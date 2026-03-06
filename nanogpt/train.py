"""
训练脚本，支持单 GPU 调试模式和多 GPU 分布式数据并行（DDP）训练。

单 GPU 运行示例：
$ python train.py --batch_size=32 --compile=False

单节点 4 GPU DDP 运行示例：
$ torchrun --standalone --nproc_per_node=4 train.py

跨 2 个节点、每节点 8 GPU 的 DDP 运行示例：
- 在主节点（IP 123.456.123.456）上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- 在工作节点上运行：
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
（如果集群没有 Infiniband 网络，在命令前加 NCCL_IB_DISABLE=1）
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 默认超参数配置，目标是在 OpenWebText 上训练 GPT-2 (124M)
# 这些值都可以被命令行参数或 config 文件覆盖（通过下方的 configurator.py）

# I/O 配置
out_dir = 'out'               # checkpoint 和日志的输出目录
eval_interval = 2000          # 每隔多少步做一次验证集评估
log_interval = 1              # 每隔多少步打印一次训练日志
eval_iters = 200              # 评估时用多少个 batch 来估算 loss（取均值）
eval_only = False             # 为 True 时只做评估，不训练，评估后立即退出
always_save_checkpoint = True # 为 True 时每次评估后都保存 checkpoint（即使 val loss 没有下降）
init_from = 'scratch'         # 初始化方式：'scratch'（随机初始化）/ 'resume'（续训）/ 'gpt2*'（从 OpenAI 权重微调）

# wandb 日志配置（需要 pip install wandb）
wandb_log = False             # 是否启用 wandb 记录训练曲线
wandb_project = 'owt'         # wandb 项目名
wandb_run_name = 'gpt2'       # wandb 运行名称

# 数据配置
dataset = 'openwebtext'       # 数据集名称，对应 data/ 下的子目录
gradient_accumulation_steps = 5 * 8  # 梯度累积步数，用于模拟更大的 batch size
                                      # 实际有效 batch = batch_size * gradient_accumulation_steps * ddp_world_size
batch_size = 12               # 单次前向的 micro-batch size（梯度累积时每步处理的样本数）
block_size = 1024             # 序列长度（上下文窗口大小），即每个样本包含多少个 token

# 模型结构超参数（GPT-2 124M 规格）
n_layer = 12                  # Transformer 层数
n_head = 12                   # 每层多头注意力的头数
n_embd = 768                  # 隐藏层维度（embedding 维度）
dropout = 0.0                 # Dropout 概率：预训练用 0，微调建议 0.1+
bias = False                  # LayerNorm 和 Linear 层是否使用 bias（False 更快且略好）

# AdamW 优化器配置
learning_rate = 6e-4          # 最大学习率（warmup 结束后的峰值）
max_iters = 600000            # 总训练步数
weight_decay = 1e-1           # 权重衰减系数（L2 正则化）
beta1 = 0.9                   # Adam 一阶矩指数衰减率
beta2 = 0.95                  # Adam 二阶矩指数衰减率（GPT-2 用 0.95，比默认 0.999 更适合语言模型）
grad_clip = 1.0               # 梯度裁剪阈值（防止梯度爆炸），0.0 表示不裁剪

# 学习率调度：cosine decay with warmup
decay_lr = True               # 是否启用学习率衰减
warmup_iters = 2000           # 线性 warmup 的步数（从 0 线性升至 learning_rate）
lr_decay_iters = 600000       # cosine 衰减结束步数，通常等于 max_iters（参考 Chinchilla 论文）
min_lr = 6e-5                 # 最小学习率（cosine 衰减的下界），通常为 learning_rate / 10

# DDP（分布式数据并行）配置
backend = 'nccl'              # 通信后端：GPU 用 'nccl'，CPU 用 'gloo'

# 系统配置
device = 'cuda'               # 训练设备：'cpu' / 'cuda' / 'cuda:0' / 'mps'（Mac）
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# 数据类型：bfloat16（A100 推荐，数值更稳定）/ float16（较老 GPU，会自动启用 GradScaler）/ float32
compile = True                # 是否使用 torch.compile()（PyTorch 2.0+），可显著提升速度
# -----------------------------------------------------------------------------

# 收集所有标量超参数的键名，供后续 configurator.py 和 wandb 使用
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # 从命令行参数或 config 文件覆盖上述默认值
config = {k: globals()[k] for k in config_keys} # 将最终配置存入字典，方便日志记录
# -----------------------------------------------------------------------------

# ---- DDP 初始化 & 基础环境配置 ----
# torchrun 启动时会自动设置 RANK 环境变量；直接 python 运行时 RANK 不存在（返回 -1）
ddp = int(os.environ.get('RANK', -1)) != -1  # 判断是否为 DDP 多进程训练
if ddp:
    # DDP 模式：初始化进程组，每个进程绑定一块 GPU
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])             # 全局进程编号（跨所有节点）
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # 本节点内的进程编号（= GPU 编号）
    ddp_world_size = int(os.environ['WORLD_SIZE']) # 总进程数（= 总 GPU 数）
    device = f'cuda:{ddp_local_rank}'              # 每个进程只操作自己对应的 GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # rank 0 进程负责日志记录、保存 checkpoint 等
    seed_offset = ddp_rank          # 每个进程使用不同的随机种子，保证数据采样多样性
    # DDP 中所有进程同时训练，因此每个进程只需做 1/world_size 的梯度累积步数
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # 单 GPU / CPU 模式
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# 每次迭代实际处理的 token 总数（用于换算训练进度）
# = 梯度累积步数 × GPU 数 × 单 GPU batch size × 序列长度
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# 只有主进程（rank 0）创建输出目录，避免多进程竞争
if master_process:
    os.makedirs(out_dir, exist_ok=True)

# 设置随机种子（DDP 时各进程种子不同，保证数据采样差异）
torch.manual_seed(1337 + seed_offset)

# 允许在矩阵乘法和 cuDNN 中使用 TF32（A100 上接近 FP32 精度但速度更快）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'  # 用于后续 torch.autocast 判断

# 将字符串类型名映射为 torch dtype 对象
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# 混合精度训练上下文：CPU 时用 nullcontext（不做任何操作），GPU 时用 autocast 自动降精度
# float16 模式下 autocast 会配合 GradScaler 防止梯度下溢
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---- 数据加载器（极简实现）----
data_dir = os.path.join('data', dataset)

def get_batch(split):
    """
    从磁盘随机采样一个 batch。
    数据文件是预先 tokenize 好的 uint16 二进制文件（prepare.py 生成）。
    x 是输入序列，y 是 x 向右移一位的目标序列（语言模型的标准监督信号）。
    """
    # 每次调用都重新创建 memmap，防止内存泄漏
    # （memmap 对象持有文件句柄，长期持有会导致内存不释放）
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # 随机采样 batch_size 个起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x: 输入 token 序列，shape = (batch_size, block_size)
    # y: 目标 token 序列，y[i] = x[i] 右移一位，即预测下一个 token
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin_memory() 将 CPU tensor 固定在内存中，配合 non_blocking=True 实现 CPU→GPU 异步传输
        # 这样 GPU 在做前向时，CPU 可以同时准备下一个 batch（数据预取）
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 训练状态变量（续训时会从 checkpoint 覆盖）
iter_num = 0         # 当前训练步数
best_val_loss = 1e9  # 历史最佳验证 loss，用于判断是否保存 checkpoint

# 尝试从数据集的 meta.pkl 中读取 vocab_size
# 字符级数据集（如 shakespeare_char）会有此文件，里面存了字符到 id 的映射
# BPE 数据集（如 openwebtext）没有此文件，vocab_size 默认为 GPT-2 的 50304
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# ---- 模型初始化（支持三种方式）----
# model_args 先用命令行/config 中的值填充，后续可能被 checkpoint 覆盖
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    # 方式一：从零随机初始化，适用于预训练
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    # 50304 = 50257 向上取整到 64 的倍数，矩阵运算更高效（对齐 GPU 内存）
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    # 方式二：从 checkpoint 续训
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # 这些结构参数必须与 checkpoint 完全一致，否则无法加载权重
    # dropout 等超参数可以在命令行重新指定（允许微调时调整）
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # torch.compile 后的模型权重 key 会带 '_orig_mod.' 前缀，需要去掉才能正常加载
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']           # 恢复训练步数
    best_val_loss = checkpoint['best_val_loss'] # 恢复历史最佳 val loss

elif init_from.startswith('gpt2'):
    # 方式三：从 OpenAI 预训练权重初始化，适用于微调
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)  # 微调时可以修改 dropout
    model = GPT.from_pretrained(init_from, override_args)
    # 从加载的模型中读回实际的结构参数，存入 model_args 以便保存到 checkpoint
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# 如果指定的 block_size 小于模型原始 block_size，裁剪位置编码（model surgery）
# 例如：从 GPT-2（block_size=1024）微调到更短序列任务
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # 确保 checkpoint 中记录的是裁剪后的值
model.to(device)

# GradScaler 用于 float16 混合精度训练，防止梯度下溢
# bfloat16 数值范围与 float32 相同，不需要 GradScaler（enabled=False 时是 no-op）
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# 创建 AdamW 优化器（内部会对参数分组：embedding/权重矩阵做 weight decay，bias/LN 不做）
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])  # 恢复优化器状态（m, v, step 等）
checkpoint = None  # 立即释放 checkpoint 占用的内存

# torch.compile()：将模型编译为优化的计算图（类似 TorchScript 但更强）
# 原理：算子融合（减少 kernel launch）+ 内存布局优化，可降低 ~30-50% 耗时
# 首次编译约需 1 分钟，之后每步更快
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model  # 保留未编译的引用，供 estimate_mfu 使用
    model = torch.compile(model)  # requires PyTorch 2.0

# 将模型包装进 DDP 容器（必须在 compile 之后）
# DDP 会在反向传播时自动对各 GPU 的梯度做 All-Reduce 求平均
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# ---- 辅助函数：评估 loss ----
@torch.no_grad()  # 评估时不需要计算梯度，节省显存和计算量
def estimate_loss():
    """
    在 train/val 集上各跑 eval_iters 个 batch，取 loss 均值。
    用多个 batch 平均是为了减少随机采样带来的方差，得到更稳定的估计值。
    """
    out = {}
    model.eval()  # 切换到评估模式（关闭 dropout，batchnorm 使用统计值）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:  # 评估也用混合精度，保持一致性
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # 切换回训练模式（恢复 dropout）
    return out

# ---- 学习率调度：线性 Warmup + Cosine Decay ----
def get_lr(it):
    """
    学习率调度策略（参考 GPT-3 / Chinchilla 论文）：
    1. 线性 Warmup：前 warmup_iters 步从 0 线性升至 learning_rate
       （避免训练初期梯度过大导致不稳定）
    2. Cosine Decay：从 learning_rate 余弦衰减至 min_lr
       （平滑衰减，比线性衰减效果更好）
    3. 衰减结束后：保持 min_lr 不变
    """
    # 阶段 1：线性 warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 阶段 3：衰减结束，保持最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 阶段 2：cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 0→1
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # cosine 系数：1→0
    return min_lr + coeff * (learning_rate - min_lr)

# 初始化 wandb（只在主进程中初始化，避免多进程重复记录）
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ============================================================
# 主训练循环
# ============================================================
X, Y = get_batch('train')  # 预取第一个 batch，避免第一步计时包含数据加载时间
t0 = time.time()
local_iter_num = 0  # 当前进程生命周期内的迭代次数（用于跳过 MFU 预热期）
raw_model = model.module if ddp else model  # DDP 包装后需要 .module 才能访问原始模型
running_mfu = -1.0  # MFU 的指数移动平均值，-1.0 表示尚未计算

while True:

    # ---- 1. 更新学习率 ----
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 手动更新每个参数组的学习率

    # ---- 2. 周期性评估 + 保存 checkpoint ----
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,  # 转换为百分比
            })
        # 当 val loss 创新低，或 always_save_checkpoint=True 时保存
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),       # 模型权重
                    'optimizer': optimizer.state_dict(),   # 优化器状态（续训必须保存）
                    'model_args': model_args,              # 模型结构参数
                    'iter_num': iter_num,                  # 当前步数
                    'best_val_loss': best_val_loss,        # 最佳 val loss
                    'config': config,                      # 完整超参数配置
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break  # eval_only 模式：只评估一次就退出

    # ---- 3. 前向 + 反向 + 梯度累积 ----
    # 梯度累积：将大 batch 拆分成多个 micro-batch，分步前向，梯度相加后统一更新
    # 等效于 batch_size * gradient_accumulation_steps 的大 batch，但显存只需单个 micro-batch
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # DDP 默认每次 backward 后都做梯度同步（All-Reduce），开销很大
            # 梯度累积时只在最后一步才需要同步，中间步骤关闭同步节省通信开销
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:  # 混合精度上下文
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # 对 loss 除以累积步数，等效于对梯度取平均

        # 在 GPU 做反向传播的同时，异步预取下一个 batch（CPU-GPU 并行）
        X, Y = get_batch('train')

        # 反向传播（float16 时 GradScaler 会自动放大 loss 防止梯度下溢）
        scaler.scale(loss).backward()

    # ---- 4. 梯度裁剪 ----
    # 防止梯度爆炸，将所有参数的梯度范数裁剪到 grad_clip 以内
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)  # 先反缩放梯度（消除 GradScaler 的放大），再裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # ---- 5. 参数更新 ----
    scaler.step(optimizer)   # 更新参数（float16 时 GradScaler 会检查梯度是否有 inf/nan）
    scaler.update()          # 动态调整 GradScaler 的缩放因子
    optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none 比置零更省内存）

    # ---- 6. 日志记录 ----
    t1 = time.time()
    dt = t1 - t0  # 本步耗时（秒）
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # loss.item() 是一次 CPU-GPU 同步点（强制等待 GPU 计算完成）
        # 乘以 gradient_accumulation_steps 还原出真实的 loss 量级
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # 跳过前几步（训练循环尚未稳定，计时不准）
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            # 指数移动平均平滑 MFU，减少单步波动
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # ---- 7. 终止条件 ----
    if iter_num > max_iters:
        break

# DDP 训练结束后销毁进程组，释放通信资源
if ddp:
    destroy_process_group()
