"""
从训练好的模型中采样生成文本。
支持两种模式：
  1. resume：从本地 checkpoint 加载自训练模型
  2. gpt2*：直接加载 OpenAI 官方 GPT-2 权重（如 'gpt2', 'gpt2-xl'）
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 采样配置（均可通过命令行参数覆盖）
init_from = 'resume'   # 'resume'（从本地 checkpoint）或 gpt2 变体（如 'gpt2-xl'）
out_dir = 'out'        # checkpoint 目录，init_from='resume' 时使用
start = "\n"           # 生成的起始 prompt；也可以是 "<|endoftext|>"
                       # 或从文件读取：start="FILE:prompt.txt"
num_samples = 10       # 生成几段独立的文本
max_new_tokens = 500   # 每段最多生成多少个 token
temperature = 0.8      # 采样温度：=1.0 不变，<1.0 更保守（更确定），>1.0 更随机（更多样）
top_k = 200            # Top-K 采样：每步只保留概率最高的 K 个 token，其余置为 0
                       # 防止采样到低概率的奇怪 token
seed = 1337            # 随机种子，保证结果可复现
device = 'cuda'        # 运行设备
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False        # 推理时通常不需要 compile（节省等待时间）
exec(open('configurator.py').read())  # 从命令行参数覆盖上述默认值
# -----------------------------------------------------------------------------

# ---- 基础环境初始化 ----
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # A100 上启用 TF32 加速矩阵乘法
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# 混合精度推理上下文（推理不需要 GradScaler，只用 autocast 降低精度提速）
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ---- 加载模型 ----
if init_from == 'resume':
    # 从本地 checkpoint 加载（自己训练的模型）
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])  # 用 checkpoint 中记录的结构参数重建模型
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # 去掉 torch.compile 引入的 '_orig_mod.' 前缀（如果有的话）
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # 从 HuggingFace 下载并加载官方 GPT-2 权重
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))  # 推理时关闭 dropout

model.eval()      # 切换到推理模式（关闭 dropout，LayerNorm 使用固定统计量）
model.to(device)
if compile:
    model = torch.compile(model)  # 可选：编译加速（推理场景收益相对较小）

# ---- 确定 tokenizer（编码/解码函数）----
# 字符级模型（如 shakespeare_char）使用自定义的字符映射表，存在 meta.pkl 中
# BPE 模型（如 openwebtext）使用 GPT-2 的 BPE tokenizer（tiktoken）
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    # 旧版 checkpoint 可能没有 'config' 字段，做兼容处理
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']  # stoi: 字符→id，itos: id→字符
    encode = lambda s: [stoi[c] for c in s]               # 字符串 → token id 列表
    decode = lambda l: ''.join([itos[i] for i in l])       # token id 列表 → 字符串
else:
    # 没有 meta.pkl，默认使用 GPT-2 的 BPE tokenizer
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# ---- 编码起始 prompt ----
if start.startswith('FILE:'):
    # 支持从文件读取 prompt：python sample.py --start=FILE:prompt.txt
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
# 转为 tensor，shape = (1, len(start_ids))，unsqueeze(0) 添加 batch 维度
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# ---- 自回归生成 ----
with torch.no_grad():  # 推理不需要梯度
    with ctx:          # 混合精度推理
        for k in range(num_samples):
            # model.generate 每步预测下一个 token 并拼接到序列末尾，重复 max_new_tokens 次
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))  # y[0] 取 batch 第 0 个，转 list 后解码为字符串
            print('---------------')
