import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# 超参数配置
# ============================================================
batch_size = 32  # 每次并行处理多少个独立序列（批大小）
block_size = 8   # 模型预测时使用的最大上下文长度（每条序列的字符数）
max_iters = 3000      # 训练总迭代次数
eval_interval = 300   # 每隔多少步评估一次损失
learning_rate = 1e-2  # 学习率（AdamW 优化器的步长）
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')  # 有 GPU 就用 GPU，否则用 CPU
eval_iters = 200      # 评估损失时，采样多少个批次取平均（减少估计方差）
# ------------------------------------------------------------

# 固定随机种子，保证实验可复现
torch.manual_seed(1337)

# 读取训练文本（莎士比亚小说）
# 可用以下命令下载：
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ============================================================
# 构建字符级词表（Vocabulary）
# ============================================================
# 找出文本中所有出现过的唯一字符，排序后作为词表
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 词表大小，即模型能处理的字符种类数

# 建立字符 <-> 整数的双向映射
stoi = { ch:i for i,ch in enumerate(chars) }  # 字符 -> 整数（string to index）
itos = { i:ch for i,ch in enumerate(chars) }  # 整数 -> 字符（index to string）

# 编码器：将字符串转换为整数列表
encode = lambda s: [stoi[c] for c in s]
# 解码器：将整数列表还原为字符串
decode = lambda l: ''.join([itos[i] for i in l])

# ============================================================
# 划分训练集和验证集
# ============================================================
data = torch.tensor(encode(text), dtype=torch.long)  # 将全文编码为整数张量
n = int(0.9*len(data))  # 前 90% 作为训练集，后 10% 作为验证集
train_data = data[:n]
val_data = data[n:]

# ============================================================
# 数据加载函数
# ============================================================
def get_batch(split):
    """
    随机采样一个小批次的输入 x 和目标 y。
    x[i] 是长度为 block_size 的字符序列，
    y[i] 是 x[i] 向右移动一位的序列（即每个位置的"下一个字符"）。
    """
    data = train_data if split == 'train' else val_data
    # 随机选取 batch_size 个起始位置（保证不越界）
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 构造输入序列 x：从每个起始位置取 block_size 个字符
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 构造目标序列 y：x 对应的"下一个字符"，即向右偏移 1 位
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ============================================================
# 损失估计函数（用于监控训练进度）
# ============================================================
@torch.no_grad()  # 评估时不需要计算梯度，节省内存
def estimate_loss():
    """
    在训练集和验证集上各采样 eval_iters 个批次，
    计算平均损失，用于监控模型是否过拟合。
    """
    out = {}
    model.eval()  # 切换为评估模式（关闭 dropout 等训练专用层）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()  # 取平均损失
    model.train()  # 切换回训练模式
    return out

# ============================================================
# Bigram 语言模型定义
# ============================================================
class BigramLanguageModel(nn.Module):
    """
    最简单的语言模型：Bigram 模型。
    核心思想：只根据"当前字符"预测"下一个字符"，
    不考虑更长的上下文（这正是 bigram 的含义：二元组）。

    实现方式：用一个嵌入表（Embedding Table）直接存储每个字符对应的
    下一字符概率分布（logits）。嵌入表的第 i 行 = 第 i 个字符作为输入时，
    预测下一字符的 logits（未归一化的概率）。
    """

    def __init__(self, vocab_size):
        super().__init__()
        # token_embedding_table：形状为 (vocab_size, vocab_size) 的查找表
        # 输入字符的整数编码 -> 输出下一字符的 logits（每个字符一个得分）
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        前向传播：给定输入序列，计算每个位置的下一字符预测分布，
        并可选地计算交叉熵损失。

        参数：
            idx:     形状 (B, T) 的整数张量，B=批大小，T=序列长度
            targets: 形状 (B, T) 的整数张量，每个位置的真实下一字符（可为 None）

        返回：
            logits: 形状 (B, T, C) 的张量，C=vocab_size，每个位置的预测得分
            loss:   标量交叉熵损失（若 targets 为 None 则返回 None）
        """
        # 通过嵌入表将每个字符编码映射为 logits，形状：(B, T, C)
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            # 推理模式：不计算损失
            loss = None
        else:
            # 训练模式：计算交叉熵损失
            B, T, C = logits.shape
            # cross_entropy 要求输入形状为 (N, C)，所以把 B 和 T 维度合并
            logits = logits.view(B*T, C)    # (B*T, C)
            targets = targets.view(B*T)     # (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        自回归生成：从给定的起始序列开始，逐步采样新字符，
        直到生成 max_new_tokens 个新字符为止。

        参数：
            idx:            形状 (B, T) 的整数张量，初始上下文
            max_new_tokens: 要生成的新字符数量

        返回：
            形状 (B, T + max_new_tokens) 的整数张量
        """
        for _ in range(max_new_tokens):
            # 1. 用当前序列做前向传播，得到所有位置的 logits
            logits, loss = self(idx)
            # 2. 只取最后一个时间步的 logits（bigram 只看最后一个字符）
            logits = logits[:, -1, :]           # (B, C)
            # 3. Softmax 将 logits 转化为概率分布
            probs = F.softmax(logits, dim=-1)   # (B, C)
            # 4. 按概率分布随机采样下一个字符的索引
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # 5. 将新采样的字符拼接到序列末尾，作为下一步的输入
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

# ============================================================
# 初始化模型与优化器
# ============================================================
model = BigramLanguageModel(vocab_size)
m = model.to(device)  # 将模型参数移到指定设备（CPU 或 GPU）

# AdamW 优化器（Adam + 权重衰减，比普通 SGD 更适合深度学习）
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ============================================================
# 训练循环
# ============================================================
for iter in range(max_iters):

    # 每隔 eval_interval 步，在训练集和验证集上评估损失并打印
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 采样一个训练批次
    xb, yb = get_batch('train')

    # 前向传播：计算预测 logits 和损失
    logits, loss = model(xb, yb)
    # 清空上一步的梯度（set_to_none=True 比置零更省内存）
    optimizer.zero_grad(set_to_none=True)
    # 反向传播：计算各参数的梯度
    loss.backward()
    # 更新参数
    optimizer.step()

# ============================================================
# 用训练好的模型生成文本
# ============================================================
# 以"字符 0"（换行符）作为起始 token，形状 (1, 1) 表示 1 条序列、长度为 1
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# 生成 500 个新字符，解码后打印
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
