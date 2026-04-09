# KV Cache：从 nanogpt 代码理解 Token 缓存的底层机制

## 0. 先看 nanogpt 当前怎么生成 token

`model.py` 第 306-330 行，`generate()` 方法：

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    logits, _ = self(idx_cond)   # 每次都把整个序列喂进去做 forward
    logits = logits[:, -1, :]    # 只取最后一个 token 的 logits
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)  # 新 token 追加到序列末尾
```

**问题**：每生成一个新 token，都把完整的 `idx_cond`（已有的全部 token）重新走一遍 `forward()`。
但实际上我们只需要最后一个位置的输出——前面所有 token 的计算结果被白白丢弃了。

---

## 1. 推理的两个阶段

理解 KV Cache 首先要区分两个阶段：

| 阶段 | 名称 | 发生时机 | 做什么 |
|------|------|---------|--------|
| 阶段一 | **Prefill（预填充）** | 输入 prompt 时 | 并行处理所有 prompt token，得到初始 KV |
| 阶段二 | **Decode（解码）** | 逐 token 生成时 | 每步只新增 1 个 token，复用之前的 KV |

nanogpt 当前的实现没有区分这两个阶段——每步 decode 都重新做了完整的 prefill，所以是浪费的。

---

## 2. KV Cache 在哪里发生作用

核心在 `CausalSelfAttention.forward()`，第 52-76 行：

```python
def forward(self, x):
    B, T, C = x.size()

    # 对输入 x 中的每个 token，同时算出 Q、K、V
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # shape: (B, n_head, T, head_size)

    # attention 计算：每个 token 的 Q 要和所有 token 的 K 做点积
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))  # causal mask
    att = F.softmax(att, dim=-1)
    y = att @ v
```

**关键观察**：

一个 token 在 attention 计算中扮演两种角色：
- 作为**查询方**（Q）：用自己的 Q 去问"我需要关注谁？"
- 作为**被查询方**（K/V）：提供 K 让别人来匹配，提供 V 传递信息

**causal mask（下三角掩码）** 保证了位置 t 只能看到位置 ≤ t 的 token。
这意味着：**已经生成的 token 的 K 和 V，对未来 token 来说永远不会变**。

---

## 3. 为什么可以缓存 K 和 V

假设已经生成了 [t₁, t₂, t₃]，现在要生成 t₄：

```
不用缓存（当前 nanogpt）：
  输入 [t₁, t₂, t₃, t₄_prompt]
  → 对全部 4 个 token 算 Q/K/V
  → 做 4×4 的 attention（受 causal mask 限制实际是下三角）
  → 取最后一行输出预测下一个 token
  → t₁, t₂, t₃ 的 K/V 被重复计算了

用 KV Cache：
  缓存中已有 [K₁,V₁], [K₂,V₂], [K₃,V₃]（上轮算好的）
  输入只有新 token [t₄_prompt]
  → 只算 Q₄, K₄, V₄（1 个 token 的计算量）
  → attention: Q₄ @ [K₁,K₂,K₃,K₄]ᵀ → 对全部历史做 softmax
  → 输出 = softmax(...) @ [V₁,V₂,V₃,V₄]
  → 预测下一个 token
```

**节省的计算**：从 O(T²) 降到 O(T)（对新 token 而言），其中 T 是已有序列长度。

---

## 4. 如何在 nanogpt 中实现 KV Cache

需要修改两处：

### 4.1 CausalSelfAttention 加入 past_kv 参数

```python
def forward(self, x, past_kv=None):
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # 如果有缓存，把历史 K/V 拼在前面
    if past_kv is not None:
        k_past, v_past = past_kv
        k = torch.cat([k_past, k], dim=2)  # 在序列维度拼接
        v = torch.cat([v_past, v], dim=2)

    # 返回当前完整的 K/V 供下一步使用
    present_kv = (k, v)

    # attention 计算（q 只有新 token，k/v 是完整历史）
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # 注意：这里不再需要 causal mask，因为 q 的位置天然在 k 之后
    att = F.softmax(att, dim=-1)
    y = att @ v

    return y, present_kv
```

### 4.2 generate() 传递并更新缓存

```python
def generate(self, idx, max_new_tokens):
    past_kvs = [None] * self.config.n_layer  # 每层一个 KV 缓存

    # Prefill 阶段：先把 prompt 整体跑一遍，建立初始缓存
    _, past_kvs = self.forward_with_cache(idx, past_kvs)

    for _ in range(max_new_tokens):
        # Decode 阶段：每次只输入最新 1 个 token
        last_token = idx[:, -1:]
        logits, past_kvs = self.forward_with_cache(last_token, past_kvs)
        # ...采样下一个 token，追加到 idx
```

---

## 5. 从单请求 KV Cache 到跨请求 Prompt Caching

单请求 KV Cache 存在 GPU 显存里，请求结束就释放。

Anthropic 的 **Prompt Caching** 本质上是把 KV Cache **持久化到服务端存储**，跨请求复用：

```
请求1（Prefill System + User(1) + User(2)）：
  → 算出每层的 K/V 矩阵
  → 存到服务端（TTL 5分钟）

请求2（有新消息 Asst(2) + User(3)）：
  → 加载请求1缓存的 K/V（跳过 Prefill 中最贵的部分）
  → 只对 Asst(2) + User(3) 做新的 Prefill
  → 把扩展后的完整 K/V 写入新缓存
```

对应到 nanogpt 代码的结构，就是：
- `CausalSelfAttention.forward()` 第 56-59 行计算的 `k, v`：这就是要缓存的内容
- `generate()` 第 316 行 `self(idx_cond)`：没有 KV Cache 时每次重新算的那部分

---

## 6. 计算量对比

以序列长度 T=1000，n_layer=12 为例：

| 方式 | 新 token 的计算量 | 旧 token 的计算量 |
|------|-----------------|-----------------|
| 无 KV Cache（当前 nanogpt）| O(T) | 每次 O(T²)，全部重算 |
| 有 KV Cache（单请求）| O(T) | 0，读缓存 |
| Prompt Caching（跨请求）| O(新增长度) | 0，读持久化缓存 |

缓存命中的 token 按 **10% 费率**计费（而不是 100%），因为省掉了 Attention 和 FFN 的计算，只需从存储加载 K/V 矩阵并写入显存。

---

## 小结

| 概念 | 对应 nanogpt 代码位置 | 作用 |
|------|---------------------|------|
| Q/K/V 的计算 | `CausalSelfAttention.forward()` L56-59 | KV Cache 缓存的就是这里算出的 k, v |
| 全序列 Attention | 同上 L62-71 | 复用缓存后，只有新 token 的 Q 参与计算 |
| 朴素生成循环 | `GPT.generate()` L312-328 | 每步重复调用 forward，是 KV Cache 要优化的对象 |
| Causal Mask | `__init__` L49-50，`forward` L68 | 保证了旧 token 的 K/V 不会因新 token 而改变，是缓存合法的前提 |
