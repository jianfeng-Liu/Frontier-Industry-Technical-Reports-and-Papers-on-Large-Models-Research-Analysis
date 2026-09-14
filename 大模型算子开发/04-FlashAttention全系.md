# 04 · FlashAttention 全系：序列长度从 4K 到 128K 的关键

> **前置**：[02 章（屋顶线）](02-屋顶线与性能剖析.md)、[03 章（GEMM 分块）](03-GEMM深度剖析.md)  
> **目标**：彻底搞懂 FlashAttention 三代的算法原理和工程实现，以及为什么它让长序列训练变得可行。

---

**本章关键词**

**在线 Softmax**（Online Softmax）—— 边扫描输入边维护最大值和归一化分母，不需要提前看到全部数据就能计算 Softmax，是 FlashAttention 的数学基础。  
**KV 分块**（KV Tiling）—— 把 K、V 矩阵切成小块逐块放进 SRAM，避免把 N×N 的注意力矩阵写入 HBM。  
**Softmax 重缩放**（Softmax Rescaling）—— 每次处理新的 KV 块时，用更新后的统计量修正之前的输出累加值，保证最终结果等价于一次性计算全量 Softmax。  
**Warpgroup MMA**（Warp 组矩阵乘累加）—— H100 Hopper 架构引入的 4 个 Warp 协作做一次 MMA 的机制，Tensor Core 吞吐翻倍。  
**GQA**（Grouped Query Attention，分组查询注意力）—— 多个 Query head 共享一组 K/V head，减少 KV Cache 大小，FA2 原生支持。  
**重计算**（Recomputation）—— 反向传播时不保存注意力矩阵，而是重新计算，用计算换显存，使训练显存从 O(N²) 降至 O(N)。

---

## 4.1 朴素 Attention 的内存问题

标准 Scaled Dot-Product Attention 的计算公式：

```
O = softmax(Q K^T / √d) · V

Q: (B, H, N, d)   ← N 是序列长度，d 是 head dimension
K: (B, H, N, d)
V: (B, H, N, d)
O: (B, H, N, d)   ← 输出
```


朴素实现的内存开销：

```
中间矩阵 S = Q K^T / √d：形状 (B, H, N, N)
  N=8192, H=32, B=1, FP16：
  8192 × 8192 × 32 × 2 Bytes = 4 GB

  N=32768（32K 上下文）：
  32768² × 32 × 2 ≈ 64 GB   ← 超过 H100 的 80 GB 显存！
```

这是长序列训练/推理的根本障碍。

---

## 4.2 在线 Softmax：分块的数学基础

标准 Softmax 需要两轮扫描：第一轮求最大值 m，第二轮求 `exp(x - m)` 的和 ℓ，然后才能算每个元素的值。这迫使朴素实现把整个 N×N 矩阵写进 HBM。

**在线 Softmax** 用一轮扫描同时维护 m 和 ℓ：

```
初始状态：m = -∞, ℓ = 0, O = 0

处理第 j 个块 S_j（大小 Bq × Bkv）：
  1. 计算这个块里的局部最大值：m_j = max(S_j)
  2. 更新全局最大值：m_new = max(m, m_j)
  3. 更新归一化分母：
     ℓ_new = ℓ × exp(m - m_new) + sum(exp(S_j - m_new))
              ↑                    ↑
         旧统计量的修正项      新块的贡献
  4. 更新输出累加值：
     O_new = O × (ℓ × exp(m - m_new) / ℓ_new) + exp(S_j - m_new) × V_j / ℓ_new
              ↑ 对之前累加的结果做重缩放            ↑ 加入新块的贡献

  m ← m_new, ℓ ← ℓ_new, O ← O_new

最终 O 就是正确的 softmax(QK^T)V
```

**关键洞察**：每次引入新块时，用 `exp(m_old - m_new)` 因子修正之前的 O，这等价于对之前的 Softmax 结果做了"重新归一化"。数学上可以证明最终结果与一次性计算整个序列的 Softmax 完全等价。

---

## 4.3 FlashAttention-1（FA1，2022）

**论文**：Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

### 算法框架

```
FlashAttention Forward Pass：

输入：Q, K, V ∈ (N, d)（省略 batch 和 head 维）
输出：O ∈ (N, d)

按行分块 Q：每块大小 Br = min(ceil(M/4d), d)   （M = SRAM 大小）
按列分块 K, V：每块大小 Bc = min(ceil(M/4d), d)

for i = 1 to ceil(N/Br):         # 外循环：Q 的块
    加载 Q_i ∈ (Br, d) 到 SRAM
    初始化 O_i = 0, ℓ_i = 0, m_i = -∞

    for j = 1 to ceil(N/Bc):     # 内循环：KV 的块
        加载 K_j, V_j ∈ (Bc, d) 到 SRAM
        S_ij = Q_i K_j^T / √d     # (Br, Bc)，留在 SRAM
        
        # 在线 Softmax 更新（如上节所述）
        m_ij = rowmax(S_ij)
        P_ij = exp(S_ij - m_ij)
        ℓ_ij = rowsum(P_ij)
        
        m_i_new = max(m_i, m_ij)
        ℓ_i_new = exp(m_i - m_i_new) × ℓ_i + exp(m_ij - m_i_new) × ℓ_ij
        O_i ← (exp(m_i - m_i_new) × ℓ_i × O_i + exp(m_ij - m_i_new) × P_ij V_j) / ℓ_i_new
        
        更新 m_i ← m_i_new, ℓ_i ← ℓ_i_new

    写回 O_i 到 HBM（每个 Q 块只写一次）
```

### FA1 的 IO 复杂度

```
朴素 Attention：
  读 Q, K, V：3 × N × d × 2 Bytes
  写读 S (N×N)：2 × N² × 2 Bytes   ← 主要开销
  总计 ≈ O(N²d)

FlashAttention-1：
  每次内循环读 K_j, V_j，外循环 ceil(N/Br) 次，共读 O(N²d/M) 次 K,V
  但 SRAM 足够大时，实际 HBM 访问 ≈ O(Nd²/M) × ... 
  简化：HBM IO 从 O(N²) 降到 O(N²d/√M)
  
  N=8192, d=128, M=50MB（SRAM）：HBM IO 减少约 9×
```

### FA1 的实现特点

- 用 CUDA 手写，利用共享内存精细控制数据布局
- 前向传播：不保存 Softmax 中间结果（N×N 的注意力矩阵）
- 反向传播：需要重新计算注意力矩阵（Recomputation），显存换算力
- 支持 causal mask（自回归掩码）

---

## 4.4 FlashAttention-2（FA2，2023）

**论文**：Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

FA1 已经解决了内存问题，但 GPU 利用率还不够高。FA2 针对以下三个点做了优化：

### 优化一：减少非矩阵乘运算量

FA1 中在线 Softmax 的更新公式包含多个逐元素操作（exp、除法、缩放），这些都是 CUDA Core 操作，Tensor Core 发挥不了作用。

FA2 重新推导了更新公式，把中间步骤的浮点运算量减少了约 2 倍。具体地，把对 O 的反复缩放合并到最后统一做一次除法，减少了每个 inner loop 里的操作数。

### 优化二：改变并行化策略

FA1：外循环（Q 块）并行，内循环（KV 块）串行。  
问题：对于短序列，Q 块数量少，无法占满所有 SM。

FA2：在 head 维度 + Q 块 + KV 块 三个维度上都并行。不同 Warp 处理不同的 KV 块，用 reduction 汇总。

```
FA1 并行：每个 Block 独立处理一个 (batch, head, Q_tile) 三元组
FA2 并行：在此基础上，Block 内部的 Warp 进一步分工处理不同 KV 块
```

结果：GPU 利用率提升，序列越短收益越明显。

### 优化三：原生支持 GQA / MQA

**GQA**（Grouped Query Attention，分组查询注意力）：多个 Q head 共享一组 K/V head。  
**MQA**（Multi-Query Attention）：所有 Q head 共享同一个 K/V head。

FA2 在内核里直接处理了 head 数不匹配的情况，不需要在调用前做显式的 K/V 广播（broadcast），节省了显存和带宽。

### FA2 实测效果（H100，BF16）

| 配置 | FA2 | 朴素 Attention | 提升 |
|---|---|---|---|
| N=2048, H=32, d=128 | 180 TFLOPS | 45 TFLOPS | 4× |
| N=8192, H=32, d=128 | 330 TFLOPS | 18 TFLOPS | 18× |
| N=32768, H=32, d=128 | 380 TFLOPS | OOM | — |

> ✅ FA2 是目前最广泛使用的版本，PyTorch 2.0+ 的 `F.scaled_dot_product_attention` 在检测到支持时会自动调用 FA2 的 CUDA 实现。

---

## 4.5 FlashAttention-3（FA3，2024）

**论文**：Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"

FA2 在 A100 上表现优秀，但在 H100 的 Hopper 架构上还没充分利用新硬件特性。FA3 专门针对 H100 重写。

### H100 的两个新硬件特性

**1. Warpgroup MMA（wgmma）指令**

H100 引入了 Warpgroup 级别的矩阵乘累加指令：4 个 Warp（128 个线程）协作完成一次大型 MMA。相比 FA2 使用的单 Warp MMA，Warpgroup MMA 的吞吐是其 2 倍以上。

FA3 把所有矩阵乘换成了 `wgmma` 指令，这是性能提升的最大来源。

**2. TMA（Tensor Memory Accelerator）**

H100 新增了硬件单元 TMA，专门负责 HBM → SRAM 的数据搬运，完全不占用计算流水线（比 FA2 里的 `cp.async` 更彻底）。

FA3 用 TMA 做所有数据加载，让加载和矩阵乘真正**流水线化**：

```
时间 →
  加载 K_j+1, V_j+1:  [────TMA异步────]
  计算 S_j, P_j V_j:          [────wgmma────]
  两者完全重叠，互不干扰
```

### FA3 解决了 FA2 的"单 Warp 低效"问题

FA2 用单 Warp MMA，但 Softmax 的非矩阵乘操作（exp、max、sum）在 CUDA Core 上运行，而且和矩阵乘是串行的：

```
FA2：[矩阵乘 QK^T][等待][Softmax 操作][等待][矩阵乘 PV][等待]...
FA3：[wgmma QK^T ─────][Softmax ──────][wgmma PV ──────]...
                              ↑
               wgmma 和 Softmax 流水线重叠（不同硬件单元）
```

### FA3 实测效果（H100 SXM5，FP16）

| 场景 | FA3 | FA2 | 提升 |
|---|---|---|---|
| 前向，N=8192 | 740 TFLOPS | 540 TFLOPS | 1.37× |
| 前向，N=32768 | 810 TFLOPS | 600 TFLOPS | 1.35× |
| 理论峰值利用率 | 75% | 55% | — |

FA3 还新增了 FP8 支持（H100 原生 FP8 Tensor Core），在 FP8 精度下前向可达 1200+ TFLOPS，是 FA2 FP16 的 2 倍以上。

---

## 4.6 反向传播：重计算换显存

FlashAttention 在反向传播中采用了一个重要的权衡：**不保存注意力矩阵，反向时重计算它**。

### 标准反向传播（不用 FA）

```
前向保存：Q, K, V, S = QK^T, P = softmax(S), O
反向需要：dP = dO V^T    → 需要 V
          dS = P ⊙ (dP - sum(P ⊙ dP, dim=-1))  → 需要 P（N×N！）
          dQ = dS K
          dK = dS^T Q
          dV = P^T dO
```

保存 P（N×N 矩阵）的显存：N=8192 时 4 GB，是训练时显存的主要占用之一。

### FlashAttention 的重计算策略

```
前向只保存：Q, K, V, O, ℓ（归一化分母，大小 N），m（最大值，大小 N）
反向时：从 Q, K, V 重新计算 S 和 P（分块，不写 HBM）
```

代价：反向传播的 FLOPs 增加约 30%（因为重新做了一次 QK^T）。
收益：显存从 O(N²) 降至 O(N)，对序列长度和 batch size 有根本性突破。

这个权衡在大多数训练场景是绝对合算的：GPU 的 FLOP 相对便宜，而 HBM 显存是硬约束。

---

## 4.7 变长序列与 Paged 注意力

### varlen（可变长度批处理）

训练/推理时一批序列长度不同，朴素做法是 padding 到最长长度，但短序列的计算就浪费了。

FA2/FA3 支持 `varlen` 模式：把一批不同长度的序列**拼接**成一条长序列，用额外的 `cu_seqlens`（累计序列长度数组）记录边界：

```
例子：seq_1 (len=3), seq_2 (len=5), seq_3 (len=2)
拼接后：[tok₀tok₁tok₂ | tok₀tok₁tok₂tok₃tok₄ | tok₀tok₁]
cu_seqlens = [0, 3, 8, 10]

每个序列内部正常做 causal attention，序列之间天然不能互相关注
（causal mask + 边界检查保证了这一点）
```

### Paged KV Cache（推理场景，见第 09 章）

推理时 KV Cache 不连续（PagedAttention），FA 的 kernel 需要支持"逻辑块 → 物理块"的查表式内存访问，即所谓 Paged FlashAttention。vLLM 实现了这一变体。

---

## 4.8 实用选型

| 场景 | 推荐 | 说明 |
|---|---|---|
| PyTorch 训练，PyTorch >= 2.0 | `F.scaled_dot_product_attention` | 自动调用 FA2，开箱即用 |
| 需要 GQA | FA2（`flash-attn` 库） | 原生支持 |
| H100，追求最高性能 | FA3（`flash-attn >= 2.6` 或 `hopper_flash`） | 需要 CUDA 12.x |
| FP8 精度 | FA3 | 原生 FP8 kernel |
| 超长序列（>100K）| Ring Attention + FA2/FA3 | 跨 GPU 分片 |
| 推理服务（vLLM/SGLang）| Paged FA2/FA3 | 框架内置 |

---

## 4.9 本章小结

```
三代 FlashAttention 的核心演进：

FA1（2022）：
  核心思路：KV 分块 + 在线 Softmax → 不写 N×N 矩阵
  收益：显存 O(N²) → O(N)，速度 2–4×
  代价：CUDA 实现复杂，反向需重计算

FA2（2023）：
  核心改进：减少非矩阵乘操作，改进并行策略，GQA 原生支持
  收益：在 FA1 基础上再提升 2×

FA3（2024）：
  核心改进：用 H100 的 Warpgroup MMA 和 TMA，流水线重叠
  收益：H100 利用率 55% → 75%，FP8 支持

数学本质：在线 Softmax 的两个递推式
  m_new = max(m_old, m_new_block)
  ℓ_new = ℓ_old × exp(m_old - m_new) + sum(exp(S_new - m_new))

反向传播的核心取舍：
  不保存 P（节省 O(N²) 显存）
  反向时重计算 P（多花 ~30% FLOPs）

推理时的变体：Paged FA，支持非连续 KV Cache（PagedAttention）

下一章：光优化单个 kernel 还不够，算子之间的数据搬运也是大头——算子融合。
```

**延伸资料**：
- [FlashAttention-1 论文](https://arxiv.org/abs/2205.14135)（Dao et al., 2022）
- [FlashAttention-2 论文](https://arxiv.org/abs/2307.08691)（Dao, 2023）
- [FlashAttention-3 论文](https://arxiv.org/abs/2407.08608)（Shah et al., 2024）
- [flash-attn GitHub](https://github.com/Dao-AILab/flash-attention)（含安装、API 文档）
- [Making FlashAttention Go Brrrr](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)（通俗图解）
