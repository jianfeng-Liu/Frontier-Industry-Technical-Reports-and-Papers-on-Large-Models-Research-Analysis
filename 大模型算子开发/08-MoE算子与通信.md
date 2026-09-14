# 08 · MoE 算子与通信：为什么 MoE 难，难在哪

> **前置**：[03 章（GEMM）](03-GEMM深度剖析.md)、[07 章（Triton）](07-Triton编程.md)  
> **目标**：搞清楚 MoE 架构在 GPU 上的核心挑战——Grouped GEMM 的不规则性、All-to-All 通信的带宽代价、Expert Parallelism 的设计权衡，以及 DeepSeek 的工程解法。

---

**本章关键词**

**MoE**（Mixture of Experts，专家混合）—— 把 FFN 层替换为多个"专家"网络，每个 token 只激活其中少数几个，计算量随激活专家数而非总专家数增长。  
**门控网络**（Gating Network）—— 决定每个 token 路由到哪些专家的小型网络，通常是一个线性层 + Softmax/TopK。  
**Expert Parallelism**（专家并行）—— 把不同专家放在不同 GPU 上，每个 GPU 只存储和计算自己负责的专家。  
**All-to-All 通信** —— 每个 GPU 要把自己的 token 发给负责对应专家的 GPU，同时接收其他 GPU 发来的 token，是 MoE 的主要通信瓶颈。  
**Grouped GEMM** —— 一批大小不同的矩阵乘法，对应不同专家收到不同数量的 token，是 MoE 的核心计算问题。  
**负载均衡**（Load Balancing）—— 让每个专家收到的 token 数量大体相等，防止"热专家"成为瓶颈，通常通过辅助损失（Auxiliary Loss）实现。  
**Token Dropping** —— 当一个专家的 token 超出容量上限时，丢弃多余的 token，是一种降低延迟但损失信息的策略。

---

## 8.1 MoE 架构回顾

标准 Dense FFN：每个 token 都经过同一套权重。

```
token x (H,)
  → W_1 (H, 4H): 全部激活
  → SiLU
  → W_2 (4H, H)
  → 输出 (H,)

FLOPs per token = 2 × H × 4H + 2 × 4H × H = 16H²
```

MoE FFN（以 8 专家、每 token 激活 Top-2 为例）：

```
token x (H,)
  → 门控网络 W_g (H, 8): softmax → top-2 专家得分 [s₁, s₂]
  → 路由到专家 E₃, E₇（举例）
  → E₃: W_1³ (H, 4H) → SiLU → W_2³ (4H, H) → 输出 o₃
  → E₇: W_1⁷ (H, 4H) → SiLU → W_2⁷ (4H, H) → 输出 o₇
  → 加权合并: x_out = s₁·o₃ + s₂·o₇

FLOPs per token = 2 × 2 × 16H² = 64H²（激活 2 个专家）
模型参数 = 8 × 2 × H × 4H = 64H²（但每个 token 只用 1/4）
```

**MoE 的吸引力**：参数量 8× 但每个 token 的 FLOPs 只有 2×（激活 2/8 专家）。在相同的计算预算下，MoE 可以部署更大的"名义参数量"，通常带来更好的能力。

---

## 8.2 MoE 的核心算子挑战：Grouped GEMM

问题的根源：**不同专家收到的 token 数量不一样**。

```
一个 batch 里，假设 S=1024 个 token，8 个专家，Top-2 路由：
  每个 token 贡献 2 个路由事件，共 2048 次路由
  理想均匀分布：每个专家 256 个 token
  
  实际分布（高度不均匀）：
  Expert 0: 312 tokens  ← 热专家
  Expert 1: 190 tokens
  Expert 2: 341 tokens  ← 最热
  Expert 3: 198 tokens
  Expert 4: 289 tokens
  Expert 5: 241 tokens
  Expert 6: 167 tokens  ← 冷专家
  Expert 7: 310 tokens
```

每个专家的 FFN 计算是一次 GEMM：

```
Expert i 的第一层：(n_i, H) × (H, 4H) = (n_i, 4H)
  其中 n_i 是到达专家 i 的 token 数
```

因为每个专家的 `n_i` 不同，这 8 次 GEMM 的 M 维度各不相同，无法直接合并为一次标准的 batched GEMM（batched GEMM 要求所有矩阵形状相同）。

### Grouped GEMM：解决不规则 GEMM

Grouped GEMM 允许一批形状不同的矩阵乘同时执行：

```
输入：K 个 GEMM 任务，每个任务 i 有：
  A_i: (m_i, k)    ← 每个专家的 token 数 m_i 不同
  B_i: (k, n)      ← 权重矩阵，同一层 n 相同
  C_i = A_i × B_i  ← 输出 (m_i, n)

CUTLASS Grouped GEMM：把 K 个 GEMM 打包在一次 kernel launch 里
```

**为什么不简单地 padding 到最大尺寸**：

```
假设最大 n_i = 341，最小 n_i = 167
  如果 pad 到 341：
  浪费计算 = (341 - 167) / 341 ≈ 51%
  对于 8 个专家，平均浪费约 25%

Grouped GEMM 按实际 m_i 分配计算资源，没有浪费。
```

**CUTLASS 3.x 的 Grouped GEMM 实现**：

```c++
// CUTLASS 3.x Grouped GEMM（简化）
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ...
    cutlass::gemm::GroupScheduler  // ← 关键：使用 Group 调度器
>;

// 传入每个组的 m, n, k 和指针
cutlass::gemm::GemmCoord problem_sizes[num_experts];
ElementA* a_ptrs[num_experts];  // 各专家的 token 指针
ElementB* b_ptrs[num_experts];  // 各专家的权重指针

for (int i = 0; i < num_experts; i++) {
    problem_sizes[i] = {n_tokens_per_expert[i], hidden_dim * 4, hidden_dim};
    a_ptrs[i] = expert_tokens[i];
    b_ptrs[i] = expert_weights[i];
}
```

---

## 8.3 All-to-All 通信：MoE 的带宽代价

在多 GPU（Expert Parallelism）设置下，不同专家放在不同 GPU 上。一个 token 路由到的专家可能在另一张 GPU 上，需要跨 GPU 发送这个 token 的激活值。

### All-to-All 的通信模式

```
设 EP（Expert Parallelism）度 = 4（4 张 GPU），每卡 2 个专家：

          GPU 0         GPU 1         GPU 2         GPU 3
专家分配: [E0, E1]     [E2, E3]     [E4, E5]     [E6, E7]

一个 batch 的路由结果：
  GPU 0 上的 token：部分要去 E2, E3（GPU 1），E4, E5（GPU 2），E6, E7（GPU 3）
  GPU 1 上的 token：部分要去 GPU 0, 2, 3
  ...

All-to-All：每张 GPU 向其他每张 GPU 发送数据，同时从每张 GPU 接收数据

通信量（以 EP=4 为例）：
  每个 token 的激活值大小 = H × 2 Bytes（FP16）= 2H Bytes
  一共 S/EP 个 token 需要发往其他 GPU（平均）
  All-to-All 通信量 ≈ 2 × S × H × 2 Bytes × (1 - 1/EP)
```

### 实际数字（DeepSeek-V3 规模）

DeepSeek-V3：H=7168，S=4096（序列长度），EP=32

```
All-to-All 通信量（前向，每层）：
  发送：4096 × 2（Top-2 路由）× 7168 × 2 Bytes / 32 × (32-1)/32
     ≈ 4096 × 7168 × 2 × 2 × 0.97 / 32
     ≈ 3.5 GB per layer（双向各一次 All-to-All）

H100 NVLink 带宽：900 GB/s（双向）
每层 All-to-All 时间 ≈ 3.5 / 900 ≈ 3.9 ms

  vs 专家 GEMM 计算时间：约 1–2 ms per layer
  
通信 > 计算！这是 MoE 扩展的核心障碍。
```

### All-to-All 的两种实现

**Two-phase All-to-All**（标准方式）：

```
Phase 1：发送 token 到目标 GPU
Phase 2：接收来自其他 GPU 的 token

GPU 上：[等待 All-to-All] → [Expert GEMM] → [等待 All-to-All]
        ↑ 纯通信等待                          ↑ 纯通信等待
```

**overlap 方式**（DeepSeek 实践）：

把 All-to-All 和 GEMM 流水线化，让通信和计算重叠：

```
时间 →
  通信：[发送 chunk 0][发送 chunk 1][发送 chunk 2]...
  计算：       [GEMM chunk 0] [GEMM chunk 1] [GEMM chunk 2]...
                  ↑
          先收到 chunk 0 就立刻开始计算，不等全部 token 到达
```

要实现这种 overlap，需要把 token 按目标 GPU 分组，用 CUDA Stream 分别发起通信和计算，是工程上较复杂的部分。

---

## 8.4 Expert Parallelism 的设计权衡

### EP 度 vs. 通信量

```
EP 度越大 → 每张 GPU 的专家越少 → 显存压力减小
           → 通信量增大（All-to-All 参与者更多）
           → 每个专家收到的 token 更少（小矩阵 GEMM，算术强度低）

EP=1：没有 All-to-All，所有专家在每张 GPU 上都有副本，显存最大
EP=N：极致分布，通信量最大，专家 GEMM 退化为极小矩阵
```

### 和其他并行的组合

MoE 训练通常同时用多种并行：

```
DP（Data Parallel）× TP（Tensor Parallel）× EP（Expert Parallel）× PP（Pipeline Parallel）

典型 DeepSeek-V3 训练配置：
  DP=16, TP=1, EP=32, PP=16
  ← TP=1 是因为 MoE 的通信已经很重，再加 TP 通信更不划算
```

**TP（张量并行）和 EP 的冲突**：
- TP 需要在每层做两次 AllReduce，通信量和 EP 的 All-to-All 叠加
- 大 EP 度下通常不用 TP，或用很小的 TP 度（2）

---

## 8.5 负载均衡：门控的艺术

路由不均匀（Load Imbalance）会严重拖累 MoE 的效率：

```
最坏情况：所有 token 都路由到 Expert 0
  Expert 0：需要处理 S 个 token（GEMM 很慢，串行瓶颈）
  Expert 1–7：空闲（浪费 GPU 算力）
  整体吞吐 ≈ 1/8 均匀情况
```

**辅助损失（Auxiliary Loss）**：

在训练时加一个额外的损失项，惩罚不均匀的路由分布：

```
L_aux = α × Σ_i (f_i × P_i)

f_i = 路由到 expert i 的 token 比例（实际分布）
P_i = 路由器对 expert i 的平均概率（理论分布）
α   = 辅助损失权重（通常 0.001–0.01）

当所有专家的 f_i 相等时，L_aux 最小（均匀分布）
```

**DeepSeek-V3 的 Expert Bias**：

DeepSeek-V3 没有用辅助损失（避免影响模型性能），而是在推理时给每个专家一个可学习的偏置 bias，在路由分数上加上 bias，使冷专家被更频繁地选择。这个 bias 在训练中保持固定，仅在推理时更新（一种在线负载均衡）。

---

## 8.6 Token Dropping vs. No-Drop

当一个专家已经达到容量上限（buffer size），新来的 token 怎么处理？

**Token Dropping（有容量限制）**：

```
每个专家设置 capacity factor c（通常 1.0–1.5）：
  最大 capacity = c × (S / num_experts)

超过 capacity 的 token 被丢弃（输出为零，或原样传递）
优点：固定计算量，GEMM 形状可预知，可以优化
缺点：丢弃信息，模型质量有损
```

**No-Drop（无容量限制）**：

```
每个专家处理所有路由到它的 token，不丢弃
优点：不损失信息
缺点：GEMM 形状随 batch 变化，需要 Grouped GEMM
     最热专家决定了整个 MoE 层的延迟（串行瓶颈）
```

DeepSeek-V3 在训练时使用 No-Drop + Grouped GEMM，在推理时用有限度的 Token Dropping（通过 Expert Bias 控制）配合固定形状优化。

---

## 8.7 共享专家：DeepSeek 的设计

DeepSeek-V2/V3 引入了"共享专家"（Shared Expert）的概念：

```
标准 MoE：N 个路由专家，每 token 选 TopK
DeepSeek MoE：Ns 个共享专家（每 token 必过）+ Nr 个路由专家（每 token 选 TopK）

优点：共享专家存储所有 GPU 上的通用知识，路由专家负责特化知识
    减少了每个专家的"被选但被丢弃"浪费

代价：每个 token 多一次 FFN forward（共享专家部分）
```

DeepSeek-V3 具体：256 个路由专家 + 1 个共享专家，Top-8 路由，即每 token 激活 8 个路由专家 + 1 个共享专家 = 9 个专家等效计算量。

---

## 8.8 实测：MoE 和 Dense 的效率对比

以 DeepSeek-V3（671B 参数，37B 激活）vs 假设的 37B Dense 模型对比：

```
指标                  DeepSeek-V3 MoE     同等激活量 Dense
────────────────────────────────────────────────────────
模型参数              671B                37B
每 token FLOPs        ~37B × 6 ≈ 222 G   ~37B × 6 ≈ 222 G
KV Cache 大小         相近（Attention 不变）相近
训练 tokens           14.8T              ~同等
推理延迟（批量）      较高（All-to-All 开销） 较低
推理吞吐（大 batch）  较低               较高
模型能力（PPL）       显著优于同激活量 Dense 基准
```

MoE 的价值在于**用更大的参数量（存储更多知识）而不增加计算量**，代价是通信复杂度和工程复杂度。

---

## 8.9 本章小结

```
MoE 的核心工程挑战：

1. Grouped GEMM：每个专家 token 数不同
   解法：CUTLASS Grouped GEMM，按实际大小分配资源

2. All-to-All 通信：跨 GPU 路由 token
   挑战：通信量 > 计算量（EP 大时）
   解法：通信-计算 overlap（按 chunk 流水线化）

3. 负载均衡：路由不均匀导致热专家串行瓶颈
   解法：辅助损失（训练时）/ Expert Bias（推理时）

4. Token Dropping vs No-Drop：
   训练用 No-Drop + Grouped GEMM（不丢信息）
   推理权衡固定形状优化和信息完整性

并行策略：
   MoE 训练：EP 为主，TP 通常不用或很小
   通信 = All-to-All（EP 内）+ AllReduce（DP 梯度）

DeepSeek 的核心工程贡献：
  FP8 训练 + 通信-计算 overlap + Expert Bias 负载均衡
  → 在 2048 张 H800 上达到 MFU 42%

下一章：推理场景的独特算子挑战——KV Cache 管理、连续批处理、投机采样。
```

**延伸资料**：
- [Switch Transformer](https://arxiv.org/abs/2101.03961)（Fedus et al., 2021，MoE 进入大模型的奠基工作）
- [Mixtral 8×7B 技术报告](https://arxiv.org/abs/2401.04088)（开源 MoE 的重要里程碑）
- [DeepSeek-V2 技术报告](https://arxiv.org/abs/2405.04434)（MLA + DeepSeekMoE 架构详解）
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)（§3 系统级 MoE 训练工程细节）
- [CUTLASS Grouped GEMM 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/24_gemm_grouped)
