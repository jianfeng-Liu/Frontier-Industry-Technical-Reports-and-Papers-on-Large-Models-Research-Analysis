# MILES 拆解：建在 slime 上的生产级 RL 框架

> **这一篇干什么**：MILES 是 slime 的下游，对标 [slime拆解.md](slime拆解.md) 的写法，重点讲清楚「MILES 加了什么、为什么要加」。
> **难度** ★★★
> **⚠️ 本文数字来源：MILES 技术报告（radixark，2026）。Dense 模型的 KL 数字、25%+ 加速数字均来自报告原文，MoE 相关改善幅度为本文推断，标 ⚠️。**

---

## 前置阅读

- [slime拆解.md](slime拆解.md) —— MILES 建在 slime 之上，slime 的架构和设计哲学不在本篇重复
- [04-训推一致性.md](04-训推一致性.md) —— Truly On-Policy 的背景，B-class / C-class 不一致的分类
- [大模型算子开发/10-RL算子.md](../大模型算子开发/10-RL算子.md) —— Rollout 阶段的算子图，Weight Offload 的代价分析

---

## 1 · 定位：slime 到 MILES 的一句话

slime 的设计目标是**快速迭代算法**——它明确拒绝抽象，换来了「上游升级零改动」。MILES 的设计目标是**生产规模稳定运行**——数百卡集群连续训练数天，不崩、不漂移、不 OOM。

两者的关系不是竞争，是分工：

```
slime：算法实验 → 跑通 → 找到 TIS/MIS 这类修正方案
         ↓
MILES：生产工程 → 稳跑 → Truly On-Policy + Online Speculative Decoding + 显存稳定性
```

✅ slime README 的 *"Ecosystem Built on slime"* 一节把 MILES 列为官方下游，不是第三方移植。

---

## 2 · slime 留下了哪些问题

slime 用 **TIS**（Token-level Importance Sampling，token 级重要性采样）和 **MIS**（Masked Importance Sampling，掩码重要性采样）解决了「算法层面的训推不一致」——当训练分布和生成分布发生偏移时，用重要性权重修正。这足以让研究性训练跑通，但有两个问题在生产规模会放大：

**问题一：算法修正是补丁，不是消除**

TIS/MIS 修正的是分布偏移的**影响**，没有消除偏移本身。K3 KL 散度（衡量训练端和推理端输出分布差距的指标，见 [04 章](04-训推一致性.md)）在 Dense 模型上仍然在 10⁻³ 量级，MoE 模型在 10⁻² 量级。对于需要精确策略梯度的长训练来说，这个持续的底噪会积累。

**问题二：Rollout 是最大的时间瓶颈，但 slime 没有加速它**

[10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8 的数据：Rollout（Decode 阶段）占 PPO 整体耗时约 45–55%，而 GPU 利用率只有 1–5%。标准 Speculative Decoding 可以加速 Rollout，但 RL 训练过程中 Actor 权重持续更新，草稿模型的接受率会随时间下滑，加速效果逐渐失效。

MILES 的三个主要模块分别对应这两个问题加上一个隐性问题（长时稳定性）。

---

## 3 · Truly On-Policy：比特级对齐

### 3.1 目标

把训练端和推理端对同一序列产生的 logits 对齐到**比特级相同**，K3 KL 散度压到 ~10⁻⁵。这不是修正，是消除偏移的根源。

### 3.2 四个算子统一

训推数值差异来自四个地方，MILES 逐一统一：

| 算子 | slime 状态 | MILES 选择 | 差异根因 |
|---|---|---|---|
| Attention | FA2（训练端）/ FA3（推理端） | 全部换成 FA3 | FA2 和 FA3 的 tile 大小与累加顺序不同，数值不等价 |
| GEMM | cuBLAS（训练端）/ DeepGEMM（推理端） | 全部换成 DeepGEMM | FP8 路径的量化缩放因子计算方式不一致 |
| Batch 不变性 | 无保证 | 引入 batch-invariant GEMM kernel | MoE expert 在不同 batch size 下排序不同，累加顺序改变结果 |
| 运行时路径 | eager mode | torch.compile | 不同运行时分支（有无梯度、有无 KV cache）走不同代码路径 |

### 3.3 Dense-only 的限制

**Truly On-Policy 目前只对 Dense 架构完全成立。**

MoE 模型存在「B-class 不一致」（见 [04 章 §4.x](04-训推一致性.md)）：专家激活是动态的，不同 batch 下同一序列可能激活不同专家组合，导致计算图的**拓扑**发生变化，单纯统一算子解决不了这个根因。

实测 K3 KL 散度对比：

| 模型架构 | slime（有 TIS/MIS） | MILES（Truly On-Policy） |
|---|---|---|
| Dense 7B | ~10⁻³ | **~10⁻⁵**（✅ 下降 2 个数量级）|
| Dense 32B | ~5×10⁻³ | **~10⁻⁵**（✅）|
| MoE 30B（A3B） | ~10⁻² | ~10⁻³（⚠️ 改善但无法归零）|

> ⚠️ MoE 的 10⁻³ 是本文基于报告「改善但未归零」描述的推断，报告未给出精确数字。

---

## 4 · Online Speculative Decoding：让草稿模型持续追随 Actor

### 4.1 问题

**标准 Speculative Decoding** 的加速原理：用一个小草稿模型（Draft Model）提前生成候选 token 序列，再由 Actor（大模型）验证。验证通过的 token 免去了 Actor 的逐 token Decode 代价。

在 RL 训练里，这个方案有一个致命缺陷：Actor 的权重每隔 K 步就更新一次，草稿模型如果保持不变，它和 Actor 之间的分布距离会单调增大，接受率持续下滑，到后期几乎没有加速效果。

### 4.2 解法：Online SFT 持续追随

MILES 在 RL 训练主循环的同时，对草稿模型做 **Online SFT**：每次 Actor 完成权重更新后，用 Actor 对当前 Rollout 序列的 logits 作为蒸馏目标，给草稿模型做一次小批量有监督微调。

```
一个 MILES 迭代：
  ① Actor Rollout（用当前 Actor + Draft 做 Speculative Decode）
  ② Policy Update（更新 Actor 权重）
  ③ Draft SFT（用新 Actor 的 logits 蒸馏 Draft）←新增
  ④ Draft 权重同步到 SGLang 推理引擎
  回到 ①
```

接受率因此能在整个训练过程中保持较高水平，而不是单调下滑。

### 4.3 四个工程要求

Online SFT 引入了四个额外的工程约束：

**① 序列打包（Sequence Packing）**

Draft SFT 的训练数据来自 Actor 的 Rollout，样本长度分布不均匀。用 varlen Attention 把不同长度的序列拼接成一条长序列，消除 padding 浪费（参见 [10-RL算子.md §10.6.1](../大模型算子开发/10-RL算子.md)）。

**② 上下文并行（Context Parallelism, CP）**

序列打包后单条序列很长，超出单卡容量。Draft SFT 需要 CP 分布式并行，不能只放在单卡上跑。

**③ 梯度隔离**

Draft 和 Actor 共享 LM Head 和 Embedding 层权重（节省显存）。Draft SFT 的梯度**不能传播**到这两层，否则会干扰 Actor 的训练分布。实现上用 `detach()` + 自定义 `requires_grad` 掩码在反向传播时截断。

**④ Megatron ↔ SGLang 权重同步**

Draft SFT 在 Megatron 训练框架里完成，但 Speculative Decode 在 SGLang 推理引擎里执行。每次 Draft SFT 之后，新权重需要同步到 SGLang。这复用了 slime 已有的 `update_weights` 机制（见 [03-权重同步.md](03-权重同步.md)），但需要额外维护一套针对 Draft 的同步路径。

### 4.4 实测效果

✅ Rollout 阶段加速约 **25%+**，在 response 长度较长（>1024 tokens）的任务上收益更显著（Decode 阶段更长，Speculative Decode 的收益窗口更大）。

---

## 5 · 显存稳定性：生产规模的防御工事

slime 的显存管理够用于研究性训练（运行数小时），不够用于连续数天的生产训练。MILES 逐一修了四类问题：

| 问题 | 根因 | MILES 解法 |
|---|---|---|
| 偶发 OOM | 长序列 Rollout 后激活显存未及时释放，下一个 Policy Update 阶段触发 OOM | 阶段切换时显式调用 `gc.collect()` + `torch.cuda.empty_cache()` |
| NCCL OOM | NCCL 通信缓冲区没有预留裕量，在 AllReduce 峰值时触发 | 配置参数 `nccl_mem_margin_bytes`，预先锁定一块裕量内存 |
| FSDP 显存泄漏 | FSDP（Fully Sharded Data Parallel，完全分片数据并行）的 `_unshard` 路径在部分情况下不释放临时 tensor | 用 weakref hook 在 forward 结束后强制触发释放 |
| Offload 灵活性不足 | 旧实现只支持全量 offload 或全量不 offload，无法细粒度控制 | 支持「按模块粒度」的 flexible offload：可以只 offload Reference Model 的部分层，保留 Actor 关键层在 GPU |

> **FSDP** = Fully Sharded Data Parallel，完全分片数据并行，PyTorch 原生的分布式训练范式。

---

## 6 · SLIME vs MILES 对比

| 维度 | SLIME | MILES |
|---|---|---|
| **定位** | 轻量级研究框架，快速算法迭代 | 生产级框架，面向大规模稳定训练 |
| **训推一致性** | TIS + MIS（算法层面修正偏移影响） | Truly On-Policy（算子层面消除偏移根因，Dense 专属）|
| **Rollout 加速** | 标准 Speculative Decoding（接受率随训练下滑）| Online Speculative Decoding（持续 SFT 维持接受率，稳定 25%+ 加速）|
| **显存稳定性** | 基础 offload + 基本 gc 调用 | OOM 恢复 + NCCL margin + FSDP fix + flexible offload |
| **适用规模** | 单机到约 64 卡集群 | 数百卡集群，连续数天训练 |
| **上游关系** | 独立框架（参见 [slime拆解.md](slime拆解.md)）| 建立在 SLIME 之上，复用调度、权重同步、TIS/MIS |
| **K3 KL（Dense）** | ~10⁻³ | **~10⁻⁵**（✅ 报告数字）|
| **K3 KL（MoE）** | ~10⁻² | ~10⁻³（⚠️ 推断，B-class 问题未根治）|

---

## 7 · 本篇小结

| 结论 | 分级 |
|---|---|
| MILES 建立在 slime 之上，复用 TIS/MIS、调度层、权重同步 | ✅ slime README 原文 |
| Truly On-Policy = 统一 FA3 + DeepGEMM + batch-invariant kernel + torch.compile | ✅ 报告 |
| Dense 模型 K3 KL：slime ~10⁻³ → MILES ~10⁻⁵，下降 2 个数量级 | ✅ 报告 |
| MoE 模型 B-class 不一致仍存在，Truly On-Policy 对 MoE 只能改善不能根治 | ✅ + ⚠️ |
| Online Speculative Decoding = 每轮 Actor 更新后对草稿模型做在线 SFT | ✅ 报告 |
| 四个工程要求：序列打包 + CP + 梯度隔离 + Megatron↔SGLang 权重同步 | ✅ 报告 |
| Rollout 加速 25%+（长序列任务收益更大）| ✅ 报告 |
| 显存四类修复：OOM gc、NCCL margin、FSDP weakref hook、flexible offload | ✅ 报告 |

**相关文档**：
- [slime拆解.md §8](slime拆解.md) —— MILES 的上游，生态关系
- [04-训推一致性.md](04-训推一致性.md) —— Truly On-Policy 的理论背景
- [大模型算子开发/10-RL算子.md](../大模型算子开发/10-RL算子.md) —— Rollout 算子图与 Speculative Decoding 算子成本
- [大模型算子开发/11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) —— B-class / C-class 不一致的算子层分析
