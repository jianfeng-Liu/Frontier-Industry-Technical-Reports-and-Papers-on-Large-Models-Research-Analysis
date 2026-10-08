# MILES 拆解：建在 slime 上的生产级 RL 框架

> **这一篇干什么**：MILES 是 slime 的下游，对标 [slime拆解.md](slime拆解.md) 的写法，重点讲清楚「MILES 加了什么、为什么要加」。
> **难度** ★★★

> ### ⚠️ 先说清楚本篇的证据等级，这一篇和另外两篇不一样
>
> | 标记 | 含义 | 本篇里指什么 |
> |---|---|---|
> | ✅ | **本次走查亲自核对过** | 只有对 **slime 源码**的引用（本地 clone 于 [源码/slime/](源码/slime/)，commit `3778dbf`） |
> | 🌐 | **转述自 MILES 技术报告**（radixark，2026） | ⚠️ **本次没有拿到一手 PDF 逐条回溯**，所有报告数字按「转述」处理，不按「实测」处理 |
> | ⚠️ | **本文的推断 / 判断** | 和报告结论分开写，不混 |
>
> ⚠️ **本地没有 MILES 的源码**（[源码/](源码/) 下只有 `slime/` 和 `OpenRLHF/`），所以本篇给不出 `文件:行号` 级别的引用。这是它和 [slime拆解.md](slime拆解.md) / [OpenRLHF拆解.md](OpenRLHF拆解.md) 最大的差别——那两篇是源码走查，这一篇是**报告走查 + 对上游 slime 源码的核对**。

---

## 0 · 七问速答：和 slime 并排看

这七个问题和 [slime拆解.md](slime拆解.md) §0、[OpenRLHF拆解.md](OpenRLHF拆解.md) §0 是同一组。

| # | 问题 | MILES 的答案 | slime（上游）的答案 |
|---|---|---|---|
| **Q1** | 架构档位 | 🌐 继承 slime 的调度层，没有换架构 | 同步 / 单步异步 / 全异步三档 |
| **Q2** | 训练后端 | 🌐 Megatron（继承） | Megatron-LM |
| **Q3** | rollout 引擎 / 怎么通信 | 🌐 SGLang（继承） | SGLang + Ray |
| **Q4** | 权重同步怎么做 | 🌐 复用 slime 的 `update_weights`，**额外维护一条给 draft model 的同步路径** | 四条通路（见 [slime拆解.md](slime拆解.md) §6） |
| **Q5** | 算法 / 加新算法 | 🌐 继承 slime；MILES 的增量不在算法层，在**算子层和工程层** | 6 个估计器 + 自定义 loss 路径 |
| **Q6** | 沙箱 / 环境 | 🌐 继承 slime 的 `slime/agent/` | `adapters/` + `harness/` 两层 |
| **Q7** | 一次迭代的时序 | ★ **比 slime 多一步**：Actor 更新完要给草稿模型做一次在线 SFT | 生成 → 训练 → 同步权重 |

⚠️ **一眼能看出的定位**：MILES **不改架构**，它改的是「同一个架构下每一步做得多干净」。所以读这一篇的正确姿势是：先读完 [slime拆解.md](slime拆解.md)，再把本篇当成一份 diff。

---

## 前置阅读

- [slime拆解.md](slime拆解.md) —— MILES 建在 slime 之上，slime 的架构和设计哲学不在本篇重复
- [04-训推一致性.md](04-训推一致性.md) —— 训推不一致的分类（A / B / C 三类）
- [大模型算子开发/10-RL算子.md](../大模型算子开发/10-RL算子.md) —— §10.8 为什么 rollout 是瓶颈，§10.10 治长尾的优先级
- [大模型算子开发/11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) —— §11.4 batch 不变性、§11.10.5「确定性 + batch 不变 kernel」那条路线的完整推演。**MILES 的 Truly On-Policy 走的就是那条路线，本篇不重复推导，只讲 MILES 在它上面加了什么。**

---

## 1 · 定位：slime 到 MILES 的一句话

slime 的设计目标是**快速迭代算法**——它明确拒绝抽象，换来了「上游升级零改动」（见 [slime拆解.md](slime拆解.md) §2）。MILES 的设计目标是**生产规模稳定运行**——数百卡集群连续训练数天，不崩、不漂移、不 OOM。

两者的关系不是竞争，是分工：

| | slime | MILES |
|---|---|---|
| **主要场景** | 算法实验：跑通、找到 TIS 这类修正方案 | 生产工程：稳跑数天 |
| **主要增量** | — | Truly On-Policy + Online Speculative Decoding + 显存稳定性 |
| **代码关系** | 上游 | 🌐 下游，复用调度、权重同步、损失函数 |

✅ slime README 的 *"Ecosystem Built on slime"* 一节把 **Miles** 列为官方下游（和 Dressage、vime、Relax、P1 等并列），不是第三方移植。这条是本次走查在 slime 仓库里亲自核对过的。

---

## 2 · slime 留下了哪些问题（以及一条要纠正的说法）

### 2.1 slime 的训推一致性做到哪一步

✅ 核对 slime 源码（`slime/backends/megatron_utils/loss.py`）：slime 在损失函数这一层提供了两种重要性采样修正：

| 函数 | 行号 | 做什么 | 对应 [04 章](04-训推一致性.md) 的哪一类 |
|---|---|---|---|
| `vanilla_tis_function` | 883 | **TIS**：把重要性比值 clamp 到 `[--tis-clip-low, --tis-clip]` 之间，超界的压回边界，**样本保留** | 统计层修正 |
| `icepop_function` | 907 | **icepop**：比值超界的 token 直接置零，**样本部分丢弃** | 统计层修正 |

> ✅ **TIS = Truncated Importance Sampling，截断重要性采样**。这是 slime 源码里的原文定义（`loss.py:944` 的 docstring：*"Optionally applies TIS (Truncated Importance Sampling) correction"*），也和 [附-速查表.md](附-速查表.md) 的表一致。
>
> ⚠️ **本文旧版把 TIS 写成了「Token-level Importance Sampling」，这是错的，已改。**

> ⚠️ **另一处**：本文旧版还写了「slime 用 TIS 和 **MIS**（Masked Importance Sampling，掩码重要性采样）」。✅ 本次在 slime 源码里**没有找到任何叫 MIS 的东西**，能找到的是上表两个函数，外加 `loss.py:1051` 注释提到的「rejection-sampling style masking（RS）」掩码。⚠️ 所以「MIS」这个缩写在本篇里**不再使用**；如果 MILES 报告里确有此词，它指的大概率就是这类掩码，但本文无法确认，不替报告下结论。

### 2.2 问题一：统计层修正是补丁，不是消除

TIS / icepop 修正的是分布偏移的**影响**，没有消除偏移本身。🌐 据 MILES 报告，K3 KL 散度（Schulman 的第三种 KL 估计量，用来衡量训练端和推理端输出分布的差距，三种估计量的对比见 [10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.7）在 Dense 模型上仍然在 10⁻³ 量级，MoE 模型在 10⁻² 量级。对于需要精确策略梯度的长训练来说，这个持续的底噪会积累。

⚠️ 这条和 [11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) §11.10.6 的那张五对策对比表是同一个判断：TIS（③）、CISPO（④）都是「统计层 / 目标函数层」的修正，栏目「消除差异？」全是「否」；**只有 ⑤「确定性 + batch 不变 kernel」那一行写的是「真消除」。** MILES 走的就是 ⑤。

### 2.3 问题二：Rollout 是最大的时间瓶颈

[10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8 的推演：Rollout 占整体耗时 **64%**（理论下界口径下是 77.3%）——它不是瓶颈之一，它就是瓶颈本身。

这里要分清两个常被混为一谈的「利用率」：Rollout 的**算力**利用率确实极低（batch=64 时约 14%，batch=1 时约 0.3%），但**带宽**利用率接近 100%。算力低是 Memory Bound（访存受限）的定义，是物理规律，不是浪费；真正被浪费掉的是**长尾**——一个批次里序列长度差异巨大，先生成完的槽位空转等最慢的那条，批次有效利用率只有 31.25%，折算下来吞掉整体约 33% 的时间（推演见 §10.2.1 与 §10.8.2）。

### 2.4 ⚠️ 这里有一处必须纠正：slime 已经支持在线训练草稿模型了

本文旧版的框架是：「标准 Speculative Decoding 的接受率会随训练下滑，**slime 没有解决，MILES 用 Online SFT 解决了**」。

✅ 本次走查在 slime 仓库里找到了直接反证——`docs/zh/advanced/speculative-decoding.md`（commit `3778dbf`）有一整节就叫**「在线 SFT draft model」**，原文：

> ✅ *"随着 RL 流程的进行，draft model 和 target model 的采样概率差异逐渐增大，能通过验证的 draft token 逐渐减少，spec 甚至可能造成负收益。**目前，slime 支持了在 RL 流程中在线训练 MTP 层，随着训练的进行同步更新 draft model，稳定提高了采样速度**。"*

✅ 对应的三个参数（`slime/utils/arguments.py:1515` 定义 `--enable-mtp-training`）：

```bash
--mtp-num-layers 1
--enable-mtp-training
--mtp-loss-scaling-factor 0.2
```

⚠️ **也就是说：「草稿模型接受率随训练下滑」这个问题，slime 自己就诊断了，也给了解法。** 把它写成「slime 留下的问题」是错的。

✅ 但同一份文档的**最后一句**给出了真正的边界：

> ✅ *"外部 draft model 的训练还在 **WIP**。"*（WIP = Work In Progress，仍在开发中）

所以准确的说法是：

| 草稿模型的形态 | slime（`3778dbf`） | MILES（🌐 报告） |
|---|---|---|
| **MTP 层**（Multi-Token Prediction，多 token 预测；和主模型同一个 checkpoint 里的附加层） | ✅ **已支持在线训练**（`--enable-mtp-training`） | 🌐 同样支持 |
| **外部独立 draft model**（单独一个小模型，和 Actor 共享 LM Head / Embedding） | ✅ 文档自陈 **WIP** | 🌐 **这才是 MILES 的增量**：完整的 Draft SFT 流程，见 §4 |

⚠️ **这是本次走查改掉的最主要一处错误。** 教训很具体：**写「下游框架 X 解决了上游 Y 的问题」之前，要先去 Y 的仓库里搜一遍，上游可能已经做了。** 两者的版本是滚动的，一份半年前写的对比表今天多半已经不成立。

---

## 3 · Truly On-Policy：把训推差异压到比特级

### 3.1 目标

🌐 把训练端和推理端对同一序列产生的 logits 对齐到**比特级相同**，K3 KL 散度压到 ~10⁻⁵。这不是修正，是消除偏移的根源。

⚠️ **这条路线的完整原理不在本篇**，在 [11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md)：§11.2 讲为什么浮点加法不满足结合律、§11.4 讲 batch 不变性、§11.10.5 讲「确定性 + batch 不变 kernel」的代价（⚠️ 该路径 −10% ~ −30% 性能）和适用范围。**本节只讲 MILES 在那条路线上具体选了什么。**

### 3.2 四个算子统一

🌐 据报告，训推数值差异来自四个地方，MILES 逐一统一：

| 算子 | 不统一时的状态 | MILES 的选择 | 差异根因 | 对应 11 章 |
|---|---|---|---|---|
| Attention | FA2（训练端）/ FA3（推理端）<br>FA = FlashAttention | 全部换成 **FA3** | 两代的 tile 大小与累加顺序不同，数值不等价 | §11.7.4 融合策略不同 |
| GEMM（General Matrix Multiply，通用矩阵乘） | cuBLAS（训练端）/ DeepGEMM（推理端） | 全部换成 **DeepGEMM** | FP8 路径的量化缩放因子计算方式不一致 | §11.6.2 精度格式 |
| Batch 不变性 | 无保证 | 引入 **batch-invariant GEMM kernel** | batch 大小会改变 kernel 的分块与 Split-K 策略，累加顺序随之改变 | §11.4 |
| 运行时路径 | eager mode（即时执行） | **torch.compile** | 不同运行时分支（有无梯度、有无 KV cache）走不同代码路径 | §11.7.4 |

⚠️ 对照 [11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) §11.10.5 那条公开参考实现（Thinking Machines Lab 的 `batch-invariant-ops`，覆盖 RMSNorm / matmul / attention 三个算子）：**MILES 的四条里有三条落在同一组算子上**，第四条（torch.compile 统一运行时路径）对应的是 §11.7.4 那条「让两端用同一套融合 kernel」。**这不是 MILES 独创的路线，是这条路线目前最完整的一次工程落地。**

### 3.3 Dense-only 的限制

🌐 **Truly On-Policy 目前只对 Dense（稠密）架构完全成立。**

MoE（Mixture of Experts，混合专家）模型存在 [04 章](04-训推一致性.md) 分类里的「B 类不一致」（同一个数学函数、两条算法路径）：专家激活是动态的，不同 batch 下同一序列可能激活不同专家组合，导致计算图的**拓扑**发生变化，单纯统一算子解决不了这个根因。

⚠️ 这条在 [11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) §11.9 有更硬的说法：**`topk` 是一个不连续函数**，它把浮点层面的小误差整流成「选了另一个专家」的大跳变，所以 MoE 上的误差不是线性放大，是跳变放大。

🌐 报告给出的 K3 KL 散度对比：

| 模型架构 | slime（有 TIS） | MILES（Truly On-Policy） | 证据等级 |
|---|---|---|---|
| Dense 7B | ~10⁻³ | **~10⁻⁵** | 🌐 报告数字，⚠️ 本文未核对一手 PDF |
| Dense 32B | ~5×10⁻³ | **~10⁻⁵** | 🌐 同上 |
| MoE 30B（A3B） | ~10⁻² | ~10⁻³ | ⚠️ **这一格是本文的推断**，见下 |

> ⚠️ **MoE 那一行的 10⁻³ 是本文根据报告「改善但未归零」的定性描述推出的数量级，报告没有给精确数字。** 不要把它当成报告结论引用。✅ 下降「2 个数量级」这个说法只对 Dense 两行成立。

---

## 4 · Online Speculative Decoding：让外部草稿模型持续追随 Actor

⚠️ 读这一节前请先看 §2.4：**MTP 层的在线训练 slime 已经有了**，MILES 在这一块的增量是**外部独立 draft model**（slime 文档自陈 WIP 的那一档）。

### 4.1 问题

**Speculative Decoding（投机解码 / 投机采样）** 的加速原理：用一个小草稿模型（Draft Model）提前生成候选 token 序列，再由 Actor（大模型）**一次批量验证**。验证通过的 token 免去了 Actor 的逐 token Decode 代价。

⚠️ 为什么这招在 RL 里值钱，[10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8.1 那张表给了答案：Rollout 的 Decode 阶段里，**权重读占整体 59.3%**，而且那是个**带宽硬下界**——每生成一个 token 就要把全部权重从 HBM 过一遍。投机解码的本质是**让一次权重读产出多个 token**，直接摊薄这个硬下界。

在 RL 训练里，这个方案有一个致命缺陷：Actor 的权重每隔 K 步就更新一次，草稿模型如果保持不变，它和 Actor 之间的分布距离会单调增大，接受率持续下滑，到后期甚至是负收益（✅ slime 自己的文档也是这么说的，见 §2.4）。

### 4.2 解法：Online SFT 持续追随

🌐 MILES 在 RL 训练主循环的同时，对草稿模型做 **Online SFT**（在线有监督微调）：每次 Actor 完成权重更新后，用 Actor 对当前 Rollout 序列的 logits 作为蒸馏目标，给草稿模型做一次小批量有监督微调。接受率因此能在整个训练过程中保持较高水平，而不是单调下滑。

### 4.3 ★ 一次 MILES 迭代的完整时序

这是本篇最该有的一张图：MILES 比 slime 多了第 ③ 步。

```
  time ---------------------------------------------------------------------------------->

  INFER |###### rollout (spec decode) #######|..... idle ......|... idle ....|= W1 =|= W2 =|
  TRAIN |............... idle ...............|# policy update #|# draft SFT #|= W1 =|= W2 =|

  legend:
    #### = busy      .... = idle
    =W1= = Actor 权重 --> SGLang
    =W2= = Draft 权重 --> SGLang    <-- 这条同步路径是 MILES 多出来的
```

逐步走一遍：

| 步 | 谁在算 | 谁在等 | 数据往哪流 | 是不是 MILES 新增 |
|---|---|---|---|---|
| ① Rollout | 推理卡：Actor 验证 + Draft 起草 | 训练卡空转 | 生成的序列 + logits 进 data buffer | 否（slime 已有） |
| ② Policy Update | 训练卡：Actor 前向/反向/优化器 | 推理卡空转 | 新 Actor 权重留在训练卡 | 否 |
| ③ **Draft SFT** | 训练卡：用**新 Actor** 的 logits 蒸馏 Draft | 推理卡继续空转 | 新 Draft 权重留在训练卡 | ★ **是** |
| ④ 权重同步（两条） | 两边：先推 Actor，再推 Draft | — | 训练卡 → SGLang | ★ **第二条是新增的** |

⚠️ **代价要说清楚：第 ③ 步是串在关键路径上的。** 它占用训练卡、延长一次迭代，而推理卡在这期间一直空着。🌐 报告给出的「Rollout 阶段加速 25%+」是**净收益**（已经减掉了 Draft SFT 的开销）还是**毛收益**，报告的表述不足以判断，⚠️ **本文无法确认，引用时请注意这个口径问题。**

⚠️ 另外注意 §4.3 这张图画的是**同步档**。如果跑在 slime 的单步异步档上（[slime拆解.md](slime拆解.md) §4.3），第 ③ 步可以和下一轮的 Rollout 重叠，代价会小得多——⚠️ 但报告没有说明实测是在哪一档下做的，**这是本文最拿不准的一处。**

### 4.4 Online SFT 引入的四个工程约束

🌐 据报告：

| # | 约束 | 为什么需要 | 延伸阅读 |
|---|---|---|---|
| ① | **序列打包**（Sequence Packing） | Draft SFT 的训练数据来自 Actor 的 Rollout，样本长度分布极不均匀。用 varlen（变长）Attention 把不同长度的序列拼成一条长序列，消除 padding 浪费 | [10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.6.1 算过这笔账：padding 浪费可达 61%~78% |
| ② | **上下文并行**（Context Parallelism, CP） | 序列打包后单条序列很长，超出单卡容量，必须沿序列维切到多卡 | [附-速查表.md](附-速查表.md) 的 RingAttention 条目是同一类手段 |
| ③ | **梯度隔离** | Draft 和 Actor 共享 LM Head 和 Embedding 层权重（节省显存）。Draft SFT 的梯度**不能传播**到这两层，否则会污染 Actor 的训练。🌐 实现上用 `detach()` + 自定义 `requires_grad` 掩码在反向传播时截断 | ⚠️ 这是「共享权重」这个省显存决策的必然代价 |
| ④ | **Megatron ↔ SGLang 权重同步** | Draft SFT 在 Megatron 里完成，投机解码在 SGLang 里执行。每次 SFT 之后新权重要同步过去。🌐 复用 slime 已有的 `update_weights` 机制，但要额外维护一条给 Draft 的同步路径 | [slime拆解.md](slime拆解.md) §6 讲了 slime 那四条通路长什么样 |

⚠️ **第 ③ 条是四条里最容易写错的。** 共享 LM Head / Embedding 省下的显存不小（词表 15 万 × hidden 4096 × 2 字节 ≈ 1.2 GB，Embedding 和 LM Head 各一份），但代价是从此每一次反向传播都要小心梯度会不会串过去。⚠️ 这类「省了显存、换来一个必须一直提防的坑」的取舍，在生产框架里非常典型。

### 4.5 效果

🌐 Rollout 阶段加速约 **25%+**，在 response 长度较长（>1024 tokens）的任务上收益更显著。

⚠️ **为什么长序列收益更大，可以自己验算**：投机解码省的是 Decode 阶段的权重读，而 Decode 的步数等于 response 长度。按 [10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8.1 的配置，Prefill 只占 1.6%、Decode 占 75.7%——**response 越长，Decode 占比越高，能被摊薄的那块越大。** ⚠️ 这个解释是本文的推演，报告只给了现象。

> ⚠️ **「25%+」这个数字没有附带测试配置**（哪个模型、多大 batch、什么 response 长度分布、哪一档架构）。按本仓库的惯例，**无配置的加速数字只能当量级参考**。

---

## 5 · 显存稳定性：生产规模的防御工事

slime 的显存管理够用于研究性训练（运行数小时），🌐 据报告不够用于连续数天的生产训练。MILES 逐一修了四类问题：

| 问题 | 根因 | 🌐 MILES 的解法 | ⚠️ 本文备注 |
|---|---|---|---|
| **偶发 OOM** | 长序列 Rollout 后激活显存未及时释放，下一个 Policy Update 阶段触发 OOM（Out Of Memory，显存溢出） | 阶段切换时显式调用 `gc.collect()` + `torch.cuda.empty_cache()` | ⚠️ 这是治症状，不是治病因；真正的病因是 PyTorch 的缓存分配器不会主动把空闲块还给驱动 |
| **NCCL OOM** | NCCL 通信缓冲区没有预留裕量，在 AllReduce 峰值时触发 | 配置参数 `nccl_mem_margin_bytes`，预先锁定一块裕量内存 | ⚠️ 本质是「显存预算要把通信库那一份算进去」。[附-速查表.md](附-速查表.md) 的显存账那一节算的是模型状态，**通信缓冲区是那张账之外的一块** |
| **FSDP 显存泄漏** | FSDP（Fully Sharded Data Parallel，完全分片数据并行）的 `_unshard` 路径在部分情况下不释放临时 tensor | 用 weakref（弱引用）hook 在 forward 结束后强制触发释放 | ⚠️ **这条值得留意**：slime 的训练后端是 Megatron 不是 FSDP（见 [slime拆解.md](slime拆解.md) §2），所以这一条说明 MILES 的适配面比 slime 宽，或者它在某些模型上走了 FSDP 路径。**本文无法从 slime 源码侧核对这一条。** |
| **Offload 灵活性不足** | 旧实现只支持全量 offload 或全量不 offload，无法细粒度控制 | 支持「按模块粒度」的 flexible offload：可以只 offload Reference Model 的部分层，保留 Actor 关键层在 GPU | ✅ 对照 slime：slime 的粒度是 `onload_weights` / `onload_kv` 两档（[slime拆解.md](slime拆解.md) §7），确实比「按模块」粗 |

---

## 6 · slime vs MILES 对比

| 维度 | slime | MILES | 证据 |
|---|---|---|---|
| **定位** | 轻量级研究框架，快速算法迭代 | 生产级框架，面向大规模稳定训练 | ✅ / 🌐 |
| **训推一致性** | TIS + icepop（统计层修正偏移的影响） | Truly On-Policy（算子层消除偏移根因，**Dense 专属**） | ✅ / 🌐 |
| **草稿模型（MTP 层）** | ✅ **已支持在线训练**（`--enable-mtp-training`） | 🌐 同样支持 | ✅ slime 文档 |
| **草稿模型（外部独立模型）** | ✅ 文档自陈 **WIP** | 🌐 完整的 Online SFT 流程（§4） | ✅ slime 文档 / 🌐 |
| **显存稳定性** | 基础 offload（`onload_weights` / `onload_kv` 两档） | OOM 恢复 + NCCL margin + FSDP fix + flexible offload | ✅ / 🌐 |
| **适用规模** | 🌐 报告口径：单机到约 64 卡 | 🌐 数百卡集群，连续数天 | ⚠️ 两个数字都来自报告的定性描述，**没有附带硬件配置** |
| **上游关系** | 独立框架 | 建在 slime 之上，复用调度、权重同步、损失函数 | ✅ slime README |
| **K3 KL（Dense）** | ~10⁻³ | **~10⁻⁵** | 🌐 报告 |
| **K3 KL（MoE）** | ~10⁻² | ~10⁻³ | ⚠️ **本文推断** |

⚠️ **这张表最容易被误读的一行是「适用规模」**：「slime 只能到 64 卡」不是 slime 的技术上限，而是报告在对比时给出的一个定性描述。✅ 事实上 slime 是 GLM-5 的生产基建（[01 章](01-格局全景.md)），GLM-5 的训练规模远超 64 卡。**所以这一行应该读成「MILES 在数百卡上做了针对性加固」，不是「slime 跑不了数百卡」。**

---

## 7 · ⚠️ 读这份报告时我拿不准的地方

把不确定的地方单独列一节，比混在正文里更诚实：

| # | 拿不准什么 | 为什么重要 |
|---|---|---|
| 1 | 「Rollout 加速 25%+」是净收益还是毛收益（扣没扣掉 Draft SFT 的开销） | 差别可能很大——Draft SFT 串在关键路径上（§4.3） |
| 2 | 那个加速数字测的是哪一档架构（同步 / 单步异步 / 全异步） | 异步档下 Draft SFT 能和 Rollout 重叠，收益口径完全不同 |
| 3 | MoE 的 K3 KL 到底是多少 | 本文填的 10⁻³ 是推断，报告只有定性描述 |
| 4 | FSDP 那条修复为什么会出现在一个 Megatron 系的框架里 | 说明 MILES 的后端覆盖面可能比 slime 宽，但无从核对 |
| 5 | 「slime 适用到约 64 卡」这个界是怎么得出的 | 和 slime 是 GLM-5 生产基建这件事表面冲突（见 §6 下方的说明） |

⚠️ **以上五条都需要一手 PDF 才能定。** 在拿到之前，本篇所有 🌐 数字请按「转述」使用，不要当成实测结论往外引。

---

## 8 · 本篇小结

| 结论 | 分级 |
|---|---|
| MILES 建在 slime 之上，被 slime README 列为官方生态项目 | ✅ slime README 原文 |
| **MILES 不改架构**，它的增量在算子层（Truly On-Policy）和工程层（显存、Draft SFT） | ⚠️ 本文判断 |
| ⚠️ **纠正**：TIS = **Truncated** Importance Sampling（截断重要性采样），旧版写成 "Token-level" 是错的 | ✅ slime `loss.py:944` docstring |
| ⚠️ **纠正**：slime 源码里**没有「MIS」这个东西**，实际是 `vanilla_tis_function` + `icepop_function` 两个函数 | ✅ slime `loss.py:883/907` |
| ⚠️ **纠正**：「草稿模型接受率随训练下滑」slime **自己就诊断了并给了解法**（`--enable-mtp-training`），不是 slime 留下的问题 | ✅ slime `docs/zh/advanced/speculative-decoding.md` |
| MILES 在投机解码上的真实增量是**外部独立 draft model**——slime 那份文档明确写着这块还是 WIP | ✅ slime 文档原文 |
| Truly On-Policy = 统一 FA3 + DeepGEMM + batch-invariant kernel + torch.compile | 🌐 报告 |
| 这条路线不是 MILES 独创，对应 [11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) §11.10.5 的「对策 ⑤」，⚠️ 该路径代价 −10%~−30% | ⚠️ 本文对照 |
| Dense 模型 K3 KL：~10⁻³ → ~10⁻⁵，下降 2 个数量级 | 🌐 报告（⚠️ 未核对一手 PDF） |
| MoE 的 B 类不一致仍存在：`topk` 不连续，小误差被整流成「选了另一个专家」的跳变 | 🌐 + ⚠️（原理见 §11.9） |
| MoE 的 K3 KL 具体值 | ⚠️ **本文推断，报告未给** |
| Online SFT 的四个工程约束：序列打包 + CP + 梯度隔离 + Draft 专用同步路径 | 🌐 报告 |
| ★ **一次 MILES 迭代比 slime 多两件事**：第 ③ 步 Draft SFT（串在关键路径上）+ 第二条权重同步路径 | 🌐 + ⚠️ 时序图为本文整理 |
| Rollout 加速 25%+，长 response 收益更大 | 🌐 报告（⚠️ **无测试配置，净/毛收益口径不明**） |
| 长序列收益更大的原因：Decode 步数 = response 长度，摊薄的是那个占 59.3% 的权重读硬下界 | ⚠️ 本文推演 |
| 显存四类修复：OOM gc、NCCL margin、FSDP weakref hook、flexible offload | 🌐 报告 |
| 「slime 适用到约 64 卡」不该读成技术上限——slime 是 GLM-5 的生产基建 | ⚠️ 本文澄清 |

**相关文档**：

- [slime拆解.md](slime拆解.md) —— MILES 的上游，§11 讲生态关系，§6 讲权重同步的四条通路
- [OpenRLHF拆解.md](OpenRLHF拆解.md) —— 同一组七问的第三份答案
- [04-训推一致性.md](04-训推一致性.md) —— A / B / C 三类不一致的分类
- [大模型算子开发/10-RL算子.md](../大模型算子开发/10-RL算子.md) —— §10.8 rollout 为什么是瓶颈，§10.10 治长尾的优先级清单
- [大模型算子开发/11-训推一致性算子层.md](../大模型算子开发/11-训推一致性算子层.md) —— §11.4 batch 不变性、§11.9 MoE 为什么特殊、§11.10 五条对策的代价对比
