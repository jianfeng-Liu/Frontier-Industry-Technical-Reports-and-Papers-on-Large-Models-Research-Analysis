# GLM-5.2 架构与技术报告解析

> 📘 **不懂模型架构？先读这份**：[大模型架构入门/](大模型架构入门/README.md) —— **17 章 + 速查表**，从零讲清 注意力 / KV cache / MLA / MoE / RoPE / MTP / Muon / 量化，**不预设任何架构基础**。本文用到的术语都能在那里找到解释。

> **重要前提**：**GLM-5.2 没有独立的技术报告**。官方发布形式是技术博客 + Hugging Face 模型卡，模型卡本身把技术细节指向两篇论文。所以本文的构成是：
>
> | 内容 | 来源 | 可靠性 |
> |---|---|---|
> | 骨干架构、训练配方、后训练、Infra | **GLM-5 技术报告 arXiv:2602.15763**（40 页，正文提取核对） | 高（一手） |
> | IndexShare 的原理与实验 | **IndexCache 论文 arXiv:2603.12201**（18 页，正文提取核对） | 高（一手，但实验在 30B 模型 + GLM-5 预实验上做） |
> | GLM-5.2 的具体增量（1M、IndexShare 配置、MTP 改进、effort 档） | 官方博客 + HF 模型卡 | 中（无论文，只有厂商自述数字） |
> | GLM-5.2 自身的参数量/训练增量/RL 数据/长上下文召回曲线 | **缺失** | — |
>
> **这是 GLM-5.2 和 Kimi K3 / DeepSeek V4 最大的差别**：后两者有完整的一手技术报告，GLM-5.2 的"5.1 → 5.2"这一段增量目前只有厂商自述，无第三方可复核的论文。
>
> **本地论文**：[papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering-arXiv-2602.15763.pdf](papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering-arXiv-2602.15763.pdf)（40 页）· [papers/GLM-5-正文提取.txt](papers/GLM-5-正文提取.txt) · [papers/IndexCache-Cross-Layer-Index-Reuse-arXiv-2603.12201.pdf](papers/IndexCache-Cross-Layer-Index-Reuse-arXiv-2603.12201.pdf)（18 页）· [papers/IndexCache-正文提取.txt](papers/IndexCache-正文提取.txt)
> **横向对比**：见 [模型技术对比.md](模型技术对比.md)（DeepSeek-V4 / Kimi K3 / GLM-5.2 / MiniMax M3 / openPangu-2.0 五方）
> **同系列**：[DeepSeek-V4.md](DeepSeek-V4.md) ｜ [Kimi-K3.md](Kimi-K3.md) ｜ [MiniMax-M3.md](MiniMax-M3.md) ｜ [openPangu-2.0-Pro.md](openPangu-2.0-Pro.md)

---

## 一、一句话总览

**GLM-5.2 = 744B 总参 / 40B 激活的稀疏 MoE，MLA + DSA 注意力，1M 上下文，MIT 许可。它是三家里唯一一个把"Agentic 工程"当成第一目标来设计的模型。**

GLM-5 报告的标题就是它的纲领：**《GLM-5: from Vibe Coding to Agentic Engineering》**——从"凭感觉写代码"到"工程化的智能体"。整篇报告的重心不在架构创新，而在**后训练与 RL 基建**：40 页里有超过一半在讲 Agentic RL、环境构建、异步 RL 基础设施。

三代演进：

| | GLM-4.5 | GLM-5 | GLM-5.1 | GLM-5.2 |
|---|---|---|---|---|
| 发布 | 2025 | 2026-02 | 2026-04 | **2026-06-16** |
| 总参 / 激活 | 355B / 32B | **744B / 40B** | 754B | 753B（HF 标注） |
| 上下文 | 128K | **200K** | 200K | **1,048,576（1M）** |
| 注意力 | GQA | **MLA-256 + DSA** | 同 | **+ IndexShare** |
| 许可 | — | — | — | **MIT** |

> **参数量口径注意**：报告 Table 10 的 744B **包含 MTP 层但不含词嵌入和输出层**；HF 模型卡标注 753B 是全量口径。两个数字都对，只是口径不同。

---

## 二、骨干架构（来自 GLM-5 报告 §2.1 + 附录 Table 10）

| 配置项 | GLM-4.5 | **GLM-5 / 5.2** |
|---|---|---|
| 总参数 | 355B | **744B** |
| 激活参数 | 32B | **40B** |
| Dense 层 | 3 | 3 |
| MoE 层 | 89 | **75** |
| MTP 层 | 1 | 1 |
| Hidden Dim | 5120 | **6144** |
| Dense 中间维 | 12288 | 12288 |
| MoE 中间维 | 1536 | **2048** |
| QK Head Dim | 128 | **192** |
| V Head Dim | 128 | **256** |
| Q LoRA Dim | – | **2048** |
| KV LoRA Dim | – | **512** |
| 注意力头数 | 96 | **64** |
| KV 头数 | 8 | –（改用 MLA） |
| Indexer 注意力头 | – | **32** |
| Indexer Head Dim | – | **128** |
| 专家总数 | 160 | **256** |
| 路由专家（每 token） | 8 | 8 |
| 共享专家 | 1 | 1 |
| 词表 | 151552 | **154880** |

**一个设计取向很明确**：GLM-5 从 GLM-4.5 的 89 层 MoE **减到 75 层**，同时把专家数从 160 加到 256。报告给的理由是**"减少专家并行的通信开销"**——层数越多，EP 通信轮次越多。这和 Kimi K3 把层数从 61 加到 93 的取向正好相反。

> **一个小的口径矛盾**：报告正文写 "reduces its layer count to **80**"，但附录 Table 10 是 3 dense + 75 MoE + 1 MTP = 79。Raschka 的笔记和 HF 配置按 78 个 Transformer 层（3+75）计。以配置表为准。

---

## 三、改进点 1：MLA-256 + Muon Split —— 让 MLA 追平 GQA

这是 GLM-5 报告里**唯一一处真正的算法创新**，而且很值得注意。

### 问题：MLA 在 Muon 下打不过 GQA

Z.ai 想用 MLA（低维 KV latent，省显存、长上下文快），但实测发现：**576 维 latent KV cache 的 MLA，性能追不上 2048 维 KV cache 的 GQA-8**。

| 数据集 | Hellaswag | MMLU | C-Eval | RACE | BBH | GSM8K | HumanEval |
|---|---|---|---|---|---|---|---|
| GQA-8 | 77.3 | 61.2 | 60.0 | **79.6** | **53.3** | **47.6** | **38.5** |
| MLA | 77.3 | 61.5 | 59.7 | 77.8 | 48.9 | 46.2 | 33.5 |
| **MLA + Muon Split** | **77.8** | **62.5** | **62.1** | **79.9** | 51.8 | 45.0 | 36.7 |
| MLA-256 + Muon Split | 77.4 | 62.0 | 59.9 | 79.6 | 51.3 | 47.5 | 36.6 |

MLA 在 BBH 上掉了 4.4 分、HumanEval 掉了 5 分。

### 解法：Muon Split —— 逐 head 正交化

原来的 Muon 配方是对**完整的** `W_UQ, W_UK, W_UV` 上投影矩阵做正交化。Z.ai 的改动：

> **把这些矩阵按 head 切成小矩阵，对每个 head 的独立小矩阵分别做正交化。**

理由：让不同注意力头的投影权重**以不同的尺度更新**。

**MLA + Muon Split 直接把 MLA 拉回 GQA-8 水平**，BBH 从 48.9 回到 51.8，HumanEval 从 33.5 回到 36.7，MMLU/C-Eval 甚至反超。

**副产品**：报告说用了 Muon Split 之后，**GLM-5 预训练全程注意力 logits 的尺度保持稳定，不需要任何 clipping 策略**。

> ⚠️ **这里有一个时间线上的重要事实**：Muon Split 出现在 **GLM-5 报告（2026-02）**，而 Kimi K3 的 **Per-Head Muon（2026-07）是同一个想法**——都是把注意力投影按 head 切开做 Newton-Schulz 正交化，都报告了"避免大尺度 head 主导更新"的同一个动机。**Z.ai 早了五个月。** 详见 [模型技术对比.md](模型技术对比.md)。

### MLA-256：为解码省算力

MLA 的另一个缺点是**解码贵**：MLA 做的是 576 维点积，而 GQA 只有 128 维。而且 DeepSeek-V3 的头数是按 H800 的 roofline 选的，换硬件就不合适。

GLM-5 的调整：**head dim 从 192 提到 256，头数减少 1/3**。

```
训练/prefill 计算量和参数量保持不变（MLA 在这两个阶段是 MHA 形态）
解码计算量下降
```

Table 1 最后一行验证了 MLA-256 在 Muon Split 下性能不掉。

---

## 四、改进点 2：MTP 参数共享

**问题**：要预测后 n 个 token，训练时需要 n 个 MTP 层，MTP 参数和 KV cache 的显存**随投机步数线性增长**。DeepSeek-V3 的做法是只训 1 个 MTP 层、推理时预测 2 个 token——**训练-推理不一致，导致第二个 token 的接受率下降**。

**GLM-5 的做法**：**训练时让 3 个 MTP 层共享参数**。

- 显存开销和 DeepSeek-V3 持平（只有一份参数）
- 接受率提升

| 模型 | Accept Length（同为 4 个投机步） |
|---|---|
| DeepSeek-V3.2 | 2.55 |
| **GLM-5** | **2.76** |

### GLM-5.2 的进一步改进（来自博客/Raschka，非论文）

- 第一个 draft step 计算 indices，**后续 draft steps 复用这些 indices 和之前的 KV cache**（给 draft 模型也用上 IndexShare，把开销压到最小）
- 消除 GLM-5.1 MTP 层的训练-推理不一致
- 配合 rejection sampling + 端到端 total-variation loss
- **消融结果：平均接受长度 4.56 → 5.47，+20%**（但报告说改动是累加的，IndexShare 单独贡献多少不清楚）

---

## 五、改进点 3：DSA 的引入方式 —— 极省的 continued pre-training

GLM-5 没有从零训稀疏注意力，而是**从 dense 基座做续训**转成 DSA。

DSA（DeepSeek Sparse Attention）的机制：一个轻量 **lightning indexer** 给每个 query 对所有前序 token 打分，选 **top-2048**，主注意力只在这个子集上算。核心注意力从 `O(L²)` 降到 `O(Lk)`。

**两阶段"dense 预热 + 稀疏适配"**：

```
预热阶段：1000 步，每步 14 条 × 202,752 tokens 的序列，最大 lr 5e-3 → 2e-4
稀疏适配：沿用 mid-training 的数据和超参，共 20B tokens，恒定 lr 1e-5
```

**最值得记的一个数字对比**：

| | 稀疏适配预算 |
|---|---|
| DeepSeek-V3.2 | **943.7B tokens** |
| **GLM-5** | **20B tokens** |

**GLM-5 只用了 DeepSeek 1/47 的预算，就把 DSA 模型适配到与原 MLA 模型持平。**

| 长上下文基准 | MQ-NIAH-128k | MV-NIAH-128k | SQuAD-128k | HotpotQA-128k |
|---|---|---|---|---|
| MLA | 100.0 | 95.5 | 79.7 | **66.3** |
| DSA | 100.0 | **97.0** | **86.0** | 63.0 |

报告还补了一步验证：用同一批 SFT 数据分别微调 DSA 和 MLA 模型，**训练 loss 和评测基准打平**。

### 附带的高价值消融：其他高效注意力方案为什么没选

报告在 GLM-9B（40 层）上系统对比了几种替代方案，**这张表对做长上下文的人很有参考价值**：

| RULER（无额外训练） | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| GLM-9B（全注意力） | 95.19 | 93.67 | 92.01 | 91.09 | **85.35** | **75.28** |
| **SWA Interleave**（固定交错） | 94.87 | 54.02 | 25.89 | 12.61 | 8.32 | **6.51** |
| **SWA Pattern**（搜索得到） | **95.78** | 92.54 | 88.92 | 82.52 | 70.23 | 53.95 |

**结论极其鲜明：固定交错的滑窗注意力在长上下文上是灾难性的**（128K 只剩 6.51），而**搜索出来的滑窗模式能到 53.95**，差了 8 倍。

SWA Pattern 的做法（受 PostNAS 启发）：beam search，beam size 8，每步优化 2 层，GLM-9B 40 层约 10 步收敛，**只在 16K 上搜索**，得到的模式是：

```
SFSSFFSSSFFFFSSFSFFFFFFSFSFSSFSSFSFSSFSSS      (S=滑窗, F=全注意力)
```

**这个只在 16K 上搜出来的模式，在所有测试长度上都保持有效**——长度泛化能力很强。

报告还提了 **SimpleGDN**：为最大化复用预训练权重而设计的极简线性化策略——**移除 Conv1d 和显式门控模块**，直接把预训练的 Q/K/V 投影权重映射进线性递归形式，不引入任何额外参数。

> 这一节和 Kimi K3 的路线选择正好构成对照：**Z.ai 系统评估过线性注意力（GDN / SimpleGDN）并选择了不用，Kimi K3 选择了全面押注线性注意力（KDA）。** 两家看的是同一批候选方案，做了相反的决定。
>
> 📌 **DSA 的直接同类是 MiniMax 的 MSA**：两者都是"精确 softmax + 动态稀疏选择"，而且都落在 **2048 token 预算**上（DSA 是 token 级 top-2048，MSA 是 16 块 × 128）。区别在底座（MLA/MQA vs GQA）、粒度（token vs 块）、选择归属（全头共用一个 Top-k vs 每个 GQA 组独立选）和 indexer 训练方式（DSA 走 LM 梯度，MSA 用 KL 对齐 + 梯度 detach）。逐条对照见 [MiniMax-M3.md](MiniMax-M3.md) §九。
> 另外，上面那张 SWA 消融表和 MiniMax M2 报告 §2.2.2 的 SWA 表是**两家独立做出来的同向证据**——固定位置的滑窗在长上下文上崩，动态按内容选不崩。

---

## 六、改进点 4：IndexShare —— GLM-5.2 的核心增量

这是 5.1 → 5.2 唯一的架构改动，背后论文是 **IndexCache（arXiv:2603.12201）**。

### 问题：DSA 省了主注意力，但 indexer 本身还是 O(L²)

DSA 把核心注意力降到 `O(Lk)`，但**每一层的 indexer 仍然要拿 query 和所有前序位置比一遍**，`N` 层总成本 `O(NL²)`，随上下文长度平方增长。

论文实测的 **indexer 占总延迟的比例**（30B DSA 模型）：

| 上下文长度 | 10K | 60K | 120K | 200K |
|---|---|---|---|---|
| **Prefill** | 27% | 50% | 68% | **81%** |
| **Decode** | 27% | 31% | 38% | 41% |

**200K 上下文下，prefill 阶段 81% 的时间花在 indexer 上。** 长上下文 DSA 的真正瓶颈已经从主注意力转移到了 indexer。

### 关键观察：相邻层选的 token 高度重合

论文计算了所有层之间 top-k 索引的两两重合度：**相邻层共享 70%~100% 的选中 token**，热力图还显示出明显的层簇结构。

> 那就没必要每层都算一次 indexer。

### 机制：Full 层 / Shared 层

```
把层分成两类：
  Full   层：跑自己的 indexer，算出 top-k 位置列表
  Shared 层：直接复用最近的前一个 Full 层的位置列表

GLM-5.2 的模式：full, shared, shared, shared   （每 4 层一个 indexer，配置开头有几处例外）
推理时只多一个条件分支
```

**注意 Shared 层省的只是"选哪些位置"**：它仍然要算自己的 query、注意力权重、value 组合、输出投影和 FFN。**只有位置列表被复用**，稀疏注意力本身每层都还在跑。

### 两种配置方法（论文提出）

| | Training-free IndexCache | Training-aware IndexCache |
|---|---|---|
| 适用 | 任何现成 DSA 模型，不改权重 | 需要训练 |
| 方法 | **贪心层选择**：直接在小校准集上最小化 LM loss 来决定保留哪些层的 indexer | **多层蒸馏 loss**：每个保留的 indexer 对着它所服务的所有层的**平均注意力分布**做训练 |
| 效果 | 论文指出**朴素均匀交错会掉点**，贪心搜索能救回来 | 让**简单的均匀交错模式也能追平全 indexer 精度** |

### 效果数字（注意这里有三组，别混）

| 场景 | 移除比例 | 收益 |
|---|---|---|
| 论文主实验（**30B DSA 模型**） | 75% indexer | **1.82× prefill / 1.48× decode 加速**，质量几乎无损 |
| 论文的 GLM-5 预实验 | 50% indexer | **约 1.2× 端到端加速**，长上下文与推理任务表现相当 |
| **GLM-5.2 出货配置**（厂商自述） | 75%（每 4 层 1 个） | **1M 上下文下每 token FLOPs 降低 2.9×** |

⚠️ **2.9× 这个数字要打折看**：

1. 它是**该上下文长度下的架构算力估算，不是端到端加速比**。
2. Z.ai 自己说明 **KV cache 显存不会等比例缩小**。1M 下的服务成本仍然包含 cache 容量、kernel、CPU 调度、cache 传输。
3. 30B 模型上的 1.82×/1.48× 是**间接旁证**，不是 GLM-5.2 的实测延迟数据。

### 训练方式：不是推理时的补丁

**IndexShare 是在 128K 序列的 continued mid-training 阶段引入的**，让每个保留下来的 indexer 有机会适应依赖它的那些层。这和"给训好的模型在推理时挂一个 cache"是两回事。

---

## 七、预训练与 Mid-Training

### 数据

- **Web**：在 GLM-4.5 pipeline 上加了一个基于句子嵌入的 DCLM 分类器；针对长尾知识专门做了一个 **World Knowledge 分类器**（用 Wikipedia 条目 + LLM 标注优化），从中低质量数据里捞有价值信息
- **Code**：刷新代码托管平台快照 + 更多含代码网页，模糊去重后**唯一 token 增加 28%**；修正 Software Heritage 的元数据对齐问题；为 Scala / Swift / Lua 等低资源语言单独训分类器
- **Math & Science**：LLM 打分只留最有教育价值的内容；长文档用 **chunk-and-aggregate 打分算法**；**严格排除合成、AI 生成、模板化数据**

### Mid-Training：三段扩上下文

```
32K   → 1T tokens
128K  → 500B tokens
200K  → 50B tokens
```

- **软件工程数据**：repo 级代码文件 + commit diff + GitHub issue + PR + 相关源文件拼成统一训练序列。放宽仓库级筛选、加强 issue 级质量过滤，得到**约 1000 万个 issue–PR 对**，过滤后**约 160B 唯一 token**
- **长上下文数据**：自然 + 合成。合成部分受 NextLong / EntropyLong 启发，把高相似文本**交错打包**构造长程依赖，缓解 lost-in-the-middle；200K 阶段额外掺入 MRCR-like 数据强化多轮召回
- **一个有意思的经验**：在 128K 阶段之后再加一个 200K mid-training 阶段，**连 128K 窗口内的表现也一起提升了**

### 超参（附录 A）

```
优化器：Muon，cosine decay，batch size warmup
预训练 lr：0 → 2e-4（warmup）→ 4e-5（decay）
mid-training lr：4e-5 → 1e-5（线性）
DSA 预热 lr：5e-3 → 2e-4
DSA 稀疏适配 lr：恒定 1e-5
```

---

## 八、后训练：五阶段 + On-Policy Cross-Stage Distillation

```
① SFT（同时开 INT4 QAT）
② Reasoning RL
③ Agentic RL
④ General RL
⑤ On-Policy Cross-Stage Distillation  ← 最后一阶段
```

### ⑤ 为什么需要这一步

**多阶段 RL 顺序优化不同目标时，会累积性地损害先前获得的能力。** 所以最后加一个 on-policy 蒸馏阶段，**把前面 SFT 和各 RL 阶段学到的技能快速找回来**。

**做法**：前面各阶段的**最终 checkpoint 作为教师模型**，训练 prompt 从对应教师的 RL 训练集里按比例混合采样。把 GRPO 目标里的 advantage 项换成：

```
Â_{i,t} = sg[ log ( π_teacher^infer(y_{i,t} | x, y_{i,<t}) / π_θ^train(y_{i,t} | x, y_{i,<t}) ) ]
```

`sg` = stop gradient。

**两个工程细节**：

- GRPO 的 **group size 设为 1**、batch size 1024。因为 advantage 直接来自与教师的差距，**不再需要每个 prompt 采一大组样本来估计 advantage**，所以可以把吞吐拉满。
- 目前用推理引擎取教师 logits；未来计划迁到训练引擎，统一用 MLA 的 MQA 模式做推理。

> ⚠️ **这个公式和 Kimi K3 的 MOPD 是同一个形式**：都是 `log(教师概率 / 学生概率)` + stop-gradient，都当成 RL 的 advantage/reward 而不是写成 KL loss。**GLM-5 报告是 2026-02，Kimi K3 是 2026-07。** 另外，GLM 用它是为了**跨阶段找回被 RL 遗忘的能力**，K3 用它是为了**把 9 个领域专家合并成一个模型**——同一个工具，两种用法。

### GLM-5.2 的增量

- **Effort 档位**：编程支持多档思考力度，平衡性能与延迟。官方称同等 token 预算下，编程能力位于 Claude Opus 4.7 和 Opus 4.8 之间
- 训练使用自研 **slime** 框架做大规模 Agentic RL 和 OPD

---

## 九、Agentic RL：这是 GLM-5 报告真正的重心

### 异步 RL 设计

**问题**：agentic rollout 有严重长尾，同步 RL 会产生巨大的 GPU 空转气泡。

**做法**：**训练引擎和推理引擎解耦到不同 GPU 设备上**。

```
推理引擎持续生成轨迹
  → 累积到阈值就把这批发给训练引擎更新模型
  → 训练引擎每 K 次梯度更新把新权重推回推理引擎
```

**代价与对策**：不同轨迹可能由不同版本的模型生成，引入严重 off-policy 问题。GLM-5 的处理是——**每次推理引擎权重更新后重置优化器**（因为 rollout 策略变了，优化问题本身也变了）。

### Server-based 多任务训练

不同 agentic 任务用不同工具集和 rollout 逻辑。GLM-5 引入 **Multi-Task Rollout Orchestrator**：

- 每个任务把自己的 rollout 和 reward 逻辑实现成**独立微服务**，注册到中央编排器
- 编排器控制**每个任务的 rollout 比例和生成速度**，保证跨任务数据均衡
- **所有 agentic 任务的轨迹统一成同一种 message-list 表示**
- 支撑 **1k+ 并发 rollout**，任务采样比例可自动动态调整

### TITO：token-in-token-out

一个容易被忽略但报告专门强调的工程点：

| | 做法 | 问题 |
|---|---|---|
| **Text-in-Text-out** | 推理引擎当黑盒返回文本，训练侧**重新 tokenize** | 重新分词会在 token 边界、空白/归一化、截断、特殊 token 位置上引入细微错配，**破坏 action 和 reward/advantage 的步骤对齐** |
| **Token-in-Token-out（GLM-5 采用）** | 训练直接消费推理引擎产出的**精确 token 流** | 保持采样内容与优化内容的 action 级严格对应 |

实现上做了一个 **TITO Gateway**，拦截所有 rollout 的生成请求并记录每条轨迹的 token ID 和元数据，把繁琐的 token ID 处理从 agent rollout 逻辑里隔离出来。报告的判断是：**TITO 对异步 RL 训练是关键性的。**

---

## 十、Infra 与国产芯片适配

### 训练 Infra（GLM-5 §2.4）

- **灵活的 MTP 放置**：MTP 输出层和主输出层**共置在最后一个 pipeline stage 上共享参数**，embedding 和 transformer 部分放在前一个 stage，平衡 pipeline 各 rank 的显存
- **Pipeline ZeRO2 梯度分片**：每个 stage 只存 1/dp 的梯度；只保留两个完整累积 buffer 做双缓冲滚动
- **Muon 分布式优化器零冗余通信**：只对每个 rank 自己拥有的参数分片做 all-gather，并与本地计算重叠
- **Pipeline 激活 offload**：前向后把激活卸到主存，反向前载回，**层粒度**，调度上避开与 P2P 通信和 MoE token 路由的争抢
- **序列分块的输出投影**：把序列切块独立算投影和 loss，块数越多峰值显存越低
- **INT4 QAT 在 SFT 阶段进行**，量化 kernel 同时用于训练和离线权重量化，**保证训练与推理 bitwise 一致**

### 国产芯片适配（GLM-5 §5，是三份报告里唯一有这一章的）

GLM-5 完成 **7 个国产芯片平台**全栈适配（华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦、燧原）；GLM-5.2 博客称扩到 **9 个**（增加平头哥、壁仞、天数智芯），且**上线首日即完成适配**。

以昇腾 Atlas 为例：

- **混合精度 W4A8 量化**：为把 750B 模型装进**单台 Atlas 800T A3**——标准 Attention 和 MLP 块用 **W8A8（INT8）**，MoE 专家压到 **W4A8（INT4）**。用 QuaRot 做离群值抑制、Flex_AWQ_SSZ 做缩放校准
- **融合 kernel 三件套**：
  - **Lightning Indexer**：把打分、ReLU、TopK 融成单 kernel，让 NPU 能重叠计算与访存
  - **Sparse Flash Attention**：并行处理 KV cache 的 TopK 选择和稀疏注意力计算
  - **MLAPO**：把 **13 个预处理小算子融成一个"超级算子"**，利用 Vector 和 Cube 单元的并行
- **推理引擎优化**：vLLM 里把 D2H 采样拷贝与下一步 decode 准备重叠，消除调度气泡；RadixCache 前缀共享 + Prefix Cache 扩展 KV 到系统内存；Attention DP + MoE EP 混合并行 + FlashComm

---

## 十一、评测（GLM-5.2 官方模型卡完整表）

对比对象：GLM-5.1、Qwen3.7-Max、MiniMax M3、DeepSeek-V4-Pro、Claude Opus 4.8、GPT-5.5、Gemini 3.1 Pro。

| Benchmark | **GLM-5.2** | GLM-5.1 | Qwen3.7-Max | MiniMax M3 | DeepSeek-V4-Pro | Opus 4.8 | GPT-5.5 | Gemini 3.1 Pro |
|---|---|---|---|---|---|---|---|---|
| **推理与知识** | | | | | | | | |
| HLE | 40.5 | 31 | 41.4 | 37 | 37.7 | **49.8\*** | 41.4\* | 45 |
| HLE (w/ Tools) | 54.7 | 52.3 | 53.5 | — | 48.2 | **57.9\*** | 52.2\* | 51.4\* |
| CritPt | 20.9 | 4.6 | 13.4 | 3.7 | 12.9 | 20.9 | **27.1** | 17.7 |
| AIME 2026 | **99.2** | 95.3 | 97 | — | 94.6 | 95.7 | 98.3 | 98.2 |
| HMMT Nov. 2025 | 94.4 | 94 | 95 | 84.4 | 94.4 | **96.5** | **96.5** | 94.8 |
| HMMT Feb. 2026 | 92.5 | 82.6 | **97.1** | 84.4 | 95.2 | 96.7 | 96.7 | 87.3 |
| IMOAnswerBench | **91.0** | 83.8 | 90 | — | 89.8 | 83.5 | — | 81 |
| GPQA-Diamond | 91.2 | 86.2 | 90 | 93 | 90.1 | 93.6 | 93.6 | **94.3** |
| **编程** | | | | | | | | |
| SWE-bench Pro | 62.1 | 58.4 | 60.6 | 59 | 55.4 | **69.2** | 58.6 | 54.2 |
| NL2Repo | 48.9 | 42.7 | 47.2 | 42.1 | 35.5 | **69.7** | 50.7 | 33.4 |
| DeepSWE | 46.2 | 18 | 18 | 20 | 8 | 58 | **70** | 10 |
| ProgramBench | 63.7 | 50.9 | — | — | 47.8 | **71.9** | 70.8 | 39.5 |
| Terminal Bench 2.1 (Terminus-2) | 81.0 | 63.5 | 75 | 65 | 64 | **85** | 84 | 74 |
| Terminal Bench 2.1 (最佳 harness) | 82.7 | 69 | — | — | — | 78.9 | **83.4** | 70.7 |
| FrontierSWE (Dominance) | 74.4 | 30.5 | — | — | 29.0 | **75.1** | 72.6 | 39.6 |
| PostTrainBench | 34.3 | 20.1 | — | — | — | **37.2** | 28.4 | 21.6 |
| SWE-Marathon | 13.0 | 1.0 | — | — | — | **26.0** | 12.0 | 4.0 |
| **Agent** | | | | | | | | |
| MCP-Atlas (Public Set) | 76.8 | 71.8 | 76.4 | 74.2 | 73.6 | **77.8** | 75.3 | 69.2 |
| Tool-Decathlon | 48.2 | 40.7 | — | — | 52.8 | **59.9** | 55.6 | 48.8 |

\* 表示全集（而非纯文本子集）结果。第三方评测另有 Long-Horizon-Terminal-Bench 31.6、RedlineBench 45.7。

### 读这张表的几个要点

1. **5.1 → 5.2 的提升幅度非常大**，而且集中在 agentic 编程：DeepSWE **18 → 46.2**、FrontierSWE **30.5 → 74.4**、Terminal-Bench 2.1 **63.5 → 81.0**（+17.5）、SWE-Marathon **1.0 → 13.0**、ProgramBench 50.9 → 63.7。这不是常规小版本迭代的幅度。
2. **对 DeepSeek-V4-Pro 在 agentic 编程上是碾压**：DeepSWE 46.2 vs 8、FrontierSWE 74.4 vs 29.0、SWE-bench Pro 62.1 vs 55.4。但注意 V4-Pro 是 4 月的版本，且 V4 的强项在 Codeforces/LiveCodeBench 这类**算法竞赛**而非仓库级工程（见 [DeepSeek-V4.md](DeepSeek-V4.md) §九）。
3. **距离 Opus 4.8 最近的三项**：MCP-Atlas 差 1.0、FrontierSWE 差 0.7、Terminal-Bench 差 4.0。**差得最远的是 SWE-Marathon（13.0 vs 26.0，差一半）**——官方博客自己承认这项"有待提升"。
4. **数学是意外的强项**：AIME 2026 **99.2 全场第一**，IMOAnswerBench **91.0 全场第一**（比 Opus 4.8 的 83.5 高 7.5 分）。
5. **知识仍是短板**：HLE 40.5 vs Opus 4.8 的 49.8，CritPt 20.9 vs GPT-5.5 的 27.1。**和 V4、K3 承认的短板结构完全一致。**

### 评测配置（模型卡披露，透明度值得肯定）

- **HLE**：temperature 1.0, top_p 0.95, max 163,840 tokens，默认纯文本子集，GPT-5.5(medium) 当 judge；带工具版用 300,000 token 上下文且**不做上下文管理**
- **SWE-bench Pro**：OpenHands harness，temp 1，32k new tokens，**400K 上下文**
- **NL2Repo**：48k new tokens，400k 上下文；规则 + LLM 双重判定，**阻断未授权 pip/curl 之类的恶意行为**
- **DeepSWE**：官方 pier + mini-swe-agent，2h 超时，400K 上下文，隔离容器（2 CPU / 8GB / 无网）
- **ProgramBench**：200 实例，Claude-Code 2.1.156，max_turns 2000，6h 超时，reasoning_effort=max
- **Terminal-Bench 2.1**：Terminus-2 用 4h 超时 / 256K 上下文；Claude Code 版去掉墙钟限制、**5 次运行取平均**
- **FrontierSWE / PostTrainBench / SWE-Marathon**：第三方（Proximal / PostTrainBench / Abundant AI）跑，1M 上下文、max effort、128K 输出

---

## 十二、开源与生态

| | |
|---|---|
| **许可** | **MIT**，可自由下载、部署、商用，官方口径"技术平权无国界" |
| 权重 | Hugging Face `zai-org/GLM-5.2`（753B，BF16/F32）+ ModelScope |
| 发布节奏 | 6/16 上线，MIT 权重与博客在首发三天后放出 |
| 推理框架 | SGLang v0.5.13.post1+、vLLM v0.23.0+、Transformers、KTransformers、Unsloth；昇腾走 vLLM-Ascend / xLLM / SGLang |
| 生态 | 月下载 218 万，131 个社区量化版本，24 个微调，100 个 Space |
| 定价 | 官方页面未公布费率；AA 平台显示 **$0.9/1M**，含在 GLM Coding Plan 内 |

**MIT 许可是它相对 Kimi K3 的一个实际优势**——商用无附加条件，这在企业自部署场景里权重不小。

---

## 十三、可靠性说明

| 内容 | 来源 | 可信度 |
|---|---|---|
| 骨干配置表、Muon Split、MLA-256、MTP 参数共享、DSA 续训预算、注意力消融、mid-training 三段、超参、后训练五阶段、OPD 公式、异步 RL、TITO、Infra、国产芯片 | **GLM-5 报告正文提取** | 一手，可在 [papers/GLM-5-正文提取.txt](papers/GLM-5-正文提取.txt) grep 核对 |
| IndexCache 原理、70~100% 重合率、indexer 延迟占比、两种配置方法、30B 上的 1.82×/1.48×、GLM-5 预实验 50%/1.2× | **IndexCache 论文正文提取** | 一手，见 [papers/IndexCache-正文提取.txt](papers/IndexCache-正文提取.txt) |
| GLM-5.2 的 1M、IndexShare 四层一 indexer、2.9× FLOPs、MTP +20%、effort 档、评测表 | 官方博客 + HF 模型卡 | **厂商自述，无论文** |
| 78 层的口径、`full,shared,shared,shared` 模式细节、4.56→5.47 消融 | Raschka 架构笔记 | 二手解读 |

**最需要保留怀疑的三处**：

1. **2.9× 是 FLOPs 估算而非端到端加速**，且 KV cache 显存不等比例下降。
2. **GLM-5.2 的 1M 长上下文召回曲线没有公开**。官方称"Solid 1M 无损"，但没有给 MRCR/RULER 之类的逐长度曲线——而 GLM-5 报告在 200K 时代是给了 NIAH 数据的。这一项目前无法核实。
3. **GLM-5 报告不能自动证明 GLM-5.2 的能力**。5.1 → 5.2 在 DeepSWE 上从 18 涨到 46.2 这种幅度，训练侧一定有超出 IndexShare 的改动，但这部分**没有任何公开材料**。

---

## 参考来源

- [GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763) — 本地：[papers/GLM-5-...pdf](papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering-arXiv-2602.15763.pdf)
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse (arXiv:2603.12201)](https://arxiv.org/abs/2603.12201) — 本地：[papers/IndexCache-...pdf](papers/IndexCache-Cross-Layer-Index-Reuse-arXiv-2603.12201.pdf)
- [zai-org/GLM-5.2 (Hugging Face)](https://huggingface.co/zai-org/GLM-5.2) — 完整评测表与评测配置
- [GLM-5.2 上线并开源：专注 Coding 与长程任务（智谱官方）](https://www.zhipuai.cn/zh/research/161)
- [GLM-5 / 5.1 / 5.2 (GitHub)](https://github.com/zai-org/GLM-5)
- [Sebastian Raschka: GLM-5.2 IndexShare Architecture Note](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html)
- [Z.AI 开发者文档 — GLM-5.2](https://docs.z.ai/guides/llm/glm-5.2)
