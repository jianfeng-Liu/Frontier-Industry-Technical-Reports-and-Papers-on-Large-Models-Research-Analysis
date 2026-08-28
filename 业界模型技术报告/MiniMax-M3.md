# MiniMax M3（含 M2 复盘）架构与技术报告解析

> 📘 **不懂模型架构？先读这份**：[大模型架构入门/](大模型架构入门/README.md) —— **17 章 + 速查表**，从零讲清 注意力 / KV cache / MLA / MoE / RoPE / MTP / Muon / 量化，**不预设任何架构基础**。本文用到的术语都能在那里找到解释。

> **本文的特殊之处**：这是五家里**唯一一个把三条路线都跑过、并且把失败原因写进正式论文的团队**。
> 线性注意力（MiniMax-01 / M1）→ 退回全注意力（M2）→ 块稀疏注意力（M3）。中间那一步"退回"，别家都没有公开记录。
>
> | 内容 | 来源 | 可靠性 |
> |---|---|---|
> | MSA 的机制、训练配方、kernel、消融、效率数字 | **MSA 论文 arXiv:2606.13392**（30 页，正文提取核对） | 高（一手，但实验模型是 109B/6B，**不是 M3 本身**） |
> | M3 的实际架构超参 | **HF `config.json` 原文** | 高（一手，出货配置） |
> | M2 退回全注意力的完整论证 + SWA 消融表 | **M2 系列报告 arXiv:2605.26494 §2.2.2 + Table 2/3**（35 页，正文提取核对） | 高（一手） |
> | M3 的 9×/15×/1-20 加速比、评测分数 | 官方博客 + HF 模型卡 | 中（厂商自述） |
> | M3 的训练数据量、训练流程、后训练配方 | **缺失** | — |
>
> **本地论文**：[papers/MiniMax-Sparse-Attention-arXiv-2606.13392.pdf](papers/MiniMax-Sparse-Attention-arXiv-2606.13392.pdf)（30 页）· [papers/MiniMax-MSA-正文提取.txt](papers/MiniMax-MSA-正文提取.txt) · [papers/MiniMax-M2-Series-arXiv-2605.26494.pdf](papers/MiniMax-M2-Series-arXiv-2605.26494.pdf)（35 页）· [papers/MiniMax-M2-正文提取.txt](papers/MiniMax-M2-正文提取.txt)
> **横向对比**：见 [模型技术对比.md](模型技术对比.md)（DeepSeek-V4 / Kimi K3 / GLM-5.2 / MiniMax M3 / openPangu-2.0 五方）
> **同系列**：[DeepSeek-V4.md](DeepSeek-V4.md) ｜ [Kimi-K3.md](Kimi-K3.md) ｜ [GLM-5.2.md](GLM-5.2.md) ｜ [openPangu-2.0-Pro.md](openPangu-2.0-Pro.md)

---

## 一、一句话总览

**MiniMax M3 = 428B 总参 / ~23B 激活的原生多模态 MoE，1M 上下文，注意力换成自研的 MSA 块稀疏注意力。**

但真正值得读的不是 M3 本身，而是**这条三段式的路线图**：

```
MiniMax-01 / M1  （2025）  Lightning Attention 线性注意力 × 全注意力混合
        ↓  失败：大规模下多跳推理出现明确缺陷，且推理基建不成熟
M2       （2025-10）      全部换回 full attention + GQA
        ↓  这一步是"退"，但退得有理有据，并写进了论文
M3       （2026-06）      MSA：保留精确 softmax，但只在选中的块上算
```

**M3 的选择本质上是一个折中**：它没有回到线性注意力（丢精度），也没有停在全注意力（太贵），而是走了"**softmax 保持精确、但把 receptive field 限制成动态选出来的 2048 个 token**"这条路。这和 DeepSeek 的 DSA、Kimi 的 KDA 是三种不同的答案。

---

## 二、先看 M2：一次公开的"撤退"

> 本节全部来自 M2 系列报告 arXiv:2605.26494 §2.2.2，可在 [papers/MiniMax-M2-正文提取.txt](papers/MiniMax-M2-正文提取.txt) 里 grep 核对。

M2 的配置本身很朴素：**229.9B 总参 / 9.8B 激活，62 层，hidden 3072，48 个 query head / 8 个 KV head 的 GQA 全注意力，全模型 RoPE，256 个细粒度专家选 8，sigmoid 门控 + 可学习专家 bias，192K 上下文，29.2T tokens 预训练**（恒定阶段 19.9T + 衰减阶段 9.3T，上下文 8K → 32K → 192K 三段扩展）。

真正的内容在报告为什么这么选。原文把理由拆成四块：

### ① 评测本身才是瓶颈

> 开发 MiniMax-Text-01 时，我们的混合注意力模型在标准 benchmark（MMLU / BBH / MATH / LongBench）上**看起来追平了全注意力**，但**在更大规模上，复杂多跳推理出现了明确的缺陷**。

于是他们做了代理指标去补这个洞，但报告紧接着说了最要命的一句：

> **代理指标和真实下游表现之间的相关性是脆弱的**——它可能在更大规模上、或在没见过的任务分布上就不成立了。

再加上：统计显著的评测所需算力随任务复杂度快速增长；不同架构和数据分布、训练配方之间的交互不可预测。**"可靠的比较异常困难"。**

这一条的分量在于：它说的不是"线性注意力不行"，而是"**我们没有能力可靠地判断它行不行**"。

### ② 基建差距

原文列了三个推理侧的未解问题（外加一个训练侧的）：

| 问题 | 原文说法 |
|---|---|
| **低精度存储** | 线性注意力对数值精度的敏感度远高于全注意力，和低精度 KV cache 的工程实践冲突 |
| **前缀缓存** | 缺乏原生 prefix caching 支持 |
| **投机解码** | 与投机解码的集成方式不明确 |
| （训练侧） | 许多线性架构**即使在训练时也是访存受限的**——不做激进的 IO 优化，理论算力优势就被抵消掉了 |

### ③ Hybrid SWA 实验：全面失败，而且给了表

这是最有价值的部分——**别家只说"我们评估过并放弃了"，MiniMax 把消融表放出来了**。

实验规模：**在 M2 架构量级上，续训了"数千亿到数万亿 tokens"**，扫过的维度包括：

- SWA / 全注意力的比例
- RoPE 设置的调整
- **层内（intra-layer）与层间（inter-layer）两种混合方式**
- 注意力模式分析（induction heads、retrieval heads）
- SWA 里加 sink token

**预训练阶段结果（报告 Table 2，全注意力基线 vs 混合 SWA）**：

| Benchmark | Baseline | w/ SWA | Δ |
|---|---|---|---|
| MMLU | 85.5 | **85.6** | +0.1 |
| MATH | 60.3 | 60.3 | 0 |
| RULER 32K CWE | 99.0 | 99.0 | 0 |
| RULER 32K MQ | 99.0 | 99.0 | 0 |
| **HELMET ICL** | **75.8** | 72.7 | **−3.1** |
| **RULER 128K CWE** | **90.0** | 72.0 | **−18.0** |
| **RULER 128K MQ** | **99.0** | 93.0 | **−6.0** |
| **MTOB K-e Bleurt** | **60.0** | 45.0 | **−15.0** |
| **MTOB e-k ChrF** | **44.8** | 27.2 | **−17.6** |

**这张表把话说得非常清楚：32K 以内完全打平，128K 直接崩，上下文学习（MTOB 是低资源语言的书内翻译）崩得最狠。**

**SFT 之后差距进一步放大（报告 Table 3）**：

| | Baseline | w/ SWA | |
|---|---|---|---|
| AIME 2025 | 86.7 | 86.7 | 打平 |
| ARC-AGI-1 | 38.9 | **39.6** | SWA 略胜 |
| **GPQA-Diamond** | **75.3** | 72.7 | −2.6 |
| MMLU-Pro | **80.5** | 80.1 | −0.4 |
| **IFBench** | 23.1 | **27.2** | **SWA 大胜 +4.1** |
| **SWE-verified** | **54.7** | 50.2 | **−4.5** |
| **Terminal-Bench** | **26.7** | 23.8 | −2.9 |
| **BrowseComp-zh** | **32.8** | 28.7 | −4.1 |
| GAIA-103 | **53.4** | 51.5 | −1.9 |
| **XBench-ds** | 58.0 | **63.0** | **SWA 大胜 +5.0** |
| **τ²-Bench retail** | 62.3 | **67.5** | **SWA 大胜 +5.2** |
| **τ²-Bench telecom** | **32.5** | 21.0 | **−11.5** |

报告自己的归纳：

> 超过 32K 上下文的 benchmark（agent 任务和复杂长上下文评测）上，**SWA 变体显著差于全注意力**。32K 以内差异是混合的、绝对值也小——SWA 在某些指令遵循和短周期 agent 任务上甚至反超，而全注意力在知识密集型评测上保持优势。

**⚠️ 这张表里有一个容易被忽略的信息**：SWA 在 IFBench、XBench-ds、τ²-retail 上是**真的赢了**，而且赢得不小。也就是说这不是"SWA 全面更差"，而是"**SWA 换来的收益和损失落在不同的任务上，而它损失的那一类恰好是 MiniMax 要做的 agent 长任务**"。同一张表换一家公司、换一个产品定位，结论完全可能相反。

### ④ 但结论不是"永不"

报告的 Outlook 一节：

> 随着上下文长度增长、GPU 算力扩展放缓，**次二次方注意力会变得越来越重要**。我们正在投入更好的长上下文数据、评测方法和基建来支撑这个转变。

**M3 就是这句话的兑现，中间只隔了 8 个月。**

### ⑤ 一个必须澄清的事实

M2 系列报告**通篇只有 §2.2.2 这一节讲注意力**，全文其余部分（35 页里的 30 页）讲的是 agent 数据管线、Forge RL 系统、和 M2.7 的自演化。**它不是一份架构报告。** 网上把它当成"MiniMax 架构报告"来引用是不准确的。

---

## 三、M3 的实际架构（来自 HF `config.json` 一手）

```json
"hidden_size": 6144,          "num_hidden_layers": 60,
"num_attention_heads": 64,    "num_key_value_heads": 4,   "head_dim": 128,
"rotary_dim": 64,             "partial_rotary_factor": 0.5,  "rope_theta": 5000000,
"vocab_size": 200064,         "max_position_embeddings": 1048576,
"num_local_experts": 128,     "num_experts_per_tok": 4,   "n_shared_experts": 1,
"intermediate_size": 3072,    "dense_intermediate_size": 12288,
"scoring_func": "sigmoid",    "use_routing_bias": true,   "routed_scaling_factor": 2.0,
"hidden_act": "swigluoai",    "swiglu_alpha": 1.702,      "swiglu_limit": 7.0,
"use_qk_norm": true,          "qk_norm_type": "per_head", "use_gemma_norm": true,
"num_mtp_modules": 7
```

整理成表，和 M2 对照：

| | **M2** | **M3** |
|---|---|---|
| 总参 / 激活 | 229.9B / 9.8B | **428B / ~23B** |
| 层数 | 62 | **60**（前 3 层 dense + 57 层 MoE） |
| Hidden dim | 3,072 | **6,144**（翻倍） |
| 注意力 | **全注意力 GQA** | **MSA 块稀疏**（前 3 层仍是全注意力） |
| Query / KV heads | 48 / 8 | **64 / 4** |
| Head dim | — | 128 |
| 位置编码 | 全 RoPE | **Partial RoPE**（前 64 维，θ=5e6） |
| QK Norm | — | **per-head QK-Norm** |
| 上下文 | 192K | **1,048,576（1M）** |
| 路由专家 | 256 选 8 | **128 选 4** + 1 共享 |
| 门控 | sigmoid + 专家 bias | 同（+ routed_scaling 2.0） |
| 激活函数 | — | **`swigluoai`**（α=1.702, limit=7.0） |
| MTP | K=1 预训练 → 权重复制扩到 K=3 | **7 个 MTP module** |
| 词表 | 200,064 | 200,064 |
| 多模态 | 纯文本 | **原生多模态**（32 层 ViT，patch 14，3D RoPE，2×2 patch merge） |

**参数量自洽性验算**（我自己按 config 算的，非官方口径）：
`57 × 128 × 3 × 6144 × 3072 ≈ 412.9B`（路由专家）`+ 3.2B`（共享专家）`+ 6.7B`（60 层注意力，含 index 分支）`+ 0.7B`（3 层 dense FFN）`+ 2.5B`（embedding + 输出头）**≈ 426B**，加上 MTP 和 ViT 与官方 428B 吻合。
激活：`57 × (4+1) × 3 × 6144 × 3072 ≈ 16.1B + 6.7B + 0.7B ≈ 23.5B`，与 ~23B 吻合。

### 三个值得单独指出的设计

**① `swigluoai` + `swiglu_limit: 7.0` —— 又一家给 SwiGLU 加上限的**

`swigluoai` 是 gpt-oss 的截断版 SwiGLU：`α=1.702` 让 `Swish` 逼近 GELU，`limit=7.0` 把两个分支都夹住。

**四家现在全都在给 SwiGLU 加界**：DeepSeek-V4 硬截断到 `[−10,10]`、Kimi K3 用 `SiTU-GLU` 的 tanh 软上限（β₁=4, β₂=25）、MiniMax M3 用 gpt-oss 式的 clamp（limit=7）。**唯一没这么做的是 GLM-5.2。** 动机完全一致：低精度算术下的激活离群值。

**② 激活参数从 9.8B 涨到 ~23B —— "mini activations"这条原则被放宽了**

M2 系列报告的标题是《**Mini Activations** Unleashing Max Real-World Intelligence》，整篇论文的中心论点就是"极小激活量也能出前沿能力"。M3 把激活量提到 2.3 倍，专家数反而从 256 减到 128、top-k 从 8 减到 4——**这是一个方向上的调整，但官方没有给出任何解释**（没有 M3 技术报告）。合理猜测是省下来的注意力算力被换成了 FFN 容量，以及原生多模态需要更大的容量，但这**纯属推测，无一手依据**。

**③ KV head 从 8 减到 4**

这一条对解码速度是**独立于 MSA 的额外收益**：M3 每层每 token 的 KV cache 只有 M2 的一半。后面讲加速比口径时会用到这一点。

---

## 四、MSA 是什么：一张图讲完

> 本节全部来自 MSA 论文 §3，公式编号沿用原文。

### 两个分支

```
                 ┌─────────────── Index Branch（轻量，负责"选"）
输入 X ──┤
                 └─────────────── Main Branch（精确 softmax，负责"算"）
```

**Index Branch**：每个 GQA 组一个 index query head，**所有组共享同一个 index key head**：

```
Q_idx = stopgrad(X) · W_q^idx  ∈ R^(N × H_kv × d_idx)     ← M3: H_kv=4, d_idx=128
K_idx = stopgrad(X) · W_k^idx  ∈ R^(N × 1   × d_idx)     ← 只有 1 个头！
```

打分 → **按块取最大值**（block-max-pool）→ 每组各自取 Top-k：

```
S^idx,(r)_{i,j} = Q_idx^(r)_i · K_idx_j^T / √d_idx
M^idx,(r)_{i,b} = max_{j ∈ B_b, j ≤ i} S^idx,(r)_{i,j}      ← 块内取 max，不是平均
I^(r)_i = TopK_b( M^idx,(r)_{i,·} , k )                     ← 每个 GQA 组独立选
```

**Main Branch**：在选中的块上做**标准的、精确的** scaled dot-product attention。没有近似。

```
O^(h)_i = softmax( Q^(h)_i · K^(r)[I^(r)_i]^T / √d_h ) · V^(r)[I^(r)_i]
```

**M3 出货配置**：`B_k = 128`（块大小），`k = 16`（每组选 16 块）→ **每个 query、每个 GQA 组恒定看 `k·B_k = 2,048` 个 KV token**。上下文从 32K 涨到 1M，这个数字不变。

### 关键设计选择：为什么是"每组独立选"

这是 MSA 和 DSA 最本质的区别。论文附录 A 给了可视化证据：

> 不同的 GQA 组追踪**不同的长程条带**，同时共享相同的局部对角线和 sink 列——说明学到的 indexer **捕捉到了 group-specific 的稀疏模式，而不是坍缩成单一的全局选择模式**。

代价是每组要各存一份 top-k 索引；收益是不同组能看不同的地方。**DSA 走的是相反的选择：所有头共用一个 Top-k**（见 [GLM-5.2.md](GLM-5.2.md) §五）。

---

## 五、训练：Top-k 不可导，怎么教 indexer

Top-k 是离散的，LM loss 传不到 `W_q^idx / W_k^idx`。MSA 用**四件套**解决：

### ① KL 对齐损失：让索引分布去追主分支

```
P_idx^(r)_{i,j} = softmax over 选中的 token 集合 ( S^idx )
P^(r)_{i,j}     = (1/G) Σ_{ℓ ∈ H_r} softmax( S^(ℓ) )      ← G 个 query 头的注意力分布在概率层面取平均

L_KL = (1/(N·H_kv)) ΣᵢΣᵣ D_KL( stopgrad(P) ‖ P_idx )

总损失：L = L_LM + λ · Σ_layers L_KL
```

注意教师 `P` 是**主分支 G 个头的概率平均**——indexer 学的是"这一组整体想看哪"。

### ② Gradient Detach：这是最关键的一个补丁

```
Q_idx = stopgrad(X) · W_q^idx
K_idx = stopgrad(X) · W_k^idx
```

**为什么必须 detach**（论文 §B.3 给了失败现象）：

如果不 detach，KL 梯度会经过 index 的 Q/K 投影回流到 hidden state，再经残差进主干。于是**KL loss 变成了对主干的一个额外目标**。两个失效模式：

1. **KL 系数大一点就炸**：偶发的 KL 梯度尖峰传进主干，几百步内引起 grad-norm 尖峰和 LM loss 发散。
2. **系数稳定时也会慢性退化**：标准短上下文 benchmark 在训练中逐渐回退。论文的解释非常精彩——

> 我们把这个退化归因于一种**自蒸馏效应**：主干可以通过**把主分支的注意力分布简化掉**来降低 KL loss，而不是通过改进 Index Branch。

**换句话说：不 detach 的话，模型会作弊——它不去学"选得更准"，而是去学"让答案变得更容易选"。**

### ③ Indexer Warmup：先用全注意力热身

论文 §B.4 观察到：训练最初阶段主分支的**注意力熵急剧下降**（从平滑分布迅速变尖），之后才进入缓慢的表示学习期。如果第 0 步就开稀疏选择，indexer 要在自己还近乎随机的时候去追一个飞快移动的目标，错误的选择又会把主分支路由到无信息的 token 上，**双向恶化**。

做法：前 `T_warm` 步两个分支都跑全注意力，只用 KL 训 index 投影；之后再切到稀疏。

**M3 的两条训练路线都用了这个**：从零训（MSA-PT）是 40B token warmup；从全注意力 checkpoint 转换（MSA-CPT）也是 40B token warmup。

### ④ Local Block：只强制一个块

每个 query 位置所在的那个（不完整的）局部块**永远被选中**，占掉 16 个名额里的 1 个，防止退化成"连自己旁边都不看"。

**⚠️ 这里有个反直觉的消融结论（论文 §C.2）**：早期版本还额外强制了 **sink 块（序列第一块）** 和一个**固定的局部窗口**。后来发现**这些先验不需要硬编码**——去掉之后模型自己就长出了这两种结构，标准指标几乎不变。所以最终配方**只强制那个不完整的自身块**。

**这一条在 M3 的出货 config 里得到了直接印证**：

```json
"sparse_init_block": 0,     ← 不强制 sink 块
"sparse_local_block": 1     ← 只强制 1 个局部块
```

---

## 六、Kernel：这篇论文一半的分量在这里

MSA 论文有整整一章（§4）讲 GPU kernel，而且开源了（[github.com/MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA)）。**这是它区别于 NSA/MoBA 这类"论文里说得通"的方案的地方。**

### ① 免 exp 的 Top-k

softmax 是保序的（`s_i ≤ s_j ⟺ softmax(s)_i ≤ softmax(s)_j`），所以选 top-k 时**根本不用算 softmax**——直接对原始分数排序。前向直接跳过 max / exp / sum。

### ② 每线程寄存器最小堆

`B_k` 和 `k` 是**和 top-k kernel 协同设计的**：`B_k` 大 → 注意力算术强度高；在这个 `B_k` 下 `k` 小 → 候选块数 `B` 和 `k` 都落在通用 top-k kernel（radix 多轮分桶 / `O(B log²B)` 双调排序）的甜点**以下**。于是自己写：warp 的 32 个 lane 各扫 1/32 步长，各维护一个 k 元素最小堆（堆顶缓存在寄存器、延迟写入），最后 k 轮 shuffle merge。共享内存布局让每个 lane 固定映射到一个 bank，**零 bank conflict**。

**基准（H800, fp32，中位数）**：deployed 配置下相对 `torch.topk` 和 TileLang radix-select 有明显优势（论文 Table 1，128K/2048/32 配置下 5378 / 3630 / **1991 μs**，即 2.7× / 1.8×）。

### ③ KV-outer 迭代 + Q gather —— 全章最重要的一条

传统做法是 query 放外层循环。MSA 反过来，**把 KV 块放外层，然后去 gather 选中了这个块的所有 query**。论文给了完整的 FLOPs/IO 推导：

```
Q 外层：  FLOPs = 4 H_q N d_h k B_k
          IO    = 2·2·H_q N d_h + 2·2·H_kv N k B_k d_h
          ⟹ FLOPs/IO ≈ G                    （G = GQA 比 = 16）

KV 外层： FLOPs 相同
          IO    = 2·2·H_kv N d_h + 2·2·H_q N k d_h + 2·H_q N (k+1) d_h
          ⟹ FLOPs/IO ≈ (2/3)·B_k            （= 85.3）
```

**算术强度差 5 倍以上**（85.3 vs 16）。所以选 KV-outer。

### ④ 预调度的 tile 分块：消灭 sink 行的负载不均

一个 CTA 一个 tile 会被 **sink 行**打死——序列开头某个 KV 块几乎被所有 query 选中，那一个 CTA 要干所有活。

做法：调度 kernel 先把每个 KV tile 沿 query 维切成**不超过 ~2·k·B_k 个 query 的 chunk**，把热点 tile 铺开到多个共享同一份 K/V 加载的 CTA 上。因为一个 query 的 k 份部分结果现在由 k 个 CTA 产出，调度器**预先给每个 (query, chunk) 分配好 `O_buf` 里的槽位 `s ∈ [0,k)`**（和 query index 打包成 32 位句柄）——于是**注意力 kernel 写部分结果时完全不需要 atomic**。

其余：两阶段 forward + combine 用 Programmatic Dependent Launch 串起来；query 拼接成 128×128 的 score MMA；LSE 融合进主分支从而**省掉 KL 的一次前向**。

---

## 七、效率：三个不同的加速数字，别混

这是整篇里最容易被二手解读弄错的地方。**目前流传着三组数字，口径完全不同**：

| 来源 | 数字 | 比较对象 | 测量对象 |
|---|---|---|---|
| **MSA 论文 §5.4** | **28.4× FLOPs / 14.2× prefill / 7.6× decode** @1M, H800 | **同配置的 dense GQA**（同样 64q/4kv/d_h=128） | **注意力这一层**的理论 FLOPs 和实测延迟 |
| **M3 官方博客 / 模型卡** | **每 token 计算量 1/20 / prefill >9× / decode >15×** @1M | **M2**（不同模型：48q/8kv、9.8B 激活、全注意力） | 未明确说明 |
| 第三方解读 | 9.7× prefill / 15.6× decode | 同上 | 同上 |

**为什么 decode 一个是 7.6× 一个是 15×？** 我的分析（**非官方说法**）：论文的 7.6× 是"MSA vs GQA、其他全部相同"的干净对照；官方的 15× 是"M3 vs M2"，而 M3 相对 M2 还额外拿到了 **KV head 从 8 减到 4** 这一项——解码是访存受限的，KV cache 读取量减半本身就能贡献接近 2× 的独立收益。7.6 × 2 ≈ 15。两个数字不矛盾，只是**一个是控制变量的，一个是端到端的**。

**论文自己非常坦率地解释了为什么实测跟不上 FLOPs 比**：

> 稀疏注意力引入了索引构建、top-k 选择、**反向索引物化**、query gather 和负载均衡的开销，而且访存模式不如稠密注意力规则。因此运行时加速小于理论 FLOPs 缩减，**但会随上下文长度增长**——因为稠密基线继续随全序列扩展，而 MSA 的主注意力预算是固定的。

**这段话应该被更多论文抄。** 28.4× 是理论上限，14.2× 是能兑现的部分，差的那一半就是稀疏本身的开销。

---

## 八、MSA 到底掉不掉点：论文的实验（注意规模）

### ⚠️ 先说清楚适用范围

**MSA 论文的所有实验都在一个 109B 总参 / 6B 激活的模型上做的，不是 M3。** 配置：41 层（3 dense + 38 MoE）、`d_model=3072`、64q/4kv/head_dim 128/RoPE 维 64、128 专家选 4 + 1 共享、200K 词表、原生多模态。总预算 3T tokens。

**这个 pilot 和 M3 的关系**：head 配比、专家结构、词表、`B_k=128`、`k=16` 全部一致，只有 `d_model`（3072 vs 6144）和层数（41 vs 60）不同。所以它是 M3 的**同族缩小版**，但**不能替 M3 的能力背书**——这一点和 [GLM-5.2.md](GLM-5.2.md) 的情形完全同构。

### 两条转换路线

| | **MSA-PT**（从零稀疏训练） | **MSA-CPT**（从全注意力转换） |
|---|---|---|
| 起点 | 从零 | 一个训了 2.6T tokens 的 GQA 全注意力 checkpoint |
| 预算 | 3T 全程稀疏（40B warmup 后） | **只续训 400B tokens**（含 40B warmup） |
| 特点 | 数学、图像、视频、长上下文检索最强 | 更保守，尽量保留原 checkpoint 行为 |

**训练动态（Fig. 2/3）**：3T tokens 全程，MSA-PT 和全注意力的 **LM loss 曲线几乎重合**（最后 50B 窗口放大后仍然叠在一起，约 1.21），梯度范数也在同一区间。CPT 路线上，warmup 阶段 KL loss 快速下降，切到稀疏后保持低位；**block recall 和 score recall 都稳定在 0.7~0.95**，且 score recall 一直高于 block recall——**说明漏掉的块本来注意力质量就低**。

### 主结果（论文 Table 2，摘要）

| | Full | MSA-PT | MSA-CPT |
|---|---|---|---|
| MMLU | 67.0 | **67.2** | 66.8 |
| BBH | **67.7** | 66.6 | 66.1 |
| GSM8K | 76.2 | **77.7** | 73.7 |
| HumanEval | 61.0 | **64.0** | 57.9 |
| **RULER-8K** | 79.8 | **84.2** | 77.2 |
| RULER-32K | 75.0 | **77.5** | 75.7 |
| VisualWebBench | 55.6 | **68.4** | 59.4 |
| EgoSchema | 29.6 | **37.6** | 25.8 |
| VideoMME | 41.1 | **45.5** | 39.7 |
| MMMU | **46.8** | 45.9 | 44.5 |
| SWE PPL ↓ | **1.216** | 1.218 | **1.216** |

**MSA-PT 在相当多项上反超了全注意力**，尤其视频和长上下文检索。论文的解释是"原生稀疏预训练能让表示适应稀疏模式"。**但要注意这是 3T tokens 预算下的对照，不是无限训练下的结论。**

### 长上下文扩展（论文 Table 3）

MSA-CPT 再训 ~140B tokens 长上下文后：

| | Full | MSA-CPT | Δ |
|---|---|---|---|
| HELMET-128K Overall | 46.53 | 45.93 | **−0.60** |
| — ICL | 70.40 | **72.80** | +2.40 |
| — Rerank/RAG | **34.60** | 32.50 | −2.10 |
| RULER-128K Overall | 72.00 | **72.12** | **+0.12** |
| — MK/MQ/MV | 96.63 | **98.87** | +2.24 |
| — CWE/FWE | **46.35** | 45.00 | −1.35 |

**在每个 query / GQA 组只看 2,048 个 KV token 的前提下，128K 的长上下文能力基本持平。** 这是全文最有说服力的一个结果。

**把这个和 M2 的 SWA 表并排看，对照极其鲜明**：同样是"限制 receptive field"，SWA 在 RULER-128K CWE 上掉了 18 分，MSA 只掉 1.35 分。**差别就在"固定位置" vs "动态按内容选"。**

论文 §B.6 还专门做了这个对照：一个 **FLOPs 对齐的滑窗基线**（去掉 Index Branch，固定看第一块 + 同样 token 预算的局部窗口），结果在 agent 任务的困惑度上**全程劣于 MSA**。

### 其他消融（论文附录）

| 消融 | 结论 |
|---|---|
| **indexer 的梯度来源**（§B.2） | 只用 LM loss → 短上下文保住了，长上下文检索很差；只用 KL loss → 检索好了，短上下文掉；**LM + KL 最平衡**。但配上 warmup 之后，Index Branch 的 value 头就不需要了 |
| **块大小**（§C.1） | `B_k` 从 32 到 128，PPL 几乎不动，RULER 无明显退化 → **可以用大块换 kernel 效率** |
| **强制 sink / 局部窗口**（§C.2） | 去掉后模型自己长出来，指标几乎不变 → **最终配方不强制** |
| **Index Branch value 头**（§C.3） | 有无差别小且随 benchmark 而异 → **去掉**。推理时 top-k 只需要 `Q_idx K_idx^T` 的块内最大值，**完全避开 value 聚合和指数运算** |
| **可学习 attention sink**（§B.5） | 试了 gpt-oss 式的可学习 sink logit，**吸走了一部分注意力质量但没能消除原有的首 token sink**，在 agent PPL 上没有一致优势 → **不采用** |

---

## 九、MSA vs DSA vs NSA vs MoBA：逐条对照

这是本文最有实用价值的一节——**MSA 和 GLM-5.2 用的 DSA 是可以直接逐项对比的**。

| | **MSA**（MiniMax M3） | **DSA**（DeepSeek → GLM-5.2） | **NSA**（DeepSeek 早期） | **MoBA**（Moonshot） |
|---|---|---|---|---|
| 底座注意力 | **GQA** | **MLA**（MQA 模式） | MQA / MHA | GQA |
| 选择粒度 | **块级**（`B_k=128`） | **token 级** | 块级 | 块级（**块很大**） |
| 谁来选 | **每个 GQA 组独立 Top-k** | **所有头共用一个 Top-k** | 三条支路 | 每组 |
| 打分方式 | index Q/K 点积 → **块内取 max** | ReLU lightning indexer | 压缩表示 | **块内 key 取平均** |
| indexer 头数 | 每组 1 个 Q 头 + **全局共享 1 个 K 头** | 32 头 / dim 128（GLM-5.2 配置） | — | — |
| 支路数 | **1 条**（只有选择） | 1 条 | **3 条**（压缩 + 选择 + 滑窗） | 1 条 |
| indexer 怎么训 | **KL 对齐 + 梯度 detach** | — | LM 梯度 | **只有 LM 梯度** |
| 每 query 预算 | **k·B_k = 16×128 = 2,048** | **top-2,048** | — | — |

**几个值得单独说的点**：

1. **两家独立走到了同一个 2048。** MSA 是 `16 块 × 128`，DSA 是直接选 2048 个 token。**不同底座、不同粒度、不同团队，落到了同一个数字上**——这大概是当前长上下文的一个经验甜点。

2. **MSA 相对 NSA 是做减法。** NSA 有压缩、选择、滑窗三条支路；MSA **只保留选择**。论文自己把这叫"deliberately streamlined"，理由是"**便于在广泛的 GPU 上高效部署**"。少一条支路 = 少一套 kernel = 直接能挂进 vLLM / SGLang。**这是一个明确的工程优先于精巧的选择。**

3. **块内 max vs 块内平均。** MoBA 用块内 key 的平均值代表整块，MSA 用**块内 token 得分的最大值**。直觉上 max 更适合"这一块里有没有我要找的东西"这个问题——平均会被块内的无关 token 稀释。这一条论文没有做直接消融，属于设计取向的差异。

4. **每组独立选 vs 全局共选。** MSA 花额外的索引存储换组间的差异化（附录 A 的可视化证明它确实被用上了），DSA 用全局单一 Top-k 换更省的实现。**没有哪个一定更好，取决于底座是 GQA 还是 MLA**——MLA 在解码时本来就是 MQA 形态，全局共选是自然选择。

---

## 十、评测（M3 官方模型卡，厂商自述）

| Benchmark | **MiniMax M3** |
|---|---|
| SWE-bench Verified | **80.5** |
| SWE-bench Pro | 59 |
| Long-Horizon-Terminal-Bench | 38.5 |
| BrowseComp | 83.5 |
| MMMU_Pro | 78.1 |
| Video-MME-v2 | 85.4 |
| claw-eval | 74.5 |
| apex-agents | 27.7 |

发布形态：`MiniMax-M3` + `MiniMax-M3-MXFP8` 量化版；minimax-community 许可；`thinking` 三档（enabled / adaptive / disabled）；推荐 temperature 1.0 / top_p 0.95；SGLang / vLLM / Transformers / KTransformers / unsloth / ATOM 支持。API 最高 1M，保障至少 512K 可用。

> **⚠️ 交叉核对提示**：[GLM-5.2.md](GLM-5.2.md) §十一的对比表里 "MiniMax M3" 一列（HLE 37、GPQA-Diamond 93、SWE-bench Pro 59、Terminal Bench 65）来自 Z.ai 的模型卡。**SWE-bench Pro 的 59 两边对得上**，可以互证。

---

## 十一、这条路线告诉我们什么

把 M1 → M2 → M3 连起来看，能读出三个别处读不到的东西：

### ① "退回去"是可以公开的，而且比"没试过"更有信息量

M2 报告花了一整节讲一个失败的实验，还附了两张消融表。**这是四家报告里唯一一处系统性的负面结果披露。** DeepSeek、Kimi、Z.ai 都只说自己选了什么，不说自己放弃了什么（GLM-5 报告的 SWA 消融是个部分例外，见 [GLM-5.2.md](GLM-5.2.md) §五）。

### ② 三家对同一个问题给了三个不同答案，而且都能跑通

| | 长上下文的解法 | 代价 |
|---|---|---|
| **DeepSeek V4** | **压缩** KV（CSA 4:1 + HCA 128:1），仍是 softmax | 压缩必然有损，需要重叠压缩、反向 RoPE 一堆补丁 |
| **Kimi K3** | **换掉** 3/4 的注意力（KDA 线性 + 1/4 MLA） | 线性注意力的固定状态会遗忘；靠 NoPE + 1/4 全局层兜底 |
| **MiniMax M3** | **限制** receptive field（MSA 选 2048 个精确算） | 选错就漏了；靠 KL 对齐 + detach + warmup 保证选得准 |
| **GLM-5.2** | **限制** + 跨层复用索引（DSA + IndexShare） | 同上，且 indexer 本身成了新瓶颈 |

**MiniMax 和 Z.ai 落在同一格里**（都是"精确 softmax + 动态稀疏选择 + 2048 预算"），DeepSeek 和 Kimi 各占一格。

### ③ MSA 论文对"评测"的态度，正好补上了 M2 报告的那个洞

M2 报告说"**发现真正的问题往往比解决它难得多**"，而 MSA 论文的做法是——不再依赖某几个 benchmark，而是给**训练动态**（LM loss 曲线重合、梯度范数同区间）、给 **indexer 的 recall 曲线**（block recall / score recall）、给 **FLOPs 对齐的对照实验**（滑窗基线）。**这是对自己两年前那个教训的直接回应。**

---

## 十二、可靠性说明

| 内容 | 来源 | 可信度 |
|---|---|---|
| MSA 的公式、训练四件套、kernel 设计、6 组消融、Table 1–6 | **MSA 论文正文提取** | 一手，可在 [papers/MiniMax-MSA-正文提取.txt](papers/MiniMax-MSA-正文提取.txt) grep 核对 |
| M2 的架构、退回全注意力的四条理由、SWA 消融 Table 2/3 | **M2 系列报告正文提取** | 一手，见 [papers/MiniMax-M2-正文提取.txt](papers/MiniMax-M2-正文提取.txt) |
| M3 的 60 层 / 6144 / 64q4kv / 128 专家选 4 / `B_k=128,k=16` / `init_block=0,local_block=1` / MTP×7 | **HF `config.json`** | 一手，出货配置 |
| 28.4× / 14.2× / 7.6× | MSA 论文 §5.4 | 一手，但**是 109B pilot 上 MSA vs GQA，不是 M3 vs M2** |
| 1/20 计算量、9× prefill、15× decode、评测分数 | 官方博客 + HF 模型卡 | **厂商自述，无论文** |
| 参数量验算、"decode 15× ≈ 7.6× × KV head 减半"的解释、"激活量从 9.8B 到 23B 是原则放宽" | **我自己的推算/分析** | **推测，无一手依据** |

**最需要保留怀疑的四处**：

1. **MSA 论文的实验模型是 109B/6B，不是 428B/23B 的 M3。** 论文全文没有一句话描述 M3 自己的训练流程、数据配比或语料规模。**它证明的是"MSA 这个机制在 109B 上不掉点"，不是"M3 很强"。** 出货 config 里的超参与论文完全一致这一点，把这个 gap 缩小了很多，但没有消除。
2. **三组加速数字口径不同**，混用会得出错误结论。引用时必须带上比较对象。
3. **M3 没有技术报告。** 激活参数为什么从 9.8B 涨到 23B、专家数为什么从 256 减到 128、原生多模态怎么从第 0 步就联合训练、后训练怎么做的——**全部无公开材料**。
4. **M2 的 SWA 消融不能推广成"SWA 不行"。** 那张表里 SWA 在 IFBench / XBench-ds / τ²-retail 上是赢的。它证明的是"**在 MiniMax 关心的长程 agent 任务上，SWA 的覆盖限制是致命的**"，这是一个和产品定位绑定的结论。

---

## 参考来源

- [MiniMax Sparse Attention (arXiv:2606.13392)](https://arxiv.org/abs/2606.13392) — 本地：[papers/MiniMax-Sparse-Attention-arXiv-2606.13392.pdf](papers/MiniMax-Sparse-Attention-arXiv-2606.13392.pdf)
- [The MiniMax-M2 Series: Mini Activations Unleashing Max Real-World Intelligence (arXiv:2605.26494)](https://arxiv.org/abs/2605.26494) — 本地：[papers/MiniMax-M2-Series-arXiv-2605.26494.pdf](papers/MiniMax-M2-Series-arXiv-2605.26494.pdf)
- [MiniMaxAI/MiniMax-M3 (Hugging Face)](https://huggingface.co/MiniMaxAI/MiniMax-M3) — `config.json` 是本文架构表的唯一一手来源
- [MiniMax-AI/MSA (GitHub)](https://github.com/MiniMax-AI/MSA) — 开源推理 kernel
- [MiniMax M3 官方博客](https://www.minimaxi.com/blog/minimax-m3)
- [Why did M2 end up as a full attention model?（官方复盘）](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) · [Hugging Face 博客镜像](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model)
- [LMSYS: MiniMax-M2 部署分析](https://lmsys.org/blog/2025-11-04-miminmax-m2/)
