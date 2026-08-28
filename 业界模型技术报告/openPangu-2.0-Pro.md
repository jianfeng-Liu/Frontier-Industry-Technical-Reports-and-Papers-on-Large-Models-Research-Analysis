# openPangu-2.0-Pro 架构与技术报告解析

> 📘 **不懂模型架构？先读这份**：[大模型架构入门/](大模型架构入门/README.md) —— **17 章 + 速查表**，从零讲清 注意力 / KV cache / MLA / MoE / RoPE / MTP / Muon / 量化，**不预设任何架构基础**。本文用到的术语都能在那里找到解释。

> **来源**：本文整理自 2026-08-05 的调研，架构、公式、超参、评测数字均从技术报告 PDF 正文提取核对（非二手解读）。
>
> | 内容 | 来源 | 可靠性 |
> |---|---|---|
> | 架构、超参、预训练课程、后训练、Infra、全部评测数字 | **openPangu-2.0 技术报告**（45 页，正文提取核对） | 高（一手） |
> | 505B/18B、512K、34T tokens、许可 | 官方 HF 模型卡 + 华为官网公告 | 高（一手） |
> | **与其他任何模型的横向对比** | **报告中完全不存在** | — |
>
> ⚠️ **本文最重要的一条前提**：**这份 45 页报告里没有任何对照模型。** 全文 `Kimi` / `MiniMax` / `Claude` / `GPT-` / `Gemini` 出现 **0 次**，`Qwen` 1 次（参考文献），`GLM` 2 次（一次参考文献，一次说明自己的 OPD 与 GLM-5 的实现差异）。Table 5 唯一的对照维度是 **Flash vs Pro、Thinking vs Non-Thinking**。
>
> **所以本文 §十二 里所有"与四家的差距"都是我跨厂商拼出来的，采样协议不同、评测子集口径不同，方向可信，具体分差不可信到小数点。** 这一点和另外四家有本质区别——它们至少在自己的报告里放了对手。
>
> **本地论文**：[papers/openPangu-2.0-Tech-Report.pdf](papers/openPangu-2.0-Tech-Report.pdf)（45 页）· [papers/openPangu-2.0-正文提取.txt](papers/openPangu-2.0-正文提取.txt)
> **横向对比**：见 [模型技术对比.md](模型技术对比.md)（DeepSeek-V4 / Kimi K3 / GLM-5.2 / MiniMax M3 / openPangu-2.0 五方）
> **同系列**：[DeepSeek-V4.md](DeepSeek-V4.md) ｜ [Kimi-K3.md](Kimi-K3.md) ｜ [GLM-5.2.md](GLM-5.2.md) ｜ [MiniMax-M3.md](MiniMax-M3.md)

---

## 一、一句话总览

**openPangu-2.0-Pro = 505B 总参 / 18B 激活的 MoE，MLA + DSA/SWA 1:2 混合注意力，mHC 4 支流残差，512K 上下文，34T tokens——全程在昇腾 NPU 上训练完成。**

报告标题是 **《openPangu-2.0: Towards Reliable and Efficient Agentic Reasoning》**，摘要里反复出现的词是 **hardware-software co-design**。这决定了它和另外四家的根本区别：

> **另外四家的报告在回答"怎么把模型做得更强/更便宜"；这份报告在回答"怎么在昇腾上把一个 500B 级 MoE 从头到尾跑通"。**

三个变体（报告 Table 8）：

| | MoE15B-A2B | openPangu-2.0-Flash | **openPangu-2.0-Pro** |
|---|---|---|---|
| 用途 | **消融与快速验证**（本报告所有消融都在它上面做） | 开源（2026-06-30） | 开源（2026-07-31） |
| 总参 / 激活 | 15B / 2B | 92B / 6B | **505B / 18B** |
| 上下文 | — | 512K | **512K** |

另有一个 **30B / 2B 激活的端侧模型**（Kirin 感知量化与剪枝，专家复用 loss 把专家切换频率降 50%，专家激活预测做预取，吞吐最高提升 5×）。

**发布时间线**：2026-06-12 HDC 发布 → 06-30 Flash 开源 → **07-31 Pro 权重 + 推理代码 + 技术报告开源**。许可是 `OPENPANGU MODEL LICENSE AGREEMENT VERSION 2.0`，**不是 MIT/Apache**。

---

## 二、骨干架构（报告 Table 8，一手）

| 配置项 | MoE15B-A2B | Flash | **Pro** |
|---|---|---|---|
| 总参数 | 15B | 92B | **505B** |
| 激活参数 | 2B | 6B | **18B** |
| 总层数 | 32 | 46 | **50** |
| Dense 层 | 2 | 2 | **3** |
| MoE 层 | 30 | 44 | **47** |
| Hidden Size | 1536 | 2560 | **5120** |
| 词表 | 151552 | 151552 | **151552** |
| 注意力头数 | 48 | 48 | **64** |
| Query 投影秩 | 384 | 1024 | **1536** |
| KV 投影秩 | 512 | 512 | **512** |
| 无位置 QK 维 | 128 | 128 | **128** |
| RoPE 维 | 64 | 64 | **64** |
| V head 维 | 128 | 128 | **128** |
| Dense FFN 中间维 | 7680 | 9216 | **16128** |
| 路由专家 | 128 | 256 | **384** |
| 共享专家 | 1 | 1 | **1** |
| 激活路由专家 | 8 | 8 | **8** |
| 专家中间维 | 768 | 1024 | **1792** |
| MTP 层 | 1 | 3 | **3** |
| **DSA : SWA** | 1:2 | 1:2 | **1:2** |
| SWA 窗口 | 512 | 512 | **512** |

### 参数量验算（配置表内部自洽）

```
路由专家   47 × 384 × 3 × 1792 × 5120 ≈ 496.8B
共享专家   47 ×   1 × 3 × 1792 × 5120 ≈   1.29B
Dense FFN   3 ×       3 × 16128 × 5120 ≈   0.74B
注意力(MLA) 50 × ≈80M                  ≈   4.0B
词嵌入+输出 2 × 151552 × 5120          ≈   1.55B
                                        ─────────
                                          ≈ 504.4B  ✓ 对得上 505B

激活：47 × 9 × 3 × 1792 × 5120 ≈ 11.6B  + 0.74 + 4.0 + 1.55 ≈ 17.9B  ✓ 对得上 18B
```

> **口径提示**：HF 侧栏标注 **541B**，是**含 3 个 MTP 层**的全量口径（每个 MTP 层 ≈ 一个完整 transformer 层 ≈ 10.7B，×3 ≈ 32B）。**505B 不含 MTP。** 这和 GLM-5 的 744B / 753B 是同一类口径差异，见 [GLM-5.2.md](GLM-5.2.md) §一。

**稀疏度**：384 选 8 = **48×**，介于 V4 的 64× 和 GLM/M3 的 32× 之间。**18B 激活是五家里最低的**（M3 23B / GLM 40B / V4 49B / K3 104B）。

---

## 三、注意力：DSA + SWA 的 1:2 交错

底座仍是 **MLA**，在它之上做了四件事。

### 3.1 层间交错：DSA–SWA–SWA

```
重复模式：DSA → SWA → SWA        （即 SWA:DSA = 2:1）
SWA 窗口固定 512
SWA 层的 RoPE base 随上下文放大：1e4 @4K → 6.4e6 @512K
```

**报告给了一条很有价值的负面经验**：

> 更大的比例（3:1、5:1）在 **4K 短序列上性能不掉**，但**扩到 32K 和 128K 时训练 loss 会退化**。

这和 GLM-9B 的 SWA 消融、MiniMax M2 §2.2.2 是**第三份同向证据**——滑窗比例越高，短上下文越看不出问题，长上下文越崩。**而且这一份补上了前两份缺的东西：它给出了"多少比例是安全的"这个量化答案（1:2，即三层里必须有一层是全局层）。** 详见 [模型技术对比.md](模型技术对比.md) §3.1.1。

### 3.2 DSA 的引入方式：只转全局层

DSA 沿用 DeepSeek 的 lightning indexer，选固定 **K = 2048** 个历史 KV。但引入方式和 DeepSeek / GLM 都不同：

> **openPangu 的基座本身就是 FULL-SWA 混合，所以它只把全注意力层转成 DSA，SWA 层原样保留。**

结果是一个**结构上解耦**的设计：**DSA 负责全局依赖，SWA 负责局部交互**。

**转换预算——第三个数据点**：

| | 稀疏适配预算 |
|---|---|
| DeepSeek-V3.2 | **943.7B tokens** |
| **openPangu-2.0** | **8B dense 预热 + 100B 稀疏训练** |
| GLM-5 | **20B tokens** |

三家的差距达到 **47×**，而三家都声称"基本恢复下游表现"。**这个量级的分歧本身就说明：DSA 适配预算目前没有共识，很可能高度依赖基座与数据。**

### 3.3 ⭐ 一条与 MiniMax MSA 的独立互证

DSA 转换时，报告给 lightning indexer 加了一个辅助目标——**拟合全注意力算出的原生注意力分数**，然后写了这么一句：

> **系数设为 0.1（因为大系数会产生过大的梯度，损害训练稳定性）。**

**这正是 MiniMax MSA 论文 §B.3 记录的失效模式。** MSA 发现：indexer 的对齐 loss 权重 λ 一大，骨干就会通过**简化主分支的注意力分布**来降低这个 loss，而不是改进 indexer，最终 LM loss 发散。MSA 的解法是**梯度 detach**（`stopgrad(X)`），从而可以放心用大 λ。

**华为独立撞上了同一堵墙，但选了另一条路——不 detach，改成把 λ 压到 0.1。** 两份报告互不引用（openPangu 全文 `MiniMax` 出现 0 次）。

> **这是本目录里第一对"同一个失效模式、两个团队、两种解法"的样本，而且两边都把它写下来了。** 强度上讲：MSA 给了机制解释和消融曲线，openPangu 只给了一句经验；但 openPangu 是在 **505B 真实出货模型**上的经验，MSA 的消融在 109B 试点上。**两者互补。** 详见 [MiniMax-M3.md](MiniMax-M3.md) §五。

### 3.4 PAS：参数化注意力汇聚点

每层引入 **128 个可学习的 key-value 对**（MLA 里是 latent KV），专门吸收多余的注意力质量。推理时它们是**固定大小的虚拟 KV 条目**，长上下文下开销可忽略。

（消融见 §四，这是本报告最有独立价值的一处实验。）

### 3.5 ModAttn：局部上下文调制

用**因果一维卷积 + 残差**在全局注意力**前后**各注入一次局部上下文：

```
Mod_θ(Z) = Z + CausalConv1D_θ(Z)

作用位置：① MLA 的压缩 latent 分支（Q 和 KV 都做）
         ② FlashAttention 全局聚合之后、输出投影之前
```

报告称其为 **low-intrusion enhancement**——不改变注意力接口，只在前后挂两个卷积。

> 值得对照：**Kimi K3 的 KDA 也在 q/k/v 上挂 ShortConv**（见 [Kimi-K3.md](Kimi-K3.md) §二）。**两家在完全不同的注意力机制上，都选择用短因果卷积补局部建模。**

### 3.6 效率分析（报告 Table 1）

报告没有报端到端加速比，而是给了**长上下文服务三项主导成本的领头系数**：

| 成本 | 同构 DSA（如 V3.2） | FULL-SWA（如 Gemma 2） | **DSA-SWA（本文）** |
|---|---|---|---|
| Prefill FLOPs（S²） | `N^Idx_head · D_Idx` | `⅓ N_head (D^MHA_QK + D^MHA_V)` | **`⅓ N^Idx_head · D_Idx`** |
| Decode cache 读取（S） | `D_Idx` | `⅓ D_KV` | **`⅓ D_Idx`** |
| Decode cache 显存（S） | `D_KV + D_Idx` | `⅓ D_KV` | **`⅓ (D_KV + D_Idx)`** |

**论证逻辑很干净**：相比同构 DSA，把 DSA 限制在 1/3 的层上，全局选择成本按比例砍到 1/3；相比 FULL-SWA，把稠密全局注意力换成"轻量索引 + 稀疏检索"，把二次项从全注意力表示降到小得多的索引表示。**两个方向各占一半便宜。**

⚠️ 但注意：**这是渐近系数分析，不是实测加速比。** 报告的实测数字只有 TPOT / TPS（见 §十一），没有和任何基线对比。

---

## 四、⭐ PAS vs gpt-oss sink：本报告最有独立价值的一处消融

这是本目录五家里**唯一一份把 attention sink 的不同实现放在一起比过的材料**。

**实验设置**：MoE15B-A2B 基线，训练 250B tokens，FULL : SWA = 1:2 混合架构。

**三条结论**：

1. **PAS 让梯度范数明显更平滑，spike 显著减少**（Fig. 9）。
2. **对全注意力基线，加 PAS 没有明显的 loss 收益。**
3. **对混合 SWA 模型，PAS 有一致的 training-loss 增益，并且明显优于 gpt-oss 使用的 sink**（Fig. 10）。

**报告给的机制解释很到位**：

> SWA 层是局部的。**没有显式 sink 时，即使窗口内没有任何值得注意的 token，注意力质量也必须分配到窗口内的真实 token 上**——这会把某些 token 推去充当"局部 sink"的角色，扰乱注意力分布。参数化 sink 给 SWA 层提供了一个**固定的吸收通道**：对 FULL/DSA 层它是稳定的全局 sink，对 SWA 层它是"局部窗口没有信息时的兜底注意力目标"。

**为什么这条值得单独记**：

- 它解释了一个此前没人解释过的现象——**sink 机制的收益是架构条件性的**。全注意力基线上加 sink 收益不明显，**只有混合了滑窗层才显现**。这意味着凡是引用"attention sink 有用"的工作，都需要说明自己的注意力架构。
- 它是**唯一一份 gpt-oss sink 的第三方对照实验**。DeepSeek-V4 用的是"每个头一个可学习 sink logit"（加在 softmax 分母上），MiniMax M3 用的是 `swigluoai` 那一系的 gpt-oss 血统组件，但**都没有做过 sink 变体之间的对比**。
- 它给了 §3.1 那条"SWA 比例不能太高"的经验一个**互补的补丁思路**：SWA 崩掉的一部分原因可能不是"看得不够远"，而是"没地方放弃看"。

⚠️ **打折提示**：消融在 **15B/2B 模型、250B tokens** 上做，规模离 505B 很远；报告只给了曲线（Fig. 9/10），**没有给数值表**，无法量化"明显优于"是多少。

---

## 五、mHC：4 支流，以及一个原论文没记录的失效模式

openPangu 直接采用 DeepSeek 的 **mHC**（Manifold-Constrained Hyper-Connections，arXiv:2512.24880），配置 **4 条并行残差流 + 20 次 Sinkhorn-Knopp 迭代**——**和 DeepSeek-V4 完全一致的迭代次数**（见 [DeepSeek-V4.md](DeepSeek-V4.md) §三）。

```
x̄ = RMSNorm(h̄)
[r_pre, r_post, r_res] = x̄ · W_mHC
H_pre  = σ(r_pre)                              ← 从哪几条流读入子层
H_post = 2σ(r_post)                            ← 子层输出写回哪几条流
H_res  = Sinkhorn-Knopp(reshape(r_res, n, n))  ← 流间混合，约束为近似双随机
```

### ⭐ 新发现：Gate Collapse

**这是 mHC 原论文和 V4 报告里都没有的内容**：

> 在原始 mHC 的放置方式下（`H_pre` 在 pre-norm 之前、`H_post` 在子层输出之后），我们观察到 **gate-collapse 现象：训练过程中，浅层的所有 `H_pre` 和 `H_post` 门控都趋近于零**，导致这些层**与残差流脱耦**，训练信号退化。

**归因**：子层输出在写回之前尺度不稳定。

**两级尺度控制解法**：

1. **Sandwich normalization**：在 `H_post` 写回**之前**对子层输出做归一化，让 `H_post` 不必再去压制过大或不稳的输出尺度。
2. **Block-level post-normalization**：在**第一层之后、之后每五层**对残差流做一次块级后归一化，显式约束多流主干沿深度累积的尺度。

**消融（报告 Table 2，MoE15B-A2B，1T tokens）**：

| 模型 | EN | ZH | MMLU | CEval | BigBench | MATH | MBPP+ | **平均** |
|---|---|---|---|---|---|---|---|---|
| Baseline（无 mHC） | 0.500 | 0.605 | 0.765 | 0.708 | 0.591 | 0.459 | 0.532 | 0.594 |
| **mHC（稳定化后）** | **0.511** | **0.608** | **0.799** | **0.743** | **0.622** | **0.472** | **0.553** | **0.615** |

**七项全赢，平均 +2.1 个点**，MMLU +3.4、CEval +3.5 最明显。

> **这是本目录里 mHC 的第二份独立验证。** V4 报告说 mHC 让墙钟时间多 6.7% 但解决了 1.6T 规模的训练稳定性；openPangu 在 15B 规模上给出了**下游指标的直接收益**，并额外贡献了一个 V4 没写的失效模式和它的解法。**对任何想用 mHC 的人，这一节比 V4 报告更实用。**

---

## 六、MoE：把 aux-loss-free 的 sign 换成 RMS 归一化

384 专家选 8 的高稀疏度对负载均衡提出了更严格的要求。openPangu 用了**两套互补机制**：

### 6.1 Adaptive Auxiliary-Loss-Free Bias

原版 aux-loss-free 用 **sign 函数**更新偏置——**不管超载多少，修正幅度都一样**：

```
原版：  b_i ← b_i − γ · sign(F_i − Q_i)

openPangu： b_i ← b_i − γ · (F_i − Q_i) / RMS(F − Q)
            其中 RMS(F − Q) = sqrt( (1/N_r) Σ_j (F_j − Q_j)² )
```

**保留了专家负载误差的相对幅度**：超载多的专家得到更大修正，接近均衡的得到更小修正，而整体更新尺度仍由学习率 γ 控制。

**消融（附录 A.3）**：两种方法**收敛速度相同、LM loss 几乎一致**，但 RMS 归一化版**始终取得更低的 batch 级负载违背度（MaxVio），且优势随训练推进而扩大**。

> ⚠️ 报告在这里引用了 **苏剑林的博客《MoE 环游记：3、换个思路来分配》**（`spaces.ac.cn/archives/10757`）作为正式参考文献 [66]。**这是本目录五份报告里唯一一次把中文技术博客列进参考文献。**

> 📌 **和 Kimi K3 的 Quantile Balancing 是同一个问题的两种解**：K3 认为 `γ` 这个固定步长是病根，于是**直接从 margin 的分位数解析解出 bias，把 γ 消掉**；华为认为病根是 sign 丢掉了误差幅度，于是**保留 γ 但让步长自适应**。K3 的解更彻底（少一个超参），华为的解改动更小（一行公式）。**两家都没引用对方。** 见 [Kimi-K3.md](Kimi-K3.md) §四。

### 6.2 EP-group 辅助损失

自适应 bias 管**全局 batch** 的均衡，另加一个 **EP-group 辅助损失**管**每个专家并行组内部**的局部均衡。权重 `2.0e-4`，另有 router z-loss `1.0e-6`。

**这些均衡约束只在 general 阶段开启，之后全部关闭。**

---

## 七、MTP：3 层，且内部走滑窗

```
L = L_main + Σ_{k=1..D} λ_k · L_MTP^(k)
出货配置 D = 3
```

**一个别家没有的细节**：**MTP 模块内部的所有注意力都用固定 2048 窗口的滑窗注意力**。理由是让辅助模块在长上下文下保持实用——窗口限住了计算与 cache 增长，同时保留足够的局部上下文来产出高质量草稿。

**分阶段调度（报告 Table 3）**：

| 阶段 | 序列长度 | MTP 层数 D | Loss 权重 λ |
|---|---|---|---|
| General | 4K | 1 | 0.3 |
| Reasoning | 8K | 1 | **0.1**（避免过度正则化主推理目标） |
| Annealing | 32K / 128K / 512K | **3** | 0.1 each（合计 0.3） |

> **四家的 MTP 现在有四种做法**：V4 深度 1、GLM 3 层**参数共享**、K3 微调成 EAGLE-3 草稿模型、M3 **7 个模块**、openPangu **3 层独立 + 内部滑窗 + 分阶段扩展**。⚠️ 报告**没有给接受长度（accept length）**，无法和 GLM 的 2.76 / DeepSeek-V3.2 的 2.55 对比。

---

## 八、Muon：分 4 组，不是逐 head

Muon 用于所有二维及以上权重张量；AdamW 负责词嵌入、预测头、**parametric sink 参数**、以及所有一维参数（RMSNorm 权重、mHC 门控系数）。

**关键差异**：

> 对 MLA 的 query 和 key-value 上投影，**注意力头被分成 4 组**后再做 Muon。

理由：**既对齐 head 的结构独立性，又保证更新矩阵足够大，正交化才稳定。**

> 📌 **这是"按 head 切分正交化"这条线上的第三个刻度**：
> ```
> GLM-5 Muon Split（2026-02） ：逐 head 切开
> Kimi K3 Per-Head Muon（07） ：逐 head 切开
> openPangu-2.0（07-31）      ：切成 4 组   ← 明确说明理由是"矩阵别切太小"
> ```
> **前两家从没讨论过"切太细会不会让 Newton-Schulz 不稳"这个问题。** openPangu 提出了它，但**没有给消融**——是经验判断。

超参：`μ = 0.95`，Nesterov，更新缩放 `γ = 0.2` 以匹配 AdamW 的 RMS 幅度，**从而两个优化器共用一套学习率调度**。Newton-Schulz 系数直接采用 `modded-nanogpt` 仓库的实现（报告在脚注给了 commit 级链接）。

### ⭐ Muon 下 attention logits 稳定性：第三种归因

已有工作报告 Muon 会导致 attention logits 爆炸（这正是 Kimi 提出 QK-Clip 的原因）。openPangu 观察到：

> 整个训练过程中 attention logits 保持稳定，**最大值始终在约 20 的有界范围内**（Fig. 12，覆盖 Pro 的完整训练全程），无需任何 clipping。

**归因**：**sandwich normalization + block post-normalization**，它们显式约束了隐状态与输出的尺度。

报告很诚实地加了一句：**"We leave a more systematic ablation of this phenomenon to future work."**

> **到这里，"Muon 下 logits 为什么稳"已经有三种互斥归因了**：
> ```
> GLM-5      → Muon Split（逐 head 正交化，从优化器侧解决）
> DeepSeek-V4 → QK-RMSNorm（直接归一化 q 和压缩 KV 条目）
> openPangu   → sandwich norm + block post-norm（约束子层输出与残差流尺度）
> Kimi K2/K3  → QK-Clip（直接裁剪，是"承认问题存在"的一方）
> ```
> **四家给出四个不同的解释，而且没有任何一家做过交叉消融。** 这是本目录里最明确的一个**悬而未决的开放问题**。详见 [模型技术对比.md](模型技术对比.md) §3.8。

---

## 九、预训练：三阶段课程，34T tokens

```
① General   : 4K 序列, 25T tokens
              Pro: batch warmup → 64M, peak lr 2.0e-4（Flash 是 36M / 3.0e-4）
              前 500B warmup → 峰值保持 17T → 最后 8T cosine 降到 1.0e-4
② Reasoning : 8K 序列,  8T tokens，提高 STEM 与代码高质量数据比例
              lr cosine 1.0e-4 → 3.0e-5
③ Annealing : 32K(800B) → 128K(400B) → 512K(200B)
              lr cosine 3.0e-5 → 8.0e-6
④ DSA 转换  : 8B replay tokens dense 预热 @ lr 1.0e-3
              100B replay tokens 稀疏训练 @ lr 8.0e-6
```

`25T + 8T + 1.4T ≈ 34.4T` ✓ 与官方"约 34T"一致。

**RoPE base 随长度逐级放大**：`10K@4K → 100K@8K → 400K@32K → 1.6M@128K → 6.4M@512K`。

**Tokenizer 的一个反直觉决定**：报告对比了三种词表，**3-digit 变体压缩率最好**（中文 4.04、多语 3.37、行业 4.51，平均 2.71），但**最终选了压缩率略差的 single-digit 变体**（平均 2.59）。理由：

> 避免多位数字按频率合并，为**数值推理、多模态定位、工具调用**提供更规则的数字级表示。

**用压缩率换数字表示的规整性**——这在 agentic 场景（坐标、bounding box、索引、工具参数）里是合理的取舍。

---

## 十、后训练：多专家 RL + OPD，以及一份训推不一致的完整解剖

```
SFT（快慢合一）
  ↓
四路专家 RL：Reasoning / General / Agentic / Code
  ↓
Multi-Expert On-Policy Distillation（OPD）合成单模型
```

### 10.1 OPD：明确对标 GLM-5

```
A_{i,t} = sg[ log ( π^train_{θ_expert}(y_{i,t} | x_i, y_{i,<t}) / π^train_θ(y_{i,t} | x_i, y_{i,<t}) ) ]
```

**和 GLM-5 的 On-Policy Cross-Stage Distillation、Kimi K3 的 MOPD 是同一个式子。** 但报告写明了自己的差异：

> **这与 GLM-5 不同——GLM-5 报告称从推理后端读取教师概率，并提到未来会从 `π^infer` 迁移到 `π^train`。**

**openPangu 直接用训练引擎前向计算教师概率，等于把 GLM-5 说的"未来工作"做了。** 这是本目录里少见的**明确的、有据可查的接力**。

另外两个加法：

- **轻量正则**：mask 掉 advantage 近零的 token；clip 掉离群的 log-ratio 值。
- **计算优化**：不给每个专家分配独立 NPU 池，而是**把专家按顺序轮流装入一个共享 NPU 池**——rollout 后按专家分组样本，逐组加载权重算 log 概率。**用时间换显存。**

### 10.2 Agentic RL

五个手段，其中一个值得注意：

- **TITO（Token-in-Token-out）**：消除轨迹生成与策略更新之间的 tokenization 错配。**这个术语和做法与 GLM-5 完全一致**（见 [GLM-5.2.md](GLM-5.2.md) §九）。
- **在线动态过滤**：持续跟踪每样本通过率，**始终被解出的样本动态踢出训练循环**，让模型永远待在学习前沿。
- **语法惩罚**：畸形工具调用（缺参数、非法 JSON）触发显式格式错误惩罚。
- **可验证评估**：奖励确定性，基于训练循环之外的客观执行结果。
- **暖启动**：Agentic RL 从 General RL 的 checkpoint 初始化。

### 10.3 ⭐ 训推不一致：一份三分类的完整解剖

**这是本报告最有分量、也是五家里最系统的一章。** 其他四家并非没做——**V4 那一份的分量其实相当重**：它专门写了一节 *High-Performance Batch-Invariant and Deterministic Kernel Libraries*，✅ 报告原文的目标是"确保训练与推理之间的**逐位可复现（bitwise reproducibility）**"，属于拿 kernel 工程正面解决问题；GLM 有 bitwise 量化 kernel，K3 有全程 QAT。**但只有 openPangu 把"为什么会不一致"拆成分类学讲了一遍**——它交付的是**诊断框架**，V4 交付的是**工程结果**，两者不是一回事。

**现有手段**（报告承认"没有单一技术能覆盖所有场景，要按架构/硬件/失配严重程度组合"）：

- **FP16 替代 BF16** ⚠️ ——配动态 loss scaling 处理梯度下溢。**这是一个逆潮流的选择**，五家里唯一一个。
- **KPop** 替代 IcePop 和常规拒绝采样，作为主要校正机制。配了一个 **shadow 超参机制**：训练初期并行评估多个候选 `φ`，先估计它们的 mask 行为再定值。经验规则：**train-rollout PCC < 0.99 用更严的 mask；> 0.99 用更松的 `φ` 以保留探索能力**。
- **Rollout Routing Replay (R3)** 的四点增强：
  - **流式传输路由专家**：通过 OpenAI 兼容流式 API 增量推送路由记录，把带宽需求摊到整个生成时间线上。
  - **可扩展数据传输**：只把元数据放进 DataProto，路由记录走 HCCL，避开 Ray 瓶颈和序列长度 padding。
  - **Masked R3** ⚠️：观察到**隐状态随 MoE 层加深、序列逐 token 推进而累积发散**，此时强制全量专家重放会**严重扭曲输出 logits、PCC 骤降，可能触发梯度爆炸或训练发散**。解法是自适应阈值——**只有当训练侧提出的专家选择与推理侧记录的重合度超过阈值时，才对该 token 重放**。
  - **Cached R3**：agentic RL 多轮推理时，已解码 token 在后续轮次会重新 prefill，而 prefill 可能给出**与当初生成时不同的路由**。解法是持久化每个 token 首次生成时的路由选择，训练时重放缓存值。

**根因三分类**：

| 类别 | 定义 | 具体成因 |
|---|---|---|
| **A. 显式定义失配** | 数学上就不等价 | 训推双引擎架构本身的公式差异 |
| **B. 计算图重排失配** | 数学等价但数值发散 | ① **部署策略**：MLA 在 Absorptive(MQA, decode) 与 Non-Absorptive(MHA, 训练/prefill) 间切换；MoE 专家训练侧按 Expert-ID 升序累加、推理侧按 Top-k 分数降序累加<br>② **并行策略**：训推偏好不同并行方式，改变浮点累加精度与顺序<br>③ **融合算子**：kernel 融合隐式重写计算图与数值精度（不同 FlashAttention 家族、MoE dispatch/combine kernel、采样算子） |
| **C. 硬件调度非确定性** | 同一次运行到下一次都会变 | NPU 按输入 shape 自动选 tiling：**SplitK**（K 轴累加拆到多个 AI Core）、**K 轴 swizzling**（交错访存提升 HBM 带宽）、**AtomicAdd**（多核异步收集，核到达顺序随机；AllReduce 的 Ring/Tree 拓扑算法进一步发散） |

**两阶段白盒对齐方法论**（报告称为 preview，尚未完成）：

```
前提：关闭随机采样 + 开启确定性通信 + 两侧并行策略完全对齐

Phase I  : 训练前向 ↔ 推理 Prefill 严格对齐（解 A 和 B 类）
           昇腾定制 tensor hook 逐层导出原始二进制 → 高精度二进制差异比对 → 反向定位
Phase II : Decode 对齐（解更深的 C 类），需要 kernel 级介入
```

> **为什么这一章值得单独读**：另外四家更多是把训推一致当成"我们做到了"来陈述，**只有这一份把它当成一个待解的工程问题、并给出了分类学**。
>
> ⚠️ **B 类里那条最值得单独拎出来的观察**——✅ 报告原文：*"MLA switching between Absorptive (MQA, decode) and Non-Absorptive (MHA, training/prefill) modes"*，即 **MLA 在 decode 走 MQA 吸收模式、在训练/prefill 走 MHA 模式**，同一份权重在两条路径上跑的是**不同的算子形状**，天然是一个失配源。
>
> ★ **这个隐性成本落在谁头上**（⚠️ 以下是我按各家报告原文的归属，2026-08 核对）：
> ```
> ✅ 有这个失配源：openPangu（报告原文如上）
>                  GLM-5.2 —— ✅ 报告写明推理侧要"统一采用 MLA 的 MQA 模式"
>                  Kimi K3 —— ⚠️ 仅限它那 1/4 的 Gated MLA 层
> ❌ 没有这个失配源：MiniMax M3 —— 用 GQA，训推同一条路径
>                    ★ DeepSeek-V4 —— 它【根本不用 MLA】：报告全文没出现过
>                      MLA，CSA/HCA 的内核直接就是共享 KV 的 MQA
> ```
> ⟹ ★ 所以这是 **MLA 这条技术路线的隐性成本**，不是"用了压缩 KV 就一定有"。V4 走压缩路线却绕开了它。

---

## 十一、推理与量化

### 11.1 量化：W8A8，**不是 4-bit**

```
静态对称的输出通道权重量化 + 动态 per-token 激活量化
Pro：权重占用 1009 GB → 517 GB，压缩 1.95×
```

校准用 channel-wise smoothing，并**放松了 SmoothQuant 的 `a + b = 1` 硬约束**，改成对解耦指数 `(a, b)` 做二维搜索；对关键的 up projection，输入侧与输出侧平滑系数**交替迭代优化至收敛**。

**保持 BF16 的模块**（报告给了明确的决策依据）：融合算子只接受 BF16 输入的、**位于离散选择之前对量化噪声敏感的（TopK 路由和 indexing）**、误差会累积的残差路径、开销大于收益的小层、已经被侧流窗口掩盖的算子。

> ⚠️ **这是五家里最保守的量化**。V4 是 FP4 存 + FP8 算、K3 是 MXFP4 + MXFP8、GLM 是 INT4 + W8A8、M3 出货 MXFP8。**openPangu 停在 W8A8，而且报告没有提 QAT——看起来是训练后量化（PTQ）。** 另外还有两篇独立的 openPangu 量化论文（arXiv:2606.21257 昇腾量化实证研究、arXiv:2512.23367 Atlas A2 上的 PTQ），说明这条线是单独在推进的。

### 11.2 实测性能（报告 Table 6/7，全部为昇腾实测）

> 📖 **先解释表头的五个缩写**（不熟悉的话，[大模型架构入门/06-KV-cache.md §6.1](大模型架构入门/06-KV-cache.md) 有完整讲解）：
>
> | 缩写 | 全称 | 是什么 |
> |---|---|---|
> | **ISL** | Input Sequence Length | 输入有多少 token（"8K+1K"里的 8K） |
> | **OSL** | Output Sequence Length | 输出有多少 token（"8K+1K"里的 1K） |
> | **TTFT** | Time To First Token | **首字延迟**：按下回车到看见第一个字，单位秒 |
> | **TPOT** | Time Per Output Token | **每字延迟**：之后每多吐一个字的间隔，单位毫秒。★ 就是你看到的"打字速度" |
> | **TPS** | Tokens Per Second | **吞吐量**：一块卡每秒总共吐出多少 token |
>
> ★ **TPOT 和 TPS 别混**：前者是"我一个人要等多久"（越小越好），后者是"这块卡一共干了多少活"（越大越好），两者靠 `TPS ≈ batch ÷ TPOT` 联系，且**互相打架**——下面两张表正好是同一个模型的两个取舍点。

**超低延迟**（牺牲吞吐换延迟）：

| 模型 | ISL + OSL | TPOT (ms) |
|---|---|---|
| Flash | 8K+1K | 5.27 |
| Flash | 128K+1K | 5.63 |
| **Pro** | 8K+1K | **9.53** |
| **Pro** | 128K+1K | **9.55** |

> **Pro 从 8K 到 128K，TPOT 只涨了 0.02 ms。** 这是 DSA+SWA 混合架构最直接的效果验证——**解码成本几乎与上下文长度无关。**
> （换算一下：9.53 ms/token ≈ **每秒 105 个字**，远快于人的阅读速度）

**低延迟吞吐**（牺牲延迟换吞吐；PD 分离部署，Pro 用 2P1D 八节点）：

| 模型 | ISL + OSL | TPOT (ms) | Batch | **TPS (tokens/s per NPU)** |
|---|---|---|---|---|
| Flash | 8K+1K | 16.2 | 48 | 2963 |
| Flash | 128K+8K | 13.0 | 24 | 1846 |
| **Pro** | 8K+1K | 23.2 | 40 | **1727** |
| **Pro** | 128K+8K | 18.1 | 24 | **1326** |

> ⚠️ **对照上面那张表**：同一个 Pro，batch 开到 40 之后 TPOT 从 **9.53 ms 变成 23.2 ms**（慢 2.4 倍），换来 1727 TPS 的吞吐。**这两组数字不能凑在一起用。**

**Prefill**：128K 单 batch 输入下，Flash **TTFT 2.6 s**、Pro **TTFT 3.0 s**——即输入 12.8 万 token 时，用户要等约 3 秒才看到第一个字。

⚠️ **这些数字全部没有对比基线**，只能自己和自己比。官方博客那句"单卡吞吐率可达业界主流开源模型的 2 倍"**在报告正文里找不到出处**。

### 11.3 昇腾侧的工程优化（择要）

- **算子**：mHC 拆成三个可重叠的算子（含专用 sinkhorn 算子）；SWA 侧提出 **intra-core 动态 micro-tile 级 mask-skip**——把 L1 上的粗粒度基本块切成与计算阵列物理维度对齐的 micro-tile，**在 L1→L0 搬运前跳过"全无效" micro-tile**，同时让 PAS 常驻 L1。
- **通信**：MoE Dispatch 改为 **token-centric 执行模型**——每个 token **只读取和量化一次**再分发给所有路由专家，消除 Top-k 相关的冗余计算，**Dispatch 延迟降 25~30%，端到端延迟最多改善 5%**。
- **图优化**：CANN 的 ACL Graph + Npugraph_ex（静态 kernel 编译 + Superkernel 融合），**合计改善 TPOT 3.2 ms**（静态编译 1.2 ms + Superkernel 2 ms）。
- **并行**：MoE 走经典 EP；注意力在 prefill 侧用 **DSA CP + SWA TP**、decode 侧用 Attention DP。CP 方案叫 **Felix**。
- 训练侧部署在 **CloudMatrix 384 超节点**。

---

## 十二、评测

### 12.1 官方全表（报告 Table 5，与 HF 模型卡完全一致）

解码设置 `top_p 0.8, temperature 1.0`。**加粗为 openPangu-2.0 系列内部最优。**

| Benchmark | Flash 非思考 | Flash 思考 | Pro 非思考 | **Pro 思考** |
|---|---|---|---|---|
| **General** | | | | |
| CL-Bench (Acc) | 15.5 | **20.4** | 17.8 | 19.7 |
| IFEval (Prompt Strict) | 89.3 | **95.9** | 90.4 | 94.5 |
| IFBench (Prompt Strict) | 54.4 | 79.6 | 59.5 | **79.9** |
| AgentIF ((CSR+ISR)/2) | 43.9 | **44.9** | 42.2 | 44.4 |
| SysBench (ISR) | 87.9 | **91.1** | 82.6 | 90.8 |
| Multichallenge (Acc) | 51.9 | **68.4** | 52.2 | 62.9 |
| SimpleQA Verified (Acc) | 21.6 | 24.5 | 27.1 | **34.9** |
| Chinese-SimpleQA (Acc) | 66.0 | 66.7 | **75.7** | 74.2 |
| **Reasoning** | | | | |
| HMMT Feb 2026 (Avg@16) | 63.2 | 76.5 | 63.3 | **86.2** |
| – w/ Python | – | 90.7 | – | **93.2** |
| AIME 2026 (Avg@16) | 86.5 | 93.3 | 87.3 | **95.4** |
| – w/ Python | – | **98.1** | – | 97.9 |
| Apex (Avg@16) | – | 1.0 | 2.1 | **17.7** |
| IMO-AnswerBench (Acc) | 62.3 | 76.5 | 60.8 | **84.3** |
| BBEH (Harmonic Mean) | 51.5 | 62.5 | 55.0 | **69.7** |
| GPQA-Diamond (Avg@4) | 79.8 | 83.7 | 81.1 | **87.9** |
| HLE (Acc) | 8.5 | 20.3 | 9.0 | **27.1** |
| **Agent** | | | | |
| TAU2-Bench (Avg@3) | 74.0 | **88.0** | 71.6 | 81.1 |
| MCP-Atlas (Acc) | 47.9 | 58.9 | 48.8 | **61.3** |
| SkillsBench (Avg@5) | 40.0 | 42.6 | 47.5 | **48.9** |
| PinchBench (Avg@3) | 82.5 | 85.6 | **88.9** | 84.0 |
| WildClawBench (Avg@3) | 35.0 | 41.5 | 43.1 | **46.5** |
| Claw-Eval (Pass^3) | 58.2 | 57.7 | 49.4 | **61.7** |
| GDPVal (Acc) | – | 51.4 | 51.2 | **74.9** |
| BrowseComp (Acc) | – | 57.0 | – | **65.7** |
| **Coding** | | | | |
| LiveCodeBench V6 (Avg@3) | 50.9 | 85.1 | 74.5 | **85.7** |
| FeatBench (Avg@3) | 45.8 | 45.9 | **48.4** | 47.1 |
| DeepCodeBench (Avg@3) | 70.9 | **76.5** | 72.6 | 74.2 |
| SWE-bench Verified (Avg@3) | 57.6 | 63.1 | 66.8 | **68.5** |

**几个内部观察**：

1. **Thinking 档的收益极度不均**。HLE `9.0 → 27.1`（3×）、Apex `2.1 → 17.7`（8.4×）、IMO `60.8 → 84.3`；但 PinchBench **反而从 88.9 掉到 84.0**、Chinese-SimpleQA `75.7 → 74.2`、FeatBench `48.4 → 47.1`。**思考不是免费的。**
2. **Flash 在多项 Agent 任务上反超 Pro**：TAU2-Bench 88.0 vs 81.1、SysBench 91.1 vs 90.8、Multichallenge 68.4 vs 62.9、DeepCodeBench 76.5 vs 74.2。**92B 在几个 agent 基准上打赢 505B，这件事报告没有解释。**
3. **SimpleQA Verified 34.9 与 Chinese-SimpleQA 74.2 差 39 分**——中英文事实性知识的巨大不对称，是所有公开数字里最刺眼的一处。

### 12.2 ⚠️ 与四家的差距（全部为跨厂商拼接，见文首前提）

Pangu 取 **Pro-Thinking**；K3 取其报告，GLM-5.2 / M3 / V4-Pro 取 Z.ai 模型卡同一张表。

| Benchmark | **Pangu-Pro** | K3 | GLM-5.2 | M3 | V4-Pro | 差距 |
|---|---|---|---|---|---|---|
| **数学 / 推理** | | | | | | |
| AIME 2026 | **95.4** | — | 99.2 | — | 94.6 | −3.8 ／ **赢 V4 +0.8** |
| HMMT Feb 2026 | 86.2 | — | 92.5 | 84.4 | 95.2 | −6.3~−9.0 ／ **赢 M3 +1.8** |
| IMO-AnswerBench | 84.3 | — | 91.0 | — | 89.8 | −5.5~−6.7 |
| GPQA-Diamond | 87.9 | 93.5 | 91.2 | 93.0 | 90.1 | −2.2 ~ **−5.6** |
| **知识** | | | | | | |
| HLE（无工具） | 27.1 | 43.5 | 40.5 | 37.0 | 37.7 | **−9.9 ~ −16.4** |
| SimpleQA Verified | 34.9 | — | — | — | 57.9 | **−23.0** |
| **Agent** | | | | | | |
| MCP-Atlas | 61.3 | — | 76.8 | 74.2 | 73.6 | **−12.3 ~ −15.5** |
| BrowseComp | 65.7 | 91.2 | — | 83.5 | — | **−17.8 ~ −25.5** |
| Claw-Eval | 61.7 | — | — | 74.5 | — | −12.8 |
| **编程** | | | | | | |
| SWE-bench Verified | 68.5 | — | — | 80.5 | — | **−12.0** |
| LiveCodeBench | 85.7 (V6) | — | — | — | 93.5（版本未标 ⚠️） | ~−7.8 |

**差距的形状比大小更有信息量**：

```
数学竞赛类   ：−3.8 ~ −9      ← 基本同档，AIME 反超 V4-Pro、HMMT 反超 M3
知识类       ：−10 ~ −23      ← 差一代
Agent / 工具 ：−12 ~ −26      ← 差一代，且是最大的一处
```

**这个形状和那四家的共性诊断不一样。** 四家收敛出的结论是"推理靠架构、知识靠数据"，短板**只在知识**（见 [模型技术对比.md](模型技术对比.md) §5.3）。**openPangu 是知识和 Agent 双短，而 Agent 恰恰是那四家最强的一格。**

一个与公开事实自洽的解释：

> **AIME/HMMT 是短轨迹、可验证、RL 便宜的任务；BrowseComp / MCP-Atlas / SWE-bench 是长周期多轮、环境昂贵、rollout 长尾严重的任务。** 后者正是那四家把半本报告砸进去的地方。余承东公开说过"留给自己的算力非常有限"，报告也承认在复杂真实软件工作流上与第一梯队有差距。**算力约束下先保短轨迹 RL，是一个说得通的取舍。**

**18B 激活是五家里最低的**，也指向同一方向。

⚠️ **不放进上表的三项**（避免误导）：

- **IFBench**：Pangu 79.9（prompt-strict）vs M2 报告消融里的 23.1/27.2——**差距太大，几乎确定是不同子集或不同判定口径**，不可比。
- **τ²-Bench**：Pangu 81.1（Avg@3，总分）vs M2 的 τ²-retail 62.3 / τ²-telecom 32.5（**分子集，且是 M2 的 SFT 消融模型**），不可比。
- **Apex**：Pangu 的 Apex 17.7 在 Reasoning 分组、Avg@16；M3 模型卡的 apex-agents 27.7 听起来是 agent 基准。**是否同一个 benchmark 未能确认**，不比。

### 12.3 第三方数据点

- **WildClawBench 榜单**：openPangu-2.0-**Flash** 41.5 排第 7，低于 GLM-5（42.6）和 MiMo-V2.5-Pro（43）。Pro 官方自测 46.5，若能复现则会超过该快照里的这两个。⚠️ 该榜单列的是 GLM-**5** 而非 5.2，且 Pro 尚未见第三方复测。
- 截至 2026-08-05，**Artificial Analysis 等独立评测平台尚未发布 Pro 的结果**。

### 12.4 报告自己的 in-house harness

报告另有 §5.2 两节：**Deep Research 与交互式可视化**、**全栈迷你应用生成**。这两节是内部 harness 的定性/对比展示，**同样没有外部模型对照**。

---

## 十三、这份报告真正的贡献在哪

**如果只看榜单，openPangu-2.0-Pro 是"数学接近第一梯队、Agent 差一代"。但这不是它的重点。**

| 它是什么 | 证据 |
|---|---|
| **一台高完成度的集成机** | mHC ← DeepSeek、DSA ← DeepSeek、Muon ← 公共、OPD 公式 ← GLM-5（明确标注）、TITO ← GLM-5、sink ← gpt-oss 谱系、R3 ← 社区。**报告全部如实引用，没有包装成原创。** |
| **昇腾全栈的可行性证明** | 505B 全程在昇腾 NPU 上训练，**开源了预训练代码、后训练代码和训练算子**——那四家**没有一家开源训练框架** |
| **四条原创增量** | ① PAS 及其 vs gpt-oss sink 的消融；② mHC 的 gate-collapse 失效模式与两级归一化解法；③ aux-loss-free 的 RMS 归一化更新；④ 训推不一致的三分类根因解剖 |
| **一份对使用者更实用的 mHC 文档** | V4 报告讲了 mHC 为什么需要，openPangu 讲了**照原样用会踩什么坑** |

**最值得记的一句**：这份报告在**它自己关心的问题上**（怎么在国产硬件上把 500B MoE 训稳、推快）披露得比另外四家都细——训推不一致那一章的颗粒度，五家里没有第二份。**它的"薄"只出现在能力对比上，而那恰好是它选择不谈的部分。**

---

## 十四、可靠性说明

| 内容 | 来源 | 可信度 |
|---|---|---|
| Table 8 架构超参、Table 1 效率系数、Table 2 mHC 消融、Table 3 MTP 调度、Table 5 全部评测、Table 6/7 延迟吞吐、三阶段课程与全部超参、OPD 公式、训推失配三分类、量化配置、附录 A.2/A.3 消融 | **报告正文提取** | 一手，可在 [papers/openPangu-2.0-正文提取.txt](papers/openPangu-2.0-正文提取.txt) grep 核对 |
| 505B/18B、512K、34T、许可、发布时间线 | HF 模型卡 + 华为官网 | 一手 |
| 541B 的口径拆解、参数量验算 | **本文推算**（基于 Table 8） | 推算，两项均与官方数字吻合 |
| §12.2 与四家的全部差距 | **跨厂商拼接** | ⚠️ **二手组合，口径不统一** |
| WildClawBench 榜单排位 | 第三方榜单快照 | 二手，且是 Flash 不是 Pro |

**最需要保留怀疑的五处**：

1. **没有任何对照模型**——所有横向结论都不是华为的主张，是我拼的。
2. **完全没有长上下文评测**。全文 `RULER` 出现 0 次，MRCR、NIAH 也没有。**它宣称 512K，但没有任何逐长度召回数据**——比 GLM-5.2 的"1M 无曲线"更彻底。而且它是五家里唯一**不到 1M** 的。
3. **所有消融都在 15B/2B 上做**（mHC 1T tokens、PAS 250B tokens、负载均衡同理），**离 505B 很远**。PAS 那一组只给曲线不给数值表。
4. **官方博客的"单卡吞吐 2 倍于业界主流开源模型"在报告正文里找不到出处**，Table 6/7 全部是自己和自己比。
5. **MTP 没有给接受长度**，无法与 GLM 的 2.76 / DeepSeek-V3.2 的 2.55 对比；**量化没有提 QAT**，看起来是 PTQ，与另外四家的训练时量化不是一回事。

---

## 参考来源

- **openPangu-2.0: Towards Reliable and Efficient Agentic Reasoning**（45 页技术报告，无 arXiv 编号）— 本地：[papers/openPangu-2.0-Tech-Report.pdf](papers/openPangu-2.0-Tech-Report.pdf)
- [openpangu/openPangu-2.0-Pro (Hugging Face)](https://huggingface.co/openpangu/openPangu-2.0-Pro) — 完整评测表与模型卡
- [openPangu-2.0-Pro 模型及技术报告正式开源上线（华为官网，2026-07-31）](https://www.huawei.com/cn/news/2026/7/openpangu)
- [Ascend Tribe 开源社区（GitCode）](https://gitcode.com/ascend-tribe) — 训练/推理框架、算子
- mHC 原论文：[Manifold-Constrained Hyper-Connections (arXiv:2512.24880)](https://arxiv.org/abs/2512.24880)（DeepSeek，V4 同款组件）
- 相关独立论文：[An Empirical Study of OpenPangu Quantization on Ascend NPUs (arXiv:2606.21257)](https://arxiv.org/abs/2606.21257) · [Post-Training Quantization of OpenPangu Models for Efficient Deployment on Atlas A2 (arXiv:2512.23367)](https://arxiv.org/abs/2512.23367)
- [如何评价华为盘古 5050 亿参数的 openPangu-2.0-Pro 模型及技术报告正式开源上线？（知乎）](https://www.zhihu.com/question/2066567142019786000)
- [Huawei Open Sources 505B openPangu AI, Drops Weights And Code (Open Source For You)](https://www.opensourceforu.com/2026/08/huawei-open-sources-505b-openpangu-ai-drops-weights-and-code/)
- [After Vowing to Make Pangu World No.1, Yu Chengdong Opens openPangu-2.0-Pro (Pandaily)](https://pandaily.com/huawei-openpangu-20-pro-open-source-ascend-jul2026)
