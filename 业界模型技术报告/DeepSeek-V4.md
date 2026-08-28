# DeepSeek-V4 架构与技术报告解析

> 📘 **不懂模型架构？先读这份**：[大模型架构入门/](大模型架构入门/README.md) —— **17 章 + 速查表**，从零讲清 注意力 / KV cache / MLA / MoE / RoPE / MTP / Muon / 量化，**不预设任何架构基础**。本文用到的术语都能在那里找到解释。

> **来源**：本文整理自 2026-08-03 的调研对话记录，内容基于技术报告 arXiv:2606.19348（2026-04-26）与官方模型卡。
> **本地论文**：[papers/arxiv-2606.19348.pdf](papers/arxiv-2606.19348.pdf)（58 页技术报告）、[papers/DeepSeek-V4-model-card-EN.pdf](papers/DeepSeek-V4-model-card-EN.pdf)（8 页官方模型卡）
> **横向对比**：见 [模型技术对比.md](模型技术对比.md)（DeepSeek-V4 / Kimi K3 / GLM-5.2 / MiniMax M3 / openPangu-2.0 五方）
> **同系列**：[Kimi-K3.md](Kimi-K3.md) ｜ [GLM-5.2.md](GLM-5.2.md) ｜ [MiniMax-M3.md](MiniMax-M3.md) ｜ [openPangu-2.0-Pro.md](openPangu-2.0-Pro.md)

---

## 一、先给结论：V4 到底改了什么

DeepSeek-V4 有两个型号，都原生 1M 上下文：

| | V4-Pro | V4-Flash |
|---|---|---|
| 总参数 / 激活 | 1.6T / 49B | 284B / 13B |
| 预训练数据 | 33T tokens | 32T tokens |
| 1M 上下文下单 token FLOPs（vs V3.2） | **27%** | 10% |
| 1M 上下文下 KV cache（vs V3.2） | **10%** | 7% |

**整个 V4 的叙事只有一句话：不是"能跑到 1M"，而是"1M 便宜到可以当默认值"。** 所有架构改动都是为这个目标服务的。

骨架仍然是 V3 那套 `DeepSeekMoE + MTP`，但在三个位置做了**替换**：

| 位置 | V3.2 | V4 | 解决什么问题 |
|---|---|---|---|
| 注意力 | MLA + DSA | **CSA + HCA 混合** | 长上下文的算力和显存 |
| 残差连接 | `x + F(x)` | **mHC**（流形约束超连接） | 1.6T 深层网络训练不稳定 |
| 优化器 | AdamW | **Muon** | 收敛更快、更稳 |

下面逐个用最小例子讲。

---

## 二、改进点 1：混合注意力 CSA + HCA（最核心）

### 问题是什么

假设上下文有 **100 万个 token**。传统注意力（哪怕是 MLA）要为每个 token 存一份 KV，生成第 1000001 个 token 时，要和前面 100 万个逐一算相似度。这就是 1M 上下文贵的根源。

V4 的思路很朴素：**先把序列"压缩"成更短的序列，再在压缩后的序列上做注意力。**

### CSA：压缩稀疏注意力（"精读"）

#### 最小例子

假设序列只有 16 个 token，压缩率 `m = 4`：

```
原始 KV:  [t0 t1 t2 t3] [t4 t5 t6 t7] [t8 t9 t10 t11] [t12 t13 t14 t15]
              ↓              ↓              ↓                ↓
压缩条目:     c0             c1             c2               c3
```

16 个 token → 4 个压缩条目，序列长度直接砍到 1/4。

压缩不是简单平均池化，而是**学出来的加权求和**：模型另外算一路"压缩权重" `Z`，加上可学习的位置偏置后做 softmax，再和 KV 做加权和。

**一个关键小设计——重叠压缩**：如果硬切成块，`t3` 和 `t4` 明明相邻却被切进不同块。所以 V4 用了两路 KV（`Ca`、`Cb`），`Cb` 取的是**前一块**的区间：

```
c1 的信息来源 = [t4 t5 t6 t7]（来自 Ca）+ [t0 t1 t2 t3]（来自 Cb）
                → softmax 在这 2m=8 个元素上统一归一化
```

所以每个压缩条目实际看到 `2m = 8` 个 token，但序列长度仍然只压到 1/4。**相邻条目共享信息，块边界的硬切断被抹平了。**

#### 压完之后：只挑重要的看

压缩之后 1M token → 25 万条目，还是多。于是用 **Lightning Indexer**（V3.2 就有，V4 沿用）打分：

```
I(t,s) = Σ_h  w_h · ReLU( q_h · K_s )
```

给每个压缩条目打个分，取 **top-k**（k 比 V3.2 更小，因为条目本身已经短了 4 倍），只对这 k 条做真正的注意力。

再加两个补丁：

- **滑动窗口**：最近 128 个 token 保留**不压缩**的原始 KV（因为最近的内容需要看细节，且同一块内因果关系被压掉了）
- **Attention sink**：每个头有个可学习的 sink logit，加在 softmax 分母上，允许某个头的注意力总和 < 1，甚至接近 0（"这一层我不想看任何东西"）

### HCA：重度压缩注意力（"泛读"）

同样的压缩思路，但压缩率拉到 `m′ = 128`，而且**不重叠、不做稀疏筛选**：

```
100 万 token / 128 ≈ 7800 个条目 → 全部都看（稠密注意力）
```

7800 条已经很便宜了，直接全量注意力，**保证全局连贯性不丢**。

### 两者交替 = 精读 + 泛读

```
CSA 层：看得细（1/4 压缩），但只挑 top-k 看 → 局部精度
HCA 层：看得糙（1/128 压缩），但全都看   → 全局视野
```

层与层之间交错排列。这就是 1M 上下文能被"扛住"的全部秘密——**没有一层同时既看得细又看得全，但堆起来两者都有。**

### 还有几个工程细节

- **QK-RMSNorm**：对 query 和压缩 KV 条目直接做 RMSNorm，防止 attention logits 爆炸。副作用是 Muon 里**不需要 QK-Clip** 了。
- **Partial RoPE + 反向 RoPE**：只在最后 64 维加 RoPE。但因为压缩条目**同时充当 key 和 value**，输出会带上绝对位置信息；于是对输出 `o_i` 的最后 64 维再施加一次位置为 `−i` 的 RoPE，把绝对位置抵消掉，恢复相对位置性质。这个 trick 挺巧。
- **KV 存储混合精度**：RoPE 维度用 BF16，其余用 FP8，KV cache 再砍一半。Indexer 的注意力计算用 FP4。
- **最终效果**：以 BF16 / GQA-8 / head_dim=128 为基线，1M 上下文下 KV cache 只有约 **2%**。

---

## 三、改进点 2：mHC —— 把残差连接换掉

### 问题：残差为什么会炸

标准残差：`x_{l+1} = x_l + F(x_l)`。只有一条"车道"，信号一路加上去。层数一深、参数量一大，数值就容易爆。

**Hyper-Connections (HC)** 的想法是把这一条车道**加宽成 n 条**：

```
x_{l+1} = B_l · x_l  +  C_l · F(A_l · x_l)
```

- `A`：把 n 条车道汇成 1 条，喂给这一层（所以层内部结构完全不用改）
- `B`：n×n 矩阵，让 n 条车道互相混合
- `C`：把这一层的输出再分发回 n 条车道

问题出在 `B`：如果它的谱范数 > 1，每过一层信号就放大一点，60 层叠下来能放大 **3000 倍**，训练直接崩。

### mHC 的解法：把 B 关进 Birkhoff 多胞体

V4 强制 `B` 是**双随机矩阵**（行和 = 1、列和 = 1、元素非负）。

**最简单的例子**：把 `B` 想成"往 4 条车道分水"。

```
        到车道0  到车道1  到车道2  到车道3   行和
车道0     0.5     0.2     0.2     0.1   = 1.0
车道1     0.2     0.4     0.3     0.1   = 1.0
车道2     0.2     0.3     0.4     0.1   = 1.0
车道3     0.1     0.1     0.1     0.7   = 1.0
列和      1.0     1.0     1.0     1.0
```

**行和为 1** ⇒ 每条车道的水全部流出去、一滴不多；**列和为 1** ⇒ 每条车道收到的水总量不变。**水的总量守恒 → 谱范数 ‖B‖₂ ≤ 1 → 永远不放大。**

而且双随机矩阵集合对乘法封闭，所以 61 层连乘下来仍然不放大。**这就把 3000× 的放大压到了 2× 以内。**

实现上：先 `exp()` 保证元素为正，再用 **Sinkhorn-Knopp 交替归一化行和列 20 次**，逼近双随机。`A = σ(Ã) ∈ (0,1)`，`C = 2σ(C̃) ∈ (0,2)`，都用 Sigmoid 卡住范围。

`A/B/C` 还是**动态的**（由当前 token 的 hidden state 算出来）+ 静态偏置 + 一个初始化为很小值的门控 `α`，让训练早期约等于普通残差，再慢慢学。

**代价**：mHC 让墙钟时间多了 **6.7%**（已经通过融合 kernel、选择性重计算、调 DualPipe 优化过了）。

---

## 四、改进点 3：Muon 优化器

**Adam 的逻辑**：对每个参数**单独**看历史梯度大小，梯度大的方向步子迈小点。

**Muon 的逻辑**：把梯度当成**矩阵** `G`，做 SVD 得 `G = UΣVᵀ`，然后把所有奇异值都**强制设为 1**，用 `UVᵀ` 去更新。

用大白话讲：**Adam 是"逐个方向调音量"，Muon 是"把所有方向的音量拉平"**——不让某几个主导方向吃掉全部更新量，更新更均衡，收敛更快。

真做 SVD 太贵，所以用 **Newton-Schulz 迭代**近似。V4 用了个**混合系数**方案，共 10 步：

- 前 8 步：`(3.4445, −4.7750, 2.0315)` —— 激进，快速把奇异值推向 1
- 后 2 步：`(2, −1.5, 0.5)` —— 温和，把奇异值稳稳收在 1

**哪些参数用 Muon**：矩阵形参数（MoE 专家、注意力投影）全用。
**哪些不用**：embedding、预测头、mHC 的静态偏置和门控、所有 RMSNorm 权重 —— 这些继续用 AdamW。

工程上有个坑：Muon 需要**完整的梯度矩阵**，而 ZeRO 会把参数切碎。V4 为此设计了混合 ZeRO 分桶（背包算法分配矩阵、保证不切分逻辑独立矩阵、padding 开销 < 10% 显存）。

---

## 五、改进点 4：MoE 的三处小调整（沿用 DeepSeekMoE）

V4-Pro 配置：**61 层、hidden 7168、384 个路由专家 + 1 共享专家、每 token 激活 6 个**。

1. **亲和度打分函数：Sigmoid → `Sqrt(Softplus(·))`**
   Sigmoid 在分数很高时会饱和（输出贴近 1，导数≈0），"很想要这个专家"和"极其想要这个专家"梯度都消失了。Sqrt(Softplus) 不饱和，梯度一直有。

2. **浅层改用哈希路由**
   前几层原本是 dense FFN，V4 换成按 **token ID 哈希**决定专家的 MoE 层。哈希是确定性的 → 天然负载均衡、零路由开销，还白赚了参数量。

3. **取消路由目标节点数限制**
   V3 限制每个 token 最多路由到 4 个节点（省通信），V4 直接取消，靠新的并行策略解决。

**MTP 完全没改**，原文说 "we adopt the same strategy for DeepSeek-V4 series without modification"，深度为 1，用于投机解码。

---

## 六、训练稳定性的两个"土办法"

报告很坦诚，这两个是**经验性的、没有理论解释**：

- **前瞻路由（Look-ahead Routing）**：第 t 步计算特征用 `θ_t`，但**路由索引用 `θ_{t−Δt}`**（一个稍旧的快照）。打破"路由决策"和"骨干网络"的同步耦合，几乎零开销地消除 loss spike。
- **SwiGLU 截断**：线性分量强行限制在 `[−10, 10]`，gate 上限也是 10。

---

## 七、Infra 层：为什么便宜

- **MegaMoE 融合 kernel**：把专家切成多个 wave 流水，做到「算当前 wave + 发上一 wave + 收下一 wave」三者并行，相比非融合基线加速 **1.50~1.73×**，RL rollout 场景最高 **1.96×**。
- **FP4 QAT**：MoE 专家权重走 `FP32 主权重 → FP4 存储 → FP8 计算`。关键点是 **FP4→FP8 反量化是无损的**（E4M3 指数位更多），所以能直接复用现有 FP8 训练框架。
- **TileLang**：自研 kernel DSL，把 kernel 启动的 CPU 侧开销从几十~几百微秒压到 **< 1 微秒**，还集成 Z3 SMT 求解器做索引约束的形式化验证。
- **逐位确定性**：注意力用双 kernel 替代 split-KV，矩阵乘用 DeepGEMM 替代 cuBLAS，反向传播给每个 SM 独立累加 buffer 消除 atomicAdd 不确定性 —— 训练/推理 bitwise 可复现。
- **推理侧缓存**：把滑动窗口和未压缩尾部视为"状态空间模型"（大小固定、只取决于当前位置），可以**预分配固定大小缓存池**；共享前缀（如同一套 system prompt）的 KV cache 存到**磁盘**复用。这就是 cache hit 价格能做到 $0.0036/M 的原因。
- **给硬件厂商的建议**（报告专门写了一节）：算力/带宽比 `C/B ≤ 6144 FLOPs/Byte` 比绝对带宽更重要；预留功耗余量；提供低延迟信令支持 push 模式；设计比 SwiGLU 更简单的激活函数。

---

## 八、后训练：领域专家 → on-policy 蒸馏合并

这块和 on-policy distillation survey 正好对上（该综述在本仓库上一级目录）。

**两阶段范式**：

```
阶段 1：分领域各练各的
  数学专家   ← SFT + GRPO（数学专用奖励）
  代码专家   ← SFT + GRPO（代码专用奖励）
  Agent 专家 ← SFT + GRPO（Agent 专用奖励）
  指令专家   ← ...

阶段 2：用 on-policy distillation 合并成一个模型
  学生自己采样 → 各领域教师给出分布 → 最小化 Σ w_i · D_KL(π_student ‖ π_teacher_i)
```

**为什么不用别的办法**：权重合并会因为参数空间不对齐而失效；混合 RL 会让不同领域的奖励互相冲突。**OPD 成本高，但效果最好。**

两个关键设计：

- **逆 KL**（`D_KL(学生 ‖ 教师)`）而非正向 KL —— 逆 KL 是 **mode-seeking**（学一个模式学得像），正向 KL 是 mode-covering（把所有模式糊在一起）。蒸馏要的是前者。
- **全词表 logit 计算 KL**，而不是 token 级采样估计 —— 梯度方差小得多，多教师场景下尤其必要。

另外几个后训练亮点：

- **GRM（生成式奖励模型）**：不训标量奖励模型，让模型**生成**一段评估判断，并对 GRM 本身也做 RL。信息量远超一个分数。
- **三档推理力度**：Non-think（8K）、Think High（128K）、Think Max（384K），**每一档单独训练专家**。
- **交错思考**：V3.2 每轮丢弃思考痕迹，V4 在工具调用场景**保留并跨轮累积**（1M 窗口才撑得住）。
- **Quick Instruction**：把意图识别、搜索触发等辅助任务编码成特殊 token 附加到输入，**复用 KV cache**，避免重新 prefill。
- **RL Infra**：可抢占 rollout 服务 + token 粒度 WAL 日志（被抢占后从最后完成的 token 恢复）；Rust 写的 DSec 沙箱平台管理数十万并发沙箱。

---

## 九、效果与代价

**强的地方**（V4-Pro-Max）：

- Codeforces Rating **3206**（开源首次匹敌闭源，约等于人类第 23 名）
- LiveCodeBench **93.5**、HMMT 2026 Feb **95.2**
- 长上下文 MRCR 1M **83.5** > Gemini-3.1-Pro 的 76.3（但 < Opus 4.6 的 92.9）
- 中文功能写作胜率 62.7% vs Gemini-3.1-Pro 34.1%

**弱的地方**：

- SimpleQA-Verified **57.9** vs Gemini-3.1-Pro 75.6，**差 17.7 分**
- MMLU-Pro 87.5 vs 91.0

报告自己给的结论很值得记：**推理能力更依赖架构与训练方法，知识能力更依赖数据。** 差距主要在训练语料的质量与知识密度上，落后前沿约"3 到 6 个月"。

**报告承认的局限**：组件堆叠导致架构"不够优雅"；前瞻路由和 SwiGLU 截断缺乏理论解释；事实性知识仍落后顶尖闭源模型。

---

## 十、一张图串起来

一个 token 走完 V4 的完整路径：

```
输入 token
   ↓
[mHC 加宽的 n 条残差车道，B 被约束为双随机矩阵]
   ↓
第 0~1 层：HCA only
   ↓
第 2~60 层：CSA / HCA 交替
   ├─ CSA：4:1 重叠压缩 → Lightning Indexer 选 top-k → MQA + 128 滑窗 + sink
   └─ HCA：128:1 压缩 → 全量稠密注意力
   ↓
每层 FFN：DeepSeekMoE（384 路由专家选 6 + 1 共享，Sqrt(Softplus) 打分，浅层哈希路由）
   ↓
MTP 头（深度 1，投机解码）
   ↓
输出

训练：Muon（矩阵参数）+ AdamW（embedding/norm/mHC 门控），FP4 存专家 / FP8 算
后训练：领域专家（SFT+GRPO）→ 全词表逆 KL 的 on-policy 蒸馏合并
```

---

## 十一、可靠性说明

1.6T/49B、284B/13B、1M 上下文、32T tokens、27%/10% 效率数字、CSA+HCA、mHC、Muon 这些来自 arXiv 摘要和官方模型卡，可靠。

而具体超参（`m=4`、`m′=128`、61 层、384 专家、滑窗 128、Sinkhorn 20 次、Newton-Schulz 系数）来自社区对报告 §4.2.1 配置表的解读——数值高度自洽，但建议以本地 PDF [papers/arxiv-2606.19348.pdf](papers/arxiv-2606.19348.pdf) 原文核对一遍。

另外网上有些解读（比如说专家数是 896、说 Engram 进了 V4）是错的：**Engram 是独立论文 arXiv:2601.07372，没有进 V4**（本地已存 [papers/arxiv-2601.07372.pdf](papers/arxiv-2601.07372.pdf)）。"896 专家"实际上是 Kimi K3 的配置，参见 [Kimi-K3.md](Kimi-K3.md)。

---

## 参考来源

- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence (arXiv:2606.19348)](https://arxiv.org/abs/2606.19348) — 本地：[papers/arxiv-2606.19348.pdf](papers/arxiv-2606.19348.pdf)
- [DeepSeek-V4-Pro 官方模型卡 (Hugging Face)](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [DeepSeek V4 官方模型卡 PDF (EN)](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf) — 本地：[papers/DeepSeek-V4-model-card-EN.pdf](papers/DeepSeek-V4-model-card-EN.pdf)
- [DeepSeek 官方发布公告：1M 上下文普惠](https://api-docs.deepseek.com/news/news260424/)
- [DeepSeek-V4 技术报告深度解析（腾讯云开发者社区）](https://cloud.tencent.com/developer/article/2661899)
- [DeepSeek V4 GA: Architecture and Inference Efficiency (Hugging Face Blog)](https://huggingface.co/blog/ResterChed/deepseek-v4-ga-architecture)
- Engram（**非 V4 组件**，独立研究）arXiv:2601.07372 — 本地：[papers/arxiv-2601.07372.pdf](papers/arxiv-2601.07372.pdf)
