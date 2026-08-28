# GLM-5.3 / GLM-5.3-Flash 架构与技术报告解析

> 📘 **不懂模型架构？先读这份**：[大模型架构入门/](大模型架构入门/README.md) —— **17 章 + 速查表**，从零讲清 注意力 / KV cache / MLA / MoE / RoPE / MTP / Muon / 量化，**不预设任何架构基础**。本文用到的术语都能在那里找到解释。

> **重要前提一：这一代是"两个模型、一篇文档"**
>
> 智谱在 2026 年 8 月连发两个模型，它们的性质完全不同，**必须分开看**：
>
> | | **GLM-5.3** | **GLM-5.3-Flash** |
> |---|---|---|
> | 发布 | 2026-08-14 | 2026-08-26 |
> | 基座 | **和 GLM-5.2 是同一个 base model**，一行架构都没改 | **全新预训练的基座**，架构大改 |
> | 增量来自 | **纯后训练**（官方原话：*"Scaling post-training is all we did for GLM-5.3."*） | 架构 + 语料 + 后训练 |
> | 规模 | 744B 总参 / 40B 激活（继承 5.2） | **320B 总参 / 18B 激活** |
> | 模态 | 纯文本 | **原生多模态**（GLM-5 系列首个） |
> | 注意力 | MLA + DSA + IndexShare（继承 5.2） | **KDA 线性注意力 + DSA 稀疏注意力 混合** |
>
> **所以"GLM-5.3 的架构"这个问题本身是个陷阱**——本体没有架构，架构在 Flash 上。本文第二~四章讲 GLM-5.3（后训练与网络安全），第五章起讲 GLM-5.3-Flash（架构）。
>
> **重要前提二：这一代没有独立的技术报告论文**
>
> | 内容 | 来源 | 可靠性 |
> |---|---|---|
> | GLM-5.3 的后训练、环境合成、网络安全、slime 基建 | **官方技术博客 [z.ai/blog/glm-5.3](https://z.ai/blog/glm-5.3)**（2026-08-14，正文逐段核对） | 中高（一手，但厂商自述，无论文） |
> | GLM-5.3-Flash 的架构主张、效率倍数、部署 | **官方技术博客 [z.ai/blog/glm-5.3-flash](https://z.ai/blog/glm-5.3-flash)** | 中高（同上） |
> | **GLM-5.3-Flash 的全部架构参数** | **Hugging Face `zai-org/GLM-5.3-Flash` 的 `config.json`** | **高（一手，可复现核对）** |
> | KDA 的内部结构 | **Kimi Linear 论文 arXiv:2510.26692** + **vLLM 源码 `vllm/models/kimi_k3/amd/kda.py`** | 高（一手论文 + 可运行代码） |
> | 骨干沿革、Muon Split、DSA 引入方式、异步 RL | **GLM-5 技术报告 arXiv:2602.15763** | 高（一手） |
> | 评测表格的完整数字 | **缺失** —— 官方博客的评测表全是**图片**，只有正文里点名的数字可文本核对 | — |
>
> **本地论文**：[papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering-arXiv-2602.15763.pdf](papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering-arXiv-2602.15763.pdf) · [papers/GLM-5-正文提取.txt](papers/GLM-5-正文提取.txt) · [papers/IndexCache-Cross-Layer-Index-Reuse-arXiv-2603.12201.pdf](papers/IndexCache-Cross-Layer-Index-Reuse-arXiv-2603.12201.pdf)
> **前一代**：[GLM-5.2.md](GLM-5.2.md)（骨干配置表、MLA-256、Muon Split、DSA 续训、IndexShare 都在那篇）
> **横向对比**：[模型技术对比.md](模型技术对比.md) ｜ **同系列**：[DeepSeek-V4.md](DeepSeek-V4.md) ｜ [Kimi-K3.md](Kimi-K3.md) ｜ [MiniMax-M3.md](MiniMax-M3.md) ｜ [openPangu-2.0-Pro.md](openPangu-2.0-Pro.md)

---

## 一、一句话总览

**GLM-5.3 证明了"后训练还能单独榨出 50% 的提升"；GLM-5.3-Flash 证明了"把线性注意力和稀疏注意力叠起来，能用 1/10 的价格接近前沿智能"。**

这是两个方向完全正交的发布：

```
GLM-5.2 基座 (744B/40B)
   │
   ├─→ GLM-5.3        ：基座冻结，只加后训练  → 编程 +50%，网络安全能力涌现
   │
   └─（另起炉灶，全新基座）
        └─→ GLM-5.3-Flash (320B/18B)：KDA+DSA 混合注意力 + mHC + 原生多模态
                                       → 智能追平 Opus 4.8，价格 1/10，跑在国产芯片上
```

**代际全表**：

| | GLM-4.5 | GLM-5 | GLM-5.1 | GLM-5.2 | **GLM-5.3** | **GLM-5.3-Flash** |
|---|---|---|---|---|---|---|
| 发布 | 2025 | 2026-02 | 2026-04 | 2026-06-16 | **2026-08-14** | **2026-08-26** |
| 总参 / 激活 | 355B / 32B | 744B / 40B | 754B | 753B | **同 5.2** | **320B / 18B** |
| 层数 | 92 | 78 | 78 | 78 | 同 5.2 | **45** |
| 上下文 | 128K | 200K | 200K | **1M** | 1M | **1M** |
| 最大输出 | — | — | — | — | **128K** | 128K |
| 注意力 | GQA | MLA + DSA | 同 | + IndexShare | 同 5.2 | **KDA + DSA + IndexPool** |
| 模态 | 文本 | 文本 | 文本 | 文本 | 文本 | **文本 + 图 + 视频** |
| 许可 | — | — | — | MIT | MIT（延后两周） | **MIT** |

---

## 二、GLM-5.3：把"后训练"当成唯一变量的一次实验

### 2.1 官方口径：基座一行没动

博客第一句就是纲领：

> ✅ **官方原文**：*"Scaling post-training is all we did for GLM-5.3."*（GLM-5.3 我们做的全部事情就是把后训练规模化。）
>
> ✅ **官方原文**：*"It uses the same base model as GLM-5.2 — every gain comes from post-training."*（它用的是和 GLM-5.2 一样的基座——所有提升都来自后训练。）

**这句话在归档角度非常值钱**：它意味着 [GLM-5.2.md](GLM-5.2.md) 里那张骨干配置表（744B/40B、78 层、MLA-256、DSA、IndexShare、MTP）**原封不动适用于 GLM-5.3**。你不需要重新核对任何一个架构数字。

> 💡 **小白提示：什么叫"基座 / base model"？**
> 大模型训练分两段：**预训练**（在几十万亿字上学语言和知识，产出的东西叫"基座"）和**后训练**（教它怎么做任务、怎么用工具、怎么听话，主要手段是强化学习）。基座相当于"大脑硬件 + 通识"，后训练相当于"职业培训"。GLM-5.3 = GLM-5.2 的同一块大脑，换了一套更长更狠的职业培训。

### 2.2 后训练规模化做了什么：环境，而不是数据

官方把瓶颈定义得很明确：

> ✅ **官方原文**：*"As agent capability improves, much of the difficulty in scaling post-training moves from the model to the environment."*（随着智能体能力提升，后训练规模化的难点从模型转移到了环境上。）

**"环境"（environment）是什么**：强化学习不像监督学习那样喂"题目+标准答案"，它需要一个能让模型**真的动手、并且能自动判分**的沙盒。写代码就得有能跑的容器、能装依赖的网络、能执行的测试。这个沙盒就叫环境。

GLM-5.3 在环境上推了三件事：

**① 任务难度上台阶** ✅

> ✅ **官方原文**：*"Some represent several days of work for an experienced engineer."*（有些任务相当于一个有经验的工程师几天的工作量。）

官方举的例子是一个 **ML 基础设施任务**：模型拿到和工程师一样的工作环境——**算力集群、存储系统、内部文档、代码库、实验结果**，要求它**诊断训练栈的瓶颈 → 实施优化 → 跑实验 → 交付一个可测量的端到端加速，且不能破坏正确性**。

**② 环境自己合成，验证器也自己合成** ✅

这是这次最工程化的一段。官方描述的流水线：

```
真实工作 → [Research Agent] 抽取任务模式
              ↓
        生成可运行的长程环境（多步依赖 + 隐藏状态）
              ↓
        [Judge Agent] 亲自做一遍，确认"这题真的能做出来"
              ↓
        [Verifier 合成] —— 关键：合成验证器时【不给参考答案】
              ↓
        用 solver 轨迹去找"奖励捷径"并堵上
              ↓
        通过 oracle / no-op / unsolved-state 三项检查
              ↓
        产出可直接训练的二值奖励
```

> 🔑 **"合成 verifier 时不给参考答案"为什么重要**：如果判分器见过标准答案，它容易退化成"和标准答案比字符串"，模型就会学会**背答案而不是解决问题**。不给答案，判分器只能从"结果对不对"这个角度写检查，奖励才是真实的。
>
> 🔑 **三项检查是防作弊的**：
> - **oracle 检查**：把正确解喂进去，验证器必须判过（否则验证器太严，正确答案也过不了）
> - **no-op 检查**：什么都不做，验证器必须判不过（否则白给分）
> - **unsolved-state 检查**：题目初始状态，验证器必须判不过（否则题目本身就是"已完成"的）
>
> ⚠️ 官方也承认这条流水线**还需要大量人工介入**：*"These pipelines still require a meaningful amount of human-in-the-loop work."*

**③ 沿用 GLM-5.2 的 RL 策略** ✅

> ✅ **官方原文**：*"It carries over the RL strategies introduced in GLM-5.2, including SAO with compaction."*

**SAO with compaction**（带上下文压缩的 SAO）——这是 GLM-5.2 引入的、用来让收益在**长程任务**而不只是短任务上站得住的机制。⚠️ 博客没展开 SAO 的全称与细节，GLM-5 报告里也没有这个缩写，**这一项目前无法从一手材料还原**。

### 2.3 评测：正文点名的数字

⚠️ **注意**：官方博客的评测大表是**图片**，无法文本核对。以下是正文里明确写出的数字，属于可核对的一手：

| 基准 | GLM-5.2 | **GLM-5.3** | 说明 |
|---|---|---|---|
| **Terminal-Bench 3.0** | 4.6 | **28.3** | 终端任务，6 倍 |
| **DeepSWE v1.1** | 46.2 | **66.9** | 真实仓库改 bug |
| **Agents' Last Exam** | 23.8 | **28.5** | 长程智能体 |
| **Z.ai Code Bench**（内部） | — | **+50%** | 官方主打数字 |

> ⚠️ **Terminal-Bench 3.0 的 4.6 → 28.3 要小心解读**：GLM-5.2 在 **Terminal-Bench 2.1** 上是 81.0（见 [GLM-5.2.md](GLM-5.2.md)），到 3.0 只剩 4.6。**这说明 3.0 是一个难度断崖式提升的新版本**，不是 5.2 突然变弱。跨版本不可比。

### 2.4 Z.ai Code Bench：这次真正有信息量的图

官方新引入了内部基准 **Z.ai Code Bench**，在**两个维度**打分：端到端任务完成率 + 细粒度 checklist 准确率。它同时画了**性能 vs. 输出 token 数**的散点——这才是重点：

| 模型 | 档位 | 分数 | 每任务输出 token |
|---|---|---|---|
| GLM-5.2 | Max | 23.4% | 96K |
| **GLM-5.3** | **Max** | **34.5%** | **~75K** |
| **GLM-5.3** | **High** | **31.4%** | **~50K** |
| Claude Opus 4.8 | 最高档 | 29.5% | 120K |
| Claude Fable 5 | Max | **39.5%** | — |

✅ 以上全部为博客正文明确写出的数字。

> 🔑 **这张表的真正卖点不是分数，是 token 效率**：GLM-5.3 在 High 档用 **50K token 拿 31.4 分**，Opus 4.8 用 **120K token 拿 29.5 分**——**分更高，token 少 2.4 倍**。对按 token 计费的 Agent 场景，这是直接的成本差。
>
> ⚠️ 但也要老实读：**Claude Fable 5 的 39.5% 仍然明显领先**，官方自己把这行放进去了。
>
> ⚠️ **这是私有基准**。官方给的理由是"避免公开测试集污染"（合理），但代价是**第三方无法复现**。

---

## 三、涌现的网络安全能力

这是 GLM-5.3 这次发布最被媒体抓住的一段，也是**最需要冷静读**的一段。

### 3.1 官方的说法：不是设计出来的，是长出来的

> ✅ **官方原文**：*"As we scaled post-training, cyber capability developed faster than we expected."*
>
> ✅ **官方原文**：*"GLM-5.3 did not simply become better at identifying isolated flaws: it began to reason across multiple stages of exploitation, forming coherent plans for complete exploitation chains."*（它不只是更会找孤立的漏洞了，而是开始跨多个利用阶段推理，形成完整利用链的连贯计划。）

他们在训练混合里加入了漏洞发现的数据和环境，预期是"更会找洞"，结果是能力沿着**利用链**往上爬。

### 3.2 三个基准，一个很一致的模式

| 基准 | 测什么 | GLM-5.2 | **GLM-5.3** | Mythos 5 | GPT-5.6 Sol |
|---|---|---|---|---|---|
| **CyberGym** | 白盒源码，能否**发现并触发**漏洞 | 77.2 | **84.5** ⭐ SOTA | 83.8 | 83.6 |
| **ExploitBench** | 对真实漏洞的**深度推理与利用** | 24.4 | **54.4**（翻倍多） | **78.0** | 76.5 |
| **ExploitGym** | 限时内**完成多少个利用任务** | 29 / 39 | **105 / 130** | **181 / 247** | — |

（ExploitGym 两个数字 = 2 小时预算 / 6 小时预算）✅ 全部为博客正文数字。

> 🔑 **模式非常清晰，官方自己也点破了**：
>
> ✅ **官方原文**：*"the further up the exploitation chain a benchmark sits, the larger the gain from GLM-5.2 — and also the wider the remaining gap to the closed frontier. Capability is growing fastest exactly where we are furthest behind."*（基准越往利用链上游，相对 5.2 的涨幅越大——同时和闭源前沿的差距也越大。我们进步最快的地方，恰恰是我们落后最多的地方。）
>
> **翻译成人话**：发现漏洞（第一步）已经追平甚至反超；真正把漏洞变成可用利用（后几步）还差一大截。这个自述是坦诚的。

> ⚠️ **ExploitGym 的方法学需要打问号**。官方脚注写明：2 小时 / 6 小时的预算是把 API 推理时间**按各模型的吞吐率重新缩放**得到的——
>
> ✅ **官方脚注原文**：*"we rescale GLM-5.3's results by 115 TPS, Kimi K3's results by 40 TPS and Qwen3.8 Max's results by 47 TPS"*（TPS = tokens per second，每秒生成 token 数，数据源为 Artificial Analysis）。
>
> **问题**：脚注只列了 GLM-5.3 / Kimi K3 / Qwen3.8 Max 三个模型的缩放系数，**Mythos 5 的缩放系数没有给出**。GLM-5.3 被赋予 115 TPS（三者中最高），意味着同样的墙钟预算里它被允许生成最多 token。⚠️ **这个折算方式对 GLM-5.3 是有利的，且无法从公开信息复核**。

### 3.3 真实世界的漏洞披露：这部分数字很硬

从 GLM-5.2 起，智谱就和国内多个安全团队合作，把模型放到真实代码库上跑。经过专家复核、筛查、去重后：

| 指标 | 数值 |
|---|---|
| **累计发现漏洞** | **2,436** 个 |
| **涉及开源项目** | **269** 个 |
| **已公开披露** | 53 个 |
| **仍在披露流程中（embargo）** | 2,383 个 |
| **中高危以上** | **1,097** 个 |

**严重程度分布** ✅（来自官方披露台账页面的原始数据）：

| 等级 | 数量 | 占比 |
|---|---|---|
| Critical（严重） | 107 | 4.39% |
| High（高危） | 990 | 40.64% |
| Medium（中危） | 1,286 | 52.79% |
| Low（低危） | 53 | 2.18% |

> 📌 **一个口径细节值得记**：博客正文写 *"1,097 medium-to-high severity issues"*，字面是"中到高危"，但 **107 + 990 = 1,097 恰好等于 Critical + High**，而台账页面上这个数字的标签就是 **"CRITICAL & HIGH"**。⚠️ **正文措辞和台账标签不一致，以台账为准：1,097 = 严重 + 高危**。

**漏洞的年龄分布**，这是最有冲击力的一组数字：

> ✅ **官方原文**：*"Findings span 45 years of impact - the oldest flaw was introduced in 1981, and on average a vulnerability lived 26.6 years before discovery."*（跨越 45 年，最老的漏洞引入于 1981 年，平均一个漏洞在被发现前已经存活了 26.6 年。）

引入年份直方图（部分）：1981: 20 个 · 1982: 21 · **1987: 47（峰值）** · 1999: 20 · 2003: 26 · **2006: 39** · 其余年份多为个位数到十几个。

覆盖范围：系统内核、操作系统、浏览器引擎、开源基础设施、Web 应用、网络协议。

> 🔑 **为什么老代码里的洞这么多**：1980~2000 年代的 C 代码大量使用无边界检查的字符串函数，且当时没有 fuzzing、没有静态分析、没有 ASan。这些代码今天还躺在 Linux、libc、各种解析器里。**LLM 的价值在于它能"读懂意图"**——它不像 fuzzer 那样盲撞，能理解"这个函数假设了什么、调用方是否保证了这个假设"。
>
> ✅ 台账地址：[cvd.z.ai](https://cvd.z.ai)（官方称持续更新）

> ⚠️ **权重延后开源的原因就在这里**。官方原话：*"We will release the weights in two weeks after launch, once safety evaluation and hardening are complete."*（发布两周后开源权重，等安全评估和加固完成。）8/14 发布 → 权重预计 **8/28**。**截至本文归档时（2026-08-27），`huggingface.co/zai-org/GLM-5.3` 仍未公开**，已通过 HF API 确认返回空。
>
> 这是**中国实验室第一次因为网络安全能力而延后开源**，本身就是一个值得记录的事件。

---

## 四、slime：让"一直加环境"这件事在工程上成立

GLM-5.3 的后训练全部跑在智谱自研的开源 RL 框架 **slime** 上（训练侧 Megatron，rollout 侧 SGLang）。

> 💡 **小白提示：什么是 rollout？**
> 强化学习的一轮 = 让模型**先去做一遍任务**（生成一串动作/文本，这个过程叫 rollout / 采样）→ 打分 → 用分数更新权重。所以 RL 系统里永远有两套东西在跑：**训练引擎**（更新权重，Megatron）和**推理引擎**（生成轨迹，SGLang）。

### 4.1 设计哲学：环境是"数据生成"，不是"训练循环的改动"

> ✅ **官方原文**：*"Its design keeps training, rollout, and the data buffer on a single dataflow, so math, code, sandboxes, verifiers, and long-horizon agentic environments plug in as data generation rather than as changes to the training loop."*

**这句话是 slime 的全部要义**：数学题、代码沙盒、验证器、长程智能体环境——全都被抽象成"往 data buffer 里灌数据的东西"。加一个新环境不需要动训练代码。

> ✅ **官方原文**：*"That is what let us keep adding environments through GLM-5.2 and GLM-5.3 without rebuilding the training stack each time."*

### 4.2 算法侧的新能力

| 能力 | 是什么 |
|---|---|
| **top-p mask** | 采样时把低概率尾部屏蔽，控制探索范围 |
| **top-k / 全词表 OPD** | OPD = Off-Policy Distillation（离策略蒸馏），用更强的"教师模型"的分布来监督。全词表版本用完整词表分布而非只用 top-k，信号更密 |
| **R3-style 配置** | ⚠️ 博客未展开，无法从一手材料还原 |
| **训推数值全对齐** | 让训练路径和 rollout 路径的数值计算完全一致 |

**最后一项给了硬数字** ✅：

> ✅ **官方原文**：*"the average difference in log probabilities (logprob) was controlled at the 1e-7 level, representing a reduction of more than 99.99% compared with previous setups."*

> 🔑 **"训推一致性"为什么是 RL 的头号工程问题**：
>
> 训练引擎（Megatron）和推理引擎（SGLang）是**两套独立实现**——算子不同、并行策略不同、精度不同。同一份权重、同一个输入，两边算出的概率会有微小差异。
>
> ```
> rollout 时：SGLang 认为 "def" 这个 token 的概率是 0.312
> 训练时  ：Megatron 认为 "def" 这个 token 的概率是 0.309
> ```
>
> RL 的更新公式里有一项是 `新策略概率 / 采样时概率`。这个比值本该反映"策略更新了多少"，**但现在它混进了"两个引擎实现不一样"的噪声**。噪声大到一定程度，梯度方向就是错的，训练会崩。
>
> 把 logprob 差异压到 **1e-7**，等于把这个噪声源基本消掉了。⚠️ "减少 99.99%" 是相对智谱自己之前的设置，**不是行业基线**。

### 4.3 系统侧：三项优化，端到端吞吐 2.3×

| 优化 | 做法 | 解决什么 |
|---|---|---|
| **本地存储做缓存层** | 把模型状态和数据分层放到本地盘，而不是全挤在 host 内存 | 内存不够 |
| **多教师 OPD 的动态切换 + 预取** | 几个教师模型轮流用，靠预取隐藏切换开销，**不用为每个教师常驻一个推理服务** | 多教师蒸馏的资源成本 |
| **router 与 slime 的联合调度 + 负载均衡** | rollout 请求长度差异极大（有的几百 token，有的几十万），做长度感知的调度 | 长尾请求把卡闲置 |
| **workload-aware 启发式** | 从每个环境的特征自动推导 prefill/decode 资源配比、并发数等吞吐关键参数 | 手调参数不 scale |

> ✅ **官方原文**：*"for long-horizon coding RL tasks, these system-level optimizations improved end-to-end RL training throughput by more than 2.3×"*
>
> ⚠️ 限定条件很明确：**长程编程 RL 任务**、**端到端训练吞吐**、**相对自己的基线**。不要当成通用加速比。

---

## 五、GLM-5.3-Flash：这一代真正的架构变化

> **从这里开始换模型了**。以下全部是 GLM-5.3-Flash，和上面的 GLM-5.3 没有任何权重关系。

### 5.1 设计目标：为"极低成本推理"重新设计

官方把对比锚点放在 GLM-4.5 而不是 5.2 上，因为它想强调"同等总参数下，激活和层数都砍半"：

> ✅ **官方原文**：*"Despite a similar total parameter count (320B vs. 355B), it nearly halves both the activated parameter count (18B vs. 32B) and the number of layers (45 vs. 92)."*

| | GLM-4.5 | **GLM-5.3-Flash** | 变化 |
|---|---|---|---|
| 总参数 | 355B | **320B** | ≈ 持平 |
| 激活参数 | 32B | **18B** | **–44%** |
| 层数 | 92 | **45** | **–51%** |

> 🔑 **"层数砍半"比"激活砍半"更值得注意**。激活参数少 → 每个 token 算得少；**层数少 → 推理时的串行深度短**。自回归解码是一层一层串着走的，45 层意味着每生成一个 token 只需要走 45 轮，延迟直接减半。这是 Flash 能做到低延迟的结构性原因，而不只是省算力。

### 5.2 骨干配置全表（来自 `config.json`，一手 ✅）

> 📌 以下每一项都可以在 [huggingface.co/zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) 的 `config.json` 里逐字核对。

**顶层**

| 配置项 | 值 |
|---|---|
| architectures | `Glm5NextForConditionalGeneration` |
| model_type | `glm5_next` |
| 权重实测总参 | **321,323,031,390**（≈321B；FP8 权重 314B + BF16 6.9B + FP32 0.3M） |
| 发布精度 | FP8（`e4m3`，block size 128×128）+ 单独的 BF16 版本 |
| 许可 | **MIT** |

**文本主干（`glm5_next_text`）**

| 配置项 | 值 | 说明 |
|---|---|---|
| 层数 | **45** | 其中 34 层 KDA + 11 层 DSA |
| Hidden Dim | 4096 | 对比 GLM-5.2 的 6144，窄了 1/3 |
| Dense 中间维 | 12288 | |
| **MoE 中间维** | **2048** | |
| **路由专家数** | **288** | GLM-5.2 是 256 |
| 共享专家 | 1 | |
| **每 token 激活专家** | **8** | |
| Dense 层（不走 MoE） | **前 3 层** | 和 GLM-5 系列一贯做法一致 |
| 路由打分函数 | sigmoid + `noaux_tc` | 无辅助损失的负载均衡 |
| 路由缩放系数 | 2.5 | |
| 词表 | 154880 | 与 GLM-5 系列一致 |
| 最大位置 | **1,048,576（1M）** | |
| MTP 层 | 1 | 投机解码用 |
| 激活函数 | SiLU + SwiGLU（clip 上限 10.0） | |

**mHC 相关**

| 配置项 | 值 |
|---|---|
| `mhc` | **true** |
| `hc_mult` | 4 |
| `hc_sinkhorn_iters` | 20 |

**视觉塔（`glm5_next_vision`）**

| 配置项 | 值 |
|---|---|
| 层数 | 24 |
| Hidden Dim | 1024 |
| 注意力头 | 16 |
| 中间维 | 4096 |
| 图像尺寸 / patch | 448 / 14 |
| 时间 patch（视频） | 2 |
| 空间 merge | 2 |
| 输出维 → 主干 | 4096 |
| 投影中间维 | 10240 |

### 5.3 混合注意力：3 层 KDA + 1 层 DSA

**官方的分工说明** ✅：

> ✅ **官方原文**：*"Linear attention captures local dependencies through state modeling, while sparse attention retrieves relevant global context through a lightweight indexer."*（线性注意力通过状态建模捕获局部依赖，稀疏注意力通过轻量索引器召回全局上下文。）

**具体排布**（`config.json` 的 `layer_types` / `kda_layers` / `full_attn_layers` 三处互相印证 ✅）：

```
层号  0    1    2    3     4    5    6    7    …   40   41   42   43   44
     KDA  KDA  KDA  DSA   KDA  KDA  KDA  DSA   …  KDA  KDA  KDA  DSA  KDA
     └────── block 1 ──┘  └────── block 2 ──┘         └── block 11 ──┘  ↑
                                                                  多出的第 45 层

kda_layers      = 0,1,2, 4,5,6, 8,9,10, … 44   （34 层）
full_attn_layers= 3, 7, 11, 15, 19, …, 43      （11 层，间隔 4）
```

**顺序是先 KDA 后 DSA**，DSA 落在每个 block 的第 4 位。

> 📌 **和 Kimi K3 的一处实质差异**：K3 是 `3 KDA + 1 Gated MLA` 循环之后，**在主干末尾额外补一层全局注意力**，保证最后一层能看全局（见 [Kimi-K3.md](Kimi-K3.md)）。GLM-5.3-Flash 的第 45 层反而是 KDA，**最后一个全局层停在第 43 层**。
>
> ⚠️ 官方没有解释这个选择。⚠️ 我的推断：Flash 的全局层是 DSA（本身就带 top-k 检索），可能不需要 K3 那种"末层兜底"；但这是推断，**无一手依据**。

### 5.4 KDA 到底是什么结构

⚠️ **重要说明**：**官方博客和模型卡从头到尾只说 "linear attention"，从未点名 KDA**。"KDA" 这个名字来自两处一手证据：

1. ✅ `config.json` 里的字段名就叫 **`kda_layers`**
2. ✅ SGLang 官方文档在描述 GLM-5.3-Flash 部署时明确写作 **KDA linear attention**

**KDA = Kimi Delta Attention（Kimi 增量注意力）**，出自 Kimi Linear 论文（arXiv:2510.26692）。它是 **GDN（Gated DeltaNet，门控增量网络）** 的改进版。

> 💡 **小白提示：线性注意力在干什么？**
> 标准注意力要把当前 token 和**前面每一个** token 都比一遍，长度 L 就是 L² 的开销，而且 KV cache 要存下所有历史。
> 线性注意力换了个思路：**把所有历史压缩进一个固定大小的矩阵状态 S**，来一个新 token 就更新一次 S。开销 O(L)，显存恒定。代价是**会遗忘**——压缩必然有损。
> 详见 [大模型架构入门/08-高效注意力.md](大模型架构入门/08-高效注意力.md) §8.5。

**KDA 的核心递推式**：

$$S_t = \left(I - \beta_t k_t k_t^\top\right)\,\mathrm{diag}(\alpha_t)\, S_{t-1} + \beta_t k_t v_t^\top$$

拆开读：

| 部分 | 名字 | 作用 |
|---|---|---|
| $\mathrm{diag}(\alpha_t) S_{t-1}$ | **逐通道遗忘门** | 先让旧状态按**每个维度各自的速率**衰减 |
| $(I - \beta_t k_t k_t^\top)$ | **delta rule 的"擦除"** | 把 $k_t$ 方向上的旧内容**减掉** |
| $+\ \beta_t k_t v_t^\top$ | **"写入"** | 再把新内容写进去 |

> 🔑 **KDA 相对 GDN 的唯一但关键的改进：门从"标量"变成"向量"**。
> - **GDN**：每个 head 一个标量遗忘门——整个 head 的记忆**同步**衰减
> - **KDA**：$\alpha_t \in (0,1)^{d_k}$，**每个特征维度一个门**——有的通道慢衰减存长期事实，有的快衰减当工作记忆
>
> Qwen3-Next 用的是 GDN 式的标量门，Kimi Linear / GLM-5.3-Flash 用的是逐通道门。

**Flash 的 KDA 层具体参数**（`linear_attn_config`，一手 ✅）：

| 配置项 | 值 | 含义 |
|---|---|---|
| `num_heads` | **64** | 64 头 × 128 维 = 8192，是 hidden(4096) 的 **2 倍** |
| `head_dim` | **128** | 每头的状态矩阵 S 是 128×128 |
| `short_conv_kernel_size` | **4** | Q/K/V 各过一个 kernel=4 的**因果一维卷积** |
| `gate_lower_bound` | **–5.0** | **衰减门的下界** |

> 🔑 **`gate_lower_bound = -5.0` 是最值得记的一个数**。门控在 log 空间被夹在 –5 以上，即每步衰减率不低于 $e^{-5} \approx 0.0067$。**目的是防止某个通道把历史一次性清空**——线性注意力的状态一旦被冲掉就永远找不回来了。这个下界机制来自 Kimi K3，GLM 原样沿用了同一个数值。

**用 vLLM 源码交叉验证结构** ✅（本地：[推理框架/源码/vLLM/vllm/models/kimi_k3/amd/kda.py](推理框架/源码/vLLM/vllm/models/kimi_k3/amd/kda.py)，类 `KimiK3DeltaAttention(GatedDeltaNetAttention)`）：

```
hidden_states
   │
   └─ in_proj_qkvgfab  ← 一个融合投影，一次算出 6 样东西：
        ├─ mixed_qkv (3×projection_size) ─→ conv1d(kernel=4) ─→ q, k, v
        ├─ g_proj_states (projection_size) ────────────→ g2  【输出门】
        ├─ f_a (head_dim=128) ─→ f_b_proj ────────────→ g1  【逐通道遗忘门，低秩】
        └─ beta (num_heads) ──────────────────────────→ β   【每头一个标量，delta 写入强度】
   │
   ├─ dt_bias / A_log  ← 门控的可学习偏置与衰减基（Mamba 系传统）
   │
   ├─ KDA 递推核（chunkwise 并行）
   │
   ├─ o_norm = FusedRMSNormGated(head_dim, activation="sigmoid")  【门控 RMSNorm 输出】
   │
   └─ o_proj
```

> 📌 **注意 `f_a` 的维度是 128（= head_dim），再由 `f_b_proj` 升回 8192**。这正是 Kimi Linear 论文里的**低秩门控分解** $\mathbb{R}^{d} \to \mathbb{R}^{d_h} \to \mathbb{R}^{n_h d_h}$——如果每个通道的门都独立参数化，光门控就要 4096×8192 的矩阵，低秩分解把它压到 1/64。
>
> 📌 **`beta` 是每头一个标量，`g1` 是每通道一个值**——两个门的粒度不同，别搞混：**β 控制"这次写多少"，α 控制"旧的忘多少"**。

### 5.5 一个容易漏掉的设计：全局层不加位置编码

`config.json` 里有一行：**`"mla_use_nope": true`**，且没有 `qk_rope_head_dim`。

**NoPE = No Positional Encoding（不加位置编码）**。也就是说，**那 11 层 DSA 全局注意力里没有 RoPE**。

> 🔑 **为什么敢这么做**：KDA 的逐通道衰减 $\alpha$ **本身就是一种学出来的位置衰减**——越久远的 token 被衰减得越多，模型天然知道"谁离我近"。既然 34 层 KDA 已经把位置信息注入了残差流，全局层就不必再叠一遍 RoPE。
>
> ✅ 这是 Kimi Linear 论文的原始设计，GLM 完整沿用。
>
> 📌 **但索引器仍然带 RoPE**：`indexer_rope_interleave: true`。⚠️ 我的理解：索引器要判断"哪些历史 token 相关"，这个判断本身依赖相对位置，所以它需要显式位置信号；而主注意力拿到的是已经带位置信息的表示，可以不加。⚠️ 官方未解释。

### 5.6 IndexPool：这一代唯一的原创增量

> ✅ **官方原文**：*"To further reduce the latency and memory overhead of the indexer at a 1M-token context length, we introduce IndexPool, which compresses four indexer key vectors into one through weighted pooling."*（为进一步降低 1M 上下文下索引器的延迟和显存开销，我们引入 IndexPool，通过加权池化把 4 个索引器 key 向量压成 1 个。）

**要理解 IndexPool，得先明白 DSA 的老问题**：

```
DSA（DeepSeek Sparse Attention）分两步：
  ① 轻量索引器给每个历史 token 打分  ← 每个 query 要和【所有】历史比一遍 → 仍是 O(L²)
  ② 只对 top-2048 个 token 算真正的注意力  ← 这步是 O(L·k)，便宜

主注意力省下来了，但【索引器本身还是平方复杂度】。
上下文 200K 时索引器占 prefill 时间的 81%（见 GLM-5.2.md）。
```

**GLM 家族解决这个问题的两次尝试**：

| 代次 | 方案 | 思路 | 压缩的对象 |
|---|---|---|---|
| GLM-5.2 | **IndexShare** | 一层算索引，**后面 3 层复用同一批 token 选择** | **层间**冗余 |
| **GLM-5.3-Flash** | **IndexPool** | 4 个索引器 key 向量**加权池化成 1 个** | **序列长度**方向 |

> 🔑 **两者不是替代关系，压缩的维度不一样**：IndexShare 省的是"算几次索引"，IndexPool 省的是"每次索引要比多少个向量"。
>
> `config.json` 印证 ✅：`index_kpool: 4`、`index_kpool_compress: true`、`index_kpool_always_select_tail: true`。
>
> 📌 **`always_select_tail` 这个开关很重要**：4 个 token 压成 1 个会丢失细粒度，但**最近的 token 不能丢**——它们是当前推理最需要的。所以尾部（最新的那些）**始终原样选中，不参与池化**。这和 [08-高效注意力.md](大模型架构入门/08-高效注意力.md) 里讲的 attention sink 是同一类"必备补丁"思路：**稀疏方案总要给某些位置开后门**。

**索引器完整配置** ✅：

| 配置项 | 值 |
|---|---|
| `index_n_heads` | 32 |
| `index_head_dim` | 128 |
| `index_topk` | **2048** |
| `index_kpool` | **4** |
| `index_kpool_compress` | true |
| `index_kpool_always_select_tail` | true |
| `index_share_for_mtp_iteration` | true（MTP 迭代间复用索引） |
| `indexer_types` | 45 层全部 `"full"` |

### 5.7 mHC：残差流上的改动

> ✅ **官方原文**：*"It also adopts Manifold-Constrained Hyper-Connections (mHC) to further improve scaling efficiency."*

**mHC = Manifold-Constrained Hyper-Connections（流形约束超连接）**。

> 💡 **小白提示**：普通 Transformer 每层是 `x_out = x_in + f(x_in)`，一条残差流从头贯到尾（见 [03-残差流.md](大模型架构入门/03-残差流.md)）。**超连接（Hyper-Connections）** 把这一条流扩成**多条并行的流**，层与层之间用一个可学习的矩阵决定"怎么混合"。好处是信息通路更多，训得更深更稳。
>
> `config.json` 里 **`hc_mult: 4`** 说明扩成了 **4 条流**；**`hc_sinkhorn_iters: 20`** 说明混合矩阵用 **Sinkhorn 迭代 20 步**归一化——Sinkhorn 是把一个矩阵反复行归一化、列归一化，逼近成**双随机矩阵**（每行每列都和为 1）的算法。**这就是"流形约束"的含义：混合权重被约束在双随机矩阵这个流形上，保证多条流之间不会有一条把其他的吞掉**。
>
> ⚠️ 官方只有一句话，**没有给 mHC 的消融实验或收益数字**。这是 Flash 架构里最缺一手材料的一项。

### 5.8 效率收益：官方给的对比方法与数字

> ✅ **官方原文**：*"For a fair comparison among different scales, we calculate the attention compute per head per layer and average KV cache size per layer (BF16). Compared with GLM-5.3, GLM-5.3-Flash reduces the attention compute and KV cache size by factors of 3.0x and 4.4x."*

| 对比项 | 相对 GLM-5.3 |
|---|---|
| 注意力计算量 | **↓ 3.01×** |
| KV cache 大小 | **↓ 4.44×** |

> 📌 **官方的自我批评值得原样保留** ✅：
>
> *"GLM-5.3-Flash has the lowest attention compute among all models compared. The KV cache size is still slightly larger than Kimi-K3 and DeepSeek-V4-Flash, leaving further room for improvement."*
>
> **注意力计算量是全场最低，但 KV cache 仍略大于 Kimi K3 和 DeepSeek-V4-Flash。** 一个厂商在自己的发布博客里承认某项指标输给对手，这个坦诚度值得记一笔。
>
> ⚠️ **口径提醒**：这是 **per head per layer** 的归一化对比，**不是端到端加速比**，也不是整机显存占用。跨模型比较时必须带上这个限定。

### 5.9 基座模型的评测

> ✅ **官方原文**：*"GLM-5.3-Flash-Base outperforms GLM-4.5-Base overall and remains competitive with GLM-5-Base across most benchmarks."*
>
> ✅ 脚注：*"Results for DeepSeek-V4-Flash-Base were evaluated using our internal evaluation framework to control for implementation differences."*

⚠️ **具体分数是图片，无法文本核对**。可核对的只有这句定性结论：**18B 激活的 Flash 基座，总体优于 32B 激活的 GLM-4.5 基座，在多数基准上与 40B 激活的 GLM-5 基座可比**。

> 🔑 **如果这个结论成立，它才是 Flash 最重要的一句话**：架构改进（KDA+DSA+mHC）+ 30T 多模态语料，让**激活参数减半的基座追平了上一代旗舰基座**。所有的成本优势最终都是从这里来的。⚠️ 但没有可核对的数字支撑。

---

## 六、原生多模态：视觉进入编程回路

这是 GLM-5 系列第一个原生多模态模型（不是外挂一个视觉编码器，而是预训练阶段就是多模态的）。官方对"为什么编程模型需要视觉"的论证很具体：

> ✅ **官方原文**：*"For tasks such as frontend development, game development, and 3D simulation, the final output is not code alone, but an interface, an interaction, or a world experienced by the user. Many failures only surface through rendering, interaction, or playtesting."*（前端、游戏、3D 仿真这类任务，最终产物不是代码本身，而是界面、交互、或用户体验到的世界。很多失败只有在渲染、交互、试玩时才暴露出来。）

**核心主张**：

> ✅ **官方原文**：*"Vision therefore needs to be natively integrated into the model, enabling it to decide when to observe and use visual feedback to guide subsequent actions."*（视觉需要原生集成进模型，让它自己决定**何时去看**，并用视觉反馈引导后续动作。）

**训练方法** ✅：

| 手段 | 内容 |
|---|---|
| 数据合成流水线 | 面向视觉编程，重点是 **self-visual judgment（自我视觉判断）** 与 test-time improvement |
| 轨迹形态 | 要求模型与环境交互 → **检查自己的输出** → 迭代改进 |
| 前端方向 | 用**环境反馈的强化学习**；再用基于真实用户流程的 agent 验证来强化 GUI 判断 |
| 验证范围的扩展 | *"beyond functional correctness to the rendered and interactive product"*（从功能正确性扩展到渲染结果和可交互产物） |

官方给的演示是一组前后对比图：**"Initial Version with Layout Issues"（初版有布局问题）→ "After Visual Self-Verification"（视觉自检后）**。

**超出编程的部分**：官方还主张视觉能力让模型进入"知识工作"——文档、表格、演示稿、仪表盘、界面、会议材料。

> ✅ **官方原文**：*"Rather than requiring users to explicitly translate their working environment into textual instructions, the model can directly interpret the artifacts associated with a task."*（不必让用户把工作环境翻译成文字指令，模型可以直接读任务相关的产物。）

**多模态评测覆盖的基准**（从脚注可见 ✅）：BabyVision、Treasury Bulletin PDF 语料（不给内嵌文本，纯靠看）、MVBench、MMVU 等。视频输入策略：原生支持视频的模型直接喂原视频，不支持的按 **1 fps 抽帧**、超限则均匀采样。

⚠️ 多模态评测的具体分数同样是图片，无法核对。

---

## 七、国产芯片上的大规模部署

这是 Flash 发布里最有产业信号意义的一段。

> ✅ **官方原文**：*"Over the past week, we have served GLM-5.3-Flash on a large-scale cluster of Chinese AI chips."*
>
> ✅ 发布前它以匿名代号 **ox-alpha** 在 OpenCode 和 OpenRouter 上做公测，*"It quickly became the most popular model of the week — with all of this traffic served on Chinese AI chips."*（一周内成为最受欢迎的模型——**全部流量都跑在国产芯片上**。）

> 🔑 **匿名公测这个做法值得记**：用户在不知道是谁家模型、也不知道跑在什么硬件上的情况下选择了它。**这比任何 benchmark 都更能说明"国产芯片推理是否已经可用"**。

### 7.1 单卡层面：为架构定制的推理引擎

> ✅ **官方原文**：*"To overcome the relatively limited compute and memory capacity of individual chips, we built a dedicated inference engine for this architecture on top of SGLang."*

**优化手段清单** ✅：

| 技术 | 作用 |
|---|---|
| **Linear Attention 与 LM head 的节点内张量并行** | 把 KDA 层和最后的输出层切到多卡 |
| **ReplaySSM** | ⚠️ 官方未展开。⚠️ 我的推断：与线性注意力的循环状态重放有关——SGLang 的命令行里确实有 `--enable-linear-replayssm-spec` / `--linear-replayssm-cache-len` 这组开关，看名字是**在投机解码时重放/回滚 SSM 状态**。线性注意力的状态是**串行更新**的，投机解码一旦回滚就必须把状态也退回去，这是纯线性层模型做投机解码的固有难点 |
| **W8A8 量化** | 权重 8bit、激活 8bit |
| **INT8 / FP8 / BF16 混合 cache 量化** | 不同的 cache 用不同精度 |
| **Layer Split** | 层切分 |

> ✅ **一个有意思的自指细节**：*"this effort was accelerated by our GLM-5.3-powered infrastructure agent, which assisted engineers in developing and optimizing kernels, diagnosing performance bottlenecks, and improving the serving stack — creating a feedback loop in which the model helped optimize the system serving the model itself."*
>
> **用 GLM-5.3 写 kernel、调性能，来优化跑 GLM-5.3-Flash 的系统。** 这个闭环第一次被一家厂商写进发布博客里。

> 🔑 **为什么国产芯片的瓶颈是"显存和带宽"而不是算力**：
>
> ✅ 官方点明：*"These chips are primarily constrained by memory capacity and bandwidth, especially when supporting context lengths of up to one million tokens."*
>
> 所以优化方向是 **compute-for-bandwidth（用算力换带宽）** 和 **communication-for-bandwidth（用通信换带宽）**——宁可多算几次、多传几次，也不要多读显存。这和 [06-KV-cache.md](大模型架构入门/06-KV-cache.md) 里"decode 是访存受限"的主线完全一致，**也正是为什么这颗模型要选线性注意力**：KV cache 小 4.44×，等于直接把最紧的那根弦松了。

### 7.2 集群层面：EPD 分离

> ✅ **官方原文**：*"our production-grade Encode–Prefill–Decode (EPD) disaggregated architecture separates multimodal encoding, prompt prefill, and token-by-token decoding into independently scheduled and scalable worker pools, enabling efficient and reliable serving across tens of thousands of domestically developed accelerators."*

**EPD = Encode–Prefill–Decode（编码–预填充–解码）三段分离**：

```
传统 PD 分离（两段）：      Prefill 池  →  Decode 池
GLM-5.3-Flash 的 EPD：Encode 池 → Prefill 池 → Decode 池
                        ↑
                   多模态编码（图/视频过视觉塔）单独成池
```

> 🔑 **为什么多模态一定要拆出第三段**：视觉编码是**纯计算密集**的（过 24 层 ViT），prefill 是**计算密集但要 KV**，decode 是**访存密集**。三者的硬件需求完全不同。挤在一起的话，一张图片的编码会把 decode 的卡堵住。**拆开后每一段可以独立扩缩容**。
>
> ✅ 规模口径：*"tens of thousands of domestically developed accelerators"*（数万张国产加速卡）。

### 7.3 结果

> ✅ **官方原文**：*"Compared with our initial baseline on the same hardware, we achieved a 3× improvement in end-to-end serving performance, reaching hardware efficiency and per-token cost comparable to mainstream NVIDIA GPUs."*

| 指标 | 数值 |
|---|---|
| 端到端服务性能 | **3×**（相对同硬件上的初始基线） |
| 单 token 成本 | **达到主流 NVIDIA GPU 水平** |

> ⚠️ **口径必须看清**：3× 是**相对自己在同一批国产硬件上的初始基线**，不是相对 NVIDIA。真正的对标声明是第二句——**单 token 成本追平主流 N 卡**。⚠️ 这个说法没有给出具体的卡型、功耗、集群规模，**无法独立验证**。

---

## 八、GLM-5.3-Flash 的评测

⚠️ 同样地，官方评测大表是图片。以下是博客正文明确写出的数字 ✅：

| 基准 | GLM-5.2 | **GLM-5.3-Flash** | Claude Opus 4.8 |
|---|---|---|---|
| **DeepSWE v1.1** | 46.2 | **63.4** | — |
| **AutomationBench** | 26.2 | **48.8** | — |
| **Z.ai Code Bench v1.0**（max effort） | — | **29.0** | **29.5** |
| **AA Intelligence Index v4.1.1** | — | **57** | — |

> ✅ **成本-智能的官方主张**：*"GLM-5.3-Flash pushes the Pareto frontier of the Artificial Analysis Intelligence Index v4.1.1, scoring 57 at just $0.045 per task (discounted) — a level of intelligence previously only available at roughly 10× the cost."*（在 AA 智能指数 v4.1.1 上拿 57 分，每任务成本仅 $0.045（折后）——这个智能水平此前大约要 10 倍的价钱。）

> 📌 **Z.ai Code Bench 上的 29.0 vs 29.5 这一行，是这次发布的"题眼"**。同一个内部基准上：
> - GLM-5.3（744B/40B，$4.4/M 输出）max 档 **34.5**
> - GLM-5.3-Flash（320B/18B，$0.50/M 输出）max 档 **29.0**
> - Claude Opus 4.8 最高档 **29.5**
>
> **Flash 用 GLM-5.3 约 1/9 的输出价格，拿到了它 84% 的分数，并追平 Opus 4.8。**
>
> ⚠️ 但注意 GLM-5.3 是 Z.ai Code Bench **v?** 、Flash 是 **v1.0**，博客对前者未标版本号，**严格说这两组数字是否同版本可比，无法确认**。

---

## 九、开源、定价与生态

### 9.1 开源状态

| 模型 | 权重 | 状态 |
|---|---|---|
| **GLM-5.3-Flash** | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)（FP8，320B-A18B） | ✅ **已开源，MIT** |
| **GLM-5.3-Flash-BF16** | [zai-org/GLM-5.3-Flash-BF16](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) | ✅ 已开源 |
| **GLM-5.3** | `zai-org/GLM-5.3` | ⚠️ **截至 2026-08-27 仍未公开**（HF API 返回空）。官方承诺"发布后两周"，即约 8/28 |

✅ HF 仓库元数据：GLM-5.3-Flash 创建于 **2026-08-25**，最后更新 **2026-08-26**，实测权重总量 **321,323,031,390** 参数。

### 9.2 API 定价（官方 [docs.z.ai 定价页](https://docs.z.ai/guides/overview/pricing)，单位 USD / 1M tokens）✅

| 模型 | 输入 | 缓存输入 | 缓存存储 | 输出 |
|---|---|---|---|---|
| **GLM-5.3-Flash** | $0.15 → **$0.075** | $0.03 → **$0.015** | 限时免费 | $0.50 → **$0.25** |
| **GLM-5.3** | $1.4 | $0.26 | 限时免费 | $4.4 |
| GLM-5.2 | $1.4 | $0.26 | 限时免费 | $4.4 |

> ✅ GLM-5.3-Flash 五折优惠至 **2026-09-09 24:00（UTC+8）**，箭头前为原价。
>
> 📌 **原价口径下 Flash 是 GLM-5.3 的 1/9（输入）到 1/8.8（输出）**；折后接近 **1/18**。官方博客说的 "one-tenth the price" 对应的是原价口径 ✅。

### 9.3 思考档位（两个模型一致）✅

| 参数 | 取值 | 默认 |
|---|---|---|
| `thinking.type` | **只接受 `enabled`** | enabled |
| `reasoning_effort` | `low` / `high` / `max` | **max** |

> ⚠️ **破坏性变更，迁移必看**：`thinking.type: "disabled"` **不再支持**。官方原文：*"If your application currently uses `thinking.type: "disabled"`, change it to `enabled` and set `reasoning_effort` to `low`. Otherwise, the request will fail."*（否则请求会直接失败。）
>
> 三档含义：low = 轻量推理，high = 增强推理，max = 深度推理。**编程任务官方推荐 max**。

### 9.4 部署框架 ✅

GLM-5.3-Flash 已支持：**SGLang**（[cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash)）、**vLLM**（[recipes](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)）、**TokenSpeed**、**KTransformers**。

⚠️ SGLang 官方文档注明：这个混合架构的 **PD 分离（prefill/decode 分离）需要同时搬运分页的 DSA KV 和 KDA 的循环状态**，目前仅在 4×GB300 上用 dummy weights 做过机械验证，**未做压测与精度测试，应视为预览特性**。

---

## 十、这一代给架构演进留下的三个判断

**① "后训练还没到头"这件事被证明了一次**

GLM-5.3 是一个干净的对照实验：**基座完全冻结，只加后训练，编程能力 +50%，并且长出了预期外的网络安全能力**。⚠️ 但要注意，这个结论的前提是**基座本身足够强且未被榨干**——它不能推广成"任何模型加后训练都能 +50%"。

**② 线性注意力的地位，从"廉价配角"升到了"主力层"**

对照 [08-高效注意力.md](大模型架构入门/08-高效注意力.md) 里记录的判断——*"这一代线性注意力的地位是：作为混合方案里的廉价层可以，作为主力不行"*，以及 MiniMax 在 M2 里从线性注意力**退回**全注意力的反例：

| 模型 | 线性层占比 | 全局层 |
|---|---|---|
| MiniMax M2 | **0%**（退回了） | 全注意力 |
| Kimi K3 | 69/93 ≈ 74% | Gated MLA |
| **GLM-5.3-Flash** | **34/45 ≈ 76%** | **DSA（top-2048 + IndexPool）** |

**这个判断需要更新**：GLM 是**第一个把线性注意力和稀疏注意力叠在一起**的开源前沿模型——之前几家都是二选一。**混合的两条腿都变便宜了**（线性层省 KV，稀疏层省计算），而不是像 K3 那样"便宜的线性层 + 昂贵的全注意力层"。

⚠️ 但仍然**没有任何一家纯用线性注意力**。76% 的比例和 K3 的 74% 几乎一样——**3:1 看起来是这一代的共识配比**。

**③ 成本竞争的战场从"参数量"转移到了"注意力结构 + 硬件协同"**

官方自己的总结值得原样保留 ✅：

> *"This is not the result of any single trick, but of three layers working together: an architecture that delivers stronger capability from less compute, a richer multimodal pre-training corpus, and infrastructure co-designed with inference hardware."*（这不是任何单一技巧的结果，而是三层共同作用：用更少算力产出更强能力的架构、更丰富的多模态预训练语料、以及与推理硬件协同设计的基础设施。）

> ✅ 并且他们明说了下一步：*"We are now scaling this recipe to larger models."*（我们正在把这套配方放大到更大的模型上。）**Flash 是试验田，下一个旗舰会是这套架构的放大版。**

---

## 十一、可靠性说明

| 内容 | 来源 | 可信度 |
|---|---|---|
| **GLM-5.3-Flash 的全部架构参数**（45 层、3:1 排布、KDA 64×128、conv=4、gate 下界 –5.0、288 专家、IndexPool 配置、mHC 参数、NoPE、视觉塔） | **HF `config.json`** | **一手，可逐字复核** |
| KDA 的递推式与实现结构 | Kimi Linear 论文 + **本地 vLLM `kimi_k3/amd/kda.py` 源码** | 一手（论文 + 可运行代码） |
| GLM-5.3 后训练方法、环境合成、网络安全三基准、漏洞台账、slime 优化 | **官方博客正文**（逐段核对） | 一手措辞，但**厂商自述，无论文** |
| GLM-5.3-Flash 的效率倍数、部署栈、国产芯片 3× | 官方博客正文 | 同上 |
| API 定价、思考档位 | 官方文档页 | 一手 |
| **所有评测大表的完整数字** | **缺失（官方全部做成图片）** | — |
| SGLang PD 分离的预览状态 | SGLang 官方 cookbook | 一手 |

**最需要保留怀疑的六处**：

1. ⚠️ **ExploitGym 的时间预算折算，独独缺了对标模型 Mythos 5 的缩放系数**，而 GLM-5.3 拿到的 115 TPS 是列出模型中最高的。这个折算对自己有利且无法复核。
2. ⚠️ **Z.ai Code Bench 是私有基准**。防污染的理由成立，但代价是第三方无法复现；且 GLM-5.3 与 Flash 的分数是否同版本可比，博客未标明。
3. ⚠️ **"3.01× / 4.44×" 是 per-head-per-layer 归一化口径**，不是端到端加速，也不是整机显存。跨模型引用必须带限定。
4. ⚠️ **国产芯片的 3× 是相对自己的初始基线**，"单 token 成本追平主流 N 卡"没有给卡型、功耗、集群规模，无法独立验证。
5. ⚠️ **mHC 只有一句话**，没有消融、没有收益数字。这是 Flash 架构里一手材料最薄的一项。
6. ⚠️ **基座评测（"18B 激活追平 GLM-5-Base"）只有定性结论**，支撑它的表是图片。而这恰恰是整个成本优势叙事的地基。

**另外两处口径不一致，已在正文标注**：

- 漏洞数 **1,097** 在正文被描述为 "medium-to-high"，但台账标签是 **"CRITICAL & HIGH"**，且 107+990 正好等于 1,097。**以台账为准**。
- 官方博客与模型卡**从未点名 KDA**，只说 "linear attention"。KDA 之名来自 `config.json` 的字段名与 SGLang 文档。

---

## 参考来源

- [GLM-5.3: Frontier Coding with Emergent Cyber Capabilities（官方博客，2026-08-14）](https://z.ai/blog/glm-5.3)
- [GLM-5.3-Flash（官方博客，2026-08-26）](https://z.ai/blog/glm-5.3-flash)
- [zai-org/GLM-5.3-Flash (Hugging Face)](https://huggingface.co/zai-org/GLM-5.3-Flash) — 架构一手来源 `config.json`
- [Z.ai Security Disclosure Ledger（漏洞披露台账）](https://cvd.z.ai)
- [GLM-5: from Vibe Coding to Agentic Engineering (arXiv:2602.15763)](https://arxiv.org/abs/2602.15763) — 本地：[papers/GLM-5-...pdf](papers/GLM-5-from-Vibe-Coding-to-Agentic-Engineering-arXiv-2602.15763.pdf)
- [Kimi Linear: An Expressive, Efficient Attention Architecture (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692) — KDA 的原始论文
- [zai-org/GLM-5 (GitHub)](https://github.com/zai-org/GLM-5)
- [Z.ai 开发者文档 — GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3) ｜ [GLM-5.3-Flash](https://docs.z.ai/guides/llm/glm-5.3-flash) ｜ [定价页](https://docs.z.ai/guides/overview/pricing)
- [SGLang Cookbook — GLM-5.3-Flash](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash)
- 本地源码交叉验证：[推理框架/源码/vLLM/vllm/models/kimi_k3/amd/kda.py](推理框架/源码/vLLM/vllm/models/kimi_k3/amd/kda.py)
