# 10 · 拿一份真实技术报告，看一线团队怎么设置 harness

> ⭐⭐ 前面九章讲的都是"原理上会出什么问题"。
> 这一章拿两份 DeepSeek 技术报告当标本，逐条对照：
> ★★★ 一个顶级团队，是【怎么把这些坑一个个填上的】。
>
> - 10.1 ~ 10.7 → **DeepSeek-V3 技术报告**（arXiv:2412.19437，53 页）
> - 10.7b → **DeepSeek-R1 技术报告**（arXiv:2501.12948，86 页）★★ 推理模型的口径有一处根本不同
>
> ★ 选它们的理由很简单：这两份报告的评测设置写得【异常详细】，
> ⟹ 详细到可以当作"该披露什么"的模板来用。

---

## 10.1 第一个发现：他们的分类，就是 03 章那三个原语

✅ 论文 §4.4.1 原文（★ 这段值得逐字读）：

> "we adopt **perplexity-based evaluation** for datasets including HellaSwag, PIQA,
> WinoGrande, RACE-Middle, RACE-High, **MMLU**, MMLU-Redux, MMLU-Pro, MMMLU,
> ARC-Easy, ARC-Challenge, C-Eval, CMMLU, C3, and CCPM, and adopt
> **generation-based evaluation** for TriviaQA, NaturalQuestions, DROP, MATH,
> **GSM8K**, MGSM, HumanEval, MBPP, LiveCodeBench-Base, CRUXEval, BBH, AGIEval,
> CLUEWSC, CMRC, and CMath. In addition, we perform **language-modeling-based
> evaluation** for Pile-test and use **Bits-Per-Byte (BPB)** as the metric to
> **guarantee fair comparison among models using different tokenizers**."

```
   ⟹ 🔑🔑 把这段和 03 章那三个抽象方法并排放：

   ┌──────────────────────────┬───────────────────────────────────┐
   │ DeepSeek 的说法           │ lm-eval-harness 的原语             │
   ├──────────────────────────┼───────────────────────────────────┤
   │ perplexity-based         │ ★★ loglikelihood                   │
   │ generation-based         │ ★★ generate_until                  │
   │ language-modeling-based  │ ★★ loglikelihood_rolling           │
   └──────────────────────────┴───────────────────────────────────┘

   ⟹ ★★★ 一字不差地对上了。
      两个互不相干的团队，各自独立收敛到【同一个三分法】。
      🔑 说明 03 章讲的不是某个框架的实现细节，
         而是【这件事本身就只有这三种做法】。
```

```
    ★ 名词：perplexity（困惑度）
      衡量"模型对一段文本有多意外"的指标。★★ 越低说明模型越觉得这段话自然。
      ⟹ 它和 loglikelihood（对数似然）是同一个东西的两种写法。

    ★ 名词：BPB = Bits-Per-Byte（每字节比特数）
      ★★ 平均编码一个【字节】需要多少比特。
```

★★★ 那个 BPB 的理由值得单独说：

```
    ⚠️ 问题：不同模型的分词器（tokenizer）不一样。
       同一句中文，A 模型切成 12 个 token，B 模型切成 20 个。
       ⟹ 🔑 那"每个 token 的困惑度"就【根本没法比】——
          分母都不是同一个东西。

    ⟹ ★★★ 解法：换成【每字节】。
       字节是物理事实，谁的分词器都改变不了它。
       ✅ 论文原话就是 "to guarantee fair comparison among models
          using different tokenizers"。

    🔑🔑 这是一个非常漂亮的"口径对齐"操作。
       ★★ 记住这个思路：★★★ 当两个数不可比时，
          找一个【双方都无法影响的分母】。
```

---

## 10.2 第二个发现：他们自己承认分数会漂

✅ 论文 §4.4.2（★ 逐字）：

> "We evaluate all these models with **our internal evaluation framework**, and
> ensure that they share the **same evaluation setting**. Note that **due to the
> changes in our evaluation framework over the past months**, the performance of
> DeepSeek-V2-Base exhibits **a slight difference from our previously reported results**."

```
    ⟹ 🔑🔑🔑 读懂这句话：

    ★★★【同一个模型（V2-Base），同一个团队，只是评测框架改了几个月，
       分数就和自己上次公布的不一样了。】

    ⟹ 模型一个权重都没动。动的全是 harness。
```

```
    ★★ 这一句话，把 01 章的中心论点从"我的说法"变成了【一线团队的书面承认】：

       ⟹ 🔑🔑 分数是【模型 + 评测方法】的联合产物，不是模型的固有属性。

    ★★ 而且请注意他们的处理方式——他们做的是【两件事】：
       ① 用同一套框架重新跑了所有对比模型（而不是抄别人报的数）
       ② ★★★ 主动声明"我们自己的历史数字对不上了"

    ⟹ ⚠️ 第 ② 条是绝大多数技术报告【不会做】的。
       看到有人这么写，可信度应该往上调。
```

---

## 10.3 第三个发现：他们给出了噪声下限

✅ 论文 Table 3 表注（★ 逐字）：

> "**Scores with a gap not exceeding 0.3 are considered to be at the same level.**"

```
    ⟹ 🔑🔑 翻译：★★★【差 0.3 分以内，我们认为是平手。】

    ★★ 为什么必须有这么一条？因为评测本身有噪声：
       采样温度、题目顺序、浮点非确定性、few-shot 抽哪几条示例……
       ⟹ 同一个模型跑两遍，分数就可能差零点几。

    ⟹ ★★★ 所以"我们比对手高 0.2 分"这种宣传，
       在 DeepSeek 自己的口径下，★★【等于没差别】。
```

```
    🔑 给你的行动项：
       ★★★ 以后看到任何两个模型的分数对比，先问一句：
          【这个差距，超过噪声了吗？】
       ⟹ 呼应 09 章 Elo 的置信区间——同一个道理的两种表达。
```

---

## 10.4 第四个发现：few-shot 数量是逐题设置的，不是一刀切

✅ 论文 Table 3 的 "# Shots" 列（★ 摘录）：

```
   ┌──────────────────┬──────────┬──────────────────────────────┐
   │ Benchmark        │ # Shots  │ ★ 为什么是这个数（我的解读⚠️）│
   ├──────────────────┼──────────┼──────────────────────────────┤
   │ PIQA (EM)        │ 0-shot   │ 常识判断，不需要格式示范      │
   │ AGIEval (EM)     │ 0-shot   │ 标准化考试，题面自带格式      │
   │ BBH (EM)         │ 3-shot   │ 推理题，要示范推理写法        │
   │ DROP (F1)        │ 3-shot   │ 阅读理解                     │
   │ MMLU (EM)        │ ★ 5-shot │ ★★ 领域惯例，为了可比性      │
   │ MMLU-Redux       │ 5-shot   │ 同上                         │
   │ MMLU-Pro         │ 5-shot   │ 同上                         │
   │ WinoGrande       │ 5-shot   │                              │
   │ TriviaQA         │ 5-shot   │                              │
   │ HellaSwag (EM)   │ 10-shot  │                              │
   │ ARC-Easy         │ ★ 25-shot│ ★★ 这个数字很大，见下        │
   │ ARC-Challenge    │ ★ 25-shot│                              │
   └──────────────────┴──────────┴──────────────────────────────┘
```

```
    ⟹ 🔑 从 0 到 25，跨度极大。★★ 这印证了 04 章的核心论点：
       ★★★ few-shot 数量【不是一个可以随便定的超参数】，
       它是【这个基准的身份证的一部分】。

    ⚠️ ARC 用 25-shot 是沿袭 HuggingFace Open LLM Leaderboard 的历史惯例
       （我的判断，论文未说明理由）。
       ⟹ ★★ 但这恰恰说明问题：★★★ 这些数字是【约定俗成】的，
          不是推导出来的。你必须照抄，否则分数不可比。

    🔑🔑 所以 04 章那句话要再强调一遍：
       ★★★ 报 MMLU 分数不说 shot 数，等于没报。
```

---

## 10.5 第五个发现：chat 模型的评法，和 base 模型完全不是一回事

★★★ 这是本章最重要的一节。

✅ 论文 §5.3.1 "Detailed Evaluation Configurations"（★ 逐字拆解）：

### ① 提示词直接借用别人的框架

> "For standard benchmarks including MMLU, DROP, GPQA, and SimpleQA, we adopt
> the evaluation prompts from the **simple-evals** framework."
> "We utilize the **Zero-Eval prompt format** for MMLU-Redux in a **zero-shot** setting."
> "For other datasets, we follow their **original evaluation protocols with default
> prompts as provided by the dataset creators**."

```
    ⟹ 🔑🔑 注意这里的对比，★★★ 同一份报告里 MMLU 出现了两次，口径不同：

    ┌──────────────┬─────────────────────────┬──────────┐
    │              │ 怎么评                   │ 分数      │
    ├──────────────┼─────────────────────────┼──────────┤
    │ V3-Base      │ perplexity-based, 5-shot│ ✅ 87.1  │
    │ V3 (chat)    │ simple-evals 提示词      │ ✅ 88.5  │
    │              │ ★★ 生成式               │          │
    └──────────────┴─────────────────────────┴──────────┘

    ⟹ ★★★ 这两个 87.1 和 88.5【不能相减说"对齐带来 +1.4"】。
       ⚠️ 因为连测量方式都换了（判别式 → 生成式）。

    🔑 这正是 01 章那个 12.6 分差示例的真实版本：
       ★★ 同一个基准名字底下，藏着完全不同的测法。
```

```
    ★ 名词：simple-evals
      ★★ OpenAI 开源的一套极简评测提示词集合。
      ⟹ 🔑 DeepSeek 直接借用它，是为了【和 OpenAI 报的数可比】。
      ★★★ 这是一个很清醒的工程决策：
         与其自己定一套更"合理"的口径，不如用对手用的那套。
```

### ② 数学题：温度、跑几遍，全部写死

> "For mathematical assessments, **AIME and CNMO 2024 are evaluated with a
> temperature of 0.7, and the results are averaged over 16 runs**, while
> **MATH-500 employs greedy decoding**."

```
   ┌──────────────┬───────────────┬────────────────────────────────┐
   │ AIME / CNMO  │ ★ T=0.7×16 次 │ ★★ 题少（AIME 只有 30 题）    │
   │              │   取平均       │ ⟹ 单次跑方差极大，必须多跑    │
   ├──────────────┼───────────────┼────────────────────────────────┤
   │ MATH-500     │ ★ greedy      │ ★★ 500 题，样本够，直接贪心   │
   │              │   （T=0）      │ ⟹ 可完全复现                  │
   └──────────────┴───────────────┴────────────────────────────────┘

    ★ 名词：greedy decoding（贪心解码）
      每一步都选概率最高的那个 token。★★ 等价于 temperature=0。
      ⟹ 🔑 同样输入必得同样输出 ⟹【可复现】。

    ⟹ 🔑🔑 这条设置的价值在于它【解释了 why】：
       ★★★ 题目越少，越必须多跑取平均；题目够多，就该用贪心保复现。
```

★★ 表注里还有一句总纲（✅ 逐字）：

> "**Benchmarks containing fewer than 1000 samples are tested multiple times
> using varying temperature settings to derive robust final results.**"

```
    ⟹ ★★★【1000 题】是他们划的线。少于这个数就必须多跑。
    ⚠️ 这条规则我在别的报告里很少见到写得这么明确。
```

### ③ 输出长度上限：一个数字定死全场

> "We allow all models to output a **maximum of 8192 tokens** for each benchmark."
> "All models are evaluated in a configuration that **limits the output length to 8K**."

```
    ⟹ 🔑 回看 05 章那个"最常见的坑"：
       ★★★ max_tokens 截断了推理过程，导致答案根本没写出来。

    ⟹ ★★ DeepSeek 的做法是【统一设成 8192，并且写进报告】。
       好处：所有模型同一条件，公平。
       ⚠️ 代价：★★★ 对特别爱长篇推理的模型，8K 可能不够，
          它会被截断，而这【看起来像是它答错了】。

    🔑🔑 所以这个数字必须公布——★★ 它是分数的一部分。
```

### ④ 闭源模型走 API

> "For closed-source models, evaluations are performed through their **respective APIs**."

```
    ⟹ ★★ 直接呼应 03 章那个"闭源 API 陷阱"：
       走 API 就【拿不到 logprobs】，做不了 perplexity-based 评测。

    ⟹ 🔑 这就是为什么 base 模型表（Table 3）里【只有开源模型】，
       而 Claude / GPT-4o 只出现在 chat 模型表（Table 6）里。
       ★★★ 不是他们不想比，是【物理上没法用同一种方法比】。
```

---

## 10.6 第六个发现：SWE-bench 那一行，暴露了 08 章的全部要害

✅ 论文 §5.3.1（★ 逐字）：

> "**SWE-Bench verified is evaluated using the agentless framework** (Xia et al., 2024)."

```
    ★ 名词：Agentless（无智能体框架）
      ★★ 一套【故意不用 agent 循环】的 SWE-bench 方案：
         固定三步——定位文件 → 生成补丁 → 挑选补丁，没有多轮自主决策。
      ⚠️ 以上是我对该方法名的解释；具体实现细节请查该论文原文。
```

```
    ⟹ 🔑🔑🔑 这一行信息量极大。它等于在说：

    ★★★【DeepSeek-V3 报的 42.0% SWE-bench Verified，
       是"V3 + Agentless 脚手架"这套系统的分数，
       不是"V3 这个模型"的分数。】

    ⟹ 换一套更强的 agent 脚手架，同一个 V3 的分数会变。
       ★★ 这正是 08 章 §8.3 那张"完整含义"表格的现实例证。
```

✅ 同表（Table 6）的 SWE Verified (Resolved) 一行：

```
   ┌───────────────┬──────────────┬──────────┬──────────────┐
   │ DeepSeek-V2.5 │ Qwen2.5 72B  │ GPT-4o   │ Claude-3.5-  │
   │               │              │  0513    │ Sonnet-1022  │
   │    22.6       │    23.8      │   38.8   │  ✅ 50.8     │
   ├───────────────┴──────────────┴──────────┴──────────────┤
   │            DeepSeek-V3 = ✅ 42.0                        │
   └────────────────────────────────────────────────────────┘
```

```
    ⚠️ 关键提醒：★★★ 这张表里【所有模型都跑的是 Agentless】。
       ⟹ 所以它是一个【口径统一的横向对比】——这很好。

    ⟹ 但 🔑🔑 它【不等于】各家自己产品的表现。
       ★★ Claude 在这里是 50.8，而 Anthropic 自家用 Claude Code
          那套脚手架报出来的数是另一回事。
       ⟹ ★★★ 两个数字都不假，但它们【回答的是不同问题】：
          ┌────────────────────┬──────────────────────────┐
          │ 统一脚手架下的 42.0 │ 这个【模型】有多强         │
          │ 各自产品报的数字     │ 这套【产品】有多好用       │
          └────────────────────┴──────────────────────────┘
```

---

## 10.7 第七个发现：训练侧也在为"能抽取"服务

★★ 这是我认为最有意思的一条，它把 05 章和训练连起来了。

✅ 论文 §5.2.1 "Rule-Based RM"（★ 逐字）：

> "For questions that can be validated using specific rules, we adopt a rule-based
> reward system... certain math problems have deterministic results, and we
> **require the model to provide the final answer within a designated format
> (e.g., in a box)**, allowing us to **apply rules to verify the correctness**."

```
    ★ "in a box" 指数学界通用的 LaTeX 写法 \boxed{42}。

    ⟹ 🔑🔑🔑 读懂这层含义：

    ┌──────────────────────────────────────────────────────────┐
    │  ★ 05 章：评测端写正则去抠答案，经常抠不准。              │
    │  ★★ DeepSeek 的做法：★★★【在训练阶段就把格式教进去】。  │
    │     强化学习的奖励规则本身就要求"答案必须写在 \boxed{} 里"│
    │  ⟹ 于是评测时那条正则【几乎不会失手】。                  │
    └──────────────────────────────────────────────────────────┘

    ⟹ ★★★ 这是一个双向的因果：
       评测方法 → 影响训练目标 → 塑造模型行为 → 让评测更准。

    ⚠️ 但也要看到硬币的另一面（我的判断）：
       ★★ 这意味着模型的输出格式，一定程度上是【被评测方式塑造的】。
       ⟹ 呼应 09 章冗长偏差那一节的推断：
          🔑 评测的偏好会顺着训练回流进模型行为。
```

---

## 10.7b 第八个发现：R1 报告推翻了"greedy 最可复现"这条常识

⭐⭐ 前面 10.1–10.7 全部取自 **DeepSeek-V3 技术报告**（arXiv:2412.19437，53 页）。
这一节换标本：**DeepSeek-R1 技术报告**（arXiv:2501.12948，86 页，已校验）。

★★ 为什么要单独看 R1？因为它是**推理模型**（reasoning model，会先输出一长段思考过程再给答案），
⟹ 🔑 而推理模型的评测口径，和普通 chat 模型**有一处根本性的不同**。

### ✅ 原文：他们明确说「不用 greedy」

> "We set the maximum generation length to **32,768 tokens** for the models.
> We found that using **greedy decoding** to evaluate long-output reasoning models
> results in **higher repetition rates and significant variability across different
> checkpoints**. Therefore, we default to **pass@k** evaluation and report pass@1
> using a **non-zero temperature**. Specifically, we use a sampling temperature of
> **0.6** and a **top-p value of 0.95** to generate k responses (typically between
> 4 and 64, depending on the test set size) for each question. Specifically, we use
> **k = 64 for AIME and GPQA, k = 16 for MATH and CodeForces, and k = 8 for LCB**."

```
   ★ 名词补全：
   ★ greedy decoding（贪心解码）= 每一步都取概率最高的那个词。
     ⟹ 同一个输入永远给同一个输出，所以看起来【最可复现】。
   ★ temperature（温度）= 采样的随机程度。0 = greedy，越大越随机。
   ★ top-p（核采样，nucleus sampling）= 只在累计概率达到 p 的那批候选词里采样。
     ⟹ 0.95 表示：把概率最低的那 5% 尾巴直接砍掉，防止采到离谱的词。
   ★ pass@k = 对同一道题采样 k 个答案。
   ★★ 但注意 R1 这里 pass@1 的算法是【k 个答案各自对错，取平均】：
```

✅ 论文给出的公式原文：

> pass@1 = (1/k) · Σᵢ pᵢ ，where pᵢ denotes the correctness of the i-th response.
> "This method provides more reliable performance estimates."

```
   ⚠️★★★ 这里有个极易误读的点，必须讲清楚：

   HumanEval 那篇论文里的 pass@k 是【k 次里只要有一次对就算对】（取最大值语义）。
   ⟹ k 越大分数越高。

   ★★ R1 这里的 pass@1 完全不是那个意思，它是【k 次的平均正确率】。
   ⟹ k 越大分数【不会变高，只会变稳】。
   ⟹ 🔑 名字叫 pass@1，含义是"单次尝试的期望正确率"，
      ★★★ 采样 64 次只是为了把这个期望估准，不是为了刷高分。

   ⚠️ 如果你看到两张榜单都写 "pass@1"，
      ★★★ 它们可能一个是 greedy 跑一次，一个是采样 64 次取平均。
      ⟹ 这两个数【不能直接比】。
```

### 🔑 为什么"最可复现"的 greedy 反而被放弃

```
   ┌─ 普通 chat 模型（V3 的做法）────────────────────────┐
   │  ★ 输出几百个 token。                                │
   │  ⟹ greedy 一次跑完，确定性强，✅ V3 的 MATH-500     │
   │     就是 greedy。                                    │
   └──────────────────────────────────────────────────────┘
   ┌─ 推理模型（R1 的处境）────────────────────────────────┐
   │  ★★ 输出可以到 32,768 token（论文原文的上限）。       │
   │  ⚠️ 长序列 + 每步都取最高概率 ⟹ 极易进入【复读循环】： │
   │     "所以答案是……所以答案是……所以答案是……"           │
   │  ✅ 论文原话就是 "higher repetition rates"。          │
   │                                                       │
   │  ⟹ 🔑🔑 结果是反直觉的：                             │
   │     ★★★ greedy 在单次层面确定，但在【模型层面不稳定】  │
   │     —— 论文说它导致 "significant variability across   │
   │     different checkpoints"，                          │
   │     ⟹ 训练过程中相邻两个 checkpoint 的分数会剧烈跳动。 │
   │                                                       │
   │  ⟹ ★★ 换成 T=0.6 采样 64 次取平均，                   │
   │     单次不确定了，但【统计量稳定了】。                 │
   └───────────────────────────────────────────────────────┘

   ⟹ 🔑🔑🔑 这是全章最值得记住的一条方法论：
   ★★★【可复现】有两种，不要混为一谈：
      ① 逐次可复现（同输入 → 同输出）—— greedy 提供的是这个
      ② 统计可复现（同配置 → 同分数）—— 多次采样取平均提供的是这个
   ⚠️ 对长输出模型，★★★ ② 比 ① 重要得多，因为你要比的是分数，不是某一条输出。
```

### 10.7b.1 R1 报告里其余的口径披露（全部 ✅）

```
   ┌──────────────┬──────────────────────────────────────────────┐
   │ 维度         │ ✅ R1 报告的原文披露                          │
   ├──────────────┼──────────────────────────────────────────────┤
   │ 最大输出长度 │ 32,768 tokens（"capped at a maximum of        │
   │              │ 32,768 tokens for each benchmark"）           │
   │ 采样参数     │ temperature 0.6, top-p 0.95                   │
   │ 采样次数 k   │ AIME/GPQA k=64，MATH/Codeforces k=16，        │
   │              │ LiveCodeBench k=8                             │
   │ AIME 额外指标│ cons@64（多数投票，consensus/majority vote）  │
   │ 提示词来源   │ MMLU/DROP/GPQA/SimpleQA → simple-evals 框架   │
   │              │ MMLU-Redux → Zero-Eval 格式，zero-shot       │
   │ ★★ 特殊改动  │ MMLU-Pro / C-Eval / CLUE-WSC 原本是 few-shot，│
   │              │ 他们【改成了 zero-shot】                      │
   │ LiveCodeBench│ CoT 格式；数据窗口 2024-08 ~ 2025-01          │
   │ Codeforces   │ 10 场 Div.2 比赛 + 专家编写的测试用例         │
   │ SWE-Verified │ agentless framework（和 V3 一致）             │
   │ Aider        │ "diff" 格式                                   │
   │ 闭源基线     │ o1-1217 因中国大陆无法访问 API，              │
   │              │ ⚠️【采用官方报告的数字】而非自测              │
   └──────────────┴──────────────────────────────────────────────┘
```

```
   ⟹ 🔑 表里有两行特别值得停下来看：

   ① ★★★【他们把 few-shot 改成了 zero-shot】
      ✅ 原文：
      > "In terms of MMLU-Pro, C-Eval and CLUE-WSC, since the original prompts
      > are few-shot, we slightly modify the prompt to the zero-shot setting.
      > **The CoT in few-shot may hurt the performance of DeepSeek-R1.**"

      ⟹ ★★ 这是 04 章"few-shot 是格式对齐"论点的一次极漂亮的反向验证：
         ★★★ 对推理模型，示例里的思维链反而【干扰】了它自己的思考方式。
      ⚠️ 后果：R1 的 MMLU-Pro 84.0 和别家 5-shot 的 MMLU-Pro，
         ★★★【不在同一个口径下】。这一点榜单上通常看不到。

   ② ⚠️⚠️【o1-1217 的数字是抄来的，不是自测的】
      ⟹ 呼应 11 章"自测还是抄来的"这个问题。
      ★★ 论文诚实地写明了原因（无法访问 API），这正是【好披露】的样子。
      ⟹ 🔑 但读者必须知道：这一列和其他列不是同一次实验的产物。
```

### 10.7b.2 一个数字对照：同一个基准，V3 报告 vs R1 报告

```
   ★★ 拿 SWE-bench Verified 这一行来看（两份报告都有 DeepSeek-V3 这一列）：

   ┌──────────────────────┬────────────┬────────────┐
   │                      │ V3 报告    │ R1 报告    │
   ├──────────────────────┼────────────┼────────────┤
   │ DeepSeek-V3          │ 42.0       │ 42.0       │
   │ Claude-3.5-Sonnet    │ 50.8       │ 50.8       │
   │ GPT-4o-0513          │ 38.8       │ 38.8       │
   └──────────────────────┴────────────┴────────────┘

   ⟹ ✅ 一致。★★ 这说明他们在两份报告之间【保持了 harness 不变】。
   ⟹ 🔑 而这恰恰反衬出 10.2 那句自曝的分量：
      ★★★ 当他们改了框架时，会明说"和之前报告的数字有出入"；
      ⚠️ 没改时，数字就一字不差。
      ⟹ 这就是一个负责任的评测口径应有的样子。
```

### 10.7b.3 R1 还补了一条 06 章的答案：他们怎么做去污染

✅ 原文：

> "DeepSeek-V3 base has a **knowledge cutoff date of July 2024**, predating
> evaluation benchmarks like CNMO 2024, and we **filtered out any text segments
> (including web pages and GitHub files) that contained matching 10-gram sequences**
> from evaluation questions or reference solutions. … in the mathematics domain alone,
> our decontamination process identified and removed approximately **six million**
> potential pre-training texts. For post-training, mathematical SFT data and RL training
> prompts were sourced **exclusively from pre-2023 competitions**…"

⚠️ 而且他们主动承认了这套办法的边界：

> "However, we acknowledge that the **n-gram based decontamination method cannot
> prevent the paraphrase of testset**. Therefore, it is possible that benchmarks
> released before 2024 may suffer from contamination issues."

```
   ⟹ 🔑 这一段把 06 章的三件事全部坐实了：
   ① ★★ 时间切分是主要手段（知识截止 2024-07 + 只用 2023 前的竞赛题）
   ② ★★ n-gram 匹配（这里是 10-gram）是主要的过滤实现
   ③ ★★★ 但换句话说的题（paraphrase）过不掉 ——
      ⚠️ 连一线团队都只能承认这一点，而不是宣称"我们完全干净"。
   ⟹ ★★★ 【会说自己方法边界的报告，比宣称零污染的报告可信得多。】
```

---

## 10.8 把这些汇成一张"该披露什么"的清单

★★★ 这是本章最实用的产出。V3 + R1 两份报告基本把这张表填满了。

```
   ┌────┬──────────────────────┬──────────────────────────────────┐
   │ #  │ 该披露的项            │ ✅ DeepSeek-V3 报告里的原话/做法  │
   ├────┼──────────────────────┼──────────────────────────────────┤
   │ 1  │ ★★ 判别式还是生成式  │ perplexity-based / generation-   │
   │    │                      │ based，逐个数据集列名             │
   │ 2  │ ★★ few-shot 数量     │ Table 3 专门一列 "# Shots"        │
   │ 3  │ ★★ 提示词从哪来       │ simple-evals / Zero-Eval /       │
   │    │                      │ 数据集原始协议                    │
   │ 4  │ ★★★ 温度和跑几遍     │ AIME T=0.7 ×16 平均；             │
   │    │                      │ MATH-500 greedy                  │
   │ 5  │ ★★★ 输出长度上限     │ 全场 8192 token                   │
   │ 6  │ ★★★ agentic 用什么脚手架│ SWE-bench → Agentless          │
   │ 7  │ ★★ 噪声下限          │ "gap not exceeding 0.3 …          │
   │    │                      │ at the same level"               │
   │ 8  │ ★★★ 对比模型是自己跑的│ "evaluate all these models with  │
   │    │  还是抄的            │ our internal evaluation framework"│
   │ 9  │ ★★★ 口径变了要声明   │ "due to the changes in our        │
   │    │                      │ evaluation framework…"           │
   │ 10 │ ★ 跨分词器怎么对齐    │ Pile-test 用 BPB                 │
   │ 11 │ ★ 闭源模型怎么测      │ "through their respective APIs"  │
   ├────┼──────────────────────┼──────────────────────────────────┤
   │    │ ★★ 以下四项由 R1 报告补齐（推理模型必须额外披露）        │
   │ 12 │ ★★★ 采样参数        │ temperature 0.6, top-p 0.95       │
   │ 13 │ ★★★ pass@1 是怎么算的│ k 次采样【取平均】，不是 k 次     │
   │    │                      │ 里"有一次对就算对"                │
   │ 14 │ ★★ 每题采样几次 k    │ AIME/GPQA 64、MATH/CF 16、LCB 8  │
   │ 15 │ ★★★ 改动了原始协议   │ MMLU-Pro/C-Eval/CLUE-WSC          │
   │    │  要说明                │ few-shot →【改成 zero-shot】     │
   │ 16 │ ★★★ 去污染怎么做的   │ 10-gram 过滤 + 只用 2023 前竞赛题 │
   │    │  以及【它的边界】     │ 且承认 paraphrase 过不掉          │
   └────┴──────────────────────┴──────────────────────────────────┘

   ⟹ 🔑🔑 拿这张表去审任何一份技术报告或宣传材料。
      ★★★ 缺得越多，那个分数越不能信。
```

---

## 10.9 那"为什么同一个模型能差十几分"

★ 回答开篇承诺的问题。把前面所有发现合起来，差异的来源是可以列举的：

```
   ┌──────────────────────────┬────────────────────────────────┐
   │ 变量                      │ 量级（⚠️ 我的估计，非论文数据） │
   ├──────────────────────────┼────────────────────────────────┤
   │ ★★★ 判别式 vs 生成式      │ 最大的一项，可达十几分          │
   │ ★★★ 有没有套对话模板       │ 套错可以让分数腰斩              │
   │ ★★ few-shot 0 vs 5        │ 几分到十几分                    │
   │ ★★ 提示词模板不同          │ 几分                            │
   │ ★★ 答案抽取正则不同        │ 几分（05 章：抽不到 = 答错）    │
   │ ★★ max_tokens 截断        │ 对推理模型可以是几十分          │
   │ ★ 温度 / 跑几遍            │ 几分（小样本基准上更大）        │
   │ ★ 框架版本升级             │ ✅ DeepSeek 自承"slight"        │
   │ ★★★ agentic：脚手架不同    │ 可以是成倍差距（08 章）         │
   └──────────────────────────┴────────────────────────────────┘

   ⟹ 🔑🔑🔑 这些【全部会叠加】。
      ★★ 所以"同一个模型在不同 harness 下差十几分"不是异常，是常态。
      ⟹ 异常的是【两个不同来源的分数恰好一样】。
```

---

## 10.10 本章总结

```
    🔑 十条带走（全部来自 DeepSeek V3 / R1 报告原文）：

    ① ★★★ 他们的三分法 = 03 章的三个原语，一字不差
       ⟹ 说明这不是框架实现，是【问题本身的结构】
    ② ✅ 他们自承：框架改了几个月，自家旧模型分数就对不上了
       ⟹ ★★★ 01 章论点的书面证据
    ③ ✅ 差 0.3 分以内算平手 ⟹ ★★ 任何对比先问"超过噪声了吗"
    ④ ✅ shot 数逐题设置，0 到 25 不等 ⟹ ★★ 它是基准身份证的一部分
    ⑤ ✅ base 用 perplexity、chat 用 simple-evals 提示词
       ⟹ ★★★ 同一份报告里的 87.1 和 88.5 不可相减
    ⑥ ✅ SWE-bench 用 Agentless ⟹ ★★★ 报的是"模型+脚手架"，08 章实锤
    ⑦ ✅ 训练时用 \boxed{} 当奖励规则
       ⟹ ★★ 评测方式会反向塑造模型行为

    ⑧ ✅ R1 明说【不用 greedy】：长输出推理模型 greedy 会复读、
       且 checkpoint 之间剧烈跳动 ⟹ 改用 T=0.6 / top-p 0.95 采样多次取平均
       ⟹ ★★★【逐次可复现】和【统计可复现】是两回事，后者更重要
    ⑨ ✅ R1 把 MMLU-Pro/C-Eval/CLUE-WSC 从 few-shot 改成 zero-shot，
       理由是"few-shot 里的 CoT 可能损害 R1 的表现"
       ⟹ ★★★ 04 章"few-shot 是格式对齐"论点的反向验证
    ⑩ ✅ 去污染：10-gram 过滤 + 知识截止 2024-07 + 只用 2023 前竞赛题，
       ⚠️ 但主动承认"n-gram 挡不住换句话说的题"
       ⟹ ★★★ 会说自己方法边界的报告，比宣称零污染的报告可信

    ⟹ 🔑🔑 最后一句：★★★ 这两份报告值得学的不是它们的分数，
       是它【把口径写清楚】这件事本身。
       ⚠️ 你自己写评测结论时，请照着 §10.8 那张表填。
```

---

> 下一章：[11-怎么读懂一张榜单.md](11-怎么读懂一张榜单.md) —— ★ 把前十章的所有教训，压成一套【看到分数先问什么】的操作流程。
> 返回 [评测入门总目录](README.md)
