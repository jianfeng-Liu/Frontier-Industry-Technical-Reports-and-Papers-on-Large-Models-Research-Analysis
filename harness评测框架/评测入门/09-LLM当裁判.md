# 09 · 让模型当裁判：LLM-as-a-Judge

> ★ 07/08 章讲的代码题能"跑测试"来判。
> ★★ 但"这封邮件写得好不好""这个解释清不清楚"没有测试可跑。
> ⟹ 🔑 于是有了一个看起来像作弊、实际却是当前主流的办法：**让另一个模型来打分**。

---

## 9.1 为什么非得这么干

```
    ★ 名词：LLM-as-a-Judge（大模型当裁判）
      LLM = Large Language Model（大语言模型）。
      ★ 用一个强模型去给另一个模型的回答打分或排序。
```

✅ 论文原文（MT-Bench，arXiv:2306.05685，NeurIPS 2023，§3）：

> "collecting human preferences can be **costly and laborious**. ...
> Given that most questions in MT-bench and Chatbot Arena are **open-ended
> without reference answers**, devising a **rule-based program** to assess the
> outputs is **extremely challenging**. Traditional evaluation metrics based on
> the similarity between outputs and reference answers (e.g., ROUGE, BLEU)
> are also **ineffective** for these questions."

★ 三条路，全堵死了：

| 路线 | 优点 | ⚠️ 为什么走不通 |
|---|---|---|
| ① 人来打分 | ★ 最准，是"金标准" | 慢且贵，无法天天跑 |
| ② 写规则 / 正则 | ★ 便宜、完全可复现 | ★★ 开放题根本没有"标准答案"可比 |
| ③ 和参考答案算相似度（BLEU / ROUGE） | ★ 全自动 | ★★ 只看字面重合，⚠️ 换个说法但意思一样 → 判低分 |
| ★★★ ④ 让模型判 | 便宜、可扩展、★ 还附带解释 | ⚠️ 本章后面全部在讲它的代价 |

```
    ★ 名词：BLEU / ROUGE
      BLEU = Bilingual Evaluation Understudy（双语评估替补，2002 年，机器翻译用）
      ROUGE = Recall-Oriented Understudy for Gisting Evaluation
              （面向召回的摘要评估替补，2004 年，自动摘要用）
      ★ 两个上古自动指标，本质都是【统计你的输出和参考答案有多少词/词组重合】。
      ★★ 对翻译/摘要还行，对开放问答基本失效。
```

★★ 论文还指出一个更根本的动机（§1）：

> "benchmarks like **MMLU and HELM cannot effectively tell the difference
> between these aligned models and the base models**."

```
    ⟹ 🔑🔑 这句话分量很重：★★★ 做完对齐（RLHF）后用户明显更喜欢的模型，
       在 MMLU 这类选择题榜上【看不出提升】。
       ⟹ 说明 01-06 章那套判别式评测，
          ★★ 测不出"好不好用"这个维度。这是 LLM 裁判存在的根本理由。

    ★ 名词：RLHF = Reinforcement Learning from Human Feedback
      （基于人类反馈的强化学习）—— 用人的偏好数据去调模型的那一步训练。
    ★ 名词：MMLU = Massive Multitask Language Understanding（大规模多任务语言理解）
    ★ 名词：HELM = Holistic Evaluation of Language Models（语言模型整体评估）
```

---

## 9.2 三种裁判形态

✅ 论文 §3.1 明确列了三种（*"We propose 3 LLM-as-a-judge variations"*）：

| # | 形态 | 怎么做 | ★ 优点 | ⚠️ 缺点（✅ 均为论文原文） |
|---|---|---|---|---|
| ① | **Pairwise comparison**（两两对比） | 给裁判一个问题 + 两份答案 → 判谁更好，或平局 | ★★ 最稳，因为"比较"比"打绝对分"容易 | ✅ *"the number of possible pairs grows **quadratically**"* ⟹ ★ 模型一多，对数爆炸 |
| ② | **Single answer grading**（单答案打分） | 直接给一份答案打个分（比如 1–10） | ★ 便宜、线性，可扩展 | ✅ *"may be unable to discern subtle differences"*、*"results may become **unstable**, as absolute scores are likely to fluctuate more than relative pairwise results **if the judge model changes**"* |
| ③ | **Reference-guided grading**（带参考答案打分） | ★★ 把一份参考答案也给裁判看，再让它判 | ⟹ 数学题上效果显著（见 §9.4 ④） | ⚠️ 需要先有参考答案 |

```
    ⚠️⚠️ 第 ③ 种有一个极易混淆的地方，必须现在就说清楚：

    ★★★ "Reference" 这个词在两处指的不是同一个东西。

    ① 论文 §3.4 的 reference-guided judge（用来降低数学题误判的那个）：
       ✅ 原文："we propose a reference-guided method, in which we first generate
       ✅        **LLM judge's answer independently**, and then display it as a
       ✅        reference answer in the judge prompt."
       ⟹ ★★ 它给裁判看的是【裁判自己先独立做一遍得到的答案】，
          ⚠️ 不是数据集里的标准答案。

    ② 工程框架里的 reference（比如下面 EvalScope 的模板）：
       ⟹ ★★ 给的是数据集里真正的 gold target（标准答案）。

    ⟹ 🔑 两者都能提升判准率，但【可用的场景完全不同】：
       ①【任何时候都能用】，因为裁判总能自己先答一遍；
       ②【只有在你手里有标准答案时才能用】。
    ⟹ ⚠️ 看到"用了 reference-guided"这句话，★★★ 要问清是哪一种。
```

★ 这三种在源码里都能看到。比如 EvalScope 的默认裁判模板 [metrics/judge/llm_judge.py:12](../源码/evalscope-1.10.0/evalscope/metrics/judge/llm_judge.py) 就是第 ③ 种的"给真标准答案"版本（★ 逐字，节选）：

```
Your job is to look at a question, a gold target, and a predicted answer,
and return a letter "A" or "B" to indicate whether the predicted answer is
correct or incorrect.

[Question] {question}
[Reference Answer] {gold}
[Predicted Answer] {pred}

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
```

```
    ★★ 注意最后一句 "Just return the letters ... with no text around it"。
    ⟹ 🔑 这是在【为 05 章的抽取环节服务】：
       裁判自己也是个模型，它的输出也要被正则抽取。
       ⟹ ★★ 所以裁判提示词里必须死死约束输出格式。
```

同文件 L32-41 还有第 ② 种模板 `DEFAULT_NUMERIC_SCORE_TEMPLATE`（★ 逐字节选）：

> "you must rate the response on a scale of 0 (worst) to 1 (best) by strictly
> following this format: `\"[[rating]]\"`, for example: `\"Rating: [[0.5]]\"`"

```
    ★★ 那个双方括号 [[ ]] 不是装饰，是【给正则用的锚点】。
    ⟹ 🔑 因为裁判会写一大段解释，必须有个不会误命中的标记把分数框出来。

    ⚠️ 而且请记住 05 章那条教训：★★★ 抽取失败要单独记，不能混进"判负"里。
       ⟹ §9.5 会给出一个源码里的反面例子：
          某个适配器里，【解析不出来的判决被直接丢掉了】。
```

---

## 9.3 裁判靠谱吗？—— 靠谱，但那个 "80%" 有口径

✅ 论文摘要（★ 逐字）：

> "Our results reveal that strong LLM judges like **GPT-4 can match both controlled
> and crowdsourced human preferences well, achieving **over 80% agreement**, the same
> level of agreement between humans."

```
    ⟹ 🔑🔑 最后半句是关键：★★★【人和人之间的一致率也就 80% 左右】。

    ★★ 这意味着：不能因为"裁判和人不完全一致"就否定它，
       因为【人和人本来也不完全一致】。
       ⟹ 开放题的"正确答案"本身就不唯一。
```

⚠️⚠️ **但这个 80% 有一个必须问清的口径**，否则你会把它读高一大截。✅ 论文 Table 5 同时报了两套设定，原文定义如下：

| 设定 | ✅ 论文原文定义 | 包含哪些投票 | ✅ 随机裁判的基线（论文记作 "R="） |
|---|---|---|---|
| **S1** | *"includes non-tie, tie, and inconsistent (due to position bias) votes and counts inconsistent as tie"* | ★ 全部投票，把"交换顺序后结论变了"的也算平局 | **R = 33%**（A 胜 / B 胜 / 平局 三种结果） |
| **S2** | *"**only includes non-tie votes**"* | ⚠️ 只留双方都给出明确胜负的那些题 | **R = 50%**（两种结果） |

✅ Table 5（MT-bench，第一轮；下方灰色小字是投票数，这里一并列出）：

| 一致率 | S1（含平局，R = 33%） | S2（只看非平局，R = 50%） |
|---|---|---|
| GPT-4 两两对比 vs GPT-4 单答案打分 | 70%（1,138 票） | 97%（662 票） |
| **GPT-4 两两对比 vs 人类专家** | **66%**（1,343 票） | **85%**（859 票） |
| GPT-4 单答案打分 vs 人类专家 | 60%（1,280 票） | 85%（739 票） |
| **人类专家 vs 人类专家** | **63%**（721 票） | **81%**（479 票） |

```
    ⟹ 🔑🔑🔑 摘要里那个 ">80%" 和 "same level as humans"，★★★ 说的是 S2 这一列：
       GPT-4 对人 85%，人对人 81%。
    ⟹ ⚠️ 而 S1（把平局和"顺序一换就翻"的题都算进去）只有 66% 对 63%。

    ★★ 两个数都是真的，但它们【回答的是不同问题】：
       S2 问：「在两个人都觉得能分出胜负的题上，裁判和人合得上吗」⟹ 合得上（85%）
       S1 问：「在全部题目上，裁判和人合得上吗」            ⟹ 只有 66%
    ⟹ ★★★ 引用时必须带上是 S1 还是 S2，以及票数（★ 注意 S2 的样本量只有 S1 的六成）。

    ⚠️ 还要注意时代：这是 2023 年的 GPT-4（gpt-4-0314 / 0613 那一代）。
       ★ 今天的裁判模型只会更强，⚠️ 不过 §9.4 那些偏差【并没有消失】。
```

★ 大规模众包那边（✅ Table 6，Chatbot Arena，3K 随机单轮投票）趋势一致：

| 裁判对 | S1（R = 33%） | S2（R = 50%） |
|---|---|---|
| GPT-4 vs 人（众包） | 64% | **87%** |
| GPT-3.5 vs 人 | 54% | 83% |
| Claude vs 人 | 53% | 84% |

```
    ⟹ ★★ 论文对这张表的解读（✅ 原文）：
    > "they reach a similar non-tie agreement ratio between humans but the number
    >  of **non-tied votes from GPT-4 is much larger**. This means that GPT-4 is
    >  more affirmative and less suffered from position bias"
    ⟹ 🔑 读法：GPT-3.5 / Claude 在"敢下结论的那些题"上也判得不错（83–84%），
       ★★ 但它们【敢下结论的题少得多】—— 剩下的都掉进了平局和自相矛盾里。
    ⟹ ★★★ 所以比较两个裁判，光看一致率不够，还要看它【判了多少题】。
```

---

## 9.3b ⭐⭐ 一致性到底怎么量：把 Cohen's kappa 走一遍

★★★ 这一节是本章最该动笔算一遍的地方。**"一致率 80%" 是一个会骗人的数字**，因为它没有扣掉"瞎猜也能蒙对"的那一部分。

### ① 先看问题长什么样：三张都是 80% 的表

★ 设定：100 道题，人类和 LLM 裁判各判"A 胜"还是"B 胜"（先不考虑平局）。三张表的**一致率全都正好是 80%**。

**表甲 · 胜负均衡**

|  | 人 = A 胜 | 人 = B 胜 | 行和 |
|---|---|---|---|
| **裁判 = A 胜** | **40** | 10 | 50 |
| **裁判 = B 胜** | 10 | **40** | 50 |
| 列和 | 50 | 50 | 100 |

**表乙 · 一边倒（多数题都是 A 胜）**

|  | 人 = A 胜 | 人 = B 胜 | 行和 |
|---|---|---|---|
| **裁判 = A 胜** | **72** | 10 | 82 |
| **裁判 = B 胜** | 10 | **8** | 18 |
| 列和 | 82 | 18 | 100 |

**表丙 · 极度一边倒**

|  | 人 = A 胜 | 人 = B 胜 | 行和 |
|---|---|---|---|
| **裁判 = A 胜** | **78** | 10 | 88 |
| **裁判 = B 胜** | 10 | **2** | 12 |
| 列和 | 88 | 12 | 100 |

```
    ⟹ 三张表的一致率（对角线 ÷ 总数）：
       表甲 (40+40)/100 = 80%
       表乙 (72+ 8)/100 = 80%
       表丙 (78+ 2)/100 = 80%
    ⟹ ⚠️ 一模一样。但它们的"裁判到底有没有本事"差得非常远。
    ⟹ 🔑 直觉检查：表丙里，一个【什么都不看、永远判 A 胜】的假裁判
       就能拿到 88% 的一致率 —— ★★★ 比这个真裁判的 80% 还高！
```

### ② Cohen's kappa：把"蒙对的那部分"扣掉

```
    ★ 名词：Cohen's kappa（科恩 kappa 系数，κ）
      1960 年 Jacob Cohen 提出，用来衡量【两个评分者的一致程度】。
      ★★ 核心思想：把"就算两人各自乱判也会碰上的那部分一致"先减掉。

      κ = (p_o − p_e) / (1 − p_e)

      p_o = observed agreement  （实测一致率 = 对角线 ÷ 总数）
      p_e = expected agreement  （★ 偶然一致率 = 把两人的边缘分布当独立事件算出来的）
      ⟹ κ = 1 完全一致；κ = 0 和瞎猜一样；κ < 0 比瞎猜还差。
```

★ p_e 怎么算：**把每一类的"行比例 × 列比例"加起来**。意思是"如果裁判和人各自按自己的习惯独立乱判，会碰上多少"。

★★★ 三张表逐张算（★ 算术全在这里，可自己验算）：

```
   表甲：
     行比例 = 50/100、50/100        列比例 = 50/100、50/100
     p_e = 0.50×0.50 + 0.50×0.50 = 0.25 + 0.25 = 0.5000
     κ   = (0.80 − 0.5000) / (1 − 0.5000) = 0.3000 / 0.5000 = ★ 0.600

   表乙：
     行比例 = 82/100、18/100        列比例 = 82/100、18/100
     p_e = 0.82×0.82 + 0.18×0.18 = 0.6724 + 0.0324 = 0.7048
     κ   = (0.80 − 0.7048) / (1 − 0.7048) = 0.0952 / 0.2952 = ★ 0.322

   表丙：
     行比例 = 88/100、12/100        列比例 = 88/100、12/100
     p_e = 0.88×0.88 + 0.12×0.12 = 0.7744 + 0.0144 = 0.7888
     κ   = (0.80 − 0.7888) / (1 − 0.7888) = 0.0112 / 0.2112 = ★ 0.053
```

★★★ 把结果摆在一起：

| 表 | 实测一致率 p_o | 偶然一致率 p_e | **Cohen's κ** | ⚠️ 结论 |
|---|---|---|---|---|
| 甲（均衡） | 80% | 50.0% | **0.600** | ★ 中等偏好，这个裁判确实有本事 |
| 乙（一边倒） | 80% | 70.5% | **0.322** | ⚠️ 勉强，★★ 大半一致是"蒙"出来的 |
| 丙（极度一边倒） | 80% | 78.9% | **0.053** | ⚠️⚠️ ★★★ 几乎等于瞎猜 |

```
    ⟹ 🔑🔑🔑 这就是那句话的来源：
       ★★★【准确率 80% 看着不错，kappa 可能只有 0.05】。

    ⟹ ★★ 什么时候最危险？【题目分布一边倒的时候】。
       ⟹ 🔑 而这恰恰是主观评测的常态：拿一个强模型去打一个弱模型，
          90% 的题都是强的赢 —— ★★★ 此时"一致率"这个指标几乎不含信息。

    ⟹ ★★★ 所以报裁判质量，光报一致率不够，必须【同时报 kappa（或同类的
       去偶然化指标）和两边的边缘分布】。⚠️ 只给一致率的，看不出上面三张表的差别。
```

★ κ 的常用解读区间（⚠️ 这是 Landis & Koch 1977 提出的**约定**，不是数学定理，不同领域标准不同）：

| κ 区间 | 常见说法 | ⚠️ 用在裁判评估上的建议 |
|---|---|---|
| < 0 | poor（比瞎猜差） | ⚠️ 裁判反着判，查提示词和抽取 |
| 0.00 – 0.20 | slight（微弱） | ⚠️⚠️ 不可用于任何结论 |
| 0.21 – 0.40 | fair（一般） | ⚠️ 只能当迭代方向的弱信号 |
| 0.41 – 0.60 | moderate（中等） | ★ 可做内部迭代信号 |
| 0.61 – 0.80 | substantial（可观） | ★★ 可做内部结论 |
| 0.81 – 1.00 | almost perfect（近乎完美） | ★★★ 主观题上基本达不到，★ 达到了先怀疑数据 |

### ③ 有平局怎么办：3×3 的例子

★★ 真实的两两对比几乎一定有"平局"。★ kappa 的公式一个字都不用改，只是对角线变成三格。

| | 人 = A 胜 | 人 = 平局 | 人 = B 胜 | 行和 |
|---|---|---|---|---|
| **裁判 = A 胜** | **40** | 12 | 3 | 55 |
| **裁判 = 平局** | 8 | **10** | 6 | 24 |
| **裁判 = B 胜** | 4 | 7 | **10** | 21 |
| 列和 | 52 | 29 | 19 | 100 |

```
   p_o = (40 + 10 + 10) / 100 = 0.6000

   p_e = (55/100)×(52/100) + (24/100)×(29/100) + (21/100)×(19/100)
       = 0.2860 + 0.0696 + 0.0399
       = 0.3955

   κ   = (0.6000 − 0.3955) / (1 − 0.3955) = 0.2045 / 0.6045 = ★ 0.338

   ⟹ ★★ 注意 p_e 从 2×2 的 0.5 掉到了 0.3955：★ 类别越多，蒙对越难，
      所以同样的一致率会换出更高的 kappa。
   ⟹ 🔑🔑 反过来说：★★★【把平局并进胜负里（论文的 S2 做法）会让
      一致率变好看，同时让偶然一致率也变高】。两个效应方向相反，
      ⚠️ 所以不算 kappa 是说不清的。
```

### ④ 把论文那些一致率换算成 kappa

★★ 论文给了每个设定下"随机裁判的一致率"（那个 `R=`），★★★ 这正好就是 p_e。于是可以直接换算：

| 出处 | 一致率 p_o | ✅ 论文给的随机基线 R = p_e | **换算出的 κ** | 读法 |
|---|---|---|---|---|
| MT-bench S2，GPT-4 vs 人 | 85% | 50% | **0.70** | ★★ substantial |
| MT-bench S2，人 vs 人 | 81% | 50% | **0.62** | ★★ substantial |
| MT-bench S1，GPT-4 vs 人 | 66% | 33% | **0.49** | ★ moderate |
| MT-bench S1，人 vs 人 | 63% | 33% | **0.45** | ★ moderate |
| Arena S2，GPT-4 vs 人 | 87% | 50% | **0.74** | ★★ substantial |
| Arena S1，GPT-4 vs 人 | 64% | 33% | **0.46** | ★ moderate |

```
   ★ 算一行给你看：(0.85 − 0.50) / (1 − 0.50) = 0.35 / 0.50 = 0.70

   ⟹ 🔑🔑 换算之后，结论比"80% 一致率"清楚得多：
      ★★★ GPT-4 当裁判的水平是 κ ≈ 0.49（全量）到 0.70（只看非平局），
      ★★ 而人类专家之间是 κ ≈ 0.45 到 0.62。
      ⟹ ★ "和人同级"这个说法成立，而且换算后【两个设定下都成立】——
         这比只看 S2 那一列更有说服力。

   ⚠️ 两条必须标出的限定：
   ① 这几行的 κ 是【我用论文给的 p_o 和 R 换算的】，论文本身没有报 kappa。
   ② ★★ 严格说，用 R（假设两边都均匀乱判）当 p_e，算出来的是
      Scott's pi / Fleiss 式的"均匀边缘"版本，而 Cohen's kappa 用的是
      【实测的边缘分布】（上面表甲乙丙那种算法）。
      ⟹ 🔑 两者在边缘分布接近均匀时几乎相同，一边倒时会差很多。
      ⟹ ⚠️ 所以自己做评估时，请用实测边缘算 Cohen's kappa，别用固定的 1/2、1/3。
```

### ⑤ 什么时候该换成 Krippendorff's alpha

```
    ★ 名词：Krippendorff's alpha（克里彭多夫 alpha，α）
      ★ 另一个去偶然化的一致性系数，公式形状是：

      α = 1 − D_o / D_e

      D_o = observed disagreement（实测分歧量）
      D_e = expected disagreement（偶然分歧量）
      ★★ 和 kappa 同一个思路（"实测 vs 偶然"），
         区别在于它是按【分歧】而不是【一致】来算，所以能带权重。
```

★ 三种情况下 kappa 不够用，要换 alpha：

| 情况 | ⚠️ kappa 的问题 | ★ alpha 怎么解决 |
|---|---|---|
| **评分者多于 2 个** | ★★ Cohen's kappa 只定义在两个评分者之间 | ★ α 原生支持任意多个评分者 |
| **有缺失值**（不是每个人都判了每一题） | ⚠️ 要么丢数据，要么两两算再平均 | ★★ α 直接支持不完整设计 |
| **分数是有序的或连续的**（比如 1–10 分） | ⚠️⚠️ ★★★ kappa 把"判 9 分 vs 判 10 分"和"判 1 分 vs 判 10 分"当成同样严重的分歧 | ★★★ α 可以换距离函数（ordinal / interval），让"差得远"罚得更重 |

```
    ⟹ 🔑 一条朴素的选择规则：
       ★★ 两个裁判 + 离散类别（A 胜 / 平 / B 胜）⟹ Cohen's kappa，够用且好解释
       ★★★ 多个裁判，或者要判的是 1–10 分这种【有序分数】⟹ Krippendorff's alpha
       ⚠️ 用 kappa 去衡量 1–10 分的单答案打分，会系统性地低估一致性。

    ⚠️ 还有一个更简单但常被忘掉的做法：★★ 对有序分数直接算
       Spearman 秩相关或 Kendall's tau。它不去偶然化，但胜在人人都会读。
       ⟹ 🔑 最稳的做法：★★★ 同时报【一致率 + 去偶然化系数 + 秩相关】三个数，
          任何一个单独拿出来都能被曲解。
```

---

## 9.4 三个必须知道的偏差（每个都配一个可操作的检测法）

★★★ 这是本章最有价值的部分。论文 §3.3 用实验把它们量化了，下面在每一条后面补一套【你自己能跑的检测流程】。

### ① 位置偏差（position bias）

✅ 定义：*"an LLM exhibits a propensity to favor certain positions over others"*

✅ 实验设置（★ 逐字）：*"we construct two similar answers to each first-turn question in MT-bench by calling **GPT-3.5 twice with a temperature of 0.7**"* —— ★★ 也就是同一个模型采两次，本来就极难分辨。

✅ Table 2（★ 三个裁判 × 两种提示词，原文数值）：

| 裁判 | 提示词 | 一致率（交换顺序后结论不变） | 偏向第一个 | 偏向第二个 | ⚠️ 输出格式错误 |
|---|---|---|---|---|---|
| Claude-v1 | default | ⚠️⚠️ **23.8%** | ★★ **75.0%** | 0.0% | 1.2% |
| Claude-v1 | **rename** | **56.2%** | 11.2% | ★ **28.7%** | 3.8% |
| GPT-3.5 | default | 46.2% | 50.0% | 1.2% | 2.5% |
| GPT-3.5 | rename | 51.2% | 38.8% | 6.2% | 3.8% |
| GPT-4 | default | ★ **65.0%** | 30.0% | 5.0% | 0.0% |
| GPT-4 | rename | 66.2% | 28.7% | 5.0% | 0.0% |

```
    ⟹ 🔑🔑 读懂 Claude-v1 的 default 那一行：只有 23.8% 的情况下判决不随顺序变，
       而 75.0% 的情况下【谁放前面谁赢】。
       ⟹ ★★★ 也就是四次里有三次，胜负是摆放顺序决定的。
       ⟹ 那一次评测得到的排名，★★ 相当大一部分是【摆放顺序造成的】。

    ⚠️ 论文自己说明这个测试很苛刻：
       ✅ "occasionally indistinguishable even to humans"
       ⟹ ★ 所以别把 23.8% 理解成"Claude 判什么都乱"。
       ✅ 论文也给了一个宽松些的结论："Only GPT-4 outputs consistent results
          in more than 60% of cases."
```

★★★ 现在看 **rename 那两行**，它藏着一个比"位置偏见"更精确的诊断：

```
    ★ rename 是什么：✅ 论文原文 —— "'rename' renames the assistants in our
      default prompt to see whether the bias is on **positions** or **names**."

    ⟹ 🔑🔑 Claude-v1 换了名字之后，★★★【偏向的方向翻过来了】：
       default：偏向第一个 75.0%，偏向第二个  0.0%
       rename ：偏向第一个 11.2%，偏向第二个 28.7%   ← ★★ 反过来了

    ⟹ ✅ 论文的结论（逐字）："Claude-v1 also shows a **name bias** which makes
       it favors **"Assistant A"**, as illustrated by the "rename" prompt."
    ⟹ ★★★ 所以 Claude-v1 那个 75% 不是"偏爱前面的位置"，
       ★★ 是【偏爱叫做 "Assistant A" 的那一个】。
       ⟹ 🔑 这是两种完全不同的病，药也不同：
          位置偏见 ⟹ 交换顺序判两遍就能抵消
          ⚠️ 名字偏见 ⟹ 交换顺序【治不了】，必须同时换标签，或者干脆用中性标签

    ⚠️ 上面"哪个标签落到了第几位"是我对 rename 设置的合理推断；
       论文只写了 "renames the assistants"，没给出具体改成了什么。
       ★ 但"存在名字偏见、偏向 Assistant A"这一条是论文明文写的。
```

★★ 论文对成因的猜测（✅ 逐字）：

> "we suspect that it could be rooted in the training data or **inherent to the
> left-to-right architecture of causal transformers**"

```
    ★ 名词：causal transformer（因果 Transformer）
      ★★ 就是现在所有主流大模型的结构：读文本时【只能看左边，不能看右边】。
      ⟹ 于是"先读到的东西"天然占据了更多的推理路径。
      ⚠️ 这只是论文的猜测（"we suspect"），未被证明。
```

**★★★ 可操作的检测流程（位置偏差 + 名字偏差）**

```
   ① 准备 N 道题，每道题两份答案 A、B
      ★★★ 关键：两份答案的质量要【尽量接近】。
      ⟹ 论文的做法是同一个模型 temperature=0.7 采两次。
      ⚠️ 拿强模型答案 vs 弱模型答案去测，测不出偏差 —— 因为差距太大，
         裁判不靠位置也能判对。

   ② 每道题判两遍：顺序 (A, B)，以及顺序 (B, A)

   ③ 把每道题归到四类里的一类，统计比例：
        一致        两遍结论指向同一份答案（含两遍都判平局）
        偏向第一位  两遍都判"排在前面的那一份"赢
        偏向第二位  两遍都判"排在后面的那一份"赢
        格式错误    至少一遍抽不出结论   ← ★★ 务必单独记（05 章那条铁律）

   ④ ★ 加一组对照实验（这一步很多人省掉，然后误诊）：
      把标签从 "Assistant A / Assistant B" 换成别的（比如 "甲 / 乙"，
      或者把 A/B 两个标签的位置互换），★★ 位置不动，只动名字。
      ⟹ 一致率变了 ⟹ 你抓到的是【名字偏见】
      ⟹ 一致率没变 ⟹ 你抓到的是【位置偏见】
```

★ 拿到"一致率"这个数之后怎么用（⚠️ 以下阈值是我给的操作建议，不是论文结论）：

| 一致率 | 怎么判 | 该怎么办 |
|---|---|---|
| < 50% | ⚠️⚠️ 结论基本由顺序决定 | ★★★ 换裁判模型，或重写提示词。两遍法救不回来 |
| 50% – 70% | ⚠️ 偏差显著 | ★★ 必须两遍法，且要接受"很多题判不出胜负" |
| 70% – 85% | ★ 偏差可控 | ★★ 仍然建议两遍法；单遍要随机化位置 |
| > 85% | ★★ 良好 | ★ 单遍可接受，但仍要随机化位置以防系统性漂移 |

**★★★ 怎么治：两遍法。但"两遍"之后怎么合，有两种做法，结果不一样。**

✅ 论文 §3.4 给的是**保守法**（逐字）：

> "A **conservative** approach is to call a judge twice by swapping the order of two
> answers and **only declare a win when an answer is preferred in both orders**.
> If the results are inconsistent after swapping, we can **call it a tie**. ...
> Another more **aggressive** approach is to **assign positions randomly**, which can be
> effective at a large scale with the correct expectations."

★★ 而 EvalScope 的 Arena-Hard 适配器用的是另一种：**两局取平均**。[arena_hard_adapter.py:41](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py) 的文档字符串直接写着（★ 逐字）：

```
- Two-game battle system (A vs B and B vs A)
```

实现在同文件 L105-138：

```python
prompt1 = GRADER_TEMPLATE.format(question=question, answer_1=reference, answer_2=filtered_prediction)
prompt2 = GRADER_TEMPLATE.format(question=question, answer_1=filtered_prediction, answer_2=reference)  # 交换
game1_response = self.llm_judge.judge(prompt1, system_prompt=GRADER_SYSTEM_PROMPT)
game2_response = self.llm_judge.judge(prompt2, system_prompt=GRADER_SYSTEM_PROMPT)
...
score1 = get_judge_score(res1, reverse=True)     # ★ 注意 reverse：第一局里基线在前
score2 = get_judge_score(res2, reverse=False)
...
score.value = {'score': (score1 + score2) / 2}   # ★★ 两局取平均
```

★ 那个 `get_judge_score` 不是 0/1，而是五档（[utils.py:22-53](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/utils.py)，★ 逐字数值）：

| 裁判输出 | 含义 | 分数（正向） | 分数（reverse） |
|---|---|---|---|
| `A>>B` | A 明显胜 | 1.00 | 0.00 |
| `A>B` | A 略胜 | 0.75 | 0.25 |
| `A=B` | 平局 | 0.50 | 0.50 |
| `B>A` | B 略胜 | 0.25 | 0.75 |
| `B>>A` | B 明显胜 | 0.00 | 1.00 |
| ⚠️ 抽取失败 | — | ★ 默认 0.5（`score_mapping.get(result, 0.5)`） | 同 |

★★★ 两种合法的对比：

| | ✅ 论文的保守法 | ✅ Arena-Hard 源码的平均法 |
|---|---|---|
| 两局都赢 | 判赢 | 高分（接近 1.0） |
| 两局矛盾 | ★★ **判平局** | ★ 落在 0.5 附近的中间分 |
| 输出的粒度 | 胜 / 平 / 负（三档） | ★ 0 到 1 的连续值 |
| ★ 优点 | ⚠️ 保守，不会把"顺序造成的胜利"记成胜利 | ★★ 保留强弱程度，分辨率更高 |
| ⚠️ 缺点 | ★★ 平局大量增多，榜单可能分不出胜负 | ⚠️ 矛盾判决被"摊平"成 0.5，★ 看不出它是矛盾还是真平局 |

```
    ⟹ 🔑 一句话：★★★ 一道题判两次，成本翻倍，换掉一个系统性偏差。
       ⚠️ 看到某个主观榜单【只判一遍】，它的结果就要打折看。
    ⟹ ⚠️ 而看到"我们用了两遍法"，★★ 还要再问一句【怎么合的】——
       保守法和平均法得出的榜单不是同一个东西。
```

★ 论文还试了另一条路：**给裁判几个示范**（few-shot judge，✅ §3.4）：

| 做法 | GPT-4 的一致率 | ⚠️ 代价（✅ 均为论文原文） |
|---|---|---|
| zero-shot（默认） | 65.0% | — |
| few-shot（3 个判例） | ★★ **77.5%** | ⚠️ *"high consistency may not imply high accuracy and we are not sure whether the few-shot examples will introduce new biases"*；⚠️ *"the longer prompts make API calls **4× more expensive**"* |

```
    ⟹ 🔑🔑 ★★★ 这一行值得单独记住："一致率变高" ≠ "判得更准"。
       ⚠️ 一个【永远判 A 赢】的裁判，一致率是 0%（交换后就翻）……
          但一个【永远判平局】的裁判，一致率是 100%。
       ⟹ ★★★ 所以一致率必须和"和人的一致率 / kappa"一起看（§9.3b），
          单独优化一致率可以优化出一个毫无用处的裁判。
```

### ② 冗长偏差（verbosity bias）

✅ 定义：*"an LLM judge favors longer, verbose responses, even if they are not as clear, high-quality, or accurate as shorter alternatives"*

★★ 论文设计了一个很妙的攻击实验："repetitive list" attack。✅ 原文的配方（逐字）：

> "We first select **23 model answers** from MT-bench that contain a numbered list.
> We then make them unnecessarily verbose by asking GPT-4 to **rephrase the list
> without adding any new information** and insert the rephrased new list to the
> beginning of the original list. For example, if the original response contains
> 5 items, then the new response will contain **10 items but the first 5 items are
> rephrased from the original 5 items**. ... We define the attack is successful if
> an LLM judge thinks the new response is better than the old response."

✅ Table 3（23 个答案上的"攻击成功率"，论文列名为 Failure rate）：

| 裁判 | Claude-v1 | GPT-3.5 | GPT-4 |
|---|---|---|---|
| 攻击成功率 | ⚠️⚠️ **91.3%** | ⚠️⚠️ **91.3%** | ★ **8.7%** |

```
    ⟹ 🔑🔑 读法：★★★ 把答案注水一倍、不加任何新信息，
       有两个裁判【九成以上会认为变好了】。

    ★ 顺手验算一下样本量：23 × 91.3% = 21.0 条，23 × 8.7% = 2.0 条。
      ⟹ ⚠️ 所以这三个数字背后各自只有 23 道题。
         ★★ 91.3% vs 8.7% 的差距足够大，结论可信；
         ⚠️ 但如果有人报"某裁判 87.0%"，那只是差 1 条，★ 别当真。

    ⟹ ★★ 这直接解释了一个你一定见过的现象：
       为什么很多模型答什么都要先来一段"这是一个很好的问题"，
       然后分点、加粗、总结、再展望。
       ⟹ 🔑 因为在【被冗长偏差污染的偏好数据】上训练过。

    ⚠️ 这是我的推断链，不是论文结论。但方向是清楚的：
       ★★★ 评测的偏差会顺着训练回流到模型行为里。
```

★★★ 论文还做了一个**必须复制的对照组**（✅ 原文）：

> "As a calibration, we find LLM judges are able to correctly judge identical
> answers (i.e., they **always return a tie for two identical answers**) but
> cannot pass the more advanced "repetitive list" attack."

```
    ⟹ 🔑 这一步的作用：★★★ 排除"裁判就是在瞎判"这个平凡解释。
       ⟹ 两份【完全相同】的答案，它稳定判平局 ⟹ 说明它有判断力。
       ⟹ 两份【信息量相同但一长一短】的答案，它判长的赢 ⟹ ★★ 它是真的被长度骗了。
```

**★★★ 可操作的检测流程（冗长偏差）**

```
   ★ 方法 A：注水攻击（论文那套，★★ 最推荐，因为它直接控住了"质量"这个混淆变量）
   ① 从你自己的评测集里挑 N 条【含编号列表或分点结构】的回答（N ≥ 20）
   ② 用另一个模型把列表"★★ 换个说法复述，不许加任何新信息"
   ③ 把复述版拼在原列表【前面】⟹ 条数翻倍，信息量不变
   ④ 让裁判两两对比"新版 vs 原版"（★ 记得用两遍法消掉位置偏差）
   ⑤ 攻击成功率 = 判"新版更好"的比例
      ★ 理想值 = 0%，★★ 理想的判决分布应该是"绝大多数平局"
   ⑥ ★★★ 必做对照：把【完全相同】的两份答案喂进去，确认它判平局。
      ⟹ 这一步不做，你无法区分"被长度骗了"和"判决本来就是噪声"

   ★ 方法 B：长度分桶（★ 便宜，用来做日常监控）
   ① 把所有被评的回答按 token 数分成 4–5 桶
   ② 看每个桶的平均得分 / 胜率随长度的走势
   ⚠️⚠️ 致命混淆：★★★【长回答可能真的更好】。
      ⟹ 所以只看"长度和分数正相关"证明不了偏差。
   ③ 修正办法：★★ 只在【人工已判定质量相同】的样本内做这个对比，
      或者反过来看——固定人工评分档位，看同档位内长度对裁判分的影响。

   ★ 方法 C：截断对照（★ 最省事的近似）
   ① 把同一份回答删掉纯冗余部分（客套话、重复总结），造一个短版
   ② 让裁判判"长版 vs 短版"
   ③ ⚠️ 比方法 A 弱，因为"删掉的到底是不是纯冗余"是人判断的，有主观性
```

**怎么治：** ⚠️ 论文没给通用解法。工程上常见的是"控制长度偏差"的评测变体——OpenCompass 里就专门有一个数据集叫 [compassbench_control_length_bias.py](../源码/opencompass-0.5.3/opencompass/datasets/subjective/compassbench_control_length_bias.py)。★ 一个偏差重要到要为它单独做一套题，说明它有多难缠。

### ③ 自我偏好（self-enhancement bias）

✅ 定义：*"LLM judges may favor the answers generated by themselves"*

✅ 论文 §3.3 实测（★ 逐字，★★ 请把后半句一起读完）：

> "GPT-4 favors itself with a **10% higher win rate**;
> Claude-v1 favors itself with a **25% higher win rate**.
> However, they also favor other models and **GPT-3.5 does not favor itself**.
> **Due to limited data and small differences, our study cannot determine whether
> the models exhibit a self-enhancement bias.** Conducting a controlled study is
> challenging because we cannot easily rephrase a response to fit the style of
> another model without changing the quality."

| 裁判 | 给"自己写的答案"的胜率，比人给的高多少 | 读法 |
|---|---|---|
| GPT-4 | **+10 个百分点** | ⚠️ 有迹象 |
| Claude-v1 | **+25 个百分点** | ⚠️⚠️ 迹象明显 |
| GPT-3.5 | ✅ **不偏袒自己** | ★ 说明这不是普适规律 |

```
    ⟹ 🔑🔑 最实用的一条推论：
       ★★★【不要用 A 家的模型当裁判去评 A 家的模型】。

    ⚠️⚠️ 但请务必别过度解读，★★★ 论文自己把结论收得很紧：
       ✅ "our study **cannot determine** whether the models exhibit a
          self-enhancement bias"
       ⟹ ★★ 也就是说论文【没有断言自我偏好存在】，只是观察到了迹象。
       ⟹ 🔑 所以这是一个【要防范的风险】，不是一条已证实的铁律。

    ⚠️ 还有一个术语精确性问题：★★ 那个 "+10%" 是【胜率的百分点差】
       （win rate 从 x% 变成 x+10%），⚠️ 不是"胜率相对提高了 10%"。
       ⟹ 🔑 百分比和百分点混用，是这一类数字最常见的误传方式。
```

**★★★ 可操作的检测流程（自我偏好）**

```
   ★ 核心工具：轮流当裁判矩阵（round-robin judge matrix）

   ① 取 k 个模型，让它们【既当选手又当裁判】
   ② 填一张 k 行（裁判）× k 列（选手）的表，每格是"该裁判给该选手打出的胜率"
   ③ ★★★ 一定要多加一行：【人类评分】那一行，当作参照基准
   ④ 自我偏好量 = 对角线那一格 − 人类那一行对应的那一格
      ⟹ ✅ 论文就是这么算的（Figure 3(b)：六个模型在不同裁判下的胜率）
```

★ 一张示意表，说明这个矩阵怎么读（⚠️ 下面的数字是我编的示例，★ 用来演示算法，不是实测）：

| 裁判 \ 选手 | 模型 M | 模型 N | 模型 P |
|---|---|---|---|
| **模型 M 当裁判** | ⚠️ **72%** ← 对角线 | 55% | 30% |
| 模型 N 当裁判 | 61% | 58% | 28% |
| 模型 P 当裁判 | 60% | 54% | 33% |
| ★★★ **人类评分** | **60%** | 56% | 31% |

```
   ⟹ ★ 逐列算"自我偏好量"：
      M 自评 72% − 人给 M 的 60% = ⚠️ +12 个百分点   ← 有自我偏好迹象
      N 自评 58% − 人给 N 的 56% = +2 个百分点        ← 可忽略
      P 自评 33% − 人给 P 的 31% = +2 个百分点        ← 可忽略

   ⟹ 🔑🔑 关键在于【必须用人类那一行当基准】，★★★ 不能用"其他裁判的平均"：
      ⚠️ 因为如果 N 和 P 都是同一家的模型、有共同的风格偏好，
         "其他裁判的平均"本身就是偏的。

   ⟹ ★★ 另一个必须做的健壮性检查（✅ 论文提到的难点）：
      去风格化后重判 —— 用第三个中性模型把所有答案改写成统一文风，再判一遍。
      ⚠️ 论文明确说这很难做好："we cannot easily rephrase a response to fit
         the style of another model **without changing the quality**"。
      ⟹ 🔑 所以这条只能当辅助证据，不能当判决。
```

### ④ 附带一个：裁判自己不会做题

✅ Table 4 原文（★ 连表题一起读，因为那个"20"是怎么来的全在表题里）：

> "Table 4: Judge failure rate on **10 math questions** with different prompts.
> We test LLaMA-13B vs. Vicuna-13B and **swap positions**.
> **A failure means when GPT-4 says an incorrect answer is correct.**"

| 提示词 | Default（直接判） | CoT（先自己想一遍再判） | Reference（给参考答案） |
|---|---|---|---|
| 失败次数 | ⚠️⚠️ **14 / 20** | 6 / 20 | ★ **3 / 20** |
| 换成百分比 | **70%** | 30% | **15%** |

```
    ★ 先把那个分母解释清楚：★★★ 20 = 10 道题 × 2（交换位置各判一遍）。
      ⟹ 🔑 所以这张表本身就是"两遍法"的产物，位置偏差已经被摊进去了。

    ⟹ ✅ 论文对这两个端点的原话："we see a significant improvement in failure
       rate (**from 70% to 15%**) over the default prompt."

    ⟹ 🔑 直接裁判：★★★ 二十次里有十四次把错的判成对。
    ⟹ ★ 让裁判先自己想一遍（CoT，Chain-of-Thought 思维链）：降到 6/20。
    ⟹ ★★ 给它一份参考答案：降到 3/20。

    ⚠️⚠️ 两处常见误读，必须纠正：

    ① ❌ 有人把 14/20 说成"比瞎猜还差"。★★★ 这个说法没有意义：
       这一列量的是【把错答案判成对】这一种单向失败的发生率，
       ⚠️ 不是"判对率"，所以没有 50% 这个"瞎猜基线"可比。
       ⟹ 🔑 正确的读法是：★★ "在明知有错答案的 20 次判决里，它放过了 14 次"。

    ② ❌ 有人把 Reference 那一列理解成"把数据集的标准答案给裁判看"。
       ★★★ 在这张表里【不是】。✅ 论文 §3.4 原文：
       > "we propose a reference-guided method, in which we first generate
       >  **LLM judge's answer independently**, and then display it as a
       >  reference answer in the judge prompt."
       ⟹ ★★ 给的是【裁判自己先独立做一遍的答案】（见 §9.2 那段说明）。
       ⟹ 🔑 这一点非常重要，因为它意味着：★★★ 就算你手里没有标准答案，
          也能拿到这 70% → 15% 的改善 —— 只要让裁判先自己答一遍。

    ⚠️ 还有一个诚实的限定：✅ 论文明说即使用了 CoT，
       "in many cases LLM makes exactly the **same mistake** as the given answers
        in its problem-solving process" ⟹ ★★ 裁判会被给定答案带着一起错。
       🔑 这也解释了为什么 Reference 比 CoT 更有效：★★★ 先独立作答再看答案，
          顺序不能颠倒。
```

```
    🔑🔑 结论：★★★ 有客观答案的题，就别让裁判"凭感觉"判。
       ⟹ 能让它先独立答一遍就先答（省不了多少钱，省很多错）；
       ⟹ 能给标准答案就给；能跑测试就跑测试（07 章）。
       ⟹ LLM 裁判是【开放题的无奈之选】，不是万能替代品。
    ⟹ ★ 12 章 §12.3 把这条规则落成了"三档期望输出"的选择流程。
```

---

## 9.5 从"两两对比"到一个排行榜：Elo

★★ 两两对比只能告诉你"A 赢了 B"。要变成一张榜，还差一步。

```
    ★ 名词：Elo 评分（Elo rating）
      由 Arpad Elo 提出，1960 年代起用于国际象棋排名。
      ★★ 核心思想：赢了高分对手加很多分，赢了低分对手加一点点。
      ⟹ 从一堆【两两胜负】推出【每个人的绝对实力分】。
```

### ① ⚠️ 源码里那个 "Elo"，其实不是国际象棋那个 Elo

源码在 [arena_hard_adapter.py:148-166](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py)：

```python
from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column
battles = pd.concat([get_battles_from_row(res.score.metadata['battle_result']) for res in sample_scores])
bootstrap_online_elo = compute_mle_elo(battles)
```

★★★ 但打开 `compute_mle_elo` 看实现（[utils.py:110-153](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/utils.py)，★ 逐字节选）：

```python
def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    ...
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X, Y)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    # set anchor as gpt4-0314 = 1000
    if 'gpt4-0314' in models.index:
        elo_scores += 1000 - elo_scores[models['gpt4-0314']]
```

```
    ⟹ ⚠️⚠️ 它是一个【逻辑回归】，不是国际象棋那种"一局一局往上加分"的算法。
       ★ MLE = Maximum Likelihood Estimation（极大似然估计）。
       ★★ 这套做法的正式名字是 Bradley-Terry 模型（BT 模型，1952 年），
          只是把系数乘以 400 再加 1000，让数值看起来像 Elo 分。

    ⟹ 🔑🔑 为什么要区分这两个？★★★ 因为它们有一个实质差别：
       ┌ 在线 Elo（增量式）：★★ 结果【依赖比赛顺序】。
       │   同一批对局换个播放顺序，最终分数不一样。
       └ BT 极大似然（这里用的）：★★ 结果【与顺序无关】，
           因为它是把所有对局一次性拟合出来的。
    ⟹ ★★ 所以看到"Elo 1247"，要问一句是哪种算法 —— 前者可复现性更差。
```

★ 顺便把那三个常量的含义讲清楚，因为它们决定了"分差多少算差得多"：

```
   SCALE = 400，BASE = 10  ⟹ 期望胜率公式（源码 predict_win_rate，utils.py:166-180）：
       胜率(A 对 B) = 1 / (1 + 10^((R_B − R_A) / 400))
```

| 分差 R_A − R_B | 期望胜率 | 算式 |
|---|---|---|
| 0 | **50.0%** | 1/(1 + 10^0) = 1/2 |
| +100 | **64.0%** | 1/(1 + 10^(−0.25)) = 1/1.5623 |
| +200 | **76.0%** | 1/(1 + 10^(−0.5)) = 1/1.3162 |
| +400 | **90.9%** | 1/(1 + 10^(−1)) = 1/1.1 |
| −100 | 36.0% | 对称 |

```
    ⟹ 🔑 拿这张表去读榜单：★★★【分差 100 分 ≈ 六四开】，
       ⚠️ 不是"强一倍"。看到两个模型差 20 分就说"明显更强"的，
       换算过来是 52.9% 对 47.1% —— ★★ 基本等于打平。
```

### ② ⚠️ 平局、抽取失败、"明显胜"，在拟合前被怎么处理

★★★ 这三条藏在 `get_battles_from_row`（[utils.py:56-107](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/utils.py)）里，每一条都是一个口径：

| 裁判输出 | 进入 BT 拟合时算几场 | ⚠️ 含义 |
|---|---|---|
| `A>B` / `B>A`（略胜） | **1 场** | 正常 |
| `A>>B` / `B>>A`（明显胜） | ★★ **3 场**（`multiplier=3`） | ★★★ "明显胜"被【当成赢了三局】，权重是略胜的 3 倍 |
| `A=B`（平局） | 1 场，标记为 tie | ★ 后续在 `compute_mle_elo` 里被拆成"一次 A 赢 + 一次 B 赢"（源码把对局表复制了一份，L114-131） |
| ⚠️ **抽取失败** | ★★★ **0 场，直接丢弃** | ⚠️⚠️ 见下 |

```
    ⚠️⚠️ 盯住最后一行。源码 utils.py:74-78：
        else:
            weight = 0
        if weight:
            results += [output] * weight
    ⟹ ★★★ 裁判输出里正则抽不出 [[A>B]] 这类标记时，
       这一局【从对局表里消失了】，既不算平局也不算失败。

    ⟹ 🔑🔑 这正是 05 章那条铁律在主观评测里的重演：
       ★★ 抽取失败被静默吞掉 ⟹ 样本量悄悄变小，而报告里看不出来。
    ⟹ ⚠️ 对比一下 07 章的做法：SWE-bench 给"补丁打不进去"立了一个
       专门的标记 APPLY_PATCH_FAIL。★★★ 那是对的做法，这里不是。
    ⟹ ★★ 你自己搭裁判评测时，务必把"抽取失败率"作为一个独立指标报出来。

    ⚠️ 注意还有一处不一致：同一份源码里，per-sample 那条路径
       （get_judge_score，utils.py:51）对抽取失败是【记 0.5 分】，
       而进入 Elo 拟合这条路径（get_battles_from_row）是【直接丢弃】。
       ⟹ 🔑 同一个框架，两条路径，两种处理方式。★★ 这就是为什么口径要逐条问。
```

### ③ ⚠️ 置信区间：bootstrap 写了，但不一定跑

```
    ★ 名词：bootstrap（自助法）
      ★★ 一种统计方法：把数据【有放回地重复抽样】很多次，
      每次算一遍分数，看这些分数散得有多开。
      ⟹ 🔑 用来算【置信区间】——也就是"这个分数上下浮动多少算正常"。
```

★★★ 但源码里的实情值得看清楚，这是个很好的教学案例：

| 适配器 | `get_bootstrap_result` | 实际报出来的指标 | 有没有区间 |
|---|---|---|---|
| **arena_hard** | ⚠️⚠️ L151 **导入了，但全文没有调用** | 一个 `winrate`（[adapter L166-172](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py)） | ❌ **没有** |
| **general_arena** | ✅ L314 真的调用了，`num_round=100` | `winrate` / `winrate_lower` / `winrate_upper`（[adapter L325-331](../源码/evalscope-1.10.0/evalscope/benchmarks/general_arena/general_arena_adapter.py)） | ✅ **95% 区间**（2.5 / 97.5 分位） |

✅ general_arena 那三行（★ 逐字）：

```python
stats.at[i, 'lower'] = np.percentile(bootstrap_model_coef[model], 2.5)
stats.at[i, 'upper'] = np.percentile(bootstrap_model_coef[model], 97.5)
...
metrics_dict['winrate']       = get_win_rate_column(stats, 'score', self.baseline).to_dict()
metrics_dict['winrate_lower'] = get_win_rate_column(stats, 'lower', self.baseline).to_dict()
metrics_dict['winrate_upper'] = get_win_rate_column(stats, 'upper', self.baseline).to_dict()
```

```
    ⟹ 🔑🔑 所以正经的主观榜单，★★★ 报的应该是
       "1247 ± 8" 或 "胜率 58.3%（95% 区间 54.1%–62.6%）"，
       ⚠️ 而不是光秃秃一个 "1247" 或 "58.3%"。
       ⚠️ 两个模型分差 3 分、区间重叠 20 分 ⟹ 【实际上没分出胜负】（11 章 §11.6）。

    ⟹ ⚠️ 但从上面这张表你也看到了：★★★【框架里有这个能力，不代表这次跑了】。
       同一个 EvalScope，两个适配器，一个报区间一个不报。
    ⟹ 🔑 所以看到一个没有区间的主观分数，第一反应不该是"他们不懂"，
       ★★ 而是"这条代码路径可能压根没算"。⟹ 你自己跑的时候要主动确认。
```

### ④ ★★ 锚点模型：省了 N² 场，代价是整张榜绑死

★★ 还要注意 `arena_hard` 源码里的这两行（[adapter L121-122](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py)）：

```python
'model_a': 'gpt4-0314',
'model_b': 'test_model',
```

★ 以及 `compute_mle_elo` 结尾那两行（utils.py:150-152）：

```python
# set anchor as gpt4-0314 = 1000
if 'gpt4-0314' in models.index:
    elo_scores += 1000 - elo_scores[models['gpt4-0314']]
```

```
    ⟹ 🔑 它不是让所有模型互相打，而是【全部去挑战同一个固定基线】。
       ★★ 这叫【锚点模型】(anchor / baseline model)。

    ⟹ ★★★ 好处：新模型加进来只要打 N 场，不是 N² 场（解决了 §9.2 ① 的爆炸问题）。
    ⟹ ⚠️ 代价一：★★ 整张榜的刻度绑死在这个锚点上。锚点一换，所有历史分数【全部不可比】。
    ⟹ ⚠️ 代价二：★★★ 如果参赛模型已经明显强过锚点（比如都赢 90%），
       ★ 那么它们彼此之间的差距会被压缩到看不出来 —— 因为大家都在打同一个沙包。
       🔑 这就是为什么主观榜单每隔一段时间就要换一次锚点，而换完就得从头积累历史。
```

---

## 9.6 裁判评测的复现性：比你想的更差

★★ 把 01 章的"分数是联合产物"套到这里，变量列表变得很长。★★★ 下面每一项，换一个值就换一张榜：

| # | 变量 | 换掉它会怎样 | ✅ 证据 / 出处 |
|---|---|---|---|
| ① | ★★ **裁判模型是哪个** | 换裁判 = 换尺子 | ✅ Table 3：同一个注水攻击，Claude-v1 上成功 91.3%，GPT-4 上只有 8.7% |
| ② | ★★★ **裁判模型的版本 / 日期** | ⚠️ 闭源 API 会静默更新，历史分数无声失效 | ⚠️ 我的判断，但 10 章 §10.5 ④ 有一线团队因此固定 API 快照的做法 |
| ③ | ★ **裁判的提示词** | 一致率可以翻倍 | ✅ Table 2：Claude-v1 default 23.8% → rename 56.2% |
| ④ | ★ **给不给示范（few-shot）** | 一致率 65.0% → 77.5%，⚠️ 但成本 ×4 且不保证更准 | ✅ 论文 §3.4 + Table 12 |
| ⑤ | ★★ **判一遍还是两遍** | 位置偏差有没有被抵消 | ✅ §3.4 的保守法 / Arena-Hard 的平均法 |
| ⑥ | ★★ **两遍怎么合** | 保守法（矛盾→平局）和平均法（矛盾→0.5）得出不同的榜 | ✅ 论文 §3.4 vs [arena_hard_adapter.py:138](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py) |
| ⑦ | ★ **裁判的 temperature** | 非 0 就逐次不可复现 | ★ 常识；⚠️ 但 10 章 §10.7b 说明"逐次可复现"未必是最该追求的那种 |
| ⑧ | ★★ **锚点模型是谁** | 换锚点整张榜作废 | ✅ `compute_mle_elo` 把 gpt4-0314 钉死在 1000 |
| ⑨ | ★ **平局怎么算分** | 0.5 分、拆成一胜一负、还是不计入 | ✅ 源码：拆成"一次 A 赢 + 一次 B 赢" |
| ⑩ | ⚠️⚠️ **"明显胜"的权重** | `multiplier=3`，改成 1 或 2 就是另一张榜 | ✅ [utils.py:56](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/utils.py) |
| ⑪ | ⚠️⚠️ **抽取失败怎么算** | 丢弃 / 记 0.5 / 记为平局，三种都能见到 | ✅ 同一份源码里两条路径两种做法（§9.5 ②） |
| ⑫ | ★ **bootstrap 跑不跑、跑几轮** | 决定有没有置信区间 | ✅ arena_hard 不跑，general_arena 跑 100 轮 |

★★★ 最后一个坑，是我核对源码时撞上的，值得单独拎出来：

```
    ⚠️⚠️ 同一个 EvalScope 里，【文档说的默认裁判】和【代码里的默认裁判】不是一个：

    ✅ arena_hard_adapter.py:49（文档字符串）：
         "Uses LLM judge (default: gpt-4-1106-preview)"
    ✅ metrics/judge/llm_judge.py:43（真正生效的代码）：
         DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
    ✅ 同文件 L87 的取值顺序：
         self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)

    ⟹ 🔑🔑 也就是说：★★★ 你不显式指定的话，用的是 Qwen3-235B-A22B，
       ⚠️ 而适配器的文档字符串还停留在原始 Arena-Hard 论文的 GPT-4 设定上。
    ⟹ ★★ 这不是谁的错——文档字符串描述的是这个基准的【原始定义】，
       代码里的是这个框架的【默认实现】。⚠️ 但两者不同，而分数只认后者。
    ⟹ ★★★ 结论：报主观分数时，★ 裁判模型要从【运行日志】里读，
       不要从文档或论文里抄。
```

```
    ⟹ 🔑 EvalScope 至少有一点做得好：★★ 默认裁判是【写死在源码里、有名有姓】的。
       ⚠️ 很多榜单只说"用了 LLM 裁判"，不说是谁、哪个版本。那种分数没法用。
```

---

## 9.7 什么时候该用、什么时候不该用

| ✅ 适合 | 为什么 |
|---|---|
| ① 开放题：写作、总结、角色扮演、多轮对话 | ★ 没有可跑的测试，也算不了相似度 |
| ② ★★ 相对比较（A 和 B 哪个好），而不是绝对分 | ✅ 论文：单答案打分 "may become unstable ... if the judge model changes" |
| ③ ★ 快速迭代：你改了提示词，想知道有没有变好 | ★★ 同一个裁判、同一天、同一批题 ⟹ 变量只剩你改的那一项 |
| ④ 大规模粗筛：先用裁判筛掉明显差的，再人工看剩下的 | ★ 把人的时间用在裁判拿不准的那一段 |

| ❌ 不适合 | 为什么 |
|---|---|
| ① ★★★ 有客观答案的（数学 / 代码） | ✅ Table 4 已证明：直判 14/20 把错的判成对 |
| ② ★★ 需要跨时间比较的 | ⚠️ 裁判会升级，刻度会漂；锚点会换，历史全废（§9.5 ④） |
| ③ ★★ 评自家模型用自家裁判 | ⚠️ 自我偏好（+10 ~ +25 个百分点的迹象，§9.4 ③） |
| ④ ★★★ 要拿去对外宣称"我们比 X 强"的场合 | ⟹ 对方可以用另一个裁判得出相反结论 |
| ⑤ ★★ 题目分布一边倒的场景（强模型 vs 弱模型） | ⚠️ 此时"一致率"几乎不含信息，κ 会掉到 0.05 那一档（§9.3b ①） |

```
    ★★★ 一条能立刻用上的自检：
    在你说出"我们的裁判和人的一致率有 XX%"之前，先回答这三个问题——
      ① 这个 XX% 是把平局算进去的（S1）还是排除掉的（S2）？
      ② 对应的 kappa 是多少？两边的边缘分布一边倒吗？
      ③ 判了一遍还是两遍？抽取失败的那些题去哪了？
    ⟹ 🔑 三个都答不上来，这个数字就不该出现在任何对外材料里。
```

---

## 9.8 本章总结

| 关键数字 | 含义 | 出处 |
|---|---|---|
| 85% / 81% | GPT-4 对人 / 人对人的一致率（★★ **S2 口径，只算非平局**） | ✅ MT-Bench Table 5 |
| 66% / 63% | 同上，⚠️ **S1 口径（含平局与自相矛盾）** | ✅ MT-Bench Table 5 |
| κ ≈ 0.70 / 0.62 | 上面 85% / 81% 换算出的 kappa（⚠️ 我的换算） | ⚠️ 由 p_o 与论文给的 R 推算 |
| 23.8% → 56.2% | Claude-v1 一致率，default → rename ⟹ ★★ 是**名字**偏见 | ✅ Table 2 |
| 91.3% / 8.7% | 注水攻击成功率：Claude-v1、GPT-3.5 / GPT-4（23 条样本） | ✅ Table 3 |
| +10 / +25 个百分点 | GPT-4 / Claude-v1 自评胜率高出人评多少 ⚠️ 论文说证据不足 | ✅ §3.3 |
| 14/20 → 6/20 → 3/20 | 数学题误判次数：直判 / CoT / 先自答再判 | ✅ Table 4 |
| 65.0% → 77.5% | few-shot 提升一致率，⚠️ 代价是成本 ×4 且不保证更准 | ✅ §3.4 |
| 100 分 ≈ 64% 胜率 | Elo 分差与期望胜率的换算 | ✅ 源码 `predict_win_rate` |

```
    🔑 九条带走：

    ① ★★ 开放题没有可跑的测试，也没法算相似度 ⟹ 只能让模型判
    ② ★ 三种形态：两两对比 / 单答案打分 / 带参考答案打分
       ⚠️ 第三种的 "reference" 有两个意思：裁判自己先答的 vs 数据集的标准答案
    ③ ✅ GPT-4 与人的一致率 >80%，★★ 和人与人之间同级 ——
       ⚠️ 但那是【排除平局】的 S2 口径，含平局只有 66%
    ④ ★★★ 一致率会骗人：★★ 同样 80% 的一致率，kappa 可能是 0.60，也可能是 0.05。
       ⟹ 🔑 报一致率必须同时报 kappa（或同类去偶然化系数）和边缘分布
    ⑤ ★★★ 三大偏差：位置（Claude-v1 一致率仅 23.8%，⚠️ 且其实是名字偏见）、
       冗长（注水攻击成功率 91.3%）、自我偏好（+10~+25 个百分点，⚠️ 论文未定论）
    ⑥ ★★ 位置偏差有解：交换顺序判两遍。⚠️ 但"两遍怎么合"有保守法和平均法两种，
       结果不同，引用时要说清
    ⑦ ★★★ 有客观答案就别用裁判：直判 70% 误判，让裁判先自己答一遍降到 15%
    ⑧ 🔑🔑 Elo + bootstrap ⟹ 看榜要看【置信区间】，区间重叠就是没分出胜负；
       ⚠️ 而且源码里的 "Elo" 是 Bradley-Terry 极大似然，不是增量式 Elo；
       ⚠️ 还有一个适配器压根没跑 bootstrap
    ⑨ ⚠️⚠️ 抽取失败的判决可能被【静默丢弃】（源码实证）。
       ⟹ 🔑 05 章那条铁律在主观评测里同样成立：★★★ 失败要单独记，不能吞

    ⟹ 🔑🔑🔑 一句话立场：
       ★★★ LLM 裁判是【便宜、可扩展、有解释】的近似人类偏好，
       但它是【近似】。★★ 用它做迭代信号很好，用它做对外结论很危险。
```

---

> 下一章：[10-DeepSeek的评测口径.md](10-DeepSeek的评测口径.md) —— ⭐⭐ 拿一份真实的技术报告，看一线团队到底怎么设置 harness，以及为什么同一个模型在不同口径下能差十几分。
> 框架侧的裁判源码逐行拆解：[OpenCompass与EvalScope拆解.md](../OpenCompass与EvalScope拆解.md)
> 返回 [评测入门总目录](README.md)
