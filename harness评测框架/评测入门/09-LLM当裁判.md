# 09 · 让模型当裁判：LLM-as-a-Judge

> ★ 07/08 章讲的代码题能"跑测试"来判。
> ★★ 但"这封邮件写得好不好""这个解释清不清楚"没有测试可跑。
> ⟹ 🔑 于是有了一个看起来像作弊、实际却是当前主流的办法：**让另一个模型来打分**。

---

## 9.1 为什么非得这么干

```
    ★ 名词：LLM-as-a-Judge（大模型当裁判）
      用一个强模型去给另一个模型的回答打分或排序。
```

✅ 论文原文（MT-Bench，arXiv:2306.05685，NeurIPS 2023，§3）：

> "collecting human preferences can be **costly and laborious**. ...
> Given that most questions in MT-bench and Chatbot Arena are **open-ended
> without reference answers**, devising a **rule-based program** to assess the
> outputs is **extremely challenging**. Traditional evaluation metrics based on
> the similarity between outputs and reference answers (e.g., ROUGE, BLEU)
> are also **ineffective** for these questions."

```
    ⟹ 🔑 三条路，全堵死了：
    ┌──────────────────┬────────────────────────────────────┐
    │ ① 人来打分        │ ★ 最准，但慢且贵，无法天天跑       │
    │ ② 写规则/正则     │ ★★ 开放题根本没有"标准答案"可比    │
    │ ③ 和参考答案算相似度│ ★★ BLEU/ROUGE 只看字面重合，       │
    │                   │   ⚠️ 换个说法但意思一样 → 判低分   │
    └──────────────────┴────────────────────────────────────┘
    ⟹ ★★★ 剩下的只有 ④：让模型判。
```

```
    ★ 名词：BLEU / ROUGE
      两个上古（2002/2004 年）自动指标，本质是【统计你的输出和参考答案
      有多少词重合】。★★ 对翻译/摘要还行，对开放问答基本失效。
```

★★ 论文还指出一个更根本的动机（§1）：

> "benchmarks like **MMLU and HELM cannot effectively tell the difference
> between these aligned models and the base models**."

```
    ⟹ 🔑🔑 这句话分量很重：★★★ 做完对齐（RLHF）后用户明显更喜欢的模型，
       在 MMLU 这类选择题榜上【看不出提升】。
       ⟹ 说明 01-06 章那套判别式评测，
          ★★ 测不出"好不好用"这个维度。这是 LLM 裁判存在的根本理由。
```

---

## 9.2 三种裁判形态

✅ 论文 §3.1 明确列了三种：

```
   ┌─ ① Pairwise comparison（两两对比）─────────────────┐
   │  给裁判一个问题 + 两份答案 → 判谁更好，或平局。     │
   │  ★★ 最稳，因为"比较"比"打绝对分"容易。            │
   │  ⚠️ 缺点：✅ "the number of possible pairs grows    │
   │     quadratically" —— ★ 模型一多，对数爆炸。       │
   ├─ ② Single answer grading（单答案打分）─────────────┤
   │  直接给一份答案打个分（比如 1-10）。               │
   │  ★ 便宜、线性。⚠️ 论文说它 "may become unstable"： │
   │     ★★ 换个裁判模型，绝对分会整体漂移。           │
   ├─ ③ Reference-guided grading（带参考答案打分）──────┤
   │  ★★ 把标准答案也给裁判看，再让它判。              │
   │  ⟹ 数学题上效果显著（见 §9.4）                     │
   └────────────────────────────────────────────────────┘
```

★ 这三种在源码里都能看到。比如 EvalScope 的默认裁判模板
[metrics/judge/llm_judge.py:12](../源码/evalscope-1.10.0/evalscope/metrics/judge/llm_judge.py)
就是第 ③ 种（★ 逐字，节选）：

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

同文件里还有第 ② 种模板 `DEFAULT_NUMERIC_SCORE_TEMPLATE`（★ 逐字节选）：

> "you must rate the response on a scale of 0 (worst) to 1 (best) by strictly
> following this format: `\"[[rating]]\"`, for example: `\"Rating: [[0.5]]\"`"

```
    ★★ 那个双方括号 [[ ]] 不是装饰，是【给正则用的锚点】。
    ⟹ 🔑 因为裁判会写一大段解释，必须有个不会误命中的标记把分数框出来。
```

---

## 9.3 裁判靠谱吗？—— 靠谱到出人意料

✅ 论文摘要：

> "strong LLM judges like **GPT-4 can match both controlled and crowdsourced
> human preferences well, achieving **over 80% agreement**, the same level of
> agreement between humans."

```
    ⟹ 🔑🔑 最后半句是关键：★★★【人和人之间的一致率也就 80% 左右】。

    ★★ 这意味着：不能因为"裁判和人不完全一致"就否定它，
       因为【人和人本来也不完全一致】。
       ⟹ 开放题的"正确答案"本身就不唯一。

    ⚠️ 但请注意时代：这是 2023 年的 GPT-4。
       ★ 今天的裁判模型只会更强，⚠️ 不过下面那些偏差【并没有消失】。
```

---

## 9.4 三个必须知道的偏差

★★★ 这是本章最有价值的部分。论文 §3.3 用实验把它们量化了。

### ① 位置偏差（position bias）

✅ 定义：*"an LLM exhibits a propensity to favor certain positions over others"*

✅ Table 2（★ 三个裁判，默认提示词）：

```
   ┌───────────┬──────────┬────────────┬────────────┐
   │ 裁判       │ 一致率    │ 偏向第一个  │ 偏向第二个  │
   ├───────────┼──────────┼────────────┼────────────┤
   │ Claude-v1  │  23.8%   │ ★★ 75.0%  │   0.0%     │
   │ GPT-3.5    │  46.2%   │   50.0%    │   1.2%     │
   │ GPT-4      │ ★ 65.0%  │   30.0%    │   5.0%     │
   └───────────┴──────────┴────────────┴────────────┘

   ★ "一致率" = 把两份答案【调换顺序】后，判决不变的比例。
```

```
    ⟹ 🔑🔑 读懂这张表：Claude-v1 只有 23.8% 的情况下judgment 不随顺序变，
       ★★★ 也就是说【四次里有三次，谁放前面谁赢】。
       ⟹ 那一次评测得到的排名，★★ 相当大一部分是【摆放顺序造成的】。

    ⚠️ 论文自己说明这个测试很苛刻：两份答案是同一个模型
       temperature=0.7 采两次生成的，本来就极难分辨。
       ✅ "occasionally indistinguishable even to humans"
       ⟹ ★ 所以别把 23.8% 理解成"Claude 判什么都乱"。
```

★★ 论文对成因的猜测（✅ 逐字）：

> "we suspect that it could be rooted in the training data or **inherent to the
> left-to-right architecture of causal transformers**"

```
    ★ 名词：causal transformer（因果 Transformer）
      ★★ 就是现在所有主流大模型的结构：读文本时【只能看左边，不能看右边】。
      ⟹ 于是"先读到的东西"天然占据了更多的推理路径。
      ⚠️ 这只是论文的猜测，未被证明。
```

**★★★ 怎么治：跑两遍，交换位置，取平均。**

这在源码里是标准做法。EvalScope 的 Arena-Hard 适配器
[arena_hard_adapter.py:41](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py)
文档字符串直接写着（★ 逐字）：

```
- Two-game battle system (A vs B and B vs A)
```

实现在同文件 L110-138：

```python
game1_response = self.llm_judge.judge(prompt1, system_prompt=GRADER_SYSTEM_PROMPT)
game2_response = self.llm_judge.judge(prompt2, system_prompt=GRADER_SYSTEM_PROMPT)
...
score1 = get_judge_score(res1, reverse=True)     # ★ 注意 reverse
score2 = get_judge_score(res2, reverse=False)
...
score.value = {'score': (score1 + score2) / 2}   # ★★ 两局取平均
```

```
    ⟹ 🔑 一句话：★★★ 一道题判两次，成本翻倍，换掉一个系统性偏差。
       ⚠️ 看到某个主观榜单【只判一遍】，它的结果就要打折看。
```

### ② 冗长偏差（verbosity bias）

✅ 定义：*"an LLM judge favors longer, verbose responses, even if they are not
as clear, high-quality, or accurate as shorter alternatives"*

★★ 论文设计了一个很妙的攻击实验："repetitive list" attack ——
把一个 5 条的列表，让 GPT-4 换个说法复述成 5 条，拼在原列表前面，
★ 变成 10 条但**信息量完全没增加**。

✅ Table 3（23 个答案上的"攻击成功率"）：

```
   ┌───────────┬───────────┬─────────┐
   │ Claude-v1 │  GPT-3.5  │  GPT-4  │
   │ ★★ 91.3% │ ★★ 91.3% │  8.7%   │
   └───────────┴───────────┴─────────┘
```

```
    ⟹ 🔑🔑 读法：★★★ 把答案注水一倍、不加任何新信息，
       有两个裁判【九成以上会认为变好了】。

    ⟹ ★★ 这直接解释了一个你一定见过的现象：
       为什么很多模型答什么都要先来一段"这是一个很好的问题"，
       然后分点、加粗、总结、再展望。
       ⟹ 🔑 因为在【被冗长偏差污染的偏好数据】上训练过。

    ⚠️ 这是我的推断链，不是论文结论。但方向是清楚的：
       ★★★ 评测的偏差会顺着训练回流到模型行为里。
```

★ 论文还做了个对照：裁判对**完全相同**的两个答案总是判平局
（✅ "always return a tie for two identical answers"），
⟹ ★★ 说明它不是乱判，是【真的被"更长"骗到了】。

**怎么治：** ⚠️ 论文没给通用解法。工程上常见的是
"控制长度偏差"的评测变体——OpenCompass 里就专门有一个数据集叫
[compassbench_control_length_bias.py](../源码/opencompass-0.5.3/opencompass/datasets/subjective/compassbench_control_length_bias.py)。
★ 一个偏差重要到要为它单独做一套题，说明它有多难缠。

### ③ 自我偏好（self-enhancement bias）

✅ 定义：*"LLM judges may favor the answers generated by themselves"*

✅ 论文 §3.3 实测：

> "GPT-4 favors itself with a **10% higher win rate**;
> Claude-v1 favors itself with a **25% higher win rate**.
> However, they also favor other models and **GPT-3.5 does not favor itself**."

```
    ⟹ 🔑🔑 最实用的一条推论：
       ★★★【不要用 A 家的模型当裁判去评 A 家的模型】。

    ⚠️ 但也别过度解读：论文自己说 GPT-3.5 就不偏袒自己，
       ✅ 而且明确写了 "Due to limited data" ——★ 样本量有限。
       ⟹ 这是一个【要防范的风险】，不是一条铁律。
```

### ④ 附带一个：裁判自己不会做题

✅ Table 4（10 道数学题，"裁判把错答案判成对"的失败次数）：

```
   ┌──────────┬─────────┬────────────┐
   │ Default  │  CoT    │ Reference  │
   │ ★★ 14/20│  6/20   │  ★ 3/20    │
   └──────────┴─────────┴────────────┘
```

```
    ⟹ 🔑 直接裁判：★★★ 二十次里有十四次把错的判成对——比瞎猜还差。
    ⟹ ★ 让裁判先自己想一遍（CoT）：降到 6/20。
    ⟹ ★★ 把标准答案给它看（Reference）：降到 3/20。

    🔑🔑 结论：★★★ 有客观答案的题，就别让裁判"凭感觉"判。
       能给参考答案就给，能跑测试就跑测试（07 章）。
       ⟹ LLM 裁判是【开放题的无奈之选】，不是万能替代品。
```

---

## 9.5 从"两两对比"到一个排行榜：Elo

★★ 两两对比只能告诉你"A 赢了 B"。要变成一张榜，还差一步。

```
    ★ 名词：Elo 评分（Elo rating）
      来自国际象棋（1960 年代，Arpad Elo 发明）。
      ★★ 核心思想：赢了高分对手加很多分，赢了低分对手加一点点。
      ⟹ 从一堆【两两胜负】推出【每个人的绝对实力分】。
```

源码里就是这么做的，[arena_hard_adapter.py:148-158](../源码/evalscope-1.10.0/evalscope/benchmarks/arena_hard/arena_hard_adapter.py)：

```python
from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column
battles = pd.concat([get_battles_from_row(res.score.metadata['battle_result']) for res in sample_scores])
bootstrap_online_elo = compute_mle_elo(battles)
```

```
    ★ 名词：bootstrap（自助法）
      ★★ 一种统计方法：把数据【有放回地重复抽样】很多次，
      每次算一遍分数，看这些分数散得有多开。
      ⟹ 🔑 用来算【置信区间】——也就是"这个分数上下浮动多少算正常"。

    ⟹ 🔑🔑 所以正经的主观榜单，★★★ 报的应该是
       "1247 ± 8" 而不是光秃秃一个 "1247"。
       ⚠️ 两个模型分差 3 分、区间重叠 20 分 ⟹ 【实际上没分出胜负】。
```

★★ 还要注意 arena_hard 源码里的这两行：

```python
'model_a': 'gpt4-0314',
'model_b': 'test_model',
```

```
    ⟹ 🔑 它不是让所有模型互相打，而是【全部去挑战同一个固定基线】。
       ★★ 这叫【锚点模型】(anchor / baseline model)。

    ⟹ ★★★ 好处：新模型加进来只要打 N 场，不是 N² 场（解决了 §9.2 的爆炸问题）。
    ⟹ ⚠️ 代价：★★ 整张榜的刻度绑死在这个锚点上。
       锚点一换，所有历史分数【全部不可比】。
```

---

## 9.6 裁判评测的复现性：比你想的更差

★★ 把 01 章的"分数是联合产物"套到这里，变量列表变得很长：

```
   ┌───────────────────────┬──────────────────────────────────┐
   │ ★★ 裁判模型是哪个？   │ 换裁判 = 换尺子                   │
   │ ★★★ 裁判模型的版本？  │ 闭源 API 会静默更新              │
   │ ★ 裁判的提示词？       │ Table 2 里 default vs rename     │
   │                        │ 就让一致率从 23.8% → 56.2%       │
   │ ★★ 判了一遍还是两遍？  │ 位置偏差有没有被抵消              │
   │ ★ 裁判的 temperature？ │ 非 0 就不可复现                   │
   │ ★★ 锚点模型是谁？      │ 换锚点整张榜作废                  │
   │ ★ 平局怎么算分？       │ 0.5 分还是不计入                  │
   └───────────────────────┴──────────────────────────────────┘

    ⟹ 🔑🔑 EvalScope 默认裁判是写死在源码里的：
       DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
       ★ 这是好事——★★ 至少它【明确写出来了】。
       ⚠️ 很多榜单只说"用了 LLM 裁判"，不说是谁。那种分数没法用。
```

---

## 9.7 什么时候该用、什么时候不该用

```
   ┌─ ✅ 适合 ────────────────────────────────────────────┐
   │ ① 开放题：写作、总结、角色扮演、多轮对话             │
   │ ② ★★ 相对比较（A 和 B 哪个好），而不是绝对分        │
   │ ③ ★ 快速迭代：你改了提示词，想知道有没有变好         │
   │ ④ 大规模粗筛：先用裁判筛掉明显差的，再人工看剩下的   │
   └──────────────────────────────────────────────────────┘
   ┌─ ❌ 不适合 ──────────────────────────────────────────┐
   │ ① ★★★ 有客观答案的（数学/代码）—— Table 4 已证明   │
   │ ② ★★ 需要跨时间比较的（裁判会升级，刻度会漂）       │
   │ ③ ★★ 评自家模型用自家裁判（自我偏好）               │
   │ ④ ★★★ 要拿去对外宣称"我们比 X 强"的场合            │
   │    ⟹ 因为对方可以用另一个裁判得出相反结论           │
   └──────────────────────────────────────────────────────┘
```

---

## 9.8 本章总结

```
    🔑 七条带走：

    ① ★★ 开放题没有可跑的测试，也没法算相似度 ⟹ 只能让模型判
    ② ★ 三种形态：两两对比 / 单答案打分 / 带参考答案打分
    ③ ✅ GPT-4 与人的一致率 >80%，★★ 和人与人之间的一致率同级
    ④ ★★★ 三大偏差：位置（Claude-v1 一致率仅 23.8%）、
       冗长（注水攻击成功率 91.3%）、自我偏好（+10%~+25%）
    ⑤ ★★ 位置偏差有解：交换顺序判两遍取平均（源码是标准做法）
    ⑥ ★★★ 有客观答案就别用裁判：直判 14/20 判错，给参考答案降到 3/20
    ⑦ 🔑🔑 Elo + bootstrap ⟹ 看榜要看【置信区间】，
       区间重叠就是没分出胜负；换锚点则整张榜作废

    ⟹ 🔑🔑🔑 一句话立场：
       ★★★ LLM 裁判是【便宜、可扩展、有解释】的近似人类偏好，
       但它是【近似】。★★ 用它做迭代信号很好，用它做对外结论很危险。
```

---

> 下一章：[10-DeepSeek的评测口径.md](10-DeepSeek的评测口径.md) —— ⭐⭐ 拿一份真实的技术报告，看一线团队到底怎么设置 harness，以及为什么同一个模型在不同口径下能差十几分。
> 返回 [评测入门总目录](README.md)
