# OpenCompass 与 EvalScope 拆解

> ★ 版本：`opencompass` 0.5.3、`evalscope` 1.10.0
> （源码在 [源码/opencompass-0.5.3/](源码/opencompass-0.5.3/)、[源码/evalscope-1.10.0/](源码/evalscope-1.10.0/)）
>
> ★★ 定位：两个**中国团队做的**评测框架。
> ⚠️ 前置阅读：[lm-eval-harness拆解.md](lm-eval-harness拆解.md)。
> ★★★ 这一篇的价值不在于"多认识两个工具"，
> 而在于：**同一个问题，三份不同的答卷，能让你看清设计权衡在哪。**
>
> 标记：✅ = 源码可查证 ｜ ⚠️ = 我的推断 ｜ ★ = 重点 ｜ 🔑 = 关键结论

---

## 0. 先说结论：三个框架各自解决什么问题

```
   ┌────────────────┬──────────────────────────────────────────────┐
   │ lm-eval-harness│ ★★ 核心问题：【模型接口太多样】               │
   │                │ ⟹ 解法：三个原语的窄腰                       │
   │                │ ⟹ 强项：判别式（loglikelihood）评测的事实标准 │
   ├────────────────┼──────────────────────────────────────────────┤
   │ OpenCompass    │ ★★★ 核心问题：【一次要跑几十个模型 × 上百个   │
   │  （司南）      │    数据集，单机跑不完】                       │
   │                │ ⟹ 解法：切分 + 调度（Partitioner + Runner）  │
   │                │ ⟹ 强项：集群规模化、主观评测（LLM 裁判）      │
   ├────────────────┼──────────────────────────────────────────────┤
   │ EvalScope      │ ★★★ 核心问题：【评测不只是准确率，            │
   │  （魔搭）      │    还得知道快不快、贵不贵】                   │
   │                │ ⟹ 解法：把"能力评测"和"性能压测"放进同一个包  │
   │                │ ⟹ 强项：Arena 类主观评测 + 推理服务压测       │
   └────────────────┴──────────────────────────────────────────────┘

   ⟹ 🔑 一句话：
      ★★★ lm_eval 解决【怎么问】，OpenCompass 解决【怎么排产】，
           EvalScope 解决【怎么判 + 怎么压测】。
```

---

# 第一部分：OpenCompass

## 1. ★★★ 它的核心抽象：Partitioner → Runner → Task

★★ 这是 OpenCompass 和 lm_eval 最根本的架构差异。

```
   ⚠️ 先想清楚它要解决的场景：

   ★ 你有 20 个模型要评，每个模型评 100 个数据集。
   ⟹ 20 × 100 = 2,000 个「模型-数据集」组合。
   ⟹ ⚠️ 用 lm_eval 的话，你得写一个 for 循环跑 2,000 次，
      而且【一次崩了就得从头再来】。

   ★★★ OpenCompass 的答案是：把评测当成【一批可调度的作业】。
```

```
   ┌─────────────────────────────────────────────────────────────┐
   │  ① Partitioner（切分器）                                    │
   │     ✅ opencompass/partitioners/                            │
   │     ★★ 职责：把「模型 × 数据集」的大矩阵，切成一堆小任务    │
   │     ⟹ 输入：models 配置 + datasets 配置                     │
   │     ⟹ 输出：一个 task 列表（每个 task 是一个 dict）         │
   ├─────────────────────────────────────────────────────────────┤
   │  ② Runner（执行器）                                         │
   │     ✅ opencompass/runners/                                 │
   │     ★★ 职责：把这些 task【投递到某种算力上】去跑            │
   │     ⟹ 本机多进程？Slurm 集群？阿里云 DLC？火山引擎？        │
   ├─────────────────────────────────────────────────────────────┤
   │  ③ Task（任务）                                             │
   │     ✅ opencompass/tasks/                                   │
   │     ★★ 职责：一个 task 内部【真正干活】                     │
   │     ⟹ OpenICLInferTask = 推理，OpenICLEvalTask = 打分       │
   └─────────────────────────────────────────────────────────────┘

   ⟹ 🔑🔑 这三层的关键在于【解耦】：
      ★★★ 换集群只需要换 Runner，不用动任何评测逻辑。
```

### ✅ Runner 的全部实现（`opencompass/runners/` 目录）

```
   __init__.py   base.py
   local.py            ← ★ 本机多进程
   local_api.py        ← ★ 本机跑 API 模型（不占 GPU）
   slurm.py            ← ★★ Slurm 集群（学术界/超算中心标配）
   slurm_sequential.py ← ★ Slurm 串行版
   dlc.py              ← ★★ 阿里云 DLC（深度学习容器）
   volc.py             ← ★★ 火山引擎
   rjob.py             ← ★ 内部调度平台
```

```
   ★ 名词：
   ★ Slurm = 一种集群作业调度系统，超算中心最常用的那个。
      ⟹ 你提交一个作业，它排队，轮到你了才给你分配 GPU。
   ★ DLC = Deep Learning Containers，阿里云的容器化训练/推理平台。

   ⟹ 🔑 从这个目录能读出 OpenCompass 的【真实使用场景】：
      ★★★ 它是给【有集群的团队】设计的，
      ⚠️ 不是给"我在自己笔记本上跑个分"设计的。
```

✅ Runner 基类的职责（`runners/base.py:11-12` 的 docstring 原文）：

```python
class BaseRunner:
    """Base class for all runners. A runner is responsible for launching
    multiple tasks.
```

✅ 它定义了三个方法（`runners/base.py:31 / :43 / :54`）：

```python
    def __call__(self, tasks): ...      # L31  ★ 入口
    def launch(self, tasks): ...        # L43  ★★ 子类必须实现：怎么把任务发出去
    def summarize(self, status): ...    # L54  ★ 汇总每个任务的返回码
```

---

## 2. ★★ Partitioner：切分策略本身就是一门学问

✅ 目录里有 7 个切分器（`opencompass/partitioners/`）：

```
   naive.py           ← ★ 最简单：一个「模型-数据集」对 = 一个任务
   size.py            ← ★★★ 按数据集大小切（最常用）
   num_worker.py      ← ★★ 按你有几个 worker 反推怎么切
   sub_naive.py       ← ★ 主观评测版
   sub_size.py        ← ★ 主观评测 + 按大小
   sub_num_worker.py  ← ★ 主观评测 + 按 worker 数
```

### ✅ 为什么需要 SizePartitioner（`partitioners/size.py:18-35` docstring 原文）

```python
class SizePartitioner(BasePartitioner):
    """Task partitioner based on the size of the dataset (with some rough
    expansion as an estimation of computational cost).

    Args:
        max_task_size (int): The maximum size of a task.
        gen_task_coef (int): The dataset cost measurement coefficient for
            generation tasks.
        strategy (str): The partition strategy. Supported strategies are:
            'heuristic' and 'split'. ...
            heuristic: split large datasets into several tasks, merge small
                datasets into one task.
            split: split large datasets into several tasks only.
    """
```

✅ 默认值（`partitioners/size.py:39-41`）：

```python
                 max_task_size: int = 40000,
                 gen_task_coef: int = 20,
                 strategy: str = 'heuristic',
```

```
   ⟹ 🔑🔑 这两个默认值里藏着一个很重要的工程认知：

   ★★★ `gen_task_coef = 20` 的意思是：
      【生成式任务的成本，按 20 倍于判别式任务来估算。】

   ⚠️ 为什么？因为：
      ★ 判别式（算 loglikelihood）：一次前向，输出 0 个 token。
      ★★ 生成式（generate_until）：要一个 token 一个 token 地往外吐，
         ⟹ 吐 256 个 token 就是 256 次前向。

   ⟹ ★★ 这正好印证了 [02 章](评测入门/02-三种评测范式.md) 那条定律的
      成本侧：★★★【生成式比判别式贵一个数量级】，
      ⚠️ 这里给出了一个具体的工程估值：约 20 倍。

   ⟹ 🔑 `strategy='heuristic'`（启发式）的两句话也很实在：
      ★★ 大数据集拆开（否则一个任务跑到天荒地老，崩了全丢）
      ★★ 小数据集合并（否则调度开销比计算还大）
      ⟹ ⚠️ 这就是典型的【负载均衡】问题。
```

---

## 3. ★★ Task 内部：OpenICL 才是真正的评测内核

✅ `opencompass/tasks/openicl_infer.py:21`：

```python
@TASKS.register_module()
class OpenICLInferTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """
```

```
   ★ 名词：
   ★★★ ICL = In-Context Learning（上下文学习）
      ⟹ 就是 [04 章](评测入门/04-few-shot与提示词.md) 讲的 few-shot：
         把几个示例塞进提示词里，让模型"照着做"。
   ★ OpenICL = OpenCompass 里实现 ICL 的那个子库。
```

### ✅ 一次推理由三个组件拼出来（`tasks/openicl_infer.py:121 / :134 / :144`）

```python
        retriever  = ICL_RETRIEVERS.build(retriever_cfg)     # L121
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)   # L134
        inferencer.inference(retriever, ...)                 # L144
```

```
   ⟹ ★★★ 拆开看这两个词：

   ┌─ Retriever（检索器）──────────────────────────────────────┐
   │  ★★ 职责：【这道题的 few-shot 示例从哪来？】              │
   │  ✅ openicl/icl_retriever/ 里有 10 个实现：               │
   │     icl_zero_retriever.py   ← ★ zero-shot，一个例子都不给 │
   │     icl_fix_k_retriever.py  ← ★ 固定取前 k 个             │
   │     icl_random_retriever.py ← ★ 随机取                    │
   │     icl_topk_retriever.py   ← ★★ 取语义最相近的 k 个      │
   │     icl_bm25_retriever.py   ← ★★ 用 BM25 关键词检索       │
   │     icl_votek_retriever.py / icl_dpp_retriever.py / ...   │
   └───────────────────────────────────────────────────────────┘
   ┌─ Inferencer（推理器）─────────────────────────────────────┐
   │  ★★ 职责：【拼好的提示词怎么送给模型、要什么形式的输出】  │
   │  ✅ openicl/icl_inferencer/ 里有 17 个实现，关键的几个：  │
   │     icl_ppl_inferencer.py   ← ★★★ 判别式：比困惑度        │
   │     icl_ll_inferencer.py    ← ★★★ 判别式：比 loglikelihood│
   │     icl_gen_inferencer.py   ← ★★★ 生成式                  │
   │     icl_chat_inferencer.py  ← ★★ 走 chat 消息格式         │
   │     icl_agent_inferencer.py ← ★★ agentic                  │
   │     icl_sc_inferencer.py    ← ★ self-consistency（自洽投票）│
   └───────────────────────────────────────────────────────────┘

   ⟹ 🔑🔑🔑 请对照 [03 章](评测入门/03-三个原语.md)：

   ★★★ ppl / ll / gen 这三个 Inferencer，
        就是 lm_eval 那三个原语（loglikelihood_rolling /
        loglikelihood / generate_until）的【同一批东西换了个名字】。

   ⟹ ⚠️ 两个独立团队，各自做设计，最后落到了同一个三分法。
      ★★ 这说明这三分法不是谁的偏好，是【问题本身的形状】。
```

```
   ⟹ 🔑 但有一个【关键区别】值得注意：

   ★★ lm_eval：三个原语在【模型接口层】（LM 基类的三个方法）。
   ★★ OpenCompass：三分法在【推理策略层】（Inferencer 类）。

   ⚠️ 后果：
      ★★★ lm_eval 的窄腰更"窄"——加一个新后端只要实现三个方法。
      ★★★ OpenCompass 的分层更"高"——它的复用点在调度，不在接口。
      ⟹ 这不是谁好谁坏，是【两边在优化不同的东西】。
```

---

## 4. ★★ OpenCompass 的独门强项：主观评测

✅ `opencompass/datasets/subjective/` 目录下有 20+ 个数据集适配：

```
   alignbench.py    alpacaeval.py    arena_hard.py    compass_arena.py
   creationbench.py flames.py        fofo.py          followbench.py
   hellobench.py    judgerbench.py   mtbench.py       mtbench101.py
   wildbench.py     writingbench.py  ...
```

✅ 并且有专门的任务类型（`opencompass/tasks/subjective_eval.py`）和
专门的切分器（`partitioners/sub_naive.py`、`sub_size.py`、`sub_num_worker.py`）。

```
   ⟹ 🔑 为什么主观评测需要【单独一套切分器】？

   ★★★ 因为客观评测是"模型 × 数据集"的二维矩阵，
        而主观评测（两两对战）是"模型 × 模型 × 数据集"的【三维】。
   ⚠️ N 个模型两两对战 = N(N-1)/2 对，
      ⟹ 10 个模型就是 45 对，20 个模型就是 190 对。
   ⟹ ★★ 组合爆炸的形状不一样，切分逻辑当然也得不一样。

   ⟹ 详见 [09 章 LLM 当裁判](评测入门/09-LLM当裁判.md)。
```

---

# 第二部分：EvalScope

## 5. 它多做了一件别人不做的事：压测

✅ `evalscope/` 顶层目录：

```
   benchmarks/   ← ★★ 173 个条目（能力评测）
   metrics/      ← ★★ 打分逻辑，含 judge/（LLM 裁判）
   models/       ← ★ 各种后端适配
   perf/         ← ★★★ 【性能压测】—— 这个别的框架没有
   backend/      ← ★★ 可以把 OpenCompass 当后端调用
   collections/  ← ★ 数据集组合
   report/  summarizer/  filters/  evaluator/  agent/  api/  cli/
```

```
   ⟹ 🔑🔑 注意两个目录：

   ★★★ `perf/` —— 它测的不是"答得对不对"，而是：
      ★ 首 token 延迟（TTFT，Time To First Token）
      ★ 吞吐（每秒能出多少 token）
      ★ 并发上去以后会不会崩
      ⟹ ⚠️ 这本质上是 [推理框架](../推理框架/) 那边的话题，
         ★★ EvalScope 把它和能力评测放在了【同一个工具】里。

   ★★ `backend/opencompass/` —— EvalScope 可以【把 OpenCompass 当成
      一个后端来调用】。
      ⟹ ⚠️ 说明它的定位是"上层统一入口"，而不是"再造一个 lm_eval"。
```

```
   ⟹ 🔑 为什么把这两件事放一起是有道理的？

   ★★★ 因为选型时你要回答的从来不是一个问题，而是三个：
      ① 这个模型【够不够聪明】？   ← 能力评测
      ② 它【够不够快】？           ← 压测
      ③ 它【够不够便宜】？         ← 压测 + 定价
   ⟹ ⚠️ 只看第①个就下结论，是 [11 章](评测入门/11-怎么读懂一张榜单.md)
      里最常见的读榜错误之一。
```

---

## 6. ★★★ LLM 裁判：EvalScope 把它做成了一等公民

★★ 这一节是本篇最值得细读的部分，因为它把 [09 章](评测入门/09-LLM当裁判.md)
讲的抽象概念，全部落成了可读的代码。

### 6.1 ✅ 裁判有四种策略（`evalscope/constants.py:110-114`）

```python
class JudgeStrategy:
    AUTO = 'auto'
    RULE = 'rule'
    LLM = 'llm'
    LLM_RECALL = 'llm_recall'
```

```
   ⚠️ 逐条解释：

   ★ RULE       = 只用规则/正则打分。★★ 便宜、确定、可复现。
   ★ LLM        = 只用大模型当裁判。★★ 贵、灵活、有噪声。
   ★★ AUTO      = 框架自己决定用哪个。
   ★★★ LLM_RECALL = 【先用规则判，规则判为"错"的再交给 LLM 复核】。

   ⟹ 🔑🔑 LLM_RECALL 这个策略非常聪明，值得单独讲：

   ★★★ 回想 [05 章](评测入门/05-答案抽取与打分.md) 那个核心痛点：
      【抽取失败 ≠ 答错，但在分数上完全等价且不可见】。

   ⟹ LLM_RECALL 正是冲着这个痛点去的：
      ★ 规则判对的 → 直接采信（省钱，且规则不会"误判为对"）
      ★★★ 规则判错的 → ⚠️ 可能是真错，也可能只是【格式没对上】
         ⟹ 这时候才花钱请 LLM 看一眼。

   ⟹ 🔑 成本收益：★★ 因为大部分题规则都能判对，
      所以只有少数题会走到 LLM，费用可控；
      ★★★ 而被"格式冤枉"的那些题，恰好全都在这少数里。
```

### 6.2 ✅ 默认裁判模型是写死的（`metrics/judge/llm_judge.py:43`）

```python
DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
```

✅ 可以用环境变量覆盖（`llm_judge.py:87`）：

```python
        self.model_id = model_id or os.environ.get('MODELSCOPE_JUDGE_LLM', DEFAULT_JUDGE_MODEL)
```

```
   ⟹ ⚠️⚠️ 这里必须提醒一个 [09 章](评测入门/09-LLM当裁判.md) 讲过的坑：

   ★★★ 【自我偏好偏见（self-preference bias）】
      = 模型当裁判时，倾向于给"和自己风格像"的答案打高分。

   ⟹ 🔑 所以：★★★ 用 Qwen 当裁判去评 Qwen 系列模型，
      ⚠️ 结论天然带利益冲突。
   ⟹ ★★ 换裁判模型跑一遍，看结论稳不稳，是必须做的健壮性检查。
   ⟹ ★ 好消息是这个框架允许你换（就是上面那个环境变量）。
```

### 6.3 ✅ 裁判的提示词长什么样（`llm_judge.py:12-30`，原文节选）

```
Your job is to look at a question, a gold target, and a predicted answer,
and return a letter "A" or "B" to indicate whether the predicted answer is
correct or incorrect.

[Question]
{question}

[Reference Answer]
{gold}

[Predicted Answer]
{pred}

Evaluate the model's answer based on correctness compared to the reference answer.
Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT

Just return the letters "A" or "B", with no text around it.
```

```
   ⟹ 🔑🔑 这个提示词的三个设计细节，全都是有意为之：

   ★★★ ① 输出被压缩成【单个字母 A/B】。
      ⟹ ⚠️ 为什么不让它输出 "correct"/"incorrect"？
      ★★ 因为字母只有 1 个 token，⟹ 便宜、快、且几乎不可能跑偏格式。
      ⟹ ★ 这和 [05 章](评测入门/05-答案抽取与打分.md) 里
         "让答案尽可能好抽"是同一个思路。

   ★★★ ② 明确写了 "with no text around it"（前后不要有别的字）。
      ⟹ ⚠️ 因为聊天模型天然爱解释，不拦着它就会输出
         "The answer is A because..."，⟹ 抽取就变复杂了。

   ★★ ③ 它是【有参考答案的打分】（给了 gold target），
      ⟹ 不是"你觉得这答案好不好"的开放式评价。
      ⟹ ★★★ 这大幅降低了裁判的主观性 ——
         ⚠️ 但代价是：这种裁判【只能用在有标准答案的题上】。
```

✅ 框架还提供了另一种模板 —— 打连续分（`llm_judge.py:34-42`，节选）：

```
you must rate the response on a scale of 0 (worst) to 1 (best) by strictly
following this format: "[[rating]]", for example: "Rating: [[0.5]]"
```

```
   ⟹ ★★ 注意 `[[  ]]` 这个双方括号。
   ⚠️ 它不是装饰，是【为了让正则能唯一定位分数】：
      ★★★ 答案文本里可能到处都是数字，但 [[0.5]] 这种形状不会误撞。
   ⟹ ✅ 对应的枚举在 `constants.py:117-119`：

        class JudgeScoreType:
            NUMERIC = 'numeric'   # 连续分
            PATTERN = 'pattern'   # 模式匹配分
```

---

## 7. ★★★ Arena Hard：一份可以逐行读的"两两对战"实现

★★ 这是全篇最推荐精读的一段源码，因为它把 [09 章](评测入门/09-LLM当裁判.md)
讲的**位置偏见对策**和 **Elo 评分**，用不到 100 行代码全实现了。

✅ `benchmarks/arena_hard/arena_hard_adapter.py:41` 的 docstring 原文：

```
- Two-game battle system (A vs B and B vs A)
```

### 7.1 ✅ 同一道题判两次，顺序对调（`arena_hard_adapter.py:105-111`）

```python
        # reference is baseline answer 'A', filtered_prediction is model answer 'B'
        prompt1 = GRADER_TEMPLATE.format(question=question, answer_1=reference, answer_2=filtered_prediction)
        # reverse the order
        prompt2 = GRADER_TEMPLATE.format(question=question, answer_1=filtered_prediction, answer_2=reference)

        # get grading response
        game1_response = self.llm_judge.judge(prompt1, system_prompt=GRADER_SYSTEM_PROMPT)
        game2_response = self.llm_judge.judge(prompt2, system_prompt=GRADER_SYSTEM_PROMPT)
```

✅ 然后把两局取平均（`arena_hard_adapter.py:144`）：

```python
        score.value = {'score': (score1 + score2) / 2}
```

```
   ⟹ 🔑🔑🔑 这四行代码在对付一个真实存在的毛病：

   ★★★【位置偏见（position bias）】
      = 大模型当裁判时，会系统性地偏爱【排在前面】的那个答案。
      ⚠️ 注意"系统性"三个字：它不是随机噪声，
         ⟹ 你跑一万次也不会自己抵消掉。

   ⟹ ★★ 对策就是上面这个：同一对答案，正着判一次、反着判一次，取平均。
   ⟹ ⚠️ 代价：★★★【裁判调用次数直接翻倍，成本翻倍】。
   ⟹ 🔑 这就是评测里典型的"用钱买正确性"。
```

### 7.2 ✅ 五档判决怎么变成分数（`benchmarks/arena_hard/utils.py:21-49`）

```python
def get_judge_score(result, reverse=False):
    if not reverse:
        score_mapping = {
            'A=B': 0.5,   # Tie
            'A>B': 0.75,  # A slightly wins
            'A>>B': 1.0,  # A significantly wins
            'B>A': 0.25,  # B slightly wins
            'B>>A': 0.0,  # B significantly wins
        }
    else:
        score_mapping = {
            'A=B': 0.5, 'A>B': 0.25, 'A>>B': 0.0,
            'B>A': 0.75, 'B>>A': 1.0,
        }
    base_score = score_mapping.get(result, 0.5)
    return base_score
```

```
   ⟹ ★★ 三个要点：

   ★★★ ① 判决不是二元的（赢/输），是【五档】：
      显著更好 / 略好 / 平 / 略差 / 显著差 → 1.0 / 0.75 / 0.5 / 0.25 / 0.0
      ⟹ ⚠️ 为什么要五档？因为"略好"和"碾压"是两码事，
         ★★ 硬压成二元会丢掉大量信息。

   ★★★ ② `reverse=True` 那份映射表是【左右镜像】的。
      ⟹ 因为第二局里 A 和 B 的身份换了，⟹ 分数也必须跟着换。
      ⚠️ 这行代码写反了的话，两局就不是互相校正而是互相抵消。

   ★★★ ③ `score_mapping.get(result, 0.5)` —— 【看这个默认值】。
      ⟹ ⚠️ 裁判要是没输出可识别的判决（比如它开始长篇大论了），
         ★★ 这一局按【平局 0.5】算。
      ⟹ 🔑 这又是一次"抽取失败被静默吞掉"：
         和 lm_eval 的 `"[invalid]"`、SWE-bench 的 `APPLY_PATCH_FAIL`
         是同一类事，只是这里连个标记都没留。
```

✅ 抽取判决用的正则（`utils.py:13-18`）：

```python
def post_process_arenahard(completion):
    result = re.findall(r'\[\[([AB<>=]+)\]\]', completion)
    if result:
        return result[0]
    else:
        return None
```

```
   ⟹ ⚠️ 注意 `result[0]` ——【取第一个匹配】。
   ★★ 如果裁判在解释过程中先写了 [[A>B]] 又改口写 [[B>A]]，
      ⟹ 采信的是【前面那个】。
   ⟹ 🔑 对照 [05 章](评测入门/05-答案抽取与打分.md)：
      ★★★ "取第一个还是取最后一个"是个真实的分数差异来源，
      ⚠️ 对会写思维链的模型，通常【最后一个】才是它的结论。
```

### 7.3 ✅ 最后汇总成 Elo 和胜率（`arena_hard_adapter.py:148-171`）

```python
    def aggregate_scores(self, sample_scores):
        from .utils import compute_mle_elo, get_battles_from_row, \
                           get_bootstrap_result, get_win_rate_column

        battles = pd.concat([get_battles_from_row(res.score.metadata['battle_result'])
                             for res in sample_scores])
        bootstrap_online_elo = compute_mle_elo(battles)
        ...
        score = get_win_rate_column(stats, 'score', 'gpt4-0314').at['test_model']

        return [AggScore(score=score, metric_name='winrate', num=len(sample_scores))]
```

✅ 锚点模型写在 `arena_hard_adapter.py:124`：

```python
            'model_a': 'gpt4-0314',
            'model_b': 'test_model',
```

```
   ★ 名词一次说清：

   ★★★ Elo（埃洛评分）= 国际象棋用的那套天梯分算法。
      ⟹ 核心思想：★★ 赢强的对手加分多，赢弱的对手加分少。
      ⟹ ⚠️ 它只能算出【相对强弱】，算不出绝对水平。

   ★★★ MLE（Maximum Likelihood Estimation，最大似然估计）
      = 一种统计方法：⟹ 找一组 Elo 分，使得
        "按这组分打，最可能打出我们实际观察到的那些胜负结果"。
      ⚠️ 比传统的"一局一局往上加"更稳，因为它一次性看全部对局。

   ★★★ bootstrap（自助法）= 从已有对局里【有放回地随机重抽】很多次，
      每次都重算一遍 Elo。
      ⟹ ⚠️ 目的：得到【置信区间】，也就是"这个分上下浮动多少算正常"。
      ⟹ 🔑🔑 这直接关系到 [11 章](评测入门/11-怎么读懂一张榜单.md)
         那个核心问题：★★★【榜单上差 1 分，到底算不算差距？】
         ⟹ 有了 bootstrap 你才知道答案，⚠️ 通常是"不算"。

   ★★★ 锚点模型（anchor）= `gpt4-0314`。
      ⟹ 所有模型都跟它对战，报的是【对它的胜率】。
      ⟹ ⚠️⚠️ 这条必须记住：
         ★★★【锚点一换，所有历史分数全部作废，不能跨版本比较。】
```

---

## 8. ⚠️ 三个框架的边界：它们都不做什么

```
   ❌ 都不负责【生成】被评的答案（除了 agentic 那条线）
      ⟹ ★★ 你得先有模型或 API。

   ❌ 都不能替你判断【题目本身合不合理】
      ⟹ ⚠️ 数据集里的错题、烂题，框架照跑不误。

   ❌ 都不能替你排除【数据污染】
      ⟹ ★★★ 见 [06 章](评测入门/06-数据污染.md)。
      ⚠️ lm_eval 有个 filters/decontamination.py，但那只是工具，
         ⟹ 是否污染仍然要你自己判断。

   ❌ 都不会告诉你【这个分数对你的业务意味着什么】
      ⟹ 🔑 这是 [12 章](评测入门/12-自己搭一套评测.md) 存在的理由。
```

---

## 9. 一句话总结

```
   ★★★ 如果你只想跑个标准分：用 lm-eval-harness。
   ★★★ 如果你有集群、要一次评几十个模型：用 OpenCompass。
   ★★★ 如果你要做主观对战评测、或者还想顺便压测服务：用 EvalScope。

   ⟹ 🔑🔑 但真正该带走的不是"选哪个"，而是这个观察：

   ★★★ 三个团队独立设计，最后都收敛到了
      【判别式 / 生成式 / agentic】这同一个三分法。
   ⚠️ 框架的名字会变，接口会变，
      ⟹ ★★ 但这三种问模型的方式，是问题本身决定的。
```

---

> 相关：[lm-eval-harness拆解.md](lm-eval-harness拆解.md) ｜ [SWE-bench拆解.md](SWE-bench拆解.md) ｜ [评测框架对比.md](评测框架对比.md)
> 返回 [harness评测框架/](README.md)
