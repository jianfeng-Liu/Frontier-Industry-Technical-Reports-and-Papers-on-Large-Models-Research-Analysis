# OpenCompass 与 EvalScope 拆解

> ★ 版本口径：`opencompass` **0.5.3**、`evalscope` **1.10.0**
> （源码在 [源码/opencompass-0.5.3/](源码/opencompass-0.5.3/)、[源码/evalscope-1.10.0/](源码/evalscope-1.10.0/)，本文所有行号都是对着这两份数的）
>
> ★★ 定位：两个**中国团队做的**评测框架。OpenCompass 来自上海人工智能实验室（中文名「司南」），EvalScope 来自阿里的 ModelScope（魔搭）社区。
> ⚠️ 前置阅读：[lm-eval-harness拆解.md](lm-eval-harness拆解.md)。
> ★★★ 这一篇的价值不在于"多认识两个工具"，
> 而在于：**同一个问题，三份不同的答卷，能让你看清设计权衡在哪。**
>
> 标记：✅ = 源码可查证 ｜ ⚠️ = 我的推断 ｜ ★ = 重点 ｜ 🔑 = 关键结论
>
> ⚠️⚠️ **行号会过期**，这两个项目迭代都很快。★★ 对不上时以你手上那份源码为准，并把版本号一起记下来。

---

## 0. 先说结论：三个框架各自解决什么问题

| 框架 | ★★ 它要解决的核心问题 | ⟹ 解法 | ⟹ 强项 |
|---|---|---|---|
| **lm-eval-harness** | ★★ 【模型接口太多样】 | 三个原语的窄腰 | 判别式（`loglikelihood`）评测的事实标准 |
| **OpenCompass**（司南） | ★★★ 【一次要跑几十个模型 × 上百个数据集，单机跑不完】 | 切分 + 调度（Partitioner + Runner） | 集群规模化、主观评测（LLM 裁判） |
| **EvalScope**（魔搭） | ★★★ 【评测不只是准确率，还得知道快不快、贵不贵】 | 把「能力评测」和「性能压测」放进同一个包 | Arena 类主观评测 + 推理服务压测 |

```
   ⟹ 🔑 一句话：
      ★★★ lm_eval 解决【怎么问】，OpenCompass 解决【怎么排产】，
           EvalScope 解决【怎么判 + 怎么压测】。
```

---

## 1. ★★★ 两家的分层结构，和 lm_eval 摆在一起看

★★ 这张表是全篇的骨架 —— **三家在"同一件事"上各自把它放在了哪一层**。

| 做的事 | lm-eval-harness | OpenCompass 0.5.3 | EvalScope 1.10.0 |
|---|---|---|---|
| ★ **任务定义放哪** | 一个 `yaml`：`lm_eval/tasks/<name>/<name>.yaml` | 一个 `py` 配置：`opencompass/configs/datasets/<name>/*.py` | 一个 `@register_benchmark(BenchmarkMeta(...))` 装饰器 + 一份 `_meta/<name>.json` |
| ★ **任务总量**（✅ 实际数过） | 212 个任务目录 | ✅ `configs/datasets/` 下 **230** 个数据集目录，★★ 一个数据集常有多份不同口径的配置 | ✅ 171 个基准目录，`@register_benchmark` 一共 **220** 次（一个目录可注册多个） |
| ★★★ **"三种问法"落在哪一层** | **模型接口层**：`LM` 基类的三个方法 | **推理策略层**：`Inferencer` 类（`ppl` / `ll` / `gen`） | **数据适配层**：`DataAdapter` 子类 |
| ★ **few-shot 示例怎么选** | 任务 yaml 里的 `fewshot_config.sampler` | ★★ 独立的 `Retriever` 组件（9 种策略可换） | `BenchmarkMeta.few_shot_num` + `sample_to_fewshot()` 方法 |
| ★ **答案怎么抠出来** | `filter_list`（★★★ 可以配**多把并存**） | `pred_postprocessor` | `extract_answer()` 方法 |
| ★ **怎么判分** | `metric_list` | `evaluator`（如 `MATHEvaluator`） | `metric_list` + `judge/`（LLM 裁判） |
| ★★ **怎么调度到算力上** | ⚠️ 没有这一层，自己写 for 循环 | ★★★ `Partitioner` + `Runner`（本机 / Slurm / 阿里云 DLC / 火山引擎） | ⚠️ 较轻；可以把 OpenCompass 当后端调用 |
| ★★ **主观评测（两两对战）** | ❌ 没有内建 | ✅ 一等公民：22 个主观数据集适配 + 专用切分器 | ✅ 一等公民：`JudgeStrategy` 四档 + Arena Hard 完整实现 |
| ★★ **性能压测** | ❌ 没有 | ❌ 没有 | ✅ **`perf/` 目录**，三家里只有它有 |

```
   ⟹ 🔑🔑 这张表最该盯住的是【第三行】：

   ★★ lm_eval：三个原语在【模型接口层】（LM 基类的三个方法）。
   ★★ OpenCompass：三分法在【推理策略层】（Inferencer 类）。
   ★★ EvalScope：靠【数据适配层】的方法重载来区分。

   ⚠️ 后果：
      ★★★ lm_eval 的窄腰更"窄"——加一个新后端只要实现三个方法。
      ★★★ OpenCompass 的分层更"高"——它的复用点在调度，不在接口。
      ⟹ 这不是谁好谁坏，是【两边在优化不同的东西】。

   ⟹ 🔑 另一个值得注意的点：★★ 三家都有「抽答案」这一层，
      ⚠️ 但只有 lm_eval 支持【同一次输出配多把尺子、出多份分数】
         （见 [lm-eval-harness拆解.md](lm-eval-harness拆解.md) §4.2）。
      ⟹ 另两家是一条流水线一个结果。
```

---

## 2. ★★★ 同一个评测在三家怎么配：GSM8K 并排看

★★ 抽象的分层表看完，马上拿一个具体例子落地。⟹ 选 GSM8K（小学数学应用题），因为三家都有，而且**其中两家的提示词一模一样**。

### 2.1 ✅ lm-eval-harness：`tasks/gsm8k/gsm8k.yaml`（节选）

```yaml
output_type: generate_until
doc_to_text: "Question: {{question}}\nAnswer:"
num_fewshot: 5
generation_kwargs:
  until: ["Question:", "</s>", "<|im_end|>"]
  do_sample: false
  temperature: 0.0
metric_list:
  - metric: exact_match
    regexes_to_ignore: [",", "\\$", "(?s).*#### ", "\\.$"]
filter_list:
  - name: "strict-match"        # ★★★ 两把尺子并存
    filter: [{function: regex, regex_pattern: "#### (\\-?[0-9\\.\\,]+)"}, {function: take_first}]
  - name: "flexible-extract"
    filter: [{function: regex, group_select: -1, regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"}, {function: take_first}]
```

### 2.2 ✅ OpenCompass：`configs/datasets/gsm8k/gsm8k_0shot_v2_gen_17d799.py`（完整原文）

```python
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
from opencompass.datasets import MATHEvaluator, math_postprocess_v2

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

gsm8k_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
)

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg,
    )
]
```

### 2.3 ✅ EvalScope：`benchmarks/gsm8k/gsm8k_adapter.py:55-75`（节选）

```python
@register_benchmark(
    BenchmarkMeta(
        name='gsm8k',
        pretty_name='GSM8K',
        dataset_id='AI-ModelScope/gsm8k',
        tags=[Tags.MATH, Tags.REASONING],
        paper_url='https://arxiv.org/abs/2110.14168',
        subset_list=['main'],
        few_shot_num=4,
        train_split='train',
        eval_split='test',
        metric_list=[{'acc': {'numeric': True}}],
        prompt_template=PROMPT_TEMPLATE,
        few_shot_prompt_template=FEWSHOT_TEMPLATE,
    )
)
class GSM8KAdapter(DefaultDataAdapter):

    def extract_answer(self, prediction: str, task_state: TaskState):
        from evalscope.metrics.math.parser import extract_answer
        return extract_answer(prediction)
```

✅ 它的提示词模板（`gsm8k_adapter.py:15-16`）：

```python
PROMPT_TEMPLATE = """{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.""".lstrip()
```

### 2.4 ★★★ 三份配置逐项对照 —— 这才是本节的重点

| 口径项 | lm_eval `gsm8k.yaml` | OpenCompass `0shot_v2` | EvalScope `gsm8k` |
|---|---|---|---|
| **shot 数** | **5** | **0**（`ZeroRetriever`） | **4** |
| **提示词** | `Question: … \nAnswer:` | `{question}\nPlease reason step by step, and put your final answer within \boxed{}.` | ★★ **和 OpenCompass 一字不差** |
| **要求的答案格式** | `#### <数字>`（GSM8K 原生格式） | `\boxed{<数字>}` | `\boxed{<数字>}` |
| **抽取方式** | ★★★ 两条正则**并存**，出两份分数 | `math_postprocess_v2` 一条流水线 | `extract_answer()`（复用数学解析器） |
| **判分器** | `exact_match` + 4 条忽略正则 | `MATHEvaluator(version='v2')` | `acc`（`numeric: True`，按数值比） |
| **数据集来源** | HuggingFace `openai/gsm8k` | `opencompass/gsm8k`（自建镜像） | ModelScope `AI-ModelScope/gsm8k` |
| **题数** | 1,319（test） | 1,319 | ✅ `_meta/gsm8k.json` 明写 `total_samples: 1319` |
| ★★ **出几个分数** | **2 个**（strict / flexible） | 1 个 | 1 个 |

```
   ⟹ 🔑🔑🔑 把这张表读一遍，一个很重要的事实就冒出来了：

   ★★★ 这三家报出来的「GSM8K 分数」，
        shot 数是 5 / 0 / 4，
        提示词有两套，
        要求的答案格式有两套（#### 和 \boxed{}），
        判分器三套，
        数据集托管在三个地方。

   ⟹ ⚠️⚠️ 所以：★★★【「某模型 GSM8K 得分 xx%」这句话，
      在没说清用哪个框架、哪份配置之前，是没有意义的。】

   ⟹ 🔑 而且注意：★★ 连「要求模型把答案写成什么样」都不一样。
      ⚠️ 一个被训成习惯写 \boxed{} 的模型，
         在 lm_eval 的 strict-match 下会被【系统性低估】；
      ⚠️ 反过来，习惯写 #### 的模型在另两家下也吃亏。
      ⟹ ★★★ 这不是模型能力差异，是【格式习惯和抽取规则对不上】。
      ⟹ 这正是 [05 章](评测入门/05-答案抽取与打分.md) 那条结论的跨框架版本。

   ⟹ ★★ 还有一条更隐蔽的：✅ OpenCompass 的 GSM8K 一个目录下就有
      【20 个】配置文件（gsm8k_gen_1d7fe4 / _701491 / _0shot_v2_gen_17d799 /
      _0shot_nocot_gen_6cbf22 / _xfinder_gen_a58960 …），
      ⚠️ 文件名后缀那串十六进制是用来区分不同口径的标记。
      ⟹ 🔑 所以在 OpenCompass 里引用分数，
         ★★★ 必须报到【文件名那一级】，报"gsm8k"等于没报。
      ⚠️ 光看这 20 个文件名就能读出几种完全不同的口径：
         0-shot 还是 few-shot、要不要思维链（nocot）、
         用规则判还是用模型判（model_postprocess / xfinder）、
         是不是 agent 模式（agent_gen）。
```

---

# 第一部分：OpenCompass

## 3. ★★★ 它的核心抽象：Partitioner → Runner → Task

★★ 这是 OpenCompass 和 lm_eval 最根本的架构差异。

```
   ⚠️ 先想清楚它要解决的场景：

   ★ 你有 20 个模型要评，每个模型评 100 个数据集。
   ⟹ 20 × 100 = 2,000 个「模型-数据集」组合。
   ⟹ ⚠️ 用 lm_eval 的话，你得写一个 for 循环跑 2,000 次，
      而且【一次崩了就得从头再来】。

   ★★★ OpenCompass 的答案是：把评测当成【一批可调度的作业】。
```

| 层 | ✅ 源码位置 | ★★ 职责 | 输入 ⟹ 输出 |
|---|---|---|---|
| ① **Partitioner**（切分器） | `opencompass/partitioners/` | 把「模型 × 数据集」的大矩阵，切成一堆小任务 | models 配置 + datasets 配置 ⟹ 一个 task 列表（每个 task 是一个 dict） |
| ② **Runner**（执行器） | `opencompass/runners/` | 把这些 task **投递到某种算力上**去跑 | task 列表 ⟹ 本机多进程？Slurm 集群？阿里云 DLC？火山引擎？ |
| ③ **Task**（任务） | `opencompass/tasks/` | 一个 task 内部**真正干活** | `OpenICLInferTask` = 推理，`OpenICLEvalTask` = 打分 |

```
   ⟹ 🔑🔑 这三层的关键在于【解耦】：
      ★★★ 换集群只需要换 Runner，不用动任何评测逻辑。
```

### 3.1 ✅ Runner 的全部实现（`opencompass/runners/` 目录）

| 文件 | 投到哪种算力 |
|---|---|
| `base.py` | 基类 |
| `local.py` | ★ 本机多进程 |
| `local_api.py` | ★ 本机跑 API 模型（不占 GPU） |
| `slurm.py` | ★★ Slurm 集群（学术界 / 超算中心标配） |
| `slurm_sequential.py` | ★ Slurm 串行版 |
| `dlc.py` | ★★ 阿里云 DLC |
| `volc.py` | ★★ 火山引擎 |
| `rjob.py` | ★ 内部调度平台 |

```
   ★ 名词：
   ★ Slurm（Simple Linux Utility for Resource Management）
      = 一种集群作业调度系统，超算中心最常用的那个。
      ⟹ 你提交一个作业，它排队，轮到你了才给你分配 GPU。
   ★ DLC = Deep Learning Containers（深度学习容器），
      阿里云的容器化训练 / 推理平台。

   ⟹ 🔑 从这个目录能读出 OpenCompass 的【真实使用场景】：
      ★★★ 它是给【有集群的团队】设计的，
      ⚠️ 不是给"我在自己笔记本上跑个分"设计的。
```

✅ Runner 基类的职责（`runners/base.py:10-12` 的 docstring 原文）：

```python
class BaseRunner:
    """Base class for all runners. A runner is responsible for launching
    multiple tasks.
```

✅ 它定义了三个方法（`runners/base.py:31 / :43 / :54`）：

```python
    def __call__(self, tasks: List[Dict[str, Any]]): ...              # L31  ★ 入口
    def launch(self, tasks: List[Dict[str, Any]]) -> ...: ...         # L43  ★★ 子类必须实现：怎么把任务发出去
    def summarize(self, status: List[Tuple[str, int]]) -> None: ...   # L54  ★ 汇总每个任务的返回码
```

---

## 4. ★★ Partitioner：切分策略本身就是一门学问

✅ 目录里有 **6 个**具体切分器（外加一个 `base.py` 基类）：

| 文件 | 策略 |
|---|---|
| `naive.py` | ★ 最简单：一个「模型-数据集」对 = 一个任务 |
| `size.py` | ★★★ 按数据集大小切（最常用） |
| `num_worker.py` | ★★ 按你有几个 worker 反推怎么切 |
| `sub_naive.py` | ★ 主观评测版 |
| `sub_size.py` | ★ 主观评测 + 按大小 |
| `sub_num_worker.py` | ★ 主观评测 + 按 worker 数 |

### 4.1 ✅ 为什么需要 SizePartitioner（`partitioners/size.py:18-35` docstring 原文）

```python
class SizePartitioner(BasePartitioner):
    """Task partitioner based on the size of the dataset (with some rough
    expansion as an estimation of computational cost).

    Args:
        out_dir (str): The output directory of tasks.
        max_task_size (int): The maximum size of a task.
        gen_task_coef (int): The dataset cost measurement coefficient for
            generation tasks.
        strategy (str): The partition strategy. Supported strategies are:
            'heuristic' and 'split'. Defaults to 'heuristic'.
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
   ★ 判别式（算 loglikelihood）：一次前向，输出 0 个新 token。
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

## 5. ★★ Task 内部：OpenICL 才是真正的评测内核

✅ `opencompass/tasks/openicl_infer.py:21-22`：

```python
@TASKS.register_module()
class OpenICLInferTask(BaseTask):
    """OpenICL Inference Task.
```

```
   ★ 名词：
   ★★★ ICL = In-Context Learning（上下文学习）
      ⟹ 就是 [04 章](评测入门/04-few-shot与提示词.md) 讲的 few-shot：
         把几个示例塞进提示词里，让模型"照着做"。
   ★ OpenICL = OpenCompass 里实现 ICL 的那个子库。
```

### 5.1 ✅ 一次推理由三步拼出来（`tasks/openicl_infer.py:121 / :134 / :144`）

```python
        retriever = ICL_RETRIEVERS.build(retriever_cfg)       # L121
        inferencer = ICL_INFERENCERS.build(inferencer_cfg)    # L134
        inferencer.inference(retriever, ...)                  # L144
```

★★ **Retriever（检索器）** —— 职责：【这道题的 few-shot 示例从哪来？】✅ `openicl/icl_retriever/` 里 10 个文件（1 个基类 + **9 种策略**）：

| 文件 | 策略 |
|---|---|
| `icl_zero_retriever.py` | ★ zero-shot，一个例子都不给 |
| `icl_fix_k_retriever.py` | ★ 固定取前 k 个 |
| `icl_random_retriever.py` | ★ 随机取 |
| `icl_sliding_k_retriever.py` | ★ 滑窗取 k 个 |
| `icl_topk_retriever.py` | ★★ 取语义最相近的 k 个 |
| `icl_bm25_retriever.py` | ★★ 用 BM25 关键词检索 |
| `icl_votek_retriever.py` | ★ vote-k 多样性采样 |
| `icl_dpp_retriever.py` | ★ 行列式点过程（DPP）选多样化示例 |
| `icl_mdl_retriever.py` | ★ 按最小描述长度（MDL）选 |

```
   ⟹ 🔑 光是"示例从哪来"就做了 9 种实现 ——
      ★★★ 这本身就说明了 [04 章](评测入门/04-few-shot与提示词.md) 那条结论
      有多实在：⚠️【换一种选示例的方式，分数就会变】。
   ⟹ ★★ 而 lm_eval 这一层只有 sampler: first_n / 随机两档。
```

★★ **Inferencer（推理器）** —— 职责：【拼好的提示词怎么送给模型、要什么形式的输出】✅ `openicl/icl_inferencer/` 里 18 个文件（1 个基类 + **17 种实现**），关键的几个：

| 文件 | 对应 lm_eval 的哪个原语 | 说明 |
|---|---|---|
| `icl_ppl_inferencer.py` | ★★★ `loglikelihood_rolling` | 判别式：比困惑度（perplexity） |
| `icl_ll_inferencer.py` | ★★★ `loglikelihood` | 判别式：比对数似然 |
| `icl_gen_inferencer.py` | ★★★ `generate_until` | 生成式 |
| `icl_chat_inferencer.py` | ⚠️ 无直接对应 | ★★ 走 chat 消息格式 |
| `icl_agent_inferencer.py` | ❌ lm_eval 没有 | ★★ agentic |
| `icl_sc_inferencer.py` | ⚠️ 相当于 `repeats` + 投票 filter | ★ self-consistency（自洽投票） |
| `icl_clp_inferencer.py` | ⚠️ 判别式变体 | 条件对数概率 |
| `icl_mink_percent_inferencer.py` | ❌ lm_eval 没有 | ★★ Min-K% 污染检测用 |

```
   ⟹ 🔑🔑🔑 请对照 [03 章](评测入门/03-三个原语.md)：

   ★★★ ppl / ll / gen 这三个 Inferencer，
        就是 lm_eval 那三个原语（loglikelihood_rolling /
        loglikelihood / generate_until）的【同一批东西换了个名字】。

   ⟹ ⚠️ 两个独立团队，各自做设计，最后落到了同一个三分法。
      ★★ 这说明这三分法不是谁的偏好，是【问题本身的形状】。

   ⟹ ★★ 但 OpenCompass 多了两样 lm_eval 没有的：
      agent（agentic 推理）和 mink_percent（污染检测，见 [06 章](评测入门/06-数据污染.md)）。
      ⚠️ 这不是三分法不成立，⟹ 是它在三分法【之外】另外长了东西。
```

---

## 6. ★★ OpenCompass 的独门强项：主观评测

✅ `opencompass/datasets/subjective/` 目录下有 **22 个**数据集适配文件（另有 `__init__.py` 和 `utils.py`）：

```
   alignbench.py    alpacaeval.py    arena_hard.py    commonbench.py
   compass_arena.py compass_arena_subjective_bench.py
   compassbench.py  compassbench_checklist.py
   compassbench_control_length_bias.py                corev2.py
   creationbench.py flames.py        fofo.py          followbench.py
   hellobench.py    judgerbench.py   mtbench.py       mtbench101.py
   multiround.py    subjective_cmp.py                 wildbench.py
   writingbench.py
```

✅ 并且有专门的任务类型（`opencompass/tasks/subjective_eval.py`）和专门的切分器（`partitioners/sub_naive.py`、`sub_size.py`、`sub_num_worker.py`）。

```
   ⟹ 🔑 为什么主观评测需要【单独一套切分器】？

   ★★★ 因为客观评测是"模型 × 数据集"的二维矩阵，
        而主观评测（两两对战）是"模型 × 模型 × 数据集"的【三维】。
   ⚠️ N 个模型两两对战 = N(N-1)/2 对，
      ⟹ 10 个模型就是 45 对，20 个模型就是 190 对。
   ⟹ ★★ 组合爆炸的形状不一样，切分逻辑当然也得不一样。

   ⟹ ⚠️ 顺手留意一个文件名：compassbench_control_length_bias.py
      ★★★ 「control length bias」= 控制长度偏见
      = 大模型当裁判时倾向于给【更长】的答案打高分。
```

⚠️ 这不是我从文件名猜的 —— ✅ 打开 `datasets/subjective/compassbench_control_length_bias.py` 能看到，它的裁判提示词里**把两个回答的字数直接算出来塞进去了**（★ 原文字段名 `prediction_cn_word_count` / `prediction_en_word_count`），并且明写了两条规矩（✅ 原文逐字）：

> 2. 在没有字数限制的情况下，回答的简洁性和直接性应被优先考虑，除非详细程度对于理解答案至关重要。
> 3. 如果两个回答都准确地解决了用户的问题，但一个回答更加简洁，而另一个回答提供了不必要的额外信息，那么简洁的回答可能会得到更高的评分。

```
   ⟹ 🔑🔑 这是一个很具体的工程手法：
      ★★★ 对付「裁判偏爱长答案」的办法不是劝它别偏心，
      ⟹ 而是【把字数变成它能看见的显式数字】，再明文要求它偏向简洁。
   ⟹ ★★ 说明 LLM 裁判要防的偏见不止一种：
      位置偏见（§9.1）、长度偏见（这里）、自我偏好偏见（§8.2）。
      ⟹ 详见 [09 章 LLM 当裁判](评测入门/09-LLM当裁判.md)。
```

---

# 第二部分：EvalScope

## 7. 它多做了一件别人不做的事：压测

✅ `evalscope/` 顶层目录：

| 目录 | 干什么 |
|---|---|
| `benchmarks/` | ★★ 171 个基准目录，`@register_benchmark` 共 220 次（能力评测） |
| `metrics/` | ★★ 打分逻辑，含 `judge/`（LLM 裁判） |
| `models/` | ★ 各种后端适配 |
| `perf/` | ★★★ **性能压测** —— 这个别的框架没有 |
| `backend/` | ★★ 可以把 OpenCompass 当后端调用（`backend/opencompass/`） |
| `collections/` | ★ 数据集组合 |
| 其余 | `report/` `summarizer/` `filters/` `evaluator/` `agent/` `api/` `cli/` `service/` `third_party/` |

```
   ⟹ 🔑🔑 注意两个目录：

   ★★★ `perf/` —— 它测的不是"答得对不对"，而是：
      ★ 首 token 延迟（TTFT，Time To First Token）
         ✅ 这个缩写在源码里是枚举常量，见 perf/utils/db_util.py:223
      ★ 吞吐（每秒能出多少 token）
      ★ 并发上去以后会不会崩
      ⟹ ✅ 目录里还有 sla/（服务等级协议）和 multi_turn_benchmark.py
         （多轮对话压测）⟹ ⚠️ 这是很明确的【生产选型】取向。
      ⟹ ⚠️ 这本质上是 [推理框架](../推理框架/) 那边的话题，
         ★★ EvalScope 把它和能力评测放在了【同一个工具】里。

   ★★ `backend/opencompass/` —— EvalScope 可以【把 OpenCompass 当成
      一个后端来调用】。
      ⟹ ⚠️ 说明它的定位是"上层统一入口"，而不是"再造一个 lm_eval"。
      ✅ 同目录下还有 rag_eval/ 和 vlm_eval_kit/ 两个后端。
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

## 8. ★★★ LLM 裁判：EvalScope 把它做成了一等公民

★★ 这一节是本篇最值得细读的部分，因为它把 [09 章](评测入门/09-LLM当裁判.md) 讲的抽象概念，全部落成了可读的代码。

### 8.1 ✅ 裁判有四种策略（`evalscope/constants.py:110-114`）

```python
class JudgeStrategy:
    AUTO = 'auto'
    RULE = 'rule'
    LLM = 'llm'
    LLM_RECALL = 'llm_recall'
```

| 策略 | 怎么判 | 成本 | 确定性 |
|---|---|---|---|
| `RULE` | 只用规则 / 正则打分 | ★★ 便宜 | ★★ 确定、可复现 |
| `LLM` | 只用大模型当裁判 | ★★ 贵 | ⚠️ 有噪声 |
| `AUTO` | 框架自己决定用哪个 | ⚠️ 看情况 | ⚠️ 看情况 |
| ★★★ `LLM_RECALL` | **先用规则判，规则判为"错"的再交给 LLM 复核** | ★★ 可控 | ★★ 大部分题走规则 |

```
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

   ⟹ ⚠️ 但要注意它引入的新问题：★★ 这是一个【单向复核】——
      只复核"规则判错"的，不复核"规则判对"的。
      ⟹ 所以规则【误判为对】的那些题，永远不会被抓出来。
      ⟹ 🔑 它系统性地往【提分】方向偏，这一点报分时该说清。
```

### 8.2 ✅ 默认裁判模型是写死的（`metrics/judge/llm_judge.py:43-44`）

```python
DEFAULT_JUDGE_MODEL = 'Qwen/Qwen3-235B-A22B'
DEFAULT_API_URL = 'https://api-inference.modelscope.cn/v1/'
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

   ⟹ ⚠️ 另外注意默认 API 指向 ModelScope 的托管服务 ——
      ★★ 这意味着【默认配置下你的测试数据会发到外部服务】。
      ⟹ 🔑 做内部评测前先确认这一条合不合规。
```

### 8.3 ✅ 裁判的提示词长什么样（`llm_judge.py:12-29`，`DEFAULT_PROMPT_TEMPLATE` 原文）

```
Your job is to look at a question, a gold target, and a predicted answer, and return a letter "A" or "B" to indicate whether the predicted answer is correct or incorrect.

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

✅ 框架还提供了另一种模板 —— 打连续分（`llm_judge.py:32-41` 的 `DEFAULT_NUMERIC_SCORE_TEMPLATE`，节选）：

```
Begin your evaluation by providing a short explanation. Be as objective as possible.
After providing your explanation, you must rate the response on a scale of 0 (worst) to 1 (best) by strictly following this format: "[[rating]]", for example: "Rating: [[0.5]]"
```

```
   ⟹ ★★ 注意 `[[  ]]` 这个双方括号。
   ⚠️ 它不是装饰，是【为了让正则能唯一定位分数】：
      ★★★ 答案文本里可能到处都是数字，但 [[0.5]] 这种形状不会误撞。
   ⟹ ⚠️ 顺带注意这个模板和上面那个 A/B 模板【方向相反】：
      ★★ A/B 模板要求"不要有别的字"，这个却要求"先给一段简短解释"。
      ⟹ 🔑 因为打连续分需要它先想一想，⟹ 这是精度和成本的权衡。
   ⟹ ✅ 对应的枚举在 `constants.py:117-119`：

        class JudgeScoreType:
            NUMERIC = 'numeric'   # numeric score
            PATTERN = 'pattern'   # pattern matching score
```

---

## 9. ★★★ Arena Hard：一份可以逐行读的「两两对战」实现

★★ 这是全篇最推荐精读的一段源码，因为它把 [09 章](评测入门/09-LLM当裁判.md) 讲的**位置偏见对策**和 **Elo 评分**，用不到 100 行代码全实现了。

✅ `benchmarks/arena_hard/arena_hard_adapter.py:41` 的 docstring 原文：

```
- Two-game battle system (A vs B and B vs A)
```

### 9.1 ✅ 同一道题判两次，顺序对调（`arena_hard_adapter.py:104-118`）

```python
        # reference is baseline answer 'A', filtered_prediction is model answer 'B'
        prompt1 = GRADER_TEMPLATE.format(question=question, answer_1=reference, answer_2=filtered_prediction)
        # reverse the order
        prompt2 = GRADER_TEMPLATE.format(question=question, answer_1=filtered_prediction, answer_2=reference)

        # get grading response
        game1_response = self.llm_judge.judge(prompt1, system_prompt=GRADER_SYSTEM_PROMPT)
        game2_response = self.llm_judge.judge(prompt2, system_prompt=GRADER_SYSTEM_PROMPT)

        # parse grading response
        res1 = post_process_arenahard(game1_response)
        res2 = post_process_arenahard(game2_response)

        score1 = get_judge_score(res1, reverse=True)
        score2 = get_judge_score(res2, reverse=False)
```

✅ 然后把两局取平均（`arena_hard_adapter.py:138`）：

```python
        score.value = {'score': (score1 + score2) / 2}
```

```
   ⟹ 🔑🔑🔑 这几行代码在对付一个真实存在的毛病：

   ★★★【位置偏见（position bias）】
      = 大模型当裁判时，会系统性地偏爱【排在前面】的那个答案。
      ⚠️ 注意"系统性"三个字：它不是随机噪声，
         ⟹ 你跑一万次也不会自己抵消掉。

   ⟹ ★★ 对策就是上面这个：同一对答案，正着判一次、反着判一次，取平均。
   ⟹ ⚠️ 代价：★★★【裁判调用次数直接翻倍，成本翻倍】。
   ⟹ 🔑 这就是评测里典型的"用钱买正确性"。

   ⟹ ★★ 注意 score1 / score2 那两行的 reverse 参数【是反着的】：
      第一局里基线排在 A 位，所以要 reverse=True 把分数翻过来，
      才能得到【被测模型】的分；第二局被测模型本来就在 A 位，reverse=False。
      ⚠️ 这两个参数写成一样的话，两局就不是互相校正而是互相叠加同一个偏见。
```

### 9.2 ✅ 五档判决怎么变成分数（`benchmarks/arena_hard/utils.py:21-53`）

```python
def get_judge_score(result, reverse=False):
    # Base score mapping - using finer-grained scores
    if not reverse:
        score_mapping = {
            'A=B': 0.5,  # Tie
            'A>B': 0.75,  # A slightly wins
            'A>>B': 1.0,  # A significantly wins
            'B>A': 0.25,  # B slightly wins
            'B>>A': 0.0,  # B significantly wins
        }
    else:
        score_mapping = {
            'A=B': 0.5,
            'A>B': 0.25,
            'A>>B': 0.0,
            'B>A': 0.75,
            'B>>A': 1.0,
        }

    base_score = score_mapping.get(result, 0.5)

    return base_score
```

★ 把两份映射表摆在一起，镜像关系一眼可见：

| 裁判的判决 | 含义 | `reverse=False` | `reverse=True` |
|---|---|---|---|
| `A>>B` | A 显著更好 | 1.00 | 0.00 |
| `A>B` | A 略好 | 0.75 | 0.25 |
| `A=B` | 平 | 0.50 | 0.50 |
| `B>A` | B 略好 | 0.25 | 0.75 |
| `B>>A` | B 显著更好 | 0.00 | 1.00 |
| ⚠️ **其他 / 解析不出来** | —— | ★★★ **0.50**（默认值） | ★★★ **0.50** |

```
   ⟹ ★★ 三个要点：

   ★★★ ① 判决不是二元的（赢/输），是【五档】：
      显著更好 / 略好 / 平 / 略差 / 显著差 → 1.0 / 0.75 / 0.5 / 0.25 / 0.0
      ⟹ ⚠️ 为什么要五档？因为"略好"和"碾压"是两码事，
         ★★ 硬压成二元会丢掉大量信息。

   ★★★ ② 两份映射表是【左右镜像】的，严格对称。

   ★★★ ③ `score_mapping.get(result, 0.5)` —— 【看这个默认值】。
      ⟹ ⚠️ 裁判要是没输出可识别的判决（比如它开始长篇大论了），
         ★★ 这一局按【平局 0.5】算。
      ⟹ 🔑 这又是一次"抽取失败被静默吞掉"：
         和 lm_eval 的 `"[invalid]"`、SWE-bench 的 `APPLY_PATCH_FAIL`
         是同一类事，⚠️ 只是这里【连个标记都没留】。
      ⟹ ★★★ 后果比那两个更坏：那两个判 0 分（偏低），
         这个判 0.5（偏向平局）⟹ 系统性地把强模型和弱模型都往中间拉。
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
   ⟹ ★ 顺带对比：lm_eval 的 GSM8K flexible-extract 专门把
      group_select 设成 -1（取最后一个），⟹ 两边做了相反的选择。
```

### 9.3 ✅ 最后汇总成 Elo 和胜率（`arena_hard_adapter.py:148-172`）

```python
    def aggregate_scores(self, sample_scores: List[SampleScore]) -> List[AggScore]:
        import pandas as pd

        from .utils import compute_mle_elo, get_battles_from_row, get_bootstrap_result, get_win_rate_column

        battles = pd.concat([get_battles_from_row(res.score.metadata['battle_result']) for res in sample_scores])

        bootstrap_online_elo = compute_mle_elo(battles)

        stats = pd.DataFrame()
        ...
        score = get_win_rate_column(stats, 'score', 'gpt4-0314').at['test_model']

        return [AggScore(score=score, metric_name='winrate', num=len(sample_scores))]
```

✅ 锚点模型写在 `arena_hard_adapter.py:121-122`：

```python
            'model_a': 'gpt4-0314',
            'model_b': 'test_model',
```

✅ 并且 `compute_mle_elo()` 里把锚点硬钉在 1000 分（`utils.py:150-152`）：

```python
    # set anchor as gpt4-0314 = 1000
    if 'gpt4-0314' in models.index:
        elo_scores += 1000 - elo_scores[models['gpt4-0314']]
```

```
   ★ 名词一次说清：

   ★★★ Elo（埃洛评分，源自匈牙利人名 Árpád Élő）
      = 国际象棋用的那套天梯分算法。
      ⟹ 核心思想：★★ 赢强的对手加分多，赢弱的对手加分少。
      ⟹ ⚠️ 它只能算出【相对强弱】，算不出绝对水平。

   ★★★ MLE（Maximum Likelihood Estimation，最大似然估计）
      = 一种统计方法：⟹ 找一组 Elo 分，使得
        "按这组分打，最可能打出我们实际观察到的那些胜负结果"。
      ⚠️ 比传统的"一局一局往上加"更稳，因为它一次性看全部对局。

   ★★★ 锚点模型（anchor）= `gpt4-0314`，被钉在 1000 分。
      ⟹ 所有模型都跟它对战，报的是【对它的胜率】。
      ⟹ ⚠️⚠️ 这条必须记住：
         ★★★【锚点一换，所有历史分数全部作废，不能跨版本比较。】
```

### 9.4 ⚠️⚠️ 一条必须指出的勘误：这里【没有】置信区间

```
   ⚠️ 上面那段代码里有两个容易骗人的地方：

   ★★ ① 变量名叫 `bootstrap_online_elo`，
      ⟹ 但它的值是 `compute_mle_elo(battles)` ——
         ★★★ 一次普通的 MLE 拟合，【没有 bootstrap】。

   ★★ ② `get_bootstrap_result` 确实被 import 了（:151），
      ⟹ ✅ 但在 arena_hard 的 aggregate_scores 里【从头到尾没被调用过】。
      ⟹ ★★★ 所以 arena_hard 这个基准输出的是【一个点估计】，
         ⚠️ 不带任何置信区间。

   ⟹ ✅ 真正调用了 bootstrap 的是另一个基准：
      general_arena（`benchmarks/general_arena/general_arena_adapter.py:314`）。
      ⟹ ⚠️ 它的 get_bootstrap_result 还多一个 baseline_model 参数。
```

★ 既然提到了，把 bootstrap 这个名词说清 —— ⟹ 它是回答「★★★ 榜单上差 1 分到底算不算差距？」这个问题的标准做法：

```
   ★★★ bootstrap（自助法）= 从已有对局里【有放回地随机重抽】很多次，
      每次都重算一遍 Elo。
      ⟹ ⚠️ 目的：得到【置信区间】，
         也就是"这个分上下浮动多少算正常"。
      ✅ 实现就在 utils.py:156-163，逻辑很短：
         battles.sample(frac=1.0, replace=True) 重抽 → 重算 → 取中位数。

   ⟹ 🔑🔑 这直接关系到 [11 章](评测入门/11-怎么读懂一张榜单.md)
      那个核心问题：★★★【榜单上差 1 分，到底算不算差距？】
      ⟹ 有了 bootstrap 你才知道答案，⚠️ 通常是"不算"。

   ⟹ ⚠️ 所以用 arena_hard 的胜率排名次时要格外小心：
      ★★★ 它给你一个很精确的数字，但【没告诉你这个数字有多抖】。
      ⟹ 🔑 一个最省事的替代判据：按题数算标准误
         `SE ≈ √(p(1−p)/n)`，见 [评测入门 · 附-速查表](评测入门/附-速查表.md) J 节。
```

---

## 10. ⚠️ 三个框架的边界：它们都不做什么

```
   ❌ 都不负责【生成】被评的答案（除了 agentic 那条线）
      ⟹ ★★ 你得先有模型或 API。

   ❌ 都不能替你判断【题目本身合不合理】
      ⟹ ⚠️ 数据集里的错题、烂题，框架照跑不误。

   ❌ 都不能替你排除【数据污染】
      ⟹ ★★★ 见 [06 章](评测入门/06-数据污染.md)。
      ⚠️ lm_eval 有个 filters/decontamination.py，
         OpenCompass 有个 icl_mink_percent_inferencer.py，
         ⟹ 但那些只是工具，是否污染仍然要你自己判断。

   ❌ ★★★ 都不会主动告诉你【这个分数有多抖】
      ⟹ ⚠️ 见 §9.4：连写了 bootstrap 函数的框架，默认路径上也没调用它。
      ⟹ 🔑 噪声下限得你自己算。

   ❌ 都不会告诉你【这个分数对你的业务意味着什么】
      ⟹ 🔑 这是 [12 章](评测入门/12-自己搭一套评测.md) 存在的理由。
```

---

## 11. 一句话总结

```
   ★★★ 如果你只想跑个标准分：用 lm-eval-harness。
   ★★★ 如果你有集群、要一次评几十个模型：用 OpenCompass。
   ★★★ 如果你要做主观对战评测、或者还想顺便压测服务：用 EvalScope。

   ⟹ 🔑🔑 但真正该带走的不是"选哪个"，而是这两个观察：

   ★★★ ① 三个团队独立设计，最后都收敛到了
      【判别式 / 生成式 / agentic】这同一个三分法。
   ⚠️ 框架的名字会变，接口会变，
      ⟹ ★★ 但这三种问模型的方式，是问题本身决定的。

   ★★★ ② 同一个 GSM8K，三家的 shot 数是 5 / 0 / 4，
      提示词两套，答案格式两套，判分器三套（§2.4）。
   ⟹ ⚠️⚠️ 所以跨框架比分数，比的从来不只是模型。
      ★★ 换框架 = 换考卷 + 换判卷标准 + 换抽取规则。
      ⟹ 🔑 唯一正确的做法是：【自己在同一套框架、同一份配置里重跑一遍】。
```

---

> 相关：[lm-eval-harness拆解.md](lm-eval-harness拆解.md) ｜ [SWE-bench拆解.md](SWE-bench拆解.md) ｜ [评测入门 09 LLM 当裁判](评测入门/09-LLM当裁判.md) ｜ [评测入门 11 怎么读懂一张榜单](评测入门/11-怎么读懂一张榜单.md)
> 返回 [harness评测框架/](README.md)
