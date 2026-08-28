# lm-evaluation-harness 拆解

> ★ 版本：`lm_eval` 0.4.12（源码在 [源码/lm_eval-0.4.12/](源码/lm_eval-0.4.12/)）
> ★★ 定位：**判别式 + 生成式** 评测的事实标准。Hugging Face Open LLM Leaderboard 用的就是它。
>
> ⚠️ 它**不做** agentic 评测（SWE-bench 那类）—— 那是 [SWE-bench拆解.md](SWE-bench拆解.md) 的事。
>
> 标记：✅ = 源码原文可查证 ｜ ⚠️ = 我的推断 ｜ ★ = 重点 ｜ 🔑 = 关键结论

---

## 1. 一句话：它解决的是一个"组合爆炸"问题

```
   ★ 问题规模：
      ✅ lm_eval/tasks/ 下有 218 个任务目录
      ✅ lm_eval/models/ 下有 28 个模型后端文件
         （huggingface / vllm / sglang / openai / anthropic / litellm /
           megatron / nemo / trtllm / gguf / mamba / neuron …）

   ⚠️ 朴素做法要写 218 × 28 = 6,104 套适配代码。
   ⟹ 🔑 实际代码量远小于此。★★★ 靠的就是下面这个设计。
```

---

## 2. ★★★ 核心设计：三个原语构成的"窄腰"

```
              218 个任务（MMLU / GSM8K / HellaSwag / …）
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │   ★★★ 只有三个原语（narrow waist）    │
        │                                       │
        │   ① loglikelihood                    │
        │   ② loglikelihood_rolling            │
        │   ③ generate_until                   │
        └───────────────────────────────────────┘
                            │
                            ▼
              28 个模型后端（HF / vLLM / OpenAI / …）

   ⟹ 🔑🔑 上面加一个任务，不用动模型代码；
      下面加一个后端，只需实现三个方法。
      ★★★ 复杂度从【乘法】降成了【加法】。
```

### ✅ 源码证据①：抽象基类只要求三个方法

`lm_eval/api/model.py`：

```
   L25   class LM(abc.ABC):
   L40       def loglikelihood(self, requests) -> list[tuple[float, bool]]
   L58       def loglikelihood_rolling(self, requests) -> list[float]
   L100      def generate_until(self, requests) -> list[str]
```

```
   ⟹ ★★ 一个新后端（比如接入一个新的推理引擎）
      只要把这三个方法实现出来，218 个任务立刻全部可跑。
```

### ✅ 源码证据②：★★★ 调度是一行 `getattr`

`lm_eval/evaluator.py:596`：

```python
        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)
```

```
   ⚠️ 解释这一行为什么重要：

   ★ reqtype 是字符串，取值只可能是那三个原语名之一。
   ★★ getattr(lm, "generate_until") 就是"把 lm 对象上叫这个名字的方法拿出来"。
   ⟹ 🔑🔑 ★★★ 整个框架的【任务侧】和【模型侧】，
      就是靠这一行字符串查表连起来的。

   ⟹ 这就是"窄腰"最直白的物理体现：
      ⚠️ 腰细到可以用一个字符串索引完成分派。
```

### ✅ 源码证据③：原语的类型定义

`lm_eval/api/instance.py`：

```
   L5-7    OutputType = Literal[
               "loglikelihood", "loglikelihood_rolling",
               "generate_until", "multiple_choice"
           ]
   L11     @dataclass class Instance:
```

```
   ⚠️ 这里有个容易困惑的点，必须说清：
   ★ 列表里有四个值，但原语只有三个。multiple_choice 不是第四个原语。

   ✅ 证据在 evaluator.py:571-576：
      # "multiple_choice" task types dispatch (several) "loglikelihood" request types
      reqtype = ("loglikelihood"
                 if task.OUTPUT_TYPE == "multiple_choice"
                 else task.OUTPUT_TYPE)

   ⟹ 🔑 ★★ multiple_choice 是【任务层的类型】，
      到了模型层会被拆成【若干个 loglikelihood 请求】
      —— 一个选项一个请求，然后比大小。
   ⟹ 这正是 [02 章](评测入门/02-三种评测范式.md) 说的"判别式"的实现方式。
```

---

## 3. 一道题的完整流动路径

★★ 把 `lm_eval --tasks gsm8k` 这条命令拆成六步。

```
   ┌─ ① 加载任务定义 ─────────────────────────────────────┐
   │  读 tasks/gsm8k/gsm8k.yaml                            │
   │  ⟹ 知道了：数据集在哪、几 shot、怎么拼提示词、        │
   │     用哪个原语、怎么抽答案、怎么算分                   │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ② 拼提示词（fewshot_context）───────────────────────┐
   │  ✅ api/task.py:447  def fewshot_context(...)         │
   │  ⟹ 把 N 条示例 + 当前题目拼成一个字符串              │
   │  ★★★ 04 章说的头号坑就在这一步                        │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ③ 造请求（construct_requests）──────────────────────┐
   │  ✅ api/task.py:382  def construct_requests(...)      │
   │  ⟹ 产出 Instance 对象，标好 request_type            │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ④ ★★★ 分派给模型 ──────────────────────────────────┐
   │  ✅ evaluator.py:596  resps = getattr(lm, reqtype)(…) │
   │  ⟹ 模型的原始输出存回 req.resps                      │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ⑤ 抽取 / 过滤（apply_filters）──────────────────────┐
   │  ✅ evaluator.py:611  task.apply_filters()            │
   │  ✅ api/task.py:505   def apply_filters(...)          │
   │  ⟹ 正则把答案抠出来，存进 filtered_resps            │
   │  ★★★ 05 章说的"最脏的一步"                            │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ⑥ 打分 + 汇总 ───────────────────────────────────────┐
   │  ✅ api/task.py:403  def process_results(...)         │
   │  ✅ api/task.py:416  def aggregation(...)             │
   │  ⟹ 逐条对错 → 平均 → 输出                            │
   └───────────────────────────────────────────────────────┘
```

```
   ⟹ 🔑 记住 ④ 和 ⑤ 是分开的两步。★★★ 这个分离是有意义的：
      ⚠️ 模型的原始输出（resps）和抽取后的答案（filtered_resps）
         【同时被保存下来】。
      ⟹ 加 --log_samples 就能把两列都导出，
         ★★ 对比这两列是排查"抽取失败"的唯一可靠办法。
```

---

## 4. ★★★ 一个任务定义长什么样：逐字段拆 gsm8k.yaml

★★ 这是全文最值得细看的一段。✅ 以下是 `tasks/gsm8k/gsm8k.yaml` 的**完整原文**：

```yaml
tag:
  - math_word_problems
task: gsm8k
dataset_path: openai/gsm8k
dataset_name: main
output_type: generate_until
training_split: train
fewshot_split: train
test_split: test
doc_to_text: "Question: {{question}}\nAnswer:"
doc_to_target: "{{answer}}"
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: false
    regexes_to_ignore:
      - ","
      - "\\$"
      - "(?s).*#### "
      - "\\.$"
generation_kwargs:
  until:
    - "Question:"
    - "</s>"
    - "<|im_end|>"
  do_sample: false
  temperature: 0.0
repeats: 1
num_fewshot: 5
filter_list:
  - name: "strict-match"
    filter:
      - function: "regex"
        regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
      - function: "take_first"
  - name: "flexible-extract"
    filter:
      - function: "regex"
        group_select: -1
        regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
      - function: "take_first"
metadata:
  version: 3.0
```

### 逐字段解释：★★★ 这 40 行里藏着前 5 章的每一个坑

```
┌────────────────────────┬──────────────────────────────────────────────┐
│ 字段                    │ 它对应教程里的哪个坑                          │
├────────────────────────┼──────────────────────────────────────────────┤
│ output_type:           │ ★★★ 03 章的三原语之一。                      │
│   generate_until       │ ⟹ 说明这是【生成式】评测，需要抽取。          │
├────────────────────────┼──────────────────────────────────────────────┤
│ num_fewshot: 5         │ ★★★ 04 章。5-shot 是 GSM8K 的惯例。          │
│ fewshot_split: train   │ ⟹ 示例从训练集里取（不能从测试集取，会泄漏）  │
├────────────────────────┼──────────────────────────────────────────────┤
│ doc_to_text:           │ ★★★ 04 章。提示词模板。                       │
│  "Question: {{q}}\n    │ ⚠️ 改一个字，分数就变。别人报的 GSM8K 分数    │
│   Answer:"             │    如果模板不同，【不可直接比较】。            │
├────────────────────────┼──────────────────────────────────────────────┤
│ until: ["Question:",   │ ★★ 02 章。停止符。                            │
│   "</s>", "<|im_end|>"]│ ⟹ 模型答完一题会想继续编下一题，              │
│                        │    看到 "Question:" 就掐掉。                  │
│                        │ ⚠️ 停止符设错 = 答案被截断 或 输出爆炸。      │
├────────────────────────┼──────────────────────────────────────────────┤
│ do_sample: false       │ ★★★ 10 章。greedy 解码，可复现。             │
│ temperature: 0.0       │ ⚠️ 但 R1 那类长输出推理模型不适合这么跑。     │
├────────────────────────┼──────────────────────────────────────────────┤
│ repeats: 1             │ ★★ 每题只跑一遍。                             │
│                        │ ⟹ 对比 DeepSeek 的 AIME：T=0.7 跑 16 遍取平均│
├────────────────────────┼──────────────────────────────────────────────┤
│ regexes_to_ignore:     │ ★★★ 05 章最经典的失败模式。                  │
│   - ","                │ ⟹ 模型答 "1,000"，标准答案是 "1000"。         │
│   - "\\$"              │ ⚠️ 不忽略逗号，这题就【判错】——              │
│                        │    但模型明明算对了。                          │
│   - "(?s).*#### "      │ ⟹ 把 "#### " 之前的推理过程全部丢掉           │
│   - "\\.$"             │ ⟹ 去掉结尾句号                                │
└────────────────────────┴──────────────────────────────────────────────┘
```

### ★★★ 最精妙的设计：两个并行的 filter

```
   ⚠️ 注意 filter_list 里有【两个】条目，不是一个：

   ┌─ strict-match（严格匹配）────────────────────────────┐
   │  regex_pattern: "#### (\-?[0-9\.\,]+)"               │
   │  ⟹ ★ 只认 GSM8K 官方格式：答案必须写在 "#### " 后面 │
   │  ⚠️ 模型不遵守这个格式 ⟹ 抽取失败 ⟹ 判错           │
   └──────────────────────────────────────────────────────┘
   ┌─ flexible-extract（宽松抽取）────────────────────────┐
   │  regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"        │
   │  group_select: -1                                     │
   │  ⟹ ★★ 抓输出里【最后一个】数字，不管格式            │
   └──────────────────────────────────────────────────────┘

   ⟹ 🔑🔑🔑 这两个 filter 跑在【同一次模型输出】上，
      ★★★ 所以你会得到【同一次运行的两个分数】。

   ⟹ ★★★ 两个分数之间的差距，就是【纯粹由抽取规则造成的分数损失】。

   ┌───────────────────────────────────────────────────────┐
   │  ⚠️ 这直接量化了 05 章那句话：                        │
   │  ★★★【抽取失败 ≠ 答错，但在分数上完全等价】          │
   │                                                        │
   │  strict 45%  /  flexible 62%                          │
   │  ⟹ 那 17 个点【不是模型不会做】，                     │
   │     是它没按 "#### " 的格式写答案。                    │
   └───────────────────────────────────────────────────────┘

   ⟹ 🔑 实践建议：★★★ 看到别人报 GSM8K 分数，
      先问是 strict-match 还是 flexible-extract。
      ⚠️ 这两个数差十几个点是常事。
```

---

## 5. 对照组：MMLU 的任务定义长什么样

✅ `tasks/mmlu/default/_default_template_yaml` 完整原文：

```yaml
dataset_path: cais/mmlu
test_split: test
fewshot_split: dev
fewshot_config:
  sampler: first_n
output_type: multiple_choice
doc_to_text: "{{question.strip()}}\nA. {{choices[0]}}\nB. {{choices[1]}}\nC. {{choices[2]}}\nD. {{choices[3]}}\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
```

```
   ⟹ ★★★ 和 gsm8k.yaml 对比，三个决定性的差别：

   ① output_type: multiple_choice（不是 generate_until）
      ⟹ ★★★ 走【判别式】路线：不让模型说话，只比四个选项的概率。

   ② ★★★【完全没有 filter_list】
      ⚠️ 因为根本不需要抽取 —— 概率最高的那个选项就是模型的答案。
      ⟹ 🔑🔑 这一条把 02 章的核心区别讲透了：
         ★★★ 判别式评测【结构上不可能出现抽取失败】。

   ③ ★★★【完全没有 generation_kwargs】
      ⚠️ 因为不生成，所以没有温度、没有停止符、没有采样。
      ⟹ 🔑 这就是为什么判别式评测【可复现性天然更高】。

   ④ fewshot_config: sampler: first_n
      ⟹ ★★ 示例取 dev 集的前 N 条，【不随机】。
      ⟹ 🔑 这是为了可复现：随机选示例 = 每次跑分数都不同。
```

```
   ┌──────────────┬───────────────────┬──────────────────┐
   │              │ gsm8k.yaml        │ MMLU template    │
   ├──────────────┼───────────────────┼──────────────────┤
   │ 原语         │ generate_until    │ multiple_choice  │
   │              │                   │ (→ loglikelihood)│
   │ 需要抽取     │ ★★★ 是（两套）   │ ❌ 否            │
   │ 采样参数     │ greedy, T=0       │ ⚠️ 不适用        │
   │ 停止符       │ 三个              │ ⚠️ 不适用        │
   │ few-shot     │ 5                 │ 由外层配置        │
   │ 示例选取     │ 未指定            │ first_n（确定）  │
   │ 闭源 API 可跑│ ✅ 可以           │ ⚠️ 通常不行      │
   └──────────────┴───────────────────┴──────────────────┘
```

---

## 6. 过滤器家族（05 章的实现细节）

✅ `lm_eval/filters/` 目录下的文件：

```
   ★ extraction.py     —— 正则抽取、答案定位（★★★ 最常用）
   ★ selection.py      —— take_first、多数投票（self-consistency）
   ★ transformation.py —— 大小写、标点等归一化
   ★ decontamination.py—— 去污染相关
   ★ custom.py         —— 自定义
```

✅ `filters/extraction.py` 的关键三行：

```
   L16    class RegexFilter(Filter):
   L26        默认正则模式
   L28        fallback: str = "[invalid]"
```

```
   ⟹ 🔑🔑 ★★★ 这个 `fallback = "[invalid]"` 是全框架最该被看见的一行。

   ⚠️ 含义：正则没匹配上时，抽取结果被填成字符串 "[invalid]"。
   ⟹ 它和标准答案永远不相等 ⟹ ★★★【判错】。

   ⟹ 所以：
      ┌──────────────────────────────────────────────────┐
      │  "模型答错了"      → filtered_resps = "42"（错的）│
      │  "抽取失败了"      → filtered_resps = "[invalid]" │
      │                                                   │
      │  ★★★ 两者在最终分数上【完全等价】，              │
      │  ⚠️ 但在 --log_samples 的输出里【清晰可辨】。    │
      └──────────────────────────────────────────────────┘

   ⟹ 🔑 排查动作：统计 filtered_resps 里 "[invalid]" 的占比。
      ★★★ 超过 5% 就说明抽取规则和这个模型的输出格式不匹配，
      ⚠️ 此时分数低【不是模型的问题】。
```

---

## 7. 命令行：七个你真正会用到的参数

✅ 全部出自 `lm_eval/_cli/run.py`，行号已核对：

| 参数 | 行 | ✅ 官方 help 原文 | 什么时候用 |
|---|---|---|---|
| `--tasks` | L66 | — | 指定跑哪些任务 |
| `--model_args` | L86 | — | 传给后端的参数（模型路径等） |
| `--apply_chat_template` | L95 | "Apply chat template to prompts (optional template name)" | ★★★ chat 模型必加 |
| `--limit` / `-L` | L104 | "Limit examples per task (integer count or fraction)" | ★★ 冒烟测试 |
| `--num_fewshot` | L123 | — | 改 shot 数 |
| `--log_samples` / `-s` | L179 | "Save all model outputs and documents for post-hoc analysis" | ★★★ 排查必加 |
| `--include_path` | L238 | "Additional directory for external tasks" | ★★ 加载自己写的任务 |

```
   ⟹ 🔑 三个参数的重要性远超其他：

   ★★★ --apply_chat_template
      ⚠️ 不加 = 用 base 模型的拼法去问 chat 模型 = 04 章的头号坑。
      ⟹ 分数可能腰斩，而且【看起来像模型不行】。

   ★★★ --log_samples
      ⟹ 唯一能看到 resps（原始输出）和 filtered_resps（抽取结果）
         这两列的办法。★★ 不加它，你对分数为什么低【一无所知】。

   ★★  --limit 20
      ⟹ 先跑 20 条确认流程通了，再跑全量。省钱省时间。
```

---

## 8. 自己加一个任务：最小步骤

```
   ① 建一个目录，比如 ./my_tasks/
   ② 写一个 yaml，最小字段：

      task: my_task
      dataset_path: json                    # 或 HuggingFace 数据集名
      dataset_kwargs:
        data_files: /abs/path/to/my.jsonl
      test_split: train
      output_type: generate_until
      doc_to_text: "问题：{{input}}\n答案："
      doc_to_target: "{{gold}}"
      generation_kwargs:
        until: ["\n问题："]
        do_sample: false
        temperature: 0.0
      metric_list:
        - metric: exact_match
          aggregation: mean
          higher_is_better: true
      filter_list:
        - name: "flexible"
          filter:
            - function: "regex"
              regex_pattern: "([ABCD])"
            - function: "take_first"

   ③ 跑：
      lm_eval --model hf --model_args pretrained=<model> \
              --include_path ./my_tasks --tasks my_task \
              --limit 20 --log_samples --output_path ./out
```

```
   ⚠️ ★★★ 但先读 [12 章](评测入门/12-自己搭一套评测.md) §12.5：

   🔑 如果你只是"跑一个模型、用 generate_until 打分"，
      ★★ 手写 100 行 Python + JSONL 比配 yaml 更灵活、更好调试。

   ⟹ 值得用 lm_eval 的三种情况：
      ① 你要测的基准框架里已经实现了（省掉重写的功夫）
      ② ★★★ 你需要 loglikelihood（判别式）——手写这个很麻烦
      ③ 你要横向比多个后端（HF / vLLM / API）而不想写适配层
```

---

## 9. 这个框架的边界（它做不了什么）

```
   ❌ ★★★ agentic 评测
      ⟹ 它的三个原语里没有"执行代码""开容器""多轮工具调用"。
      ⟹ SWE-bench 那类必须用另一套东西，见 [SWE-bench拆解.md](SWE-bench拆解.md)。

   ❌ ★★ LLM-as-a-Judge 的完整工作流
      ⟹ 可以勉强用 generate_until 拼，但没有内建的
         两两对战、位置交换、Elo 汇总。
      ⟹ 这些在 EvalScope / OpenCompass 里是一等公民，
         见 [OpenCompass与EvalScope拆解.md](OpenCompass与EvalScope拆解.md)。

   ⚠️ ★★ 闭源 API 上的判别式评测
      ⟹ 不是框架的问题，是 API 不返回 logprobs。
      ⟹ 🔑 结构性限制，换框架也解决不了。
```

---

## 10. 一句话总结

```
   ★★★ lm-evaluation-harness 的全部价值，在于它证明了一件事：
   🔑🔑 上百个评测集和几十种模型后端之间，
        只需要【三个原语】就能连通。

   ⟹ 而这三个原语不是设计出来的巧合 ——
      ✅ DeepSeek-V3 技术报告独立提出的三分法
      （perplexity-based / generation-based / language-modeling-based）
      ★★★ 和它们【一一对应】。
      ⟹ 见 [10 章 §10.1](评测入门/10-DeepSeek的评测口径.md)。
```

---

> 相关：[评测入门 03 三个原语](评测入门/03-三个原语.md) ｜ [05 答案抽取与打分](评测入门/05-答案抽取与打分.md)
> 返回 [harness评测框架/](README.md)
