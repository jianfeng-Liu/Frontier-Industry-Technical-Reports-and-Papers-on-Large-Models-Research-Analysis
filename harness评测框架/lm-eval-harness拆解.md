# lm-evaluation-harness 拆解

> ★ 版本口径：`lm_eval` **0.4.12**（源码在 [源码/lm_eval-0.4.12/](源码/lm_eval-0.4.12/)，本文所有行号都是对着这一份数的）
> ★★ 定位：**判别式 + 生成式** 评测的事实标准。Hugging Face Open LLM Leaderboard 用的就是它。
>
> ⚠️ 它**不做** agentic 评测（SWE-bench 那类）—— 那是 [SWE-bench拆解.md](SWE-bench拆解.md) 的事。
>
> 标记：✅ = 源码原文可查证 ｜ ⚠️ = 我的推断 ｜ ★ = 重点 ｜ 🔑 = 关键结论
>
> ⚠️⚠️ **行号会过期**。这个项目迭代很快（0.4.x 内部就改过目录结构，比如命令行从 `__main__.py` 搬到了 `lm_eval/_cli/`）。★★ 对不上时以你手上那份源码为准，并把版本号一起记下来。

---

## 1. 一句话：它解决的是一个「组合爆炸」问题

```
   ★ 问题规模（✅ 对 0.4.12 实际数过）：
      ✅ lm_eval/tasks/ 下有 212 个任务目录
         ⚠️ 目录里还有 6 个散文件（README 等），
            所以 `ls | wc -l` 会得到 218，别把那 6 个也算成任务
      ✅ lm_eval/models/ 下有 26 个文件带 @register_model 装饰器，
         一共注册了 36 个后端名
         （hf / vllm / sglang / openai-chat-completions / anthropic-chat /
           litellm / megatron_lm / nemo_lm / trtllm / gguf / mamba_ssm /
           neuronx / watsonx_llm / winml …）

   ⚠️ 朴素做法要写 212 × 36 ≈ 7,600 套适配代码。
   ⟹ 🔑 实际代码量远小于此。★★★ 靠的就是下面这个设计。
```

---

## 2. ★★★ 核心设计：三个原语构成的「窄腰」

★ 名词：**窄腰（narrow waist）**= 一种架构比喻。沙漏中间最细的那一段，上下两边都只认它，所以两边可以各自随便加东西。互联网的 IP 协议就是经典例子。

```
              212 个任务（MMLU / GSM8K / HellaSwag / …）
                            │
                            ▼
              ★★★ 只有三个原语（narrow waist）
                            │
                ├── ① loglikelihood
                ├── ② loglikelihood_rolling
                └── ③ generate_until
                            │
                            ▼
              36 个模型后端名（HF / vLLM / OpenAI / …）

   ⟹ 🔑🔑 上面加一个任务，不用动模型代码；
      下面加一个后端，只需实现三个方法。
      ★★★ 复杂度从【乘法】降成了【加法】。
```

### ✅ 源码证据①：抽象基类只要求三个方法

`lm_eval/api/model.py`：

```
   L25   class LM(abc.ABC):
   L40       def loglikelihood(self, requests: list["Instance"]) -> list[tuple[float, bool]]
   L58       def loglikelihood_rolling(self, requests: list["Instance"]) -> list[float]
   L100      def generate_until(self, requests: list["Instance"]) -> list[str]
```

★ 三个原语分别在问模型什么：

| 原语 | 中文 | ★★ 它问模型的问题 | 返回什么 | 典型用在 |
|---|---|---|---|---|
| `loglikelihood` | 对数似然 | 「**给定这段上文，你觉得接下来是这段指定文字的可能性有多大**」 | `(对数概率, 是否是贪心首选)` | ★★★ 选择题：每个选项算一次，比大小 |
| `loglikelihood_rolling` | 滚动对数似然 | 「**这整段文字在你眼里有多自然**」（不给上文） | 一个对数概率 | 困惑度类指标、语言建模评测 |
| `generate_until` | 生成直到 | 「**接着往下写，写到遇见这些停止符为止**」 | 生成出来的字符串 | ★★★ 数学题、代码题、开放问答 |

```
   ⟹ ★★ 一个新后端（比如接入一个新的推理引擎）
      只要把这三个方法实现出来，212 个任务立刻全部可跑。
   ⚠️ 反过来也成立：★★★ 后端【做不到】的事，
      整个框架就做不到 —— 见 §9 的第三条。
```

### ✅ 源码证据②：★★★ 调度是一行 `getattr`

`lm_eval/evaluator.py:584-596`：

```python
    # execute each type of request
    for reqtype, reqs in requests.items():
        ...
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

✅ 而那个字符串是从哪来的？`evaluator.py:560-562`：

```python
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)
```

⟹ ★★ 它来自**每个 `Instance` 对象自己带的 `request_type` 字段**，在造请求的时候就定好了。

### ✅ 源码证据③：原语的类型定义

`lm_eval/api/instance.py:5-7`：

```python
OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]
```

```
   ⚠️ 这里有个容易困惑的点，必须说清：
   ★ 列表里有四个值，但原语只有三个。multiple_choice 不是第四个原语。
```

✅ 真正的证据在 `lm_eval/api/task.py:1433-1445` —— 一个 `multiple_choice` 任务造出来的是**一串** `Instance`，每一个的 `request_type` 都被写成 `"loglikelihood"`：

```python
        if self.OUTPUT_TYPE == "multiple_choice":
            request_list = [
                Instance(
                    request_type="loglikelihood",   # ★★★ 就是这里
                    doc=doc,
                    arguments=arg,
                    idx=i,
                    **kwargs,
                )
                for i, arg in enumerate(arguments)
            ]

            return request_list
```

✅ 而那串 `arguments` 是一个选项一条（`api/task.py:1390`）：

```python
                arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]
```

```
   ⟹ 🔑 ★★ 所以 multiple_choice 是【任务层的类型】，
      到了模型层会被拆成【若干个 loglikelihood 请求】
      —— 一个选项一个请求，然后比大小。
   ⟹ 这正是 [02 章](评测入门/02-三种评测范式.md) 说的"判别式"的实现方式。

   ⚠️⚠️ 一条勘误：★★ evaluator.py:571-576 那段
      `reqtype = "loglikelihood" if task.OUTPUT_TYPE == "multiple_choice" else ...`
      【不是】真正的分派代码。
      ⟹ 它在 `if lm.world_size > 1:` 块里面，只用来算
         多卡场景下要补几个 padding 请求。
      ⟹ 🔑 真正定下 request_type 的地方就是上面那段 :1433-1445。
```

---

## 3. ★★★ 一次评测从入口到出分：每一步谁改了什么状态

★★ 这一节是全篇的主线。⚠️ 把 `lm_eval --tasks gsm8k` 这条命令拆开，看**一道题的状态是怎么一步步被改写的**。

### 3.1 ✅ 调用链

```
   命令行   lm_eval --model hf --tasks gsm8k --log_samples
        │
        ▼
   加载任务定义  读 tasks/gsm8k/gsm8k.yaml
        │   ⟹ 知道了：数据集在哪、几 shot、怎么拼提示词、
        │      用哪个原语、怎么抽答案、怎么算分
        ▼
   ConfigurableTask.__init__  api/task.py:757-776
        │   ★★ 把 yaml 里的 filter_list 编译成若干个 FilterEnsemble
        ▼
   fewshot_context()          api/task.py:447（基类）/ :933（Configurable）
        │   把 N 条示例 + 当前题目拼成一个字符串 ctx
        ▼
   construct_requests()       api/task.py:382（基类）/ :1362（Configurable）
        │   产出 Instance 对象，标好 request_type
        ▼
   【分派给模型】              evaluator.py:596
        │   resps = getattr(lm, reqtype)(cloned_reqs)
        │   ⟹ 模型原始输出写进 req.resps（:600）
        ▼
   task.apply_filters()       evaluator.py:611 → api/task.py:505 / :1160
        │   ⟹ 每个 FilterEnsemble 跑一遍，
        │      结果写进 req.filtered_resps[这把尺子的名字]
        ▼
   for filter_key in ...      evaluator.py:624
        │   ★★★ 按【尺子】循环 —— 有几把尺子就出几份分数
        ▼
   process_results()          api/task.py:403（基类）/ :1455（Configurable）
        │   逐条判对错
        ▼
   aggregation()              api/task.py:416 / :1666
        │   取平均
        ▼
   最终那张表：每个 (metric, filter) 组合一行
```

### 3.2 ★★★ 状态怎么变：逐步表

★★ 跟着 GSM8K 的一道真题走一遍。⚠️ 这张表是本篇最该记住的东西 —— **排查「分数为什么低」，就是回来找卡在哪一行**。

| 步 | 代码位置 | 读了什么 | ★★ 写出了什么状态 | ⚠️ 这一步出问题的典型表现 |
|---|---|---|---|---|
| 0 | 读 `gsm8k.yaml` | 配置文件 | `TaskConfig` 对象 | ★ yaml 里字段拼错 ⟹ 启动就报错（这是**好事**，早失败） |
| 1 | `ConfigurableTask.__init__` task.py:757 | `filter_list` | ★★★ `self._filters` = 一个 **FilterEnsemble 列表**（GSM8K 是 2 个） | ★★ yaml 里**没写** `filter_list` ⟹ 自动塞一个名叫 `"none"` 的默认尺子（:776） |
| 2 | 下载/加载数据集 | `dataset_path: openai/gsm8k` | 内存里的 doc 列表 | ⚠️ 网络 / 数据集版本不对 ⟹ 题目都变了，分数无从比较 |
| 3 | `fewshot_context()` task.py:447 | `num_fewshot: 5` + `fewshot_split: train` | 一个字符串 `ctx`：5 条示例 + 本题题干 | ★★★ **04 章的头号坑**：chat 模型没加 `--apply_chat_template`，这一步拼出的是 base 模型的格式 |
| 4 | `construct_requests()` task.py:382 | `ctx` + `output_type` | `Instance` 对象（`request_type="generate_until"`，`arguments=(ctx, generation_kwargs)`） | ★ `output_type` 写错 ⟹ 走错整条链路 |
| 5 | `getattr(lm, reqtype)()` evaluator.py:596 | `Instance.arguments` | ★★★ `req.resps` = **模型的原始输出**（一字不改） | ★★ 停止符设错 ⟹ 答案被截断，或输出爆炸烧钱 |
| 6 | `FilterEnsemble.apply()` api/filter.py:45-56 | `req.resps` | ★★★ `req.filtered_resps["strict-match"]` 和 `req.filtered_resps["flexible-extract"]` —— **两个 key** | ★★★ 正则没匹配上 ⟹ 填成 `"[invalid]"`，见 §6 |
| 7 | `for filter_key in ...` evaluator.py:624 | `filtered_resps` 的 **key 集合** | 对**每把尺子**各跑一遍第 8–9 步 | ⟹ 🔑 这就是「同一次运行出两个分数」的物理原因 |
| 8 | `process_results()` task.py:403 | `filtered_resps[filter_key]` + 标准答案 | 每题一个 0/1 | ★★ `regexes_to_ignore` 没配好 ⟹ `"1,000"` ≠ `"1000"`，算对的题被判错 |
| 9 | `acc["raw_metrics"][(metric, filter_key)]` evaluator.py:665 | 每题的 0/1 | ★★★ 按 **(指标, 尺子)** 二元组分桶 | —— |
| 10 | `aggregation()` task.py:416 | 一桶 0/1 | 一个百分数 | ⟹ 这才是你看到的那个数 |

```
   ⟹ 🔑 记住第 5 步和第 6 步是【分开的两步】。★★★ 这个分离是有意义的：
      ⚠️ 模型的原始输出（resps）和抽取后的答案（filtered_resps）
         【同时被保存下来】。
      ✅ 证据在 evaluator.py:645-648 —— --log_samples 导出的每条样本里
         同时有 "resps" 和 "filtered_resps" 两个字段。
      ⟹ ★★ 对比这两列是排查"抽取失败"的唯一可靠办法。

   ⟹ 🔑🔑 再记住第 7 步：★★★ 循环的是【尺子】，不是题。
      ⚠️ 所以 lm_eval 输出的从来不是"一个分数"，
         而是【每个 (指标, 尺子) 组合一个分数】。
      ⟹ 看到别人只报一个 GSM8K 数字，说明他省略了尺子名字。
```

---

## 4. ★★★ 一个任务定义长什么样：逐字段拆 gsm8k.yaml

★★ 这是全文最值得细看的一段。✅ 以下是 `tasks/gsm8k/gsm8k.yaml` 的**完整原文**（含源码里那行注释）：

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
doc_to_target: "{{answer}}" #" {{answer.split('### ')[-1].rstrip()}}"
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

### 4.1 ★★★ 配置项 → 运行时行为映射表

★★ 这张表是本节的核心：**每个 yaml 字段，在 §3.2 的第几步生效、改的是什么状态**。⟹ 🔑 看懂它，你就看懂了「为什么改一行配置分数就变」。

| yaml 字段 | ★★ 在第几步生效 | 它具体改了什么 | ⚠️ 配错的后果 |
|---|---|---|---|
| `dataset_path` / `dataset_name` / `test_split` | 第 2 步 | 决定**题目是哪些** | ⚠️ 换了版本 = 换了考卷，分数不可比 |
| `output_type: generate_until` | 第 4 步 | 决定 `Instance.request_type`，也就是走哪条原语 | ★★★ 决定了**需不需要抽取**（见 §5 的对照） |
| `num_fewshot: 5` | 第 3 步 | 拼几条示例进提示词 | ★★★ 改成 0 ⟹ 模型不知道要写成 `#### 66` ⟹ `strict-match` 会大面积抠不到 |
| `fewshot_split: train` | 第 3 步 | 示例从**训练集**取 | ★★ 从测试集取 = 泄漏，分数虚高 |
| `doc_to_text` | 第 3 步 | 提示词模板本体 | ★★★ 改一个字分数就变。别人报的 GSM8K 分数如果模板不同，**不可直接比较** |
| `doc_to_target` | 第 8 步 | 标准答案怎么从数据集字段里取 | ⚠️ 这里取的是整段 `answer`（含推理过程），靠 `regexes_to_ignore` 去掉前缀 |
| `generation_kwargs.until` | 第 5 步 | 停止符。模型答完一题会想继续编下一题，看到 `"Question:"` 就掐掉 | ★★ 设错 = 答案被截断，或输出爆炸烧钱 |
| `do_sample: false` / `temperature: 0.0` | 第 5 步 | greedy（贪心）解码，可复现 | ⚠️ 但 R1 那类长输出推理模型不适合这么跑，见 [10 章](评测入门/10-DeepSeek的评测口径.md) §10.7b |
| `repeats: 1` | 第 5 步 | 每题只跑一遍 | ★★ 对比 DeepSeek 的 AIME：T=0.6 跑 64 遍取平均 —— ⟹ 那种数和这种数**不是一回事** |
| `filter_list`（两条） | 第 1 步编译、第 6 步执行 | ★★★ 决定**怎么从输出里抠答案**，以及**出几份分数** | 见 §4.2，这是本篇最重要的一段 |
| `metric_list.metric: exact_match` | 第 8 步 | 判对错的方式：字符串完全相等 | ★ 换 metric 就是换判卷标准 |
| `regexes_to_ignore` | 第 8 步 | 比较**之前**先把这些模式从两边删掉 | ★★★ 见下面的 4.1b，这是 05 章最经典的失败模式 |
| `ignore_case: true` | 第 8 步 | 忽略大小写 | ★ 数学题上无所谓，文本题上影响很大 |
| `aggregation: mean` | 第 10 步 | 逐题 0/1 取平均 | —— |
| `metadata.version: 3.0` | —— | ★★ **任务定义自己的版本号** | ⟹ 🔑 引用 GSM8K 分数时，这个号也该一起报 |

### 4.1b ⚠️ `regexes_to_ignore` 这四条正则到底在干什么

| 正则 | 它删掉什么 | ★★ 为什么必须删 |
|---|---|---|
| `","` | 所有逗号 | ★★★ 模型答 `"1,000"`，标准答案是 `"1000"`。⚠️ 不删这题就**判错**，但模型明明算对了 |
| `"\\$"` | 美元符号 | 模型答 `"$18"`，答案是 `"18"` |
| `"(?s).*#### "` | 从开头一直到 `"#### "` 的全部内容 | ★★★ 把推理过程整段丢掉，只留最后那个数。`(?s)` 让 `.` 也能匹配换行 |
| `"\\.$"` | 结尾的句号 | 模型答 `"18."`，答案是 `"18"` |

```
   ⟹ 🔑🔑 这四行是一个很好的缩影：
      ★★★【判别"对不对"这件事，在生成式评测里是靠一堆正则糊出来的。】
   ⚠️ 少一条，一批算对的题就被判错；
      多一条、写宽了，一批答错的题就被判对。
   ⟹ ★★ 这就是 [05 章](评测入门/05-答案抽取与打分.md) 称它为"最脏的一步"的原因。
```

### 4.2 ★★★ 最精妙的设计：两把尺子在源码里是怎么并存的

⚠️ 注意 `filter_list` 里有【两个】条目，不是一个：

✅ 两条正则原文（放在代码块里，免得表格里的竖线被当成分隔符）：

```
   strict-match      regex_pattern: "#### (\-?[0-9\.\,]+)"
   flexible-extract  regex_pattern: "(-?[$0-9.,]{2,})|(-?[0-9]+)"
                     group_select:  -1
```

| | `strict-match`（严格匹配） | `flexible-extract`（宽松抽取） |
|---|---|---|
| ✅ 正则在找什么 | 必须出现 `#### ` 再跟一串数字 | 任意一串数字（两种写法取其一） |
| `group_select` | 默认 `0`（取第一个匹配） | **`-1`**（取**最后一个**匹配） |
| ★ 它认什么 | 只认 GSM8K 官方格式：答案必须写在 `"#### "` 后面 | 抓输出里**最后一个**数字，不管格式 |
| ⚠️ 模型不遵守格式时 | 抽取失败 ⟹ 填 `"[invalid]"` ⟹ 判错 | 大概率还能抠到 |
| ⚠️ 它自己的失效模式 | 对不写 `####` 的模型**系统性低估** | ★★ 模型最后又写了句「所以答案是 18 元，比预算 20 少」⟹ 抠到 `20`，**判错** |

#### ✅ 源码侧：两把尺子为什么能并存，只需要三行代码

★★★ 这是本节真正的看点。⟹ 关键在于 `filtered_resps` **不是一个值，是一个字典**。

✅ 第一步，`Instance` 上的字段（`api/instance.py:19-20`）：

```python
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)      # ★★★ 是 dict
```

✅ 第二步，yaml 里每个 `filter_list` 条目被编译成一个 `FilterEnsemble`，**名字就是 yaml 里的 `name`**（`api/task.py:757-769`）：

```python
        if self.config.filter_list is not None:
            self._filters = []
            for filter_config in self.config.filter_list:
                filter_name = filter_config["name"]          # "strict-match" / "flexible-extract"
                ...
                filter_pipeline = build_filter_ensemble(filter_name, components)
                self._filters.append(filter_pipeline)
```

✅ 第三步，每个 ensemble 把结果写进**以自己名字为 key 的槽位**（`api/filter.py:45-56`，含源码注释）：

```python
    def apply(self, instances: List[Instance]) -> None:
        resps, docs = zip(*((inst.resps, inst.doc) for inst in instances))
        resps, docs = list(resps), list(docs)

        for f in self.filters:
            # apply filters in sequence
            resps = f().apply(resps, docs)

        # add the end results after filtering to filtered_requests of their respective source instances.
        # has key `self.name`: each FilterEnsemble applied in a given run should use a different name.
        for inst, resp in zip(instances, resps):
            inst.filtered_resps[self.name] = resp            # ★★★ 就是这一行
```

✅ 第四步，打分的时候按 key 循环（`evaluator.py:624` 和 `:665`）：

```python
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps:
            ...
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
            ...
                for metric, value in metrics.items():
                    acc["raw_metrics"][(metric, filter_key)].append(value)
```

```
   ⟹ 🔑🔑🔑 四步连起来看，机制就一句话：

   ★★★ 模型只被调用【一次】（resps 只生成一遍），
        但 filtered_resps 这个字典里有【几个 key】，
        最终就出【几份分数】。

   ⟹ ⚠️ 代价为零：不多花一次模型调用，
      因为两把尺子量的是【同一批原始输出】。
   ⟹ ★★ 这也是为什么 lm_eval 敢默认给 GSM8K 配两把尺子 ——
      多一把尺子只多一点 CPU 正则时间。

   ⟹ 🔑 反过来说：★★★ 两个分数之间的差距，
      就是【纯粹由抽取规则造成的分数损失】，
      和模型能力一点关系都没有。
```

#### ★★ 这两把尺子到底能差多少：去看算过的例子

⚠️ **这里不重复造数字。** [评测入门 01 章](评测入门/01-为什么评测是个难题.md) §1.3b 用 GSM8K 真实的这两条正则，在一个 4 题的迷你集上**逐题算过一遍**：同一批模型输出，`strict-match` **25.0%** / 取第一个匹配 **50.0%** / `flexible-extract` **75.0%** —— ⟹ ★★★ **差 50.0 个百分点，只因为换了抽取规则**。

```
   ⟹ 🔑 那一节还顺手指出了一件更反直觉的事：
      ★★ flexible-extract 在那批题上分最高（75%），
      ⚠️ 但它在其中一道题上【把算对的判错了】——
      ⟹ 因为模型最后又多写了一个数字，而它取的是最后一个。

   ⟹ ★★★ 所以"宽松的尺子分数更高"不等于"宽松的尺子更准"。
      两把尺子都有自己的系统性偏差，只是方向相反。
```

⟹ 🔑 实践建议：★★★ 看到别人报 GSM8K 分数，先问是 `strict-match` 还是 `flexible-extract`。⚠️ 这两个数差十几个点是常事。★★ 如果对方报的是**一个**数字又没说尺子名，那个数字就不该进对比表。

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
   ⟹ ★★★ 和 gsm8k.yaml 对比，四个决定性的差别：

   ① output_type: multiple_choice（不是 generate_until）
      ⟹ ★★★ 走【判别式】路线：不让模型说话，只比四个选项的概率。
      ✅ 实现见 §2 源码证据③：拆成 4 个 loglikelihood 请求。

   ② ★★★【完全没有 filter_list】
      ⚠️ 因为根本不需要抽取 —— 概率最高的那个选项就是模型的答案。
      ⟹ 🔑🔑 这一条把 [02 章](评测入门/02-三种评测范式.md) 的核心区别讲透了：
         ★★★ 判别式评测【结构上不可能出现抽取失败】。
      ⚠️ 但注意：没写 filter_list 不等于没有 filter ——
         ✅ api/task.py:776 会塞一个名叫 "none" 的默认 take_first 尺子。
         ⟹ 所以输出表里你会看到 filter 列是 "none"。

   ③ ★★★【完全没有 generation_kwargs】
      ⚠️ 因为不生成，所以没有温度、没有停止符、没有采样。
      ⟹ 🔑 这就是为什么判别式评测【可复现性天然更高】。

   ④ fewshot_config: sampler: first_n
      ⟹ ★★ 示例取 dev 集的前 N 条，【不随机】。
      ⟹ 🔑 这是为了可复现：随机选示例 = 每次跑分数都不同。
      ⚠️ 反观 gsm8k.yaml 【没有】这一项。
```

★ 两份配置逐项对照：

| | `gsm8k.yaml` | MMLU `_default_template_yaml` |
|---|---|---|
| 原语 | `generate_until` | `multiple_choice` ⟹ 拆成多个 `loglikelihood` |
| 需要抽取 | ★★★ 是，而且配了**两套** | ❌ 否（只有默认的 `"none"` 尺子） |
| 采样参数 | greedy，T=0 | ⚠️ 不适用（不生成） |
| 停止符 | 三个 | ⚠️ 不适用 |
| few-shot 数 | 5（写死在任务里） | 由外层配置（`--num_fewshot`） |
| 示例选取 | ⚠️ 未指定 | `first_n`（确定） |
| 判分指标 | `exact_match` + 4 条忽略正则 | `acc`（比概率大小，没有正则） |
| ★★ 闭源 API 上能跑吗 | ✅ 可以（只要能生成文本） | ⚠️ **通常不行**，见 §9 第三条 |
| 任务定义版本 | `3.0` | `1.0` |

```
   ⟹ 🔑🔑 把这张表读完，一句话结论：
      ★★★ GSM8K 那一列里"可能出错的地方"比 MMLU 多了一倍，
      ⚠️ 而这些地方【全都不在模型里，全都在配置里】。
```

---

## 6. 过滤器家族（05 章的实现细节）

✅ `lm_eval/filters/` 目录下的文件：

| 文件 | 干什么 | ★ 常用函数名 |
|---|---|---|
| `extraction.py` | ★★★ 正则抽取、答案定位（最常用） | `regex`、`regex_pos` 等 |
| `selection.py` | `take_first`、多数投票（self-consistency 自洽投票） | `take_first` |
| `transformation.py` | 大小写、标点等归一化 | —— |
| `decontamination.py` | 去污染相关 | —— |
| `custom.py` | 自定义 | —— |

✅ `filters/extraction.py:24-37` 的构造函数原文 —— ★★★ 注意第三个参数：

```python
    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        """Compile `regex_pattern` and set the fallback for non-matches.

        `fallback` defines the output returned if no matches for the regex are located.
        """
```

✅ 它在哪里被用上（`extraction.py:47-58`）：

```python
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    ...
                    match = match.strip()
                else:
                    match = self.fallback          # ★★★ 没匹配上就填这个
```

```
   ⟹ 🔑🔑 ★★★ 这个 `fallback = "[invalid]"` 是全框架最该被看见的一行。

   ⚠️ 含义：正则没匹配上时，抽取结果被填成字符串 "[invalid]"。
   ⟹ 它和标准答案永远不相等 ⟹ ★★★【判错】。
```

★ 两种"判错"在代码里的区别：

| 情况 | `resps`（原始输出） | `filtered_resps`（抽取结果） | 最终分数 | ★★ 能分辨吗 |
|---|---|---|---|---|
| 模型**答错了** | `"…所以答案是 #### 42"` | `"42"`（错的） | 0 分 | —— |
| ★★★ **抽取失败了** | `"…所以答案是 42。"`（没写 `####`） | `"[invalid]"` | 0 分 | ✅ **能**，加 `--log_samples` 就看得见 |

```
   ⟹ ★★★ 两者在最终分数上【完全等价】，
      ⚠️ 但在 --log_samples 的输出里【清晰可辨】。
      ✅ 两列都在导出里：evaluator.py:645-648 的 "resps" 和 "filtered_resps"。

   ⟹ 🔑 排查动作：统计 filtered_resps 里 "[invalid]" 的占比。
      ★★★ 超过 5% 就说明抽取规则和这个模型的输出格式不匹配，
      ⚠️ 此时分数低【不是模型的问题】。

   ⟹ ⚠️ 顺手记一句 `group_select` 的默认值：0，也就是【取第一个匹配】。
      ★★ GSM8K 的 flexible-extract 专门把它设成 -1（取最后一个）。
      ⟹ 🔑 对会写思维链的模型，通常【最后一个】才是它的结论，
         ⚠️ 但"最后一个数字"也可能是它顺手提到的别的数（§4.2）。
```

---

## 7. 命令行：七个你真正会用到的参数

✅ 全部出自 `lm_eval/_cli/run.py`（⚠️ 0.4.12 里命令行已经搬到这个目录，老资料里的 `lm_eval/__main__.py` 是旧位置），行号已核对：

| 参数 | 行 | ✅ 官方 help 原文 | 什么时候用 |
|---|---|---|---|
| `--model` | L78 | — | 选后端（`hf` / `vllm` / `local-chat-completions` …） |
| `--tasks` | L66 | — | 指定跑哪些任务 |
| `--model_args` | L86 | — | 传给后端的参数（模型路径等） |
| `--apply_chat_template` | L95 | "Apply chat template to prompts (optional template name)" | ★★★ chat 模型必加 |
| `--limit` / `-L` | L104 | "Limit examples per task (integer count or fraction)" | ★★ 冒烟测试 |
| `--num_fewshot` | L123 | — | 改 shot 数 |
| `--log_samples` / `-s` | L179 | "Save all model outputs and documents for post-hoc analysis" | ★★★ 排查必加 |
| `--include_path` | L238 | "Additional directory for external tasks" | ★★ 加载自己写的任务 |
| `--output_path` | L171 | — | 结果落盘位置（`--log_samples` 要配合它） |

```
   ⟹ 🔑 三个参数的重要性远超其他：

   ★★★ --apply_chat_template
      ⚠️ 不加 = 用 base 模型的拼法去问 chat 模型 = 04 章的头号坑。
      ⟹ 分数可能腰斩，而且【看起来像模型不行】。
      ⟹ ★ 对应 §3.2 状态表的第 3 步。

   ★★★ --log_samples
      ⟹ 唯一能看到 resps（原始输出）和 filtered_resps（抽取结果）
         这两列的办法。★★ 不加它，你对分数为什么低【一无所知】。
      ⟹ ★ 对应第 5、6 步。

   ★★  --limit 20
      ⟹ 先跑 20 条确认流程通了，再跑全量。省钱省时间。
      ⚠️ 但 --limit 的结果【不能当分数报】——
         20 题的标准误大到约 11 个百分点（见速查表 J 节）。
```

---

## 8. 自己加一个任务：最小步骤

```yaml
# ./my_tasks/my_task.yaml
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
```

```bash
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

   ⟹ 🔑 还有一个容易被忽略的理由：★★ 它自带【两把尺子并存】这套机制（§4.2）。
      ⚠️ 自己手写的话，大多数人只会写一把，
      ⟹ 于是永远看不到"抽取规则吃掉了多少分"。
```

---

## 9. 这个框架的边界（它做不了什么）

```
   ❌ ★★★ agentic 评测
      ⟹ 它的三个原语里没有"执行代码""开容器""多轮工具调用"。
      ⟹ SWE-bench 那类必须用另一套东西，见 [SWE-bench拆解.md](SWE-bench拆解.md)。

   ❌ ★★ LLM-as-a-Judge（大模型当裁判）的完整工作流
      ⟹ 可以勉强用 generate_until 拼，但没有内建的
         两两对战、位置交换、Elo 汇总。
      ⟹ 这些在 EvalScope / OpenCompass 里是一等公民，
         见 [OpenCompass与EvalScope拆解.md](OpenCompass与EvalScope拆解.md)。

   ⚠️ ★★ 闭源 API 上的判别式评测
      ⟹ 不是框架的问题，是 API 不返回 logprobs（每个 token 的对数概率）。
      ⟹ 🔑 结构性限制，换框架也解决不了。
      ⟹ ★★★ 回到 §2 那句话：后端做不到的事，窄腰以上全都做不到。

   ❌ ★★ 它不负责判断【题目本身合不合理】，也不负责排除【数据污染】
      ⚠️ filters/decontamination.py 只是个工具，
      ⟹ 是否污染仍然要你自己判断，见 [06 章](评测入门/06-数据污染.md)。
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
      ✅ OpenCompass 的 ppl / ll / gen 三个 Inferencer 也是同一批东西，
         见 [OpenCompass与EvalScope拆解.md](OpenCompass与EvalScope拆解.md)。

   ⟹ 🔑🔑 但本篇真正该带走的是 §4.2：
      ★★★ 这个框架给 GSM8K 默认配了【两把尺子】，
         因为它的作者清楚"一把尺子量不准"。
      ⚠️ 而榜单上只报一个数 ——
      ⟹ 那个数是哪把尺子量的，几乎从来没人写。
```

---

> 相关：[评测入门 01 §1.3b 三把尺子算一遍](评测入门/01-为什么评测是个难题.md) ｜ [03 三个原语](评测入门/03-三个原语.md) ｜ [05 答案抽取与打分](评测入门/05-答案抽取与打分.md) ｜ [附-速查表 J 节 噪声下限](评测入门/附-速查表.md)
> 返回 [harness评测框架/](README.md)
