# SWE-bench 拆解

> ★ 版本口径：`swebench` **5.0.2**（源码在 [源码/swebench-5.0.2/](源码/swebench-5.0.2/)，本文所有行号都是对着这一份数的）
> ★★ 定位：**agentic 评测**的事实标准。Claude Code、Codex 这类工具的分数就来自它。
>
> ⚠️ 它和 [lm-eval-harness](lm-eval-harness拆解.md) **机制上完全不同**：
> ★★★ 那边是「拼提示词 → 抽答案 → 比字符串」；这边是「起容器 → 打补丁 → 跑测试」。
>
> 标记：✅ = 源码/论文原文可查证 ｜ ⚠️ = 我的推断 ｜ ★ = 重点 ｜ 🔑 = 关键结论
>
> ⚠️⚠️ **行号会过期**。5.0.2 相对老版本动过结构（比如 `TestSpec` 从 `harness/test_spec.py` 搬到了 `swebench/types.py`，`eval_script` 改成由数据集直接提供）。★★ 对不上时以你手上那份源码为准，并把版本号一起记下来。
>
> ★ 本篇是**源码侧**拆解。概念与方法论侧在 [评测入门 07 章](评测入门/07-agentic编码评测与SWE-bench.md)，两边有意不重复：那边讲「这套评测为什么这么设计」，这边讲「代码到底怎么跑的」。

---

## 1. 它在测什么

```
   ★ 一道 SWE-bench 题 = 一个真实的 GitHub issue（问题单）+ 一个真实的代码仓库快照

   给模型：  ① 仓库在 bug 未修复时的完整代码
             ② issue 的文字描述（用户报告的问题）
   要模型：  产出一个 patch（补丁，即 git diff 格式的代码改动）
   怎么判：  ★★★ 把补丁打到仓库上，【真的跑单元测试】

   ⟹ 🔑 和选择题的根本区别：
      ★★★ 它不比较文本相似度，它【执行】。
      ⚠️ 模型写的代码和人类的官方修复长得完全不一样也没关系，
         ★★ 测试过了就是过了。
```

✅ 论文摘要原文（arXiv:2310.06770，52 页，已校验）：

> "SWE-bench, an evaluation framework consisting of **2,294** software engineering
> problems drawn from real GitHub issues and corresponding pull requests across
> **12 popular Python repositories**. … **Claude 2** … resolves a mere **1.96%**
> of the issues."

### ✅ 一道题在数据集里长什么样：12 个字段

★★ 这是全篇的起点 —— 后面每一步都只是在读写这 12 个字段里的某几个。✅ 源码 `swebench/types.py:9-21` 原文：

```python
class SWEbenchInstance(TypedDict):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str
    version: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    environment_setup_commit: str
```

★ 逐个字段说清它是干什么的，以及**谁能看到它**：

| 字段 | 中文 | 是什么 | ★★★ 模型/agent 能看吗 |
|---|---|---|---|
| `repo` | 仓库名 | 比如 `django/django` | ✅ 能 |
| `instance_id` | 题号 | 比如 `django__django-11099` | ✅ 能 |
| `base_commit` | 基准提交 | ★★★ bug **还没修**时的那个 commit | ✅ 能（必须从这里开始干活） |
| `patch` | 官方修复 | 人类当年真实合入的那个 diff | ❌ **绝对不能**，这就是标准答案 |
| `test_patch` | 测试补丁 | 那个 PR 里新增/改动的测试文件 | ❌ **绝对不能**，见 §9 的坑② |
| `problem_statement` | 问题陈述 | issue 的文字 | ✅ 能，这是题干 |
| `hints_text` | 提示文本 | issue 下面的讨论跟帖 | ⚠️ 看你的脚手架给不给，**给不给会影响分数**，属于必须披露的口径 |
| `created_at` | 创建时间 | PR 的日期 | ✅ 能（查污染要用） |
| `version` | 版本 | 该仓库的版本标签，决定装哪套依赖 | ⚠️ 一般不给 |
| `FAIL_TO_PASS` | 尺子① | 打补丁前失败、打完应该通过的测试名单 | ❌ **绝对不能** |
| `PASS_TO_PASS` | 尺子② | 打补丁前就通过、打完必须还通过的测试名单 | ❌ **绝对不能** |
| `environment_setup_commit` | 环境提交 | 用来装依赖的那个 commit（可能和 `base_commit` 不同） | ⚠️ 一般不给 |

```
   ⟹ 🔑🔑 把这张表的最后一列单独看一遍，你就明白了 SWE-bench 的
      ★★★【信息隔离】设计：

   ★★ 能给模型的只有 4 个字段（repo / instance_id / base_commit / problem_statement）
      + 一个有争议的 hints_text。
   ⚠️ 剩下的 7 个字段全是【判卷用的】，一旦漏给模型，分数立刻失去意义。

   ⟹ ★★★ 而这个隔离，harness 代码【管不了】——
      它只在你那一侧（脚手架）生效。见 §4。
```

---

## 2. ★★★ 判卷：两把尺子，不是一把

★★ 这是整个 SWE-bench 设计里最值得学的一点。

| | 尺子① `FAIL_TO_PASS` | 尺子② `PASS_TO_PASS` |
|---|---|---|
| 中文 | 原本失败 → 应该变通过 | 原本通过 → 必须还通过 |
| ✅ 常量定义 | `harness/constants/__init__.py:30` | `harness/constants/__init__.py:32` |
| ★★ 它在问什么 | 这个 bug **修好了吗**？ | ★★★ 你有没有**把别的功能搞坏**？ |
| 打补丁**前**的状态 | 全部失败 | 全部通过 |
| 打补丁**后**要求 | 应该全过 | **必须**还是全过 |
| ✅ 判卷时允许打折吗 | ★★ 允许（有 PARTIAL 一档） | ❌ **一个都不许掉** |

```
   ⟹ 🔑🔑 为什么必须有第二把尺子？
      ⚠️ 因为「让这个测试通过」有一个作弊解法：★★★ 把断言删掉。
      ⟹ 只有尺子① 的话，删测试、改配置、注释掉校验都能得分。
      ⟹ ★★ 尺子② 堵死了这条路 —— 你一动别的东西，它立刻报警。
```

### ✅ 判卷逻辑的完整源码（`harness/grading.py:309-326`）

```python
def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
    Determine resolved status of an evaluation instance

    Criteria:
        - If fail-to-pass (Resolution) = 1 and pass-to-pass (Maintenance) = 1 -> FULL
        - If (fail-to-pass (Resolution) < 1 and > 0) and pass-to-pass (Maintenance) = 1 -> PARTIAL
        - Otherwise -> NO
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)

    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    elif f2p < 1 and f2p > 0 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    else:
        return ResolvedStatus.NO.value
```

✅ 两个比率的算法（`grading.py:288-295` 和 `:298-306`，两个函数结构完全对称）：

```python
def compute_fail_to_pass(report):
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total
```

⚠️ ★★ 注意 `if total == 0: return 1` 这一行：**名单为空时直接返回满分**。✅ `compute_pass_to_pass` 里官方自己在这一行上留了 `# TODO: Don't factor in p2p metrics`，说明他们知道这是个将就的处理。

✅ 三种结果的枚举（`constants/__init__.py:6-9`）：

```python
class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"
```

★ 把三个分支摊成一张真值表，看清那个不对称：

| `f2p`（尺子①） | `p2p`（尺子②） | ⟹ 判定 | 人话 |
|---|---|---|---|
| `== 1` 全过 | `== 1` 全过 | ✅ `RESOLVED_FULL` | 修好了，也没弄坏 |
| `0 < f2p < 1` 部分过 | `== 1` 全过 | `RESOLVED_PARTIAL` | ★★ 修了一半，没弄坏 |
| `== 1` 全过 | ⚠️ `< 1` 掉了哪怕一个 | ❌ `RESOLVED_NO` | ★★★ **修好了也判 NO** |
| `== 0` 全不过 | 任意 | ❌ `RESOLVED_NO` | 没修对 |

```
   ⟹ 🔑🔑🔑 请注意上面这张表里【最狠的一行】：第三行。

   ★★★ 三个分支里，p2p 的条件全部是 `p2p == 1`，一个都不放松。

   ⟹ 换句话说：⚠️【只要弄坏了一个原本通过的测试，
      不管 bug 修得多漂亮，直接判 NO。】
   ⟹ ★★ 而 f2p 是允许部分的（PARTIAL 那一档）。

   ★★★ 这个不对称设计翻译成人话就是：
      「修不完可以商量，弄坏了没得商量。」
   ⚠️ 这恰好也是真实软件工程的规矩 ——
      ★★ 回归（regression，指改动把原本好用的功能弄坏了）
      比未完成更不可接受。

   ⚠️ 补充一个读榜单时的注意点：
   ★★ 榜单上报的 "% Resolved" 通常只算 FULL，PARTIAL 不计分。
```

---

## 3. ★★★ 一次评测从入口到出分：每一步谁改了什么状态

★★ 这一节是全篇的主线。⚠️ 源码拆解最容易变成「模块罗列」，所以这里换个问法：**跑一次 `run_evaluation`，一条数据依次经过哪些对象和函数，每一步之后世界上多了什么东西？**

> ★ 同一条流程的**概念版**（谁做、最容易出分歧在哪）在 [评测入门 07 章](评测入门/07-agentic编码评测与SWE-bench.md) §7.2b，那张表讲的是「方法论上的分歧点」。这一节讲的是「代码上的状态变化」，两边配着看。

### 3.1 ✅ 函数级调用链

```
   命令行入口   python -m swebench.harness.run_evaluation
        │
        ▼
   main()                                        run_evaluation.py:699
        │   ★★★ 整个签名里只有 predictions_path 一个「模型侧」输入
        ▼
   get_predictions_from_file()                   harness/utils.py:37
        │   读 JSONL  ⟹  dict{instance_id: pred}       （调用点 :740）
        ▼
   write_run_metadata()                          调用点 run_evaluation.py:744
        │   落盘 run.json ——「这次跑的是哪个数据集、哪个 split」
        ▼
   get_dataset_from_preds()                      run_evaluation.py:542
        │   ★★ 按 instance_id 把【预测】和【题目】对齐
        │   ⚠️ 已经有 report.json 的题会被跳过（可续跑）
        ▼
   make_test_spec()                              harness/utils.py:251
        │   dict  ⟹  TestSpec 对象（定义在 swebench/types.py:25）
        ▼
   run_instances()                               run_evaluation.py:432
        │   线程池并发，--max_workers 默认 4
        ▼
   run_instance()  ← ★★★ 一道题的全部动作都在这个函数里
        │                                        run_evaluation.py:229
        ├─① create_container()                   调用点 :284（定义 :72）
        ├─② patch_file.write_text(...)           :291
        ├─③ copy_to_container()                  :295  （docker_utils.py:16）
        ├─④ for git_apply_cmd in GIT_APPLY_CMDS  :301
        ├─⑤ git apply --check --reverse 兜底      :323
        ├─⑥ exec_run_with_timeout("/bin/bash /eval.sh")
        │                                        :367（docker_utils.py:109）
        └─⑦ get_eval_report()                    :399（grading.py:329）
                 ├─ get_logs_eval()              grading.py:113
                 ├─ get_eval_tests_report()      grading.py:179
                 └─ get_resolution_status()      grading.py:309
        │
        ▼
   make_run_report()                             reporting.py:16（调用点 :794）
        │   汇总全部题目  ⟹  一个总报告 json
        ▼
   最终那个百分数
```

### 3.2 ★★★ 状态怎么变：逐步表

★★ 同一条流程，按「这一步之后磁盘/容器里多了什么」摊开。⚠️ 这张表是本篇最该记住的东西 —— **排查任何一次失败，都是回到这张表里找「卡在哪一行」**。

| 步 | 代码位置 | 读了什么 | ★★ 写出了什么状态（在哪） | ⚠️ 这一步失败时留下的痕迹 |
|---|---|---|---|---|
| 0 | `main()` :699 | 命令行参数 | —— | ★ 连 `run_id` 都没有就 `assert` 挂掉（:733） |
| 1 | `get_predictions_from_file()` utils.py:37 | 你的 `preds.jsonl` | 内存里的 `{instance_id: pred}` | ⚠️ JSONL 格式错 / `instance_id` 拼错 ⟹ 这题**根本不会被跑**，榜单上算 0 分但日志里什么都没有 |
| 2 | `write_run_metadata()` 调用点 :744 | 数据集名、split | 磁盘：`run.json` | ★ 这个文件是后面 `swebench report` 重新判卷的依据 |
| 3 | `get_dataset_from_preds()` :542 | 预测 + 数据集 | 内存里的待跑题目列表 | ★★ 已有 `report.json` 的题被静默跳过 ⟹ **改了脚手架重跑却没换 `--run_id`，拿到的是上次的旧分数** |
| 4 | `make_test_spec()` utils.py:251 | 数据集那一行 | `TestSpec` 对象：`eval_script_list` / `FAIL_TO_PASS` / `PASS_TO_PASS` / `log_parser` | ⚠️ 5.0.2 里 `eval_script` 是**数据集直接给的**，harness 只负责解析（见 §3.3） |
| 5 | `create_container()` :72 / 调用点 :284 | `TestSpec.image` | ★★★ 一个跑起来的容器，里面是该仓库在 `base_commit` 上、依赖装好的状态 | ★★ 镜像拉不下来 / 装不上 ⟹ 进 `except Exception`（:418），这题**报错，不是「答错」** |
| 6 | `patch_file.write_text()` :291 | `pred["model_patch"]` | 磁盘：`logs/run_evaluation/<run_id>/<model_name>/<id>/patch.diff` | ★ `model_patch` 是空串也照写（`or ""`），⟹ 空补丁会一路走到判卷，稳定判 NO |
| 7 | `GIT_APPLY_CMDS` 循环 :301 | `patch.diff` | 容器里的工作树被改动 | ★★★ 四级全败 ⟹ 日志写入 `APPLY_PATCH_FAIL`，抛 `EvaluationError`（:333）。**这是「格式问题」而不是「能力问题」，见 §5** |
| 7b | `git apply --check --reverse` :323 | `patch.diff` | —— | ★★ 5.0.2 新增的兜底：四条命令全报非零、但补丁其实已经打上去了，这里能救回来 |
| 8 | `git diff`（前） :342 | 容器工作树 | 日志里的 `Git diff before` | ★ 用来和跑完测试之后对比，看测试本身有没有改动代码 |
| 9 | `exec_run_with_timeout()` :367 | `/eval.sh` | 磁盘：`test_output.txt`（**原始测试输出**） | ★★★ 超时 ⟹ `TESTS_TIMEOUT`，抛 `EvaluationError`（:377）。默认上限 `--timeout` = 1800 秒 |
| 10 | `get_logs_eval()` grading.py:113 | `test_output.txt` | 内存里的 `status_map`：每个测试名 → PASSED / FAILED / SKIPPED / ERROR / XFAIL | ★★ 见到 `APPLY_PATCH_FAIL` / `RESET_FAILED` / `TESTS_ERROR` / `TESTS_TIMEOUT` 任一标记就直接返回 `({}, False)`（:141） |
| 11 | `get_eval_tests_report()` grading.py:179 | `status_map` + 两份名单 | 四个桶：F2P / P2P 各自的 `success` 和 `failure` | ★ 名单里有、但日志里查不到的测试，怎么算取决于 `eval_type`（见 §3.4） |
| 12 | `get_resolution_status()` grading.py:309 | 四个桶 | 一个字符串：`RESOLVED_FULL` / `PARTIAL` / `NO` | —— |
| 13 | 写 `report.json` :411 | 上面全部 | 磁盘：这题的 `report.json` | ★★ 有了这个文件，这题下次就会被跳过（回到第 3 步） |
| 14 | `make_run_report()` reporting.py:16 | 所有题的 `report.json` | 磁盘：总报告 | ⟹ 这才是你看到的那个百分数 |

```
   ⟹ 🔑🔑 读完这张表，三件事应该变得很清楚：

   ★★★ ① 分数是【第 12 步】算出来的，但能毁掉分数的地方
      从【第 1 步】就开始了。
      ⚠️ 第 1、3、6、7、9 步任意一步出问题，
         最终都表现为「这题 0 分」，而原因完全不同。

   ★★★ ② 唯一可靠的排查入口是磁盘上那三个文件：
      patch.diff（你给了什么）
      test_output.txt（测试真实输出）
      report.json（判卷结论）
      ⟹ 🔑 不看这三个文件，对着总分猜原因是浪费时间。

   ★★★ ③ 第 5–14 步全是官方代码，谁跑都一样。
      ⚠️ 而第 1 步那个 JSONL 【是你自己造的】——
      ⟹ 所有的不可比性都藏在那个文件是怎么产生的里面。这就是 §4。
```

### 3.3 ⚠️ 一个必须讲的版本差异：`eval_script` 从哪来

★★ 这一条是读老资料时最容易被坑的地方。

✅ 5.0.2 的 `make_test_spec()`（`harness/utils.py:251-272`）原文片段：

```python
def make_test_spec(instance: dict) -> TestSpec:
    """
    Build a TestSpec from a dataset instance.

    The instance dict must contain: instance_id, image, repo, version,
    FAIL_TO_PASS, PASS_TO_PASS, log_parser, eval_type, eval_script.
    """
```

```
   ⟹ 🔑 注意 docstring 里要求的字段：`image`、`eval_script`、`log_parser`。
      ★★★ 这三个【不在】§1 那张 12 字段表里。

   ⚠️ 含义：5.0.2 里，「用哪个镜像、跑什么测试命令、用哪个日志解析器」
      是【数据集直接提供的】，harness 只做解析和加工。

   ⟹ ★★ 而老版本（2.x 时代）是 harness 自己按
      base_commit + test_patch + 仓库版本【现场拼出】这段脚本的。

   ⟹ 🔑🔑 所以：★★★ 如果你读到的教程里讲「harness 会 git checkout
      base_commit、再 apply test_patch」，⚠️ 那是老版本的行为。
      5.0.2 里这些动作已经被固化进镜像和 eval_script 了。
      ⟹ 这也是为什么镜像那么大（§6）。
```

✅ 加工的那一步也值得看一眼 —— `record_test_exit_code()`（`harness/utils.py:222-241`）的 docstring 原文：

> "Eval scripts end with a `git checkout` that resets the test files, and run
> under `set -uxo pipefail` without `-e`, so the script's exit status is the
> reset's, not the tests'."

```
   ⚠️ 翻译：★★ 脚本最后会 git checkout 把测试文件还原回去，
      而且没开 `set -e`，⟹ 整个脚本的退出码是【那个还原动作的】，
      不是【测试的】。
   ⟹ 🔑 所以 harness 在测试命令后面插了一行 `$?` 把真实退出码捞出来。
      ★★★ 这是一个很典型的「脚本退出码不等于你以为的那个退出码」陷阱。
```

### 3.4 ⚠️ 两个容易漏掉的判卷开关

✅ `constants/__init__.py:20-22`：

```python
class EvalType(Enum):
    PASS_AND_FAIL = "pass_and_fail"
    FAIL_ONLY = "fail_only"
```

```
   ★ PASS_AND_FAIL = 两把尺子都查（标准模式）。
   ★★ FAIL_ONLY   = 只查 FAIL_TO_PASS。
      ⚠️ 这一档下「名单里有、日志里没有」的测试会被当成【通过】。

   ⟹ 🔑🔑 5.0.2 在 get_logs_eval() 里专门为此加了一道防线
      （✅ grading.py:155-160 的注释原文）：
      "Under EvalType.FAIL_ONLY an absent test counts as success, so
       without this a suite that never started (e.g. a browser that fails
       to launch) scores every F2P test as resolved."

   ⟹ ⚠️ 翻译：★★★ 测试套件【根本没启动】的情况下，
      如果不加这道防线，每一个 F2P 测试都会被判成「过了」——
      ⟹ 一道题凭「什么都没跑」拿到满分。
   ⟹ 🔑 这类 bug 才是评测框架里最危险的：★★ 它不报错，它给高分。
```

✅ 另一道同类防线在 `grading.py:166-175`：测试日志里一个 FAILED 都没有、但测试命令的退出码非零 ⟹ 整题作废（`return {}, False`）。源码注释写明了动机：**补丁自己可以打印假的 "PASSED" 行**（比如塞一个 `conftest.py` 钩子）。

---

## 4. ★★★ 一个源码事实：harness 根本不知道 agent 长什么样

★★ 这一节直接回答「Claude Code 是怎么被评的」。

✅ `harness/run_evaluation.py:699-712` 的 `main()` 签名，输入里**只有一个文件路径**是「模型侧」的：

```python
def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,          # ★★★ 就是它
    max_workers: int,
    open_file_limit: int,
    run_id: str,
    timeout: int,
    rewrite_reports: bool,
    modal: bool,
    report_dir: str = ".",
    task_repo: str | None = None,
):
```

✅ 每条预测的数据结构（`run_evaluation.py:244` 的 docstring 原文）：

```
    # pred (dict): Prediction w/ model_name_or_path, model_patch, instance_id
```

✅ 而 `model_name_or_path` 只被用来当日志目录名（`run_evaluation.py:253-254`）：

```python
    model_name_or_path = pred.get("model_name_or_path", "None").replace("/", "__")
    log_dir = RUN_EVALUATION_LOG_DIR / run_id / model_name_or_path / instance_id
```

```
   ⟹ 🔑🔑🔑 把这三条拼起来，结论无法回避：

   ★★★ SWE-bench 的评测程序收到的，是一个 JSONL 文件，
        每行三个字段：instance_id、model_patch、一个【纯粹当文件夹名用的字符串】。

   ⚠️ 它【完全不知道】：
      ❌ 这个补丁是哪个模型生成的（那个字段是自由文本，你写什么都行）
      ❌ 用了什么脚手架、什么工具、什么提示词
      ❌ 跑了一次还是五十次
      ❌ 花了 $0.13 还是 $2.59
      ❌ ★★★ 生成补丁的时候，有没有偷看 FAIL_TO_PASS 的测试内容

   ⟹ 🔑🔑 所以：★★★【SWE-bench 分数评的是「模型 + 脚手架」这一整套系统，
      而榜单通常只写模型的名字。】
```

### ✅ 这个结论有正反两面的实证

**正面：同一个权重，换脚手架差 6.7 倍**（✅ SWE-agent 论文 arXiv:2405.15793 Table 1 原文数值，GPT-4 Turbo × **SWE-bench Lite 300 题**）

| 设定 | 这个设定是什么 | % Resolved | ✅ $ Avg. Cost |
|---|---|---|---|
| RAG | 检索增强，非交互：先替模型挑好文件，一次性生成 | **2.67** | $0.13 |
| Shell-only，不给示范 | 裸 shell 交互，没有示范轨迹 | 7.33 | $0.79 |
| Shell-only + 示范 | 裸 shell 交互 + 一条示范轨迹 | 11.00 | $1.46 |
| ★★★ SWE-agent 完整 ACI | 专门为模型设计的编辑/搜索/翻页命令 | **18.00** | $1.67 |

✅ 论文自己的措辞（★ 逐字）："compared to RAG on Lite, SWE-agent is 8-13x more costly but yields a **6.7-fold improved % Resolved rate**."

⚠️⚠️ **口径两条必须跟着一起写**：① 这是 **2024 年**的 GPT-4 Turbo，证明的是「差异存在且量级很大」，**不是**「今天换个脚手架还能差 6.7 倍」；② ✅ 论文把 `$ Avg. Cost` 定义成 **"averaged over all successfully resolved instances"** —— 即**只对解决了的题取平均**，⟹ ★★ 它**不是**「每道题平均花这么多」，用它估全量预算会严重偏低。

**反面：一线团队自己也知道必须披露脚手架**

✅ DeepSeek-V3 报告原文：*"SWE-Bench verified is evaluated using the **agentless framework**"* ⟹ ★★ 连报告作者都明白，不写这一句，那个分数就没法被解读。

⟹ 这条线完整展开在 [评测入门 08 章](评测入门/08-ClaudeCode与Codex怎么被评.md) §8.4.1，以及 [agent 脚手架目录](agent脚手架/README.md)（那一整个目录就是在讲 harness 不问的这一段）。

---

## 5. ★★ 一个特别有教育意义的细节：打补丁本身会失败

✅ `run_evaluation.py:54-59` 的完整原文：

```python
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]
```

⚠️ 逐条解释这四级「容错阶梯」（越往下越宽松）：

| 档 | 命令 | 宽松到什么程度 |
|---|---|---|
| ① | `git apply --verbose` | ★ 严格模式：上下文必须一字不差地对上 |
| ② | `git apply --verbose --3way` | ★★ 三方合并：允许用 git 的历史信息去推断该打在哪 |
| ③ | `git apply --verbose --reject` | ★★ 能打的部分先打上，打不上的丢到 `.rej` 文件里 |
| ④ | `patch --batch --forward --fuzz=5 -p1 -i` | ★★★ 换成古老的 `patch` 工具，`fuzz=5` 表示**上下文允许有 5 行对不上**也照打 |

✅ `run_evaluation.py:301` 是一个循环：前一个失败就试下一个。✅ 并且 5.0.2 在 `:302-309` 加了一步**清场**——注释原文：

> "a failed attempt (notably --reject) leaves partial state behind, which makes
> every later command fail; restart from a pristine tree"

```
   ⚠️ 翻译：★★ 第 ③ 档失败时会在工作树里留下打了一半的烂摊子，
      ⟹ 不清掉的话，第 ④ 档必然也失败。
   ⟹ 🔑 这是一个很实在的工程细节：★★★【重试之前必须先回到干净状态】，
      否则「四级容错」在真实情况下退化成「一级」。
```

✅ 全部失败时的标记（`constants/__init__.py:44`）和超时标记（`:50`）：

```python
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
TESTS_TIMEOUT    = ">>>>> Tests Timed Out"
```

★★★ 这个设计和 [评测入门 05 章](评测入门/05-答案抽取与打分.md) 讲的「抽取失败」是**同构**的：

| | lm-eval-harness | SWE-bench | EvalScope 的 Arena Hard |
|---|---|---|---|
| 场景 | 模型答对了但格式不合规 | 补丁思路对了但 diff 格式不合规 | 裁判给了判决但格式不合规 |
| 代码里发生什么 | 抽取结果被填成 `"[invalid]"` | 日志写入 `APPLY_PATCH_FAIL` | 分数按默认值 `0.5` 算 |
| ⟹ 最终分数 | ★★ 判错 | ★★ 判错 | ⚠️ 判成平局 |
| ★★★ 日志里分得清吗 | ✅ 分得清（`--log_samples` 能看到） | ✅ **分得清**（有专门标记） | ❌ **分不清**，连标记都没留 |

```
   ⟹ ★★★ 三个框架都在做同一件事：
      【把"格式问题"和"能力问题"在最终分数上混成一个数】。

   ⚠️ 但 SWE-bench 做得比另两个好的一点：
      ★★ 它给了 APPLY_PATCH_FAIL 一个【专门的标记】，
      ⟹ 所以你翻日志能分清"没修对"和"补丁根本没打上去"。
      🔑 这正是 05 章希望 lm_eval 也能做到的事。

   ⟹ 🔑 排查建议：★★★ 自己跑 SWE-bench 时，
      先统计日志里 APPLY_PATCH_FAIL 的比例。
      ⚠️ 比例高 ⟹ 你的脚手架输出 diff 的格式有问题，
         【不是模型不会修 bug】。
```

✅ 一个参照量级：SWE-bench 论文 Table 5 里，Claude 2 在 BM25 设定下的 **% Apply 只有 43.07%**（全量）/ **33.00%**（Lite）—— ⚠️ 也就是说**一半以上的补丁根本没打进去**。★★ 这个指标和 % Resolved 一样重要，但榜单上几乎从不出现。

---

## 6. 为什么这种评测这么贵

✅ 源码里的成本证据（`run_evaluation.py:61-62`）：

```python
DOCKER_CLIENT_TIMEOUT   = int(os.environ.get("SWEBENCH_DOCKER_TIMEOUT", "1800"))
DOCKER_CLIENT_POOL_SIZE = int(os.environ.get("SWEBENCH_DOCKER_POOL_SIZE", "128"))
```

```
   ⟹ ⚠️ 从这两个默认值能读出很多东西：

   ★★ 单个容器操作的超时是 1800 秒 = 30 分钟。
      ⟹ 说明【一道题跑几十分钟是正常的】。
      ✅ 命令行的 --timeout 默认值也是 1800（run_evaluation.py:843）。
   ★★ 连接池默认 128。
      ⟹ 说明设计上就预期【上百个容器并发】。
      ⚠️ 但 --max_workers 默认只有 4（:833），help 里写着
         "should be <= 75% of CPU cores" ⟹ 实际并发受你的 CPU 限制。

   ⟹ 🔑 成本三件套：
      ① ★★★ 镜像：12 个仓库 × 多个历史版本，拉下来【几十 GB】
         ⚠️ 5.0.2 里连 eval_script 都固化进镜像了（§3.3），所以更大
      ② ★★  时间：2,294 道题 × 每题分钟级
      ③ ★★★ 模型调用：agentic 是多轮的，一道题几十轮对话
         ✅ SWE-agent 论文把单题预算上限设成了 $4
         ⚠️ 超过就自动把当前改动交卷

   ⟹ 🔑🔑 对照 [02 章](评测入门/02-三种评测范式.md) 那条定律：
      ★★★【越真实 ⟺ 越贵 ⟺ 越难复现】。
      ⚠️ MMLU 一道题的成本是它的几万分之一。
```

---

## 7. ⚠️⚠️ 题集有好几个，绝对不能混着比

★★★ 这一节是读 SWE-bench 分数时**第一个该问的问题**。

| 名字 | 题数 | ✅ 出处 | 它是怎么来的 | ⚠️ 读分数时要注意 |
|---|---|---|---|---|
| **SWE-bench**（全量） | **2,294** | ✅ 论文 §2.1，从 ~90,000 个 PR 过滤而来 | 12 个热门 Python 仓库的真实 issue + PR | ★ 最难，也最贵 |
| **SWE-bench Lite** | **300** | ✅ **就在原论文里**：论文 §2.4 专节 + 论文 Appendix A.7 完整筛选条件 | 从全量里抽样，挑**更自包含**的、聚焦**功能性 bug 修复**的题 | ★★★ ✅ 论文原文：**"covers 11 of the original 12 repositories"** ⟹ **只覆盖 12 个仓库里的 11 个**，少一个 |
| **SWE-bench Verified** | **500** | ⚠️ **不在原论文里**。由 **OpenAI 于 2024 年 8 月**另行发布 | 请专业开发者人工逐题核验「题目本身是否可解、测试是否公允」后留下的子集 | ★★★ 分数**系统性高于**全量和 Lite |
| **SWE-bench Multimodal** | ⚠️ 未在本篇核实 | ✅ 源码里有专门分支（`run_evaluation.py:716`） | issue 里带图片的题 | ⚠️ 官方建议用托管评测（`sb-cli`）提交 |
| **SWE-bench Multilingual** | ⚠️ 未在本篇核实 | ✅ 源码别名表里有（`cli/_datasets.py:8`） | 非 Python 语言 | ⚠️ 和上面几个完全不可比 |

✅ Lite 的出处原文（论文 §2.4，★ 逐字）：

> "To encourage adoption of SWE-bench, we create a **Lite** subset of **300 instances**
> from SWE-bench that have been sampled to be more self-contained, with a focus on
> evaluating functional bug fixes. … **SWE-bench Lite covers 11 of the original 12
> repositories**, with a similar diversity and distribution of task instances across
> repositories as the original."

```
   ⟹ 🔑🔑 两个最容易搞错的点，单独拎出来：

   ★★★ ① 「Lite 是后来社区搞的」是【错的】。
      ⚠️ Lite 写在原论文 §2.4 里，筛选细节在 Appendix A.7。
      ⟹ 它是官方的、第一天就有的。

   ★★★ ② 「Verified 是 SWE-bench 作者做的」也是【错的】。
      ⚠️ Verified 是 OpenAI 2024-08 另行发布的人工核验子集，
         ⟹ 原论文（ICLR 2024）里【完全没有这个东西】。
      ⟹ 🔑 所以引用 Verified 的分数时，别把它的来源记成那篇论文。

   ⟹ ★★★ Verified 上的分数【系统性高于】全量和 Lite，
      因为它主动剔除了描述不清、信息不足、测试不公允、根本做不了的题。
      ⚠️ 这不是「模型变强了」，是【题变可做了】。

   ⟹ 🔑 看到 "SWE-bench 42%" 这样一个数，
      ★★★ 必须先问是哪个子集。⚠️ 差别可以是十几个点。
```

⚠️⚠️ **还有一个比子集更隐蔽的问题：题数直接决定了噪声下限。**

| 子集 | 题数 n | p=0.5 时的标准误 SE | ⟹ 差多少才值得当真 |
|---|---|---|---|
| Lite | 300 | 2.9 个百分点 | ★★★ **差 5 个点约等于噪声** |
| Verified | 500 | 2.2 个百分点 | 差 4 个点以下要慎重 |
| 全量 | 2,294 | 1.0 个百分点 | 差 2 个点以下要慎重 |

★ 算法是 `SE ≈ √(p(1−p)/n)`，完整的一张表和四条使用限制在 [评测入门 · 附-速查表](评测入门/附-速查表.md) J 节。⟹ 🔑 **这意味着 Lite 上那些「差 2、3 个点」的榜单名次，基本读不出信息。**

---

## 8. ⚠️ 一个必须警惕的坑：上下文是怎么给的

★★ 模型不可能把整个仓库塞进上下文 —— ✅ 论文 Table 1：一个仓库平均 **3,010 个非测试文件 / 438K 行非测试代码**，而官方修复平均只改 **32.8 行**。

```
   ⟹ 🔑 先把这个比例感受一遍：
      438,000 行里只改 32.8 行  ⟹  ★★★ 99.99% 是噪音。
   ⟹ ⚠️ 所以「挑出可能相关的文件」这一步，本身就是半个题目。
```

✅ 论文 §4.1 原文（关于 BM25 检索基线）：

> "We observe that in approximately **40% of instances**, BM25 retrieves a superset
> of the oracle files for the 27,000-token context limit. However, in **almost half
> of the instances** with the 27,000-token limit, it retrieves **none of the files**
> from the 'oracle' context."

```
   ★ 两个名词先说清：
   ★ BM25（Best Matching 25）= 一种经典的关键词检索算法，不用神经网络。
   ★★ oracle context（先知上下文）= 官方修复实际改动的那些文件，
      ⟹ 相当于"标准答案涉及哪几个文件"。

   ⟹ 🔑🔑 论文这句话的意思是：
      ★★★ 用 BM25 检索时，接近一半的题目里，
      ⚠️【真正该改的文件一个都没被检索出来】。
      ★ （✅ 论文这句限定在 27,000 token 的上下文预算下）
      ⟹ 这种情况下模型再强也做不对 —— 它压根没看到该改的代码。

   ⟹ 🔑 所以：★★★ "怎么给上下文"本身就是脚手架的一部分，
      而且是【影响最大的那一部分】。
      ⚠️ 这也解释了 §4 里 RAG 2.67% vs SWE-agent 18.00% 的巨大差距 ——
         SWE-agent 让模型【自己去搜、自己去翻】，
         而不是提前替它决定该看什么。
```

⟹ ★★ 回到 §3.2 那张状态表：**这一整节讲的事情，发生在第 1 步之前** —— 也就是 harness 完全看不见的那一段。

---

## 9. 如果你想自己跑一次

### 9.1 ✅ 5.0.2 有了一个新的命令行入口

✅ `pyproject.toml:84-85` 注册了 `swebench` 这个命令（`swebench.cli.cli:main`）。✅ `cli/evaluate.py:49-57` 的 help 里自带示例：

```bash
# ① 装
pip install swebench

# ② ★★★ 先用官方修复跑一遍，确认你的环境是好的
#    --gold 表示"评测参考补丁"，✅ cli/evaluate.py:18
swebench eval verified --gold -i <某个instance_id>

# ③ 再跑自己的预测
swebench eval verified -p preds.jsonl --run-id my-first-run -j 16
```

```
   ⟹ 🔑🔑 ★★★ 第 ② 步是最值得强调的一条建议：
      先跑 --gold。
   ⚠️ 如果官方修复在你这儿都判不出 RESOLVED_FULL，
      ⟹ 那是你的环境/镜像/Docker 有问题，
         【不是你的 agent 不行】。
   ⟹ ★★ 这一步能省掉后面几小时的瞎猜。
```

⚠️ **别名表里没有 `lite`**（✅ `cli/_datasets.py:5-10` 只有 `full` / `verified` / `multilingual` / `multimodal`），⟹ 跑 Lite 要写全名。★ 另外注意数据集的 HuggingFace 组织名**改过**：`run_evaluation.py:807` 现在的默认值是 `SWE-bench/SWE-bench_Lite`，老资料里的 `princeton-nlp/...` 是旧名。

### 9.2 老的模块式入口仍然可用

```bash
# 准备预测文件 preds.jsonl，每行三个字段：
# {"instance_id": "...", "model_patch": "<diff 文本>",
#  "model_name_or_path": "my-agent-v1"}

python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Lite \
    --predictions_path ./preds.jsonl \
    --max_workers 4 \
    --run_id my-first-run \
    --instance_ids <id1> <id2>
```

### 9.3 ⚠️ 四个新手一定会踩的坑

| # | 坑 | ⚠️ 后果 | ★ 对应 §3.2 状态表的哪一行 |
|---|---|---|---|
| ① | ★★★ 没让 agent 在 `base_commit` 上工作 | 直接 clone 主分支 = 拿到的是**已修复的代码** = 分数假到离谱 | 发生在第 1 步之前 |
| ② | ★★★ 让 agent 看到了 `FAIL_TO_PASS` 的测试内容或 `test_patch` | 等于把答案给它。🔑 **更隐蔽的版本**：跑 N 次然后用 f2p 测试挑最好的那次 —— ★★★ 这也是看答案，只是换了个姿势 | 同上，harness 查不出来 |
| ③ | ★★ 重跑却没换 `--run_id` | 已有 `report.json` 的题被静默跳过，你拿到的是**上次的旧分数** | 第 3 步 |
| ④ | ★★ Docker 镜像会拉几十 GB | 磁盘/网络先确认；⚠️ `--max_workers` 开太大会把机器打满 | 第 5 步 |

```
   ⟹ 🔑 坑 ② 值得再说一遍，因为它是【唯一一个 harness 无法防御的作弊】：

   ★★★ 第 §3.2 表的第 5–14 步全是官方代码，谁跑都一样。
   ⚠️ 但「生成 preds.jsonl」这件事完全在你手里，
      ⟹ 你偷看了答案，harness 一个字段都察觉不到。
   ⟹ ★★ 所以 agentic 榜单的可信度，
      最终取决于【报分的人愿不愿意如实披露口径】，
      而不是取决于评测框架写得多严。
```

---

## 10. 一句话总结

```
   ★★★ SWE-bench 的核心贡献不是"出了 2,294 道题"，
   🔑🔑 而是把评测的判定权，从【文本比对】交给了【程序执行】。

   ⟹ ★★ 代价是：贵、慢、难复现。
   ⟹ ★★★ 收益是：分数造假的空间被压缩到了极小
      —— 你没法"说服"一个单元测试。

   ⚠️ 但它换来了一个新问题：
      ★★★ 由于 harness 只收补丁不问来源（§4），
      【分数从此变成了「模型 + 脚手架 + 子集 + 预算」的联合产物】，
      ⟹ 而这一点，榜单上是看不出来的。

   ⟹ 🔑🔑 所以本篇真正该带走的是 §3.2 那张状态表：
      ★★★ 它告诉你分数是在哪一行算出来的，
         也告诉你能毁掉分数的那几行【根本不在官方代码里】。
```

---

> 相关：[评测入门 07 agentic 编码评测](评测入门/07-agentic编码评测与SWE-bench.md)（概念侧，与本篇互补） ｜ [08 Claude Code 和 Codex 怎么被评](评测入门/08-ClaudeCode与Codex怎么被评.md) ｜ [agent 脚手架目录](agent脚手架/README.md)（harness 不问的那一段） ｜ [lm-eval-harness 拆解](lm-eval-harness拆解.md)
> 返回 [harness评测框架/](README.md)
