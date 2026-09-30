# 07 · agentic 编码评测：SWE-bench 是怎么工作的

> ⭐⭐ 这是全篇最重要的一章。
> 前六章讲的是【怎么给答案打分】，这一章开始讲【怎么给"干活"打分】。
> 你想知道的 Claude Code / Codex 怎么被评，答案的地基就在这里。

---

## 7.1 先把名词摆清楚

```
    ★ 名词：agentic（智能体式的）

      形容词，来自 agent（智能体 / 代理人）。
      ★ 指【模型不是回答一个问题，而是被放进一个环境里，
         自己看、自己动手、自己改，反复多轮，直到把事办成】。

    ★ 名词：SWE-bench
      SWE = Software Engineering（软件工程），bench = benchmark（基准测试集）。
      读作 "S-W-E bench"。
      ★ 一套【真实 GitHub issue（问题单）→ 让模型改代码 → 跑真单元测试】的评测集。

    ★ 名词：issue（问题单）
      GitHub 上用户提交的一条 bug 报告或功能请求。就是一段自然语言描述。

    ★ 名词：pull request，简称 PR（合并请求）
      开发者提交的一份【代码修改提案】。被仓库维护者审核通过后合入主干。

    ★ 名词：patch（补丁）
      一份纯文本的差异描述，写明"第几行删掉什么、加上什么"。
      ★ 模型的输出就是这个，不是"完整的新文件"。

    ★ 名词：unit test（单元测试）
      仓库里自带的一段代码，专门用来检查某个函数行为对不对。
      跑通=通过，报错=失败。★★ 它是 SWE-bench 唯一的裁判。
```

★★ 还有三个词会在本章反复出现，先给全称，免得后面卡住：

| 缩写 / 术语 | 英文全称 | 中文 | 一句话 |
|---|---|---|---|
| **% Resolved** | percentage of resolved task instances | 解决率 | ★★ SWE-bench 的唯一主指标：有百分之几的题最终判定为"已解决" |
| **% Apply** | percentage of patches applied successfully | 补丁应用成功率 | ★★★ 论文里和 % Resolved 并列报的第二列。★ 补丁连打都打不进去，就不可能解决（§7.5 详讲） |
| **BM25** | Best Matching 25 | 最佳匹配 25（一种稀疏检索算法） | ★ 1994 年的关键词检索打分公式，只看词频、不理解语义。SWE-bench 用它来"从仓库里挑几个文件塞给模型"（§7.9 详讲） |
| **Docker** | —（产品名，无缩写） | 容器 | 把"操作系统 + 依赖库 + 代码"整个打包冻住的技术。★★ SWE-bench 每道题起一个（§7.6 详讲） |
| **harness** | —（英文常用词） | 评测框架 / 挽具 | ★ 指"把模型和题目、判分器接起来"的那套外围程序。⚠️ 这个词在 agent 语境下还有另一个意思（脚手架），见 08 章 §8.2 |

---

## 7.2 一句话说清 SWE-bench 在干什么

| 环节 | 内容 |
|---|---|
| **给模型** | ① 一段 issue 文字　② 一整个真实代码仓库（checkout 到出问题那个 commit） |
| **要模型** | 产出一个 patch（纯文本 diff） |
| **怎么判** | 把 patch 打进仓库 → ★★ 真的把测试跑一遍 |
| **分数是** | ★ 有百分之几的题，测试全绿（% Resolved） |

✅ 论文原文（arXiv:2310.06770，ICLR 2024，摘要）：

> "an evaluation framework consisting of **2,294 software engineering problems**
> drawn from real GitHub issues and corresponding pull requests across
> **12 popular Python repositories**"

✅ 同篇 §2.2「Evaluation metrics」对"算解决了"的定义（这是最该记住的一句）：

> "If the patch applies successfully and **all of these tests pass** we consider
> the proposed solution to have successfully resolved the issue.
> The metric for our benchmark is **the percentage of task instances that are resolved**."

```
    ⟹ 🔑 注意这里【完全没有出现"答案"两个字】。
       ★★ 没有标准答案，没有正则抽取，没有裁判模型。
       只有一句话：测试跑没跑绿。

    ⟹ 🔑🔑 05 章讲的所有"抽取失败"的脏活，在这里【一次性全部消失】。
       这就是执行式验证（execution-based verification）的全部魅力。
```

---

## 7.2b ⭐⭐ 一道题从「给 issue」到「判 resolved」，完整走一遍

★★★ 这一节是本章的骨架。后面每一节都是在放大其中某一格。

```
   ① 出题（离线，只做一次）
        │  从真实 PR 里抽出：issue 文字 + base commit + 测试名单
        ▼
   ② 给模型看：issue 文字 + 一部分仓库内容
        │  ⚠️ "一部分"怎么选 —— 这一步就能让分数差几倍（§7.9）
        ▼
   ③ 模型/agent 干活 ← ★★ 这一格 07 章不管，08 章专门讲
        │  产出一份 patch 文本
        ▼
   ④ 起一个 Docker 容器，装好该仓库该版本的全部依赖
        │
        ▼
   ⑤ 把 patch 打进去（4 档容错，全失败 → APPLY_PATCH_FAIL，§7.5）
        │
        ▼
   ⑥ 用【出题时就定好的测试名单】跑测试（≤ 30 分钟，超时 → TESTS_TIMEOUT）
        │
        ▼
   ⑦ 判卷：FAIL_TO_PASS 全绿 且 PASS_TO_PASS 全绿 ⟹ RESOLVED_FULL（§7.4）
```

★ 把同一条流程按「谁做什么」摊开，并标出哪一步最容易出分歧：

| 步骤 | 谁做 | 输入 | 输出 | ⚠️ 最容易出分歧的地方 |
|---|---|---|---|---|
| ① 出题 | SWE-bench 作者（离线） | 90,000 个真实 PR | 2,294 条题目 | ⚠️ **题目本身可能不可解**——这就是后来要做 Verified 子集的原因（§7.8） |
| ② 给上下文 | ★★ **报分的人自己决定** | 仓库 3,010 个文件 / 438K 行 | 塞进提示词的那几个文件 | ★★★ **本章最大的分歧点**。BM25 检索 1.96%、oracle 4.80%、oracle-collapsed 5.93%——同一个 Claude 2（§7.9） |
| ③ 生成补丁 | ★★ **报分的人自己决定** | issue + 上下文 + 工具 | 一份 patch | ★★★ **和②并列的最大分歧点**。轮数、工具、重试、预算全在这里，SWE-bench 一概不记录（08 章） |
| ④ 起容器 | 官方 harness | 题目的 `instance_id` | 一个装好依赖的容器 | ★ 镜像拉不下来 / 装不上 ⟹ 整题报错，不是"答错" |
| ⑤ 打补丁 | 官方 harness | patch 文本 | 打进去 / 打不进去 | ★★ **格式问题冒充能力问题**。Claude 2 在 BM25 设定下只有 43.07% 的补丁打得进去（§7.5） |
| ⑥ 跑测试 | 官方 harness | 容器 + 测试名单 | 每个测试 PASS / FAIL | ★ 测试有随机性（flaky）时结果会抖；超时算超时不算失败 |
| ⑦ 判卷 | 官方 harness | 测试结果 | FULL / PARTIAL / NO | ★ 榜单通常只报 FULL 的占比，PARTIAL 不计分（§7.4） |

```
    🔑🔑 读这张表最该带走的一句：

    ★★★ 第 ④⑤⑥⑦ 步是【官方代码，谁跑都一样】。
       第 ②③ 步是【每家自己搭，完全不受约束】。
    ⟹ 所以"某模型在 SWE-bench 上 xx%"这句话，
       ★★ 在没说清②③之前，是【一个没有答案的问题】。这就是 08 章。
```

> ★ 想看 ④–⑦ 步对应的具体函数名和行号（`get_dataset_from_preds` → `import docker` → `GIT_APPLY_CMDS` → `get_eval_report`），去 [SWE-bench拆解.md](../SWE-bench拆解.md) 第 4 节。那篇是源码侧拆解，本章是概念与方法论侧，两边不重复。

---

## 7.3 题目是怎么造出来的：三级漏斗

★★ 这套流程值得看懂，因为它解释了"为什么这些题是干净的"。

✅ 论文 §2.1 原文：*"we use a **3-stage pipeline**"*

```
   Stage I  抓取
        │   从 12 个热门开源 Python 仓库抓 PR
        │   ✅ "producing about ~90,000 PRs in total"
        ▼
   Stage II  属性过滤
        │   只留【已合入（merged）】且同时满足两个条件的 PR
        ▼
   Stage III  执行过滤
        │   真的跑一遍：打补丁前 vs 打补丁后
        ▼
   2,294 条（留存率 2,294 / 90,000 = 2.5%）
```

★ 逐级看过滤条件：

| 阶段 | 过滤条件 | ✅ 论文原文 | 这一条在防什么 |
|---|---|---|---|
| **I** | 12 个热门仓库、>90% Python 代码 | "we focus on popular repositories as they tend be better maintained ... and have better test coverage" | ★ 冷门仓库测试覆盖差，跑不出信号 |
| **II-①** | PR 关联了一个 GitHub issue | "selecting the **merged** PRs that (1) resolve a GitHub issue" | ★ 保证有一段自然语言题面 |
| **II-②** | ★★★ **这个 PR 自己改了测试文件** | "and (2) make changes to the **test files** of the repository, which indicates that the user likely contributed tests to check whether the issue has been resolved" | ★★★ 保证有【真人写的验收标准】 |
| **III-①** | 至少一个测试【由失败变成通过】 | "We filter out task instances without at least one test where its status changes from a fail to pass (henceforth referred to as **fail-to-pass** test)" | ★★ 保证这道题"改对了是看得出来的" |
| **III-②** | 装得上、跑得起来 | "We also filter out instances that result in installation or runtime errors" | ★ 环境坏的题直接丢 |

```
    🔑 Stage II 的 ②【必须改了测试文件】是整个设计的灵魂。

    ★★ 因为它意味着：验收标准不是评测团队编的，
       是【当年那个真人开发者自己写的】。
       ⟹ 所以这套题基本没有"出题人偏见"。

    ⚠️ 但注意它换来了一个副作用：★★ 题目被限制在"作者顺手写了测试的那类改动"上。
       ⟹ 纯重构、纯性能优化、纯文档、改不出测试的 UI 调整，全部进不来。
       这是 SWE-bench 的【覆盖面边界】，不是缺陷，但报分时值得记着。
```

★ 论文 Table 1 给了这 2,294 条题的画像（✅ 原文数值，均为微平均）：

| 属性 | 平均 | 最大 | 读法 |
|---|---|---|---|
| issue 文字长度（词） | 195.1 | 4,477 | ★ 题面很短 |
| 仓库非测试文件数 | 3,010 | 5,890 | ★★ 但仓库很大 |
| 仓库非测试行数 | 438K | 886K | ★★★ 装不进任何上下文窗口 |
| 标准答案改了几行 | 32.8 | 5,888 | ★ 正解通常很小 |
| 标准答案改了几个文件 | 1.7 | 31 | ★★ 多数题只需改 1–2 个文件 |
| fail-to-pass 测试数 | 9.1 | 1,633 | ★ 平均 9 个"该由红变绿"的测试 |
| 该题总测试数 | 120.8 | 9,459 | ★★★ 另外一百多个测试是"不许弄坏"的（§7.4） |

```
    ⟹ 🔑🔑 把第 3 行和第 5 行放一起看，这道题的真正难度就出来了：
       ★★★【在 438K 行里，找到那 1.7 个文件、改对那 32.8 行】。
    ⟹ 所以"怎么找到该改的文件"本身就是题的一大半（§7.9）。
```

---

## 7.4 判卷：两把尺子，不是一把

★★★ 这是本章最实用的知识点。

源码在 [swebench/harness/constants/__init__.py](../源码/swebench-5.0.2/swebench/harness/constants/__init__.py)：

```python
FAIL_TO_PASS = "FAIL_TO_PASS"      # L30
FAIL_TO_FAIL = "FAIL_TO_FAIL"      # L31  ← ★ 定义了，但不参与判定
PASS_TO_PASS = "PASS_TO_PASS"      # L32
PASS_TO_FAIL = "PASS_TO_FAIL"      # L33  ← ★ 同上
```

| 尺子 | 中文 | 它在问什么 | 平均有几个测试（Table 1） |
|---|---|---|---|
| **FAIL_TO_PASS** | ★ 修复度（resolution） | 原来坏的，你修好了吗？ | 9.1 个 |
| **PASS_TO_PASS** | ★ 维持度（maintenance） | 原来好的，你弄坏了吗？ | ★★ 约 110 个（120.8 − 9.1） |

```
    ⟹ 🔑🔑 一句话：★★★【不但要修好这个 bug，还不能弄坏别的】。

    ★★ 为什么必须有第二把尺子？
       因为只看第一把的话，模型有个作弊捷径：
       把那个函数整个删掉重写成"永远返回测试期望的值"。
       ⟹ 目标测试绿了，其余 110 个测试全红。
       ⟹ PASS_TO_PASS 就是堵这个洞的。
```

源码里两者被压成 0~1 的比例，[grading.py:288 / :298](../源码/swebench-5.0.2/swebench/harness/grading.py)（★ 逐字）：

```python
def compute_fail_to_pass(report: dict[str, dict[str, Any]]) -> float:
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1                       # ← ⚠️ 没有 f2p 测试时，默认给满分
    return len(report[FAIL_TO_PASS]["success"]) / total

def compute_pass_to_pass(report: dict[str, dict[str, Any]]) -> float:
    total = len(report[PASS_TO_PASS]["success"]) + len(report[PASS_TO_PASS]["failure"])
    if total == 0:
        # TODO: Don't factor in p2p metrics
        return 1                       # ← ⚠️ 同上，而且源码自己挂了个 TODO
    return len(report[PASS_TO_PASS]["success"]) / total
```

```
    ⚠️ 注意那两个 `if total == 0: return 1`：
       ★ 一道题如果压根没有 p2p 测试，维持度会【默认记满分】。
       ⟹ 这是个合理的工程折中，但它是一条【口径】，不是数学真理。
       ★★ 源码里那句 "TODO: Don't factor in p2p metrics" 说明作者自己也觉得这里没最终定论。
```

然后 [grading.py:309-326](../源码/swebench-5.0.2/swebench/harness/grading.py) 做最终裁定（★ 逐字摘录）：

```python
def get_resolution_status(report: dict[str, dict[str, Any]]) -> str:
    """
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

三种结局定义在同文件树的 [constants/__init__.py:6-9](../源码/swebench-5.0.2/swebench/harness/constants/__init__.py)：

```python
class ResolvedStatus(Enum):
    NO      = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL    = "RESOLVED_FULL"
```

```
    ★★ 读懂这段逻辑，你会发现一个不对称：

       p2p 必须【严格等于 1】才有可能拿 PARTIAL。
       ⟹ 🔑 只要你弄坏了任何一个原本好的测试，
          不管 bug 修得多漂亮，直接 NO。

    ⟹ 🔑🔑 这个基准对"改坏东西"是零容忍的。
       ★★ 这比大多数人想象的严格得多，也是分数普遍偏低的原因之一。

    ⚠️ 另外：榜单上通常只报 FULL 的占比（即 % Resolved），
       PARTIAL 一般【不计分】。这是我的观察，不是论文明文规定。
```

---

## 7.4b ⭐⭐ 两把尺子交叉出六种结局：把论文的分类走一遍

★★★ 上面那段 `if/elif/else` 看着是三档，但论文附录 C.5 把它摊成了**六种结局**，值得对照记住——因为它告诉你"没解决"这三个字底下藏着完全不同的四种失败。

✅ 论文 Table 22 原文分类（行 = P2P 通过几个，列 = F2P 通过几个）：

| P2P（不许弄坏的） \ F2P（该修好的） | 全通过 | 部分通过 | 全不通过 |
|---|---|---|---|
| **全通过** | ✅ **Resolved**<br>（唯一计分的） | Partially Resolved | No-Op |
| **部分通过** | ⚠️ Breaking Resolved | Work in Progress | ⚠️ Regression |
| **全不通过** | ⚠️ Breaking Resolved | Work in Progress | ⚠️ Regression |

★ 六个名字的中文与含义，以及它们在 `get_resolution_status()` 里落到哪一档：

| 论文术语 | 中文 | 发生了什么 | 源码判定 |
|---|---|---|---|
| Resolved | 完全解决 | 修好了，且什么都没弄坏 | `RESOLVED_FULL` ✅ 计分 |
| Partially Resolved | 部分解决 | 只修好了一部分，但没弄坏东西 | `RESOLVED_PARTIAL` ⚠️ 通常不计分 |
| **Breaking Resolved** | ★★★ 修好了但砸了别的 | f2p 全绿，p2p 有红 | `RESOLVED_NO` ❌ |
| Work in Progress | 半成品 | f2p 部分绿，p2p 有红 | `RESOLVED_NO` ❌ |
| **No-Op** | ★★ 空操作 | p2p 全绿，但 f2p 一个都没修好 | `RESOLVED_NO` ❌ |
| **Regression** | ★★ 回归（改倒退了） | f2p 一个都没修好，还把 p2p 弄红了 | `RESOLVED_NO` ❌ |

★★★ 现在把真实数字填进去。✅ 论文 Table 23，Claude 2 那一列（**所有"补丁成功打进去了"的生成**）：

| 结局 | 条数 | 占比（÷1,078） | 一句话 |
|---|---|---|---|
| Resolved | 110 | **10.2%** | ★ 只有这一档计分 |
| Breaking Resolved | 26 | 2.4% | ★★★ 修好了，但砸了别的 ⟹ 0 分 |
| Partially Resolved | 15 | 1.4% | 部分修好 |
| Work in Progress | 20 | 1.9% | 半成品 |
| **No-Op** | **471** | **43.7%** | ★★★ 补丁打进去了，一个目标测试都没修好 |
| **Regression** | **436** | **40.4%** | ★★★ 没修好，还把原来好的弄红了 |
| **合计** | **1,078** | 100% | 110+26+15+20+471+436 = 1,078 ✅ 自洽 |

```
    🔑🔑🔑 三条从这张表里直接算出来的结论：

    ① No-Op + Regression = 471 + 436 = 907 条 = ★★★ 84.1%
       ⟹ 【补丁打进去了】的生成里，有 84% 连一个目标测试都没修好。
       ⟹ 🔑 所以那个 1.96% 的低分里，绝大部分不是"差一点"，是"完全没碰到点上"。

    ② Regression 一项就占 40.4%。
       ⟹ ★★★ 五分之二的补丁【把原来能跑的东西弄坏了】。
       ⟹ 🔑 这就是 PASS_TO_PASS 这把尺子存在的全部理由。
          如果只看 FAIL_TO_PASS，这 436 条里有多少会被误判成"无害"？全部。

    ③ Breaking Resolved 有 26 条。
       ⟹ ★★ 这 26 条【issue 真的修好了】，但因为碰坏了别的测试，得 0 分。
       ⟹ 26 / (110+26) = 19.1% —— ★★★ 在"真把 issue 修好了"的 136 条里，
          有将近两成因为"顺手砸了别的"而一分不得。这就是零容忍的代价。

    ⚠️ 口径提醒（我核对时发现的不一致，必须说明）：
       论文 Table 23 的 "Applied" 写 1,078，但 Table 18 说 Claude 2 在 oracle 设定下
       % Apply = 62.82%（≈1,441 条），Table 5 说 BM25 设定下 43.07%（≈988 条）。
       ⟹ ⚠️ Table 23 没写明用的是哪个检索设定，三个数对不上。
       ⟹ 所以上面这张表【只能按内部占比读】（它自己加得起来），
          ★★ 不要拿 1,078 这个绝对数去和别处的 % Apply 相乘。
```

---

## 7.5 一个特别有教育意义的细节：打补丁本身会失败

模型输出的是一份文本 patch。★★ 但这份 patch 经常【打不进去】——行号对不上、上下文差一个空格、缩进错了。

[run_evaluation.py:54](../源码/swebench-5.0.2/swebench/harness/run_evaluation.py) 里是这样兜底的（★ 逐字）：

```python
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]
```

| 档 | 命令 | 宽松程度 | 它容忍什么 |
|---|---|---|---|
| 第 1 档 | `git apply` | 最严 | ★ 一个字符不对就拒绝 |
| 第 2 档 | `git apply --3way` | 中 | ★ 三方合并，容忍一点上下文漂移 |
| 第 3 档 | `git apply --reject` | 宽 | 能打的打，打不了的单独记下来 |
| 第 4 档 | `patch --fuzz=5` | ★★ 最宽松 | ★★ 允许模糊匹配，上下文最多漂 5 行 |

[run_evaluation.py:301](../源码/swebench-5.0.2/swebench/harness/run_evaluation.py) 里逐档尝试：`for attempt, git_apply_cmd in enumerate(GIT_APPLY_CMDS):`

★ 而且每换一档之前会先把工作区洗干净（同文件 L302-309，★ 逐字注释）：

```python
if attempt:
    # a failed attempt (notably --reject) leaves partial state behind,
    # which makes every later command fail; restart from a pristine tree
    container.exec_run(["/bin/bash", "-c", "git checkout -- . ; git clean -fd"], ...)
```

四档全失败，就打上这个标记（[constants/__init__.py:44](../源码/swebench-5.0.2/swebench/harness/constants/__init__.py)）：

```python
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
```

★★★ 这件事到底有多严重？✅ 论文 Table 5 把 % Apply 和 % Resolved 并列报了出来（BM25 检索设定）：

| 模型 | SWE-bench 全量 % Resolved | SWE-bench 全量 **% Apply** | ★ 打进去的里面有几成解决了 |
|---|---|---|---|
| Claude 3 Opus | 3.79 | 46.56 | 3.79 / 46.56 = 8.1% |
| Claude 2 | 1.97 | 43.07 | 1.97 / 43.07 = 4.6% |
| GPT-4-turbo | 1.31 | 26.90 | 1.31 / 26.90 = 4.9% |
| ChatGPT-3.5 | 0.17 | **26.33** | 0.17 / 26.33 = 0.6% |
| SWE-Llama 7b | 0.70 | 51.74 | 0.70 / 51.74 = 1.4% |

```
    ⟹ 🔑🔑🔑 盯住 % Apply 这一列：
       ★★★ ChatGPT-3.5 有【73.7% 的补丁连打都打不进去】。
       ⟹ 也就是说，在它那 99.83% 的"没解决"里，
          ★★ 有四分之三根本没走到"跑测试"这一步。

    ⟹ ★★★ 结论：在 2023 年那批分数里，
       【输出格式能力】和【解决问题能力】是被搅在一个数字里的。
       ⚠️ 看到一个很低的 agentic 分数，第一个该问的不是"它笨不笨"，
          而是"它的补丁打进去了几成"。
```

★★ 论文还做了一个更直接的证据：把"输出 patch"换成"输出整个文件"，分数直接腰斩。

> ✅ 论文 §5 原文：*"we find that models generally perform worse at this task than when generating patch files; for instance, **Claude 2 scores at 2.2% compared to 4.8%** in the main table for "oracle" retrieval."*

| 同一个 Claude 2、同一套 oracle 上下文 | % Resolved |
|---|---|
| 要它输出 patch（diff 格式） | ✅ 4.8% |
| 要它输出改好的整个文件 | ✅ 2.2% |

```
    ⟹ 🔑🔑 【输出格式一换，分数掉一半】。模型一个字节都没变。
    ⟹ ★★★ 这和 05 章"抽取规则换一个，分数换一个"是【完全同一件事】，
       只是从"答案怎么抽"搬到了"改动怎么表达"。
```

```
    🔑🔑 这段代码在讲一个和 05 章【完全同构】的道理：

       ★ 05 章：模型答对了，但正则没抽出来 ⟹ 记成答错
       ★ 07 章：模型改对了，但 patch 格式歪了 ⟹ 记成没解决

    ⟹ ★★★ 换了范式，"格式问题冒充能力问题"这个幽灵还在，
       只是从【抽取层】搬到了【应用层】。

    ⟹ 但 SWE-bench 做了一件对的事：它【给这种失败单独立了一个标记】，
       而且【在论文里单独报了一列 % Apply】。
       ⚠️ 而不是像 "[invalid]" 那样，混进"答错"里看不见了。
       ★★ 你自己搭评测时，请学这两点。
```

还有一个同级标记（[constants/__init__.py:50](../源码/swebench-5.0.2/swebench/harness/constants/__init__.py)）：

```python
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
```

★ 测试卡死也要和"测试失败"分开记。同一个思想。同文件里还并列放着 `RESET_FAILED` / `TESTS_ERROR` / `TESTS_FAILED` / `TESTS_PASSED`（L46-49）——★★ 五种失败方式，五个不同的标记，一个都不混。

---

## 7.6 为什么这种评测这么贵

[run_evaluation.py:3](../源码/swebench-5.0.2/swebench/harness/run_evaluation.py) 第一眼就说明了一切：

```python
import docker
```

同文件 L61-62 的两个常量：

```python
DOCKER_CLIENT_TIMEOUT   = int(os.environ.get("SWEBENCH_DOCKER_TIMEOUT",   "1800"))
DOCKER_CLIENT_POOL_SIZE = int(os.environ.get("SWEBENCH_DOCKER_POOL_SIZE", "128"))
```

```
    ★ 1800 秒 = 30 分钟。★★【单道题】判卷环节的超时上限。
    ★ 128 = 容器连接池大小，说明设计上就是要【大规模并发】的。
```

⟹ 把 02 章 §2.4 的成本论断落到实处：

| 范式 | 一道题的开销 | 需要什么 | 跑一轮的现实节奏 |
|---|---|---|---|
| ① 判别式（选择题比概率） | 1 次前向 | 一张卡 | ★ 分钟级 |
| ② 生成式（自由作答） | 几百~几千 token | 一张卡 | ★ 小时级 |
| ③ **agentic** | ★★ 几十轮对话 + 一整个容器 | ★★★ 容器 + 装依赖 + 跑测试 + 最多 30 分钟 | ★★★ 小时到天级 |

```
   ⟹ 🔑 这就是为什么 MMLU 每天都能跑，而 SWE-bench 一个季度才跑一次。

   ⚠️ 而且请注意：那 30 分钟【只是判卷的超时】。
      ★★★ 生成补丁那一段花了多久、烧了多少 token，
      SWE-bench 完全不记录、也不限制 —— 这是 08 章 §8.5 的主题。
```

---

## 7.7 分数的历史：从 1.96% 到 70%+

✅ 论文摘要（2023 年 10 月）：

> "The best-performing model, **Claude 2**, is able to solve a mere **1.96%** of the issues."

✅ 论文 Table 2（BM25 检索，三种上下文长度上限）：

| 模型 | 13k 上下文 | 27k 上下文 | 50k 上下文 |
|---|---|---|---|
| Claude 2 | **1.96** | 1.87 | 1.22 |
| SWE-Llama 7b | 0.70 | 0.31 | 0.00 |
| SWE-Llama 13b | 0.70 | 0.48 | 0.00 |

```
    ⚠️⚠️ 先别急着看那个 1.96，★★★ 先看这张表【从左往右是在往下走的】。
       ⟹ 上下文给得越多，分数越低。三个模型，三条线，方向一致。
       ⟹ 🔑 这一条反直觉的事实，§7.9 会给出论文自己的解释。
```

```
    ⚠️ 一个必须指出的口径小坑：论文自己有两个略微不同的数。
       摘要和 §5 正文写 Claude 2 = 1.96%（对应 Table 2 的 13k 那一格），
       ✅ 而 Table 5（主结果表）里 Claude 2 = 1.97%。
       ⟹ ★ 差 0.01 个百分点不影响任何结论，
          但它提醒你：★★ 同一篇论文里【引哪张表】都要写清楚。
```

```
    ⚠️ 到 2025-2026 年，前沿模型 + 成熟脚手架在 SWE-bench Verified 上
       ★ 已经普遍报到 70% 以上。这是我依据公开报道的概括，
       ⚠️ 不是来自本篇论文，具体数字请以各家发布当天的原始说明为准。
    ⚠️ 而且那类数字通常是【厂商自报】，不是第三方复现——
       ★★ 看到模型卡上的 SWE-bench 分数，先问一句"谁跑的"。

    ★★ 但请注意：1.96% → 70% 这个跨度里，
       至少有【四件事】同时变了，而榜单把它们混成了一个数字：

       ① 模型变强了
       ② 脚手架变强了（1.96% 那一档根本没有 agent 循环，08 章 §8.2）
       ③ 题集换了（全量 2,294 → Verified 500，§7.8）
       ④ 上下文给法换了（BM25 → agent 自己去仓库里翻，§7.9）

    ⟹ 🔑 这正是下一章要拆的东西。
```

---

## 7.7b ⭐⭐ pass@k：把那个组合数公式用小数字走一遍

★★★ 这一节和 SWE-bench 本身无关，但**所有代码类评测的分数都印着这个指标**，而它有两三种算法、数值不同。不走一遍，你永远会把两个不可比的数放在一起比大小。

### ① 名词与三种算法

```
    ★ 名词：pass@k
      ★ 让模型对同一道题答 k 次，★★ 只要有一次通过测试就算这道题过。
      再把所有题的"过了没"平均起来，就是 pass@k。
      ⟹ 它出自 HumanEval / Codex 论文（arXiv:2107.03374，OpenAI，2021）。
         ★ 你问的 Codex，这个指标的源头就在那篇。

    ★ 记号约定（下面全用这三个字母）：
      n = 一道题实际采了几个样本
      c = 这 n 个里有几个通过了测试（correct）
      k = 你想报的那个 pass@k 里的 k
```

★★★ 同一个名字 `pass@k`，公开材料里至少有三种算法：

| # | 算法 | 公式 | ✅ HumanEval 论文怎么评价它 |
|---|---|---|---|
| ① | **直接法**：就采 k 个，看有没有对的 | 至少一个通过 → 1，否则 → 0 | ✅ 无偏，但 *"computing pass@k in this way can have **high variance**"* |
| ② | **无偏估计量**（论文正式定义） | 1 − C(n−c, k) / C(n, k)，要求 n ≥ k | ✅ *"calculate the **unbiased estimator**"*；论文自己用 n = 200、k ≤ 100 |
| ③ | **插值法**（看着最自然，但错） | 1 − (1 − c/n)^k | ⚠️ ✅ *"we show that it is **biased** in Appendix A"* ⟹ **系统性低估** |

```
    ⚠️⚠️ 这里有一处广为流传的说法需要纠正：

       常见说法：「直接采 k 次那种算法【有偏】且方差大」。
       ✅ 论文的原话不是这样：★★ 直接法（①）是【无偏】的，
          ✅ Appendix A 原文："only the empirical estimate used by
             Kulal et al. (2019), and (1) are unbiased."
          （其中 (1) 就是②那个组合数公式）
       ⟹ ① 的问题是【方差大】，不是有偏。
       ⟹ ★★★ 真正【有偏】的是③那个 1−(1−p̂)^k 插值法，而且偏的方向是【偏低】。
       ⟹ 🔑 记住这个区分，因为"有偏"和"方差大"要用完全不同的办法对付：
          方差大 ⟹ 多采样本；有偏 ⟹ 换公式，采再多也不收敛到真值。
```

### ② 用 n = 10、c = 3 走一遍

★★ 设定：一道题采了 10 个样本，3 个通过测试。现在分别算 pass@1 / pass@5 / pass@10。

**② 无偏估计量的直觉**：`C(n−c, k) / C(n, k)` 读作「从这 10 个样本里**不放回**地抽 k 个，★ 恰好全抽到失败样本的概率」。1 减去它，就是「抽到的 k 个里至少有一个是对的」的概率。

| k | C(7, k)（只从 7 个失败样本里抽） | C(10, k)（从全部 10 个里抽） | 比值 = 全抽到失败的概率 | **② 无偏估计量 = 1 − 比值** | **③ 插值法 = 1 − 0.7^k** | ⚠️ 两者差（百分点） |
|---|---|---|---|---|---|---|
| **1** | C(7,1) = 7 | C(10,1) = 10 | 7/10 = 0.700 | **30.0%** | 1 − 0.7 = **30.0%** | 0.0 |
| **5** | C(7,5) = 21 | C(10,5) = 252 | 21/252 = 0.0833 | **91.7%** | 1 − 0.16807 = **83.2%** | ⚠️ **8.5** |
| **10** | C(7,10) = **0** | C(10,10) = 1 | 0/1 = 0 | **100%** | 1 − 0.02825 = **97.2%** | ⚠️ **2.8** |

★ 三行都可以自己验算，算术全在这里：

```
k = 1 :  C(7,1)/C(10,1) = 7/10 = 0.7        ⟹ 1 − 0.7 = 0.300 = 30.0%
         ★ 注意 k=1 时无偏估计量正好等于 c/n = 3/10，和插值法重合。
         ⟹ 🔑 所以 pass@1 是三种算法唯一不会打架的那个 k。

k = 5 :  C(7,5)  = 7!/(5!·2!)  = 21
         C(10,5) = 10!/(5!·5!) = 252
         21 / 252 = 1/12 = 0.08333          ⟹ 1 − 0.08333 = 0.9167 = 91.7%
         插值法：0.7^5 = 0.16807            ⟹ 1 − 0.16807 = 0.8319 = 83.2%
         ⚠️ 差 91.67 − 83.19 = 8.48 个百分点。

k = 10:  C(7,10) = 0   ← ★★ 只有 7 个失败样本，抽不出 10 个全失败的组合
         ⟹ 1 − 0 = 1.000 = 100%
         ★ 直觉检查：我们【已经知道】这 10 个里有 3 个通过，
           所以"从这 10 个里取 10 个、里面有对的"当然是必然事件。
         插值法：0.7^10 = 0.028248          ⟹ 97.2%，⚠️ 差 2.8 个百分点。
```

```
    🔑🔑 从这张表带走三条：

    ① ★★★ 插值法（③）在中间的 k 上偏得最狠（这里 8.5 个百分点），
       而且【永远偏低】。⚠️ 两家一家用②一家用③，差的这 8 个百分点
       会被读成"模型差一截"。

    ② ★ k = n 时，②退化成①：因为"从 n 个里取 n 个"没有随机性了，
       结果就是"这 n 次里有没有一次对"的那个 0/1 指示量。
       ⟹ 🔑 所以 n = k 的 pass@k 报出来，方差是最大的那一档。

    ③ ★★ k = 1 时三种算法一致 ⟹ 这就是为什么绝大多数榜单只报 pass@1：
       ⚠️ 不是因为它最有信息量，而是因为它最不容易吵架。
       （⚠️ 但 pass@1 自己也有两种定义，见 [附-速查表.md](附-速查表.md) D 节）
```

### ②b ⭐ 用最小的例子把「有偏」证死：n = 2、k = 2、p = 0.5

★★ 上面那张表只说明"两个公式给的数不一样"，还没证明哪个偏。★★★ 下面这个例子小到可以把**所有可能情况**列完，于是"偏"这件事就没得争了。

★ 设定：一道题的真实单次通过率 p = 0.5，每道题采 n = 2 个样本，要报 pass@2。★ 真值是 1 − (1 − 0.5)² = **0.75**。

| 采样结果 | 出现概率（二项分布） | **② 无偏估计量** 1 − C(2−c, 2)/C(2,2) | **③ 插值法** 1 − (1 − c/2)² |
|---|---|---|---|
| c = 0（两个都错） | 0.5 × 0.5 = **0.25** | 1 − C(2,2)/1 = 1 − 1 = **0** | 1 − (1 − 0)² = **0** |
| c = 1（一对一错） | 2 × 0.5 × 0.5 = **0.50** | 1 − C(1,2)/1 = 1 − 0 = **1** | 1 − (1 − 0.5)² = **0.75** |
| c = 2（两个都对） | 0.5 × 0.5 = **0.25** | 1 − C(0,2)/1 = 1 − 0 = **1** | 1 − (1 − 1)² = **1** |

★ 按概率加权求期望（★ 算术全在这里，可自行验算）：

```
   ② 的期望 = 0.25×0 + 0.50×1    + 0.25×1 = 0 + 0.500 + 0.25 = 0.750
              ⟹ ★★★ 正好等于真值 0.75。这就是"无偏"的含义。

   ③ 的期望 = 0.25×0 + 0.50×0.75 + 0.25×1 = 0 + 0.375 + 0.25 = 0.625
              ⟹ ⚠️ 0.625 < 0.750，★★★ 系统性低估 12.5 个百分点。

   ⟹ 🔑🔑 差别全在 c = 1 那一行：
      ★★ 真实情况是"这 2 个样本里有一个是对的"，
         所以"从这 2 个里取 2 个、里面有对的"是【必然事件】= 1。
      ② 算出 1，是对的。③ 算出 0.75，★★★ 因为它假装这 2 次抽样是
         【有放回、互相独立】的——而它们不是，它们就是同一批样本。
      ✅ 论文 Appendix A 原文一句话点破：
      > "The interpretation of this estimator is that we draw k samples
      >  **with replacement** from a pool of n candidates, but the k samples
      >  **are not independent**."
```

```
    ⚠️ 两个必须补上的限定，否则容易走到另一个极端：

    ① ★ ③ 是【有偏但一致（consistent）】的：n 越大偏差越小。
       ⚠️ 但 ✅ 论文说："The gap **doesn't fully close even when n > 5k**,
          and results can seem better with more samples."
       ⟹ 🔑 也就是说：★★★ 用③的人【多采样本就能显得分数更高】，
          这正是不能拿③和②比的原因。

    ② ★ ② 自己也是有噪声的。前面 n=10、c=3 那一行算出 91.7%，
       而若真实 p = 0.3，pass@5 的真值是 83.2% —— ★★ 这 8.5 个百分点的差
       是②的【方差】，不是②的偏差。
       ⟹ 🔑 区别在于：★★★ 把整个"采 10 个"的实验重复很多次再平均，
          ② 会收敛到 83.2%；③ 不会。
```

### ③ 为什么①的「方差大」是个真问题

★ 接着上面的设定，假设这道题的真实单次通过率 p = 0.3，那么真实 pass@5 = 1 − 0.7^5 = 83.2%。

| 算法 | 一道题上能给出什么值 | 用了多少信息 |
|---|---|---|
| ① 直接采 5 个看有没有对的 | ★★ 只能是 **0 或 1** | 5 个样本，而且只留一个 bit |
| ② 用 n = 10 算组合数 | ★ 一个连续值（91.7%） | ★★ 10 个样本，等价于把 C(10,5) = **252 种抽法全部平均了一遍** |

★ 把①的噪声算出来（★ 纯二项分布算术，可自行验算）：

```
   单题方差  = pass@5 × (1 − pass@5) = 0.832 × 0.168 = 0.1398
   单题标准差 = √0.1398 = 0.374

   在 HumanEval 那样的 164 道题上取平均：
   平均值的标准差 = 0.374 / √164 = 0.0292 ≈ ★★ ±2.9 个百分点（1 个标准差）

   ⟹ 🔑 也就是说：两套系统真实水平完全一样，
      用①这种算法各跑一次，报出来的 pass@5 相差 5–6 个百分点
      ★★★ 是完全正常的统计波动，不代表谁强。
   ⟹ ② 用同样的样本预算把这个噪声压下来，这就是它存在的理由。
```

### ④ 回到 SWE-bench：它的 pass@1 是哪一种？

✅ 论文 Appendix D.2 原文（这一句是 SWE-bench 全部分数的口径）：

> "Since generations are relatively expensive, we only generate **a single patch file per instance**.
> Following precedent in code generation for evaluation in Pass@1 (Chen et al., 2021; Rozière et al., 2023),
> we simply use **greedy decoding** for all models."

| 口径项 | SWE-bench 原论文的取值 |
|---|---|
| 用哪种 pass@k 算法 | ★★ **①直接法，k = 1**（不是②那个估计量） |
| n（每题采几个） | **1** |
| 解码方式 | **greedy（贪心，温度 0）** |
| 所以 % Resolved 等于 | ★ 「跑一遍、看解决了几道」——最朴素的那种 |

```
    ⟹ 🔑🔑 所以：★★★【SWE-bench 榜单上的 % Resolved 就是 pass@1，
       而且是 n=1 greedy 的那种 pass@1】。
    ⟹ ⚠️ 它和"采 6 次取平均的 pass@1"、"采 64 次取平均的 pass@1"
       都不是一个东西。后两种见 [附-速查表.md](附-速查表.md) D 节和 10 章 §10.7b。

    ⚠️ 一个必须提防的操作：有人会跑 N 遍、挑最好的一份交上去，
       然后仍然把结果写成 "pass@1"。
       ★★★ 那其实是 pass@N，而且如果挑选时看了 f2p 测试，连 pass@N 都算不上。
       ⟹ 这一整套坑，08 章 §8.5 用实测数字拆开讲。
```

> ✅ ①②③ 三种算法的原文依据、numpy 稳定实现、以及偏差的完整推导，都在 [papers/HumanEval-Codex-arXiv-2107.03374.pdf](../papers/HumanEval-Codex-arXiv-2107.03374.pdf) 的 §2.1 与 Appendix A。

---

## 7.8 为什么会有 "SWE-bench Verified"

```
    ★ 名词：SWE-bench Verified（人工校验子集）
      ★★ 从原始 2,294 条里，由专业工程师逐条人工审核后留下的 500 条。
      ⚠️ 它【不是本篇论文的产物】，是后来（2024 年）由 OpenAI 发布的另一套子集。
         ⟹ 具体的标注流程、标注人数、剔除标准，请以官方公告为准，
            本章只说它的性质和它对分数的影响。

    ⚠️ 它解决的是原始集的三个毛病（我的归纳）：
      ① 有些 issue 描述【本身就说不清要什么】，人来了也做不对
      ② 有些 fail-to-pass 测试【超出 issue 的要求范围】，等于超纲
      ③ 有些题的环境配置本身就是坏的

    ⟹ 🔑 所以 "SWE-bench 45%" 和 "SWE-bench Verified 45%" ★★ 不是一回事，
       后者是被清洗过的、更容易的子集。
       ⚠️ 看到分数第一件事：★★★ 问清楚是哪个子集。
```

论文里已经埋了这个伏笔——作者自己先做了个 Lite 子集：

> ✅ "we create a **Lite** subset of **300 instances** from SWE-bench that have been
> sampled to be more self-contained, with a focus on evaluating functional bug fixes."

★ 于是同一个名字下至少有三套题：

| 题集名 | 题数 | 谁做的 / 什么时候 | 怎么挑的 | 相对难度 |
|---|---|---|---|---|
| **SWE-bench**（full） | 2,294 | ✅ 原论文（2023-10） | 三级漏斗的全部产出（§7.3） | ★ 基准 |
| **SWE-bench Lite** | 300 | ✅ 原论文附录 A.7 | *"sampled to be more self-contained, with a focus on evaluating functional bug fixes"*；✅ 覆盖原 12 个仓库中的 11 个 | ★★ 更容易 |
| **SWE-bench Verified** | 500 | ⚠️ OpenAI（2024，非本论文） | ⚠️ 专业工程师逐条人工审核，剔除描述不清 / 测试超纲 / 环境损坏的题 | ★★★ 最容易 |

```
   ★★ 三者分数不可直接比较。

   ⚠️ 为什么 Verified 系统性更高？因为它【剔掉的正是模型做不出来的那些题】。
      ⟹ 🔑 这不是"作弊"，这是在回答一个不同的问题：
         full  问的是「在真实 GitHub 的全部噪声里，你能搞定几成」
         ★★ Verified 问的是「在一道【确认可解】的题上，你能搞定几成」
      ⟹ ★★★ 两个问题都合理，但【把两个答案放一张表里比大小】不合理。
```

> ⚠️ 这三个题集的对照，[附-速查表.md](附-速查表.md) D 节 ③ 也有一份；本节额外给了"谁做的 / 怎么挑的"两列，用来判断一个分数能不能横向比。

---

## 7.9 一个必须警惕的坑：上下文是怎么给的

★★★ 这是我认为新手最容易忽略、但影响最大的一件事。

论文 §4.1 提供了两种给上下文的方式（后来附录里又多了第三种）：

| 设定 | 怎么挑文件给模型 | ✅ 论文自己的评价 | 现实性 |
|---|---|---|---|
| **① BM25 稀疏检索** | ★ 拿 issue 文字当查询词，用关键词匹配从几千个文件里挑几个 | ✅ *"in almost half of the instances with the 27,000-token limit, it retrieves **none of the files** from the 'oracle' context"* | ★★ 接近真实（人也得自己找） |
| **② "Oracle" 检索**（先知设定） | ★★★ 直接把【标准答案改过的那些文件】给模型。✅ D.1：*"file paths are simply extracted directly from the reference solution's patch file excluding test files"* | ✅ *"less realistic"* | ⚠️ 等于告诉考生"答案在第 37 页" |
| **③ "Oracle"-collapsed** | ★ 在②的基础上，把那些文件里【不是标准答案改过的部分】折叠掉（只留改动处 ±15 行） | ✅ 论文把它当"输入消融实验" | ⚠️ 比②还要更近一步 |

### ⚠️⚠️ 先看那个反直觉的事实：上下文给得越多，分数越低

★★★ 把论文 Table 2 和 Table 3 并排放，方向正好相反：

| 上下文长度上限 | **Table 3：BM25 召回率**（挑中的文件里覆盖了多少 oracle 文件） | **Table 2：Claude 2 的 % Resolved** |
|---|---|---|
| 13k | 29.58 | **1.96** |
| 27k | 44.41 ↑ | 1.87 ↓ |
| 50k | 51.06 ↑ | **1.22** ↓↓ |

```
    ⟹ 🔑🔑🔑 读这张表：★★★【该给的文件给对得更多了，分数反而更低了】。

    ✅ 论文自己的解释（原文）：
    > "Even when increasing the maximum context size for BM25 would increase
    >  recall with respect to the oracle files, performance drops, as shown in
    >  Table 2, as models are **simply ineffective at localizing problematic code**."
    ✅ 以及：
    > "models become **distracted by additional context** and may be sensitive to
    >  the relative location of target sequences"

    ⟹ ★★★ 这直接反驳了"上下文窗口越大越好"的朴素想法。
       🔑 定位能力是一项【独立的能力】，不会随窗口变大而自动获得。
    ⟹ ★★ 08 章 §8.4.2 会看到同一条规律在 agent 侧的重演：
       SWE-agent 把"整个文件塞进去"的分数（12.7）反而低于"只给 100 行窗口"（18.0）。
```

### ★★★ 三种设定下，同一个 Claude 2 的分数差多少

✅ 全部来自本篇论文（Table 2 / Table 18 / Table 6），模型权重一个字节没改：

| 上下文设定 | Claude 2 % Resolved | 相对 BM25 的倍数 | % Apply | 出处 |
|---|---|---|---|---|
| ① BM25 检索（13k） | **1.96** | 1.0× | 43.07 | Table 2 / Table 5 |
| ② Oracle 检索 | **4.80** | ★★ **2.4×** | 62.82 | Table 18 |
| ③ Oracle-collapsed | **5.93** | ★★★ **3.0×** | — | Table 6 |

★ 同一组设定换别的模型，方向完全一致（✅ Table 18 与 Table 6）：

| 模型 | ① BM25 | ② Oracle | ③ Oracle-collapsed |
|---|---|---|---|
| Claude 3 Opus | 3.79 | — | **9.39** |
| Claude 2 | 1.96 | 4.80 | 5.93 |
| GPT-4 ⚠️ | 1.31 | 1.74 | 3.40 |
| ChatGPT-3.5 | 0.17 | 0.52 | 1.09 |
| SWE-Llama 13b | 0.70 | 3.97 | — |

```
    ⚠️ 那个打 ⚠️ 的 GPT-4 行必须加口径说明。✅ 论文 Table 18 脚注原文：
    > "Due to budget constraints we evaluate GPT-4 on a **25% random subset** of
    >  SWE-bench in the 'Oracle' and BM25 27K retriever settings only."
    ⟹ ★★ 也就是 574 道题，不是 2,294 道。
    ⟹ 🔑 论文诚实地把这件事写在了表下面。★★★ 这就是"报分要报口径"的正面样板：
       ⚠️ 同一张表里有一行是【不同样本量】的，不标出来就是在骗人。
```

```
    ⚠️ 论文 Table 1 说仓库平均 3,010 个非测试文件、438K 行。
    ⟹ 🔑🔑 所以【怎么找到该改的文件】本身就是这道题的一大半难度。

    ⟹ ★★★ 结论，现在有数字了：同一个模型，在 oracle 设定下的分数
       是"自己去仓库里翻"设定下的 ★★ 2.4 到 3.0 倍。
       这不是模型的差别，是【题面难度的差别】。

    ⟹ 这正好呼应 01 章的总论点：
       ★★★【分数是"模型+评测方法"的联合产物】。
       在 agentic 评测里，"评测方法"这一项的权重比前面任何一章都大。
```

---

## 7.10 本章总结

```
   ★★ 一图流：

   issue 文字 + 整个仓库（438K 行）
        │
        ▼  ② 挑哪几个文件给它看     ← ★★★ BM25 / oracle，差 2.4–3.0 倍（§7.9）
   ┌─────────────────
   │ 模型 / agent 干活              ← ★★ 这一格 07 章没管，08 章专门讲
   └─────────────────
        │  产出 patch
        ▼
   ┌─────────────────
   │ 起容器、打补丁                 ← 4 档 fallback；失败 → APPLY_PATCH_FAIL
   └─────────────────                  ★★★ % Apply 低到 26%–52%（§7.5）
        │
        ▼
   ┌─────────────────
   │ 跑测试（≤ 30 min）            ← 卡死 → TESTS_TIMEOUT
   └─────────────────
        │
        ▼
   FAIL_TO_PASS = 1 且 PASS_TO_PASS = 1 ⟹ RESOLVED_FULL
   其余五种结局全部算 RESOLVED_NO（§7.4b）
```

★ 本章出现过的关键数字，集中放一遍（全部 ✅ 来自 arXiv:2310.06770 或 swebench-5.0.2 源码）：

| 数字 | 含义 | 出处 |
|---|---|---|
| 90,000 → 2,294 | 三级漏斗的留存（2.5%） | ✅ §2.1 / §2.2 |
| 12 | 原始仓库数（Lite 覆盖 11 个） | ✅ 摘要 / 附录 A.7 |
| 3,010 / 438K | 平均非测试文件数 / 行数 | ✅ Table 1 |
| 9.1 / 120.8 | 平均 f2p 测试数 / 总测试数 | ✅ Table 1 |
| 1,800 秒 / 128 | 单题判卷超时 / 容器池大小 | ✅ run_evaluation.py:61-62 |
| 43.07% | Claude 2 在 BM25 下的 % Apply | ✅ Table 5 |
| 84.1% | 打进去的补丁里"一个目标测试都没修好"的比例 | ✅ Table 23 推算 |
| 1.96 / 4.80 / 5.93 | Claude 2 在三种上下文设定下的 % Resolved | ✅ Table 2 / 18 / 6 |
| 2,294 / 300 / 500 | full / Lite / Verified 的题数 | ✅ 论文 + ⚠️ OpenAI 公告 |

```
    🔑 六条带走：

    ① ★★ 判卷靠【真跑测试】，不靠答案比对 ⟹ 05 章的抽取脏活全免
    ② ★★★ 两把尺子：修好了（f2p）+ 没弄坏（p2p），后者零容忍。
       ✅ 实证：Table 23 里有 40.4% 的补丁属于 Regression（没修好还弄坏了）
    ③ ★★ 格式问题依然存在（patch 打不进去），但它被【单独标记】并【单独报了一列】。
       ✅ 实证：ChatGPT-3.5 有 73.7% 的补丁连打都打不进去
    ④ ★★ 贵在容器：每题一个 Docker、判卷超时 30 分钟 ⟹ 季度级而非日级。
       ⚠️ 而【生成阶段】的时间和 token 花费，SWE-bench 完全不记录
    ⑤ 🔑🔑 ★★★ full / Lite / Verified 是三套题；BM25 / oracle / oracle-collapsed
       是三种难度。⟹ 报分数不说清这两件事，等于没报
    ⑥ ★★ pass@k 至少有三种算法（§7.7b）：直接法无偏但方差大、组合数估计量无偏、
       插值法【系统性偏低】。⚠️ SWE-bench 自己用的是"n=1、greedy"的直接法
```

---

> 下一章：[08-ClaudeCode与Codex怎么被评.md](08-ClaudeCode与Codex怎么被评.md) —— ⭐⭐ 上面那个"模型/agent 干活"的格子里到底是什么，以及为什么"评的其实不只是模型"。
> 源码侧的逐函数拆解：[SWE-bench拆解.md](../SWE-bench拆解.md)
> 返回 [评测入门总目录](README.md)
