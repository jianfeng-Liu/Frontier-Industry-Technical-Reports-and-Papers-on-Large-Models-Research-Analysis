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

---

## 7.2 一句话说清 SWE-bench 在干什么

```
   ┌──────────────────────────────────────────────────────────┐
   │  给模型：① 一段 issue 文字   ② 一整个真实代码仓库        │
   │  要模型：产出一个 patch                                   │
   │  怎么判：把 patch 打进仓库 → ★★ 真的把测试跑一遍         │
   │  分数是：★ 有百分之几的题，测试全绿                       │
   └──────────────────────────────────────────────────────────┘
```

✅ 论文原文（arXiv:2310.06770，ICLR 2024，摘要）：

> "an evaluation framework consisting of **2,294 software engineering problems**
> drawn from real GitHub issues and corresponding pull requests across
> **12 popular Python repositories**"

✅ 同篇 §2.2 对"算解决了"的定义（这是最该记住的一句）：

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

## 7.3 题目是怎么造出来的：三级漏斗

★★ 这套流程值得看懂，因为它解释了"为什么这些题是干净的"。

✅ 论文 §2.1 原文：*"we use a **3-stage pipeline**"*

```
   ┌─ Stage I  抓取 ───────────────────────────────────┐
   │  从 12 个热门 Python 仓库抓 PR                     │
   │  ✅ "producing about ~90,000 PRs in total"         │
   └───────────────────────────────────────────────────┘
                        ↓
   ┌─ Stage II  属性过滤 ──────────────────────────────┐
   │  只留【已合入】且同时满足：                        │
   │    ① 关联了一个 issue                              │
   │    ② ★★ 这个 PR 自己改了测试文件                  │
   │  ⟹ 说明作者顺手写了"验收标准"                      │
   └───────────────────────────────────────────────────┘
                        ↓
   ┌─ Stage III  执行过滤 ─────────────────────────────┐
   │  ★★★ 真的跑一遍：打补丁前 vs 打补丁后             │
   │  必须至少有一个测试【由失败变成通过】              │
   │  ✅ "henceforth referred to as fail-to-pass test"  │
   │  装不上、跑不起来的，全丢掉                        │
   └───────────────────────────────────────────────────┘
                        ↓
              ★ 90,000 → 2,294 条（留存率 2.5%）
```

```
    🔑 Stage II 的 ②【必须改了测试文件】是整个设计的灵魂。

    ★★ 因为它意味着：验收标准不是评测团队编的，
       是【当年那个真人开发者自己写的】。
       ⟹ 所以这套题没有"出题人偏见"，也几乎不可能出错。
```

---

## 7.4 判卷：两把尺子，不是一把

★★★ 这是本章最实用的知识点。

源码在 [swebench/harness/constants/__init__.py](../源码/swebench-5.0.2/swebench/harness/constants/__init__.py)：

```python
FAIL_TO_PASS = "FAIL_TO_PASS"      # L30
PASS_TO_PASS = "PASS_TO_PASS"      # L32
```

```
   ┌──────────────┬──────────────────────┬────────────────────────┐
   │ 尺子          │ 中文                  │ 它在问什么              │
   ├──────────────┼──────────────────────┼────────────────────────┤
   │ FAIL_TO_PASS │ ★ 修复度（resolution）│ 原来坏的，你修好了吗？  │
   │ PASS_TO_PASS │ ★ 维持度（maintenance）│ 原来好的，你弄坏了吗？  │
   └──────────────┴──────────────────────┴────────────────────────┘

    ⟹ 🔑🔑 一句话：★★★【不但要修好这个 bug，还不能弄坏别的】。

    ★★ 为什么必须有第二把尺子？
       因为只看第一把的话，模型有个作弊捷径：
       把那个函数整个删掉重写成"永远返回测试期望的值"。
       ⟹ 目标测试绿了，其余 100 多个测试全红。
       ⟹ PASS_TO_PASS 就是堵这个洞的。
```

源码里两者被压成 0~1 的比例，[grading.py:288 / :298](../源码/swebench-5.0.2/swebench/harness/grading.py)：
`compute_fail_to_pass` 和 `compute_pass_to_pass`。

然后 [grading.py:309-328](../源码/swebench-5.0.2/swebench/harness/grading.py) 做最终裁定（★ 逐字摘录）：

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

## 7.5 一个特别有教育意义的细节：打补丁本身会失败

模型输出的是一份文本 patch。★★ 但这份 patch 经常【打不进去】——
行号对不上、上下文差一个空格、缩进错了。

[run_evaluation.py:54](../源码/swebench-5.0.2/swebench/harness/run_evaluation.py) 里是这样兜底的（★ 逐字）：

```python
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]
```

```
   ┌───────────────────────────────┬──────────────────────────────┐
   │ 第 1 档  git apply            │ 严格：一个字符不对就拒绝      │
   │ 第 2 档  --3way               │ ★ 三方合并，容忍一点上下文漂移│
   │ 第 3 档  --reject             │ 能打的打，打不了的单独记下来  │
   │ 第 4 档  patch --fuzz=5       │ ★★ 最宽松：允许模糊匹配 5 行 │
   └───────────────────────────────┴──────────────────────────────┘
```

[run_evaluation.py:301](../源码/swebench-5.0.2/swebench/harness/run_evaluation.py) 里逐档尝试：
`for attempt, git_apply_cmd in enumerate(GIT_APPLY_CMDS):`

四档全失败，就打上这个标记（constants/__init__.py:44）：

```python
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
```

```
    🔑🔑 这段代码在讲一个和 05 章【完全同构】的道理：

       ★ 05 章：模型答对了，但正则没抽出来 ⟹ 记成答错
       ★ 07 章：模型改对了，但 patch 格式歪了 ⟹ 记成没解决

    ⟹ ★★★ 换了范式，"格式问题冒充能力问题"这个幽灵还在，
       只是从【抽取层】搬到了【应用层】。

    ⟹ 但 SWE-bench 做了一件对的事：它【给这种失败单独立了一个标记】。
       ⚠️ 而不是像 "[invalid]" 那样，混进"答错"里看不见了。
       ★★ 你自己搭评测时，请学这一点。
```

还有一个同级标记（constants/__init__.py:50）：

```python
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
```

★ 测试卡死也要和"测试失败"分开记。同一个思想。

---

## 7.6 为什么这种评测这么贵

[run_evaluation.py:3](../源码/swebench-5.0.2/swebench/harness/run_evaluation.py) 第一眼就说明了一切：

```python
import docker
```

```
    ★ 名词：Docker（容器）
      一种把"操作系统+依赖库+代码"整个打包冻住的技术。
      ★★ 每道题一个容器，互不干扰，环境完全可复现。
```

同文件里的两个常量：

```python
DOCKER_CLIENT_TIMEOUT   = int(os.environ.get("SWEBENCH_DOCKER_TIMEOUT",   "1800"))
DOCKER_CLIENT_POOL_SIZE = int(os.environ.get("SWEBENCH_DOCKER_POOL_SIZE", "128"))
```

```
    ★ 1800 秒 = 30 分钟。★★【单道题】的超时上限。
    ★ 128 = 容器连接池大小，说明设计上就是要【大规模并发】的。

   ⟹ 把 02 章 §2.4 的成本论断落到实处：
   ┌────────────────┬────────────────┬───────────────────────────┐
   │ 范式            │ 一道题的开销    │ 需要什么                   │
   ├────────────────┼────────────────┼───────────────────────────┤
   │ ① 判别式        │ 1 次前向        │ 一张卡                     │
   │ ② 生成式        │ 几百~几千 token │ 一张卡                     │
   │ ③ agentic       │ ★★ 几十轮对话  │ ★★★ 容器 + 装依赖 + 跑测试│
   │                 │  + 一整个容器   │  + 最多 30 分钟            │
   └────────────────┴────────────────┴───────────────────────────┘

   ⟹ 🔑 这就是为什么 MMLU 每天都能跑，而 SWE-bench 一个季度才跑一次。
```

---

## 7.7 分数的历史：从 1.96% 到 70%+

✅ 论文摘要（2023 年 10 月）：

> "The best-performing model, **Claude 2**, is able to solve a mere **1.96%** of the issues."

✅ 论文 Table 2（BM25 检索，13k 上下文）：

```
    Claude 2        1.96
    SWE-Llama 7b    0.70
    SWE-Llama 13b   0.70
```

```
    ⚠️ 到 2025-2026 年，前沿模型 + 成熟脚手架在 SWE-bench Verified 上
       ★ 已经普遍报到 70% 以上。这是我依据公开报道的概括，
       ⚠️ 不是来自本篇论文，具体数字请以各家发布当天的原始说明为准。

    ★★ 但请注意：1.96% → 70% 这个跨度里，
       【模型变强】和【脚手架变强】是两件事，而榜单把它们混在一起了。
       ⟹ 🔑 这正是下一章要拆的东西。
```

---

## 7.8 为什么会有 "SWE-bench Verified"

```
    ★ 名词：SWE-bench Verified（人工校验子集）
      ★★ 从原始 2,294 条里，由专业工程师逐条人工审核后留下的 500 条。

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

```
   ⟹ ★ 于是同一个名字下至少有三套题：
   ┌──────────────────┬────────┬──────────────────────────────┐
   │ SWE-bench (full) │ 2,294  │ 原始全集                      │
   │ SWE-bench Lite   │   300  │ ✅ 作者抽的"更自足"子集       │
   │ SWE-bench Verified│  500  │ ⚠️ 人工校验子集（后出的）     │
   └──────────────────┴────────┴──────────────────────────────┘
   ★★ 三者分数不可直接比较。
```

---

## 7.9 一个必须警惕的坑：上下文是怎么给的

★★★ 这是我认为新手最容易忽略、但影响最大的一件事。

论文 §4.1 提供了两种给上下文的方式：

```
   ┌─ ① BM25 稀疏检索 ────────────────────────────────┐
   │  ★ 用关键词匹配，从几千个文件里挑几个塞给模型     │
   │  ✅ "in almost half of the instances with the      │
   │      27,000-token limit, it retrieves none of      │
   │      the files from the 'oracle' context"          │
   │  ⟹ 🔑 将近一半的题，★★ 该改的文件根本没给模型看  │
   └──────────────────────────────────────────────────┘

   ┌─ ② "Oracle" 检索（先知设定）─────────────────────┐
   │  ★ 直接把【标准答案改过的那些文件】给模型         │
   │  ✅ 论文自己说："less realistic"                   │
   │  ⟹ ★★ 相当于告诉考生"答案在第 37 页"             │
   └──────────────────────────────────────────────────┘
```

```
    ⚠️ 论文 Table 1 说仓库平均 3,010 个非测试文件、438K 行。
    ⟹ 🔑🔑 所以【怎么找到该改的文件】本身就是这道题的一大半难度。

    ⟹ ★★★ 结论：同一个模型，在 oracle 设定下的分数
       和在"自己去仓库里翻"的设定下的分数，可能差【好几倍】。
       这不是模型的差别，是【题面难度的差别】。

    ⟹ 这正好呼应 01 章的总论点：
       ★★★【分数是"模型+评测方法"的联合产物】。
       在 agentic 评测里，"评测方法"这一项的权重比前面任何一章都大。
```

---

## 7.10 本章总结

```
   ★★ 一图流：

   issue 文字 + 整个仓库
        ↓
   ┌─────────────────┐
   │ 模型/agent 干活  │  ← ★★ 这一格里装什么，07 章没管，08 章专门讲
   └─────────────────┘
        ↓  产出 patch
   ┌─────────────────┐
   │ 起容器、打补丁   │  ← 4 档 fallback；失败→ APPLY_PATCH_FAIL
   └─────────────────┘
        ↓
   ┌─────────────────┐
   │ 跑测试（≤30min）│  ← 卡死 → TESTS_TIMEOUT
   └─────────────────┘
        ↓
   FAIL_TO_PASS=1 且 PASS_TO_PASS=1 ⟹ RESOLVED_FULL
```

```
    🔑 五条带走：

    ① ★★ 判卷靠【真跑测试】，不靠答案比对 ⟹ 05 章的抽取脏活全免
    ② ★★★ 两把尺子：修好了（f2p）+ 没弄坏（p2p），后者零容忍
    ③ ★★ 格式问题依然存在（patch 打不进去），但它被【单独标记】了
    ④ ★★ 贵在容器：每题一个 Docker、超时 30 分钟 ⟹ 季度级而非日级
    ⑤ 🔑🔑 ★★★ full / Lite / Verified 是三套题；oracle / BM25 是两种难度。
       ⟹ 报分数不说清这两件事，等于没报。
```

---

> 下一章：[08-ClaudeCode与Codex怎么被评.md](08-ClaudeCode与Codex怎么被评.md) —— ⭐⭐ 上面那个"模型/agent 干活"的格子里到底是什么，以及为什么"评的其实不只是模型"。
> 返回 [评测入门总目录](README.md)
