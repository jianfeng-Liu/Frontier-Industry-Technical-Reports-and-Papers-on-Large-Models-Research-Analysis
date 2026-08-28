# SWE-bench 拆解

> ★ 版本：`swebench` 5.0.2（源码在 [源码/swebench-5.0.2/](源码/swebench-5.0.2/)）
> ★★ 定位：**agentic 评测**的事实标准。Claude Code、Codex 这类工具的分数就来自它。
>
> ⚠️ 它和 [lm-eval-harness](lm-eval-harness拆解.md) **机制上完全不同**：
> ★★★ 那边是「拼提示词 → 抽答案 → 比字符串」；这边是「起容器 → 打补丁 → 跑测试」。
>
> 标记：✅ = 源码/论文原文可查证 ｜ ⚠️ = 我的推断 ｜ ★ = 重点 ｜ 🔑 = 关键结论

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

---

## 2. ★★★ 判卷：两把尺子，不是一把

★★ 这是整个 SWE-bench 设计里最值得学的一点。

```
   ┌─ 尺子① FAIL_TO_PASS（原本失败 → 应该变通过）───────────┐
   │  ✅ constants/__init__.py:30                            │
   │  ★★ 含义：这个 bug【修好了吗】？                        │
   │  ⟹ 这批测试在打补丁前是失败的，打完应该全过。          │
   └─────────────────────────────────────────────────────────┘
   ┌─ 尺子② PASS_TO_PASS（原本通过 → 必须还通过）───────────┐
   │  ✅ constants/__init__.py:32                            │
   │  ★★★ 含义：你有没有【把别的功能搞坏】？                │
   │  ⟹ 这批测试打补丁前就是过的，打完必须还是过的。        │
   └─────────────────────────────────────────────────────────┘

   ⟹ 🔑🔑 为什么必须有第二把尺子？
      ⚠️ 因为「让这个测试通过」有一个作弊解法：★★★ 把断言删掉。
      ⟹ 只有尺子① 的话，删测试、改配置、注释掉校验都能得分。
      ⟹ ★★ 尺子② 堵死了这条路 —— 你一动别的东西，它立刻报警。
```

### ✅ 判卷逻辑的完整源码（`harness/grading.py:309-328`）

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

✅ 两个比率的算法（`grading.py:288` 和 `:298`）：

```python
def compute_fail_to_pass(report):
    total = len(report[FAIL_TO_PASS]["success"]) + len(report[FAIL_TO_PASS]["failure"])
    if total == 0:
        return 1
    return len(report[FAIL_TO_PASS]["success"]) / total
```

✅ 三种结果的枚举（`constants/__init__.py:6-9`）：

```python
class ResolvedStatus(Enum):
    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"
```

```
   ⟹ 🔑🔑🔑 请注意上面代码里【最狠的一个细节】：

   ★★★ 三个分支里，p2p 的条件全部是 `p2p == 1`，一个都不放松。

   ⟹ 换句话说：⚠️【只要弄坏了一个原本通过的测试，
      不管 bug 修得多漂亮，直接判 NO。】
   ⟹ ★★ 而 f2p 是允许部分的（PARTIAL 那一档）。

   ┌────────────────────────────────────────────────────┐
   │  ★★★ 这个不对称设计翻译成人话就是：                │
   │  「修不完可以商量，弄坏了没得商量。」               │
   │                                                     │
   │  ⚠️ 这恰好也是真实软件工程的规矩 ——                │
   │     ★★ 回归（regression）比未完成更不可接受。      │
   └────────────────────────────────────────────────────┘

   ⚠️ 补充一个读榜单时的注意点：
   ★★ 榜单上报的 "% Resolved" 通常只算 FULL，PARTIAL 不计分。
```

---

## 3. ★★★ 一个源码事实：harness 根本不知道 agent 长什么样

★★ 这一节直接回答"Claude Code 是怎么被评的"。

✅ `harness/run_evaluation.py:699` 的 `main()` 签名，输入里**只有一个文件路径**：

```python
def main(
    dataset_name: str,
    split: str,
    instance_ids: list,
    predictions_path: str,          # ★★★ 就是它
    max_workers: int,
    ...
)
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

   ⟹ 🔑🔑 所以：★★★【SWE-bench 分数评的是「模型 + 脚手架」这一整套系统，
      而榜单通常只写模型的名字。】
```

### ✅ 这个结论有正反两面的实证

```
   ┌─ 正面（SWE-agent 论文 Table 1，✅ 原文数值）───────────┐
   │  ★★★ 同一个 GPT-4 Turbo，在 SWE-bench Lite 上：       │
   │     2.67%  ← RAG（检索增强，非交互）                   │
   │     7.33%  ← 裸 shell，不给示范                        │
   │    11.00%  ← 裸 shell + 示范                           │
   │    18.00%  ← SWE-agent 完整 ACI                        │
   │  ⚠️ 权重一个字节没改，分数差 6.7 倍。                  │
   └────────────────────────────────────────────────────────┘
   ┌─ 反面（DeepSeek-V3 报告，✅ 原文）─────────────────────┐
   │  "SWE-Bench verified is evaluated using the            │
   │   **agentless framework**"                             │
   │  ⟹ ★★ 一线团队自己也知道必须披露脚手架。              │
   └────────────────────────────────────────────────────────┘

   ⟹ 详见 [08 章 §8.4.1](评测入门/08-ClaudeCode与Codex怎么被评.md)。
```

---

## 4. 执行流水线：一道题是怎么跑完的

```
   ┌─ ① 读预测文件 ────────────────────────────────────────┐
   │  ✅ get_dataset_from_preds()  run_evaluation.py:542    │
   │  ⟹ 把 JSONL 里的补丁和数据集里的题目对齐              │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ② 起 Docker 容器 ────────────────────────────────────┐
   │  ✅ import docker            run_evaluation.py:3       │
   │  ⟹ ★★★ 每道题一个独立容器，装好该仓库该版本的全部依赖│
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ③ ★★ 把补丁写进去并尝试应用 ─────────────────────────┐
   │  ✅ run_evaluation.py:291                              │
   │     patch_file.write_text(pred["model_patch"] or "")   │
   │  ✅ run_evaluation.py:301                              │
   │     for attempt, git_apply_cmd in enumerate(GIT_APPLY_CMDS):│
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ④ 跑测试，抓日志 ────────────────────────────────────┐
   │  ✅ get_logs_eval()          grading.py:113            │
   │  ⟹ 用 log_parsers/ 里对应语言的解析器读测试输出       │
   └───────────────────────────────────────────────────────┘
                            ▼
   ┌─ ⑤ 判卷 ──────────────────────────────────────────────┐
   │  ✅ get_eval_report()        grading.py:329            │
   │  ✅ get_resolution_status()  grading.py:309            │
   └───────────────────────────────────────────────────────┘
```

---

## 5. ★★ 一个特别有教育意义的细节：打补丁本身会失败

✅ `run_evaluation.py:54` 的完整原文：

```python
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
]
```

```
   ⚠️ 逐条解释这四级"容错阶梯"（越往下越宽松）：

   ① git apply --verbose
      ⟹ ★ 严格模式：上下文必须一字不差地对上。

   ② git apply --verbose --3way
      ⟹ ★★ 三方合并：允许用 git 的历史信息去推断该打在哪。

   ③ git apply --verbose --reject
      ⟹ ★★ 能打的部分先打上，打不上的丢到 .rej 文件里。

   ④ patch --batch --forward --fuzz=5 -p1 -i
      ⟹ ★★★ 换成古老的 patch 工具，fuzz=5 表示
         【上下文允许有 5 行对不上】也照打。

   ⟹ ✅ 源码在 L301 是一个循环：前一个失败就试下一个。
```

✅ 全部失败时的标记（`constants/__init__.py:44`）：

```python
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
```

✅ 超时的标记（`constants/__init__.py:50`）：

```python
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
```

```
   ⟹ 🔑🔑 这个设计和 05 章的"抽取失败"是【同构】的：

   ┌────────────────┬──────────────────────┬──────────────────────┐
   │                │ lm-eval-harness      │ SWE-bench            │
   ├────────────────┼──────────────────────┼──────────────────────┤
   │ 模型答对了但   │ 抽取失败             │ 补丁应用失败          │
   │ 格式不合规     │ → "[invalid]"        │ → APPLY_PATCH_FAIL   │
   │ 结果           │ ★★ 判错             │ ★★ 判错             │
   └────────────────┴──────────────────────┴──────────────────────┘

   ⟹ ★★★ 两个框架都在做同一件事：
      【把"格式问题"和"能力问题"在最终分数上混成一个数】。

   ⚠️ 但 SWE-bench 做得比 lm_eval 更好的一点：
      ★★ 它给了 APPLY_PATCH_FAIL 一个【专门的标记】，
      ⟹ 所以你翻日志能分清"没修对"和"补丁根本没打上去"。
      🔑 这正是 05 章希望 lm_eval 也能做到的事。
```

```
   ⟹ 🔑 排查建议：★★★ 自己跑 SWE-bench 时，
      先统计日志里 APPLY_PATCH_FAIL 的比例。
      ⚠️ 比例高 ⟹ 你的脚手架输出 diff 的格式有问题，
         【不是模型不会修 bug】。
```

---

## 6. 为什么这种评测这么贵

✅ 源码里的成本证据（`run_evaluation.py:61-68`）：

```python
DOCKER_CLIENT_TIMEOUT   = int(os.environ.get("SWEBENCH_DOCKER_TIMEOUT", "1800"))
DOCKER_CLIENT_POOL_SIZE = int(os.environ.get("SWEBENCH_DOCKER_POOL_SIZE", "128"))
```

```
   ⟹ ⚠️ 从这两个默认值能读出很多东西：

   ★★ 单个容器操作的超时是 1800 秒 = 30 分钟。
      ⟹ 说明【一道题跑几十分钟是正常的】。
   ★★ 连接池默认 128。
      ⟹ 说明设计上就预期【上百个容器并发】。

   ⟹ 🔑 成本三件套：
      ① ★★★ 镜像：12 个仓库 × 多个历史版本，拉下来【几十 GB】
      ② ★★  时间：2,294 道题 × 每题分钟级
      ③ ★★★ 模型调用：agentic 是多轮的，一道题几十轮对话
         ✅ SWE-agent 论文 Table 1：SWE-bench Lite 单题平均 $1.67

   ⟹ 🔑🔑 对照 02 章那条定律：
      ★★★【越真实 ⟺ 越贵 ⟺ 越难复现】。
      ⚠️ MMLU 一道题的成本是它的几万分之一。
```

---

## 7. ⚠️⚠️ 三个题集，绝对不能混着比

```
   ┌──────────────┬────────┬────────────────────────────────┐
   │ 名字         │ 题数    │ 说明                            │
   ├──────────────┼────────┼────────────────────────────────┤
   │ SWE-bench    │ 2,294  │ ✅ 全量，论文原始版本            │
   │ Lite         │   300  │ ✅ 论文给出的精简子集            │
   │ Verified     │   500  │ ⚠️ 人工核验过"题目本身可解"的子集│
   └──────────────┴────────┴────────────────────────────────┘

   ⚠️ ★★★ Verified 上的分数【系统性高于】全量，
      因为它剔除了描述不清、信息不足、根本做不了的题。

   ⟹ 🔑 看到 "SWE-bench 42%" 这样一个数，
      ★★★ 必须先问是哪个子集。⚠️ 差别可以是十几个点。
```

✅ 源码里还有第四个变体（`run_evaluation.py:716`）：SWE-bench Multimodal 有专门分支。

---

## 8. ⚠️ 一个必须警惕的坑：上下文是怎么给的

★★ 模型不可能把整个仓库塞进上下文。所以必须先【挑出可能相关的文件】。

✅ 论文原文（关于 BM25 检索基线）：

> "We observe that in approximately **40% of instances**, BM25 retrieves a superset
> of the oracle files for the 27,000-token context limit. However, in **almost half
> of the instances** with the 27,000-token limit, it retrieves **none of the files**
> from the 'oracle' context."

```
   ★ 名词：
   ★ BM25 = 一种经典的关键词检索算法（不用神经网络）
   ★★ oracle context（先知上下文）= 官方修复实际改动的那些文件
      ⟹ 相当于"标准答案涉及哪几个文件"

   ⟹ 🔑🔑 论文这句话的意思是：
      ★★★ 用 BM25 检索时，接近一半的题目里，
      ⚠️【真正该改的文件一个都没被检索出来】。
      ★ （✅ 论文这句限定在 27,000 token 的上下文预算下）
      ⟹ 这种情况下模型再强也做不对 —— 它压根没看到该改的代码。

   ⟹ 🔑 所以：★★★ "怎么给上下文"本身就是脚手架的一部分，
      而且是【影响最大的那一部分】。
      ⚠️ 这也解释了 §3 里 RAG 2.67% vs SWE-agent 18.00% 的巨大差距 ——
         SWE-agent 让模型【自己去搜、自己去翻】，而不是提前替它决定看什么。
```

---

## 9. 如果你想自己跑一次

```bash
# ① 装
pip install swebench

# ② 准备预测文件 preds.jsonl，每行：
# {"instance_id": "...", "model_patch": "<diff 文本>",
#  "model_name_or_path": "my-agent-v1"}

# ③ 跑（先只跑几道题）
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path ./preds.jsonl \
    --max_workers 4 \
    --run_id my-first-run \
    --instance_ids <id1> <id2>
```

```
   ⚠️ 三个新手一定会踩的坑：

   ① ★★★ 必须让 agent 在【base commit】上工作。
      ⚠️ 直接 clone 主分支 = 拿到的是已修复的代码 = 分数假到离谱。

   ② ★★★ 绝对不能让 agent 看到 FAIL_TO_PASS 的测试内容。
      ⚠️ 那等于把答案给它。
      ⟹ 🔑 更隐蔽的版本：跑 N 次然后【用 f2p 测试挑最好的那次】——
         ★★★ 这也是看答案，只是换了个姿势。

   ③ ★★ Docker 镜像会拉几十 GB。⚠️ 先确认磁盘和网络。
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
      ★★★ 由于 harness 只收补丁不问来源，
      【分数从此变成了「模型 + 脚手架」的联合产物】，
      ⟹ 而这一点，榜单上是看不出来的。
```

---

> 相关：[评测入门 07 agentic 编码评测](评测入门/07-agentic编码评测与SWE-bench.md) ｜ [08 Claude Code 和 Codex 怎么被评](评测入门/08-ClaudeCode与Codex怎么被评.md)
> 返回 [harness评测框架/](README.md)
