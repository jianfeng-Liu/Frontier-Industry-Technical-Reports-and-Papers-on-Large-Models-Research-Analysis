# 07 · DeepSeek 的脚手架：一份意外详细的披露

> ⭐⭐ 本章一手来源：**DeepSeek-V4 技术报告**
> 本地存档 [`业界模型技术报告/papers/arxiv-2606.19348.pdf`](../../业界模型技术报告/papers/arxiv-2606.19348.pdf)（58 页），
> 引用集中在 **§5.2.5 沙箱基础设施** 和 **§5.3.1 评测设置**。
>
> ★★★ 一句话预告：**这一章是全目录唯一一处，
> 你能同时看到【脚手架配置】和【它跑出来的分数】的地方。**

---

## 7.0 为什么这章值得单独写

```
   ⟹ 回顾一下 README 里的三级可查证度：
      ① SWE-agent  ⟹ 论文 + 源码，全可查
      ② Claude Code / Codex ⟹ 行为可查、实现不可查
      ③ DeepSeek   ⟹ 我原本以为"基本查不到"

   ⚠️ 我错了。★★★ V4 报告披露的脚手架细节，
      比 Claude Code 和 Codex 加起来还具体——
      因为它写的是【跑分那一次到底怎么设的】。

   ⟹ 🔑 ★★ 这是一个结构性差异：
      产品文档写的是"你可以怎么配"；
      技术报告写的是"我们那次是怎么配的"。
      ⟹ ★★★ 后者才是【复现分数】需要的东西。
```

⚠️ 但它也有它查不到的地方，先说在前面：

```
   ❌ 这套内部框架【不开源】，没有仓库可读。
   ❌ 提示词模板、工具的具体参数签名，报告里没有。
   ⟹ 所以本章能给你的是【规格参数】，不是【实现】。
```

---

## 7.1 ★★★ 核心披露：跑 SWE-bench 用的是自研框架

✅ 报告 §5.3.1 原文（★ 逐字）：

> "For code agent tasks (SWE-Verified, Terminal-Bench, SWE-Pro, SWE Multilingual),
> we evaluate DeepSeek-V4 series using an internally developed evaluation framework.
> This framework provides a minimal set of tools — a bash tool and a file-edit tool.
> The maximum number of interaction steps is set to 500, and the maximum context
> length is set to 512K tokens."

⟹ **对于代码 agent 任务（SWE-Verified、Terminal-Bench、SWE-Pro、SWE 多语言版），我们使用一个内部开发的评测框架来评测 DeepSeek-V4 系列。该框架提供一套极简工具集 —— 一个 bash 工具和一个文件编辑工具。最大交互步数设为 500，最大上下文长度设为 512K token。**

```
   ⟹ 🔑🔑🔑 ★★★ 拆开看，这一句话给了【四个具体参数】：

   ┌────────────────────┬──────────────────────────────────┐
   │ ① 工具集           │ ★★★ 只有 2 个：bash + 文件编辑   │
   │ ② 最大交互步数     │ 500                              │
   │ ③ 最大上下文       │ 512K token                       │
   │ ④ 框架归属         │ ⚠️ 自研，不开源                  │
   └────────────────────┴──────────────────────────────────┘

   ⟹ ★★ 这四个数，正好对应本目录前四章讲的四件事：
      ①→03 章工具集   ②→02 章退出条件
      ③→04 章上下文   ④→本章
```

---

## 7.2 ★★★ 第一个反差：只有两个工具

★★ 请把这一行，和 03 章那张表放在一起看：

```
   ┌───────────────┬────────────────────────────────────────┐
   │ SWE-agent     │ 9 个专门设计的工具                     │
   │ （2024 论文） │ find_file / search_file / search_dir /  │
   │               │ open / scroll_down / scroll_up / goto / │
   │               │ create / edit                          │
   ├───────────────┼────────────────────────────────────────┤
   │ Claude Code   │ 十几个内置工具（Read/Edit/Bash/Glob/   │
   │ （2026 产品） │ Grep/Task/WebFetch…）                   │
   ├───────────────┼────────────────────────────────────────┤
   │ DeepSeek-V4   │ ★★★ 2 个：bash + 文件编辑              │
   │ （2026 跑分） │ ✅ 报告自称 "a minimal set of tools"    │
   └───────────────┴────────────────────────────────────────┘
```

```
   ⟹ 🔑🔑 ★★★ 这看起来像"倒退"，但它其实是【立场】：

   ★★ SWE-agent 的 ACI 论点是：
      "模型不会用人类界面，所以要给它专门设计的界面。"
      ⟹ 03 章那个 7.7 分的消融就是这么来的。

   ★★ DeepSeek 这里的隐含论点相反：
      "给它 bash 就行，剩下的它自己会。"

   ⟹ ★★★ 谁对？我认为【两个都对，但说的是不同年份的模型】：
      ⚠️ 以下是我的推断，报告没有这样说 ——
      2024 年的 GPT-4 Turbo 需要脚手架把 bash 包装成 9 个安全工具；
      2026 年的模型，本身就是在【大量 bash 轨迹上训出来的】。
      ⟹ 能力从【脚手架】搬进了【模型权重】。
```

★★ 有一条旁证，就在同一份报告里（✅ 逐字）：

> "for each target domain — such as mathematics, coding, agent, and instruction
> following — a separate expert model is trained independently."

⟹ **对每个目标领域 —— 例如数学、编程、agent、指令遵循 —— 都单独训练一个专家模型。**

```
   ⟹ 🔑 ★★★ 注意 "agent" 被列为一个和数学、编程并列的【训练领域】。
      ⟹ ★★ 也就是说：【"会用工具"是被专门训出来的一项能力】，
         不再是靠脚手架临时兜住的。
      ⟹ 这直接解释了为什么 2026 年的框架敢只给 2 个工具。
```

---

## 7.3 ★★ 第二个反差：500 步、512K 上下文

★★ 把这两个数和 SWE-agent 论文对比：

| | SWE-agent（2024） | DeepSeek-V4（2026） |
|---|---|---|
| 上下文窗口 | ✅ GPT-4 Turbo **128K** / Claude 3 Opus 200K | ✅ **512K** |
| 步数上限 | ⚠️ 论文的成本上限约束下，实测成功案例约 **12 步**、失败约 21 步 | ✅ **500 步** |
| 上下文策略 | ✅ 只留最近 5 条观察 | ⚠️ 代码 agent 任务未说明；★★ 但**搜索** agent 任务明确说了（见下） |

```
   ⟹ 🔑🔑 ★★★ 500 步 vs 12 步 —— 这不是同一个数量级的东西。

   ⟹ ★★ 回想 02 章那条 "Agents succeed quickly and fail slowly"
      （agent 成功得快，失败得慢）：
      2024 年成功案例平均 12 步，所以步数上限设 20~30 就够了。
   ⟹ ★★★ 把上限抬到 500，意味着设计者认为
      【有一类任务本来就需要几百步才能做完】。
      ⚠️ 这是我的推断。但报告里的 Terminal-Bench（终端基准）
         确实是这类长任务。
```

**⭐ 关于上下文策略，报告在【搜索 agent】那一段给了明确答案**（✅ 逐字）：

> "For search agent tasks (BrowseComp, HLE w/ tool), we also use an in-house harness
> with websearch and Python tool, and set maximum interaction steps to 500 and the
> maximum context length to 512K tokens. For BrowseComp, we use the same
> discard-all context management strategy as DeepSeek-V3.2."

⟹ **对于搜索 agent 任务（BrowseComp、带工具的 HLE），我们同样使用一个内部 harness，配备网页搜索和 Python 工具，最大交互步数设为 500，最大上下文长度设为 512K token。对于 BrowseComp，我们使用与 DeepSeek-V3.2 相同的「全部丢弃」上下文管理策略。**

```
   ★ 名词：discard-all（全部丢弃）
     ⟹ ★★★ 04 章三种策略里【最激进的截断】：
        不是折叠成一行，是直接扔。

   ⟹ 🔑 ★★ 注意这里用词是 "harness"——
      ⚠️ 这正是 01 章讲的那个同名歧义：
      报告里这个 harness 指的是【脚手架】，不是【评测框架】。
      ⟹ ★★★ 同一份报告里，两个意思都出现了，靠上下文区分。
```

---

## 7.4 ⭐⭐ 第三个披露：脚手架的写法，能让模型的特性【失效】

★★★ 这是我在整份报告里认为**最重要**的一段，因为它是唯一一处
明确写出"脚手架设计错了会怎样"的地方。

先看 V4 的一个模型侧特性（✅ 逐字，§ Interleaved Thinking）：

> "Tool-Calling Scenarios. … all reasoning content is fully preserved throughout the
> entire conversation. Unlike DeepSeek-V3.2, which discarded thinking traces upon
> each new user turn, DeepSeek-V4 series retain the complete reasoning history across
> all rounds, including across user message boundaries."

⟹ **工具调用场景：全部推理内容在整段对话中被完整保留。与 DeepSeek-V3.2 在每个新用户回合到来时丢弃思考轨迹不同，DeepSeek-V4 系列跨所有轮次保留完整的推理历史，包括跨越用户消息边界。**

```
   ★ 名词：interleaved thinking（交错思考）
     ⟹ 模型的"思考过程"和"工具调用"交替进行，
        且思考内容在后续轮次里仍然可见。
   ★ 名词：thinking trace / reasoning trace（思考轨迹）
     ⟹ 模型在给出动作之前那段"内心独白"。
     ⟹ ★★ 保留它 = 模型记得"我上一步为什么那么想"。
```

★★★ 然后是那句关键的警告（✅ 逐字）：

> "agent frameworks that simulate tool interactions via user messages (e.g., Terminus)
> may not trigger the tool-calling context path and thus may not benefit from enhanced
> reasoning persistence. We continue to recommend non-think models for such
> architectures."

⟹ **那些通过「用户消息」来模拟工具交互的 agent 框架（例如 Terminus），可能不会触发工具调用的上下文路径，因而可能享受不到增强的推理持久性。对这类架构，我们仍然建议使用非思考模型。**

```
   ⟹ 🔑🔑🔑 ★★★ 请务必读懂这一段。它在说：

   ┌──────────────────────────────────────────────────────────┐
   │ 同一个模型，同一道题，                                    │
   │ 只因为脚手架【把工具结果塞在哪个字段里】不一样——         │
   │                                                          │
   │   写法 A：作为 tool result（工具结果）回传                │
   │      ⟹ ✅ 触发工具调用路径 ⟹ 思考轨迹保留 ⟹ 更强        │
   │                                                          │
   │   写法 B：伪装成 user message（用户消息）回传             │
   │      ⟹ ❌ 不触发 ⟹ 思考轨迹被丢弃 ⟹ 更弱                │
   │      ⟹ ★★★ 官方甚至建议这种情况【别用思考模型】          │
   └──────────────────────────────────────────────────────────┘

   ⟹ ★★ 回到 02 章讲的循环：
      我在那一章说"工具结果被拼回对话，作为一条新消息"。
      ⟹ 🔑 现在你知道了：【这条消息挂在哪个角色下面，是有后果的。】

   ⟹ ★★★ 这是全目录最硬的一条证据，证明本目录的立论：
      【脚手架不是模型的包装纸，它是分数的一部分。】
```

⚠️ 一句提醒：`Terminus` 是报告点名的一个第三方 agent 框架。我没有独立核实
它的实现，只转述报告的说法。

---

## 7.5 ⭐ 第四个披露：DSec —— 一整套沙箱基础设施

★★ 报告 §5.2.5 用了整整一节讲沙箱平台。✅ 逐字：

> "we build a production-grade sandbox platform, DeepSeek Elastic Compute (DSec).
> DSec comprises three Rust components — the API gateway (Apiserver), per-host agent
> (Edge), and the cluster monitor (Watcher) … In production, a single DSec cluster
> manages hundreds of thousands of concurrent sandbox instances."

⟹ **我们构建了一个生产级沙箱平台 DeepSeek Elastic Compute（DSec）。DSec 由三个 Rust 组件构成 —— API 网关（Apiserver）、每主机代理（Edge）和集群监视器（Watcher）……在生产环境中，单个 DSec 集群管理数十万个并发沙箱实例。**

```
   ⟹ ★★ 顺带一提：DeepSeek 用 Rust 这件事是 ✅ 报告明写的。
      ⚠️ 对照 06 章：Codex 用 Rust 我反而【没能】一手确认。
```

**✅ 四种执行基底（逐字对照）：**

| 基底 | 报告原文 ⟹ 中文 | ★ 隔离强度 |
|---|---|---|
| **Function Call** | "dispatches stateless invocations to a pre-warmed container pool, eliminating cold-start overhead" ⟹ 把无状态调用派发到预热容器池，消除冷启动开销 | 最弱、最快 |
| **Container** | "fully Docker-compatible" ⟹ 完全兼容 Docker | ★★ 与 SWE-agent 同级 |
| **microVM** | 基于 Firecracker，"adds VM-level isolation for security-sensitive, high-density deployments" ⟹ 为安全敏感、高密度部署提供虚拟机级隔离 | ★★★ 更强 |
| **fullVM** | 基于 QEMU，"supports arbitrary guest operating systems" ⟹ 支持任意客户操作系统 | 最强、最重 |

✅ 且四者接口统一：

> "All four share a common API surface — command execution, file transfer, and TTY
> access — and switching between them requires only a parameter change."

⟹ **四者共享同一套 API —— 命令执行、文件传输、TTY 访问 —— 在它们之间切换只需要改一个参数。**

```
   ⟹ 🔑🔑 ★★★ 请回想 05 章那句 SWE-agent 论文的坦白：
      "not considered as secure as virtualized hardware isolation"
      ⟹【不如硬件虚拟化隔离安全。】

   ⟹ ★★ DSec 的 microVM / fullVM 两档，正是把那个"不如"补上了。
      ⟹ 两年后，学术论文承认的缺口，在工业界成了一个【可选参数】。
```

**★★★ 还有一条我认为很关键的设计 —— 轨迹日志与确定性重放**（✅ 逐字）：

> "DSec maintains a globally ordered trajectory log for each sandbox, persistently
> recording every command invocation and its results."

⟹ **DSec 为每个沙箱维护一份全局有序的轨迹日志，持久记录每一次命令调用及其结果。**

报告说这份日志有三个用途，第三个是：

> "deterministic replay — any historical session can be faithfully reproduced from its
> trajectory."

⟹ **确定性重放 —— 任何历史会话都能从它的轨迹被忠实复现。**

```
   ⟹ 🔑🔑 ★★★ 这直接回答了【复现性】这个老问题。

   ⟹ ★★ 对照 05 章：SWE-agent 用"一次性容器"保证【起点相同】；
      DSec 用"轨迹日志"保证【整个过程可回放】。
      ⟹ 前者能复现【环境】，后者能复现【那一次运行】。

   ⚠️ 但注意：这是 DeepSeek 的【内部】能力。
      ⟹ ★★★ 报告没有公开任何一条轨迹，
         所以外部的人仍然无法复现他们的分数。
      ⟹ 🔑 能复现 ≠ 已公开。
```

---

## 7.6 那这套框架跑出来是多少分？

✅ 报告 Table 6，Agentic（agent 能力）一段，**SWE Verified — Resolved（解决率 %）**：

| Opus-4.6 Max | GPT-5.4 xHigh | Gemini-3.1-Pro High | K2.6 Thinking | GLM-5.1 Thinking | **DS-V4-Pro Max** |
|---|---|---|---|---|---|
| **80.8** | — | 80.6 | 80.2 | — | **80.6** |

同表 **Terminal Bench 2.0 — Acc（准确率 %）**：

| Opus-4.6 | GPT-5.4 | Gemini-3.1-Pro | K2.6 | GLM-5.1 | **DS-V4-Pro** |
|---|---|---|---|---|---|
| 65.4 | **75.1** | 68.5 | 66.7 | 63.5 | 67.9 |

```
   ⚠️⚠️ ★★★ 但这张表【必须】配一句警告才能读：

   ✅ 报告只说了【DeepSeek 自己】用的是内部框架、2 个工具、
      500 步、512K 上下文。
   ⚠️ 它【没有说】Opus-4.6 那个 80.8 是在什么脚手架下跑的。

   ⟹ 🔑 ★★★ 所以这一栏的可比性，取决于一个报告没交代的前提：
      【别家的数字，是他们自己报的，还是 DeepSeek 用同一套框架重跑的？】

   ⟹ ★★ 对照一下：报告在【长上下文】那一段是明说了的（✅ 逐字）——
      "We re-evaluate Claude Opus 4.6 and Gemini 3.1 Pro on these tasks with the
       goal of standardizing the configuration across all models."
      ⟹【我们在这些任务上重新评测了 Claude Opus 4.6 和 Gemini 3.1 Pro，
         目的是统一所有模型的配置。】
   ⟹ ★★★ 也就是说：长上下文那一段【明确声明了统一口径】，
      而 agent 那一段【没有同样的声明】。
      ⚠️ 这个差别，我认为读者应该自己去原文确认，而不是替它假设。
```

---

## 7.7 ⭐ 报告自己回应了"内部框架 = 自己给自己出题"的质疑

★★★ 我认为这是这份报告在方法论上最有意思的一手（✅ 逐字）：

> "It is worth noting that DeepSeek-V4-Pro performs well on MCP Atlas and Toolathlon
> — two evaluation test sets that include a wide range of tools and MCP services —
> indicating that our model has excellent generalization capability and does not
> perform well only on internal frameworks."

⟹ **值得注意的是，DeepSeek-V4-Pro 在 MCP Atlas 和 Toolathlon 上表现良好 —— 这两个测试集包含大量工具和 MCP 服务 —— 这表明我们的模型具有出色的泛化能力，而不是只在内部框架上表现好。**

```
   ⟹ 🔑🔑 ★★★ 请注意这句话的结构，它是一个【自辩】：

   ① 潜在质疑：★★"你用自研框架跑分，是不是模型被调成
      只会用你那两个工具了？"
   ② 报告的回应："我们在别人的、工具很多的基准上也不差。"

   ⟹ ★★ 我认为这个回应【方向是对的】：
      在陌生工具集上仍然强，确实是泛化的证据。

   ⚠️ 但它不是【完整的】证据。真正干净的做法应该是：
      ★★★【用同一套公开脚手架（比如开源的 SWE-agent），
         把自己和对手都重跑一遍。】
      ⟹ 这样才能把"模型强"和"脚手架配得好"分开。
   ⟹ ⚠️ 这是我的评论，不是报告的表述。
```

★★ 顺带，报告还主动承认了一个环境问题（✅ 逐字，很少见）：

> "Regarding Terminal-Bench 2.0, we acknowledge the environment-related issues noted
> by GLM-5.1. Nevertheless, we report our performance on the original Terminal-Bench
> 2.0 dataset for consistency. On the Terminal-Bench 2.0 Verified subset,
> DeepSeek-V4-Pro achieves a score of approximately 72.0."

⟹ **关于 Terminal-Bench 2.0，我们承认 GLM-5.1 指出的环境相关问题。尽管如此，为保持一致性，我们仍在原始 Terminal-Bench 2.0 数据集上报告成绩。在 Terminal-Bench 2.0 Verified 子集上，DeepSeek-V4-Pro 得分约为 72.0。**

```
   ⟹ ★★ 67.9（原始集）vs 约 72.0（Verified 子集）。
      🔑 ★★★ 同一个模型、同一次运行，换个子集差 4 分。
      ⟹ 这就是"口径"两个字的重量。
   ⟹ ★★ 而且报告选择报【低的那个】以保持一致性——
      这一点我认为值得肯定。
```

---

## 7.8 和 [评测入门 10 章](../评测入门/10-DeepSeek的评测口径.md) 的分工

```
   ┌──────────────────────────────────────────────────────────┐
   │ 评测入门 10 章                                            │
   │   来源：DeepSeek-V3 / R1 报告                             │
   │   角度：★【评测框架】—— 温度、few-shot、输出长度、       │
   │        去污染、greedy vs sampling                         │
   │   关于 SWE-bench 的结论：                                 │
   │     ✅ "SWE-Bench verified is evaluated using the         │
   │        agentless framework"                               │
   │     ⟹ V3 的 42.0 是【V3 + Agentless】的分数              │
   ├──────────────────────────────────────────────────────────┤
   │ 本章（agent脚手架 07）                                     │
   │   来源：DeepSeek-V4 报告                                  │
   │   角度：★★★【agent 脚手架】—— 工具数、步数、上下文、    │
   │        沙箱基底、思考轨迹保留                             │
   │   关于 SWE-bench 的结论：                                 │
   │     ✅ V4 换成了【自研 code agent 框架】，2 工具/500 步    │
   └──────────────────────────────────────────────────────────┘

   ⟹ 🔑🔑 ★★★ 把两章接起来，你会看到一条演进线：

      V3（Agentless，不用 agent 循环）  ⟹ 42.0
      V4（自研 agent 框架，500 步）     ⟹ 80.6

   ⚠️⚠️ 但【绝对不能】把这两个数直接相减说"进步了 38.6 分"：
      ① 模型换了（V3 → V4）
      ② 脚手架换了（Agentless → 自研 agent）
      ③ ⚠️ 题目集是否完全一致，需要回原文核对
   ⟹ ★★★ 三个变量同时变了，这个差值【不归因于任何单一原因】。
   ⟹ 🔑 这正是 08 章要讲的核心问题。
```

---

## 7.9 本章总结

```
    🔑 七条带走：

    ① ★★★ ✅ V4 报告披露了跑分那次的四个参数：
       只有 2 个工具（bash + 文件编辑）、500 步上限、
       512K 上下文、内部自研框架（⚠️ 不开源）。

    ② ★★★ 2 个工具 vs SWE-agent 的 9 个 ——
       ⚠️ 我的解读：能力从【脚手架】搬进了【模型权重】。
       ✅ 旁证：报告把 "agent" 列为一个独立的训练领域。

    ③ ★★ 500 步 vs 2024 年的约 12 步，不是一个数量级。

    ④ ★★★ 🔑 全目录最硬的一条证据：
       ✅ 工具结果如果被脚手架"伪装成用户消息"回传，
       模型的思考轨迹保留特性【不触发】，官方甚至建议
       这种情况别用思考模型。
       ⟹【脚手架把结果挂在哪个角色下，是有分数后果的。】

    ⑤ ★★ ✅ DSec：四种执行基底（函数调用/容器/microVM/fullVM），
       统一 API，切换只改一个参数。
       ⟹ 补上了 SWE-agent 论文承认的"不如硬件虚拟化"那个缺口。

    ⑥ ★★★ ✅ 轨迹日志支持"确定性重放"。
       ⚠️ 但轨迹没公开 ⟹ 【他们能复现，你不能】。

    ⑦ ⚠️ ★★★ Table 6 的横向对比，报告在【长上下文】那段声明了
       统一口径，在【agent】那段没有同样的声明。
       ⟹ 读这张表时，这个差别必须自己去原文确认。
```

---

> 上一章：[06 Claude Code 与 Codex 逐项对照](06-ClaudeCode与Codex逐项对照.md) ｜ 下一章：[08 脚手架怎么影响榜单分数](08-脚手架怎么影响榜单分数.md)
