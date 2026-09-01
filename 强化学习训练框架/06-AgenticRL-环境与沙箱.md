# 06 · Agentic RL：环境与沙箱

> **这一篇干什么**：讲 RL 训练里「模型之外」的那一半——环境、工具、沙箱。
> **难度** ★★ · **前置**：[02-架构主轴](02-架构主轴-同步到全异步.md)
> **★ 这一篇会挂回 [harness评测框架/agent脚手架/](../harness评测框架/agent脚手架/)**——评测用的 harness 和训练用的 environment，是同一件东西的两面。
> **本篇 🌐 材料统一出处**：[Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape)（下称「HF 横评」），访问于 2026-08，以官方仓库现状为准。

---

## 6.1 Agentic RL 和普通 RL 差在哪

普通 RL 的一次 rollout：

```
题目 ──► 模型生成答案 ──► 打分
```

Agentic RL 的一次 rollout：

```
题目 ──► 模型想 ──► 调工具 ──► 环境执行 ──► 返回结果 ──► 模型再想 ──► ... ──► 打分
                       ▲                        │
                       └────── 几十到几百轮 ─────┘
```

差别不是「多轮」这么简单，而是三件事同时变了：

| | 普通 RL | Agentic RL |
|---|---|---|
| 一条轨迹多长 | 几千 token | ✅ MiniMax：**最长 192K token**；✅ Kimi-K3：**1M 上下文** |
| 谁在生成 | 只有模型 | 模型 + **环境**（环境返回的内容也在轨迹里，但不该算模型的梯度） |
| 生成期间在干嘛 | GPU 满载 | ✅ Kimi：等模型推理结果的时间**可占沙箱生命周期的 98%**；反过来说，agent 大量时间在等工具执行 |

⚠️ 第三条是整章的关键：**Agentic RL 里，长尾不再只是「有的答案长」，而是「有的工具跑得慢」。** 而工具执行是 CPU/IO 负载，不占 GPU——这意味着一个纯 GPU 视角的调度器根本看不见这部分延迟。

---

## 6.2 ✅ 环境该放在架构的哪一层：GLM 给了明确答案

✅ GLM-5 §4.1.1 的 Multi-Task Rollout Orchestrator（多任务 rollout 编排器）设计：

> *"we standardize trajectories from all agentic tasks into a **unified message-list representation**. This enables joint training of complex agentic frameworks (e.g., Software Engineering task) while also supporting centralized post-processing and logging for heterogeneous workloads. This design **cleanly isolates task-specific logic from the core training loop**, enabling seamless integration with multi-task RL training."*

拆开看：

| 设计 | 意思 |
|---|---|
| 每个任务 = 独立微服务，向中央编排器注册 | 加一个新任务不用改训练代码 |
| 统一的 message-list 表示 | 不管是搜索、编程还是数学，交给训练侧的格式是同一种 |
| 编排器控制每任务的 rollout 配比和生成速度 | 多任务混训时不会某个任务把资源吃光 |
| **task-specific logic 与 core training loop 彻底隔离** | ★ 这是全篇最重要的一条架构原则 |

⚠️ **环境应该以「数据生成」的身份接入，而不是以「训练循环的修改」的身份接入。**
这条原则的实际含义是：写一个新环境的人，不需要懂分布式训练；改训练循环的人，不需要懂任何一个具体环境。两边通过一个固定的 message-list 契约解耦。

✅ 规模上，这个编排器**支撑超过 1000 路并发 rollout**，✅ 而 GLM-5 为编程任务建了**超过 10,000 个可验证训练场景**（*"over 10,000 verifiable training scenarios"*）。

### ✅ MiniMax 的白盒 / 黑盒二分

MiniMax Forge §6.2.3 给了另一个角度——agent 有两种接法：

| 类型 | 定义 | 框架能看到什么 |
|---|---|---|
| **白盒 agent** | 暴露自己的上下文管理逻辑：`s_{t+1} = f_CM(concat(s_t, a_t, o_t))` | 框架知道上下文是怎么拼的，可以做前缀复用、精确裁剪 |
| **黑盒 agent** | 不透明的轨迹生产者，只把补全请求路由给 Gateway | 框架只看到「有个东西在请求补全」 |

Forge 用同一个 **Gateway Server** 同时接住两种。✅ 报告原文：

> *"This design has been validated across **hundreds of distinct agent scaffolds** and **thousands of tool invocation formats**."*

⚠️ 这句话是 Agentic RL 基建设计的现实依据：**你不可能为每个 agent 脚手架写一套适配。** 唯一可扩展的做法是定义一个足够窄的接口（「给我补全」），让所有脚手架都能塞进来。这和 GLM 的 message-list 是同一个思路的两种实现。

---

## 6.3 ★★ 沙箱：三家的完整方案

环境要跑真实代码，就必须隔离。这一节把三家的沙箱基建放一起看——**这是五份报告里工程细节最密集的部分之一**。

### ✅ DeepSeek DSec：规模最大的一个

✅ 原文：*"we build a production-grade sandbox platform, **DeepSeek Elastic Compute (DSec)**"*

```
DSec 的三个 Rust 组件（自定义 RPC 协议连接，横向扩展）
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Apiserver   │   │     Edge     │   │   Watcher    │
│   API 网关   │   │ 每主机 agent │   │   集群监控   │
└──────────────┘   └──────────────┘   └──────────────┘
              全部架在 3FS 分布式文件系统之上
```

✅ **单个 DSec 集群管理数十万（hundreds of thousands）并发沙箱实例。**

✅ 设计动机是四点观察（直接对应四个工程难题）：

1. agentic 负载**高度异构**——从轻量函数调用到完整的软件工程任务
2. 环境镜像**又多又大**，但必须快速加载且支持迭代定制
3. **高密度部署**要求高效的 CPU 与内存利用
4. **沙箱生命周期必须与 GPU 训练调度协同**，包括抢占和基于 checkpoint 的恢复

✅ 三种运行时，按隔离强度分档：

| 运行时 | 基座 | 特点 |
|---|---|---|
| **Container** | 完全 Docker 兼容 | **EROFS 按需加载**镜像；预热容器池消除冷启动 |
| **microVM** | **Firecracker** | VM 级隔离，面向安全敏感的高密度部署 |
| **fullVM** | **QEMU** | 支持任意定制 |

✅ 两处密度优化（都很具体）：
- 缓解虚拟化环境下**重复的 page cache 占用**，并做内存回收以支持安全超售
- 缓解容器运行时的**自旋锁争用**，降低单沙箱 CPU 开销，显著提升单机装箱密度

✅ **轨迹日志与抢占安全恢复**：DSec 为每个沙箱维护一份**全局有序的轨迹日志**，持久记录每一次命令调用及其结果。用途之一是「客户端快进」——训练任务被抢占时沙箱资源仍然保留，恢复时 DSec **重放已完成命令的缓存结果**，加速任务恢复。

### ✅ Kimi AgentENV：细节最锋利的一个

✅ Kimi-K3 §5.3.2 用三种沙箱运行时（传统容器 / GPU 沙箱 / microVM），其中重点是与合作伙伴共建的 **AgentENV**，基于 **Firecracker**。三个设计目标：

**① 高保真隔离。** ✅ 原文给了一个很实在的理由：

> *"in our early experiments with traditional container-based sandbox runtimes, we observed several **kernel panics and deadlocks** caused by unintended agent operations."*

⚠️ 注意这句话的因果：不是「为了安全」而上 microVM，是**容器真的被 agent 搞崩过**。而且 Kimi 明确说他们**想尽可能允许探索**（agent 会尝试 reward hacking、会激进探索），所以不能靠限制 agent 来解决，只能靠更强的隔离。

**② 灵活的沙箱生命周期。** ✅ 增量 checkpoint：只保存自上次 checkpoint 以来**脏掉的内存页**，达到 —

> ✅ **checkpoint 延迟低至 133 ms，resume 延迟低至 49 ms**

在此之上提供三个高层操作：

| 操作 | 作用 | ✅ 关键数字 / 用途 |
|---|---|---|
| **Pause / Resume** | 暂停的沙箱**不占内存也不占 CPU** | agent 等模型推理结果的时间**可占沙箱生命周期的 98%**——这段时间沙箱可以完全释放 |
| **Fork** | 从当前状态精确复制一个新沙箱，原来的继续跑 | **用于奖励判定而不产生副作用** |
| **Snapshot** | 定期存快照 | 错误恢复 |

**③ 高效率高密度。** ✅ 数万个沙箱、每个镜像都不同，可能需要在数秒内创建完。用 **OverlayBD** 作为镜像格式，配自研的 **ublk 驱动**实现。

⚠️ 「Pause 能省 98%」是整章最值得记的一个数字。它说明：**Agentic RL 里沙箱资源的主要浪费不是执行，是等待。** 而这个等待恰好是可以被精确识别的（agent 在等模型返回），所以能被彻底优化掉。

⚠️ 「Fork 用于奖励判定」是一个很聪明的设计：判定奖励往往要跑测试、要改文件，这些操作有副作用。fork 一份出来判，原环境不受影响——省掉了「判完再恢复」的整套逻辑。

### ✅ GLM-5：把环境故障当成数据质量问题

GLM 的角度不同——它关心的是**沙箱失败会污染训练信号**。✅ 原文：

> *"coding-agent sandboxes can be inherently unstable and may fail for reasons unrelated to the model (e.g., environment crashes). Such failures introduce **noisy training signals** because they reflect **environment instability rather than the model's capability**. To mitigate this, we **record the failure reason for each sample and exclude samples that fail due to environment collapse**."*

⚠️ 这是一条被普遍低估的工程实践：**环境崩了不等于模型做错了。** 如果不区分，模型会学到「这类任务做不成」——而实际上只是基建不稳。要区分就必须**给每个样本记录失败原因**，这又是一条对数据通路契约的新要求（呼应 [04 章 §4.4](04-训推一致性.md)）。

### 三家对照

| | DeepSeek DSec | Kimi AgentENV | GLM-5 |
|---|---|---|---|
| 隔离技术 | Container / **Firecracker** / QEMU 三档 | **Firecracker** microVM | 未细述 |
| 规模 | ✅ 单集群**数十万**并发 | ✅ **数万**个，秒级创建 | ✅ **1k+** 并发 rollout |
| 恢复机制 | ✅ 全局有序轨迹日志 + 命令结果重放 | ✅ 增量内存页 checkpoint（133ms/49ms） | ✅ 心跳驱动的容错 |
| 独特的一招 | 抢占安全的「客户端快进」 | **Pause 省 98%** + **Fork 做无副作用奖励判定** | **按失败原因过滤样本** |

⚠️ 三家的侧重完全不同，合起来才是一套完整方案：**DeepSeek 解决规模，Kimi 解决单沙箱效率，GLM 解决数据质量。**

---

## 6.4 ✅ 开源框架里的 agent 层

### veRL：`agent_loop/` + `tools/`

✅ 本地实测（commit `b256ebf`）：

| 文件 | 行数 | 作用 |
|---|---|---|
| `verl/experimental/agent_loop/agent_loop.py` | **1,261** | agent 主循环 |
| `verl/experimental/agent_loop/tool_parser.py` | **816** | ★ 解析模型吐出来的工具调用 |
| `verl/experimental/agent_loop/tool_agent_loop.py` | 506 | 带工具的循环 |
| `verl/experimental/agent_loop/single_turn_agent_loop.py` | 107 | 单轮（退化情形） |
| `verl/tools/function_tool.py` | 258 | 函数工具 |
| `verl/tools/schemas.py` | 127 | 工具 schema 定义 |
| `verl/tools/tool_registry.py` | 101 | 工具注册表 |
| `verl/tools/base_tool.py` | 93 | 工具基类 |

⚠️ **`tool_parser.py` 816 行**这个数字值得停一下：解析工具调用比实现工具本身（`base_tool.py` 93 行）还复杂 8 倍多。因为不同模型吐工具调用的格式各不相同——有的用 JSON、有的用 XML 标签、有的用特殊 token，还要处理流式解析和格式错误。这正是 [harness评测框架/agent脚手架/03-工具集与ACI.md](../harness评测框架/agent脚手架/03-工具集与ACI.md) 讲的那个问题，在训练侧的同一副面孔。

### ★★ slime：直接把 Claude Code 和 Codex 当训练环境

这是本次源码走查里最意外的一个发现。✅ 本地实测（commit `3778dbf`），`slime/agent/` 共 13 个文件、2,727 行，其中：

```
slime/agent/
├── sandbox.py           399 行  沙箱抽象
├── trajectory.py        508 行  轨迹
├── parsing.py           114 行
├── adapters/                    ← 按 API 协议适配
│   ├── common.py        523 行
│   ├── openai.py        378 行
│   └── anthropic.py     350 行
└── harness/                     ← ★ 按【真实 agent CLI】适配
    ├── common.py        178 行
    ├── claude_code.py    86 行  ★ Claude Code
    └── codex.py          71 行  ★ Codex
```

✅ `claude_code.py` 里的 `ClaudeCodeHarness` 类，真的是在沙箱里装一个 Node 运行时、装 Claude Code 的 npm 包，然后用这些参数启动它：

```python
launch_flags = (
    "--permission-mode bypassPermissions "
    "--output-format stream-json --include-partial-messages "
    "--include-hook-events --verbose"
)
```

✅ `codex.py` 的注释更实在，把踩过的两个坑写在了文件头：

> *"Two non-obvious bits: the provider base_url must be inline in the TOML (Codex only honours env vars for the default OpenAI provider), and the config is written via a base64 round-trip to dodge shell-quoting traps."*

⚠️ **这件事的意义远超一个适配文件。** 它说明训练环境已经走到了这一步：不再是「我们自己写一个简化的 agent 循环来训练」，而是**直接把生产环境里真实用的那个 CLI 拉进沙箱训练**。模型是在它将来真正会被部署进去的那个 harness 里学习的。

✅ Kimi-K3 独立给出了同一个方向的判断——它的可验证任务：

> *"Every task runs in a containerized sandbox and is rolled out under **diverse agent scaffolds rather than a single fixed harness, to promote cross-scaffold generalization**."*

⚠️ 两家的选择合起来读很有意思：slime 是「用真实的 harness 训」，Kimi 是「用多个不同的 harness 训」。**前者防的是 sim-to-real 差距，后者防的是过拟合到单一脚手架。** 两个问题都真实存在，理想做法大概是两者都做。

✅ slime 的沙箱抽象刻意做得很窄，文件头注释：

> *"The public sandbox contract is **intentionally small**: async context management, command execution, and file read/write. Agent examples can build task-specific setup, runner, and evaluator logic on top of this **without depending directly on one sandbox provider**."*

⚠️ 这和 GLM 报告里那句「task-specific logic 与 core training loop 隔离」是同一条原则的源码实现——**报告说的和代码写的对上了**。

---

## 6.5 ★ 挂回 `harness评测框架/`

本仓库另有一个目录 [harness评测框架/agent脚手架/](../harness评测框架/agent脚手架/) 在讲 agent 脚手架。⚠️ 那边讲的东西和这一章是**同一件事的两面**：

| 同一个概念 | 在评测里叫 | 在训练里叫 | 关心什么 |
|---|---|---|---|
| 让模型能操作计算机的那套东西 | **harness / 脚手架** | **environment / 环境** | 评测关心公平可复现；训练关心吞吐与可恢复 |
| 工具定义与返回裁剪 | **ACI** | 工具 schema + tool parser | 两边**完全一样**，见 [03-工具集与ACI.md](../harness评测框架/agent脚手架/03-工具集与ACI.md) |
| 隔离执行 | 沙箱 | 沙箱 | 评测要防作弊；训练要防崩 + 要能 pause/fork |
| 上下文怎么拼 | 上下文管理 | 白盒 agent 的 `f_CM` | 见 [04-上下文管理.md](../harness评测框架/agent脚手架/04-上下文管理.md) |

⚠️ 三条推论：

1. **评测 harness 的差异会直接变成训练环境的差异。** 那边 [08-脚手架怎么影响榜单分数.md](../harness评测框架/agent脚手架/08-脚手架怎么影响榜单分数.md) 讲的「同一个模型换个脚手架分数就变」，在训练侧对应的是「同一个模型换个环境学到的东西就不一样」。

2. **slime 的 `harness/claude_code.py` 是这条连接的实证。** 那边 [06-ClaudeCode与Codex逐项对照.md](../harness评测框架/agent脚手架/06-ClaudeCode与Codex逐项对照.md) 逐项对比了两个 CLI 的设计，而 slime 把这两个 CLI **原封不动搬进了训练环路**——那份对照读下来的每一条差异，都会成为训练信号的差异。

3. **训练侧比评测侧多两个硬需求**：可 pause/resume（评测跑完就完了，训练要跨步续跑）和高并发（评测几百条，训练几十万条）。这就是为什么 Kimi 要自己做 AgentENV、DeepSeek 要自己做 DSec，而不是直接用现成的评测沙箱。

---

## 6.6 ✅ 一处闭环：逐样本版本号

[02 章 §2.5](02-架构主轴-同步到全异步.md) 引了 HF 横评的结论：**逐样本打版本号是必需的**，batch 级门控不够。当时那是 🌐 材料。

✅ GLM-5 §4.1.2 给了这条的一手实证——而且比博客说的更细：

> *"we log the policy weight version used by the rollout engine at generation time. Specifically, for each response we record **the sequence of model versions involved, (w₀, ..., wₖ) with w₀ < ... < wₖ**. Let w′ denote the current policy version. We discard a sample if its **oldest** rollout version is too stale, i.e., if **w′ − w₀ > τ**."*

⚠️ 注意两个细节：

1. **一条轨迹记的不是一个版本号，是一个版本序列。** 因为 agentic 轨迹跨越多次权重更新——第一轮工具调用用的是 w₀，第十轮可能已经是 w₃ 了。这正是 [03 章 §3.5](03-权重同步.md) 「永不停」中断模型的直接后果。
2. **判据用的是最老的那个 w₀，不是平均、也不是最新。** 这是保守选择：只要轨迹里有任何一段太旧，整条丢掉。

**🌐 博客说「要逐样本打版本号」，✅ GLM 的实现是「要逐样本打版本序列，并按最旧的那个判」——一手材料比二手材料更严格。**

---

## 6.7 本章小结

| 结论 | 分级 |
|---|---|
| Agentic RL 的长尾来自工具执行（CPU/IO），纯 GPU 视角的调度器看不见 | ⚠️ |
| 环境应以「数据生成」身份接入，与训练循环彻底隔离（GLM：统一 message-list） | ✅ + ⚠️ |
| MiniMax 白盒/黑盒二分 + Gateway 统一，已验证于数百种脚手架、数千种工具格式 | ✅ |
| DSec：三个 Rust 组件 + 3FS，单集群数十万并发沙箱，三档运行时 | ✅ |
| AgentENV：Firecracker microVM，checkpoint 133ms / resume 49ms | ✅ |
| **Pause 能省掉沙箱 98% 的生命周期**——浪费在等待而非执行 | ✅ |
| **Fork 用于无副作用的奖励判定** | ✅ |
| Kimi 上 microVM 的真实理由：容器被 agent 搞出过 kernel panic 和死锁 | ✅ |
| GLM 按失败原因排除「环境崩溃」样本——环境不稳不等于模型不行 | ✅ |
| veRL `tool_parser.py` 816 行 > `base_tool.py` 93 行：解析工具调用比实现工具难 8 倍 | ✅ 实测 + ⚠️ |
| **slime 直接把 Claude Code / Codex 真实 CLI 拉进沙箱当训练环境** | ✅ 实测 |
| slime「用真实 harness 训」vs Kimi「用多样 harness 训」——防的是两个不同的问题 | ✅ 双方 + ⚠️ |
| 训练环境比评测 harness 多两个硬需求：可 pause/resume + 高并发 | ⚠️ |
| GLM 的逐样本版本**序列** `(w₀..wₖ)`，按最旧的 w₀ 判 `w′−w₀>τ`——比博客的说法更严格 | ✅ ↔ 🌐 |

**下一篇** → [07-选型与实践.md](07-选型与实践.md)
