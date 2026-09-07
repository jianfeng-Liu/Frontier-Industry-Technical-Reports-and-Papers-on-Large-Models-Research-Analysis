# 07 · DeepSeek 的脚手架：从"查不到"到"可以逐行读"

> ⭐⭐⭐ 本章有 **两个** 一手来源，这在全目录是独一份：
>
> **来源 A｜论文**：DeepSeek-V4 技术报告
> 本地存档 [`业界模型技术报告/papers/arxiv-2606.19348.pdf`](../../业界模型技术报告/papers/arxiv-2606.19348.pdf)（58 页），
> 引用集中在 **§5.2.5 沙箱基础设施**、**§5.3.1 评测设置** 和 **交错思考** 一节。
> 本章所有 ✅ 逐字引文，我都用 `pypdf` 从这份 PDF 重新抽过一遍原文核对。
>
> **来源 B｜源码**：`deepseek-ai/deepseek-harness`（简称 **dsh**）
> ✅ 2026 年开源，MIT 协议，仓库地址 https://github.com/deepseek-ai/deepseek-harness
> 本章 ✅ 逐字源码引文来自我在 **2026-09-04** 拉取的 `master`，
> 提交 `76fda72`（2026-09-03），根包版本 `0.1.2-rc.1`。
>
> ⚠️ **2026-09-05 我又拉了一次**（`git clone --depth 1`，浅克隆），
> 已经走到提交 `d347e70`（2026-09-04，`release/dsh-0.1.3-alpha.1` 的合并），
> 根包版本 **`0.1.3-alpha.1`**。变动见 7.1 表，
> 新核实到的凭据/API key 机制见 **7.2b**（该节是第二次拉取时新写的）。
> ★ 本章其余 ✅ 引文我没有在新提交上逐条重抽，遇到对不上以仓库现状为准。
>
> ★★★ 一句话预告：**这一章是全目录唯一一处，
> 你能同时看到【脚手架配置】【它的源码实现】和【它跑出来的分数】三件东西的地方。**

---

## 7.0 ⚠️ 这一章被推翻过一次，请先读这一段

```
   ⟹ 本目录 README 原来把可查证度分成三档，DeepSeek 被放在最底下：

      ① SWE-agent            ✅ 论文 + 源码，全可查
      ② Claude Code / Codex  ✅ 行为可查 / ⚠️ 实现不可查
      ③ DeepSeek             ⚠️ 基本查不到 ← 【这一条现在错了】

   ⚠️⚠️ 2026 年 8 月，DeepSeek 把自家的 agent 脚手架整个开源了，
        仓库名 deepseek-harness，命令行叫 dsh，MIT 协议。
   ⟹ ★★★ 于是 DeepSeek 从【最底档】直接跳到了【比第①档还全】的位置：
        论文（规格）＋ 源码（实现）＋ 分数（结果），三样齐全。
        SWE-agent 有论文和源码，但它不是任何一家前沿模型的跑分脚手架；
        Claude Code / Codex 是产品脚手架，但源码/分数对不上号。
```

⚠️ 但请立刻注意一条边界，它决定了本章后面每一节的措辞：

```
   ┌─────────────────────────────────────────────────────────────┐
   │ ✅ 报告说：跑分用的是 "an internally developed evaluation    │
   │    framework"（一个内部开发的评测框架）。                     │
   │                                                             │
   │ ✅ 仓库存在：dsh，且里面有一个配置档位（sdk-minimal）        │
   │    和报告描述的那套参数【逐项吻合】。                        │
   │                                                             │
   │ ⚠️⚠️ 但 —— 报告【没有】说那个内部框架就是 dsh，              │
   │    dsh 仓库里也【没有】任何一处提到 SWE-bench、              │
   │    Terminal-Bench 或跑分复现。我 grep 过，零命中。           │
   │                                                             │
   │ ⟹ ★★★ 所以本章的写法是：                                   │
   │    【报告说了什么】和【仓库里有什么】分开列，                 │
   │    两者对得上的地方我标 ⚠️「高度吻合，但这是我的推断」。      │
   └─────────────────────────────────────────────────────────────┘

   ★ 时间线也支持"不是同一份代码"这个谨慎读法：
     报告 arXiv 编号 2606（2026 年 6 月）⟹ 跑分在 6 月之前；
     dsh 首次公开在 2026 年 8 月，且仍标注 developer preview。
   ⟹ ⚠️ 我的判断：dsh 更像那套内部框架【产品化之后的后代】，
     而不是当初那份跑分脚本本身。
```

---

## 7.1 ✅ dsh 是什么：仓库事实速览

★ 名词：**dsh** = **D**eep**S**eek **H**arness（DeepSeek 脚手架）的命令名。
⟹ ★★ 注意它自称 harness 而不是 agent —— 这正是 [01 章](01-两个harness不是一个东西.md) 讲的歧义词，
   这里是【agent 脚手架】那个意思，不是【评测框架】。

| 事实 | 值 | 我怎么核实的 |
|---|---|---|
| 协议 | ✅ MIT，`Copyright (c) 2026 DeepSeek` | `LICENSE` 首行 |
| 根包 | ✅ `@deepseek-ai/dsh-root`。版本：09-04 读到 `0.1.2-rc.1`，⚠️ 09-05 已是 **`0.1.3-alpha.1`** | `package.json` |
| 运行时 | ✅ Node.js `^22.19.0 \|\| >=24.0.0`，包管理器锁定 `pnpm@11.7.0` | `package.json` 的 `engines` / `packageManager` |
| 规模 | ✅ `packages/` 下的 `package.json`：09-04 是 **257** 个，⚠️ 09-05 是 **262** 个 | `find packages -name package.json \| wc -l` |
| 成熟度 | ⚠️ README 自称 *developer preview*，并全大写警告 **"THERE WILL BE COMPATIBILITY-BREAKING CHANGES"**（将会有破坏兼容性的改动） | `README.md` |
| 热度 | ⚠️ 我 2026-09-04 读到 **211.4k star / 24.8k fork / 906 watching / 14,981 commits**。★ 这是网页读数，会变，别把它当固定事实 | GitHub 仓库页 |
| 外部贡献 | ✅ `CONTRIBUTING.md` 明写 **"we cannot accept external pull requests at the moment"**（目前无法接受外部 PR），引导去 Discussions 和写插件 | `CONTRIBUTING.md` |

**★★★ 它的架构主张只有一句话：everything is a plugin（万物皆插件）。**

✅ `docs/architecture.md` 逐字：

> "Every part of the product is a plugin, including the model adapter, the tool registry,
> the session log, and the agent loop itself, so each is replaceable from configuration.
> There is no privileged core to patch."

⟹ **产品的每一个部分都是插件，包括模型适配器、工具注册表、会话日志，以及 agent 循环本身，因此每一个都可以通过配置替换。不存在一个享有特权的、需要你去打补丁的内核。**

```
   ★ 名词：Cordis
     ⟹ dsh 底下的插件框架，✅ README 说其设计发表于
        arXiv 2608.25512《A Programming Paradigm for Spatiotemporal Composability》
        （一种面向时空可组合性的编程范式）。
     ⚠️ 这篇论文我【没有】独立核实，只转述 README 的指向。

   ★ 名词：profile（配置档）/ bundle（插件包）
     ⟹ ✅ 一次 dsh 运行 = 按顺序叠起来的一棵插件树。
        profile 列出它要叠哪些 bundle，bundle 是一组配置行 + 对应代码。
     ⟹ ★★ 官方随包发的 profile 有五个：
        web / headless / sdk / sdk-minimal / acp。
     ⟹ 🔑🔑 ★★★ 记住 sdk-minimal 这个名字，下一节整节都在讲它。
```

---

## 7.2 ★★★ 本章最硬的一节：报告里那四个参数，在仓库里对应哪几行

先把报告那句话再放一遍（✅ 逐字，我已用 pypdf 从 PDF 重新抽取核对）：

> "For code agent tasks (SWE-Verified, Terminal-Bench, SWE-Pro, SWE Multilingual),
> we evaluate DeepSeek-V4 series using an internally developed evaluation framework.
> This framework provides a minimal set of tools — a bash tool and a file-edit tool.
> The maximum number of interaction steps is set to 500, and the maximum context
> length is set to 512K tokens."

⟹ **对于代码 agent 任务（SWE-Verified、Terminal-Bench、SWE-Pro、SWE 多语言版），我们使用一个内部开发的评测框架来评测 DeepSeek-V4 系列。该框架提供一套极简工具集 —— 一个 bash 工具和一个文件编辑工具。最大交互步数设为 500，最大上下文长度设为 512K token。**

★★★ 现在看 `packages/bundle/sdk-minimal/README.md` 的开头一行（✅ 逐字）：

> "Standalone two-tool SDK profile for users who need a minimal cross-platform coding
> agent without the shared base bundle."

⟹ **独立的双工具 SDK 配置档，面向需要一个极简跨平台编码 agent、又不想要共享 base 插件包的用户。**

```
   ⟹ 🔑🔑🔑 ★★★ "two-tool"（双工具）。
      ⚠️ 报告说的 "a bash tool and a file-edit tool"，
         在仓库里有一个【自称双工具】的官方 profile。
      ⟹ ★★ 这是我认为两者同源的最强线索，但仍然只是线索。
```

**✅ 把 `packages/bundle/sdk-minimal/cordis.patch.yml` 里相关的行摘出来，逐项对照：**

| 报告说的 | 仓库里对应的配置行 | 对得上吗 |
|---|---|---|
| bash 工具 | `id: persistent-bash` → `@deepseek-ai/dsh-tool-bash-persistent`，`timeoutMs: 300000`（300 秒）；Windows 上换成 `persistent-pwsh` | ✅ 对上 |
| 文件编辑工具 | `id: str-replace-editor` → `@deepseek-ai/dsh-tool-str-replace-editor`，`maxOutputChars: 16000` | ✅ 对上 |
| "minimal set"（极简工具集） | 整棵树里**只有这两个** model-facing 工具 | ✅ 对上 |
| 512K 上下文 | ⚠️ **对不上**：`defaultContextWindow` 默认取 `DSH_CONTEXT_WINDOW ?? 1000000`（100 万） | ⚠️ 见下 |
| 500 步上限 | ❌ **仓库里根本没有这个开关**（详见 7.4） | ❌ 没有 |

★★ 顺便把这个 profile 的其它设定也列出来，因为每一条都是【跑分口径】：

```
   ✅ 系统提示词：DSH_SYSTEM_PROMPT，缺省是一句
      "You are a helpful software engineer assistant."
      （你是一个乐于助人的软件工程师助手。）
      ⟹ 🔑 ★★★ 就这一句。没有几千字的 persona。
      ⟹ ★★ 对照 06 章讲的 Claude Code —— 完全相反的路线。

   ✅ 沙箱策略：mode: danger-full-access（危险·完全访问）
      ⟹ ⚠️ README 自己警告："use it only with an isolated workspace"
         （只在隔离的工作区里用）。
      ⟹ ★★ 这在跑分场景是合理的：题目本来就跑在一次性容器里，
         再套一层进程级限制只会让 bash 用不顺。

   ✅ 明确【不装】的东西（README 逐条列出）：
      Web 界面、设置、托管凭据、遥测、compaction（上下文压缩）、
      工作区指令、skills、jobs、subagents 全部缺席。
   ⟹ 🔑🔑 ★★★ 请把这一行读三遍：
      【跑分档位是把功能一个个【拆掉】拆出来的，不是攒出来的。】

   ✅ 会话持久化：未压缩 JSONL，落在 $DSH_HOME/sessions
   ✅ 流空闲超时：streamIdleTimeoutMs: 172800000
      ⟹ ★★ 换算一下 = 48 小时。
      🔑 一个允许模型思考两天不吭声的超时值，
         本身就说明设计者预期的是【超长任务】。
```

**⭐ 还有一份一行的文档，位置很说明问题 —— 仓库根目录的 `BENCHMARK.md`**（✅ 全文只有这么多）：

> "Follow *Get started with the Python SDK* to install the SDK and run the
> `jsonrpc-agent` minimal variant. Use separate workspaces and session IDs for
> independent benchmark tasks."

⟹ **按《Python SDK 上手》安装 SDK，并运行 `jsonrpc-agent` 的 minimal 变体。为互相独立的基准任务使用各自独立的工作区和会话 ID。**

```
   ⟹ 🔑🔑 ★★★ 官方仓库里【叫 BENCHMARK.md 的那份文档】，
      指向的正是 sdk-minimal 这个 profile。
   ⟹ ★★ 也就是说：DeepSeek 自己认为"要跑分就用这个双工具档位"。
      ⚠️ 但它仍然没说"V4 报告里那些分数是这么跑出来的"。
```

★ `docs/user/guide/python-sdk.md` 里那张表把这个档位的口径写得更全（✅ 逐字摘要）：
模型缺省 `deepseek-v4-flash`，示例里 `max_tokens=49_152`，shell 超时 300 秒，
编辑器输出上限 16,000 字符，**"Runtime context and compaction: Absent"**（运行时上下文与上下文压缩：无）。

---

## 7.2b ⭐⭐ 补录（2026-09-05）：API key 放哪，暴露了 sdk-minimal 又少一样东西

> ★ 这一节是我第二次拉仓库、真正**把 dsh 配起来跑通一次**之后补的。
> 它不改变 7.2 的结论，而是给"跑分档位是拆出来的"这句话再添一条证据。

★ 名词：**凭据引用（credential reference）**
⟹ dsh 的配置文件里【从不写 key 本身】，只写一个环境变量名（例如 `apiKeyEnv: DEEPSEEK_API_KEY`），
   真正的值由另一类插件（凭据提供方）保管。所以"key 放哪"这个问题，
   在 dsh 里等价于"**哪个 profile 挂了凭据插件**"。

**✅ 完整档位（base 组合包，即 web / headless）有四层取值，先命中者赢：**

| 层 | 位置 | 可写？ |
|---|---|---|
| ① 启动进程的环境变量 | `DEEPSEEK_API_KEY=… dsh` | ❌ 只读 |
| ② 受管凭据文件 | `$DSH_HOME/.credentials.yaml` | ✅ Web 界面写的就是它 |
| ③ 项目 `.env` | `<调用目录>/.env` | ❌ |
| ④ home `.env` | `$DSH_HOME/.env` | ❌ |

✅ 逐字，`apps/cli/reference/README.md`：

> "Provider credentials resolve from the inherited environment, `$DSH_HOME/.credentials.yaml`,
> the invoking directory's `.env`, then `$DSH_HOME/.env`; the managed document is never
> materialized into `process.env`."

⟹ **凭据依次从继承环境、`$DSH_HOME/.credentials.yaml`、调用目录的 `.env`、`$DSH_HOME/.env` 解析；受管文档从不物化进 `process.env`。**

```
   🔑🔑 ★★★ 但 sdk-minimal【没有凭据插件】。
      ✅ 我 grep 过 packages/bundle/sdk-minimal/cordis.patch.yml，
         关键词 credential 零命中。

   ⟹ ✅ 对应源码 packages/llm/llm-deepseek/src/index.ts:433-447：
        先问 ctx.get('credentials')，拿不到这个服务就【直接退回读启动环境变量】，
        再没有就抛 MISSING_CREDENTIAL。

   ⟹ ★★★ 所以跑分档位里，环境变量是【唯一】入口，
      .credentials.yaml 和两个 .env 全部不生效。
```

⟹ 🔑 **把这条并进 7.2 那份"明确不装的东西"清单**：
   Web 界面、设置、托管凭据、遥测、compaction、工作区指令、skills、jobs、subagents ——
   ★★ 其中"托管凭据"这一条，现在有了源码级的确证（缺插件 + 缺插件时的回退路径）。
   ⟹ ★★ 这也很合理：跑分跑在一次性容器里，key 本来就该从 `-e` 注进去，
      再存一个磁盘文件只是多一个要清理的东西。

**⚠️ 顺带纠正一处仓库自身的文档错误**（这条值得单独记住）：

```
   ⚠️ packages/llm/llm-deepseek/README.md 的字段表写：
      "baseURL … $DEEPSEEK_BASE_URL wins when set"（环境变量优先）

   ✅ 但源码 packages/llm/llm-deepseek/src/index.ts:378-380 是：
      config.baseURL  ??  $DEEPSEEK_BASE_URL  ??  https://api.deepseek.com
      ⟹ 【配置 / settings 层最高，环境变量只排第二】—— 和 README 写反了。

   ⟹ ★★ 我用实验确认过源码版本：把 $DSH_HOME/settings.yaml 的 baseURL
      改成一个不存在的域名，dsh 报错时喊的正是那个假域名：
        dsh: TRANSPORT: DeepSeek API request to https://…invalid/v1 failed
   ⟹ 🔑 教训：dsh 的 README 字段表【不可尽信】，以源码为准。
```

★ 另外两条第二次拉取时顺带核实到的小事实：
✅ base 组合包里 `agent-default-model` 的缺省就是 `provider: deepseek-official` / `model: deepseek-v4-flash`
（`packages/bundle/base/cordis.patch.yml:75-79`），所以开箱即用不需要选模型；
✅ 内置 `web_search` 工具**复用同一个 `DEEPSEEK_API_KEY`**，另接受 `DEEPSEEK_SEARCH_BASE_URL`。

★ 还有一条和 7.7（轨迹日志）呼应的观察：
✅ `headless` 档位落的会话日志是 **zstd 压缩**的（`session.jsonl.zstd`），
而 7.2 引的 `sdk-minimal` 是**未压缩 JSONL** —— 两个档位的持久化策略确实不同。

---

## 7.3 ★★★ 第一个反差：2 个工具 vs 同一个仓库里的 62 个

★★ 03 章那张表现在可以补完了 —— 而且最有意思的是，**最后两行来自同一个仓库**：

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
   │ dsh 全量目录  │ ★★★ ✅ 62 个 model-visible 工具名，     │
   │ （2026 源码） │ 分布在 24 个工具包里                    │
   ├───────────────┼────────────────────────────────────────┤
   │ dsh           │ ★★★ ✅ 2 个：bash + str_replace_editor  │
   │ sdk-minimal   │ 官方称 "two-tool profile"               │
   └───────────────┴────────────────────────────────────────┘
```

✅ 那 62 个名字，我从 `docs/tool-catalog.md` 里 `grep '^### '` 数出来的，包括：

```
   ask_user_question  bash  pwsh  read  write  edit  read_image  glob  grep
   run_code  exit_plan_mode  lsp  skill  todo_write  workflow  ralph
   web_search  web_fetch  job_list  job_output  job_kill
   terminal_open/read/send/close/list/signal（6 个）
   subagent  list_subagent_models  send_message  interrupt_agent  list_agents
   create_goal / get_goal / update_goal
   schedule_create / delete / list
   session_search / session_trace / session_event_read / _search / _trace
   spawn_teammate  team_task_create / get / list / update  wait_agent
   cordis_define / undefine / run / stop / inspect_list / inspect_query / inspect_self
```

```
   ⟹ 🔑🔑🔑 ★★★ 这才是真正的结论，比原来那版"2 个 vs 9 个"深一层：

   ⚠️ 不是【DeepSeek 认为 2 个工具就够了】。
   ✅ 是【同一家公司、同一个仓库，产品档位给 62 个，跑分档位给 2 个】。

   ⟹ ★★★ 工具数量不是一个技术信仰，是一个【按场景拨的旋钮】。
      ★★ 产品要好用 ⟹ 工具多，给人省事。
      ★★ 跑分要干净 ⟹ 工具少，减少变量、减少 prompt 里的 schema 占用。
```

⚠️ 一个必须说清的细节：这 62 个不是"默认全开"。
✅ `docs/tool-catalog.md` 里明确标注了哪些**不在任何随包配置里**（例如 `cordis_*` 那七个，
原文 "Not in any shipped tree"，⟹ 不在任何随包发布的插件树里），
以及哪些默认关闭（例如 Agent Teams 那九个，"The shipped dsh-base bundle keeps the package disabled"）。
✅ 实际的 `dsh-base` 插件包一共 **85 行配置行**（我 grep `- id:` 数的）。
⟹ ★★ 所以"62"是**目录规模**，不是**某一次运行的工具数**。这个区别在读任何脚手架时都要守住。

★★ 还有一条旁证仍然成立，它解释了"为什么 2 个就敢跑"（✅ V4 报告逐字）：

> "for each target domain — such as mathematics, coding, agent, and instruction
> following — a separate expert model is trained independently."

⟹ **对每个目标领域 —— 例如数学、编程、agent、指令遵循 —— 都单独训练一个专家模型。**

```
   ⟹ 🔑 ★★★ "agent" 被列为一个和数学、编程并列的【训练领域】。
      ⟹ ★★【"会用工具"是被专门训出来的能力】，不再靠脚手架兜。
      ⟹ ⚠️ 我的推断：能力从【脚手架】搬进了【模型权重】，
         所以跑分时才敢把脚手架拆到只剩两个工具。
```

---

## 7.4 ⭐⭐ 第二个反差：500 步 —— 而仓库里【没有】这个开关

★★ 先看数字对比：

| | SWE-agent（2024） | DeepSeek-V4（2026） |
|---|---|---|
| 上下文窗口 | ✅ GPT-4 Turbo **128K** / Claude 3 Opus 200K | ✅ 报告：**512K** ／ ⚠️ dsh 默认回退值 **1M** |
| 步数上限 | ⚠️ 成本上限约束下，实测成功案例约 **12 步**、失败约 21 步 | ✅ 报告：**500 步** |
| 上下文策略 | ✅ 只留最近 5 条观察 | ✅ 搜索 agent 明说 discard-all；⚠️ 代码 agent 未说明 |

**★★★ 但现在有源码了，就能问一个原来问不出的问题：这个 500 是配在哪儿的？**

```
   ⚠️⚠️ 答案是：【不在 dsh 里】。

   ✅ 我 grep 了整个仓库：maxSteps / max_steps / stepLimit / maxTurns
      —— 全部零命中。
   ✅ agent-loop 这个"唯一的具体循环"只有一个上限类配置：
      maxParallelToolCalls，默认 10，
      ⟹ 管的是"一步里最多并发几个工具调用"，不是"总共几步"。
   ✅ 它的 README 自己说：
      "It is the harness's only concrete loop — everything beyond
       'call the model, run the tools, repeat' belongs to plugins."
      ⟹【它是本脚手架唯一的具体循环 —— 除了"调模型、跑工具、重复"
         之外的一切，都属于插件。】

   ⟹ 🔑🔑🔑 ★★★ 所以"500 步"这个数，
      不是脚手架的属性，是【跑分脚本套在脚手架外面的一层】。

   ⟹ ★★ 这条结论比数字本身更有用。它说明：
      ┌────────────────────────────────────────────────┐
      │ 一次跑分的口径，至少分布在三层：                 │
      │   ① 模型（V4-Pro / reasoningEffort 档位）       │
      │   ② 脚手架（工具集、超时、沙箱、上下文策略）     │
      │   ③ 跑分驱动脚本（步数上限、并发、题目集、重试） │
      │ ⟹ ★★★ 开源了②，不等于能复现③。               │
      └────────────────────────────────────────────────┘
```

⚠️ 关于 512K vs 1M 那个差异，我的读法是：
`DSH_CONTEXT_WINDOW ?? 1000000` 是"模型不在适配器目录里时的**回退值**"，
不是跑分时实际用的窗口；报告的 512K 应该是跑分脚本显式设的。
✅ 报告标题本身就是 "Towards Highly Efficient Million-Token Context Intelligence"
（迈向高效的百万 token 上下文智能），所以 1M 这个回退默认值和模型定位是一致的。
⚠️ 这一段是我的推断，报告和仓库都没有把两者接起来。

**⭐ 上下文策略这一段，报告在【搜索 agent】那里给了明确答案**（✅ 逐字）：

> "For search agent tasks (BrowseComp, HLE w/ tool), we also use an in-house harness
> with websearch and Python tool, and set maximum interaction steps to 500 and the
> maximum context length to 512K tokens. For BrowseComp, we use the same
> discard-all context management strategy as DeepSeek-V3.2."

⟹ **对于搜索 agent 任务（BrowseComp、带工具的 HLE），我们同样使用一个内部 harness，配备网页搜索和 Python 工具，最大交互步数设为 500，最大上下文长度设为 512K token。对于 BrowseComp，我们使用与 DeepSeek-V3.2 相同的「全部丢弃」上下文管理策略。**

```
   ★ 名词：discard-all（全部丢弃）
     ⟹ ★★★ 04 章三种策略里【最激进的截断】：不是折叠成一行，是直接扔。

   ⟹ ★★ 现在可以对照仓库了 —— dsh 里的上下文管理有三个包：
      ✅ dsh-compaction-basic
         ⟹ 压力上来时，把最老的一段【总结成一段摘要】，
            代价是"one extra model request"（一次额外的模型请求）。
      ✅ dsh-compaction-tool-result-pruner
         ⟹ 只裁工具输出：留头 + 一句 "middle pruned" + 留尾，
            ★★★ 且【完整原文仍留在 session log 里】供回放。
            🔑 不用调模型，所以有时光靠它就把压力泄掉了。
      ✅ dsh-command-compact
         ⟹ 人手打 /compact 立刻压一次。

   ⟹ 🔑🔑 ★★★ 请注意这里的反差：
      ✅ dsh 有一整套【摘要式】压缩；
      ✅ 而跑分用的 sdk-minimal 明确写着 compaction【缺席】；
      ✅ 报告说 BrowseComp 用的是【全部丢弃】。
   ⟹ ★★ 三处口径三个样。
      🔑 这正是 04 章的核心论点：上下文策略是配置，不是产品属性。
```

---

## 7.5 ⭐⭐⭐ 第三个披露：脚手架的写法能让模型特性【失效】——现在能看到实现了

★★★ 这仍然是我认为整份报告里最重要的一段。先看模型侧特性（✅ 逐字，§ Interleaved Thinking）：

> "Tool-Calling Scenarios. … all reasoning content is fully preserved throughout the
> entire conversation. Unlike DeepSeek-V3.2, which discarded thinking traces upon
> each new user turn, DeepSeek-V4 series retain the complete reasoning history across
> all rounds, including across user message boundaries."

⟹ **工具调用场景：全部推理内容在整段对话中被完整保留。与 DeepSeek-V3.2 在每个新用户回合到来时丢弃思考轨迹不同，DeepSeek-V4 系列跨所有轮次保留完整的推理历史，包括跨越用户消息边界。**

```
   ★ 名词：interleaved thinking（交错思考）
     ⟹ 模型的"思考过程"和"工具调用"交替进行，且思考内容在后续轮次里仍可见。
   ★ 名词：thinking trace / reasoning trace（思考轨迹）
     ⟹ 模型在给出动作之前那段"内心独白"。
     ⟹ ★★ 保留它 = 模型记得"我上一步为什么那么想"。
```

★★★ 然后是那句关键警告（✅ 逐字，我已从 PDF 原文核对）：

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
```

**★★★ 而现在，这条"要保留思考轨迹"的要求，在 dsh 里能看到是怎么落实的。**

✅ `packages/llm/llm-deepseek/README.md` 逐字：

> "Reasoning content from a prior assistant turn is passed back verbatim, whether or
> not that turn called a tool."

⟹ **来自上一个助手回合的推理内容会被【原样】回传，无论那一回合是否调用了工具。**

```
   ⟹ 🔑🔑 ★★★ "verbatim"（逐字原样）+ "whether or not that turn called a tool"
      （无论有没有调工具）—— 这正是报告承诺的那个特性的实现侧。
   ⟹ ★★ 官方适配器把它做成了【无条件行为】，
      而不是留给使用者去配。⟹ 这就是"自家脚手架配自家模型"的价值。
```

⚠️ 但同一份 README 也写了它的代价（✅ 逐字）：

> "reasoning passback appends on every reasoned turn"

⟹ **每一个有推理的回合，回传都会往上追加。**

```
   ⟹ ★★ 上下文写在"是否命中 prompt cache（提示词缓存）"那一段里。
      ★ 名词：prompt cache（提示词缓存）
        ⟹ 请求前缀没变时，服务端可以复用上一次的计算，价格更便宜。
   ⟹ 🔑 ★★ 所以【保留思考轨迹】不是白拿的：
      它让上下文单调变长，也参与缓存命中的判定。
   ⟹ ⚠️ 报告只讲了它的好处，源码 README 讲了它的账。
      ★★★ 这就是"有源码"比"只有论文"多出来的那部分信息。
```

★★ 顺带一个只有源码才看得到的口径：`reasoningEffort`（推理力度）有四档 —— ✅ `off / low / high / max`，
适配器默认 `high`；`off` 不会以 `reasoning_effort: 'off'` 上线，而是发 `thinking: { type: 'disabled' }`。
⟹ 🔑 ★★★ 读任何 DeepSeek 的榜单数字时，除了"哪个模型"，还得问"哪一档 effort"。
✅ 这也解释了 7.7 那张表里为什么写着 "DS-V4-Pro **Max**"、"GPT-5.4 **xHigh**"、"Gemini-3.1-Pro **High**"
—— 那些后缀就是这个旋钮。

⚠️ 关于 `Terminus`：它是报告点名的第三方 agent 框架。我没有独立核实它的实现，只转述报告的说法。

---

## 7.6 ⚠️ 沙箱：报告里的 DSec 和开源的 dsh，【不是同一个东西】

★★ 报告 §5.2.5 用整整一节讲沙箱平台。✅ 逐字：

> "we build a production-grade sandbox platform, DeepSeek Elastic Compute (DSec).
> DSec comprises three Rust components — the API gateway (Apiserver), per-host agent
> (Edge), and the cluster monitor (Watcher) … In production, a single DSec cluster
> manages hundreds of thousands of concurrent sandbox instances."

⟹ **我们构建了一个生产级沙箱平台 DeepSeek Elastic Compute（DSec）。DSec 由三个 Rust 组件构成 —— API 网关（Apiserver）、每主机代理（Edge）和集群监视器（Watcher）……在生产环境中，单个 DSec 集群管理数十万个并发沙箱实例。**

✅ 补一条原来漏掉的细节：报告说这三个组件"scale horizontally atop the **3FS** distributed filesystem"
（在 3FS 分布式文件系统之上横向扩展）—— 3FS 是 DeepSeek 早前开源的另一个项目。

**✅ 报告披露的四种执行基底：**

| 基底 | 报告原文 ⟹ 中文 | ★ 隔离强度 |
|---|---|---|
| **Function Call** | "dispatches stateless invocations to a pre-warmed container pool, eliminating cold-start overhead" ⟹ 把无状态调用派发到预热容器池，消除冷启动开销 | 最弱、最快 |
| **Container** | "fully Docker-compatible" ⟹ 完全兼容 Docker | ★★ 与 SWE-agent 同级 |
| **microVM** | 基于 Firecracker，"adds VM-level isolation for security-sensitive, high-density deployments" ⟹ 为安全敏感、高密度部署提供虚拟机级隔离 | ★★★ 更强 |
| **fullVM** | 基于 QEMU，"supports arbitrary guest operating systems" ⟹ 支持任意客户操作系统 | 最强、最重 |

**⚠️⚠️ ★★★ 现在是这一节的重点：我在 dsh 仓库里 grep 了 `DSec` 和 `Elastic Compute`，零命中。**

```
   ⟹ 🔑🔑🔑 ★★★ 【开源的是脚手架，不是那套沙箱基础设施。】

   ✅ dsh 自带的隔离，是【同一台机器上的进程级约束】：
      packages/sandbox/sandbox-local 逐字：
        Linux  ⟹ bwrap（bubblewrap），不行就退到 Landlock 启动器
        macOS  ⟹ Seatbelt（sandbox-exec）
        Windows⟹ ACL 受限令牌（restricted token）
      三档策略：read-only / workspace-write / danger-full-access

   ✅ 而 packages/sandbox/sandbox 的 README 自己把边界划得很清（逐字）：
      "Confinement is same-world only — backends share the host kernel and
       filesystem, while containers, microVMs, and remote executors replace
       whole capabilities instead."
      ⟹【约束仅限于同一世界内 —— 各后端共享宿主内核与文件系统；
         而容器、microVM、远程执行器是【替换整块能力】，不是在这里加后端。】

   ✅ 一条我很欣赏的设计：fail closed（失败即关闭）。
      "When the requested mode cannot be enforced, the call fails closed with a
       SANDBOX_UNAVAILABLE error instead of running unconfined."
      ⟹【当请求的模式无法被强制执行时，调用以 SANDBOX_UNAVAILABLE 错误
         失败关闭，而不是在无约束状态下运行。】
      ⟹ ★★ 对照 05 章：这正是"宁可不跑，也不静默降级"。

   ✅ 想要真隔离怎么办？仓库给了远程档位：
      packages/e2b（dsh-e2b / fs-e2b / subprocess-e2b）⟹ 一个共享的远程
      Linux 沙箱，文件、命令、终端全在里面跑。
      ⚠️ 但 README 明写 "no shipped composition enables this family by default"
      （没有任何随包配置默认启用这一族）—— 而且 E2B 是第三方服务。
```

```
   ⟹ 🔑 ★★★ 所以 05 章那句 SWE-agent 论文的坦白：
      "not considered as secure as virtualized hardware isolation"
      （不如硬件虚拟化隔离安全）——
   ⟹ ★★ DSec 的 microVM / fullVM 把这个缺口补上了，
      ⚠️ 但【补上那部分没有开源】。
      dsh 开源的这一层，隔离强度和 SWE-agent 是同一量级，
      甚至更轻（进程级 > 容器级只在"不需要装 Docker"这点上占优）。

   ⟹ ★★★ 一句话记住分界线：
      【DeepSeek 开源了"模型怎么被驱动"，没有开源"命令跑在哪里"。】
```

---

## 7.7 ⭐ 轨迹日志与确定性重放：报告里的承诺，在 dsh 里变成了架构不变量

✅ 报告逐字：

> "DSec maintains a globally ordered trajectory log for each sandbox, persistently
> recording every command invocation and its results."
> … "deterministic replay — any historical session can be faithfully reproduced from
> its trajectory."

⟹ **DSec 为每个沙箱维护一份全局有序的轨迹日志，持久记录每一次命令调用及其结果。……确定性重放 —— 任何历史会话都能从它的轨迹被忠实复现。**

★★★ 现在看 dsh 的架构文档，同样的思想被写成了一条**强制约束**（✅ `docs/architecture.md` 逐字）：

> "**Model-visible means logged.** Anything that reaches a model request must be
> reconstructable from the log, and a runtime invariant asserts it."

⟹ **【模型可见即已记录。】任何进入模型请求的东西都必须能从日志中重建，并且有一条运行时不变量在断言这一点。**

```
   ★ 名词：runtime invariant（运行时不变量）
     ⟹ 程序运行时持续自检的一条断言：一旦被违反就报错，不是写在注释里的君子协定。

   ⟹ 🔑🔑 ★★★ 这句话的分量：
      它不是"我们记了日志"，而是
      【凡是模型看得见的，必须能从日志重建，否则运行时报错】。
   ⟹ ★★ 配套机制：
      ✅ session log 是 append-only（只追加）的 SessionEvent 流；
      ✅ deriveMessages() 从这条流【投影】出模型历史；
      ✅ 连 assistant/chunk（流式片段）原样保留，用于回放和界面还原；
      ✅ 前面提到的 tool-result pruner 裁掉的只是【模型看到的那份】，
         完整原文留在日志里 —— 这正是这条不变量在起作用。
```

⚠️ 但请注意，这里有一个**新出现的、必须说的东西**：

```
   ✅ docs/deepseek-llm-api-wire-extensions.md 披露了两个私有请求扩展字段：

   ┌─────────────────────┬──────────┬──────────────────────────────┐
   │ dsh_plugin_packages │ ✅ 默认开 │ 上报当前激活的插件包名+版本   │
   │ dsh_session_log     │ ⚠️ 默认关 │ 上报会话日志的一段连续后缀    │
   └─────────────────────┴──────────┴──────────────────────────────┘

   ✅ 文档自己列了 dsh_session_log 打开后会暴露什么（逐字）：
      "the Session working directory, system-prompt snapshots, user and assistant
       content, raw assistant chunks, tool arguments and results, compaction
       summaries, feedback, and plugin-owned events."
      ⟹【会话工作目录、系统提示词快照、用户与助手内容、原始助手片段、
         工具参数与结果、压缩摘要、反馈，以及插件自有事件。】

   ⟹ 🔑 ★★★ 我认为这一段值得单独指出来，两个原因：
      ① ✅ 默认关闭、且文档主动把暴露面逐项写出来 —— 这是负责任的做法。
      ② ⚠️ 但它也说明：这条"确定性重放"的链路，
         在设计上是【可以把整份会话日志送回服务端的】。
      ⟹ ★★ 自己部署时，这是必须自己拍板的一个开关。
```

```
   ⟹ 🔑 回到复现性这个老问题，现在的结论比原来精确了：

      ✅ 环境可复现  ⟹ SWE-agent 用一次性容器
      ✅ 过程可复现  ⟹ DSec 轨迹日志 / dsh 的 append-only session log
      ⚠️ 但【他们的那次跑分】仍然不可复现：
         报告没公开任何一条轨迹，dsh 里也没有跑分驱动脚本。
   ⟹ ★★★ 🔑 能复现 ≠ 已公开。这一条【开源之后依然成立】。
```

---

## 7.8 那这套框架跑出来是多少分？

✅ 报告 Table 6，Agentic（agent 能力）一段，**SWE Verified — Resolved（解决率 %）**：

| Opus-4.6 Max | GPT-5.4 xHigh | Gemini-3.1-Pro High | K2.6 Thinking | GLM-5.1 Thinking | **DS-V4-Pro Max** |
|---|---|---|---|---|---|
| **80.8** | — | 80.6 | 80.2 | — | **80.6** |

同表 **Terminal Bench 2.0 — Acc（准确率 %）**：

| Opus-4.6 | GPT-5.4 | Gemini-3.1-Pro | K2.6 | GLM-5.1 | **DS-V4-Pro** |
|---|---|---|---|---|---|
| 65.4 | **75.1** | 68.5 | 66.7 | 63.5 | 67.9 |

```
   ⚠️⚠️ ★★★ 这张表【必须】配一句警告才能读：

   ✅ 报告只说了【DeepSeek 自己】用的是内部框架、2 个工具、
      500 步、512K 上下文。
   ⚠️ 它【没有说】Opus-4.6 那个 80.8 是在什么脚手架下跑的。

   ⟹ 🔑 ★★★ 这一栏的可比性，取决于一个报告没交代的前提：
      【别家的数字，是他们自己报的，还是 DeepSeek 用同一套框架重跑的？】

   ⟹ ★★ 对照一下：报告在【长上下文】那一段是明说了的（✅ 逐字）——
      "We re-evaluate Claude Opus 4.6 and Gemini 3.1 Pro on these tasks with the
       goal of standardizing the configuration across all models."
      ⟹【我们在这些任务上重新评测了 Claude Opus 4.6 和 Gemini 3.1 Pro，
         目的是统一所有模型的配置。】
   ⟹ ★★★ 长上下文那段【明确声明了统一口径】，
      agent 那段【没有同样的声明】。
      ⚠️ 这个差别，读者应该自己去原文确认，而不是替它假设。

   ⟹ 🔑🔑 ★★★ 开源改变了什么？改变了【一半】：
      ✅ 现在你可以自己 pip install deepseek-harness-sdk，
         用 profile="sdk-minimal" 把 SWE-bench 跑一遍。
      ⚠️ 但你复现出来的数字如果对不上，你仍然分不清是因为
         ① 模型版本/effort 档位不同 ② 步数上限等驱动层参数不同
         ③ 题目集口径不同 —— 因为②③ 依然没有公开。
```

---

## 7.9 ⭐ "内部框架 = 自己给自己出题"的质疑，开源之后怎么看

★★★ 报告在方法论上最有意思的一手（✅ 逐字）：

> "It is worth noting that DeepSeek-V4-Pro performs well on MCP Atlas and Toolathlon
> — two evaluation test sets that include a wide range of tools and MCP services —
> indicating that our model has excellent generalization capability and does not
> perform well only on internal frameworks."

⟹ **值得注意的是，DeepSeek-V4-Pro 在 MCP Atlas 和 Toolathlon 上表现良好 —— 这两个测试集包含大量工具和 MCP 服务 —— 这表明我们的模型具有出色的泛化能力，而不是只在内部框架上表现好。**

```
   ★ 名词：MCP（Model Context Protocol，模型上下文协议）
     ⟹ 一套让 agent 接入外部工具/数据源的通用协议。
     ⟹ ✅ dsh 自己也内置了 MCP 客户端：packages/mcp/mcp-client。

   ⟹ 🔑🔑 ★★★ 这句话的结构是一个【自辩】：
   ① 潜在质疑：★★"你用自研框架跑分，是不是模型被调成只会用你那两个工具了？"
   ② 报告的回应："我们在别人的、工具很多的基准上也不差。"

   ⟹ ★★ 方向是对的：在陌生工具集上仍然强，确实是泛化的证据。
```

**★★★ 而开源这件事，对这个质疑的回应力度，比那句自辩强得多 —— 但仍然不是完整回应：**

```
   ✅ 开源【真正解决】的部分：
      ① 双工具跑分档位可以被任何人拿去跑别的模型
         ⟹ 🔑 这是关键：sdk-minimal 里的模型适配器是可换的插件，
            换掉 llm-deepseek 那一行就能跑别家模型。
         ⟹ ★★★ 也就是说，"用同一套公开脚手架把自己和对手都重跑一遍"
            这件事，现在【外部第三方可以做】了。
      ② 工具的确切 schema、bash 的确切描述文本、编辑器的确切语义，
         全部可读 —— 原来这些是黑盒。

   ⚠️ 开源【没有解决】的部分：
      ① 报告没说跑分用的就是这份代码（7.0 那条边界）。
      ② 跑分驱动脚本（500 步、题目集、重试策略）不在仓库里。
      ③ DSec 沙箱不在仓库里。
      ④ ⚠️ 仓库不接受外部 PR，且是 developer preview
         ⟹ ★★ 意味着"你今天复现出来的口径，下个版本可能就变了"。

   ⟹ ★★★ 所以我原来那句评论要改一个字：
      原来我说"真正干净的做法应该是用同一套公开脚手架重跑"——
      ⟹ 🔑 现在这套公开脚手架【存在了】，
         但【做这件事的人还得是社区，不是 DeepSeek 自己】。
   ⟹ ⚠️ 这是我的评论，不是报告或仓库的表述。
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
   ⟹ ★★ 而且报告选择报【低的那个】以保持一致性 —— 这一点我认为值得肯定。
```

---

## 7.10 和 [评测入门 10 章](../评测入门/10-DeepSeek的评测口径.md) 的分工

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
   │   来源：DeepSeek-V4 报告 ＋ deepseek-harness 源码          │
   │   角度：★★★【agent 脚手架】—— 工具数、步数、上下文、    │
   │        沙箱基底、思考轨迹保留、插件架构                    │
   │   关于 SWE-bench 的结论：                                 │
   │     ✅ V4 换成了【自研 code agent 框架】，2 工具/500 步    │
   │     ✅ 且这套脚手架（或其后代）现在可以逐行读              │
   └──────────────────────────────────────────────────────────┘

   ⟹ 🔑🔑 ★★★ 把两章接起来，你会看到一条演进线：

      V3（Agentless，不用 agent 循环）  ⟹ 42.0
      V4（自研 agent 框架，500 步）     ⟹ 80.6

   ⚠️⚠️ 但【绝对不能】把这两个数直接相减说"进步了 38.6 分"：
      ① 模型换了（V3 → V4）
      ② 脚手架换了（Agentless → 自研 agent 循环）
      ③ ⚠️ 题目集是否完全一致，需要回原文核对
   ⟹ ★★★ 三个变量同时变了，这个差值【不归因于任何单一原因】。
   ⟹ 🔑 这正是 08 章要讲的核心问题。

   ⟹ ★★ 补一句演进线上的第三段：
      V3 用 Agentless（无 agent 循环）⟹ V4 自研 agent 循环 ⟹ 把循环开源。
      🔑 三步走完，DeepSeek 从"评测口径披露最细的一家"
         变成了"脚手架实现披露最细的一家"。
```

---

## 7.11 本章总结

```
    🔑 十条带走：

    ① ★★★ ✅ V4 报告披露了跑分那次的四个参数：
       只有 2 个工具（bash + 文件编辑）、500 步上限、512K 上下文、
       内部自研框架。⟹ ⚠️ 原来这里写"不开源"，现在这句话作废。

    ② ★★★ ✅ deepseek-harness（dsh）已开源，MIT，Node.js，
       everything-is-a-plugin，packages/ 下 257 个包。
       ⟹ DeepSeek 在本目录的可查证度从【最底档】升到【最高档】。

    ③ ⚠️⚠️ ★★★ 但报告【没说】跑分用的就是 dsh，
       dsh 里也零处提及 SWE-bench。
       ⟹ 🔑 高度吻合 ≠ 同一份代码。本章所有对照都按这条边界写。

    ④ ★★★ ✅ 报告的"minimal set of tools"在仓库里有实体：
       profile `sdk-minimal`，官方自称 "two-tool profile"，
       只有 bash-persistent（300 秒超时）+ str_replace_editor
       （输出上限 16000 字符），系统提示词只有一句话，
       compaction / subagent / skills / 遥测【全部拆掉】。
       ✅ 根目录 BENCHMARK.md 就指向这个 profile。

    ⑤ ★★★ 🔑 2 vs 62：同一个仓库里，工具目录有 62 个 model-visible
       工具，跑分档位只留 2 个。
       ⟹【工具数量不是技术信仰，是按场景拨的旋钮。】

    ⑥ ★★★ ⚠️ "500 步"在 dsh 里【找不到对应开关】
       （maxSteps 全仓零命中，只有 maxParallelToolCalls: 10）。
       ⟹ 🔑 它属于【跑分驱动脚本】那一层。
          ★★★ 开源了脚手架 ≠ 能复现跑分。

    ⑦ ★★★ 🔑 全目录最硬的一条证据仍然成立，而且现在有实现侧了：
       ✅ 报告：工具结果若被"伪装成用户消息"回传，思考轨迹保留特性
          不触发，官方建议这种情况别用思考模型。
       ✅ 源码：官方适配器把"上一回合的推理内容原样回传，无论
          那一回合有没有调工具"做成了无条件行为。
       ⚠️ 代价源码也写了：每个推理回合都追加，影响提示词缓存命中。

    ⑧ ⚠️⚠️ ★★★ 开源的是脚手架，【不是 DSec】。
       ✅ 报告的 DSec：四种基底（函数调用/容器/microVM/fullVM）+ 3FS。
       ✅ dsh 的沙箱：同机进程级（bwrap/Landlock/Seatbelt/Windows ACL），
          README 自己写明 "same-world only"，失败即关闭。
       ⟹ 🔑【开源了"模型怎么被驱动"，没开源"命令跑在哪里"。】

    ⑨ ★★★ ✅ DSec 的"确定性重放"，在 dsh 里以架构不变量存在：
       "Model-visible means logged"（模型可见即已记录）。
       ⚠️ 但同一套日志也能通过 dsh_session_log 字段回传服务端
          （默认关闭，文档主动列出了暴露面）—— 自部署时要自己拍板。

    ⑩ ⚠️ ★★★ Table 6 的横向对比，报告在【长上下文】那段声明了
       统一口径，在【agent】那段没有同样的声明。
       ⟹ 开源之后，"用同一套公开脚手架把各家重跑一遍"这件事
          第一次变得可做 —— 🔑 但做的人得是社区，
          而且仓库是 developer preview、不收外部 PR。
```

---

## 附：本章新增引用的源码位置速查

| 主题 | 文件 |
|---|---|
| 插件架构 / 不变量 / 回合与步的流程 | `docs/architecture.md` |
| 62 个工具的完整 schema | `docs/tool-catalog.md` |
| 双工具跑分档位（完整配置树） | `packages/bundle/sdk-minimal/cordis.patch.yml` + 同目录 `README.md` |
| 跑分入口 | `BENCHMARK.md` → `docs/user/guide/python-sdk.md` |
| 唯一的 agent 循环 | `packages/core/agent-loop/README.md` |
| 思考轨迹回传 / reasoningEffort 四档 | `packages/llm/llm-deepseek/README.md` |
| 私有请求扩展字段（含会话日志回传） | `docs/deepseek-llm-api-wire-extensions.md` |
| 沙箱边界 | `packages/sandbox/sandbox/README.md`、`packages/sandbox/sandbox-local/README.md` |
| 远程沙箱（默认关闭） | `packages/e2b/*/README.md` |
| 三种上下文压缩策略 | `packages/compaction/{compaction-basic,compaction-tool-result-pruner,command-compact}/README.md` |
| 凭据四层 / 各 profile 的 CLI 行为 | `apps/cli/reference/README.md` |
| 凭据层次与 `.credentials.yaml` 格式 | `packages/credentials/credentials-local/README.md` |
| baseURL 与 key 的实际解析顺序（⚠️ 与 README 表格矛盾，以此为准） | `packages/llm/llm-deepseek/src/index.ts:378-380`、`:433-447` |
| 缺省 provider / model | `packages/bundle/base/cordis.patch.yml:75-79` |
| 不收外部 PR 的说明 | `CONTRIBUTING.md` |

---

> 上一章：[06 Claude Code 与 Codex 逐项对照](06-ClaudeCode与Codex逐项对照.md) ｜ 下一章：[08 脚手架怎么影响榜单分数](08-脚手架怎么影响榜单分数.md)
