# 06 · Claude Code 与 Codex 逐项对照

> ⭐⭐ 这一章不讲原理，只做一件事：
> **把两个产品【公开可查】的部分，摆进同一张表。**
>
> ⚠️ 全章纪律：写进表里的每一项，都必须是官方文档里能查到的
> 名字 / 取值 / 路径。**看不到的实现，一律标 ⚠️ 或直接留空。**

---

## 6.0 先说清楚这一章能做到什么、做不到什么

```
   ┌────────────────────────────────────────────────────────────┐
   │ ✅ 能做到：                                                 │
   │   对照【配置面】——模式名、配置键、文件路径、工具名。      │
   │   ⟹ 这些是产品对外承诺的接口，写在文档里，可以逐字引用。  │
   ├────────────────────────────────────────────────────────────┤
   │ ❌ 做不到：                                                 │
   │   ① 对照【系统提示词】——两家都不公开                      │
   │      ✅ Claude Code 文档原话："Claude Code's system prompt │
   │         isn't published." ⟹ 系统提示词不公开。            │
   │   ② 对照【压缩算法】【分类器怎么判】【工具内部实现】      │
   │   ③ 对照【谁在 SWE-bench 上更强】——见 08 章，公开提交    │
   │      记录不注明用的是哪个档位                              │
   └────────────────────────────────────────────────────────────┘

   ⟹ 🔑 ★★ 所以本章的用法是：
      【当你要在两者之间做技术选型时，这张表告诉你各自把哪些
       旋钮交到了你手上。】不是"谁更强"的排行榜。
```

---

## 6.1 开源程度 ★★★

| | Claude Code | Codex |
|---|---|---|
| 主程序 | ⚠️ 不开源 | ✅ 开源：`openai/codex` |
| 官方定位 | — | ✅ "The primary home for Codex open-source development" ⟹ Codex 开源开发的主仓库 |
| SDK | ✅ Claude Agent SDK（公开） | ✅ `openai/codex/codex-sdk` |
| 技能 / 插件 | ✅ 机制公开、可自建 | ✅ `openai/skills`、`openai/plugins` |
| 云端环境 | ⚠️ 不开源 | ✅ `openai/codex-universal`（云端基础环境）；⚠️ **Codex cloud 本身**文档标注 "Not open source" |
| IDE 扩展 | ⚠️ 不开源 | ⚠️ 文档标注 "Not open source" |

```
   ⟹ 🔑 ★★★ 这是两者最大的结构性差异：
      Codex 的 CLI（命令行程序）本体是开源的，你可以读源码；
      Claude Code 的本体不开源，只能读文档。

   ⚠️ 但请注意【开源的是壳，不是脑】：
      两家的【模型】都不开源。开源的只是那层脚手架。
      ⟹ ★★ 也就是说，就算你读了 codex 全部源码，
         你仍然不知道 GPT-5 是怎么被训成会用这些工具的。
```

**✅ 关于"Codex 是 Rust 写的"这个说法：已一手核实**

```
   ✅ 核实方式（2026-08-27）：直接取仓库里的构建清单文件

     curl https://raw.githubusercontent.com/openai/codex/main/codex-rs/Cargo.toml

   返回内容开头：

     [workspace]
     members = [
         "aws-auth", "analytics", "agent-graph-store", "agent-identity",
         "agent-roles", "backend-client", "bwrap", "build-info",
         "ansi-escape", "async-utils", "app-server",
         "app-server-transport", ...
     ]

   ★ 术语：Cargo 是 Rust 的官方构建工具与包管理器
          （名字来自"货物"，没有缩写含义）；
          Cargo.toml 是它的清单文件，只有 Rust 项目才有。
   ★ 术语：workspace（工作区）= 一个仓库里放多个互相依赖的 Rust 子包，
          members 就是子包列表。

   ⟹ ★★ 结论：✅ 【Codex CLI 本体确实是 Rust 写的，且是一个多包工作区。】

   ⚠️ 顺带一个观察：members 里有一个子包直接叫 `bwrap`
      ⟹ 呼应 [05 章](05-权限与沙箱.md)：bubblewrap 是 Linux 上的沙箱工具，
         Codex 把对它的封装单独做成了一个 Rust 子包。
      ⚠️ 但"有这个子包"只能说明它用到了 bubblewrap，
         不能推断它在 Linux 上的默认沙箱就是 bubblewrap —— 这一层我没核实。
```

> ★ 备注：GitHub 的**语言统计接口**（`api.github.com/repos/.../languages`）
> 在未登录时限额只有 **60 次/小时**，用完就返回 403。
> ⟹ 绕开办法：`raw.githubusercontent.com` 取具体文件**不占这个限额**。

---

## 6.2 配置文件 ★★

| | Claude Code | Codex |
|---|---|---|
| 格式 | ✅ **JSON** | ✅ **TOML** |
| 主文件 | ✅ `settings.json` | ✅ `config.toml` |
| 用户级 | ✅ `~/.claude/settings.json` | ✅ `~/.codex/config.toml` |
| 项目级 | ✅ `.claude/settings.json`（提交进 git 共享） | ✅ `.codex/`（项目作用域） |
| 个人不提交的那份 | ✅ `.claude/settings.local.json` | — |
| 组织强制 | ✅ managed settings（托管设置，个人改不掉） | ✅ 有企业侧控制 |
| 主目录环境变量 | `CLAUDE_CONFIG_DIR` | ✅ `CODEX_HOME`（默认 `~/.codex`） |

```
   ★ 名词：JSON / TOML
     ⟹ 两种配置文件格式。JSON 是 {"a": {"b": 1}} 这种嵌套花括号；
        TOML 是 [section] 下面 key = value 这种分节形式。
     ⟹ ★★ 纯口味差异，不影响能力。

   ★ 名词：managed settings（托管设置）
     ⟹ 由公司 IT 下发、放在系统目录里的配置。
     ★★★ 关键性质：【个人设置覆盖不了它】。
```

★★ Claude Code 有一条我认为很值得注意的行为（✅ 文档逐字）：

> "Claude Code watches your settings files and reloads them when they change, so it
> applies most edits to the running session without a restart"

⟹ **Claude Code 会监视你的设置文件并在变更时重载，所以大多数编辑会应用到正在运行的会话，无需重启。**

---

## 6.3 项目说明文件 ★★★

★★ 这是"你怎么把项目规矩告诉 agent"的入口，两家做法差别很大。

| | Claude Code | Codex |
|---|---|---|
| 文件名 | ✅ `CLAUDE.md` | ✅ `AGENTS.md` |
| 何时读 | ✅ 每次会话开始 | ✅ "Codex reads `AGENTS.md` files before doing any work." ⟹ 干任何活之前读 |
| 全局层 | ✅ `~/.claude/CLAUDE.md` | ✅ `~/.codex/`（或 `CODEX_HOME`）下的 `AGENTS.override.md`，回退到 `AGENTS.md` |
| 项目层 | ✅ `./CLAUDE.md` 或 `./.claude/CLAUDE.md` | ✅ 从项目根（通常是 Git 根）**逐级向下走到当前目录** |
| 个人层 | ✅ `./CLAUDE.local.md`（加进 .gitignore） | ✅ 每层先试 `AGENTS.override.md` 再试 `AGENTS.md` |
| 组织层 | ✅ 托管策略路径（如 macOS `/Library/Application Support/ClaudeCode/CLAUDE.md`），★★★ 个人**不能**排除 | — |
| 合并方式 | ✅ "All discovered files are concatenated into context rather than overriding each other." ⟹ 全部拼接，不互相覆盖 | ✅ "Codex concatenates files from the root down, joining them with blank lines." ⟹ 从根往下拼接，用空行连接 |
| 谁优先 | ✅ 从文件系统根往下排，**越靠近工作目录的越晚被读到** | ✅ "Files closer to your current directory override earlier guidance because they appear later in the combined prompt." ⟹ 越近的越靠后，因而覆盖前面的 |
| 大小上限 | ✅ 建议每个文件 200 行以内；超过 4 MiB 直接跳过 | ✅ `project_doc_max_bytes`，**默认 32 KiB**，到上限就不再加文件 |
| 引入其他文件 | ✅ `@path/to/file` 语法，最多递归 4 层 | ✅ `project_doc_fallback_filenames`（备选文件名列表） |

```
   ⟹ 🔑🔑 ★★★ 两家的合并规则【一模一样】：
      ①【拼接不覆盖】 ②【越具体的越靠后】
   ⟹ ★★ 为什么"靠后 = 优先"？因为模型读的是一段线性文本，
      后面的话在语境上自然覆盖前面的话。
      ⟹ 这不是代码逻辑，这是【提示词工程】。

   ⚠️ 注意一个实际差异：
      Codex 明确给了 32 KiB 的硬上限并会【截断】；
      Claude Code 给的是"建议 200 行"（软），硬线在 4 MiB。
```

★★★ 关于跨产品互通，Claude Code 文档专门写了一段（✅ 逐字）：

> "Claude Code reads `CLAUDE.md`, not `AGENTS.md`. If your repository already uses
> `AGENTS.md` for other coding agents, create a `CLAUDE.md` that imports it so both
> tools read the same instructions without duplicating them."

⟹ **Claude Code 读 `CLAUDE.md`，不读 `AGENTS.md`。如果你的仓库已经为其他编码 agent 用了 `AGENTS.md`，那就建一个 `CLAUDE.md` 去引入它，这样两个工具读同一份指令而不必重复写。**

```
   ⟹ ★★ 文档给的两种写法：
      写法① 在 CLAUDE.md 第一行写：@AGENTS.md
      写法② 建符号链接：ln -s AGENTS.md CLAUDE.md
      ⚠️ Windows 上做符号链接需要管理员权限，用写法①。

   ⟹ 🔑 ★★★ 这段话本身就是一条情报：
      【AGENTS.md 事实上正在成为跨产品的通用格式。】
      Anthropic 在自己的文档里教你怎么兼容它。
```

---

## 6.4 许可与沙箱 ★★★（05 章详版，这里只做并排）

| 维度 | Claude Code | Codex |
|---|---|---|
| 许可/审批档位 | ✅ **6 档**：`default`(Manual) / `acceptEdits` / `plan` / `auto` / `dontAsk` / `bypassPermissions` | ✅ **3 档**：`approval_policy` = `untrusted` / `on-request` / `never` |
| 沙箱档位 | ✅ **2 档**：auto-allow / regular permissions | ✅ **3 档**：`sandbox_mode` = `read-only` / `workspace-write` / `danger-full-access` |
| 模型审模型 | ✅ **classifier**（分类器），auto 模式的核心 | ✅ `approvals_reviewer = "auto_review"` ⟹ 交给 reviewer agent |
| macOS 机制 | ✅ Seatbelt | ✅ Seatbelt |
| Linux/WSL2 机制 | ✅ `bubblewrap` + `socat`（+ 可选 seccomp） | ✅ `bubblewrap`；"uses the first `bwrap` executable it finds on `PATH`" |
| 原生 Windows | ⚠️ **不支持**，"Native Windows is not supported."，需走 WSL2 | ✅ **有原生 Windows 沙箱**（PowerShell 下） |
| 扩大可写范围 | ✅ `sandbox.filesystem.allowWrite` | ✅ `sandbox_workspace_write.writable_roots` |
| 网络默认 | ✅ **一个域名都不预先放行**，首次访问时问 | ✅ 有允许清单机制 |
| 网络配置键 | ✅ `sandbox.network.allowedDomains` / `deniedDomains` / `strictAllowlist` | ✅ 企业侧 "Allow public internet access" 开关 |

```
   ⟹ 🔑 ★★★ 一句话概括这张表：
      【两家在同一片沙上盖房子（Seatbelt / bubblewrap 都是现成的
       操作系统能力），区别在于把多少个旋钮露出来给你。】

   ⟹ ⚠️ 唯一的能力级差异是 Windows：
      Claude Code 必须走 WSL2，Codex 有原生方案。
```

---

## 6.5 上下文管理 ★★★（04 章详版）

| | Claude Code | Codex |
|---|---|---|
| 自动压缩 | ✅ auto-compact | ✅ `model_auto_compact_token_limit` |
| 手动触发 | ✅ `/compact`，可带指令 `/compact <重点>` | ⚠️ 未在我查到的文档中列出等价命令 |
| 阈值口径 | ⚠️ 未公开数值 | ✅ `model_auto_compact_token_limit_scope` = `total` \| `body_after_prefix` |
| 换压缩提示词 | ✅ `CLAUDE.md` 里的 `# Compact instructions` | ✅ `compact_prompt` / `experimental_compact_prompt_file` |
| 单条工具输出上限 | ⚠️ 未公开配置键 | ✅ `tool_output_token_limit` |
| 清空重开 | ✅ `/clear` | ⚠️ — |
| 查看占用 | ✅ `/context` | ⚠️ — |
| 压缩前后钩子 | ✅ `PreCompact`（Claude Code 也有） | ✅ `PreCompact` / `PostCompact` |
| 窗口大小配置 | ⚠️ — | ✅ `model_context_window` |

```
   ⟹ ★★ 这一格上，Codex 暴露的【数值旋钮】更多（阈值、口径、
      单条输出上限都能调）；
      Claude Code 暴露的【交互命令】更多（/compact /clear /context）。

   ⟹ 🔑 ⚠️ 我的解读：
      这反映了两个产品预设的用户不一样——
      Codex 更像给【脚本和 CI】用的（写在配置文件里，一次配好）；
      Claude Code 更像给【坐在终端前的人】用的（随时敲斜杠命令）。
      ⚠️ 这是我的推断，不是任何一家的官方表述。
```

---

## 6.6 子 agent ★★★

| | Claude Code | Codex |
|---|---|---|
| 叫法 | ✅ subagent | ✅ subagent，工作线程叫 **agent thread**（agent 线程） |
| 官方定义 | ✅ "Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions." | ✅ "A delegated agent that Codex starts to handle a specific task." ⟹ Codex 启动来处理某个特定任务的受委派 agent |
| 上下文隔离 | ✅ 明说 "fresh, isolated context window"，且看不到主对话历史 | ✅ 强调返回**摘要**而非原始中间输出 |
| 定义在哪 | ✅ `.claude/agents/`（Markdown + YAML 前置元数据） | ✅ `~/.codex/agents/`（个人）或 `.codex/agents/`（项目），**每个 agent 一个 TOML 文件** |
| 必填字段 | ✅ name / description | ✅ `name`、`description`、`developer_instructions` |
| 内置角色 | ✅ 有内置类型（如通用型、探索型） | ✅ **三个内置**：`default`（通用兜底）/ `worker`（执行实现和修复）/ `explorer`（读多的代码库探索） |
| 并发上限 | ⚠️ 未公开数值 | ✅ `agents.max_concurrent_threads_per_session` |
| 指定模型 | ✅ 可为子 agent 指定模型 | ✅ `agents.default_subagent_model` |
| 指定思考强度 | ⚠️ — | ✅ `agents.default_subagent_reasoning_effort` |
| 总开关 | ⚠️ — | ✅ `agents.enabled`（默认 `true`） |
| 子 agent 的沙箱 | ✅ "independent permissions" ⟹ 独立许可 | ✅ 每个 agent 文件可写 `sandbox_mode` |

★★★ 两家都主动写了**代价**，我认为这一点比功能本身更值得读：

> ✅ Claude Code：**Warning** — "Running many subagents that each return detailed
> results can consume significant context."
> ⟹ 跑很多子 agent、每个都返回详细结果，一样会吃掉大量上下文。

> ✅ Codex："Because each subagent does its own model and tool work, subagent
> workflows consume more tokens than comparable single-agent runs."
> ⟹ 因为每个子 agent 都在做自己的模型和工具工作，子 agent 工作流比同等的单 agent 运行消耗更多 token。

```
   ⟹ 🔑🔑 ★★★ 两家用不同的话说了同一件事：
      【隔离不是免费的，它把成本从"上下文"挪到了"token 总量"。】

   ⟹ ★★ 回到 04 章的三种策略：
      截断省钱但丢信息；压缩保信息但要多花一次调用；
      隔离保主对话干净但【总花费更高】。
      ⟹ 没有免费午餐，只有【把成本挪到哪里】。
```

---

## 6.7 其他扩展点 ★★

| | Claude Code | Codex |
|---|---|---|
| 钩子（hooks） | ✅ 有，事件包括 `PreToolUse` / `PreCompact` / `ConfigChange` / `InstructionsLoaded` 等 | ✅ 有，✅ "Run scripts or MCP tools during the Codex lifecycle" ⟹ 在 Codex 生命周期中运行脚本或 MCP 工具 |
| MCP | ✅ 支持 | ✅ 支持，文档页 `/docs/extend/mcp.md` |
| 技能（skills） | ✅ 有，按需加载 | ✅ 有，仓库 `openai/skills` |
| 插件（plugins） | ✅ 有 | ✅ 有，仓库 `openai/plugins` |
| 斜杠命令 | ✅ 有 | ✅ 有（⚠️ 但 "Custom Prompts" 页已标注**弃用**，官方引导改用 skills） |
| 非交互模式 | ✅ `claude -p "<提示>"` | ✅ `codex exec`（文档页 "Non-interactive mode"） |
| 自动记忆 | ✅ **auto memory**，Claude 自己写笔记到 `~/.claude/projects/<项目>/memory/` | ⚠️ 未查到等价机制 |

```
   ★ 名词：hook（钩子）
     ⟹ 在固定的生命周期节点上自动执行的脚本。
     ★★★ 关键性质：✅ Claude Code 文档原话——
        "Hooks execute as shell commands at fixed lifecycle events and apply
         regardless of what Claude decides."
        ⟹【钩子在固定生命周期事件上作为 shell 命令执行，
           不管 Claude 决定做什么，它都生效。】
     ⟹ 🔑 ★★ 对比 CLAUDE.md：
        ✅ "CLAUDE.md instructions shape Claude's behavior but are not a hard
           enforcement layer." ⟹ 塑造行为，但不是硬性执行层。
        ⟹ ★★★ 想【建议】它 → 写 CLAUDE.md；
           想【强制】它 → 写 hook。这两者不能互相替代。

   ★ 名词：MCP（Model Context Protocol，模型上下文协议）
     ⟹ 一个开放协议，让第三方把自己的工具接进 agent。
     ⟹ ★★ 两家都支持同一个协议 = 工具生态是【互通】的。
```

---

## 6.8 有一件事这张表故意没做

```
   ⚠️⚠️ ★★★ 我【没有】在这一章比较"谁在 SWE-bench 上分更高"。

   ⟹ 理由（详见 08 章）：
      ① 两家的官方分数往往用【不同的题目子集】
         （SWE-bench Verified / Lite / full 不是一回事）
      ② ⚠️ 都不注明跑分时用的许可档位和预算上限
      ③ ⚠️ 都不注明模型版本和脚手架版本的对应关系
      ④ ★★★ 而 03/04 章已经证明：
         【同一个模型，换脚手架能差 7.7 个点。】
      ⟹ 所以"A 产品 x 分 vs B 产品 y 分"这个比较，
         在缺少上述三项信息时【是没有意义的】。

   ⟹ 🔑 ★★ 这就是本目录 README 里那条纪律的实际后果：
      能查证的，逐字写；查不到的，说清楚查不到。
      【不用一个看起来精确的数字，去填一个实际上是空的格子。】
```

---

## 6.9 本章总结

```
    🔑 七条带走：

    ① ★★★ ✅ 结构性差异：Codex 的 CLI 本体开源（openai/codex），
       Claude Code 不开源。⚠️ 但两家的【模型】都不开源。

    ② ✅ Codex CLI 本体用 Rust 写，已一手核实：
       `codex-rs/Cargo.toml` 是一个 Rust 多包工作区清单。

    ③ ★★ 配置：JSON/settings.json vs TOML/config.toml，纯口味差异。

    ④ ★★★ 项目说明：CLAUDE.md vs AGENTS.md，
       但【合并规则完全一致】：拼接不覆盖、越具体越靠后。
       ✅ 且 Claude Code 官方教你用 @AGENTS.md 兼容。

    ⑤ ★★★ 许可/沙箱：底层机制几乎相同（Seatbelt / bubblewrap），
       差异只在露出多少旋钮。⚠️ 唯一能力差：原生 Windows。

    ⑥ ★★★ 子 agent：两家都主动写明【隔离要多花 token】。
       ⟹ 没有免费午餐，只有把成本挪到哪里。

    ⑦ ★★★ 想【建议】模型 → CLAUDE.md / AGENTS.md；
       想【强制】模型 → hook。✅ 官方原话区分了这两层。
```

---

> 上一章：[05 权限与沙箱](05-权限与沙箱.md) ｜ 下一章：[07 DeepSeek 的脚手架](07-DeepSeek的脚手架.md)
