# slime 拆解：GLM 系列背后的 RL 框架

> **这一篇干什么**：源码级走查。对标 [训练框架/Megatron-LM拆解.md](../训练框架/Megatron-LM拆解.md) 的写法。
> **难度** ★★
> **✅ 本文所有行号钉在 commit `3778dbf`（tag `v0.3.2`，2026-08-28）**。行号会随版本漂移，换版本请重新定位。

---

## 1 · 规模实测

✅ 本地 clone 后实测：

```
整仓：332 个 .py 文件 · 72,468 行 · 18 MB

顶层：
  train.py             99 行   ← 同步训练主循环
  train_async.py       81 行   ← 异步训练主循环
  slime/                       ← 主包，152 文件 / 35,708 行
  slime_plugins/  examples/  tests/  tools/  scripts/  docs/
```

`slime/` 内部：

| 目录 | 文件数 | 行数 | 干什么 |
|---|---|---|---|
| `backends/` | 68 | **18,322** | 训练与推理引擎的适配 |
| `utils/` | 30 | 7,900 | 工具 |
| `rollout/` | 21 | 3,015 | 生成流程 |
| `agent/` | 13 | **2,727** | agent 与沙箱（见 [06 章](06-AgenticRL-环境与沙箱.md)） |
| `observability/` | 12 | 2,516 | 日志、追踪、profiling |
| `ray/` | 7 | **1,228** | Ray 编排 |

★ `backends/` 内部有一个很说明问题的不对称：

```
slime/backends/megatron_utils/   16,220 行   ← 训练侧
slime/backends/sglang_utils/      2,101 行   ← 推理侧
                                  ────────
                                  7.7 倍
```

⚠️ **接一个训练引擎，比接一个推理引擎难 7.7 倍。**
原因不难理解：推理引擎是个服务，接口是「发请求、收 token」；训练引擎要处理并行切分、优化器状态、checkpoint、梯度累积——而且这些都要和权重同步对上。这也解释了为什么开源框架里「多推理后端」比「多训练后端」常见得多。

**对照组**（✅ 见 [01 章 §1.2](01-格局全景.md)）：veRL 171,584 行 · slime 72,468 行 · OpenRLHF 13,411 行。

---

## 2 · ✅ 设计哲学：明确拒绝抽象

slime 是本目录里**唯一把设计取舍写在 README 第一屏**的框架。这些不是我的归纳，是它自己的原话：

### ① 只接一个推理后端，是为了不丢特性

> ✅ *"By choosing **one** rollout backend, slime can use SGLang-specific capabilities directly instead of **flattening multiple inference engines into a lowest-common-denominator abstraction**."*
> （选定单一 rollout 后端，slime 就能直接用 SGLang 的专属能力，而不是把多个推理引擎压平成一个最小公分母的抽象。）

⚠️ 这句话是对 veRL 那条路线的正面回应。veRL 接了 4 个推理引擎，代价就是：任何一个引擎的新特性，都要先在抽象层里找到位置才能用上。slime 用「少一个自由度」换「零延迟接入 SGLang 新能力」。

### ② 参数直通（pass-through），不做包装

> ✅ *"every argument supported by the installed SGLang can be used by adding the `--sglang-` prefix"*
> ✅ *"slime reads Megatron arguments directly, so Megatron-side parallelism, optimizer, checkpointing, and model options remain available **without wrapper code**"*

⚠️ 这是「拒绝抽象」的具体落地：slime 不重新定义参数体系，SGLang 有什么参数，加个 `--sglang-` 前缀就能用。**上游升级了新功能，slime 一行代码都不用改就能用上。**

### ③ 环境「不 fork 训练内核」

> ✅ *"math, code, search, tools, sandboxes, verifiers, environments, multi-agent systems, and long-horizon agentic workflows plug in as **data generation or reward workflows**. They **do not fork the training kernel**."*

⚠️ 这和 ✅ GLM-5 报告里那句 *"cleanly isolates task-specific logic from the core training loop"*（见 [06 章 §6.2](06-AgenticRL-环境与沙箱.md)）是**同一条原则的两处表述**——一处在论文里，一处在 README 里。**报告与源码互相印证。**

### ④ 一句值得抄下来的话

> ✅ *"**RL bugs are often silent.** slime keeps the dataflow explicit, supports separate rollout-only and train-only debugging paths."*

⚠️ 「RL 的 bug 通常是沉默的」——这正是 [04 章](04-训推一致性.md) 开头讲的那件事。而 slime 的应对是**工程手段**：把 rollout 和 train 拆成可以独立调试的两条路径，出问题时能二分定位。

---

## 3 · ★★ 三个模块：training / rollout / data buffer

✅ README 的 Architecture Overview 只列了三个模块：

```
┌──────────────────────────┐        ┌──────────────────────────┐
│                          │        │                          │
│   training (Megatron)    │        │  rollout (SGLang+router) │
│                          │        │                          │
│ 从 Data Buffer 读数据    │        │ 生成新数据（含 reward /  │
│ 训完把参数同步给 rollout │        │ verifier 输出）          │
│                          │        │                          │
└───────────┬──────────────┘        └─────────────┬────────────┘
            │                                     │
            │  读                             写  │
            │      ┌──────────────────────┐       │
            └─────►│     data buffer      │◄──────┘
                   │                      │
                   │ 管理 prompt 初始化、 │
                   │ 自定义数据、生成方法 │
                   └──────────────────────┘
```

⚠️ **注意这里没有「weight sync」这个模块。** 权重同步在 slime 里不是一个平级模块，而是 training 模块的一个动作（`actor_model.update_weights()`）。这和 veRL 把 `checkpoint_engine/` 单独抽一层（见 [03 章 §3.3](03-权重同步.md)）是不同的组织方式。

⚠️ **data buffer 是唯一的数据桥。** ✅ README 原话：*"Custom generate functions can wrap this with multi-turn loops, tool calls, environment/sandbox interaction, and verifier-based reward"*——不管你的环境多复杂，最后都是往 data buffer 里放样本。**这就是「环境即数据生成」的架构实现。**

✅ 源码里对应的是 `slime/rollout/data_source.py`（229 行）：

```
class DataSource(abc.ABC)                  # 抽象接口：get_samples / add_samples / save / load / __len__
    └─ class RolloutDataSource             # 基础实现
         └─ class RolloutDataSourceWithBuffer   # ★ 带 buffer 的版本
              ├─ _get_samples_from_buffer()
              └─ get_buffer_length()
```

⚠️ 类名的继承关系直接对应了 [02 章 §2.2](02-架构主轴-同步到全异步.md) 的 buffer 深度：**不带 buffer 的是深度 0，带 buffer 的才有深度。**

---

## 4 · ★★★ 主循环：99 行 vs 81 行

这是 slime 最好读的地方，也是本目录里**能把「同步 vs 异步」讲得最清楚的一处源码**。两个主循环几乎一样长，差别集中在两处。

### 同步版 `train.py`（99 行）

✅ 关键行：

| 行号 | 代码 | 对应什么 |
|---|---|---|
| 14 | `pgs = create_placement_groups(args)` | 分配 GPU |
| 19 | `rollout_manager, num_rollout_per_epoch = create_rollout_manager(...)` | 建 rollout 管理器（内含 SGLang 引擎） |
| 21 | `actor_model, critic_model = create_training_models(...)` | 建训练模型 |
| 27 | `actor_model.update_weights()` | 开跑前先推一次权重 |
| **49** | `for rollout_id in range(args.start_rollout_id, args.num_rollout):` | **主循环** |
| **53** | `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))` | ★ **阻塞式生成** |
| 69 | `ray.get(actor_model.async_train(rollout_id, rollout_data_ref))` | 训练 |
| **85** | `actor_model.update_weights()` | ★ **每一步都同步权重** |

⚠️ 第 53 行的 `ray.get(...)` 是整个同步架构的全部秘密：**它在这里等着，直到所有样本生成完。** 这一等，训练卡就空转了。

### 异步版 `train_async.py`（81 行）

✅ 差别只有三处，但每一处都对应本目录的一个概念：

**差别一 · L11：异步必须分离部署**

```python
assert not args.colocate, "Colocation is not supported for async training."
```

⚠️ 一行断言，把 [02 章](02-架构主轴-同步到全异步.md) 那条「同置 vs 分离」的选择变成了硬约束：**要异步，就必须分开放。**

**差别二 · L32 / L36 / L40：一步离策略的教科书实现**

```python
# L32  循环【开始之前】就先发出第一次生成请求（不 ray.get，拿的是 future）
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # L36  等上一次的生成结果
    rollout_data_curr_ref = ray.get(rollout_data_next_future)

    # L39-40  ★ 立刻发出【下一次】生成请求，然后才去训练
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    # ... 训练当前这批（此时下一批正在生成）
```

⚠️ **这就是 [02 章 §2.2](02-架构主轴-同步到全异步.md) 的「buffer 深度 = 1」，用 8 行代码实现。**
`next_future` 这个变量名本身就是那个深度为 1 的缓冲区。同步版是「生成 → 训练 → 生成 → 训练」，异步版是「生成 N+1 ‖ 训练 N」。

**差别三 · L66-70：权重同步频率是个参数**

```python
if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
    # sync generate before update weights to prevent update weight in the middle of generation
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None
    actor_model.update_weights()
```

★ 这五行同时印证了两个前面章节的结论：

1. **`args.update_weights_interval`** = ✅ GLM-5 报告里那句 *"pushes the new weights back to the inference engine **every K gradient updates**"*（见 [02 章 §2.4](02-架构主轴-同步到全异步.md)）。**报告里的 K，在源码里就是这个参数。**

2. **那行注释** —— *"sync generate before update weights to prevent update weight in the middle of generation"*（换权重前先把生成同步掉，避免在生成中途换权重）——正是 [03 章 §3.5](03-权重同步.md) 中断模型的**第 3 档「软暂停 / 排空」**。slime 选的不是「永不停」，而是排空后再换。

⚠️ 把三处差别合起来看：**从同步走到一步离策略，slime 只改了 18 行代码。** 这和 [02 章 §2.3](02-架构主轴-同步到全异步.md) 那个观察（veRL `one_step_off_policy/` 只有 553 行，`fully_async_policy/` 却有 4,335 行）说的是同一件事——**这一档的性价比极高。**

✅ 而 L9 的注释指出真正的全异步在别处：

```python
# The framework supports other asynchronous approaches such as fully async
# (which is shown in examples/full_async).
```

✅ 对应 `slime/rollout/fully_async_rollout.py`（274 行）和 `examples/fully_async/`。

---

## 5 · Ray 编排层：1,228 行

✅ `slime/ray/` 只有 7 个文件：

| 文件 | 行数 | 作用 |
|---|---|---|
| `rollout.py` | **495** | `RolloutManager` |
| `actor_group.py` | 269 | `RayTrainGroup`，训练侧分发 |
| `placement_group.py` | 253 | GPU 资源分配 |
| `train_actor.py` | 126 | 训练 actor |
| `utils.py` | 75 | — |
| `ray_actor.py` | 10 | — |

✅ `RolloutManager`（`slime/ray/rollout.py:38`）的方法列表，把「rollout 侧要管什么」列得很全：

| 方法组 | 方法 | 说明 |
|---|---|---|
| 主流程 | `generate` (L163) · `eval` (L184) | 生成与评测 |
| **显存管理** | `offload` (L207) · `onload` (L212) · **`onload_weights`** (L216) · **`onload_kv`** (L220) | ★ 权重和 KV cache **分开**换入换出 |
| 权重 | `check_weights` (L251) | 校验训推权重一致 |
| **容错** | `recover_updatable_engines` (L224) · `health_monitoring_pause` (L243) · `health_monitoring_resume` (L247) | ★ 对应 ✅ GLM-5 的「心跳驱动容错」 |
| 数据加工 | `_post_process_rewards` (L279) · `_convert_samples_to_train_data` (L306) | 样本 → 训练数据 |
| 并行适配 | `set_train_parallel_config` (L425) · `_split_train_data_by_dp` (L428) | 按训练侧 DP 切数据 |

⚠️ 两处值得单独记：

1. **`onload_weights` 和 `onload_kv` 是分开的两个方法。** 同置模式下，训练要用显存时，推理引擎必须把权重和 KV cache 都吐出来；但恢复时**先恢复权重、再恢复 KV**——因为权重恢复完就能开始接请求了，KV 可以慢慢来。✅ `train.py` L84 和 L88 正是分两次调用的。这是个很细但很实在的优化。

2. **`health_monitoring_pause/resume`** 对应 ✅ GLM-5 §3.6.3 的 *"heartbeat-driven rollout fault tolerance and router-level server lifecycle management"*——**报告里的容错机制，在源码里能找到对应的方法名。**

---

## 6 · rollout 层：一个目录读懂 slime 支持什么

✅ `slime/rollout/`（21 文件 / 3,015 行），文件名几乎就是功能清单：

| 文件 | 行数 | 对应本目录哪一章 |
|---|---|---|
| `sglang_rollout.py` | **649** | 主力生成路径 |
| `fully_async_rollout.py` | **274** | [02 章](02-架构主轴-同步到全异步.md) 全异步 |
| `data_source.py` | 229 | [02 章](02-架构主轴-同步到全异步.md) data buffer |
| `sglang_streaming_rollout.py` | 167 | 流式生成 |
| `forge_load.py` | 114 | 负载调度 |
| `on_policy_distillation.py` | **67** | [05 章 §5.6](05-算法与框架的接口.md) OPD |
| `sft_rollout.py` | 68 | SFT 混训 |
| `sample_hooks.py` | 50 | 自定义钩子 |
| `sleep_rollout.py` | 12 | 睡眠模式 |
| `filter_hub/` · `rm_hub/` | — | 样本过滤 / 奖励模型 |

⚠️ **`on_policy_distillation.py` 只有 67 行**，这是 [05 章 §5.6](05-算法与框架的接口.md) 那个结论的最硬证据：**在策略蒸馏就是把奖励换成教师概率，其他一切照旧。** 复用了整套 rollout、buffer、权重同步，所以只需要 67 行。

⚠️ `filter_hub/` 的存在对应 [05 章 §5.5](05-算法与框架的接口.md) 讲的动态过滤，也对应 ✅ GLM-5 那条「按失败原因排除环境崩溃样本」（[06 章 §6.3](06-AgenticRL-环境与沙箱.md)）。

---

## 7 · agent 层：把真实 CLI 拉进训练

这一块在 [06 章 §6.4](06-AgenticRL-环境与沙箱.md) 已经详细讲过，这里只放结构：

```
slime/agent/  (13 文件 / 2,727 行)
├── trajectory.py        508    轨迹
├── sandbox.py           399    沙箱抽象（"intentionally small" 的接口）
├── parsing.py           114
├── aiohttp_threaded.py   98
├── adapters/                   按 API 协议适配
│   ├── common.py        523
│   ├── openai.py        378
│   └── anthropic.py     350
└── harness/                    ★ 按真实 agent CLI 适配
    ├── common.py        178
    ├── claude_code.py    86    ★ Claude Code
    └── codex.py          71    ★ Codex
```

⚠️ **`adapters/` 和 `harness/` 是两层不同的抽象**，这个区分很关键：
- `adapters/` 解决「模型怎么被调用」——OpenAI 协议还是 Anthropic 协议
- `harness/` 解决「agent 程序怎么被跑起来」——装 Node、装 npm 包、拼命令行参数

**一个 agent CLI 可以走任意一种 API 协议，所以这两层必须分开。**

---

## 8 · ✅ 生态：MILES 其实建在 slime 上

✅ slime README 有一节 *"Ecosystem Built on slime"*，列了 11 个下游项目：

Dressage · **Miles** · vime · Relax · OpenClaw-RL · P1 · RLVE · TritonForge · APRIL · qqr · ART(AWS)

⚠️ **这里有一个需要修正 [01 章](01-格局全景.md) 那张开源总表的地方**：🌐 HF 横评把 **MILES** 列为一个独立的库（radixark 出品，~950 star），✅ 而 slime 的 README 把它列在「建立在 slime 之上的生态」里。

**所以「16 个独立框架」这个数字是偏高的**——其中至少有一个是另一个的下游。⚠️ 我的判断：真正的**架构谱系**远少于框架数量，很多「新框架」是在某个基座上换了调度策略或加了环境层。选型时应该先问「它的基座是什么」，而不是把它们当成 18 个平行的选项。

（这条是本次源码走查带来的、单看联网材料看不出来的发现。）

---

## 9 · 怎么读这份源码

✅ slime README 自己给了一条阅读路径，实测和源码结构对得上：

```
train.py: train
├─ slime/ray/placement_group.py        Ray 资源与 worker 初始化
├─ slime/ray/rollout.py                RolloutManager.generate：rollout 编排
│  └─ slime/rollout/sglang_rollout.py    样本生成与奖励计算
└─ slime/ray/actor_group.py            RayTrainGroup.async_train：训练分发
   └─ slime/backends/megatron_utils/actor.py
      ├─ model.py                      Megatron 模型执行
      └─ loss.py                       RL 损失与优势
```

✅ 并给了两条「可以先跳过」的建议：`slime/backends/sglang_utils/` 的部署细节和 `update_weight/` 的权重同步实现，等要改那块时再看。

⚠️ 我的补充建议——**如果你的目的是理解本目录讲的概念，最高效的读法是只读两个文件**：

```
train.py (99 行) + train_async.py (81 行)  =  180 行
```

**这 180 行里包含了：同置 vs 分离、buffer 深度 0 vs 1、权重同步频率 K、中断模型的软暂停、offload/onload 的显存管理。** 本目录前五章的核心概念，在这里全都有一行代码对应。

---

## 10 · 本篇小结

| 结论 | 分级 |
|---|---|
| 72,468 行，介于 veRL（171,584）和 OpenRLHF（13,411）之间 | ✅ 实测 |
| 训练后端适配 16,220 行 vs 推理后端 2,101 行 = **7.7 倍**，接训练引擎难得多 | ✅ 实测 + ⚠️ |
| 明确拒绝多后端抽象：*"instead of flattening ... into a lowest-common-denominator abstraction"* | ✅ 原文 |
| 参数直通：`--sglang-` 前缀 + Megatron 参数直读，上游升级零改动 | ✅ 原文 |
| 环境「不 fork 训练内核」↔ GLM-5 报告的「隔离 task-specific logic」，README 与论文互证 | ✅ 双向 |
| 三模块：training / rollout / **data buffer**；权重同步不是平级模块 | ✅ + ⚠️ |
| `train.py` 99 行 vs `train_async.py` 81 行，差别只有 18 行 | ✅ 实测 |
| `assert not args.colocate` —— 异步必须分离部署，一行断言 | ✅ L11 |
| `args.update_weights_interval` = GLM-5 报告里那个「每 K 次梯度更新」的 K | ✅ L66 ↔ 报告 |
| 换权重前先排空生成（L67 注释）= 中断模型第 3 档「软暂停」 | ✅ L67 |
| `onload_weights` / `onload_kv` 分开：先恢复权重再恢复 KV | ✅ 实测 |
| `health_monitoring_pause/resume` ↔ GLM-5 报告的心跳驱动容错 | ✅ 双向 |
| `on_policy_distillation.py` 只有 **67 行**——OPD 确实复用整套 RL 管线 | ✅ 实测 |
| `adapters/`（API 协议）与 `harness/`（agent CLI）是两层独立抽象 | ✅ + ⚠️ |
| **MILES 建立在 slime 之上**，所以「16 个独立框架」偏高，架构谱系远少于框架数 | ✅ 修正 🌐 |
| 只读 `train.py` + `train_async.py` 共 180 行，能覆盖本目录前五章的核心概念 | ⚠️ 建议 |

**下一篇** → [OpenRLHF拆解.md](OpenRLHF拆解.md)
