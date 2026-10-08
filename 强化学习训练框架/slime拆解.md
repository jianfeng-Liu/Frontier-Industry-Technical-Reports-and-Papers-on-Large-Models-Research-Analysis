# slime 拆解：GLM 系列背后的 RL 框架

> **这一篇干什么**：源码级走查。对标 [训练框架/Megatron-LM拆解.md](../训练框架/Megatron-LM拆解.md) 的写法。
> **难度** ★★
> **✅ 本文所有行号钉在 commit `3778dbf`**（本地 clone 于 [源码/slime/](源码/slime/)，tag `v0.3.2`，2026-08-28）。
> ⚠️ **RL 框架迭代极快，行号几个月就会失效。** 本文每条引用都同时给出**文件名 + 符号名**（类名 / 函数名 / 参数名），换版本时请按符号名重新定位，不要照抄行号。

---

## 0 · 七问速答：三份拆解共用的同一组问题

本目录的三份源码拆解（slime / [OpenRLHF](OpenRLHF拆解.md) / [MILES](MILES拆解.md)）回答的是**同一组七个问题**，这样你可以把三张表并排放着横向比。先看 slime 的答案：

| # | 问题 | slime 的答案 | 展开于 |
|---|---|---|---|
| **Q1** | 架构档位：同步 / 单步异步 / 全异步？ | **三档都有**，但不是同一套代码：`train.py` 同步、`train_async.py` 单步异步、`examples/fully_async/` + `slime/rollout/fully_async_rollout.py` 全异步 | §4 · §5 |
| **Q2** | 训练后端是谁？ | **只有 Megatron-LM**，不做抽象层，Megatron 的参数直读 | §2 · §1 |
| **Q3** | rollout 引擎用谁？训练进程怎么和它说话？ | **只有 SGLang**（带 router）。通过 **Ray actor 调用**：`RolloutManager.generate.remote(rollout_id)` 返回一个 Ray ObjectRef | §3 · §7 |
| **Q4** | 权重同步怎么做？要不要停推理？ | **四条通路可选**（Ray IPC / NCCL 广播 / 磁盘全量 / 磁盘增量），由两个参数组合决定；**要停**——换权重前先把在飞的生成排空 | §6 |
| **Q5** | 支持哪些算法？加一个新算法要动哪些文件？ | 6 个优势估计器（内置 choices）。加估计器动 **2 个文件**；加一个全新 loss **0 个文件**（有 `--custom-loss-function-path` 逃生口） | §10 |
| **Q6** | 沙箱 / 环境怎么接？ | 接成**数据生成函数**，不碰训练内核。`slime/agent/` 分 `adapters/`（API 协议）和 `harness/`（真实 CLI）两层 | §9 |
| **Q7** | 一次迭代的时序长什么样？谁在等？ | 同步档：训练卡等 rollout，**空转约 2/3 的时间**；单步异步档：两边重叠，迭代时间被 rollout 单独决定 | §4 |

---

## 1 · 规模实测

✅ 本地 clone 后实测（commit `3778dbf`）：

```
整仓：332 个 .py 文件 · 72,468 行 · 18 MB

顶层：
  train.py             99 行   <-- 同步训练主循环
  train_async.py       81 行   <-- 异步训练主循环
  slime/                       <-- 主包，152 文件 / 35,708 行
  slime_plugins/  examples/  tests/  tools/  scripts/  docs/
```

`slime/` 内部：

| 目录 | 文件数 | 行数 | 干什么 |
|---|---|---|---|
| `backends/` | 68 | **18,322** | 训练与推理引擎的适配 |
| `utils/` | 30 | 7,900 | 工具（含 `ppo_utils.py`、`arguments.py`） |
| `rollout/` | 21 | 3,015 | 生成流程 |
| `agent/` | 13 | **2,727** | agent 与沙箱（见 [06 章](06-AgenticRL-环境与沙箱.md)） |
| `observability/` | 12 | 2,516 | 日志、追踪、profiling |
| `ray/` | 7 | **1,228** | Ray 编排 |

★ `backends/` 内部有一个很说明问题的不对称：

| 适配对象 | 目录 | 行数 |
|---|---|---|
| 训练侧（Megatron） | `slime/backends/megatron_utils/` | **16,220** |
| 推理侧（SGLang） | `slime/backends/sglang_utils/` | **2,101** |
| **比值** | | **7.7 倍** |

⚠️ **接一个训练引擎，比接一个推理引擎难 7.7 倍。**

原因不难理解：推理引擎是个服务，接口是「发请求、收 token」；训练引擎要处理并行切分、优化器状态、checkpoint、梯度累积——而且这些都要和权重同步对上。这也解释了为什么开源框架里「多推理后端」比「多训练后端」常见得多。

**对照组**（✅ 见 [01 章](01-格局全景.md)）：veRL 171,584 行 · slime 72,468 行 · OpenRLHF 13,411 行。

---

## 2 · ✅ 设计哲学：明确拒绝抽象

slime 是本目录里**唯一把设计取舍写在 README 第一屏**的框架。这些不是我的归纳，是它自己的原话：

### 2.1 只接一个推理后端，是为了不丢特性

> ✅ *"By choosing **one** rollout backend, slime can use SGLang-specific capabilities directly instead of **flattening multiple inference engines into a lowest-common-denominator abstraction**."*
> （选定单一 rollout 后端，slime 就能直接用 SGLang 的专属能力，而不是把多个推理引擎压平成一个最小公分母的抽象。）

⚠️ 这句话是对 veRL 那条路线的正面回应。veRL 接了 4 个推理引擎，代价就是：任何一个引擎的新特性，都要先在抽象层里找到位置才能用上。slime 用「少一个自由度」换「零延迟接入 SGLang 新能力」。

### 2.2 参数直通（pass-through），不做包装

> ✅ *"every argument supported by the installed SGLang can be used by adding the `--sglang-` prefix"*
> ✅ *"slime reads Megatron arguments directly, so Megatron-side parallelism, optimizer, checkpointing, and model options remain available **without wrapper code**"*

⚠️ 这是「拒绝抽象」的具体落地：slime 不重新定义参数体系，SGLang 有什么参数，加个 `--sglang-` 前缀就能用。**上游升级了新功能，slime 一行代码都不用改就能用上。**

### 2.3 环境「不 fork 训练内核」

> ✅ *"math, code, search, tools, sandboxes, verifiers, environments, multi-agent systems, and long-horizon agentic workflows plug in as **data generation or reward workflows**. They **do not fork the training kernel**."*

⚠️ 这和 ✅ GLM-5 报告里那句 *"cleanly isolates task-specific logic from the core training loop"*（见 [06 章](06-AgenticRL-环境与沙箱.md)）是**同一条原则的两处表述**——一处在论文里，一处在 README 里。**报告与源码互相印证。**

### 2.4 一句值得抄下来的话

> ✅ *"**RL bugs are often silent.** slime keeps the dataflow explicit, supports separate rollout-only and train-only debugging paths."*

⚠️ 「RL 的 bug 通常是沉默的」——这正是 [04 章](04-训推一致性.md) 开头讲的那件事。而 slime 的应对是**工程手段**：把 rollout 和 train 拆成可以独立调试的两条路径，出问题时能二分定位。

---

## 3 · ★★ 三个模块：training / rollout / data buffer

✅ README 的 Architecture Overview 只列了三个模块。把它们的职责摊成一张表：

| 模块 | 跑在哪 | 它做什么 | 和 data buffer 的关系 |
|---|---|---|---|
| **training** | Megatron（训练卡） | 从 data buffer **读**数据做梯度更新；训完把参数推给 rollout | 只读 |
| **rollout** | SGLang + router（推理卡） | 生成新数据，含 reward / verifier 的输出 | 只写 |
| **data buffer** | 中间层 | 管理 prompt 初始化、自定义数据、生成方法 | 是**唯一**的数据桥 |

连接关系（纯连接线，无右边框）：

```
   training (Megatron)            rollout (SGLang + router)
          |                                 |
          | read                      write |
          +--------->  data buffer  <-------+
```

⚠️ **注意这里没有「weight sync」这个模块。** 权重同步在 slime 里不是一个平级模块，而是 training 模块的一个动作（`actor_model.update_weights()`）。这和 veRL 把 `checkpoint_engine/` 单独抽一层（见 [03 章](03-权重同步.md)）是不同的组织方式。

⚠️ **data buffer 是唯一的数据桥。** ✅ README 原话：*"Custom generate functions can wrap this with multi-turn loops, tool calls, environment/sandbox interaction, and verifier-based reward"*——不管你的环境多复杂，最后都是往 data buffer 里放样本。**这就是「环境即数据生成」的架构实现。**

✅ 源码里对应的是 `slime/rollout/data_source.py`（229 行），三个类是一条继承链：

| 层级 | 类名 | 关键方法 | 说明 |
|---|---|---|---|
| 1 | `DataSource(abc.ABC)` | `get_samples` / `add_samples` / `save` / `load` / `__len__` | 抽象接口 |
| 2 | `RolloutDataSource` | 继承上面五个 | 基础实现，**不带 buffer** |
| 3 | `RolloutDataSourceWithBuffer` | `_get_samples_from_buffer()` · `get_buffer_length()` | ★ **带 buffer 的版本** |

⚠️ 类名的继承关系直接对应了 [02 章](02-架构主轴-同步到全异步.md) 讲的 buffer 深度：**不带 buffer 的是深度 0，带 buffer 的才有深度。** 深度这个概念在 slime 里不是一个数字参数，而是「你用哪个类」。

---

## 4 · ★★★ 一次迭代的完整时序：谁在算、谁在等

这是拆解文档最该有、却通常没有的一张图。先看同步档（`train.py`），再看单步异步档（`train_async.py`）。

### 4.1 同步档：训练卡大部分时间在空转

分离部署（训练卡和推理卡是两批卡）+ 同步主循环，一次迭代的时间轴：

```
  time ------------------------------------------------------------>
            0s                        32s          46s   47s

  TRAIN   |..........idle...........|#### train ####|=sync=|
  ROLLOUT |##### generate N ########|.... idle .....|=sync=|

  legend:  #### = busy     .... = idle     =sync= = weight sync
```

每一段对应 `train.py` 的哪一行：

| 段 | 训练卡在干什么 | 推理卡在干什么 | 源码位置 |
|---|---|---|---|
| ① | 卡在 `ray.get(...)` 上**空转** | 生成这一轮全部样本 | `train.py:53` `rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))` |
| ② | 前向 + 反向 + 优化器 | **空转**（同置时则被 offload 吐出显存） | `train.py:69` `ray.get(actor_model.async_train(...))` |
| ③ | 把新权重推给推理引擎 | 接收权重、重建 KV cache 池 | `train.py:85` `actor_model.update_weights()` |

⚠️ **第 53 行那个 `ray.get(...)` 是整个同步架构的全部秘密**：它在这里等着，直到所有样本生成完。这一等，训练卡就空转了。

### 4.2 ⚠️ 本文推演：同步档的空转到底有多贵

用 [大模型算子开发/10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8.2 那套 8B 模型的分阶段耗时做一次可验算的推演。⚠️ **下面是本文的推算，不是 slime 项目公布的数字**，效率因子全部来自那一节的 ⚠️ 量级估计。

**第 1 步 · 把一次迭代的耗时按「跑在哪批卡上」分成两堆**：

| 阶段 | 耗时 | 跑在哪 |
|---|---|---|
| Rollout（Prefill + Decode 权重读 + KV 读） | 1.06 + 24.5 + 6.8 = **32.4 s** | 推理卡 |
| Reference 前向 | **3.71 s** | 训练卡 |
| Re-forward + Backward + AdamW | 13.9 + 0.02 = **13.9 s** | 训练卡 |
| 打分（CPU 验证器） | 0.3 s | CPU |
| 权重同步 | 0.3 s | 两边都参与 |
| **合计** | **≈ 50.6 s** | |

**第 2 步 · 算两批卡各自的忙时**：

```
训练卡忙 = 3.71 + 13.9        = 17.6 s
推理卡忙 = 32.4               = 32.4 s
一次迭代总长（同步，串行） = 50.6 s
```

**第 3 步 · 算利用率**：

```
同步档：
  训练卡利用率 = 17.6 / 50.6 = 34.8%     <-- 空转 65.2%
  推理卡利用率 = 32.4 / 50.6 = 64.0%     <-- 空转 36.0%
```

**第 4 步 · 单步异步档：两边重叠，迭代时间由慢的那一边单独决定**：

```
稳态迭代时间 = max(训练卡忙, 推理卡忙) = max(17.6, 32.4) = 32.4 s
```

**第 5 步 · 换算成收益**（⚠️ 这里是本目录最容易写错的一处换算）：

```
时间减少比例 = 1 - 32.4 / 50.6 = 36.0%
加速倍数     = 50.6 / 32.4     = 1.56 倍
交叉验算     = 1 / (1 - 0.360) = 1.5625  ✓ 对上
```

⚠️ **「时间减少 36%」不等于「加速 36%」**，它等于「加速 1.56 倍」。公式是 `加速倍数 = 1 / (1 − 时间减少比例)`。这条换算在 [附-速查表.md](附-速查表.md) 末尾有专门一节。

⚠️ 这个 1.56 倍是**上界**，三件事会把它吃掉：① rollout 本身的长尾（同一批里最慢的那条决定整批何时结束）；② 权重同步时要排空在飞的生成（见 §6）；③ 离策略带来的样本质量损失——异步省下的时间，有一部分会以「需要更多步才收敛」的形式还回去。

### 4.3 单步异步档的时序

```
  time ------------------------------------------------------------>
            0s                 32s                64s              96s

  TRAIN   |...idle(warmup)...|### train N ###...|### train N+1 ##..|
  ROLLOUT |#### gen N #######|#### gen N+1 #####|#### gen N+2 #####|

  数据流： gen N 产出 --> train N 消费（晚一拍）
```

⚠️ 「晚一拍」就是 buffer 深度 = 1，也就是 [02 章](02-架构主轴-同步到全异步.md) 讲的**一步离策略**：训练 N 用的样本，是上一版权重生成的。

---

## 5 · ★★★ 两个主循环：99 行 vs 81 行

这是 slime 最好读的地方，也是本目录里**能把「同步 vs 异步」讲得最清楚的一处源码**。

### 5.1 同步版 `train.py`（99 行）

✅ 关键行（commit `3778dbf`，⚠️ 行号会漂移，按右列的符号名定位）：

| 行号 | 符号 | 对应什么 |
|---|---|---|
| 14 | `create_placement_groups(args)` | 分配 GPU |
| 19 | `create_rollout_manager(...)` | 建 rollout 管理器（内含 SGLang 引擎） |
| 21 | `create_training_models(...)` | 建训练模型（actor + 可选 critic） |
| 27 | `actor_model.update_weights()` | 开跑前先推一次权重 |
| **49** | `for rollout_id in range(...)` | **主循环** |
| **53** | `ray.get(rollout_manager.generate.remote(...))` | ★ **阻塞式生成** |
| 69 | `ray.get(actor_model.async_train(...))` | 训练 |
| **85** | `actor_model.update_weights()` | ★ **每一步都同步权重** |

### 5.2 异步版 `train_async.py`（81 行）：三处改动

**改动一 · L11：异步必须分离部署**

```python
assert not args.colocate, "Colocation is not supported for async training."
```

⚠️ 一行断言，把 [02 章](02-架构主轴-同步到全异步.md) 那条「同置 vs 分离」的选择变成了硬约束：**要异步，就必须分开放。**

**改动二 · L32 / L35-36 / L39-40：一步离策略的教科书实现**

```python
# L32  循环【开始之前】就先发出第一次生成请求（不 ray.get，拿的是 future）
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # L35-36  等上一次的生成结果
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)

    # L39-40  ★ 立刻发出【下一次】生成请求，然后才去训练
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    # ... 训练当前这批（此时下一批正在生成）
```

⚠️ **这就是 §4.3 那张图的 8 行实现。** `next_future` 这个变量名本身就是那个深度为 1 的缓冲区。

**改动三 · L66-70：权重同步频率是个参数**

```python
if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
    # sync generate before update weights to prevent update weight in the middle of generation
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None
    actor_model.update_weights()
```

★ 这五行同时印证了两件事：

1. **`args.update_weights_interval`**（定义在 `slime/utils/arguments.py:547`）= ✅ GLM-5 报告里那句 *"pushes the new weights back to the inference engine **every K gradient updates**"*（见 [02 章](02-架构主轴-同步到全异步.md)）。**报告里的 K，在源码里就是这个参数。**
2. **那行注释** —— *"sync generate before update weights to prevent update weight in the middle of generation"*（换权重前先把生成同步掉，避免在生成中途换权重）——说明 slime 选的**不是「永不停」，而是排空后再换**，对应 [03 章](03-权重同步.md) 中断模型里「软暂停 / 排空」那一档。

### 5.3 ⚠️ 一处要纠正的说法：异步版不是「在同步版上改了 18 行」

本文早先的版本写过「从同步走到一步离策略，slime 只改了 18 行代码」。✅ 逐行对读两个文件后，这个说法**不准确**，真实情况是：

| 变化 | 内容 | 行数量级 |
|---|---|---|
| **加** | L11 断言 + L32/35-36/39-40 的 future 调度 | 约 +8 行 |
| **改** | L66-70 的权重同步条件（从「每步」改成「每 K 步，且先排空」） | 约 5 行 |
| **删** | **整套 offload / onload 机制**：`train.py` 的 L23-24、L32-33、L39-46 的 `offload_train()`、L55-56、L82-84、L87-88 | 约 −25 行 |

⚠️ **异步版更短，不是因为它更简单，而是因为 L11 那条断言禁掉了同置——不同置就不需要显存换入换出，`offload` / `onload_weights` / `onload_kv` 那一整块直接消失了。**

⚠️ 这个修正反而让结论更强：[02 章](02-架构主轴-同步到全异步.md) 观察到的「veRL 的 `one_step_off_policy/` 只有 553 行，`fully_async_policy/` 却有 4,335 行」说的是同一件事——**从同步走到一步离策略，代价极小；真正的复杂度出现在再往下一档。**

✅ 而 `train_async.py` L9 的注释明确指出真正的全异步在别处：

```python
# The framework supports other asynchronous approaches such as fully async
# (which is shown in examples/full_async).
```

✅ 对应 `slime/rollout/fully_async_rollout.py`（274 行）和 `examples/fully_async/`。

---

## 6 · ★★★ 权重同步：四条通路，两个参数决定走哪条

[03 章](03-权重同步.md) 讲了权重同步的各种做法。slime 把它们收在一个目录里：`slime/backends/megatron_utils/update_weight/`（2,093 行，8 个文件）。

### 6.1 选择逻辑写在一个工厂函数里

✅ `update_weight/__init__.py` 的 `create_weight_updater()`，整个选择逻辑只看两个参数：

| 参数 | 取值 | 定义在 |
|---|---|---|
| `--update-weight-mode` | `full`（默认）/ `delta` | `slime/utils/arguments.py:134` |
| `--update-weight-transport` | `nccl`（默认）/ `disk` | `slime/utils/arguments.py:144` |

组合结果（⚠️ `--colocate` 会抢在前面覆盖 transport 的选择）：

| 配置 | 选中的类 | 数据实际怎么走 | 适合什么场景 |
|---|---|---|---|
| `--colocate` | `UpdateWeightFromTensor`（432 行） | GPU→CPU 序列化 → Gloo gather → **Ray IPC 句柄**交给同一张卡上的 SGLang 引擎 | 卡少、同置、单机 |
| 默认（分离 + `nccl` + `full`） | `UpdateWeightFromDistributed`（373 行） | 按 PP rank 建进程组 `slime-pp_{pp_rank}`，只有 `DP=0 且 TP=0` 的 rank 负责发，**NCCL 分块广播** | 分离部署的标准路径 |
| `transport=disk` + `mode=full` | `UpdateWeightFromDisk`（95 行） | 写一份**完整的 HF 格式 checkpoint**，引擎自己 reload | 跨集群、推理池弹性扩缩容 |
| `mode=delta`（**只能配 disk**） | `UpdateWeightFromDiskDelta`（304 行） | 和上一次的 pinned-CPU 快照**做差**，只发变化的字节 | 大模型、慢网络 |

✅ 三条硬约束直接写成了断言（`update_weight/__init__.py` 与 `slime/utils/arguments.py:2059-2075`）：

```python
# delta 模式只能走 disk
assert update_weight_transport == "disk", "--update-weight-mode=delta requires --update-weight-transport=disk"
# delta 模式不能和同置一起用
assert not args.colocate, "--update-weight-mode=delta is not supported with --colocate"
# 非 disk、非同置的情况下，full 模式只支持 nccl
assert update_weight_transport == "nccl", f"unsupported weight sync mode/transport: ..."
```

⚠️ **「Delta Weight Sync」在 [附-速查表.md](附-速查表.md) 里被列为 slime 的专有名词，这里是它的源码落点。** 注意它有个反直觉的限制：**增量同步只能走磁盘，不能走 NCCL。** 原因在 `__init__.py` 的注释里写着——delta 的落地方式是「每台主机把发布出来的 delta 应用进本机的 checkpoint，引擎再走普通的 `update_weights_from_disk` 重载」，这条路径天然是文件语义，不是集合通信语义。

### 6.2 NCCL 路径的两跳：先 all-gather，再 broadcast

✅ `update_weight_from_distributed.py` 的 `connect_rollout_engines()` 里有一段注释把 TP 情形说得很清楚：

```
For TP:
  1. AllGather parameters to rank 0
  2. Broadcast parameters from rank 0 to all sglang engines
```

⚠️ 这两跳就是 [03 章](03-权重同步.md) 讲的 **resharding**（重切分）的实现：训练侧的张量是按 Megatron 的 TP 切开的，推理侧的切法不一样，所以必须先在训练侧把一个完整参数拼回来（AllGather），再广播出去。**只有 `DP=0 且 TP=0` 的 rank 参与发送**，其余 rank 不占网络——这是个很实在的优化，否则 DP 组里每个副本都发一遍，流量翻 DP 倍。

### 6.3 要不要停推理：slime 的答案是「停，但先排空」

| 档位 | 做法 | slime 选哪个 |
|---|---|---|
| ① 永不停 | 边生成边换权重，一条轨迹可能横跨两个版本 | ✗ |
| ② 硬打断 | 立即中止在飞的生成，丢掉或存起来 | ✗ |
| ③ **软暂停 / 排空** | **先把在飞的生成同步掉，再换权重** | ✅ `train_async.py:67` 注释 |

⚠️ 代价很直白：排空期间推理卡在收尾、训练卡在等，两边都不满负荷。**这就是 §4.2 推演里那个 1.56 倍上界被吃掉的第二个来源。**

---

## 7 · Ray 编排层：1,228 行

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

| 方法组 | 方法（行号） | 说明 |
|---|---|---|
| 主流程 | `generate` (163) · `eval` (184) | 生成与评测 |
| 数据持久化 | `save` (201) · `load` (204) | 断点续训 |
| **显存管理** | `offload` (207) · `onload` (212) · **`onload_weights`** (216) · **`onload_kv`** (220) | ★ 权重和 KV cache **分开**换入换出 |
| 权重 | `check_weights` (251) | 校验训推权重一致 |
| **容错** | `recover_updatable_engines` (224) · `health_monitoring_pause` (243) · `health_monitoring_resume` (247) | ★ 对应 ✅ GLM-5 的「心跳驱动容错」 |
| 数据加工 | `_post_process_rewards` (279) · `_convert_samples_to_train_data` (306) | 样本 → 训练数据 |
| 并行适配 | `set_train_parallel_config` (425) · `_split_train_data_by_dp` (428) | 按训练侧 DP 切数据 |

⚠️ 两处值得单独记：

1. **`onload_weights` 和 `onload_kv` 是分开的两个方法。** 同置模式下，训练要用显存时，推理引擎必须把权重和 KV cache 都吐出来；但恢复时**先恢复权重、再恢复 KV**——因为权重恢复完就能开始接请求了，KV 可以慢慢来。✅ `train.py:84` 和 `train.py:88` 正是分两次调用的，中间夹着 `actor_model.update_weights()`（L85）。**顺序是：onload 权重 → 推新权重 → onload KV。** 这是个很细但很实在的优化。
2. **`health_monitoring_pause` / `health_monitoring_resume`** 对应 ✅ GLM-5 技术报告里的 *"heartbeat-driven rollout fault tolerance and router-level server lifecycle management"*——**报告里描述的容错机制，在源码里能找到同名的方法。** ⚠️ 注意这是「名字对得上」级别的互证，不是「我验证了它的行为」。

---

## 8 · rollout 层：一个目录读懂 slime 支持什么

✅ `slime/rollout/`（21 文件 / 3,015 行），文件名几乎就是功能清单：

| 文件 | 行数 | 干什么 |
|---|---|---|
| `sglang_rollout.py` | **649** | 主力生成路径 |
| `fully_async_rollout.py` | **274** | 全异步（见 [02 章](02-架构主轴-同步到全异步.md)） |
| `data_source.py` | 229 | data buffer（见 §3） |
| `sglang_streaming_rollout.py` | 167 | 流式生成 |
| `forge_load.py` | 114 | 负载调度 |
| `sft_rollout.py` | 68 | SFT 混训 |
| `on_policy_distillation.py` | **67** | 在策略蒸馏（见 [05 章](05-算法与框架的接口.md)） |
| `sample_hooks.py` | 50 | 自定义钩子 |
| `sleep_rollout.py` | 12 | 睡眠模式 |
| `filter_hub/` · `rm_hub/` | — | 样本过滤 / 奖励模型 |

⚠️ **`on_policy_distillation.py` 只有 67 行**，这是 [05 章](05-算法与框架的接口.md) 那个结论的最硬证据：**在策略蒸馏（OPD）就是把奖励换成教师概率，其他一切照旧。** 复用了整套 rollout、buffer、权重同步，所以只需要 67 行。

✅ 而且在 commit `3778dbf` 这一版里，OPD 已经被做成**和优势估计器正交**的一个开关——`slime/utils/arguments.py:952` 那个 `--advantage-estimator` 的 help 原话是：*"on-policy distillation (OPD) is now orthogonal to the advantage estimator. Use `--opd-kl-coef > 0` to enable OPD on top of any estimator."* **也就是说 OPD 可以叠在 GRPO / GSPO / CISPO 任意一个上面。**

⚠️ `filter_hub/` 的存在对应 [05 章](05-算法与框架的接口.md) 讲的动态过滤，也对应 ✅ GLM-5 那条「按失败原因排除环境崩溃样本」（[06 章](06-AgenticRL-环境与沙箱.md)）。

---

## 9 · agent 层：沙箱和真实 CLI 怎么接进来

这一块在 [06 章](06-AgenticRL-环境与沙箱.md) 已经详细讲过，这里只放结构。`slime/agent/`：13 文件 / 2,727 行。

```
slime/agent/
├── trajectory.py
├── sandbox.py
├── parsing.py
├── aiohttp_threaded.py
├── adapters/
│   ├── common.py
│   ├── openai.py
│   └── anthropic.py
└── harness/
    ├── common.py
    ├── claude_code.py
    └── codex.py
```

行数和职责（标注写在图外，避免对齐问题）：

| 文件 | 行数 | 干什么 |
|---|---|---|
| `trajectory.py` | 508 | 轨迹数据结构 |
| `sandbox.py` | 399 | 沙箱抽象（README 自称 *"intentionally small"* 的接口） |
| `parsing.py` | 114 | 输出解析 |
| `aiohttp_threaded.py` | 98 | 异步 HTTP 客户端 |
| `adapters/common.py` | 523 | 协议适配公共部分 |
| `adapters/openai.py` | 378 | OpenAI 协议 |
| `adapters/anthropic.py` | 350 | Anthropic 协议 |
| `harness/common.py` | 178 | CLI 拉起的公共部分 |
| `harness/claude_code.py` | **86** | ★ Claude Code |
| `harness/codex.py` | **71** | ★ Codex |

⚠️ **`adapters/` 和 `harness/` 是两层不同的抽象**，这个区分很关键：

- `adapters/` 解决「**模型**怎么被调用」——走 OpenAI 协议还是 Anthropic 协议
- `harness/` 解决「**agent 程序**怎么被跑起来」——装 Node、装 npm 包、拼命令行参数

**一个 agent CLI 可以走任意一种 API 协议，所以这两层必须分开。** 也正因为分开了，接一个新的 agent CLI 只要写几十行（`claude_code.py` 86 行、`codex.py` 71 行）。

⚠️ 回到 §2.3 那条原则：这一整个目录都挂在 **rollout 侧**，训练内核一行都没动。这就是 *"They do not fork the training kernel"* 的实际样子。

---

## 10 · ★★ 加一个新算法，要动哪些文件

这是选框架时最该问、却最少被回答的问题。slime 的答案分三档。

### 10.1 内置了哪些

✅ `slime/utils/arguments.py:952` 的 `--advantage-estimator`，choices 写死了 6 个：

| 取值 | 中文 | 要不要 critic |
|---|---|---|
| `grpo`（默认） | 分组相对策略优化 | 否 |
| `gspo` | 分组序列策略优化 | 否 |
| `cispo` | 裁剪重要性采样策略优化 | 否 |
| `reinforce_plus_plus` | REINFORCE++ | 否 |
| `reinforce_plus_plus_baseline` | REINFORCE++ 带基线 | 否 |
| `ppo` | 近端策略优化 | **是** |

✅ 这个「要不要 critic」不用你自己记——`arguments.py:1913` 一行就推出来了：

```python
args.use_critic = args.advantage_estimator == "ppo"
```

⚠️ **六选一里只有 PPO 需要 critic。** 对照 [05 章](05-算法与框架的接口.md) 讲的「去 critic 潮流」：这一行代码就是那个潮流的结果——critic 从「默认配置」退成了「一个取值的特例」。

### 10.2 三档改动量

| 你要加什么 | 要动的文件 | 量级 |
|---|---|---|
| **一个新的优势估计器** | ① `slime/utils/arguments.py` 的 choices 列表加一项；② `slime/utils/ppo_utils.py` 加对应的 `get_xxx_advantages()` 函数 | **2 个文件** |
| **一个新的策略损失** | `slime/backends/megatron_utils/loss.py`：`policy_loss_function()`（L933）里加分支，`loss_function()`（L1282）的 `match args.loss_type` 分发（L1326）里挂上 | **1 个文件** |
| **一个完全自定义的 loss** | ⭐ **不用改仓库**：`--loss-type custom_loss` + `--custom-loss-function-path <你的函数路径>`（`arguments.py:926` 和 `:936`） | **0 个文件** |

⚠️ 第三档那个逃生口很值得注意：**slime 给算法研究者留了一条「不 fork 仓库也能换损失函数」的路**。这和 §2 那条「拒绝抽象」的哲学是一致的——不替你设计一套算法注册体系，而是给一个函数路径参数让你自己挂。

⚠️ **对照 veRL**：veRL 有 14 个优势估计器 + 12 个策略损失的注册表（见 [05 章](05-算法与框架的接口.md)）。**slime 的 6 个不是「能力不足」，是「研究者自己挂」的另一种设计选择。**

---

## 11 · ✅ 生态：MILES 其实建在 slime 上

✅ slime README 有一节 *"Ecosystem Built on slime"*，列了 11 个下游项目：

Dressage · **Miles** · vime · Relax · OpenClaw-RL · P1 · RLVE · TritonForge · APRIL · qqr · ART(AWS)

⚠️ **这里有一个需要修正 [01 章](01-格局全景.md) 那张开源总表的地方**：🌐 HF 横评把 **MILES** 列为一个独立的库（radixark 出品，~950 star），✅ 而 slime 的 README 把它列在「建立在 slime 之上的生态」里。

**所以「16 个独立框架」这个数字是偏高的**——其中至少有一个是另一个的下游。

⚠️ **这里要分清两件事**：*"Ecosystem Built on slime"* 这个标题是 ✅ slime README 的原文；而由此推出的「架构谱系远少于框架数量」是 ⚠️ **本文的判断，不是 slime 项目或任何论文的结论**。我的判断是：很多「新框架」是在某个基座上换了调度策略或加了环境层，选型时应该先问「**它的基座是什么**」，而不是把它们当成 16 个平行的选项。

（这条是本次源码走查带来的、单看联网材料看不出来的发现。MILES 相对 slime 具体加了什么，见 [MILES拆解.md](MILES拆解.md)。）

---

## 12 · 怎么读这份源码

✅ slime README 自己给了一条阅读路径，实测和源码结构对得上：

```
train.py : train()
├── slime/ray/placement_group.py
├── slime/ray/rollout.py
│   └── slime/rollout/sglang_rollout.py
└── slime/ray/actor_group.py
    └── slime/backends/megatron_utils/actor.py
        ├── model.py
        └── loss.py
```

每一层在干什么：

| 文件 | 这一层负责 |
|---|---|
| `slime/ray/placement_group.py` | Ray 资源与 worker 初始化 |
| `slime/ray/rollout.py` | `RolloutManager.generate`：rollout 编排 |
| `slime/rollout/sglang_rollout.py` | 样本生成与奖励计算 |
| `slime/ray/actor_group.py` | `RayTrainGroup.async_train`：训练分发 |
| `slime/backends/megatron_utils/actor.py` | 训练侧主体 |
| `.../model.py` | Megatron 模型执行 |
| `.../loss.py` | RL 损失与优势 |

✅ README 并给了两条「可以先跳过」的建议：`slime/backends/sglang_utils/` 的部署细节和 `update_weight/` 的权重同步实现，等要改那块时再看。

⚠️ 我的补充建议——**如果你的目的是理解本目录讲的概念，最高效的读法是只读两个文件**：

```
train.py (99 行) + train_async.py (81 行)  =  180 行
```

**这 180 行里能一行代码对应一个概念：** 同置 vs 分离（`assert not args.colocate`）、buffer 深度 0 vs 1（`ray.get` vs `next_future`）、权重同步频率 K（`update_weights_interval`）、中断模型的排空档（L67 注释）、offload / onload 的显存管理（`onload_weights` / `onload_kv`）。本目录前五章的核心概念，在这里全都有一行代码对应。

---

## 13 · 本篇小结

| 结论 | 分级 |
|---|---|
| 72,468 行，介于 veRL（171,584）和 OpenRLHF（13,411）之间 | ✅ 实测 |
| 训练后端适配 16,220 行 vs 推理后端 2,101 行 = **7.7 倍**，接训练引擎难得多 | ✅ 实测 + ⚠️ 推断原因 |
| 明确拒绝多后端抽象：*"instead of flattening ... into a lowest-common-denominator abstraction"* | ✅ README 原文 |
| 参数直通：`--sglang-` 前缀 + Megatron 参数直读，上游升级零改动 | ✅ README 原文 |
| 环境「不 fork 训练内核」↔ GLM-5 报告的「隔离 task-specific logic」，README 与报告互证 | ✅ 双向 |
| 三模块：training / rollout / **data buffer**；权重同步不是平级模块，是 training 的一个动作 | ✅ + ⚠️ |
| **同步档训练卡利用率 34.8%、空转 65.2%；单步异步把迭代从 50.6 s 压到 32.4 s** | ⚠️ **本文推演**（基于 [10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8.2 的分阶段耗时） |
| **时间减少 36% = 加速 1.56 倍**，不是「加速 36%」 | ⚠️ 换算 |
| `assert not args.colocate` —— 异步必须分离部署，一行断言 | ✅ `train_async.py:11` |
| **异步版更短不是因为更简单**：禁掉同置后整套 offload/onload 消失了（约 −25 行），净加约 8 行 | ✅ 逐行对读（**纠正旧版「只改了 18 行」**） |
| `args.update_weights_interval` = GLM-5 报告里那个「每 K 次梯度更新」的 K | ✅ `arguments.py:547` ↔ 报告 |
| 换权重前先排空生成 = 中断模型的「软暂停 / 排空」档，不是「永不停」 | ✅ `train_async.py:67` |
| **权重同步四条通路**：Ray IPC（同置）/ NCCL 广播（默认）/ 磁盘全量 / 磁盘增量；增量**只能走磁盘** | ✅ `update_weight/__init__.py` |
| NCCL 路径是两跳：先 TP 内 AllGather 回完整参数，再从 `DP=0,TP=0` 广播 | ✅ `update_weight_from_distributed.py` 注释 |
| `onload_weights` / `onload_kv` 分开：先恢复权重、推完新权重、再恢复 KV | ✅ `train.py:84/85/88` |
| `health_monitoring_pause/resume` ↔ GLM-5 报告的心跳驱动容错 | ✅ 名字互证（⚠️ 未验证行为） |
| `on_policy_distillation.py` 只有 **67 行**；且 OPD 已和优势估计器正交（`--opd-kl-coef`） | ✅ 实测 + README |
| 6 个优势估计器，**只有 PPO 需要 critic**（`args.use_critic = estimator == "ppo"`） | ✅ `arguments.py:1913` |
| 加估计器动 2 个文件；加自定义 loss **0 个文件**（`--custom-loss-function-path`） | ✅ 实测 |
| `adapters/`（API 协议）与 `harness/`（agent CLI）是两层独立抽象 | ✅ + ⚠️ |
| **MILES 建立在 slime 之上** | ✅ slime README 原文 |
| 由此推出「架构谱系远少于框架数」 | ⚠️ **本文判断**，非项目结论 |
| 只读 `train.py` + `train_async.py` 共 180 行，能覆盖本目录前五章的核心概念 | ⚠️ 建议 |

**下一篇** → [OpenRLHF拆解.md](OpenRLHF拆解.md) ｜ **下游** → [MILES拆解.md](MILES拆解.md)
