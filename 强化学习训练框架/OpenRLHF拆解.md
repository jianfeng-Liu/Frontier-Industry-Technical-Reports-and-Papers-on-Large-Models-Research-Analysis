# OpenRLHF 拆解：架构最直白的那一个

> **这一篇干什么**：源码级走查。OpenRLHF 是本目录里代码量最小的框架，**因此它是理解 RL 框架骨架的最佳入口**。
> **难度** ★★
> **✅ 本文所有行号钉在 commit `3c3be62`（tag `v0.11.0`，2026-08-13）**。行号会随版本漂移。

---

## 1 · 规模实测：小一个数量级

✅ 本地 clone 后实测：

```
整仓：65 个 .py 文件 · 13,411 行 · 1.7 MB
```

| | veRL | slime | **OpenRLHF** |
|---|---|---|---|
| 行数 | 171,584 | 72,468 | **13,411** |
| 相对 OpenRLHF | **12.8×** | 5.4× | 1× |

`openrlhf/` 内部：

| 目录 | 文件数 | 行数 | 干什么 |
|---|---|---|---|
| `trainer/` | 20 | **5,196** | 所有训练器（PPO / DPO / SFT / RM）+ Ray 编排 |
| `utils/` | 14 | 2,772 | 工具，含 `agent.py` |
| `cli/` | 7 | 1,967 | 命令行入口 |
| `models/` | 6 | 1,352 | 模型与损失 |
| `datasets/` | 5 | 641 | 数据集 |

⚠️ **12.8 倍的差距不在算法，在后端适配。** [01 章 §1.2](01-格局全景.md) 已经论证过这点：veRL 接了 6 个训练引擎 × 4 个推理引擎 × 6 个 checkpoint 后端；OpenRLHF 只认 **DeepSpeed + vLLM** 一条路。

**这不是缺点，是定位。** 如果你要在多种硬件、多种引擎上跑，veRL 是唯一选择；如果你要读懂 RL 框架到底在干什么，**从 13,411 行开始远比从 171,584 行开始现实**。

---

## 2 · ✅ 三个基础设施组件

OpenRLHF 的 README 把技术栈说得很明确：

| 组件 | 角色 | ✅ 原文要点 |
|---|---|---|
| **Ray** | 分布式调度与控制 | *"separates the Actor, Reward, Reference, and Critic models across different GPUs, enabling scalable training for models up to **70B+ parameters**"* |
| **vLLM** | 高性能推理 | *"RLHF training spends **80% of the time on sample generation**"* |
| **DeepSpeed** | 显存高效训练 | ZeRO-3 + deepcompile + AutoTP + RingAttention |
| **NCCL / CUDA IPC** | 高速通信 | 见 [03 章 §3.2](03-权重同步.md) |

✅ 那句「生成占 80% 时间」是本目录反复引用的那个数字的**框架侧一手出处**（[02 章 §2.1](02-架构主轴-同步到全异步.md) 用的是 🌐 博客给的量化数据，这里是框架 README 的直接陈述）。

### ★ Hybrid Engine：另一条省钱的路

✅ README 原文：

> *"**Hybrid Engine Scheduling**: All models and vLLM engines can **share GPU resources**—minimizing idle time and maximizing GPU utilization. This allows running full RLHF pipelines on **limited hardware**."*

⚠️ 这是 [02 章](02-架构主轴-同步到全异步.md) 那条轴上被忽略的一个位置：**GLM-5 靠「分离 + 异步」提高利用率，OpenRLHF 的 Hybrid Engine 靠「全部同置 + 换入换出」提高利用率。** 两者的目标一样（别让卡闲着），手段完全相反。

⚠️ 而且注意最后一句 *"on limited hardware"*——这和 ✅ Kimi-K3 的 *"within a few hundred GPUs"*（[01 章 §1.3](01-格局全景.md)）是同一个约束下的同一类答案。**卡少的时候，同置几乎总是对的。**

✅ 源码里对应的是四个开关（`openrlhf/cli/train_ppo_ray.py`）：

```
--train.colocate_actor_ref      actor 和 reference 共享 GPU
--train.colocate_critic_reward  critic 和 reward 共享 GPU
--train.colocate_all            全部共享
--vllm.enable_sleep             ★ vLLM 睡眠模式（不用时释放显存）
```

⚠️ 前两个开关很有意思：**actor 和 reference 是天然的一对**（reference 只做前向，不占优化器状态），**critic 和 reward 也是一对**。这个配对不是随便定的——它对应 [14 章 §14.2](../训练框架/分布式训练入门/14-RL训练框架.md) 那张显存账里 16Ψ 和 2Ψ 的搭配。

---

## 3 · ★★★ 同步主循环：595 行读懂 PPO

✅ `openrlhf/trainer/ppo_trainer.py` 的 `fit()`（L499 起）是全目录**最容易读懂的 RL 主循环**：

```python
for episode in range(start_episode, self.args.train.num_episodes):      # L516
    while True:
        # ① 生成一批样本（阻塞）
        rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
            self.samples_generator.generate_samples(**self.generate_kwargs)
        )

        if not rollout_samples:
            if is_exhausted: break
            continue

        # ② 用这批样本做一次 PPO 更新
        status, global_step = self.train_step(rollout_samples, global_step)

        # ③ 记录耗时（★ 生成时间是被单独计时的）
        status["timing/generation"] = generation_time

        # ④ 存 checkpoint / 评测
        self.save_logs_and_checkpoints(global_step, status, client_states)
```

⚠️ **这就是 [02 章 §2.2](02-架构主轴-同步到全异步.md) 的「buffer 深度 0」，20 行说完。**
`generate_samples()` 是阻塞的，它返回之前训练什么都干不了。和 slime `train.py` L53 那个 `ray.get(...generate...)` 是完全一样的形状（见 [slime拆解 §4](slime拆解.md)）。

✅ 两个值得注意的细节：

1. **`status["timing/generation"]` 被单独计时**——框架自己就认为「生成占多久」是首要指标。
2. **`filter_pass_rate`** 是 `generate_samples` 的返回值之一，配 `--algo.dynamic_filtering_enable`。这就是 [05 章 §5.5](05-算法与框架的接口.md) 讲的动态过滤，**而 pass rate 被当作一个要记录的训练指标**——⚠️ 说明这个数字在实践中确实需要盯着。

---

## 4 · ★★★ 异步主循环：三个源码级的概念印证

✅ `openrlhf/trainer/ppo_trainer_async.py`（353 行）是本目录**印证密度最高的一个文件**。

### ① `--train.async_queue_size` 就是 buffer 深度

✅ `openrlhf/cli/train_ppo_ray.py:275`：

```python
parser.add_argument("--train.async_queue_size", type=int, default=1,
                    help="Queue size for async sampler<->trainer")
```

✅ 用在 `ppo_trainer_async.py:287-297`：

```python
queue_size = getattr(strategy.args.train, "async_queue_size", 1)
if queue_size <= 0:
    raise ValueError(f"async_queue_size must be positive, got {queue_size}")

self.rollout_queue = Queue(maxsize=queue_size)

# Token pool (counting semaphore) for queue capacity.
self.rollout_slots = Queue(maxsize=queue_size)
for _ in range(queue_size):
    self.rollout_slots.put(0, block=True)
```

★ **[02 章 §2.2](02-架构主轴-同步到全异步.md) 那张「buffer 深度四档」的表，在这里是一个整数参数。**

| 配置 | 对应哪一档 |
|---|---|
| 不开 `--train.async_enable` | 深度 0（同步） |
| `--train.async_queue_size 1`（默认） | **深度 1**（一步离策略） |
| `--train.async_queue_size K` | **深度 2~K**（有界队列） |

⚠️ 而且实现用了**两个队列**：`rollout_queue` 装样本，`rollout_slots` 是一个计数信号量（counting semaphore）——生成侧必须先拿到一个 slot 才能开始生成。这正是 [02 章 §2.5](02-架构主轴-同步到全异步.md) 讲的「② 深度限制：队列满了就阻塞生成」的标准实现。**深度上界是硬保证的，不是靠速率碰运气。**

### ② `VLLMLock` 就是中断模型

✅ `ppo_trainer_async.py:20-34`：

```python
@ray.remote(num_cpus=0)
class VLLMLock:
    """Cross-actor mutex for vLLM critical section.

    Ensures generation and weight broadcast do not overlap on the same vLLM engines,
    so every sample in a batch is generated with consistent weights.
    """
```

⚠️ 注释的最后半句是重点：*"so every sample in a batch is generated with **consistent weights**"*（保证一个批次里的每个样本都由一致的权重生成）。

**这是一个明确的架构立场**：OpenRLHF 不接受 [03 章 §3.5](03-权重同步.md) 的档 1「永不停」——它宁可用一把互斥锁把生成和权重广播隔开，也不让一条轨迹跨两个版本。

### ③ `partial_rollout` 模式下才允许打断

✅ `ppo_trainer_async.py:254-265`：

```python
def broadcast_to_vllm(self):
    # Lock prevents weight broadcast from overlapping with eval generation.
    ray.get(self.vllm_lock.acquire.remote())
    if self._partial_rollout:
        batch_vllm_engine_call(self.vllm_engines, "pause_generation")
    try:
        super().broadcast_to_vllm()
    finally:
        if self._partial_rollout:
            batch_vllm_engine_call(self.vllm_engines, "resume_generation")
        ray.get(self.vllm_lock.release.remote())
```

★ **`pause_generation` / `resume_generation` = [03 章 §3.5](03-权重同步.md) 的中断模型第 3 档「软暂停」，源码级实证。**

⚠️ 而且注意这两行只在 `self._partial_rollout` 为真时才执行。也就是说：**只有开了 partial rollout，OpenRLHF 才允许「生成到一半被打断」**——因为 partial rollout 本来就设计了「存下来下轮接着生成」的机制（[02 章 §2.4](02-架构主轴-同步到全异步.md) Kimi 那一节）。没开的话，就老老实实等生成完。

### ④ 参数之间的约束关系写进了断言

✅ `openrlhf/cli/train_ppo_ray.py:667-698` 这一段，把本目录几章的结论直接写成了运行时断言：

| 行号 | 断言 | 印证了什么 |
|---|---|---|
| 670-671 | `if async_enable: assert not vllm.enable_sleep` | 异步模式下推理引擎不能睡——它得一直在生成 |
| 673-674 | `if partial_rollout_enable: assert async_enable` | **partial rollout 依赖异步**，同步模式下没有「下一轮接着生成」的概念 |
| 694-698 | `if vllm_generate_batch_size > rollout.batch_size: assert async_enable`<br>*"(over-sampling needs async queue to buffer extra batches)"* | ★ **超采样必须有队列接着** |
| 667-668 | `if colocate_all and async_enable:` 打印警告：*"only colocates DeepSpeed models"* | 异步下 vLLM 引擎不能再和训练同置 |

⚠️ 最后一条和 slime 的 `assert not args.colocate`（[slime拆解 §4](slime拆解.md)）是同一个约束的两种态度：**slime 直接禁止，OpenRLHF 给个警告然后部分支持。** slime 更硬，OpenRLHF 更宽松——⚠️ 对新手来说 slime 那种「早失败」更友好。

---

## 5 · ★★ Agent 抽象：把「多轮」和「算法」拆成正交两维

这是 OpenRLHF 自称的**核心创新**，也是本目录 [05 章 §5.2](05-算法与框架的接口.md) 引过的那个设计。

✅ README 原文：*"OpenRLHF is **the first RLHF framework** to implement a **unified agent-based paradigm**. Every training run—whether standard PPO or complex multi-turn reasoning—follows a consistent agent execution pipeline."*

✅ 源码里就是 `openrlhf/utils/agent.py`（**356 行**）：

```
openrlhf/utils/agent.py
├─ L12   class AgentExecutorBase(ABC)          ← 抽象基类（TITO 核心）
│         └─ execute()  抽象方法
├─ L31   class MultiTurnAgentExecutor(...)     ← 多轮：reset() + step()
└─ L184  class SingleTurnAgentExecutor(...)    ← 单轮：可选 reward_func()
```

⚠️ **356 行就实现了整个 agent 抽象层。** 对比 veRL 的 `agent_loop/` 2,831 行 + `tools/` 593 行（[06 章 §6.4](06-AgenticRL-环境与沙箱.md)）——差 9.6 倍。差距主要在 veRL 有 816 行的 `tool_parser.py`，而 OpenRLHF 把工具解析留给用户自己实现的 executor。

✅ 用户接入的方式是一个命令行参数：

```
--train.agent_func_path <你的 AgentExecutorBase 子类>
```

✅ `openrlhf/trainer/ray/vllm_engine.py:19-29` 负责加载并校验：

```python
def _load_agent_executor(agent_func_path: str) -> AgentExecutorBase:
    ...
    assert issubclass(agent_executor_cls, AgentExecutorBase), \
        "AgentExecutor must inherit from AgentExecutorBase"
```

⚠️ 这个设计的关键在于**它把用户代码放在了推理引擎那一侧**（`ray/vllm_engine.py`），而不是训练侧。**用户写的 agent 逻辑离生成最近、离训练最远**——这和 ✅ GLM-5 的 *"cleanly isolates task-specific logic from the core training loop"*（[06 章 §6.2](06-AgenticRL-环境与沙箱.md)）是同一条原则。**三家（GLM 报告 / slime README / OpenRLHF 源码）独立表述了同一件事。**

### ✅ 正交性写进了 README

✅ 原文标题就是：*"Two Execution Modes (**Orthogonal to RL Algorithms**)"*

```
        执行模式 →     Single-Turn        Multi-Turn
  算法 ↓                （默认，99% 场景）   （reset + step）
  PPO                       ✓                  ✓
  REINFORCE++               ✓                  ✓
  REINFORCE++-baseline      ✓                  ✓
  GRPO                      ✓                  ✓
  RLOO                      ✓                  ✓
  Dr. GRPO                  ✓                  ✓
```

✅ 并且 README 明确写了 *"**Token-in-Token-out**: All sampling produces token-level trajectories → **Zero text-level mismatch**"*——即 [04 章 §4.5](04-训推一致性.md) 讲的 TITO，在 OpenRLHF 里是**架构基石**而不是可选项。⚠️ 这也是它敢把 agent 抽象做到只有 356 行的原因之一：**因为不做文本层的转换，就不需要处理转换带来的一大堆边界情况。**

---

## 6 · 算法与损失：都在 `models/loss.py`

✅ `openrlhf/models/`（6 文件 / 1,352 行）：

| 文件 | 行数 |
|---|---|
| `loss.py` | **335** |
| `model.py` | 319 |
| `actor.py` | 305 |
| `utils.py` | 186 |
| `ring_attn_utils.py` | 182 |

✅ 支持的算法（README 表，用 `--algo.advantage.estimator` 切换）：

| 算法 | 参数值 | 特点 |
|---|---|---|
| PPO | （默认） | 完整 critic |
| REINFORCE++ | `reinforce` | ✅ **由 OpenRLHF 团队于 2024/12 提出** |
| REINFORCE++-baseline | `reinforce_baseline` | 均值基线，适合 RLVR 推理任务 |
| RLOO | `rloo` | 逐 token KL + PPO clip |
| GRPO | `group_norm` | 组归一化 |
| Dr. GRPO | `dr_grpo` | 去掉组内 `/std` 归一化 |

⚠️ 对比 veRL 的 14 个优势估计器 + 12 个策略损失（[05 章 §5.2](05-算法与框架的接口.md)）：**OpenRLHF 只有 6 个，但 335 行就写完了。** 这个对比很能说明两个框架的定位差异——veRL 要做算法研究的公共平台，OpenRLHF 要做「能跑起来的最短路径」。

✅ 另外 `openrlhf/trainer/ppo_utils/` 里有几个值得注意的文件：

| 文件 | 行数 | 说明 |
|---|---|---|
| `experience_maker.py` | 411 | 把 rollout 变成训练用的 experience |
| `samples_generator.py` | 314 | 生成侧 |
| `experience.py` | 283 | experience 数据结构 |
| `replay_buffer.py` | **177** | ★ 回放缓冲区 |
| `length_penalty.py` | 153 | 长度惩罚 |
| `kl_controller.py` | 29 | KL 系数自适应 |

⚠️ `replay_buffer.py` 只有 177 行，而它就是 [01 章 §1.1](01-格局全景.md) 五模块里的第 ③ 块。**在同步框架里，data buffer 确实只需要这么多代码**——复杂度全在异步之后才出现（对照 slime 的 `data_source.py` 229 行 + 整个 `fully_async_rollout.py` 274 行）。

---

## 7 · ✅ 版本时间线：能看出这个领域的演进

✅ README 的 News 部分是一条有价值的行业时间线：

| 时间 | 事件 | ⚠️ 对应本目录哪个主题 |
|---|---|---|
| 2024/12 | 提出 **REINFORCE++** | 去 critic 潮流（[05 章](05-算法与框架的接口.md)） |
| 2025/4 | *"Clean OpenRLHF: Refactored based on **Single Controller** and Unified Packing Samples"* | 单控制器范式（[14 章 §14.4](../训练框架/分布式训练入门/14-RL训练框架.md)） |
| 2025/5 | v0.8.0：`--train.async_enable`（异步 RLHF）+ `--train.agent_func_path`（异步 agent RLHF） | ★ **异步与 agent 同一个版本落地** |
| 2025/10 | ScaleRL | — |
| 2026/2 | ProRL V2 | — |
| 2026/4 | v0.10：多轮 VLM RL + VLM RLHF | 多模态 |

⚠️ **2025/5 那一行最值得注意：异步和 agent 支持是在同一个版本一起出来的。**
这不是巧合——[06 章 §6.1](06-AgenticRL-环境与沙箱.md) 讲过，agentic 任务的轨迹又长又参差，同步架构根本扛不住。**agent 支持在工程上必须以异步为前提**，这条在 OpenRLHF 的断言里也写着（`partial_rollout_enable` 要求 `async_enable`，见 §4④）。

⚠️ 而 2025/4 的「基于单控制器重构」说明：**单控制器不是从一开始就有的**，是这个领域走了一段之后才收敛出来的范式。[14 章](../训练框架/分布式训练入门/14-RL训练框架.md) 讲的 HybridFlow 单/多控制器混合，在 OpenRLHF 这里是一次明确的重构事件。

---

## 8 · 怎么读这份源码

⚠️ 我的建议路径（按依赖顺序，读完约需一两个小时）：

```
① openrlhf/trainer/ppo_trainer.py   L499 fit()        ← 同步主循环，20 行看懂 RL
② openrlhf/trainer/ppo_trainer_async.py                ← 异步版，看三处差别（§4）
③ openrlhf/utils/agent.py           356 行             ← agent 抽象全貌
④ openrlhf/models/loss.py           335 行             ← 算法都在这
⑤ openrlhf/cli/train_ppo_ray.py     L660-700           ← ★ 参数间的约束关系
```

⚠️ **第 ⑤ 步最容易被跳过，但信息量最大。** 那几十行断言把「哪些配置组合是非法的」写得清清楚楚，而每一条非法组合背后都是一个架构约束。读参数校验，往往比读实现更快理解一个框架的边界。

---

## 9 · 本篇小结

| 结论 | 分级 |
|---|---|
| 13,411 行，是 veRL 的 1/12.8；差距在后端适配而非算法 | ✅ 实测 + ⚠️ |
| Ray + vLLM + DeepSpeed 三件套；README 自陈「生成占 80% 时间」 | ✅ 原文 |
| **Hybrid Engine**：全部同置 + 换入换出，与 GLM 的「分离 + 异步」目标同、手段反 | ✅ + ⚠️ |
| 同置开关成对出现：actor+ref、critic+reward，对应 16Ψ/2Ψ 的显存搭配 | ✅ + ⚠️ |
| 同步 `fit()` 20 行说完 buffer 深度 0 | ✅ L499 |
| **`--train.async_queue_size`（默认 1）就是 buffer 深度参数** | ✅ `train_ppo_ray.py:275` |
| 用 `rollout_queue` + `rollout_slots` 计数信号量硬保证深度上界 | ✅ L292-297 |
| **`VLLMLock`** 明确拒绝「永不停」：一批样本必须由一致的权重生成 | ✅ L20-34 |
| **`pause_generation`/`resume_generation`** = 中断模型第 3 档软暂停，且只在 partial rollout 下启用 | ✅ L254-265 |
| 参数断言把架构约束写死：异步禁 sleep、partial rollout 依赖异步、超采样依赖队列 | ✅ L667-698 |
| `agent.py` 仅 **356 行**实现整个 agent 抽象（veRL 对应部分 3,424 行，差 9.6 倍） | ✅ 实测 |
| 用户 agent 代码挂在**推理引擎侧**，离生成最近离训练最远——与 GLM/slime 同一原则 | ✅ + ⚠️ |
| 「执行模式 × RL 算法」正交表 + TITO 作为架构基石 | ✅ 原文 |
| 6 个算法 335 行 vs veRL 26 个注册项——定位差异，不是能力差异 | ✅ + ⚠️ |
| `replay_buffer.py` 仅 177 行：同步框架的 buffer 本来就简单 | ✅ + ⚠️ |
| **2025/5 异步与 agent 支持同版本落地**——agent 在工程上以异步为前提 | ✅ + ⚠️ |
| 单控制器是 2025/4 的一次重构，不是与生俱来 | ✅ |
| 读源码先读 `train_ppo_ray.py:660-700` 的参数断言，比读实现更快理解边界 | ⚠️ 建议 |

**下一篇** → [07-选型与实践.md](07-选型与实践.md)
