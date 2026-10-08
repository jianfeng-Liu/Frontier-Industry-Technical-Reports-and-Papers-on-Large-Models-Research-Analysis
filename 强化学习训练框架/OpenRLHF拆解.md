# OpenRLHF 拆解：架构最直白的那一个

> **这一篇干什么**：源码级走查。OpenRLHF 是本目录里代码量最小的框架，**因此它是理解 RL 框架骨架的最佳入口**。
> **难度** ★★
> **✅ 本文所有行号钉在 commit `3c3be62`**（本地 clone 于 [源码/OpenRLHF/](源码/OpenRLHF/)，tag `v0.11.0`，2026-08-13）。
> ⚠️ **RL 框架迭代极快，行号几个月就会失效。** 本文每条引用都同时给出**文件名 + 符号名**（类名 / 函数名 / 参数名），换版本时请按符号名重新定位，不要照抄行号。

---

## 0 · 七问速答：和 slime 并排看

这七个问题和 [slime拆解.md](slime拆解.md) §0、[MILES拆解.md](MILES拆解.md) §0 是同一组，把三张表并排放着就能横向比。

| # | 问题 | OpenRLHF 的答案 | slime 的答案（对照） | 展开于 |
|---|---|---|---|---|
| **Q1** | 架构档位 | **同步 + 单步/有界异步**，由 `--train.async_enable` + `--train.async_queue_size` 一个整数控制；**没有全异步** | 同步 / 单步异步 / 全异步三档三套代码 | §5 |
| **Q2** | 训练后端 | **只有 DeepSpeed**（ZeRO-3 + AutoTP + RingAttention） | 只有 Megatron-LM | §2 |
| **Q3** | rollout 引擎 / 怎么通信 | **只有 vLLM**。通过 **Ray actor 调用** + 一个 `Queue`（Ray 的队列 actor）在采样器和训练器之间传样本 | 只有 SGLang，Ray ObjectRef | §5 |
| **Q4** | 权重同步怎么做 / 停不停推理 | **逐参数流式**：NCCL 广播或 CUDA IPC 句柄，一个参数一次。**默认停**（互斥锁等生成完）；**开了 partial rollout 才不等**（并接受一条轨迹横跨两个版本） | 四条通路可选；排空后再换 | §6 |
| **Q5** | 算法 / 加新算法动哪些文件 | 6 个优势估计器，全部在 `models/loss.py`（335 行）。加一个要动 **3 个文件**，没有「自定义 loss 路径」逃生口 | 6 个估计器；自定义 loss **0 个文件** | §8 |
| **Q6** | 沙箱 / 环境怎么接 | 写一个 `AgentExecutorBase` 子类，用 `--train.agent_func_path` 指过去；**代码挂在推理引擎侧** | 接成数据生成函数，挂在 rollout 侧 | §7 |
| **Q7** | 一次迭代的时序 / 谁在等 | **三种部署三种时序**，其中 `--train.colocate_all`（Hybrid Engine）是本目录独一份：同一批卡上训练和推理**轮流换入换出** | 分离为主，异步强制分离 | §3 |

⚠️ 一眼能看出的定位差异：**slime 把自由度留给算法研究者**（自定义 loss 路径、四条权重同步通路），**OpenRLHF 把自由度留给「卡少的人」**（Hybrid Engine 全同置）。

---

## 1 · 规模实测：小一个数量级

✅ 本地 clone 后实测（commit `3c3be62`）：

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

⚠️ **12.8 倍的差距不在算法，在后端适配。** [01 章](01-格局全景.md) 已经论证过这点：veRL 接了 6 个训练引擎 × 4 个推理引擎 × 6 个 checkpoint 后端；OpenRLHF 只认 **DeepSpeed + vLLM** 一条路。

**这不是缺点，是定位。** 如果你要在多种硬件、多种引擎上跑，veRL 是唯一选择；如果你要读懂 RL 框架到底在干什么，**从 13,411 行开始远比从 171,584 行开始现实**。

---

## 2 · ✅ 三个基础设施组件

OpenRLHF 的 README 把技术栈说得很明确：

| 组件 | 全称 | 角色 | ✅ 原文要点 |
|---|---|---|---|
| **Ray** | — | 分布式调度与控制 | *"separates the Actor, Reward, Reference, and Critic models across different GPUs, enabling scalable training for models up to **70B+ parameters**"* |
| **vLLM** | — | 高性能推理 | *"RLHF training spends **80% of the time on sample generation**"* |
| **DeepSpeed** | — | 显存高效训练 | ZeRO-3 + deepcompile + AutoTP + RingAttention |
| **NCCL / CUDA IPC** | NVIDIA Collective Communications Library / CUDA Inter-Process Communication | 高速通信 | 见 §6 与 [03 章](03-权重同步.md) |

✅ 那句「生成占 80% 时间」是本目录反复引用的那个数字的**框架侧一手出处**（[02 章](02-架构主轴-同步到全异步.md) 用的是 🌐 博客给的量化数据，这里是框架 README 的直接陈述）。

### 2.1 ★ Hybrid Engine：另一条省钱的路

✅ README 原文：

> *"**Hybrid Engine Scheduling**: All models and vLLM engines can **share GPU resources**—minimizing idle time and maximizing GPU utilization. This allows running full RLHF pipelines on **limited hardware**."*

⚠️ 这是 [02 章](02-架构主轴-同步到全异步.md) 那条轴上被忽略的一个位置：**GLM-5 靠「分离 + 异步」提高利用率，OpenRLHF 的 Hybrid Engine 靠「全部同置 + 换入换出」提高利用率。** 两者的目标一样（别让卡闲着），手段完全相反。

⚠️ 而且注意最后一句 *"on limited hardware"*——这和 ✅ Kimi-K3 的 *"within a few hundred GPUs"*（[01 章](01-格局全景.md)）是同一个约束下的同一类答案。**卡少的时候，同置几乎总是对的。**

✅ 源码里对应的是四个开关（定义在 `openrlhf/cli/train_ppo_ray.py`，行号见右列）：

| 开关 | 行号 | 作用 |
|---|---|---|
| `--train.colocate_actor_ref` | 207 | actor 和 reference 共享 GPU |
| `--train.colocate_critic_reward` | 218 | critic 和 reward 共享 GPU |
| `--train.colocate_all` | 224 | 全部共享（含 vLLM 引擎） |
| `--vllm.enable_sleep` | 245 | ★ vLLM 睡眠模式（不用时释放显存） |

⚠️ 前两个开关很有意思：**actor 和 reference 是天然的一对**（reference 只做前向，不占优化器状态），**critic 和 reward 也是一对**。这个配对不是随便定的——它对应 [训练框架/分布式训练入门/14-RL训练框架.md](../训练框架/分布式训练入门/14-RL训练框架.md) 那张显存账里 16Ψ（训练态）和 2Ψ（冻结）的搭配：**把一个 16Ψ 的和一个 2Ψ 的放一起，显存曲线才不会两头撞峰。**

✅ 而第四个开关有一条自动纠偏（`train_ppo_ray.py:663-665`）：

```python
if args.vllm.enable_sleep and not args.train.colocate_all:
    print("Set args.vllm.enable_sleep to False when args.train.colocate_all is disabled.")
    args.vllm.enable_sleep = False
```

⚠️ **睡眠模式只在全同置下有意义**——分离部署时推理卡本来就没人跟它抢显存，让它睡觉纯属自找麻烦。框架直接替你把这个开关按回去了。

---

## 3 · ★★★ 一次迭代的完整时序：三种部署，三种等法

OpenRLHF 的特别之处是它**三种部署都支持**，而三种部署的时序完全不同。这一节把三张图放在一起。

### 3.1 部署 A · 全同置（Hybrid Engine，`--train.colocate_all`）

同一批卡，训练和推理轮流用。切换时要把对方的显存吐出来。

```
  同一批 GPU，时间 ------------------------------------------->

  GPU   |## vLLM generate ##|<swap>|## DeepSpeed train ##|<swap>|=S=|

  swap  = vLLM sleep (释放 KV cache + 权重) / DeepSpeed reload states
  =S=   = weight sync (这里是最便宜的一档：同卡，走 CUDA IPC)
```

| 阶段 | 发生了什么 | 代价 |
|---|---|---|
| generate | vLLM 独占显存做生成 | DeepSpeed 的优化器状态此时在 CPU 上（`offload_states()`） |
| swap → train | `--vllm.enable_sleep` 让 vLLM 释放 KV cache 和权重；DeepSpeed `reload_states()` 把优化器状态搬回 GPU | ⚠️ **这一下是纯开销**，两边都没在算 |
| train | DeepSpeed 独占显存做梯度更新 | vLLM 睡着 |
| swap → generate + sync | DeepSpeed `offload_states()`；权重通过 CUDA IPC 交给同卡的 vLLM | 同卡 IPC 几乎零传输成本 |

⚠️ **全同置的利用率高，不是因为没有空转，而是因为「空转的那张卡」被另一个角色占着。** 代价转移到了 swap：每次切换都要搬优化器状态和 KV cache。✅ 源码里这两个动作是 `openrlhf/trainer/ray/ppo_actor.py` 的 `offload_states()` / `reload_states()`。

### 3.2 部署 B · 分离 + 同步（默认）

```
  time ------------------------------------------------------------>

  TRAIN   |..........idle...........|#### train ####|=S=|
  ROLLOUT |##### generate ##########|.... idle .....|=S=|
```

✅ 这就是 `ppo_trainer.py` 的 `fit()`：`generate_samples()` 不返回，训练什么都干不了。和 slime `train.py:53` 的 `ray.get(...generate...)` 是**完全一样的形状**（见 [slime拆解.md](slime拆解.md) §4.1）。

### 3.3 部署 C · 分离 + 异步（`--train.async_enable`）

```
  time ------------------------------------------------------------>

  TRAIN   |..idle(warmup)..|### train N ###|### train N+1 ###|
  ROLLOUT |### gen N ######|### gen N+1 ###|### gen N+2 #####|
                           |
                           +--> rollout_queue, maxsize = --train.async_queue_size
```

⚠️ 和 slime 异步版的区别在**中间那个 queue 是显式的**：slime 用一个 Python 变量 `rollout_data_next_future` 当深度 1 的缓冲，OpenRLHF 用一个真的 `Queue(maxsize=K)`，**所以它的 buffer 深度可以调到大于 1**。细节见 §5①。

### 3.4 ⚠️ 本文推演：三种部署该怎么选

用 [大模型算子开发/10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8.2 的 8B 分阶段耗时（rollout 32.4 s、训练侧 17.6 s、合计 50.6 s）推一次。⚠️ **以下是本文推算，不是 OpenRLHF 项目公布的数字。**

| 部署 | 需要多少卡 | 稳态迭代时间 | 相对同步分离 |
|---|---|---|---|
| A 全同置 | **N 张**（一份卡干两件事） | 50.6 s + swap 开销（⚠️ 实测才知道，无公开数字） | 卡数减半，时间略增 |
| B 分离同步 | **2N 张**（粗略按训练卡 : 推理卡 = 1 : 1） | 50.6 s | 基准 |
| C 分离异步 | 2N 张 | `max(17.6, 32.4)` = 32.4 s | 时间减少 36.0%，**加速 1.56 倍** |

⚠️ 换算提醒：**「时间减少 36%」不等于「加速 36%」**，`加速倍数 = 1 / (1 − 时间减少比例) = 1 / 0.64 = 1.56`。这条换算在 [附-速查表.md](附-速查表.md) 末尾有专门一节。

⚠️ 结论很简单：**卡少选 A，卡够且追吞吐选 C，调试和对拍选 B。** 这也正是 README 那句 *"on limited hardware"* 想说的事。

---

## 4 · ★★★ 同步主循环：20 行读懂 PPO

✅ `openrlhf/trainer/ppo_trainer.py` 的 `fit()`（L499 起，全文件 595 行）是全目录**最容易读懂的 RL 主循环**：

```python
for episode in range(start_episode, self.args.train.num_episodes):      # L516
    while True:
        # ① 生成一批样本（阻塞）
        t_gen_start = time.time()
        rollout_samples, filter_pass_rate, prompts_consumed, is_exhausted = (
            self.samples_generator.generate_samples(**self.generate_kwargs)
        )
        generation_time = time.time() - t_gen_start

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

⚠️ **这就是 [02 章](02-架构主轴-同步到全异步.md) 的「buffer 深度 0」，20 行说完。** `generate_samples()` 是阻塞的，它返回之前训练什么都干不了。

✅ 两个值得注意的细节：

1. **`status["timing/generation"]` 被单独计时**——框架自己就认为「生成占多久」是首要指标。
2. **`filter_pass_rate`** 是 `generate_samples()` 的返回值之一，配 `--algo.dynamic_filtering_enable`。这就是 [05 章](05-算法与框架的接口.md) 讲的动态过滤，**而 pass rate 被当作一个要记录的训练指标**——⚠️ 说明这个数字在实践中确实需要盯着。

---

## 5 · ★★★ 异步主循环：四个源码级的概念印证

✅ `openrlhf/trainer/ppo_trainer_async.py`（353 行）是本目录**印证密度最高的一个文件**。

### 5.1 ① `--train.async_queue_size` 就是 buffer 深度

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

★ **[02 章](02-架构主轴-同步到全异步.md) 那张「buffer 深度四档」的表，在这里是一个整数参数。**

| 配置 | 对应哪一档 |
|---|---|
| 不开 `--train.async_enable` | 深度 0（同步） |
| `--train.async_queue_size 1`（默认） | **深度 1**（一步离策略） |
| `--train.async_queue_size K` | **深度 2~K**（有界队列） |

⚠️ 而且实现用了**两个队列**：`rollout_queue` 装样本，`rollout_slots` 是一个**计数信号量**（counting semaphore，一种「令牌池」：拿到令牌才能干活，干完还回去）——生成侧必须先拿到一个 slot 才能开始生成。这正是 [02 章](02-架构主轴-同步到全异步.md) 讲的「深度限制：队列满了就阻塞生成」的标准实现。**深度上界是硬保证的，不是靠速率碰运气。**

### 5.2 ② `VLLMLock` 是默认路径的中断模型

✅ `ppo_trainer_async.py:19-35`：

```python
@ray.remote(num_cpus=0)
class VLLMLock:
    """Cross-actor mutex for vLLM critical section.

    Ensures generation and weight broadcast do not overlap on the same vLLM engines,
    so every sample in a batch is generated with consistent weights.
    """
```

⚠️ 注释的最后半句是重点：*"so every sample in a batch is generated with **consistent weights**"*（保证一个批次里的每个样本都由一致的权重生成）。

**这是默认路径上的一个明确架构立场**：不开 partial rollout 时，OpenRLHF 不接受 [03 章](03-权重同步.md) 的「永不停」档——它宁可用一把互斥锁把生成和权重广播隔开，也不让一条轨迹跨两个版本。

### 5.3 ③ ⚠️ 要纠正的一条：partial rollout 下，这个保证被**主动放弃**了

✅ `ppo_trainer_async.py:255-265`：

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

本文早先的版本把这段读成「`pause_generation` / `resume_generation` = 软暂停档，所以 OpenRLHF 始终拒绝跨版本轨迹」。✅ **对照参数的 help 文本，这个读法不完整**——`train_ppo_ray.py` 的 `--train.partial_rollout_enable` 原文是：

> ✅ *"Enable partial rollout in async mode. Uses vLLM **pause/resume for weight sync instead of locking**, allowing generation to **overlap with training**. **In-flight samples may contain tokens from both old and new weights.**"*

⚠️ **最后一句是明说的**：开了 partial rollout，一条在飞的轨迹**可以**横跨两个权重版本。所以准确的说法是：

| 模式 | 生成要不要等 | 一条轨迹会不会跨版本 | 对应中断模型 |
|---|---|---|---|
| 异步，**不开** partial rollout | **要等**（生成侧持锁，广播要排队） | ✅ 不会 | 「等生成完」档 |
| 异步，**开** partial rollout | 不等（生成侧不持锁，只被 pause 一下） | ⚠️ **会，而且是设计如此** | 「软暂停」档 |

⚠️ **这是本次走查改掉的最主要一处错误**：旧版把「框架提供了一个可选档」写成了「框架的架构立场」。OpenRLHF 的真实立场是**两档都给，让你选**，默认那一档保守。

### 5.4 ④ 参数之间的约束关系写进了断言

✅ `openrlhf/cli/train_ppo_ray.py:663-698` 这一段，把本目录几章的结论直接写成了运行时检查：

| 行号 | 检查 | 印证了什么 |
|---|---|---|
| 663-665 | `if enable_sleep and not colocate_all:` 自动改回 False | 睡眠模式只在全同置下有意义 |
| 667-668 | `if colocate_all and async_enable:` 打印警告 *"only colocates DeepSpeed models"* | 异步下 vLLM 引擎不能再和训练同置 |
| 670-671 | `if async_enable: assert not enable_sleep` | 异步模式下推理引擎不能睡——它得一直在生成 |
| 673-674 | `if partial_rollout_enable: assert async_enable` | **partial rollout 依赖异步**，同步模式下没有「下一轮接着生成」的概念 |
| 694-698 | `if vllm_generate_batch_size > rollout.batch_size: assert async_enable`<br>✅ *"(over-sampling needs async queue to buffer extra batches)"* | ★ **超采样必须有队列接着** |

⚠️ 667-668 那条和 slime 的 `assert not args.colocate`（[slime拆解.md](slime拆解.md) §5.2）是同一个约束的两种态度：**slime 直接禁止，OpenRLHF 给个警告然后部分支持**（DeepSpeed 的几个模型仍然同置，vLLM 不再同置）。slime 更硬，OpenRLHF 更宽松——⚠️ 对新手来说 slime 那种「早失败」更友好。

---

## 6 · ★★ 权重同步：逐参数流式，两条通路

[03 章](03-权重同步.md) 讲权重同步时提到「全量广播 vs 逐参数流式 vs 共享显存」三种形态。OpenRLHF 的实现**一个文件就能读完**：`openrlhf/trainer/ray/ppo_actor.py` 的 `broadcast_to_vllm()`（L408 起）。

### 6.1 它是逐参数的，不是一次性整包

✅ 函数体里是一个对参数的循环，每个参数调一次 `_broadcast_param()` 或 `_handle_cuda_ipc()`：

```python
def _broadcast_param(param, count, num_params):
    if torch.distributed.get_rank() == 0:
        shape = param.shape if zero_stage != 3 else param.ds_shape
        refs = [
            engine.update_weight.remote(name, dtype=param.dtype, shape=shape,
                                        empty_cache=count == num_params)
            for engine in self.vllm_engines
        ]
        self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
        ray.get(refs)
```

⚠️ 三个细节值得记：

1. **只有 rank 0 发。** 和 slime 的 `DP=0 且 TP=0` 才发是同一个优化——否则 DP 组里每个副本都广播一遍，流量翻 DP 倍。
2. **`empty_cache=count == num_params`**：只在**最后一个**参数上让引擎清一次显存缓存。逐参数清的话，每个参数都要同步一次，开销不可接受。
3. **ZeRO-3 下要用 `param.ds_shape`**：ZeRO-3 把参数切片存着，`param.shape` 看到的是切片的形状，完整形状在 `ds_shape` 里。⚠️ 这是 resharding（重切分）在 DeepSpeed 侧的具体样子，对照 [03 章](03-权重同步.md)。

### 6.2 两条通路

| 通路 | 触发条件 | 怎么传 | 适合 |
|---|---|---|---|
| **NCCL 广播** | 默认 | 训练进程组和 vLLM 引擎组成一个 `_model_update_group`，rank 0 广播 | 分离部署、跨机 |
| **CUDA IPC** | 同机同卡 | `reduce_tensor()` 做出 IPC 句柄 → `all_gather_object` 收齐 → 交给引擎的 `update_weight_cuda_ipc()` | 全同置（Hybrid Engine） |
| **Ray collective** | `--vllm.sync_with_ray` | 用 `ray.util.collective.broadcast` 替代 PyTorch 的进程组 | 和 Ray 调度耦合更紧时 |

⚠️ **CUDA IPC 那条通路才是 Hybrid Engine 便宜的真正原因**：权重根本没离开这张卡的显存，传的只是一个「指针 + 元数据」的句柄。对照 [附-速查表.md](附-速查表.md) §3.1：这属于「机内」那一行，**一点 RDMA 都不沾**。

### 6.3 一个容易漏的副作用：prefix cache 必须清

✅ `broadcast_to_vllm()` 的头几行：

```python
use_prefix_cache = getattr(self.strategy.args.vllm, "enable_prefix_caching", False)
if use_prefix_cache and torch.distributed.get_rank() == 0:
    for engine in self.vllm_engines:
        cache_reset_refs.append(engine.reset_prefix_cache.remote())
```

⚠️ **换了权重，旧权重算出来的 prefix cache（前缀缓存）就全部失效了**——它缓存的是「这段前缀在旧权重下的 KV」。不清的话，新权重会读到旧权重的中间结果，这正是 [04 章](04-训推一致性.md) 说的那种「沉默的 bug」：不报错，只是慢慢训歪。**这三行是权重同步里最容易被自研框架漏掉的一条。**

---

## 7 · ★★ Agent 抽象：把「多轮」和「算法」拆成正交两维

这是 OpenRLHF 自称的**核心创新**，也是本目录 [05 章](05-算法与框架的接口.md) 引过的那个设计。

✅ README 原文：*"OpenRLHF is **the first RLHF framework** to implement a **unified agent-based paradigm**. Every training run—whether standard PPO or complex multi-turn reasoning—follows a consistent agent execution pipeline."*

✅ 源码里就是 `openrlhf/utils/agent.py`（**356 行**），三个类：

| 行号 | 类 | 关键方法 | 说明 |
|---|---|---|---|
| 12 | `AgentExecutorBase(ABC)` | `execute()`（抽象） | 抽象基类，TITO 的核心 |
| 31 | `MultiTurnAgentExecutor` | `reset()` + `step()` | 多轮：标准 RL 环境接口 |
| 184 | `SingleTurnAgentExecutor` | 可选 `reward_func()` | 单轮：默认路径 |

⚠️ **356 行就实现了整个 agent 抽象层。** 对比 veRL 的 `agent_loop/` 2,831 行 + `tools/` 593 行（[06 章](06-AgenticRL-环境与沙箱.md)）——差 9.6 倍。差距主要在 veRL 有 816 行的 `tool_parser.py`，而 OpenRLHF 把工具解析留给用户自己实现的 executor。

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

⚠️ 这个设计的关键在于**它把用户代码放在了推理引擎那一侧**（`ray/vllm_engine.py`），而不是训练侧。**用户写的 agent 逻辑离生成最近、离训练最远**——这和 ✅ GLM-5 的 *"cleanly isolates task-specific logic from the core training loop"*（[06 章](06-AgenticRL-环境与沙箱.md)）、以及 ✅ slime README 的 *"They do not fork the training kernel"*（[slime拆解.md](slime拆解.md) §2.3）是同一条原则。**三家（GLM 报告 / slime README / OpenRLHF 源码）独立表述了同一件事。**

✅ 还有一条联动：`train_ppo_ray.py:596-597`，只要设了 `agent_func_path`，奖励来源自动切成 `"agent"`：

```python
if args.train.agent_func_path:
    args.reward.remote_url = "agent"
```

⚠️ 也就是说在 OpenRLHF 里，**「接环境」和「接奖励」是同一个口子**——环境执行完自己给分，不需要另外挂一个奖励模型。这正是 RLVR（可验证奖励强化学习）的架构形态。

### 7.1 ✅ 正交性写进了 README

✅ 原文标题就是：*"Two Execution Modes (**Orthogonal to RL Algorithms**)"*。原表是一张 `✓` 网格，这里改写成 markdown 表（✓ 属于 Ambiguous 宽度字符，放在要列对齐的 ASCII 图里会错位）：

| 算法 \ 执行模式 | Single-Turn（默认，99% 场景） | Multi-Turn（`reset` + `step`） |
|---|---|---|
| PPO | 支持 | 支持 |
| REINFORCE++ | 支持 | 支持 |
| REINFORCE++-baseline | 支持 | 支持 |
| GRPO | 支持 | 支持 |
| RLOO | 支持 | 支持 |
| Dr. GRPO | 支持 | 支持 |

✅ 并且 README 明确写了 *"**Token-in-Token-out**: All sampling produces token-level trajectories → **Zero text-level mismatch**"*——即 [04 章](04-训推一致性.md) 讲的 **TITO**（Token-in-Token-out，token 进 token 出：训练直接吃推理吐出的 token id，不经过「解码成文字再重新分词」），在 OpenRLHF 里是**架构基石**而不是可选项。

⚠️ 这也是它敢把 agent 抽象做到只有 356 行的原因之一：**因为不做文本层的转换，就不需要处理转换带来的一大堆边界情况。**

---

## 8 · 算法与损失：都在 `models/loss.py`

✅ `openrlhf/models/`（6 文件 / 1,352 行）：

| 文件 | 行数 |
|---|---|
| `loss.py` | **335** |
| `model.py` | 319 |
| `actor.py` | 305 |
| `utils.py` | 186 |
| `ring_attn_utils.py` | 182 |

### 8.1 六个优势估计器

✅ `openrlhf/cli/train_ppo_ray.py:478-481`，`--algo.advantage.estimator` 的 choices：

| 参数值 | 算法 | 中文 | 要不要 critic |
|---|---|---|---|
| `gae`（**默认**） | PPO + GAE | 广义优势估计 | **是** |
| `reinforce` | REINFORCE++ | — | 否 |
| `reinforce_baseline` | REINFORCE++-baseline | 均值基线，适合 RLVR 推理任务 | 否 |
| `rloo` | RLOO | 留一法 REINFORCE | 否 |
| `group_norm` | GRPO | 分组相对策略优化 | 否 |
| `dr_grpo` | Dr. GRPO | 去掉组内 `/std` 归一化 | 否 |

✅ `train_ppo_ray.py:599-600` 一行就把 critic 的去留决定了：

```python
if args.algo.advantage.estimator not in ["gae"]:
    args.critic.model_name_or_path = None
```

⚠️ 和 slime 的 `args.use_critic = args.advantage_estimator == "ppo"` 是**同一个判断的两种写法**：六选一里只有一个需要 critic。[05 章](05-算法与框架的接口.md) 讲的「去 critic 潮流」，在两家框架里都退成了「一个取值的特例」。

### 8.2 ★ 一条写进断言的沉默 bug

✅ `train_ppo_ray.py:606-615`，这条注释值得整段抄下来：

```python
# These estimators compute a per-prompt-group baseline (mean / std / leave-one-out),
# so with n_samples_per_prompt == 1 every advantage collapses to 0 and training is a
# silent no-op. dr_grpo subtracts the group mean too (see experience_maker), so it
# belongs here as well.
if args.algo.advantage.estimator in ["rloo", "reinforce_baseline", "group_norm", "dr_grpo"]:
    assert args.rollout.n_samples_per_prompt > 1, \
        f"{args.algo.advantage.estimator} requires n_samples_per_prompt > 1"
```

⚠️ **「每组只采 1 个样本 + 组内基线」= 优势全部等于 0 = 训练是个静默的空操作。** 这正是 slime README 那句 *"RL bugs are often silent"*（RL 的 bug 通常是沉默的）的教科书例子：不报错、loss 有数、曲线也在动，**但梯度是零，模型纹丝不动**。

⚠️ **OpenRLHF 把它写成了启动时的断言。** 这是本文认为最值得抄走的一条工程实践：**凡是「会静默失效」的参数组合，都应该在启动时就 assert 掉，而不是留给你三天后看曲线纳闷。**

### 8.3 ✅ 训推不一致的修正也是一个参数

✅ `train_ppo_ray.py:255-271`，三个参数一组：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--algo.advantage.is_correction_enable` | False | 开关 |
| `--algo.advantage.is_correction_threshold` | `[0.5, 5.0]` | 重要性采样比值的上下截断阈值 |
| `--algo.advantage.is_correction_type` | `tis` | 三选一：`tis` / `icepop` / `seq-mask-tis` |

✅ 三种修正方式的 help 原文：

| 取值 | 全称 | ✅ 原文说明 | 粒度 |
|---|---|---|---|
| `tis` | Truncated Importance Sampling（截断重要性采样） | *"token-level clamp"* | token 级，**截断** |
| `icepop` | — | *"token-level filter"* | token 级，**丢弃** |
| `seq-mask-tis` | — | *"sequence-level geom mean"* | 序列级，几何平均 |

⚠️ 注意 `tis` 和 `icepop` 的区别：**截断是把超界的比值压回阈值（样本保留），过滤是把超界的 token 直接扔掉（样本部分丢失）。** 这个区分在 [04 章](04-训推一致性.md) 讲 IS 修正时是核心取舍。

✅ 源码里这组参数上方还留了一条出处注释，是 🌐 一篇博客：*"Your Efficient RL Framework Secretly Brings You Off-Policy RL Training"*（`https://fengyao.notion.site/off-policy-rl`）。⚠️ 这条是源码注释里的引用，本文**没有去核对博客内容**。

### 8.4 加一个新算法要动哪些文件

| 你要加什么 | 要动的文件 | 量级 |
|---|---|---|
| 一个新的优势估计器 | ① `cli/train_ppo_ray.py` 的 choices（L480）；② `trainer/ppo_utils/experience_maker.py` 算优势；③ 可能还要在 L599 那条 critic 判断里加例外 | **3 个文件** |
| 一个新的策略损失 | `models/loss.py` 的 `PolicyLoss`（L116） | **1 个文件** |
| 一个完全自定义的 loss | ⚠️ **没有逃生口**——OpenRLHF 没有 slime 那种 `--custom-loss-function-path`，只能改仓库 | **必须 fork** |

⚠️ **这是 slime 和 OpenRLHF 最实在的一处定位差异**：slime 留了「不改仓库也能换损失函数」的口子（见 [slime拆解.md](slime拆解.md) §10.2），OpenRLHF 没有。⚠️ 对照 veRL 的 14 个优势估计器 + 12 个策略损失（[05 章](05-算法与框架的接口.md)）：**OpenRLHF 只有 6 个，但 335 行就写完了。** veRL 要做算法研究的公共平台，OpenRLHF 要做「能跑起来的最短路径」。

### 8.5 PPO 周边

✅ `openrlhf/trainer/ppo_utils/` 里几个值得注意的文件：

| 文件 | 行数 | 说明 |
|---|---|---|
| `experience_maker.py` | 411 | 把 rollout 变成训练用的 experience（优势在这里算） |
| `samples_generator.py` | 314 | 生成侧 |
| `experience.py` | 283 | experience 数据结构 |
| `replay_buffer.py` | **177** | ★ 回放缓冲区 |
| `length_penalty.py` | 153 | 长度惩罚 |
| `kl_controller.py` | 29 | KL 系数自适应 |

⚠️ `replay_buffer.py` 只有 177 行，而它就是 [01 章](01-格局全景.md) 五模块里的 data buffer 那一块。**在同步框架里，data buffer 确实只需要这么多代码**——复杂度全在异步之后才出现（对照 slime 的 `data_source.py` 229 行 + `fully_async_rollout.py` 274 行）。

---

## 9 · ✅ 版本时间线：能看出这个领域的演进

✅ README 的 News 部分是一条有价值的行业时间线：

| 时间 | 事件 | ⚠️ 对应本目录哪个主题 |
|---|---|---|
| 2024/12 | 提出 **REINFORCE++** | 去 critic 潮流（[05 章](05-算法与框架的接口.md)） |
| 2025/4 | *"Clean OpenRLHF: Refactored based on **Single Controller** and Unified Packing Samples"* | 单控制器范式（[14 章](../训练框架/分布式训练入门/14-RL训练框架.md)） |
| 2025/5 | v0.8.0：`--train.async_enable`（异步 RLHF）+ `--train.agent_func_path`（异步 agent RLHF） | ★ **异步与 agent 同一个版本落地** |
| 2025/10 | ScaleRL | — |
| 2026/2 | ProRL V2 | — |
| 2026/4 | v0.10：多轮 VLM RL + VLM RLHF | 多模态 |

⚠️ **2025/5 那一行最值得注意：异步和 agent 支持是在同一个版本一起出来的。**

这不是巧合——[06 章](06-AgenticRL-环境与沙箱.md) 讲过，agentic 任务的轨迹又长又参差，同步架构根本扛不住。**agent 支持在工程上必须以异步为前提**，这条在 OpenRLHF 的断言里也写着（`partial_rollout_enable` 要求 `async_enable`，见 §5.4）。

⚠️ 而 2025/4 的「基于单控制器重构」说明：**单控制器不是从一开始就有的**，是这个领域走了一段之后才收敛出来的范式。[14 章](../训练框架/分布式训练入门/14-RL训练框架.md) 讲的 HybridFlow 单/多控制器混合，在 OpenRLHF 这里是一次明确的重构事件。

---

## 10 · 怎么读这份源码

⚠️ 我的建议路径（按依赖顺序，读完约需一两个小时）：

| 顺序 | 文件与位置 | 读它干什么 |
|---|---|---|
| ① | `openrlhf/trainer/ppo_trainer.py` L499 `fit()` | 同步主循环，20 行看懂 RL |
| ② | `openrlhf/trainer/ppo_trainer_async.py` | 异步版，看 §5 那四处 |
| ③ | `openrlhf/utils/agent.py`（356 行） | agent 抽象全貌 |
| ④ | `openrlhf/models/loss.py`（335 行） | 算法都在这 |
| ⑤ | `openrlhf/trainer/ray/ppo_actor.py` L408 `broadcast_to_vllm()` | 权重同步的全部实现 |
| ⑥ | `openrlhf/cli/train_ppo_ray.py` L596-700 | ★ **参数间的约束关系** |

⚠️ **第 ⑥ 步最容易被跳过，但信息量最大。** 那一百行断言把「哪些配置组合是非法的」写得清清楚楚，而每一条非法组合背后都是一个架构约束——§8.2 那条「组内基线 + 单样本 = 静默空操作」就藏在里面。**读参数校验，往往比读实现更快理解一个框架的边界。**

---

## 11 · 本篇小结

| 结论 | 分级 |
|---|---|
| 13,411 行，是 veRL 的 1/12.8；差距在后端适配而非算法 | ✅ 实测 + ⚠️ |
| Ray + vLLM + DeepSpeed 三件套；README 自陈「生成占 80% 时间」 | ✅ README 原文 |
| **Hybrid Engine**：全部同置 + 换入换出，与 GLM 的「分离 + 异步」目标同、手段反 | ✅ + ⚠️ |
| 同置开关成对出现：actor+ref、critic+reward，对应 16Ψ/2Ψ 的显存搭配 | ✅ 开关 + ⚠️ 原因推断 |
| 睡眠模式不开全同置就被自动改回 False | ✅ `train_ppo_ray.py:663` |
| **三种部署三种时序**：全同置靠 swap、分离同步靠等、分离异步靠队列 | ✅ 源码 + ⚠️ 时序为本文整理 |
| 同步分离 50.6 s → 异步 32.4 s，**时间减少 36.0% = 加速 1.56 倍** | ⚠️ **本文推演**（基于 [10-RL算子.md](../大模型算子开发/10-RL算子.md) §10.8.2） |
| 同步 `fit()` 20 行说完 buffer 深度 0 | ✅ `ppo_trainer.py:499` |
| **`--train.async_queue_size`（默认 1）就是 buffer 深度**，且可以大于 1 | ✅ `train_ppo_ray.py:275` |
| 用 `rollout_queue` + `rollout_slots` 计数信号量硬保证深度上界 | ✅ `ppo_trainer_async.py:292-297` |
| **`VLLMLock`**：默认路径上一批样本必须由一致的权重生成 | ✅ `ppo_trainer_async.py:19-35` |
| ⚠️ **纠正**：开了 partial rollout，`pause/resume` 取代锁，✅ help 明说 *"In-flight samples may contain tokens from both old and new weights"*——**跨版本轨迹是设计如此，不是 OpenRLHF 拒绝的东西** | ✅ 参数 help 原文（**纠正旧版说法**） |
| 参数断言把架构约束写死：异步禁 sleep、partial rollout 依赖异步、超采样依赖队列 | ✅ `train_ppo_ray.py:663-698` |
| **权重同步是逐参数流式**：rank 0 发、只在最后一个参数清缓存、ZeRO-3 下用 `ds_shape` | ✅ `ppo_actor.py:408` |
| 三条通路：NCCL 广播 / CUDA IPC（同卡，Hybrid Engine 便宜的原因）/ Ray collective | ✅ 实测 |
| 换权重必须清 **prefix cache**，否则是典型的「沉默 bug」 | ✅ 源码 + ⚠️ 后果为本文推断 |
| `agent.py` 仅 **356 行**实现整个 agent 抽象（veRL 对应部分 3,424 行，差 9.6 倍） | ✅ 实测 |
| 用户 agent 代码挂在**推理引擎侧**，离生成最近离训练最远——与 GLM/slime 同一原则 | ✅ + ⚠️ |
| 设了 `agent_func_path`，奖励来源自动变成 `"agent"`——接环境和接奖励是同一个口子 | ✅ `train_ppo_ray.py:596` |
| 「执行模式 × RL 算法」正交 + TITO 作为架构基石 | ✅ README 原文 |
| 6 个估计器里**只有 `gae`（PPO）要 critic** | ✅ `train_ppo_ray.py:599` |
| ★ **组内基线 + `n_samples_per_prompt == 1` = 优势全为 0 = 静默空操作**，被写成启动断言 | ✅ 源码注释原文 |
| 训推不一致的修正也是参数：`tis`（截断）/ `icepop`（过滤）/ `seq-mask-tis`（序列级几何平均） | ✅ `train_ppo_ray.py:255-271` |
| 加自定义 loss **必须 fork**——没有 slime 那种 `--custom-loss-function-path` | ✅ 实测 |
| `replay_buffer.py` 仅 177 行：同步框架的 buffer 本来就简单 | ✅ + ⚠️ |
| **2025/5 异步与 agent 支持同版本落地**——agent 在工程上以异步为前提 | ✅ README + ⚠️ |
| 单控制器是 2025/4 的一次重构，不是与生俱来 | ✅ README |
| 读源码先读 `train_ppo_ray.py:596-700` 的参数校验，比读实现更快理解边界 | ⚠️ 建议 |

**上一篇** → [slime拆解.md](slime拆解.md) ｜ **下一篇** → [07-选型与实践.md](07-选型与实践.md)
