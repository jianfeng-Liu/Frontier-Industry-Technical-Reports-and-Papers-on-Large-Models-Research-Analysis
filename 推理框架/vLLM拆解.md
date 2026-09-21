# vLLM 源码拆解

> **一句话**：整个 vLLM 就是四个对象在循环——`AsyncLLM` 收请求、`Scheduler.schedule()` 分 token 预算和 KV 块、`GPUModelRunner` 跑一次前向、`Scheduler.update_from_output()` 收结果判结束。**其余所有模块都挂在这条循环上。**

> **版本**：commit `bd8865a`（2026-08-20 拉取，本地副本在 [源码/vLLM/](源码/vLLM/)），4,280 个 `.py` 文件 / 1,432,376 行。

> ⚠️ **行号是我在这个 commit 上实际 `grep` 出来的，可以直接点开对照——但 vLLM 迭代极快，换个版本行号必然偏移。** 所以每处引用都同时写了**文件名 + 符号名**；行号对不上时用符号名搜，别信行号。

> **这篇不讲概念。** 分页与前缀缓存的原理看 [05 章](推理优化入门/05-PagedAttention与前缀缓存.md)，调度与 chunked prefill 看 [04 章](推理优化入门/04-连续批处理与调度.md)，并行策略看 [06 章](推理优化入门/06-推理中的并行策略.md)，算子层的账看 [大模型算子开发/09-推理侧算子.md](../大模型算子开发/09-推理侧算子.md)。**本文只回答一件事：那些机制在源码里长什么样。**

---

## 一、先看目录长什么样

```
    vllm/
    ├── v1/              ★★ 主战场。V1 是重写过的引擎，现在的默认路径
    │   ├── engine/      ★ 请求的入口和出口
    │   ├── core/        ★★ 调度器 + KV Cache 块管理  ← 本文重点
    │   ├── worker/      ★ 真正跑模型的地方（GPUModelRunner）
    │   ├── attention/   注意力后端（FlashAttention / FlashInfer / Mamba / 线性注意力 ...）
    │   ├── spec_decode/ ★ 投机解码
    │   ├── sample/      采样
    │   ├── structured_output/  结构化输出（JSON Schema 等）
    │   └── kv_offload/  KV 换出
    ├── distributed/     ★★ 并行状态、KV 传输、EPLB  ← 第五节的源头
    ├── model_executor/  ★ 各种并行线性层、量化
    ├── models/          具体模型定义
    ├── config/          ★ 配置（CUDAGraphMode 在这里）
    ├── engine/          V0 遗留（★ 读代码时别走错）
    └── entrypoints/     OpenAI 兼容的 HTTP 服务
```

> 🔑 **第一个坑**：`vllm/engine/` 是 **V0 的老代码**，`vllm/v1/engine/` 才是现在用的。目录名几乎一样，很容易读错文件。**本文讲的全部是 v1。**

> 📖 上面出现的两个缩写：**KV Cache**（Key-Value Cache，键值缓存）——把已算过的注意力 K/V 存下来，见 [02 章](推理优化入门/02-KV-Cache与显存账.md)；**EPLB**（Expert Parallelism Load Balancer，专家并行负载均衡器）——见第六节。**不认识的缩写都可以先去 [附-速查表](推理优化入门/附-速查表.md) 查。**

---

## 二、★★ 主线：一个请求走完全程

源码拆解最容易读成「模块罗列」。**先把这条主线走完，后面每一节才挂得上去。**

### 2.1 调用链：七幕

一个 HTTP 请求进来，到 SSE 流里吐出第一个字，依次经过这些函数。

> 📖 **SSE**（Server-Sent Events，服务器推送事件）——HTTP 上的单向流式协议，OpenAI 兼容接口用它做「一个字一个字往外冒」的效果。

```
    HTTP POST /v1/chat/completions
      │   entrypoints/openai/
      ▼
  【进程 A：前端异步进程】
  ① AsyncLLM.add_request()               v1/engine/async_llm.py:283
      ├─ 分词 → 建 Request 对象
      ├─ 在 OutputProcessor 里登记一个 RequestState   output_processor.py:131
      └─ 建一个 RequestOutputCollector 当出口          output_processor.py:47
      │
      ▼  经进程间队列（EngineCoreClient）送过去
  【进程 B：引擎进程】
  ② EngineCoreProc.run_busy_loop()       v1/engine/core.py:1391   ★ 永不停的主循环
      ├─ _process_input_queue()          core.py:1417   把新请求收进调度器
      └─ _process_engine_step()          core.py:1448
          │
          ▼
  ③ EngineCore.step()                    v1/engine/core.py:583    ★★ 一步的本体，只有 20 行
      ├─ Scheduler.schedule()            sched/scheduler.py:477   ★★ 决定这一步跑谁（第三节）
      ├─ GPUModelRunner.execute_model()  gpu_model_runner.py:4288  ★ 组张量、前向、采样
      └─ Scheduler.update_from_output()  sched/scheduler.py:1737  ★ 收结果、判结束、释放块
      │
      ▼  EngineCoreOutputs 经进程间队列回传
  【回到进程 A】
  ④ AsyncLLM._run_output_handler()       v1/engine/async_llm.py:665   后台常驻任务
      ▼
  ⑤ OutputProcessor.process_outputs()    v1/engine/output_processor.py:598
      └─ 增量 detokenize、判停止串、组 RequestOutput
      ▼
  ⑥ RequestOutputCollector.put()         output_processor.py:64
      ▼
  ⑦ AsyncLLM.generate() 里的 async for   v1/engine/async_llm.py:550
      ▼
    HTTP SSE 逐块吐字
```

> ★★ **注意这条链跨了两个进程**：①④⑤⑥⑦ 在异步前端进程，②③ 在引擎进程，中间是进程间队列。**这就是为什么调度器是纯同步代码**——它根本不在 asyncio 的世界里，不用担心 `await` 把循环切走。

### 2.2 `step()` 的内部时间线

`EngineCore.step()`（[core.py:583](源码/vLLM/vllm/v1/engine/core.py)）短到可以整段抄下来看清楚：

```python
# ✅ L592-613
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())   # L594
future = self.model_executor.execute_model(scheduler_output, non_block=True)   # L595 ★ 先不等
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)          # L596
with (...):
    model_output = future.result()                                            # L601 ★ 这里才等 GPU
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)       # L603
self._process_aborts_queue()                                                  # L607
engine_core_outputs = self.scheduler.update_from_output(                      # L608
    scheduler_output, model_output)
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

```
    ★ 时间线上真正重要的是两处「错位」：

    L595 execute_model(non_block=True)  ── 只把活扔给 GPU，立刻返回一个 future
      │
      │   ← ★★ 这段 CPU 时间是白赚的：
      ▼      L596 在算结构化输出的掩码，GPU 同时在跑前向
    L601 future.result()               ── 现在才真的等 GPU

    L607 _process_aborts_queue()       ── ★ 前向那几毫秒里到达的 abort，
                                          在写回结果【之前】统一处理
                                          ⟹ 避免给已取消的请求写 token
```

### 2.3 一个请求的状态怎么变

`RequestStatus`（[v1/request.py:359](源码/vLLM/vllm/v1/request.py)）是个 `IntEnum`，值的大小自己带语义：

```python
# ✅ v1/request.py:362-382
WAITING / WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR / WAITING_FOR_REMOTE_KVS
WAITING_FOR_STREAMING_REQ / RUNNING / PREEMPTED
# Note: anything after PREEMPTED will be considered as a finished status.
FINISHED_STOPPED / FINISHED_LENGTH_CAPPED / FINISHED_ABORTED
FINISHED_IGNORED / FINISHED_ERROR / FINISHED_REPETITION

@staticmethod
def is_finished(status):
    return status > RequestStatus.PREEMPTED        # ★★ 判结束就是一次比大小
```

> ★ **枚举顺序本身是契约**：`is_finished` 不是查表，是 `> PREEMPTED`。往这个枚举里插值得插在对的位置——插错地方，一个未结束的状态会被判成已结束。**读这类代码时，枚举的顺序要当代码看。**

下面把一个具体请求逐步跟完。⚠️ **参数是我为讲解构造的**：prompt 40 个 token、`block_size = 16`、`max_num_batched_tokens = 32`（故意开小，好触发 chunked prefill）、生成 3 个 token 后遇到 EOS。

| # | 这一步 `schedule()` 算出什么 | `status` | `num_computed_tokens` | 持有的 KV 块 | 这一步产出 |
|---|---|---|---|---|---|
| 入队 | — | `WAITING` | 0 | 0 | — |
| step 1 | `num_new_tokens = min(40-0, 32, …) = 32`，算 token 0..31 | `WAITING` → `RUNNING` | 0 → **32** | 申请 2 块，**两块都写满** | ★ 无输出 token（prompt 没算完） |
| — | `update_from_output` 之后 | `RUNNING` | 32 | 2 个满块被 `cache_full_blocks` **写进哈希表** | — |
| step 2 | `min(40-32, 32, …) = 8`，算 token 32..39 | `RUNNING` | 32 → **40** | 申请第 3 块，写到 8/16 | ★ **第 1 个输出 token**（TTFT 在这里定） |
| step 3 | `min(41-40, …) = 1`，纯 decode | `RUNNING` | 40 → **41** | 第 3 块 9/16 | 第 2 个输出 token |
| step 4 | 纯 decode | `RUNNING` → `FINISHED_STOPPED` | 41 → 42 | ★ `update_from_output` 里**全部释放** | EOS，流关闭 |

```
    ★★ 这张表里有三件事值得单独记住：

    ① 半块永远不进缓存表。step 2 之后第 3 块只有 8/16，
       它的哈希算出来下一步就失效了 ⟹ 只有满块能进（见第四节）

    ② 「第 1 个输出 token」出现在 prompt 算完的那一步，不是第一步。
       ⟹ 把 max_num_batched_tokens 开小，会把 TTFT 拆成多步累加

    ③ 释放不等于回收。块回到池子里、哈希还留在表里，
       ⟹ 下一个同前缀的请求还能命中它 —— 这才是前缀缓存的来源
```

> 📖 **TTFT**（Time To First Token，首 token 延迟）——从请求到达到用户看见第一个字的时间。指标口径见 [13 章](推理优化入门/13-性能指标与压测.md)。

> ★ **被抢占时，状态回到哪里**：`_preempt_request`（L1340）把 `status` 设成 `PREEMPTED` 且 `num_computed_tokens = 0`——**回到上表的「入队」那一行**，重走 step 1。看着很浪费，但块很可能还在缓存表里，实际不真的重算（见第三节第 3 步）。

### 2.4 数据并行模式下的主循环

```
    ✅ core.py 里的真实继承链（grep "^class " 出来的）：

    EngineCore                                      L104   一步的本体（step()）
    └─ EngineCoreProc(EngineCore)                   L1007  ★ 加上主循环，跑在独立进程里
       ├─ DPEngineCoreProc(EngineCoreProc)          L1986  ★★ 数据并行专用
       │  │                                                （它重写了 run_busy_loop，L2170）
       │  └─ DPMoEEngineCoreActor(Mixin, DPEngine…) L2519  ★ MoE + DP 的组合
       └─ EngineCoreActor(Mixin, EngineCoreProc)    L2542  Ray actor 形态
    EngineCoreActorMixin                            L2387  给上面两个 Actor 提供的共用部分
```

> 📖 **DP**（data parallelism，数据并行）——同一份完整模型复制多份，每份处理不同的请求。
> 📖 **MoE**（Mixture of Experts，混合专家）——每个 token 只激活一小部分「专家」子网络的模型结构，见 [11 章](推理优化入门/11-MoE推理与专家并行.md)。

> ★★ `DPEngineCoreProc` 存在的理由，源码注释自己写在了 [parallel_state.py:1821](源码/vLLM/vllm/distributed/parallel_state.py)：*"all the ranks in the same DP group should generate simultaneously, i.e. the `generate` call in the same DP group should be called together, otherwise it will cause deadlock"*。**DP 组必须一起 generate，否则死锁**——所以需要一个专门的主循环来保证所有 DP rank 步调一致。这条约束在 [06 章](推理优化入门/06-推理中的并行策略.md) 讲过，这里是它的实现。

---

## 三、★★ 调度器：一次 `schedule()` 到底做了什么

这是 vLLM 最值得读的一个函数。文件：[v1/core/sched/scheduler.py](源码/vLLM/vllm/v1/core/sched/scheduler.py)（`class Scheduler` 在 L69，`def schedule` 在 **L477**）。

**先看它一步之内做完的事**：

| 顺序 | 位置 | 做什么 | 结束时改了什么 |
|---|---|---|---|
| 1 | L497-500 | 开三个预算 | 局部变量 `token_budget` / `input_budget` / `draft_slots` |
| 2 | L559-575 | 扫 **running** 队列（老用户优先） | 每个在跑的请求分到 `num_new_tokens`，扣预算 |
| 3 | L1340 | 装不下就**抢占**（从 running 尾部踢） | `status = PREEMPTED`，`num_computed_tokens = 0`，块全放 |
| 4 | L444 / L1034 | 扫 **waiting** 队列（放新人进来） | 先查前缀缓存，再 `allocate_slots` 拿块，`status = RUNNING` |
| 5 | L1170-1180 | 一组断言守门 | 什么都不改——**只证明上面所有 `min()` 自洽** |

产出一个 `SchedulerOutput`（[sched/output.py](源码/vLLM/vllm/v1/core/sched/output.py)），交给 `GPUModelRunner`。

### 第 1 步：开预算

```python
# ✅ L497-500
token_budget = self.max_num_scheduled_tokens
spec = self.vllm_config.speculative_config
draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
input_budget = self.scheduler_config.max_num_batched_tokens
```

```
    ★ 三个预算：
      token_budget  —— 这一步总共能算多少 token（chunked prefill 的粒度）
      input_budget  —— 输入张量的容量上限
      ★ draft_slots —— 给投机解码的草稿 token 预留的位置
```

> ⚠️ **口径提醒：这三个预算全是「算力侧」的闸门**，管的是一步塞多少 token 进前向。**显存侧的闸门是另一个**——KV 块够不够，由第 4 步的 `allocate_slots` 说话。两者经常一起把并发压下来，但**原因不同、调的参数也不同**，说「利用率低」时必须先说清是哪一侧受限（算力受限还是带宽/显存受限，见 [大模型算子开发/09-推理侧算子.md](../大模型算子开发/09-推理侧算子.md)）。

### 第 2 步：先排 running 队列（老用户优先，保证吐字不断）

```python
# ✅ L559-575（每个 min 都是跨多行的写法）
num_new_tokens = (request.num_tokens_with_spec
                  + request.num_output_placeholders
                  - request.num_computed_tokens)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold   # ★ 单条封顶
num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)  # ★ 总预算夹紧
num_new_tokens = min(num_new_tokens, self.max_model_len - request.num_computed_tokens)  # ★ 位置不越界
```

```
    ★★ 这几行就是 chunked prefill 的全部核心：一次计算 + 三道夹紧
       ① 算出「这个请求还欠多少 token 没算」
       ② ★ 单条超长 prompt 先自己封顶（long_prefill_token_threshold）
          ⟹ 防止一条 32K 的 prompt 独占整步
       ③ ★ 再被总预算夹一次，并减掉 draft_slots ⟹ 给投机解码留位置
       ④ 最后夹一次 max_model_len ⟹ 投机解码时位置可能越界，必须防
```

```python
# ✅ L553-557 —— 同一个循环里的一处 continue，值得单独看
if defer_prefills and request.is_prefill_chunk:
    # DP prefill balancing: defer this in-progress prefill chunk to a
    # cadence-aligned step; decodes still run to fill this step.
    req_index += 1
    continue
```

```
    ★ defer_prefills 打开时，正在做 prefill 的 chunk 会被【跳过】，
      让这一步只跑 decode。
    ⟹ 这是数据并行下的 prefill 均衡：把 prefill 攒到对齐的步上一起做，
       decode 照跑不误 ⟹ ★ 各 DP rank 的步型更一致
    ⟹ ★★ 注意它和 SGLang 的「Prefill 优先」正好是一对镜像：
       vLLM 默认混着跑、必要时【推迟 prefill】；
       SGLang 默认分开跑、必要时【推迟 prefill】（prefill_delayer.py）。
       两边最后都在同一个地方拧同一颗螺丝。
```

### 第 3 步：装不下就抢占

```python
# ✅ L1340-1360（节选）
def _preempt_request(self, request, timestamp, drop_stale_output=False):
    assert request.status == RequestStatus.RUNNING
    self._free_request_blocks(request)       # ★ 释放全部 KV 块
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0          # ★★ 归零 = 走【重算】路线
```

```
    ★ 从 running 队列【尾部】选（最晚进来的先踢）⟹ 保住 FCFS 语义
    ★★ num_computed_tokens = 0 说明 v1 主推【重算】而非【换出到 CPU】
       原因：有前缀缓存兜底，重算时块很可能还在，不真的重算
```

> 📖 **FCFS**（first come first serve，先来先服务）——最朴素的排队规则，先到的先服务。

### 第 4 步：再排 waiting 队列（放新人进来）

新请求要先问「有多少前缀已经算过了」，再申请剩下的块：

```python
# ✅ L444-454 —— 查前缀缓存的那个辅助函数
def _get_local_prefix_cache_hit(self, request):
    connector = self.connector
    if connector is not None and connector.supports_divergent_local_hybrid_hits:
        return self.kv_cache_manager.get_computed_blocks_for_connector(request)
    blocks, num_local, shared_prefix_boundary = (
        self.kv_cache_manager.get_computed_blocks(request))    # ★ 查前缀缓存
    return blocks, num_local, shared_prefix_boundary, False

# ✅ L1034 —— 拿到命中长度之后，只申请剩下的部分
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks, ...)            # ★ 申请剩下的块
if new_blocks is None:
    break                                                    # ★★ 显存不够 ⟹ 整个 waiting 循环停住
```

```
    ★★ 注意 new_blocks is None 时是 break 而不是 continue：
       一旦显存装不下当前这个请求，【后面的都不再试】
    ⟹ 保住 FCFS：不允许「小请求插队钻进大请求的空隙」
```

`get_computed_blocks` / `allocate_slots` 的定义在 [v1/core/kv_cache_manager.py](源码/vLLM/vllm/v1/core/kv_cache_manager.py)（**L232** / **L347**）。

### 第 5 步：断言守门

```python
# ✅ L1171-1176
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert token_budget >= 0
assert input_budget >= 0
assert len(self.running) <= self.max_num_running_reqs
```

> ★ 这组断言是「预算制」的最后一道保险。读代码时看到它，就知道**上面所有分支的 `min()` 加起来必须自洽**。**这几行比任何注释都能说明这个函数的契约是什么。**

---

## 四、★ KV Cache 管理：块池 + 链式哈希

> 概念（为什么要分页、前缀缓存怎么判命中）在 [05 章](推理优化入门/05-PagedAttention与前缀缓存.md)。这里只看它在源码里长什么样。

### 块池

文件：[v1/core/block_pool.py](源码/vLLM/vllm/v1/core/block_pool.py)

| 符号 | 行 | 是什么 |
|---|---|---|
| `class BlockHashToBlockMap` | L33 | ★ 哈希 → 块，前缀缓存的查找表 |
| `class BlockPool` | L143 | ★ 所有 KV 块的池子 |
| `def cache_full_blocks` | L225 | ★★ 只有**写满**的块才进缓存 |
| `def cache_partial_block` | L445 | ★ 半满的块走另一条路 |

> 🔑 **为什么要区分满块和半块？** 半满的块内容还会变（下一个 token 就写进来了），**它的哈希此刻算出来，下一步就失效了。** 所以只有写满、内容冻结的块才能进缓存表。这就是第 2.3 节那张表里「step 2 之后第 3 块不进表」的原因。

### 链式哈希

文件：[v1/core/kv_cache_utils.py](源码/vLLM/vllm/v1/core/kv_cache_utils.py)

| 符号 | 行 | 是什么 |
|---|---|---|
| `NONE_HASH` | L102 | ★ 哈希链的起点 |
| `DEFAULT_NONE_HASH_SEED` | L106 | 种子字符串 `"vllm-none-hash"` |
| `generate_block_hash_extra_keys` | L580 | ★ 额外键：多模态输入、LoRA id 等 |
| `hash_block_tokens` | L618 | ★★ 块哈希函数本体 |

> 📖 **LoRA**（Low-Rank Adaptation，低秩适配）——给基座模型挂一组小的增量权重做微调。同一个基座上挂不同 LoRA 的请求，KV 不能互相复用，所以 LoRA id 必须进哈希的额外键。

```
    ★★ 哈希是【链式】的：
       block_hash[i] = hash( block_hash[i-1], 这 16 个 token, extra_keys )
                              ↑ 上一块的哈希

    ⟹ 只有【整条前缀路径完全相同】才会命中，
       中间任何一个 token 不同，后面全部不命中。★ 这是正确性的保证。

    ⚠️ 也是一个常见坑的成因：prompt 开头放时间戳
       ⟹ 第 1 块哈希就不同 ⟹ ★ 整条链全废。
```

### 上层管理

```
    v1/core/kv_cache_manager.py            对外接口（get_computed_blocks L232 / allocate_slots L347）
    v1/core/kv_cache_coordinator.py        ★ 协调多种 KV 类型
    v1/core/single_type_kv_cache_manager.py
```

> ★ 为什么需要「多种 KV 类型」？因为现在的模型不再是清一色的注意力了——Mamba/线性注意力的状态、滑动窗口注意力的有限窗口、全注意力，**三者的缓存生命周期完全不同**，需要分开管理。
> ✅ **印证**：`v1/attention/backends/` 里确实有 `mamba1_attn.py` / `mamba2_attn.py` / `mamba_attn.py` / `linear_attn.py` / `short_conv_attn.py`。

> 📖 **SWA**（sliding window attention，滑动窗口注意力）——每个 token 只看最近 W 个 token，超出窗口的 KV 可以丢掉。**它会丢弃旧 KV，所以缓存语义和全注意力完全不同。**

---

## 五、★★ 并行状态：06 章的源头

文件：[distributed/parallel_state.py](源码/vLLM/vllm/distributed/parallel_state.py)，函数 `initialize_model_parallel`（**L1751**）。

> 📖 **TP**（tensor parallelism，张量并行）——把单个矩阵按行或列切到多张卡上。
> 📖 **PP**（pipeline parallelism，流水线并行）——把模型按层切段，一段一张卡。
> 📖 **EP**（expert parallelism，专家并行）——MoE 的专家分散到不同卡上。
> 📖 **PCP / DCP**（prefill / decode context parallelism，Prefill / Decode 上下文并行）——把**一条序列的长度**切到多张卡上；vLLM 把 Prefill 和 Decode 两段分开成两个维度。
> 📖 **ExternalDP**（external data parallelism，外部数据并行）——不属于模型本身的 DP 层，各 rank 可以完全独立 generate（源码注释举的例子是 verl 训练框架的集成）。

**核心就是一次 reshape 加若干次 transpose**：

```python
# ✅ L1817 —— 全文最重要的一行注释
# the layout order is: ExternalDP x DP x PP x PCP x TP

# ✅ L1826
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size)
```

代码注释还直接给了取组的通用手法（L1824-1825）：

> *"to get group_ranks for each dimension, transpose that dimension to the last dimension, then reshape to 2D, then unbind the last dimension"*

**于是各组的构造就是一行一个**：

| 组 | 表达式所在行 | 手法 | 备注 |
|---|---|---|---|
| TP | L1837 | `all_ranks.view(-1, tp).unbind(0)` | ★ 已在最后一维，**不用转置** |
| DCP | L1858-1859 | `transpose(-1, -2)` 再按 `dcp` 切 | ★ 只在 `dcp_size > 1` 时建组 |
| PCP | L1872 | `transpose(3, 4)` | |
| PP | L1891-1892 | `transpose(2, 4)` | |
| DP | L1908 | `transpose(1, 4)` | |
| EP | L1927-1934 | `transpose(1, 2).reshape(-1, dp × pcp × tp)` | ★★ **只有 MoE 模型才建这个组**（L1926 的 `is_moe` 判断） |

```
    ★ EP 组 = 同一个 PP 段内的所有卡。
      transpose(1, 2) 把 DP 和 PP 换位 ⟹ 布局变成 ExtDP × PP × DP × PCP × TP
      再 reshape(-1, dp×pcp×tp) ⟹ 每组正好是「一个 PP 段里的全部卡」
```

**文档字符串里的例子**（8 卡、TP=2、PP=4，✅ L1768-1775 原文）：

```
    TP 组：[g0,g1] [g2,g3] [g4,g5] [g6,g7]     ★ 相邻，走 NVLink
    PP 组：[g0,g2,g4,g6] [g1,g3,g5,g7]         步长 2
```

> ★★ [06 章](推理优化入门/06-推理中的并行策略.md) 的全部结论都从这几行推出来。**如果只读一个文件，读这个。**

### 并行线性层

文件：[model_executor/layers/linear.py](源码/vLLM/vllm/model_executor/layers/linear.py)

| 类 | 行 | 是什么 |
|---|---|---|
| `ColumnParallelLinear` | L407 | ★ 列切；`gather_output` **默认 `False`**（L443） |
| `DCPGroupColumnParallelLinear` | L604 | ★ 给 DCP 用的变体 |
| `MergedColumnParallelLinear` | L645 | 多个列切层合并（MLP 的 gate + up） |
| `QKVParallelLinear` | L971 | ★ QKV 三个矩阵合成一个列切层 |
| `RowParallelLinear` | L1510 | ★★ 行切；**唯一的通信在 L1660** |

```python
# ✅ L1659-1662 —— 推理 TP 的全部通信，就这一句
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
else:
    output = output_parallel
```

```
    ★★ 请注意三件事：
    ① 整个文件里【没有任何反向传播的通信】
       ⟹ 一层里有两个 RowParallelLinear（注意力的 o_proj、MLP 的 down_proj）
          ⟹ ★ 推理每层 2 次 all-reduce，且【只有前向这 2 次】
       ⟹ 这就是 01 章「推理 TP 每层 2 次，训练 4 次」的源码证据
          （训练那 2 次多出来的在反向，这个文件里根本不存在）
    ② ColumnParallelLinear 的 gather_output 默认 False（L443）
       ⟹ ★ 列切的输出【故意不聚合】，直接喂给下一个行切层
       ⟹ 一个「列切 + 行切」的组合，全程只需要【最后那一次】all-reduce
    ③ ★ 连那一次都能关：reduce_results=False 时直接返回本地分片
       ⟹ 调用方自己决定什么时候聚合（融合层、DP attention 都用这个口子）
```

---

## 六、其余模块的地图

挂到第二节那条主线上：**投机解码改的是 `num_tokens_with_spec`（第 2 步的输入），KV 传输改的是 `get_computed_blocks` 的命中来源（第 4 步），EPLB 和量化都在 `execute_model` 内部。**

### 投机解码 `v1/spec_decode/`

```
    eagle.py              ★ EAGLE
    medusa.py             Medusa 多头
    ngram_proposer.py     ★ n-gram 查表，零成本
    ngram_proposer_gpu.py   GPU 版
    suffix_decoding.py    ★ 后缀解码
    draft_model.py        独立小模型
    llm_base_proposer.py  通用的「拿一个 LLM 当草稿模型」基类
    dynamic/              ★★ 自适应——按负载动态调整
    metrics.py            接受率统计
```

> ★ `dynamic/` 的存在印证了 [08 章](推理优化入门/08-投机解码.md) 那条规律：**投机解码在大 batch 下会变亏，必须按负载动态开关。**

### KV 传输 `distributed/kv_transfer/kv_connector/v1/`

```
    base.py                  ★ 统一接口
    nixl/                    ★ NVIDIA 的 RDMA 传输库（主流）
    mooncake/                ★ 月之暗面的 KV 池
    lmcache_connector.py     LMCache（还有 lmcache_mp_connector.py / lmcache_integration/）
    moriio/  hf3fs/  flexkv_connector.py     其他后端
    offloading/  simple_cpu_offload_connector.py   换出到 CPU/SSD
    ★ multi_connector.py     ★★ 串联多个后端：先查本地，再查远端
```

> 📖 **RDMA**（Remote Direct Memory Access，远程直接内存访问）——绕过 CPU 和内核，网卡直接读写对端显存/内存，是 PD 分离传 KV 的主力手段，见 [10 章](推理优化入门/10-PD分离.md)。

### EPLB `distributed/eplb/`

> 📖 **EPLB**（Expert Parallelism Load Balancer，专家并行负载均衡器）——MoE 里热门专家会把某几张卡压满，EPLB 负责把专家搬一搬让负载均匀。

```
    eplb_state.py            当前专家分布
    eplb_communicator.py     搬运用的通信
    policy/                  重排算法
    rebalance_execute.py     ★ 执行搬运
    ★ async_worker.py        ★★ 异步做，不阻塞推理 ← 能上线的关键
```

### 量化 `model_executor/layers/quantization/`

```
    fp8.py / input_quant_fp8.py / fbgemm_fp8.py   ★ FP8 一族（H100 主战场）
    auto_awq.py / auto_gptq.py                    ★ 权重量化
    ★ kv_cache.py                                 ★★ KV Cache 量化
    mxfp4.py                                      Blackwell
    moe_wna16.py / experts_int8.py                MoE 专用
    compressed_tensors/ / quark/ / inc/           厂商与外部工具链
    torchao.py / modelopt.py                      外部工具链
```

详见 [09 章 量化](推理优化入门/09-量化.md)。

### CUDA Graph `config/compilation.py:53`

```python
# ✅ L53-63
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)         # ★ 元组 = 两种批次两种模式
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # ★★ chunked prefill 下的常用值
```

```
    ★ 元组的两位分别是「纯 decode 步用什么模式」和「混合步用什么模式」：
      decode_mode()  取 value[0]      # ✅ L65-66
      mixed_mode()   取 value[1]      # ✅ L68-69
    ⟹ FULL_DECODE_ONLY = 纯 decode 走全图，混合步不走图
    ⟹ 直接对应第 2.3 节那张表：step 3/4 能走图，step 1/2 不能
```

详见 [12 章 CUDA Graph 与图编译](推理优化入门/12-CUDA-Graph与图编译.md)。

---

## 七、★ 读源码的建议路线

```
    第 1 天：v1/engine/core.py 的 step()（L583-613，只有 30 行）
            ⟹ 先把第二节那条主线在真代码里走一遍。这是骨架。
            ★ 顺手翻一下 v1/request.py:359 的 RequestStatus

    第 2 天：v1/core/sched/scheduler.py 的 schedule()
            ⟹ 从 L477 一路读到 L1180，看预算怎么被消耗
            ★ 重点看每一处 min() / continue / break

    第 3 天：distributed/parallel_state.py 的 L1751-1960
            ⟹ 把 rank 布局彻底搞清楚（对照第五节那张表）

    第 4 天：v1/core/block_pool.py + kv_cache_utils.py
            ⟹ 搞清楚一个块从申请到被缓存到被淘汰的完整生命

    第 5 天：model_executor/layers/linear.py
            ⟹ 验证「每层只有 1 次 all-reduce」

    第 6 天：v1/worker/gpu_model_runner.py（★ 这个文件很大，4000+ 行）
            ⟹ 只看 execute_model()（L4288），看 SchedulerOutput
               怎么变成张量
```

> ⚠️ **不要一开始就读 `models/`**。那里是各个模型的具体定义，长得都差不多，读了对理解系统没帮助。

---

## 八、和 SGLang 的关键差异（速览）

| | vLLM |
|---|---|
| 进程结构 | ★ 异步前端进程 + 引擎进程，中间进程间队列 |
| 一次 step | ★★ **decode 和 prefill 混在同一个批次里**（✅ `enable_chunked_prefill` 默认 `True`，[config/scheduler.py:74](源码/vLLM/vllm/config/scheduler.py)） |
| 并行网格 | ★ 固定五维 reshape：`ExtDP × DP × PP × PCP × TP` |
| 前缀缓存 | ★ 链式哈希 + 哈希表 |
| 调度依据 | 显存够不够 + token 预算 |
| 抢占 | ★ 重算（`num_computed_tokens = 0`） |
| 设计取向 | ★ 并行策略是**部署配置**，先定网格，模型往里填 |

详细对比见 [推理框架对比.md](推理框架对比.md)。

---

> 返回 [推理框架总目录](README.md) ｜ 下一篇 [SGLang拆解.md](SGLang拆解.md)
