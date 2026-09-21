# SGLang 源码拆解

> **一句话**：SGLang 把「缓存」当一等公民——**一棵基数树装下所有请求的 KV 前缀**，调度器在放请求进来之前就能问出「这个请求能省多少 Prefill」；并行上它只固定 `tp × pp`，剩下的维度全从 TP 组**内部**再切。

> **版本**：commit `04444ee`（2026-08-20 拉取，本地副本在 [源码/SGLang/](源码/SGLang/)），5,674 个 `.py` 文件 / 1,836,778 行。代码主体在 `python/sglang/srt/`（**srt = SGLang RunTime**）。

> ⚠️ **行号是我在这个 commit 上实际 `grep` 出来的。SGLang 迭代比 vLLM 更快，`scheduler.py` 单文件就有四千多行，换个版本行号必然偏移。** 每处引用都同时写了**文件名 + 符号名**；行号对不上时用符号名搜。

> **读法建议**：先读完 [vLLM拆解.md](vLLM拆解.md) 再读这篇——SGLang 的很多设计只有和 vLLM 对照才看得出用意。概念侧（基数树 vs 哈希表、缓存感知调度）在 [05 章](推理优化入门/05-PagedAttention与前缀缓存.md) 和 [04 章](推理优化入门/04-连续批处理与调度.md)，**这篇只回答：那些机制在源码里长什么样。**

---

## 一、目录长什么样

```
    srt/
    ├── managers/        ★★ 调度器、批次、策略  ← 相当于 vLLM 的 v1/core/sched
    │   └── scheduler_components/   ★ 调度器被拆出来的 20 个 .py
    ├── mem_cache/       ★★ 内存池 + 各种前缀缓存  ← 本文重点，40 个 .py 文件
    ├── distributed/     ★★ 并行状态  ← 第五节的源头
    ├── layers/          ★ 各种层，含 dp_attention.py
    ├── speculative/     ★ 投机解码（34 个 .py + 2 个子目录）
    ├── disaggregation/  ★ PD 分离
    ├── eplb/            ★ 专家负载均衡（★ 含离线模拟器）
    ├── elastic_ep/      ★ 弹性专家并行
    ├── model_executor/  执行器
    ├── compilation/     图编译
    └── entrypoints/     HTTP 服务
```

> ★ **第一印象就能看出取向**：`mem_cache/` 有 **40 个 `.py` 文件**，而 vLLM 对应的 `v1/core/` 只有 **8 个 `.py` 加一个 `sched/` 子目录（7 个文件）**。**SGLang 把「缓存」当成了一等公民**，这和它的论文主张（RadixAttention）完全一致。

> 📖 上面出现的缩写：**PD 分离**（Prefill / Decode disaggregation）——把 Prefill 和 Decode 两段放到不同机器上分别部署，见 [10 章](推理优化入门/10-PD分离.md)；**EPLB**（Expert Parallelism Load Balancer，专家并行负载均衡器）——见第八节；**EP**（expert parallelism，专家并行）——MoE 的专家分散到不同卡上。**不认识的缩写都可以先去 [附-速查表](推理优化入门/附-速查表.md) 查。**

---

## 二、★★ 主线：一个请求走完全程

和 vLLM 的两进程结构不同，**SGLang 是三类进程 + ZMQ 消息**。先把这条主线走完，后面每一节才挂得上去。

> 📖 **ZMQ**（ZeroMQ）——一个轻量消息库。SGLang 用它在前端、调度器、反分词三类进程之间传结构化消息，而不是共享内存队列。

### 2.1 调用链：三个进程，一圈消息

```
    HTTP POST /generate  或  /v1/chat/completions
      │   entrypoints/http_server.py
      ▼
  【进程 1：TokenizerManager（异步前端，一个）】
  ① TokenizerManager.generate_request()      managers/tokenizer_manager.py:765
      ├─ _tokenize_one_request()             L995    分词 → TokenizedGenerateReqInput
      ├─ _send_one_request()                 L1597   ★ 经 ZMQ 发给调度器进程
      └─ _wait_one_response()                L1733   ★ 挂在 asyncio 上等，边等边 yield
      │
      ▼  ZMQ
  【进程 2：Scheduler（每个 TP rank 一个，同步代码）】
  ② Scheduler.event_loop_normal()            managers/scheduler.py:1743   ★★ 主循环
      ├─ request_receiver.recv_requests()             L1750
      ├─ process_input_requests()                     L1901
      │     └─ handle_generate_request()              L2396   建 Req 对象
      │           └─ _add_request_to_queue()          L2767   ★ 进 waiting_queue（L2774）
      ├─ get_next_batch_to_run()                      L3064   ★★ 决定这一步跑谁
      │     ├─ get_new_batch_prefill()                L3209   Prefill 候选
      │     │     ├─ SchedulePolicy.calc_priority()   L3285   ★★ 排序（含 LPM）
      │     │     └─ PrefillAdder                     L3309   ★ 逐个往批次里加
      │     └─ update_running_batch()                 L3550   Decode 批次
      ├─ run_batch()                                  L3695   → 前向 + 采样
      └─ process_batch_result()                       L3991   写回 Req、判结束
      │                                                        └─ 结束时 release_kv_cache()
      │                                                           mem_cache/common.py:198
      │                                                           → RadixCache.cache_finished_req()
      ▼  ZMQ
  【进程 3：DetokenizerManager（一个）】
  ③ DetokenizerManager.event_loop()          managers/detokenizer_manager.py:167
      └─ handle_batch_token_id_out()         L431    ★ 增量反分词：token id → 文本片段
      ▼  ZMQ
  【回到进程 1】
  ④ TokenizerManager._wait_one_response() 收到片段 → HTTP SSE 逐块吐字
```

> 📖 **SSE**（Server-Sent Events，服务器推送事件）——HTTP 上的单向流式协议。

```
    ★★ 三进程切分的两个后果，值得单独记住：

    ① 调度器是【纯同步】代码，一个 while True 跑到底（L1745）。
       ⟹ 读它不用考虑 await 把循环切走，控制流是线性的
       ⟹ 代价：调度器进程里任何一次慢操作都直接顶住整步

    ② 反分词【单独一个进程】。
       ⟹ tokenizer 的 CPU 开销（尤其是中文和多字节 token 的增量解码）
          不占调度器进程的时间
       ⟹ 这是 vLLM 没有的一刀：vLLM 的 detokenize 在前端进程里做
          （OutputProcessor 内部）
```

### 2.2 ★★ 一次循环里发生什么：Prefill 优先

`event_loop_normal`（[scheduler.py:1743](源码/SGLang/python/sglang/srt/managers/scheduler.py)）的循环体只有五步：

```python
# ✅ L1745-1773（节选）
while True:
    recv_reqs = self.request_receiver.recv_requests()      # ① 收请求
    self.process_input_requests(recv_reqs)                 # ② 入队
    plan = self.get_next_batch_to_run(                     # ③ 组批 ★★
        running_batch=self.running_batch, last_batch=self.last_batch)
    self.running_batch = plan.running_batch
    batch = plan.batch_to_run
    if batch:
        result = self.run_batch(batch)                     # ④ 前向
        self.process_batch_result(batch, result)           # ⑤ 收结果
    self.last_batch = batch
```

**第 ③ 步是全文最关键的分叉**，代码自己写了注释：

```python
# ✅ L3175-3184
if new_batch is not None:
    # Run prefill first if possible
    ret = new_batch
else:
    # Run decode (skip for prefill-only batches)
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        running_batch = self.update_running_batch(running_batch)
        ret = running_batch if not running_batch.is_empty() else None
    else:
        ret = None
```

```
    ★★★ 读懂这 10 行 = 读懂 SGLang 和 vLLM 最大的行为差异：

    SGLang：一步要么【全是 Prefill】，要么【全是 Decode】。
            有 Prefill 可做就先做 Prefill，没有才跑 Decode。

    vLLM：  一步里 Decode 和 Prefill chunk【混在同一个批次】，
            共用一个 token 预算（见 vLLM 拆解的第三节第 2 步）。
```

> ★ SGLang 也能混，但**要显式打开**：`enable_mixed_chunk`（源码里在 [scheduler.py:1182-1184](源码/SGLang/python/sglang/srt/managers/scheduler.py) 算出 `self.is_mixed_chunk`）。**默认是分开的。**

⚠️ **下面这张时间线是我按上面的代码结构画的示意，不是实测 trace**。设定：一个长请求正在 Decode，第 3 步来了两个新请求。

```
  step 1   [ Decode  ]  running_batch 里 1 个请求，吐 1 个字
  step 2   [ Decode  ]  吐 1 个字
           ← 两个新请求到达，进 waiting_queue
  step 3   [ Prefill ]  ★★ get_new_batch_prefill 有货 ⟹ 这一步【不 Decode】
                        ⟹ 老请求这一步【一个字都没吐】←★ TBT 尖峰来源
  step 4   [ Decode  ]  last_batch 是 extend ⟹ 先 merge 进 running_batch（L3136-3141）
                        ⟹ 现在 3 个请求一起 Decode
  step 5   [ Decode  ]  ...
```

```
    ★ 于是 SGLang 有一组 vLLM 没有的旋钮，专门治这个尖峰：
      managers/prefill_delayer.py          ★ 故意延迟 Prefill（攒批）
      managers/min_free_slots_delayer.py   ★ 保底空闲槽位，防抢占颠簸
    ⟹ 两个都是主动「不做事」的组件。
    ⟹ ★★ vLLM 用 chunked prefill 把长 Prefill 切碎混进去，
       SGLang 用「延迟 + 分步」在时间上错开。两条路治的是同一个病。
```

> 📖 **TBT**（Time Between Tokens，token 间隔）——相邻两个输出 token 之间的时间。一步只跑 Prefill 会让所有在 Decode 的请求这一步不出字，表现为 TBT 尖峰。指标口径见 [13 章](推理优化入门/13-性能指标与压测.md)。

### 2.3 状态怎么变：两个共享前缀的请求

⚠️ **下面这张表的数字是我为讲解构造的**，用来看清树、引用计数、`waiting_queue` 三者怎么联动。设定：系统提示词 30 个 token，请求 A 和 B 都带它，A 先到 10 ms。

| # | 发生了什么 | 基数树 | A 的状态 | B 的状态 |
|---|---|---|---|---|
| 1 | A 到达，`_add_request_to_queue`（L2767） | 空树 | 在 `waiting_queue` | — |
| 2 | `calc_priority` → `match_prefix`（radix_cache.py:377） | 空树，匹配长度 **0** | `num_matched_prefix_tokens = 0` | — |
| 3 | `PrefillAdder` 收下 A，`inc_lock_ref`（L623）锁住 A 用到的路径 | — | Prefill 全部 30 token + 提示词后的正文 | — |
| 4 | A 进 Decode。`maybe_cache_unfinished_req`（mem_cache/common.py:107） | ★ A 已算的前缀**插进树**，边上挂着 KV | Decode 中，锁仍持有 | — |
| 5 | B 到达，`match_prefix` | ★★ 匹配到 **30**（整段提示词） | — | `num_matched_prefix_tokens = 30` |
| 6 | `_sort_by_longest_prefix`（L381）把 B 排前面 | — | — | ★ 排到队首 |
| 7 | `PrefillAdder` 收 B：**只 Prefill 提示词之后的部分** | ★ `_split_node`（L705）在分叉点劈边 | — | ★★ 省掉 30 token 的 Prefill |
| 8 | A 结束：`release_kv_cache` → `cache_finished_req`（L459） | ★ A 的 KV **交还给树**，`dec_lock_ref` 解锁 | `FINISHED` | Decode 中 |
| 9 | 显存紧张，`evict`（L593）跑 LRU | ★ **只淘汰 `evictable_size()`（L659）统计到的部分**——B 锁住的路径不动 | — | 安全 |

```
    ★★ 这张表里的因果链，是 SGLang 全部设计的浓缩：

    树能一次问出「匹配多长」（第 5 步）
      ⟹ 调度器才能按匹配长度排序（第 6 步）
      ⟹ 才能「先放最省事的那批进来」（第 7 步）

    而共享一旦发生，淘汰就必须知道谁在用（第 9 步）
      ⟹ 所以必须有引用计数锁（第 3 / 8 步）
    ⟹ ★ 树和锁是一套东西，不能只抄一半。
```

---

## 三、★★ RadixAttention：基数树前缀缓存

这是 SGLang 的招牌，也是它和 vLLM 最本质的分歧点。

> 📖 **基数树**（radix tree，也叫压缩前缀树 / Patricia trie）——把公共前缀合并成一条边的树。原理和图示在 [05 章](推理优化入门/05-PagedAttention与前缀缓存.md)。

文件：[mem_cache/radix_cache.py](源码/SGLang/python/sglang/srt/mem_cache/radix_cache.py)（**共 863 行**）

| 符号 | 行 | 是什么 |
|---|---|---|
| `class RadixKey` | L59 | ★ 树的键（token 序列 + 额外标识） |
| `class TreeNode` | L238 | ★ 树节点 |
| `TreeNode.evicted` | L269 | 该节点的 KV 是否已被淘汰 |
| `class RadixCache` | L303 | ★★ 主类 |
| `def match_prefix` | L377 | ★★ 最长前缀匹配 ← **核心操作** |
| `def insert` | L437 | 插入 |
| `def cache_finished_req` | L459 | ★ 请求结束时把 KV 交还给树 |
| `def cache_unfinished_req` | L516 | ★ 请求进行中也能先缓存 |
| `def evict` | L593 | ★ LRU 淘汰 |
| `def inc_lock_ref` | L623 | ★★ 引用计数锁：在用的块不许淘汰 |
| `def dec_lock_ref` | L638 | 解锁 |
| `def evictable_size` | L659 | ★ 可淘汰的量（调度器要看这个） |
| `def protected_size` | L662 | ★ 被锁住、动不了的量 |
| `def _split_node` | L705 | ★★ 边分裂 ← 基数树的标志性操作 |

```
    ★ evictable_size (L659) + protected_size (L662) 这一对，
      和 total_size (L590) 一起构成调度器的显存视图：
      总量 = 能淘汰的 + 被锁住的
    ⟹ ★★ 调度器判断「还能不能放人进来」，看的是【能淘汰的】那一半。
```

### ★ 为什么是树，而不是哈希表

```
    vLLM：块 → 哈希 → 查表
      ⟹ 要知道命中多长，得【逐块】算哈希去试
      ⟹ ★ 而且只能回答「命中/不命中」，很难提前估算「能命中多少」

    ★★ SGLang：整个缓存是一棵树，公共前缀天然合并成一条边
      ⟹ match_prefix() 一次遍历，直接返回【最长匹配长度】
      ⟹ ★★★ 于是调度器可以在【放请求进来之前】就问：
             「这个请求能省多少 Prefill？」
```

**`_split_node`（L705）是理解基数树的关键**：

```
    树里已有一条边： "你是一个助手。请介绍"
    新请求来了：     "你是一个助手。今天"

    ⟹ 公共前缀是 "你是一个助手。"
    ⟹ ★ 必须把原来那条边【劈成两段】：
         "你是一个助手。" → 分叉 → "请介绍" / "今天"
    ⟹ 这就是 _split_node 干的事。
```

### ★ 引用计数锁（L623）

```
    ⚠️ 问题：某个请求正在用节点 X 的 KV，
       此时显存不够，淘汰算法想把 X 删掉 ⟹ ★ 正在跑的请求崩溃

    ★ 解法：inc_lock_ref / dec_lock_ref
       请求开始用 → 沿着树【从该节点一路往上到根】全部加锁
       请求结束   → 全部解锁
    ⟹ evictable_size()（L659）只统计【没被锁住】的部分
    ⟹ ★★ 调度器看的是 evictable_size，而不是「缓存总大小」
```

### ★ 缓存家族：一棵树不够用

`mem_cache/` 里的树有一整个家族，每个对应一种模型结构：

| 文件 | 管什么 |
|---|---|
| `radix_cache.py` | ★ 标准版 |
| `radix_cache_cpp.py` | ★ C++ 实现（降低树操作的 CPU 开销） |
| `hiradix_cache.py` | ★★ 分层（hierarchical）：GPU → CPU → SSD 三级 |
| `swa_radix_cache.py` | 滑动窗口注意力专用 |
| `pure_swa_radix_cache.py` | 纯滑动窗口 |
| `mamba_radix_cache.py` | ★ Mamba 状态专用 |
| `unified_radix_cache.py` | ★ 混合结构模型（一部分注意力 + 一部分 Mamba） |
| `chunk_cache.py` | **不用树**的简单版（对照组） |

> ⚠️ 上面是 **7 棵树 + 1 个非树对照组**。另外 `mem_cache/storage/` 下还有接外部存储的子类（`flexkv_radix_cache.py` / `lmc_radix_cache.py`），它们是在 `RadixCache` 上再包一层。

> 📖 **SWA**（sliding window attention，滑动窗口注意力）——每个 token 只看最近 W 个 token，超出窗口的旧 KV 可以丢掉。

> 🔑 **为什么要这么多种？** [05 章](推理优化入门/05-PagedAttention与前缀缓存.md) 讲的树，隐含假设是「KV 只增不减、可以任意复用」。**滑动窗口注意力会丢弃旧 KV，Mamba 根本没有 KV 只有状态**——这两种情况下树的语义完全不同，必须分开实现。★ 这也解释了为什么 `mem_cache/` 有 40 个文件：**现代模型结构已经不统一了。**

---

## 四、★★ 缓存感知调度：树结构的真正回报

文件：[managers/schedule_policy.py](源码/SGLang/python/sglang/srt/managers/schedule_policy.py)

**源码把调度策略明确分成了两类**，这个划分本身就是论点：

```python
# ✅ L200-213
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""
    LPM = "lpm"                # ★★ longest prefix match（最长前缀匹配优先）
    DFS_WEIGHT = "dfs-weight"  # ★ 深度优先加权

class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""
    FCFS = "fcfs"              # 先来先服务
    LOF = "lof"                # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # prioritize by routing key frequency in running batch
```

> 📖 **LPM**（longest prefix match，最长前缀匹配）／**FCFS**（first come first serve，先来先服务）／**LOF**（longest output first，预计输出最长的先跑）。

```
    ★★ LPM 策略在做什么：
       等待队列里有 100 个请求，显存只够放 10 个。
       ⟹ ★ 优先放【前缀命中最长】的那 10 个
       ⟹ 它们的 Prefill 几乎不用算 ⟹ ★ 立刻就能开始吐字

    ⟹ ★★★ 这就是 vLLM 的哈希表方案给不了的东西。
       没有树，你就不知道「谁命中最长」。
```

> ⚠️ **但要说清两件源码里写着、宣传里不提的事。**

**⚠️ 一：LPM 不是默认值。** [server_args.py:839-854](源码/SGLang/python/sglang/srt/server_args.py) 里 `schedule_policy` 的默认值是 **`"fcfs"`**，可选值是 `lpm / random / fcfs / dfs-weight / lof / priority / routing-key`。★ 要用 LPM 得显式加 `--schedule-policy lpm`。⟹ **「SGLang 天生缓存感知」这句话，对默认配置不成立。**

**⚠️ 二：LPM 会自动退化。**

```python
# ✅ schedule_policy.py:290-294
def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
    if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
        # Turn off the expensive prefix matching and sorting when the #queue is large.
        return CacheAgnosticPolicy.FCFS
    return self.policy
```

```
    ⟹ ★★ 等待队列超过 128 就退回 FCFS——前缀匹配加排序本身不是免费的。
    ⟹ 也就是说：★ 最需要挑活干的高负载时刻，恰恰是它关掉的时刻。
    ⟹ ★ 读框架源码时，这类「自我保护的退化分支」比主路径更值得记：
       它标出了作者认为主路径会失效的边界。
```

> ⚠️ **「同样的显存能多服务多少请求」我没有可回溯的数字**（本地 `papers/` 里的 RadixAttention 论文给的是特定负载下的吞吐倍数，不能当通用结论）。**方向上 LPM 确实省 Prefill，倍数请以你自己的负载实测为准**，压测方法见 [13 章](推理优化入门/13-性能指标与压测.md)。

**排序的实现**就是一行 `sort`：

```python
# ✅ L381-391
def _sort_by_longest_prefix(waiting_queue, temporary_deprioritized):
    """Sorts the waiting queue based on the longest prefix match."""
    waiting_queue.sort(key=lambda r: (
        -r.num_matched_prefix_tokens
        if r.rid not in temporary_deprioritized else float("inf")))
```

```
    ★ 注意 float("inf") 这个分支：被【临时降优先级】的请求排到最后。
    ⟹ 这是防饿死/防抖动的口子——不然一批同前缀的请求会永久压住别人。
```

> ★ **`ROUTING_KEY` 这个策略特别有意思**（注释原文：*"prioritize by routing key frequency in running batch"*，实现在 `_sort_by_routing_key`，**L451**）：按 **MoE 路由键在当前批次里的出现频率**排序。⟹ 把会激活**同一批专家**的请求放到同一步里跑 ⟹ **减少 all-to-all 的通信量和专家负载的抖动**。**这是把 [11 章](推理优化入门/11-MoE推理与专家并行.md) 的 MoE 负载问题，搬到调度层来解决。**

> 📖 **MoE**（Mixture of Experts，混合专家）——每个 token 只激活一小部分专家子网络的模型结构。

### 相关文件

```
    managers/scheduler.py             class Scheduler  L383   ★ 主调度器（4000+ 行）
    managers/scheduler_components/    ★★ 被拆出来的 20 个 .py
      ├─ request_receiver.py          收请求（主循环 L1750 调它）
      ├─ batch_result_processor.py    ★ 写回结果、释放 KV
      └─ output_sender.py             往 ZMQ 发输出
    managers/schedule_batch.py        批次的数据结构
    managers/scheduler_pp_mixin.py    ★ PP 相关逻辑单独拆成 mixin
    managers/prefill_delayer.py       ★ 故意延迟 Prefill（攒批）
    managers/min_free_slots_delayer.py ★ 保底空闲槽位，防抢占颠簸
    managers/data_parallel_controller.py ★ DP 层面的请求分发
```

> ★ `prefill_delayer.py` 和 `min_free_slots_delayer.py` 这两个名字值得注意——它们都是**主动「不做事」**的组件。这是成熟系统才会有的东西：**有时候等一等比立刻做更快。** 为什么 SGLang 特别需要它们，见第 2.2 节那条时间线。

---

## 五、★★ 并行状态：和 vLLM 完全不同的思路

文件：[distributed/parallel_state.py](源码/SGLang/python/sglang/srt/distributed/parallel_state.py)，函数 `initialize_model_parallel`（**L2328**）。

> 📖 **TP / PP / EP / DP**（tensor / pipeline / expert / data parallelism，张量 / 流水线 / 专家 / 数据并行）——四种切法，见 [06 章](推理优化入门/06-推理中的并行策略.md)。
> 📖 **CP**（context parallelism，上下文并行）——把**一条序列的长度**切到多张卡上。
> 📖 **DCP**（decode context parallelism）——只在 Decode 阶段把 KV Cache 切开的变体。

### 第一行就分道扬镳了

```python
# ✅ L2408-2413
if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
    raise RuntimeError(
        f"world_size ({world_size}) is not equal to "
        f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
        f"pipeline_model_parallel_size ({pipeline_model_parallel_size})")
```

```
    ★★★ 读懂它：SGLang 的全局网格里【只有 TP 和 PP】。
        world_size 必须正好等于 tp × pp，没有 DP 维度的位置。

    ⟹ 对比 vLLM 的五维 reshape（ExtDP × DP × PP × PCP × TP），
       这是两种世界观。
```

### 那 DP 从哪来？——从 TP 组【内部】再切

```python
# ✅ L2501 —— 注意力侧的分解
attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

# ✅ L2587 —— MoE 侧的分解
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

**同一批 TP 卡，被两套不同的分解同时看待**：

| 谁在看 | 怎么分解这 `tp` 张卡 | 源码 |
|---|---|---|
| 注意力层 | `tp = attn_dp × attn_cp × attn_tp` | L2501 |
| MoE 层 | `tp = moe_dp × moe_ep × moe_tp` | L2587 |

```
    ⟹ 同样 8 张卡，注意力层可能是「8 个 DP 副本」，
       MoE 层可能是「8 路专家并行」。★ 同时成立。
```

**文档字符串给了一个完整的 8 卡例子**（✅ L2375-2390 原文），把上面那张表变成具体的 rank 列表——设 `tp = 8`、`attn_cp = 2`、`attn_tp = 4`（于是 `attn_dp = 8 ÷ 2 ÷ 4 = 1`）、`moe_dp = 2`、`moe_ep = 4`（于是 `moe_tp = 8 ÷ 4 ÷ 2 = 1`）：

```
    1 个 TP 组：        [g0,g1,g2,g3,g4,g5,g6,g7]      ← ★ 全局只有这一个
    4 个 attn CP 组：   [g0,g4] [g1,g5] [g2,g6] [g3,g7]
    2 个 attn TP 组：   [g0..g3] [g4..g7]
    2 个 moe EP 组：    [g0..g3] [g4..g7]
    4 个 moe DP 组：    [g0,g4] [g1,g5] [g2,g6] [g3,g7]

    ★★ 盯着后四行看：
       attn CP 组 和 moe DP 组 —— 完全是同一批 rank 组合
       attn TP 组 和 moe EP 组 —— 也完全是同一批
    ⟹ 同一条通信链路，在注意力层叫「上下文并行」，
       在 MoE 层叫「数据并行」/「专家并行」。
    ⟹ ★ 这就是「模型部位自己声明怎么切」的字面意思：
       ★★ 卡没动、NCCL 组没动，只是【同一组卡在两个层里承担不同角色】。
```

**各通信组的构造位置**（行号指 `init_model_parallel_group(...)` 那一句）：

| 组 | 行 | 备注 |
|---|---|---|
| `_TP` | L2446 | 连续切：`range(idx*tp, (idx+1)*tp)` |
| `_DCP` | L2486 | ★ 只在 `decode_context_parallel_size > 1` 时建（L2479） |
| `_ATTN_CP` | L2526 | 注意力的上下文并行 |
| `_ATTN_TP` | L2571 | ★ 注意力的张量并行 |
| `_MOE_DP` | L2608 | MoE 的数据并行 |
| `_MOE_EP` | L2636 | ★ MoE 的专家并行 |
| `_MOE_TP` | L2666 | MoE 的张量并行 |
| `_PP` | L2689 | 流水线并行；步长切：`range(idx, world, num_pp_groups)` |

**一个值得学的小优化**——源码里出现了整整五次：

```python
# ✅ L2554-2555（同型写法还有 L2507、L2596、L2621、L2650）
if attn_tp_size == tensor_model_parallel_size:
    _ATTN_TP = _TP      # ★ 不切的话，直接复用 TP 组，不新建 NCCL 通信组
```

> ★ 新建一个 NCCL 通信组是有实打实开销的（握手、显存、句柄）。这个写法出现五次说明作者很在意通信组的数量——**因为 SGLang 的组本来就多**（上面那张表有 8 个）。

### DP Attention 的实现

文件：[layers/dp_attention.py](源码/SGLang/python/sglang/srt/layers/dp_attention.py)

| 符号 | 行 | 是什么 |
|---|---|---|
| `class DpPaddingMode` | L81 | ★ 各 DP rank 序列长度不同时怎么补齐 |
| `compute_dp_attention_world_info` | L326 | ★ 算出本 rank 在 DP-attention 里的身份 |
| `initialize_dp_attention` | L343 | ★★ 初始化 |
| `_dp_gather_via_all_reduce` | L489 | ★ 聚合路径一 |
| `_dp_gather_via_all_gather` | L531 | ★ 聚合路径二 |
| `_dp_gatherv_sizes` | L730 | 变长聚合 |

```
    ★ 为什么要两条聚合路径（L489 / L531）？
      注意力算完，各 DP rank 手里是不同请求的结果，要拼回一个完整批次。
      ⟹ all_reduce 路径：把本地结果填进全长张量的对应位置，其余置零，再 all-reduce
      ⟹ all_gather 路径：直接收集各 rank 的片段
    ⚠️（我的判断，源码没写取舍依据）哪条快取决于【批次形状】：
         各 rank 长度接近 ⟹ all_gather 更省（传的是实际数据）
         各 rank 长度悬殊 ⟹ all_reduce 不用处理变长，实现更简单
    ⟹ ★ 框架两条都实现了，运行时选。
```

> ★★ **`DpPaddingMode`（L81）是 DP Attention 最脏的地方**：DP 的本意是各 rank 独立，但 MoE 层的 all-to-all 又要求大家步调一致 ⟹ **必须把各 rank 的 token 数补齐到一样**，才能进入 MoE 层。这就是 [06 章](推理优化入门/06-推理中的并行策略.md)「DP 组必须一起 generate」在 SGLang 侧的对应体现——**vLLM 用一个专门的主循环（`DPEngineCoreProc`）解决它，SGLang 用一个 padding 模式枚举解决它。**

---

## 六、投机解码：EAGLE 的重兵投入

目录 `speculative/` 有 **34 个 `.py` + 2 个子目录**，**光名字里带 `eagle` 的就有 10 个**：

```
    ★ EAGLE 一族（10 个）
      eagle_info.py / eagle_utils.py / eagle_worker_v2.py / eagle_worker_common.py
      ★ eagle_draft_cuda_graph_runner.py          ← 给草稿模型上 CUDA Graph
      eagle_draft_extend_cuda_graph_runner.py
      multi_layer_eagle_utils.py                  ★ 多层 EAGLE
      multi_layer_eagle_worker_v2.py
      multi_layer_eagle_draft_extend_cuda_graph_runner.py
      ★ eagle_disaggregation.py                   ← ★★ 投机解码 + PD 分离

    ★ MTP（DeepSeek 的多 token 预测）
      frozen_kv_mtp_worker_v2.py / frozen_kv_mtp_info.py
      frozen_kv_mtp_utils.py / frozen_kv_mtp_cuda_graph_runner.py

    ★ n-gram
      ngram_worker.py / ngram_info.py
      ★ cpp_ngram/                                ← C++ 实现，降低查表开销

    ★★ 自适应
      adaptive_spec_params.py
      adaptive_runtime_state.py

    ★ 其他
      base_spec_worker.py / spec_registry.py      基类与策略注册表
      ragged_verify.py                            变长验证
      decoupled_spec_io.py                        解耦的输入输出
      dflash_* / dspark_* / standalone_worker_v2.py  其他算法
```

> 📖 **EAGLE**——投机解码里最主流的草稿方案，用「特征级自回归」的小模型猜下几个 token（论文 arXiv 2401.15077，本地 PDF 在 [papers/](papers/)；⚠️ 缩写的英文全称我没从论文正文核实到，这里不写）。**MTP**（Multi-Token Prediction，多 token 预测）——DeepSeek 的做法，让主模型自己多预测几个 token。两者都见 [08 章](推理优化入门/08-投机解码.md)。

> ★★ **三个值得注意的信号**：
> ① `*_cuda_graph_runner.py` 有 **4 个** —— **草稿模型不上 CUDA Graph，投机解码的收益会被 CPU 派发开销吃掉**（见 [12 章](推理优化入门/12-CUDA-Graph与图编译.md)）。
> ② `eagle_disaggregation.py` / `dflash_disaggregation.py` / `dspark_disaggregation.py` —— 投机解码和 PD 分离**每种算法都要单独做组合**，不是各开各的就行。
> ③ `adaptive_*` —— 印证 [08 章](推理优化入门/08-投机解码.md) 那条规律：**投机解码必须按负载动态开关。**

---

## 七、PD 分离 `disaggregation/`

```
    prefill.py                       ★ P 端主逻辑
    decode.py                        ★ D 端主逻辑
    decode_schedule_batch_mixin.py   D 端的批次组织
    ★ decode_kvcache_offload_manager.py  D 端 KV 换出到 CPU/SSD
    decode_hicache_mixin.py          D 端接分层缓存
    kv_events.py                     KV 传输事件
    base/  common/  utils.py         公共层

    传输后端：
      nixl/        ★ NVIDIA RDMA 传输库（主流）
      mooncake/    ★ 月之暗面的 KV 池
      mori/        ★ AMD 的传输库
      ascend/      ★ 华为昇腾
      ★ fake/      ★★ 假后端 —— 没有 RDMA 也能跑功能测试

    还有：
      encode_server.py / encode_receiver.py / encode_grpc_server.py
      ⟹ ★ 多模态的编码阶段也被拆出去了（不只是 P 和 D 两段）
```

> 📖 **RDMA**（Remote Direct Memory Access，远程直接内存访问）——网卡绕过 CPU 直接读写对端内存/显存，是跨机传 KV 的主力手段。

> ★ `fake/` 这个目录是很好的工程实践：**PD 分离依赖专用硬件，但功能正确性不该依赖硬件才能测。** 值得借鉴。

---

## 八、EPLB `eplb/`：比 vLLM 分得更细

> 📖 **EPLB**（Expert Parallelism Load Balancer，专家并行负载均衡器）——MoE 里热门专家会把某几张卡压满，EPLB 负责把专家搬一搬让负载均匀。

```
    expert_distribution.py       ★ 统计：谁热谁冷
    expert_location.py           ★ 记录：专家现在在哪张卡
    expert_location_dispatch.py  按位置分发 token
    expert_location_updater.py   ★ 在线更新位置（不停服）
    eplb_manager.py              总控
    eplb_algorithms/             多种均衡算法
    lplb_solver.py               求解器
    ★ eplb_simulator/            ★★ 离线模拟器
```

```
    ★★ eplb_simulator/ 是最务实的一个设计：
       拿【历史负载数据】离线跑一遍，先算出最优的专家布局，
       ⟹ 上线时直接用，★ 不用在生产环境试错
       ⟹ 也可以用来回答「加一台机器能好多少」这类容量规划问题
```

> ★ 另外注意顶层还有一个 `elastic_ep/` 目录——**弹性专家并行**，说明 SGLang 在往「运行时增减 EP 规模」这个方向走。**vLLM 侧的对应物是 `parallel_state.py` 里那些 `enable_elastic_ep` 分支**（见 vLLM 拆解第五节）。

---

## 九、★ 读源码的建议路线

```
    第 1 天：managers/scheduler.py 的 event_loop_normal（L1743，只有 30 行）
            + get_next_batch_to_run 的 L3175-3184
            ⟹ 先把第二节那条主线和「Prefill 优先」搞清楚。这是骨架。

    第 2 天：mem_cache/radix_cache.py 全文（863 行，值得通读）
            ⟹ 重点：match_prefix (L377)、_split_node (L705)、
                    inc_lock_ref (L623)、evictable_size (L659)
            ★ 这是 SGLang 的灵魂

    第 3 天：managers/schedule_policy.py
            ⟹ 从 L200 的两个 Enum 开始读，理解「缓存感知」是什么意思
            ★ 一定要读 L290-294 那个退化分支
            ⟹ 再看 PrefillAdder (L511) 怎么往批次里加请求

    第 4 天：distributed/parallel_state.py 的 L2328-2700
            ⟹ 对照 vLLM 的 L1751-1960 一起读
            ★ 两种世界观的直接对比；先读 L2375-2390 的文档字符串例子

    第 5 天：layers/dp_attention.py
            ⟹ 理解「注意力层用 DP」到底是怎么落地的
            ★ 重点看 DpPaddingMode (L81) 解决什么问题

    第 6 天：按需选一个：speculative/ 或 disaggregation/ 或 eplb/
```

---

## 十、★★ 和 vLLM 的核心差异

| | vLLM | SGLang |
|---|---|---|
| **进程结构** | 异步前端 + 引擎，进程间队列 | ★ 前端 / 调度器 / 反分词**三类进程**，ZMQ |
| **一次 step** | ★★ Prefill chunk 和 Decode **混在同一批次**（✅ `enable_chunked_prefill` 默认 `True`） | ★★ 一步**要么全 Prefill 要么全 Decode**；混合要显式开 `enable_mixed_chunk` |
| **并行网格** | 固定五维 `ExtDP×DP×PP×PCP×TP` | ★ 只有 `tp × pp`，其余维度从 TP 组内再切 |
| **谁决定切法** | ★ 部署配置（先定网格） | ★ 模型部位（注意力和 MoE 各自声明） |
| **前缀缓存** | 链式哈希 + 哈希表 | ★★ 基数树（7 棵树 + 1 个非树对照） |
| **调度依据** | 显存 + token 预算 | ★★ 显存 + **前缀命中长度**（LPM，⚠️ 非默认） |
| **缓存层次** | 块池 + 可插拔 offload | ★ `hiradix`：GPU→CPU→SSD 三级内建 |
| **代码规模** | 143 万行 / 4,280 个 `.py` | 184 万行 / 5,674 个 `.py` |
| **最擅长** | ★ 稠密模型、通用部署 | ★ 超大 MoE、多轮对话、共享前缀 |

```
    ⚠️（我的判断，非官方表述）一句话概括两种哲学：

    ★ vLLM：  并行策略是【部署配置】——先定好网格，模型往里填。
              ⟹ 简单、可预测、容易运维

    ★ SGLang：并行策略是【模型属性】——每一部分自己声明想怎么切。
              ⟹ 灵活、能榨出 MoE 的性能，但要跟踪 8 个通信组

    ⟹ 两条路都在收敛：vLLM 补上了 PCP/DCP 和 EP 的独立处理，
       SGLang 也在补 PP（scheduler_pp_mixin.py）。
       ★ 差异在缩小，但设计基因还在。
```

> ⚠️ **别拿这张表做选型**。它描述的是 `bd8865a` / `04444ee` 这两个 commit 的形态，两个项目每个季度都在互相补齐功能。选型请按你自己的负载压测，方法见 [13 章](推理优化入门/13-性能指标与压测.md) 和 [推理框架对比.md](推理框架对比.md)。

---

> 返回 [推理框架总目录](README.md) ｜ 选型见 [推理框架对比.md](推理框架对比.md)
