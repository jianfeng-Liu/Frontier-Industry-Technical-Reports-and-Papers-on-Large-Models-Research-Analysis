# SGLang 源码拆解

> 版本：commit `04444ee`（2026-08-20 拉取），5,674 个 `.py` 文件 / 1,836,778 行。
> **读法建议**：先读完 [vLLM拆解.md](vLLM拆解.md) 再读这篇。
> SGLang 的很多设计只有和 vLLM 对照才看得出用意。

---

## 一、目录长什么样

代码主体在 `python/sglang/srt/`（**srt = SGLang RunTime**）：

```
    srt/
    ├── managers/        ★★ 调度器、批次、策略  ← 相当于 vLLM 的 v1/core/sched
    ├── mem_cache/       ★★ 内存池 + 各种前缀缓存  ← 本文重点，40+ 个文件
    ├── distributed/     ★★ 并行状态  ← 06 章的源头
    ├── layers/          ★ 各种层，含 dp_attention.py
    ├── speculative/     ★ 投机解码（35+ 文件，EAGLE 占 8 个）
    ├── disaggregation/  ★ PD 分离
    ├── eplb/            ★ 专家负载均衡（★ 含离线模拟器）
    ├── model_executor/  执行器
    ├── compilation/     图编译
    ├── elastic_ep/      ★ 弹性专家并行
    └── entrypoints/     HTTP 服务
```

> ★ **第一印象就能看出取向**：`mem_cache/` 有 40+ 个文件（vLLM 对应的 `v1/core/` 只有 8 个）。
> **SGLang 把「缓存」当成了一等公民**，这和它的论文主张（RadixAttention）完全一致。

---

## 二、★★ RadixAttention：基数树前缀缓存

这是 SGLang 的招牌，也是它和 vLLM 最本质的分歧点。

文件：[mem_cache/radix_cache.py](源码/SGLang/python/sglang/srt/mem_cache/radix_cache.py)

```
    class RadixKey                L59    ★ 树的键（token 序列 + 额外标识）
    class TreeNode                L238   ★ 树节点
        def evicted               L269
    class RadixCache              L303   ★★ 主类
        def match_prefix          L377   ★★ 最长前缀匹配 ← 核心操作
        def insert                L437   插入
        def cache_finished_req    L459   ★ 请求结束时把 KV 交还给树
        def cache_unfinished_req  L516   ★ 请求进行中也能先缓存
        def evict                 L593   ★ LRU 淘汰
        def inc_lock_ref          L623   ★★ 引用计数锁：在用的块不许淘汰
        def evictable_size        L659   ★ 可淘汰的量（调度器要看这个）
        def _split_node           L705   ★★ 边分裂 ← 基数树的标志性操作
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

```
    radix_cache.py              ★ 标准版
    radix_cache_cpp.py          ★ C++ 实现（降低树操作的 CPU 开销）
    hiradix_cache.py            ★★ 分层（hierarchical）：GPU → CPU → SSD 三级
    swa_radix_cache.py          滑动窗口注意力专用
    pure_swa_radix_cache.py     纯滑动窗口
    mamba_radix_cache.py        ★ Mamba 状态专用
    unified_radix_cache.py      ★ 混合结构模型（一部分注意力 + 一部分 Mamba）
    chunk_cache.py              不用树的简单版（对照组）
```

> 🔑 **为什么要这么多种？**
> 05 章讲的树，隐含假设是「KV 只增不减、可以任意复用」。
> **滑动窗口注意力会丢弃旧 KV，Mamba 根本没有 KV 只有状态**——
> 这两种情况下树的语义完全不同，必须分开实现。
> ★ 这也解释了为什么 `mem_cache/` 有 40 个文件：**现代模型结构已经不统一了。**

---

## 三、★★ 缓存感知调度：树结构的真正回报

文件：[managers/schedule_policy.py](源码/SGLang/python/sglang/srt/managers/schedule_policy.py)

**源码把调度策略明确分成了两类**，这个划分本身就是论点：

```python
# ✅ L200
class CacheAwarePolicy(Enum):
    """Scheduling policies that are aware of the tree cache."""
    LPM = "lpm"                # ★★ longest prefix match（最长前缀匹配优先）
    DFS_WEIGHT = "dfs-weight"  # ★ 深度优先加权

# ✅ L207
class CacheAgnosticPolicy(Enum):
    """Scheduling policies that are not aware of the tree cache."""
    FCFS = "fcfs"              # 先来先服务
    LOF = "lof"                # longest output first
    RANDOM = "random"
    ROUTING_KEY = "routing-key"  # ★ 按 MoE 路由键的热度排（★ 见下）
```

```
    ★★ LPM 策略在做什么：
       等待队列里有 100 个请求，显存只够放 10 个。
       ⟹ ★ 优先放【前缀命中最长】的那 10 个
       ⟹ 它们的 Prefill 几乎不用算 ⟹ ★ 立刻就能开始吐字
       ⟹ 同样的显存，服务的请求数显著更多

    ⟹ ★★★ 这就是 vLLM 的哈希表方案给不了的东西。
       没有树，你就不知道「谁命中最长」。
```

> ★ **`ROUTING_KEY` 这个策略特别有意思**（注释：*"prioritize by routing key frequency in running batch"*）：
> 按 **MoE 路由键在当前批次里的出现频率**排序。
> ⟹ 把会激活**同一批专家**的请求放到同一步里跑
> ⟹ **减少 all-to-all 的通信量和专家负载的抖动**（11 章）。
> **这是把 11 章的 MoE 负载问题，搬到调度层来解决。**

其他相关文件：

```
    managers/scheduler.py             class Scheduler  L383   ★ 主调度器
    managers/schedule_batch.py        批次的数据结构
    managers/scheduler_pp_mixin.py    ★ PP 相关逻辑单独拆成 mixin
    managers/prefill_delayer.py       ★ 故意延迟 Prefill（攒批）
    managers/min_free_slots_delayer.py ★ 保底空闲槽位，防抢占颠簸
    managers/data_parallel_controller.py ★ DP 层面的请求分发
```

> ★ `prefill_delayer.py` 和 `min_free_slots_delayer.py` 这两个名字值得注意——
> 它们都是**主动"不做事"**的组件。这是成熟系统才会有的东西：
> **有时候等一等比立刻做更快。**

---

## 四、★★ 并行状态：和 vLLM 完全不同的思路

文件：[distributed/parallel_state.py](源码/SGLang/python/sglang/srt/distributed/parallel_state.py)

```
    def initialize_model_parallel    L2328
```

### 第一行就分道扬镳了

```python
# ✅ L2408
if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
    raise RuntimeError(...)
```

```
    ★★★ 读懂它：SGLang 的全局网格里【只有 TP 和 PP】。
        没有 DP 维度的位置。

    ⟹ 对比 vLLM 的 5 维 reshape（ExtDP×DP×PP×PCP×TP），
       这是两种世界观。
```

### 那 DP 从哪来？——从 TP 组【内部】再切

```python
# ✅ L2587
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

```
    ★★ 同一批 TP 卡，被【两套不同的分解】同时看待：

    ┌────────────────────────────────────────────────────┐
    │  注意力层看到的：  tp = attn_dp × attn_cp × attn_tp  │
    │  MoE 层看到的：    tp = moe_dp × moe_ep × moe_tp    │
    └────────────────────────────────────────────────────┘

    ⟹ 同样 8 张卡，注意力层可能是「8 个 DP 副本」，
       MoE 层可能是「8 路专家并行」。★ 同时成立。
```

**各通信组的构造位置**：

```
    _TP        L2446    张量并行组
    _ATTN_CP   L2526    注意力的上下文并行
    _ATTN_TP   L2571    ★ 注意力的张量并行
    _MOE_DP    ~L2620   MoE 的数据并行
    _MOE_EP    L2636    ★ MoE 的专家并行
    _MOE_TP    L2666    MoE 的张量并行
    _PP        L2689    流水线并行
```

**一个值得学的小优化**（L2554）：

```python
if attn_tp_size == tensor_model_parallel_size:
    _ATTN_TP = _TP      # ★ 不切的话，直接复用 TP 组，不新建 NCCL 通信组
```

> ★ 新建一个 NCCL 通信组是有实打实开销的（握手、显存、句柄）。
> 这一行说明作者很在意通信组的数量——**因为 SGLang 的组本来就多**。

### DP Attention 的实现

文件：[layers/dp_attention.py](源码/SGLang/python/sglang/srt/layers/dp_attention.py)

```
    class DpPaddingMode            L81    ★ 各 DP rank 序列长度不同时怎么补齐
    compute_dp_attention_world_info L326  ★ 算出本 rank 在 DP-attention 里的身份
    initialize_dp_attention        L343   ★★ 初始化
    _dp_gather_via_all_reduce      L489   ★ 聚合路径一
    _dp_gather_via_all_gather      L531   ★ 聚合路径二
    _dp_gatherv_sizes              L730   变长聚合
```

```
    ★ 为什么要两条聚合路径（L489 / L531）？
      注意力算完，各 DP rank 手里是不同请求的结果，要拼回一个完整批次。
      ⟹ all_reduce 路径：把本地结果填进全长张量的对应位置，其余置零，再 all-reduce
      ⟹ all_gather 路径：直接收集各 rank 的片段
      ★★ 哪条快取决于【批次形状】：
         各 rank 长度接近 ⟹ all_gather 更省（传的是实际数据）
         各 rank 长度悬殊 ⟹ all_reduce 可能更简单（不用处理变长）
    ⟹ ★ 框架两条都实现了，运行时选。
```

> ★★ **`DpPaddingMode`（L81）是 DP Attention 最脏的地方**：
> DP 的本意是各 rank 独立，但 MoE 层的 all-to-all 又要求大家步调一致
> ⟹ **必须把各 rank 的 token 数补齐到一样**，才能进入 MoE 层。
> 这就是 06 章「DP 组必须一起 generate」在 SGLang 侧的对应体现。

---

## 五、投机解码：EAGLE 的重兵投入

目录 `speculative/` 有 35+ 个文件，**光 EAGLE 相关就有 8 个**：

```
    ★ EAGLE 一族
      eagle_info.py / eagle_utils.py / eagle_worker_v2.py
      eagle_worker_common.py
      ★ eagle_draft_cuda_graph_runner.py          ← 给草稿模型上 CUDA Graph
      eagle_draft_extend_cuda_graph_runner.py
      multi_layer_eagle_utils.py                  ★ 多层 EAGLE
      multi_layer_eagle_draft_extend_cuda_graph_runner.py
      ★ eagle_disaggregation.py                   ← ★★ 投机解码 + PD 分离

    ★ MTP（DeepSeek 的多 token 预测）
      frozen_kv_mtp_worker_v2.py / frozen_kv_mtp_info.py
      frozen_kv_mtp_cuda_graph_runner.py

    ★ n-gram
      ngram_worker.py / ngram_info.py
      ★ cpp_ngram/                                ← C++ 实现，降低查表开销

    ★★ 自适应（08 章 §8.6 的关键）
      adaptive_spec_params.py
      adaptive_runtime_state.py

    ★ 其他
      ragged_verify.py       变长验证
      spec_registry.py       策略注册表
      decoupled_spec_io.py   解耦的输入输出
```

> ★★ **三个值得注意的信号**：
> ① `*_cuda_graph_runner.py` 有 5 个 —— **草稿模型不上 CUDA Graph，投机解码的收益会被 CPU 派发开销吃掉**（12 章 §12.7）。
> ② `eagle_disaggregation.py` —— 投机解码和 PD 分离**要专门做组合**，不是各开各的就行。
> ③ `adaptive_*` —— 印证 08 章那条规律：**投机解码必须按负载动态开关。**

---

## 六、PD 分离 `disaggregation/`

```
    prefill.py                       ★ P 端主逻辑
    decode.py                        ★ D 端主逻辑
    decode_schedule_batch_mixin.py   D 端的批次组织
    ★ decode_kvcache_offload_manager.py  D 端 KV 换出到 CPU/SSD
    decode_hicache_mixin.py          D 端接分层缓存
    kv_events.py                     KV 传输事件

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

> ★ `fake/` 这个目录是很好的工程实践：**PD 分离依赖专用硬件，
> 但功能正确性不该依赖硬件才能测。** 值得借鉴。

---

## 七、EPLB `eplb/`：比 vLLM 分得更细

```
    expert_distribution.py     ★ 统计：谁热谁冷
    expert_location.py         ★ 记录：专家现在在哪张卡
    expert_location_dispatch.py  按位置分发 token
    expert_location_updater.py ★ 在线更新位置（不停服）
    eplb_manager.py            总控
    eplb_algorithms/           多种均衡算法
    lplb_solver.py             求解器
    ★ eplb_simulator/          ★★ 离线模拟器
```

```
    ★★ eplb_simulator/ 是最务实的一个设计：
       拿【历史负载数据】离线跑一遍，先算出最优的专家布局，
       ⟹ 上线时直接用，★ 不用在生产环境试错
       ⟹ 也可以用来回答「加一台机器能好多少」这类容量规划问题
```

> ★ 另外注意顶层还有一个 `elastic_ep/` 目录——**弹性专家并行**，
> 说明 SGLang 在往「运行时增减 EP 规模」这个方向走。

---

## 八、★ 读源码的建议路线

```
    第 1 天：mem_cache/radix_cache.py 全文（★ 只有 700 多行，值得通读）
            ⟹ 重点：match_prefix (L377)、_split_node (L705)、
                    inc_lock_ref (L623)
            ★ 这是 SGLang 的灵魂

    第 2 天：managers/schedule_policy.py
            ⟹ 从 L200 的两个 Enum 开始读，理解「缓存感知」是什么意思
            ⟹ 再看 PrefillAdder (L511) 怎么往批次里加请求

    第 3 天：distributed/parallel_state.py L2328-2700
            ⟹ 对照 vLLM 的 L1751-1960 一起读
            ★ 两种世界观的直接对比

    第 4 天：layers/dp_attention.py
            ⟹ 理解「注意力层用 DP」到底是怎么落地的
            ★ 重点看 DpPaddingMode (L81) 解决什么问题

    第 5 天：按需选一个：speculative/ 或 disaggregation/ 或 eplb/
```

---

## 九、★★ 和 vLLM 的核心差异

| | vLLM | SGLang |
|---|---|---|
| **并行网格** | 固定五维 `ExtDP×DP×PP×PCP×TP` | ★ 只有 `tp × pp`，DP 从 TP 组内再切 |
| **谁决定切法** | ★ 部署配置（先定网格） | ★ 模型部位（注意力和 MoE 各自声明） |
| **前缀缓存** | 链式哈希 + 哈希表 | ★★ 基数树（8 个变体） |
| **调度依据** | 显存 + token 预算 | ★★ 显存 + **前缀命中长度**（LPM） |
| **缓存层次** | 块池 + 可插拔 offload | ★ hiradix：GPU→CPU→SSD 三级内建 |
| **代码规模** | 143 万行 | 184 万行 |
| **最擅长** | ★ 稠密模型、通用部署 | ★ 超大 MoE、多轮对话、共享前缀 |

```
    ⚠️（我的判断，非官方表述）一句话概括两种哲学：

    ★ vLLM：  并行策略是【部署配置】——先定好网格，模型往里填。
              ⟹ 简单、可预测、容易运维

    ★ SGLang：并行策略是【模型属性】——每一部分自己声明想怎么切。
              ⟹ 灵活、能榨出 MoE 的性能，但要跟踪 6+ 个通信组

    ⟹ 两条路都在收敛：vLLM 补上了 DCP/PCP 和 EP 的独立处理，
       SGLang 也在补 PP。★ 差异在缩小，但设计基因还在。
```

---

> 返回 [推理框架总目录](README.md) ｜ 选型见 [推理框架对比.md](推理框架对比.md)
