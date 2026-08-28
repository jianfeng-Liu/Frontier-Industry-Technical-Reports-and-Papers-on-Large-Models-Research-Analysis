# vLLM 源码拆解

> 版本：commit `bd8865a`（2026-08-20 拉取），4,280 个 `.py` 文件 / 1,432,376 行。
> **本文只走主干**：一个请求从进来到吐字，中间经过哪些类、哪些文件、哪几行。
> 所有行号都是我在这个 commit 上实际 `grep`/`sed` 出来的，可以直接点开对照。

---

## 一、先看目录长什么样

```
    vllm/
    ├── v1/              ★★ 主战场。V1 是重写过的引擎，现在的默认路径
    │   ├── engine/      ★ 请求的入口和出口
    │   ├── core/        ★★ 调度器 + KV Cache 块管理  ← 本文重点
    │   ├── worker/      ★ 真正跑模型的地方（GPUModelRunner）
    │   ├── attention/   注意力后端（FlashAttention / FlashInfer / MLA ...）
    │   ├── spec_decode/ ★ 投机解码
    │   ├── sample/      采样
    │   └── kv_offload/  KV 换出
    ├── distributed/     ★★ 并行状态、KV 传输、EPLB  ← 06/10/11 章的源头
    ├── model_executor/  ★ 各种并行线性层、量化
    ├── models/          具体模型定义
    ├── config/          ★ 配置（CUDAGraphMode 在这里）
    ├── engine/          V0 遗留（★ 读代码时别走错）
    └── entrypoints/     OpenAI 兼容的 HTTP 服务
```

> 🔑 **第一个坑**：`vllm/engine/` 是 **V0 的老代码**，`vllm/v1/engine/` 才是现在用的。
> 目录名几乎一样，很容易读错文件。**本文讲的全部是 v1。**

---

## 二、一个请求的完整旅程

```
    HTTP 请求
       │  entrypoints/openai/
       ▼
    ┌──────────────────────────────────────────┐
    │  AsyncLLM        v1/engine/async_llm.py:72│  ★ 异步入口，管流式输出
    └───────────────────┬──────────────────────┘
                        │
    ┌───────────────────▼──────────────────────┐
    │  EngineCore      v1/engine/core.py:104   │  ★ 引擎主循环
    │   └─ EngineCoreProc      (L1007)         │  跑在独立进程里
    │   └─ ★ DPEngineCoreProc  (L1986)         │  ★★ DP 模式的主循环
    └───────────────────┬──────────────────────┘
                        │  每一步循环调用
    ┌───────────────────▼──────────────────────┐
    │  Scheduler.schedule()                    │  ★★ 决定这一步跑谁
    │      v1/core/sched/scheduler.py:477      │     （见第三节）
    └───────────────────┬──────────────────────┘
                        │  产出 SchedulerOutput
    ┌───────────────────▼──────────────────────┐
    │  GPUModelRunner.execute_model()          │  ★ 组织张量、跑前向
    │      v1/worker/gpu_model_runner.py:4288  │
    │      （类定义在 L501）                     │
    └───────────────────┬──────────────────────┘
                        │  产出 logits → 采样
    ┌───────────────────▼──────────────────────┐
    │  Scheduler.update_from_output()          │  ★ 收结果、判结束、释放块
    │      v1/core/sched/scheduler.py:1737     │
    └──────────────────────────────────────────┘
                        │
                        ▼  回到 EngineCore 下一步循环
```

> ★ **注意 `DPEngineCoreProc`（L1986）这个类**。它就是 06 章 §6.3 说的
> 「DP 组必须一起 generate，否则死锁」的实现所在——DP 模式下需要一个专门的主循环
> 来保证所有 DP rank 步调一致。还有 `DPMoEEngineCoreActor`（L2519），
> ★ 名字里带 MoE，说明 MoE + DP 的组合有专门处理。

---

## 三、★★ 调度器：一次 `schedule()` 到底做了什么

这是 vLLM 最值得读的一个函数。文件：[v1/core/sched/scheduler.py](源码/vLLM/vllm/v1/core/sched/scheduler.py)。

```
    class Scheduler              L69
    def schedule(self)           ★★ L477
```

### 第 1 步：开预算

```python
# ✅ L497
token_budget = self.max_num_scheduled_tokens
spec = self.vllm_config.speculative_config
draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
input_budget = self.scheduler_config.max_num_batched_tokens
```

```
    ★ 三个预算：
      token_budget  —— 这一步总共能算多少 token（03 章的 chunked prefill 粒度）
      input_budget  —— 输入张量的容量上限
      ★ draft_slots —— 给投机解码的草稿 token 预留的位置（08 章）
```

### 第 2 步：先排 running 队列（老用户优先，保证吐字不断）

```python
# ✅ L559-568（min 是跨三行的写法）
num_new_tokens = (request.num_tokens_with_spec
                  + request.num_output_placeholders
                  - request.num_computed_tokens)
if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold   # ★ 单条封顶
num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)  # ★ 总预算夹紧
```

```
    ★★ 这四行就是 chunked prefill 的全部核心：
       ① 算出「这个请求还欠多少 token 没算」
       ② ★ 单条超长 prompt 先自己封顶（long_prefill_token_threshold）
          ⟹ 防止一条 32K 的 prompt 独占整步
       ③ ★ 再被总预算夹一次
       ④ 减掉 draft_slots ⟹ 给投机解码留位置
```

### 第 3 步：装不下就抢占

```python
# ✅ L1340
def _preempt_request(self, request, timestamp, drop_stale_output=False):
    ...
    self._free_request_blocks(request)      # ★ 释放全部 KV 块
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0          # ★★ 归零 = 走【重算】路线
```

```
    ★ 从 running 队列【尾部】选（最晚进来的先踢）⟹ 保住 FCFS 语义
    ★★ num_computed_tokens = 0 说明 v1 主推【重算】而非【换出到 CPU】
       原因见 04 章：有前缀缓存兜底，重算时块很可能还在，不真的重算
```

### 第 4 步：再排 waiting 队列（放新人进来）

新请求进来要先申请 KV 块，这里会尝试命中前缀缓存（见第四节）。

### 第 5 步：断言守门

```python
# ✅ L1172
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
```

> ★ 这条断言是「预算制」的最后一道保险。读代码时看到它，就知道
> **上面所有分支的 `min()` 加起来必须自洽**。

---

## 四、★ KV Cache 管理：块池 + 链式哈希

### 块池

文件：[v1/core/block_pool.py](源码/vLLM/vllm/v1/core/block_pool.py)

```
    class BlockHashToBlockMap    L33    ★ 哈希 → 块，前缀缓存的查找表
    class BlockPool              L143   ★ 所有 KV 块的池子
    def cache_full_blocks        L225   ★★ 只有【写满】的块才进缓存
    def cache_partial_block      L445   ★ 半满的块走另一条路
```

> 🔑 **为什么要区分满块和半块？**
> 半满的块内容还会变（下一个 token 就写进来了），
> **它的哈希此刻算出来，下一步就失效了。** 所以只有写满、内容冻结的块才能进缓存表。

### 链式哈希

文件：[v1/core/kv_cache_utils.py](源码/vLLM/vllm/v1/core/kv_cache_utils.py)

```
    NONE_HASH                       L102   ★ 哈希链的起点
    DEFAULT_NONE_HASH_SEED          L106   种子字符串 "vllm-none-hash"
    generate_block_hash_extra_keys  L580   ★ 额外键：多模态输入、LoRA id 等
    hash_block_tokens               L618   ★★ 块哈希函数本体
```

```
    ★★ 哈希是【链式】的：
       block_hash[i] = hash( block_hash[i-1], 这 16 个 token, extra_keys )
                              ↑ 上一块的哈希

    ⟹ 只有【整条前缀路径完全相同】才会命中，
       中间任何一个 token 不同，后面全部不命中。★ 这是正确性的保证。

    ⚠️ 也是 13 章那个坑的成因：prompt 开头放时间戳
       ⟹ 第 1 块哈希就不同 ⟹ ★ 整条链全废。
```

### 上层管理

```
    v1/core/kv_cache_manager.py            对外接口
    v1/core/kv_cache_coordinator.py        ★ 协调多种 KV 类型
    v1/core/single_type_kv_cache_manager.py
```

> ★ 为什么需要「多种 KV 类型」？因为现在的模型不再是清一色的注意力了——
> Mamba/线性注意力的状态、滑动窗口注意力的有限窗口、全注意力，
> **三者的缓存生命周期完全不同**，需要分开管理。
> 印证：`v1/attention/backends/` 里有 `mamba1_attn.py` / `mamba2_attn.py` /
> `linear_attn.py` / `short_conv_attn.py`。

---

## 五、★★ 并行状态：06 章的源头

文件：[distributed/parallel_state.py](源码/vLLM/vllm/distributed/parallel_state.py)

```
    def initialize_model_parallel    L1751
```

**核心就是一次 reshape 加若干次 transpose**：

```python
# ✅ L1817 —— 全文最重要的一行注释
# the layout order is: ExternalDP x DP x PP x PCP x TP

# ✅ L1826
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size)
```

代码注释还给了取组的通用手法（L1824 附近）：

> *"to get group_ranks for each dimension, transpose that dimension to the last dimension, then reshape to 2D, then unbind the last dimension"*

```
    ★ 于是各组的构造就是一行一个：
      TP   L1843   all_ranks.view(-1, tp).unbind(0)        ← ★ 已在最后，不用转置
      PCP          transpose(3, 4)
      PP   L1891   all_ranks.transpose(2, 4)
      DP   L1908   all_ranks.transpose(1, 4)
      EP   L1923   all_ranks.transpose(1, 2).reshape(-1, dp*pcp*tp)
                   ★ EP 组 = 同一个 PP 段内的所有卡
```

**文档字符串里的例子**（8 卡 TP=2 PP=4）：

```
    TP 组：[g0,g1] [g2,g3] [g4,g5] [g6,g7]     ★ 相邻，走 NVLink
    PP 组：[g0,g2,g4,g6] [g1,g3,g5,g7]         步长 2
```

> ★★ 06 章的全部结论都从这几行推出来。**如果只读一个文件，读这个。**

### 并行线性层

文件：[model_executor/layers/linear.py](源码/vLLM/vllm/model_executor/layers/linear.py)

```
    ColumnParallelLinear         L407    ★ 列切；gather_output 默认 False
    ├─ MergedColumnParallelLinear L645   多个列切层合并（MLP 的 gate+up）
    └─ QKVParallelLinear          L971   ★ QKV 三个矩阵合成一个列切层
    DCPGroupColumnParallelLinear  L604   ★ 给 DCP 用的变体
    RowParallelLinear            L1510   ★★ 行切；唯一的通信在 L1660
```

```python
# ✅ L1660 —— 推理 TP 的全部通信，就这一句
output = tensor_model_parallel_all_reduce(output_parallel)
```

```
    ★★ 请注意两件事：
    ① 整个文件里【没有任何反向传播的通信】
       ⟹ 这就是 01 章「推理 TP 每层 2 次 all-reduce，训练 4 次」的源码证据
    ② ColumnParallelLinear 的 gather_output 默认 False
       ⟹ ★ 列切的输出【故意不聚合】，直接喂给下一个行切层
       ⟹ 一个「列切 + 行切」的组合，全程只需要【最后那一次】all-reduce
```

---

## 六、其余模块的地图

### 投机解码 `v1/spec_decode/`

```
    eagle.py              ★ EAGLE（08 章说的最强方案）
    medusa.py             Medusa 多头
    ngram_proposer.py     ★ n-gram 查表，零成本
    ngram_proposer_gpu.py   GPU 版
    suffix_decoding.py    ★ 后缀解码
    draft_model.py        独立小模型
    dynamic/              ★★ 自适应——按负载动态调整（08 章 §8.6）
    metrics.py            接受率统计
```

### KV 传输 `distributed/kv_transfer/kv_connector/v1/`

```
    base.py                  ★ 统一接口
    nixl/                    ★ NVIDIA 的 RDMA 传输库（主流）
    mooncake/                ★ 月之暗面的 KV 池
    lmcache_connector.py     LMCache
    moriio/  hf3fs/          其他后端
    offloading/              换出到 CPU/SSD
    ★ multi_connector.py     ★★ 串联多个后端：先查本地，再查远端
```

### EPLB `distributed/eplb/`

```
    eplb_state.py            当前专家分布
    policy/                  重排算法
    rebalance_execute.py     ★ 执行搬运
    ★ async_worker.py        ★★ 异步做，不阻塞推理 ← 能上线的关键
```

### 量化 `model_executor/layers/quantization/`

```
    fp8.py / input_quant_fp8.py / fbgemm_fp8.py   ★ FP8 一族（H100 主战场）
    auto_awq.py / auto_gptq.py                    ★ 权重量化
    ★ kv_cache.py                                 ★★ KV Cache 量化（09 章）
    mxfp4.py                                      Blackwell
    moe_wna16.py / experts_int8.py                MoE 专用
    torchao.py / modelopt.py                      外部工具链
```

### CUDA Graph `config/compilation.py:53`

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)         # ★ 元组 = 两种批次两种模式
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)  # ★★ chunked prefill 下的推荐值
```

详见 [12 章](推理优化入门/12-CUDA-Graph与图编译.md) §12.4。

---

## 七、★ 读源码的建议路线

```
    第 1 天：只读 distributed/parallel_state.py 的 L1751-1960
            ⟹ 把 06 章的 rank 布局彻底搞清楚。这是骨架。

    第 2 天：读 v1/core/sched/scheduler.py 的 schedule()
            ⟹ 从 L477 一路读到 L700，看预算怎么被消耗
            ★ 重点看每一处 min() 和 continue

    第 3 天：读 v1/core/block_pool.py + kv_cache_utils.py
            ⟹ 搞清楚一个块从申请到被缓存到被淘汰的完整生命

    第 4 天：读 model_executor/layers/linear.py
            ⟹ 对照 06 章验证「每层只有 1 次 all-reduce」

    第 5 天：v1/worker/gpu_model_runner.py（★ 这个文件很大，4000+ 行）
            ⟹ 只看 execute_model() (L4288)，看 SchedulerOutput
               怎么变成张量
```

> ⚠️ **不要一开始就读 `models/`**。那里是各个模型的具体定义，
> 长得都差不多，读了对理解系统没帮助。

---

## 八、和 SGLang 的关键差异（速览）

| | vLLM |
|---|---|
| 并行网格 | ★ 固定五维 reshape：`ExtDP×DP×PP×PCP×TP` |
| 前缀缓存 | ★ 链式哈希 + 哈希表 |
| 调度依据 | 显存够不够 + token 预算 |
| 抢占 | ★ 重算（`num_computed_tokens = 0`） |
| 设计取向 | ★ 并行策略是**部署配置**，先定网格，模型往里填 |

详细对比见 [推理框架对比.md](推理框架对比.md)。

---

> 返回 [推理框架总目录](README.md) ｜ 下一篇 [SGLang拆解.md](SGLang拆解.md)
