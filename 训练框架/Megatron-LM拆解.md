> [← 返回目录](README.md) ｜ [分布式训练入门教程](分布式训练入门/README.md) ｜ [训练框架对比](训练框架对比.md)

# Megatron-LM 源码拆解

> **这份文档的定位**：教程讲的是**原理**，这份讲的是**这些原理在代码里长什么样**。
> 所有目录、行号、类名、参数名，均来自 [源码/Megatron-LM/](源码/Megatron-LM/) 实际走查（✅ 一手）。
>
> ✅ **本次走查的版本**：Megatron-Core **0.20.0**，commit `c9676f4`
> ⚠️ 行号会随版本变化，**结构和文件名相对稳定**，看不到对应行号时按类名/函数名搜索即可。

---

## 0 · 先说清楚三件事

```
① Megatron-LM ≠ Megatron-Core，这是两个东西
   megatron/core/     ← Megatron-Core，可被别的框架当【库】引用（NeMo 就是这么用的）
   megatron/training/ ← 训练脚本层，参数解析、主循环、日志、存档
   ★ 你在别的项目里看到 "from megatron.core import ..."，用的就是前者

② ✅ 代码规模：638 个 .py 文件，约 24 万行
   ⚠️ 不要试图从头读。本文给的是【按需切入点】。

③ ✅ 命令行参数有 419 个（arguments.py 里 add_argument 的个数）
   ★ 这就是为什么训练脚本动辄几十行参数 —— 但真正需要你决定的只有十来个
```

---

## 1 · ★ 一分钟看懂目录结构

```
Megatron-LM/
├── pretrain_gpt.py          ★★ 入口。想知道"训练是怎么跑起来的"，从这里开始
├── pretrain_vlm.py             多模态版入口
├── pretrain_mamba.py           Mamba 版入口
│
├── megatron/
│   ├── core/                ★★★ 核心库（可复用），下面细讲
│   ├── training/            训练循环、参数、checkpoint（约 2.6 万行）
│   ├── inference/           推理
│   ├── rl/                  ★ RL 后训练（约 7 千行，14 章讲的那些）
│   ├── post_training/       量化感知训练等
│   └── elastification/      弹性伸缩
│
├── examples/                ★ 各模型的启动脚本，抄参数用
├── tools/                   数据预处理、checkpoint 转换
└── tests/                   ★ 想确认某个功能怎么用，测试文件常比文档清楚
```

### megatron/core/ 的分工（✅ 实测行数，按大小排序）

| 目录 | 行数 | 管什么 | 对应教程章节 |
|---|---:|---|---|
| `transformer/` | 44819 | 模型结构本身：attention、MLP、MoE、MLA、MTP | — |
| `inference/` | 35909 | 推理引擎 | — |
| `models/` | 18925 | GPT/BERT/T5/多模态的组装 | — |
| `distributed/` | 16989 | **DP、梯度桶、FSDP** | [04](分布式训练入门/04-数据并行DP.md)、[05](分布式训练入门/05-ZeRO与FSDP.md) |
| `optimizer/` | 9900 | 分布式优化器（= ZeRO-1） | [05](分布式训练入门/05-ZeRO与FSDP.md) |
| `pipeline_parallel/` | 8069 | **PP 的调度算法** | [07](分布式训练入门/07-流水线并行PP.md) |
| `dist_checkpointing/` | 7917 | 分布式存档 | [13](分布式训练入门/13-断点续训与容错.md) |
| `tensor_parallel/` | 7330 | **TP 的切法与通信** | [06](分布式训练入门/06-张量并行TP.md) |
| `datasets/` | 5659 | 数据集与索引 | — |

> ★ **一个反直觉的观察**：`tensor_parallel/` 只有 **7330 行**，是核心目录里最小的之一。
> 张量并行这个"听起来最玄"的东西，**代码量其实最少** —— 因为它的本质就是
> "矩阵按行切还是按列切" + "什么时候插一次 all-reduce"，逻辑非常收敛。
> ⚠️ 真正臃肿的是 `transformer/`，因为它要支持几十种模型变体。

---

## 2 · ★★★ 主线：一次训练是怎么跑起来的

**这是全篇最重要的一节。搞清楚这条链，你就能自己往下钻任何一环。**

```
pretrain_gpt.py  __main__                          ← ✅ 第 508 行
      │
      │  把三个"提供者函数"传给 pretrain()：
      │    - train_valid_test_datasets_provider  怎么造数据集
      │    - model_provider                      怎么造模型
      │    - forward_step                        一个 micro-batch 怎么前向
      ▼
megatron/training/training.py  pretrain()           ← ✅ 第 1500 行
      │
      ├─ initialize_megatron()      解析 419 个参数、初始化 NCCL、建各种并行组
      ├─ setup_model_and_optimizer()← ✅ 第 2661 行：造模型、切分、包 DDP、造优化器
      ├─ build_train_valid_test_data_iterators()
      ▼
                    train()                         ← ✅ 第 4114 行  【主循环】
                      │
                      │  while iteration < train_iters:
                      ▼
                  train_step()                      ← ✅ 第 2997 行  【一步】
                      │
                      ├─ ① optimizer.zero_grad()
                      ├─ ② forward_backward_func(...)   ★★ 这里进入流水线调度
                      ├─ ③ 梯度规约（DP all-reduce / reduce-scatter）
                      ├─ ④ optimizer.step()
                      └─ ⑤ 学习率调度
                      ▼
megatron/core/pipeline_parallel/schedules.py
                  get_forward_backward_func()       ← ✅ 第 53 行
```

### ★★ 全库最值得读的 8 行代码

`megatron/core/pipeline_parallel/schedules.py` 第 161-168 行（✅ 原文）：

```python
    if pp_size > 1:
        if vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func
```

```
★★★ 这 8 行就是 [07 章](分布式训练入门/07-流水线并行PP.md) 的全部内容：

    PP = 1                → forward_backward_no_pipelining
                            （没有流水线，前向完直接反向）

    PP > 1，没开 vpp      → forward_backward_pipelining_without_interleaving
                            （标准 1F1B，气泡率 (p−1)/(m+p−1)）

    PP > 1，开了 vpp      → forward_backward_pipelining_with_interleaving
                            （交错式，气泡率再除以 vpp）
                            ✅ 这个函数从第 1008 行开始，长达 1000 多行
                              —— 交错式调度确实是最复杂的那个

★ 教学价值：一个框架的"策略选择"往往就浓缩在这样一小段分支里。
  找到这段分支，等于找到了整个模块的地图。
```

---

## 3 · 张量并行：`tensor_parallel/`

### 3.1 ★★ 两个类，就是 TP 的全部

✅ `megatron/core/tensor_parallel/layers.py`：

| 类 | 行号 | 论文原文的定义 |
|---|---:|---|
| `ColumnParallelLinear` | 906 | *"The linear layer is defined as **Y = XA + b**. A is parallelized along its **second dimension** as A = [A_1, ..., A_p]"* |
| `RowParallelLinear` | 1294 | *"A is parallelized along its **first dimension** and X along its **second dimension**. A = transpose([A_1 .. A_p]), X = [X_1, ..., X_p]"* |
| `VocabParallelEmbedding` | 230 | 词表切开（词表大时很关键） |

```
★★ 对照 [06 章](分布式训练入门/06-张量并行TP.md)：

   MLP 里的两个矩阵：
      第一个 h → 4h    用 ColumnParallelLinear（按列切）
      第二个 4h → h    用 RowParallelLinear（按行切）

   ★ 为什么必须是这个顺序？
     列切的输出天然是"每卡拿一部分列"，正好就是行切需要的输入形式，
     ⇒ ★★ 中间【不需要通信】，只在最后做一次 all-reduce。
        反过来排就要通信两次。

   代码印证：RowParallelLinear 有个参数叫
     input_is_parallel:
       "If true, we assume that the input is already split
        across the GPUs and we do not split again"
     ★ 这个参数的存在，就是上面那句话的代码化。
```

### 3.2 通信原语在 `mappings.py`

✅ 这个文件里的 `torch.autograd.Function` 子类，是 TP 的通信全集：

```
_CopyToModelParallelRegion        前向 = 恒等，反向 = all-reduce
_ReduceFromModelParallelRegion    前向 = all-reduce，反向 = 恒等
_ScatterToModelParallelRegion     前向 = split，反向 = all-gather
_GatherFromModelParallelRegion    前向 = all-gather，反向 = split

★★★ 注意这四个是【两两对偶】的：
    前向做 all-reduce 的，反向什么都不做；
    前向什么都不做的，反向做 all-reduce。
    ⇒ 这就是 [06 章](分布式训练入门/06-张量并行TP.md) 里 f / f̄ 那对算子的实现。

序列并行（[08 章](分布式训练入门/08-序列并行与上下文并行.md)）多出来的三个：
_ScatterToSequenceParallelRegion
_GatherFromSequenceParallelRegion         ← all-gather
_ReduceScatterToSequenceParallelRegion    ← reduce-scatter

★ 印证 08 章的核心结论：SP 把一次 all-reduce
  拆成了 reduce-scatter + all-gather，【总通信量不变】，但激活省了。

_AllToAll                                  ← MoE 用（[09 章](分布式训练入门/09-专家并行EP.md)）
```

### 3.3 ⚠️ 一个容易被忽略的文件：`random.py`

```
★ tensor_parallel/random.py 管的是【随机数状态】。

  为什么 TP 需要专门管随机数？
      TP 组内的各卡持有同一层的不同切片
        ↓
      Dropout 如果各卡各自随机，掩码就对不上
        ↓
      ⇒ 必须让"同一份数据的 dropout 掩码"在 TP 组内一致，
        而"不同数据"之间又要真的随机
        ↓
      ⇒ 需要维护【两套 RNG 状态】并来回切换

⚠️ 这也是 [13 章](分布式训练入门/13-断点续训与容错.md) 说的
   "checkpoint 必须存随机数状态"的原因之一 ——
   ★ 不存的话，续训后 dropout 模式变了，虽然不影响收敛，
     但会让"复现某个 bug"变得不可能。
```

---

## 4 · 流水线并行：`pipeline_parallel/`

```
schedules.py                  ★★ 三个调度算法（见 §2）
p2p_communication.py          ★ 点对点收发的封装
                                 → PP 的通信只有 send/recv，没有集合通信
                                   这是它通信量最小的原因（[07 章](分布式训练入门/07-流水线并行PP.md)）
combined_1f1b.py              1F1B 的组合优化
fine_grained_activation_offload.py   ★ 细粒度激活卸载（[11 章](分布式训练入门/11-重计算与激活优化.md)）
hybrid_cp_schedule.py         ★ CP 和 PP 混合时的调度
bridge_communicator.py        跨"网格"通信（多模块模型，如 VLM 的视觉塔与语言塔）
```

### ★ 一个细节：`deallocate_output_tensor()`

✅ `schedules.py` 第 171 行。

```
流水线里，一个 stage 的输出发给下一个 stage 之后，
本地这份【还得留着】—— 因为反向要用。

★ 但其实不用留【整个张量】，只需要留住它的"壳"（shape/dtype/grad_fn），
  数据部分可以立刻释放。

⇒ 这个函数干的就是这件事。
★★ 教学价值：这是"显存优化"最典型的形态 ——
   不是换个算法，而是发现【某块内存其实没人再读了】。
   [11 章](分布式训练入门/11-重计算与激活优化.md) 的所有技术，本质都是这个思路的变种。
```

---

## 5 · 数据并行与优化器：`distributed/` + `optimizer/`

```
distributed/
├── distributed_data_parallel.py    ★ Megatron 自己的 DDP（不是 PyTorch 那个）
├── param_and_grad_buffer.py        ★★ 梯度桶（bucket）
│      → [04 章](分布式训练入门/04-数据并行DP.md) 讲的"攒够一桶再发一次"
├── finalize_model_grads.py         ★ 梯度规约的收尾（含跨 PP 的 embedding 梯度同步）
├── fsdp/                           Megatron 自己的 FSDP 实现
└── torch_fully_sharded_data_parallel.py   ★ 也支持直接用 PyTorch 原生 FSDP

optimizer/                          ★★ "分布式优化器" = ZeRO-1
       → 开关是 --use-distributed-optimizer
       → [05 章](分布式训练入门/05-ZeRO与FSDP.md)：优化器状态 12Ψ 切成 12Ψ/d
```

> ⚠️ **命名陷阱**：Megatron 说的 **"distributed optimizer"** 指的是 **ZeRO-1**（只切优化器状态），
> 不是 ZeRO-3。想要 ZeRO-3 级别的显存节省，要走 `distributed/fsdp/` 那条线。
> ★ 这个术语混淆在实践中造成的误解非常多。

### ★ 重叠通信的三个开关（✅ 实际参数名）

```
--overlap-grad-reduce                     梯度规约与反向重叠
--overlap-param-gather                    参数收集与前向重叠
--overlap-param-gather-with-optimizer-step  参数收集与优化器步重叠

★★ 对照 [12 章 §12.6](分布式训练入门/12-并行策略组合与MFU.md) 的 MegaScale Table 3：
   三种重叠（TP/PP/DP）总共带来约 4 个百分点的 MFU，
   而且【不改模型、不损精度】。
   ⇒ 这几个开关属于"没有理由不开"的那一类。
```

---

## 6 · MoE：`transformer/moe/`

```
moe_layer.py           MoE 层的组装
router.py              ★★ 路由：决定每个 token 去哪几个专家
token_dispatcher.py    ★★★ 分发：把 token 发到专家所在的卡（all-to-all）
experts.py             专家本体（GroupedMLP / SequentialMLP）
shared_experts.py      ★ 共享专家（DeepSeek 系列的做法）
fused_a2a.py           ★ 融合的 all-to-all
moe_utils.py           负载均衡损失等
```

### ✅ 一手数字：README.md 第 430 行

> *"**EP All-to-All can consume 30-40% of training time without optimization.** These features hide or reduce EP communication overhead."*

```
★★ 这个数字论文里没有，只在源码 README 里。
   它解释了 [09 章](分布式训练入门/09-专家并行EP.md) 的核心矛盾：
   EP 省了显存，但把通信从 all-reduce 换成了 all-to-all，
   而 all-to-all 是【最难优化的集合通信】—— 每张卡给每张卡发不同的数据。

✅ 缓解手段（README 第 434 行）：
   --overlap-moe-expert-parallel-comm --delay-wgrad-compute
   原理："Overlaps All-to-All with computation by merging
          FWD-BWD passes of adjacent microbatches"
   ★ 即：把相邻两个 micro-batch 的前向和反向搅在一起，
     用一个的计算去盖另一个的通信。

✅ 另一条 README 原文（第 191 行）：
   "For very large MoE models like DeepSeek-V3, the EP communication
    may exceed the NVLink bandwidth."
   ⚠️ 注意它说的是【超过 NVLink 带宽】，不是超过跨机带宽 ——
      说明大 MoE 的 EP 通信压力已经大到机内互联都吃不消。
```

---

## 7 · ★★ 并行组是怎么建出来的：`parallel_state.py`

**这是理解"多维并行"最关键的一个文件。**

✅ 文件里给了一个极好的例子（第 755-769 行，原文）：

> *"Let's say we have a total of **16 GPUs** denoted by g0 ... g15 and we use **2 GPUs to parallelize the model tensor**, and **4 GPUs to parallelize the model pipeline**. The present function will create 8 tensor model-parallel groups, 4 pipeline model-parallel groups and 8 data-parallel groups as:"*

```
                TP = 2,  PP = 4,  DP = 2      （2 × 4 × 2 = 16 ✅）

  8 个 TP 组：[g0,g1] [g2,g3] [g4,g5] [g6,g7] [g8,g9] [g10,g11] [g12,g13] [g14,g15]
                ↑ ★★ 相邻 rank！所以一定在同一台机器里

  8 个 DP 组：[g0,g2] [g1,g3] [g4,g6] [g5,g7] [g8,g10] [g9,g11] [g12,g14] [g13,g15]
                ↑ 隔 2 个

  4 个 PP 组：[g0,g4,g8,g12] [g1,g5,g9,g13] [g2,g6,g10,g14] [g3,g7,g11,g15]
                ↑ ★ 隔 4 个，跨得最远
```

✅ 紧接着的原文警告（第 766-769 行）：

> *"Note that **for efficiency, the caller should make sure adjacent ranks are on the same DGX box**. For example if we are using 2 DGX-1 boxes with a total of 16 GPUs, rank 0 to 7 belong to the first box and ranks 8 to 15 belong to the second box."*

```
★★★ 这段注释把 [12 章](分布式训练入门/12-并行策略组合与MFU.md) 的 Takeaway #1 变成了代码事实：

   rank 排布顺序：TP → CP → EP → DP → PP
                  ↑最内层                ↑最外层

   TP 在最内层  ⇒  TP 组的 rank 号最紧密相邻
                ⇒  TP 组落在同一台机器里，走 NVLink
                ⇒  ✅ 而 TP 恰恰是通信最频繁的那个（每层 2 次 all-reduce）

   PP 在最外层  ⇒  PP 组跨机器
                ⇒  ✅ 而 PP 恰恰是通信最少的那个（每 stage 边界 1 次 send/recv）

★★ 一句话：【通信越频繁的并行方式，排得越靠内层】。
   这个排布不是随意的，是把通信频次和链路带宽做了匹配。
```

> ⚠️ **实践提醒**：如果你手工改 rank 映射（比如某些调度器会打乱），
> 可能在不知情的情况下把 TP 组拆到了两台机器上，MFU 会莫名其妙掉一大截。
> ★ [12 章 §12.5](分布式训练入门/12-并行策略组合与MFU.md) 排查清单的第 4 条查的就是这个。

---

## 8 · 容错与存档

```
core/dist_checkpointing/       ★★ 分布式 checkpoint
       → --ckpt-format torch_dist
       → [13 章](分布式训练入门/13-断点续训与容错.md)：存取时并行度可以不同

core/rerun_state_machine.py    ★ 出错重跑的状态机
core/fault_injector.py         ★ 故障注入 —— 主动制造故障来测容错
core/README_STRAGGLER.md       ★★ 慢卡检测的专门文档
core/energy_monitor.py         功耗监控
core/telemetry/                遥测
```

```
★★ 看到 fault_injector.py 和 README_STRAGGLER.md 这两个文件，
   就能明白 [13 章](分布式训练入门/13-断点续训与容错.md) 那句
   ✅ "Failures and stragglers are the norm rather than the exception"
   不是修辞 —— 框架里专门有模块来【制造故障】和【抓慢卡】。

★ 一个健康的判断标准：
  一个训练框架有没有"故障注入"和"慢卡检测"，
  基本能说明它有没有真正在万卡规模跑过。
```

---

## 9 · ★ 按需切入表：我想改 X，该看哪个文件

| 我想…… | 去看 |
|---|---|
| 知道训练怎么跑起来的 | `pretrain_gpt.py` → `training/training.py:1500` |
| 改流水线调度 | `pipeline_parallel/schedules.py:53` 起 |
| 理解 TP 怎么切矩阵 | `tensor_parallel/layers.py:906` / `:1294` |
| 理解 TP 的通信 | `tensor_parallel/mappings.py` |
| 加一种新并行组 | `core/parallel_state.py` |
| 改重计算策略 | `core/recompute.py` + `transformer/transformer_block.py` |
| 改 MoE 路由 | `transformer/moe/router.py` |
| 改 MoE 通信 | `transformer/moe/token_dispatcher.py` |
| 改优化器分片 | `core/optimizer/` |
| 改 checkpoint 格式 | `core/dist_checkpointing/` |
| 抄一份能跑的参数 | `examples/` 下对应模型目录 |
| 确认某功能怎么用 | `tests/` 里搜函数名 ★ 常比文档准 |

---

## 10 · ⚠️ 读这份源码时的四个坑

```
① Megatron 有【两套】DDP 和【两套】FSDP
   - megatron/core/distributed/distributed_data_parallel.py   自己的 DDP
   - megatron/core/distributed/fsdp/                          自己的 FSDP
   - torch_fully_sharded_data_parallel.py                     PyTorch 原生 FSDP 的封装
   ⚠️ 看代码时先确认你的配置走的是哪一条，否则会读错分支。

② "distributed optimizer" 是 ZeRO-1，不是 ZeRO-3（见 §5）

③ 很多算子有 Transformer Engine 和 原生 PyTorch 两套实现
   transformer/custom_layers/ 和 core/extensions/ 下是 TE 版
   ⚠️ 生产配置基本都走 TE 版，但 TE 版更难读。
   ★ 想理解【原理】读原生版，想理解【性能】读 TE 版。

④ 419 个参数里，绝大多数有合理默认值
   ⚠️ 不要因为看到一个参数就觉得需要调它。
   ★ 真正需要你决策的，就是 [12 章 §12.4](分布式训练入门/12-并行策略组合与MFU.md)
     那七步里涉及的十来个。
```

---

## 11 · 本章小结

```
① Megatron-Core 是【库】，megatron/training 是【脚本】，两者可分开用。

② ★★ 主线只有一条：
   pretrain_gpt.py → pretrain() → train() → train_step()
                                              → forward_backward_func()

③ ★★★ 全库最值得读的是 schedules.py 里那 8 行分派 ——
   PP 的三种调度在那里一目了然。

④ ★★ TP 的全部实现就是两个类（Column/RowParallelLinear）
   加一组对偶的通信算子（mappings.py）。代码量小得出人意料。

⑤ ★★★ parallel_state.py 的 rank 排布（TP 最内、PP 最外）
   是"通信频次匹配链路带宽"这条原则的代码化，
   也是 PTD-P Takeaway #1 能成立的工程前提。

⑥ ✅ 源码 README 里藏着论文没有的数字，
   比如 MoE 的 "EP All-to-All 未优化时占 30-40% 训练时间"。

⑦ ★ 有没有 fault_injector 和 straggler 检测，
   是判断一个框架有没有真上过万卡的实用标志。
```

---

> [← 返回目录](README.md) ｜ [分布式训练入门教程](分布式训练入门/README.md) ｜ [训练框架对比](训练框架对比.md)
