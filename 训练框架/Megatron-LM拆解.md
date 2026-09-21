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

③ ✅ arguments.py 里显式写出的 add_argument 有 419 处
   ⚠️ 但【419 不等于参数总数】。0.20.0 还会从配置 dataclass 自动生成一批参数：
      megatron/training/argument_utils.py 的 ArgumentGroupFactory
      会把一个 dataclass 的字段逐个展开成命令行参数，
      arguments.py 里这样挂了 15 个配置类（TransformerConfig 在第 2327 行挂上去）。
      ✅ 仅 TransformerConfig 一个类就声明了约 268 个带类型标注的字段。
   ⇒ 所以真实可用的参数【明显多于 419】，
     而且你在 --help 里看到的很多参数，在 arguments.py 里【搜不到】——
     ★ 搜不到的时候，去 megatron/core/transformer/transformer_config.py
       和 megatron/core/model_parallel_config.py 里按字段名（下划线版）搜。
   ★ 这就是为什么训练脚本动辄几十行参数 —— 但真正需要你决定的只有十来个
```

> ⚠️ **一个具体例子，说明上面这条为什么重要**：`--ckpt-format`、`--overlap-moe-expert-parallel-comm`、`--delay-wgrad-compute` 这三个参数都真实可用，但在 `arguments.py` 里都**搜不到 `add_argument`**。前者来自 `CheckpointConfig`（`arguments.py:2949` 挂上），后两者是 `ModelParallelConfig` 的字段（`model_parallel_config.py:343` 和 `:348`），经 `TransformerConfig` 继承后被自动展开。✅ `arguments.py:2967` 那个 `--dist-ckpt-format` 反而是**已废弃**的旧名（`dest='dist_ckpt_format_deprecated'`，help 里直接写 "Deprecated: see --ckpt-format"）。

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
> 张量并行这个"听起来最玄"的东西，**代码量其实最少** —— 因为它的本质就是"矩阵按行切还是按列切" + "什么时候插一次 all-reduce"，逻辑非常收敛。
> ⚠️ 真正臃肿的是 `transformer/`，因为它要支持几十种模型变体。

---

## 2 · ★★★ 主线：一次训练是怎么跑起来的

**这是全篇最重要的一节。搞清楚这条链，你就能自己往下钻任何一环。**

### 2.1 ★★ 调用链：从进程启动到主循环

```
pretrain_gpt.py  __main__                        <- (1) 第 508 行
      |
      |  把三个"提供者函数"传给 pretrain()：
      |    - train_valid_test_datasets_provider    怎么造数据集
      |    - model_provider                        怎么造模型
      |    - forward_step                          一个 micro-batch 怎么前向
      v
megatron/training/training.py  pretrain()        <- (2) 第 1500 行
      |
      +-- initialize_megatron()                     解析参数、初始化 NCCL、建并行组
      +-- setup_model_and_optimizer()             <- (3) 第 2661 行
      |                                             造模型、切分、包 DDP、造优化器
      +-- build_train_valid_test_data_iterators() <- (4) 第 5503 行
      v
            train()                              <- (5) 第 4114 行 【主循环】
              |
              |  while iteration < train_iters:
              v
            train_step()                         <- (6) 第 2997 行 【一步】
              |
              v
megatron/core/pipeline_parallel/schedules.py
            get_forward_backward_func()           <- (7) 第 53 行
```

> ✅ 上面 7 个行号均取自 Megatron-Core 0.20.0（commit `c9676f4`）实测，用 `grep -n "^def pretrain" megatron/training/training.py` 这类命令可逐个复核。

### 2.2 ★★★ 一次 step 到底改了哪些状态

**这是全篇最重要的一张表。** 上面那张图只告诉你"经过了谁"，但学框架真正要搞清的是"**每一步之后，内存里什么东西变了**"。

⚠️ **先纠正一个很容易画错的地方**：很多拆解（包括本文的上一版）会把"梯度规约"画成 `train_step()` 里位于前反向和 `optimizer.step()` **之间**的一个独立步骤。✅ 源码里不是这样：`train_step()` 从 `forward_backward_func(...)` 返回时，**DP 规约已经做完了**。规约是在调度函数内部、前反向循环跑完之后，通过 `config.finalize_model_grads_func` 这个回调触发的。

```
train_step()  training.py:2997
   |
   | (a) while rerun_state_machine.should_run_forward_backward(...)
   |     ★ 前反向被包在一个【重跑状态机】的循环里，不是直接执行
   |
   +-- (b) for model_chunk in model: model_chunk.zero_grad_buffer()
   |        ★ 先清【梯度桶】，不是先清 param.grad
   +-- (c) optimizer.zero_grad()
   |
   +-- (d) forward_backward_func(...)   <- 进入 schedules.py，跑完全部 micro-batch
   |          |
   |          |  ... 前反向按调度算法交错执行，梯度累加进 param.main_grad ...
   |          |
   |          +-- schedules.py:852  if config.finalize_model_grads_func is not None:
   |                     |
   |                     v
   |              finalize_model_grads()   distributed/finalize_model_grads.py:560
   |                     |   ★★ 梯度规约在这里发生，【还在 forward_backward 里面】
   |                     +-- (d1) model_chunk.finish_grad_sync()      DP all-reduce / reduce-scatter
   |                     +-- (d2) _allreduce_conditional_embedding_grads()
   |                     +-- (d3) _allreduce_non_tensor_model_parallel_grads()   LayerNorm 梯度，走 TP 组
   |                     +-- (d4) _allreduce_word_embedding_grads()     首尾 PP stage 之间
   |                     +-- (d5) _update_router_expert_bias()          MoE 才有
   |                     +-- (d6) model_chunk.scale_gradients(1/num_tokens)
   |
   +-- (e) optimizer.step()             <- training.py:3152
   |          返回 (update_successful, grad_norm, num_zeros_in_grad)
   |
   +-- (f) opt_param_scheduler.step(increment=...)   仅当 update_successful 为真
   v
```

✅ `config.finalize_model_grads_func` 的默认值在 `training.py:4325` 被设为 `finalize_model_grads`；调度函数里的触发点在 `schedules.py` 第 852、2081、2475 行（三种调度各一处）。

**逐步状态变化表**（✅ 均对应上面的源码位置）：

| 步 | 调用 | 调用后，什么状态变了 |
|---|---|---|
| a | `should_run_forward_backward()` | 无数据变化；决定这一轮前反向要不要**重跑**（用于 NaN 复查、结果校验） |
| b | `zero_grad_buffer()` | **梯度桶**（`param_and_grad_buffer.py` 里那块连续显存）被清零 |
| c | `optimizer.zero_grad()` | 优化器侧持有的梯度引用被清 |
| d | `forward_backward_func()` | 激活值被创建又释放；`param.main_grad` 从 0 累加到**已规约、已按 token 数缩放**的最终梯度 |
| d1 | `finish_grad_sync()` | 桶内梯度跨 **DP 组**规约。开了 distributed optimizer 走 reduce-scatter（每卡只留自己那片），否则走 all-reduce |
| d3 | `_allreduce_non_tensor_model_parallel_grads()` | **LayerNorm / 非 TP 切分参数**的梯度跨 **TP 组** all-reduce（开 SP 时必需：这些参数在 TP 组内是复制的，梯度必须对齐） |
| d4 | `_allreduce_word_embedding_grads()` | 词嵌入权重共享时，**第一个和最后一个 PP stage** 的 embedding 梯度对齐 |
| d6 | `scale_gradients(1/num_tokens)` | 梯度整体除以全局 batch 的**非 padding token 总数**（per-token loss 归一） |
| e | `optimizer.step()` | **参数本体更新**；Adam 的 m/v 更新；返回梯度范数。失败（溢出）时**不更新**，返回 `update_successful=False` |
| f | `opt_param_scheduler.step()` | 学习率、权重衰减推进一格；`update_successful=False` 时**跳过**，学习率不动 |

```
★★★ 这张表最值得记住的三件事：

  ① 梯度规约不在 train_step 的主干上，在 forward_backward 内部的回调里。
     ⇒ 想插手梯度（裁剪、监控、自定义规约），入口是
       config.finalize_model_grads_func，不是去 train_step 里加一行。

  ② 一次 step 里有【四种不同范围】的梯度通信，不是一次：
       DP 组   全部梯度        (d1)  ← 量最大
       TP 组   LayerNorm 类    (d3)
       PP 首尾 共享 embedding  (d4)
       EP 相关 router bias     (d5)
     ⇒ ★ 新手常以为"梯度同步"就是一次 all-reduce，源码告诉你是四次不同的。

  ③ optimizer.step() 可能【什么都不做】。
     混合精度下梯度溢出就跳过这一步，学习率也跟着不推进。
     ⇒ 这就是日志里 "skipped iterations" 的来源。
```

### 2.3 ★★ 全库最值得读的 8 行代码

`megatron/core/pipeline_parallel/schedules.py` 第 161-168 行（✅ 原文，逐字核对）：

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

> ⚠️ **补一个这 8 行之前就返回的分支**（容易漏）：`schedules.py` 第 154-155 行有一个前置判断 —— 如果传进来的 `schedule_pg_collection` 是 `MultiModuleProcessGroupCollection`（多模块 / 跨"网格"模型，比如 VLM 的视觉塔 + 语言塔分别占不同卡组），**直接返回 `forward_backward_pipelining_without_interleaving`，不再看 PP/VPP**。✅ 所以严格说是"三选一 + 一个前置短路"，共四条路径。

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

✅ 这个文件里的 `torch.autograd.Function` 子类，就是 TP 的通信全集 —— **一共 10 个，行号逐个实测**：

| 类（`mappings.py`） | 行号 | 前向 | 反向 | 切/拼哪一维 | 服务于 |
|---|---:|---|---|---|---|
| `_CopyToModelParallelRegion` | 201 | 恒等 | all-reduce | — | TP |
| `_ReduceFromModelParallelRegion` | 221 | all-reduce | 恒等 | — | TP |
| `_ScatterToModelParallelRegion` | 240 | split | all-gather | 最后一维（hidden） | TP |
| `_GatherFromModelParallelRegion` | 260 | all-gather | split | 最后一维（hidden） | TP |
| `_AllGatherFromTensorParallelRegion` | 384 | all-gather | reduce-scatter | 最后一维（hidden） | TP |
| `_ReduceScatterToTensorParallelRegion` | 404 | reduce-scatter | all-gather | 最后一维（hidden） | TP |
| `_ScatterToSequenceParallelRegion` | 280 | split | all-gather | **第一维（序列）** | SP |
| `_GatherFromSequenceParallelRegion` | 300 | all-gather | reduce-scatter 或 split | **第一维（序列）** | SP |
| `_ReduceScatterToSequenceParallelRegion` | 355 | reduce-scatter | all-gather | **第一维（序列）** | SP |
| `_AllToAll` | 424 | all-to-all | all-to-all | 指定维 | MoE / EP |

> ★★ **这张表里最容易被忽略、又最能说明 TP 和 SP 区别的是"切哪一维"那列**：TP 那几个切的是 `_split_along_last_dim`（**隐藏维**，因为 TP 切的是矩阵的列/行），SP 那几个切的是 `_split_along_first_dim`（**序列维**，因为 SP 切的是 token）。✅ 函数名本身就写着 `last` 和 `first` —— 这是"TP 切宽度、SP 切长度"这句话在代码里的直接证据。
>
> ⚠️ `_GatherFromSequenceParallelRegion` 的反向有**两条路**（源码里按 `tensor_parallel_output_grad` 标志二选一）：要么 `_reduce_scatter_along_first_dim`，要么 `_split_along_first_dim`。表里写"或"不是含糊，是源码真有两个分支。

```
★★★ 这张表真正要看的是【前向和反向那两列互为对偶】：

    前向 恒等        <-> 反向 all-reduce        (_CopyTo...)
    前向 all-reduce  <-> 反向 恒等              (_ReduceFrom...)
    前向 split       <-> 反向 all-gather        (_ScatterTo...)
    前向 all-gather  <-> 反向 split             (_GatherFrom...)

  ⇒ 这就是 [06 章](分布式训练入门/06-张量并行TP.md) 里 f / f̄ 那对算子的实现：
    你只需要写【前向怎么通信】，反向的通信是它的对偶，照抄另一行即可。

★ 再看 SP 那三行（[08 章](分布式训练入门/08-序列并行与上下文并行.md)）：
  出现了 reduce-scatter，而 TP 那四行里没有。
  ⇒ 印证 08 章的核心结论：SP 把一次 all-reduce
    拆成了 reduce-scatter + all-gather，【总通信量不变】，但激活省了。

★ 最后一行 _AllToAll 只有一个，前向反向都是 all-to-all
  —— 它自己就是自己的对偶（[09 章](分布式训练入门/09-专家并行EP.md)）。
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
      ⇒ 需要维护【两套 RNG（Random Number Generator，随机数生成器）状态】并来回切换

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
       → 开关是 --use-distributed-optimizer（✅ arguments.py 里实测存在）
       → [05 章](分布式训练入门/05-ZeRO与FSDP.md)：优化器状态 12Ψ 切成 12Ψ/d
```

> **⚠️ 进门口径声明（Ψ 是什么）**：本节沿用 [05 章](分布式训练入门/05-ZeRO与FSDP.md) 的口径 —— **Ψ = 参数的个数**（不是字节数）。在这个口径下，混合精度 + Adam 的常驻显存是 **16 字节/参数**：FP32 主权重 4 + Adam 一阶动量 m **4** + 二阶动量 v **4**（★ m 和 v 是**各** 4 字节，合计 8，这里最容易漏一半）+ BF16 参数 2 + BF16 梯度 2。其中被 ZeRO-1 切开的"优化器状态"是前三项 = **12 字节/参数**，写成 **12Ψ**，切到 d 路数据并行上就是 **12Ψ/d**。⚠️ 本目录 03 / 05 章在讲通信量时用的是另一个口径（Ψ = 一份梯度的**字节数**），BF16 下与本节差 2 倍 —— **看到 Ψ 先确认口径**。

> ⚠️ **命名陷阱**：Megatron 说的 **"distributed optimizer"** 指的是 **ZeRO-1**（只切优化器状态），不是 ZeRO-3。想要 ZeRO-3 级别的显存节省，要走 `distributed/fsdp/` 那条线。
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

### ✅ 一手数字：`megatron/core/transformer/moe/README.md` 第 430 行

> ⚠️ **注意这个路径**：是 MoE 目录自己的 README，**不是仓库根目录的 `README.md`**（根 README 里没有这段）。本文上一版只写了"README.md 第 430 行"，会让人在根目录里白找一遍。

> *"**EP All-to-All can consume 30-40% of training time without optimization.** These features hide or reduce EP communication overhead."*

```
★★ 这个数字论文里没有，只在源码 README 里。
   它解释了 [09 章](分布式训练入门/09-专家并行EP.md) 的核心矛盾：
   EP 省了显存，但把通信从 all-reduce 换成了 all-to-all，
   而 all-to-all 是【最难优化的集合通信】—— 每张卡给每张卡发不同的数据。

✅ 缓解手段（同一个 README 第 434 行的表格行）：
   --overlap-moe-expert-parallel-comm --delay-wgrad-compute
   原理："Overlaps All-to-All with computation by merging
          FWD-BWD passes of adjacent microbatches"
   ★ 即：把相邻两个 micro-batch 的前向和反向搅在一起，
     用一个的计算去盖另一个的通信。
   ⚠️ 这两个参数在 arguments.py 里搜不到 add_argument ——
      它们是 ModelParallelConfig 的字段
      （model_parallel_config.py:343 和 :348），由 dataclass 自动展开（见 §0）。
      ✅ 源码里还有一条硬约束：delay_wgrad_compute 必须和
        overlap_moe_expert_parallel_comm 一起开
        （transformer_config.py:3025 的断言原文
         "overlap_moe_expert_parallel_comm must be enabled when
          enabling delay_wgrad_compute"）。

✅ 另一条 README 原文（第 191 行）：
   "For very large MoE models like DeepSeek-V3, the EP communication
    may exceed the NVLink bandwidth."
   ⚠️ 注意它说的是【超过 NVLink 带宽】，不是超过跨机带宽 ——
      说明大 MoE 的 EP 通信压力已经大到机内互联都吃不消。
```

> ★ **把"超过 NVLink 带宽"换成数字感受一下**（⚠️ 以下为标称值推算，非实测）：NVLink 常被引用的 **900 GB/s 是双向聚合**口径，**算单向要用 450 GB/s**；而跨机的 **InfiniBand（IB，无限带宽，一种高速网络互联标准）** 400 **Gb**/s ÷ 8 = **50 GB/s**。⇒ 同口径下 **NVLink : IB ≈ 9 倍**（不是 18 倍 —— 18 倍是拿 NVLink 的双向值去比 IB 的单向值，两边口径不一致）。所以 README 那句话的分量是：EP 的 all-to-all 已经吃满了那个**快 9 倍**的机内链路，一旦它溢出到机外，差距是数量级的。

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

### 7.1 ⚠️ 先把 `order` 这件事说准确

很多资料（包括本文上一版）会写成"Megatron 的 rank 排布顺序是 `tp-cp-ep-dp-pp`"，好像它是一个写死的常量。✅ **源码里不是常量，是一个默认参数值**：

```python
# megatron/core/parallel_state.py:617
def initialize_model_parallel(
    ...
    order: str = "tp-cp-ep-dp-pp",
    ...
) -> None:
```

⇒ 它是 `initialize_model_parallel()` 的**第 17 个参数的默认值**，调用方可以传别的顺序进去。★ 这个区别很重要：**它是可配置的策略，不是框架的硬性约定** —— 你想验证"TP 排外层会掉多少 MFU"，改这一个字符串就行。

⚠️ **而且这一个字符串会被拆成【两套】排布，不是一套。** 这是本节最容易读错的地方：

| | 稠密（decoder）排布 | 专家（expert）排布 |
|---|---|---|
| 构造位置 | `parallel_state.py:859` | `parallel_state.py:887` |
| 实际 order 串 | `tp-gtp_remat-cp-ep-dp-pp` | `tp-cp-ep-gtp_remat-dp-pp` |
| `ep` 取值 | **强制为 1** | `expert_model_parallel_size` |
| `cp` 取值 | `context_parallel_size` | **强制为 1** |
| `dp` 取值 | `data_parallel_size` | `world_size ÷ (etp × ep × pp × egtp)` |
| `tp` 取值 | `tensor_model_parallel_size` | `expert_tensor_parallel_size`（默认同 TP） |

✅ 证据有两条。一是 `RankGenerator.__init__`（`parallel_state.py:479-482`）开头就断言两者不能共存，断言消息把原因写得很清楚：

> *"Both EP and CP > 1 in not allow in one rank generator. **CP is only included in default RankGenerator, and EP only in expert RankGenerator.**"*

二是 `_inject_gtp_remat_axis()`（`parallel_state.py:582`）的 docstring 解释了两套串为什么插在不同位置：

> *"Position controls locality (**leftmost token = smallest stride = most adjacent ranks**)"* —— 稠密侧插在 `tp` 后，专家侧插在 `ep` 后，*"so EP keeps more-local placement than EGTP (the MoE EP all-to-all is the heavier expert-side collective)"*。

```
★★ 所以"排布顺序"的正确读法是：

   一个 order 字符串 "tp-cp-ep-dp-pp"
        |
        +--> 稠密 RankGenerator：ep 置 1，CP 生效  → 管 TP/CP/DP/PP 组
        +--> 专家 RankGenerator：cp 置 1，EP 生效  → 管 MoE 那套 ETP/EP/EDP 组

   ⇒ ⚠️ 不要以为 CP 和 EP 在同一个 5 维网格里各占一维 ——
     它们分属两套网格，只是共用同一个顺序声明。
```

### 7.2 ★★★ 用具体 rank 号算一遍（16 卡，TP=2 / PP=2 / DP=4）

上面 docstring 给的是 TP=2 / PP=4 / DP=2。换一组更常见的配置手推一遍，规律就清楚了。

**推导规则**（从 order 串直接得到）：`tp` 在最左 ⇒ **步长最小、变化最快**；`pp` 在最右 ⇒ **步长最大、变化最慢**。CP=EP=1 时：

```
rank = tp_idx + TP x (dp_idx + DP x pp_idx)
     = tp_idx + 2 x (dp_idx + 4 x pp_idx)
```

⇒ 16 张卡的归属（`g0`–`g15`，假设 8 卡一台机，`g0`–`g7` 在机器 A，`g8`–`g15` 在机器 B）：

| rank | tp_idx | dp_idx | pp_idx | 在哪台机 |
|---|---:|---:|---:|---|
| g0 | 0 | 0 | 0 | A |
| g1 | 1 | 0 | 0 | A |
| g2 | 0 | 1 | 0 | A |
| g3 | 1 | 1 | 0 | A |
| g4 | 0 | 2 | 0 | A |
| g5 | 1 | 2 | 0 | A |
| g6 | 0 | 3 | 0 | A |
| g7 | 1 | 3 | 0 | A |
| g8 | 0 | 0 | 1 | B |
| g9 | 1 | 0 | 1 | B |
| g10 | 0 | 1 | 1 | B |
| g11 | 1 | 1 | 1 | B |
| g12 | 0 | 2 | 1 | B |
| g13 | 1 | 2 | 1 | B |
| g14 | 0 | 3 | 1 | B |
| g15 | 1 | 3 | 1 | B |

**⇒ 三种组的完整成员表**：

| 组类型 | 组数 | 每组大小 | 具体成员 | rank 间隔 | 落在哪 |
|---|---:|---:|---|---|---|
| **TP 组** | 8 | 2 | `[g0,g1] [g2,g3] [g4,g5] [g6,g7]` `[g8,g9] [g10,g11] [g12,g13] [g14,g15]` | **1（紧邻）** | ✅ 必定同机，走 NVLink |
| **DP 组** | 4 | 4 | `[g0,g2,g4,g6]` `[g1,g3,g5,g7]` `[g8,g10,g12,g14]` `[g9,g11,g13,g15]` | 2 | 同机 |
| **PP 组** | 8 | 2 | `[g0,g8] [g1,g9] [g2,g10] [g3,g11]` `[g4,g12] [g5,g13] [g6,g14] [g7,g15]` | **8（跨得最远）** | ⚠️ 必定跨机，走 IB |

```
★★ 自检：组数 x 组大小应该等于 16
     TP: 8 x 2 = 16  ✅
     DP: 4 x 4 = 16  ✅
     PP: 8 x 2 = 16  ✅
   ★ 这个自检很有用 —— 手推 rank 表时一旦哪个乘不出 world_size，就是推错了。
```

> ✅ **这套推导方法是可验证的**：用同样的公式去算源码 docstring 里那组（TP=2 / PP=4 / DP=2），得到的 DP 组是 `[g0,g2] [g1,g3] [g4,g6] [g5,g7] [g8,g10] [g9,g11] [g12,g14] [g13,g15]`、PP 组是 `[g0,g4,g8,g12] [g1,g5,g9,g13] [g2,g6,g10,g14] [g3,g7,g11,g15]` —— **和 §7 开头引用的 docstring 逐字一致**。所以上面那张 TP=2/PP=2/DP=4 的表可以放心用。

### 7.3 ★★ 为什么是这个顺序

```
★★★ 这段注释把 [12 章](分布式训练入门/12-并行策略组合与MFU.md) 的 Takeaway #1 变成了代码事实：

   order = "tp-cp-ep-dp-pp"
            ^最内层          ^最外层
           （步长最小）      （步长最大）

   TP 在最内层  ⇒  TP 组的 rank 号最紧密相邻（上表里间隔 = 1）
                ⇒  TP 组落在同一台机器里，走 NVLink
                ⇒  ✅ 而 TP 恰恰是通信最频繁的那个（每层 2 次 all-reduce）

   PP 在最外层  ⇒  PP 组跨机器（上表里间隔 = 8）
                ⇒  ✅ 而 PP 恰恰是通信最少的那个（每 stage 边界 1 次 send/recv）

★★ 一句话：【通信越频繁的并行方式，排得越靠内层】。
   这个排布不是随意的，是把通信频次和链路带宽做了匹配。

✅ 还有一条源码级的印证：parallel_state.py:898-902 有一句断言，
   要求满足以下三者之一：
     ① order 以 "pp" 结尾（即 PP 在最外层），或
     ② PP = 1（没开流水线，顺序无所谓），或
     ③ 专家侧 DP 大小 == 稠密侧 DP 大小
   ⚠️ 注意它不是"PP 必须在最外层"的死规定，条件 ③ 留了口子。
   ✅ 断言消息原文说明了它真正防的是什么：
     "the data parallel size of the attention and moe layers must be the same"
   ⇒ 把 PP 挪走会让稠密侧和专家侧的 DP 大小对不上，
     两套网格就没法共用同一组 PP 组了
     （紧接着第 904 行还有一句断言，直接检查两套生成器算出的 PP 组必须相同）。
```

> ⚠️ **实践提醒**：如果你手工改 rank 映射（比如某些调度器会打乱），可能在不知情的情况下把 TP 组拆到了两台机器上，MFU 会莫名其妙掉一大截。
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
| 知道训练怎么跑起来的 | `pretrain_gpt.py:508` → `training/training.py:1500` |
| 看一次 step 的完整状态变化 | `training/training.py:2997`（`train_step`）★ 配合本文 §2.2 那张表 |
| 改流水线调度 | `pipeline_parallel/schedules.py:53` 起 |
| **插手梯度（裁剪/监控/自定义规约）** | `distributed/finalize_model_grads.py:560` ★★ 入口是 `config.finalize_model_grads_func`，**不要**去 `train_step` 里加行（见 §2.2） |
| 理解 TP 怎么切矩阵 | `tensor_parallel/layers.py:906`（Column）/ `:1294`（Row） |
| 理解 TP 的通信 | `tensor_parallel/mappings.py` ★ 10 个类的对偶表见 §3.2 |
| 加一种新并行组 | `core/parallel_state.py` ⚠️ 注意有**两套** RankGenerator（见 §7.1） |
| 查一个参数为什么搜不到 | `core/transformer/transformer_config.py` + `core/model_parallel_config.py` ★ 见 §0 |
| 改重计算策略 | `core/recompute.py` + `transformer/transformer_block.py` |
| 改 MoE 路由 | `transformer/moe/router.py` |
| 改 MoE 通信 | `transformer/moe/token_dispatcher.py` |
| 改优化器分片 | `core/optimizer/` |
| 改 checkpoint 格式 | `core/dist_checkpointing/` |
| 抄一份能跑的参数 | `examples/` 下对应模型目录 |
| 确认某功能怎么用 | `tests/` 里搜函数名 ★ 常比文档准 |

> ✅ 表里所有行号均为 Megatron-Core 0.20.0（`c9676f4`）实测。⚠️ 换版本后行号会漂，但**类名和函数名**在这几年里相当稳定，按名字搜即可。

---

## 10 · ⚠️ 读这份源码时的六个坑

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

④ 参数极多，但绝大多数有合理默认值
   ⚠️ 不要因为看到一个参数就觉得需要调它。
   ★ 真正需要你决策的，就是 [12 章 §12.4](分布式训练入门/12-并行策略组合与MFU.md)
     那七步里涉及的十来个。

⑤ ⚠️ 参数不止 arguments.py 里那 419 处 add_argument
   还有一大批从配置 dataclass 自动展开（见 §0）。
   ⇒ 在 arguments.py 里搜不到某个参数【不等于它不存在】，
     要去 transformer_config.py / model_parallel_config.py 按下划线名再搜一遍。

⑥ ⚠️ 别把"梯度规约"当成 train_step 主干上的一步
   它在 forward_backward_func 内部的回调里（见 §2.2）。
   ★ 这个误解会直接导致你改错地方 —— 在 train_step 里加的那行
     会在梯度【已经规约完】之后才执行。
```

---

## 11 · 本章小结

```
① Megatron-Core 是【库】，megatron/training 是【脚本】，两者可分开用。

② ★★ 主线只有一条：
   pretrain_gpt.py → pretrain() → train() → train_step()
                                              → forward_backward_func()

③ ★★★ 一次 step 的状态变化比调用链更值得记（§2.2）：
   清梯度桶 → 前反向累加 main_grad → 【规约在 forward_backward 内部的
   回调里】→ optimizer.step() 改参数 → 学习率推进。
   ⚠️ 规约不在 train_step 主干上，这是最容易画错的一处。

④ ★★ 一次 step 里有【四种范围】的梯度通信，不是一次：
   DP 组（全部梯度）/ TP 组（LayerNorm 类）/ PP 首尾（共享 embedding）
   / EP（router bias）。

⑤ ★★★ 全库最值得读的是 schedules.py 第 161-168 行那 8 行分派 ——
   PP 的三种调度在那里一目了然（⚠️ 另有一个多模块前置短路，见 §2.3）。

⑥ ★★ TP 的全部实现就是两个类（Column/RowParallelLinear）
   加一组对偶的通信算子（mappings.py 里 10 个类）。代码量小得出人意料。
   ★ 而 TP 与 SP 的分界，就写在函数名的 last / first 上（§3.2）。

⑦ ★★★ parallel_state.py 的 rank 排布（TP 最内、PP 最外）
   是"通信频次匹配链路带宽"这条原则的代码化，
   也是 PTD-P Takeaway #1 能成立的工程前提。
   ⚠️ 但要说准确：order 是【默认参数值】而非常量，
     而且同一个字符串会拆成【稠密 + 专家两套】排布（§7.1）。

⑧ ✅ 源码 README 里藏着论文没有的数字，
   比如 MoE 的 "EP All-to-All 未优化时占 30-40% 训练时间"
   （出处是 moe/README.md，不是根 README）。

⑨ ⚠️ "419 个参数"只是 arguments.py 里 add_argument 的处数，
   不是参数总数 —— 还有一大批从 dataclass 自动展开（§0）。

⑩ ★ 有没有 fault_injector 和 straggler 检测，
   是判断一个框架有没有真上过万卡的实用标志。
```

---

> [← 返回目录](README.md) ｜ [分布式训练入门教程](分布式训练入门/README.md) ｜ [训练框架对比](训练框架对比.md)
