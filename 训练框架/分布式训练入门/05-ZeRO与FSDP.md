> [← 04 数据并行 DP](04-数据并行DP.md) ｜ [目录](README.md) ｜ 下一章 → [06 张量并行 TP](06-张量并行TP.md)

# 05 · ZeRO 与 FSDP：把冗余切掉

> **这一章的目标**：讲清一个非常朴素、但威力巨大的想法。
> 读完你应该能回答：ZeRO 三个阶段各切了什么、各省多少？为什么 stage 1 和 2 是"白赚"？stage 3 那 1.5 倍通信量是怎么来的？以及 FSDP 和 ZeRO-3 到底是不是一回事？

> 📌 **ZeRO** = **Ze**ro **R**edundancy **O**ptimizer，零冗余优化器。DeepSpeed 的核心技术。
> 📌 **FSDP** = **F**ully **S**harded **D**ata **P**arallel，完全分片数据并行。PyTorch 官方的等价实现。

---

## 5.1 那个朴素的问题

**回忆 [04 章](04-数据并行DP.md)：N 张卡做数据并行，每张卡上的模型状态是完全一样的。**

```
    卡0：参数 P、梯度 G、优化器状态 O
    卡1：参数 P、梯度 G、优化器状态 O     ← 和卡0 一模一样
    卡2：参数 P、梯度 G、优化器状态 O     ← 一模一样
    卡3：参数 P、梯度 G、优化器状态 O     ← 一模一样

    ★★ 那为什么要存四份？★★
```

**ZeRO 的回答**：**不存四份。每张卡只存 1/4，需要用完整的时候再临时凑。**

> ✅ ZeRO 论文（[papers/ZeRO-...-arXiv-1910.02054.pdf](../papers/ZeRO-Memory-Optimizations-Trillion-Parameter-arXiv-1910.02054.pdf)）原文：
> *"ZeRO-DP removes the memory state redundancies across data-parallel processes by partitioning the model states instead of replicating them, and it retains the compute/communication efficiency by retaining the computational granularity and communication volume of DP using a dynamic communication schedule during training."*

> 🔑 **一句话**：**把"复制"换成"切分"，缺的部分用通信临时补上。**

---

## 5.2 三个阶段

**回忆 [02 章](02-显存去哪了.md)那 16 字节：参数 2 + 梯度 2 + 优化器 12。ZeRO 分三步把它们切掉。**

```
┌──────────────────────────────────────────────────────────────────────┐
│  基线（普通 DP）：每卡 16 Ψ 字节                                       │
│  ████████████████  参数2 ▏梯度2 ▏───────── 优化器 12 ─────────       │
├──────────────────────────────────────────────────────────────────────┤
│  ZeRO-1（Pos）：只切【优化器状态】                                     │
│  █████▏░░░░░░░░░░   参数2 ▏梯度2 ▏优化器 12/N                        │
│  → 每卡 4 + 12/N 字节        ★ N=64 时 ≈ 4.2 字节，省了 3.8 倍       │
├──────────────────────────────────────────────────────────────────────┤
│  ZeRO-2（Pos+g）：再切【梯度】                                         │
│  ███▏░░░░░░░░░░░░   参数2 ▏梯度 2/N ▏优化器 12/N                     │
│  → 每卡 2 + 14/N 字节        ★ N=64 时 ≈ 2.2 字节，省了 7.3 倍       │
├──────────────────────────────────────────────────────────────────────┤
│  ZeRO-3（Pos+g+p）：连【参数】也切                                     │
│  ▏░░░░░░░░░░░░░░░   全部 16/N                                        │
│  → 每卡 16/N 字节            ★★ N=64 时 = 0.25 字节，省了 64 倍 ★★   │
└──────────────────────────────────────────────────────────────────────┘
```

> ✅ **论文原文，逐字对上**：
> *"When enabled cumulatively: 1) Optimizer State Partitioning (Pos): 4x memory reduction, same communication volume as DP; 2) Add Gradient Partitioning (Pos+g): 8x memory reduction, same communication volume as DP; 3) Add Parameter Partitioning (Pos+g+p): Memory reduction is linear with DP degree Nd. For example, splitting across 64 GPUs (Nd = 64) will yield a 64x memory reduction."*

```
    ★ 论文说的 "4x / 8x"，就是我上面算的 16→4 和 16→2（N 很大时）。
      两个数完全对得上。
```

> ✅ 论文的结论句：*"With all three stages enabled, ZeRO can train a trillion-parameter model on just 1024 NVIDIA GPUs."*
> ```
> ★ 验算：1T 参数 × 16 字节 = 16 TB ÷ 1024 张 = 15.6 GB/卡
>   ✅ 论文自己也算了这笔账：
>      "A trillion-parameter model with an optimizer like Adam in 16-bit precision
>       requires approximately 16 terabytes (TB) of memory... 16TB divided by
>       1024 is 16GB, which is well within a reasonable bound for a GPU"
> ```

---

## 5.3 ★ 关键问题：切了之后怎么算

**这是全章最需要想明白的地方。参数都切碎了，前向传播怎么做？**

### ZeRO-1、ZeRO-2：参数没切，正常算

```
    参数完整地在每张卡上  →  前向、反向都和普通 DP 一模一样  ✅

    只有【最后一步】不同：

      普通 DP：  All-Reduce 梯度（每卡拿到完整梯度）→ 每卡都做完整的更新
      ZeRO-2：   Reduce-Scatter 梯度（每卡只拿 1/N 段）→ 每卡只更新自己那 1/N 段
                 → 再 All-Gather 更新后的参数，让每卡重新拥有完整参数

    ★★ 回忆 03 章那个恒等式：All-Reduce = Reduce-Scatter + All-Gather
       → ZeRO-2 只是把这一个 all-reduce【拆成了两半】，中间插了一次更新
       → ★ 通信总量【完全没变】，还是 2Ψ
```

> 🔑 **★★ 这就是为什么 ZeRO-1 / ZeRO-2 是"白赚"：省 4 倍 / 8 倍显存，通信量一分钱不多花。**
> ✅ 论文原文两次强调 *"same communication volume as DP"*。
>
> **实践建议：ZeRO-1 和 ZeRO-2 应该是默认开启的，没有理由不开。**


#### ★★ 追问：凭什么"每卡只更新 1/N"就够了？

这句话听起来太便宜了——**优化器凭什么可以只看一段参数就把它更新对？**
答案藏在 Adam 的更新公式里，值得单独拆开看：

```
    Adam 对【第 i 个参数】做的事：

      m[i] ← β₁·m[i] + (1−β₁)·g[i]           动量
      v[i] ← β₂·v[i] + (1−β₂)·g[i]²          二阶动量
      θ[i] ← θ[i] − lr · m̂[i] / (√v̂[i] + ε)   更新

    ★★ 通读一遍，注意一件事：等号右边【只出现下标 i】。
       没有 g[j]、没有 sum、没有任何跨参数的项。
```

> 🔑 **这就是 ZeRO 能成立的全部数学基础：优化器更新是【逐元素】的（element-wise）。**
>
> 名词解释：**逐元素**——每个参数的新值只依赖它自己的旧值和它自己的梯度，
> 参数之间互不相干。
> ⟹ 所以把参数切成 N 段、每卡负责一段，**算出来的结果和单卡完全一致**，
>    不是近似，是**逐比特相同**（⚠️ 忽略浮点归约顺序带来的差异）。

**还有一个更容易被忽略的时序问题**：

```
    ⚠️ 常见误解：「动量是历史信息，切开会不会丢上下文？」

    ★ 不会。看清楚 m 的更新时机：
       m 是在【梯度算完之后】才更新的，
       而梯度此刻已经被 Reduce-Scatter 归约成【全局正确】的值了。

    ⟹ 每张卡拿到的 g[i]，和单卡训练时的 g[i] 是同一个数
    ⟹ 于是它算出的 m[i]、v[i] 也和单卡时相同
    ⟹ ★★ 动量根本不需要"跨卡拼接"，它只是逐元素地跟着梯度走

    ⟹ 🔑 所以 ZeRO-1 才能做到【零新增通信】：
       优化器状态被切开这件事，通信层【完全感知不到】。
       它切的是"谁来算"，不是"要传什么"。
```

> ⚠️ **反例提醒**：如果优化器不是逐元素的，这套就不成立。
> 比如需要看整层参数矩阵的二阶方法（K-FAC、Shampoo 这类），
> 切开之后就必须额外通信才能算对。
> ★ ZeRO 的"白赚"是**建立在 Adam/SGD 这类逐元素优化器之上的**，不是普适定理。

### ZeRO-3：参数切了，需要临时拼

```
    前向传播，算到第 k 层时：

        ① All-Gather：把第 k 层的参数从 N 张卡上凑齐    ← 临时通信
        ② 用完整的第 k 层参数做计算
        ③ ★ 立刻把不属于自己的那部分【扔掉】             ← 释放显存
        ④ 走到第 k+1 层，重复

    反向传播时同理，再 All-Gather 一次。

    ★ 任何时刻，显存里只有【一层】的完整参数，其余全是碎片。
```

> ✅ FSDP 论文（[papers/PyTorch-FSDP-arXiv-2304.11277.pdf](../papers/PyTorch-FSDP-arXiv-2304.11277.pdf)）把这件事说得很清楚：
> *"This approach ensures that FSDP only needs to materialize parameters from one unit at a time, which significantly reduces peak memory consumption."*

**通信量账**：

```
    前向 All-Gather 参数：       Ψ
    反向 All-Gather 参数：       Ψ    ← ★ 多出来的就是这一次
    反向 Reduce-Scatter 梯度：   Ψ
    ────────────────────────────────
    合计  3Ψ    vs  普通 DP 的 2Ψ

    ★ 贵 1.5 倍的通信，换 N 倍的显存。

#### ★★ 这笔账很容易算错：为什么是 3Ψ，不是 4Ψ

**一个几乎人人都会踩的坑**：

```
    ✗ 错误的算法：
       「普通 DP 是 2Ψ，ZeRO-3 又加了前向和反向两次 all-gather，
         所以是 2Ψ + 2Ψ = 4Ψ」

    ⟹ ★ 错在哪？错在【以为只是往上加】。
       ZeRO-3 不只是加，它还【删掉】了一样东西。
```

**正确的算法是「一删两增」**：

```
    起点：普通 DP = 2Ψ
          （回忆 03 章：All-Reduce = Reduce-Scatter Ψ + All-Gather Ψ）

    ┌──────────────────────────────────────────────────────────┐
    │  ★ 一删：  −Ψ   删掉梯度 All-Reduce 里的 All-Gather 那半  │
    │  ★ 两增：  +Ψ   前向 All-Gather 参数                      │
    │            +Ψ   反向 All-Gather 参数                      │
    └──────────────────────────────────────────────────────────┘
                    2Ψ − Ψ + Ψ + Ψ = 3Ψ   ✅
```

> 🔑 **★★★ 那个 −Ψ 是全章最容易漏掉的一项，也是最能说明问题的一项。**
>
> **为什么可以删？** 因为 ZeRO-3 的参数**本来就是切片状态**：
> ```
>     ZeRO-2：每卡更新完自己那 1/N 段参数
>             ⟹ ★ 必须 All-Gather 一次，把完整参数拼回每张卡
>                （因为下一轮前向需要完整参数）
>
>     ZeRO-3：每卡更新完自己那 1/N 段参数
>             ⟹ ★★ 就这样放着，不拼！
>                （因为下一轮前向【本来就要逐层临时 all-gather】）
> ```
> ⟹ **ZeRO-3 把"更新后拼回参数"这一次通信，合并进了"前向逐层拼参数"里。**
> ⟹ ★ 所以两次前/反向 all-gather 里，**有一次是"白拿"的**——
>    它顶替了原本就要付的那一次。
>
> **这也是为什么 ZeRO-3 的代价是 1.5 倍而不是 2 倍。**

⚠️ **但账面便宜不等于实际便宜**（这点比数字更重要）：

```
    ★ 3Ψ 和 2Ψ 的差距是【总量】上的 1.5 倍，
      但这两种通信的【可隐藏性】完全不同：

      普通 DP 的 2Ψ：  梯度通信 ⟹ ★ 可以和反向计算完全重叠
                       （算完第 k 层的梯度就开始传，同时算第 k−1 层）
                       ⟹ 理想情况下【几乎不占用额外时间】

      ZeRO-3 的前向 Ψ：★★ 在【关键路径】上
                       ⟹ 第 k 层的参数没拼完，第 k 层就【不能开始算】
                       ⟹ 只能靠"提前一层预取"来藏（prefetch），
                          藏得住藏不住取决于【单层计算时间 vs 单层通信时间】

    ⟹ ⚠️（我的判断）所以实践中 ZeRO-3 的减速常常【超过】1.5 倍这个账面值，
       尤其是小模型 / 慢网络 —— 因为单层计算太快，来不及掩盖 all-gather。
    ⟹ 🔑 反过来说，模型越大、层越厚、网络越快，ZeRO-3 越划算。

```

> 🔑 **所以 ZeRO-3 的取舍很清楚**：
> ```
> ✅ 用它：模型状态实在装不下，或者你只有单一维度的并行可用
> ⚠️ 慎用：通信量涨 1.5 倍，而且 all-gather 在【前向的关键路径】上
>          （不像 DP 的梯度通信可以完全藏在反向计算里）
> ```


### ★★ 把三个阶段的 All-Gather 摆在一条时间轴上

前面分开讲了三个阶段，但**最容易混淆的是"那次 all-gather 到底发生在什么时候"**。
把它们并排放，一眼就清楚了：

| | 梯度归约用什么 | **参数 All-Gather 在哪** | 次数 | 能否和计算重叠 |
|---|---|---|---|---|
| **普通 DP** | All-Reduce (2Ψ) | ★ 不需要（参数从没被切） | 0 | — |
| **ZeRO-1** | All-Reduce (2Ψ) | ★ 不需要 | 0 | — |
| **ZeRO-2** | Reduce-Scatter (Ψ) | ★ **优化器 step 的最后**，一次性拼回全部参数 | 1 次/步 | ⚠️ 难（在步与步之间） |
| **ZeRO-3** | Reduce-Scatter (Ψ) | ★★ **前向每一层之前 + 反向每一层之前** | 2×L 次/步 | ✅ 可预取，但在关键路径 |

> ★★ **读这张表最该注意的是"次数"那一列**：
> ```
>     ZeRO-2：1 次大的  ⟹ 一次传 Ψ，★ 带宽利用率高，但【完全无法重叠】
>                          （此刻前向还没开始，没有计算可以拿来掩盖）
>     ZeRO-3：2L 次小的 ⟹ 每次传 Ψ/L，★ 可以和前一层的计算重叠
>                          ⚠️ 但小消息多 ⟹ 通信【延迟】开销被放大 L 倍
> ```
> 🔑 **同样是 Ψ 的通信量，切成多少块、在什么时候发，性能可以差好几倍。**
> ⟹ ★ 这正是 FSDP 里 `reshard_after_forward`、DeepSpeed 里
>    `stage3_prefetch_bucket_size` 这类参数在调的东西。

**✅ 源码实证：这两个时间点在代码里长这样**

（DeepSpeed `v0.19.5`，见 [源码/DeepSpeed/](../源码/DeepSpeed/)）

```python
# ── ZeRO-2：all-gather 在 step() 的末尾 ──────────────────────
# ✅ deepspeed/runtime/zero/stage_1_and_2.py:2367-2370
#    （所在函数 step() 从 L2253 开始）
self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
# Gather the updated weights from everyone.
# Then all partitions of the model parameters are updated and ready for next round forward.
all_gather_dp_groups(groups_flat=self.bit16_groups_flat, ...)
```

```
    ★ 注意源码那句注释：
      "ready for next round forward"（为下一轮前向做好准备）
    ⟹ ★★ 一句话点明了 ZeRO-2 和 ZeRO-3 的分界：
       ZeRO-2 必须【提前】把参数备齐，ZeRO-3 选择【用到再说】。
```

```python
# ── ZeRO-3：all-gather 挂在每个子模块的前后 ──────────────────
# ✅ deepspeed/runtime/zero/parameter_offload.py
def pre_sub_module_forward_function(self, sub_module):    # L525  ★ 进这层前：拼
    ...
def post_sub_module_forward_function(self, sub_module):   # L549  ★ 出这层后：放
    ...
def pre_sub_module_backward_function(self, sub_module):   # L567  ★ 反向前：再拼一次
```

> 🔑 **对照 §5.5 讲的 hook 机制**：
> 这三个函数就是被 `register_forward_pre_hook` / `register_forward_hook`
> 挂上去的（`parameter_offload.py:353/357`）。
> ⟹ ★★ **"每层临时拼回来"这句话，在源码里就是这三个函数。**

---

---

## 5.4 ZeRO-3 和 FSDP 是一回事吗

**思想上：是。工程上：不是同一份代码。**

| | **ZeRO-3** | **FSDP** |
|---|---|---|
| 出处 | DeepSpeed（微软），2019 | PyTorch 官方，2023 |
| 论文 | [arXiv:1910.02054](../papers/ZeRO-Memory-Optimizations-Trillion-Parameter-arXiv-1910.02054.pdf) | [arXiv:2304.11277](../papers/PyTorch-FSDP-arXiv-2304.11277.pdf) |
| 切分单位 | 按参数张量切 | 按 **FlatParameter**（一组模块的参数拍平成一个大张量）切 |
| 集成度 | 外挂库，需要 `deepspeed.initialize()` | ★ **PyTorch 原生**，和 autograd / 显存分配器深度耦合 |
| 现状 | 仍广泛用于 RLHF 等场景 | ★ 新项目的默认选择；FSDP2 基于 `DTensor` 重写 |

> ✅ FSDP 论文强调它的差异化在于"原生集成"：
> *"FSDP has been closely co-designed with several key PyTorch core components including Tensor implementation, dispatcher system, and CUDA memory caching allocator, to provide non-intrusive user experiences and high training efficiency."*
>
> ✅ 效果：*"FSDP is capable of achieving comparable performance to Distributed Data Parallel while providing support for significantly larger models with near-linear scalability in terms of TFLOPS."*
> ✅ 实验规模：*"utilizing up to 512 80GB A100 GPUs"*

> 📌 **一个容易踩的坑**：FSDP 论文提到显存碎片问题——
> ✅ *"operating near GPU memory capacity significantly increases the chance to trigger defragmentations"*
> **实践含义**：把显存用到 99% 反而会变慢（CUDA 分配器要整理碎片）。**留 5~10% 余量是常规做法。** ⚠️ 具体留多少论文没给数字，这是我的经验判断。

---

## 5.5 代码长什么样

```python
# ── DeepSpeed ZeRO ──────────────────────────────────
# ds_config.json:
# {"zero_optimization": {"stage": 2},          ← 改这一个数字就换阶段
#  "bf16": {"enabled": true}}
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model, config="ds_config.json")
for batch in loader:
    loss = model(batch)
    model.backward(loss)        # ★ 不是 loss.backward()
    model.step()                # ★ 不是 optimizer.step()


# ── PyTorch FSDP（FSDP2 / DTensor 风格）─────────────
from torch.distributed.fsdp import fully_shard
for layer in model.layers:
    fully_shard(layer)          # ★ 每一层是一个"分片单元"
fully_shard(model)
# 之后就和普通训练一样：loss.backward(); optimizer.step()
```

> 🔑 **"每一层包一次"这个写法很重要**：它决定了 §5.3 里"一次 all-gather 拼多大一块"。
> ```
> 包得太细（每个 Linear 包一次）→ 通信次数太多，延迟开销大
> 包得太粗（整个模型包一次）    → 等于没切，显存峰值又回去了
> ★ 一层 transformer 包一次是标准做法
> ```

---

### ✅ 源码实证：ZeRO-1 和 ZeRO-2 在代码里只差一个布尔值

上面那句"改这一个数字就换阶段"，在源码里比你想的还要字面。
（以下来自本地实测的 DeepSpeed 官方源码包 `v0.19.5`，见 [源码/DeepSpeed/](../源码/DeepSpeed/)）

`deepspeed/runtime/zero/` 目录长这样：

```
stage_1_and_2.py    3,153 行   ← ZeRO-1 和 ZeRO-2 共用这一个文件
stage3.py           3,857 行   ← ZeRO-3 是完全独立的另一套
partition_parameters.py 2,520 行  ← ZeRO-3 专用
```

`stage_1_and_2.py:222-223`：

```python
self.partition_gradients = partition_grads
self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"
```

⇒ ★★ **连"我是几阶段"这个名字，都是这个布尔值现算出来的。**
这正好印证了 §5.2 的说法：ZeRO-2 只是在 ZeRO-1 基础上"顺手把梯度也切了"，
而 ZeRO-3 是换一整套做法（约 7,700 行新代码）—— 因为参数被切之后，
前向传播本身就得改（§5.3 讲的"临时拼回来"）。

★ 这也解释了一个实践现象：**很多团队停在 ZeRO-2**。
不是不知道 ZeRO-3 更省显存，而是它的通信量和踩坑面确实是另一个量级。

**另外，ZeRO-3 是怎么做到"你的模型代码一行都不用改"的？** 两个 PyTorch 原生机制：

```python
# ① 构造期：参数刚创建出来就切碎（partition_parameters.py:1026 官方示例）
with deepspeed.zero.Init():
    model = MyLargeModel()          # ★ 模型从未在任何一张卡上完整存在过

# ② 运行期：用 hook 在每层前后自动"拼回来 / 放掉"
#    （parameter_offload.py:439-442）
module.register_forward_pre_hook(_pre_forward_module_hook)   # 进这层前拼
module.register_forward_hook(_post_forward_module_hook)      # 出这层后放
```

> 名词解释：**hook（钩子）**——
> PyTorch 允许你在某个模块 forward 之前/之后自动插入一段函数，
> 模型作者不知情，也不用配合。
> ⇒ ★ 这就是 DeepSpeed "接入成本极低"的技术来源。

---

## 5.6 两个亲戚：Offload 与 Infinity

**如果卡实在太少，还有两条"用速度换可行性"的路**：

| 论文 | 做什么 | 定位 |
|---|---|---|
| **ZeRO-Offload**（[arXiv:2101.06840](../papers/ZeRO-Offload-arXiv-2101.06840.pdf)） | 把优化器状态和优化器计算**放到 CPU 内存**上 | 单卡/少卡也能训大模型 |
| **ZeRO-Infinity**（[arXiv:2104.07857](../papers/ZeRO-Infinity-arXiv-2104.07857.pdf)） | 再往下一层，**用 NVMe 固态硬盘**当显存 | 极限情况 |

> 🔑 **回忆 [01 章 §1.6](01-为什么需要分布式训练.md)：PCIe 带宽比显存带宽窄 55 倍。**
> **所以这两条路一定是慢的。它们的价值不是"更快"，是"原本跑不了的现在能跑"。**
> ⚠️ **工业界大规模训练不用它们。** 但如果你想在 1~2 张卡上微调一个大模型，它们是标准答案。

---

## 5.7 ⚠️ ZeRO 和 TP 是什么关系

**这是一个常见困惑：ZeRO-3 都省 64 倍了，为什么还要张量并行？**

> ✅ ZeRO 论文自己也问了这个问题：*"ZeRO and MP: Since ZeRO eliminates the memory inefficiency in DP, it is natural to ask: Do we still need MP, and when?"*
>
> ✅ 但 Megatron 第二篇论文（[papers/Megatron-2-...-arXiv-2104.04473.pdf](../papers/Megatron-2-PTD-P-Efficient-Large-Scale-Training-arXiv-2104.04473.pdf)）给了实测答案：
> *"We also compared to ZeRO, and found that our approach outperforms ZeRO-3 by 70% for models with 175 and 530 billion parameters due to less cross-node communication."*

```
    ★ 关键词是 "less cross-node communication"（更少的跨机通信）

    回忆 03 章：跨机带宽比机内 NVLink 慢 18 倍。

    ZeRO-3 的 all-gather 是在【整个数据并行组】上做的
      → 数据并行组通常横跨很多台机器
      → ★ 每一层的参数 all-gather 都要走慢速网络

    TP 的 all-reduce 只在【一台机器内】做（8 张卡）
      → 走 NVLink

    ⚠️ 我的理解：这就是那 70% 差距的主要来源。论文归因于 "cross-node
       communication"，但没有拆解出各部分占比。
```

> 🔑 **实践上的结论（这也是工业界的标准配方）**：
> ```
>   按【通信要走多远】排：
>
>     机器内 8 卡之间：  TP（张量并行）    ← 高频通信，必须走 NVLink
>     若干台机器之间：  PP（流水线并行）  ← 低频通信，能忍受慢网络
>     整个集群铺开：    DP + ZeRO-1       ← ★ 只切优化器状态，白赚，不加通信
>
>   ★ 注意最后一行：大规模训练里 ZeRO 常常【只开到 stage 1】，
>     因为参数和梯度已经被 TP/PP 切掉了，不需要 ZeRO 再切一次。
> ```

> ⚠️ **一个必须提前说清的措辞陷阱**（否则读到 12 章一定会打架）：
>
> 「**最外层**」这个词在两个完全不同的语境里都被用，指的却是**不同的东西**：
>
> | 说法 | 在讲什么 | 谁在"外" |
> |---|---|---|
> | 本章这里 | ★ **通信要走多远**（机内 → 机间 → 全集群） | **DP** 铺得最开 |
> | [12 章 §12.8](12-并行策略组合与MFU.md) | ★ **rank 编号的排布轴**（谁变得最慢） | **PP** 变得最慢 |
>
> ```
>     ⟹ ★★ 这两句话【同时成立】，因为它们量的根本不是同一把尺子：
>
>        「DP 铺得最开」  说的是：DP 组的成员分布在【物理上最远】的机器上
>        「PP 变得最慢」  说的是：rank 号从小到大走一遍，PP 这一维【最后才翻页】
>
>     ⚠️ 别把它们叠在一张图上理解，会自相矛盾。
>     ★ 本章之后统一说「铺得最开 / 走得最远」，
>       把「最外层」这个词【让给 12 章】的 rank 轴。
> ```

---

## 5.8 本章小结

```
① ZeRO = Zero Redundancy Optimizer。想法极朴素：
   ★ 数据并行的 N 张卡存着完全一样的模型状态 —— 把"复制"换成"切分"

② 三个阶段（✅ 论文原文数字）：
   ZeRO-1  切优化器状态   4× 显存节省   通信量【不变】  ★ 白赚
   ZeRO-2  再切梯度       8× 显存节省   通信量【不变】  ★ 白赚
   ZeRO-3  再切参数       N× 显存节省   通信量 2Ψ→3Ψ   ⚠️ 贵 1.5 倍
   ✅ 三阶段全开：1024 张卡可训 1T 参数模型（论文自己算的 16TB÷1024=16GB）

③ ★ ZeRO-1/2 为什么白赚：
   All-Reduce = Reduce-Scatter + All-Gather（03 章那个恒等式）
   → ZeRO-2 只是把这一个 all-reduce 拆成两半，中间插一次局部更新
   → 通信总量一分钱没多花

④ ★ ZeRO-3 怎么工作：算到第 k 层才 all-gather 第 k 层的参数，
   用完立刻扔掉 → 任何时刻显存里只有【一层】的完整参数
   代价：前向 Ψ + 反向 Ψ + 梯度 Ψ = 3Ψ，且在关键路径上

⑤ FSDP ≈ PyTorch 原生版的 ZeRO-3，思想相同、代码不同
   差异在"和 autograd / 显存分配器深度耦合"
   ⚠️ 坑：显存用到 99% 会触发碎片整理反而变慢，留 5~10% 余量

⑥ ZeRO-Offload / ZeRO-Infinity：往 CPU 内存 / NVMe 上放
   ★ 定位是"用速度换可行性"，工业界大规模训练不用

⑦ ★★ 为什么有了 ZeRO-3 还要 TP：
   ✅ Megatron-2 实测：PTD-P 比 ZeRO-3 快 70%（175B 和 530B 模型上）
   ✅ 归因："less cross-node communication"
   → 标准配方：机内 TP + 机间 PP + 铺满集群的 DP/ZeRO-1
```

**下一章讲 Megatron 的立身之本：张量并行。** 这是"把一个矩阵切开"这件事——听起来吓人，其实只是**矩阵分块乘法**。

---

> [← 04 数据并行 DP](04-数据并行DP.md) ｜ [目录](README.md) ｜ 下一章 → [06 张量并行 TP](06-张量并行TP.md)
