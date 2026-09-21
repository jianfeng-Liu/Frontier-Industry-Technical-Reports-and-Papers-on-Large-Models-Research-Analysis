# 13 · Ascend C 编程：把 Triton 的"一块数据"心智换成"三段流水"

> **前置**：[12 · 昇腾硬件基础](12-昇腾硬件基础.md) 必读——本章从第一行代码起就在用 AI Core / UB / MTE 这几个词。[07 · Triton 编程](07-Triton编程.md) 是本章的对照章，两者是同一件事（写一个 kernel）的两种写法。
> **目标**：写出、看懂、算清楚、调优你的第一个昇腾算子。读完之后你应该能回答三个问题：**这一行代码在硬件上让谁动了？分块参数是怎么算出来的？为什么不开双缓冲性能就腰斩？**
> **读法**：本章每个抽象概念后面都跟一个带具体数字的小例子，算术都能自己验算。**§13.4.3（双缓冲的逐时间单位推演）、§13.5.2（从 UB 容量反推 tileLength）、§13.5.3（尾块的 32 字节账）这三处请务必拿纸笔算一遍**——它们分别对应昇腾算子最常见的一个性能 bug 和一个精度 bug。
> **版本与可信度**：Ascend C 的 API 随 CANN 版本变动很大。本章按 **CANN 8.0 / 8.1 系列的官方 Kernel API 文档**校对，凡版本敏感的写法单独标 ⚠️。硬件参数沿用本仓库纪律：**✅ = 华为官方产品页 / CANN 官方文档；⚠️ = 第三方、社区实测或为演示而设的假设值**。凡是要写进方案的数字，请按 [12 章 §12.4.3](12-昇腾硬件基础.md) 在自己的卡上实测。

---

**本章关键词**

**CANN**（Compute Architecture for Neural Networks，神经网络计算架构）—— 昇腾软件栈总称，对标 CUDA 全家桶。
**Ascend C** —— 基于 C++ 的昇腾算子编程语言，对标 CUDA C++。注意它**不是** C 语言，是一套 C++ 模板库。
**AscendCL**（Ascend Computing Language，昇腾计算语言）—— 昇腾的运行时 API，对标 CUDA Runtime API，函数名都以 `acl` 开头。
**aclnn** —— CANN 提供的单算子 API 层（acl neural network），PyTorch 的每个算子最终落到这里。
**AI Core** —— 昇腾的基本计算单元，地位相当于 NVIDIA 的 SM（Streaming Multiprocessor，流式多处理器）。
**AIC / AIV**（AI Cube / AI Vector，矩阵核 / 向量核）—— 910B 起 Cube 和 Vector 拆成两种独立的核，比例 1 : 2（✅ CANN 文档）。
**GM**（Global Memory，全局内存）—— 昇腾对片外主存（HBM，高带宽显存）的称呼。
**UB**（Unified Buffer，统一缓冲区）—— Vector 单元唯一的片上工作区，相当于 GPU 共享内存的一部分角色。⚠️ 注意昇腾体系里 UB 这个缩写撞车了，CloudMatrix 超节点的 Unified **Bus**（统一总线）也叫 UB。
**MTE**（Memory Transfer Engine，存储转换引擎）—— 专门负责各级存储之间搬运和格式转换的**独立硬件流水线**。MTE2 管搬入（GM→UB），MTE3 管搬出（UB→GM）。
**TPipe**（Task Pipe，片上内存与同步事件管理器）—— 一个算子里通常只有一个，所有片上内存都从它手里申请。⚠️ T 系列这几个类的英文全称官方文档没有正式展开，这里给的是社区通行读法。
**TQue**（Task Queue，任务队列）—— 连接流水线各阶段的队列，**既传数据也做同步**，双缓冲就靠它实现。
**TBuf**（Temporary Buffer，临时缓冲区）—— 不进队列的临时片上内存，用于中间变量。
**TPosition**（逻辑位置）—— `VECIN` / `VECOUT` / `VECCALC` / `A1` / `B1` 等标签，抽象掉物理存储（UB / L1 / L0）的位置名。
**LocalTensor / GlobalTensor** —— 片上张量 / 片外（GM）张量，Ascend C 里最基本的两种数据句柄。
**DataCopy / DataCopyPad** —— 搬运 API。`DataCopy` 要求 **32 字节对齐**，`DataCopyPad` 是它的非对齐版本，专门用来处理尾块。
**Tiling**（分块）—— 把大张量切成片上放得下的小块，并把切分参数从 Host（主机 CPU）传给 Device（NPU）的整套机制。
**blockDim** —— 昇腾语境下指**参与计算的核数（或 AIC+AIV 组合数）**，**不是** CUDA 里的"每个 block 多少线程"。这个词撞名了，极易误解。
**Double Buffer**（双缓冲）—— 把片上缓冲一分为二，让搬运和计算重叠。在昇腾上这不是优化技巧，是基本写法。
**ISASI**（Instruction Set Architecture Specific Interface，体系结构相关接口）—— CANN 文档给部分 API 打的标签，意思是"这个接口跟具体芯片指令集绑定，**不保证跨硬件版本兼容**"。看到这个标签就要查自己的型号。
**soc_version** —— 芯片型号标识（如 `Ascend910B3`）。算子是按它编译的，编出来的包不能跨型号用。
**msprof / MindStudio Insight** —— 昇腾的性能采集与可视化工具，对标 Nsight Systems / Nsight Compute。
**CPU 孪生调试** —— 同一份 kernel 代码在 x86 CPU 上以单核串行方式跑起来，可以 gdb 单步、可以 printf。这是昇腾相对 CUDA 的一个真实优势。

---

## 13.1 CANN 软件栈：先找到自己站在哪一层

从 GPU 过来的人第一件事是建立栈的对应关系：

```
   NVIDIA                              华为昇腾
─────────────────────────────────────────────────────────────
   PyTorch                             PyTorch + torch_npu
      │                                      │
   torch.compile / Inductor            TorchAir（GE 图模式）
      │                                      │
   cuDNN / cuBLAS / CUTLASS            aclnn 算子库 / cann-ops-adv
      │                                      │
   CUDA Runtime API                    AscendCL（acl*）
      │                                      │
   Triton                              triton-ascend（⚠️ 新，见 §13.9.4）
      │                                      │
   CUDA C++                            ★ Ascend C ★  ← 本章在这一层
      │                                      │
   PTX / SASS                          昇腾指令集
      │                                      │
   NCCL                                HCCL
   Nsight Compute / Systems            msprof / MindStudio Insight
```

层级对应大致齐整，有两个例外要注意：

1. **Triton 那一层曾经是空的，现在不是了。** `triton-lang/triton-ascend` 已开源，项目文档口径是"支持 85% 以上的 Triton Python API"（⚠️ 该比例出自项目自述，未见独立复现）。但成熟度还在快速变动，**昇腾上的生产算子目前仍以 Ascend C 为主**。详见 §13.9.4。
2. **`cann-ops-adv` 是开源的**。这是华为的高阶融合算子库（adv = advanced），FlashAttention 类算子都在里面，可以直接读源码——这一点比读 cuDNN 闭源二进制友好得多。

**Ascend C 在"抽象层级"这根尺子上的位置**，是理解本章所有代码膨胀的前提：

```
更抽象（省事，但控制不了）
  │
  │  PyTorch 算子        —— 一行 torch.add()，什么都不用想
  │  Triton              —— 你管"切多大块"，编译器管搬运 / 片上内存 / 流水
  │  ★ Ascend C ★       —— 你管"切多大块" + 搬运 + 片上内存 + 流水同步
  │  CUDA C++            —— 你管上面全部 + 每个线程拿哪个元素
  │  PTX / 昇腾指令集     —— 你管每一条指令
  │
更具体（啥都能控制，但啥都要写）
```

> ★ **记住这个位置**：Ascend C **像 Triton 一样以"块"为单位思考**（没有线程、没有 warp、没有 `threadIdx`），但**像 CUDA 一样要你亲手写搬运和流水**。本章后面所有多出来的代码，都在填 Triton 那一行"编译器管"变成"你管"之后留下的坑。

---

## 13.2 三段式：先看数据在硬件上怎么流

### 13.2.1 一个小块的完整旅程

先不看代码，只看一个数据块（tile）从进来到出去走了哪条路。任务就是最简单的 `z = x + y`。

```
GM（Global Memory，片外 HBM）
 │   x[...]        y[...]
 │
 │   ① CopyIn     执行者 = MTE2（搬入引擎）
 │      把 x 的一小段、y 的一小段，从 GM 搬进 UB
 ▼
UB（Unified Buffer，片上，Vector 单元唯一的工作区）
 │   xLocal[...]   yLocal[...]   zLocal[...]
 │
 │   ② Compute    执行者 = Vector（向量计算单元）
 │      zLocal = xLocal + yLocal，全程不出 UB
 │
 │   ③ CopyOut    执行者 = MTE3（搬出引擎）
 │      把 zLocal 从 UB 搬回 GM
 ▼
GM
     z[...]
```

**这三段就是 Ascend C 的全部骨架。** 不管算子多复杂——RMSNorm、Softmax、FlashAttention——结构都是这三段，只是 Compute 段里的内容变多。

> ★ CANN 官方把这个叫**矢量编程范式**（✅ 官方术语）。涉及矩阵乘时是**矩阵编程范式**，多两段：`CopyIn → Split → Compute → Aggregate → CopyOut`，多出来的两段是"把 L1 里的数据切给 L0A/L0B"和"把 L0C 里的结果汇总出来"（[12 章 §12.3.1](12-昇腾硬件基础.md) 那条 Cube 数据流）。本章只讲矢量范式，它是打底的。

### 13.2.2 为什么非要分成三段：三段跑在三条不同的流水线上

如果三段只是"先搬、再算、再搬"，那分段没有任何意义——一个函数从头写到尾就行了。分段的全部理由是：**MTE2、Vector、MTE3 是三套独立的硬件，能真正同时工作**（[12 章 §12.2.1](12-昇腾硬件基础.md)）。

```
一个 AI Core 内部，同一时刻可以是这样的：

  MTE2   正在把第 3 块从 GM 搬进 UB
  Vector 正在算第 2 块
  MTE3   正在把第 1 块从 UB 搬回 GM
         └─ 三件事互不等待，因为用的是三套不同的硬件
```

而如果你把三段写成一个函数、一块数据走完全程再处理下一块，那就是这样：

```
  MTE2   搬第 1 块 ████        空转            空转        搬第 2 块 ████
  Vector    空转         算第 1 块 ██          空转           空转
  MTE3      空转            空转         写第 1 块 ██         空转
         └─ 任意时刻只有 1/3 的硬件在干活
```

> ★ **这就是三段式存在的唯一理由**：把代码分成三段，是为了让**硬件的三条流水线各自有活干**。分段只是必要条件，真正让它们重叠起来的是 TQue 和双缓冲（§13.4）。

**和 GPU 对照一下，这件事的地位完全不同**：

| | GPU（CUDA / Triton） | 昇腾（Ascend C） |
|---|---|---|
| 搬运和计算能否重叠 | 能（`cp.async` 异步拷贝、TMA） | 能（MTE 是独立硬件） |
| 谁来安排重叠 | 编译器（Triton 的 `num_stages`）或库 | **你，手写** |
| 不安排会怎样 | 慢一些，但仍有 warp 轮转兜底（[01 章 §1.6](01-GPU硬件基础.md) 的 Occupancy） | **没有兜底**，昇腾没有 occupancy 这回事，硬件直接空转 |

> ★ 这一行是 GPU 工程师最容易低估的差别。GPU 上就算你写得很笨，硬件还有"64 个 warp 轮着上"这一层自动的延迟掩盖；**昇腾上延迟掩盖只有"手工排流水"这一个办法**，你不排，就没人排。

### 13.2.3 TPipe / TQue / TBuf：谁管什么

这三个类最容易混。一张表定死分工：

| | 中文 | 管什么 | 怎么初始化 | 怎么取数据 | 带同步吗 |
|---|---|---|---|---|---|
| **TPipe** | 片上内存管理器 | **整块 UB 的所有权**。所有片上内存都从它手里切出来 | 声明一个成员变量就行 | 不直接取 | 它管事件，但你不直接用 |
| **TQue** | 任务队列 | 阶段**之间**传递的数据（CopyIn 给 Compute 的、Compute 给 CopyOut 的） | `pipe.InitBuffer(que, BUFFER_NUM, 字节数)` | `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` | **带**，`DeQue` 会阻塞等上一级 |
| **TBuf** | 临时缓冲区 | 阶段**内部**的中间变量（比如求平方和时的临时张量） | `pipe.InitBuffer(buf, 字节数)` | `buf.Get<T>()` | **不带**，同一段代码里直接用 |

注意 `InitBuffer` 的两种重载——**这是区分 TQue 和 TBuf 最直观的标志**：

```cpp
pipe.InitBuffer(inQueueX, BUFFER_NUM, 256);   // TQue：三个参数，中间那个是缓冲份数
pipe.InitBuffer(tmpBuf, 256);                 // TBuf：两个参数，没有"份数"的概念
```

为什么 TBuf 不需要"份数"？因为它不跨阶段——它的生命周期就在 Compute 函数里面，没有"上一级还没放进来"这种事，所以不需要队列，也不需要同步，也就不需要双缓冲。

**把 UB 画出来**，以 §13.3 那个例子为例（每块 tile 是 128 个 `half`，即 256 字节，三个张量各开双缓冲）：

```
UB（假设 192 KB ⚠️，本例只用到 1536 字节，用满还差得远）

  TPipe 从 UB 里切出六块，分给三个 TQue：

  inQueueX   ├── 缓冲 0   256 B      ← MTE2 正在往这里搬第 k+1 块
             └── 缓冲 1   256 B      ← Vector 正在读这里的第 k 块
  inQueueY   ├── 缓冲 0   256 B
             └── 缓冲 1   256 B
  outQueueZ  ├── 缓冲 0   256 B      ← Vector 正在往这里写第 k 块
             └── 缓冲 1   256 B      ← MTE3 正在把第 k-1 块搬出去

  合计 3 个队列 × 2 份 × 256 B = 1536 B
  ★ 一份 tile 的 UB 开销 = 张量个数 × BUFFER_NUM × tileLength × 元素字节数
    这个式子就是 §13.5.2 反推分块大小的全部依据
```

> ★ **"UB 一分为二"的真正含义**在这张图上：不是把整块 UB 劈成两半，而是**每一个队列各自持有两份缓冲，MTE2 写其中一份的同时 Vector 读另一份**。这也解释了为什么开双缓冲要把 `tileLength` 减半——总 UB 容量没变，份数翻倍，每份就得减半。

### 13.2.4 TPosition：为什么不直接说"放到 UB"

`TQue` 的模板参数第一位是 **TPosition**（逻辑位置）。你不写"放到 UB"，而是写"这是 VECIN（Vector 计算的输入）"，编译器把逻辑位置映射到物理存储：

| TPosition | 含义 | 实际落在 |
|---|---|---|
| `VECIN` | 矢量计算的输入 | UB |
| `VECOUT` | 矢量计算的输出 | UB |
| `VECCALC` | 矢量计算的临时变量（TBuf 常用这个） | UB |
| `A1` / `B1` | 矩阵计算的左 / 右矩阵，第一级 | L1 Buffer |
| `A2` / `B2` | 矩阵计算的左 / 右矩阵，第二级 | L0A / L0B |
| `CO1` / `CO2` | 矩阵计算的结果 | L0C / GM |

好处是**代码不写死物理存储**。[12 章 §12.2.3](12-昇腾硬件基础.md) 提到下一代昇腾 950 据称要改 L1 的共享策略（⚠️ 推断），到时候 `A1` 映射到哪里由编译器改，你的代码不用动。

> ⚠️ 逻辑位置不等于"随便写哪个都行"。`VECIN` 和 `VECOUT` 虽然都落在 UB 上，但**搬运方向的合法性是按逻辑位置校验的**——`DataCopy` 从 GM 到 `VECOUT` 不是所有版本都允许。按语义写：搬进来的用 `VECIN`，要搬出去的用 `VECOUT`，纯中间量用 `VECCALC`。

---

## 13.3 第一个算子：逐元素加，每一行都说清楚让谁动了

### 13.3.1 先把数字定下来

代码里的每个变量都会有具体值，这样你能对着算：

```
任务：z[i] = x[i] + y[i]，数据类型 half（FP16，2 字节）

  totalLength  = 8192        总元素数
  coreNum      = 8           用 8 个核（真机 910B3 是 40 个 AIV ⚠️，
                             这里取 8 只为了手算方便）
  BUFFER_NUM   = 2           双缓冲

  ↓ 推出来的

  blockLength  = 8192 / 8 = 1024      每个核负责 1024 个元素
  tileNum      = 4                    （官方样例的参数，见下面的 ⚠️）
  tileLength   = 1024 / 4 / 2 = 128   每次搬 128 个元素 = 256 字节
  loopCount    = 4 × 2 = 8            每个核循环 8 次，处理 8 个 tile

  校验：8 个 tile × 128 个元素 = 1024 ✓ 正好是本核的份额
  校验：256 字节 ÷ 32 = 8，是 32 字节的整数倍 ✓（DataCopy 的对齐要求）
```

> ⚠️ **官方样例里 `tileNum` 这个名字很坑**：它不是"一共几块"，而是"块数 ÷ BUFFER_NUM"。真正的块数是 `tileNum × BUFFER_NUM`。第一次读官方 `add_custom` 样例时几乎所有人都在这里卡过。本章沿用官方写法以便你对照样例，但心里要记住：**块数是 `loopCount`，不是 `tileNum`**。

### 13.3.2 完整代码

```cpp
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;   // ← 双缓冲。这一行是性能的一半，理由见 §13.4

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // ── ① 本核负责哪一段？总长度按核数均分
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum     = tileNum;
        this->tileLength  = this->blockLength / tileNum / BUFFER_NUM;

        // ── ② 把 GM 上属于本核的那一段挂成 GlobalTensor
        //    GetBlockIdx() 返回本核编号，相当于 CUDA 的 blockIdx.x
        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ half*)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + offset, this->blockLength);

        // ── ③ 向 TPipe 申请片上内存。第二个参数 = BUFFER_NUM，就是在开双缓冲
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;    // = 8
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);      // 这三个调用【看起来】是串行的，
            Compute(i);     // 但因为 TQue 的存在，硬件上它们重叠执行
            CopyOut(i);     // 为什么能重叠，见 §13.4
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);     // 通知 Compute：这块数据我搬完了，可以取
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();   // 阻塞，等 CopyIn
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);       // Vector 单元干活

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);   // 必须还！否则双缓冲的另一半拿不到内存
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength, tileNum, tileLength;
};

// 核函数入口。__global__ __aicore__ 的组合相当于 CUDA 的 __global__
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);    // 解析 Host 侧传来的切分参数
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
```

### 13.3.3 逐行说明：这一行在硬件上让谁动了

这是本章最该慢慢读的一张表。**左边是你写的代码，右边是硬件上真实发生的事。**

| 代码 | 在硬件上让谁动了 | 备注 |
|---|---|---|
| `GetBlockNum()` | **什么都没动**，读一个由 Host 侧 `SetBlockDim` 写入的常量 | 相当于 CUDA 的 `gridDim.x` |
| `GetBlockIdx()` | **什么都没动**，读本核编号 | 相当于 `blockIdx.x`。⚠️ MIX 模式下有坑，见 §13.5.6 |
| `SetGlobalBuffer(ptr, len)` | **什么都没动**。只是记下"GM 上这段地址归我管" | 不产生搬运。就像 C 里给指针赋值 |
| `pipe.InitBuffer(que, 2, 256)` | **Scalar 单元**算地址，从 UB 里划出 2×256 字节记账 | 只在 Init 里执行一次，不在循环里 |
| `inQueueX.AllocTensor<half>()` | **Scalar 单元**从队列里取一个空闲缓冲的句柄。**若两份都没还回来，这里会阻塞** | 这是最容易造成假串行的地方（§13.4.5） |
| `DataCopy(xLocal, xGm[off], 128)` | **MTE2 发起一次搬运**：GM → UB，256 字节。发起之后 Scalar 立刻继续往下跑，不等搬完 | 异步。真正的等待发生在 `DeQue` |
| `inQueueX.EnQue(xLocal)` | **插一个同步事件**："这块的 MTE2 搬运完成后，Compute 可以取" | 硬件层面是流水线之间的 flag |
| `inQueueX.DeQue<half>()` | **阻塞等待**上面那个事件。MTE2 没搬完，Vector 就在这儿等 | 这才是唯一真正会"停下来"的地方 |
| `Add(zLocal, xLocal, yLocal, 128)` | **Vector 单元**干活。128 个 `half` = 256 字节，Vector 单次迭代处理 256 字节，所以这是 **1 次迭代** | 计算全程在 UB 内，一个字节都不出片 |
| `outQueueZ.EnQue(zLocal)` | 插事件："Vector 算完了，MTE3 可以搬" | — |
| `inQueueX.FreeTensor(xLocal)` | **把缓冲还给队列**，让 MTE2 能拿去装下一块 | **漏了这一行，第三块就永远卡住** |
| `DataCopy(zGm[off], zLocal, 128)` | **MTE3 发起一次搬运**：UB → GM | 同样异步 |

> ★ **把这张表的信息压成一句话**：`DataCopy` 是"发起"不是"完成"，`EnQue` 是"插旗"，`DeQue` 是"等旗"，`FreeTensor` 是"还钥匙"。**四个动作里只有 `DeQue` 会真的停下来等**——这正是三段能重叠的原因，也是漏写 `FreeTensor` 会让重叠失效的原因。

> ⚠️ `AscendC::Add` 有三种参数形式（✅ CANN 官方 API 文档）：整个 tensor 的运算符重载（`dst = src0 + src1`）、**tensor 前 n 个元素**（上面用的这种，第四个参数叫 `calCount`）、**tensor 高维切分计算**（传 `mask` + `repeatTimes` + `BinaryRepeatParams`）。第三种才是性能调优时真正要用的，它能控制迭代次数和步长。本章为了讲清骨架只用第二种。

### 13.3.4 把第 0 块的完整时序摊开

上面那张表是"每一行干什么"，这里是"按时间顺序发生了什么"。假设 ⚠️ MTE2 搬一块要 2 个时间单位、Vector 算一块要 1 个、MTE3 搬一块要 1 个（逐元素加要搬进 2 个张量、搬出 1 个，所以 MTE2 最重，这个比例是合理的）：

```
时间 →

单位 1   Scalar: AllocTensor 拿到 xLocal / yLocal 的缓冲 0
         MTE2:   DataCopy 启动，开始搬第 0 块的 x 和 y      ████
         Vector: 卡在 DeQue 上等                            ....

单位 2   MTE2:   继续搬                                     ████
         Vector: 还在等                                     ....
                 ↑ MTE2 搬完，EnQue 的旗子立起来

单位 3   Vector: DeQue 通过，Add 执行                        ██
         MTE2:   已经在搬第 1 块了（用缓冲 1）              ████
                 ↑ 这就是双缓冲：MTE2 不等 Vector

单位 4   MTE3:   DeQue 通过，搬出第 0 块                     ██
         Vector: 在算第 1 块吗？不——第 1 块要到单位 5 才搬完
```

> ★ 注意单位 3：**MTE2 已经在搬第 1 块了，而 Vector 在算第 0 块**。这两件事同时发生，靠的是 `BUFFER_NUM = 2` 提供的第二份缓冲。如果只有一份，MTE2 必须等 Vector 把缓冲腾出来才能开工——那就是 §13.4 要算的那笔账。

### 13.3.5 和 Triton 七行的逐项对照

[07 章 §7.4.1](07-Triton编程.md) 的同一个算子：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)      # 搬运，编译器管
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)  # 搬运，编译器管
```

**七行 Triton 对七十行 Ascend C。** 这个膨胀比是真实的，而且多出来的六十行全在干同一件事：**显式管理搬运和流水同步**。

| 这件事 | Triton | Ascend C | 多出来的代码量 |
|---|---|---|---|
| 我负责哪一段 | `tl.program_id(0)` | `GetBlockIdx()` + 手算 `blockLength` 偏移 | +3 行 |
| 片上内存从哪来 | 编译器分配，你看不见 | `pipe.InitBuffer` × 3 + `AllocTensor` / `FreeTensor` | +9 行 |
| 搬进来 | `tl.load` | `DataCopy` + `EnQue` | +4 行 |
| 搬出去 | `tl.store` | `DataCopy` + `DeQue` + `FreeTensor` | +3 行 |
| 搬运和计算重叠 | `num_stages=2`（一个参数，[07 章 §7.7.3](07-Triton编程.md)） | `BUFFER_NUM=2` + 三段函数 + 队列同步 | +30 行（整个三段式结构） |
| 尾块 / 越界 | `mask=offs < n`（一行） | **没有对应物**，要在 Host 侧 tiling 里算，见 §13.5.3 | +Host 侧一整个函数 |
| 分块参数在哪定 | `BLOCK=1024` 写在启动参数里 | **Host 侧独立的 Tiling 函数**，见 §13.5.5 | +20 行 Host 代码 |

> ★ **Triton 的 `num_stages=2` 和 Ascend C 的 `BUFFER_NUM=2` 是同一个东西的两种形态**——都是双缓冲（[03 章 §3.5](03-GEMM深度剖析.md)）。区别是：Triton 里你给个数字，编译器把循环拆成"预取 + 计算"重叠的形式；Ascend C 里你得自己把代码拆成三段、自己布队列、自己记着还缓冲。**这一个参数的差别，就是六十行代码。**

---

## 13.4 双缓冲：为什么 `BUFFER_NUM = 2` 不是优化而是基本功

### 13.4.1 把假设值定下来

要算这笔账，得给三段各一个耗时。**下面的数字是 ⚠️ 假设值，仅为演示算法**，真机数字要用 msprof 测（§13.8）：

```
⚠️ 假设（仅为演示）：

  T_MTE2   = 2 个时间单位   搬入 x 和 y 两个张量，各 256 B，共 512 B
  T_Vector = 1 个时间单位   128 个 half 的加法，Vector 一次迭代搞定
  T_MTE3   = 1 个时间单位   搬出 z 一个张量，256 B

  本核要处理 N = 8 个 tile（§13.3.1 算出来的 loopCount）

  为什么 MTE2 最重：逐元素加要读 2 份、写 1 份，搬入量是搬出量的 2 倍。
  这个"MTE2 是瓶颈"的形态，是逐元素算子在昇腾上的典型状态。
```

### 13.4.2 单缓冲：逐时间单位推演

`BUFFER_NUM = 1` 时，每个队列只有一份缓冲。MTE2 要往里装第 1 块，必须等 Vector 把第 0 块读完、`FreeTensor` 还回来。于是：

| 时间单位 | MTE2 | Vector | MTE3 | 空转的流水线数 |
|---|---|---|---|---|
| 1–2 | **搬块 0** | 空转（缓冲里还没数据） | 空转 | 2 |
| 3 | 空转（缓冲被 Vector 占着） | **算块 0** | 空转 | 2 |
| 4 | 空转 | 空转 | **写块 0** | 2 |
| 5–6 | **搬块 1** | 空转 | 空转 | 2 |
| 7 | 空转 | **算块 1** | 空转 | 2 |
| 8 | 空转 | 空转 | **写块 1** | 2 |
| … | … | … | … | … |
| 29–30 | **搬块 7** | 空转 | 空转 | 2 |
| 31 | 空转 | **算块 7** | 空转 | 2 |
| 32 | 空转 | 空转 | **写块 7** | 2 |

```
单缓冲总耗时 = N × (T_MTE2 + T_Vector + T_MTE3)
             = 8 × (2 + 1 + 1)
             = 32 个时间单位

各流水线占用率：
  MTE2   忙 8×2 = 16 单位 / 32 = 50%
  Vector 忙 8×1 =  8 单位 / 32 = 25%
  MTE3   忙 8×1 =  8 单位 / 32 = 25%
  ★ 任何时刻都只有一条流水线在动，另外两条纯空转
```

### 13.4.3 双缓冲：同样的推演，错开一格

`BUFFER_NUM = 2` 时，每个队列有两份缓冲。MTE2 装满缓冲 0 之后，立刻可以去装缓冲 1，不用等 Vector。

| 时间单位 | MTE2 | Vector | MTE3 |
|---|---|---|---|
| 1–2 | **搬块 0** → 缓冲 0 | — | — |
| 3 | **搬块 1** → 缓冲 1 | **算块 0**（读缓冲 0，算完 Free） | — |
| 4 | （继续搬块 1） | — | **写块 0** |
| 5 | **搬块 2** → 缓冲 0 | **算块 1** | — |
| 6 | （继续搬块 2） | — | **写块 1** |
| 7 | **搬块 3** → 缓冲 1 | **算块 2** | — |
| 8 | （继续搬块 3） | — | **写块 2** |
| … | 每 2 个单位搬完一块，**一刻不停** | 每 2 个单位算一块 | 每 2 个单位写一块 |
| 15–16 | **搬块 7** | 算块 6（单位 15） | 写块 6（单位 16） |
| 17 | — | **算块 7** | — |
| 18 | — | — | **写块 7** |

```
双缓冲总耗时 = 18 个时间单位

各流水线占用率：
  MTE2   忙 16 单位 / 18 = 89%   ← 几乎满载，它是瓶颈
  Vector 忙  8 单位 / 18 = 44%
  MTE3   忙  8 单位 / 18 = 44%

加速比 = 32 / 18 = 1.78 倍
```

**通用公式**（自己代一遍就记住了）：

```
单缓冲 = N × Σ T_i                              （三段严格串行，逐块相加）
双缓冲 = (Σ T_i − max T_i) + N × max T_i        （只有瓶颈级在累加，其余被藏住）

  代入本例：Σ = 4，max = 2，N = 8
    单缓冲 = 8 × 4         = 32
    双缓冲 = (4 − 2) + 8×2 = 18   ✓ 和表格一致

N 很大时，加速比 → Σ T_i / max T_i = 4 / 2 = 2 倍
```

> ★ **这就是"耗时从 `T搬入 + T计算 + T搬出` 降到 `max(三者)`"的准确含义**——每块的**平均**耗时降到了 max，而不是总耗时。开头结尾那 `(Σ − max)` 个单位是流水线的填充和排空，块数一多就摊薄了。

> ★ **1.78 倍是"白捡"的**：一行 `BUFFER_NUM = 2` 加上三段式结构，没改算法、没改数据格式、没换 API。反过来说，**不写双缓冲等于主动放弃将近一半的性能**。这就是为什么 §13.8 的排查清单里，"确认 `BUFFER_NUM=2` 真的开了"排在第一条。

### 13.4.4 为什么是 2 而不是 3、4

Triton 的 `num_stages` 可以开到 3、4、5（[07 章 §7.7.3](07-Triton编程.md)），昇腾上 `BUFFER_NUM` 为什么基本就停在 2？

```
从 1 到 2：收益巨大
  1 → 2：把"严格串行"变成"完全重叠"，加速比接近 Σ/max（本例 2 倍）

从 2 到 3：收益很小
  2 份缓冲已经足够让瓶颈级一刻不停（见 §13.4.3 的表：MTE2 占用 89%）
  第 3 份只能吸收"抖动"——某一块偶然搬得慢了一点

代价却是线性的：
  UB 占用 = 张量数 × BUFFER_NUM × tileLength × 元素字节数
  BUFFER_NUM 从 2 到 3，同样的 UB 容量下 tileLength 要减到 2/3
  而 tileLength 变小 → 单次搬运变小 → MTE 效率下降（§13.8）
```

> ★ **判断标准**：`BUFFER_NUM = 2` 已经让瓶颈级接近满载时，加到 3 只会因为 `tileLength` 变小而变慢。**先看 msprof 的流水图确认瓶颈级占用率**：85% 以上就别加了，加了也没地方省。

> ⚠️ 上面这套推演是**理想模型**，忽略了三件事：① `EnQue` / `DeQue` 的同步本身有开销；② MTE2 和 MTE3 虽是两条流水线，但可能争抢同一条对 GM 的总线，不完全独立；③ Vector 的实际耗时和数据类型、指令种类强相关。所以**真实加速比会低于 1.78**，具体多少只能实测。这里要你记住的是**量级和机理**，不是那个数。

### 13.4.5 三种把双缓冲写废的写法

双缓冲最难受的地方是：**写错了不报错，代码看起来完全正常，只是慢一半。** 三个高频错法：

**① 忘了 `FreeTensor`，或者还得太晚。**

```cpp
// ✗ 错：算完没还缓冲
__aicore__ inline void Compute(int32_t progress) {
    auto xLocal = inQueueX.DeQue<half>();
    auto zLocal = outQueueZ.AllocTensor<half>();
    AscendC::Add(zLocal, xLocal, xLocal, this->tileLength);
    outQueueZ.EnQue(zLocal);
    // 漏了 inQueueX.FreeTensor(xLocal);
}
```

后果：`inQueueX` 的两份缓冲被占满之后，`CopyIn` 里的 `AllocTensor` 永久阻塞——**这个不是变慢，是直接挂死**。相对好查。真正难查的是"还得太晚"：把 `FreeTensor` 放到 Compute 函数最末尾、放在一大堆计算之后，那这段时间 MTE2 就一直拿不到缓冲。**原则：一个 LocalTensor 最后一次被读之后，立刻 `FreeTensor`。**

**② 三段合成一段写。**

```cpp
// ✗ 错：看起来省事，双缓冲完全失效
__aicore__ inline void ProcessOneTile(int32_t i) {
    auto xLocal = inQueueX.AllocTensor<half>();
    AscendC::DataCopy(xLocal, xGm[i * tileLength], tileLength);
    inQueueX.EnQue(xLocal);
    xLocal = inQueueX.DeQue<half>();          // ← 立刻 DeQue，等于同步等搬完
    AscendC::Add(...);
    // ...
}
```

后果：`EnQue` 之后马上 `DeQue`，同步事件立即被消费，MTE2 根本没机会去搬下一块。**这就退化成 §13.4.2 的单缓冲时间线，耗时 32 而不是 18。** 而且代码"能跑、结果对"，只有测性能才发现。

> ★ **三段式必须真的分成三个函数、在 `Process()` 的循环里依次调用**，靠的就是"`CopyIn(i)` 里 `EnQue` 之后函数就返回了，MTE2 的搬运还在后台跑"这件事。把它们揉成一段，就把异步揉没了。

**③ `tileLength` 没有为双缓冲留出空间。**

```
✗ 错：按 UB 全容量算 tileLength，再开 BUFFER_NUM=2
  → UB 超分配，编译期或运行期报错（好的情况），
    或者把别的队列的内存踩了（坏的情况）

✓ 对：tileLength 的分母里要带 BUFFER_NUM，见 §13.5.2
```

> ★ 三段式和 GPU 侧的对应关系值得单独记一句：[03 章 §3.5](03-GEMM深度剖析.md) 的 GEMM 双缓冲、[04 章 §4.3](04-FlashAttention全系.md) 的 FlashAttention 分块、本章的 `BUFFER_NUM=2`，**思想完全相同：用片上容量换搬运次数，用流水重叠掩盖延迟**。区别只在于 GPU 上编译器和库帮你做了大半，昇腾上全要自己写。

---

## 13.5 Tiling：昇腾算子开发真正的工作量所在

### 13.5.1 为什么 Tiling 在 Host 侧

CUDA 里分块参数是启动参数，一拍脑袋填个 `<<<4, 256>>>` 就完事。Triton 里是 `BLOCK=1024` 加一个 `@triton.autotune`。昇腾不一样：**Tiling 是 Host 侧一个独立的 C++ 函数，产出一个结构体，通过一块内存传给 Device。**

```
Host 侧（CPU）                          Device 侧（NPU）
─────────────────────────────────────────────────────────────
① 拿到输入 shape
      │
② 查平台参数：几个核？UB 多大？
      │   PlatformAscendC::GetCoreNumAiv()
      │   PlatformAscendC::GetCoreMemSize(CoreMemType::UB, &ubSize)
      │
③ 算 tiling：每核多少、每块多少、几块、尾块多长
      │
④ SetBlockDim(coreNum)  ← 决定启动几个核
      │
⑤ tiling.SaveToBuffer(...)  把结构体写进一块内存
      │
      └──────── 这块内存随 kernel 一起下发 ────────►
                                            ⑥ GET_TILING_DATA(tilingData, tiling)
                                                 解析出 totalLength / tileNum / tailLength
                                            ⑦ Init() 按这些参数划 UB、算偏移
```

为什么要这么麻烦？因为**昇腾的分块参数不是一两个数，而是一个几十个字段的结构体**：每一级切多大、循环多少次、尾块多长、要不要走非对齐搬运路径、核间怎么分。这些只能在 Host 侧、拿到真实 shape 之后算，算完一次性告诉 Device。

> ★ **对照 Triton 的 autotune**：Triton 把"选哪组分块参数"交给运行时 benchmark（[07 章 §7.8](07-Triton编程.md)）；Ascend C 把它交给你在 Host 侧手写的一段确定性代码。**Ascend C 没有内置的 autotune**，要扫参只能自己写循环实测。

### 13.5.2 从 UB 容量反推 tileLength：完整推演

这是昇腾算子开发的第一道算术题。**给定 UB 容量和数据类型，反推每次能搬多少元素。**

```
第 1 步：拿到 UB 容量

  ✅ 正确做法（代码里查，不写死）：
     uint64_t ubSize;
     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  ⚠️ 本例假设 UB = 192 KB = 196608 字节（910B 系列的社区常见口径，
     仅为演示算法。真值请用上面这个 API 查，或看 platform_config/*.ini）

第 2 步：数清楚同时要在 UB 上活着的张量

  逐元素加需要三块：xLocal、yLocal、zLocal   → 张量数 = 3
  每块都要开双缓冲                            → BUFFER_NUM = 2
  数据类型 half                               → 元素字节数 = 2

第 3 步：写出 UB 占用式子，反解 tileLength

  UB 占用 = 张量数 × BUFFER_NUM × tileLength × 元素字节数
          = 3 × 2 × tileLength × 2
          = 12 × tileLength   字节

  要求 UB 占用 ≤ 196608
    → tileLength ≤ 196608 / 12 = 16384 个元素

第 4 步：留余量

  ⚠️ 不要按上界取。UB 上还有编译器的临时空间、Scalar 的栈、
     高阶 API 的内部 workspace。社区经验是留 10–20%。

  取 80%：16384 × 0.8 = 13107 个元素

第 5 步：对齐到 32 字节

  DataCopy 要求搬运长度和 UB 上的起始地址都是 32 字节的整数倍（✅ CANN 文档）
  32 字节 = 16 个 half
  13107 ÷ 16 = 819.2 → 向下取整 819 → 819 × 16 = 13104 个元素

  ★ 最终上界：tileLength ≤ 13104
```

**上界不等于最优值。** 实际取多少还要看两件事：

| 考虑 | 倾向 | 理由 |
|---|---|---|
| MTE 搬运效率 | `tileLength` **大**一点好 | 单次搬运越大，启动开销摊得越薄（§13.8 的 MTE Bound 对策） |
| 能不能整除 | 让 `blockLength` 被 `tileLength` 整除 | 整除就没有尾块，省掉 §13.5.3 那一整套麻烦 |
| 流水填充比例 | 块数 N 不能太少 | 回看 §13.4.3 的公式：`(Σ−max)` 那部分是固定开销，N 太小就摊不薄 |

> ★ **三个考虑互相打架**，这就是为什么昇腾 tiling 通常是"算出上界 → 在 2 的幂里挑几个候选 → 实测扫一遍"。**上界必须手算，最优值只能实测。**

### 13.5.3 尾块：静默精度 bug 的头号来源

这一节是本章最该记住的一节。**先纠正一个流传很广的说法。**

> ⚠️ **"昇腾没有 mask"这句话不准确，需要分成两层说**：
> - **计算层面有 mask**（✅ CANN 官方 API 文档）。`Add` / `Mul` / `Exp` 这类矢量 API 的"tensor 高维切分计算"形式接受 `mask` 参数，有**逐 bit 模式**（`uint64_t mask[2]`，按位控制哪些元素参与）和**连续模式**（`uint64_t mask`，表示参与计算的连续元素个数）两种。所以"算的时候只算前 n 个元素"是完全能表达的——最简单的甚至只要用 `calCount` 那个形式。
> - **搬运层面没有 mask**。`DataCopy` **没有**逐元素开关，它只有一个长度，而且这个长度**必须是 32 字节的整数倍**。Triton 里 `tl.load(ptr + offs, mask=offs < n)` 那种"越界的位置就别读"的能力，`DataCopy` 没有。
>
> **所以真正的坑不在"没有 mask"，而在"搬运有 32 字节对齐的硬要求"**——尾块的元素个数凑不齐 32 字节时，你必须显式处理，而处理方式的默认行为会静默给出错误结果。

**把账算出来。** 沿用 §13.5.2 的设定，换一组不整除的数字：

```
totalLength = 1,000,000 个 half
coreNum     = 40（910B3 的 AIV 数 ⚠️）
tileLength  = 2048（在 §13.5.2 算出的上界 13104 以内，且是 2 的幂）

第 1 步：每核多少
  blockLength = 1000000 / 40 = 25000 个元素   ✓ 整除，先不考虑核间不均

第 2 步：每核切几块
  25000 / 2048 = 12.207...
  → 12 个整块（12 × 2048 = 24576 个元素）
  → 尾块 = 25000 − 24576 = 424 个元素

第 3 步：尾块对不对齐？这一步是关键
  424 个 half × 2 字节 = 848 字节
  848 / 32 = 26.5              ✗ 不是整数！
  848 = 32 × 26 + 16           差 16 字节才能凑齐 27 个 32 字节块

  → 对齐到 32 字节需要 864 字节 = 432 个 half
  → 也就是要多搬 8 个元素，而这 8 个元素在 GM 上是【别人的数据或越界】
```

**三种处理方式，逐个看代价和风险：**

| 方式 | 怎么做 | 代价 | 风险 |
|---|---|---|---|
| **① 在 tiling 里保证整除** | 调整 `tileLength`，或者 Host 侧 pad 输入到整数倍 | 要么 `tileLength` 不自由，要么多一次 pad 的搬运 | 最安全，但很多时候做不到（shape 是外面给的） |
| **② 尾块单独走 `DataCopyPad`** | 主块用 `DataCopy`（快路径），尾块切到 `DataCopyPad` | 多一个分支；`DataCopyPad` 比 `DataCopy` 慢 | **⚠️ 填充值不保证为 0，见下** |
| **③ 尾块也用整块长度搬，靠计算侧的 `calCount` / `mask` 挡住** | 搬 432 个进来，只算前 424 个 | 读了 8 个不属于自己的元素 | 搬入侧是**越界读**；搬出侧如果不挡就是**越界写**，会踩坏别人的数据 |

**方式 ② 是官方推荐路径，但它有一个必须知道的陷阱：**

```cpp
// 尾块搬入：DataCopyPad（✅ CANN 7.0+ 提供，⚠️ 主要在训练卡上支持，
//                        310 系列推理卡不支持这类非对齐 API）
AscendC::DataCopyExtParams    copyParams{1, tailLength * (uint32_t)sizeof(half), 0, 0, 0};
//                            ↑ blockCount, blockLen【单位是字节，不是元素个数】
AscendC::DataCopyPadExtParams<half> padParams{true, 0, (uint8_t)(alignLen - tailLength), (half)0};
//                                  ↑ isPad, leftPadding, rightPadding, paddingValue
AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
```

> ⚠️ **这是本章最值钱的一句：`DataCopyPad` 为了凑 32 字节对齐而多搬进来的那几个位置（官方文档叫 dummy），值是不确定的。** CANN 文档的原话是：`leftPadding` / `rightPadding` 都为 0 时，dummy **默认填的是待搬运数据块的第一个元素值**；`leftPadding` / `rightPadding` 不为 0 而 `isPad` 为 `false` 时，填的是**随机值**。**无论哪种，都不是 0。**
>
> 后果：如果你搬进尾块之后直接做 `ReduceSum` 这类**整块归约**，那几个 dummy 会被加进和里。这个错误的表现是：
> - **不报错**、不 NaN，只是最后几个元素的归一化结果偏了一点；
> - 而且只在 shape 不整除时出现——用 1024、4096 这类整齐的 shape 测，**永远测不出来**。
>
> **两个必须做的动作**：① 上面代码里 `isPad = true` + `paddingValue = 0` 显式指定填 0；② 计算侧用 `calCount = tailLength`（或 mask 连续模式）把有效长度传进去，**别让归约看到尾巴**。两个都做，不要只做一个。

> ★ **搬出方向是安全的**（✅ CANN 文档）：`DataCopyPad` 从 UB 搬到 GM 时，框架自动补齐到 32 字节再搬，**到了 GM 会把补的那部分丢弃**，不会污染尾块之后的数据。所以只有**搬入**方向需要你操心填充值。

**Host 侧配套的 tiling 字段**（最小可用版）：

```cpp
TILING_DATA_FIELD_DEF(uint32_t, totalLength);   // 总元素数
TILING_DATA_FIELD_DEF(uint32_t, tileLength);    // 整块长度
TILING_DATA_FIELD_DEF(uint32_t, mainTileNum);   // 整块有几块（本例 12）
TILING_DATA_FIELD_DEF(uint32_t, tailLength);    // 尾块长度（本例 424），0 表示没有尾块
```

kernel 里最后一次循环判断一下：

```cpp
for (int32_t i = 0; i < loopCount; i++) {
    uint32_t len = (i == loopCount - 1 && tailLength > 0) ? tailLength : tileLength;
    CopyIn(i, len);     // len != tileLength 时内部切到 DataCopyPad 路径
    Compute(i, len);    // 把 len 作为 calCount 传给 Add，不算尾巴
    CopyOut(i, len);
}
```

> ⚠️ **测试纪律**：昇腾算子的单元测试**必须包含一组不整除的 shape**。用 `N = 1000000`、`N = 1023`、`N = 424` 这类数字，而不是清一色的 1024 / 4096。[14 章 §14.4](14-GPU到NPU迁移实战.md) 的静默错误清单里，"尾块处理漏了"排在前列，就是因为大家测的 shape 都太整齐了。

### 13.5.4 核间负载均衡：为什么它比单核效率更重要

`totalLength / GetBlockNum()` 除不尽时，总有核要多干活。**而整个算子的耗时由最慢的那个核决定。**

```
例：totalLength = 1,000,000，coreNum = 40
  1000000 / 40 = 25000        ✓ 整除，每核一样

例：totalLength = 1,000,033，coreNum = 40
  1000033 / 40 = 25000.825
  笨办法：前 39 核各 25000，最后 1 核 25000 + 33 = 25033
    → 最后那个核多干 0.13%，几乎无感 ✓

  真正危险的是"块数不均"：
  假设 tileLength = 2048，每核 25000 → 12 块 + 尾块
  但某个核只分到 2000 个元素 → 1 块都不到，全是尾块
    → 这个核走的全是慢速的 DataCopyPad 路径
    → 而它拖累的是整个算子
```

**变长场景才是真战场。** [09 章 §9.5](09-推理侧算子.md) 讲的连续批处理里，一个 batch 里每条序列长度都不同。如果按"一条序列一个核"分，那 20 个核里有 1 个拿到超长序列、慢 30%，**整个算子就慢 30%**——另外 19 个核算完在干等。

```
✗ 按序列分核（负载极不均）
  核 0: seq_len = 4096  ████████████████████████
  核 1: seq_len =  128  █
  核 2: seq_len =  256  ██
  ...
  整体耗时 = 核 0 的耗时，其余 19 个核大部分时间空转

✓ 按总 token 数分核（先把所有序列拼成一条，再按 token 均分）
  核 0: token 0     – 2047   ████████████
  核 1: token 2048  – 4095   ████████████
  核 2: token 4096  – 6143   ████████████
  ...
  整体耗时 ≈ 单核耗时，全核满载
  代价：一个核可能跨越两条序列的边界，Attention 这类算子要额外处理边界
```

> ★ 这就是为什么昇腾上 FlashAttention 类算子的优化文章**通篇在讲"核间负载均衡"**，而 GPU 那边讲的是"把 softmax 留在 SRAM 里"（[04 章 §4.9.1](04-FlashAttention全系.md) 解释了这个分岔）。GPU 有几千个 Block 去填 132 个 SM，天然有调度弹性；昇腾只有 40 个核，一个核慢就全慢，**没有任何弹性**。

### 13.5.5 Tiling 代码：两个文件

**定义结构体**（Host / Device 共用）：

```cpp
// add_custom_tiling.h
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);   // 总元素数
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);       // 见 §13.3.1 的 ⚠️：这是块数 ÷ BUFFER_NUM
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
```

**Host 侧 Tiling 函数**：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTilingData tiling;

    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // ── 关键决策 1：用几个核？
    //    注意昇腾的 blockDim = 核数（或 AIC+AIV 组合数），不是 CUDA 的"每 block 线程数"
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();   // 纯 Vector 算子取 AIV 数
    context->SetBlockDim(coreNum);

    // ── 关键决策 2：UB 有多大？—— 别写死，查出来（§13.5.2 第 1 步）
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // 到这里就可以按 §13.5.2 的式子反解 tileLength 上界了：
    //   tileLength ≤ ubSize / (张量数 × BUFFER_NUM × sizeof(元素))
    // 再留 20% 余量、对齐到 32 字节，最后在候选值里实测挑一个

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(8);                 // ⚠️ 演示值。生产代码要按上面反解出来的上界挑

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
```

> ⚠️ `GetCoreNumAiv()` 取的是**向量核数**，`GetCoreNumAic()` 取的是**矩阵核数**，比例 1 : 2（[12 章 §12.2.3](12-昇腾硬件基础.md)）。纯 Vector 算子用 AIV 数，纯 Cube 算子用 AIC 数。**混合算子（MIX）要用 AIC 数**，理由见下一节。

### 13.5.6 `blockDim` 这个词的三层含义

这是全章最容易出错的命名问题，必须拆开说。

| 场景 | `SetBlockDim(n)` 里的 n 是什么 | `GetBlockNum()` 返回什么 | 实际启动几个核 |
|---|---|---|---|
| CUDA（对照） | 每个 Block 的**线程数** | — | — |
| 昇腾 · 纯 Vector 算子 | **AIV 核数** | AIV 核数 | n 个 AIV |
| 昇腾 · 纯 Cube 算子 | **AIC 核数** | AIC 核数 | n 个 AIC |
| 昇腾 · MIX（Cube+Vector 混合） | **"AIC+AIV 组合"的个数** | 组合数 | n 个 AIC + **2n 个 AIV** |

```
分离架构（910B/910C）上一个"组合"长这样：

  组合 k  ├── 1 个 AIC（矩阵核）
          ├── 1 个 AIV（子编号 0）
          └── 1 个 AIV（子编号 1）

  某处理器有 20 个 AIC + 40 个 AIV → 建议 SetBlockDim(20)
  → 启动 20 个组合 = 20 个 AIC + 40 个 AIV
```

> ⚠️ **MIX 模式下 `GetBlockIdx()` 不够用**（✅ CANN 官方 API 说明）。同一个组合里的两个 AIV，`GetBlockIdx()` 返回**相同的值**，要用 `GetSubBlockIdx()`（返回 0 或 1）区分。所以在 MIX 算子里，AIV 的全局唯一编号要写成：
>
> ```cpp
> uint32_t aivId = AscendC::GetBlockIdx() * 2 + AscendC::GetSubBlockIdx();
> uint32_t aivNum = AscendC::GetBlockNum() * 2;
> ```
>
> **只写 `GetBlockIdx()` 的后果**：同一组合里的两个 AIV 算了**同一段数据**，另一段数据谁都没算。表现是结果里有一半是脏数据——又是一个静默错误。
>
> ⚠️ 配套地，Host 侧用 `MultiCoreMatmulTiling` 时 `SetDim()` 要填 **AIV 数**（= `SetBlockDim` 的 2 倍），因为 Matmul 的切分是按 AIV 视角做的。这两个数不一致是 MIX 算子最高频的配置错误。

---

## 13.6 稍微真实一点：RMSNorm 的 Compute 段

逐元素加用不到 Cube，也看不出 Vector 指令的丰富程度。看一个大模型里天天用的 **RMSNorm**（Root Mean Square Normalization，均方根归一化）：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) × gamma

按 hidden 维（一行）做，每行独立。假设 hiddenSize = 4096。
```

数据流比逐元素加多了一步"沿 hidden 维归约"：

```
UB 上同时要活着的张量：

  xLocal[4096]      ← 从 TQue 取（跨阶段，进队列）
  gammaLocal[4096]  ← 从 TQue 取（权重，也要搬进来）
  yLocal[4096]      ← 往 TQue 放（跨阶段）
  sqLocal[4096]     ← TBuf，x² 的中间结果（不跨阶段）
  sumLocal[小]      ← TBuf，归约结果 + ReduceSum 的工作空间
  workLocal[小]     ← TBuf，ReduceSum 要求的工作空间

  ★ 注意 TQue 和 TBuf 在这里第一次分清了：
    xLocal / gammaLocal / yLocal 要在阶段之间传，用 TQue；
    sqLocal / sumLocal / workLocal 只在 Compute 里活着，用 TBuf。
```

Compute 段的骨架：

```cpp
__aicore__ inline void Compute(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal     = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
    AscendC::LocalTensor<float> yLocal     = outQueueY.AllocTensor<float>();
    // 临时变量用 TBuf，不进队列、不需要 BUFFER_NUM
    AscendC::LocalTensor<float> sqLocal   = tmpBuf.Get<float>();
    AscendC::LocalTensor<float> sumLocal  = sumBuf.Get<float>();
    AscendC::LocalTensor<float> workLocal = workBuf.Get<float>();

    // ① x²                                                  Vector，4096 个元素
    AscendC::Mul(sqLocal, xLocal, xLocal, this->hiddenSize);
    // ② 沿 hidden 维求和（归约）                            Vector，log 级步骤数
    AscendC::ReduceSum<float>(sumLocal, sqLocal, workLocal, this->hiddenSize);
    // ③ mean + eps                                          Vector，只算 1 个元素
    AscendC::Muls(sumLocal, sumLocal, 1.0f / (float)this->hiddenSize, 1);
    AscendC::Adds(sumLocal, sumLocal, this->eps, 1);
    // ④ 1/sqrt(...)                                         Vector，只算 1 个元素
    AscendC::Rsqrt(sumLocal, sumLocal, 1);
    // ⑤ x × rsqrt × gamma
    float rms = sumLocal.GetValue(0);                  // ⚠️ 标量回读，见陷阱 1
    AscendC::Muls(yLocal, xLocal, rms, this->hiddenSize);
    AscendC::Mul(yLocal, yLocal, gammaLocal, this->hiddenSize);

    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueGamma.FreeTensor(gammaLocal);
}
```

> ⚠️ **这是演示骨架，不是可直接编译的生产代码。** 省略了 `Init` 和 TBuf 的声明；`ReduceSum` 的 `workLocal` 大小要求**随芯片型号而变**（✅ CANN 文档：Atlas 训练系列要按公式算最小空间，Atlas A2 的"tensor 前 n 个元素"接口可以传任意大小）；⚠️ **CANN 9.1.0 起归约基础 API 改过名**（`WholeReduceSum` → `ReduceRepeat<SUM>`，`BlockReduceSum` → `ReduceDataBlock<SUM>`），跨版本移植要查。**以你所装 CANN 版本的官方 Kernel API 手册和 `samples` 仓库的 RmsNorm 样例为准。**

这段代码暴露了三个昇腾特有的陷阱。

**陷阱 1：标量回读（`GetValue`）会打断流水。**

```
第 ⑤ 步的 sumLocal.GetValue(0) 做了什么：

  Vector 单元 ──► UB ──► Scalar 单元的寄存器
                          ↑ 要读到这里，UB 上的值必须已经写完
                          ↑ 于是硬件必须【等 Vector 流水排空】

  代价：前面所有 Vector 指令的延迟都暴露出来，不再被流水掩盖
```

生产代码的做法是**不回读**：把 `rsqrt` 的结果留在 LocalTensor 里，用广播乘法（`BroadCast` 把 1 个值铺成 4096 个）或者 `Div` 直接做张量级运算。多一次 Vector 操作，换回不打断流水——通常是赚的。

> ⚠️ 但也别一刀切。`GetValue` 一次的开销是固定的，如果整个 Compute 段有几十条 Vector 指令，一次回读摊下来可能无关紧要。**先测再改**，这是 §13.8 的工作。

**陷阱 2：归约（`ReduceSum`）是 Vector 上的昂贵操作。**

它有 log 级的步骤数、需要额外的 UB 工作空间，而且**累加顺序按二叉树**（✅ CANN 文档），所以和 GPU 上的结果可能有微小的数值差异——做训推一致性比对时（[11 章](11-训推一致性算子层.md)）这一点要记住。

RMSNorm、Softmax、LayerNorm 这类算子在昇腾上**全是 Vector Bound**：它们根本碰不到 Cube，[12 章 §12.4](12-昇腾硬件基础.md) 那个"320 TFLOPS 峰值算力"对它们毫无意义。

**陷阱 3：想和前后的 GEMM 融合，会撞上 AIC/AIV 分离。**

```
GPU 上（05 章）：GEMM + RMSNorm 融合
  GEMM 的结果还在寄存器/共享内存里 → 顺手做 RMSNorm → 写回 HBM
  收益 = 省掉一次 HBM 往返 ✓

910B 上（12 章 §12.2.3）：
  GEMM 跑在 AIC，RMSNorm 跑在 AIV
  AIC 和 AIV 之间【没有直连通道】，中间结果必须绕 GM（经 L2）
  → "省掉一次往返"这个收益，直接没了 ✗

于是昇腾的做法换成 CV 流水并行：
  把数据切块，让 AIC 算第 k+1 块的 GEMM，同时 AIV 处理第 k 块的 RMSNorm
  → 收益从"消除搬运"变成"掩盖搬运"
```

> ★ **这是本仓库反复出现的一个主题**：同样叫"融合"，GPU 上是**消除中间结果落盘**，昇腾上是**用流水重叠掩盖搬运**。物理约束不同，优化的着力点就不同。完整讨论见 [12 章 §12.2.3](12-昇腾硬件基础.md) 和 [14 章 §14.2](14-GPU到NPU迁移实战.md)。

---

## 13.7 编译与运行

两条路，按目的选。

### 13.7.1 Kernel 直调与 CPU 孪生调试

不走算子工程，直接在一个 cpp 文件里调核函数，最快验证功能：

```cpp
// ① CPU 孪生调试：在 x86 上跑，能用 gdb 单步、能 printf
ICPU_RUN_KF(add_custom, blockDim, x, y, z, workspace, tiling);

// ② NPU 上板运行
ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice,
                                workspace, tilingDevice);
```

> ⚠️ `ACLRT_LAUNCH_KERNEL` 是较新 CANN（8.x）的写法。早期版本用的是 `add_custom<<<blockDim, l2ctrl, stream>>>(...)` 这种类 CUDA 的三尖括号语法。**看到老教程里的三尖括号不要奇怪，它是同一件事的旧写法。**

**CPU 孪生调试（CPU Twin Debugging）是昇腾相对 CUDA 的一个真实优势。** 同一份 kernel 代码能在 CPU 上以**单核串行**方式跑起来：

```
CPU 孪生模式下发生了什么：

  ① 所有 __aicore__ 函数编译成普通 x86 函数
  ② UB / L1 / L0 用普通的堆内存模拟
  ③ MTE2 / MTE3 的 DataCopy 退化成 memcpy
  ④ TQue 的 EnQue / DeQue 退化成【立即返回】—— 因为是单核串行，没有并发
  ⑤ blockDim 个核变成一个 for 循环，一个一个跑

  于是你能做：gdb 单步、打断点、printf、valgrind
  你不能做：看性能（③④ 把流水全抹掉了，CPU 上的耗时和 NPU 毫无关系）
```

> ★ **标准流程是两步走**：先在 CPU 上把**功能和精度**调对（尾块、边界、数值），再上板调**性能**（流水、tiling）。CUDA 侧没有等价物——`cuda-gdb` 是在设备上调，体验差很多，Triton 的 `TRITON_INTERPRET=1` 只验证逻辑且不同版本支持程度不一（[07 章 §7.9.2](07-Triton编程.md)）。

> ⚠️ **CPU 孪生调试测不出并发 bug**。上面第 ④ 条是关键：CPU 上 `DeQue` 立即返回，所以"漏写 `FreeTensor` 导致死锁""`EnQue` / `DeQue` 配对错误"这类问题在 CPU 上**可能跑得好好的**，一上板就挂。**同步相关的问题必须上板验证。**

### 13.7.2 算子工程（交付用）

```bash
# 1) 用 msOpGen 从算子原型 JSON 生成工程骨架
msopgen gen -i add_custom.json -c ai_core-Ascend910B3 -lan cpp -out ./AddCustom

# 2) 目录结构
#    AddCustom/
#      op_host/      ← Tiling 函数、算子原型注册（§13.5.5 那两段）
#      op_kernel/    ← 核函数（Device 侧，就是 §13.3.2 那段）
#      CMakeLists.txt

# 3) 编译，产出自定义算子包
./build.sh

# 4) 安装到 CANN 的自定义算子目录
./custom_opp_<arch>.run

# 5) PyTorch 侧调用：通过 aclnn 接口或 torch_npu 的自定义算子注册
```

### 13.7.3 `soc_version`：一个真实的运维负担

注意 `-c ai_core-Ascend910B3` 这个参数——**算子是按 `soc_version` 编译的**。

```
CUDA 的兼容机制：
  源码 ──► PTX（虚拟汇编，跨代通用）──► SASS（运行时由驱动 JIT 编译）
  → 一份二进制能在未来的卡上跑（前向兼容）

昇腾的机制：
  源码 ──► 直接编成目标型号的指令
  → 910B3 编的包【不能】直接在 910B4 上跑
  → 多型号混合的集群里，每个型号都要单独编一份
```

> ⚠️ 这一条在生产环境的实际影响：一个自定义算子要覆盖 910B2/B3/B4 + 910C，就要维护四份编译产物和四条 CI 流水线。**做迁移方案评估时要把这个成本算进去**（[14 章 §14.1](14-GPU到NPU迁移实战.md)）。另外 §13.2.4 提到的 **ISASI** 标签也在这一层起作用：打了 ISASI 的 API 不保证跨硬件版本兼容，换型号时它们是首先要复查的。

---

## 13.8 调优：四类瓶颈的排查路径

[12 章 §12.6](12-昇腾硬件基础.md) 给了昇腾的四类瓶颈（Cube / Vector / MTE / 下发）。这一节是完整的工具链路径和判读方法。

### 13.8.1 采集三件套

```bash
# ① 采集：上板
msprof op --output=./prof_out ./my_op_test

# ①' 采集：仿真模式（不占卡，能看到指令级流水，但只能在 0 卡跑）
msprof op simulator --soc-version=Ascend910B3 --output=./prof_out ./my_op_test

# ② 解析
msprof --export=on --output=./prof_out

# ③ 可视化：MindStudio Insight 打开，看「指令流水图」
#    这是昇腾调优最核心的一张图，对应 Nsight Compute 的 warp stall 分析

# ④ 模型级：自动瓶颈诊断 + 专家建议 + GPU/NPU 性能比对
msprof-analyze advisor all -d ./prof_out
```

### 13.8.2 排查顺序：先排除不在算子里的问题

**这个顺序和 GPU 侧不一样，不能照搬 [02 章 §2.7](02-屋顶线与性能剖析.md) 的三步漏斗。**

```
第 0 步：算子总耗时之和 ≈ 端到端耗时吗？
  │
  ├─ 差很多（算子加起来远小于端到端）
  │    └─► 【下发 Bound】瓶颈在 Host 侧，不在算子里
  │         对策：开图模式（ACLGraph / TorchAir，见 14 章 §14.5）、合并小算子
  │         ★ 到这里就别再调算子了，调也没用
  │
  └─ 差不多 → 继续
       │
第 1 步：看指令流水图，哪一级占比最高？
       │
       ├─ MTE2 / MTE3 高  →  【MTE Bound】  见 §13.8.4 案例一
       ├─ Vector 高       →  【Vector Bound】见 §13.8.4 案例三
       ├─ Cube 高         →  【Cube Bound】  见 §13.8.4 案例二（理想状态）
       └─ 三级都不高      →  回第 0 步，或查同步开销 / 标量回读
```

> ★ **昇腾要先排除 Host 侧，GPU 可以直接看 kernel。** 原因是昇腾单算子的下发开销相对更显眼，而且推理 Decode 阶段算子又多又小（[09 章 §9.9](09-推理侧算子.md)）。**很大比例的"昇腾算子慢"，最后查出来根本不是算子慢。**

### 13.8.3 指令流水图怎么读

Insight 里的指令流水图，每一行是一条硬件流水线，横轴是时间，实心代表在干活：

```
读法：看哪一行最满，那就是瓶颈。

  占比 = 该流水线的忙碌时间 ÷ 算子总耗时

  ★ 判据不是"占比高不高"，而是"最高的那一级是哪个，它有多接近 100%"
    最高的那级 > 85%  →  瓶颈明确，按它的对策走
    最高的那级 < 50%  →  没有单一瓶颈，去查同步开销和下发
```

### 13.8.4 三个案例

**案例一：MTE Bound（最常见）**

```
时间 →
MTE2  ████████████████████████████░░░░   占比 88%   ← 瓶颈在这
Cube  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  0%   ← 纯 Vector 算子，没用到
Vector███████░░░░░░░░░░░░░░░░░░░░░░░░░   占比 24%
MTE3  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░   占比 20%

判定：MTE Bound（搬入受限）
对策按优先级：
  1. 确认 BUFFER_NUM=2 真的开了，而且没被 §13.4.5 那三种写法写废
     —— 最常见的低级错误，先查这个，代价最低
  2. 调大 tileLength —— 单次搬运越大，MTE 启动开销摊得越薄
     受 UB 容量限制，上界按 §13.5.2 算
  3. 检查 32 字节对齐 —— 不对齐会走 DataCopyPad 的慢路径（§13.5.3）
  4. 检查数据格式 —— ND ↔ FRACTAL_NZ 的隐式转换是不是在这里发生
     （12 章 §12.3.3；torch_npu 里查 npu_format_cast 的调用）
```

> ★ 注意这个形态**和 §13.4.3 那张双缓冲时间线一模一样**：MTE2 占 89%，Vector 和 MTE3 各占 44%。**这说明流水已经排满了，MTE Bound 是这个算子的正常终点**——逐元素算子的算术强度只有 0.1 量级（[02 章 §2.3](02-屋顶线与性能剖析.md)），它注定是搬运受限的。这时候对策 1 和 2 都已经做到位了，真正的出路是**算子融合**（[05 章](05-算子融合.md)）：既然搬一趟这么贵，就在这一趟里把能算的都算完。

**案例二：Cube Bound（理想状态）**

```
MTE2  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比 12%
Cube  ██████████████████████████████░░   占比 92%
Vector░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  3%

判定：Cube Bound —— 恭喜，这是矩阵算子该有的样子
对策：已接近硬件上限。剩下的收益在换更大 tiling、提高 Cube 利用效率
     （数据按 FRACTAL_NZ 摆好，减少格式转换）
```

**案例三：Vector Bound**

```
MTE2  ████████░░░░░░░░░░░░░░░░░░░░░░░░   占比 25%
Cube  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  6%
Vector██████████████████████████████░░   占比 91%

判定：Vector Bound —— AIV 忙死，AIC 闲着
对策：
  1. 减少 Vector 指令数 —— 查有没有多余的类型转换（Cast）、
     有没有能合并的 Muls + Adds
  2. 检查标量回读（GetValue）—— 它会让 Vector 流水排空（§13.6 陷阱 1）
  3. 检查 mask / repeatTimes 用法 —— 用"tensor 前 n 个元素"这种简单形式时，
     每次迭代的元素数由 API 自己定；改用高维切分形式手工控制 repeat，
     有时能减少迭代次数
  4. CV 流水并行 —— 把能挪的计算挪给闲着的 Cube（§13.6 陷阱 3）
```

**案例四：下发 Bound（第 0 步就该抓到）**

```
MTE2  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  9%
Cube  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  6%
Vector███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比 10%
（三级都不满，但端到端耗时很长）

判定：下发 Bound —— 瓶颈在 Host 侧，不在算子里
对策：开图模式减少下发次数（见 [14 章 §14.5](14-GPU到NPU迁移实战.md)），或合并小算子
     ★ 这一类不要在算子里找答案
```

### 13.8.5 和 GPU 侧诊断流程的对照

| | GPU（[02 章 §2.7](02-屋顶线与性能剖析.md)） | 昇腾（本节） |
|---|---|---|
| 第一步 | `nsys` 看全局时间线，找最慢的 kernel | **先确认瓶颈是否在 Host 侧** |
| 主判据 | 屋顶线：Compute Bound / Memory Bound 二选一 | **指令流水图：四级占比，四选一** |
| 瓶颈类别数 | 2 类（+ 下发） | 4 类 |
| 核心工具 | Nsight Compute 的 warp stall 分析 | MindStudio Insight 的指令流水图 |
| 延迟掩盖靠什么 | Occupancy（多 warp 轮转） | **流水重叠**（昇腾没有 occupancy） |
| 最常见的误诊 | 以为 Occupancy 低就是问题 | 以为算子慢，其实是下发慢 |

---

## 13.9 Ascend C vs Triton vs CUDA：三方对照

### 13.9.1 同一个算子，三种写法

任务都是 `z[i] = x[i] + y[i]`。

| | CUDA C++ | Triton | Ascend C |
|---|---|---|---|
| 代码量 | ~30 行（kernel + 启动器） | **~7 行** | ~70 行 kernel + ~25 行 Host Tiling |
| 你站在哪一级写 | **一个线程** | **一个 Block（program）** | **一个核 + 三段流水** |
| 一行代码描述的是 | 一个标量的操作 | 一整块数据的操作 | 一整块数据的操作 + 它在哪条流水线上 |
| 越界怎么挡 | `if (i < N)` | `mask=offs < n` | **Host 侧算尾块 + `DataCopyPad` + `calCount`** |

### 13.9.2 谁替你做了什么：这是全章的收束

| 这件事 | CUDA C++ | Triton | Ascend C |
|---|---|---|---|
| 元素 → 线程的映射 | **你**（`threadIdx`） | 编译器 | **不存在这个问题**（没有线程） |
| 合并访存 | **你**（[01 章 §1.4](01-GPU硬件基础.md)） | 编译器 | **不存在**；换成 32 字节对齐要求 |
| Bank Conflict | **你**（padding / swizzle） | 编译器 | **不存在**（[12 章 §12.3.2](12-昇腾硬件基础.md)） |
| 片上内存分配 | **你**（`__shared__`） | 编译器 | **你**（`TPipe` + `InitBuffer`） |
| 搬运 | **你**（或 `cp.async`） | 编译器（`tl.load`） | **你**（`DataCopy`） |
| 同步屏障 | **你**（`__syncthreads()`） | 编译器 | **你**（`EnQue` / `DeQue`） |
| 流水重叠 | **你**（手写软件流水） | 半自动（`num_stages` 一个参数） | **你**（`BUFFER_NUM` + 三段式结构） |
| 选矩阵指令 | **你**（`mma` / `wgmma`） | 编译器 | 高阶 API（`Matmul`）或你自己摆 L0A/L0B |
| 分块参数在哪定 | kernel 启动参数 | `tl.constexpr` + `@triton.autotune` | **Host 侧独立的 Tiling 函数** |
| 自动调优 | 无（手工或 CUTLASS） | **`@triton.autotune` 内置** | **无内置**，要自己写循环实测 |
| 延迟怎么掩盖 | Occupancy（多 warp 轮转） | Occupancy + `num_stages` | **只有流水重叠**，没有兜底 |
| CPU 上能调试吗 | 不能（`cuda-gdb` 在设备上） | ⚠️ `TRITON_INTERPRET=1`，仅验逻辑 | **能（CPU 孪生调试）** |
| 学习曲线 | 陡 | 平缓 | **最陡** |

> ★ **把这张表读成一句话**：Triton 从 CUDA 手里拿走了"元素怎么摊到线程上"这一整类问题；Ascend C 也拿走了这一类（因为昇腾根本没有线程），**但它把"搬运和流水"还给了你**。所以 Ascend C 既没有 CUDA 的细粒度控制力，也没有 Triton 的省事——它在两者之间，而且是两者的坏处相加。

> ⚠️ **但"难写"不等于"不该用"。** 昇腾上没有第二个成熟选择（§13.9.4），而且 Ascend C 的显式流水在**矩阵 + 向量混合的融合算子**上确实给了 Triton 给不了的控制力（CV 流水并行没法交给编译器）。

### 13.9.3 行数差在哪：把六十行拆开

前面 §13.3.5 给过一次，这里换个角度——**按"这些代码在填哪个抽象层的坑"来分**：

```
Triton 7 行
  │
  │ + 3 行   本核负责哪一段（Triton 里是 tl.program_id 一行）
  │ + 9 行   片上内存的申请和归还（Triton 里编译器包办）
  │ + 7 行   DataCopy + EnQue + DeQue + FreeTensor（Triton 里 tl.load/store 两行）
  │ + 30 行  三段式类结构：构造函数、Init、Process、三个私有方法、成员声明
  │          （Triton 里 num_stages=2 一个参数）
  │ + 8 行   核函数入口 + GET_TILING_DATA
  │ + 25 行  Host 侧 Tiling 结构定义 + Tiling 函数（Triton 里 BLOCK=1024 一个参数）
  ▼
Ascend C 约 95 行（kernel 70 + Host 25）
```

> ★ **最大的一块是那 30 行三段式结构，它换回来的是 §13.4 算出的 1.78 倍。** 这笔交易在昇腾上是强制的——不做这个交易，你就付 1.78 倍的耗时。在 Triton 上这笔交易是一个参数。**这个差别就是"编译器能不能替你排流水"，也是 [00 章 §0.3](00-总纲.md) 讲两种硬件哲学时的落脚点。**

### 13.9.4 triton-ascend：这一节的旧说法已经过时

> ⚠️ **本章原稿（以及本仓库早期版本）说"昇腾没有 Triton 的等价物"，这一条已经过时。**
>
> `triton-lang/triton-ascend`（面向昇腾 NPU 的 Triton 语言与编译器）**已开源**，项目文档口径是**"支持 85% 以上的 Triton Python API"**（⚠️ 该比例出自项目自述，未见独立复现），覆盖 MatMul、FlashAttention、LayerNorm 等核心大模型算子，并已适配 vLLM 等开源仓里的 Triton 算子。
>
> ⚠️ **但它的成熟度、API 覆盖度和性能水平都还在快速变动**（本仓库写作时的预发布版本是 3.2.0rc4）。**昇腾上的生产算子目前仍以 Ascend C 为主**，理由有三：
> 1. **性能上限**。CV 流水并行、AIC/AIV 分工、FRACTAL_NZ 布局这些昇腾特有的优化，编译器还没学会自动做。
> 2. **生态**。`cann-ops-adv` 里的融合算子、MindSpeed / MindIE 的加速层，都是 Ascend C 写的。
> 3. **可诊断性**。msprof 的指令流水图是按 Ascend C 的流水概念组织的，Triton 生成的 kernel 在这张图上不那么好对应。

**所以选型阶梯是这样的**（完整版见 [14 章 §14.2](14-GPU到NPU迁移实战.md)）：

```
从上往下试，能停在哪层就停在哪层

① torch_npu 现成算子 / aclnn                  ← 零开发
② cann-ops-adv 的融合算子（FA 类都在里面）     ← 零开发，开源可读
③ 图模式（TorchAir / ACLGraph）解决下发问题    ← 零 kernel 开发
④ triton-ascend 试一把                        ← ⚠️ 成熟度待验证，但试的成本很低
⑤ 从零写 Ascend C                             ← 本章，最后的手段
```

> ★ **结论没变，理由变了**：以前说"必须写 Ascend C，因为没别的"；现在说"**优先不写 Ascend C，写不动了再写**"。Ascend C 的开发效率确实明显低于 Triton 和 CUDA，这是昇腾生态目前最真实的短板，缓解办法是尽量不写。

---

## 13.10 本章小结

**一个骨架，三段流水**

```
CopyIn (MTE2)  →  Compute (Vector/Cube)  →  CopyOut (MTE3)
  AllocTensor       DeQue                    DeQue
  DataCopy          计算 API                  DataCopy
  EnQue             EnQue + FreeTensor        FreeTensor

★ 分三段的唯一理由：MTE2 / Vector / MTE3 是三套独立硬件，能同时干活
★ 四个动作里只有 DeQue 会真的停下来等 —— 这就是重叠的来源
```

**谁管什么**

| | 管什么 | 初始化 | 带同步吗 |
|---|---|---|---|
| **TPipe** | 整块 UB 的所有权 | 声明成员变量 | 管事件，不直接用 |
| **TQue** | 阶段**之间**传的数据 | `InitBuffer(que, BUFFER_NUM, bytes)` | **带**，`DeQue` 阻塞 |
| **TBuf** | 阶段**内部**的临时变量 | `InitBuffer(buf, bytes)` | 不带 |

**核心公式，全章就这三个**

```
① UB 占用 = 张量数 × BUFFER_NUM × tileLength × 元素字节数
   → 反解 tileLength 上界，再留 20% 余量、对齐到 32 字节（§13.5.2）

② 单缓冲耗时 = N × Σ T_i
   双缓冲耗时 = (Σ T_i − max T_i) + N × max T_i
   → N 大时加速比 → Σ T_i / max T_i（§13.4.3，本章例子是 1.78 倍）

③ MIX 模式下 AIV 全局编号 = GetBlockIdx() × 2 + GetSubBlockIdx()
   AIV 总数 = GetBlockNum() × 2                （§13.5.6）
```

**四个静默错误（都不报错，只是结果错或性能塌）**

| 错误 | 症状 | 出处 |
|---|---|---|
| 尾块没处理 / `DataCopyPad` 没显式填 0 | 张量末尾少量元素是脏数据；**整齐 shape 测不出来** | §13.5.3 |
| 三段揉成一段写，或 `FreeTensor` 还得太晚 | 结果对，但耗时退化成单缓冲（本例 1.78 倍） | §13.4.5 |
| MIX 模式只用 `GetBlockIdx()` | 一半数据被算两遍，另一半没算 | §13.5.6 |
| `SetDim` 和 `SetBlockDim` 差 2 倍没对上 | Matmul 切分和实际核数不匹配 | §13.5.6 |

**四类瓶颈，排查顺序不能换**

```
第 0 步：算子耗时之和 ≈ 端到端？不是 → 【下发 Bound】，别在算子里找答案
第 1 步：看指令流水图，最满的那一级
   MTE 高    → 查 BUFFER_NUM → 调大 tileLength → 查 32B 对齐 → 查 ND↔NZ
   Cube 高   → 理想状态，剩余收益在更大 tiling
   Vector 高 → 减指令数、去掉 GetValue 回读、CV 流水并行
   都不高    → 回第 0 步，或查同步开销
```

**和 GPU 的三条关键差异**

- **没有兜底的延迟掩盖**。GPU 有 64 个 warp 轮转，昇腾只有手工排流水。所以 `BUFFER_NUM=2` 不是优化，是基本功。
- **搬运没有 mask**。计算侧有 mask（逐 bit / 连续两种模式），搬运侧只有 32 字节对齐的硬要求，尾块必须显式处理。
- **融合的含义变了**。GPU 上融合是消除中间结果落盘；910B 上 AIC 和 AIV 之间必须绕 GM，融合变成用 CV 流水并行掩盖搬运。

**优势与短板**

- **优势**：CPU 孪生调试（功能和精度能在 x86 上用 gdb 调），`cann-ops-adv` 开源可读。
- **短板**：代码量和学习曲线明显高于 Triton；无内置 autotune；算子按 `soc_version` 编译，多型号集群要维护多份产物。

**自测三题**

1. 一个算子在 UB 上同时要活着 4 个 `float`（4 字节）张量，全部开双缓冲。UB 容量按 192 KB（⚠️ 假设值）算。`tileLength` 的上界是多少个元素？留 20% 余量并对齐到 32 字节之后是多少？
2. 某算子三段的耗时是 `T_MTE2 = 1`、`T_Vector = 3`、`T_MTE3 = 1`（⚠️ 假设值），每核处理 8 个 tile。单缓冲和双缓冲各耗时多少？加速比是多少？瓶颈是哪一级？按 §13.8.4，你应该走哪个案例的对策？
3. `totalLength = 100000` 个 `half`，40 个核，`tileLength = 1024`。每核几个整块、尾块多少个元素？尾块的字节数是 32 的整数倍吗？要凑齐需要多补几个元素？

#### 参考答案

1. UB 占用 = `4 (张量) × 2 (双缓冲) × tileLength × 4 (字节)` = `32 × tileLength`。上界 = `196608 ÷ 32` = **6144 个元素**。留 20% → `6144 × 0.8 = 4915.2`；32 字节 = 8 个 `float`，`4915 ÷ 8 = 614.4` → 向下取整 614 → `614 × 8` = **4912 个元素**。实践中会在 4096（2 的幂，好算好测）和 4912 之间实测挑一个。参见 §13.5.2。
2. `Σ T_i = 1 + 3 + 1 = 5`，`max T_i = 3`（Vector）。单缓冲 = `8 × 5` = **40 个时间单位**；双缓冲 = `(5 − 3) + 8 × 3` = `2 + 24` = **26 个时间单位**；加速比 = `40 ÷ 26` ≈ **1.54 倍**。瓶颈是 **Vector**，Vector 占用 = `24 ÷ 26` ≈ 92%。应走 §13.8.4 **案例三（Vector Bound）**的对策：减少 Vector 指令数、去掉 `GetValue` 标量回读、把能挪的计算挪给闲着的 Cube——注意**不是**去调大 `tileLength`，那是 MTE Bound 的对策，在这里没用（甚至有害，因为 Vector 的工作量按元素数线性增长）。参见 §13.4.3 和 §13.8.4。
3. 每核 = `100000 ÷ 40` = 2500 个元素。`2500 ÷ 1024 = 2.44` → **2 个整块**（2048 个元素），尾块 = `2500 − 2048` = **452 个元素**。字节数 = `452 × 2` = **904 字节**；`904 ÷ 32 = 28.25`，**不是整数**（`904 = 32 × 28 + 8`）。凑齐需要 `32 × 29 = 928` 字节 = 464 个 `half`，即**多补 12 个元素**。这 12 个位置就是 `DataCopyPad` 的 dummy 区，**必须用 `isPad = true` + `paddingValue = 0` 显式填 0，并且计算侧用 `calCount = 452`**，否则归约会把它们算进去——而这个错误用 `N = 4096` 这种整齐 shape 永远测不出来。参见 §13.5.3。

---

> **下一章**：不自己写算子的那条路——昇腾现成的大模型融合算子有哪些、怎么用、什么时候必须退回本章。以及一份从 GPU 迁过来的完整 checklist 和静默错误清单。见 [14 · GPU 到 NPU 迁移实战](14-GPU到NPU迁移实战.md)。

> **GPU 对照回看**：本章的 `BUFFER_NUM=2` 对应 [03 章 §3.5](03-GEMM深度剖析.md) 的双缓冲和 [07 章 §7.7.3](07-Triton编程.md) 的 `num_stages`；Host 侧 Tiling 对应 [03 章 §3.3](03-GEMM深度剖析.md) 的分块和 [07 章 §7.8](07-Triton编程.md) 的 autotune；本章的"没有 occupancy、只有流水重叠"，对照 [01 章 §1.6](01-GPU硬件基础.md) 和 [12 章 §12.3.2](12-昇腾硬件基础.md)。两种硬件为什么分岔，见 [00 章 §0.3](00-总纲.md)。

**延伸资料**：
- [CANN · Ascend C 编程范式](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0016.html)（官方，三段式与矩阵五段式的权威说明）
- [Ascend C API 参考 · 基础 API](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/)（**本章所有 API 细节以你所装 CANN 版本的这份手册为准**。注意区分"tensor 前 n 个元素"和"tensor 高维切分计算"两种参数形式，以及打了 ISASI 标签的接口）
- [DataCopyPad 官方文档](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0253.html)（§13.5.3 尾块处理的一手来源，dummy 填充语义就在这一页）
- [如何高效处理 Ascend C 非对齐数据](https://www.hiascend.com/developer/techArticles/20250627-1)（官方技术文章，非对齐搬运的实操与选型）
- [Ascend/samples 算子样例仓](https://github.com/Ascend/samples)（`add_custom`、`RmsNorm` 等样例的一手代码，本章 §13.3 和 §13.6 的对照基准）
- [cann-ops-adv 融合算子库](https://gitee.com/ascend/cann-ops-adv)（开源，读 FlashAttention 类算子源码的入口）
- [msProf 算子调优工具概述](https://www.hiascend.com/document/detail/zh/mindstudio/700/ODtools/Operatordevelopmenttools/atlasopdev_16_0082.html)（官方）
- [msprof-analyze 自动瓶颈诊断](https://gitcode.com/Ascend/mstt)（开源，含 GPU/NPU 性能比对 recipe）
- [基于 Ascend C 的 FlashAttention 算子性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240607-1)（官方技术文章，讲清了 CV 流水并行和核间负载均衡的实操）
- [triton-ascend](https://github.com/triton-lang/triton-ascend)（昇腾的 Triton 后端，§13.9.4 的对象）
