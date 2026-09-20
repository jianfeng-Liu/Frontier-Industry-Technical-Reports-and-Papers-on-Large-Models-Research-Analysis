# 13 · Ascend C 编程：把第 07 章的 Triton 心智换成三段式流水

> **前置**：[12 · 昇腾硬件基础](12-昇腾硬件基础.md) 必读；[07 · Triton 编程](07-Triton编程.md) 用于对照。
> **目标**：写出、编译、跑通、调优你的第一个昇腾算子，并且知道它为什么快或为什么慢。

---

**本章关键词**

**CANN**（Compute Architecture for Neural Networks，神经网络计算架构）—— 昇腾软件栈总称，对标 CUDA 全家桶。
**Ascend C** —— 基于 C++ 的昇腾算子编程语言，对标 CUDA C++。注意它**不是** C 语言，是一套 C++ 模板库。
**AscendCL**（Ascend Computing Language）—— 昇腾的运行时 API，对标 CUDA Runtime API。
**aclnn** —— CANN 提供的单算子 API 层，PyTorch 的每个算子最终落到这里。
**TPipe** —— Ascend C 的片上内存与同步事件管理器，一个算子里通常只有一个。
**TQue**（Tensor Queue，张量队列）—— 连接流水线各阶段的队列，双缓冲就靠它实现。
**TBuf**（Tensor Buffer，张量缓冲）—— 不进队列的临时片上内存，用于中间变量。
**TPosition**（逻辑位置）—— VECIN / VECOUT / VECCALC 等，抽象掉物理存储（UB/L1/L0）的位置标签。
**LocalTensor / GlobalTensor** —— 片上张量 / 片外（GM）张量，Ascend C 里最基本的两种数据句柄。
**Tiling**（分块）—— 把大张量切成片上放得下的小块，并把切分参数从 Host 传给 Device 的整套机制。
**blockDim** —— 昇腾语境下指**参与计算的核数**，**不是** CUDA 里的"每个 block 多少线程"。这个词撞名了，极易误解。
**Double Buffer**（双缓冲）—— 把片上缓冲一分为二，让搬运和计算重叠的基本手法。
**msprof / MindStudio Insight** —— 昇腾的性能采集与可视化工具，对标 Nsight Compute / Nsight Systems。

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
   CUDA C++ / Triton                   ★ Ascend C ★  ← 本章在这一层
      │                                      │
   PTX / SASS                          昇腾指令集
      │                                      │
   NCCL                                HCCL
   Nsight Compute / Systems            msprof / MindStudio Insight
```

**层级对应大致齐整，但有两个例外要注意**：

1. **没有 Triton 那一层**。Triton 在 GPU 侧的价值是"用 Python 写、编译器自动管共享内存和流水"。昇腾上没有等价物被广泛使用——Ascend C 的抽象层级介于 CUDA C++ 和 Triton 之间：比 CUDA 高（不用管线程），但比 Triton 低（搬运和双缓冲要自己写）。
2. **`cann-ops-adv` 是开源的**。这是华为的高阶融合算子库（adv = advanced），FlashAttention 类算子都在里面，可以直接读源码——这一点比读 cuDNN 闭源二进制友好得多。

---

## 13.2 Ascend C 的编程范式：CopyIn → Compute → CopyOut

第 07 章里 Triton 的心智是"我是一个 program，负责处理一块数据，`tl.load` / `tl.store` 自动搞定搬运"。Ascend C 的心智完全不同：

```
Triton（GPU）：
   一个 program 处理一块 → load → 算 → store
   共享内存怎么用、要不要异步预取，编译器替你决定

Ascend C（昇腾）：
   一个核处理一批块，每块走三个阶段，三个阶段并行流水
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ CopyIn  │──►│ Compute │──►│ CopyOut │
   │  MTE2   │   │ Vector  │   │  MTE3   │
   │ GM→UB   │   │ UB→UB   │   │ UB→GM   │
   └─────────┘   └─────────┘   └─────────┘
      ↑ 三个阶段跑在不同的硬件流水线上，能真正同时执行
      ↑ 你的任务：把它们排满，别让任何一级空转
```

阶段之间靠 **TQue** 串起来。TQue 不只是数据容器，它还承担**同步**职责：`EnQue` 表示"这块数据我处理完了，下一级可以取"，`DeQue` 会阻塞直到上一级放进来。硬件层面这对应着流水线之间的 flag 同步，Ascend C 把它包装成了队列操作。

**TPosition**（逻辑位置）是另一个关键抽象。你不直接说"放到 UB"，而是说"这是 VECIN（矢量计算的输入）"，编译器把逻辑位置映射到物理存储：

| TPosition | 含义 | 实际落在 |
|---|---|---|
| `VECIN` | 矢量计算的输入 | UB |
| `VECOUT` | 矢量计算的输出 | UB |
| `VECCALC` | 矢量计算的临时变量 | UB |
| `A1` / `B1` | 矩阵计算的左/右矩阵，第一级 | L1 |
| `A2` / `B2` | 矩阵计算的左/右矩阵，第二级 | L0A / L0B |
| `CO1` / `CO2` | 矩阵计算的结果 | L0C / GM |

这样做的好处是代码不写死物理存储，换代芯片时（第 12 章提到 950 改了 L1 共享策略）不用重写。

---

## 13.3 第一个算子：向量加法的完整代码

先看 Triton 版本（第 07 章的写法），作为对照：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)      # 搬运，编译器管
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)  # 搬运，编译器管
```

同样的功能，Ascend C 版本：

```cpp
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;   // ← 双缓冲。这一行就是性能的一半

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // 第一步：本核负责哪一段？总长度按核数均分
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum     = tileNum;
        // 每次搬多少：本核的份额，再切 tileNum 份，再除以双缓冲的 2
        this->tileLength  = this->blockLength / tileNum / BUFFER_NUM;

        // 第二步：把 GM 上属于本核的那一段挂成 GlobalTensor
        //         GetBlockIdx() 返回本核的编号，相当于 CUDA 的 blockIdx.x
        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + this->blockLength * AscendC::GetBlockIdx(),
                            this->blockLength);

        // 第三步：向 TPipe 申请片上内存。第二个参数 = BUFFER_NUM，开启双缓冲
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);      // 这三个调用看起来是串行的，
            Compute(i);     // 但因为 TQue 的存在，实际在硬件上重叠执行
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        // DataCopy：GM → UB，由 MTE2 执行
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);     // 通知 Compute 阶段：可以取了
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();   // 阻塞等待
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);       // Vector 单元干活

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);   // 及时还回去，否则双缓冲的另一半拿不到内存
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        // UB → GM，由 MTE3 执行
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
    GET_TILING_DATA(tilingData, tiling);    // 从 Host 传来的切分参数
    KernelAdd op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
```

**七行 Triton 变成了七十行 Ascend C**。这个膨胀比是真实的，也是昇腾算子开发的日常。多出来的代码全部在做同一件事：**显式管理数据搬运和流水同步**。

值得逐条对照的差异：

| | Triton | Ascend C |
|---|---|---|
| 搬运 | `tl.load` / `tl.store`，编译器管 | `DataCopy`，你写 |
| 片上内存 | 编译器分配 | `pipe.InitBuffer` + `AllocTensor` / `FreeTensor` |
| 流水重叠 | 编译器自动做软件流水 | `BUFFER_NUM = 2` + `EnQue` / `DeQue` |
| 越界处理 | `mask=offs < n` | **没有 mask**，要靠 tiling 保证整除，或单独处理尾块 |
| 本核负责哪段 | `tl.program_id(0)` | `AscendC::GetBlockIdx()` |
| 一共几个核 | grid 参数 | `AscendC::GetBlockNum()`，由 Host 侧 `SetBlockDim` 决定 |

**没有 mask** 这一条最容易翻车。Triton 里你随手加个 `mask` 就处理了不整除的尾块；Ascend C 里你必须在 tiling 阶段就算清楚——要么保证整除，要么写一段单独的尾块逻辑。这是新手最常见的精度 bug 来源。

---

## 13.4 双缓冲：为什么 `BUFFER_NUM = 2` 是性能的一半

第 12 章说过，MTE 是独立的硬件流水线。双缓冲就是把这个事实变现：

```
BUFFER_NUM = 1（无双缓冲）：一块 UB 同时只能被一个阶段占用
  时间 →
  MTE2:  [搬块0]         [搬块1]         [搬块2]
  Vector:       [算块0]         [算块1]         [算块2]
  MTE3:               [写块0]         [写块1]
         └─ 每一级都在等上一级，硬件大部分时间空转

BUFFER_NUM = 2（双缓冲）：UB 一分为二，交替使用
  时间 →
  MTE2:  [搬块0][搬块1][搬块2][搬块3][搬块4]
  Vector:       [算块0][算块1][算块2][算块3]
  MTE3:               [写块0][写块1][写块2]
         └─ 三级同时在跑，总耗时 ≈ 最慢那一级的耗时
```

理想情况下，双缓冲把耗时从 `T_搬入 + T_计算 + T_搬出` 降到 `max(T_搬入, T_计算, T_搬出)`。对于逐元素算子这类典型的 MTE Bound 场景，这基本等于**性能翻倍以上**。

代价是片上内存占用翻倍——`tileLength` 要相应减半，所以上面代码里 `tileLength = blockLength / tileNum / BUFFER_NUM`。这是 tiling 时必须一起算进去的约束。

> 这里和第 03 章 GEMM 的三级分块、第 04 章 FlashAttention 的分块在思想上是同一件事：**用片上容量换搬运次数，用流水重叠掩盖延迟**。区别只是 GPU 上编译器和库帮你做了大半，昇腾上要自己写。

---

## 13.5 Tiling：昇腾算子开发真正的工作量所在

Ascend C 的一个特点是 **Tiling 在 Host 侧计算**，通过一块内存传给 Device。这和 CUDA 把 block/grid 尺寸当成启动参数一拍脑袋定不一样——昇腾的 tiling 参数往往是一个几十个字段的结构体，包含每一级切多大、循环多少次、尾块怎么处理。

### 13.5.1 定义 Tiling 结构

```cpp
// add_custom_tiling.h
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);   // 总元素数
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);       // 每核切几块
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)
}
```

### 13.5.2 Host 侧 Tiling 函数

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTilingData tiling;

    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 关键决策 1：用几个核？
    //   一般取平台可用的 AI Core 数（第 12 章的 platform_config 里写着）
    //   注意昇腾的 blockDim = 核数，不是 CUDA 的"每 block 线程数"
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();   // Vector 算子取 AIV 数
    context->SetBlockDim(coreNum);

    // 关键决策 2：每核切几块？
    //   块太大 → UB 放不下；块太小 → 搬运次数多、每次搬运效率低
    //   实践中靠"算 UB 容量 + 实测扫一遍"确定
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(8);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
```

### 13.5.3 Tiling 的三条经验

**① UB 容量是硬约束，要手算。**

```
单核 UB 容量（查 platform_config/*.ini，910B 量级约 192 KB ⚠️ 以实际 ini 为准）

本例需要同时放：xLocal、yLocal、zLocal 三块，每块开双缓冲
  所需 UB = 3 (张量) × 2 (双缓冲) × tileLength × sizeof(half)

反解：tileLength ≤ UB容量 / (3 × 2 × 2 Byte)
```

算出上界后往下取整到 32 Byte 对齐（昇腾 DataCopy 有对齐要求），再实测微调。

**② 核间负载均衡比单核效率更重要。**

`totalLength / GetBlockNum()` 除不尽时，最后一个核会多干活，整个算子的耗时由它决定。20 个核里有 1 个慢 30%，整体就慢 30%。变长场景（比如第 09 章讲的连续批处理里，每条序列长度不同）尤其要小心，这是昇腾上 FlashAttention 类算子优化的主要战场之一。

**③ 尾块要单独想。**

没有 mask 兜底，`totalLength` 不是 `coreNum × tileNum × BUFFER_NUM × tileLength` 的整数倍时，要么在 tiling 里保证整除，要么写尾块分支。**建议做法**：让 tiling 函数输出一个 `tailLength` 字段，kernel 里最后一次循环用它。

---

## 13.6 稍微真实一点的例子：RMSNorm 的计算段

逐元素加法用不到 Cube，也看不出 Vector 指令的丰富程度。看一个大模型里天天用的 RMSNorm（Root Mean Square Normalization，均方根归一化）：

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
```

Compute 段的骨架：

```cpp
__aicore__ inline void Compute(int32_t progress)
{
    AscendC::LocalTensor<float> xLocal   = inQueueX.DeQue<float>();
    AscendC::LocalTensor<float> yLocal   = outQueueY.AllocTensor<float>();
    // 临时变量用 TBuf，不进队列
    AscendC::LocalTensor<float> sqLocal  = tmpBuf.Get<float>();
    AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();

    // ① x²
    AscendC::Mul(sqLocal, xLocal, xLocal, this->hiddenSize);
    // ② 沿 hidden 维求和（规约）
    AscendC::ReduceSum<float>(sumLocal, sqLocal, sqLocal, this->hiddenSize);
    // ③ mean + eps
    float meanScale = 1.0f / static_cast<float>(this->hiddenSize);
    AscendC::Muls(sumLocal, sumLocal, meanScale, 1);
    AscendC::Adds(sumLocal, sumLocal, this->eps, 1);
    // ④ rsqrt
    AscendC::Sqrt(sumLocal, sumLocal, 1);
    float rms = sumLocal.GetValue(0);          // 标量回读，有同步开销，慎用
    // ⑤ x / rms * gamma
    AscendC::Muls(yLocal, xLocal, 1.0f / rms, this->hiddenSize);
    AscendC::Mul(yLocal, yLocal, gammaLocal, this->hiddenSize);

    outQueueY.EnQue<float>(yLocal);
    inQueueX.FreeTensor(xLocal);
}
```

> ⚠️ 这是演示骨架，不是可直接编译的生产代码：省略了 Init/TBuf 声明，`ReduceSum` 的工作空间参数按版本有差异，`GetValue` 回读标量在实际算子里通常要避免（见下）。以官方 `samples` 仓库的 RmsNorm 样例为准。

这段代码暴露了三个昇腾特有的性能陷阱：

**陷阱 1：标量回读（`GetValue`）会打断流水。** 第 ④ 步把结果读回 Scalar 单元算倒数，这会强制等待 Vector 流水排空。生产代码里要用 Vector 的 `Div` 或广播乘法避免回读。

**陷阱 2：规约（`ReduceSum`）是 Vector 上的昂贵操作。** 它有 log 级的步骤数，且中间要用额外的 UB。RMSNorm、Softmax、LayerNorm 这类算子在昇腾上都是 Vector Bound，第 12 章说的"峰值算力"对它们毫无意义——它们根本碰不到 Cube。

**陷阱 3：这类算子适合和前后的 GEMM 融合，但受 AIC/AIV 分离的限制。** 第 05 章讲 GPU 上 `GEMM + RMSNorm` 融合能省一次 HBM 往返；910B 上 GEMM 在 AIC、RMSNorm 在 AIV，中间结果**仍然要过 GM**。所以昇腾的做法是 CV 流水并行——切块后让 AIC 算第 k+1 块的 GEMM，AIV 同时处理第 k 块的 RMSNorm。

---

## 13.7 编译与运行

两条路，按目的选：

### 13.7.1 Kernel 直调（调试用）

不走算子工程，直接在一个 cpp 文件里调核函数，最快验证功能：

```cpp
// CPU 孪生调试：在 x86 上跑，能用 gdb、能 printf
ICPU_RUN_KF(add_custom, blockDim, x, y, z, workspace, tiling);

// NPU 上板运行
ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice,
                                workspace, tilingDevice);
```

**CPU 孪生调试是昇腾相对 CUDA 的一个明显优势**：同一份 kernel 代码能在 CPU 上以单核串行方式跑起来，可以单步、可以打印。CUDA 侧没有等价物（`cuda-gdb` 是在设备上调，体验差很多）。先在 CPU 上把功能调对，再上板调性能，是标准流程。

### 13.7.2 算子工程（交付用）

```bash
# 1) 用 msOpGen 从算子原型 JSON 生成工程骨架
msopgen gen -i add_custom.json -c ai_core-Ascend910B3 -lan cpp -out ./AddCustom

# 2) 目录结构
#    AddCustom/
#      op_host/      ← tiling 函数、算子原型注册（Host 侧）
#      op_kernel/    ← 核函数（Device 侧，就是 §13.3 那段）
#      CMakeLists.txt

# 3) 编译，产出自定义算子包
./build.sh

# 4) 安装到 CANN 的自定义算子目录
./custom_opp_<arch>.run

# 5) PyTorch 侧调用：通过 aclnn 接口或 torch_npu 的自定义算子注册
```

注意 `-c ai_core-Ascend910B3` 这个参数——**算子是按 soc_version 编译的**，910B3 编的包不能直接在 910B4 上跑。这在多型号混合的集群里是个真实的运维负担，和 CUDA 的 PTX 前向兼容机制很不一样。

---

## 13.8 调优：四类瓶颈的排查路径

第 12 章给了昇腾的四类瓶颈（Cube / Vector / MTE / 下发）。工具链上的完整路径：

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

指令流水图怎么读：

```
时间 →
MTE2  ████████████████████████████░░░░   占比 88%  ← 瓶颈在这
Cube  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  0%  ← 没用到（纯 Vector 算子）
Vector███████░░░░░░░░░░░░░░░░░░░░░░░░░   占比 24%
MTE3  ██████░░░░░░░░░░░░░░░░░░░░░░░░░░   占比 20%

判定：MTE Bound（搬入受限）
对策优先级：
  1. 确认 BUFFER_NUM=2 真的开了（最常见的低级错误）
  2. 调大 tileLength —— 单次搬运越大，MTE 效率越高（但受 UB 容量限制）
  3. 检查数据是否 32 Byte 对齐，不对齐会触发低效搬运
  4. 检查数据格式，ND ↔ NZ 的隐式转换是不是在这里发生
```

```
MTE2  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比 12%
Cube  ██████████████████████████████░░   占比 92%
Vector░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  3%

判定：Cube Bound —— 恭喜，这是理想状态（矩阵算子的目标）
对策：已接近硬件上限，剩下的收益在换更大 tiling 或提高 Cube 利用效率
```

```
MTE2  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  9%
Cube  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比  6%
Vector███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   占比 10%
（三级都不满，但整体耗时很长）

判定：下发 Bound —— 瓶颈在 Host 侧，不在算子里
对策：开图模式减少下发次数（见 [14 章 §14.5](14-GPU到NPU迁移实战.md)），或合并小算子
```

> 一个经验判断：昇腾上很大比例的"算子慢"，最后查出来根本不是算子本身慢，而是**调度下发开销**和**HBM 读写**。这和第 02 章讲 GPU 的诊断顺序（先看屋顶线定位 Compute/Memory Bound）不同——昇腾要先确认瓶颈是不是压根不在 Device 上。

---

## 13.9 Ascend C vs Triton vs CUDA：一张对照表

| 维度 | CUDA C++ | Triton | Ascend C |
|---|---|---|---|
| 语言 | C++ | Python DSL | C++ 模板库 |
| 抽象层级 | 线程级 | 块级（编译器管块内） | 块级 + 显式流水 |
| 并行单位 | thread / warp / block | program | **核（AI Core）** |
| 谁管片上内存 | 你（`__shared__`） | 编译器 | 你（`TPipe` + `TQue`） |
| 谁管搬运 | 你（或 `cp.async`） | 编译器 | 你（`DataCopy`） |
| 谁管流水重叠 | 你（软件流水） | 编译器 | 你（`BUFFER_NUM`） |
| 越界保护 | 手写 if | `mask=` | **无，靠 tiling 保证** |
| 分块参数在哪定 | kernel 启动参数 | `tl.constexpr` + autotune | **Host 侧 Tiling 函数** |
| 自动调优 | 无（手工或 CUTLASS） | `@triton.autotune` | 无内置，靠实测扫参 |
| CPU 上能调试吗 | 不能 | 不能（有 interpreter 模式） | **能（CPU 孪生调试）** |
| 写一个逐元素算子 | ~30 行 | ~7 行 | ~70 行 |
| 学习曲线 | 陡 | 平缓 | **最陡** |

结论很直白：**Ascend C 的开发效率明显低于 Triton，也低于 CUDA**。这是昇腾生态目前最真实的短板。缓解办法是尽量不写——优先用 `cann-ops-adv` 里已有的融合算子，选用阶梯见 [14 章 §14.2](14-GPU到NPU迁移实战.md)。

---

## 13.10 小结

```
CANN ≈ CUDA 全家桶；Ascend C ≈ CUDA C++（没有 Triton 那一层）

编程范式：CopyIn → Compute → CopyOut 三段式
  TPipe 管内存，TQue 管数据传递 + 同步，TBuf 管临时变量
  TPosition（VECIN/VECOUT/A1/B1/...）抽象掉物理存储

双缓冲 BUFFER_NUM=2 是基本功，不是优化
  耗时从 T搬入+T计算+T搬出 降到 max(三者)

Tiling 在 Host 侧算，是真正的工作量
  UB 容量硬约束要手算；核间负载均衡；尾块没有 mask 兜底

blockDim = 核数，不是每 block 线程数（撞名，极易误解）

调优：msprof 采集 → Insight 看指令流水图 → 四类 bound 对症下药
  很多"算子慢"其实是下发慢，先排除 Host 侧

优势：CPU 孪生调试；劣势：代码量和学习曲线明显高于 Triton

下一章：不自己写算子的那条路——昇腾现成的大模型融合算子有哪些、怎么用。
```

---

**延伸资料**：
- [CANN · Ascend C 编程范式](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0016.html)（官方，三段式的权威说明）
- [cann-ops-adv 融合算子库](https://gitee.com/ascend/cann-ops-adv)（开源，读 FlashAttention 类算子源码的入口）
- [msProf 算子调优工具概述](https://www.hiascend.com/document/detail/zh/mindstudio/700/ODtools/Operatordevelopmenttools/atlasopdev_16_0082.html)（官方）
- [msprof-analyze 自动瓶颈诊断](https://gitcode.com/Ascend/mstt)（开源，含 GPU/NPU 性能比对 recipe）
- [基于 Ascend C 的 FlashAttention 算子性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240607-1)（官方技术文章，讲清了 CV 流水并行的实操）
