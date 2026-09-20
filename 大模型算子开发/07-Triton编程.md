# 07 · Triton 编程：用 Python 写 GPU kernel 的正确方式

> **前置**：[01 章（GPU 硬件基础）](01-GPU硬件基础.md)、[02 章（屋顶线）](02-屋顶线与性能剖析.md)。强烈建议先读 [03 章（分块与双缓冲）](03-GEMM深度剖析.md)和 [05 章（融合账本）](05-算子融合.md)，本章的 `num_stages`、`BLOCK_SIZE`、`GROUP_SIZE_M` 三个旋钮全部是那两章概念的 Triton 化身。  
> **目标**：在脑子里建立"**我写的是一个 Block 的代码，不是一个 Thread 的代码**"这一个心智模型，然后从 vector add 一路演进到 fused softmax 和 matmul，最后能判断什么时候该用 Triton、什么时候该退回 CUDA。  
> **读法**：本章每引入一个新概念，都会先说它**对应硬件上的什么**，再给一个能拿纸笔验算的小例子。§7.2.3（元素怎么摊到线程上）、§7.3.1（mask 的逐 lane 推演）、§7.6.4（GROUP_SIZE_M 的 90 vs 54）这三处请一定自己算一遍。  
> **版本**：Triton 的 API 随版本变动很大。本章按 **Triton 3.3 及以上**校对，写作时 PyPI 最新版是 **3.8.0（2026-08）**。凡是版本敏感的写法都单独标了 ⚠️。

---

**本章关键词**

**Triton** —— 一门嵌在 Python 里的 GPU kernel 语言 + 编译器，最早由 Philippe Tillet 在 OpenAI 发布（2021），现在由社区维护在 `triton-lang/triton`，OpenAI、Meta、AMD、Intel、NVIDIA 都在提交代码。  
**DSL**（Domain-Specific Language，领域专用语言）—— 为某一类问题定制的小语言。Triton 是"写 GPU kernel 的 DSL"，它长得像 Python，但不是 Python。  
**Program（程序实例）** —— Triton 里并行的最小调度单位，**一个 program 就是 CUDA 里的一个 Block**。全书里凡是说"一个 program"，都可以直接读成"一个 Block"。  
**Block 级编程模型**（Block-level Programming）—— 本章唯一需要建立的心智模型：你写的每一行代码，描述的是**一整个 Block 对一整块数据**做什么，而不是一个线程对一个标量做什么。  
**tl**（Triton Language）—— Triton 的算子库，`import triton.language as tl`。`tl.load` / `tl.store` / `tl.dot` 是三个最核心的操作。  
**JIT**（Just-In-Time，即时编译）—— 第一次真正调用函数时才编译，而不是提前编好。`@triton.jit` 装饰的函数在首次以某组参数类型/常量调用时才会被编译。  
**constexpr**（Compile-time Constant Expression，编译期常量表达式）—— 标注为 `tl.constexpr` 的参数会被"烤进"生成的机器码里，每换一个值就重新编译一份 kernel。`BLOCK_SIZE` 必须是它。  
**mask（掩码）** —— `tl.load` / `tl.store` 的按元素开关，是 Triton 里 [01 章 §1.2.3](01-GPU硬件基础.md) 那个 `if (i < N)` 的替代品，**不写就会越界**。  
**IR**（Intermediate Representation，中间表示）—— 编译器内部的程序形态，介于源码和机器码之间。Triton 有两层自己的 IR：**TTIR**（Triton IR，还不知道 GPU 长什么样）和 **TTGIR**（TritonGPU IR，已经决定了数据怎么摊到线程上）。  
**MLIR**（Multi-Level Intermediate Representation，多层中间表示）—— LLVM 项目里的一套"造 IR 的框架"，Triton 的 TTIR / TTGIR 都是用它搭的。  
**LLVM IR** —— 通用编译器中间表示，Triton 的后端把 TTGIR 降到这里，再交给厂商后端生成汇编。  
**PTX**（Parallel Thread Execution，并行线程执行）—— NVIDIA 的虚拟汇编，跨代通用。  
**SASS**（Streaming ASSembler）—— 真正的 GPU 机器码，由 CUDA 工具链里的 `ptxas` 从 PTX 编译得到，绑定具体架构。  
**num_warps** —— 一个 program 用多少个 Warp（线程束）去执行。`num_warps=4` 就是 128 个线程。对应 CUDA 的 `blockDim`。  
**num_stages** —— 编译器给循环做**软件流水**的级数。`num_stages=2` 就是 [03 章 §3.5](03-GEMM深度剖析.md) 的**双缓冲**，更大的值就是多级流水。  
**autotune（自动调优）** —— `@triton.autotune`，把一组候选配置在真机上各跑一遍，挑最快的那个记下来。  
**MMA**（Matrix Multiply-Accumulate，矩阵乘累加）—— Tensor Core 的硬件指令。Triton 里 `tl.dot` 会被编译成 `mma.sync`（Ampere）或 `wgmma`（Hopper）。  
**TorchInductor** —— `torch.compile` 的默认 GPU 后端，它生成的就是 Triton 代码（CPU 侧生成 C++/OpenMP）。

---

## 7.1 先并排看一眼：同一个算子，CUDA 和 Triton 各写一遍

不讲道理，先看东西。任务就是 [01 章 §1.2.3](01-GPU硬件基础.md) 那个：`z[i] = x[i] + y[i]`，N = 1000。

**CUDA 版**：

```c
__global__ void add_kernel(const float* x, const float* y, float* z, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 我是第几个线程
    if (i < N) {                                     // 越界拦截
        z[i] = x[i] + y[i];                          // 我负责 1 个元素
    }
}
// 启动：add_kernel<<<4, 256>>>(x, y, z, 1000);
```

**Triton 版**：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                  # 我是第几个 Block
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 我负责哪 BLOCK 个元素（一个向量）
    mask = offs < N                              # 越界拦截（逐元素的开关）
    x = tl.load(x_ptr + offs, mask=mask)         # 一次读进 BLOCK 个
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)     # 一次写出 BLOCK 个

# 启动：add_kernel[(triton.cdiv(1000, 256),)](x, y, z, 1000, BLOCK=256)
```

逐行对照：

| CUDA | Triton | 差在哪 |
|---|---|---|
| `blockIdx.x` | `tl.program_id(0)` | **一模一样**，都是"第几个 Block" |
| `blockDim.x` | 没有对应物（由 `BLOCK` 和 `num_warps` 间接决定） | Triton 不让你看线程数 |
| `threadIdx.x` | **不存在** | 这是全章最大的分水岭 |
| `i`（一个标量下标） | `offs`（一个 BLOCK 长的**向量**下标） | 标量 → 向量 |
| `if (i < N)` | `mask = offs < N` | 标量分支 → 逐元素掩码 |
| `x[i]`（读 1 个 float） | `tl.load(x_ptr + offs, mask=mask)`（读 BLOCK 个） | 标量访存 → 块访存 |
| `<<<4, 256>>>` | `[(4,)]` + `BLOCK=256` | grid 照写，block 大小换了个位置 |

> ★ **两段代码启动的 Block 数量完全一样（都是 4 个），做的事也完全一样。唯一的区别是：CUDA 里你站在"一个线程"的位置上写代码，Triton 里你站在"一个 Block"的位置上写代码。** 这一句话就是整章的中心。下一节把它讲透。

---

## 7.2 心智模型：你写的是一个 Block 的代码，不是一个 Thread 的代码

### 7.2.1 两种视角，同一块硬件

硬件本身没变——还是 [01 章 §1.2.3](01-GPU硬件基础.md) 那三级：Grid → Block → Warp → Thread。变的是**你站在哪一级上写代码**。

```
硬件的实际层次                CUDA 让你站在哪         Triton 让你站在哪
──────────────────────────────────────────────────────────────────────
Grid（整块 GPU）
  │
  ├── Block  ← 一个 SM 上                            ★ 你在这里
  │     │                                              一行代码 = 一整块数据的操作
  │     ├── Warp（32 线程）
  │     │     │
  │     │     └── Thread   ★ 你在这里
  │     │                    一行代码 = 一个标量的操作
```

同一个"加法"，两种视角下脑子里的画面完全不同：

```
CUDA 视角（1024 个线程，每人捧着 1 个数）

  线程0   线程1   线程2   ...  线程1023
   [x0]    [x1]    [x2]         [x1023]
    +       +       +             +
   [y0]    [y1]    [y2]         [y1023]
    ↓       ↓       ↓             ↓
   [z0]    [z1]    [z2]         [z1023]

  你写的一行 z[i] = x[i] + y[i] 描述的是【其中一个人】干了什么。


Triton 视角（1 个 program，捧着一整条长度 1024 的向量）

  program 0
   [ x0 x1 x2 ............... x1023 ]     ← 一个 BLOCK 长的张量
                  +
   [ y0 y1 y2 ............... y1023 ]
                  ↓
   [ z0 z1 z2 ............... z1023 ]

  你写的一行 x + y 描述的是【整条向量】的操作。
  这条向量最后怎么摊到 32 / 128 / 256 个线程上去，是编译器的事，你看不到。
```

### 7.2.2 名词对照表

| CUDA 概念 | Triton 里的对应 | 说明 |
|---|---|---|
| Grid | 启动时的 `grid` 元组 | `kernel[(1024,)](...)` 或 `kernel[(gm, gn)](...)`，最多三维 |
| `gridDim.x/y/z` | `tl.num_programs(0/1/2)` | 一共有多少个 program |
| `blockIdx.x/y/z` | `tl.program_id(0/1/2)` | 我是第几个 |
| Block | **program（程序实例）** | 名字换了，东西是同一个 |
| `blockDim.x` | `num_warps × 32`（启动参数，kernel 里看不到） | 默认 `num_warps=4` → 128 线程 |
| `threadIdx.x` | **没有，也不该有** | 见 §7.2.3 |
| `__shared__` 数组 | **没有直接对应物** | 编译器按需自己分配，见 §7.2.4 |
| `__syncthreads()` | **没有，也不用写** | 编译器自动插屏障，见 §7.2.4 |
| `warpSize` / shuffle | **没有** | 归约用 `tl.sum` / `tl.max`，编译器决定用 shuffle 还是共享内存 |
| `if (i < N)` | `mask=` 参数 | 见 §7.3 |
| `__launch_bounds__` | `Config(..., maxnreg=...)` | ⚠️ Triton 3.x 的 `Config` 有 `maxnreg`，不是所有后端都支持 |

### 7.2.3 为什么没有 `threadIdx`：把 1024 个元素摊到 128 个线程上

这是新手最想不通的地方：**`BLOCK = 1024`，但硬件上只有 128 个线程（`num_warps=4`），那 1024 个元素到底谁处理谁？**

答案是**编译器决定**，而且它会挑一个能同时满足"合并访存"和"向量化指令"的摊法。以 `BLOCK=1024`、FP32、`num_warps=4`（128 线程）为例，编译器的典型选择是：**每个线程一次拿 4 个连续的 float（正好 16 字节 = 128 位，一条向量访存指令的宽度），128 个线程一轮覆盖 512 个元素，跑两轮覆盖 1024 个。**

| 线程号 | 第 1 轮拿的元素下标 | 第 2 轮拿的元素下标 |
|---|---|---|
| 0 | 0 – 3 | 512 – 515 |
| 1 | 4 – 7 | 516 – 519 |
| 2 | 8 – 11 | 520 – 523 |
| … | … | … |
| 31 | 124 – 127 | 636 – 639 |
| … | … | … |
| 127 | 508 – 511 | 1020 – 1023 |

**核对一下合并访存**（对照 [01 章 §1.4.1](01-GPU硬件基础.md)）：第 0 个 Warp 的 32 个线程，第 1 轮覆盖元素 0–127，也就是字节 0–511，正好 **16 个连续的 32 字节扇区，一个不多一个不少，效率 100%** ✓。

> ★ **这就是"为什么不给你 `threadIdx`"的答案**：元素到线程的映射是一件**性能攸关但纯机械**的事，人来做容易做错（[01 章 §1.4.3](01-GPU硬件基础.md) 那个"换个下标位置慢 8 倍"的例子），编译器来做既不会错又能跟着硬件代际自动变。Triton 把这个自由度收走，换来的是你再也写不出未合并的访存。

> ⚠️ 上面这张表是**按 Triton 的 blocked layout 规则推出来的典型结果，不是保证**。具体的 `sizePerThread`（每线程连续几个元素）由编译器根据 dtype、`num_warps`、访问模式和版本决定。要确认真实布局，用 `TRITON_KERNEL_DUMP=1` 把 TTGIR 打出来看 `#blocked` 属性（§7.9）。

### 7.2.4 编译器替你做了哪五件事

这五件事，正好是 01 章和 03 章里最折磨人的那几节：

| 编译器替你做的 | 对应本书哪一节 | Triton 里怎么做到的 |
|---|---|---|
| **① 合并访存** | [01 章 §1.4](01-GPU硬件基础.md) | `tritongpu-coalesce` pass：分析每个 `tl.load` 的地址表达式，挑一个让相邻 lane 落在相邻地址的 layout |
| **② bank conflict 规避** | [01 章 §1.5](01-GPU硬件基础.md)、[03 章 §3.4.3](03-GEMM深度剖析.md) | 需要经共享内存中转时（主要是 `tl.dot` 的操作数），编译器给它挑一个**带 swizzle 的共享内存布局**，不用你写 `[32][33]` |
| **③ 寄存器分配** | [01 章 §1.3](01-GPU硬件基础.md)（寄存器溢出） | 降到 LLVM IR 后由 NVIDIA 后端分配；`Config(maxnreg=...)` 可以给它设上限 |
| **④ 选 Tensor Core 指令** | [01 章 §1.7](01-GPU硬件基础.md)、[03 章 §3.6](03-GEMM深度剖析.md) | `tritongpu-accelerate-matmul` pass：把 `tl.dot` 换成带 MMA layout 的形式，后端发射 `mma.sync` 或 `wgmma` |
| **⑤ 插同步屏障** | [01 章 §1.3](01-GPU硬件基础.md)（`__syncthreads()` 的两个坑） | membar pass：分析共享内存的读写依赖，在必须同步的地方自动插屏障。[03 章 §3.3.4](03-GEMM深度剖析.md) 那两个"一个都不能省"的 `__syncthreads()`，在 Triton 里你根本不用想 |

再加一件半自动的：

| **⑥ 软件流水（双缓冲）** | [03 章 §3.5](03-GEMM深度剖析.md) | 不是全自动——你给 `num_stages`，编译器负责把循环拆成"预取 + 计算"重叠的形式，并发出 `cp.async` / TMA |

顺带说清楚**共享内存在 Triton 里到底什么时候会被用上**（原稿这里说得很含糊）：

```
tl.load 到的数据默认放在【寄存器】里，不经过共享内存。

编译器会动用共享内存的四种场合：
  ① tl.dot 的操作数          —— MMA 指令要求特定的片上布局，必须中转
  ② 跨 Warp 的归约            —— tl.sum / tl.max 跨越多个 Warp 时
  ③ 布局转换（layout convert）—— 前后两个算子想要的数据摆法不同
  ④ num_stages > 1 的流水缓冲 —— 预取的下一块 tile 要有地方放

前三种你控制不了，第四种由 num_stages 直接决定（§7.7.3）。
标准 Triton 语言里没有 __shared__ 的对应物，你不能手动开一块共享内存。
```

### 7.2.5 编译器**不**替你做的四件事

> ⚠️ **"Triton 帮你管掉了硬件细节"不等于"Triton 帮你写出了好算法"。** 下面四件事，全都得你自己来，而且它们决定了性能的上限：

| 你必须自己决定的 | 为什么编译器做不了 |
|---|---|
| **BLOCK 怎么切、grid 怎么分** | 这是算法设计。一行一个 program？一块 128×128 一个 program？编译器不知道你的数据语义 |
| **算法本身** | FlashAttention 的在线 softmax（[04 章 §4.2](04-FlashAttention全系.md)）是数学改写，不是编译优化。编译器不会帮你想出来 |
| **mask** | 忘了写 `mask=`，编译器不会警告，直接越界（§7.3） |
| **索引表达式本身是否连续** | 编译器能优化"元素→线程"的映射，**但改不了你写的地址公式**。你写 `tl.load(p + offs * 32)`，它照样是 [01 章 §1.4.2](01-GPU硬件基础.md) 的情况 B，12.5% 效率，编译器无能为力 |

最后一条特别容易被误解，单独强调一次：

> ⚠️ **Triton 不能把"跨步访问"变成"连续访问"。** 它保证的是"在你给定的地址表达式下，线程与元素的对应关系是最优的"。数据本身在 HBM 里就是跨着摆的（比如你要按列读一个行优先矩阵），那还是跨步，还是慢。这种情况下的正确做法和 CUDA 里一样：改数据布局，或者借片上存储中转（[01 章 §1.4.4](01-GPU硬件基础.md)）。

---

## 7.3 `mask`：Triton 版的 `if (i < N)`

### 7.3.1 一个 N 不整除 BLOCK 的完整推演

沿用 [01 章 §1.2.3](01-GPU硬件基础.md) 的设定：**N = 1000，BLOCK = 256**，所以 `grid = ceil(1000/256) = 4` 个 program。

先把四个 program 的 `offs` 范围列出来（`offs = pid * 256 + tl.arange(0, 256)`）：

| pid | `tl.arange(0, 256)` | `offs` 范围 | `mask = offs < 1000` | 情况 |
|---|---|---|---|---|
| 0 | 0 – 255 | 0 – 255 | 全 True | 全部有效 |
| 1 | 0 – 255 | 256 – 511 | 全 True | 全部有效 |
| 2 | 0 – 255 | 512 – 767 | 全 True | 全部有效 |
| 3 | 0 – 255 | 768 – 1023 | **前 232 个 True，后 24 个 False** | 后 24 个越界 |

把 program 3 的尾巴放大看——`mask` 是一个长度 256 的**布尔向量**，不是一个标量条件：

```
program 3 的 offs 向量（只画尾部 8 个位置）：

  向量位置:  248   249   250   251   252   253   254   255
  offs 值:   1016  1017  1018  1019  1020  1021  1022  1023
  < 1000?    False False False False False False False False
                ↑ 这 8 个位置的 load 不会发出请求，store 不会写出去

  向量位置:  230   231   232   233
  offs 值:   998   999   1000  1001
  < 1000?    True  True  False False
                          ↑ 分界线正好落在向量中间，不在 Warp 边界上
```

**注意最后一行**：分界线落在向量的第 232 个位置。这个位置属于第 7 个 Warp（232 ÷ 32 = 7.25）的中间，也就是说**同一个 Warp 里有些 lane 有效、有些无效**。这不是问题——掩码本来就是逐 lane 的，硬件天生支持（[01 章 §1.2.5](01-GPU硬件基础.md) 讲 Warp 分歧时说的"不该干活的线程用掩码屏蔽掉"，就是这个机制）。

> ★ **`mask` 和 CUDA 的 `if (i < N)` 是同一件事的两种写法**，都翻译成硬件上的 lane 谓词（predicate）。区别只在于：CUDA 里它长得像一个分支，Triton 里它长得像一个参数。**两者都不是可选的，漏了都会越界写显存**——而且都不报错，只是安静地写坏别人的数据。

### 7.3.2 `other=` 怎么选：三个场景，选错就是静默错误

被 mask 屏蔽掉的位置，`tl.load` 不会真的去读内存，但那个位置在返回的向量里**总得填点什么**。填什么由 `other=` 决定，默认是 `0`。

| 后面要拿它做什么 | `other=` 该填 | 填错会怎样 |
|---|---|---|
| 求和 / 累加 / `tl.dot` | `0.0` | 填别的会把垃圾加进和里 |
| 求最大值 `tl.max` | `-float('inf')` | **填 0.0 时，如果整行都是负数，最大值会错成 0** |
| 求最小值 `tl.min` | `float('inf')` | 同理 |
| 读一个权重向量，后面做乘法 | `0.0`（配合 store 的 mask） | 一般无所谓，因为写出去的时候也被 mask 挡了 |

> ⚠️ **求最大值那一行是本节最值钱的一句。** softmax 的第一步是 `x - max(x)`（[04 章 §4.2.5](04-FlashAttention全系.md) 讲过为什么非减不可）。如果一行的有效长度是 781、BLOCK 是 1024，尾部 243 个位置用 `other=0.0` 填，而这一行的真实数据恰好全是负数，那么 `tl.max` 会返回 0 而不是真正的最大值——**结果是错的，但不会报错，也不会 NaN，只是数值偏了**。官方 softmax 教程里写 `other=-float('inf')` 就是为了这个。

### 7.3.3 一个反复被问的问题：`mask` 会不会导致 Warp 分歧

不会——至少不会以 [01 章 §1.2.5](01-GPU硬件基础.md) 那种"两条路都走一遍"的方式。

```
Warp 分歧（01 章 §1.2.5）：
    if (cond) { A(); } else { B(); }
    → 两段【不同的指令序列】，硬件必须各执行一遍，耗时相加

mask：
    tl.load(ptr + offs, mask=m)
    → 只有【一条指令】，带一个 32 位的谓词寄存器
    → 谓词为 0 的 lane 这一拍不发访存请求，仅此而已
    → 不存在"两条路"，耗时不相加
```

代价只有一个：被屏蔽的 lane 这一拍白跑了（通道利用率下降）。N=1000/BLOCK=256 时，浪费 24/1024 = 2.3%，和 [01 章 §1.2.3](01-GPU硬件基础.md) 算出来的完全一样。

---

## 7.4 第一步：vector add —— 把骨架走通

### 7.4.1 完整代码（kernel + 启动器）

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, z_ptr,      # 三个张量的首地址（Triton 收到的是裸指针）
    n_elements,               # 运行期参数：向量长度
    BLOCK: tl.constexpr,      # 编译期常量：每个 program 处理多少个元素
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)     # 长度为 BLOCK 的下标向量
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.is_contiguous()
    z = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)               # cdiv = 向上取整除法
    add_kernel[grid](x, y, z, n, BLOCK=1024, num_warps=4)
    return z
```

四个语法点，都是后面两节要反复用的：

| 写法 | 含义 |
|---|---|
| `kernel[grid](...)` | 方括号里是 grid（一个元组，1–3 维），圆括号里是参数。这是 Triton 的启动语法 |
| 张量直接当指针传 | Triton 会自动取 `data_ptr()`。所以 kernel 里 `x_ptr + offs` 是**按元素**偏移，不是按字节 |
| `BLOCK: tl.constexpr` | 编译期常量。**必须是 2 的幂**（`tl.arange` 的硬性要求），每换一个值重编译一次 |
| `num_warps=4` | 启动参数，不是 kernel 的形参。决定这个 program 用多少线程（4×32 = 128） |

> ⚠️ **`tl.arange(0, BLOCK)` 要求 BLOCK 是 2 的幂**，这是 Triton 从头到尾的硬约束。所以处理"一行 781 个元素"这种形状时，标准做法是 `BLOCK = triton.next_power_of_2(781) = 1024`，再用 mask 把尾巴挡掉——不是"BLOCK 取 781"。

### 7.4.2 这段代码在硬件上对应什么

| 这一行 | 硬件上发生了什么 |
|---|---|
| `tl.program_id(0)` | 读 `%ctaid.x` 特殊寄存器，就是 CUDA 的 `blockIdx.x` |
| `tl.arange(0, 1024)` | **不产生任何指令**。它是编译期的形状信息，告诉编译器"接下来是一个长 1024 的张量" |
| `tl.load(x_ptr + offs, mask)` | 编译成若干条带谓词的向量访存（典型是 `ld.global.v4.b32`，每线程一次取 128 位），全部合并 |
| `x + y` | 若干条 `add.f32`，每个线程对自己手里那 8 个元素各加一次 |
| `tl.store(...)` | 对称的 `st.global.v4.b32` |

### 7.4.3 算一笔账：这个 kernel 能跑多快

**任务**：N = 1 亿个 FP32，H100 SXM5。

```
搬运量 = 读 x (400 MB) + 读 y (400 MB) + 写 z (400 MB) = 1.2 GB
计算量 = 1 亿次加法 = 0.1 GFLOP

算术强度 = 0.1e9 ÷ 1.2e9 = 0.083 FLOP/Byte
        ≪ 屋脊点 295（01 章 §1.3）→ 彻头彻尾的 Memory Bound

理论耗时 = 1.2 GB ÷ 3350 GB/s = 358 µs
理论算力占用 = (0.1e9 ÷ 358e-6) ÷ 67e12 = 279 GFLOPS ÷ 67 TFLOPS = 0.42%
              （对照 01 章 §1.3 的 y = x*2 是 0.63%：那里每元素搬 8 字节，
                这里搬 12 字节，算术强度低一些，占比也就低一些 ✓）
```

**再看波次**（[01 章 §1.2.4](01-GPU硬件基础.md)）：

```
grid = ceil(1e8 / 1024) = 97,657 个 program
假设每 SM 能同时驻留 8 个（num_warps=4 → 每 program 4 个 Warp，8×4 = 32 Warp，未超 64）
一个波次 = 132 SM × 8 = 1056 个 program
波次数 = ceil(97657 / 1056) = 93 个

最后一波只有 97657 − 92×1056 = 505 个 program（装了 48%）
机器利用率 = 97657 ÷ (93 × 1056) = 99.4%    ← 尾巴被 93 个波次摊薄了，可以忽略
```

> ★ **写到这里这个 kernel 就完了——没有任何优化空间。** 它已经是 100% 合并访存、100% 带宽受限、尾巴可以忽略。你能做的只剩"别再多读写一遍"，也就是 [05 章](05-算子融合.md)的融合。**这正好说明 Triton 的价值在哪：它不能让 Memory Bound 的算子超过屋顶线，它只是让你用十行 Python 就稳稳地贴到屋顶线上。**

---

## 7.5 第二步：fused softmax —— 新增"行内归约"和"一个 program 一整行"

从 vector add 到 softmax，只加两个新概念：**归约**（`tl.max` / `tl.sum`）和**一个 program 负责一整行**。

### 7.5.1 先算账：为什么值得融

Softmax 按定义写是五步：求最大值 → 减最大值 → 取指数 → 求和 → 除。如果每步一个 PyTorch 算子，HBM 往返就是这样（记一份 `[M, N]` 张量为 `T`，[05 章](05-算子融合.md)的记法）：

| 步骤 | 读 | 写 |
|---|---|---|
| `m = x.max(dim=-1)` | 1 T | ~0（结果只有 M 个数） |
| `t = x - m` | 1 T | 1 T |
| `e = t.exp()` | 1 T | 1 T |
| `s = e.sum(dim=-1)` | 1 T | ~0 |
| `out = e / s` | 1 T | 1 T |
| **合计** | **5 T** | **3 T** |

融合成一个 kernel 之后：**读 1 T、写 1 T，合计 2 T**。

代进真实形状（M = 4096 行，N = 4096 列，BF16，`T = 4096 × 4096 × 2 B = 33.55 MB`）：

```
不融合：8 T = 268.4 MB ÷ 3350 GB/s = 80.1 µs
融  合：2 T =  67.1 MB ÷ 3350 GB/s = 20.0 µs
                                     ────────
                                     快 4.0 倍
```

**这 4 倍不来自任何算法技巧，只来自"中间结果不写出去"。** 和 [05 章 §5.2](05-算子融合.md) 那条 LayerNorm 链是同一个故事。

### 7.5.2 代码：最朴素的一行一 program 版本

```python
@triton.jit
def softmax_kernel(
    out_ptr, in_ptr,
    in_row_stride, out_row_stride,   # 相邻两行之间隔多少个【元素】
    n_cols,
    BLOCK: tl.constexpr,             # 必须 >= n_cols，且是 2 的幂
):
    row = tl.program_id(0)                      # 一个 program 负责一整行
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # ① 唯一一次从 HBM 读这一行
    row_ptr = in_ptr + row * in_row_stride
    x = tl.load(row_ptr + cols, mask=mask, other=-float('inf'))
    #                                      ↑ 见 §7.3.2：求 max 必须填 -inf

    # ② 三步归约 + 逐元素运算，全程在片上，一个字节都不落 HBM
    x = x - tl.max(x, axis=0)                   # 减最大值（04 章 §4.2.5）
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)

    # ③ 唯一一次写回 HBM
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    num_warps = 8 if BLOCK >= 2048 else 4        # 行越长，多派点线程
    softmax_kernel[(n_rows,)](                   # grid = 行数，一行一个 program
        out, x, x.stride(0), out.stride(0), n_cols,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out
```

### 7.5.3 新概念：归约在硬件上是怎么做的

`tl.max(x, axis=0)` 这一行在源码里只有 14 个字符，编译出来是一棵**两级归约树**：

```
设 BLOCK = 1024，num_warps = 4（128 线程），每线程手里 8 个元素。

第一级：线程内归约（纯寄存器，最便宜）
    每个线程把自己手里 8 个元素两两比，得 1 个局部最大值
    128 个线程 → 128 个局部值        代价：8 条 max 指令

第二级 a：Warp 内归约（用 shuffle 指令，不经过内存）
    一个 Warp 的 32 个局部值 → 1 个
    用 __shfl_xor_sync 做 5 轮（log2(32) = 5）
    4 个 Warp → 4 个值               代价：5 条 shuffle + 5 条 max

第二级 b：跨 Warp 归约（这里才动用共享内存）
    4 个 Warp 各把自己的结果写进共享内存（4 个 float = 16 字节）
    插一个屏障
    再读回来，4 个值归约成 1 个，广播给全部 128 个线程

  ★ 这三步、那个屏障、那 16 字节共享内存，你一行都没写。
    在 CUDA 里这是一个至少 30 行、且极容易写错的 blockReduce 模板。
```

> ★ **这就是 Triton 最实在的省力之处**：归约是 GPU 编程里最容易写错的一类代码（错一个屏障就是随机结果），而 Triton 把它压成了一个函数调用。`tl.sum` / `tl.max` / `tl.min` / `tl.argmax` / `tl.cumsum` 都是同一套机制。

> ⚠️ 上面"5 轮 shuffle + 共享内存中转"是**按归约树原理推出的典型形态**，具体用几轮 shuffle、是否走共享内存，取决于 `num_warps` 和编译器版本。要确认请看 TTGIR / PTX（§7.9）。

### 7.5.4 这个写法的硬约束：一整行必须装进片上

`BLOCK >= n_cols` 意味着**一整行必须同时活在一个 program 的寄存器里**。算一下预算：

```
N = 4096，BF16 输入，Triton 内部 exp 走 FP32：
  一行转成 FP32 = 4096 × 4 B = 16 KB
  同时活着的中间量 x / num / out 大约 2–3 份
  → ⚠️ 约 32–48 KB / program（估算，编译器会复用，实际更少）

寄存器文件 = 256 KB/SM（01 章 §1.2.2）
→ 一个 SM 大概放得下 5–8 个这样的 program

N = 32768 时：一行 FP32 就是 128 KB，加中间量直接爆掉 ✗
```

> ⚠️ **N 太大时这个写法会直接编译失败或性能塌方**（症状是 `out of resource: shared memory`，或者编译通过但寄存器大量溢出，[01 章 §1.3](01-GPU硬件基础.md) 的 spill）。出路是退回**两趟扫描**：第一趟只扫一遍求 max 和 sum，第二趟再扫一遍做除法。代价是 x 要读两遍，融合收益从 4× 降到约 2.7×（8 T → 3 T）。和 [05 章 §5.7](05-算子融合.md) 末尾 LayerNorm 遇到的是同一个墙。**更好的出路是在线算法**——一趟就把 max 和 sum 同时算出来，那正是 [04 章 §4.2](04-FlashAttention全系.md) 的在线 Softmax。

### 7.5.5 进阶：persistent kernel，以及一次 Occupancy 手算

上面的写法是 `grid = (n_rows,)`——行数多的时候会启动几万个 program，每个只干一点点活，[01 章 §1.8.2](01-GPU硬件基础.md) 说的**下发成本**开始显眼。官方教程用的是 **persistent kernel（常驻 kernel）**：只启动"刚好填满一个波次"的 program，每个 program 内部用循环去领活。

```python
@triton.jit
def softmax_persistent(out_ptr, in_ptr, in_stride, out_stride, n_rows, n_cols,
                       BLOCK: tl.constexpr, NUM_STAGES: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)                       # 一共启动了多少个 program
    for row in tl.range(row_start, n_rows, row_step, num_stages=NUM_STAGES):
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float('inf'))
        x = x - tl.max(x, axis=0)
        num = tl.exp(x)
        tl.store(out_ptr + row * out_stride + cols,
                 num / tl.sum(num, axis=0), mask=mask)
```

这就是 [01 章 §1.2.4](01-GPU硬件基础.md) 提到的 **Grid-Stride Loop（网格跨步循环）**。关键问题变成：**应该启动多少个 program？** 答案是"刚好填满一个波次"，而这要靠 Occupancy 反推。

**手算一遍**（H100，`num_warps=8`，假设编译后每线程 32 个寄存器、每 program 用 16 KB 共享内存）：

```
① 寄存器能装几个 program
   每 program 用的寄存器 = 32 × 32 × 8 = 8192 个
                            ↑每线程 ↑每Warp ↑Warp数
   65536 ÷ 8192 = 8 个 program/SM

② 共享内存能装几个
   228 KB ÷ 16 KB = 14 个 program/SM

③ Warp 上限
   64 ÷ 8 = 8 个 program/SM         ← 01 章 §1.2.2 的每 SM 64 Warp 上限

取最小值：min(8, 14, 8) = 8 个 program/SM
驻留 Warp = 8 × 8 = 64 → Occupancy = 100%

一个波次 = 132 SM × 8 = 1056 个 program
→ grid = min(1056, n_rows)
```

官方教程正是这么做的：先 `kernel.warmup(...)` 编译一遍，从编译产物里读出 `n_regs` 和 `metadata.shared`，再代入上面这个式子算 `num_programs`。

> ⚠️ 官方教程用到的 `kernel.warmup()` / `kernel._init_handles()` / `kernel.n_regs` / `kernel.metadata.shared` 是**偏内部的 API，带下划线的那个尤其不稳定**，不同 Triton 版本改过。生产代码里建议把算出来的 grid 数写死或做成配置项，不要在运行时反射编译产物。

> ⚠️ 官方教程的那个 occupancy 公式**只除了寄存器和共享内存，没有对 64 Warp 上限做 min**。`num_warps` 小的时候它可能算出超过硬件上限的值。自己用的时候记得补上第 ③ 条。

---

## 7.6 第三步：matmul —— 新增"二维指针"、`tl.dot`、K 循环、L2 分组

这一步一次加了四个概念，但每一个都在 [03 章](03-GEMM深度剖析.md)见过原型。

### 7.6.1 二维指针：`[:, None]` 和 `[None, :]` 到底在干什么

这是 Triton 里唯一需要动一下脑筋的语法。`offs_m[:, None]` 把一个长度 BM 的向量变成 `BM × 1` 的矩阵，`offs_k[None, :]` 把长度 BK 的向量变成 `1 × BK`，两者相加触发**广播**，得到一个 `BM × BK` 的地址矩阵。

**手算一遍。** 设 A 是行优先的 `M × K` 矩阵，`stride_am = K = 4`，`stride_ak = 1`；取 `BM = 2`，`BK = 3`，`pid_m = 0`，`k = 0`：

```
offs_m = tl.arange(0, 2)            = [0, 1]
offs_k = tl.arange(0, 3)            = [0, 1, 2]

offs_m[:, None] * stride_am  =  [[0],     * 4  =  [[0],
                                 [1]]              [4]]

offs_k[None, :] * stride_ak  =  [[0, 1, 2]] * 1 = [[0, 1, 2]]

两者相加（广播）：
                 [[0],        [[0, 1, 2]]      [[0, 1, 2],
                  [4]]    +                =    [4, 5, 6]]

a_ptrs = a_ptr + 上面这个 2×3 的偏移矩阵
```

对照一下 A 在内存里的实际摆法（`A[r][c]` 在第 `r*4 + c` 个元素），确认无误：`A[0][0..2]` 在 0,1,2；`A[1][0..2]` 在 4,5,6 ✓。

> ★ **一个记忆口诀**：`[:, None]` 管"行走哪一维"，`[None, :]` 管"列走哪一维"，各自乘上对应的 stride。这个模式在本章、在所有 Triton GEMM/Attention 代码里会出现几十次，记熟它，剩下的都是体力活。

### 7.6.2 `tl.dot` 和 03 章三级分块的对应关系

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
    acc = tl.dot(a, b, acc)                    # 累加进 acc，不是 acc += tl.dot(a, b)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```

这段循环就是 [03 章 §3.3.4](03-GEMM深度剖析.md) 那段伪代码的实现。对照表：

| 03 章的概念 | Triton 里是谁 | 谁决定 |
|---|---|---|
| **Block Tile**（BM × BN × BK） | `BLOCK_M` / `BLOCK_N` / `BLOCK_K` | **你**（或 autotune） |
| **Warp Tile**（WM × WN） | 无对应写法 | **编译器**（由 `num_warps` 间接影响） |
| **Thread Tile**（TM × TN） | 无对应写法 | **编译器** |
| 搬进共享内存 + `__syncthreads()` | 无对应写法 | **编译器**（`tl.dot` 的操作数自动中转 + 自动插屏障） |
| 共享内存 swizzle（§3.4.3） | 无对应写法 | **编译器** |
| 双缓冲 / N 级流水（§3.5） | `num_stages` | **你**（给个数字，编译器实现） |
| Tensor Core 的 `mma` / `wgmma` | `tl.dot` | **编译器**选指令 |
| Threadblock Swizzle（§3.3.6） | `GROUP_SIZE_M` + 手写的 pid 重排 | **你**（§7.6.3） |

> ★ **三级分块里，Triton 只让你管第一级。** 03 章里那个"Thread Tile 为什么正好是 8×8"的精妙推算（§3.4.2），在 Triton 里你既不用算也没法控制——这既是省力，也是天花板所在（§7.11）。

**三个必须知道的 `tl.dot` 约束**：

| 约束 | 说明 |
|---|---|
| **M、N、K 三个维度都必须 ≥ 16** | 这是 Tensor Core MMA 指令形状决定的（[01 章 §1.7.2](01-GPU硬件基础.md)）。`[16,8] × [8,16]` 会因为 K=8 报错 |
| **没有矩阵—向量路径** | `BLOCK_N = 1` 的 GEMV 用不了 `tl.dot`，得写成 `tl.sum(a * b[None, :], axis=1)` |
| **累加器用 FP32** | `tl.zeros(..., dtype=tl.float32)`。BF16 累加会掉精度，而 Tensor Core 本来就支持 FP32 累加，不花钱 |

> ⚠️ FP8 输入在部分后端要求 K ≥ 32，不是 16。⚠️ `tl.dot` 的 `allow_tf32=True` 参数在新版本里**已标记为废弃**，等价写法是 `input_precision="tf32"`（NVIDIA 后端的默认值就是 `"tf32"`；要严格 FP32 精度得显式写 `input_precision="ieee"`）。

### 7.6.3 `GROUP_SIZE_M`：03 章 §3.3.6 那个 L2 优化的 Triton 版

[03 章 §3.3.6](03-GEMM深度剖析.md) 说过：**Block 的执行顺序会显著影响 L2 命中率**，按行扫描是最差的选择。Triton 官方 GEMM 教程把这件事交给你手写，参数叫 `GROUP_SIZE_M`。

**把账算出来。** 设 C 被切成 9 × 9 = 81 个块，一个波次能同时跑 9 个 program。问：这 9 个 program 一共要从 HBM 读进多少个 A/B 的小块？

```
方案一：按行扫描（pid_m = pid // 9, pid_n = pid % 9）
  这 9 个 program 覆盖 C 的第 0 行，一整行 9 个块

  要的 A 块：C 第 0 行只用到 A 的第 0 条横条 → 1 行 × 9 个 K 块 =  9 块
  要的 B 块：用到 B 的全部 9 条竖条        → 9 列 × 9 个 K 块 = 81 块
                                                              ───────
                                                        合计   90 块

方案二：按 3×3 分组扫描（GROUP_SIZE_M = 3）
  这 9 个 program 挤在 C 的一个 3×3 方块里

  要的 A 块：3 条横条 × 9 个 K 块 = 27 块
  要的 B 块：3 条竖条 × 9 个 K 块 = 27 块
                                    ───────
                              合计   54 块

  90 ÷ 54 = 1.67 倍的 HBM 读量差距
```

> ★ **同样的 9 个输出块，换个遍历顺序，要读的输入少了 40%。** 一行算法都没改，纯粹是让同时在跑的那批 Block 挤在一个方块里，互相蹭 L2。官方教程报告在某些架构上这一招带来 **10% 以上**的端到端提速。这就是 [03 章 §3.3.6](03-GEMM深度剖析.md) 那张"按行扫描 vs 4×2 分组扫描"图的可执行版本。

实现（这段是模板，照抄就行，但要看懂它在算什么）：

```python
pid = tl.program_id(axis=0)                    # 注意：grid 是一维的
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n    # 一个"组"里有多少个 program
group_id = pid // num_pid_in_group             # 我在第几组
first_pid_m = group_id * GROUP_SIZE_M          # 本组从 C 的第几行块开始
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)   # 最后一组可能不满
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

> ⚠️ `group_size_m` 那一行的 `min` 不能省：`num_pid_m` 不是 `GROUP_SIZE_M` 的整数倍时，最后一组会短一截，不处理就会算出越界的 `pid_m`。

### 7.6.4 完整的 matmul kernel

```python
@triton.autotune(
    configs=[
        # 注意：num_stages / num_warps 是 Config 的【独立关键字参数】，
        #       不能塞进第一个字典里（§7.8.1 讲这个坑）
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
                      num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],          # 这三个值一变就重新调优（§7.8.2）
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ── ① 分组重排 pid，为了 L2 命中率（§7.6.3）
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ── ② 构造 A / B 的二维指针（§7.6.1）
    #    这里用 % M 和 % N 把越界行列绕回有效范围，避免读到映射外的地址；
    #    读进来的是"无所谓的值"，最后 store 时再用 c_mask 挡掉不写。
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # ── ③ K 方向主循环（03 章 §3.3.4），编译器按 num_stages 给它做软件流水
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)                  # 累加器传进去，省一次加法
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ── ④ Epilogue：累加器还在寄存器里，这里可以顺手做激活/加 bias/量化（03 章 §3.7.1）
    c = acc.to(tl.float16)

    # ── ⑤ 写回，这里才是真正需要边界 mask 的地方
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # autotune 会改 BLOCK_M/BLOCK_N，所以 grid 必须写成接收 META 的 lambda
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )
    return c
```

> ⚠️ **第 ② 步那两个 `% M` / `% N` 不是可有可无的。** 如果直接写 `offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`，而 K 方向的 mask 又只检查了 K（这是最常见的写法），那么最后一个 M 块里超出 M 的那些行会用**越界的地址**去 `tl.load`——mask 没挡住这一维。**这是越界读，属于未定义行为**，多数时候读到相邻内存不崩溃，所以极难发现。用 `% M` 把它们折回有效行，读到的值虽然是错的，但 ⑤ 步的 `c_mask` 保证它们不会被写出去。

> ⚠️ **grid 必须写成 `lambda META: ...`。** 配了 autotune 之后 `BLOCK_M` 不再是固定值，写死的 grid 元组会和实际选中的配置对不上，结果是算漏或算重。这是 autotune 最高频的踩坑点。

### 7.6.5 该不该用 Triton 写 GEMM

官方 GEMM 教程在方阵上能跑到和 cuBLAS 几乎持平的水平（教程自己的 benchmark 里 4096³ 是 Triton 217 TFLOPS vs cuBLAS 219 TFLOPS）。

> ⚠️ **这个数字不能直接拿来用**：官方教程页面没有明确标注跑这组数据的卡型（同页另一处提到的 220–245 TFLOPS 是 A100 上的结果），而 H100 的 BF16 稠密峰值是 989 TFLOPS，[03 章 §3.8.2](03-GEMM深度剖析.md) 里 cuBLAS 在 4096³ 上的量级是 ~875 TFLOPS。**217 这个数和 H100 对不上，多半是 A100 或者非 Hopper 优化路径的结果。** 引用性能数字之前先确认硬件——这是本系列反复强调的纪律。

本系列的一贯建议不变（[03 章 §3.8.3](03-GEMM深度剖析.md)）：

- **标准形状的 GEMM 用 cuBLAS**，Triton 重写通常打不过，维护成本还高。
- **需要非标准 Epilogue 或非标准形状**（分组 GEMM、稀疏、自定义量化 scale、fused MoE 路由）时用 Triton——这时候 cuBLAS 根本没有对应的 kernel，"慢 10%"和"没有"之间没得比。
- 上面这段 matmul 代码的真正用途是**当模板**：把 ④ 步换成你要的 Epilogue，就是一个新算子。

---

## 7.7 三个旋钮：`BLOCK_*` / `num_warps` / `num_stages` 各在调什么

这三个参数经常被一起写在 Config 里，但它们调的是**三件完全不同的事**。

| 旋钮 | 调的是什么 | 对应本书哪一节 | 调大的好处 | 调大的代价 |
|---|---|---|---|---|
| `BLOCK_M/N/K` | 一个 program 干多少活 | [03 章 §3.3](03-GEMM深度剖析.md) 分块 | 算术强度上升，HBM 流量下降 | 寄存器 + 共享内存吃紧；program 数变少，可能填不满 SM |
| `num_warps` | 这些活派多少线程去干 | [01 章 §1.6](01-GPU硬件基础.md) Occupancy | 每线程分到的元素少、寄存器压力小 | 跨 Warp 归约变贵；program 数一定时驻留数下降 |
| `num_stages` | 搬运和计算重叠几级 | [03 章 §3.5](03-GEMM深度剖析.md) 双缓冲 | HBM 延迟被藏住 | 共享内存 **× num_stages**，很容易爆 |

### 7.7.1 `BLOCK_*`：直接照搬 03 章的公式

[03 章 §3.3.2](03-GEMM深度剖析.md) 那条公式在 Triton 里一字不改地成立：

```
算术强度 = 调和平均(BLOCK_M, BLOCK_N) ÷ 每元素字节数

BLOCK_M = BLOCK_N = 128，BF16（2 B）：
    调和平均 = 2 ÷ (1/128 + 1/128) = 128
    算术强度 = 128 ÷ 2 = 64 FLOP/Byte
    对比屋脊点 295 → 还差 4.6 倍，靠 L2 和 GROUP_SIZE_M 补（§7.6.3）
```

注意 `BLOCK_K` **不影响 HBM 总流量**（[03 章 §3.3.5](03-GEMM深度剖析.md) 的反直觉结论），它只影响共享内存占用和流水线粒度——这也是为什么官方配置表里 `BLOCK_K` 在 32/64/128 之间变来变去。

### 7.7.2 `num_warps`：同样的活，派多少人

固定 `BLOCK = 1024`、FP32，看 `num_warps` 变化时每个线程的负担：

| `num_warps` | 线程数 | 每线程元素数 | 光装数据就要的寄存器 | 仅按 64 Warp 上限算，每 SM 能驻留几个 program |
|---|---|---|---|---|
| 1 | 32 | 32 | 32 个 | 64 |
| 2 | 64 | 16 | 16 个 | 32 |
| **4（默认）** | **128** | **8** | **8 个** | **16** |
| 8 | 256 | 4 | 4 个 | 8 |
| 16 | 512 | 2 | 2 个 | 4 |
| 32 | 1024 | 1 | 1 个 | 2 |

两头都有毛病：

```
num_warps 太小（1–2）：
  每线程要扛几十个元素 → 寄存器压力大 → 可能寄存器溢出（01 章 §1.3）
  好处：归约几乎不用跨 Warp，共享内存和屏障都省了

num_warps 太大（16–32）：
  每线程只有一两个元素 → 指令级并行不足，循环/地址开销摊不薄
  跨 Warp 归约要走共享内存 + 屏障，代价随 Warp 数上升
  一个 Block 最多 1024 线程（01 章 §1.2.2），所以 num_warps ≤ 32 是硬上限
```

经验起点：**逐元素算子用默认的 4；行内归约且行很长（BLOCK ≥ 2048）用 8；GEMM 类大 tile 用 8**。但真正的答案是交给 autotune（§7.8）。

### 7.7.3 `num_stages`：就是 03 章的双缓冲，而且共享内存按它翻倍

`num_stages` 直接对应 [03 章 §3.5](03-GEMM深度剖析.md)：

```
num_stages = 1   单缓冲：搬 → 算 → 搬 → 算，一半时间在等         （03 章 §3.5.1）
num_stages = 2   双缓冲：搬下一块的同时算这一块                   （03 章 §3.5.2 那张时间线）
num_stages = 3+  多级流水：同时有好几块在路上，抖动也能吸收       （03 章 §3.5.3 的 CUTLASS Stages）

底层靠的是同一套硬件机制：Ampere 的 cp.async、Hopper 的 TMA。
你不用写 __pipeline_memcpy_async，编译器按 num_stages 展开循环并插指令。
```

**代价是共享内存乘以 stages 数，这是最容易爆的地方。** 算一笔账（H100，每 Block 最大共享内存 228 KB）：

```
配置：BLOCK_M = 128, BLOCK_N = 256, BLOCK_K = 64，BF16（2 字节）

每一级流水要缓存的 tile：
    A 块 = 128 × 64 × 2 B = 16,384 B
    B 块 =  64 × 256 × 2 B = 32,768 B
                             ─────────
    一级合计               = 49,152 B = 48 KB

num_stages = 2  →  96 KB   ✓
num_stages = 3  → 144 KB   ✓   ← 官方这组配置用的就是 3
num_stages = 4  → 192 KB   ✓   勉强
num_stages = 5  → 240 KB   ✗   超过 228 KB，编译报错
```

> ⚠️ **超了会得到一个运行期错误**，形如 `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 245760, Hardware limit: 232448`。看到它就是 `num_stages` 或 `BLOCK_*` 开大了。注意报的是**字节**：232448 B = 227 KB，就是 H100 单 Block 共享内存上限（本系列表里的 228 KB 是这个数的取整说法）。

> ★ **一句话分工**：`BLOCK_*` 决定"要搬多少"（带宽问题），`num_stages` 决定"搬的时候在不在干活"（延迟问题），`num_warps` 决定"干活的人手"（并行度问题）。[03 章 §3.5.4](03-GEMM深度剖析.md) 那句"双缓冲藏延迟，藏不掉带宽不够"在 Triton 里同样成立——`num_stages` 调到 8 也救不了一个 `BLOCK` 开太小的 kernel。

---

## 7.8 autotune：怎么用，以及三个必踩的坑

### 7.8.1 正确的写法（原稿这里是错的）

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8},
                      num_stages=3, num_warps=8),
        #             └── 第一个参数：传给 kernel 的 constexpr 元参数（字典）
        #                                       └── 这两个是 Config 自己的关键字参数
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

> ⚠️ **`num_warps` 和 `num_stages` 绝对不能写进第一个字典里。** `triton.Config` 的签名是 `Config(kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None, ir_override=None)`——字典只装"传给 kernel 的 constexpr 参数"。把 `'num_warps': 8` 塞进字典，它要么被当成 kernel 不认识的形参，要么被 Config 自己的默认值 4 覆盖掉；无论哪种，**你以为调了 8 个 Warp，实际跑的是 4 个**，而且不报错。本章原稿就是这么写的，是一个静默失效的配置错误。

> ⚠️ 装饰器顺序固定：`@triton.autotune` 在**外**，`@triton.jit` 在**内**。反了不生效。

### 7.8.2 `key=` 的语义：什么时候重新调优

`key` 是一个**参数名列表**。autotuner 把这些参数的当前取值组成一个元组当缓存键——**键没见过就把所有 config 跑一遍挑最快的，见过就直接用记下来的那个**。

```
key=['M', 'N', 'K'] 时：

  第 1 次 matmul(4096×4096, 4096×4096)
      → 键 (4096, 4096, 4096) 没见过
      → 把 4 个 config 各编译 + benchmark 一遍，挑最快的记下
      → 这次调用很慢（编译 + 试跑，⚠️ 量级通常是几百毫秒到几秒，随 config 数和形状而定）

  第 2 次 同样形状
      → 命中缓存，直接跑，零额外开销

  第 3 次 matmul(4096×4096, 4096×8192)
      → 键 (4096, 8192, 4096) 是新的 → 又要全跑一遍
```

### 7.8.3 三个坑

**坑一：`key` 选太细，变成每次调用都在调优。**

> ⚠️ 变长序列（[04 章 §4.8.1](04-FlashAttention全系.md) 的 varlen）场景下，如果 `key=['seq_len']` 而 `seq_len` 每个 batch 都不同，那就是**每个 batch 都重新 benchmark 一遍全部 config**，比不调优还慢几个数量级。解法：要么把 `seq_len` 从 key 里拿掉（用一个对长度不敏感的配置），要么先把长度分桶（比如向上取整到 512 的倍数）再传进去。

**坑二：以为结果自动存盘了。**

> ⚠️ **默认情况下 autotune 的结果只活在进程内存里，进程一退就没了。** 原稿说的 `~/.triton/autotune_cache` 这个路径**不存在**——`~/.triton/cache` 是编译产物（PTX/cubin）的缓存，和 autotune 的计时结果是两回事。Triton 3.3 起 `triton.autotune` 有一个 `cache_results=True` 参数可以把计时写到磁盘，但**默认是 `False`**。所以：服务每次重启，第一批请求都会付一遍调优的钱。生产做法是服务启动时用代表性形状预热一遍，或者干脆跳过 autotune、把实测最优的配置硬编码进 `@triton.jit` 的调用参数里。

**坑三：config 列表太长。**

```
config 数 × 不同形状数 = 要跑的 benchmark 次数

4 个 config × 3 种形状  = 12 次   ✓ 可以接受
30 个 config × 50 种形状 = 1500 次 ✗ 启动阶段能卡上几分钟
```

实用做法是先用大列表离线扫一遍，看清楚哪几个配置在你的形状区间里真的领先，再把线上的列表砍到 3–5 个。`prune_configs_by={'early_config_prune': ...}` 可以按共享内存预算之类的规则提前剪掉必然失败的配置。

> ★ 调试时打开 `TRITON_PRINT_AUTOTUNING=1`，它会在每个 kernel 调优结束后打印**胜出的配置和总耗时**。这是确认"autotune 到底选了什么"的唯一可靠手段——不要靠猜。

---

## 7.9 编译流水线与调试：怎么看见编译器干了什么

### 7.9.1 五级降级路径

原稿这里只画了三级且跳过了 MLIR，补全如下：

```
@triton.jit 装饰的 Python 函数
        │  遍历 Python AST（抽象语法树），不是解释执行
        ▼
TTIR   (Triton IR)          —— 还是"块"的世界：张量有形状，但不知道有几个线程
        │  tritongpu-coalesce / accelerate-matmul / pipeline / membar ... 一串 MLIR pass
        ▼
TTGIR  (TritonGPU IR)       —— 已经决定了 layout：谁拿哪些元素、共享内存怎么 swizzle
        │
        ▼
LLVM IR                     —— 通用中间表示，已经是"线程"的世界了
        │
        ▼
PTX    (虚拟汇编)            —— 这里能 grep 到 mma / wgmma / cp.async
        │  ptxas（CUDA 工具链自带，Triton 直接调用，不经过 nvcc）
        ▼
cubin / SASS (机器码)        —— 绑定具体架构，最终执行的东西
```

> ★ **`tl.arange`、`tl.load`、`mask`、layout 这些概念全部活在 TTIR / TTGIR 这两层。** 想知道"编译器到底把我的 1024 个元素怎么分的"，就去看 TTGIR 里的 `#blocked` / `#shared` / `#mma` 属性；想知道"有没有用上 Tensor Core"，就去 PTX 里 grep。

### 7.9.2 环境变量（全部出自官方 README）

| 环境变量 | 作用 |
|---|---|
| `TRITON_KERNEL_DUMP=1` | 把每一级 IR 和最终的 PTX 全部导出（配合 `TRITON_DUMP_DIR=某个目录`） |
| `MLIR_ENABLE_DUMP=1` | 每个 MLIR pass 前都 dump 一次 IR，看 pass 之间的变化（配合 `MLIR_DUMP_PATH`） |
| `LLVM_IR_ENABLE_DUMP=1` | 每个 LLVM pass 前 dump 一次 LLVM IR |
| `TRITON_INTERPRET=1` | 走**解释器**而不是 GPU，可以在 kernel 里下 Python 断点、`print` |
| `TRITON_PRINT_AUTOTUNING=1` | 打印每个 kernel 调优的胜出配置和总耗时 |
| `TRITON_ALWAYS_COMPILE=1` | 绕过缓存强制重编译——**dump 类的变量看不到输出时，八成就是命中缓存了** |
| `TRITON_HOME=某个目录` | 改 `.triton` 目录的位置（缓存和下载都在里面） |
| `PTXAS_OPTIONS=...` | 透传选项给 `ptxas`，比如 `-v` 看寄存器用量和 spill |

> ⚠️ `MLIR_ENABLE_DUMP` 和 `TRITON_KERNEL_DUMP` 最常见的"不生效"原因是**编译缓存命中了**——kernel 没重新编译，自然没有 dump。清掉 `~/.triton/cache/*` 或者加上 `TRITON_ALWAYS_COMPILE=1`。

> ⚠️ 关于 `TRITON_INTERPRET`：官方 README 的原话是"运行 Triton 解释器而不是在 GPU 上运行，可以在 kernel 代码里插 Python 断点"。**它并不承诺"完全不需要 GPU 环境"**——不同版本对 CPU 张量的支持程度不一样，而且解释器只验证**逻辑正确性**，不反映任何性能行为。原稿说的"在 CPU 上模拟执行（无需真实 GPU）"是过度承诺。

### 7.9.3 三个最常做的确认动作

**动作一：确认 Tensor Core 真的用上了**（[03 章 §3.6.3](03-GEMM深度剖析.md) 说的静默退化）

```bash
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/tdump TRITON_ALWAYS_COMPILE=1 python run.py
grep -E "mma|wgmma" /tmp/tdump/**/*.ptx | head
```

看到 `mma.sync.aligned.m16n8k16` 就是走了 Ampere 路径的 Tensor Core；看到 `wgmma.mma_async` 就是走了 Hopper 的 Warpgroup MMA（[03 章 §3.6.2](03-GEMM深度剖析.md)）。**什么都没看到，说明 `tl.dot` 退回了普通 CUDA Core 的循环**，多半是形状没对齐到 16 或者 dtype 不支持。

> ⚠️ Triton 在 Hopper 上什么时候发 `wgmma`、什么时候仍然发 `mma.sync`，随版本、dtype、tile 形状而变，**没有"用了 Triton 就一定用上 wgmma"这回事**。唯一可靠的确认方式就是 grep PTX。

**动作二：确认寄存器和共享内存用量**

```python
compiled = matmul_kernel[grid](...)      # 启动返回编译产物句柄
print(compiled.n_regs)                    # 每线程寄存器数
print(compiled.metadata.shared)           # 每 program 共享内存字节数
print(compiled.asm.keys())                # dict_keys(['ttir','ttgir','llir','ptx','cubin'])
print(compiled.asm['ptx'][:2000])
```

> ⚠️ `.asm` / `.n_regs` / `.metadata` 这几个属性在 Triton 3.x 上可用，但**跨大版本改过名字**（2.x 时代的结构不同）。生产代码里不要依赖它们，只在调试时用。

**动作三：性能对比**

```python
ms = triton.testing.do_bench(lambda: matmul(a, b))          # 返回毫秒（默认取中位数）
ms_ref = triton.testing.do_bench(lambda: torch.matmul(a, b))
print(f"triton {ms:.3f} ms, torch {ms_ref:.3f} ms")
```

`triton.testing.do_bench` 会自己做预热和多次重复，比手写 `torch.cuda.synchronize()` + `time.time()` 可靠（[01 章 §1.8.1](01-GPU硬件基础.md) 讲过异步启动的计时陷阱）。要出对比曲线图用 `@triton.testing.perf_report` + `triton.testing.Benchmark`。

> ★ **完整的性能排查还是要回 [02 章 §2.7](02-屋顶线与性能剖析.md) 的 Nsight 三步漏斗。** Triton 生成的 kernel 在 `ncu` 里和手写 CUDA kernel 没有任何区别，所有指标都能正常采集，kernel 名字就是你的 Python 函数名。

---

## 7.10 torch.compile 与 TorchInductor：自动生成的 Triton

### 7.10.1 管线

```
Python 模型代码
    │  TorchDynamo：改写 Python 字节码，抽出张量操作
    ▼
FX Graph（计算图）
    │  AOTAutograd（Ahead-Of-Time Autograd，提前微分）：把反向也展开成图
    ▼
前向图 + 反向图
    │  TorchInductor：后端代码生成
    ├──→ Triton kernel（GPU）    ← 本章讲的东西，就是它生成的
    └──→ C++ / OpenMP（CPU）
```

细节和"怎么确认真的融了"（看 kernel 名字里的 `triton_poi_fused_...`）见 [05 章 §5.6](05-算子融合.md)，这里不重复。

### 7.10.2 两处需要修正的常见说法

> ⚠️ **"Inductor 遇到 GEMM 一定调 cuBLAS"——不准确。** 默认模式下 Inductor 确实把 `matmul` 交给 ATen（也就是 cuBLAS）。但在 `torch.compile(model, mode="max-autotune")` 下，Inductor 会把 **cuBLAS、Triton 模板 GEMM、（视版本）CUTLASS 模板**都 benchmark 一遍，挑最快的；Triton 模板赢的情况并不罕见，尤其是需要把 Epilogue 融进去的时候。代价是编译时间大幅拉长。

> ⚠️ **"Inductor 会生成 CUDA Graph"——只在 `mode="reduce-overhead"` 下才会**，不是默认行为。见 [05 章 §5.6.3](05-算子融合.md)。

### 7.10.3 一个很实用的用法：让 Inductor 帮你写初稿

```bash
TORCH_LOGS="output_code" python your_script.py
```

这会把 Inductor 生成的**完整 Triton 源码**打印出来。它可读性一般（变量名是 `tmp0`、`tmp1`），但**结构是对的、mask 是对的、grid 是对的**，非常适合当手写 kernel 的起点：先让 `torch.compile` 融一版，把代码捞出来，再手工改进它的分块和 Epilogue。这比从空白文件开始快得多。

> ⚠️ 原稿那张"典型 LLaMA 前向：torch.compile 1.3–1.5×、+FlashAttention 1.5–2.0×、+INT8 量化 2.0–3.5×"的表格**没有给出模型版本、序列长度、batch size、硬件和 PyTorch 版本，也没有来源**，本次改写移除。`torch.compile` 的收益区间和它的适用条件见 [05 章 §5.6.4](05-算子融合.md)，那里的数字同样标了"社区量级"。

---

## 7.11 什么时候用 Triton，什么时候退回 CUDA

### 7.11.1 决策树

```
你要写的是什么？
│
├─ 现成算子能拼出来，只是嫌 kernel 太多太碎
│    └─→ 先试 torch.compile（默认模式）
│          └─ 还不够 → mode="max-autotune" 或 "reduce-overhead"
│                └─ 还不够 → 往下走
│
├─ 标准 GEMM / 标准 Attention / 标准 Conv
│    └─→ 用现成库：cuBLAS / FlashAttention / cuDNN
│          自己写几乎一定更慢（03 章 §3.8.3、04 章 §4.8.4）
│
├─ 逐元素链的融合、归一化、自定义 Attention 变体、
│  自定义量化 scale、MoE 路由、RL 的 logprob / KL 计算
│    └─→ ★ 这是 Triton 的主场，写它
│
└─ 需要精确控制片上布局 / 需要 wgmma + TMA + Warp Specialization /
   需要跨 Block 的复杂同步 / 要榨最后 10–20%
     └─→ 退回 CUTLASS 或手写 CUDA
```

### 7.11.2 六条"该退回 CUDA/CUTLASS"的具体信号

| 信号 | 为什么 Triton 不行 |
|---|---|
| **grep PTX 发现没发 `wgmma`，而你确实需要它** | Triton 选哪条 MMA 指令你控制不了（§7.9.3）。[03 章 §3.6.2](03-GEMM深度剖析.md)：H100 上只用 `mma.sync` 的天花板明显低于 cuBLAS |
| **需要手工设计共享内存布局** | 标准 Triton 语言里没有 `__shared__`，swizzle 由编译器定（§7.2.4）。[03 章 §3.4.3](03-GEMM深度剖析.md) 那种精确的布局控制做不了 |
| **需要 Warp Specialization（生产者/消费者分工）** | 这要求你能指名道姓地控制"哪几个 Warp 只搬数、哪几个只算"。⚠️ 新版 Triton 在部分后端有实验性支持，但接口不稳定，别在生产上赌 |
| **需要跨 Block 的同步或原子协议** | Triton 有 `tl.atomic_add` 等，但复杂的 grid 级协作（比如 Stream-K 的跨 SM 归约，[03 章 §3.7.5](03-GEMM深度剖析.md)）写起来很别扭 |
| **autotune 扫完所有配置还是差现成库 30% 以上** | 说明瓶颈在 Triton 抽象层管不到的地方（Warp Tile、指令调度、寄存器分配），继续调旋钮是浪费时间 |
| **要交付一个跨多个 CUDA 版本的稳定二进制** | Triton 是 JIT，首次运行要编译，且行为随版本变动大（本章标了十几个 ⚠️ 就是证据） |

### 7.11.3 三条"该用 Triton"的正面理由

- **算子在库里不存在。** 这是最强的理由。自定义 mask 的 Attention、带 per-group scale 的 W4A8 GEMM（[06 章](06-量化算子.md)）、MoE 的 Grouped GEMM（[08 章](08-MoE算子与通信.md)）——这些形状 cuBLAS 里没有，"Triton 慢 15%"和"根本没有"之间没得选。
- **迭代速度。** 改一个 Epilogue，Triton 是改三行 Python 重跑；CUTLASS 是改模板参数重编译几分钟；手写 CUDA 是重写一遍。做实验时这个差距是决定性的。
- **跟着硬件走。** 新一代 GPU 出来，Triton 升个版本就可能自动用上新指令；手写的 PTX 要你自己改。

### 7.11.4 和其他方案的横向对照

| 维度 | PyTorch eager | torch.compile | Triton | CUTLASS | 手写 CUDA |
|---|---|---|---|---|---|
| 写一个融合算子 | 不可能 | 零代码 | ~30 行 Python | ~200 行 C++ 模板 | ~300 行 C++ |
| 你要懂什么 | 无 | 无 | 分块 + 屋顶线（01–03 章） | 全套 GPU 架构 | 全套 + PTX |
| 谁管合并访存 / bank conflict | — | 编译器 | **编译器** | 你（用库封装好的 swizzle） | **你** |
| 谁管 Warp/Thread Tile | — | 编译器 | **编译器** | 模板参数 | **你** |
| 谁管流水重叠 | — | 编译器 | 你给 `num_stages` | 模板参数 `Stages` | **你** |
| 能用上 wgmma + TMA 吗 | — | 看 Triton | ⚠️ 看版本和形状，不可控 | ✓ 完全可控 | ✓ 完全可控 |
| 可达性能 | 低 | 中 | ⚠️ 通常 cuBLAS 的 80–90%（[03 章 §3.8.3](03-GEMM深度剖析.md)） | 接近 cuBLAS | 上限最高 |
| 调试手段 | Python | `TORCH_LOGS` | `TRITON_INTERPRET` + ncu | cuda-gdb + ncu | cuda-gdb + ncu |

---

## 7.12 本章小结

**一个心智模型**

```
你写的是一个 Block 的代码，不是一个 Thread 的代码。

  tl.program_id(0)  =  blockIdx.x         （一模一样）
  threadIdx.x       =  不存在             （元素→线程的映射交给编译器）
  if (i < N)        =  mask=              （逐 lane 谓词，不是分支）
  一个标量          =  一条长 BLOCK 的向量

★ 这个交换的本质：你放弃"元素摊到哪个线程"的控制权，
  换来编译器包办合并访存、bank conflict、屏障、MMA 选指令。
```

**编译器管什么，你管什么**

| 归编译器 | 归你 |
|---|---|
| 合并访存（[01 章 §1.4](01-GPU硬件基础.md)） | BLOCK 怎么切、grid 怎么分 |
| bank conflict / swizzle（[01 章 §1.5](01-GPU硬件基础.md)、[03 章 §3.4.3](03-GEMM深度剖析.md)） | **mask**（忘了就越界） |
| 同步屏障（[03 章 §3.3.4](03-GEMM深度剖析.md) 那两个 `__syncthreads()`） | 算法本身（在线 softmax 这类数学改写） |
| Warp Tile / Thread Tile（[03 章 §3.4](03-GEMM深度剖析.md)） | 地址表达式本身连不连续 |
| 选 `mma` / `wgmma`（[03 章 §3.6](03-GEMM深度剖析.md)） | 三个旋钮 `BLOCK_*` / `num_warps` / `num_stages` |

**三个旋钮，三件不同的事**

| 旋钮 | 调什么 | 对应 | 爆掉的症状 |
|---|---|---|---|
| `BLOCK_M/N/K` | 算术强度（带宽问题） | [03 章 §3.3](03-GEMM深度剖析.md) | 寄存器溢出 / program 太少填不满 SM |
| `num_warps` | 并行度与寄存器压力 | [01 章 §1.6](01-GPU硬件基础.md) | 太小溢出，太大归约变贵（上限 32） |
| `num_stages` | 搬运与计算重叠（延迟问题） | [03 章 §3.5](03-GEMM深度剖析.md) | `out of resource: shared memory` |

**五个静默错误（都不报错，只是结果错或性能塌）**

| 错误 | 症状 | 出处 |
|---|---|---|
| 漏写 `mask=` | 越界读写，随机污染别的数据 | §7.3.1 |
| 求 max 时 `other=0.0` | 整行为负时最大值错成 0 | §7.3.2 |
| `num_warps` 塞进 `Config` 的字典里 | 静默用回默认的 4 | §7.8.1 |
| 配了 autotune 却写死 grid | 算漏或算重 | §7.6.4 |
| M/N 维不 `% M` 就用裸 offs 构造指针 | 最后一块越界读 | §7.6.4 |

**三级演进，每步只加一个概念**

| 算子 | 新增的概念 | 对应硬件上的什么 |
|---|---|---|
| vector add | `program_id` / `arange` / `mask` / `load` / `store` | 合并访存、lane 谓词 |
| fused softmax | `tl.max` / `tl.sum` 归约、一个 program 一整行 | 两级归约树（shuffle + 共享内存 + 屏障） |
| matmul | 二维指针广播、`tl.dot`、K 循环、`GROUP_SIZE_M` | Tensor Core MMA、软件流水、L2 命中率 |

**自测三题**

1. 一个 Triton kernel 用 `BLOCK = 2048`、`num_warps = 8`、FP32。每个线程分到多少个元素？如果编译器让每个线程一次取 4 个连续的 float，一个 Warp 一轮覆盖多少字节、多少个 32 字节扇区？合并效率是多少？
2. 某 matmul 用 `BLOCK_M = 256`、`BLOCK_N = 128`、`BLOCK_K = 64`、BF16。每一级流水要多少共享内存？在 H100（单 Block 上限约 227 KB）上 `num_stages` 最大能开到几？如果想开到 5 级，`BLOCK_K` 最大只能取多少（仍是 2 的幂）？
3. C 被切成 12 × 12 = 144 个输出块，一个波次能同时跑 16 个 program。按行扫描时，这 16 个 program 一共要读进多少个 A/B 小块？改成 `GROUP_SIZE_M = 4` 的分组扫描呢？省了百分之几？

#### 参考答案

1. 线程数 = 8 × 32 = **256 个**，每线程 2048 ÷ 256 = **8 个元素**。一个 Warp 一轮 = 32 线程 × 4 个 float × 4 字节 = **512 字节** = **16 个扇区**，全部字节都被用到，**效率 100%**。参见 §7.2.3 和 [01 章 §1.4.1](01-GPU硬件基础.md)。
2. 每级 = `(256 × 64 + 64 × 128) × 2 B` = `(16384 + 8192) × 2` = **49,152 B = 48 KB**。`227 ÷ 48 = 4.7` → `num_stages` 最大 **4**（192 KB ✓，5 级 240 KB ✗）。要开 5 级，每级预算 = 227 ÷ 5 = 45.4 KB；`BLOCK_K = 32` 时每级 = `(256 × 32 + 32 × 128) × 2` = 24,576 B = **24 KB**，5 级共 120 KB ✓，所以 `BLOCK_K` 最大取 **32**。参见 §7.7.3。
3. 按行扫描：16 个 program 覆盖 C 第 0 行的 12 个块 + 第 1 行的前 4 个块。要的 A 块 = 2 条横条 × 12 个 K 块 = 24；B 块 = 12 条竖条 × 12 个 K 块 = 144；**合计 168 块**。分组扫描（`GROUP_SIZE_M = 4`）：16 个 program 挤在 C 的一个 4 × 4 方块里，A 块 = 4 × 12 = 48，B 块 = 4 × 12 = 48，**合计 96 块**。省了 `(168 − 96) ÷ 168` = **43%**。参见 §7.6.3 和 [03 章 §3.3.6](03-GEMM深度剖析.md)。

> **下一章**：MoE（Mixture of Experts，混合专家）把一次大 GEMM 拆成几十个形状不齐的小 GEMM，还要在卡之间做 All-to-All 通信——这正是"库里没有现成 kernel、必须自己写 Triton"的典型场景。见 [08 · MoE 算子与通信](08-MoE算子与通信.md)。

> **昇腾对照**：⚠️ 本章原稿说"昇腾没有 Triton 的等价物"，这一条**已经过时**。`triton-lang/triton-ascend`（面向昇腾 NPU 的 Triton 语言与编译器）已开源，**项目文档口径是「支持 85% 以上的 Triton Python API」**（⚠️ 该比例出自项目自述，未见独立复现），覆盖 MatMul、FlashAttention、LayerNorm 等核心大模型算子，并已适配 vLLM 等开源仓里的 Triton 算子。⚠️ 但它的成熟度、API 覆盖度和性能水平都还在快速变动中（本章写作时的预发布版本是 3.2.0rc4），**昇腾上的生产算子目前仍以 Ascend C 为主**。Ascend C 的抽象层级介于 CUDA C++ 和 Triton 之间：它像 Triton 一样以"块"为单位思考，但片上内存、搬运、流水都要你自己写。同一个逐元素算子 Triton 约 7 行、Ascend C 约 70 行，三者的完整对照见 [13 章 §13.9](13-AscendC编程.md)。根本原因（编译器能不能替你排流水）见 [00 章 §0.3](00-总纲.md)。

**延伸资料**：
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)（01-vector-add、02-fused-softmax、03-matrix-multiplication 对应本章 §7.4–§7.6，必看；06-fused-attention 是 FlashAttention 的官方 Triton 实现，配合 [04 章](04-FlashAttention全系.md)读）
- [Triton Python API 参考](https://triton-lang.org/main/python-api/triton.language.html)（`tl.*` 的权威签名。**本章所有 API 细节以这里为准，不要相信任何博客上的旧写法**）
- [Triton GitHub](https://github.com/triton-lang/triton)（注意：仓库已从 `openai/triton` 迁到 `triton-lang/triton`。README 末尾的 "Tips for Hacking" 一节是 §7.9.2 那张环境变量表的一手来源）
- [TorchInductor 设计文档](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- [triton-ascend](https://github.com/triton-lang/triton-ascend)（昇腾 NPU 的 Triton 后端）
- ⚠️ [FlashAttention 仓库](https://github.com/Dao-AILab/flash-attention)里的 Triton 实现路径随版本变动（历史上在 `flash_attn/flash_attn_triton.py`），要学复杂 kernel 优先看上面官方教程的 06-fused-attention
