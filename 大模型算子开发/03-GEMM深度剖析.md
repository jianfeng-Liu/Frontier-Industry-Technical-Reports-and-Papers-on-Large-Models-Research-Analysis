# 03 · GEMM 深度剖析：大模型 90% 算力的来源

> **前置**：[01 章（GPU 硬件基础）](01-GPU硬件基础.md)、[02 章（屋顶线与性能剖析）](02-屋顶线与性能剖析.md)  
> **目标**：彻底搞清楚矩阵乘法在 GPU 上如何从朴素实现一步步优化到接近硬件峰值，以及 cuBLAS / CUTLASS 的架构思路。

---

**本章关键词**

**GEMM**（General Matrix Multiply）—— `C = α·A·B + β·C`，深度学习里最核心的算子。  
**分块**（Tiling）—— 把大矩阵切成小块装进 SRAM，提高数据复用率的核心手段。  
**三级分块**（3-Level Tiling）—— Thread Block Tile → Warp Tile → Thread Tile，对应 GPU 的三级内存层次。  
**双缓冲**（Double Buffering）—— 让数据加载和计算流水线重叠，消除内存等待。  
**Epilogue**（收尾计算）—— GEMM 计算完成后的融合操作，如加 bias、激活函数、量化等。  
**cuBLAS** —— NVIDIA 官方 BLAS 库，最优化但不可定制。  
**CUTLASS** —— NVIDIA 开源的 GEMM 模板库，可定制 Epilogue，是 TensorRT 和 FlashAttention 的底层。  
**Stream-K** —— 解决 GEMM 在 GPU SM 之间负载不均衡问题的分解方式（CUTLASS 3.x 引入）。

---

## 3.1 为什么 GEMM 是大模型的核心

大模型的一次前向传播（以 Transformer 为例）：

```
输入 x (B×S×H)
  ↓  线性变换 W_Q, W_K, W_V  ← 3 次 GEMM (B·S × H) × (H × H)
  ↓  QKᵀ                     ← 1 次 GEMM (B·H × S) × (S × H) [Attention]
  ↓  ·V                      ← 1 次 GEMM
  ↓  W_O 投影                 ← 1 次 GEMM
  ↓  FFN 第一层 W_1           ← 1 次 GEMM，输出维度扩 4×
  ↓  FFN 第二层 W_2           ← 1 次 GEMM，输出维度缩回
  ↓  LM head                 ← 1 次 GEMM (B·S × H) × (H × V)  [词表通常最大]
```

一个 32 层的 Transformer，每层 8 次 GEMM = 256 次 GEMM。**约 90% 的 FLOP 都是矩阵乘法**。

---

## 3.2 朴素实现：为什么慢

```c
// 最朴素的 CUDA 实现：每个线程负责 C 的一个元素
__global__ void naive_gemm(float *A, float *B, float *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // C 的行
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // C 的列

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];  // ← 每次都从 HBM 读
        }
        C[row * N + col] = sum;
    }
}
```

**性能分析**：

每个线程计算一个 C 的元素，需要读 A 的一行（K 个元素）和 B 的一列（K 个元素）。

```
对于 M=N=K=4096，FP32：
  总 FLOPs = 2 × 4096³ ≈ 137 G FLOP
  
  每个线程读 A：4096 × 4 = 16 KB
  每个线程读 B：4096 × 4 = 16 KB
  总共有 4096×4096 = 16M 个线程
  
  A 的每一行被读 N=4096 次
  B 的每一列被读 M=4096 次
  总内存读写 ≈ 2 × 4096² × 4096 × 4 Bytes ≈ 512 GB
  
  算术强度 = 137 G / 512 G ≈ 0.27 FLOP/Byte   ← 极低！
```

朴素实现的算术强度只有 0.27，而 H100 需要 295 才能打满算力。慢 1000 倍。

---

## 3.3 第一级优化：共享内存分块

**核心思路**：让同一个 Thread Block 的线程**共同**把 A、B 的一块装进共享内存，然后各自用这块数据计算，复用后再换下一块。

### 单级分块

```
把 C 切成 (BLOCK_M × BLOCK_N) 的块，每个 Thread Block 负责一块。

Thread Block 处理 C[i:i+BM, j:j+BN]：
  for k_tile in range(0, K, BLOCK_K):
    # 阶段1：协作加载（所有线程一起把数据搬进 SRAM）
    从 HBM 加载 A[i:i+BM, k_tile:k_tile+BK] 进 smem_A
    从 HBM 加载 B[k_tile:k_tile+BK, j:j+BN] 进 smem_B
    __syncthreads()  # 确保所有线程加载完毕
    
    # 阶段2：每个线程用 SRAM 里的数据计算自己负责的元素
    for k in range(BLOCK_K):
        C_thread += smem_A[thread_row, k] * smem_B[k, thread_col]
    
    __syncthreads()  # 确保所有线程用完当前 tile 再换下一块
```

**算术强度提升**：

```
A 的 tile 大小：BM × BK × 4 Bytes
B 的 tile 大小：BK × BN × 4 Bytes
这块数据被 BM × BN 个线程各用一次，其中 A 的每一行被用 BN 次，B 的每一列被用 BM 次

算术强度 ≈ 2 × BM × BN × BK / (BM×BK + BK×BN) × (1/4Byte)
         ≈ BM × BN / (BM + BN) × 0.5   [当 BM=BN=128 时]
         ≈ 32 FLOP/Byte   ← 比朴素提升 100 倍
```

实际 BM=BN=128，BK=16 时算术强度约 32。还不够，继续优化。

---

## 3.4 三级分块：对应 GPU 的三级内存层次

GPU 有三级内存：HBM → 共享内存（SRAM）→ 寄存器。高效 GEMM 需要三级分块全部用上。

```
Level 1：Thread Block Tile（Block 级别分块）
  A_tile: BM × BK  装进共享内存
  B_tile: BK × BN  装进共享内存
  一次从 HBM 加载，供整个 Block 复用

Level 2：Warp Tile（Warp 级别分块）
  每个 Warp 从共享内存里取自己负责的子块
  WM × WN 大小，供 32 个线程复用

Level 3：Thread Tile（线程级别分块）
  每个线程用寄存器存一个 TM × TN 的小矩阵
  直接在寄存器里做外积（outer product）累加
```

### 外积（Outer Product）计算模式

高效的线程级矩阵乘采用**外积**而非点积：

```
朴素（点积模式）：
  每个线程算 C 的一个元素
  c[i][j] = sum_k a[i][k] * b[k][j]   ← 每次迭代 A 和 B 各用 1 次

外积模式：
  每个线程负责 C 的 TM×TN 的小块
  对每个 k：
    从 A 取一列（TM 个元素）存到寄存器 a_reg[TM]
    从 B 取一行（TN 个元素）存到寄存器 b_reg[TN]
    做外积：c_reg[i][j] += a_reg[i] * b_reg[j]  （TM×TN 次乘加）
    
  ← 每次迭代：读 TM + TN 个元素，做 TM×TN 次乘加
  算术强度 = TM×TN / (TM + TN)
  当 TM=TN=8：算术强度 = 64/16 = 4
  加上 Block 分块的复用，总算术强度可达 128+
```

### 数字化示例（CUTLASS 典型配置）

```
H100 SXM5 上的高效 GEMM（BF16）：

Block Tile：  BM=128, BN=256, BK=64
Warp Tile：   WM=64, WN=64
Thread Tile： TM=8, TN=8

每个 Block 有 (128/64)×(256/64) = 2×4 = 8 个 Warp
每个 Warp 有 (64/8)×(64/8) = 8×8 = 64 个线程（≈ 2 个 Warp，取整为 32 线程/Warp）

共享内存用量：
  smem_A = 128×64×2 Bytes = 16 KB（BF16）
  smem_B = 64×256×2 Bytes = 32 KB
  合计 48 KB / Block，低于 H100 的 228 KB/SM 上限 ✓

寄存器用量：
  每个线程：TM×TN = 64 个 BF16 寄存器（c_reg）
            + TM + TN = 16 个寄存器（a_reg, b_reg）
  合计约 80 个寄存器/线程，在 65536/256 = 256 上限内 ✓
```

---

## 3.5 双缓冲：消除计算等待内存

单缓冲的问题：每次 k_tile 迭代，所有线程先等数据加载完（HBM → SRAM），再计算，再等下一轮加载……计算和加载串行。

**双缓冲**（Software Pipelining）：在计算当前 tile 的同时，**异步**预加载下一个 tile：

```
时间 →
  单缓冲：[加载 tile₀][计算 tile₀][加载 tile₁][计算 tile₁][加载 tile₂]...
  双缓冲：[加载 tile₀][加载 tile₁   ←异步→  ][计算 tile₀][加载 tile₂][计算 tile₁]...
                                  └── 两个 SRAM 缓冲区交替使用 ──┘
```

实现时维护两个共享内存缓冲区（buffer A 和 buffer B），当前迭代用 A 计算时，用 CUDA 异步拷贝（`cp.async`，H100 支持）把下一个 tile 加载进 B，完成后切换。

H100 的 `cp.async` 指令允许在不占用计算流水线的情况下发起 HBM → SRAM 的数据搬运，是实现双缓冲的硬件基础。

---

## 3.6 Tensor Core 的接入

上面讨论的都是用 CUDA Core 做矩阵乘，实际高效 GEMM 必须用 Tensor Core。

### Warp-level MMA（矩阵乘累加）

H100 的 Tensor Core 支持 `m16n8k16` 的 MMA 操作：一个 Warp（32 个线程）协作完成一次 16×16×16 的矩阵乘累加（BF16 输入，FP32 累加）。

一次 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 指令：
- 输入：A fragment (16×16)，B fragment (16×8)
- 累加：C fragment (16×8，FP32)
- 计算：C += A × B
- Warp 里 32 个线程各持有 fragment 的一部分（分布式寄存器）

### 要"喂饱" Tensor Core

Tensor Core 理论峰值 989 TFLOPS（BF16，H100），但 Warp 调度器必须不断给它送 MMA 指令才能维持高吞吐。如果 MMA 指令之间有气泡（等待内存、Warp 分歧、寄存器依赖），性能就会下降。

实际高效的 GEMM（如 cuBLAS）能达到 85–90% 的峰值利用率，关键就是把双缓冲、分块参数调优、MMA 指令发射密度三者调到最优配合。

---

## 3.7 CUTLASS：可定制的 GEMM 模板库

### 为什么不直接用 cuBLAS

cuBLAS 的 GEMM 极度优化，但只能做 `C = α·A·B + β·C`。大模型里的常见需求：

- `C = GEMM(A, B) + bias`（加 bias）
- `C = activation(GEMM(A, B))`（融合激活）
- `C = int8_quantize(GEMM(A, B))`（融合量化）
- `C = GEMM(A, B) * gate`（SwiGLU 里的门控）

这些都需要在 GEMM 的最后一步（写出 C 之前）做额外计算——这就是 **Epilogue**（收尾）。

cuBLAS 不支持自定义 Epilogue，而 **CUTLASS**（CUDA Templates for Linear Algebra Subroutines and Solvers）专为此设计：

```
CUTLASS 的分层结构：

Device Level:   GemmUniversal           ← 用户接口
    ↓
Kernel Level:   GemmUniversal<...>      ← 整个 kernel 的逻辑
    ↓
Collective:     MainloopSm90 + Epilogue  ← 主循环 + 收尾，可替换
    ↓
Tile:           TiledMMA + SmemLayout    ← 分块参数，模板参数传入
    ↓
Atom:           MMA_Atom (m16n8k16)      ← 底层 PTX 指令
```

### 自定义 Epilogue 示例

```c++
// CUTLASS 的 Epilogue 支持 "访问者模式"（EVT：Epilogue Visitor Tree）
// 可以把任意 element-wise 操作链接在 GEMM 后面

// 示例：GEMM + bias + ReLU + 输出量化为 INT8
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,           // 输出类型 INT8
    128 / sizeof(ElementOutput) * 8,  // 向量化宽度
    ElementAccumulator,      // 累加器类型 FP32
    ElementComputeEpilogue,  // 中间计算精度 FP32
    cutlass::epilogue::thread::ScaleType::NoBetaScaling
>;
```

### Stream-K：解决尾部负载不均

标准的 GEMM 分块方式（Data-Parallel 模式）把 C 矩阵按 Block Tile 均匀分配给各 SM。但当矩阵大小不是 Block Tile 的整数倍时，最后一批 Block 数量少于 SM 数，造成 SM 空闲。

```
例子：M=N=1000, BM=BN=128
  需要 ceil(1000/128)² = 8×8 = 64 个 Block
  H100 有 132 个 SM
  第一轮：132 个 SM 全部运行（但只有 64 个 Block）
  → 有 132 - 64 = 68 个 SM 空转！
```

**Stream-K**（CUTLASS 3.x）把工作按"SM 数 × 若干轮"切片，每个 SM 分到近似等量的 k 方向工作，用 reduction 汇总跨 SM 的部分和，消除尾部空闲。对中小矩阵效果显著。

---

## 3.8 实测数字与选型建议

### H100 SXM5 的 GEMM 实测（BF16，方阵）

| 矩阵大小 | cuBLAS | CUTLASS 3 | 理论峰值 | 利用率 |
|---|---|---|---|---|
| 4096×4096×4096 | 875 TFLOPS | 858 TFLOPS | 989 TFLOPS | 88% |
| 8192×8192×8192 | 921 TFLOPS | 908 TFLOPS | 989 TFLOPS | 93% |
| 1024×4096×4096 | 380 TFLOPS | 412 TFLOPS | 989 TFLOPS | 42% |
| 1×4096×4096 | 4.2 TFLOPS | 4.8 TFLOPS | 989 TFLOPS | 0.5% |

> ⚠️ batch=1 的推理场景，GEMM 利用率不足 1%。这是推理优化的核心挑战，见第 09 章。

### 选型建议

| 场景 | 推荐 | 原因 |
|---|---|---|
| 训练，标准形状 | cuBLAS（通过 PyTorch 自动调用） | 最优化，开箱即用 |
| 训练，需要融合 Epilogue | CUTLASS | 模板化 Epilogue，性能接近 cuBLAS |
| 推理，需要融合量化/激活 | CUTLASS 或 TensorRT-LLM 的内置 kernel | 融合量化收益大 |
| 实验性自定义 kernel | Triton（见第 07 章） | 开发速度快 |
| MoE Grouped GEMM | grouped_gemm 库（基于 CUTLASS） | 处理不规则 batch 大小 |

---

## 3.9 本章小结

```
核心要点：

GEMM 优化的本质 = 提高算术强度 = 让每字节数据做更多计算

三级分块对应三级内存：
  Block Tile → 共享内存（SRAM）
  Warp Tile  → 寄存器文件
  Thread Tile → 寄存器（外积计算）

双缓冲 = 让加载和计算流水线重叠，消除等待

Tensor Core 需要 m16n8k16 的 Warp 级 MMA 指令，
维度必须是 16 的倍数才能用上

cuBLAS：最快但不可定制
CUTLASS：可定制 Epilogue，Stream-K 解决负载均衡

batch=1 推理时 GEMM 算术强度约 1，是推理性能的根本瓶颈

下一章：注意力机制的 GEMM 有特殊性，FlashAttention 怎么把它优化到极致。
```

**延伸资料**：
- [CUTLASS 官方文档与示例](https://github.com/NVIDIA/cutlass)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)（强烈推荐，通俗版屋顶线 + GEMM 分析）
- [Dissecting the Ampere GPU Architecture](https://developer.nvidia.com/blog/dissecting-the-ampere-gpu-architecture-through-microbenchmarking/)
- Stream-K 论文：[*Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU*](https://arxiv.org/abs/2301.03598)
