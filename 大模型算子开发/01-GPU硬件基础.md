# 01 · GPU 硬件基础：在写第一行 CUDA 之前必须搞清楚的事

> **前置**：无。这是本系列的起点，后面所有章节都建立在这里。  
> **目标**：让你在脑子里有一张清晰的 GPU 硬件地图，知道"一个线程在哪里运行、数据在哪里存、为什么快、为什么会慢"。

---

**本章关键词**

**SM**（Streaming Multiprocessor，流式多处理器）—— GPU 的基本计算单元，一块 H100 有 132 个 SM。  
**Warp** —— SM 内部同时执行的 32 个线程，是 GPU 调度的最小单位。  
**Thread / Block / Grid** —— CUDA 的三级线程组织结构，对应 GPU 的物理执行层次。  
**寄存器**（Register）—— 每个线程私有的最快存储，容量极小（约 256 KB/SM）。  
**共享内存**（Shared Memory / SRAM）—— 一个 Block 内所有线程共享，类似 L1 缓存。  
**HBM**（High Bandwidth Memory）—— GPU 的主内存，大（40–80 GB）但相对慢。  
**Bank Conflict**（存储体冲突）—— 共享内存访问的经典性能杀手。  
**Occupancy**（占用率）—— SM 上实际运行的 Warp 数量 / 理论最大值，影响延迟隐藏能力。  
**Tensor Core** —— 专门做矩阵乘累加的硬件单元，从 V100 开始出现，是现代深度学习的算力来源。

---

## 1.1 为什么 GPU 能快：一个最核心的比喻

CPU 是"几个超级专家"：4–64 个强大的核，每个核能独立解决复杂问题，擅长分支判断、串行依赖、单线程性能。

GPU 是"成千上万个普通工人"：H100 有 16896 个 CUDA Core，每个核很弱（没有大缓存、没有复杂的分支预测），但它们可以**同时**做同一件事。

```
CPU（比喻）：
  [超级厨师] [超级厨师] [超级厨师] [超级厨师]
   能做任何菜    独立运作    快速思考    灵活应变
   适合：单道复杂菜肴

GPU（比喻）：
  [普工][普工][普工]...[普工]  ← 16896 个
   只能切菜     只能炒    只能装盘
   适合：同时切 16896 根胡萝卜
```

深度学习恰好是"切 16896 根胡萝卜"的问题：矩阵里的每个元素计算相互独立，天然适合并行。

---

## 1.2 GPU 的物理结构：从芯片到线程

### 1.2.1 SM：GPU 的基本单元

一块 H100 SXM5 GPU 有 **132 个 SM**（Streaming Multiprocessor）。每个 SM 是一个独立的小型处理器，包含：

```
┌──────────────────── 一个 SM ────────────────────┐
│                                                 │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ │
│  │ CUDA    │ │ CUDA    │ │ Tensor  │ │  SFU  │ │
│  │ Core ×  │ │ Core ×  │ │ Core ×  │ │（特殊 │ │
│  │  32     │ │  32     │ │   4     │ │ 函数）│ │
│  └─────────┘ └─────────┘ └─────────┘ └───────┘ │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │         共享内存 / L1 缓存（256 KB）     │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │         寄存器文件（~256 KB）            │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  Warp 调度器 × 4    ← 每周期可以发射 4 条指令  │
└─────────────────────────────────────────────────┘
```

H100 的关键数字：
- 132 个 SM
- 每个 SM 有 128 个 CUDA Core（FP32）和 4 个 Tensor Core
- 每个 SM 最多同时执行 64 个 Warp（= 2048 个线程）
- 共享内存：最大 228 KB/SM（可配置与 L1 的比例）
- 寄存器：65536 个 32-bit 寄存器/SM

### 1.2.2 CUDA 的三级线程组织

CUDA 把线程组织成三级：

```
Grid（整个 kernel 的所有线程）
 └── Block（一组协作的线程，在同一个 SM 上运行）
      └── Thread（最小执行单元）
```

**Thread**：一个 CUDA 线程，有自己的寄存器，执行一小块计算（比如算矩阵的一个元素）。

**Block**（线程块）：一组线程，分配到同一个 SM 上运行，可以共享该 SM 的共享内存，可以通过 `__syncthreads()` 同步。Block 里通常有 128–1024 个线程。

**Grid**（网格）：所有 Block 的集合，就是一次 kernel 调用的全部工作量。Grid 可以是一维、二维或三维的（方便处理向量、矩阵、3D 数据）。

```python
# PyTorch / CUDA 里启动 kernel 的逻辑（伪代码）
kernel<<<grid_dim, block_dim>>>(args...)
# grid_dim：有多少个 Block
# block_dim：每个 Block 有多少个 Thread
```

**对应关系**：一个 Block → 一个 SM（SM 可以同时容纳多个 Block）；一个 Grid → 整块 GPU。

### 1.2.3 Warp：调度的真正最小单位

Block 里的 32 个线程构成一个 **Warp**。GPU 调度器以 Warp 为单位发射指令——同一个 Warp 里的 32 个线程**永远执行相同的指令**，只是处理不同的数据（SIMT 执行模型：Single Instruction Multiple Threads）。

这有一个重要推论：**Warp 分歧**（Warp Divergence）是性能杀手。

```c
// 反例：同一 Warp 里的线程走了不同分支
if (threadIdx.x % 2 == 0) {
    doExpensiveWork();   // 偶数线程执行这里
} else {
    doOtherWork();       // 奇数线程执行这里
}
```

GPU 的处理方式：先让偶数线程执行 `doExpensiveWork()`，奇数线程**等待**（掩码屏蔽）；然后让奇数线程执行 `doOtherWork()`，偶数线程等待。两段都执行完才算结束。实际时间 = 两段时间之和，而不是较长的那段。

> ★ 设计 kernel 时要尽量让同一 Warp 里的线程走相同的分支。

---

## 1.3 内存层级：从快到慢

GPU 的内存是一个金字塔，越快越小越贵：

```
访问速度 ↑      容量 ↓
  极快   │ 寄存器（Register）     每线程 ~256 个 32-bit，约 1 周期
         │ 共享内存（Shared Mem） 每 SM 228 KB，约 5–30 周期（无 bank conflict）
         │ L2 缓存               50 MB（H100），约 100–200 周期
  较快   │ HBM（主显存）          80 GB（H100 SXM5），约 300–600 周期
  较慢   │ PCIe → 主机 DRAM       约 6 GB/s，延迟 µs 级
访问速度 ↓      容量 ↑
```

### 寄存器

每个线程独享，完全私有，访问速度等同于计算速度（1 周期）。

**限制**：每个 SM 最多 65536 个 32-bit 寄存器，线程越多，每个线程能用的寄存器就越少。如果一个线程用了太多寄存器，编译器会把溢出部分存到 **Local Memory**（本地内存，物理上在 HBM，很慢）。这叫 **Register Spilling**（寄存器溢出），是 kernel 性能崩坏的常见原因之一。

### 共享内存（SRAM）

Block 内所有线程共享，生命周期等同于 Block。访问速度比 HBM 快约 10–100 倍，是手动控制数据复用的核心工具。

**使用方式**（CUDA C）：
```c
__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];  // 声明共享内存
tile[threadIdx.y][threadIdx.x] = A[row * K + col];  // 从 HBM 加载
__syncthreads();  // 等所有线程加载完
// 现在可以安全地读 tile，反复复用
float val = tile[threadIdx.y][k];
```

### HBM（高带宽显存）

GPU 的主内存。H100 SXM5 有 80 GB HBM3，带宽约 3.35 TB/s。看起来很高，但和算力（2000 TFLOPS BF16）比，只够喂饱算术强度约 600 FLOP/Byte 的算子（见第 02 章）。大多数算子算术强度远低于 600，所以 HBM 带宽是瓶颈。

---

## 1.4 Bank Conflict：共享内存的经典陷阱

共享内存在物理上被分成 32 个"存储体"（Bank），编号 0–31，每个 Bank 的宽度是 4 字节（32-bit）。地址分配规则：第 0 个 4 字节在 Bank 0，第 1 个在 Bank 1，……，第 32 个又回到 Bank 0。

**Bank Conflict**：同一 Warp 内，如果多个线程同时访问**同一个 Bank 里的不同地址**，这些访问会被串行化——原本应该同时完成的访问变成了一个接一个。

```
正常访问（无冲突）：
  线程 0  → Bank 0  ← 地址 0×4
  线程 1  → Bank 1  ← 地址 1×4
  ...
  线程 31 → Bank 31 ← 地址 31×4
  所有访问同时完成 ✓

2-way Bank Conflict：
  线程 0  → Bank 0  ← 地址 0×4
  线程 16 → Bank 0  ← 地址 32×4   ← 冲突！同一 Bank 两个不同地址
  ...
  需要 2 个周期完成，性能减半 ✗

32-way Bank Conflict（最坏情况）：
  所有线程访问同一 Bank 的不同地址
  需要 32 个周期，性能变为 1/32 ✗✗✗
```

**典型场景**：矩阵转置时，按列读取共享内存 tile 就会触发 Bank Conflict。

**解决方法**：在 tile 的列数上加一个 Padding（+1 或 +2），让每列偏移到不同的 Bank：

```c
// 有 Bank Conflict 的版本
__shared__ float tile[BLOCK][BLOCK];  // tile[0][0], tile[1][0]... 都在 Bank 0

// 消除 Bank Conflict：加一列 padding
__shared__ float tile[BLOCK][BLOCK + 1];  // 偏移后每列落在不同 Bank ✓
```

---

## 1.5 Occupancy：延迟隐藏的关键

**Occupancy**（占用率）= SM 上实际活跃的 Warp 数 / SM 支持的最大 Warp 数（H100 是 64）。

为什么 Occupancy 重要？GPU 隐藏内存延迟的方式是**切换 Warp**：当一个 Warp 在等 HBM 数据（几百个周期），调度器立刻切换到另一个就绪的 Warp 继续计算。如果 SM 上只有 2 个 Warp，等待时调度器无牌可出，SM 就空转了。

```
低 Occupancy（比喻）：
  只有 2 个服务员，一个去取菜（等待 HBM），另一个也在等，厨房空闲。

高 Occupancy（比喻）：
  有 16 个服务员，取菜的等待期间，其他服务员继续上菜，厨房一直满负荷。
```

**影响 Occupancy 的三个因素**：

1. **寄存器用量**：每个线程用越多寄存器，SM 能容纳的线程越少，Occupancy 越低。
2. **共享内存用量**：一个 Block 用越多共享内存，SM 能同时放的 Block 越少。
3. **Block 大小**：Block 太小（比如 32 个线程），一个 SM 上需要很多 Block 才能填满；Block 太大，资源（寄存器/共享内存）被一个 Block 独占，其他 Block 进不来。

> ⚠️ Occupancy 高不等于性能好。Compute Bound 的算子满载运行时，提高 Occupancy 没有收益；有时候适当降低 Occupancy（让每个线程用更多寄存器，减少寄存器溢出）反而更快。Occupancy 是延迟隐藏的工具，不是目标本身。

**经验值**：大多数高效 kernel 的 Occupancy 在 50–75% 之间就够了。

---

## 1.6 Tensor Core：矩阵乘法的专用加速器

从 Volta（V100）开始，每个 SM 内置了 **Tensor Core**——专门执行小矩阵乘累加（Matrix Multiply-Accumulate, MMA）的硬件单元。

```
普通 CUDA Core（FP32）：
  每个周期：1 次乘法 + 1 次加法 = 2 FLOP
  每个 SM 128 个 CUDA Core = 256 FLOP/周期

Tensor Core（BF16，H100 Hopper）：
  每个周期：一次 16×16 MMA = 16×16×2 = 512 FLOP
  每个 SM 4 个 Tensor Core = 2048 FLOP/周期  ← 快 8 倍
```

Tensor Core 要求矩阵维度是特定值的倍数（不同精度不同要求）：

| 精度 | 矩阵维度要求 | H100 峰值吞吐 |
|---|---|---|
| FP32 | 无（用 CUDA Core） | 约 67 TFLOPS |
| TF32 | M,N,K 是 16 的倍数 | 约 494 TFLOPS |
| BF16 / FP16 | M,N,K 是 16 的倍数 | 约 989 TFLOPS |
| FP8 | M,N,K 是 16 的倍数 | 约 1979 TFLOPS |
| INT8 | M,N,K 是 16 的倍数 | 约 1979 TOPS |

**这就是为什么大模型的隐藏维度都是 128 的倍数**（128 = 8 × 16，满足所有精度的要求，也方便分块）。

Tensor Core 在 Warp 级别（32 个线程）协作操作：一个 Warp 的 32 个线程共同完成一次 16×16 的 MMA。要用上 Tensor Core，必须用 `wmma`（Warp Matrix Multiply-Accumulate）API 或更高层的 cuBLAS / CUTLASS / Triton。

---

## 1.7 一个完整的执行流程

把以上内容串起来，看一次 kernel 调用从代码到 GPU 的完整路径：

```
① Python/PyTorch：torch.matmul(A, B)
      ↓
② 调度到 cuBLAS GEMM kernel
      ↓
③ CPU 向 GPU 发送 kernel 启动命令（约 5–10 µs 延迟）
      ↓
④ GPU 的工作分配器（GigaThread Engine）把 Grid 里的 Block 分配给各 SM
      ↓
⑤ 每个 SM 接收若干 Block，每个 Block 分成若干 Warp
      ↓
⑥ SM 的 Warp 调度器选择就绪的 Warp 发射指令
      ↓
⑦ 指令命中缓存 → 直接用；未命中 → 发送 HBM 请求（切换到其他 Warp）
      ↓
⑧ 数据回来 → Warp 重新就绪 → 调度器再次选择它
      ↓
⑨ 最终结果写回 HBM → kernel 完成 → CPU 读取结果
```

---

## 1.8 本章小结

```
核心概念速查：

SM         → GPU 的基本计算单元，H100 有 132 个
Warp       → 32 个线程，调度的最小单位，执行相同指令
Block      → 一组 Warp，在同一 SM 上运行，共享 SRAM
Grid       → 所有 Block，整个 kernel 的工作量

内存层级（快 → 慢）：
  寄存器（~1 周期）→ 共享内存（~5 周期）→ L2（~100 周期）→ HBM（~400 周期）

Bank Conflict → 多线程访问同一 Bank 的不同地址，性能串行化，加 Padding 解决
Occupancy    → SM 上活跃 Warp 比例，高 = 更好的延迟隐藏，但不是越高越好
Tensor Core  → 专用矩阵乘加速器，要求维度是 16 的倍数，是现代大模型算力的来源

下一章：知道了硬件长什么样，接着看怎么判断一个算子是被算力限制还是被内存限制。
```

**延伸资料**：
- NVIDIA [H100 架构白皮书](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)（§5 Memory Model 最重要）
- [GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)（NVIDIA 官方性能文档）
