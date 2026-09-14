# 07 · Triton 编程：用 Python 写 GPU kernel 的正确方式

> **前置**：[01 章（GPU 硬件基础）](01-GPU硬件基础.md)、[02 章（屋顶线）](02-屋顶线与性能剖析.md)  
> **目标**：搞懂 Triton 的编程模型，能写出比 PyTorch 原生快的自定义 kernel，了解它和 CUDA 的本质区别以及 torch.compile 是如何用它的。

---

**本章关键词**

**Triton** —— OpenAI 开源的 GPU 编程语言/编译器，用 Python 写 kernel，自动处理内存对齐、Warp 分组、bank conflict 等细节，比 CUDA 开发效率高得多。  
**Block 级并行**（Block-level Programming）—— Triton 的核心抽象：程序员只写"一个 Block 做什么"，Triton 负责把无数 Block 分派到 GPU 上并行执行。  
**tl（Triton Language）** —— Triton 的 Python API，`tl.load`、`tl.store`、`tl.dot` 是最核心的三个操作。  
**@triton.jit** —— 把 Python 函数编译为 PTX（NVIDIA GPU 汇编）的装饰器，首次调用时触发 JIT 编译。  
**autotuner** —— Triton 内置的超参数搜索工具，自动找到最优的 Block 大小、warp 数等配置。  
**TorchInductor** —— torch.compile 的默认后端，负责把 PyTorch 图编译为 Triton kernel（CPU 则编译为 C++/OpenMP）。

---

## 7.1 为什么需要 Triton

CUDA 很强，但门槛高：
- 需要精通 C++，还要懂 GPU 架构细节（Warp 调度、Bank Conflict、异步内存）
- 一个高效的 GEMM kernel 动辄数千行 C++ 模板代码
- 调试困难，没有真正的 Python 调试器

PyTorch 算子扩展（`torch.autograd.Function`）可以用 Python 写前向/反向，但底层还是要用 C++/CUDA 写真正的 kernel。

**Triton 的定位**：用 Python 写 GPU kernel，但性能接近手写 CUDA。

```
不同层次的 GPU 编程：

性能  ↑    纯 CUDA C++     最高性能，最难写
      │    CUTLASS         次高性能，C++ 模板
      │    Triton          80–95% CUDA 性能，Python 接口
      │    Torch Inductor  自动生成 Triton，无需手动写 kernel
      ↓    PyTorch         最易用，性能受算子融合限制
开发效率 ↓
```

---

## 7.2 Triton 的编程模型

### 核心抽象：Block 级并行

Triton kernel 里写的是"**一个 Block** 怎么处理数据"。不同的 Block 对应数据的不同分块，并行执行：

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, z_ptr,    # 三个向量的内存指针
    n_elements,              # 向量长度
    BLOCK_SIZE: tl.constexpr # Block 大小（编译期常量）
):
    # 每个程序实例（即每个 Block）处理一段数据
    # program_id(0) 是当前 Block 在第 0 个维度上的编号
    pid = tl.program_id(axis=0)
    
    # 这个 Block 负责处理的偏移量范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 边界检查（向量长度不一定是 BLOCK_SIZE 的倍数）
    mask = offsets < n_elements
    
    # 从 HBM 加载（mask 防止越界访问）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 计算
    z = x + y
    
    # 写回 HBM
    tl.store(z_ptr + offsets, z, mask=mask)
```

**调用方式**：

```python
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    
    # 启动 ceil(n / BLOCK_SIZE) 个 Block
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    vector_add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z
```

**和 CUDA 的对比**：
- CUDA 需要写 `threadIdx`、`blockIdx`、手动处理 Warp 对齐
- Triton 只需要 `program_id`（Block 编号），线程层次对程序员**完全透明**
- Triton 自动把 `tl.arange(0, BLOCK_SIZE)` 的操作向量化到合适的 Warp 执行

---

## 7.3 矩阵乘法：Triton 的核心用例

下面是一个简化的 FP16 GEMM kernel，展示 Triton 如何做分块矩阵乘：

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A 的步长（行步长, 列步长）
    stride_bk, stride_bn,   # B 的步长
    stride_cm, stride_cn,   # C 的步长
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 哪个 Block 处理 C 的哪个分块
    pid_m = tl.program_id(axis=0)   # C 的行方向 Block 编号
    pid_n = tl.program_id(axis=1)   # C 的列方向 Block 编号
    
    # C[pid_m * BM : (pid_m+1) * BM, pid_n * BN : (pid_n+1) * BN]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 用于 K 方向的指针偏移量
    offs_k = tl.arange(0, BLOCK_K)
    
    # 构造 A 和 B 的指针（二维指针运算）
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    # 累加寄存器，初始化为 0
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K 方向分块循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 A 和 B 的分块（边界 mask）
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        
        # 矩阵乘累加（Triton 会自动调用 Tensor Core 的 MMA 指令）
        acc += tl.dot(a, b)
        
        # 更新指针到下一个 K 块
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # 转换精度并写回
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

**关键细节**：
- `tl.dot(a, b)` —— Triton 自动分析形状，在支持的 GPU 上调用 `wmma`/`mma` PTX 指令（即 Tensor Core）
- 无需手写 Warp Tile 和 Thread Tile，Triton 编译器处理
- 没有显式共享内存管理，Triton 自动把 tile 放入 SRAM（通过 `tl.load` 的 cache 策略）

---

## 7.4 autotuner：自动找最优配置

手动选 BLOCK_M/BLOCK_N/BLOCK_K 很费力。Triton 提供了 `@triton.autotune` 装饰器：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_warps': 4}),
        # ... 更多配置
    ],
    key=['M', 'N', 'K'],  # 按这些维度分开缓存调优结果
)
@triton.jit
def matmul_kernel(...):
    ...
```

第一次以特定 `(M, N, K)` 调用时，Triton 会把所有配置都跑一遍，选出最快的，并把结果缓存到磁盘（`~/.triton/autotune_cache`）。后续相同形状的调用直接用缓存。

---

## 7.5 Triton 生成的代码路径

```
@triton.jit Python 函数
        ↓
Triton 编译器（LLVM IR 中间层）
        ↓
PTX（NVIDIA GPU 汇编）
        ↓
SASS（GPU 的机器码，由 nvcc 的 ptxas 编译）
        ↓
H100 Tensor Core / CUDA Core 执行
```

Triton 编译器做的主要工作：
1. 把 `tl.arange` 和向量运算展开为 SIMT 线程执行
2. 把 `tl.load`/`tl.store` 转为带 cache 策略的 `ld.global.cs`/`st.global.cs` PTX 指令
3. 把 `tl.dot` 转为 `mma.sync` 或 `wgmma` PTX 指令（视目标架构）
4. 自动插入 `__syncthreads()` 式同步（通过 LLVM Pass 分析）

---

## 7.6 torch.compile 与 TorchInductor

`torch.compile` 是 PyTorch 2.0 引入的编译栈，把 Python 的 PyTorch 代码编译为更快的等价实现：

```
torch.compile 内部管道：

Python 代码（model.forward()）
    ↓
Torch Dynamo（FX Graph 捕获，追踪 Python 字节码）
    ↓
AOTAutograd（提前生成 forward + backward 的 FX 图）
    ↓
TorchInductor（后端，把 FX 图编译为 Triton kernel）
    ↓
Triton PTX + CUDA kernel
    ↓
H100 执行
```

**Inductor 做了什么**：
- 把多个 element-wise 操作融合为一个 Triton kernel（消灭算子间的带宽税）
- 把 Reduction 和前序操作融合
- 对 GEMM 插入 cuBLAS 调用（不用 Triton 重写 GEMM，cuBLAS 更优化）
- 生成 CUDA Graph 以减少 kernel launch 开销

**实际效果**（典型 LLaMA 前向）：

| 模式 | 吞吐（tokens/s）| 备注 |
|---|---|---|
| PyTorch eager | 基准 1× | 无融合 |
| torch.compile | 1.3–1.5× | 自动融合 element-wise |
| + FlashAttention | 1.5–2.0× | 手写 Triton/CUDA kernel |
| + 量化（INT8）| 2.0–3.5× | 视 batch size |

---

## 7.7 实际开发一个 Triton kernel 的流程

以实现 RMSNorm 为例（一个典型的 Memory Bound 算子，融合进 GEMM 后有明显收益）：

```python
@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': bs}) for bs in [256, 512, 1024, 2048]],
    key=['N'],
)
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    M, N,               # M = batch * seq_len, N = hidden_dim
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 Block 处理 x 的一行（一个 token）
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    
    # 分块计算平方和
    sq_sum = tl.zeros((), dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sq_sum += tl.sum(x * x, axis=0)
    
    # RMS 归一化因子
    rms = tl.sqrt(sq_sum / N + eps)
    
    # 归一化并写出
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        out = (x / rms) * w
        tl.store(out_ptr + row_start + cols, out.to(tl.float16), mask=mask)
```

**调试技巧**：
- 先用 `TRITON_INTERPRET=1` 环境变量在 CPU 上模拟执行（无需真实 GPU，便于 print 调试）
- `triton.testing.perf_report` 生成性能对比图
- `triton.compiler.compile` 查看生成的 PTX，确认 Tensor Core 指令被发射

---

## 7.8 Triton vs CUDA：选哪个

| 维度 | Triton | CUDA |
|---|---|---|
| 开发效率 | 高（Python，不用管 Warp/Bank） | 低（C++，需要精细调控） |
| 可达性能 | ~80–95% 手写 CUDA | 100%（可以精调每个细节） |
| 调试体验 | 较好（TRITON_INTERPRET 模式） | 较差（需要 cuda-gdb/Nsight） |
| 新硬件适配 | 快（Triton 自动适配 wgmma 等新指令） | 慢（需手动更新代码用新 PTX 指令）|
| 适用场景 | Element-wise 融合、自定义 Attention、RMSNorm | 极致性能 GEMM、需要精确控制 SRAM 布局的场景 |
| 典型用户 | AI 研究员、ML 工程师 | GPU 内核工程师 |

**实用建议**：
- 大多数算子融合（RMSNorm + dropout + residual、自定义 Attention 变体）用 Triton，够用且快
- GEMM 用 cuBLAS / CUTLASS，不要在 Triton 里重写（即使 Triton 版本只差 5–10%，维护成本不值得）
- 完全不懂 GPU 但想加速自定义操作：先试 `torch.compile`，不满足再写 Triton

---

## 7.9 本章小结

```
Triton 的核心价值：
  写 Block 级别的逻辑，Triton 处理 Warp/Bank/对齐，生成 PTX
  性能 ~80–95% 手写 CUDA，开发效率远高于 CUDA

三个核心 API：
  tl.load(ptr + offsets, mask=mask)   从 HBM 加载
  tl.store(ptr + offsets, val, mask)  写入 HBM
  tl.dot(a, b)                        触发 Tensor Core MMA

torch.compile 管道：
  Dynamo 捕获 FX 图 → AOTAutograd 生成 forward/backward
  → Inductor 生成 Triton kernel → JIT 编译 → GPU 执行

autotuner：对每种矩阵形状自动搜索最优 BLOCK 配置，结果磁盘缓存

不要用 Triton 重写 GEMM（用 cuBLAS），
用 Triton 做算子融合、自定义 Attention、归一化层。

下一章：MoE 的通信和算子挑战——All-to-All 和 Grouped GEMM。
```

**延伸资料**：
- [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/index.html)（Vector Add、GEMM、Softmax 三个 tutorial 必看）
- [Triton GitHub](https://github.com/openai/triton)
- [TorchInductor 设计文档](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- [FlashAttention 的 Triton 实现](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)（学习 Triton 写复杂 kernel 的最佳范本）
