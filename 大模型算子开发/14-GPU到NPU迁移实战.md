# 14 · GPU→NPU 迁移实战：一份按顺序执行的手册

> **前置**：[12 · 昇腾硬件基础](12-昇腾硬件基础.md)。算子原理散在各章的「在昇腾上」小节里：
> Attention → [04 章 §4.9](04-FlashAttention全系.md)；量化 → [06 章 §6.10](06-量化算子.md)；MoE 与通信 → [08 章 §8.9](08-MoE算子与通信.md)。
> **目标**：把一个在 GPU 上跑通的模型搬到昇腾上，按顺序该做什么、每一步会撞上什么。

---

**本章关键词**

**torch_npu** —— PyTorch 的昇腾后端扩展（Ascend Extension for PyTorch），提供 `.npu()`、NPU 算子和自定义 API。
**aclnn** —— CANN 的单算子 API 层，PyTorch 每个算子最终落到这里。
**ACLGraph** —— 昇腾版的 CUDA Graph：录制一次算子下发序列后重放，消除 Host 下发开销。
**TorchAir** —— 基于 PyTorch Dynamo 的昇腾图模式扩展，把 FX 图转成 GE 图执行。
**GE**（Graph Engine，图引擎）—— CANN 的整图编译执行引擎。
**MindIE** —— 昇腾的推理引擎/加速套件，对标 TensorRT-LLM。
**MindSpeed** —— 昇腾的大模型训练加速库，对标 Megatron-LM 的加速插件层。
**soc_version** —— 芯片子型号标识（如 `Ascend910B3`）。**算子按 soc_version 编译**，这是昇腾特有的运维约束。
**静默错误**（Silent Error）—— 不抛异常、不报警告，只是结果不对的错误。本章 §14.4 单列一节，因为它是迁移中最贵的一类问题。

---

## 14.1 迁移前：先判断这活值不值得干

在改第一行代码之前，用这四个问题估工作量。它们决定了迁移是"两天"还是"两个月"：

```
Q1  模型里有自定义 CUDA kernel 吗？
    没有 → 好消息，主要工作是调参和验证
    有   → 这是最大的工作量，每个 kernel 都要用 Ascend C 重写（13 章）
           先评估能不能用 cann-ops-adv 的现成算子替代

Q2  用到 FP8 了吗？
    是 → 910B/910C 没有 FP8 硬件支持（06 章 §6.10）
         必须退回 BF16，或改走 INT8 W8A8 路线，精度要重新验证

Q3  是 MoE 且规模很大吗？
    是 → 可能是好事。超节点上 EP 的设计空间比 GPU 宽（08 章 §8.9）
         但并行策略要重新调，GPU 上的最优配比不能直接搬

Q4  对逐位一致性有要求吗（如 RL 训练的训推一致）？
    是 → 11 章讲的所有问题在昇腾上同样存在，且实现不同
         跨平台复现基本不可能，要重新建立基线
```

一个经验判断：**没有自定义 kernel、不依赖 FP8 的标准 Transformer 推理部署，迁移是可行且成熟的**；**训练侧、或者依赖大量自定义算子的场景，成本要按"高一个档次"来估**（理由见 §14.7）。

---

## 14.2 第一原则：能不写 kernel 就不写

[13 章](13-AscendC编程.md) 的结论是 Ascend C 的开发效率明显低于 Triton 和 CUDA——同一个逐元素算子，Triton 七行，Ascend C 七十行。所以正确的顺序是**从上往下试，能停在哪层就停在哪层**：

```
① torch_npu 现成的融合算子接口              ← 90% 的情况到这一步就够了
      npu_fusion_attention / npu_rms_norm / npu_rotary_mul ...
      ↓ 没有覆盖
② cann-ops-adv 里的算子（改参数、改布局）
      开源可读，先确认是不是接口用法不对
      ↓ 没有覆盖
③ MindSpeed（训练）/ MindIE、vLLM-Ascend（推理）的上层封装
      ↓ 没有覆盖
④ 读 cann-ops-adv 源码，改一个出来
      它是开源的，这点比 cuDNN 闭源二进制友好
      ↓ 实在没有
⑤ 从零写 Ascend C（13 章）
```

每往下一级，工作量大约翻一倍。**在第 ⑤ 级之前，务必确认前四级都真的走过了**——迁移项目里最常见的浪费，是花两周手写了一个 `cann-ops-adv` 里已经有的算子。

---

## 14.3 四轮迁移 checklist

按顺序执行。**每一轮做完再进下一轮**，尤其不要跳过第二轮直接看性能。

### 第一轮：能不能跑起来

```
□ torch_npu 版本和 CANN 版本对得上吗
   两者强耦合，版本矩阵要查官方文档，错配的报错信息通常没有指向性
□ device 改成 "npu"：.cuda() → .npu()，"cuda:0" → "npu:0"
□ torch.distributed 的 backend：nccl → hccl
□ 自定义 CUDA kernel 有没有？→ 回到 §14.2 的阶梯
□ 用到 FP8 吗？→ 910B/C 上没有，退回 BF16 或改 INT8
□ 用到 GPU 特有 API 吗（如 __shfl_sync、warp 原语）→ 没有对应物，要重写
```

### 第二轮：结果对不对 ← 最重要的一轮，别跳

```
□ atten_mask 取反了吗                    （04 章 §4.9.3 坑 1）
□ sparse_mode 按 flash-attn 版本选对了吗  （04 章 §4.9.3 坑 2）
□ 拿同一批输入，和 GPU 结果做逐层比对
   工具：TorchAir 的精度 dump，或 torch_npu 的 dump 能力
□ 比对不要只看 loss —— 短序列小 batch 上 loss 可能看着正常
□ 量化模型：重新做校准，不要直接搬 GPU 的量化产物（06 章 §6.10.1）
□ 确定性：11 章讲的算子非确定性在昇腾上同样存在，且实现不同
```

### 第三轮：跑得快不快

```
□ 先用 msprof 采集，判断是不是 Host bound（下发瓶颈）
   这是昇腾特有的高频问题，见 §14.5
   → 是：开 ACLGraph 或 TorchAir
   → 否：看指令流水图定位 Cube / Vector / MTE bound（13 章 §13.8）
□ 变长场景用 TND 布局了吗，还是在 padding 到 128 倍数（04 章 §4.9.3 坑 3）
□ 有没有隐式的 ND ↔ NZ 格式转换（12 章 §12.3.3）
   模型里反复横跳是一类典型的隐性损失
□ 用的是融合算子还是一堆小算子拼的（回到 §14.2）
□ MoE：EP 规模重新调了吗、Dispatch/Combine 融合启用了吗（≥64 卡）
```

### 第四轮：放不放得下

```
□ 910B 单卡 64 GB vs H100 80 GB —— 同样的并行配置可能放不下
   910B4 只有 32 GB，差距更大
□ 重计算、KV Cache 量化、W8A8 的取舍重新算
□ 910C 是双 die，卡内有 NUMA 效应（12 章 §12.4.2）
   把一张卡当成均匀加速器做切分会踩坑
```

---

## 14.4 静默错误清单

这一节单列，因为它们是迁移中**最贵**的问题：不抛异常、不报警告，只是结果慢慢不对，往往要到训练跑了几天、或者线上效果掉了才被发现。

| 静默错误 | 现象 | 检查方法 | 出处 |
|---|---|---|---|
| `atten_mask` 没取反 | loss 能降但降不到位；长序列效果差 | 拿 GPU 结果逐元素比对 attention 输出 | [04 §4.9.3](04-FlashAttention全系.md) |
| `sparse_mode` 选错 | 非方阵（S_q ≠ S_kv）时结果错，方阵时正常 | 用 S_q ≠ S_kv 的 case 专门测 | [04 §4.9.3](04-FlashAttention全系.md) |
| 量化产物直接搬运 | 精度掉但不崩 | 在昇腾上重新校准，比对 PPL | [06 §6.10.1](06-量化算子.md) |
| 尾块处理漏了 | 张量末尾少量元素是脏数据 | 用非整除的 shape 专门测 | [13 §13.5.3](13-AscendC编程.md) |
| ND/NZ 格式假设错 | 结果完全错乱（这个通常不静默） | 检查 `npu_format_cast` 调用 | [12 §12.3.3](12-昇腾硬件基础.md) |

**统一的防御手段只有一个：拿 GPU 结果做逐层比对，而不是看 loss 曲线。** 建议在迁移第一天就把比对脚本搭起来，固定一批输入，逐层 dump、逐层比，把误差量级记录下来作为基线。[11 章](11-训推一致性算子层.md) 讲的方法论在这里直接适用——只不过那里比的是"训练 vs 推理"，这里比的是"GPU vs NPU"。

---

## 14.5 图模式：昇腾特有的下发瓶颈

[13 章 §13.8](13-AscendC编程.md) 提到，昇腾上很大比例的性能问题是**下发（dispatch）瓶颈**——Host 侧一个一个把算子发给 Device，Device 算得比发得快。这在 Decode 阶段尤其严重：每步只算一个 token，算子多而小。

这不是昇腾独有的问题（GPU 上 CUDA Graph 就是为此而生，见 [05 章 §5.5](05-算子融合.md)），但在昇腾上更突出，因为单算子下发路径更长。

⚠️ 以下三种模式的对比来自社区经验与工具文档，官方未给出统一的横向对比表：

```
① 单算子模式（Eager）
   torch_npu 逐个下发 aclnn 算子
   ✔ 调试友好、灵活
   ✘ Host 下发开销大，小算子场景必然 host bound

② ACLGraph  ← 对应 CUDA Graph
   录制一次下发序列，之后重放
   ✔ 接入成本低、编译快、消除下发开销
   ✘ 仍走单算子 kernel，不做融合
   ✘ 要求 shape / 地址固定，需要静态内存池配合

③ GE 图模式（TorchAir）  ← 对应 torch.compile
   Dynamo 捕获 FX 图 → 转成 GE 图 → CANN 图编译器做融合、内存复用、整图调度
   ✔ 优化程度最高
   ✘ 编译耗时长
   ✘ 动态 shape 要配分档，否则反复重编译
```

选型（⚠️ 社区口径）：

| 场景 | 推荐 | 理由 |
|---|---|---|
| 开发调试 | 单算子模式 | 能单步、能打印 |
| **Decode 阶段、小 shape、host bound** | **ACLGraph** | 收益明显且接入简单，是性价比最高的一步 |
| 训练 / Prefill，shape 稳定 | TorchAir GE 图模式 | 要极致融合，能接受编译耗时 |

> **一个额外收益**：TorchAir 提供图导出和**精度 dump**能力，可以把 GE 图里每个节点的输出落盘。这是 §14.4 逐层比对的主力工具，也是排查 [11 章](11-训推一致性算子层.md) 训推一致性问题的利器。

---

## 14.6 映射总表

把散在各章的对应关系汇总到一处，迁移时对着查。

### 14.6.1 软件栈

| NVIDIA | 昇腾 | 说明 |
|---|---|---|
| CUDA（全家桶） | CANN | 软件栈总称 |
| CUDA Runtime API | AscendCL（`acl*`） | 运行时 |
| CUDA C++ | Ascend C | kernel 编程语言（[13 章](13-AscendC编程.md)） |
| **Triton** | **无广泛采用的等价物** | 昇腾最明显的生态缺口 |
| cuBLAS / cuDNN | aclnn 算子库 | 基础算子 |
| CUTLASS | cann-ops-adv（**开源**） | 高阶/融合算子 |
| NCCL | HCCL | 集合通信（[08 章 §8.9.1](08-MoE算子与通信.md)） |
| CUDA Graph | ACLGraph | 消除下发开销 |
| torch.compile / Inductor | TorchAir（GE 图） | 图编译 |
| TensorRT-LLM | MindIE | 推理引擎 |
| Megatron-LM 加速层 | MindSpeed | 训练加速 |
| Nsight Compute / Systems | msprof + MindStudio Insight | 性能剖析 |
| `nvidia-smi` | `npu-smi info` | 设备查询 |
| AutoGPTQ / llm-compressor | msModelSlim / AMCT | 量化工具（[06 章 §6.10](06-量化算子.md)） |

### 14.6.2 PyTorch 侧代码改动

| GPU | NPU |
|---|---|
| `import torch` | `import torch; import torch_npu` |
| `.cuda()` / `.to("cuda")` | `.npu()` / `.to("npu")` |
| `torch.cuda.xxx` | `torch_npu.npu.xxx` |
| `backend="nccl"` | `backend="hccl"` |
| `F.scaled_dot_product_attention` | `torch_npu.npu_fusion_attention`（**注意 mask 取反**） |
| `flash_attn_func` | `torch_npu.npu_fusion_attention` |
| `flash_attn_varlen_func` | `npu_prompt_flash_attention` 或 TND 布局的 `npu_fusion_attention` |
| PagedAttention decode | `npu_incre_flash_attention` |
| `torch.cuda.CUDAGraph` | ACLGraph |
| `torch.compile` | TorchAir |

### 14.6.3 排查命令

```bash
npu-smi info                          # 对应 nvidia-smi
npu-smi info -t usages -i 0           # AICore/AIV 利用率、HBM 带宽占用

msprof op --output=./prof ./my_test   # 对应 ncu，采集
msprof --export=on --output=./prof    # 解析
msprof-analyze advisor all -d ./prof  # 自动诊断 + 专家建议（GPU/NPU 比对 recipe 也在这）
# 可视化：MindStudio Insight → 指令流水图

# 查芯片真实参数（唯一可信来源，12 章 §12.4.3）
find /usr/local/Ascend -type d -name "platform_config" 2>/dev/null
```

---

## 14.7 诚实的现状评估

把全系列关于昇腾的事实汇总，回答"到底能不能用"。

**已经成熟**

- Attention 融合算子齐备（训练 / Prefill / Decode 三件套 + 统一算子），且**开源可读**。
- HCCL 功能对齐 NCCL，NHR 算法在非 2 次幂集群上有优势，全硬化调度理论上重叠更干净。
- W8A8 量化工具链完整，DeepSeek 等主流模型有现成方案。
- vLLM、SGLang 都已有 Ascend backend，MindIE 作为官方推理引擎。
- **超节点在大规模 MoE 场景下是结构性优势**（[08 章 §8.9.3](08-MoE算子与通信.md)）。
- CPU 孪生调试是昇腾相对 CUDA 的一个真实优势（[13 章 §13.7.1](13-AscendC编程.md)）。

**明确的短板**

| 短板 | 影响 | 出处 |
|---|---|---|
| **算子开发效率** | Ascend C 比 Triton 陡峭得多，无广泛采用的高层 DSL | [13 章](13-AscendC编程.md) |
| **卡间互联带宽** | ⚠️ ~392 GB/s vs NVLink 900 GB/s，单机内 TP 规模受限 | [12 章 §12.5.1](12-昇腾硬件基础.md) |
| **没有 FP8** | 低精度落后一代，HiF8 要等下一代硬件 | [06 章 §6.10](06-量化算子.md) |
| **型号碎片化** | 910B 的 B1/B2/B3/B4 核数带宽差异巨大，算子按 soc_version 编译 | [12 章 §12.4.1](12-昇腾硬件基础.md) |
| **规格不透明** | 无公开完整数据手册，二手数字互相矛盾 | [12 章 §12.4](12-昇腾硬件基础.md) |
| **生态滞后** | 新模型、新算法的 Day-0 支持基本都在 CUDA 上 | — |

**一句话结论**：主流的训练和推理路径已经能跑通且有优化空间；但**任何需要自己写 kernel 的工作，成本比 GPU 高一个档次**。做技术选型时，这个成本要老实算进去，不要只对比硬件标称算力。

---

## 14.8 小结

```
迁移前先估工作量：自定义 kernel？FP8？MoE 规模？逐位一致性要求？
  没有自定义 kernel 的标准推理部署 → 可行且成熟
  训练侧 / 大量自定义算子           → 成本高一个档次

第一原则：能不写 kernel 就不写
  torch_npu 接口 → cann-ops-adv → 上层框架 → 改源码 → 从零写
  每往下一级工作量翻倍；最常见的浪费是手写了已经有的算子

四轮 checklist，按顺序做，别跳第二轮
  ① 跑起来（版本匹配、device、backend、FP8 退路）
  ② 结果对（逐层比对，不看 loss）  ← 最重要
  ③ 跑得快（先排除 Host bound，再看流水图）
  ④ 放得下（64 GB vs 80 GB、910C 双 die NUMA）

静默错误是最贵的
  atten_mask 取反、sparse_mode、量化产物直接搬、尾块
  唯一防御：第一天就搭好逐层比对脚本

图模式治下发瓶颈
  Eager 调试 / ACLGraph 性价比最高 / TorchAir 求极致融合
  TorchAir 的精度 dump 是逐层比对的主力工具

现状：主流路径能用，写 kernel 的成本高一个档次
```

---

**延伸资料**：
- [Ascend Extension for PyTorch 迁移调优指南](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)（官方，迁移坑的权威说明）
- [cann-ops-adv 融合算子库](https://gitee.com/ascend/cann-ops-adv)（开源，先在这里找现成算子）
- [msprof-analyze](https://gitcode.com/Ascend/mstt)（开源，含 GPU/NPU 性能比对 recipe）
- [vLLM Ascend backend RFC](https://github.com/vllm-project/vllm/issues/7692)（社区，看接入方式与遗留问题）
