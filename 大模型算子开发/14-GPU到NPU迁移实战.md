# 14 · GPU→NPU 迁移实战：一条能照着走的排查路径

> **前置**：[12 · 昇腾硬件基础](12-昇腾硬件基础.md)、[13 · Ascend C 编程](13-AscendC编程.md)。算子原理散在各章的「在昇腾上」小节里：
> Attention → [04 章 §4.9](04-FlashAttention全系.md)；量化 → [06 章 §6.10](06-量化算子.md)；MoE 与通信 → [08 章 §8.9](08-MoE算子与通信.md)。
> **目标**：把一个在 GPU 上跑通的模型搬到昇腾上。本章给的不是一张名词清单，是**一条可执行的排查路径**：每一步该做什么、**怎么判断这一步过了**、过不去的时候怎么定位。
> **读法**：本章的核心是 §14.4（精度对齐）。如果你只有时间读一节，读那一节——迁移项目里绝大多数的时间损失都发生在那里。

---

**本章关键词**

**CANN**（Compute Architecture for Neural Networks，神经网络计算架构）—— 昇腾的软件栈总称，地位相当于 CUDA 全家桶。
**torch_npu** —— PyTorch 的昇腾后端扩展（Ascend Extension for PyTorch），提供 `.npu()`、NPU 算子和自定义 API。
**aclnn** —— CANN 的单算子 API 层，PyTorch 每个算子最终落到这里。
**ACLGraph** —— 昇腾版的 CUDA Graph：录制一次算子下发序列后重放，消除 Host 下发开销。
**TorchAir**（Torch Ascend Intermediate Representation）—— 基于 PyTorch Dynamo 的昇腾图模式扩展，把 FX 图转成 GE 图执行。
**GE**（Graph Engine，图引擎）—— CANN 的整图编译执行引擎。
**MindIE**（Mind Inference Engine）—— 昇腾的推理引擎/加速套件，对标 TensorRT-LLM。
**MindSpeed** —— 昇腾的大模型训练加速库，对标 Megatron-LM 的加速插件层。
**PFA**（Prompt Flash Attention，提示词闪电注意力）—— 昇腾的推理 Prefill 阶段 Attention 算子。
**IFA**（Incremental Flash Attention，增量闪电注意力）—— 昇腾的推理 Decode 阶段 Attention 算子，query 序列长固定为 1。
**msprobe**（MindStudio Probe，精度调试工具）—— mstt 工具链里做精度 dump 与 GPU/NPU 逐层比对的工具。**本章第 2 关的主力工具**。
**msFmkTransplt**（MindStudio Framework Transplant，框架迁移工具）—— 官方的 PyTorch 脚本自动迁移与算子支持度分析工具，也叫 PyTorch GPU2Ascend。
**msprof / MindStudio Insight** —— 昇腾的性能采集与可视化工具，对标 Nsight Compute / Nsight Systems。
**soc_version** —— 芯片子型号标识（如 `Ascend910B3`）。**算子按 soc_version 编译**，这是昇腾特有的运维约束。
**静默错误**（Silent Error）—— 不抛异常、不报警告，只是结果不对的错误。本章 §14.4 单列一节，因为它是迁移中最贵的一类问题。

> **关于本章数字的可信度标记**（沿用 [12 章](12-昇腾硬件基础.md) 的纪律）
> **✅ 披露** = 华为官方产品页 / CANN 官方文档 / 官方工具文档明确写出的；
> **⚠️ 推断** = 第三方分析、社区实测、工具源码常量，或本章为了演示流程而设的假设值。
> 本章出现的所有「人天」「周」「百分比门槛」**全部是 ⚠️ 假设值，只用来演示怎么算这笔账，不要直接抄进你的项目计划**。

---

## 14.1 迁移全景：五道关卡，每道关卡有一条硬标准

迁移之所以容易失控，不是因为哪一步特别难，而是因为**没人说得清「这一步到底算不算做完了」**。下面先把关卡和验收标准摆出来。

### 14.1.1 一张总流程图

```
                      GPU 上跑得好好的模型
                              │
                              ▼
   ┌─── 第 0 关  环境 ────────────────────────────────
   │    驱动 / CANN / torch / torch_npu 四者版本对上
   │    硬标准：npu-smi 看得到卡，一个张量能 .npu()
   │    过不去的典型症状：undefined symbol、找不到设备
   └──────────────┬───────────────────────────────────
                  ▼
   ┌─── 第 1 关  跑通（Functional）───────────────────
   │    .cuda→.npu、nccl→hccl、自定义 kernel 有没有替代
   │    硬标准：前向出数不 NaN；训练侧连跑 50 步不崩
   │    过不去的典型症状：算子 not implemented、报错码
   └──────────────┬───────────────────────────────────
                  ▼
   ┌─── 第 2 关  精度对齐（Numerical）★ 最难最贵 ─────
   │    和 GPU 逐层比对，找出第一个不达标的层并解释它
   │    硬标准：见 §14.4.4 的指标与阈值
   │    过不去的典型症状：**没有症状** —— 它不报错
   └──────────────┬───────────────────────────────────
                  ▼
   ┌─── 第 3 关  性能（Performance）──────────────────
   │    先答「瓶颈在 Host 还是 Device」，再答「哪一级流水」
   │    硬标准：Device 计算时间占端到端 > 80%
   │    过不去的典型症状：利用率低，但说不出瓶颈在哪
   └──────────────┬───────────────────────────────────
                  ▼
   ┌─── 第 4 关  上线（Capacity & Stability）─────────
   │    显存放得下、多卡起得来、长时间跑得住
   │    硬标准：峰值显存 ≤ 单卡容量 × 0.9，24 小时不掉卡
   │    过不去的典型症状：跑一小时后 OOM
   └──────────────┬───────────────────────────────────
                  ▼
                 上线
```

### 14.1.2 「这一关过了没有」：五条硬标准

下面这张表是本章最该贴在工位上的一张。**门槛数字全部是 ⚠️ 经验值**，不同项目可以调，但一定要在开工前把它们写下来——否则第 2 关永远没有「做完」的那一天。

| 关 | 名字 | 硬标准（⚠️ 建议门槛） | 用什么确认 | 大概花多久（⚠️ 假设） |
|---|---|---|---|---|
| 第 0 关 | 环境 | `npu-smi info` 能列出全部卡；`torch.randn(4).npu() * 2` 不报错 | 一条命令 + 一行 Python | 0.5–2 天 |
| 第 1 关 | 跑通 | 单步前向输出无 `NaN` / `Inf`；训练侧连跑 50 步不崩，loss 量级和 GPU 同一档 | 裸跑 + 一个 `assert torch.isfinite(out).all()` | 2–8 天 |
| 第 2 关 | 精度 | 逐层比对里**第一个**不达标的条目能被解释清楚（是浮点噪声，还是语义不同）；端到端固定输入贪心解码，输出 token 与 GPU 的一致长度 ≥ 你能接受的值 | `msprobe compare` | 5–15 天 |
| 第 3 关 | 性能 | Device 计算时间 ÷ 端到端时间 > 80%；指令流水图里最高的那一级占比 > 70% | `msprof` + MindStudio Insight | 10–20 天 |
| 第 4 关 | 上线 | 峰值显存 ≤ 单卡容量 × 0.9；目标并发下连续 24 小时不 OOM、不掉卡 | `npu-smi info -t usages` + 长稳测试 | 5–10 天 |

> ⚠️ 第 2 关那条「输出 token 一致长度」要现实一点。BF16 下 GPU 和 NPU **不可能逐位一致**（原因见 [11 章 §11.4](11-训推一致性算子层.md)：累加顺序、SplitK 切分、归约拓扑都不同），所以不要把「完全一致」写进验收标准。可行的写法是：贪心解码前 N 个 token 一致（N 取 32 或 64 都算合理），或者固定测试集上困惑度（PPL）相对差 < 0.5%。

### 14.1.3 顺序为什么不能乱：把「跳过第 2 关」的代价算出来

最常见的错误是**先调性能、最后才验精度**。把两条时间线并排放，损失就看得清了（⚠️ 周数为假设值，只演示这个顺序的价值）。

```
团队 A：跑通 → 直接调性能 → 上线前才验精度
  第 1 周     跑通
  第 2–4 周   性能：换 TND 布局、开 ACLGraph、调 EP 配比、扫 tiling
              吞吐从 1.0 提到 2.3，写了一份很漂亮的调优报告
  第 5 周     上线前做精度验收 → 发现 atten_mask 没取反（§14.4.6 坑 1）
              修完 mask 之后，attention 走的分支变了
              → 第 2–4 周那些结论（TND 收益多少、EP 开几、tiling 取多大）
                全部是在错误数值上测出来的，必须重测
  净损失 ≈ 3 周

团队 B：跑通 → 精度对齐 → 再调性能
  第 1 周     跑通
  第 2 周     搭逐层比对脚本；第 3 天查出 atten_mask，第 5 天查出 sparse_mode
  第 3–5 周   性能
  合计 5 周，且每一个性能结论都建立在「数值正确」的基线上
```

> ★ **精度是性能的前置条件，不是和性能并列的一项。** 理由很机械：性能优化的每一个动作（换布局、换算子、开图模式、改并行度）都会改变数值路径，如果基线数值本身是错的，你没有任何办法判断「性能变好的同时精度变没变」。所以**第 2 关必须在第 3 关之前关闭**，这是本章唯一一条不可协商的顺序。

### 14.1.4 开工前四问，和一张工作量估算表

在改第一行代码之前，用这四个问题估量级。它们决定了迁移是「两周」还是「两个季度」：

```
Q1  模型里有自定义 CUDA kernel 吗？
    ├── 没有 → 好消息。主要工作在精度对齐和调参
    └── 有   → 这是最大的一块。每个 kernel 都要用 Ascend C 重写（13 章）
               先走一遍 §14.2 的阶梯，确认不能用现成算子替代

Q2  用到 FP8 了吗？
    ├── 没有 → 跳过
    └── 用了 → 910B/910C 没有 FP8 硬件支持（✅，见 06 章 §6.10）
               必须退回 BF16，或改走 INT8 W8A8 路线
               ★ 注意这不只是"改个 dtype"：精度要重新验证，
                 量化校准要在昇腾上重做（§14.4.6 坑 5）

Q3  是 MoE 且规模很大吗（≥ 64 卡）？
    ├── 不是 → 跳过
    └── 是   → 可能是好事。超节点上 EP 的设计空间比 GPU 宽（08 章 §8.9.3）
               但 GPU 上调出来的 EP/TP 最优配比不能直接搬，要重新扫

Q4  对逐位一致性有要求吗（如 RL 训练的训推一致）？
    ├── 没有 → 跳过
    └── 有   → 11 章讲的所有问题在昇腾上同样存在，且实现不同
               ★ 跨平台复现基本不可能，要在昇腾上重新建立基线，
                 不要拿 GPU 的基线当验收标准
```

有了这四个答案，工作量就可以按 §14.2 的阶梯**逐个算子累加**。算法是：

```
总工作量 = 固定成本（环境 + 精度对齐 + 长稳）
         + Σ（该级算子个数 × 该级单个成本）

阶梯级别（§14.2.1）              单个算子成本（⚠️ 全部为假设值，仅演示算法）
  ① torch_npu 有现成接口            0.5 人天   —— 改调用 + 核对返回值
  ② 现成但要改参数 / 改布局          1–3 人天   —— 主要花在核语义
  ③ 上层框架（MindSpeed 等）已封装   0.5 人天   —— 查文档 + 验证
  ④ 读 cann-ops-adv 源码改一个       5–10 人天
  ⑤ 从零写 Ascend C                 10–20 人天
```

**案例 A：Llama 结构 7B 稠密模型，纯推理部署，无自定义 kernel**（⚠️ 全表为假设值）

| 环节 | 内容 | 人天 |
|---|---|---|
| 环境 | 镜像、四方版本对齐 | 2 |
| 跑通 | 自动迁移 + 逐个 device / backend | 3 |
| 算子替换 | 约 12 个算子，全在 ① 级 | 6 |
| **精度对齐** | 搭比对脚本 3 + 逐层排查 5 | **8** |
| 性能 | ACLGraph + TND 布局 + 读流水图 | 10 |
| 上线 | 显存预算、长稳 | 5 |
| | **合计** | **≈ 34 人天** |

**案例 B：DeepSeek 结构 MoE 训练，含 2 个自研融合 kernel**（⚠️ 全表为假设值）

| 环节 | 内容 | 人天 |
|---|---|---|
| 环境 | 多机、HCCL 组网 | 4 |
| 跑通 | 分布式 + MindSpeed 接入 | 8 |
| 算子替换 | MoE 链路 6 个在 ② 级 | 12 |
| **自研 kernel 重写** | 2 个 × 15 人天（⑤ 级） | **30** |
| **精度对齐** | 训练侧要看长期 loss 曲线，不是跑一遍就完 | **15** |
| 性能 | EP 配比重扫 + 通算融合 + 流水图 | 20 |
| 上线 | 长稳、断点续训、掉卡恢复 | 10 |
| | **合计** | **≈ 99 人天** |

> ★ 两个案例差了近 3 倍，差距**不在模型大小**，在两件事：**自研 kernel 的个数**（30 人天，占案例 B 的三分之一），和**训练 vs 推理**（训练的精度对齐要观察几天的 loss 曲线，推理跑一遍就知道）。所以 Q1 和「训练还是推理」是估工作量时最该先问清的两个问题，比参数量重要得多。

> ★ 还有一行值得单独看：**「精度对齐」是唯一一项不随模型大小缩放的固定成本**。搬一个 0.5B 的小模型，精度对齐照样要 5–8 人天——比对脚本要搭、指标要定、坑要一个个踩。所以「先拿个小模型试试水」这个直觉，省不下多少时间。

---

## 14.2 第一原则：能不写 kernel 就不写

[13 章 §13.9](13-AscendC编程.md) 的结论是 Ascend C 的开发效率明显低于 Triton 和 CUDA——同一个逐元素算子，Triton 约 7 行，Ascend C 约 70 行。所以正确的顺序是**从上往下试，能停在哪层就停在哪层**。

### 14.2.1 五级阶梯，每一级配一个「怎么确认这级真的没有」的动作

迁移时真正卡人的从来不是「不知道有阶梯」，而是「怎么确认第 ① 级真的没覆盖」——判断不了，就只好凭感觉往下走一级，于是就有了「花两周手写一个已经存在的算子」这种事。所以下面每一级后面都配一个确认动作：

```
① torch_npu 现成的融合算子接口                  ← 90% 的情况到这一步就够了
     npu_fusion_attention / npu_rms_norm / npu_rotary_mul / npu_swiglu ...
     怎么确认这级没有：
       a) 把本版本提供的所有自定义算子列出来（见 §14.2.2 命令 ①）
       b) 去昇腾社区「Ascend Extension for PyTorch 自定义 API」列表里按
          **你装的版本号**查 —— 同一个算子在 6.0.RC1 和 7.x 的参数表差异很大
     ↓ 确认没有
② cann-ops-adv 里的算子（改参数、改布局）
     它是开源的，先确认是不是自己接口用法不对，而不是真的缺算子
     怎么确认：clone 下来 grep（见 §14.2.2 命令 ②）
     ↓ 确认没有
③ MindSpeed（训练）/ MindIE、vLLM-Ascend（推理）的上层封装
     这些仓库已经替你踩过大部分坑，它们怎么调，你就怎么调
     怎么确认：grep 这些仓库里的 npu_ 调用（见 §14.2.2 命令 ③）
     ↓ 确认没有
④ 读 cann-ops-adv 源码，改一个出来
     开源这一点比 cuDNN 闭源二进制友好得多
     ↓ 实在没有
⑤ 从零写 Ascend C（13 章）
```

> ★ 每往下一级，工作量大约翻一倍（§14.1.4 的成本表就是按这个规律填的）。**在第 ⑤ 级之前，务必确认前四级都真的走过了**——迁移项目里最常见的浪费，是花两周手写了一个 `cann-ops-adv` 里已经有的算子。

### 14.2.2 三条命令，三分钟确认有没有现成算子

这一节的价值在于：把「有没有现成的」从一个**需要问人的问题**，变成一个**三分钟能自己回答的问题**。

```bash
# ① 列出你这个 torch_npu 版本提供的全部自定义算子（最快，先做这个）
python3 -c "import torch, torch_npu; print('\n'.join(sorted(n for n in dir(torch_npu) if n.startswith('npu_'))))"

# ①' 找到之后看它的签名和文档字符串
python3 -c "import torch, torch_npu; help(torch_npu.npu_grouped_matmul)"

# ② 在开源的 cann-ops-adv 里搜（算子名、也可以搜论文里的算法名）
git clone https://gitee.com/ascend/cann-ops-adv
grep -ril "rmsnorm" cann-ops-adv/ | head -20

# ③ 看别人是怎么调的 —— 这一步常常比读文档快
#    vLLM-Ascend / MindSpeed / SGLang 的 NPU 分支里都有真实调用
grep -rn "npu_fusion_attention" <vllm-ascend 或 MindSpeed 仓库路径>
```

> ⚠️ **命令 ① 只能告诉你「接口在不在」，不能告诉你「这个版本的参数长什么样」。** `npu_grouped_matmul` 是个现成的例子：6.0.RC1 的签名只有 `x / weight / bias / group_list / split_item / output_dtype` 几个参数，到 7.x 已经扩到十几个（多了 `scale`、`per_token_scale`、`group_type`、`group_list_type`、`tuning_config` 等，✅ 官方 API 文档）。**照着网上搜到的代码片段抄，撞上版本不同是迁移里最高频的时间浪费之一**——永远以 `help()` 输出和你装的版本对应的那份文档为准。

### 14.2.3 一个 Llama 结构的逐算子盘点：开箱即用 / 要改 / 要重写

把一层 Decoder 从头到尾走一遍，每个算子落在阶梯的哪一级。这张表就是 §14.1.4 那个工作量公式的输入。

| 环节 | GPU 侧典型写法 | 昇腾侧 | 阶梯 | 要注意什么 |
|---|---|---|---|---|
| Embedding | `nn.Embedding` | 原生支持 | ① 开箱 | — |
| RMSNorm | 手写或 apex 融合版 | `torch_npu.npu_rms_norm` | ① | ⚠️ 返回值个数按版本查（不一定只返回 y） |
| RoPE | 手写 `apply_rotary_pos_emb` | `torch_npu.npu_rotary_mul` | ① | 输入 layout 约定和 GPU 侧不一定一样，**必须核** |
| QKV / O Linear | `F.linear` | 原生支持 | ① | 权重要不要转 NZ，见 §14.5.4 |
| **Attention（训练）** | `flash_attn_func` | `npu_fusion_attention` | ① | ★ **四个坑全在这一行**，见 §14.4.6 |
| Attention（Prefill） | `flash_attn_varlen_func` | `npu_prompt_flash_attention`，或 TND 布局的 `npu_fusion_attention` | ① | 变长必须走 TND |
| Attention（Decode） | PagedAttention kernel | `npu_incre_flash_attention` | ① | `block_table` / `block_size` 要对齐 |
| SwiGLU | `silu(gate) * up` | `torch_npu.npu_swiglu`（✅ 官方 API） | ① | 签名 `npu_swiglu(self, dim=-1)` |
| MoE 路由 | `torch.topk` + 排序 | `npu_moe_gating_top_k_softmax` + `npu_moe_init_routing`(`_v2`) | ② | `group_list` 语义有 cumsum / count 两种，版本相关 |
| MoE Grouped GEMM | grouped_gemm / Triton | `npu_grouped_matmul`（✅ 官方 API） | ② | 参数表版本差异大（见 §14.2.2 的 ⚠️） |
| MoE 结果合并 | unpermute | `npu_moe_finalize_routing` | ② | — |
| EP 通信 | DeepEP dispatch / combine | HCCL + 通算融合（≥64 卡，[08 §8.9.2](08-MoE算子与通信.md)） | ② | GPU 上的 EP 配比要重扫 |
| W8A8 量化 | llm-compressor / AutoGPTQ | msModelSlim **重新校准**（[06 §6.10.1](06-量化算子.md)） | ② | ★ 不能搬 GPU 的量化产物 |
| FP8 | 原生 | ❌ 无硬件支持 | — | 退 BF16 或改 INT8 |
| 自研 fused kernel | `.cu` | 无对应物 | ⑤ | 每个 10–20 人天（⚠️ 假设值） |

> ★ **数一下落在 ⑤ 级那几行有几个，基本就能估出这个项目的量级。** 一个都没有 → 案例 A 那一档；有两三个 → 案例 B 那一档。

---

## 14.3 逐关排查：每一条 checklist 都配一个确认动作

迁移 checklist 最没用的形态是一串名词（「检查版本」「检查 backend」）——你没法判断自己到底检查过了没有。这一节按 §14.1 的关卡把 checklist 重排，**每一条都配一个能执行的确认动作**：一条命令、一个 assert，或者一段要 grep 的关键字。**第 2 关（精度）在 §14.4，第 3 关（性能）在 §14.5**，这里讲第 0、1、4 关。

### 14.3.1 第 0 关 · 环境：版本矩阵，和错配长什么样

昇腾侧有**四个东西的版本要同时对上**：驱动/固件、CANN、PyTorch、torch_npu。其中 torch_npu 的版本号前三段跟着 PyTorch 走（形如 `2.1.0.postX`），这一点是多数人第一天就会撞到的。

```bash
# ① 卡和驱动在不在
npu-smi info

# ② 三件套的版本各是多少
python3 -c "import torch; print('torch      ', torch.__version__)"
python3 -c "import torch, torch_npu; print('torch_npu  ', torch_npu.__version__)"
ls -d /usr/local/Ascend/ascend-toolkit/*            # CANN 版本目录

# ③ 最小冒烟：一个张量能不能上卡、能不能算
python3 -c "import torch, torch_npu; print(torch_npu.npu.is_available(), torch_npu.npu.device_count())"
python3 -c "import torch, torch_npu; print(torch.randn(4).npu() * 2)"

# ④ 这张卡到底是哪一款（12 章 §12.4.1：910B 的 B1/B2/B3/B4 差异巨大）
python3 -c "import torch, torch_npu; print(torch_npu.npu.get_device_name(0))"
```

错配的报错信息通常没有指向性，下面这张对照表能省掉半天（⚠️ 症状归纳自社区经验，具体错误码随版本变化）：

| 症状 | 多半是什么 | 下一步 |
|---|---|---|
| `import torch_npu` 报 `undefined symbol: _ZN3c10...` | torch 和 torch_npu 的 C++ ABI 对不上，即大版本不匹配 | 按 torch 版本重装对应的 torch_npu |
| `import torch_npu` 卡很久然后报驱动错 | 驱动/固件版本和 CANN 不匹配 | `npu-smi info` 看驱动版本，对 CANN 的配套表 |
| `torch_npu.npu.is_available()` 返回 `False` 但 `npu-smi` 正常 | 环境变量没 source（`set_env.sh`） | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 某个算子报 `aclnnXxx` 找不到 | CANN 太老，这个版本还没有这个算子 | 回到 §14.2 的阶梯找替代 |
| 形如 `EZ****` / `EE****` 的错误码，Python 栈没有信息 | 错误发生在 CANN 内部，Python 侧只拿到码 | **去看 plog**，见下 |

**最后一行最值得单独说**：Python 侧的报错常常只有一个错误码，真正的信息在 CANN 的 plog 里。

```bash
# plog 默认落在用户目录下（⚠️ 路径随版本和环境变量变，先找一下）
find ~ -maxdepth 4 -type d -name "plog" 2>/dev/null
# 找到之后按时间取最新的一个，搜 ERROR
ls -t <plog目录>/*/ | head
```

> ★ **「Python 报错看不懂就去看 plog」是昇腾侧最该养成的一个习惯**，相当于 CUDA 侧的 `CUDA_LAUNCH_BLOCKING=1` + `cuda-gdb`。第 0 关和第 1 关卡住的时候，十有八九答案就在 plog 里。

### 14.3.2 第 1 关 · 跑通：三种改法，先用哪一种

| 改法 | 怎么做 | 适用 | 代价 |
|---|---|---|---|
| **A 自动迁移** | 训练入口文件里，`import torch` 之后加一行 | 没有自定义 kernel、没有 jit 的常规场景 | 有几条已知限制（见下） |
| **B 工具迁移** | `msFmkTransplt` / PyTorch GPU2Ascend | 想要一份算子支持度报告 | 要装几个 Python 依赖 |
| **C 手工改** | 逐处替换 | 有自定义算子、要精细控制 | 改动点多而分散 |

**改法 A：自动迁移**（✅ CANN 官方「分析迁移工具 · 自动迁移」文档）

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu   # ← 必须在 import torch 之后
```

原理是 monkey-patch：运行时把 `torch.cuda.*`、`.cuda()`、`"cuda:0"` 这些接口注入重定向到 NPU 对应实现。官方明确列出的限制（✅）有三条，每一条都会在迁移时真实撞上：

```
限制 1  backend="nccl" 会在 init_process_group 之后被自动换成 hccl，
        但代码里形如 assert backend in ['gloo', 'nccl'] 或 if backend == 'nccl'
        的判断【不会】被改 —— 这些地方要手动改
        ★ 确认动作：grep -rn "nccl" 你的代码库，逐个看

限制 2  不支持 channels_last，要用 contiguous() 替代
        ★ 确认动作：grep -rn "channels_last" 你的代码库

限制 3  和 torch.jit.script 冲突，工具会屏蔽 jit
        ★ 确认动作：grep -rn "jit.script\|jit.trace" 你的代码库
        撞上了就改走改法 B 或 C
```

**改法 B：工具迁移**（✅ CANN 官方 `msFmkTransplt`）

```bash
cd /usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/
./pytorch_gpu2npu.sh -i <原始脚本目录> -o <迁移输出目录> -v 2.1.0
#                     ↑ 输入          ↑ 输出           ↑ 原脚本的 torch 版本
# 依赖（✅ 官方列出）：pip3 install pandas>=1.2.4 libcst prettytable jedi
```

产出三样东西：迁移后的脚本目录（后缀 `_msft`）、`msFmkTranspltlog.txt`（转换日志）、**`unsupported_op.xlsx`（不支持算子清单）**。

> ★ **就算你打算全部手工改，也值得先跑一遍改法 B**——它产出的 `unsupported_op.xlsx` 就是 §14.2.3 那张盘点表的自动版本，五分钟换来一份「哪些算子要单独处理」的清单，这是本关性价比最高的一个动作。

> ⚠️ 官方文档口径是该工具支持 PyTorch 1.11.0 / 2.1.0 版本的训练脚本（版本清单随 CANN 版本变动，以你装的那版文档为准）。另外官方点名 APEX 的 `FusedAdam` 优化器不支持自动迁移，要自己改。

**改法 C：手工改**——改动点见 §14.6.2 的对照表。

**第 1 关的验收动作**（对应 §14.1.2 的硬标准）：

```python
# 推理侧：一次前向，确认没有 NaN / Inf
out = model(fixed_input)
assert torch.isfinite(out).all(), "第 1 关没过：输出里有 NaN 或 Inf"

# 训练侧：连跑 50 步，确认 loss 不发散、量级和 GPU 同一档
#   ★ 注意这一步【不能】用来判断精度对不对 —— 那是第 2 关的事。
#     这里只回答"跑不跑得动"，不回答"跑得对不对"。
```

### 14.3.3 第 4 关 · 放得下：把显存预算算出来

910B3 单卡 64 GB vs H100 80 GB，差 20%——听起来不多，但**吃掉这 20% 的是 KV Cache，而 KV Cache 决定最大并发**，所以实际影响被放大了。把账算一遍（⚠️ HBM 容量取自 [12 章 §12.4.1](12-昇腾硬件基础.md) 的推断值，仅演示算法）：

```
场景：70B BF16 模型推理，TP = 8

  权重      70e9 × 2 Byte = 140 GB，切 8 张卡 → 17.5 GB/卡
  按 90% 可用率留余量：

    H100    80 GB × 0.9 − 17.5 = 54.5 GB  可给 KV Cache
    910B3   64 GB × 0.9 − 17.5 = 40.1 GB  可给 KV Cache
    910B4   32 GB × 0.9 − 17.5 = 11.3 GB  可给 KV Cache

  KV Cache 预算之比：
    910B3 / H100 = 40.1 / 54.5 = 74%    → 同样的序列长度，最大并发少 26%
    910B4 / H100 = 11.3 / 54.5 = 21%    → 只剩五分之一
```

> ★ 注意最后一行的含义：**同一份部署方案，在 910B3 上跑得动、在 910B4 上根本起不来。** 这不是「性能差一点」，是「起不来」。所以第 0 关的 ④「这张卡到底是哪一款」不是走过场——**它直接决定第 4 关的结论**，而 910B 系列的子型号差异在采购单上经常看不出来（[12 章 §12.4.1](12-昇腾硬件基础.md)）。

不够用的时候，三个旋钮按这个顺序拧：

| 旋钮 | 省多少 | 代价 | 出处 |
|---|---|---|---|
| KV Cache 量化（INT8） | KV Cache 减半 | 长上下文下有精度损失，要验 | IFA 算子原生带反量化参数（[06 §6.10.3](06-量化算子.md)） |
| W8A8 权重量化 | 权重减半（17.5 → 8.75 GB/卡） | 必须在昇腾上**重新校准** | [06 §6.10.1](06-量化算子.md) |
| 提高 TP 度 | 线性减少 | 卡间互联带宽是短板（⚠️ ~392 GB/s vs NVLink 900 GB/s），TP 规模受限 | [12 §12.5.1](12-昇腾硬件基础.md) |

> ⚠️ **910C 是双 die 封装，卡内也有 NUMA（Non-Uniform Memory Access，非统一内存访问）效应**（[12 章 §12.4.2](12-昇腾硬件基础.md)）。把一张 910C 当成「一个均匀的加速器」来做切分会踩坑——两个 die 之间的通信代价和 die 内不同。做 TP 切分和通信编排时要把它当成 2 个单元看。

### 14.3.4 四关 checklist 汇总

把上面的内容压成一张可以直接对着划勾的表。**每一行都有确认动作**，没有确认动作的条目不算条目。

**第 0 关 · 环境**

```
□ npu-smi info 能列出全部卡                    → 列不出来先查驱动
□ torch / torch_npu 版本匹配                   → §14.3.1 命令 ②
□ set_env.sh 已 source                         → is_available() 返回 True
□ 确认卡的子型号（B2/B3/B4 / 910C）            → get_device_name(0)，影响第 4 关
□ 知道 plog 在哪                               → find ~ -name "plog"
```

**第 1 关 · 跑通**

```
□ device 改成 npu：.cuda() → .npu()，"cuda:0" → "npu:0"
□ 分布式 backend：nccl → hccl                  → 并 grep 'nccl' 看有没有硬编码判断
□ 自定义 CUDA kernel 有没有                    → 回到 §14.2 的阶梯
□ 用到 FP8 吗                                  → 910B/C 没有，退 BF16 或 INT8
□ 用到 GPU 特有 API 吗（__shfl_sync、warp 原语）→ 昇腾没有 warp（12 章 §12.3.2），要重写
□ 前向输出无 NaN/Inf                           → assert torch.isfinite(out).all()
□ 训练侧连跑 50 步不崩
```

**第 2 关 · 精度** → 详见 §14.4

```
□ 两侧用【同一份】输入（存成 .pt，不要各自随机生成）
□ msprobe 两侧对称采集（task / level / step 一致）
□ 先 L0 模块级扫全网，再对可疑模块 L1 细看
□ 按前向顺序找【第一个】不达标的条目，并能解释它
□ atten_mask 取反了吗                          → §14.4.6 坑 1（有免工具自检）
□ sparse_mode 按 flash-attn 版本选对了吗       → §14.4.6 坑 2
□ 量化模型：在昇腾上重新校准了吗               → §14.4.6 坑 5
□ 端到端：固定输入贪心解码，和 GPU 对前 N 个 token
```

**第 3 关 · 性能** → 详见 §14.5

```
□ 先答"瓶颈在 Host 还是 Device"                → §14.5.2，有个不用工具的土办法
□ Host bound → 开 ACLGraph 或 TorchAir
□ Device bound → 看指令流水图定四类 bound      → §14.5.3 对应表
□ 变长场景用 TND 布局了吗                      → §14.5.4，padding 浪费是平方级的
□ 有没有隐式的 ND ↔ NZ 转换                    → §14.5.4，看 TransData 占比
□ 用的是融合算子还是一堆小算子拼的             → 回到 §14.2
□ MoE：EP 规模重扫了吗、通算融合启用了吗（≥64 卡）
```

**第 4 关 · 上线**

```
□ 峰值显存 ≤ 单卡容量 × 0.9                    → §14.3.3 的账
□ 确认子型号（910B4 只有 32 GB，差距是数量级的）
□ 910C 双 die 的 NUMA 效应算进切分了吗
□ 24 小时长稳：不 OOM、不掉卡、吞吐不衰减
□ 多机：HCCL 组网、掉卡恢复 / 断点续训验过了吗
```

---

## 14.4 第 2 关 · 精度对齐：逐层比对的完整方法，和四个迁移坑

这是全章最难、最贵、最不能跳的一节。它贵在一个特性上：**这一关的失败没有症状**。第 0、1、3、4 关过不去都会以某种形式报出来——报错、慢、OOM；只有第 2 关，模型会安安静静地给你一个错的答案，一直给到上线之后。

### 14.4.1 为什么必须逐层比对：看 loss 会漏掉什么

先用一个具体的坑说明「看 loss」为什么不够。以 `atten_mask` 语义相反为例（[04 章 §4.9.3](04-FlashAttention全系.md) 坑 1）：GPU 侧 PyTorch SDPA 的约定是 **True = 参与计算**，NPU 侧 `npu_fusion_attention` 的 bool 型 `atten_mask` 是 **True = 屏蔽**（✅ 接口语义，见 04 章）。同一个张量，两边读出相反的意思。

把一个 4 token 的因果掩码（causal mask，下三角）代进去，逐格看会发生什么：

```
GPU 侧构造的因果 mask（True = 参与）：      NPU 按 True = 屏蔽 读同一个张量：

        k0  k1  k2  k3                              k0  k1  k2  k3
   q0    T   F   F   F                         q0   屏蔽 参与 参与 参与
   q1    T   T   F   F                         q1   屏蔽 屏蔽 参与 参与
   q2    T   T   T   F                         q2   屏蔽 屏蔽 屏蔽 参与
   q3    T   T   T   T                         q3   屏蔽 屏蔽 屏蔽 屏蔽
        ↑ 每个 token 只看自己和过去            ↑ 每个 token 只看【未来】，
                                                 最后一行【什么都看不到】
```

后果分两种，都不报错：

```
情况 1：因果掩码取反 —— 模型能看到未来（信息泄漏）
  训练 loss 不但会降，还可能【降得比 GPU 更快】—— 因为它在偷看答案
  但一做自回归生成就崩：推理时没有"未来"可看
  ⚠️ 这条是机理推导，具体表现随实现而异，但"不报错"是确定的

情况 2：padding 掩码取反 —— 真实 token 被屏蔽、padding 参与计算
  loss 能降但降不到位
  ★ 而且它有一个盲区：batch 内序列【等长】时没有 padding，
    这个 bug 完全不显形 —— 这正是"短序列小 batch 上看着正常"的原因
```

> ★ **这就是 loss 曲线的根本问题：它是一个标量，是几十亿次计算聚合出来的一个数。** 它能告诉你「有东西不对」，但既不能告诉你「哪里不对」，也无法排除「看起来对，其实不对」。第 2 关要的是逐层的向量级比对，不是一条曲线。

> ★ **一个不需要 GPU、不需要任何工具的自检**：因果注意力里，**第 0 个 token 只能看到自己**，所以 softmax 是对单个元素做的，结果恒等于 1，于是 `out[0] == V[0]`，逐元素相等（BF16 下允许 1e-2 量级的误差）。这条性质对任何正确的 causal attention 实现都成立。**如果你在 NPU 上跑出来 `out[0] != V[0]`，mask 一定错了**，连 GPU 基准都不用准备。建议把它写成一个单元测试，放在迁移的第一天。

### 14.4.2 逐层比对的完整流程

```
        GPU 侧（基准 / bench）              NPU 侧（被测）
              │                                  │
              └────────── 同一份输入 ────────────┘
                 （存成 .pt 文件，两边都 load 它，
                   ★ 绝对不要两边各自随机生成）
              │                                  │
        msprobe PrecisionDebugger          msprobe PrecisionDebugger
        config: task / level / step        ★ 配置必须【完全一致】，
                                              否则 compare 对不上名字
              │                                  │
              ▼                                  ▼
         bench_dump/                        npu_dump/
              └────────────┬─────────────────────┘
                           ▼
          msprobe -f pytorch compare -i ./compare.json -o ./out
                           ▼
              compare_result_<时间戳>.xlsx   （全部条目 + 五个指标）
              advisor_<时间戳>.txt           （工具给的可疑点提示）
                           ▼
          按【前向顺序】找第一个不达标的条目  ← §14.4.5
                           ▼
         ┌─────────────────┴──────────────────┐
         ▼                                    ▼
   cos 仍 ≈ 1，误差随层数缓慢累积        cos 在某一层【断崖下跌】
         │                                    │
    这是浮点噪声                         这是语义/实现不同
    记进基线，放行                       ★ 就是它，继续往下钻
                                              ▼
                              在这一层上收紧：L0 模块级 → L1 算子级
                                              ▼
                              把这一层的输入 dump 成 .pt，
                              写 20 行脚本单独调这个算子对比
                                              ▼
                                        定位到具体参数
```

### 14.4.3 怎么 dump：msprobe 的最小可用配置

工具是 **msprobe**（MindStudio Probe），属于 mstt 工具链（✅ 官方，开源）。

```bash
pip install mindstudio-probe          # ✅ 官方包名
```

配置文件 `config.json`（✅ 结构取自华为云 ModelArts「GPU 业务迁移至昇腾」官方最佳实践）：

```json
{
  "task": "statistics",
  "dump_path": "/xxx/msprobe_dump",
  "rank": [],
  "step": [0],
  "level": "L1",
  "seed": 1234,
  "is_deterministic": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

训练脚本里插桩（✅ 官方写法）：

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all()                                              # 固定随机性，★ 两侧都要调
debugger = PrecisionDebugger(config_path='./config.json')

for step, batch in enumerate(loader):
    debugger.start()
    loss = model(batch).loss
    loss.backward()
    debugger.stop()
    debugger.step()        # ⚠️ 官方提示：必须放在反向【之后】，早了会丢反向数据
```

`task` 和 `level` 这两个字段决定了这次 dump 的粒度和体积，先搞清楚（✅ 官方口径）：

| 字段 | 取值 | 采什么 | 数据量（✅ 官方口径） |
|---|---|---|---|
| `task` | `statistics` | 只存 max / min / mean / 方差等统计量 | 几 MB 到几十 MB |
| `task` | `tensor` | 存真实张量 | GB 级，官方说明适合单卡小规模 |
| `level` | `L0` | 模块（`nn.Module`）级，另出 `construct.json` | 小 |
| `level` | `L1` | API（算子）级 | 中 |
| `level` | `mix` | 两者都要（做图可视化比对时必需） | 大 |

> ★ **实践顺序：先粗后细。** 第一遍用 `task=statistics` + `level=L0` 扫全网，目标是定位到「哪一个模块」；找到之后，再对这个模块用 `task=tensor` + `level=L1` 取真实张量细看。**一上来就 `tensor` 全网 dump 是新手第一个会踩的坑**——一个 70B 模型单步的全部张量能把盘塞满，而且 compare 会跑到天亮。

两侧采完之后比对（✅ 官方命令）：

```bash
# compare.json（单卡）
# {
#   "npu_path":   "./npu_dump/dump.json",
#   "bench_path": "./bench_dump/dump.json",
#   "stack_path": "./npu_dump/stack.json",
#   "is_print_compare_log": true
# }
# 多卡时三个 path 改成指向 step 目录，如 "./npu_dump/step0"

msprobe -f pytorch compare -i ./compare.json -o ./output -s
```

产出两个文件（✅ 官方）：`compare_result_<时间戳>.xlsx`（全部条目和指标）和 `advisor_<时间戳>.txt`（工具给出的可疑 API 提示）。

> ⚠️ **msprobe 在 Megatron / MindSpeed / ModelLink 这类加速库里有一个高频报错**：msprobe 通过包装（wrap）torch 的 API 来插桩，会改变这些 API 的类型和地址；而加速库里有些 API 在工具生效前类型就已固定，加速库自己的类型检查会因此报错。官方给的两个办法：把 `PrecisionDebugger` 的实例化放在文件最开头（紧跟 import），确保所有 API 都被包装到；或者编辑 msprobe 的 `support_wrap_ops.yaml`，把冲突的算子（社区反馈常见的是 `gelu` / `silu`）注释掉让工具跳过。

> **还有一个可视化的比对入口**：`msprobe graph_visualize -tp <被测> -gp <基准> -o <输出>`，产出 `.vis.db` 用 `tensorboard --logdir <输出>` 打开，能在图上高亮差异大的节点（✅ vLLM-Ascend 官方文档口径）。它要求 `level` 是 `L0` 或 `mix`，因为需要 `construct.json`。层数多的模型上，这个比翻 Excel 快。

### 14.4.4 比什么指标、阈值取多少

`msprobe compare` 输出五个指标（✅ 官方列出）：

| 指标 | 含义 | 它对什么敏感 |
|---|---|---|
| **Cosine**（余弦相似度） | 两个张量拉平成向量后的夹角余弦 | **方向**。实现逻辑、算法语义不同会让它掉 |
| **MaxAbsErr**（最大绝对误差） | `max｜a−b｜` | 单个离群点 |
| **MaxRelativeErr**（最大相对误差） | 相对量级的最大偏差 | 小数值上的偏差会被放大 |
| **One Thousandth Err Ratio**（双千分之一） | 相对误差 ≤ 1e-3 的元素**占比** | 误差是**普遍**的还是**个别**的 |
| **Five Thousandths Err Ratio**（双千分之五） | 同上，阈值放宽到 5e-3 | 低精度（BF16/FP16）场景的兜底 |

五个指标不用平均使力。**最该先看的是 Cosine 和 双千分之一占比这一对**，因为它们组合起来能区分三类完全不同的问题：

| Cosine | 双千分之一占比 | 判断 | 对应 11 章的分类 |
|---|---|---|---|
| ≈ 1 | 高（≈1） | 正常的浮点噪声，随层数缓慢累积 | — |
| ≈ 1 | **明显偏低** | 普遍性的小偏差：精度格式或累加顺序不同 | A 类（定义失配）/ C 类（硬件非确定性） |
| **明显 < 1** | 任意 | **方向都变了，这不是噪声** | B 类（计算图重排）或语义错误 ★ 优先查 |
| ≈ 1 | 高 但 MaxAbsErr 很大 | 个别离群点，看是不是集中在某些位置（padding？尾块？） | 查 §14.4.6 坑 3 / 坑 6 |

阈值（⚠️ **官方文档只说「工具按阈值过滤」，没有给出数值门槛**；下面是工具源码里的常量口径 + 经验值，只能当参考）：

```
Cosine          ≥ 0.9999     ⚠️
MaxAbsErr       < 0.001      ⚠️  注意这是【绝对值】，张量本身量级大时要放宽
双千分之一占比   ≥ 0.99       ⚠️  经验值
```

> ⚠️ **别把阈值当判据用死。** 绝对误差 0.001 对一个取值范围在 [−1, 1] 的 RMSNorm 输出是严重问题；对一个取值上千的 logits 是纯噪声。**正确的做法是：先看工具在 `Result` 和 `Err_Message` 两列给的结论，再结合这一层输出的实际量级判断。** 一个更稳的习惯是先跑一遍「GPU vs GPU（同一份代码跑两遍）」拿到噪声基线，知道正常情况下这些指标长什么样，再拿 NPU 的结果去对。

### 14.4.5 发现不一致之后：怎么二分定位

打开 `compare_result.xlsx` 看到一片红是常态。关键是知道**只有一行有信息量**。

```
第一步：按【前向顺序】排序，找第一个不达标的条目
        ★ 只看第一个。它后面的条目几乎必然也不达标 —— 误差会往下传。
          新手最常见的动作是看到满屏红色就慌，挨个查，
          其实第 2 行往后的红色 90% 是被第 1 行带坏的，没有独立信息。

第二步：在这一层内部再二分
        假设 compare 报第一个坏掉的是 layer 12 的 attention 输出。
        这一层内部依次是：QKV Linear → RoPE → FlashAttention → O Linear
        在这四个点上分别看指标：

            QKV Linear      cos = 1.0000   ✓
            RoPE            cos = 1.0000   ✓
            FlashAttention  cos = 0.9813   ✗  ← 就是它
            O Linear        cos = 0.9752   ✗  （被上一步带坏的，不用管）

        ★ 这一步就是普通的二分查找，只是查找空间是"前向顺序"。
          40 层的模型，二分 6 次就能定位到层；层内 4 个算子，再 2 次定位到算子。
          总共 8 次 compare，这是有工具的情况下最坏的代价。

第三步：脱离模型，做最小复现
        把这一层的全部输入（q / k / v / atten_mask / 各种标志位）
        从 dump 里取出来存成 .pt，然后写一个 20 行的脚本：
            输入 load 进来 → 分别调 npu_fusion_attention 和 GPU 侧参考实现
            → 打印 cos 和 MaxAbsErr
        在这个脚本上试参数：sparse_mode 取 2 还是 3、mask 要不要取反、
        layout 是 BNSD 还是 TND、scale 传对没有。

        ★ 为什么一定要做第三步：在一个 20 行脚本上试一组参数要 3 秒，
          在一个 70B 模型上试一组参数要 20 分钟。
          迭代速度差两到三个数量级 —— 这一步省下的时间，
          通常比前两步加起来还多。
```

> ★ **第三步的产物是可以留下来的资产**：这个 20 行脚本改一改就是一个算子级单元测试。迁移完成之后，它能防止下一次升级 CANN 时同样的坑再来一遍。

### 14.4.6 六个坑，逐个给复现方法——先说清哪几个是静默的

迁移资料里常把下面这些坑一股脑装进「静默错误」一张表，但它们的**性质其实不一样**——有的不报错、有的会报错、有的只是慢。混在一起会误导排查优先级，所以先分类：

```
真正的静默【精度】错误（不报错，结果不对）  ← 最贵，必须靠逐层比对才能发现
  坑 1  atten_mask 语义相反
  坑 2  sparse_mode 选错
  坑 5  量化产物直接从 GPU 搬过来
  坑 6  尾块处理漏了（自己写 Ascend C 算子时）

会报错的（其实是好事，早发现早修）
  坑 3  定长 kernel 要求序列长度是 128 的倍数

静默【性能】损失（结果是对的，只是慢）
  坑 4  head_dim 对齐要求比 GPU 严，落不到快路径
```

---

**坑 1：`atten_mask` 语义相反**（[04 §4.9.3](04-FlashAttention全系.md)，静默精度错误）

机理和后果见 §14.4.1。修法：

```python
# GPU（PyTorch SDPA）：attn_mask 为 True 表示「参与计算」
out_gpu = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

# NPU：bool 型 atten_mask 为 True 表示「屏蔽掉」—— 必须取反
atten_mask_npu = torch.logical_not(mask)
out_npu = torch_npu.npu_fusion_attention(
    q, k, v, head_num,
    input_layout="BNSD",
    atten_mask=atten_mask_npu,          # ← 取反后的
    scale=1.0 / math.sqrt(head_dim),
    keep_prob=1.0,
)[0]                                     # ← 返回元组，第 0 个才是 attention 输出
```

**最小复现（两个，都很便宜）**：

```
复现 A（不需要 GPU）：
  构造 q = k = v = 随机张量，用标准 causal mask 跑一次
  断言 out[0] ≈ V[0]   ← §14.4.1 那条数学性质
  不成立 → mask 错了

复现 B（需要 GPU 基准）：
  用一个【有 padding 的变长 batch】（比如长度 [4, 7, 2] padding 到 8），
  两侧跑同一份输入，比对输出
  ★ 关键：batch 内序列【必须不等长】。等长 batch 是这个 bug 的盲区，
    用等长数据测一万遍也测不出来。
```

---

**坑 2：`sparse_mode` 要按 flash-attn 版本选**（[04 §4.9.3](04-FlashAttention全系.md)，静默精度错误）

```
替换 flash-attn ≤ 2.0 的代码 → sparse_mode = 2
替换 flash-attn ≥ 2.1 的代码 → sparse_mode = 3
```

原因是 flash-attn 2.1 改过因果掩码在**非方阵**（S_q ≠ S_kv）时的对齐方式：三角形到底贴左上角还是右下角。用一个 S_q = 2、S_kv = 4 的小例子把两种对齐画出来（这正是 chunked prefill 的典型形状：已经有 2 个 kv，这一轮再来 2 个 q）：

```
左上角对齐（老语义，sparse_mode = 2）：       右下角对齐（新语义，sparse_mode = 3）：
   q_i 能看到 k_j 当 j ≤ i                      q_i 能看到 k_j 当 j ≤ i + (S_kv − S_q)

        k0  k1  k2  k3                               k0  k1  k2  k3
   q0    ✓   ×   ×   ×                          q0    ✓   ✓   ✓   ×
   q1    ✓   ✓   ×   ×                          q1    ✓   ✓   ✓   ✓

   两者相差 (S_kv − S_q) = 2 列。
   ★ S_q = S_kv 时这个差值是 0，两张图【完全重合】——
     这就是"单元测试只测方阵就永远发现不了"的数学原因。
```

**最小复现**：

```
写一个参数化的测试，形状只取 S_q ≠ S_kv 的组合，例如
    (S_q, S_kv) = (1, 8), (2, 4), (128, 512)
对每组分别跑 sparse_mode = 2 和 3，和 GPU 参考实现比 cos。
★ 正确的那个 mode 会给出 cos ≈ 1，错的那个会明显偏离 —— 一次就能选对。
★ 顺带把 (S_q, S_kv) = (4, 4) 这种方阵也测一遍，
  你会看到两个 mode 都通过 —— 亲眼看一次，这个坑就再也不会踩了。
```

---

**坑 3：定长 kernel 要求序列长度是 128 的倍数**（[04 §4.9.3](04-FlashAttention全系.md) 坑 3，**会报错**）

SGLang 社区踩过的真实问题：某些定长 Attention kernel 要求序列长度是 128 的倍数，而 Decode 请求的 `q_len = 1`，于是 [09 章 §9.8](09-推理侧算子.md) 讲的 chunked prefill + decode 混合批直接触发 kernel 错误。

**最小复现**：拿 `S = 1`、`S = 127`、`S = 129` 各跑一次定长 kernel，看报不报错。**它会报错，这是好消息**——它属于第 1 关（跑通）就会暴露的问题，不会拖到线上。

解法是换变长布局，这同时也是 §14.5.4 的性能要求：

```
BNSD 布局：[Batch, Num_heads, Seq, Dim]     定长，要 padding 到 128 的倍数
TND  布局：[Total_tokens, Num_heads, Dim]   变长，不 padding
           配 actual_seq_qlen / actual_seq_kvlen 传每条序列的真实长度
```

---

**坑 4：head_dim 对齐要求比 GPU 严**（[04 §4.9.3](04-FlashAttention全系.md) 坑 4，**静默性能损失**）

[12 章 §12.3.3](12-昇腾硬件基础.md) 讲的 FRACTAL_NZ 分形格式对维度有硬要求，非标准 `head_dim` 可能落不到快路径。**注意它和坑 1/2 不是一回事：结果是对的，只是慢。**

**最小复现（这是一次性能扫描，不是精度测试）**：

```
固定 batch / seq / num_heads，只扫 head_dim：
    head_dim ∈ {64, 80, 96, 112, 128, 160, 256}
每个跑 100 次取中位耗时，画一条曲线。
★ 要找的特征：耗时【不是】随 head_dim 单调上升的平滑曲线，
  而是在 16 或 128 的倍数上有明显的台阶 —— 台阶下面是快路径，上面是慢路径。
  这和 01 章 §1.7.2 讲的"Tensor Core 对不齐就静默退回慢路径"是同一类现象，
  只是昇腾这边的代价更大（不是退化，是要触发额外的补齐搬运）。
```

---

**坑 5：量化产物直接从 GPU 搬过来**（[06 §6.10.1](06-量化算子.md)，静默精度错误）

GPU 上用 AutoGPTQ / llm-compressor 量化出来的权重和 scale，不能直接拿到昇腾上用。必须用 msModelSlim 在昇腾侧**重新校准**。

**最小复现**：在一个固定测试集上分别算 BF16 基线、GPU 量化产物直接搬、昇腾重新校准三者的困惑度（PPL）。前者和后者的差距通常远大于量化本身应有的损失。⚠️ 具体差多少和模型强相关，本章不给数字。

> ⚠️ 顺带一个 CANN 版本相关的坑（✅ 见 06 章 §6.10.1）：**8.0.RC3 及之前 msModelSlim 内置在 CANN 包里，之后的版本要单独装开源版。** 用新 CANN 却按老文档去 import 内置模块，会找不到。

---

**坑 6：尾块处理漏了**（[13 §13.5.3](13-AscendC编程.md)，静默精度错误，只在自己写 Ascend C 算子时出现）

Ascend C **没有 Triton 的 `mask=` 兜底**（[13 章 §13.3](13-AscendC编程.md) 的对照表）。数据长度不是分块参数的整数倍时，张量末尾的少量元素会是脏数据。

**最小复现**：用**非整除**的 shape 专门测。比如算子按 `tileLength = 256` 切分，就拿 `N = 1000`（= 3×256 + 232）、`N = 257`、`N = 1` 各跑一次，**只看输出张量的最后几十个元素**。整除的 shape（`N = 1024`）测一万遍也测不出来。

### 14.4.7 六个坑汇总

| 坑 | 报错吗 | 现象 | 最小复现（关键在"用什么数据测"） | 出处 |
|---|---|---|---|---|
| `atten_mask` 没取反 | **否** | 因果掩码：信息泄漏，loss 反而降得快；padding 掩码：loss 降不到位 | `out[0] == V[0]` 自检；或用**不等长** batch 比对 | [04 §4.9.3](04-FlashAttention全系.md) |
| `sparse_mode` 选错 | **否** | 非方阵时结果错，方阵时完全正常 | 只测 **S_q ≠ S_kv** 的形状 | [04 §4.9.3](04-FlashAttention全系.md) |
| 量化产物直接搬 | **否** | 精度掉但不崩 | 比 BF16 / 搬运 / 重校准三者的 PPL | [06 §6.10.1](06-量化算子.md) |
| 尾块漏处理 | **否** | 张量末尾少量元素是脏数据 | 用**非整除** shape，只看输出末尾 | [13 §13.5.3](13-AscendC编程.md) |
| 序列长非 128 倍数 | **是** | kernel 直接报错 | `S = 1 / 127 / 129` | [04 §4.9.3](04-FlashAttention全系.md) |
| `head_dim` 不对齐 | 否（**性能**） | 结果对，但慢 | 扫 head_dim 画耗时曲线，找台阶 | [04 §4.9.3](04-FlashAttention全系.md) |
| ND/NZ 格式假设错 | 通常**是** | 结果完全错乱 | 检查 `npu_format_cast` 调用 | [12 §12.3.3](12-昇腾硬件基础.md) |

> ★ **四个静默精度错误有一个共同特征：它们都有「盲区数据」。** 等长 batch 测不出坑 1，方阵测不出坑 2，整除 shape 测不出坑 6。所以迁移期的单元测试有一条铁律——**专挑不整齐的数据测**：不等长、非方阵、非整除、带 padding。整齐的数据让所有人都感觉良好，然后在上线那天把账一次性结清。

> ★ **统一的防御手段只有一个：拿 GPU 结果做逐层比对，而不是看 loss 曲线。** 建议在迁移第一天就把 §14.4.2 的比对脚本搭起来，固定一批输入，逐层 dump、逐层比，把正常状态下的误差量级**记录下来作为基线**。[11 章](11-训推一致性算子层.md) 讲的方法论在这里直接适用——只不过那里比的是「训练 vs 推理」，这里比的是「GPU vs NPU」。

---

## 14.5 第 3 关 · 性能：先答「瓶颈在不在 Device 上」

进入这一关的前提是第 2 关已经关闭（§14.1.3）。

### 14.5.1 两步排查顺序

拿到「NPU 上比 GPU 慢」这个结论之后，**不要先去看算子**。按下面的顺序问两个问题：

```
  第 1 问：瓶颈在 Host 还是 Device？
      工具：msprof 采集 → 看 Device 计算时间占端到端的比例
      也有不用工具的土办法，见 §14.5.2
           │
           ├── Device 占比 < 50%   → Host bound（下发受限）
           │                          去 §14.5.2，★ 这时候碰算子是白费力气
           ├── Device 占比 > 80%   → Device bound，进第 2 问
           └── 50% – 80%           → 两边都有。先治 Host，它便宜得多
                                        │
                                        ▼
  第 2 问：Device 上哪一级流水是瓶颈？
      工具：MindStudio Insight 的【指令流水图】（13 章 §13.8 有读图示例）
           │
           ├── MTE2 / MTE3 那一行最高  → MTE Bound（搬运受限）
           ├── Cube 最高               → Cube Bound（理想状态）
           ├── Vector 最高             → Vector Bound
           └── 三行都不高              → 回到第 1 问，其实还是 Host
```

> ★ **这个顺序和 GPU 侧不一样。** [02 章 §2.7](02-屋顶线与性能剖析.md) 的 GPU 工作流是先用屋顶线定 Compute / Memory Bound，因为 GPU 上 kernel 启动是流水化的、下发很少成为瓶颈（[01 章 §1.8.2](01-GPU硬件基础.md) 算过这笔账）。昇腾上单算子下发路径更长，**必须先确认瓶颈是不是压根就不在 Device 上**（[13 章 §13.8](13-AscendC编程.md) 的经验判断）。把这一步跳过去，最典型的结果是花两周把一个算子从 88% 优化到 92%，而端到端一点没变——因为那 88% 的时间里，Device 本来就在等 Host 发命令。

### 14.5.2 第 1 问：Host bound 的判定，和三种执行模式

**怎么判定 Host bound**，三个办法，按成本从低到高：

```
① 最土也最快的办法（两分钟，不需要任何工具）：
     把 batch size 开大一倍，测端到端耗时。
     耗时几乎不变  → 时间不花在"算"上 → Host bound，去开图模式
     耗时接近翻倍  → Device bound，去读流水图
   ★ 建议先做这一条。它不需要装工具、不需要解析 profiling，
     而且结论和用 msprof 测出来的方向基本一致。

② npu-smi info -t usages -i 0
     AICore 利用率低，但端到端很慢 → 大概率 Host bound

③ msprof 采集后看时间线
     Device 侧算子之间有明显空隙（"锯齿状"），Host 侧却是满的 → Host bound
```

**为什么 Decode 阶段特别容易 Host bound**：一个 70B 模型，一层十几个算子，80 层就是上千个算子；Decode 每步只算一个 token，**每个算子的 Device 计算时间只有几微秒，而下发一个算子的开销是固定的**。算子多而小，正是下发开销占比最高的形状。这和 [09 章 §9.9](09-推理侧算子.md) 讲的 GPU 侧下发开销是同一件事，只是昇腾上更突出。

⚠️ 以下三种模式的对比来自社区经验与工具文档，官方未给出统一的横向对比表：

```
① 单算子模式（Eager）
   torch_npu 逐个下发 aclnn 算子
   ✔ 调试友好、灵活、能单步能打印
   ✘ Host 下发开销大，小算子场景必然 host bound

② ACLGraph  ← 对应 CUDA Graph（05 章 §5.5）
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
| 开发调试、正在做第 2 关 | 单算子模式 | 能单步、能打印，msprobe 插桩也最稳 |
| **Decode 阶段、小 shape、host bound** | **ACLGraph** | 收益明显且接入简单，是性价比最高的一步 |
| 训练 / Prefill，shape 稳定 | TorchAir GE 图模式 | 要极致融合，能接受编译耗时 |

怎么开（⚠️ **接口随 torch_npu / torchair / vllm-ascend 版本变动明显，以你装的版本文档为准**）：

```python
# TorchAir GE 图模式
import torch, torch_npu, torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
model = torch.compile(model, backend=npu_backend, dynamic=False)
#                                                 ↑ 动态 shape 要单独配分档，
#                                                   不配就会反复重编译（见 §14.5.3 的现象表）
```

在 vLLM-Ascend 里，ACLGraph 的开关就是「**不加 `--enforce-eager`**」——加了这个参数就退回单算子模式（⚠️ 以你用的 vllm-ascend 版本为准）。做第 2 关精度排查时通常要加它，做完记得摘掉。

> **一个额外收益**：TorchAir 提供图导出能力，配合 §14.4.3 的 `msprobe graph_visualize` 可以在图上做节点级比对。这是 §14.4 逐层比对的补充工具，也是排查 [11 章](11-训推一致性算子层.md) 训推一致性问题的利器。

### 14.5.3 第 2 问：Device 上四类瓶颈——看哪张图、找什么特征、改什么

[12 章 §12.6](12-昇腾硬件基础.md) 给出了昇腾特有的四类瓶颈（比 GPU 多一类 **MTE Bound**，因为搬运由独立的 MTE 流水线执行），[13 章 §13.8](13-AscendC编程.md) 给了读图示例。这里把它整理成迁移时能直接对着查的形式：

| 瓶颈 | 在哪张图上看 | 找什么特征 | 改什么 |
|---|---|---|---|
| **下发 Bound** | msprof 时间线 | Device 行有空隙（锯齿状），三级流水都不满，但总耗时长 | 开 ACLGraph → 不够再上 TorchAir（§14.5.2） |
| **MTE Bound** | Insight 指令流水图 | **MTE2 或 MTE3** 那一行占比最高 | ① 确认双缓冲 `BUFFER_NUM=2` 真的开了 ② 调大 `tileLength` ③ 查 32 Byte 对齐 ④ 查 ND↔NZ 隐式转换（§14.5.4） |
| **Vector Bound** | 同上 | **Vector（AIV）** 行高，Cube（AIC）空转 | CV 流水并行；或把部分计算挪给 Cube（[13 §13.6](13-AscendC编程.md) 陷阱 3） |
| **Cube Bound** | 同上 | **Cube** 行占比最高 | 恭喜，这是矩阵算子的理想状态。剩余收益在换更大 tiling |

> ⚠️ **MTE Bound 是 GPU 侧没有的一类，最容易被误诊成 Memory Bound。** 区别在于：Memory Bound 是 HBM 带宽跑满了；MTE Bound 是**HBM 带宽没跑满、Cube 也没跑满，但 MTE2 流水占满了**——数据堵在 GM → L1 → L0 的路上（[12 章 §12.6](12-昇腾硬件基础.md)）。用 GPU 的思路去「优化访存」是治不了它的，要治的是搬运的**次数和粒度**。

迁移时最常遇到的几个现象，和它们对应的瓶颈：

| 迁移时观察到的现象 | 最可能的瓶颈 | 先查哪一条 |
|---|---|---|
| Decode 吞吐远低于 GPU，Prefill 还行 | 下发 Bound | 先做 §14.5.2 的「土办法」，然后开 ACLGraph |
| Prefill 慢，序列越长越慢 | MTE Bound，或根本没用融合算子 | 确认 Attention 走的是 `npu_fusion_attention` 而不是一堆小算子拼的（§14.2） |
| LayerNorm / Softmax / RMSNorm 占比异常高 | Vector Bound | 这类算子在昇腾上**根本碰不到 Cube**（[13 §13.6](13-AscendC编程.md) 陷阱 2），「峰值算力」对它们没有意义。考虑和前后的 GEMM 做 CV 流水并行 |
| 换个 batch 或 seq，性能忽好忽坏 | 图模式在反复重编译 | 动态 shape 分档没配（§14.5.2） |
| 算子列表里出现大量你没写过的算子 | ND ↔ NZ 隐式转换 | §14.5.4 ① |
| 变长 batch 下吞吐远低于预期 | padding 浪费 | §14.5.4 ② |

### 14.5.4 两个昇腾特有的性能陷阱

**陷阱 ①：ND ↔ NZ 反复横跳**

```
现象：msprof 解析出来的算子列表里，出现大量 TransData / Cast 之类
      你在代码里从来没写过的算子，加起来占了可观的耗时

原因：12 章 §12.3.3 —— 昇腾片上矩阵用 FRACTAL_NZ 分形格式，
      而 PyTorch 侧张量是 ND（普通行优先）。
      某些算子吃 NZ、某些吃 ND，一条链上交替出现就会反复转换。

怎么确认：把 msprof 解析出的算子表按总耗时排序，看 TransData 的占比。
          ★ 这是一个纯查表动作，两分钟。

怎么改，两个方向：
  a) 全程走 ND，不让私有格式扩散：
         torch_npu.npu.config.allow_internal_format = False
     ⚠️ 这个开关的默认值随 torch_npu 版本变过，用之前先 print 一下当前值；
        接口位置也可能随版本变，以你装的版本文档为准。
  b) 反过来：对确实能从 NZ 受益、且【生命周期长】的张量
     （权重、KV Cache 这类只转一次就长期复用的），
     用 npu_format_cast 显式转一次，之后不再转。

★ 原则：要么全程 ND，要么在一个明确的边界上转【一次】。
  最差的情况是每一层都转一次 —— 转换本身是有成本的，
  而这个成本在 PyTorch 代码里完全看不出来（12 章 §12.3.3 称之为
  "昇腾上一类典型的隐性性能损失"）。
```

**陷阱 ②：padding 到 128 的浪费是平方级的**

坑 3（§14.4.6）说的 128 对齐要求，除了会报错，还有一个性能面。**注意力的计算量正比于序列长度的平方**（[04 章 §4.8.1](04-FlashAttention全系.md)），所以 padding 的代价不是线性的：

```
一个变长 batch，序列长度 = [10, 37, 512, 8]

定长布局（BNSD），全部 padding 到 512（且已是 128 的倍数）：
    注意力计算量 ∝ 4 × 512²  = 4 × 262144 = 1,048,576

真实需要的计算量：
    10² + 37² + 512² + 8²  = 100 + 1369 + 262144 + 64 = 263,677

    浪费倍数 = 1,048,576 / 263,677 ≈ 4.0 倍
             ↑ 注意 token 数只浪费了 2048/567 ≈ 3.6 倍，
               但【计算量】浪费了 4.0 倍 —— 平方放大的效果

解法：TND 布局 + actual_seq_qlen / actual_seq_kvlen（04 章 §4.9.3 坑 3）
     TND 就是 04 章 §4.8.1 那个 cu_seqlens 机制的昇腾版本
```

> ★ 这不是一个「顺手优化一下」的事。做连续批处理（[09 章 §9.5](09-推理侧算子.md)）时，batch 内的长度分布往往比上面这个例子更悬殊（一条 4000 token 的长 prompt 和一堆 `q_len=1` 的 decode 请求混在一起），浪费倍数能到两位数。**变长场景在昇腾上必须用 TND，这是硬要求，不是优化项。**

---

## 14.6 映射总表

把散在各章的对应关系汇总到一处，迁移时对着查。

### 14.6.1 软件栈

| NVIDIA | 昇腾 | 说明 |
|---|---|---|
| CUDA（全家桶） | CANN | 软件栈总称 |
| CUDA Runtime API | AscendCL（`acl*`） | 运行时 |
| CUDA C++ | Ascend C | kernel 编程语言（[13 章](13-AscendC编程.md)） |
| Triton | **triton-ascend**（⚠️ 见下） | 曾是昇腾最明显的生态缺口，现已有开源后端 |
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
| **（无直接对应）** | **msprobe** | GPU/NPU 逐层精度比对（§14.4）——这一项昇腾侧反而更成体系 |
| **（无直接对应）** | **msFmkTransplt** | 脚本自动迁移 + 算子支持度分析（§14.3.2） |

> ⚠️ **Triton 那一行是近期变化最大的**。`triton-lang/triton-ascend`（面向昇腾 NPU 的 Triton 语言与编译器）已开源，项目文档口径是「支持 85% 以上的 Triton Python API」（⚠️ 该比例出自项目自述，未见独立复现），覆盖 MatMul / FlashAttention / LayerNorm 等核心算子。但成熟度仍在快速变动，**昇腾上的生产算子目前仍以 Ascend C 为主**。详见 [07 章](07-Triton编程.md) 章末与 [13 章 §13.9](13-AscendC编程.md)。★ 很多迁移资料（包括本仓库早前的版本）在这一行写的是「无广泛采用的等价物」，**这个说法已经过时**，但也别据此以为可以照搬 GPU 的 Triton 算子——先按 §14.2.2 命令 ① 的思路，在你的环境里实测一遍覆盖度。

### 14.6.2 PyTorch 侧代码改动

| GPU | NPU |
|---|---|
| `import torch` | `import torch; import torch_npu` |
| `.cuda()` / `.to("cuda")` | `.npu()` / `.to("npu")` |
| `"cuda:0"` | `"npu:0"` |
| `torch.cuda.xxx` | `torch_npu.npu.xxx` |
| `backend="nccl"` | `backend="hccl"`（★ 并 grep 代码里硬编码的 `'nccl'` 判断） |
| `F.scaled_dot_product_attention` | `torch_npu.npu_fusion_attention`（★ **mask 要取反**，返回元组取 `[0]`） |
| `flash_attn_func` | `torch_npu.npu_fusion_attention` |
| `flash_attn_varlen_func` | `npu_prompt_flash_attention` 或 TND 布局的 `npu_fusion_attention` |
| PagedAttention decode | `npu_incre_flash_attention` |
| `torch.cuda.CUDAGraph` | ACLGraph |
| `torch.compile` | TorchAir（`torchair.get_npu_backend`） |
| `__shfl_sync` 等 warp 原语 | **无对应物**（昇腾没有 warp，[12 章 §12.3.2](12-昇腾硬件基础.md)），要重写 |

### 14.6.3 命令速查（按关卡组织）

```bash
# ── 第 0 关 环境 ─────────────────────────────────────
npu-smi info                                    # 对应 nvidia-smi
npu-smi info -t usages -i 0                     # AICore/AIV 利用率、HBM 带宽占用
python3 -c "import torch,torch_npu; print(torch_npu.__version__, torch_npu.npu.get_device_name(0))"
find ~ -maxdepth 4 -type d -name "plog"         # CANN 详细日志在哪（报错码看不懂时）

# ── 第 1 关 跑通 ─────────────────────────────────────
# 自动迁移：在 import torch 之后加 from torch_npu.contrib import transfer_to_npu
cd /usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/
./pytorch_gpu2npu.sh -i <原脚本> -o <输出> -v 2.1.0    # 产出 unsupported_op.xlsx

# ── 第 2 关 精度 ─────────────────────────────────────
pip install mindstudio-probe
msprobe -f pytorch compare -i ./compare.json -o ./output -s    # 逐层比对
msprobe graph_visualize -tp <被测> -gp <基准> -o <输出>        # 图上比对（需 level=L0/mix）
tensorboard --logdir <输出>

# ── 第 3 关 性能 ─────────────────────────────────────
msprof op --output=./prof ./my_test             # 对应 ncu，采集
msprof --export=on --output=./prof              # 解析
msprof-analyze advisor all -d ./prof            # 自动诊断 + 专家建议
                                                # （GPU/NPU 性能比对 recipe 也在这）
# 可视化：MindStudio Insight → 指令流水图

# ── 通用：查芯片真实参数（唯一可信来源，12 章 §12.4.3）──
find /usr/local/Ascend -type d -name "platform_config" 2>/dev/null
```

---

## 14.7 诚实的现状评估

把全系列关于昇腾的事实汇总，回答「到底能不能用」。

**已经成熟**

- Attention 融合算子齐备（训练 / Prefill / Decode 三件套 + 统一算子），且**开源可读**（[04 章 §4.9.2](04-FlashAttention全系.md)）。
- HCCL 功能对齐 NCCL，NHR 算法在非 2 次幂集群上有优势，全硬化调度理论上重叠更干净（[08 章 §8.9.1](08-MoE算子与通信.md)）。
- W8A8 量化工具链完整，DeepSeek 等主流模型有现成方案（[06 章 §6.10.1](06-量化算子.md)）。
- vLLM、SGLang 都已有 Ascend backend，MindIE 作为官方推理引擎。
- **超节点在大规模 MoE 场景下是结构性优势**（[08 章 §8.9.3](08-MoE算子与通信.md)）。
- CPU 孪生调试是昇腾相对 CUDA 的一个真实优势（[13 章 §13.7.1](13-AscendC编程.md)）。
- **迁移工具链本身比预期成体系**：`msFmkTransplt` 做脚本迁移和算子支持度分析、`msprobe` 做 GPU/NPU 逐层精度比对、`msprof-analyze advisor` 做自动瓶颈诊断并自带 GPU/NPU 性能比对 recipe。**这一项 NVIDIA 侧反而没有直接对应物**——因为 NVIDIA 不需要「往别处迁」。

**明确的短板**

| 短板 | 影响 | 出处 |
|---|---|---|
| **算子开发效率** | Ascend C 比 Triton 陡峭得多（7 行 vs 70 行）；triton-ascend 已开源但尚未成为生产主流 | [13 章 §13.9](13-AscendC编程.md) |
| **卡间互联带宽** | ⚠️ ~392 GB/s vs NVLink 900 GB/s，单机内 TP 规模受限 | [12 章 §12.5.1](12-昇腾硬件基础.md) |
| **没有 FP8** | 低精度落后一代，HiF8 要等下一代硬件 | [06 章 §6.10](06-量化算子.md) |
| **型号碎片化** | 910B 的 B1/B2/B3/B4 核数带宽差异巨大，算子按 soc_version 编译；32 GB 和 64 GB 的差别能决定方案起不起得来（§14.3.3） | [12 章 §12.4.1](12-昇腾硬件基础.md) |
| **规格不透明** | 无公开完整数据手册，二手数字互相矛盾 | [12 章 §12.4](12-昇腾硬件基础.md) |
| **API 版本漂移快** | 同一个算子在 6.0.RC1 和 7.x 的参数表差异巨大，网上的代码片段极易过时 | §14.2.2 |
| **生态滞后** | 新模型、新算法的 Day-0 支持基本都在 CUDA 上 | — |

**一句话结论**：主流的训练和推理路径已经能跑通且有优化空间；但**任何需要自己写 kernel 的工作，成本比 GPU 高一个档次**（§14.1.4 的案例 B 里，2 个自研 kernel 吃掉了三分之一的人天）。做技术选型时，这个成本要老实算进去，不要只对比硬件标称算力。

---

## 14.8 本章小结

**五道关卡和它们的硬标准**

| 关 | 硬标准（⚠️ 建议门槛） | 主力工具 | 最容易犯的错 |
|---|---|---|---|
| 0 环境 | `npu-smi` 看得到卡，张量能 `.npu()` | `npu-smi info` + plog | 不知道 plog 在哪，对着没有信息的错误码硬猜 |
| 1 跑通 | 前向无 NaN/Inf；训练连跑 50 步不崩 | `transfer_to_npu` / `msFmkTransplt` | 把「跑通」当成「跑对」 |
| **2 精度** | 第一个不达标的条目能被解释 | **msprobe compare** | **跳过这一关先去调性能** |
| 3 性能 | Device 占端到端 > 80%；最高流水 > 70% | msprof + Insight | 不先判 Host/Device 就开始改算子 |
| 4 上线 | 峰值显存 ≤ 容量 × 0.9；24h 长稳 | `npu-smi -t usages` | 不看子型号（B4 只有 32 GB） |

**顺序不能乱的那条铁律**

```
精度是性能的【前置条件】，不是并列项。
  理由：性能优化的每一个动作都会改变数值路径。
        基线数值错了，你没有任何办法判断
        "性能变好的同时精度变没变"。
  代价：§14.1.3 那个例子里，跳过第 2 关净损失 3 周。
```

**逐层比对的五步法**

```
① 两侧同一份输入（.pt 文件），msprobe 配置完全对称
② 先粗后细：task=statistics + level=L0 扫全网
           → 定位到模块后，task=tensor + level=L1 细看
③ 五个指标里先看 Cosine 和 双千分之一占比这一对
     cos ≈ 1，占比高     → 浮点噪声，放行
     cos ≈ 1，占比低     → 精度格式/累加顺序不同
     cos 明显 < 1        → 语义或算法不同 ★ 优先查
④ 按前向顺序找【第一个】不达标的条目 —— 只有它有信息量
⑤ 脱离模型做最小复现：20 行脚本调单算子，迭代快两三个数量级
```

**六个坑，按性质分类**

```
静默【精度】错误（不报错，靠逐层比对才能发现）
  atten_mask 取反   盲区：等长 batch     自检：out[0] == V[0]
  sparse_mode 选错  盲区：方阵           测法：只测 S_q ≠ S_kv
  量化产物直接搬     盲区：小测试集       测法：比三者 PPL
  尾块漏处理         盲区：整除 shape     测法：用非整除 shape，只看末尾

会报错的（好事）      序列长非 128 倍数   —— 第 1 关就暴露
静默【性能】损失      head_dim 不对齐     —— 扫 head_dim 画耗时曲线找台阶

★ 四个静默精度错误的共同点：都有"盲区数据"。
  所以迁移期的单测有一条铁律 —— 专挑不整齐的数据测：
  不等长、非方阵、非整除、带 padding。
```

**性能排查的两步走**

```
第 1 问  瓶颈在 Host 还是 Device？
         土办法：batch 开大一倍，耗时不变 → Host bound
         Host bound → ACLGraph（性价比最高）→ TorchAir（要极致融合）
第 2 问  Device 上哪一级流水满了？看指令流水图
         MTE2/MTE3 高 → MTE Bound（GPU 上没有这一类，别误诊成 Memory Bound）
         Vector 高    → Vector Bound（Norm/Softmax 这类算子碰不到 Cube）
         Cube 高      → 理想状态

两个昇腾特有的性能陷阱
  ND↔NZ 横跳：看算子表里 TransData 的占比，要么全程 ND，要么只转一次
  padding 到 128：注意力计算量正比于 S²，浪费是平方级的，变长必须用 TND
```

**自测三题**（答案在本章对应小节）

1. 因果注意力里，为什么第 0 个 token 的输出 `out[0]` 一定等于 `V[0]`？如果在 NPU 上跑出来不相等，最可能踩了哪个坑？这个自检需要准备 GPU 基准吗？
2. `S_q = 3`、`S_kv = 6` 的因果掩码，左上角对齐和右下角对齐分别允许 `q1` 看到哪几个 `k`？两者相差几列？为什么只用方阵做单元测试永远测不出这个差别？
3. 一个变长 batch，序列长度是 `[64, 200, 1000]`。定长 kernel 要求 padding 到 128 的倍数。算一下 padding 之后的注意力计算量是真实需要的几倍。

#### 参考答案

1. 因果掩码下 `q0` 只能看到 `k0`（它自己），所以 softmax 是对**单个元素**做的，结果恒等于 1，于是 `out[0] = 1 × V[0] = V[0]`，逐元素相等（BF16 下允许 1e-2 量级误差）。不相等说明 mask 让 `q0` 看到了不该看的位置或屏蔽了自己——最可能是 **`atten_mask` 语义取反**（§14.4.6 坑 1）。**不需要 GPU 基准**：这条性质是数学推出来的，对任何正确实现都成立，所以它是迁移第一天就能加上的免费自检。参见 §14.4.1。
2. 左上角对齐的规则是 `j ≤ i`，所以 `q1` 看到 `k0, k1`，共 **2 个**。右下角对齐的规则是 `j ≤ i + (S_kv − S_q) = i + 3`，所以 `q1` 看到 `k0…k4`，共 **5 个**。两者相差 `S_kv − S_q = ` **3 列**。方阵时 `S_kv − S_q = 0`，两种对齐完全重合，所以只测 `S_q = S_kv` 的用例永远发现不了它——这就是 `sparse_mode` 成为静默错误的原因。参见 §14.4.6 坑 2。
3. 最长序列是 1000，向上取整到 128 的倍数 = **1024**。padding 后：`3 × 1024² = 3 × 1,048,576 = 3,145,728`。真实需要：`64² + 200² + 1000² = 4096 + 40000 + 1,000,000 = 1,044,096`。比值 = `3,145,728 ÷ 1,044,096 ≈ ` **3.01 倍**。（对比一下 token 数的浪费只有 `3072 ÷ 1264 ≈ 2.43` 倍——计算量的浪费比 token 的浪费更严重，因为注意力是 `S²` 的。）参见 §14.5.4 陷阱 ②。

> **写在最后**：本系列到这里结束。如果把 14 章压成一句话，那就是——**所有的优化，本质上都是在回答同一个问题：数据搬了几次，每次搬进来被用了多少回。** 从 [01 章](01-GPU硬件基础.md) 的合并访存，到 [03 章](03-GEMM深度剖析.md) 的分块、[04 章](04-FlashAttention全系.md) 的在线 Softmax、[05 章](05-算子融合.md) 的融合，再到 [13 章](13-AscendC编程.md) 的双缓冲和本章的 ND↔NZ 横跳，换的只是硬件和词汇，问题没变过。硬件会换代，本章提到的每一个 API 都可能在两个版本之后改名，但这个问题不会过时。

---

**延伸资料**：
- [Ascend Extension for PyTorch 迁移调优指南](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)（官方，迁移坑的权威说明）
- [华为云 · GPU 业务迁移至昇腾训练推理 最佳实践](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_2525.html)（官方，§14.4 的 msprobe 命令与配置出处）
- [mstt / msprobe 精度调试工具链](https://gitcode.com/Ascend/mstt)（开源，逐层比对与 GPU/NPU 性能比对 recipe 都在这）
- [cann-ops-adv 融合算子库](https://gitee.com/ascend/cann-ops-adv)（开源，§14.2 阶梯第 ②④ 级的入口）
- [triton-ascend](https://github.com/triton-lang/triton-ascend)（昇腾的 Triton 后端，生态变化最快的一块）
- [vLLM Ascend backend](https://github.com/vllm-project/vllm-ascend)（社区，看接入方式、真实调用与遗留问题）
