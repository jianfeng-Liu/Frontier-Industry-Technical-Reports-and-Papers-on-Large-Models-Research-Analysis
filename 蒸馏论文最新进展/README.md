# 蒸馏论文最新进展（论文收集，尚未成文）

> ⚠️ **这个目录还没有笔记，只有论文收集。** 17 篇 PDF 都在本地，但**按 [.gitignore](../.gitignore) 规则不上传**（他人版权内容）。所以在 GitHub 上你只能看到这份清单。

**想看已经写好的蒸馏内容 → [后训练/08-On-Policy 蒸馏（OPD）：五家共同的收口](../后训练/08-蒸馏与合并-OPD.md)**
那一章讲清了 OPD 是什么、为什么 2026 年五家一手报告在流水线最后一步用了几乎同一个数学式子。**这个目录是那一章的延伸阅读池。**

---

## 先说清楚：蒸馏 / OPD 是什么

**知识蒸馏（Knowledge Distillation）** —— 用一个已经很强的**大模型（老师）**去教一个**小模型（学生）**。不是让学生去背标准答案，而是让它模仿老师的"思路分布"：老师认为下一个词有 60% 可能是 A、30% 是 B，学生就学这个 60/30 的倾向，而不是死记"答案是 A"。

**OPD = On-Policy Distillation（在策略蒸馏）** —— 关键区别在于**谁写的句子**：

```
  离线蒸馏（off-policy）：老师先写一堆句子 → 学生照着背
                          ↑ 学生从没见过自己会犯的错

  在策略蒸馏（on-policy）：学生自己写句子 → 老师逐字打分 → 学生改
                          ↑ 纠的是学生【真实会犯】的错
```

一句话理解 OPD 的价值：**把老师当成一个免费的、逐 token 的稠密奖励模型**（详见 [后训练/08](../后训练/08-蒸馏与合并-OPD.md)）。

---

## 论文清单（17 篇，本地 `*.pdf`）

### 综述与失败模式分析

| 论文 |
|---|
| A Survey of On-Policy Distillation for Large Language Models |
| Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes |
| Why Does Self-Distillation (Sometimes) Degrade the Reasoning Capability of LLMs? |

### 自蒸馏（Self-Distillation）

模型自己教自己 —— 没有更强的老师，用自己的某种"更好状态"当老师。

| 论文 |
|---|
| Embarrassingly Simple Self-Distillation Improves Code Generation |
| Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models |
| Reinforcement Learning via Self-Distillation |
| Unifying Group-Relative and Self-Distillation Policy Optimization via Sample Routing |

### 特权信息蒸馏（Privileged Information）

老师能看到学生看不到的东西（比如标准答案、提示），用这个信息差来教。

| 论文 |
|---|
| Privileged Information Distillation for Language Models |
| HDPO: Hybrid Distillation Policy Optimization via Privileged Self-Distillation |
| POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration |

### OPD 的方法改进

| 论文 |
|---|
| On-Policy Delta Distillation |
| Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe |
| Lightning OPD: Efficient Post-Training for Large Reasoning Models with Offline On-Policy Distillation |
| Rubric-based On-policy Distillation |

### 与 RL 结合 / Agentic 方向

| 论文 |
|---|
| Reinforcement-aware Knowledge Distillation for LLM Reasoning |
| Expanding the Capabilities of Reinforcement Learning via Text Feedback |
| The Physics of Multi-Turn Long-Horizon Planning: From Pre-training to Post-training via Single- and Multi-Teacher On-Policy Agentic Distillation |

---

## 怎么获取这些 PDF

按标题在 [arXiv](https://arxiv.org/) 搜索即可。**本仓库不分发论文原文。**

---

> 上一级：[仓库总目录](../README.md)｜已成文的蒸馏内容：[后训练/08](../后训练/08-蒸馏与合并-OPD.md)
