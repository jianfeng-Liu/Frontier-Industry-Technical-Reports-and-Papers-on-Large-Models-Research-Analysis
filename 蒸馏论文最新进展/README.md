# 蒸馏论文最新进展

> ⚠️ **17 篇 PDF 在本地，但按 [.gitignore](../.gitignore) 规则不上传**（他人版权内容）。所以在 GitHub 上你看不到 PDF 本身 —— 但**正文笔记已经写好了**，就在下面的 01–05 章里，每个数字都注明了出自哪篇论文的哪一节/哪张表。

**先读基础 → [后训练/08 · On-Policy 蒸馏（OPD）](../后训练/08-蒸馏与合并-OPD.md)**
那一章讲清了 OPD 是什么、为什么 2026 年**四家**一手报告在流水线最后一步用了几乎同一个数学式子（⚠️ 是四家不是五家：MiniMax M3 没有技术报告，那一格是「未披露」而不是「没做」）。**这个目录是那一章的延伸：从五家报告的「工程共识」，走到论文前沿的「还在吵什么」。**

---

## 📖 本目录的五章

| 章 | 讲什么 | 基于几篇 | 一句话 |
|---|---|---|---|
| **[01](01-综述与失败模式.md)** | 综述的分类体系 + ⭐ **OPD 什么时候不 work** | 3 | 「什么时候失败」的材料在中文世界里特别稀缺，这一章写得最足 |
| **[02](02-自蒸馏.md)** | 没有更强的老师，用自己的「更好状态」当老师 | 4 | ⚠️ 和 01 章那篇「自蒸馏会**损害**推理能力」直接打架，两章互相引用、把矛盾摆出来了 |
| **[03](03-特权信息蒸馏.md)** | 老师能看到学生看不到的东西（答案、提示） | 3 | 推理时撤掉特权，学生到底学到了什么 |
| **[04](04-OPD方法改进.md)** | 四篇各自改进 OPD 的哪一环 | 4 | ⚠️ 含 Lightning OPD 的 `Offline` × `on-policy` 矛盾怎么解 |
| **[05](05-与RL结合与Agentic.md)** | 蒸馏接在 RL 的哪一环、长程信用分配 | 3 | ⚠️ Text Feedback **没有替代**标量奖励，是叠加的第二条信道 |
| **[附](附-速查表.md)** | 17 篇总表 + 缩写 + ⭐ **「这个领域在吵什么」** | — | 总表里专门有一列标**阅读深度** |

> ⚠️⚠️ **先看这一条再读正文**：17 篇里只有 **6 篇做了全文通读**（01 的 3 篇 + 05 的 3 篇），02/03/04 的若干篇读了正文主干，**其余只读了第 1 页摘要**。[附-速查表](附-速查表.md) 的总表里**逐篇标注了阅读深度**，凡「⚠️ 仅摘要」的条目只能当**线索**用，不能当结论引。
>
> ★ 这不是谦虚，是这个目录的使用方式：**它帮你决定该亲自去读哪几篇**，不替你读完。

---

## ⚖️ 这个目录和别处的分界

| 想知道什么 | 去哪 |
|---|---|
| OPD 是什么、工业界怎么用 | [后训练/08](../后训练/08-蒸馏与合并-OPD.md) |
| 奖励从哪来、rubric 与 LLM-as-judge | [后训练/06](../后训练/06-奖励从哪来.md) |
| RLVR、难题采不到正样本的死穴 | [后训练/05](../后训练/05-RLVR-可验证奖励.md) |
| agentic RL 的环境为什么是瓶颈 | [后训练/07](../后训练/07-AgenticRL-环境成为瓶颈.md) |
| 这些算法在系统上怎么跑起来 | [强化学习训练框架/](../强化学习训练框架/) |
| **论文前沿、互相矛盾的结论** | ⭐ **本目录** |

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
