> [← 03 存储格式](03-存储格式.md) ｜ [目录](README.md) ｜ 下一章 → [05 语言识别与启发式过滤](05-语言识别与启发式过滤.md)

# 04 · ⭐ 训练格式：`.bin` + `.idx` 与三个索引

> **这是全教程工程性最强的一章，也是最有实用价值的一章。**
> 训练时 GPU 真正读的就是这两个文件。**读懂这一章，你就能看懂 Megatron-LM / NeMo 的整个数据模块。**

---

## 4.1 先想清楚：训练时的访问模式有多苛刻

```
训练一步要什么？
   "给我第 8,231,904 号样本"      ← 一个随机的下标
   
一步要几条？   B = 512（全局 batch）
一秒几步？     ~ 2 步
几千张卡？     每张卡各要各的

  ⟹  ★ 每秒几万次【完全随机】的读，而且必须比 GPU 算得还快
      否则 GPU 就在等数据（叫 "data starvation"，数据饥饿）
```

而 [03 章](03-存储格式.md)那些格式全都做不到：

```
JSONL     → 想要第 800 万条？从头数 800 万个 \n。❌
Parquet   → 要解压 + 解码 + 反序列化。每条几十微秒。❌
WebDataset→ tar，纯顺序，根本没有"第 n 条"这个概念。❌
```

**所以训练格式必须重新设计。** 设计目标只有一条：

> ★★ **取第 n 条样本 = O(1) 次内存访问，零解析、零解压、零反序列化。**

---

## 4.2 ★ 核心思路：把整个语料变成一个巨大的一维数组

关键的观察是：**tokenize 之后，数据不再是"文本"了，是一串整数。**

而整数是**定长**的。定长的东西，可以直接当数组用。

```
【转换前】三篇文档，长度不一的文本
    doc0 = "番茄炒蛋是家常菜"
    doc1 = "def f(): pass"
    doc2 = "The cat sat"

【tokenize】每篇变成一串整数，末尾加一个 <eod>（end of document，文档结束符）
    doc0 → [4521, 889, 15, 3204, 992,  2]        长度 6
    doc1 → [7712, 331, 88, 21,  2]               长度 5
    doc2 → [1103, 5540, 902, 2]                  长度 4

【落盘】★ 全部首尾相接，拍平成一个巨大的一维数组，存进 .bin
    .bin:  4521 889 15 3204 992 2 7712 331 88 21 2 1103 5540 902 2
           └────── doc0 ──────┘ └──── doc1 ────┘ └─── doc2 ───┘
    下标:   0   1   2   3   4  5   6   7   8  9 10   11   12  13 14

【问题】.bin 里已经没有"文档边界"这个信息了！
        → 所以要有第二个文件 .idx 专门记边界
    .idx:  文档 i 的长度 = [6, 5, 4]
           文档 i 的起始字节偏移 = [0, 12, 22]    （每个 token 2 字节时）
```

> ★★★ **这就是 `.bin` + `.idx` 的全部思想：**
> **`.bin` = 一根拍平的、没有任何结构的整数长条**
> **`.idx` = 告诉你这根长条上哪里到哪里是一篇文档**

**取第 n 篇文档 = 查 `.idx` 拿到 (offset, length) → 在 `.bin` 上切一刀。两次内存访问，结束。**

---

## 4.3 两个文件的具体内容

### `.bin` —— 纯数据，没有任何头部

```
┌──────────────────────────────────────────────────────────┐
│  就是一个 numpy 数组直接 dump 到磁盘。没有魔数、没有头。   │
│  等价于：   np.array(all_tokens, dtype=T).tofile("x.bin") │
└──────────────────────────────────────────────────────────┘

★ dtype T 由【词表大小】决定，Megatron 里是自动选的：
      词表 ≤ 65535        →  uint16   每个 token 2 字节
      词表  > 65535        →  int32    每个 token 4 字节
✅ Megatron 源码里这个函数叫 DType.optimal_dtype(tokenizer.vocab_size)

⚠️ 这个自动选择很重要，算一笔账（我的估算）：
      15 T token × 2 字节 =  30 TB
      15 T token × 4 字节 =  60 TB
   ★ 词表从 65535 涨到 65537，你的数据集存储费直接【翻倍】
   而现在的大模型词表普遍 15 万左右 → 只能用 int32 → 只能认这个 60 TB
```

### `.idx` —— 元数据，很小

```
┌─ 文件头 ────────────────────────────────────────────────┐
│  魔数 magic   b"MMIDIDX\x00\x00"   ← 用来认格式和版本    │
│  版本号                                                  │
│  dtype 编码   （告诉你 .bin 里是 uint16 还是 int32）     │
│  序列条数 S                                              │
│  文档篇数 D                                              │
├─ 数组区 ────────────────────────────────────────────────┤
│  sequence_lengths[S]    每条序列有多少个 token   int32   │
│  sequence_pointers[S]   每条序列在 .bin 里的字节偏移 int64│
│  document_indices[D+1]  文档 ↔ 序列 的分界        int64  │
└─────────────────────────────────────────────────────────┘

⚠️ 上面的字段名与顺序是我依据 Megatron-LM 的 IndexedDataset 实现整理的，
   ✅ 但"两个文件（.bin 存数据 / .idx 存元数据）"与"靠 8 字节魔数识别实现"
      是 Megatron 官方文档明确写的。
   ★ 字节级布局请以 megatron/core/datasets/indexed_dataset.py 源码为准，
     不同版本改过。

体积感：.idx 通常只有 .bin 的 千分之一 量级 ⚠️（我的估算）
        → 所以 .idx 可以整个读进内存，.bin 用 mmap
```

### 路径约定：只给"前缀"

```
你的磁盘上：   /data/zh_web_text_document.bin
              /data/zh_web_text_document.idx

配置里写的：   --data-path /data/zh_web_text_document
                            ↑ ★ 不带 .bin / .idx 后缀
✅ Megatron 官方文档原话：the path should be a basename to which
   both .idx and .bin can be appended.

⚠️ 这是新手最常见的报错之一：写了带后缀的路径 → 找不到文件。
```

---

## 4.4 三个索引：从"一根长条"到"第 n 号训练样本"

`.bin/.idx` 解决的是**"取第 n 篇文档"**。但训练要的不是文档，是**定长 4096 的样本**。

一篇文档可能 300 token，也可能 50000 token。怎么把它们变成整整齐齐的 4096？

✅ Megatron 的 `GPTDataset` 用**三个索引数组**来做这件事。这是本章最需要理解的部分。

### 先看它要满足的四个约束

```
① 样本必须【定长 L】（比如 4096）
② 要能训【E 个 epoch】——数据不够就重复用
③ 每个 epoch 内文档顺序要【随机】，且每个 epoch 的随机顺序【不同】
④ ★★ 必须【完全确定】：给定随机种子，第 n 号样本永远是同一批 token
   （否则断点续训就废了，见 14 章）
```

### 三个索引各管一件事

```
┌─ ① document_index（文档索引） ── 管"顺序"和"重复几轮" ────────┐
│                                                               │
│   Do_idx = [文档编号, 文档编号, ...]                           │
│   长度 = E × 文档总数，E = 让样本数够用的最小 epoch 数          │
│                                                               │
│   例：3 篇文档，要训 2 个 epoch                                 │
│       Do_idx = [ 2, 0, 1,   1, 2, 0 ]                         │
│                 └epoch 0┘  └epoch 1┘                          │
│                 ↑ 每个 epoch 内部各自独立打乱                   │
│   ✅ Megatron 文档：E is the minimum number of epochs such      │
│      that E × |indexed_indices| >= N                           │
└───────────────────────────────────────────────────────────────┘

┌─ ② sample_index（样本索引） ── 管"从哪切到哪" ────────────────┐
│                                                               │
│   把 Do_idx 指定的文档【首尾相接】成一条无限长的流，            │
│   每 L 个 token 切一刀。记下每一刀的位置：                     │
│                                                               │
│   Sa_idx[j] = (在 Do_idx 里的下标 i,  该文档内的字符偏移 off)  │
│                                                               │
│   ★ 一个样本可能【跨越好几篇文档】，也可能【切在一篇文档中间】  │
└───────────────────────────────────────────────────────────────┘

┌─ ③ shuffle_index（打乱索引） ── 管"样本的呈现顺序" ───────────┐
│                                                               │
│   Sh_idx = [0..N-1] 的一个随机排列                             │
│   训练第 j 步要的是第 Sh_idx[j] 号样本                         │
│                                                               │
│   ★ 为什么②之后还要③？                                        │
│     因为②切出来的样本【在语料里是连续的】——                   │
│     样本 100 和样本 101 很可能来自同一篇长文档。                │
│     不打乱的话，一个 batch 里全是同一个主题 → 梯度有偏。       │
└───────────────────────────────────────────────────────────────┘
```

### 一张图串起来

```
  训练第 j 步
       │
       ▼
  ③ Sh_idx[j] = 8231904            "我要第 8231904 号样本"
       │
       ▼
  ② Sa_idx[8231904]   = (i=4021, off=173)
    Sa_idx[8231905]   = (i=4023, off=2044)   ← 下一个样本的起点
       │  ★ 这两条一起，界定了本样本的范围
       ▼
  ① Do_idx[4021] = 88123   Do_idx[4022] = 51    Do_idx[4023] = 7742
       │           └ 文档编号 ┘
       ▼
  .idx → 查这三篇文档各自的 (offset, length)
       │
       ▼
  .bin → mmap 上切三刀，拼接：
         doc88123[173:] + doc51[全部] + doc7742[:2044]
       │
       ▼
     恰好 4096 个 token 的 numpy 数组  ★
```

✅ Megatron 官方文档对这一步的描述与上图一致：
`sample += indexed_dataset[Do_idx[i]][offset:]`，然后是中间的整篇文档，最后 `indexed_dataset[Do_idx[i_next]][:offset_next]`。

### 索引的构建与缓存

```
✅ Megatron 的做法（官方文档原话的意思）：
   ① 三个索引在【一个 rank 上顺序构建】，然后落盘缓存
   ② 其他所有 rank【并行地加载】这个缓存
   ③ 缓存文件名带一个 hash，hash 由 (数据路径、样本数、序列长度、
      随机种子、split 比例 ...) 算出

★ 这个 hash 是个大坑：
   你改了 --seq-length 或 --seed 或数据路径 → hash 变了
   → 索引【全部重建】。TB 级语料上这可能要几十分钟。
⚠️ 而且：如果你【只加了一个新的数据文件】，整个索引也要重建。
```

---

## 4.5 多路数据混合：`BlendedDataset`

真实训练不会只有一个 `.bin`，而是几十个（网页、代码、数学、多语言……），要按配比混（[10 章](10-数据配比.md)讲配比怎么定，这里只讲**怎么实现**）。

```
配置长这样（Megatron 的 --data-path 或 blend 字段）：
    0.70  /data/web_text_document
    0.20  /data/code_text_document
    0.10  /data/math_text_document
    └权重┘ └────── 前缀（不带后缀）──────┘

✅ 官方文档：BlendedDataset is parameterized by the underlying
   MegatronDataset instances D, the weights W, and the size S;
   it draws samples from contributing datasets in proportion to the weights.
```

实现上，它又是一层索引：

```
  BlendedDataset 也预先算好两个数组：
      dataset_index[j]        第 j 个混合样本来自【哪个】数据集
      dataset_sample_index[j] 它是那个数据集里的【第几个】样本

  取第 j 条：
      d = dataset_index[j]                 # 比如 = 1（code）
      k = dataset_sample_index[j]          # 比如 = 55231
      return datasets[d][k]                # 转给 GPTDataset 走 4.4 那套

  ★ 同样是【完全确定】的：给定种子和权重，第 j 条永远来自同一处
```

> ⚠️ **一个很容易踩的坑**：权重是**按 token 数**还是**按样本数**？
> Megatron 的 blend 权重生效在**样本层面**。而不同数据集的**平均文档长度不同**——
> 如果你以为"代码权重 0.2 = 代码占 20% 的 token"，在文档长度差异大时会有偏差。
> ★ **稳妥做法：先统计每个 `.bin` 的实际 token 总数（`.idx` 里就有），再反推权重。**

### 三个配置项，先解释清楚

```
--split 9999,8,2
   把数据按 99.99% / 0.08% / 0.02% 切成 train / valid / test
   ★ 是按【文档】切的连续区间，不是随机抽

--reset-position-ids  /  --reset-attention-mask
   ★ 这两个就是 13 章要讲的【文档掩码】开关：
     开了之后，一条样本里跨文档的位置编码会重置、注意力不能跨文档
     默认【关闭】—— 也就是说默认情况下文档之间是会"串味"的

--eod-mask-loss
   要不要在 <eod> 那个位置上算损失。通常关掉
```

---

## 4.6 完整的伪代码

```python
# ============ 离线：写 .bin / .idx ============
class IndexedDatasetBuilder:
    def __init__(self, prefix, dtype):
        self.bin = open(prefix + ".bin", "wb")
        self.dtype = dtype
        self.lengths, self.pointers = [], []
        self.offset = 0

    def add_document(self, token_ids):
        arr = np.array(token_ids, dtype=self.dtype)
        self.bin.write(arr.tobytes(order="C"))     # ★ 直接追加到长条末尾
        self.lengths.append(len(arr))
        self.pointers.append(self.offset)
        self.offset += arr.nbytes

    def finalize(self, prefix):
        self.bin.close()
        with open(prefix + ".idx", "wb") as f:
            f.write(MAGIC); write_header(f, self.dtype, len(self.lengths))
            np.array(self.lengths,  dtype=np.int32).tofile(f)
            np.array(self.pointers, dtype=np.int64).tofile(f)

# 用法
b = IndexedDatasetBuilder("train", np.int32)
for line in open("clean.jsonl", encoding="utf-8"):
    ids = tokenizer.encode(json.loads(line)["text"]) + [EOD]
    b.add_document(ids)
b.finalize("train")


# ============ 在线：读 ============
class IndexedDataset:
    def __init__(self, prefix):
        self.lengths, self.pointers, dtype = read_idx(prefix + ".idx")
        # ★★ 关键：mmap —— 不把 60 TB 读进内存，只建立映射
        self.bin = np.memmap(prefix + ".bin", dtype=dtype, mode="r")
        self.item_size = np.dtype(dtype).itemsize

    def __getitem__(self, doc_id):                  # ★ O(1)
        start = self.pointers[doc_id] // self.item_size
        return self.bin[start : start + self.lengths[doc_id]]


class GPTDataset:
    def __init__(self, indexed, num_samples, seq_len, seed):
        self.ds = indexed
        self.Do_idx, self.Sa_idx, self.Sh_idx = build_indices(
            indexed, num_samples, seq_len, seed)      # ← 4.4 那三个

    def __getitem__(self, j):
        j = self.Sh_idx[j]                            # ③ 打乱
        i,      off      = self.Sa_idx[j]             # ② 起点
        i_next, off_next = self.Sa_idx[j + 1]         # ② 终点
        if i == i_next:                               # 整条样本在同一篇文档里
            sample = self.ds[self.Do_idx[i]][off:off_next]
        else:
            parts = [self.ds[self.Do_idx[i]][off:]]   # 第一篇的尾巴
            parts += [self.ds[self.Do_idx[k]]         # 中间的整篇
                      for k in range(i + 1, i_next)]
            parts += [self.ds[self.Do_idx[i_next]][:off_next]]  # 最后一篇的头
            sample = np.concatenate(parts)
        return {"tokens": sample}                     # 长度恰好 seq_len(+1)
```

---

## 4.7 ⚠️ 五个真实会踩的坑

```
① 路径带了后缀
   --data-path /d/train.bin  ❌     --data-path /d/train  ✅

② 换 tokenizer 忘了重跑 preprocess
   .bin 里是整数 id，换词表后同一个 id 指向【完全不同的词】
   ★ 现象：loss 从头就是随机水平，且不下降。这个坑非常隐蔽。

③ 索引 hash 失效导致重建
   改 seq_len / seed / 加一个数据文件 → 索引全部重建，TB 级要几十分钟
   ★ 建议：预处理产物和索引缓存都放共享存储，别放本机 /tmp

④ 所有 rank 都要参与构建
   ✅ 官方文档明确警告：all ranks should attempt to build the dataset
      via the builder or the program will hang（不然会挂死）
   ★ 现象：训练启动后卡住不动、没有报错。这是典型的集合通信死锁。

⑤ 混合权重的口径
   blend 权重作用在【样本】层面；你以为的是【token】占比
   → 各数据集平均文档长度差异大时会跑偏。先统计 token 数再反推权重
```

---

## 4.8 挂回真实系统

| 框架 | 训练格式 | 备注 |
|---|---|---|
| **Megatron-LM / Megatron-Core** | ✅ `.bin` + `.idx`（`IndexedDataset`） | 本章讲的就是它，事实标准 |
| **NVIDIA NeMo / Megatron-Bridge** | 同上 | NeMo 是 Megatron-Core 的上层封装 |
| **Megatron-DeepSpeed** | 同上（有 `lazy`/`cached`/`mmap` 三种后端） | ✅ 靠 8 字节魔数识别用的是哪种 |
| **MosaicML LLM Foundry** | MDS | 见 [03 章](03-存储格式.md) |
| **HuggingFace `Trainer`** | Arrow / 在线 tokenize | ⚠️ 小规模够用，大规模会成为瓶颈 |
| **torchtitan** | 在线 tokenize + HF datasets | 偏研究用途 |

> ★ **国内外几乎所有千亿级模型的预训练，数据侧都是 Megatron 这套 `.bin/.idx`**（或它的私有变体）。所以这套东西非常值得吃透。

---

## 4.9 小结

```
① 训练时的访问模式：每秒几万次【完全随机】读，必须快过 GPU
   ⟹ 目标：取第 n 条 = O(1) 内存访问，零解析零解压

② ★★ 核心思想：tokenize 后数据变成【定长整数】→ 可以当数组用
     .bin = 所有文档首尾相接拍平成的一根整数长条（无头无结构）
     .idx = 记录"哪里到哪里是一篇文档"的边界表
     取文档 = 查 .idx 拿 (offset,len) → 在 .bin 上切一刀

③ dtype 由词表大小自动决定：≤65535 用 uint16，否则 int32
   ★ 现在词表普遍 15 万 → 只能 int32 → 15 T token 要 60 TB

④ ★ 三个索引把"长条"变成"第 n 号定长样本"：
     ① document_index  —— 顺序 + 重复几个 epoch（每 epoch 独立打乱）
     ② sample_index    —— 每 L 个 token 切一刀，记 (文档下标, 偏移)
     ③ shuffle_index   —— 再打乱样本的呈现顺序（因为②切出来是连续的）
   全部【确定性】：给定种子，第 n 条永远一样 → 断点续训才成立

⑤ BlendedDataset 是再上一层索引：(来自哪个数据集, 是它的第几条)
   ⚠️ 权重作用在【样本】而非 token 层面，长度差异大时要先统计再反推

⑥ 五个坑：路径带后缀 / 换 tokenizer 没重跑 / 索引 hash 失效重建 /
          不是所有 rank 都 build 会挂死 / 混合权重口径
```

---

> [← 03 存储格式](03-存储格式.md) ｜ [目录](README.md) ｜ 下一章 → [05 语言识别与启发式过滤](05-语言识别与启发式过滤.md)
