# 设计说明：混合检索与重排序

> 面向要改这套检索链的人。用户向的安装与使用见 [README](../README.md)，
> 评估方法与实测数据见 [eval.md](eval.md)。

## 1. 两阶段检索总览

检索分两阶段，开关都在 [config_data.py](../config_data.py)：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `hybrid_search_enabled` | `True` | 阶段一：BM25 + 向量双路粗召回，用 `EnsembleRetriever`（RRF）融合 |
| `rerank_enabled` | `False` | 阶段二：对粗召回结果做精排 |

于是有三条路径，也正好是评估脚本对比的三种模式：

| 模式 | `hybrid_search_enabled` | `rerank_enabled` | 实际走什么 |
| --- | --- | --- | --- |
| baseline | `False` | 任意 | 原渐进式三阶段纯向量检索（**逐字节未改**） |
| +hybrid | `True` | `False` | 双路粗召回（RRF 融合）→ 取前 `rerank_top_n` 篇 |
| +hybrid+rerank | `True` | `True` | 双路粗召回 → 精排 → 取前 `rerank_top_n` 篇 |

分流点在 `rag._retrieve_docs_progressively_with_debug`：开关为假时直接走原有三阶段逻辑，
一行未改；为真时交给 `_retrieve_hybrid_with_debug`。

## 2. 为什么 hybrid 走独立路径，而不是复用原阈值

原渐进式检索靠**相关度分数阈值**（`0.8 / 0.65 / 0.45`）决定取几篇。但：

- `EnsembleRetriever` 融合后输出的是 RRF 排名分（0.0x 量级）；
- 精排分数的量级随模型差异极大——实测同一个明显相关的文档，
  `qwen3.7-text-rerank` 给 **0.908**，`gte-rerank-v2` 只给 **0.360**。

把绝对阈值套到这两种分数上必然全部落空，所以 hybrid 路径**不用阈值**，改成
「粗召回固定篇数 → 精排固定篇数」。`hybrid_search_enabled=False` 时渐进式那套逻辑
一行未改，可以直接当 baseline 用。

两种 hybrid 模式最终都取 `rerank_top_n` 篇，这样它们对比时只差「有没有精排」一个变量。

## 3. BM25 的中文分词（坑）

`BM25Retriever` 默认的 `preprocess_func` 就是 `text.split()`，对中文会把**整句当成一个
token**，BM25 这一路等于失效。实测：

```
"珠海校区哪里有好吃的" -> ['珠海', '校区', '哪里', '有', '好吃', '的']   # jieba
"退货政策是签收后七天内可无理由退货，生鲜类商品除外。" -> [整句一个 token]   # 默认
```

所以换成 jieba（`bm25_tokenizer = "jieba"`）。jieba 没装时自动降级为字符二元组，
模块仍可正常导入。

## 4. BM25 索引与 category 的冲突

`BM25Retriever` 是纯内存索引，**不支持 metadata 过滤**，`EnsembleRetriever` 也不会透传。
两种解法里选了后者：**按 category 分别建索引**（`dict[category, BM25Retriever]`，
`category=None` 用 `__all__` 全量索引）。另一种「检索后再按 metadata 过滤」会破坏
BM25 的 top-k 语义（过滤后可能只剩很少几条）。

索引懒加载 + 双重检查锁（与 `knowledge_base._md5_lock` 同一模式），语料直接从 Chroma
取，保证与向量库同源。检索时复用语料向量、只换 `k`，避免并发下改坏共享对象。

## 5. 重排序后端选择

`rerank_backend` 目前只实现 `"dashscope"`：复用现有的 `DASHSCOPE_API_KEY`，
不需要 torch，也不用下载模型。`"local"` 分支留了 TODO 注释（接
`HuggingFaceCrossEncoder` + `CrossEncoderReranker`），代价是 torch +
sentence-transformers + transformers 以及约 2.3GB 权重，本次未实现。

`rerank_dashscope_model` 可换。三个模型都实测可用，**当前默认 `qwen3-rerank`**
（`qwen3.7-text-rerank` 的免费额度已在测试中跑完）：

| 模型 | 同一次实测的分数 | 备注 |
| --- | --- | --- |
| `qwen3-rerank` | 0.915 / 0.539 / 0.322 | **当前默认** |
| `qwen3.7-text-rerank` | 0.908 / 0.777 / 0.080 | 区分度最好，但额度已耗尽 |
| `gte-rerank-v2` | 0.360 / 0.189 / 0.048 | 偏低且压缩，能力较旧 |

地域显式写在 `rerank_dashscope_base_url`（华北2/北京）。该 SDK 的默认值本来就是北京站，
这一项是可审计的显式化，不是修复。

实现上只按接口返回的**顺序**取前 N 篇，不对分数设任何绝对阈值——理由同第 2 节。
结果异常偏少时按原顺序补齐，保证不会比不重排更差。

## 6. 状态一致性

BM25 索引是进程内缓存，写操作会让它变旧。失效逻辑放在 `app_core.RAGApp._after_write`：

- **为什么在编排层**：索引缓存挂在 `VectorStoreService` 上，而写操作走
  `KnowledgeBaseService`，两者是各自独立的实例，彼此不知道对方；`RAGApp` 是唯一同时
  持有两者的地方。
- **失效范围**：指定 category 时，同时失效该分类与 `__all__` 跨源索引（跨源索引覆盖
  所有分类，任何写都会让它变旧）。
- **不失效的后果**：长驻进程里「刚上传的文件」用 BM25 那一路检索不到——向量那一路能查到，
  所以表现为「有时查得到有时查不到」。

注意这只覆盖**同进程**。多 worker 部署时每个 worker 各持一份索引且互不同步，
限制见 README 的「部署注意事项」。

## 7. 默认值依据

- `hybrid_search_enabled = True`：困难集上 +hybrid 把 baseline 未命中的题救回 **3/4**，
  召回命中率 89.2% → 91.9%。
- `rerank_enabled = False`：
  - 原 20 题**饱和集**（旧口径）上重排是负收益：MRR 1.000 → 0.950 → 0.842；
  - 困难集 37 题上重排是正收益：MRR 0.820 → 0.901、召回 89.2% → 97.3%，且 **0 题被弄丢**；
  - 也就是说重排的收益高度依赖「题有多难」——**简单题上可能退步**，所以保守默认关闭。
    困难场景（用户提问与原文措辞差异大）建议打开，数据见 [eval.md](eval.md)。

## 8. 延迟与冷启动

- BM25 索引首次构建要读全量语料并分词：当前语料（5825 篇 / 约 206 万字）**约 5-6 秒**；
  之后缓存在进程内，单次 BM25 检索约 20ms。
- 所以**进程启动后的第一次混合检索会明显偏慢**，之后恢复正常。
- DashScope 精排每次多一个网络往返，实测约 0.3s；困难集上三模式的平均检索延迟为
  340ms / 388ms / 732ms（仅检索，不含大模型生成）。
- **重排按输入 token 计费**，而粗召回有 `max(bm25_top_k, vector_top_k) = 30` 篇。
  `rerank_max_candidates`（默认 10）用来截断送进去的篇数——不设上限的话每 37 题要约
  41 万 token，设成 10 之后降到约 13.7 万。这个截断**会影响结果**（排在更后面的正确文档
  可能被丢掉），是有意的成本/质量权衡。
