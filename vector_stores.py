import threading

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config_data as config

# jieba 是可选依赖：没装时自动降级为字符二元组，保证本模块始终能被 import。
try:
    import jieba
    _JIEBA_READY = True
except ImportError:  # pragma: no cover
    jieba = None
    _JIEBA_READY = False

# BM25 索引在内存里缓存，构建时要读全量语料，必须加锁避免并发重复构建。
# 与 knowledge_base._md5_lock 同一模式。
_bm25_lock = threading.Lock()

# category=None 时用这个 key 表示「跨所有源」的全量索引
_BM25_ALL = "__all__"


def _tokenize(text: str) -> list[str]:
    """BM25 分词函数。

    BM25Retriever 默认的 preprocess_func 是 text.split()，对中文会把整句当成
    一个 token，等于让 BM25 这一路彻底失效，所以必须替换掉。
    """
    if _JIEBA_READY and str(getattr(config, "bm25_tokenizer", "jieba")).lower() == "jieba":
        return [token for token in jieba.lcut(text) if token.strip()]

    # 降级方案：字符二元组。零依赖，对中文检索仍然有效。
    cleaned = "".join(text.split())
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding:嵌入模型的传入        
        """
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config.collection_name, 
            embedding_function=self.embedding, 
            persist_directory=config.persist_directory
        )
        # BM25 索引缓存：{category 或 "__all__": BM25Retriever}
        self._bm25_indexes: dict = {}

    def get_retriever(self):
        """
        返回向量检索器，方便加入链
        """
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_filenum})

    # ---------------- 纯向量检索（保持原样，作为 baseline） ----------------

    def similarity_search_with_relevance_scores(self, query: str, k: int, category: str | None = None) -> list[tuple[Document, float]]:
        """返回带相关度分数的检索结果。

        category 不为空时只在该分类内检索（Chroma 的 metadata 过滤），
        为空时跨所有分类检索，与改造前行为一致。
        """
        where = {"category": category} if category else None
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k, filter=where)

    # ---------------- BM25 ----------------

    @staticmethod
    def _bm25_key(category: str | None) -> str:
        return category or _BM25_ALL

    def _load_corpus(self, category: str | None) -> tuple[list[str], list[dict]]:
        """从 Chroma 取语料，保证 BM25 索引与向量库同源。"""
        where = {"category": category} if category else None
        got = self.vector_store.get(where=where, include=["documents", "metadatas"])
        return got.get("documents") or [], got.get("metadatas") or []

    def build_bm25_index(self, category: str | None = None):
        """构建（或复用）某个分类的 BM25 索引，懒加载。

        BM25Retriever 是纯内存索引，**不支持 metadata 过滤**，所以按 category
        分别建索引；category=None 用全量索引。双重检查加锁，避免并发重复构建。
        语料为空时返回 None。
        """
        key = self._bm25_key(category)
        cached = self._bm25_indexes.get(key)
        if cached is not None:
            return cached

        with _bm25_lock:
            # 双重检查：等锁期间可能已经被别的线程建好了
            cached = self._bm25_indexes.get(key)
            if cached is not None:
                return cached

            texts, metadatas = self._load_corpus(category)
            if not texts:
                return None

            from langchain_community.retrievers import BM25Retriever

            index = BM25Retriever.from_texts(
                texts=texts,
                metadatas=metadatas,
                preprocess_func=_tokenize,
            )
            index.k = int(config.bm25_top_k)
            self._bm25_indexes[key] = index
            return index

    def invalidate_bm25_index(self, category: str | None = None) -> None:
        """让 BM25 缓存失效。增/改/删之后必须调用，否则会检索到旧语料。

        - 指定 category：失效该分类的索引 **以及跨源全量索引**（全量索引也包含它）
        - category=None：清空全部缓存
        """
        with _bm25_lock:
            # 跨源索引覆盖所有分类，任何写操作都会让它变旧
            self._bm25_indexes.pop(_BM25_ALL, None)
            if category is None:
                self._bm25_indexes.clear()
            else:
                self._bm25_indexes.pop(self._bm25_key(category), None)

    def bm25_search(self, query: str, k: int, category: str | None = None) -> list[Document]:
        """BM25 单路检索。"""
        index = self.build_bm25_index(category)
        if index is None:
            return []

        # 复用已建好的语料向量，只换 k —— 避免并发下改坏共享对象。
        from langchain_community.retrievers import BM25Retriever

        retriever = BM25Retriever(
            vectorizer=index.vectorizer,
            docs=index.docs,
            k=int(k),
            preprocess_func=index.preprocess_func,
        )
        return retriever.invoke(query)

    # ---------------- 混合检索 ----------------

    def _vector_retriever(self, k: int, category: str | None):
        where = {"category": category} if category else None
        return self.vector_store.as_retriever(
            search_kwargs={"k": int(k), "filter": where}
        )

    def hybrid_search(self, query: str, k: int, category: str | None = None) -> list[Document]:
        """BM25 + 向量双路粗召回，用 EnsembleRetriever（RRF）融合。

        返回的是 Document 列表，**没有可比的相关度分数**，
        所以调用方不能再对结果套 progressive 的分数阈值。
        """
        from langchain_classic.retrievers import EnsembleRetriever

        retrievers = []
        weights = []

        bm25 = self.build_bm25_index(category)
        if bm25 is not None:
            retrievers.append(
                self._bm25_retriever_with_k(bm25, int(config.bm25_top_k))
            )
            weights.append(float(config.bm25_weight))

        retrievers.append(self._vector_retriever(int(config.vector_top_k), category))
        weights.append(float(config.vector_weight))

        if len(retrievers) == 1:
            # 语料为空时 BM25 建不出来，退化为纯向量检索
            return retrievers[0].invoke(query)[:k]

        ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)
        return ensemble.invoke(query)[:k]

    @staticmethod
    def _bm25_retriever_with_k(index, k: int):
        from langchain_community.retrievers import BM25Retriever

        return BM25Retriever(
            vectorizer=index.vectorizer,
            docs=index.docs,
            k=int(k),
            preprocess_func=index.preprocess_func,
        )

    # ---------------- 重排序 ----------------

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """对粗召回结果做精排，返回前 top_n 篇。

        【成本】送去重排的候选会被 config.rerank_max_candidates 截断：
        重排按输入 token 计费，粗召回可能有 30 篇，全塞进去每次要 1 万多 token。
        截断会改变结果（排在更后面的正确文档可能被丢掉），这是有意的成本权衡。
        """
        if not docs:
            return []

        max_candidates = int(getattr(config, "rerank_max_candidates", len(docs)))
        if max_candidates > 0:
            docs = docs[:max_candidates]

        limit = min(int(top_n), len(docs))
        if limit <= 0:
            return []

        backend = str(getattr(config, "rerank_backend", "dashscope")).lower()
        if backend == "dashscope":
            return self._rerank_dashscope(query, docs, limit)

        # TODO(local): 本地 Cross-Encoder 分支，本次未实现。
        #   from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        #   from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        #   model = HuggingFaceCrossEncoder(model_name=config.rerank_model)
        #   compressor = CrossEncoderReranker(model=model, top_n=limit)
        #   代价：torch + sentence-transformers + transformers，以及约 2.3GB 的模型权重。
        raise NotImplementedError(
            f"rerank_backend='{backend}' 尚未实现，当前只支持 'dashscope'"
        )

    def _rerank_dashscope(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """用 DashScope TextReRank 精排。

        实测分数绝对值随模型差异极大（qwen3.7-text-rerank 可达 0.9，
        gte-rerank-v2 只有 0.3 上下），所以这里只按返回顺序取前 top_n，
        **不对分数设任何绝对阈值**。
        """
        import dashscope
        from dashscope import TextReRank

        # 显式声明地域（默认值本来就是华北2/北京，这里是可审计的显式化）
        base_url = getattr(config, "rerank_dashscope_base_url", "")
        if base_url:
            dashscope.base_http_api_url = base_url

        resp = TextReRank.call(
            model=config.rerank_dashscope_model,
            query=query,
            documents=[doc.page_content for doc in docs],
            top_n=top_n,
            return_documents=False,
        )

        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(
                "DashScope rerank 失败：status=%s code=%s message=%s"
                % (
                    getattr(resp, "status_code", None),
                    getattr(resp, "code", None),
                    getattr(resp, "message", None),
                )
            )

        output = getattr(resp, "output", None) or {}
        results = output.get("results") if isinstance(output, dict) else getattr(output, "results", None)
        if not results:
            return docs[:top_n]

        ranked: list[Document] = []
        used = set()
        for item in results:
            index = item.get("index") if isinstance(item, dict) else getattr(item, "index", None)
            if index is None or not (0 <= index < len(docs)) or index in used:
                continue
            used.add(index)
            ranked.append(docs[index])

        # 精排返回异常偏少时按原顺序补齐，保证不会比不重排更差
        for index, doc in enumerate(docs):
            if len(ranked) >= top_n:
                break
            if index not in used:
                ranked.append(doc)
        return ranked[:top_n]


if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我的体重180斤，身高1米8，给出尺码推荐")
    print(res)
