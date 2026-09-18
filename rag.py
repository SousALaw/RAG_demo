from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from langchain_core.output_parsers import StrOutputParser


# 日志里最多打印这么多字符的 prompt 正文，避免服务端日志随上下文一起膨胀。
PROMPT_LOG_PREVIEW_CHARS = 500


def print_prompt(prompt):
    text = prompt.to_string()
    print("="*20)
    print(f"[prompt] 总字符数：{len(text)}")
    print(text[:PROMPT_LOG_PREVIEW_CHARS] + ("..." if len(text) > PROMPT_LOG_PREVIEW_CHARS else ""))
    print("="*20)
    return prompt


def _clip(text: str, limit: int, marker: str) -> str:
    """把 text 裁到不超过 limit 个字符，截断处补 marker，且 marker 计入额度。"""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(marker):
        return text[:limit]
    return text[: limit - len(marker)] + marker


def build_context(docs: list[Document]) -> str:
    """把检索结果拼成参考资料，并保证总长度不超过配置上限。

    约定：docs 已按相关度降序排列（_filter_candidates 返回时就是这个顺序），
    所以「从尾部丢弃」等价于「按相关度从低到高截断」。
    """
    if not docs:
        return "无相关资料"

    total_limit = int(config.context_max_chars)
    per_doc_limit = int(config.context_per_doc_max_chars)
    marker = getattr(config, "context_truncate_marker", "……（已截断）")

    parts: list[str] = []
    used = 0
    for doc in docs:
        remaining = total_limit - used
        if remaining <= 0:
            break

        block = f"文档内容：{doc.page_content}\n 文档元数据：{doc.metadata}\n\n"
        # 先受单篇上限约束，再受剩余总预算约束。
        block = _clip(block, min(per_doc_limit, remaining), marker)
        if not block:
            continue

        parts.append(block)
        used += len(block)

    return "".join(parts) if parts else "无相关资料"


class RAGService(object):
    def __init__(self):
        
        self.vector_store_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model)
        )
        self.retriever = self.vector_store_service.get_retriever()

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，专业并且简洁简要地回答用户问题。参考资料：{context}"),
                ("system", "并且我提供用户的会话历史记录，如下："),
                MessagesPlaceholder("history"),
                ("user", "请回答用户提问：{input}"),
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """
        获取最终执行的链
        """
        def format_for_retriever(value:dict):
            return value["input"] 
        
        def format_for_prompt(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | RunnableLambda(self.retrieve_docs_progressively) | build_context,
            } | RunnableLambda(format_for_prompt) | self.prompt_template | print_prompt |self.chat_model | StrOutputParser()
        )
        
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain

    @staticmethod
    def _normalize_relevance_score(score: float) -> float:
        """将不同区间的分数归一化到 0~1，值越大表示相关度越高。"""
        if score < 0:
            return 0.0
        if score <= 1:
            return score
        # 某些向量库可能返回距离分数（越小越相关），这里做兜底转换。
        return 1 / (1 + score)

    @staticmethod
    def _filter_candidates(
        candidates: list[tuple[Document, float]],
        max_docs: int,
        max_sources: int,
        relative_ratio: float,
        min_floor: float,
    ) -> tuple[list[Document], dict]:
        """按相对分数和来源数量限制筛选候选文档，并返回调试信息。"""
        if not candidates:
            return [], {
                "best_score": 0.0,
                "dynamic_threshold": 0.0,
                "selected": [],
                "dropped": [],
            }

        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        best_score = sorted_candidates[0][1]
        dynamic_threshold = max(min_floor, best_score * relative_ratio)

        selected_docs: list[Document] = []
        selected_sources: set[str] = set()
        selected_keys: set[str] = set()
        selected_debug: list[dict] = []
        dropped_debug: list[dict] = []

        for doc, score in sorted_candidates:
            source = doc.metadata.get("source", "") if doc.metadata else ""
            key = f"{source}::{hash(doc.page_content)}"

            if score < dynamic_threshold:
                dropped_debug.append(
                    {
                        "source": source,
                        "score": round(score, 4),
                        "reason": "below_dynamic_threshold",
                    }
                )
                continue

            if key in selected_keys:
                dropped_debug.append(
                    {
                        "source": source,
                        "score": round(score, 4),
                        "reason": "duplicate_chunk",
                    }
                )
                continue

            if source and source not in selected_sources and len(selected_sources) >= max_sources:
                dropped_debug.append(
                    {
                        "source": source,
                        "score": round(score, 4),
                        "reason": "source_limit",
                    }
                )
                continue

            selected_keys.add(key)
            if source:
                selected_sources.add(source)
            selected_docs.append(doc)
            selected_debug.append(
                {
                    "source": source,
                    "score": round(score, 4),
                    "reason": "selected",
                }
            )

            if len(selected_docs) >= max_docs:
                break

        return selected_docs, {
            "best_score": round(best_score, 4),
            "dynamic_threshold": round(dynamic_threshold, 4),
            "selected": selected_debug,
            "dropped": dropped_debug,
        }

    def _retrieve_docs_progressively_with_debug(self, query: str, category: str | None = None) -> tuple[list[Document], dict]:
        """按渐进式披露策略动态检索文档，并返回调试信息。

        category 为空时跨所有数据源检索；指定时只在该分类内检索。

        hybrid_search_enabled=True 时走「双路粗召回 -> 精排」的独立路径
        （见 _retrieve_hybrid_with_debug）；=False 时走下面这套原有三阶段逻辑。
        """
        if getattr(config, "hybrid_search_enabled", False):
            return self._retrieve_hybrid_with_debug(query, category=category)

        stage_topk = list(getattr(config, "progressive_stage_topk", [1, 3, 5]))
        stage_thresholds = list(getattr(config, "progressive_stage_thresholds", [0.8, 0.65, 0.45]))
        min_docs = int(getattr(config, "progressive_min_docs", 1))
        max_docs = int(getattr(config, "progressive_max_docs", 4))
        max_sources = int(getattr(config, "progressive_max_sources", 2))
        relative_ratio = float(getattr(config, "progressive_relative_score_ratio", 0.9))
        min_floor = float(getattr(config, "progressive_min_relevance_floor", 0.55))

        stage_count = min(len(stage_topk), len(stage_thresholds))
        if stage_count == 0:
            # 走 service 而不是 self.retriever，否则 category 过滤不生效。
            pairs = self.vector_store_service.similarity_search_with_relevance_scores(
                query=query, k=config.similarity_filenum, category=category
            )
            docs = [doc for doc, _ in pairs]
            return docs, {
                "mode": "fallback_retriever",
                "stages": [],
                "filter": {},
                "selected_sources": [doc.metadata.get("source", "") for doc in docs if doc.metadata],
                "hybrid_used": False,
                "rerank_used": False,
                "coarse_count": len(docs),
                "reranked_count": len(docs),
                "score_source": "vector",
            }

        candidates: list[tuple[Document, float]] = []
        candidate_keys: set[str] = set()
        stage_debug: list[dict] = []

        for i in range(stage_count):
            top_k = int(stage_topk[i])
            threshold = float(stage_thresholds[i])
            if top_k <= 0:
                continue

            pairs = self.vector_store_service.similarity_search_with_relevance_scores(
                query=query, k=top_k, category=category
            )
            current_stage = {
                "stage": i + 1,
                "top_k": top_k,
                "threshold": threshold,
                "accepted": 0,
                "inspected": len(pairs),
            }

            for doc, raw_score in pairs:
                score = self._normalize_relevance_score(raw_score)
                if score < threshold:
                    continue

                source = doc.metadata.get("source", "") if doc.metadata else ""
                key = f"{source}::{hash(doc.page_content)}"
                if key in candidate_keys:
                    continue

                candidate_keys.add(key)
                candidates.append((doc, score))
                current_stage["accepted"] += 1

                if len(candidates) >= max_docs * 2:
                    break

            stage_debug.append(current_stage)

            if len(candidates) >= min_docs:
                break

        selected_docs, filter_debug = self._filter_candidates(
            candidates=candidates,
            max_docs=max_docs,
            max_sources=max_sources,
            relative_ratio=relative_ratio,
            min_floor=min_floor,
        )
        if selected_docs:
            selected_sources = []
            for doc in selected_docs:
                source = doc.metadata.get("source", "") if doc.metadata else ""
                if source and source not in selected_sources:
                    selected_sources.append(source)

            debug = {
                "mode": "progressive",
                "stages": stage_debug,
                "filter": filter_debug,
                "selected_sources": selected_sources,
                "hybrid_used": False,
                "rerank_used": False,
                "coarse_count": len(candidates),
                "reranked_count": len(selected_docs),
                "score_source": "vector",
            }
            return selected_docs, debug

        # 三阶段都未命中阈值时兜底返回，避免模型完全无上下文。
        fallback_k = max(stage_topk)
        pairs = self.vector_store_service.similarity_search_with_relevance_scores(
            query=query, k=fallback_k, category=category
        )
        fallback_candidates = [(doc, self._normalize_relevance_score(raw_score)) for doc, raw_score in pairs]
        selected_docs, filter_debug = self._filter_candidates(
            candidates=fallback_candidates,
            max_docs=max_docs,
            max_sources=max_sources,
            relative_ratio=0.0,
            min_floor=0.0,
        )
        selected_docs = selected_docs[:max_docs]
        selected_sources = []
        for doc in selected_docs:
            source = doc.metadata.get("source", "") if doc.metadata else ""
            if source and source not in selected_sources:
                selected_sources.append(source)

        debug = {
            "mode": "fallback",
            "stages": stage_debug,
            "filter": filter_debug,
            "selected_sources": selected_sources,
            "hybrid_used": False,
            "rerank_used": False,
            "coarse_count": len(fallback_candidates),
            "reranked_count": len(selected_docs),
            "score_source": "vector",
        }
        return selected_docs, debug

    def _retrieve_hybrid_with_debug(self, query: str, category: str | None = None) -> tuple[list[Document], dict]:
        """混合检索路径：BM25 + 向量双路粗召回 ->（可选）精排。

        这条路径**不使用 progressive 的分数阈值**：EnsembleRetriever 融合后是 RRF
        排名分，DashScope rerank 的分数量级又随模型差异极大（实测
        qwen3.7-text-rerank 最高可到 0.9，gte-rerank-v2 只在 0.3 上下），
        套 0.8/0.65/0.45 这种绝对阈值一定会全部落空。
        """
        coarse_k = max(int(config.bm25_top_k), int(config.vector_top_k))
        coarse = self.vector_store_service.hybrid_search(query, k=coarse_k, category=category)

        rerank_used = bool(getattr(config, "rerank_enabled", False))
        final_n = int(config.rerank_top_n)
        if rerank_used:
            docs = self.vector_store_service.rerank(query, coarse, final_n)
            score_source = "rerank"
        else:
            # 与重排模式保持同样的最终篇数，这样两种 hybrid 模式对比时
            # 只差「有没有精排」这一个变量。
            docs = coarse[:final_n]
            score_source = "rrf"

        selected_sources = []
        for doc in docs:
            source = doc.metadata.get("source", "") if doc.metadata else ""
            if source and source not in selected_sources:
                selected_sources.append(source)

        debug = {
            "mode": "hybrid",
            "stages": [],
            "filter": {},
            "selected_sources": selected_sources,
            "hybrid_used": True,
            "rerank_used": rerank_used,
            "coarse_count": len(coarse),
            "reranked_count": len(docs),
            "score_source": score_source,
            "bm25_weight": float(config.bm25_weight),
            "vector_weight": float(config.vector_weight),
        }
        return docs, debug

    def retrieve_docs_progressively(self, query: str, category: str | None = None) -> list[Document]:
        """按渐进式披露策略动态检索文档。"""
        docs, _ = self._retrieve_docs_progressively_with_debug(query, category=category)
        return docs

    def search_kb(self, query: str, k: int = 5, category: str | None = None) -> list[dict]:
        """按相似度返回命中切片，供 Agent Tool 做知识库检索。

        与问答链路不同，这里不做阈值过滤、不限制来源数量，
        只返回原始 top-k 命中，避免「为喂给模型而做的裁剪」影响检索结果本身。
        category 为空时跨所有数据源检索。
        """
        if not query or not query.strip():
            return []

        limit = int(k) if k else 5
        if limit <= 0:
            return []

        pairs = self.vector_store_service.similarity_search_with_relevance_scores(
            query=query, k=limit, category=category
        )

        results: list[dict] = []
        for doc, raw_score in pairs:
            metadata = doc.metadata or {}
            results.append(
                {
                    "source": metadata.get("source", ""),
                    "category": metadata.get("category", ""),
                    "content": doc.page_content,
                    # 复用问答链路的归一化口径，保证 score 落在 0~1 且越大越相关。
                    "score": round(self._normalize_relevance_score(raw_score), 4),
                }
            )
        return results

    def get_references_and_debug(self, query: str, category: str | None = None) -> tuple[list[str], dict]:
        """一次检索同时返回参考来源与调试信息。"""
        docs, debug = self._retrieve_docs_progressively_with_debug(query, category=category)
        source_names = []
        for doc in docs:
            source = doc.metadata.get("source") if doc.metadata else None
            if source and source not in source_names:
                source_names.append(source)
        return source_names, debug

    def get_progressive_debug_report(self, query: str, category: str | None = None) -> dict:
        """返回渐进式检索调试报告。"""
        _, debug = self._retrieve_docs_progressively_with_debug(query, category=category)
        return debug

    def get_reference_sources(self, query: str, category: str | None = None) -> list[str]:
        """根据检索结果返回去重后的知识库文件名列表。"""
        source_names, _ = self.get_references_and_debug(query, category=category)
        return source_names
    

if __name__ == "__main__":
    # session_id配置
    session_config = {
        "configurable":{
            "session_id": "user_001",
        }
    }
    res = RAGService().chain.invoke({"input": "夏天穿什么衣服"}, config=session_config)
    print(res)