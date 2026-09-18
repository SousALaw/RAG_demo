md5_path = "./md5.text"
data_directory = "./data"

# ===== 知识库数据源分类（逻辑隔离）=====
# 单 Chroma 集合 + metadata 的 category 字段做过滤；本地文件放 data/{category}/。
# 加新数据源只需在这里加一行，其余代码不用动。
knowledge_categories = {
    "market": "集市帖子",
    "iwiki": "iwiki 文档",
    "college": "学院知识库",
    "turing": "图灵知识库",
    "third_party": "第三方网站",
}

#Chroma向量库配置
collection_name = "rag"  # 向量库名称
persist_directory = "./chroma_db"  # 向量库持久化存储

#文本分割配置
chunk_size = 1000      # 每个文本块的大小
chunk_overlap = 100    # 文本块之间的重叠部分大小
separators = ["\n\n", "\n", " ", ".", "?", "!", "", "。", "？", "！"]  # 文本分割的分隔符列表
max_split_char_num = 1000    # 文本分割的阈值

# 检索返回匹配的文档数量
similarity_filenum = 1 

# 渐进式披露检索配置（高置信 -> 中置信 -> 兜底扩展）
progressive_stage_topk = [1, 3, 5]
progressive_stage_thresholds = [0.8, 0.65, 0.45]
progressive_min_docs = 1
progressive_max_docs = 4
progressive_max_sources = 2
progressive_relative_score_ratio = 0.9
progressive_min_relevance_floor = 0.55

# ===== 混合检索与重排序 =====
# hybrid_search_enabled=False 时走原来的渐进式纯向量检索（逐字节不变，可作 baseline）；
# =True 时走「双路粗召回 -> 精排」的独立路径，不再套 progressive 的分数阈值。
hybrid_search_enabled = True       # 是否启用混合检索（BM25 + 向量双路粗召回）
bm25_weight = 0.3                  # BM25 权重
vector_weight = 0.7                # 向量权重
bm25_top_k = 30                    # BM25 粗召回数量
vector_top_k = 30                  # 向量粗召回数量
# 中文分词方式：jieba（默认）| bigram（字符二元组）。jieba 未安装时自动降级为 bigram。
# 注意：BM25Retriever 默认分词是 text.split()，对中文会把整句当一个 token，必须替换。
bm25_tokenizer = "jieba"

rerank_enabled = False             # 是否启用重排序
# 默认 False 的依据：在 eval/eval_dataset.json（20 题）上实测，开启重排后
# MRR@5 从 1.000 掉到 0.842、召回命中率从 100% 掉到 95%，没有一题变好。
# 那套评估集对 baseline 已 100% 饱和、缺乏区分度，等 eval/eval_hard.json
# （困难子集）跑完再据此重新决定默认值。
rerank_backend = "dashscope"       # dashscope（本次实现）| local（预留，见 vector_stores.rerank 的 TODO）
rerank_model = "BAAI/bge-reranker-v2-m3"   # local 后端用的模型，本次未实现
# dashscope 后端模型。注意：qwen3.7-text-rerank 的免费额度已被测试跑完，
# 现改用 qwen3-rerank。换行即可切换（三个都实测可用）：
#   qwen3-rerank         当前默认
#   qwen3.7-text-rerank  分数分离度最好（0.908 / 0.777 / 0.080），额度已耗尽
#   gte-rerank-v2        分数偏低且压缩（0.360 / 0.189 / 0.048），能力较旧
rerank_dashscope_model = "qwen3-rerank"
# 【成本控制，会影响结果】送去重排的候选篇数上限。
# 重排是按输入 token 计费的，而粗召回有 max(bm25_top_k, vector_top_k)=30 篇；
# 不限的话每次调用要 1 万多 token，而最终只需要 rerank_top_n 篇。
# 调大 = 更全但更贵，调小 = 更省但可能漏掉排在后面的正确文档。
rerank_max_candidates = 10
# 显式声明华北2（北京）地域：该 SDK 默认值本来就是北京站，这里是可审计的显式化。
rerank_dashscope_base_url = "https://dashscope.aliyuncs.com/api/v1"
rerank_top_n = 5                   # 精排后保留数量

# ===== 上下文长度控制（防止多轮对话 Token / 延迟无限增长）=====
history_max_messages = 10          # 注入 prompt 的历史消息条数上限，10 条 = 5 轮
history_disk_max_messages = 100    # 落到磁盘的历史消息条数上限，100 条 = 50 轮
context_max_chars = 4000           # 参考资料（context）总字符数上限
context_per_doc_max_chars = 1500   # 单个文档块的字符数上限
context_truncate_marker = "……（已截断）"  # 发生截断时补在末尾，便于日志确认

#向量模型配置
embedding_model = "text-embedding-v4"  # 向量模型名称

# chat模型配置
chat_model = "qwen3-max-2026-01-23"  # chat模型名称，qwen3新用户有免费额度适合练手

# session_id配置
session_config = {
    "configurable":{
        "session_id": "user_001",
    }
}