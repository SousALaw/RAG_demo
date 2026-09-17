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