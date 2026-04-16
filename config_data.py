md5_path = "./md5.text"
data_directory = "./data"

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