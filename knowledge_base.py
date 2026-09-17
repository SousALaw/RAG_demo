"""

知识库

按 category（数据源）做逻辑隔离：
- 所有分类共用同一个 Chroma 集合，靠 metadata 的 category 字段过滤；
- 本地文件放在 data/{category}/ 下；
- md5 去重按 (content_md5, category) 隔离，不同源可以有相同内容。

"""
import os
import config_data as config
import hashlib
import threading
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

# md5.text 是「读全部 -> 改写」的共享文件；FastAPI 的同步路由跑在线程池里会真并发，
# 这里用模块级锁保护下面的读改写段。
# 注意：锁只在单进程内有效，API 进程与 Agent 进程各起一份时并不互斥。
_md5_lock = threading.Lock()

# list_kb_files 只列 .txt，并跳过超过该大小的文件：
# 否则像 data/market/market_data.csv（4.4MB）这种大文件会把前端 text_area 拖死。
KB_FILE_SUFFIX = ".txt"
KB_FILE_MAX_BYTES = 1024 * 1024


def normalize_category(category: str) -> str:
    """写操作用：category 必填且必须在白名单内，否则抛 ValueError。

    category 会参与路径拼接（data/{category}/），所以必须做白名单校验，
    否则传 category="../../" 就能把文件写到 data/ 之外。
    """
    name = (category or "").strip()
    if not name:
        raise ValueError("category 必填")
    if name not in config.knowledge_categories:
        allowed = "、".join(config.knowledge_categories.keys())
        raise ValueError(f"未知的 category：{name}，可选值：{allowed}")
    return name


def optional_category(category) -> str | None:
    """读操作用：None 或空字符串表示不做分类过滤（跨所有源）。"""
    if category is None or not str(category).strip():
        return None
    return normalize_category(category)


def _md5_line(md5_str: str, category: str) -> str:
    """md5.text 的一行：content_md5<TAB>category。"""
    return f"{md5_str}\t{category}"


def check_md5(md5_str: str, category: str):
    """
    检查 (content_md5, category) 是否已经处理过。
    返回false表示没有处理过，true表示已经处理过

    按整行 "md5<TAB>category" 精确匹配：旧格式（没有制表符的裸 hash 行）
    永远命中不了，因此不会被误判成已处理。
    """
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()  # 创建一个空的md5文件
        return False  # 如果文件不存在，说明没有处理过任何md5字符串
    
    else:
        target = _md5_line(md5_str, category)
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():
            line = line.strip()     #处理字符串前后的空格和换行符
            if line == target:
                return True  # 该 (md5, 分类) 已经处理过
        return False  # 没有处理过



def save_md5(md5_str: str, category: str):
    """保存 (content_md5, category) 到 md5.text"""
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(_md5_line(md5_str, category) + "\n")


def remove_md5(md5_str: str, category: str):
    """从 md5.text 中删除指定的 (content_md5, category) 行（如果存在）。"""
    if not os.path.exists(config.md5_path):
        return

    target = _md5_line(md5_str, category)
    with open(config.md5_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    new_lines = [line for line in lines if line != target]

    with open(config.md5_path, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")


def get_string_md5(input_str: str, encoding="utf-8"):
    """将传入的字符串转换为md5字符串"""
    # 将字符串转换为字节数组
    byte_str = input_str.encode(encoding = encoding)
    # 计算md5哈希值
    md5_hash = hashlib.md5()
    md5_hash.update(byte_str)  # 更新哈希对象的状态，传入字节数据进行计算
    # 返回十六进制字符串
    return md5_hash.hexdigest()


class KnowledgeBaseService(object):

    """知识库类"""

    def __init__(self):
        # 如果文件夹不存在则创建，如果存在则继续使用
        os.makedirs(config.persist_directory, exist_ok=True)  # 创建持久化存储目录
        self.data_directory = getattr(config, "data_directory", "./data")
        os.makedirs(self.data_directory, exist_ok=True)
        self.chroma = Chroma(
                collection_name=config.collection_name,   #向量库名称
                embedding_function=DashScopeEmbeddings(model="text-embedding-v4"),
                persist_directory=config.persist_directory,
        )  # 向量存储的实例Chroma向量库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,    #文本块的大小
            chunk_overlap = config.chunk_overlap,  #文本块之间的重叠部分大小
            separators = config.separators,  #文本分割的分隔符列表
            length_function=len,
        )  # 文本分割器实例

    def _split_to_chunks(self, data: str) -> list[str]:
        if len(data) > config.max_split_char_num:
            return self.spliter.split_text(data)
        return [data]

    def _build_metadata(self, filename: str, category: str, content_md5: str, operator: str = "admin") -> dict:
        return {
            "source": filename,
            "category": category,
            "content_md5": content_md5,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": operator,
        }

    def _category_dir(self, category: str, create: bool = False) -> str:
        """返回 data/{category} 目录，必要时创建。"""
        path = os.path.join(self.data_directory, category)
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def _data_file_path(self, filename: str, category: str) -> str:
        return os.path.join(self._category_dir(category), filename)

    @staticmethod
    def _category_where(filename: str, category: str) -> dict:
        """按 (来源文件, 分类) 精确定位切片。

        必须带上 category：不同分类允许存在同名文件，
        只按 source 过滤会把别的分类里的同名文件一起删掉。
        """
        return {"$and": [{"source": filename}, {"category": category}]}

    def _list_one_category(self, category: str) -> list[dict]:
        """列出单个分类下的合格文件：只 .txt，且不超过大小上限。"""
        directory = self._category_dir(category)
        if not os.path.isdir(directory):
            return []

        files = []
        for filename in os.listdir(directory):
            if not filename.lower().endswith(KB_FILE_SUFFIX):
                continue

            file_path = os.path.join(directory, filename)
            if not os.path.isfile(file_path):
                continue

            stat = os.stat(file_path)
            if stat.st_size > KB_FILE_MAX_BYTES:
                continue

            files.append(
                {
                    "name": filename,
                    "category": category,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "update_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        return files

    def list_kb_files(self, category: str | None = None) -> list[dict]:
        """列出本地知识库文件。category=None 时遍历所有分类。"""
        target = optional_category(category)
        categories = [target] if target else list(config.knowledge_categories.keys())

        files = []
        for cat in categories:
            files.extend(self._list_one_category(cat))

        files.sort(key=lambda x: x["update_time"], reverse=True)
        return files

    def search_kb_files(self, keyword: str = "", category: str | None = None) -> list[dict]:
        """按文件名关键词过滤知识库文件。"""
        keyword = (keyword or "").strip().lower()
        file_list = self.list_kb_files(category=category)
        if not keyword:
            return file_list
        return [item for item in file_list if keyword in item["name"].lower()]

    def get_file_content(self, filename: str, category: str | None = None, encoding: str = "utf-8") -> str:
        """读取知识库文件内容。category=None 时跨分类查找，返回首个命中。"""
        target = optional_category(category)
        if target:
            file_path = self._data_file_path(filename, target)
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()

        # 跨分类查找：同名文件可能存在于多个分类，这里返回第一个命中的。
        for cat in config.knowledge_categories:
            file_path = self._data_file_path(filename, cat)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()

        raise FileNotFoundError(f"文件不存在：{filename}")

    def upload_by_str(self, data: str, filename: str, category: str | None = None):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        return self.add_new_file(filename=filename, data=data, category=category)

    def add_new_file(self, filename: str, data: str, category: str | None = None, operator: str = "admin") -> str:
        """新增文件并写入向量库。category 必填，为空或不在白名单内会抛 ValueError。"""
        cat = normalize_category(category)

        data = data.strip()
        if not data:
            return "[失败]，文件内容不能为空"

        os.makedirs(self._category_dir(cat), exist_ok=True)
        file_path = self._data_file_path(filename, cat)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)

        # 计算字符串的md5值
        md5_hex = get_string_md5(data)
        # 检查 (md5, 分类) 是否已经被处理过（check_md5 在文件缺失时会创建文件，属写操作）
        with _md5_lock:
            if check_md5(md5_hex, cat):
                return (f"[跳过]，文件{filename}已经被处理过了")

        knowledge_chunks = self._split_to_chunks(data)

        metadata = self._build_metadata(filename=filename, category=cat, content_md5=md5_hex, operator=operator)
        
        self.chroma.add_texts(  #内容加载到向量库中
            texts = knowledge_chunks, 
            metadatas=[metadata for _ in knowledge_chunks]
        )  # 将文本块添加到向量库中，并设置元数据
        
        # 这里没有和上面的 check 合并成同一临界区，因为中间隔着 add_texts 的网络调用；
        # 所以并发重复上传仍可能各写一次记录，锁只保证 md5.text 本身不被写坏。
        with _md5_lock:
            save_md5(md5_hex, cat)  # 保存 (md5, 分类) 到 md5.text

        return (f"[成功]，文件{filename}已经被成功处理了") # 返回True表示处理成功

    def delete_file(self, filename: str, category: str | None = None) -> str:
        """删除知识库文件（本地文件 + 该分类下的向量切片）。category 必填。"""
        cat = normalize_category(category)

        file_path = self._data_file_path(filename, cat)
        old_text = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_text = f.read()
            os.remove(file_path)

        # 按 (source, category) 删除切片：只按 source 会误删其他分类里的同名文件
        self.chroma.delete(where=self._category_where(filename, cat))

        if old_text:
            with _md5_lock:
                remove_md5(get_string_md5(old_text), cat)

        return f"[成功]，文件{filename}已删除"

    def update_file(self, filename: str, new_data: str, category: str | None = None, operator: str = "admin") -> str:
        """更新知识库文件内容（覆盖本地文件并重建该文件向量）。category 必填。"""
        cat = normalize_category(category)

        new_data = new_data.strip()
        if not new_data:
            return "[失败]，文件内容不能为空"

        os.makedirs(self._category_dir(cat), exist_ok=True)
        file_path = self._data_file_path(filename, cat)
        old_data = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_data = f.read()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_data)

        new_md5 = get_string_md5(new_data)

        # 先删除旧切片（限定 category），再写入新切片。
        self.chroma.delete(where=self._category_where(filename, cat))

        knowledge_chunks = self._split_to_chunks(new_data)
        metadata = self._build_metadata(filename=filename, category=cat, content_md5=new_md5, operator=operator)
        self.chroma.add_texts(
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        # remove / check / save 都作用于同一个 md5.text，放进同一临界区，
        # 否则与并发的 add/delete 交错时会丢记录。
        with _md5_lock:
            if old_data:
                remove_md5(get_string_md5(old_data), cat)

            if not check_md5(new_md5, cat):
                save_md5(new_md5, cat)

        return f"[成功]，文件{filename}已更新"


if __name__ == "__main__":  
    service = KnowledgeBaseService()
    service.upload_by_str("这是一个测试字符串，用于测试知识库的上传功能。", "test.txt", category="market")
    