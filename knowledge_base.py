"""

知识库

"""
import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str: str):
    """
    检查输入的md5字符串是否已经被处理过
    返回false表示没有处理过，true表示已经处理过
    """
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()  # 创建一个空的md5文件
        return False  # 如果文件不存在，说明没有处理过任何md5字符串
    
    else:
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():
            line = line.strip()     #处理字符串前后的空格和换行符
            if line == md5_str:
                return True  # 如果md5字符串已经存在，说明已经处理过
        return False  # 如果md5字符串不存在，说明没有处理过



def save_md5(md5_str: str):
    """保存md5字符串到数据库"""
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")


def remove_md5(md5_str: str):
    """从md5文件中删除指定哈希值（如果存在）。"""
    if not os.path.exists(config.md5_path):
        return

    with open(config.md5_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    new_lines = [line for line in lines if line != md5_str]

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

    def _build_metadata(self, filename: str, operator: str = "admin") -> dict:
        return {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": operator,
        }

    def _data_file_path(self, filename: str) -> str:
        return os.path.join(self.data_directory, filename)

    def list_kb_files(self) -> list[dict]:
        """列出本地知识库文件。"""
        files = []
        for filename in os.listdir(self.data_directory):
            file_path = self._data_file_path(filename)
            if not os.path.isfile(file_path):
                continue

            stat = os.stat(file_path)
            files.append(
                {
                    "name": filename,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "update_time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        files.sort(key=lambda x: x["update_time"], reverse=True)
        return files

    def search_kb_files(self, keyword: str = "") -> list[dict]:
        """按文件名关键词过滤知识库文件。"""
        keyword = (keyword or "").strip().lower()
        file_list = self.list_kb_files()
        if not keyword:
            return file_list
        return [item for item in file_list if keyword in item["name"].lower()]

    def get_file_content(self, filename: str, encoding: str = "utf-8") -> str:
        """读取知识库文件内容。"""
        file_path = self._data_file_path(filename)
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()

    def upload_by_str(self, data: str, filename: str):
        """将传入的字符串，进行向量化，存入向量数据库中"""
        return self.add_new_file(filename=filename, data=data)
    
    def add_new_file(self, filename: str, data: str, operator: str = "admin") -> str:
        """新增文件并写入向量库。"""
        data = data.strip()
        if not data:
            return "[失败]，文件内容不能为空"

        file_path = self._data_file_path(filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)

        # 计算字符串的md5值
        md5_hex = get_string_md5(data)
        # 检查md5值是否已经被处理过
        if check_md5(md5_hex):
            return (f"[跳过]，文件{filename}已经被处理过了")

        knowledge_chunks = self._split_to_chunks(data)

        metadata = self._build_metadata(filename=filename, operator=operator)
        
        self.chroma.add_texts(  #内容加载到向量库中
            texts = knowledge_chunks, 
            metadatas=[metadata for _ in knowledge_chunks]
        )  # 将文本块添加到向量库中，并设置元数据
        
        save_md5(md5_hex)  # 将md5字符串保存到数据库中

        return (f"[成功]，文件{filename}已经被成功处理了") # 返回True表示处理成功

    def delete_file(self, filename: str) -> str:
        """删除知识库文件（本地文件 + 向量库数据）。"""
        file_path = self._data_file_path(filename)
        old_text = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_text = f.read()
            os.remove(file_path)

        # 按元数据删除对应切片
        self.chroma.delete(where={"source": filename})

        if old_text:
            remove_md5(get_string_md5(old_text))

        return f"[成功]，文件{filename}已删除"

    def update_file(self, filename: str, new_data: str, operator: str = "admin") -> str:
        """更新知识库文件内容（覆盖本地文件并重建该文件向量）。"""
        new_data = new_data.strip()
        if not new_data:
            return "[失败]，文件内容不能为空"

        file_path = self._data_file_path(filename)
        old_data = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                old_data = f.read()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_data)

        # 先删除旧切片，再写入新切片。
        self.chroma.delete(where={"source": filename})

        knowledge_chunks = self._split_to_chunks(new_data)
        metadata = self._build_metadata(filename=filename, operator=operator)
        self.chroma.add_texts(
            texts=knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks],
        )

        if old_data:
            remove_md5(get_string_md5(old_data))

        new_md5 = get_string_md5(new_data)
        if not check_md5(new_md5):
            save_md5(new_md5)

        return f"[成功]，文件{filename}已更新"


if __name__ == "__main__":  
    service = KnowledgeBaseService()
    service.upload_by_str("这是一个测试字符串，用于测试知识库的上传功能。", "test.txt")
    