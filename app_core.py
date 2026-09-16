"""
RAGApp：统一编排入口。

api_server.py（HTTP 接口）与 agent_tools.py（Agent Tool）都只调用这里，
业务逻辑只实现一遍，两条通道不重复写。
所有方法都是同步方法。
"""

import os

from dotenv import load_dotenv

from knowledge_base import KnowledgeBaseService
from rag import RAGService

# 后端接口与 Agent Tool 都需要 DASHSCOPE_API_KEY，统一在这里加载项目根目录的 .env，
# 这样 api_server.py / agent_tools.py / 自定义 Agent 脚本都不必各自记得加载。
# Key 只在构造服务时（RAGApp.__init__）被读取，所以放在 import 之后是安全的。
load_dotenv()

DEFAULT_SESSION_ID = "user_001"


def normalize_filename(name: str) -> str:
    """只保留文件名部分，阻断 ../ 之类的路径穿越。

    data/ 目录下的读取、覆盖、删除都先经过这里。
    """
    raw = (name or "").strip().replace("\\", "/")
    return os.path.basename(raw)


class RAGApp(object):
    """编排 RAGService 与 KnowledgeBaseService，对外提供一组同步能力。"""

    def __init__(self, session_id: str = DEFAULT_SESSION_ID):
        self.session_id = session_id
        self.rag = RAGService()
        self.kb = KnowledgeBaseService()

    # ---------------- 问答域 ----------------

    def ask(self, query: str, session_id: str = None) -> dict:
        """回答问题，返回 answer、references、retrieval_debug、session_id。"""
        sid = session_id or self.session_id
        config = {"configurable": {"session_id": sid}}
        answer = self.rag.chain.invoke({"input": query}, config=config)
        # 一次调用同时拿到来源与调试信息，避免为了取来源再检索一遍。
        references, retrieval_debug = self.rag.get_references_and_debug(query)
        return {
            "answer": answer,
            "references": references,
            "session_id": sid,
            "retrieval_debug": retrieval_debug,
        }

    def search_kb(self, query: str, k: int = 5) -> list:
        """检索知识库切片，返回 [{source, content, score}]。

        注意：这个方法只有 Agent Tool 通道在用，HTTP 路由表里没有对应接口。
        """
        return self.rag.search_kb(query=query, k=k)

    # ---------------- 知识库域 ----------------

    def list_files(self) -> list:
        """列出 knowledge base 文件。"""
        return self.kb.list_kb_files()

    def get_file_content(self, name: str) -> str:
        """读取单个知识库文件内容，文件不存在时抛 FileNotFoundError。"""
        return self.kb.get_file_content(normalize_filename(name))

    def add_file(self, name: str, content: str) -> dict:
        """新增文件并写入向量库。"""
        safe_name = normalize_filename(name)
        message = self.kb.add_new_file(filename=safe_name, data=content)
        return self._result(safe_name, message)

    def update_file(self, name: str, content: str) -> dict:
        """覆盖文件内容并重建该文件的向量。"""
        safe_name = normalize_filename(name)
        message = self.kb.update_file(filename=safe_name, new_data=content)
        return self._result(safe_name, message)

    def delete_file(self, name: str) -> dict:
        """删除本地文件与向量库切片。"""
        safe_name = normalize_filename(name)
        message = self.kb.delete_file(safe_name)
        return self._result(safe_name, message)

    @staticmethod
    def _result(name: str, message: str) -> dict:
        """把服务层的中文结果字符串归类成调用方可判断的状态。

        这段判定以前散落在前端（判断 "[成功]" / "[跳过]"），现在只留一份。
        """
        if "[成功]" in message:
            status = "success"
        elif "[跳过]" in message:
            status = "skipped"
        else:
            status = "error"
        return {"name": name, "status": status, "message": message}
