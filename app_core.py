"""
RAGApp：统一编排入口。

api_server.py（HTTP 接口）与 agent_tools.py（Agent Tool）都只调用这里，
业务逻辑只实现一遍，两条通道不重复写。
所有方法都是同步方法。
"""

import os

from dotenv import load_dotenv

import config_data as config
from knowledge_base import KnowledgeBaseService, optional_category
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

    def ask(self, query: str, session_id: str = None, category: str | None = None) -> dict:
        """回答问题，返回 answer、references、retrieval_debug、session_id。

        category 为空时跨所有数据源检索。
        """
        # 这里也要校验：检索链不会自己校验 category，
        # 不校验的话非法分类会静默返回空结果而不是报错。
        category = optional_category(category)
        sid = session_id or self.session_id
        config = {"configurable": {"session_id": sid}}
        answer = self.rag.chain.invoke({"input": query}, config=config)
        # 一次调用同时拿到来源与调试信息，避免为了取来源再检索一遍。
        references, retrieval_debug = self.rag.get_references_and_debug(query, category=category)
        return {
            "answer": answer,
            "references": references,
            "session_id": sid,
            "retrieval_debug": retrieval_debug,
        }

    def search_kb(self, query: str, k: int = 5, category: str | None = None) -> list:
        """检索知识库切片，返回 [{source, category, content, score}]。

        注意：这个方法只有 Agent Tool 通道在用，HTTP 路由表里没有对应接口。
        """
        return self.rag.search_kb(query=query, k=k, category=optional_category(category))

    # ---------------- 知识库域 ----------------

    def list_categories(self) -> dict:
        """返回配置里的数据源分类字典 {category: 说明}。"""
        return dict(config.knowledge_categories)

    def list_files(self, category: str | None = None) -> list:
        """列出知识库文件。category 为空时遍历所有分类。"""
        return self.kb.list_kb_files(category=category)

    def get_file_content(self, name: str, category: str | None = None) -> str:
        """读取单个知识库文件内容，文件不存在时抛 FileNotFoundError。

        category 为空时跨分类查找，返回首个命中。
        """
        return self.kb.get_file_content(normalize_filename(name), category=category)

    def add_file(self, name: str, content: str, category: str | None = None) -> dict:
        """新增文件并写入向量库。category 必填，为空会抛 ValueError。"""
        safe_name = normalize_filename(name)
        message = self.kb.add_new_file(filename=safe_name, data=content, category=category)
        return self._result(safe_name, message, category)

    def update_file(self, name: str, content: str, category: str | None = None) -> dict:
        """覆盖文件内容并重建该文件的向量。category 必填。"""
        safe_name = normalize_filename(name)
        message = self.kb.update_file(filename=safe_name, new_data=content, category=category)
        return self._result(safe_name, message, category)

    def delete_file(self, name: str, category: str | None = None) -> dict:
        """删除本地文件与向量库切片。category 必填。"""
        safe_name = normalize_filename(name)
        message = self.kb.delete_file(safe_name, category=category)
        return self._result(safe_name, message, category)

    @staticmethod
    def _result(name: str, message: str, category: str | None = None) -> dict:
        """把服务层的中文结果字符串归类成调用方可判断的状态。

        这段判定以前散落在前端（判断 "[成功]" / "[跳过]"），现在只留一份。
        """
        if "[成功]" in message:
            status = "success"
        elif "[跳过]" in message:
            status = "skipped"
        else:
            status = "error"
        return {"name": name, "category": category or "", "status": status, "message": message}
