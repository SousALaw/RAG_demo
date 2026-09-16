"""
Agent Tool 组：供内部 AI 助手（小秋）调用。

与 api_server.py 的区别：这里进程内直调 RAGApp，不经过 HTTP。
每个 tool 都返回 dict：
    成功 -> {"status": "ok", "data": ...}
    失败 -> {"status": "error", "error": "..."}
任何异常都在这里捕获，不抛给 Agent。
"""

import functools
import json

from app_core import DEFAULT_SESSION_ID, RAGApp

# RAGApp 在首次调用时才构造：这样 import agent_tools 不会因为缺少模型服务而失败，
# 也让 Agent 进程与 API 进程各自持有一份实例。
_rag_app = None


def _get_app() -> RAGApp:
    global _rag_app
    if _rag_app is None:
        _rag_app = RAGApp()
    return _rag_app


# ---------------- Tools ----------------

def tool_ask(query: str, session_id: str = DEFAULT_SESSION_ID) -> dict:
    """基于知识库回答问题，返回答案与参考来源文件。"""
    try:
        data = _get_app().ask(query=query, session_id=session_id)
    except Exception as exc:
        return {"status": "error", "error": f"问答失败：{exc}"}
    return {"status": "ok", "data": data}


def tool_search_kb(query: str, k: int = 5) -> dict:
    """按相似度检索知识库切片，返回 [{source, content, score}]。"""
    try:
        data = _get_app().search_kb(query=query, k=k)
    except Exception as exc:
        return {"status": "error", "error": f"检索失败：{exc}"}
    return {"status": "ok", "data": data}


def tool_list_files() -> dict:
    """列出知识库中的全部文件。"""
    try:
        data = _get_app().list_files()
    except Exception as exc:
        return {"status": "error", "error": f"列出文件失败：{exc}"}
    return {"status": "ok", "data": data}


def tool_get_file_content(name: str) -> dict:
    """读取指定知识库文件的完整内容。"""
    try:
        data = _get_app().get_file_content(name)
    except Exception as exc:
        return {"status": "error", "error": f"读取文件失败：{exc}"}
    return {"status": "ok", "data": {"name": name, "content": data}}


def tool_add_file(name: str, content: str) -> dict:
    """新增知识库文件（落盘并向量化入库）。"""
    try:
        data = _get_app().add_file(name=name, content=content)
    except Exception as exc:
        return {"status": "error", "error": f"新增文件失败：{exc}"}
    return {"status": "ok", "data": data}


def tool_delete_file(name: str) -> dict:
    """删除知识库文件及其向量。"""
    try:
        data = _get_app().delete_file(name)
    except Exception as exc:
        return {"status": "error", "error": f"删除文件失败：{exc}"}
    return {"status": "ok", "data": data}


# ---------------- 工具描述 ----------------

TOOL_SPECS = [
    {
        "name": "tool_ask",
        "description": "基于知识库回答问题，返回答案与参考来源文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户问题"},
                "session_id": {
                    "type": "string",
                    "description": f"会话标识，默认 {DEFAULT_SESSION_ID}",
                },
            },
            "required": ["query"],
        },
        "returns": "{status, data:{answer, references, session_id}}",
    },
    {
        "name": "tool_search_kb",
        "description": "按相似度检索知识库切片，不生成回答。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或问句"},
                "k": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": ["query"],
        },
        "returns": "{status, data:[{source, content, score}]}",
    },
    {
        "name": "tool_list_files",
        "description": "列出知识库中的全部文件。",
        "parameters": {"type": "object", "properties": {}, "required": []},
        "returns": "{status, data:[{name, size_kb, update_time}]}",
    },
    {
        "name": "tool_get_file_content",
        "description": "读取指定知识库文件的完整内容。",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "文件名"}},
            "required": ["name"],
        },
        "returns": "{status, data:{name, content}}",
    },
    {
        "name": "tool_add_file",
        "description": "新增知识库文件，内容会被向量化入库。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "文件名"},
                "content": {"type": "string", "description": "文件文本内容"},
            },
            "required": ["name", "content"],
        },
        "returns": "{status, data:{name, status, message}}",
    },
    {
        "name": "tool_delete_file",
        "description": "删除知识库文件及其向量。",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "文件名"}},
            "required": ["name"],
        },
        "returns": "{status, data:{name, status, message}}",
    },
]

_TOOL_FUNCS = {
    "tool_ask": tool_ask,
    "tool_search_kb": tool_search_kb,
    "tool_list_files": tool_list_files,
    "tool_get_file_content": tool_get_file_content,
    "tool_add_file": tool_add_file,
    "tool_delete_file": tool_delete_file,
}


# ---------------- 可选：LangChain 适配器 ----------------

def _json_text_wrapper(func):
    """LangChain Tool 期望返回文本，这里把 dict 序列化成 JSON 字符串。"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return json.dumps(func(*args, **kwargs), ensure_ascii=False)

    return wrapper


def to_langchain_tools() -> list:
    """【可选】把上面的纯函数包成 LangChain Tool。

    默认不启用；环境里没有 langchain 时返回空列表，不抛异常。
    上面 6 个纯函数 + TOOL_SPECS 才是本模块的主接口。
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        return []

    tools = []
    for spec in TOOL_SPECS:
        func = _TOOL_FUNCS.get(spec["name"])
        if func is None:
            continue
        tools.append(
            StructuredTool.from_function(
                func=_json_text_wrapper(func),
                name=spec["name"],
                description=spec["description"],
            )
        )
    return tools
