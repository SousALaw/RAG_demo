"""
前端共享 HTTP 客户端。

Streamlit 页面只通过这里访问后端 API：
- 不 import RAGService / KnowledgeBaseService；
- 不直接读写 data/、chroma_db/、md5.text。

API 地址从环境变量 RAG_API_BASE_URL 读取，默认 http://127.0.0.1:8000。
"""

import os
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

# 前端自身不需要模型 Key，但允许把 RAG_API_BASE_URL 写进项目根目录的 .env。
load_dotenv()

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_SESSION_ID = "user_001"

# /qa/ask 内部要做检索并调用大模型，超时给宽一点。
TIMEOUT_SECONDS = 120.0


def get_base_url() -> str:
    """返回当前配置的 API 地址。"""
    return os.getenv("RAG_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _detail(response: httpx.Response) -> str:
    """尽量取出后端给出的失败原因，取不到就退回响应正文片段。"""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:200]

    if isinstance(payload, dict):
        return str(payload.get("detail", payload))
    return str(payload)


def _request(method: str, path: str, **kwargs):
    url = f"{get_base_url()}{path}"
    kwargs.setdefault("timeout", TIMEOUT_SECONDS)

    response = httpx.request(method, url, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} {_detail(response)}")
    return response.json()


def _file_path(name: str) -> str:
    # 文件名可能是中文，这里做一次百分号编码。
    return f"/kb/files/{quote(name, safe='')}"


def _params(category: str | None) -> dict:
    """只在指定了分类时才带上 category 查询参数。"""
    return {"category": category} if category else {}


# ---------------- 问答域 ----------------

def ask(query: str, session_id: str = DEFAULT_SESSION_ID, category: str | None = None) -> dict:
    """提问，返回 {answer, references, session_id, retrieval_debug}。

    category 留空表示跨所有数据源检索。
    """
    payload = {"query": query, "session_id": session_id}
    if category:
        payload["category"] = category
    return _request("POST", "/qa/ask", json=payload)


# ---------------- 知识库域 ----------------

def list_categories() -> dict:
    """列出所有数据源分类，返回 {category: 中文说明}。"""
    return _request("GET", "/kb/categories").get("categories", {})


def list_files(category: str | None = None) -> list:
    """列出知识库文件，返回 [{name, category, size_kb, update_time}]。

    category 留空表示遍历所有分类。
    """
    return _request("GET", "/kb/files", params=_params(category)).get("files", [])


def get_file_content(name: str, category: str | None = None) -> str:
    """读取文件内容。category 留空时后端跨分类查找，返回首个命中。"""
    return _request("GET", _file_path(name), params=_params(category)).get("content", "")


def add_file(name: str, content: str, category: str | None = None) -> dict:
    """新增文件，返回 {name, category, status, message}。category 必填。"""
    return _request(
        "POST", "/kb/files", json={"name": name, "content": content, "category": category}
    )


def update_file(name: str, content: str, category: str | None = None) -> dict:
    """覆盖文件内容，返回 {name, category, status, message}。category 必填。"""
    return _request("PUT", _file_path(name), params=_params(category), json={"content": content})


def delete_file(name: str, category: str | None = None) -> dict:
    """删除文件，返回 {name, category, status, message}。category 必填。"""
    return _request("DELETE", _file_path(name), params=_params(category))
