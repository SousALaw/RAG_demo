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


# ---------------- 问答域 ----------------

def ask(query: str, session_id: str = DEFAULT_SESSION_ID) -> dict:
    """提问，返回 {answer, references, session_id}。"""
    return _request("POST", "/qa/ask", json={"query": query, "session_id": session_id})


# ---------------- 知识库域 ----------------

def list_files() -> list:
    """列出知识库文件，返回 [{name, size_kb, update_time}]。"""
    return _request("GET", "/kb/files").get("files", [])


def get_file_content(name: str) -> str:
    """读取文件内容。"""
    return _request("GET", _file_path(name)).get("content", "")


def add_file(name: str, content: str) -> dict:
    """新增文件，返回 {name, status, message}。"""
    return _request("POST", "/kb/files", json={"name": name, "content": content})


def update_file(name: str, content: str) -> dict:
    """覆盖文件内容，返回 {name, status, message}。"""
    return _request("PUT", _file_path(name), json={"content": content})


def delete_file(name: str) -> dict:
    """删除文件，返回 {name, status, message}。"""
    return _request("DELETE", _file_path(name))
