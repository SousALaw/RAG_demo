"""
FastAPI 双接口层。

问答域：/qa/*
知识库域：/kb/*

路由一律用 def（同步），由 FastAPI 丢到线程池执行；
底层 Chroma / 向量化 / 大模型调用本身都是同步的。
"""

import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app_core import normalize_filename, RAGApp

# .env 由 app_core 统一加载（DASHSCOPE_API_KEY 在 RAGApp 构造时用到）。

# 监听地址可用 RAG_API_HOST 覆盖，默认只绑本机回环。
# 本服务没有任何鉴权，改成 0.0.0.0 等于对局域网开放，必须自行用防火墙兜底。
API_HOST = os.getenv("RAG_API_HOST", "127.0.0.1")
API_PORT = 8000

app = FastAPI(title="RAG Demo API", version="0.1.0")

# 允许本地前端调用。注意：Streamlit 是用 httpx 从服务端进程调 API 的，
# 属进程间调用，走不到 CORS；这里主要是给将来浏览器 / JS 直连留的路。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_app = RAGApp()


# ---------------- 请求 / 响应模型 ----------------

class AskRequest(BaseModel):
    query: str = Field(min_length=1, description="用户问题")
    session_id: str = Field(default="user_001", min_length=1, description="会话标识")
    category: str | None = Field(default=None, description="只在该数据源内检索；留空则跨所有源")


class AskResponse(BaseModel):
    answer: str
    references: list[str]
    session_id: str
    # 检索调试信息：前端「显示检索调试信息」面板在用。
    # 这是对约定字段的加法超集，删掉它前端只会少一个面板。
    retrieval_debug: dict = Field(default_factory=dict)


class FileInfo(BaseModel):
    name: str
    category: str
    size_kb: float
    update_time: str


class FileListResponse(BaseModel):
    files: list[FileInfo]
    total: int


class CategoryListResponse(BaseModel):
    categories: dict[str, str]
    total: int


class FileContentResponse(BaseModel):
    name: str
    content: str


class AddFileRequest(BaseModel):
    name: str = Field(min_length=1, description="文件名")
    content: str = Field(min_length=1, description="文件文本内容")
    category: str | None = Field(default=None, description="数据源分类；必填")


class UpdateFileRequest(BaseModel):
    content: str = Field(min_length=1, description="新的文件文本内容")


class MutationResponse(BaseModel):
    name: str
    category: str = ""
    status: str
    message: str


# ---------------- 问答域 ----------------

@app.post("/qa/ask", response_model=AskResponse)
def qa_ask(req: AskRequest) -> AskResponse:
    """基于知识库回答问题。"""
    try:
        result = rag_app.ask(query=req.query, session_id=req.session_id, category=req.category)
    except ValueError as exc:
        # category 为空或不在白名单内
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # 未做结构化错误设计，只是让调用方能看见失败原因（例如缺少 API Key、超时）。
        raise HTTPException(status_code=502, detail=f"问答链路失败：{exc}")
    return AskResponse(**result)


# ---------------- 知识库域 ----------------

@app.get("/kb/categories", response_model=CategoryListResponse)
def kb_list_categories() -> CategoryListResponse:
    """列出所有数据源分类。"""
    categories = rag_app.list_categories()
    return CategoryListResponse(categories=categories, total=len(categories))


@app.get("/kb/files", response_model=FileListResponse)
def kb_list_files(category: str | None = Query(default=None, description="只看某个数据源；留空遍历所有源")) -> FileListResponse:
    """列出知识库文件。"""
    try:
        files = rag_app.list_files(category=category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return FileListResponse(files=files, total=len(files))


@app.get("/kb/files/{name}", response_model=FileContentResponse)
def kb_get_file(name: str, category: str | None = Query(default=None, description="数据源分类；留空跨分类查找")) -> FileContentResponse:
    """读取单个知识库文件内容。"""
    try:
        content = rag_app.get_file_content(name, category=category)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"文件不存在：{normalize_filename(name)}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return FileContentResponse(name=normalize_filename(name), content=content)


@app.post("/kb/files", response_model=MutationResponse)
def kb_add_file(req: AddFileRequest) -> MutationResponse:
    """新增知识库文件（落盘 + 向量化入库）。category 必填。"""
    try:
        return MutationResponse(**rag_app.add_file(name=req.name, content=req.content, category=req.category))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/kb/files/{name}", response_model=MutationResponse)
def kb_update_file(name: str, req: UpdateFileRequest, category: str | None = Query(default=None, description="数据源分类；必填")) -> MutationResponse:
    """覆盖知识库文件内容并重建向量。category 必填。"""
    try:
        return MutationResponse(**rag_app.update_file(name=name, content=req.content, category=category))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/kb/files/{name}", response_model=MutationResponse)
def kb_delete_file(name: str, category: str | None = Query(default=None, description="数据源分类；必填")) -> MutationResponse:
    """删除知识库文件与对应向量。category 必填。"""
    try:
        return MutationResponse(**rag_app.delete_file(name, category=category))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
