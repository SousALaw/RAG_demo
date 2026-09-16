# RAG 知识库问答服务（双接口）

一个中文 RAG 示例项目：知识库入库 -> 向量检索 -> 结合大模型回答。

改造后的重点是**一套业务逻辑、两种接入方式**：

| 接入方式 | 面向 | 入口 | 是否走 HTTP |
| --- | --- | --- | --- |
| REST API | 外部系统 | [api_server.py](api_server.py) | 是 |
| Agent Tool | 内部 AI 助手（小秋） | [agent_tools.py](agent_tools.py) | 否，进程内直调 |

两条通道都只调用 [app_core.py](app_core.py) 里的 `RAGApp`，业务逻辑只实现一遍。
前端（Streamlit）已与后端解耦，只通过 HTTP 调 API，不 import 任何服务层代码。

架构图见 [archify/rag-demo-runtime.architecture.html](archify/rag-demo-runtime.architecture.html)（可直接用浏览器打开）。

## 项目结构

```
消费者        前端 / Agent
             │
接入层        app_qa.py  app_file_uploader.py ── api_client.py (httpx)
             │                                        │ HTTP
接口层       agent_tools.py (进程内直调)          api_server.py (FastAPI)
             │                                        │
编排层       └──────────────► app_core.py : RAGApp ◄──┘
                                  │
服务层                  rag.py ──┴── knowledge_base.py
                          │              │
存储层              chroma_db/      data/ · md5.text
```

| 文件 | 作用 |
| --- | --- |
| [app_core.py](app_core.py) | `RAGApp`：唯一编排入口，同时包装 `RAGService` 与 `KnowledgeBaseService` |
| [api_server.py](api_server.py) | FastAPI 接口层，问答域 `/qa/*` + 知识库域 `/kb/*` |
| [agent_tools.py](agent_tools.py) | 6 个 Agent Tool + `TOOL_SPECS` |
| [api_client.py](api_client.py) | 前端共享 httpx 客户端，后端地址从环境变量读 |
| [app_qa.py](app_qa.py) | 智能问答页（Streamlit） |
| [app_file_uploader.py](app_file_uploader.py) | 知识库管理页（Streamlit） |
| [rag.py](rag.py) | 检索与生成链路，`RAGService` |
| [knowledge_base.py](knowledge_base.py) | 切分、入库、增删改，`KnowledgeBaseService` |
| [vector_stores.py](vector_stores.py) | Chroma 检索封装，`VectorStoreService` |
| [file_history_store.py](file_history_store.py) | 会话历史落盘 |
| [config_data.py](config_data.py) | 模型名、分块、检索阈值等常量 |

## 环境要求

- Python 3.11+（开发环境实测 3.11.16）
- 能访问 DashScope（阿里云百炼）的网络
- 一个可用的 `DASHSCOPE_API_KEY`
- 无需外部数据库：Chroma 以本地文件方式持久化

## 安装依赖

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

依赖包含：`fastapi`、`uvicorn`、`httpx`、`python-dotenv`、`streamlit`、`langchain` 系列、`langchain-chroma`、`dashscope`。

> 若你的虚拟环境里没有 pip（`No module named pip`），可用 [uv](https://github.com/astral-sh/uv) 安装：
> `uv pip install -r requirements.txt --python .venv\Scripts\python.exe`

## 配置 .env

复制模板并填入真实 Key：

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

`.env` 内容说明：

| 变量 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `DASHSCOPE_API_KEY` | 是 | — | 后端与 Agent Tool 调用模型用 |
| `RAG_API_BASE_URL` | 否 | `http://127.0.0.1:8000` | **前端**要访问的后端地址 |
| `RAG_API_HOST` | 否 | `127.0.0.1` | **后端**监听地址 |

要点：

- `.env` 已被 `.gitignore` 忽略，**不要**把真实 Key 提交进仓库。
- 前端本身不需要 Key，它只发 HTTP 请求；Key 只被后端进程读取。
- `.env` 由 `app_core.py` 统一 `load_dotenv()`，因此 API 进程、Agent 进程、Streamlit 进程都能读到。

## 启动后端和前端

两个进程分开跑，通过 HTTP 通信。

### 1. 启动后端

```bash
uvicorn api_server:app --host 127.0.0.1 --port 8000
```

或者用模块自带入口（等价，监听地址读 `RAG_API_HOST`）：

```bash
python api_server.py
```

### 2. 启动前端

```bash
streamlit run app_qa.py
```

浏览器打开 `http://localhost:8501`，左侧可在「智能问答」和「知识库管理」之间切换。
知识库管理页也可以单独启动：

```bash
streamlit run app_file_uploader.py
```

### 3. 首次使用必须先建知识库

`data/`、`chroma_db/`、`md5.text` 都是运行时状态且不入库（见 `.gitignore`），
**克隆下来是空的**。此时直接提问会得到「无相关资料」，因为向量库里没有任何内容。

请先在「知识库管理」页上传 txt，或调 `POST /kb/files`，再提问。

## 验证方式

后端起来后访问交互式文档：

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

在 `/docs` 里可以直接填参数试接口。最快的端到端验证：

1. `GET /kb/files` — 应返回 `{"files": [], "total": 0}`（首次为空）
2. `POST /kb/files`，body `{"name": "demo.txt", "content": "退货政策：签收后 7 天内可无理由退货。"}`
3. `POST /qa/ask`，body `{"query": "退货政策是几天？", "session_id": "user_001"}`
   — `answer` 应基于刚上传的内容作答，`references` 里应出现 `demo.txt`

## 接口清单

问答域：

| 方法 | 路径 | 请求体 | 响应 |
| --- | --- | --- | --- |
| POST | `/qa/ask` | `{query, session_id}` | `{answer, references, session_id, retrieval_debug}` |

知识库域：

| 方法 | 路径 | 请求体 | 响应 |
| --- | --- | --- | --- |
| GET | `/kb/files` | — | `{files: [{name, size_kb, update_time}], total}` |
| GET | `/kb/files/{name}` | — | `{name, content}`；文件不存在返回 404 |
| POST | `/kb/files` | `{name, content}` | `{name, status, message}` |
| PUT | `/kb/files/{name}` | `{content}` | `{name, status, message}` |
| DELETE | `/kb/files/{name}` | — | `{name, status, message}` |

约定：

- 请求与响应都由 Pydantic 模型定义，字段缺失或类型错误返回 422。
- `status` 取值 `success` / `skipped` / `error`；内容重复时返回 `skipped`。
- 所有路由都是同步 `def`，由 FastAPI 丢到线程池执行。
- `session_id` 用于区分会话历史，缺省 `user_001`。

## Agent Tool 说明

[agent_tools.py](agent_tools.py) 提供 6 个工具，**进程内直调 `RAGApp`，不经过 HTTP**。
Agent 与小秋所在的进程需要能 import 本项目，并且能读到 `.env`（`app_core` 已统一加载）。

| Tool | 入参 | 成功时 `data` |
| --- | --- | --- |
| `tool_ask` | `query`, `session_id="user_001"` | `{answer, references, session_id, retrieval_debug}` |
| `tool_search_kb` | `query`, `k=5` | `[{source, content, score}]` |
| `tool_list_files` | — | `[{name, size_kb, update_time}]` |
| `tool_get_file_content` | `name` | `{name, content}` |
| `tool_add_file` | `name`, `content` | `{name, status, message}` |
| `tool_delete_file` | `name` | `{name, status, message}` |

返回结构统一为信封格式，**任何异常都会被捕获后返回，不会抛给 Agent**：

```python
{"status": "ok",    "data": ...}
{"status": "error", "error": "读取文件失败：..."}
```

`TOOL_SPECS` 是这 6 个工具的手写 JSON Schema 描述（名称、说明、参数、返回），
供任意 Agent 框架注册使用，不绑定具体框架。

可选适配器：`to_langchain_tools()` 会把它们包成 LangChain `StructuredTool`，
环境里没有 langchain 时返回空列表、不报错。默认不使用。

`tool_search_kb` 是纯检索（不生成回答），**目前只有 Agent 通道有，没有对应的 HTTP 路由**。

## 部署注意事项

**无鉴权，不要暴露公网。** 这一点最重要：

- 服务没有任何认证、授权、限流和租户隔离，`session_id` 由调用方自行传入。
- 默认只监听 `127.0.0.1`，也就是仅本机可访问。

关于 `--host 0.0.0.0`：

- 它会让服务监听所有网卡，**在无鉴权前提下等于对局域网（乃至公网）开放**，
  任何能连上 8000 端口的人都能读取、修改、删除你的知识库。
- 只有在容器编排、内网跳板、或已确认有防火墙 / 安全组 / 反向代理鉴权兜底时再用。
- 若确实需要，同时设置 `RAG_API_HOST=0.0.0.0`（或用 `--host 0.0.0.0` 启动），
  并务必限制来源 IP。

其他：

- **上传才能入库**：文件必须通过「知识库管理」页或 `POST /kb/files` 上传。
  直接把 txt 拷进 `data/` 目录**不会**被向量化——它会出现在文件列表里，但检索永远命中不到。
- 运行时状态不入库：`data/`、`chroma_db/`、`chat_histories/`、`md5.text`、`.env` 均已被 `.gitignore` 忽略。
- 迁移或备份时，`data/`、`chroma_db/`、`md5.text` 三者的状态需要一起带走，否则会出现「有文件没向量」的不一致。
- 本地状态是明文的：向量库、原文、会话历史都没有加密。
- 未做并发设计：同步路由跑在线程池里会真并发，`md5.text` 的读改写加了进程内锁（跨进程无效），
  但 Chroma 侧没有加锁，不适合多进程同时大量写。
- 环境变量 `DASHSCOPE_API_KEY` 缺失时，服务会在构造模型客户端时就启动失败——这是预期行为。

## 说明

本项目定位是架构可行性验证与入门示例，不是生产级系统：
没有鉴权、没有并发控制、没有结构化错误码、没有流式输出（问答为整段返回）。
