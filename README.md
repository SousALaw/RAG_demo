# RAG 知识库问答服务（双接口 + 多数据源）

一个中文 RAG 示例项目：知识库入库 -> 向量检索 -> 结合大模型回答。两个特点：

**一、一套业务逻辑，两种接入方式**

| 接入方式 | 面向 | 入口 | 是否走 HTTP |
| --- | --- | --- | --- |
| REST API | 外部系统 | [api_server.py](api_server.py) | 是 |
| Agent Tool | 内部 AI 助手（小秋） | [agent_tools.py](agent_tools.py) | 否，进程内直调 |

两条通道都只调用 [app_core.py](app_core.py) 里的 `RAGApp`，业务逻辑只实现一遍。
前端（Streamlit）已与后端解耦，只通过 HTTP 调 API，不 import 任何服务层代码。

**二、多数据源逻辑隔离**

集市帖子、iwiki、学院知识库、图灵知识库、第三方网站**共用一个 Chroma 集合**，
靠每个切片 metadata 里的 `category` 字段过滤；原文分目录放在 `data/{category}/` 下。
检索可以不指定（跨所有源）或只查某一个源。详见「[数据源分类](#数据源分类category)」一节。

架构图见 [archify/rag-demo-runtime.architecture.html](archify/rag-demo-runtime.architecture.html)（可直接用浏览器打开）。

详细设计与评估见 [docs/](docs/)：
[设计说明](docs/design.md)（混合检索与重排序、BM25 分词坑、默认值依据、状态一致性）｜
[评估说明](docs/eval.md)（评估口径、实测数据、困难集构造）。

本项目基于 [dwgu-ai/RAG_demo](https://github.com/dwgu-ai/RAG_demo) 修改，原项目 MIT 许可（见 [LICENSE](LICENSE)）。

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
存储层              chroma_db/      data/{category}/ · md5.text
                  （单集合 + category 过滤）
```

| 文件 | 作用 |
| --- | --- |
| [app_core.py](app_core.py) | `RAGApp`：唯一编排入口，同时包装 `RAGService` 与 `KnowledgeBaseService` |
| [api_server.py](api_server.py) | FastAPI 接口层，问答域 `/qa/*` + 知识库域 `/kb/*` |
| [agent_tools.py](agent_tools.py) | 7 个 Agent Tool + `TOOL_SPECS` |
| [api_client.py](api_client.py) | 前端共享 httpx 客户端，后端地址从环境变量读 |
| [app_qa.py](app_qa.py) | 智能问答页（Streamlit），带数据源下拉（默认「全部」） |
| [app_file_uploader.py](app_file_uploader.py) | 知识库管理页（Streamlit），四个 Tab 按选中数据源操作 |
| [rag.py](rag.py) | 检索与生成链路，`RAGService` |
| [knowledge_base.py](knowledge_base.py) | 切分、入库、增删改、category 隔离，`KnowledgeBaseService` |
| [vector_stores.py](vector_stores.py) | Chroma 检索封装，`VectorStoreService`；另含 BM25 索引、混合检索 `hybrid_search`、精排 `rerank` |
| [file_history_store.py](file_history_store.py) | 会话历史落盘（滑动窗口 + 磁盘上限） |
| [config_data.py](config_data.py) | 数据源字典、模型名、分块、检索阈值、混合检索/重排开关、上下文长度等配置 |
| [eval/run_eval.py](eval/run_eval.py) | 评估脚本：三种检索模式对比，含 MRR@5 / Recall@K |

## 数据源分类（category）

当前配置了 5 个数据源，定义在 [config_data.py](config_data.py) 的 `knowledge_categories` 里：

| category | 说明 |
| --- | --- |
| `market` | 集市帖子 |
| `iwiki` | iwiki 文档 |
| `college` | 学院知识库 |
| `turing` | 图灵知识库 |
| `third_party` | 第三方网站 |

### 存储布局

```
chroma_db/            # 只有一个集合（config_data.collection_name）
                      # 每个切片的 metadata 带 category / content_md5
data/
  market/             # data/{category}/{filename}
  iwiki/
  college/
  turing/
  third_party/
md5.text              # 每行 "content_md5<TAB>category"
```

约定：

- **单集合 + metadata 过滤**，不为每个源建独立 Chroma 实例，检索时用 `where={"category": ...}`。
- **去重按 (内容, 分类) 隔离**：同一份内容可以同时存在于多个源，互不影响。
- **不同源可以有同名文件**：删除/更新时用 `$and` 同时限定 `source` 与 `category`，
  不会误删其他源的同名文件。
- **加新数据源只改 `knowledge_categories` 一行**，其余代码不用动。

### 参数语义

| 场景 | `category` |
| --- | --- |
| 检索（`/qa/ask`、`tool_search_kb`） | 可选；不传 = 跨所有源 |
| 新增 / 修改 / 删除 | **必填**；不传或不在白名单内 → API 返回 `400`，Python 抛 `ValueError` |
| 读文件内容 / 列文件 | 可选；不传 = 跨分类查找 / 遍历所有分类 |

### 列表过滤

`GET /kb/files` 只返回 `.txt` 且不超过 1MB 的文件（常量见 [knowledge_base.py](knowledge_base.py) 的
`KB_FILE_SUFFIX` / `KB_FILE_MAX_BYTES`），避免像 `data/market/market_data.csv`（4.4MB）
这种大文件把前端文本框拖死。

## 环境要求

- Python 3.11+（开发环境实测 3.11.16）
- 能访问 DashScope（阿里云百炼）的网络
- 一个可用的 `DASHSCOPE_API_KEY`
- 无需外部数据库：Chroma 以本地文件方式持久化

## 安装依赖

依赖由 [pyproject.toml](pyproject.toml) 声明、[uv.lock](uv.lock) 精确锁定（134 个包），
推荐用 [uv](https://github.com/astral-sh/uv) 安装：

```bash
# 1. 安装 uv（二选一）
pipx install uv
# 或官方脚本：https://docs.astral.sh/uv/getting-started/installation/

# 2. 按 uv.lock 复现环境（会自动创建 .venv）
uv sync
```

如果 `pypi.org` 访问不稳定，先指定镜像源再安装（本项目实测该镜像可用）：

```powershell
# Windows PowerShell
$env:UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
uv sync
```
```bash
# macOS / Linux
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
uv sync
```

依赖包含：`streamlit`、`fastapi`、`uvicorn`、`httpx`、`pydantic`、`python-dotenv`、
`dashscope`、`langchain-core` / `langchain-community` / `langchain-classic`（`EnsembleRetriever`
在这里）/ `langchain-chroma` / `langchain-text-splitters`，以及混合检索用的 `rank-bm25`
与中文分词用的 `jieba`。

<details>
<summary>旧方式（已弃用）：<code>pip install -r requirements.txt</code></summary>

`requirements.txt` 仍然保留，但**已弃用**——它不再是安装入口，内容也不再随依赖更新维护，
请使用上面的 `uv sync`。若确实要沿用 pip：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

</details>

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

请先在「知识库管理」页选好数据源再上传 txt，或调 `POST /kb/files`（记得带 `category`），再提问。

## 验证方式

后端起来后访问交互式文档：

- Swagger UI：<http://127.0.0.1:8000/docs>
- ReDoc：<http://127.0.0.1:8000/redoc>

在 `/docs` 里可以直接填参数试接口。最快的端到端验证：

1. `GET /kb/categories` — 应返回 5 个数据源
2. `GET /kb/files?category=iwiki` — 首次为空：`{"files": [], "total": 0}`
3. `POST /kb/files`，body
   `{"name": "demo.txt", "content": "退货政策：签收后 7 天内可无理由退货。", "category": "iwiki"}`
4. `POST /qa/ask`，body `{"query": "退货政策是几天？", "session_id": "user_001", "category": "iwiki"}`
   — `answer` 应基于刚上传的内容作答，`references` 里应出现 `demo.txt`
5. 把 `category` 换成 `market` 再问一次 — 同一问题应当**答不出来**，这就证明分类隔离生效了

## 接口清单

问答域：

| 方法 | 路径 | 请求体 | 响应 |
| --- | --- | --- | --- |
| POST | `/qa/ask` | `{query, session_id, category?}` | `{answer, references, session_id, retrieval_debug}` |

知识库域：

| 方法 | 路径 | 参数 | 响应 |
| --- | --- | --- | --- |
| GET | `/kb/categories` | — | `{categories: {market: "集市帖子", ...}, total}` |
| GET | `/kb/files` | `category?`（query） | `{files: [{name, category, size_kb, update_time}], total}` |
| GET | `/kb/files/{name}` | `category?`（query） | `{name, content}`；文件不存在 404 |
| POST | `/kb/files` | body `{name, content, category}` | `{name, category, status, message}` |
| PUT | `/kb/files/{name}` | `category`（query）+ body `{content}` | `{name, category, status, message}` |
| DELETE | `/kb/files/{name}` | `category`（query） | `{name, category, status, message}` |

带 `?` 的是可选。约定：

- 请求与响应都由 Pydantic 模型定义，字段缺失或类型错误返回 `422`。
- `category` 缺失、为空或不在白名单内：读写类接口返回 `400`，响应体形如 `{"detail": "category 必填"}`。
- `status` 取值 `success` / `skipped` / `error`；同源同内容重复上传返回 `skipped`。
- 所有路由都是同步 `def`，由 FastAPI 丢到线程池执行。
- `session_id` 用于区分会话历史，缺省 `user_001`；它与 `category` 互不影响（历史不按数据源隔离）。

## Agent Tool 说明

[agent_tools.py](agent_tools.py) 提供 7 个工具，**进程内直调 `RAGApp`，不经过 HTTP**。
Agent 与小秋所在的进程需要能 import 本项目，并且能读到 `.env`（`app_core` 已统一加载）。

| Tool | 入参 | 成功时 `data` |
| --- | --- | --- |
| `tool_ask` | `query`, `session_id="user_001"`, `category?` | `{answer, references, session_id, retrieval_debug}` |
| `tool_search_kb` | `query`, `k=5`, `category?` | `[{source, category, content, score}]` |
| `tool_list_categories` | — | `{market: "集市帖子", ...}` |
| `tool_list_files` | `category?` | `[{name, category, size_kb, update_time}]` |
| `tool_get_file_content` | `name`, `category?` | `{name, category, content}` |
| `tool_add_file` | `name`, `content`, `category` | `{name, category, status, message}` |
| `tool_delete_file` | `name`, `category` | `{name, category, status, message}` |

带 `?` 的是可选，含义与 HTTP 接口一致：不传 `category` 表示跨所有数据源。

返回结构统一为信封格式，**任何异常都会被捕获后返回，不会抛给 Agent**：

```python
{"status": "ok",    "data": ...}
{"status": "error", "error": "新增文件失败：category 必填"}
```

`TOOL_SPECS` 是这 7 个工具的手写 JSON Schema 描述（名称、说明、参数、返回），
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

### 检索相关配置

检索开关都在 [config_data.py](config_data.py)：默认 `hybrid_search_enabled = True`、
`rerank_enabled = True`、`rerank_adaptive_enabled = True`：

- **重排序默认开启，但靠门控兜底。** 它在困难场景（用户提问与原文措辞差异大）上实测有效
  （困难集 MRR 0.820 → 0.901、召回 89.2% → 97.3%），但在简单题上会退步——所以**必须和门控一起用**。
  只把 `rerank_adaptive_enabled` 关掉的话，就退回「每题都精排」，简单题上的负收益会回来。
- 门控开启后注意**重排按输入 token 计费**，粗召回有 30 篇，`rerank_max_candidates`（默认 10）
  用来截断送进去的篇数。
- **自适应门控**：`rerank_adaptive_enabled = True`（默认开）会在重排前先判难度——
  粗召回的首篇向量余弦低于 `rerank_activation_threshold`（默认 0.72）才真的调重排，
  全量 57 题上省约 **63%** 的重排调用，简单题误触发 1/20。
  它**只在 `rerank_enabled = True` 时才有意义**；判据是纯本地计算的真余弦，
  文档向量本地取，不额外花 API。`rerank_activation_signal` 默认 `"top1"`，
  可切 `"mean"` / `"top3"`，但实测 `mean` 分不开简单题与困难题，不建议改。
  阈值依据与取舍（门控会漏判一部分难题）见 [docs/design.md](docs/design.md) 第 6 节。

依据与数据见 [docs/design.md](docs/design.md) 与 [docs/eval.md](docs/eval.md)。

### 已知限制

- **上传才能入库**：文件必须通过「知识库管理」页或 `POST /kb/files` 上传（并带上 `category`）。
  直接把 txt 拷进 `data/{category}/` 目录**不会**被向量化——它会出现在文件列表里，但检索永远命中不到。
- **同内容重复上传不会报错，但也不会重复入库**：`add_new_file` 先落盘再查 md5，
  重复内容返回 `skipped`，于是文件在磁盘上存在、却没有对应向量。
  这是去重的预期行为，但排查「列表里有、检索查不到」时先看这一条。
- 运行时状态不入库：`data/`、`chroma_db/`、`chat_histories/`、`md5.text`、`.env` 均已被 `.gitignore` 忽略。
- 迁移或备份时，`data/{category}/`、`chroma_db/`、`md5.text` 三者的状态要一起带走，
  否则会出现「有文件没向量」的不一致。
- 本地状态是明文的：向量库、原文、会话历史都没有加密。
- **category 是逻辑隔离，不是安全边界**：所有数据源共用同一个进程和同一份 API Key，
  无鉴权的调用方可以查询任意 `category`。要真正隔离，得靠独立的服务与凭证。
- 未做并发设计：同步路由跑在线程池里会真并发，`md5.text` 的读改写加了进程内锁（跨进程无效），
  但 Chroma 侧没有加锁，不适合多进程同时大量写。
- **BM25 索引是进程内内存**：当前是单 worker 部署，所以没问题；一旦改成 `--workers N`，
  每个 worker 都会各自持有一份 BM25 索引（内存 ×N）且彼此不同步——A worker 里上传的文件，
  B worker 要等自己的索引失效重建后才用 BM25 检得到。多 worker 部署前需要先解决这一点。
- 环境变量 `DASHSCOPE_API_KEY` 缺失时，服务会在构造模型客户端时就启动失败——这是预期行为。

## 说明

本项目定位是架构可行性验证与入门示例，不是生产级系统：
没有鉴权、没有并发控制、没有结构化错误码、没有流式输出（问答为整段返回）。
