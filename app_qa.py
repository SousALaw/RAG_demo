# app_qa.py
from dotenv import load_dotenv
load_dotenv()  # 自动读取项目根目录下的 .env 文件
import streamlit as st

import api_client
from app_file_uploader import render_uploader_page

SESSION_ID = "user_001"


def render_reference_expanders(references: list[str]) -> None:
    """在回答下方渲染可展开的参考文件内容（内容通过 API 读取）。"""
    if not references:
        return

    st.markdown("\n\n参考知识库文件：")
    for filename in references:
        content = None
        error = None
        try:
            content = api_client.get_file_content(filename)
        except Exception as exc:
            error = str(exc)

        with st.expander(filename, expanded=False):
            if error:
                st.warning(f"读取失败：{error}")
            else:
                st.text(content)
                st.caption(f"来源：data/{filename}")


def render_retrieval_debug_panel(debug_info: dict, in_sidebar: bool = False) -> None:
    """渲染渐进式检索调试信息（数据来自 /qa/ask 的 retrieval_debug）。"""
    if not debug_info:
        return

    host = st.sidebar if in_sidebar else st
    panel = host.expander("检索调试信息", expanded=False)
    with panel:
        panel.caption(f"检索模式：{debug_info.get('mode', 'unknown')}")

        stages = debug_info.get("stages", [])
        if stages:
            panel.markdown("阶段命中：")
            panel.table(stages)

        filter_info = debug_info.get("filter", {})
        if filter_info:
            panel.markdown(
                f"筛选阈值：best={filter_info.get('best_score', 0)} / dynamic={filter_info.get('dynamic_threshold', 0)}"
            )

            selected = filter_info.get("selected", [])
            if selected:
                panel.markdown("入选候选：")
                panel.table(selected)

            dropped = filter_info.get("dropped", [])
            if dropped:
                panel.markdown("被过滤候选：")
                panel.table(dropped)

        selected_sources = debug_info.get("selected_sources", [])
        if selected_sources:
            panel.markdown("最终来源文件：")
            panel.write("、".join(selected_sources))


def render_qa_page() -> None:
    # 标题
    st.title("智能客服问答系统")
    st.divider()  # 分隔符
    show_retrieval_debug = st.sidebar.toggle("显示检索调试信息", value=False)
    st.caption(f"后端 API：{api_client.get_base_url()}")

    # 数据源选择：默认「全部」= 跨所有源检索
    try:
        categories = api_client.list_categories()
    except Exception as exc:
        categories = {}
        st.sidebar.caption(f"分类获取失败：{exc}")

    picked = st.sidebar.selectbox(
        "数据源（检索范围）",
        options=["全部"] + list(categories.keys()),
        index=0,
        key="qa_category",
    )
    category = None if picked == "全部" else picked
    if category:
        st.caption(f"当前只在「{categories.get(category, category)}」内检索")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "您好！我是智能客服，有什么可以帮助您的吗？"}
        ]
        # 每条消息是一个字典，包含"role"和"content"两个字段

    if "latest_retrieval_debug" not in st.session_state:
        st.session_state["latest_retrieval_debug"] = {}

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])  # 根据消息的角色，调用不同的消息组件来展示消息内容
            if msg["role"] == "assistant":
                render_reference_expanders(msg.get("references", []))

    # 在页面最下方提供用户输入栏
    prompt = st.chat_input()

    if prompt:
        # 在页面输出用户的提问
        st.chat_message("user").write(prompt)

        # 把用户输入的消息添加到session_state中
        st.session_state["messages"].append({"role": "user", "content": prompt})

        result = None
        with st.spinner("AI正在思考..."):
            # 通过 HTTP 调后端 API；本页不再持有 RAGService 实例。
            try:
                result = api_client.ask(query=prompt, session_id=SESSION_ID, category=category)
            except Exception as exc:
                st.error(f"请求失败：{exc}")

        if result is not None:
            answer = result.get("answer", "")
            references = result.get("references", [])
            retrieval_debug = result.get("retrieval_debug", {})

            with st.chat_message("assistant"):
                st.write(answer)  # 整段渲染，不做流式
                render_reference_expanders(references)

            st.session_state["latest_retrieval_debug"] = retrieval_debug

            # 把AI的回复添加到session_state中
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "references": references,
                    "retrieval_debug": retrieval_debug,
                }
            )

    if show_retrieval_debug:
        latest_debug = st.session_state.get("latest_retrieval_debug", {})
        if not latest_debug:
            for item in reversed(st.session_state["messages"]):
                if item.get("role") == "assistant" and item.get("retrieval_debug"):
                    latest_debug = item["retrieval_debug"]
                    break
        if latest_debug:
            render_retrieval_debug_panel(latest_debug, in_sidebar=True)
        else:
            st.sidebar.caption("暂无检索调试数据，请先提一个问题。")


module = st.sidebar.radio(
    "功能模块",
    ["智能问答", "知识库管理"],
)

if module == "智能问答":
    render_qa_page()
else:
    render_uploader_page()
