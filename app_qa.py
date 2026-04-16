import streamlit as st
from rag import RAGService
import time
import os
import config_data as config
from app_file_uploader import render_uploader_page


def read_kb_file_content(filename: str) -> str:
    """读取 data 目录下知识库文件内容。"""
    # 仅允许读取 data 目录下的文件名，避免路径穿越。
    safe_name = os.path.basename(filename)
    data_dir = getattr(config, "data_directory", "./data")
    file_path = os.path.join(data_dir, safe_name)
    if not os.path.isfile(file_path):
        return "[文件不存在] 可能已被删除或重命名。"

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def render_reference_expanders(references: list[str]) -> None:
    """在回答下方渲染可展开的参考文件内容。"""
    if not references:
        return

    st.markdown("\n\n参考知识库文件：")
    for index, filename in enumerate(references):
        content = read_kb_file_content(filename)
        with st.expander(filename, expanded=False):
            st.text(content)
            st.caption(f"来源：data/{os.path.basename(filename)}")


def render_retrieval_debug_panel(debug_info: dict, in_sidebar: bool = False) -> None:
    """渲染渐进式检索调试信息。"""
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

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "您好！我是智能客服，有什么可以帮助您的吗？"}]
        # 每条消息是一个字典，包含"role"和"content"两个字段

    if "latest_retrieval_debug" not in st.session_state:
        st.session_state["latest_retrieval_debug"] = {}

    if "rag" not in st.session_state:
        st.session_state["rag"] = RAGService()  # 实例化RAGService类，创建一个RAG服务对象

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

        ai_res_list = []  # 用来缓存AI回复的列表

        with st.spinner("AI正在思考..."):
            # 调用RAG服务对象的chain链来处理用户输入的消息，并传入会话配置
            time.sleep(1)  # 模拟AI思考的时间
            res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config=config.session_config)

            # 定义一个生成器函数，用来捕获AI回复的流式输出，并将每个输出块添加到缓存列表中
            def capture(generator, cache_list):
                for chunk in generator:
                    cache_list.append(chunk)
                    yield chunk

            with st.chat_message("assistant"):
                st.write_stream(capture(res_stream, ai_res_list))  # 把AI的回复以流式的方式展示在页面上

                # 基于同一问题再做一次检索，补充可展开的参考文件。
                references, retrieval_debug = st.session_state["rag"].get_references_and_debug(prompt)
                render_reference_expanders(references)
                st.session_state["latest_retrieval_debug"] = retrieval_debug

            # 把AI的回复添加到session_state中
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": "".join(ai_res_list),
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