"""
基于 Streamlit 完成 Web 端文件上传功能。
"""
import time
import streamlit as st
from knowledge_base import KnowledgeBaseService


def decode_text_file(file_bytes: bytes) -> str:
    """优先使用 UTF-8 解码，失败时回退 GBK。"""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("gbk")


def render_uploader_page() -> None:
    st.title("知识库管理")
    st.caption("在此模块中可对知识库文件进行增删改查。")

    # 删除成功后，下一轮脚本开始时再清理控件状态，避免修改已实例化控件引发异常。
    if st.session_state.pop("kb_reset_delete_state", False):
        st.session_state.pop("kb_delete_confirm", None)
        st.session_state.pop("kb_delete_select", None)
        st.session_state.pop("kb_delete_batch_confirm", None)
        st.session_state.pop("kb_delete_batch_select", None)

    if "kb_action_message" in st.session_state:
        action_msg = st.session_state.pop("kb_action_message")
        if "[成功]" in action_msg:
            st.success(action_msg)
        elif "[跳过]" in action_msg:
            st.info(action_msg)
        else:
            st.error(action_msg)

    if "service" not in st.session_state:
        st.session_state["service"] = KnowledgeBaseService()

    service = st.session_state["service"]
    tab_create, tab_read, tab_update, tab_delete = st.tabs(["新增", "查询", "修改", "删除"])

    with tab_create:
        st.subheader("新增知识库文件")
        left_col, right_col = st.columns([3, 2], gap="large")

        with left_col:
            st.markdown("#### 上传文件（支持单个/批量）")
            selected_files = st.file_uploader(
                "请选择一个或多个 txt 文件",
                type=["txt"],
                accept_multiple_files=True,
                key="kb_create_uploader",
            )

            if selected_files:
                st.write(f"已选择 {len(selected_files)} 个文件")
                st.write("文件清单：" + "、".join([f.name for f in selected_files]))

                preview_file = selected_files[0]
                preview_raw_bytes = preview_file.getvalue()
                if preview_raw_bytes:
                    try:
                        preview_text = decode_text_file(preview_raw_bytes)
                        preview = preview_text[:200] + ("..." if len(preview_text) > 200 else "")
                        st.text_area("首个文件内容预览", value=preview, height=120, disabled=True)
                    except UnicodeDecodeError:
                        st.warning(f"首个文件 {preview_file.name} 编码无法识别，上传时会记录为失败。")

                if st.button("开始上传", type="primary", key="kb_create_btn"):
                    success_count = 0
                    skip_count = 0
                    fail_count = 0
                    detail_messages = []

                    with st.spinner("正在上传文件..."):
                        for file_item in selected_files:
                            raw_bytes = file_item.getvalue()
                            if not raw_bytes:
                                fail_count += 1
                                detail_messages.append(f"[失败]，文件{file_item.name}内容为空")
                                continue

                            try:
                                text = decode_text_file(raw_bytes)
                                result = service.add_new_file(filename=file_item.name, data=text)
                                detail_messages.append(result)
                                if "[成功]" in result:
                                    success_count += 1
                                elif "[跳过]" in result:
                                    skip_count += 1
                                else:
                                    fail_count += 1
                            except UnicodeDecodeError:
                                fail_count += 1
                                detail_messages.append(f"[失败]，文件{file_item.name}编码无法识别")

                        time.sleep(0.3)

                    st.success(f"上传完成：成功 {success_count}，跳过 {skip_count}，失败 {fail_count}")
                    with st.expander("查看上传明细"):
                        for msg in detail_messages:
                            st.write(msg)

        with right_col:
            st.markdown("### 已上传文件")
            uploaded_files = service.list_kb_files()
            st.caption(f"当前共 {len(uploaded_files)} 个文件")

            if not uploaded_files:
                st.info("暂无已上传文件")
            else:
                st.dataframe(uploaded_files, width="stretch", height=260)

    with tab_read:
        st.subheader("查询知识库文件")
        keyword = st.text_input("按文件名关键词筛选", key="kb_search_keyword")
        file_items = service.search_kb_files(keyword)

        if not file_items:
            st.info("暂无匹配文件。")
        else:
            st.dataframe(file_items, width="stretch")
            selected_name = st.selectbox(
                "选择一个文件查看内容",
                options=[item["name"] for item in file_items],
                key="kb_read_select",
            )
            try:
                content = service.get_file_content(selected_name)
                st.text_area("文件内容", value=content, height=260, disabled=True)
            except UnicodeDecodeError:
                st.error("该文件不是UTF-8编码，暂不支持预览。")
            except FileNotFoundError:
                st.warning("文件不存在，可能已被删除。")

    with tab_update:
        st.subheader("修改知识库文件")
        file_items = service.list_kb_files()
        if not file_items:
            st.info("暂无可修改的文件。")
        else:
            update_name = st.selectbox(
                "选择要修改的文件",
                options=[item["name"] for item in file_items],
                key="kb_update_select",
            )

            try:
                old_content = service.get_file_content(update_name)
                new_content = st.text_area(
                    "编辑文件内容",
                    value=old_content,
                    height=280,
                    key=f"kb_update_content_{update_name}",
                )
                if st.button("保存修改", type="primary", key="kb_update_btn"):
                    with st.spinner("正在更新文件..."):
                        result = service.update_file(filename=update_name, new_data=new_content)
                    if "[成功]" in result:
                        st.success(result)
                    else:
                        st.error(result)
            except UnicodeDecodeError:
                st.error("该文件不是UTF-8编码，暂不支持在线修改。")
            except FileNotFoundError:
                st.warning("文件不存在，可能已被删除。")

    with tab_delete:
        st.subheader("删除知识库文件")
        file_items = service.list_kb_files()
        if not file_items:
            st.info("暂无可删除的文件。")
        else:
            all_file_names = [item["name"] for item in file_items]
            st.markdown("#### 删除文件（支持单个/批量）")
            delete_batch_names = st.multiselect(
                "选择要删除的文件",
                options=all_file_names,
                key="kb_delete_batch_select",
            )
            confirm_batch = st.checkbox("确认删除所选文件", key="kb_delete_batch_confirm")

            if st.button("执行删除", type="primary", key="kb_delete_batch_btn"):
                if not delete_batch_names:
                    st.warning("请先选择至少一个文件。")
                elif not confirm_batch:
                    st.warning("请先勾选确认后再删除。")
                else:
                    with st.spinner("正在删除文件..."):
                        success_files = []
                        fail_files = []
                        for filename in delete_batch_names:
                            try:
                                result = service.delete_file(filename)
                                if "[成功]" in result:
                                    success_files.append(filename)
                                else:
                                    fail_files.append(f"{filename}：{result}")
                            except Exception as e:
                                fail_files.append(f"{filename}：{e}")

                    if fail_files:
                        summary = f"[失败]，删除完成：成功 {len(success_files)}，失败 {len(fail_files)}"
                        summary += "；失败详情：" + "；".join(fail_files)
                    else:
                        summary = f"[成功]，删除完成：成功 {len(success_files)}，失败 0"

                    st.session_state["kb_action_message"] = summary
                    st.session_state["kb_reset_delete_state"] = True
                    st.rerun()


if __name__ == "__main__":
    render_uploader_page()