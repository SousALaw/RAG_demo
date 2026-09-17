"""
基于 Streamlit 完成 Web 端文件上传功能（知识库管理页）。

只负责 UI：增删改查全部通过 api_client 调后端 API，
不 import KnowledgeBaseService，也不直接读写 data/、chroma_db/、md5.text。
"""
import streamlit as st

import api_client


def decode_text_file(file_bytes: bytes) -> str:
    """优先使用 UTF-8 解码，失败时回退 GBK。

    这属于浏览器上传输入的编码处理，与 RAG / 知识库业务逻辑无关。
    """
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("gbk")


def _load_files(category: str | None = None) -> list:
    """从 API 取文件列表，失败时提示并返回空列表。"""
    try:
        return api_client.list_files(category=category)
    except Exception as exc:
        st.error(f"请求失败：{exc}")
        return []


def _render_mutation(result: dict) -> None:
    """按 success / skipped / error 三态给出提示。"""
    message = result.get("message", "")
    status = result.get("status", "error")
    if status == "success":
        st.success(message)
    elif status == "skipped":
        st.info(message)
    else:
        st.error(message)


def render_uploader_page() -> None:
    st.title("知识库管理")
    st.caption(f"在此模块中可对知识库文件进行增删改查。后端 API：{api_client.get_base_url()}")

    # 数据源选择：下面四个 Tab 全部只操作选中的分类。
    # 这里不提供「全部」——新增文件必须落到某个具体的分类里。
    try:
        categories = api_client.list_categories()
    except Exception as exc:
        st.error(f"请求失败：{exc}")
        return

    if not categories:
        st.warning("后端没有返回任何数据源分类，请检查 config_data.knowledge_categories。")
        return

    category = st.selectbox(
        "数据源分类",
        options=list(categories.keys()),
        format_func=lambda key: f"{categories[key]}（{key}）",
        key="kb_category",
    )

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
                            except UnicodeDecodeError:
                                fail_count += 1
                                detail_messages.append(f"[失败]，文件{file_item.name}编码无法识别")
                                continue

                            try:
                                result = api_client.add_file(
                                    name=file_item.name, content=text, category=category
                                )
                            except Exception as exc:
                                fail_count += 1
                                detail_messages.append(f"[失败]，文件{file_item.name}：{exc}")
                                continue

                            detail_messages.append(result.get("message", ""))
                            status = result.get("status")
                            if status == "success":
                                success_count += 1
                            elif status == "skipped":
                                skip_count += 1
                            else:
                                fail_count += 1

                    st.success(f"上传完成：成功 {success_count}，跳过 {skip_count}，失败 {fail_count}")
                    with st.expander("查看上传明细"):
                        for msg in detail_messages:
                            st.write(msg)

        with right_col:
            st.markdown(f"### 已上传文件（{categories[category]}）")
            uploaded_files = _load_files(category)
            st.caption(f"当前共 {len(uploaded_files)} 个文件")

            if not uploaded_files:
                st.info("暂无已上传文件")
            else:
                st.dataframe(uploaded_files, width="stretch", height=260)

    with tab_read:
        st.subheader("查询知识库文件")
        keyword = st.text_input("按文件名关键词筛选", key="kb_search_keyword")
        file_items = _load_files(category)

        # 关键词过滤只作用于已取回的列表，属于展示逻辑。
        keyword = (keyword or "").strip().lower()
        if keyword:
            file_items = [item for item in file_items if keyword in item["name"].lower()]

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
                content = api_client.get_file_content(selected_name, category=category)
                st.text_area("文件内容", value=content, height=260, disabled=True)
            except Exception as exc:
                st.error(f"请求失败：{exc}")

    with tab_update:
        st.subheader("修改知识库文件")
        file_items = _load_files(category)
        if not file_items:
            st.info("暂无可修改的文件。")
        else:
            update_name = st.selectbox(
                "选择要修改的文件",
                options=[item["name"] for item in file_items],
                key="kb_update_select",
            )

            old_content = None
            try:
                old_content = api_client.get_file_content(update_name, category=category)
            except Exception as exc:
                st.error(f"请求失败：{exc}")

            if old_content is not None:
                new_content = st.text_area(
                    "编辑文件内容",
                    value=old_content,
                    height=280,
                    key=f"kb_update_content_{update_name}",
                )
                if st.button("保存修改", type="primary", key="kb_update_btn"):
                    with st.spinner("正在更新文件..."):
                        try:
                            result = api_client.update_file(
                                name=update_name, content=new_content, category=category
                            )
                        except Exception as exc:
                            st.error(f"请求失败：{exc}")
                        else:
                            _render_mutation(result)

    with tab_delete:
        st.subheader("删除知识库文件")
        file_items = _load_files(category)
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
                    success_files = []
                    fail_files = []
                    with st.spinner("正在删除文件..."):
                        for filename in delete_batch_names:
                            try:
                                result = api_client.delete_file(filename, category=category)
                            except Exception as exc:
                                fail_files.append(f"{filename}：{exc}")
                                continue

                            if result.get("status") == "success":
                                success_files.append(filename)
                            else:
                                fail_files.append(f"{filename}：{result.get('message', '')}")

                    if fail_files:
                        st.error(
                            f"删除完成：成功 {len(success_files)}，失败 {len(fail_files)}"
                            "；失败详情：" + "；".join(fail_files)
                        )
                    else:
                        st.success(f"删除完成：成功 {len(success_files)}，失败 0")


if __name__ == "__main__":
    render_uploader_page()
