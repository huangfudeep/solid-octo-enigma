from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="知识库 AI 助手", page_icon="🎓", layout="wide")

load_dotenv()

# Streamlit Cloud Secrets are not automatically exported as environment variables.
# Populate os.environ before importing db_utils, which reads its configuration at import time.
try:
    for _key in (
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "CHAT_MODEL", "EMBED_PROVIDER",
        "EMBED_MODEL", "EMBED_DIMENSIONS", "EMBED_INSTRUCTIONS",
        "LOCAL_EMBED_MODEL", "SIMILARITY_THRESHOLD", "DB_PATH", "CHROMA_DIR",
        "SUPABASE_DB_URL", "ADMIN_PASSWORD",
    ):
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    # Local run without a Streamlit secrets file is expected.
    pass

from db_utils import (
    THRESHOLD,
    add_qa,
    delete_qa,
    generate_answer,
    get_all_qa,
    get_api_enabled,
    get_unanswered,
    get_all_unanswered,
    ignore_unanswered,
    init_db,
    load_json_to_kb,
    record_unanswered,
    resolve_unanswered,
    search_similar,
    set_api_enabled,
)

# Streamlit Cloud Secrets 优先，其次读取环境变量/.env。
def get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)

ADMIN_PASSWORD = get_setting("ADMIN_PASSWORD")
SAMPLE_JSON = Path(__file__).with_name("sample_qa.json")

try:
    init_db()
except Exception as exc:
    st.error(f"数据库初始化失败：{exc}")
    st.stop()

if "role" not in st.session_state:
    st.session_state.role = None


def logout() -> None:
    st.session_state.role = None
    st.rerun()


def answer_question(question: str, *, student: bool) -> None:
    """学生优先展示最高相似度答案；未达阈值时提交管理员待办。"""
    api_enabled = get_api_enabled()
    try:
        # 学生端阈值固定为 0.60；管理员测试和 JSON 导入仍使用原全局阈值。
        matches = search_similar(
            question, top_k=1 if student else 5,
            api_enabled=api_enabled, threshold=0.6 if student else None,
        )
        if matches:
            best = matches[0]
            st.markdown("### 知识库回答")
            st.markdown(best["answer"])
            st.caption(f"匹配问题：{best['question']} · 相似度：{best['score']:.0%}")
            # 按钮由学生端在提交后持续渲染；若放在表单提交分支内，点击按钮触发 rerun 后会消失。
            if student:
                st.session_state["last_matched_student_question"] = question
            return

        if student:
            # 未达到学生端阈值时自动加入管理员待办；关闭 API 时不展示低相似度候选答案。
            record_unanswered(question)
            if api_enabled:
                with st.spinner("正在生成临时回答…"):
                    answer = generate_answer(question, [], allow_free=True, api_enabled=True)
                st.markdown("### 临时回答")
                st.markdown(answer)
                st.warning("临时回答，管理员待确认。问题已自动加入管理员待办。")
            else:
                st.warning("暂无答案匹配。")
                st.info("问题已自动加入管理员待办，等待管理员补充正式答案。")
            return

        # 管理员测试端保留原来的行为。
        if api_enabled:
            with st.spinner("正在生成临时回答…"):
                answer = generate_answer(question, [], allow_free=True, api_enabled=True)
            st.markdown("### 临时回答")
            st.markdown(answer)
        else:
            st.warning("知识库中暂时没有找到合适答案。当前已关闭方舟 API。")
    except Exception as exc:
        st.error(f"处理问题时发生错误：{exc}")
        st.caption("若已关闭 API，请检查本地向量模型依赖；开启 API 时请检查 Secrets 配置。")


# 身份选择
if st.session_state.role is None:
    st.title("🎓 知识库 AI 助手")
    st.write("面向高校助农主播赛事的智能问答与知识库管理系统")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("学生端")
            st.write("查询赛事规则、集训安排、账号开通和直播操作等问题。")
            if st.button("进入学生端", type="primary", use_container_width=True):
                st.session_state.role = "student"
                st.rerun()
    with right:
        with st.container(border=True):
            st.subheader("管理员端")
            st.write("处理待回答问题、维护问答知识库、测试检索效果。")
            if st.button("管理员登录", use_container_width=True):
                st.session_state.role = "admin_login"
                st.rerun()
    st.caption("提示：首次部署后，请管理员登录并导入 sample_qa.json 示例题库。")

# 管理员登录
elif st.session_state.role == "admin_login":
    st.title("🔐 管理员登录")
    if not ADMIN_PASSWORD:
        st.warning("尚未设置管理员密码。请在项目根目录 .env 或 Streamlit Cloud 的 Secrets 中设置 ADMIN_PASSWORD。")
    with st.form("admin_login_form"):
        password = st.text_input("管理员密码", type="password")
        submitted = st.form_submit_button("登录", type="primary")
    if submitted:
        if ADMIN_PASSWORD and password == ADMIN_PASSWORD:
            st.session_state.role = "admin"
            st.rerun()
        else:
            st.error("密码错误或管理员密码尚未配置。")
    if st.button("返回首页"):
        logout()

# 学生端
elif st.session_state.role == "student":
    st.title("🎓 学生问答端")
    st.write("请输入你想了解的赛事问题。")
    with st.form("student_question_form", clear_on_submit=False):
        question = st.text_area("问题", placeholder="例如：快手 0 粉丝可以参赛吗？", height=100)
        submitted = st.form_submit_button("提交提问", type="primary")
    if submitted:
        if question.strip():
            st.session_state["last_student_question"] = question.strip()
            st.session_state.pop("last_matched_student_question", None)
            answer_question(question.strip(), student=True)
        else:
            st.warning("请先输入问题。")
    elif st.session_state.get("last_student_question"):
        # 维持最近一次结果和反馈按钮，避免 Streamlit rerun 后按钮消失。
        answer_question(st.session_state["last_student_question"], student=True)

    matched_question = st.session_state.get("last_matched_student_question")
    if matched_question:
        if st.button("不是我想要的答案", key="not_answer_feedback"):
            record_unanswered(matched_question)
            st.session_state["last_matched_student_question"] = None
            st.success("已将你的问题提交给管理员待办，管理员确认后会补充正式答案。")

    st.sidebar.info("学生端：仅可提问和查看答案")
    if st.sidebar.button("退出并返回首页"):
        logout()

# 管理员后台
elif st.session_state.role == "admin":
    st.title("🔐 管理员后台")
    current_api_enabled = get_api_enabled()
    api_enabled_ui = st.toggle(
        "启用方舟 API（向量检索 + 大模型临时回答）",
        value=current_api_enabled,
        key="admin_api_toggle",
        help="关闭后所有访客改用本地向量模型；未命中只提交管理员，不调用方舟 API。",
    )
    if api_enabled_ui != current_api_enabled:
        set_api_enabled(api_enabled_ui)
        st.success("已开启方舟 API。" if api_enabled_ui else "已关闭方舟 API，切换到本地检索。")
        st.rerun()
    st.caption("当前模式：" + ("方舟 API 已开启" if current_api_enabled else "本地检索模式（不调用方舟 API）"))

    tab_test, tab_pending, tab_kb = st.tabs(["问答测试", "待回答问题处理", "知识库管理"])

    with tab_test:
        st.subheader("问答检索测试")
        with st.form("admin_test_form"):
            test_question = st.text_input("输入测试问题")
            test_submitted = st.form_submit_button("测试检索", type="primary")
        if test_submitted:
            if test_question.strip():
                answer_question(test_question.strip(), student=False)
            else:
                st.warning("请先输入测试问题。")
        st.caption(f"当前相似度阈值：{THRESHOLD:.2f}。低于阈值的问题会视为未命中。")

    with tab_pending:
        st.subheader("学生提问、尚未入库的问题")
        pending_list = get_unanswered(status="pending")
        if not pending_list:
            st.success("暂无待处理问题")
        else:
            for item in pending_list:
                with st.expander(f"【待处理】{item['question']}"):
                    st.caption(f"提问时间：{item['created_at']}")
                    with st.form(f"resolve_form_{item['id']}"):
                        new_answer = st.text_area("填写正式答案，保存后将加入知识库", key=f"answer_{item['id']}", height=120)
                        save_clicked = st.form_submit_button("✅ 保存答案并入库", type="primary")
                    if save_clicked:
                        if new_answer.strip():
                            try:
                                resolve_unanswered(item["id"], new_answer.strip(), api_enabled=get_api_enabled())
                                st.success("已保存到知识库！")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"保存失败：{exc}")
                        else:
                            st.warning("答案不能为空。")
                    if st.button("❌ 忽略该问题", key=f"ignore_{item['id']}"):
                        ignore_unanswered(item["id"])
                        st.rerun()

    with tab_kb:
        st.subheader("📦 数据备份与导出")
        import json
        from datetime import datetime, timezone
        backup_payload = {
            "backup_format": "streamlit_qa_backup_v1",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "knowledge_base": get_all_qa(),
            "unanswered_questions": get_all_unanswered(),
            "settings": {"api_enabled": get_api_enabled()},
        }
        st.download_button(
            "⬇️ 一键导出 JSON 完整备份",
            data=json.dumps(backup_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"qa_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="备份包含知识库问答、待处理问题记录和 API 开关状态。API 密钥及管理员密码不会被导出。",
        )
        st.caption("建议定期下载并妥善保存备份文件；导出的 JSON 可用于恢复知识库问答。")
        st.divider()
        st.subheader("新增知识库条目")
        with st.form("add_qa_form", clear_on_submit=True):
            question_add = st.text_input("问题")
            answer_add = st.text_area("答案", height=120)
            add_clicked = st.form_submit_button("添加到知识库", type="primary")
        if add_clicked:
            if question_add.strip() and answer_add.strip():
                try:
                    add_qa(question_add.strip(), answer_add.strip(), source="manual", api_enabled=get_api_enabled())
                    st.success("添加成功")
                    st.rerun()
                except Exception as exc:
                    st.error(f"添加失败：{exc}")
            else:
                st.warning("问题和答案均不能为空。")

        st.divider()
        st.subheader("📂 上传本地 JSON 问答库")
        st.caption("支持拖拽或选择 .json 文件。系统会自动跳过问答完全一致的重复项，并列出相似问题供管理员确认。")
        uploaded_json = st.file_uploader("拖动 JSON 文件到此处，或点击选择文件", type=["json"], key="json_kb_uploader")
        use_model_for_import = st.checkbox(
            "本次导入使用方舟 API 向量检索（关闭则使用本地模型）",
            value=get_api_enabled(),
            help="开启会调用已配置的方舟向量 API；关闭时不调用方舟 API。若本地模型首次运行，会下载模型文件。",
            key="json_import_use_api",
        )
        if uploaded_json is not None:
            import json
            try:
                parsed = json.loads(uploaded_json.getvalue().decode("utf-8-sig"))
                if isinstance(parsed, dict):
                    # Also accept common wrapper formats such as {"data": [...]}.
                    parsed = parsed.get("knowledge_base", parsed.get("data", parsed.get("qa", parsed.get("items", parsed))))
                if not isinstance(parsed, list):
                    st.error("JSON 格式不正确：顶层必须是问答数组，或包含 data/qa/items 数组的对象。")
                else:
                    clean_records = []
                    for idx, row in enumerate(parsed, start=1):
                        if not isinstance(row, dict):
                            continue
                        q = str(row.get("question", row.get("问题", ""))).strip()
                        a = str(row.get("answer", row.get("答案", ""))).strip()
                        if q and a:
                            clean_records.append({"question": q, "answer": a, "row_num": idx})
                    if not clean_records:
                        st.warning("文件中没有识别到有效的 question/answer（或 问题/答案）字段。")
                    else:
                        existing = get_all_qa()
                        exact_pairs = {(x["question"].strip(), x["answer"].strip()) for x in existing}
                        existing_questions = {x["question"].strip() for x in existing}
                        new_records = []
                        exact_skipped = 0
                        for record in clean_records:
                            pair = (record["question"], record["answer"])
                            if pair in exact_pairs:
                                exact_skipped += 1
                                continue
                            # Duplicate pairs inside the uploaded file are also skipped.
                            if pair in {(x["question"], x["answer"]) for x in new_records}:
                                exact_skipped += 1
                                continue
                            new_records.append(record)
                        st.info(f"识别到 {len(clean_records)} 条有效问答；完全一致的重复项将自动跳过：{exact_skipped} 条。")
                        if new_records:
                            st.write("**相似项检查（相似问题会列出已有答案，导入前请确认）**")
                            reviewed = []
                            for i, record in enumerate(new_records):
                                q, a = record["question"], record["answer"]
                                same_question = next((x for x in existing if x["question"].strip() == q), None)
                                try:
                                    similar = search_similar(q, top_k=3, api_enabled=use_model_for_import)
                                except Exception as exc:
                                    similar = []
                                    st.warning(f"第 {record['row_num']} 条相似检索失败：{exc}")
                                # Exact question + exact answer is already skipped; same question but different answer requires human review.
                                with st.expander(f"第 {record['row_num']} 条：{q}", expanded=bool(same_question or similar)):
                                    st.markdown("**上传文件中的答案**")
                                    st.write(a)
                                    if same_question:
                                        st.warning("知识库已有完全相同的问题，但答案不同，请重点确认。")
                                        st.markdown("**本地库已有答案**")
                                        st.write(same_question["answer"])
                                    else:
                                        candidates = [m for m in similar if m.get("question", "").strip() != q]
                                        if candidates:
                                            st.markdown("**本地库相似问题与已有回答（供确认）**")
                                            for m in candidates:
                                                st.write(f"相似度 {m.get('score', 0):.0%} · {m['question']}")
                                                st.write(m["answer"])
                                        else:
                                            st.caption("未检索到明显相似项。")
                                    checked = st.checkbox("确认导入此条", value=not bool(same_question), key=f"json_import_{uploaded_json.name}_{i}")
                                    reviewed.append((record, checked, same_question))
                            if st.button("✅ 导入已确认问答", type="primary", key="confirm_json_import"):
                                imported, skipped_conflict, errors = 0, 0, []
                                for record, checked, same_question in reviewed:
                                    if not checked:
                                        skipped_conflict += 1
                                        continue
                                    try:
                                        # Existing question with different answer must not silently overwrite the knowledge base.
                                        add_qa(record["question"], record["answer"], source=f"json_import:{uploaded_json.name}", api_enabled=use_model_for_import)
                                        imported += 1
                                    except Exception as exc:
                                        errors.append(f"{record['question']}：{exc}")
                                st.success(f"导入完成：新增 {imported} 条；未勾选/待确认 {skipped_conflict} 条；自动跳过完全重复 {exact_skipped} 条。")
                                if errors:
                                    st.warning("以下项目未能导入：\n" + "\n".join(errors[:10]))
                                st.rerun()
                        else:
                            st.success("文件中的问答都已存在且完全一致，无需重复导入。")
            except Exception as exc:
                st.error(f"无法读取 JSON：{exc}")

        st.subheader("现有知识库")
        all_items = get_all_qa()
        st.metric("知识库问答数量", len(all_items))
        if not all_items:
            st.info("知识库暂时为空，请导入示例题库或手动添加条目。")
        for item in all_items:
            with st.expander(f"{item['id']}. {item['question']}"):
                st.write(item["answer"])
                st.caption(f"来源：{item['source'] or '未知'} · 创建时间：{item['created_at']}")
                if st.button("删除此条目", key=f"delete_{item['id']}"):
                    try:
                        delete_qa(item["id"])
                        st.success("已删除")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"删除失败：{exc}")

    st.sidebar.info("管理员端：处理待回答问题、维护知识库、批量导入题库")
    if st.sidebar.button("退出登录"):
        logout()
