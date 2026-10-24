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
        "ADMIN_PASSWORD",
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
    get_unanswered,
    ignore_unanswered,
    init_db,
    load_json_to_kb,
    record_unanswered,
    resolve_unanswered,
    search_similar,
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
    """优先返回超过相似度阈值的知识库答案；未命中时记录待处理问题。"""
    try:
        matches = search_similar(question)
        if matches:
            best = matches[0]
            st.markdown("### 回答")
            st.markdown(best["answer"])
            st.caption(f"知识库匹配度：{best['score']:.0%}")
            with st.expander("查看匹配到的知识库问题"):
                st.write(best["question"])
            return

        if student:
            record_unanswered(question)
        with st.spinner("正在生成临时回答…"):
            answer = generate_answer(question, [], allow_free=True)
        st.markdown("### 临时回答")
        st.markdown(answer)
        st.info("该回答尚未经过管理员确认，请以赛事群和官方通知为准。问题已提交管理员处理。" if student else "这是临时回答，尚未加入知识库。")
    except Exception as exc:
        st.error(f"处理问题时发生错误：{exc}")
        st.caption("请检查 Streamlit Secrets 中的模型 API、向量模型和网络配置。")


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
            answer_question(question.strip(), student=True)
        else:
            st.warning("请先输入问题。")
    st.sidebar.info("学生端：仅可提问和查看答案")
    if st.sidebar.button("退出并返回首页"):
        logout()

# 管理员后台
elif st.session_state.role == "admin":
    st.title("🔐 管理员后台")
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
                                resolve_unanswered(item["id"], new_answer.strip())
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
        st.subheader("新增知识库条目")
        with st.form("add_qa_form", clear_on_submit=True):
            question_add = st.text_input("问题")
            answer_add = st.text_area("答案", height=120)
            add_clicked = st.form_submit_button("添加到知识库", type="primary")
        if add_clicked:
            if question_add.strip() and answer_add.strip():
                try:
                    add_qa(question_add.strip(), answer_add.strip(), source="manual")
                    st.success("添加成功")
                    st.rerun()
                except Exception as exc:
                    st.error(f"添加失败：{exc}")
            else:
                st.warning("问题和答案均不能为空。")

        st.divider()
        st.subheader("批量导入示例题库")
        if st.button("📥 一键导入 sample_qa.json"):
            if not SAMPLE_JSON.exists():
                st.error(f"找不到文件：{SAMPLE_JSON.name}。请确认该文件已和 app.py 一起部署。")
            else:
                try:
                    count = load_json_to_kb(str(SAMPLE_JSON))
                    st.success(f"导入完成！新增 {count} 条问答，重复的问题会自动跳过。")
                    st.rerun()
                except Exception as exc:
                    st.error(f"导入失败：{exc}")

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
