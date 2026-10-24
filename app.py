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
        if _key in st.secrets:
            # Streamlit Cloud Secrets 优先级高于本地环境变量/.env
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
