"""SQLite + ChromaDB helpers for the Streamlit knowledge-base assistant."""
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_PATH = os.getenv("DB_PATH", "data/assistant.db")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai").strip().lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS")) if os.getenv("EMBED_DIMENSIONS") else None
EMBED_INSTRUCTIONS = os.getenv("EMBED_INSTRUCTIONS", "")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Avoid a network call at import time; the API key is checked when a request is made.
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "missing-api-key")
_http = httpx.Client(timeout=45.0)
_embedding_model = None


def _get_local_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("本地向量模型需要安装 sentence-transformers。") from exc
        _embedding_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _embedding_model


def _require_api_key() -> None:
    if not LLM_API_KEY:
        raise RuntimeError("未配置 OPENAI_API_KEY。请在 .env 或 Streamlit Cloud Secrets 中设置模型 API 密钥。")


def get_embedding(text: str, api_enabled: bool = True) -> list[float]:
    if not text or not text.strip():
        raise ValueError("向量化文本不能为空。")

    # API 关闭时严格使用本地模型，不调用方舟向量 API。
    if not api_enabled or EMBED_PROVIDER == "local":
        return _get_local_embedding_model().encode(text, normalize_embeddings=True).tolist()

    _require_api_key()
    if EMBED_PROVIDER == "ark":
        payload: dict[str, Any] = {
            "model": EMBED_MODEL,
            "input": [{"type": "text", "text": text}],
            "encoding_format": "float",
        }
        if EMBED_DIMENSIONS:
            payload["dimensions"] = EMBED_DIMENSIONS
        if EMBED_INSTRUCTIONS:
            payload["instructions"] = EMBED_INSTRUCTIONS
        response = _http.post(
            LLM_BASE_URL.rstrip("/") + "/embeddings/multimodal",
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        # Support common response shape variants from compatible embedding providers.
        if "data" in body and body["data"] and "embedding" in body["data"][0]:
            return body["data"][0]["embedding"]
        if "data" in body and isinstance(body["data"].get("embedding"), list):
            return body["data"]["embedding"]
        raise RuntimeError("向量接口返回格式不符合预期，请检查模型服务商文档。")

    if EMBED_PROVIDER in {"openai", "api"}:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
            **({"dimensions": EMBED_DIMENSIONS} if EMBED_DIMENSIONS else {}),
        )
        return response.data[0].embedding

    raise ValueError("EMBED_PROVIDER 仅支持 openai、ark 或 local。")


def _connect() -> sqlite3.Connection:
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS qa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL UNIQUE,
                answer TEXT NOT NULL,
                source TEXT,
                created_at TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS unanswered (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )



def get_api_enabled() -> bool:
    """站点级 API 开关，存储于 SQLite，所有访客共用。"""
    with _connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = conn.execute("SELECT value FROM app_settings WHERE key='api_enabled'").fetchone()
        if row is None:
            conn.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('api_enabled', '1')")
            return True
        return row["value"] == "1"


def set_api_enabled(enabled: bool) -> None:
    with _connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO app_settings (key, value) VALUES ('api_enabled', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if enabled else "0",),
        )


def _get_collection(name: str):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


api_collection = _get_collection("qa_collection")
local_collection = _get_collection("qa_local_collection")


def _sync_local_index() -> None:
    """从 SQLite 补齐本地向量索引，确保关闭 API 后也能检索原知识库。"""
    with _connect() as conn:
        rows = conn.execute("SELECT id, question, answer FROM qa ORDER BY id").fetchall()
    existing = set(local_collection.get(include=[])["ids"])
    for row in rows:
        qid = str(row["id"])
        if qid not in existing:
            vector = get_embedding(row["question"], api_enabled=False)
            local_collection.upsert(
                ids=[qid], embeddings=[vector], documents=[row["question"]],
                metadatas=[{"answer": row["answer"]}],
            )


def _upsert_vector(qid: int | str, question: str, answer: str, api_enabled: bool = True) -> None:
    vector = get_embedding(question, api_enabled=api_enabled)
    target = api_collection if api_enabled else local_collection
    target.upsert(ids=[str(qid)], embeddings=[vector], documents=[question], metadatas=[{"answer": answer}])


def add_qa(question: str, answer: str, source: str = "manual", api_enabled: bool = True) -> int:
    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        raise ValueError("问题和答案不能为空。")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        existing = conn.execute("SELECT id FROM qa WHERE question = ?", (question,)).fetchone()
        if existing:
            raise ValueError("该问题已存在于知识库中。")
        cursor = conn.execute(
            "INSERT INTO qa (question, answer, source, created_at) VALUES (?, ?, ?, ?)",
            (question, answer, source, now),
        )
        qid = int(cursor.lastrowid)
    try:
        _upsert_vector(qid, question, answer, api_enabled=api_enabled)
    except Exception:
        # Keep SQLite and Chroma in sync if embedding/vector storage fails.
        with _connect() as conn:
            conn.execute("DELETE FROM qa WHERE id = ?", (qid,))
        raise
    return qid


def get_all_qa() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, answer, source, created_at FROM qa ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_qa(qid: int | str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM qa WHERE id = ?", (qid,))
    try:
        collection.delete(ids=[str(qid)])
    except Exception:
        # A missing vector can occur after an interrupted import; the SQL deletion still stands.
        pass


def search_similar(query: str, top_k: int = 3, api_enabled: bool = True, threshold: float | None = None) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []
    target = api_collection if api_enabled else local_collection
    if not api_enabled:
        _sync_local_index()
    total = target.count()
    if total == 0:
        return []
    embedding = get_embedding(query.strip(), api_enabled=api_enabled)
    result = target.query(
        query_embeddings=[embedding], n_results=min(max(1, top_k), total),
        include=["distances", "documents", "metadatas"],
    )
    output = []
    ids = (result.get("ids") or [[]])[0]
    for idx, qid in enumerate(ids):
        metadata = (result.get("metadatas") or [[]])[0][idx] or {}
        document = (result.get("documents") or [[]])[0][idx] or ""
        distance = (result.get("distances") or [[]])[0][idx]
        score = 1.0 - float(distance)
        if score >= (THRESHOLD if threshold is None else threshold):
            output.append({"id": qid, "question": document, "answer": metadata.get("answer", ""), "score": score})
    return output


def generate_answer(question: str, ctx_list: list[dict[str, Any]] | None, allow_free: bool = False, api_enabled: bool = True) -> str:
    if ctx_list:
        context_text = "\n\n".join(f"问题：{item['question']}\n答案：{item['answer']}" for item in ctx_list)
        prompt = (
            "你是赛事知识库助手。请优先依据下列知识库内容回答；若内容不足，请明确说明，不要编造赛事政策。\n\n"
            f"参考知识库：\n{context_text}\n\n用户问题：{question}"
        )
    elif allow_free:
        prompt = (
            "你是赛事问答助手。请谨慎回答；涉及赛事时间、资格、费用、平台政策等可能变化的信息，"
            "如果没有可靠依据，请明确提示用户以官方通知为准，不要声称已经联网检索。\n\n"
            f"用户问题：{question}"
        )
    else:
        return "知识库中没有相关内容。"

    if not api_enabled:
        return "知识库未命中；当前 API 已关闭，问题应交由管理员处理。"
    _require_api_key()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "暂时无法生成回答，请稍后重试。"


def record_unanswered(question: str) -> None:
    """加入管理员待办；同一问题已有待处理记录时不重复创建。"""
    question = question.strip()
    if not question:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        exists = conn.execute(
            "SELECT id FROM unanswered WHERE question = ? AND status = 'pending' LIMIT 1",
            (question,),
        ).fetchone()
        if exists:
            return
        conn.execute(
            "INSERT INTO unanswered (question, status, created_at) VALUES (?, ?, ?)",
            (question, "pending", now),
        )


def get_unanswered(status: str = "pending") -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, created_at FROM unanswered WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    return [dict(row) for row in rows]


def resolve_unanswered(uid: int | str, answer: str, api_enabled: bool = True) -> None:
    with _connect() as conn:
        row = conn.execute("SELECT question FROM unanswered WHERE id = ?", (uid,)).fetchone()
        if not row:
            raise ValueError("待处理问题不存在。")
        question = row["question"]
    add_qa(question, answer, source="unanswered_review", api_enabled=api_enabled)
    with _connect() as conn:
        conn.execute("UPDATE unanswered SET status = 'resolved' WHERE id = ?", (uid,))


def ignore_unanswered(uid: int | str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE unanswered SET status = 'ignored' WHERE id = ?", (uid,))


def import_records(json_list: list[dict[str, Any]], api_enabled: bool = True) -> int:
    count = 0
    for item in json_list:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue
        try:
            add_qa(question, answer, source="json_import", api_enabled=api_enabled)
            count += 1
        except ValueError as exc:
            if "已存在" not in str(exc):
                raise
    return count


def load_json_to_kb(json_path: str, api_enabled: bool = True) -> int:
    """Import JSON question-answer pairs, skipping existing questions."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层格式必须是问答数组。")
    return import_records(data, api_enabled=api_enabled)
