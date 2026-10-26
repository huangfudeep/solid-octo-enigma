"""Supabase PostgreSQL + persistent embedding storage helpers for Streamlit KB."""
import json
import math
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
import psycopg2
import streamlit as st
from psycopg2.extras import Json, RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai").strip().lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS")) if os.getenv("EMBED_DIMENSIONS") else None
EMBED_INSTRUCTIONS = os.getenv("EMBED_INSTRUCTIONS", "")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "missing-api-key")
_http = httpx.Client(timeout=45.0)


def _connect():
    if not DB_URL:
        raise RuntimeError(
            "未配置 SUPABASE_DB_URL。请在 Streamlit Cloud → Settings → Secrets 中添加 Supabase PostgreSQL 连接字符串。"
        )
    return psycopg2.connect(DB_URL, connect_timeout=15, cursor_factory=RealDictCursor, sslmode="require")


@contextmanager
def _db():
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


@st.cache_resource(show_spinner="正在加载本地语义模型…")
def _get_local_embedding_model(model_name: str):
    """Cache one model process-wide across Streamlit reruns/sessions."""
    try:
        # Limit CPU thread pools on small Streamlit Cloud instances.
        import torch
        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # PyTorch may already have initialized its inter-op pool.
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("本地向量模型需要安装 sentence-transformers。") from exc
    return SentenceTransformer(model_name, device="cpu")


def _require_api_key() -> None:
    if not LLM_API_KEY:
        raise RuntimeError("未配置 OPENAI_API_KEY。请在 Streamlit Cloud Secrets 中设置模型 API 密钥。")


def get_embedding(text: str, api_enabled: bool = True) -> list[float]:
    if not text or not text.strip():
        raise ValueError("向量化文本不能为空。")
    if not api_enabled or EMBED_PROVIDER == "local":
        model = _get_local_embedding_model(LOCAL_EMBED_MODEL)
        vector = model.encode(
            text, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=1,
        )
        return vector.tolist()
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
        if body.get("data") and isinstance(body["data"], list) and "embedding" in body["data"][0]:
            return body["data"][0]["embedding"]
        if isinstance(body.get("data", {}).get("embedding"), list):
            return body["data"]["embedding"]
        raise RuntimeError("向量接口返回格式不符合预期，请检查模型服务商文档。")
    if EMBED_PROVIDER in {"openai", "api"}:
        response = client.embeddings.create(
            model=EMBED_MODEL, input=text,
            **({"dimensions": EMBED_DIMENSIONS} if EMBED_DIMENSIONS else {}),
        )
        return response.data[0].embedding
    raise ValueError("EMBED_PROVIDER 仅支持 openai、ark 或 local。")


def init_db() -> None:
    """Create tables if missing. Never drops or truncates existing data."""
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa (
                    id BIGSERIAL PRIMARY KEY,
                    question TEXT NOT NULL UNIQUE,
                    answer TEXT NOT NULL,
                    source TEXT,
                    created_at TEXT,
                    embedding_api JSONB,
                    embedding_local JSONB
                )
            """)
            cur.execute("ALTER TABLE qa ADD COLUMN IF NOT EXISTS embedding_api JSONB")
            cur.execute("ALTER TABLE qa ADD COLUMN IF NOT EXISTS embedding_local JSONB")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS unanswered (
                    id BIGSERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)


def get_api_enabled() -> bool:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO app_settings (key, value) VALUES ('api_enabled', '1') ON CONFLICT (key) DO NOTHING")
            cur.execute("SELECT value FROM app_settings WHERE key = 'api_enabled'")
            row = cur.fetchone()
    return bool(row and row["value"] == "1")


def set_api_enabled(enabled: bool) -> None:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO app_settings (key, value) VALUES ('api_enabled', %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                ("1" if enabled else "0",),
            )


def _embedding_column(api_enabled: bool) -> str:
    return "embedding_api" if api_enabled and EMBED_PROVIDER != "local" else "embedding_local"


def _ensure_embedding(row: dict[str, Any], api_enabled: bool) -> list[float]:
    col = _embedding_column(api_enabled)
    vector = row.get(col)
    if vector:
        return [float(v) for v in vector]
    vector = get_embedding(row["question"], api_enabled=api_enabled)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE qa SET {col} = %s WHERE id = %s AND {col} IS NULL", (Json(vector), row["id"]))
    return vector


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def add_qa(question: str, answer: str, source: str = "manual", api_enabled: bool = True) -> int:
    question, answer = question.strip(), answer.strip()
    if not question or not answer:
        raise ValueError("问题和答案不能为空。")
    vector = get_embedding(question, api_enabled=api_enabled)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    col = _embedding_column(api_enabled)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM qa WHERE question = %s", (question,))
            if cur.fetchone():
                raise ValueError("该问题已存在于知识库中。")
            if col == "embedding_api":
                cur.execute(
                    "INSERT INTO qa (question, answer, source, created_at, embedding_api) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (question, answer, source, now, Json(vector)),
                )
            else:
                cur.execute(
                    "INSERT INTO qa (question, answer, source, created_at, embedding_local) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (question, answer, source, now, Json(vector)),
                )
            return int(cur.fetchone()["id"])


def get_all_qa() -> list[dict[str, Any]]:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer, source, created_at FROM qa ORDER BY id DESC")
            return [dict(row) for row in cur.fetchall()]


def delete_qa(qid: int | str) -> None:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM qa WHERE id = %s", (qid,))


def search_similar(query: str, top_k: int = 3, api_enabled: bool = True, threshold: float | None = None) -> list[dict[str, Any]]:
    """Bounded-memory cosine search. Fetch only the active embedding column in small batches.

    For local mode, missing vectors are encoded in small batches and persisted, avoiding one
    model inference per row. Results preserve the existing threshold/top-k behavior.
    """
    if not query or not query.strip():
        return []

    query_vector = get_embedding(query.strip(), api_enabled=api_enabled)
    cutoff = THRESHOLD if threshold is None else threshold
    batch_size = max(25, min(100, int(os.getenv("SEARCH_BATCH_SIZE", "50"))))
    top_n = max(1, int(top_k))
    col = _embedding_column(api_enabled)
    best: list[dict[str, Any]] = []
    offset = 0

    while True:
        # Do not fetch both API and local vectors: JSONB vectors add significant transient memory.
        with _db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, question, answer, {col} AS embedding FROM qa ORDER BY id LIMIT %s OFFSET %s",
                    (batch_size, offset),
                )
                rows = [dict(row) for row in cur.fetchall()]
        if not rows:
            break

        missing = [row for row in rows if not row.get("embedding")]
        if missing and col == "embedding_local":
            model = _get_local_embedding_model(LOCAL_EMBED_MODEL)
            vectors = model.encode(
                [row["question"] for row in missing],
                batch_size=8, normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            updates = []
            for row, vector in zip(missing, vectors):
                row["embedding"] = vector.tolist()
                updates.append((Json(row["embedding"]), row["id"]))
            if updates:
                with _db() as conn:
                    with conn.cursor() as cur:
                        cur.executemany(
                            "UPDATE qa SET embedding_local = %s WHERE id = %s AND embedding_local IS NULL",
                            updates,
                        )
        elif missing:
            # API mode only computes missing embeddings; local-model memory remains untouched.
            for row in missing:
                row["embedding"] = get_embedding(row["question"], api_enabled=api_enabled)
                with _db() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"UPDATE qa SET {col} = %s WHERE id = %s AND {col} IS NULL",
                            (Json(row["embedding"]), row["id"]),
                        )

        for row in rows:
            vector = row.get("embedding")
            if not vector:
                continue
            score = _cosine_similarity(query_vector, [float(v) for v in vector])
            if score >= cutoff:
                best.append({
                    "id": str(row["id"]), "question": row["question"],
                    "answer": row["answer"], "score": score,
                })
        best.sort(key=lambda item: item["score"], reverse=True)
        del rows, missing
        best = best[:top_n]
        offset += batch_size

    return best


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
    question = question.strip()
    if not question:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM unanswered WHERE question = %s AND status = 'pending' LIMIT 1", (question,))
            if cur.fetchone():
                return
            cur.execute(
                "INSERT INTO unanswered (question, status, created_at) VALUES (%s, 'pending', %s)",
                (question, now),
            )


def get_unanswered(status: str = "pending") -> list[dict[str, Any]]:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, created_at FROM unanswered WHERE status = %s ORDER BY id DESC", (status,))
            return [dict(row) for row in cur.fetchall()]


def get_all_unanswered() -> list[dict[str, Any]]:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, status, created_at FROM unanswered ORDER BY id DESC")
            return [dict(row) for row in cur.fetchall()]


def resolve_unanswered(uid: int | str, answer: str, api_enabled: bool = True) -> None:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT question FROM unanswered WHERE id = %s", (uid,))
            row = cur.fetchone()
    if not row:
        raise ValueError("待处理问题不存在。")
    add_qa(row["question"], answer, source="unanswered_review", api_enabled=api_enabled)
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE unanswered SET status = 'resolved' WHERE id = %s", (uid,))


def ignore_unanswered(uid: int | str) -> None:
    with _db() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE unanswered SET status = 'ignored' WHERE id = %s", (uid,))


def import_records(json_list: list[dict[str, Any]], api_enabled: bool = True) -> int:
    count = 0
    for item in json_list:
        question = str(item.get("question", item.get("问题", ""))).strip()
        answer = str(item.get("answer", item.get("答案", ""))).strip()
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
    path = __import__("pathlib").Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get("knowledge_base", data.get("data", data.get("qa", data.get("items", data))))
    if not isinstance(data, list):
        raise ValueError("JSON 顶层格式必须是问答数组，或包含 knowledge_base/data/qa/items 数组的对象。")
    return import_records(data, api_enabled=api_enabled)
