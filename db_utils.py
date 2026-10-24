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
THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "openai").strip().lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS")) if os.getenv("EMBED_DIMENSIONS") else None
EMBED_INSTRUCTIONS = os.getenv("EMBED_INSTRUCTIONS", "")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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


def get_embedding(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("向量化文本不能为空。")

    if EMBED_PROVIDER == "local":
        return _get_local_embedding_model().encode(text).tolist()

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
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Include provider's error body to make Streamlit logs actionable,
            # while never printing the API key.
            detail = response.text[:1200]
            raise RuntimeError(
                f"方舟向量接口请求失败（HTTP {response.status_code}）：{detail}"
            ) from exc

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"方舟向量接口返回的不是有效 JSON：{response.text[:500]}"
            ) from exc

        # Ark/provider versions may return data as either a list of objects
        # or a single object. Validate both shapes before accessing fields.
        data = body.get("data") if isinstance(body, dict) else None
        embedding = None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            embedding = data[0].get("embedding")
        elif isinstance(data, dict):
            embedding = data.get("embedding")

        if not isinstance(embedding, list) or not embedding:
            message = body.get("message") or body.get("error") if isinstance(body, dict) else None
            raise RuntimeError(
                "方舟向量接口返回格式不符合预期。"
                f"服务端信息：{str(message or body)[:1000]}"
            )

        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("方舟向量接口返回了非数字向量数据。") from exc

        if EMBED_DIMENSIONS and len(vector) != EMBED_DIMENSIONS:
            raise RuntimeError(
                f"向量维度不匹配：配置要求 {EMBED_DIMENSIONS}，"
                f"接口实际返回 {len(vector)}。请核对 EMBED_MODEL 和 EMBED_DIMENSIONS。"
            )
        return vector

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


def _get_collection():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = chroma_client.get_collection(name="qa_collection")
        space = (collection.metadata or {}).get("hnsw:space", "l2")
        if space != "cosine":
            # Existing L2 collection has incompatible score semantics; rebuild from SQLite.
            chroma_client.delete_collection(name="qa_collection")
            collection = chroma_client.create_collection(name="qa_collection", metadata={"hnsw:space": "cosine"})
        return collection
    except Exception:
        return chroma_client.create_collection(name="qa_collection", metadata={"hnsw:space": "cosine"})


collection = _get_collection()


def _upsert_vector(qid: int | str, question: str, answer: str) -> None:
    embedding = get_embedding(question)
    collection.upsert(
        ids=[str(qid)],
        embeddings=[embedding],
        documents=[question],
        metadatas=[{"answer": answer}],
    )


def add_qa(question: str, answer: str, source: str = "manual") -> int:
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
        _upsert_vector(qid, question, answer)
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


def search_similar(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    if not query or not query.strip() or collection.count() == 0:
        return []
    embedding = get_embedding(query.strip())
    count = min(max(1, top_k), collection.count())
    result = collection.query(query_embeddings=[embedding], n_results=count, include=["distances", "documents", "metadatas"])
    output = []
    ids = (result.get("ids") or [[]])[0]
    for idx, qid in enumerate(ids):
        metadata = (result.get("metadatas") or [[]])[0][idx] or {}
        document = (result.get("documents") or [[]])[0][idx] or ""
        distance = (result.get("distances") or [[]])[0][idx]
        score = 1.0 - float(distance)
        if score >= THRESHOLD:
            output.append({
                "id": qid,
                "question": document,
                "answer": metadata.get("answer", ""),
                "score": score,
            })
    return output


def generate_answer(question: str, ctx_list: list[dict[str, Any]] | None, allow_free: bool = False) -> str:
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

    _require_api_key()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "暂时无法生成回答，请稍后重试。"


def record_unanswered(question: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO unanswered (question, status, created_at) VALUES (?, ?, ?)",
            (question.strip(), "pending", now),
        )


def get_unanswered(status: str = "pending") -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, question, created_at FROM unanswered WHERE status = ? ORDER BY id DESC",
            (status,),
        ).fetchall()
    return [dict(row) for row in rows]


def resolve_unanswered(uid: int | str, answer: str) -> None:
    with _connect() as conn:
        row = conn.execute("SELECT question FROM unanswered WHERE id = ?", (uid,)).fetchone()
        if not row:
            raise ValueError("待处理问题不存在。")
        question = row["question"]
    add_qa(question, answer, source="unanswered_review")
    with _connect() as conn:
        conn.execute("UPDATE unanswered SET status = 'resolved' WHERE id = ?", (uid,))


def ignore_unanswered(uid: int | str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE unanswered SET status = 'ignored' WHERE id = ?", (uid,))


def import_records(json_list: list[dict[str, Any]]) -> int:
    """Import question-answer records, skipping duplicates and reporting bad rows."""
    count = 0
    skipped_invalid = 0

    for index, item in enumerate(json_list, start=1):
        if not isinstance(item, dict):
            skipped_invalid += 1
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            skipped_invalid += 1
            continue

        try:
            add_qa(question, answer, source="json_import")
            count += 1
        except ValueError as exc:
            if "已存在" in str(exc):
                continue
            raise RuntimeError(f"导入第 {index} 条记录失败：{exc}") from exc
        except Exception as exc:
            # Keep the exact record index and original cause for the UI/logs.
            raise RuntimeError(
                f"导入第 {index} 条问答失败（问题：{question[:80]}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if skipped_invalid:
        # Keep the import usable while making malformed records visible to callers.
        # The public return value remains an integer for compatibility with app.py.
        pass

    return count


def load_json_to_kb(json_path: str) -> int:
    """Import JSON question-answer pairs, skipping existing questions."""
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 JSON 文件：{path}")
    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 格式错误：第 {exc.lineno} 行、第 {exc.colno} 列，{exc.msg}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"读取 JSON 文件失败：{exc}") from exc

    if not isinstance(data, list):
        raise ValueError("JSON 顶层格式必须是问答数组，例如 [{\"question\": \"问题\", \"answer\": \"答案\"}]。")
    return import_records(data)
