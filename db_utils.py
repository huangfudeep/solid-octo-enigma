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
