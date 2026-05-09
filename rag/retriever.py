"""rag/retriever.py — pgvector-backed retriever

Primary retriever: PgVectorRetriever
  - Stores chunk embeddings in Supabase via pgvector.
  - fit()     → encode texts + upsert to rag_chunks table
  - retrieve() → cosine similarity search using <=> operator

Fallback chain:
  1. PgVectorRetriever (pgvector on Supabase)
  2. TFIDFFallback     (pure-NumPy, no DB needed)

TFIDFRetriever is kept as an alias for backward compatibility with engine.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import pickle
import threading
from typing import Dict, List, Optional

import numpy as np

from rag.knowledge_builder import Chunk

logger = logging.getLogger(__name__)

_CACHE_DIR = pathlib.Path(".embedding_cache")
_CACHE_DIR.mkdir(exist_ok=True)

_model_lock   = threading.Lock()
_shared_model = None


def _get_model():
    """Lazy-load sentence-transformers model. Thread-safe singleton."""
    global _shared_model
    if _shared_model is not None:
        return _shared_model
    with _model_lock:
        if _shared_model is not None:
            return _shared_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model all-MiniLM-L6-v2 ...")
            _shared_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("Dense embedding model loaded.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — TF-IDF fallback active. "
                "Run: pip install sentence-transformers"
            )
            _shared_model = None
    return _shared_model


def _encode(texts: List[str]) -> Optional[np.ndarray]:
    """Encode a list of texts; uses disk cache to avoid re-encoding."""
    model = _get_model()
    if model is None:
        return None

    cache_key  = hashlib.md5("".join(texts).encode()).hexdigest()
    cache_path = _CACHE_DIR / f"{cache_key}.pkl"

    if cache_path.exists():
        logger.info("Embeddings loaded from cache: %s", cache_key[:8])
        return pickle.loads(cache_path.read_bytes())

    raw = model.encode(
        texts, batch_size=64, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    embs = raw.astype(np.float32)
    cache_path.write_bytes(pickle.dumps(embs))
    logger.info("Embeddings encoded and cached: %s", cache_key[:8])
    return embs


# ── pgvector helpers ──────────────────────────────────────────────────────────

_TABLE = "rag_chunks"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    chunk_id  text    NOT NULL,
    layer     text    NOT NULL,
    text      text    NOT NULL,
    metadata  jsonb   NOT NULL DEFAULT '{{}}',
    embedding vector(384),
    PRIMARY KEY (chunk_id, layer)
);
"""

_table_ensured = False
_table_lock    = threading.Lock()


def _get_conn():
    """Lazy import of DB connection — keeps rag layer free of Streamlit at import time."""
    from core.database import get_connection
    return get_connection()


def _ensure_table() -> bool:
    """Create rag_chunks table if it does not exist. Runs at most once per process."""
    global _table_ensured
    if _table_ensured:
        return True
    with _table_lock:
        if _table_ensured:
            return True
        try:
            with _get_conn() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(_CREATE_TABLE_SQL)
                conn.commit()
            _table_ensured = True
            logger.info("pgvector table ready: %s", _TABLE)
            return True
        except Exception as exc:
            logger.warning("pgvector table setup failed: %s", exc)
            return False


# ── PgVectorRetriever ─────────────────────────────────────────────────────────

class PgVectorRetriever:
    """
    Retriever backed by pgvector on Supabase PostgreSQL.

    Parameters
    ----------
    layer : 'static' or 'dynamic'
        Partitions the rag_chunks table so static and dynamic chunks
        don't overwrite each other on fit().
    """

    def __init__(self, layer: str = "dynamic") -> None:
        self.layer    = layer
        self._chunks:   List[Chunk]            = []
        self._fallback: Optional[TFIDFFallback] = None
        self._ready   = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, chunks: List[Chunk]) -> "PgVectorRetriever":
        self._chunks   = chunks
        self._fallback = None
        self._ready    = False

        if not chunks:
            return self

        embeddings = _encode([c.text for c in chunks])

        if embeddings is None:
            logger.warning("No embedding model — TF-IDF fallback (layer=%s)", self.layer)
            self._fallback = TFIDFFallback().fit(chunks)
            return self

        if not _ensure_table():
            logger.warning("pgvector unavailable — TF-IDF fallback (layer=%s)", self.layer)
            self._fallback = TFIDFFallback().fit(chunks)
            return self

        try:
            self._upsert(chunks, embeddings)
            self._ready = True
            logger.info("pgvector fit: layer=%s, %d chunks", self.layer, len(chunks))
        except Exception as exc:
            logger.error("pgvector upsert failed (%s) — TF-IDF fallback", exc)
            self._fallback = TFIDFFallback().fit(chunks)

        return self

    def retrieve(self, query: str, k: int = 6) -> List[Chunk]:
        if self._fallback is not None:
            return self._fallback.retrieve(query, k)

        if not self._ready or not self._chunks:
            return []

        model = _get_model()
        if model is None:
            return []

        q_vec = model.encode(
            [query], show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )[0].astype(np.float32)

        try:
            return self._query(q_vec, k)
        except Exception as exc:
            logger.error("pgvector query failed (%s)", exc)
            return []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _upsert(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        from psycopg2.extras import execute_values

        rows = [
            (
                c.chunk_id,
                self.layer,
                c.text,
                json.dumps(c.metadata),
                embeddings[i].tolist(),
            )
            for i, c in enumerate(chunks)
        ]

        with _get_conn() as conn:
            if conn is None:
                raise RuntimeError("No DB connection available")
            with conn.cursor() as cur:
                # Replace all chunks for this layer atomically
                cur.execute(f"DELETE FROM {_TABLE} WHERE layer = %s", (self.layer,))
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {_TABLE} (chunk_id, layer, text, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (chunk_id, layer) DO UPDATE
                        SET text      = EXCLUDED.text,
                            metadata  = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                    """,
                    rows,
                )
            conn.commit()

    def _query(self, q_vec: np.ndarray, k: int) -> List[Chunk]:
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in q_vec.tolist()) + "]"
        sql = f"""
            SELECT chunk_id,
                   text,
                   metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM   {_TABLE}
            WHERE  layer = %s
            ORDER  BY embedding <=> %s::vector
            LIMIT  %s
        """
        with _get_conn() as conn:
            if conn is None:
                return []
            with conn.cursor() as cur:
                cur.execute(sql, (vec_literal, self.layer, vec_literal, k))
                rows = cur.fetchall()

        return [
            Chunk(
                chunk_id=row[0],
                text=row[1],
                metadata=row[2] if isinstance(row[2], dict) else {},
                score=float(row[3]),
            )
            for row in rows
        ]


# ── TF-IDF fallback (pure NumPy, no dependencies) ────────────────────────────

class TFIDFFallback:
    """Pure-NumPy TF-IDF retriever — used when pgvector or sentence-transformers
    is unavailable."""

    def __init__(self) -> None:
        self._chunks:  List[Chunk]    = []
        self._vocab:   Dict[str, int] = {}
        self._idf:     np.ndarray     = np.array([])
        self._matrix:  np.ndarray     = np.array([])

    def fit(self, chunks: List[Chunk]) -> "TFIDFFallback":
        self._chunks = chunks
        if not chunks:
            return self

        vocab: Dict[str, int] = {}
        for chunk in chunks:
            for tok in self._tokenize(chunk.text):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab

        V, N = len(vocab), len(chunks)
        tf = np.zeros((N, V), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            for tok in self._tokenize(chunk.text):
                if tok in vocab:
                    tf[i, vocab[tok]] += 1
            row_sum = tf[i].sum()
            if row_sum > 0:
                tf[i] /= row_sum

        df_counts  = (tf > 0).sum(axis=0).astype(np.float32)
        self._idf  = np.log((N + 1) / (df_counts + 1)) + 1.0

        tfidf  = tf * self._idf
        norms  = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._matrix = tfidf / norms
        return self

    def retrieve(self, query: str, k: int = 6) -> List[Chunk]:
        if not self._chunks or self._matrix.size == 0:
            return []
        V  = len(self._vocab)
        qv = np.zeros(V, dtype=np.float32)
        for tok in self._tokenize(query):
            if tok in self._vocab:
                qv[self._vocab[tok]] += 1
        qv   = qv * self._idf
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv /= norm
        scores  = self._matrix.dot(qv)
        top_k   = min(k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            Chunk(
                chunk_id=self._chunks[i].chunk_id,
                text=self._chunks[i].text,
                metadata=self._chunks[i].metadata,
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        tokens  = re.findall(r"[a-z0-9_]+", text.lower())
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams


# Backward-compatible alias used by engine.py
TFIDFRetriever = PgVectorRetriever
