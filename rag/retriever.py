from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from rag.knowledge_builder import Chunk

logger = logging.getLogger(__name__)

import hashlib
import pickle
import pathlib

_CACHE_DIR = pathlib.Path(".embedding_cache")
_CACHE_DIR.mkdir(exist_ok=True)

_model_lock  = threading.Lock()
_shared_model = None   


def _get_model():
    """
    Lazy-load sentence-transformers model.
    Thread-safe singleton — model chỉ load 1 lần dù có nhiều retriever instances.
    """
    global _shared_model
    if _shared_model is not None:
        return _shared_model

    with _model_lock:
        if _shared_model is not None:   
            return _shared_model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers model all-MiniLM-L6-v2 ...")
            _shared_model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",         
            )
            logger.info("Dense embedding model loaded successfully.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to TF-IDF retriever. "
                "Run: pip install sentence-transformers"
            )
            _shared_model = None
    return _shared_model


#  Dense Retriever 

class DenseRetriever:
    """
    Cosine-similarity retriever over dense sentence embeddings.
    """

    def __init__(self) -> None:
        self._chunks:     List[Chunk]   = []
        self._embeddings: Optional[np.ndarray] = None   # shape (N, 384)

    #  Public API (same as TFIDFRetriever) 

    def fit(self, chunks: List[Chunk]) -> "DenseRetriever":
        self._chunks = chunks
        if not chunks:
            self._embeddings = None
            return self

        model = _get_model()
        if model is None:
            self._embeddings = None
            self._tfidf_fallback = TFIDFFallback()
            self._tfidf_fallback.fit(chunks)
            return self

        # ── Cache lookup ──────────────────────────────────────
        texts     = [c.text for c in chunks]
        cache_key = hashlib.md5("".join(texts).encode()).hexdigest()
        cache_path = _CACHE_DIR / f"{cache_key}.pkl"

        if cache_path.exists():
            logger.info("Loading embeddings from cache: %s", cache_key[:8])
            self._embeddings = pickle.loads(cache_path.read_bytes())
            return self

        raw = model.encode(
            texts, batch_size=64, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        self._embeddings = raw.astype(np.float32)
        cache_path.write_bytes(pickle.dumps(self._embeddings))
        logger.info("Embeddings cached: %s", cache_key[:8])
        return self

    def retrieve(self, query: str, k: int = 6) -> List[Chunk]:
        """Return top-k chunks by cosine similarity to query."""
        if not self._chunks:
            return []

        if self._embeddings is None:
            if hasattr(self, "_tfidf_fallback"):
                return self._tfidf_fallback.retrieve(query, k)
            return []

        model = _get_model()
        if model is None:
            return []

        q_vec = model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)                  # shape (384,)

        scores = self._embeddings.dot(q_vec)     # shape (N,) — cosine similarity

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



class TFIDFFallback:
    """
    Pure-NumPy TF-IDF retriever — fallback khi sentence-transformers chưa cài.
    Code giữ nguyên từ phiên bản cũ.
    """

    def __init__(self) -> None:
        from typing import Dict
        self._chunks:  List[Chunk]      = []
        self._vocab:   Dict[str, int]   = {}
        self._idf:     np.ndarray       = np.array([])
        self._matrix:  np.ndarray       = np.array([])

    def fit(self, chunks: List[Chunk]) -> "TFIDFFallback":
        self._chunks = chunks
        if not chunks:
            return self

        vocab: dict = {}
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

        df_counts = (tf > 0).sum(axis=0).astype(np.float32)
        self._idf = np.log((N + 1) / (df_counts + 1)) + 1.0

        tfidf = tf * self._idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._matrix = tfidf / norms
        return self

    def retrieve(self, query: str, k: int = 6) -> List[Chunk]:
        if not self._chunks or self._matrix.size == 0:
            return []
        V = len(self._vocab)
        qv = np.zeros(V, dtype=np.float32)
        for tok in self._tokenize(query):
            if tok in self._vocab:
                qv[self._vocab[tok]] += 1
        qv = qv * self._idf
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv /= norm
        scores = self._matrix.dot(qv)
        top_k  = min(k, len(self._chunks))
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
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

TFIDFRetriever = DenseRetriever