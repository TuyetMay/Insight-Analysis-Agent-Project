"""
rag/retriever.py
Lightweight TF-IDF retriever (pure NumPy — no sklearn dependency).
Supports unigrams + bigrams for better phrase matching.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from rag.knowledge_builder import Chunk


class TFIDFRetriever:
    """Index a list of Chunks and retrieve the top-k most relevant ones for a query."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._matrix: np.ndarray = np.array([])

    # ── Public API ────────────────────────────────────────────

    def fit(self, chunks: List[Chunk]) -> "TFIDFRetriever":
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
        top_k = min(k, len(self._chunks))
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

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams
