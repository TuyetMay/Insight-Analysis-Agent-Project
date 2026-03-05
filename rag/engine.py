"""
rag/engine.py
RAGEngine: high-level facade that wires KnowledgeBaseBuilder + TFIDFRetriever,
manages the rolling chat-history buffer, and produces RAGContext objects
ready for Gemini prompt injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from rag.knowledge_builder import Chunk, KnowledgeBaseBuilder
from rag.retriever import TFIDFRetriever


# ─────────────────────────────────────────────────────────────
# RAGContext — what callers receive
# ─────────────────────────────────────────────────────────────

@dataclass
class RAGContext:
    """Retrieval result: relevant chunks + condensed chat history."""

    query: str
    chunks: List[Chunk]
    chat_summary: str = ""

    def as_prompt_section(self, max_chunks: int = 6) -> str:
        """Render to a plain-text block suitable for embedding in an LLM prompt."""
        lines: List[str] = []
        if self.chat_summary:
            lines += ["[Recent conversation]", self.chat_summary]
        if self.chunks:
            lines += ["\n[Verified data facts from dashboard]"]
            lines += [f"  - {c.text}" for c in self.chunks[:max_chunks]]
        return "\n".join(lines)

    def chunk_texts(self, max_chunks: int = 6) -> List[str]:
        return [c.text for c in self.chunks[:max_chunks]]


# ─────────────────────────────────────────────────────────────
# RAGEngine
# ─────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Usage:
        rag = RAGEngine()
        rag.build(df, kpis, filters)          # rebuild when filters change
        rag.add_turn("user", "What is ...")
        rag.add_turn("assistant", "Total ...")
        ctx = rag.retrieve("question", k=6)
        section = ctx.as_prompt_section()
    """

    _MAX_TURNS: int = 8
    _MUST_HAVE_IDS = frozenset({"schema_metrics", "schema_dimensions", "kpi_summary"})

    def __init__(self) -> None:
        self._retriever = TFIDFRetriever()
        self._all_chunks: List[Chunk] = []
        self._history: List[Dict[str, str]] = []
        self._built: bool = False

    # ── Build / rebuild ───────────────────────────────────────

    def build(self, df: pd.DataFrame, kpis: Dict[str, Any], filters: Dict[str, Any]) -> None:
        """(Re)build the knowledge base. Call whenever filters change."""
        self._all_chunks = KnowledgeBaseBuilder().build(df, kpis, filters)
        self._retriever.fit(self._all_chunks)
        self._built = True

    @property
    def total_chunks(self) -> int:
        return len(self._all_chunks)

    # ── Chat history ──────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Append a conversation turn. Automatically trims to the rolling window."""
        self._history.append({"role": role, "content": content})
        max_messages = self._MAX_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        self._history = []

    # ── Retrieve ──────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 6, min_score: float = 0.03) -> RAGContext:
        """Return the top-k most relevant chunks plus a summary of recent chat."""
        if not self._built:
            return RAGContext(query=query, chunks=[], chat_summary=self._history_summary())

        candidates = self._retriever.retrieve(query, k=k + 4)
        seen: set = set()
        results: List[Chunk] = []

        # High-scoring candidates first
        for c in candidates:
            if c.score >= min_score and c.chunk_id not in seen:
                results.append(c)
                seen.add(c.chunk_id)

        # Always include schema + KPI chunks
        for c in self._all_chunks:
            if c.chunk_id in self._MUST_HAVE_IDS and c.chunk_id not in seen:
                results.append(c)
                seen.add(c.chunk_id)

        must = [c for c in results if c.chunk_id in self._MUST_HAVE_IDS]
        rest = sorted(
            [c for c in results if c.chunk_id not in self._MUST_HAVE_IDS],
            key=lambda x: x.score, reverse=True,
        )
        return RAGContext(query=query, chunks=must + rest,
                         chat_summary=self._history_summary())

    def retrieve_for_suggestions(self, last_question: str, last_answer: str,
                                 k: int = 8) -> RAGContext:
        """Retrieve context optimised for follow-up suggestion generation."""
        combined = f"{last_question} {last_answer}"
        return self.retrieve(combined, k=k, min_score=0.02)

    # ── Internal ──────────────────────────────────────────────

    def _history_summary(self, max_turns: int = 5) -> str:
        recent = self._history[-(max_turns * 2):]
        parts = []
        for msg in recent:
            label = "User" if msg["role"] == "user" else "Bot"
            text = msg["content"][:200] + ("..." if len(msg["content"]) > 200 else "")
            parts.append(f"{label}: {text}")
        return "\n".join(parts)
