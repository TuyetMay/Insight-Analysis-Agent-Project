from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set
import pandas as pd
from rag.knowledge_builder import Chunk, KnowledgeBaseBuilder
from rag.retriever import TFIDFRetriever
logger = logging.getLogger(__name__)

 
@dataclass
class RAGContext:
    """Retrieval result: relevant chunks + condensed chat history."""
    query: str
    chunks: List[Chunk]
    chat_summary: str = ""
 
    def as_prompt_section(self, max_chunks: int = 6) -> str:
        lines: List[str] = []
        if self.chat_summary:
            lines += ["[Recent conversation]", self.chat_summary]
        if self.chunks:
            lines += ["\n[Verified data facts from dashboard]"]
            lines += [f"  - {c.text}" for c in self.chunks[:max_chunks]]
        return "\n".join(lines)
 
    def chunk_texts(self, max_chunks: int = 6) -> List[str]:
        return [c.text for c in self.chunks[:max_chunks]]
 
 
class RAGEngine:
    """
    Two-layer RAG:
      - Static layer  : schema facts, dimension values — built ONCE at startup
      - Dynamic layer : KPI values, trends — rebuilt only when filters change
    Retrieval: metadata pre-filter → TF-IDF search trong subset.
    """
 
    _MAX_TURNS:    int      = 8
    _STATIC_TYPES: Set[str] = {"schema", "filter"}   # chunk types không đổi theo filter
    _MUST_HAVE_IDS           = frozenset({"schema_metrics", "schema_dimensions", "kpi_summary"})
 
    def __init__(self) -> None:
        self._static_chunks:  List[Chunk] = []
        self._dynamic_chunks: List[Chunk] = []
        self._static_retriever  = TFIDFRetriever()
        self._dynamic_retriever = TFIDFRetriever()
        self._history:  List[Dict[str, str]] = []
        self._built:    bool = False
        self._static_built: bool = False
 
    # ── Build ─────────────────────────────────────────────────
 
    def build_static(self, df: pd.DataFrame) -> None:
        """
        Gọi 1 lần khi app khởi động — build schema + dimension facts.
        Không phụ thuộc vào filter.
        """
        builder = KnowledgeBaseBuilder()
        self._static_chunks = builder.build_static(df)
        self._static_retriever.fit(self._static_chunks)
        self._static_built = True
 
    def build(self, df: pd.DataFrame, kpis: Dict[str, Any],
          filters: Dict[str, Any]) -> None:
        if not self._static_built:
            self.build_static(df)

        builder = KnowledgeBaseBuilder()
        new_chunks = builder.build_dynamic(df, kpis, filters)

        # ── Chỉ re-fit khi nội dung chunk thực sự thay đổi ──
        new_texts = {c.chunk_id: c.text for c in new_chunks}
        old_texts = {c.chunk_id: c.text for c in self._dynamic_chunks}

        self._dynamic_chunks = new_chunks

        if new_texts != old_texts:
            self._dynamic_retriever.fit(self._dynamic_chunks)
            logger.info("Dynamic retriever re-fitted (%d chunks)", len(new_chunks))
        else:
            logger.info("Chunk content unchanged — skipping re-fit")

        self._built = True
    
    @property
    def total_chunks(self) -> int:
        return len(self._static_chunks) + len(self._dynamic_chunks)
 
    # ── Chat history ──────────────────────────────────────────
 
    def add_turn(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        max_messages = self._MAX_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]
 
    def clear_history(self) -> None:
        self._history = []
 
    # ── Retrieve ──────────────────────────────────────────────
 
    def retrieve(self, query: str, k: int = 6,
             min_score: float = 0.03,
             metadata_filter: Optional[Dict[str, Any]] = None) -> RAGContext:
        if not self._built:
            return RAGContext(query=query, chunks=[], chat_summary=self._history_summary())

        # ── FIX: dùng pre-built retrievers, KHÔNG tạo instance mới ──
        static_hits  = self._static_retriever.retrieve(query,  k=k // 2 + 2)
        dynamic_hits = self._dynamic_retriever.retrieve(query, k=k)

        # Metadata filter áp dụng POST-retrieval
        if metadata_filter:
            static_hits  = [c for c in static_hits
                            if all(c.metadata.get(fk) == fv
                                for fk, fv in metadata_filter.items())]
            dynamic_hits = [c for c in dynamic_hits
                            if all(c.metadata.get(fk) == fv
                                for fk, fv in metadata_filter.items())]

        seen: set        = set()
        results: List[Chunk] = []

        for c in dynamic_hits + static_hits:
            if c.score >= min_score and c.chunk_id not in seen:
                results.append(c)
                seen.add(c.chunk_id)

        all_chunks = self._static_chunks + self._dynamic_chunks
        for c in all_chunks:
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
        combined = f"{last_question} {last_answer}"
        return self.retrieve(combined, k=k, min_score=0.02)
 
    # ── Internal helpers ──────────────────────────────────────
 
    @staticmethod
    def _apply_metadata_filter(chunks: List[Chunk],
                               flt: Optional[Dict[str, Any]]) -> List[Chunk]:
        """
        Trả về chunks khớp TẤT CẢ key-value trong flt.
        Nếu flt=None hoặc {}, trả về toàn bộ.
        """
        if not flt:
            return chunks
        result = []
        for c in chunks:
            if all(c.metadata.get(k) == v for k, v in flt.items()):
                result.append(c)
        return result if result else chunks   # fallback: không filter nếu không có gì match
 
    def _history_summary(self, max_turns: int = 5) -> str:
        recent = self._history[-(max_turns * 2):]
        parts  = []
        for msg in recent:
            label = "User" if msg["role"] == "user" else "Bot"
            text  = msg["content"][:200] + ("..." if len(msg["content"]) > 200 else "")
            parts.append(f"{label}: {text}")
        return "\n".join(parts)