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
      - Static layer  : schema facts — built ONCE at startup
      - Dynamic layer : KPI values, trends — rebuilt only when filters change
    Retrieval: dense embedding search → intent-based must-have injection.

    Changes vs v1:
      - _MUST_HAVE_IDS: removed kpi_summary (score=0.000 in ALL 20 test queries)
      - Added _INTENT_MUST_HAVES: inject relevant chunks based on query intent
      - retrieve(): added intent parameter for intent-aware injection
      - Intent kpi_detail now injects anomaly_loss_subcat_summary as must-have
    """

    _MAX_TURNS:    int      = 8
    _STATIC_TYPES: Set[str] = {"schema", "filter"}

    # v2: kpi_summary removed — it scored 0.000 in every query (audit confirmed).
    # schema_metrics and schema_dimensions stay because they provide grounding for
    # Tier-3 Gemini plan generation and are small (25–36 tokens each).
    _MUST_HAVE_IDS = frozenset({
        "schema_metrics",
        "schema_dimensions",
        # REMOVED: "kpi_summary" — score=0.000, force-inject was meaningless
    })

    # Intent-specific must-haves — injected when intent is known.
    # These are the most relevant entry-point chunks per query type.
    _INTENT_MUST_HAVES: Dict[str, List[str]] = {
        "kpi_value":   ["kpi_sales_snapshot", "kpi_profit_snapshot"],
        "kpi_trend":   ["trend_overview_sales_yearly"],
        "kpi_rank":    [],   # retriever decides based on dimension in plan
        "kpi_compare": ["filter_active"],
        # CRITICAL: inject anomaly entry point for kpi_detail so retriever
        # doesn't have to find it — audit showed kpi_detail had 0 ⭐ hits.
        "kpi_detail":  ["anomaly_loss_subcat_summary"],
        "clarify":     [],
    }

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
        Call once at startup — builds schema chunks only.
        Independent of filters, does not change.
        """
        builder = KnowledgeBaseBuilder()
        self._static_chunks = builder.build_static(df)
        self._static_retriever.fit(self._static_chunks)
        self._static_built = True

    def build(self, df: pd.DataFrame, kpis: Dict[str, Any],
              filters: Dict[str, Any]) -> None:
        """Build dynamic layer — call whenever filters change."""
        if not self._static_built:
            self.build_static(df)

        builder   = KnowledgeBaseBuilder()
        new_chunks = builder.build_dynamic(df, kpis, filters)

        # Only re-fit if content actually changed (avoid unnecessary recompute)
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
                 metadata_filter: Optional[Dict[str, Any]] = None,
                 intent: Optional[str] = None) -> RAGContext:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query   : natural language query
        k       : number of candidates from each retriever
        min_score : minimum cosine similarity to include
        metadata_filter : optional post-retrieval filter on chunk metadata
        intent  : optional query intent (kpi_value/kpi_trend/kpi_rank/
                  kpi_compare/kpi_detail) — enables intent-based must-haves

        Changes vs v1
        -------------
        - Added intent parameter
        - Intent-aware must-have injection at end of method
        - kpi_detail intent injects anomaly_loss_subcat_summary automatically
        """
        if not self._built:
            return RAGContext(query=query, chunks=[],
                              chat_summary=self._history_summary())

        # ── Semantic retrieval from both layers ───────────────
        static_hits  = self._static_retriever.retrieve(query,  k=k // 2 + 2)
        dynamic_hits = self._dynamic_retriever.retrieve(query, k=k)

        # Optional post-retrieval metadata filter
        if metadata_filter:
            static_hits  = [c for c in static_hits
                            if all(c.metadata.get(fk) == fv
                                   for fk, fv in metadata_filter.items())]
            dynamic_hits = [c for c in dynamic_hits
                            if all(c.metadata.get(fk) == fv
                                   for fk, fv in metadata_filter.items())]

        # Deduplicate and filter by min_score
        seen: Set[str] = set()
        results: List[Chunk] = []

        for c in dynamic_hits + static_hits:
            if c.score >= min_score and c.chunk_id not in seen:
                results.append(c)
                seen.add(c.chunk_id)

        # ── Unconditional must-haves ──────────────────────────
        all_chunks = self._static_chunks + self._dynamic_chunks
        for c in all_chunks:
            if c.chunk_id in self._MUST_HAVE_IDS and c.chunk_id not in seen:
                results.append(c)
                seen.add(c.chunk_id)

        # ── Split: must-haves first, rest sorted by score ─────
        must = [c for c in results if c.chunk_id in self._MUST_HAVE_IDS]
        rest = sorted(
            [c for c in results if c.chunk_id not in self._MUST_HAVE_IDS],
            key=lambda x: x.score,
            reverse=True,
        )

        # ── Intent-based must-haves (NEW in v2) ──────────────
        # These are injected at the front because they are the most
        # relevant entry-point chunks for the specific query type.
        # Example: kpi_detail → inject anomaly_loss_subcat_summary
        # so the LLM always has access to the loss-making items list.
        if intent and intent in self._INTENT_MUST_HAVES:
            intent_ids  = self._INTENT_MUST_HAVES[intent]
            all_chunk_map = {c.chunk_id: c for c in all_chunks}
            for cid in intent_ids:
                if cid not in seen and cid in all_chunk_map:
                    must.insert(0, all_chunk_map[cid])
                    seen.add(cid)

        return RAGContext(query=query,
                          chunks=must + rest,
                          chat_summary=self._history_summary())

    def retrieve_for_suggestions(self, last_question: str, last_answer: str,
                                 k: int = 8) -> RAGContext:
        combined = f"{last_question} {last_answer}"
        return self.retrieve(combined, k=k, min_score=0.02)

    # ── Internal helpers ──────────────────────────────────────

    @staticmethod
    def _apply_metadata_filter(chunks: List[Chunk],
                               flt: Optional[Dict[str, Any]]) -> List[Chunk]:
        if not flt:
            return chunks
        result = [c for c in chunks
                  if all(c.metadata.get(k) == v for k, v in flt.items())]
        return result if result else chunks

    def _history_summary(self, max_turns: int = 5) -> str:
        recent = self._history[-(max_turns * 2):]
        parts  = []
        for msg in recent:
            label = "User" if msg["role"] == "user" else "Bot"
            text  = msg["content"][:200] + ("..." if len(msg["content"]) > 200 else "")
            parts.append(f"{label}: {text}")
        return "\n".join(parts)