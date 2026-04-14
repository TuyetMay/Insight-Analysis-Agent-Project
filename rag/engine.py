"""
rag/engine.py  —  v3
────────────────────────────────────────────────────────────────────────────────
Two-layer Retrieval-Augmented Generation engine.

v2 → v3 changes
────────────────
1. ExampleStore integration
   Past query→plan pairs are retrieved alongside data-fact chunks.
   Populated into RAGContext.example_section → injected before data facts
   in the Tier-3 Gemini prompt.  This gives the LLM concrete plan examples
   to copy rather than inferring structure from scratch.

2. Soft re-ranking (priority-weighted scoring)
   Replaces the v2 hard metadata filter with MetadataPreFilter.score_chunks().
   Every chunk receives an intent-priority weight (from a lookup table)
   and a dimension/grain boost.  The final ranking is:
     adjusted_score = cosine_similarity × priority_weight × dimension_boost

3. ExampleStore.add() called by DashboardChatbot on successful Tier-3 plans
   (integration point in chatbot/orchestrator.py — see note below).

4. retrieve() signature unchanged → backward compatible.

NOTE FOR ORCHESTRATOR INTEGRATION
──────────────────────────────────
After a successful Tier-3 plan execution, call:
    self._rag.record_example(question, plan["intent"], plan)
This keeps the example store growing with user-verified plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from rag.hyde import HyDEExpander
from rag.knowledge_builder import Chunk, KnowledgeBaseBuilder
from rag.metadata_filter import MetadataPreFilter, ScoredChunk
from rag.retriever import TFIDFRetriever
from rag.example_store import ExampleStore

logger = logging.getLogger(__name__)


# ── RAGContext ────────────────────────────────────────────────────────────────

@dataclass
class RAGContext:
    """
    Retrieval result: relevant chunks + condensed chat history + example section.

    example_section is injected BEFORE data-fact chunks in as_prompt_section()
    so that the LLM sees working plan patterns before domain facts.
    """
    query: str
    chunks: List[Chunk]
    chat_summary: str = ""
    example_section: str = ""      # v3 NEW: few-shot examples from ExampleStore

    def as_prompt_section(self, max_chunks: int = 6) -> str:
        lines: List[str] = []

        # 1. Few-shot examples (v3 addition)
        if self.example_section:
            lines.append(self.example_section)

        # 2. Recent conversation
        if self.chat_summary:
            lines += ["[Recent conversation]", self.chat_summary, ""]

        # 3. Verified data facts
        if self.chunks:
            lines += ["\n[Verified data facts from dashboard]"]
            lines += [f"  - {c.text}" for c in self.chunks[:max_chunks]]

        return "\n".join(lines)

    def chunk_texts(self, max_chunks: int = 6) -> List[str]:
        return [c.text for c in self.chunks[:max_chunks]]


# ── RAGEngine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Retrieval pipeline:
      build_static()  → schema chunks (once at startup)
      build()         → KPI / trend / anomaly chunks (on filter change)
      retrieve()      → HyDE expand → soft-ranked retrieval → example inject
    """

    _MAX_TURNS:    int       = 8
    _STATIC_TYPES: Set[str]  = {"schema", "filter"}
    _MUST_HAVE_IDS: frozenset = frozenset()

    _SCHEMA_CHUNK_IDS: frozenset = frozenset({
        "schema_metrics",
        "schema_dimensions",
    })

    _INTENT_MUST_HAVES: Dict[str, List[str]] = {
        "kpi_value":   ["kpi_sales_snapshot", "kpi_profit_snapshot"],
        "kpi_trend":   ["trend_overview_sales_yearly"],
        "kpi_rank":    ["top10_sub_category_profit"],
        "kpi_compare": ["filter_active", "trend_overview_sales_yearly"],
        "kpi_detail":  ["anomaly_loss_subcat_summary"],
        "clarify":     [],
    }

    _BREAKDOWN_RANK_CHUNKS: Dict[str, List[str]] = {
        "region":       ["region_ranked_by_sales",    "region_ranked_by_profit"],
        "segment":      ["segment_ranked_by_sales",   "segment_ranked_by_profit"],
        "category":     ["category_ranked_by_sales",  "category_ranked_by_profit"],
        "sub_category": ["top10_sub_category_profit", "top10_sub_category_sales"],
    }

    def __init__(self) -> None:
        self._static_chunks:  List[Chunk] = []
        self._dynamic_chunks: List[Chunk] = []
        self._static_retriever  = TFIDFRetriever()
        self._dynamic_retriever = TFIDFRetriever()
        self._history: List[Dict[str, str]] = []
        self._built:       bool = False
        self._static_built: bool = False
        self._hyde        = HyDEExpander()
        self._meta_filter = MetadataPreFilter()    # v3: soft scoring
        self._examples    = ExampleStore()          # v3: NEW

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_static(self, df: pd.DataFrame) -> None:
        builder = KnowledgeBaseBuilder()
        self._static_chunks = builder.build_static(df)
        self._static_retriever.fit(self._static_chunks)
        self._static_built = True

    def build(self, df: pd.DataFrame, kpis: Dict[str, Any],
              filters: Dict[str, Any]) -> None:
        if not self._static_built:
            self.build_static(df)

        builder    = KnowledgeBaseBuilder()
        new_chunks = builder.build_dynamic(df, kpis, filters)

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

    # ── Chat history ──────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        max_messages = self._MAX_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        self._history = []

    # ── Example recording (v3 NEW) ────────────────────────────────────────────

    def record_example(
        self,
        question: str,
        intent: str,
        plan: Dict[str, Any],
        sql_pattern: str = "",
    ) -> None:
        """
        Register a successful query→plan pair into the ExampleStore.
        Call this from DashboardChatbot after a verified Tier-3 execution.
        """
        self._examples.add(
            question=question,
            intent=intent,
            plan=plan,
            sql_pattern=sql_pattern,
        )

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 6,
        min_score: float = 0.03,
        metadata_filter: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        breakdown_by: Optional[str] = None,
        tier: int = 2,
        grain: Optional[str] = None,
        inject_examples: bool = True,   # v3 NEW: inject few-shot into context
    ) -> RAGContext:
        """
        Retrieve relevant chunks.

        v3 pipeline
        ───────────
        1. HyDE expansion (semantic vocabulary augmentation)
        2. Dense cosine retrieval from both static + dynamic layers
        3. Soft re-ranking: score × priority_weight × dimension_boost
        4. Schema injection (tier-3 only)
        5. Intent must-have injection
        6. Breakdown rank chunk injection (kpi_rank + known dimension)
        7. Example section retrieval (inject_examples=True and tier=3)
        8. Return RAGContext with example_section populated

        Parameters
        ──────────
        grain           : time grain hint for trend scoring boost
        inject_examples : include few-shot examples (default True for tier=3 callers)
        """
        if not self._built:
            return RAGContext(
                query=query, chunks=[],
                chat_summary=self._history_summary(),
            )

        # ── Step 1: HyDE expand ───────────────────────────────────────────────
        expanded_query = self._hyde.expand(query, intent=intent)
        if expanded_query != query:
            logger.debug("HyDE: %r → %r", query[:50], expanded_query[:70])

        # ── Step 2: Dense retrieval ───────────────────────────────────────────
        static_hits  = self._static_retriever.retrieve(expanded_query,  k=k // 2 + 2)
        dynamic_hits = self._dynamic_retriever.retrieve(expanded_query, k=k)

        if metadata_filter:
            static_hits  = [c for c in static_hits
                            if all(c.metadata.get(fk) == fv
                                   for fk, fv in metadata_filter.items())]
            dynamic_hits = [c for c in dynamic_hits
                            if all(c.metadata.get(fk) == fv
                                   for fk, fv in metadata_filter.items())]

        # ── Step 3: Soft re-ranking with priority weights (v3) ────────────────
        _blocked = self._SCHEMA_CHUNK_IDS if tier < 3 else frozenset()

        # Deduplicate and score
        seen: Set[str] = set()
        raw_candidates: List[Chunk] = []
        for c in dynamic_hits + static_hits:
            if (c.score >= min_score
                    and c.chunk_id not in seen
                    and c.chunk_id not in _blocked):
                raw_candidates.append(c)
                seen.add(c.chunk_id)

        # Apply soft priority scoring
        if intent:
            scored = self._meta_filter.score_chunks(
                raw_candidates,
                intent=intent,
                breakdown_by=breakdown_by,
                grain=grain,
            )
            # Sort by adjusted score descending
            scored.sort(key=lambda x: x.adjusted_score, reverse=True)
            results = [sc.chunk for sc in scored]

            logger.debug(
                "[MetadataFilter-v3] intent=%s | raw=%d → scored=%d | "
                "top weights: %s",
                intent, len(raw_candidates), len(results),
                [f"{sc.chunk.chunk_id}×{sc.priority_weight:.1f}"
                 for sc in scored[:4]],
            )
        else:
            results = sorted(raw_candidates, key=lambda c: c.score, reverse=True)

        # ── Step 4: Schema injection (tier-3 only) ────────────────────────────
        all_chunk_map = {c.chunk_id: c
                         for c in self._static_chunks + self._dynamic_chunks}

        schema_injected: List[Chunk] = []
        if tier >= 3:
            for cid in self._SCHEMA_CHUNK_IDS:
                if cid not in seen and cid in all_chunk_map:
                    schema_injected.append(all_chunk_map[cid])
                    seen.add(cid)

        # ── Step 5: Intent must-haves ─────────────────────────────────────────
        intent_injected: List[Chunk] = []
        if intent and intent in self._INTENT_MUST_HAVES:
            for cid in self._INTENT_MUST_HAVES[intent]:
                if cid not in seen and cid in all_chunk_map:
                    intent_injected.append(all_chunk_map[cid])
                    seen.add(cid)

        # ── Step 6: Breakdown rank chunks ─────────────────────────────────────
        breakdown_injected: List[Chunk] = []
        if intent == "kpi_rank" and breakdown_by and breakdown_by in self._BREAKDOWN_RANK_CHUNKS:
            for cid in self._BREAKDOWN_RANK_CHUNKS[breakdown_by]:
                if cid not in seen and cid in all_chunk_map:
                    breakdown_injected.append(all_chunk_map[cid])
                    seen.add(cid)

        # ── Final assembly: schema → intent → breakdown → semantic ────────────
        final = schema_injected + intent_injected + breakdown_injected + results

        # Trim to k (keep injected chunks + top-k semantic)
        n_injected = len(schema_injected) + len(intent_injected) + len(breakdown_injected)
        final = final[:n_injected + k]

        # ── Step 7: Example section (tier-3 or explicit inject_examples) ─────
        example_section = ""
        if (inject_examples and tier >= 3) or (inject_examples and intent == "kpi_detail"):
            example_section = self._examples.as_prompt_section(
                question=query,
                k=3,
                intent=intent,
            )

        # ── Logging ───────────────────────────────────────────────────────────
        _token_est = sum(len(c.text) // 4 for c in final)
        logger.debug(
            "retrieve() tier=%d intent=%s | final=%d chunks "
            "(schema=%d intent=%d breakdown=%d semantic=%d) | ~%dt | examples=%s",
            tier, intent, len(final),
            len(schema_injected), len(intent_injected),
            len(breakdown_injected), len(results),
            _token_est,
            "yes" if example_section else "no",
        )

        return RAGContext(
            query=query,
            chunks=final,
            chat_summary=self._history_summary(),
            example_section=example_section,
        )

    def retrieve_for_suggestions(
        self,
        last_question: str,
        last_answer: str,
        k: int = 8,
    ) -> RAGContext:
        """Suggestions don't need schema or examples — always tier=2."""
        combined = f"{last_question} {last_answer}"
        return self.retrieve(combined, k=k, min_score=0.02, tier=2,
                             inject_examples=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _history_summary(self, max_turns: int = 5) -> str:
        recent = self._history[-(max_turns * 2):]
        parts  = []
        for msg in recent:
            label = "User" if msg["role"] == "user" else "Bot"
            text  = msg["content"][:200] + ("..." if len(msg["content"]) > 200 else "")
            parts.append(f"{label}: {text}")
        return "\n".join(parts)

    @staticmethod
    def _apply_metadata_filter(
        chunks: List[Chunk],
        flt: Optional[Dict[str, Any]],
    ) -> List[Chunk]:
        if not flt:
            return chunks
        result = [c for c in chunks
                  if all(c.metadata.get(k) == v for k, v in flt.items())]
        return result if result else chunks