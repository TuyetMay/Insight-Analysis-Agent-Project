from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Set
import pandas as pd
from rag.hyde import HyDEExpander
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

    Step 2.4 changes vs v2:
    ──────────────────────────────────────────────────────────────────
    PROBLEM (v2): schema_metrics + schema_dimensions were ALWAYS injected
    via _MUST_HAVE_IDS into EVERY retrieve() call — including Tier-2 rule-based
    queries that never need schema (intent already parsed, no LLM prompt).

    This wasted ~106 tokens per query (43 + 63) and diluted attention on the
    actual facts the LLM needs.

    FIX (v3 / Step 2.4):
      1. _MUST_HAVE_IDS → empty (no unconditional injection)
      2. _SCHEMA_CHUNK_IDS → new set, injected ONLY when tier=3 (Gemini path)
      3. retrieve() gains `tier: int` parameter (default=2)
      4. _INTENT_MUST_HAVES → refined per intent (kept from v2, + kpi_rank improvement)
      5. kpi_rank → inject dimension_ranked_by_* chunk if breakdown_by known

    TOKEN SAVINGS:
      Tier-2 queries  : −106 tokens/query  (schema_metrics + schema_dimensions)
      Tier-3 queries  : ±0 (schema still injected, needed for Gemini prompt)
      Expected savings: ~80% of queries are Tier-2 → net −85 tokens average
    ──────────────────────────────────────────────────────────────────
    """

    _MAX_TURNS:    int      = 8
    _STATIC_TYPES: Set[str] = {"schema", "filter"}

    # ── v3 Step 2.4: Empty — no unconditional injection ───────────────────────
    # Previously: frozenset({"schema_metrics", "schema_dimensions"})
    # Reason removed: these are only needed by Tier-3 Gemini prompt, not Tier-2
    # rule-based. Injecting them unconditionally wasted 106 tokens per query
    # and diluted retrieval attention.
    _MUST_HAVE_IDS: frozenset = frozenset()

    # ── v3 Step 2.4: Schema chunks injected ONLY at tier=3 ───────────────────
    # These provide grounding for Gemini's JSON plan generation.
    # Rule-based parser (Tier-2) does not need them — intent is already parsed.
    _SCHEMA_CHUNK_IDS: frozenset = frozenset({
        "schema_metrics",
        "schema_dimensions",
    })

    # ── Intent-specific must-haves (refined from v2) ──────────────────────────
    # Injected at the front of results when intent is known.
    # Each entry is the "entry-point chunk" for that query type.
    #
    # Changes vs v2:
    #   kpi_value   : added kpi_margin_snapshot (margin queries need it)
    #   kpi_rank    : added top10_sub_category_profit (most common rank target)
    #   kpi_compare : added trend_overview_sales_yearly (YoY compare needs context)
    #   kpi_detail  : unchanged — anomaly_loss_subcat_summary is the entry point
    _INTENT_MUST_HAVES: Dict[str, List[str]] = {
        "kpi_value":   ["kpi_sales_snapshot",  "kpi_profit_snapshot"],
        "kpi_trend":   ["trend_overview_sales_yearly"],
        "kpi_rank":    ["top10_sub_category_profit"],
        "kpi_compare": ["filter_active", "trend_overview_sales_yearly"],
        "kpi_detail":  ["anomaly_loss_subcat_summary"],
        "clarify":     [],
    }

    # ── Breakdown → dimension rank chunk mapping (for kpi_rank) ──────────────
    # When breakdown_by is known, also inject the pre-computed ranking chunk.
    # e.g. breakdown_by="region" → inject "region_ranked_by_sales"
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
        self._history:  List[Dict[str, str]] = []
        self._built:    bool = False
        self._static_built: bool = False
        self._hyde = HyDEExpander()  

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

    # ── Chat history ──────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        max_messages = self._MAX_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def clear_history(self) -> None:
        self._history = []

    # ── Retrieve ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 6,
        min_score: float = 0.03,
        metadata_filter: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        breakdown_by: Optional[str] = None,
        tier: int = 2,
    ) -> RAGContext:
        """
        Retrieve relevant chunks for a query.

        Parameters
        ----------
        query        : natural language query
        k            : number of candidates from each retriever
        min_score    : minimum cosine similarity to include
        metadata_filter : optional post-retrieval filter on chunk metadata
        intent       : query intent → enables intent-based must-have injection
        breakdown_by : dimension for kpi_rank → injects pre-computed rank chunk
        tier         : execution tier (default=2)
                       2 = rule-based (no schema injection)
                       3 = Gemini LLM (schema injected for prompt grounding)

        Step 2.4 changes
        ----------------
        - tier parameter added
        - Schema chunks (_SCHEMA_CHUNK_IDS) only injected when tier >= 3
        - _MUST_HAVE_IDS is now empty (removed unconditional schema injection)
        - kpi_rank + breakdown_by → inject dimension-specific rank chunks
        - Injection order: schema (tier3) → intent must-haves → semantic results
        """
        if not self._built:
            return RAGContext(query=query, chunks=[],
                              chat_summary=self._history_summary())
        
        expanded_query = self._hyde.expand(query, intent=intent)
        if expanded_query != query:
            logger.debug("HyDE expanded: %r → %r", query[:50], expanded_query[:80])

        # ── Semantic retrieval from both layers ───────────────
        static_hits  = self._static_retriever.retrieve(expanded_query,  k=k // 2 + 2)
        dynamic_hits = self._dynamic_retriever.retrieve(expanded_query, k=k)

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

        _blocked = self._SCHEMA_CHUNK_IDS if tier < 3 else frozenset()

        for c in dynamic_hits + static_hits:
            if (c.score >= min_score
                    and c.chunk_id not in seen
                    and c.chunk_id not in _blocked):
                results.append(c)
                seen.add(c.chunk_id)

        all_chunks    = self._static_chunks + self._dynamic_chunks
        all_chunk_map = {c.chunk_id: c for c in all_chunks}

        # ── Step 2.4: Schema injection — ONLY for Tier 3 ─────
        # Tier-3 = Gemini path: schema facts are needed for the JSON plan prompt.
        # Tier-2 = rule-based: intent already parsed, schema wastes tokens.
        schema_injected: List[Chunk] = []
        if tier >= 3:
            for cid in self._SCHEMA_CHUNK_IDS:
                if cid not in seen and cid in all_chunk_map:
                    schema_injected.append(all_chunk_map[cid])
                    seen.add(cid)
            logger.debug(
                "Tier-3: injected %d schema chunks (%s)",
                len(schema_injected),
                [c.chunk_id for c in schema_injected],
            )
        else:
            logger.debug("Tier-%d: schema chunks SKIPPED (saved ~106 tokens)", tier)

        # ── Intent-based must-haves ───────────────────────────
        intent_injected: List[Chunk] = []
        if intent and intent in self._INTENT_MUST_HAVES:
            for cid in self._INTENT_MUST_HAVES[intent]:
                if cid not in seen and cid in all_chunk_map:
                    intent_injected.append(all_chunk_map[cid])
                    seen.add(cid)

        # ── Step 2.4: Breakdown-specific rank chunks ──────────
        # kpi_rank with breakdown_by="region" → inject "region_ranked_by_sales"
        # so the LLM always has the pre-computed ranking available.
        breakdown_injected: List[Chunk] = []
        if intent == "kpi_rank" and breakdown_by and breakdown_by in self._BREAKDOWN_RANK_CHUNKS:
            for cid in self._BREAKDOWN_RANK_CHUNKS[breakdown_by]:
                if cid not in seen and cid in all_chunk_map:
                    breakdown_injected.append(all_chunk_map[cid])
                    seen.add(cid)

        # ── Sort semantic results by score ────────────────────
        semantic_sorted = sorted(
            results,
            key=lambda x: x.score,
            reverse=True,
        )

        # ── Final order: schema → intent → breakdown → semantic ─
        # Rationale:
        #   - schema first = grounding for Gemini
        #   - intent must-haves = most relevant entry-point chunks
        #   - breakdown = dimension-specific rank context
        #   - semantic = best matches by embedding similarity
        final = schema_injected + intent_injected + breakdown_injected + semantic_sorted

        _token_estimate = sum(len(c.text) // 4 for c in final[:k + len(schema_injected) + len(intent_injected)])
        logger.debug(
            "retrieve() tier=%d intent=%s | chunks=%d schema=%d intent_must=%d breakdown=%d semantic=%d | ~%d tokens",
            tier, intent,
            len(final),
            len(schema_injected), len(intent_injected),
            len(breakdown_injected), len(semantic_sorted),
            _token_estimate,
        )

        return RAGContext(
            query=query,
            chunks=final,
            chat_summary=self._history_summary(),
        )

    def retrieve_for_suggestions(self, last_question: str, last_answer: str,
                                 k: int = 8) -> RAGContext:
        """Suggestions don't need schema — always tier=2."""
        combined = f"{last_question} {last_answer}"
        return self.retrieve(combined, k=k, min_score=0.02, tier=2)

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