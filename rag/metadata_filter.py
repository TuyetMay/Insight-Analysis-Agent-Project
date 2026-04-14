"""
rag/metadata_filter.py  —  v2
────────────────────────────────────────────────────────────────────────────────
Pre-filters the candidate chunk pool by query intent.

v1 vs v2 changes
────────────────
v1: HARD filter — chunks not in allowlist are completely removed.
    Problem: a kpi_detail query like "which products are bleeding money" could
    miss a `discount_impact` insight chunk that is highly relevant even though
    it's not in the kpi_detail allowlist.

v2: SOFT filter — chunks not in allowlist get their similarity score multiplied
    by a penalty factor (default 0.3) instead of being removed.
    Chunks with score × penalty still above min_score pass through.

    New method: `score_chunks()` returns (chunk, adjusted_score) pairs
    that the retriever uses for re-ranking.

    Backward compat: `filter()` still works as before for callers that
    don't want the soft-scoring pipeline.

PRIORITY WEIGHTS
────────────────
Each chunk type is given a priority weight per intent.
Weight > 1.0: boosted (injected first), = 1.0: neutral, < 1.0: penalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rag.knowledge_builder import Chunk


# ── Allowlists (unchanged from v1, kept for hard-filter backward compat) ─────

_INTENT_ALLOWLIST: Dict[str, Set[str]] = {
    "kpi_value": {
        "kpi_snapshot", "dimension_value", "dimension_rank",
        "cross_fact", "insight", "filter_context", "trend_overview",
        "time_period_fact",
    },
    "kpi_trend": {
        "time_period_fact", "trend_transition", "trend_overview",
        "dimension_rank", "filter_context",
    },
    "kpi_rank": {
        "dimension_rank", "dimension_value", "anomaly_fact",
        "kpi_snapshot", "filter_context",
    },
    "kpi_detail": {
        "anomaly_fact", "insight", "filter_context",
    },
    "kpi_compare": {
        "time_period_fact", "trend_transition", "trend_overview",
        "kpi_snapshot", "filter_context",
    },
    "clarify": set(),
}

# ── Priority weights: (chunk_type, intent) → multiplier ───────────────────────
# Values > 1.0 boost, < 1.0 penalise, 1.0 neutral.
# Unlisted = _DEFAULT_PRIORITY (0.4 — light penalty).

_DEFAULT_PRIORITY = 0.4

_PRIORITY_TABLE: Dict[Tuple[str, str], float] = {
    # kpi_detail
    ("anomaly_fact",     "kpi_detail"):  2.0,
    ("insight",          "kpi_detail"):  1.5,
    ("filter_context",   "kpi_detail"):  1.2,
    ("time_period_fact", "kpi_detail"):  0.2,
    ("trend_transition", "kpi_detail"):  0.2,
    ("kpi_snapshot",     "kpi_detail"):  0.3,
    # kpi_trend
    ("trend_transition", "kpi_trend"):   2.0,
    ("trend_overview",   "kpi_trend"):   1.8,
    ("time_period_fact", "kpi_trend"):   1.5,
    ("dimension_rank",   "kpi_trend"):   1.0,
    ("kpi_snapshot",     "kpi_trend"):   0.3,
    ("anomaly_fact",     "kpi_trend"):   0.3,
    # kpi_rank
    ("dimension_rank",   "kpi_rank"):    2.0,
    ("dimension_value",  "kpi_rank"):    1.5,
    ("anomaly_fact",     "kpi_rank"):    1.2,
    ("kpi_snapshot",     "kpi_rank"):    0.8,
    ("trend_transition", "kpi_rank"):    0.2,
    ("time_period_fact", "kpi_rank"):    0.2,
    # kpi_value
    ("kpi_snapshot",     "kpi_value"):   2.0,
    ("dimension_value",  "kpi_value"):   1.8,
    ("dimension_rank",   "kpi_value"):   1.5,
    ("cross_fact",       "kpi_value"):   1.2,
    ("trend_transition", "kpi_value"):   0.2,
    # kpi_compare
    ("trend_transition", "kpi_compare"): 2.0,
    ("trend_overview",   "kpi_compare"): 1.8,
    ("time_period_fact", "kpi_compare"): 1.5,
    ("kpi_snapshot",     "kpi_compare"): 1.0,
    ("anomaly_fact",     "kpi_compare"): 0.3,
    ("dimension_value",  "kpi_compare"): 0.3,
}

# Safety floor: never reduce below this fraction of original score
_SOFT_FLOOR = 0.05

_MIN_POOL = 2


@dataclass
class ScoredChunk:
    """Chunk with adjusted relevance score after intent-priority re-weighting."""
    chunk: Chunk
    original_score: float
    priority_weight: float
    adjusted_score: float


class MetadataPreFilter:
    """
    Stateless pre-filter; instantiate once, call score_chunks() per retrieve().
    """

    # ── New primary method: soft scoring ─────────────────────────────────────

    def score_chunks(
        self,
        chunks: List[Chunk],
        intent: Optional[str],
        breakdown_by: Optional[str] = None,
        grain: Optional[str] = None,
        soft_penalty: float = 0.3,
    ) -> List[ScoredChunk]:
        """
        Return ScoredChunk list with adjusted scores.

        Scoring formula per chunk:
          priority_weight = _PRIORITY_TABLE.get((type, intent), _DEFAULT_PRIORITY)
                            × dimension_boost (1.2 if dimension matches breakdown_by)
                            × grain_boost     (1.2 if grain matches time_grain)

          adjusted_score = max(original_score × priority_weight, original_score × SOFT_FLOOR)

        For chunks in the allowlist, priority_weight ≥ 1.0 (neutral or boost).
        For chunks outside the allowlist, priority_weight is the _DEFAULT_PRIORITY
        unless the table assigns an explicit value.
        """
        if not intent or intent not in _INTENT_ALLOWLIST:
            return [ScoredChunk(c, c.score, 1.0, c.score) for c in chunks]

        scored: List[ScoredChunk] = []
        for chunk in chunks:
            ctype = chunk.metadata.get("type", "")

            # Base priority from table
            weight = _PRIORITY_TABLE.get((ctype, intent), _DEFAULT_PRIORITY)

            # Dimension boost: chunk matches requested dimension
            if breakdown_by and chunk.metadata.get("dimension") == breakdown_by:
                weight *= 1.2

            # Grain boost: trend chunk matches requested time grain
            if grain and grain != "none" and chunk.metadata.get("grain") == grain:
                weight *= 1.2

            # filter_context and must-have injected chunks always neutral
            if ctype in ("filter_context", "schema_fact"):
                weight = max(weight, 1.0)

            adj_score = max(chunk.score * weight, chunk.score * _SOFT_FLOOR)
            scored.append(ScoredChunk(chunk, chunk.score, weight, adj_score))

        return scored

    # ── Backward-compat: hard filter (kept from v1) ───────────────────────────

    def filter(
        self,
        chunks: List[Chunk],
        intent: Optional[str],
        breakdown_by: Optional[str] = None,
        metric: Optional[str] = None,
        grain: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Hard-filter fallback (v1 behaviour).
        Kept for callers that use the old API.
        New code should prefer score_chunks().
        """
        if not intent or intent not in _INTENT_ALLOWLIST:
            return chunks

        allowlist = _INTENT_ALLOWLIST[intent]
        if not allowlist:
            return chunks

        filtered = [c for c in chunks if c.metadata.get("type", "") in allowlist]
        if len(filtered) < _MIN_POOL:
            return chunks

        # Dimension sub-filter
        if breakdown_by:
            dim_matched = [
                c for c in filtered
                if c.metadata.get("dimension") == breakdown_by
                or c.metadata.get("dimension") is None
                or c.metadata.get("type") in (
                    "kpi_snapshot", "filter_context", "cross_fact",
                    "insight", "trend_overview", "anomaly_fact",
                )
            ]
            if len(dim_matched) >= _MIN_POOL:
                filtered = dim_matched

        # Grain sub-filter
        if grain and grain != "none" and intent == "kpi_trend":
            grain_matched = [
                c for c in filtered
                if c.metadata.get("grain") == grain
                or c.metadata.get("type") not in (
                    "time_period_fact", "trend_transition", "trend_overview",
                )
            ]
            if len(grain_matched) >= _MIN_POOL:
                filtered = grain_matched

        return filtered

    def stats(
        self,
        original: List[Chunk],
        filtered: List[Chunk],
        intent: str,
    ) -> str:
        """Debug helper — returns one-line summary."""
        removed = len(original) - len(filtered)
        pct     = removed / len(original) * 100 if original else 0
        return (
            f"[MetadataFilter] intent={intent} | "
            f"{len(original)} → {len(filtered)} chunks "
            f"(-{removed}, -{pct:.0f}%)"
        )