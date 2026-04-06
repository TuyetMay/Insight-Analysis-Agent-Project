"""
rag/metadata_filter.py
Step 2.1 — Metadata Pre-filtering

Reduces the candidate chunk pool BEFORE semantic search by keeping only
chunk types relevant to the query intent.

WHY:
  Without pre-filtering, a kpi_detail query ("which products lose money")
  competes against 46 chunks including irrelevant time_period_fact chunks
  (year_2014_sales_fact, trend_year_sales_2014_2015...) that dilute the
  semantic space and push anomaly_fact chunks down the ranking.

DESIGN:
  - Each intent has an allowlist of chunk types
  - Optional sub-filters: dimension match, grain match, metric match
  - Safety fallback: if filtered pool < MIN_POOL, skip filter entirely
    (prevents over-filtering from breaking retrieval)
  - filter_context always included (date range context is always useful)
  - schema_fact excluded here (handled separately by tier logic)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from rag.knowledge_builder import Chunk


# ── Intent → allowed chunk type allowlists ─────────────────────────────────
# Types NOT in the allowlist are excluded from the candidate pool.
# filter_context is always implicitly included (added in filter()).

_INTENT_ALLOWLIST: Dict[str, Set[str]] = {
    "kpi_value": {
        "kpi_snapshot",
        "dimension_value",
        "dimension_rank",
        "cross_fact",
        "insight",
        "filter_context",
        "trend_overview",    
        "time_period_fact",

    },
    "kpi_trend": {
        "time_period_fact",
        "trend_transition",
        "trend_overview",
        "dimension_rank",   # needed for breakdown trends
        "filter_context",
    },
    "kpi_rank": {
        "dimension_rank",
        "dimension_value",
        "anomaly_fact",     # loss-making rank queries
        "kpi_snapshot",
        "filter_context",
    },
    "kpi_detail": {
        "anomaly_fact",
        "insight",
        "filter_context",
    },
    "kpi_compare": {
        "time_period_fact",
        "trend_transition",
        "trend_overview",
        "kpi_snapshot",
        "filter_context",
    },
    "clarify": set(),  # no filter — pass everything
}

# Minimum candidate pool after filtering.
# If fewer chunks survive, skip the filter (safety fallback).
_MIN_POOL = 2


class MetadataPreFilter:
    """
    Stateless pre-filter: call filter() per retrieve() call.
    """

    def filter(
        self,
        chunks: List[Chunk],
        intent: Optional[str],
        breakdown_by: Optional[str] = None,
        metric: Optional[str] = None,
        grain: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Return a filtered subset of chunks relevant to the query context.

        Parameters
        ----------
        chunks      : full candidate pool (static + dynamic, pre-dedup)
        intent      : query intent from NL parser
        breakdown_by: dimension (region/segment/category/sub_category)
        metric      : primary metric (sales/profit/orders/profit_margin)
        grain       : time grain (year/quarter/month/week)

        Returns
        -------
        Filtered list, or original list if filter produces < _MIN_POOL results.
        """
        if not intent or intent not in _INTENT_ALLOWLIST:
            return chunks

        allowlist = _INTENT_ALLOWLIST[intent]

        # clarify = no filter
        if not allowlist:
            return chunks

        # ── Pass 1: type allowlist ────────────────────────────
        filtered = [
            c for c in chunks
            if c.metadata.get("type", "") in allowlist
        ]

        # Safety fallback
        if len(filtered) < _MIN_POOL:
            return chunks

        # ── Pass 2: dimension sub-filter ──────────────────────
        # When breakdown_by is known, prefer chunks for that dimension.
        # Does NOT hard-exclude other dimensions — just boosts recall
        # by keeping dimension-matched chunks + non-dimension chunks.
        if breakdown_by:
            dim_matched = [
                c for c in filtered
                if c.metadata.get("dimension") == breakdown_by
                or c.metadata.get("dimension") is None  # non-dimensional chunks
                or c.metadata.get("type") in ("kpi_snapshot", "filter_context",
                                               "cross_fact", "insight",
                                               "trend_overview", "anomaly_fact")
            ]
            if len(dim_matched) >= _MIN_POOL:
                filtered = dim_matched

        # ── Pass 3: grain sub-filter ──────────────────────────
        # For trend queries: prefer chunks matching the requested time grain.
        if grain and grain != "none" and intent == "kpi_trend":
            grain_matched = [
                c for c in filtered
                if c.metadata.get("grain") == grain
                or c.metadata.get("type") not in ("time_period_fact",
                                                    "trend_transition",
                                                    "trend_overview")
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