"""
rag/hyde.py
Template-based HyDE (Hypothetical Document Embedding) expander.

No LLM required — uses regex pattern matching to detect paraphrases
and appends canonical business terms before embedding.

WHY:
  Dense embeddings fail on metaphorical queries:
    "bleeding money" → embedding far from "negative profit loss-making"
  Fix: append canonical terms → averaged embedding moves closer to chunk space.

DESIGN:
  HyDEExpander.expand(query, intent) → augmented_query_string
  The augmented string is passed to the retriever instead of raw query.
  Original query is preserved as prefix → no meaning is lost.
"""

from __future__ import annotations

import re
from typing import Optional


class HyDEExpander:
    """
    Template-based query expander.
    Appends canonical business vocabulary to paraphrase queries.
    """

    # ── Paraphrase → canonical expansion ─────────────────────
    # Each pattern maps to terms that appear verbatim in chunk text.
    # Order matters: more specific patterns first.
    _PARAPHRASE_PATTERNS: list = [
        # Loss / negative profit metaphors
        (
            r'\b(bleeding|burning|wasting|draining)\s+(money|cash|revenue|profit)\b',
            "loss-making unprofitable negative profit sub-category losing money"
        ),
        (
            r'\b(at\s+a\s+loss|in\s+the\s+red|losing\s+money|unprofitable|loss.making)\b',
            "negative profit loss-making sub-category unprofitable"
        ),
        # Revenue synonyms
        (
            r'\b(income|earnings|turnover|proceeds)\b',
            "sales revenue"
        ),
        # Geographic synonyms
        (
            r'\b(area|zone|market|territory|district)\b',
            "region area"
        ),
        # Growth/trend synonyms
        (
            r'\b(evolving|progressing|changing|moving|shifted)\b',
            "trend growth year-over-year sales"
        ),
        # "Make money" / "earn" → profit/revenue rank
        (
            r'\b(make|makes|making|earn|earns|earning)\s+(the\s+)?(most|highest|best)\s+(money|profit|revenue|cash)\b',
            "highest profit ranking top region profitable"
        ),
        # "Most money" without verb
        (
            r'\b(most|highest|best)\s+(money|cash|revenue|profit)\b',
            "highest profit ranking top profitable"
        ),
        # Discount / price cut metaphors
        (
            r'\b(heavy\s+discount|deep\s+discount|slashing\s+prices|price\s+cut)\b',
            "discount impact profit margin loss"
        ),
    ]

    # ── Intent-level suffix (fallback when no pattern matched) ─
    # Applied only when intent is known and query is short/vague.
    _INTENT_SUFFIXES: dict = {
        "kpi_detail":  "loss-making negative profit unprofitable sub-category",
        "kpi_trend":   "trend growth year sales over time",
        "kpi_rank":    "top ranked highest profit sales region",
        "kpi_compare": "comparison period versus previous year change",
        "kpi_value":   "total sales profit revenue breakdown",
    }

    # Minimum query length to trigger intent suffix (avoid over-expanding short tokens)
    _MIN_LEN_FOR_INTENT_SUFFIX: int = 8

    def expand(self, query: str, intent: Optional[str] = None) -> str:
        """
        Return augmented query string.

        Strategy:
          1. Try paraphrase patterns first (most specific)
          2. Fall back to intent suffix if no pattern matched and query is vague
          3. Return original query unchanged if nothing applies

        The original query is always preserved as the prefix so its meaning
        is not lost — expansion only adds canonical terms.
        """
        if not query or not query.strip():
            return query

        # ── Pass 1: paraphrase pattern matching ───────────────
        additions: list = []
        for pattern, expansion in self._PARAPHRASE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                additions.append(expansion)

        if additions:
            return f"{query} {' '.join(additions)}"

        # ── Pass 2: intent-level fallback ─────────────────────
        if (intent
                and intent in self._INTENT_SUFFIXES
                and len(query.strip()) >= self._MIN_LEN_FOR_INTENT_SUFFIX):
            # Only apply if query uses vague/indirect vocabulary
            # (heuristic: no metric keyword found)
            metric_words = re.compile(
                r'\b(sales|profit|revenue|orders|margin|loss|income|earn)\b',
                re.IGNORECASE
            )
            if not metric_words.search(query):
                suffix = self._INTENT_SUFFIXES[intent]
                return f"{query} {suffix}"

        return query