"""
rag/hyde.py  —  v2
────────────────────────────────────────────────────────────────────────────────
Hypothetical Document Expansion (template-based, no LLM required).

v1 vs v2 differences
─────────────────────
v1: simple keyword append
    "bleeding money" → "loss-making unprofitable negative profit sub-category"

v2 additions:
  1. Semantic phrase table  — maps *any* natural-language variant to canonical
     business vocabulary (35+ patterns vs 8 in v1)
  2. Multi-pass expansion   — runs ALL matching patterns, not just the first
  3. Intent-context suffix  — still fires when no pattern matched, but now
     only for queries that lack metric vocabulary AND are long enough
  4. Negation guard         — "which region is NOT losing money" should not
     expand to "loss-making" vocabulary

DESIGN INVARIANT
────────────────
Original query is always the prefix.  Expansion only ADDS terms.
No information is removed or overwritten.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


# ── Pattern table  ───────────────────────────────────────────────────────────
# Each entry: (regex pattern, canonical expansion string)
# Multiple patterns can match a single query — all expansions are appended.

_PARAPHRASE_PATTERNS: List[Tuple[str, str]] = [

    # ── Loss / negative profit ────────────────────────────────────────────
    (
        r'\b(bleeding|burning|wasting|draining|hemorrhaging)\s+(money|cash|revenue|profit|funds)\b',
        "loss-making unprofitable negative profit sub-category losing money",
    ),
    (
        r'\b(at\s+a\s+loss|in\s+the\s+red|losing\s+money|unprofitable|loss[\s-]making)\b',
        "negative profit loss-making sub-category unprofitable",
    ),
    (
        r'\b(negative\s+profit|profit\s+<\s*0|profit\s+is\s+negative|below\s+zero\s+profit)\b',
        "negative profit loss-making sub-category",
    ),
    (
        r'\b(which\s+(products?|items?|sub[\s-]?categories?)\s+(are\s+)?(hurting|dragging|pulling\s+down))\b',
        "loss-making negative profit unprofitable sub-category",
    ),

    # ── Revenue / sales synonyms ──────────────────────────────────────────
    (
        r'\b(income|earnings|turnover|proceeds|receipts|gross\s+revenue)\b',
        "sales revenue total",
    ),
    (
        r'\b(top[\s-]line|topline)\b',
        "sales revenue",
    ),

    # ── Geographic synonyms ───────────────────────────────────────────────
    (
        r'\b(area|zone|territory|district|market|geo|geography|location)\b',
        "region area geography",
    ),

    # ── Trend / change synonyms ───────────────────────────────────────────
    (
        r'\b(evolving|progressing|changing|shifting|moving|trajectory)\b',
        "trend growth year-over-year sales over time",
    ),
    (
        r'\b(over\s+time|through\s+the\s+(years?|months?|quarters?)|historically)\b',
        "trend time series year sales",
    ),
    (
        r'\b(momentum|acceleration|deceleration|growth\s+rate)\b',
        "trend growth rate year-over-year change",
    ),

    # ── Profit / profitability synonyms ──────────────────────────────────
    (
        r'\b(bottom[\s-]line|bottomline|net\s+income|net\s+earnings)\b',
        "profit profitability margin",
    ),
    (
        r'\b(margin\s+compression|margin\s+squeeze)\b',
        "profit margin decline decrease profitability",
    ),
    (
        r'\b(most\s+profitable|highest\s+margin|best\s+margin)\b',
        "profit margin highest top rank profitable",
    ),

    # ── "Make money" / earn patterns ─────────────────────────────────────
    (
        r'\b(make|makes|making|earn|earns|earning|generate[sd]?)\s+(the\s+)?(most|highest|best)\s+(money|profit|revenue|cash)\b',
        "highest profit revenue ranking top profitable",
    ),
    (
        r'\b(most|highest|best)\s+(money|cash)\b',
        "highest profit revenue ranking",
    ),

    # ── Discount / pricing ───────────────────────────────────────────────
    (
        r'\b(heavy\s+discount|deep\s+discount|slashing\s+prices|price\s+cut|over[\s-]?discounted)\b',
        "discount impact profit margin loss high discount",
    ),
    (
        r'\b(discount[s]?\s+(destroying|hurting|killing|eating)\s+(profit|margin|revenue))\b',
        "discount impact negative profit margin loss",
    ),
    (
        r'\b(discount[\s-]?(driven|related|caused)\s+loss)\b',
        "discount negative profit loss-making",
    ),

    # ── Transaction / order synonyms ─────────────────────────────────────
    (
        r'\b(transactions?|purchases?|invoices?|ticket[s]?|deals?)\b',
        "orders transactions count",
    ),
    (
        r'\b(how\s+many\s+orders?|order\s+count|order\s+volume)\b',
        "total orders count transactions",
    ),

    # ── Category / product synonyms ──────────────────────────────────────
    (
        r'\b(items?|products?|SKUs?|goods?|merchandise)\b',
        "sub_category product items",
    ),
    (
        r'\b(product\s+(line|lines?|group[s]?|mix))\b',
        "category sub_category product",
    ),

    # ── Underperformance ─────────────────────────────────────────────────
    (
        r'\b(underperform(ing)?|lagging|weak\s+(performer|region|segment)|poor\s+performance)\b',
        "underperform low profit negative profit rank bottom",
    ),
    (
        r'\b(drag(ging)?\s+(down|on)|pulling?\s+down|weighing?\s+on)\b',
        "low profit underperform negative drag",
    ),

    # ── Comparison / benchmarking ─────────────────────────────────────────
    (
        r'\b(year[\s-]over[\s-]year|yoy|same\s+period\s+last\s+year)\b',
        "year-over-year comparison previous period change",
    ),
    (
        r'\b(month[\s-]over[\s-]month|mom|vs\s+last\s+month)\b',
        "month-over-month comparison previous period change",
    ),

    # ── Concentration / distribution ─────────────────────────────────────
    (
        r'\b(distribution|breakdown|split|proportion|share)\b',
        "breakdown distribution percentage share",
    ),
    (
        r'\b(concentration|dominate|dominant|majority|bulk\s+of)\b',
        "top concentration highest percentage breakdown",
    ),

    # ── KPI / metric synonyms ─────────────────────────────────────────────
    (
        r'\b(kpi[s]?|key\s+metric[s]?|key\s+indicator[s]?|dashboard\s+metric[s]?)\b',
        "total sales profit orders profit_margin KPI",
    ),
    (
        r'\b(performance\s+(overview|summary|snapshot))\b',
        "total sales profit orders margin summary",
    ),
]

# ── Negation guard  ──────────────────────────────────────────────────────────
# When these precede a matched phrase, suppress loss/negative expansion.
_NEGATION_RE = re.compile(
    r'\b(not?|no\s+longer|without|except|exclude|isn\'?t|aren\'?t|doesn\'?t)\b',
    re.IGNORECASE,
)

# ── Loss-related patterns that should be guarded ────────────────────────────
_LOSS_PATTERN_INDICES = {0, 1, 2, 3}   # indices in _PARAPHRASE_PATTERNS


# ── Intent-level suffix (fallback) ───────────────────────────────────────────
_INTENT_SUFFIXES: dict = {
    "kpi_detail":  "loss-making negative profit unprofitable sub-category discount",
    "kpi_trend":   "trend growth rate year sales over time CAGR",
    "kpi_rank":    "top ranked highest profit sales region sub-category",
    "kpi_compare": "comparison period versus previous year change percent",
    "kpi_value":   "total sales profit revenue breakdown summary",
}

# Minimum query length to trigger intent suffix
_MIN_LEN_FOR_SUFFIX = 8

# Metric vocabulary — if ANY of these are present, intent suffix is skipped
_METRIC_RE = re.compile(
    r'\b(sales|profit|revenue|orders?|margin|loss|income|earn|discount|trend)\b',
    re.IGNORECASE,
)


class HyDEExpander:
    """
    Template-based query expander.
    Appends canonical business vocabulary to paraphrase queries so that
    dense embeddings land closer to the relevant chunk representations.
    """

    def expand(self, query: str, intent: Optional[str] = None) -> str:
        """
        Return augmented query string.

        Processing order:
          1. Multi-pass paraphrase pattern matching (all hits, with negation guard)
          2. Intent-level suffix (fires only when no patterns matched + vague vocab)
          3. Return original unchanged if nothing applies
        """
        if not query or not query.strip():
            return query

        additions: List[str] = []
        has_negation = bool(_NEGATION_RE.search(query))

        for idx, (pattern, expansion) in enumerate(_PARAPHRASE_PATTERNS):
            if re.search(pattern, query, re.IGNORECASE):
                # Guard: suppress loss-pattern expansions after negation
                if has_negation and idx in _LOSS_PATTERN_INDICES:
                    continue
                if expansion not in additions:   # dedup
                    additions.append(expansion)

        if additions:
            return f"{query} {' '.join(additions)}"

        # ── Fallback: intent-level suffix ─────────────────────────────────
        if (
            intent
            and intent in _INTENT_SUFFIXES
            and len(query.strip()) >= _MIN_LEN_FOR_SUFFIX
            and not _METRIC_RE.search(query)
        ):
            return f"{query} {_INTENT_SUFFIXES[intent]}"

        return query