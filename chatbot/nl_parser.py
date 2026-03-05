"""
chatbot/nl_parser.py
Natural-language → plan-dict parsing.
Three tiers (fastest to slowest):
  1. Fast KPI path   — regex, no LLM, for simple total/margin questions
  2. Rule-based path — keyword matching, no LLM
  3. Gemini path     — full LLM with RAG-grounded prompt
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd
from google.genai import types as genai_types

if TYPE_CHECKING:
    from rag.engine import RAGContext


# ── Constants ─────────────────────────────────────────────────

_METRICS    = {"sales", "profit", "orders", "profit_margin"}
_BREAKDOWNS = {"region", "segment", "category", "sub_category"}
_GRAINS     = {"none", "week", "month", "quarter", "year"}

_METRIC_KEYWORDS: Dict[str, str] = {
    "sale": "sales", "sales": "sales", "revenue": "sales", "income": "sales",
    "profit": "profit", "margin": "profit_margin", "profit margin": "profit_margin",
    "profitability": "profit_margin",
    "order": "orders", "orders": "orders", "transaction": "orders",
}
_BREAKDOWN_KEYWORDS: Dict[str, str] = {
    "region": "region", "regions": "region",
    "segment": "segment", "segments": "segment",
    "category": "category", "categories": "category",
    "sub-category": "sub_category", "subcategory": "sub_category",
    "sub_category": "sub_category", "sub category": "sub_category",
    "subcategories": "sub_category", "sub-categories": "sub_category",
    "product": "sub_category",
}
_GRAIN_KEYWORDS: Dict[str, str] = {
    "daily": "week", "weekly": "week", "week": "week",
    "monthly": "month", "month": "month",
    "quarterly": "quarter", "quarter": "quarter",
    "yearly": "year", "annual": "year", "year": "year",
}
_COMPARE_KEYWORDS: Dict[str, str] = {
    "yoy": "yoy", "year over year": "yoy", "year-over-year": "yoy",
    "mom": "mom", "month over month": "mom", "month-over-month": "mom",
    "previous period": "prev_period", "prior period": "prev_period",
    "last period": "prev_period", "vs previous": "prev_period",
}

_BREAKDOWN_RE = re.compile(
    r"\b(by|per|across|for each|breakdown|group by|split by)\s+"
    r"(region|segment|category|sub.?category|product|state|city|ship\s*mode)\b",
    re.IGNORECASE,
)
_TOP_RE      = re.compile(r"\btop[\s-]?(\d+)\b", re.IGNORECASE)
_COMPARE_RE  = re.compile(r"\b(compare|vs|versus|compared|growth|change|difference)\b")
_TIME_YEAR   = re.compile(r"\b(20\d{2})\b")
_TIME_YM     = re.compile(r"\b(\d{4}[-/](0?[1-9]|1[0-2]))\b")
_TIME_GRAIN_RE = re.compile(
    r"\b(q[1-4]|quarter|month|year|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b"
)
_TIME_REL    = re.compile(r"\b(in|during|between|from|since|until|last|previous|this)\b")


class NLParser:
    """
    Three-tier NL → plan parser. Callers provide shared state (df, filters, gemini).
    """

    def __init__(self, df: pd.DataFrame, filters: Dict[str, Any],
                 gemini_client: Any = None, model_name: str = "") -> None:
        self.df           = df
        self.filters      = filters
        self.gemini_client = gemini_client
        self.model_name   = model_name or ""

    # ── Tier 1: fast KPI ─────────────────────────────────────

    def fast_kpi_answer(self, q: str) -> Optional[str]:
        """Return a plain-text answer for simple KPI totals — no DB round-trip needed."""
        ql = (q or "").strip().lower()
        if not ql or self._has_breakdown_hint(q) or self._has_time_hint(q):
            return None

        kpis = self._compute_kpis()
        ts, tp = float(kpis["total_sales"]), float(kpis["total_profit"])
        to_    = int(kpis["total_orders"])
        pm     = float(kpis["profit_margin"])

        def money(x: float) -> str: return f"${x:,.0f}"

        if re.search(r"\b(kpi|summary|overview|dashboard)\b", ql):
            return (
                "Here's a quick overview of your current KPIs:\n\n"
                f"- **Total Sales:** {money(ts)}\n"
                f"- **Total Profit:** {money(tp)}\n"
                f"- **Total Orders:** {to_:,}\n"
                f"- **Profit Margin:** {pm:.2f}%"
            )
        if re.search(r"\b(total\s+sales|sales\s+total|revenue)\b", ql):
            return f"Total sales came in at **{money(ts)}**."
        if re.search(r"\b(total\s+profit|profit\s+total)\b", ql):
            return f"Total profit reached **{money(tp)}**."
        if re.search(r"\b(total\s+orders|orders\s+total|number\s+of\s+orders)\b", ql):
            return f"The total number of orders was **{to_:,}**."
        if re.search(r"\b(profit\s+margin|margin)\b", ql):
            return f"The profit margin stood at **{pm:.2f}%**."
        return None

    # ── Tier 2: rule-based plan ───────────────────────────────

    def rule_based_plan(self, q: str) -> Optional[Dict[str, Any]]:
        """Parse simple NL into a plan dict — no LLM required."""
        ql = (q or "").lower().strip()
        s0, e0 = self._date_range()
        regions, segments, categories = self._filter_lists()

        plan: Dict[str, Any] = {
            "intent": "kpi_value", "metrics": ["sales"], "time_grain": "none",
            "breakdown_by": None, "start_date": s0, "end_date": e0,
            "compare_period": None, "top_k": None, "order_by": "sales",
            "filters": {"region": [], "segment": [], "category": []},
        }

        # Metric
        detected_metric = "sales"
        for kw, metric in sorted(_METRIC_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_metric = metric
                break
        plan["metrics"]  = [detected_metric]
        plan["order_by"] = detected_metric

        # Breakdown
        detected_breakdown: Optional[str] = None
        for kw, dim in sorted(_BREAKDOWN_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_breakdown = dim
                break
        if detected_breakdown is None:
            m = _BREAKDOWN_RE.search(q)
            if m:
                raw = m.group(2).lower().replace(" ", "_").replace("-", "_")
                if "sub" in raw:
                    raw = "sub_category"
                if raw in _BREAKDOWNS:
                    detected_breakdown = raw

        # Top-k
        top_match = _TOP_RE.search(ql)
        top_k: Optional[int] = int(top_match.group(1)) if top_match else None

        # Time grain
        detected_grain = "none"
        for kw, grain in sorted(_GRAIN_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_grain = grain
                break

        # Compare period
        detected_compare: Optional[str] = None
        if _COMPARE_RE.search(ql):
            for kw, cp in sorted(_COMPARE_KEYWORDS.items(), key=lambda x: -len(x[0])):
                if kw in ql:
                    detected_compare = cp
                    break
            if detected_compare is None:
                detected_compare = "prev_period"

        # Build intent
        if detected_compare:
            plan.update(intent="kpi_compare", compare_period=detected_compare)
        elif top_k and detected_breakdown:
            plan.update(intent="kpi_rank", breakdown_by=detected_breakdown, top_k=top_k)
        elif detected_grain != "none":
            plan.update(intent="kpi_trend", time_grain=detected_grain, breakdown_by=detected_breakdown)
        elif detected_breakdown:
            plan.update(intent="kpi_value", breakdown_by=detected_breakdown)
        else:
            return None  # nothing recognisable — let Gemini handle it

        return plan

    # ── Tier 3: Gemini plan ───────────────────────────────────

    def gemini_plan(self, q: str, rag_context: "RAGContext") -> Dict[str, Any]:
        if not self.gemini_client:
            raise RuntimeError("Gemini API key not configured.")
        prompt = self._gemini_prompt(q, rag_context)
        resp = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=400),
        )
        return self._extract_json((getattr(resp, "text", None) or "").strip())

    # ── Helpers ───────────────────────────────────────────────

    def _has_breakdown_hint(self, q: str) -> bool:
        return bool(_BREAKDOWN_RE.search(q))

    def _has_time_hint(self, q: str) -> bool:
        ql = q.lower()
        return bool(
            _TIME_YEAR.search(ql) or _TIME_YM.search(ql)
            or _TIME_GRAIN_RE.search(ql) or _TIME_REL.search(ql)
        )

    def _compute_kpis(self) -> Dict[str, Any]:
        if self.df.empty:
            return {"total_sales": 0, "total_profit": 0, "total_orders": 0, "profit_margin": 0}
        ts = float(self.df["sales"].sum())
        tp = float(self.df["profit"].sum())
        to_ = int(self.df["order_id"].nunique())
        pm = (tp / ts * 100) if ts else 0
        return {"total_sales": ts, "total_profit": tp, "total_orders": to_, "profit_margin": pm}

    def _filter_lists(self):
        f = self.filters or {}
        return list(f.get("region") or []), list(f.get("segment") or []), list(f.get("category") or [])

    def _date_range(self):
        f = self.filters or {}
        dr = f.get("date_range")
        if dr and isinstance(dr, (tuple, list)) and len(dr) == 2:
            s, e = dr
            fmt = lambda d: d.strftime("%Y-%m-%d") if isinstance(d, (date, datetime)) else str(d)
            return fmt(s), fmt(e)
        if "order_date" in self.df.columns and not self.df.empty:
            s = pd.to_datetime(self.df["order_date"].min()).date().strftime("%Y-%m-%d")
            e = pd.to_datetime(self.df["order_date"].max()).date().strftime("%Y-%m-%d")
            return s, e
        return "1900-01-01", "2100-01-01"

    def _gemini_prompt(self, q: str, rag_context: "RAGContext") -> str:
        regions, segments, categories = self._filter_lists()
        s0, e0 = self._date_range()
        rag_section = rag_context.as_prompt_section(max_chunks=7)
        return f"""
You are a STRICT JSON query planner for a Postgres "Superstore" analytics dataset.
Output ONLY one valid JSON object. No markdown, no SQL, no explanations.

=== VERIFIED DATA FACTS ===
{rag_section}

=== SUPPORTED INTENTS ===
kpi_value | kpi_trend | kpi_rank | kpi_compare | clarify

=== SUPPORTED VALUES ===
metrics: sales | profit | orders | profit_margin
time_grain: none | week | month | quarter | year
breakdown_by: null | region | segment | category | sub_category
compare_period: null | prev_period | mom | yoy

=== CURRENT DASHBOARD FILTERS ===
regions={json.dumps(regions)}
segments={json.dumps(segments)}
categories={json.dumps(categories)}
default_start="{s0}", default_end="{e0}"

=== RULES ===
- filters values must be ONLY from current selections above ([] = no extra filter)
- intent="clarify" when question cannot be answered or is ambiguous
- kpi_compare: metrics must have exactly 1 item
- kpi_rank: requires breakdown_by AND top_k; time_grain must be "none"
- If intent="clarify": add "clarifying_question": "<one question>"

=== JSON SCHEMA ===
{{"intent":"kpi_value","metrics":["sales"],"time_grain":"none","breakdown_by":null,"start_date":"{s0}","end_date":"{e0}","compare_period":null,"top_k":null,"order_by":"sales","filters":{{"region":[],"segment":[],"category":[]}}}}

USER QUESTION: {q}

Return ONLY the JSON object:""".strip()

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        for attempt in (text, re.sub(r"```(?:json)?", "", text).strip()):
            try:
                return json.loads(attempt)
            except Exception:
                pass
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Gemini did not return valid JSON.")
