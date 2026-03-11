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

    def _build_context_line(self) -> str:
        f = self.filters or {}
        parts = []
        dr = f.get("date_range")
        if dr and isinstance(dr, (tuple, list)) and len(dr) == 2:
            fmt = lambda d: d.strftime("%b %d, %Y") if hasattr(d, "strftime") else str(d)
            parts.append(f"from **{fmt(dr[0])}** to **{fmt(dr[1])}**")

        regions    = list(f.get("region")    or [])
        segments   = list(f.get("segment")   or [])
        categories = list(f.get("category")  or [])

        if 0 < len(regions) <= 2:
            parts.append(f"in **{', '.join(regions)}**")
        if 0 < len(segments) <= 2:
            parts.append(f"for **{', '.join(segments)}** segment")
        if 0 < len(categories) <= 2:
            parts.append(f"across **{', '.join(categories)}**")

        return " ".join(parts) if parts else ""

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

        ctx = self._build_context_line()

        if re.search(r"\b(kpi|summary|overview|dashboard)\b", ql):
            return (
                f"Here's a snapshot of your current business performance{' — ' + ctx if ctx else ''}:\n\n"
                f"- 💰 **Total Sales:** {money(ts)}\n"
                f"- 📈 **Total Profit:** {money(tp)}\n"
                f"- 📦 **Total Orders:** {to_:,}\n"
                f"- 📊 **Profit Margin:** {pm:.2f}%"
            )
        if re.search(r"\b(total\s+sales|sales\s+total|revenue)\b", ql):
            return (
                f"Total revenue came in at **{money(ts)}**"
                + (f", generated {ctx}" if ctx else "")
                + f". That reflects **{to_:,} orders** at an average of "
                + f"**{money(ts / to_ if to_ else 0)}** per order."
            )
        if re.search(r"\b(total\s+profit|profit\s+total)\b", ql):
            health = "healthy" if pm >= 10 else "tight"
            return (
                f"Total profit reached **{money(tp)}**"
                + (f", recorded {ctx}" if ctx else "")
                + f" — representing a **{pm:.2f}% margin** on **{money(ts)}** in revenue."
                f" Profitability looks **{health}** for this period."
            )
        if re.search(r"\b(total\s+orders|orders\s+total|number\s+of\s+orders)\b", ql):
            avg_val = ts / to_ if to_ else 0
            return (
                f"A total of **{to_:,} orders** were placed"
                + (f" {ctx}" if ctx else "")
                + f", with an average order value of **{money(avg_val)}**."
                f" These orders generated **{money(ts)}** in revenue and **{money(tp)}** in profit."
            )
        if re.search(r"\b(profit\s+margin|margin)\b", ql):
            benchmark = "above" if pm >= 12 else "below"
            return (
                f"The profit margin stands at **{pm:.2f}%**"
                + (f" {ctx}" if ctx else "")
                + f", with **{money(tp)}** profit on **{money(ts)}** revenue."
                f" This is **{benchmark} the typical 12% retail benchmark**."
            )
        return None

    # ── Tier 2: rule-based plan ───────────────────────────────

    def rule_based_plan(self, q: str) -> Optional[Dict[str, Any]]:
        """Parse simple NL into a plan dict — no LLM required."""
        ql = (q or "").lower().strip()
        s0, e0 = self._date_range()

        plan: Dict[str, Any] = {
            "intent": "kpi_value", "metrics": ["sales"], "time_grain": "none",
            "breakdown_by": None, "start_date": s0, "end_date": e0,
            "compare_period": None, "top_k": None, "order_by": "sales",
            "filters": {"region": [], "segment": [], "category": [], "sub_category": []},
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

        # ── Sub-category value detection (no keyword needed) ─────
        # User can name sub_category values directly without saying "by sub_category"
        # e.g. "profit of envelopes and paper and copiers"
        # Detect this BEFORE the intent-building so it can set detected_breakdown.
        if detected_breakdown is None and not self.df.empty and "sub_category" in self.df.columns:
            all_sub_cats = [
                str(v) for v in self.df["sub_category"].dropna().unique().tolist()
            ]
            mentioned_sub = [v for v in all_sub_cats if v.lower() in ql]
            if mentioned_sub and len(mentioned_sub) < len(all_sub_cats):
                detected_breakdown = "sub_category"

        # ── Same logic for region/segment/category without keyword ─
        if detected_breakdown is None:
            f = self.filters or {}
            for dim, all_vals in [
                ("region",   list(f.get("region")   or [])),
                ("segment",  list(f.get("segment")  or [])),
                ("category", list(f.get("category") or [])),
            ]:
                mentioned = [v for v in all_vals if v.lower() in ql]
                if mentioned and len(mentioned) < len(all_vals):
                    detected_breakdown = dim
                    break

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

        # ✅ FIX: also inject filters on the rule-based path
        return self._inject_mentioned_filters(plan, q)

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
        plan = self._extract_json((getattr(resp, "text", None) or "").strip())
        # ✅ FIX 2: _inject_mentioned_filters now always overrides (see method below)
        return self._inject_mentioned_filters(plan, q)

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

        === CURRENT DASHBOARD FILTERS (available values only) ===
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
        - IMPORTANT: If the user explicitly names specific dimension values (e.g. "west, east and south"):
          (a) set filters to ONLY those named values — do NOT include unlisted ones.
          (b) ALSO set breakdown_by to that dimension so each value gets its own row.
          Example: "total sales of west, east and south" →
            breakdown_by="region", filters.region=["West","East","South"]

        === JSON SCHEMA ===
        {{"intent":"kpi_value","metrics":["sales"],"time_grain":"none","breakdown_by":null,"start_date":"{s0}","end_date":"{e0}","compare_period":null,"top_k":null,"order_by":"sales","filters":{{"region":[],"segment":[],"category":[]}}}}

        USER QUESTION: {q}

        Return ONLY the JSON object:""".strip()

    def _inject_mentioned_filters(self, plan: Dict[str, Any], q: str) -> Dict[str, Any]:
        """
        Detects explicit dimension values named in the user's question and:
          1. Injects them as SQL filters (narrows the WHERE clause)
          2. Sets breakdown_by to that dimension if Gemini left it null

        Handles: region, segment, category (from dashboard sidebar filters)
                 sub_category (extracted live from self.df)
        """
        ql = q.lower()
        f  = self.filters or {}

        all_regions    = list(f.get("region")    or [])
        all_segments   = list(f.get("segment")   or [])
        all_categories = list(f.get("category")  or [])

        # sub_category values come from the loaded DataFrame, not sidebar filters
        all_sub_categories: List[str] = []
        if not self.df.empty and "sub_category" in self.df.columns:
            all_sub_categories = [
                str(v) for v in self.df["sub_category"].dropna().unique().tolist()
            ]

        filters = plan.get("filters") or {
            "region": [], "segment": [], "category": [], "sub_category": []
        }
        if "sub_category" not in filters:
            filters["sub_category"] = []

        _DIM_MAP = {
            "region":       all_regions,
            "segment":      all_segments,
            "category":     all_categories,
            "sub_category": all_sub_categories,
        }

        for key, all_vals in _DIM_MAP.items():
            if not all_vals:
                continue
            mentioned = [v for v in all_vals if v.lower() in ql]
            # Narrow filter when user explicitly named a STRICT SUBSET
            if mentioned and len(mentioned) < len(all_vals):
                filters[key] = mentioned
                # Infer breakdown_by when Gemini left it null
                if plan.get("breakdown_by") is None and plan.get("intent") in (
                    "kpi_value", "kpi_trend", None
                ):
                    plan["breakdown_by"] = key

        plan["filters"] = filters
        return plan

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