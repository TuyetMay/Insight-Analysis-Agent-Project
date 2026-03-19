"""
chatbot/nl_parser.py
Natural-language → plan-dict parsing.
Three tiers (fastest to slowest):
  1. Fast KPI path   — regex, no LLM, for simple total/margin questions
  2. Rule-based path — keyword matching, no LLM
  3. Gemini path     — full LLM with RAG-grounded prompt
"""

from __future__ import annotations

import calendar
import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
from google.genai import types as genai_types

if TYPE_CHECKING:
    from rag.engine import RAGContext


# ── Constants ─────────────────────────────────────────────────

_METRICS    = {"sales", "profit", "orders", "profit_margin"}
_BREAKDOWNS = {"region", "segment", "category", "sub_category", "state"}
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
    "state": "state", "states": "state", "province": "state",
}
_GRAIN_KEYWORDS: Dict[str, str] = {
    "daily": "week", "weekly": "week", "week": "week",
    "monthly": "month", "month": "month",
    "quarterly": "quarter", "quarter": "quarter",
    "yearly": "year", "annual": "year", "year": "year",
}
_COMPARE_KEYWORDS: Dict[str, str] = {
    "yoy": "yoy", "year over year": "yoy", "year-over-year": "yoy",
    "same period last year": "yoy", "same period": "yoy",   # ← add these
    "same time last year": "yoy", "prior year": "yoy", "last year": "yoy",
    "mom": "mom", "month over month": "mom", "month-over-month": "mom",
    "previous period": "prev_period", "prior period": "prev_period",
    "last period": "prev_period", "vs previous": "prev_period",
}

# ── NEW: Negative-profit / drill-down detection ───────────────
# Matches queries like: "which orders have negative profit",
# "show loss-making products", "what is generating losses", etc.
_DETAIL_NEGATIVE_RE = re.compile(
    r"\b("
    r"negative\s+profit|loss[\s-]?making|at\s+a\s+loss|generating\s+loss"
    r"|which\s+orders|which\s+products|which\s+items"
    r"|losing\s+money|unprofitable|loss\s+orders"
    r"|negative\s+margin|profit\s+<\s*0|profit\s+is\s+negative"
    r"|what.*generating.*loss|what.*causing.*loss"
    r")\b",
    re.IGNORECASE,
)

# Detects "highest and lowest", "best and worst", "top and bottom" etc.
_EXTREMES_RE = re.compile(
    r"\b(highest.*lowest|lowest.*highest|best.*worst|worst.*best"
    r"|top.*bottom|bottom.*top|maximum.*minimum|min.*max"
    r"|most.*least|least.*most)\b",
    re.IGNORECASE,
)

# Month name → number
_MONTH_MAP: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
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

_MONTH_YEAR_RE = re.compile(
    r"\b(?:("
    + "|".join(_MONTH_MAP.keys())
    + r")\s+(20\d{2})|(20\d{2})\s+("
    + "|".join(_MONTH_MAP.keys())
    + r"))\b",
    re.IGNORECASE,
)
_QUARTER_YEAR_RE = re.compile(r"\bq([1-4])\s*(20\d{2})\b", re.IGNORECASE)
_YEAR_ONLY_RE    = re.compile(r"\bin\s+(20\d{2})\b|\bfor\s+(20\d{2})\b|\b(20\d{2})\b")

# Relative date patterns: "past 30 days", "last 2 weeks", "this month"
_RELATIVE_DATE_RE = re.compile(
    r"\b(?:past|last|previous|over\s+(?:the\s+)?past)\s+(\d+)\s*(day|week|month|year)s?\b"
    r"|\b(this|current)\s+(week|month|quarter|year)\b",
    re.IGNORECASE,
)

# Growth/rank keywords for "which X has strongest growth"
_GROWTH_RANK_RE = re.compile(
    r"\b(strongest|highest|most|best|biggest|largest|fastest)\s+"
    r"(growth|increase|improvement|gain|rise|growing)\b"
    r"|\b(growth|increase|improvement)\s+(?:by|per|across|for each)\b",
    re.IGNORECASE,
)

# Numeric date patterns
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_DATE_RANGE_SEP_RE = re.compile(
    r"\b(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4})\s*(?:to|until|–|—|-{1,2})\s*(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
    re.IGNORECASE,
)


def _parse_numeric_date(raw: str) -> Optional[date]:
    raw = raw.strip()
    normalised = re.sub(r"[/.]", "-", raw)
    parts = normalised.split("-")
    if len(parts) != 3:
        return None
    a, b, c = parts
    if len(a) == 4:
        try:
            return datetime.strptime(f"{a}-{b.zfill(2)}-{c.zfill(2)}", "%Y-%m-%d").date()
        except ValueError:
            return None
    if len(c) == 4:
        ai, bi, ci = int(a), int(b), int(c)
        if 1 <= ai <= 31 and 1 <= bi <= 12:
            try:
                return date(ci, bi, ai)
            except ValueError:
                pass
        if 1 <= ai <= 12 and 1 <= bi <= 31:
            try:
                return date(ci, ai, bi)
            except ValueError:
                pass
    return None


class NLParser:
    """Three-tier NL → plan parser."""

    def __init__(self, df: pd.DataFrame, filters: Dict[str, Any],
                 gemini_client: Any = None, model_name: str = "") -> None:
        self.df            = df
        self.filters       = filters
        self.gemini_client = gemini_client
        self.model_name    = model_name or ""

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
        """Return instant KPI answer — skip if it's a detail/drill-down query."""
        ql = (q or "").strip().lower()
        # Never short-circuit detail queries
        if _DETAIL_NEGATIVE_RE.search(q):
            return None
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
        ql = (q or "").lower().strip()
        s0, e0 = self._date_range()

        # ── NEW: Detect negative-profit drill-down ────────────
        if _DETAIL_NEGATIVE_RE.search(q):
            explicit_dates = self._extract_date_range(ql)
            if explicit_dates:
                s0, e0 = explicit_dates
            # Detect breakdown preference
            breakdown = "sub_category"
            for kw, dim in sorted(_BREAKDOWN_KEYWORDS.items(), key=lambda x: -len(x[0])):
                if kw in ql:
                    breakdown = dim
                    break
            plan = {
                "intent":         "kpi_detail",
                "condition":      "profit_negative",
                "metrics":        ["sales", "profit"],
                "time_grain":     "none",
                "breakdown_by":   breakdown,
                "start_date":     s0,
                "end_date":       e0,
                "compare_period": None,
                "top_k":          15,
                "order_by":       "profit",
                "filters":        {"region": [], "segment": [], "category": [], "sub_category": []},
            }
            return self._inject_mentioned_filters(plan, q)

        explicit_dates = self._extract_date_range(ql) or self._extract_relative_date(ql)
        if explicit_dates:
            s0, e0 = explicit_dates

        show_extremes = bool(_EXTREMES_RE.search(q))
        is_growth_rank = bool(_GROWTH_RANK_RE.search(q))

        plan: Dict[str, Any] = {
            "intent": "kpi_value", "metrics": ["sales"], "time_grain": "none",
            "breakdown_by": None, "start_date": s0, "end_date": e0,
            "compare_period": None, "top_k": None, "order_by": "sales",
            "show_extremes": show_extremes,
            "filters": {"region": [], "segment": [], "category": [], "sub_category": []},
        }

        detected_metrics: List[str] = []
        for kw, metric in sorted(_METRIC_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql and metric not in detected_metrics:
                detected_metrics.append(metric)
        if not detected_metrics:
            detected_metrics = ["sales"]
        detected_metrics = detected_metrics[:2]

        plan["metrics"]  = detected_metrics
        plan["order_by"] = detected_metrics[0]

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

        top_match = _TOP_RE.search(ql)
        top_k: Optional[int] = int(top_match.group(1)) if top_match else None

        detected_grain = "none"
        for kw, grain in sorted(_GRAIN_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_grain = grain
                break

        detected_compare: Optional[str] = None
        if _COMPARE_RE.search(ql):
            for kw, cp in sorted(_COMPARE_KEYWORDS.items(), key=lambda x: -len(x[0])):
                if kw in ql:
                    detected_compare = cp
                    break
            if detected_compare is None:
                detected_compare = "prev_period"

        if detected_breakdown is None and not self.df.empty and "sub_category" in self.df.columns:
            all_sub_cats = [str(v) for v in self.df["sub_category"].dropna().unique().tolist()]
            mentioned_sub = [v for v in all_sub_cats if v.lower() in ql]
            if mentioned_sub and len(mentioned_sub) < len(all_sub_cats):
                detected_breakdown = "sub_category"

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

        # ── Growth rank: "which state/region has strongest growth" ──
        # Build a kpi_compare plan but ranked by breakdown — simulate by
        # returning a kpi_trend so the user sees period change per dimension.
        # Best approximation: kpi_rank over current period vs previous, broken by dimension.
        if is_growth_rank and detected_breakdown:
            # Use top-10 by default for growth ranking
            plan.update(
                intent="kpi_rank",
                breakdown_by=detected_breakdown,
                top_k=top_k or 10,
                metrics=["sales"],
                order_by="sales",
            )
            # If no explicit date range, use last 30 days relative to data
            if not explicit_dates:
                rel = self._extract_relative_date("past 30 days")
                if rel:
                    plan["start_date"], plan["end_date"] = rel
            result = self._inject_mentioned_filters(plan, q)
            # Clear the breakdown dimension filter
            if detected_breakdown in ("region", "segment", "category", "state"):
                ql_check = q.lower()
                f_check = self.filters or {}
                all_vals = list(f_check.get(detected_breakdown, []) or [])
                import re as _re
                mentioned = [v for v in all_vals if _re.search(r"\b" + _re.escape(v.lower()) + r"\b", ql_check)]
                if not mentioned:
                    result["filters"][detected_breakdown] = []
            return result

        if detected_compare:
            if detected_breakdown:
                # "Which REGION is underperforming vs same period?" → compare per region
                plan.update(
                    intent="kpi_compare",
                    compare_period=detected_compare,
                    breakdown_by=detected_breakdown,
                    metrics=[detected_metrics[0]],
                )
            else:
                plan.update(
                    intent="kpi_compare",
                    compare_period=detected_compare,
                    metrics=[detected_metrics[0]],
                )
        elif top_k and detected_breakdown:
            plan.update(intent="kpi_rank", breakdown_by=detected_breakdown, top_k=top_k)
        elif detected_grain != "none":
            plan.update(intent="kpi_trend", time_grain=detected_grain,
                        breakdown_by=detected_breakdown)
        elif detected_breakdown:
            plan.update(intent="kpi_value", breakdown_by=detected_breakdown)
        elif explicit_dates or len(detected_metrics) > 1:
            plan.update(intent="kpi_value")
        else:
            return None

        # ── When comparing across a dimension (show_extremes or breakdown query),
        # clear that dimension's filter so ALL values are returned, not just
        # the ones currently selected in the sidebar.
        plan = self._inject_mentioned_filters(plan, q)

        if plan.get("breakdown_by") and plan.get("intent") == "kpi_value":
            dim = plan["breakdown_by"]
            # Only clear if filter was set to a subset (i.e. restricting results)
            # This ensures "sales by region" returns ALL regions even if sidebar filters
            f_copy = self.filters or {}
            all_vals = list(f_copy.get(dim, []) or [])
            current_filter = plan.get("filters", {}).get(dim, [])
            # If current filter equals all available values or is a subset,
            # clear it so the breakdown shows everything
            if dim in ("region", "segment", "category") and len(all_vals) > 0:
                # If no explicit mention of specific values (i.e. not "show me West and East"),
                # clear the dimension filter to get all values
                ql_check = q.lower()
                mentioned_specific = [
                        v for v in all_vals
                        if re.search(r"\b" + re.escape(v.lower()) + r"\b", ql_check)
                    ]
                if not mentioned_specific:
                    plan["filters"][dim] = []

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
        plan = self._extract_json((getattr(resp, "text", None) or "").strip())
        return self._inject_mentioned_filters(plan, q)

    # ── Date extraction ───────────────────────────────────────

    def _extract_date_range(self, ql: str) -> Optional[Tuple[str, str]]:
        range_match = _DATE_RANGE_SEP_RE.search(ql)
        if range_match:
            d1 = _parse_numeric_date(range_match.group(1))
            d2 = _parse_numeric_date(range_match.group(2))
            if d1 and d2:
                start, end = (d1, d2) if d1 <= d2 else (d2, d1)
                return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        all_numeric = _NUMERIC_DATE_RE.finditer(ql)
        parsed_dates = []
        for m in all_numeric:
            d = _parse_numeric_date(m.group(0))
            if d:
                parsed_dates.append(d)

        if len(parsed_dates) == 2:
            start, end = (parsed_dates[0], parsed_dates[1]) \
                if parsed_dates[0] <= parsed_dates[1] \
                else (parsed_dates[1], parsed_dates[0])
            return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        if len(parsed_dates) == 1:
            d = parsed_dates[0]
            return d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")

        m = _MONTH_YEAR_RE.search(ql)
        if m:
            if m.group(1) and m.group(2):
                month_name, year = m.group(1).lower(), int(m.group(2))
            else:
                month_name, year = m.group(4).lower(), int(m.group(3))
            month_num = _MONTH_MAP.get(month_name)
            if month_num:
                last_day = calendar.monthrange(year, month_num)[1]
                return f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day:02d}"

        m = _QUARTER_YEAR_RE.search(ql)
        if m:
            q_num, year = int(m.group(1)), int(m.group(2))
            qs = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
            qe = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
            sm, sd = qs[q_num]; em, ed = qe[q_num]
            return f"{year}-{sm:02d}-{sd:02d}", f"{year}-{em:02d}-{ed:02d}"

        year_matches = _TIME_YEAR.findall(ql)
        if len(year_matches) == 1:
            year = int(year_matches[0])
            return f"{year}-01-01", f"{year}-12-31"

        return None

    def _extract_relative_date(self, ql: str):
        """
        Parse relative expressions like 'past 30 days', 'last 2 weeks', 'last month',
        'this month' into (start, end) using the DataFrame's max date as 'today'.
        """
        from datetime import timedelta

        if "order_date" in self.df.columns and not self.df.empty:
            today = pd.to_datetime(self.df["order_date"]).max().date()
        else:
            from datetime import date as _d
            today = _d.today()

        m = _RELATIVE_DATE_RE.search(ql)
        if not m:
            return None

        # Group layout (3 alternatives in the regex):
        # Alt 1 "past 30 days"  → groups 1,2 = n, unit
        # Alt 2 "last month"    → group 3 = unit (n=1 implied)
        # Alt 3 "this month"    → groups 4,5 = keyword, grain
        if m.group(1) and m.group(2):
            n, unit = int(m.group(1)), m.group(2).lower()
        elif m.group(3):
            n, unit = 1, m.group(3).lower()
        elif m.group(4) and m.group(5):
            grain = m.group(5).lower()
            ts = pd.Timestamp(today)
            if grain == "week":
                start = (ts - pd.tseries.offsets.Week(weekday=0)).date()
            elif grain == "month":
                start = today.replace(day=1)
            elif grain == "quarter":
                q_month = ((today.month - 1) // 3) * 3 + 1
                start = today.replace(month=q_month, day=1)
            elif grain == "year":
                start = today.replace(month=1, day=1)
            else:
                return None
            return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        else:
            return None

        if unit == "day":
            start = today - timedelta(days=n)
        elif unit == "week":
            start = today - timedelta(weeks=n)
        elif unit == "month":
            start = (pd.Timestamp(today) - pd.DateOffset(months=n)).date()
        elif unit == "year":
            start = (pd.Timestamp(today) - pd.DateOffset(years=n)).date()
        else:
            return None
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")


    # ── Helpers ───────────────────────────────────────────────

    def _has_breakdown_hint(self, q: str) -> bool:
        return bool(_BREAKDOWN_RE.search(q))

    def _has_time_hint(self, q: str) -> bool:
        ql = q.lower()
        return bool(
            _TIME_YEAR.search(ql) or _TIME_YM.search(ql)
            or _TIME_GRAIN_RE.search(ql) or _TIME_REL.search(ql)
            or _NUMERIC_DATE_RE.search(ql)
        )

    def _compute_kpis(self) -> Dict[str, Any]:
        if self.df.empty:
            return {"total_sales": 0, "total_profit": 0, "total_orders": 0, "profit_margin": 0}
        ts  = float(self.df["sales"].sum())
        tp  = float(self.df["profit"].sum())
        to_ = int(self.df["order_id"].nunique())
        pm  = (tp / ts * 100) if ts else 0
        return {"total_sales": ts, "total_profit": tp, "total_orders": to_, "profit_margin": pm}

    def _filter_lists(self):
        f = self.filters or {}
        return list(f.get("region") or []), list(f.get("segment") or []), list(f.get("category") or [])

    def _date_range(self):
        f  = self.filters or {}
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
        kpi_value | kpi_trend | kpi_rank | kpi_compare | kpi_detail | clarify

        === SUPPORTED VALUES ===
        metrics: sales | profit | orders | profit_margin
        time_grain: none | week | month | quarter | year
        breakdown_by: null | region | segment | category | sub_category
        compare_period: null | prev_period | mom | yoy
        condition (only for kpi_detail): profit_negative | high_discount | loss_orders

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
        - kpi_detail: use when user asks "which orders/products have negative profit",
          "what is losing money", "show loss-making items" etc.
          Set condition="profit_negative", breakdown_by="sub_category"
        - If intent="clarify": add "clarifying_question": "<one question>"
        - If dates given in any format (DD/MM/YYYY etc.), parse to YYYY-MM-DD

        === JSON SCHEMA ===
        {{"intent":"kpi_value","metrics":["sales"],"time_grain":"none","breakdown_by":null,"start_date":"{s0}","end_date":"{e0}","compare_period":null,"top_k":null,"order_by":"sales","filters":{{"region":[],"segment":[],"category":[]}}}}

        USER QUESTION: {q}

        Return ONLY the JSON object:""".strip()

    def _inject_mentioned_filters(self, plan: Dict[str, Any], q: str) -> Dict[str, Any]:
        ql = q.lower()
        f  = self.filters or {}

        all_regions        = list(f.get("region")    or [])
        all_segments       = list(f.get("segment")   or [])
        all_categories     = list(f.get("category")  or [])
        all_sub_categories: List[str] = []
        if not self.df.empty and "sub_category" in self.df.columns:
            all_sub_categories = [str(v) for v in self.df["sub_category"].dropna().unique().tolist()]

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
            # Use word-boundary matching to avoid false positives like
            # "west" matching inside "lowest" or "east" inside "northeast"
            mentioned = [
                v for v in all_vals
                if re.search(r"\b" + re.escape(v.lower()) + r"\b", ql)
            ]
            if mentioned and len(mentioned) < len(all_vals):
                filters[key] = mentioned
                if plan.get("breakdown_by") is None and plan.get("intent") in (
                    "kpi_value", "kpi_trend", None
                ):
                    plan["breakdown_by"] = key

        plan["filters"] = filters
        return plan

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
        if matches:
            try:
                return json.loads(matches[-1].group(0))
            except Exception:
                pass
        raise ValueError("Gemini did not return valid JSON.")