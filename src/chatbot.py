# src/chatbot.py
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from google import genai
from google.genai import types as genai_types

from config import Config
from src.chatbot_suggest_question import (
    RAGSuggestionEngine,
    RuleBasedSuggestionEngine,
    Suggestion,
    SuggestionEngine,
)
from src.database import execute_query
from src.rag_engine import RAGEngine, RAGContext


class DashboardChatbot:
    _METRICS = {"sales", "profit", "orders", "profit_margin"}
    _TIME_GRAINS = {"none", "week", "month", "quarter", "year"}
    _DIM_FILTERS = {"region", "segment", "category", "sub_category"}
    _DIM_BREAKDOWNS = {"region", "segment", "category", "sub_category"}

    _BREAKDOWN_PATTERN = re.compile(
        r"\b(by|per|across|for each|breakdown|group by|split by)\s+"
        r"(region|segment|category|sub.?category|product|state|city|ship\s*mode)\b",
        re.IGNORECASE,
    )

    # Keyword maps for rule-based NL parsing
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
    _TOP_PATTERN = re.compile(r"\btop[\s-]?(\d+)\b", re.IGNORECASE)
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
        "compared to previous": "prev_period",
    }

    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any], filters: Dict[str, Any]):
        self.df = df.copy()
        self.kpis = kpis
        self.filters = filters
        self.last_plan: Optional[Dict[str, Any]] = None
        self._last_question: str = ""
        self._last_answer: str = ""

        if "order_date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["order_date"]):
            self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors="coerce")

        self.gemini_ready = False
        api_key = getattr(Config, "GOOGLE_API_KEY", "")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_name = getattr(Config, "GEMINI_MODEL", "gemini-1.5-flash")
            self.gemini_ready = True

        self.rag = RAGEngine()
        self.rag.build(df=self.df, kpis=self.kpis, filters=self.filters)

        self._rule_engine = RuleBasedSuggestionEngine(
            allowed_metrics=list(self._METRICS),
            allowed_breakdowns=list(self._DIM_BREAKDOWNS),
            allowed_compare_periods=["prev_period", "mom", "yoy"],
            max_suggestions=4,
        )
        self._rag_suggest_engine: Optional[RAGSuggestionEngine] = None
        if self.gemini_ready:
            self._rag_suggest_engine = RAGSuggestionEngine(
                gemini_client=self.client,
                model_name=self.model_name,
                rule_engine=self._rule_engine,
                max_suggestions=4,
            )

        self.suggestion_engine = SuggestionEngine(
            allowed_metrics=list(self._METRICS),
            allowed_breakdowns=list(self._DIM_BREAKDOWNS),
            allowed_compare_periods=["prev_period", "mom", "yoy"],
            max_suggestions=4,
        )

    # ─────────────────────────────────────────────────────────
    # Fast path
    # ─────────────────────────────────────────────────────────

    def _filter_summary(self) -> str:
        f = self.filters or {}
        parts: List[str] = []
        dr = f.get("date_range")
        if dr and isinstance(dr, (tuple, list)) and len(dr) == 2:
            parts.append(f"Date: {dr[0]} to {dr[1]}")

        def fmt(name: str, values: Any) -> Optional[str]:
            if not values:
                return None
            vals = ", ".join(str(x) for x in values) if isinstance(values, (list, tuple, set)) else str(values)
            return f"{name}: {vals}"

        for label, key in [("Region", "region"), ("Segment", "segment"), ("Category", "category")]:
            s = fmt(label, f.get(key))
            if s:
                parts.append(s)

        return " | ".join(parts) if parts else "No filters applied"

    def _has_breakdown_hint(self, q: str) -> bool:
        return bool(self._BREAKDOWN_PATTERN.search(q))

    def _has_time_hint(self, q: str) -> bool:
        ql = q.lower()
        if re.search(r"\b(20\d{2})\b", ql):
            return True
        if re.search(r"\b(\d{4}[-/](0?[1-9]|1[0-2]))\b", ql):
            return True
        if re.search(r"\b(q[1-4]|quarter|month|year|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", ql):
            return True
        if re.search(r"\b(in|during|between|from|since|until|last|previous|this)\b", ql):
            return True
        return False

    def _fast_kpi_answer(self, q: str) -> Optional[str]:
        ql = (q or "").strip().lower()
        if not ql:
            return None
        if self._has_breakdown_hint(q):
            return None
        if self._has_time_hint(q):
            return None

        def money(x: float) -> str:
            return f"${x:,.0f}"

        ts = float(self.kpis.get("total_sales", 0) or 0)
        tp = float(self.kpis.get("total_profit", 0) or 0)
        to_ = int(self.kpis.get("total_orders", 0) or 0)
        pm = float(self.kpis.get("profit_margin", 0) or 0)

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

    # ─────────────────────────────────────────────────────────
    # Rule-based NL → Plan  (no LLM, handles simple queries)
    # ─────────────────────────────────────────────────────────

    def _rule_based_plan(self, q: str) -> Optional[Dict[str, Any]]:
        """
        Parse simple natural-language queries into a plan dict without calling Gemini.
        Handles patterns like:
          - "sales by region"
          - "total profit by category"
          - "top 5 sub-category by profit"
          - "profit trend by month"
          - "compare sales yoy"
        Returns a validated plan dict, or None if pattern not recognized.
        """
        ql = q.lower().strip()
        s0, e0 = self._get_dashboard_date_range()
        regions, segments, categories = self._get_current_filter_lists()

        base_plan: Dict[str, Any] = {
            "intent": "kpi_value",
            "metrics": ["sales"],
            "time_grain": "none",
            "breakdown_by": None,
            "start_date": s0,
            "end_date": e0,
            "compare_period": None,
            "top_k": None,
            "order_by": "sales",
            "filters": {"region": [], "segment": [], "category": []},
        }

        # ── Detect metric ────────────────────────────────────
        detected_metric = "sales"
        for kw, metric in sorted(self._METRIC_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_metric = metric
                break
        base_plan["metrics"] = [detected_metric]
        base_plan["order_by"] = detected_metric

        # ── Detect breakdown ─────────────────────────────────
        detected_breakdown: Optional[str] = None
        for kw, dim in sorted(self._BREAKDOWN_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_breakdown = dim
                break
        # Also check via the existing regex pattern
        if detected_breakdown is None:
            m = self._BREAKDOWN_PATTERN.search(q)
            if m:
                raw = m.group(2).lower().replace(" ", "_").replace("-", "_")
                # Normalize "sub_category" variants
                if "sub" in raw:
                    raw = "sub_category"
                if raw in self._DIM_BREAKDOWNS:
                    detected_breakdown = raw

        # ── Detect top-k ─────────────────────────────────────
        top_match = self._TOP_PATTERN.search(ql)
        top_k: Optional[int] = int(top_match.group(1)) if top_match else None

        # ── Detect time grain ────────────────────────────────
        detected_grain = "none"
        for kw, grain in sorted(self._GRAIN_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in ql:
                detected_grain = grain
                break

        # ── Detect compare period ────────────────────────────
        detected_compare: Optional[str] = None
        compare_triggers = re.search(
            r"\b(compare|vs|versus|compared|growth|change|difference)\b", ql
        )
        if compare_triggers:
            for kw, cp in sorted(self._COMPARE_KEYWORDS.items(), key=lambda x: -len(x[0])):
                if kw in ql:
                    detected_compare = cp
                    break
            if detected_compare is None:
                # default compare to prev_period if "compare" detected but no period specified
                detected_compare = "prev_period"

        # ── Build intent ──────────────────────────────────────
        if detected_compare:
            if len(base_plan["metrics"]) != 1:
                base_plan["metrics"] = [detected_metric]
            base_plan["intent"] = "kpi_compare"
            base_plan["compare_period"] = detected_compare
        elif top_k and detected_breakdown:
            base_plan["intent"] = "kpi_rank"
            base_plan["breakdown_by"] = detected_breakdown
            base_plan["top_k"] = top_k
        elif detected_grain != "none":
            base_plan["intent"] = "kpi_trend"
            base_plan["time_grain"] = detected_grain
            base_plan["breakdown_by"] = detected_breakdown
        elif detected_breakdown:
            base_plan["intent"] = "kpi_value"
            base_plan["breakdown_by"] = detected_breakdown
        else:
            # Nothing recognizable beyond maybe a metric — let Gemini handle it
            return None

        # ── Must have at least breakdown or grain or compare ──
        has_something = (
            detected_breakdown is not None
            or detected_grain != "none"
            or detected_compare is not None
        )
        if not has_something:
            return None

        try:
            return self._validate_plan(base_plan)
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────
    # Gemini: NL → Plan JSON
    # ─────────────────────────────────────────────────────────

    def _plan_prompt_with_rag(self, user_question: str, rag_context: RAGContext) -> str:
        regions, segments, categories = self._get_current_filter_lists()
        default_start, default_end = self._get_dashboard_date_range()
        rag_section = rag_context.as_prompt_section(max_chunks=7)

        return f"""
    You are a STRICT JSON query planner for a Postgres "Superstore" analytics dataset.

    You must output ONLY ONE valid JSON object that matches the schema below.
    No markdown. No SQL. No explanations. No extra keys.
    If the question cannot be answered with the schema (missing required info or unsupported dimension), output intent="clarify".

    === VERIFIED DATA FACTS (use these only; do not invent facts) ===
    {rag_section}

    === SUPPORTED INTENTS ===
    - kpi_value   : totals for 1–2 metrics, optionally grouped by breakdown_by
    - kpi_trend   : time series for 1–2 metrics (requires time_grain != "none")
    - kpi_rank    : ranked top-k list (requires breakdown_by and top_k)
    - kpi_compare : compare 1 metric to a prior period (requires compare_period)
    - clarify     : ask ONE clarifying question when needed

    === SUPPORTED METRICS ===
    sales | profit | orders | profit_margin

    === SUPPORTED TIME GRAIN ===
    none | week | month | quarter | year

    === SUPPORTED BREAKDOWN (dimension) ===
    null | region | segment | category | sub_category

    === SUPPORTED COMPARE PERIOD ===
    null | prev_period | mom | yoy

    === FILTER RULES (STRICT) ===
    - filters must include exactly: region, segment, category
    - values must be chosen ONLY from the CURRENT dashboard selections below
    - [] means "no extra filtering beyond current dashboard state"

    Current selectable values:
    regions={json.dumps(regions)}
    segments={json.dumps(segments)}
    categories={json.dumps(categories)}

    === DATE RULES ===
    - Dates must be YYYY-MM-DD
    - If user does not specify dates, use the default range:
    start_date="{default_start}", end_date="{default_end}"
    - If user mentions an ambiguous period (e.g., "last month", "Q2"), and VERIFIED DATA FACTS do not clearly resolve it,
    use intent="clarify" instead of guessing.

    === IMPORTANT BEHAVIOR ===
    - If user asks for an unsupported breakdown like state/city/ship mode/product:
    - If it can be reasonably mapped to sub_category (only when user clearly means product category level),
        use breakdown_by="sub_category".
    - Otherwise use intent="clarify".
    - If user requests "top N" but does not specify a breakdown, use intent="clarify".
    - For kpi_compare: metrics must contain exactly 1 item.
    - For kpi_rank: time_grain must be "none".

    === OUTPUT JSON SCHEMA (copy keys exactly) ===
    {{
    "intent": "kpi_value",
    "metrics": ["sales"],
    "time_grain": "none",
    "breakdown_by": null,
    "start_date": "{default_start}",
    "end_date": "{default_end}",
    "compare_period": null,
    "top_k": null,
    "order_by": "sales",
    "filters": {{"region": [], "segment": [], "category": []}}
    }}

    === USER QUESTION ===
    {user_question}

    Return ONLY the JSON object:
    """.strip()

    def _gemini_plan(self, q: str, rag_context: RAGContext) -> Dict[str, Any]:
        if not self.gemini_ready:
            raise RuntimeError("Gemini API key not configured.")

        prompt = self._plan_prompt_with_rag(q, rag_context)
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=400,
            ),
        )
        text = (getattr(resp, "text", None) or "").strip()
        return self._extract_json_object(text)

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            pass
        cleaned = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError("Gemini did not return valid JSON.")
        return json.loads(m.group(0))

    # ─────────────────────────────────────────────────────────
    # Plan validation
    # ─────────────────────────────────────────────────────────

    def _parse_date(self, s: str) -> Optional[date]:
        if not isinstance(s, str):
            return None
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date()
        except Exception:
            return None

    def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object.")

        intent = plan.get("intent")
        if intent not in {"kpi_value", "kpi_trend", "kpi_rank", "kpi_compare", "clarify"}:
            raise ValueError(f"Invalid intent: {intent}")

        if intent == "clarify":
            cq = (plan.get("clarifying_question") or "").strip()
            if not cq:
                raise ValueError("clarifying_question required when intent='clarify'.")
            plan.setdefault("metrics", ["sales"])
            plan.setdefault("time_grain", "none")
            plan.setdefault("breakdown_by", None)
            plan.setdefault("compare_period", None)
            plan.setdefault("top_k", None)
            plan.setdefault("order_by", plan["metrics"][0])
            plan.setdefault("filters", {"region": [], "segment": [], "category": []})
            s0, e0 = self._get_dashboard_date_range()
            plan.setdefault("start_date", s0)
            plan.setdefault("end_date", e0)
            return plan

        metrics = plan.get("metrics")
        if isinstance(metrics, str):
            metrics = [metrics]
        if not isinstance(metrics, list) or not metrics:
            raise ValueError("metrics must be a non-empty array.")
        metrics = [str(m) for m in metrics]
        if len(metrics) > 2:
            raise ValueError("metrics supports at most 2 items.")
        for m in metrics:
            if m not in self._METRICS:
                raise ValueError(f"Invalid metric: {m}")

        time_grain = plan.get("time_grain", "none")
        if time_grain not in self._TIME_GRAINS:
            raise ValueError(f"Invalid time_grain: {time_grain}")

        if intent == "kpi_value" and time_grain != "none":
            intent = "kpi_trend"
            plan["intent"] = intent

        breakdown_by = plan.get("breakdown_by")
        if breakdown_by is not None:
            breakdown_by = str(breakdown_by)
            if breakdown_by not in self._DIM_BREAKDOWNS:
                raise ValueError(f"Invalid breakdown_by: {breakdown_by}")

        compare_period = plan.get("compare_period")
        if compare_period is not None:
            compare_period = str(compare_period)
            if compare_period not in {"prev_period", "mom", "yoy"}:
                raise ValueError(f"Invalid compare_period: {compare_period}")

        top_k = plan.get("top_k")
        if top_k is not None:
            try:
                top_k = int(top_k)
            except Exception:
                raise ValueError("top_k must be an integer or null.")
            if not (1 <= top_k <= 50):
                raise ValueError("top_k must be between 1 and 50.")

        order_by = plan.get("order_by") or metrics[0]
        if order_by not in metrics:
            order_by = metrics[0]

        s0, e0 = self._get_dashboard_date_range()
        start_date = plan.get("start_date") or s0
        end_date = plan.get("end_date") or e0
        sd = self._parse_date(start_date)
        ed = self._parse_date(end_date)
        if sd is None or ed is None:
            raise ValueError("start_date/end_date must be YYYY-MM-DD.")
        if sd > ed:
            raise ValueError("start_date must be <= end_date.")

        filters = plan.get("filters")
        if not isinstance(filters, dict):
            raise ValueError("filters must be an object.")

        current_regions, current_segments, current_categories = self._get_current_filter_lists()
        allowed = {
            "region": set(current_regions),
            "segment": set(current_segments),
            "category": set(current_categories),
        }
        norm_filters: Dict[str, List[str]] = {"region": [], "segment": [], "category": []}
        for dim in ["region", "segment", "category"]:
            vals = filters.get(dim) or []
            if not isinstance(vals, list):
                raise ValueError(f"filters.{dim} must be an array.")
            vals = [str(v) for v in vals]
            if vals and not set(vals).issubset(allowed[dim]):
                bad = sorted(set(vals) - allowed[dim])
                raise ValueError(f"filters.{dim} contains values not in dashboard: {bad}")
            norm_filters[dim] = vals

        if intent == "kpi_rank":
            if not breakdown_by:
                raise ValueError("breakdown_by required for intent='kpi_rank'.")
            if top_k is None:
                raise ValueError("top_k required for intent='kpi_rank'.")
            if time_grain != "none":
                raise ValueError("kpi_rank only supports time_grain='none'.")
        if intent == "kpi_compare":
            if compare_period is None:
                raise ValueError("compare_period required for intent='kpi_compare'.")
            if len(metrics) != 1:
                raise ValueError("kpi_compare supports exactly 1 metric.")

        return {
            "intent": intent,
            "metrics": metrics,
            "time_grain": time_grain,
            "breakdown_by": breakdown_by,
            "start_date": sd.strftime("%Y-%m-%d"),
            "end_date": ed.strftime("%Y-%m-%d"),
            "compare_period": compare_period,
            "top_k": top_k,
            "order_by": order_by,
            "filters": norm_filters,
        }

    # ─────────────────────────────────────────────────────────
    # SQL builder
    # ─────────────────────────────────────────────────────────

    def _build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        table = getattr(Config, "DB_TABLE", "superstore")
        intent = plan["intent"]
        metrics = plan["metrics"]
        time_grain = plan["time_grain"]
        breakdown_by = plan.get("breakdown_by")
        f = plan["filters"]
        top_k = plan.get("top_k")
        order_by = plan.get("order_by") or metrics[0]

        def metric_expr(m: str) -> str:
            return {
                "sales": "SUM(sales)",
                "profit": "SUM(profit)",
                "orders": "COUNT(DISTINCT order_id)",
                "profit_margin": "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END",
            }[m]

        where_parts = ["order_date >= %(start)s", "order_date <= %(end)s"]
        params: Dict[str, Any] = {"start": plan["start_date"], "end": plan["end_date"]}

        if f.get("region"):
            where_parts.append("region = ANY(%(regions)s)")
            params["regions"] = f["region"]
        if f.get("segment"):
            where_parts.append("segment = ANY(%(segments)s)")
            params["segments"] = f["segment"]
        if f.get("category"):
            where_parts.append("category = ANY(%(categories)s)")
            params["categories"] = f["category"]

        where_sql = " AND ".join(where_parts)

        bucket_sql = None
        if time_grain != "none":
            grain_map = {"week": "week", "month": "month", "quarter": "quarter", "year": "year"}
            if time_grain not in grain_map:
                raise ValueError(f"Unsupported time_grain: {time_grain}")
            bucket_sql = f"DATE_TRUNC('{grain_map[time_grain]}', order_date)"

        select_parts: List[str] = []
        group_parts: List[str] = []

        if bucket_sql:
            select_parts.append(f"{bucket_sql} AS period")
            group_parts.append("period")

        if breakdown_by:
            select_parts.append(f"{breakdown_by} AS breakdown")
            group_parts.append("breakdown")

        for m in metrics:
            select_parts.append(f"{metric_expr(m)} AS {m}")

        sql_lines = [
            "SELECT " + ", ".join(select_parts),
            f"FROM {table}",
            f"WHERE {where_sql}",
        ]
        if group_parts:
            sql_lines.append("GROUP BY " + ", ".join(group_parts))

        if intent == "kpi_rank":
            sql_lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
            sql_lines.append("LIMIT %(top_k)s")
            params["top_k"] = top_k
        elif breakdown_by and not bucket_sql:
            sql_lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
        elif bucket_sql:
            if breakdown_by:
                sql_lines.append("ORDER BY period ASC, breakdown ASC")
            else:
                sql_lines.append("ORDER BY period ASC")

        return "\n".join(sql_lines), params

    def _run_plan(self, plan: Dict[str, Any]) -> pd.DataFrame:
        if plan["intent"] == "kpi_compare":
            sql, params = self._build_sql(plan)
            cur = execute_query(sql, params=params)

            sd = datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
            ed = datetime.strptime(plan["end_date"], "%Y-%m-%d").date()
            cp = plan["compare_period"]

            if cp == "prev_period":
                delta = (ed - sd).days + 1
                prev_end = sd.fromordinal(sd.toordinal() - 1)
                prev_start = prev_end.fromordinal(prev_end.toordinal() - (delta - 1))
            elif cp == "mom":
                prev_start = (pd.Timestamp(sd) - pd.DateOffset(months=1)).date()
                prev_end = (pd.Timestamp(ed) - pd.DateOffset(months=1)).date()
            elif cp == "yoy":
                prev_start = (pd.Timestamp(sd) - pd.DateOffset(years=1)).date()
                prev_end = (pd.Timestamp(ed) - pd.DateOffset(years=1)).date()
            else:
                raise ValueError(f"Unsupported compare_period: {cp}")

            plan_prev = {**plan, "start_date": prev_start.strftime("%Y-%m-%d"), "end_date": prev_end.strftime("%Y-%m-%d")}
            sql2, params2 = self._build_sql(plan_prev)
            prev = execute_query(sql2, params=params2)

            metric = plan["metrics"][0]
            cur_val = float(cur.iloc[0][metric]) if cur is not None and not cur.empty else 0.0
            prev_val = float(prev.iloc[0][metric]) if prev is not None and not prev.empty else 0.0

            return pd.DataFrame([{
                "metric": metric,
                "current_start": plan["start_date"],
                "current_end": plan["end_date"],
                "prev_start": plan_prev["start_date"],
                "prev_end": plan_prev["end_date"],
                "current": cur_val,
                "previous": prev_val,
            }])

        sql, params = self._build_sql(plan)
        return execute_query(sql, params=params)

    # ─────────────────────────────────────────────────────────
    # Insight generator  (LLM-powered, rule-based fallback)
    # ─────────────────────────────────────────────────────────

    def _generate_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Generate a natural-language analytical insight using Gemini LLM.
        Falls back to rule-based insight if Gemini is unavailable or fails.
        """
        if df is None or df.empty:
            return ""

        # Try LLM insight first
        if self.gemini_ready:
            llm_insight = self._llm_insight(plan, df)
            if llm_insight:
                return f"\n\n---\n💡 **Insight:** {llm_insight}"

        # Fallback: rule-based insight
        rule_insight = self._rule_based_insight(plan, df)
        if rule_insight:
            return f"\n\n---\n💡 **Insight:** " + "  \n".join(rule_insight)
        return ""

    def _build_data_summary(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        """Serialize the query result into a compact text table for the prompt."""
        intent = plan.get("intent")
        metrics = plan.get("metrics", ["sales"])
        m0 = plan.get("order_by") or metrics[0]
        breakdown_by = plan.get("breakdown_by")

        def fmt(v: Any, col: str) -> str:
            try:
                fv = float(v)
                if col in {"sales", "profit"}:
                    return f"${fv:,.0f}"
                if col == "profit_margin":
                    return f"{fv:.1f}%"
                if col == "orders":
                    return f"{int(fv):,}"
                return f"{fv:,.2f}"
            except Exception:
                return str(v)

        lines: List[str] = []

        if intent == "kpi_compare" and "current" in df.columns:
            row = df.iloc[0]
            metric = str(row.get("metric", m0))
            cur = float(row["current"])
            prev = float(row["previous"])
            chg = ((cur - prev) / abs(prev) * 100) if prev != 0 else None
            lines.append(f"Metric: {metric}")
            lines.append(f"Current ({row['current_start']} – {row['current_end']}): {fmt(cur, metric)}")
            lines.append(f"Previous ({row['prev_start']} – {row['prev_end']}): {fmt(prev, metric)}")
            lines.append(f"Change: {chg:+.1f}%" if chg is not None else "Change: n/a")

        elif "breakdown" in df.columns:
            sorted_df = df.sort_values(by=m0, ascending=False).head(15).reset_index(drop=True)
            total = float(sorted_df[m0].sum()) if m0 in sorted_df.columns else 0
            lines.append(f"Breakdown by: {breakdown_by}  |  Ranked by: {m0}  |  Total: {fmt(total, m0)}")
            for i, (_, r) in enumerate(sorted_df.iterrows(), 1):
                vals = "  ".join(
                    f"{col}={fmt(r[col], col)}"
                    for col in metrics
                    if col in r
                )
                lines.append(f"  {i}. {r['breakdown']}: {vals}")

        elif "period" in df.columns:
            show = df.sort_values("period").tail(24)
            for _, r in show.iterrows():
                p = str(r.get("period", ""))[:7]
                vals = "  ".join(
                    f"{col}={fmt(r[col], col)}"
                    for col in metrics
                    if col in r
                )
                lines.append(f"  {p}: {vals}")

        else:
            # Single aggregate
            r0 = df.iloc[0]
            for col in metrics:
                if col in r0:
                    lines.append(f"{col}: {fmt(r0[col], col)}")

        return "\n".join(lines)

    def _llm_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Call Gemini to generate a short, varied analytical insight
        based on the actual query result data.
        """
        data_summary = self._build_data_summary(plan, df)
        intent = plan.get("intent", "kpi_value")
        metrics = plan.get("metrics", ["sales"])
        breakdown_by = plan.get("breakdown_by")
        compare_period = plan.get("compare_period")
        time_grain = plan.get("time_grain", "none")
        ctx = self._natural_context(plan)

        intent_hint = {
            "kpi_rank":    "This is a ranked list. Comment on the leader, notable gaps, concentration, or any surprising entries.",
            "kpi_value":   f"This is a breakdown by {breakdown_by}. Comment on the distribution, dominant players, or any imbalance.",
            "kpi_trend":   f"This is a time-series by {time_grain}. Comment on the overall trajectory, peak, recent momentum, or seasonality.",
            "kpi_compare": f"This compares {metrics[0]} across two periods ({compare_period}). Comment on the magnitude of change, what it signals, and whether it's concerning or positive.",
        }.get(intent, "Provide a brief analytical observation.")

        prompt = f"""You are a concise business analyst. Write an insight based ONLY on the numbers provided.

            === RESULT (VERIFIED) ===
            Context: {ctx or 'full dataset'}
            {data_summary}

            === TASK ===
            {intent_hint}

            Hard rules:
            - Exactly 2 or 3 sentences. No more.
            - Use the actual figures/names from the RESULT (do not invent).
            - No table reprint. No bullets. No headings.
            - Avoid generic openers like "The data shows" or "Overall".
            - Bold (**...**) only the most important 1–3 names/numbers.
            - If the result is not sufficient to infer a meaningful insight, write:
            "Not enough signal in this slice to draw a strong conclusion."

            Output plain text only:"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,          # Higher temp = more varied phrasing
                    max_output_tokens=150,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            # Sanity check: must be non-trivial
            if len(text) > 20:
                return text
        except Exception:
            pass
        return ""

    def _rule_based_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """
        Deterministic fallback insight generator.
        Returns a list of insight strings (joined with line breaks by caller).
        """
        intent = plan.get("intent")
        metrics = plan.get("metrics", ["sales"])
        m0 = plan.get("order_by") or metrics[0]
        breakdown_by = plan.get("breakdown_by")

        def fmt_val(v: float, metric: str) -> str:
            if metric in {"sales", "profit"}:
                return f"${v:,.0f}"
            if metric == "profit_margin":
                return f"{v:.1f}%"
            return f"{int(v):,}"

        insights: List[str] = []

        if intent in {"kpi_rank", "kpi_value"} and breakdown_by and "breakdown" in df.columns and m0 in df.columns:
            sorted_df = df.sort_values(by=m0, ascending=False).reset_index(drop=True)
            n = len(sorted_df)
            if n >= 1:
                top_name = sorted_df.iloc[0]["breakdown"]
                top_val = float(sorted_df.iloc[0][m0])
                insights.append(f"**{top_name}** leads with {fmt_val(top_val, m0)}.")
            if n >= 2:
                second_name = sorted_df.iloc[1]["breakdown"]
                second_val = float(sorted_df.iloc[1][m0])
                if top_val != 0:
                    gap_pct = abs(top_val - second_val) / abs(top_val) * 100
                    if gap_pct > 50:
                        insights.append(f"Gap to **{second_name}** ({fmt_val(second_val, m0)}) is significant ({gap_pct:.0f}%).")
                    elif gap_pct < 10:
                        insights.append(f"**{second_name}** ({fmt_val(second_val, m0)}) is very close behind.")
                    else:
                        insights.append(f"**{second_name}** follows at {fmt_val(second_val, m0)}.")
            if n >= 3:
                total_val = float(sorted_df[m0].sum())
                top2_val = float(sorted_df.head(2)[m0].sum())
                if total_val > 0:
                    share = top2_val / total_val * 100
                    if share >= 60:
                        top2 = " & ".join(str(sorted_df.iloc[i]["breakdown"]) for i in range(2))
                        insights.append(f"**{top2}** together account for {share:.0f}% of total.")
            if m0 in {"profit", "profit_margin"}:
                negatives = sorted_df[sorted_df[m0] < 0]
                if not negatives.empty:
                    neg_names = ", ".join(f"**{r['breakdown']}**" for _, r in negatives.iterrows())
                    insights.append(f"⚠️ {neg_names} {'is' if len(negatives) == 1 else 'are'} at a loss.")

        elif intent == "kpi_trend" and "period" in df.columns and m0 in df.columns:
            sorted_df = df.sort_values("period").reset_index(drop=True)
            n = len(sorted_df)
            if n >= 2:
                first_val = float(sorted_df.iloc[0][m0])
                last_val = float(sorted_df.iloc[-1][m0])
                if first_val != 0:
                    chg = (last_val - first_val) / abs(first_val) * 100
                    direction = "grown" if chg > 0 else "declined"
                    insights.append(f"Overall {direction} by **{abs(chg):.1f}%** over the period.")
            if n >= 1:
                peak_idx = sorted_df[m0].idxmax()
                peak_row = sorted_df.loc[peak_idx]
                insights.append(f"Peak: **{str(peak_row['period'])[:7]}** at {fmt_val(float(peak_row[m0]), m0)}.")

        elif intent == "kpi_compare" and "current" in df.columns:
            row = df.iloc[0]
            cur, prev = float(row["current"]), float(row["previous"])
            metric = str(row.get("metric", m0))
            if prev != 0:
                chg = (cur - prev) / abs(prev) * 100
                if abs(chg) >= 20:
                    word = "strong growth" if chg > 0 else "sharp decline"
                    insights.append(f"**{word}** of {abs(chg):.1f}% — worth investigating drivers.")
                elif abs(chg) < 5:
                    insights.append("Performance is **stable** with minimal period-over-period change.")
                else:
                    d = "ahead of" if chg > 0 else "behind"
                    insights.append(f"Current period is **{d}** the prior by {abs(chg):.1f}%.")

        return insights

    # ─────────────────────────────────────────────────────────
    # Answer formatter
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_money(x: float) -> str:
        return f"${x:,.0f}"

    def _natural_context(self, plan: Dict[str, Any]) -> str:
        """
        Builds a concise, readable context line.
        Examples:
            "Jan 2014 – Dec 2017"
            "Jan 2014 – Dec 2017  ·  West, East"
        Returns empty string when no meaningful restriction to surface.
        """
        sd = plan.get("start_date", "")
        ed = plan.get("end_date", "")

        def fmt_date(d: str) -> str:
            try:
                return datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y")
            except Exception:
                return d

        date_part = f"{fmt_date(sd)} – {fmt_date(ed)}" if sd and ed else ""

        f = plan.get("filters") or {}
        regions, segments, categories = self._get_current_filter_lists()

        filter_parts: List[str] = []
        fv_r = f.get("region") or []
        fv_s = f.get("segment") or []
        fv_c = f.get("category") or []

        if fv_r and set(fv_r) != set(regions):
            filter_parts.append(", ".join(fv_r))
        if fv_s and set(fv_s) != set(segments):
            filter_parts.append(", ".join(fv_s))
        if fv_c and set(fv_c) != set(categories):
            filter_parts.append(", ".join(fv_c))

        parts = [p for p in [date_part] + filter_parts if p]
        return "  ·  ".join(parts)

    # legacy alias
    def _explain_query(self, plan: Dict[str, Any]) -> str:
        return self._natural_context(plan)

    def _format_answer(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        if plan["intent"] == "clarify":
            return plan.get("clarifying_question", "Could you clarify your question?")

        ctx = self._natural_context(plan)
        ctx_line = f"\n*{ctx}*" if ctx else ""

        intent = plan["intent"]
        metrics = plan["metrics"]
        time_grain = plan["time_grain"]
        breakdown_by = plan.get("breakdown_by")

        metric_labels = {
            "orders": "Total Orders",
            "sales": "Total Sales",
            "profit": "Total Profit",
            "profit_margin": "Profit Margin",
        }

        # Natural sentence openers for single-value answers
        metric_openers = {
            "sales":         "Total sales came in at",
            "profit":        "Total profit reached",
            "orders":        "The total number of orders was",
            "profit_margin": "The profit margin stood at",
        }

        if df is None or df.empty:
            return f"No data found for the selected filters.{ctx_line}"

        # ── kpi_compare ───────────────────────────────────────
        if intent == "kpi_compare":
            row = df.iloc[0]
            m = row["metric"]
            cur = float(row["current"])
            prev = float(row["previous"])
            change = ((cur - prev) / abs(prev) * 100.0) if prev != 0 else None

            def fmt_val(v: float) -> str:
                if m in {"sales", "profit"}:
                    return self._fmt_money(v)
                if m == "profit_margin":
                    return f"{v:.2f}%"
                return f"{int(v):,}"

            cp_label = {
                "yoy": "year-over-year",
                "mom": "month-over-month",
                "prev_period": "vs the previous period",
            }.get(plan.get("compare_period", ""), "vs previous")

            delta_s = f"{change:+.1f}%" if change is not None else "n/a"
            direction = "up" if (change or 0) >= 0 else "down"

            answer = "\n".join([
                f"Here's the **{metric_labels[m]}** comparison ({cp_label}):{ctx_line}",
                "",
                f"- **Current** ({row['current_start']} – {row['current_end']}): {fmt_val(cur)}",
                f"- **Previous** ({row['prev_start']} – {row['prev_end']}): {fmt_val(prev)}",
                f"- **Change:** {delta_s} ({direction})",
            ])
            answer += self._generate_insight(plan, df)
            return answer

        # ── kpi_value (no time grain) ─────────────────────────
        if intent == "kpi_value" and time_grain == "none":
            if breakdown_by:
                dim_label = breakdown_by.replace("_", " ").title()
                m0 = plan.get("order_by") or metrics[0]
                lines = [
                    f"Here's how **{metric_labels[metrics[0]]}** breaks down by **{dim_label}**:{ctx_line}",
                    "",
                ]
                show = df.sort_values(by=m0, ascending=False).head(20)
                for rank, (_, r) in enumerate(show.iterrows(), start=1):
                    b = r.get("breakdown", "—")
                    val_parts = []
                    for m in metrics:
                        v = r.get(m)
                        if v is None:
                            continue
                        if m in {"sales", "profit"}:
                            val_parts.append(f"{metric_labels[m]}: {self._fmt_money(float(v))}")
                        elif m == "profit_margin":
                            val_parts.append(f"{metric_labels[m]}: {float(v):.2f}%")
                        else:
                            val_parts.append(f"{metric_labels[m]}: {int(v):,}")
                    lines.append(f"{rank}. **{b}** — " + " | ".join(val_parts))
                answer = "\n".join(lines)
                answer += self._generate_insight(plan, df)
                return answer

            # Single aggregate
            r0 = df.iloc[0]
            if len(metrics) == 1:
                m = metrics[0]
                v = float(r0[m])
                vs = (
                    self._fmt_money(v) if m in {"sales", "profit"}
                    else (f"{v:.2f}%" if m == "profit_margin" else f"{int(v):,}")
                )
                opener = metric_openers.get(m, f"{metric_labels[m]} is")
                return f"{opener} **{vs}**.{ctx_line}"

            # Multiple metrics together
            lines = [f"Here's a summary of the selected metrics:{ctx_line}", ""]
            for m in metrics:
                v = float(r0[m])
                vs = (
                    self._fmt_money(v) if m in {"sales", "profit"}
                    else (f"{v:.2f}%" if m == "profit_margin" else f"{int(v):,}")
                )
                lines.append(f"- **{metric_labels[m]}:** {vs}")
            return "\n".join(lines)

        # ── kpi_trend ─────────────────────────────────────────
        if intent == "kpi_trend" and time_grain != "none":
            grain_label = {
                "week": "week", "month": "month",
                "quarter": "quarter", "year": "year",
            }.get(time_grain, time_grain)
            metric_str = " & ".join(metric_labels[m] for m in metrics)

            lines = [
                f"Here's the **{metric_str}** trend by {grain_label}:{ctx_line}",
                "",
            ]
            show = df.copy()
            if "period" in show.columns:
                show = show.sort_values("period").tail(24)
            if breakdown_by:
                m0 = metrics[0]
                top_b = show.groupby("breakdown")[m0].sum().sort_values(ascending=False).head(5).index.tolist()
                show = show[show["breakdown"].isin(top_b)]

            for _, r in show.iterrows():
                p = str(r.get("period", ""))[:7]
                val_parts = []
                for m in metrics:
                    v = float(r[m])
                    vs = (
                        self._fmt_money(v) if m in {"sales", "profit"}
                        else (f"{v:.2f}%" if m == "profit_margin" else f"{int(v):,}")
                    )
                    val_parts.append(vs)
                prefix = p
                if breakdown_by:
                    prefix += f" | {r.get('breakdown')}"
                lines.append(f"- **{prefix}:** " + " / ".join(val_parts))

            if len(lines) > 32:
                lines = lines[:32] + ["- *(truncated – showing most recent 30 periods)*"]

            answer = "\n".join(lines)
            answer += self._generate_insight(plan, df)
            return answer

        # ── kpi_rank ──────────────────────────────────────────
        if intent == "kpi_rank":
            m0 = metrics[0]
            dim_label = (breakdown_by or "item").replace("_", " ").title()
            lines = [
                f"Here are the **top {plan['top_k']} {dim_label}s** ranked by {metric_labels[m0]}:{ctx_line}",
                "",
            ]
            for i, (_, r) in enumerate(df.head(plan["top_k"]).iterrows(), start=1):
                b = r.get("breakdown", "—")
                v = float(r[m0])
                vs = (
                    self._fmt_money(v) if m0 in {"sales", "profit"}
                    else (f"{v:.2f}%" if m0 == "profit_margin" else f"{int(v):,}")
                )
                lines.append(f"{i}. **{b}** — {vs}")

            answer = "\n".join(lines)
            answer += self._generate_insight(plan, df)
            return answer

        return f"Done.{ctx_line}"

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _get_current_filter_lists(self) -> Tuple[List[str], List[str], List[str]]:
        f = self.filters or {}
        return list(f.get("region") or []), list(f.get("segment") or []), list(f.get("category") or [])

    def _get_dashboard_date_range(self) -> Tuple[str, str]:
        f = self.filters or {}
        dr = f.get("date_range")
        if dr and isinstance(dr, (tuple, list)) and len(dr) == 2:
            s, e = dr
            s = s.strftime("%Y-%m-%d") if isinstance(s, (datetime, date)) else str(s)
            e = e.strftime("%Y-%m-%d") if isinstance(e, (datetime, date)) else str(e)
            return s, e
        if "order_date" in self.df.columns and not self.df.empty:
            s = pd.to_datetime(self.df["order_date"].min()).date().strftime("%Y-%m-%d")
            e = pd.to_datetime(self.df["order_date"].max()).date().strftime("%Y-%m-%d")
            return s, e
        return "1900-01-01", "2100-01-01"

    def _get_dashboard_defaults(self) -> Dict[str, Any]:
        s0, e0 = self._get_dashboard_date_range()
        return {
            "start_date": s0,
            "end_date": e0,
            "filters": {
                "region": self.filters.get("region", []),
                "segment": self.filters.get("segment", []),
                "category": self.filters.get("category", []),
            },
        }

    # ─────────────────────────────────────────────────────────
    # Public: get_response
    # ─────────────────────────────────────────────────────────

    def get_response(self, user_question: str) -> str:
        q = (user_question or "").strip()
        self.last_plan = None
        self._last_question = q
        self._last_answer = ""

        if not q:
            return "Ask me about Sales, Profit, Orders, or Profit Margin."

        # ── 1. Fast KPI path (no breakdown, no time filter) ──────────
        fast = self._fast_kpi_answer(q)
        if fast:
            self._last_answer = fast
            self.rag.add_turn("user", q)
            self.rag.add_turn("assistant", fast)
            return fast

        # ── 2. Rule-based NL parser (instant, no LLM) ────────────────
        rule_plan = self._rule_based_plan(q)
        if rule_plan and not self.gemini_ready:
            # No Gemini → rule-based is the only option
            try:
                self.last_plan = rule_plan
                result_df = self._run_plan(rule_plan)
                answer = self._format_answer(rule_plan, result_df)
            except Exception as e:
                answer = f"❌ Sorry, I couldn't answer that. ({str(e)})"
            self._last_answer = answer
            self.rag.add_turn("user", q)
            self.rag.add_turn("assistant", answer)
            return answer

        if not self.gemini_ready:
            return "⚠️ Gemini API Key not configured in .env"

        # ── 3. Gemini plan (richer NL understanding) ──────────────────
        rag_ctx = self.rag.retrieve(q, k=7)
        gemini_ok = False
        try:
            plan = self._gemini_plan(q, rag_ctx)
            plan = self._validate_plan(plan)
            self.last_plan = plan
            result_df = self._run_plan(plan)
            answer = self._format_answer(plan, result_df)
            gemini_ok = True
        except Exception as gemini_err:
            # ── 4. Fallback: rule-based plan if Gemini failed ─────────
            if rule_plan:
                try:
                    self.last_plan = rule_plan
                    result_df = self._run_plan(rule_plan)
                    answer = self._format_answer(rule_plan, result_df)
                except Exception as e:
                    answer = f"❌ Sorry, I couldn't answer that. ({str(e)})"
            else:
                answer = f"❌ Sorry, I couldn't answer that. ({str(gemini_err)})"

        self._last_answer = answer
        self.rag.add_turn("user", q)
        self.rag.add_turn("assistant", answer)
        return answer

    # ─────────────────────────────────────────────────────────
    # Public: get_suggestions
    # ─────────────────────────────────────────────────────────

    def get_suggestions(self, *, language: str = "en") -> List[Dict[str, Any]]:
        dashboard_defaults = self._get_dashboard_defaults()

        if not self._last_question:
            suggs = self._rule_engine.suggest(
                plan=self.last_plan or {},
                dashboard_defaults=dashboard_defaults,
            )
            return [{"text": s.text, "plan": s.plan} for s in suggs]

        rag_ctx = self.rag.retrieve_for_suggestions(
            last_question=self._last_question,
            last_answer=self._last_answer,
            k=8,
        )

        if self._rag_suggest_engine:
            suggs = self._rag_suggest_engine.suggest(
                last_question=self._last_question,
                last_answer=self._last_answer,
                rag_context=rag_ctx,
                last_plan=self.last_plan,
                dashboard_defaults=dashboard_defaults,
            )
        else:
            suggs = self._rule_engine.suggest(
                plan=self.last_plan or {},
                dashboard_defaults=dashboard_defaults,
            )

        return [{"text": s.text, "plan": s.plan} for s in suggs]

    # ─────────────────────────────────────────────────────────
    # Public: run suggestion plan directly
    # ─────────────────────────────────────────────────────────

    def get_response_from_plan(self, plan: Dict[str, Any]) -> str:
        try:
            plan = self._validate_plan(plan)
            self.last_plan = plan
            df = self._run_plan(plan)
            answer = self._format_answer(plan, df)
            self._last_answer = answer
            self.rag.add_turn("assistant", answer)
            return answer
        except Exception as e:
            self.last_plan = None
            return f"❌ Could not run that suggestion. ({str(e)})"

    # ─────────────────────────────────────────────────────────
    # Public: get_insights
    # ─────────────────────────────────────────────────────────

    def get_insights(self) -> str:
        print("gemini_ready =", self.gemini_ready, "model =", getattr(self, "model_name", None))

        if not self.gemini_ready:
            return "Configure Gemini API Key to see auto-insights."
        if self.df.empty:
            return "No data available."

        rag_ctx = self.rag.retrieve("insights overview summary performance", k=6)
        rag_section = rag_ctx.as_prompt_section(max_chunks=6)

        prompt = f"""
            You are a business analyst. Using ONLY the verified data facts below, write exactly 3 bullet-point insights.

            === VERIFIED FACTS ===
            {rag_section}

            Rules:
            - Output exactly 3 lines.
            - Each line must start with "- ".
            - Numbers-first: each bullet must include at least one numeric value from the VERIFIED FACTS.
            - No fluff, no generic statements, no recommendations without evidence.
            - English only.

            Output:""".strip()
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                ),
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception as e:
            return f"Could not generate insights at this time. ({type(e).__name__}: {e})"

    def rebuild_rag(self) -> None:
        self.rag.build(df=self.df, kpis=self.kpis, filters=self.filters)

    @property
    def rag_total_chunks(self) -> int:
        return self.rag.total_chunks