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
    # Gemini: NL → Plan JSON
    # ─────────────────────────────────────────────────────────

    def _plan_prompt_with_rag(self, user_question: str, rag_context: RAGContext) -> str:
        regions, segments, categories = self._get_current_filter_lists()
        default_start, default_end = self._get_dashboard_date_range()
        rag_section = rag_context.as_prompt_section(max_chunks=7)

        return f"""
You are a query planner for a Postgres Superstore analytics database.
Return ONLY a valid JSON object. Do NOT include markdown. Do NOT write SQL.

=== REAL DATA FACTS FROM DASHBOARD ===
{rag_section}

=== VALID INTENTS ===
- "kpi_value"   : one or more KPI totals (optionally grouped by breakdown_by)
- "kpi_trend"   : time series of one or two metrics
- "kpi_rank"    : ranked top-k list (requires breakdown_by and top_k)
- "kpi_compare" : compare metric to a prior period (requires compare_period)
- "clarify"     : when the question is ambiguous or missing required info

=== VALID METRICS ===
"sales" | "profit" | "orders" | "profit_margin"

=== VALID TIME GRAIN ===
"none" | "week" | "month" | "quarter" | "year"

=== VALID BREAKDOWN ===
null | "region" | "segment" | "category" | "sub_category"

=== VALID COMPARE PERIOD ===
null | "prev_period" | "mom" | "yoy"

=== DATE RULES ===
- Dates must be "YYYY-MM-DD"
- Default range when not specified: start="{default_start}", end="{default_end}"
- Use real data facts above to pick correct year/month when user refers to a period

=== FILTER RULES ===
- filters object must have keys: region, segment, category
- Values must ONLY come from current dashboard selections:
  regions={json.dumps(regions)}
  segments={json.dumps(segments)}
  categories={json.dumps(categories)}
- Empty array [] = include all (current filter)

=== JSON SCHEMA ===
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

OUTPUT (JSON only):""".strip()

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

            return "\n".join([
                f"Here's the **{metric_labels[m]}** comparison ({cp_label}):{ctx_line}",
                "",
                f"- **Current** ({row['current_start']} – {row['current_end']}): {fmt_val(cur)}",
                f"- **Previous** ({row['prev_start']} – {row['prev_end']}): {fmt_val(prev)}",
                f"- **Change:** {delta_s} ({direction})",
            ])

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
                return "\n".join(lines)

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
            return "\n".join(lines)

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
            return "\n".join(lines)

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

        fast = self._fast_kpi_answer(q)
        if fast:
            self._last_answer = fast
            self.rag.add_turn("user", q)
            self.rag.add_turn("assistant", fast)
            return fast

        if not self.gemini_ready:
            return "⚠️ Gemini API Key not configured in .env"

        rag_ctx = self.rag.retrieve(q, k=7)

        try:
            plan = self._gemini_plan(q, rag_ctx)
            plan = self._validate_plan(plan)
            self.last_plan = plan
            result_df = self._run_plan(plan)
            answer = self._format_answer(plan, result_df)
        except Exception as e:
            answer = f"❌ Sorry, I couldn't answer that. ({str(e)})"

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
        if not self.gemini_ready:
            return "Configure Gemini API Key to see auto-insights."
        if self.df.empty:
            return "No data available."

        rag_ctx = self.rag.retrieve("insights overview summary performance", k=6)
        rag_section = rag_ctx.as_prompt_section(max_chunks=6)

        prompt = f"""
You are a business analyst. Based on the verified data facts below,
write exactly 3 concise bullet-point insights. Numbers-first, English only.

{rag_section}
""".strip()

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
        except Exception:
            return "Could not generate insights at this time."

    def rebuild_rag(self) -> None:
        self.rag.build(df=self.df, kpis=self.kpis, filters=self.filters)

    @property
    def rag_total_chunks(self) -> int:
        return self.rag.total_chunks