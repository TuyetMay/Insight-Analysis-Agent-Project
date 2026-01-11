# src/chatbot.py
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import google.generativeai as genai

from config import Config
from src.database import execute_query


class DashboardChatbot:
    """Hybrid Chatbot for the Superstore Dashboard (English-only).

    Architecture
    - Fast path (no Gemini): common KPI questions answered directly from `kpis`.
    - Query path (Gemini -> Plan JSON -> SQL): for time-based KPI questions such as
      "orders in March 2017" / "profit in 2017" / "profit margin Q1 2016".

    Security & reliability
    - Gemini NEVER returns raw SQL.
    - Gemini returns a small JSON "plan".
    - The app validates the plan using allowlists and then builds parameterized SQL.

    Notes
    - This chatbot assumes Postgres.
    - It respects the dashboard's current filters (region/segment/category) always.
    - For date filtering:
        * If the question includes an explicit date range/year/month/quarter, the plan can
          override the dashboard date range.
        * Otherwise, it uses the current dashboard date_range.
    """

    # ---- Allowlists (keep small & explicit) ----
    _METRICS = {"sales", "profit", "orders", "profit_margin"}
    _TIME_GRAINS = {"none", "week", "month", "quarter", "year"}
    _DIM_FILTERS = {"region", "segment", "category"}
    _DIM_BREAKDOWNS = {'region', 'segment', 'category'}

    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any], filters: Dict[str, Any]):
        self.df = df.copy()
        self.kpis = kpis
        self.filters = filters

        # Ensure datetime in df (used for fallback and fast trend computations if needed)
        if "order_date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["order_date"]):
            self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors="coerce")

        # Configure Gemini
        self.gemini_ready = False
        api_key = getattr(Config, "GOOGLE_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            self.model_name = getattr(Config, "GEMINI_MODEL", "gemini-1.5-flash")
            self.model = genai.GenerativeModel(self.model_name)
            self.gemini_ready = True

    # ---------------------------
    # Fast path: KPI card answers
    # ---------------------------
    def _fast_kpi_answer(self, q: str) -> Optional[str]:
        """Answer simple KPI questions directly from `self.kpis`.

        Only triggers when the question doesn't appear to request a specific time period.
        """
        ql = (q or "").strip().lower()
        if not ql:
            return None

        # If the question contains any explicit time hint, skip fast path.
        # Examples: 2017, 2017-03, March 2017, Q1, quarter, last month, etc.
        if re.search(r"\b(20\d{2})\b", ql):
            return None
        if re.search(r"\b(\d{4}[-/](0?[1-9]|1[0-2]))\b", ql):
            return None
        if re.search(r"\b(q[1-4]|quarter|month|year|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", ql):
            return None
        if re.search(r"\b(in|during|between|from|to|since|until|last|previous|this)\b", ql):
            return None

        def money(x: float) -> str:
            return f"${x:,.0f}"

        total_sales = float(self.kpis.get("total_sales", 0) or 0)
        total_profit = float(self.kpis.get("total_profit", 0) or 0)
        total_orders = int(self.kpis.get("total_orders", 0) or 0)
        profit_margin = float(self.kpis.get("profit_margin", 0) or 0)

        if re.search(r"\b(kpi|summary|overview|dashboard)\b", ql):
            return (
                f"Current KPIs (based on your filters):\n"
                f"- Total Sales: {money(total_sales)}\n"
                f"- Total Profit: {money(total_profit)}\n"
                f"- Total Orders: {total_orders:,}\n"
                f"- Profit Margin: {profit_margin:.2f}%"
            )

        if re.search(r"\b(total\s+sales|sales\s+total|revenue)\b", ql):
            return f"Total Sales: {money(total_sales)} (based on current filters)."

        if re.search(r"\b(total\s+profit|profit\s+total)\b", ql):
            return f"Total Profit: {money(total_profit)} (based on current filters)."

        if re.search(r"\b(total\s+orders|orders\s+total|number\s+of\s+orders)\b", ql):
            return f"Total Orders: {total_orders:,} (based on current filters)."

        if re.search(r"\b(profit\s+margin|margin)\b", ql):
            return f"Profit Margin: {profit_margin:.2f}% (based on current filters)."

        return None

    # ---------------------------
    # Gemini: NL -> Plan JSON
    # ---------------------------
    def _plan_prompt(self, user_question: str) -> str:
        """Prompt Gemini to produce a strict JSON plan (no markdown, no SQL)."""
        regions, segments, categories = self._get_current_filter_lists()
        default_start, default_end = self._get_dashboard_date_range()

        return f"""
You are a query planner for a Postgres Superstore analytics database.
Return ONLY a valid JSON object. Do NOT include markdown. Do NOT include SQL.

You must choose ONE of these intents:
- "kpi_value": return one or more KPI values for the period (optionally grouped by breakdown_by).
- "kpi_trend": return a time series for one or more metrics.
- "kpi_rank": return a ranked list (requires breakdown_by and top_k).
- "kpi_compare": compare a metric to a prior period (compare_period must be "mom" or "yoy" or "prev_period").
- "clarify": ask a clarification question when the user's question is ambiguous or missing required info.

Allowed metrics (use "metrics" as an array, 1 or 2 items):
- "sales" (SUM(sales))
- "profit" (SUM(profit))
- "orders" (COUNT(DISTINCT order_id))
- "profit_margin" (SUM(profit)/NULLIF(SUM(sales),0)*100)

Allowed time_grain:
- "none", "week", "month", "quarter", "year"

Allowed breakdown_by (optional):
- null, "region", "segment", "category"

Allowed compare_period (only for intent="kpi_compare"):
- null, "prev_period", "mom", "yoy"

Date rules:
- start_date/end_date must be "YYYY-MM-DD".
- If the question does NOT specify dates, use the dashboard default range:
  start_date="{default_start}", end_date="{default_end}".

Filter rules:
- filters must be an object with keys: region, segment, category
- each value must be an array
- IMPORTANT: you may ONLY choose values from the CURRENT dashboard selections below (do not invent new ones).
  regions={json.dumps(regions)}
  segments={json.dumps(segments)}
  categories={json.dumps(categories)}

Output JSON schema (examples):

1) Single KPI value:
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

2) Trend of 2 metrics by month:
{{
  "intent": "kpi_trend",
  "metrics": ["sales","profit"],
  "time_grain": "month",
  "breakdown_by": null,
  "start_date": "{default_start}",
  "end_date": "{default_end}",
  "compare_period": null,
  "top_k": null,
  "order_by": "sales",
  "filters": {{"region": [], "segment": [], "category": []}}
}}

3) Top 5 categories by profit:
{{
  "intent": "kpi_rank",
  "metrics": ["profit"],
  "time_grain": "none",
  "breakdown_by": "category",
  "start_date": "{default_start}",
  "end_date": "{default_end}",
  "compare_period": null,
  "top_k": 5,
  "order_by": "profit",
  "filters": {{"region": [], "segment": [], "category": []}}
}}

4) YoY compare:
{{
  "intent": "kpi_compare",
  "metrics": ["sales"],
  "time_grain": "none",
  "breakdown_by": null,
  "start_date": "{default_start}",
  "end_date": "{default_end}",
  "compare_period": "yoy",
  "top_k": null,
  "order_by": "sales",
  "filters": {{"region": [], "segment": [], "category": []}}
}}

5) Clarify (when needed):
{{
  "intent": "clarify",
  "clarifying_question": "Which metric do you want: sales, profit, orders, or profit margin?",
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

Now plan this user question:
{user_question}
""".strip()

    def _gemini_plan(self, q: str) -> Dict[str, Any]:
        if not self.gemini_ready:
            raise RuntimeError("Gemini API key missing")

        prompt = self._plan_prompt(q)
        resp = self.model.generate_content(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 300},
        )

        text = (resp.text or "").strip()

        # Tolerate accidental wrapping text by extracting the first JSON object.
        plan = self._extract_json_object(text)
        return plan

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        """Extract the first JSON object from a string."""
        # Common: model returns exactly JSON; fast path
        try:
            return json.loads(text)
        except Exception:
            pass

        # Otherwise, find first {...}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("Gemini did not return JSON.")
        return json.loads(m.group(0))

    # ---------------------------
    # Validation + SQL generation
    # ---------------------------
    def _validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the plan; raise if invalid."""
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object.")

        intent = plan.get("intent")
        if intent not in {"kpi_value", "kpi_trend", "kpi_rank", "kpi_compare", "clarify"}:
            raise ValueError(f"Invalid intent: {intent}")

        # Clarify path (LLM should include a question)
        if intent == "clarify":
            cq = (plan.get("clarifying_question") or "").strip()
            if not cq:
                raise ValueError("clarifying_question is required when intent='clarify'.")
            # still normalize the rest so the app can show context if desired
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

        time_grain = plan.get("time_grain")
        if time_grain not in self._TIME_GRAINS:
            raise ValueError(f"Invalid time_grain: {time_grain}")

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
            # keep it simple: only allow ordering by one of the chosen metrics
            order_by = metrics[0]

        # Date parsing (allow empty -> default dashboard range)
        s0, e0 = self._get_dashboard_date_range()
        start_date = plan.get("start_date") or s0
        end_date = plan.get("end_date") or e0

        sd = self._parse_date(start_date)
        ed = self._parse_date(end_date)
        if sd is None or ed is None:
            raise ValueError("start_date/end_date must be YYYY-MM-DD.")
        if sd > ed:
            raise ValueError("start_date must be <= end_date.")

        # Validate filters: must be subsets of current selected filters
        filters = plan.get("filters")
        if not isinstance(filters, dict):
            raise ValueError("filters must be an object.")

        current_regions, current_segments, current_categories = self._get_current_filter_lists()
        allowed = {"region": set(current_regions), "segment": set(current_segments), "category": set(current_categories)}

        norm_filters: Dict[str, List[str]] = {"region": [], "segment": [], "category": []}
        for dim in ["region", "segment", "category"]:
            vals = filters.get(dim, [])
            if vals is None:
                vals = []
            if not isinstance(vals, list):
                raise ValueError(f"filters.{dim} must be an array.")
            vals = [str(v) for v in vals]
            if vals and not set(vals).issubset(allowed[dim]):
                bad = sorted(set(vals) - allowed[dim])
                raise ValueError(f"filters.{dim} contains values not in current dashboard selection: {bad}")
            norm_filters[dim] = vals

        # Intent-specific constraints
        if intent == "kpi_rank":
            if not breakdown_by:
                raise ValueError("breakdown_by is required for intent='kpi_rank'.")
            if top_k is None:
                raise ValueError("top_k is required for intent='kpi_rank'.")
            if time_grain != "none":
                raise ValueError("intent='kpi_rank' currently supports time_grain='none' only.")
        if intent == "kpi_compare":
            if compare_period is None:
                raise ValueError("compare_period is required for intent='kpi_compare'.")
            if len(metrics) != 1:
                raise ValueError("intent='kpi_compare' supports exactly 1 metric.")

        # Normalize plan
        plan_n = {
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
        return plan_n

    def _build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build a safe SQL query from a validated plan (no free-form identifiers)."""
        table = getattr(Config, "DB_TABLE", "superstore")

        intent = plan["intent"]
        metrics = plan["metrics"]
        time_grain = plan["time_grain"]
        breakdown_by = plan.get("breakdown_by")
        start = plan["start_date"]
        end = plan["end_date"]
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
        params: Dict[str, Any] = {"start": start, "end": end}

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

        # Time bucket
        bucket_sql = None
        if time_grain != "none":
            if time_grain == "week":
                bucket_sql = "DATE_TRUNC('week', order_date)"
            elif time_grain == "month":
                bucket_sql = "DATE_TRUNC('month', order_date)"
            elif time_grain == "quarter":
                bucket_sql = "DATE_TRUNC('quarter', order_date)"
            elif time_grain == "year":
                bucket_sql = "DATE_TRUNC('year', order_date)"
            else:
                raise ValueError(f"Unsupported time_grain: {time_grain}")

        # SELECT parts
        select_parts: List[str] = []
        group_parts: List[str] = []
        order_parts: List[str] = []

        if bucket_sql:
            select_parts.append(f"{bucket_sql} AS period")
            group_parts.append("period")
            order_parts.append("period ASC")

        if breakdown_by:
            select_parts.append(f"{breakdown_by} AS breakdown")
            group_parts.append("breakdown")

        for m in metrics:
            select_parts.append(f"{metric_expr(m)} AS {m}")

        # Base query
        sql_lines = [
            "SELECT " + ", ".join(select_parts),
            f"FROM {table}",
            f"WHERE {where_sql}",
        ]
        if group_parts:
            sql_lines.append("GROUP BY " + ", ".join(group_parts))

        # Ordering / ranking
        if intent in {"kpi_rank"}:
            # rank always desc by the metric
            sql_lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
            sql_lines.append("LIMIT %(top_k)s")
            params["top_k"] = top_k
        elif breakdown_by and not bucket_sql:
            # grouped value -> show largest first
            sql_lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
        elif bucket_sql:
            # already period asc; if also breakdown, keep stable ordering
            if breakdown_by:
                sql_lines.append("ORDER BY period ASC, breakdown ASC")
            else:
                sql_lines.append("ORDER BY period ASC")

        return "\n".join(sql_lines), params

    def _run_plan(self, plan: Dict[str, Any]) -> pd.DataFrame:
        intent = plan["intent"]

        if intent == "kpi_compare":
            # Current period
            sql, params = self._build_sql(plan)
            cur = execute_query(sql, params=params)

            # Prior period dates
            sd = datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
            ed = datetime.strptime(plan["end_date"], "%Y-%m-%d").date()

            cp = plan["compare_period"]
            if cp == "prev_period":
                delta = (ed - sd).days + 1
                prev_end = sd.fromordinal(sd.toordinal() - 1)
                prev_start = prev_end.fromordinal(prev_end.toordinal() - (delta - 1))
            elif cp == "mom":
                # shift by 1 month (same calendar dates where possible)
                prev_start = (pd.Timestamp(sd) - pd.DateOffset(months=1)).date()
                prev_end = (pd.Timestamp(ed) - pd.DateOffset(months=1)).date()
            elif cp == "yoy":
                prev_start = (pd.Timestamp(sd) - pd.DateOffset(years=1)).date()
                prev_end = (pd.Timestamp(ed) - pd.DateOffset(years=1)).date()
            else:
                raise ValueError(f"Unsupported compare_period: {cp}")

            plan_prev = dict(plan)
            plan_prev["start_date"] = prev_start.strftime("%Y-%m-%d")
            plan_prev["end_date"] = prev_end.strftime("%Y-%m-%d")

            sql2, params2 = self._build_sql(plan_prev)
            prev = execute_query(sql2, params=params2)

            # Return a single-row frame with both values
            metric = plan["metrics"][0]
            cur_val = float(cur.iloc[0][metric]) if cur is not None and not cur.empty else 0.0
            prev_val = float(prev.iloc[0][metric]) if prev is not None and not prev.empty else 0.0
            out = pd.DataFrame(
                [{
                    "metric": metric,
                    "current_start": plan["start_date"],
                    "current_end": plan["end_date"],
                    "prev_start": plan_prev["start_date"],
                    "prev_end": plan_prev["end_date"],
                    "current": cur_val,
                    "previous": prev_val,
                }]
            )
            return out

        # Normal query
        sql, params = self._build_sql(plan)
        return execute_query(sql, params=params)

    @staticmethod
    def _format_money(x: float) -> str:
        return f"${x:,.0f}"

    def _explain_query(self, plan: Dict[str, Any]) -> str:
        parts = []
        parts.append(f"Period: {plan['start_date']} → {plan['end_date']}")
        f = plan.get("filters") or {}
        for dim in ["region", "segment", "category"]:
            vals = f.get(dim) or []
            if vals:
                parts.append(f"{dim}={', '.join(vals)}")
        if plan.get("breakdown_by"):
            parts.append(f"breakdown_by={plan['breakdown_by']}")
        if plan.get("time_grain") and plan["time_grain"] != "none":
            parts.append(f"time_grain={plan['time_grain']}")
        if plan.get("intent") == "kpi_compare":
            parts.append(f"compare={plan.get('compare_period')}")
        return " • ".join(parts)

    def _format_answer(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        # Clarify path
        if plan["intent"] == "clarify":
            return plan.get("clarifying_question", "Could you clarify your question?")

        explain = self._explain_query(plan)
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

        if df is None or df.empty:
            return f"No data found. ({explain})"

        # Compare
        if intent == "kpi_compare":
            row = df.iloc[0]
            m = row["metric"]
            cur = float(row["current"])
            prev = float(row["previous"])
            if prev == 0:
                change = None
            else:
                change = (cur - prev) / abs(prev) * 100.0

            if m in {"sales", "profit"}:
                cur_s = self._format_money(cur)
                prev_s = self._format_money(prev)
            elif m == "profit_margin":
                cur_s = f"{cur:.2f}%"
                prev_s = f"{prev:.2f}%"
            else:
                cur_s = f"{int(cur):,}"
                prev_s = f"{int(prev):,}"

            if change is None:
                delta_s = "n/a"
            else:
                delta_s = f"{change:+.1f}%"

            return "\n".join([
                f"**{metric_labels[m]}** ({explain})",
                f"- Current: {cur_s}",
                f"- Previous: {prev_s}",
                f"- Change: {delta_s}",
            ])

        # KPI value (optionally grouped)
        if intent == "kpi_value" and time_grain == "none":
            header = f"**{', '.join(metric_labels[m] for m in metrics)}** ({explain})"
            if breakdown_by:
                lines = [header]
                # Expect: breakdown + metric columns
                show = df.copy()
                show = show.sort_values(by=plan.get("order_by") or metrics[0], ascending=False).head(20)
                for _, r in show.iterrows():
                    b = r.get("breakdown")
                    vals = []
                    for m in metrics:
                        v = r.get(m)
                        if m in {"sales", "profit"}:
                            vals.append(self._format_money(float(v)))
                        elif m == "profit_margin":
                            vals.append(f"{float(v):.2f}%")
                        else:
                            vals.append(f"{int(v):,}")
                    lines.append(f"- {b}: " + " | ".join(f"{metric_labels[m]} {vals[i]}" for i, m in enumerate(metrics)))
                return "\n".join(lines)

            # Single row
            r0 = df.iloc[0]
            lines = [header]
            for m in metrics:
                v = float(r0[m])
                if m in {"sales", "profit"}:
                    vs = self._format_money(v)
                elif m == "profit_margin":
                    vs = f"{v:.2f}%"
                else:
                    vs = f"{int(v):,}"
                lines.append(f"- {metric_labels[m]}: {vs}")
            return "\n".join(lines)

        # Trend
        if intent == "kpi_trend" and time_grain != "none":
            header = f"**Trend** ({explain})"
            lines = [header]
            show = df.copy()

            # Keep only recent points to avoid spam
            if "period" in show.columns:
                show = show.sort_values("period").tail(24)

            if breakdown_by:
                # Show top few breakdowns
                # We will pivot: for each breakdown, last value of first metric
                m0 = metrics[0]
                last = show.groupby("breakdown")[m0].tail(1)
                top_breakdowns = (
                    show.groupby("breakdown")[m0].sum().sort_values(ascending=False).head(5).index.tolist()
                )
                show = show[show["breakdown"].isin(top_breakdowns)]

            for _, r in show.iterrows():
                p = r.get("period")
                p = str(p)[:10] if p is not None else ""
                vals = []
                for m in metrics:
                    v = float(r[m])
                    if m in {"sales", "profit"}:
                        vals.append(self._format_money(v))
                    elif m == "profit_margin":
                        vals.append(f"{v:.2f}%")
                    else:
                        vals.append(f"{int(v):,}")
                if breakdown_by:
                    b = r.get("breakdown")
                    lines.append(f"- {p} • {b}: " + " | ".join(f"{metric_labels[m]} {vals[i]}" for i, m in enumerate(metrics)))
                else:
                    lines.append(f"- {p}: " + " | ".join(f"{metric_labels[m]} {vals[i]}" for i, m in enumerate(metrics)))

            if len(lines) > 30:
                lines = lines[:30] + ["- ... (truncated)"]
            return "\n".join(lines)

        # Rank (top-k)
        if intent == "kpi_rank":
            m0 = metrics[0]
            header = f"**Top {plan['top_k']} by {metric_labels[m0]}** ({explain})"
            lines = [header]
            show = df.copy().head(plan["top_k"])
            for i, (_, r) in enumerate(show.iterrows(), start=1):
                b = r.get("breakdown")
                v = float(r[m0])
                if m0 in {"sales", "profit"}:
                    vs = self._format_money(v)
                elif m0 == "profit_margin":
                    vs = f"{v:.2f}%"
                else:
                    vs = f"{int(v):,}"
                lines.append(f"{i}. {b}: {vs}")
            return "\n".join(lines)

        # Fallback
        return f"Done. ({explain})"

    # ---------------------------
    # Helpers: current filters
    # ---------------------------
    def _get_current_filter_lists(self) -> Tuple[List[str], List[str], List[str]]:
        """Return the currently selected region/segment/category lists from dashboard filters."""
        f = self.filters or {}
        regions = list(f.get("region") or [])
        segments = list(f.get("segment") or [])
        categories = list(f.get("category") or [])
        return regions, segments, categories

    def _get_dashboard_date_range(self) -> Tuple[str, str]:
        """Return current dashboard date range as strings YYYY-MM-DD."""
        f = self.filters or {}
        dr = f.get("date_range")
        if dr and isinstance(dr, (tuple, list)) and len(dr) == 2:
            s, e = dr
            # Streamlit date_input returns date objects; accept strings too
            if isinstance(s, (datetime, date)):
                s = s.strftime("%Y-%m-%d")
            else:
                s = str(s)
            if isinstance(e, (datetime, date)):
                e = e.strftime("%Y-%m-%d")
            else:
                e = str(e)
            return s, e

        # Fallback to dataset range if missing
        if "order_date" in self.df.columns and not self.df.empty:
            s = pd.to_datetime(self.df["order_date"].min()).date().strftime("%Y-%m-%d")
            e = pd.to_datetime(self.df["order_date"].max()).date().strftime("%Y-%m-%d")
            return s, e
        return "1900-01-01", "2100-01-01"

    # ---------------------------
    # Public: main response
    # ---------------------------
    def get_response(self, user_question: str) -> str:
        q = (user_question or "").strip()
        if not q:
            return "Ask me about Total Sales, Total Profit, Total Orders, or Profit Margin."

        # 1) Fast KPI path
        fast = self._fast_kpi_answer(q)
        if fast:
            return fast

        # 2_toggle: Gemini must be configured for query planning
        if not self.gemini_ready:
            return "⚠️ Gemini API Key is missing in .env. I cannot parse questions into queries."

        # 2) Query path: Gemini -> plan -> validate -> SQL -> answer
        try:
            plan = self._gemini_plan(q)
            plan = self._validate_plan(plan)
            result_df = self._run_plan(plan)
            return self._format_answer(plan, result_df)
        except Exception as e:
            return f"❌ Sorry — I couldn't answer that. ({str(e)})"

    # Optional: keep your old insights feature (still works, but now it's separate)
    def get_insights(self) -> str:
        if not self.gemini_ready:
            return "Configure Gemini API Key to see auto-insights."

        # Minimal context for insights (English-only).
        # Use the filtered df you pass in (already respects sidebar filters).
        df = self.df.copy()
        if df.empty:
            return "No data available for insights."

        # Simple aggregates (keep short to reduce tokens)
        try:
            by_region = df.groupby("region")[["sales", "profit"]].sum().sort_values("profit", ascending=False).head(5)
            by_segment = df.groupby("segment")[["sales", "profit"]].sum().sort_values("profit", ascending=False).head(3)
            prompt = f"""
You are a business analyst. Provide exactly 3 bullet insights based on the aggregates below.
Keep it concise, numbers-first, English only.

Top regions:
{by_region.to_string()}

Top segments:
{by_segment.to_string()}
""".strip()

            resp = self.model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 200})
            return (resp.text or "").strip()
        except Exception:
            return "Could not generate insights at this moment."
