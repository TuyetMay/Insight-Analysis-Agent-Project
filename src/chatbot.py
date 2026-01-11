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
    _TIME_GRAINS = {"none", "month", "year"}
    _DIM_FILTERS = {"region", "segment", "category"}

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
        # Give Gemini the currently active non-date filters so it doesn't try to invent them.
        regions, segments, categories = self._get_current_filter_lists()

        # Dashboard date range (used as default if question doesn't specify)
        default_start, default_end = self._get_dashboard_date_range()

        # Keep the instruction extremely strict.
        return f"""
You are a query planner for a Postgres Superstore analytics database.
Return ONLY a valid JSON object. Do NOT include markdown. Do NOT include SQL.

Goal: Convert the user's question into a small plan for computing one KPI.

Allowed metrics (choose exactly one):
- "sales" (SUM(sales))
- "profit" (SUM(profit))
- "orders" (COUNT(DISTINCT order_id))
- "profit_margin" (SUM(profit)/SUM(sales)*100)

Allowed time_grain:
- "none" (one number)
- "month" (time series by month)
- "year" (time series by year)

Date logic:
- If the user explicitly mentions a year/month/quarter/range, set start_date/end_date accordingly.
- Otherwise use the default dashboard range: start_date="{default_start}" end_date="{default_end}".

Filters:
- Always respect the dashboard's current filters for region/segment/category.
- You MUST NOT invent new filter values.
- Use ONLY these exact lists:

region options currently selected: {regions}
segment options currently selected: {segments}
category options currently selected: {categories}

Output schema (all keys required):
{{
  "metric": "sales|profit|orders|profit_margin",
  "time_grain": "none|month|year",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "filters": {{
    "region": [..],      // must be subset of the selected region list
    "segment": [..],     // must be subset of the selected segment list
    "category": [..]     // must be subset of the selected category list
  }},
  "intent": "kpi_value|kpi_trend"
}}

Rules:
- If the user asks for a single period (e.g., \"March 2017\"), return time_grain="none" and that month range.
- If the user asks for a trend (e.g., \"monthly trend in 2017\"), return time_grain="month" and intent="kpi_trend".
- If unclear, pick the safest interpretation: one number (intent="kpi_value", time_grain="none").
- Return strict JSON only.

User question:
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

        metric = plan.get("metric")
        time_grain = plan.get("time_grain")
        start_date = plan.get("start_date")
        end_date = plan.get("end_date")
        intent = plan.get("intent")
        filters = plan.get("filters")

        if metric not in self._METRICS:
            raise ValueError(f"Invalid metric: {metric}")
        if time_grain not in self._TIME_GRAINS:
            raise ValueError(f"Invalid time_grain: {time_grain}")
        if intent not in {"kpi_value", "kpi_trend"}:
            raise ValueError(f"Invalid intent: {intent}")

        # Parse dates
        sd = self._parse_date(start_date)
        ed = self._parse_date(end_date)
        if sd is None or ed is None:
            raise ValueError("start_date/end_date must be YYYY-MM-DD.")
        if sd > ed:
            raise ValueError("start_date must be <= end_date.")

        # Validate filters: must be subsets of current selected filters
        if not isinstance(filters, dict):
            raise ValueError("filters must be an object.")

        current_regions, current_segments, current_categories = self._get_current_filter_lists()
        allowed = {
            "region": set(current_regions),
            "segment": set(current_segments),
            "category": set(current_categories),
        }

        norm_filters: Dict[str, List[str]] = {}
        for k in self._DIM_FILTERS:
            vals = filters.get(k, [])
            if vals is None:
                vals = []
            if not isinstance(vals, list):
                raise ValueError(f"filters.{k} must be a list.")
            # Only allow subset; if empty -> use current selection (safer UX)
            if len(vals) == 0:
                norm_filters[k] = list(allowed[k])
                continue
            if not set(vals).issubset(allowed[k]):
                raise ValueError(f"filters.{k} includes values outside current dashboard selection.")
            norm_filters[k] = vals

        return {
            "metric": metric,
            "time_grain": time_grain,
            "start_date": sd.strftime("%Y-%m-%d"),
            "end_date": ed.strftime("%Y-%m-%d"),
            "filters": norm_filters,
            "intent": intent,
        }

    @staticmethod
    def _parse_date(s: Any) -> Optional[date]:
        if not isinstance(s, str):
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    def _build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build parameterized SQL for Postgres."""
        metric = plan["metric"]
        time_grain = plan["time_grain"]
        start = plan["start_date"]
        end = plan["end_date"]
        f = plan["filters"]

        table = getattr(Config, "DB_TABLE", "superstore")

        metric_expr_map = {
            "orders": "COUNT(DISTINCT order_id)",
            "sales": "SUM(sales)",
            "profit": "SUM(profit)",
            "profit_margin": "CASE WHEN SUM(sales) = 0 THEN 0 ELSE (SUM(profit) / SUM(sales)) * 100 END",
        }
        metric_expr = metric_expr_map[metric]

        where_parts = ["order_date >= %(start)s", "order_date <= %(end)s"]
        params: Dict[str, Any] = {"start": start, "end": end}

        # Always apply current filters (validated)
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

        if time_grain == "none":
            sql = f"""
                SELECT {metric_expr} AS value
                FROM {table}
                WHERE {where_sql}
            """.strip()
            return sql, params

        # Trend query
        if time_grain == "month":
            bucket = "DATE_TRUNC('month', order_date)"
        elif time_grain == "year":
            bucket = "DATE_TRUNC('year', order_date)"
        else:
            raise ValueError("Unsupported time_grain")

        sql = f"""
            SELECT {bucket} AS period, {metric_expr} AS value
            FROM {table}
            WHERE {where_sql}
            GROUP BY period
            ORDER BY period
        """.strip()
        return sql, params

    # ---------------------------
    # Execution + formatting
    # ---------------------------
    def _run_plan(self, plan: Dict[str, Any]) -> pd.DataFrame:
        sql, params = self._build_sql(plan)
        return execute_query(sql, params=params)

    @staticmethod
    def _format_money(x: float) -> str:
        return f"${x:,.0f}"

    def _format_answer(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        metric = plan["metric"]
        time_grain = plan["time_grain"]
        start = plan["start_date"]
        end = plan["end_date"]

        metric_label = {
            "orders": "Total Orders",
            "sales": "Total Sales",
            "profit": "Total Profit",
            "profit_margin": "Profit Margin",
        }[metric]

        if df is None or df.empty:
            return f"No data found for {metric_label} between {start} and {end} (based on current filters)."

        if time_grain == "none":
            val = df.iloc[0]["value"]
            if metric in {"sales", "profit"}:
                return f"{metric_label} ({start} to {end}): {self._format_money(float(val))} (based on current filters)."
            if metric == "profit_margin":
                return f"{metric_label} ({start} to {end}): {float(val):.2f}% (based on current filters)."
            return f"{metric_label} ({start} to {end}): {int(val):,} (based on current filters)."

        # Trend formatting (keep concise)
        lines = [f"{metric_label} trend ({start} to {end}) by {time_grain} (based on current filters):"]
        for _, row in df.iterrows():
            period = row["period"]
            if isinstance(period, str):
                p = period
            else:
                try:
                    p = pd.to_datetime(period).strftime("%Y-%m")
                    if time_grain == "year":
                        p = pd.to_datetime(period).strftime("%Y")
                except Exception:
                    p = str(period)

            val = row["value"]
            if metric in {"sales", "profit"}:
                v = self._format_money(float(val))
            elif metric == "profit_margin":
                v = f"{float(val):.2f}%"
            else:
                v = f"{int(val):,}"
            lines.append(f"- {p}: {v}")

        # avoid overly long messages
        if len(lines) > 25:
            lines = lines[:25] + ["- ... (truncated)"]

        return "\n".join(lines)

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
