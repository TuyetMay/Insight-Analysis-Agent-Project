"""
chatbot/sql_builder.py
Converts a validated plan dict into parameterised SQL and executes it.
Pure data logic — no LLM, no UI.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

from config import Config
from core.database import execute_query


_METRIC_EXPR: Dict[str, str] = {
    "sales":         "SUM(sales)",
    "profit":        "SUM(profit)",
    "orders":        "COUNT(DISTINCT order_id)",
    "profit_margin": "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END",
}

_GRAIN_MAP: Dict[str, str] = {
    "week": "week", "month": "month", "quarter": "quarter", "year": "year",
}


class SQLBuilder:
    """Build and run SQL for validated query plans."""

    def __init__(self, table: str = "") -> None:
        self.table = table or Config.DB_TABLE

    # ── Public ────────────────────────────────────────────────

    def run(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """Execute plan and return a DataFrame."""
        if plan["intent"] == "kpi_compare":
            return self._run_compare(plan)
        sql, params = self.build_sql(plan)
        return execute_query(sql, params)

    def build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Translate a validated plan into (sql_string, params_dict)."""
        metrics       = plan["metrics"]
        time_grain    = plan["time_grain"]
        breakdown_by  = plan.get("breakdown_by")
        f             = plan["filters"]
        top_k         = plan.get("top_k")
        order_by      = plan.get("order_by") or metrics[0]

        where_parts = ["order_date >= %(start)s", "order_date <= %(end)s"]
        params: Dict[str, Any] = {"start": plan["start_date"], "end": plan["end_date"]}

        # ✅ FIX: added sub_category to WHERE clause builders
        for col in ("region", "segment", "category", "sub_category"):
            vals = f.get(col)
            if vals:
                placeholders = ", ".join(f"%({col}_{i})s" for i in range(len(vals)))
                where_parts.append(f"{col} IN ({placeholders})")
                for i, v in enumerate(vals):
                    params[f"{col}_{i}"] = v

        where_sql = " AND ".join(where_parts)

        bucket_sql = (
            f"DATE_TRUNC('{_GRAIN_MAP[time_grain]}', order_date)"
            if time_grain != "none" else None
        )

        select_parts: List[str] = []
        group_parts: List[str] = []

        if bucket_sql:
            select_parts.append(f"{bucket_sql} AS period")
            group_parts.append("period")
        if breakdown_by:
            select_parts.append(f"{breakdown_by} AS breakdown")
            group_parts.append("breakdown")
        for m in metrics:
            select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        lines = [
            "SELECT " + ", ".join(select_parts),
            f"FROM {self.table}",
            f"WHERE {where_sql}",
        ]
        if group_parts:
            lines.append("GROUP BY " + ", ".join(group_parts))
        if plan["intent"] == "kpi_value" and not bucket_sql and not breakdown_by:
            supporting = [m for m in ["sales", "profit"] if m not in metrics]
            for m in supporting:
                select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")
        elif plan["intent"] == "kpi_rank":
            lines += [f"ORDER BY {order_by} DESC NULLS LAST", "LIMIT %(top_k)s"]
            params["top_k"] = top_k
        elif breakdown_by and not bucket_sql:
            lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
        elif bucket_sql:
            order = "period ASC, breakdown ASC" if breakdown_by else "period ASC"
            lines.append(f"ORDER BY {order}")

        return "\n".join(lines), params

    # ── Period comparison ─────────────────────────────────────

    def _run_compare(self, plan: Dict[str, Any]) -> pd.DataFrame:
        sql, params = self.build_sql(plan)
        cur_df = execute_query(sql, params)

        prev_plan = {**plan, **self._prev_dates(plan)}
        sql2, params2 = self.build_sql(prev_plan)
        prev_df = execute_query(sql2, params2)

        metric = plan["metrics"][0]

        def safe_float(df: pd.DataFrame) -> float:
            if df is None or df.empty:
                return 0.0
            val = df.iloc[0].get(metric)
            try:
                return float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        cur_val  = safe_float(cur_df)
        prev_val = safe_float(prev_df)

        return pd.DataFrame([{
            "metric":        metric,
            "current_start": plan["start_date"],
            "current_end":   plan["end_date"],
            "prev_start":    prev_plan["start_date"],
            "prev_end":      prev_plan["end_date"],
            "current":       cur_val,
            "previous":      prev_val,
        }])

    @staticmethod
    def _prev_dates(plan: Dict[str, Any]) -> Dict[str, str]:
        sd = datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
        ed = datetime.strptime(plan["end_date"],   "%Y-%m-%d").date()
        cp = plan["compare_period"]

        if cp == "prev_period":
            delta = (ed - sd).days + 1
            prev_end   = date.fromordinal(sd.toordinal() - 1)
            prev_start = date.fromordinal(prev_end.toordinal() - (delta - 1))
        elif cp == "mom":
            prev_start = (pd.Timestamp(sd) - pd.DateOffset(months=1)).date()
            prev_end   = (pd.Timestamp(ed) - pd.DateOffset(months=1)).date()
        elif cp == "yoy":
            prev_start = (pd.Timestamp(sd) - pd.DateOffset(years=1)).date()
            prev_end   = (pd.Timestamp(ed) - pd.DateOffset(years=1)).date()
        else:
            raise ValueError(f"Unsupported compare_period: {cp!r}")

        return {"start_date": prev_start.strftime("%Y-%m-%d"),
                "end_date":   prev_end.strftime("%Y-%m-%d")}