"""
chatbot/sql_builder.py
Converts a validated plan dict into parameterised SQL and executes it.
Pure data logic — no LLM, no UI.
"""

from __future__ import annotations

from datetime import date, datetime
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import re as _re

from config import Config
from core.database import execute_query

logger = logging.getLogger(__name__)

_METRIC_EXPR: Dict[str, str] = {
    "sales":         "SUM(sales)",
    "profit":        "SUM(profit)",
    "orders":        "COUNT(DISTINCT order_id)",
    "profit_margin": "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END",
}
_SAFE_TABLE_RE = _re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

_GRAIN_MAP: Dict[str, str] = {
    "week": "week", "month": "month", "quarter": "quarter", "year": "year",
}

_CONDITION_WHERE: Dict[str, str] = {
    "profit_negative": "profit < 0",
    "loss_orders":     "profit < 0",
    "profit_positive": "profit > 0",
    "high_discount":   "discount > 0.2",
}

_ALLOWED_BREAKDOWN_COLS: frozenset = frozenset({
    "region", "segment", "category", "sub_category", "state", "ship_mode"
})
 

class SQLBuilder:
    """Build and run SQL for validated query plans."""

    def __init__(self, table: str = "") -> None:
        raw = table or Config.DB_TABLE
        if not _SAFE_TABLE_RE.match(raw):
            raise ValueError(f"Unsafe table name: {raw!r}")
        self.table = raw

    # ── Public ────────────────────────────────────────────────

    def run(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """Execute plan and return a DataFrame."""
        if plan["intent"] == "kpi_compare":
            return self._run_compare(plan)
        if plan["intent"] == "kpi_detail":
            return self._run_detail(plan)
        # Cross-breakdown: GROUP BY primary × secondary
        if plan.get("secondary_breakdown"):
            return self._run_cross(plan)
        sql, params = self.build_sql(plan)
        return execute_query(sql, params)

    def build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Translate a validated plan into (sql_string, params_dict)."""
        metrics             = plan["metrics"]
        time_grain          = plan["time_grain"]
        breakdown_by        = plan.get("breakdown_by")
        secondary_breakdown = plan.get("secondary_breakdown")  # may be None
        f                   = plan["filters"]
        top_k               = plan.get("top_k")
        order_by            = plan.get("order_by") or metrics[0]

        # ── WHERE ─────────────────────────────────────────────
        where_parts = ["order_date >= %(start)s", "order_date <= %(end)s"]
        params: Dict[str, Any] = {"start": plan["start_date"], "end": plan["end_date"]}

        for col in ("region", "segment", "category", "sub_category", "state"):
            vals = f.get(col)
            if vals:
                placeholders = ", ".join(f"%({col}_{i})s" for i in range(len(vals)))
                where_parts.append(f"{col} IN ({placeholders})")
                for i, v in enumerate(vals):
                    params[f"{col}_{i}"] = v

        where_sql = " AND ".join(where_parts)

        # ── SELECT / GROUP BY ─────────────────────────────────
        bucket_sql = (
            f"DATE_TRUNC('{_GRAIN_MAP[time_grain]}', order_date)"
            if time_grain != "none" else None
        )

        select_parts: List[str] = []
        group_parts:  List[str] = []

        if bucket_sql:
            select_parts.append(f"{bucket_sql} AS period")
            group_parts.append("period")

        if breakdown_by:
            select_parts.append(f"{breakdown_by} AS breakdown")
            group_parts.append("breakdown")

        # secondary_breakdown adds a second grouping column
        if secondary_breakdown:
            select_parts.append(f"{secondary_breakdown} AS breakdown2")
            group_parts.append("breakdown2")

        for m in metrics:
            select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        # ── Supporting metrics for bare kpi_value ─────────────
        if plan["intent"] == "kpi_value" and not bucket_sql and not breakdown_by and not secondary_breakdown:
            for m in ["sales", "profit"]:
                if m not in metrics:
                    select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        # ── Assemble SQL ──────────────────────────────────────
        lines = [
            "SELECT " + ", ".join(select_parts),
            f"FROM {self.table}",
            f"WHERE {where_sql}",
        ]

        if group_parts:
            lines.append("GROUP BY " + ", ".join(group_parts))

        if plan["intent"] == "kpi_rank":
            lines += [f"ORDER BY {order_by} DESC NULLS LAST", "LIMIT %(top_k)s"]
            params["top_k"] = top_k
        elif secondary_breakdown:
            # cross-breakdown: sort by primary then metric desc
            lines.append(f"ORDER BY breakdown ASC, {order_by} DESC NULLS LAST")
        elif breakdown_by and not bucket_sql:
            lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
        elif bucket_sql:
            order = "period ASC, breakdown ASC" if breakdown_by else "period ASC"
            lines.append(f"ORDER BY {order}")

        return "\n".join(lines), params

    # ── Cross-breakdown query (e.g. region × category) ────────

    def _run_cross(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Run a GROUP BY primary × secondary query and return a tidy DataFrame
        with columns: breakdown, breakdown2, <metric(s)>.
        Called by run() when plan["secondary_breakdown"] is set.
        """
        sql, params = self.build_sql(plan)
        df = execute_query(sql, params)
        # Tag so formatter knows this is a cross result
        if not df.empty:
            df.attrs["cross_breakdown"] = True
            df.attrs["primary_dim"]     = plan.get("breakdown_by", "")
            df.attrs["secondary_dim"]   = plan.get("secondary_breakdown", "")
        return df

    # ── Detail query (negative profit / drill-down) ───────────

    def _run_detail(self, plan: Dict[str, Any]) -> pd.DataFrame:
        condition = plan.get("condition", "profit_negative")
        breakdown = plan.get("breakdown_by") or "sub_category"
 
        # ── THÊM: whitelist check ─────────────────────────────
        if breakdown not in _ALLOWED_BREAKDOWN_COLS:
            logger.warning("Blocked unsafe breakdown column: %r", breakdown)
            breakdown = "sub_category"
        # ─────────────────────────────────────────────────────
 
        f     = plan["filters"]
        top_k = plan.get("top_k") or 15
 
        base_parts: List[str] = [
            "order_date >= %(start)s",
            "order_date <= %(end)s",
        ]
        params: Dict[str, Any] = {"start": plan["start_date"], "end": plan["end_date"]}
 
        for col in ("region", "segment", "category", "sub_category", "state"):
            vals = f.get(col)
            if vals:
                placeholders = ", ".join(f"%({col}_{i})s" for i in range(len(vals)))
                base_parts.append(f"{col} IN ({placeholders})")
                for i, v in enumerate(vals):
                    params[f"{col}_{i}"] = v
 
        base_where = " AND ".join(base_parts)
        params_g = {**params, "top_k": top_k}
 
        # ── Dùng f-string CHỈ cho whitelisted column name ────
        # breakdown đã được validate ở trên → safe
        group_sql = f"""
            SELECT
                {breakdown}        AS breakdown,
                category,
                COUNT(DISTINCT order_id)                          AS orders,
                ROUND(CAST(SUM(sales)   AS DECIMAL), 2)          AS sales,
                ROUND(CAST(SUM(profit)  AS DECIMAL), 2)          AS profit,
                ROUND(CAST(AVG(discount) * 100 AS DECIMAL), 1)   AS avg_discount_pct,
                CASE WHEN SUM(sales) = 0 THEN 0
                     ELSE ROUND(CAST(SUM(profit)/SUM(sales)*100 AS DECIMAL), 2)
                END                                               AS profit_margin
            FROM {self.table}
            WHERE {base_where}
            GROUP BY {breakdown}, category
            HAVING SUM(profit) < 0
            ORDER BY SUM(profit) ASC
            LIMIT %(top_k)s
        """
        grouped_df = execute_query(group_sql, params_g)
 
        # Sample orders — giữ nguyên logic cũ, chỉ dùng f-string cho breakdown
        row_where  = base_where + " AND profit < 0"
        params_s   = {**params, "sample_k": 10}
        sample_df  = pd.DataFrame()
 
        sample_sql = f"""
            SELECT order_id, order_date,
                   {breakdown} AS breakdown, category, product_name,
                   sales, profit, discount
            FROM {self.table}
            WHERE {row_where}
            ORDER BY profit ASC
            LIMIT %(sample_k)s
        """
        try:
            sample_df = execute_query(sample_sql, params_s)
        except Exception:
            pass
 
        if not grouped_df.empty:
            grouped_df.attrs["detail_type"]   = "grouped_summary"
            grouped_df.attrs["condition"]     = condition
            grouped_df.attrs["sample_orders"] = sample_df
 
        return grouped_df

    # ── Period comparison ─────────────────────────────────────

    def _run_compare(self, plan: Dict[str, Any]) -> pd.DataFrame:
        breakdown = plan.get("breakdown_by")

        if breakdown:
            cur_sql, cur_params = self.build_sql(plan)
            prev_plan = {**plan, **self._prev_dates(plan)}
            prv_sql, prv_params = self.build_sql(prev_plan)

            cur_df = execute_query(cur_sql, cur_params)
            prv_df = execute_query(prv_sql, prv_params)

            metric = plan["metrics"][0]
            if cur_df.empty:
                return pd.DataFrame()

            merged = cur_df[["breakdown", metric]].rename(columns={metric: "current"})
            if not prv_df.empty:
                merged = merged.merge(
                    prv_df[["breakdown", metric]].rename(columns={metric: "previous"}),
                    on="breakdown", how="left"
                )
            else:
                merged["previous"] = 0.0

            merged["previous"]   = merged["previous"].fillna(0.0)
            merged["change_pct"] = merged.apply(
                lambda r: ((r["current"] - r["previous"]) / abs(r["previous"]) * 100)
                if r["previous"] != 0 else None,
                axis=1,
            )
            merged["metric"]        = metric
            merged["current_start"] = plan["start_date"]
            merged["current_end"]   = plan["end_date"]
            merged["prev_start"]    = prev_plan["start_date"]
            merged["prev_end"]      = prev_plan["end_date"]
            return merged.sort_values("change_pct", ascending=True)

        sql, params = self.build_sql(plan)
        cur_df = execute_query(sql, params)
        prev_plan = {**plan, **self._prev_dates(plan)}
        sql2, params2 = self.build_sql(prev_plan)
        prev_df = execute_query(sql2, params2)

        metric = plan["metrics"][0]

        def safe_float(df):
            if df is None or df.empty: return 0.0
            val = df.iloc[0].get(metric)
            try:   return float(val) if val is not None else 0.0
            except: return 0.0

        return pd.DataFrame([{
            "metric":        metric,
            "current_start": plan["start_date"],
            "current_end":   plan["end_date"],
            "prev_start":    prev_plan["start_date"],
            "prev_end":      prev_plan["end_date"],
            "current":       safe_float(cur_df),
            "previous":      safe_float(prev_df),
        }])

    @staticmethod
    def _prev_dates(plan: Dict[str, Any]) -> Dict[str, str]:
        sd = datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
        ed = datetime.strptime(plan["end_date"],   "%Y-%m-%d").date()
        cp = plan["compare_period"]

        if cp == "prev_period":
            delta      = (ed - sd).days + 1
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