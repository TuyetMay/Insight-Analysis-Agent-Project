from __future__ import annotations
import logging
import re as _re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from core.database import execute_query

logger = logging.getLogger(__name__)

_SAFE_TABLE_RE = _re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

_METRIC_EXPR: Dict[str, str] = {
    "sales":         "SUM(sales)",
    "profit":        "SUM(profit)",
    "orders":        "COUNT(DISTINCT order_id)",
    "profit_margin": "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END",
}

_GRAIN_MAP: Dict[str, str] = {
    "week": "week", "month": "month", "quarter": "quarter", "year": "year",
}

_ALLOWED_BREAKDOWN_COLS: frozenset = frozenset({
    "region", "segment", "category", "sub_category", "state", "ship_mode",
})

_SEMANTIC_MAP: Dict[str, str] = {
    "revenue": "sales", "income": "sales", "earnings": "sales", "turnover": "sales",
    "gross": "sales", "net profit": "profit", "gain": "profit",
    "margin": "profit_margin", "profitability": "profit_margin",
    "transactions": "orders", "purchases": "orders",
    "area": "region", "zone": "region", "territory": "region",
    "market": "region", "geo": "region",
    "customer type": "segment", "customer segment": "segment",
    "product type": "category", "product group": "category",
    "item": "sub_category", "sku": "sub_category",
}


_DISTRIBUTION_BUCKETS: Dict[str, List[Tuple[str, str]]] = {
    "discount": [
        ("0%",    "discount = 0"),
        ("1-10%",  "discount > 0 AND discount <= 0.10"),
        ("11-20%", "discount > 0.10 AND discount <= 0.20"),
        ("21-30%", "discount > 0.20 AND discount <= 0.30"),
        (">30%",   "discount > 0.30"),
    ],
    "profit_margin": [
        ("<0%",    "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END < 0"),
        ("0-5%",   "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END BETWEEN 0 AND 5"),
        ("5-15%",  "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END BETWEEN 5 AND 15"),
        (">15%",   "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END > 15"),
    ],
    "sales": [   
        ("<$100",    "sales < 100"),
        ("$100-500", "sales BETWEEN 100 AND 500"),
        ("$500-1k",  "sales BETWEEN 500 AND 1000"),
        (">$1k",     "sales > 1000"),
    ],
}


class SQLBuilder:

    def __init__(self, table: str = "") -> None:
        raw = table or Config.DB_TABLE
        if not _SAFE_TABLE_RE.match(raw):
            raise ValueError(f"Unsafe table name: {raw!r}")
        self.table = raw

    def run(self, plan: Dict[str, Any]) -> pd.DataFrame:
        plan  = self._apply_semantic_map(plan)
        intent = plan.get("intent", "")

        if intent == "clarify":
            return pd.DataFrame()

        result = self._dispatch(plan)

        if result.empty and intent not in ("kpi_detail", "clarify",
                                           "kpi_distribution", "kpi_correlation"):
            widened = self._widen_date(plan)
            if widened:
                logger.info("Empty result — retrying with widened date")
                result = self._dispatch(widened)
                if not result.empty:
                    result.attrs["date_widened"] = True

        return result

    # Dispatch

    def _dispatch(self, plan: Dict[str, Any]) -> pd.DataFrame:
        intent = plan.get("intent", "")

        if intent == "kpi_compare":
            return self._run_compare(plan)
        if intent == "kpi_detail":
            return self._run_detail(plan)
        if intent == "kpi_distribution":
            return self._run_distribution(plan)
        if intent == "kpi_correlation":
            return self._run_correlation(plan)

        breakdown_cols = self._get_breakdown_cols(plan)
        if len(breakdown_cols) >= 2:
            return self._run_multi_breakdown(plan, breakdown_cols)

        if intent == "kpi_trend":
            return self._run_trend(plan)
        if intent == "kpi_rank":
            return self._run_rank(plan)

        return self._run_value(plan)

    def _run_distribution(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Histogram-style distribution.

        Supports:
          - Bucketing a metric (discount, sales, profit_margin) across orders
          - Counting orders + summing profit/sales per bucket
          - Optional breakdown_by for faceting (e.g. by region)

        SQL pattern (CASE WHEN buckets — works in all PostgreSQL versions):
          SELECT
            CASE
              WHEN discount = 0       THEN '0%'
              WHEN discount <= 0.10   THEN '1-10%'
              ...
            END AS bucket,
            COUNT(DISTINCT order_id) AS orders,
            SUM(profit)              AS profit,
            SUM(sales)               AS sales
          FROM superstore
          WHERE ...
          GROUP BY bucket
          ORDER BY MIN(discount) ASC
        """
        dist_metric = plan.get("dist_metric", "discount") 
        breakdown   = plan.get("breakdown_by")
        where_sql, params = self._build_where(plan)

        buckets = _DISTRIBUTION_BUCKETS.get(dist_metric, _DISTRIBUTION_BUCKETS["discount"])

        if dist_metric == "profit_margin":
            return self._run_distribution_margin(plan, where_sql, params, breakdown, buckets)

        case_parts = " ".join(
            f"WHEN {cond} THEN {repr(label)}"
            for label, cond in buckets
        )
        case_sql = f"CASE {case_parts} ELSE 'other' END"

        select = [
            f"{case_sql} AS bucket",
            "COUNT(DISTINCT order_id) AS orders",
            "SUM(sales)               AS sales",
            "ROUND(CAST(SUM(profit) AS DECIMAL), 2) AS profit",
        ]
        group_by = ["bucket"]

        if breakdown and breakdown in _ALLOWED_BREAKDOWN_COLS:
            select.insert(1, f"{breakdown} AS breakdown")
            group_by.append("breakdown")

        # Sort by bucket order (not alphabetically)
        bucket_labels = [label for label, _ in buckets]
        order_case = " ".join(
            f"WHEN {repr(label)} THEN {i}"
            for i, label in enumerate(bucket_labels)
        )
        order_sql = f"CASE bucket {order_case} ELSE 99 END ASC"
        if breakdown:
            order_sql = f"breakdown ASC, {order_sql}"

        sql = (
            f"SELECT {', '.join(select)}\n"
            f"FROM {self.table}\n"
            f"WHERE {where_sql}\n"
            f"GROUP BY {', '.join(group_by)}\n"
            f"ORDER BY {order_sql}"
        )

        if not self._validate_sql(sql, params):
            return pd.DataFrame()

        df = execute_query(sql, params)
        if not df.empty:
            df.attrs["dist_metric"]  = dist_metric
            df.attrs["bucket_order"] = bucket_labels
        return df

    def _run_distribution_margin(self, plan, where_sql, params,
                                  breakdown, buckets) -> pd.DataFrame:
        """Margin distribution requires aggregating first, then bucketing."""
        group_col = f"{breakdown}, " if breakdown and breakdown in _ALLOWED_BREAKDOWN_COLS else ""
        inner_breakdown = f"{breakdown} AS breakdown," if breakdown and breakdown in _ALLOWED_BREAKDOWN_COLS else ""

        inner = (
            f"SELECT {inner_breakdown}\n"
            f"  sub_category,\n"
            f"  SUM(sales) AS s, SUM(profit) AS p\n"
            f"FROM {self.table}\n"
            f"WHERE {where_sql}\n"
            f"GROUP BY {group_col}sub_category\n"
            f"HAVING SUM(sales) > 0"
        )

        sql = (
            f"SELECT\n"
            f"  CASE\n"
            f"    WHEN p/s*100 < 0      THEN '<0%'\n"
            f"    WHEN p/s*100 <= 5     THEN '0-5%'\n"
            f"    WHEN p/s*100 <= 15    THEN '5-15%'\n"
            f"    ELSE '>15%'\n"
            f"  END AS bucket,\n"
            f"  COUNT(*) AS orders,\n"
            f"  SUM(s) AS sales,\n"
            f"  SUM(p) AS profit\n"
            f"FROM ({inner}) sub\n"
            f"GROUP BY bucket\n"
            f"ORDER BY MIN(p/s*100)"
        )
        if not self._validate_sql(sql, params):
            return pd.DataFrame()
        df = execute_query(sql, params)
        if not df.empty:
            df.attrs["dist_metric"] = "profit_margin"
        return df

    def _run_correlation(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Dual-metric scatter data for correlation analysis.

        Returns one row per breakdown value (or per sub_category by default)
        with both metrics — suitable for scatter plots.

        SQL pattern:
          SELECT
            sub_category AS breakdown,
            AVG(discount) AS x_metric,
            SUM(profit)/SUM(sales)*100 AS y_metric,
            COUNT(DISTINCT order_id) AS orders
          FROM superstore
          GROUP BY sub_category
          ORDER BY x_metric
        """
        metrics    = plan.get("metrics", ["sales", "profit"])
        breakdown  = plan.get("breakdown_by") or "sub_category"
        x_metric   = plan.get("x_metric", "discount")  # horizontal axis
        y_metric   = plan.get("y_metric", metrics[0] if metrics else "profit")
        where_sql, params = self._build_where(plan)

        if breakdown not in _ALLOWED_BREAKDOWN_COLS:
            breakdown = "sub_category"

        # Build X metric expression
        x_expr_map = {
            "discount":      "ROUND(CAST(AVG(CASE WHEN discount <= 1 THEN discount ELSE discount/100 END)*100 AS DECIMAL), 1)",
            "sales":         "ROUND(CAST(SUM(sales) AS DECIMAL), 0)",
            "orders":        "COUNT(DISTINCT order_id)",
            "profit_margin": "ROUND(CAST(CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END AS DECIMAL), 1)",
        }
        y_expr_map = {
            "profit":        "ROUND(CAST(SUM(profit) AS DECIMAL), 0)",
            "profit_margin": "ROUND(CAST(CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END AS DECIMAL), 1)",
            "sales":         "ROUND(CAST(SUM(sales) AS DECIMAL), 0)",
            "orders":        "COUNT(DISTINCT order_id)",
        }

        x_expr = x_expr_map.get(x_metric, x_expr_map["discount"])
        y_expr = y_expr_map.get(y_metric, y_expr_map["profit"])

        sql = (
            f"SELECT\n"
            f"  {breakdown} AS breakdown,\n"
            f"  {x_expr} AS x_val,\n"
            f"  {y_expr} AS y_val,\n"
            f"  COUNT(DISTINCT order_id) AS orders,\n"
            f"  SUM(sales) AS sales,\n"
            f"  SUM(profit) AS profit\n"
            f"FROM {self.table}\n"
            f"WHERE {where_sql}\n"
            f"GROUP BY {breakdown}\n"
            f"ORDER BY x_val ASC"
        )

        if not self._validate_sql(sql, params):
            return pd.DataFrame()

        df = execute_query(sql, params)
        if not df.empty:
            df.attrs["x_metric"] = x_metric
            df.attrs["y_metric"] = y_metric
            df.attrs["breakdown"] = breakdown

        return df

    def _get_breakdown_cols(self, plan: Dict[str, Any]) -> List[str]:
        """
        Resolve the list of breakdown columns from the plan.
        """
        explicit = plan.get("breakdown_cols")
        if isinstance(explicit, list) and explicit:
            return [c for c in explicit if c in _ALLOWED_BREAKDOWN_COLS]

        b1 = plan.get("breakdown_by")
        b2 = plan.get("secondary_breakdown")
        cols = []
        if b1 and b1 in _ALLOWED_BREAKDOWN_COLS:
            cols.append(b1)
        if b2 and b2 in _ALLOWED_BREAKDOWN_COLS and b2 != b1:
            cols.append(b2)
        return cols

    def _run_multi_breakdown(self, plan: Dict[str, Any],
                              breakdown_cols: List[str]) -> pd.DataFrame:
        """
        GROUP BY N columns.

        Returns columns: breakdown_0, breakdown_1, ..., breakdown_{N-1}, <metrics>

        N=2 → compatible with existing _run_cross() formatter
               (breakdown = col[0], breakdown2 = col[1])
        N≥3 → new format: breakdown_0..breakdown_{N-1}
        """
        metrics  = plan.get("metrics", ["sales"])
        order_by = plan.get("order_by") or metrics[0]
        where_sql, params = self._build_where(plan)

        # Validate all columns
        safe_cols = [c for c in breakdown_cols if c in _ALLOWED_BREAKDOWN_COLS]
        if not safe_cols:
            return self._run_value(plan)

        select_parts: List[str] = []
        group_parts:  List[str] = []

        for i, col in enumerate(safe_cols):
            alias = "breakdown" if i == 0 else (f"breakdown2" if i == 1 else f"breakdown_{i}")
            select_parts.append(f"{col} AS {alias}")
            group_parts.append(col)

        for m in metrics:
            select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        # Always include support metrics
        for m in ("sales", "profit"):
            if m not in metrics:
                select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        sql = (
            f"SELECT {', '.join(select_parts)}\n"
            f"FROM {self.table}\n"
            f"WHERE {where_sql}\n"
            f"GROUP BY {', '.join(group_parts)}\n"
            f"ORDER BY breakdown ASC, {order_by} DESC NULLS LAST"
        )

        if not self._validate_sql(sql, params):
            return pd.DataFrame()

        df = execute_query(sql, params)
        if not df.empty:
            df.attrs["breakdown_cols"]  = safe_cols
            df.attrs["n_breakdowns"]    = len(safe_cols)
            df.attrs["primary_dim"]     = safe_cols[0]
            df.attrs["secondary_dim"]   = safe_cols[1] if len(safe_cols) > 1 else None
            # Backward compat
            df.attrs["cross_breakdown"] = True

        return df

    def _run_trend(self, plan: Dict[str, Any]) -> pd.DataFrame:
        grain = plan.get("time_grain") or "year"
        if grain not in _GRAIN_MAP:
            grain = "year"

        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        where_sql, params = self._build_where(plan)
        order_by  = plan.get("order_by") or metrics[0]

        bucket = f"DATE_TRUNC('{_GRAIN_MAP[grain]}', order_date)"
        select = [f"{bucket} AS period"]
        group  = ["period"]

        if breakdown and breakdown in _ALLOWED_BREAKDOWN_COLS:
            select.append(f"{breakdown} AS breakdown")
            group.append("breakdown")

        for m in metrics:
            select.append(f"{_METRIC_EXPR[m]} AS {m}")

        sql = (
            f"SELECT {', '.join(select)}\n"
            f"FROM {self.table}\n"
            f"WHERE {where_sql}\n"
            f"GROUP BY {', '.join(group)}\n"
            f"ORDER BY period ASC" + (", breakdown ASC" if breakdown else "")
        )
        if not self._validate_sql(sql, params):
            return pd.DataFrame()
        return execute_query(sql, params)

    def _run_rank(self, plan: Dict[str, Any]) -> pd.DataFrame:
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        top_k     = plan.get("top_k")
        order_by  = plan.get("order_by") or metrics[0]

        if not isinstance(top_k, int) or not (1 <= top_k <= 50):
            top_k = 10
        if not breakdown or breakdown not in _ALLOWED_BREAKDOWN_COLS:
            breakdown = "sub_category"

        where_sql, params = self._build_where(plan)
        params["top_k"] = top_k

        supporting = {m for m in ("sales", "profit") if m not in metrics}
        all_metrics = list(metrics) + list(supporting)

        select = [
            f"{breakdown} AS breakdown",
            *[f"{_METRIC_EXPR[m]} AS {m}" for m in all_metrics],
        ]
        sql = (
            f"SELECT {', '.join(select)}\n"
            f"FROM {self.table}\n"
            f"WHERE {where_sql}\n"
            f"GROUP BY {breakdown}\n"
            f"ORDER BY {order_by} DESC NULLS LAST\n"
            f"LIMIT %(top_k)s"
        )
        if not self._validate_sql(sql, params):
            return pd.DataFrame()
        return execute_query(sql, params)

    def _run_value(self, plan: Dict[str, Any]) -> pd.DataFrame:
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        order_by  = plan.get("order_by") or metrics[0]
        top_k     = plan.get("top_k")

        where_sql, params = self._build_where(plan)
        supporting = {m for m in ("sales", "profit") if m not in metrics}
        all_metrics = list(metrics) + list(supporting)

        select: List[str] = []
        group:  List[str] = []

        if breakdown and breakdown in _ALLOWED_BREAKDOWN_COLS:
            select.append(f"{breakdown} AS breakdown")
            group.append(breakdown)

        select += [f"{_METRIC_EXPR[m]} AS {m}" for m in all_metrics]

        sql_parts = [
            f"SELECT {', '.join(select)}",
            f"FROM {self.table}",
            f"WHERE {where_sql}",
        ]
        if group:
            sql_parts.append(f"GROUP BY {', '.join(group)}")
            sql_parts.append(f"ORDER BY {order_by} DESC NULLS LAST")

        if isinstance(top_k, int) and 1 <= top_k <= 50:
            sql_parts.append("LIMIT %(top_k)s")
            params["top_k"] = top_k

        sql = "\n".join(sql_parts)
        if not self._validate_sql(sql, params):
            return pd.DataFrame()
        return execute_query(sql, params)

    def _run_cross(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """Legacy 2-column cross — now delegates to _run_multi_breakdown."""
        cols = self._get_breakdown_cols(plan)
        if len(cols) >= 2:
            return self._run_multi_breakdown(plan, cols)
        sql, params = self.build_sql(plan)
        df = execute_query(sql, params)
        if not df.empty:
            df.attrs["cross_breakdown"] = True
        return df

    def _run_detail(self, plan: Dict[str, Any]) -> pd.DataFrame:
        breakdown = plan.get("breakdown_by") or "sub_category"
        if breakdown not in _ALLOWED_BREAKDOWN_COLS:
            breakdown = "sub_category"

        top_k = plan.get("top_k") or 15
        where_sql, params = self._build_where(plan)
        params["top_k"] = top_k

        group_sql = f"""
            SELECT
                {breakdown} AS breakdown,
                category,
                COUNT(DISTINCT order_id) AS orders,
                ROUND(CAST(SUM(sales) AS DECIMAL), 2) AS sales,
                ROUND(CAST(SUM(profit) AS DECIMAL), 2) AS profit,
                ROUND(CAST(AVG(CAST(discount AS numeric)) * 100 AS DECIMAL), 1) AS avg_discount_pct,
                CASE WHEN SUM(sales) = 0 THEN 0
                     ELSE ROUND(CAST(SUM(profit)/SUM(sales)*100 AS DECIMAL), 2)
                END AS profit_margin
            FROM {self.table}
            WHERE {where_sql}
            GROUP BY {breakdown}, category
            HAVING SUM(profit) < 0
            ORDER BY SUM(profit) ASC
            LIMIT %(top_k)s
        """
        grouped_df = execute_query(group_sql, params)
        sample_sql = f"""
            SELECT order_id, order_date,
                   {breakdown} AS breakdown, category, product_name,
                   sales, profit, discount
            FROM {self.table}
            WHERE {where_sql} AND profit < 0
            ORDER BY profit ASC LIMIT %(sample_k)s
        """
        try:
            sample_df = execute_query(sample_sql, {**params, "sample_k": 10})
        except Exception:
            sample_df = pd.DataFrame()

        if not grouped_df.empty:
            grouped_df.attrs["detail_type"]   = "grouped_summary"
            grouped_df.attrs["condition"]     = plan.get("condition", "profit_negative")
            grouped_df.attrs["sample_orders"] = sample_df
        return grouped_df

    def _run_compare(self, plan: Dict[str, Any]) -> pd.DataFrame:
        breakdown = plan.get("breakdown_by")
        metric    = plan["metrics"][0]

        def _run(p):
            sql, params = self.build_sql(p)
            return execute_query(sql, params)

        cur_df  = _run(plan)
        prev_pl = {**plan, **self._prev_dates(plan)}
        prv_df  = _run(prev_pl)

        if breakdown:
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
                if r["previous"] != 0 else None, axis=1,
            )
            merged["metric"] = metric
            merged["current_start"] = plan["start_date"]
            merged["current_end"]   = plan["end_date"]
            merged["prev_start"]    = prev_pl["start_date"]
            merged["prev_end"]      = prev_pl["end_date"]
            return merged.sort_values("change_pct", ascending=True)

        def safe_float(df):
            if df is None or df.empty: return 0.0
            val = df.iloc[0].get(metric)
            try: return float(val) if val is not None else 0.0
            except: return 0.0

        return pd.DataFrame([{
            "metric": metric,
            "current_start": plan["start_date"],
            "current_end":   plan["end_date"],
            "prev_start":    prev_pl["start_date"],
            "prev_end":      prev_pl["end_date"],
            "current":       safe_float(cur_df),
            "previous":      safe_float(prv_df),
        }])

    # Helpers

    def build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Raw SQL builder — for agent tools and _run_compare."""
        metrics             = plan["metrics"]
        time_grain          = plan.get("time_grain", "none")
        breakdown_by        = plan.get("breakdown_by")
        secondary_breakdown = plan.get("secondary_breakdown")
        top_k               = plan.get("top_k")
        order_by            = plan.get("order_by") or metrics[0]
        where_sql, params   = self._build_where(plan)

        bucket_sql = (
            f"DATE_TRUNC('{_GRAIN_MAP[time_grain]}', order_date)"
            if time_grain not in (None, "none") else None
        )

        select_parts: List[str] = []
        group_parts:  List[str] = []

        if bucket_sql:
            select_parts.append(f"{bucket_sql} AS period")
            group_parts.append("period")
        if breakdown_by and breakdown_by in _ALLOWED_BREAKDOWN_COLS:
            select_parts.append(f"{breakdown_by} AS breakdown")
            group_parts.append("breakdown")
        if secondary_breakdown and secondary_breakdown in _ALLOWED_BREAKDOWN_COLS:
            select_parts.append(f"{secondary_breakdown} AS breakdown2")
            group_parts.append("breakdown2")

        for m in metrics:
            select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        if plan.get("intent") == "kpi_value" and not bucket_sql and not breakdown_by:
            for m in ["sales", "profit"]:
                if m not in metrics:
                    select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        lines = [
            "SELECT " + ", ".join(select_parts),
            f"FROM {self.table}",
            f"WHERE {where_sql}",
        ]
        if group_parts:
            lines.append("GROUP BY " + ", ".join(group_parts))

        if plan.get("intent") == "kpi_rank":
            lines += [f"ORDER BY {order_by} DESC NULLS LAST", "LIMIT %(top_k)s"]
            params["top_k"] = top_k
        elif secondary_breakdown:
            lines.append(f"ORDER BY breakdown ASC, {order_by} DESC NULLS LAST")
        elif breakdown_by and not bucket_sql:
            lines.append(f"ORDER BY {order_by} DESC NULLS LAST")
        elif bucket_sql:
            order = "period ASC, breakdown ASC" if breakdown_by else "period ASC"
            lines.append(f"ORDER BY {order}")

        return "\n".join(lines), params

    def _build_where(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        parts  = ["order_date >= %(start)s", "order_date <= %(end)s"]
        params: Dict[str, Any] = {
            "start": plan.get("start_date", "1900-01-01"),
            "end":   plan.get("end_date",   "2100-01-01"),
        }
        f = plan.get("filters") or {}
        for col in ("region", "segment", "category", "sub_category", "state"):
            vals = f.get(col)
            if vals:
                ph = ", ".join(f"%({col}_{i})s" for i in range(len(vals)))
                parts.append(f"{col} IN ({ph})")
                for i, v in enumerate(vals):
                    params[f"{col}_{i}"] = v
        return " AND ".join(parts), params

    def _validate_sql(self, sql: str, params: Dict[str, Any]) -> bool:
        upper = sql.upper()
        if "SELECT" not in upper or "FROM" not in upper:
            logger.error("SQL validation: missing SELECT/FROM")
            return False
        if "LIMIT" in upper and "%(top_k)s" in sql and not isinstance(params.get("top_k"), int):
            logger.error("SQL validation: LIMIT with non-int top_k")
            return False
        if ";" in sql.replace("%(", "").replace(")s", ""):
            logger.error("SQL validation: semicolon injection")
            return False
        required = set(_re.findall(r'%\((\w+)\)s', sql))
        missing  = required - set(params.keys())
        if missing:
            logger.error("SQL validation: missing params %s", missing)
            return False
        return True

    def _apply_semantic_map(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        plan = dict(plan)
        if "metrics" in plan:
            plan["metrics"] = [_SEMANTIC_MAP.get(m.lower(), m) for m in plan["metrics"]]
        if "order_by" in plan:
            plan["order_by"] = _SEMANTIC_MAP.get((plan["order_by"] or "").lower(), plan["order_by"])
        if "breakdown_by" in plan:
            plan["breakdown_by"] = _SEMANTIC_MAP.get(
                (plan.get("breakdown_by") or "").lower(), plan.get("breakdown_by"))
        if "breakdown_cols" in plan and isinstance(plan["breakdown_cols"], list):
            plan["breakdown_cols"] = [
                _SEMANTIC_MAP.get(c.lower(), c) for c in plan["breakdown_cols"]
            ]
        return plan

    def _widen_date(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            sd = datetime.strptime(plan["start_date"], "%Y-%m-%d")
            ed = datetime.strptime(plan["end_date"],   "%Y-%m-%d")
        except (KeyError, ValueError):
            return None
        if (ed - sd).days >= 365:
            return None
        return {**plan,
                "start_date": (sd - timedelta(days=183)).strftime("%Y-%m-%d"),
                "end_date":   (ed + timedelta(days=183)).strftime("%Y-%m-%d")}

    @staticmethod
    def _prev_dates(plan: Dict[str, Any]) -> Dict[str, str]:
        sd = datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
        ed = datetime.strptime(plan["end_date"],   "%Y-%m-%d").date()
        cp = plan.get("compare_period", "prev_period")
        if plan.get("_override_prev_start"):
            return {"start_date": plan["_override_prev_start"],
                    "end_date":   plan["_override_prev_end"]}
        if cp == "prev_period":
            delta      = (ed - sd).days + 1
            prev_end   = sd.__class__.fromordinal(sd.toordinal() - 1)
            prev_start = sd.__class__.fromordinal(prev_end.toordinal() - (delta - 1))
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