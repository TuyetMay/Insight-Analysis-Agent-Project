from __future__ import annotations

import logging
import re as _re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from core.database import execute_query

logger = logging.getLogger(__name__)

# ── Safe table name guard ─────────────────────────────────────
_SAFE_TABLE_RE = _re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# ── Metric SQL expressions ────────────────────────────────────
_METRIC_EXPR: Dict[str, str] = {
    "sales":         "SUM(sales)",
    "profit":        "SUM(profit)",
    "orders":        "COUNT(DISTINCT order_id)",
    "profit_margin": "CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END",
}

# ── Time grain → DATE_TRUNC argument ─────────────────────────
_GRAIN_MAP: Dict[str, str] = {
    "week": "week", "month": "month",
    "quarter": "quarter", "year": "year",
}

# ── Allowed breakdown columns (SQL injection guard) ───────────
_ALLOWED_BREAKDOWN_COLS: frozenset = frozenset({
    "region", "segment", "category", "sub_category", "state", "ship_mode",
})

# ── Semantic synonym map
_SEMANTIC_MAP: Dict[str, str] = {
    # metric synonyms
    "revenue":       "sales",
    "income":        "sales",
    "earnings":      "sales",
    "turnover":      "sales",
    "gross":         "sales",
    "net profit":    "profit",
    "gain":          "profit",
    "margin":        "profit_margin",
    "profitability": "profit_margin",
    "transactions":  "orders",
    "purchases":     "orders",
    # dimension synonyms
    "area":          "region",
    "zone":          "region",
    "territory":     "region",
    "market":        "region",
    "geo":           "region",
    "customer type": "segment",
    "customer segment": "segment",
    "product type":  "category",
    "product group": "category",
    "item":          "sub_category",
    "sku":           "sub_category",
}


class SQLBuilder:
    """Build and execute parameterised SQL for validated query plans."""

    def __init__(self, table: str = "") -> None:
        raw = table or Config.DB_TABLE
        if not _SAFE_TABLE_RE.match(raw):
            raise ValueError(f"Unsafe table name: {raw!r}")
        self.table = raw

    # ─────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────

    def run(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Execute a validated plan and return a DataFrame.
        """
        # 1. Semantic normalisation before anything else
        plan = self._apply_semantic_map(plan)

        intent = plan.get("intent", "")

        # 2. Short-circuit: clarify never hits DB
        if intent == "clarify":
            return pd.DataFrame()

        # 3. Intent dispatch
        result = self._dispatch(plan)

        # 4. Self-correction: empty result on a narrow window → retry once
        if result.empty and intent not in ("kpi_detail", "clarify"):
            widened = self._widen_date(plan)
            if widened:
                logger.info(
                    "Empty result for %s — retrying with widened date %s→%s",
                    intent, widened["start_date"], widened["end_date"],
                )
                result = self._dispatch(widened)
                if not result.empty:
                    result.attrs["date_widened"] = True

        return result

    # Intent dispatch router

    def _dispatch(self, plan: Dict[str, Any]) -> pd.DataFrame:
        intent = plan.get("intent", "")

        if intent == "kpi_compare":
            return self._run_compare(plan)
        if intent == "kpi_detail":
            return self._run_detail(plan)
        if plan.get("secondary_breakdown"):
            return self._run_cross(plan)
        if intent == "kpi_trend":
            return self._run_trend(plan)
        if intent == "kpi_rank":
            return self._run_rank(plan)
        # kpi_value (and any unknown intent as safe fallback)
        return self._run_value(plan)

    # kpi_trend handler

    def _run_trend(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Time-series aggregation.
        """
        grain      = plan.get("time_grain") or "year"
        if grain not in _GRAIN_MAP:
            logger.warning("Unknown grain %r — defaulting to year", grain)
            grain = "year"

        metrics    = plan.get("metrics", ["sales"])
        breakdown  = plan.get("breakdown_by")
        f          = plan.get("filters", {})
        order_by   = plan.get("order_by") or metrics[0]

        where_sql, params = self._build_where(plan)

        bucket = f"DATE_TRUNC('{_GRAIN_MAP[grain]}', order_date)"
        select = [f"{bucket} AS period"]
        group  = ["period"]

        if breakdown:
            if breakdown not in _ALLOWED_BREAKDOWN_COLS:
                logger.warning("Blocked breakdown %r in trend — skipping", breakdown)
                breakdown = None
            else:
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
        """
        Top-K ranking.
        """
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        top_k     = plan.get("top_k")
        order_by  = plan.get("order_by") or metrics[0]

        # Guard: top_k must be a valid int
        if not isinstance(top_k, int) or not (1 <= top_k <= 50):
            logger.warning("kpi_rank called with top_k=%r — defaulting to 10", top_k)
            top_k = 10

        # Guard: breakdown required for rank
        if not breakdown or breakdown not in _ALLOWED_BREAKDOWN_COLS:
            logger.warning("kpi_rank missing valid breakdown — using sub_category")
            breakdown = "sub_category"

        where_sql, params = self._build_where(plan)
        params["top_k"] = top_k

        # Always include sales + profit as supporting context
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

    # kpi_value handler

    def _run_value(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """
        Aggregate value — with or without breakdown.
        """
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        order_by  = plan.get("order_by") or metrics[0]
        top_k     = plan.get("top_k")
        show_ext  = plan.get("show_extremes", False)

        where_sql, params = self._build_where(plan)

        supporting = {m for m in ("sales", "profit") if m not in metrics}
        all_metrics = list(metrics) + list(supporting)

        select: List[str] = []
        group:  List[str] = []

        if breakdown:
            if breakdown not in _ALLOWED_BREAKDOWN_COLS:
                logger.warning("Blocked breakdown %r in value — skipping", breakdown)
                breakdown = None
            else:
                select.append(f"{breakdown} AS breakdown")
                group.append("breakdown")

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

    # build_sql — kept for backward-compat 

    def build_sql(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Translate a validated plan into (sql_string, params_dict).
        Kept for callers that need the raw SQL string (agent tools).
        Internal callers should prefer run() → _dispatch().
        """
        metrics             = plan["metrics"]
        time_grain          = plan["time_grain"]
        breakdown_by        = plan.get("breakdown_by")
        secondary_breakdown = plan.get("secondary_breakdown")
        top_k               = plan.get("top_k")
        order_by            = plan.get("order_by") or metrics[0]

        where_sql, params = self._build_where(plan)

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

        if secondary_breakdown:
            select_parts.append(f"{secondary_breakdown} AS breakdown2")
            group_parts.append("breakdown2")

        for m in metrics:
            select_parts.append(f"{_METRIC_EXPR[m]} AS {m}")

        if plan["intent"] == "kpi_value" and not bucket_sql and not breakdown_by and not secondary_breakdown:
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

        if plan["intent"] == "kpi_rank":
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

    # Unchanged internal handlers (_run_compare, _run_detail, _run_cross)

    def _run_cross(self, plan: Dict[str, Any]) -> pd.DataFrame:
        sql, params = self.build_sql(plan)
        df = execute_query(sql, params)
        if not df.empty:
            df.attrs["cross_breakdown"] = True
            df.attrs["primary_dim"]     = plan.get("breakdown_by", "")
            df.attrs["secondary_dim"]   = plan.get("secondary_breakdown", "")
        return df

    def _run_detail(self, plan: Dict[str, Any]) -> pd.DataFrame:
        condition = plan.get("condition", "profit_negative")
        breakdown = plan.get("breakdown_by") or "sub_category"

        if breakdown not in _ALLOWED_BREAKDOWN_COLS:
            logger.warning("Blocked unsafe breakdown %r", breakdown)
            breakdown = "sub_category"

        f     = plan.get("filters", {})
        top_k = plan.get("top_k") or 15

        where_sql, params = self._build_where(plan)
        params["top_k"] = top_k

        group_sql = f"""
            SELECT
                {breakdown}        AS breakdown,
                category,
                COUNT(DISTINCT order_id)                          AS orders,
                ROUND(CAST(SUM(sales)   AS DECIMAL), 2)          AS sales,
                ROUND(CAST(SUM(profit)  AS DECIMAL), 2)          AS profit,
                ROUND(CAST(AVG(CAST(discount AS numeric)) * 100 AS DECIMAL), 1) AS avg_discount_pct,
                CASE WHEN SUM(sales) = 0 THEN 0
                     ELSE ROUND(CAST(SUM(profit)/SUM(sales)*100 AS DECIMAL), 2)
                END                                               AS profit_margin
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
            ORDER BY profit ASC
            LIMIT %(sample_k)s
        """
        sample_params = {**params, "sample_k": 10}
        try:
            sample_df = execute_query(sample_sql, sample_params)
        except Exception:
            sample_df = pd.DataFrame()

        if not grouped_df.empty:
            grouped_df.attrs["detail_type"]   = "grouped_summary"
            grouped_df.attrs["condition"]     = condition
            grouped_df.attrs["sample_orders"] = sample_df

        return grouped_df

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


    def _build_where(self, plan: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Extract WHERE clause and params from a plan.
        Centralised so every handler uses identical filter logic.
        """
        parts: List[str] = [
            "order_date >= %(start)s",
            "order_date <= %(end)s",
        ]
        params: Dict[str, Any] = {
            "start": plan.get("start_date", "1900-01-01"),
            "end":   plan.get("end_date",   "2100-01-01"),
        }

        f = plan.get("filters") or {}
        for col in ("region", "segment", "category", "sub_category", "state"):
            vals = f.get(col)
            if vals:
                placeholders = ", ".join(f"%({col}_{i})s" for i in range(len(vals)))
                parts.append(f"{col} IN ({placeholders})")
                for i, v in enumerate(vals):
                    params[f"{col}_{i}"] = v

        return " AND ".join(parts), params

    def _validate_sql(self, sql: str, params: Dict[str, Any]) -> bool:
        """
        Lightweight pre-flight check before executing SQL.
        Catches the most common plan → SQL bugs without hitting DB.

        ChatBI §3.3 — validation layer.
        """
        # Must have SELECT and FROM
        upper = sql.upper()
        if "SELECT" not in upper or "FROM" not in upper:
            logger.error("SQL validation failed — missing SELECT/FROM: %s", sql[:120])
            return False

        # LIMIT must be accompanied by a concrete int, not NULL
        if "LIMIT" in upper:
            if "%(top_k)s" in sql and not isinstance(params.get("top_k"), int):
                logger.error("SQL validation failed — LIMIT with non-int top_k: %r", params.get("top_k"))
                return False

        # No stray semicolons (injection guard)
        if ";" in sql.replace("%(", "").replace(")s", ""):
            logger.error("SQL validation failed — semicolon in query")
            return False

        # All %(key)s placeholders must have a matching param
        required = set(_re.findall(r'%\((\w+)\)s', sql))
        missing  = required - set(params.keys())
        if missing:
            logger.error("SQL validation failed — missing params: %s", missing)
            return False

        return True

    def _apply_semantic_map(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalise metrics/breakdown using the semantic synonym map.
        """
        plan = dict(plan)  

        # Normalise metrics list
        if "metrics" in plan:
            plan["metrics"] = [
                _SEMANTIC_MAP.get(m.lower(), m) for m in plan["metrics"]
            ]

        # Normalise order_by
        if "order_by" in plan:
            plan["order_by"] = _SEMANTIC_MAP.get(
                (plan["order_by"] or "").lower(), plan["order_by"]
            )

        # Normalise breakdown_by
        if "breakdown_by" in plan:
            plan["breakdown_by"] = _SEMANTIC_MAP.get(
                (plan.get("breakdown_by") or "").lower(), plan.get("breakdown_by")
            )

        return plan

    def _widen_date(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            sd = datetime.strptime(plan["start_date"], "%Y-%m-%d")
            ed = datetime.strptime(plan["end_date"],   "%Y-%m-%d")
        except (KeyError, ValueError):
            return None

        span_days = (ed - sd).days
        if span_days >= 365:
            return None 

        new_sd = sd - timedelta(days=183)
        new_ed = ed + timedelta(days=183)

        return {
            **plan,
            "start_date": new_sd.strftime("%Y-%m-%d"),
            "end_date":   new_ed.strftime("%Y-%m-%d"),
        }

    @staticmethod
    def _prev_dates(plan: Dict[str, Any]) -> Dict[str, str]:
        """Compute previous period dates for kpi_compare."""
        sd = datetime.strptime(plan["start_date"], "%Y-%m-%d").date()
        ed = datetime.strptime(plan["end_date"],   "%Y-%m-%d").date()
        cp = plan.get("compare_period", "prev_period")

        if plan.get("_override_prev_start"):
            return {
                "start_date": plan["_override_prev_start"],
                "end_date":   plan["_override_prev_end"],
            }

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

        return {
            "start_date": prev_start.strftime("%Y-%m-%d"),
            "end_date":   prev_end.strftime("%Y-%m-%d"),
        }