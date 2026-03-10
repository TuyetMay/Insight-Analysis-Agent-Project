"""
core/data_loader.py
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config import Config
from core.database import execute_query


def _build_where(filters: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """Build parameterised WHERE clause. Returns (where_sql, cache_key, params)."""
    conditions: List[str] = []
    params: Dict[str, Any] = {}
    cache_parts: List[str] = []

    date_range = filters.get("date_range")
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start, end = date_range[0], date_range[1]
        conditions += ["order_date >= %(start_date)s", "order_date <= %(end_date)s"]
        params["start_date"] = start
        params["end_date"]   = end
        cache_parts.append(f"dr={start}_{end}")

    for col, key in [("region", "region"), ("segment", "segment"), ("category", "category")]:
        values = filters.get(key)
        if values:  # non-empty list = user filtered
            sorted_vals = sorted(str(v) for v in values)
            # Build explicit IN clause — most compatible with psycopg2
            placeholders = ", ".join(f"%({key}_{i})s" for i in range(len(sorted_vals)))
            conditions.append(f"{col} IN ({placeholders})")
            for i, v in enumerate(sorted_vals):
                params[f"{key}_{i}"] = v
            cache_parts.append(f"{key}={'|'.join(sorted_vals)}")

    where    = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    cache_key = ";".join(cache_parts)
    return where, cache_key, params


@st.cache_data(ttl=600)
def load_filtered_data(cache_key: str, filters_serialised: str) -> pd.DataFrame:
    """
    cache_key  — stable string used by Streamlit to detect changes
    filters_serialised — JSON string to reconstruct filters inside the cached fn
    """
    import json
    filters = json.loads(filters_serialised)

    # Restore date_range as strings (JSON loses date objects — DB accepts strings fine)
    where, _, params = _build_where(filters)
    sql = f"SELECT * FROM {Config.DB_TABLE} {where} ORDER BY order_date"
    df  = execute_query(sql, params)

    if not df.empty:
        for col in ("order_date", "ship_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


def load_filtered_data_safe(filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Public wrapper — converts filters to a stable cache key + JSON, then calls cached fn.
    Call this from app.py instead of load_filtered_data directly.
    """
    import json
    _, cache_key, _ = _build_where(filters)

    # Serialise: convert date objects → string so JSON works
    safe_filters: Dict[str, Any] = {}
    dr = filters.get("date_range")
    if isinstance(dr, (tuple, list)) and len(dr) == 2:
        safe_filters["date_range"] = [
            dr[0].strftime("%Y-%m-%d") if hasattr(dr[0], "strftime") else str(dr[0]),
            dr[1].strftime("%Y-%m-%d") if hasattr(dr[1], "strftime") else str(dr[1]),
        ]
    for key in ("region", "segment", "category"):
        safe_filters[key] = sorted(str(v) for v in (filters.get(key) or []))

    return load_filtered_data(cache_key, json.dumps(safe_filters, sort_keys=True))


@st.cache_data(ttl=3600)
def get_filter_options() -> Dict[str, Any]:
    sql = f"""
        SELECT
            ARRAY_AGG(DISTINCT region)   AS regions,
            ARRAY_AGG(DISTINCT segment)  AS segments,
            ARRAY_AGG(DISTINCT category) AS categories,
            MIN(order_date)              AS min_date,
            MAX(order_date)              AS max_date
        FROM {Config.DB_TABLE}
    """
    df = execute_query(sql)
    if df.empty:
        return {}
    row = df.iloc[0]
    return {
        "region":   sorted(row["regions"]   or []),
        "segment":  sorted(row["segments"]  or []),
        "category": sorted(row["categories"] or []),
        "min_date": row["min_date"],
        "max_date": row["max_date"],
    }


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"total_sales": 0, "total_profit": 0, "total_orders": 0, "profit_margin": 0}
    total_sales   = df["sales"].sum()
    total_profit  = df["profit"].sum()
    total_orders  = df["order_id"].nunique()
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0
    return {
        "total_sales":   total_sales,
        "total_profit":  total_profit,
        "total_orders":  total_orders,
        "profit_margin": profit_margin,
    }