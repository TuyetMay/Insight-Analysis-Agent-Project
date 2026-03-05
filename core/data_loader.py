"""
core/data_loader.py
Data-access layer: filtered queries, filter option discovery, KPI calculation.
Pure data logic — no chart or UI concerns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config import Config
from core.database import execute_query

# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _build_where(filters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Build a parameterised WHERE clause from a filters dict."""
    conditions: List[str] = []
    params: Dict[str, Any] = {}

    date_range = filters.get("date_range")
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        conditions += ["order_date >= %(start_date)s", "order_date <= %(end_date)s"]
        params["start_date"], params["end_date"] = date_range

    for col, key in [("region", "region"), ("segment", "segment"), ("category", "category")]:
        values = filters.get(key)
        if values:
            conditions.append(f"{col} IN %({key}_tuple)s")
            params[f"{key}_tuple"] = tuple(values)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where, params


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def load_filtered_data(filters: Dict[str, Any]) -> pd.DataFrame:
    """Load the filtered dataset from the database."""
    where, params = _build_where(filters)
    sql = f"SELECT * FROM {Config.DB_TABLE} {where} ORDER BY order_date"
    df = execute_query(sql, params)

    if not df.empty:
        for col in ("order_date", "ship_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(ttl=3600)
def get_filter_options() -> Dict[str, Any]:
    """Return all distinct filter values and the full date range."""
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
    """Compute summary KPIs from an already-filtered DataFrame."""
    if df.empty:
        return {"total_sales": 0, "total_profit": 0, "total_orders": 0, "profit_margin": 0}

    total_sales  = df["sales"].sum()
    total_profit = df["profit"].sum()
    total_orders = df["order_id"].nunique()
    profit_margin = (total_profit / total_sales * 100) if total_sales else 0

    return {
        "total_sales":  total_sales,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "profit_margin": profit_margin,
    }
