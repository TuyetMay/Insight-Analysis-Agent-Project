"""
charts/trends.py
Time-series chart generators: sales/profit trend, orders per month.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts._utils import ensure_datetime, ensure_numeric


def create_sales_profit_trend(df: pd.DataFrame) -> go.Figure:
    """Monthly sales & profit line chart."""
    df = ensure_datetime(ensure_numeric(df.copy(), ["sales", "profit"]), "order_date")
    df = df[df["order_date"].notna()]

    monthly = (
        df.groupby(df["order_date"].dt.to_period("M"))
        .agg({"sales": "sum", "profit": "sum"})
        .reset_index()
    )
    monthly["order_date"] = monthly["order_date"].dt.to_timestamp()

    fig = go.Figure()
    for col, color, name in [
        ("sales",  "#1f77b4", "Sales"),
        ("profit", "#17becf", "Profit"),
    ]:
        fig.add_trace(go.Scatter(
            x=monthly["order_date"], y=monthly[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=4),
        ))

    fig.update_layout(
        title="Sales & Profit Trend — Monthly",
        xaxis_title="Month", yaxis_title="USD",
        hovermode="x unified", height=400,
    )
    return fig


def create_orders_by_month(df: pd.DataFrame) -> go.Figure:
    """Total orders per month bar chart."""
    df = ensure_datetime(df.copy(), "order_date")
    df = df[df["order_date"].notna()]

    monthly = (
        df.groupby(df["order_date"].dt.to_period("M"))
        .agg({"order_id": "nunique"})
        .reset_index()
    )
    monthly["order_date"] = monthly["order_date"].dt.to_timestamp()

    fig = px.bar(
        monthly, x="order_date", y="order_id",
        title="Orders per Month",
        labels={"order_id": "Orders", "order_date": "Month"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=400)
    return fig
