"""
charts/products.py
Product-level insight charts: discount impact, sub-category ranking, category distribution.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts._utils import ensure_numeric, normalize_discount


def create_discount_impact(df: pd.DataFrame) -> go.Figure:
    df = ensure_numeric(df.copy(), ["discount", "sales", "profit"])
    df["discount"] = normalize_discount(df["discount"])
    df = df[df["discount"].notna() & df["discount"].between(0, 1)]

    df["discount_bucket"] = pd.cut(
        df["discount"],
        bins=[0, 0.1, 0.2, 0.3, 1.0],
        labels=["0-10%", "10-20%", "20-30%", ">30%"],
        include_lowest=True,
    )
    data = (
        df.groupby("discount_bucket", observed=True)
        .agg({"sales": "mean", "profit": "mean"})
        .reset_index()
    )

    fig = go.Figure()
    for col, color, name in [("sales", "#1f77b4", "Avg Sales"), ("profit", "#17becf", "Avg Profit")]:
        fig.add_trace(go.Scatter(
            x=data["discount_bucket"], y=data[col],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Avg Sales & Profit by Discount Level",
        xaxis_title="Discount Bucket", yaxis_title="USD",
        height=400,
    )
    return fig


def create_top_subcategories(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    df = ensure_numeric(df.copy(), ["profit"])
    data = df.groupby("sub_category")["profit"].sum().nlargest(top_n).reset_index()
    fig = px.bar(
        data, x="profit", y="sub_category", orientation="h",
        title=f"Top {top_n} Sub-Categories by Profit",
        labels={"profit": "Profit (USD)", "sub_category": "Sub-Category"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=400)
    return fig


def create_category_distribution(df: pd.DataFrame) -> go.Figure:
    df = ensure_numeric(df.copy(), ["profit"])
    data = df.groupby("category")["profit"].sum().reset_index()
    fig = px.pie(
        data, values="profit", names="category",
        title="Profit Distribution by Category",
        hole=0.3,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400)
    return fig
