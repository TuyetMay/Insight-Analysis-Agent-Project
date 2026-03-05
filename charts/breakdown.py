"""
charts/breakdown.py
Dimensional breakdown charts: region, segment, region×segment heatmap.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from charts._utils import ensure_numeric


def create_profit_by_region(df: pd.DataFrame) -> go.Figure:
    df = ensure_numeric(df.copy(), ["profit"])
    data = df.groupby("region")["profit"].sum().reset_index()
    fig = px.bar(
        data, x="region", y="profit",
        title="Profit by Region",
        labels={"profit": "Profit (USD)", "region": "Region"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=400)
    return fig


def create_profit_by_segment(df: pd.DataFrame) -> go.Figure:
    df = ensure_numeric(df.copy(), ["profit"])
    data = df.groupby("segment")["profit"].sum().reset_index()
    fig = px.bar(
        data, x="segment", y="profit",
        title="Profit by Segment",
        labels={"profit": "Profit (USD)", "segment": "Segment"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=400)
    return fig


def create_profit_heatmap(df: pd.DataFrame) -> go.Figure:
    df = ensure_numeric(df.copy(), ["profit"])
    pivot = df.pivot_table(
        values="profit", index="segment", columns="region", aggfunc="sum"
    )
    fig = px.imshow(
        pivot,
        labels=dict(x="Region", y="Segment", color="Profit"),
        title="Profit Heatmap — Region × Segment",
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(height=400)
    return fig
