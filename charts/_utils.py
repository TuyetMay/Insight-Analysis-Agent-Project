"""
charts/_utils.py
Shared helpers used across all chart modules.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            s = (
                df[col].astype(str)
                .str.strip()
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(s, errors="coerce")
    return df


def normalize_discount(series: pd.Series) -> pd.Series:
    """Convert discount to 0-1 range if it was stored as 0-100."""
    if pd.notna(series.max()) and series.max() > 1:
        return series / 100.0
    return series
