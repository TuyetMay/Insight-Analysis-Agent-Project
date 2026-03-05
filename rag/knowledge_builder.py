"""
rag/knowledge_builder.py
Converts tabular Superstore data into searchable text chunks for RAG retrieval.
Each chunk describes one verified numerical fact derived from the actual data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Chunk:
    """A single unit of retrievable knowledge."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class KnowledgeBaseBuilder:
    """Aggregate a DataFrame into plain-text Chunks the retriever can index."""

    def build(self, df: pd.DataFrame, kpis: Dict[str, Any], filters: Dict[str, Any]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunks += self._schema_chunks()
        chunks += self._filter_context_chunks(filters)
        chunks += self._kpi_chunks(kpis)
        if not df.empty:
            chunks += self._time_range_chunks(df)
            chunks += self._yearly_trend_chunks(df)
            chunks += self._monthly_chunks(df)
            chunks += self._quarterly_chunks(df)
            for dim in ("region", "segment", "category"):
                chunks += self._dimension_chunks(df, dim)
            chunks += self._top_k_chunks(df, "sub_category", k=10)
            chunks += self._top_k_chunks(df, "category", k=5)
            chunks += self._top_k_chunks(df, "region", k=4)
            chunks += self._discount_impact_chunks(df)
            chunks += self._segment_category_cross_chunks(df)
        return chunks

    # ── Schema facts ──────────────────────────────────────────

    def _schema_chunks(self) -> List[Chunk]:
        return [
            Chunk("schema_metrics",
                  "Queryable metrics: sales (total revenue USD), profit (can be negative), "
                  "orders (COUNT DISTINCT order_id), profit_margin (profit/sales × 100, %).",
                  {"type": "schema"}),
            Chunk("schema_dimensions",
                  "Supported breakdowns: region, segment, category, sub_category. "
                  "Time grains: week, month, quarter, year.",
                  {"type": "schema"}),
            Chunk("schema_compare",
                  "Period comparisons: yoy (vs same period last year), "
                  "mom (vs prior month), prev_period (vs prior window of equal length).",
                  {"type": "schema"}),
            Chunk("schema_intents",
                  "Supported query intents: kpi_value (aggregates), kpi_trend (time series), "
                  "kpi_rank (top-k by dimension), kpi_compare (period-over-period).",
                  {"type": "schema"}),
        ]

    # ── Active filter context ─────────────────────────────────

    def _filter_context_chunks(self, filters: Dict[str, Any]) -> List[Chunk]:
        dr = filters.get("date_range", ())
        date_str = f"{dr[0]} to {dr[1]}" if (dr and len(dr) == 2) else "full history"
        parts = [f"Date range: {date_str}"]
        for label, key in [("Region", "region"), ("Segment", "segment"), ("Category", "category")]:
            vals = filters.get(key, [])
            if vals:
                parts.append(f"{label}: {', '.join(str(v) for v in vals)}")
        return [Chunk("filter_context", "Current dashboard filters: " + "; ".join(parts) + ".",
                      {"type": "filter"})]

    # ── KPI summary ───────────────────────────────────────────

    def _kpi_chunks(self, kpis: Dict[str, Any]) -> List[Chunk]:
        ts  = float(kpis.get("total_sales", 0) or 0)
        tp  = float(kpis.get("total_profit", 0) or 0)
        to_ = int(kpis.get("total_orders", 0) or 0)
        pm  = float(kpis.get("profit_margin", 0) or 0)
        return [
            Chunk("kpi_summary",
                  f"KPI summary (current filters): Total Sales=${ts:,.0f}, "
                  f"Total Profit=${tp:,.0f}, Total Orders={to_:,}, Profit Margin={pm:.2f}%.",
                  {"type": "kpi"}),
            Chunk("kpi_profitability",
                  f"Net profit ${tp:,.0f} on revenue ${ts:,.0f} → margin {pm:.2f}%. "
                  f"{'Profitable.' if tp > 0 else 'Operating at a loss.'}",
                  {"type": "kpi", "topic": "profitability"}),
        ]

    # ── Time range ────────────────────────────────────────────

    def _time_range_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns:
            return []
        try:
            dates = pd.to_datetime(df["order_date"], errors="coerce").dropna()
            if dates.empty:
                return []
            years = sorted(dates.dt.year.unique().tolist())
            return [Chunk("time_range",
                          f"Data spans {dates.min().date()} to {dates.max().date()}. "
                          f"{len(years)} years: {years}. {dates.dt.to_period('M').nunique()} months total.",
                          {"type": "time", "years": years})]
        except Exception:
            return []

    # ── Yearly trend ──────────────────────────────────────────

    def _yearly_trend_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]
            agg = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")
            if "order_id" in df2.columns:
                agg["orders"] = ("order_id", "nunique")
            yearly = df2.groupby(df2["order_date"].dt.year).agg(**agg).reset_index()
            yearly.rename(columns={"order_date": "year"}, inplace=True)

            chunks = []
            for _, row in yearly.iterrows():
                yr = int(row["year"])
                s = float(row["sales"])
                p = float(row.get("profit", 0))
                o = int(row.get("orders", 0))
                pm = (p / s * 100) if s else 0
                chunks.append(Chunk(f"yearly_{yr}",
                    f"Year {yr}: Sales=${s:,.0f}, Profit=${p:,.0f}, Orders={o:,}, Margin={pm:.1f}%.",
                    {"type": "trend", "grain": "year", "year": yr}))

            if len(yearly) >= 2:
                first_s, last_s = float(yearly.iloc[0]["sales"]), float(yearly.iloc[-1]["sales"])
                growth = ((last_s - first_s) / first_s * 100) if first_s else 0
                chunks.append(Chunk("yearly_growth",
                    f"Revenue growth {int(yearly.iloc[0]['year'])}→{int(yearly.iloc[-1]['year'])}: "
                    f"{growth:+.1f}% (${first_s:,.0f} → ${last_s:,.0f}).",
                    {"type": "trend", "topic": "growth"}))
            return chunks
        except Exception:
            return []

    # ── Monthly / quarterly summaries ────────────────────────

    def _monthly_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]
            agg = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")
            monthly = df2.groupby(df2["order_date"].dt.to_period("M")).agg(**agg).reset_index()
            if monthly.empty:
                return []

            chunks = []
            best = monthly.loc[monthly["sales"].idxmax()]
            chunks.append(Chunk("monthly_best_sales",
                f"Peak month by sales: {best['order_date']} — ${float(best['sales']):,.0f}"
                + (f", profit ${float(best['profit']):,.0f}." if "profit" in best else "."),
                {"type": "time", "grain": "month"}))

            if "profit" in monthly.columns:
                worst = monthly.loc[monthly["profit"].idxmin()]
                if float(worst["profit"]) < 0:
                    chunks.append(Chunk("monthly_worst_profit",
                        f"Worst month by profit: {worst['order_date']} — ${float(worst['profit']):,.0f}.",
                        {"type": "time", "grain": "month"}))
            return chunks
        except Exception:
            return []

    def _quarterly_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]
            agg = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")
            quarterly = df2.groupby(df2["order_date"].dt.to_period("Q")).agg(**agg).reset_index().tail(8)
            if quarterly.empty:
                return []
            rows = "; ".join(
                f"{r['order_date']}: Sales=${float(r['sales']):,.0f}"
                + (f", Profit=${float(r['profit']):,.0f}" if "profit" in r else "")
                for _, r in quarterly.iterrows()
            )
            return [Chunk("quarterly_summary",
                f"Last 8 quarters: {rows}.", {"type": "trend", "grain": "quarter"})]
        except Exception:
            return []

    # ── Dimension aggregations ────────────────────────────────

    def _dimension_chunks(self, df: pd.DataFrame, dim: str) -> List[Chunk]:
        if dim not in df.columns or "sales" not in df.columns:
            return []
        try:
            agg = {"sales": ("sales", "sum")}
            if "profit" in df.columns:
                agg["profit"] = ("profit", "sum")
            if "order_id" in df.columns:
                agg["orders"] = ("order_id", "nunique")
            grp = df.groupby(dim).agg(**agg).reset_index().sort_values("sales", ascending=False)
            values = sorted(df[dim].dropna().unique().tolist())
            detail = "; ".join(
                f"{r[dim]}: Sales=${float(r['sales']):,.0f}"
                + (f", Profit=${float(r['profit']):,.0f}" if "profit" in r else "")
                for _, r in grp.iterrows()
            )
            return [Chunk(f"dim_{dim}",
                f"Dimension '{dim}' values: {values}. Detail: {detail}.",
                {"type": "dimension", "dimension": dim, "values": values})]
        except Exception:
            return []

    def _top_k_chunks(self, df: pd.DataFrame, dim: str, k: int = 10) -> List[Chunk]:
        if dim not in df.columns or "profit" not in df.columns:
            return []
        try:
            grp = df.groupby(dim).agg(sales=("sales", "sum"), profit=("profit", "sum")).reset_index()
            chunks = []
            for metric, label in [("profit", "profit"), ("sales", "sales")]:
                top = grp.nlargest(k, metric)
                items = ", ".join(
                    f"{r[dim]} (${float(r[metric]):,.0f})" for _, r in top.iterrows()
                )
                chunks.append(Chunk(f"top{k}_{dim}_{metric}",
                    f"Top {k} {dim.replace('_', ' ')} by {label}: {items}.",
                    {"type": "rank", "dimension": dim, "metric": metric, "k": k}))
            return chunks
        except Exception:
            return []

    # ── Discount impact ───────────────────────────────────────

    def _discount_impact_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "discount" not in df.columns or "profit" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["discount"] = pd.to_numeric(df2["discount"], errors="coerce")
            df2 = df2[df2["discount"].notna()]
            if df2["discount"].max() > 1:
                df2["discount"] /= 100
            df2["bucket"] = pd.cut(
                df2["discount"],
                bins=[0, 0.1, 0.2, 0.3, 1.0],
                labels=["0-10%", "10-20%", "20-30%", ">30%"],
                include_lowest=True,
            )
            buckets = df2.groupby("bucket", observed=True).agg(
                avg_profit=("profit", "mean"), avg_sales=("sales", "mean"), n=("profit", "count")
            ).reset_index()
            rows = "; ".join(
                f"Discount {r['bucket']}: avg profit=${float(r['avg_profit']):,.0f}, "
                f"avg sales=${float(r['avg_sales']):,.0f} ({int(r['n'])} orders)"
                for _, r in buckets.iterrows() if int(r["n"]) > 0
            )
            return [Chunk("discount_impact", f"Discount impact on profit: {rows}.",
                          {"type": "insight", "topic": "discount"})]
        except Exception:
            return []

    # ── Cross-dimension ───────────────────────────────────────

    def _segment_category_cross_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if not {"segment", "category", "profit"}.issubset(df.columns):
            return []
        try:
            cross = (
                df.groupby(["segment", "category"])
                .agg(profit=("profit", "sum"), sales=("sales", "sum"))
                .reset_index()
                .sort_values("profit", ascending=False)
                .head(9)
            )
            rows = "; ".join(
                f"{r['segment']}/{r['category']}: Profit=${float(r['profit']):,.0f}"
                for _, r in cross.iterrows()
            )
            return [Chunk("segment_category_cross",
                f"Top 9 Segment×Category combos by profit: {rows}.",
                {"type": "cross", "dimensions": ["segment", "category"]})]
        except Exception:
            return []
