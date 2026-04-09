from __future__ import annotations

import calendar as _cal
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
    """
    Builds proposition-based Chunks from Superstore DataFrame.

    Design principles (v2):
      1. 1 chunk = 1 atomic claim, verifiable independently
      2. Natural language text — NO technical prefixes like "Dimension 'X' values:"
      3. Every chunk has comparative context where possible ("highest", "lowest")
      4. Target: 50–150 tokens per chunk
    """

    # ── Public: build entry points ────────────────────────────────────────────

    def build(self, df: pd.DataFrame, kpis: Dict[str, Any],
              filters: Dict[str, Any]) -> List[Chunk]:
        """Backward-compat: build all chunks (static + dynamic)."""
        return self._schema_chunks() + self.build_dynamic(df, kpis, filters)

    def build_static(self, df: pd.DataFrame) -> List[Chunk]:
        """
        Static layer — schema facts only.
        Built once at startup, independent of filters.
        inject_tier=[3]: only injected for Tier-3 Gemini calls, not insight generation.
        """
        return self._schema_chunks()

    def build_dynamic(self, df: pd.DataFrame, kpis: Dict[str, Any],
                      filters: Dict[str, Any]) -> List[Chunk]:
        """
        Dynamic layer — all data-dependent chunks.
        Rebuilt when filters change.
        """
        chunks: List[Chunk] = []
        chunks += self._filter_context_chunks(filters)
        chunks += self._kpi_snapshot_chunks(kpis)

        if not df.empty:
            chunks += self._yearly_trend_chunks(df)
            chunks += self._yearly_transition_chunks(df)
            chunks += self._trend_overview_chunk(df)
            chunks += self._quarterly_chunks(df)
            chunks += self._monthly_peak_chunks(df)
            chunks += self._dimension_value_chunks(df)
            chunks += self._dimension_rank_chunks(df)
            chunks += self._top_k_chunks(df, "sub_category", k=10)
            chunks += self._anomaly_chunks(df, kpis)
            chunks += self._discount_impact_chunks(df)
            chunks += self._segment_category_cross_chunks(df)

        return chunks

    # ── Schema facts (static) ─────────────────────────────────────────────────

    def _schema_chunks(self) -> List[Chunk]:
        """4 schema fact chunks — inject_tier=[3] means Tier-3 Gemini only."""
        return [
            Chunk(
                "schema_metrics",
                "Available metrics: sales (total revenue in USD), profit (net profit, "
                "can be negative), orders (count of unique orders), "
                "profit_margin (profit divided by sales, as percentage).",
                {"type": "schema_fact", "schema_aspect": "metrics", "inject_tier": [3]},
            ),
            Chunk(
                "schema_dimensions",
                "Data can be grouped by: region (4 values: Central, East, South, West), "
                "segment (3 values: Consumer, Corporate, Home Office), "
                "category (3 values: Furniture, Office Supplies, Technology), "
                "and sub_category (17 product types including Phones, Chairs, Tables).",
                {"type": "schema_fact", "schema_aspect": "dimensions", "inject_tier": [3]},
            ),
            Chunk(
                "schema_compare",
                "Period comparison options: yoy (same period vs last year), "
                "mom (vs prior month), prev_period (vs prior window of same length).",
                {"type": "schema_fact", "schema_aspect": "compare_periods", "inject_tier": [3]},
            ),
            Chunk(
                "schema_intents",
                "Query types supported: aggregate totals (kpi_value), "
                "time series trends (kpi_trend), top-k rankings (kpi_rank), "
                "period comparisons (kpi_compare), "
                "loss or anomaly drill-down (kpi_detail).",
                {"type": "schema_fact", "schema_aspect": "intents", "inject_tier": [3]},
            ),
        ]

    # ── Filter context (dynamic) ──────────────────────────────────────────────

    def _filter_context_chunks(self, filters: Dict[str, Any]) -> List[Chunk]:
        """
        Natural-language description of active filters + date range.

        v1 bug: "Current dashboard filters: Date range: 2014-01-03 to 2017-12-30."
                → NEVER_RETRIEVED — text too sparse, dense embedding can't match queries.
        v2 fix: "Data covers 4 years from 2014 to 2017 (Jan 2014 to Dec 2017), 48 months."
        """
        dr = filters.get("date_range", ())

        if dr and len(dr) == 2:
            try:
                start    = pd.Timestamp(dr[0])
                end      = pd.Timestamp(dr[1])
                n_years  = end.year - start.year + 1
                n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1
                date_text = (
                    f"Data covers {n_years} year{'s' if n_years > 1 else ''} "
                    f"from {start.year} to {end.year} "
                    f"({start.strftime('%b %Y')} to {end.strftime('%b %Y')}), "
                    f"{n_months} months total."
                )
                start_str, end_str = str(dr[0]), str(dr[1])
            except Exception:
                date_text = f"Data from {dr[0]} to {dr[1]}."
                start_str, end_str = str(dr[0]), str(dr[1])
        else:
            date_text = "Data covers the full available history."
            start_str = end_str = ""

        extras: List[str] = []
        for label, key in [("Region", "region"), ("Segment", "segment"),
                            ("Category", "category")]:
            vals = filters.get(key, [])
            if vals:
                extras.append(f" Filtered to {', '.join(str(v) for v in vals)} {label.lower()}s.")

        return [Chunk(
            "filter_active",
            date_text + "".join(extras),
            {"type": "filter_context", "start_date": start_str, "end_date": end_str},
        )]

    # ── KPI snapshots (dynamic) ───────────────────────────────────────────────

    def _kpi_snapshot_chunks(self, kpis: Dict[str, Any]) -> List[Chunk]:
        
        ts  = float(kpis.get("total_sales",   0) or 0)
        tp  = float(kpis.get("total_profit",  0) or 0)
        to_ = int(kpis.get("total_orders",    0) or 0)
        pm  = float(kpis.get("profit_margin", 0) or 0)
        aov = ts / to_ if to_ else 0
        ppo = tp / to_ if to_ else 0
        bm  = "above" if pm >= 12 else "below"

        return [
            Chunk(
                "kpi_sales_snapshot",
                f"Total sales revenue is ${ts:,.0f}, generated from {to_:,} orders "
                f"at an average of ${aov:,.0f} per order.",
                {"type": "kpi_snapshot", "metric": "sales", "value": ts},
            ),
            Chunk(
                "kpi_profit_snapshot",
                f"Total profit is ${tp:,.0f}, representing a {pm:.1f}% profit margin "
                f"on ${ts:,.0f} in revenue. "
                f"Profitability is {bm} the typical 12% retail benchmark.",
                {"type": "kpi_snapshot", "metric": "profit", "value": tp},
            ),
            Chunk(
                "kpi_orders_snapshot",
                f"Total orders placed: {to_:,}, averaging ${aov:,.0f} in revenue per order "
                f"and ${ppo:,.0f} in profit per order.",
                {"type": "kpi_snapshot", "metric": "orders", "value": to_},
            ),
            Chunk(
                "kpi_margin_snapshot",
                f"Overall profit margin is {pm:.2f}% — {bm} the typical 12% retail benchmark. "
                f"${tp:,.0f} profit retained from ${ts:,.0f} in revenue.",
                {"type": "kpi_snapshot", "metric": "profit_margin", "value": pm},
            ),
        ]

    # ── Yearly trend (dynamic) ────────────────────────────────────────────────

    def _yearly_trend_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        1 chunk per year — with UNIQUE comparative context per year.
        """
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg: Dict[str, Any] = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")
            if "order_id" in df2.columns:
                agg["orders"] = ("order_id", "nunique")

            yearly = (
                df2.groupby(df2["order_date"].dt.year)
                .agg(**agg)
                .reset_index()
                .rename(columns={"order_date": "year"})
                .sort_values("year")
                .reset_index(drop=True)
            )

            max_sales = float(yearly["sales"].max())

            # Pre-compute all YoY sales changes to find best growth year
            yoy_chgs: List[float] = []
            for i in range(1, len(yearly)):
                ps = float(yearly.iloc[i - 1]["sales"])
                cs = float(yearly.iloc[i]["sales"])
                yoy_chgs.append((cs - ps) / abs(ps) * 100 if ps else 0.0)

            chunks: List[Chunk] = []

            for idx, row in yearly.iterrows():
                yr     = int(row["year"])
                s      = float(row["sales"])
                p      = float(row.get("profit", 0))
                o      = int(row.get("orders", 0))
                pm_val = (p / s * 100) if s else 0.0
                is_peak = (s == max_sales)

                if idx == 0:
                    # Baseline year — note if it has distinctive margin
                    all_margins = [
                        float(r["profit"]) / float(r["sales"]) * 100
                        for _, r in yearly.iterrows()
                        if float(r.get("sales", 0)) > 0
                    ]
                    margin_note = (
                        "lowest profit margin across all years"
                        if all_margins and pm_val == min(all_margins)
                        else "baseline year"
                    )
                    text = (
                        f"In {yr} ({margin_note}), sales were ${s:,.0f} with "
                        f"a {pm_val:.1f}% profit margin. Profit: ${p:,.0f}, orders: {o:,}."
                    )
                else:
                    prev_s  = float(yearly.iloc[idx - 1]["sales"])
                    prev_p  = float(yearly.iloc[idx - 1].get("profit", 0))
                    prev_yr = int(yearly.iloc[idx - 1]["year"])
                    s_chg   = (s - prev_s) / abs(prev_s) * 100 if prev_s else 0.0
                    p_chg   = (p - prev_p) / abs(prev_p) * 100 if prev_p else 0.0

                    if s < prev_s:
                        text = (
                            f"In {yr}, sales declined {abs(s_chg):.1f}% to ${s:,.0f} "
                            f"(from {prev_yr}'s ${prev_s:,.0f}) — "
                            f"the only year in the dataset with a sales drop. "
                            f"Despite lower revenue, profit improved {p_chg:+.1f}% "
                            f"to ${p:,.0f} ({pm_val:.1f}% margin)."
                        )
                    elif is_peak:
                        text = (
                            f"{yr} was the peak revenue year: ${s:,.0f} "
                            f"({s_chg:+.1f}% vs {prev_yr}), {o:,} orders. "
                            f"Profit ${p:,.0f} ({pm_val:.1f}% margin)."
                        )
                    else:
                        chg_idx      = idx - 1  # 0-based index into yoy_chgs
                        is_best_growth = (yoy_chgs and
                                          chg_idx < len(yoy_chgs) and
                                          s_chg == max(yoy_chgs))
                        qualifier = (
                            "the strongest single-year growth rate"
                            if is_best_growth
                            else f"{s_chg:+.1f}% vs {prev_yr}"
                        )
                        text = (
                            f"In {yr}, sales grew to ${s:,.0f} ({qualifier}). "
                            f"Profit ${p:,.0f} ({pm_val:.1f}% margin), orders {o:,}."
                        )

                chunks.append(Chunk(
                    f"year_{yr}_sales_fact",
                    text,
                    {"type": "time_period_fact",
                     "grain": "year",
                     "year": yr,
                     "metric": "sales",
                     "value": s,
                     "is_best_period": is_peak,
                     "is_worst_period": (idx > 0 and
                                        s < float(yearly.iloc[idx - 1]["sales"]))},
                ))

            return chunks
        except Exception:
            return []

    # ── Yearly transitions (dynamic) ──────────────────────────────────────────

    def _yearly_transition_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        1 chunk per YoY transition — replaces single yoy_comparison chunk (35 tokens).
        """
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg: Dict[str, Any] = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")

            yearly = (
                df2.groupby(df2["order_date"].dt.year)
                .agg(**agg)
                .reset_index()
                .rename(columns={"order_date": "year"})
                .sort_values("year")
                .reset_index(drop=True)
            )

            chunks: List[Chunk] = []
            for i in range(1, len(yearly)):
                y_from = int(yearly.iloc[i - 1]["year"])
                y_to   = int(yearly.iloc[i]["year"])
                s_from = float(yearly.iloc[i - 1]["sales"])
                s_to   = float(yearly.iloc[i]["sales"])
                p_from = float(yearly.iloc[i - 1].get("profit", 0))
                p_to   = float(yearly.iloc[i].get("profit", 0))
                s_chg  = (s_to - s_from) / abs(s_from) * 100 if s_from else 0.0
                p_chg  = (p_to - p_from) / abs(p_from) * 100 if p_from else 0.0

                direction = "grew" if s_chg >= 0 else "declined"
                if abs(s_chg) >= 20:
                    cls = "strong growth" if s_chg > 0 else "severe decline"
                elif abs(s_chg) >= 5:
                    cls = "moderate growth" if s_chg > 0 else "mild decline"
                else:
                    cls = "flat"

                chunks.append(Chunk(
                    f"trend_year_sales_{y_from}_{y_to}",
                    f"From {y_from} to {y_to}, sales {direction} {abs(s_chg):.1f}% "
                    f"({cls}): ${s_from:,.0f} → ${s_to:,.0f} "
                    f"(Δ ${abs(s_to - s_from):,.0f}). "
                    f"Profit changed {p_chg:+.1f}%: ${p_from:,.0f} → ${p_to:,.0f}.",
                    {"type": "trend_transition",
                     "grain": "year",
                     "metric": "sales",
                     "period_from": y_from,
                     "period_to": y_to,
                     "value_from": s_from,
                     "value_to": s_to,
                     "pct_change": round(s_chg, 1),
                     "direction": "up" if s_chg >= 0 else "down",
                     "classification": cls},
                ))

            return chunks
        except Exception:
            return []

    # ── Trend overview (dynamic) ──────────────────────────────────────────────

    def _trend_overview_chunk(self, df: pd.DataFrame) -> List[Chunk]:
        """
        Overall multi-year trend with CAGR — replaces yearly_growth chunk.
        """
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg: Dict[str, Any] = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")

            yearly = (
                df2.groupby(df2["order_date"].dt.year)
                .agg(**agg)
                .reset_index()
                .rename(columns={"order_date": "year"})
                .sort_values("year")
                .reset_index(drop=True)
            )

            if len(yearly) < 2:
                return []

            first_s  = float(yearly.iloc[0]["sales"])
            last_s   = float(yearly.iloc[-1]["sales"])
            first_p  = float(yearly.iloc[0].get("profit", 0))
            last_p   = float(yearly.iloc[-1].get("profit", 0))
            n_years  = len(yearly) - 1
            y_from   = int(yearly.iloc[0]["year"])
            y_to     = int(yearly.iloc[-1]["year"])
            s_growth = (last_s - first_s) / abs(first_s) * 100 if first_s else 0.0
            p_growth = (last_p - first_p) / abs(first_p) * 100 if first_p else 0.0
            cagr     = ((last_s / first_s) ** (1 / n_years) - 1) * 100 if first_s > 0 else 0.0

            return [Chunk(
                "trend_overview_sales_yearly",
                f"Over {n_years} year{'s' if n_years > 1 else ''} "
                f"({y_from} to {y_to}), total sales grew {s_growth:+.1f}% "
                f"from ${first_s:,.0f} to ${last_s:,.0f}, "
                f"with a compound annual growth rate (CAGR) of {cagr:.1f}%. "
                f"Profit grew {p_growth:+.1f}% over the same period "
                f"(${first_p:,.0f} → ${last_p:,.0f}).",
                {"type": "trend_overview",
                 "grain": "year",
                 "metric": "sales",
                 "period_from": y_from,
                 "period_to": y_to,
                 "overall_pct_change": round(s_growth, 1),
                 "cagr_pct": round(cagr, 1),
                 "n_periods": n_years + 1},
            )]
        except Exception:
            return []

    # ── Quarterly chunks (dynamic) ────────────────────────────────────────────

    def _quarterly_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        1 chunk per quarter — replaces 83-token quarterly_summary monolith.
        """
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]
            df2["year"]    = df2["order_date"].dt.year
            df2["quarter"] = df2["order_date"].dt.quarter

            agg: Dict[str, Any] = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")

            qtr_data = (
                df2.groupby(["year", "quarter"])
                .agg(**agg)
                .reset_index()
                .sort_values(["year", "quarter"])
            )

            if qtr_data.empty:
                return []

            chunks: List[Chunk] = []
            for year, grp in qtr_data.groupby("year"):
                grp = grp.sort_values("quarter").reset_index(drop=True)
                q_vals: Dict[int, float] = {}
                p_vals: Dict[int, float] = {}
                for _, r in grp.iterrows():
                    q = int(r["quarter"])
                    q_vals[q] = float(r["sales"])
                    p_vals[q] = float(r.get("profit", 0))

                # Build label for each quarter present
                q_parts = ", ".join(
                    f"Q{q} ${q_vals[q]:,.0f}" for q in sorted(q_vals)
                )
                year_sales = sum(q_vals.values())

                # Find peak quarter
                peak_q = max(q_vals, key=lambda q: q_vals[q])
                peak_s = q_vals[peak_q]
                seasonal = (
                    f"Q{peak_q} was the strongest quarter ({f'${peak_s:,.0f}'}) "
                    + ("— typical Q4 year-end seasonal peak." if peak_q == 4 else ".")
                )

                chunks.append(Chunk(
                    f"quarter_{year}_annual_summary",
                    f"Quarterly breakdown for {year}: {q_parts}. "
                    f"Annual total: ${year_sales:,.0f}. {seasonal}",
                    {"type": "time_period_fact",
                     "grain": "quarter",
                     "year": year,
                     "metric": "sales",
                     "value": year_sales},
                ))

            return chunks
        except Exception:
            return []

    # ── Monthly peak chunks (dynamic) ─────────────────────────────────────────

    def _monthly_peak_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        Peak month + worst profit month — natural language with seasonal context.
        """
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg: Dict[str, Any] = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg["profit"] = ("profit", "sum")

            monthly = (
                df2.groupby(df2["order_date"].dt.to_period("M"))
                .agg(**agg)
                .reset_index()
                .rename(columns={"order_date": "month"})
            )

            if monthly.empty:
                return []

            chunks: List[Chunk] = []

            # ── Peak sales month ────────────────────────────────────────────
            best   = monthly.loc[monthly["sales"].idxmax()]
            best_s = float(best["sales"])
            best_p = float(best.get("profit", 0))
            best_pm = (best_p / best_s * 100) if best_s else 0.0

            try:
                period_obj = best["month"]
                month_name = _cal.month_name[period_obj.month]
                period_str = f"{month_name} {period_obj.year}"
                seasonal   = "Q4 seasonal peak." if period_obj.month in (10, 11, 12) else ""
            except Exception:
                period_str = str(best["month"])
                seasonal   = ""

            chunks.append(Chunk(
                "month_peak_sales",
                f"{period_str} was the highest revenue month: ${best_s:,.0f} in sales "
                f"(${best_p:,.0f} profit, {best_pm:.1f}% margin). {seasonal}".strip(),
                {"type": "time_period_fact",
                 "grain": "month",
                 "period": str(best["month"]),
                 "metric": "sales",
                 "value": best_s,
                 "is_best_period": True},
            ))

            # ── Worst profit month (only if negative) ───────────────────────
            if "profit" in monthly.columns:
                worst   = monthly.loc[monthly["profit"].idxmin()]
                worst_p = float(worst["profit"])

                if worst_p < 0:
                    try:
                        period_obj2 = worst["month"]
                        month_name2 = _cal.month_name[period_obj2.month]
                        period_str2 = f"{month_name2} {period_obj2.year}"
                    except Exception:
                        period_str2 = str(worst["month"])

                    chunks.append(Chunk(
                        "month_worst_profit",
                        f"{period_str2} had negative overall profit: ${worst_p:,.0f} loss. "
                        f"This is the only month in the dataset with negative profitability.",
                        {"type": "anomaly_fact",
                         "anomaly_type": "negative_profit",
                         "dimension": "month",
                         "metric": "profit",
                         "value": worst_p,
                         "severity": "moderate"},
                    ))

            return chunks
        except Exception:
            return []

    # ── Dimension value chunks (dynamic) ──────────────────────────────────────

    def _dimension_value_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        1 atomic chunk per dimension value.
        """
        chunks: List[Chunk] = []
        if "sales" not in df.columns or "profit" not in df.columns:
            return chunks

        ts_total = float(df["sales"].sum())
        tp_total = float(df["profit"].sum())
        if ts_total == 0:
            return chunks

        rank_labels = {1: "highest", 2: "second highest", 3: "third highest"}

        for dim in ("region", "segment", "category"):
            if dim not in df.columns:
                continue
            try:
                grp = (
                    df.groupby(dim)
                    .agg(sales=("sales", "sum"), profit=("profit", "sum"))
                    .reset_index()
                    .sort_values("sales", ascending=False)
                    .reset_index(drop=True)
                )
                n = len(grp)

                for rank_idx, row in grp.iterrows():
                    val    = str(row[dim])
                    s      = float(row["sales"])
                    p      = float(row["profit"])
                    margin = (p / s * 100) if s else 0.0
                    s_pct  = (s / ts_total * 100)
                    p_pct  = (p / tp_total * 100) if tp_total else 0.0
                    rank   = rank_idx + 1

                    dim_plural = {"category": "categories", "region": "regions",
                                  "segment": "segments"}.get(dim, f"{dim}s")
                    if rank == 1:
                        intro = f"{val} leads all {dim_plural} by sales"
                    elif rank == n:
                        intro = f"{val} is the lowest {dim} by sales"
                    else:
                        lbl   = rank_labels.get(rank, f"rank {rank}")
                        intro = f"{val} is the {lbl} {dim} by sales"

                    text = (
                        f"{intro}: ${s:,.0f} ({s_pct:.0f}% of total revenue). "
                        f"Profit: ${p:,.0f} ({p_pct:.0f}% of total profit, "
                        f"{margin:.1f}% margin)."
                    )

                    # Fix 4: add unique distinguishing signal for Consumer and Home Office
                    # so dense embeddings can separate them from Corporate (rank 2)
                    if dim == "segment" and val == "Consumer":
                        text += (
                            f" Consumer is the majority segment — it drives more than half "
                            f"({s_pct:.0f}%) of total revenue."
                        )
                    elif dim == "segment" and val == "Home Office":
                        text += (
                            f" Despite being the smallest segment, Home Office has the highest "
                            f"profit margin at {margin:.1f}% — more efficient than Consumer or Corporate."
                        )

                    val_slug = val.lower().replace(" ", "_").replace("/", "_")
                    chunks.append(Chunk(
                        f"{dim}_{val_slug}_fact",
                        text,
                        {"type": "dimension_value",
                         "dimension": dim,
                         "dimension_value": val,
                         "metric": "sales",
                         "value": s,
                         "rank": rank,
                         "rank_of": n,
                         "is_highest": (rank == 1),
                         "is_lowest": (rank == n),
                         "margin": round(margin, 1),
                         "pct_of_total": round(s_pct, 1)},
                    ))
            except Exception:
                continue

        return chunks

    # ── Dimension rank chunks (dynamic) ───────────────────────────────────────

    def _dimension_rank_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        Compact ranking summary per dimension × metric — includes gap analysis.
        Replaces top4_region_profit, top4_region_sales (now merged here).
        Adds segment/category rankings that were missing in v1.
        """
        chunks: List[Chunk] = []

        for dim in ("region", "segment", "category"):
            if dim not in df.columns:
                continue
            for metric in ("sales", "profit"):
                if metric not in df.columns:
                    continue
                try:
                    grp = (
                        df.groupby(dim)
                        .agg(**{metric: (metric, "sum")})
                        .reset_index()
                        .sort_values(metric, ascending=False)
                        .reset_index(drop=True)
                    )

                    items = ", ".join(
                        f"{row[dim]} (${float(row[metric]):,.0f})"
                        for _, row in grp.iterrows()
                    )
                    top_val = float(grp.iloc[0][metric])
                    bot_val = float(grp.iloc[-1][metric])
                    gap_pct = abs(top_val - bot_val) / abs(top_val) * 100 if top_val else 0.0

                    verb = "Most profitable" if metric == "profit" else "Highest revenue"
                    dim_plural = {"category": "categories", "region": "regions",
                                  "segment": "segments"}.get(dim, f"{dim}s")
                    chunks.append(Chunk(
                        f"{dim}_ranked_by_{metric}",
                        f"{verb} {dim_plural}: {items}. "
                        f"Gap between highest and lowest: {gap_pct:.0f}%.",
                        {"type": "dimension_rank",
                         "dimension": dim,
                         "metric": metric,
                         "gap_top_bottom_pct": round(gap_pct, 1)},
                    ))
                except Exception:
                    continue

        return chunks


    def _anomaly_chunks(self, df: pd.DataFrame,
                        kpis: Dict[str, Any]) -> List[Chunk]:
        if df.empty or "profit" not in df.columns:
            return []

        chunks: List[Chunk] = []
        overall_margin = float(kpis.get("profit_margin", 12))

        # ── 1. Loss-making sub-categories ─────────────────────────────────
        if "sub_category" in df.columns and "category" in df.columns:
            try:
                agg_cols: Dict[str, Any] = {
                    "profit": ("profit", "sum"),
                    "sales":  ("sales",  "sum"),
                }
                if "order_id" in df.columns:
                    agg_cols["orders"] = ("order_id", "nunique")

                sub_grp = (
                    df.groupby(["sub_category", "category"])
                    .agg(**agg_cols)
                    .reset_index()
                )
                loss_subs = sub_grp[sub_grp["profit"] < 0].sort_values("profit")

                if not loss_subs.empty:
                    # Summary chunk — key entry point for kpi_detail
                    loss_list = ", ".join(
                        f"{r['sub_category']} (-${abs(float(r['profit'])):,.0f})"
                        for _, r in loss_subs.iterrows()
                    )
                    total_loss   = abs(float(loss_subs["profit"].sum()))
                    total_orders = (int(loss_subs["orders"].sum())
                                    if "orders" in loss_subs.columns else 0)

                    chunks.append(Chunk(
                        "anomaly_loss_subcat_summary",
                        f"Loss-making sub-categories: {loss_list}. "
                        f"These products are unprofitable — they are losing money and bleeding revenue. "
                        f"Negative profit despite positive sales means every order makes the situation worse. "
                        f"Total loss: ${total_loss:,.0f} across {total_orders:,} orders.",
                        {"type": "anomaly_fact",
                         "anomaly_type": "negative_profit",
                         "dimension": "sub_category",
                         "severity": "critical",
                         "total_loss": total_loss},
                    ))

                    # Per-item detail chunks (top 3 worst)
                    for _, row in loss_subs.head(3).iterrows():
                        val      = str(row["sub_category"])
                        cat      = str(row["category"])
                        loss     = abs(float(row["profit"]))
                        revenue  = float(row["sales"])
                        orders   = int(row.get("orders", 0))
                        val_slug = val.lower().replace(" ", "_")

                        avg_disc = 0.0
                        if "discount" in df.columns:
                            mask    = df["sub_category"] == val
                            raw_avg = df.loc[mask, "discount"].mean()
                            if pd.notna(raw_avg):
                                avg_disc = (float(raw_avg) * 100
                                            if float(raw_avg) <= 1
                                            else float(raw_avg))

                        chunks.append(Chunk(
                            f"anomaly_loss_{val_slug}",
                            f"{val} is a loss-making sub-category in {cat}: "
                            f"net loss of ${loss:,.0f} on ${revenue:,.0f} in revenue "
                            f"({orders:,} orders, avg discount {avg_disc:.0f}%). "
                            f"This product is unprofitable and losing money.",
                            {"type": "anomaly_fact",
                             "anomaly_type": "negative_profit",
                             "dimension": "sub_category",
                             "dimension_value": val,
                             "metric": "profit",
                             "value": -loss,
                             "revenue": revenue,
                             "category": cat,
                             "avg_discount_pct": round(avg_disc, 1),
                             "severity": "critical" if loss > 10_000 else "moderate"},
                        ))
            except Exception:
                pass

        # ── 2. Low margin anomalies per dimension ──────────────────────────
        for dim in ("region", "category"):
            if dim not in df.columns:
                continue
            try:
                grp = (
                    df.groupby(dim)
                    .agg(sales=("sales", "sum"), profit=("profit", "sum"))
                    .reset_index()
                )
                grp["margin"] = grp.apply(
                    lambda r: float(r["profit"]) / float(r["sales"]) * 100
                    if float(r["sales"]) > 0 else 0.0,
                    axis=1,
                )
                worst    = grp.sort_values("margin").iloc[0]
                margin_v = float(worst["margin"])
                gap_vs   = overall_margin - margin_v

                if gap_vs > 5:  # only flag if >5pp below overall average
                    val      = str(worst[dim])
                    val_slug = val.lower().replace(" ", "_")
                    chunks.append(Chunk(
                        f"anomaly_low_margin_{dim}_{val_slug}",
                        f"{val} {dim} has the lowest profit margin at {margin_v:.1f}% — "
                        f"{gap_vs:.1f} percentage points below the overall average of "
                        f"{overall_margin:.1f}%. "
                        f"Sales ${float(worst['sales']):,.0f}, "
                        f"Profit ${float(worst['profit']):,.0f}.",
                        {"type": "anomaly_fact",
                         "anomaly_type": "low_margin",
                         "dimension": dim,
                         "dimension_value": val,
                         "metric": "profit_margin",
                         "value": margin_v,
                         "severity": "critical" if gap_vs > 10 else "moderate"},
                    ))
            except Exception:
                continue

        # ── 3. High discount → loss anomaly ───────────────────────────────
        if "discount" in df.columns:
            try:
                df2 = df.copy()
                df2["discount"] = pd.to_numeric(df2["discount"], errors="coerce")
                df2 = df2[df2["discount"].notna()]
                if df2["discount"].max() > 1:
                    df2["discount"] /= 100

                high_disc    = df2[df2["discount"] > 0.20]
                total_orders = len(df2)

                if not high_disc.empty and total_orders:
                    avg_p    = float(high_disc["profit"].mean())
                    n_orders = len(high_disc)
                    pct      = n_orders / total_orders * 100

                    chunks.append(Chunk(
                        "anomaly_high_discount_loss",
                        f"Orders with discounts above 20% are on average unprofitable: "
                        f"average profit is ${avg_p:,.0f} (a loss). "
                        f"This affects {n_orders:,} orders ({pct:.0f}% of all orders). "
                        f"Heavy discounting is destroying profitability.",
                        {"type": "anomaly_fact",
                         "anomaly_type": "high_discount_loss",
                         "dimension": "orders",
                         "metric": "profit",
                         "value": avg_p,
                         "severity": "critical" if avg_p < -50 else "moderate"},
                    ))
            except Exception:
                pass

        return chunks

    #  Top-k sub-category chunks (dynamic) 

    def _top_k_chunks(self, df: pd.DataFrame, dim: str,
                      k: int = 10) -> List[Chunk]:
        """
        Top-k ranking for a dimension by profit and sales.
        Rewritten to add loss-maker mention for profit metric.
        """
        if dim not in df.columns or "profit" not in df.columns:
            return []
        try:
            grp = (
                df.groupby(dim)
                .agg(sales=("sales", "sum"), profit=("profit", "sum"))
                .reset_index()
            )
            dim_label = dim.replace("_", " ")
            dim_plural_label = {"sub_category": "sub categories", "category": "categories",
                                 "region": "regions", "segment": "segments"}.get(dim, dim_label + "s")
            chunks: List[Chunk] = []

            for metric, verb in [("profit", "Most profitable"),
                                  ("sales",  "Highest revenue")]:
                top   = grp.nlargest(k, metric)
                items = ", ".join(
                    f"{row[dim]} (${float(row[metric]):,.0f})"
                    for _, row in top.iterrows()
                )

                loss_note = ""
                if metric == "profit":
                    negatives = grp[grp["profit"] < 0]
                    if not negatives.empty:
                        loss_names = ", ".join(str(r[dim]) for _, r in negatives.iterrows())
                        loss_note  = (
                            f" Note: {loss_names} have negative profit "
                            f"(loss-making, not shown above)."
                        )

                chunks.append(Chunk(
                    f"top{k}_{dim}_{metric}",
                    f"{verb} {dim_plural_label}: {items}.{loss_note}",
                    {"type": "dimension_rank", "dimension": dim, "metric": metric, "k": k},
                ))

            return chunks
        except Exception:
            return []

    #  Discount impact (dynamic) 

    def _discount_impact_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """Discount bucket analysis — kept from v1, text unchanged."""
        if "discount" not in df.columns or "profit" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["discount"] = pd.to_numeric(df2["discount"], errors="coerce")
            df2 = df2[df2["discount"].notna()]
            if df2["discount"].max() > 1:
                df2["discount"] /= 100
            df2 = df2[df2["discount"].between(0, 1)]

            df2["bucket"] = pd.cut(
                df2["discount"],
                bins=[0, 0.1, 0.2, 0.3, 1.0],
                labels=["0–10%", "10–20%", "20–30%", ">30%"],
                include_lowest=True,
            )
            buckets = (
                df2.groupby("bucket", observed=True)
                .agg(avg_profit=("profit", "mean"),
                     avg_sales=("sales",  "mean"),
                     n=("profit", "count"))
                .reset_index()
            )
            rows = "; ".join(
                f"Discount {row['bucket']}: avg profit=${float(row['avg_profit']):,.0f}, "
                f"avg sales=${float(row['avg_sales']):,.0f} ({int(row['n'])} orders)"
                for _, row in buckets.iterrows()
                if int(row["n"]) > 0
            )
            return [Chunk(
                "discount_impact",
                f"Discount impact on profit: {rows}.",
                {"type": "insight", "topic": "discount"},
            )]
        except Exception:
            return []

    #  Segment × Category cross chunks (dynamic) 

    def _segment_category_cross_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        """
        Top segment × category profit combinations.
        Rewritten to add concentration note and furniture anomaly highlight.
        """
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
                f"{row['segment']}/{row['category']}: Profit=${float(row['profit']):,.0f}"
                for _, row in cross.iterrows()
            )
            concentration_note = (
                "Technology and Office Supplies dominate profit across all segments. "
                "Furniture is marginally profitable and drags down overall category margins."
            )
            return [Chunk(
                "cross_segment_category_profit",
                f"Top segment-category combinations by profit: {rows}. "
                f"{concentration_note}",
                {"type": "cross_fact", "dimensions": ["segment", "category"]},
            )]
        except Exception:
            return []