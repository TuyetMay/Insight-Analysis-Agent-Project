"""
chatbot/insight_generator.py  — v3
Improvements vs v2:
  - _data_summary(): richer context per intent
    * kpi_detail: loss_ratio, discount vs profitable benchmark
    * kpi_rank: margin per entity, HHI concentration
    * kpi_trend: acceleration, CAGR significance
    * kpi_compare: volume vs price decomposition context
    * kpi_value: margin per breakdown, revenue concentration
  - _llm_insight(): per-intent prompt templates
    * Each intent has specific analysis tasks
    * Forbidden phrases list to avoid generic output
    * Output format enforced per intent
"""

from __future__ import annotations
import logging
import re as _re
from typing import Any, Dict, List, Optional

import pandas as pd
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class InsightGenerator:

    def __init__(self, gemini_client: Any = None, model_name: str = "") -> None:
        self.client     = gemini_client
        self.model_name = model_name

    # ── Public ────────────────────────────────────────────────

    def generate(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return ""
        if self.client and self.model_name:
            text = self._llm_insight(plan, df)
            if text:
                return f"\n\n---\n💡 **Insight:**\n\n{text}"
        lines = self._rule_insight(plan, df)
        if lines:
            return "\n\n---\n💡 **Insight:**\n\n" + "\n\n".join(lines)
        return ""

    # ── LLM insight ───────────────────────────────────────────

    def _llm_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        if plan.get("intent") == "kpi_rank" and len(df) == 1:
            return ""

        summary   = self._data_summary(plan, df)
        intent    = plan.get("intent", "kpi_value")
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        grain     = plan.get("time_grain", "none")
        cp        = plan.get("compare_period")
        m0        = plan.get("order_by") or metrics[0]

        # ── Per-intent prompt ─────────────────────────────────
        if intent == "kpi_detail":
            task = f"""You are analysing LOSS-MAKING sub-categories for an executive.

REQUIRED — cover ALL 5 points, in this order:
1. SCALE: Total financial damage in $ and as % of total revenue ({self._fv(float(df['sales'].sum()) if 'sales' in df.columns else 0, 'sales')} revenue). Is this material?
2. PRIMARY DRIVER: Which sub-category causes the most damage? What % of total loss is it responsible for?
3. ROOT CAUSE: For each loss-maker, is discount the cause (avg_discount > 20% = YES) or structural (discount low but still losing)?
4. PATTERN: Are losses concentrated (1 sub-category = 80%+ of loss) or spread?
5. ACTION: One specific action per loss-maker — either "cap discount at X%" or "review cost structure for Y".

FORBIDDEN phrases: "it is worth noting", "overall", "in conclusion", "the data shows", "it appears".
Every sentence must have a specific $ or % number.
Format: 5 numbered paragraphs, each 2-3 sentences."""

        elif intent == "kpi_rank":
            dim = breakdown or "item"
            task = f"""You are analysing a {dim} performance ranking for an executive.

REQUIRED — cover ALL 4 points:
1. LEADER: Who leads, exact value, % share of total. Is this concentration healthy or a risk?
2. GAP ANALYSIS: Gap between #1 and #last in $ and %. Is it a winner-take-all market or balanced?
3. UNDERPERFORMERS: Any entity with negative profit or margin below overall average? Name them with exact numbers.
4. STRATEGIC IMPLICATION: Should the business double-down on leader, fix underperformers, or rebalance? Give specific reasoning with numbers.

FORBIDDEN: "it is worth noting", "overall performance", "further investigation needed".
Every sentence must have at least one $ or % number from the DATA.
Format: 4 bullet points, each 2-3 sentences. Lead each bullet with **bold label**."""

        elif intent == "kpi_trend":
            if breakdown:
                task = f"""You are analysing a {grain}-level trend broken down by {breakdown}.

REQUIRED — cover ALL 5 points:
1. BEST PERFORMER: Which {breakdown} grew most (% and absolute $)? What drove it?
2. WORST PERFORMER: Which {breakdown} grew least or declined? Is it structural or cyclical?
3. DIVERGENCE: Did all {breakdown}s move together or did they diverge? Quote the spread.
4. MOMENTUM CHECK: Is growth accelerating or decelerating in the most recent period? Use last 2 transitions.
5. DRIVER: Which {breakdown} contributed most $ to the total change (not just % growth)?

Every sentence must have a specific number. Max 8 bullet points."""
            else:
                task = f"""You are analysing a {grain}-level sales trend for an executive.

REQUIRED — cover ALL 4 points IN ORDER:
1. OVERALL: Total % change and absolute $ from first to last period. State CAGR if multi-year.
2. PERIOD-BY-PERIOD: For EACH transition, state: % change, classify (🚀≥20% / 📈5-19% / ➡️flat / 📉decline), and one specific business interpretation (not "growth was strong" — say WHY it could have happened based on the data pattern).
3. MOMENTUM: Compare last 2 periods to prior 2. Is momentum accelerating or decelerating? Quote exact averages.
4. FORWARD VIEW: Based on CAGR of {self._compute_cagr(df, m0):.1f}%, what is the base-case next period? State the number.

FORBIDDEN: "the data shows", "it is worth noting", "monitor", "investigate further".
Every sentence must have a $ or % number. Max 10 sentences total."""

        elif intent == "kpi_compare":
            task = f"""You are analysing a period-over-period comparison for an executive.

REQUIRED — cover ALL 4 points:
1. MAGNITUDE: State the change in % AND absolute $. Is {abs(self._get_compare_chg(df)):.1f}% change large, moderate, or small for this business context (compare to CAGR or typical variance)?
2. DECOMPOSITION: Is this change likely driven by VOLUME (more orders) or VALUE (higher per-order amount)? Use any available data to reason.
3. SUSTAINABILITY: Is this a one-time spike or consistent trend? Cross-reference with trend data if available.
4. RISK/OPPORTUNITY: What specific action should the business take in the NEXT period based on this change?

FORBIDDEN: "monitor whether momentum continues", "it is worth noting", "further analysis needed".
Every sentence must have a number. Max 6 sentences."""

        elif intent == "kpi_value":
            if breakdown:
                task = f"""You are analysing {metrics[0]} broken down by {breakdown} for an executive.

REQUIRED — cover ALL 4 points:
1. LEADER: Who leads, exact value, % share of {self._fv(float(df[m0].sum()) if m0 in df.columns else 0, m0)} total. State margin if available.
2. CONCENTRATION RISK: Do top 2 entities control >60% of total? If yes, flag as concentration risk with exact %.
3. UNDERPERFORMER: Who is lowest, exact value, % share. Is the gap between #1 and #last normal or alarming?
4. MARGIN DIVERGENCE: Which entity has highest/lowest margin? Does high sales = high profit, or is there a disconnect?

FORBIDDEN: "competitive parity", "it is worth noting", "further investigation".
Every sentence must have a $ or % number. Max 5 bullet points."""
            else:
                task = "Provide a 3-sentence summary of the key metric with margin context and one action."

        else:
            task = "Provide a thorough analytical observation using the data. Include specific numbers."

        # ── Shared writing rules ──────────────────────────────
        max_tokens = {
            "kpi_trend":   2000,
            "kpi_compare": 1000,
            "kpi_rank":    1200,
            "kpi_detail":  1500,
            "kpi_value":   1200,
        }.get(intent, 1000)

        prompt = f"""You are a senior business analyst writing for a C-suite executive audience.
Using ONLY the numbers in the DATA section, write the insight requested.

=== DATA ===
{summary}

=== TASK ===
{task}

=== ABSOLUTE RULES ===
- Use ONLY numbers from the DATA section above. Do NOT invent figures.
- Bold ALL numbers, percentages, entity names using **markdown**.
- Never use: "The data shows", "It is worth noting", "Overall", "In conclusion", "It appears", "Monitor", "Investigate further".
- Every sentence must contain at least one specific number.
- Be direct and specific. No hedging language.

Write the complete insight now:"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.05,
                    max_output_tokens=max_tokens,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()

            if len(text) > 30:
                score = self._verify_grounding(text, summary)
                if score < 0.4:
                    logger.warning(
                        "Grounding score %.2f < 0.4 — falling back to rule-based", score
                    )
                    return ""
                return self._trim_to_complete_sentence(text)
        except Exception as exc:
            logger.warning("LLM insight failed: %s", exc)

        return ""

    # ── Helpers for prompt context ─────────────────────────────

    def _compute_cagr(self, df: pd.DataFrame, metric: str) -> float:
        """Compute CAGR from trend df if available."""
        try:
            if "period" not in df.columns or metric not in df.columns:
                return 0.0
            sdf = df.sort_values("period").reset_index(drop=True)
            if len(sdf) < 2:
                return 0.0
            first = float(sdf.iloc[0][metric])
            last  = float(sdf.iloc[-1][metric])
            n     = len(sdf) - 1
            if first <= 0 or n == 0:
                return 0.0
            return ((last / first) ** (1 / n) - 1) * 100
        except Exception:
            return 0.0

    def _get_compare_chg(self, df: pd.DataFrame) -> float:
        """Get % change from compare df."""
        try:
            if "current" not in df.columns or "previous" not in df.columns:
                return 0.0
            row  = df.iloc[0]
            cur  = float(row["current"])
            prev = float(row["previous"])
            return (cur - prev) / abs(prev) * 100 if prev else 0.0
        except Exception:
            return 0.0

    # ── Data summary — v3: richer context per intent ──────────

    def _data_summary(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        intent    = plan.get("intent")
        metrics   = plan.get("metrics", ["sales"])
        m0        = plan.get("order_by") or metrics[0]
        grain     = plan.get("time_grain", "none")
        breakdown = plan.get("breakdown_by")
        lines: List[str] = []

        # ── kpi_detail ────────────────────────────────────────
        if intent == "kpi_detail" and "breakdown" in df.columns:
            total_loss   = float(df["profit"].sum()) if "profit" in df.columns else 0
            total_sales  = float(df["sales"].sum())  if "sales"  in df.columns else 0
            total_orders = int(df["orders"].sum())   if "orders" in df.columns else 0
            loss_ratio   = abs(total_loss) / total_sales * 100 if total_sales else 0

            lines.append(f"=== LOSS ANALYSIS ===")
            lines.append(f"Total loss: ${abs(total_loss):,.0f} on ${total_sales:,.0f} revenue")
            lines.append(f"Loss-to-revenue ratio: {loss_ratio:.1f}% (every $100 of sales loses ${loss_ratio:.2f})")
            lines.append(f"Affected sub-categories: {len(df)} | Affected orders: {total_orders:,}")
            lines.append("")

            # Per item with loss ratio
            lines.append("Per sub-category breakdown:")
            for i, (_, r) in enumerate(df.iterrows(), 1):
                name   = r.get("breakdown", "—")
                cat    = r.get("category", "")
                profit = float(r.get("profit", 0))
                sales  = float(r.get("sales", 0))
                orders = int(r.get("orders", 0))
                disc   = float(r.get("avg_discount_pct", 0))
                margin = float(r.get("profit_margin", 0))
                loss_share = abs(profit) / abs(total_loss) * 100 if total_loss else 0
                loss_per_order = abs(profit) / orders if orders else 0

                lines.append(
                    f"  {i}. {name} ({cat}):"
                    f"\n     loss=${abs(profit):,.0f} ({loss_share:.0f}% of total loss)"
                    f"\n     sales=${sales:,.0f} | orders={orders:,}"
                    f"\n     avg_discount={disc:.0f}% | margin={margin:.1f}%"
                    f"\n     loss_per_order=${loss_per_order:,.0f}"
                )
            return "\n".join(lines)

        # ── kpi_compare ───────────────────────────────────────
        if intent == "kpi_compare" and "current" in df.columns:
            row      = df.iloc[0]
            m        = str(row.get("metric", m0))
            cur      = float(row["current"])
            prev     = float(row["previous"])
            chg      = ((cur - prev) / abs(prev) * 100) if prev else None
            abs_diff = abs(cur - prev)
            direction = "increase" if (chg or 0) >= 0 else "decrease"

            lines += [
                f"=== PERIOD COMPARISON ===",
                f"Metric: {m}",
                f"Current  ({row['current_start']} → {row['current_end']}): ${cur:,.0f}",
                f"Previous ({row['prev_start']} → {row['prev_end']}): ${prev:,.0f}",
                f"Change: {chg:+.1f}% ({direction})" if chg is not None else "Change: n/a",
                f"Absolute delta: ${abs_diff:,.0f}",
                f"Context: {abs(chg):.1f}% {'is above' if abs(chg or 0) > 15 else 'is below'} typical 10-15% annual growth",
            ]
            return "\n".join(lines)

        # ── kpi_trend ─────────────────────────────────────────
        if intent == "kpi_trend" and "period" in df.columns:
            if breakdown and "breakdown" in df.columns:
                return self._data_summary_trend_breakdown(df, metrics, m0, grain, breakdown)

            sdf = df.sort_values("period").reset_index(drop=True)
            n   = len(sdf)

            lines.append(f"=== TREND ANALYSIS ===")
            lines.append(f"Time grain: {grain} | Metric: {m0} | Periods: {n}")

            if n >= 2:
                first_v   = float(sdf.iloc[0][m0])
                last_v    = float(sdf.iloc[-1][m0])
                total_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
                abs_chg   = abs(last_v - first_v)
                lines.append(f"Total change: {total_chg:+.1f}% (${first_v:,.0f} → ${last_v:,.0f}, Δ=${abs_chg:,.0f})")

                if grain == "year" and n >= 2:
                    years = n - 1
                    cagr  = ((last_v / first_v) ** (1 / years) - 1) * 100 if first_v > 0 else 0
                    lines.append(f"CAGR: {cagr:.2f}% over {years} year(s)")

            lines.append("")
            lines.append("Period-by-period:")
            transitions = []
            for i, (_, r) in enumerate(sdf.iterrows()):
                p    = _fmt_period(r.get("period", ""), grain)
                v    = float(r[m0])
                if i > 0:
                    prev_v = float(sdf.iloc[i-1][m0])
                    pct    = (v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                    abs_d  = abs(v - prev_v)
                    cls    = "🚀 strong" if pct >= 20 else ("📈 moderate" if pct >= 5 else ("📉 decline" if pct < 0 else "➡️ flat"))
                    transitions.append(pct)
                    lines.append(f"  {p}: ${v:,.0f} ({pct:+.1f}%, Δ=${abs_d:,.0f}) — {cls}")
                else:
                    lines.append(f"  {p}: ${v:,.0f} (baseline)")

            # Acceleration check
            if len(transitions) >= 2:
                accel = transitions[-1] - transitions[-2]
                direction = "accelerating" if accel > 0 else "decelerating"
                lines.append(f"Momentum: {direction} ({transitions[-2]:+.1f}% → {transitions[-1]:+.1f}%, Δ={accel:+.1f}pp)")

            # Recent vs prior average
            if n >= 4:
                recent_avg = float(sdf.tail(2)[m0].mean())
                prior_avg  = float(sdf.iloc[-4:-2][m0].mean())
                momentum   = (recent_avg - prior_avg) / abs(prior_avg) * 100 if prior_avg else 0
                lines.append(f"Recent 2-period avg: ${recent_avg:,.0f} vs prior 2-period avg: ${prior_avg:,.0f} ({momentum:+.1f}%)")

            peak   = sdf.loc[sdf[m0].idxmax()]
            trough = sdf.loc[sdf[m0].idxmin()]
            lines.append(f"Peak: {_fmt_period(peak.get('period',''), grain)} at ${float(peak[m0]):,.0f}")
            lines.append(f"Trough: {_fmt_period(trough.get('period',''), grain)} at ${float(trough[m0]):,.0f}")
            lines.append(f"Peak-to-trough swing: ${abs(float(peak[m0]) - float(trough[m0])):,.0f}")
            return "\n".join(lines)

        # ── kpi_rank / kpi_value with breakdown ───────────────
        if "breakdown" in df.columns and m0 in df.columns:
            sdf   = df.sort_values(by=m0, ascending=False).reset_index(drop=True)
            total = float(sdf[m0].sum())
            n     = len(sdf)

            lines.append(f"=== {'RANKING' if intent == 'kpi_rank' else 'BREAKDOWN'} ANALYSIS ===")
            lines.append(f"Dimension: {breakdown or 'item'} | Metric: {m0}")
            lines.append(f"Grand total: ${total:,.0f} | Entries: {n}")
            lines.append("")

            # Concentration index (top 2 share)
            if n >= 2:
                top2 = float(sdf.head(2)[m0].sum())
                top2_pct = top2 / total * 100 if total else 0
                lines.append(f"Concentration: top 2 hold {top2_pct:.0f}% of total ({'HIGH RISK' if top2_pct > 70 else 'moderate' if top2_pct > 50 else 'balanced'})")

            # Gap top-to-bottom
            if n >= 2:
                top_v = float(sdf.iloc[0][m0])
                bot_v = float(sdf.iloc[-1][m0])
                gap   = (top_v - bot_v) / abs(top_v) * 100 if top_v else 0
                lines.append(f"Gap #1 vs #last: ${abs(top_v - bot_v):,.0f} ({gap:.0f}%)")

            lines.append("")
            lines.append("Ranked entries:")

            for i, (_, r) in enumerate(sdf.iterrows(), 1):
                b   = r.get("breakdown", "—")
                v   = float(r[m0])
                pct = v / total * 100 if total else 0

                # Compute margin if sales+profit available
                margin_str = ""
                if "sales" in r and "profit" in r and m0 not in ("profit_margin",):
                    s_v = float(r.get("sales", 0))
                    p_v = float(r.get("profit", 0))
                    if s_v > 0:
                        margin = p_v / s_v * 100
                        margin_str = f" | margin={margin:.1f}%"

                flag = " ⚠️ LOSS" if v < 0 else (" 🏆 LEADER" if i == 1 else "")
                lines.append(
                    f"  {i}. {b}: ${v:,.0f} ({pct:.1f}% of total){margin_str}{flag}"
                )

            return "\n".join(lines)

        # ── kpi_value single ──────────────────────────────────
        r0 = df.iloc[0]
        lines.append(f"=== KPI SNAPSHOT ===")
        for m in metrics + ["profit", "orders"]:
            if m in r0:
                lines.append(f"{m}: {self._fv(float(r0[m]), m)}")
        if "sales" in r0 and "profit" in r0:
            ts = float(r0["sales"])
            tp = float(r0["profit"])
            if ts:
                margin = tp / ts * 100
                lines.append(f"profit_margin: {margin:.2f}%")
                lines.append(f"benchmark: {'ABOVE' if margin >= 12 else 'BELOW'} typical 12% retail margin")

        return "\n".join(lines)

    def _data_summary_trend_breakdown(self, df: pd.DataFrame, metrics: List[str],
                                      m0: str, grain: str, breakdown: str) -> str:
        dim_label     = breakdown.replace("_", " ").title()
        lines: List[str] = []
        period_totals = df.groupby("period")[m0].sum().sort_index()
        n_periods     = len(period_totals)
        n_segments    = df["breakdown"].nunique()

        lines.append(f"=== TREND BY {dim_label.upper()} ===")
        lines.append(f"Time grain: {grain} | Metric: {m0}")
        lines.append(f"Periods: {n_periods} | Segments: {n_segments}")

        if n_periods >= 2:
            total_first = float(period_totals.iloc[0])
            total_last  = float(period_totals.iloc[-1])
            total_chg   = (total_last - total_first) / abs(total_first) * 100 if total_first else 0
            p_first     = _fmt_period(period_totals.index[0], grain)
            p_last      = _fmt_period(period_totals.index[-1], grain)
            lines.append(
                f"Total across all {dim_label}s: "
                f"${total_first:,.0f} ({p_first}) → ${total_last:,.0f} ({p_last}) "
                f"= {total_chg:+.1f}%"
            )

        lines.append("")
        lines.append(f"Per-{dim_label} performance:")

        segment_changes = []
        for bk_name, bk_df in df.groupby("breakdown"):
            bk_sorted = bk_df.sort_values("period")
            if len(bk_sorted) < 2:
                continue
            first_v = float(bk_sorted.iloc[0][m0])
            last_v  = float(bk_sorted.iloc[-1][m0])
            chg     = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            abs_chg = abs(last_v - first_v)
            segment_changes.append((bk_name, chg, abs_chg, first_v, last_v))

            per_period = []
            prev_v = None
            for _, r in bk_sorted.iterrows():
                p = _fmt_period(r.get("period", ""), grain)
                v = float(r.get(m0, 0))
                if prev_v is not None and prev_v != 0:
                    pct = (v - prev_v) / abs(prev_v) * 100
                    per_period.append(f"{p}:${v:,.0f}({pct:+.1f}%)")
                else:
                    per_period.append(f"{p}:${v:,.0f}(baseline)")
                prev_v = v

            lines.append(
                f"  {bk_name}: overall {chg:+.1f}% "
                f"(${first_v:,.0f}→${last_v:,.0f}, Δ=${abs_chg:,.0f})"
            )
            lines.append(f"    {' | '.join(per_period)}")

        # Contribution analysis
        if len(segment_changes) >= 2:
            total_abs = sum(abs(s[2]) for s in segment_changes)
            lines.append("")
            lines.append("Contribution to total change:")
            for name, chg, abs_chg, fv, lv in sorted(segment_changes, key=lambda x: -x[2]):
                contrib = abs_chg / total_abs * 100 if total_abs else 0
                direction = "growth" if chg > 0 else "drag"
                lines.append(f"  {name}: {contrib:.0f}% of total change (${abs_chg:,.0f} {direction})")

        return "\n".join(lines)

    # ── Rule-based insight (giữ nguyên logic, improve text) ───

    def _rule_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        intent    = plan.get("intent")
        metrics   = plan.get("metrics", ["sales"])
        m0        = plan.get("order_by") or metrics[0]
        breakdown = plan.get("breakdown_by")
        secondary = plan.get("secondary_breakdown")
        insights: List[str] = []

        if (intent in {"kpi_rank", "kpi_value"}
                and breakdown and secondary
                and "breakdown" in df.columns
                and "breakdown2" in df.columns
                and m0 in df.columns):

            grand_total   = float(df[m0].sum())
            region_totals = df.groupby("breakdown2")[m0].sum().sort_values(ascending=False)
            top_region    = region_totals.index[0]
            top_region_v  = float(region_totals.iloc[0])
            bot_region    = region_totals.index[-1]
            bot_region_v  = float(region_totals.iloc[-1])
            gap_pct       = (top_region_v - bot_region_v) / top_region_v * 100 if top_region_v else 0

            insights.append(
                f"**{top_region}** leads with {self._fv(top_region_v, m0)} "
                f"({top_region_v / grand_total * 100:.0f}% of total) — "
                f"**{gap_pct:.0f}% ahead** of weakest region "
                f"**{bot_region}** ({self._fv(bot_region_v, m0)})."
            )

            cat_totals     = df.groupby("breakdown")[m0].sum().sort_values(ascending=False)
            top_cat_global = cat_totals.index[0]
            top2_share     = float(cat_totals.iloc[:2].sum()) / grand_total * 100 if grand_total else 0

            outlier_regions = []
            for region_val in region_totals.index:
                rdf     = df[df["breakdown2"] == region_val]
                top_row = rdf.sort_values(m0, ascending=False).iloc[0]
                if top_row.get("breakdown") != top_cat_global:
                    outlier_regions.append(f"**{region_val}** (led by {top_row.get('breakdown','—')})")

            if outlier_regions:
                insights.append(
                    f"While **{top_cat_global}** dominates globally ({top2_share:.0f}% of total), "
                    + ", ".join(outlier_regions)
                    + " show different patterns — suggesting distinct regional demand."
                )
            else:
                insights.append(
                    f"**{top_cat_global}** dominates across all regions — "
                    f"top 2 categories hold **{top2_share:.0f}%** of total, "
                    "indicating high concentration risk."
                )
            return insights

        if (intent == "kpi_trend"
                and breakdown and "breakdown" in df.columns
                and "period" in df.columns and m0 in df.columns):
            return self._rule_insight_trend_breakdown(df, m0, breakdown, plan)

        if intent in {"kpi_rank", "kpi_value"} and breakdown and "breakdown" in df.columns and m0 in df.columns:
            sdf   = df.sort_values(by=m0, ascending=False).reset_index(drop=True)
            n     = len(sdf)
            total = float(sdf[m0].sum())

            if n >= 1:
                top_val   = float(sdf.iloc[0][m0])
                top_share = top_val / total * 100 if total else 0
                insights.append(
                    f"**{sdf.iloc[0]['breakdown']}** leads with {self._fv(top_val, m0)}, "
                    f"representing **{top_share:.0f}%** of total {self._fv(total, m0)}."
                )

            if n >= 2:
                tv  = float(sdf.iloc[0][m0])
                sv  = float(sdf.iloc[1][m0])
                gap = abs(tv - sv) / abs(tv) * 100 if tv else 0
                if gap > 40:
                    insights.append(
                        f"**{gap:.0f}% gap** separates leader from #2 "
                        f"**{sdf.iloc[1]['breakdown']}** ({self._fv(sv, m0)}) — "
                        "highly uneven distribution."
                    )
                else:
                    insights.append(
                        f"**{sdf.iloc[1]['breakdown']}** follows at {self._fv(sv, m0)} "
                        f"— **{gap:.0f}%** behind, indicating competitive parity."
                    )

            if n >= 3:
                top2_sum = float(sdf.head(2)[m0].sum())
                top2_pct = top2_sum / total * 100 if total > 0 else 0
                if top2_pct >= 60:
                    insights.append(
                        f"⚠️ Top 2 entities control **{top2_pct:.0f}%** of total — "
                        "concentration risk: underperformance by either would significantly impact results."
                    )

            if m0 in {"profit", "profit_margin"}:
                negatives = sdf[sdf[m0] < 0]
                if not negatives.empty:
                    names = ", ".join(f"**{r['breakdown']}**" for _, r in negatives.iterrows())
                    worst = float(negatives.iloc[-1][m0])
                    insights.append(
                        f"⚠️ {names} operating at a loss (worst: {self._fv(worst, m0)}) — "
                        "requires immediate discount/cost review."
                    )

            if n >= 4:
                bottom_v    = float(sdf.iloc[-1][m0])
                bottom_name = str(sdf.iloc[-1]["breakdown"])
                bottom_pct  = bottom_v / total * 100 if total else 0
                insights.append(
                    f"**{bottom_name}** contributes only {self._fv(bottom_v, m0)} "
                    f"({bottom_pct:.0f}% of total) — assess whether this reflects "
                    "market size or structural underperformance."
                )

        elif intent == "kpi_trend" and "period" in df.columns and m0 in df.columns:
            sdf = df.sort_values("period").reset_index(drop=True)
            n   = len(sdf)
            if n < 2:
                return insights

            first_v    = float(sdf.iloc[0][m0])
            last_v     = float(sdf.iloc[-1][m0])
            chg        = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            abs_change = abs(last_v - first_v)
            word       = "grown" if chg > 0 else "declined"
            grain      = plan.get("time_grain", "none")

            insights.append(
                f"Over **{n} periods**, **{_label(m0)}** {word} **{abs(chg):.1f}%** — "
                f"from {self._fv(first_v, m0)} to {self._fv(last_v, m0)} "
                f"(Δ {self._fv(abs_change, m0)})."
            )

            if grain in ("year", "quarter", "month") and n <= 24:
                period_lines, best_chg, best_idx, worst_chg, worst_idx = [], -999, 1, 999, 1
                transitions = []
                for i in range(1, n):
                    prev_v = float(sdf.iloc[i-1][m0])
                    curr_v = float(sdf.iloc[i][m0])
                    pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                    abs_d  = abs(curr_v - prev_v)
                    transitions.append(pct)
                    if pct > best_chg:  best_chg,  best_idx  = pct, i
                    if pct < worst_chg: worst_chg, worst_idx = pct, i
                    prev_p = _fmt_period(sdf.iloc[i-1].get("period",""), grain)
                    curr_p = _fmt_period(sdf.iloc[i].get("period",""), grain)
                    tag    = ("🚀 Strong" if pct>=20 else "📈 Moderate" if pct>=5
                              else "📉 Decline" if pct<0 else "➡️ Flat")
                    period_lines.append(
                        f"  - **{prev_p}→{curr_p}:** {pct:+.1f}% "
                        f"({self._fv(prev_v,m0)}→{self._fv(curr_v,m0)}, Δ{self._fv(abs_d,m0)}) — {tag}"
                    )
                insights.append("**Period transitions:**\n" + "\n".join(period_lines))

                if len(transitions) >= 2:
                    accel = transitions[-1] - transitions[-2]
                    direction = "accelerating 📈" if accel > 0 else "decelerating 📉"
                    insights.append(
                        f"Momentum is **{direction}** — last transition "
                        f"({transitions[-1]:+.1f}%) vs prior ({transitions[-2]:+.1f}%), "
                        f"Δ={accel:+.1f}pp."
                    )

            peak   = sdf.loc[sdf[m0].idxmax()]
            trough = sdf.loc[sdf[m0].idxmin()]
            insights.append(
                f"🏆 Peak: **{_fmt_period(peak.get('period',''), grain)}** at {self._fv(float(peak[m0]),m0)} | "
                f"📉 Trough: **{_fmt_period(trough.get('period',''), grain)}** at {self._fv(float(trough[m0]),m0)} — "
                f"swing of **{self._fv(abs(float(peak[m0])-float(trough[m0])),m0)}**."
            )

            if grain == "year" and n >= 2 and first_v > 0:
                years = n - 1
                cagr  = ((last_v / first_v) ** (1/years) - 1)
                proj  = last_v * (1 + cagr)
                last_year = _fmt_period(sdf.iloc[-1].get("period",""), grain)
                try:
                    next_year = str(int(last_year) + 1)
                except Exception:
                    next_year = "next period"
                insights.append(
                    f"📊 At **{cagr*100:.1f}% CAGR**, base-case forecast for "
                    f"**{next_year}**: **{self._fv(proj, m0)}**."
                )

        elif intent == "kpi_compare" and "current" in df.columns:
            row  = df.iloc[0]
            cur  = float(row["current"])
            prev = float(row["previous"])
            if prev == 0:
                return []
            m    = str(row.get("metric", m0))
            chg  = (cur - prev) / abs(prev) * 100
            cp   = plan.get("compare_period", "prev_period")
            cp_label = {"yoy":"year-over-year","mom":"month-over-month",
                        "prev_period":"vs previous period"}.get(cp, cp)

            direction = "increased" if chg > 0 else "decreased"
            insights.append(
                f"**{m.replace('_',' ').title()}** {direction} **{abs(chg):.1f}%** {cp_label} — "
                f"from {self._fv(prev,m)} to {self._fv(cur,m)} (Δ {self._fv(abs(cur-prev),m)})."
            )

            if abs(chg) >= 20:
                insights.append(
                    f"A **{abs(chg):.0f}%** swing is significant — "
                    "likely driven by volume change, pricing shift, or seasonal effect. "
                    "Cross-reference with order count to decompose."
                )
            elif abs(chg) >= 10:
                insights.append(
                    f"**{abs(chg):.1f}%** is a meaningful move. "
                    + ("Sustain momentum by identifying the top-performing segment." if chg > 0
                       else "Investigate whether decline is concentrated in one region or broad-based.")
                )
            else:
                insights.append(
                    f"Change of **{abs(chg):.1f}%** is within normal variance. "
                    "No immediate action required — watch for sustained drift over 2+ periods."
                )

        elif intent == "kpi_value" and not breakdown and "sales" in df.columns:
            r0  = df.iloc[0]
            ts  = float(r0.get("sales", 0))
            tp  = float(r0.get("profit", 0))
            pm  = (tp / ts * 100) if ts else 0
            to_ = int(r0.get("orders", 0)) if "orders" in r0 else None

            bm_label = "ABOVE" if pm >= 12 else "BELOW"
            insights.append(
                f"Profit margin **{pm:.1f}%** is **{bm_label}** the 12% retail benchmark — "
                f"{self._fv(tp,'profit')} retained from {self._fv(ts,'sales')} revenue."
            )
            if to_:
                aov = ts / to_
                insights.append(
                    f"**{to_:,} orders** at avg **{self._fv(aov,'sales')}** each. "
                    f"A 10% AOV increase would add **{self._fv(ts*0.1,'sales')}** revenue."
                )

        return insights

    def _rule_insight_trend_breakdown(self, df: pd.DataFrame, m0: str,
                                      breakdown: str, plan: Dict[str, Any]) -> List[str]:
        grain    = plan.get("time_grain", "year")
        insights: List[str] = []

        segment_stats = []
        for bk_name, bk_df in df.groupby("breakdown"):
            bk_sorted = bk_df.sort_values("period")
            if len(bk_sorted) < 2:
                continue
            first_v = float(bk_sorted.iloc[0][m0])
            last_v  = float(bk_sorted.iloc[-1][m0])
            chg     = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            segment_stats.append({
                "name": bk_name, "first_v": first_v, "last_v": last_v,
                "change": chg, "n_periods": len(bk_sorted),
                "abs_change": abs(last_v - first_v),
            })

        if not segment_stats:
            return insights

        segment_stats.sort(key=lambda x: x["change"], reverse=True)
        dim_label = breakdown.replace("_", " ").title()
        best  = segment_stats[0]
        worst = segment_stats[-1]

        insights.append(
            f"**{best['name']}** leads with **{best['change']:+.1f}%** growth "
            f"({self._fv(best['first_v'],m0)} → {self._fv(best['last_v'],m0)}). "
            f"**{worst['name']}** lags at **{worst['change']:+.1f}%** "
            f"({self._fv(worst['first_v'],m0)} → {self._fv(worst['last_v'],m0)})."
        )

        growers   = [s for s in segment_stats if s["change"] > 5]
        decliners = [s for s in segment_stats if s["change"] < -5]

        if growers and decliners:
                insights.append(
                    f"⚠️ **Divergence:** {', '.join([f'**{s["name"]}**' for s in growers])} grew "
                    f"while {', '.join([f'**{s["name"]}**' for s in decliners])} declined — "
                    f"overall trend masks segment-level variation."
                )

        period_totals = df.groupby("period")[m0].sum().sort_index()
        if len(period_totals) >= 2:
            total_first = float(period_totals.iloc[0])
            total_last  = float(period_totals.iloc[-1])
            if total_first:
                total_chg = (total_last - total_first) / abs(total_first) * 100
                p_first   = _fmt_period(period_totals.index[0], grain)
                p_last    = _fmt_period(period_totals.index[-1], grain)
                insights.append(
                    f"Total {_label(m0)}: **{total_chg:+.1f}%** overall "
                    f"({self._fv(total_first,m0)} in {p_first} → {self._fv(total_last,m0)} in {p_last})."
                )

        period_totals = df.groupby("period")[m0].sum().sort_index()
        if len(segment_stats) >= 2 and len(period_totals) >= 2:
            total_abs = float(period_totals.iloc[-1]) - float(period_totals.iloc[0])
            if total_abs != 0:
                contributions = []
                for s in segment_stats:
                    abs_chg_s  = s["last_v"] - s["first_v"]
                    contrib_pct = abs_chg_s / total_abs * 100 if total_abs else 0
                    contributions.append((s["name"], contrib_pct, abs_chg_s))
                contributions.sort(key=lambda x: abs(x[2]), reverse=True)
                top = contributions[0]
                insights.append(
                    f"**{top[0]}** drove **{abs(top[1]):.0f}%** of total change "
                    f"({self._fv(abs(top[2]),m0)}) — "
                    f"the primary {'growth driver' if top[2] > 0 else 'drag'}."
                )

        return insights

    # ── Grounding verifier ────────────────────────────────────

    def _verify_grounding(self, insight_text: str, data_summary: str) -> float:
        def _normalize_nums(text: str) -> set:
            nums: set = set()
            for m in _re.finditer(r'\$?([\d][\d,]*(?:\.\d+)?)', text):
                raw = m.group(1).replace(",", "")
                try:
                    nums.add(round(float(raw), 1))
                except ValueError:
                    pass
            for m in _re.finditer(r'([\d]+\.?[\d]*)%', text):
                try:
                    nums.add(round(float(m.group(1)), 1))
                except ValueError:
                    pass
            return nums

        insight_nums = _normalize_nums(insight_text)
        summary_nums = _normalize_nums(data_summary)
        if not insight_nums:
            return 1.0
        matched = insight_nums & summary_nums
        return len(matched) / len(insight_nums)

    @staticmethod
    def _trim_to_complete_sentence(text: str) -> str:
        import re
        text = text.strip()
        if text and text[-1] in ".!?":
            return text
        matches = list(re.finditer(r'[.!?](\*{0,2}[\)\s]|$)', text))
        if matches:
            return text[:matches[-1].end()].strip()
        return text

    @staticmethod
    def _fv(v: float, metric: str) -> str:
        if metric in {"sales", "profit"}:
            return f"\\${v:,.0f}"
        if metric == "profit_margin":
            return f"{v:.2f}%"
        return f"{int(v):,}"


# ── Module-level helpers ──────────────────────────────────────

def _label(metric: str) -> str:
    return {
        "sales":         "Total Sales",
        "profit":        "Total Profit",
        "orders":        "Total Orders",
        "profit_margin": "Profit Margin",
    }.get(metric, metric.replace("_", " ").title())


def _fmt_period(raw: Any, grain: str) -> str:
    s = str(raw)
    if grain == "year":    return s[:4]
    elif grain == "quarter":
        try:
            from datetime import datetime
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return f"{dt.year} Q{(dt.month-1)//3+1}"
        except Exception:
            return s[:7]
    elif grain == "month": return s[:7]
    return s[:10]