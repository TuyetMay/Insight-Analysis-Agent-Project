"""
chatbot/insight_generator.py
Generates a deep analytical insight for a query result.
Tries Gemini first; falls back to deterministic rule-based insight on failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from google.genai import types as genai_types


class InsightGenerator:
    """
    Usage:
        gen = InsightGenerator(gemini_client, model_name)
        insight = gen.generate(plan, result_df)
        # returns "\n\n---\n💡 **Insight:** ..." or ""
    """

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
        summary   = self._data_summary(plan, df)
        intent    = plan.get("intent", "kpi_value")
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        grain     = plan.get("time_grain", "none")
        cp        = plan.get("compare_period")

        intent_guidance = {
            "kpi_detail": (
                "Analyse these loss-making products/sub-categories deeply:\n"
                "- What is the total financial damage (sum of losses)?\n"
                "- Which sub-category or product is the biggest loss driver?\n"
                "- Is heavy discounting (avg_discount_pct) the main cause?\n"
                "- Are losses concentrated in one category or spread broadly?\n"
                "- What action should the business take to fix this?"
            ),
            "kpi_rank": (
                "Analyse the ranking deeply:\n"
                "- Who leads and by how much (gap %)?\n"
                "- Are results concentrated (top 2 dominating) or spread evenly?\n"
                "- Which entries are underperforming or at a loss?\n"
                "- What strategic action does this ranking suggest?"
            ),
            "kpi_value": (
                f"Analyse the breakdown across {breakdown or 'all data'} deeply:\n"
                "- Identify the dominant player and its share of total.\n"
                "- Point out any underperformer or outlier (loss-making, far below average).\n"
                "- Compare the spread: is there a large gap between top and bottom?\n"
                "- What business implication does this distribution carry?"
            ),
            "kpi_trend": (
                f"Analyse this {grain}-level trend DEEPLY and SPECIFICALLY using the numbers provided:\n"
                "1. **Overall trajectory**: Is the trend growing, declining, or volatile across the full period? "
                "State the total cumulative growth from first to last period with exact numbers.\n"
                "2. **Best and worst periods**: Name the specific peak period and trough period with their exact values. "
                "Calculate the gap between them.\n"
                "3. **Year-by-year breakdown** (if year grain): For EACH period transition, state the exact % change "
                "and classify it (strong growth / moderate growth / decline / recovery). "
                "Example: '2014→2015: -2.8% (slight dip), 2015→2016: +29.5% (breakout year)'.\n"
                "4. **CAGR** (if multiple years): Calculate the Compound Annual Growth Rate from first to last year.\n"
                "5. **Momentum and forecast**: Based on the most recent 2–3 periods, is momentum accelerating or slowing? "
                "What does this imply for the next period?\n"
                "6. **Root cause hypothesis**: What business factors typically explain this pattern "
                "(seasonality, market expansion, operational changes)?\n"
                "Be specific with numbers. Do NOT use vague phrases like 'strong performance' without a number."
            ),
            "kpi_compare": (
                f"Analyse this {cp} period comparison deeply:\n"
                "- State the magnitude of change and whether it's significant.\n"
                "- Is this a one-time spike/dip or part of a sustained trend?\n"
                "- What operational or market factors could explain this shift?\n"
                "- What should the business monitor or act on next?"
            ),
        }.get(intent, "Provide a thorough analytical observation using the data.")

        prompt = f"""You are a senior business analyst writing a performance commentary for an executive audience.
Using ONLY the numbers in the DATA section below, write a **deep analytical insight**.

=== DATA ===
{summary}

=== ANALYSIS TASK ===
{intent_guidance}

=== WRITING RULES ===
- Write 6–10 sentences OR 5–7 bullet points. Be SUBSTANTIVE — go deep, not wide.
- Lead with the single most important finding with a specific number.
- Use **bold** for ALL critical numbers, percentages, and period names.
- Every sentence must contain at least one number from the DATA section.
- Include CAGR if multiple years of data are present.
- Include a forward-looking or actionable observation in the final 1–2 sentences.
- Do NOT invent figures that are not in the DATA section.
- Do NOT use filler openers like "The data shows", "Overall", "In conclusion", "Notably".
- Use clear, direct business language. No jargon.
- Format: flowing prose OR a short bulleted list — pick whichever fits better for trend data use bullets.

Write the insight now:"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.55,
                    max_output_tokens=900,   # ← increased from 450 to allow full analysis
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if len(text) > 30:
                return text
        except Exception:
            pass
        return ""

    # ── Rule-based insight ────────────────────────────────────

    def _rule_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        intent    = plan.get("intent")
        metrics   = plan.get("metrics", ["sales"])
        m0        = plan.get("order_by") or metrics[0]
        breakdown = plan.get("breakdown_by")
        insights: List[str] = []

        if intent in {"kpi_rank", "kpi_value"} and breakdown and "breakdown" in df.columns and m0 in df.columns:
            sdf   = df.sort_values(by=m0, ascending=False).reset_index(drop=True)
            n     = len(sdf)
            total = float(sdf[m0].sum()) if m0 in sdf.columns else 0

            if n >= 1:
                top_val   = float(sdf.iloc[0][m0])
                top_share = (top_val / total * 100) if total else 0
                insights.append(
                    f"**{sdf.iloc[0]['breakdown']}** leads with {self._fv(top_val, m0)}, "
                    f"representing **{top_share:.0f}%** of the total {self._fv(total, m0)}."
                )

            if n >= 2:
                tv  = float(sdf.iloc[0][m0])
                sv  = float(sdf.iloc[1][m0])
                gap = abs(tv - sv) / abs(tv) * 100 if tv else 0
                if gap > 40:
                    insights.append(
                        f"There is a significant **{gap:.0f}% gap** between the leader "
                        f"and **{sdf.iloc[1]['breakdown']}** ({self._fv(sv, m0)}), "
                        f"suggesting a highly uneven distribution of {m0}."
                    )
                else:
                    insights.append(
                        f"**{sdf.iloc[1]['breakdown']}** follows closely at {self._fv(sv, m0)} "
                        f"— only **{gap:.0f}%** behind the leader, indicating competitive parity."
                    )

            if n >= 3:
                top2_sum = float(sdf.head(2)[m0].sum())
                if total > 0:
                    top2_pct = top2_sum / total * 100
                    top2_names = " & ".join(str(sdf.iloc[i]["breakdown"]) for i in range(2))
                    if top2_pct >= 60:
                        insights.append(
                            f"**{top2_names}** together account for **{top2_pct:.0f}%** of total — "
                            f"performance is heavily concentrated in these two, "
                            f"posing a risk if either declines."
                        )
                    else:
                        insights.append(
                            f"The top two (**{top2_names}**) hold {top2_pct:.0f}% of total, "
                            f"indicating a relatively balanced spread across all {n} entries."
                        )

            if m0 in {"profit", "profit_margin"}:
                negatives = sdf[sdf[m0] < 0]
                if not negatives.empty:
                    names = ", ".join(f"**{r['breakdown']}**" for _, r in negatives.iterrows())
                    worst = float(negatives.iloc[-1][m0])
                    insights.append(
                        f"⚠️ {names} {'is' if len(negatives)==1 else 'are'} operating at a loss "
                        f"(worst: {self._fv(worst, m0)}). "
                        f"These require urgent cost or pricing review."
                    )

            if n >= 4:
                bottom_val = float(sdf.iloc[-1][m0])
                bottom_name = str(sdf.iloc[-1]["breakdown"])
                insights.append(
                    f"At the bottom, **{bottom_name}** contributes only {self._fv(bottom_val, m0)}. "
                    f"Investigate whether this reflects a structural weakness or simply smaller market size."
                )

        elif intent == "kpi_trend" and "period" in df.columns and m0 in df.columns:
            sdf = df.sort_values("period").reset_index(drop=True)
            n   = len(sdf)

            if n < 2:
                return insights

            first_v = float(sdf.iloc[0][m0])
            last_v  = float(sdf.iloc[-1][m0])
            chg     = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            word    = "grown" if chg > 0 else "declined"

            # ── 1. Overall growth ─────────────────────────────
            insights.append(
                f"Over the full period, **{_label(m0)}** has {word} by **{abs(chg):.1f}%** — "
                f"from {self._fv(first_v, m0)} to {self._fv(last_v, m0)} "
                f"(absolute change: {self._fv(abs(last_v - first_v), m0)})."
            )

            # ── 2. CAGR (for year grain or multi-year span) ───
            grain = plan.get("time_grain", "none")
            if grain == "year" and n >= 2:
                years = n - 1
                cagr  = ((last_v / first_v) ** (1 / years) - 1) * 100 if first_v > 0 else 0
                insights.append(
                    f"The **Compound Annual Growth Rate (CAGR)** over {years} year{'s' if years > 1 else ''} "
                    f"is **{cagr:.1f}%**, indicating "
                    f"{'healthy' if cagr >= 10 else 'moderate' if cagr >= 5 else 'slow'} long-term expansion."
                )

            # ── 3. Year-by-year or period-by-period breakdown ─
            if grain == "year" and n <= 10:
                period_lines = []
                for i in range(1, n):
                    prev_p = str(sdf.iloc[i - 1].get("period", ""))[:4]
                    curr_p = str(sdf.iloc[i].get("period", ""))[:4]
                    prev_v = float(sdf.iloc[i - 1][m0])
                    curr_v = float(sdf.iloc[i][m0])
                    pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                    tag    = (
                        "🚀 strong growth" if pct >= 20
                        else "📈 moderate growth" if pct >= 5
                        else "📉 decline" if pct < 0
                        else "➡️ flat"
                    )
                    period_lines.append(
                        f"  - **{prev_p}→{curr_p}:** {pct:+.1f}% "
                        f"({self._fv(prev_v, m0)} → {self._fv(curr_v, m0)}) — {tag}"
                    )
                if period_lines:
                    insights.append(
                        "**Period-by-period breakdown:**\n" + "\n".join(period_lines)
                    )

            # ── 4. Peak and trough ────────────────────────────
            peak   = sdf.loc[sdf[m0].idxmax()]
            trough = sdf.loc[sdf[m0].idxmin()]
            peak_p   = _fmt_period(peak.get("period", ""), grain)
            trough_p = _fmt_period(trough.get("period", ""), grain)
            range_v  = abs(float(peak[m0]) - float(trough[m0]))
            insights.append(
                f"**Peak:** {peak_p} at {self._fv(float(peak[m0]), m0)} | "
                f"**Trough:** {trough_p} at {self._fv(float(trough[m0]), m0)} — "
                f"range of **{self._fv(range_v, m0)}** ({range_v / abs(float(peak[m0])) * 100:.0f}% swing)."
            )

            # ── 5. Recent momentum ────────────────────────────
            if n >= 4:
                recent_avg = float(sdf.tail(2)[m0].mean())
                prior_avg  = float(sdf.iloc[-4:-2][m0].mean())
                if prior_avg:
                    momentum = (recent_avg - prior_avg) / abs(prior_avg) * 100
                    direction = (
                        "**accelerating 📈**" if momentum > 10
                        else "**decelerating 📉**" if momentum < -10
                        else "**stable ➡️**"
                    )
                    insights.append(
                        f"Recent momentum is {direction} — the last 2 periods averaged "
                        f"{self._fv(recent_avg, m0)} vs {self._fv(prior_avg, m0)} in the prior 2 "
                        f"({momentum:+.1f}%). "
                        f"{'Expect continued growth if conditions hold.' if momentum > 5 else 'Watch for further slowdown.' if momentum < -5 else 'Growth is steady but not accelerating.'}"
                    )

        elif intent == "kpi_compare" and "current" in df.columns:
            row  = df.iloc[0]
            cur  = float(row["current"])
            prev = float(row["previous"])
            m    = str(row.get("metric", m0))
            cp   = plan.get("compare_period", "prev_period")

            if prev:
                chg = (cur - prev) / abs(prev) * 100
                direction = "increased" if chg > 0 else "decreased"
                cp_label  = {"yoy": "year-over-year", "mom": "month-over-month",
                             "prev_period": "vs the previous period"}.get(cp, cp)

                insights.append(
                    f"**{m.replace('_', ' ').title()}** has {direction} by **{abs(chg):.1f}%** {cp_label} — "
                    f"from {self._fv(prev, m)} to {self._fv(cur, m)}."
                )

                if abs(chg) >= 30:
                    driver = "significant structural shift or seasonality" if abs(chg) >= 50 else "notable operational change"
                    insights.append(
                        f"A **{abs(chg):.0f}% swing** is material and likely reflects a {driver}. "
                        f"Cross-referencing with volume (orders) and margin data is recommended."
                    )
                elif abs(chg) >= 10:
                    insights.append(
                        f"A **{abs(chg):.1f}%** move is meaningful. "
                        f"{'Monitor whether this momentum continues next period.' if chg > 0 else 'Investigate root causes — pricing, volume, or mix shift.'}"
                    )
                else:
                    insights.append(
                        f"The change is modest (**{abs(chg):.1f}%**), suggesting relative stability. "
                        f"No immediate action required, but watch for further drift."
                    )

                delta_abs = abs(cur - prev)
                if delta_abs > 0:
                    insights.append(
                        f"In absolute terms, the difference is **{self._fv(delta_abs, m)}** — "
                        f"{'a meaningful contribution to the bottom line.' if m == 'profit' else 'which reflects real business volume change.'}"
                    )

        elif intent == "kpi_value" and not breakdown and "sales" in df.columns:
            r0 = df.iloc[0]
            ts = float(r0.get("sales", 0))
            tp = float(r0.get("profit", 0))
            pm = (tp / ts * 100) if ts else 0
            to_ = int(r0.get("orders", 0)) if "orders" in r0 else None

            if pm < 5:
                insights.append(
                    f"⚠️ The profit margin of **{pm:.1f}%** is dangerously low. "
                    f"On **{self._fv(ts, 'sales')}** in revenue, only **{self._fv(tp, 'profit')}** "
                    f"reaches the bottom line — review pricing strategy and cost structure immediately."
                )
            elif pm > 20:
                insights.append(
                    f"The profit margin of **{pm:.1f}%** is strong and well above typical retail benchmarks (~12%). "
                    f"This means **{self._fv(tp, 'profit')}** is retained from **{self._fv(ts, 'sales')}** in revenue — "
                    f"a sign of healthy pricing power and cost control."
                )
            else:
                insights.append(
                    f"The **{pm:.1f}% profit margin** is within normal retail range. "
                    f"**{self._fv(tp, 'profit')}** profit on **{self._fv(ts, 'sales')}** revenue "
                    f"leaves room for improvement — targeting 15–20% should be a medium-term goal."
                )

            if to_:
                avg = ts / to_
                insights.append(
                    f"With **{to_:,} orders** and an average order value of **{self._fv(avg, 'sales')}**, "
                    f"growing either order volume or average basket size would have an outsized impact on total profit."
                )

        return insights

    # ── Data summary for LLM ──────────────────────────────────

    def _data_summary(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        intent  = plan.get("intent")
        metrics = plan.get("metrics", ["sales"])
        m0      = plan.get("order_by") or metrics[0]
        grain   = plan.get("time_grain", "none")
        lines: List[str] = []

        if intent == "kpi_detail" and "breakdown" in df.columns:
            total_loss  = float(df["profit"].sum()) if "profit" in df.columns else 0
            total_sales = float(df["sales"].sum())  if "sales"  in df.columns else 0
            lines.append(f"Total loss: \\${abs(total_loss):,.0f} on \\${total_sales:,.0f} revenue")
            lines.append(f"Sub-categories in loss: {len(df)}")
            for i, (_, r) in enumerate(df.head(10).iterrows(), 1):
                disc = float(r.get("avg_discount_pct", 0))
                lines.append(
                    f"  {i}. {r.get('breakdown','—')} ({r.get('category','')}) — "
                    f"profit=\\${float(r.get('profit',0)):,.0f} | "
                    f"sales=\\${float(r.get('sales',0)):,.0f} | "
                    f"orders={int(r.get('orders',0))} | avg_discount={disc:.0f}%"
                )
            return "\n".join(lines)

        if intent == "kpi_compare" and "current" in df.columns:
            row = df.iloc[0]
            m   = str(row.get("metric", m0))
            cur, prev = float(row["current"]), float(row["previous"])
            chg = ((cur - prev) / abs(prev) * 100) if prev else None
            lines += [
                f"Metric: {m}",
                f"Current period ({row['current_start']} – {row['current_end']}): {self._fv(cur, m)}",
                f"Previous period ({row['prev_start']} – {row['prev_end']}): {self._fv(prev, m)}",
                f"Change: {chg:+.1f}%" if chg is not None else "Change: n/a",
                f"Absolute delta: {self._fv(abs(cur - prev), m)}",
            ]

        elif intent == "kpi_trend" and "period" in df.columns:
            # ── Provide rich trend context for LLM ─────────────
            all_rows = df.sort_values("period").reset_index(drop=True)
            n = len(all_rows)
            lines.append(f"Time grain: {grain} | Metric: {m0} | Total periods: {n}")

            if n >= 2:
                first_v = float(all_rows.iloc[0][m0])
                last_v  = float(all_rows.iloc[-1][m0])
                total_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
                lines.append(f"Total change first→last: {total_chg:+.1f}% ({self._fv(first_v, m0)} → {self._fv(last_v, m0)})")
                if grain == "year" and n >= 2:
                    years = n - 1
                    cagr  = ((last_v / first_v) ** (1 / years) - 1) * 100 if first_v > 0 else 0
                    lines.append(f"CAGR over {years} year(s): {cagr:.1f}%")

            lines.append("Period-by-period data:")
            for i, (_, r) in enumerate(all_rows.iterrows()):
                p    = _fmt_period(r.get("period", ""), grain)
                vals = "  ".join(f"{c}={self._fv(float(r[c]), c)}" for c in metrics if c in r)
                if i > 0:
                    prev_v = float(all_rows.iloc[i - 1][m0])
                    curr_v = float(r[m0])
                    pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                    lines.append(f"  {p}: {vals}  (change vs prior: {pct:+.1f}%)")
                else:
                    lines.append(f"  {p}: {vals}  (baseline)")

            peak    = all_rows.loc[all_rows[m0].idxmax()]
            trough  = all_rows.loc[all_rows[m0].idxmin()]
            lines.append(f"Peak period: {_fmt_period(peak.get('period',''), grain)} at {self._fv(float(peak[m0]), m0)}")
            lines.append(f"Trough period: {_fmt_period(trough.get('period',''), grain)} at {self._fv(float(trough[m0]), m0)}")

        elif "breakdown" in df.columns:
            sdf   = df.sort_values(by=m0, ascending=False).head(20).reset_index(drop=True)
            total = float(sdf[m0].sum()) if m0 in sdf.columns else 0
            lines.append(f"Breakdown dimension: {plan.get('breakdown_by')} | Primary metric: {m0} | Grand total: {self._fv(total, m0)}")
            lines.append(f"Number of entries: {len(sdf)}")
            for i, (_, r) in enumerate(sdf.iterrows(), 1):
                vals = "  ".join(f"{c}={self._fv(float(r[c]), c)}" for c in metrics if c in r)
                share = (float(r[m0]) / total * 100) if total else 0
                lines.append(f"  {i}. {r['breakdown']}: {vals} ({share:.1f}% of total)")
        else:
            r0 = df.iloc[0]
            for m in metrics + ["profit", "orders"]:
                if m in r0:
                    lines.append(f"{m}: {self._fv(float(r0[m]), m)}")

        return "\n".join(lines)

    @staticmethod
    def _fv(v: float, metric: str) -> str:
        if metric in {"sales", "profit"}:  return f"\\${v:,.0f}"
        if metric == "profit_margin":      return f"{v:.2f}%"
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
    if grain == "year":
        return s[:4]
    elif grain == "quarter":
        try:
            from datetime import datetime
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return f"{dt.year} Q{(dt.month - 1) // 3 + 1}"
        except Exception:
            return s[:7]
    elif grain == "month":
        return s[:7]
    return s[:10]