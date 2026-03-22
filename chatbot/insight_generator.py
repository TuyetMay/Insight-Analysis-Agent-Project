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
                (
                    # Cross-breakdown case: 2 dimensions
                    f"Analyse this {breakdown} × {plan.get('secondary_breakdown')} cross-breakdown deeply. "
                    "Cover ALL of these points:\n"
                    "- **Overall leader**: which primary dimension leads and by how much ($ and %).\n"
                    "- **Category dominance per dimension**: for each primary value, which secondary "
                    "dimension leads and what % share does it hold.\n"
                    "- **Pattern**: is the category mix consistent across all primary values, or does "
                    "any primary value have a notably different distribution?\n"
                    "- **Outlier**: identify any primary+secondary combination that is unusually high "
                    "or low compared to peers.\n"
                    "- **Action**: 1 concrete recommendation per major finding.\n"
                    "Write 6–9 bullet points. Every bullet must contain at least one number. "
                    "Each bullet must end with a period."
                ) if plan.get("secondary_breakdown") else (
                    f"Analyse the breakdown across {breakdown or 'all data'}. "
                    "Cover exactly these 3 points in 4–5 bullet points total:\n"
                    "- **Leader**: who leads, their exact value, and their % share of total.\n"
                    "- **Spread**: gap between top and bottom ($ and %), is it concentrated or balanced?\n"
                    "- **Action**: one concrete business recommendation based on this distribution.\n"
                    "Max 5 bullet points. Each bullet must end with a period. No open-ended sentences."
                )
            ),
            "kpi_trend": (
                f"Analyse this {grain}-level YoY trend concisely. Cover exactly 3 things:\n"
                "1. **Overall change**: Total % and absolute $ from first to last period.\n"
                "2. **Period-by-period**: For each transition, state % change, classify it "
                "(🚀 ≥20% / 📈 5-19% / ➡️ flat / 📉 decline), and one short phrase explaining it.\n"
                "3. **Key pattern**: One sentence on the dominant trend (e.g. dip then recovery, "
                "sustained growth, etc.) and what it implies for the business.\n\n"
                "Do NOT include: CAGR, forecasts, peak/trough detail, growth concentration %, "
                "or root cause hypotheses unless the data strongly demands it.\n"
                "Max 6–8 sentences total. Every sentence must have a number."
            ),
            "kpi_compare": (
                f"Analyse this {cp} period comparison deeply:\n"
                "- State the magnitude of change and whether it's significant.\n"
                "- Is this a one-time spike/dip or part of a sustained trend?\n"
                "- What operational or market factors could explain this shift?\n"
                "- What should the business monitor or act on next?"
            ),
        }.get(intent, "Provide a thorough analytical observation using the data.")

        # ── Token budget scales with intent ───────────────────
        max_tokens = {
            "kpi_trend":   2800,   
            "kpi_compare": 3000,
            "kpi_rank":    1500,
            "kpi_detail":  1800,
            "kpi_value":   1500,
        }.get(intent, 1500)

        num_sentences = {
            "kpi_trend":  "12–20 sentences OR 10–15 detailed bullet points",
            "kpi_detail": "8–12 sentences",
        }.get(intent, "6–10 sentences OR 5–7 bullet points")

        prompt = f"""You are a senior business analyst writing a performance commentary for an executive audience.
            Using ONLY the numbers in the DATA section below, write a **deep, comprehensive analytical insight**.

            === DATA ===
            {summary}

            === ANALYSIS TASK ===
            {intent_guidance}

            === WRITING RULES ===
            - Write {num_sentences}. Be SUBSTANTIVE and EXHAUSTIVE — cover every angle described above.
            - Lead with the single most important finding with a specific number.
            - Use **bold** for ALL critical numbers, percentages, and period names.
            - Every sentence must contain at least one number from the DATA section.
            - For trend analysis: address EACH period transition individually — do NOT summarise multiple transitions into one sentence.
            - Include CAGR if multiple years of data are present.
            - Include a 2-period forward projection in the final section.
            - Do NOT invent figures that are not in the DATA section.
            - Do NOT use filler openers like "The data shows", "Overall", "In conclusion", "Notably".
            - Use clear, direct business language. No jargon.
            - Format: use numbered sections with headers for kpi_trend, bullet points for others.

            Write the complete, full-length insight now (do not truncate):"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.55,
                    max_output_tokens=max_tokens,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if len(text) > 30:
                return self._trim_to_complete_sentence(text)
        except Exception:
            pass
        return ""

    # ── Rule-based insight ────────────────────────────────────

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

            grand_total = float(df[m0].sum())

            # ── 1. Region tổng cao nhất ───────────────────────────────
            region_totals = df.groupby("breakdown2")[m0].sum().sort_values(ascending=False)
            top_region    = region_totals.index[0]
            top_region_v  = float(region_totals.iloc[0])
            bot_region    = region_totals.index[-1]
            bot_region_v  = float(region_totals.iloc[-1])
            gap_pct       = (top_region_v - bot_region_v) / top_region_v * 100 if top_region_v else 0
            insights.append(
                f"**{top_region}** is the strongest region at {self._fv(top_region_v, m0)} "
                f"({top_region_v/grand_total*100:.0f}% of total), "
                f"**{gap_pct:.0f}%** ahead of the weakest region "
                f"**{bot_region}** ({self._fv(bot_region_v, m0)})."
            )

            # ── 2. Category tổng cao nhất + regional spread ───────────
            cat_totals = df.groupby("breakdown")[m0].sum().sort_values(ascending=False)
            top_cat    = cat_totals.index[0]
            top_cat_v  = float(cat_totals.iloc[0])
            # Tìm region dẫn đầu cho top category
            top_cat_df  = df[df["breakdown"] == top_cat].sort_values(m0, ascending=False)
            top_cat_reg = top_cat_df.iloc[0].get("breakdown2", "—")
            top_cat_reg_v = float(top_cat_df.iloc[0][m0])
            insights.append(
                f"**{top_cat}** leads all categories at {self._fv(top_cat_v, m0)} "
                f"({top_cat_v/grand_total*100:.0f}% of total), "
                f"with **{top_cat_reg}** as its strongest market "
                f"({self._fv(top_cat_reg_v, m0)})."
            )

            # ── 3. Region phụ thuộc 1 category (concentration risk) ───
            concentration_lines = []
            for region_val in region_totals.index:
                rdf      = df[df["breakdown2"] == region_val]
                r_total  = float(rdf[m0].sum())
                top_row  = rdf.sort_values(m0, ascending=False).iloc[0]
                top_share = float(top_row[m0]) / r_total * 100 if r_total else 0
                if top_share >= 40:
                    concentration_lines.append(
                        f"**{region_val}** relies heavily on "
                        f"**{top_row.get('breakdown','—')}** "
                        f"({top_share:.0f}% of region sales)"
                    )
            if concentration_lines:
                insights.append(
                    "⚠️ **Concentration risk:** " + "; ".join(concentration_lines) +
                    " — a decline in this category would disproportionately impact the region."
                )

            # ── 4. Category phân bổ đều nhất giữa các region ─────────
            cat_std = {}
            for cat_val in cat_totals.index:
                cdf    = df[df["breakdown"] == cat_val]
                shares = []
                for region_val in region_totals.index:
                    r_total = float(region_totals[region_val])
                    cell    = cdf[cdf["breakdown2"] == region_val][m0].sum()
                    shares.append(float(cell) / r_total * 100 if r_total else 0)
                import statistics
                cat_std[cat_val] = statistics.stdev(shares) if len(shares) > 1 else 0

            most_even = min(cat_std, key=cat_std.get)
            most_conc = max(cat_std, key=cat_std.get)
            insights.append(
                f"**{most_even}** has the most balanced regional distribution "
                f"(std dev {cat_std[most_even]:.1f}pp across regions), "
                f"while **{most_conc}** is most unevenly spread "
                f"(std dev {cat_std[most_conc]:.1f}pp) — "
                f"suggesting {most_conc} growth is concentrated in specific markets."
            )

            return insights


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

            grain = plan.get("time_grain", "none")

            # ── 1. Overall trajectory ─────────────────────────
            abs_change = abs(last_v - first_v)
            insights.append(
                f"Over the full **{n}-period** span, **{_label(m0)}** has {word} by "
                f"**{abs(chg):.1f}%** — from {self._fv(first_v, m0)} to {self._fv(last_v, m0)} "
                f"(an absolute change of **{self._fv(abs_change, m0)}**)."
            )

            # ── 3. Period-by-period breakdown ─────────────────
            if grain in ("year", "quarter", "month") and n <= 24:
                period_lines = []
                best_chg, best_idx = -999, 1
                worst_chg, worst_idx = 999, 1

                for i in range(1, n):
                    prev_p = _fmt_period(sdf.iloc[i - 1].get("period", ""), grain)
                    curr_p = _fmt_period(sdf.iloc[i].get("period", ""), grain)
                    prev_v = float(sdf.iloc[i - 1][m0])
                    curr_v = float(sdf.iloc[i][m0])
                    pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                    abs_d  = abs(curr_v - prev_v)

                    if pct > best_chg:
                        best_chg, best_idx = pct, i
                    if pct < worst_chg:
                        worst_chg, worst_idx = pct, i

                    tag = (
                        "🚀 Strong growth" if pct >= 20
                        else "📈 Moderate growth" if pct >= 5
                        else "📉 Decline" if pct < 0
                        else "➡️ Flat"
                    )
                    period_lines.append(
                        f"  - **{prev_p} → {curr_p}:** {pct:+.1f}% "
                        f"({self._fv(prev_v, m0)} → {self._fv(curr_v, m0)}, "
                        f"Δ {self._fv(abs_d, m0)}) — {tag}"
                    )

                insights.append(
                    "**Period-by-period breakdown:**\n" + "\n".join(period_lines)
                )

                # Highlight best and worst transitions
                if n >= 3:
                    bp = _fmt_period(sdf.iloc[best_idx].get("period", ""), grain)
                    wp = _fmt_period(sdf.iloc[worst_idx].get("period", ""), grain)
                    insights.append(
                        f"The **strongest single-period jump** was into **{bp}** "
                        f"({best_chg:+.1f}%), while the **weakest** was into **{wp}** "
                        f"({worst_chg:+.1f}%)."
                    )

            # ── 4. Peak and trough ────────────────────────────
            peak   = sdf.loc[sdf[m0].idxmax()]
            trough = sdf.loc[sdf[m0].idxmin()]
            peak_p   = _fmt_period(peak.get("period", ""), grain)
            trough_p = _fmt_period(trough.get("period", ""), grain)
            range_v  = abs(float(peak[m0]) - float(trough[m0]))
            swing_pct = range_v / abs(float(peak[m0])) * 100 if float(peak[m0]) else 0
            insights.append(
                f"🏆 **Peak:** {peak_p} at {self._fv(float(peak[m0]), m0)} | "
                f"📉 **Trough:** {trough_p} at {self._fv(float(trough[m0]), m0)} — "
                f"a range of **{self._fv(range_v, m0)}** ({swing_pct:.0f}% swing from peak to trough)."
            )

            

            # ── 6. Recent momentum ────────────────────────────
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
                    r1 = _fmt_period(sdf.iloc[-2].get("period", ""), grain)
                    r2 = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
                    p1 = _fmt_period(sdf.iloc[-4].get("period", ""), grain)
                    p2 = _fmt_period(sdf.iloc[-3].get("period", ""), grain)
                    insights.append(
                        f"**Recent momentum is {direction}** — the last 2 periods "
                        f"(**{r1}** & **{r2}**) averaged {self._fv(recent_avg, m0)}, "
                        f"vs {self._fv(prior_avg, m0)} in the prior 2 (**{p1}** & **{p2}**) "
                        f"({momentum:+.1f}%). "
                        f"{'Sustained momentum suggests continued growth if market conditions hold.' if momentum > 5 else 'Slowing momentum warrants a closer look at pipeline and demand signals.' if momentum < -5 else 'Steady growth — no dramatic acceleration or deceleration detected.'}"
                    )

            # ── 7. Forward projection ─────────────────────────
            if grain == "year" and n >= 2 and first_v > 0:
                years = n - 1
                cagr  = ((last_v / first_v) ** (1 / years) - 1) if first_v > 0 else 0
                proj_base = last_v * (1 + cagr)
                proj_high = last_v * (1 + cagr * 1.3)
                proj_low  = last_v * (1 + cagr * 0.7)
                last_year = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
                try:
                    next_year = str(int(last_year) + 1)
                    yr_after  = str(int(last_year) + 2)
                except Exception:
                    next_year = "next period"
                    yr_after  = "the period after"
                insights.append(
                    f"📊 **Forward projection:** Applying the **{cagr*100:.1f}% CAGR** to {last_year}'s "
                    f"{self._fv(last_v, m0)}, the base-case forecast for **{next_year}** is "
                    f"**{self._fv(proj_base, m0)}** "
                    f"(range: {self._fv(proj_low, m0)} low → {self._fv(proj_high, m0)} high). "
                    f"By **{yr_after}**, the base case reaches **{self._fv(proj_base * (1 + cagr), m0)}**."
                )

        elif intent == "kpi_compare" and "current" in df.columns:
            row  = df.iloc[0]
            cur  = float(row["current"])
            prev = float(row["previous"])
            if prev == 0:
                return []
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
            # ── Rich trend context for LLM ──────────────────
            all_rows = df.sort_values("period").reset_index(drop=True)
            n = len(all_rows)
            lines.append(f"Time grain: {grain} | Metric: {m0} | Total periods: {n}")

            if n >= 2:
                first_v = float(all_rows.iloc[0][m0])
                last_v  = float(all_rows.iloc[-1][m0])
                total_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
                abs_chg   = abs(last_v - first_v)
                lines.append(f"Total change first→last: {total_chg:+.1f}% ({self._fv(first_v, m0)} → {self._fv(last_v, m0)}, absolute Δ={self._fv(abs_chg, m0)})")
                if grain == "year" and n >= 2:
                    years = n - 1
                    cagr  = ((last_v / first_v) ** (1 / years) - 1) * 100 if first_v > 0 else 0
                    lines.append(f"CAGR over {years} year(s): {cagr:.2f}%")

            lines.append("Period-by-period data:")
            for i, (_, r) in enumerate(all_rows.iterrows()):
                p    = _fmt_period(r.get("period", ""), grain)
                vals = "  ".join(f"{c}={self._fv(float(r[c]), c)}" for c in metrics if c in r)
                if i > 0:
                    prev_v = float(all_rows.iloc[i - 1][m0])
                    curr_v = float(r[m0])
                    pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                    abs_d  = abs(curr_v - prev_v)
                    lines.append(f"  {p}: {vals}  (Δ vs prior: {pct:+.1f}%, abs Δ={self._fv(abs_d, m0)})")
                else:
                    lines.append(f"  {p}: {vals}  (baseline)")

            peak    = all_rows.loc[all_rows[m0].idxmax()]
            trough  = all_rows.loc[all_rows[m0].idxmin()]
            lines.append(f"Peak period: {_fmt_period(peak.get('period',''), grain)} at {self._fv(float(peak[m0]), m0)}")
            lines.append(f"Trough period: {_fmt_period(trough.get('period',''), grain)} at {self._fv(float(trough[m0]), m0)}")
            lines.append(f"Peak-to-trough range: {self._fv(abs(float(peak[m0]) - float(trough[m0])), m0)}")

            # Recent momentum
            if n >= 4:
                recent_avg = float(all_rows.tail(2)[m0].mean())
                prior_avg  = float(all_rows.iloc[-4:-2][m0].mean())
                momentum   = (recent_avg - prior_avg) / abs(prior_avg) * 100 if prior_avg else 0
                lines.append(f"Recent momentum (last 2 vs prior 2): {momentum:+.1f}% (recent avg={self._fv(recent_avg, m0)}, prior avg={self._fv(prior_avg, m0)})")

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
    @staticmethod
    def _trim_to_complete_sentence(text: str) -> str:
        """
        Nếu text kết thúc giữa câu (không có dấu . ! ? sau từ cuối),
        cắt về dấu kết thúc câu cuối cùng tìm được.
        """
        import re
        text = text.strip()
        # Nếu đã kết thúc bằng dấu câu → OK
        if text and text[-1] in ".!?":
            return text
        # Tìm vị trí kết thúc câu cuối cùng
        # Xét cả . ! ? kể cả khi sau đó có ký tự như ** hoặc )
        matches = list(re.finditer(r'[.!?](\*{0,2}[\)\s]|$)', text))
        if matches:
            last = matches[-1]
            return text[:last.end()].strip()
        # Không tìm được → trả về nguyên
        return text


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

