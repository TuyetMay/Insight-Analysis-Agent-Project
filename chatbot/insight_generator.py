"""
chatbot/insight_generator.py
Generates a short analytical insight for a query result.
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
                return f"\n\n---\n💡 **Insight:** {text}"
        lines = self._rule_insight(plan, df)
        if lines:
            return "\n\n---\n💡 **Insight:** " + "  \n".join(lines)
        return ""

    # ── LLM insight ───────────────────────────────────────────

    def _llm_insight(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        summary   = self._data_summary(plan, df)
        intent    = plan.get("intent", "kpi_value")
        metrics   = plan.get("metrics", ["sales"])
        breakdown = plan.get("breakdown_by")
        grain     = plan.get("time_grain", "none")
        cp        = plan.get("compare_period")

        hint = {
            "kpi_rank":    "Comment on the leader, notable gaps, concentration, or surprising entries.",
            "kpi_value":   f"Comment on the distribution across {breakdown}: dominant player, imbalance.",
            "kpi_trend":   f"Comment on the trajectory by {grain}: peak, momentum, seasonality.",
            "kpi_compare": f"Comment on the {metrics[0]} change ({cp}): magnitude and business signal.",
        }.get(intent, "Provide a brief analytical observation.")

        prompt = f"""You are a concise business analyst. Write an insight from ONLY the numbers below.

=== RESULT ===
{summary}

=== TASK ===
{hint}

Rules:
- Exactly 2-3 sentences. No more.
- Use actual figures/names from RESULT (do not invent).
- No table reprint, no bullets, no headings.
- Avoid openers like "The data shows" or "Overall".
- Bold (**...**) the most important 1-3 names/numbers.
- If insufficient signal: write "Not enough signal in this slice to draw a strong conclusion."

Plain text only:"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.7, max_output_tokens=150),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if len(text) > 20:
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
            sdf = df.sort_values(by=m0, ascending=False).reset_index(drop=True)
            n = len(sdf)
            if n >= 1:
                insights.append(f"**{sdf.iloc[0]['breakdown']}** leads with {self._fv(float(sdf.iloc[0][m0]), m0)}.")
            if n >= 2:
                tv  = float(sdf.iloc[0][m0])
                sv  = float(sdf.iloc[1][m0])
                gap = abs(tv - sv) / abs(tv) * 100 if tv else 0
                if gap > 50:
                    insights.append(f"Gap to **{sdf.iloc[1]['breakdown']}** ({self._fv(sv, m0)}) is significant ({gap:.0f}%).")
                else:
                    insights.append(f"**{sdf.iloc[1]['breakdown']}** follows at {self._fv(sv, m0)}.")
            if n >= 3:
                total = float(sdf[m0].sum())
                top2  = float(sdf.head(2)[m0].sum())
                if total > 0 and top2 / total >= 0.6:
                    top2_names = " & ".join(str(sdf.iloc[i]["breakdown"]) for i in range(2))
                    insights.append(f"**{top2_names}** together account for {top2/total*100:.0f}% of total.")
            if m0 in {"profit", "profit_margin"}:
                negatives = sdf[sdf[m0] < 0]
                if not negatives.empty:
                    names = ", ".join(f"**{r['breakdown']}**" for _, r in negatives.iterrows())
                    verb  = "is" if len(negatives) == 1 else "are"
                    insights.append(f"⚠️ {names} {verb} at a loss.")

        elif intent == "kpi_trend" and "period" in df.columns and m0 in df.columns:
            sdf = df.sort_values("period").reset_index(drop=True)
            if len(sdf) >= 2:
                first, last = float(sdf.iloc[0][m0]), float(sdf.iloc[-1][m0])
                if first:
                    chg = (last - first) / abs(first) * 100
                    word = "grown" if chg > 0 else "declined"
                    insights.append(f"Overall {word} by **{abs(chg):.1f}%** over the period.")
            peak = sdf.loc[sdf[m0].idxmax()]
            insights.append(f"Peak: **{str(peak['period'])[:7]}** at {self._fv(float(peak[m0]), m0)}.")

        elif intent == "kpi_compare" and "current" in df.columns:
            row  = df.iloc[0]
            cur, prev = float(row["current"]), float(row["previous"])
            m = str(row.get("metric", m0))
            if prev:
                chg = (cur - prev) / abs(prev) * 100
                if abs(chg) >= 20:
                    word = "strong growth" if chg > 0 else "sharp decline"
                    insights.append(f"**{word}** of {abs(chg):.1f}% — worth investigating drivers.")
                elif abs(chg) < 5:
                    insights.append("Performance is **stable** with minimal period-over-period change.")
                else:
                    d = "ahead of" if chg > 0 else "behind"
                    insights.append(f"Current period is **{d}** the prior by {abs(chg):.1f}%.")

        return insights

    # ── Data summary for LLM ──────────────────────────────────

    def _data_summary(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        intent  = plan.get("intent")
        metrics = plan.get("metrics", ["sales"])
        m0      = plan.get("order_by") or metrics[0]
        lines: List[str] = []

        if intent == "kpi_compare" and "current" in df.columns:
            row = df.iloc[0]
            m   = str(row.get("metric", m0))
            cur, prev = float(row["current"]), float(row["previous"])
            chg = ((cur - prev) / abs(prev) * 100) if prev else None
            lines += [
                f"Metric: {m}",
                f"Current ({row['current_start']} – {row['current_end']}): {self._fv(cur, m)}",
                f"Previous ({row['prev_start']} – {row['prev_end']}): {self._fv(prev, m)}",
                f"Change: {chg:+.1f}%" if chg is not None else "Change: n/a",
            ]
        elif "breakdown" in df.columns:
            sdf   = df.sort_values(by=m0, ascending=False).head(15).reset_index(drop=True)
            total = float(sdf[m0].sum()) if m0 in sdf.columns else 0
            lines.append(f"Breakdown: {plan.get('breakdown_by')} | Metric: {m0} | Total: {self._fv(total, m0)}")
            for i, (_, r) in enumerate(sdf.iterrows(), 1):
                vals = "  ".join(f"{c}={self._fv(float(r[c]), c)}" for c in metrics if c in r)
                lines.append(f"  {i}. {r['breakdown']}: {vals}")
        elif "period" in df.columns:
            for _, r in df.sort_values("period").tail(24).iterrows():
                p    = str(r.get("period", ""))[:7]
                vals = "  ".join(f"{c}={self._fv(float(r[c]), c)}" for c in metrics if c in r)
                lines.append(f"  {p}: {vals}")
        else:
            r0 = df.iloc[0]
            for m in metrics:
                if m in r0:
                    lines.append(f"{m}: {self._fv(float(r0[m]), m)}")

        return "\n".join(lines)

    @staticmethod
    def _fv(v: float, metric: str) -> str:
        if metric in {"sales", "profit"}:  return f"${v:,.0f}"
        if metric == "profit_margin":      return f"{v:.1f}%"
        return f"{int(v):,}"
