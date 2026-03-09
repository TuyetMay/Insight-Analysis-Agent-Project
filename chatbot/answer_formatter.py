"""
chatbot/answer_formatter.py
Converts a validated plan + result DataFrame into a nicely formatted Markdown string.
Pure presentation logic — no LLM, no DB calls.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


_METRIC_LABELS = {
    "orders":        "Total Orders",
    "sales":         "Total Sales",
    "profit":        "Total Profit",
    "profit_margin": "Profit Margin",
}
_METRIC_OPENERS = {
    "sales":         "Total sales came in at",
    "profit":        "Total profit reached",
    "orders":        "The total number of orders was",
    "profit_margin": "The profit margin stood at",
}
_CP_LABELS = {
    "yoy":         "year-over-year",
    "mom":         "month-over-month",
    "prev_period": "vs the previous period",
}


class AnswerFormatter:
    """Stateless: all context is passed in per call."""

    # ── Public ────────────────────────────────────────────────

    def format(self, plan: Dict[str, Any], df: pd.DataFrame,
               insight: str = "") -> str:
        if plan["intent"] == "clarify":
            return plan.get("clarifying_question", "Could you clarify your question?")
        if df is None or df.empty:
            ctx = self.natural_context(plan)
            return f"No data found for the selected filters." + (f"\n*{ctx}*" if ctx else "")

        intent    = plan["intent"]
        metrics   = plan["metrics"]
        grain     = plan["time_grain"]
        breakdown = plan.get("breakdown_by")
        ctx_line  = f"\n*{self.natural_context(plan)}*" if self.natural_context(plan) else ""

        if intent == "kpi_compare":
            body = self._format_compare(plan, df)
        elif intent == "kpi_value" and grain == "none":
            body = self._format_value(plan, df, breakdown, metrics, ctx_line)
        elif intent == "kpi_trend":
            body = self._format_trend(plan, df, metrics, grain, breakdown, ctx_line)
        elif intent == "kpi_rank":
            body = self._format_rank(plan, df, metrics, ctx_line)
        else:
            body = f"Done.{ctx_line}"

        return body + insight

    def natural_context(self, plan: Dict[str, Any]) -> str:
        """e.g. 'Jan 2014 – Dec 2017  ·  West, East'"""
        sd = plan.get("start_date", "")
        ed = plan.get("end_date", "")

        def fmt_date(d: str) -> str:
            try:
                return datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y")
            except Exception:
                return d

        date_part = f"{fmt_date(sd)} – {fmt_date(ed)}" if (sd and ed) else ""
        parts = [date_part] if date_part else []

        # Only show filter context that differs from "all selected"
        # (we don't track dashboard-wide selections here, so show non-empty filters)
        f = plan.get("filters") or {}
        for key in ("region", "segment", "category"):
            vals = f.get(key) or []
            if vals:
                parts.append(", ".join(vals))

        return "  ·  ".join(parts)

    # ── Intent-specific formatters ────────────────────────────

    def _format_compare(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        row  = df.iloc[0]
        m    = row["metric"]
        cur  = float(row["current"])
        prev = float(row["previous"])
        chg  = ((cur - prev) / abs(prev) * 100.0) if prev else None

        def fv(v: float) -> str:
            if m in {"sales", "profit"}:   return f"${v:,.0f}"
            if m == "profit_margin":        return f"{v:.2f}%"
            return f"{int(v):,}"

        cp_label  = _CP_LABELS.get(plan.get("compare_period", ""), "vs previous")
        delta_s   = f"{chg:+.1f}%" if chg is not None else "n/a"
        direction = "up" if (chg or 0) >= 0 else "down"
        ctx_line  = f"\n*{self.natural_context(plan)}*" if self.natural_context(plan) else ""

        return "\n".join([
            f"Here's the **{_METRIC_LABELS[m]}** comparison ({cp_label}):{ctx_line}",
            "",
            f"- **Current** ({row['current_start']} – {row['current_end']}): {fv(cur)}",
            f"- **Previous** ({row['prev_start']} – {row['prev_end']}): {fv(prev)}",
            f"- **Change:** {delta_s} ({direction})",
        ])

    def _format_value(self, plan: Dict[str, Any], df: pd.DataFrame,
                      breakdown: Any, metrics: List[str], ctx_line: str) -> str:
        if breakdown:
            m0   = plan.get("order_by") or metrics[0]
            dim_label = breakdown.replace("_", " ").title()
            lines = [f"Here's how **{_METRIC_LABELS[metrics[0]]}** breaks down by **{dim_label}**:{ctx_line}", ""]
            for rank, (_, r) in enumerate(df.sort_values(by=m0, ascending=False).head(20).iterrows(), 1):
                b    = r.get("breakdown", "—")
                vals = " | ".join(
                    f"{_METRIC_LABELS[m]}: {self._fv(float(r[m]), m)}"
                    for m in metrics if m in r
                )
                lines.append(f"{rank}. **{b}** — {vals}")
            return "\n".join(lines)

        # Single aggregate
        r0 = df.iloc[0]
        if len(metrics) == 1:
            m  = metrics[0]
            vs = self._fv(float(r0[m]), m)
            return f"{_METRIC_OPENERS.get(m, f'{_METRIC_LABELS[m]} is')} **{vs}**.{ctx_line}"

        lines = [f"Here's a summary of the selected metrics:{ctx_line}", ""]
        for m in metrics:
            lines.append(f"- **{_METRIC_LABELS[m]}:** {self._fv(float(r0[m]), m)}")
        return "\n".join(lines)

    def _format_trend(self, plan: Dict[str, Any], df: pd.DataFrame,
                      metrics: List[str], grain: str, breakdown: Any, ctx_line: str) -> str:
        grain_label = {"week": "week", "month": "month", "quarter": "quarter", "year": "year"}.get(grain, grain)
        metric_str  = " & ".join(_METRIC_LABELS[m] for m in metrics)
        lines = [f"Here's the **{metric_str}** trend by {grain_label}:{ctx_line}", ""]

        show = df.copy()
        if "period" in show.columns:
            show = show.sort_values("period").tail(24)
        if breakdown and "breakdown" in show.columns:
            m0   = metrics[0]
            top5 = show.groupby("breakdown")[m0].sum().sort_values(ascending=False).head(5).index.tolist()
            show = show[show["breakdown"].isin(top5)]

        for _, r in show.iterrows():
            p    = str(r.get("period", ""))[:7]
            vals = " / ".join(self._fv(float(r[m]), m) for m in metrics if m in r)
            prefix = f"{p} | {r.get('breakdown')}" if breakdown else p
            lines.append(f"- **{prefix}:** {vals}")

        if len(lines) > 32:
            lines = lines[:32] + ["- *(truncated — showing most recent 30 periods)*"]
        return "\n".join(lines)

    def _format_rank(self, plan: Dict[str, Any], df: pd.DataFrame,
                     metrics: List[str], ctx_line: str) -> str:
        m0        = metrics[0]
        breakdown = plan.get("breakdown_by") or "item"
        dim_label = breakdown.replace("_", " ").title()
        lines = [
            f"Here are the **top {plan['top_k']} {dim_label}s** ranked by {_METRIC_LABELS[m0]}:{ctx_line}",
            "",
        ]
        for i, (_, r) in enumerate(df.head(plan["top_k"]).iterrows(), 1):
            b  = r.get("breakdown", "—")
            vs = self._fv(float(r[m0]), m0)
            lines.append(f"{i}. **{b}** — {vs}")
        return "\n".join(lines)

    # ── Value formatter ───────────────────────────────────────

    @staticmethod
    def _fv(v: float, metric: str) -> str:
        if metric in {"sales", "profit"}:  return f"\\${v:,.0f}"   # escape $
        if metric == "profit_margin":      return f"{v:.2f}%"
        return f"{int(v):,}"