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

        intent    = plan["intent"]

        # kpi_detail has its own empty-check inside _format_detail
        if intent != "kpi_detail" and (df is None or df.empty):
            ctx = self.natural_context(plan)
            return f"No data found for the selected filters." + (f"\n*{ctx}*" if ctx else "")
        metrics   = plan["metrics"]
        grain     = plan["time_grain"]
        breakdown = plan.get("breakdown_by")
        ctx_line  = f"\n*{self.natural_context(plan)}*" if self.natural_context(plan) else ""

        if intent == "kpi_detail":
            if df is None or df.empty:
                ctx = self.natural_context(plan)
                return (
                    "No loss-making sub-categories found for the selected filters."
                    + (f"\n*{ctx}*" if ctx else "")
                    + "\n\nAll sub-categories are profitable in this period. 🎉"
                )
            body = self._format_detail(plan, df, ctx_line)
        elif intent == "kpi_compare":
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
        sd = plan.get("start_date", "")
        ed = plan.get("end_date", "")

        def fmt_date(d: str) -> str:
            try:
                return datetime.strptime(d, "%Y-%m-%d").strftime("%b %Y")
            except Exception:
                return d

        date_part = f"{fmt_date(sd)} – {fmt_date(ed)}" if (sd and ed) else ""
        parts = [date_part] if date_part else []

        f = plan.get("filters") or {}
        for key in ("region", "segment", "category"):
            vals = f.get(key) or []
            if vals:
                parts.append(", ".join(vals))

        return "  ·  ".join(parts)

    # ── kpi_detail: negative profit drill-down ────────────────

    def _format_detail(self, plan: Dict[str, Any], df: pd.DataFrame,
                       ctx_line: str) -> str:
        condition  = plan.get("condition", "profit_negative")
        breakdown  = plan.get("breakdown_by") or "sub_category"
        dim_label  = breakdown.replace("_", " ").title()
        sample_df  = df.attrs.get("sample_orders", pd.DataFrame())

        cond_label = {
            "profit_negative": "negative profit (loss-making)",
            "loss_orders":     "negative profit (loss-making)",
            "high_discount":   "high discount (>20%)",
            "profit_positive": "positive profit",
        }.get(condition, condition.replace("_", " "))

        total_orders  = int(df["orders"].sum()) if "orders" in df.columns else 0
        total_loss    = float(df["profit"].sum()) if "profit" in df.columns else 0
        total_sales   = float(df["sales"].sum()) if "sales" in df.columns else 0

        lines = [
            f"Found **{total_orders:,} orders** with {cond_label} across **{len(df)} {dim_label}s**,{ctx_line}",
            f"generating a total loss of **\\${abs(total_loss):,.0f}** on **\\${total_sales:,.0f}** in revenue.",
            "",
            f"**Breakdown by {dim_label} — worst performers:**",
            "",
        ]

        for rank, (_, r) in enumerate(df.iterrows(), 1):
            name     = r.get("breakdown", "—")
            cat      = r.get("category", "")
            profit   = float(r.get("profit", 0))
            sales    = float(r.get("sales", 0))
            orders   = int(r.get("orders", 0))
            disc     = float(r.get("avg_discount_pct", 0))
            margin   = float(r.get("profit_margin", 0))
            cat_str  = f" *(in {cat})*" if cat and cat.lower() != name.lower() else ""

            # Reason indicator
            reason = ""
            if disc >= 30:
                reason = f"  ⚠️ *avg discount {disc:.0f}% — heavy discounting eroding margin*"
            elif disc >= 15:
                reason = f"  ⚠️ *avg discount {disc:.0f}% — discounting likely driving losses*"
            elif margin < -20:
                reason = f"  ⚠️ *margin {margin:.1f}% — structural cost or pricing issue*"

            lines.append(
                f"{rank}. **{name}**{cat_str} — Loss: **\\${abs(profit):,.0f}** "
                f"| Sales: \\${sales:,.0f} | Orders: {orders:,} | "
                f"Avg Discount: {disc:.0f}%{reason}"
            )

        # ── Sample of worst individual orders ─────────────────
        if not sample_df.empty:
            lines += [
                "",
                "**Sample of worst individual orders:**",
                "",
            ]
            for _, r in sample_df.head(8).iterrows():
                oid     = r.get("order_id", "—")
                odate   = str(r.get("order_date", ""))[:10]
                prod    = str(r.get("product_name", "—"))[:45]
                profit  = float(r.get("profit", 0))
                disc    = float(r.get("discount", 0)) * 100
                lines.append(
                    f"- `{oid}` ({odate}) — *{prod}* — "
                    f"Loss: **\\${abs(profit):,.0f}** | Discount: {disc:.0f}%"
                )

        return "\n".join(lines)

    # ── Intent-specific formatters ────────────────────────────

    def _format_compare(self, plan: Dict[str, Any], df: pd.DataFrame) -> str:
        metric    = plan["metrics"][0]
        cp_label  = _CP_LABELS.get(plan.get("compare_period", ""), "vs previous")
        ctx_line  = f"\n*{self.natural_context(plan)}*" if self.natural_context(plan) else ""

        # ── Per-dimension comparison (e.g. by region) ──────────
        if "breakdown" in df.columns:
            dim_label = (plan.get("breakdown_by") or "dimension").replace("_", " ").title()
            first = df.iloc[0]
            header = (
                f"Here's **{_METRIC_LABELS.get(metric, metric)}** by **{dim_label}** "
                f"— {cp_label}:{ctx_line}\n"
                f"*(Current: {first['current_start']} – {first['current_end']} | "
                f"Previous: {first['prev_start']} – {first['prev_end']})*\n"
            )
            lines = [header]
            for _, r in df.iterrows():
                cur  = float(r["current"])
                prev = float(r["previous"])
                chg  = r.get("change_pct")
                if chg is not None:
                    arrow = "📉" if chg < 0 else ("📈" if chg > 5 else "➡️")
                    delta = f"{chg:+.1f}%"
                else:
                    arrow, delta = "➡️", "n/a"
                flag = " ⚠️ **UNDERPERFORMING**" if (chg is not None and chg < 0) else ""
                lines.append(
                    f"- **{r['breakdown']}** — "
                    f"Current: {self._fv(cur, metric)} | "
                    f"Previous: {self._fv(prev, metric)} | "
                    f"{arrow} {delta}{flag}"
                )
            return "\n".join(lines)

        # ── Original single-aggregate comparison ───────────────
        row  = df.iloc[0]
        m    = row["metric"]
        cur  = float(row["current"])
        prev = float(row["previous"])
        chg  = ((cur - prev) / abs(prev) * 100.0) if prev else None

        def fv(v: float) -> str:
            if m in {"sales", "profit"}:  return f"${v:,.0f}"
            if m == "profit_margin":       return f"{v:.2f}%"
            return f"{int(v):,}"

        delta_s   = f"{chg:+.1f}%" if chg is not None else "n/a"
        direction = "up" if (chg or 0) >= 0 else "down"

        return "\n".join([
            f"Here's the **{_METRIC_LABELS.get(m, m)}** comparison ({cp_label}):{ctx_line}",
            "",
            f"- **Current** ({row['current_start']} – {row['current_end']}): {fv(cur)}",
            f"- **Previous** ({row['prev_start']} – {row['prev_end']}): {fv(prev)}",
            f"- **Change:** {delta_s} ({direction})",
        ])

    def _format_value(self, plan: Dict[str, Any], df: pd.DataFrame,
                      breakdown: Any, metrics: List[str], ctx_line: str) -> str:
        if breakdown:
            m0        = plan.get("order_by") or metrics[0]
            dim_label = breakdown.replace("_", " ").title()
            show_extremes = plan.get("show_extremes", False)

            sorted_df = df.sort_values(by=m0, ascending=False)

            if show_extremes and len(sorted_df) == 1:
                # Only one entry — sidebar filter is restricting results
                only_name = sorted_df.iloc[0].get("breakdown", "—")
                only_val  = self._fv(float(sorted_df.iloc[0][m0]), m0)
                return (
                    f"Only **{only_name}** is shown ({only_val}) because the sidebar "
                    f"**{dim_label} filter** is set to a single value.\n\n"
                    f"To compare all {dim_label}s, please select all options in the "
                    f"**🔍 Filters → {dim_label}** multiselect in the sidebar, then ask again."
                )

            if show_extremes and len(sorted_df) >= 2:
                # Show highest + lowest clearly
                metric_label = _METRIC_LABELS.get(metrics[0], metrics[0])
                lines = [
                    f"Here's the **highest and lowest {metric_label}** by **{dim_label}**:{ctx_line}",
                    "",
                    f"🏆 **Highest {dim_label}:**",
                ]
                # Top entries (up to 3 if many regions, else just 1)
                n_top = min(3, max(1, len(sorted_df) // 3))
                for rank, (_, r) in enumerate(sorted_df.head(n_top).iterrows(), 1):
                    b    = r.get("breakdown", "—")
                    vals = " | ".join(
                        f"{_METRIC_LABELS.get(m, m)}: {self._fv(float(r[m]), m)}"
                        for m in metrics if m in r
                    )
                    lines.append(f"  {rank}. **{b}** — {vals}")

                lines += ["", f"📉 **Lowest {dim_label}:**"]
                # Bottom entries
                for rank, (_, r) in enumerate(sorted_df.tail(n_top).iloc[::-1].iterrows(), 1):
                    b    = r.get("breakdown", "—")
                    vals = " | ".join(
                        f"{_METRIC_LABELS.get(m, m)}: {self._fv(float(r[m]), m)}"
                        for m in metrics if m in r
                    )
                    lines.append(f"  {rank}. **{b}** — {vals}")

                # Add gap context if exactly 2+ entries
                if len(sorted_df) >= 2:
                    top_val = float(sorted_df.iloc[0][m0])
                    bot_val = float(sorted_df.iloc[-1][m0])
                    gap_pct = ((top_val - bot_val) / abs(top_val) * 100) if top_val else 0
                    lines += [
                        "",
                        f"**Gap:** {self._fv(top_val - bot_val, m0)} ({gap_pct:.0f}% difference "
                        f"between highest and lowest)",
                    ]

                # Also show full ranking if more than 2 entries
                if len(sorted_df) > 2:
                    lines += ["", f"**Full {dim_label} ranking:**"]
                    for rank, (_, r) in enumerate(sorted_df.iterrows(), 1):
                        b    = r.get("breakdown", "—")
                        vals = " | ".join(
                            f"{_METRIC_LABELS.get(m, m)}: {self._fv(float(r[m]), m)}"
                            for m in metrics if m in r
                        )
                        lines.append(f"  {rank}. **{b}** — {vals}")

                return "\n".join(lines)

            # Normal breakdown display
            lines = [f"Here's how **{_METRIC_LABELS.get(metrics[0], metrics[0])}** breaks down by **{dim_label}**:{ctx_line}", ""]
            if len(sorted_df) == 1:
                only_name = sorted_df.iloc[0].get("breakdown", "—")
                lines.append(
                    f"⚠️ Only **{only_name}** is shown because the sidebar "
                    f"**{dim_label} filter** is set to a single value. "
                    f"Select all {dim_label}s in the sidebar to see the full breakdown."
                )
            for rank, (_, r) in enumerate(sorted_df.head(20).iterrows(), 1):
                b    = r.get("breakdown", "—")
                vals = " | ".join(
                    f"{_METRIC_LABELS.get(m, m)}: {self._fv(float(r[m]), m)}"
                    for m in metrics if m in r
                )
                lines.append(f"{rank}. **{b}** — {vals}")
            return "\n".join(lines)

        r0 = df.iloc[0]
        if len(metrics) == 1:
            m  = metrics[0]
            vs = self._fv(float(r0[m]), m)
            opener = _METRIC_OPENERS.get(m, f"{_METRIC_LABELS.get(m, m)} is")

            extras = []
            if m == "sales" and "profit" in r0:
                extras.append(f"profit of {self._fv(float(r0['profit']), 'profit')}")
            if m == "profit" and "sales" in r0:
                sales_v = float(r0["sales"])
                prof_v  = float(r0[m])
                margin  = (prof_v / sales_v * 100) if sales_v else 0
                extras.append(f"a **{margin:.1f}% margin** on {self._fv(sales_v, 'sales')} revenue")
            if m == "orders" and "sales" in r0:
                orders_v = float(r0[m])
                sales_v  = float(r0["sales"])
                avg      = sales_v / orders_v if orders_v else 0
                extras.append(f"averaging **{self._fv(avg, 'sales')}** per order")

            extra_str = f", with {extras[0]}" if extras else ""
            return f"{opener} **{vs}**{extra_str}.{ctx_line}"

        lines = [f"Here's a summary of the requested metrics:{ctx_line}", ""]
        for m in metrics:
            if m in r0:
                label = _METRIC_LABELS.get(m, m.replace("_", " ").title())
                lines.append(f"- **{label}:** {self._fv(float(r0[m]), m)}")

        if "sales" in r0 and "profit" in r0 and "profit_margin" not in metrics:
            sales_v = float(r0["sales"])
            prof_v  = float(r0["profit"])
            margin  = (prof_v / sales_v * 100) if sales_v else 0
            lines.append(f"- **Profit Margin:** {margin:.2f}%")

        return "\n".join(lines)

    def _format_trend(self, plan: Dict[str, Any], df: pd.DataFrame,
                      metrics: List[str], grain: str, breakdown: Any, ctx_line: str) -> str:
        grain_label = {"week": "week", "month": "month", "quarter": "quarter", "year": "year"}.get(grain, grain)
        metric_str  = " & ".join(_METRIC_LABELS.get(m, m) for m in metrics)
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
        top_k     = plan.get("top_k", 10)

        # ── Direct answer for top-1 queries ──────────────────────
        if top_k == 1 and not df.empty:
            r   = df.iloc[0]
            b   = r.get("breakdown", "—")
            val = self._fv(float(r[m0]), m0)
            secondary = ""
            if m0 == "sales" and "profit" in r:
                pm = (float(r["profit"]) / float(r[m0]) * 100) if float(r[m0]) else 0
                secondary = f", with a **{pm:.1f}% profit margin**"
            elif m0 == "profit" and "sales" in r:
                secondary = f" on {self._fv(float(r['sales']), 'sales')} in revenue"
            return (
                f"**{b}** is the best-performing {dim_label} by "
                f"{_METRIC_LABELS.get(m0, m0)}{ctx_line}, "
                f"with **{val}**{secondary}."
            )

        # ── Original multi-rank display ───────────────────────────
        lines = [
            f"Here are the **top {top_k} {dim_label}s** ranked by "
            f"{_METRIC_LABELS.get(m0, m0)}:{ctx_line}",
            "",
        ]
        for i, (_, r) in enumerate(df.head(top_k).iterrows(), 1):
            b  = r.get("breakdown", "—")
            vs = self._fv(float(r[m0]), m0)
            secondary = ""
            if m0 == "sales" and "profit" in r:
                pm = (float(r["profit"]) / float(r[m0]) * 100) if float(r[m0]) else 0
                secondary = f" *(margin: {pm:.1f}%)*"
            elif m0 == "profit" and "sales" in r:
                secondary = f" *(on {self._fv(float(r['sales']), 'sales')} revenue)*"
            lines.append(f"{i}. **{b}** — {vs}{secondary}")
        return "\n".join(lines)

    # ── Value formatter ───────────────────────────────────────

    @staticmethod
    def _fv(v: float, metric: str) -> str:
        if metric in {"sales", "profit"}:  return f"\\${v:,.0f}"
        if metric == "profit_margin":      return f"{v:.2f}%"
        return f"{int(v):,}"