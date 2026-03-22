# chatbot/quick_insight.py
"""
Quick KPI insight handler.

Fix v3:
1. _generate_action: quality gate — reject output that has no numbers (fallback)
2. Stronger fallback_action with % and $ values
3. Store metadata as instance vars (best_period, top_region, etc.)
   → orchestrator reads these to build analyst-level follow-up suggestions
"""
from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
from chatbot.sql_builder import SQLBuilder
from chatbot.insight_generator import InsightGenerator


def _detect_grain(filters: Dict[str, Any]) -> str:
    dr = filters.get("date_range")
    if not dr or len(dr) < 2:
        return "year"
    try:
        start = pd.Timestamp(dr[0])
        end   = pd.Timestamp(dr[1])
        days  = (end - start).days
        if days > 365:   return "year"
        elif days > 31:  return "month"
        else:            return "week"
    except Exception:
        return "year"


def _grain_label(grain: str) -> str:
    return {
        "year":  "year-over-year",
        "month": "month-over-month",
        "week":  "week-over-week",
    }.get(grain, grain)


def _fmt_period(raw: Any, grain: str) -> str:
    from datetime import datetime as _dt
    s = str(raw)
    if grain == "year":    return s[:4]
    if grain == "month":   return s[:7]
    if grain == "week":    return s[:10]
    if grain == "quarter":
        try:
            dt = _dt.strptime(s[:10], "%Y-%m-%d")
            return f"{dt.year} Q{(dt.month-1)//3+1}"
        except Exception:
            return s[:7]
    return s[:10]


def _period_name(sdf: Any, idx: int, grain: str) -> str:
    try:
        return _fmt_period(sdf.iloc[idx].get("period", ""), grain)
    except Exception:
        return "that period"


class QuickInsightHandler:
    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any],
                 filters: Dict[str, Any],
                 gemini_client: Any = None, model_name: str = "") -> None:
        self.df      = df
        self.kpis    = kpis
        self.filters = filters
        self.sql     = SQLBuilder()
        self.insight = InsightGenerator(gemini_client, model_name)
        self.grain   = _detect_grain(filters)
        self._s0, self._e0 = self._date_range()

        # ── Metadata exposed to orchestrator for analyst suggestions ──
        self.kpi_name:            str   = ""
        self.best_period:         str   = ""
        self.top_region:          str   = ""
        self.top_product:         str   = ""
        self.overall_change_pct:  float = 0.0
        self.is_decelerating:     bool  = False
        self.last_transition_pct: float = 0.0
        self.prev_transition_pct: float = 0.0
        self.first_label:         str   = ""
        self.last_label:          str   = ""
        self.first_value:         float = 0.0
        self.last_value:          float = 0.0

    def _date_range(self):
        dr = self.filters.get("date_range")
        if dr and len(dr) == 2:
            fmt = lambda d: d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            return fmt(dr[0]), fmt(dr[1])
        dates = pd.to_datetime(self.df["order_date"], errors="coerce").dropna()
        return str(dates.min().date()), str(dates.max().date())

    def _base_filters(self):
        f = self.filters or {}
        return {
            "region":       list(f.get("region",   []) or []),
            "segment":      list(f.get("segment",  []) or []),
            "category":     list(f.get("category", []) or []),
            "sub_category": [],
        }

    def generate(self, kpi: str) -> str:
        self.kpi_name = kpi
        handlers = {
            "sales":         self._sales_insight,
            "profit":        self._profit_insight,
            "orders":        self._orders_insight,
            "profit_margin": self._margin_insight,
        }
        fn = handlers.get(kpi)
        return fn() if fn else "❌ Unknown KPI."

    # ── Sales ─────────────────────────────────────────────────

    def _sales_insight(self) -> str:
        grain        = self.grain
        grain_lbl    = _grain_label(grain)
        base_filters = self._base_filters()

        trend_plan = {
            "intent": "kpi_trend", "metrics": ["sales"],
            "time_grain": grain, "breakdown_by": None,
            "secondary_breakdown": None,
            "start_date": self._s0, "end_date": self._e0,
            "compare_period": None, "top_k": None,
            "order_by": "sales", "filters": base_filters,
            "show_extremes": False,
        }
        trend_df   = self.sql.run(trend_plan)
        top_cat_df = self.sql.run({**trend_plan, "intent": "kpi_rank", "time_grain": "none", "breakdown_by": "sub_category", "top_k": 1})
        top_reg_df = self.sql.run({**trend_plan, "intent": "kpi_rank", "time_grain": "none", "breakdown_by": "region",       "top_k": 1})

        total = float(self.kpis.get("total_sales", 0))
        lines = []

        if trend_df.empty or "period" not in trend_df.columns:
            return "⚠️ No trend data available for the selected filters."

        sdf         = trend_df.sort_values("period").reset_index(drop=True)
        first_v     = float(sdf.iloc[0]["sales"])
        last_v      = float(sdf.iloc[-1]["sales"])
        overall_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
        word        = "grown" if overall_chg >= 0 else "declined"
        n           = len(sdf)

        transitions = []
        for i in range(1, n):
            pv  = float(sdf.iloc[i-1]["sales"])
            cv  = float(sdf.iloc[i]["sales"])
            pct = (cv - pv) / abs(pv) * 100 if pv else 0
            tag = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
            transitions.append((pct, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0])
        best_pct    = transitions[best][0]
        best_period = _period_name(sdf, best + 1, grain)
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        top_reg   = top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—"
        top_prod  = top_cat_df.iloc[0].get("breakdown", "—") if not top_cat_df.empty else "—"
        is_decel  = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

        # ── Store metadata for orchestrator ───────────────────
        self.best_period         = best_period
        self.top_region          = top_reg
        self.top_product         = top_prod
        self.overall_change_pct  = overall_chg
        self.is_decelerating     = is_decel
        self.last_transition_pct = transitions[-1][0] if transitions else 0
        self.prev_transition_pct = transitions[-2][0] if len(transitions) >= 2 else 0
        self.first_label         = _fmt_period(sdf.iloc[0].get("period", ""), grain)
        self.last_label          = last_period
        self.first_value         = first_v
        self.last_value          = last_v

        # ── 📌 Summary ────────────────────────────────────────
        lines += [
            f"📌 Sales {word} **{abs(overall_chg):.0f}%** over {n} periods — "
            f"from **\\${first_v:,.0f}** to **\\${last_v:,.0f}**.",
            "",
            f"🎯 **Overview** ({grain_lbl}):",
        ]
        for i, (_, r) in enumerate(sdf.iterrows()):
            p = _fmt_period(r.get("period", ""), grain)
            v = float(r["sales"])
            if i == 0:
                lines.append(f"  - {p}: **\\${v:,.0f}** *(baseline)*")
            else:
                pct, tag = transitions[i-1]
                lines.append(f"  - {p}: **\\${v:,.0f}** {tag} {pct:+.1f}%")

        # ── 💡 Insight ─────────────────────────────────────────
        lines += ["", "💡 **Insight:**"]

        lines.append(
            f"  - Strongest growth was in **{best_period}** (**{best_pct:+.1f}%**) — "
            f"compare order count vs AOV between **{best_period}** and the prior period "
            f"to determine whether this was volume-driven or value-driven."
        )

        if len(transitions) >= 2 and is_decel:
            lines.append(
                f"  - Growth rate eased from **{transitions[-2][0]:+.1f}%** to "
                f"**{transitions[-1][0]:+.1f}%** — still strong, but the narrowing "
                f"gap warrants checking whether it reflects market saturation or seasonal pattern."
            )
        else:
            lines.append(
                f"  - Momentum is **sustained** — last period (**{last_period}**) "
                f"continued the positive trajectory with **{transitions[-1][0]:+.1f}%** growth."
            )

        if not top_reg_df.empty:
            top_reg_v  = float(top_reg_df.iloc[0].get("sales", 0))
            reg_share  = (top_reg_v / total * 100) if total else 0
            lines.append(
                f"  - **{top_reg}** contributes **{reg_share:.0f}%** of total sales "
                f"(\\${top_reg_v:,.0f}) — verify whether its **{best_period}** performance "
                f"spiked specifically, which would confirm it drove the headline growth."
            )

        if not top_cat_df.empty:
            top_prod_v  = float(top_cat_df.iloc[0].get("sales", 0))
            prod_share  = (top_prod_v / total * 100) if total else 0
            lines.append(
                f"  - **{top_prod}** holds **{prod_share:.0f}%** of total sales "
                f"(\\${top_prod_v:,.0f}) — if its share grew in **{best_period}**, "
                f"it's a product-mix driver; if flat, growth was market-wide."
            )

        # ── 🚀 Action ──────────────────────────────────────────
        transitions_text = "\n".join(
            f"  {_fmt_period(sdf.iloc[i-1].get('period',''), grain)} → "
            f"{_fmt_period(sdf.iloc[i].get('period',''), grain)}: "
            f"{transitions[i-1][0]:+.1f}%"
            for i in range(1, n)
        )
        action_context = {
            "period_range":        f"{self.first_label} – {last_period}",
            "overall_change_pct":  overall_chg,
            "first_label":         self.first_label,
            "first_value":         first_v,
            "last_label":          last_period,
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "best_period":         best_period,
            "top_product":         top_prod,
            "top_product_value":   float(top_cat_df.iloc[0].get("sales", 0)) if not top_cat_df.empty else 0,
            "top_product_share":   (float(top_cat_df.iloc[0].get("sales", 0)) / total * 100) if (not top_cat_df.empty and total) else 0,
            "top_region":          top_reg,
            "top_region_value":    float(top_reg_df.iloc[0].get("sales", 0)) if not top_reg_df.empty else 0,
            "top_region_share":    (float(top_reg_df.iloc[0].get("sales", 0)) / total * 100) if (not top_reg_df.empty and total) else 0,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
        }
        action_text = self._generate_action("sales", action_context)
        lines += ["", "🚀 **Action:**", action_text]

        return "\n".join(lines)

    # ── Profit ────────────────────────────────────────────────

    def _profit_insight(self) -> str:
        grain        = self.grain
        grain_lbl    = _grain_label(grain)
        base_filters = self._base_filters()

        trend_plan = {
            "intent": "kpi_trend", "metrics": ["profit"],
            "time_grain": grain, "breakdown_by": None,
            "secondary_breakdown": None,
            "start_date": self._s0, "end_date": self._e0,
            "compare_period": None, "top_k": None,
            "order_by": "profit", "filters": base_filters,
            "show_extremes": False,
        }
        trend_df   = self.sql.run(trend_plan)
        top_reg_df = self.sql.run({**trend_plan, "intent": "kpi_rank", "time_grain": "none", "breakdown_by": "region", "top_k": 3})

        total_profit   = float(self.kpis.get("total_profit", 0))
        total_sales    = float(self.kpis.get("total_sales", 1))
        overall_margin = (total_profit / total_sales * 100)
        lines          = []

        if trend_df.empty or "period" not in trend_df.columns:
            return "⚠️ No trend data available for the selected filters."

        sdf         = trend_df.sort_values("period").reset_index(drop=True)
        first_v     = float(sdf.iloc[0]["profit"])
        last_v      = float(sdf.iloc[-1]["profit"])
        overall_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
        word        = "grown" if overall_chg >= 0 else "declined"
        n           = len(sdf)

        transitions = []
        for i in range(1, n):
            pv  = float(sdf.iloc[i-1]["profit"])
            cv  = float(sdf.iloc[i]["profit"])
            pct = (cv - pv) / abs(pv) * 100 if pv else 0
            tag = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
            transitions.append((pct, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else self.last_label
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        top_reg     = top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—"
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

        # ── Store metadata ─────────────────────────────────────
        self.best_period         = best_period
        self.top_region          = top_reg
        self.top_product         = ""
        self.overall_change_pct  = overall_chg
        self.is_decelerating     = is_decel
        self.last_transition_pct = transitions[-1][0] if transitions else 0
        self.prev_transition_pct = transitions[-2][0] if len(transitions) >= 2 else 0
        self.first_label         = _fmt_period(sdf.iloc[0].get("period", ""), grain)
        self.last_label          = last_period
        self.first_value         = first_v
        self.last_value          = last_v

        margin_health = "healthy" if overall_margin >= 12 else "tight"
        lines += [
            f"📌 Profit {word} **{abs(overall_chg):.0f}%** {grain_lbl} — "
            f"margin currently **{overall_margin:.1f}%** ({margin_health}).",
            "",
            f"🎯 **Overview** ({grain_lbl}):",
        ]
        for i, (_, r) in enumerate(sdf.iterrows()):
            p = _fmt_period(r.get("period", ""), grain)
            v = float(r["profit"])
            if i == 0:
                lines.append(f"  - {p}: **\\${v:,.0f}** *(baseline)*")
            else:
                pct, tag = transitions[i-1]
                lines.append(f"  - {p}: **\\${v:,.0f}** {tag} {pct:+.1f}%")

        lines += ["", "💡 **Insight:**"]

        negatives = sdf[sdf["profit"] < 0]
        if not negatives.empty:
            lines.append(
                f"  - ⚠️ **{len(negatives)} period(s)** recorded negative profit — "
                f"cross-reference with discount rate in those periods to confirm causation."
            )
        else:
            lines.append(
                f"  - Profit remained **positive across all {n} periods** — "
                f"no structural loss-making detected in this timeframe."
            )

        if overall_margin < 10:
            lines.append(
                f"  - ⚠️ Margin of **{overall_margin:.1f}%** is below the 12% retail benchmark — "
                f"identify sub-categories with discount > 20% as the most likely drag."
            )
        elif overall_margin > 18:
            lines.append(
                f"  - Margin of **{overall_margin:.1f}%** is strong — "
                f"pricing power is intact; priority is scaling order volume."
            )
        else:
            lines.append(
                f"  - Margin of **{overall_margin:.1f}%** is within healthy range — "
                f"targeting 15–18% is achievable with discount discipline."
            )

        if not top_reg_df.empty:
            top_reg_v = float(top_reg_df.iloc[0].get("profit", 0))
            reg_share = (top_reg_v / total_profit * 100) if total_profit else 0
            lines.append(
                f"  - **{top_reg}** leads profitability at **\\${top_reg_v:,.0f}** "
                f"({reg_share:.0f}% of total profit) — verify whether its margin trend "
                f"in **{last_period}** mirrors the overall {word} pattern."
            )

        transitions_text = "\n".join(
            f"  {_fmt_period(sdf.iloc[i-1].get('period',''), grain)} → "
            f"{_fmt_period(sdf.iloc[i].get('period',''), grain)}: {transitions[i-1][0]:+.1f}%"
            for i in range(1, n)
        )
        action_context = {
            "period_range":        f"{self.first_label} – {last_period}",
            "overall_change_pct":  overall_chg,
            "first_label":         self.first_label,
            "first_value":         first_v,
            "last_label":          last_period,
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "overall_margin":      overall_margin,
            "best_period":         best_period,
            "top_region":          top_reg,
            "top_region_value":    float(top_reg_df.iloc[0].get("profit", 0)) if not top_reg_df.empty else 0,
            "top_region_share":    (float(top_reg_df.iloc[0].get("profit", 0)) / total_profit * 100) if (not top_reg_df.empty and total_profit) else 0,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
        }
        action_text = self._generate_action("profit", action_context)
        lines += ["", "🚀 **Action:**", action_text]

        return "\n".join(lines)

    # ── Orders ────────────────────────────────────────────────

    def _orders_insight(self) -> str:
        grain        = self.grain
        grain_lbl    = _grain_label(grain)
        base_filters = self._base_filters()

        trend_plan = {
            "intent": "kpi_trend", "metrics": ["orders"],
            "time_grain": grain, "breakdown_by": None,
            "secondary_breakdown": None,
            "start_date": self._s0, "end_date": self._e0,
            "compare_period": None, "top_k": None,
            "order_by": "orders", "filters": base_filters,
            "show_extremes": False,
        }
        trend_df = self.sql.run(trend_plan)

        total_orders = int(self.kpis.get("total_orders", 0))
        total_sales  = float(self.kpis.get("total_sales", 1))
        aov          = total_sales / total_orders if total_orders else 0
        lines        = []

        if trend_df.empty or "period" not in trend_df.columns:
            return "⚠️ No trend data available for the selected filters."

        sdf         = trend_df.sort_values("period").reset_index(drop=True)
        first_v     = float(sdf.iloc[0]["orders"])
        last_v      = float(sdf.iloc[-1]["orders"])
        overall_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
        word        = "grown" if overall_chg >= 0 else "declined"
        n           = len(sdf)

        transitions = []
        for i in range(1, n):
            pv  = float(sdf.iloc[i-1]["orders"])
            cv  = float(sdf.iloc[i]["orders"])
            pct = (cv - pv) / abs(pv) * 100 if pv else 0
            tag = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
            transitions.append((pct, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else ""
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

        # ── Store metadata ─────────────────────────────────────
        self.best_period         = best_period
        self.top_region          = ""
        self.top_product         = ""
        self.overall_change_pct  = overall_chg
        self.is_decelerating     = is_decel
        self.last_transition_pct = transitions[-1][0] if transitions else 0
        self.prev_transition_pct = transitions[-2][0] if len(transitions) >= 2 else 0
        self.first_label         = _fmt_period(sdf.iloc[0].get("period", ""), grain)
        self.last_label          = last_period
        self.first_value         = first_v
        self.last_value          = last_v

        lines += [
            f"📌 Orders {word} **{abs(overall_chg):.0f}%** {grain_lbl} — "
            f"**{total_orders:,}** total orders, avg **\\${aov:,.0f}** per order.",
            "",
            f"🎯 **Overview** ({grain_lbl}):",
        ]
        for i, (_, r) in enumerate(sdf.iterrows()):
            p = _fmt_period(r.get("period", ""), grain)
            v = int(r["orders"])
            if i == 0:
                lines.append(f"  - {p}: **{v:,} orders** *(baseline)*")
            else:
                pct, tag = transitions[i-1]
                lines.append(f"  - {p}: **{v:,} orders** {tag} {pct:+.1f}%")

        lines += ["", "💡 **Insight:**"]
        lines.append(
            f"  - AOV of **\\${aov:,.0f}** — "
            + ("above $400 suggests high-value items; cross-sell opportunity is strong."
               if aov > 400 else
               "below $400 suggests lower-value items; bundle strategy could increase basket size.")
        )
        if len(transitions) >= 2:
            recent_avg = sum(t[0] for t in transitions[-2:]) / 2
            if recent_avg > 10:
                lines.append(
                    f"  - Recent momentum is **accelerating** — last 2 periods "
                    f"averaged **+{recent_avg:.0f}%** growth, above the full-period average."
                )
            elif recent_avg < -5:
                lines.append(
                    f"  - ⚠️ Recent momentum is **slowing** — last 2 periods "
                    f"averaged **{recent_avg:.0f}%**; check demand signals in **{last_period}**."
                )
            else:
                lines.append(
                    f"  - Order volume is **stable** — last 2 periods averaged "
                    f"**{recent_avg:+.0f}%**, no dramatic acceleration or reversal."
                )

        transitions_text = "\n".join(
            f"  {_fmt_period(sdf.iloc[i-1].get('period',''), grain)} → "
            f"{_fmt_period(sdf.iloc[i].get('period',''), grain)}: {transitions[i-1][0]:+.1f}%"
            for i in range(1, n)
        )
        action_context = {
            "period_range":        f"{self.first_label} – {last_period}",
            "overall_change_pct":  overall_chg,
            "first_label":         self.first_label,
            "first_value":         first_v,
            "last_label":          last_period,
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "best_period":         best_period,
            "aov":                 aov,
            "total_orders":        total_orders,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
        }
        action_text = self._generate_action("orders", action_context)
        lines += ["", "🚀 **Action:**", action_text]

        return "\n".join(lines)

    # ── Profit Margin ─────────────────────────────────────────

    def _margin_insight(self) -> str:
        grain        = self.grain
        grain_lbl    = _grain_label(grain)
        base_filters = self._base_filters()

        trend_plan = {
            "intent": "kpi_trend", "metrics": ["profit_margin"],
            "time_grain": grain, "breakdown_by": None,
            "secondary_breakdown": None,
            "start_date": self._s0, "end_date": self._e0,
            "compare_period": None, "top_k": None,
            "order_by": "profit_margin", "filters": base_filters,
            "show_extremes": False,
        }
        trend_df   = self.sql.run(trend_plan)
        top_cat_df = self.sql.run({**trend_plan,
            "intent": "kpi_rank", "time_grain": "none",
            "metrics": ["profit_margin"], "breakdown_by": "category",
            "top_k": 3, "order_by": "profit_margin"})

        overall_margin = float(self.kpis.get("profit_margin", 0))
        vs_benchmark   = overall_margin - 12.0
        lines          = []

        if trend_df.empty or "period" not in trend_df.columns:
            return "⚠️ No trend data available for the selected filters."

        sdf     = trend_df.sort_values("period").reset_index(drop=True)
        first_v = float(sdf.iloc[0]["profit_margin"])
        last_v  = float(sdf.iloc[-1]["profit_margin"])
        delta   = last_v - first_v
        word    = "improved" if delta >= 0 else "compressed"
        n       = len(sdf)

        transitions = []
        for i in range(1, n):
            pv   = float(sdf.iloc[i-1]["profit_margin"])
            cv   = float(sdf.iloc[i]["profit_margin"])
            diff = cv - pv
            tag  = "📈" if diff >= 1 else ("📉" if diff <= -1 else "➡️")
            transitions.append((diff, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else ""
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        lowest_cat  = top_cat_df.iloc[-1].get("breakdown", "—") if not top_cat_df.empty else "—"
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

        # ── Store metadata ─────────────────────────────────────
        self.best_period         = best_period
        self.top_region          = ""
        self.top_product         = lowest_cat
        self.overall_change_pct  = delta
        self.is_decelerating     = is_decel
        self.last_transition_pct = transitions[-1][0] if transitions else 0
        self.prev_transition_pct = transitions[-2][0] if len(transitions) >= 2 else 0
        self.first_label         = _fmt_period(sdf.iloc[0].get("period", ""), grain)
        self.last_label          = last_period
        self.first_value         = first_v
        self.last_value          = last_v

        benchmark_str = "above" if vs_benchmark >= 0 else "below"
        lines += [
            f"📌 Profit margin {word} from **{first_v:.1f}%** to **{last_v:.1f}%** "
            f"({delta:+.1f}pp) — currently **{abs(vs_benchmark):.1f}pp {benchmark_str}** "
            f"the 12% retail benchmark.",
            "",
            f"🎯 **Overview** ({grain_lbl}):",
        ]
        for i, (_, r) in enumerate(sdf.iterrows()):
            p = _fmt_period(r.get("period", ""), grain)
            v = float(r["profit_margin"])
            if i == 0:
                lines.append(f"  - {p}: **{v:.1f}%** *(baseline)*")
            else:
                diff, tag = transitions[i-1]
                lines.append(f"  - {p}: **{v:.1f}%** {tag} {diff:+.1f}pp")

        lines += ["", "💡 **Insight:**"]
        if delta < -2:
            lines.append(
                f"  - ⚠️ Margin compressed **{abs(delta):.1f}pp** over the period — "
                f"check whether average discount rate rose or product mix shifted "
                f"toward lower-margin items in the same window."
            )
        elif delta > 2:
            lines.append(
                f"  - Margin expanded **{delta:.1f}pp** — pricing improvements or "
                f"a shift toward higher-margin products is working; protect this gain."
            )
        else:
            lines.append(
                f"  - Margin is **stable** ({delta:+.1f}pp total change) — "
                f"no structural shift detected, but still {abs(vs_benchmark):.1f}pp "
                f"{'above' if vs_benchmark >= 0 else 'below'} the 12% benchmark."
            )

        if not top_cat_df.empty:
            cat_lines = [
                f"**{r.get('breakdown','—')}** ({float(r.get('profit_margin',0)):.1f}%)"
                for _, r in top_cat_df.head(3).iterrows()
            ]
            lowest_m = float(top_cat_df.iloc[-1].get("profit_margin", 0))
            lines.append(
                f"  - Category ranking: {' > '.join(cat_lines)} — "
                f"**{lowest_cat}** at {lowest_m:.1f}% is the priority for discount control."
            )

        transitions_text = "\n".join(
            f"  {_fmt_period(sdf.iloc[i-1].get('period',''), grain)} → "
            f"{_fmt_period(sdf.iloc[i].get('period',''), grain)}: {transitions[i-1][0]:+.1f}pp"
            for i in range(1, n)
        )
        action_context = {
            "period_range":        f"{self.first_label} – {last_period}",
            "overall_change_pct":  delta,
            "first_label":         self.first_label,
            "first_value":         first_v,
            "last_label":          last_period,
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "overall_margin":      overall_margin,
            "vs_benchmark":        vs_benchmark,
            "best_period":         best_period,
            "lowest_margin_cat":   lowest_cat,
            "lowest_margin_val":   float(top_cat_df.iloc[-1].get("profit_margin", 0)) if not top_cat_df.empty else 0,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
        }
        action_text = self._generate_action("profit_margin", action_context)
        lines += ["", "🚀 **Action:**", action_text]

        return "\n".join(lines)

    # ── Action generator ──────────────────────────────────────

    def _generate_action(self, kpi: str, context: dict) -> str:
        """
        LLM-generated action with quality gate:
        - Must contain at least 1 number per action
        - Must have 2 lines
        - Falls back to rule-based if quality check fails
        """
        if not self.insight.client or not self.insight.model_name:
            return self._fallback_action(kpi, context)

        is_decel    = context.get("is_decelerating", False)
        best_period = context.get("best_period", "peak period")
        top_region  = context.get("top_region",  "top region")
        top_product = context.get("top_product", "top sub-category")

        prompt = f"""You are a senior business analyst. Write exactly 2 SPECIFIC next actions.

DATA:
- KPI: {kpi.upper()}
- Period: {context.get('period_range', 'N/A')}
- Trend: {context.get('first_label')} ${context.get('first_value', 0):,.0f} → {context.get('last_label')} ${context.get('last_value', 0):,.0f} ({context.get('overall_change_pct', 0):+.1f}%)
- Period transitions:
{context.get('transitions_text', 'N/A')}
- Strongest period: {best_period}
- Top region: {top_region} (${context.get('top_region_value', 0):,.0f}, {context.get('top_region_share', 0):.0f}%)
- Top sub-category: {top_product} (${context.get('top_product_value', 0):,.0f}, {context.get('top_product_share', 0):.0f}%)
- Overall margin: {context.get('overall_margin', 'N/A')}
- Deceleration: {is_decel} (last: {context.get('last_transition_pct', 0):+.1f}%, prev: {context.get('prev_transition_pct', 0):+.1f}%)

RULES (ALL MANDATORY — violation = your output is discarded):
1. EACH action MUST contain at least one specific number (%, $, or count) from DATA
2. Start with: Compare / Drill into / Audit / Cap / Shift / Investigate / Quantify
3. One sentence per action, ending with a period
4. "Investigate root causes" = REJECTED (too vague, no number)
5. "Monitor trends" = REJECTED (too vague, no number)
6. Action 1: diagnose {f"WHERE the {context.get('prev_transition_pct',0):+.1f}%→{context.get('last_transition_pct',0):+.1f}% deceleration is concentrated" if is_decel else f"HOW to scale the {context.get('last_transition_pct',0):+.1f}% growth from {best_period}"}
7. Action 2: target {top_product} ({context.get('top_product_share',0):.0f}% of total) or {top_region}

FORMAT: Exactly 2 lines. Each line: "- <action sentence>"
"""

        import re as _re
        from google.genai import types as genai_types
        try:
            resp = self.insight.client.models.generate_content(
                model=self.insight.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.2, max_output_tokens=200),
            )
            text = (getattr(resp, "text", "") or "").strip()

            if text and len(text) > 20:
                out_lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Strip prefixes
                    line = line.lstrip("0123456789.)•–—*·").strip()
                    if line.startswith("- "):
                        line = line[2:].strip()
                    if line:
                        out_lines.append(line)

                # ── QUALITY GATE: reject generic lines ────────
                valid_lines = []
                for line in out_lines[:3]:
                    has_number    = bool(_re.search(r'\d', line))
                    is_too_vague  = bool(_re.search(
                        r'\b(investigate root causes?|monitor trends?|review strategy|'
                        r'explore opportunities?|consider|evaluate|assess)\b',
                        line, _re.IGNORECASE
                    ))
                    if has_number and not is_too_vague:
                        valid_lines.append(f"  - {line}")

                if len(valid_lines) >= 2:
                    return "\n".join(valid_lines[:2])
                # Partial: use what's valid + fill rest from fallback
                if len(valid_lines) == 1:
                    fallback_lines = self._fallback_action(kpi, context).strip().splitlines()
                    fallback_lines = [l for l in fallback_lines if l.strip()]
                    if fallback_lines:
                        return valid_lines[0] + "\n" + fallback_lines[-1]

        except Exception:
            pass

        return self._fallback_action(kpi, context)

    def _fallback_action(self, kpi: str, context: dict) -> str:
        """Rule-based fallback — always produces specific numbers."""
        top_region  = context.get("top_region",  "top region")
        top_product = context.get("top_product", "top sub-category")
        best_period = context.get("best_period", "peak period")
        is_decel    = context.get("is_decelerating", False)
        last_pct    = context.get("last_transition_pct", 0)
        prev_pct    = context.get("prev_transition_pct", 0)

        if kpi == "sales":
            if is_decel:
                a1 = (
                    f"  - Drill into **{top_region}** order data — compare order count vs AOV "
                    f"between **{best_period}** and {context.get('last_label','last period')} "
                    f"to explain the {prev_pct:+.1f}%→{last_pct:+.1f}% deceleration."
                )
            else:
                a1 = (
                    f"  - Quantify **{top_region}**'s ({context.get('top_region_share',0):.0f}% of total) "
                    f"contribution to the **{last_pct:+.1f}%** growth in "
                    f"**{context.get('last_label','last period')}** — if outsized, replicate its playbook."
                )
            a2 = (
                f"  - Audit **{top_product}** ({context.get('top_product_share',0):.0f}% of total, "
                f"\\${context.get('top_product_value',0):,.0f}) share in **{best_period}** vs baseline — "
                f"rising share = product-mix driver, flat share = market-wide lift."
            )

        elif kpi == "profit":
            a1 = (
                f"  - Cap discounts at 20% in **{top_region}** "
                f"(\\${context.get('top_region_value',0):,.0f} profit, "
                f"{context.get('top_region_share',0):.0f}% of total) — "
                f"this is the fastest lever to close the gap to 15% margin "
                f"from the current **{context.get('overall_margin',0):.1f}%**."
            )
            a2 = (
                f"  - Shift sales mix in **{best_period}** toward highest-margin categories to lift "
                f"overall margin from **{context.get('overall_margin',0):.1f}%** toward 15%."
            )

        elif kpi == "orders":
            aov = context.get("aov", 0)
            a1 = (
                f"  - Compare order count vs AOV between **{best_period}** and the prior period — "
                f"determine whether the **{last_pct:+.1f}%** {'deceleration' if is_decel else 'growth'} "
                f"is volume-driven or basket-size-driven at \\${aov:,.0f} AOV."
            )
            a2 = (
                f"  - Launch bundles targeting AOV > \\${aov*1.1:,.0f} (+10%) — "
                f"applied to **{context.get('total_orders',0):,}** orders, "
                f"that's a direct \\${aov*0.1*context.get('total_orders',0):,.0f} revenue lift."
            )

        else:  # profit_margin
            a1 = (
                f"  - Cap discounts at 20% in **{context.get('lowest_margin_cat','lowest-margin category')}** "
                f"({context.get('lowest_margin_val',0):.1f}% margin) — fastest lever to recover "
                f"the **{abs(context.get('vs_benchmark',0)):.1f}pp gap** vs the 12% benchmark."
            )
            a2 = (
                f"  - Shift sales mix toward highest-margin category to lift overall "
                f"margin from **{context.get('overall_margin',0):.1f}%** toward 15%."
            )

        return f"{a1}\n{a2}"