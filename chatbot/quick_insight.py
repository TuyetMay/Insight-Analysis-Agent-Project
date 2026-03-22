# chatbot/quick_insight.py
"""
Quick KPI insight handler — fixed version.

Changes vs previous:
1. top_product & top_region bullets connected to YoY trend (not isolated facts)
2. _generate_action: robust output normalisation (handles "- " and "  - ")
3. _fallback_action: specific numbers, no generic advice
4. Removed _infer_cause (was speculating without data)
"""
from __future__ import annotations
from typing import Any, Dict
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

        # ── 💡 Insight — mỗi bullet connect với trend ─────────
        lines += ["", "💡 **Insight:**"]

        # 1. Strongest growth — không đoán, chỉ quan sát
        lines.append(
            f"  - Strongest growth was in **{best_period}** (**{best_pct:+.1f}%**) — "
            f"root cause is unclear from aggregates alone; compare order count vs AOV "
            f"between **{best_period}** and the prior period to identify the driver."
        )

        # 2. Deceleration — balanced, không negative bias
        if len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]:
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

        # 3. Top region — connected to trend
        if not top_reg_df.empty:
            top_reg   = top_reg_df.iloc[0].get("breakdown", "—")
            top_reg_v = float(top_reg_df.iloc[0].get("sales", 0))
            reg_share = (top_reg_v / total * 100) if total else 0
            lines.append(
                f"  - **{top_reg}** contributes **{reg_share:.0f}%** of total sales "
                f"(\\${top_reg_v:,.0f}) — verify whether its **{best_period}** performance "
                f"spiked specifically, which would confirm it drove the headline growth."
            )

        # 4. Top product — connected to trend
        if not top_cat_df.empty:
            top_prod   = top_cat_df.iloc[0].get("breakdown", "—")
            top_prod_v = float(top_cat_df.iloc[0].get("sales", 0))
            prod_share = (top_prod_v / total * 100) if total else 0
            lines.append(
                f"  - **{top_prod}** holds **{prod_share:.0f}%** of total sales "
                f"(\\${top_prod_v:,.0f}) — if its share grew in **{best_period}**, "
                f"it's a product-mix driver; if flat, growth was market-wide."
            )

        # ── 🚀 Action — LLM-generated ─────────────────────────
        transitions_text = "\n".join(
            f"  {_fmt_period(sdf.iloc[i-1].get('period',''), grain)} → "
            f"{_fmt_period(sdf.iloc[i].get('period',''), grain)}: "
            f"{transitions[i-1][0]:+.1f}%"
            for i in range(1, n)
        )
        action_context = {
            "period_range":        f"{_fmt_period(sdf.iloc[0].get('period',''), grain)} – {_fmt_period(sdf.iloc[-1].get('period',''), grain)}",
            "overall_change_pct":  overall_chg,
            "first_label":         _fmt_period(sdf.iloc[0].get("period", ""), grain),
            "first_value":         first_v,
            "last_label":          last_period,
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "best_period":         best_period,
            "top_product":         top_cat_df.iloc[0].get("breakdown", "—") if not top_cat_df.empty else "—",
            "top_product_value":   float(top_cat_df.iloc[0].get("sales", 0)) if not top_cat_df.empty else 0,
            "top_product_share":   (float(top_cat_df.iloc[0].get("sales", 0)) / total * 100) if (not top_cat_df.empty and total) else 0,
            "top_region":          top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—",
            "top_region_value":    float(top_reg_df.iloc[0].get("sales", 0)) if not top_reg_df.empty else 0,
            "top_region_share":    (float(top_reg_df.iloc[0].get("sales", 0)) / total * 100) if (not top_reg_df.empty and total) else 0,
            "is_decelerating":     len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0],
            "last_transition_pct": transitions[-1][0] if transitions else 0,
            "prev_transition_pct": transitions[-2][0] if len(transitions) >= 2 else 0,
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
            top_reg   = top_reg_df.iloc[0].get("breakdown", "—")
            top_reg_v = float(top_reg_df.iloc[0].get("profit", 0))
            reg_share = (top_reg_v / total_profit * 100) if total_profit else 0
            last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
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
            "period_range":        f"{_fmt_period(sdf.iloc[0].get('period',''), grain)} – {_fmt_period(sdf.iloc[-1].get('period',''), grain)}",
            "overall_change_pct":  overall_chg,
            "first_label":         _fmt_period(sdf.iloc[0].get("period", ""), grain),
            "first_value":         first_v,
            "last_label":          _fmt_period(sdf.iloc[-1].get("period", ""), grain),
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "overall_margin":      overall_margin,
            "top_region":          top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—",
            "top_region_value":    float(top_reg_df.iloc[0].get("profit", 0)) if not top_reg_df.empty else 0,
            "top_region_share":    (float(top_reg_df.iloc[0].get("profit", 0)) / total_profit * 100) if (not top_reg_df.empty and total_profit) else 0,
            "is_decelerating":     len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0],
            "last_transition_pct": transitions[-1][0] if transitions else 0,
            "prev_transition_pct": transitions[-2][0] if len(transitions) >= 2 else 0,
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

        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)

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
            "period_range":        f"{_fmt_period(sdf.iloc[0].get('period',''), grain)} – {_fmt_period(sdf.iloc[-1].get('period',''), grain)}",
            "overall_change_pct":  overall_chg,
            "first_label":         _fmt_period(sdf.iloc[0].get("period", ""), grain),
            "first_value":         first_v,
            "last_label":          last_period,
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "aov":                 aov,
            "total_orders":        total_orders,
            "is_decelerating":     len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0],
            "last_transition_pct": transitions[-1][0] if transitions else 0,
            "prev_transition_pct": transitions[-2][0] if len(transitions) >= 2 else 0,
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
            lowest_cat = top_cat_df.iloc[-1].get("breakdown", "—")
            lowest_m   = float(top_cat_df.iloc[-1].get("profit_margin", 0))
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
            "period_range":        f"{_fmt_period(sdf.iloc[0].get('period',''), grain)} – {_fmt_period(sdf.iloc[-1].get('period',''), grain)}",
            "overall_change_pct":  delta,
            "first_label":         _fmt_period(sdf.iloc[0].get("period", ""), grain),
            "first_value":         first_v,
            "last_label":          _fmt_period(sdf.iloc[-1].get("period", ""), grain),
            "last_value":          last_v,
            "transitions_text":    transitions_text,
            "overall_margin":      overall_margin,
            "vs_benchmark":        vs_benchmark,
            "lowest_margin_cat":   top_cat_df.iloc[-1].get("breakdown", "—") if not top_cat_df.empty else "—",
            "lowest_margin_val":   float(top_cat_df.iloc[-1].get("profit_margin", 0)) if not top_cat_df.empty else 0,
            "is_decelerating":     len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0],
            "last_transition_pct": transitions[-1][0] if transitions else 0,
            "prev_transition_pct": transitions[-2][0] if len(transitions) >= 2 else 0,
        }
        action_text = self._generate_action("profit_margin", action_context)
        lines += ["", "🚀 **Action:**", action_text]

        return "\n".join(lines)

    # ── Action generator (LLM + fallback) ────────────────────

    def _generate_action(self, kpi: str, context: dict) -> str:
        if not self.insight.client or not self.insight.model_name:
            return self._fallback_action(kpi, context)

        prompt = f"""You are a senior business analyst. Based on the actual data below,
write exactly 2 specific, actionable recommendations.

=== DATA ===
KPI: {kpi.upper()}
Period: {context.get('period_range', 'N/A')}
Overall change: {context.get('overall_change_pct', 0):+.1f}%
  ({context.get('first_label', '')}: ${context.get('first_value', 0):,.0f} →
   {context.get('last_label', '')}: ${context.get('last_value', 0):,.0f})

Period-by-period transitions:
{context.get('transitions_text', 'N/A')}

Top sub-category: {context.get('top_product', 'N/A')} (${context.get('top_product_value', 0):,.0f}, {context.get('top_product_share', 0):.0f}% of total)
Top region: {context.get('top_region', 'N/A')} (${context.get('top_region_value', 0):,.0f}, {context.get('top_region_share', 0):.0f}% of total)
Overall margin: {context.get('overall_margin', 'N/A')}
Deceleration detected: {context.get('is_decelerating', False)}
Latest period growth: {context.get('last_transition_pct', 0):+.1f}%
Prior period growth: {context.get('prev_transition_pct', 0):+.1f}%

=== RULES ===
- MUST cite at least one specific number from the data above in each action
- No generic advice ("monitor", "review strategy" are NOT acceptable)
- Start each action with a strong verb (Compare, Audit, Cap, Investigate, Shift, etc.)
- If deceleration detected: focus on root-cause diagnosis
- If strong growth: focus on scaling what's working
- Max 1 sentence per action

Format: exactly 2 lines. Each line starts with "- " (dash space).

Write 2 actions now:"""

        from google.genai import types as genai_types
        try:
            resp = self.insight.client.models.generate_content(
                model=self.insight.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=180),
            )
            text = (getattr(resp, "text", "") or "").strip()

            # ✅ FIX 3: Robust normalisation — handles "- ", "  - ", "• ", numbered, etc.
            if text and len(text) > 20:
                out_lines = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Strip any existing prefix bullets/numbers
                    line = line.lstrip("0123456789.)•–—*·").strip()
                    if line.startswith("- "):
                        line = line[2:].strip()
                    if line:
                        out_lines.append(f"  - {line}")
                if len(out_lines) >= 1:
                    return "\n".join(out_lines[:2])
        except Exception:
            pass

        return self._fallback_action(kpi, context)

    def _fallback_action(self, kpi: str, context: dict) -> str:
        """Rule-based fallback với specific numbers — không dùng câu chung chung."""
        top_region  = context.get("top_region",  "top region")
        top_product = context.get("top_product", "top sub-category")
        is_decel    = context.get("is_decelerating", False)
        best_period = context.get("best_period", "peak period")

        if kpi == "sales":
            if is_decel:
                a1 = (
                    f"  - Compare order count vs AOV in **{top_region}** between "
                    f"**{best_period}** and the subsequent period to determine whether "
                    f"the slowdown from **{context.get('prev_transition_pct',0):+.1f}%** "
                    f"to **{context.get('last_transition_pct',0):+.1f}%** is volume- or value-driven."
                )
            else:
                a1 = (
                    f"  - Scale the strategy that drove **{context.get('last_transition_pct',0):+.1f}%** "
                    f"growth in **{top_region}** ({context.get('top_region_share',0):.0f}% of total) — "
                    f"identify which channel or product mix changed and replicate it."
                )
            a2 = (
                f"  - Audit **{top_product}** ({context.get('top_product_share',0):.0f}% of total sales) "
                f"YoY share in **{best_period}** vs baseline — rising share = product-mix driver, "
                f"flat share = market-wide lift."
            )

        elif kpi == "profit":
            a1 = (
                f"  - Investigate sub-categories in **{top_region}** "
                f"(\\${context.get('top_region_value',0):,.0f} profit, "
                f"{context.get('top_region_share',0):.0f}% of total) "
                f"with discount > 20% — cap those discounts to protect the "
                f"**{context.get('overall_margin',0):.1f}%** margin."
            )
            a2 = (
                f"  - Shift sales mix toward the highest-margin category to lift "
                f"overall margin from **{context.get('overall_margin',0):.1f}%** toward the 15% target."
            )

        elif kpi == "orders":
            aov = context.get("aov", 0)
            a1 = (
                f"  - Compare order count vs AOV between the last 2 periods to confirm "
                f"whether the **{context.get('last_transition_pct',0):+.1f}%** order growth "
                f"came from more customers or higher basket size at \\${aov:,.0f} AOV."
            )
            a2 = (
                f"  - Launch bundle promotions on top sub-categories to push AOV "
                f"beyond \\${aov:,.0f} — a 10% AOV lift on "
                f"**{context.get('total_orders',0):,}** orders adds significant revenue."
            )

        else:  # profit_margin
            a1 = (
                f"  - Cap discounts at 20% in **{context.get('lowest_margin_cat','lowest-margin category')}** "
                f"({context.get('lowest_margin_val',0):.1f}% margin) — this is the fastest "
                f"lever to recover the **{abs(context.get('vs_benchmark',0)):.1f}pp gap** "
                f"vs the 12% benchmark."
            )
            a2 = (
                f"  - Shift sales mix toward the highest-margin category to lift overall "
                f"margin from **{context.get('overall_margin',0):.1f}%** toward 15%."
            )

        return f"{a1}\n{a2}"