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

def _trend_label(n: int, grain_lbl: str) -> str:
    if n < 2:   return "single data point"
    if n == 2:  return f"period-over-period comparison ({grain_lbl})"
    return f"overview ({grain_lbl})"

def _best_period_label(best_pct: float) -> str:
    if best_pct > 10:   return "peak growth period"
    elif best_pct > 0:  return "strongest growth period"
    elif best_pct > -10: return "most stable period"
    else:               return "least-decline period"

def _classify_situation(overall_chg: float, n: int) -> str:
    if overall_chg >= 20:   return "growth_strong"
    elif overall_chg >= 5:  return "growth_mild"
    elif overall_chg >= -10: return "stable"
    elif overall_chg >= -40: return "decline_mild"
    else:                    return "decline_severe"


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
        top_reg_df = self.sql.run({**trend_plan, "intent": "kpi_rank", "time_grain": "none", "breakdown_by": "region", "top_k": 1})

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
        situation   = _classify_situation(overall_chg, n)

        transitions = []
        for i in range(1, n):
            pv  = float(sdf.iloc[i-1]["sales"])
            cv  = float(sdf.iloc[i]["sales"])
            pct = (cv - pv) / abs(pv) * 100 if pv else 0
            tag = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
            transitions.append((pct, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_pct    = transitions[best][0] if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else ""
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        top_reg     = top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—"
        top_prod    = top_cat_df.iloc[0].get("breakdown", "—") if not top_cat_df.empty else "—"
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

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

        overview_label = _trend_label(n, grain_lbl)
        lines += [
            f"📌 Sales {word} **{abs(overall_chg):.0f}%** — "
            f"from **\\${first_v:,.0f}** to **\\${last_v:,.0f}**.",
            "",
            f"🎯 **{overview_label.capitalize()}:**",
        ]
        for i, (_, r) in enumerate(sdf.iterrows()):
            p = _fmt_period(r.get("period", ""), grain)
            v = float(r["sales"])
            if i == 0:
                lines.append(f"  - {p}: **\\${v:,.0f}** *(baseline)*")
            else:
                pct, tag = transitions[i-1]
                lines.append(f"  - {p}: **\\${v:,.0f}** {tag} {pct:+.1f}%")

        lines += ["", "💡 **Insight:**"]

        if situation == "decline_severe":
            lines.append(
                f"  - ⚠️ Sales dropped **{abs(overall_chg):.0f}%** — a decline of this magnitude "
                f"indicates a significant structural shift, not normal fluctuation. "
                f"Drill down by region and category to isolate the primary source."
            )
        elif situation == "decline_mild":
            lines.append(
                f"  - Sales declined **{abs(overall_chg):.0f}%** — check whether this is "
                f"concentrated in one region or category, or spread across the business."
            )
        elif best_period and best_pct > 0:
            lines.append(
                f"  - Strongest growth was in **{best_period}** (**{best_pct:+.1f}%**) — "
                f"compare order count vs AOV to determine whether this was volume-driven or value-driven."
            )

        if n >= 2 and len(transitions) >= 2:
            if is_decel and situation not in ("decline_mild", "decline_severe"):
                lines.append(
                    f"  - Growth rate eased from **{transitions[-2][0]:+.1f}%** to "
                    f"**{transitions[-1][0]:+.1f}%** — check whether this reflects market saturation "
                    f"or a seasonal pattern."
                )
            elif not is_decel and overall_chg > 0:
                lines.append(
                    f"  - Momentum is **sustained** — last period (**{last_period}**) "
                    f"continued the positive trajectory with **{transitions[-1][0]:+.1f}%** growth."
                )

        if not top_reg_df.empty:
            top_reg_v = float(top_reg_df.iloc[0].get("sales", 0))
            reg_share = (top_reg_v / total * 100) if total else 0
            lines.append(
                f"  - **{top_reg}** contributes **{reg_share:.0f}%** of total sales "
                f"(\\${top_reg_v:,.0f}) — "
                + ("verify whether it drove the headline decline." if overall_chg < 0
                   else f"verify whether its **{best_period}** performance spiked specifically.")
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
            "best_pct":            best_pct,
            "top_product":         top_prod,
            "top_product_value":   float(top_cat_df.iloc[0].get("sales", 0)) if not top_cat_df.empty else 0,
            "top_product_share":   (float(top_cat_df.iloc[0].get("sales", 0)) / total * 100) if (not top_cat_df.empty and total) else 0,
            "top_region":          top_reg,
            "top_region_value":    float(top_reg_df.iloc[0].get("sales", 0)) if not top_reg_df.empty else 0,
            "top_region_share":    (float(top_reg_df.iloc[0].get("sales", 0)) / total * 100) if (not top_reg_df.empty and total) else 0,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
            "situation":           situation,   
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
        situation   = _classify_situation(overall_chg, n)  # FIX-6

        transitions = []
        for i in range(1, n):
            pv  = float(sdf.iloc[i-1]["profit"])
            cv  = float(sdf.iloc[i]["profit"])
            pct = (cv - pv) / abs(pv) * 100 if pv else 0
            tag = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
            transitions.append((pct, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_pct    = transitions[best][0] if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else self.last_label
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        top_reg     = top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—"
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

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

        overview_label = _trend_label(n, grain_lbl)

        if n == 2:
            summary_period = f"from **{self.first_label}** to **{last_period}**"
        else:
            summary_period = grain_lbl

        lines += [
            f"📌 Profit {word} **{abs(overall_chg):.0f}%** {summary_period} — "
            f"margin at **{overall_margin:.1f}%**.",
            "",
            f"🎯 **{overview_label.capitalize()}:**",
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
        if situation == "decline_severe":
            lines.append(
                f"  - ⚠️ Profit dropped **{abs(overall_chg):.0f}%** — a decline of this "
                f"magnitude indicates a significant structural issue, not seasonal variation. "
                f"While profit remains positive, the scale of the drop requires immediate investigation."
            )
        elif negatives.empty:
            lines.append(
                f"  - Profit remained **positive across all {n} {'periods' if n > 1 else 'period'}** — "
                f"no structural loss-making detected in this timeframe."
            )
        else:
            lines.append(
                f"  - ⚠️ **{len(negatives)} period(s)** recorded negative profit — "
                f"cross-reference with discount rate in those periods to confirm causation."
            )

        if n >= 3:
            if is_decel and overall_chg > 0:
                lines.append(
                    f"  - Profit growth is **slowing** ({transitions[-2][0]:+.1f}% → {transitions[-1][0]:+.1f}%) "
                    f"— check whether this reflects margin compression or a natural volume plateau."
                )
            elif overall_chg > 30:
                lines.append(
                    f"  - Strong profit growth of **{overall_chg:.0f}%** over the period — "
                    f"verify whether margin has been maintained or if growth is purely volume-driven."
                )
        elif n == 2 and situation not in ("decline_severe", "decline_mild"):
            lines.append(
                f"  - Single-period comparison: **{self.first_label}** (\\${first_v:,.0f}) → "
                f"**{last_period}** (\\${last_v:,.0f}). More periods needed to confirm a trend."
            )

        if not top_reg_df.empty:
            top_reg_v = float(top_reg_df.iloc[0].get("profit", 0))
            reg_share = (top_reg_v / total_profit * 100) if total_profit else 0
            lines.append(
                f"  - **{top_reg}** leads profitability at **\\${top_reg_v:,.0f}** "
                f"({reg_share:.0f}% of total profit) — "
                + ("check how much it contributed to the overall decline." if overall_chg < -10
                   else f"verify whether its margin trend in **{last_period}** mirrors the overall {word} pattern.")
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
            "best_pct":            best_pct,                    
            "top_region":          top_reg,
            "top_region_value":    float(top_reg_df.iloc[0].get("profit", 0)) if not top_reg_df.empty else 0,
            "top_region_share":    (float(top_reg_df.iloc[0].get("profit", 0)) / total_profit * 100) if (not top_reg_df.empty and total_profit) else 0,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
            "situation":           situation,                  
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
        situation   = _classify_situation(overall_chg, n)

        transitions = []
        for i in range(1, n):
            pv  = float(sdf.iloc[i-1]["orders"])
            cv  = float(sdf.iloc[i]["orders"])
            pct = (cv - pv) / abs(pv) * 100 if pv else 0
            tag = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
            transitions.append((pct, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_pct    = transitions[best][0] if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else ""
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

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

        overview_label = _trend_label(n, grain_lbl)
        if n == 2:
            summary_period = f"from **{self.first_label}** to **{last_period}**"
        else:
            summary_period = grain_lbl

        lines += [
            f"📌 Orders {word} **{abs(overall_chg):.0f}%** {summary_period} — "
            f"**{total_orders:,}** total orders, avg **\\${aov:,.0f}** per order.",
            "",
            f"🎯 **{overview_label.capitalize()}:**",
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

        if situation == "decline_severe":
            lines.append(
                f"  - ⚠️ Orders dropped **{abs(overall_chg):.0f}%** — "
                f"from **{int(first_v):,}** to **{int(last_v):,}**. "
                f"This is a significant demand reduction; check whether it's segment-specific "
                f"or business-wide."
            )
        else:
            lines.append(
                f"  - AOV of **\\${aov:,.0f}** — "
                + ("above $400 suggests high-value items; cross-sell opportunity is strong."
                   if aov > 400 else
                   "below $400 suggests lower-value items; bundle strategy could increase basket size.")
            )

        if n >= 3 and len(transitions) >= 2:
            recent_avg = sum(t[0] for t in transitions[-2:]) / 2
            if recent_avg > 10:
                lines.append(
                    f"  - Recent momentum is **accelerating** — last 2 periods "
                    f"averaged **+{recent_avg:.0f}%** growth."
                )
            elif recent_avg < -5:
                lines.append(
                    f"  - ⚠️ Recent momentum is **slowing** — last 2 periods "
                    f"averaged **{recent_avg:.0f}%**; check demand signals in **{last_period}**."
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
            "best_pct":            best_pct,
            "aov":                 aov,
            "total_orders":        total_orders,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
            "situation":           situation,
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
        lines          = []

        if trend_df.empty or "period" not in trend_df.columns:
            return "⚠️ No trend data available for the selected filters."

        sdf     = trend_df.sort_values("period").reset_index(drop=True)
        first_v = float(sdf.iloc[0]["profit_margin"])
        last_v  = float(sdf.iloc[-1]["profit_margin"])
        delta   = last_v - first_v
        word    = "improved" if delta >= 0 else "compressed"
        n       = len(sdf)
        situation = _classify_situation(delta * 5, n)  
        transitions = []
        for i in range(1, n):
            pv   = float(sdf.iloc[i-1]["profit_margin"])
            cv   = float(sdf.iloc[i]["profit_margin"])
            diff = cv - pv
            tag  = "📈" if diff >= 1 else ("📉" if diff <= -1 else "➡️")
            transitions.append((diff, tag))

        best        = max(range(len(transitions)), key=lambda i: transitions[i][0]) if transitions else 0
        best_pct    = transitions[best][0] if transitions else 0
        best_period = _period_name(sdf, best + 1, grain) if transitions else ""
        last_period = _fmt_period(sdf.iloc[-1].get("period", ""), grain)
        lowest_cat  = top_cat_df.iloc[-1].get("breakdown", "—") if not top_cat_df.empty else "—"
        is_decel    = len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]

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

        overview_label = _trend_label(n, grain_lbl)

        if n == 2:
            summary_period = f"from **{self.first_label}** to **{last_period}**"
        else:
            summary_period = grain_lbl

        lines += [
            f"📌 Profit margin {word} from **{first_v:.1f}%** to **{last_v:.1f}%** "
            f"({delta:+.1f}pp) — current margin **{overall_margin:.1f}%**.",
            "",
            f"🎯 **{overview_label.capitalize()}:**",
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

        if n < 3:
            lines.append(
                f"  - Only **{n} period(s)** available — further data needed to confirm a trend direction."
            )
        elif delta < -2:
            lines.append(
                f"  - ⚠️ Margin compressed **{abs(delta):.1f}pp** — check whether average "
                f"discount rate rose or product mix shifted toward lower-margin items."
            )
        elif delta > 2:
            lines.append(
                f"  - Margin expanded **{delta:.1f}pp** — pricing improvements or "
                f"a shift toward higher-margin products is working; protect this gain."
            )
        else:
            lines.append(
                f"  - Margin is **stable** ({delta:+.1f}pp total change) — "
                f"no structural shift detected in this period."
            )

        if not top_cat_df.empty and n >= 2:
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
            "best_period":         best_period,
            "best_pct":            best_pct,
            "lowest_margin_cat":   lowest_cat,
            "lowest_margin_val":   float(top_cat_df.iloc[-1].get("profit_margin", 0)) if not top_cat_df.empty else 0,
            "is_decelerating":     is_decel,
            "last_transition_pct": self.last_transition_pct,
            "prev_transition_pct": self.prev_transition_pct,
            "situation":           situation,
        }
        action_text = self._generate_action("profit_margin", action_context)
        lines += ["", "🚀 **Action:**", action_text]
        return "\n".join(lines)

    # ── Action generator ──────────────────────────────────────

    def _generate_action(self, kpi: str, context: dict) -> str:
        if not self.insight.client or not self.insight.model_name:
            return self._fallback_action(kpi, context)

        situation   = context.get("situation", "stable")
        best_period = context.get("best_period", "peak period")
        best_pct    = context.get("best_pct", 0)
        last_period = context.get("last_label", "most recent period")
        is_decel    = context.get("is_decelerating", False)

        best_period_label = _best_period_label(best_pct)

        prompt = f"""You are a senior business analyst. Write exactly 2 SPECIFIC next actions.

        DATA:
        - KPI: {kpi.upper()}
        - Situation: {situation}
        - Period: {context.get('period_range', 'N/A')}
        - Trend: {context.get('first_label')} ${context.get('first_value', 0):,.0f} → {context.get('last_label')} ${context.get('last_value', 0):,.0f} ({context.get('overall_change_pct', 0):+.1f}%)
        - Period transitions:
        {context.get('transitions_text', 'N/A')}
        - {best_period_label}: {best_period} ({best_pct:+.1f}%)
        - Most recent period: {last_period}
        - Top region: {context.get('top_region', 'N/A')} (${context.get('top_region_value', 0):,.0f}, {context.get('top_region_share', 0):.0f}%)
        - Top sub-category: {context.get('top_product', 'N/A')} (${context.get('top_product_value', 0):,.0f}, {context.get('top_product_share', 0):.0f}%)
        - Deceleration: {is_decel}

        SITUATION-SPECIFIC RULES:
        {
        "- Situation is DECLINE_SEVERE (>40% drop). Actions must be DIAGNOSTIC, not prescriptive." +
        "\\n- DO NOT suggest 'cap discounts' — we don't have evidence discount is the cause." +
        "\\n- Action 1: identify WHERE the decline is concentrated (region? category? segment?)" +
        "\\n- Action 2: identify WHAT changed — sales volume, margin, or both?"
        if situation == "decline_severe" else
        "- Situation is " + situation + ". Use standard analytical approach."
        }

        MANDATORY RULES (violation = output discarded):
        1. Each action MUST contain at least one specific number from DATA
        2. Start with: Compare / Drill into / Audit / Cap / Shift / Quantify / Break down
        3. One sentence per action, ending with a period
        4. TIME RULE: Never say 'in {best_period}' as action target. Use 'going forward' or '{last_period}'
        5. {best_period} is a {best_period_label} — label it correctly if referenced

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
                    line = line.lstrip("0123456789.)•–—*·").strip()
                    if line.startswith("- "):
                        line = line[2:].strip()
                    if line:
                        out_lines.append(line)

                valid_lines = []
                for line in out_lines[:3]:
                    has_number    = bool(_re.search(r'\d', line))
                    is_too_vague  = bool(_re.search(
                        r'\b(investigate root causes?|monitor trends?|review strategy|'
                        r'explore opportunities?)\b',
                        line, _re.IGNORECASE
                    ))
                    is_past_action = (
                        bool(_re.search(r'\bin\s+\d{4}\b(?!.*compared|.*vs|.*versus|.*than)', line, _re.IGNORECASE))
                        and best_period in line
                    )
                    if has_number and not is_too_vague and not is_past_action:
                        valid_lines.append(f"  - {line}")

                if len(valid_lines) >= 2:
                    return "\n".join(valid_lines[:2])
                if len(valid_lines) == 1:
                    fallback_lines = self._fallback_action(kpi, context).strip().splitlines()
                    fallback_lines = [l for l in fallback_lines if l.strip()]
                    if fallback_lines:
                        return valid_lines[0] + "\n" + fallback_lines[-1]

        except Exception:
            pass

        return self._fallback_action(kpi, context)

    def _fallback_action(self, kpi: str, context: dict) -> str:
        top_region  = context.get("top_region",  "top region")
        top_product = context.get("top_product", "top sub-category")
        best_period = context.get("best_period", "peak period")
        best_pct    = context.get("best_pct", 0.0)
        last_period = context.get("last_label",  "most recent period")
        is_decel    = context.get("is_decelerating", False)
        last_pct    = context.get("last_transition_pct", 0)
        prev_pct    = context.get("prev_transition_pct", 0)
        situation   = context.get("situation", "stable")
        overall_chg = context.get("overall_change_pct", 0)
        best_period_label = _best_period_label(best_pct)   

        if kpi == "sales":
            if situation == "decline_severe":
                a1 = (
                    f"  - Break down the **{abs(overall_chg):.0f}%** sales decline by region — "
                    f"determine whether **{top_region}** ({context.get('top_region_share',0):.0f}% "
                    f"of total) led the drop or if it was spread across all regions."
                )
                a2 = (
                    f"  - Compare order count vs AOV between **{best_period}** "
                    f"({best_period_label}, {best_pct:+.1f}%) and **{last_period}** — "
                    f"a volume drop means demand loss; an AOV drop means pricing or mix shift."
                )
            elif situation == "decline_mild":
                a1 = (
                    f"  - Drill into **{top_region}** ({context.get('top_region_share',0):.0f}% of total) "
                    f"to identify whether the **{last_pct:+.1f}%** change is concentrated there "
                    f"or spread across all regions."
                )
                a2 = (
                    f"  - Audit **{top_product}** ({context.get('top_product_share',0):.0f}% of total, "
                    f"\\${context.get('top_product_value',0):,.0f}) performance in **{last_period}** "
                    f"vs baseline to check for category-level weakness."
                )
            else:
                if is_decel:
                    a1 = (
                        f"  - Drill into **{top_region}** order data — compare order count vs AOV "
                        f"between **{best_period}** and **{last_period}** to explain "
                        f"the {prev_pct:+.1f}%→{last_pct:+.1f}% deceleration."
                    )
                else:
                    a1 = (
                        f"  - Quantify **{top_region}**'s ({context.get('top_region_share',0):.0f}% of total) "
                        f"contribution to the **{last_pct:+.1f}%** growth in **{last_period}** — "
                        f"if outsized, replicate its playbook going forward."
                    )
                a2 = (
                    f"  - Audit **{top_product}** ({context.get('top_product_share',0):.0f}% of total, "
                    f"\\${context.get('top_product_value',0):,.0f}) share compared to **{best_period}** "
                    f"({best_period_label}) — rising share = product-mix driver, flat = market-wide lift."
                )

        elif kpi == "profit":
            if situation == "decline_severe":
                a1 = (
                    f"  - Break down the **{abs(overall_chg):.0f}%** profit decline by region and category — "
                    f"**{top_region}** holds {context.get('top_region_share',0):.0f}% of remaining profit "
                    f"(\\${context.get('top_region_value',0):,.0f}); determine if it also led the decline."
                )
                a2 = (
                    f"  - Compare category mix in **{best_period}** ({best_period_label}) vs "
                    f"**{last_period}** — identify whether higher-margin categories shrank in share "
                    f"or whether discounting rose, then prioritise the confirmed driver."
                )
            elif situation == "decline_mild":
                a1 = (
                    f"  - Drill into **{top_region}** (\\${context.get('top_region_value',0):,.0f}, "
                    f"{context.get('top_region_share',0):.0f}% of total profit) — "
                    f"check whether its margin compressed in **{last_period}** vs **{best_period}**."
                )
                a2 = (
                    f"  - Review discount levels in the current period vs **{best_period}** "
                    f"({best_period_label}) — if avg discount rose, that is the most likely driver "
                    f"of the **{abs(overall_chg):.0f}%** profit drop."
                )
            else:
                a1 = (
                    f"  - Cap discounts at 20% in **{top_region}** "
                    f"(\\${context.get('top_region_value',0):,.0f} profit, "
                    f"{context.get('top_region_share',0):.0f}% of total) — "
                    f"fastest lever to improve margin from the current "
                    f"**{context.get('overall_margin',0):.1f}%**."
                )
                a2 = (
                    f"  - Shift sales mix going forward toward highest-margin categories — "
                    f"compare category mix in **{best_period}** ({best_period_label}) vs "
                    f"**{last_period}** to identify which categories to prioritise."
                )

        elif kpi == "orders":
            aov = context.get("aov", 0)
            if situation in ("decline_severe", "decline_mild"):
                a1 = (
                    f"  - Break down the **{abs(overall_chg):.0f}%** order decline by segment — "
                    f"determine whether demand dropped uniformly or is concentrated in specific customer groups."
                )
                a2 = (
                    f"  - Compare order count and AOV between **{best_period}** ({best_period_label}) "
                    f"and **{last_period}** — if AOV held but volume fell, the issue is acquisition; "
                    f"if AOV fell, it's pricing or mix."
                )
            else:
                a1 = (
                    f"  - Compare order count vs AOV between **{best_period}** ({best_period_label}) "
                    f"and **{last_period}** — determine whether the **{last_pct:+.1f}%** "
                    f"{'deceleration' if is_decel else 'growth'} is volume-driven or basket-size-driven "
                    f"at \\${aov:,.0f} AOV."
                )
                a2 = (
                    f"  - Launch bundles targeting AOV > \\${aov*1.1:,.0f} (+10%) — "
                    f"applied to **{context.get('total_orders',0):,}** orders, "
                    f"that's a direct \\${aov*0.1*context.get('total_orders',0):,.0f} revenue lift."
                )

        else:  
            if situation in ("decline_severe", "decline_mild"):
                a1 = (
                    f"  - Break down margin by category in **{last_period}** vs **{best_period}** "
                    f"({best_period_label}) — **{context.get('lowest_margin_cat','lowest-margin category')}** "
                    f"at {context.get('lowest_margin_val',0):.1f}% is the priority to investigate."
                )
                a2 = (
                    f"  - Check whether average discount rose between **{best_period}** and "
                    f"**{last_period}** — if yes, discounting is the driver; "
                    f"if no, the issue is product mix or cost structure."
                )
            else:
                a1 = (
                    f"  - Cap discounts at 20% in **{context.get('lowest_margin_cat','lowest-margin category')}** "
                    f"({context.get('lowest_margin_val',0):.1f}% margin) — "
                    f"fastest lever to recover margin compression."
                )
                a2 = (
                    f"  - Compare category mix in **{best_period}** ({best_period_label}) vs "
                    f"**{last_period}** — identify whether highest-margin categories grew or "
                    f"shrank in share and adjust the sales focus accordingly."
                )

        return f"{a1}\n{a2}"