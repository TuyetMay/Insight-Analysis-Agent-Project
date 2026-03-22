# chatbot/quick_insight.py
"""
Quick KPI insight handler.
Runs 4-5 sub-queries in parallel and assembles a structured answer
in the 📌🎯💡🚀 format.
"""
from __future__ import annotations
from typing import Any, Dict
import pandas as pd
from chatbot.sql_builder import SQLBuilder
from chatbot.insight_generator import InsightGenerator


def _detect_grain(filters: Dict[str, Any]) -> str:
    """
    Adaptive grain based on date range span:
      multi-year  → year
      single year → month
      single month → week
    """
    dr = filters.get("date_range")
    if not dr or len(dr) < 2:
        return "year"
    try:
        start = pd.Timestamp(dr[0])
        end   = pd.Timestamp(dr[1])
        days  = (end - start).days
        if days > 365:
            return "year"
        elif days > 31:
            return "month"
        else:
            return "week"
    except Exception:
        return "year"


def _grain_label(grain: str) -> str:
    return {"year": "year-over-year", "month": "month-over-month", "week": "week-over-week"}.get(grain, grain)


class QuickInsightHandler:
    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any],
                 filters: Dict[str, Any],
                 gemini_client: Any = None, model_name: str = "") -> None:
        self.df       = df
        self.kpis     = kpis
        self.filters  = filters
        self.sql      = SQLBuilder()
        self.insight  = InsightGenerator(gemini_client, model_name)
        self.grain    = _detect_grain(filters)
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
            "region":       list(f.get("region", []) or []),
            "segment":      list(f.get("segment", []) or []),
            "category":     list(f.get("category", []) or []),
            "sub_category": [],
        }

    def generate(self, kpi: str) -> str:
        """
        kpi: "sales" | "profit" | "orders" | "profit_margin"
        Returns formatted 📌🎯💡🚀 string.
        """
        handlers = {
            "sales":         self._sales_insight,
            "profit":        self._profit_insight,
            "orders":        self._orders_insight,
            "profit_margin": self._margin_insight,
        }
        fn = handlers.get(kpi)
        if not fn:
            return "❌ Unknown KPI."
        return fn()

    # ── Sales ─────────────────────────────────────────────────

    def _sales_insight(self) -> str:
        grain       = self.grain
        grain_lbl   = _grain_label(grain)
        base_filters = self._base_filters()

        # 1. Trend over time
        trend_plan = {
            "intent": "kpi_trend", "metrics": ["sales"],
            "time_grain": grain, "breakdown_by": None,
            "secondary_breakdown": None,
            "start_date": self._s0, "end_date": self._e0,
            "compare_period": None, "top_k": None,
            "order_by": "sales", "filters": base_filters,
            "show_extremes": False,
        }
        trend_df = self.sql.run(trend_plan)

        # 2. Top sub_category by sales
        top_cat_plan = {**trend_plan,
            "intent": "kpi_rank", "time_grain": "none",
            "breakdown_by": "sub_category", "top_k": 1}
        top_cat_df = self.sql.run(top_cat_plan)

        # 3. Top region by sales
        top_reg_plan = {**trend_plan,
            "intent": "kpi_rank", "time_grain": "none",
            "breakdown_by": "region", "top_k": 1}
        top_reg_df = self.sql.run(top_reg_plan)

        # ── Build output ───────────────────────────────────────
        total = float(self.kpis.get("total_sales", 0))
        lines = []

        # Trend data
        if not trend_df.empty and "period" in trend_df.columns:
            sdf = trend_df.sort_values("period").reset_index(drop=True)
            first_v = float(sdf.iloc[0]["sales"])
            last_v  = float(sdf.iloc[-1]["sales"])
            overall_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            word = "grown" if overall_chg >= 0 else "declined"
            n = len(sdf)

            # Period transitions
            transitions = []
            for i in range(1, n):
                prev_v = float(sdf.iloc[i-1]["sales"])
                curr_v = float(sdf.iloc[i]["sales"])
                pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                tag    = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
                transitions.append((pct, tag))

            # Strongest jump
            best = max(range(len(transitions)), key=lambda i: transitions[i][0])
            best_pct, best_tag = transitions[best]

            # ── Detect cause ───────────────────────────────────
            cause = _infer_cause(transitions)

            # ── Format ─────────────────────────────────────────
            tldr = f"Sales {word} **{abs(overall_chg):.0f}%** over {n} periods — " \
                   f"from **\\${first_v:,.0f}** to **\\${last_v:,.0f}**."

            lines += [
                f"📌 {tldr}",
                "",
                f"🎯 **Overview** ({grain_lbl}):",
            ]
            for i, (_, r) in enumerate(sdf.iterrows()):
                p  = _fmt_period(r.get("period", ""), grain)
                v  = float(r["sales"])
                if i == 0:
                    lines.append(f"  - {p}: **\\${v:,.0f}** *(baseline)*")
                else:
                    pct, tag = transitions[i-1]
                    lines.append(f"  - {p}: **\\${v:,.0f}** {tag} {pct:+.1f}%")

            # Insight
            lines += [
                "",
                "💡 **Insight:**",
            ]
            lines.append(f"  - The strongest growth was in **{_period_name(sdf, best+1, grain)}** "
             f"(**{best_pct:+.1f}%**). Root cause requires deeper analysis — "
             f"check order volume vs AOV to determine whether acquisition or upsell drove it.")
            if len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0]:
               lines.append(f"  - Growth rate slowed from **{transitions[-2][0]:+.1f}%** to "
                f"**{transitions[-1][0]:+.1f}%** — still strong, but worth monitoring "
                f"whether this reflects market saturation or a seasonal effect.")

            else:
                lines.append(f"  - Momentum is **sustained** across the last 2 periods — "
                             f"a positive signal for continued trajectory.")

        # Top product
        if not top_cat_df.empty:
            top_prod = top_cat_df.iloc[0].get("breakdown", "—")
            top_prod_v = float(top_cat_df.iloc[0].get("sales", 0))
            prod_share = (top_prod_v / total * 100) if total else 0
            lines.append(f"  - **{top_prod}** is the top sub-category "
                         f"(**\\${top_prod_v:,.0f}**, {prod_share:.0f}% of total).")

        # Top region
        if not top_reg_df.empty:
            top_reg = top_reg_df.iloc[0].get("breakdown", "—")
            top_reg_v = float(top_reg_df.iloc[0].get("sales", 0))
            reg_share = (top_reg_v / total * 100) if total else 0
            lines.append(f"  - **{top_reg}** leads all regions "
                         f"(**\\${top_reg_v:,.0f}**, {reg_share:.0f}% of total).")

        # Action
        transitions_text = "\n".join(
            f"  {_fmt_period(sdf.iloc[i-1].get('period',''), grain)} → "
            f"{_fmt_period(sdf.iloc[i].get('period',''), grain)}: "
            f"{transitions[i-1][0]:+.1f}%"
            for i in range(1, n)
        )

        action_context = {
            "period_range":         f"{_fmt_period(sdf.iloc[0].get('period',''), grain)} – {_fmt_period(sdf.iloc[-1].get('period',''), grain)}",
            "overall_change_pct":   overall_chg,
            "first_label":          _fmt_period(sdf.iloc[0].get("period", ""), grain),
            "first_value":          first_v,
            "last_label":           _fmt_period(sdf.iloc[-1].get("period", ""), grain),
            "last_value":           last_v,
            "transitions_text":     transitions_text,
            "top_product":          top_cat_df.iloc[0].get("breakdown", "—") if not top_cat_df.empty else "—",
            "top_product_value":    float(top_cat_df.iloc[0].get("sales", 0)) if not top_cat_df.empty else 0,
            "top_product_share":    (float(top_cat_df.iloc[0].get("sales", 0)) / total * 100) if (not top_cat_df.empty and total) else 0,
            "top_region":           top_reg_df.iloc[0].get("breakdown", "—") if not top_reg_df.empty else "—",
            "top_region_value":     float(top_reg_df.iloc[0].get("sales", 0)) if not top_reg_df.empty else 0,
            "top_region_share":     (float(top_reg_df.iloc[0].get("sales", 0)) / total * 100) if (not top_reg_df.empty and total) else 0,
            "is_decelerating":      len(transitions) >= 2 and transitions[-1][0] < transitions[-2][0],
            "last_transition_pct":  transitions[-1][0] if transitions else 0,
            "prev_transition_pct":  transitions[-2][0] if len(transitions) >= 2 else 0,
        }
        action_text = self._generate_action("sales", action_context)

        lines += [
            "",
            "🚀 **Action:**",
            action_text,
        ]


        return "\n".join(lines)

    # ── Profit ────────────────────────────────────────────────

    def _profit_insight(self) -> str:
        grain     = self.grain
        grain_lbl = _grain_label(grain)
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
        trend_df = self.sql.run(trend_plan)

        top_reg_plan = {**trend_plan,
            "intent": "kpi_rank", "time_grain": "none",
            "breakdown_by": "region", "top_k": 3}
        top_reg_df = self.sql.run(top_reg_plan)

        total_profit = float(self.kpis.get("total_profit", 0))
        total_sales  = float(self.kpis.get("total_sales", 1))
        overall_margin = (total_profit / total_sales * 100)

        lines = []

        if not trend_df.empty and "period" in trend_df.columns:
            sdf = trend_df.sort_values("period").reset_index(drop=True)
            first_v = float(sdf.iloc[0]["profit"])
            last_v  = float(sdf.iloc[-1]["profit"])
            overall_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            word = "grown" if overall_chg >= 0 else "declined"
            n = len(sdf)

            transitions = []
            for i in range(1, n):
                prev_v = float(sdf.iloc[i-1]["profit"])
                curr_v = float(sdf.iloc[i]["profit"])
                pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                tag    = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
                transitions.append((pct, tag))

            margin_health = "healthy" if overall_margin >= 12 else "tight"
            tldr = f"Profit {word} **{abs(overall_chg):.0f}%** {grain_lbl} — " \
                   f"margin currently at **{overall_margin:.1f}%** ({margin_health})."

            lines += [
                f"📌 {tldr}",
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

            # Negative profit check
            negatives = sdf[sdf["profit"] < 0]
            if not negatives.empty:
                lines.append(f"  - ⚠️ **{len(negatives)} period(s)** recorded negative profit — "
                             f"likely driven by high discounting or cost spikes.")
            else:
                lines.append(f"  - Profit remained **positive across all periods** — "
                             f"no structural loss-making detected.")

            # Margin signal
            if overall_margin < 10:
                lines.append(f"  - ⚠️ Margin of **{overall_margin:.1f}%** is below the 12% retail benchmark — "
                             f"review discount strategy and product mix.")
            elif overall_margin > 18:
                lines.append(f"  - Margin of **{overall_margin:.1f}%** is strong — "
                             f"pricing power is intact, focus on scaling volume.")
            else:
                lines.append(f"  - Margin of **{overall_margin:.1f}%** is within healthy range — "
                             f"targeting 15-18% is achievable with discount control.")

        # Top regions
        if not top_reg_df.empty:
            top_reg = top_reg_df.iloc[0].get("breakdown", "—")
            top_reg_v = float(top_reg_df.iloc[0].get("profit", 0))
            lines.append(f"  - **{top_reg}** leads profitability at **\\${top_reg_v:,.0f}**.")

        lines += [
            "",
            "🚀 **Action:**",
            "  - Identify sub-categories with discount > 20% and profit < 0 → reduce or restructure pricing.",
            f"  - Expand sales mix toward high-margin products to push overall margin above **15%**.",
        ]

        return "\n".join(lines)

    # ── Orders ────────────────────────────────────────────────

    def _orders_insight(self) -> str:
        grain     = self.grain
        grain_lbl = _grain_label(grain)
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

        lines = []

        if not trend_df.empty and "period" in trend_df.columns:
            sdf = trend_df.sort_values("period").reset_index(drop=True)
            first_v = float(sdf.iloc[0]["orders"])
            last_v  = float(sdf.iloc[-1]["orders"])
            overall_chg = (last_v - first_v) / abs(first_v) * 100 if first_v else 0
            word = "grown" if overall_chg >= 0 else "declined"
            n = len(sdf)

            transitions = []
            for i in range(1, n):
                prev_v = float(sdf.iloc[i-1]["orders"])
                curr_v = float(sdf.iloc[i]["orders"])
                pct    = (curr_v - prev_v) / abs(prev_v) * 100 if prev_v else 0
                tag    = "🚀" if pct >= 20 else ("📈" if pct >= 5 else ("📉" if pct < 0 else "➡️"))
                transitions.append((pct, tag))

            tldr = (f"Orders {word} **{abs(overall_chg):.0f}%** {grain_lbl} — "
                    f"**{total_orders:,}** total orders, avg **\\${aov:,.0f}** per order.")

            lines += [
                f"📌 {tldr}",
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

            # Volume vs value signal
            lines.append(f"  - AOV of **\\${aov:,.0f}** suggests customers buy " +
                         ("**high-value items** — cross-sell opportunity is strong." 
                          if aov > 400 else 
                          "**lower-value items** — bundle strategy could increase AOV."))

            # Trend pattern
            if len(transitions) >= 2:
                recent_avg = sum(t[0] for t in transitions[-2:]) / 2
                if recent_avg > 10:
                    lines.append(f"  - Recent momentum is **accelerating** — "
                                 f"last 2 periods averaged **+{recent_avg:.0f}%** growth.")
                elif recent_avg < -5:
                    lines.append(f"  - ⚠️ Recent momentum is **slowing** — "
                                 f"last 2 periods averaged **{recent_avg:.0f}%**. Check demand signals.")
                else:
                    lines.append(f"  - Order volume is **stable** — "
                                 f"growth is steady but not accelerating.")

        lines += [
            "",
            "🚀 **Action:**",
            f"  - If AOV is below target, launch **bundle/combo promotions** for top sub-categories.",
            "  - Identify seasonal peaks in order volume to time campaigns for maximum impact.",
        ]

        return "\n".join(lines)

    # ── Profit Margin ─────────────────────────────────────────

    def _margin_insight(self) -> str:
        grain     = self.grain
        grain_lbl = _grain_label(grain)
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
        trend_df = self.sql.run(trend_plan)

        top_cat_plan = {**trend_plan,
            "intent": "kpi_rank", "time_grain": "none",
            "metrics": ["profit_margin"],
            "breakdown_by": "category", "top_k": 3,
            "order_by": "profit_margin"}
        top_cat_df = self.sql.run(top_cat_plan)

        overall_margin = float(self.kpis.get("profit_margin", 0))
        benchmark = 12.0
        vs_benchmark = overall_margin - benchmark
        lines = []

        if not trend_df.empty and "period" in trend_df.columns:
            sdf = trend_df.sort_values("period").reset_index(drop=True)
            first_v = float(sdf.iloc[0]["profit_margin"])
            last_v  = float(sdf.iloc[-1]["profit_margin"])
            delta   = last_v - first_v
            word    = "improved" if delta >= 0 else "compressed"
            n = len(sdf)

            transitions = []
            for i in range(1, n):
                prev_v = float(sdf.iloc[i-1]["profit_margin"])
                curr_v = float(sdf.iloc[i]["profit_margin"])
                diff   = curr_v - prev_v  # pp change
                tag    = "📈" if diff >= 1 else ("📉" if diff <= -1 else "➡️")
                transitions.append((diff, tag))

            benchmark_str = ("above" if vs_benchmark >= 0 else "below")
            tldr = (f"Profit margin {word} from **{first_v:.1f}%** to **{last_v:.1f}%** "
                    f"({delta:+.1f}pp) — currently **{abs(vs_benchmark):.1f}pp {benchmark_str}** "
                    f"the 12% retail benchmark.")

            lines += [
                f"📌 {tldr}",
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

            # Compression signal
            if delta < -2:
                lines.append(f"  - ⚠️ Margin has **compressed {abs(delta):.1f}pp** — "
                             f"likely caused by rising discounts or shifting product mix toward lower-margin items.")
            elif delta > 2:
                lines.append(f"  - Margin has **expanded {delta:.1f}pp** — "
                             f"pricing improvements or mix shift toward higher-margin products is working.")
            else:
                lines.append(f"  - Margin is **relatively stable** ({delta:+.1f}pp change) — "
                             f"no major structural shift detected.")

        # Category margin breakdown
        if not top_cat_df.empty:
            cat_lines = []
            for _, r in top_cat_df.head(3).iterrows():
                cat  = r.get("breakdown", "—")
                marg = float(r.get("profit_margin", 0))
                cat_lines.append(f"**{cat}** ({marg:.1f}%)")
            lines.append(f"  - By category: {' > '.join(cat_lines)} — "
                         f"focus pricing efforts on the lowest-margin category.")

        lines += [
            "",
            "🚀 **Action:**",
            "  - Cap discounts at **20%** for categories where margin is already below 10%.",
            "  - Shift sales mix toward the highest-margin category to lift overall margin.",
        ]

        return "\n".join(lines)
    
    def _generate_action(self, kpi: str, context: dict) -> str:
        """
        Gọi Gemini để generate action recommendations dựa trên actual data context.
        Fallback về rule-based nếu Gemini fail hoặc không có client.
        """
        if not self.insight.client or not self.insight.model_name:
            return self._fallback_action(kpi, context)

        # Build prompt với đầy đủ context
        prompt = f"""You are a senior business analyst. Based on the following actual data, 
    write exactly 2 specific, actionable recommendations.

    === DATA CONTEXT ===
    KPI: {kpi.upper()}
    Period: {context.get('period_range', 'N/A')}
    Overall change: {context.get('overall_change_pct', 0):+.1f}% 
    ({context.get('first_label','')}: ${context.get('first_value',0):,.0f} → 
    {context.get('last_label','')}: ${context.get('last_value',0):,.0f})

    Growth transitions:
    {context.get('transitions_text', 'N/A')}

    Top sub-category: {context.get('top_product', 'N/A')} 
    (${context.get('top_product_value', 0):,.0f}, {context.get('top_product_share', 0):.0f}% of total)

    Top region: {context.get('top_region', 'N/A')} 
    (${context.get('top_region_value', 0):,.0f}, {context.get('top_region_share', 0):.0f}% of total)

    Deceleration detected: {context.get('is_decelerating', False)}
    Latest period growth: {context.get('last_transition_pct', 0):+.1f}%
    Prior period growth: {context.get('prev_transition_pct', 0):+.1f}%

    === RULES ===
    - Each action must reference specific numbers from the data above
    - No generic advice (e.g. "monitor performance" is NOT acceptable)
    - Focus on what to do NEXT based on what the data shows
    - Max 1 sentence per action, starting with a verb
    - Format: exactly 2 lines, each starting with "  - "

    Write 2 actions:"""

        from google.genai import types as genai_types
        try:
            resp = self.insight.client.models.generate_content(
                model=self.insight.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=150,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text and "  - " in text:
                return text
        except Exception:
            pass

        return self._fallback_action(kpi, context)


    def _fallback_action(self, kpi: str, context: dict) -> str:
        """Rule-based fallback — chỉ dùng khi Gemini fail."""
        top_region  = context.get("top_region", "top region")
        top_product = context.get("top_product", "top sub-category")
        is_decel    = context.get("is_decelerating", False)

        if kpi == "sales":
            a1 = (f"  - Investigate why growth slowed from "
                f"**{context.get('prev_transition_pct',0):+.1f}%** to "
                f"**{context.get('last_transition_pct',0):+.1f}%** — "
                f"compare order volume vs AOV in **{top_region}** to isolate the cause."
                if is_decel else
                f"  - Scale acquisition in **{top_region}** ({context.get('top_region_share',0):.0f}% "
                f"of sales) by replicating the strategy that drove the strongest growth period.")
            a2 = (f"  - Audit **{top_product}** ({context.get('top_product_share',0):.0f}% of total) "
                f"to check if its share is growing or flat — rising share = product-mix win, "
                f"flat = market-wide lift.")
        elif kpi == "profit":
            a1 = (f"  - Review discount policy for products in **{top_region}** where "
                f"margin pressure is highest — cap discounts at 20% for sub-categories below 10% margin.")
            a2 = (f"  - Shift sales mix toward **{top_product}** "
                f"({context.get('top_product_share',0):.0f}% of profit) to lift overall margin.")
        elif kpi == "orders":
            aov = context.get("aov", 0)
            a1 = (f"  - Launch bundle promotions for **{top_product}** to increase AOV "
                f"beyond current **${aov:,.0f}** — even a 10% AOV lift = significant revenue gain.")
            a2 = (f"  - Identify peak order months in **{top_region}** and concentrate "
                f"campaigns there to maximise conversion efficiency.")
        else:  # profit_margin
            a1 = (f"  - Cap discounts at 20% in the lowest-margin category to recover "
                f"at least 2-3pp of margin.")
            a2 = (f"  - Prioritise upsell toward **{top_product}** which shows highest "
                f"margin contribution — shift marketing budget accordingly.")

        return f"{a1}\n{a2}"


# ── Module helpers ────────────────────────────────────────────

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


def _infer_cause(transitions: list) -> str:
    """Simple heuristic to suggest a cause for the strongest jump."""
    if not transitions:
        return "investigate contributing factors"
    max_pct = max(t[0] for t in transitions)
    if max_pct >= 30:
        return "likely a new market, product launch, or seasonal surge"
    elif max_pct >= 15:
        return "possibly driven by expanded product mix or improved retention"
    else:
        return "consistent execution and demand growth"
    
