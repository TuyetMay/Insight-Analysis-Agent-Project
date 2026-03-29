# chatbot/suggestions/rule_engine.py — PATCHED v3
"""
FIX-10: Grammar "the the" bug
  OLD: period_verb="contributed to the" + template "the {best_period}" = "the the 2016"
  NEW: template never adds "the" — period_verb owns all articles

FIX-11: Decline context uses worst_period, not best_period
  OLD: "Which region drove the 2015 spike?" when 2015 was -2.8% (not a spike, not worst)
  NEW: For decline situations, reference worst_period (the period with the most negative transition)

FIX-12: Partial period context awareness in suggestions
  NEW: When has_partial_period=True, suggestion 3 is "Compare same months YoY"
       instead of a generic YoY comparison
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set

from chatbot.suggestions.models import Suggestion


class RuleBasedSuggestionEngine:
    _METRICS:         Set[str] = {"sales", "profit", "orders", "profit_margin"}
    _BREAKDOWNS:      Set[str] = {"region", "segment", "category", "sub_category"}
    _COMPARE_PERIODS: Set[str] = {"prev_period", "mom", "yoy"}

    _METRIC_LABELS = {
        "sales":         "Sales",
        "profit":        "Profit",
        "orders":        "Orders",
        "profit_margin": "Profit Margin",
    }
    _DIM_LABELS = {
        "region":       "Region",
        "segment":      "Segment",
        "category":     "Category",
        "sub_category": "Sub-Category",
    }
    _GRAIN_LABELS = {
        "month": "Month", "quarter": "Quarter",
        "year": "Year",   "week": "Week",
    }
    _COMPARE_LABELS = {
        "yoy":         "YoY (vs last year)",
        "mom":         "MoM (vs last month)",
        "prev_period": "vs previous period",
    }

    def __init__(self, *, allowed_metrics: Optional[List[str]] = None,
                 allowed_breakdowns: Optional[List[str]] = None,
                 allowed_compare_periods: Optional[List[str]] = None,
                 max_suggestions: int = 4) -> None:
        self.allowed_metrics         = set(allowed_metrics         or self._METRICS)
        self.allowed_breakdowns      = set(allowed_breakdowns      or self._BREAKDOWNS)
        self.allowed_compare_periods = set(allowed_compare_periods or self._COMPARE_PERIODS)
        self.max_suggestions         = max(1, int(max_suggestions))

    def suggest(self, plan: Dict[str, Any],
                dashboard_defaults: Optional[Dict[str, Any]] = None) -> List[Suggestion]:
        if not isinstance(plan, dict) or not plan.get("intent"):
            return self._fallback(dashboard_defaults)

        base = self._normalize(plan, dashboard_defaults or {})
        qctx = plan.get("_quick_context")
        if qctx:
            return self._dedup(self._analyst_suggestions(base, qctx))

        if base.get("breakdown_by") and base.get("secondary_breakdown"):
            return self._dedup(self._cross_breakdown_suggestions(base))

        b = base.get("breakdown_by")
        candidates: List[Suggestion] = []
        if b:
            candidates += self._rank_from_breakdown(base)
            candidates += self._compare(base)
            candidates += self._time_grains(base)
        else:
            candidates += self._breakdowns(base)
            candidates += self._compare(base)
            candidates += self._time_grains(base)
            candidates += self._metric_switch(base)

        if base.get("intent") == "kpi_trend":
            candidates += self._breakdowns(base)
        elif base.get("intent") == "kpi_rank":
            candidates += self._rank_variations(base)

        return self._dedup(candidates)

    # ── Analyst suggestions for Quick Insight ─────────────────

    def _analyst_suggestions(self, base: Dict[str, Any],
                              ctx: Dict[str, Any]) -> List[Suggestion]:
        kpi           = ctx.get("kpi", "sales")
        m             = base["metrics"][0]
        best_period   = ctx.get("best_period", "")
        worst_period  = ctx.get("worst_period", "")           # FIX-11
        top_region    = ctx.get("top_region", "")
        top_product   = ctx.get("top_product", "")
        is_decel      = ctx.get("is_decelerating", False)
        overall_chg   = ctx.get("overall_change_pct", 0.0)
        last_pct      = ctx.get("last_transition_pct", 0.0)
        has_partial   = ctx.get("has_partial_period", False)  # FIX-12

        # ── FIX-10 + FIX-11: Build period event phrase ────────
        # Rules:
        #   1. No double "the" — the phrase must include its own articles
        #   2. For decline context, reference worst_period (biggest drop)
        #      not best_period (which is the "least bad" period)
        #   3. Threshold for "spike" = overall_chg >= 20%, not just positive
        #
        # Build a complete phrase: "drove the 2016 growth spike"
        # instead of "drove" + "the" + "2016" + "spike" (template fragmentation = bugs)

        if overall_chg >= 20:
            reference_period = best_period
            period_phrase    = f"drove the {reference_period} growth spike"
        elif overall_chg >= 5:
            reference_period = best_period
            period_phrase    = f"drove the {reference_period} growth"
        elif overall_chg >= -5:
            reference_period = best_period or worst_period
            period_phrase    = f"drove the {reference_period} change"
        elif overall_chg >= -40:
            # Use worst_period (the period with the biggest drop) for decline
            reference_period = worst_period or best_period
            period_phrase    = f"contributed to the {reference_period} decline"
        else:
            # Severe decline — worst_period is where the drop happened
            reference_period = worst_period or best_period
            period_phrase    = f"contributed to the {reference_period} drop"

        suggs: List[Suggestion] = []

        # ── 1. Region breakdown — context-aware phrase ─────────
        if reference_period:
            suggs.append(Suggestion(
                f"Which region {period_phrase}?",
                self._clone(base, intent="kpi_rank", breakdown_by="region",
                            top_k=5, secondary_breakdown=None, time_grain="none"),
            ))
        else:
            suggs.append(Suggestion(
                f"Top 5 regions by {self._lm(m)}",
                self._clone(base, intent="kpi_rank", breakdown_by="region",
                            top_k=5, secondary_breakdown=None, time_grain="none"),
            ))

        # ── 2. Sub-category drill-down — context-aware ─────────
        if overall_chg < -10:
            suggs.append(Suggestion(
                f"Which sub-categories drove the {self._lm(m)} drop?",
                self._clone(base, intent="kpi_rank", breakdown_by="sub_category",
                            top_k=5, secondary_breakdown=None, time_grain="none"),
            ))
        elif best_period and top_product:
            suggs.append(Suggestion(
                f"Top sub-categories by {self._lm(m)} in {best_period}",
                self._clone(base, intent="kpi_rank", breakdown_by="sub_category",
                            top_k=5, secondary_breakdown=None, time_grain="none"),
            ))
        else:
            suggs.append(Suggestion(
                f"Top 5 sub-categories by {self._lm(m)}",
                self._clone(base, intent="kpi_rank", breakdown_by="sub_category",
                            top_k=5, secondary_breakdown=None, time_grain="none"),
            ))

        # ── 3. FIX-12: Partial-period aware comparison ─────────
        if has_partial and best_period:
            # When last period is partial, the most useful next question
            # is a like-for-like comparison (same months, different year)
            suggs.append(Suggestion(
                f"{self._lm(m)} — same date range last year",
                self._clone(base, intent="kpi_compare", compare_period="yoy",
                            breakdown_by=None, secondary_breakdown=None, metrics=[m]),
            ))
        elif overall_chg < -10:
            suggs.append(Suggestion(
                f"{self._lm(m)} — YoY comparison",
                self._clone(base, intent="kpi_compare", compare_period="yoy",
                            breakdown_by=None, secondary_breakdown=None, metrics=[m]),
            ))
        elif is_decel and best_period:
            suggs.append(Suggestion(
                "Orders trend — is volume also slowing?",
                self._clone(base, intent="kpi_trend", time_grain="year",
                            metrics=["orders"], order_by="orders",
                            breakdown_by=None, secondary_breakdown=None),
            ))
        elif kpi in ("sales", "profit") and overall_chg > 10:
            suggs.append(Suggestion(
                f"{self._lm(m)} by region — full breakdown",
                self._clone(base, intent="kpi_value", breakdown_by="region",
                            secondary_breakdown=None, time_grain="none"),
            ))
        else:
            suggs.append(Suggestion(
                f"{self._lm(m)} — YoY comparison",
                self._clone(base, intent="kpi_compare", compare_period="yoy",
                            breakdown_by=None, secondary_breakdown=None, metrics=[m]),
            ))

        # ── 4. Complementary metric ────────────────────────────
        if kpi == "sales":
            lbl = "is profitability also declining?" if overall_chg < -10 else "is growth profitable?"
            suggs.append(Suggestion(
                f"Profit margin by region — {lbl}",
                self._clone(base, intent="kpi_value", breakdown_by="region",
                            metrics=["profit_margin"], order_by="profit_margin",
                            secondary_breakdown=None, time_grain="none"),
            ))
        elif kpi == "profit":
            suggs.append(Suggestion(
                "Which sub-categories are loss-making?",
                self._clone(base, intent="kpi_value", breakdown_by="sub_category",
                            metrics=["profit"], order_by="profit",
                            secondary_breakdown=None, time_grain="none"),
            ))
        elif kpi == "orders":
            suggs.append(Suggestion(
                "Sales by region — does AOV vary by region?",
                self._clone(base, intent="kpi_value", breakdown_by="region",
                            metrics=["sales"], order_by="sales",
                            secondary_breakdown=None, time_grain="none"),
            ))
        elif kpi == "profit_margin":
            suggs.append(Suggestion(
                "Profit by sub-category — worst margin offenders",
                self._clone(base, intent="kpi_rank", breakdown_by="sub_category",
                            metrics=["profit"], order_by="profit",
                            top_k=10, secondary_breakdown=None, time_grain="none"),
            ))
        else:
            suggs.append(Suggestion(
                f"{self._lm(m)} by category",
                self._clone(base, intent="kpi_value", breakdown_by="category",
                            secondary_breakdown=None, time_grain="none"),
            ))

        return suggs

    # ── Private helpers (unchanged from v2) ───────────────────

    def _lm(self, m: str) -> str:
        return self._METRIC_LABELS.get(m, m.replace("_", " ").title())

    def _ld(self, d: str) -> str:
        return self._DIM_LABELS.get(d, d.replace("_", " ").title())

    def _lg(self, g: str) -> str:
        return self._GRAIN_LABELS.get(g, g.title())

    def _lc(self, c: str) -> str:
        return self._COMPARE_LABELS.get(c, c)

    def _fallback(self, defaults: Optional[Dict[str, Any]]) -> List[Suggestion]:
        metric = (defaults or {}).get("last_metric") or "sales"
        if metric not in self.allowed_metrics:
            metric = "sales"
        base = {
            "intent": "kpi_value", "metrics": [metric], "time_grain": "none",
            "breakdown_by": None, "compare_period": None, "top_k": None,
            "order_by": metric,
            "start_date": (defaults or {}).get("start_date"),
            "end_date":   (defaults or {}).get("end_date"),
            "filters": (defaults or {}).get("filters") or
                       {"region": [], "segment": [], "category": []},
        }
        return self.suggest(base, defaults)

    def _normalize(self, plan: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
        p = copy.deepcopy(plan)
        metrics = p.get("metrics", ["sales"])
        if isinstance(metrics, str):
            metrics = [metrics]
        p["metrics"] = [m for m in metrics if m in self.allowed_metrics] or ["sales"]
        p.setdefault("time_grain", "none")
        bd = p.get("breakdown_by")
        p["breakdown_by"] = bd if bd in self.allowed_breakdowns else None
        p.setdefault("compare_period", None)
        p.setdefault("top_k", None)
        p.setdefault("order_by", p["metrics"][0])
        p.setdefault("start_date", defaults.get("start_date"))
        p.setdefault("end_date",   defaults.get("end_date"))
        p.setdefault("filters",
                     defaults.get("filters") or {"region": [], "segment": [], "category": []})
        return p

    @staticmethod
    def _clone(base: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
        p = copy.deepcopy(base)
        p.pop("_quick_context", None)
        p.update(updates)
        return p

    def _dedup(self, candidates: List[Suggestion]) -> List[Suggestion]:
        seen: set = set()
        result: List[Suggestion] = []
        for s in candidates:
            if s.text and s.text not in seen:
                seen.add(s.text)
                result.append(s)
            if len(result) >= self.max_suggestions:
                break
        return result

    def _breakdowns(self, base: Dict[str, Any]) -> List[Suggestion]:
        m, current = base["metrics"][0], base.get("breakdown_by")
        return [
            Suggestion(f"{self._lm(m)} by {self._ld(b)}",
                       self._clone(base, intent="kpi_value", breakdown_by=b, top_k=None))
            for b in ["region", "segment", "category", "sub_category"]
            if b in self.allowed_breakdowns and b != current
        ]

    def _time_grains(self, base: Dict[str, Any]) -> List[Suggestion]:
        m, current = base["metrics"][0], base.get("time_grain") or "none"
        return [
            Suggestion(f"{self._lm(m)} trend by {self._lg(grain)}",
                       self._clone(base, intent="kpi_trend", time_grain=grain,
                                   breakdown_by=None, top_k=None))
            for grain in ["month", "quarter", "year"] if grain != current
        ]

    def _compare(self, base: Dict[str, Any]) -> List[Suggestion]:
        m = base["metrics"][0]
        return [
            Suggestion(f"{self._lm(m)} — {self._lc(c)}",
                       self._clone(base, intent="kpi_compare", compare_period=c,
                                   top_k=None, metrics=[m]))
            for c in ["yoy", "mom", "prev_period"]
            if c in self.allowed_compare_periods and base.get("compare_period") != c
        ]

    def _rank_from_breakdown(self, base: Dict[str, Any]) -> List[Suggestion]:
        m, b = base["metrics"][0], base.get("breakdown_by")
        if not b or b not in self.allowed_breakdowns:
            return []
        return [
            Suggestion(f"Top {k} {self._ld(b)} by {self._lm(m)}",
                       self._clone(base, intent="kpi_rank", top_k=k, order_by=m))
            for k in (3, 5)
        ]

    def _rank_variations(self, base: Dict[str, Any]) -> List[Suggestion]:
        m = base["metrics"][0]
        b = base.get("breakdown_by") or "sub_category"
        return [
            Suggestion(f"Top {k} {self._ld(b)} by {self._lm(m)}",
                       self._clone(base, intent="kpi_rank", top_k=k, order_by=m, breakdown_by=b))
            for k in (3, 5, 10) if base.get("top_k") != k
        ]

    def _metric_switch(self, base: Dict[str, Any]) -> List[Suggestion]:
        current = base["metrics"][0]
        return [
            Suggestion(f"View {self._lm(m)}", self._clone(base, metrics=[m], order_by=m))
            for m in ["sales", "profit", "orders", "profit_margin"]
            if m in self.allowed_metrics and m != current
        ]

    def _cross_breakdown_suggestions(self, base: Dict[str, Any]) -> List[Suggestion]:
        m  = base["metrics"][0]
        b1 = base.get("breakdown_by") or "region"
        b2 = base.get("secondary_breakdown") or "category"
        return [
            Suggestion(f"Top 5 {self._ld(b1)} by {self._lm(m)}",
                       self._clone(base, intent="kpi_rank", breakdown_by=b1,
                                   top_k=5, secondary_breakdown=None)),
            Suggestion(f"{self._lm(m)} by {self._ld(b1)}",
                       self._clone(base, intent="kpi_value", breakdown_by=b1,
                                   secondary_breakdown=None)),
            Suggestion(f"{self._lm(m)} by {self._ld(b2)}",
                       self._clone(base, intent="kpi_value", breakdown_by=b2,
                                   secondary_breakdown=None)),
            Suggestion(f"{self._lm(m)} — YoY (vs last year)",
                       self._clone(base, intent="kpi_compare", compare_period="yoy",
                                   breakdown_by=b1, secondary_breakdown=None, metrics=[m])),
        ]