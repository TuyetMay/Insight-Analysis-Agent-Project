"""
chatbot/suggestions/rule_engine.py
Deterministic suggestion engine — no LLM required.
Derives follow-up suggestions by applying fixed transformations to the last plan.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set

from chatbot.suggestions.models import Suggestion


class RuleBasedSuggestionEngine:
    _METRICS:         Set[str] = {"sales", "profit", "orders", "profit_margin"}
    _BREAKDOWNS:      Set[str] = {"region", "segment", "category", "sub_category"}
    _COMPARE_PERIODS: Set[str] = {"prev_period", "mom", "yoy"}

    _METRIC_LABELS  = {"sales": "Sales", "profit": "Profit", "orders": "Orders",
                       "profit_margin": "Profit Margin"}
    _DIM_LABELS     = {"region": "Region", "segment": "Segment",
                       "category": "Category", "sub_category": "Sub-Category"}
    _GRAIN_LABELS   = {"month": "Month", "quarter": "Quarter", "year": "Year", "week": "Week"}
    _COMPARE_LABELS = {"yoy": "YoY (vs last year)", "mom": "MoM (vs last month)",
                       "prev_period": "vs previous period"}

    def __init__(self, *, allowed_metrics: Optional[List[str]] = None,
                 allowed_breakdowns: Optional[List[str]] = None,
                 allowed_compare_periods: Optional[List[str]] = None,
                 max_suggestions: int = 4) -> None:
        self.allowed_metrics  = set(allowed_metrics  or self._METRICS)
        self.allowed_breakdowns = set(allowed_breakdowns or self._BREAKDOWNS)
        self.allowed_compare_periods = set(allowed_compare_periods or self._COMPARE_PERIODS)
        self.max_suggestions = max(1, int(max_suggestions))

    # ── Public ────────────────────────────────────────────────

    def suggest(self, plan: Dict[str, Any],
                dashboard_defaults: Optional[Dict[str, Any]] = None) -> List[Suggestion]:
        if not isinstance(plan, dict) or not plan.get("intent"):
            return self._fallback(dashboard_defaults)

        base = self._normalize(plan, dashboard_defaults or {})
        b    = base.get("breakdown_by")

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

    # ── Private helpers ───────────────────────────────────────

    def _lm(self, m: str) -> str: return self._METRIC_LABELS.get(m, m.replace("_", " ").title())
    def _ld(self, d: str) -> str: return self._DIM_LABELS.get(d, d.replace("_", " ").title())
    def _lg(self, g: str) -> str: return self._GRAIN_LABELS.get(g, g.title())
    def _lc(self, c: str) -> str: return self._COMPARE_LABELS.get(c, c)

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
            "filters": (defaults or {}).get("filters") or {"region": [], "segment": [], "category": []},
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
        p.setdefault("filters", defaults.get("filters") or {"region": [], "segment": [], "category": []})
        return p

    @staticmethod
    def _clone(base: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
        p = copy.deepcopy(base)
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
                       self._clone(base, intent="kpi_trend", time_grain=grain, breakdown_by=None, top_k=None))
            for grain in ["month", "quarter", "year"] if grain != current
        ]

    def _compare(self, base: Dict[str, Any]) -> List[Suggestion]:
        m = base["metrics"][0]
        return [
            Suggestion(f"Compare {self._lm(m)} {self._lc(c)}",
                       self._clone(base, intent="kpi_compare", compare_period=c, top_k=None, metrics=[m]))
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
        m  = base["metrics"][0]
        b  = base.get("breakdown_by") or "sub_category"
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
