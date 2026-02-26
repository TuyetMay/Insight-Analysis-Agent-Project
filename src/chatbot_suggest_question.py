from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import copy


@dataclass(frozen=True)
class Suggestion:
    """A safe follow-up suggestion.

    - `text` is what you show in UI (chip/button).
    - `plan` is an OPTIONAL machine-readable payload you can send back to your backend
      to run without calling the LLM again.
    """
    text: str
    plan: Optional[Dict[str, Any]] = None


class SuggestionEngine:
    """Plan-aware suggestion engine (anti-hallucination).

    Design principles
    - Suggestions are derived from a *validated plan* (or dashboard state) using a small
      set of allowed transformations (relations).
    - Every suggested plan stays within allow-lists (metrics/dimensions/compare_period, etc.)
      so it can be executed safely.

    Typical usage
    - After you produce an `answer_payload` (or at least a validated `plan`),
      call `engine.suggest_from_plan(validated_plan, dashboard_defaults=...)`.
    - Render `Suggestion.text` as clickable chips.
    - On click, send `Suggestion.plan` back to backend and run it through your existing
      validate -> SQL generation -> execution pipeline (NO LLM needed).
    """

    def __init__(
        self,
        *,
        allowed_metrics: Optional[List[str]] = None,
        allowed_breakdowns: Optional[List[str]] = None,
        allowed_compare_periods: Optional[List[str]] = None,
        max_suggestions: int = 4,
    ) -> None:
        self.allowed_metrics = set(allowed_metrics or ["sales", "profit", "orders", "profit_margin"])
        self.allowed_breakdowns = set(allowed_breakdowns or ["region", "segment", "category", "sub_category"])
        self.allowed_compare_periods = set(allowed_compare_periods or ["prev_period", "mom", "yoy"])
        self.max_suggestions = max(1, int(max_suggestions))

    # -------------------------
    # Public API
    # -------------------------
    def suggest_from_plan(
        self,
        plan: Dict[str, Any],
        *,
        dashboard_defaults: Optional[Dict[str, Any]] = None,
        language: str = "vi",
    ) -> List[Suggestion]:
        """Generate follow-up suggestions from an already validated plan.

        `dashboard_defaults` (optional) can include:
          - start_date, end_date (if plan doesn't have)
          - filters (if plan doesn't have)
        """
        if not isinstance(plan, dict) or not plan.get("intent"):
            return self.suggest_from_dashboard_state(dashboard_defaults or {}, language=language)

        base = self._normalize_base_plan(plan, dashboard_defaults=dashboard_defaults or {})
        intent = base.get("intent")

        suggestions: List[Suggestion] = []

        # 1) Core transformations (context-aware)
        b = base.get("breakdown_by")

        # If the user already asked with a breakdown (e.g., profit by region),
        # keep suggestions on the same topic: rank/compare/trend. Avoid metric switching.
        if b:
            suggestions.extend(self._suggest_rank_if_breakdown_exists(base, language=language))
            suggestions.extend(self._suggest_compare(base, language=language))
            suggestions.extend(self._suggest_time_grain_variations(base, language=language))
        else:
            # No breakdown yet: suggest breakdowns and (optionally) metric switches.
            suggestions.extend(self._suggest_breakdowns(base, language=language))
            suggestions.extend(self._suggest_compare(base, language=language))
            suggestions.extend(self._suggest_time_grain_variations(base, language=language))
            suggestions.extend(self._suggest_metric_switch(base, language=language))

        # 2) Intent-specific transformations
        if intent == "kpi_value":
            # handled in core transformations above
            pass
        elif intent == "kpi_trend":
            suggestions.extend(self._suggest_breakdowns(base, language=language))
        elif intent == "kpi_rank":
            suggestions.extend(self._suggest_rank_variations(base, language=language))
            suggestions.extend(self._suggest_breakdowns(base, language=language))
        elif intent == "kpi_compare":
            # Show breakdown compare or trend compare
            suggestions.extend(self._suggest_breakdowns(base, language=language))
        # intent == "clarify": you likely won't suggest; user must answer clarifying question

        # De-duplicate by text
        uniq: List[Suggestion] = []
        seen = set()
        for s in suggestions:
            if not s.text or s.text in seen:
                continue
            seen.add(s.text)
            uniq.append(s)

        return uniq[: self.max_suggestions]

    def suggest_from_dashboard_state(self, state: Dict[str, Any], *, language: str = "vi") -> List[Suggestion]:
        """Fallback when you do not have a validated plan (e.g., Fast Path KPI answer).

        `state` can include:
          - last_metric: e.g. "profit"
          - filters: current dashboard filters
          - start_date/end_date: current dashboard date range
        """
        metric = (state or {}).get("last_metric") or "sales"
        if metric not in self.allowed_metrics:
            metric = "sales"

        base = {
            "intent": "kpi_value",
            "metrics": [metric],
            "time_grain": "none",
            "breakdown_by": None,
            "compare_period": None,
            "top_k": None,
            "order_by": metric,
            "start_date": (state or {}).get("start_date"),
            "end_date": (state or {}).get("end_date"),
            "filters": (state or {}).get("filters") or {"region": [], "segment": [], "category": []},
        }
        return self.suggest_from_plan(base, dashboard_defaults=state, language=language)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _normalize_base_plan(self, plan: Dict[str, Any], *, dashboard_defaults: Dict[str, Any]) -> Dict[str, Any]:
        """Make a best-effort base plan (assumes plan was validated upstream)."""
        p = copy.deepcopy(plan)
        p.setdefault("metrics", ["sales"])
        if isinstance(p["metrics"], str):
            p["metrics"] = [p["metrics"]]
        p["metrics"] = [m for m in p["metrics"] if m in self.allowed_metrics] or ["sales"]

        p.setdefault("time_grain", "none")
        p.setdefault("breakdown_by", None)
        if p["breakdown_by"] is not None and p["breakdown_by"] not in self.allowed_breakdowns:
            p["breakdown_by"] = None

        p.setdefault("compare_period", None)
        if p["compare_period"] is not None and p["compare_period"] not in self.allowed_compare_periods:
            p["compare_period"] = None

        p.setdefault("top_k", None)
        p.setdefault("order_by", p["metrics"][0])

        # Dates and filters
        p.setdefault("start_date", dashboard_defaults.get("start_date"))
        p.setdefault("end_date", dashboard_defaults.get("end_date"))
        p.setdefault("filters", dashboard_defaults.get("filters") or {"region": [], "segment": [], "category": []})
        return p

    def _mk_text(self, template_key: str, *, language: str, **kwargs: Any) -> str:
        """Very small i18n for suggestion phrasing."""
        metric = kwargs.get("metric")
        breakdown = kwargs.get("breakdown")
        grain = kwargs.get("grain")
        topk = kwargs.get("topk")
        compare = kwargs.get("compare")

        if language.lower().startswith("vi"):
            mapping = {
                "trend": f"Xu hướng {metric} theo {grain}",
                "breakdown": f"{metric} theo {breakdown}",
                "rank": f"Top {topk} {breakdown} theo {metric}",
                "compare": f"So sánh {metric} với kỳ {compare}",
                "switch_metric": f"Xem {metric} thay vì chỉ số hiện tại",
            }
        else:
            mapping = {
                "trend": f"Trend of {metric} by {grain}",
                "breakdown": f"{metric} by {breakdown}",
                "rank": f"Top {topk} {breakdown} by {metric}",
                "compare": f"Compare {metric} vs {compare}",
                "switch_metric": f"Switch metric to {metric}",
            }

        return mapping.get(template_key, "")

    def _clone_with(self, base: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
        p = copy.deepcopy(base)
        for k, v in updates.items():
            p[k] = v
        return p

    # -------------------------
    # Suggestion builders
    # -------------------------
    def _suggest_breakdowns(self, base: Dict[str, Any], *, language: str) -> List[Suggestion]:
        metric = base["metrics"][0]
        current = base.get("breakdown_by")
        out: List[Suggestion] = []

        # If already has breakdown, suggest drill-down if possible
        drill_map = {"category": "sub_category"}
        if current in drill_map and drill_map[current] in self.allowed_breakdowns:
            b = drill_map[current]
            p = self._clone_with(base, intent="kpi_value", breakdown_by=b, top_k=None)
            text = self._mk_text("breakdown", language=language, metric=metric.title(), breakdown=b.replace("_", " ").title())
            out.append(Suggestion(text=text, plan=p))

        # Otherwise propose common breakdowns not currently used
        for b in ["region", "segment", "category", "sub_category"]:
            if b not in self.allowed_breakdowns:
                continue
            if b == current:
                continue
            p = self._clone_with(base, intent="kpi_value", breakdown_by=b, top_k=None)
            text = self._mk_text("breakdown", language=language, metric=metric.title(), breakdown=b.replace("_", " ").title())
            out.append(Suggestion(text=text, plan=p))

        return out

    def _suggest_time_grain_variations(self, base: Dict[str, Any], *, language: str) -> List[Suggestion]:
        metric = base["metrics"][0]
        out: List[Suggestion] = []
        current = base.get("time_grain") or "none"
        # If not already a trend, suggest a monthly trend (safe default)
        for grain in ["month", "quarter", "year"]:
            if grain == current:
                continue
            p = self._clone_with(base, intent="kpi_trend", time_grain=grain, breakdown_by=None, top_k=None)
            text = self._mk_text("trend", language=language, metric=metric.title(), grain=grain.title())
            out.append(Suggestion(text=text, plan=p))
        return out

    def _suggest_compare(self, base: Dict[str, Any], *, language: str) -> List[Suggestion]:
        metric = base["metrics"][0]
        out: List[Suggestion] = []
        for c in ["yoy", "mom", "prev_period"]:
            if c not in self.allowed_compare_periods:
                continue
            if base.get("compare_period") == c:
                continue
            p = self._clone_with(base, intent="kpi_compare", compare_period=c, top_k=None)
            compare_label = {"yoy": "năm trước", "mom": "tháng trước", "prev_period": "kỳ trước"}.get(c, c)
            if not language.lower().startswith("vi"):
                compare_label = {"yoy": "YoY", "mom": "MoM", "prev_period": "previous period"}.get(c, c)
            text = self._mk_text("compare", language=language, metric=metric.title(), compare=compare_label)
            out.append(Suggestion(text=text, plan=p))
        return out

    def _suggest_rank_if_breakdown_exists(self, base: Dict[str, Any], *, language: str) -> List[Suggestion]:
        """If the user already asks by a breakdown, suggest top-k ranking."""
        metric = base["metrics"][0]
        b = base.get("breakdown_by")
        if not b:
            return []
        if b not in self.allowed_breakdowns:
            return []

        out: List[Suggestion] = []
        for k in (3, 5):
            p = self._clone_with(base, intent="kpi_rank", top_k=k, order_by=metric)
            text = self._mk_text(
                "rank",
                language=language,
                metric=metric.title(),
                breakdown=b.replace("_", " ").title(),
                topk=k,
            )
            out.append(Suggestion(text=text, plan=p))
        return out

    def _suggest_rank_variations(self, base: Dict[str, Any], *, language: str) -> List[Suggestion]:
        metric = base["metrics"][0]
        b = base.get("breakdown_by") or "sub_category"
        if b not in self.allowed_breakdowns:
            b = next(iter(self.allowed_breakdowns))
        out: List[Suggestion] = []
        for k in [3, 5, 10]:
            if base.get("top_k") == k:
                continue
            p = self._clone_with(base, intent="kpi_rank", top_k=k, order_by=metric, breakdown_by=b)
            text = self._mk_text("rank", language=language, metric=metric.title(), breakdown=b.replace("_", " ").title(), topk=k)
            out.append(Suggestion(text=text, plan=p))
        return out

    def _suggest_metric_switch(self, base: Dict[str, Any], *, language: str) -> List[Suggestion]:
        """Suggest switching to other core metrics while keeping filters/time/breakdown."""
        current = base["metrics"][0]
        candidates = ["sales", "profit", "orders", "profit_margin"]
        out: List[Suggestion] = []
        for m in candidates:
            if m not in self.allowed_metrics:
                continue
            if m == current:
                continue
            p = self._clone_with(base, metrics=[m], order_by=m)
            text = self._mk_text("switch_metric", language=language, metric=m.replace("_", " ").title())
            out.append(Suggestion(text=text, plan=p))
        return out
