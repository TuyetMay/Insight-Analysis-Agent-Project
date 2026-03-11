"""
chatbot/plan_validator.py
Validates and normalises a raw plan dict from any NL parser tier.
Raises ValueError with a descriptive message on any schema violation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

_METRICS    = {"sales", "profit", "orders", "profit_margin"}
_GRAINS     = {"none", "week", "month", "quarter", "year"}
_BREAKDOWNS = {"region", "segment", "category", "sub_category"}
_COMPARES   = {"prev_period", "mom", "yoy"}
_INTENTS    = {"kpi_value", "kpi_trend", "kpi_rank", "kpi_compare", "clarify"}


class PlanValidator:
    """
    Stateless validator; instantiate once per chatbot session (holds allowed filter values).
    """

    def __init__(self, df_date_range: Tuple[str, str],
                 allowed_regions: List[str],
                 allowed_segments: List[str],
                 allowed_categories: List[str],
                 allowed_sub_categories: Optional[List[str]] = None) -> None:
        self._date_range = df_date_range
        self._allowed = {
            "region":       set(allowed_regions),
            "segment":      set(allowed_segments),
            "category":     set(allowed_categories),
            # sub_category is open — validated as non-empty strings only
            # (values come from df at runtime, not a fixed enum)
            "sub_category": set(allowed_sub_categories or []),
        }

    # ── Public ────────────────────────────────────────────────

    def validate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object.")

        intent = plan.get("intent")
        if intent not in _INTENTS:
            raise ValueError(f"Invalid intent: {intent!r}")

        if intent == "clarify":
            cq = (plan.get("clarifying_question") or "").strip()
            if not cq:
                raise ValueError("clarifying_question is required when intent='clarify'.")
            s0, e0 = self._date_range
            return {
                "intent": "clarify",
                "clarifying_question": cq,
                "metrics": ["sales"],
                "time_grain": "none", "breakdown_by": None,
                "compare_period": None, "top_k": None,
                "order_by": "sales",
                "start_date": s0, "end_date": e0,
                "filters": {"region": [], "segment": [], "category": [], "sub_category": []},
            }

        metrics = self._validated_metrics(plan.get("metrics"))
        time_grain = self._validated_grain(plan.get("time_grain", "none"))

        # kpi_value with a grain is really a trend
        if intent == "kpi_value" and time_grain != "none":
            intent = "kpi_trend"

        breakdown_by   = self._validated_breakdown(plan.get("breakdown_by"))
        compare_period = self._validated_compare(plan.get("compare_period"))
        top_k          = self._validated_top_k(plan.get("top_k"))
        order_by       = plan.get("order_by") or metrics[0]
        if order_by not in metrics:
            order_by = metrics[0]

        sd, ed = self._validated_dates(plan.get("start_date"), plan.get("end_date"))
        filters = self._validated_filters(plan.get("filters"))

        # Intent-specific constraints
        if intent == "kpi_rank":
            if not breakdown_by:
                raise ValueError("breakdown_by is required for intent='kpi_rank'.")
            if top_k is None:
                raise ValueError("top_k is required for intent='kpi_rank'.")
            if time_grain != "none":
                raise ValueError("kpi_rank only supports time_grain='none'.")
        if intent == "kpi_compare":
            if compare_period is None:
                raise ValueError("compare_period is required for intent='kpi_compare'.")
            if len(metrics) != 1:
                raise ValueError("kpi_compare supports exactly 1 metric.")

        return {
            "intent": intent, "metrics": metrics,
            "time_grain": time_grain, "breakdown_by": breakdown_by,
            "start_date": sd, "end_date": ed,
            "compare_period": compare_period, "top_k": top_k,
            "order_by": order_by, "filters": filters,
        }

    # ── Field validators ─────────────────────────────────────

    def _validated_metrics(self, raw: Any) -> List[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise ValueError("metrics must be a non-empty array.")
        metrics = [str(m) for m in raw]
        if len(metrics) > 2:
            raise ValueError("metrics supports at most 2 items.")
        for m in metrics:
            if m not in _METRICS:
                raise ValueError(f"Invalid metric: {m!r}")
        return metrics

    @staticmethod
    def _validated_grain(raw: Any) -> str:
        grain = str(raw or "none")
        if grain not in _GRAINS:
            raise ValueError(f"Invalid time_grain: {grain!r}")
        return grain

    @staticmethod
    def _validated_breakdown(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        bd = str(raw)
        if bd not in _BREAKDOWNS:
            raise ValueError(f"Invalid breakdown_by: {bd!r}")
        return bd

    @staticmethod
    def _validated_compare(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        cp = str(raw)
        if cp not in _COMPARES:
            raise ValueError(f"Invalid compare_period: {cp!r}")
        return cp

    @staticmethod
    def _validated_top_k(raw: Any) -> Optional[int]:
        if raw is None:
            return None
        try:
            k = int(raw)
        except Exception:
            raise ValueError("top_k must be an integer or null.")
        if not (1 <= k <= 50):
            raise ValueError("top_k must be between 1 and 50.")
        return k

    def _validated_dates(self, start: Any, end: Any) -> Tuple[str, str]:
        s0, e0 = self._date_range
        start = str(start or s0)
        end   = str(end   or e0)
        sd = self._parse_date(start)
        ed = self._parse_date(end)
        if sd is None or ed is None:
            raise ValueError("start_date / end_date must be YYYY-MM-DD.")
        if sd > ed:
            raise ValueError("start_date must be ≤ end_date.")
        return sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d")

    def _validated_filters(self, raw: Any) -> Dict[str, List[str]]:
        if not isinstance(raw, dict):
            raise ValueError("filters must be an object.")
        result: Dict[str, List[str]] = {}

        # ✅ FIX: added sub_category to filter dimensions
        for dim in ("region", "segment", "category", "sub_category"):
            vals = raw.get(dim) or []
            if not isinstance(vals, list):
                raise ValueError(f"filters.{dim} must be an array.")
            vals = [str(v) for v in vals]

            # region/segment/category: validate against known allowed set
            # sub_category: allow any non-empty string (values are dynamic from df)
            if dim != "sub_category":
                bad = sorted(set(vals) - self._allowed[dim])
                if bad:
                    raise ValueError(f"filters.{dim} contains unknown values: {bad}")
            result[dim] = vals
        return result

    @staticmethod
    def _parse_date(s: str) -> Optional[date]:
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date()
        except Exception:
            return None