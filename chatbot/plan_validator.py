"""
chatbot/plan_validator.py — PATCHED

FIX-PV-1: Thêm redirect cho kpi_distribution → kpi_rank
           và kpi_correlation → kpi_value để tránh lỗi "Invalid intent"
FIX-PV-2: Thêm kpi_distribution vào _INTENTS để hỗ trợ đầy đủ
FIX-PV-3: Validate kpi_distribution plan → trả về plan hợp lệ cho SQLBuilder
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

_METRICS    = {"sales", "profit", "orders", "profit_margin"}
_GRAINS     = {"none", "week", "month", "quarter", "year"}
_BREAKDOWNS = {"region", "segment", "category", "sub_category", "state"}
_COMPARES   = {"prev_period", "mom", "yoy"}
_INTENTS    = {
    "kpi_value", "kpi_trend", "kpi_rank", "kpi_compare",
    "kpi_detail", "kpi_distribution", "kpi_correlation", "clarify",
}
_CONDITIONS = {"profit_negative", "profit_positive", "high_discount", "loss_orders"}

# FIX-PV-1: Redirect unsupported/legacy intents
_INTENT_REDIRECTS: Dict[str, str] = {
    # Nếu SQLBuilder chưa hỗ trợ, redirect về intent tương đương
    # Hiện tại sql_builder.py đã có _run_distribution() và _run_correlation()
    # nên không cần redirect — nhưng giữ lại làm safety net
}


class PlanValidator:
    """
    Stateless validator; instantiate once per chatbot session.
    PATCHED: hỗ trợ kpi_distribution và kpi_correlation.
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
            "sub_category": set(allowed_sub_categories or []),
        }

    # ── Public ────────────────────────────────────────────────

    def validate(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object.")

        intent = plan.get("intent")

        # FIX-PV-1: Safety redirect cho intent không hợp lệ
        if intent not in _INTENTS:
            # Thử redirect trước
            redirected = _INTENT_REDIRECTS.get(intent)
            if redirected:
                plan = dict(plan)
                plan["intent"] = redirected
                intent = redirected
            else:
                # Fallback thông minh dựa vào nội dung plan
                intent = self._infer_fallback_intent(plan)
                plan = dict(plan)
                plan["intent"] = intent

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

        # ── kpi_detail ────────────────────────────────────────
        if intent == "kpi_detail":
            sd, ed = self._validated_dates(plan.get("start_date"), plan.get("end_date"))
            filters = self._validated_filters(plan.get("filters") or {})
            condition = str(plan.get("condition") or "profit_negative")
            if condition not in _CONDITIONS:
                condition = "profit_negative"
            top_k = self._validated_top_k(plan.get("top_k") or 15)
            return {
                "intent": "kpi_detail",
                "condition": condition,
                "metrics": ["sales", "profit"],
                "time_grain": "none",
                "breakdown_by": plan.get("breakdown_by") or "sub_category",
                "start_date": sd, "end_date": ed,
                "compare_period": None,
                "top_k": top_k,
                "order_by": "profit",
                "filters": filters,
            }

        # ── kpi_distribution (FIX-PV-2) ───────────────────────
        if intent == "kpi_distribution":
            sd, ed = self._validated_dates(plan.get("start_date"), plan.get("end_date"))
            filters = self._validated_filters(plan.get("filters") or {})
            metrics = self._validated_metrics(plan.get("metrics") or ["profit_margin"])
            breakdown = self._validated_breakdown(plan.get("breakdown_by"))

            # dist_metric: metric để bucket (discount, sales, profit_margin)
            dist_metric = plan.get("dist_metric") or metrics[0]
            if dist_metric not in {"discount", "sales", "profit_margin", "profit"}:
                dist_metric = metrics[0]

            return {
                "intent": "kpi_distribution",
                "dist_metric": dist_metric,
                "metrics": metrics,
                "time_grain": "none",
                "breakdown_by": breakdown,
                "secondary_breakdown": None,
                "start_date": sd, "end_date": ed,
                "compare_period": None,
                "top_k": None,
                "order_by": metrics[0],
                "filters": filters,
                "show_extremes": False,
            }

        # ── kpi_correlation (FIX-PV-2) ────────────────────────
        if intent == "kpi_correlation":
            sd, ed = self._validated_dates(plan.get("start_date"), plan.get("end_date"))
            filters = self._validated_filters(plan.get("filters") or {})
            metrics = self._validated_metrics(plan.get("metrics") or ["sales", "profit"])
            breakdown = self._validated_breakdown(plan.get("breakdown_by")) or "sub_category"

            x_metric = plan.get("x_metric", "discount")
            y_metric = plan.get("y_metric", metrics[0])

            return {
                "intent": "kpi_correlation",
                "metrics": metrics,
                "x_metric": x_metric,
                "y_metric": y_metric,
                "time_grain": "none",
                "breakdown_by": breakdown,
                "secondary_breakdown": None,
                "start_date": sd, "end_date": ed,
                "compare_period": None,
                "top_k": None,
                "order_by": metrics[0],
                "filters": filters,
                "show_extremes": False,
            }

        # ── Standard intents ───────────────────────────────────
        metrics = self._validated_metrics(plan.get("metrics"))
        time_grain = self._validated_grain(plan.get("time_grain", "none"))

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

        if intent == "kpi_rank":
            if not breakdown_by:
                breakdown_by = "sub_category"   # FIX: default thay vì raise
            if top_k is None:
                top_k = 10                       # FIX: default thay vì raise
            if time_grain != "none":
                time_grain = "none"              # FIX: reset thay vì raise

        if intent == "kpi_compare":
            if compare_period is None:
                compare_period = "yoy"           # FIX: default thay vì raise
            if len(metrics) != 1:
                metrics = [metrics[0]]           # FIX: trim thay vì raise

        return {
            "intent": intent, "metrics": metrics,
            "time_grain": time_grain, "breakdown_by": breakdown_by,
            "secondary_breakdown": plan.get("secondary_breakdown"),
            "start_date": sd, "end_date": ed,
            "compare_period": compare_period, "top_k": top_k,
            "order_by": order_by, "filters": filters,
            "show_extremes": bool(plan.get("show_extremes", False)),
        }

    # ── Fallback intent inference ─────────────────────────────

    @staticmethod
    def _infer_fallback_intent(plan: Dict[str, Any]) -> str:
        """
        Khi intent không hợp lệ, infer intent tốt nhất từ các field trong plan.
        """
        has_breakdown = bool(plan.get("breakdown_by"))
        has_top_k     = plan.get("top_k") is not None
        has_grain     = plan.get("time_grain") not in (None, "none")
        has_compare   = plan.get("compare_period") is not None
        has_condition = plan.get("condition") is not None

        if has_condition:
            return "kpi_detail"
        if has_compare:
            return "kpi_compare"
        if has_grain:
            return "kpi_trend"
        if has_top_k and has_breakdown:
            return "kpi_rank"
        if has_breakdown:
            return "kpi_value"
        return "kpi_value"

    # ── Field validators ──────────────────────────────────────

    def _validated_metrics(self, raw: Any) -> List[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            return ["sales"]    # FIX: default thay vì raise
        metrics = [str(m) for m in raw]
        if len(metrics) > 2:
            metrics = metrics[:2]
        valid = [m for m in metrics if m in _METRICS]
        if not valid:
            return ["sales"]    # FIX: default thay vì raise
        return valid

    @staticmethod
    def _validated_grain(raw: Any) -> str:
        grain = str(raw or "none")
        if grain not in _GRAINS:
            return "none"       # FIX: default thay vì raise
        return grain

    @staticmethod
    def _validated_breakdown(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        bd = str(raw)
        if bd not in _BREAKDOWNS:
            return None         # FIX: None thay vì raise
        return bd

    @staticmethod
    def _validated_compare(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        cp = str(raw)
        if cp not in _COMPARES:
            return None         # FIX: None thay vì raise
        return cp

    @staticmethod
    def _validated_top_k(raw: Any) -> Optional[int]:
        if raw is None:
            return None
        try:
            k = int(raw)
        except Exception:
            return None         # FIX: None thay vì raise
        if not (1 <= k <= 50):
            return min(max(k, 1), 50)  # FIX: clamp thay vì raise
        return k

    def _validated_dates(self, start: Any, end: Any) -> Tuple[str, str]:
        s0, e0 = self._date_range
        start = str(start or s0)
        end   = str(end   or e0)
        sd = self._parse_date(start)
        ed = self._parse_date(end)
        if sd is None:
            sd = self._parse_date(s0)
        if ed is None:
            ed = self._parse_date(e0)
        if sd is None or ed is None:
            raise ValueError("start_date / end_date must be YYYY-MM-DD.")
        if sd > ed:
            sd, ed = ed, sd     # FIX: swap thay vì raise
        return sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d")

    def _validated_filters(self, raw: Any) -> Dict[str, List[str]]:
        if not isinstance(raw, dict):
            return {"region": [], "segment": [], "category": [], "sub_category": []}
        result: Dict[str, List[str]] = {}

        for dim in ("region", "segment", "category", "sub_category"):
            vals = raw.get(dim) or []
            if not isinstance(vals, list):
                vals = []
            vals = [str(v) for v in vals]

            if dim != "sub_category":
                valid_vals = [v for v in vals if v in self._allowed[dim]]
                result[dim] = valid_vals
            else:
                result[dim] = vals
        return result

    @staticmethod
    def _parse_date(s: str) -> Optional[date]:
        try:
            return datetime.strptime(s.strip(), "%Y-%m-%d").date()
        except Exception:
            return None