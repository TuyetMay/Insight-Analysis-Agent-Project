from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AUDITABLE_FIELDS = {
    "intent", "compare_period", "time_grain",
    "breakdown_by", "breakdown_cols", "top_k",
    "start_date", "end_date", "condition",
}

_VALID_INTENTS = {
    "kpi_value", "kpi_trend", "kpi_rank", "kpi_compare",
    "kpi_detail", "kpi_distribution", "kpi_correlation", "clarify",
}

_VALID_GRAINS    = {"none", "week", "month", "quarter", "year"}
_VALID_COMPARES  = {"prev_period", "mom", "yoy"}
_VALID_BREAKDOWNS = {"region", "segment", "category", "sub_category", "state"}
_VALID_CONDITIONS = {"profit_negative", "profit_positive", "high_discount", "loss_orders"}


class LLMPlanAuditor:
    """
    Validates a rule-based plan with Gemini.
    Stateless — instantiate once, call audit() per query.
    """

    def __init__(self, gemini_client: Any, model_name: str) -> None:
        self.client = gemini_client
        self.model  = model_name

    def audit(self, question: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return corrected plan. Falls back to original plan on any error.
        
        Only corrects: intent, compare_period, time_grain, breakdown_by,
                       breakdown_cols, top_k, start_date, end_date, condition
        """
        if not self.client or not plan:
            return plan

        # Skip audit for plans that are already clearly correct
        if self._is_trivially_correct(question, plan):
            return plan

        try:
            corrected = self._call_gemini(question, plan)
            if corrected and self._is_improvement(plan, corrected):
                logger.info(
                    "LLM audit corrected plan: intent %r→%r, compare_period %r→%r",
                    plan.get("intent"), corrected.get("intent"),
                    plan.get("compare_period"), corrected.get("compare_period"),
                )
                return self._merge(plan, corrected)
        except Exception as exc:
            logger.warning("LLM audit failed — using original plan: %s", exc)

        return plan

    # ── Internal ──────────────────────────────────────────────

    def _is_trivially_correct(self, question: str, plan: Dict[str, Any]) -> bool:
        """Skip audit for obvious cases to save API calls."""
        intent = plan.get("intent", "")
        ql = question.lower()

        # Already a complex intent — rule-based got it right
        if intent in ("kpi_compare", "kpi_distribution", "kpi_correlation", "kpi_detail"):
            return True

        # Short, unambiguous KPI queries
        if len(ql.split()) <= 4 and intent == "kpi_value":
            return True

        # No temporal or comparison signals → nothing to fix
        has_temporal = bool(re.search(r'\b(20\d{2}|vs|versus|compared|last year|yoy|mom)\b', ql))
        has_complex  = bool(re.search(r'\b(distribution|histogram|bucket|correlation|scatter|relationship|between)\b', ql))
        if not has_temporal and not has_complex:
            return True

        return False

    def _call_gemini(self, question: str, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        from google.genai import types as genai_types

        prompt = self._build_prompt(question, plan)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=250,
            ),
        )
        raw = (getattr(resp, "text", "") or "").strip()
        return self._parse_correction(raw)

    def _build_prompt(self, question: str, plan: Dict[str, Any]) -> str:
        plan_clean = {k: v for k, v in plan.items()
                      if k in _AUDITABLE_FIELDS}
        return f"""You are a strict BI query plan auditor.

USER QUESTION: "{question}"

CURRENT PLAN (from rule-based parser):
{json.dumps(plan_clean, indent=2)}

TASK: Check if the plan correctly reflects the question.
Output ONLY a JSON object with fields that need correction.
Output {{}} if the plan is already correct.

VALID VALUES:
- intent: {sorted(_VALID_INTENTS)}
- compare_period: {sorted(_VALID_COMPARES)} or null
- time_grain: {sorted(_VALID_GRAINS)}
- breakdown_by: {sorted(_VALID_BREAKDOWNS)} or null
- breakdown_cols: list of breakdown dimensions (for multi-group queries), e.g. ["region", "category"]

COMMON CORRECTIONS:
- "X compared to Y" / "X vs Y" / "2023 vs 2022" → intent: kpi_compare, compare_period: prev_period or yoy
- "distribution / histogram / buckets" → intent: kpi_distribution
- "correlation / relationship between X and Y" → intent: kpi_correlation
- "trend / over time / monthly / yearly" → intent: kpi_trend
- "by region AND by category" (2 dimensions) → breakdown_cols: ["region", "category"]
- "top N" → intent: kpi_rank, top_k: N

FORBIDDEN: Do not change metrics, filters, start_date, end_date unless clearly wrong.
Output ONLY the corrections as JSON, or {{}} if nothing needs fixing:""".strip()

    def _parse_correction(self, raw: str) -> Optional[Dict[str, Any]]:
        for attempt in (raw, re.sub(r"```(?:json)?", "", raw).strip()):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    def _is_improvement(self, original: Dict, corrected: Dict) -> bool:
        """Only apply if at least one auditable field changed."""
        if not corrected:
            return False
        for k, v in corrected.items():
            if k in _AUDITABLE_FIELDS and original.get(k) != v:
                return True
        return False

    def _merge(self, original: Dict, corrections: Dict) -> Dict[str, Any]:
        """Apply only validated corrections to original plan."""
        merged = dict(original)
        for k, v in corrections.items():
            if k not in _AUDITABLE_FIELDS:
                continue
            # Type-safe validation before applying
            if k == "intent" and v not in _VALID_INTENTS:
                continue
            if k == "compare_period" and v not in _VALID_COMPARES and v is not None:
                continue
            if k == "time_grain" and v not in _VALID_GRAINS:
                continue
            if k == "breakdown_by" and v not in _VALID_BREAKDOWNS and v is not None:
                continue
            if k == "breakdown_cols":
                if not isinstance(v, list):
                    continue
                v = [b for b in v if b in _VALID_BREAKDOWNS]
            if k == "top_k":
                try:
                    v = int(v)
                    if not (1 <= v <= 50):
                        continue
                except Exception:
                    continue
            if k == "condition" and v not in _VALID_CONDITIONS:
                continue
            merged[k] = v

        # Post-correction consistency fixes
        if merged.get("intent") == "kpi_compare" and not merged.get("compare_period"):
            merged["compare_period"] = "prev_period"
        if merged.get("intent") == "kpi_rank" and not merged.get("top_k"):
            merged["top_k"] = 10
        if merged.get("intent") in ("kpi_distribution", "kpi_correlation"):
            merged.setdefault("time_grain", "none")

        return merged