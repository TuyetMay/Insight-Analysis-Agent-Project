"""
chatbot/suggestions/rag_engine.py — IMPROVED

CHANGES vs original:
  - max_suggestions tăng từ 4 → 10
  - max_output_tokens tăng từ 800 → 1500 để đủ chỗ cho 10 suggestions
  - Prompt cải thiện: yêu cầu đa dạng hơn, cover nhiều angle hơn
  - Thêm diversity enforcement trong prompt
  - Cải thiện fallback: nếu Gemini trả về < 5, bổ sung từ rule engine
  - Fix: validate_plan() chấp nhận kpi_detail intent
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from google.genai import types as genai_types

from chatbot.suggestions.models import Suggestion
from chatbot.suggestions.rule_engine import RuleBasedSuggestionEngine

if TYPE_CHECKING:
    from rag.engine import RAGContext


_VALID_INTENTS   = {"kpi_value", "kpi_trend", "kpi_rank", "kpi_compare", "kpi_detail"}  # thêm kpi_detail
_VALID_METRICS   = {"sales", "profit", "orders", "profit_margin"}
_VALID_BREAKDOWNS= {"region", "segment", "category", "sub_category"}
_VALID_GRAINS    = {"none", "week", "month", "quarter", "year"}
_VALID_COMPARES  = {"prev_period", "mom", "yoy"}


class RAGSuggestionEngine:
    """
    Generate suggestions với Gemini + RAG context.
    IMPROVED: max 10 suggestions, better diversity, smarter fallback.
    """

    def __init__(self, gemini_client: Any, model_name: str,
                 rule_engine: Optional[RuleBasedSuggestionEngine] = None,
                 max_suggestions: int = 10) -> None:  # ← tăng default lên 10
        self.client      = gemini_client
        self.model_name  = model_name
        self.rule_engine = rule_engine or RuleBasedSuggestionEngine(max_suggestions=max_suggestions)
        self.max_suggestions = max_suggestions

    # ── Public ────────────────────────────────────────────────

    def suggest(self, last_question: str, last_answer: str,
                rag_context: "RAGContext",
                last_plan: Optional[Dict[str, Any]] = None,
                dashboard_defaults: Optional[Dict[str, Any]] = None) -> List[Suggestion]:
        """
        IMPROVED: Nếu Gemini trả về ít hơn target, bổ sung từ rule engine.
        """
        gemini_suggestions: List[Suggestion] = []
        try:
            gemini_suggestions = self._gemini_suggest(
                last_question, last_answer, rag_context, last_plan, dashboard_defaults
            )
        except Exception:
            pass

        # Nếu Gemini trả đủ → dùng luôn
        if len(gemini_suggestions) >= self.max_suggestions:
            return gemini_suggestions[:self.max_suggestions]

        # Bổ sung từ rule engine nếu thiếu
        if last_plan:
            rule_suggestions = self.rule_engine.suggest(
                last_plan, dashboard_defaults, last_answer=last_answer
            )
            # Merge: Gemini trước, rule sau, dedup by text
            seen_texts = {s.text for s in gemini_suggestions}
            for s in rule_suggestions:
                if s.text not in seen_texts and len(gemini_suggestions) < self.max_suggestions:
                    gemini_suggestions.append(s)
                    seen_texts.add(s.text)

        return gemini_suggestions[:self.max_suggestions]

    # ── Gemini call ───────────────────────────────────────────

    def _gemini_suggest(self, last_question: str, last_answer: str,
                        rag_context: "RAGContext",
                        last_plan: Optional[Dict[str, Any]],
                        dashboard_defaults: Optional[Dict[str, Any]]) -> List[Suggestion]:
        prompt = self._build_prompt(last_question, last_answer, rag_context,
                                    last_plan, dashboard_defaults)
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.4,           # tăng nhẹ để có diversity
                max_output_tokens=1500,    # tăng từ 800 → 1500 cho 10 items
            ),
        )
        items = self._parse_json_array((getattr(resp, "text", None) or "").strip())
        return [
            Suggestion(
                text=(item.get("text") or "").strip(),
                plan=self._validate_plan(item.get("plan"), dashboard_defaults),
            )
            for item in items
            if (item.get("text") or "").strip()
        ]

    def _build_prompt(self, last_question: str, last_answer: str,
                      rag_context: "RAGContext",
                      last_plan: Optional[Dict[str, Any]],
                      dashboard_defaults: Optional[Dict[str, Any]]) -> str:
        defaults   = dashboard_defaults or {}
        start_date = defaults.get("start_date", "unknown")
        end_date   = defaults.get("end_date",   "unknown")
        filters    = defaults.get("filters", {})
        plan_str   = f"\nLast structured plan: {json.dumps(last_plan)}" if last_plan else ""

        is_agent_response = (
            "Diagnostic Analysis" in last_answer
            or "Root Cause" in last_answer
            or "Supporting Evidence" in last_answer
            or last_answer.startswith("⚠️ **Data Check")
            or last_answer.startswith("❌ Could not")
        )

        if is_agent_response:
            task_instructions = f"""Generate exactly {self.max_suggestions} follow-up questions after a DIAGNOSTIC analysis.

The user just asked: "{last_question}"
The system gave a diagnostic/causal answer (see PREVIOUS Q&A above).

Rules for diagnostic follow-ups:
1. Drill deeper into the root cause identified — ask for evidence or breakdown
2. Ask for data to VERIFY or CHALLENGE the diagnosis
3. Suggest related anomalies the user hasn't checked yet
4. Ask for actionable next steps based on what was found
5. Each question should be different from the original question

DIVERSITY REQUIREMENT — cover ALL of these angles:
  - At least 2 questions drill into a specific dimension (region/category/segment)
  - At least 1 question asks for a time trend
  - At least 1 question asks for a comparison (YoY or period)
  - At least 1 question explores a related metric (if profit was asked, ask about margin or orders)
  - At least 1 question asks about top/bottom performers

BAD examples (too generic): "Show sales by region", "Profit trend over years"
GOOD examples: "Which sub-category drove the discount increase in that period?",
               "Compare Tables vs Bookcases loss by region — is it concentrated?",
               "What is the profit margin trend for Furniture category year by year?"
"""
        else:
            task_instructions = f"""Generate exactly {self.max_suggestions} smart follow-up questions.

Rules:
1. Only suggest questions the DATA ABOVE can answer — no hallucination.
2. Each question explores a COMPLETELY DIFFERENT angle.
3. DIVERSITY REQUIREMENT — must include:
   - At least 2 different breakdown dimensions (region, segment, category, sub_category)
   - At least 1 time trend question (monthly/quarterly/yearly)
   - At least 1 comparison question (YoY, MoM, or period vs period)
   - At least 1 ranking question (top N)
   - At least 1 related metric (if sales was asked, suggest profit or margin)
   - At least 1 drill-down into loss-making or anomaly data
4. Prioritise: actionable insights over generic data views.
5. English, max 70 chars per text.
"""

        return f"""You are a BI assistant for the Superstore Dashboard.

=== VERIFIED DATA FACTS ===
{rag_context.as_prompt_section(max_chunks=8)}

=== RECENT CONVERSATION ===
{rag_context.chat_summary or "(none)"}

=== PREVIOUS Q&A ===
User: {last_question}
Bot: {last_answer[:500]}...{plan_str}

=== TASK ===
{task_instructions}

=== CONSTRAINTS ===
Date range: {start_date} to {end_date}
Active filters: {json.dumps(filters)}
Valid metrics: sales, profit, orders, profit_margin
Valid dimensions: region, segment, category, sub_category
Valid time grains: week, month, quarter, year
Valid compare periods: yoy, mom, prev_period
Valid intents: kpi_value, kpi_trend, kpi_rank, kpi_compare, kpi_detail

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array of exactly {self.max_suggestions} objects.
No markdown, no explanation, no text outside the array.

Schema for each object:
{{"text":"<question under 70 chars>","plan":{{"intent":"kpi_value","metrics":["sales"],"time_grain":"none","breakdown_by":null,"start_date":"{start_date}","end_date":"{end_date}","compare_period":null,"top_k":null,"order_by":"sales","filters":{{"region":[],"segment":[],"category":[]}}}}}}

JSON array (exactly {self.max_suggestions} items):""".strip()


    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        for attempt in (raw, re.sub(r"```(?:json)?", "", raw).strip()):
            try:
                result = json.loads(attempt)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        return []

    def _validate_plan(self, plan: Any,
                       defaults: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(plan, dict):
            return None
        intent = plan.get("intent")
        if intent not in _VALID_INTENTS:
            return None

        metrics = plan.get("metrics", [])
        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = [m for m in metrics if m in _VALID_METRICS]
        if not metrics:
            metrics = ["sales"]  # default thay vì return None

        time_grain    = plan.get("time_grain", "none")
        if time_grain not in _VALID_GRAINS:
            time_grain = "none"

        breakdown_by  = plan.get("breakdown_by")
        if breakdown_by not in _VALID_BREAKDOWNS:
            breakdown_by = None

        compare_period = plan.get("compare_period")
        if compare_period not in _VALID_COMPARES:
            compare_period = None

        top_k = plan.get("top_k")
        if top_k is not None:
            try:
                top_k = max(1, min(50, int(top_k)))
            except Exception:
                top_k = None

        # Fix: kpi_rank cần breakdown_by và top_k
        if intent == "kpi_rank":
            if not breakdown_by:
                breakdown_by = "sub_category"  # default thay vì return None
            if top_k is None:
                top_k = 10  # default thay vì return None

        # Fix: kpi_compare cần compare_period
        if intent == "kpi_compare" and compare_period is None:
            compare_period = "yoy"  # default thay vì return None

        d = defaults or {}
        raw_filters = plan.get("filters") or {}
        return {
            "intent": intent, "metrics": metrics, "time_grain": time_grain,
            "breakdown_by": breakdown_by,
            "start_date": plan.get("start_date") or d.get("start_date", "2000-01-01"),
            "end_date":   plan.get("end_date")   or d.get("end_date",   "2100-01-01"),
            "compare_period": compare_period, "top_k": top_k,
            "order_by": plan.get("order_by") or metrics[0],
            "filters": {
                "region":   list(raw_filters.get("region",   []) or []),
                "segment":  list(raw_filters.get("segment",  []) or []),
                "category": list(raw_filters.get("category", []) or []),
            },
        }


# ── Backward-compat alias ─────────────────────────────────────

class SuggestionEngine(RuleBasedSuggestionEngine):
    """Legacy alias — new code should use RuleBasedSuggestionEngine directly."""

    def suggest_from_plan(self, plan: Dict[str, Any],
                          dashboard_defaults: Optional[Dict[str, Any]] = None,
                          language: str = "en") -> List[Suggestion]:
        return self.suggest(plan, dashboard_defaults)

    def suggest_from_dashboard_state(self, state: Dict[str, Any],
                                     language: str = "en") -> List[Suggestion]:
        return self._fallback(state)