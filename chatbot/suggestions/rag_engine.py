"""
chatbot/suggestions/rag_engine.py
RAG-powered suggestion engine: uses Gemini + verified data facts to generate
follow-up questions that the dashboard CAN actually answer.
Falls back to RuleBasedSuggestionEngine on any error.
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


_VALID_INTENTS   = {"kpi_value", "kpi_trend", "kpi_rank", "kpi_compare"}
_VALID_METRICS   = {"sales", "profit", "orders", "profit_margin"}
_VALID_BREAKDOWNS= {"region", "segment", "category", "sub_category"}
_VALID_GRAINS    = {"none", "week", "month", "quarter", "year"}
_VALID_COMPARES  = {"prev_period", "mom", "yoy"}


class RAGSuggestionEngine:
    """
    Generate suggestions with Gemini + RAG context.
    Falls back to rule-based on any failure.
    """

    def __init__(self, gemini_client: Any, model_name: str,
                 rule_engine: Optional[RuleBasedSuggestionEngine] = None,
                 max_suggestions: int = 4) -> None:
        self.client      = gemini_client
        self.model_name  = model_name
        self.rule_engine = rule_engine or RuleBasedSuggestionEngine(max_suggestions=max_suggestions)
        self.max_suggestions = max_suggestions

    # ── Public ────────────────────────────────────────────────

    def suggest(self, last_question: str, last_answer: str,
                rag_context: "RAGContext",
                last_plan: Optional[Dict[str, Any]] = None,
                dashboard_defaults: Optional[Dict[str, Any]] = None) -> List[Suggestion]:
        try:
            suggestions = self._gemini_suggest(
                last_question, last_answer, rag_context, last_plan, dashboard_defaults
            )
            if suggestions:
                return suggestions[:self.max_suggestions]
        except Exception:
            pass
        return self.rule_engine.suggest(last_plan or {}, dashboard_defaults)

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
            config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=800),
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
 
        # Detect response type để điều chỉnh prompt
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
 
BAD examples (too generic): "Show sales by region", "Profit trend over years"
GOOD examples: "Which sub-category drove the discount increase in that period?",
               "Compare Tables vs Bookcases loss by region — is it concentrated?",
               "What is the profit margin trend for Furniture category year by year?"
"""
        else:
            task_instructions = f"""Generate exactly {self.max_suggestions} smart follow-up questions.
Rules:
1. Only suggest questions the DATA ABOVE can answer — no hallucination.
2. Each question explores a DIFFERENT angle not yet covered.
3. Prioritise: comparisons, trends, rankings, drill-downs.
4. English, max 60 chars per text.
"""
 
        return f"""You are a BI assistant for the Superstore Dashboard.
 
=== VERIFIED DATA FACTS ===
{rag_context.as_prompt_section(max_chunks=8)}
 
=== RECENT CONVERSATION ===
{rag_context.chat_summary or "(none)"}
 
=== PREVIOUS Q&A ===
User: {last_question}
Bot: {last_answer[:400]}...{plan_str}
 
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
 
=== OUTPUT ===
Return ONLY a valid JSON array — no markdown, no explanation.
Each element:
{{"text":"<question under 60 chars>","plan":{{"intent":"kpi_value","metrics":["sales"],"time_grain":"none","breakdown_by":null,"start_date":"{start_date}","end_date":"{end_date}","compare_period":null,"top_k":null,"order_by":"sales","filters":{{"region":[],"segment":[],"category":[]}}}}}}
 
JSON array:""".strip()
 


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
            return None

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

        if intent == "kpi_rank" and (not breakdown_by or top_k is None):
            return None
        if intent == "kpi_compare" and compare_period is None:
            return None

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
