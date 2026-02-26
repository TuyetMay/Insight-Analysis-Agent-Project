# src/chatbot_suggest_question.py
"""
Suggestion Engine - two modes:

1. RuleBasedSuggestionEngine  (fast, deterministic, no LLM needed)
   - Derives suggestions from a validated plan by applying fixed transformations.
   - Used as fallback when Gemini is unavailable.

2. RAGSuggestionEngine  (smart, grounded in real data)
   - Uses Gemini + RAG context to generate suggestions the chatbot CAN answer.
   - Each suggestion carries a plan JSON so it can be re-run without calling LLM again.
   - Falls back to rule-based on any error.

Output format: List[Suggestion]  →  [{"text": "...", "plan": {...}|None}]
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from google.genai import types as genai_types

if TYPE_CHECKING:
    from src.rag_engine import RAGContext


@dataclass(frozen=True)
class Suggestion:
    """
    A follow-up suggestion rendered as a clickable chip.
    - text : what the user sees.
    - plan : optional JSON plan that can be executed without calling the LLM again.
    """
    text: str
    plan: Optional[Dict[str, Any]] = None


# ─────────────────────────────────────────────────────────────
# 1. Rule-based engine  (deterministic, fallback)
# ─────────────────────────────────────────────────────────────

class RuleBasedSuggestionEngine:
    """
    Generates suggestions by applying a fixed set of transformations to a validated plan.
    Safe, offline, no LLM required.
    """

    _METRICS = {"sales", "profit", "orders", "profit_margin"}
    _BREAKDOWNS = {"region", "segment", "category", "sub_category"}
    _COMPARE_PERIODS = {"prev_period", "mom", "yoy"}

    _METRIC_LABELS = {
        "sales": "Sales", "profit": "Profit",
        "orders": "Orders", "profit_margin": "Profit Margin",
    }
    _DIM_LABELS = {
        "region": "Region", "segment": "Segment",
        "category": "Category", "sub_category": "Sub-Category",
    }
    _GRAIN_LABELS = {
        "month": "Month", "quarter": "Quarter", "year": "Year", "week": "Week",
    }
    _COMPARE_LABELS = {
        "yoy": "YoY (vs last year)",
        "mom": "MoM (vs last month)",
        "prev_period": "vs previous period",
    }

    def __init__(
        self,
        *,
        allowed_metrics: Optional[List[str]] = None,
        allowed_breakdowns: Optional[List[str]] = None,
        allowed_compare_periods: Optional[List[str]] = None,
        max_suggestions: int = 4,
    ) -> None:
        self.allowed_metrics = set(allowed_metrics or self._METRICS)
        self.allowed_breakdowns = set(allowed_breakdowns or self._BREAKDOWNS)
        self.allowed_compare_periods = set(allowed_compare_periods or self._COMPARE_PERIODS)
        self.max_suggestions = max(1, int(max_suggestions))

    def suggest(
        self,
        plan: Dict[str, Any],
        dashboard_defaults: Optional[Dict[str, Any]] = None,
    ) -> List[Suggestion]:
        if not isinstance(plan, dict) or not plan.get("intent"):
            return self._fallback(dashboard_defaults)

        base = self._normalize(plan, dashboard_defaults or {})
        intent = base.get("intent")
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

        if intent == "kpi_trend":
            candidates += self._breakdowns(base)
        elif intent == "kpi_rank":
            candidates += self._rank_variations(base)

        return self._dedup(candidates)

    # ── helpers ───────────────────────────────────────────────

    def _fallback(self, defaults: Optional[Dict[str, Any]]) -> List[Suggestion]:
        metric = (defaults or {}).get("last_metric") or "sales"
        if metric not in self.allowed_metrics:
            metric = "sales"
        base = {
            "intent": "kpi_value", "metrics": [metric], "time_grain": "none",
            "breakdown_by": None, "compare_period": None, "top_k": None,
            "order_by": metric,
            "start_date": (defaults or {}).get("start_date"),
            "end_date": (defaults or {}).get("end_date"),
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
        if bd is not None and bd not in self.allowed_breakdowns:
            bd = None
        p["breakdown_by"] = bd
        p.setdefault("compare_period", None)
        p.setdefault("top_k", None)
        p.setdefault("order_by", p["metrics"][0])
        p.setdefault("start_date", defaults.get("start_date"))
        p.setdefault("end_date", defaults.get("end_date"))
        p.setdefault("filters", defaults.get("filters") or {"region": [], "segment": [], "category": []})
        return p

    def _clone(self, base: Dict[str, Any], **updates: Any) -> Dict[str, Any]:
        p = copy.deepcopy(base)
        p.update(updates)
        return p

    def _lm(self, m: str) -> str:
        return self._METRIC_LABELS.get(m, m.replace("_", " ").title())

    def _ld(self, d: str) -> str:
        return self._DIM_LABELS.get(d, d.replace("_", " ").title())

    def _lg(self, g: str) -> str:
        return self._GRAIN_LABELS.get(g, g.title())

    def _lc(self, c: str) -> str:
        return self._COMPARE_LABELS.get(c, c)

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

    # ── builders ──────────────────────────────────────────────

    def _breakdowns(self, base: Dict[str, Any]) -> List[Suggestion]:
        metric = base["metrics"][0]
        current = base.get("breakdown_by")
        out: List[Suggestion] = []
        for b in ["region", "segment", "category", "sub_category"]:
            if b not in self.allowed_breakdowns or b == current:
                continue
            p = self._clone(base, intent="kpi_value", breakdown_by=b, top_k=None)
            out.append(Suggestion(text=f"{self._lm(metric)} by {self._ld(b)}", plan=p))
        return out

    def _time_grains(self, base: Dict[str, Any]) -> List[Suggestion]:
        metric = base["metrics"][0]
        current = base.get("time_grain") or "none"
        out: List[Suggestion] = []
        for grain in ["month", "quarter", "year"]:
            if grain == current:
                continue
            p = self._clone(base, intent="kpi_trend", time_grain=grain, breakdown_by=None, top_k=None)
            out.append(Suggestion(text=f"{self._lm(metric)} trend by {self._lg(grain)}", plan=p))
        return out

    def _compare(self, base: Dict[str, Any]) -> List[Suggestion]:
        metric = base["metrics"][0]
        out: List[Suggestion] = []
        for c in ["yoy", "mom", "prev_period"]:
            if c not in self.allowed_compare_periods or base.get("compare_period") == c:
                continue
            p = self._clone(base, intent="kpi_compare", compare_period=c, top_k=None, metrics=[metric])
            out.append(Suggestion(text=f"Compare {self._lm(metric)} {self._lc(c)}", plan=p))
        return out

    def _rank_from_breakdown(self, base: Dict[str, Any]) -> List[Suggestion]:
        metric = base["metrics"][0]
        b = base.get("breakdown_by")
        if not b or b not in self.allowed_breakdowns:
            return []
        out: List[Suggestion] = []
        for k in [3, 5]:
            p = self._clone(base, intent="kpi_rank", top_k=k, order_by=metric)
            out.append(Suggestion(text=f"Top {k} {self._ld(b)} by {self._lm(metric)}", plan=p))
        return out

    def _rank_variations(self, base: Dict[str, Any]) -> List[Suggestion]:
        metric = base["metrics"][0]
        b = base.get("breakdown_by") or "sub_category"
        out: List[Suggestion] = []
        for k in [3, 5, 10]:
            if base.get("top_k") == k:
                continue
            p = self._clone(base, intent="kpi_rank", top_k=k, order_by=metric, breakdown_by=b)
            out.append(Suggestion(text=f"Top {k} {self._ld(b)} by {self._lm(metric)}", plan=p))
        return out

    def _metric_switch(self, base: Dict[str, Any]) -> List[Suggestion]:
        current = base["metrics"][0]
        out: List[Suggestion] = []
        for m in ["sales", "profit", "orders", "profit_margin"]:
            if m not in self.allowed_metrics or m == current:
                continue
            p = self._clone(base, metrics=[m], order_by=m)
            out.append(Suggestion(text=f"View {self._lm(m)}", plan=p))
        return out


# ─────────────────────────────────────────────────────────────
# 2. RAGSuggestionEngine  (Gemini + RAG context)
# ─────────────────────────────────────────────────────────────

class RAGSuggestionEngine:
    """
    Generates suggestions using Gemini + RAG context.

    Flow:
        last Q + last A + chat history
                 |
         RAG Retrieve → rag_context  (real verified data facts)
                 |
         Gemini Prompt:
           "Given these real data facts: [rag_context]
            Generate N follow-up questions the data CAN answer.
            Return JSON array with plan per suggestion."
                 |
         Parse JSON → List[{text, plan}]
                 |
         Fallback: RuleBasedSuggestionEngine
    """

    _VALID_INTENTS = {"kpi_value", "kpi_trend", "kpi_rank", "kpi_compare"}
    _VALID_METRICS = {"sales", "profit", "orders", "profit_margin"}
    _VALID_BREAKDOWNS = {"region", "segment", "category", "sub_category"}
    _VALID_GRAINS = {"none", "week", "month", "quarter", "year"}
    _VALID_COMPARES = {"prev_period", "mom", "yoy"}

    def __init__(
        self,
        gemini_client: Any,
        model_name: str,
        rule_engine: Optional[RuleBasedSuggestionEngine] = None,
        max_suggestions: int = 4,
    ) -> None:
        self.client = gemini_client
        self.model_name = model_name
        self.rule_engine = rule_engine or RuleBasedSuggestionEngine(max_suggestions=max_suggestions)
        self.max_suggestions = max_suggestions

    def suggest(
        self,
        last_question: str,
        last_answer: str,
        rag_context: "RAGContext",
        last_plan: Optional[Dict[str, Any]] = None,
        dashboard_defaults: Optional[Dict[str, Any]] = None,
    ) -> List[Suggestion]:
        try:
            suggestions = self._gemini_suggest(
                last_question=last_question,
                last_answer=last_answer,
                rag_context=rag_context,
                last_plan=last_plan,
                dashboard_defaults=dashboard_defaults,
            )
            if suggestions:
                return suggestions[:self.max_suggestions]
        except Exception:
            pass

        # Fallback: rule-based
        return self.rule_engine.suggest(
            plan=last_plan or {},
            dashboard_defaults=dashboard_defaults,
        )

    def _build_prompt(
        self,
        last_question: str,
        last_answer: str,
        rag_context: "RAGContext",
        last_plan: Optional[Dict[str, Any]],
        dashboard_defaults: Optional[Dict[str, Any]],
    ) -> str:
        defaults = dashboard_defaults or {}
        start_date = defaults.get("start_date", "unknown")
        end_date = defaults.get("end_date", "unknown")
        filters = defaults.get("filters", {})

        data_facts = rag_context.as_prompt_section(max_chunks=8)
        chat_history = rag_context.chat_summary or "(no history)"

        plan_str = ""
        if last_plan and isinstance(last_plan, dict):
            plan_str = f"\nLast executed plan: {json.dumps(last_plan)}"

        return f"""
You are a Business Intelligence assistant for the Superstore Dashboard.

=== VERIFIED DATA FACTS FROM DATABASE ===
{data_facts}

=== RECENT CONVERSATION HISTORY ===
{chat_history}

=== PREVIOUS Q&A ===
User asked: {last_question}
Bot answered: {last_answer[:300]}...{plan_str}

=== TASK ===
Generate exactly {self.max_suggestions} smart follow-up questions. RULES:
1. Only suggest questions that THE DATA ABOVE CAN ANSWER. No hallucination.
2. Each question must explore a different angle not yet covered.
3. Prioritize: comparisons, trends, rankings, breakdowns by dimension.
4. Write in English, concise (max 60 chars per text).
5. Base questions on the actual data facts shown above.

=== TECHNICAL CONSTRAINTS ===
- Dashboard date range: {start_date} to {end_date}
- Active filters: {json.dumps(filters)}
- Valid metrics: sales, profit, orders, profit_margin
- Valid dimensions: region, segment, category, sub_category
- Valid time grains: week, month, quarter, year
- Valid compare periods: yoy, mom, prev_period
- Valid intents: kpi_value, kpi_trend, kpi_rank, kpi_compare

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array. No markdown. No explanation.
Each element:
{{
  "text": "Short question text shown to user (< 60 chars)",
  "plan": {{
    "intent": "kpi_value|kpi_trend|kpi_rank|kpi_compare",
    "metrics": ["sales"],
    "time_grain": "none",
    "breakdown_by": null,
    "start_date": "{start_date}",
    "end_date": "{end_date}",
    "compare_period": null,
    "top_k": null,
    "order_by": "sales",
    "filters": {{"region": [], "segment": [], "category": []}}
  }}
}}

OUTPUT (JSON array only):""".strip()

    def _gemini_suggest(
        self,
        last_question: str,
        last_answer: str,
        rag_context: "RAGContext",
        last_plan: Optional[Dict[str, Any]],
        dashboard_defaults: Optional[Dict[str, Any]],
    ) -> List[Suggestion]:
        prompt = self._build_prompt(
            last_question=last_question,
            last_answer=last_answer,
            rag_context=rag_context,
            last_plan=last_plan,
            dashboard_defaults=dashboard_defaults,
        )

        # ✅ Correct SDK: use config=GenerateContentConfig(...)
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=800,
            ),
        )

        raw = (getattr(resp, "text", None) or "").strip()
        items = self._parse_json_array(raw)

        suggestions: List[Suggestion] = []
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            plan = item.get("plan")
            validated_plan = self._validate_plan_lightly(plan, dashboard_defaults)
            suggestions.append(Suggestion(text=text, plan=validated_plan))

        return suggestions

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return result
        except Exception:
            pass

        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except Exception:
            pass

        m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        return []

    def _validate_plan_lightly(
        self,
        plan: Any,
        dashboard_defaults: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(plan, dict):
            return None

        intent = plan.get("intent")
        if intent not in self._VALID_INTENTS:
            return None

        metrics = plan.get("metrics", [])
        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = [m for m in metrics if m in self._VALID_METRICS]
        if not metrics:
            return None

        time_grain = plan.get("time_grain", "none")
        if time_grain not in self._VALID_GRAINS:
            time_grain = "none"

        breakdown_by = plan.get("breakdown_by")
        if breakdown_by is not None and breakdown_by not in self._VALID_BREAKDOWNS:
            breakdown_by = None

        compare_period = plan.get("compare_period")
        if compare_period is not None and compare_period not in self._VALID_COMPARES:
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

        defaults = dashboard_defaults or {}
        start_date = plan.get("start_date") or defaults.get("start_date", "2000-01-01")
        end_date = plan.get("end_date") or defaults.get("end_date", "2100-01-01")

        raw_filters = plan.get("filters") or {}
        filters = {
            "region": list(raw_filters.get("region") or []),
            "segment": list(raw_filters.get("segment") or []),
            "category": list(raw_filters.get("category") or []),
        }

        return {
            "intent": intent,
            "metrics": metrics,
            "time_grain": time_grain,
            "breakdown_by": breakdown_by,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "compare_period": compare_period,
            "top_k": top_k,
            "order_by": plan.get("order_by") or metrics[0],
            "filters": filters,
        }


# ─────────────────────────────────────────────────────────────
# Backward compat alias
# ─────────────────────────────────────────────────────────────

class SuggestionEngine(RuleBasedSuggestionEngine):
    """
    Backward-compat alias for old app.py imports.
    New code should use RuleBasedSuggestionEngine or RAGSuggestionEngine.
    """

    def suggest_from_plan(
        self,
        plan: Dict[str, Any],
        dashboard_defaults: Optional[Dict[str, Any]] = None,
        language: str = "en",
    ) -> List[Suggestion]:
        return self.suggest(plan, dashboard_defaults)

    def suggest_from_dashboard_state(
        self,
        state: Dict[str, Any],
        language: str = "en",
    ) -> List[Suggestion]:
        return self._fallback(state)