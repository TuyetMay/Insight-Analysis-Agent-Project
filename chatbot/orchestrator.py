"""
chatbot/orchestrator.py
DashboardChatbot — the single public entry point for the chatbot feature.
Coordinates NLParser → PlanValidator → SQLBuilder → AnswerFormatter → InsightGenerator.
Manages shared state: last plan, last Q&A, RAG engine, suggestion engines.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai

from config import Config
from chatbot.nl_parser import NLParser
from chatbot.plan_validator import PlanValidator
from chatbot.sql_builder import SQLBuilder
from chatbot.answer_formatter import AnswerFormatter
from chatbot.insight_generator import InsightGenerator
from chatbot.suggestions.models import Suggestion
from chatbot.suggestions.rule_engine import RuleBasedSuggestionEngine
from chatbot.suggestions.rag_engine import RAGSuggestionEngine
from rag.engine import RAGEngine


class DashboardChatbot:
    """
    Public API:
        chatbot = DashboardChatbot(df, kpis, filters)
        answer  = chatbot.get_response("What is profit by region?")
        chips   = chatbot.get_suggestions()
    """

    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any], filters: Dict[str, Any]) -> None:
        self.df      = df.copy()
        self.kpis    = kpis
        self.filters = filters

        if "order_date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["order_date"]):
            self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors="coerce")

        # ── Gemini setup ──────────────────────────────────────
        api_key = getattr(Config, "GOOGLE_API_KEY", "") or ""
        self._gemini_ready = bool(api_key)
        self._gemini_client = genai.Client(api_key=api_key) if self._gemini_ready else None
        self._model         = getattr(Config, "GEMINI_MODEL", "gemini-1.5-flash")

        # ── Core services ─────────────────────────────────────
        regions, segments, categories = self._filter_lists()
        s0, e0 = self._date_range()

        self._parser    = NLParser(self.df, self.filters, self._gemini_client, self._model)
        self._validator = PlanValidator((s0, e0), regions, segments, categories)
        self._sql       = SQLBuilder()
        self._formatter = AnswerFormatter()
        self._insights  = InsightGenerator(self._gemini_client, self._model)

        # ── RAG engine ────────────────────────────────────────
        self._rag = RAGEngine()
        self._rag.build(self.df, self.kpis, self.filters)

        # ── Suggestion engines ────────────────────────────────
        self._rule_suggestions = RuleBasedSuggestionEngine(
            allowed_metrics=["sales", "profit", "orders", "profit_margin"],
            allowed_breakdowns=["region", "segment", "category", "sub_category"],
            allowed_compare_periods=["prev_period", "mom", "yoy"],
            max_suggestions=4,
        )
        self._rag_suggestions: Optional[RAGSuggestionEngine] = (
            RAGSuggestionEngine(self._gemini_client, self._model, self._rule_suggestions, 4)
            if self._gemini_ready else None
        )

        # ── Session state ─────────────────────────────────────
        self._last_plan:     Optional[Dict[str, Any]] = None
        self._last_question: str = ""
        self._last_answer:   str = ""

    # ── Public: get_response ──────────────────────────────────

    def get_response(self, user_question: str) -> str:
        q = (user_question or "").strip()
        self._last_plan     = None
        self._last_question = q
        self._last_answer   = ""

        if not q:
            return "Ask me about Sales, Profit, Orders, or Profit Margin."

        # Tier 1 — instant KPI answer (no DB hit)
        fast = self._parser.fast_kpi_answer(q)
        if fast:
            self._record(q, fast)
            return fast

        # Tier 2 — rule-based plan
        rule_plan = self._parser.rule_based_plan(q)

        if not self._gemini_ready:
            return self._execute_plan(rule_plan, q) or "⚠️ Gemini API Key not configured in .env"

        # Tier 3 — Gemini plan (with RAG context)
        rag_ctx = self._rag.retrieve(q, k=7)
        try:
            raw_plan  = self._parser.gemini_plan(q, rag_ctx)
            plan      = self._validator.validate(raw_plan)
            result_df = self._sql.run(plan)
            insight   = self._insights.generate(plan, result_df)
            answer    = self._formatter.format(plan, result_df, insight)
            self._last_plan = plan
            self._record(q, answer)
            return answer
        except Exception as gemini_err:
            # Tier 4 — fallback to rule-based
            fallback = self._execute_plan(rule_plan, q)
            if fallback:
                return fallback
            answer = f"❌ Sorry, I couldn't answer that. ({gemini_err})"
            self._record(q, answer)
            return answer

    # ── Public: get_suggestions ───────────────────────────────

    def get_suggestions(self, *, language: str = "en") -> List[Dict[str, Any]]:
        defaults = self._dashboard_defaults()
        if not self._last_question:
            suggs = self._rule_suggestions.suggest(self._last_plan or {}, defaults)
            return self._serialise(suggs)

        rag_ctx = self._rag.retrieve_for_suggestions(
            self._last_question, self._last_answer, k=8
        )
        engine = self._rag_suggestions or self._rule_suggestions
        if isinstance(engine, RAGSuggestionEngine):
            suggs = engine.suggest(self._last_question, self._last_answer,
                                   rag_ctx, self._last_plan, defaults)
        else:
            suggs = engine.suggest(self._last_plan or {}, defaults)
        return self._serialise(suggs)

    # ── Public: run a suggestion plan directly ────────────────

    def get_response_from_plan(self, plan: Dict[str, Any]) -> str:
        try:
            plan      = self._validator.validate(plan)
            result_df = self._sql.run(plan)
            insight   = self._insights.generate(plan, result_df)
            answer    = self._formatter.format(plan, result_df, insight)
            self._last_plan   = plan
            self._last_answer = answer
            self._rag.add_turn("assistant", answer)
            return answer
        except Exception as exc:
            self._last_plan = None
            return f"❌ Could not run that suggestion. ({exc})"

    # ── Public: auto-insights summary ────────────────────────

    def get_insights(self) -> str:
        if not self._gemini_ready:
            return "Configure a Gemini API Key to enable auto-insights."
        if self.df.empty:
            return "No data available."

        rag_ctx = self._rag.retrieve("insights overview summary performance", k=6)
        prompt = f"""You are a business analyst. Using ONLY the verified data facts below, write exactly 3 bullet-point insights.

=== VERIFIED FACTS ===
{rag_ctx.as_prompt_section(max_chunks=6)}

Rules:
- Output exactly 3 lines, each starting with "- ".
- Each bullet must include at least one numeric value from the VERIFIED FACTS.
- No generic statements, no recommendations without evidence.
- English only.

Output:""".strip()

        from google.genai import types as genai_types
        try:
            resp = self._gemini_client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.3, max_output_tokens=300),
            )
            return (getattr(resp, "text", "") or "").strip()
        except Exception as exc:
            return f"Could not generate insights. ({type(exc).__name__}: {exc})"

    def rebuild_rag(self) -> None:
        self._rag.build(df=self.df, kpis=self.kpis, filters=self.filters)

    @property
    def rag_total_chunks(self) -> int:
        return self._rag.total_chunks

    # ── Private helpers ───────────────────────────────────────

    def _execute_plan(self, rule_plan: Optional[Dict[str, Any]], q: str) -> Optional[str]:
        if not rule_plan:
            return None
        try:
            plan      = self._validator.validate(rule_plan)
            result_df = self._sql.run(plan)
            insight   = self._insights.generate(plan, result_df)
            answer    = self._formatter.format(plan, result_df, insight)
            self._last_plan = plan
            self._record(q, answer)
            return answer
        except Exception as exc:
            answer = f"❌ Sorry, I couldn't answer that. ({exc})"
            self._record(q, answer)
            return answer

    def _record(self, question: str, answer: str) -> None:
        self._last_answer = answer
        self._rag.add_turn("user", question)
        self._rag.add_turn("assistant", answer)

    def _filter_lists(self):
        f = self.filters or {}
        return list(f.get("region") or []), list(f.get("segment") or []), list(f.get("category") or [])

    def _date_range(self):
        f  = self.filters or {}
        dr = f.get("date_range")
        if dr and isinstance(dr, (tuple, list)) and len(dr) == 2:
            s, e = dr
            fmt  = lambda d: d.strftime("%Y-%m-%d") if isinstance(d, (date, datetime)) else str(d)
            return fmt(s), fmt(e)
        if "order_date" in self.df.columns and not self.df.empty:
            s = pd.to_datetime(self.df["order_date"].min()).date().strftime("%Y-%m-%d")
            e = pd.to_datetime(self.df["order_date"].max()).date().strftime("%Y-%m-%d")
            return s, e
        return "1900-01-01", "2100-01-01"

    def _dashboard_defaults(self) -> Dict[str, Any]:
        s0, e0 = self._date_range()
        f = self.filters or {}
        return {
            "start_date": s0, "end_date": e0,
            "filters": {
                "region":   list(f.get("region",   []) or []),
                "segment":  list(f.get("segment",  []) or []),
                "category": list(f.get("category", []) or []),
            },
        }

    @staticmethod
    def _serialise(suggs: List[Suggestion]) -> List[Dict[str, Any]]:
        return [{"text": s.text, "plan": s.plan} for s in suggs]
