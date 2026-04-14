from __future__ import annotations

from datetime import date, datetime
import logging
import streamlit as st
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai

# ── NEW: SmartRouter + HybridExecutor replace old QueryRouter ─
from chatbot.smart_router import SmartRouter, RouteDecision
from chatbot.hybrid_executor import HybridExecutor

from chatbot.agent.orchestrator import AgentOrchestrator
from chatbot.agent.suggestions import get_diagnostic_suggestions
from chatbot.llm_plan_auditor import LLMPlanAuditor
from chatbot.quick_insight import QuickInsightHandler
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

logger = logging.getLogger(__name__)

_QUICK_TOKENS = {
    "__quick_sales__":   "sales",
    "__quick_profit__":  "profit",
    "__quick_orders__":  "orders",
    "__quick_margin__":  "profit_margin",
}


class DashboardChatbot:
    def __init__(self, df: pd.DataFrame, kpis: Dict[str, Any], filters: Dict[str, Any]) -> None:
        self.df      = df.copy()
        self.kpis    = kpis
        self.filters = filters
        self._plan_history: List[Dict[str, Any]] = []

        self._last_was_agent = False

        if "order_date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["order_date"]):
            self.df["order_date"] = pd.to_datetime(self.df["order_date"], errors="coerce")

        api_key = getattr(Config, "GOOGLE_API_KEY", "") or ""
        self._gemini_ready  = bool(api_key)
        self._gemini_client = genai.Client(api_key=api_key) if self._gemini_ready else None
        self._model         = getattr(Config, "GEMINI_MODEL", "gemini-1.5-flash")

        regions, segments, categories = self._filter_lists()
        s0, e0 = self._date_range()

        self._parser    = NLParser(self.df, self.filters, self._gemini_client, self._model)
        self._validator = PlanValidator((s0, e0), regions, segments, categories)
        self._sql       = SQLBuilder()
        self._formatter = AnswerFormatter()
        self._insights  = InsightGenerator(self._gemini_client, self._model)

        self._rag = RAGEngine()
        self._rag.build_static(self.df)
        self._rag.build(self.df, self.kpis, self.filters)

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

        self._last_plan:     Optional[Dict[str, Any]] = None
        self._last_question: str = ""
        self._last_answer:   str = ""

        self._agent = AgentOrchestrator(
            gemini_client  = self._gemini_client,
            model_name     = self._model,
            default_start  = s0,
            default_end    = e0,
        ) if self._gemini_ready else None

        self._plan_auditor = LLMPlanAuditor(self._gemini_client, self._model)

        # ── NEW: Smart router (replaces QueryRouter) ──────────────────
        self._smart_router = SmartRouter(
            gemini_client = self._gemini_client,
            model_name    = self._model,
        )

        self._hybrid = HybridExecutor(
            structured_runner = self._run_structured_query,
            agent_runner      = (
                lambda q: self._agent.run(q)
                if self._agent
                else "❌ Agent not available (no API key)."
            ),
            gemini_client = self._gemini_client,
            model_name    = self._model,
        )

    # ── Public: get_response ──────────────────────────────────

    def get_response(self, user_question: str) -> str:
        q = (user_question or "").strip()
        self._last_plan      = None
        self._last_question  = q
        self._last_answer    = ""
        self._last_was_agent = False

        try:
            st.session_state["is_generating"] = True
        except Exception:
            pass

        if not q:
            self._clear_stop()
            return "Ask me about Sales, Profit, Orders, or Profit Margin."

        if self._is_stop_requested():
            self._clear_stop()
            return "⏹️ *Generation stopped.*"

        # ── Quick token handler ───────────────────────────────
        if q in _QUICK_TOKENS:
            kpi_name = _QUICK_TOKENS[q]
            handler  = QuickInsightHandler(
                self.df, self.kpis, self.filters,
                self._gemini_client, self._model
            )
            answer = handler.generate(kpi_name)
            self._clear_stop()
            return answer

        # ── Smart routing: LLM classifies into structured / agent / hybrid ──
        if self._gemini_ready:
            decision = self._smart_router.classify(q)
            logger.debug(
                "SmartRouter: mode=%s fallback=%s query=%r",
                decision.mode, decision.used_fallback, q[:60]
            )

            # ── Agent path ────────────────────────────────────
            if decision.mode == "agent":
                if self._is_stop_requested():
                    self._clear_stop()
                    return "⏹️ *Generation stopped.*"
                answer = self._agent.run(q)
                self._last_was_agent = True
                self._record(q, answer)
                self._clear_stop()
                return answer

            # ── Hybrid path (NEW) ─────────────────────────────
            elif decision.mode == "hybrid":
                if self._is_stop_requested():
                    self._clear_stop()
                    return "⏹️ *Generation stopped.*"
                answer = self._hybrid.execute(decision, q)
                self._last_was_agent = True   # use diagnostic suggestions
                self._record(q, answer)
                self._clear_stop()
                return answer

            # ── Structured path falls through to Tier 1/2/3 ──

        # ── Tier 1: Fast KPI ──────────────────────────────────
        fast = self._parser.fast_kpi_answer(q)
        if fast:
            self._record(q, fast)
            self._clear_stop()
            return fast

        if self._is_stop_requested():
            self._clear_stop()
            return "⏹️ *Generation stopped.*"

        # ── Tier 2: Rule-based ────────────────────────────────
        rule_plan = self._parser.rule_based_plan(q)
        if rule_plan:
            rule_plan = self._plan_auditor.audit(q, rule_plan)

        if not self._gemini_ready:
            result = self._execute_plan(rule_plan, q) or "⚠️ Gemini API Key not configured."
            self._clear_stop()
            return result

        if rule_plan is not None:
            result = self._execute_plan(rule_plan, q)
            if result:
                self._clear_stop()
                return result

        if self._is_stop_requested():
            self._clear_stop()
            return "⏹️ *Generation stopped.*"

        # ── Tier 3: Gemini ────────────────────────────────────
        _grain_hint = (rule_plan or {}).get("time_grain") or None
        rag_ctx = self._rag.retrieve(q, k=15, tier=3, grain=_grain_hint)
        try:
            raw_plan  = self._parser.gemini_plan(q, rag_ctx)
            plan      = self._validator.validate(raw_plan)
            result_df = self._sql.run(plan)
            insight   = self._insights.generate(plan, result_df)
            answer    = self._formatter.format(plan, result_df, insight)
            self._last_plan = plan
            self._rag.record_example(q, plan.get("intent", ""), plan)
            self._record(q, answer)
            self._clear_stop()
            return answer
        except Exception as gemini_err:
            answer = f"❌ Sorry, I couldn't answer that. ({gemini_err})"
            self._record(q, answer)
            self._clear_stop()
            return answer

    def get_suggestions(self, *, language: str = "en") -> List[Dict[str, Any]]:
        if self._last_was_agent and self._last_question:
            diag_suggs = get_diagnostic_suggestions(
                question=self._last_question,
                agent_response=self._last_answer,
                plan_defaults=self._dashboard_defaults(),
            )
            if diag_suggs:
                return diag_suggs

        defaults = self._dashboard_defaults()
        if not self._last_question:
            suggs = self._rule_suggestions.suggest(self._last_plan or {}, defaults)
            return self._serialise(suggs)

        rag_ctx = self._rag.retrieve_for_suggestions(
            self._last_question, self._last_answer, k=8
        )
        engine = self._rag_suggestions or self._rule_suggestions
        if isinstance(engine, RAGSuggestionEngine):
            suggs = engine.suggest(
                self._last_question, self._last_answer,
                rag_ctx, self._last_plan, defaults
            )
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
            self._last_was_agent = False
            self._rag.add_turn("assistant", answer)
            return answer
        except Exception as exc:
            self._last_plan = None
            return f"❌ Could not run that suggestion. ({exc})"

    def get_insights(self) -> str:
        if not self._gemini_ready:
            return "Configure a Gemini API Key to enable auto-insights."
        if self.df.empty:
            return "No data available."

        rag_ctx = self._rag.retrieve(
            "insights overview summary performance", k=6, tier=2
        )
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

    # ── NEW: Structured query runner (used by HybridExecutor) ─

    def _run_structured_query(self, query: str) -> str:
        """
        Execute a query through the structured pipeline (Tier 1 → 2 → 3).
        Called by HybridExecutor as the structured_runner callable.
        Returns formatted answer string.
        """
        # Tier 1: Fast KPI
        fast = self._parser.fast_kpi_answer(query)
        if fast:
            return fast

        # Tier 2: Rule-based
        rule_plan = self._parser.rule_based_plan(query)
        if rule_plan:
            rule_plan = self._plan_auditor.audit(query, rule_plan)
            result = self._execute_plan(rule_plan, query)
            if result and not result.startswith("❌"):
                return result

        # Tier 3: Gemini plan
        if self._gemini_ready:
            try:
                rag_ctx = self._rag.retrieve(query, k=7, tier=3, inject_examples=True)
                raw_plan  = self._parser.gemini_plan(query, rag_ctx)
                plan      = self._validator.validate(raw_plan)
                result_df = self._sql.run(plan)
                insight   = self._insights.generate(plan, result_df)
                return self._formatter.format(plan, result_df, insight)
            except Exception as exc:
                logger.warning("_run_structured_query tier3 failed: %s", exc)

        return "❌ Could not retrieve structured data."

    # ── Stop request helpers ───────────────────────────────────

    def _is_stop_requested(self) -> bool:
        try:
            return bool(st.session_state.get("stop_requested", False))
        except Exception:
            return False

    def _clear_stop(self) -> None:
        try:
            st.session_state["stop_requested"] = False
            st.session_state["is_generating"]  = False
        except Exception:
            pass

    def rebuild_rag(self) -> None:
        self._rag.build(df=self.df, kpis=self.kpis, filters=self.filters)

    @property
    def rag_total_chunks(self) -> int:
        return self._rag.total_chunks

    # ── Private helpers ───────────────────────────────────────

    def _execute_plan(self, rule_plan, q: str, _retry: int = 0) -> Optional[str]:
        if not rule_plan:
            return None
        try:
            plan      = self._validator.validate(rule_plan)
            result_df = self._sql.run(plan)

            if result_df.empty and _retry < 2:
                repaired = self._repair_plan(plan)
                if repaired:
                    return self._execute_plan(repaired, q, _retry + 1)

            insight = self._insights.generate(plan, result_df)
            answer  = self._formatter.format(plan, result_df, insight)
            self._last_plan = plan
            self._record(q, answer)
            return answer
        except Exception as exc:
            return f"❌ {exc}"

    def _repair_plan(self, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Widen date range by 1 year when result is empty."""
        from datetime import timedelta
        try:
            sd   = datetime.strptime(plan["start_date"], "%Y-%m-%d")
            ed   = datetime.strptime(plan["end_date"],   "%Y-%m-%d")
            span = (ed - sd).days
            if span < 365:
                return {**plan,
                        "start_date": (sd - timedelta(days=365)).strftime("%Y-%m-%d"),
                        "end_date":   (ed + timedelta(days=365)).strftime("%Y-%m-%d")}
        except Exception:
            pass
        return None

    def _record(self, question: str, answer: str) -> None:
        self._last_answer = answer
        self._rag.add_turn("user",      question)
        self._rag.add_turn("assistant", answer)

    def _filter_lists(self):
        f = self.filters or {}
        return (
            list(f.get("region",   []) or []),
            list(f.get("segment",  []) or []),
            list(f.get("category", []) or []),
        )

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
            "start_date": s0,
            "end_date":   e0,
            "filters": {
                "region":   list(f.get("region",   []) or []),
                "segment":  list(f.get("segment",  []) or []),
                "category": list(f.get("category", []) or []),
            },
        }

    @staticmethod
    def _serialise(suggs: List[Suggestion]) -> List[Dict[str, Any]]:
        return [{"text": s.text, "plan": s.plan} for s in suggs]