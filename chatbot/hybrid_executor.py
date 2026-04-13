from __future__ import annotations

import logging
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.smart_router import RouteDecision

logger = logging.getLogger(__name__)

_DATA_HEADER    = "📊 **Data Overview:**"
_EXPLAIN_HEADER = "🔍 **Analysis & Explanation:**"
_DIVIDER        = "\n\n---\n\n"

_PRE_DATA_PREFIX = "=== PRE-QUERIED DATA (use ONLY these numbers) ==="
_PRE_DATA_SUFFIX = "=== END PRE-QUERIED DATA ==="

# ── Section strippers ─────────────────────────────────────────

_KEY_METRICS_BLOCK_RE = re.compile(
    r"\*\*📊\s*Key Metrics:\*\*.*?(?=\n\*\*[🔍📉✅]|\Z)",
    re.DOTALL,
)

_RECOMMENDED_ACTIONS_RE = re.compile(
    r"\*\*✅\s*Recommended Actions:\*\*.*?(?=\n\*\*[📊🔍📉]|\Z)",
    re.DOTALL,
)

_SOURCE_FOOTNOTE_RE = re.compile(
    r"\n\*Based on \d+ data (?:query|queries)\*\s*$",
    re.IGNORECASE,
)


class HybridExecutor:

    def __init__(self, structured_runner, agent_runner) -> None:
        self._run_structured = structured_runner
        self._run_agent      = agent_runner

    def execute(self, decision: "RouteDecision", original_query: str) -> str:
        structured_q = decision.structured_query or original_query
        explain_q    = decision.explain_query    or original_query

        structured_answer = ""
        try:
            structured_answer = self._run_structured(structured_q)
        except Exception as exc:
            logger.warning("Hybrid: structured failed: %s", exc)

        agent_answer = ""
        try:
            if structured_answer and not structured_answer.startswith("❌"):
                agent_input = self._build_agent_input(explain_q, structured_answer)
            else:
                agent_input = explain_q
            agent_answer = self._run_agent(agent_input)
        except Exception as exc:
            logger.warning("Hybrid: agent failed: %s", exc)

        return self._merge(structured_answer, agent_answer, original_query)

    @staticmethod
    def _build_agent_input(explain_q: str, structured_data: str) -> str:
        snippet = structured_data[:2000] if len(structured_data) > 2000 else structured_data
        return (
            f"{_PRE_DATA_PREFIX}\n"
            f"{snippet}\n"
            f"{_PRE_DATA_SUFFIX}\n\n"
            f"Question: {explain_q}"
        )

    @staticmethod
    def _clean_agent_answer(raw: str) -> str:
        text = raw

        for prefix in ("🔍 **Diagnostic Analysis:**\n\n", "🔍 **Diagnostic Analysis:**"):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip("\n")
                break

        text = _KEY_METRICS_BLOCK_RE.sub("", text)

        text = _RECOMMENDED_ACTIONS_RE.sub("", text)

        text = _SOURCE_FOOTNOTE_RE.sub("", text)

        return text.strip()

    @staticmethod
    def _merge(structured_answer: str, agent_answer: str, original_query: str) -> str:
        has_structured = bool(structured_answer and not structured_answer.startswith("❌"))
        has_agent      = bool(agent_answer      and not agent_answer.startswith("❌"))

        if has_structured and has_agent:
            clean_agent = HybridExecutor._clean_agent_answer(agent_answer)
            return (
                f"{_DATA_HEADER}\n\n"
                f"{structured_answer}"
                f"{_DIVIDER}"
                f"{_EXPLAIN_HEADER}\n\n"
                f"{clean_agent}"
            )

        if has_structured and not has_agent:
            return f"{structured_answer}\n\n*Note: Could not generate explanation — showing data only.*"

        if not has_structured and has_agent:
            return agent_answer

        return (
            f"❌ Could not answer: *{original_query}*\n\n"
            f"Try rephrasing, or ask the data and explanation questions separately."
        )