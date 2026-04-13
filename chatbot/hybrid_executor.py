"""
chatbot/hybrid_executor.py — v3

ARCHITECTURE (vs v2):

  v2: 2 LLM calls (classify + extract) → template rewrite → text to agent
  v3: 1 LLM call (classify + extract merged) → LLM rewrite → JSON to agent

  3 specific upgrades from review:
    1. Merge classify + extract into one JSON call   (-50% latency, -50% cost)
    2. LLM-based rewrite instead of templates        (flexible, no hardcode)
    3. Reason on structure: agent receives JSON data  (grounded, less hallucination)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.smart_router import RouteDecision

logger = logging.getLogger(__name__)

_DATA_HEADER    = "📊 **Data Overview:**"
_EXPLAIN_HEADER = "🔍 **Analysis & Explanation:**"
_DIVIDER        = "\n\n---\n\n"

_PRE_DATA_PREFIX = "=== PRE-QUERIED DATA (use ONLY these numbers) ==="
_PRE_DATA_SUFFIX = "=== END PRE-QUERIED DATA ==="

_KEY_METRICS_BLOCK_RE = re.compile(
    r"\*\*📊\s*Key Metrics:\*\*.*?(?=\n\*\*[🔍📉✅]|\Z)", re.DOTALL,
)
_RECOMMENDED_ACTIONS_RE = re.compile(
    r"\*\*✅\s*Recommended Actions:\*\*.*?(?=\n\*\*[📊🔍📉]|\Z)", re.DOTALL,
)
_SOURCE_FOOTNOTE_RE = re.compile(
    r"\n\*Based on \d+ data (?:query|queries)\*\s*$", re.IGNORECASE,
)

# ── Upgrade 1: Single merged prompt ───────────────────────────────────────
#
# One LLM call returns BOTH intent AND entities.
# Input: ~200 tokens. Output: ~100 tokens (JSON).
# vs v2: 2 calls × ~130 tokens = ~260 tokens.
#
_MERGED_ANALYSIS_PROMPT = """You are an analytics query analyzer.
Given a question and a data answer, return a single JSON object.

Question: "{question}"

Data answer (truncated):
\"\"\"{answer}\"\"\"

Return ONLY this JSON (no markdown, no explanation):
{{
  "intent": "<one of: bottom_performer | top_performer | loss_analysis | trend_analysis | comparison | general>",
  "top":    "<name of best/highest entity from data, or null>",
  "bottom": "<name of worst/lowest entity from data, or null>",
  "all_entities": ["<all entity names in rank order, best first>"],
  "loss_items":   ["<entity names that have negative profit>"],
  "metric":    "<main metric: profit | sales | orders | margin, or null>",
  "dimension": "<grouping dimension: region | segment | category | sub_category, or null>",
  "key_facts": ["<up to 3 key numeric facts from the data, e.g. Central profit=$39,706>"]
}}

Rules:
- Entity names must be copied exactly as they appear in the answer
- intent = 'general' when no specific pattern detected
- key_facts must contain real numbers from the answer"""


# ── Upgrade 2: LLM-based rewrite (replaces templates) ─────────────────────
#
# LLM rewrites the generic question using actual entities + intent.
# Flexible — handles any phrasing, no hardcoded strings.
#
_REWRITE_PROMPT = """Rewrite this vague analytical question into a precise, data-grounded one.

Original question: "{question}"
Intent detected:   {intent}
Entities from data:
{entities_json}

Rules:
1. Replace vague phrases ("the lowest region", "these products") with the ACTUAL names
2. Keep the question concise — one sentence, max 25 words
3. End with: "Use the exact figures from the data provided."
4. Do NOT answer the question — only rewrite it

Return ONLY the rewritten question, nothing else."""


# ── Upgrade 3: Structured data for agent ──────────────────────────────────
#
# Build a compact JSON payload instead of injecting raw text.
# Agent receives: structured facts + enriched question.
#
def _build_structured_payload(
    enriched_q: str,
    structured_answer: str,
    analysis: Dict[str, Any],
) -> str:
    """
    Build the agent input as a structured JSON block rather than raw text.

    This separates DATA (structured facts extracted by LLM) from QUESTION,
    so the agent reasons on clean numbers rather than on formatted prose.
    """
    entities_block = {
        "top":        analysis.get("top"),
        "bottom":     analysis.get("bottom"),
        "all_ranked": analysis.get("all_entities", []),
        "loss_items": analysis.get("loss_items", []),
        "metric":     analysis.get("metric"),
        "dimension":  analysis.get("dimension"),
        "key_facts":  analysis.get("key_facts", []),
    }

    # Remove nulls to keep payload clean
    entities_block = {k: v for k, v in entities_block.items() if v}

    structured_section = json.dumps(entities_block, ensure_ascii=False, indent=2)

    # Also include a snippet of the original text for any facts the JSON missed
    text_snippet = structured_answer[:800] if len(structured_answer) > 800 else structured_answer

    return (
        f"{_PRE_DATA_PREFIX}\n"
        f"[Structured entities]\n"
        f"{structured_section}\n\n"
        f"[Full answer text — use for additional context]\n"
        f"{text_snippet}\n"
        f"{_PRE_DATA_SUFFIX}\n\n"
        f"Question: {enriched_q}"
    )


# ── Core functions ─────────────────────────────────────────────────────────

def _analyze_once(
    explain_q: str,
    structured_answer: str,
    gemini_client: Any,
    model_name: str,
) -> Dict[str, Any]:
    """
    Single LLM call that classifies intent AND extracts entities simultaneously.

    Upgrade 1 vs v2: replaces 2 separate calls with 1.
    Returns the full analysis dict or an empty fallback.
    """
    _empty: Dict[str, Any] = {
        "intent": "general", "top": None, "bottom": None,
        "all_entities": [], "loss_items": [], "key_facts": [],
        "metric": None, "dimension": None,
    }

    if not gemini_client or not model_name:
        return _empty

    try:
        from google.genai import types as genai_types
        prompt = _MERGED_ANALYSIS_PROMPT.format(
            question=explain_q,
            answer=structured_answer[:1200],
        )
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=220,
            ),
        )
        raw     = (getattr(resp, "text", "") or "").strip()
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            result = json.loads(m.group(0)) if m else {}

        merged = {**_empty, **result}
        logger.debug(
            "Hybrid analyze_once: intent=%s top=%s bottom=%s loss=%s",
            merged["intent"], merged["top"], merged["bottom"], merged["loss_items"],
        )
        return merged
    except Exception as exc:
        logger.warning("analyze_once failed (%s) — using empty analysis", exc)
        return _empty


def _rewrite_question_llm(
    explain_q: str,
    analysis: Dict[str, Any],
    gemini_client: Any,
    model_name: str,
) -> str:
    """
    Upgrade 2: LLM-based rewrite instead of hardcoded templates.

    Replaces the template dict in v2 with a flexible LLM call that handles
    any question phrasing, including complex or compound questions.
    Falls back to explain_q on failure.
    """
    if not gemini_client or not model_name:
        return explain_q

    intent = analysis.get("intent", "general")
    if intent == "general":
        return explain_q

    # Build compact entities summary for the prompt
    entities_lines = []
    if analysis.get("top"):
        entities_lines.append(f"  top entity:   {analysis['top']}")
    if analysis.get("bottom"):
        entities_lines.append(f"  bottom entity: {analysis['bottom']}")
    if analysis.get("all_entities"):
        entities_lines.append(f"  all ranked:   {', '.join(analysis['all_entities'])}")
    if analysis.get("loss_items"):
        entities_lines.append(f"  loss-making:  {', '.join(analysis['loss_items'])}")
    if analysis.get("key_facts"):
        entities_lines.append(f"  key facts:    {'; '.join(analysis['key_facts'])}")
    if analysis.get("metric"):
        entities_lines.append(f"  metric:       {analysis['metric']}")

    entities_json = "\n".join(entities_lines) if entities_lines else "  (none found)"

    try:
        from google.genai import types as genai_types
        prompt = _REWRITE_PROMPT.format(
            question=explain_q,
            intent=intent,
            entities_json=entities_json,
        )
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=80,
            ),
        )
        rewritten = (getattr(resp, "text", "") or "").strip()
        if rewritten and len(rewritten) > 10:
            logger.debug("Hybrid rewrite: %r → %r", explain_q[:50], rewritten[:70])
            return rewritten
        return explain_q
    except Exception as exc:
        logger.warning("LLM rewrite failed (%s) — using original", exc)
        return explain_q


def _enrich_explain_query(
    explain_q: str,
    structured_answer: str,
    original_query: str,
    gemini_client: Any = None,
    model_name: str = "",
) -> tuple[str, Dict[str, Any]]:
    """
    v3 enrichment pipeline — returns (enriched_question, analysis_dict):

      1. _analyze_once()          — 1 LLM call: intent + entities merged
      2. _rewrite_question_llm()  — 1 LLM call: LLM-based rewrite (not template)

    Total: 2 LLM calls per hybrid query (same as v2 total, but smarter).
    The returned analysis dict is used in step 3 to build the structured payload.

    Degrades gracefully:
      - No LLM → (original explain_q, empty analysis)
      - LLM fails at any step → falls back at that step
    """
    _empty_analysis: Dict[str, Any] = {
        "intent": "general", "top": None, "bottom": None,
        "all_entities": [], "loss_items": [], "key_facts": [],
        "metric": None, "dimension": None,
    }

    if not gemini_client or not model_name:
        return explain_q, _empty_analysis

    # Step 1: single merged call
    analysis = _analyze_once(explain_q, structured_answer, gemini_client, model_name)

    if analysis.get("intent") == "general":
        return explain_q, analysis

    has_entities = bool(
        analysis.get("bottom") or analysis.get("top") or analysis.get("loss_items")
    )
    if not has_entities:
        return explain_q, analysis

    # Step 2: LLM-based rewrite
    enriched_q = _rewrite_question_llm(explain_q, analysis, gemini_client, model_name)

    return enriched_q, analysis


# ── HybridExecutor v3 ──────────────────────────────────────────────────────

class HybridExecutor:
    """
    Hybrid query executor v3.

    Upgrades from v2:
      - 1 merged LLM call (intent + entities) instead of 2 separate calls
      - LLM-based question rewrite instead of hardcoded templates
      - Structured JSON payload sent to agent (reason on structure, not text)
    """

    def __init__(
        self,
        structured_runner,
        agent_runner,
        gemini_client: Any = None,
        model_name: str = "",
    ) -> None:
        self._run_structured = structured_runner
        self._run_agent      = agent_runner
        self._gemini_client  = gemini_client
        self._model_name     = model_name

    def execute(self, decision: "RouteDecision", original_query: str) -> str:
        structured_q = decision.structured_query or original_query
        explain_q    = decision.explain_query    or original_query

        # Step 1: Get structured data (SQL pipeline)
        structured_answer = ""
        try:
            structured_answer = self._run_structured(structured_q)
        except Exception as exc:
            logger.warning("Hybrid: structured failed: %s", exc)

        # Steps 2-3: Enrich + build agent input
        agent_answer = ""
        try:
            if structured_answer and not structured_answer.startswith("❌"):
                enriched_q, analysis = _enrich_explain_query(
                    explain_q, structured_answer, original_query,
                    gemini_client=self._gemini_client,
                    model_name=self._model_name,
                )
                # Upgrade 3: structured JSON payload instead of raw text
                agent_input = _build_structured_payload(
                    enriched_q, structured_answer, analysis
                )
            else:
                agent_input = explain_q
            agent_answer = self._run_agent(agent_input)
        except Exception as exc:
            logger.warning("Hybrid: agent failed: %s", exc)

        return self._merge(structured_answer, agent_answer, original_query)

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
                f"{_DATA_HEADER}\n\n{structured_answer}"
                f"{_DIVIDER}"
                f"{_EXPLAIN_HEADER}\n\n{clean_agent}"
            )
        if has_structured and not has_agent:
            return f"{structured_answer}\n\n*Note: Could not generate explanation — showing data only.*"
        if not has_structured and has_agent:
            return agent_answer
        return (
            f"❌ Could not answer: *{original_query}*\n\n"
            f"Try rephrasing, or ask the data and explanation questions separately."
        )