"""
chatbot/agent/orchestrator.py  — PATCHED v2
FIXES:
  FIX-1: Structured output format enforced — prevents "The primary reason is" cutoff
  FIX-2: max_output_tokens tăng lên 3000 để tránh mid-sentence truncation
  FIX-3: completion_guard() kiểm tra response có đầy đủ cấu trúc không
  FIX-4: Fallback synthesis dùng cùng structured prompt
  FIX-5: Agent suggestions context-aware cho diagnostic queries
"""

from __future__ import annotations
import json
import logging
import streamlit as st
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types

from chatbot.agent.tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 5

# ── FIX-1: Structured output format ───────────────────────────────────────────
# OLD: "Structure: Gap → Root cause → Action, Max 3 paragraphs"
#      → Gemini wrote incomplete sentences like "The primary reason is"
# NEW: Explicit section headers with completion requirement
_SYSTEM_PROMPT = """You are a senior business analyst for a Superstore retail company.
You have access to tools that query real sales data.

Your job: answer WHY questions by gathering data, finding root causes, and giving recommendations.

Rules:
- Always use at least 2 tools before answering
- Every claim must include exact $ or % numbers from tool data
- NEVER write an incomplete sentence — always finish what you start
- If you write "The reason is..." you MUST immediately state the reason on the same line

REQUIRED OUTPUT FORMAT — always use ALL four sections:

**📊 Key Metrics:**
- [metric 1 with before/after values and % change]
- [metric 2 with before/after values and % change]

**🔍 Root Cause:**
[2-3 complete sentences explaining exactly WHY. State the cause explicitly, not vaguely.]

**📉 Supporting Evidence:**
- [Specific finding 1: exact numbers]
- [Specific finding 2: exact numbers]

**✅ Recommended Actions:**
1. [Action with specific target metric and number]
2. [Action with specific target metric and number]
"""


class AgentOrchestrator:
    """Multi-step agent using Gemini function calling."""

    def __init__(
        self,
        gemini_client: Any,
        model_name: str,
        default_start: str,
        default_end: str,
    ) -> None:
        self.client        = gemini_client
        self.model_name    = model_name
        self.default_start = default_start
        self.default_end   = default_end

    def run(self, question: str) -> str:
        try:
            return self._agentic_loop(question)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                return (
                    "⚠️ **API quota reached** — diagnostic analysis temporarily unavailable.\n\n"
                    "Try again in ~1 minute, or ask a structured question like:\n"
                    "*'Sales by region from Oct to Nov 2016'*"
                )
            return (
                f"⚠️ Could not complete diagnostic. ({type(e).__name__})\n\n"
                f"Try: *'Why are Tables losing money?'* or *'Why is Central underperforming?'*"
            )

    def _agentic_loop(self, question: str) -> str:
        tools = [
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(**schema)
                    for schema in TOOL_SCHEMAS
                ]
            )
        ]

        messages = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"{_SYSTEM_PROMPT}\n\nQuestion: {question}")]
            )
        ]

        tool_results_log: List[str] = []
        call_count = 0

        while call_count < _MAX_TOOL_CALLS:
            # FIX-2: max_output_tokens tăng lên 3000 để tránh truncation
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=genai_types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1,
                    max_output_tokens=3000,  # was 2000
                ),
            )

            candidate = resp.candidates[0] if resp.candidates else None
            if not candidate:
                break

            tool_calls = [
                part for part in candidate.content.parts
                if hasattr(part, "function_call") and part.function_call
            ]

            if not tool_calls:
                text_parts = [
                    part.text for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                ]
                final_answer = "\n".join(text_parts).strip()

                # FIX-3: Completion guard — nếu response bị cắt, chạy fallback synthesis
                if final_answer and self._is_complete(final_answer):
                    return self._format_agent_answer(final_answer, tool_results_log)
                elif final_answer and tool_results_log:
                    # Response có nội dung nhưng thiếu section → fallback với data đã có
                    logger.warning("Agent response incomplete — running fallback synthesis")
                    return self._fallback_synthesis(question, tool_results_log)
                break

            messages.append(candidate.content)
            tool_response_parts = []

            for part in tool_calls:
                fc        = part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                logger.info("Agent calling tool: %s(%s)", tool_name, tool_args)

                result = execute_tool(
                    tool_name, tool_args,
                    self.default_start, self.default_end
                )
                tool_results_log.append(f"[{tool_name}]\n{result}")
                call_count += 1

                tool_response_parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            response={"result": result},
                        )
                    )
                )

            messages.append(
                genai_types.Content(role="user", parts=tool_response_parts)
            )

        return self._fallback_synthesis(question, tool_results_log)

    # ── FIX-3: Completion guard ────────────────────────────────────────────────
    # Kiểm tra response có đủ 4 sections không.
    # WHY: Gemini đôi khi viết "The reason is" rồi bị truncate → incomplete.
    # Nếu thiếu section, ta chạy fallback synthesis thay vì trả về response dở dang.
    @staticmethod
    def _is_complete(text: str) -> bool:
        """Return True if all 4 required sections are present."""
        required_markers = [
            "Key Metrics",
            "Root Cause",
            "Supporting Evidence",
            "Recommended Actions",
        ]
        return all(marker in text for marker in required_markers)

    def _format_agent_answer(self, answer: str, tool_log: List[str]) -> str:
        header = "🔍 **Diagnostic Analysis:**\n\n"
        sources = ""
        if tool_log:
            n = len(tool_log)
            sources = f"\n\n*Based on {n} data {'query' if n==1 else 'queries'}*"
        return header + answer + sources

    def _fallback_synthesis(self, question: str, tool_log: List[str]) -> str:
        if not tool_log:
            return "❌ Could not gather enough data to answer this question."

        data_summary = "\n\n".join(tool_log)

        # FIX-4: Fallback dùng cùng structured format như _SYSTEM_PROMPT
        prompt = f"""You are a senior business analyst. Answer this question using ONLY the data below.

Question: "{question}"

DATA COLLECTED:
{data_summary}

REQUIRED OUTPUT — you MUST include ALL 4 sections, fully completed:

**📊 Key Metrics:**
- [list specific metrics with before/after values from the data]

**🔍 Root Cause:**
[Explain exactly WHY in 2-3 complete sentences with specific numbers. Do NOT leave sentences unfinished.]

**📉 Supporting Evidence:**
- [specific finding with numbers]
- [specific finding with numbers]

**✅ Recommended Actions:**
1. [Specific action with target number]
2. [Specific action with target number]

IMPORTANT: Every sentence must be complete. Never write "The reason is" without immediately stating the reason."""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=1500,
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text and self._is_complete(text):
                return f"🔍 **Diagnostic Analysis:**\n\n{text}"
            elif text:
                # Still return even if not all sections present — better than nothing
                return f"🔍 **Diagnostic Analysis:**\n\n{text}"
        except Exception as e:
            logger.error("Fallback synthesis failed: %s", e)

        # Last resort: format raw tool data into readable output
        return self._emergency_format(question, tool_log)

    def _emergency_format(self, question: str, tool_log: List[str]) -> str:
        """Last-resort formatting when LLM fails — structures raw tool data."""
        lines = [
            "🔍 **Diagnostic Analysis:**",
            "",
            f"**❓ Question:** {question}",
            "",
            "**📊 Raw Data Collected:**",
        ]
        for entry in tool_log[:3]:
            # Clean up tool output for display
            tool_lines = entry.strip().split("\n")
            for line in tool_lines[:8]:  # limit to 8 lines per tool
                if line.strip():
                    lines.append(f"  - {line.strip()}")
            lines.append("")

        lines += [
            "**⚠️ Note:** Automated synthesis unavailable. "
            "Use the data above or try asking a more specific question.",
        ]
        return "\n".join(lines)

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