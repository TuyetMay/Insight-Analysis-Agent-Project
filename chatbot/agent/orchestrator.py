"""
chatbot/agent/orchestrator.py
AI Agent: Gemini tự plan + gọi tools + synthesize answer.

Flow:
  1. Gửi query + tool schemas → Gemini plans
  2. Gemini gọi tools (function calling)
  3. Execute tools → collect results
  4. Gemini synthesize → final answer

Max 5 tool calls để tránh infinite loop.
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

_SYSTEM_PROMPT = """You are a senior business analyst for a Superstore retail company.
You have access to tools that query real sales data.

Your job: answer WHY questions by:
1. Gathering relevant data using tools
2. Comparing entities (regions, categories) to identify gaps
3. Finding anomalies (loss-makers, high discounts)
4. Synthesizing findings into a clear diagnosis

Rules:
- Always use at least 2 tools before answering
- Every claim must be backed by tool data
- Be specific: use exact numbers
- Structure: Gap → Root cause → Action
- Max 3 paragraphs
"""


class AgentOrchestrator:
    """
    Multi-step agent using Gemini function calling.
    """

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
                    "**Based on available data:**\n"
                    f"- Ask again in ~1 minute when quota resets\n"
                    f"- Or try a structured question: *'sales trend from Dec 2016 to Feb 2017'*"
                )
            return (
                f"⚠️ Could not complete diagnostic. ({type(e).__name__})\n\n"
                f"Try: *'Why are Tables losing money?'* or *'Why is Central underperforming?'*"
            )

    def _agentic_loop(self, question: str) -> str:
        """
        Gemini function-calling loop.
        Continues until Gemini stops requesting tools or max calls reached.
        """
        # Build tool declarations cho Gemini
        tools = [
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(**schema)
                    for schema in TOOL_SCHEMAS
                ]
            )
        ]

        # Conversation history
        messages = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"{_SYSTEM_PROMPT}\n\nQuestion: {question}")]
            )
        ]

        tool_results_log: List[str] = []
        call_count = 0

        while call_count < _MAX_TOOL_CALLS:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=genai_types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.1,
                    max_output_tokens=2000,
                ),
            )

            candidate = resp.candidates[0] if resp.candidates else None
            if not candidate:
                break

            # Check if Gemini wants to call tools
            tool_calls = [
                part for part in candidate.content.parts
                if hasattr(part, "function_call") and part.function_call
            ]

            if not tool_calls:
                # No more tool calls — extract final text answer
                text_parts = [
                    part.text for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                ]
                final_answer = "\n".join(text_parts).strip()
                if final_answer:
                    return self._format_agent_answer(final_answer, tool_results_log)
                break

            # Execute each tool call
            messages.append(candidate.content)  # add assistant message
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

            # Add tool results to conversation
            messages.append(
                genai_types.Content(role="user", parts=tool_response_parts)
            )

        # Fallback: nếu loop kết thúc không có answer
        return self._fallback_synthesis(question, tool_results_log)

    def _format_agent_answer(self, answer: str, tool_log: List[str]) -> str:
        """Wrap agent answer với header và data sources."""
        header = "🔍 **Diagnostic Analysis:**\n\n"
        sources = ""
        if tool_log:
            n = len(tool_log)
            sources = f"\n\n*Based on {n} data {'query' if n==1 else 'queries'}*"
        return header + answer + sources

    def _fallback_synthesis(self, question: str, tool_log: List[str]) -> str:
        """Nếu agent loop kết thúc không answer được — synthesize từ tool results."""
        if not tool_log:
            return "❌ Could not gather enough data to answer this question."

        data_summary = "\n\n".join(tool_log)
        prompt = f"""Based on this data, answer the question: "{question}"

        DATA COLLECTED:
        {data_summary}

        Write a concise diagnostic answer (2-3 paragraphs).
        Lead with the most important finding.
        Include specific numbers.
        End with 1 actionable recommendation."""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1, max_output_tokens=800
                ),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return f"🔍 **Diagnostic Analysis:**\n\n{text}"
        except Exception:
            pass

        return f"🔍 **Data collected but synthesis failed.**\n\n" + "\n\n".join(tool_log[:2])

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