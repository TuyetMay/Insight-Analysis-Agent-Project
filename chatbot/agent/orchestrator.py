"""
chatbot/agent/orchestrator.py  — PATCHED v3
NEW: Assumption Validator — detects premise vs data contradiction
NEW: Clean emergency format (no internal error messages shown to user)
"""

from __future__ import annotations
import logging
import streamlit as st
from typing import Any, List

from google.genai import types as genai_types

from chatbot.agent.tools import TOOL_SCHEMAS, execute_tool
from chatbot.agent.assumption_validator import validate_assumptions

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 5

_SYSTEM_PROMPT = """You are a senior business analyst for a Superstore retail company.
You have access to tools that query real sales data.

Your job: answer WHY questions by gathering data, finding root causes, and giving recommendations.

Rules:
- Always use at least 2 tools before answering
- Every claim must include exact $ or % numbers from tool data
- NEVER write an incomplete sentence — always finish what you start

REQUIRED OUTPUT FORMAT — always use ALL four sections:

**📊 Key Metrics:**
- [metric 1 with before/after values and % change]
- [metric 2 with before/after values and % change]

**🔍 Root Cause:**
[2-3 complete sentences explaining exactly WHY with specific numbers.]

**📉 Supporting Evidence:**
- [Specific finding 1: exact numbers]
- [Specific finding 2: exact numbers]

**✅ Recommended Actions:**
1. [Action with specific target metric and number]
2. [Action with specific target metric and number]
"""


class AgentOrchestrator:

    def __init__(self, gemini_client: Any, model_name: str,
                 default_start: str, default_end: str) -> None:
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
                    "⚠️ **API quota reached** — try again in ~1 minute, or ask:\n"
                    "*'Sales by region from Oct to Nov 2016'*"
                )
            return f"⚠️ Could not complete diagnostic. ({type(e).__name__})"

    def _agentic_loop(self, question: str) -> str:
        tools = [
            genai_types.Tool(
                function_declarations=[
                    genai_types.FunctionDeclaration(
                        name=schema["name"],
                        description=schema["description"],
                        parameters=genai_types.Schema(
                            type=genai_types.Type.OBJECT,
                            properties={
                                k: genai_types.Schema(type=genai_types.Type.STRING)
                                for k in schema["parameters"].get("properties", {})
                            },
                        ),
                    )
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
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=genai_types.GenerateContentConfig(
                    tools=tools, temperature=0.1, max_output_tokens=3000,
                ),
            )
            candidate = resp.candidates[0] if resp.candidates else None
            if not candidate:
                break

            tool_calls = [
                p for p in candidate.content.parts
                if hasattr(p, "function_call") and p.function_call
            ]

            if not tool_calls:
                text_parts = [
                    p.text for p in candidate.content.parts
                    if hasattr(p, "text") and p.text
                ]
                final_answer = "\n".join(text_parts).strip()

                if final_answer:
                    # Validate assumptions BEFORE returning
                    contradiction = validate_assumptions(question, tool_results_log)
                    if contradiction:
                        return contradiction

                    if self._is_complete(final_answer):
                        return self._format_agent_answer(final_answer, tool_results_log)
                    elif tool_results_log:
                        return self._fallback_synthesis(question, tool_results_log)
                break

            messages.append(candidate.content)
            tool_response_parts = []

            for part in tool_calls:
                fc        = part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                result    = execute_tool(tool_name, tool_args, self.default_start, self.default_end)
                tool_results_log.append(f"[{tool_name}]\n{result}")
                call_count += 1
                tool_response_parts.append(
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name, response={"result": result},
                        )
                    )
                )

            messages.append(genai_types.Content(role="user", parts=tool_response_parts))

        # Also check on fallback path
        if tool_results_log:
            contradiction = validate_assumptions(question, tool_results_log)
            if contradiction:
                return contradiction

        return self._fallback_synthesis(question, tool_results_log)

    @staticmethod
    def _is_complete(text: str) -> bool:
        return all(m in text for m in ["Key Metrics", "Root Cause", "Supporting Evidence", "Recommended Actions"])

    def _format_agent_answer(self, answer: str, tool_log: List[str]) -> str:
        header  = "🔍 **Diagnostic Analysis:**\n\n"
        sources = f"\n\n*Based on {len(tool_log)} data {'query' if len(tool_log)==1 else 'queries'}*" if tool_log else ""
        return header + answer + sources

    def _fallback_synthesis(self, question: str, tool_log: List[str]) -> str:
        if not tool_log:
            return "❌ Could not gather enough data to answer this question."

        prompt = f"""You are a senior business analyst. Answer: "{question}"

DATA:
{chr(10).join(tool_log)}

Output ALL 4 sections — complete sentences only:

**📊 Key Metrics:**
- [before/after values with % change]

**🔍 Root Cause:**
[WHY — 2-3 sentences with numbers]

**📉 Supporting Evidence:**
- [finding with numbers]

**✅ Recommended Actions:**
1. [action with number]
2. [action with number]"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=1500),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return f"🔍 **Diagnostic Analysis:**\n\n{text}"
        except Exception as e:
            logger.error("Fallback synthesis failed: %s", e)

        return self._emergency_format(tool_log)

    def _emergency_format(self, tool_log: List[str]) -> str:
        """Clean data display — no internal error messages."""
        lines = ["🔍 **Data retrieved for your question:**", ""]
        for entry in tool_log[:4]:
            entry_lines = entry.strip().split("\n")
            if entry_lines:
                header = entry_lines[0].strip("[]").replace("_", " ").title()
                lines.append(f"**{header}:**")
                for line in entry_lines[1:7]:
                    if line.strip():
                        lines.append(f"  {line.strip()}")
                lines.append("")
        lines.append("*Use the suggested follow-ups below to explore further.*")
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