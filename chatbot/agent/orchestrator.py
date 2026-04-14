from __future__ import annotations
import logging
import re
import streamlit as st
from typing import Any, List, Optional, Tuple

from google.genai import types as genai_types

from chatbot.agent.tools import TOOL_SCHEMAS, execute_tool
from chatbot.agent.assumption_validator import validate_assumptions

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 8

_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_LAST_DAY = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

_MONTH_YEAR_RE = re.compile(
    r'\b(january|february|march|april|may|june|july|august|september|october|november|december'
    r'|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(20\d{2})\b',
    re.IGNORECASE
)
_YEAR_ONLY_RE = re.compile(r'\b(20\d{2})\b')
_FROM_TO_RE   = re.compile(
    r'\bfrom\s+(\w+\s+20\d{2})\s+to\s+(\w+\s+20\d{2})\b', re.IGNORECASE
)

def _parse_month_year(text: str) -> Optional[Tuple[str, str]]:
    m = _MONTH_YEAR_RE.search(text)
    if m:
        month = _MONTH_MAP[m.group(1).lower()]
        year  = int(m.group(2))
        last  = _MONTH_LAST_DAY.get(month, 30)
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"
    return None

def _extract_date_pairs(question: str) -> List[Tuple[str, str]]:
    ql = question.lower()
    pairs: List[Tuple[str, str]] = []
    for m in _MONTH_YEAR_RE.finditer(ql):
        month = _MONTH_MAP[m.group(1).lower()]
        year  = int(m.group(2))
        last  = _MONTH_LAST_DAY.get(month, 30)
        pairs.append((f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"))
    if not pairs:
        for m in _YEAR_ONLY_RE.finditer(ql):
            yr = int(m.group(1))
            pairs.append((f"{yr}-01-01", f"{yr}-12-31"))
    return pairs[:2]


def _sort_current_previous(
    pairs: List[Tuple[str, str]]
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    a, b = pairs[0], pairs[1]
    if a[0] >= b[0]:
        return a, b
    else:
        return b, a

def _is_hallucinated_round_number(text: str) -> bool:
    if re.search(r'\$\d+\.0[MB]\b', text):
        return True
    round_thousands = re.findall(r'\$(\d+),000\b', text)
    for n in round_thousands:
        if int(n) % 5 == 0 and int(n) >= 95:
            return True
    if re.search(r'\b(1[,.]?000|1[,.]?100)\s+orders\b', text, re.IGNORECASE):
        return True
    return False


_FORBIDDEN_FIELD_RE = re.compile(
    r'\b(cogs|cost of goods sold|inventory|headcount|employees?|'
    r'operating expenses?|capex|depreciation|tax rate|overhead)\b',
    re.IGNORECASE,
)

def _scrub_forbidden_fields(text: str) -> str:
    if _FORBIDDEN_FIELD_RE.search(text):
        disclaimer = (
            "\n\n> ⚠️ *Note: The Superstore dataset does not contain COGS, inventory, "
            "operating expenses, or tax data. Root-cause analysis is based only on "
            "sales, profit, discount, and order volume.*"
        )
        cleaned = re.sub(
            r'[^.!?]*\b(cogs|cost of goods sold|inventory|operating expenses?)\b[^.!?]*[.!?]',
            '',
            text,
            flags=re.IGNORECASE,
        )
        return cleaned.strip() + disclaimer
    return text


_SYSTEM_PROMPT = """You are a senior business analyst and data analyst with access to a LIVE Superstore database.

CRITICAL RULES — violation = your answer is discarded:
1. NEVER invent, estimate, or approximate any number. Every $ and % MUST come from a tool call.
2. ALWAYS call compare_periods or get_trend FIRST using the EXACT dates from the question.
3. Date mapping: "October to December 2016" → current_start="2016-10-01", current_end="2016-12-31", previous_start="2015-10-01", previous_end="2015-12-31".
4. PREMISE CHECK FIRST: Before explaining WHY, confirm the premise is true from tool results.
   - If profit grew FASTER than sales → premise "profit grew slower" is FALSE. State the correction.
   - If orders DECREASED → premise "orders increased" is FALSE. State the correction.
   - Do NOT produce a root-cause analysis for a false premise.
5. Do NOT use placeholder numbers like $100,000 or 1,000 orders. These are not in the database.
6. FORBIDDEN FIELDS — these do NOT exist in Superstore. NEVER mention them:
   COGS, cost of goods sold, inventory, operating expenses, overhead, headcount, employees, tax, depreciation.
7. MARGIN MATH: profit_margin = profit / sales x 100. Never invert.
8. Use AT LEAST 3 tool calls before writing your final answer.

REQUIRED OUTPUT FORMAT — use ALL four sections with REAL numbers from tools:

**📊 Key Metrics:**
- [exact metric] before: $X → after: $Y (Z% change, from tool result)

**🔍 Root Cause:**
[2-3 sentences with the actual numbers from tool results explaining WHY]

**📉 Supporting Evidence:**
- [Specific finding with exact numbers from tool calls]

**✅ Recommended Actions:**
1. [Action with specific target]
2. [Action with specific target]
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

        forced_tool_results = self._run_forced_queries(question)

        forced_context = ""
        if forced_tool_results:
            forced_context = (
                "\n\n=== PRE-QUERIED DATA (use ONLY these numbers) ===\n"
                + "\n".join(forced_tool_results)
                + "\n=== END PRE-QUERIED DATA ===\n\n"
                "The numbers above are from the live database. "
                "Use them directly. Do not invent alternative numbers.\n\n"
            )

        messages = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=f"{_SYSTEM_PROMPT}\n\n{forced_context}Question: {question}")]
            )
        ]
        tool_results_log: List[str] = list(forced_tool_results)
        call_count = len(forced_tool_results)

        while call_count < _MAX_TOOL_CALLS:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=genai_types.GenerateContentConfig(
                    tools=tools, temperature=0.0, max_output_tokens=3000,
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
                    if _is_hallucinated_round_number(final_answer):
                        logger.warning("Detected hallucinated round numbers — falling back to synthesis")
                        return self._fallback_synthesis(question, tool_results_log)

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
                # FIX: fc.args may be a protobuf MapComposite in newer google-genai
                # versions — dict() raises TypeError, use .items() instead.
                if fc.args:
                    try:
                        tool_args = dict(fc.args)
                    except TypeError:
                        tool_args = {k: str(v) for k, v in fc.args.items()}
                else:
                    tool_args = {}

                result = execute_tool(tool_name, tool_args, self.default_start, self.default_end)
                tool_results_log.append(f"[{tool_name}]\n{result}")   # ← was missing
                call_count += 1                                         # ← was missing
                tool_response_parts.append(                             # ← was missing
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name, response={"result": result},
                        )
                    )
                )

            messages.append(genai_types.Content(role="user", parts=tool_response_parts))

        if tool_results_log:
            contradiction = validate_assumptions(question, tool_results_log)
            if contradiction:
                return contradiction

        return self._fallback_synthesis(question, tool_results_log)

    def _run_forced_queries(self, question: str) -> List[str]:
        results: List[str] = []
        date_pairs = _extract_date_pairs(question)

        if not date_pairs:
            return results

        if len(date_pairs) >= 2:
            current_pair, previous_pair = _sort_current_previous(date_pairs)
            for metric in ("sales", "profit", "orders"):
                args = {
                    "metric":         metric,
                    "current_start":  current_pair[0],
                    "current_end":    current_pair[1],
                    "previous_start": previous_pair[0],
                    "previous_end":   previous_pair[1],
                }
                result = execute_tool("compare_periods", args, self.default_start, self.default_end)
                results.append(
                    f"[compare_periods / {metric}]\n"
                    f"NOTE: current=LATER period ({current_pair[0]}..{current_pair[1]}), "
                    f"previous=EARLIER period ({previous_pair[0]}..{previous_pair[1]})\n"
                    f"{result}"
                )
        else:
            for metric in ("sales", "profit", "orders"):
                args = {
                    "metric":        metric,
                    "current_start": date_pairs[0][0],
                    "current_end":   date_pairs[0][1],
                }
                result = execute_tool("compare_periods", args, self.default_start, self.default_end)
                results.append(f"[compare_periods / {metric}]\n{result}")

        return results

    @staticmethod
    def _is_complete(text: str) -> bool:
        return all(m in text for m in ["Key Metrics", "Root Cause", "Supporting Evidence", "Recommended Actions"])

    def _format_agent_answer(self, answer: str, tool_log: List[str]) -> str:
        header  = "🔍 **Diagnostic Analysis:**\n\n"
        sources = f"\n\n*Based on {len(tool_log)} data {'query' if len(tool_log)==1 else 'queries'}*" if tool_log else ""
        return header + _scrub_forbidden_fields(answer) + sources

    def _fallback_synthesis(self, question: str, tool_log: List[str]) -> str:
        if not tool_log:
            return "❌ Could not gather enough data to answer this question."

        date_pairs = _extract_date_pairs(question)
        period_hint = ""
        if date_pairs:
            period_hint = f"\nThe question asks about the period(s): {date_pairs}\n"

        prompt = f"""You are a senior business analyst. Answer: "{question}"
{period_hint}
=== REAL DATA FROM DATABASE (USE ONLY THESE NUMBERS) ===
{chr(10).join(tool_log)}
=== END DATA ===

RULES:
- Use ONLY the numbers shown above. Do NOT invent any figures.
- If the user's premise contradicts the data, state the correction clearly at the top.
- Output ALL 4 sections using only real numbers from the data above:

**📊 Key Metrics:**
- [before/after values with % change — from data above]

**🔍 Root Cause:**
[WHY — 2-3 sentences with exact numbers from data]

**📉 Supporting Evidence:**
- [finding with numbers from data]

**✅ Recommended Actions:**
1. [action with number]
2. [action with number]"""

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=1500),
            )
            text = (getattr(resp, "text", "") or "").strip()
            if text and not _is_hallucinated_round_number(text):
                return f"🔍 **Diagnostic Analysis:**\n\n{_scrub_forbidden_fields(text)}"
            elif text:
                return (
                    "⚠️ *Note: Some numbers below may not match database exactly — "
                    "verify by asking a specific date-range question.*\n\n"
                    f"🔍 **Diagnostic Analysis:**\n\n{text}"
                )
        except Exception as e:
            logger.error("Fallback synthesis failed: %s", e)

        return self._emergency_format(tool_log)

    def _emergency_format(self, tool_log: List[str]) -> str:
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