"""
chatbot/answer_validator.py
────────────────────────────────────────────────────────────────────────────────
Three-layer answer validation for the structured query pipeline.

Layer 1 — Rule-based result_df checks (no LLM, no cost):
  - Empty result          → caller retries with wider date or relaxed filters
  - All-zero metrics      → data likely outside filter window, caller retries
  - Invalid profit margin → physically impossible value (>500%), add disclaimer

Layer 2 — Grounding score (no LLM, no cost):
  - Extracts $ amounts and % values from the formatted answer text
  - Extracts the same from result_df numeric columns
  - Score = fraction of answer numbers traceable back to the DataFrame
  - Score < GROUNDING_THRESHOLD → caller prepends a ⚠️ warning to the answer

Layer 3 — Hybrid question-answer consistency (no LLM first, LLM if uncertain):
  Step A  Rule check (free):
    - FAIL_CERTAIN  → retry immediately (escalate tier), no LLM call needed
    - FAIL_UNCERTAIN→ hand off to LLM judge
    - PASS          → return answer as-is
  Step B  LLM judge (1 API call, only on FAIL_UNCERTAIN):
    - PASS  → return answer as-is
    - FAIL  → escalate tier

All three layers are stateless — instantiate once, call per query.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

GROUNDING_THRESHOLD = 0.40   # answers with score < this get a warning tag
_MIN_METRIC_VALUE   = 100.0  # ignore numbers below this (rank indices, years…)
_MAX_MARGIN_ABS     = 500.0  # profit margins beyond ±500 % are data errors

# Metric columns we trust for grounding (others may be auxiliary labels)
_METRIC_COLS = frozenset({
    "sales", "profit", "orders",
    "current", "previous",
    "profit_margin", "change_pct", "avg_discount_pct",
})

# Dollar pattern  — handles \$1,234,567  or  $1,234,567  (Streamlit markdown)
_DOLLAR_RE  = re.compile(r'\\?\$([\d,]+(?:\.\d+)?)')
# Percentage pattern  — handles 12.3%  or  +12.3%  or  -5.0%
_PERCENT_RE = re.compile(r'([+-]?\d+\.?\d*)%')


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    is_valid: bool
    reason: str     # "ok" | "empty" | "all_zero" | "invalid_margin"


# ── Validator ─────────────────────────────────────────────────────────────────

class AnswerValidator:
    """
    Layer 1: validate result_df before formatting.
    Layer 2: compute grounding score after formatting.
    """

    # ── Layer 1 ───────────────────────────────────────────────────────────────

    def validate_result_df(
        self,
        plan: Dict[str, Any],
        result_df: Optional[pd.DataFrame],
    ) -> ValidationResult:
        """
        Check whether result_df is usable.

        Returns ValidationResult(is_valid=False, reason=...) on failure so
        the caller can decide the recovery action:
          "empty"          → widen date range (already handled by _repair_plan)
          "all_zero"       → widen dimension filters
          "invalid_margin" → add a disclaimer, but still show the answer
        """
        # Check 1 — empty DataFrame
        if result_df is None or result_df.empty:
            return ValidationResult(False, "empty")

        intent = plan.get("intent", "")

        # Check 2 — all metric values are zero (suspicious for metric queries)
        if intent in ("kpi_value", "kpi_rank", "kpi_trend", "kpi_compare"):
            metric_cols = [
                c for c in result_df.columns
                if c in ("sales", "profit", "orders", "current", "previous")
            ]
            if metric_cols:
                total_abs = result_df[metric_cols].abs().sum().sum()
                if total_abs < 1.0:
                    return ValidationResult(False, "all_zero")

        # Check 3 — profit margin outside physically possible range
        if "profit_margin" in result_df.columns:
            margins = result_df["profit_margin"].dropna()
            if len(margins) > 0 and (margins.abs() > _MAX_MARGIN_ABS).any():
                return ValidationResult(False, "invalid_margin")

        return ValidationResult(True, "ok")

    # ── Layer 2 ───────────────────────────────────────────────────────────────

    def grounding_score(
        self,
        answer_text: str,
        result_df: Optional[pd.DataFrame],
    ) -> float:
        """
        Fraction of numeric claims in answer_text that can be traced to result_df.

        Returns 1.0 (pass) when:
          - answer_text contains no numbers  →  nothing to verify
          - result_df is empty or None       →  no data to compare against
        """
        if not answer_text or result_df is None or result_df.empty:
            return 1.0

        ans_nums = self._nums_from_text(answer_text)
        if not ans_nums:
            return 1.0

        df_nums = self._nums_from_df(result_df)
        if not df_nums:
            return 1.0

        matched = sum(
            1 for a in ans_nums
            if any(self._close(a, d) for d in df_nums)
        )
        return matched / len(ans_nums)

    def warning_tag(self, score: float) -> str:
        """Return a ⚠️ prefix string when grounding is low, else empty string."""
        if score < GROUNDING_THRESHOLD:
            return (
                "⚠️ *Some figures in this answer may not exactly match the "
                "underlying data — please verify key numbers against the "
                "dashboard.*\n\n"
            )
        return ""

    def disclaimer_tag(self) -> str:
        """Return disclaimer for invalid_margin results."""
        return (
            "\n\n> ⚠️ *Note: one or more profit margin values appear "
            "outside a realistic range. The data may contain division "
            "anomalies for very small sales amounts.*"
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _nums_from_text(text: str) -> Set[float]:
        """Extract significant dollar amounts and percentages from answer text."""
        nums: Set[float] = set()

        for m in _DOLLAR_RE.finditer(text):
            raw = m.group(1).replace(",", "")
            try:
                v = round(float(raw), 0)
                if v >= _MIN_METRIC_VALUE:
                    nums.add(v)
            except ValueError:
                pass

        for m in _PERCENT_RE.finditer(text):
            try:
                v = round(abs(float(m.group(1))), 1)
                if 0.1 <= v <= 200.0:
                    nums.add(v)
            except ValueError:
                pass

        return nums

    @staticmethod
    def _nums_from_df(df: pd.DataFrame) -> Set[float]:
        """Extract numeric values from known metric columns of result_df."""
        nums: Set[float] = set()
        cols = [c for c in df.columns if c in _METRIC_COLS]

        for col in cols:
            for raw in df[col].dropna():
                try:
                    v = float(raw)
                    if col in ("profit_margin", "change_pct", "avg_discount_pct"):
                        pct = round(abs(v), 1)
                        if 0.1 <= pct <= 200.0:
                            nums.add(pct)
                    elif abs(v) >= _MIN_METRIC_VALUE:
                        nums.add(round(v, 0))
                except (TypeError, ValueError):
                    pass

        return nums

    @staticmethod
    def _close(a: float, b: float, rtol: float = 0.01, atol: float = 1.0) -> bool:
        """True when a and b differ by at most rtol relatively or atol absolutely."""
        if abs(b) < atol:
            return abs(a - b) <= atol
        return abs(a - b) / abs(b) <= rtol


# ── Layer 3 ───────────────────────────────────────────────────────────────────

# Dimension values present in Superstore — used by breakdown-presence check
_REGION_VALUES   = {"West", "East", "Central", "South"}
_SEGMENT_VALUES  = {"Consumer", "Corporate", "Home Office"}
_CATEGORY_VALUES = {"Furniture", "Technology", "Office Supplies"}
_ALL_DIM_VALUES  = _REGION_VALUES | _SEGMENT_VALUES | _CATEGORY_VALUES

# Regex helpers for rule checks
_ERROR_PREFIX_RE    = re.compile(r'^(❌|⚠️ Could not|Error:)', re.IGNORECASE)
_NO_DATA_RE         = re.compile(r'no data found|no loss-making|could not retrieve', re.IGNORECASE)
_COMPARISON_Q_RE    = re.compile(r'\b(compare|vs\.?|versus|2014.*2015|2015.*2016|2016.*2017|year.over.year|yoy)\b', re.IGNORECASE)
_COMPARISON_A_RE    = re.compile(r'\b(current|previous|change|vs\.?|compared|grew|declined|increased|decreased)\b', re.IGNORECASE)
_WHY_Q_RE           = re.compile(r'^why\b|why did|why (is|are|does|do|was|were)\b', re.IGNORECASE)
_CAUSAL_A_RE        = re.compile(r'\b(because|due to|caused by|driven by|result of|root cause|contributing|factor|reason)\b', re.IGNORECASE)
_BREAKDOWN_Q_RE     = re.compile(r'\b(by region|by segment|by category|breakdown|split by|per region|per segment)\b', re.IGNORECASE)
_TOPN_Q_RE          = re.compile(r'\btop\s*(\d+)\b', re.IGNORECASE)
_NUMBER_A_RE        = re.compile(r'\\?\$[\d,]+|\d+%|\d{4,}')
_METRIC_Q_RE        = re.compile(r'\b(total|how much|how many|revenue|profit|sales|orders|margin)\b', re.IGNORECASE)

# Answer patterns that signal "already validated" — skip Layer 3
_AGENT_ANSWER_RE    = re.compile(r'🔍 \*\*Diagnostic Analysis', re.IGNORECASE)


class Layer3Validator:
    """
    Hybrid question-answer consistency validator (Layer 3).

    validate() flow:
      1. Rule check  (free, instant)
         PASS          → return immediately
         FAIL_CERTAIN  → return immediately (caller escalates tier)
         FAIL_UNCERTAIN→ proceed to LLM judge
      2. LLM judge   (1 Gemini call, only on FAIL_UNCERTAIN)
         PASS / FAIL   → return result

    Verdicts:
      PASS          — answer is good, send to user
      FAIL_CERTAIN  — definitely wrong, retry without LLM cost
      FAIL_UNCERTAIN— might be wrong, LLM used to confirm
    """

    PASS          = "pass"
    FAIL_CERTAIN  = "fail_certain"
    FAIL_UNCERTAIN = "fail_uncertain"

    # ── Public ────────────────────────────────────────────────────────────────

    def validate(
        self,
        question: str,
        answer: str,
        plan: Dict[str, Any],
        gemini_client: Any = None,
        model_name: str = "",
    ) -> Tuple[str, str]:
        """
        Returns (verdict, reason).
        verdict ∈ {PASS, FAIL_CERTAIN, FAIL_UNCERTAIN}
        reason  is an empty string on PASS.
        """
        # Agent answers have their own internal validation — skip
        if _AGENT_ANSWER_RE.search(answer):
            return self.PASS, ""

        # Step A: Rule check
        verdict, reason = self._rule_check(question, answer, plan)
        logger.debug("Layer3 rule_check: %s — %s", verdict, reason)

        if verdict != self.FAIL_UNCERTAIN:
            return verdict, reason

        # Step B: LLM judge (only when rule is uncertain)
        if gemini_client and model_name:
            verdict, reason = self._llm_judge(question, answer, gemini_client, model_name)
            logger.debug("Layer3 llm_judge: %s — %s", verdict, reason)
            return verdict, reason

        # No LLM available → pass through to avoid blocking
        return self.PASS, ""

    # ── Step A: Rule check ────────────────────────────────────────────────────

    def _rule_check(
        self,
        question: str,
        answer: str,
        plan: Dict[str, Any],
    ) -> Tuple[str, str]:

        # ── Certain fails (retry immediately, no LLM needed) ─────────────────

        # 1. System error / exception in answer
        if _ERROR_PREFIX_RE.search(answer):
            return self.FAIL_CERTAIN, "system error in answer"

        # 2. No-data response for a question that should have data
        if _NO_DATA_RE.search(answer) and not self._is_detail_intent(plan):
            return self.FAIL_CERTAIN, "answer reports no data"

        # 3. Metric question but answer contains zero numbers
        if _METRIC_Q_RE.search(question) and not _NUMBER_A_RE.search(answer):
            return self.FAIL_CERTAIN, "metric question but answer has no numbers"

        # 4. Breakdown question but answer lacks any known dimension value
        if _BREAKDOWN_Q_RE.search(question):
            has_dim = any(dim in answer for dim in _ALL_DIM_VALUES)
            # Also accept if answer has numbered list (could be sub-categories)
            has_list = bool(re.search(r'^\s*\d+\.', answer, re.MULTILINE))
            if not has_dim and not has_list:
                return self.FAIL_CERTAIN, "breakdown question but no dimension values in answer"

        # ── Uncertain (rule cannot decide — need LLM judge) ───────────────────

        # 5. Comparison question but answer seems to miss both-period structure
        if _COMPARISON_Q_RE.search(question) and not _COMPARISON_A_RE.search(answer):
            return self.FAIL_UNCERTAIN, "comparison question may lack both-period comparison"

        # 6. "Why" question but answer has no causal language
        if _WHY_Q_RE.search(question) and not _CAUSAL_A_RE.search(answer):
            return self.FAIL_UNCERTAIN, "why-question may lack root-cause explanation"

        # 7. "Top N" request but answer might be truncated
        m = _TOPN_Q_RE.search(question)
        if m:
            requested_n = int(m.group(1))
            # Count numbered list items in answer
            list_items = re.findall(r'^\s*\d+\.', answer, re.MULTILINE)
            if len(list_items) < min(requested_n, 3):  # allow slack for small N
                return self.FAIL_UNCERTAIN, f"top-{requested_n} request but answer has {len(list_items)} items"

        return self.PASS, ""

    # ── Step B: LLM judge ─────────────────────────────────────────────────────

    def _llm_judge(
        self,
        question: str,
        answer: str,
        gemini_client: Any,
        model_name: str,
    ) -> Tuple[str, str]:
        try:
            from google.genai import types as genai_types

            prompt = f"""You are a strict QA checker for a BI chatbot.

Question: "{question}"
Answer (first 500 chars): "{answer[:500]}"

Does this answer CORRECTLY and COMPLETELY address the question?

Rules:
- A comparison question (vs / compare / year-over-year) MUST show numbers for BOTH periods.
- A "why" / "what caused" question MUST include a root-cause explanation (not just numbers).
- A breakdown question (by region / by segment) MUST list values for that dimension.
- A KPI / total question MUST include at least one specific dollar or percentage figure.
- A "top N" question MUST list at least min(N, 3) items.

Reply with ONE line only — no explanation, no preamble:
PASS
or
FAIL: <one-line reason>"""

            resp = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=60,
                ),
            )
            raw = (getattr(resp, "text", "") or "").strip()
            return self._parse_llm_verdict(raw)

        except Exception as exc:
            logger.warning("Layer3 LLM judge failed — defaulting to PASS: %s", exc)
            return self.PASS, ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_verdict(raw: str) -> Tuple[str, str]:
        upper = raw.upper()
        if upper.startswith("PASS"):
            return Layer3Validator.PASS, ""
        if upper.startswith("FAIL"):
            reason = raw[4:].lstrip(": ").strip()
            return Layer3Validator.FAIL_CERTAIN, reason or "LLM judge: answer incomplete"
        # Ambiguous response → pass through (fail-safe)
        return Layer3Validator.PASS, ""

    @staticmethod
    def _is_detail_intent(plan: Dict[str, Any]) -> bool:
        return plan.get("intent") == "kpi_detail"
