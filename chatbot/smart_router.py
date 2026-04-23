"""
chatbot/smart_router.py
───────────────────────────────────────────────────────────────

FIXES vs original:
  FIX-SR-1: "Why did/does/is/are X" queries bây giờ luôn route "agent"
             TRƯỚC khi gọi LLM. LLM trước đây phân loại sai chúng là
             "structured" vì thấy metric + date. Thêm pre-LLM shortcut.
  
  FIX-SR-2: "What caused/drove X" queries cũng route agent ngay.
  
  FIX-SR-3: Regex fallback được cải thiện — khi has_agent=True và không
             có has_structured, luôn route agent (không check thêm điều kiện).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

RouteMode = Literal["structured", "agent", "hybrid"]


@dataclass
class RouteDecision:
    mode: RouteMode
    structured_query: Optional[str] = None
    explain_query: Optional[str] = None
    confidence: float = 1.0
    raw_llm_output: str = ""
    used_fallback: bool = False


_FEW_SHOT_EXAMPLES = """
EXAMPLES:

Q: "total sales by region"
→ {"mode":"structured","structured_query":"total sales by region","explain_query":null}

Q: "profit trend over years"
→ {"mode":"structured","structured_query":"profit trend over years","explain_query":null}

Q: "top 5 sub-categories by profit"
→ {"mode":"structured","structured_query":"top 5 sub-categories by profit","explain_query":null}

Q: "why did profit drop in 2016?"
→ {"mode":"agent","structured_query":null,"explain_query":"why did profit drop in 2016?"}

Q: "what caused the revenue decline last quarter?"
→ {"mode":"agent","structured_query":null,"explain_query":"what caused the revenue decline last quarter?"}

Q: "sales increased but profit decreased — why?"
→ {"mode":"agent","structured_query":null,"explain_query":"sales increased but profit decreased — why?"}

Q: "why does heavy discounting hurt profitability?"
→ {"mode":"agent","structured_query":null,"explain_query":"why does heavy discounting hurt profitability?"}

Q: "why is Furniture underperforming?"
→ {"mode":"agent","structured_query":null,"explain_query":"why is Furniture underperforming?"}

Q: "explain the sales trend by region"
→ {"mode":"hybrid","structured_query":"sales trend by region","explain_query":"explain why sales trend differs across regions"}

Q: "which region contributes least to profit and why?"
→ {"mode":"hybrid","structured_query":"profit by region","explain_query":"why does the lowest profit region underperform?"}

Q: "show me loss-making products and what is causing the losses"
→ {"mode":"hybrid","structured_query":"loss-making sub-categories with profit and discount","explain_query":"what is causing these sub-categories to lose money?"}

Q: "compare 2016 vs 2017 sales and explain the difference"
→ {"mode":"hybrid","structured_query":"compare 2016 vs 2017 sales","explain_query":"explain why sales changed between 2016 and 2017"}

Q: "which segment has the best margin and why?"
→ {"mode":"hybrid","structured_query":"profit margin by segment","explain_query":"why does that segment have a better margin than others?"}

Q: "is the profit margin healthy?"
→ {"mode":"agent","structured_query":null,"explain_query":"is the profit margin healthy compared to benchmarks?"}

Q: "what should we focus on to improve profit?"
→ {"mode":"agent","structured_query":null,"explain_query":"what should we focus on to improve profit?"}
"""

_CLASSIFIER_PROMPT = """You are a query intent classifier for a business analytics dashboard.

Classify the user's question into one of three modes:
- "structured": pure data retrieval — aggregates, trends, rankings, comparisons with no explanation needed
- "agent": causal analysis, root cause, recommendations, "why", "explain", "what caused", "should we"
- "hybrid": the question has BOTH a data part AND an explanation part — needs structured data first, then agent explanation

For "hybrid" mode, decompose the question into:
- structured_query: the data retrieval sub-question (SQL-answerable)
- explain_query: the explanation sub-question (agent-answerable, should reference the data)

RULES:
1. Words like "explain", "describe", "tell me about", "what drives" = agent or hybrid signal
2. "why", "what caused", "reason", "root cause" = strong agent signal
3. If the question has BOTH a quantitative lookup AND a causal/qualitative request → hybrid
4. Simple "show me", "what is", "how much", "top N", "trend", "compare" → structured
5. Ambiguous questions lean toward "agent" (safer — agent can also show data)
6. Return ONLY valid JSON. No markdown, no explanation.

{few_shot}

USER QUESTION: "{query}"

Return JSON only:"""


# ── Pre-LLM shortcut patterns (FIX-SR-1, FIX-SR-2) ──────────────────────────
# Các pattern này LUÔN route agent mà không cần gọi LLM.
# Ngăn LLM phân loại sai "Why did profit drop" → structured.

_WHY_VERB_RE = re.compile(
    r'^why\s+(did|does|do|is|are|was|were|has|have|had|would|should|could|can)\b',
    re.IGNORECASE,
)

_CAUSAL_START_RE = re.compile(
    r'^(what\s+(caused?|drove|is\s+causing|triggered|led\s+to|drove)|'
    r'how\s+come\b|'
    r'reason\s+(for|why)\b)',
    re.IGNORECASE,
)


class SmartRouter:
    """
    LLM-based query router with regex fallback.

    FIX-SR-1: Thêm pre-LLM shortcut cho "why did/does/is" và "what caused" queries.
    Các query này LUÔN là agent — không cần tốn LLM call để confirm.
    """

    def __init__(
        self,
        gemini_client: Any,
        model_name: str,
        fallback_router: Optional[Any] = None,
        cache_size: int = 128,
    ) -> None:
        self.client = gemini_client
        self.model_name = model_name
        self._fallback = fallback_router
        self._cache: Dict[str, RouteDecision] = {}
        self._cache_size = cache_size

    def classify(self, query: str) -> RouteDecision:
        q = (query or "").strip()
        if not q:
            return RouteDecision(mode="structured")

        # Cache hit
        if q in self._cache:
            return self._cache[q]

        # ── FIX-SR-1: Pre-LLM shortcut cho causal queries ─────────────────
        # "Why did/does/is/are X" và "What caused X" LUÔN là agent.
        # LLM trước đây phân loại sai khi thấy metric+date → "structured".
        if _WHY_VERB_RE.search(q) or _CAUSAL_START_RE.search(q):
            decision = RouteDecision(
                mode="agent",
                explain_query=q,
                confidence=0.97,
                used_fallback=False,
            )
            self._store_cache(q, decision)
            return decision

        # Try LLM classification
        decision = self._llm_classify(q)
        self._store_cache(q, decision)
        return decision

    def _store_cache(self, q: str, decision: RouteDecision) -> None:
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[q] = decision

    def route(self, query: str) -> str:
        decision = self.classify(query)
        if decision.mode == "hybrid":
            return "agent"
        return decision.mode

    def _llm_classify(self, query: str) -> RouteDecision:
        if not self.client:
            return self._regex_fallback(query)

        try:
            prompt = _CLASSIFIER_PROMPT.format(
                few_shot=_FEW_SHOT_EXAMPLES,
                query=query,
            )
            try:
                from google.genai import types as genai_types
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=120,
                    ),
                )
            except ImportError:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

            raw = (getattr(resp, "text", "") or "").strip()
            return self._parse_llm_response(raw, query)

        except Exception as exc:
            logger.warning("SmartRouter LLM call failed (%s) — using regex fallback", exc)
            return self._regex_fallback(query)

    def _parse_llm_response(self, raw: str, original_query: str) -> RouteDecision:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except Exception:
                    logger.warning("SmartRouter: could not parse LLM output: %r", raw[:100])
                    return self._regex_fallback(original_query)
            else:
                logger.warning("SmartRouter: no JSON found in LLM output: %r", raw[:100])
                return self._regex_fallback(original_query)

        mode = obj.get("mode", "structured")
        if mode not in ("structured", "agent", "hybrid"):
            logger.warning("SmartRouter: invalid mode %r — falling back", mode)
            return self._regex_fallback(original_query)

        return RouteDecision(
            mode=mode,
            structured_query=obj.get("structured_query") or None,
            explain_query=obj.get("explain_query") or None,
            confidence=0.9,
            raw_llm_output=raw,
            used_fallback=False,
        )

    # ── Regex fallback ─────────────────────────────────────────────────────────

    _AGENT_RE = re.compile(
        r"\b("
        r"why\s+is|why\s+are|why\s+did|why\s+does|why\s+do"
        r"|what\s+caused|what\s+is\s+causing|what\s+drove"
        r"|how\s+come|reason\s+for|explain\s+why"
        r"|should\s+we|should\s+i|what\s+should"
        r"|what\s+if|what\s+would\s+happen|if\s+we"
        r"|is\s+it\s+worth|is\s+this\s+normal"
        r"|underperform|root\s+cause|diagnosis|diagnose"
        r"|why\s+\w+"
        r")\b",
        re.IGNORECASE,
    )
    _DIVERGENCE_RE = re.compile(
        r"\b(increase|up|rise|grew|higher).{1,40}(decrease|down|fall|drop|lower|decline)"
        r"|\b(decrease|down|fall|drop|lower|decline).{1,40}(increase|up|rise|grew|higher)\b",
        re.IGNORECASE,
    )
    _STRUCTURED_RE = re.compile(
        r"\b("
        r"total|sum|count|average|trend|top|rank|compare|breakdown"
        r"|show\s+me|give\s+me|what\s+is\s+the\s+total"
        r"|how\s+much|how\s+many|list|by\s+region|by\s+segment"
        r")\b",
        re.IGNORECASE,
    )
    _EXPLAIN_RE = re.compile(
        r"\b("
        r"explain|describe\s+why|describe\s+the"
        r"|what\s+drives?|what\s+is\s+driving"
        r"|what\s+accounts\s+for|break\s+down\s+why"
        r"|and\s+why|and\s+what\s+caused|what\s+reason"
        r"|tell\s+me\s+why|tell\s+me\s+what"
        r")\b",
        re.IGNORECASE,
    )
    _COMPOUND_CAUSAL_RE = re.compile(
        r"\b(and\s+why|and\s+what\s+(caused|is\s+causing|drove|drives?|accounts?\s+for)"
        r"|,\s*what\s+(caused|is\s+causing|drove|drives?|accounts?\s+for)"
        r"|what\s+is\s+causing\s+the"
        r"|what\s+accounts?\s+for\s+(it|this|that|the)"
        r")\b",
        re.IGNORECASE,
    )
    _STANDALONE_EXPLAIN_RE = re.compile(
        r"^(explain|what\s+drives?|what\s+is\s+driving|describe\s+the)\b",
        re.IGNORECASE,
    )
    _QUALITATIVE_RE = re.compile(
        r"\b(healthy|normal|good|bad|worth\s+it|too\s+(high|low)"
        r"|performing\s+well|underperform|struggling|concern"
        r")\b",
        re.IGNORECASE,
    )

    def _regex_fallback(self, query: str) -> RouteDecision:
        has_agent      = bool(self._AGENT_RE.search(query))
        has_divergence = bool(self._DIVERGENCE_RE.search(query))
        has_structured = bool(self._STRUCTURED_RE.search(query))
        has_explain    = bool(self._EXPLAIN_RE.search(query))
        has_compound   = bool(self._COMPOUND_CAUSAL_RE.search(query))
        has_standalone = bool(self._STANDALONE_EXPLAIN_RE.search(query))
        has_qualitative = bool(self._QUALITATIVE_RE.search(query))

        # Divergence pattern → always agent
        if has_divergence:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # Compound causal → hybrid
        if has_compound:
            return RouteDecision(
                mode="hybrid",
                structured_query=query,
                explain_query=query,
                used_fallback=True,
            )

        # Qualitative without structured → agent
        if has_qualitative and not has_structured:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # FIX-SR-3: Agent pattern (has_agent=True) WITHOUT structured → luôn agent
        # Original code chỉ check "why" khi has_structured cũng True, nhưng
        # "Why is Furniture underperforming?" không match _STRUCTURED_RE → phải agent
        if has_agent and not has_structured:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # explain + structured → hybrid
        if has_explain and has_structured:
            return RouteDecision(
                mode="hybrid",
                structured_query=query,
                explain_query=query,
                used_fallback=True,
            )

        # Standalone explain → agent or hybrid
        if has_standalone and not has_structured:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        if has_explain:
            has_why = bool(re.search(r"\bwhy\b", query, re.IGNORECASE))
            if has_why:
                return RouteDecision(mode="agent", used_fallback=True,
                                     explain_query=query)
            has_dimension = bool(re.search(
                r"\b(by|across|per|for each|between|among|within)\s+"
                r"(region|segment|category|sub.?category|area|zone|market)s?\b"
                r"|\b(region|segment|category|sub.?category)s?\s+"
                r"(differ|difference|variation|comparison|breakdown)\b",
                query, re.IGNORECASE
            ))
            if has_dimension:
                return RouteDecision(
                    mode="hybrid",
                    structured_query=query,
                    explain_query=query,
                    used_fallback=True,
                )
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # Agent + structured: nếu có "why" → agent wins
        if has_agent and has_structured:
            if re.search(r"\bwhy\b", query, re.IGNORECASE):
                return RouteDecision(mode="agent", used_fallback=True,
                                     explain_query=query)
            return RouteDecision(mode="structured", used_fallback=True)

        return RouteDecision(mode="structured", used_fallback=True)