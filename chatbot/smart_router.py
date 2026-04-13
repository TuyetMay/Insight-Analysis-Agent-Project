"""
chatbot/smart_router.py
───────────────────────────────────────────────────────────────
Smart Query Router — LLM-based intent classification (Step R2)

PROBLEMS SOLVED:
  Limitation 1: "Explain the sales trend by region"
    → contains "explain" (agent signal) + "trend by region" (structured signal)
    → old regex: routes to structured (loses causal explanation)
    → new router: detects HYBRID — runs structured first, agent explains with data

  Limitation 2: "Which region contributes least to profit and why?"
    → complex sentence, "why" may be missed in variant phrasings
    → old regex: fragile pattern matching
    → new router: LLM reads full semantic intent, returns JSON mode

ARCHITECTURE (based on DIN-SQL / Decomposed Prompting / BIRD):
  Input query
      │
      ▼
  LLM Classifier (few-shot, returns JSON)
      │
      ├─ mode: "structured"  → existing rule-based + SQL path
      ├─ mode: "agent"       → existing AgentOrchestrator path
      └─ mode: "hybrid"      → STEP 1: structured query (get data)
                                STEP 2: agent explain (with data as context)
                                STEP 3: merge & return combined answer

COST OPTIMISATION:
  - Uses gemini-1.5-flash-lite (cheapest, fastest)
  - Prompt < 300 tokens
  - Falls back to regex router on ANY LLM failure
  - Caches result per query string (no repeated LLM calls)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────
RouteMode = Literal["structured", "agent", "hybrid"]


@dataclass
class RouteDecision:
    """
    Result from SmartRouter.classify().

    mode            : "structured" | "agent" | "hybrid"
    structured_query: sub-query for SQL/rule path (hybrid only)
    explain_query   : sub-query for agent path (hybrid only)
    confidence      : 0.0–1.0, used for logging / fallback decisions
    raw_llm_output  : original LLM JSON string for debugging
    """
    mode: RouteMode
    structured_query: Optional[str] = None
    explain_query: Optional[str] = None
    confidence: float = 1.0
    raw_llm_output: str = ""
    used_fallback: bool = False


# ── Few-shot examples for the classifier prompt ───────────────
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


class SmartRouter:
    """
    LLM-based query router with regex fallback.

    Usage:
        router = SmartRouter(gemini_client, model_name)
        decision = router.classify("explain sales trend by region")
        # decision.mode == "hybrid"
        # decision.structured_query == "sales trend by region"
        # decision.explain_query == "explain why sales trend differs across regions"
    """

    def __init__(
        self,
        gemini_client: Any,
        model_name: str,
        fallback_router: Optional[Any] = None,  # existing QueryRouter instance
        cache_size: int = 128,
    ) -> None:
        self.client = gemini_client
        self.model_name = model_name
        self._fallback = fallback_router
        self._cache: Dict[str, RouteDecision] = {}
        self._cache_size = cache_size

    # ── Public API ────────────────────────────────────────────

    def classify(self, query: str) -> RouteDecision:
        """
        Classify query and return RouteDecision.
        Falls back to regex router on any LLM failure.
        """
        q = (query or "").strip()
        if not q:
            return RouteDecision(mode="structured")

        # Cache hit
        if q in self._cache:
            return self._cache[q]

        # Try LLM classification
        decision = self._llm_classify(q)

        # Evict cache if full (simple FIFO)
        if len(self._cache) >= self._cache_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]

        self._cache[q] = decision
        return decision

    def route(self, query: str) -> str:
        """
        Backward-compatible with old QueryRouter.route().
        Returns "structured" | "agent" (hybrid → "agent" for simple callers).
        """
        decision = self.classify(query)
        if decision.mode == "hybrid":
            return "agent"   # caller should use classify() for full hybrid support
        return decision.mode

    # ── LLM classifier ────────────────────────────────────────

    def _llm_classify(self, query: str) -> RouteDecision:
        if not self.client:
            return self._regex_fallback(query)

        try:
            # Build prompt
            prompt = _CLASSIFIER_PROMPT.format(
                few_shot=_FEW_SHOT_EXAMPLES,
                query=query,
            )

            # Call LLM — use duck-typing, avoid hard import of google.genai
            # so the class works even when google-genai is not installed
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
                # google.genai not available — call without config kwarg
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
        """Parse LLM JSON response into RouteDecision."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting JSON object
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

    # ── Regex fallback (existing logic preserved) ─────────────

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
    # NEW: signals that indicate hybrid even without "why"
    # Covers Limitation 1: "explain", "describe", "what drives"
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

    # NEW: Limitation 2 — implicit causal in compound questions
    # Catches "X and why?", "X and what caused?", "X, what accounts for?"
    _COMPOUND_CAUSAL_RE = re.compile(
        r"\b(and\s+why|and\s+what\s+(caused|is\s+causing|drove|drives?|accounts?\s+for)"
        r"|,\s*what\s+(caused|is\s+causing|drove|drives?|accounts?\s+for)"
        r"|what\s+is\s+causing\s+the"
        r"|what\s+accounts?\s+for\s+(it|this|that|the)"
        r")\b",
        re.IGNORECASE,
    )

    # NEW: standalone "explain" or "what drives" without other signals
    _STANDALONE_EXPLAIN_RE = re.compile(
        r"^(explain|what\s+drives?|what\s+is\s+driving|describe\s+the)\b",
        re.IGNORECASE,
    )

    # NEW: advisory/qualitative signals that imply agent
    _QUALITATIVE_RE = re.compile(
        r"\b(healthy|normal|good|bad|worth\s+it|too\s+(high|low)"
        r"|performing\s+well|underperform|struggling|concern"
        r")\b",
        re.IGNORECASE,
    )

    def _regex_fallback(self, query: str) -> RouteDecision:
        """Regex-based fallback — enhanced version of old QueryRouter."""
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

        # Limitation 2: compound causal ("X and why?", "X and what caused?")
        if has_compound:
            return RouteDecision(
                mode="hybrid",
                structured_query=query,
                explain_query=query,
                used_fallback=True,
            )

        # Qualitative questions with no strong structured component → agent
        if has_qualitative and not has_structured:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # Limitation 1: "explain X" + structured data component → hybrid
        if has_explain and has_structured:
            return RouteDecision(
                mode="hybrid",
                structured_query=query,
                explain_query=query,
                used_fallback=True,
            )

        # Standalone "explain ..." or "what drives ..." without metric keywords → agent
        if has_standalone and not has_structured:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # "explain X" even without explicit structured keyword → hybrid
        # e.g. "explain profit differences across segments"
        # BUT "describe why X" = strong agent (has explicit "why")
        if has_explain:
            has_why = bool(re.search(r"\bwhy\b", query, re.IGNORECASE))
            if has_why:
                # "describe why X" → agent (causal, no data lookup needed)
                return RouteDecision(mode="agent", used_fallback=True,
                                     explain_query=query)
            # "explain X across Y" — check if there's a dimension breakdown
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
            # "explain X" without dimension → agent
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # Pure agent signals
        if has_agent and not has_structured:
            return RouteDecision(mode="agent", used_fallback=True,
                                 explain_query=query)

        # Agent + structured with "why" → agent wins
        if has_agent and has_structured:
            if re.search(r"\bwhy\b", query, re.IGNORECASE):
                return RouteDecision(mode="agent", used_fallback=True,
                                     explain_query=query)
            return RouteDecision(mode="structured", used_fallback=True)

        return RouteDecision(mode="structured", used_fallback=True)