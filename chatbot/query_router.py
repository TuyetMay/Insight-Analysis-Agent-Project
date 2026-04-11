from __future__ import annotations
import re
from typing import Literal

QueryMode = Literal["structured", "agent"]

# Core diagnostic patterns
_AGENT_PATTERNS = re.compile(
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

_DIVERGENCE_PATTERN = re.compile(
    r"\b(increase|up|rise|grew|higher).{1,40}(decrease|down|fall|drop|lower|decline)"
    r"|\b(decrease|down|fall|drop|lower|decline).{1,40}(increase|up|rise|grew|higher)\b",
    re.IGNORECASE,
)

_STRUCTURED_OVERRIDE = re.compile(
    r"\b("
    r"total|sum|count|average|trend|top|rank|compare|breakdown"
    r"|show\s+me|give\s+me|what\s+is\s+the\s+total"
    r"|how\s+much|how\s+many|list|by\s+region|by\s+segment"
    r")\b",
    re.IGNORECASE,
)


class QueryRouter:
    """Stateless — call route() per query."""

    def route(self, query: str) -> QueryMode:
        if not query or not query.strip():
            return "structured"

        has_agent      = bool(_AGENT_PATTERNS.search(query))
        has_divergence = bool(_DIVERGENCE_PATTERN.search(query))
        has_structured = bool(_STRUCTURED_OVERRIDE.search(query))

        if has_divergence:
            return "agent"

        # Agent pattern without structured override → agent
        if has_agent and not has_structured:
            return "agent"

        # Agent pattern WITH structured override: agent wins if "why" is present
       
        if has_agent and has_structured:
            if re.search(r"\bwhy\b", query, re.IGNORECASE):
                return "agent"

        return "structured"