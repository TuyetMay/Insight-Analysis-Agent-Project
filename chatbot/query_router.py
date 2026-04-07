"""
chatbot/query_router.py
Phân loại query → "structured" hoặc "agent"

Structured: what/how → intent-based (giữ nguyên)
Agent:      why/should/what-if → AI agent layer
"""

from __future__ import annotations
import re
from typing import Literal

QueryMode = Literal["structured", "agent"]

# Patterns chỉ ra "why/diagnostic" queries
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
    r"|\w+\s+(decrease|drop|fall|decline)\s+(from|in|during)" 
    r")\b",

    re.IGNORECASE,
)

# Patterns chỉ ra "structured" queries — override agent nếu match
_STRUCTURED_OVERRIDE = re.compile(
    r"\b("
    r"total|sum|count|average|trend|top|rank|compare|breakdown"
    r"|show\s+me|give\s+me|what\s+is\s+the\s+total"
    r"|how\s+much|how\s+many|list|by\s+region|by\s+segment"
    r")\b",
    re.IGNORECASE,
)


class QueryRouter:
    """Stateless — gọi route() mỗi query."""

    def route(self, query: str) -> QueryMode:
        """
        Returns "agent" hoặc "structured".

        Logic:
          1. Nếu có agent pattern → agent
          2. Nếu có structured override → structured
          3. Default → structured (safe fallback)
        """
        if not query or not query.strip():
            return "structured"

        has_agent      = bool(_AGENT_PATTERNS.search(query))
        has_structured = bool(_STRUCTURED_OVERRIDE.search(query))

        if has_agent and not has_structured:
            return "agent"
        return "structured"