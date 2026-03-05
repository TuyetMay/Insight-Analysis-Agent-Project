"""
chatbot/suggestions/models.py
Shared data model for suggestions across both rule-based and RAG engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Suggestion:
    """
    A follow-up question chip shown to the user.
    - text : what the user sees (< 60 chars)
    - plan : optional pre-validated plan JSON that can be re-run without calling the LLM
    """
    text: str
    plan: Optional[Dict[str, Any]] = None
