"""
chatbot/agent/suggestions.py  — NEW FILE
Context-aware suggested follow-ups for diagnostic (agent) responses.

WHY:
  After "why did sales increase but profit decrease from Oct to Nov 2016?",
  the old suggestions were generic:
    - Sales by Region
    - Sales by Segment
    - Sales by Category
    - Sales by Sub-Category
  
  These are useless — user already knows sales increased. They need to understand
  the PROFIT SIDE: discount impact, loss-making products, margin by category.

  NEW suggestions are context-aware based on:
    1. The diagnostic finding (what was the root cause)
    2. The metrics involved (sales↑ profit↓ = margin/discount issue)
    3. The time period (specific date range drill-down)
"""

from __future__ import annotations

import re
from typing import List, Dict, Any, Optional


# ── Patterns to detect diagnostic context from question ────────────────────────

_SALES_UP_PROFIT_DOWN = re.compile(
    r"(sales|revenue).{1,50}(increase|up|rose|grew).{1,80}(profit).{1,50}(decrease|down|fell|drop)",
    re.IGNORECASE | re.DOTALL,
)

_PROFIT_ISSUE = re.compile(
    r"\b(profit|margin|loss|losing|unprofitable|bleeding)\b",
    re.IGNORECASE,
)

_UNDERPERFORM = re.compile(
    r"\b(underperform|behind|lag|weak|poor|low|worst)\b",
    re.IGNORECASE,
)

_DATE_SPECIFIC = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
    r"|\d{4}[-/]\d{1,2})\b",
    re.IGNORECASE,
)


def get_diagnostic_suggestions(
    question: str,
    agent_response: str,
    plan_defaults: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate context-aware follow-up suggestions after a diagnostic (agent) response.
    
    Returns list of {text, plan} dicts compatible with the existing suggestion format.
    """
    d = plan_defaults or {}
    start = d.get("start_date", "2016-01-01")
    end   = d.get("end_date",   "2017-12-31")
    filters = d.get("filters", {"region": [], "segment": [], "category": []})

    suggestions = []

    # ── Case 1: Sales up, profit down → discount/margin investigation ──────────
    if _SALES_UP_PROFIT_DOWN.search(question):
        suggestions = [
            {
                "text": "Which products had heaviest discounts in this period?",
                "plan": {
                    "intent": "kpi_rank", "metrics": ["profit"], "time_grain": "none",
                    "breakdown_by": "sub_category", "top_k": 10, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Profit margin by category — where did margin compress?",
                "plan": {
                    "intent": "kpi_value", "metrics": ["profit_margin"], "time_grain": "none",
                    "breakdown_by": "category", "top_k": None, "order_by": "profit_margin",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Which sub-categories are loss-making in this period?",
                "plan": {
                    "intent": "kpi_detail", "metrics": ["sales", "profit"],
                    "condition": "profit_negative", "breakdown_by": "sub_category",
                    "time_grain": "none", "top_k": 15, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Profit by region — which region dragged down profitability?",
                "plan": {
                    "intent": "kpi_value", "metrics": ["profit"], "time_grain": "none",
                    "breakdown_by": "region", "top_k": None, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
        ]

    # ── Case 2: General profit/loss investigation ───────────────────────────────
    elif _PROFIT_ISSUE.search(question):
        suggestions = [
            {
                "text": "Loss-making sub-categories — full breakdown",
                "plan": {
                    "intent": "kpi_detail", "metrics": ["sales", "profit"],
                    "condition": "profit_negative", "breakdown_by": "sub_category",
                    "time_grain": "none", "top_k": 15, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Profit margin by category",
                "plan": {
                    "intent": "kpi_value", "metrics": ["profit_margin"], "time_grain": "none",
                    "breakdown_by": "category", "top_k": None, "order_by": "profit_margin",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Profit trend — year over year comparison",
                "plan": {
                    "intent": "kpi_compare", "metrics": ["profit"], "time_grain": "none",
                    "breakdown_by": None, "top_k": None, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": "yoy",
                    "filters": filters,
                }
            },
            {
                "text": "Top 10 sub-categories by profit",
                "plan": {
                    "intent": "kpi_rank", "metrics": ["profit"], "time_grain": "none",
                    "breakdown_by": "sub_category", "top_k": 10, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
        ]

    # ── Case 3: Region/segment underperformance ─────────────────────────────────
    elif _UNDERPERFORM.search(question):
        suggestions = [
            {
                "text": "Profit by region — full comparison",
                "plan": {
                    "intent": "kpi_value", "metrics": ["profit", "profit_margin"],
                    "time_grain": "none", "breakdown_by": "region",
                    "top_k": None, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Sales by segment — identify lagging segments",
                "plan": {
                    "intent": "kpi_value", "metrics": ["sales", "profit"],
                    "time_grain": "none", "breakdown_by": "segment",
                    "top_k": None, "order_by": "sales",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Loss-making sub-categories in this period",
                "plan": {
                    "intent": "kpi_detail", "metrics": ["sales", "profit"],
                    "condition": "profit_negative", "breakdown_by": "sub_category",
                    "time_grain": "none", "top_k": 15, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Sales YoY comparison — is this period worse than last year?",
                "plan": {
                    "intent": "kpi_compare", "metrics": ["sales"], "time_grain": "none",
                    "breakdown_by": None, "top_k": None, "order_by": "sales",
                    "start_date": start, "end_date": end, "compare_period": "yoy",
                    "filters": filters,
                }
            },
        ]

    # ── Default fallback ────────────────────────────────────────────────────────
    else:
        suggestions = [
            {
                "text": "Loss-making sub-categories — which products are unprofitable?",
                "plan": {
                    "intent": "kpi_detail", "metrics": ["sales", "profit"],
                    "condition": "profit_negative", "breakdown_by": "sub_category",
                    "time_grain": "none", "top_k": 15, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Profit margin by category",
                "plan": {
                    "intent": "kpi_value", "metrics": ["profit_margin"], "time_grain": "none",
                    "breakdown_by": "category", "top_k": None, "order_by": "profit_margin",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Profit by region — best and worst regions",
                "plan": {
                    "intent": "kpi_value", "metrics": ["profit"], "time_grain": "none",
                    "breakdown_by": "region", "top_k": None, "order_by": "profit",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
            {
                "text": "Sales vs profit trend — monthly breakdown",
                "plan": {
                    "intent": "kpi_trend", "metrics": ["sales", "profit"],
                    "time_grain": "month", "breakdown_by": None,
                    "top_k": None, "order_by": "sales",
                    "start_date": start, "end_date": end, "compare_period": None,
                    "filters": filters,
                }
            },
        ]

    return suggestions[:4]