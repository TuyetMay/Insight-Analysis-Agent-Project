"""
Tool wrappers cho AI Agent.
Mỗi tool = 1 SQL call hoặc 1 RAG lookup.
Agent tự quyết định gọi tool nào.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
from chatbot.sql_builder import SQLBuilder

sql = SQLBuilder()


TOOL_SCHEMAS = [
    {
        "name": "query_metric",
        "description": (
            "Get aggregated metric (sales/profit/orders/profit_margin) "
            "optionally broken down by a dimension (region/segment/category/sub_category). "
            "Use this to get benchmark data for comparison."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["sales", "profit", "orders", "profit_margin"],
                    "description": "The metric to query",
                },
                "breakdown_by": {
                    "type": "string",
                    "enum": ["region", "segment", "category", "sub_category"],
                    "description": "Optional dimension to group by",
                },
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date":   {"type": "string", "description": "YYYY-MM-DD"},
                "filters": {
                    "type": "object",
                    "description": "Optional filters e.g. {region: ['West']}",
                },
            },
            "required": ["metric"],
        },
    },
    {
        "name": "find_anomalies",
        "description": (
            "Find loss-making or underperforming items. "
            "Use this to diagnose WHY something is underperforming."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "breakdown_by": {
                    "type": "string",
                    "enum": ["sub_category", "region", "segment", "category"],
                },
                "start_date": {"type": "string"},
                "end_date":   {"type": "string"},
                "filters":    {"type": "object"},
            },
            "required": [],
        },
    },
    {
        "name": "get_trend",
        "description": (
            "Get time series data for a metric. "
            "Use this to check if something is getting better or worse over time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric":    {"type": "string", "enum": ["sales", "profit", "orders", "profit_margin"]},
                "time_grain": {"type": "string", "enum": ["year", "quarter", "month"]},
                "breakdown_by": {"type": "string"},
                "start_date": {"type": "string"},
                "end_date":   {"type": "string"},
                "filters":    {"type": "object"},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "compare_periods",
        "description": (
            "Compare a metric between two time periods. "
            "Use this to quantify how much something changed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "metric":         {"type": "string"},
                "current_start":  {"type": "string", "description": "YYYY-MM-DD"},
                "current_end":    {"type": "string", "description": "YYYY-MM-DD"},
                "previous_start": {"type": "string", "description": "YYYY-MM-DD"},
                "previous_end":   {"type": "string", "description": "YYYY-MM-DD"},
                "breakdown_by":   {"type": "string"},
                "filters":        {"type": "object"},
            },
            "required": ["metric", "current_start", "current_end"],
        },
    },
]


# ── Tool executors ────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    default_start: str,
    default_end: str,
) -> str:
    """
    Execute a tool call and return result as string for Gemini.
    Always returns string — never raises (returns error message instead).
    """
    try:
        if tool_name == "query_metric":
            return _query_metric(tool_args, default_start, default_end)
        elif tool_name == "find_anomalies":
            return _find_anomalies(tool_args, default_start, default_end)
        elif tool_name == "get_trend":
            return _get_trend(tool_args, default_start, default_end)
        elif tool_name == "compare_periods":
            return _compare_periods(tool_args, default_start, default_end)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool error ({tool_name}): {str(e)}"


def _fmt_metric(metric: str, val: float) -> str:
    if metric == "orders":
        return f"{val:,.0f} orders"
    if metric == "profit_margin":
        return f"{val:.1f}%"
    return f"${val:,.0f}"


def _build_filters(raw: Any) -> Dict[str, list]:
    if not isinstance(raw, dict):
        return {"region": [], "segment": [], "category": [], "sub_category": []}
    return {
        "region":       list(raw.get("region",       []) or []),
        "segment":      list(raw.get("segment",      []) or []),
        "category":     list(raw.get("category",     []) or []),
        "sub_category": list(raw.get("sub_category", []) or []),
    }


def _query_metric(args: Dict, s0: str, e0: str) -> str:
    plan = {
        "intent":              "kpi_value",
        "metrics":             [args["metric"]],
        "time_grain":          "none",
        "breakdown_by":        args.get("breakdown_by"),
        "secondary_breakdown": None,
        "start_date":          args.get("start_date", s0),
        "end_date":            args.get("end_date",   e0),
        "compare_period":      None,
        "top_k":               None,
        "order_by":            args["metric"],
        "filters":             _build_filters(args.get("filters")),
        "show_extremes":       False,
    }
    df = sql.run(plan)
    if df.empty:
        return "No data found."

    metric = args["metric"]
    if "breakdown" in df.columns:
        rows = []
        for _, r in df.iterrows():
            val = float(r[metric])
            rows.append(f"{r['breakdown']}: ${val:,.0f}" if metric in ("sales","profit")
                        else f"{r['breakdown']}: {val:.1f}{'%' if metric=='profit_margin' else ''}")
        return f"{metric} by {args.get('breakdown_by','dimension')}:\n" + "\n".join(rows)

    r0  = df.iloc[0]
    val = float(r0[metric])
    return f"Total {metric}: {_fmt_metric(metric, val)}"


def _find_anomalies(args: Dict, s0: str, e0: str) -> str:
    plan = {
        "intent":       "kpi_detail",
        "condition":    "profit_negative",
        "metrics":      ["sales", "profit"],
        "time_grain":   "none",
        "breakdown_by": args.get("breakdown_by", "sub_category"),
        "start_date":   args.get("start_date", s0),
        "end_date":     args.get("end_date",   e0),
        "compare_period": None,
        "top_k":        15,
        "order_by":     "profit",
        "filters":      _build_filters(args.get("filters")),
    }
    df = sql.run(plan)
    if df.empty:
        return "No loss-making items found."

    lines = [f"Loss-making items ({len(df)} found):"]
    for _, r in df.iterrows():
        disc = float(r.get("avg_discount_pct", 0))
        lines.append(
            f"  - {r['breakdown']} ({r.get('category','')}): "
            f"loss=${abs(float(r['profit'])):,.0f}, "
            f"sales=${float(r['sales']):,.0f}, "
            f"avg_discount={disc:.0f}%"
        )
    return "\n".join(lines)


def _get_trend(args: Dict, s0: str, e0: str) -> str:
    plan = {
        "intent":              "kpi_trend",
        "metrics":             [args.get("metric", "sales")],
        "time_grain":          args.get("time_grain", "year"),
        "breakdown_by":        args.get("breakdown_by"),
        "secondary_breakdown": None,
        "start_date":          args.get("start_date", s0),
        "end_date":            args.get("end_date",   e0),
        "compare_period":      None,
        "top_k":               None,
        "order_by":            args.get("metric", "sales"),
        "filters":             _build_filters(args.get("filters")),
        "show_extremes":       False,
    }
    df = sql.run(plan)
    if df.empty or "period" not in df.columns:
        return "No trend data found."

    metric = args.get("metric", "sales")
    sdf    = df.sort_values("period")
    lines  = [f"{metric} trend:"]
    prev_v = None
    for _, r in sdf.iterrows():
        p = str(r["period"])[:7]
        v = float(r[metric])
        fv = _fmt_metric(metric, v)
        if prev_v:
            chg = (v - prev_v) / abs(prev_v) * 100 if prev_v else 0
            lines.append(f"  {p}: {fv} ({chg:+.1f}%)")
        else:
            lines.append(f"  {p}: {fv} (baseline)")
        prev_v = v
    return "\n".join(lines)


def _compare_periods(args: Dict, s0: str, e0: str) -> str:
    from datetime import datetime, timedelta
    import pandas as _pd

    metric = args.get("metric", "sales")
    cs     = args.get("current_start",  s0)
    ce     = args.get("current_end",    e0)

    # Default previous = same length window before current
    explicit_previous = "previous_start" in args
    if not explicit_previous:
        sd  = datetime.strptime(cs, "%Y-%m-%d")
        ed  = datetime.strptime(ce, "%Y-%m-%d")
        gap = (ed - sd).days + 1
        ps  = (sd - timedelta(days=gap)).strftime("%Y-%m-%d")
        pe  = (sd - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        ps = args["previous_start"]
        pe = args["previous_end"]

    filters = _build_filters(args.get("filters"))

    def _run(start, end):
        plan = {
            "intent": "kpi_value", "metrics": [metric],
            "time_grain": "none", "breakdown_by": args.get("breakdown_by"),
            "secondary_breakdown": None,
            "start_date": start, "end_date": end,
            "compare_period": None, "top_k": None,
            "order_by": metric, "filters": filters, "show_extremes": False,
        }
        df = sql.run(plan)
        if df.empty:
            return 0.0
        return float(df.iloc[0][metric]) if metric in df.columns else 0.0

    cur = _run(cs, ce)

    # If computed previous period predates available data, skip comparison
    if not explicit_previous and ps < s0:
        return (
            f"{metric} ({cs} → {ce}): {_fmt_metric(metric, cur)}\n"
            f"  (no prior comparison period available for this date range)"
        )

    prev = _run(ps, pe)
    chg  = (cur - prev) / abs(prev) * 100 if prev else 0

    return (
        f"{metric} comparison:\n"
        f"  Current  ({cs} → {ce}): {_fmt_metric(metric, cur)}\n"
        f"  Previous ({ps} → {pe}): {_fmt_metric(metric, prev)}\n"
        f"  Change: {chg:+.1f}%"
    )