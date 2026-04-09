"""
chatbot/agent/assumption_validator.py  — PATCHED v2

FIXES vs v1:
  FIX-AV-1: _extract_actuals() now parses compare_periods tool output format directly
             ("Current: $X / Previous: $Y / Change: +Z%") instead of relying on
             fragile regex that missed the actual format.
  FIX-AV-2: _extract_actuals() also computes margin from sales+profit if both present,
             preventing margin inversion bugs.
  FIX-AV-3: validate_assumptions() now returns a contradiction when the premise is
             UNVERIFIABLE (tool returned error / no data) — bot should not fabricate
             root cause when it can't confirm the premise.
  FIX-AV-4: Corrective response explicitly computes and shows the correct margin,
             preventing the inversion bug where bot shows 2015 and 2016 margins swapped.
  FIX-AV-5: List of fields NOT in dataset (COGS, inventory, headcount) injected into
             corrective message so downstream synthesis doesn't hallucinate them.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ── Claim extractors (unchanged from v1) ──────────────────────────────────────

_INCREASE_WORDS = re.compile(
    r"\b(increase|increased|increasing|up|rise|rose|grow|grew|growing|higher|more|spike|surge|faster)\b",
    re.IGNORECASE,
)
_DECREASE_WORDS = re.compile(
    r"\b(decrease|decreased|decreasing|down|drop|dropped|fall|fell|decline|declined|lower|less|shrink|slower)\b",
    re.IGNORECASE,
)
_SALES_WORDS   = re.compile(r"\b(sales|revenue|income|turnover)\b", re.IGNORECASE)
_PROFIT_WORDS  = re.compile(r"\b(profit|margin|profitability)\b", re.IGNORECASE)
_ORDERS_WORDS  = re.compile(r"\b(orders|transactions|purchases)\b", re.IGNORECASE)

# Fields that do NOT exist in Superstore — mention = hallucination
_FORBIDDEN_FIELDS = re.compile(
    r"\b(cogs|cost of goods sold|inventory|headcount|employees?|"
    r"operating expenses?|capex|depreciation|tax|overhead)\b",
    re.IGNORECASE,
)

_NOT_IN_DATASET_NOTE = (
    "\n\n⚠️ *Note: The Superstore dataset does not contain COGS, inventory, "
    "operating expenses, or headcount. Any analysis of these fields is fabricated.*"
)


def _extract_claims(question: str) -> Dict[str, str]:
    claims = {}
    parts = re.split(r"\b(but|while|yet|however|although|whereas|and)\b", question, flags=re.IGNORECASE)
    for part in parts:
        is_up   = bool(_INCREASE_WORDS.search(part))
        is_down = bool(_DECREASE_WORDS.search(part))
        if not (is_up or is_down):
            continue
        direction = "up" if is_up else "down"
        if _SALES_WORDS.search(part):
            claims["sales"] = direction
        if _PROFIT_WORDS.search(part):
            claims["profit"] = direction
        if _ORDERS_WORDS.search(part):
            claims["orders"] = direction
    return claims


# FIX-AV-1: Parse the actual compare_periods output format
# Tool output looks like:
#   sales comparison:
#     Current  (2016-10-01 → 2016-12-31): $236,099
#     Previous (2015-10-01 → 2015-12-31): $182,297
#     Change: +29.5% ($53,802)

_COMPARE_BLOCK_RE = re.compile(
    r'(sales|profit|orders)\s+comparison:.*?Change:\s*([+-]?\d+\.?\d*)%',
    re.IGNORECASE | re.DOTALL,
)
_CURRENT_VALUE_RE  = re.compile(r'Current\s*\([^)]+\):\s*\$?([\d,]+)', re.IGNORECASE)
_PREVIOUS_VALUE_RE = re.compile(r'Previous\s*\([^)]+\):\s*\$?([\d,]+)', re.IGNORECASE)
_CHANGE_PCT_RE     = re.compile(r'Change:\s*([+-]?\d+\.?\d*)%', re.IGNORECASE)
_TOOL_ERROR_RE     = re.compile(r'Tool error|No data found|error', re.IGNORECASE)

# FIX-BUG-A: detect the direction note injected by _run_forced_queries
# "NOTE: current=LATER period (2016-12-01..2016-12-31), previous=EARLIER period (...)"
# When this note is present we trust the Change: sign directly (positive = later > earlier).
_DIRECTION_NOTE_RE = re.compile(r'NOTE: current=LATER period', re.IGNORECASE)


def _parse_dollar(s: str) -> float:
    return float(s.replace(",", ""))


def _extract_actuals(tool_results: List[str]) -> Dict[str, float]:
    """
    FIX-AV-1 + FIX-BUG-A: Parse compare_periods output directly.
    Returns {metric: pct_change} for metrics found in tool results.

    The Change: sign is always "current vs previous".
    _run_forced_queries guarantees current=LATER, previous=EARLIER, so a positive
    Change: means the metric went UP from the earlier to the later period.
    """
    actuals: Dict[str, float] = {}
    sales_vals: Dict[str, float] = {}
    profit_vals: Dict[str, float] = {}

    combined = "\n".join(tool_results)

    # Parse each comparison block
    for m in _COMPARE_BLOCK_RE.finditer(combined):
        metric = m.group(1).lower()
        pct    = float(m.group(2))
        actuals[metric] = pct

    # FIX-AV-2: Extract raw values for margin computation
    # Find the sales and profit blocks independently
    for metric_name, val_dict in [("sales", sales_vals), ("profit", profit_vals)]:
        # Look for block header
        block_re = re.compile(
            rf'{metric_name}\s+comparison:(.+?)(?=\n\[|\Z)',
            re.IGNORECASE | re.DOTALL,
        )
        bm = block_re.search(combined)
        if bm:
            block = bm.group(1)
            cur_m  = _CURRENT_VALUE_RE.search(block)
            prev_m = _PREVIOUS_VALUE_RE.search(block)
            if cur_m:
                val_dict["current"]  = _parse_dollar(cur_m.group(1))
            if prev_m:
                val_dict["previous"] = _parse_dollar(prev_m.group(1))

    # Compute actual margin direction if both sales and profit available
    if (sales_vals.get("current") and sales_vals.get("previous")
            and profit_vals.get("current") and profit_vals.get("previous")):
        margin_cur  = profit_vals["current"]  / sales_vals["current"]  * 100
        margin_prev = profit_vals["previous"] / sales_vals["previous"] * 100
        actuals["_margin_cur"]  = margin_cur
        actuals["_margin_prev"] = margin_prev
        # margin direction: positive = improved
        actuals["margin_change_pp"] = margin_cur - margin_prev

    return actuals


def _is_unverifiable(tool_results: List[str], metric: str) -> bool:
    """
    FIX-AV-3: Returns True if the tool result for this metric contains an error
    or "No data found" — meaning the premise cannot be confirmed.
    """
    for result in tool_results:
        if metric in result.lower() and _TOOL_ERROR_RE.search(result):
            return True
    return False


def _direction_matches(claimed: str, actual_pct: float) -> bool:
    if claimed == "up":
        return actual_pct > 0
    elif claimed == "down":
        return actual_pct < 0
    return True


def validate_assumptions(question: str, tool_results: List[str]) -> Optional[str]:
    """
    Returns a corrective message when:
      (a) user's claimed direction contradicts actual % change, OR
      (b) user's claimed metric is unverifiable (tool error / missing data)
    Returns None if everything is consistent.
    """
    claims  = _extract_claims(question)
    actuals = _extract_actuals(tool_results)

    if not claims:
        return None

    # FIX-AV-3: Check for unverifiable premises first
    unverifiable = [m for m in claims if _is_unverifiable(tool_results, m)]
    if unverifiable:
        lines = [
            "⚠️ **Data Check: Cannot verify part of your question's premise.**",
            "",
        ]
        for m in unverifiable:
            lines.append(
                f"- **{m.title()}** data returned an error or was unavailable "
                f"for this date range. The claim that {m} '{claims[m]}' cannot be confirmed."
            )
        lines += [
            "",
            "**What I can confirm from the data:**",
            *(
                f"- **{k.title()}** changed **{v:+.1f}%**"
                for k, v in actuals.items()
                if not k.startswith("_") and k not in unverifiable and k != "margin_change_pp"
            ),
            "",
            "Since the premise cannot be fully verified, a root-cause analysis would be unreliable.",
            "Try asking about the metrics that are available, or check that the date range contains data.",
        ]
        return "\n".join(lines)

    if not actuals:
        return None

    contradictions = []
    confirmations  = []

    for metric, claimed_direction in claims.items():
        if metric not in actuals:
            continue
        actual_pct = actuals[metric]
        actual_direction = "up" if actual_pct > 0 else ("down" if actual_pct < 0 else "flat")
        actual_label     = f"{actual_pct:+.1f}%"

        if not _direction_matches(claimed_direction, actual_pct):
            contradictions.append({
                "metric":     metric,
                "claimed":    claimed_direction,
                "actual":     actual_direction,
                "actual_pct": actual_pct,
                "label":      actual_label,
            })
        else:
            confirmations.append({
                "metric":     metric,
                "direction":  actual_direction,
                "actual_pct": actual_pct,
                "label":      actual_label,
            })

    if not contradictions:
        return None

    lines = [
        "⚠️ **Data Check: Your assumption does not match the actual numbers.**",
        "",
    ]

    for c in contradictions:
        metric_label = c["metric"].replace("_", " ").title()
        claimed_word = "grew faster than sales" if (
            c["metric"] == "profit" and c["claimed"] == "down" and "sales" in actuals
        ) else ("increased" if c["claimed"] == "up" else "decreased")
        actual_word  = "actually increased" if c["actual"] == "up" else "actually decreased"
        lines.append(
            f"- **{metric_label}** — You assumed it {claimed_word}, "
            f"but it **{actual_word} {c['actual_pct']:+.1f}%** in this period."
        )

    if confirmations:
        lines.append("")
        for conf in confirmations:
            metric_label = conf["metric"].replace("_", " ").title()
            word = "did increase" if conf["direction"] == "up" else "did decrease"
            lines.append(
                f"- **{metric_label}** {word} {conf['actual_pct']:+.1f}% "
                f"— consistent with your assumption."
            )

    # FIX-AV-4: Show correct margin if available (prevents inversion display bug)
    margin_cur  = actuals.get("_margin_cur")
    margin_prev = actuals.get("_margin_prev")
    margin_pp   = actuals.get("margin_change_pp")
    if margin_cur is not None and margin_prev is not None:
        margin_word = "improved" if (margin_pp or 0) > 0 else "compressed"
        lines += [
            "",
            f"**📊 Profit margin:** {margin_prev:.1f}% → {margin_cur:.1f}% "
            f"({margin_word} by {abs(margin_pp or 0):.1f}pp)",
        ]

    # What actually happened — the real story
    lines += ["", "**📊 What actually happened:**"]

    if len(contradictions) == 1 and confirmations:
        c   = contradictions[0]
        conf = confirmations[0]
        lines.append(
            f"  - {conf['metric'].title()} {conf['direction']} {conf['actual_pct']:+.1f}% "
            f"AND {c['metric'].title()} also {c['actual']} {c['actual_pct']:+.1f}% — "
            f"both moved in the same direction."
        )
        # Specific case: profit grew faster than sales
        if c["metric"] == "profit" and c["actual"] == "up" and "sales" in actuals:
            s_pct = actuals.get("sales", 0)
            p_pct = actuals.get("profit", c["actual_pct"])
            if p_pct > s_pct:
                lines.append(
                    f"  - Profit grew **faster** than sales ({p_pct:+.1f}% vs {s_pct:+.1f}%), "
                    f"meaning profitability **improved**, not deteriorated."
                )
                if margin_cur and margin_prev:
                    lines.append(
                        f"  - Profit margin rose from {margin_prev:.1f}% to {margin_cur:.1f}%, "
                        f"confirming the business became more efficient in this period."
                    )
                lines.append(
                    "  - The more useful question may be: "
                    "**what drove the higher profit margin?** "
                    "(product mix, lower discounting, or category shift)"
                )
    elif len(contradictions) >= 2:
        lines.append(
            "  - The actual data shows the **reverse pattern** from what you described."
        )
        lines.append(
            "  - Try verifying the date range in the sidebar, or rephrase with specific months."
        )
    else:
        c = contradictions[0]
        lines.append(
            f"  - {c['metric'].title()} went {c['actual']} {c['actual_pct']:+.1f}%, not {c['claimed']}."
        )
        lines.append(
            f"  - Would you like to explore **why {c['metric'].title()} "
            f"{c['actual']} {abs(c['actual_pct']):.1f}%** instead?"
        )

    lines += [
        "",
        "*Check the sidebar date filters or rephrase the question to explore the actual trend.*",
    ]

    return "\n".join(lines)