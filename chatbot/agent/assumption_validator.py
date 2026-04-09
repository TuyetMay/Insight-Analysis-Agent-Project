"""
chatbot/agent/assumption_validator.py
Detects when user's premise contradicts actual data collected by tools.

Examples:
  User: "why is sales increasing but profit decreasing?"
  Data: sales -18.9%, profit +76.5%
  → Contradiction detected → return corrective response

  User: "why did sales drop in October?"
  Data: sales actually +14% in October
  → Contradiction detected → correct the user

This is an "Assumption Validation Engine" — advanced BI behavior that
distinguishes a professional assistant from a basic chatbot.
"""

from __future__ import annotations

import re
from typing import Optional


# ── Claim extractors ───────────────────────────────────────────────────────────

_INCREASE_WORDS = re.compile(
    r"\b(increase|increased|increasing|up|rise|rose|grow|grew|growing|higher|more|spike|surge)\b",
    re.IGNORECASE,
)
_DECREASE_WORDS = re.compile(
    r"\b(decrease|decreased|decreasing|down|drop|dropped|fall|fell|decline|declined|lower|less|shrink)\b",
    re.IGNORECASE,
)

# Metric extractors
_SALES_WORDS   = re.compile(r"\b(sales|revenue|income|turnover)\b", re.IGNORECASE)
_PROFIT_WORDS  = re.compile(r"\b(profit|margin|profitability)\b", re.IGNORECASE)
_ORDERS_WORDS  = re.compile(r"\b(orders|transactions|purchases)\b", re.IGNORECASE)


def _extract_claims(question: str) -> dict:
    """
    Parse user question to extract directional claims per metric.
    Returns: {metric: "up" | "down" | None}

    Example:
      "why is sales up but profit down" → {"sales": "up", "profit": "down"}
      "why did profit drop"             → {"profit": "down"}
    """
    claims = {}

    # Split on "but", "while", "yet", "however", "although", "whereas"
    # to handle compound claims like "sales up but profit down"
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


def _extract_actuals(tool_results: list[str]) -> dict:
    """
    Parse tool results to extract actual % changes per metric.
    Returns: {metric: float_pct_change}

    Looks for patterns like:
      "sales comparison: ... Change: -18.9%"
      "profit comparison: ... Change: +76.5%"
    """
    actuals = {}
    combined = "\n".join(tool_results)

    # Pattern: "sales comparison:" or "profit comparison:" followed by "Change: ±X%"
    for metric in ("sales", "profit", "orders"):
        # Find the section for this metric
        section_pattern = rf"{metric}\s+comparison:.*?Change:\s*([+-]?\d+\.?\d*)%"
        m = re.search(section_pattern, combined, re.IGNORECASE | re.DOTALL)
        if m:
            actuals[metric] = float(m.group(1))
            continue

        # Alternative: "Total sales: +X%" or direct trend numbers
        # get_trend output: "2016-10: $X (+Y%)" style
        trend_pattern = rf"{metric}\s+trend:.*?(\d{{4}}-\d{{2}}.*?\(([+-]?\d+\.?\d*)%\))"
        m = re.search(trend_pattern, combined, re.IGNORECASE | re.DOTALL)
        if m:
            # Take the last % change found for this metric
            all_pcts = re.findall(rf"([+-]\d+\.?\d*)%", combined)
            if all_pcts:
                actuals[metric] = float(all_pcts[-1])

    return actuals


def _direction_matches(claimed: str, actual_pct: float) -> bool:
    """Does the claimed direction match the actual % change?"""
    if claimed == "up":
        return actual_pct > 0
    elif claimed == "down":
        return actual_pct < 0
    return True


def validate_assumptions(question: str, tool_results: list[str]) -> Optional[str]:
    """
    Main entry point. Returns a corrective message if the user's premise
    contradicts the data, or None if everything is consistent.

    Args:
        question:     Original user question
        tool_results: List of strings from tool executions

    Returns:
        Corrective response string, or None if no contradiction found
    """
    claims  = _extract_claims(question)
    actuals = _extract_actuals(tool_results)

    if not claims or not actuals:
        return None

    contradictions = []
    confirmations  = []

    for metric, claimed_direction in claims.items():
        if metric not in actuals:
            continue
        actual_pct = actuals[metric]
        actual_direction = "up" if actual_pct > 0 else ("down" if actual_pct < 0 else "flat")
        actual_label = f"{actual_pct:+.1f}%"

        if not _direction_matches(claimed_direction, actual_pct):
            contradictions.append({
                "metric":    metric,
                "claimed":   claimed_direction,
                "actual":    actual_direction,
                "actual_pct": actual_pct,
                "label":     actual_label,
            })
        else:
            confirmations.append({
                "metric":    metric,
                "direction": actual_direction,
                "actual_pct": actual_pct,
                "label":     actual_label,
            })

    if not contradictions:
        return None

    # Build corrective response
    lines = [
        "⚠️ **Data Check: Your assumption may not match the actual numbers.**",
        "",
    ]

    for c in contradictions:
        metric_label = c["metric"].replace("_", " ").title()
        claimed_word = "increased" if c["claimed"] == "up" else "decreased"
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
            lines.append(f"- **{metric_label}** {word} {conf['actual_pct']:+.1f}% — consistent with your assumption.")

    # Add reframed analysis based on actual data
    lines += ["", "**📊 What actually happened:**"]

    if contradictions and confirmations:
        # Mixed: some match, some don't
        matched_metric    = confirmations[0]["metric"]
        matched_pct       = confirmations[0]["actual_pct"]
        contradicted      = contradictions[0]
        c_metric          = contradicted["metric"]
        c_pct             = contradicted["actual_pct"]
        c_matched_label   = matched_metric.replace("_", " ").title()
        c_contract_label  = c_metric.replace("_", " ").title()

        lines.append(
            f"  - {c_matched_label} moved {matched_pct:+.1f}% as expected, "
            f"but {c_contract_label} moved in the opposite direction ({c_pct:+.1f}%)."
        )
        lines.append(
            f"  - The more interesting question may be: "
            f"**why did {c_contract_label} {contradicted['actual']} when {c_matched_label} went the other way?**"
        )
    elif len(contradictions) >= 2:
        # Both contradicted: the real data is opposite to what user thinks
        lines.append(
            "  - The actual data shows the **reverse pattern** from what you described."
        )
        lines.append(
            "  - This often happens with date range confusion or cached filter selections."
        )
        lines.append(
            "  - Try verifying the date range in the sidebar, or rephrase with specific months."
        )
    else:
        # Single contradiction
        c = contradictions[0]
        metric_label = c["metric"].replace("_", " ").title()
        lines.append(
            f"  - {metric_label} went {c['actual']} {c['actual_pct']:+.1f}%, not {c['claimed']}."
        )
        lines.append(
            f"  - Would you like to explore **why {metric_label} {c['actual']} "
            f"{abs(c['actual_pct']):.1f}%** instead?"
        )

    lines += [
        "",
        "*Check the sidebar date filters or ask about the actual trend to continue.*",
    ]

    return "\n".join(lines)