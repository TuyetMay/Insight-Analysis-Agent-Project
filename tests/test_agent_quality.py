"""
tests/test_agent_quality.py  — v2
────────────────────────────────────────────────────────────────────────────────
Fixes vs v1:

  FIX-AQ-1: Exclude "no data" responses from factual accuracy pool
    v1: "❌ Could not gather enough data" → 0 claims → factual_acc = 1.0 (inflated)
    v2: These responses counted separately as "agent_failed" — excluded.

  FIX-AQ-2: Exclude pure corrective responses from factual accuracy pool
    v1: AssumptionValidator short corrections fail DB verification due to
        different date window → artificially depresses grounded accuracy.
    v2: Responses starting with "⚠️ Data Check" tagged as "corrective" — excluded.

  FIX-AQ-3: Dynamic DB verifier — extract dates from query + answer
    v1: Hardcoded SQL with fixed year windows → Q4 values vs full-year verifier
    v2: Extract year/quarter/month from combined query+answer text → correct window.

  FIX-AQ-4: Tolerance 2% → 5% for percentage values
    v1: 2% tolerance too strict for computed ratios (margin%, change%)
    v2: Dollar values 2%, percentage values 5%.

  FIX-AQ-5: Report factual accuracy only on answerable diagnostic queries.
"""

from __future__ import annotations

import os
import sys
import re
import json
import time
import calendar as _cal
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from config import Config
from google import genai
from core.data_loader import load_filtered_data_safe, calculate_kpis, get_filter_options
from core.database import execute_query
from chatbot.agent.orchestrator import AgentOrchestrator, _is_hallucinated_round_number
from chatbot.agent.assumption_validator import validate_assumptions

# ── Setup ─────────────────────────────────────────────────────────────────────
gemini_client = genai.Client(
    vertexai=True,
    project=Config.GCP_PROJECT,
    location=Config.GCP_LOCATION,
)
model_name = Config.GEMINI_MODEL

filter_options = get_filter_options()
filters = {
    "date_range": (filter_options["min_date"], filter_options["max_date"]),
    "region":     filter_options["region"],
    "segment":    filter_options["segment"],
    "category":   filter_options["category"],
}
df   = load_filtered_data_safe(filters)
kpis = calculate_kpis(df)

s0 = str(filter_options["min_date"])[:10]
e0 = str(filter_options["max_date"])[:10]
TABLE = Config.DB_TABLE

agent = AgentOrchestrator(
    gemini_client=gemini_client,
    model_name=model_name,
    default_start=s0,
    default_end=e0,
)
print("Setup complete.")

# ── Response type classifier ──────────────────────────────────────────────────

def classify_response(answer: str) -> str:
    """
    FIX-AQ-1 + FIX-AQ-2:
      'no_data'    — agent failed, excluded from accuracy
      'corrective' — AssumptionValidator intercepted, excluded from accuracy
      'diagnostic' — full response, included in accuracy
    """
    if not answer:
        return "no_data"
    stripped = answer.strip()
    if stripped.startswith("❌") or "Could not gather" in stripped:
        return "no_data"
    if stripped.startswith("⚠️ **Data Check") or stripped.startswith("⚠️ **API quota"):
        return "corrective"
    return "diagnostic"

# ── Dynamic DB verifier (FIX-AQ-3) ───────────────────────────────────────────

_YEAR_RE    = re.compile(r'\b(201[4-9])\b')
_QUARTER_RE = re.compile(r'\bQ([1-4])\s*(201[4-9])\b', re.IGNORECASE)
_MONTH_MAP  = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
}
_MONTH_RE = re.compile(
    r'\b(' + '|'.join(_MONTH_MAP.keys()) + r')\s+(201[4-9])\b',
    re.IGNORECASE,
)


def _extract_date_window(text: str) -> Optional[Tuple[str, str]]:
    """Extract most specific date window (month > quarter > year)."""
    m = _MONTH_RE.search(text)
    if m:
        month    = _MONTH_MAP[m.group(1).lower()]
        year     = int(m.group(2))
        last_day = _cal.monthrange(year, month)[1]
        return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last_day:02d}"

    m = _QUARTER_RE.search(text)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        qs = {1:(1,1), 2:(4,1), 3:(7,1), 4:(10,1)}
        qe = {1:(3,31), 2:(6,30), 3:(9,30), 4:(12,31)}
        sm, sd = qs[q]; em, ed = qe[q]
        return f"{year}-{sm:02d}-{sd:02d}", f"{year}-{em:02d}-{ed:02d}"

    years = _YEAR_RE.findall(text)
    if years:
        year = max(int(y) for y in years)
        return f"{year}-01-01", f"{year}-12-31"

    return None


def _scalar(sql: str, params: dict = None) -> Optional[float]:
    df_r = execute_query(sql, params)
    if df_r is None or df_r.empty:
        return None
    try:
        return float(df_r.iloc[0, 0])
    except Exception:
        return None


def _build_dynamic_verifiers(query: str, answer: str) -> List[float]:
    """Build verifier values using dynamic date window from query+answer."""
    window = _extract_date_window(f"{query} {answer}") or (s0, e0)
    start, end = window
    params = {"start": start, "end": end}
    base   = f"FROM {TABLE} WHERE order_date >= %(start)s AND order_date <= %(end)s"

    db_values = []
    for sql in [
        f"SELECT SUM(sales)                          {base}",
        f"SELECT SUM(profit)                         {base}",
        f"SELECT COUNT(DISTINCT order_id)            {base}",
        f"SELECT SUM(profit)/SUM(sales)*100          {base}",
    ]:
        v = _scalar(sql, params)
        if v is not None:
            db_values.extend([v, round(v, -2), round(v, -3)])
            if abs(v) > 100:
                db_values.extend([round(v/1000, 1), round(v/1000000, 2)])

    # Breakdown values
    for dim in ("region", "segment", "category"):
        sql = (
            f"SELECT {dim}, SUM(sales) s, SUM(profit) p "
            f"FROM {TABLE} "
            f"WHERE order_date >= %(start)s AND order_date <= %(end)s "
            f"GROUP BY {dim}"
        )
        df_r = execute_query(sql, params)
        if df_r is not None and not df_r.empty:
            for _, row in df_r.iterrows():
                for col in ("s", "p"):
                    try:
                        val = float(row[col])
                        db_values.extend([val, round(val, -2), round(val, -3)])
                    except Exception:
                        pass

    return list(set(v for v in db_values if v is not None))


def _approx(a: float, b: float, tol: float) -> bool:
    if b == 0:
        return abs(a) < 1
    return abs(a - b) / abs(b) <= tol


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    nums = []
    for m in re.finditer(r'\$([\d,]+(?:\.\d+)?)', text):
        try:
            nums.append(("dollar", float(m.group(1).replace(",", ""))))
        except Exception:
            pass
    for m in re.finditer(r'([\d]+\.?[\d]*)%', text):
        try:
            nums.append(("pct", float(m.group(1))))
        except Exception:
            pass
    return nums


def verify_claim(claim_type: str, value: float, db_values: List[float]) -> bool:
    """FIX-AQ-4: 5% tolerance for pct, 2% for dollar."""
    tol = 0.05 if claim_type == "pct" else 0.02
    return any(_approx(value, db_val, tol) for db_val in db_values)


def check_completeness(text: str) -> bool:
    sections = ["Key Metrics", "Root Cause", "Supporting Evidence", "Recommended Actions"]
    return all(s in text for s in sections)

# ── QUERIES ───────────────────────────────────────────────────────────────────

AGENT_QUERIES = [
    {"q": "Why did profit drop in Q4 2016?",                           "category": "false_premise"},
    {"q": "What caused the revenue decline in the Central region?",    "category": "false_premise"},
    {"q": "Why is the Furniture category underperforming?",            "category": "genuine_diagnostic"},
    {"q": "What drove the sales spike in November 2017?",              "category": "genuine_diagnostic"},
    {"q": "Why does the South region have lower profit than the West?","category": "genuine_diagnostic"},
    {"q": "What is causing the margin compression in 2016?",           "category": "false_premise"},
    {"q": "Why are Tables and Bookcases losing money?",                "category": "genuine_diagnostic"},
    {"q": "What caused the high discount rate in the Central region?", "category": "genuine_diagnostic"},
    {"q": "Why did orders drop in Q1 2017?",                           "category": "false_premise"},
    {"q": "What is driving the profit growth in the West region?",     "category": "genuine_diagnostic"},
    {"q": "Why is the Home Office segment more profitable than Consumer?", "category": "genuine_diagnostic"},
    {"q": "Why did profit decline in 2016 despite revenue growth?",    "category": "false_premise"},
    {"q": "Why does heavy discounting hurt profitability?",            "category": "genuine_diagnostic"},
    {"q": "What is causing loss-making in the Tables sub-category?",   "category": "genuine_diagnostic"},
    {"q": "Why is Technology the most profitable category?",           "category": "genuine_diagnostic"},
    {"q": "Why did Q4 always outperform other quarters in sales?",     "category": "genuine_diagnostic"},
    {"q": "What drove Consumer segment growth from 2014 to 2017?",     "category": "genuine_diagnostic"},
    {"q": "What caused the 2016 to 2017 profit margin improvement?",   "category": "genuine_diagnostic"},
    {"q": "Why is the East region second in profitability?",           "category": "genuine_diagnostic"},
    {"q": "Why does the Corporate segment have lower margin than Home Office?", "category": "false_premise"},
]

FALSE_PREMISE_QUERIES = [
    {"q": "Why did sales drop from 2015 to 2016?",
     "false_metric": "sales",  "false_direction": "down",
     "current_sql":  f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
     "previous_sql": f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'"},
    {"q": "Why did revenue decline in 2017 compared to 2016?",
     "false_metric": "sales",  "false_direction": "down",
     "current_sql":  f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
     "previous_sql": f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'"},
    {"q": "Why did orders fall in 2016?",
     "false_metric": "orders", "false_direction": "down",
     "current_sql":  f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
     "previous_sql": f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'"},
    {"q": "Why did profit decrease from 2016 to 2017?",
     "false_metric": "profit", "false_direction": "down",
     "current_sql":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
     "previous_sql": f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'"},
    {"q": "Why did profit fall in Q1 2016 compared to Q1 2015?",
     "false_metric": "profit", "false_direction": "down",
     "current_sql":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-03-31'",
     "previous_sql": f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-03-31'"},
    {"q": "Why did sales increase but profit decrease from 2016 to 2017?",
     "false_metric": "profit", "false_direction": "down",
     "current_sql":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
     "previous_sql": f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'"},
    {"q": "Why did revenue grow but orders declined in 2016?",
     "false_metric": "orders", "false_direction": "down",
     "current_sql":  f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
     "previous_sql": f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'"},
    {"q": "Why did profit collapse in 2014?",
     "false_metric": "profit", "false_direction": "down",
     "current_sql":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2014-01-01' AND '2014-12-31'",
     "previous_sql": None},
    {"q": "Why did West region sales decline in 2017?",
     "false_metric": "sales",  "false_direction": "down",
     "current_sql":  f"SELECT SUM(sales) FROM {TABLE} WHERE region='West' AND order_date BETWEEN '2017-01-01' AND '2017-12-31'",
     "previous_sql": f"SELECT SUM(sales) FROM {TABLE} WHERE region='West' AND order_date BETWEEN '2016-01-01' AND '2016-12-31'"},
    {"q": "Why did Consumer segment profit shrink from 2015 to 2016?",
     "false_metric": "profit", "false_direction": "down",
     "current_sql":  f"SELECT SUM(profit) FROM {TABLE} WHERE segment='Consumer' AND order_date BETWEEN '2016-01-01' AND '2016-12-31'",
     "previous_sql": f"SELECT SUM(profit) FROM {TABLE} WHERE segment='Consumer' AND order_date BETWEEN '2015-01-01' AND '2015-12-31'"},
]

# ── PART 1: Factual Accuracy ──────────────────────────────────────────────────

def run_agent_eval(queries, label, use_forced_prequery=True):
    print(f"\n{'='*80}")
    print(f"AGENT EVALUATION — {label}")
    print(f"{'='*80}")

    import chatbot.agent.orchestrator as _orch_mod
    _orig = _orch_mod.AgentOrchestrator._run_forced_queries

    if not use_forced_prequery:
        _orch_mod.AgentOrchestrator._run_forced_queries = lambda self, q: []

    results = []
    print(f"\n{'Q':<4} {'Category':<22} {'RespType':<12} {'FactAcc':>8} {'Complete':>9}  Query")
    print("-"*85)

    for i, item in enumerate(queries, 1):
        q        = item["q"]
        category = item.get("category", "unknown")

        try:
            answer = agent.run(q)
        except Exception as e:
            answer = f"❌ ERROR: {e}"

        resp_type = classify_response(answer)
        halluc    = _is_hallucinated_round_number(answer)
        complete  = check_completeness(answer) if resp_type == "diagnostic" else False

        if resp_type == "diagnostic":
            db_values = _build_dynamic_verifiers(q, answer)
            claims    = extract_numbers(answer)
            if claims and db_values:
                supported = sum(
                    1 for ctype, v in claims
                    if verify_claim(ctype, v, db_values)
                )
                fact_acc = supported / len(claims)
            else:
                fact_acc  = 1.0 if not claims else 0.0
                supported = 0
        else:
            fact_acc  = None
            supported = 0
            claims    = []

        acc_str = f"{fact_acc:.3f}" if fact_acc is not None else "n/a    "
        print(f"Q{i:02d}  {category:<22} {resp_type:<12} {acc_str:>8} {'Y' if complete else 'N':>9}  {q[:38]}")

        time.sleep(2.0)

        results.append({
            "query": q, "category": category, "answer": answer,
            "resp_type": resp_type,
            "n_claims": len(claims), "n_supported": supported,
            "factual_acc": round(fact_acc, 3) if fact_acc is not None else None,
            "hallucinated": halluc, "complete": complete,
        })

    if not use_forced_prequery:
        _orch_mod.AgentOrchestrator._run_forced_queries = _orig

    n_total      = len(results)
    n_diagnostic = sum(1 for r in results if r["resp_type"] == "diagnostic")
    n_corrective = sum(1 for r in results if r["resp_type"] == "corrective")
    n_no_data    = sum(1 for r in results if r["resp_type"] == "no_data")

    diag         = [r for r in results if r["resp_type"] == "diagnostic"]
    mean_acc     = sum(r["factual_acc"] for r in diag) / n_diagnostic if n_diagnostic else 0.0
    fully_acc    = sum(1 for r in diag if r["factual_acc"] == 1.0)
    halluc_n     = sum(1 for r in results if r["hallucinated"])
    complete_n   = sum(1 for r in results if r["complete"])

    print(f"\nSUMMARY ({label}):")
    print(f"  Total queries          : {n_total}")
    print(f"  Diagnostic responses   : {n_diagnostic}  ← factual accuracy evaluated on these")
    print(f"  Corrective responses   : {n_corrective}  ← AssumptionValidator intercepted")
    print(f"  No-data responses      : {n_no_data}   ← agent failed")
    print(f"  ────────────────────────────────────────")
    print(f"  Mean factual accuracy  : {mean_acc*100:.1f}%  (on {n_diagnostic} diagnostic)")
    print(f"  Fully accurate         : {fully_acc}/{n_diagnostic}")
    print(f"  Hallucination rate     : {halluc_n}/{n_total} = {halluc_n/n_total*100:.1f}%")
    print(f"  Response completeness  : {complete_n}/{n_total} = {complete_n/n_total*100:.1f}%")

    return results, {
        "label": label, "n_total": n_total,
        "n_diagnostic": n_diagnostic, "n_corrective": n_corrective, "n_no_data": n_no_data,
        "mean_factual_acc_pct": round(mean_acc * 100, 1),
        "fully_accurate": fully_acc,
        "hallucination_count": halluc_n,
        "hallucination_rate_pct": round(halluc_n / n_total * 100, 1),
        "completeness_count": complete_n,
        "completeness_rate_pct": round(complete_n / n_total * 100, 1),
    }


print("\nRunning GROUNDED agent...")
results_grounded, summary_grounded = run_agent_eval(AGENT_QUERIES, "WITH PRE-QUERY (grounded)", True)

print("\nRunning BASELINE agent...")
results_baseline, summary_baseline = run_agent_eval(AGENT_QUERIES, "WITHOUT PRE-QUERY (baseline)", False)

# ── PART 2: AssumptionValidator ───────────────────────────────────────────────

print(f"\n{'='*80}")
print("ASSUMPTION VALIDATOR — n=10 false-premise queries")
print(f"{'='*80}")
print(f"{'Q':<4} {'Detected':>8} {'Corrected':>10} {'Dir':>8}  Query")
print("-"*70)

av_results = []
for i, item in enumerate(FALSE_PREMISE_QUERIES, 1):
    q, false_metric, false_dir = item["q"], item["false_metric"], item["false_direction"]

    cur_val  = _scalar(item["current_sql"])  if item["current_sql"]  else None
    prev_val = _scalar(item["previous_sql"]) if item["previous_sql"] else None

    if cur_val is not None and prev_val is not None:
        actual_dir   = "up" if cur_val > prev_val else "down"
        premise_false = actual_dir != false_dir
    else:
        actual_dir, premise_false = "unknown", False

    fake_results = []
    if cur_val and prev_val:
        chg = (cur_val - prev_val) / abs(prev_val) * 100
        fake_results.append(
            f"{false_metric} comparison:\n"
            f"  Current  (...): ${cur_val:,.0f}\n"
            f"  Previous (...): ${prev_val:,.0f}\n"
            f"  Change: {chg:+.1f}%"
        )

    correction = validate_assumptions(q, fake_results) if fake_results else None
    detected   = correction is not None
    corrected  = False
    if detected and correction:
        actual_word = "increased" if actual_dir == "up" else "decreased"
        corrected   = actual_word in correction.lower() or "actually" in correction.lower()

    ptype = (
        "divergence" if ("but" in q.lower() or "despite" in q.lower())
        else ("sales_drop" if false_metric in ("sales", "orders") else "profit_drop")
    )

    print(f"Q{i:02d}  {'✓' if detected else '✗':>8}  {'✓' if corrected else '✗':>10}  {actual_dir:>8}  {q[:42]}")
    av_results.append({
        "id": i, "query": q, "false_metric": false_metric, "false_direction": false_dir,
        "actual_direction": actual_dir, "premise_is_false": premise_false,
        "detected": detected, "corrected": corrected, "type": ptype,
    })
    time.sleep(0.2)

types_map = {"sales_drop": "Sales direction inverted", "profit_drop": "Profit direction inverted", "divergence": "Divergence claim"}
av_by_type = {}
print("\nASSUMPTION VALIDATOR SUMMARY:")
for ptype, label in types_map.items():
    subset = [r for r in av_results if r["type"] == ptype]
    n = len(subset)
    if n == 0:
        continue
    det = sum(1 for r in subset if r["detected"])
    cor = sum(1 for r in subset if r["corrected"])
    mis = n - det
    print(f"  {label}: n={n}  detected={det/n*100:.1f}%  corrected={cor/n*100:.1f}%  missed={mis/n*100:.1f}%")
    av_by_type[ptype] = {"n": n, "detected_pct": round(det/n*100,1), "corrected_pct": round(cor/n*100,1), "missed_pct": round(mis/n*100,1)}

av_n   = len(av_results)
av_det = sum(1 for r in av_results if r["detected"])
av_cor = sum(1 for r in av_results if r["corrected"])
av_mis = av_n - av_det
print(f"  OVERALL: n={av_n}  detected={av_det/av_n*100:.1f}%  corrected={av_cor/av_n*100:.1f}%  missed={av_mis/av_n*100:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────

g, b = summary_grounded, summary_baseline
delta_acc  = round(g["mean_factual_acc_pct"] - b["mean_factual_acc_pct"], 1)
delta_comp = round(g["completeness_rate_pct"] - b["completeness_rate_pct"], 1)
delta_hall = round(g["hallucination_rate_pct"] - b["hallucination_rate_pct"], 1)

output = {
    "methodology": {
        "factual_accuracy_scope": "Diagnostic responses only (no_data and corrective excluded)",
        "db_verifier": "Dynamic date window from query+answer text",
        "tolerance": "Dollar 2%, Percentage 5%",
    },
    "factual_accuracy": {
        "grounded": {**g, "per_query": results_grounded},
        "baseline": {**b, "per_query": results_baseline},
        "delta": {"factual_acc_pp": delta_acc, "completeness_pp": delta_comp, "hallucination_pp": delta_hall},
    },
    "assumption_validator": {
        "n": av_n, "detected": av_det, "detected_pct": round(av_det/av_n*100,1),
        "corrected": av_cor, "corrected_pct": round(av_cor/av_n*100,1),
        "missed": av_mis, "missed_pct": round(av_mis/av_n*100,1),
        "by_type": av_by_type, "per_query": av_results,
    }
}

out_json = os.path.join(_PROJECT_ROOT, "agent_results.json")
out_txt  = os.path.join(_PROJECT_ROOT, "agent_results.txt")

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

with open(out_txt, "w", encoding="utf-8") as f:
    f.write("AGENT EVALUATION RESULTS — Section 5.6  (v2)\n")
    f.write("="*60 + "\n\n")
    f.write("METHODOLOGY NOTE\n")
    f.write(f"  Factual accuracy measured on DIAGNOSTIC responses only.\n")
    f.write(f"  No-data and corrective responses excluded from accuracy pool.\n\n")
    f.write("RESPONSE BREAKDOWN\n")
    f.write(f"  Grounded:  diagnostic={g['n_diagnostic']}  corrective={g['n_corrective']}  no_data={g['n_no_data']}  /20\n")
    f.write(f"  Baseline:  diagnostic={b['n_diagnostic']}  corrective={b['n_corrective']}  no_data={b['n_no_data']}  /20\n\n")
    f.write("TABLE 5.12 — Factual Accuracy & Response Quality\n")
    f.write(f"  Factual accuracy:     grounded={g['mean_factual_acc_pct']}%  baseline={b['mean_factual_acc_pct']}%  delta={delta_acc:+.1f}pp\n")
    f.write(f"  Fully accurate:       grounded={g['fully_accurate']}/{g['n_diagnostic']}  baseline={b['fully_accurate']}/{b['n_diagnostic']}\n")
    f.write(f"  Hallucination rate:   grounded={g['hallucination_rate_pct']}%  baseline={b['hallucination_rate_pct']}%  delta={delta_hall:+.1f}pp\n")
    f.write(f"  Completeness:         grounded={g['completeness_rate_pct']}%  baseline={b['completeness_rate_pct']}%  delta={delta_comp:+.1f}pp\n\n")
    f.write("TABLE 5.13 — AssumptionValidator\n")
    for ptype, s in av_by_type.items():
        f.write(f"  {types_map[ptype]}: n={s['n']}  detected={s['detected_pct']}%  corrected={s['corrected_pct']}%  missed={s['missed_pct']}%\n")
    f.write(f"  OVERALL: n={av_n}  detected={av_det/av_n*100:.1f}%  corrected={av_cor/av_n*100:.1f}%  missed={av_mis/av_n*100:.1f}%\n")

print(f"\nSaved → {out_txt}")
print(f"Saved → {out_json}")
print("="*60)