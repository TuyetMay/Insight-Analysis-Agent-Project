"""
test_agent_quality.py
────────────────────────────────────────────────────────────────────────────────
Evaluates diagnostic agent response quality (Section 5.6).

Metrics:
  1. Factual accuracy  — % numeric claims verified against live DB (FActScore-style)
  2. Hallucination rate — % responses flagged by round-number detector
  3. Premise correction — AssumptionValidator catch rate on 10 false-premise queries
  4. Response completeness — % responses with all 4 required sections

Produces:
  agent_results.txt   — table values ready to paste into Section 5.6
  agent_results.json  — full per-query detail

Usage:
    python test_agent_quality.py

Place in project root. Requires .env with GCP_PROJECT and GCP_LOCATION.
"""

import os, sys, re, json, time
from dotenv import load_dotenv
load_dotenv()
sys.path.append('..')

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
model_name    = Config.GEMINI_MODEL

filter_options = get_filter_options()
filters = {
    "date_range": (filter_options["min_date"], filter_options["max_date"]),
    "region": filter_options["region"],
    "segment": filter_options["segment"],
    "category": filter_options["category"],
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

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scalar(sql, params=None):
    df_r = execute_query(sql, params)
    if df_r.empty: return None
    return float(df_r.iloc[0, 0])

def extract_numbers(text):
    """Extract all $ and % numeric values from text."""
    nums = []
    for m in re.finditer(r'\$([\d,]+(?:\.\d+)?)', text):
        try: nums.append(("dollar", float(m.group(1).replace(",", ""))))
        except: pass
    for m in re.finditer(r'([\d]+\.?[\d]*)%', text):
        try: nums.append(("pct", float(m.group(1))))
        except: pass
    return nums

def verify_claim(claim_type, value, db_verifiers):
    """
    Check if value is within 2% tolerance of any DB verifier result.
    db_verifiers: list of scalar DB values relevant to this response.
    """
    for db_val in db_verifiers:
        if db_val is None: continue
        if db_val == 0:
            if abs(value) < 1: return True
        else:
            if abs(value - db_val) / abs(db_val) <= 0.02: return True
    return False

def check_completeness(text):
    """True if all 4 required sections are present."""
    sections = ["Key Metrics", "Root Cause", "Supporting Evidence", "Recommended Actions"]
    return all(s in text for s in sections)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

# 20 diagnostic queries + ground-truth DB verifier SQL for each
AGENT_QUERIES = [
    {
        "q": "Why did profit drop in Q4 2016?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-10-01' AND '2016-12-31'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2015-10-01' AND '2015-12-31'",
            f"SELECT SUM(sales)  FROM {TABLE} WHERE order_date BETWEEN '2016-10-01' AND '2016-12-31'",
        ]
    },
    {
        "q": "What caused the revenue decline in the Central region?",
        "db_verifiers": [
            f"SELECT SUM(sales)  FROM {TABLE} WHERE region='Central'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='Central'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE region='Central'",
        ]
    },
    {
        "q": "Why is the Furniture category underperforming?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE category='Furniture'",
            f"SELECT SUM(sales)  FROM {TABLE} WHERE category='Furniture'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE category='Furniture'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE}",
        ]
    },
    {
        "q": "What drove the sales spike in November 2017?",
        "db_verifiers": [
            f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2017-11-01' AND '2017-11-30'",
            f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-11-01' AND '2016-11-30'",
        ]
    },
    {
        "q": "Why does the South region have lower profit than the West?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='South'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='West'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE region='South'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE region='West'",
        ]
    },
    {
        "q": "What is causing the margin compression in 2016?",
        "db_verifiers": [
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'",
        ]
    },
    {
        "q": "Why are Tables and Bookcases losing money?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE sub_category='Tables'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE sub_category='Bookcases'",
            f"SELECT AVG(discount) FROM {TABLE} WHERE sub_category='Tables'",
        ]
    },
    {
        "q": "What caused the high discount rate in the Central region?",
        "db_verifiers": [
            f"SELECT AVG(discount)*100 FROM {TABLE} WHERE region='Central'",
            f"SELECT AVG(discount)*100 FROM {TABLE}",
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='Central'",
        ]
    },
    {
        "q": "Why did orders drop in Q1 2017?",
        "db_verifiers": [
            f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-03-31'",
            f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-03-31'",
        ]
    },
    {
        "q": "What is driving the profit growth in the West region?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='West'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE region='West'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='West' AND order_date BETWEEN '2016-01-01' AND '2016-12-31'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='West' AND order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        ]
    },
    {
        "q": "Why is the Home Office segment more profitable than Consumer?",
        "db_verifiers": [
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE segment='Home Office'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE segment='Consumer'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE segment='Home Office'",
        ]
    },
    {
        "q": "Why did profit decline in 2016 despite revenue growth?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'",
            f"SELECT SUM(sales)  FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
            f"SELECT SUM(sales)  FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'",
        ]
    },
    {
        "q": "Why does heavy discounting hurt profitability?",
        "db_verifiers": [
            f"SELECT AVG(profit) FROM {TABLE} WHERE discount > 0.20",
            f"SELECT AVG(profit) FROM {TABLE} WHERE discount <= 0.10",
            f"SELECT AVG(discount)*100 FROM {TABLE}",
        ]
    },
    {
        "q": "What is causing loss-making in the Tables sub-category?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE sub_category='Tables'",
            f"SELECT AVG(discount)*100 FROM {TABLE} WHERE sub_category='Tables'",
            f"SELECT SUM(sales) FROM {TABLE} WHERE sub_category='Tables'",
        ]
    },
    {
        "q": "Why is Technology the most profitable category?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE category='Technology'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE category='Technology'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE category='Furniture'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE category='Office Supplies'",
        ]
    },
    {
        "q": "Why did Q4 always outperform other quarters in sales?",
        "db_verifiers": [
            f"SELECT SUM(sales) FROM {TABLE} WHERE EXTRACT(QUARTER FROM order_date)=4",
            f"SELECT SUM(sales) FROM {TABLE} WHERE EXTRACT(QUARTER FROM order_date)=1",
            f"SELECT SUM(sales) FROM {TABLE} WHERE EXTRACT(QUARTER FROM order_date)=2",
            f"SELECT SUM(sales) FROM {TABLE} WHERE EXTRACT(QUARTER FROM order_date)=3",
        ]
    },
    {
        "q": "What drove Consumer segment growth from 2014 to 2017?",
        "db_verifiers": [
            f"SELECT SUM(sales) FROM {TABLE} WHERE segment='Consumer' AND order_date BETWEEN '2014-01-01' AND '2014-12-31'",
            f"SELECT SUM(sales) FROM {TABLE} WHERE segment='Consumer' AND order_date BETWEEN '2017-01-01' AND '2017-12-31'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE segment='Consumer'",
        ]
    },
    {
        "q": "What caused the 2016 to 2017 profit margin improvement?",
        "db_verifiers": [
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        ]
    },
    {
        "q": "Why is the East region second in profitability?",
        "db_verifiers": [
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='East'",
            f"SELECT SUM(profit) FROM {TABLE} WHERE region='West'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE region='East'",
        ]
    },
    {
        "q": "Why does the Corporate segment have lower margin than Home Office?",
        "db_verifiers": [
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE segment='Corporate'",
            f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE segment='Home Office'",
            f"SELECT AVG(discount)*100 FROM {TABLE} WHERE segment='Corporate'",
            f"SELECT AVG(discount)*100 FROM {TABLE} WHERE segment='Home Office'",
        ]
    },
]

# 10 false-premise queries for AssumptionValidator
FALSE_PREMISE_QUERIES = [
    # "sales dropped" when sales actually grew
    {
        "q": "Why did sales drop from 2015 to 2016?",
        "false_metric": "sales", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
        "actual_sql_previous": f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'",
    },
    {
        "q": "Why did revenue decline in 2017 compared to 2016?",
        "false_metric": "sales", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        "actual_sql_previous": f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
    },
    {
        "q": "Why did orders fall in 2016?",
        "false_metric": "orders", "false_direction": "down",
        "actual_sql_current":  f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
        "actual_sql_previous": f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'",
    },
    # "profit decreased" when profit grew
    {
        "q": "Why did profit decrease from 2016 to 2017?",
        "false_metric": "profit", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        "actual_sql_previous": f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
    },
    {
        "q": "Why did profit fall in Q1 2016 compared to Q1 2015?",
        "false_metric": "profit", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-03-31'",
        "actual_sql_previous": f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-03-31'",
    },
    # "sales up, profit down" divergence when both moved same direction
    {
        "q": "Why did sales increase but profit decrease from 2016 to 2017?",
        "false_metric": "profit", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        "actual_sql_previous": f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
    },
    {
        "q": "Why did revenue grow but orders declined in 2016?",
        "false_metric": "orders", "false_direction": "down",
        "actual_sql_current":  f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'",
        "actual_sql_previous": f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'",
    },
    # "profit dropped" in specific period
    {
        "q": "Why did profit collapse in 2014?",
        "false_metric": "profit", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2014-01-01' AND '2014-12-31'",
        "actual_sql_previous": None,  # no prior year to compare, should be passed through
    },
    {
        "q": "Why did West region sales decline in 2017?",
        "false_metric": "sales", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(sales) FROM {TABLE} WHERE region='West' AND order_date BETWEEN '2017-01-01' AND '2017-12-31'",
        "actual_sql_previous": f"SELECT SUM(sales) FROM {TABLE} WHERE region='West' AND order_date BETWEEN '2016-01-01' AND '2016-12-31'",
    },
    {
        "q": "Why did Consumer segment profit shrink from 2015 to 2016?",
        "false_metric": "profit", "false_direction": "down",
        "actual_sql_current":  f"SELECT SUM(profit) FROM {TABLE} WHERE segment='Consumer' AND order_date BETWEEN '2016-01-01' AND '2016-12-31'",
        "actual_sql_previous": f"SELECT SUM(profit) FROM {TABLE} WHERE segment='Consumer' AND order_date BETWEEN '2015-01-01' AND '2015-12-31'",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Factual accuracy + hallucination rate + completeness
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_eval(queries, label="WITH PRE-QUERY"):
    """Run all queries through agent.run(), compute metrics."""
    print(f"\n{'='*80}")
    print(f"AGENT EVALUATION — {label}")
    print(f"{'='*80}")
    print(f"{'Q':<4} {'Factual Acc':>12} {'Halluc':>7} {'Complete':>9}  {'#Claims':>7}  Query")
    print("-"*80)

    results = []
    for item in queries:
        q            = item["q"]
        verifier_sqls = item["db_verifiers"]

        # Get DB ground-truth values for this query
        db_values = []
        for sql in verifier_sqls:
            v = _scalar(sql)
            if v is not None:
                db_values.append(v)
                # Also add percentage versions if dollar value
                if v > 100:
                    db_values.append(round(v / 1000, 1))   # thousands
                    db_values.append(round(v / 1000000, 1)) # millions

        # Run agent
        try:
            answer = agent.run(q)
        except Exception as e:
            answer = f"ERROR: {e}"

        # Hallucination check
        halluc = _is_hallucinated_round_number(answer)

        # Completeness check
        complete = check_completeness(answer)

        # Factual accuracy: extract numbers and verify each
        claims = extract_numbers(answer)
        if claims and db_values:
            supported = sum(1 for _, v in claims if verify_claim(None, v, db_values))
            acc = supported / len(claims)
        else:
            supported = 0
            acc = 1.0 if not claims else 0.0  # no claims = trivially accurate; claims with no DB = 0

        print(f"Q{len(results)+1:02d}  {acc:>12.3f} {'Y' if halluc else 'N':>7} {'Y' if complete else 'N':>9}  {len(claims):>7}  {q[:45]}")
        time.sleep(2.0)  # avoid rate limit

        results.append({
            "query": q, "answer": answer,
            "n_claims": len(claims), "n_supported": supported,
            "factual_acc": round(acc, 3),
            "hallucinated": halluc,
            "complete": complete,
            "db_values_used": len(db_values),
        })

    # Aggregate
    n = len(results)
    mean_acc   = sum(r["factual_acc"]  for r in results) / n
    fully_acc  = sum(1 for r in results if r["factual_acc"] == 1.0)
    halluc_n   = sum(1 for r in results if r["hallucinated"])
    complete_n = sum(1 for r in results if r["complete"])

    print(f"\nSUMMARY ({label}):")
    print(f"  Mean factual accuracy : {mean_acc:.3f}  ({mean_acc*100:.1f}%)")
    print(f"  Fully accurate        : {fully_acc} / {n}")
    print(f"  Hallucination rate    : {halluc_n}/{n}  ({halluc_n/n*100:.1f}%)")
    print(f"  Response completeness : {complete_n}/{n}  ({complete_n/n*100:.1f}%)")

    return results, {
        "label": label, "n": n,
        "mean_factual_acc_pct": round(mean_acc * 100, 1),
        "fully_accurate": fully_acc,
        "hallucination_count": halluc_n,
        "hallucination_rate_pct": round(halluc_n / n * 100, 1),
        "completeness_count": complete_n,
        "completeness_rate_pct": round(complete_n / n * 100, 1),
    }

print("\nRunning GROUNDED agent (with forced pre-query)...")
results_grounded, summary_grounded = run_agent_eval(AGENT_QUERIES, "WITH PRE-QUERY (grounded)")

# ── WITHOUT pre-query: monkey-patch to disable forced queries ─────────────────
print("\nRunning BASELINE agent (without forced pre-query)...")
import chatbot.agent.orchestrator as _orch_mod
_original_run_forced = _orch_mod.AgentOrchestrator._run_forced_queries

def _no_forced_queries(self, question):
    return []  # disable pre-query

_orch_mod.AgentOrchestrator._run_forced_queries = _no_forced_queries
results_baseline, summary_baseline = run_agent_eval(AGENT_QUERIES, "WITHOUT PRE-QUERY (baseline)")
_orch_mod.AgentOrchestrator._run_forced_queries = _original_run_forced  # restore

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: AssumptionValidator on false-premise queries
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'='*80}")
print("ASSUMPTION VALIDATOR — False-Premise Queries (n=10)")
print(f"{'='*80}")
print(f"{'Q':<4} {'Detected':>8} {'Corrected':>10} {'Actual Dir':>11}  Query")
print("-"*80)

av_results = []
FALSE_PREMISE_TYPES = {
    "sales_drop":      [],
    "profit_drop":     [],
    "divergence":      [],
}

for i, item in enumerate(FALSE_PREMISE_QUERIES, 1):
    q             = item["q"]
    false_metric  = item["false_metric"]
    false_dir     = item["false_direction"]

    # Get actual direction from DB
    cur_sql  = item["actual_sql_current"]
    prev_sql = item["actual_sql_previous"]
    cur_val  = _scalar(cur_sql) if cur_sql else None
    prev_val = _scalar(prev_sql) if prev_sql else None

    if cur_val is not None and prev_val is not None:
        actual_direction = "up" if cur_val > prev_val else "down"
        premise_is_false = (actual_direction != false_dir)
    else:
        actual_direction = "unknown"
        premise_is_false = False

    # Simulate forced queries result string (simplified)
    fake_tool_results = []
    if cur_val and prev_val:
        chg = (cur_val - prev_val) / abs(prev_val) * 100 if prev_val else 0
        fake_tool_results.append(
            f"{false_metric} comparison:\n"
            f"  Current  (...): ${cur_val:,.0f}\n"
            f"  Previous (...): ${prev_val:,.0f}\n"
            f"  Change: {chg:+.1f}% (${abs(cur_val-prev_val):,.0f})"
        )

    # Run AssumptionValidator
    correction = validate_assumptions(q, fake_tool_results) if fake_tool_results else None

    detected  = correction is not None
    # "Corrected" = detected AND the correction text mentions the actual direction
    corrected = False
    if detected and correction:
        actual_word = "increased" if actual_direction == "up" else "decreased"
        corrected = actual_word in correction.lower() or "actually" in correction.lower()

    print(f"Q{i:02d}  {'✓' if detected else '✗':>8}  {'✓' if corrected else '✗':>10}  {actual_direction:>11}  {q[:45]}")
    time.sleep(0.2)

    ptype = "divergence" if "but" in q.lower() or "despite" in q.lower() else \
            ("sales_drop" if false_metric == "sales" or false_metric == "orders" else "profit_drop")

    av_results.append({
        "id": i, "query": q,
        "false_metric": false_metric, "false_direction": false_dir,
        "actual_direction": actual_direction,
        "premise_is_false": premise_is_false,
        "detected": detected,
        "corrected": corrected,
        "type": ptype,
    })

# Aggregate by type
print("\nASSUMPTION VALIDATOR SUMMARY:")
types_map = {
    "sales_drop":  "Sales direction inverted",
    "profit_drop": "Profit direction inverted",
    "divergence":  "Divergence claim",
}
av_by_type = {}
for ptype, label in types_map.items():
    subset = [r for r in av_results if r["type"] == ptype]
    n = len(subset)
    if n == 0: continue
    det = sum(1 for r in subset if r["detected"])
    cor = sum(1 for r in subset if r["corrected"])
    mis = n - det
    print(f"  {label}: n={n}  detected={det/n*100:.1f}%  corrected={cor/n*100:.1f}%  missed={mis/n*100:.1f}%")
    av_by_type[ptype] = {"n": n, "detected_pct": round(det/n*100, 1),
                         "corrected_pct": round(cor/n*100, 1), "missed_pct": round(mis/n*100, 1)}

av_n   = len(av_results)
av_det = sum(1 for r in av_results if r["detected"])
av_cor = sum(1 for r in av_results if r["corrected"])
av_mis = av_n - av_det
print(f"\n  OVERALL: n={av_n}  detected={av_det/av_n*100:.1f}%  corrected={av_cor/av_n*100:.1f}%  missed={av_mis/av_n*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────

output = {
    "factual_accuracy": {
        "grounded":   {**summary_grounded,  "per_query": results_grounded},
        "baseline":   {**summary_baseline,  "per_query": results_baseline},
        "delta": {
            "factual_acc_pp":     round(summary_grounded["mean_factual_acc_pct"] - summary_baseline["mean_factual_acc_pct"], 1),
            "hallucination_pp":   round(summary_grounded["hallucination_rate_pct"] - summary_baseline["hallucination_rate_pct"], 1),
            "completeness_pp":    round(summary_grounded["completeness_rate_pct"] - summary_baseline["completeness_rate_pct"], 1),
        }
    },
    "assumption_validator": {
        "n": av_n,
        "detected": av_det, "detected_pct": round(av_det/av_n*100, 1),
        "corrected": av_cor, "corrected_pct": round(av_cor/av_n*100, 1),
        "missed": av_mis, "missed_pct": round(av_mis/av_n*100, 1),
        "by_type": av_by_type,
        "per_query": av_results,
    }
}

with open("agent_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

with open("agent_results.txt", "w", encoding="utf-8") as f:
    f.write("AGENT EVALUATION RESULTS — Section 5.6\n")
    f.write("="*60 + "\n\n")

    f.write("TABLE 5.12 — Factual Accuracy & Hallucination\n")
    g = summary_grounded; b = summary_baseline
    f.write(f"  Factual accuracy:     grounded={g['mean_factual_acc_pct']}%  baseline={b['mean_factual_acc_pct']}%  delta=+{g['mean_factual_acc_pct']-b['mean_factual_acc_pct']:.1f}pp\n")
    f.write(f"  Fully accurate:       grounded={g['fully_accurate']}/20  baseline={b['fully_accurate']}/20\n")
    f.write(f"  Hallucination rate:   grounded={g['hallucination_rate_pct']}%  baseline={b['hallucination_rate_pct']}%  delta={g['hallucination_rate_pct']-b['hallucination_rate_pct']:.1f}pp\n")
    f.write(f"  Completeness:         grounded={g['completeness_rate_pct']}%  baseline={b['completeness_rate_pct']}%  delta=+{g['completeness_rate_pct']-b['completeness_rate_pct']:.1f}pp\n\n")

    f.write("TABLE 5.13 — AssumptionValidator\n")
    for ptype, s in av_by_type.items():
        f.write(f"  {types_map.get(ptype,ptype)}: n={s['n']}  detected={s['detected_pct']}%  corrected={s['corrected_pct']}%  missed={s['missed_pct']}%\n")
    f.write(f"  OVERALL: n={av_n}  detected={av_det/av_n*100:.1f}%  corrected={av_cor/av_n*100:.1f}%  missed={av_mis/av_n*100:.1f}%\n")

print("\nSaved → agent_results.txt  agent_results.json")
print("="*60)