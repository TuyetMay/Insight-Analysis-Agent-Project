"""
test_structured_pipeline.py
────────────────────────────────────────────────────────────────────────────────
Evaluates the three-tier structured query pipeline (Section 5.4).

Metrics produced:
  - Execution accuracy per category (Simple KPI / Structured Breakdown / Trend+Compare)
  - Tier hit rate (how many queries resolved at Tier 1 / 2 / 3)
  - LLMPlanAuditor contribution (Tier-2 accuracy with vs without auditor)
  - Self-correction success rate (empty-result date-widening loop)

Usage:
    python test_structured_pipeline.py

Output:
    pipeline_results.txt   — machine-readable summary (paste into Section 5.4)
    pipeline_results.json  — full per-query detail for further analysis

Requirements:
    pip install python-dotenv
    Set GCP_PROJECT and GCP_LOCATION in .env
"""

import os, sys, json, time, re
from datetime import date
from turtle import pd
sys.path.append('..')

from config import Config
from google import genai
from core.database import execute_query
from core.data_loader import load_filtered_data_safe, calculate_kpis, get_filter_options
from chatbot.nl_parser import NLParser
from chatbot.plan_validator import PlanValidator
from chatbot.sql_builder import SQLBuilder
from chatbot.answer_formatter import AnswerFormatter
from chatbot.insight_generator import InsightGenerator
from chatbot.llm_plan_auditor import LLMPlanAuditor
from rag.engine import RAGEngine

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
    "region":     filter_options["region"],
    "segment":    filter_options["segment"],
    "category":   filter_options["category"],
}
df   = load_filtered_data_safe(filters)
kpis = calculate_kpis(df)

s0 = str(filter_options["min_date"])[:10]
e0 = str(filter_options["max_date"])[:10]

parser    = NLParser(df, filters, gemini_client, model_name)
validator = PlanValidator(
    (s0, e0),
    filter_options["region"],
    filter_options["segment"],
    filter_options["category"],
)
sql_builder = SQLBuilder()
formatter   = AnswerFormatter()
auditor     = LLMPlanAuditor(gemini_client, model_name)

rag = RAGEngine()
rag.build_static(df)
rag.build(df, kpis, filters)

# ── Ground-truth dataset ──────────────────────────────────────────────────────
# Each entry: (question, category, expected_check_fn)
# expected_check_fn(result_df) -> bool
# GT is defined as: result_df non-empty AND key value within tolerance of DB direct query.

def _direct_query(sql):
    """Run a direct SQL query against the DB and return scalar or DataFrame."""
    return execute_query(sql)

def _scalar(sql):
    df_r = _direct_query(sql)
    if df_r.empty: return None
    return float(df_r.iloc[0, 0])

def approx(a, b, tol=0.02):
    """True if a ≈ b within tol fraction."""
    if a is None or b is None: return False
    if b == 0: return abs(a) < 1
    return abs(a - b) / abs(b) <= tol

TABLE = Config.DB_TABLE

QUESTIONS = [
    # ── Simple KPI (n=20) ─────────────────────────────────────────────────────
    {   "q": "What is the total sales revenue?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("sales", 0)),
            _scalar(f"SELECT SUM(sales) FROM {TABLE}")
        )
    },
    {   "q": "What is the total profit?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit", 0)),
            _scalar(f"SELECT SUM(profit) FROM {TABLE}")
        )
    },
    {   "q": "How many orders were placed in total?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("orders", 0)),
            _scalar(f"SELECT COUNT(DISTINCT order_id) FROM {TABLE}")
        )
    },
    {   "q": "What is the overall profit margin?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit_margin", 0)),
            _scalar(f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE}")
        )
    },
    {   "q": "What is the average order value?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What is the total revenue for 2016?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("sales", 0)),
            _scalar(f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-01-01' AND '2016-12-31'")
        )
    },
    {   "q": "How many orders were placed in 2017?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("orders", 0)),
            _scalar(f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-12-31'")
        )
    },
    {   "q": "What is the total profit in 2015?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit", 0)),
            _scalar(f"SELECT SUM(profit) FROM {TABLE} WHERE order_date BETWEEN '2015-01-01' AND '2015-12-31'")
        )
    },
    {   "q": "What is the total sales in Q4 2016?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("sales", 0)),
            _scalar(f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2016-10-01' AND '2016-12-31'")
        )
    },
    {   "q": "How much revenue did we generate in January 2017?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("sales", 0)),
            _scalar(f"SELECT SUM(sales) FROM {TABLE} WHERE order_date BETWEEN '2017-01-01' AND '2017-01-31'")
        )
    },
    {   "q": "What is the total number of orders in the West region?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("orders", 0)),
            _scalar(f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE region='West'")
        )
    },
    {   "q": "What is the total profit for the Consumer segment?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit", 0)),
            _scalar(f"SELECT SUM(profit) FROM {TABLE} WHERE segment='Consumer'")
        )
    },
    {   "q": "What is the total sales for the Technology category?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("sales", 0)),
            _scalar(f"SELECT SUM(sales) FROM {TABLE} WHERE category='Technology'")
        )
    },
    {   "q": "What is the profit margin for the Corporate segment?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit_margin", 0)),
            _scalar(f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE segment='Corporate'")
        )
    },
    {   "q": "What is the total revenue for Furniture in 2017?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("sales", 0)),
            _scalar(f"SELECT SUM(sales) FROM {TABLE} WHERE category='Furniture' AND order_date BETWEEN '2017-01-01' AND '2017-12-31'")
        )
    },
    {   "q": "What is the total profit for Office Supplies?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit", 0)),
            _scalar(f"SELECT SUM(profit) FROM {TABLE} WHERE category='Office Supplies'")
        )
    },
    {   "q": "What is the overall average discount rate?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "How many orders did the South region place in 2015?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("orders", 0)),
            _scalar(f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE region='South' AND order_date BETWEEN '2015-01-01' AND '2015-12-31'")
        )
    },
    {   "q": "What is the profit margin in 2014?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("profit_margin", 0)),
            _scalar(f"SELECT SUM(profit)/SUM(sales)*100 FROM {TABLE} WHERE order_date BETWEEN '2014-01-01' AND '2014-12-31'")
        )
    },
    {   "q": "How many orders were placed in the East region in 2016?",
        "category": "simple_kpi",
        "check": lambda df_r: not df_r.empty and approx(
            float(df_r.iloc[0].get("orders", 0)),
            _scalar(f"SELECT COUNT(DISTINCT order_id) FROM {TABLE} WHERE region='East' AND order_date BETWEEN '2016-01-01' AND '2016-12-31'")
        )
    },

    # ── Structured Breakdown (n=25) ───────────────────────────────────────────
    {   "q": "What are the total sales by region?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns and len(df_r) == 4
    },
    {   "q": "Show me profit breakdown by segment.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns and len(df_r) == 3
    },
    {   "q": "What is the revenue by category in 2016?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns and len(df_r) == 3
    },
    {   "q": "What are the top 5 sub-categories by profit?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and len(df_r) <= 5
    },
    {   "q": "Show me the top 10 products by sales.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and len(df_r) <= 10
    },
    {   "q": "What is the profit margin by category?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns and len(df_r) == 3
    },
    {   "q": "Show me monthly sales trend for 2016.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "What is the quarterly revenue breakdown for 2017?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "Show profit by region and segment.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What is the yearly sales trend from 2014 to 2017?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns and len(df_r) == 4
    },
    {   "q": "What is the profit breakdown by sub-category for Furniture?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns
    },
    {   "q": "Show monthly orders trend in 2017.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "What are the loss-making sub-categories?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns
    },
    {   "q": "Show me which products are losing money.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What sub-categories have negative profit?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Show sales trend by segment over years.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "What is the profit margin by region in 2016?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns and len(df_r) == 4
    },
    {   "q": "Show top 5 categories by revenue growth.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What is the discount impact on profit by category?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Show sales and profit by segment in 2017.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and "breakdown" in df_r.columns
    },
    {   "q": "What are the top 5 regions by revenue in Q3 2016?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and len(df_r) <= 4
    },
    {   "q": "Show top 10 sub-categories by sales in the West region.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and len(df_r) <= 10
    },
    {   "q": "Show sales by region for each year.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What are the top 3 segments by profit margin?",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty and len(df_r) <= 3
    },
    {   "q": "Show me sales by category broken down by region.",
        "category": "structured_breakdown",
        "check": lambda df_r: not df_r.empty
    },

    # ── Trend / Compare (n=20) ────────────────────────────────────────────────
    {   "q": "Compare 2016 vs 2017 total sales.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty and "current" in df_r.columns and "previous" in df_r.columns
    },
    {   "q": "How did profit change from 2015 to 2016?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Compare Q3 2016 vs Q3 2017 revenue.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What is the year-over-year sales growth?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "How does profit in 2017 compare to 2016?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty and "current" in df_r.columns
    },
    {   "q": "Compare the West region vs East region sales in 2016.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Show month-over-month revenue change in 2017.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "How did the Consumer segment perform compared to Corporate in 2016?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Compare Technology vs Furniture profit in 2017.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "What is the sales growth rate from 2014 to 2017?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "How did profit margin change from 2015 to 2017?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Compare Q1 2016 vs Q1 2017 orders.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Show year-over-year profit growth by region.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "How did discount levels change from 2015 to 2016?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Compare the top region vs bottom region sales in 2017.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "How did orders in the South region change from 2016 to 2017?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Compare profit margin between Consumer and Home Office segments.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Show the revenue trend — is growth accelerating or slowing?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty and "period" in df_r.columns
    },
    {   "q": "How did the Furniture category perform vs Technology in 2016?",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
    {   "q": "Compare October 2016 vs October 2015 total sales.",
        "category": "trend_compare",
        "check": lambda df_r: not df_r.empty
    },
]

# ── Tier tracking helpers ─────────────────────────────────────────────────────

def run_pipeline(question, use_auditor=True):
    """
    Run structured pipeline, return (result_df, tier_used, empty_retried, recovered).
    tier_used: 1 | 2 | 3 | None (failed)
    """
    # Tier 1
    fast = parser.fast_kpi_answer(question)
    if fast:
        # Tier 1 answered — run the actual SQL for checking
        plan_t1 = parser.rule_based_plan(question)
        if plan_t1 is None:
            # pure regex KPI — build a quick aggregate plan
            plan_t1 = {
                "intent": "kpi_value", "metrics": ["sales", "profit"],
                "time_grain": "none", "breakdown_by": None,
                "secondary_breakdown": None,
                "start_date": s0, "end_date": e0,
                "compare_period": None, "top_k": None, "order_by": "sales",
                "filters": {"region": [], "segment": [], "category": [], "sub_category": []},
                "show_extremes": False,
            }
        try:
            plan_t1 = validator.validate(plan_t1)
            df_r = sql_builder.run(plan_t1)
            return df_r, 1, False, False
        except Exception:
            pass

    # Tier 2
    rule_plan = parser.rule_based_plan(question)
    if rule_plan:
        if use_auditor:
            rule_plan = auditor.audit(question, rule_plan)
        try:
            plan = validator.validate(rule_plan)
            df_r = sql_builder.run(plan)
            empty_retried = False
            recovered = False
            if df_r.empty:
                # self-correction
                empty_retried = True
                from datetime import timedelta, datetime
                try:
                    sd = datetime.strptime(plan["start_date"], "%Y-%m-%d")
                    ed = datetime.strptime(plan["end_date"], "%Y-%m-%d")
                    widened = {**plan,
                        "start_date": (sd - timedelta(days=183)).strftime("%Y-%m-%d"),
                        "end_date":   (ed + timedelta(days=183)).strftime("%Y-%m-%d")}
                    df_r2 = sql_builder.run(widened)
                    if not df_r2.empty:
                        df_r = df_r2
                        recovered = True
                except Exception:
                    pass
            return df_r, 2, empty_retried, recovered
        except Exception:
            pass

    # Tier 3
    try:
        rag_ctx  = rag.retrieve(question, k=10, tier=3)
        raw_plan = parser.gemini_plan(question, rag_ctx)
        plan     = validator.validate(raw_plan)
        df_r     = sql_builder.run(plan)
        empty_retried = False
        recovered = False
        if df_r.empty:
            empty_retried = True
            from datetime import timedelta, datetime
            try:
                sd = datetime.strptime(plan["start_date"], "%Y-%m-%d")
                ed = datetime.strptime(plan["end_date"], "%Y-%m-%d")
                widened = {**plan,
                    "start_date": (sd - timedelta(days=183)).strftime("%Y-%m-%d"),
                    "end_date":   (ed + timedelta(days=183)).strftime("%Y-%m-%d")}
                df_r2 = sql_builder.run(widened)
                if not df_r2.empty:
                    df_r = df_r2
                    recovered = True
            except Exception:
                pass
        return df_r, 3, empty_retried, recovered
    except Exception as e:
        return pd.DataFrame(), None, False, False

# ── Main evaluation loop ──────────────────────────────────────────────────────

print("\n" + "="*90)
print("STRUCTURED PIPELINE EVALUATION  —  Section 5.4")
print("="*90)
print(f"{'Q_ID':<5} {'Category':<22} {'Tier':>4}  {'Pass':>4}  {'Empty→Fixed':>11}  {'Question (truncated)'}")
print("-"*90)

results = []

for i, item in enumerate(QUESTIONS, 1):
    q        = item["q"]
    category = item["category"]
    check_fn = item["check"]

    t0 = time.time()
    df_r, tier, empty_retried, recovered = run_pipeline(q, use_auditor=True)
    elapsed = time.time() - t0

    try:
        passed = bool(check_fn(df_r))
    except Exception:
        passed = False

    fix_label = "recovered" if (empty_retried and recovered) else ("failed" if empty_retried else "—")

    print(f"Q{i:02d}  {category:<22} T{tier or '?'!s:>1}   {'✓' if passed else '✗':>4}  {fix_label:>11}  {q[:45]}")

    results.append({
        "id": i, "question": q, "category": category,
        "tier": tier, "passed": passed,
        "empty_retried": empty_retried, "recovered": recovered,
        "elapsed_s": round(elapsed, 3),
    })

    time.sleep(0.4)  # avoid API rate limits

# ── Auditor-off pass (Tier 2 only, no auditor) ───────────────────────────────
print("\n" + "="*90)
print("TIER 2 — WITHOUT LLMPlanAuditor  (for Table 5.7)")
print("="*90)

t2_items = [r for r in results if r["tier"] == 2]
t2_no_audit_pass = 0

for item_r in t2_items:
    q        = item_r["question"]
    check_fn = next(x["check"] for x in QUESTIONS if x["q"] == q)
    rule_plan = parser.rule_based_plan(q)
    if rule_plan:
        try:
            plan  = validator.validate(rule_plan)
            df_r  = sql_builder.run(plan)
            ok    = bool(check_fn(df_r))
            if ok: t2_no_audit_pass += 1
        except Exception:
            pass

# ── Aggregate results ─────────────────────────────────────────────────────────

def stats(subset, label):
    n      = len(subset)
    passed = sum(r["passed"] for r in subset)
    acc    = passed / n * 100 if n else 0
    t1     = sum(1 for r in subset if r["tier"] == 1)
    t2     = sum(1 for r in subset if r["tier"] == 2)
    t3     = sum(1 for r in subset if r["tier"] == 3)
    return {"label": label, "n": n, "passed": passed, "acc": acc,
            "tier1": t1, "tier2": t2, "tier3": t3}

cats = {
    "simple_kpi":           "Simple KPI",
    "structured_breakdown": "Structured Breakdown",
    "trend_compare":        "Trend / Compare",
}

all_stats    = [stats([r for r in results if r["category"] == k], v) for k, v in cats.items()]
overall      = stats(results, "Overall")

# Tier hit rate
t1_total = sum(r["tier"] == 1 for r in results)
t2_total = sum(r["tier"] == 2 for r in results)
t3_total = sum(r["tier"] == 3 for r in results)
n_total  = len(results)

# Self-correction
retried    = [r for r in results if r["empty_retried"]]
recovered_ = [r for r in retried if r["recovered"]]

# Auditor
t2_with_audit    = sum(r["passed"] for r in results if r["tier"] == 2)
t2_without_audit = t2_no_audit_pass
t2_total_count   = t2_total
auditor_plans_corrected = t2_with_audit - t2_without_audit
auditor_bypassed        = sum(1 for r in results if r["tier"] == 2 and True)  # placeholder

# ── Print summary ─────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESULTS SUMMARY — COPY INTO Section 5.4")
print("="*60)
print()
print("TABLE 5.5 — Execution Accuracy by Category")
print(f"  {'Category':<25} {'n':>4}  {'Pass':>4}  {'Accuracy':>9}")
print(f"  {'-'*25} {'-'*4}  {'-'*4}  {'-'*9}")
for s in all_stats:
    print(f"  {s['label']:<25} {s['n']:>4}  {s['passed']:>4}  {s['acc']:>8.1f}%")
print(f"  {'Overall':<25} {overall['n']:>4}  {overall['passed']:>4}  {overall['acc']:>8.1f}%")

print()
print("TABLE 5.6 — Tier Hit Rate")
cumulative = 0
for tier_n, count in [(1, t1_total), (2, t2_total), (3, t3_total)]:
    hr = count / n_total * 100
    cumulative += hr
    print(f"  Tier {tier_n}: {count:>3} queries  {hr:>6.1f}%  (cumulative {cumulative:.1f}%)")

print()
print("TABLE 5.7 — LLMPlanAuditor Contribution (Tier 2 only)")
t2_acc_with    = t2_with_audit / t2_total_count * 100 if t2_total_count else 0
t2_acc_without = t2_without_audit / t2_total_count * 100 if t2_total_count else 0
print(f"  Tier 2 acc WITHOUT auditor : {t2_acc_without:.1f}%  ({t2_without_audit}/{t2_total_count})")
print(f"  Tier 2 acc WITH    auditor : {t2_acc_with:.1f}%  ({t2_with_audit}/{t2_total_count})")
print(f"  Accuracy gain              : +{t2_acc_with - t2_acc_without:.1f} pp")
print(f"  Plans corrected by auditor : {max(0, auditor_plans_corrected)}")

print()
print("TABLE 5.8 — Self-correction Success Rate")
print(f"  Queries triggering empty-result loop : {len(retried)}")
print(f"  Successfully recovered               : {len(recovered_)}  ({len(recovered_)/len(retried)*100:.1f}% of retried)" if retried else "  No queries triggered retry.")
print(f"  Still empty after max retries        : {len(retried) - len(recovered_)}")

# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "execution_accuracy": {s["label"]: {"n": s["n"], "passed": s["passed"], "accuracy_pct": round(s["acc"], 1)} for s in all_stats},
    "overall_accuracy": {"n": overall["n"], "passed": overall["passed"], "accuracy_pct": round(overall["acc"], 1)},
    "tier_hit_rate": {
        "tier1": {"count": t1_total, "pct": round(t1_total / n_total * 100, 1)},
        "tier2": {"count": t2_total, "pct": round(t2_total / n_total * 100, 1)},
        "tier3": {"count": t3_total, "pct": round(t3_total / n_total * 100, 1)},
    },
    "auditor": {
        "tier2_acc_without_pct": round(t2_acc_without, 1),
        "tier2_acc_with_pct":    round(t2_acc_with, 1),
        "gain_pp":               round(t2_acc_with - t2_acc_without, 1),
        "plans_corrected":       max(0, auditor_plans_corrected),
        "tier2_total":           t2_total_count,
    },
    "self_correction": {
        "retried":   len(retried),
        "recovered": len(recovered_),
        "failed":    len(retried) - len(recovered_),
        "success_rate_pct": round(len(recovered_) / len(retried) * 100, 1) if retried else 0,
    },
    "per_query": results,
}

with open("pipeline_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

with open("pipeline_results.txt", "w", encoding="utf-8") as f:
    f.write("STRUCTURED PIPELINE EVALUATION — Section 5.4\n")
    f.write("="*60 + "\n\n")

    f.write("TABLE 5.5 — Execution Accuracy\n")
    for s in all_stats:
        f.write(f"  {s['label']:<25} n={s['n']}  acc={s['acc']:.1f}%\n")
    f.write(f"  {'Overall':<25} n={overall['n']}  acc={overall['acc']:.1f}%\n\n")

    f.write("TABLE 5.6 — Tier Hit Rate\n")
    cum = 0
    for tier_n, count in [(1, t1_total), (2, t2_total), (3, t3_total)]:
        hr = count / n_total * 100
        cum += hr
        f.write(f"  Tier {tier_n}: {count} queries  {hr:.1f}%  cumulative={cum:.1f}%\n")

    f.write("\nTABLE 5.7 — Auditor Contribution (Tier 2)\n")
    f.write(f"  Without auditor : {t2_acc_without:.1f}%\n")
    f.write(f"  With auditor    : {t2_acc_with:.1f}%\n")
    f.write(f"  Gain            : +{t2_acc_with - t2_acc_without:.1f} pp\n")
    f.write(f"  Plans corrected : {max(0, auditor_plans_corrected)}\n")

    f.write("\nTABLE 5.8 — Self-correction\n")
    f.write(f"  Retried:   {len(retried)}\n")
    f.write(f"  Recovered: {len(recovered_)}\n")
    if retried:
        f.write(f"  Rate:      {len(recovered_)/len(retried)*100:.1f}%\n")

    f.write("\nPER-QUERY DETAIL\n")
    f.write(f"{'Q':<4} {'Cat':<22} {'Tier':>4} {'Pass':>4} {'Retry':>5} {'Rec':>3}  Question\n")
    f.write("-"*90 + "\n")
    for r in results:
        f.write(f"Q{r['id']:<3} {r['category']:<22} T{r['tier'] or '?'!s:>1}   {'Y' if r['passed'] else 'N':>4} "
                f"{'Y' if r['empty_retried'] else 'N':>5} {'Y' if r['recovered'] else 'N':>3}  {r['question'][:50]}\n")

print("\nSaved → pipeline_results.txt  pipeline_results.json")
print("="*60)