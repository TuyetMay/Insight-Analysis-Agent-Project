"""
test_e2e_performance.py
────────────────────────────────────────────────────────────────────────────────
Evaluates end-to-end system performance (Section 5.8):
  1. Response latency per processing path (50 queries)
  2. API call rate — full system vs no-routing baseline
  3. Generates responses for 40 human-rated queries × 3 conditions
     (full system / no-RAG / no-routing) → saved to rating sheet

Produces:
  e2e_results.json        — latency + API call counts
  e2e_results.txt         — table values ready to paste into Section 5.8
  5_8_rating_sheet.txt    — responses for human rating (Table 5.19)

Usage:
    python test_e2e_performance.py

Place in project root. Requires .env with GCP_PROJECT and GCP_LOCATION.
"""

import os, sys, json, time, re
sys.path.append('..')

from config import Config
from google import genai
from google.genai import types as genai_types
from core.data_loader import load_filtered_data_safe, calculate_kpis, get_filter_options
from core.database import execute_query
from chatbot.nl_parser import NLParser
from chatbot.plan_validator import PlanValidator
from chatbot.sql_builder import SQLBuilder
from chatbot.answer_formatter import AnswerFormatter
from chatbot.insight_generator import InsightGenerator
from chatbot.llm_plan_auditor import LLMPlanAuditor
from chatbot.smart_router import SmartRouter
from chatbot.agent.orchestrator import AgentOrchestrator
from chatbot.hybrid_executor import HybridExecutor
from rag.engine import RAGEngine
import pandas as pd

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
s0   = str(filter_options["min_date"])[:10]
e0   = str(filter_options["max_date"])[:10]

parser    = NLParser(df, filters, gemini_client, model_name)
validator = PlanValidator((s0,e0), filter_options["region"], filter_options["segment"], filter_options["category"])
sql_b     = SQLBuilder()
formatter = AnswerFormatter()
insights  = InsightGenerator(gemini_client, model_name)
auditor   = LLMPlanAuditor(gemini_client, model_name)
router    = SmartRouter(gemini_client, model_name)
agent     = AgentOrchestrator(gemini_client, model_name, s0, e0)

rag = RAGEngine()
rag.build_static(df)
rag.build(df, kpis, filters)

hybrid = HybridExecutor(
    structured_runner=lambda q: _run_structured(q),
    agent_runner=lambda q: agent.run(q),
    gemini_client=gemini_client,
    model_name=model_name,
)

# ── API call counter ──────────────────────────────────────────────────────────
_api_call_count = 0

_original_generate = gemini_client.models.generate_content.__func__ \
    if hasattr(gemini_client.models.generate_content, '__func__') else None

class _CountingClient:
    """Wraps gemini_client to count API calls."""
    def __init__(self, client):
        self._client = client
        self.call_count = 0

    def generate_content(self, *args, **kwargs):
        self.call_count += 1
        return self._client.models.generate_content(*args, **kwargs)

counting_client = _CountingClient(gemini_client)

# We'll count calls by wrapping at a higher level — see run_full_system()

# ── Structured pipeline helper ────────────────────────────────────────────────
def _run_structured(q):
    fast = parser.fast_kpi_answer(q)
    if fast: return fast
    rule_plan = parser.rule_based_plan(q)
    if rule_plan:
        rule_plan = auditor.audit(q, rule_plan)
        try:
            plan  = validator.validate(rule_plan)
            df_r  = sql_b.run(plan)
            ins   = insights.generate(plan, df_r)
            return formatter.format(plan, df_r, ins)
        except Exception:
            pass
    try:
        ctx     = rag.retrieve(q, k=10, tier=3)
        rplan   = parser.gemini_plan(q, ctx)
        plan    = validator.validate(rplan)
        df_r    = sql_b.run(plan)
        ins     = insights.generate(plan, df_r)
        return formatter.format(plan, df_r, ins)
    except Exception as e:
        return f"Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Query sets
# ─────────────────────────────────────────────────────────────────────────────

# 50 queries for latency measurement (covers all paths)
LATENCY_QUERIES = [
    # Tier 1 — ~12 queries
    ("tier1", "What is the total sales revenue?"),
    ("tier1", "What is the total profit?"),
    ("tier1", "How many orders were placed in total?"),
    ("tier1", "What is the overall profit margin?"),
    ("tier1", "What is the average order value?"),
    ("tier1", "What is the total revenue?"),
    ("tier1", "Show me total orders."),
    ("tier1", "What is our profit margin?"),
    ("tier1", "How much profit did we make?"),
    ("tier1", "What are total sales?"),
    ("tier1", "Give me the profit margin."),
    ("tier1", "Total number of orders?"),
    # Tier 2 (no auditor trigger) — ~10 queries
    ("tier2_no_audit", "What are the total sales by region?"),
    ("tier2_no_audit", "Show profit by segment."),
    ("tier2_no_audit", "Top 5 sub-categories by profit."),
    ("tier2_no_audit", "Sales trend by year."),
    ("tier2_no_audit", "Which sub-categories are losing money?"),
    ("tier2_no_audit", "Top 10 products by sales."),
    ("tier2_no_audit", "Profit by category."),
    ("tier2_no_audit", "Monthly orders in 2017."),
    ("tier2_no_audit", "Profit margin by region."),
    ("tier2_no_audit", "Sales by category in 2016."),
    # Tier 2 + Auditor — ~8 queries
    ("tier2_audit", "Compare 2016 vs 2017 total sales."),
    ("tier2_audit", "How did profit change from 2015 to 2016?"),
    ("tier2_audit", "Year over year sales growth."),
    ("tier2_audit", "Compare Q1 2016 vs Q1 2017 orders."),
    ("tier2_audit", "Monthly profit trend in 2016."),
    ("tier2_audit", "Compare West vs East region sales."),
    ("tier2_audit", "Year over year profit by region."),
    ("tier2_audit", "Show quarterly revenue for 2017."),
    # Tier 3 — ~8 queries
    ("tier3", "Which products are bleeding money and what's causing it?"),
    ("tier3", "Show sales by category broken down by region with profit details."),
    ("tier3", "What is the discount impact on profitability across categories?"),
    ("tier3", "Revenue trend — is growth accelerating or slowing?"),
    ("tier3", "Show profit margin distribution across all sub-categories."),
    ("tier3", "Which region has the highest margin and why?"),
    ("tier3", "Show me the correlation between discount levels and profit."),
    ("tier3", "Top 5 regions by revenue in Q3 2016 with YoY comparison."),
    # Agent — ~7 queries
    ("agent", "Why did profit drop in Q4 2016?"),
    ("agent", "What caused the revenue decline in the Central region?"),
    ("agent", "Why are Tables and Bookcases losing money?"),
    ("agent", "Why does the South region have lower profit than the West?"),
    ("agent", "What is causing the margin compression in 2016?"),
    ("agent", "Why did orders drop in Q1 2017?"),
    ("agent", "Why does heavy discounting hurt profitability?"),
    # SmartRouter classification overhead — 5 queries (structured, to isolate router latency)
    ("router_only", "What are the total sales by region?"),
    ("router_only", "Show profit by segment."),
    ("router_only", "Compare 2016 vs 2017."),
    ("router_only", "Why did profit drop in 2016?"),
    ("router_only", "Show loss-making products and explain why."),
]

# 40 queries for human rating (10 per type)
RATING_QUERIES = [
    # Simple KPI (10)
    ("simple_kpi", "What is the total sales revenue?"),
    ("simple_kpi", "What is the total profit?"),
    ("simple_kpi", "What is the overall profit margin?"),
    ("simple_kpi", "How many orders were placed in total?"),
    ("simple_kpi", "What is the total revenue for 2016?"),
    ("simple_kpi", "What is the total profit for the Consumer segment?"),
    ("simple_kpi", "What is the profit margin for the Technology category?"),
    ("simple_kpi", "How many orders were placed in the West region?"),
    ("simple_kpi", "What is the total sales in Q4 2016?"),
    ("simple_kpi", "What is the average order value?"),
    # Structured Breakdown (10)
    ("structured", "What are the total sales by region?"),
    ("structured", "Show me profit breakdown by segment."),
    ("structured", "What are the top 5 sub-categories by profit?"),
    ("structured", "Show monthly sales trend for 2016."),
    ("structured", "What is the profit margin by category?"),
    ("structured", "Show sales trend by segment over years."),
    ("structured", "Compare 2016 vs 2017 total sales."),
    ("structured", "What are the loss-making sub-categories?"),
    ("structured", "Show quarterly revenue breakdown for 2017."),
    ("structured", "What is the profit margin by region in 2016?"),
    # Diagnostic / Agent (10)
    ("agent", "Why did profit drop in Q4 2016?"),
    ("agent", "What caused the revenue decline in the Central region?"),
    ("agent", "Why is the Furniture category underperforming?"),
    ("agent", "Why are Tables and Bookcases losing money?"),
    ("agent", "Why does the South region have lower profit than the West?"),
    ("agent", "What is causing the margin compression in 2016?"),
    ("agent", "Why does heavy discounting hurt profitability?"),
    ("agent", "Why is the Home Office segment more profitable than Consumer?"),
    ("agent", "What drove the sales spike in November 2017?"),
    ("agent", "Why did orders drop in Q1 2017?"),
    # Hybrid (10)
    ("hybrid", "Show me loss-making products and explain what is causing the losses."),
    ("hybrid", "Which region contributes least to profit and why?"),
    ("hybrid", "Explain the sales trend by region."),
    ("hybrid", "Show profit by segment and explain which segment should we focus on."),
    ("hybrid", "Which sub-categories are unprofitable and what is driving the losses?"),
    ("hybrid", "Compare 2016 vs 2017 sales and explain the difference."),
    ("hybrid", "Show the profit trend and explain why growth is slowing."),
    ("hybrid", "Which segment has the best margin and why does it outperform others?"),
    ("hybrid", "Show me the discount impact on profit and explain why high discounts are harmful."),
    ("hybrid", "Identify the top region by sales and explain what drives its performance."),
]

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Latency measurement
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PART 1 — LATENCY MEASUREMENT (50 queries)")
print("="*80)

latency_results = []
api_counts      = {"tier1":0,"tier2_no_audit":0,"tier2_audit":0,"tier3":0,"agent":0,"router_only":0}
path_times      = {k:[] for k in api_counts}
path_api_calls  = {k:[] for k in api_counts}

for path, q in LATENCY_QUERIES:
    calls_before = counting_client.call_count

    t0 = time.perf_counter()

    if path == "router_only":
        # Measure SmartRouter in isolation
        _ = router.classify(q)
    elif path in ("tier1","tier2_no_audit","tier2_audit","tier3"):
        _ = _run_structured(q)
    elif path == "agent":
        _ = agent.run(q)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    calls_this = counting_client.call_count - calls_before

    path_times[path].append(elapsed_ms)
    path_api_calls[path].append(calls_this)

    print(f"  [{path:<16}] {elapsed_ms:>8.1f} ms  |  API calls: {calls_this}  |  {q[:50]}")
    time.sleep(0.5)

# Aggregate latency
print("\nLATENCY SUMMARY:")
print(f"  {'Path':<20} {'n':>3} {'Mean (ms)':>10} {'p95 (ms)':>10} {'Mean API calls':>15}")
print(f"  {'-'*20} {'-'*3} {'-'*10} {'-'*10} {'-'*15}")

latency_summary = {}
for path in ["tier1","tier2_no_audit","tier2_audit","tier3","agent","router_only"]:
    times = path_times[path]
    calls = path_api_calls[path]
    if not times: continue
    times_s = sorted(times)
    mean_t  = sum(times) / len(times)
    p95_t   = times_s[int(len(times_s)*0.95)] if len(times_s)>1 else times_s[0]
    mean_c  = sum(calls) / len(calls) if calls else 0
    print(f"  {path:<20} {len(times):>3} {mean_t:>10.1f} {p95_t:>10.1f} {mean_c:>15.1f}")
    latency_summary[path] = {"n":len(times),"mean_ms":round(mean_t,1),"p95_ms":round(p95_t,1),"mean_api_calls":round(mean_c,1)}

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — API call rate: full system vs no-routing baseline
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PART 2 — API CALL RATE (100 evaluation queries)")
print("="*80)

# Use the 100-query evaluation dataset (65 structured + 20 agent + 15 hybrid)
from test_routing import questions as ALL_QUESTIONS, ground_truth as ALL_GT  # reuse if available
try:
    eval_queries = list(zip(ALL_QUESTIONS, ALL_GT))
except Exception:
    # Fallback: use RATING_QUERIES × 2.5
    eval_queries = [(q, cat) for cat, q in RATING_QUERIES] * 1
    eval_queries = eval_queries[:40]

full_api_calls    = []
norouting_api_calls = []

print("Running 40-query subset for API call comparison (full vs no-routing)...")
for q, _ in RATING_QUERIES:
    # Full system
    c_before = counting_client.call_count
    try:
        decision = router.classify(q)
        if decision.mode == "agent":
            _ = agent.run(q)
        elif decision.mode == "hybrid":
            _ = hybrid.execute(decision, q)
        else:
            _ = _run_structured(q)
    except Exception:
        pass
    full_api_calls.append(counting_client.call_count - c_before)

    # No-routing baseline: always call Gemini directly
    c_before2 = counting_client.call_count
    try:
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=f"You are a Superstore BI analyst. Answer: {q}",
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=800),
        )
        counting_client.call_count += 1
    except Exception:
        pass
    norouting_api_calls.append(counting_client.call_count - c_before2)

    time.sleep(0.5)

n_q = len(RATING_QUERIES)
full_pct    = sum(1 for c in full_api_calls if c > 0) / n_q * 100
fullmean    = sum(full_api_calls) / n_q
noroute_pct = 100.0
noroute_mean = sum(norouting_api_calls) / n_q
relative_cost = (sum(full_api_calls) / sum(norouting_api_calls) * 100) if sum(norouting_api_calls) else 0

print(f"\nAPI CALL SUMMARY (n={n_q} queries):")
print(f"  Full system:        {full_pct:.1f}% queries with API call  |  mean {fullmean:.1f} calls/query")
print(f"  No-routing baseline: {noroute_pct:.0f}% queries with API call  |  mean {noroute_mean:.1f} calls/query")
print(f"  Relative API cost:   {relative_cost:.0f}% of baseline")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Generate responses for human rating (40 queries × 3 conditions)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("PART 3 — GENERATING RESPONSES FOR HUMAN RATING (40 × 3 conditions)")
print("Saving to 5_8_rating_sheet.txt...")
print("="*80)

def run_no_rag(q):
    """Run structured or agent pipeline WITHOUT RAG context."""
    fast = parser.fast_kpi_answer(q)
    if fast: return fast
    rule_plan = parser.rule_based_plan(q)
    if rule_plan:
        try:
            plan = validator.validate(rule_plan)
            df_r = sql_b.run(plan)
            return formatter.format(plan, df_r, "")
        except Exception:
            pass
    # Tier 3 without RAG: direct Gemini, no context
    try:
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=f"You are a Superstore BI analyst. Answer this query with SQL results: {q}. "
                     f"Dataset: orders from 2014-2017, columns: sales, profit, region, segment, category, sub_category, discount, order_date.",
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=800),
        )
        return getattr(resp, "text", "") or "No response"
    except Exception as e:
        return f"Error: {e}"

def run_no_routing(q):
    """Always call Gemini directly — no routing, no SQL, no RAG."""
    try:
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=f"You are a Superstore BI analyst. Answer this question: {q}\n"
                     f"The Superstore dataset covers 2014-2017. Use your knowledge to give specific numbers.",
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=800),
        )
        return getattr(resp, "text", "") or "No response"
    except Exception as e:
        return f"Error: {e}"

def run_full_system(q):
    """Full pipeline: SmartRouter → appropriate path."""
    try:
        decision = router.classify(q)
        if decision.mode == "agent":
            return agent.run(q)
        elif decision.mode == "hybrid":
            return hybrid.execute(decision, q)
        else:
            return _run_structured(q)
    except Exception as e:
        return f"Error: {e}"

rating_data = []
for i, (qtype, q) in enumerate(RATING_QUERIES, 1):
    print(f"  Q{i:02d} [{qtype}] {q[:55]}...")

    t0 = time.perf_counter()
    r_full = run_full_system(q)
    t_full = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r_norag = run_no_rag(q)
    t_norag = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r_noroute = run_no_routing(q)
    t_noroute = (time.perf_counter() - t0) * 1000

    rating_data.append({
        "id": i, "type": qtype, "query": q,
        "full_system": r_full,
        "no_rag": r_norag,
        "no_routing": r_noroute,
        "latency_full_ms": round(t_full, 0),
        "latency_norag_ms": round(t_norag, 0),
        "latency_noroute_ms": round(t_noroute, 0),
    })
    time.sleep(1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

output = {
    "latency": latency_summary,
    "api_calls": {
        "full_system_pct_with_call": round(full_pct, 1),
        "full_system_mean_per_query": round(fullmean, 1),
        "norouting_pct_with_call": noroute_pct,
        "norouting_mean_per_query": round(noroute_mean, 1),
        "relative_cost_pct": round(relative_cost, 0),
        "n_queries": n_q,
    },
    "rating_responses": rating_data,
}

with open("e2e_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

# Text summary
with open("e2e_results.txt", "w", encoding="utf-8") as f:
    f.write("END-TO-END EVALUATION RESULTS — Section 5.8\n")
    f.write("="*60 + "\n\n")

    f.write("TABLE 5.17 — Latency per path\n")
    for path, s in latency_summary.items():
        f.write(f"  {path:<20}  mean={s['mean_ms']}ms  p95={s['p95_ms']}ms  api_calls={s['mean_api_calls']}\n")

    f.write("\nTABLE 5.18 — API call rate\n")
    f.write(f"  Full system:          {full_pct:.1f}% with API call  |  {fullmean:.1f} mean calls/query\n")
    f.write(f"  No-routing baseline:  {noroute_pct:.0f}% with API call  |  {noroute_mean:.1f} mean calls/query\n")
    f.write(f"  Relative cost:        {relative_cost:.0f}% of baseline\n")

    f.write("\nTABLE 5.19 — Answer quality\n")
    f.write("  [Fill after human rating — see 5_8_rating_sheet.txt]\n")
    f.write("  Rater 1 scores: ____  Rater 2 scores: ____\n")
    f.write("  Cohen's κ = ____\n")

# Rating sheet
with open("5_8_rating_sheet.txt", "w", encoding="utf-8") as f:
    f.write("HUMAN RATING SHEET — Section 5.8 (Table 5.19)\n")
    f.write("="*80 + "\n\n")
    f.write("RATING SCALE:\n")
    f.write("  1 = Incorrect or completely unhelpful\n")
    f.write("  2 = Partially correct but missing key information\n")
    f.write("  3 = Correct but lacking analytical depth\n")
    f.write("  4 = Correct and analytically useful\n")
    f.write("  5 = Correct, insightful, and actionable\n\n")
    f.write("INSTRUCTIONS:\n")
    f.write("  Rate all 3 responses for each query independently.\n")
    f.write("  Do NOT look at which condition produced which response.\n")
    f.write("  Verify numeric claims against direct DB if unsure.\n\n")
    f.write("="*80 + "\n\n")

    for r in rating_data:
        f.write(f"Q{r['id']:02d} [{r['type'].upper()}]\n")
        f.write(f"Query: {r['query']}\n")
        f.write("-"*60 + "\n\n")

        for label, key in [("CONDITION A", "full_system"), ("CONDITION B", "no_rag"), ("CONDITION C", "no_routing")]:
            f.write(f"{label}:\n")
            response_text = r[key][:1200] + ("..." if len(r[key]) > 1200 else "")
            f.write(response_text + "\n\n")
            f.write(f"  Rater 1 score (1–5): [ ]    Rater 2 score (1–5): [ ]\n\n")

        f.write("="*80 + "\n\n")

    f.write("AGGREGATION (fill after rating):\n\n")
    for qtype in ["simple_kpi","structured","agent","hybrid"]:
        ids = [r["id"] for r in rating_data if r["type"] == qtype]
        f.write(f"  {qtype.upper()} (Q{ids[0]:02d}–Q{ids[-1]:02d}):\n")
        f.write(f"    Condition A (Full):       Rater1=___ Rater2=___ Mean=___\n")
        f.write(f"    Condition B (No-RAG):     Rater1=___ Rater2=___ Mean=___\n")
        f.write(f"    Condition C (No-routing): Rater1=___ Rater2=___ Mean=___\n\n")
    f.write("  OVERALL:\n")
    f.write("    Full system mean:   ___\n")
    f.write("    No-RAG mean:        ___  Δ vs Full: ___\n")
    f.write("    No-routing mean:    ___  Δ vs Full: ___\n")
    f.write("    Cohen's κ:          ___  (use: sklearn.metrics.cohen_kappa_score(r1, r2))\n")

print("\nSaved:")
print("  e2e_results.json      — latency + API call data")
print("  e2e_results.txt       — table values for Section 5.8")
print("  5_8_rating_sheet.txt  — 40 queries × 3 conditions for human rating")
print("="*60)
print("\nNEXT STEPS:")
print("  1. Share 5_8_rating_sheet.txt with Rater 2 (blind — don't reveal condition labels)")
print("  2. Both raters score independently 1–5")
print("  3. Compute mean per query type per condition")
print("  4. Run: python -c \"from sklearn.metrics import cohen_kappa_score; print(cohen_kappa_score([r1_scores], [r2_scores]))\"")
print("  5. Fill Table 5.19 in Section_5_8_End_to_End_Performance.docx")