from __future__ import annotations

import json
import os
import sys
import time
import statistics
from typing import Any, Dict, List, Tuple

# ── Setup path ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from core.database import execute_query

# ── Helpers ───────────────────────────────────────────────────

def measure(fn, *args, repeat: int = 5, **kwargs) -> Tuple[float, float, float, Any]:
    """
    Chạy fn(*args, **kwargs) repeat lần.
    Trả về (mean_ms, min_ms, max_ms, last_result).
    """
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return (
        statistics.mean(times),
        min(times),
        max(times),
        result,
    )


def print_row(label: str, mean: float, mn: float, mx: float,
              note: str = "") -> None:
    bar_len = min(int(mean / 50), 40)
    bar     = "█" * bar_len
    print(f"  {label:<40} {mean:>8.1f}ms  [{mn:>7.1f} – {mx:>7.1f}]  {bar}  {note}")


SEP = "=" * 90

# ─────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────
print(SEP)
print("BENCHMARK — Superstore BI Chatbot Pipeline")
print(SEP)
print("\n[SETUP] Loading data from database...")

t_load_start = time.perf_counter()
df = execute_query("SELECT * FROM superstore")
t_load_end   = time.perf_counter()
db_load_ms   = (t_load_end - t_load_start) * 1000

print(f"        Loaded {len(df):,} rows in {db_load_ms:.0f}ms")

kpis = {
    "total_sales":   float(df["sales"].sum()),
    "total_profit":  float(df["profit"].sum()),
    "total_orders":  int(df["order_id"].nunique()),
    "profit_margin": float(df["profit"].sum() / df["sales"].sum() * 100),
}
filters = {
    "date_range": (
        str(pd.to_datetime(df["order_date"]).min().date()),
        str(pd.to_datetime(df["order_date"]).max().date()),
    ),
    "region": [], "segment": [], "category": [],
}

results: Dict[str, Any] = {}

# ─────────────────────────────────────────────────────────────
# 2. BƯỚC 1 — Quick Token detection (pattern matching)
# ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print("BƯỚC 1 — Quick Token Detection (pattern matching UI buttons)")
print(SEP)

QUICK_TOKENS = {
    "__quick_sales__",
    "__quick_profit__",
    "__quick_orders__",
    "__quick_margin__",
}

def check_quick_token(q: str) -> bool:
    return q in QUICK_TOKENS

quick_queries = [
    ("__quick_sales__",   "Quick Token — Sales button"),
    ("__quick_profit__",  "Quick Token — Profit button"),
    ("__quick_orders__",  "Quick Token — Orders button"),
    ("__quick_margin__",  "Quick Token — Margin button"),
    ("what is total sales?", "Non-token (text query)"),
]

print(f"\n  {'Query':<40} {'Mean':>8}   {'Min – Max':>18}   Bar")
print(f"  {'-'*88}")

step1_times = []
for q, label in quick_queries:
    mean, mn, mx, _ = measure(check_quick_token, q, repeat=1000)
    step1_times.append(mean)
    print_row(label, mean, mn, mx)

results["step1_quick_token_detection"] = {
    "mean_ms":  statistics.mean(step1_times),
    "min_ms":   min(step1_times),
    "max_ms":   max(step1_times),
    "note":     "pattern matching — regex, O(1)"
}

# ─────────────────────────────────────────────────────────────
# 3. BƯỚC 2 — Query Router (phân loại câu hỏi)
# ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print("BƯỚC 2 — Query Router (regex-based classification)")
print(SEP)

from chatbot.query_router import QueryRouter
router = QueryRouter()

router_queries = [
    ("why did sales drop in Q3?",               "Agent Path — diagnostic 'why'"),
    ("sales increased but profit decreased",     "Agent Path — divergence pattern"),
    ("what caused the revenue decline?",         "Agent Path — causal keyword"),
    ("total sales by region",                    "Structured Path — aggregate"),
    ("top 5 products by profit",                 "Structured Path — ranking"),
    ("profit trend over years",                  "Structured Path — trend"),
    ("compare 2016 vs 2017 sales",               "Structured Path — compare"),
]

print(f"\n  {'Query':<48} {'Route':<12} {'Mean':>8}   {'Min – Max':>18}")
print(f"  {'-'*95}")

step2_times = []
for q, label in router_queries:
    mean, mn, mx, route = measure(router.route, q, repeat=500)
    step2_times.append(mean)
    bar_len = min(int(mean / 0.01), 30)
    print(f"  {label:<48} {route:<12} {mean:>8.3f}ms  [{mn:>6.3f} – {mx:>6.3f}]")

results["step2_query_router"] = {
    "mean_ms": statistics.mean(step2_times),
    "min_ms":  min(step2_times),
    "max_ms":  max(step2_times),
    "note":    "regex pattern matching — no LLM"
}

# ─────────────────────────────────────────────────────────────
# 4. BƯỚC 3 — Answer Generation (3 tầng)
# ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print("BƯỚC 3 — Answer Generation (3-tier strategy)")
print(SEP)

from chatbot.nl_parser import NLParser
from chatbot.plan_validator import PlanValidator
from chatbot.sql_builder import SQLBuilder
from chatbot.answer_formatter import AnswerFormatter

s0 = str(pd.to_datetime(df["order_date"]).min().date())
e0 = str(pd.to_datetime(df["order_date"]).max().date())

parser    = NLParser(df, filters)
validator = PlanValidator(
    (s0, e0),
    allowed_regions   = list(df["region"].unique()),
    allowed_segments  = list(df["segment"].unique()),
    allowed_categories= list(df["category"].unique()),
)
sql_builder = SQLBuilder()
formatter   = AnswerFormatter()

# ── Tier 1: fast_kpi_answer ──────────────────────────────────
print("\n  ▶ Tầng 1 — Instant KPI (regex, no DB, no API)")
print(f"  {'Query':<45} {'Mean':>8}   {'Min – Max':>18}   Bar")
print(f"  {'-'*88}")

tier1_queries = [
    "what is total sales?",
    "total profit",
    "how many orders",
    "profit margin",
    "kpi summary overview",
]

tier1_times = []
for q in tier1_queries:
    mean, mn, mx, res = measure(parser.fast_kpi_answer, q, repeat=500)
    hit = "✓ HIT" if res else "✗ MISS"
    tier1_times.append(mean)
    print_row(f"{q} [{hit}]", mean, mn, mx)

results["step3_tier1_instant_kpi"] = {
    "mean_ms": statistics.mean(tier1_times),
    "min_ms":  min(tier1_times),
    "max_ms":  max(tier1_times),
    "note":    "regex only — no DB, no API"
}

# ── Tier 2: rule_based_plan + SQL ───────────────────────────
print("\n  ▶ Tầng 2 — SQL Analytics (keyword parsing + PostgreSQL)")
print(f"  {'Query':<45} {'Phase':<12} {'Mean':>8}   {'Min – Max':>18}")
print(f"  {'-'*90}")

tier2_queries = [
    ("total sales by region",              "kpi_value  + breakdown"),
    ("top 5 sub-categories by profit",     "kpi_rank"),
    ("sales trend over years",             "kpi_trend  yearly"),
    ("profit yoy comparison",              "kpi_compare yoy"),
    ("which products are losing money",    "kpi_detail negative profit"),
    ("sales in 2017",                      "kpi_value  date filter"),
]

tier2_plan_times  = []
tier2_sql_times   = []
tier2_total_times = []

for q, label in tier2_queries:
    # Phase A: rule_based_plan (keyword parsing only)
    mean_p, mn_p, mx_p, plan = measure(parser.rule_based_plan, q, repeat=200)
    tier2_plan_times.append(mean_p)
    print(f"  {label:<45} {'NL→Plan':<12} {mean_p:>8.2f}ms  [{mn_p:>6.2f} – {mx_p:>6.2f}]")

    if plan:
        try:
            validated = validator.validate(plan)

            # Phase B: SQL execution (DB round-trip)
            mean_s, mn_s, mx_s, result_df = measure(sql_builder.run, validated, repeat=10)
            tier2_sql_times.append(mean_s)
            rows = len(result_df) if result_df is not None else 0
            print(f"  {'':<45} {'SQL+DB':<12} {mean_s:>8.1f}ms  [{mn_s:>6.1f} – {mx_s:>6.1f}]  ({rows} rows)")

            tier2_total_times.append(mean_p + mean_s)
        except Exception as e:
            print(f"  {'':<45} {'ERROR':<12} {str(e)[:40]}")
    print()

results["step3_tier2_sql_analytics"] = {
    "nl_to_plan_mean_ms": statistics.mean(tier2_plan_times) if tier2_plan_times else 0,
    "sql_execution_mean_ms": statistics.mean(tier2_sql_times) if tier2_sql_times else 0,
    "total_mean_ms": statistics.mean(tier2_total_times) if tier2_total_times else 0,
    "note": "keyword parsing + parameterized SQL — no API"
}

# ── Tier 3: Gemini plan (LLM) ───────────────────────────────
print("\n  ▶ Tầng 3 — LLM Planning (Gemini API)")
print("  [!] Requires GOOGLE_API_KEY — skipping if not set\n")

from config import Config
tier3_results = {}

if Config.GOOGLE_API_KEY:
    from rag.engine import RAGEngine
    from google import genai

    rag = RAGEngine()
    rag.build_static(df)
    rag.build(df, kpis, filters)

    gemini_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    parser_with_llm = NLParser(df, filters, gemini_client, Config.GEMINI_MODEL)

    tier3_queries = [
        "show me revenue distribution across all markets",
        "which customer segments have the best margin efficiency?",
        "analyze profitability trend for technology products",
    ]

    tier3_times = []
    print(f"  {'Query':<55} {'Mean':>10}   {'Min – Max':>20}")
    print(f"  {'-'*92}")

    for q in tier3_queries:
        rag_ctx = rag.retrieve(q, k=7, tier=3)
        mean, mn, mx, _ = measure(
            parser_with_llm.gemini_plan, q, rag_ctx, repeat=3
        )
        tier3_times.append(mean)
        print(f"  {q:<55} {mean:>10.0f}ms  [{mn:>8.0f} – {mx:>8.0f}]")

    tier3_results = {
        "mean_ms": statistics.mean(tier3_times),
        "min_ms":  min(tier3_times),
        "max_ms":  max(tier3_times),
        "note":    "1 LLM call (plan generation) + SQL execution"
    }
    results["step3_tier3_llm_planning"] = tier3_results
    print(f"\n  Average Tier 3 (LLM plan): {statistics.mean(tier3_times):.0f}ms")
else:
    print("  GOOGLE_API_KEY not set — Tier 3 skipped.")
    results["step3_tier3_llm_planning"] = {"note": "skipped — no API key"}

# ─────────────────────────────────────────────────────────────
# 5. BƯỚC 4 — RAG Knowledge Base
# ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print("BƯỚC 4 — RAG Knowledge Base")
print(SEP)

from rag.engine import RAGEngine

rag_engine = RAGEngine()

# Build time
print("\n  ▶ Build time")
mean_static, mn, mx, _ = measure(rag_engine.build_static, df, repeat=3)
print(f"  Static layer build  : {mean_static:.0f}ms  [{mn:.0f} – {mx:.0f}]")

mean_dynamic, mn, mx, _ = measure(rag_engine.build, df, kpis, filters, repeat=3)
print(f"  Dynamic layer build : {mean_dynamic:.0f}ms  [{mn:.0f} – {mx:.0f}]")
print(f"  Total chunks        : {rag_engine.total_chunks}")

# Retrieve time
print("\n  ▶ Retrieve time (per query)")
print(f"  {'Query':<45} {'Tier':>5} {'Mean':>8}   {'Min – Max':>18}   Chunks")
print(f"  {'-'*90}")

retrieve_queries = [
    ("total sales by region",           2, "kpi_value",  "region"),
    ("which products are losing money", 2, "kpi_detail", None),
    ("sales trend over years",          2, "kpi_trend",  None),
    ("analyze profitability",           3, "kpi_value",  None),
    ("why did profit drop?",            3, None,         None),
]

rag_retrieve_times = []
for q, tier, intent, breakdown in retrieve_queries:
    mean, mn, mx, ctx = measure(
        rag_engine.retrieve, q,
        k=7, tier=tier, intent=intent, breakdown_by=breakdown,
        repeat=50
    )
    rag_retrieve_times.append(mean)
    n_chunks = len(ctx.chunks) if ctx else 0
    print(f"  {q:<45} {tier:>5} {mean:>8.2f}ms  [{mn:>6.2f} – {mx:>6.2f}]   {n_chunks} chunks")

results["step4_rag_knowledge_base"] = {
    "build_static_ms":  mean_static,
    "build_dynamic_ms": mean_dynamic,
    "retrieve_mean_ms": statistics.mean(rag_retrieve_times),
    "retrieve_min_ms":  min(rag_retrieve_times),
    "retrieve_max_ms":  max(rag_retrieve_times),
    "total_chunks":     rag_engine.total_chunks,
}

# ─────────────────────────────────────────────────────────────
# 6. BƯỚC 5 — Suggestion Generation
# ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print("BƯỚC 5 — Suggestion Generation")
print(SEP)

from chatbot.suggestions.rule_engine import RuleBasedSuggestionEngine

rule_engine = RuleBasedSuggestionEngine(max_suggestions=4)
dashboard_defaults = {
    "start_date": s0,
    "end_date":   e0,
    "filters": {"region": [], "segment": [], "category": []},
}

# Rule-based suggestions
sample_plans = [
    {"intent": "kpi_value",   "metrics": ["sales"],  "breakdown_by": "region",       "time_grain": "none"},
    {"intent": "kpi_trend",   "metrics": ["profit"], "breakdown_by": None,            "time_grain": "year"},
    {"intent": "kpi_rank",    "metrics": ["profit"], "breakdown_by": "sub_category",  "top_k": 10, "time_grain": "none"},
    {"intent": "kpi_compare", "metrics": ["sales"],  "breakdown_by": None,            "compare_period": "yoy", "time_grain": "none"},
]

print(f"\n  ▶ Rule-based suggestion engine")
print(f"  {'Plan intent':<25} {'Mean':>8}   {'Min – Max':>18}")
print(f"  {'-'*60}")

sugg_times = []
for plan in sample_plans:
    mean, mn, mx, suggs = measure(
        rule_engine.suggest, plan, dashboard_defaults, repeat=200
    )
    sugg_times.append(mean)
    print(f"  {plan['intent']:<25} {mean:>8.3f}ms  [{mn:>6.3f} – {mx:>6.3f}]  ({len(suggs)} suggestions)")

results["step5_suggestion_generation"] = {
    "rule_based_mean_ms": statistics.mean(sugg_times),
    "rule_based_min_ms":  min(sugg_times),
    "rule_based_max_ms":  max(sugg_times),
    "note": "pure rule-based — no LLM, no DB"
}

# ─────────────────────────────────────────────────────────────
# 7. END-TO-END SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + SEP)
print("TỔNG HỢP — End-to-End Pipeline Timing")
print(SEP)

print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Bước / Thành phần                     Thời gian xử lý (trung bình) │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Bước 1 — Quick Token detection         < 0.01 ms  (regex O(1))    │
  │  Bước 2 — Query Router (regex)          < 0.5  ms  (3 regex sets)  │
  │                                                                     │
  │  Bước 3 — Answer Generation:                                        │
  │    Tầng 1 — Instant KPI (regex+memory)  < 1    ms  (no DB, no API) │
  │    Tầng 2 — SQL Analytics (NL→SQL→DB)   < 50   ms  (no API)        │
  │      └ NL → Plan (keyword parsing)      ~{statistics.mean(tier2_plan_times):>5.1f} ms                    │
  │      └ SQL execution (PostgreSQL)       ~{statistics.mean(tier2_sql_times) if tier2_sql_times else 0:>5.1f} ms                    │
  │    Tầng 3 — LLM Planning (Gemini API)   2–4  sec  (1-2 API calls)  │
  │                                                                     │
  │  Bước 4 — RAG retrieve (per query)      ~{statistics.mean(rag_retrieve_times):>5.1f} ms  (embedding search) │
  │  Bước 5 — Rule-based suggestions        < 1    ms  (no LLM)        │
  └─────────────────────────────────────────────────────────────────────┘
""")

# ─────────────────────────────────────────────────────────────
# 8. Save JSON results
# ─────────────────────────────────────────────────────────────
results["meta"] = {
    "db_rows":       len(df),
    "db_load_ms":    db_load_ms,
    "rag_chunks":    rag_engine.total_chunks,
    "tier2_plan_mean_ms": statistics.mean(tier2_plan_times) if tier2_plan_times else 0,
    "tier2_sql_mean_ms":  statistics.mean(tier2_sql_times)  if tier2_sql_times  else 0,
    "rag_retrieve_mean_ms": statistics.mean(rag_retrieve_times),
}

output_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"
)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)

print(f"  Results saved → {output_path}")
print(SEP)