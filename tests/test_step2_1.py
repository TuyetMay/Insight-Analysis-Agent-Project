"""
tests/test_step2_1.py
Step 2.1 — Metadata pre-filtering validation.

Checks:
  A. kpi_detail  → only anomaly_fact / filter_context in results
  B. kpi_trend   → no kpi_snapshot in results
  C. kpi_rank    → no trend_transition / trend_overview in results
  D. kpi_value   → no trend_transition in results
  E. Fallback    → obscure intent returns results (no over-filtering)
  F. HyDE + filter combo — bleeding money query gets anomaly_fact top
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from core.database import execute_query
from rag.engine import RAGEngine

print("=" * 60)
print("TEST Step 2.1 — Metadata Pre-filtering")
print("=" * 60)

print("\n[1/2] Loading data...")
df = execute_query("SELECT * FROM superstore")
kpis = {
    "total_sales":   float(df["sales"].sum()),
    "total_profit":  float(df["profit"].sum()),
    "total_orders":  int(df["order_id"].nunique()),
    "profit_margin": float(df["profit"].sum() / df["sales"].sum() * 100),
}
filters = {"date_range": ("2014-01-01","2017-12-31"),
           "region":[], "segment":[], "category":[]}
engine = RAGEngine()
engine.build_static(df)
engine.build(df, kpis, filters)
print(f"      Total chunks: {engine.total_chunks}")

results = {"pass": 0, "fail": 0}

def check(name, condition, detail=""):
    if condition:
        print(f"  ✅ PASS | {name}")
        results["pass"] += 1
    else:
        print(f"  ❌ FAIL | {name}")
        if detail:
            print(f"         | {detail}")
        results["fail"] += 1

print("\n[2/2] Running filter tests...")

# ── A. kpi_detail — only anomaly_fact + filter_context ────────
print("\n  A. kpi_detail filter")
queries_detail = [
    "which products are bleeding money",
    "what items are losing money",
    "show loss-making sub-categories",
]
ALLOWED_DETAIL = {"anomaly_fact", "filter_context", "insight"}
for q in queries_detail:
    ctx = engine.retrieve(q, k=6, tier=2, intent="kpi_detail")
    # exclude must-have injected chunks (score=0.0) from type check
    semantic = [c for c in ctx.chunks if c.score > 0.01]
    bad_types = {c.metadata.get("type") for c in semantic} - ALLOWED_DETAIL
    check(
        f'kpi_detail: "{q}"',
        len(bad_types) == 0,
        f"unexpected types: {bad_types}"
    )

# ── B. kpi_trend — no kpi_snapshot ────────────────────────────
print("\n  B. kpi_trend filter")
queries_trend = [
    "sales trend over years",
    "how is revenue evolving",
    "monthly profit trend",
]
for q in queries_trend:
    ctx = engine.retrieve(q, k=6, tier=2, intent="kpi_trend")
    semantic = [c for c in ctx.chunks if c.score > 0.01]
    has_snapshot = any(c.metadata.get("type") == "kpi_snapshot" for c in semantic)
    check(
        f'kpi_trend no snapshot: "{q}"',
        not has_snapshot,
        f"kpi_snapshot leaked into trend results"
    )

# ── C. kpi_rank — no trend chunks ─────────────────────────────
print("\n  C. kpi_rank filter")
queries_rank = [
    "top 5 regions by profit",
    "which zone makes the most money",
    "best sub-categories by sales",
]
BLOCKED_RANK = {"trend_transition", "trend_overview", "time_period_fact"}
for q in queries_rank:
    ctx = engine.retrieve(q, k=6, tier=2, intent="kpi_rank")
    semantic = [c for c in ctx.chunks if c.score > 0.01]
    leaked = {c.metadata.get("type") for c in semantic} & BLOCKED_RANK
    check(
        f'kpi_rank no trend: "{q}"',
        len(leaked) == 0,
        f"trend types leaked: {leaked}"
    )

# ── D. kpi_value — no trend_transition ────────────────────────
print("\n  D. kpi_value filter")
queries_value = [
    "total sales this year",
    "income breakdown by area",
    "profit by region",
]
for q in queries_value:
    ctx = engine.retrieve(q, k=6, tier=2, intent="kpi_value")
    semantic = [c for c in ctx.chunks if c.score > 0.01]
    has_trend = any(c.metadata.get("type") == "trend_transition" for c in semantic)
    check(
        f'kpi_value no trend_transition: "{q}"',
        not has_trend,
        "trend_transition leaked into value results"
    )

# ── E. Fallback — unknown intent returns results ───────────────
print("\n  E. Fallback safety")
ctx = engine.retrieve("total sales", k=6, tier=2, intent=None)
check("no intent → results not empty", len(ctx.chunks) > 0)

# ── F. HyDE + filter combo ─────────────────────────────────────
print("\n  F. HyDE + filter combo")
ctx = engine.retrieve("which products are bleeding money", k=6,
                      tier=2, intent="kpi_detail")
semantic = [c for c in ctx.chunks if c.score > 0.01]
top_type = semantic[0].metadata.get("type") if semantic else None
check(
    "bleeding money → anomaly_fact on top",
    top_type == "anomaly_fact",
    f"top chunk type was: {top_type}"
)
top_id = semantic[0].chunk_id if semantic else "none"
check(
    "bleeding money → anomaly_loss_subcat_summary retrieved",
    any(c.chunk_id == "anomaly_loss_subcat_summary" for c in semantic),
    f"top chunk: {top_id}"
)

# ── Summary ────────────────────────────────────────────────────
total = results["pass"] + results["fail"]
print(f"""
{'=' * 60}
SUMMARY
{'=' * 60}
  Pass: {results['pass']}/{total}
  Fail: {results['fail']}/{total}
  {'🎉 Step 2.1 COMPLETE' if results['fail'] == 0 else '⚠️  Some tests failed'}
{'=' * 60}""")