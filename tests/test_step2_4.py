"""
test_step2_4.py — Verify Step 2.4: Conditional schema chunk injection

Chạy từ root project:
    python test_step2_4.py

Kiểm tra:
  A. Tier-2 queries: schema chunks KHÔNG được inject
  B. Tier-3 queries: schema chunks ĐƯỢC inject
  C. Intent must-haves: đúng chunks được inject theo intent
  D. Token savings: đo lường actual savings
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from core.database import execute_query
from rag.engine import RAGEngine

print("=" * 60)
print("TEST Step 2.4 — Conditional Must-Have Injection")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────
print("\n[1/4] Loading data...")
df = execute_query("SELECT * FROM superstore")
kpis = {
    "total_sales":   float(df["sales"].sum()),
    "total_profit":  float(df["profit"].sum()),
    "total_orders":  int(df["order_id"].nunique()),
    "profit_margin": float(df["profit"].sum() / df["sales"].sum() * 100),
}
filters = {
    "date_range": (str(pd.to_datetime(df["order_date"]).min().date()),
                   str(pd.to_datetime(df["order_date"]).max().date())),
    "region": [], "segment": [], "category": [],
}

engine = RAGEngine()
engine.build_static(df)
engine.build(df, kpis, filters)
print(f"      Total chunks: {engine.total_chunks}")

SCHEMA_IDS = {"schema_metrics", "schema_dimensions"}

def count_tokens(chunks):
    return sum(len(c.text) // 4 for c in chunks)

def has_schema(chunks):
    ids = {c.chunk_id for c in chunks}
    return ids & SCHEMA_IDS

# ── Test A: Tier-2 — schema KHÔNG inject ──────────────────────
print("\n[2/4] TEST A — Tier-2 queries (schema should NOT be injected)")
print("-" * 55)

tier2_queries = [
    ("total sales this year",         "kpi_value",   None),
    ("sales trend over years",         "kpi_trend",   None),
    ("top 5 regions by profit",        "kpi_rank",    "region"),
    ("which products are losing money","kpi_detail",  None),
    ("compare this year vs last year", "kpi_compare", None),
]

tier2_pass = 0
tier2_total_tokens_saved = 0

for query, intent, breakdown in tier2_queries:
    ctx = engine.retrieve(query, k=6, intent=intent, breakdown_by=breakdown, tier=2)
    schema_found = has_schema(ctx.chunks)
    tokens = count_tokens(ctx.chunks[:8])

    # Estimate what tier-3 would add
    schema_tokens = sum(len(c.text) // 4
                        for c in (engine._static_chunks + engine._dynamic_chunks)
                        if c.chunk_id in SCHEMA_IDS)

    status = "✅ PASS" if not schema_found else "❌ FAIL (schema leaked)"
    tier2_total_tokens_saved += schema_tokens if not schema_found else 0
    if not schema_found:
        tier2_pass += 1

    print(f"  {status} | ~{tokens}t | intent={intent}")
    print(f"           query='{query[:50]}'")
    if schema_found:
        print(f"           ⚠️  SCHEMA FOUND: {schema_found}")
    chunk_ids = [c.chunk_id for c in ctx.chunks[:5]]
    print(f"           top chunks: {chunk_ids}")
    print()

print(f"  Tier-2 pass rate: {tier2_pass}/{len(tier2_queries)}")
print(f"  Estimated tokens saved vs old behavior: ~{tier2_total_tokens_saved} tokens across {len(tier2_queries)} queries")

# ── Test B: Tier-3 — schema ĐƯỢC inject ───────────────────────
print("\n[3/4] TEST B — Tier-3 queries (schema SHOULD be injected)")
print("-" * 55)

tier3_queries = [
    ("what are the trends in sales?",  "kpi_trend",  None),
    ("which region is most profitable?","kpi_rank",   "region"),
    ("compare profits year over year", "kpi_compare", None),
]

tier3_pass = 0
for query, intent, breakdown in tier3_queries:
    ctx = engine.retrieve(query, k=7, intent=intent, breakdown_by=breakdown, tier=3)
    schema_found = has_schema(ctx.chunks)
    tokens = count_tokens(ctx.chunks[:10])

    status = "✅ PASS" if schema_found else "❌ FAIL (schema missing)"
    if schema_found:
        tier3_pass += 1

    print(f"  {status} | ~{tokens}t | intent={intent}")
    print(f"           query='{query[:50]}'")
    print(f"           schema injected: {schema_found}")
    chunk_ids = [c.chunk_id for c in ctx.chunks[:5]]
    print(f"           top chunks: {chunk_ids}")
    print()

print(f"  Tier-3 pass rate: {tier3_pass}/{len(tier3_queries)}")

# ── Test C: Intent must-haves ─────────────────────────────────
print("\n[4/4] TEST C — Intent must-have injection")
print("-" * 55)

EXPECTED_MUST_HAVES = {
    "kpi_value":   {"kpi_sales_snapshot", "kpi_profit_snapshot"},
    "kpi_trend":   {"trend_overview_sales_yearly"},
    "kpi_rank":    {"top10_sub_category_profit"},
    "kpi_detail":  {"anomaly_loss_subcat_summary"},
    "kpi_compare": {"filter_active", "trend_overview_sales_yearly"},
}

intent_pass = 0
for intent, expected_ids in EXPECTED_MUST_HAVES.items():
    query = f"test query for {intent}"
    ctx   = engine.retrieve(query, k=6, intent=intent, tier=2)
    found = {c.chunk_id for c in ctx.chunks}
    injected = expected_ids & found
    missing  = expected_ids - found

    if not missing:
        intent_pass += 1
        status = "✅ PASS"
    else:
        status = f"⚠️  PARTIAL ({len(injected)}/{len(expected_ids)})"

    print(f"  {status} | intent={intent}")
    print(f"           expected : {sorted(expected_ids)}")
    print(f"           injected : {sorted(injected)}")
    if missing:
        print(f"           missing  : {sorted(missing)}")
    print()

print(f"  Intent must-have pass rate: {intent_pass}/{len(EXPECTED_MUST_HAVES)}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  A. Tier-2 no-schema :  {tier2_pass}/{len(tier2_queries)} pass")
print(f"  B. Tier-3 schema     :  {tier3_pass}/{len(tier3_queries)} pass")
print(f"  C. Intent must-haves :  {intent_pass}/{len(EXPECTED_MUST_HAVES)} pass")

# Token savings estimate (production scale)
schema_tokens_each = 106  # ~43 + 63 tokens
print(f"\n  Token savings (Tier-2 queries):")
print(f"    Per query   : ~{schema_tokens_each} tokens saved")
print(f"    Per 100 queries: ~{schema_tokens_each * 100:,} tokens saved")
print(f"    (Assuming ~80% Tier-2): ~{int(schema_tokens_each * 80):,} tokens/100 queries")

total_pass = tier2_pass + tier3_pass + intent_pass
total_tests = len(tier2_queries) + len(tier3_queries) + len(EXPECTED_MUST_HAVES)
print(f"\n  Overall: {total_pass}/{total_tests} tests passed")
if total_pass == total_tests:
    print("  🎉 Step 2.4 COMPLETE — all tests passed!")
else:
    print("  ⚠️  Some tests failed — check output above")
print("=" * 60)