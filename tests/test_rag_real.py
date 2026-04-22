# test_rag_real.py
import os

from core.database import execute_query

from rag.engine import RAGEngine

# Lấy sample data thật
df = execute_query("SELECT * FROM superstore LIMIT 500")
kpis = {
    "total_sales":   df["sales"].sum(),
    "total_profit":  df["profit"].sum(),
    "total_orders":  df["order_id"].nunique(),
    "profit_margin": df["profit"].sum() / df["sales"].sum() * 100,
}

engine = RAGEngine()
engine.build_static(df)
engine.build(df, kpis, filters={})

# Những query test đồng nghĩa
queries = [
    "revenue by region",           # "revenue" ≠ "sales" với TF-IDF
    "which area makes most money", # paraphrase hoàn toàn
    "show me losses",              # "losses" → profit negative
    "how many transactions",       # "transactions" → orders
    "income trend over years",     # "income" → sales trend
]

print("=== RAG Engine Test với data thật ===\n")
for q in queries:
    ctx = engine.retrieve(q, k=3)
    print(f"Query: '{q}'")
    for i, c in enumerate(ctx.chunks[:3], 1):
        print(f"  {i}. [{c.chunk_id}] score={c.score:.3f} — {c.text[:80]}...")
    print()