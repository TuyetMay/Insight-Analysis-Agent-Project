"""
test_rag_quality.py
────────────────────────────────────────────────────────────────────────────────
Evaluates RAG retrieval quality (Section 5.5).

Produces:
  rag_results.txt   — summary table values (paste into Section 5.5)
  rag_results.json  — full per-query detail

Metrics:
  1. Precision@6: dense retriever vs TF-IDF fallback (30 queries)
  2. HyDE expansion impact on informal queries (5 queries)
  3. Grounding score: with vs without forced pre-query injection (20 queries)

Usage:
    python test_rag_quality.py

Place in project root. Requires .env with GCP_PROJECT and GCP_LOCATION.
"""

import os, sys, json, re, time
from dotenv import load_dotenv
load_dotenv()
sys.path.append('..')

from config import Config
from google import genai
from core.data_loader import load_filtered_data_safe, calculate_kpis, get_filter_options
from rag.engine import RAGEngine
from rag.retriever import DenseRetriever, TFIDFFallback
from rag.knowledge_builder import KnowledgeBaseBuilder
from rag.hyde import HyDEExpander
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

# Build chunks
builder        = KnowledgeBaseBuilder()
static_chunks  = builder.build_static(df)
dynamic_chunks = builder.build_dynamic(df, kpis, filters)
all_chunks     = static_chunks + dynamic_chunks

# Build RAG engine (dense)
rag = RAGEngine()
rag.build_static(df)
rag.build(df, kpis, filters)

# Build TF-IDF retriever over same chunks
tfidf_static  = TFIDFFallback()
tfidf_dynamic = TFIDFFallback()
tfidf_static.fit(static_chunks)
tfidf_dynamic.fit(dynamic_chunks)

hyde = HyDEExpander()

print("Setup complete.")
print(f"  Total chunks: {len(all_chunks)} (static={len(static_chunks)}, dynamic={len(dynamic_chunks)})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Precision@6: Dense vs TF-IDF (30 queries)
# ─────────────────────────────────────────────────────────────────────────────
#
# relevant_chunk_ids: manually labelled chunk IDs that a correct answer MUST use.
# Rule: a chunk is relevant if it contains a numeric fact or schema element
#       that the ground-truth answer depends on.
# These IDs come from the chunk dump (run rag/audit_test_chunk.py first).
#
# FORMAT: { "query": "...", "intent": "...", "relevant_ids": ["chunk_id1", ...] }
#
# ⚠️  Fill in relevant_ids after reading audit_chunks_dump.txt
#     Look at the chunk text and decide which top-6 chunks should appear.

PRECISION_QUERIES = [
    # ── Simple KPI / Snapshot (n=6) ──────────────────────────────────────────
    {
        "q": "What is the total sales revenue?",
        "intent": "simple_kpi",
        "relevant_ids": ["kpi_sales_snapshot", "kpi_profit_snapshot", "filter_active"],
    },
    {
        "q": "What is the overall profit margin?",
        "intent": "simple_kpi",
        "relevant_ids": ["kpi_margin_snapshot", "kpi_profit_snapshot", "filter_active"],
    },
    {
        "q": "How many orders were placed in total?",
        "intent": "simple_kpi",
        "relevant_ids": ["kpi_orders_snapshot", "kpi_sales_snapshot", "filter_active"],
    },
    {
        "q": "What is the total profit in 2016?",
        "intent": "simple_kpi",
        "relevant_ids": ["year_2016_sales_fact", "kpi_profit_snapshot", "trend_year_sales_2015_2016"],
    },
    {
        "q": "What is total revenue for Q4 2016?",
        "intent": "simple_kpi",
        "relevant_ids": ["quarter_2016_annual_summary", "kpi_sales_snapshot", "year_2016_sales_fact"],
    },
    {
        "q": "What is the profit margin for the Consumer segment?",
        "intent": "simple_kpi",
        "relevant_ids": ["segment_consumer_fact", "segment_ranked_by_profit", "kpi_margin_snapshot"],
    },

    # ── Structured Breakdown / Rank (n=8) ─────────────────────────────────────
    {
        "q": "What are the total sales by region?",
        "intent": "structured_breakdown",
        "relevant_ids": ["region_ranked_by_sales", "region_west_fact", "region_east_fact",
                         "region_central_fact", "region_south_fact"],
    },
    {
        "q": "Show profit by segment.",
        "intent": "structured_breakdown",
        "relevant_ids": ["segment_ranked_by_profit", "segment_consumer_fact",
                         "segment_corporate_fact", "segment_home_office_fact"],
    },
    {
        "q": "What are the top 5 sub-categories by profit?",
        "intent": "structured_breakdown",
        "relevant_ids": ["top10_sub_category_profit", "anomaly_loss_subcat_summary"],
    },
    {
        "q": "What is the profit margin by category?",
        "intent": "structured_breakdown",
        "relevant_ids": ["category_ranked_by_profit", "kpi_margin_snapshot",
                         "cross_segment_category_profit"],
    },
    {
        "q": "Show loss-making sub-categories.",
        "intent": "structured_breakdown",
        "relevant_ids": ["anomaly_loss_subcat_summary", "anomaly_high_discount_loss",
                         "top10_sub_category_profit"],
    },
    {
        "q": "Which sub-categories have negative profit?",
        "intent": "structured_breakdown",
        "relevant_ids": ["anomaly_loss_subcat_summary", "anomaly_high_discount_loss"],
    },
    {
        "q": "Show sales and profit by segment in 2017.",
        "intent": "structured_breakdown",
        "relevant_ids": ["segment_ranked_by_sales", "segment_ranked_by_profit",
                         "year_2017_sales_fact"],
    },
    {
        "q": "What is the discount impact on profit by category?",
        "intent": "structured_breakdown",
        "relevant_ids": ["discount_impact", "anomaly_high_discount_loss",
                         "category_ranked_by_profit"],
    },

    # ── Trend / Compare (n=6) ─────────────────────────────────────────────────
    {
        "q": "What is the yearly sales trend from 2014 to 2017?",
        "intent": "trend_compare",
        "relevant_ids": ["trend_overview_sales_yearly", "year_2014_sales_fact",
                         "year_2015_sales_fact", "year_2016_sales_fact", "year_2017_sales_fact"],
    },
    {
        "q": "Compare 2016 vs 2017 total sales.",
        "intent": "trend_compare",
        "relevant_ids": ["trend_year_sales_2016_2017", "year_2016_sales_fact",
                         "year_2017_sales_fact", "trend_overview_sales_yearly"],
    },
    {
        "q": "How did profit change from 2015 to 2016?",
        "intent": "trend_compare",
        "relevant_ids": ["trend_year_sales_2015_2016", "year_2015_sales_fact",
                         "year_2016_sales_fact"],
    },
    {
        "q": "Show monthly sales trend for 2016.",
        "intent": "trend_compare",
        "relevant_ids": ["quarter_2016_annual_summary", "year_2016_sales_fact",
                         "trend_overview_sales_yearly"],
    },
    {
        "q": "What is the year-over-year sales growth?",
        "intent": "trend_compare",
        "relevant_ids": ["trend_overview_sales_yearly", "trend_year_sales_2015_2016",
                         "trend_year_sales_2016_2017"],
    },
    {
        "q": "Compare Q3 2016 vs Q3 2017 revenue.",
        "intent": "trend_compare",
        "relevant_ids": ["quarter_2016_annual_summary", "quarter_2017_annual_summary",
                         "trend_year_sales_2016_2017"],
    },

    # ── Loss / Anomaly (n=5) ──────────────────────────────────────────────────
    {
        "q": "Which products are losing money?",
        "intent": "loss_anomaly",
        "relevant_ids": ["anomaly_loss_subcat_summary", "anomaly_high_discount_loss",
                         "top10_sub_category_profit"],
    },
    {
        "q": "What sub-categories are unprofitable?",
        "intent": "loss_anomaly",
        "relevant_ids": ["anomaly_loss_subcat_summary", "anomaly_high_discount_loss"],
    },
    {
        "q": "Show me all loss-making items and their discounts.",
        "intent": "loss_anomaly",
        "relevant_ids": ["anomaly_loss_subcat_summary", "discount_impact",
                         "anomaly_high_discount_loss"],
    },
    {
        "q": "Which region has the lowest profit margin?",
        "intent": "loss_anomaly",
        "relevant_ids": ["region_ranked_by_profit", "anomaly_low_margin_region_central",
                         "region_central_fact"],
    },
    {
        "q": "Are there any anomalies in discount behaviour?",
        "intent": "loss_anomaly",
        "relevant_ids": ["anomaly_high_discount_loss", "discount_impact",
                         "anomaly_loss_subcat_summary"],
    },

    # ── Informal / Paraphrase (n=5) ───────────────────────────────────────────
    {
        "q": "which products are bleeding money",
        "intent": "informal",
        "relevant_ids": ["anomaly_loss_subcat_summary", "anomaly_high_discount_loss",
                         "top10_sub_category_profit"],
    },
    {
        "q": "which region makes the most cash",
        "intent": "informal",
        "relevant_ids": ["region_ranked_by_sales", "region_west_fact",
                         "region_ranked_by_profit"],
    },
    {
        "q": "what is draining our profitability",
        "intent": "informal",
        "relevant_ids": ["anomaly_high_discount_loss", "anomaly_loss_subcat_summary",
                         "discount_impact"],
    },
    {
        "q": "orders that are hurting us",
        "intent": "informal",
        "relevant_ids": ["anomaly_high_discount_loss", "anomaly_loss_subcat_summary"],
    },
    {
        "q": "which items are making money",
        "intent": "informal",
        "relevant_ids": ["top10_sub_category_profit", "region_ranked_by_profit",
                         "kpi_profit_snapshot"],
    },
]

# ── Retrieve helper ────────────────────────────────────────────────────────────

def retrieve_dense(query, k=6):
    """Use the RAGEngine (dense) to get top-k chunks."""
    ctx = rag.retrieve(query, k=k, inject_examples=False)
    return [c.chunk_id for c in ctx.chunks[:k]]

def retrieve_tfidf(query, k=6):
    """Use TF-IDF fallback on same chunk pool."""
    expanded = hyde.expand(query)
    static_hits  = tfidf_static.retrieve(expanded,  k=k // 2 + 2)
    dynamic_hits = tfidf_dynamic.retrieve(expanded, k=k)
    seen, combined = set(), []
    for c in dynamic_hits + static_hits:
        if c.chunk_id not in seen:
            combined.append(c)
            seen.add(c.chunk_id)
    return [c.chunk_id for c in combined[:k]]

def retrieve_dense_no_hyde(query, k=6):
    """Dense retrieval WITHOUT HyDE expansion."""
    # Temporarily bypass HyDE by passing raw query to retriever directly
    ctx = rag.retrieve(query, k=k, inject_examples=False)
    # We can't easily bypass HyDE inside engine, so we replicate retrieval
    raw_static  = rag._static_retriever.retrieve(query,  k=k // 2 + 2)
    raw_dynamic = rag._dynamic_retriever.retrieve(query, k=k)
    seen, combined = set(), []
    for c in raw_dynamic + raw_static:
        if c.chunk_id not in seen:
            combined.append(c)
            seen.add(c.chunk_id)
    return [c.chunk_id for c in combined[:k]]

def precision_at_k(retrieved_ids, relevant_ids, k=6):
    relevant_set = set(relevant_ids)
    hits = sum(1 for cid in retrieved_ids[:k] if cid in relevant_set)
    return hits / k

# ── Run Precision@6 evaluation ────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 1 — Precision@6: Dense vs TF-IDF (30 queries)")
print("="*80)
print(f"{'Q':<4} {'Intent':<22} {'Dense P@6':>9} {'TFIDF P@6':>10}  {'Delta':>6}  Query")
print("-"*80)

p6_results = []

for item in PRECISION_QUERIES:
    q           = item["q"]
    intent      = item["intent"]
    relevant    = item["relevant_ids"]

    dense_ids = retrieve_dense(q, k=6)
    tfidf_ids = retrieve_tfidf(q, k=6)

    dense_p = precision_at_k(dense_ids, relevant)
    tfidf_p = precision_at_k(tfidf_ids, relevant)
    delta   = dense_p - tfidf_p

    print(f"     {intent:<22} {dense_p:>9.2f} {tfidf_p:>10.2f}  {delta:>+6.2f}  {q[:40]}")
    p6_results.append({
        "query": q, "intent": intent,
        "relevant_ids": relevant,
        "dense_retrieved": dense_ids,
        "tfidf_retrieved": tfidf_ids,
        "dense_p6": round(dense_p, 3),
        "tfidf_p6": round(tfidf_p, 3),
        "delta": round(delta, 3),
    })
    time.sleep(0.1)

# Aggregate by intent group
intent_groups = ["simple_kpi", "structured_breakdown", "trend_compare", "loss_anomaly", "informal"]
intent_labels = {
    "simple_kpi": "Simple KPI / Snapshot",
    "structured_breakdown": "Structured Breakdown / Rank",
    "trend_compare": "Trend / Compare",
    "loss_anomaly": "Loss / Anomaly",
    "informal": "Informal / Paraphrase",
}

print("\nAGGREGATED BY INTENT:")
print(f"  {'Intent':<30} {'n':>3} {'Dense':>7} {'TFIDF':>7} {'Delta':>7}")
print(f"  {'-'*30} {'-'*3} {'-'*7} {'-'*7} {'-'*7}")
agg_p6 = {}
for grp in intent_groups:
    subset = [r for r in p6_results if r["intent"] == grp]
    n = len(subset)
    d_mean = sum(r["dense_p6"] for r in subset) / n if n else 0
    t_mean = sum(r["tfidf_p6"] for r in subset) / n if n else 0
    delta  = d_mean - t_mean
    print(f"  {intent_labels[grp]:<30} {n:>3} {d_mean:>7.3f} {t_mean:>7.3f} {delta:>+7.3f}")
    agg_p6[grp] = {"n": n, "dense_mean": round(d_mean, 3),
                   "tfidf_mean": round(t_mean, 3), "delta": round(delta, 3)}

overall_dense = sum(r["dense_p6"] for r in p6_results) / len(p6_results)
overall_tfidf = sum(r["tfidf_p6"] for r in p6_results) / len(p6_results)
overall_delta = overall_dense - overall_tfidf
print(f"  {'Overall mean P@6':<30} {len(p6_results):>3} {overall_dense:>7.3f} {overall_tfidf:>7.3f} {overall_delta:>+7.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — HyDE expansion impact (5 informal queries)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 2 — HyDE Expansion Impact (informal queries)")
print("="*80)

INFORMAL_QUERIES = [r for r in PRECISION_QUERIES if r["intent"] == "informal"]

hyde_results = []
print(f"{'Query':<45} {'No HyDE':>8} {'HyDE':>8} {'Delta':>7}  Top expansion term")
print("-"*80)

for item in INFORMAL_QUERIES:
    q        = item["q"]
    relevant = item["relevant_ids"]

    # Without HyDE
    no_hyde_ids = retrieve_dense_no_hyde(q, k=6)
    p_no_hyde   = precision_at_k(no_hyde_ids, relevant)

    # With HyDE
    hyde_ids = retrieve_dense(q, k=6)
    p_hyde   = precision_at_k(hyde_ids, relevant)

    # Get expansion term
    expanded    = hyde.expand(q)
    added_terms = expanded[len(q):].strip() if expanded != q else "(none)"
    top_term    = " ".join(added_terms.split()[:4]) if added_terms != "(none)" else "(none)"

    delta = p_hyde - p_no_hyde
    print(f"{q:<45} {p_no_hyde:>8.3f} {p_hyde:>8.3f} {delta:>+7.3f}  {top_term}")
    hyde_results.append({
        "query": q,
        "relevant_ids": relevant,
        "p6_no_hyde": round(p_no_hyde, 3),
        "p6_hyde":    round(p_hyde, 3),
        "delta":      round(delta, 3),
        "expansion_added": added_terms[:80],
    })

hyde_mean_no  = sum(r["p6_no_hyde"] for r in hyde_results) / len(hyde_results)
hyde_mean_yes = sum(r["p6_hyde"]    for r in hyde_results) / len(hyde_results)
hyde_mean_d   = hyde_mean_yes - hyde_mean_no
print(f"{'Mean (informal)':<45} {hyde_mean_no:>8.3f} {hyde_mean_yes:>8.3f} {hyde_mean_d:>+7.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Grounding score (20 agent-style queries)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("SECTION 3 — Grounding Score: with vs without forced pre-query")
print("="*80)

from chatbot.agent.orchestrator import AgentOrchestrator
from chatbot.insight_generator import InsightGenerator
import re as _re

s0 = str(filter_options["min_date"])[:10]
e0 = str(filter_options["max_date"])[:10]

agent = AgentOrchestrator(
    gemini_client=gemini_client,
    model_name=model_name,
    default_start=s0,
    default_end=e0,
)
insight_gen = InsightGenerator(gemini_client, model_name)

AGENT_QUERIES = [
    "Why did profit drop in Q4 2016?",
    "What caused the revenue decline in the Central region?",
    "Why is the Furniture category underperforming?",
    "What drove the sales spike in November 2017?",
    "Why does the South region have lower profit than the West?",
    "What is causing the margin compression in 2016?",
    "Why are Tables and Bookcases losing money?",
    "Why did sales increase but profit decrease from 2015 to 2016?",
    "What caused the high discount rate in the Central region?",
    "Why did orders drop in Q1 2017?",
    "What is driving the profit growth in the West region?",
    "Why is the Home Office segment more profitable than Consumer?",
    "Why did profit decline in 2016 despite revenue growth?",
    "What caused sales to fall in 2017?",
    "Why does heavy discounting hurt profitability?",
    "What is causing loss-making in Tables sub-category?",
    "Why did Q4 always outperform other quarters?",
    "What drove Consumer segment growth?",
    "Why is Technology the most profitable category?",
    "What caused the 2016 to 2017 profit margin improvement?",
]

def _normalize_nums(text):
    nums = set()
    for m in _re.finditer(r'\$?([\d][\d,]*(?:\.\d+)?)', text):
        raw = m.group(1).replace(",", "")
        try: nums.add(round(float(raw), 1))
        except: pass
    for m in _re.finditer(r'([\d]+\.?[\d]*)%', text):
        try: nums.add(round(float(m.group(1)), 1))
        except: pass
    return nums

def grounding_score(answer_text, context_text):
    ans_nums = _normalize_nums(answer_text)
    ctx_nums = _normalize_nums(context_text)
    if not ans_nums: return 1.0
    matched = ans_nums & ctx_nums
    return len(matched) / len(ans_nums)

grounding_results = []
THRESHOLD = 0.40

print(f"\n{'Q':<4} {'Grounded':>9} {'Baseline':>9} {'Pass(G)':>8} {'Pass(B)':>8}  Query")
print("-"*75)

for i, q in enumerate(AGENT_QUERIES, 1):
    print(f"  Running Q{i:02d}: {q[:55]}...")

    # ── GROUNDED: run full agent (uses forced pre-query + RAG) ──────────────
    try:
        answer_grounded = agent.run(q)
        # Build context summary from RAG
        rag_ctx  = rag.retrieve(q, k=8, inject_examples=False)
        ctx_text = " ".join(c.text for c in rag_ctx.chunks)
        score_g  = grounding_score(answer_grounded, ctx_text)
    except Exception as e:
        answer_grounded = f"ERROR: {e}"
        score_g = 0.0

    # ── BASELINE: simulate ungrounded — just ask Gemini directly ─────────────
    try:
        from google.genai import types as genai_types
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=f"You are a business analyst. Answer this question about Superstore data: {q}\nGive specific numbers and percentages.",
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=600),
        )
        answer_baseline = getattr(resp, "text", "") or ""
        score_b = grounding_score(answer_baseline, ctx_text)
    except Exception as e:
        answer_baseline = f"ERROR: {e}"
        score_b = 0.0

    pass_g = score_g >= THRESHOLD
    pass_b = score_b >= THRESHOLD

    print(f"Q{i:02d}  {score_g:>9.3f} {score_b:>9.3f} {'✓' if pass_g else '✗':>8} {'✓' if pass_b else '✗':>8}  {q[:40]}")

    grounding_results.append({
        "id": i, "query": q,
        "score_grounded": round(score_g, 3),
        "score_baseline": round(score_b, 3),
        "pass_grounded": pass_g,
        "pass_baseline": pass_b,
    })

    time.sleep(1.5)  # rate limit buffer

# Aggregate
n  = len(grounding_results)
mean_g  = sum(r["score_grounded"] for r in grounding_results) / n
mean_b  = sum(r["score_baseline"]  for r in grounding_results) / n
pass_g  = sum(r["pass_grounded"]   for r in grounding_results)
pass_b  = sum(r["pass_baseline"]   for r in grounding_results)
fall_g  = n - pass_g   # fallback triggered (score < 0.40)
fall_b  = n - pass_b

print(f"\nGROUNDING SUMMARY (n={n}, threshold={THRESHOLD})")
print(f"  {'Condition':<35} {'Mean Score':>11} {'Pass rate':>10} {'Fallback rate':>14}")
print(f"  {'-'*35} {'-'*11} {'-'*10} {'-'*14}")
print(f"  {'With forced pre-query + RAG':<35} {mean_g:>11.3f} {pass_g/n*100:>9.1f}% {fall_g/n*100:>13.1f}%")
print(f"  {'Without pre-query (baseline)':<35} {mean_b:>11.3f} {pass_b/n*100:>9.1f}% {fall_b/n*100:>13.1f}%")
print(f"  {'Delta':<35} {mean_g-mean_b:>+11.3f} {(pass_g-pass_b)/n*100:>+9.1f}pp {(fall_g-fall_b)/n*100:>+13.1f}pp")

# ─────────────────────────────────────────────────────────────────────────────
# Save output
# ─────────────────────────────────────────────────────────────────────────────

output = {
    "precision_at_6": {
        "per_query": p6_results,
        "by_intent": agg_p6,
        "overall": {
            "n": len(p6_results),
            "dense_mean_p6": round(overall_dense, 3),
            "tfidf_mean_p6": round(overall_tfidf, 3),
            "delta": round(overall_delta, 3),
        }
    },
    "hyde_impact": {
        "per_query": hyde_results,
        "mean_no_hyde": round(hyde_mean_no, 3),
        "mean_hyde":    round(hyde_mean_yes, 3),
        "mean_delta":   round(hyde_mean_d, 3),
    },
    "grounding": {
        "per_query": grounding_results,
        "n": n,
        "threshold": THRESHOLD,
        "grounded": {
            "mean_score": round(mean_g, 3),
            "pass_count": pass_g,
            "pass_rate_pct": round(pass_g/n*100, 1),
            "fallback_rate_pct": round(fall_g/n*100, 1),
        },
        "baseline": {
            "mean_score": round(mean_b, 3),
            "pass_count": pass_b,
            "pass_rate_pct": round(pass_b/n*100, 1),
            "fallback_rate_pct": round(fall_b/n*100, 1),
        },
        "delta": {
            "mean_score": round(mean_g - mean_b, 3),
            "pass_rate_pp": round((pass_g - pass_b) / n * 100, 1),
            "fallback_rate_pp": round((fall_g - fall_b) / n * 100, 1),
        }
    }
}

with open("rag_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

with open("rag_results.txt", "w", encoding="utf-8") as f:
    f.write("RAG EVALUATION RESULTS — Section 5.5\n")
    f.write("="*60 + "\n\n")

    f.write("TABLE 5.9 — Precision@6 by intent\n")
    for grp in intent_groups:
        s = agg_p6[grp]
        f.write(f"  {intent_labels[grp]:<30} n={s['n']}  dense={s['dense_mean']:.3f}  tfidf={s['tfidf_mean']:.3f}  delta={s['delta']:+.3f}\n")
    f.write(f"  {'Overall mean P@6':<30} n={len(p6_results)}  dense={overall_dense:.3f}  tfidf={overall_tfidf:.3f}  delta={overall_delta:+.3f}\n\n")

    f.write("TABLE 5.10 — HyDE impact (informal queries)\n")
    for r in hyde_results:
        f.write(f"  p6_no_hyde={r['p6_no_hyde']:.3f}  p6_hyde={r['p6_hyde']:.3f}  delta={r['delta']:+.3f}  q={r['query']}\n")
    f.write(f"  MEAN: no_hyde={hyde_mean_no:.3f}  hyde={hyde_mean_yes:.3f}  delta={hyde_mean_d:+.3f}\n\n")

    f.write("TABLE 5.11 — Grounding score\n")
    f.write(f"  WITH pre-query:    mean={mean_g:.3f}  pass={pass_g/n*100:.1f}%  fallback={fall_g/n*100:.1f}%\n")
    f.write(f"  WITHOUT pre-query: mean={mean_b:.3f}  pass={pass_b/n*100:.1f}%  fallback={fall_b/n*100:.1f}%\n")
    f.write(f"  DELTA:             mean={mean_g-mean_b:+.3f}  pass={( pass_g-pass_b)/n*100:+.1f}pp  fallback={(fall_g-fall_b)/n*100:+.1f}pp\n")

print("\nSaved → rag_results.txt  rag_results.json")
print("="*60)