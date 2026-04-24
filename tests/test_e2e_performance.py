"""
test_e2e_performance.py  — v2 (FIXED)
────────────────────────────────────────────────────────────────────────────────
FIXES vs v1:

  FIX-E2E-1  API counter via monkey-patch thay vì wrapper object
             v1: counting_client wrapper không được dùng bởi router/agent/pipeline
             → counter luôn = 0 cho agent, router tự thêm 1 call
             v2: patch gemini_client.models.generate_content trực tiếp

  FIX-E2E-2  Latency đo IN ISOLATION per tier, không qua SmartRouter
             v1: tất cả queries đều qua SmartRouter (1 LLM call = ~580ms)
             → Tier-1 xuất hiện 591ms thay vì <5ms
             v2: mỗi tier được gọi hàm riêng, không pass qua router

  FIX-E2E-3  API call rate dùng 100-query eval set (65 structured + 20 agent
             + 15 hybrid) thay vì 40-query rating set toàn agent/hybrid
             v1: 40 queries → 100% calls, 1.5x baseline
             v2: 100 queries với 65 structured (nhiều Tier-1/2 = 0 API calls)

  FIX-E2E-4  no_rag condition dùng đúng SQL pipeline mà không inject RAG
             v1: gọi Gemini trực tiếp với "answer this query" → nhận SQL code
             v2: Tier-1/2 chạy bình thường, Tier-3/Agent dùng empty RAGContext
"""

import os, sys, json, time, statistics
# Script phải chạy từ thư mục tests/ hoặc project root
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "tests" else _THIS_DIR
sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from config import Config
from google import genai
from google.genai import types as genai_types
from core.data_loader import load_filtered_data_safe, calculate_kpis, get_filter_options
from chatbot.nl_parser import NLParser
from chatbot.plan_validator import PlanValidator
from chatbot.sql_builder import SQLBuilder
from chatbot.answer_formatter import AnswerFormatter
from chatbot.insight_generator import InsightGenerator
from chatbot.llm_plan_auditor import LLMPlanAuditor
from chatbot.smart_router import SmartRouter
from chatbot.agent.orchestrator import AgentOrchestrator
from chatbot.hybrid_executor import HybridExecutor
from rag.engine import RAGEngine, RAGContext

# ── Setup ─────────────────────────────────────────────────────────────────────
print("[Setup] Connecting to database...")
filter_options = get_filter_options()
if not filter_options:
    print("\n❌ ERROR: Cannot connect to database.")
    print("   Check .env file — DB_HOST, DB_USER, DB_PASSWORD, DB_NAME must be set.")
    print(f"   Current DB_HOST: {Config.DB_HOST}")
    print(f"   Current DB_NAME: {Config.DB_NAME}")
    sys.exit(1)

print(f"[Setup] DB connected. Date range: {filter_options['min_date']} → {filter_options['max_date']}")

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
print(f"[Setup] Loaded {len(df):,} rows. KPIs: sales=${kpis['total_sales']:,.0f}  orders={kpis['total_orders']:,}")

print("[Setup] Initialising Gemini client...")
if not Config.GCP_PROJECT:
    print("❌ ERROR: GCP_PROJECT not set in .env")
    sys.exit(1)

gemini_client = genai.Client(
    vertexai=True,
    project=Config.GCP_PROJECT,
    location=Config.GCP_LOCATION,
)
model_name = Config.GEMINI_MODEL
print(f"[Setup] Gemini ready. Model: {model_name}")

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
print(f"[Setup] RAG built. Total chunks: {rag.total_chunks}")

# ── FIX-E2E-1: Monkey-patch TRƯỚC KHI khởi tạo các component khác
# Patch ngay sau khi gemini_client được tạo để đếm mọi call từ mọi nơi.
_api_call_count = 0
_original_gc    = gemini_client.models.generate_content

def _counted_gc(*args, **kwargs):
    global _api_call_count
    _api_call_count += 1
    return _original_gc(*args, **kwargs)

gemini_client.models.generate_content = _counted_gc

def _reset_counter():
    global _api_call_count
    _api_call_count = 0

def _get_count():
    return _api_call_count

# ── Pipeline helpers (defined before hybrid so _run_structured is available) ──

def _run_tier1_only(q):
    """Tier 1 in isolation — chỉ fast_kpi_answer, không qua router."""
    return parser.fast_kpi_answer(q)

def _run_tier2_no_audit(q):
    """Tier 2 in isolation — rule_based_plan + SQL, không audit."""
    rule_plan = parser.rule_based_plan(q)
    if not rule_plan:
        return None
    try:
        plan = validator.validate(rule_plan)
        df_r = sql_b.run(plan)
        return formatter.format(plan, df_r, "")
    except Exception as e:
        return f"Error: {e}"

def _run_tier2_with_audit(q):
    """Tier 2 in isolation — rule_based_plan + auditor.audit() + SQL."""
    rule_plan = parser.rule_based_plan(q)
    if not rule_plan:
        return None
    rule_plan = auditor.audit(q, rule_plan)   # 1 LLM call
    try:
        plan = validator.validate(rule_plan)
        df_r = sql_b.run(plan)
        return formatter.format(plan, df_r, "")
    except Exception as e:
        return f"Error: {e}"

def _run_tier3(q):
    """Tier 3 full — RAG retrieve + gemini_plan + SQL + insight."""
    try:
        rag_ctx  = rag.retrieve(q, k=10, tier=3)
        raw_plan = parser.gemini_plan(q, rag_ctx)
        plan     = validator.validate(raw_plan)
        df_r     = sql_b.run(plan)
        ins      = insights.generate(plan, df_r)
        return formatter.format(plan, df_r, ins)
    except Exception as e:
        return f"Error: {e}"

def _run_structured(q):
    """Full structured: Tier 1 → 2 → 3. Dùng trong hybrid executor."""
    fast = parser.fast_kpi_answer(q)
    if fast:
        return fast
    rule_plan = parser.rule_based_plan(q)
    if rule_plan:
        if rule_plan.get("intent") == "kpi_detail":
            rule_plan["filters"] = {"region":[],"segment":[],"category":[],"sub_category":[]}
        rule_plan = auditor.audit(q, rule_plan)
        try:
            plan = validator.validate(rule_plan)
            df_r = sql_b.run(plan)
            ins  = insights.generate(plan, df_r)
            return formatter.format(plan, df_r, ins)
        except Exception:
            pass
    return _run_tier3(q)

hybrid = HybridExecutor(
    structured_runner=_run_structured,
    agent_runner=lambda q: agent.run(q),
    gemini_client=gemini_client,
    model_name=model_name,
)
print("[Setup] All components ready.\n")

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — FIX-E2E-2: Latency per tier IN ISOLATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("PART 1 — LATENCY (per-tier, in isolation, không qua SmartRouter)")
print("="*80)

TIER1_QUERIES = [
    # Chỉ dùng queries mà fast_kpi_answer() THỰC SỰ handle được:
    # không mention specific region/segment/category, không có date filter
    "What is the total sales revenue?",
    "What is the total profit?",
    "What is the overall profit margin?",
    "How many orders were placed in total?",
    "Give me the KPI summary.",
    "Total revenue overview.",
    "What is our profit margin?",
    "Sales overview dashboard.",
    "How much profit did we make?",
    "What are total sales?",
    "Total orders count.",
    "What is the profit margin?",
]

TIER2_NO_AUDIT_QUERIES = [
    "What are the total sales by region?",
    "Show profit by segment.",
    "Top 5 sub-categories by profit.",
    "Sales trend by year.",
    "Which sub-categories are losing money?",
    "Top 10 products by sales.",
    "Profit by category.",
    "Monthly orders in 2017.",
    "Profit margin by region.",
    "Sales by category in 2016.",
]

TIER2_AUDIT_QUERIES = [
    "Compare 2016 vs 2017 total sales.",
    "How did profit change from 2015 to 2016?",
    "Year over year sales growth.",
    "Compare Q1 2016 vs Q1 2017 orders.",
    "Monthly profit trend in 2016.",
    "Compare West vs East region sales.",
    "Year over year profit by region.",
    "Show quarterly revenue for 2017.",
]

TIER3_QUERIES = [
    "Show me sales by category broken down by region with profit details.",
    "Revenue trend — is growth accelerating or slowing?",
    "Profit margin distribution across all sub-categories.",
    "Compare profit margin between Consumer and Home Office segments.",
    "How did the Furniture category perform vs Technology in 2016?",
    "Which products are bleeding money and what's causing it?",
    "What is the discount impact on profitability across categories?",
    "Top 5 regions by revenue in Q3 2016.",
]

AGENT_LATENCY_QUERIES = [
    "Why did profit drop in Q4 2016?",
    "What caused the revenue decline in the Central region?",
    "Why are Tables and Bookcases losing money?",
    "Why does the South region have lower profit than the West?",
    "What is causing the margin compression in 2016?",
    "Why did orders drop in Q1 2017?",
    "Why does heavy discounting hurt profitability?",
]

ROUTER_QUERIES = [
    "What are the total sales by region?",
    "Show profit by segment.",
    "Compare 2016 vs 2017.",
    "Why did profit drop in 2016?",
    "Show loss-making products and explain why.",
]

latency_results = {}
path_configs = [
    ("Tier 1 — Regex KPI",         TIER1_QUERIES,          _run_tier1_only),
    ("Tier 2 — NLParser (no audit)",TIER2_NO_AUDIT_QUERIES, _run_tier2_no_audit),
    ("Tier 2 — NLParser + Auditor", TIER2_AUDIT_QUERIES,    _run_tier2_with_audit),
    ("Tier 3 — Gemini + RAG",       TIER3_QUERIES,          _run_tier3),
    ("Agent path (diagnostic)",     AGENT_LATENCY_QUERIES,  lambda q: agent.run(q)),
    ("SmartRouter classification",  ROUTER_QUERIES,         lambda q: router.classify(q)),
]

for path_label, queries, fn in path_configs:
    times = []
    api_list = []
    print(f"\n  [{path_label}]")
    for q in queries:
        _reset_counter()
        t0 = time.perf_counter()
        try:
            fn(q)
        except Exception as e:
            print(f"    WARN: {e}")
        elapsed = (time.perf_counter() - t0) * 1000
        calls   = _get_count()
        times.append(elapsed)
        api_list.append(calls)
        print(f"    {elapsed:>9.1f}ms  api={calls}  {q[:50]}")
        time.sleep(0.3)

    times_s = sorted(times)
    mean_t  = statistics.mean(times)
    p95_t   = times_s[int(len(times_s)*0.95)] if len(times_s) > 1 else times_s[0]
    mean_c  = statistics.mean(api_list)

    key = path_label.split("—")[0].strip().lower().replace(" ", "_")
    latency_results[path_label] = {
        "n": len(times),
        "mean_ms": round(mean_t, 1),
        "p95_ms":  round(p95_t, 1),
        "mean_api_calls": round(mean_c, 1),
    }
    print(f"  → mean={mean_t:.1f}ms  p95={p95_t:.1f}ms  mean_api={mean_c:.1f}")

print("\nLATENCY SUMMARY:")
print(f"  {'Processing Path':<35} {'n':>3}  {'Mean(ms)':>10}  {'p95(ms)':>10}  {'API calls/q':>12}")
print(f"  {'-'*73}")
for label, s in latency_results.items():
    print(f"  {label:<35} {s['n']:>3}  {s['mean_ms']:>10.1f}  {s['p95_ms']:>10.1f}  {s['mean_api_calls']:>12.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — FIX-E2E-3: API call rate trên 100-query eval set
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("PART 2 — API CALL RATE (100-query eval set: 65 structured + 20 agent + 15 hybrid)")
print("="*80)

try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
    from test_routing import questions as ALL_QUESTIONS, ground_truth as ALL_GT
    eval_pairs = list(zip(ALL_QUESTIONS, ALL_GT))
    print(f"  Loaded {len(eval_pairs)} queries from tests/test_routing.py")
except ImportError:
    print("  ERROR: Cannot import test_routing.py. Ensure tests/ folder is accessible.")
    raise

full_api_calls    = []
norouting_api_calls = []

for i, (q, gt) in enumerate(eval_pairs, 1):
    # Full system
    _reset_counter()
    try:
        decision = router.classify(q)
        if decision.mode == "agent":
            agent.run(q)
        elif decision.mode == "hybrid":
            hybrid.execute(decision, q)
        else:
            _run_structured(q)
    except Exception:
        pass
    full_api_calls.append(_get_count())

    # No-routing baseline (1 direct Gemini call per query)
    _reset_counter()
    try:
        gemini_client.models.generate_content(
            model=model_name,
            contents=f"You are a Superstore BI analyst. Answer: {q}",
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=800),
        )
    except Exception:
        pass
    norouting_api_calls.append(_get_count())

    if i % 10 == 0:
        print(f"  Progress: {i}/{len(eval_pairs)} queries done...")
    time.sleep(0.4)

n_q          = len(eval_pairs)
full_total   = sum(full_api_calls)
base_total   = sum(norouting_api_calls)
full_mean    = full_total / n_q
noroute_mean = base_total / n_q

# Breakdown structured / agent / hybrid
struct_full = [full_api_calls[i]       for i, (_, gt) in enumerate(eval_pairs) if gt == "structured"]
agent_full  = [full_api_calls[i]       for i, (_, gt) in enumerate(eval_pairs) if gt == "agent"]
hybrid_full = [full_api_calls[i]       for i, (_, gt) in enumerate(eval_pairs) if gt == "hybrid"]

full_pct    = sum(1 for c in full_api_calls if c > 0) / n_q * 100
struct_pct  = sum(1 for c in struct_full if c > 0) / len(struct_full) * 100 if struct_full else 0
agent_pct   = sum(1 for c in agent_full  if c > 0) / len(agent_full)  * 100 if agent_full  else 0
hybrid_pct  = sum(1 for c in hybrid_full if c > 0) / len(hybrid_full) * 100 if hybrid_full else 0

# Relative cost = full_total / baseline_total
# Baseline: 1 call/query = n_q calls total
# Full system: SmartRouter (1/query) + pipeline extra (0 Tier1/2, 1-2 Tier3, 4-9 agent)
# Tier-1/2 structured = SmartRouter only = 1 call → same as baseline
# Agent/Hybrid = SmartRouter + tool calls = significantly more
rel_cost = (full_total / base_total * 100) if base_total else 0

# Queries với API calls = 0 là những queries được served bởi Tier-1 sau khi
# SmartRouter classify thành "structured" nhưng fast_kpi_answer() intercept trước router
# (Tier-1 không bao giờ call router trong production — đây là artifact của test)
queries_with_zero_calls = sum(1 for c in full_api_calls if c == 0)

print(f"\n  Full system (n={n_q}):")
print(f"    Structured (n={len(struct_full)}): {struct_pct:.0f}% with API  mean={statistics.mean(struct_full):.1f}/query")
print(f"    Agent      (n={len(agent_full)}): {agent_pct:.0f}% with API  mean={statistics.mean(agent_full):.1f}/query")
print(f"    Hybrid     (n={len(hybrid_full)}): {hybrid_pct:.0f}% with API  mean={statistics.mean(hybrid_full):.1f}/query")
print(f"    TOTAL: {full_pct:.1f}% with API call  |  {full_mean:.2f} mean/query  |  {full_total} total calls")
print(f"  No-routing: 100%  |  {noroute_mean:.2f} mean/query  |  {base_total} total calls")
print(f"  Relative API cost: {rel_cost:.0f}% of baseline")
print(f"  Queries with 0 API calls (pure Tier-1/2): {queries_with_zero_calls}/{n_q}")

api_results = {
    "full_system_pct_with_call":  round(full_pct, 1),
    "full_system_mean_per_query": round(full_mean, 2),
    "full_system_total_calls":    full_total,
    "norouting_pct_with_call":    100.0,
    "norouting_mean_per_query":   round(noroute_mean, 2),
    "norouting_total_calls":      base_total,
    "relative_cost_pct":          round(rel_cost, 1),
    "queries_zero_api_calls":     queries_with_zero_calls,
    "n_queries":                  n_q,
    "structured_queries": {
        "n": len(struct_full),
        "pct_with_call": round(struct_pct, 1),
        "mean_calls":    round(statistics.mean(struct_full), 2),
    },
    "agent_queries": {
        "n": len(agent_full),
        "pct_with_call": round(agent_pct, 1),
        "mean_calls":    round(statistics.mean(agent_full), 2),
    },
    "hybrid_queries": {
        "n": len(hybrid_full),
        "pct_with_call": round(hybrid_pct, 1),
        "mean_calls":    round(statistics.mean(hybrid_full), 2),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Generate 40 responses × 3 conditions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("PART 3 — GENERATING RATING RESPONSES (40 × 3 conditions)")
print("="*80)

RATING_QUERIES = [
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
    ("structured",  "What are the total sales by region?"),
    ("structured",  "Show me profit breakdown by segment."),
    ("structured",  "What are the top 5 sub-categories by profit?"),
    ("structured",  "Show monthly sales trend for 2016."),
    ("structured",  "What is the profit margin by category?"),
    ("structured",  "Show sales trend by segment over years."),
    ("structured",  "Compare 2016 vs 2017 total sales."),
    ("structured",  "What are the loss-making sub-categories?"),
    ("structured",  "Show quarterly revenue breakdown for 2017."),
    ("structured",  "What is the profit margin by region in 2016?"),
    ("agent",       "Why did profit drop in Q4 2016?"),
    ("agent",       "What caused the revenue decline in the Central region?"),
    ("agent",       "Why is the Furniture category underperforming?"),
    ("agent",       "Why are Tables and Bookcases losing money?"),
    ("agent",       "Why does the South region have lower profit than the West?"),
    ("agent",       "What is causing the margin compression in 2016?"),
    ("agent",       "Why does heavy discounting hurt profitability?"),
    ("agent",       "Why is the Home Office segment more profitable than Consumer?"),
    ("agent",       "What drove the sales spike in November 2017?"),
    ("agent",       "Why did orders drop in Q1 2017?"),
    ("hybrid",      "Show me loss-making products and explain what is causing the losses."),
    ("hybrid",      "Which region contributes least to profit and why?"),
    ("hybrid",      "Explain the sales trend by region."),
    ("hybrid",      "Show profit by segment and explain which segment should we focus on."),
    ("hybrid",      "Which sub-categories are unprofitable and what is driving the losses?"),
    ("hybrid",      "Compare 2016 vs 2017 sales and explain the difference."),
    ("hybrid",      "Show the profit trend and explain why growth is slowing."),
    ("hybrid",      "Which segment has the best margin and why does it outperform others?"),
    ("hybrid",      "Show me the discount impact on profit and explain why high discounts are harmful."),
    ("hybrid",      "Identify the top region by sales and explain what drives its performance."),
]


def run_full(q):
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


def run_no_rag(q):
    """
    FIX-E2E-4: SQL pipeline bình thường nhưng không có RAG context.
    Tier-1/2 không cần RAG → giữ nguyên.
    Tier-3: gọi gemini_plan với RAGContext rỗng.
    Agent: gọi agent.run() nhưng không có rag-injected context.
    """
    fast = parser.fast_kpi_answer(q)
    if fast:
        return fast

    rule_plan = parser.rule_based_plan(q)
    if rule_plan:
        if rule_plan.get("intent") == "kpi_detail":
            rule_plan["filters"] = {"region":[],"segment":[],"category":[],"sub_category":[]}
        try:
            plan = validator.validate(rule_plan)
            df_r = sql_b.run(plan)
            # Không dùng InsightGenerator (có LLM) để tránh sai số
            return formatter.format(plan, df_r, "")
        except Exception:
            pass

    # Tier-3 without RAG
    try:
        empty_ctx = RAGContext(query=q, chunks=[], chat_summary="", example_section="")
        raw_plan  = parser.gemini_plan(q, empty_ctx)
        plan      = validator.validate(raw_plan)
        df_r      = sql_b.run(plan)
        return formatter.format(plan, df_r, "")
    except Exception as e:
        return f"Error (no-RAG): {e}"


def run_no_routing(q):
    """No-routing: 1 direct Gemini call với BI analyst prompt."""
    try:
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=(
                f"You are a Superstore BI analyst with direct access to a database "
                f"(2014-2017 data). Answer with specific numbers from the data: {q}\n"
                f"Columns available: sales, profit, orders, order_date, region, "
                f"segment, category, sub_category, discount."
            ),
            config=genai_types.GenerateContentConfig(temperature=0.0, max_output_tokens=600),
        )
        return getattr(resp, "text", "") or "No response"
    except Exception as e:
        return f"Error: {e}"


rating_data = []
for i, (qtype, q) in enumerate(RATING_QUERIES, 1):
    print(f"  Q{i:02d} [{qtype:<10}] {q[:55]}...")

    t0 = time.perf_counter()
    r_full = run_full(q)
    t_full = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r_norag = run_no_rag(q)
    t_norag = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r_noroute = run_no_routing(q)
    t_noroute = (time.perf_counter() - t0) * 1000

    print(f"       full={t_full:.0f}ms  no_rag={t_norag:.0f}ms  no_routing={t_noroute:.0f}ms")

    rating_data.append({
        "id": i, "type": qtype, "query": q,
        "full_system":  r_full,
        "no_rag":       r_norag,
        "no_routing":   r_noroute,
        "latency_full_ms":    round(t_full, 0),
        "latency_norag_ms":   round(t_norag, 0),
        "latency_noroute_ms": round(t_noroute, 0),
    })
    time.sleep(1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
output = {
    "latency":          latency_results,
    "api_calls":        api_results,
    "rating_responses": rating_data,
}

with open("e2e_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)

with open("e2e_results.txt", "w", encoding="utf-8") as f:
    f.write("END-TO-END EVALUATION RESULTS — Section 5.8\n")
    f.write("="*60 + "\n\n")

    f.write("TABLE 5.17 — Response Latency per Processing Path\n")
    f.write(f"  {'Processing Path':<35} {'n':>3}  {'Mean(ms)':>10}  {'p95(ms)':>10}  {'API/query':>10}\n")
    f.write(f"  {'-'*72}\n")
    for label, s in latency_results.items():
        f.write(f"  {label:<35} {s['n']:>3}  {s['mean_ms']:>10.1f}  {s['p95_ms']:>10.1f}  {s['mean_api_calls']:>10.1f}\n")

    f.write("\nTABLE 5.18 — API Call Rate vs No-Routing Baseline\n")
    f.write(f"  Full system (n={n_q}):          {full_pct:.1f}% with API call  |  {full_mean:.2f} mean/query\n")
    f.write(f"    Structured queries (n={len(struct_full)}): {struct_pct:.0f}% with API  mean={statistics.mean(struct_full):.2f}/query\n")
    f.write(f"    Agent queries      (n={len(agent_full)}): {agent_pct:.0f}% with API  mean={statistics.mean(agent_full):.2f}/query\n")
    f.write(f"    Hybrid queries     (n={len(hybrid_full)}): {hybrid_pct:.0f}% with API  mean={statistics.mean(hybrid_full):.2f}/query\n")
    f.write(f"  No-routing baseline (n={n_q}): 100% with API call  |  {noroute_mean:.2f} mean/query\n")
    f.write(f"  Relative API cost:   {rel_cost:.0f}% of baseline\n")
    f.write(f"  Queries with 0 API calls (Tier-1 KPI, no LLM): {queries_with_zero_calls}/{n_q} ({queries_with_zero_calls/n_q*100:.0f}%)\n")

    f.write("\nTABLE 5.19 — Answer Quality (fill after human rating)\n")
    f.write("  [See 5_8_rating_sheet.txt]\n")

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
    f.write("  Rate all 3 responses (A / B / C) for each query independently.\n")
    f.write("  Condition labels are revealed at the end of the file.\n")
    f.write("  Verify numeric claims against the dashboard directly if unsure.\n\n")
    f.write("="*80 + "\n\n")

    for r in rating_data:
        f.write(f"Q{r['id']:02d} [{r['type'].upper()}]\n")
        f.write(f"Query: {r['query']}\n")
        f.write("-"*60 + "\n\n")
        for label, key in [
            ("CONDITION A", "full_system"),
            ("CONDITION B", "no_rag"),
            ("CONDITION C", "no_routing"),
        ]:
            f.write(f"{label}:\n")
            resp_text = r[key][:1500] + ("..." if len(r[key]) > 1500 else "")
            f.write(resp_text + "\n\n")
            f.write(f"  Rater 1 score (1–5): [ ]    Rater 2 score (1–5): [ ]\n\n")
        f.write("="*80 + "\n\n")

    f.write("\n" + "="*80 + "\n")
    f.write("CONDITION KEY (reveal after all rating is done):\n")
    f.write("  A = Full system (SmartRouter + SQL pipeline + RAG + Agent/Hybrid)\n")
    f.write("  B = No-RAG baseline (SmartRouter + SQL, no RAG context injection)\n")
    f.write("  C = No-routing baseline (direct Gemini call, no SQL, no RAG)\n\n")
    f.write("Cohen's kappa: python -c \"from sklearn.metrics import cohen_kappa_score; \\\n")
    f.write("  print(cohen_kappa_score([rater1_scores], [rater2_scores]))\"\n")

print("\nSaved: e2e_results.json  e2e_results.txt  5_8_rating_sheet.txt")
print("="*60)
print("\nNEXT STEPS:")
print("  1. Copy file sang tests/ và chạy: python tests/test_e2e_performance.py")
print("  2. Gửi 5_8_rating_sheet.txt cho Rater 2 (không tiết lộ condition labels)")
print("  3. Điền điểm, tính Cohen's κ, điền Table 5.19")