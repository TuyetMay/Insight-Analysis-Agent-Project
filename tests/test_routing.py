"""
tests/test_routing.py
────────────────────────────────────────────────────────────────────────────────
So sánh SmartRouter (LLM-based) vs QueryRouter (regex-based).

Cách chạy:
    cd /path/to/project
    python tests/test_routing.py          # dùng SmartRouter nếu có GCP_PROJECT
    python tests/test_routing.py --regex  # chỉ chạy regex, không cần GCP

Nếu GCP_PROJECT không được cấu hình trong .env, script tự động
fallback về chế độ regex-only (không lỗi).
"""

import os
import sys
import argparse

# ── Path setup: chạy từ bất kỳ thư mục nào cũng được ─────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# Load .env từ project root

from config import Config
from chatbot.query_router import QueryRouter

# ── Parse args ────────────────────────────────────────────────────────────────
parser_args = argparse.ArgumentParser()
parser_args.add_argument("--regex", action="store_true",
                         help="Chỉ chạy regex router, bỏ qua SmartRouter")
args, _ = parser_args.parse_known_args()

# ── Khởi tạo SmartRouter nếu GCP_PROJECT có sẵn ──────────────────────────────
smart_router  = None
use_smart     = False
gemini_client = None

if not args.regex and Config.GCP_PROJECT:
    try:
        from google import genai
        from chatbot.smart_router import SmartRouter

        gemini_client = genai.Client(
            vertexai=True,
            project=Config.GCP_PROJECT,
            location=Config.GCP_LOCATION,
        )
        smart_router = SmartRouter(
            gemini_client=gemini_client,
            model_name=Config.GEMINI_MODEL,
        )
        use_smart = True
        print(f"✅ SmartRouter khởi tạo thành công (project={Config.GCP_PROJECT})")
    except Exception as e:
        print(f"⚠️  Không thể khởi tạo SmartRouter: {e}")
        print("   → Chỉ chạy regex router.\n")
else:
    if args.regex:
        print("ℹ️  Chế độ --regex: bỏ qua SmartRouter.")
    else:
        print("ℹ️  GCP_PROJECT không được cấu hình → chỉ chạy regex router.")

regex_router = QueryRouter()

# ── Dataset ───────────────────────────────────────────────────────────────────
questions = [
    # SIMPLE KPI (1-20) — GT: structured
    "What is the total sales revenue?",
    "What is the total profit?",
    "How many orders were placed in total?",
    "What is the overall profit margin?",
    "What is the average order value?",
    "What is the total revenue for 2016?",
    "How many orders were placed in 2017?",
    "What is the total profit in 2015?",
    "What is the profit margin in 2014?",
    "What is the total sales in Q4 2016?",
    "How much revenue did we generate in January 2017?",
    "What is the total number of orders in the West region?",
    "What is the total profit for the Consumer segment?",
    "What is the total sales for the Technology category?",
    "How many orders did the South region place in 2015?",
    "What is the profit margin for the Corporate segment?",
    "What is the total revenue for Furniture in 2017?",
    "How many orders were placed in the East region in 2016?",
    "What is the total profit for Office Supplies?",
    "What is the overall average discount rate?",
    # STRUCTURED BREAKDOWN (21-45) — GT: structured
    "What are the total sales by region?",
    "Show me profit breakdown by segment.",
    "What is the revenue by category in 2016?",
    "Show sales by region for each year.",
    "What are the top 5 sub-categories by profit?",
    "Show me the top 10 products by sales.",
    "What is the profit margin by category?",
    "Show sales and profit by segment in 2017.",
    "What are the top 5 regions by revenue in Q3 2016?",
    "Show monthly sales trend for 2016.",
    "What is the quarterly revenue breakdown for 2017?",
    "Show profit by region and segment.",
    "What are the top 3 segments by profit margin?",
    "Show me sales by category broken down by region.",
    "What is the yearly sales trend from 2014 to 2017?",
    "Show top 10 sub-categories by sales in the West region.",
    "What is the profit breakdown by sub-category for Furniture?",
    "Show monthly orders trend in 2017.",
    "What are the loss-making sub-categories?",
    "Show me which products are losing money.",
    "What sub-categories have negative profit?",
    "Show sales trend by segment over years.",
    "What is the profit margin by region in 2016?",
    "Show top 5 categories by revenue growth.",
    "What is the discount impact on profit by category?",
    # TREND / COMPARE (46-65) — GT: structured
    "Compare 2016 vs 2017 total sales.",
    "How did profit change from 2015 to 2016?",
    "Compare Q3 2016 vs Q3 2017 revenue.",
    "What is the year-over-year sales growth?",
    "How does profit in 2017 compare to 2016?",
    "Compare the West region vs East region sales in 2016.",
    "Show month-over-month revenue change in 2017.",
    "How did the Consumer segment perform compared to Corporate in 2016?",
    "Compare Technology vs Furniture profit in 2017.",
    "What is the sales growth rate from 2014 to 2017?",
    "How did profit margin change from 2015 to 2017?",
    "Compare Q1 2016 vs Q1 2017 orders.",
    "Show year-over-year profit growth by region.",
    "How did discount levels change from 2015 to 2016?",
    "Compare the top region vs bottom region sales in 2017.",
    "How did orders in the South region change from 2016 to 2017?",
    "Compare profit margin between Consumer and Home Office segments.",
    "Show the revenue trend — is growth accelerating or slowing?",
    "How did the Furniture category perform vs Technology in 2016?",
    "Compare October 2016 vs October 2015 total sales.",
    # DIAGNOSTIC / AGENT (66-85) — GT: agent
    "Why did profit drop in Q4 2016?",
    "What caused the revenue decline in the Central region?",
    "Why is the Furniture category underperforming?",
    "What drove the sales spike in November 2017?",
    "Why does the South region have lower profit than the West?",
    "What is causing the margin compression in 2016?",
    "Why are Tables and Bookcases losing money?",
    "What should we do to improve profit margin?",
    "Is the current profit margin healthy?",
    "Why did sales increase but profit decrease from 2015 to 2016?",
    "What caused the high discount rate in the Central region?",
    "Why did orders drop in Q1 2017?",
    "What is driving the profit growth in the West region?",
    "Should we stop selling Bookcases given the losses?",
    "Why is the Home Office segment more profitable than Consumer?",
    "What would happen if we capped discounts at 20%?",
    "Why did profit decline in 2016 despite revenue growth?",
    "What caused sales to fall in 2017?",
    "Is it worth expanding into the South region?",
    "Why does heavy discounting hurt profitability?",
    # HYBRID (86-100) — GT: hybrid
    "Show me loss-making products and explain what is causing the losses.",
    "Which region contributes least to profit and why?",
    "Explain the sales trend by region over the years.",
    "Show profit by segment and explain which segment should we focus on.",
    "Which sub-categories are unprofitable and what is driving the losses?",
    "Compare 2016 vs 2017 sales and explain the difference.",
    "Show the profit trend and explain why growth is slowing.",
    "Which segment has the best margin and why does it outperform others?",
    "Show me the discount impact on profit and explain why high discounts are harmful.",
    "Identify the top region by sales and explain what drives its performance.",
    "Show loss-making products in the Central region and explain the root cause.",
    "Compare Consumer vs Corporate profit and explain the performance gap.",
    "Show the quarterly revenue breakdown for 2016 and explain the Q4 spike.",
    "Which categories are most profitable and explain what drives that profitability.",
    "Show me the sales trend in the West region and explain whether momentum is sustainable.",
]

ground_truth = (
    ["structured"] * 20 +  # Simple KPI
    ["structured"] * 25 +  # Structured Breakdown
    ["structured"] * 20 +  # Trend/Compare
    ["agent"]      * 20 +  # Diagnostic
    ["hybrid"]     * 15    # Hybrid
)

# ── Chạy evaluation ───────────────────────────────────────────────────────────
header_cols = "Q_ID | GT         | REGEX"
if use_smart:
    header_cols = "Q_ID | GT         | SMART      | REGEX"

print()
print(f"{'Q':>4}  {header_cols:<60}  Question (truncated)")
print("-" * 100)

results = []
for i, (q, gt) in enumerate(zip(questions, ground_truth), 1):
    regex_mode = regex_router.route(q)

    # regex chỉ có structured/agent — map hybrid → agent cho fair comparison
    regex_correct = (regex_mode == gt) or (gt == "hybrid" and regex_mode == "agent")

    if use_smart:
        try:
            smart_decision = smart_router.classify(q)
            smart_mode     = smart_decision.mode
        except Exception as e:
            smart_mode = "structured"  # fallback an toàn
            print(f"  ⚠️  SmartRouter lỗi Q{i}: {e}")

        smart_correct = (smart_mode == gt)
        print(f"Q{i:02d}  | {gt:<10} | {smart_mode:<10} | {regex_mode:<10}  {q[:45]}")
    else:
        smart_mode    = None
        smart_correct = None
        print(f"Q{i:02d}  | {gt:<10} | {regex_mode:<10}  {q[:45]}")

    results.append({
        "id": i, "q": q, "gt": gt,
        "smart": smart_mode,
        "regex": regex_mode,
        "smart_correct": smart_correct,
        "regex_correct": regex_correct,
    })

# ── Tổng kết ──────────────────────────────────────────────────────────────────
n           = len(results)
regex_total = sum(r["regex_correct"] for r in results)

print(f"\n{'='*60}")
print("OVERALL ACCURACY")
print(f"{'='*60}")

if use_smart:
    smart_total = sum(r["smart_correct"] for r in results)
    print(f"  SmartRouter : {smart_total}/{n} = {smart_total/n*100:.1f}%")

print(f"  RegexRouter : {regex_total}/{n} = {regex_total/n*100:.1f}%")

# Per-class breakdown
for mode in ["structured", "agent", "hybrid"]:
    subset = [r for r in results if r["gt"] == mode]
    if not subset:
        continue
    r_acc = sum(r["regex_correct"] for r in subset)
    print(f"\n  [{mode.upper()}] n={len(subset)}")
    if use_smart:
        s_acc = sum(r["smart_correct"] for r in subset)
        print(f"    SmartRouter : {s_acc}/{len(subset)} = {s_acc/len(subset)*100:.1f}%")
    print(f"    RegexRouter : {r_acc}/{len(subset)} = {r_acc/len(subset)*100:.1f}%")

# ── Ghi file kết quả ─────────────────────────────────────────────────────────
output_path = os.path.join(_PROJECT_ROOT, "routing_results.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for r in results:
        smart_col = f"Smart: {r['smart']:<10} | " if use_smart else ""
        f.write(f"Q{r['id']:02d}: {smart_col}Regex: {r['regex']:<10} | GT: {r['gt']}\n")

    f.write(f"\n{'='*60}\n")
    f.write("OVERALL ACCURACY\n")
    if use_smart:
        smart_total = sum(r["smart_correct"] for r in results)
        f.write(f"  SmartRouter : {smart_total}/{n} = {smart_total/n*100:.1f}%\n")
    f.write(f"  RegexRouter : {regex_total}/{n} = {regex_total/n*100:.1f}%\n")

    for mode in ["structured", "agent", "hybrid"]:
        subset = [r for r in results if r["gt"] == mode]
        if not subset:
            continue
        r_acc = sum(r["regex_correct"] for r in subset)
        f.write(f"\n  [{mode.upper()}] n={len(subset)}\n")
        if use_smart:
            s_acc = sum(r["smart_correct"] for r in subset)
            f.write(f"    SmartRouter : {s_acc}/{len(subset)} = {s_acc/len(subset)*100:.1f}%\n")
        f.write(f"    RegexRouter : {r_acc}/{len(subset)} = {r_acc/len(subset)*100:.1f}%\n")

print(f"\nĐã lưu → {output_path}")