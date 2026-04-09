from __future__ import annotations

import os
import sys
import re
import json
import hashlib
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

from core.database import execute_query

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# 1. LOAD DATA & BUILD CHUNKS

print("=" * 60)
print("RAG AUDIT — Step 1.1")
print("=" * 60)

print("\n[1/5] Connecting to DB and loading data...")

df = execute_query("SELECT * FROM superstore")
print(f"      Loaded {len(df):,} rows from superstore table")

if df.empty:
    print("ERROR: No data loaded. Check DB connection.")
    sys.exit(1)

kpis = {
    "total_sales":   float(df["sales"].sum()),
    "total_profit":  float(df["profit"].sum()),
    "total_orders":  int(df["order_id"].nunique()),
    "profit_margin": float(df["profit"].sum() / df["sales"].sum() * 100),
}

min_date = str(pd.to_datetime(df["order_date"]).min().date())
max_date = str(pd.to_datetime(df["order_date"]).max().date())

filters = {
    "date_range": (min_date, max_date),
    "region":     [],
    "segment":    [],
    "category":   [],
}

print("\n[2/5] Building chunks...")
from rag.knowledge_builder import KnowledgeBaseBuilder, Chunk

builder = KnowledgeBaseBuilder()
static_chunks  = builder.build_static(df)
dynamic_chunks = builder.build_dynamic(df, kpis, filters)
all_chunks     = static_chunks + dynamic_chunks

print(f"      Static  chunks: {len(static_chunks)}")
print(f"      Dynamic chunks: {len(dynamic_chunks)}")
print(f"      TOTAL   chunks: {len(all_chunks)}")


def count_tokens(text: str) -> int:
    """Approximate token count: len / 4"""
    return max(1, len(text) // 4)


# DUMP ALL CHUNKS TO FILE
with open("audit_chunks_dump.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("FULL CHUNK DUMP — RAG Audit Step 1.1\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"STATIC CHUNKS ({len(static_chunks)} total)\n")
    f.write("-" * 60 + "\n\n")
    for c in static_chunks:
        tokens = count_tokens(c.text)
        f.write(f"chunk_id   : {c.chunk_id}\n")
        f.write(f"type       : {c.metadata.get('type', 'unknown')}\n")
        f.write(f"tokens     : ~{tokens}\n")
        f.write(f"metadata   : {json.dumps({k: v for k, v in c.metadata.items() if k != 'type'}, default=str)}\n")
        f.write(f"text       :\n{c.text}\n")
        f.write("\n" + "-" * 40 + "\n\n")

    f.write(f"\nDYNAMIC CHUNKS ({len(dynamic_chunks)} total)\n")
    f.write("-" * 60 + "\n\n")
    for c in dynamic_chunks:
        tokens = count_tokens(c.text)
        f.write(f"chunk_id   : {c.chunk_id}\n")
        f.write(f"type       : {c.metadata.get('type', 'unknown')}\n")
        f.write(f"tokens     : ~{tokens}\n")
        f.write(f"metadata   : {json.dumps({k: v for k, v in c.metadata.items() if k != 'type'}, default=str)}\n")
        f.write(f"text       :\n{c.text}\n")
        f.write("\n" + "-" * 40 + "\n\n")

print("      Done → audit_chunks_dump.txt")
print("\n[4/5] Analysing chunks...")

type_counter: Counter = Counter()
layer_counter: Counter = Counter({"static": len(static_chunks), "dynamic": len(dynamic_chunks)})
for c in all_chunks:
    type_counter[c.metadata.get("type", "unknown")] += 1

TOKEN_THRESHOLD = 500
long_chunks = []
for c in all_chunks:
    t = count_tokens(c.text)
    if t > TOKEN_THRESHOLD:
        long_chunks.append((c.chunk_id, c.metadata.get("type", "?"), t, c.text[:120]))

id_counts: Counter = Counter(c.chunk_id for c in all_chunks)
duplicate_ids = [(cid, cnt) for cid, cnt in id_counts.items() if cnt > 1]

content_hashes: Dict[str, List[str]] = defaultdict(list)
for c in all_chunks:
    normalized = re.sub(r'\d+', 'N', c.text[:200].lower().strip())
    h = hashlib.md5(normalized.encode()).hexdigest()[:8]
    content_hashes[h].append(c.chunk_id)

overlapping = {h: ids for h, ids in content_hashes.items() if len(ids) > 1}

missing_metadata = []
for c in all_chunks:
    issues = []
    if not c.metadata.get("type"):
        issues.append("missing_type")
    if c.metadata.get("type") in ("trend", "dimension") and not c.metadata.get("dimension") and not c.metadata.get("grain"):
        issues.append("missing_dimension_or_grain")
    if issues:
        missing_metadata.append((c.chunk_id, issues))

total_tokens = sum(count_tokens(c.text) for c in all_chunks)
avg_tokens   = total_tokens // len(all_chunks) if all_chunks else 0

print("\n[5/5] Running retrieval tests (test queries)...")

# Build RAG engine
from rag.engine import RAGEngine
engine = RAGEngine()
engine.build_static(df)
engine.build(df, kpis, filters)

# Test queries covering all intent types
TEST_QUERIES = [
    # kpi_value
    ("kpi_value",   "total sales this year"),
    ("kpi_value",   "what is total profit and margin"),
    ("kpi_value",   "how many orders"),
    # kpi_trend
    ("kpi_trend",   "sales trend over years"),
    ("kpi_trend",   "monthly profit trend"),
    ("kpi_trend",   "quarterly revenue growth"),
    # kpi_rank
    ("kpi_rank",    "top 5 regions by profit"),
    ("kpi_rank",    "best sub-categories by sales"),
    ("kpi_rank",    "which segment has highest margin"),
    # kpi_compare
    ("kpi_compare", "compare this year vs last year sales"),
    ("kpi_compare", "yoy profit growth"),
    # kpi_detail
    ("kpi_detail",  "which products are losing money"),
    ("kpi_detail",  "loss-making sub-categories"),
    ("kpi_detail",  "negative profit orders"),
    # semantic/paraphrase
    ("semantic",    "revenue by area"),
    ("semantic",    "which region makes most money"),
    ("semantic",    "income trend over time"),
    ("semantic",    "transactions per month"),
    ("semantic",    "which items are bleeding money"),
    ("semantic",    "discount impact on profitability"),
]

retrieved_chunk_ids: Counter = Counter()
retrieval_log: List[Dict] = []

for intent, query in TEST_QUERIES:
    ctx = engine.retrieve(query, k=6)
    top_ids = [c.chunk_id for c in ctx.chunks[:6]]
    scores  = [round(c.score, 3) for c in ctx.chunks[:6]]
    retrieved_chunk_ids.update(top_ids)
    retrieval_log.append({
        "intent": intent,
        "query":  query,
        "top_chunks": list(zip(top_ids, scores)),
    })

all_ids     = set(c.chunk_id for c in all_chunks)
retrieved   = set(retrieved_chunk_ids.keys())
never_retrieved = all_ids - retrieved

with open("audit_retrieval_log.txt", "w", encoding="utf-8") as f:
    f.write("RETRIEVAL LOG — Test Queries\n")
    f.write("=" * 80 + "\n\n")
    for entry in retrieval_log:
        f.write(f"[{entry['intent'].upper():12s}] {entry['query']}\n")
        for cid, score in entry["top_chunks"]:
            flag = "⭐" if score > 0.4 else ("  " if score > 0.15 else "❌")
            f.write(f"  {flag} {score:.3f}  {cid}\n")
        f.write("\n")

print("      Done → audit_retrieval_log.txt")

print("\nWriting audit_report.txt...")

SEP = "=" * 80

with open("audit_report.txt", "w", encoding="utf-8") as f:

    f.write(SEP + "\n")
    f.write("RAG CHUNK AUDIT REPORT — Step 1.1\n")
    f.write(SEP + "\n\n")

    f.write("A. TỔNG QUAN\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Total chunks     : {len(all_chunks)}\n")
    f.write(f"  Static chunks    : {len(static_chunks)}\n")
    f.write(f"  Dynamic chunks   : {len(dynamic_chunks)}\n")
    f.write(f"  Total ~tokens    : {total_tokens:,}\n")
    f.write(f"  Avg tokens/chunk : {avg_tokens}\n")
    f.write(f"  Max tokens/chunk : {max(count_tokens(c.text) for c in all_chunks)}\n")
    f.write(f"  Min tokens/chunk : {min(count_tokens(c.text) for c in all_chunks)}\n\n")

    f.write("B. PHÂN BỐ THEO TYPE\n")
    f.write("-" * 40 + "\n")
    for t, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_chunks) * 100
        f.write(f"  {t:20s} : {cnt:3d} chunks  ({pct:.1f}%)\n")
    f.write("\n")

    f.write("C. BẢNG TẤT CẢ CHUNKS — chunk_id | type | tokens | layer | issues\n")
    f.write("-" * 80 + "\n")
    f.write(f"  {'chunk_id':<35} {'type':<15} {'~tokens':>7} {'layer':<8} issues\n")
    f.write(f"  {'-'*35} {'-'*15} {'-'*7} {'-'*8} {'-'*20}\n")

    for c in all_chunks:
        layer   = "static" if c in static_chunks else "dynamic"
        tokens  = count_tokens(c.text)
        ctype   = c.metadata.get("type", "?")
        issues  = []
        if tokens > TOKEN_THRESHOLD:
            issues.append(f"LONG({tokens}t)")
        if id_counts[c.chunk_id] > 1:
            issues.append("DUPLICATE_ID")
        if c.chunk_id in never_retrieved:
            issues.append("NEVER_RETRIEVED")
        normalized = re.sub(r'\d+', 'N', c.text[:200].lower().strip())
        h = hashlib.md5(normalized.encode()).hexdigest()[:8]
        if len(content_hashes[h]) > 1:
            peer = [x for x in content_hashes[h] if x != c.chunk_id]
            issues.append(f"OVERLAP({peer[0][:20]})")

        issue_str = ", ".join(issues) if issues else "ok"
        f.write(f"  {c.chunk_id:<35} {ctype:<15} {tokens:>7} {layer:<8} {issue_str}\n")
    f.write("\n")

    f.write("D. CHUNKS QUÁ DÀI (> 500 tokens) — cần split hoặc rút gọn\n")
    f.write("-" * 60 + "\n")
    if long_chunks:
        for cid, ctype, tokens, preview in sorted(long_chunks, key=lambda x: -x[2]):
            f.write(f"  [{tokens:4d}t] {cid} ({ctype})\n")
            f.write(f"           Preview: {preview}...\n\n")
    else:
        f.write("  Không có chunk nào vượt ngưỡng 500 tokens.\n\n")

    f.write("E. DUPLICATE CHUNK IDs — cùng ID xuất hiện > 1 lần\n")
    f.write("-" * 60 + "\n")
    if duplicate_ids:
        for cid, cnt in sorted(duplicate_ids, key=lambda x: -x[1]):
            f.write(f"  {cid} xuất hiện {cnt} lần\n")
        f.write("\n")
    else:
        f.write("  Không có duplicate chunk ID.\n\n")

    f.write("F. OVERLAPPING CHUNKS — nội dung gần giống nhau\n")
    f.write("-" * 60 + "\n")
    if overlapping:
        for h, ids in overlapping.items():
            f.write(f"  Hash group [{h}]: {ids}\n")
        f.write("\n")
    else:
        f.write("  Không phát hiện overlap đáng kể.\n\n")

    f.write("G. CHUNKS KHÔNG BAO GIỜ ĐƯỢC RETRIEVE (trong test queries)\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Test queries: {len(TEST_QUERIES)}\n")
    f.write(f"  Never retrieved: {len(never_retrieved)} / {len(all_chunks)} chunks\n\n")
    for cid in sorted(never_retrieved):
        # Tìm chunk
        chunk = next((c for c in all_chunks if c.chunk_id == cid), None)
        if chunk:
            ctype  = chunk.metadata.get("type", "?")
            tokens = count_tokens(chunk.text)
            preview = chunk.text[:100].replace("\n", " ")
            f.write(f"  [{ctype:12s}] {cid}\n")
            f.write(f"    {tokens}t | {preview}...\n\n")

    # ── H. Retrieval frequency ────────────────────────────────
    f.write("H. RETRIEVAL FREQUENCY — chunk được retrieve nhiều nhất\n")
    f.write("-" * 60 + "\n")
    for cid, cnt in retrieved_chunk_ids.most_common(20):
        chunk = next((c for c in all_chunks if c.chunk_id == cid), None)
        ctype = chunk.metadata.get("type", "?") if chunk else "?"
        f.write(f"  {cnt:2d}x  {cid:<40} ({ctype})\n")
    f.write("\n")

    # ── I. Missing metadata ───────────────────────────────────
    f.write("I. CHUNKS CÓ METADATA THIẾU\n")
    f.write("-" * 60 + "\n")
    if missing_metadata:
        for cid, issues in missing_metadata:
            f.write(f"  {cid}: {issues}\n")
        f.write("\n")
    else:
        f.write("  Tất cả chunks đều có metadata cơ bản.\n\n")

    # ── J. Tóm tắt vấn đề ─────────────────────────────────────
    f.write("J. TÓM TẮT VẤN ĐỀ PHÁT HIỆN\n")
    f.write("-" * 60 + "\n")
    problems = []
    if long_chunks:
        problems.append(f"  🔴 {len(long_chunks)} chunks quá dài (>{TOKEN_THRESHOLD} tokens) → gây noise khi đưa vào prompt")
    if duplicate_ids:
        problems.append(f"  🔴 {len(duplicate_ids)} duplicate chunk IDs → index bị ghi đè")
    if overlapping:
        problems.append(f"  🟡 {len(overlapping)} nhóm chunks overlap nội dung → redundancy, waste context window")
    if never_retrieved:
        problems.append(f"  🟡 {len(never_retrieved)} chunks không bao giờ được retrieve → có thể chunk quá chung chung hoặc query coverage thiếu")
    if missing_metadata:
        problems.append(f"  🟡 {len(missing_metadata)} chunks thiếu metadata → không thể filter chính xác")

    # Kiểm tra thiếu anomaly chunks
    anomaly_types = [c for c in all_chunks if "anomaly" in c.chunk_id.lower() or "loss" in c.chunk_id.lower()]
    if not anomaly_types:
        problems.append("  🔴 THIẾU anomaly_fact chunks — loss-making/negative profit không có chunk riêng")

    # Kiểm tra thiếu per-dimension-value chunks (mỗi region/segment có chunk riêng không?)
    per_value_chunks = [c for c in all_chunks if c.metadata.get("dimension_value")]
    if not per_value_chunks:
        problems.append("  🔴 THIẾU per-dimension-value chunks — mỗi region/segment không có chunk atomic riêng")

    if problems:
        for p in problems:
            f.write(p + "\n")
    else:
        f.write("  Không phát hiện vấn đề nghiêm trọng.\n")
    f.write("\n")

    f.write(SEP + "\n")
    f.write("END OF REPORT\n")
    f.write(SEP + "\n")


# ─────────────────────────────────────────────────────────────
# 8. IN TÓM TẮT RA CONSOLE
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("AUDIT COMPLETE — TÓM TẮT NHANH")
print("=" * 60)
print(f"\nTổng chunks     : {len(all_chunks)} (static={len(static_chunks)}, dynamic={len(dynamic_chunks)})")
print(f"Tổng ~tokens    : {total_tokens:,}  (avg={avg_tokens}/chunk)")
print(f"\nPhân bố type:")
for t, cnt in sorted(type_counter.items(), key=lambda x: -x[1]):
    print(f"  {t:<20}: {cnt} chunks")

print(f"\n🔍 Vấn đề phát hiện:")
print(f"  Chunks quá dài (>{TOKEN_THRESHOLD}t) : {len(long_chunks)}")
print(f"  Duplicate IDs            : {len(duplicate_ids)}")
print(f"  Overlapping content      : {len(overlapping)} nhóm")
print(f"  Never retrieved          : {len(never_retrieved)} / {len(all_chunks)}")
print(f"  Missing metadata         : {len(missing_metadata)}")

print(f"\n📂 Files tạo ra:")
print("  audit_chunks_dump.txt   — full chunk text")
print("  audit_report.txt        — bảng tổng hợp + vấn đề")
print("  audit_retrieval_log.txt — retrieval test log")
print("\nĐọc audit_report.txt để xem chi tiết và báo kết quả.")
print("=" * 60)