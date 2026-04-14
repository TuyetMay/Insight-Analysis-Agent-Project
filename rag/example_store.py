from __future__ import annotations

import json
import logging
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PLAN_DISPLAY_KEYS = {
    "intent", "metrics", "breakdown_by", "secondary_breakdown",
    "time_grain", "compare_period", "condition", "top_k",
}


@dataclass
class QueryExample:
    """One stored query-plan pair."""
    question: str
    intent: str
    plan: Dict[str, Any]
    sql_pattern: str = ""          # representative SQL (not always populated)
    success: bool = True           # False = was corrected by user / fallback
    score: float = 0.0             # retrieval score (populated at query time)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.question.lower().encode()).hexdigest()[:8]

    def plan_summary(self) -> Dict[str, Any]:
        return {k: v for k, v in self.plan.items() if k in _PLAN_DISPLAY_KEYS and v is not None}


# ── Pre-seeded canonical examples ────────────────────────────────────────────

_SEED_EXAMPLES: List[Dict[str, Any]] = [
    {
        "question":    "What is the total sales by region?",
        "intent":      "kpi_value",
        "plan":        {"intent": "kpi_value", "metrics": ["sales"], "breakdown_by": "region", "time_grain": "none"},
        "sql_pattern": "SELECT region AS breakdown, SUM(sales) AS sales FROM superstore GROUP BY region ORDER BY sales DESC",
    },
    {
        "question":    "Show profit by segment",
        "intent":      "kpi_value",
        "plan":        {"intent": "kpi_value", "metrics": ["profit"], "breakdown_by": "segment", "time_grain": "none"},
        "sql_pattern": "SELECT segment AS breakdown, SUM(profit) AS profit FROM superstore GROUP BY segment ORDER BY profit DESC",
    },
    {
        "question":    "Which sub-categories are losing money?",
        "intent":      "kpi_detail",
        "plan":        {"intent": "kpi_detail", "condition": "profit_negative", "breakdown_by": "sub_category"},
        "sql_pattern": "SELECT sub_category AS breakdown, SUM(profit) AS profit FROM superstore GROUP BY sub_category HAVING SUM(profit) < 0 ORDER BY profit ASC",
    },
    {
        "question":    "Show me the sales trend over years",
        "intent":      "kpi_trend",
        "plan":        {"intent": "kpi_trend", "metrics": ["sales"], "time_grain": "year"},
        "sql_pattern": "SELECT DATE_TRUNC('year', order_date) AS period, SUM(sales) AS sales FROM superstore GROUP BY period ORDER BY period",
    },
    {
        "question":    "Monthly profit and sales trend",
        "intent":      "kpi_trend",
        "plan":        {"intent": "kpi_trend", "metrics": ["sales", "profit"], "time_grain": "month"},
        "sql_pattern": "SELECT DATE_TRUNC('month', order_date) AS period, SUM(sales) AS sales, SUM(profit) AS profit FROM superstore GROUP BY period ORDER BY period",
    },
    {
        "question":    "Top 10 sub-categories by profit",
        "intent":      "kpi_rank",
        "plan":        {"intent": "kpi_rank", "metrics": ["profit"], "breakdown_by": "sub_category", "top_k": 10},
        "sql_pattern": "SELECT sub_category AS breakdown, SUM(profit) AS profit FROM superstore GROUP BY sub_category ORDER BY profit DESC LIMIT 10",
    },
    {
        "question":    "Best 5 regions by sales",
        "intent":      "kpi_rank",
        "plan":        {"intent": "kpi_rank", "metrics": ["sales"], "breakdown_by": "region", "top_k": 5},
        "sql_pattern": "SELECT region AS breakdown, SUM(sales) AS sales FROM superstore GROUP BY region ORDER BY sales DESC LIMIT 5",
    },
    {
        "question":    "Compare 2016 vs 2017 sales",
        "intent":      "kpi_compare",
        "plan":        {"intent": "kpi_compare", "metrics": ["sales"], "compare_period": "prev_period"},
        "sql_pattern": "",
    },
    {
        "question":    "Sales year over year growth",
        "intent":      "kpi_compare",
        "plan":        {"intent": "kpi_compare", "metrics": ["sales"], "compare_period": "yoy"},
        "sql_pattern": "",
    },
    {
        "question":    "Total profit margin across all orders",
        "intent":      "kpi_value",
        "plan":        {"intent": "kpi_value", "metrics": ["profit_margin"], "time_grain": "none"},
        "sql_pattern": "SELECT CASE WHEN SUM(sales)=0 THEN 0 ELSE SUM(profit)/SUM(sales)*100 END AS profit_margin FROM superstore",
    },
    {
        "question":    "Quarterly revenue breakdown by category",
        "intent":      "kpi_trend",
        "plan":        {"intent": "kpi_trend", "metrics": ["sales"], "time_grain": "quarter", "breakdown_by": "category"},
        "sql_pattern": "",
    },
    {
        "question":    "Which orders have negative profit due to heavy discounts?",
        "intent":      "kpi_detail",
        "plan":        {"intent": "kpi_detail", "condition": "profit_negative", "breakdown_by": "sub_category"},
        "sql_pattern": "",
    },
]


# ── ExampleStore ──────────────────────────────────────────────────────────────

class ExampleStore:
    """
    Maintains a ring-buffer of QueryExample objects and serves top-k
    similar examples for few-shot injection into the Tier-3 LLM prompt.

    Thread-safe reads; single-writer assumed (main Streamlit session).
    """

    _MAX_RUNTIME = 300   # max runtime-added examples (seeds not counted)
    _MIN_SCORE   = 0.30  # minimum cosine similarity to include in results

    def __init__(self) -> None:
        self._seeds: List[QueryExample] = [
            QueryExample(**{k: deepcopy(v) for k, v in s.items()})
            for s in _SEED_EXAMPLES
        ]
        self._runtime: List[QueryExample] = []
        self._seed_embeddings:    Optional[np.ndarray] = None
        self._runtime_embeddings: Optional[np.ndarray] = None
        self._seeds_dirty   = True
        self._runtime_dirty = False

    # ── Public ───────────────────────────────────────────────────────────────

    def add(self, question: str, intent: str,
            plan: Dict[str, Any], sql_pattern: str = "") -> None:
        """
        Record a verified (question, plan) pair at runtime.
        Called by DashboardChatbot after a successful Tier-3 execution.
        """
        # Evict oldest if at capacity
        if len(self._runtime) >= self._MAX_RUNTIME:
            self._runtime = self._runtime[-self._MAX_RUNTIME + 1:]
            self._runtime_embeddings = None

        ex = QueryExample(
            question=question, intent=intent,
            plan=deepcopy(plan), sql_pattern=sql_pattern, success=True,
        )
        self._runtime.append(ex)
        self._runtime_dirty = True
        logger.debug("ExampleStore: added runtime example intent=%s q='%s'", intent, question[:60])

    def retrieve(
        self,
        question: str,
        k: int = 3,
        intent: Optional[str] = None,
    ) -> List[QueryExample]:
        """
        Return top-k examples ordered by descending similarity.
        Prefers intent-matching examples when intent is known.
        Falls back gracefully to TF-IDF if sentence-transformers unavailable.
        """
        all_examples = self._seeds + self._runtime

        # Intent filtering: prefer same-intent but don't hard-exclude
        if intent:
            intent_examples = [e for e in all_examples if e.intent == intent]
            other_examples  = [e for e in all_examples if e.intent != intent]
        else:
            intent_examples = all_examples
            other_examples  = []

        try:
            from rag.retriever import _get_model
            model = _get_model()
            if model is None:
                raise ImportError("model not available")

            q_vec = model.encode(
                [question], convert_to_numpy=True, normalize_embeddings=True
            )[0].astype(np.float32)

            results: List[QueryExample] = []

            for pool, dirty_flag, emb_attr in [
                (intent_examples, "_seeds_dirty",   "_seed_embeddings"),
                (other_examples,  "_runtime_dirty", "_runtime_embeddings"),
            ]:
                if not pool:
                    continue
                embs = self._get_embeddings(pool, model)
                if embs is None:
                    continue

                # Only score the subset we care about
                pool_all   = self._seeds + self._runtime
                pool_idxs  = [pool_all.index(e) for e in pool if e in pool_all]
                emb_subset = embs[pool_idxs] if embs is not None and pool_idxs else embs

                scores = emb_subset.dot(q_vec) if emb_subset is not None else np.array([])
                for i, score in enumerate(scores):
                    if float(score) >= self._MIN_SCORE:
                        ex_copy       = deepcopy(pool[i])
                        ex_copy.score = float(score)
                        results.append(ex_copy)

            results.sort(key=lambda x: x.score, reverse=True)

            # Deduplicate by chunk_id
            seen: set = set()
            deduped: List[QueryExample] = []
            for ex in results:
                if ex.chunk_id not in seen:
                    seen.add(ex.chunk_id)
                    deduped.append(ex)

            return deduped[:k]

        except Exception as exc:
            logger.debug("ExampleStore dense retrieval failed (%s) — using TF-IDF", exc)
            return self._tfidf_retrieve(question, intent_examples + other_examples, k)

    def as_prompt_section(
        self,
        question: str,
        k: int = 3,
        intent: Optional[str] = None,
    ) -> str:
        """Return a formatted block for injection into the Gemini prompt."""
        examples = self.retrieve(question, k=k, intent=intent)
        if not examples:
            return ""

        lines = ["[Similar past queries — use these plan structures as reference]"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"\nExample {i} (similarity {ex.score:.2f}):")
            lines.append(f"  Q: {ex.question}")
            lines.append(f"  Plan: {json.dumps(ex.plan_summary(), ensure_ascii=False)}")
            if ex.sql_pattern:
                snippet = ex.sql_pattern[:100] + ("..." if len(ex.sql_pattern) > 100 else "")
                lines.append(f"  SQL: {snippet}")
        lines.append("")
        return "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_embeddings(
        self,
        pool: List[QueryExample],
        model: Any,
    ) -> Optional[np.ndarray]:
        """Compute / return cached embeddings for a pool of examples."""
        if not pool:
            return None
        try:
            texts = [e.question for e in pool]
            embs  = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embs.astype(np.float32)
        except Exception as exc:
            logger.warning("ExampleStore: embedding failed (%s)", exc)
            return None

    def _tfidf_retrieve(
        self,
        question: str,
        pool: List[QueryExample],
        k: int,
    ) -> List[QueryExample]:
        """Jaccard-overlap fallback when dense model is unavailable."""
        q_tokens = set(question.lower().split())
        scored: List[QueryExample] = []

        for ex in pool:
            ex_tokens = set(ex.question.lower().split())
            union     = len(q_tokens | ex_tokens)
            overlap   = len(q_tokens & ex_tokens) / union if union else 0.0
            if overlap >= 0.15:
                ex_copy       = deepcopy(ex)
                ex_copy.score = overlap
                scored.append(ex_copy)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]