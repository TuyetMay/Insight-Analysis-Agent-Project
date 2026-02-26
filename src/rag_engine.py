# src/rag_engine.py
"""
RAG Engine cho Superstore Dashboard Chatbot.

Database gốc là dữ liệu BẢNG (tabular), không phải PDF/text.
Pipeline:
  1. KnowledgeBaseBuilder  : aggregate DataFrame -> text chunks mô tả số liệu thực
  2. TFIDFRetriever        : cosine similarity search (chỉ dùng numpy)
  3. RAGEngine             : facade - build index, quản lý chat history, retrieve context

Ví dụ chunk được tạo ra:
  "Top 5 sub_category theo profit: Copiers ($55,618), Phones ($44,516), ..."
  "Nam 2017: sales=$609,206, profit=$81,795"
  "Dimension 'region': values = [Central, East, South, West]. Highest profit: West ($108k)"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """Mot don vi kien thuc co the retrieve."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RAGContext:
    """
    Ket qua retrieve: chunks lien quan + tom tat chat history.
    Dung as_prompt_section() de inject vao Gemini prompt.
    """
    query: str
    chunks: List[Chunk]
    chat_summary: str = ""

    def as_prompt_section(self, max_chunks: int = 6) -> str:
        lines: List[str] = []
        if self.chat_summary:
            lines.append("[Lich su hoi thoai gan day]")
            lines.append(self.chat_summary)
        if self.chunks:
            lines.append("\n[Du lieu thuc te da xac nhan tu dashboard]")
            for c in self.chunks[:max_chunks]:
                lines.append(f"  - {c.text}")
        return "\n".join(lines)

    def chunk_texts(self, max_chunks: int = 6) -> List[str]:
        return [c.text for c in self.chunks[:max_chunks]]


# ─────────────────────────────────────────────────────────────
# KnowledgeBaseBuilder  -  tabular data -> text chunks
# ─────────────────────────────────────────────────────────────

class KnowledgeBaseBuilder:
    """
    Chuyen doi DataFrame bang Superstore thanh cac text chunks.
    Moi chunk = mot su that da duoc tinh toan tu du lieu thuc.
    """

    def build(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunks.extend(self._schema_chunks())
        chunks.extend(self._filter_context_chunks(filters))
        chunks.extend(self._kpi_chunks(kpis, filters))
        if not df.empty:
            chunks.extend(self._time_range_chunks(df))
            chunks.extend(self._yearly_trend_chunks(df))
            chunks.extend(self._monthly_chunks(df))
            chunks.extend(self._quarterly_chunks(df))
            chunks.extend(self._dimension_chunks(df, "region"))
            chunks.extend(self._dimension_chunks(df, "segment"))
            chunks.extend(self._dimension_chunks(df, "category"))
            chunks.extend(self._top_k_chunks(df, "sub_category", k=10))
            chunks.extend(self._top_k_chunks(df, "category", k=5))
            chunks.extend(self._top_k_chunks(df, "region", k=4))
            chunks.extend(self._discount_impact_chunks(df))
            chunks.extend(self._segment_category_cross_chunks(df))
        return chunks

    # -- Schema (kha nang he thong) -----------------------------------

    def _schema_chunks(self) -> List[Chunk]:
        return [
            Chunk(
                chunk_id="schema_metrics",
                text=(
                    "Cac metrics co the truy van: "
                    "sales (tong doanh thu USD), "
                    "profit (loi nhuan, co the am), "
                    "orders (so don hang = COUNT DISTINCT order_id), "
                    "profit_margin (= profit/sales x 100, don vi %)."
                ),
                metadata={"type": "schema", "topic": "metrics"},
            ),
            Chunk(
                chunk_id="schema_dimensions",
                text=(
                    "Cac chieu phan tich (breakdown): region, segment, category, sub_category. "
                    "Cac moc thoi gian (time_grain): week, month, quarter, year."
                ),
                metadata={"type": "schema", "topic": "dimensions"},
            ),
            Chunk(
                chunk_id="schema_compare",
                text=(
                    "So sanh ky: yoy (year-over-year voi cung ky nam truoc), "
                    "mom (month-over-month voi thang truoc), "
                    "prev_period (so voi ky lien truoc cung do dai)."
                ),
                metadata={"type": "schema", "topic": "comparison"},
            ),
            Chunk(
                chunk_id="schema_intents",
                text=(
                    "Cac loai cau hoi co the tra loi: "
                    "kpi_value (gia tri tong hop), "
                    "kpi_trend (xu huong theo thoi gian), "
                    "kpi_rank (xep hang top-k theo dimension), "
                    "kpi_compare (so sanh ky nay vs ky truoc)."
                ),
                metadata={"type": "schema", "topic": "intents"},
            ),
        ]

    # -- Filter context ------------------------------------------------

    def _filter_context_chunks(self, filters: Dict[str, Any]) -> List[Chunk]:
        dr = filters.get("date_range", ())
        date_str = f"{dr[0]} den {dr[1]}" if (dr and len(dr) == 2) else "toan bo lich su"
        regions = filters.get("region", [])
        segments = filters.get("segment", [])
        categories = filters.get("category", [])

        parts = [f"Khoang thoi gian: {date_str}"]
        if regions:
            parts.append(f"Region: {', '.join(str(r) for r in regions)}")
        if segments:
            parts.append(f"Segment: {', '.join(str(s) for s in segments)}")
        if categories:
            parts.append(f"Category: {', '.join(str(c) for c in categories)}")

        return [Chunk(
            chunk_id="filter_context",
            text="Bo loc dashboard hien tai: " + "; ".join(parts) + ".",
            metadata={"type": "filter"},
        )]

    # -- KPI tong quan -------------------------------------------------

    def _kpi_chunks(self, kpis: Dict[str, Any], filters: Dict[str, Any]) -> List[Chunk]:
        ts = float(kpis.get("total_sales", 0) or 0)
        tp = float(kpis.get("total_profit", 0) or 0)
        to_ = int(kpis.get("total_orders", 0) or 0)
        pm = float(kpis.get("profit_margin", 0) or 0)

        return [
            Chunk(
                chunk_id="kpi_summary",
                text=(
                    f"Tong quan KPI theo bo loc hien tai: "
                    f"Total Sales=${ts:,.0f}, "
                    f"Total Profit=${tp:,.0f}, "
                    f"Total Orders={to_:,}, "
                    f"Profit Margin={pm:.2f}%."
                ),
                metadata={"type": "kpi", "metrics": ["sales", "profit", "orders", "profit_margin"]},
            ),
            Chunk(
                chunk_id="kpi_profitability",
                text=(
                    f"Loi nhuan tong: ${tp:,.0f} tren doanh thu ${ts:,.0f}. "
                    f"Bien loi nhuan = {pm:.2f}%. "
                    f"{'Kinh doanh co lai.' if tp > 0 else 'Dang lo.'}"
                ),
                metadata={"type": "kpi", "topic": "profitability"},
            ),
        ]

    # -- Thoi gian ------------------------------------------------------

    def _time_range_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns:
            return []
        try:
            dates = pd.to_datetime(df["order_date"], errors="coerce").dropna()
            if dates.empty:
                return []
            min_d = dates.min().date()
            max_d = dates.max().date()
            years = sorted(dates.dt.year.unique().tolist())
            months_count = dates.dt.to_period("M").nunique()
            return [Chunk(
                chunk_id="time_range",
                text=(
                    f"Du lieu trai dai tu {min_d} den {max_d}. "
                    f"Co {len(years)} nam: {years}. "
                    f"Tong {months_count} thang co du lieu."
                ),
                metadata={"type": "time", "years": years},
            )]
        except Exception:
            return []

    def _yearly_trend_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg_dict = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg_dict["profit"] = ("profit", "sum")
            if "order_id" in df2.columns:
                agg_dict["orders"] = ("order_id", "nunique")

            yearly = df2.groupby(df2["order_date"].dt.year).agg(**agg_dict).reset_index()
            yearly.rename(columns={"order_date": "year"}, inplace=True)

            chunks = []
            for _, row in yearly.iterrows():
                yr = int(row["year"])
                s = float(row["sales"])
                p = float(row.get("profit", 0))
                o = int(row.get("orders", 0))
                pm = (p / s * 100) if s > 0 else 0
                chunks.append(Chunk(
                    chunk_id=f"yearly_{yr}",
                    text=(
                        f"Nam {yr}: Sales=${s:,.0f}, Profit=${p:,.0f}, "
                        f"Orders={o:,}, Profit Margin={pm:.1f}%."
                    ),
                    metadata={"type": "trend", "grain": "year", "year": yr},
                ))

            if len(yearly) >= 2:
                first_s = float(yearly.iloc[0]["sales"])
                last_s = float(yearly.iloc[-1]["sales"])
                growth = ((last_s - first_s) / first_s * 100) if first_s > 0 else 0
                first_yr = int(yearly.iloc[0]["year"])
                last_yr = int(yearly.iloc[-1]["year"])
                chunks.append(Chunk(
                    chunk_id="yearly_growth_summary",
                    text=(
                        f"Tang truong doanh thu tu {first_yr} den {last_yr}: "
                        f"{growth:+.1f}% (tu ${first_s:,.0f} len ${last_s:,.0f})."
                    ),
                    metadata={"type": "trend", "topic": "growth"},
                ))
            return chunks
        except Exception:
            return []

    def _monthly_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg_dict = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg_dict["profit"] = ("profit", "sum")
            monthly = df2.groupby(df2["order_date"].dt.to_period("M")).agg(**agg_dict).reset_index()

            if monthly.empty:
                return []
            best = monthly.loc[monthly["sales"].idxmax()]
            chunks = [
                Chunk(
                    chunk_id="monthly_best_sales",
                    text=(
                        f"Thang co doanh thu cao nhat: {best['order_date']} "
                        f"voi Sales=${float(best['sales']):,.0f}"
                        + (f", Profit=${float(best['profit']):,.0f}." if "profit" in best else ".")
                    ),
                    metadata={"type": "time", "grain": "month", "topic": "best"},
                ),
            ]
            if "profit" in monthly.columns:
                worst_p = monthly.loc[monthly["profit"].idxmin()]
                if float(worst_p["profit"]) < 0:
                    chunks.append(Chunk(
                        chunk_id="monthly_worst_profit",
                        text=(
                            f"Thang lo nhieu nhat: {worst_p['order_date']} "
                            f"voi Profit=${float(worst_p['profit']):,.0f} (am)."
                        ),
                        metadata={"type": "time", "grain": "month", "topic": "worst"},
                    ))
            return chunks
        except Exception:
            return []

    def _quarterly_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "order_date" not in df.columns or "sales" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["order_date"] = pd.to_datetime(df2["order_date"], errors="coerce")
            df2 = df2[df2["order_date"].notna()]

            agg_dict = {"sales": ("sales", "sum")}
            if "profit" in df2.columns:
                agg_dict["profit"] = ("profit", "sum")

            quarterly = (
                df2.groupby(df2["order_date"].dt.to_period("Q"))
                .agg(**agg_dict)
                .reset_index()
                .tail(8)
            )
            if quarterly.empty:
                return []
            rows = "; ".join(
                f"{row['order_date']}: Sales=${float(row['sales']):,.0f}"
                + (f", Profit=${float(row['profit']):,.0f}" if "profit" in row else "")
                for _, row in quarterly.iterrows()
            )
            return [Chunk(
                chunk_id="quarterly_summary",
                text=f"Tom tat theo quy (8 quy gan nhat): {rows}.",
                metadata={"type": "trend", "grain": "quarter"},
            )]
        except Exception:
            return []

    # -- Dimension stats -----------------------------------------------

    def _dimension_chunks(self, df: pd.DataFrame, dim: str) -> List[Chunk]:
        if dim not in df.columns or "sales" not in df.columns:
            return []
        try:
            agg_dict = {"sales": ("sales", "sum")}
            if "profit" in df.columns:
                agg_dict["profit"] = ("profit", "sum")
            if "order_id" in df.columns:
                agg_dict["orders"] = ("order_id", "nunique")

            grp = df.groupby(dim).agg(**agg_dict).reset_index().sort_values("sales", ascending=False)
            values = sorted(df[dim].dropna().unique().tolist())

            detail = "; ".join(
                f"{row[dim]}: Sales=${float(row['sales']):,.0f}"
                + (f", Profit=${float(row['profit']):,.0f}" if "profit" in row else "")
                for _, row in grp.iterrows()
            )

            text = f"Dimension '{dim}' co cac gia tri: {values}. Chi tiet: {detail}."

            return [Chunk(
                chunk_id=f"dim_{dim}",
                text=text,
                metadata={"type": "dimension", "dimension": dim, "values": values},
            )]
        except Exception:
            return []

    def _top_k_chunks(self, df: pd.DataFrame, dim: str, k: int = 10) -> List[Chunk]:
        if dim not in df.columns or "profit" not in df.columns:
            return []
        try:
            grp = df.groupby(dim).agg(
                sales=("sales", "sum"),
                profit=("profit", "sum"),
            ).reset_index()

            top_profit = grp.nlargest(k, "profit")
            top_sales = grp.nlargest(k, "sales")

            profit_str = ", ".join(
                f"{row[dim]} (${float(row['profit']):,.0f})"
                for _, row in top_profit.iterrows()
            )
            sales_str = ", ".join(
                f"{row[dim]} (${float(row['sales']):,.0f})"
                for _, row in top_sales.iterrows()
            )

            return [
                Chunk(
                    chunk_id=f"top{k}_{dim}_profit",
                    text=f"Top {k} {dim.replace('_',' ')} theo profit: {profit_str}.",
                    metadata={"type": "rank", "dimension": dim, "metric": "profit", "k": k},
                ),
                Chunk(
                    chunk_id=f"top{k}_{dim}_sales",
                    text=f"Top {k} {dim.replace('_',' ')} theo sales: {sales_str}.",
                    metadata={"type": "rank", "dimension": dim, "metric": "sales", "k": k},
                ),
            ]
        except Exception:
            return []

    # -- Discount impact -----------------------------------------------

    def _discount_impact_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "discount" not in df.columns or "profit" not in df.columns:
            return []
        try:
            df2 = df.copy()
            df2["discount"] = pd.to_numeric(df2["discount"], errors="coerce")
            df2 = df2[df2["discount"].notna()]
            if df2["discount"].max() > 1:
                df2["discount"] = df2["discount"] / 100

            df2["discount_bucket"] = pd.cut(
                df2["discount"],
                bins=[0, 0.1, 0.2, 0.3, 1.0],
                labels=["0-10%", "10-20%", "20-30%", ">30%"],
                include_lowest=True,
            )
            buckets = df2.groupby("discount_bucket", observed=True).agg(
                avg_profit=("profit", "mean"),
                avg_sales=("sales", "mean"),
                count=("profit", "count"),
            ).reset_index()

            lines = "; ".join(
                f"Giam gia {row['discount_bucket']}: avg profit=${float(row['avg_profit']):,.0f}, "
                f"avg sales=${float(row['avg_sales']):,.0f} ({int(row['count'])} don)"
                for _, row in buckets.iterrows()
                if int(row["count"]) > 0
            )
            return [Chunk(
                chunk_id="discount_impact",
                text=f"Anh huong cua discount len profit: {lines}.",
                metadata={"type": "insight", "topic": "discount"},
            )]
        except Exception:
            return []

    # -- Cross dimension -----------------------------------------------

    def _segment_category_cross_chunks(self, df: pd.DataFrame) -> List[Chunk]:
        if "segment" not in df.columns or "category" not in df.columns or "profit" not in df.columns:
            return []
        try:
            cross = (
                df.groupby(["segment", "category"])
                .agg(profit=("profit", "sum"), sales=("sales", "sum"))
                .reset_index()
                .sort_values("profit", ascending=False)
                .head(9)
            )
            lines = "; ".join(
                f"{row['segment']}/{row['category']}: Profit=${float(row['profit']):,.0f}"
                for _, row in cross.iterrows()
            )
            return [Chunk(
                chunk_id="segment_category_cross",
                text=f"Loi nhuan theo Segment x Category (top 9): {lines}.",
                metadata={"type": "cross", "dimensions": ["segment", "category"]},
            )]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────
# TFIDFRetriever  -  cosine similarity, pure numpy
# ─────────────────────────────────────────────────────────────

class TFIDFRetriever:
    """TF-IDF retriever nhe, khong can sklearn."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._matrix: np.ndarray = np.array([])

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, chunks: List[Chunk]) -> "TFIDFRetriever":
        self._chunks = chunks
        if not chunks:
            return self

        vocab: Dict[str, int] = {}
        for c in chunks:
            for tok in self._tokenize(c.text):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._vocab = vocab
        V = len(vocab)
        N = len(chunks)

        tf = np.zeros((N, V), dtype=np.float32)
        for i, c in enumerate(chunks):
            for tok in self._tokenize(c.text):
                if tok in vocab:
                    tf[i, vocab[tok]] += 1
            row_len = tf[i].sum()
            if row_len > 0:
                tf[i] /= row_len

        df_counts = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((N + 1) / (df_counts + 1)) + 1.0
        self._idf = idf

        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._matrix = tfidf / norms
        return self

    def retrieve(self, query: str, k: int = 6) -> List[Chunk]:
        if not self._chunks or self._matrix.size == 0:
            return []
        V = len(self._vocab)
        qv = np.zeros(V, dtype=np.float32)
        for tok in self._tokenize(query):
            if tok in self._vocab:
                qv[self._vocab[tok]] += 1
        qv = qv * self._idf
        norm = np.linalg.norm(qv)
        if norm > 0:
            qv /= norm

        scores = self._matrix.dot(qv)
        top_k = min(k, len(self._chunks))
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            Chunk(
                chunk_id=self._chunks[i].chunk_id,
                text=self._chunks[i].text,
                metadata=self._chunks[i].metadata,
                score=float(scores[i]),
            )
            for i in top_idx
        ]


# ─────────────────────────────────────────────────────────────
# RAGEngine  -  facade cao cap
# ─────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Facade tong hop:
    - Xay dung knowledge base tu tabular data
    - Quan ly chat history buffer
    - Retrieve context cho moi cau hoi -> RAGContext

    Cach dung:
        rag = RAGEngine()
        rag.build(df, kpis, filters)          # goi khi filter thay doi
        rag.add_turn("user", "cau hoi")
        rag.add_turn("assistant", "cau tra loi")
        ctx = rag.retrieve("cau hoi moi", k=6)
        prompt_section = ctx.as_prompt_section()
    """

    MAX_TURNS = 8

    def __init__(self) -> None:
        self._retriever = TFIDFRetriever()
        self._all_chunks: List[Chunk] = []
        self._chat_history: List[Dict[str, str]] = []
        self._built = False

    def build(
        self,
        df: pd.DataFrame,
        kpis: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> None:
        """Rebuild toan bo knowledge base. Goi lai khi filter thay doi."""
        builder = KnowledgeBaseBuilder()
        self._all_chunks = builder.build(df, kpis, filters)
        self._retriever.fit(self._all_chunks)
        self._built = True

    @property
    def total_chunks(self) -> int:
        return len(self._all_chunks)

    def add_turn(self, role: str, content: str) -> None:
        """Them mot luot chat (role = 'user' | 'assistant')."""
        self._chat_history.append({"role": role, "content": content})
        if len(self._chat_history) > self.MAX_TURNS * 2:
            self._chat_history = self._chat_history[-(self.MAX_TURNS * 2):]

    def clear_history(self) -> None:
        self._chat_history = []

    def _summarize_history(self, max_turns: int = 5) -> str:
        recent = self._chat_history[-(max_turns * 2):]
        parts: List[str] = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Bot"
            text = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            parts.append(f"{role_label}: {text}")
        return "\n".join(parts)

    def retrieve(self, query: str, k: int = 6, min_score: float = 0.03) -> RAGContext:
        """Retrieve top-k chunks lien quan den query."""
        if not self._built:
            return RAGContext(query=query, chunks=[], chat_summary=self._summarize_history())

        candidates = self._retriever.retrieve(query, k=k + 4)

        # Luon dam bao co schema + kpi
        must_have_ids = {"schema_metrics", "schema_dimensions", "kpi_summary"}
        must_have_chunks = [c for c in self._all_chunks if c.chunk_id in must_have_ids]

        seen_ids: set = set()
        final: List[Chunk] = []

        for c in candidates:
            if c.score >= min_score and c.chunk_id not in seen_ids:
                final.append(c)
                seen_ids.add(c.chunk_id)

        for c in must_have_chunks:
            if c.chunk_id not in seen_ids:
                final.append(c)
                seen_ids.add(c.chunk_id)

        must_final = [c for c in final if c.chunk_id in must_have_ids]
        rest = [c for c in final if c.chunk_id not in must_have_ids]
        rest.sort(key=lambda x: x.score, reverse=True)

        return RAGContext(
            query=query,
            chunks=must_final + rest,
            chat_summary=self._summarize_history(),
        )

    def retrieve_for_suggestions(
        self, last_question: str, last_answer: str, k: int = 8
    ) -> RAGContext:
        """
        Retrieve context dac biet cho viec tao suggestions.
        Combine query tu ca cau hoi lan cau tra loi de bat context day du hon.
        """
        combined_query = f"{last_question} {last_answer}"
        return self.retrieve(combined_query, k=k, min_score=0.02)