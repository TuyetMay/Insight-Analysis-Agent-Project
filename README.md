# Superstore Business Intelligence Dashboard

An AI-powered interactive analytics platform built with Streamlit, enabling natural language querying over Superstore sales data through a multi-tier LLM pipeline with RAG-based retrieval and autonomous agent capabilities.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Database Setup](#database-setup)
  - [Running the App](#running-the-app)
- [How It Works](#how-it-works)
  - [Query Processing Pipeline](#query-processing-pipeline)
  - [RAG System](#rag-system)
  - [Agent Orchestration](#agent-orchestration)
  - [Suggestions Engine](#suggestions-engine)
- [Testing](#testing)
- [Configuration Reference](#configuration-reference)
- [Contributing](#contributing)

---

## Overview

This project is a full-stack Business Intelligence dashboard for the classic **Superstore sales dataset**. Users can filter data across dimensions (date range, region, segment, category) and interact with an AI assistant through a chat sidebar to ask business questions in plain English.

Under the hood, queries are processed through a **three-tier LLM pipeline** — fast rule-based KPI answers, structured SQL execution, and full Gemini LLM inference — with automatic fallback between tiers. A SmartRouter classifies each query and can escalate it to a **multi-turn autonomous agent** for complex, multi-step analysis.

---

## Features

### Dashboard
- **Live KPI cards** — Total Sales, Total Profit, Order Count, Profit Margin, each with an "Ask AI" shortcut
- **Interactive Plotly charts** — Sales & profit trends, regional/segment breakdown, category heatmaps, top subcategories, discount impact analysis
- **Multi-dimensional filters** — Date range, region, segment, category (sidebar); all charts and KPIs update in real time
- **Expandable raw data table** — Paginated view of the filtered dataset

### AI Chat Assistant
- **Natural language queries** — Ask anything about the filtered data in plain English
- **Multi-tier processing** with automatic fallback:
  - Tier 1: Instant rule-based KPI answers (no LLM call)
  - Tier 2: NL-parsed structured SQL execution
  - Tier 3: Full Gemini LLM inference with RAG context
- **Hybrid mode** — Combines structured query results with LLM-generated explanations
- **Autonomous agent** — Multi-turn reasoning for complex, open-ended questions
- **Follow-up suggestions** — Contextual next-question recommendations after every answer

### AI Infrastructure
- **SmartRouter** — LLM-based query classifier routing to `structured`, `agent`, or `hybrid` modes
- **RAG engine** — TFIDF retrieval with HyDE, metadata soft-ranking, and ExampleStore few-shot learning
- **Plan validation & LLM auditing** — Validates generated execution plans before running them
- **Assumption validator** — Checks agent-generated assumptions against actual data

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (app.py)                   │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  KPI Cards   │  │  Plotly Charts  │  │   Chat Sidebar   │  │
│  └──────────────┘  └─────────────────┘  └────────┬─────────┘  │
└───────────────────────────────────────────────────┼────────────┘
                                                    │ user question
                                                    ▼
                              ┌────────────────────────────────┐
                              │     DashboardChatbot           │
                              │       orchestrator.py          │
                              └──────────────┬─────────────────┘
                                             │
                              ┌──────────────▼─────────────────┐
                              │         SmartRouter             │
                              │  "structured" / "agent" /       │
                              │  "hybrid" / quick-token check   │
                              └──────┬──────────┬──────────┬───┘
                                     │          │          │
                         ┌───────────▼─┐  ┌─────▼────┐  ┌─▼──────────────┐
                         │  Tier 1/2/3 │  │  Agent   │  │    Hybrid      │
                         │  Structured │  │ Orchestr.│  │   Executor     │
                         │  Pipeline   │  │          │  │                │
                         └──────┬──────┘  └────┬─────┘  └───────┬────────┘
                                │              │                 │
                         ┌──────▼──────────────▼─────────────────▼────────┐
                         │              Core Data Layer                    │
                         │    PostgreSQL (Supabase) via SQLAlchemy Pool    │
                         └─────────────────────────────────────────────────┘
```

### Query Processing Layers

| Layer | Component | Trigger |
|-------|-----------|---------|
| Quick Token | `NLParser` | `__quick_sales__`, `__quick_profit__`, etc. |
| Tier 1 | `NLParser` fast path | Simple KPI questions |
| Tier 2 | `NLParser` + `SQLBuilder` | Structured queries with filters |
| Tier 3 | Gemini + RAG + `SQLBuilder` | Complex or ambiguous questions |
| Agent | `AgentOrchestrator` | Multi-step reasoning |
| Hybrid | `HybridExecutor` | Data + narrative combined |

---

## Tech Stack

| Category | Library / Service | Version |
|----------|------------------|---------|
| Web Framework | Streamlit | 1.52.2 |
| Language | Python | 3.11 |
| Database | PostgreSQL (Supabase cloud) | — |
| ORM / Driver | SQLAlchemy + psycopg2-binary | 2.0.45 / 2.9.11 |
| LLM | Google Gemini (Vertex AI) | gemini-1.5-flash-002 |
| LLM Framework | LangChain + LangSmith | 0.1.20 / 0.1.147 |
| LLM Alt. | OpenAI | 1.30.1 |
| Embeddings | sentence-transformers | 3.4.1 |
| Deep Learning | PyTorch | 2.7.0 |
| Visualization | Plotly / Plotly Express | 6.5.0 / 0.4.1 |
| Data Processing | Pandas / NumPy | 2.3.3 / 1.26.4 |
| Statistics | SciPy / Statsmodels | 1.16.3 / 0.14.6 |
| Config | python-dotenv | 1.2.1 |
| Tokenization | tiktoken | 0.12.0 |

---

## Project Structure

```
Superstore_streamlit/
│
├── app.py                          # Main Streamlit entry point
├── config.py                       # Centralized configuration
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version pin (3.11)
├── .env.example                    # Environment variable template
│
├── core/                           # Data layer
│   ├── database.py                 # ThreadedConnectionPool, health checks
│   ├── data_loader.py              # Cached data loading, KPI calculation
│   └── __init__.py
│
├── chatbot/                        # AI / LLM pipeline
│   ├── orchestrator.py             # DashboardChatbot — main entry point
│   ├── smart_router.py             # LLM-based query classifier
│   ├── nl_parser.py                # Natural language → structured plan
│   ├── sql_builder.py              # Plan → parameterized SQL
│   ├── plan_validator.py           # Pre-execution plan validation
│   ├── llm_plan_auditor.py         # LLM-based plan audit
│   ├── answer_formatter.py         # Result → display-ready text
│   ├── insight_generator.py        # LLM narrative synthesis
│   ├── quick_insight.py            # Fast KPI answer handler
│   ├── hybrid_executor.py          # Structured + agent combined execution
│   ├── query_router.py             # Legacy fallback router
│   │
│   ├── agent/                      # Autonomous agent subsystem
│   │   ├── orchestrator.py         # Multi-turn agent loop
│   │   ├── tools.py                # Agent tool definitions
│   │   ├── assumption_validator.py # Validates agent assumptions vs data
│   │   └── suggestions.py          # Diagnostic suggestion generator
│   │
│   └── suggestions/                # Follow-up suggestion engines
│       ├── rule_engine.py          # Rule-based suggestions
│       ├── rag_engine.py           # RAG-based suggestions
│       └── models.py               # Suggestion data models
│
├── rag/                            # Retrieval-Augmented Generation
│   ├── engine.py                   # Two-layer retrieval + ExampleStore
│   ├── retriever.py                # TFIDF retriever
│   ├── knowledge_builder.py        # Knowledge base chunking
│   ├── hyde.py                     # Hypothetical Document Embeddings
│   ├── example_store.py            # Few-shot query→plan store
│   ├── metadata_filter.py          # Soft re-ranking with priority weights
│   └── audit_test_chunk.py         # Retrieval audit utility
│
├── charts/                         # Visualization modules
│   ├── trends.py                   # Sales/profit/orders over time
│   ├── breakdown.py                # Regional, segment, heatmap charts
│   ├── products.py                 # Category, subcategory, discount charts
│   ├── _utils.py                   # Shared chart utilities
│   └── __init__.py
│
├── ui/                             # Streamlit UI components
│   ├── components.py               # render_filters(), render_chat_sidebar()
│   ├── styles.py                   # CSS injection
│   └── __init__.py
│
├── tests/                          # Test suite
│   ├── test_structured_pipeline.py # Structured query pipeline E2E tests
│   ├── test_agent_quality.py       # Agent response quality evaluation
│   ├── test_agent_quality_v3.py    # Agent quality v3 evaluation
│   ├── test_rag_quality.py         # RAG retrieval quality metrics
│   ├── test_rag_real.py            # Live RAG execution tests
│   ├── test_e2e_performance.py     # End-to-end performance benchmarking
│   ├── test_routing.py             # SmartRouter vs QueryRouter comparison
│   ├── kappa_calculator.py         # Inter-rater agreement (Cohen's Kappa)
│   └── generate_table_5_19.py      # Evaluation table generator
│
├── benchmark/
│   └── benchmark_pipeline.py       # Pipeline performance measurements
│
└── .devcontainer/
    └── devcontainer.json           # Dev Container configuration
```

---

## Getting Started

### Prerequisites

- Python 3.11
- PostgreSQL database (or a [Supabase](https://supabase.com) project)
- Google Cloud project with Vertex AI enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)

### Installation

```bash
# Clone the repository
git clone git@github.com:TuyetMay/Insight-Analysis-Agent-Project.git
cd Insight-Analysis-Agent-Project

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# ── Database ──────────────────────────────────────────────────
DB_HOST=db.<your-supabase-project-id>.supabase.co
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=<your-database-password>
DB_TABLE=superstore

# ── Google Vertex AI ──────────────────────────────────────────
GCP_PROJECT=<your-gcp-project-id>
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-1.5-flash-002

# ── OpenAI (optional fallback) ────────────────────────────────
OPENAI_API_KEY=<your-openai-api-key>

# ── LangSmith (optional tracing) ─────────────────────────────
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-langsmith-api-key>
LANGCHAIN_PROJECT=superstore-bi
```

### Database Setup

1. Create a PostgreSQL database (Supabase free tier works out of the box).
2. Import the Superstore dataset:

```bash
# Using psql
psql -h <DB_HOST> -U postgres -d postgres -f superstore.sql

# Or load directly from CSV via the provided data loader
python -c "from core.data_loader import load_csv_to_db; load_csv_to_db('Superstore_clean.csv')"
```

3. Verify the `superstore` table exists and contains data before starting the app.

### Running the App

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## How It Works

### Query Processing Pipeline

Every user message sent to the chat sidebar flows through the following pipeline:

```
User message
    │
    ▼
1. Quick-token check (e.g. "__quick_sales__")   ← instant, no LLM
    │  no match
    ▼
2. SmartRouter (Gemini LLM)
    │  classifies as: "structured" | "agent" | "hybrid"
    │
    ├─► "agent"    → AgentOrchestrator (multi-turn loop)
    │
    ├─► "hybrid"   → HybridExecutor
    │                  ├─ structured SQL execution
    │                  └─ LLM narrative explanation
    │
    └─► "structured" ──►
            │
            ▼
        Tier 1: NLParser fast path
            fast KPI answer?  YES → return immediately
            │  NO
            ▼
        Tier 2: NLParser rule-based plan
            valid plan? YES → SQLBuilder → PostgreSQL → AnswerFormatter
            │  NO
            ▼
        Tier 3: Gemini + RAG
            RAGEngine.retrieve() → build context
            Gemini generates plan → PlanValidator → LLMPlanAuditor
            SQLBuilder → PostgreSQL execute
            InsightGenerator → narrative
            AnswerFormatter → final response
    │
    ▼
SuggestionEngine (rule + RAG) → 3 follow-up questions
    │
    ▼
Display in Streamlit chat
```

### RAG System

The RAG engine uses a **two-layer retrieval** approach:

1. **TFIDF Retriever** — Scores knowledge chunks against the query using TF-IDF similarity.
2. **MetadataFilter** — Re-ranks results using soft priority weights (entity matches, recency, source type).
3. **HyDE (Hypothetical Document Embeddings)** — Generates a hypothetical answer to the query, then retrieves documents similar to that answer, improving recall for vague questions.
4. **ExampleStore** — Stores successful `(question → execution plan)` pairs as few-shot examples injected into the LLM prompt.

### Agent Orchestration

When SmartRouter classifies a query as `"agent"`, the `AgentOrchestrator` takes over:

- Maintains a **multi-turn reasoning loop** with access to a set of tools (data queries, aggregations, comparisons).
- Each tool call result is fed back into the context for the next reasoning step.
- `AssumptionValidator` checks every intermediate claim against actual database values before surfacing them to the user.
- The loop terminates when the agent produces a final answer or exceeds the maximum number of steps.

### Suggestions Engine

After every response, the chatbot generates **3 follow-up question suggestions** using two parallel engines:

| Engine | Mechanism |
|--------|-----------|
| `RuleEngine` | Pattern-matches response type (metric, breakdown, time-period) → selects template suggestions |
| `RAGEngine` | Retrieves semantically similar past questions from the ExampleStore → surfaces related follow-ups |

The two lists are merged and deduplicated before being shown to the user.

---

## Testing

The test suite covers the full pipeline from query routing to answer quality.

```bash
# Run all tests
pytest tests/

# Structured query pipeline
pytest tests/test_structured_pipeline.py -v

# Agent response quality
pytest tests/test_agent_quality.py -v

# RAG retrieval quality
pytest tests/test_rag_quality.py -v

# Routing accuracy (SmartRouter vs QueryRouter)
pytest tests/test_routing.py -v

# End-to-end performance benchmarks
pytest tests/test_e2e_performance.py -v
```

Test outputs (JSON + plaintext) are written to the project root:

| File | Contents |
|------|----------|
| `agent_results.json` / `.txt` | Agent quality evaluation results |
| `rag_results.json` / `.txt` | RAG retrieval evaluation results |
| `pipeline_results.json` / `.txt` | Structured pipeline evaluation results |

Inter-rater agreement scores are computed with `tests/kappa_calculator.py` (Cohen's Kappa).

---

## Configuration Reference

All configuration is loaded from environment variables via `config.py`. The table below lists every supported variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | — | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `postgres` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | — | Database password |
| `DB_TABLE` | `superstore` | Target table name |
| `APP_TITLE` | `Superstore Business Intelligence Dashboard` | Streamlit page title |
| `APP_ICON` | `📊` | Browser tab icon |
| `GCP_PROJECT` | — | Google Cloud project ID |
| `GCP_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | `gemini-1.5-flash-002` | Gemini model ID |
| `OPENAI_API_KEY` | — | OpenAI key (optional fallback) |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | — | LangSmith project name |

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and add tests where appropriate.
3. Ensure all existing tests pass: `pytest tests/`
4. Commit with a clear message and open a pull request against `main`.

Please keep PRs focused — one logical change per PR makes reviews faster.

---

*Built with Streamlit · Powered by Google Gemini · Data: Sample Superstore Dataset*
