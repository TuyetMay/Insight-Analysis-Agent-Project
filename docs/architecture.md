# System Architecture — Superstore BI Assistant

## Mermaid Diagram (paste vào https://mermaid.live hoặc draw.io)

```mermaid
flowchart TD
    User(["👤 User"])

    %% ── FRONTEND ──────────────────────────────────────────────
    subgraph FE["🖥️  FRONT-END  (Streamlit)"]
        direction LR
        Dashboard["📊 Dashboard\nKPIs · Charts · Filters"]
        ChatUI["💬 AI Chat Assistant\nChat history · Quick buttons"]
    end

    User -- Prompt --> ChatUI
    Dashboard -- displays --> User

    %% ── BACK-END ──────────────────────────────────────────────
    subgraph BE["⚙️  BACK-END  (Python / Orchestrator)"]
        direction TB

        %% Layer 1 — Routing
        subgraph L1["① ROUTING LAYER"]
            SR["🔀 SmartRouter\n— Gemini LLM call —\nOutputs: structured | agent | hybrid"]
        end

        %% Layer 2 — Execution
        subgraph L2["② EXECUTION LAYER"]
            direction TB

            subgraph SP["Structured Path  (waterfall fallback)"]
                direction LR
                T1["⚡ Tier 1\nFast KPI\nRegex · No API"]
                T2["🔧 Tier 2\nRule SQL\n+ LLMPlanAuditor\nNo API"]
                T3["🧩 Tier 3\nGemini Plan\n+ RAG Context\nAPI call"]
                T1 -- FAIL --> T2
                T2 -- FAIL --> T3
            end

            subgraph AP["Agent Path"]
                AGT["🧠 AgentOrchestrator\nReAct Loop  (max 12 calls)\nTools: query_metric · find_anomalies\nget_trend · compare_periods\n+ AssumptionValidator"]
            end

            subgraph HP["Hybrid Path"]
                HYB["🔄 HybridExecutor\nSQL data  +  Agent explanation\n(1 merged LLM call)"]
            end

            subgraph KB["📚 RAG Engine  (feeds Tier 3)"]
                direction LR
                HyDE["HyDE\nQuery Expander"]
                TFIDF["TF-IDF\nRetriever"]
                MF["MetadataPreFilter\nSoft re-ranker"]
                EX["ExampleStore\nFew-shot plans"]
                HyDE --> TFIDF --> MF --> EX
            end

            T3 -- retrieve context --> KB
        end

        %% Layer 3 — Validation
        subgraph L3["③ VALIDATION LAYER  (Answer Quality Gates)"]
            direction LR
            V1["Layer 1\nResult checks\nempty · all-zero · margin"]
            V2["Layer 2\nGrounding score\nnumber traceability"]
            V3["Layer 3\nHybrid QA\nRule → LLM Judge\n(Gemini)"]
            V1 -- ok --> V2 -- ok --> V3
        end

        %% Layer 4 — Response
        subgraph L4["④ RESPONSE LAYER"]
            direction LR
            FMT["📝 AnswerFormatter\n+ InsightGenerator"]
            SGG["💡 Suggestion Engine\nRule-based + RAG-based"]
        end

        %% Escalation
        V3 -- "FAIL → escalate" --> AGT
        V1 -- "FAIL empty" --> T2
        V1 -- "FAIL all-zero" --> T3
    end

    %% ── External ──────────────────────────────────────────────
    DB[("🗄️  PostgreSQL\nSupabase")]
    GAPI["🔑 Google\nGemini API"]

    %% ── Connections ───────────────────────────────────────────
    ChatUI -- question --> SR

    SR -- structured --> SP
    SR -- agent      --> AP
    SR -- hybrid     --> HP

    SP -- answer candidate --> L3
    AP -- answer candidate --> L3
    HP -- answer candidate --> L3

    L3 -- PASS --> L4
    L4 -- answer + suggestions --> ChatUI
    L4 -- KPI data --> Dashboard

    SP -- SQL queries --> DB
    AP -- SQL queries --> DB
    KB -- pgvector queries --> DB

    GAPI -- LLM classify  --> SR
    GAPI -- plan generation --> T3
    GAPI -- tool calling   --> AGT
    GAPI -- LLM judge      --> V3
    GAPI -- insight        --> FMT
    GAPI -- suggestions    --> SGG
```

---

## ASCII Fallback (dùng khi không render được Mermaid)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FRONT-END  (Streamlit)                           │
│   ┌──────────────────────────┐    ┌──────────────────────────────────┐  │
│   │   📊 Dashboard           │    │   💬 AI Chat Assistant           │  │
│   │   KPIs · Charts          │    │   Chat history · Quick buttons   │  │
│   └──────────────────────────┘    └──────────────┬───────────────────┘  │
└──────────────────────────────────────────────────┼─────────────────────┘
                                                   │ question
┌──────────────────────────────────────────────────▼─────────────────────┐
│                        BACK-END  (Python)                               │
│                                                                         │
│  ╔═══════════════════════════════════════════════════════════════════╗  │
│  ║  ① ROUTING LAYER                                                 ║  │
│  ║     SmartRouter  ─── Gemini LLM call ───                         ║  │
│  ║     → structured | agent | hybrid                                ║  │
│  ╚════════════╤══════════════════╤═══════════════╤══════════════════╝  │
│               │ structured       │ agent         │ hybrid              │
│  ╔════════════▼══════════════════▼═══════════════▼══════════════════╗  │
│  ║  ② EXECUTION LAYER                                               ║  │
│  ║                                                                   ║  │
│  ║  STRUCTURED PATH (waterfall fallback):                           ║  │
│  ║  ┌───────────┐  FAIL  ┌───────────────────┐  FAIL  ┌──────────┐ ║  │
│  ║  │  Tier 1   │───────▶│      Tier 2        │───────▶│  Tier 3  │ ║  │
│  ║  │ Fast KPI  │        │  Rule SQL          │        │ Gemini   │ ║  │
│  ║  │ (Regex)   │        │  + LLMPlanAuditor  │        │ + RAG    │ ║  │
│  ║  │  No API   │        │  No API            │        │ API call │ ║  │
│  ║  └───────────┘        └───────────────────┘        └────┬─────┘ ║  │
│  ║                                                          │ retrieve ║  │
│  ║  ┌───────────────────────────────────────────────────────▼──────┐ ║  │
│  ║  │  📚 RAG ENGINE                                               │ ║  │
│  ║  │  HyDE Expander → TF-IDF Retriever → MetadataPreFilter       │ ║  │
│  ║  │  ExampleStore (few-shot)           pgvector (Supabase)       │ ║  │
│  ║  └──────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                   ║  │
│  ║  AGENT PATH:                                                      ║  │
│  ║  ┌───────────────────────────────────────────────────────────┐   ║  │
│  ║  │  🧠 AgentOrchestrator  (ReAct loop, max 12 calls)        │   ║  │
│  ║  │  Tools: query_metric · find_anomalies · get_trend        │   ║  │
│  ║  │         compare_periods · AssumptionValidator             │   ║  │
│  ║  └───────────────────────────────────────────────────────────┘   ║  │
│  ║                                                                   ║  │
│  ║  HYBRID PATH:                                                     ║  │
│  ║  ┌───────────────────────────────────────────────────────────┐   ║  │
│  ║  │  🔄 HybridExecutor  (SQL data  +  Agent explanation)      │   ║  │
│  ║  └───────────────────────────────────────────────────────────┘   ║  │
│  ╚═══════════════════════════════╤═══════════════════════════════════╝  │
│                                  │ answer candidate                     │
│  ╔═══════════════════════════════▼═══════════════════════════════════╗  │
│  ║  ③ VALIDATION LAYER  (Answer Quality Gates)                      ║  │
│  ║                                                                   ║  │
│  ║  Layer 1 ─────────────────── Layer 2 ─────────────────── Layer 3 ║  │
│  ║  Rule-based result checks    Grounding score             Hybrid   ║  │
│  ║  empty · all-zero · margin   number traceability         QA check ║  │
│  ║  → retry wider query         → ⚠️ warning tag            Rule +   ║  │
│  ║                                                          LLM Judge║  │
│  ║                                              FAIL ──────────────▶ ║  │
│  ║                                              escalate to Agent    ║  │
│  ╚═══════════════════════════════╤═══════════════════════════════════╝  │
│                                  │ PASS                                 │
│  ╔═══════════════════════════════▼═══════════════════════════════════╗  │
│  ║  ④ RESPONSE LAYER                                                ║  │
│  ║  AnswerFormatter · InsightGenerator                               ║  │
│  ║  Suggestion Engine: Rule-based + RAG-based                        ║  │
│  ╚═══════════════════════════════════════════════════════════════════╝  │
│                                                                         │
│  ═══════════════ External Services ═════════════════════════════════    │
│  🗄️  PostgreSQL (Supabase) ← SQL queries from Tier 1/2/3, Agent, RAG  │
│  🔑  Google Gemini API    ← SmartRouter · Tier 3 · Agent · Layer 3    │
└─────────────────────────────────────────────────────────────────────────┘
                                   │ answer + suggestions
                         ┌─────────▼──────────────────────┐
                         │     FRONT-END (Streamlit)       │
                         └────────────────────────────────┘
                                   │
                                👤 User
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 🔑 Gemini API call | Costs money, adds latency |
| No API label | Free, rule/regex-based |
| FAIL → | Escalation / fallback trigger |
| PASS → | Answer cleared, proceeds to next layer |
| → | Data flow direction |

## Component → File mapping

| Component | File |
|-----------|------|
| SmartRouter | `chatbot/smart_router.py` |
| Tier 1 Fast KPI | `chatbot/nl_parser.py` → `fast_kpi_answer()` |
| Tier 2 Rule SQL + LLMPlanAuditor | `chatbot/nl_parser.py` + `chatbot/llm_plan_auditor.py` |
| Tier 3 Gemini Plan | `chatbot/nl_parser.py` → `gemini_plan()` |
| AgentOrchestrator | `chatbot/agent/orchestrator.py` |
| Agent Tools | `chatbot/agent/tools.py` |
| AssumptionValidator | `chatbot/agent/assumption_validator.py` |
| HybridExecutor | `chatbot/hybrid_executor.py` |
| RAG Engine | `rag/engine.py` |
| HyDE Expander | `rag/hyde.py` |
| TF-IDF Retriever | `rag/retriever.py` |
| MetadataPreFilter | `rag/metadata_filter.py` |
| ExampleStore | `rag/example_store.py` |
| Validation L1/L2 | `chatbot/answer_validator.py` → `AnswerValidator` |
| Validation L3 | `chatbot/answer_validator.py` → `Layer3Validator` |
| AnswerFormatter | `chatbot/answer_formatter.py` |
| InsightGenerator | `chatbot/insight_generator.py` |
| Suggestion Engine | `chatbot/suggestions/rule_engine.py` + `rag_engine.py` |
| Orchestrator (glue) | `chatbot/orchestrator.py` |
