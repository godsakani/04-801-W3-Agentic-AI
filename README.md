# Alumni RAG Agent v2.0

**CMU Africa Alumni Tracking and Support System** — A RAG-enabled multi-agent system for discovering alumni career updates and initiating personalized outreach.

## What's New in v2.0 (HW3)

| Feature | Description |
|---------|-------------|
| **Persistent Memory** | Cross-session continuity via MongoDB — the agent remembers past interactions |
| **Role Separation** | Planner → Executor → Critic architecture with distinct responsibilities |
| **Evaluation Framework** | 4 metrics (groundedness, tool accuracy, efficiency, completion) across 5 test cases |
| **Adaptive Control** | Closed-loop behavior: RE_RETRIEVE, RE_PLAN, ESCALATE, and CLARIFY actions |
| **Automated Discovery** | Tavily Search → LLM Extraction → Auto-Ingestion pipeline |

---

## Architecture

The system implements a **ReAct (Reason+Act)** loop with role-separated nodes:

```
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (ReActAgent)                │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐            │
│  │  PLANNER  │ →  │ EXECUTOR  │ →  │  CRITIC   │ → (loop)   │
│  │  (Plan)   │    │  (Act)    │    │ (Evaluate) │            │
│  └───────────┘    └───────────┘    └───────────┘            │
└─────────────────────────────────────────────────────────────┘
         │                │                │
         ▼                ▼                ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ RETRIEVAL│     │  TOOLS   │     │  VERIFY  │
   │  MODULE  │     │  MODULE  │     │  MODULE  │
   └──────────┘     └──────────┘     └──────────┘
         │                │                │
         ▼                ▼                ▼
   MongoDB Atlas    Tavily/Email/    Groundedness
   Vector Search    Survey/LinkedIn   Scorer
```

### Tool Status

| Tool | Status | Description |
|------|--------|-------------|
| **Tavily Search** | ✅ Active | Alumni discovery via Tavily API |
| **Email Sender** | ✅ Active | Email delivery with prerequisite validation |
| **LinkedIn Tool** | ✅ Active | Profile data formatting from search results |
| **Survey Tool** | ✅ Active | Career update survey generation |

### Role Separation

- **PlannerNode** — LLM-driven reasoning: decides which tool to call or whether to give a final answer
- **ExecutorNode** — Validates prerequisites, executes tools, handles errors
- **CriticNode** — Evaluates groundedness, recommends next action (proceed / re-plan / re-retrieve / escalate)

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Required variables:

| Variable | Purpose |
|----------|---------|
| `MONGODB_URI` | MongoDB Atlas connection string |
| `OPENAI_API_KEY` | OpenAI API key (or CMU AI Gateway) |
| `LANGCHAIN_API_KEY` | LangSmith tracing key |
| `TAVILY_API_KEY` | Tavily API key for alumni discovery |

### 4. Setup MongoDB Atlas

1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create database `alumni_db` with collections:
   - `alumni_vectors` — Vector store for alumni profiles
   - `agent_memory` — Persistent memory for session data
3. Create a Vector Search index named `alumni_vector_index`:

```json
{
  "type": "vectorSearch",
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1536, "similarity": "cosine"},
    {"type": "filter", "path": "metadata.alumni_id"},
    {"type": "filter", "path": "metadata.doc_type"}
  ]
}
```

---

## Usage

### Quick Start

```python
from src.agent import AlumniAgent, SAMPLE_ALUMNI

# Initialize agent (includes persistent memory + role separation)
agent = AlumniAgent()

# Option 1: Manual ingestion
agent.ingest_alumni(SAMPLE_ALUMNI)

# Option 2: Automated discovery pipeline
profiles = agent.discover_and_ingest(program="MSIT", year=2023)

# Run a query
result = agent.run("Tell me about alumni working in fintech")
print(result["response"])
print(f"Groundedness: {result['verification'].score}")
print(f"Session ID: {result['session_id']}")
```

### Running the Notebook

Open `notebooks/hw3_agent_execute.ipynb` for a full interactive demo covering:
1. Environment setup & agent initialization
2. Automated alumni discovery & ingestion
3. Annotated execution traces (Session 1 & 2)
4. Cross-session memory demonstration
5. Evaluation framework (5 test cases)
6. Adaptive control demonstration
7. Trace export

---

## Project Structure

```
alumni-rag-agent/
├── src/
│   ├── __init__.py               # Package init (v2.0.0)
│   ├── agent.py                  # AlumniAgent facade
│   ├── orchestrator.py           # ReActAgent loop
│   ├── models/                   # Data models
│   │   ├── state.py              #   AgentState dataclass
│   │   └── outputs.py            #   PlanOutput, ExecutionResult, CriticOutput
│   ├── nodes/                    # Role-separated nodes
│   │   ├── planner.py            #   PlannerNode (LLM reasoning)
│   │   ├── executor.py           #   ExecutorNode (tool execution)
│   │   └── critic.py             #   CriticNode (evaluation)
│   ├── memory/                   # Persistent memory
│   │   └── agent_memory.py       #   PersistentMemory (MongoDB)
│   ├── evaluation/               # Evaluation framework
│   │   ├── evaluation_framework.py  # EvaluationFramework + metrics
│   │   └── test_cases.py         #   TEST_CASES definitions
│   ├── retrieval/
│   │   └── mongodb_vector.py     # Vector search wrapper
│   ├── tools/
│   │   ├── tavily_search.py      # Tavily discovery tool
│   │   ├── linkedin.py           # LinkedIn profile tool
│   │   ├── email.py              # Email sender tool
│   │   └── survey.py             # Survey tool
│   ├── verification/
│   │   └── groundedness.py       # Groundedness scorer
│   ├── utils/
│   │   └── helpers.py            # Shared utilities
│   └── data/
│       └── sample_data.py        # Sample alumni profiles
├── notebooks/
│   └── hw3_agent_execute.ipynb   # Main demo notebook
├── docs/                         # Documentation
├── logs/                         # Exported traces
├── requirements.txt
├── .env.example
└── README.md
```

---

## Evaluation Results

The evaluation framework runs 5 structured test cases:

| Test | Groundedness | Tool Accuracy | Efficiency | Pass/Fail |
|------|:-----------:|:-------------:|:----------:|:---------:|
| Alumni Info Retrieval | 1.00 | 1.0 | 0.80 | ✅ PASS |
| Email Outreach | 0.11 | 0.0 | 0.00 | ❌ FAIL |
| LinkedIn Profile Check | 1.00 | 1.0 | 0.00 | ✅ PASS |
| Survey Distribution | 0.50 | 1.0 | 0.00 | ✅ PASS |
| Vague Request (Failure) | 1.00 | 0.0 | 0.80 | ❌ FAIL |

**Aggregate**: 60% pass rate · 0.72 avg groundedness · 1.00 task completion

> Test 2 fails due to prerequisite validation blocking `email_sender` (missing `personalization` param).
> Test 5 is an **intentional failure case** — the vague query demonstrates adaptive control and clarification behavior.

---

## LangSmith Tracing

All LangChain operations are traced when `LANGCHAIN_TRACING_V2=true`.

- **Project**: `alumni-rag-agent-hw3`
- **Dashboard**: https://smith.langchain.com/

---

## HW3 Deliverables

- [ ] Persistent Memory — MongoDB-backed cross-session continuity
- [ ] Role Separation — Planner / Executor / Critic nodes
- [ ] Evaluation Framework — 4 metrics, 5 test cases
- [ ] Adaptive Control — Closed-loop with RE_RETRIEVE, RE_PLAN, ESCALATE
- [ ] Automated Discovery — Tavily Search → LLM Extraction → Ingestion
- [ ] Annotated Execution Traces — Session 1 & 2 with role labels
- [ ] Cross-Session Memory Demo — Session 2 references Session 1

---

## License

Educational use only — CMU Africa.
