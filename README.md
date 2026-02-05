# Alumni RAG Agent

This document describes the implementation of a RAG-enabled agent for the **CMU Africa Alumni Tracking and Support System**. The system automates the process of discovering alumni career updates and initiating personalized outreach.

## Overview

This project implements a three-module agentic system:

1. **Retrieval Module** - MongoDB Atlas Vector Search for alumni data
2. **Tool-Calling Module** - LinkedIn, Email, CRM, Survey tools with ReAct loop
3. **Verification Module** - Groundedness scoring and hallucination detection

## Architecture

The system implements a **ReAct (Reason+Act)** loop with the following tool verification status:

| Tool | Status | Description |
|------|--------|-------------|
| **Tavily Search** | Completed and Used | Uses Tavily API for robust alumni discovery |
| **Email Sender** | Completed and Used | Uses `smtplib` for actual email delivery |
| **LinkedIn Tool** | Completed but Not Used | This was to format the LinkedIn profile data to be used by the agent from Google Search |
| **Survey Tool** | Completed but Not Used | This requires the use of Google Forms API which is not available for use |

```
--------
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
│  ┌──────────────────────────────────────────────────┐   │
│  │ OBSERVE → REASON → DECIDE → ACT → UPDATE → LOOP  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ RETRIEVAL│   │  TOOLS   │   │  VERIFY  │
   │  MODULE  │   │  MODULE  │   │  MODULE  │
   └──────────┘   └──────────┘   └──────────┘

-----

## Setup

### 1. Setting Virtual Environment

```bash
 Create virtual environment
python -m venv venv

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy .env.example to .env and fill in your keys
cp .env.example .env
```

Required variables:
- `MONGODB_URI` - MongoDB Atlas connection string
- `OPENAI_API_KEY` - OpenAI API key
- `LANGCHAIN_API_KEY` - LangSmith tracing key
- `TAVILY_API_KEY` - Tavily API Key (for Discovery)
- `SMTP_HOST` - SMTP Server (e.g. smtp.gmail.com)
- `SMTP_USER` - SMTP Username
- `SMTP_PASSWORD` - SMTP App Password

### 4. Setup MongoDB Atlas

1. Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create database `alumni_db` with collection `alumni_vectors`
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

## Usage

### Quick Start

```python
from src.agent import AlumniAgent, SAMPLE_ALUMNI

# Initialize agent
agent = AlumniAgent()

# Ingest sample data
agent.ingest_alumni(SAMPLE_ALUMNI)

# Run a query
result = agent.run("Find alumni who work in fintech and send a check-in email")

print(result["response"])
print(f"Verification Score: {result['verification'].score}")
```

### Using the Notebook

Open `project.ipynb` for an interactive demo with all modules.

## Project Structure

```
alumni-rag-agent/
├── src/
│   ├── __init__.py
│   ├── agent.py              # Main agent with ReAct loop
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── mongodb_vector.py # Vector search wrapper
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── tavily_search.py  # Tavily search tool
│   │   ├── linkedin.py       # LinkedIn scraper
│   │   ├── email.py          # Email sender
│   │   └── survey.py         # Survey tool
│   └── verification/
│       ├── __init__.py
│       └── groundedness.py   # Groundedness scorer
├── notebooks/
│   ├── agent_excute.ipynb    # Agent Execution Notebook for running the agent
│   ├── useful.ipynb          # Contains useful code snippets and functions
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── README.md
├── technical_brief_report.md        # Technical Brief Report
```

## LangSmith Tracing

All LangChain operations are automatically traced when `LANGCHAIN_TRACING_V2=true`.

View traces at: https://smith.langchain.com/

## HW2 Deliverables

- [ ] Project-Integrated Code 
- [ ] Implementation Trace (export from LangSmith)
- [ ] Technical Brief (use `technical_brief_report.md`)

## Pending Tasks

- [ ] Search for alternative data scraping tools, sources for Alumni data
- [ ] Intergrate survey tool with Google Forms API
- [ ] Update the search tool as google custom search API and scraping of data from the web disabled
- [ ]  Add more tools to the agent such as analytics tools
- [ ] Implement Authentication and Authorization for the system access
- [ ]  Implement Complete dashboard for CMU Africa admin to interact with the agent

## License

Educational use only - CMU Africa.
