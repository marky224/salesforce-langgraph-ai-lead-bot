# salesforce-langgraph-ai-lead-bot

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4+-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Salesforce](https://img.shields.io/badge/Salesforce-Agentforce-00A1E0?logo=salesforce&logoColor=white)](https://www.salesforce.com)
[![Azure](https://img.shields.io/badge/Azure-Container%20Apps-0078D4?logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/en-us/products/container-apps)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end AI sales lead generation system that qualifies prospects through natural conversation and automatically creates enriched records in Salesforce — with intelligent follow-up powered by Agentforce.

---

## Demo

> 🚧 **Live demo coming soon** — the chat widget will be embedded on [markandrewmarquez.com](https://markandrewmarquez.com)

<!-- TODO: Add screenshot/gif of the chat widget in action -->
<!-- TODO: Add screenshot of the Salesforce Lead + Tasks created -->

---

## How It Works

A visitor lands on the website and clicks the floating chat bubble. Alex, the AI solutions advisor, guides them through a natural conversation — exploring their challenges, understanding their timeline and budget, and collecting contact information. Once the conversation wraps up, the system automatically:

1. **Scores the lead** (0–100) based on budget, timeline, company size, decision-maker status, and pain points
2. **Creates a Lead** in Salesforce with all qualification data in custom fields
3. **Attaches the full transcript** as a linked Task record
4. **Triggers an Agentforce agent** that analyzes the conversation and creates prioritized follow-up Tasks, drafts a personalized email, and optionally creates an Opportunity for high-value leads

The conversation feels natural — the bot adapts if the visitor volunteers information early, handles objections with empathy, and gracefully captures whatever data is available if someone needs to leave mid-conversation.

---

## Architecture

```mermaid
graph TB
    subgraph "GitHub Pages"
        A[markandrewmarquez.com<br/>embed snippet]
    end

    subgraph "Azure Static Web Apps"
        B[widget.js + chat-widget.css<br/>nlux chat widget]
    end

    subgraph "Azure Container Apps"
        C[FastAPI Server]
        D[LangGraph Engine]
        E[LLM Provider<br/>Anthropic / OpenAI / Groq / xAI]
    end

    subgraph "Salesforce"
        F[Lead + Task Records]
        G[Record-Triggered Flow]
        H[Agentforce Agent]
        I[Follow-up Tasks<br/>+ Email + Opportunity]
    end

    A -->|loads cross-origin| B
    B -->|SSE stream| C
    C --> D
    D -->|API call| E
    D -->|Create Lead + Task| F
    F -->|triggers| G
    G -->|invokes| H
    H -->|creates| I
```

### Data Flow

```
Visitor types message
  → widget.js sends POST /chat/stream
    → FastAPI receives request
      → LangGraph: extraction node (parse new data from message)
      → LangGraph: router node (decide next conversation stage)
      → LangGraph: conversation node (generate AI reply)
    ← SSE stream: tokens sent back in real-time
  ← Chat bubble displays reply with typing effect

On conversation completion:
  → LangGraph: scoring node (compute 0-100 lead score)
  → LangGraph: salesforce node (create Lead + Task via API)
  → Salesforce: Record-Triggered Flow fires
    → Agentforce: analyzes transcript + creates follow-up actions
```

---

## Features

**Conversational AI Chatbot**
- Stateful multi-turn conversations with persistent memory across page reloads
- 7 conversation stages: greeting → discovery → qualification → objection handling → lead capture → confirmation → completion
- Natural stage transitions — adapts if the visitor shares information out of order
- Soft re-engagement when visitors try to exit early (one attempt, then graceful capture)
- Swappable LLM provider via environment variable (Anthropic, OpenAI, Groq, xAI)

**Lead Qualification & Scoring**
- Deterministic 0-100 scoring rubric across 6 dimensions
- Real-time data extraction from conversation context using structured JSON prompts
- Incremental data accumulation — never overwrites previously captured information
- Qualification completeness tracking to determine conversation readiness

**Salesforce Integration**
- OAuth 2.0 Connected App with Client Credentials Flow
- Lead creation with 6 custom fields (score, budget, timeline, pain points, company size, transcript ID)
- Full conversation transcript attached as a linked Task record
- Async API calls via thread pool to avoid blocking the FastAPI event loop

**Agentforce Automation**
- Record-Triggered Flow invokes the agent on Lead creation
- 4 agent topics: Lead Analysis, Follow-Up Tasks, Email Drafting, Opportunity Creation
- Priority-scaled actions: High leads get 3 tasks + email + opportunity, Low leads get 1 nurture task
- Personalized email drafts referencing specific pain points from the conversation

**Embeddable Widget**
- Floating chat bubble with dark theme matched to the host site
- Cross-origin deployment: widget files on Azure Static Web Apps, embed snippet on GitHub Pages
- SSE streaming for real-time token-by-token display
- Mobile responsive (fullscreen on small screens)
- Lazy-loaded — nlux and the greeting only initialize when the bubble is first clicked

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | [nlux](https://nlux.dev) | Embeddable chat widget with streaming support |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com) | REST API with SSE streaming endpoints |
| **AI Engine** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful agent graph with checkpointed memory |
| **LLM Abstraction** | [LangChain](https://python.langchain.com) | Swappable provider support |
| **CRM** | [Salesforce](https://salesforce.com) + [Agentforce](https://www.salesforce.com/agentforce/) | Lead management + AI-powered follow-up |
| **Hosting** | [Azure Container Apps](https://azure.microsoft.com/en-us/products/container-apps) | Serverless container deployment |
| **Widget Hosting** | [Azure Static Web Apps](https://azure.microsoft.com/en-us/products/app-service/static) | CDN-hosted widget files |
| **Website** | [GitHub Pages](https://pages.github.com) | Portfolio site with embed snippet |

---

## Project Structure

```
salesforce-langgraph-ai-lead-bot/
├── README.md
├── architecture.md
├── .env.example
├── .gitignore
│
├── backend/
│   ├── app/
│   │   ├── server.py                # FastAPI entrypoint
│   │   ├── config.py                # Settings + LLM provider factory
│   │   ├── graph/
│   │   │   ├── state.py             # LangGraph state schema
│   │   │   ├── nodes.py             # All graph node functions
│   │   │   ├── edges.py             # Conditional routing logic
│   │   │   ├── graph.py             # Graph builder + compilation
│   │   │   └── prompts.py           # System prompts for each stage
│   │   ├── tools/
│   │   │   ├── salesforce.py        # Salesforce API wrappers
│   │   │   └── qualification.py     # Deterministic lead scoring
│   │   └── models/
│   │       └── schemas.py           # Pydantic models + enums
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_graph.py            # 39 node unit tests
│   │   ├── test_tools.py            # 21 tool + scoring tests
│   │   └── test_e2e.py              # 3 end-to-end conversation tests
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── DEPLOY.md
│
├── frontend/
│   ├── widget.js                    # nlux chat widget + SSE adapter
│   ├── chat-widget.css              # Widget styles (dark theme)
│   ├── embed-snippet.html           # Minimal code for GitHub Pages
│   └── index.html                   # Standalone demo page
│
└── salesforce/
    ├── connected-app-setup.md
    ├── custom-fields.md
    ├── agentforce/
    │   ├── agent-instructions.md
    │   ├── agent-topics.md
    │   └── agent-setup.md
    └── flows/
        ├── Lead_Created_Flow.md
        └── flow-setup.md
```

---

## Setup

### Prerequisites

- Python 3.12+
- Docker (for containerized deployment)
- A Salesforce Developer Edition org ([free signup](https://developer.salesforce.com/signup))
- An API key for at least one LLM provider
- An Azure subscription (for deployment)

### 1. Clone and configure

```bash
git clone https://github.com/marky224/salesforce-langgraph-ai-lead-bot.git
cd salesforce-langgraph-ai-lead-bot
cp .env.example .env
# Edit .env with your API keys and Salesforce credentials
```

### 2. Set up Salesforce

Follow these guides in order:

1. [Custom Fields](salesforce/custom-fields.md) — create the 6 custom fields on the Lead object
2. [Connected App](salesforce/connected-app-setup.md) — set up OAuth authentication
3. [Record-Triggered Flow](salesforce/flows/flow-setup.md) — create the automation flow
4. [Agentforce Agent](salesforce/agentforce/agent-setup.md) — configure the AI follow-up agent

### 3. Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.server:app --reload --port 8000
```

The API is now running at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 4. Test the chat widget

Open `frontend/index.html` in a browser (or use VS Code Live Server). The chat bubble should appear in the bottom-right corner. Click it to start a conversation.

### 5. Run tests

```bash
cd backend
pytest tests/ -v
```

All 63 tests should pass in under 2 seconds.

### 6. Deploy

See [DEPLOY.md](backend/DEPLOY.md) for full Azure Container Apps + Static Web Apps deployment instructions.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Synchronous chat — send message, get full reply |
| `POST` | `/chat/stream` | Streaming chat via SSE — real-time token delivery |
| `POST` | `/chat/init` | Start a new conversation and get the greeting |
| `GET` | `/health` | Liveness probe — returns version + timestamp |
| `GET` | `/health/salesforce` | Salesforce connectivity check + API usage |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: Chat Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "We need help automating our CRM workflows"}'
```

```json
{
  "reply": "Hey there! I'm Alex. That sounds like a great area to explore...",
  "thread_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "stage": "discovery",
  "is_complete": false,
  "lead_id": null
}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_PROVIDER` | No | `anthropic` | LLM backend: `anthropic`, `openai`, `groq`, `xai` |
| `ANTHROPIC_API_KEY` | Yes* | — | API key for the active provider |
| `SF_INSTANCE_URL` | Yes | — | Salesforce org URL |
| `SF_CLIENT_ID` | Yes | — | Connected App consumer key |
| `SF_CLIENT_SECRET` | Yes | — | Connected App consumer secret |
| `SF_USERNAME` | Yes | — | Integration user email |
| `SF_PASSWORD` | Yes | — | Integration user password |
| `SF_SECURITY_TOKEN` | No | — | Security token (if IP not relaxed) |
| `CORS_ORIGINS` | No | `localhost` | Comma-separated allowed origins |
| `LOG_LEVEL` | No | `INFO` | Python logging level |

\* Only the key for the configured `LLM_PROVIDER` is required.

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit and push
6. Open a Pull Request

---

## License

[MIT](LICENSE)

---

## Author

**Mark Marquez** — [markandrewmarquez.com](https://markandrewmarquez.com) · [GitHub](https://github.com/marky224) · [LinkedIn](https://www.linkedin.com/in/markandrewmarquez/)
