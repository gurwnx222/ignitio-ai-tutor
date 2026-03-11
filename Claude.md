# CLAUDE.md — ignitio-ai-tutor

## Project Overview

**ignitio-ai-tutor** is an AI-powered LangChain learning assistant built with LangGraph. It takes a user's meme request, generates a meme and identifies 3 core LangChain concepts used to build it, then teaches those concepts through explanations and code examples — followed by a learning test to verify understanding.

The system follows an **orchestrator → builder → teaching → critic** pipeline pattern, with a single conditional reflective loop on learning test failure.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.14 |
| Package Manager | `uv` (see `uv.lock`, `pyproject.toml`) |
| LLM Framework | LangChain + LangChain Community + LangChain Core |
| Graph / State Machine | LangGraph |
| Data Validation | Pydantic v2 |
| Environment Config | `dotenv` |

---

> **Note:** There is no standalone `edges.py`. All graph topology — node registration, edge definitions, and conditional routing — lives in `graph/builder.py`.

---

## Graph State Schema

Defined in `graph/state.py` using Pydantic v2:

```python
from pydantic import BaseModel, Field

class graph_state(BaseModel):
    user_query: str = Field(..., description="The user's query")
    meme_url: str = Field(..., description="The URL of the generated meme")
    meme_text: str = Field(..., description="The text used in the meme")
    concept_map: dict = Field(..., description="The concept map generated from the user's query")
    explanation: dict = Field(..., description="The explanation of the concept map — explaining which concepts are used in building the generated meme.")
    code_examples: dict = Field(..., description="The code examples used to generate the meme, including code for each concept in the concept map.")
    learning_test: dict = Field(..., description="Testing the user's understanding after explaining the concepts with code examples.")
```

> **Important:** `explanation`, `code_examples`, and `learning_test` are only populated after the Builder agent completes — never pre-generated. The Critic agent's pass/fail result determines whether the Teaching agent loops once for a simpler re-explanation.

---

## Node Responsibilities

### Orchestrator (`nodes/orchestrator/`)
- Receives the raw `user_query`
- Validates whether the query is a meme request (returns `NON_MEME_RESPONSE` if not)
- If valid, dispatches a structured task plan to the Builder, Teaching, and Critic agents
- **Runs once** at the start of the session

### Builder (`nodes/builder/`)
- Receives the validated `user_query`
- Generates `meme_url` — the URL of the produced meme image
- Generates `meme_text` — the top/bottom text used in the meme
- Builds `concept_map` — exactly 3 LangChain concepts that underpin the meme generation
- **Runs once** after the Orchestrator

### Teaching (`nodes/teaching/`)
- Receives the 3 concepts from `concept_map`
- Populates `explanation` — one clear educational explanation per concept, contextualised to the meme
- Populates `code_examples` — one practical LangChain code example per concept
- **Runs on-demand** — triggered after the Builder completes, and again (once only) if the Critic returns a fail result with a simpler explanation

### Critic (`nodes/critic/`)
- Receives `explanation` and `code_examples` after the Teaching agent completes
- Runs a `learning_test` — 3 questions to assess user understanding
- Evaluates the user's answers and returns pass or fail
- On **fail**: triggers one Teaching agent retry at a simpler level — the loop does not repeat more than once
- On **pass**: session ends successfully
- **Runs on-demand** — once after the initial Teaching pass, and once more after the retry if triggered

---

## Data Flow

```
user_query
    │
    ▼
[Orchestrator]
    │  validates meme request, dispatches agent task plan
    ▼
[Builder Agent]
    │  generates: meme_url, meme_text, concept_map (3 concepts)
    ▼
[Teaching Agent]
    │  populates: explanation, code_examples (per concept)
    ▼
[Critic Agent]
    │  runs: learning_test
    │
    ├── PASS ──▶ session complete
    │
    └── FAIL ──▶ [Teaching Agent — retry once, simpler level]
                     │
                     ▼
                 [Critic Agent — final evaluation]
                     │
                     ▼
                 session complete (no further loops)
```

---

## Key Conventions

- **No pre-generation:** Never populate `explanation`, `code_examples`, or `learning_test` before the Builder agent has completed. All teaching content is derived from the live `concept_map`.
- **Single retry loop:** The Critic → Teaching reflective loop runs **at most once**. Hard-enforce this via a `retry_count` flag in graph state or edge logic — never allow a second loop.
- **Prompt isolation:** Each node's prompt lives in its own `prompt.py` — keep business logic out of prompts and vice versa.
- **State immutability:** Treat `graph_state` as append-only where possible; avoid overwriting previously generated content.
- **LLM config centralised:** All LLM initialisation goes in `core/llm.py`. Nodes import from there — never instantiate models inline.
- **Environment variables:** All secrets and API keys go in `.env` and are loaded via `dotenv`. Never hardcode credentials.

---

## Environment Setup

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Copy and fill in environment variables
cp .env.example .env
```

### Required `.env` Variables
```
OPENAI_API_KEY=...        # or whichever LLM provider is configured in core/llm.py
# Add any other provider keys here
```

---

## Running the Project

```bash
uv main.py
```

---

## Development Notes for Claude Code

- When adding a **new node**, follow the `nodes/<name>/node.py` + `nodes/<name>/prompt.py` pattern
- When modifying **graph topology** (adding edges, changing flow), edit `graph/builder.py` — this is the single file for both graph construction and conditional routing
- The `graph_state` schema in `graph/state.py` is the **single source of truth** for what data exists in the graph — always update it when adding new state fields
- LangGraph nodes should accept and return `graph_state` (or a partial dict compatible with it)
- Keep prompts in `prompt.py` files as plain strings or `PromptTemplate` objects — no logic
- The Critic → Teaching retry loop **must be gated** — use a `retry_count` or `has_retried` field in state if needed, and enforce the one-loop cap in `graph/builder.py`