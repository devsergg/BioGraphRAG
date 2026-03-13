# Biotech GraphRAG Synthesizer

A hybrid Retrieval-Augmented Generation (RAG) system for neurological pain research. It answers complex questions about clinical trials by combining **semantic vector search** (Pinecone) with **knowledge graph traversal** (Neo4j), reranking results with a cross-encoder, and generating cited answers via GPT-4o-mini — all served through a FastAPI backend with a Streamlit chat frontend that shows a full reasoning trace.

## Architecture

```
User Query
    │
    ├── Vector Retrieval  (Pinecone + text-embedding-3-small)  → top 5 semantic matches
    ├── Graph Traversal   (Neo4j + GraphCypherQAChain)         → structured relationships
    │
    └── Combine → CrossEncoder Reranker → top 3
                                              │
                                              └── GPT-4o-mini → Cited Answer
```

## Prerequisites

- Python 3.11+
- [Pinecone](https://pinecone.io) account (free tier)
- [Neo4j Aura](https://neo4j.com/cloud/aura/) account (free tier)
- [OpenAI](https://platform.openai.com) API key
- [LangSmith](https://smith.langchain.com) account (free tier, optional but recommended)

## Setup

```bash
# 1. Clone / create the project directory and navigate into it
cd "Biotech GraphRAG Synthesizer"

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies (~800MB due to PyTorch via sentence-transformers)
pip install -r requirements.txt

# 4. Fill in your credentials
cp .env .env.backup   # .env already has the template
# Edit .env with your actual API keys
```

## .env Configuration

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=pain-trials
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=biotech-graphrag
```

## Seed the Databases (run once)

Fetches pain neurology trials from ClinicalTrials.gov and populates both Pinecone and Neo4j:

```bash
python scripts/seed_databases.py
```

This takes 5–15 minutes. Expect 100–300 unique trials across 18 pain-focused search terms.

## Run the Application

**Terminal 1 — FastAPI backend:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Streamlit frontend:**
```bash
streamlit run streamlit_app.py
```

Visit **http://localhost:8501** in your browser.

### API Health Check
```bash
curl http://localhost:8000/health
# → {"status": "ok", "version": "0.1.0"}
```

### Example Query
```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What Phase 2 trials are investigating Nav1.7 inhibitors for neuropathic pain?"}'
```

## Run Evaluations

After seeding and filling in `eval/ground_truth.py` with actual NCT IDs:

```bash
python eval/run_evals.py
```

Results appear in your [LangSmith dashboard](https://smith.langchain.com).

## Project Structure

```
├── app/
│   ├── config.py              # Pydantic Settings (.env loader)
│   ├── main.py                # FastAPI app + CORS + LangSmith init
│   ├── routes/
│   │   ├── ingest.py          # POST /api/ingest
│   │   └── query.py           # POST /api/query (full hybrid pipeline)
│   └── services/
│       ├── data_fetcher.py    # ClinicalTrials.gov + PubChem API
│       ├── pinecone_service.py # Embed + upsert + retrieve
│       ├── neo4j_service.py   # Graph CRUD + GraphCypherQAChain
│       ├── reranker.py        # CrossEncoder reranking
│       └── chain.py           # LangChain LCEL answer generation
├── scripts/
│   └── seed_databases.py      # One-time data population script
├── eval/
│   ├── ground_truth.py        # 30 Q&A pairs (fill after seeding)
│   └── run_evals.py           # LangSmith evaluation runner
├── streamlit_app.py           # Chat UI with reasoning trace
├── requirements.txt
└── .env                       # API keys (never commit)
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Orchestration | LangChain 1.x (LCEL) |
| Vector DB | Pinecone (serverless, text-embedding-3-small) |
| Graph DB | Neo4j Aura (GraphCypherQAChain via langchain-neo4j) |
| LLM | OpenAI gpt-4o-mini |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Tracing | LangSmith |
| Frontend | Streamlit |
