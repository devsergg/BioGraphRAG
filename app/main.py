import os

# LangSmith env vars MUST be set before any LangChain imports
from app.config import settings

os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import query

app = FastAPI(
    title="Biotech GraphRAG Synthesizer",
    description="Hybrid RAG system for neurological pain research — peer-reviewed papers + biological knowledge graph",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}
