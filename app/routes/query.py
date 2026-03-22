import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.services.pinecone_service import init_pinecone, vector_retrieve
from app.services.neo4j_service import Neo4jService
from app.services.reranker import rerank
from app.services.chain import generate_answer

router = APIRouter()

# ── Module-level singletons ───────────────────────────────────────────────────
# Initialised once on first request and reused for all subsequent requests.
# Avoids re-creating connections and re-downloading models on every query.
_vectorstore = None
_neo4j: Neo4jService | None = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = init_pinecone()
    return _vectorstore


def _get_neo4j() -> Neo4jService:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jService()
    return _neo4j
# ─────────────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    history: list[dict] = []   # [{role: "user"|"assistant", content: "..."}]


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    reasoning_trace: dict


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Full hybrid RAG pipeline:
        1. Vector retrieval + Graph traversal  (run in parallel)
        2. Reranking         (CrossEncoder cross-encoder/ms-marco-MiniLM-L-6-v2)
        3. LLM generation    (gpt-4o-mini with cited answer)
    """
    try:
        # --- Steps 1 & 2: Vector retrieval + Graph traversal in parallel ---
        # Both are blocking (network + LLM calls) so we offload each to the
        # thread pool and await them together — cuts latency by ~40-50%.
        vectorstore = _get_vectorstore()
        neo4j       = _get_neo4j()

        vector_task = asyncio.to_thread(
            vector_retrieve, request.question, vectorstore, request.top_k
        )
        graph_task = asyncio.to_thread(neo4j.graph_query, request.question)

        vector_docs, graph_response = await asyncio.gather(vector_task, graph_task)

        vector_summaries = [
            {"content": doc.page_content[:200], "metadata": doc.metadata}
            for doc in vector_docs
        ]

        graph_result  = graph_response["result"]
        graph_cypher  = graph_response["cypher"]
        graph_db_rows = graph_response["db_results"]

        # --- Step 3: Reranking ---
        reranked = rerank(request.question, vector_docs, top_k=3)
        context = "\n\n---\n\n".join(r["content"] for r in reranked)

        # --- Step 4: LLM generation ---
        answer = generate_answer(
            question=request.question,
            context=context,
            graph_context=graph_result,
            history=request.history,
        )

        # --- Build response ---
        sources = []
        for r in reranked:
            meta = r["metadata"]
            sources.append({
                "record_type":     "paper",
                "pmid":            meta.get("pmid", ""),
                "title":           meta.get("title", ""),
                "journal":         meta.get("journal", ""),
                "year":            meta.get("year", ""),
                "source":          meta.get("source", ""),
                "doi":             meta.get("doi", ""),
                "relevance_score": r["score"],
                "content_preview": r["content"][:300],
            })

        reasoning_trace = {
            "vector_search": {
                "retrieved": len(vector_docs),
                "top_results": vector_summaries,
            },
            "graph_search": {
                "result":     graph_result,
                "cypher":     graph_cypher,
                "db_results": graph_db_rows[:10],  # cap for readability
            },
            "reranking": {
                "input_count": len(vector_docs),
                "output_count": len(reranked),
                "scores": [r["score"] for r in reranked],
            },
        }

        return QueryResponse(answer=answer, sources=sources, reasoning_trace=reasoning_trace)

    except Exception as e:
        logger.exception("Query pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))
