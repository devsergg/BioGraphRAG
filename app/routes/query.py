import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.services.pinecone_service import init_pinecone, vector_retrieve
from app.services.neo4j_service import Neo4jService
from app.services.reranker import rerank
from app.services.chain import generate_answer

router = APIRouter()


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
        1. Vector retrieval  (Pinecone semantic search)
        2. Graph traversal   (Neo4j Cypher via LLM)
        3. Reranking         (CrossEncoder cross-encoder/ms-marco-MiniLM-L-6-v2)
        4. LLM generation    (gpt-4o-mini with cited answer)
    """
    try:
        # --- Step 1: Vector retrieval ---
        vectorstore = init_pinecone()
        vector_docs = vector_retrieve(request.question, vectorstore, k=request.top_k)
        vector_summaries = [
            {
                "content": doc.page_content[:200],
                "metadata": doc.metadata,
            }
            for doc in vector_docs
        ]

        # --- Step 2: Graph traversal ---
        neo4j = Neo4jService()
        graph_response = neo4j.graph_query(request.question)
        neo4j.close()
        graph_result   = graph_response["result"]
        graph_cypher   = graph_response["cypher"]
        graph_db_rows  = graph_response["db_results"]

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
        # Sources can be trials (record_type="trial") or papers (record_type="paper")
        sources = []
        for r in reranked:
            meta = r["metadata"]
            record_type = meta.get("record_type", "trial")

            if record_type == "paper":
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
            else:
                sources.append({
                    "record_type":     "trial",
                    "nct_id":          meta.get("nct_id", ""),
                    "title":           meta.get("title", ""),
                    "sponsor":         meta.get("sponsor", ""),
                    "phase":           meta.get("phase", ""),
                    "status":          meta.get("status", ""),
                    "compound":        meta.get("search_compound", ""),
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
