from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.services.data_fetcher import fetch_all_trials, fetch_trials
from app.services.pinecone_service import init_pinecone, add_trials_to_pinecone
from app.services.neo4j_service import Neo4jService

router = APIRouter()


class IngestRequest(BaseModel):
    search_terms: Optional[list[str]] = None


class IngestResponse(BaseModel):
    trials_fetched: int
    pinecone_upserted: int
    neo4j_upserted: int
    message: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest = IngestRequest()):
    """
    Trigger data ingestion from ClinicalTrials.gov into Pinecone and Neo4j.
    If search_terms is omitted, ingests all default SEARCH_TERMS.
    """
    try:
        if request.search_terms:
            # Partial ingest for specific terms
            raw_trials: list[dict] = []
            for term in request.search_terms:
                raw_trials.extend(fetch_trials(term))
            # Deduplicate by nct_id
            seen: dict[str, dict] = {}
            for t in raw_trials:
                if t["nct_id"] not in seen:
                    seen[t["nct_id"]] = t
            trials = list(seen.values())
        else:
            trials = fetch_all_trials()

        if not trials:
            raise HTTPException(
                status_code=422,
                detail="No trials fetched. Check search terms or API availability.",
            )

        # Upsert to Pinecone
        vectorstore = init_pinecone()
        add_trials_to_pinecone(trials, vectorstore)

        # Upsert to Neo4j
        neo4j = Neo4jService()
        neo4j.upsert_trials_batch(trials)
        neo4j.close()

        return IngestResponse(
            trials_fetched=len(trials),
            pinecone_upserted=len(trials),
            neo4j_upserted=len(trials),
            message="Ingest complete",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
