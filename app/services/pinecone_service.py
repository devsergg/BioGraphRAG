"""
Pinecone service — vector store for trials AND papers.

Vector ID prefixes
------------------
  "nct_{nct_id}"  — ClinicalTrials.gov trial
  "pmid_{pmid}"   — peer-reviewed paper

Both record types share the same index (pain-trials) and are distinguished
by the `record_type` metadata field ("trial" | "paper").
"""
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.config import settings

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
BATCH_SIZE      = 100


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=settings.openai_api_key,
    )


def init_pinecone() -> PineconeVectorStore:
    """
    Create the Pinecone index if it doesn't exist, poll until ready,
    and return a PineconeVectorStore instance.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"  Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Poll until ready — create_index() returns immediately
        # Pinecone 7.x: status is an object (.ready bool), not a dict
        while not pc.describe_index(index_name).status.ready:
            print("  Waiting for index to be ready...")
            time.sleep(5)
        print("  Pinecone index ready.")
    else:
        print(f"  Pinecone index '{index_name}' already exists.")

    index = pc.Index(index_name)
    embeddings = _get_embeddings()
    return PineconeVectorStore(index=index, embedding=embeddings)


# ─── Trial documents ─────────────────────────────────────────────────────────

def _create_trial_text(trial: dict) -> str:
    """Build a rich text string combining all trial fields for semantic embedding."""
    parts = [
        f"Title: {trial.get('title', '')}",
        f"Description: {trial.get('description', '')}",
        f"Conditions: {', '.join(trial.get('conditions', []))}",
        f"Interventions: {', '.join(trial.get('interventions', []))}",
        f"Compound: {trial.get('search_compound', '')}",
        f"Sponsor: {trial.get('sponsor', '')}",
        f"Phase: {', '.join(trial.get('phase', []))}",
        f"Status: {trial.get('status', '')}",
    ]
    return " | ".join(parts)


def trials_to_documents(trials: list[dict]) -> list[Document]:
    """Convert trial dicts to LangChain Documents with metadata."""
    docs = []
    for trial in trials:
        text = _create_trial_text(trial)
        metadata = {
            "record_type":     "trial",
            "nct_id":          trial.get("nct_id", ""),
            "title":           trial.get("title", ""),
            "sponsor":         trial.get("sponsor", ""),
            "phase":           ", ".join(trial.get("phase", [])),
            "status":          trial.get("status", ""),
            "search_compound": trial.get("search_compound", ""),
            "conditions":      ", ".join(trial.get("conditions", [])),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def add_trials_to_pinecone(trials: list[dict], vectorstore: PineconeVectorStore):
    """Batch upsert trial documents into Pinecone (100 per batch)."""
    documents = trials_to_documents(trials)
    total = len(documents)
    print(f"  Upserting {total} trial documents to Pinecone in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        # nct_ prefix prevents ID collision with pmid_ paper IDs
        ids = [f"nct_{doc.metadata['nct_id']}" for doc in batch]
        vectorstore.add_documents(batch, ids=ids)
        print(f"  Pinecone trials: {min(i + BATCH_SIZE, total)}/{total} upserted")
        time.sleep(0.5)


# ─── Paper documents ──────────────────────────────────────────────────────────

def _create_paper_text(paper: dict) -> str:
    """Build an embedding text string for a peer-reviewed paper."""
    abstract = paper.get("abstract", "")
    # Truncate abstract to avoid hitting embedding token limits
    if len(abstract) > 2000:
        abstract = abstract[:2000]
    parts = [
        f"Title: {paper.get('title', '')}",
        f"Abstract: {abstract}",
        f"Journal: {paper.get('journal', '')}",
        f"Year: {paper.get('year', '')}",
        f"Source: {paper.get('source', '')}",
    ]
    return " | ".join(parts)


def papers_to_documents(papers: list[dict]) -> list[Document]:
    """Convert paper dicts to LangChain Documents with metadata."""
    docs = []
    for paper in papers:
        text = _create_paper_text(paper)
        metadata = {
            "record_type": "paper",
            "pmid":        paper.get("pmid", ""),
            "title":       paper.get("title", ""),
            "journal":     paper.get("journal", ""),
            "year":        paper.get("year") or "",
            "source":      paper.get("source", ""),
            "doi":         paper.get("doi") or "",
            "authors":     ", ".join(paper.get("authors", []))[:500],
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def add_papers_to_pinecone(papers: list[dict], vectorstore: PineconeVectorStore):
    """Batch upsert paper documents into Pinecone (100 per batch)."""
    documents = papers_to_documents(papers)
    total = len(documents)
    print(f"  Upserting {total} paper documents to Pinecone in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        # pmid_ prefix prevents ID collision with nct_ trial IDs
        ids = [f"pmid_{doc.metadata['pmid']}" for doc in batch]
        vectorstore.add_documents(batch, ids=ids)
        print(f"  Pinecone papers: {min(i + BATCH_SIZE, total)}/{total} upserted")
        time.sleep(0.5)


# ─── Vector retrieval ─────────────────────────────────────────────────────────

def vector_retrieve(query: str, vectorstore: PineconeVectorStore, k: int = 5) -> list[Document]:
    """Return top-k semantically similar documents for the given query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
