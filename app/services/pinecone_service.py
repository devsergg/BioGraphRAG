"""
Pinecone service — vector store for peer-reviewed papers.

Vector ID scheme
----------------
  "pmid_{pmid}_chunk_{i}"  — i-th semantic chunk of a paper's abstract
"""
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from app.config import settings
from app.services.chunker import semantic_chunk

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


# ─── Paper documents ──────────────────────────────────────────────────────────

def papers_to_documents(papers: list[dict]) -> tuple[list[Document], list[str]]:
    """
    Semantically chunk each paper's abstract and return one Document per chunk.
    Multiple chunks from the same paper share identical metadata (pmid, title, etc.)
    so any retrieved chunk can be traced back to its source paper.

    Returns (documents, ids) where ids follow the scheme pmid_{pmid}_chunk_{i}.
    """
    docs: list[Document] = []
    ids:  list[str]      = []

    for paper in papers:
        title    = paper.get("title", "")
        abstract = paper.get("abstract", "")
        pmid     = paper.get("pmid", "")

        chunks = semantic_chunk(title, abstract)
        if not chunks:
            continue

        metadata = {
            "record_type": "paper",
            "pmid":        pmid,
            "title":       title,
            "journal":     paper.get("journal", ""),
            "year":        paper.get("year") or "",
            "source":      paper.get("source", ""),
            "doi":         paper.get("doi") or "",
            "authors":     ", ".join(paper.get("authors", []))[:500],
        }

        for i, chunk in enumerate(chunks):
            docs.append(Document(page_content=chunk, metadata=metadata))
            ids.append(f"pmid_{pmid}_chunk_{i}")

    return docs, ids


def add_papers_to_pinecone(papers: list[dict], vectorstore: PineconeVectorStore):
    """Semantically chunk and batch-upsert paper documents into Pinecone."""
    documents, ids = papers_to_documents(papers)
    total = len(documents)
    print(f"  Upserting {total} chunks from {len(papers)} papers in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_ids  = ids[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch_docs, ids=batch_ids)
        print(f"  Pinecone chunks: {min(i + BATCH_SIZE, total)}/{total} upserted")
        time.sleep(0.5)


# ─── Vector retrieval ─────────────────────────────────────────────────────────

def vector_retrieve(query: str, vectorstore: PineconeVectorStore, k: int = 5) -> list[Document]:
    """Return top-k semantically similar documents for the given query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
