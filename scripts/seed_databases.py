#!/usr/bin/env python3
"""
Seed script — populates Pinecone and Neo4j with peer-reviewed papers
from PubMed, Europe PMC, and/or Semantic Scholar.

Steps
-----
  [0] Clear Neo4j graph + delete/recreate Pinecone index  (skipped with --no-clear)
  [1] Fetch papers from selected sources, deduplicate by PMID
  [2] Extract biological entities from papers (GPT-4o-mini)
  [3] Upsert papers to Neo4j
  [4] Upsert papers to Pinecone
  [5] Print stats

Usage
-----
  # Initial test — PubMed only, clears databases first (default):
  python scripts/seed_databases.py --sources pubmed

  # After testing, append Europe PMC + Semantic Scholar WITHOUT clearing:
  python scripts/seed_databases.py --sources europepmc semanticscholar --no-clear

  # Full seed from scratch (all 3 sources, clears first):
  python scripts/seed_databases.py

  # Full seed appending to existing data (no clear):
  python scripts/seed_databases.py --no-clear

Available sources: pubmed, europepmc, semanticscholar
"""
import sys
import os
import time
import argparse

# Allow imports from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pinecone import Pinecone

from app.config import settings
from app.services.pubmed_fetcher import fetch_pubmed_papers
from app.services.europepmc_fetcher import fetch_europepmc_papers
from app.services.semanticscholar_fetcher import fetch_semanticscholar_papers
from app.services.entity_extractor import extract_entities_batch
from app.services.neo4j_service import Neo4jService
from app.services.pinecone_service import (
    init_pinecone,
    add_papers_to_pinecone,
)

# ─── Source registry ──────────────────────────────────────────────────────────
ALL_SOURCES = {
    "pubmed":          (fetch_pubmed_papers,          "PubMed"),
    "europepmc":       (fetch_europepmc_papers,       "EuropePMC"),
    "semanticscholar": (fetch_semanticscholar_papers, "SemanticScholar"),
}

# ─── Search terms (shared by all paper sources) ───────────────────────────────
SEARCH_TERMS = [
    "neuropathic pain",
    "chronic pain",
    "central sensitization",
    "spinal cord stimulation",
    "dorsal root ganglion",
    "sodium channel pain",
    "Nav1.7",
    "Nav1.8",
    "CGRP pain",
    "substance P pain",
    "ketamine pain",
    "low dose naltrexone",
    "gabapentin neuropathy",
    "pregabalin neuropathy",
    "glial cell pain",
    "neuroinflammation pain",
    "TRPV1 pain",
    "endocannabinoid pain",
]


# ─── Step 0: Clear databases ─────────────────────────────────────────────────

def reset_pinecone() -> None:
    """Delete the Pinecone index (if it exists) and poll until it's gone."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        print(f"  Deleting Pinecone index '{index_name}'...")
        pc.delete_index(index_name)
        while index_name in [idx.name for idx in pc.list_indexes()]:
            print("  Waiting for index deletion...")
            time.sleep(5)
        print("  Pinecone index deleted.")
    else:
        print("  Pinecone index does not exist — skipping deletion.")


# ─── Step 1: Fetch papers ─────────────────────────────────────────────────────

def fetch_papers_from_sources(
    sources: list[str],
    seen_pmids: set[str] | None = None,
) -> list[dict]:
    """
    Fetch up to 25 papers per (source × search_term) for the given sources.
    Deduplicates by PMID across all results.

    Pass `seen_pmids` to skip PMIDs already loaded in a previous run.
    """
    if seen_pmids is None:
        seen_pmids = set()

    papers: list[dict] = []
    total_terms = len(SEARCH_TERMS)

    for idx, term in enumerate(SEARCH_TERMS, 1):
        print(f"  Fetching papers for '{term}' [{idx}/{total_terms}]...")

        for source_key in sources:
            fetcher_fn, source_name = ALL_SOURCES[source_key]
            try:
                results = fetcher_fn(term)
            except Exception as exc:
                print(f"    {source_name} error: {exc}")
                results = []

            new = 0
            for paper in results:
                pmid = paper.get("pmid")
                if not pmid or pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                papers.append(paper)
                new += 1
            if new:
                print(f"    {source_name}: +{new} new papers ({len(papers)} total this run)")

        time.sleep(0.5)

    return papers


# ─── Main orchestration ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed Pinecone + Neo4j with peer-reviewed papers."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(ALL_SOURCES.keys()),
        default=list(ALL_SOURCES.keys()),
        metavar="SOURCE",
        help=(
            "Paper sources to fetch from. "
            "Choices: pubmed, europepmc, semanticscholar. "
            "Default: all three."
        ),
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help=(
            "Skip clearing Neo4j and Pinecone before seeding. "
            "Use this to APPEND new sources to an existing dataset "
            "without wiping what's already there."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_names = [ALL_SOURCES[s][1] for s in args.sources]

    print("=" * 60)
    print("Biotech GraphRAG Synthesizer — Database Seeding")
    print(f"Sources:  {', '.join(source_names)}")
    print(f"Clear DB: {'NO (--no-clear)' if args.no_clear else 'YES'}")
    print("=" * 60)

    # ── [0] Clear both databases (skipped with --no-clear) ────────────────────
    neo4j = Neo4jService()

    if args.no_clear:
        print("\n[0/4] Skipping database clear (--no-clear).")
        print("  Neo4j and Pinecone will be APPENDED to (MERGE prevents duplicates).")
    else:
        print("\n[0/4] Clearing databases...")
        neo4j.clear_graph()
        reset_pinecone()

    # ── [1] Fetch papers ──────────────────────────────────────────────────────
    print(f"\n[1/4] Fetching papers from: {', '.join(source_names)} × {len(SEARCH_TERMS)} terms...")
    papers = fetch_papers_from_sources(args.sources)
    print(f"  Total unique papers fetched this run: {len(papers)}")
    if not papers:
        print("  WARNING: No papers fetched. Check your network / API keys.")

    # ── [2] Extract entities ──────────────────────────────────────────────────
    if papers:
        print(f"\n[2/4] Extracting biological entities from {len(papers)} papers (GPT-4o-mini)...")
        print("  ~1 LLM call per paper, batches of 10 with 1.5s pause between batches")
        paper_entities = extract_entities_batch(papers)
        print(f"  Entity extraction complete for {len(paper_entities)} papers.")
    else:
        paper_entities = []
        print("\n[2/4] No papers to extract entities from — skipping.")

    # ── [3] Upsert papers to Neo4j ────────────────────────────────────────────
    if paper_entities:
        print(f"\n[3/4] Upserting {len(paper_entities)} papers to Neo4j...")
        neo4j.upsert_papers_batch(paper_entities)
        print("  Papers upserted to Neo4j.")
    else:
        print("\n[3/4] No papers to upsert to Neo4j — skipping.")

    neo4j_stats = neo4j.get_stats()
    neo4j.close()

    # ── [4] Init Pinecone + upsert papers ─────────────────────────────────────
    print("\n[4/4] Initializing Pinecone...")
    vectorstore = init_pinecone()
    if papers:
        print(f"  Upserting {len(papers)} papers to Pinecone...")
        add_papers_to_pinecone(papers, vectorstore)
        print("  Papers upserted to Pinecone.")
    else:
        print("  No papers to upsert to Pinecone — skipping.")

    # ── [5] Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Seeding complete!")
    print(f"  Papers ingested this run:  {len(papers)}")
    print("\n  Neo4j node counts (cumulative):")
    for label, count in sorted(neo4j_stats.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count}")
    print("=" * 60)
    print("\nVerification Cypher queries for Neo4j Browser:")
    print("  MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC")
    print("  MATCH (p:Paper)-[:MENTIONS]->(e) RETURN p.title, labels(e)[0], e.name LIMIT 20")

    if args.no_clear:
        print("\nTo add remaining sources later:")
        remaining = [s for s in ALL_SOURCES if s not in args.sources]
        if remaining:
            print(f"  python scripts/seed_databases.py --sources {' '.join(remaining)} --no-clear")


if __name__ == "__main__":
    main()
