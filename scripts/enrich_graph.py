#!/usr/bin/env python3
"""
Co-occurrence graph enrichment — no re-seed required.

Scans the existing (:Paper)-[:MENTIONS]->(entity) edges and creates
(:entityA)-[:ASSOCIATED_WITH]->(:entityB) relationships between every
entity pair that is co-mentioned in ≥ MIN_PAPERS papers.

Pairs that already have a specific biological relationship
(BINDS_TO, INHIBITS, ACTIVATES, MODULATES, EXPRESSED_IN, ENCODED_BY,
INVOLVED_IN, ACTIVE_IN, UNDERLIES, COMORBID_WITH, TARGETS) are skipped —
the specific relationship is more informative than a generic ASSOCIATED_WITH.

Idempotent: MERGE prevents duplicates, so this is safe to re-run.

Usage
-----
  # Default (threshold = 2 shared papers):
  python scripts/enrich_graph.py

  # Stricter threshold — only very frequent co-occurrences:
  python scripts/enrich_graph.py --min-papers 3

  # Preview what WOULD be created without writing to Neo4j:
  python scripts/enrich_graph.py --dry-run

  # Dry run with a custom threshold:
  python scripts/enrich_graph.py --dry-run --min-papers 2
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase
from app.config import settings


# ─── Cypher queries ───────────────────────────────────────────────────────────

# id(e1) < id(e2) ensures each unordered pair is processed exactly once.
# The undirected [-] match on biological types catches relationships written
# in either direction, so we never overlay a generic edge on a specific one.

_ENRICH_QUERY = """\
MATCH (e1)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(e2)
WHERE id(e1) < id(e2)
WITH e1, e2, count(DISTINCT p) AS shared_papers
WHERE shared_papers >= $min_papers
  AND NOT (e1)-[:BINDS_TO|INHIBITS|ACTIVATES|MODULATES|EXPRESSED_IN
              |ENCODED_BY|INVOLVED_IN|ACTIVE_IN|UNDERLIES
              |COMORBID_WITH|TARGETS]-(e2)
MERGE (e1)-[:ASSOCIATED_WITH]->(e2)
RETURN count(*) AS created
"""

# Dry-run: identical WHERE clauses but returns preview rows instead of writing.
_DRY_RUN_QUERY = """\
MATCH (e1)<-[:MENTIONS]-(p:Paper)-[:MENTIONS]->(e2)
WHERE id(e1) < id(e2)
WITH e1, e2, count(DISTINCT p) AS shared_papers
WHERE shared_papers >= $min_papers
  AND NOT (e1)-[:BINDS_TO|INHIBITS|ACTIVATES|MODULATES|EXPRESSED_IN
              |ENCODED_BY|INVOLVED_IN|ACTIVE_IN|UNDERLIES
              |COMORBID_WITH|TARGETS]-(e2)
RETURN
  labels(e1)[0]  AS from_type,
  e1.name        AS from_name,
  labels(e2)[0]  AS to_type,
  e2.name        AS to_name,
  shared_papers
ORDER BY shared_papers DESC
LIMIT 60
"""

# Count ASSOCIATED_WITH edges grouped by (from_label → to_label) pair.
_ASSOC_STATS_QUERY = """\
MATCH (a)-[r:ASSOCIATED_WITH]->(b)
RETURN
  labels(a)[0] + ' → ' + labels(b)[0] AS pair_type,
  count(r) AS count
ORDER BY count DESC
"""

# Full relationship-type breakdown (before / after comparison).
_ALL_REL_COUNTS_QUERY = """\
MATCH ()-[r]->()
RETURN type(r) AS rel_type, count(r) AS count
ORDER BY count DESC
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_rel_counts(session) -> dict[str, int]:
    result = session.run(_ALL_REL_COUNTS_QUERY)
    return {row["rel_type"]: row["count"] for row in result}


def _get_assoc_stats(session) -> list[dict]:
    result = session.run(_ASSOC_STATS_QUERY)
    return [{"pair_type": row["pair_type"], "count": row["count"]} for row in result]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ASSOCIATED_WITH edges from paper co-occurrence data. "
            "Safe to run multiple times (MERGE is idempotent)."
        )
    )
    parser.add_argument(
        "--min-papers",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Minimum number of papers in which two entities must both appear "
            "before an ASSOCIATED_WITH edge is created (default: 2)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the top candidate pairs without writing anything to Neo4j.",
    )
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("Biotech GraphRAG Synthesizer — Graph Enrichment")
    print(f"Mode:        {'DRY RUN (no writes)' if args.dry_run else 'WRITE'}")
    print(f"Min papers:  {args.min_papers}")
    print("=" * 60)

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    try:
        if args.dry_run:
            # ── Dry run: preview only ─────────────────────────────────────
            print(f"\n[DRY RUN] Top candidate pairs (≥ {args.min_papers} shared papers):\n")
            with driver.session() as session:
                rows = session.run(_DRY_RUN_QUERY, min_papers=args.min_papers).data()

            if not rows:
                print(
                    "  No qualifying pairs found. "
                    "Try lowering --min-papers or check that MENTIONS edges exist."
                )
                return

            # Group and display
            print(f"  {'From type':<16} {'From entity':<35} {'To type':<16} {'To entity':<35} {'Papers':>6}")
            print("  " + "-" * 112)
            for row in rows:
                print(
                    f"  {row['from_type']:<16} {row['from_name']:<35} "
                    f"  {row['to_type']:<16} {row['to_name']:<35} {row['shared_papers']:>6}"
                )

            print(f"\n  (Showing up to 60 rows. Run without --dry-run to write these relationships.)")

        else:
            # ── Write mode ────────────────────────────────────────────────

            # Snapshot relationship counts before enrichment
            with driver.session() as session:
                counts_before = _get_rel_counts(session)

            assoc_before = counts_before.get("ASSOCIATED_WITH", 0)
            total_before  = sum(counts_before.values())

            print(f"\nRelationships before enrichment: {total_before:,}")
            print(f"  ASSOCIATED_WITH already present: {assoc_before:,}")

            # Run enrichment
            print(f"\nCreating ASSOCIATED_WITH edges (min_papers={args.min_papers})...")
            with driver.session() as session:
                result = session.run(
                    _ENRICH_QUERY, min_papers=args.min_papers
                ).single()
                created = result["created"] if result else 0

            # Snapshot after
            with driver.session() as session:
                counts_after = _get_rel_counts(session)
                assoc_stats  = _get_assoc_stats(session)

            assoc_after  = counts_after.get("ASSOCIATED_WITH", 0)
            total_after  = sum(counts_after.values())

            print(f"\nEnrichment complete!")
            print(f"  ASSOCIATED_WITH edges created this run: {created:,}")
            print(f"  ASSOCIATED_WITH total (including prior runs): {assoc_after:,}")
            print(f"  Total relationships: {total_before:,} → {total_after:,}  (+{total_after - total_before:,})")

            # Breakdown by entity-type pair
            if assoc_stats:
                print("\nASSOCIATED_WITH breakdown by entity-type pair:")
                for row in assoc_stats:
                    print(f"  {row['pair_type']:<40} {row['count']:>5}")

            # Full relationship type summary
            print("\nAll relationship types (after enrichment):")
            for rel_type, count in counts_after.items():
                delta = count - counts_before.get(rel_type, 0)
                delta_str = f"  (+{delta})" if delta > 0 else ""
                print(f"  {rel_type:<30} {count:>6}{delta_str}")

            print("\nVerification Cypher:")
            print("  MATCH (a)-[:ASSOCIATED_WITH]->(b)")
            print("  RETURN labels(a)[0], a.name, labels(b)[0], b.name LIMIT 20")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
