"""
Neo4j service — biological knowledge graph for neurological pain research.

Schema
------
Nodes (9 types):
  Paper       — peer-reviewed papers (key: pmid)
  Trial       — ClinicalTrials.gov entries (key: nct_id)
  BrainRegion — named brain/spinal regions (key: name)
  Receptor    — ion channels & receptors (key: name)
  GeneProtein — genes, proteins, cytokines (key: name)
  Disorder    — pain conditions & comorbidities (key: name)
  Pathway     — biological pathways & mechanisms (key: name)
  Compound    — drug/compound from trials (key: name)
  Intervention— treatment from papers (key: name)

Relationships:
  (:Paper)-[:MENTIONS]->(BrainRegion|Receptor|GeneProtein|Disorder|Pathway|Intervention)
  (:Trial)-[:INVESTIGATES]->(:Compound)
  (:Trial)-[:STUDIES]->(:Disorder)
  (:Compound|Intervention)-[:BINDS_TO|INHIBITS|ACTIVATES|MODULATES]->(:Receptor|Pathway)
  (:Receptor|GeneProtein)-[:EXPRESSED_IN]->(:BrainRegion)
  (:Receptor)-[:ENCODED_BY]->(:GeneProtein)
  (:GeneProtein)-[:INVOLVED_IN]->(:Pathway)
  (:Pathway)-[:ACTIVE_IN]->(:BrainRegion)
  (:Pathway)-[:UNDERLIES]->(:Disorder)
  (:Disorder)-[:COMORBID_WITH]->(:Disorder)
  (:Intervention)-[:TARGETS]->(:BrainRegion|Receptor)

Dropped from v1: Organization, Condition, SPONSORS_TRIAL, RESEARCHES, STUDIES_CONDITION
"""
from __future__ import annotations

from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

from app.config import settings
from app.services.entity_extractor import (
    PaperEntities,
    filter_valid_relationships,
)

# ─── Labels for entity nodes merged via Paper→MENTIONS ───────────────────────
_MENTION_LABELS: dict[str, str] = {
    "brain_regions":  "BrainRegion",
    "receptors":      "Receptor",
    "genes_proteins": "GeneProtein",
    "disorders":      "Disorder",
    "pathways":       "Pathway",
    "interventions":  "Intervention",
}

_CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Paper)        REQUIRE n.pmid     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Trial)        REQUIRE n.nct_id   IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:BrainRegion)  REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Receptor)     REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeneProtein)  REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disorder)     REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Pathway)      REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Compound)     REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Intervention) REQUIRE n.name     IS UNIQUE",
]


class Neo4jService:
    """Wraps a Neo4j driver + LangChain GraphCypherQAChain."""

    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self._lc_graph: Neo4jGraph | None = None
        self._chain: GraphCypherQAChain | None = None
        self._create_constraints()

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        with self.driver.session() as session:
            for cypher in _CONSTRAINTS:
                session.run(cypher)

    # ─── Graph clearing ───────────────────────────────────────────────────────

    def clear_graph(self):
        """
        Delete all nodes and relationships in batches of 10 000.
        Required for Neo4j Aura free tier (limited heap — bulk deletes OOM).
        """
        print("  Clearing Neo4j graph...")
        while True:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
                )
                deleted = result.single()["deleted"]
            print(f"    Deleted {deleted} nodes")
            if deleted == 0:
                break
        print("  Neo4j graph cleared.")

    # ─── Paper upsert ─────────────────────────────────────────────────────────

    def upsert_paper(self, paper: dict):
        """Merge a Paper node with its core metadata."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {pmid: $pmid})
                SET p.title   = $title,
                    p.journal = $journal,
                    p.year    = $year,
                    p.source  = $source,
                    p.doi     = $doi,
                    p.authors = $authors
                """,
                pmid=paper["pmid"],
                title=paper.get("title", ""),
                journal=paper.get("journal", ""),
                year=paper.get("year"),
                source=paper.get("source", ""),
                doi=paper.get("doi"),
                authors=", ".join(paper.get("authors", [])),
            )

    def upsert_paper_entities(self, paper: dict, entities: PaperEntities):
        """
        Merge all entity nodes extracted from a paper and create
        (:Paper)-[:MENTIONS]->(entity) edges.

        Entity-to-entity relationships from the LLM are whitelisted and
        written using f-string Cypher (safe because values are whitelist-validated).
        """
        pmid = paper["pmid"]

        # 1. MENTIONS edges for each entity category
        with self.driver.session() as session:
            for field, label in _MENTION_LABELS.items():
                names: list[str] = getattr(entities, field, [])
                for name in names:
                    if not name.strip():
                        continue
                    # f-string safe: label is a hardcoded dict value
                    session.run(
                        f"""
                        MERGE (e:{label} {{name: $name}})
                        WITH e
                        MATCH (p:Paper {{pmid: $pmid}})
                        MERGE (p)-[:MENTIONS]->(e)
                        """,
                        name=name.strip(),
                        pmid=pmid,
                    )

        # 2. Entity-to-entity biological relationships
        valid_rels = filter_valid_relationships(entities.relationships)
        with self.driver.session() as session:
            for rel in valid_rels:
                from_label = rel.from_type
                to_label   = rel.to_type
                relation   = rel.relation
                # All three values are whitelist-validated — f-string is safe
                session.run(
                    f"""
                    MERGE (a:{from_label} {{name: $from_name}})
                    MERGE (b:{to_label}   {{name: $to_name}})
                    MERGE (a)-[:{relation}]->(b)
                    """,
                    from_name=rel.from_entity.strip(),
                    to_name=rel.to_entity.strip(),
                )

    def upsert_papers_batch(self, paper_entities: list[tuple[dict, PaperEntities]]):
        """Upsert a list of (paper, entities) tuples."""
        total = len(paper_entities)
        for i, (paper, entities) in enumerate(paper_entities, 1):
            self.upsert_paper(paper)
            self.upsert_paper_entities(paper, entities)
            if i % 50 == 0 or i == total:
                print(f"  Neo4j papers: {i}/{total} upserted")

    # ─── Trial upsert ─────────────────────────────────────────────────────────

    def upsert_trial(self, trial: dict):
        """Merge a Trial node linked to Compound and Disorder nodes."""
        nct_id = trial.get("nct_id")
        if not nct_id:
            return

        with self.driver.session() as session:
            # Core Trial node
            session.run(
                """
                MERGE (t:Trial {nct_id: $nct_id})
                SET t.title       = $title,
                    t.phase       = $phase,
                    t.status      = $status,
                    t.description = $description
                """,
                nct_id=nct_id,
                title=trial.get("title", ""),
                phase=", ".join(trial.get("phase", [])),
                status=trial.get("status", ""),
                description=trial.get("description", "")[:2000],
            )

            # (:Trial)-[:INVESTIGATES]->(:Compound)
            interventions = trial.get("interventions", [])
            if isinstance(interventions, str):
                interventions = [interventions]
            for compound in interventions[:10]:
                compound = compound.strip()
                if compound:
                    session.run(
                        """
                        MERGE (c:Compound {name: $name})
                        WITH c
                        MATCH (t:Trial {nct_id: $nct_id})
                        MERGE (t)-[:INVESTIGATES]->(c)
                        """,
                        name=compound,
                        nct_id=nct_id,
                    )

            # (:Trial)-[:STUDIES]->(:Disorder)
            conditions = trial.get("conditions", [])
            if isinstance(conditions, str):
                conditions = [conditions]
            for condition in conditions[:10]:
                condition = condition.strip()
                if condition:
                    session.run(
                        """
                        MERGE (d:Disorder {name: $name})
                        WITH d
                        MATCH (t:Trial {nct_id: $nct_id})
                        MERGE (t)-[:STUDIES]->(d)
                        """,
                        name=condition,
                        nct_id=nct_id,
                    )

    def upsert_trials_batch(self, trials: list[dict]):
        """Upsert a list of trial dicts."""
        total = len(trials)
        for i, trial in enumerate(trials, 1):
            self.upsert_trial(trial)
            if i % 50 == 0 or i == total:
                print(f"  Neo4j trials: {i}/{total} upserted")

    # ─── Graph query (Cypher QA) ──────────────────────────────────────────────

    def graph_query(self, question: str) -> dict:
        """
        Use GraphCypherQAChain to translate a natural language question into
        a Cypher query, execute it, and return a dict with:
          - result:     the LLM-generated answer string
          - cypher:     the Cypher query that was generated and executed
          - db_results: the raw rows returned by Neo4j (list of dicts)

        Schema is refreshed on every call so stale cached schemas never
        cause the LLM to generate queries for the wrong node/rel types.
        """
        if self._chain is None:
            self._lc_graph = Neo4jGraph(
                url=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
            )
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=settings.openai_api_key,
            )
            self._chain = GraphCypherQAChain.from_llm(
                llm=llm,
                graph=self._lc_graph,
                allow_dangerous_requests=True,
                return_intermediate_steps=True,  # exposes generated Cypher
                verbose=False,
            )
        else:
            # Refresh so the LLM always sees the current schema
            self._lc_graph.refresh_schema()

        try:
            raw = self._chain.invoke({"query": question})
            answer     = raw.get("result", "")
            steps      = raw.get("intermediate_steps", [])

            # intermediate_steps is a list of dicts; first has "query" key
            cypher     = ""
            db_results = []
            for step in steps:
                if "query" in step and not cypher:
                    cypher = step["query"]
                if "context" in step:
                    db_results = step["context"]

            return {"result": answer, "cypher": cypher, "db_results": db_results}
        except Exception as exc:
            return {"result": f"Graph query error: {exc}", "cypher": "", "db_results": []}

    # ─── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return node-count breakdown by label."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
            )
            return {row["label"]: row["count"] for row in result}
