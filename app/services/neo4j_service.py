"""
Neo4j service — biological knowledge graph for neurological pain research.

Schema
------
Nodes (7 types):
  Paper       — peer-reviewed papers (key: pmid)
  BrainRegion — named brain/spinal regions (key: name)
  Receptor    — ion channels & receptors (key: name)
  GeneProtein — genes, proteins, cytokines (key: name)
  Disorder    — pain conditions & comorbidities (key: name)
  Pathway     — biological pathways & mechanisms (key: name)
  Intervention— treatment from papers (key: name)

Relationships:
  (:Paper)-[:MENTIONS]->(BrainRegion|Receptor|GeneProtein|Disorder|Pathway|Intervention)
  (:Intervention)-[:BINDS_TO|INHIBITS|ACTIVATES|MODULATES]->(:Receptor|Pathway)
  (:Receptor|GeneProtein)-[:EXPRESSED_IN]->(:BrainRegion)
  (:Receptor)-[:ENCODED_BY]->(:GeneProtein)
  (:GeneProtein)-[:INVOLVED_IN]->(:Pathway)
  (:Pathway)-[:ACTIVE_IN]->(:BrainRegion)
  (:Pathway)-[:UNDERLIES]->(:Disorder)
  (:Disorder)-[:COMORBID_WITH]->(:Disorder)
  (:Intervention)-[:TARGETS]->(:BrainRegion|Receptor)
"""
from __future__ import annotations

from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from app.config import settings

# ─── Custom Cypher generation prompt ─────────────────────────────────────────
# The default prompt generates exact name matches ({name: "value"}) which
# fail because nodes are stored with varied capitalisation and phrasing.
# This prompt enforces toLower() CONTAINS matching so "chronic pain" will
# match "Chronic Pain", "Chronic Back Pain Syndrome", etc.
_CYPHER_GENERATION_TEMPLATE = """You are an expert Neo4j Cypher query writer for a biomedical knowledge graph.

VALID RELATIONSHIP PATTERNS — only these (source)-[rel]->(target) combinations exist.
Do NOT invent patterns outside this list:

  (:Paper)-[:MENTIONS]->(BrainRegion|Receptor|GeneProtein|Disorder|Pathway|Intervention)
  (:Intervention)-[:BINDS_TO|INHIBITS|ACTIVATES|MODULATES|TARGETS]->(Receptor|Pathway|BrainRegion)
  (:Receptor)-[:EXPRESSED_IN]->(BrainRegion)
  (:GeneProtein)-[:EXPRESSED_IN]->(BrainRegion)
  (:Receptor)-[:ENCODED_BY]->(GeneProtein)
  (:GeneProtein)-[:INVOLVED_IN]->(Pathway)
  (:Pathway)-[:ACTIVE_IN]->(BrainRegion)
  (:Pathway)-[:UNDERLIES]->(Disorder)
  (:Disorder)-[:COMORBID_WITH]->(Disorder)
  (:Intervention)-[:TARGETS]->(BrainRegion|Receptor)

NODE LABELS and their meaning:
  BrainRegion  — brain/spinal regions (e.g. "Dorsal Horn", "Periaqueductal Gray")
  Receptor     — ion channels, receptors (e.g. "Nav1.7", "NMDA Receptor", "TRPV1")
  GeneProtein  — genes, proteins, cytokines (e.g. "BDNF", "TNF-alpha", "Substance P")
  Disorder     — pain conditions (e.g. "Neuropathic Pain", "Fibromyalgia")
  Pathway      — biological mechanisms (e.g. "Descending Pain Modulation", "Neuroinflammation")
  Intervention — drugs, treatments, devices (e.g. "Ketamine", "Gabapentin", "Spinal Cord Stimulation")

STRICT RULES:
1. NEVER use exact property matching like {{name: "value"}}.
   ALWAYS use: WHERE toLower(n.name) CONTAINS toLower("value")
2. ONLY use the relationship patterns listed above — never invent new ones.
3. Return only the raw Cypher query — no explanation, no markdown.
4. For multi-hop queries, chain patterns from the list above.

Examples:
  MATCH (p:Pathway)-[:ACTIVE_IN]->(b:BrainRegion) WHERE toLower(p.name) CONTAINS toLower("descending pain") RETURN p.name, b.name
  MATCH (i:Intervention)-[:INHIBITS|MODULATES]->(r:Receptor) WHERE toLower(i.name) CONTAINS toLower("ketamine") RETURN i.name, r.name
  MATCH (p:Pathway)-[:UNDERLIES]->(d:Disorder) WHERE toLower(d.name) CONTAINS toLower("neuropathic") RETURN p.name, d.name
  MATCH (r:Receptor)-[:EXPRESSED_IN]->(b:BrainRegion) WHERE toLower(r.name) CONTAINS toLower("nav1.7") RETURN r.name, b.name
  MATCH (g:GeneProtein)-[:INVOLVED_IN]->(p:Pathway)-[:UNDERLIES]->(d:Disorder) WHERE toLower(d.name) CONTAINS toLower("chronic pain") RETURN g.name, p.name

Question: {question}
Cypher:"""

_CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_CYPHER_GENERATION_TEMPLATE,
)
# ─────────────────────────────────────────────────────────────────────────────
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
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:BrainRegion)  REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Receptor)     REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:GeneProtein)  REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disorder)     REQUIRE n.name     IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Pathway)      REQUIRE n.name     IS UNIQUE",
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
                cypher_prompt=_CYPHER_GENERATION_PROMPT,
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
