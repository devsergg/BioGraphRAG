"""
LLM-powered entity extractor for neurological pain research papers.

Uses GPT-4o-mini with structured output (Pydantic schema) to extract
biological entities and relationships from paper abstracts.

Returns list[tuple[dict, PaperEntities]] — paper dict co-located with
its extracted entities (avoids index alignment bugs).
"""
import time

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

from app.config import settings

# ─── Node type and relationship whitelists ────────────────────────────────────
# These must stay in sync with neo4j_service.py
VALID_NODE_LABELS = {
    "BrainRegion", "Receptor", "GeneProtein", "Disorder",
    "Pathway", "Intervention", "Compound",
}
VALID_RELATIONS = {
    "BINDS_TO", "INHIBITS", "ACTIVATES", "MODULATES",
    "EXPRESSED_IN", "ENCODED_BY", "INVOLVED_IN", "ACTIVE_IN",
    "UNDERLIES", "COMORBID_WITH", "TARGETS", "ASSOCIATED_WITH",
}

# ─── Label normalization ──────────────────────────────────────────────────────
# Maps common LLM variations → canonical VALID_NODE_LABELS values.
# Prevents silently dropping relationships where the LLM uses a near-synonym.
_LABEL_ALIASES: dict[str, str] = {
    # GeneProtein variations
    "gene":          "GeneProtein",
    "protein":       "GeneProtein",
    "gene/protein":  "GeneProtein",
    "geneprotein":   "GeneProtein",
    "cytokine":      "GeneProtein",
    "neuropeptide":  "GeneProtein",
    # Disorder variations
    "disease":       "Disorder",
    "condition":     "Disorder",
    "syndrome":      "Disorder",
    "disorder":      "Disorder",
    "pain condition":"Disorder",
    # Receptor variations
    "ion channel":   "Receptor",
    "channel":       "Receptor",
    "receptor":      "Receptor",
    # BrainRegion variations
    "brain region":  "BrainRegion",
    "brain area":    "BrainRegion",
    "spinal region": "BrainRegion",
    "neural region": "BrainRegion",
    "brainregion":   "BrainRegion",
    # Pathway variations
    "mechanism":     "Pathway",
    "signaling pathway": "Pathway",
    "cascade":       "Pathway",
    # Intervention/Compound variations
    "drug":          "Compound",
    "medication":    "Compound",
    "treatment":     "Intervention",
    "therapy":       "Intervention",
    "procedure":     "Intervention",
}

def _normalize_label(label: str) -> str:
    """Normalize a node type string to a canonical VALID_NODE_LABELS value."""
    stripped = label.strip()
    # Exact match first
    if stripped in VALID_NODE_LABELS:
        return stripped
    # Case-insensitive alias lookup
    return _LABEL_ALIASES.get(stripped.lower(), stripped)


BATCH_SIZE    = 10
SLEEP_BETWEEN = 1.5   # seconds between batches


# ─── Pydantic output models ───────────────────────────────────────────────────

class Relationship(BaseModel):
    from_entity: str
    from_type:   str   # must be a VALID_NODE_LABELS value
    relation:    str   # must be a VALID_RELATIONS value
    to_entity:   str
    to_type:     str   # must be a VALID_NODE_LABELS value


class PaperEntities(BaseModel):
    brain_regions:  list[str]
    receptors:      list[str]
    genes_proteins: list[str]
    disorders:      list[str]
    pathways:       list[str]
    interventions:  list[str]
    relationships:  list[Relationship]


# ─── Extraction prompt ────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """\
You are a biomedical knowledge-graph builder specializing in neurological pain research.

Given a paper title and abstract, extract biological entities and their relationships.

━━━ ENTITY TYPES ━━━
- brain_regions:  Named brain/spinal cord regions
    Examples: "Anterior Cingulate Cortex", "Dorsal Horn", "Periaqueductal Gray",
              "Thalamus", "Insular Cortex", "Spinal Cord", "Amygdala"
- receptors:      Ion channels, receptors, transporters
    Examples: "TRPV1", "Nav1.7", "Nav1.8", "NMDA Receptor", "CGRP Receptor",
              "mu-opioid receptor", "GABA-A Receptor", "AMPA Receptor"
- genes_proteins: Genes, proteins, peptides, cytokines, growth factors
    Examples: "BDNF", "SCN9A", "TNF-alpha", "Substance P", "CGRP", "IL-6",
              "NGF", "TrkA", "P2X3"
- disorders:      Pain conditions, neurological disorders, psychiatric comorbidities
    Examples: "Neuropathic Pain", "Fibromyalgia", "CRPS", "Central Sensitization",
              "Depression", "Anxiety", "Diabetic Neuropathy", "Migraine"
- pathways:       Biological pathways, mechanisms, physiological processes
    Examples: "Descending Pain Modulation", "Neuroinflammation", "Central Sensitization",
              "Wind-Up", "Long-Term Potentiation", "Synaptic Plasticity",
              "Glial Activation", "Endocannabinoid Signaling"
- interventions:  Drugs, devices, therapies, stimulation methods
    Examples: "Ketamine", "Gabapentin", "Spinal Cord Stimulation",
              "Low-Dose Naltrexone", "Pregabalin", "Transcranial Magnetic Stimulation"

━━━ RELATIONSHIP TYPES (use ONLY these exact strings) ━━━
  BINDS_TO        — molecule physically binds to target
  INHIBITS        — entity suppresses/blocks another
  ACTIVATES       — entity activates/upregulates another
  MODULATES       — entity changes activity of another (direction unspecified)
  EXPRESSED_IN    — receptor or gene expressed in a brain region
  ENCODED_BY      — receptor encoded by a gene
  INVOLVED_IN     — entity participates in a pathway
  ACTIVE_IN       — pathway or receptor is functionally active in a region
  UNDERLIES       — pathway/mechanism underlies a disorder
  COMORBID_WITH   — disorder co-occurs with another disorder
  TARGETS         — intervention/compound targets a receptor or region
  ASSOCIATED_WITH — general association when no more specific type fits

━━━ NODE TYPE VALUES (use ONLY these exact strings for from_type / to_type) ━━━
  BrainRegion | Receptor | GeneProtein | Disorder | Pathway | Intervention | Compound

━━━ RELATIONSHIP EXAMPLES ━━━
  {"from_entity": "Nav1.7",          "from_type": "Receptor",     "relation": "EXPRESSED_IN",  "to_entity": "Dorsal Root Ganglion",         "to_type": "BrainRegion"}
  {"from_entity": "BDNF",            "from_type": "GeneProtein",  "relation": "ACTIVATES",     "to_entity": "NMDA Receptor",               "to_type": "Receptor"}
  {"from_entity": "Ketamine",        "from_type": "Intervention", "relation": "INHIBITS",      "to_entity": "NMDA Receptor",               "to_type": "Receptor"}
  {"from_entity": "Neuroinflammation","from_type": "Pathway",      "relation": "UNDERLIES",     "to_entity": "Neuropathic Pain",            "to_type": "Disorder"}
  {"from_entity": "TNF-alpha",       "from_type": "GeneProtein",  "relation": "INVOLVED_IN",   "to_entity": "Neuroinflammation",           "to_type": "Pathway"}
  {"from_entity": "Fibromyalgia",    "from_type": "Disorder",     "relation": "COMORBID_WITH", "to_entity": "Depression",                  "to_type": "Disorder"}
  {"from_entity": "TRPV1",           "from_type": "Receptor",     "relation": "ACTIVE_IN",     "to_entity": "Dorsal Horn",                 "to_type": "BrainRegion"}

━━━ RULES ━━━
1. Extract what is explicitly stated OR strongly implied by the study design/findings
   (e.g., if a paper studies how Gabapentin reduces TRPV1 activity, extract INHIBITS)
2. Use canonical biomedical names — full names preferred (write "TRPV1" not "the receptor")
3. from_type and to_type MUST be one of the 7 node type values listed above — no other values
4. relation MUST be one of the 12 relationship type strings listed above — no other values
5. Be generous with relationships — aim for 3–8 relationships per paper when the data supports it
6. If nothing relevant exists in a category, return an empty list
"""


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )


def extract_entities(paper: dict) -> PaperEntities:
    """
    Extract biological entities from a single paper's title + abstract.
    Returns empty PaperEntities on any failure — never raises.
    """
    llm = _build_llm()
    structured_llm = llm.with_structured_output(PaperEntities)

    user_msg = (
        f"Title: {paper.get('title', '')}\n\n"
        f"Abstract: {paper.get('abstract', '')}"
    )

    try:
        result = structured_llm.invoke([
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user",   "content": user_msg},
        ])
        return result
    except Exception:
        return PaperEntities(
            brain_regions=[], receptors=[], genes_proteins=[],
            disorders=[], pathways=[], interventions=[], relationships=[],
        )


def extract_entities_batch(
    papers: list[dict],
) -> list[tuple[dict, PaperEntities]]:
    """
    Extract entities for a list of papers in batches of BATCH_SIZE.
    Sleeps SLEEP_BETWEEN seconds between batches to avoid rate-limiting.

    Returns list of (paper_dict, PaperEntities) tuples — co-located so
    downstream code never has index alignment issues.
    """
    results: list[tuple[dict, PaperEntities]] = []
    total = len(papers)

    for start in range(0, total, BATCH_SIZE):
        batch = papers[start : start + BATCH_SIZE]
        print(f"  Entity extraction: {min(start + BATCH_SIZE, total)}/{total} papers...")
        for paper in batch:
            entities = extract_entities(paper)
            results.append((paper, entities))
        if start + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN)

    return results


def filter_valid_relationships(
    relationships: list[Relationship],
) -> list[Relationship]:
    """
    Normalize node type labels then filter out any relationships where
    the (normalized) types or relation are not in the canonical whitelists.

    Normalization catches common LLM variations before filtering, so
    fewer valid relationships are silently dropped.
    """
    valid = []
    for rel in relationships:
        from_type = _normalize_label(rel.from_type)
        to_type   = _normalize_label(rel.to_type)
        relation  = rel.relation.strip().upper()

        if (
            from_type in VALID_NODE_LABELS
            and to_type   in VALID_NODE_LABELS
            and relation  in VALID_RELATIONS
        ):
            # Return a copy with normalized types so neo4j_service writes them correctly
            valid.append(Relationship(
                from_entity=rel.from_entity,
                from_type=from_type,
                relation=relation,
                to_entity=rel.to_entity,
                to_type=to_type,
            ))
    return valid
