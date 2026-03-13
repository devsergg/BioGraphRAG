"""
Ground truth Q&A pairs for evaluating the Biotech GraphRAG Synthesizer.

IMPORTANT: This file should be filled in AFTER running scripts/seed_databases.py.
The expected_answer and relevant_nct_ids fields need to be verified against
actual data in your Pinecone and Neo4j instances.

Categories:
  - relational: Tests graph traversal (Neo4j relationships)
  - semantic:   Tests vector similarity (Pinecone embeddings)
  - hybrid:     Requires both graph and vector retrieval
"""

GROUND_TRUTH_QA: list[dict] = [
    # -------------------------------------------------------------------------
    # RELATIONAL (10) — Tests graph traversal, sponsor/compound/condition links
    # -------------------------------------------------------------------------
    {
        "id": "rel_001",
        "question": "Which organizations are sponsoring clinical trials investigating Nav1.7 sodium channel inhibitors?",
        "expected_answer": "TODO: fill after seeding — query Neo4j: MATCH (o:Organization)-[:SPONSORS_TRIAL]->(t:Trial)-[:INVESTIGATES]->(c:Compound) WHERE c.name CONTAINS 'Nav1.7' RETURN o.name, t.nct_id",
        "relevant_nct_ids": [],  # TODO: populate from DB
        "category": "relational",
    },
    {
        "id": "rel_002",
        "question": "What conditions are being studied in ketamine pain trials?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_003",
        "question": "Which organizations research spinal cord stimulation for chronic pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_004",
        "question": "What compounds are being investigated for neuropathic pain conditions?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_005",
        "question": "Which sponsors are running trials that target CGRP for pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_006",
        "question": "What conditions does low-dose naltrexone target in clinical trials?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_007",
        "question": "Which organizations are researching dorsal root ganglion stimulation?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_008",
        "question": "What interventions are being tested for central sensitization?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_009",
        "question": "Which sponsors are running gabapentin or pregabalin neuropathy trials?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },
    {
        "id": "rel_010",
        "question": "What trials is the NIH sponsoring related to neuroinflammation and pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "relational",
    },

    # -------------------------------------------------------------------------
    # SEMANTIC (10) — Tests vector similarity search, concept-level questions
    # -------------------------------------------------------------------------
    {
        "id": "sem_001",
        "question": "Summarize recent clinical approaches to treating neuropathic pain using ion channel blockers.",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_002",
        "question": "What is the current research landscape for TRPV1 antagonists in pain management?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_003",
        "question": "Describe trials investigating the endocannabinoid system for chronic pain relief.",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_004",
        "question": "What is known from clinical trials about ketamine's efficacy for treatment-resistant pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_005",
        "question": "How is neuroinflammation being targeted in current pain clinical trials?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_006",
        "question": "Describe trials studying glial cell modulation for pain treatment.",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_007",
        "question": "What approaches are being studied for substance P inhibition in pain trials?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_008",
        "question": "What are the most common conditions studied alongside neuropathic pain in clinical trials?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_009",
        "question": "Summarize the use of spinal cord stimulation technologies in pain management trials.",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },
    {
        "id": "sem_010",
        "question": "What neuromodulation approaches are being investigated for chronic pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "semantic",
    },

    # -------------------------------------------------------------------------
    # HYBRID (10) — Requires both graph relationships and semantic context
    # -------------------------------------------------------------------------
    {
        "id": "hyb_001",
        "question": "Which Phase 2 or Phase 3 trials are investigating Nav1.7 or Nav1.8 inhibitors, and who are the sponsors?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_002",
        "question": "Compare the organizations running CGRP pain trials versus those running ketamine pain trials.",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_003",
        "question": "What Phase 2+ trials are studying central sensitization, and what compounds are being used?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_004",
        "question": "Which recruiting trials are investigating endocannabinoids for neuropathic pain, and what are their sponsors?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_005",
        "question": "Compare the trial phases of spinal cord stimulation trials versus dorsal root ganglion stimulation trials.",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_006",
        "question": "What are the completed trials for low-dose naltrexone in pain conditions, and which organizations ran them?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_007",
        "question": "Which Phase 1 trials are testing novel sodium channel blockers for neuropathic pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_008",
        "question": "What conditions are targeted in neuroinflammation pain trials, and which sponsors lead this research?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_009",
        "question": "Which organizations are conducting both CGRP and TRPV1 pain research, and in what trial phases?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
    {
        "id": "hyb_010",
        "question": "What glial cell-targeting compounds are in active recruiting trials for chronic pain?",
        "expected_answer": "TODO: fill after seeding",
        "relevant_nct_ids": [],
        "category": "hybrid",
    },
]
