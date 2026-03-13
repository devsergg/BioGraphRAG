from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings

# How many past messages to keep in context (= MAX_HISTORY // 2 turns)
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """You are a specialized biomedical research assistant with deep expertise in \
neurological pain research, including neuropathic pain mechanisms, analgesic drug development, \
and clinical trial methodology.

Your knowledge base contains two complementary data sources:
1. Peer-reviewed research papers from PubMed, Europe PMC, and Semantic Scholar — covering \
   basic science, mechanisms, and translational findings
2. Clinical trial records from ClinicalTrials.gov — covering active, completed, and \
   recruiting interventional studies

Your role is to synthesize information across both sources to answer questions about:
- Neuropathic and chronic pain conditions (CRPS, fibromyalgia, diabetic neuropathy, central sensitization)
- Brain regions in pain processing: anterior cingulate cortex (ACC), periaqueductal gray (PAG), \
  dorsal horn, thalamus, insular cortex
- Ion channel targets: Nav1.7, Nav1.8, TRPV1, NMDA Receptor, CGRP Receptor, mu-opioid receptor
- Genes and proteins: BDNF, SCN9A, TNF-alpha, substance P, CGRP, IL-6
- Pain pathways: descending pain modulation, neuroinflammation cascade, wind-up, \
  long-term potentiation in pain
- Analgesic treatments: CGRP antagonists, ketamine, low-dose naltrexone, gabapentinoids, \
  endocannabinoids, sodium channel blockers
- Neuromodulation: spinal cord stimulation, dorsal root ganglion stimulation

Guidelines:
1. Cite NCT IDs (e.g., NCT01234567) when referencing clinical trials
2. Cite PMIDs (e.g., PMID:12345678) when referencing peer-reviewed papers
3. Clearly distinguish between established findings (completed trials / published papers) \
   and ongoing or early-phase research
4. Be precise about trial phases: Phase 1 = safety, Phase 2 = efficacy signals, \
   Phase 3 = confirmatory, Phase 4 = post-market
5. If the provided context does not contain enough information to answer the question, \
   respond with: "I don't have enough information in my current knowledge base to answer \
   this accurately."
6. Do not speculate or extrapolate beyond what the evidence in the context supports

Context from vector search (semantically relevant papers and trial records):
{context}

Context from knowledge graph (structured biological relationships):
{graph_context}
"""


def _to_lc_messages(history: list[dict]) -> list[HumanMessage | AIMessage]:
    """Convert [{role, content}] dicts to LangChain message objects."""
    messages = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def generate_answer(
    question: str,
    context: str,
    graph_context: str,
    history: list[dict] | None = None,
) -> str:
    """
    Generate a cited, evidence-grounded answer using the LCEL chain.

    Args:
        question:     The user's current natural language question.
        context:      Reranked vector search results as a concatenated string.
        graph_context: Neo4j graph query result string.
        history:      Prior conversation turns as [{role, content}] dicts.
                      Capped at MAX_HISTORY_MESSAGES entries.

    Returns:
        The LLM-generated answer string.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    # MessagesPlaceholder inserts prior turns between the system prompt and the
    # current question, giving the model full conversational context.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "question":     question,
        "context":      context,
        "graph_context": graph_context,
        "history":      _to_lc_messages(history or []),
    })
