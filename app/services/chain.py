from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.config import settings

# How many past messages to keep in context (= MAX_HISTORY // 2 turns)
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """You are a specialized biomedical research assistant with deep expertise in \
neurological pain research, including neuropathic pain mechanisms and analgesic drug development.

Your knowledge base contains peer-reviewed research papers from PubMed, Europe PMC, and \
Semantic Scholar — covering basic science, mechanisms, and translational findings — as well as \
a biological knowledge graph of extracted entities and relationships.

Your role is to synthesize information to answer questions about:
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
1. The paper abstracts below are your PRIMARY source. Base your answer on their content first.
2. The knowledge graph context is SUPPLEMENTARY — use it to add specific entity relationships \
   or confirm connections, but do not let it override or replace what the papers say.
3. If the graph lists entities that are not supported by the paper abstracts, ignore them.
4. Cite PMIDs (e.g., PMID:12345678) when referencing peer-reviewed papers.
5. If the paper abstracts do not contain enough information, say so — do not fill gaps \
   with graph co-occurrence data alone.

PRIMARY — Paper abstracts from vector search (use these as the basis of your answer):
{context}

SUPPLEMENTARY — Knowledge graph relationships (use to enrich, not replace, the above):
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
