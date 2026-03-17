import streamlit as st
import requests

# In production (Streamlit Cloud), add this under Settings → Secrets:
#   API_BASE = "https://your-app.railway.app/api"
#
# Locally, falls back to localhost so development workflow is unchanged.
API_BASE = st.secrets.get("API_BASE", "http://localhost:8000/api")

st.set_page_config(
    page_title="Biotech GraphRAG Synthesizer",
    page_icon="🧠",
    layout="wide",
)

st.title("Biotech GraphRAG Synthesizer")
st.caption(
    "Hybrid RAG for neurological pain research — "
    "papers (PubMed · EuropePMC · Semantic Scholar) + trials (ClinicalTrials.gov) "
    "· Neo4j biological graph · GPT-4o-mini"
)


def render_sources(sources: list[dict]):
    """Render a list of mixed paper/trial sources."""
    if not sources:
        return
    st.subheader("Sources")
    for src in sources:
        score = src.get("relevance_score", 0)
        title = src.get("title", "Untitled")
        record_type = src.get("record_type", "trial")

        if record_type == "paper":
            pmid    = src.get("pmid", "")
            journal = src.get("journal", "")
            year    = src.get("year", "")
            source  = src.get("source", "")
            label   = f"PMID:{pmid}" if pmid else "Paper"
            st.markdown(
                f"- 📄 **{label}** — {title}  "
                f"*({journal}, {year} | {source} | Score: {score:.3f})*"
            )
        else:
            nct_id  = src.get("nct_id", "")
            phase   = src.get("phase", "")
            status  = src.get("status", "")
            label   = nct_id if nct_id else "Trial"
            st.markdown(
                f"- 🧪 **{label}** — {title}  "
                f"*(Phase: {phase} | {status} | Score: {score:.3f})*"
            )


def render_reasoning_trace(trace: dict):
    """Render the full reasoning trace in an expander."""
    if not trace:
        return
    with st.expander("View Reasoning Trace"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Vector Search")
            vs = trace.get("vector_search", {})
            st.caption(f"Retrieved {vs.get('retrieved', 0)} documents")
            st.json(vs)

            st.subheader("🕸️ Graph Search")
            gs = trace.get("graph_search", {})
            graph_result = gs.get("result", "No result.")
            st.write(graph_result)

            # Generated Cypher — key for debugging
            cypher = gs.get("cypher", "")
            if cypher:
                st.caption("Generated Cypher query:")
                st.code(cypher, language="cypher")
            else:
                st.caption("No Cypher was generated.")

            # Raw DB rows returned by Neo4j
            db_rows = gs.get("db_results", [])
            if db_rows:
                st.caption(f"Neo4j returned {len(db_rows)} row(s):")
                st.json(db_rows)
            else:
                st.caption("Neo4j returned 0 rows.")

        with col2:
            st.subheader("📊 Reranking Scores")
            scores = trace.get("reranking", {}).get("scores", [])
            if scores:
                st.bar_chart({"Relevance Score": scores})
            else:
                st.write("No reranking scores available.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            render_reasoning_trace(message.get("reasoning_trace", {}))

# Chat input
if prompt := st.chat_input("Ask about pain mechanisms, receptors, brain regions, or clinical trials..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("Searching papers + graph + trials..."):
            try:
                # Build history: all messages before the current one (which we
                # just appended), capped at 10 entries (5 turns).  We only
                # send role + content — not sources or reasoning_trace.
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m["role"] in ("user", "assistant")
                ][-10:]

                response = requests.post(
                    f"{API_BASE}/query",
                    json={"question": prompt, "history": history},
                    timeout=90,  # CrossEncoder first-load can be slow
                )
                response.raise_for_status()
                data = response.json()

                answer          = data["answer"]
                sources         = data.get("sources", [])
                reasoning_trace = data.get("reasoning_trace", {})

                st.markdown(answer)
                render_sources(sources)
                render_reasoning_trace(reasoning_trace)

                st.session_state.messages.append({
                    "role":            "assistant",
                    "content":         answer,
                    "sources":         sources,
                    "reasoning_trace": reasoning_trace,
                })

            except requests.exceptions.Timeout:
                st.error("Request timed out (90s). The first query may be slow due to model loading. Try again.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure FastAPI is running: `uvicorn app.main:app --reload --port 8000`")
            except requests.exceptions.RequestException as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"API error: {str(e)}\n\n**Detail:** {detail}")
