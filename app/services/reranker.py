from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# Lazy singleton — avoids 80MB model download on every cold start.
# The model is loaded on the first call to rerank() and cached in memory.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print("  [reranker] Loading CrossEncoder model (first time, ~80MB)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  [reranker] Model loaded.")
    return _model


def rerank(query: str, documents: list[Document], top_k: int = 3) -> list[dict]:
    """
    Score each (query, document) pair using the cross-encoder.
    Returns top_k results sorted by score descending.

    Each result dict:
        {
            "document": Document,
            "score": float,        # cast from numpy float32
            "content": str,
            "metadata": dict,
        }
    """
    if not documents:
        return []

    model = _get_model()
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)  # numpy array of float32

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "document": doc,
            "score": float(score),  # numpy float32 → Python float for JSON serialization
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc, score in ranked[:top_k]
    ]
