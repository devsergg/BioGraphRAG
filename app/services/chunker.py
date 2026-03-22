"""
Semantic chunker for biomedical paper abstracts.

Algorithm
---------
1. Split abstract into sentences (regex, no extra deps).
2. Embed each sentence with a lightweight local bi-encoder
   (all-MiniLM-L6-v2 — already pulled in by sentence-transformers).
3. Compute cosine similarity between every pair of adjacent sentences.
4. Mark breakpoints where similarity falls in the bottom 25th percentile
   (semantic valleys = topic shifts).
5. Merge the resulting segments greedily until each chunk approaches
   `target_tokens` (~200 by default).
6. Prepend the paper title to every chunk so each vector is self-contained.

Why a local model for chunking?
--------------------------------
The final Pinecone embeddings use OpenAI text-embedding-3-small, but that
model is only needed for the vectors themselves. Finding semantic breakpoints
only requires relative similarity between adjacent sentences, which the local
MiniLM model handles accurately and cheaply (no API calls, ~80ms per abstract).
"""
from __future__ import annotations

import re

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

TARGET_TOKENS = 200

# ─── Lazy singletons ──────────────────────────────────────────────────────────

_sentence_model: SentenceTransformer | None = None
_tokenizer: tiktoken.Encoding | None = None


def _get_sentence_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is None:
        print("  [chunker] Loading sentence encoder (all-MiniLM-L6-v2)...")
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [chunker] Sentence encoder loaded.")
    return _sentence_model


def _get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        # cl100k_base is used by text-embedding-3-small
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by a capital letter."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text))


# ─── Public API ───────────────────────────────────────────────────────────────

def semantic_chunk(
    title: str,
    abstract: str,
    target_tokens: int = TARGET_TOKENS,
) -> list[str]:
    """
    Split an abstract into semantically coherent chunks of ~target_tokens each.
    The paper title is prepended to every chunk so each vector is self-contained.

    Returns a list of chunk strings. If the abstract is too short to chunk
    (≤ 2 sentences or already within target_tokens), returns a single chunk.
    """
    sentences = _split_sentences(abstract)

    # Too short to chunk meaningfully
    if not sentences:
        return []
    if len(sentences) <= 2 or _count_tokens(abstract) <= target_tokens:
        return [f"{title}\n\n{abstract}"]

    # Embed sentences with normalized vectors so dot product = cosine similarity
    model = _get_sentence_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Cosine similarity between each adjacent pair
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Breakpoints at semantic valleys (bottom 25th percentile of similarity)
    threshold = float(np.percentile(similarities, 25))
    breakpoints = {i + 1 for i, sim in enumerate(similarities) if sim < threshold}

    # Group sentences into segments at breakpoints
    segments: list[str] = []
    start = 0
    for bp in sorted(breakpoints):
        segments.append(" ".join(sentences[start:bp]))
        start = bp
    segments.append(" ".join(sentences[start:]))

    # Greedily merge small adjacent segments toward target_tokens
    chunks: list[str] = []
    buffer = segments[0]
    for seg in segments[1:]:
        candidate = buffer + " " + seg
        if _count_tokens(candidate) <= target_tokens:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = seg
    chunks.append(buffer)

    return [f"{title}\n\n{chunk}" for chunk in chunks]
