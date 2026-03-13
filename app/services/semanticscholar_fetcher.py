"""
Semantic Scholar fetcher.

Uses the Semantic Scholar Graph API (search endpoint).
With an API key: 100 req/s.  Without: ~1 req/s.

Only papers with a PubMed ID are returned (for stable cross-source dedup).
Papers without an abstract are skipped.

Returns canonical paper dicts; never raises — returns [] on any error.
"""
import requests

from app.config import settings

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_RESULTS   = 25
FIELDS        = "title,abstract,year,authors,venue,externalIds"


def fetch_semanticscholar_papers(term: str) -> list[dict]:
    """
    Search Semantic Scholar for `term` and return up to MAX_RESULTS canonical paper dicts.
    """
    headers = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    params = {
        "query":  term,
        "limit":  MAX_RESULTS,
        "fields": FIELDS,
    }

    try:
        resp = requests.get(S2_SEARCH_URL, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    papers_raw = data.get("data", [])
    papers = []

    for paper in papers_raw:
        # Skip papers without an abstract
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue

        # Only keep papers indexed in PubMed (needed for dedup key)
        external_ids = paper.get("externalIds") or {}
        pmid = external_ids.get("PubMed")
        if not pmid:
            continue

        title = (paper.get("title") or "").strip()
        year_raw = paper.get("year")
        try:
            year = int(year_raw) if year_raw is not None else None
        except (ValueError, TypeError):
            year = None

        journal = (paper.get("venue") or "").strip()

        authors = []
        for a in (paper.get("authors") or []):
            name = (a.get("name") or "").strip()
            if name:
                authors.append(name)

        doi = external_ids.get("DOI") or None

        papers.append({
            "pmid":     str(pmid),
            "title":    title,
            "abstract": abstract,
            "journal":  journal,
            "year":     year,
            "authors":  authors,
            "source":   "semanticscholar",
            "doi":      doi,
        })

    return papers
