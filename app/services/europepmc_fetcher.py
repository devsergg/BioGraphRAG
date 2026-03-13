"""
Europe PMC fetcher.

Uses the Europe PMC REST API to retrieve up to 25 papers per search term.
`resultType=core` is required to include the abstractText field.

Returns canonical paper dicts; never raises — returns [] on any error.
"""
import requests

EUROPEPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
MAX_RESULTS   = 25


def fetch_europepmc_papers(term: str) -> list[dict]:
    """
    Search Europe PMC for `term` and return up to MAX_RESULTS canonical paper dicts.
    """
    params = {
        "query":      term,
        "resultType": "core",       # required for abstractText
        "pageSize":   MAX_RESULTS,
        "format":     "json",
        "sort":       "RELEVANCE",
    }
    try:
        resp = requests.get(EUROPEPMC_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return []

    results = data.get("resultList", {}).get("result", [])
    papers = []

    for result in results:
        # Filter to journal articles with a PubMed ID for stable dedup
        pmid = result.get("pmid")
        if not pmid:
            continue

        abstract = result.get("abstractText", "").strip()
        if not abstract:
            continue

        title = result.get("title", "").strip().rstrip(".")
        journal = result.get("journalTitle", "").strip()

        # pubYear is a string in the API response
        raw_year = result.get("pubYear", None)
        try:
            year = int(raw_year) if raw_year else None
        except (ValueError, TypeError):
            year = None

        # Authors list
        authors = []
        author_list = result.get("authorList", {}).get("author", [])
        for a in author_list:
            full = a.get("fullName", "").strip()
            if full:
                authors.append(full)

        # DOI
        doi = result.get("doi", None) or None

        papers.append({
            "pmid":     str(pmid),
            "title":    title,
            "abstract": abstract,
            "journal":  journal,
            "year":     year,
            "authors":  authors,
            "source":   "europepmc",   # canonical value — not epmc_source
            "doi":      doi,
        })

    return papers
