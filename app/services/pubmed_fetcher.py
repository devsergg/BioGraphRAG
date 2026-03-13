"""
PubMed fetcher — two-phase esearch → efetch.

Phase 1: esearch returns a list of PMIDs for a search term (max 25).
Phase 2: efetch retrieves all PMIDs in a single XML request.

Returns canonical paper dicts; never raises — returns [] on any error.
"""
import time
import xml.etree.ElementTree as ET

import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
MAX_RESULTS = 25
SLEEP_BETWEEN = 0.34  # polite 3 req/s


def _esearch(term: str) -> list[str]:
    """Return up to MAX_RESULTS PMIDs for the given search term."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": MAX_RESULTS,
        "retmode": "json",
        "sort": "relevance",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except requests.RequestException:
        return []


def _efetch(pmids: list[str]) -> list[dict]:
    """Fetch full records for a list of PMIDs in one XML request."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    papers = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text.strip() if pmid_elem is not None else None
        if not pmid:
            continue

        # Title — inline tags like <i> are common
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

        # Abstract — may be multiple <AbstractText> elements (structured abstracts)
        abstract_parts = []
        for at in article.findall(".//AbstractText"):
            part = "".join(at.itertext()).strip()
            if part:
                abstract_parts.append(part)
        abstract = " ".join(abstract_parts)

        # Skip papers with no useful abstract
        if not abstract:
            continue

        # Journal
        journal_elem = article.find(".//Journal/Title")
        if journal_elem is None:
            journal_elem = article.find(".//MedlineTA")
        journal = journal_elem.text.strip() if journal_elem is not None else ""

        # Year: prefer <Year>, fallback to first 4 digits of <MedlineDate>
        year = None
        year_elem = article.find(".//PubDate/Year")
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text.strip())
            except ValueError:
                pass
        if year is None:
            medline_elem = article.find(".//PubDate/MedlineDate")
            if medline_elem is not None and medline_elem.text:
                digits = "".join(filter(str.isdigit, medline_elem.text))
                if len(digits) >= 4:
                    try:
                        year = int(digits[:4])
                    except ValueError:
                        pass

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "").strip()
            fore = author.findtext("ForeName", "").strip()
            if last:
                authors.append(f"{last} {fore}".strip())

        # DOI
        doi = None
        for aid in article.findall(".//ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        papers.append({
            "pmid":     pmid,
            "title":    title,
            "abstract": abstract,
            "journal":  journal,
            "year":     year,
            "authors":  authors,
            "source":   "pubmed",
            "doi":      doi,
        })

    return papers


def fetch_pubmed_papers(term: str) -> list[dict]:
    """
    Search PubMed for `term` and return up to 25 canonical paper dicts.
    Sleeps 0.34s after each HTTP call to stay within the 3 req/s polite limit.
    """
    pmids = _esearch(term)
    time.sleep(SLEEP_BETWEEN)
    if not pmids:
        return []
    papers = _efetch(pmids)
    time.sleep(SLEEP_BETWEEN)
    return papers
