import time
import requests
from typing import Optional

SEARCH_TERMS = [
    "neuropathic pain",
    "chronic pain",
    "central sensitization",
    "spinal cord stimulation",
    "dorsal root ganglion",
    "sodium channel pain",
    "Nav1.7",
    "Nav1.8",
    "CGRP pain",
    "substance P pain",
    "ketamine pain",
    "low dose naltrexone",
    "gabapentin neuropathy",
    "pregabalin neuropathy",
    "glial cell pain",
    "neuroinflammation pain",
    "TRPV1 pain",
    "endocannabinoid pain",
]

CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"


def fetch_trials(search_term: str, max_results: int = 20) -> list[dict]:
    """Fetch trials from ClinicalTrials.gov for a single search term."""
    params = {
        "query.term": search_term,
        "pageSize": max_results,
        "format": "json",
    }
    try:
        resp = requests.get(CLINICALTRIALS_BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        records = []
        for study in studies:
            record = extract_trial_fields(study, search_term)
            if record is not None:
                records.append(record)
        return records
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch trials for '{search_term}': {e}")
        return []


def extract_trial_fields(study: dict, search_compound: str) -> Optional[dict]:
    """Safely extract fields using .get() chains. Returns None if essential fields are missing."""
    protocol = study.get("protocolSection", {})

    id_module = protocol.get("identificationModule", {})
    nct_id = id_module.get("nctId")
    title = id_module.get("briefTitle")

    # Skip records without essential identifiers
    if not nct_id or not title:
        return None

    desc_module = protocol.get("descriptionModule", {})
    description = desc_module.get("briefSummary", "")

    conditions_module = protocol.get("conditionsModule", {})
    conditions = conditions_module.get("conditions", [])

    arms_module = protocol.get("armsInterventionsModule", {})
    interventions_raw = arms_module.get("interventions", [])
    interventions = [i.get("name", "") for i in interventions_raw if i.get("name")]

    sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
    lead_sponsor = sponsor_module.get("leadSponsor", {})
    sponsor = lead_sponsor.get("name", "")

    design_module = protocol.get("designModule", {})
    phase = design_module.get("phases", [])

    status_module = protocol.get("statusModule", {})
    status = status_module.get("overallStatus", "")

    return {
        "nct_id": nct_id,
        "title": title,
        "description": description,
        "conditions": conditions,
        "interventions": interventions,
        "sponsor": sponsor,
        "phase": phase,
        "status": status,
        "search_compound": search_compound,
    }


def fetch_pubchem_data(compound_name: str) -> dict:
    """Fetch basic compound info from PubChem. Returns {} on any failure."""
    try:
        url = f"{PUBCHEM_BASE}/{requests.utils.quote(compound_name)}/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        data = resp.json()
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return {}
        compound = compounds[0]
        props = {p["urn"]["label"]: p.get("value", {}) for p in compound.get("props", [])}
        return {
            "cid": compound.get("id", {}).get("id", {}).get("cid"),
            "molecular_formula": props.get("Molecular Formula", {}).get("sval", ""),
            "iupac_name": props.get("IUPAC Name", {}).get("sval", ""),
        }
    except Exception:
        return {}


def fetch_all_trials() -> list[dict]:
    """Iterate all SEARCH_TERMS, deduplicate by nct_id, return unique trial list."""
    seen: dict[str, dict] = {}
    total_fetched = 0

    for i, term in enumerate(SEARCH_TERMS, 1):
        print(f"  [{i}/{len(SEARCH_TERMS)}] Fetching: '{term}'")
        trials = fetch_trials(term)
        new_count = 0
        for trial in trials:
            nct_id = trial["nct_id"]
            if nct_id not in seen:
                seen[nct_id] = trial
                new_count += 1
        total_fetched += len(trials)
        print(f"    → {len(trials)} fetched, {new_count} new unique")
        time.sleep(0.5)  # Polite rate limiting

    unique_trials = list(seen.values())
    print(f"\nTotal fetched: {total_fetched} | Unique trials: {len(unique_trials)}")
    return unique_trials
