"""
Prospective validation monitor.

For every OpenCure prediction older than a minimum age, re-query PubMed
and ClinicalTrials.gov for papers/trials published AFTER the prediction
was made. If new evidence appears, the prediction is marked as a
"prospective validation hit" — a genuinely independent signal because
OpenCure made the prediction before the evidence existed.

This is the most credible efficacy claim a drug-repurposing platform can
make. By running this monthly, we build a rolling precision@K that isn't
retrospectively data-mined.

Usage:
    python3 scripts/prospective_monitor.py

Outputs:
    data/prospective/validation_log.jsonl   (append-only log)
    data/prospective/summary.json           (rolling precision)

Schedule: intended to run via agents/ cron monthly on the 1st.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


RESULTS_DIR = Path("experiments/results")
LOG_PATH = Path("data/prospective/validation_log.jsonl")
SUMMARY_PATH = Path("data/prospective/summary.json")

# Minimum age of a prediction before it counts as "prospective"
MIN_AGE_DAYS: int = 90

# How many top candidates to track per disease
TOP_K: int = 10


def _prediction_age_days(jf: Path) -> int:
    try:
        return int((time.time() - jf.stat().st_mtime) / 86400)
    except Exception:
        return 0


def _prediction_date(jf: Path) -> str:
    return datetime.fromtimestamp(jf.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")


def _load_existing_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    out = []
    with LOG_PATH.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _is_already_logged(log: list[dict], drug: str, disease: str) -> bool:
    return any(e.get("drug") == drug and e.get("disease") == disease for e in log)


def _query_pubmed(drug: str, disease: str, min_date: str) -> int:
    """Return PubMed count for (drug AND disease) published on/after min_date."""
    try:
        import urllib.parse
        import urllib.request
    except ImportError:
        return 0
    q = f"({urllib.parse.quote(drug)}) AND ({urllib.parse.quote(disease)}) AND ({min_date}[dp]:3000[dp])"
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={q}&retmode=json&retmax=1"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.load(r)
        return int(data.get("esearchresult", {}).get("count", 0))
    except Exception:
        return 0


def _query_trials(drug: str, disease: str, min_date: str) -> int:
    """Return ClinicalTrials.gov v2 count for trials posted after min_date."""
    try:
        import urllib.parse
        import urllib.request
    except ImportError:
        return 0
    q = urllib.parse.quote(f"{drug} {disease}")
    url = (
        "https://clinicaltrials.gov/api/v2/studies?"
        f"query.term={q}&filter.advanced=AREA%5BStudyFirstPostDate%5DRANGE%5B{min_date},MAX%5D"
        "&pageSize=1&countTotal=true"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        return int(data.get("totalCount", 0))
    except Exception:
        return 0


def check_disease_file(jf: Path, log: list[dict]) -> list[dict]:
    """Check each top-K candidate in a disease file for post-prediction evidence."""
    try:
        data = json.loads(jf.read_text())
    except Exception:
        return []
    cands = data.get("candidates", []) if isinstance(data, dict) else data
    if not cands:
        return []

    prediction_date = _prediction_date(jf)
    disease = jf.stem.replace("_", " ")
    age = _prediction_age_days(jf)
    if age < MIN_AGE_DAYS:
        return []

    new_entries = []
    for i, c in enumerate(cands[:TOP_K], start=1):
        drug = c.get("drug_name", "")
        if not drug:
            continue
        if _is_already_logged(log, drug, disease):
            continue
        # Rate-limit: small sleep
        time.sleep(0.35)
        pm_post = _query_pubmed(drug, disease, prediction_date)
        time.sleep(0.35)
        tr_post = _query_trials(drug, disease, prediction_date)
        validated = pm_post > 0 or tr_post > 0
        entry = {
            "drug": drug,
            "disease": disease,
            "rank": i,
            "prediction_date": prediction_date,
            "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "age_days": age,
            "new_pubmed_after_prediction": pm_post,
            "new_trials_after_prediction": tr_post,
            "validated": validated,
            "combined_score": c.get("combined_score", 0),
        }
        new_entries.append(entry)
    return new_entries


def update_summary(log: list[dict]) -> dict:
    total = len(log)
    validated = sum(1 for e in log if e.get("validated"))
    by_k = {5: [0, 0], 10: [0, 0]}  # [validated, total]
    for e in log:
        rank = e.get("rank", 99)
        for k in by_k:
            if rank <= k:
                by_k[k][1] += 1
                if e.get("validated"):
                    by_k[k][0] += 1
    return {
        "last_update": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_predictions_checked": total,
        "total_validated": validated,
        "precision_at_5":  round(by_k[5][0]  / by_k[5][1],  3) if by_k[5][1]  else None,
        "precision_at_10": round(by_k[10][0] / by_k[10][1], 3) if by_k[10][1] else None,
    }


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = _load_existing_log()
    print(f"Loaded {len(log)} prior log entries")

    new_entries: list[dict] = []
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if any(x in jf.name for x in ("opencure", "screening", "novel", "mechanism")):
            continue
        entries = check_disease_file(jf, log + new_entries)
        if entries:
            print(f"  {jf.stem}: +{len(entries)} checked  ({sum(1 for e in entries if e['validated'])} validated)")
            new_entries.extend(entries)

    if new_entries:
        with LOG_PATH.open("a") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")

    full_log = log + new_entries
    summary = update_summary(full_log)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {LOG_PATH} (+{len(new_entries)} entries)")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
