"""
Crowd-sourced validation endpoint (v5).

FastAPI router exposing a public submission form where researchers,
clinicians, or anyone running a repurposing experiment can report:

  "I tested DRUG X on DISEASE Y in [model]. Result: [IC50/EC50/no effect]"

Submissions append to data/crowd_validation.jsonl (append-only) and are
tagged `pending_moderation`. A human moderator flips `approved` before
the record surfaces publicly. The dashboard shows counts only (approved
+ pending) until moderation happens.

Mounted into the existing FastAPI app in opencure/web/; the dashboard
HTML adds a simple form linking to the endpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field, EmailStr
except ImportError:
    # Minimal stubs — module still importable on systems without fastapi
    class APIRouter:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def get(self, *a, **k): return lambda f: f
        def post(self, *a, **k): return lambda f: f
    class HTTPException(Exception):  # type: ignore
        def __init__(self, status_code, detail): self.status_code = status_code
    class BaseModel: pass  # type: ignore
    def Field(*a, **k): return None  # type: ignore
    EmailStr = str  # type: ignore


router = APIRouter(prefix="/crowd", tags=["crowd_validation"])

LOG_PATH = Path("data/crowd_validation.jsonl")


class ValidationSubmission(BaseModel):  # type: ignore
    drug_name: str = Field(..., min_length=1, max_length=200)
    drug_id: Optional[str] = None  # DrugBank ID preferred
    disease_name: str = Field(..., min_length=1, max_length=200)
    disease_id: Optional[str] = None  # MeSH preferred
    result: str = Field(..., description="One of: efficacy, no_effect, mixed, safety_signal")
    experimental_model: str = Field(..., description="e.g. 'Vero E6 + SARS-CoV-2', 'U87 xenograft', 'patient-derived organoid'")
    readout: str = Field(..., description="e.g. 'IC50=3.2 µM', 'no effect at 10 µM', 'tumor volume −42%'")
    n_replicates: Optional[int] = None
    reference: Optional[str] = Field(None, description="DOI / PMID / preprint URL / lab notebook ref")
    researcher_name: Optional[str] = None
    researcher_email: Optional[EmailStr] = None
    institution: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=2000)


def append_submission(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")


@router.post("/submit")
def submit(entry: ValidationSubmission):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "submission_id": str(uuid.uuid4())[:12],
        "received_at": now,
        "approved": False,
        **entry.dict(),
    }
    append_submission(record)
    return {
        "status": "received",
        "submission_id": record["submission_id"],
        "note": (
            "Thank you. Your submission is queued for moderation (typically 1-3 "
            "days) before it surfaces publicly. Contribution will be cited in "
            "the next prospective-validation summary with the reference you provided."
        ),
    }


@router.get("/stats")
def stats():
    """Public stats (counts only)."""
    if not LOG_PATH.exists():
        return {"total": 0, "approved": 0, "pending": 0, "by_result": {}}

    total = 0
    approved = 0
    by_result: dict[str, int] = {}
    with LOG_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            total += 1
            if r.get("approved"):
                approved += 1
                key = r.get("result", "")
                by_result[key] = by_result.get(key, 0) + 1
    return {
        "total": total,
        "approved": approved,
        "pending": total - approved,
        "by_result": by_result,
    }


@router.get("/recent")
def recent(limit: int = 20):
    """Public feed — APPROVED submissions only, sorted newest first."""
    if not LOG_PATH.exists():
        return []
    records = []
    with LOG_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("approved"):
                continue
            records.append({
                "drug_name": r.get("drug_name"),
                "disease_name": r.get("disease_name"),
                "result": r.get("result"),
                "experimental_model": r.get("experimental_model"),
                "readout": r.get("readout"),
                "reference": r.get("reference"),
                "institution": r.get("institution"),
                "received_at": r.get("received_at"),
            })
    records.sort(key=lambda x: x.get("received_at", ""), reverse=True)
    return records[:limit]
