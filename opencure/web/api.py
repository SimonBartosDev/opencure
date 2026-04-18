"""
OpenCure programmatic API (v5).

FastAPI app exposing the core query endpoints:

    POST /score         — score compounds for a disease (runs full pipeline)
    GET  /predictions/<disease> — top-k predictions from latest snapshot
    GET  /clusters      — cross-disease mechanism clusters
    GET  /drug/<db_id>  — full drug profile (DDI + pharmacogenomics + dose)
    GET  /health        — readiness check
    /crowd/*            — crowd validation endpoints (mounted from crowd_endpoint)

Usage:
    uvicorn opencure.web.api:app --host 0.0.0.0 --port 8000

Behind FastAPI, uses the SAME search module as the CLI, so results are
bit-identical. Designed for pharma/biotech integration workflows:

    curl -X POST https://opencure.example/score \\
        -H 'Content-Type: application/json' \\
        -d '{"disease": "Glioblastoma", "top_k": 20}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = Exception
    BaseModel = object


RESULTS_DIR = Path("experiments/results")


if FastAPI is not None:
    app = FastAPI(
        title="OpenCure v5 API",
        version="5.0.0",
        description=(
            "Open-source drug repurposing platform. Returns calibrated "
            "predictions + mechanism paths + clinical guardrails. All "
            "Apache-2.0, no rate limits for non-commercial use."
        ),
    )

    # Mount crowd validation sub-router
    try:
        from opencure.web.crowd_endpoint import router as crowd_router
        app.include_router(crowd_router)
    except Exception:
        pass

    class ScoreRequest(BaseModel):
        disease: str
        top_k: int = 20
        include_mechanism_path: bool = True
        include_clinical_layer: bool = True

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "5.0.0"}

    @app.post("/score")
    def score(req: ScoreRequest):
        """Run full v5 scoring pipeline for a disease. Heavy — ~5-30 min."""
        try:
            from opencure.search import search
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"import failed: {e}")

        try:
            result = search(req.disease, top_k=req.top_k)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {
            "disease": req.disease,
            "candidates": result.get("candidates", []),
            "metadata": {
                "pipeline_version": "v5",
                "n_pillars": 12,
            },
        }

    @app.get("/predictions/{disease}")
    def predictions(disease: str, top_k: int = 10):
        """Return cached top-K predictions for a disease (no re-scoring)."""
        safe = disease.replace(" ", "_").replace("/", "_")
        jf = RESULTS_DIR / f"{safe}.json"
        if not jf.exists():
            raise HTTPException(status_code=404, detail=f"No cached predictions for '{disease}'")
        data = json.loads(jf.read_text())
        cands = data.get("candidates", [])[:top_k]
        return {"disease": disease, "candidates": cands, "cached": True}

    @app.get("/clusters")
    def clusters(min_strength: float = 0.5, limit: int = 50):
        """Return cross-disease mechanism clusters (polypharmacology)."""
        p = RESULTS_DIR / "mechanism_clusters.json"
        if not p.exists():
            return {"clusters": [], "note": "Run opencure.scoring.mechanism_cluster first"}
        data = json.loads(p.read_text())
        filt = [c for c in data if c.get("cluster_strength", 0) >= min_strength][:limit]
        return {"clusters": filt, "total": len(data), "filtered": len(filt)}

    @app.get("/drug/{drugbank_id}")
    def drug_profile(drugbank_id: str):
        """Full drug profile: DDI + pharmacogenomic + dose + ChEMBL phase."""
        from opencure.evidence.ddi_warnings import get_ddi_warnings
        from opencure.evidence.pharmacogenomics_v5 import get_pharmacogenomic_flags
        from opencure.evidence.dose_plausibility import get_dose_plausibility

        # Resolve name
        names_path = Path("data/drkg/drug_names_cache.tsv")
        name = drugbank_id
        if names_path.exists():
            for line in names_path.open():
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] == drugbank_id:
                    name = parts[1]
                    break

        return {
            "drugbank_id": drugbank_id,
            "drug_name": name,
            "ddi": get_ddi_warnings(drugbank_id, top_k=10),
            "pharmacogenomics": get_pharmacogenomic_flags(name),
            "dose": get_dose_plausibility(drugbank_id),
        }

else:
    app = None
