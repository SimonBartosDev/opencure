"""OpenCure web application."""

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from opencure.search import search, search_simple
from opencure.data.drkg import DISEASE_NAME_MAP
from opencure.evidence.report import generate_evidence_report

WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = FastAPI(title="OpenCure", description="AI-powered drug repurposing")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Pre-warm data on startup
_warmed = False


def _warm():
    global _warmed
    if not _warmed:
        search_simple("Alzheimer's disease", top_k=1)
        _warmed = True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    _warm()
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/search")
async def api_search(disease: str, top_k: int = 25):
    """Search for drug repurposing candidates."""
    _warm()
    start = time.time()
    results = search(
        disease,
        top_k=top_k,
        use_molecular_similarity=True,
        use_evidence=True,
    )
    elapsed = round(time.time() - start, 2)

    # Cache for report lookups
    _last_search["disease"] = disease
    _last_search["results"] = results

    return JSONResponse({
        "disease": disease,
        "results": results,
        "count": len(results),
        "elapsed_seconds": elapsed,
    })


# Cache last search results for evidence report lookups
_last_search = {"disease": "", "results": []}


@app.get("/api/report")
async def api_report(disease: str, drug_id: str):
    """Generate a full evidence report for a drug-disease pair (on-demand)."""
    _warm()

    # Find the result from the last search, or do a quick search
    result = None
    if _last_search["disease"] == disease:
        for r in _last_search["results"]:
            if r["drug_id"] == drug_id:
                result = r
                break

    if result is None:
        # Quick search to find this drug
        results = search(disease, top_k=50, use_molecular_similarity=True, use_evidence=True)
        for r in results:
            if r["drug_id"] == drug_id:
                result = r
                break

    if result is None:
        return JSONResponse({"error": f"Drug {drug_id} not found for {disease}"}, status_code=404)

    start = time.time()
    report = generate_evidence_report(result, disease)
    elapsed = round(time.time() - start, 2)

    return JSONResponse({
        "report": report.to_dict(),
        "elapsed_seconds": elapsed,
    })


@app.get("/api/explain")
async def api_explain(disease: str, drug_id: str):
    """Generate a mechanistic explanation for a drug-disease prediction (on-demand)."""
    _warm()
    from opencure.data.drkg import load_triplets, load_embeddings, find_disease_entities
    from opencure.evidence.llm_explainer import explain_prediction
    from opencure.search import _get_data

    data = _get_data()

    # Find the drug and disease entities
    drug_entity = f"Compound::{drug_id}"
    disease_matches = find_disease_entities(data["entity_to_id"], disease)
    if not disease_matches:
        return JSONResponse({"error": f"Disease '{disease}' not found"}, status_code=404)

    disease_entity = disease_matches[0][0]

    # Get MR score if available
    mr_score = 0
    mr_targets = 0
    if _last_search["disease"] == disease:
        for r in _last_search["results"]:
            if r["drug_id"] == drug_id:
                mr_score = r.get("mr_score", 0) or 0
                mr_targets = r.get("mr_genetic_targets", 0) or 0
                break

    start = time.time()
    explanation = explain_prediction(
        drug_entity, disease_entity, disease,
        data["triplets"], data["drug_names"],
        mr_score=mr_score, mr_targets=mr_targets,
    )
    elapsed = round(time.time() - start, 2)

    return JSONResponse({
        "explanation": explanation,
        "elapsed_seconds": elapsed,
    })


@app.get("/api/combinations")
async def api_combinations(disease: str, drug_id: str):
    """Find synergistic drug combination partners."""
    _warm()
    from opencure.scoring.drug_combinations import find_synergistic_partners
    from opencure.search import _get_data

    data = _get_data()
    drug_entity = f"Compound::{drug_id}"

    # Get other candidates from last search
    candidates = _last_search.get("results", [])
    if not candidates or _last_search.get("disease") != disease:
        candidates = search(disease, top_k=50, use_molecular_similarity=True, use_evidence=False)

    start = time.time()
    partners = find_synergistic_partners(
        drug_entity, disease, candidates,
        data["triplets"], data["drug_names"],
    )
    elapsed = round(time.time() - start, 2)

    return JSONResponse({
        "drug_id": drug_id,
        "disease": disease,
        "partners": partners,
        "elapsed_seconds": elapsed,
    })


@app.get("/api/diseases")
async def api_diseases():
    """Get list of supported disease names for autocomplete."""
    diseases = sorted(DISEASE_NAME_MAP.keys())
    return JSONResponse({"diseases": diseases})
