"""
PAINS / structural-alert annotation.

Pan-Assay INterference compoundS (PAINS) are molecules known to
frequently produce false positives in bioactivity assays. Classic v1
filters (Baell & Holloway, 2010) cover ~480 substructures; we add the
NIH Nuisance / Brenk / ZINC / Dundee alert sets where useful.

This is an **annotation pillar, not a hard filter** — FDA-approved
drugs do sometimes match PAINS patterns (e.g., curcumin). We flag the
hit and surface it in output so reviewers can weight accordingly, but
we don't drop the compound. A *deliberate* design choice.

Output per SMILES: dict with keys
  - ``has_pains``     : bool
  - ``pains_families``: list of short matched-family names
  - ``n_alerts``      : int  — total matched patterns across all catalogs
  - ``alert_names``   : list of the first few pattern names (for
                        debugging; cap at 5)

Usage
-----
    from opencure.filters.pains import annotate_structural_alerts
    alerts = annotate_structural_alerts("O=C(O)c1ccccc1O")  # salicylic acid
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _catalog():
    """Build a composite RDKit FilterCatalog covering PAINS + Brenk + NIH."""
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    return FilterCatalog(params)


def annotate_structural_alerts(smiles: str) -> dict:
    """Return structural-alert annotation for a SMILES string.

    Safe to call repeatedly — RDKit mol construction and the catalog
    are both cheap; the catalog itself is lru-cached across calls.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {"has_pains": False, "pains_families": [], "n_alerts": 0,
                "alert_names": [], "error": "rdkit_not_installed"}
    if not smiles:
        return {"has_pains": False, "pains_families": [], "n_alerts": 0,
                "alert_names": []}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"has_pains": False, "pains_families": [], "n_alerts": 0,
                "alert_names": [], "error": "invalid_smiles"}

    catalog = _catalog()
    entries = catalog.GetMatches(mol)
    names = [e.GetDescription() for e in entries]
    # Short family labels: first token, uppercased (PAINS_A, BRENK, NIH, …)
    families = sorted({n.split("_")[0].upper() if "_" in n else n.split()[0].upper()
                       for n in names})
    return {
        "has_pains": any(f.startswith("PAINS") for f in families),
        "pains_families": families,
        "n_alerts": len(names),
        "alert_names": names[:5],
    }
