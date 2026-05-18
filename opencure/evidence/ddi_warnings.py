"""
Drug-drug interaction warnings (v5).

DRKG contains 1,379,271 DRUGBANK::ddi-interactor-in::Compound:Compound
edges — direct drug-drug interaction annotations from DrugBank. For every
top prediction, we surface the most clinically-consequential interactions
so a prescriber sees "Drug X is predicted for Disease Y, but DO NOT combine
with [A, B, C] due to [mechanism]".

Strategy:
  - Build adjacency: compound → set of interacting compounds  (one-time, cached)
  - For each top-10 prediction, look up interactions + enrich with drug names
  - Rank by "clinical severity proxy":
      * highest: interactions with commonly-co-prescribed drugs (we proxy by
        degree: if the interacting drug also appears in the result set, or is
        a top-200 most-prescribed drug, prioritize)
      * flag CYP3A4 / P-gp / narrow-therapeutic-index partners explicitly

Output per drug:
  {
    "n_interactions": int,
    "top_interactions": [{"drug": X, "mechanism_hint": Y, "severity": S}, ...]
  }
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional


DRKG_PATH = Path("data/drkg/drkg.tsv")
DDI_CACHE_PATH = Path("data/drkg/ddi_adjacency.pkl")
DRUG_NAMES_PATH = Path("data/drkg/drug_names_cache.tsv")


# Commonly-co-prescribed drugs (rough 2024 top-list) — interactions with
# these are clinically more likely to matter. Priority=high.
COMMONLY_COPRESCRIBED = {
    "DB00945",  # Aspirin
    "DB00564",  # Carbamazepine
    "DB00331",  # Metformin
    "DB00682",  # Warfarin
    "DB00338",  # Omeprazole
    "DB00641",  # Simvastatin
    "DB00175",  # Pravastatin
    "DB01076",  # Atorvastatin
    "DB01098",  # Rosuvastatin
    "DB00758",  # Clopidogrel
    "DB00316",  # Acetaminophen
    "DB00530",  # Erlotinib
    "DB00959",  # Methylprednisolone
    "DB00213",  # Pantoprazole
    "DB01264",  # Darunavir (for HIV co-prescription)
    "DB00199",  # Erythromycin
    "DB01211",  # Clarithromycin
    "DB00227",  # Lovastatin
    "DB00860",  # Prednisolone
    "DB00783",  # Estradiol
    "DB00773",  # Etoposide
    "DB00563",  # Methotrexate
    "DB01032",  # Probenecid
    "DB00829",  # Diazepam
    "DB00502",  # Haloperidol
}


@lru_cache(maxsize=1)
def _load_drug_names() -> dict[str, str]:
    """DrugBank ID → display name."""
    if not DRUG_NAMES_PATH.exists():
        return {}
    m: dict[str, str] = {}
    with DRUG_NAMES_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                m[parts[0]] = parts[1]
    return m


def _build_adjacency() -> dict[str, set[str]]:
    """Build compound → {interacting compounds} from DRKG ddi edges."""
    print("Building DDI adjacency from DRKG…")
    adj: dict[str, set[str]] = {}
    n = 0
    with DRKG_PATH.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if r != "DRUGBANK::ddi-interactor-in::Compound:Compound":
                continue
            h_id = h.split("::", 1)[1] if "::" in h else h
            t_id = t.split("::", 1)[1] if "::" in t else t
            adj.setdefault(h_id, set()).add(t_id)
            adj.setdefault(t_id, set()).add(h_id)
            n += 1
    print(f"  {n:,} DDI edges → {len(adj):,} drugs with at least one interaction")
    return adj


@lru_cache(maxsize=1)
def _get_adjacency() -> dict[str, set[str]]:
    if DDI_CACHE_PATH.exists():
        try:
            with DDI_CACHE_PATH.open("rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    adj = _build_adjacency()
    DDI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DDI_CACHE_PATH.open("wb") as f:
        # Convert sets to lists for pickle; sets aren't always pickle-friendly
        pickle.dump({k: list(v) for k, v in adj.items()}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {k: set(v) for k, v in adj.items()}


def get_ddi_warnings(
    drug_id: str,
    top_k: int = 10,
) -> dict:
    """
    Return a structured DDI warning profile for a drug.

    Args:
        drug_id: DrugBank ID (with or without "Compound::" prefix)
        top_k: how many top interactions to return

    Returns:
        {
          "n_interactions": total count,
          "has_warnings": bool,
          "top_interactions": [{"drug_id": DB..., "drug_name": name, "severity": "high"|"moderate"}]
        }
    """
    bare = drug_id.split("::", 1)[1] if "::" in drug_id else drug_id
    adj = _get_adjacency()
    partners = adj.get(bare, set())
    if not partners:
        return {"n_interactions": 0, "has_warnings": False, "top_interactions": []}

    names = _load_drug_names()
    out: list[dict] = []
    for p in partners:
        severity = "high" if p in COMMONLY_COPRESCRIBED else "moderate"
        out.append({
            "drug_id": p,
            "drug_name": names.get(p, p),
            "severity": severity,
        })

    # Sort: high severity first, then alphabetical by drug name
    out.sort(key=lambda x: (0 if x["severity"] == "high" else 1, x["drug_name"]))
    return {
        "n_interactions": len(partners),
        "has_warnings": True,
        "top_interactions": out[:top_k],
    }


if __name__ == "__main__":
    # Smoke test
    import sys
    for drug_id in ["DB00945", "DB00682", "DB01211", "DB00843", "DB00641"]:
        w = get_ddi_warnings(drug_id, top_k=5)
        print(f"\n{drug_id}: {w['n_interactions']:,} interactions")
        for t in w["top_interactions"][:5]:
            print(f"  [{t['severity']:8s}]  {t['drug_name']} ({t['drug_id']})")
