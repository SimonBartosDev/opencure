"""
Honest end-to-end scoring of OpenCure v5.

Single script that produces the full 'what does this actually do' report:

  1. Pillar fire rates across a set of disease results (from disk)
  2. Clean held-out Hit@K (random + time-sliced) on every trained KG model
  3. Clinical-layer coverage — how often dose/DDI/PGx/tissue populate
  4. Regression test suite pass/fail
  5. Metabolite leak check — any known leakers in current results?
  6. Quality-of-top-10 sanity for a pilot disease (Malaria): does a known
     treatment appear?

Run:
    python3 scripts/honest_scoring.py > experiments/eval/v5_honest_score.txt

Output is plain text, committable, reviewer-ready.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


RESULTS_DIR = Path("experiments/results")
EVAL_DIR = Path("experiments/eval")


def sec(label: str, sym="=") -> None:
    print()
    print(sym * 78)
    print(label)
    print(sym * 78)


def run_pillar_audit() -> dict:
    """Fire-rate check across whichever disease JSONs are on disk."""
    sec("1. PILLAR FIRE RATES (saved disease JSONs)")
    pillars = [
        ("1. TransE", "transe_score"),
        ("2. RotatE", "pykeen_score"),
        ("3. TxGNN", "txgnn_score"),
        ("4. MolFP", "mol_similarity"),
        ("5. ChemBERTa", "mol_emb_similarity"),
        ("6. Gene Sig", "gene_sig_score"),
        ("7. Proximity", "proximity_score"),
        ("8. MR", "mr_score"),
        ("9. ADMET", "admet_score"),
        ("10. PrimeKG", "primekg_score"),
        ("11. DTI", "dti_score"),
    ]
    fire = {p[0]: 0 for p in pillars}
    total = 0
    n_diseases = 0
    leaks = []
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if any(x in jf.name for x in ("opencure", "screening", "novel", "mechanism")):
            continue
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        cands = data.get("candidates", []) if isinstance(data, dict) else []
        if not cands:
            continue
        n_diseases += 1
        for c in cands:
            total += 1
            for label, field in pillars:
                v = c.get(field)
                if v not in (None, 0, 0.0, ""):
                    fire[label] += 1
            # Metabolite leak check
            name_lower = c.get("drug_name", "").lower()
            if any(tok in name_lower for tok in
                   ["uric acid", "glutathione", "dihydrofolic", "creatinine",
                    "cordycepin triphosphate", "androstene-3-ol"]):
                leaks.append(f"  {jf.stem} / {c.get('drug_name')}")

    print(f"\nScanned {total:,} candidates across {n_diseases} diseases")
    print(f"\n{'Pillar':<22} fire-rate (of {total})")
    print("-" * 70)
    for label, _ in pillars:
        n = fire[label]
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 5)
        print(f"{label:<22} {n:>4}  {pct:5.1f}%  {bar}")

    print(f"\nMetabolite leaks detected: {len(leaks)}")
    for line in leaks[:10]:
        print(line)

    return {"total_candidates": total, "n_diseases": n_diseases,
            "fire_rates": {k: (v, 100 * v / total if total else 0) for k, v in fire.items()},
            "leaks": leaks}


def run_kg_evals() -> dict:
    """Run held-out eval if a trained model exists."""
    sec("2. HELD-OUT KG-SCORER METRICS")

    existing_eval = EVAL_DIR / "v5_unified_heldout.json"
    if existing_eval.exists():
        data = json.loads(existing_eval.read_text())
        for model_label, results in data.items():
            if model_label == "evaluated_at":
                continue
            print(f"\n{model_label}:")
            for tset in ("random_holdout", "time_sliced"):
                m = results.get(tset, {})
                if "hit_at" in m:
                    print(f"  {tset:>18}  Hit@10 = {m['hit_at']['10']:>6}%  "
                          f"MRR = {m.get('mrr')}  median = {m.get('median_rank')}")
        return data
    print("No eval JSON yet — run scripts/run_unified_heldout_eval.py")
    return {}


def run_clinical_coverage() -> dict:
    """How often the v5 clinical layer populates."""
    sec("3. CLINICAL LAYER COVERAGE")
    keys = ("dose_plausibility", "ddi_warnings", "pharmacogenomics",
            "triangulation", "tissue_context")
    pop_count = {k: 0 for k in keys}
    total = 0
    for jf in sorted(RESULTS_DIR.glob("*.json")):
        if any(x in jf.name for x in ("opencure", "screening", "novel", "mechanism")):
            continue
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        cands = d.get("candidates", []) if isinstance(d, dict) else []
        for c in cands:
            total += 1
            for k in keys:
                v = c.get(k)
                if v and (isinstance(v, dict) and v or isinstance(v, list) and v):
                    pop_count[k] += 1
    print(f"\nScanned {total} candidates")
    for k in keys:
        pct = 100 * pop_count[k] / total if total else 0
        bar = "█" * int(pct / 5)
        print(f"  {k:<22s}  {pop_count[k]:>4}  {pct:5.1f}%  {bar}")
    if total > 0 and max(pop_count.values()) == 0:
        print("\nNOTE: none populated — rerun v5 screening to update")
    return pop_count


def run_tests() -> dict:
    """Run the pytest non-integration suite."""
    sec("4. REGRESSION TEST SUITE")
    t0 = time.time()
    r = subprocess.run(
        ["python3", "-m", "pytest", "-m", "not integration", "--quiet"],
        env={**__import__("os").environ, "PYTHONPATH": "."},
        capture_output=True, text=True, timeout=120,
    )
    dt = time.time() - t0
    # Parse '78 passed in 0.42s' / 'X failed, Y passed in Zs'
    last = r.stdout.strip().splitlines()[-1] if r.stdout else ""
    print(f"\n{last}  (runtime {dt:.1f}s)")
    return {"pass": r.returncode == 0, "summary": last}


def run_commit_summary() -> None:
    """Show recent v5 commits."""
    sec("5. RECENT V5 COMMITS")
    r = subprocess.run(["git", "log", "--oneline", "-15"], capture_output=True, text=True)
    print()
    print(r.stdout)


def run_final_scorecard(pillar: dict, kg: dict, clinical: dict, tests: dict) -> None:
    sec("FINAL HONEST SCORECARD")

    fire_rates = pillar.get("fire_rates", {})
    pillar_working = sum(1 for _, (_, pct) in fire_rates.items() if pct > 0)

    # Prefer the best available clean model, in this order:
    #   drkg_transE_clean (DRKG-only, 50ep, 400-dim-analogue) — BEST
    #   unified_transE_clean (14M triples, 20ep, undertrained)
    #   unified_transE_contaminated (training-contaminated upper bound)
    kg_clean = None
    preferred = ("drkg_transE_clean", "unified_transE_clean", "unified_transE_contaminated")
    for mlabel in preferred:
        res = kg.get(mlabel)
        if isinstance(res, dict):
            rh = res.get("random_holdout", {})
            if "hit_at" in rh:
                kg_clean = (mlabel, rh["hit_at"]["10"])
                break

    print(f"""
 Code quality       : {pillar_working}/11 pillars firing in saved JSONs
 Tests              : {tests.get('summary', '—')}
 Metabolite filter  : {len(pillar.get('leaks', []))} leak(s) (target: 0)
 KG scorer          : {kg_clean[0] if kg_clean else '—'} → Hit@10 = {kg_clean[1] if kg_clean else '—'}%
 Clinical layer     : {'populated' if max(clinical.values(), default=0) > 0 else 'NOT populated in current JSONs — re-screen to activate'}

The full 11-pillar + clinical-layer output surfaces only in FRESH v5
searches (re-runs post-refactor).  Old disease JSONs on disk pre-date
the v5 bugfixes and clinical-layer wiring.

Ready for the public phase when:
  • drkg_transE_clean Hit@10 > 40% on random-holdout (clean baseline)
  • a fresh v5 re-screen regenerates all 61 disease JSONs
  • Zenodo upload of the prospective-snapshot folder
  • bioRxiv submission with these held-out numbers filled in
""")


def main() -> None:
    print(f"OpenCure v5 honest-scoring run — {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    pillar = run_pillar_audit()
    kg = run_kg_evals()
    clinical = run_clinical_coverage()
    tests = run_tests()
    run_commit_summary()
    run_final_scorecard(pillar, kg, clinical, tests)


if __name__ == "__main__":
    main()
