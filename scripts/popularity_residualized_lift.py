"""
Popularity-residualized conditional-lift test for the genetics-anchored ranker.

WHY THIS EXISTS
---------------
The Phase-3 scorecard (genetics_anchored_benchmark_v3.py) reports the genetics-
anchored ranker beating a popularity baseline ~5x on the genetics-covered subset
(Hit@10 20.8% vs 3.8%). Two problems with that headline, both surfaced by an
adversarial review:

  1. SELF-INFLATED BASELINE. The popularity baseline degree is built from
     `load_triplets()` -> data/drkg/drkg.tsv, the FULL graph, which still
     contains the 999 held-out `treats` edges. Each held-out true (drug,disease)
     edge adds +1 to that drug's degree, so the comparator is inflated by the
     very answers being predicted. The honest baseline must use
     data/drkg/drkg_stripped.tsv.

  2. NO CONDITIONAL-LIFT NUMBER. The scorecard reports genetics and popularity
     as two side-by-side rankings, never the within-pair INCREMENTAL lift of
     genetics OVER degree. And its "genetic-signal subset" is defined by
     pool_score(true_drug) > 0 — conditioning the eval subset on the feature
     value at the label (selection bias).

This script answers the honest question: AFTER controlling for popularity, does
the genetics score add real rank lift? It does so three leak-free ways:

  (A) CORRECTED RANKING. Re-rank with the popularity baseline degree taken from
      the STRIPPED graph. Report genetics vs corrected-popularity overall and on
      the genetics-covered subset, and quantify how much the inflation mattered.
  (B) DEGREE-STRATIFIED Hit@10. Bin held-out positives by their true drug's
      (stripped) degree quartile; within each quartile compare genetics Hit@10
      to popularity Hit@10. A signal that wins in the LOW-degree quartiles is
      genuinely independent of popularity (residualized).
  (C) PAIRED HEAD-TO-HEAD + LOGISTIC LIFT. On the genetics-covered positives,
      the fraction where genetics ranks the true drug better than degree does
      (bootstrap CI over diseases); plus, if statsmodels is available, the
      partial coefficient of genetics_score in is_true ~ log_degree +
      genetics_score over the pooled candidate sets.

Leak control is INHERITED unchanged from genetics_anchored_benchmark_v3.py
(disease side = genetics datasources only; drug side = ChEMBL mechanism target;
no treats/indication edge is read). The ONLY thing this script changes is the
popularity comparator (now leak-clean) and the metric (now conditional).

OUTPUT: experiments/eval/conditional_lift_report.json
        experiments/eval/conditional_lift_pairs.jsonl  (per-pair tuples)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import genetics_anchored_benchmark_v3 as gab  # reuse the leak-free machinery

STRIPPED = Path("data/drkg/drkg_stripped.tsv")
FULL = Path("data/drkg/drkg.tsv")
OUT = Path("experiments/eval/conditional_lift_report.json")
PAIRS_OUT = Path("experiments/eval/conditional_lift_pairs.jsonl")
RNG = np.random.default_rng(0)


def degree_from(path: Path) -> Counter:
    """Node degree from a DRKG triplet TSV (head<TAB>relation<TAB>tail)."""
    deg: Counter = Counter()
    with path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            deg[parts[0]] += 1
            deg[parts[2]] += 1
    return deg


def hit_at(ranks: np.ndarray, k: int) -> float:
    return round(100 * float(np.mean(ranks <= k)), 1) if len(ranks) else 0.0


def summarise(ranks: list[float]) -> dict:
    if not ranks:
        return {"n": 0}
    a = np.array(ranks, dtype=float)
    return {
        "n": len(a),
        "hit_at_10": hit_at(a, 10),
        "hit_at_30": hit_at(a, 30),
        "hit_at_100": hit_at(a, 100),
        "median_rank": int(np.median(a)),
        "mrr": round(float(np.mean(1.0 / a)), 4),
    }


def compute_logistic_lift(reg_rows: list) -> dict:
    """Partial coefficient of genetics_score controlling for log_degree.

    is_true ~ log_degree + genetics_score, pooled over the covered diseases'
    candidate sets. Features standardized so coefficients are comparable.
    statsmodels gives an analytic CI; if absent we fall back to sklearn with a
    disease-cluster bootstrap CI (resample diseases, refit). A positive
    genetics coefficient whose CI excludes 0 = genetics carries signal beyond
    popularity. Disease clustering is honored only in the bootstrap CI, so this
    is read alongside the degree-stratified table, not instead of it.
    """
    if not reg_rows:
        return {"available": False, "reason": "no covered diseases"}
    diseases = [r[0] for r in reg_rows]
    X = np.array([[r[1], r[2]] for r in reg_rows], dtype=float)  # log_deg, gscore
    y = np.array([r[3] for r in reg_rows], dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    try:
        import statsmodels.api as sm  # type: ignore
        model = sm.Logit(y, sm.add_constant(Xs)).fit(disp=0, maxiter=300)
        conf = model.conf_int(alpha=0.10)
        return {
            "available": True, "engine": "statsmodels",
            "n_rows": len(reg_rows), "n_positives": int(y.sum()),
            "standardized_coef_log_degree": round(float(model.params[1]), 4),
            "standardized_coef_genetics_score": round(float(model.params[2]), 4),
            "genetics_score_p_value": round(float(model.pvalues[2]), 6),
            "genetics_score_90ci": [round(float(conf[2][0]), 4),
                                    round(float(conf[2][1]), 4)],
        }
    except ModuleNotFoundError:
        pass

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}

    def fit_gcoef(Xm, ym):
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        clf.fit(Xm, ym)
        return float(clf.coef_[0][0]), float(clf.coef_[0][1])  # logdeg, gscore

    ld_coef, g_coef = fit_gcoef(Xs, y)
    # disease-cluster bootstrap for the genetics-score coefficient CI
    by_dis_idx: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(diseases):
        by_dis_idx[d].append(i)
    keys = list(by_dis_idx)
    boots = []
    for _ in range(200):
        samp = RNG.choice(len(keys), len(keys), replace=True)
        idx = [i for di in samp for i in by_dis_idx[keys[di]]]
        if len(set(y[idx])) < 2:
            continue
        try:
            _, gc = fit_gcoef(Xs[idx], y[idx])
            boots.append(gc)
        except Exception:  # noqa: BLE001
            continue
    ci = ([round(float(np.quantile(boots, 0.05)), 4),
           round(float(np.quantile(boots, 0.95)), 4)] if boots else None)
    return {
        "available": True, "engine": "sklearn (unpenalized) + cluster bootstrap",
        "n_rows": len(reg_rows), "n_positives": int(y.sum()),
        "standardized_coef_log_degree": round(ld_coef, 4),
        "standardized_coef_genetics_score": round(g_coef, 4),
        "genetics_score_90ci_bootstrap_over_diseases": ci,
        "n_bootstrap_fits": len(boots),
        "note": "Standardized coefficients (comparable units). A positive "
                "genetics coefficient with CI excluding 0 means genetics adds "
                "signal beyond popularity. Extreme class imbalance (one true "
                "drug per disease vs the full pool) makes the magnitude less "
                "interpretable than the sign and the degree-stratified table.",
    }


def main() -> None:
    # optional CLI: holdout path + output tag (default = random split)
    holdout_path = Path(sys.argv[1]) if len(sys.argv) > 1 else gab.HOLDOUT
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    out = OUT if not tag else OUT.with_name(f"conditional_lift_report{tag}.json")
    pairs_out = PAIRS_OUT if not tag else \
        PAIRS_OUT.with_name(f"conditional_lift_pairs{tag}.jsonl")
    print(f"Holdout: {holdout_path}  ->  {out.name}")

    # ---- held-out pairs + crosswalks (reuse v3, leak-free) ---------------
    heldout = [(d["compound"], d["disease"])
               for d in (json.loads(l) for l in holdout_path.open())]
    by_disease: dict[str, list[str]] = defaultdict(list)
    for drug, dis in heldout:
        by_disease[dis].append(drug)
    print(f"Held-out pairs: {len(heldout)} across {len(by_disease)} diseases")

    print("Building crosswalks from local Open Targets parquet ...")
    mesh2efo, _ = gab.build_mesh_to_efo()
    db2chembl = gab.build_drugbank_to_chembl()
    chembl2tgt = gab.build_chembl_mechanism()
    sym2ens, _ = gab.build_gene_crosswalk()

    # ---- candidate pool + BOTH degree vectors ----------------------------
    deg_strip = degree_from(STRIPPED)
    deg_full = degree_from(FULL)
    all_compounds = sorted(c for c in set(deg_strip) | set(deg_full)
                           if c.startswith("Compound::"))
    cand_pos = {c: i for i, c in enumerate(all_compounds)}
    cand_deg_strip = np.array([deg_strip.get(c, 0) for c in all_compounds], float)
    cand_deg_full = np.array([deg_full.get(c, 0) for c in all_compounds], float)
    print(f"DRKG compound pool: {len(all_compounds)}  "
          f"(stripped treats edges removed from degree)")

    # ---- mechanism-only drug -> target (the honest high-precision config)-
    targets_mech: dict[str, set[str]] = {}
    for comp in all_compounds:
        if not comp.startswith("Compound::DB"):
            continue
        dbid = comp.split("::")[1]
        mech: set[str] = set()
        for chembl in db2chembl.get(dbid, ()):
            mech |= chembl2tgt.get(chembl, set())
        if mech:
            targets_mech[comp] = mech

    # ---- genetics datasource scores (cache-warm => offline) --------------
    needed: set[str] = set()
    for dis in by_disease:
        for efo in mesh2efo.get(dis.split(":")[-1], []):
            needed.add(efo)
    # optional 3rd CLI arg: a pre-built {disease_id: {gene: score}} genetics
    # snapshot (e.g. the Feb-2020 leak-free snapshot) instead of the live API.
    gen_json = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if gen_json:
        allg = json.loads(gen_json.read_text())
        genetics = {e: allg[e] for e in needed if e in allg}
        print(f"Genetics source: {gen_json} (PRE-2020 snapshot) — "
              f"{len(genetics)}/{len(needed)} needed diseases covered")
    else:
        genetics = gab.fetch_datasource_genetics(needed)

    # ---- score every held-out positive, persist per-pair tuples ----------
    pairs = []  # one row per evaluable held-out positive
    # regression dataset rows: (disease, log_degree, gscore, is_true)
    reg_rows = []
    for dis, drugs in by_disease.items():
        efos = mesh2efo.get(dis.split(":")[-1], [])
        if not efos:
            continue
        gene_scores: dict[str, float] = {}
        for efo in efos:
            for g, s in genetics.get(efo, {}).items():
                if s > gene_scores.get(g, 0.0):
                    gene_scores[g] = s
        if not gene_scores:
            continue
        pool_scores = gab.score_pool(gene_scores, targets_mech, all_compounds)
        held_set = set(drugs)
        # regression: candidates whose genetics score > 0 are informative; to
        # keep the design balanced and within-disease we use the full pool but
        # only for diseases with a covered held-out positive (added below).
        any_covered = any(d in cand_pos and pool_scores[cand_pos[d]] > 0
                          for d in drugs)
        for drug in drugs:
            if drug not in cand_pos:
                continue
            i = cand_pos[drug]
            g_rank = gab.midrank(pool_scores, pool_scores[i])
            pr_strip = gab.midrank(cand_deg_strip, cand_deg_strip[i])
            pr_full = gab.midrank(cand_deg_full, cand_deg_full[i])
            row = {
                "disease": dis, "drug": drug,
                "genetics_score": float(pool_scores[i]),
                "genetics_rank": g_rank,
                "pop_rank_stripped": pr_strip,
                "pop_rank_full": pr_full,
                "degree_stripped": float(cand_deg_strip[i]),
                "covered": bool(pool_scores[i] > 0),
            }
            pairs.append(row)
        # build the within-disease choice set for the logistic lift, only for
        # diseases that have at least one covered held-out positive
        if any_covered:
            for c in all_compounds:
                j = cand_pos[c]
                reg_rows.append((
                    dis,
                    float(np.log1p(cand_deg_strip[j])),
                    float(pool_scores[j]),
                    1 if c in held_set else 0,
                ))

    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    with pairs_out.open("w") as fh:
        for r in pairs:
            fh.write(json.dumps(r) + "\n")

    # ---- (A) corrected ranking ------------------------------------------
    g_all = [p["genetics_rank"] for p in pairs]
    pop_strip_all = [p["pop_rank_stripped"] for p in pairs]
    pop_full_all = [p["pop_rank_full"] for p in pairs]
    covered = [p for p in pairs if p["covered"]]
    g_cov = [p["genetics_rank"] for p in covered]
    pop_strip_cov = [p["pop_rank_stripped"] for p in covered]
    pop_full_cov = [p["pop_rank_full"] for p in covered]

    # ---- (B) degree-stratified Hit@10 on the covered subset --------------
    cov_deg = np.array([p["degree_stripped"] for p in covered])
    strat = {}
    if len(covered) >= 8:
        qs = np.quantile(cov_deg, [0.25, 0.5, 0.75])
        bins = np.digitize(cov_deg, qs)  # 0..3
        for b in range(4):
            idx = [k for k in range(len(covered)) if bins[k] == b]
            if not idx:
                continue
            gr = np.array([covered[k]["genetics_rank"] for k in idx])
            pr = np.array([covered[k]["pop_rank_stripped"] for k in idx])
            strat[f"degree_q{b+1}"] = {
                "n": len(idx),
                "degree_range": [float(cov_deg[idx].min()),
                                 float(cov_deg[idx].max())],
                "genetics_hit_at_10": hit_at(gr, 10),
                "popularity_hit_at_10": hit_at(pr, 10),
                "genetics_median": int(np.median(gr)),
                "popularity_median": int(np.median(pr)),
            }

    # ---- (C) paired head-to-head + bootstrap CI over diseases -----------
    def paired_win_fraction(rows):
        # fraction of covered positives where genetics ranks the true drug
        # strictly better than the clean popularity baseline
        wins = sum(1 for p in rows if p["genetics_rank"] < p["pop_rank_stripped"])
        ties = sum(1 for p in rows if p["genetics_rank"] == p["pop_rank_stripped"])
        return wins, ties, len(rows)

    wins, ties, n_cov = paired_win_fraction(covered)
    # bootstrap over diseases (cluster resample)
    by_dis_cov: dict[str, list] = defaultdict(list)
    for p in covered:
        by_dis_cov[p["disease"]].append(p)
    dis_keys = list(by_dis_cov)
    boot = []
    if dis_keys:
        for _ in range(2000):
            sample_dis = RNG.choice(len(dis_keys), len(dis_keys), replace=True)
            rows = [r for di in sample_dis for r in by_dis_cov[dis_keys[di]]]
            w = sum(1 for p in rows if p["genetics_rank"] < p["pop_rank_stripped"])
            boot.append(w / len(rows) if rows else 0.0)
        ci = [round(float(np.quantile(boot, 0.05)), 3),
              round(float(np.quantile(boot, 0.95)), 3)]
    else:
        ci = [0.0, 0.0]

    logit = compute_logistic_lift(reg_rows)

    report = {
        "purpose": "Popularity-residualized conditional-lift test. The ONLY "
                   "change vs genetics_anchored_benchmark_v3.py is a leak-clean "
                   "popularity baseline (degree from drkg_stripped.tsv, not the "
                   "full graph) and a conditional metric. Ranker leak control is "
                   "inherited unchanged.",
        "baseline_correction": {
            "issue": "Phase-3 popularity degree was built from the FULL DRKG "
                     "graph (drkg.tsv) which still contains the 999 held-out "
                     "treats edges, inflating the comparator by the answers.",
            "fix": "degree recomputed from data/drkg/drkg_stripped.tsv.",
            "held_out_treats_edges_removed": 999,
        },
        "n_evaluable_heldout_positives": len(pairs),
        "n_genetics_covered": len(covered),
        "A_corrected_ranking": {
            "overall": {
                "genetics": summarise(g_all),
                "popularity_CLEAN_stripped": summarise(pop_strip_all),
                "popularity_INFLATED_full": summarise(pop_full_all),
            },
            "genetics_covered_subset": {
                "genetics": summarise(g_cov),
                "popularity_CLEAN_stripped": summarise(pop_strip_cov),
                "popularity_INFLATED_full": summarise(pop_full_cov),
            },
        },
        "B_degree_stratified_hit_at_10_covered": strat,
        "C_conditional_lift": {
            "paired_head_to_head_covered": {
                "n": n_cov,
                "genetics_better_than_popularity": wins,
                "ties": ties,
                "win_fraction": round(wins / n_cov, 3) if n_cov else 0.0,
                "win_fraction_90ci_bootstrap_over_diseases": ci,
                "interpretation": "Fraction of covered held-out positives where "
                                  "genetics ranks the true drug strictly better "
                                  "than the CLEAN popularity baseline. >0.5 with "
                                  "CI excluding 0.5 = genetics adds lift over "
                                  "popularity head-to-head.",
            },
            "logistic_partial_coefficient": logit,
        },
    }
    out.write_text(json.dumps(report, indent=2))

    # ---- console summary -------------------------------------------------
    print("\n" + "=" * 72)
    print("POPULARITY-RESIDUALIZED CONDITIONAL-LIFT REPORT")
    print("=" * 72)
    print(f"Evaluable held-out positives: {len(pairs)}  "
          f"genetics-covered: {len(covered)}")
    print("\n(A) GENETICS-COVERED SUBSET, corrected baseline:")
    print(f"    genetics            Hit@10={summarise(g_cov).get('hit_at_10')}%"
          f"  median={summarise(g_cov).get('median_rank')}")
    print(f"    popularity CLEAN    Hit@10={summarise(pop_strip_cov).get('hit_at_10')}%"
          f"  median={summarise(pop_strip_cov).get('median_rank')}")
    print(f"    popularity INFLATED Hit@10={summarise(pop_full_cov).get('hit_at_10')}%"
          f"  median={summarise(pop_full_cov).get('median_rank')}   <- old comparator")
    print("\n(B) degree-stratified Hit@10 (covered subset):")
    for k, v in strat.items():
        print(f"    {k} (n={v['n']}, deg {v['degree_range'][0]:.0f}-"
              f"{v['degree_range'][1]:.0f}): genetics {v['genetics_hit_at_10']}% "
              f"vs popularity {v['popularity_hit_at_10']}%")
    print("\n(C) paired head-to-head (covered):")
    print(f"    genetics beats clean popularity on {wins}/{n_cov} "
          f"(win frac {round(wins/n_cov,3) if n_cov else 0}, "
          f"90% CI {ci})")
    if logit.get("available"):
        ci = logit.get("genetics_score_90ci") or \
            logit.get("genetics_score_90ci_bootstrap_over_diseases")
        print(f"    logistic std-coef(genetics | log_degree)="
              f"{logit['standardized_coef_genetics_score']}  90%CI {ci}  "
              f"(log_degree std-coef {logit['standardized_coef_log_degree']}, "
              f"engine: {logit['engine']})")
    print(f"\nSaved: {out}\n       {pairs_out}")


if __name__ == "__main__":
    main()
