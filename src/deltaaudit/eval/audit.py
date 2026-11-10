"""Phase 8: seed-averaged per-perturbation scores, F2 diagnostics, F3 comparisons, decision rules.

Inputs are the per-run `scores.parquet` files under results/<dataset>/; nothing is refitted.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from deltaaudit.data import go_graph
from deltaaudit.eval.ceilings import describe
from deltaaudit.eval.ladder import ladder
from deltaaudit.eval.stats import (
    bootstrap_ci,
    cluster_test,
    go_clusters,
    holm,
    paired_test,
    spearman_ci,
)
from deltaaudit.pipeline import PROCESSED, RESULTS, fit_mu

KEYS = ["dataset", "model", "split_type", "fold", "analysis_set", "condition"]
FLAGS = ["fallback_to_mu", "degenerate_raw", "degenerate_pspec"]
SINGLE = "unseen_single"


def collect(root: Path, name: str) -> pd.DataFrame:
    files = sorted((root / RESULTS / name).glob("*/**/scores.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def seed_average(scores: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """One row per (model, perturbation, analysis set): metric means over runs (seeds, and the
    permutations for M3; prereg 4). Flags become the fraction of runs flagged."""
    g = scores.groupby(KEYS, sort=True, dropna=False)
    out = g[metrics].mean().join(g[FLAGS].mean()).assign(n_runs=g.size())
    return out.reset_index()


def fold_ceilings(root: Path, data: dict, splits: list[dict]) -> pd.DataFrame:
    """Split-half ceilings per test perturbation; ceil_pspec uses that fold's mu (prereg 5)."""
    d = root / PROCESSED / data["name"]
    half_a, half_b = (pd.read_parquet(d / f"pb_half_{h}.parquet") for h in "AB")
    frames = []
    for s in splits:
        c = describe(data["delta"].loc[s["test"]], half_a, half_b, data["meta"],
                     data["genes"], mu=fit_mu(data["delta"], s))  # fmt: skip
        frames.append(
            c[["condition", "ceil_raw", "ceil_pspec"]].assign(
                split_type=s["split_type"]
            )
        )
    return pd.concat(frames, ignore_index=True)


def per_perturbation(
    root: Path, data: dict, splits: list[dict], metrics: list[str]
) -> pd.DataFrame:
    pp = seed_average(collect(root, data["name"]), metrics)
    pp = pp.merge(
        fold_ceilings(root, data, splits), on=["split_type", "condition"], how="left"
    )
    sys_cos = pd.read_parquet(root / PROCESSED / data["name"] / "descriptive.parquet")
    pp = pp.merge(sys_cos[["condition", "sys_cos"]], on="condition", how="left")
    b3 = pp[(pp.model == "B3") & (pp.analysis_set == "all")]
    has = dict(zip(b3.condition, b3.fallback_to_mu == 0, strict=True))
    return pp.assign(has_go_neighbour=pp.condition.map(has))


def best_learned(root: Path, name: str, candidates: list[str]) -> dict:
    """Prereg 6: the C1 model is the candidate with the higher mean inner-validation raw_r
    over folds and seeds. Read from the training manifests; no test score is used."""
    vals = {
        m: [json.loads(p.read_text())["fit_info"]["val_raw_r"]
            for p in sorted((root / RESULTS / name / m).glob("seed_*/fold_*/MANIFEST.json"))]
        for m in candidates
    }  # fmt: skip
    means = {m: float(np.mean(v)) for m, v in vals.items() if v}
    return {"val_raw_r": means, "n_runs": {m: len(v) for m, v in vals.items()},
            "selected": max(means, key=means.get) if means else None}  # fmt: skip


def _wide(pp: pd.DataFrame, metric: str) -> pd.DataFrame:
    return pp.pivot(index="condition", columns="model", values=metric)


def _frac_ceiling(pp: pd.DataFrame) -> dict:
    out = {}
    for m, g in pp.groupby("model"):
        ok = g[["pspec_r", "ceil_pspec"]].dropna()
        out[m] = float(ok.pspec_r.mean() / ok.ceil_pspec.mean()) if len(ok) else None
    return out


def diagnose(
    pp: pd.DataFrame, set_name: str, best: dict, stats_cfg: dict, sec_cfg: dict
) -> dict:
    """BUILD_GUIDE F2 on one dataset's singles, one analysis set."""
    s = pp[(pp.split_type == SINGLE) & (pp.analysis_set == set_name)]
    if s.empty:
        return {}
    boot = (stats_cfg["bootstrap_B"], stats_cfg["bootstrap_seed"], stats_cfg["ci"])
    b1 = s[s.model == "B1"].dropna(subset=["raw_r", "sys_cos"])
    rho, rlo, rhi = spearman_ci(b1.raw_r, b1.sys_cos, *boot)
    means = s.groupby("model")[["raw_r", "pspec_r"]].mean()
    rank = {
        m: means[m].rank(ascending=False, method="min").astype(int).to_dict()
        for m in means
    }
    models = list(means.index)
    flips = [[a, b] for i, a in enumerate(models) for b in models[i + 1:]
             if np.sign(means.raw_r[a] - means.raw_r[b]) * np.sign(means.pspec_r[a] - means.pspec_r[b]) < 0]  # fmt: skip
    return {
        "n_perturbations": int(s.condition.nunique()),
        "B1_raw_r": {
            "mean": float(b1.raw_r.mean()),
            "ci": list(bootstrap_ci(b1.raw_r, *boot)),
        },
        "spearman_B1_raw_r_vs_sys_cos": {"rho": rho, "ci": [rlo, rhi], "n": len(b1)},
        "mean": {m: means.loc[m].to_dict() for m in models},
        "rank": {m: {k: rank[k][m] for k in rank} for m in models},
        "flipped_pairs_raw_vs_pspec": flips,
        "frac_of_noise_ceiling_pspec": _frac_ceiling(s),
        "mean_ceil_pspec": float(s[s.model == "B1"].ceil_pspec.mean()),
        "n_undefined": {m: {k: int(v) for k, v in g[["raw_r", "pspec_r"]].isna().sum().items()}
                        for m, g in s.groupby("model")},  # fmt: skip
        "example": _example(
            s, best["selected"], sec_cfg["example_perturbation"]["max_abs_pspec_r"]
        ),
    }


def _example(s: pd.DataFrame, model: str | None, max_abs: float) -> dict | None:
    """F2 (iv): the best-learned model's highest-raw_r perturbation with |pspec_r| < max_abs."""
    if model is None:
        return None
    m = s[(s.model == model) & (s.pspec_r.abs() < max_abs)]
    if m.empty:
        return None
    row = m.loc[m.raw_r.idxmax()]
    b1 = s[(s.model == "B1") & (s.condition == row.condition)].raw_r
    return {"rule": f"{model}: max raw_r with |pspec_r| < {max_abs}", "condition": row.condition,
            "model": model, "raw_r": float(row.raw_r), "pspec_r": float(row.pspec_r),
            "B1_raw_r": float(b1.iloc[0]) if len(b1) else None, "sys_cos": float(row.sys_cos),
            "ceil_raw": float(row.ceil_raw), "ceil_pspec": float(row.ceil_pspec)}  # fmt: skip


def compare(
    pp: pd.DataFrame, comps: list[dict], group: dict, stats_cfg: dict,
    clusters: dict | None = None,
) -> list[dict]:  # fmt: skip
    """Paired tests of d = A - B on one set of perturbations; Holm within the `comps` list."""
    boot = (stats_cfg["bootstrap_B"], stats_cfg["bootstrap_seed"], stats_cfg["ci"])
    rows = []
    for c in comps:
        w = _wide(pp, c["metric"])
        if c["a"] not in w or c["b"] not in w:
            continue
        both = w[[c["a"], c["b"]]].dropna(how="all")
        pair = both.dropna()
        d = (pair[c["a"]] - pair[c["b"]]).to_numpy()
        if not len(d):
            continue
        row = {**group, **c, "n_dropped": len(both) - len(pair),
               "mean_a": float(pair[c["a"]].mean()), "mean_b": float(pair[c["b"]].mean()),
               **paired_test(d, *boot)}  # fmt: skip
        if clusters is not None:
            cl = cluster_test(d, pair.index.map(clusters).to_numpy(), *boot)
            row.update({f"cluster_{k}": v for k, v in cl.items() if k != "obs"})
        rows.append(row)
    p_holm = holm(np.array([r["p_raw"] for r in rows])) if rows else []
    for r, p in zip(rows, p_holm, strict=True):
        r["p_holm"] = float(p)
    return rows


def family(eval_cfg: dict, best: str | None) -> list[dict]:
    return [{**c, "a": best if c["a"] == "best_learned" else c["a"], "a_label": c["a"]}
            for c in eval_cfg["family"]]  # fmt: skip


def compare_all(
    pp: pd.DataFrame, eval_cfg: dict, sec_cfg: dict, best: dict, clusters: dict
) -> pd.DataFrame:
    """Primary family per analysis set and dataset, out-of-panel C4/C5 strata, doubles."""
    stats_cfg, primary = eval_cfg["statistics"], eval_cfg["analysis_sets"]["primary"]
    rows = []
    for ds, g in pp.groupby("dataset"):
        fam = family(eval_cfg, best[ds]["selected"])
        singles = g[g.split_type == SINGLE]
        for set_name in sec_cfg["family_sets"]:
            s = singles[singles.analysis_set == set_name]
            is_primary = ds == eval_cfg["primary_dataset"] and set_name == primary
            grp = {"dataset": ds, "analysis_set": set_name, "stratum": "",
                   "family": "primary" if is_primary else "secondary"}  # fmt: skip
            rows += compare(s, fam, grp, stats_cfg, clusters.get(ds))
            if set_name == "out_of_panel":
                strata = sec_cfg["out_of_panel_strata"]
                sub = [c for c in fam if c["id"] in strata["comparisons"]]
                for val in (True, False):
                    st = s[s[strata["by"]] == val]
                    rows += compare(st, sub, {**grp, "stratum": f"{strata['by']}={val}"},
                                    stats_cfg, clusters.get(ds))  # fmt: skip
        dbl = g[g.split_type != SINGLE]
        if not dbl.empty:
            cfg = sec_cfg["doubles"]
            comps = [{"id": f"D{i + 1}", "a": a, "b": b, "metric": m, "hypothesis": "none", "a_label": a}
                     for i, ((a, b), m) in enumerate((p, m) for m in cfg["metrics"] for p in cfg["pairs"])]  # fmt: skip
            grp = {
                "dataset": ds,
                "analysis_set": "doubles",
                "stratum": "",
                "family": "doubles",
            }
            rows += compare(dbl, comps, grp, stats_cfg)
    return pd.DataFrame(rows)


def cluster_labels(root: Path, data: dict, gaf: Path, go_cfg: dict, cut: float) -> dict:
    """GO-similarity cluster per single perturbation (prereg 8 cluster bootstrap)."""
    meta = data["meta"]
    singles = meta[meta.targets.map(len) == 1].condition.tolist()
    terms, syn = go_graph.read_gaf(gaf, go_cfg["aspects"], go_cfg["exclude_evidence"])
    sets, _ = go_graph.resolve([meta.at[c, "targets"][0] for c in singles], terms, syn)
    return dict(zip(singles, go_clusters(sets, cut).tolist(), strict=True))


def decisions(comps: pd.DataFrame, diagnosis: dict, eval_cfg: dict, ds: str) -> dict:
    """Prereg 7 decision rules on one dataset's primary analysis set."""
    alpha, rules = eval_cfg["statistics"]["alpha"], eval_cfg["decision_rules"]
    primary = eval_cfg["analysis_sets"]["primary"]
    c = comps[
        (comps.dataset == ds) & (comps.analysis_set == primary) & (comps.stratum == "")
    ]
    c = {r["id"]: r for r in c.to_dict("records")}
    diag = diagnosis[ds][primary]
    sig = lambda i: i in c and c[i]["obs"] > 0 and c[i]["p_holm"] < alpha  # noqa: E731
    rho = diag["spearman_B1_raw_r_vs_sys_cos"]["rho"]
    h1 = (
        "C1" in c
        and c["C1"]["ci_hi"] < rules["H1"]["equivalence_margin"]
        and rho > rules["H1"]["spearman_threshold"]
    )
    b2 = diag["mean"].get("B2", {}).get("pspec_r", np.nan)
    far = b2 < rules["H2"]["far_below_ceiling_fraction"] * diag["mean_ceil_pspec"]
    out = {
        "H1": {"supported": bool(h1), "C1_ci_hi": c.get("C1", {}).get("ci_hi"), "rho": rho,
               "best_learned": c.get("C1", {}).get("a")},
        "H2": {"supported": bool(sig("C2") and far), "C2_significant_positive": sig("C2"),
               "B2_mean_pspec_r": b2, "mean_ceil_pspec": diag["mean_ceil_pspec"],
               "far_below_ceiling": bool(far)},
        "H3": {"status": "contradicted" if sig("C3") else "stands"},
        "H4a": {"supported": sig("C4")},
        "H4b": {"supported": sig("C5")},
        "C6": {k: c["C6"][k] for k in ("obs", "ci_lo", "ci_hi", "p_holm")} if "C6" in c else None,
    }  # fmt: skip
    return out


SUMMARY = Path("results/summary")


def run_evaluate(root: Path, names: list[str], eval_cfg: dict, sec_cfg: dict) -> dict:
    """-> results/summary/{scores.csv, model_table.csv, diagnosis.json}."""
    from deltaaudit.pipeline import load_dataset, load_splits, summarize

    frames, diagnosis = [], {}
    for name in names:
        data, splits = load_dataset(root, name), load_splits(root, name)
        pp = per_perturbation(root, data, splits, eval_cfg["metrics"])
        best = best_learned(root, name, eval_cfg["best_learned"]["candidates"])
        diagnosis[name] = {"best_learned": best} | {
            s: diagnose(pp, s, best, eval_cfg["statistics"], sec_cfg)
            for s in sec_cfg["family_sets"]
        }
        frames.append(pp)
    scores = pd.concat(frames, ignore_index=True)
    out = root / SUMMARY
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False, float_format="%.6g")
    table = summarize(scores, eval_cfg["metrics"], eval_cfg["statistics"])
    table.to_csv(out / "model_table.csv", index=False, float_format="%.6g")
    (out / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    return {"scores": scores, "table": table, "diagnosis": diagnosis}


def run_stats(
    root: Path,
    eval_cfg: dict,
    sec_cfg: dict,
    rungs: list[dict],
    gaf: Path,
    go_cfg: dict,
) -> dict:
    """-> results/summary/{comparisons.csv, decisions.json, ladder.csv}; needs run_evaluate first."""
    from deltaaudit.pipeline import load_dataset

    out = root / SUMMARY
    scores = pd.read_csv(out / "scores.csv")
    diagnosis = json.loads((out / "diagnosis.json").read_text())
    names = sorted(scores.dataset.unique())
    cut = eval_cfg["cluster_bootstrap"]["cut_height"]
    clusters = {
        n: cluster_labels(root, load_dataset(root, n), gaf, go_cfg, cut) for n in names
    }
    best = {n: diagnosis[n]["best_learned"] for n in names}
    comps = compare_all(scores, eval_cfg, sec_cfg, best, clusters)
    comps.to_csv(out / "comparisons.csv", index=False, float_format="%.6g")
    dec = {n: decisions(comps, diagnosis, eval_cfg, n) for n in names}
    dec["n_clusters"] = {n: len(set(c.values())) for n, c in clusters.items()}
    (out / "decisions.json").write_text(json.dumps(dec, indent=2, default=float) + "\n")
    primary = eval_cfg["analysis_sets"]["primary"]
    sel = lambda n: scores[(scores.dataset == n) & (scores.split_type == SINGLE)  # noqa: E731
                           & (scores.analysis_set == primary)]  # fmt: skip
    lad = ladder(sel(eval_cfg["primary_dataset"]), sel(sec_cfg["replication_dataset"]),
                 rungs, eval_cfg["statistics"])  # fmt: skip
    lad.insert(1, "dataset", eval_cfg["primary_dataset"])
    lad.insert(2, "analysis_set", primary)
    lad.to_csv(out / "ladder.csv", index=False)
    return {"comparisons": comps, "decisions": dec, "ladder": lad}
