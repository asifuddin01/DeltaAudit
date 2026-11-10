"""Phase 9 sensitivity analyses (prereg 8): mu_all reference, top-20 genes, full set vs common
support, cluster bootstrap, HVG re-selection on control + training cells (fold 0).

Each analysis re-runs the C1-C6 family and states whether any verdict (+, -, 0 at Holm alpha)
differs from its reference. Nothing here is Holm-gated with the primary family.
"""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from deltaaudit.data import go_graph, hvg
from deltaaudit.data.leakage import SPLITS_DIR
from deltaaudit.eval.audit import (
    SINGLE,
    SUMMARY,
    best_learned,
    compare,
    family,
    per_perturbation,
    seed_average,
)
from deltaaudit.eval.stats import holm
from deltaaudit.pipeline import (
    PROCESSED,
    RESULTS,
    evaluate_split,
    load_dataset,
    load_splits,
    split_tag,
)

TOP20 = {"raw_r": "raw_r20", "pspec_r": "pspec_r20"}


def verdict(obs: float, p_holm: float, alpha: float) -> str:
    return "0" if not p_holm < alpha or obs == 0 else ("+" if obs > 0 else "-")


def rescore(
    root: Path, data: dict, splits: list[dict], eval_cfg: dict, mu: np.ndarray | None
) -> pd.DataFrame:
    """Re-score every saved prediction (pred.npy) with `mu` as the pspec reference
    (None = the fold's train+val mean, i.e. the stored scores). No model is refitted."""
    by_tag = {split_tag(s): s for s in splits}
    base = root / RESULTS / data["name"]
    frames = []
    for man in sorted(base.glob("*/**/MANIFEST.json")):
        parts = man.parent.relative_to(base).parts
        model, run, tag = parts[0], "/".join(parts[1:-1]), parts[-1]
        if tag not in by_tag:
            continue
        split = by_tag[tag]
        assert json.loads(man.read_text())["test_rows"] == split["test"], man
        stored = pd.read_parquet(man.parent / "scores.parquet")
        first = stored[
            stored.analysis_set
            == ("all" if split["split_type"] == SINGLE else "doubles")
        ]
        pred = np.load(man.parent / "pred.npy").astype(np.float64)
        s = evaluate_split(
            pred, first.fallback_to_mu.to_numpy(), split, data, eval_cfg, mu=mu
        )
        s.insert(1, "model", model)
        s.insert(2, "run", run)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def _family_rows(pp, eval_cfg, best, set_name, analysis, metric_map=None) -> list[dict]:
    fam = family(eval_cfg, best)
    if metric_map:
        fam = [{**c, "metric": metric_map[c["metric"]]} for c in fam]
    s = pp[(pp.split_type == SINGLE) & (pp.analysis_set == set_name)]
    if s.empty:
        return []
    ds = s.dataset.iloc[0]
    grp = {"analysis": analysis, "dataset": ds, "analysis_set": set_name}
    return compare(s, fam, grp, eval_cfg["statistics"])


def hvg_panels(
    root: Path,
    ds_cfg: dict,
    cfg: dict,
    splits: list[dict],
    go_cfg: dict,
    m3b: dict,
    gaf: Path,
) -> dict[str, str]:
    """Build <ds>__hvg_all and <ds>__hvg_train (top n_top of the panel by dispersion, from all
    cells vs from control + fold-k train+val cells). Returns variant -> dataset name."""
    name = ds_cfg["name"]
    base = root / PROCESSED / name
    split = next(
        s for s in splits if s["split_type"] == SINGLE and s["fold"] == cfg["fold"]
    )
    a = ad.read_h5ad(root / ds_cfg["h5ad"], backed="r")
    try:
        cond = hvg.cell_conditions(a)
        ctrl = cond == "ctrl"
        masks = {"all": np.ones(a.n_obs, bool),
                 "train": ctrl | np.isin(cond, split["train"] + split["val"])}  # fmt: skip
        mom, xtx, csum = hvg.gene_moments(
            a, masks, ctrl, ds_cfg["preprocess"]["chunk_cells"]
        )
        ens = a.var_names.to_numpy().astype(str)
    finally:
        a.file.close()
    genes = pd.read_parquet(base / "genes.parquet")
    assert (genes.ensembl_id.to_numpy() == ens).all()
    out = {}
    for variant, (n, s1, s2) in mom.items():
        keep = hvg.select(hvg.dispersion_score(n, s1, s2, cfg["n_bins"]), cfg["n_top"])
        dname = f"{name}__hvg_{variant}"
        d = root / PROCESSED / dname
        hvg.derive(
            base,
            d,
            keep,
            xtx,
            csum,
            int(ctrl.sum()),
            ds_cfg["preprocess"]["ctrl_embed_k"],
        )
        hvg.copy_targets_graph(base, d, go_graph.GO_DIR)
        meta, sub_genes = pd.read_parquet(d / "meta.parquet"), pd.read_parquet(
            d / "genes.parquet"
        )
        union = d / go_graph.GO_UNION_DIR
        go_graph.build(go_graph.union_nodes(meta, sub_genes), gaf, go_cfg, union)
        _, adj = go_graph.load(d, go_graph.GO_UNION_DIR)
        sp.save_npz(
            union / "rewired.npz",
            go_graph.rewire(adj, m3b["nswap_per_edge"], m3b["seed"]),
        )
        sd = root / SPLITS_DIR / dname
        sd.mkdir(parents=True, exist_ok=True)
        for s in splits:
            if s["split_type"] == SINGLE:
                (sd / f"{split_tag(s)}.json").write_text(json.dumps(s, indent=1) + "\n")
        out[variant] = dname
    return out


def hvg_overlap(root: Path, names: dict[str, str]) -> dict:
    g = {v: set(pd.read_parquet(root / PROCESSED / n / "genes.parquet").ensembl_id)
         for v, n in names.items()}  # fmt: skip
    a, b = g["all"], g["train"]
    return {"n_top": len(a), "n_shared": len(a & b), "jaccard": len(a & b) / len(a | b)}


def run_sensitivity(
    root: Path, eval_cfg: dict, sec_cfg: dict, hvg_names: dict[str, str] | None
) -> pd.DataFrame:
    """-> results/summary/{sensitivity.csv, sensitivity.md}. Needs evaluate + stats first."""
    out = root / SUMMARY
    alpha = eval_cfg["statistics"]["alpha"]
    primary = eval_cfg["analysis_sets"]["primary"]
    scores = pd.read_csv(out / "scores.csv")
    main = pd.read_csv(out / "comparisons.csv").fillna({"stratum": ""})
    diag = json.loads((out / "diagnosis.json").read_text())
    names = list(
        dict.fromkeys([eval_cfg["primary_dataset"], sec_cfg["replication_dataset"]])
    )
    ref = {
        (r.dataset, r.analysis_set, r.id): r
        for r in main[main.stratum == ""].itertuples()
    }
    rows: list[dict] = []
    notes: dict[tuple[str, str], str] = {}
    for ds in names:
        best = diag[ds]["best_learned"]["selected"]
        pp = scores[scores.dataset == ds]
        # 1. mu = centroid of all perturbations
        data, splits = load_dataset(root, ds), load_splits(root, ds)
        mu_all = data["delta"].mean(axis=0).to_numpy(np.float64)
        n_dbl = int(data["meta"].is_double.sum())
        notes[("mu_all", ds)] = (
            f"μ_all averages {len(data['delta'])} perturbations ({n_dbl} doubles)"
        )
        re = seed_average(
            rescore(root, data, splits, eval_cfg, mu_all), eval_cfg["metrics"]
        )
        rows += [
            {**r, "ref": "main"}
            for r in _family_rows(re, eval_cfg, best, primary, "mu_all")
        ]
        # 2. top-20 genes instead of all genes
        rows += [{**r, "ref": "main"}
                 for r in _family_rows(pp, eval_cfg, best, primary, "top20_genes", TOP20)]  # fmt: skip
        # 3. full set instead of common support (already in comparisons.csv)
        for r in main[
            (main.dataset == ds) & (main.analysis_set == "all") & (main.stratum == "")
        ].to_dict("records"):
            rows.append(
                {**r, "analysis": "full_set", "ref_set": primary, "ref": "main"}
            )
        # 4. cluster bootstrap on the primary set
        m = main[
            (main.dataset == ds) & (main.analysis_set == primary) & (main.stratum == "")
        ]
        ph = holm(m.cluster_p_raw.to_numpy())
        notes[("cluster_bootstrap", ds)] = (
            f"{int(m.cluster_n_clusters.iloc[0])} GO clusters for {int(m.n.iloc[0])} perturbations"
        )
        for r, p in zip(m.to_dict("records"), ph, strict=True):
            rows.append({**r, "analysis": "cluster_bootstrap", "p_raw": r["cluster_p_raw"],
                         "ci_lo": r["cluster_ci_lo"], "ci_hi": r["cluster_ci_hi"], "p_holm": p, "ref": "main"})  # fmt: skip
    # 5. HVG re-selection, one fold: train-cell panel vs all-cell panel vs full panel
    if hvg_names:
        ds = eval_cfg["primary_dataset"]
        fold = sec_cfg["sensitivity"]["hvg_reselection"]["fold"]
        full = scores[(scores.dataset == ds) & (scores.fold == fold)]
        best = diag[ds]["best_learned"]["selected"]
        full_rows = {}
        for set_name in (primary, "all"):
            for r in _family_rows(
                full, eval_cfg, best, set_name, f"full_panel_fold{fold}"
            ):
                full_rows[(set_name, r["id"])] = r
                rows.append({**r, "ref": "none"})
        pps = {}
        for variant, dname in hvg_names.items():
            data, splits = load_dataset(root, dname), load_splits(root, dname)
            splits = [s for s in splits if s["fold"] == fold]
            pps[variant] = (per_perturbation(root, data, splits, eval_cfg["metrics"]),
                            best_learned(root, dname, eval_cfg["best_learned"]["candidates"])["selected"])  # fmt: skip
        for set_name in (primary, "all"):
            for variant, refname in (
                ("all", f"full_panel_fold{fold}"),
                ("train", "hvg_all"),
            ):
                pp, b = pps[variant]
                for r in _family_rows(pp, eval_cfg, b, set_name, f"hvg_{variant}"):
                    rows.append({**r, "dataset": ds, "ref": refname})
    df = pd.DataFrame(rows)
    df["verdict"] = [
        verdict(o, p, alpha) for o, p in zip(df.obs, df.p_holm, strict=True)
    ]
    refs = [_ref(r, df, ref, alpha, primary) for r in df.to_dict("records")]
    df["ref_verdict"], df["ref_obs"], df["ref_n"] = zip(*refs, strict=True)
    cols = ["analysis", "dataset", "analysis_set", "id", "a", "b", "metric", "n", "obs", "ci_lo",
            "ci_hi", "p_raw", "p_holm", "verdict", "ref", "ref_verdict", "ref_obs", "ref_n"]  # fmt: skip
    df = df[cols]
    df.to_csv(out / "sensitivity.csv", index=False, float_format="%.6g")
    overlap = hvg_overlap(root, hvg_names) if hvg_names else None
    (out / "sensitivity.md").write_text(render(df, overlap, notes))
    return df


def _ref(r: dict, df: pd.DataFrame, ref: dict, alpha: float, primary: str) -> tuple:
    """(verdict, obs, n) of the row's reference comparison; blanks if it has none."""
    if r["ref"] == "main":
        m = ref[(r["dataset"], primary, r["id"])]
        return verdict(m.obs, m.p_holm, alpha), m.obs, m.n
    m = df[
        (df.analysis == r["ref"])
        & (df.analysis_set == r["analysis_set"])
        & (df.id == r["id"])
    ]
    if r["ref"] == "none" or len(m) != 1:
        return "", np.nan, np.nan
    return verdict(m.obs.item(), m.p_holm.item(), alpha), m.obs.item(), m.n.item()


def why(v: str, ref_v: str, obs: float, ref_obs: float) -> str:
    """Mechanical reason a verdict changed."""
    if ref_v == "0":
        return "becomes significant"
    if v == "0":
        same = np.sign(obs) == np.sign(ref_obs)
        return (
            "same direction, no longer significant"
            if same
            else "direction reversed, not significant"
        )
    return "direction reversed"


LABELS = {
    "mu_all": "pspec reference μ = centroid of all perturbations (instead of the fold's train+val mean)",
    "top20_genes": "top-20 genes by |Δ| (raw_r20, pspec_r20) instead of all genes",
    "full_set": "all perturbations (B2 falls back to μ out of panel) instead of common support",
    "cluster_bootstrap": "cluster bootstrap over GO-similarity clusters instead of single perturbations",
    "hvg_all": "fold 0, panel re-selected from all cells (vs the full 5,000-gene panel, fold 0)",
    "hvg_train": "fold 0, panel re-selected from control + train+val cells only (vs the all-cell panel)",
}


def render(df: pd.DataFrame, overlap: dict | None, notes: dict | None = None) -> str:
    """One line per (analysis, dataset, set): 'conclusion unchanged' or what changed and why."""
    lines = ["# Sensitivity analyses (Phase 9)", "",
             "Generated by `deltaaudit sensitivity` from `results/summary/sensitivity.csv`. "
             "Verdict per comparison: `+`/`-` = significant at Holm α within that analysis, "
             "`0` = not significant. Not Holm-gated with the primary family.", ""]  # fmt: skip
    if overlap:
        lines += [f"HVG panels: top {overlap['n_top']} of the 5,000-gene panel; the all-cell and "
                  f"train-cell panels share {overlap['n_shared']} genes (Jaccard {overlap['jaccard']:.3f}).", ""]  # fmt: skip
    for (an, ds, st), g in df[df.ref != "none"].groupby(
        ["analysis", "dataset", "analysis_set"], sort=False
    ):
        pattern = ", ".join(f"{r.id} {r.verdict}" for r in g.itertuples())
        diff = g[g.verdict != g.ref_verdict]
        if diff.empty:
            what = "**conclusion unchanged**"
        else:
            parts = [f"{r.id} {r.ref_verdict}→{r.verdict}, {why(r.verdict, r.ref_verdict, r.obs, r.ref_obs)} "
                     f"(d = {r.obs:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], p_holm = {r.p_holm:.3g}; "
                     f"n = {r.n} vs {r.ref_n:.0f} in the reference)" for r in diff.itertuples()]  # fmt: skip
            what = "**changed because** " + "; ".join(parts)
        note = (notes or {}).get((an, ds))
        ctx = f" ({note})" if note else ""
        lines.append(
            f"- `{an}` · {ds} · {st}: {LABELS.get(an, an)}{ctx}. {what}. Verdicts: {pattern}."
        )
    return "\n".join(lines) + "\n"


def fit_fold(
    root: Path, name: str, eval_cfg: dict, models_cfg: dict, fold: int,
    seeds: list[int], device: str, log=print,
) -> None:  # fmt: skip
    """Every single-perturbation model on one fold of a derived dataset (resumable for nets)."""
    from deltaaudit.models.baselines import run_baselines
    from deltaaudit.models.train import VARIANTS, run_learned

    data = load_dataset(root, name)
    splits = [
        s
        for s in load_splits(root, name)
        if s["split_type"] == SINGLE and s["fold"] == fold
    ]
    run_baselines(root, data, splits, eval_cfg, models_cfg, log)
    run_learned(
        root,
        data,
        splits,
        eval_cfg,
        models_cfg["learned"],
        list(VARIANTS),
        seeds,
        device,
        log,
    )
