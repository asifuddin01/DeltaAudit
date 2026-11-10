"""Fit -> predict -> score loop shared by every model (baselines, linear, nets).

A predictor gets the processed data, one split, and the fit rows (train+val, via `fit_rows`)
and returns predicted deltas for some or all test perturbations. Missing rows fall back to mu
and are flagged (the pre-registered rule for B2 on out-of-panel targets).
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from deltaaudit.data import go_graph
from deltaaudit.data.leakage import SPLITS_DIR, fit_rows
from deltaaudit.eval.metrics import score, target_mask
from deltaaudit.eval.stats import bootstrap_ci
from deltaaudit.manifest import build_manifest, up_to_date

PROCESSED = Path("data/processed")
RESULTS = Path("results")
IN_PANEL = "target_in_expression_genes"

Predictor = Callable[[dict, dict, pd.DataFrame], pd.DataFrame]


def load_dataset(root: Path, name: str) -> dict:
    d = root / PROCESSED / name
    tables = ("delta", "meta", "genes", "pb_mean")
    data = {k: pd.read_parquet(d / f"{k}.parquet") for k in tables}
    data["name"] = name
    data["inputs"] = [f"{PROCESSED}/{name}/{k}.parquet" for k in tables]
    if (d / "ctrl_embed.parquet").exists():
        data["ctrl_embed"] = pd.read_parquet(d / "ctrl_embed.parquet")
        data["inputs"].append(f"{PROCESSED}/{name}/ctrl_embed.parquet")
    for key, sub in (("go", go_graph.GO_DIR), ("go_union", go_graph.GO_UNION_DIR)):
        if (d / sub / "graph.npz").exists():
            data[key] = go_graph.load(d, sub)
            data["inputs"] += [
                f"{PROCESSED}/{name}/{sub}/{f}" for f in ("graph.npz", "nodes.json")
            ]
    if (d / go_graph.GO_UNION_DIR / "rewired.npz").exists():
        data["go_rewired"] = sp.load_npz(
            d / go_graph.GO_UNION_DIR / "rewired.npz"
        ).tocsr()
        data["inputs"].append(f"{PROCESSED}/{name}/{go_graph.GO_UNION_DIR}/rewired.npz")
    data["symbols"] = (
        data["genes"]
        .set_index("ensembl_id")
        .loc[data["delta"].columns, "gene_symbol"]
        .to_numpy()
    )
    data["meta"] = data["meta"].set_index("condition", drop=False)
    return data


def load_splits(root: Path, name: str) -> list[dict]:
    files = sorted((root / SPLITS_DIR / name).glob("*.json"))
    return [json.loads(f.read_text()) for f in files if f.name != "summary.json"]


def split_tag(split: dict) -> str:
    """Result folder / split file stem: fold_k, or doubles_seen_2of2."""
    if split["split_type"] == "unseen_single":
        return f"fold_{split['fold']}"
    return f"doubles_{split['split_type']}"


def analysis_sets(split: dict, meta: pd.DataFrame) -> dict[str, np.ndarray]:
    test = split["test"]
    if split["split_type"] == "seen_2of2":
        return {"doubles": np.ones(len(test), dtype=bool)}
    in_panel = meta.loc[test, IN_PANEL].to_numpy(bool)
    return {
        "common_support": in_panel,
        "all": np.ones(len(test), dtype=bool),
        "out_of_panel": ~in_panel,
    }


def fit_mu(delta: pd.DataFrame, split: dict) -> np.ndarray:
    """mu = mean delta over the fold's train+val perturbations (pre-registered)."""
    return fit_rows(delta, split).mean(axis=0).to_numpy(np.float64)


def evaluate_split(
    pred: np.ndarray, fallback: np.ndarray, split: dict, data: dict, eval_cfg: dict
) -> pd.DataFrame:
    test = split["test"]
    truth = data["delta"].loc[test].to_numpy(np.float64)
    keep = ~target_mask(data["meta"].loc[test, "targets"].tolist(), data["symbols"])
    mu = fit_mu(data["delta"], split)
    frames = []
    for set_name, rows in analysis_sets(split, data["meta"]).items():
        if not rows.any():
            continue
        s = score(
            pred[rows],
            truth[rows],
            mu,
            keep[rows],
            eval_cfg["top_k_genes"],
            eval_cfg["degenerate_std"],
        )
        s.insert(0, "condition", np.array(test)[rows])
        s.insert(1, "analysis_set", set_name)
        s["fallback_to_mu"] = fallback[rows]
        frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "dataset", data["name"])
    out.insert(2, "split_type", split["split_type"])
    out.insert(3, "fold", split["fold"])
    return out


def run_model(
    root: Path,
    data: dict,
    splits: list[dict],
    eval_cfg: dict,
    model: str,
    predict: Predictor,
    config: dict,
    run: str = "",
    resume: bool = False,
    log=lambda _: None,
) -> pd.DataFrame:
    """Fit/predict/score every split; write
    results/<dataset>/<model>/[<run>/]<split>/{pred.npy, scores.parquet, MANIFEST.json}.
    With resume, a split whose MANIFEST matches config, code and inputs is read back, not refit.
    """
    all_scores = []
    for split in splits:
        out = root / RESULTS / data["name"] / model / run / split_tag(split)
        inputs = data["inputs"] + [
            f"{SPLITS_DIR}/{data['name']}/{split_tag(split)}.json"
        ]
        conf = {"model": model, "config": config, "eval": eval_cfg}
        if resume and up_to_date(out / "MANIFEST.json", root, conf, inputs):
            all_scores.append(pd.read_parquet(out / "scores.parquet"))
            log(f"{data['name']}/{model}/{run}/{split_tag(split)}: up to date, skipped")
            continue
        t0 = time.time()
        fit = fit_rows(data["delta"], split)
        test = split["test"]
        raw = predict(data, split, fit)
        got = raw.reindex(index=test, columns=data["delta"].columns)
        fallback = got.isna().all(axis=1).to_numpy()
        pred = got.to_numpy(np.float64)
        pred[fallback] = fit_mu(data["delta"], split)
        scores = evaluate_split(pred, fallback, split, data, eval_cfg)
        scores.insert(1, "model", model)
        scores.insert(2, "run", run)

        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "pred.npy", pred.astype(np.float32))
        scores.to_parquet(out / "scores.parquet")
        manifest = build_manifest(root, conf, inputs, t0)
        manifest["test_rows"] = test
        manifest["fit_info"] = raw.attrs.get("fit_info", {})
        if "device" in manifest["fit_info"]:
            manifest["device"] = manifest["fit_info"][
                "device"
            ]  # the device actually used
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        all_scores.append(scores)
        log(
            f"{data['name']}/{model}/{run}/{split_tag(split)}: {manifest['wall_time_s']:.1f} s"
        )
    return pd.concat(all_scores, ignore_index=True)


def summarize(
    scores: pd.DataFrame, metrics: list[str], stats_cfg: dict
) -> pd.DataFrame:
    """Mean and bootstrap CI per (dataset, model, split_type, analysis_set, fold|pooled, metric)."""
    keys = ["dataset", "model", "run", "split_type", "analysis_set"]
    scores = scores.assign(run=scores["run"].fillna("") if "run" in scores else "")
    rows = []
    for key, g in scores.groupby(keys, sort=True):
        parts = [("pooled", g)] + [(str(f), gf) for f, gf in g.groupby("fold")]
        for fold, gf in parts:
            for m in metrics:
                x = gf[m].dropna().to_numpy()
                lo, hi = (
                    bootstrap_ci(
                        x,
                        stats_cfg["bootstrap_B"],
                        stats_cfg["bootstrap_seed"],
                        stats_cfg["ci"],
                    )
                    if len(x)
                    else (np.nan, np.nan)
                )
                rows.append(
                    {**dict(zip(keys, key, strict=True)), "fold": fold, "metric": m, "n": len(x),
                     "mean": float(x.mean()) if len(x) else np.nan, "ci_lo": lo, "ci_hi": hi,
                     "n_fallback_to_mu": int(gf["fallback_to_mu"].sum())}  # fmt: skip
                )
    return pd.DataFrame(rows)
