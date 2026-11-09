"""B0 no-change, B1 train-mean, B3 kNN-GO, B4 additive; plus the registry of every baseline.

Predictors follow `pipeline.Predictor`: (data, split, fit_rows) -> predicted deltas for test rows.
Rows a predictor leaves out fall back to mu in the pipeline and are flagged.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from deltaaudit.models.linear import make_predictor
from deltaaudit.pipeline import run_model

SINGLE, DOUBLE = "unseen_single", "seen_2of2"


def predict_b0(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=split["test"], columns=fit.columns)


def predict_b1(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
    mu = fit.mean(axis=0).to_numpy(np.float64)
    return pd.DataFrame(
        np.tile(mu, (len(split["test"]), 1)), index=split["test"], columns=fit.columns
    )


def predict_b3(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
    """Jaccard-weighted mean delta of the target's GO neighbours among train+val perturbations."""
    nodes, adj = data["go"]
    idx = {n: i for i, n in enumerate(nodes)}
    targets = data["meta"]["targets"]
    test_nodes = [idx[targets[c][0]] for c in split["test"]]
    w = adj[test_nodes][:, [idx[targets[c][0]] for c in fit.index]].toarray()
    s = w.sum(axis=1)
    has = s > 0
    pred = (w[has] @ fit.to_numpy(np.float64)) / s[has, None]
    return pd.DataFrame(pred, index=np.array(split["test"])[has], columns=fit.columns)


def predict_b4(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
    """Additive: delta_A + delta_B from the training singles."""
    pairs = [data["meta"].at[c, "targets"] for c in split["test"]]
    pred = [
        fit.loc[a].to_numpy(np.float64) + fit.loc[b].to_numpy(np.float64)
        for a, b in pairs
    ]
    return pd.DataFrame(np.array(pred), index=split["test"], columns=fit.columns)


def registry(models_cfg: dict, eval_cfg: dict) -> dict[str, tuple]:
    """model -> (predictor, split types it runs on, config recorded in the manifest)."""
    tol = eval_cfg["degenerate_std"]
    b2 = models_cfg["B2"]
    return {
        "B0": (predict_b0, {SINGLE, DOUBLE}, {}),
        "B1": (predict_b1, {SINGLE, DOUBLE}, {}),
        "B2": (make_predictor(b2, b2["defaults"], "gene", tol), {SINGLE}, b2),
        "B2-GO": (
            make_predictor(models_cfg["B2-GO"], b2["defaults"], "go", tol),
            {SINGLE},
            {**models_cfg["B2-GO"], "go_graph": models_cfg["go_graph"]},
        ),
        "B3": (
            predict_b3,
            {SINGLE},
            {**models_cfg["B3"], "go_graph": models_cfg["go_graph"]},
        ),
        "B4": (predict_b4, {DOUBLE}, {}),
    }


def run_baselines(
    root: Path,
    data: dict,
    splits: list[dict],
    eval_cfg: dict,
    models_cfg: dict,
    log=print,
) -> pd.DataFrame:
    frames = []
    for model, (predict, types, cfg) in registry(models_cfg, eval_cfg).items():
        mine = [s for s in splits if s["split_type"] in types]
        if not mine or (model in ("B2-GO", "B3") and "go" not in data):
            continue
        frames.append(run_model(root, data, mine, eval_cfg, model, predict, cfg))
        log(f"{data['name']}/{model}: {len(mine)} split(s) scored")
    return pd.concat(frames, ignore_index=True)
