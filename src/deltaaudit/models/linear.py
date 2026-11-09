"""B2 ridge-linear and B2-GO, following const-ae/linear_perturbation_prediction-Paper @ 95e831e
(benchmark/src/run_linear_pretrained_model.R):

    G = prcomp(train expression, genes x conditions incl. ctrl)$x[, 1:K]
    Y = change (delta) of training conditions with an embedding, plus ctrl (delta 0, p = 0)
    K = (G'G + lI)^-1 G' (Y - c) P' (PP' + lI)^-1,  c = rowMeans(Y)
    delta_hat = G K p + c

B2: p = row of G for the target gene. B2-GO: p = spectral GO embedding of the target.
Test perturbations without an embedding are not predicted (pipeline falls back to mu, flagged).
K and l are chosen per fold by mean inner-validation pspec_r (fit on train, mu = train mean),
then refit on train+val.
"""

import numpy as np
import pandas as pd

from deltaaudit.data.go_graph import spectral_embedding
from deltaaudit.data.leakage import CTRL, fit_rows
from deltaaudit.eval.metrics import pearson_or_zero, target_mask


def gene_embedding(expr: np.ndarray, k: int) -> np.ndarray:
    """PCA scores of genes: expr is conditions x genes; prcomp centres each condition column.
    Returns min(k, n_conditions) columns."""
    x = expr.T - expr.T.mean(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :k] * s[:k]


def solve_y_axb(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided ridge for Y ~ A K B (Y genes x perts); returns (K, row-mean centre)."""
    centre = y.mean(axis=1)
    left = np.linalg.solve(a.T @ a + ridge * np.eye(a.shape[1]), a.T)
    right = np.linalg.solve(b @ b.T + ridge * np.eye(b.shape[0]), b).T
    return left @ (y - centre[:, None]) @ right, centre


def fit_predict(
    y_fit: np.ndarray, g: np.ndarray, p_fit: np.ndarray, p_new: np.ndarray, ridge: float
) -> np.ndarray:
    """y_fit perts x genes, g genes x k, p_* k x perts -> predicted deltas (new perts x genes)."""
    k, centre = solve_y_axb(y_fit.T, g, p_fit, ridge)
    return (g @ k @ p_new + centre[:, None]).T


def _pert_embedding(
    data: dict, conditions: list[str], kind: str, g: np.ndarray
) -> np.ndarray:
    """k x n; NaN columns where a target has no embedding. Doubles sum their targets."""
    k = g.shape[1]
    if kind == "gene":
        cols = {}
        for i, s in enumerate(data["symbols"]):
            cols.setdefault(s, i)  # duplicated symbol (TBCE): first column

        def emb(t: str) -> np.ndarray:
            return g[cols[t]] if t in cols else np.full(k, np.nan)

    else:
        if "go_emb" not in data:
            nodes, adj = data["go"]
            data["go_emb"] = (
                {n: i for i, n in enumerate(nodes)},
                spectral_embedding(adj, data["go_k_max"]),
            )
        idx, e = data["go_emb"]

        def emb(t: str) -> np.ndarray:
            return e[idx[t], :k] if t in idx else np.full(k, np.nan)

    targets = data["meta"].loc[conditions, "targets"]
    return np.array([sum(emb(t) for t in ts) for ts in targets]).reshape(-1, k).T


def _fits(
    data: dict, split: dict, include_val: bool, new: list[str], kind: str,
    ks: list[int], ridges: list[float],
) -> dict[tuple[int, float], pd.DataFrame]:  # fmt: skip
    """Predictions for `new` (rows with an embedding) for every (K, ridge), fitted on
    train (+val) only."""
    delta = fit_rows(data["delta"], split, include_val)
    expr = pd.concat(
        [fit_rows(data["pb_mean"], split, include_val), data["pb_mean"].loc[[CTRL]]]
    )
    g_full = gene_embedding(expr.to_numpy(np.float64), max(ks))
    k_max = g_full.shape[1]
    if kind == "go":
        _pert_embedding(data, [], kind, g_full)  # builds the cached spectral embedding
        k_max = min(k_max, data["go_emb"][1].shape[1])
    out = {}
    for k in (
        k for k in ks if k <= k_max
    ):  # larger K is not identifiable here; skipped, logged
        g = g_full[:, :k]
        p_fit = _pert_embedding(data, delta.index.tolist(), kind, g)
        ok = np.isfinite(p_fit).all(axis=0)
        y = np.vstack([delta.to_numpy(np.float64)[ok], np.zeros(delta.shape[1])])
        p = np.hstack([p_fit[:, ok], np.zeros((k, 1))])
        p_new = _pert_embedding(data, new, kind, g)
        ok_new = np.isfinite(p_new).all(axis=0)
        rows = list(np.array(new)[ok_new])
        for lam in ridges:
            pred = (
                fit_predict(y, g, p, p_new[:, ok_new], lam)
                if rows
                else np.empty((0, y.shape[1]))
            )
            out[k, lam] = pd.DataFrame(pred, index=rows, columns=data["delta"].columns)
    return out


def _val_pspec(data: dict, split: dict, pred: pd.DataFrame, tol: float) -> float:
    mu = (
        fit_rows(data["delta"], split, include_val=False)
        .mean(axis=0)
        .to_numpy(np.float64)
    )
    truth = data["delta"].loc[pred.index].to_numpy(np.float64)
    keep = ~target_mask(
        data["meta"].loc[pred.index, "targets"].tolist(), data["symbols"]
    )
    r, _ = pearson_or_zero(pred.to_numpy() - mu, truth - mu, keep, tol)
    return float(r.mean())


def make_predictor(cfg: dict, defaults: dict, kind: str, degenerate_std: float):
    """B2 (kind='gene') or B2-GO (kind='go') as a pipeline.Predictor."""
    ks, ridges = cfg["K_grid"], cfg["ridge_grid"]

    def predict(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
        data.setdefault("go_k_max", max(ks))
        val = _fits(data, split, False, split["val"], kind, ks, ridges)
        scores = {
            kl: _val_pspec(data, split, p, degenerate_std)
            for kl, p in val.items()
            if len(p)
        }
        k, lam = (
            max(scores, key=scores.get)
            if scores
            else (defaults["K"], defaults["ridge"])
        )
        out = _fits(data, split, True, split["test"], kind, [k], [lam])[k, lam]
        out.attrs["fit_info"] = {
            "K": int(k), "ridge": float(lam),
            "K_skipped": sorted({a for a in ks} - {a for a, _ in val}),
            "n_val_scored": len(next(iter(val.values()))) if scores else 0,
            "val_pspec_r": {f"K={a},ridge={b}": v for (a, b), v in scores.items()},
        }  # fmt: skip
        return out

    return predict
