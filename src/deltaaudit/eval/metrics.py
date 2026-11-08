"""Per-perturbation metrics. The target gene is excluded from every metric (CLAUDE.md rule 6):
callers pass `keep`, a perturbations x genes boolean mask that is False on target columns.
"""

import numpy as np
import pandas as pd


def target_mask(targets: list[list[str]], symbols: np.ndarray) -> np.ndarray:
    """True where a column's gene symbol is one of that perturbation's targets.
    A symbol mapping to several columns (e.g. duplicated TBCE) masks all of them."""
    mask = np.zeros((len(targets), len(symbols)), dtype=bool)
    for i, ts in enumerate(targets):
        if len(ts):
            mask[i] = np.isin(symbols, list(ts))
    return mask


def rowwise_pearson(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Pearson r per row over kept columns. NaN where either side has zero variance."""
    n = keep.sum(axis=1)
    am = np.where(keep, a, 0.0).sum(axis=1) / n
    bm = np.where(keep, b, 0.0).sum(axis=1) / n
    ac = np.where(keep, a - am[:, None], 0.0)
    bc = np.where(keep, b - bm[:, None], 0.0)
    den = np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ac * bc).sum(axis=1) / den, np.nan)


def rowwise_cosine(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Cosine similarity per row over kept columns. NaN for a zero vector."""
    a, b = np.where(keep, a, 0.0), np.where(keep, b, 0.0)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=1) / den, np.nan)


def topk_mask(values: np.ndarray, keep: np.ndarray, k: int) -> np.ndarray:
    """Per row, the k kept columns with the largest |values|."""
    v = np.where(keep, np.abs(values), -np.inf)
    k = min(k, v.shape[1])
    idx = np.argpartition(-v, k - 1, axis=1)[:, :k]
    mask = np.zeros_like(keep)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask & keep


def _row_std(x: np.ndarray, keep: np.ndarray) -> np.ndarray:
    n = keep.sum(axis=1)
    m = np.where(keep, x, 0.0).sum(axis=1) / n
    return np.sqrt(np.where(keep, (x - m[:, None]) ** 2, 0.0).sum(axis=1) / n)


def pearson_or_zero(pred, truth, keep, tol):
    """Pearson r; a constant prediction (std < tol) scores 0 and is flagged degenerate."""
    degenerate = _row_std(pred, keep) < tol
    r = rowwise_pearson(pred, truth, keep)
    return np.where(degenerate, 0.0, r), degenerate


def sign_agreement(pred, truth, mask) -> np.ndarray:
    return np.where(mask, np.sign(pred) == np.sign(truth), False).sum(
        axis=1
    ) / mask.sum(axis=1)


def masked_mse(pred, truth, mask) -> np.ndarray:
    return np.where(mask, (pred - truth) ** 2, 0.0).sum(axis=1) / mask.sum(axis=1)


def distance_matrix(
    pred: np.ndarray, truth: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """D[p, q] = squared distance between pred_p and truth_q over p's kept genes."""
    k = keep.astype(float)
    pk = pred * k
    return (pk * pred).sum(1)[:, None] - 2 * pk @ truth.T + k @ (truth**2).T


def centroid_acc(d: np.ndarray) -> np.ndarray:
    """1 if a perturbation's own truth is the nearest; ties share credit (expected accuracy)."""
    near = np.isclose(d, d.min(axis=1, keepdims=True), rtol=1e-12, atol=1e-12)
    return np.diag(near) / near.sum(axis=1)


def pds(d: np.ndarray) -> np.ndarray:
    """Normalized rank of the own distance among all candidates: 0 best, ~0.5 chance."""
    from scipy.stats import rankdata

    n = d.shape[0]
    if n < 2:
        return np.full(n, np.nan)
    own = rankdata(d, method="average", axis=1)[np.arange(n), np.arange(n)]
    return (own - 1) / (n - 1)


def score(
    pred: np.ndarray,
    truth: np.ndarray,
    mu: np.ndarray,
    keep: np.ndarray,
    top_k: int,
    degenerate_std: float,
) -> pd.DataFrame:
    """BUILD_GUIDE F1 metrics for one set of perturbations scored together (one fold/subset).
    Ranking metrics (centroid_acc, pds) compare against the other rows passed in."""
    pred, truth = pred.astype(np.float64), truth.astype(np.float64)
    ps, ts = pred - mu, truth - mu
    top_raw, top_spec = topk_mask(truth, keep, top_k), topk_mask(ts, keep, top_k)
    raw_r, deg_raw = pearson_or_zero(pred, truth, keep, degenerate_std)
    raw_r20, _ = pearson_or_zero(pred, truth, top_raw, degenerate_std)
    pspec_r, deg_spec = pearson_or_zero(ps, ts, keep, degenerate_std)
    pspec_r20, _ = pearson_or_zero(ps, ts, top_spec, degenerate_std)
    d = distance_matrix(pred, truth, keep)
    return pd.DataFrame(
        {
            "raw_r": raw_r,
            "raw_r20": raw_r20,
            "mse": masked_mse(pred, truth, keep),
            "mse20": masked_mse(pred, truth, top_raw),
            "pspec_r": pspec_r,
            "pspec_r20": pspec_r20,
            "dir20": sign_agreement(pred, truth, top_raw),
            "pspec_dir20": sign_agreement(ps, ts, top_spec),
            "centroid_acc": centroid_acc(d),
            "pds": pds(d),
            "degenerate_raw": deg_raw,
            "degenerate_pspec": deg_spec,
            "n_eval_genes": keep.sum(axis=1),
        }
    )
