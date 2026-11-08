"""Per-perturbation metrics. The target gene is excluded from every metric (CLAUDE.md rule 6):
callers pass `keep`, a perturbations x genes boolean mask that is False on target columns.
"""

import numpy as np


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
