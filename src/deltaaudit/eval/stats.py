"""Bootstrap statistics over held-out perturbations (CLAUDE.md rule 5, prereg 6 and 8).

Every test resamples perturbations (or GO clusters of them), never cells or genes.
"""

from functools import lru_cache

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, wilcoxon

from deltaaudit.data.go_graph import jaccard_matrix


@lru_cache(maxsize=8)  # same n reused across models/metrics
def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _ci(means: np.ndarray, ci: float) -> tuple[float, float]:
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def bootstrap_ci(
    x: np.ndarray, n_boot: int, seed: int, ci: float
) -> tuple[float, float]:
    """Percentile CI of the mean, resampling perturbations."""
    x = np.asarray(x, dtype=float)
    return _ci(x[bootstrap_indices(len(x), n_boot, seed)].mean(axis=1), ci)


def _p_null_centred(means: np.ndarray, obs: float) -> float:
    """Two-sided p = (1 + #{|mean(d*) - obs| >= |obs|}) / (B + 1)."""
    return float((1 + (np.abs(means - obs) >= abs(obs)).sum()) / (len(means) + 1))


def paired_test(d: np.ndarray, n_boot: int, seed: int, ci: float) -> dict:
    """d_p = score_A(p) - score_B(p): mean, percentile CI, bootstrap p, Wilcoxon p, median, frac > 0."""
    d = np.asarray(d, dtype=float)
    obs = float(d.mean())
    means = d[bootstrap_indices(len(d), n_boot, seed)].mean(axis=1)
    lo, hi = _ci(means, ci)
    return {
        "n": len(d), "obs": obs, "ci_lo": lo, "ci_hi": hi,
        "p_raw": _p_null_centred(means, obs),
        "p_wilcoxon": float(wilcoxon(d).pvalue) if (d != 0).any() else np.nan,
        "median_d": float(np.median(d)), "frac_pos": float((d > 0).mean()),
    }  # fmt: skip


def cluster_test(
    d: np.ndarray, labels: np.ndarray, n_boot: int, seed: int, ci: float
) -> dict:
    """Cluster bootstrap: resample clusters with replacement; the mean is over all their members."""
    d = np.asarray(d, dtype=float)
    _, inv = np.unique(labels, return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    obs = float(d.mean())
    lo, hi = _ci(means, ci)
    return {"n_clusters": len(s), "obs": obs, "ci_lo": lo, "ci_hi": hi,
            "p_raw": _p_null_centred(means, obs)}  # fmt: skip


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m, order = len(p), np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def spearman_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, ci: float
) -> tuple[float, float, float]:
    """Spearman rho with a percentile CI from resampling perturbations (pairs)."""
    from deltaaudit.eval.metrics import rowwise_pearson

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = rowwise_pearson(
        rankdata(x)[None], rankdata(y)[None], np.ones((1, len(x)), bool)
    )
    idx = bootstrap_indices(len(x), n_boot, seed)
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    boot = rowwise_pearson(rx, ry, np.ones_like(rx, dtype=bool))
    return (float(rho[0]), *_ci(boot[np.isfinite(boot)], ci))


def go_clusters(term_sets: list[set[str]], cut_height: float) -> np.ndarray:
    """Average-linkage clusters on 1 - GO Jaccard, cut at `cut_height` (prereg 8).
    Unannotated genes have distance 1 to everything, so they end up as singletons."""
    if len(term_sets) < 2:
        return np.ones(len(term_sets), dtype=int)
    dist = 1.0 - jaccard_matrix(term_sets)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="average")
    return fcluster(z, t=cut_height, criterion="distance")
