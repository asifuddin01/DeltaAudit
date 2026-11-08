"""Bootstrap statistics over held-out perturbations (CLAUDE.md rule 5). Paired tests and
Holm come in Phase 8; this holds the shared resampling core."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)  # same n reused across models/metrics
def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(
    x: np.ndarray, n_boot: int, seed: int, ci: float
) -> tuple[float, float]:
    """Percentile CI of the mean, resampling perturbations."""
    x = np.asarray(x, dtype=float)
    means = x[bootstrap_indices(len(x), n_boot, seed)].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
