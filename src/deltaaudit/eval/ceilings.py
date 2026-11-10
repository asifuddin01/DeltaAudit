"""Split-half noise ceilings and systematic variation (BUILD_GUIDE F2).

Descriptive, computed on all perturbations with mu = mean delta over all of them; no model,
no split, so it is safe before pre-registration. Control is split in halves too, so the two
half-deltas share no control cells (shared control noise would inflate the ceiling).
"""

import numpy as np
import pandas as pd

from deltaaudit.eval.metrics import rowwise_cosine, rowwise_pearson, target_mask

CTRL = "ctrl"


def describe(
    delta: pd.DataFrame,
    half_a: pd.DataFrame,
    half_b: pd.DataFrame,
    meta: pd.DataFrame,
    genes: pd.DataFrame,
    mu: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per perturbation: sys_cos = cos(delta_p, mu_all); ceil_raw and ceil_pspec from half-splits.
    `mu` (e.g. a fold's train+val mean) replaces mu_all in ceil_pspec when given."""
    perts = delta.index
    targets = meta.set_index("condition").loc[perts, "targets"].tolist()
    keep = ~target_mask(
        targets,
        genes.set_index("ensembl_id").loc[delta.columns, "gene_symbol"].to_numpy(),
    )
    d = delta.to_numpy(np.float64)
    mu_all = d.mean(axis=0, keepdims=True)
    mu = mu_all if mu is None else np.asarray(mu, dtype=np.float64)[None]
    da = (half_a.loc[perts] - half_a.loc[CTRL]).to_numpy(np.float64)
    db = (half_b.loc[perts] - half_b.loc[CTRL]).to_numpy(np.float64)
    return pd.DataFrame(
        {
            "condition": perts,
            "sys_cos": rowwise_cosine(d, np.broadcast_to(mu_all, d.shape), keep),
            "ceil_raw": rowwise_pearson(da, db, keep),
            "ceil_pspec": rowwise_pearson(da - mu, db - mu, keep),
            "n_eval_genes": keep.sum(axis=1),
        }
    )


def summarize(desc: pd.DataFrame, quantiles: list[float]) -> dict:
    out = {"n_perturbations": len(desc)}
    for col in ("sys_cos", "ceil_raw", "ceil_pspec"):
        s = desc[col].dropna()
        out[col] = {
            "n": int(len(s)),
            "mean": float(s.mean()),
            **{f"q{int(q * 100):02d}": float(s.quantile(q)) for q in quantiles},
        }
    return out
