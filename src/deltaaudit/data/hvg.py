"""HVG re-selection sensitivity (BUILD_GUIDE D3, prereg 8; Phase 9).

GEARS picked its 5,000-gene panel from all cells, test perturbations included. The h5ad holds
only that panel, so the check re-selects a smaller panel *inside* it twice — from all cells,
and from control + one fold's train+val cells — and compares the conclusions.
One backed, chunked pass over the h5ad (CLAUDE.md rule 9).
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from deltaaudit.data.card import canonical_condition
from deltaaudit.data.pseudobulk import ctrl_embedding
from deltaaudit.eval.ceilings import describe

TABLES = ("pb_mean", "pb_half_A", "pb_half_B", "delta")


def cell_conditions(adata) -> np.ndarray:
    raw = adata.obs["condition"].astype(str)
    return raw.map({c: canonical_condition(c) for c in raw.unique()}).to_numpy()


def gene_moments(
    adata, masks: dict[str, np.ndarray], ctrl_mask: np.ndarray, chunk_cells: int
):
    """Per mask: (n cells, sum, sum of squares) of expm1(X) per gene (Seurat dispersion works on
    the de-logged scale). Also control X^T X and column sums on log X, as in preprocess.
    """
    g = adata.n_vars
    mom = {k: [0, np.zeros(g), np.zeros(g)] for k in masks}
    xtx, csum = np.zeros((g, g)), np.zeros(g)
    for s in range(0, adata.n_obs, chunk_cells):
        e = min(s + chunk_cells, adata.n_obs)
        x = sp.csr_matrix(adata.X[s:e], dtype=np.float64)
        xe = x.copy()
        xe.data = np.expm1(xe.data)
        for k, m in masks.items():
            sub = xe[m[s:e]]
            mom[k][0] += sub.shape[0]
            mom[k][1] += np.asarray(sub.sum(axis=0)).ravel()
            mom[k][2] += np.asarray(sub.multiply(sub).sum(axis=0)).ravel()
        xc = x[ctrl_mask[s:e]]
        xtx += (xc.T @ xc).toarray()
        csum += np.asarray(xc.sum(axis=0)).ravel()
        del x, xe, xc
    return {k: tuple(v) for k, v in mom.items()}, xtx, csum


def dispersion_score(n: int, s1: np.ndarray, s2: np.ndarray, n_bins: int) -> np.ndarray:
    """Seurat-flavour normalized dispersion: log(var/mean), z-scored within bins of log1p(mean).
    Genes with zero mean get NaN (never selected); a one-gene bin scores 0."""
    mean = s1 / n
    var = (s2 - n * mean**2) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(mean > 0, np.log(np.where(mean > 0, var / mean, 1.0)), np.nan)
    df = pd.DataFrame({"d": disp, "b": pd.cut(np.log1p(mean), n_bins)})
    g = df.groupby("b", observed=True)["d"]
    sd = g.transform("std")
    return (
        ((df.d - g.transform("mean")) / sd.where(sd > 0))
        .fillna(df.d.where(df.d.isna(), 0.0))
        .to_numpy()
    )


def select(score: np.ndarray, n_top: int) -> np.ndarray:
    """Top n_top genes by score (stable on ties); NaN never selected."""
    order = np.argsort(-np.nan_to_num(score, nan=-np.inf), kind="stable")[:n_top]
    keep = np.zeros(len(score), dtype=bool)
    keep[order[np.isfinite(score[order])]] = True
    return keep


def derive(
    base: Path, out: Path, keep: np.ndarray, xtx: np.ndarray, csum: np.ndarray,
    n_ctrl: int, k: int,
) -> None:  # fmt: skip
    """A processed dataset on the gene subset `keep` (aligned with base genes.parquet):
    pseudobulk columns subset (exact, per-gene means), target-in-panel recomputed,
    control embedding recomputed from the subset's control X^T X, descriptive table."""
    out.mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(base / "genes.parquet")
    ens = genes.ensembl_id[keep].tolist()
    t = {n: pd.read_parquet(base / f"{n}.parquet")[ens] for n in TABLES}
    sub_genes = genes[keep].reset_index(drop=True)
    panel = set(sub_genes.gene_symbol)
    meta = pd.read_parquet(base / "meta.parquet")
    meta["target_in_expression_genes"] = meta.targets.map(
        lambda ts: all(g in panel for g in ts)
    )
    emb = ctrl_embedding(xtx[np.ix_(keep, keep)], csum[keep], n_ctrl, k)
    ctrl = pd.DataFrame(emb.astype(np.float32), index=pd.Index(ens, name="ensembl_id"),
                        columns=[f"pc{i:02d}" for i in range(emb.shape[1])])  # fmt: skip
    for name, df in {**t, "genes": sub_genes, "meta": meta, "ctrl_embed": ctrl}.items():
        df.to_parquet(out / f"{name}.parquet")
    desc = describe(t["delta"], t["pb_half_A"], t["pb_half_B"], meta, sub_genes)
    desc.to_parquet(out / "descriptive.parquet")


def copy_targets_graph(base: Path, out: Path, sub: str) -> None:
    """The targets-only GO graph (B2-GO, B3) does not depend on the expression panel."""
    shutil.copytree(base / sub, out / sub, dirs_exist_ok=True)
