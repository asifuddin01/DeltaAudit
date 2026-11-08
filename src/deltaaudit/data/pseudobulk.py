"""Backed, chunked pseudobulk + seeded half-splits + control-only gene embedding (Phase 2).

One pass over the h5ad in row chunks (CLAUDE.md rule 9). Per chunk, a sparse
condition-indicator matrix times X accumulates per-condition sums for half A and half B;
control rows also accumulate X^T X for the control gene-gene correlation.
Outputs follow BUILD_GUIDE C4; columns are Ensembl ids (symbols are not unique, e.g. TBCE).
"""

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from deltaaudit.data.card import CTRL, canonical_condition, parse_condition, scan_h5ad
from deltaaudit.manifest import build_manifest


def half_assignment(codes: np.ndarray, seed: int) -> np.ndarray:
    """True = half A. Each condition's cells are shuffled; the first floor(n/2) go to A."""
    rng = np.random.default_rng(seed)
    is_a = np.zeros(len(codes), dtype=bool)
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes))[:-1]
    for group in np.split(order, bounds):
        is_a[rng.permutation(group)[: len(group) // 2]] = True
    return is_a


def chunked_sums(adata, codes, is_a, ctrl_mask, n_conds, chunk_cells):
    """Per-condition sums for halves A/B, plus control X^T X and column sums. float64 accumulators."""
    g = adata.n_vars
    sum_a, sum_b = np.zeros((n_conds, g)), np.zeros((n_conds, g))
    xtx, csum = np.zeros((g, g)), np.zeros(g)
    for s in range(0, adata.n_obs, chunk_cells):
        e = min(s + chunk_cells, adata.n_obs)
        x = sp.csr_matrix(adata.X[s:e], dtype=np.float64)
        c, a = codes[s:e], is_a[s:e]
        rows = np.arange(e - s)
        for target, sel in ((sum_a, a), (sum_b, ~a)):
            ind = sp.csr_matrix(
                (np.ones(sel.sum()), (c[sel], rows[sel])), shape=(n_conds, e - s)
            )
            target += (ind @ x).toarray()
        xc = x[ctrl_mask[s:e]]
        xtx += (xc.T @ xc).toarray()
        csum += np.asarray(xc.sum(axis=0)).ravel()
        del x, xc
    return sum_a, sum_b, xtx, csum


def ctrl_embedding(xtx: np.ndarray, csum: np.ndarray, n: int, k: int) -> np.ndarray:
    """Top-k PCs of the control-cell gene-gene correlation, scaled by sqrt(eigenvalue).
    Genes with zero variance in control get a zero row. Signs fixed for determinism."""
    mean = csum / n
    cov = xtx / n - np.outer(mean, mean)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    ok = sd > 0
    corr = np.zeros_like(cov)
    corr[np.ix_(ok, ok)] = cov[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
    vals, vecs = np.linalg.eigh(corr)
    top = np.argsort(vals)[::-1][:k]
    vals, vecs = np.clip(vals[top], 0.0, None), vecs[:, top]
    flip = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(len(top))])
    return vecs * flip * np.sqrt(vals)


def preprocess(root: Path, cfg: dict, out_dir: Path) -> dict:
    """Write pb_mean, pb_half_A/B, delta, meta, genes, ctrl_embed (.parquet), qc.json, MANIFEST.json."""
    t0 = time.time()
    pre, qc = cfg["preprocess"], cfg["qc"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = root / cfg["h5ad"]

    adata = ad.read_h5ad(path, backed="r")
    try:
        ens = adata.var_names.to_numpy().astype(str)
        symbols = (
            adata.var["gene_name"].astype(str).to_numpy()
            if "gene_name" in adata.var
            else ens
        )
        raw = adata.obs["condition"].astype(str)
        cond = raw.map({c: canonical_condition(c) for c in raw.unique()}).to_numpy()
        conds, codes = np.unique(cond, return_inverse=True)
        ctrl_mask = cond == CTRL
        is_a = half_assignment(codes, pre["half_split_seed"])
        sum_a, sum_b, xtx, csum = chunked_sums(
            adata, codes, is_a, ctrl_mask, len(conds), pre["chunk_cells"]
        )
    finally:
        adata.file.close()

    n_a = np.bincount(codes[is_a], minlength=len(conds))
    n_b = np.bincount(codes[~is_a], minlength=len(conds))
    n = n_a + n_b
    keep = (n >= qc["min_cells"]) | (conds == CTRL)

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            values[keep].astype(np.float32),
            index=pd.Index(conds[keep], name="condition"),
            columns=ens,
        )

    pb_mean = frame((sum_a + sum_b) / n[:, None])
    half_a = frame(sum_a / np.maximum(n_a, 1)[:, None])
    half_b = frame(sum_b / np.maximum(n_b, 1)[:, None])
    delta = pb_mean.drop(index=CTRL) - pb_mean.loc[CTRL]

    per, _ = scan_h5ad(str(path), pre["chunk_cells"], qc["knockdown_pseudocount"])
    in_panel = set(symbols)
    counts = pd.DataFrame(
        {"condition": conds, "n_cells": n, "n_cells_A": n_a, "n_cells_B": n_b}
    )
    meta = (
        counts[keep & (conds != CTRL)]
        .merge(
            per[["condition", "target_knockdown_log2fc"]], on="condition", how="left"
        )
        .assign(
            targets=lambda m: m["condition"].map(parse_condition),
            is_double=lambda m: m["targets"].str.len() == 2,
            target_in_expression_genes=lambda m: m["targets"].map(
                lambda t: all(g in in_panel for g in t)
            ),
        )
    )
    genes = pd.DataFrame({"ensembl_id": ens, "gene_symbol": symbols, "is_hvg": True})
    emb = ctrl_embedding(xtx, csum, int(ctrl_mask.sum()), pre["ctrl_embed_k"])
    ctrl_embed = pd.DataFrame(
        emb.astype(np.float32),
        index=pd.Index(ens, name="ensembl_id"),
        columns=[f"pc{i:02d}" for i in range(emb.shape[1])],
    )

    for name, df in {
        "pb_mean": pb_mean, "pb_half_A": half_a, "pb_half_B": half_b, "delta": delta,
        "meta": meta, "genes": genes, "ctrl_embed": ctrl_embed,
    }.items():  # fmt: skip
        df.to_parquet(out_dir / f"{name}.parquet")

    dropped = counts[~keep]
    qc_report = {
        "min_cells": qc["min_cells"],
        "n_conditions_total": int((conds != CTRL).sum()),
        "n_kept": int(len(meta)),
        "n_ctrl_cells": int(ctrl_mask.sum()),
        "dropped": [
            {
                "condition": c,
                "n_cells": int(k),
                "reason": f"n_cells < {qc['min_cells']}",
            }
            for c, k in zip(dropped["condition"], dropped["n_cells"], strict=True)
        ],
    }
    (out_dir / "qc.json").write_text(json.dumps(qc_report, indent=2) + "\n")
    rel = path.relative_to(root).as_posix()
    manifest = build_manifest(root, cfg, [rel], t0)
    manifest["shapes"] = {"pb_mean": list(pb_mean.shape), "delta": list(delta.shape)}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"qc": qc_report, "manifest": manifest}
