"""Dataset card: cells, perturbations, genes, controls and knockdown efficiency.

Reads the h5ad backed and in row chunks (CLAUDE.md rule 9); never densifies the full matrix.
Descriptive only (no model, no split), so it is safe before pre-registration.
"""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

CTRL = "ctrl"
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def parse_condition(cond: str) -> list[str]:
    """GEARS condition string -> target genes. 'A+ctrl' -> [A]; 'A+B' -> [A, B]; 'ctrl' -> []."""
    return [g for g in cond.split("+") if g != CTRL]


def canonical_condition(cond: str) -> str:
    """One name per perturbation: GEARS lists 'A+ctrl' and 'ctrl+A' (and 'A+B'/'B+A') apart."""
    return "+".join(sorted(parse_condition(cond))) or CTRL


def _expm1_rows(x) -> sp.csr_matrix:
    # csr_matrix() accepts both backed sparse slices and dense chunks
    x = sp.csr_matrix(x, dtype=np.float32)
    x.data = np.expm1(x.data)
    return x


def scan_h5ad(
    path: str, chunk_cells: int, pseudocount: float
) -> tuple[pd.DataFrame, dict]:
    """Per-condition table (n_cells, targets, knockdown) plus a few whole-file facts.

    Knockdown: log2((mean_pert + pc) / (mean_ctrl + pc)) of the target gene's normalized
    expression (expm1 of the stored log1p values), for targets inside the expression panel.
    """
    adata = ad.read_h5ad(path, backed="r")
    try:
        genes = (
            adata.var["gene_name"].astype(str).to_numpy()
            if "gene_name" in adata.var
            else adata.var_names.to_numpy()
        )
        col_of = {g: i for i, g in enumerate(genes)}
        raw_cond = adata.obs["condition"].astype(str)
        n_raw_conditions = raw_cond.nunique()
        cond = raw_cond.map(
            {c: canonical_condition(c) for c in raw_cond.unique()}
        ).to_numpy()
        conds, codes = np.unique(cond, return_inverse=True)
        n_cells = np.bincount(codes, minlength=len(conds))
        ctrl_mask = cond == CTRL
        n_ctrl = int(ctrl_mask.sum())

        # One "pair" per (condition, in-panel target gene); one entry per (cell, pair).
        pair_cond, pair_col, pair_gene = [], [], []
        for ci, c in enumerate(conds):
            for g in parse_condition(c):
                if g in col_of:
                    pair_cond.append(ci)
                    pair_col.append(col_of[g])
                    pair_gene.append(g)
        pair_cond = np.array(pair_cond, dtype=np.int64)
        pair_col = np.array(pair_col, dtype=np.int64)
        cells_of = np.split(np.argsort(codes, kind="stable"), np.cumsum(n_cells)[:-1])
        entry_cell = np.concatenate(
            [cells_of[ci] for ci in pair_cond] or [np.zeros(0, int)]
        )
        entry_pair = np.repeat(np.arange(len(pair_cond)), n_cells[pair_cond])
        order = np.argsort(entry_cell, kind="stable")
        entry_cell, entry_pair = entry_cell[order], entry_pair[order]

        pair_sum = np.zeros(len(pair_cond))
        ctrl_sum = np.zeros(len(genes))
        x_max, x_is_int, x_kind = 0.0, True, "empty"
        for s in range(0, adata.n_obs, chunk_cells):
            e = min(s + chunk_cells, adata.n_obs)
            chunk = adata.X[s:e]
            if s == 0:
                x_kind = "sparse" if sp.issparse(chunk) else "dense"
            raw = sp.csr_matrix(chunk)
            del chunk
            if raw.nnz:
                x_max = max(x_max, float(raw.data.max()))
                x_is_int = x_is_int and bool(np.all(raw.data == np.round(raw.data)))
            x = _expm1_rows(raw)
            del raw
            ctrl_sum += np.asarray(x[ctrl_mask[s:e]].sum(axis=0)).ravel()
            lo, hi = np.searchsorted(entry_cell, [s, e])
            if hi > lo:
                vals = np.asarray(
                    x[entry_cell[lo:hi] - s, pair_col[entry_pair[lo:hi]]]
                ).ravel()
                np.add.at(pair_sum, entry_pair[lo:hi], vals)
        obs_cols = list(adata.obs.columns)
        n_obs, n_vars = adata.n_obs, adata.n_vars
    finally:
        adata.file.close()

    ctrl_mean = ctrl_sum / max(n_ctrl, 1)
    pert_mean = pair_sum / n_cells[pair_cond]
    log2fc = np.log2((pert_mean + pseudocount) / (ctrl_mean[pair_col] + pseudocount))

    kd = pd.DataFrame(
        {
            "condition": conds[pair_cond],
            "gene": pair_gene,
            "log2fc": log2fc,
            "target_ctrl_mean": ctrl_mean[pair_col],
        }
    )
    per = pd.DataFrame({"condition": conds, "n_cells": n_cells})
    per["targets"] = [parse_condition(c) for c in conds]
    per["is_double"] = per["targets"].str.len() == 2
    per["all_targets_in_panel"] = [all(g in col_of for g in t) for t in per["targets"]]
    per = per.merge(
        kd.groupby("condition")["log2fc"].mean().rename("target_knockdown_log2fc"),
        on="condition",
        how="left",
    )
    facts = {
        "n_cells": int(n_obs),
        "n_genes": int(n_vars),
        "n_ctrl_cells": n_ctrl,
        "n_raw_condition_labels": int(n_raw_conditions),
        "x_storage": x_kind,
        "x_max": x_max,
        "x_integer_valued": x_is_int,
        "obs_columns": obs_cols,
        "guide_columns": [
            c for c in obs_cols if "guide" in c.lower() or "sgrna" in c.lower()
        ],
    }
    return per[per["condition"] != CTRL].reset_index(drop=True), facts


def _quantiles(s: pd.Series) -> dict:
    return {f"q{int(p * 100):02d}": float(s.quantile(p)) for p in QUANTILES}


def _fmt(d: dict) -> str:
    return ", ".join(f"{k} {v:.3g}" for k, v in d.items())


def summarize(per: pd.DataFrame, facts: dict, min_cells: int) -> dict:
    singles = per[~per["is_double"]]
    kd = singles["target_knockdown_log2fc"].dropna()
    return {
        **facts,
        "n_perturbations": len(per),
        "n_single": int((~per["is_double"]).sum()),
        "n_double": int(per["is_double"].sum()),
        "n_unique_target_genes": len({g for t in per["targets"] for g in t}),
        "n_perturbations_all_targets_in_panel": int(per["all_targets_in_panel"].sum()),
        "min_cells_threshold": min_cells,
        "n_perturbations_below_min_cells": int((per["n_cells"] < min_cells).sum()),
        "cells_per_perturbation": _quantiles(per["n_cells"]),
        "single_knockdown_n": int(len(kd)),
        "single_knockdown_log2fc": _quantiles(kd) if len(kd) else {},
        "single_frac_log2fc_below_minus1": float((kd < -1).mean()) if len(kd) else None,
        "single_frac_log2fc_above_plus1": float((kd > 1).mean()) if len(kd) else None,
    }


def render_card(cfg: dict, s: dict) -> str:
    rows = [
        ("Modality", cfg.get("modality", "")),
        ("Cells", f"{s['n_cells']:,}"),
        ("Control cells", f"{s['n_ctrl_cells']:,}"),
        ("Genes (expression panel)", f"{s['n_genes']:,}"),
        ("Perturbations (excl. ctrl, order-merged)", f"{s['n_perturbations']:,}"),
        ("Raw GEARS condition labels (incl. ctrl)", f"{s['n_raw_condition_labels']:,}"),
        ("  single / double", f"{s['n_single']:,} / {s['n_double']:,}"),
        ("Distinct target genes", f"{s['n_unique_target_genes']:,}"),
        (
            "Perturbations with all targets in panel",
            f"{s['n_perturbations_all_targets_in_panel']:,}",
        ),
        (
            f"Perturbations with < {s['min_cells_threshold']} cells (dropped in Phase 2)",
            f"{s['n_perturbations_below_min_cells']:,}",
        ),
        ("Cells per perturbation (quantiles)", _fmt(s["cells_per_perturbation"])),
        (
            "Single-target log2FC of target, in-panel (n)",
            f"{s['single_knockdown_n']:,}",
        ),
        (
            "  quantiles",
            (
                _fmt(s["single_knockdown_log2fc"])
                if s["single_knockdown_log2fc"]
                else "n/a"
            ),
        ),
        (
            "  fraction log2FC < -1 / > +1",
            (
                f"{s['single_frac_log2fc_below_minus1']:.3f} / {s['single_frac_log2fc_above_plus1']:.3f}"
                if s["single_knockdown_n"]
                else "n/a"
            ),
        ),
        (
            "X storage / max value / integer-valued",
            f"{s['x_storage']} / {s['x_max']:.3g} / {s['x_integer_valued']}",
        ),
        ("obs columns", ", ".join(s["obs_columns"])),
        ("Guide id columns", ", ".join(s["guide_columns"]) or "none"),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (
        f"# Dataset card: {cfg['name']}\n\n"
        f"_Generated by `deltaaudit card {cfg['name']}`; do not edit. "
        f"Machine-readable copy: `{cfg['name']}.json`._\n\n"
        f"{cfg.get('description', '')}\n\n"
        f"| Field | Value |\n|---|---|\n{table}\n\n"
        f"**Guides.** {cfg.get('guides_note', '')}\n\n"
        "**Knockdown** is log2((mean_pert + pc) / (mean_ctrl + pc)) of the target gene's "
        f"normalized expression, pc = {cfg['qc']['knockdown_pseudocount']}. Weak knockdowns are "
        "reported here, not dropped.\n"
    )
