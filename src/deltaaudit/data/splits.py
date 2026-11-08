"""Perturbation-level CV splits (BUILD_GUIDE D1).

- unseen_single: K folds over single-gene perturbations, stratified by whether the target is in
  the expression panel; each fold holds out 10% of its training perturbations as inner validation.
  Doubles are not part of these splits (for Norman this is the strict single-gene setting).
- seen_2of2 (Norman): test = doubles whose two singles are both measured; train = all singles.
Lists are sorted condition names; split files are committed under data/splits/<dataset>/.
"""

import numpy as np
import pandas as pd

from deltaaudit.data.leakage import genes_of

STRATUM = "target_in_expression_genes"


def _fold_of(units: list[str], strata: list[bool], n_folds: int, rng) -> dict[str, int]:
    """Round-robin over a shuffled order within each stratum, continuing across strata,
    so both fold sizes and per-stratum counts differ by at most one."""
    fold_of, offset = {}, 0
    for s in sorted(set(strata)):
        members = sorted(u for u, st in zip(units, strata, strict=True) if st == s)
        for i, j in enumerate(rng.permutation(len(members))):
            fold_of[members[j]] = (i + offset) % n_folds
        offset += len(members)
    return fold_of


def unseen_single_splits(meta: pd.DataFrame, cfg: dict, dataset: str) -> list[dict]:
    singles = meta[~meta["is_double"]]
    units = singles["condition"].tolist()
    strata = singles[STRATUM].astype(bool).tolist()
    stratum_of = dict(zip(units, strata, strict=True))
    fold_of = _fold_of(
        units, strata, cfg["n_folds"], np.random.default_rng(cfg["seed"])
    )

    out = []
    for k in range(cfg["n_folds"]):
        rng = np.random.default_rng([cfg["seed"], k])
        test = sorted(u for u in units if fold_of[u] == k)
        rest = sorted(u for u in units if fold_of[u] != k)
        val = []
        for s in sorted(set(strata)):
            members = [u for u in rest if stratum_of[u] == s]
            n_val = round(cfg["val_frac"] * len(members))
            val += rng.choice(members, n_val, replace=False).tolist() if n_val else []
        val = sorted(val)
        train = sorted(set(rest) - set(val))
        out.append(
            {
                "dataset": dataset,
                "split_type": "unseen_single",
                "fold": k,
                "n_folds": cfg["n_folds"],
                "seed": cfg["seed"],
                "val_frac": cfg["val_frac"],
                "stratified_by": STRATUM,
                "train": train,
                "val": val,
                "test": test,
            }
        )
    return out


def seen_2of2_split(meta: pd.DataFrame, dataset: str) -> dict:
    singles = sorted(meta.loc[~meta["is_double"], "condition"])
    seen = genes_of(singles)
    doubles = meta.loc[meta["is_double"], "condition"]
    test = sorted(d for d in doubles if genes_of([d]) <= seen)
    return {
        "dataset": dataset,
        "split_type": "seen_2of2",
        "fold": 0,
        "n_folds": 1,
        "seed": None,
        "train": singles,
        "val": [],
        "test": test,
    }


def split_summary(splits: list[dict], meta: pd.DataFrame) -> dict:
    """Counts per split file, including the common-support (target in panel) subset of test."""
    in_panel = meta.set_index("condition")[STRATUM]
    return {
        f"{s['split_type']}_fold_{s['fold']}": {
            "n_train": len(s["train"]),
            "n_val": len(s["val"]),
            "n_test": len(s["test"]),
            "n_test_target_in_panel": int(in_panel.loc[s["test"]].sum()),
        }
        for s in splits
    }
