"""Leakage assertions (BUILD_GUIDE D2), used by tests, by `deltaaudit verify`, and by
`fit_rows`, the single gate through which fitted quantities get their training rows."""

import json
from pathlib import Path

import pandas as pd

CTRL = "ctrl"
SPLITS_DIR = Path("data/splits")


class LeakageError(Exception):
    pass


def genes_of(conditions) -> set[str]:
    """Target genes of a set of conditions; a double contributes both genes."""
    return {g for c in conditions for g in c.split("+") if g != CTRL}


def check_split(split: dict, meta: pd.DataFrame | None = None) -> list[str]:
    """Problems within one split file (empty list = clean)."""
    name = f"{split['dataset']}/{split['split_type']}/fold_{split['fold']}"
    lists = {k: split[k] for k in ("train", "val", "test")}
    problems = []
    for k, items in lists.items():
        if len(items) != len(set(items)):
            problems.append(f"{name}: duplicate conditions in {k}")
        if CTRL in items:
            problems.append(f"{name}: ctrl in {k}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if both := set(lists[a]) & set(lists[b]):
            problems.append(f"{name}: condition overlap {a}/{b}: {sorted(both)[:5]}")

    fit = lists["train"] + lists["val"]
    if split["split_type"] == "unseen_single":
        if shared := genes_of(fit) & genes_of(lists["test"]):
            problems.append(
                f"{name}: gene overlap train+val/test: {sorted(shared)[:5]}"
            )
    elif split["split_type"] == "seen_2of2":
        singles = {c for c in fit if "+" not in c}
        missing = {g for d in lists["test"] for g in d.split("+") if g not in singles}
        if missing:
            problems.append(
                f"{name}: test double has an unseen single: {sorted(missing)[:5]}"
            )
    else:
        problems.append(f"{name}: unknown split_type {split['split_type']}")

    if (
        meta is not None
        and "guides" in meta.columns
        and split["split_type"] == "unseen_single"
    ):
        guides = meta.set_index("condition")["guides"]
        g_fit = {g for c in fit for g in guides.get(c, [])}
        g_test = {g for c in lists["test"] for g in guides.get(c, [])}
        if shared := g_fit & g_test:
            problems.append(
                f"{name}: guide overlap train+val/test: {sorted(shared)[:5]}"
            )
    return problems


def check_folds(folds: list[dict], meta: pd.DataFrame | None = None) -> list[str]:
    """Across the unseen_single folds of one dataset: every unit tested exactly once, every
    fold covers the same universe, and (with meta) that universe is exactly the singles.
    """
    problems = []
    universes = [frozenset(f["train"] + f["val"] + f["test"]) for f in folds]
    universe = universes[0]
    if any(u != universe for u in universes):
        problems.append("folds do not share one universe of perturbations")
    tests = [c for f in folds for c in f["test"]]
    if len(tests) != len(set(tests)):
        problems.append("a perturbation is tested in more than one fold")
    if set(tests) != universe:
        problems.append("test folds do not partition the universe")
    if meta is not None:
        singles = set(meta.loc[~meta["is_double"].astype(bool), "condition"])
        if singles != universe:
            problems.append(
                "split universe != singles in meta (re-run `deltaaudit split`?)"
            )
    return problems


def fit_rows(df: pd.DataFrame, split: dict, include_val: bool = True) -> pd.DataFrame:
    """The only rows anything may be fitted on: train (+val). Raises on a leaky split."""
    if problems := check_split(split):
        raise LeakageError("; ".join(problems))
    rows = split["train"] + (split["val"] if include_val else [])
    return df.loc[rows]


def verify_split_files(root: Path) -> list[str]:
    """Check every committed split file, and fold sets per dataset; use meta when present."""
    problems = []
    for d in sorted(p for p in (root / SPLITS_DIR).glob("*") if p.is_dir()):
        meta_path = root / "data" / "processed" / d.name / "meta.parquet"
        meta = pd.read_parquet(meta_path) if meta_path.exists() else None
        splits = [
            json.loads(f.read_text())
            for f in sorted(d.glob("*.json"))
            if f.name != "summary.json"
        ]
        for s in splits:
            problems += check_split(s, meta)
        folds = [s for s in splits if s["split_type"] == "unseen_single"]
        if folds:
            problems += [f"{d.name}: {p}" for p in check_folds(folds, meta)]
    return problems


def require_prereg(root: Path) -> None:
    """CLAUDE.md rule 1: refuse to score test folds unless the `prereg` tag exists and HEAD
    descends from it."""
    import subprocess

    def ok(*args: str) -> bool:
        return (
            subprocess.run(["git", *args], cwd=root, capture_output=True).returncode
            == 0
        )

    if not ok("rev-parse", "-q", "--verify", "refs/tags/prereg^{commit}"):
        raise LeakageError(
            "no `prereg` tag: commit and tag reports/preregistration.md first"
        )
    if not ok("merge-base", "--is-ancestor", "prereg", "HEAD"):
        raise LeakageError("HEAD does not descend from the `prereg` tag")
