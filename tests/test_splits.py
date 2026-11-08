import numpy as np
import pandas as pd
import pytest

from deltaaudit.data.leakage import check_folds, check_split, genes_of
from deltaaudit.data.splits import seen_2of2_split, unseen_single_splits

CFG = {"n_folds": 5, "val_frac": 0.1, "seed": 0}


def _meta(n_in=40, n_out=23, doubles=(("G00", "G01"), ("G02", "G03"), ("G04", "G10"))):
    singles = [f"G{i:02d}" for i in range(n_in)] + [f"X{i:02d}" for i in range(n_out)]
    rows = [
        {
            "condition": s,
            "targets": [s],
            "is_double": False,
            "target_in_expression_genes": s.startswith("G"),
        }
        for s in singles
    ]
    rows += [
        {
            "condition": "+".join(d),
            "targets": list(d),
            "is_double": True,
            "target_in_expression_genes": True,
        }
        for d in doubles
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def folds():
    return unseen_single_splits(_meta(), CFG, "toy")


def test_folds_partition_singles_exactly_once(folds):
    meta = _meta()
    singles = set(meta.loc[~meta["is_double"], "condition"])
    tests = [set(f["test"]) for f in folds]
    assert set().union(*tests) == singles
    assert sum(len(t) for t in tests) == len(singles)  # pairwise disjoint
    for f in folds:
        assert set(f["train"]) | set(f["val"]) | set(f["test"]) == singles
        assert not (set(f["train"]) & set(f["val"]))
    assert check_folds(folds, meta) == []


def test_doubles_excluded_from_single_gene_splits(folds):
    for f in folds:
        assert not any("+" in c for c in f["train"] + f["val"] + f["test"])


def test_stratified_by_panel_membership(folds):
    meta = _meta().set_index("condition")
    in_panel = [meta.loc[f["test"], "target_in_expression_genes"].sum() for f in folds]
    sizes = [len(f["test"]) for f in folds]
    assert max(in_panel) - min(in_panel) <= 1
    assert max(sizes) - min(sizes) <= 1


def test_val_is_about_ten_percent_of_training(folds):
    for f in folds:
        n_rest = len(f["train"]) + len(f["val"])
        assert abs(len(f["val"]) - CFG["val_frac"] * n_rest) <= 2


def test_seeded_and_deterministic():
    a = unseen_single_splits(_meta(), CFG, "toy")
    b = unseen_single_splits(_meta(), CFG, "toy")
    c = unseen_single_splits(_meta(), {**CFG, "seed": 1}, "toy")
    assert a == b
    assert [f["test"] for f in a] != [f["test"] for f in c]


def test_split_lists_are_sorted_and_record_settings(folds):
    f = folds[0]
    assert f["train"] == sorted(f["train"]) and f["test"] == sorted(f["test"])
    assert (f["split_type"], f["fold"], f["n_folds"], f["seed"]) == (
        "unseen_single",
        0,
        5,
        0,
    )
    for f in folds:
        assert check_split(f) == []


def test_seen_2of2_doubles_split():
    meta = _meta()
    s = seen_2of2_split(meta, "toy")
    assert sorted(s["test"]) == ["G00+G01", "G02+G03", "G04+G10"]
    assert s["val"] == [] and not any("+" in c for c in s["train"])
    assert genes_of(s["test"]) <= genes_of(s["train"])
    assert check_split(s) == []


def test_seen_2of2_skips_doubles_with_an_unseen_single():
    s = seen_2of2_split(_meta(doubles=(("G00", "NEW"),)), "toy")
    assert s["test"] == []


def test_real_world_sizes_are_balanced():
    # 1,070 singles, 406 in panel, like Replogle after QC.
    rng = np.random.default_rng(0)
    flags = np.zeros(1070, dtype=bool)
    flags[rng.choice(1070, 406, replace=False)] = True
    meta = pd.DataFrame(
        {
            "condition": [f"P{i:04d}" for i in range(1070)],
            "targets": [[f"P{i:04d}"] for i in range(1070)],
            "is_double": False,
            "target_in_expression_genes": flags,
        }
    )
    folds = unseen_single_splits(meta, CFG, "toy")
    assert sorted(len(f["test"]) for f in folds) == [214] * 5
    assert check_folds(folds, meta) == []
