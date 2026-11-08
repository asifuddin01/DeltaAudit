import json

import numpy as np
import pandas as pd
import pytest

from deltaaudit.data.leakage import (
    LeakageError,
    check_folds,
    check_split,
    fit_rows,
    genes_of,
    verify_split_files,
)


def _split(train, val, test, split_type="unseen_single", fold=0):
    return {"dataset": "toy", "split_type": split_type, "fold": fold, "n_folds": 2, "seed": 0,
            "train": train, "val": val, "test": test}  # fmt: skip


def test_genes_of_counts_both_genes_of_a_double():
    assert genes_of(["A", "B+C", "ctrl"]) == {"A", "B", "C"}


def test_clean_split_passes():
    assert check_split(_split(["A", "B"], ["C"], ["D"])) == []


def test_double_in_train_sharing_a_gene_with_test_is_caught():
    problems = check_split(_split(["A+D", "B"], ["C"], ["D"]))
    assert any("gene overlap" in p for p in problems)


def test_val_counts_as_training_for_gene_overlap():
    assert any("gene overlap" in p for p in check_split(_split(["A"], ["D+E"], ["D"])))


def test_ctrl_anywhere_is_caught():
    assert any("ctrl" in p for p in check_split(_split(["A"], [], ["ctrl"])))
    assert any("ctrl" in p for p in check_split(_split(["A", "ctrl"], [], ["B"])))


def test_same_condition_in_two_lists_and_duplicates_are_caught():
    assert any("overlap" in p for p in check_split(_split(["A", "B"], [], ["B"])))
    assert any("duplicate" in p for p in check_split(_split(["A", "A"], [], ["B"])))


def test_guide_overlap_is_caught_when_guides_exist():
    meta = pd.DataFrame({"condition": ["A", "B"], "guides": [["g1"], ["g1"]]})
    assert any(
        "guide overlap" in p for p in check_split(_split(["A"], [], ["B"]), meta)
    )
    meta["guides"] = [["g1"], ["g2"]]
    assert check_split(_split(["A"], [], ["B"]), meta) == []


def test_seen_2of2_requires_both_singles_in_train():
    ok = _split(["A", "B"], [], ["A+B"], split_type="seen_2of2")
    bad = _split(["A"], [], ["A+B"], split_type="seen_2of2")
    assert check_split(ok) == []
    assert any("unseen single" in p for p in check_split(bad))


def test_check_folds_catches_gaps_and_double_testing():
    f0 = _split(["C", "D"], [], ["A", "B"], fold=0)
    f1 = _split(["A", "B"], [], ["C", "D"], fold=1)
    assert check_folds([f0, f1]) == []
    gap = _split(["A", "B"], [], ["C"], fold=1)
    assert any("partition" in p or "universe" in p for p in check_folds([f0, gap]))
    twice = _split(["C", "D"], [], ["A", "B"], fold=1)
    assert any("more than one" in p for p in check_folds([f0, twice]))
    meta = pd.DataFrame({"condition": ["A", "B", "C", "D", "E"], "is_double": False})
    assert any("meta" in p for p in check_folds([f0, f1], meta))


def test_fit_rows_ignores_corrupted_test_rows():
    """Rule 3 / D2.4: anything fitted sees train (+val) rows only."""
    df = pd.DataFrame(np.arange(12.0).reshape(4, 3), index=["A", "B", "C", "D"])
    s = _split(["A", "B"], ["C"], ["D"])
    clean = fit_rows(df, s).mean()
    corrupted = df.copy()
    corrupted.loc["D"] = np.nan
    np.testing.assert_array_equal(fit_rows(corrupted, s).mean(), clean)
    assert list(fit_rows(df, s, include_val=False).index) == ["A", "B"]


def test_fit_rows_refuses_a_leaky_split():
    df = pd.DataFrame(np.zeros((2, 2)), index=["A", "A+B"])
    with pytest.raises(LeakageError):
        fit_rows(df, _split(["A+B"], [], ["A"]))


def test_verify_split_files_on_disk(tmp_path):
    d = tmp_path / "data" / "splits" / "toy"
    d.mkdir(parents=True)
    for f in (
        _split(["C", "D"], [], ["A", "B"], fold=0),
        _split(["A", "B"], [], ["C", "D"], fold=1),
    ):
        (d / f"fold_{f['fold']}.json").write_text(json.dumps(f))
    assert verify_split_files(tmp_path) == []
    leaky = _split(["C+A", "D"], [], ["A", "B"], fold=0)
    (d / "fold_0.json").write_text(json.dumps(leaky))
    assert verify_split_files(tmp_path) != []
