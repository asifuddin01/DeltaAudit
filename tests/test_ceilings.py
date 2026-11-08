import numpy as np
import pandas as pd
import pytest

from deltaaudit.eval.ceilings import describe
from deltaaudit.eval.metrics import rowwise_cosine, rowwise_pearson, target_mask


def test_rowwise_pearson_matches_numpy_on_kept_genes():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(5, 30)), rng.normal(size=(5, 30))
    keep = rng.random((5, 30)) > 0.2
    r = rowwise_pearson(a, b, keep)
    for i in range(5):
        k = keep[i]
        assert r[i] == pytest.approx(np.corrcoef(a[i, k], b[i, k])[0, 1])


def test_rowwise_cosine_and_constant_rows():
    a = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    keep = np.ones_like(a, dtype=bool)
    assert rowwise_cosine(a, 2 * a, keep) == pytest.approx([1.0, 1.0])
    assert np.isnan(rowwise_pearson(a, a, keep)[1])  # zero variance -> undefined


def test_target_mask_hits_every_column_with_that_symbol():
    symbols = np.array(["A", "B", "A", "C"])
    m = target_mask([["A"], ["B", "C"], []], symbols)
    assert m.tolist() == [
        [True, False, True, False],
        [False, True, False, True],
        [False, False, False, False],
    ]


def test_describe_excludes_target_gene():
    # Two perturbations; gene 0 is p1's target with a huge change that would dominate.
    genes = pd.DataFrame(
        {"ensembl_id": ["e0", "e1", "e2", "e3"], "gene_symbol": ["T", "a", "b", "c"]}
    )
    delta = pd.DataFrame(
        [[-9.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]],
        index=["T", "x"],
        columns=genes["ensembl_id"],
    )
    meta = pd.DataFrame({"condition": ["T", "x"], "targets": [["T"], ["Q"]]})
    halves = delta.copy()
    ctrl = pd.DataFrame([[0.0] * 4], index=["ctrl"], columns=genes["ensembl_id"])
    half = pd.concat([halves, ctrl])
    d = describe(delta, half, half, meta, genes).set_index("condition")
    # Without the target, both rows equal (1,2,3) on genes a-c, as does their mean -> cos = 1.
    assert d.loc["T", "sys_cos"] == pytest.approx(1.0)
    assert d.loc["T", "ceil_raw"] == pytest.approx(1.0)
    assert d.loc["T", "n_eval_genes"] == 3 and d.loc["x", "n_eval_genes"] == 4
