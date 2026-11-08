import numpy as np
import pytest

from deltaaudit.eval.metrics import distance_matrix, score
from deltaaudit.eval.stats import bootstrap_ci

TOL = 1e-8


def _data(p=6, g=40, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(p, g))
    mu = truth.mean(axis=0)
    keep = np.ones((p, g), dtype=bool)
    return truth, mu, keep


def test_perfect_prediction():
    truth, mu, keep = _data()
    s = score(truth.copy(), truth, mu, keep, top_k=20, degenerate_std=TOL)
    for m in (
        "raw_r",
        "raw_r20",
        "pspec_r",
        "pspec_r20",
        "dir20",
        "pspec_dir20",
        "centroid_acc",
    ):
        np.testing.assert_allclose(s[m], 1.0)
    np.testing.assert_allclose(s[["mse", "mse20", "pds"]].to_numpy(), 0.0, atol=1e-12)
    assert not s["degenerate_raw"].any() and not s["degenerate_pspec"].any()


def test_train_mean_prediction_is_degenerate_on_pspec():
    truth, mu, keep = _data()
    pred = np.tile(mu, (len(truth), 1))
    s = score(pred, truth, mu, keep, top_k=20, degenerate_std=TOL)
    assert (s["pspec_r"] == 0).all() and (s["pspec_r20"] == 0).all()
    assert s["degenerate_pspec"].all() and not s["degenerate_raw"].any()
    assert (s["pspec_dir20"] == 0).all()  # sign(0) never matches a nonzero sign
    assert s["raw_r"].notna().all()


def test_zero_prediction_is_degenerate_on_raw():
    truth, mu, keep = _data()
    s = score(np.zeros_like(truth), truth, mu, keep, top_k=20, degenerate_std=TOL)
    assert (s["raw_r"] == 0).all() and s["degenerate_raw"].all()
    assert (s["dir20"] == 0).all()


def test_constant_prediction_is_at_chance_on_ranking_metrics():
    truth, mu, keep = _data(p=8)
    pred = np.tile(
        np.random.default_rng(1).normal(size=truth.shape[1]), (len(truth), 1)
    )
    s = score(pred, truth, mu, keep, top_k=20, degenerate_std=TOL)
    assert s["centroid_acc"].mean() == pytest.approx(1 / 8)
    assert s["pds"].mean() == pytest.approx(0.5)


def test_target_gene_is_excluded():
    truth, mu, keep = _data()
    pred = truth.copy()
    pred[0, 3] += 100.0  # huge error on perturbation 0's target gene
    keep[0, 3] = False
    s = score(pred, truth, mu, keep, top_k=20, degenerate_std=TOL)
    assert s.loc[0, "raw_r"] == pytest.approx(1.0)
    assert s.loc[0, "mse"] == pytest.approx(0.0)
    assert s.loc[0, "pds"] == pytest.approx(0.0)


def test_top20_uses_largest_true_changes():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(1, 30)) * 0.1
    truth[0, :20] += (
        np.sign(truth[0, :20]) * 5
    )  # first 20 genes have the largest |delta|
    pred = truth.copy()
    pred[0, 20:] = rng.normal(size=10) * 3  # garbage outside the top 20
    keep = np.ones_like(truth, dtype=bool)
    s = score(pred, truth, np.zeros(30), keep, top_k=20, degenerate_std=TOL)
    assert s.loc[0, "raw_r20"] == pytest.approx(1.0)
    assert s.loc[0, "raw_r"] < 0.99


def test_pds_ties_get_average_rank():
    truth = np.array([[1.0, 0.0], [1.0, 0.0], [5.0, 5.0]])
    pred = truth.copy()
    keep = np.ones_like(truth, dtype=bool)
    s = score(pred, truth, truth.mean(0), keep, top_k=2, degenerate_std=TOL)
    # Row 0: distances [0, 0, 41]; own rank = average(1, 2) = 1.5 -> (1.5 - 1) / 2.
    assert s.loc[0, "pds"] == pytest.approx(0.25)
    assert s.loc[0, "centroid_acc"] == pytest.approx(0.5)  # tie between two candidates


def test_distance_matrix_respects_row_masks():
    pred = np.array([[1.0, 100.0], [0.0, 0.0]])
    truth = np.array([[1.0, 0.0], [0.0, 0.0]])
    keep = np.array([[True, False], [True, True]])
    d = distance_matrix(pred, truth, keep)
    assert d.tolist() == [[0.0, 1.0], [1.0, 0.0]]


def test_bootstrap_ci_is_seeded_and_sane():
    x = np.random.default_rng(3).normal(1.0, 1.0, 200)
    a = bootstrap_ci(x, n_boot=2000, seed=7, ci=0.95)
    assert a == bootstrap_ci(x, n_boot=2000, seed=7, ci=0.95)
    assert a[0] < x.mean() < a[1]
    assert bootstrap_ci(np.full(10, 0.3), n_boot=100, seed=0, ci=0.95) == pytest.approx(
        (0.3, 0.3)
    )
