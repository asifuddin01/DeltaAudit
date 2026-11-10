"""Paired bootstrap, Holm, cluster bootstrap, Spearman CI and GO clusters (prereg 6, 8)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from deltaaudit.eval.stats import (
    cluster_test,
    go_clusters,
    holm,
    paired_test,
    spearman_ci,
)

B, SEED = 2000, 20261109


def test_paired_test_known_answer():
    d = np.r_[np.full(30, 0.2), np.full(10, -0.1)]
    t = paired_test(d, B, SEED, 0.95)
    assert t["n"] == 40 and t["obs"] == pytest.approx(0.125)
    assert t["median_d"] == pytest.approx(0.2) and t["frac_pos"] == pytest.approx(0.75)
    assert t["ci_lo"] < t["obs"] < t["ci_hi"] and t["ci_lo"] > 0
    assert t["p_raw"] == pytest.approx(
        1 / (B + 1)
    )  # no null-centred resample reaches |obs|
    assert t["p_wilcoxon"] < 1e-3


def test_paired_test_null_and_all_zero():
    rng = np.random.default_rng(0)
    t = paired_test(rng.normal(0, 1, 200), B, SEED, 0.95)
    assert t["ci_lo"] < 0 < t["ci_hi"] and t["p_raw"] > 0.05
    z = paired_test(
        np.zeros(20), B, SEED, 0.95
    )  # e.g. B2 vs B1 where B2 fell back to mu
    assert z["obs"] == 0 and z["p_raw"] == 1.0 and np.isnan(z["p_wilcoxon"])


def test_p_value_formula_matches_prereg():
    d = np.random.default_rng(1).normal(0.1, 1, 50)
    t = paired_test(d, B, SEED, 0.95)
    idx = np.random.default_rng(SEED).integers(0, 50, size=(B, 50))
    means = d[idx].mean(axis=1)
    p = (1 + (np.abs(means - d.mean()) >= abs(d.mean())).sum()) / (B + 1)
    assert t["p_raw"] == pytest.approx(p)


def test_holm_known_answer():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm(np.array([0.5, 0.9])), [1.0, 1.0])


def test_cluster_test_one_cluster_per_item_equals_paired_ci_width():
    d = np.random.default_rng(2).normal(0.3, 1, 100)
    a = paired_test(d, B, SEED, 0.95)
    c = cluster_test(d, np.arange(100), B, SEED, 0.95)
    assert c["obs"] == pytest.approx(a["obs"]) and c["n_clusters"] == 100
    assert c["ci_hi"] - c["ci_lo"] == pytest.approx(a["ci_hi"] - a["ci_lo"], rel=0.15)


def test_cluster_test_wider_when_clusters_are_correlated():
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(10), 20)
    d = np.repeat(rng.normal(0, 1, 10), 20) + rng.normal(0, 0.05, 200)
    a, c = paired_test(d, B, SEED, 0.95), cluster_test(d, labels, B, SEED, 0.95)
    assert (c["ci_hi"] - c["ci_lo"]) > 3 * (a["ci_hi"] - a["ci_lo"])


def test_spearman_ci():
    rng = np.random.default_rng(4)
    x = rng.normal(size=300)
    y = x + rng.normal(0, 0.5, 300)
    rho, lo, hi = spearman_ci(x, y, B, SEED, 0.95)
    assert rho == pytest.approx(spearmanr(x, y).statistic) and lo < rho < hi


def test_go_clusters_cut_on_one_minus_jaccard():
    sets = [{"a", "b"}, {"a", "b", "c"}, {"x", "y"}, {"x", "y"}, set(), set()]
    lab = go_clusters(sets, cut_height=0.8)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2]
    assert len({lab[4], lab[5], lab[0], lab[2]}) == 4  # unannotated -> singletons
