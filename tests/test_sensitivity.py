"""Phase 9: HVG re-selection on control + training cells, mu_all rescoring, verdicts."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from conftest import EVAL, MODELS, ROOT, fold0  # noqa: E402

from deltaaudit.data.hvg import (
    cell_conditions,
    derive,
    dispersion_score,
    gene_moments,
    select,
)
from deltaaudit.data.leakage import fit_rows
from deltaaudit.data.pseudobulk import ctrl_embedding
from deltaaudit.eval.sensitivity import rescore, verdict
from deltaaudit.models.baselines import run_baselines
from deltaaudit.pipeline import PROCESSED, evaluate_split, load_dataset, load_splits

H5AD = ROOT / "tests/fixtures/tiny.h5ad"


def test_gene_moments_match_dense():
    a = ad.read_h5ad(H5AD, backed="r")
    try:
        cond = cell_conditions(a)
        rng = np.random.default_rng(0)
        masks = {"all": np.ones(a.n_obs, bool), "some": rng.random(a.n_obs) < 0.3}
        mom, xtx, csum = gene_moments(a, masks, cond == "ctrl", chunk_cells=333)
        x = np.asarray(a.X[:].todense(), dtype=np.float64)
    finally:
        a.file.close()
    for k, m in masks.items():
        n, s1, s2 = mom[k]
        e = np.expm1(x[m])
        assert n == m.sum()
        np.testing.assert_allclose(s1, e.sum(0), rtol=1e-10)
        np.testing.assert_allclose(s2, (e**2).sum(0), rtol=1e-10)
    xc = x[cond == "ctrl"]
    np.testing.assert_allclose(xtx, xc.T @ xc, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(csum, xc.sum(0), rtol=1e-10)


def test_dispersion_ranks_by_var_over_mean_within_bin():
    n = 100
    mean = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    var = np.array([1.0, 4.0, 2.0, 3.0, 0.0])
    s1, s2 = mean * n, (var * (n - 1) + n * mean**2)
    score = dispersion_score(n, s1, s2, n_bins=1)
    assert list(np.argsort(-np.nan_to_num(score, nan=-np.inf))[:4]) == [1, 3, 2, 0]
    keep = select(score, 2)
    assert keep.tolist() == [False, True, False, True, False]


def test_derive_subsets_columns_and_recomputes_panel(tiny_root, tmp_path):
    base = tiny_root / PROCESSED / "tiny"
    genes = pd.read_parquet(base / "genes.parquet")
    keep = np.zeros(len(genes), bool)
    keep[::2] = True  # G000, G002, ...: odd-numbered targets leave the panel
    xtx, csum = np.eye(len(genes)), np.ones(len(genes))
    derive(base, tmp_path, keep, xtx, csum, n_ctrl=10, k=4)
    d = pd.read_parquet(tmp_path / "delta.parquet")
    full = pd.read_parquet(base / "delta.parquet")
    assert list(d.columns) == list(genes.ensembl_id[keep])
    pd.testing.assert_frame_equal(d, full[d.columns])
    meta = pd.read_parquet(tmp_path / "meta.parquet").set_index("condition")
    assert meta.at["G002", "target_in_expression_genes"]
    assert not meta.at["G003", "target_in_expression_genes"]
    emb = pd.read_parquet(tmp_path / "ctrl_embed.parquet")
    np.testing.assert_allclose(
        emb.to_numpy(),
        ctrl_embedding(xtx[np.ix_(keep, keep)], csum[keep], 10, 4),
        atol=1e-6,
    )
    for f in ("pb_mean", "pb_half_A", "pb_half_B", "genes", "descriptive"):
        assert (tmp_path / f"{f}.parquet").exists()


def test_evaluate_split_mu_override(tiny_root):
    data, split = load_dataset(tiny_root, "tiny"), fold0(tiny_root)
    pred = np.tile(
        fit_rows(data["delta"], split).mean(0).to_numpy(), (len(split["test"]), 1)
    )
    fb = np.zeros(len(split["test"]), bool)
    a = evaluate_split(pred, fb, split, data, EVAL)
    b = evaluate_split(pred, fb, split, data, EVAL, mu=data["delta"].mean(0).to_numpy())
    np.testing.assert_allclose(a.raw_r, b.raw_r)
    assert (a.pspec_r == 0).all() and not (
        b.pspec_r == 0
    ).all()  # B1 is no longer degenerate


def test_rescore_with_fold_mu_reproduces_stored_scores(tiny_root):
    data, splits = load_dataset(tiny_root, "tiny"), load_splits(tiny_root, "tiny")
    stored = run_baselines(tiny_root, data, splits, EVAL, MODELS, log=lambda _: None)
    got = rescore(tiny_root, data, splits, EVAL, mu=None)
    got = got[got.model.isin(stored.model.unique())]
    k = ["model", "split_type", "fold", "analysis_set", "condition"]
    m = stored.merge(got, on=k, suffixes=("", "_re"))
    assert len(m) == len(stored)
    for c in ("raw_r", "pspec_r", "pspec_r20", "pds"):
        np.testing.assert_allclose(m[c], m[f"{c}_re"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "obs,p,want",
    [(0.1, 0.01, "+"), (-0.1, 0.01, "-"), (0.1, 0.2, "0"), (0.0, 1.0, "0")],
)
def test_verdict(obs, p, want):
    assert verdict(obs, p, 0.05) == want


def test_why():
    from deltaaudit.eval.sensitivity import why

    assert why("+", "0", 0.1, 0.05) == "becomes significant"
    assert why("0", "+", 0.02, 0.1) == "same direction, no longer significant"
    assert why("0", "+", -0.02, 0.1) == "direction reversed, not significant"
    assert why("-", "+", -0.1, 0.1) == "direction reversed"
