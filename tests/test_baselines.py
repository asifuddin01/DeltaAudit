import json

import numpy as np
import pandas as pd
import pytest
from conftest import EVAL, MODELS, fold0  # noqa: E402

from deltaaudit.data.leakage import LeakageError, fit_rows, require_prereg
from deltaaudit.models.baselines import predict_b1, registry, run_baselines
from deltaaudit.pipeline import fit_mu, load_dataset, load_splits, summarize


@pytest.fixture(scope="module")
def scores(tiny_root):
    data, splits = load_dataset(tiny_root, "tiny"), load_splits(tiny_root, "tiny")
    all_ = run_baselines(tiny_root, data, splits, EVAL, MODELS, log=lambda _: None)
    return {m: g for m, g in all_.groupby("model")}


def test_every_single_tested_once_and_doubles_scored(scores):
    s = scores["B1"]
    singles = s[(s.split_type == "unseen_single") & (s.analysis_set == "all")]
    assert singles["condition"].is_unique and len(singles) == 44
    assert len(s[s.analysis_set == "common_support"]) == 40
    assert len(s[s.analysis_set == "doubles"]) == 6


def test_b1_is_degenerate_on_pspec_and_b0_on_raw(scores):
    assert (scores["B1"]["pspec_r"] == 0).all() and scores["B1"][
        "degenerate_pspec"
    ].all()
    assert (scores["B0"]["raw_r"] == 0).all() and scores["B0"]["degenerate_raw"].all()
    # The tiny data has a strong shared shift, so the mean beats no-change on raw_r.
    assert scores["B1"]["raw_r"].mean() > 0.3


def test_results_written_with_manifest(tiny_root, scores):
    d = tiny_root / "results/tiny/B1/fold_0"
    m = json.loads((d / "MANIFEST.json").read_text())
    assert np.load(d / "pred.npy").shape == (len(m["test_rows"]), 200)
    assert (d / "scores.parquet").exists() and "code_hash" in m


def test_b1_ignores_corrupted_test_rows(tiny_root):
    """Rule 3: corrupting test rows with NaN must not change what is fitted."""
    data = load_dataset(tiny_root, "tiny")
    split = fold0(tiny_root)
    clean = fit_mu(data["delta"], split)
    data["delta"].loc[split["test"]] = np.nan
    np.testing.assert_array_equal(fit_mu(data["delta"], split), clean)
    pred = predict_b1(data, split, fit_rows(data["delta"], split))
    assert np.isfinite(pred.to_numpy()).all()


def test_summary_has_cis(scores):
    stats = {**EVAL["statistics"], "bootstrap_B": 500}
    t = summarize(pd.concat(scores.values()), ["raw_r", "pspec_r"], stats)
    pooled = t[(t.fold == "pooled") & (t.metric == "raw_r")]
    assert ((pooled.ci_lo <= pooled["mean"]) & (pooled["mean"] <= pooled.ci_hi)).all()
    assert set(t.fold) == {"pooled", "0", "1", "2", "3", "4"}


def test_require_prereg_refuses_without_tag(tmp_path):
    with pytest.raises(LeakageError):
        require_prereg(tmp_path)


def test_registry_runs_each_model_on_its_splits(scores):
    assert set(scores) == {"B0", "B1", "B2", "B2-GO", "B3", "B4"}
    assert set(scores["B4"].split_type) == {"seen_2of2"}
    for m in ("B2", "B2-GO", "B3"):
        assert set(scores[m].split_type) == {"unseen_single"}


def test_b2_predicts_in_panel_and_falls_back_out_of_panel(scores):
    s = scores["B2"][lambda d: d.analysis_set == "all"]
    out = s.condition.str.startswith("X")
    assert (
        s.loc[out, "fallback_to_mu"].all() and not s.loc[~out, "fallback_to_mu"].any()
    )
    assert not s.loc[~out, "degenerate_pspec"].any()


def test_b2_go_and_b3_fall_back_only_without_go_neighbours(scores):
    for m in ("B2-GO", "B3"):
        s = scores[m][lambda d: d.analysis_set == "all"]
        assert set(s.loc[s.fallback_to_mu, "condition"]) == {"X03"}


def test_b2_selection_recorded_in_manifest(tiny_root, scores):
    m = json.loads((tiny_root / "results/tiny/B2/fold_0/MANIFEST.json").read_text())
    info = m["fit_info"]
    assert (
        info["K"] in MODELS["B2"]["K_grid"]
        and info["ridge"] in MODELS["B2"]["ridge_grid"]
    )
    assert info["K_skipped"] == [
        50
    ]  # tiny has ~36 training conditions: K=50 is not identifiable
    assert len(info["val_pspec_r"]) == 12 and info["n_val_scored"] > 0
    assert info["val_pspec_r"][f"K={info['K']},ridge={info['ridge']:g}"] == max(
        info["val_pspec_r"].values()
    )


@pytest.mark.parametrize("model", ["B2", "B2-GO", "B3"])
def test_fitted_models_ignore_corrupted_test_rows(tiny_root, model):
    """Rule 3: NaN in test rows (delta and expression) must not change any prediction."""
    predict = registry(MODELS, EVAL)[model][0]
    split = fold0(tiny_root)
    clean_data = load_dataset(tiny_root, "tiny")
    clean = predict(clean_data, split, fit_rows(clean_data["delta"], split))
    data = load_dataset(tiny_root, "tiny")
    data["delta"].loc[split["test"]] = np.nan
    data["pb_mean"].loc[split["test"]] = np.nan
    got = predict(data, split, fit_rows(data["delta"], split))
    assert np.isfinite(got.to_numpy()).all()
    pd.testing.assert_frame_equal(got, clean)


def test_b4_is_sum_of_singles(tiny_root):
    data = load_dataset(tiny_root, "tiny")
    split = next(
        s for s in load_splits(tiny_root, "tiny") if s["split_type"] == "seen_2of2"
    )
    pred = registry(MODELS, EVAL)["B4"][0](data, split, fit_rows(data["delta"], split))
    d = data["delta"]
    np.testing.assert_allclose(pred.loc["G002+G003"], d.loc["G002"] + d.loc["G003"])
