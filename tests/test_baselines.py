import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from deltaaudit.data.leakage import LeakageError, require_prereg
from deltaaudit.data.pseudobulk import preprocess
from deltaaudit.data.splits import seen_2of2_split, unseen_single_splits
from deltaaudit.models.baselines import PREDICTORS, predict_b1
from deltaaudit.pipeline import fit_mu, load_dataset, load_splits, run_model, summarize

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "tests/fixtures/tiny.yaml").read_text())
EVAL = yaml.safe_load((ROOT / "configs/eval.yaml").read_text())
SPLITS = {"n_folds": 5, "val_frac": 0.1, "seed": 0}


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    preprocess(ROOT, CFG, root / "data/processed/tiny")
    import pandas as pd

    meta = pd.read_parquet(root / "data/processed/tiny/meta.parquet")
    out = root / "data/splits/tiny"
    out.mkdir(parents=True)
    for s in unseen_single_splits(meta, SPLITS, "tiny"):
        (out / f"fold_{s['fold']}.json").write_text(json.dumps(s))
    (out / "doubles_seen_2of2.json").write_text(
        json.dumps(seen_2of2_split(meta, "tiny"))
    )
    return root


@pytest.fixture(scope="module")
def scores(tiny_root):
    data, splits = load_dataset(tiny_root, "tiny"), load_splits(tiny_root, "tiny")
    return {
        m: run_model(tiny_root, data, splits, EVAL, m, f, {})
        for m, f in PREDICTORS.items()
    }


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
    split = load_splits(tiny_root, "tiny")[0]
    clean = fit_mu(data["delta"], split)
    data["delta"].loc[split["test"]] = np.nan
    np.testing.assert_array_equal(fit_mu(data["delta"], split), clean)
    from deltaaudit.data.leakage import fit_rows

    pred = predict_b1(data, split, fit_rows(data["delta"], split))
    assert np.isfinite(pred.to_numpy()).all()


def test_summary_has_cis(scores):
    import pandas as pd

    stats = {**EVAL["statistics"], "bootstrap_B": 500}
    t = summarize(pd.concat(scores.values()), ["raw_r", "pspec_r"], stats)
    pooled = t[(t.fold == "pooled") & (t.metric == "raw_r")]
    assert ((pooled.ci_lo <= pooled["mean"]) & (pooled["mean"] <= pooled.ci_hi)).all()
    assert set(t.fold) == {"pooled", "0", "1", "2", "3", "4"}


def test_require_prereg_refuses_without_tag(tmp_path):
    with pytest.raises(LeakageError):
        require_prereg(tmp_path)
