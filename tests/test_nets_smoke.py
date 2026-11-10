"""Phase 6 smoke: every learned variant trains on the tiny fixture (CPU, few epochs)."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import EVAL, MODELS, fold0  # noqa: E402

from deltaaudit.data.leakage import fit_rows
from deltaaudit.models.train import VARIANTS, make_learned, run_learned
from deltaaudit.pipeline import load_dataset

LEARNED = {
    **MODELS["learned"],
    "max_epochs": 15,
    "early_stopping": {"metric": "val_pspec_r", "patience": 5},
}


def _predict(root, variant, seed=0, corrupt=False):
    data = load_dataset(root, "tiny")
    split = fold0(root)
    if corrupt:
        data["delta"].loc[split["test"]] = np.nan
    predict = make_learned(LEARNED, variant, seed, "cpu", EVAL["degenerate_std"])
    return predict(data, split, fit_rows(data["delta"], split))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variant_trains_and_restores_best(tiny_root, variant):
    pred = _predict(tiny_root, variant)
    info = pred.attrs["fit_info"]
    assert np.isfinite(pred.to_numpy()).all() and len(pred) == 9
    assert 1 <= info["best_epoch"] <= info["epochs_run"] <= LEARNED["max_epochs"]
    assert info["val_pspec_r_restored"] == pytest.approx(
        info["best_val_pspec_r"], abs=1e-5
    )


def test_deterministic_and_ignores_corrupted_test_rows(tiny_root):
    a = _predict(tiny_root, "M2")
    pd.testing.assert_frame_equal(a, _predict(tiny_root, "M2"))
    pd.testing.assert_frame_equal(a, _predict(tiny_root, "M2", corrupt=True))
    assert not np.allclose(a.to_numpy(), _predict(tiny_root, "M2", seed=1).to_numpy())


def test_m1_and_m2_differ_only_by_graph(tiny_root):
    assert not np.allclose(
        _predict(tiny_root, "M1").to_numpy(), _predict(tiny_root, "M2").to_numpy()
    )


def test_run_learned_writes_runs_and_resumes(tiny_root):
    data, splits = load_dataset(tiny_root, "tiny"), [fold0(tiny_root)]
    kw = dict(models=["M1", "M3"], seeds=[0], device="cpu", log=lambda _: None)
    s = run_learned(tiny_root, data, splits, EVAL, LEARNED, **kw)
    assert set(s["run"]) == {
        "seed_0",
        "perm_0/seed_0",
        "perm_1/seed_0",
        "perm_2/seed_0",
    }
    m = tiny_root / "results/tiny/M3/perm_1/seed_0/fold_0/MANIFEST.json"
    info = json.loads(m.read_text())
    assert info["fit_info"]["epochs_run"] >= 1 and info["device"]
    before = m.stat().st_mtime_ns
    again = run_learned(tiny_root, data, splits, EVAL, LEARNED, **kw)
    assert m.stat().st_mtime_ns == before  # up-to-date runs are skipped
    pd.testing.assert_frame_equal(
        s.reset_index(drop=True), again.reset_index(drop=True)
    )
