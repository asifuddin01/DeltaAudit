"""Phase 8 on the tiny fixture: seed averaging, fold ceilings, comparisons, decisions, ladder."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import EVAL, MODELS, ROOT  # noqa: E402

from deltaaudit.eval.audit import SUMMARY, run_evaluate, run_stats
from deltaaudit.eval.ladder import ladder
from deltaaudit.models.baselines import run_baselines
from deltaaudit.models.train import run_learned
from deltaaudit.pipeline import RESULTS, load_dataset, load_splits

E = {
    **EVAL,
    "primary_dataset": "tiny",
    "statistics": {**EVAL["statistics"], "bootstrap_B": 500},
}
SEC = {
    **yaml.safe_load((ROOT / "configs/secondary.yaml").read_text()),
    "replication_dataset": "tiny",
}
RUNGS = yaml.safe_load((ROOT / "configs/ladder.yaml").read_text())["rungs"]
LEARNED = {**MODELS["learned"], "max_epochs": 15,
           "early_stopping": {"metric": "val_pspec_r", "patience": 5}}  # fmt: skip


@pytest.fixture(scope="module")
def out(tiny_root):
    data, splits = load_dataset(tiny_root, "tiny"), load_splits(tiny_root, "tiny")
    run_baselines(tiny_root, data, splits, E, MODELS, log=lambda _: None)
    singles = [s for s in splits if s["split_type"] == "unseen_single"]
    run_learned(tiny_root, data, singles, E, LEARNED, ["M1", "M2", "M3", "M3b"], [0, 1],
                "cpu", log=lambda _: None)  # fmt: skip
    ev = run_evaluate(tiny_root, ["tiny"], E, SEC)
    st = run_stats(
        tiny_root, E, SEC, RUNGS, tiny_root / "tiny.gaf.gz", MODELS["go_graph"]
    )
    return {**ev, **st}


def test_summary_files_written(tiny_root, out):
    for f in ("scores.csv", "model_table.csv", "diagnosis.json", "comparisons.csv",
              "decisions.json", "ladder.csv"):  # fmt: skip
        assert (tiny_root / SUMMARY / f).exists(), f


def test_one_row_per_model_perturbation_and_set(out):
    s = out["scores"]
    assert not s.duplicated(["model", "split_type", "analysis_set", "condition"]).any()
    runs = s.groupby("model").n_runs.first()
    assert (
        runs["M1"] == 2 and runs["M3"] == 6 and runs["B2"] == 1
    )  # M3: 3 perms x 2 seeds


def test_seed_average_is_mean_over_runs(tiny_root, out):
    files = sorted(
        (tiny_root / RESULTS / "tiny" / "M3").glob(
            "perm_*/seed_*/fold_0/scores.parquet"
        )
    )
    raw = pd.concat(pd.read_parquet(f) for f in files)
    raw = raw[raw.analysis_set == "all"]
    c = raw.condition.iloc[0]
    s = out["scores"]
    got = s[
        (s.model == "M3") & (s.analysis_set == "all") & (s.condition == c)
    ].pspec_r.item()
    assert got == pytest.approx(raw[raw.condition == c].pspec_r.mean())


def test_fold_ceiling_uses_fold_mu(tiny_root, out):
    desc = pd.read_parquet(
        tiny_root / "data/processed/tiny/descriptive.parquet"
    ).set_index("condition")
    s = (
        out["scores"]
        .query("model == 'B1' and analysis_set == 'all'")
        .set_index("condition")
    )
    np.testing.assert_allclose(s.ceil_raw, desc.loc[s.index, "ceil_raw"], rtol=1e-5)
    assert not np.allclose(s.ceil_pspec, desc.loc[s.index, "ceil_pspec"])


def test_primary_family_complete(out):
    c = out["comparisons"]
    fam = c[(c.family == "primary") & (c.stratum == "")]
    assert list(fam.id) == ["C1", "C2", "C3", "C4", "C5", "C6"]
    cols = [
        "obs",
        "ci_lo",
        "ci_hi",
        "p_raw",
        "p_holm",
        "p_wilcoxon",
        "median_d",
        "frac_pos",
    ]
    assert fam[cols[:5]].notna().all().all()
    assert (fam.p_holm >= fam.p_raw - 1e-12).all()
    assert fam.cluster_n_clusters.notna().all()
    best = json.loads(json.dumps(out["diagnosis"]))["tiny"]["best_learned"]
    assert best["selected"] in ("M1", "M2") and best["n_runs"]["M1"] == 10
    assert fam.set_index("id").at["C1", "a"] == best["selected"]
    assert fam.set_index("id").at["C1", "metric"] == "raw_r"


def test_secondary_sets_strata_and_doubles(out):
    c = out["comparisons"]
    assert set(c[c.family == "doubles"].id) == {f"D{i}" for i in range(1, 7)}
    strata = c[c.stratum != ""]
    assert set(strata.id) <= {"C4", "C5"}


def test_decisions_and_diagnosis(out):
    d = out["decisions"]["tiny"]
    assert {"H1", "H2", "H3", "H4a", "H4b", "C6"} <= set(d)
    assert d["H3"]["status"] in ("stands", "contradicted")
    g = out["diagnosis"]["tiny"]["common_support"]
    assert -1 <= g["spearman_B1_raw_r_vs_sys_cos"]["rho"] <= 1
    assert g["rank"]["B1"]["pspec_r"] >= 1


def test_ladder_from_tiny(out):
    lad = out["ladder"].set_index("model")
    assert lad.L0.all() and lad.L1.all()
    assert not lad.at["B0", "L2"] and lad.at["B0", "highest_rung"] == "L1"
    assert not lad.at["B1", "L3"]
    assert not lad.loc[lad.index != "M2", "L6"].any()


def test_ladder_known_answer():
    rng = np.random.default_rng(0)
    n = 60
    truth = rng.normal(size=n)
    rows = []
    for m, raw, spec, mse in [("B0", 0.0, 0.1, 2.0), ("B1", 0.5, 0.0, 1.0), ("B2", 0.6, 0.2, 0.9),
                              ("M2", 0.8, 0.5, 0.5), ("M3", 0.6, 0.2, 0.9)]:  # fmt: skip
        for i in range(n):
            e = 0.01 * truth[i]
            rows.append({"model": m, "condition": f"P{i}", "raw_r": raw + e,
                         "pspec_r": spec + e, "mse": mse + e})  # fmt: skip
    s = pd.DataFrame(rows)
    lad = ladder(s, s, RUNGS, {"bootstrap_B": 500, "bootstrap_seed": 0, "ci": 0.95})
    lad = lad.set_index("model")
    assert (
        lad.at["M2", "highest_rung"] == "L7"
    )  # clears everything; replication = same data
    assert lad.at["B2", "highest_rung"] == "L4"  # cannot beat itself at L5
    assert lad.at["B1", "highest_rung"] == "L2"
    assert not lad.at["M3", "L5"]  # ties B2: CI not above 0


def test_sensitivity_end_to_end(tiny_root, out):
    """Phase 9 on tiny: HVG panels (fold 0), fold-0 refits, every analysis rendered."""
    from conftest import CFG

    from deltaaudit.eval.sensitivity import (
        LABELS,
        fit_fold,
        hvg_panels,
        run_sensitivity,
    )

    hcfg = {"dataset": "tiny", "fold": 0, "n_top": 120, "n_bins": 20}
    ds_cfg = {**CFG, "h5ad": str(ROOT / "tests/fixtures/tiny.h5ad")}
    names = hvg_panels(tiny_root, ds_cfg, hcfg, load_splits(tiny_root, "tiny"),
                       MODELS["go_graph"], MODELS["learned"]["M3b"], tiny_root / "tiny.gaf.gz")  # fmt: skip
    assert set(names) == {"all", "train"}
    for n in names.values():
        genes = pd.read_parquet(tiny_root / "data/processed" / n / "genes.parquet")
        assert len(genes) == 120
        fit_fold(
            tiny_root,
            n,
            E,
            {**MODELS, "learned": LEARNED},
            0,
            [0],
            "cpu",
            log=lambda _: None,
        )
    df = run_sensitivity(
        tiny_root, E, {**SEC, "sensitivity": {"hvg_reselection": hcfg}}, names
    )
    assert set(df.verdict) <= {"+", "-", "0"}
    assert set(LABELS) <= set(df.analysis)
    md = (tiny_root / SUMMARY / "sensitivity.md").read_text()
    for an in LABELS:
        assert f"`{an}`" in md
    assert "conclusion unchanged" in md or "changed because" in md
    mu = df[df.analysis == "mu_all"].set_index("id")
    main = (
        out["comparisons"]
        .query("family == 'primary' and stratum == ''")
        .set_index("id")
    )
    assert mu.at["C1", "obs"] == pytest.approx(
        main.at["C1", "obs"]
    )  # raw_r does not use mu
