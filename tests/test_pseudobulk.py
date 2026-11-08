import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from deltaaudit.data.card import canonical_condition
from deltaaudit.data.pseudobulk import preprocess

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "tests/fixtures/tiny.yaml").read_text())


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny")
    preprocess(ROOT, CFG, out)
    return out


def _read(out, name):
    return pd.read_parquet(out / f"{name}.parquet")


def _dense_groupby_mean():
    a = ad.read_h5ad(ROOT / CFG["h5ad"])
    cond = a.obs["condition"].astype(str).map(canonical_condition)
    return pd.DataFrame(a.X.toarray(), columns=a.var_names).groupby(cond.values).mean()


def test_pseudobulk_equals_dense_recomputation(processed):
    pb = _read(processed, "pb_mean")
    dense = _dense_groupby_mean().loc[pb.index, pb.columns]
    np.testing.assert_allclose(pb.to_numpy(), dense.to_numpy(), rtol=1e-5, atol=1e-6)


def test_qc_drops_small_perturbations_and_merges_orderings(processed):
    meta = _read(processed, "meta").set_index("condition")
    assert "G040" not in meta.index and "G041" not in meta.index
    qc = json.loads((processed / "qc.json").read_text())
    assert {d["condition"] for d in qc["dropped"]} == {"G040", "G041"}
    assert meta.loc["G005", "n_cells"] == 40  # 'G005+ctrl' and 'ctrl+G005' merged
    assert "G002+G003" in meta.index  # written 'G003+G002' in the raw file
    assert len(meta) == 44 + 6
    assert meta.loc["G002+G003", "is_double"]
    assert not meta.loc["X00", "target_in_expression_genes"]
    assert meta.loc["G001", "target_knockdown_log2fc"] < -1


def test_half_splits_partition_cells(processed):
    meta = _read(processed, "meta").set_index("condition")
    assert (meta["n_cells_A"] + meta["n_cells_B"] == meta["n_cells"]).all()
    assert ((meta["n_cells_A"] - meta["n_cells_B"]).abs() <= 1).all()
    pb, a, b = (_read(processed, n) for n in ("pb_mean", "pb_half_A", "pb_half_B"))
    p = meta.index
    na, nb = (
        meta["n_cells_A"].to_numpy()[:, None],
        meta["n_cells_B"].to_numpy()[:, None],
    )
    recombined = (a.loc[p].to_numpy() * na + b.loc[p].to_numpy() * nb) / (na + nb)
    np.testing.assert_allclose(recombined, pb.loc[p].to_numpy(), rtol=1e-5, atol=1e-6)


def test_delta_is_mean_minus_ctrl(processed):
    pb, delta = _read(processed, "pb_mean"), _read(processed, "delta")
    assert "ctrl" not in delta.index
    np.testing.assert_allclose(
        delta.to_numpy(), (pb.loc[delta.index] - pb.loc["ctrl"]).to_numpy(), atol=1e-6
    )


def test_genes_and_ctrl_embed(processed):
    genes = _read(processed, "genes")
    assert genes["ensembl_id"].is_unique and (genes["gene_symbol"] == "G198").sum() == 2
    emb = _read(processed, "ctrl_embed")
    assert emb.shape == (200, CFG["preprocess"]["ctrl_embed_k"])
    assert list(emb.index) == list(genes["ensembl_id"])


def test_chunk_size_and_seed_determinism(processed, tmp_path):
    cfg = {**CFG, "preprocess": {**CFG["preprocess"], "chunk_cells": 7}}
    preprocess(ROOT, cfg, tmp_path)
    for name in ("pb_mean", "pb_half_A", "pb_half_B", "ctrl_embed"):
        np.testing.assert_allclose(
            _read(tmp_path, name).to_numpy(),
            _read(processed, name).to_numpy(),
            atol=1e-5,
        )
    manifest = json.loads((processed / "MANIFEST.json").read_text())
    assert manifest["peak_rss_bytes"] > 0 and "code_hash" in manifest


def test_manifest_up_to_date_tracks_config(processed):
    from deltaaudit.manifest import up_to_date

    m = processed / "MANIFEST.json"
    assert up_to_date(m, ROOT, CFG, [CFG["h5ad"]])
    changed = {**CFG, "qc": {**CFG["qc"], "min_cells": 31}}
    assert not up_to_date(m, ROOT, changed, [CFG["h5ad"]])
