import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from deltaaudit.data.card import (
    canonical_condition,
    parse_condition,
    scan_h5ad,
    summarize,
)


def test_parse_condition():
    assert parse_condition("ctrl") == []
    assert parse_condition("AARS+ctrl") == ["AARS"]
    assert parse_condition("ctrl+AARS") == ["AARS"]
    assert parse_condition("A+B") == ["A", "B"]
    assert canonical_condition("ctrl+A") == canonical_condition("A+ctrl") == "A"
    assert canonical_condition("B+A") == "A+B"
    assert canonical_condition("ctrl") == "ctrl"


# Chunk boundaries must not change the answers.
@pytest.mark.parametrize("chunk", [1, 3, 100])
def test_scan_known_knockdown(tmp_path, chunk):
    # Genes G0..G2 (stored as log1p). Control: G0 = 4 -> mean 4.
    # "G0+ctrl": G0 = 1 -> log2((1+pc)/(4+pc)); "G1+G2": both targets in panel;
    # "OUT+ctrl": target outside panel -> no knockdown value.
    pc = 0.0
    counts = np.array(
        [
            [4, 2, 2],  # ctrl
            [4, 2, 2],  # ctrl
            [1, 2, 2],  # G0+ctrl
            [1, 2, 2],  # G0+ctrl
            [4, 1, 1],  # G1+G2
            [4, 2, 2],  # OUT+ctrl
        ],
        dtype=np.float32,
    )
    cond = ["ctrl", "ctrl", "G0+ctrl", "ctrl+G0", "G2+G1", "OUT+ctrl"]
    a = ad.AnnData(
        X=sp.csr_matrix(np.log1p(counts)),
        obs=pd.DataFrame(
            {"condition": pd.Categorical(cond)}, index=[f"c{i}" for i in range(6)]
        ),
        var=pd.DataFrame({"gene_name": ["G0", "G1", "G2"]}, index=["G0", "G1", "G2"]),
    )
    path = tmp_path / "t.h5ad"
    a.write_h5ad(path)

    per, facts = scan_h5ad(str(path), chunk_cells=chunk, pseudocount=pc)
    per = per.set_index("condition")
    assert per.loc["G0", "n_cells"] == 2 and facts["n_raw_condition_labels"] == 5
    assert facts["n_ctrl_cells"] == 2 and facts["x_storage"] == "sparse"
    assert per.loc["G0", "target_knockdown_log2fc"] == pytest.approx(np.log2(1 / 4))
    assert per.loc["G1+G2", "target_knockdown_log2fc"] == pytest.approx(np.log2(1 / 2))
    assert np.isnan(per.loc["OUT", "target_knockdown_log2fc"])
    assert not per.loc["OUT", "all_targets_in_panel"]
    assert per.loc["G1+G2", "is_double"]

    s = summarize(per.reset_index(), facts, min_cells=2)
    assert (s["n_perturbations"], s["n_single"], s["n_double"]) == (3, 2, 1)
    assert s["n_perturbations_below_min_cells"] == 2
    assert s["single_knockdown_n"] == 1
