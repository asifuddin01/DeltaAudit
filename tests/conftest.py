"""Shared tiny-fixture root: preprocessed data, splits, and synthetic GO graphs."""

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from deltaaudit.data import go_graph
from deltaaudit.data.pseudobulk import preprocess
from deltaaudit.data.splits import seen_2of2_split, unseen_single_splits

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "tests/fixtures/tiny.yaml").read_text())
EVAL = yaml.safe_load((ROOT / "configs/eval.yaml").read_text())
MODELS = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
SPLITS = {"n_folds": 5, "val_frac": 0.1, "seed": 0}


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    processed = root / "data/processed/tiny"
    preprocess(ROOT, CFG, processed)
    meta = pd.read_parquet(processed / "meta.parquet")
    genes = pd.read_parquet(processed / "genes.parquet")
    out = root / "data/splits/tiny"
    out.mkdir(parents=True)
    for s in unseen_single_splits(meta, SPLITS, "tiny"):
        (out / f"fold_{s['fold']}.json").write_text(json.dumps(s))
    (out / "doubles_seen_2of2.json").write_text(
        json.dumps(seen_2of2_split(meta, "tiny"))
    )
    # Synthetic GAF: G-genes share a group term (i % 5) plus a private one; X03 is unannotated.
    annotated = [f"G{i:03d}" for i in range(199)] + ["X00", "X01", "X02"]
    rows = [(g, f"GO:{i % 5}") for i, g in enumerate(annotated)]
    rows += [(g, f"GO:1{i:03d}") for i, g in enumerate(annotated)]
    gaf = root / "tiny.gaf.gz"
    with gzip.open(gaf, "wt") as f:
        for g, t in rows:
            cols = ["UniProtKB", g, g, "involved_in", t, "", "IDA", "", "P", "", ""]
            f.write("\t".join(cols) + "\n")
    cfg = MODELS["go_graph"]
    go_graph.build(go_graph.target_nodes(meta), gaf, cfg, processed / go_graph.GO_DIR)
    union = processed / go_graph.GO_UNION_DIR
    go_graph.build(go_graph.union_nodes(meta, genes), gaf, cfg, union)
    _, adj = go_graph.load(processed, go_graph.GO_UNION_DIR)
    m3b = MODELS["learned"]["M3b"]
    sp.save_npz(
        union / "rewired.npz", go_graph.rewire(adj, m3b["nswap_per_edge"], m3b["seed"])
    )
    return root


def fold0(root: Path) -> dict:
    """Unseen-single fold 0 (load_splits sorts by file name, so doubles_seen_2of2 comes first)."""
    from deltaaudit.pipeline import load_splits

    return next(
        s
        for s in load_splits(root, "tiny")
        if s.get("fold") == 0 and s["split_type"] == "unseen_single"
    )
