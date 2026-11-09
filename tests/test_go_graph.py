import gzip

import numpy as np
import pytest

from deltaaudit.data.go_graph import (
    coverage,
    jaccard_knn,
    permute_labels,
    read_gaf,
    resolve,
    spectral_embedding,
)

CFG = {
    "aspects": ["P", "F", "C"],
    "exclude_evidence": [],
    "top_k": 2,
    "min_jaccard": 0.1,
}


def _gaf_line(symbol, go, evidence="IDA", aspect="P", qualifier="involved_in", syn=""):
    cols = ["UniProtKB", f"ID_{symbol}", symbol, qualifier, go, "PMID:1", evidence, "",
            aspect, "name", syn, "protein", "taxon:9606", "20260101", "X", "", ""]  # fmt: skip
    return "\t".join(cols)


@pytest.fixture
def gaf(tmp_path):
    lines = ["!gaf-version: 2.2", "!comment with a ! inside"]
    lines += [_gaf_line("A", t) for t in ("GO:1", "GO:2", "GO:3")]
    lines += [_gaf_line("B", t) for t in ("GO:1", "GO:2")]
    lines += [_gaf_line("C", t) for t in ("GO:3", "GO:4")]
    lines += [_gaf_line("NEWD", "GO:4", syn="OLDD|D2")]
    lines += [_gaf_line("E", "GO:9")]
    lines += [_gaf_line("B", "GO:7", qualifier="NOT|involved_in")]  # NOT is dropped
    lines += [_gaf_line("C", "GO:8", evidence="IEA", aspect="F")]
    p = tmp_path / "tiny.gaf.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return p


def test_read_gaf_drops_not_and_filters(gaf):
    terms, syn = read_gaf(gaf, ["P", "F", "C"], [])
    assert terms["B"] == {"GO:1", "GO:2"}
    assert terms["C"] == {"GO:3", "GO:4", "GO:8"}
    assert syn["OLDD"] == "NEWD"
    terms, _ = read_gaf(gaf, ["P"], ["IEA"])
    assert terms["C"] == {"GO:3", "GO:4"}


def test_resolve_uses_synonyms_only_as_fallback(gaf):
    terms, syn = read_gaf(gaf, ["P", "F", "C"], [])
    sets, n_syn = resolve(["A", "OLDD", "ZZZ"], terms, syn)
    assert sets == [{"GO:1", "GO:2", "GO:3"}, {"GO:4"}, set()] and n_syn == 1


def _graph(gaf):
    terms, syn = read_gaf(gaf, CFG["aspects"], CFG["exclude_evidence"])
    nodes = ["A", "B", "C", "OLDD", "E", "ZZZ"]
    sets, _ = resolve(nodes, terms, syn)
    return nodes, jaccard_knn(sets, CFG["top_k"], CFG["min_jaccard"])


def test_graph_symmetric_no_self_edges_and_weights(gaf):
    nodes, a = _graph(gaf)
    d = a.toarray()
    np.testing.assert_array_equal(d, d.T)
    assert (np.diag(d) == 0).all()
    assert d[0, 1] == pytest.approx(2 / 3)  # A={1,2,3}, B={1,2}
    assert d[0, 2] == pytest.approx(1 / 5)  # A, C={3,4,8}
    assert d[2, 3] == pytest.approx(1 / 3)  # C, NEWD={4}
    assert d[4].sum() == 0 and d[5].sum() == 0  # E shares nothing; ZZZ unannotated


def test_top_k_then_symmetrize_max(gaf):
    terms, _ = read_gaf(gaf, CFG["aspects"], [])
    sets = [terms["A"], terms["B"], terms["C"]] + [{"GO:1", "GO:2", "GO:3"}] * 3
    d = jaccard_knn(sets, top_k=1, min_jaccard=0.1).toarray()
    assert ((d > 0).sum(axis=1) >= 1).all()  # each row keeps >= its own top-1
    np.testing.assert_array_equal(d, d.T)


def test_coverage_reported(gaf):
    nodes, a = _graph(gaf)
    terms, syn = read_gaf(gaf, CFG["aspects"], [])
    sets, _ = resolve(nodes, terms, syn)
    c = coverage(a, sets)
    assert c["n_nodes"] == 6 and c["n_annotated"] == 5 and c["n_with_edge"] == 4
    assert c["frac_with_edge"] == pytest.approx(4 / 6)


def test_label_permutation_preserves_degree_sequence(gaf):
    _, a = _graph(gaf)
    p = permute_labels(a, seed=3)
    deg = lambda m: np.sort((m.toarray() > 0).sum(axis=1))  # noqa: E731
    np.testing.assert_array_equal(deg(a), deg(p))
    np.testing.assert_allclose(np.sort(a.data), np.sort(p.data))
    assert not np.array_equal(a.toarray(), p.toarray())


def test_spectral_embedding_isolated_nodes_missing(gaf):
    _, a = _graph(gaf)
    e = spectral_embedding(a, k=2)
    assert e.shape == (6, 2)
    assert np.isnan(e[4]).all() and np.isnan(e[5]).all()
    assert np.isfinite(e[:4]).all()
