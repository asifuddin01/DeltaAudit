"""GO Jaccard kNN graphs, plus label-shuffle, rewired and spectral views.

Two node sets: the targets of the QC-passing singles (B2-GO, B3), and panel genes + targets
(learned models). Test targets are included: the graph uses only GO annotations, never
expression, so it carries no fold information.
Edge weight = Jaccard of GO term sets; keep each node's top_k neighbours with J >= min_jaccard,
then symmetrize by max. No self-loops (the SGC adds its own in Phase 6).
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

GO_DIR = "go"  # under data/processed/<dataset>/: targets only (B2-GO, B3)
GO_UNION_DIR = "go_union"  # panel genes + targets (learned models)
# GAF 2.x columns (0-based)
SYMBOL, QUALIFIER, GO_ID, EVIDENCE, ASPECT, SYNONYMS = 2, 3, 4, 6, 8, 10


def read_gaf(
    path: Path, aspects: list[str], exclude_evidence: list[str]
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """symbol -> GO terms (NOT annotations dropped), and unambiguous synonym -> symbol."""
    terms: dict[str, set[str]] = {}
    syn: dict[str, set[str]] = {}
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if not row or row[0].startswith("!"):
                continue
            if "NOT" in row[QUALIFIER].split("|"):
                continue
            if row[ASPECT] not in aspects or row[EVIDENCE] in exclude_evidence:
                continue
            terms.setdefault(row[SYMBOL], set()).add(row[GO_ID])
            for s in filter(None, row[SYNONYMS].split("|")):
                syn.setdefault(s, set()).add(row[SYMBOL])
    unique = {
        s: next(iter(v)) for s, v in syn.items() if len(v) == 1 and s not in terms
    }
    return terms, unique


def resolve(
    nodes: list[str], terms: dict[str, set[str]], synonyms: dict[str, str]
) -> tuple[list[set[str]], int]:
    """GO term set per node: primary symbol first, else an unambiguous synonym (old GEARS names)."""
    out, n_syn = [], 0
    for n in nodes:
        if n in terms:
            out.append(terms[n])
        elif n in synonyms:
            out.append(terms[synonyms[n]])
            n_syn += 1
        else:
            out.append(set())
    return out, n_syn


def jaccard_matrix(term_sets: list[set[str]]) -> np.ndarray:
    """Dense n x n Jaccard of term sets; 0 where both are empty, diagonal kept."""
    # ponytail: dense n x n Jaccard; fine for ~1k perturbation targets, go sparse for genome-wide.
    vocab = {t: i for i, t in enumerate(sorted(set().union(*term_sets)))}
    rows = [i for i, s in enumerate(term_sets) for _ in s]
    cols = [vocab[t] for s in term_sets for t in s]
    m = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(len(term_sets), len(vocab))
    )
    inter = (m @ m.T).toarray()
    size = np.diag(inter).copy()
    union = size[:, None] + size[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def jaccard_knn(
    term_sets: list[set[str]], top_k: int, min_jaccard: float
) -> sp.csr_matrix:
    j = jaccard_matrix(term_sets)
    np.fill_diagonal(j, 0.0)
    j[j < min_jaccard] = 0.0
    keep = np.argsort(-j, axis=1, kind="stable")[:, :top_k]
    a = np.zeros_like(j)
    r = np.arange(len(j))[:, None]
    a[r, keep] = j[r, keep]
    return sp.csr_matrix(np.maximum(a, a.T))


def coverage(adj: sp.csr_matrix, term_sets: list[set[str]]) -> dict:
    deg = np.diff(adj.indptr)
    n = adj.shape[0]
    return {
        "n_nodes": int(n),
        "n_annotated": int(sum(bool(s) for s in term_sets)),
        "n_with_edge": int((deg > 0).sum()),
        "frac_with_edge": float((deg > 0).mean()) if n else 0.0,
        "mean_degree": float(deg.mean()) if n else 0.0,
        "n_edges": int(adj.nnz // 2),
    }


def permute_labels(adj: sp.csr_matrix, seed: int) -> sp.csr_matrix:
    """Same topology, gene labels shuffled (M3 null): A' = P A P^T."""
    perm = np.random.default_rng(seed).permutation(adj.shape[0])
    return adj[perm][:, perm].tocsr()


def spectral_embedding(adj: sp.csr_matrix, k: int) -> np.ndarray:
    """Eigenvectors 2..k+1 (smallest eigenvalues, first dropped) of the normalized Laplacian
    over non-isolated nodes; isolated nodes get NaN rows (no embedding -> fallback).
    Fewer than k+1 connected nodes -> only n_connected - 1 columns."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    idx = np.flatnonzero(deg > 0)
    k = min(k, len(idx) - 1)
    a = adj[idx][:, idx].toarray()
    dinv = 1.0 / np.sqrt(deg[idx])
    lap = np.eye(len(idx)) - dinv[:, None] * a * dinv[None, :]
    _, vec = np.linalg.eigh(lap)
    vec = vec[:, 1 : k + 1]
    vec *= np.where(
        vec[np.abs(vec).argmax(axis=0), range(k)] < 0, -1.0, 1.0
    )  # fix signs
    out = np.full((adj.shape[0], k), np.nan)
    out[idx] = vec
    return out


def target_nodes(meta: pd.DataFrame) -> list[str]:
    """Targets of the QC-passing singles (graph for B2-GO and B3)."""
    return sorted({t for ts in meta.loc[~meta["is_double"], "targets"] for t in ts})


def union_nodes(meta: pd.DataFrame, genes: pd.DataFrame) -> list[str]:
    """Panel genes plus targets (graph for the learned models, BUILD_GUIDE E2)."""
    return sorted(set(genes["gene_symbol"]) | set(target_nodes(meta)))


def rewire(adj: sp.csr_matrix, nswap_per_edge: int, seed: int) -> sp.csr_matrix:
    """Degree-preserving double-edge swap (M3b null); the original weights are reassigned
    at random to the rewired edges."""
    import networkx as nx

    g = nx.from_scipy_sparse_array(adj)
    n_edges = g.number_of_edges()
    nx.double_edge_swap(
        g,
        nswap=nswap_per_edge * n_edges,
        max_tries=100 * nswap_per_edge * n_edges,
        seed=seed,
    )
    weights = np.random.default_rng(seed).permutation(sp.triu(adj, k=1).data)
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()))
    a = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=adj.shape)
    return (a + a.T).tocsr()


def build(nodes: list[str], gaf_path: Path, go_cfg: dict, out_dir: Path) -> dict:
    """Write out_dir/{graph.npz, nodes.json}; return coverage facts."""
    terms, syn = read_gaf(gaf_path, go_cfg["aspects"], go_cfg["exclude_evidence"])
    sets, n_syn = resolve(nodes, terms, syn)
    adj = jaccard_knn(sets, go_cfg["top_k"], go_cfg["min_jaccard"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "graph.npz", adj)
    (out_dir / "nodes.json").write_text(json.dumps(nodes) + "\n")
    return {**coverage(adj, sets), "n_via_synonym": n_syn, "release": go_cfg["release"]}


def load(processed: Path, sub: str = GO_DIR) -> tuple[list[str], sp.csr_matrix]:
    d = processed / sub
    return (
        json.loads((d / "nodes.json").read_text()),
        sp.load_npz(d / "graph.npz").tocsr(),
    )
