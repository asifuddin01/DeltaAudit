"""Per-fold node features for the learned models (BUILD_GUIDE E2, prereg M1).

Nodes = panel genes + targets (the go_union graph). Features per node:
E_delta (PCA of delta_train^T, train perturbations only), E_ctrl (control-cell correlation
embedding from preprocess), and a missing indicator for nodes outside the panel (features 0).
Each feature column is scaled to unit std over panel nodes.
"""

import numpy as np
import scipy.sparse as sp

from deltaaudit.data.leakage import fit_rows
from deltaaudit.models.linear import gene_embedding


def node_features(data: dict, split: dict, cfg: dict) -> np.ndarray:
    nodes = data["go_union"][0]
    k_d, k_c = cfg["E_delta_k"], cfg["E_ctrl_k"]
    train = fit_rows(data["delta"], split, include_val=False)  # E_delta_fit_on: train
    e_d = gene_embedding(train.to_numpy(np.float64), k_d)
    e_d = np.pad(e_d, ((0, 0), (0, k_d - e_d.shape[1])))  # fewer train perts than k_d
    e_c = data["ctrl_embed"].loc[data["delta"].columns].to_numpy(np.float64)[:, :k_c]
    e = np.hstack([e_d, e_c])
    std = e.std(axis=0)
    e /= np.where(std > 0, std, 1.0)
    col: dict[str, int] = {}
    for i, s in enumerate(data["symbols"]):
        col.setdefault(s, i)  # duplicated symbol (TBCE): first column
    x = np.zeros((len(nodes), e.shape[1] + 1))
    for i, n in enumerate(nodes):
        if n in col:
            x[i, :-1] = e[col[n]]
        else:
            x[i, -1] = 1.0
    return x if cfg["missing_indicator"] else x[:, :-1]


def propagate(adj: sp.csr_matrix, x: np.ndarray, hops: int) -> np.ndarray:
    """SGC: A_hat^hops X with A_hat = D^-1/2 (A + I) D^-1/2 (fixed, so precomputed)."""
    a = adj + sp.identity(adj.shape[0], format="csr")
    dinv = sp.diags(1.0 / np.sqrt(np.asarray(a.sum(axis=1)).ravel()))
    a_hat = (dinv @ a @ dinv).tocsr()
    for _ in range(hops):
        x = a_hat @ x
    return x
