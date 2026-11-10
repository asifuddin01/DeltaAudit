import numpy as np
import scipy.sparse as sp
from conftest import MODELS, fold0  # noqa: E402

from deltaaudit.data.features import node_features, propagate
from deltaaudit.pipeline import load_dataset

FEAT = MODELS["learned"]["features"]


def test_features_use_train_rows_only(tiny_root):
    """Rule 3: E_delta is fitted on train only, so corrupting val and test rows changes nothing."""
    split = fold0(tiny_root)
    data = load_dataset(tiny_root, "tiny")
    clean = node_features(data, split, FEAT)
    data["delta"].loc[split["val"] + split["test"]] = np.nan
    np.testing.assert_array_equal(node_features(data, split, FEAT), clean)
    assert np.isfinite(clean).all()


def test_missing_indicator_and_zero_features_off_panel(tiny_root):
    data = load_dataset(tiny_root, "tiny")
    x = node_features(data, fold0(tiny_root), FEAT)
    nodes = data["go_union"][0]
    off = np.array([n.startswith("X") for n in nodes])
    assert x.shape == (len(nodes), FEAT["E_delta_k"] + data["ctrl_embed"].shape[1] + 1)
    assert (x[off, -1] == 1).all() and (x[off, :-1] == 0).all()
    assert (x[~off, -1] == 0).all() and (np.abs(x[~off, :-1]).sum(axis=1) > 0).all()


def test_propagate_identity_without_edges_and_sym_norm():
    x = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(propagate(sp.csr_matrix((3, 3)), x, hops=2), x)
    a = sp.csr_matrix(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 0]]))
    # A + I on nodes 0-1 has degree 2: A_hat = [[.5,.5],[.5,.5]]; node 2 keeps itself.
    np.testing.assert_allclose(
        propagate(a, x, hops=1), [[1.0, 2.0], [1.0, 2.0], [4.0, 5.0]]
    )
