import numpy as np
import pytest

from deltaaudit.models.linear import fit_predict, gene_embedding, solve_y_axb


def test_solve_y_axb_recovers_exact_bilinear_map():
    rng = np.random.default_rng(0)
    a, b, k0 = (
        rng.normal(size=(60, 4)),
        rng.normal(size=(3, 25)),
        rng.normal(size=(4, 3)),
    )
    b -= b.mean(
        axis=1, keepdims=True
    )  # exact only when the embedding is mean-zero over perts
    y = a @ k0 @ b + 2.0  # constant offset goes into the centre
    k, centre = solve_y_axb(y, a, b, ridge=1e-10)
    pred = a @ k @ b + centre[:, None]
    np.testing.assert_allclose(pred, y, atol=1e-6)


def test_solve_y_axb_matches_closed_form():
    """K = (A'A + lI)^-1 A' (Y - c) B' (BB' + lI)^-1, c = rowMeans(Y) (const-ae solve_y_axb)."""
    rng = np.random.default_rng(1)
    y, a, b = (
        rng.normal(size=(30, 12)),
        rng.normal(size=(30, 5)),
        rng.normal(size=(4, 12)),
    )
    lam = 0.7
    c = y.mean(axis=1)
    ref = (
        np.linalg.inv(a.T @ a + lam * np.eye(5)) @ a.T @ (y - c[:, None]) @ b.T
        @ np.linalg.inv(b @ b.T + lam * np.eye(4))
    )  # fmt: skip
    k, centre = solve_y_axb(y, a, b, ridge=lam)
    np.testing.assert_allclose(k, ref, atol=1e-10)
    np.testing.assert_allclose(centre, c)


def test_gene_embedding_is_column_centred_pca():
    rng = np.random.default_rng(2)
    expr = rng.normal(size=(15, 40))  # conditions x genes
    g = gene_embedding(expr, k=3)
    x = expr.T - expr.T.mean(axis=0)  # prcomp(genes x conditions), centre=TRUE
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    np.testing.assert_allclose(np.abs(g), np.abs(u[:, :3] * s[:3]), atol=1e-10)


def test_fit_predict_known_answer():
    """Noise-free: delta_p = G W g_target(p) + c, so a held-out target is predicted exactly."""
    rng = np.random.default_rng(3)
    genes, k = 50, 4
    g = rng.normal(size=(genes, k))
    w, c = rng.normal(size=(k, k)), rng.normal(size=genes)
    p_fit, p_new = rng.normal(size=(k, 20)), rng.normal(size=(k, 3))
    p_fit -= p_fit.mean(axis=1, keepdims=True)  # so rowMeans(Y) = c exactly
    y_fit = (g @ w @ p_fit + c[:, None]).T  # perts x genes
    pred = fit_predict(y_fit, g, p_fit, p_new, ridge=1e-10)
    np.testing.assert_allclose(pred, (g @ w @ p_new + c[:, None]).T, atol=1e-6)


@pytest.mark.parametrize("ridge", [0.01, 10.0])
def test_fit_predict_zero_embedding_returns_centre(ridge):
    rng = np.random.default_rng(4)
    y = rng.normal(size=(8, 10))
    g, p = rng.normal(size=(10, 3)), rng.normal(size=(3, 8))
    pred = fit_predict(y, g, p, np.zeros((3, 1)), ridge=ridge)
    np.testing.assert_allclose(pred[0], y.mean(axis=0))
