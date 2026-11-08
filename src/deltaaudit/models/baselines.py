"""B0 no-change and B1 train-mean (Phase 4); B3 kNN-GO and B4 additive come in Phase 5.

Predictors follow `pipeline.Predictor`: (data, split, fit_rows) -> predicted deltas for test rows.
"""

import numpy as np
import pandas as pd


def predict_b0(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=split["test"], columns=fit.columns)


def predict_b1(data: dict, split: dict, fit: pd.DataFrame) -> pd.DataFrame:
    mu = fit.mean(axis=0).to_numpy(np.float64)
    return pd.DataFrame(
        np.tile(mu, (len(split["test"]), 1)), index=split["test"], columns=fit.columns
    )


PREDICTORS = {"B0": predict_b0, "B1": predict_b1}
