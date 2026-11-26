"""Four N/R regimes and adjusted failure probability per regime.

Regimes come from median splits of N and R, with the medians computed on the
training questions only. Code = 2 * [N high] + [R high]
(0: low N / low R, 1: low N / high R, 2: high N / low R, 3: high N / high R).

The adjusted failure probability of a regime is the mean predicted
probability on the held-out rows when every row is assigned that regime, from
a logistic model with the baseline covariates plus regime indicators. This is
standardization over the held-out covariate distribution. The CI is a cluster
bootstrap over held-out questions conditional on the fitted model (it does
not include refitting variability). The estimates are model-based
associations within a linear-logit specification; they are not causal
effects of N or R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.predictive_value import (
    BASE_CAT, BASE_LINEAR, BASE_SPLINE, DEFAULT_N_BOOT, CI_LEVEL,
    cluster_bootstrap_means, resample_question_indices, split_questions,
)

REGIME_LABELS = {0: "low N / low R", 1: "low N / high R",
                 2: "high N / low R", 3: "high N / high R"}
REGIME_C = 1.0


def median_thresholds(train: pd.DataFrame) -> dict:
    return {"N": float(train["N"].median()), "R": float(train["R"].median())}


def regime_codes(df: pd.DataFrame, thresholds: dict) -> np.ndarray:
    return (2 * (df["N"].to_numpy() > thresholds["N"]).astype(int)
            + (df["R"].to_numpy() > thresholds["R"]).astype(int))


def _design(df: pd.DataFrame, codes: np.ndarray, cols: list, cat: list) -> pd.DataFrame:
    x = df[cols].astype(float).reset_index(drop=True)
    for k in (1, 2, 3):
        x[f"regime_{k}"] = (codes == k).astype(float)
    for c in cat:
        x = x.join(pd.get_dummies(df[c].astype(str).reset_index(drop=True),
                                  prefix=c, dtype=float))
    return x


def adjusted_failure_by_regime(data: pd.DataFrame, seed: int = 0,
                               test_frac: float = 0.2,
                               n_boot: int = DEFAULT_N_BOOT) -> dict:
    """Return thresholds, raw rates and adjusted probabilities with CIs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_ids, test_ids = split_questions(data["question_id"], seed, test_frac)
    in_test = data["question_id"].isin(set(test_ids))
    train, test = data.loc[~in_test], data.loc[in_test].reset_index(drop=True)
    thr = median_thresholds(train)
    cols = BASE_SPLINE + BASE_LINEAR
    cols = [c for c in cols if train[c].nunique() > 1]

    x_train = _design(train, regime_codes(train, thr), cols, BASE_CAT)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=REGIME_C, max_iter=5000).fit(
        scaler.transform(x_train), train["y"])

    cols_all = list(x_train.columns)
    cat = [c for c in BASE_CAT]
    codes_test = regime_codes(test, thr)
    preds = {}
    for k in REGIME_LABELS:
        x = _design(test, np.full(len(test), k), cols, cat)
        x = x.reindex(columns=cols_all, fill_value=0.0)
        preds[k] = model.predict_proba(scaler.transform(x))[:, 1]

    rng = np.random.default_rng(seed + 104729)
    idx = resample_question_indices(test["question_id"].nunique(), n_boot, rng)
    lo_q, hi_q = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    out = {}
    for k, label in REGIME_LABELS.items():
        reps = cluster_bootstrap_means(test["question_id"].to_numpy(), preds[k], idx)
        sel = codes_test == k
        out[label] = {
            "adjusted_failure_prob": float(preds[k].mean()),
            "ci95": [float(np.percentile(reps, lo_q)), float(np.percentile(reps, hi_q))],
            "raw_failure_rate_heldout": float(test.loc[sel, "y"].mean()) if sel.any() else None,
            "n_heldout_rows": int(sel.sum()),
        }
    return {"thresholds_from_train": thr, "regimes": out,
            "ci_note": "cluster bootstrap over held-out questions, conditional on the fitted model"}
