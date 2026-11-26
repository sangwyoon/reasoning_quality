"""Incremental predictive value of necessity (N) and robustness (R).

Question: once a set of baseline covariates is in the model, how much does
adding N and/or R reduce held-out log loss when predicting y (an item that is
correct with the intact trace becomes incorrect under an independent
meaning-breaking edit)?

Design
------
* One row per (model, question). Rows with y undefined (items that are not
  correct with the intact trace) are dropped, and the number dropped is
  reported.
* The split is by question_id: every model's row for a question lands in the
  same fold. Held-out question ids are written to ``split.json``.
* Every predictor is a penalized (L2) logistic regression. Continuous
  features enter through B-spline bases (``SplineTransformer`` with the
  ``n_knots`` / ``degree`` given in the config); the inverse regularization
  strength C is chosen per model by inner grouped CV on the training
  questions only.
* Baselines: ``B`` (size, trace length, intact/question-only confidence,
  entropy, answer agreement, correctness flags, family, benchmark) and
  ``B*`` = ``B`` + log P_S(y) + log P_0(y) + their difference + interactions
  with ``correct_intact``. In ``B*`` the raw log-probabilities from which N is
  built are already available to the predictor.
* Gains are differences in mean held-out log loss, in bits per prediction
  (nats / ln 2): dN|R = L(base+R) - L(base+N+R), dR|N = L(base+N) -
  L(base+N+R), plus the marginal and joint gains.

Two kinds of uncertainty (reported separately, labelled differently)
-------------------------------------------------------------------
(a) ``conditional_on_fitted_predictor``: one split, predictors fitted once;
    a cluster bootstrap over held-out questions (all model rows of a
    question resampled together) gives paired CIs for the loss differences.
    It captures sampling variability of the held-out questions only, not
    variability from refitting.
(b) ``refit_over_random_splits``: the whole split + fit (including choice of
    C) is repeated over S random question splits; mean / SD / percentiles of
    the gains are reported. The splits overlap, so the SD is a
    split-sensitivity measure, not a sampling standard error.

What this can and cannot support
--------------------------------
A positive gain means that, within the chosen predictor class (penalized
logistic regression with spline terms on these columns), N or R carries
information about y that the listed covariates do not expose to that class.
It does not show that N or R is a causal or mechanistic quantity. If N is a
deterministic transform of the raw log-probabilities, a gain over ``B`` can
be feature engineering that a flexible enough predictor could reproduce from
the raw inputs; the ``B*`` comparison is the check for that, but a gain over
``B*`` is also limited to the predictor class, the spline resolution and the
sample size. A gain near zero is not proof of no information: it can reflect
limited power. Items outside the intact-correct subset are not covered.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

LN2 = math.log(2.0)
PROB_EPS = 1e-6

DEFAULT_SEED = 0
DEFAULT_TEST_FRAC = 0.2
DEFAULT_N_KNOTS = 5
DEFAULT_DEGREE = 3
DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_INNER_FOLDS = 5
DEFAULT_N_BOOT = 2000
DEFAULT_N_SPLITS = 50
CI_LEVEL = 95.0

REQUIRED_COLUMNS = (
    "question_id", "model", "family", "benchmark", "log_size", "log_trace_len",
    "N", "R", "p_intact_max", "p_q_max", "entropy_intact", "correct_intact",
    "correct_qonly", "logp_gold_intact", "logp_gold_q", "agree_rate", "y",
)

BASE_SPLINE = ["log_size", "log_trace_len", "p_intact_max", "p_q_max",
               "entropy_intact", "agree_rate"]
BASE_LINEAR = ["correct_intact", "correct_qonly"]
BASE_CAT = ["family", "benchmark"]
STAR_SPLINE = ["logp_gold_intact", "logp_gold_q", "logp_diff"]
STAR_LINEAR = ["int_logp_gold_intact", "int_logp_gold_q", "int_logp_diff"]


@dataclass(frozen=True)
class Config:
    seed: int = DEFAULT_SEED
    test_frac: float = DEFAULT_TEST_FRAC
    n_knots: int = DEFAULT_N_KNOTS
    degree: int = DEFAULT_DEGREE
    c_grid: tuple = DEFAULT_C_GRID
    inner_folds: int = DEFAULT_INNER_FOLDS
    n_boot: int = DEFAULT_N_BOOT
    n_splits: int = DEFAULT_N_SPLITS


def _blocks():
    base = (BASE_SPLINE, BASE_LINEAR, BASE_CAT)
    star = (BASE_SPLINE + STAR_SPLINE, BASE_LINEAR + STAR_LINEAR, BASE_CAT)
    return {"B": base, "B*": star}


def model_specs() -> dict:
    """Map model name -> (spline cols, linear cols, categorical cols)."""
    specs = {}
    for name, (sp, li, ca) in _blocks().items():
        specs[name] = (list(sp), list(li), list(ca))
        specs[name + "+N"] = (sp + ["N"], list(li), list(ca))
        specs[name + "+R"] = (sp + ["R"], list(li), list(ca))
        specs[name + "+N+R"] = (sp + ["N", "R"], list(li), list(ca))
    return specs


def gain_definitions() -> dict:
    """Gain name -> (loss of the smaller model, loss of the larger model)."""
    defs = {}
    for b in ("B", "B*"):
        defs[f"{b}: dN|R"] = (f"{b}+R", f"{b}+N+R")
        defs[f"{b}: dR|N"] = (f"{b}+N", f"{b}+N+R")
        defs[f"{b}: dN (marginal)"] = (b, f"{b}+N")
        defs[f"{b}: dR (marginal)"] = (b, f"{b}+R")
        defs[f"{b}: dN+R (joint)"] = (b, f"{b}+N+R")
    return defs


# ----------------------------------------------------------------- data

def load_records(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input extension: {path.suffix}")


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate columns, keep rows with y defined, add derived columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    n_total = len(df)
    data = df.loc[df["y"].notna()].copy()
    n_y_defined = len(data)
    feature_cols = [c for c in REQUIRED_COLUMNS if c not in ("y",)]
    data = data.dropna(subset=feature_cols)
    data["y"] = data["y"].astype(int)
    data["question_id"] = data["question_id"].astype(str)
    ci = data["correct_intact"].astype(float)
    data["correct_intact"] = ci
    data["logp_diff"] = data["logp_gold_intact"] - data["logp_gold_q"]
    data["int_logp_gold_intact"] = data["logp_gold_intact"] * ci
    data["int_logp_gold_q"] = data["logp_gold_q"] * ci
    data["int_logp_diff"] = data["logp_diff"] * ci
    data = data.reset_index(drop=True)
    info = {
        "rows_input": int(n_total),
        "rows_y_defined": int(n_y_defined),
        "rows_dropped_y_undefined": int(n_total - n_y_defined),
        "rows_dropped_missing_features": int(n_y_defined - len(data)),
        "rows_used": int(len(data)),
        "questions_used": int(data["question_id"].nunique()),
    }
    return data, info


def split_questions(question_ids: Sequence, seed: int,
                    test_frac: float = DEFAULT_TEST_FRAC) -> tuple[list, list]:
    """Question-level split. Returns (train_ids, test_ids), each sorted."""
    uniq = np.array(sorted(set(str(q) for q in question_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(test_frac * len(uniq))))
    test = sorted(uniq[perm[:n_test]].tolist())
    train = sorted(uniq[perm[n_test:]].tolist())
    return train, test


# ------------------------------------------------------------- modelling

def _make_pipeline(spline_cols, linear_cols, cat_cols, C, cfg: Config):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    parts = []
    if spline_cols:
        spl = Pipeline([
            ("spline", SplineTransformer(n_knots=cfg.n_knots, degree=cfg.degree,
                                         knots="uniform", extrapolation="constant",
                                         include_bias=False)),
            ("scale", StandardScaler()),
        ])
        parts.append(("spline", spl, list(spline_cols)))
    if linear_cols:
        parts.append(("linear", StandardScaler(), list(linear_cols)))
    if cat_cols:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols)))
    return Pipeline([
        ("features", ColumnTransformer(parts, remainder="drop")),
        ("logit", LogisticRegression(C=C, max_iter=5000)),
    ])


def _varying(train: pd.DataFrame, cols: Sequence[str]) -> list:
    return [c for c in cols if train[c].nunique() > 1]


def per_row_logloss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), PROB_EPS, 1 - PROB_EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def select_c(train: pd.DataFrame, spec, cfg: Config) -> float:
    """Choose C by inner grouped CV (groups = question_id) on train only."""
    from sklearn.model_selection import GroupKFold

    sp, li, ca = (_varying(train, c) for c in spec)
    groups = train["question_id"].to_numpy()
    n_folds = min(cfg.inner_folds, len(set(groups)))
    folds = list(GroupKFold(n_splits=n_folds).split(train, groups=groups))
    best_c, best_loss = None, np.inf
    for C in cfg.c_grid:
        losses = []
        for tr, va in folds:
            a, b = train.iloc[tr], train.iloc[va]
            if a["y"].nunique() < 2:
                continue
            pipe = _make_pipeline(sp, li, ca, C, cfg).fit(a, a["y"])
            losses.append(per_row_logloss(b["y"], pipe.predict_proba(b)[:, 1]))
        score = float(np.mean(np.concatenate(losses)))
        if score < best_loss:
            best_c, best_loss = C, score
    return float(best_c)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, spec, cfg: Config):
    C = select_c(train, spec, cfg)
    sp, li, ca = (_varying(train, c) for c in spec)
    pipe = _make_pipeline(sp, li, ca, C, cfg).fit(train, train["y"])
    return pipe.predict_proba(test)[:, 1], C


def run_split(data: pd.DataFrame, seed: int, cfg: Config) -> dict:
    """Fit every model on the train questions; return held-out per-row losses."""
    train_ids, test_ids = split_questions(data["question_id"], seed, cfg.test_frac)
    in_test = data["question_id"].isin(set(test_ids))
    train, test = data.loc[~in_test], data.loc[in_test].reset_index(drop=True)
    losses, chosen = {}, {}
    for name, spec in model_specs().items():
        p, C = fit_predict(train, test, spec, cfg)
        losses[name] = per_row_logloss(test["y"], p)
        chosen[name] = C
    return {"train_ids": train_ids, "test_ids": test_ids, "train": train,
            "test": test, "losses": losses, "chosen_c": chosen}


# ----------------------------------------------------------- uncertainty

def resample_question_indices(n_questions: int, n_boot: int,
                              rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_questions) indices of questions drawn with replacement."""
    return rng.integers(0, n_questions, size=(n_boot, n_questions))


def cluster_bootstrap_means(question_ids, values, idx: np.ndarray) -> np.ndarray:
    """Bootstrap replicates of the row-mean of ``values``.

    Each replicate resamples questions (``idx`` row) and takes every row of
    each drawn question, so rows of one question are never split.
    """
    codes, uniq = pd.factorize(pd.Series(np.asarray(question_ids)))
    n_q = len(uniq)
    sums = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=n_q)
    cnts = np.bincount(codes, minlength=n_q).astype(float)
    return sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)


def _ci(reps: np.ndarray) -> list:
    lo, hi = np.percentile(reps, [(100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2])
    return [float(lo), float(hi)]


def gains_with_ci(question_ids, losses: Mapping[str, np.ndarray], n_boot: int,
                  rng: np.random.Generator) -> dict:
    """Paired gains (bits/prediction) with cluster-bootstrap percentile CIs."""
    n_q = len(set(np.asarray(question_ids).tolist()))
    idx = resample_question_indices(n_q, n_boot, rng)
    out = {}
    for name, (small, large) in gain_definitions().items():
        diff = (losses[small] - losses[large]) / LN2
        reps = cluster_bootstrap_means(question_ids, diff, idx)
        out[name] = {"gain_bits": float(diff.mean()), "ci95": _ci(reps)}
    return out


def point_gains(losses: Mapping[str, np.ndarray]) -> dict:
    return {n: float(((losses[a] - losses[b]) / LN2).mean())
            for n, (a, b) in gain_definitions().items()}


def refit_summary(data: pd.DataFrame, cfg: Config) -> dict:
    """Repeat split + fit over ``cfg.n_splits`` seeds; summarise the gains."""
    rows = [point_gains(run_split(data, cfg.seed + 1 + s, cfg)["losses"])
            for s in range(cfg.n_splits)]
    frame = pd.DataFrame(rows)
    summary = {}
    for col in frame.columns:
        v = frame[col].to_numpy()
        summary[col] = {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "p2.5": float(np.percentile(v, 2.5)),
            "p50": float(np.percentile(v, 50)),
            "p97.5": float(np.percentile(v, 97.5)),
            "frac_positive": float((v > 0).mean()),
        }
    return {"n_splits": cfg.n_splits, "gains_bits": summary}


# ---------------------------------------------------------------- report

def build_report(data: pd.DataFrame, info: dict, cfg: Config,
                 skip_refit: bool = False) -> tuple[dict, dict]:
    split = run_split(data, cfg.seed, cfg)
    test, train, losses = split["test"], split["train"], split["losses"]
    rng = np.random.default_rng(cfg.seed + 7919)
    qids = test["question_id"].to_numpy()

    per_family = {}
    for fam, sub in test.groupby("family"):
        sel = sub.index.to_numpy()
        per_family[str(fam)] = {
            "test_rows": int(len(sub)),
            "test_questions": int(sub["question_id"].nunique()),
            "gains": gains_with_ci(sub["question_id"].to_numpy(),
                                   {k: v[sel] for k, v in losses.items()},
                                   cfg.n_boot, rng),
        }

    report = {
        "config": {
            "seed": cfg.seed, "test_frac": cfg.test_frac, "n_knots": cfg.n_knots,
            "degree": cfg.degree, "c_grid": list(cfg.c_grid),
            "inner_folds": cfg.inner_folds, "n_boot": cfg.n_boot,
            "n_splits": cfg.n_splits, "spline_knots": "uniform",
        },
        "data": info,
        "counts": {
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "train_questions": len(split["train_ids"]),
            "test_questions": len(split["test_ids"]),
        },
        "prevalence": {
            "overall": float(data["y"].mean()),
            "per_family": {str(k): float(v) for k, v in
                           data.groupby("family")["y"].mean().items()},
        },
        "chosen_c": split["chosen_c"],
        "absolute_log_loss": {
            n: {"nats": float(l.mean()), "bits": float(l.mean() / LN2)}
            for n, l in losses.items()
        },
        "conditional_on_fitted_predictor": {
            "description": "single split, predictors fitted once; cluster "
                           "bootstrap over held-out questions",
            "gains": gains_with_ci(qids, losses, cfg.n_boot, rng),
        },
        "per_family_conditional_on_fitted_predictor": per_family,
    }
    if not skip_refit:
        report["refit_over_random_splits"] = refit_summary(data, cfg)
    return report, split


def render_markdown(report: dict) -> str:
    c, d, n = report["config"], report["data"], report["counts"]
    L = ["# Incremental predictive value of N and R", ""]
    L += [f"- rows used: {d['rows_used']} (dropped, y undefined: "
          f"{d['rows_dropped_y_undefined']}; dropped, missing features: "
          f"{d['rows_dropped_missing_features']}); questions: {d['questions_used']}",
          f"- train: {n['train_rows']} rows / {n['train_questions']} questions; "
          f"test: {n['test_rows']} rows / {n['test_questions']} questions",
          f"- splines: n_knots={c['n_knots']}, degree={c['degree']}; "
          f"seed={c['seed']}; bootstrap reps={c['n_boot']}",
          f"- prevalence of y: {report['prevalence']['overall']:.4f}", ""]
    L += ["| family | prevalence |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in report["prevalence"]["per_family"].items()]
    L += ["", "## Chosen C and held-out log loss", "",
          "| model | C | loss (nats) | loss (bits) |", "|---|---|---|---|"]
    for m, v in report["absolute_log_loss"].items():
        L.append(f"| {m} | {report['chosen_c'][m]} | {v['nats']:.4f} | {v['bits']:.4f} |")
    L += ["", "## (a) Gains, conditional on the fitted predictor "
          "(cluster bootstrap over held-out questions)", "",
          "| gain | bits/prediction | 95% CI |", "|---|---|---|"]
    for k, v in report["conditional_on_fitted_predictor"]["gains"].items():
        L.append(f"| {k} | {v['gain_bits']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] |")
    if "refit_over_random_splits" in report:
        r = report["refit_over_random_splits"]
        L += ["", f"## (b) Gains, refit over {r['n_splits']} random question splits", "",
              "| gain | mean | SD | p2.5 | p50 | p97.5 | frac > 0 |",
              "|---|---|---|---|---|---|---|"]
        for k, v in r["gains_bits"].items():
            L.append(f"| {k} | {v['mean']:.4f} | {v['sd']:.4f} | {v['p2.5']:.4f} | "
                     f"{v['p50']:.4f} | {v['p97.5']:.4f} | {v['frac_positive']:.2f} |")
    L += ["", "## Per-family gains (conditional on the fitted predictor)", ""]
    for fam, v in report["per_family_conditional_on_fitted_predictor"].items():
        L += [f"### {fam} ({v['test_rows']} rows, {v['test_questions']} questions)", "",
              "| gain | bits/prediction | 95% CI |", "|---|---|---|"]
        for k, g in v["gains"].items():
            L.append(f"| {k} | {g['gain_bits']:.4f} | "
                     f"[{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}] |")
        L.append("")
    return "\n".join(L) + "\n"


def write_outputs(report: dict, split: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps({
        "seed": report["config"]["seed"], "test_frac": report["config"]["test_frac"],
        "heldout_question_ids": split["test_ids"]}, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "report.md").write_text(render_markdown(report))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="records.parquet or records.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    p.add_argument("--n-knots", type=int, default=DEFAULT_N_KNOTS)
    p.add_argument("--degree", type=int, default=DEFAULT_DEGREE)
    p.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    p.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    p.add_argument("--skip-refit", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg = Config(seed=a.seed, test_frac=a.test_frac, n_knots=a.n_knots,
                 degree=a.degree, c_grid=tuple(a.c_grid), inner_folds=a.inner_folds,
                 n_boot=a.n_boot, n_splits=a.n_splits)
    data, info = prepare(load_records(a.input))
    print(f"rows used {info['rows_used']} (dropped y undefined: "
          f"{info['rows_dropped_y_undefined']})", file=sys.stderr)
    report, split = build_report(data, info, cfg, skip_refit=a.skip_refit)
    write_outputs(report, split, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
