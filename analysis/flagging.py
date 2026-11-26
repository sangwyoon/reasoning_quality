"""Fixed-budget flagging of items that flip from correct to incorrect.

Uses the same records and the same question-level split as
``analysis.predictive_value`` (``prepare`` and ``split_questions``): rows
with y undefined are dropped, held-out questions are chosen by seed.

For each budget k (share of held-out rows) the top-k rows by risk are
flagged under two scores:

* ``confidence``: confidence alone, risk = ``1 - p_intact_max`` (or
  ``entropy_intact`` with ``--confidence entropy``). No fitting.
* ``logistic``: an L2 logistic regression on the training questions using N,
  R and the same confidence column; risk = predicted probability of y = 1.

The catch rate is the share of held-out y = 1 rows that are flagged. The
interval for the difference (logistic - confidence) is a percentile cluster
bootstrap over held-out questions: questions are resampled with replacement
and the top-k flagging is recomputed inside each replicate. It is conditional
on the fitted logistic model (no refitting) and on the single split. Ties in
risk are broken by row order.

Outputs ``flagging.json`` and ``flagging.md``. Seeded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import predictive_value as pv

DEFAULT_BUDGETS = (0.05, 0.10, 0.20)
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0
DEFAULT_C = 1.0
CI_LEVEL = 95.0


def confidence_risk(df: pd.DataFrame, kind: str = "p_intact_max") -> np.ndarray:
    if kind == "entropy":
        return df["entropy_intact"].to_numpy(dtype=float)
    return 1.0 - df["p_intact_max"].to_numpy(dtype=float)


def n_flagged(n_rows: int, budget: float) -> int:
    return min(n_rows, max(1, int(math.ceil(budget * n_rows))))


def catch_rate(y: np.ndarray, risk: np.ndarray, budget: float) -> float:
    """Share of y == 1 rows among the top ``budget`` fraction by risk."""
    total = y.sum()
    if total == 0:
        return float("nan")
    k = n_flagged(len(y), budget)
    top = np.argsort(-risk, kind="stable")[:k]
    return float(y[top].sum() / total)


def fit_logistic_risk(train: pd.DataFrame, test: pd.DataFrame, conf_kind: str,
                      C: float = DEFAULT_C) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def feats(d):
        return np.column_stack([d["N"].to_numpy(float), d["R"].to_numpy(float),
                                confidence_risk(d, conf_kind)])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    pipe.fit(feats(train), train["y"].to_numpy())
    return pipe.predict_proba(feats(test))[:, 1]


def bootstrap_difference(question_ids, y, risk_a, risk_b, budget: float,
                         n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Replicates of catch(b) - catch(a), resampling whole questions."""
    codes, uniq = pd.factorize(pd.Series(np.asarray(question_ids)))
    members = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    reps = np.full(n_boot, np.nan)
    for r in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[i] for i in draw])
        ya = y[idx]
        reps[r] = (catch_rate(ya, risk_b[idx], budget)
                   - catch_rate(ya, risk_a[idx], budget))
    return reps


def build_report(data: pd.DataFrame, info: dict, budgets=DEFAULT_BUDGETS,
                 seed: int = DEFAULT_SEED, test_frac: float = pv.DEFAULT_TEST_FRAC,
                 n_boot: int = DEFAULT_N_BOOT,
                 confidence: str = "p_intact_max") -> dict:
    train_ids, test_ids = pv.split_questions(data["question_id"], seed, test_frac)
    in_test = data["question_id"].isin(set(test_ids))
    train, test = data.loc[~in_test], data.loc[in_test].reset_index(drop=True)
    y = test["y"].to_numpy(dtype=int)
    risk_conf = confidence_risk(test, confidence)
    risk_log = fit_logistic_risk(train, test, confidence)
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - CI_LEVEL) / 2, 100 - (100 - CI_LEVEL) / 2
    rows = []
    for b in budgets:
        reps = bootstrap_difference(test["question_id"], y, risk_conf, risk_log,
                                    b, n_boot, rng)
        reps = reps[np.isfinite(reps)]
        c_conf, c_log = catch_rate(y, risk_conf, b), catch_rate(y, risk_log, b)
        rows.append({
            "budget": float(b), "n_flagged": n_flagged(len(y), b),
            "catch_confidence": c_conf, "catch_logistic": c_log,
            "difference": c_log - c_conf,
            "difference_ci": ([float(np.percentile(reps, lo_q)),
                               float(np.percentile(reps, hi_q))]
                              if len(reps) else None)})
    return {"data": info,
            "config": {"seed": seed, "test_frac": test_frac, "n_boot": n_boot,
                       "confidence": confidence, "ci_level": CI_LEVEL},
            "heldout_rows": int(len(test)), "heldout_positives": int(y.sum()),
            "heldout_question_ids": test_ids, "budgets": rows}


def render_markdown(report: dict) -> str:
    lines = ["# Fixed-budget flagging on held-out questions", "",
             f"Held-out rows {report['heldout_rows']}, y=1 rows "
             f"{report['heldout_positives']}. Catch rate = share of y=1 rows "
             f"flagged. The interval is a {report['config']['ci_level']:g}% "
             f"percentile cluster bootstrap over held-out questions for "
             f"(logistic - confidence), conditional on the fitted model.", "",
             "| budget | flagged rows | catch (confidence) | catch (logistic "
             "N, R, confidence) | difference | interval |",
             "|---|---|---|---|---|---|"]
    for r in report["budgets"]:
        ci = r["difference_ci"]
        ci_s = "NA" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {r['budget']:.0%} | {r['n_flagged']} | "
                     f"{r['catch_confidence']:.3f} | {r['catch_logistic']:.3f} | "
                     f"{r['difference']:.3f} | {ci_s} |")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "flagging.json").write_text(json.dumps(report, indent=2))
    (out / "flagging.md").write_text(render_markdown(report))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="records.parquet or records.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-frac", type=float, default=pv.DEFAULT_TEST_FRAC)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--budgets", type=float, nargs="+",
                   default=list(DEFAULT_BUDGETS))
    p.add_argument("--confidence", choices=["p_intact_max", "entropy"],
                   default="p_intact_max")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    data, info = pv.prepare(pv.load_records(a.input))
    report = build_report(data, info, tuple(a.budgets), a.seed, a.test_frac,
                          a.n_boot, a.confidence)
    write_outputs(report, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
