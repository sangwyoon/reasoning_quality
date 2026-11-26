import json

import numpy as np
import pandas as pd
import pytest

try:
    import sklearn  # noqa: F401
except Exception as exc:  # missing or binary-incompatible install
    pytest.skip(f"scikit-learn unusable: {exc}", allow_module_level=True)

from analysis import flagging as fl
from analysis import predictive_value as pv


def make_records(informative=True, n_q=300, n_models=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        for m in range(n_models):
            n_feat, r_feat = rng.normal(), rng.normal()
            logit = -2.0 + (1.5 * n_feat + 1.0 * r_feat if informative else 0.0)
            rows.append(dict(
                question_id=f"q{q}", model=f"m{m}", family="f0", benchmark="b0",
                log_size=float(m), log_trace_len=rng.normal(), N=n_feat, R=r_feat,
                p_intact_max=rng.uniform(0.3, 1), p_q_max=rng.uniform(0.2, 1),
                entropy_intact=rng.uniform(0, 1.5), correct_intact=1.0,
                correct_qonly=0.0, logp_gold_intact=rng.normal(),
                logp_gold_q=rng.normal(), agree_rate=rng.uniform(0, 1),
                y=float(rng.random() < 1 / (1 + np.exp(-logit)))))
    return pd.DataFrame(rows)


def run(df, **kw):
    data, info = pv.prepare(df)
    return fl.build_report(data, info, n_boot=200, seed=1, **kw)


def test_catch_rate_basic():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    risk = np.array([0.9, 0.1, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert fl.catch_rate(y, risk, 0.2) == 1.0
    assert fl.catch_rate(y, risk, 0.1) == 0.5
    assert np.isnan(fl.catch_rate(np.zeros(5), risk[:5], 0.2))


def test_informative_score_catches_more_than_confidence():
    rep = run(make_records(informative=True, n_q=1000))
    for r in rep["budgets"]:
        assert r["catch_logistic"] > r["catch_confidence"]
    big = rep["budgets"][-1]
    assert big["difference_ci"][0] > 0


def test_noise_scores_catch_near_budget_and_difference_ci_covers_zero():
    rep = run(make_records(informative=False, seed=4))
    for r in rep["budgets"]:
        assert abs(r["catch_logistic"] - r["budget"]) < 0.12
        lo, hi = r["difference_ci"]
        assert lo <= 0 <= hi or abs(r["difference"]) < 0.1


def test_split_uses_heldout_questions_only():
    df = make_records()
    rep = run(df)
    data, _ = pv.prepare(df)
    train_ids, test_ids = pv.split_questions(data["question_id"], 1, 0.2)
    assert rep["heldout_question_ids"] == test_ids
    assert rep["heldout_rows"] == int(data["question_id"].isin(test_ids).sum())


def test_cli(tmp_path):
    src = tmp_path / "r.csv"
    make_records(n_q=100).to_csv(src, index=False)
    out = tmp_path / "o"
    assert fl.main(["--input", str(src), "--out", str(out), "--n-boot", "30",
                    "--confidence", "entropy"]) == 0
    rep = json.loads((out / "flagging.json").read_text())
    assert [b["budget"] for b in rep["budgets"]] == [0.05, 0.10, 0.20]
    assert (out / "flagging.md").exists()
