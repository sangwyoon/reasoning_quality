import numpy as np
import pandas as pd
import pytest

try:
    import sklearn  # noqa: F401
except Exception as exc:  # missing or binary-incompatible install
    pytest.skip(f"scikit-learn unusable: {exc}", allow_module_level=True)

from analysis import predictive_value as pv
from analysis import regimes

FAST = pv.Config(seed=3, n_knots=4, degree=3, c_grid=(0.1, 1.0, 10.0),
                 inner_folds=3, n_boot=300, n_splits=3)


def make_records(kind, n_q=500, n_models=3, seed=0):
    """Synthetic records. kind: 'n_in_logp' (y depends on N = logp difference,
    N is a deterministic function of the raw log-probs) or 'r_indep' (y depends
    on R, independent of every baseline column)."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        q_eff = rng.normal(0, 0.3)
        for m in range(n_models):
            gi = rng.normal(-1.0, 1.0)
            gq = rng.normal(-2.0, 1.5)
            n_feat = gi - gq
            r_feat = rng.normal()
            logit = -0.5 + q_eff
            logit += 1.2 * n_feat if kind == "n_in_logp" else 1.2 * r_feat
            y = float(rng.random() < 1 / (1 + np.exp(-logit)))
            ci = float(rng.random() < 0.85)
            rows.append(dict(
                question_id=f"q{q}", model=f"m{m}", family=f"f{m % 2}",
                benchmark="b0", log_size=float(m), log_trace_len=rng.normal(),
                N=n_feat, R=r_feat, p_intact_max=rng.uniform(0.3, 1),
                p_q_max=rng.uniform(0.2, 1), entropy_intact=rng.uniform(0, 1.5),
                correct_intact=ci, correct_qonly=float(rng.random() < 0.5),
                logp_gold_intact=gi, logp_gold_q=gq,
                agree_rate=rng.uniform(0, 1), y=y if ci else np.nan))
    return pd.DataFrame(rows)


def test_prepare_reports_filtered_rows():
    df = make_records("r_indep", n_q=50)
    data, info = pv.prepare(df)
    assert info["rows_dropped_y_undefined"] == int(df["y"].isna().sum()) > 0
    assert info["rows_used"] == len(data) == int(df["y"].notna().sum())
    assert set(data["y"].unique()) <= {0, 1}


def test_split_is_question_grouped():
    data, _ = pv.prepare(make_records("r_indep", n_q=80))
    train, test = pv.split_questions(data["question_id"], seed=1)
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(data["question_id"])
    res = pv.run_split(data, 1, FAST)
    assert not set(res["train"]["question_id"]) & set(res["test"]["question_id"])


def test_split_deterministic_with_seed():
    data, _ = pv.prepare(make_records("r_indep", n_q=80))
    assert pv.split_questions(data["question_id"], 5) == pv.split_questions(data["question_id"], 5)
    assert pv.split_questions(data["question_id"], 5) != pv.split_questions(data["question_id"], 6)
    a = pv.run_split(data, 5, FAST)
    b = pv.run_split(data, 5, FAST)
    assert a["chosen_c"] == b["chosen_c"]
    for k in a["losses"]:
        np.testing.assert_array_equal(a["losses"][k], b["losses"][k])


def test_bootstrap_resamples_whole_questions():
    qids = np.array(["a", "a", "a", "b", "c", "c"])
    vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 40.0])
    idx = pv.resample_question_indices(3, 50, np.random.default_rng(0))
    reps = pv.cluster_bootstrap_means(qids, vals, idx)
    sums, cnts = np.array([6.0, 10.0, 60.0]), np.array([3.0, 1.0, 2.0])
    expected = np.array([sums[i].sum() / cnts[i].sum() for i in idx])
    np.testing.assert_allclose(reps, expected)


def test_bootstrap_ci_wider_when_rows_share_question_effect():
    rng = np.random.default_rng(0)
    q = np.repeat(np.arange(100), 5)
    shared = np.repeat(rng.normal(size=100), 5)
    idx_c = pv.resample_question_indices(100, 1000, np.random.default_rng(1))
    clustered = pv.cluster_bootstrap_means(q, shared, idx_c)
    naive = np.array([shared[np.random.default_rng(i).integers(0, 500, 500)].mean()
                      for i in range(300)])
    assert clustered.std() > 1.5 * naive.std()


def test_gain_over_B_but_not_Bstar_when_N_is_encoded_in_logprobs():
    data, _ = pv.prepare(make_records("n_in_logp", n_q=700))
    split = pv.run_split(data, 3, FAST)
    g = pv.gains_with_ci(split["test"]["question_id"].to_numpy(), split["losses"],
                         300, np.random.default_rng(0))
    over_b = g["B: dN|R"]
    over_star = g["B*: dN|R"]
    assert over_b["ci95"][0] > 0 and over_b["gain_bits"] > 0.05
    assert abs(over_star["gain_bits"]) < 0.02
    assert over_star["ci95"][0] <= 0.0 <= over_star["ci95"][1]


def test_gain_for_feature_independent_of_baseline():
    data, _ = pv.prepare(make_records("r_indep", n_q=700))
    split = pv.run_split(data, 3, FAST)
    g = pv.gains_with_ci(split["test"]["question_id"].to_numpy(), split["losses"],
                         300, np.random.default_rng(0))
    for key in ("B: dR|N", "B*: dR|N"):
        assert g[key]["gain_bits"] > 0.05 and g[key]["ci95"][0] > 0


def test_report_structure_and_outputs(tmp_path):
    df = make_records("r_indep", n_q=120)
    data, info = pv.prepare(df)
    cfg = pv.Config(seed=1, n_knots=4, c_grid=(0.1, 1.0), inner_folds=3,
                    n_boot=50, n_splits=2)
    report, split = pv.build_report(data, info, cfg)
    pv.write_outputs(report, split, tmp_path)
    assert (tmp_path / "split.json").exists() and (tmp_path / "report.md").exists()
    assert report["counts"]["train_questions"] + report["counts"]["test_questions"] == info["questions_used"]
    assert "refit_over_random_splits" in report
    assert set(report["per_family_conditional_on_fitted_predictor"]) == {"f0", "f1"}


def test_regimes_adjusted_probabilities():
    data, _ = pv.prepare(make_records("r_indep", n_q=300))
    out = regimes.adjusted_failure_by_regime(data, seed=2, n_boot=100)
    assert set(out["regimes"]) == set(regimes.REGIME_LABELS.values())
    for v in out["regimes"].values():
        assert 0 <= v["ci95"][0] <= v["adjusted_failure_prob"] <= v["ci95"][1] <= 1
    r = out["regimes"]
    assert r["low N / high R"]["adjusted_failure_prob"] > r["low N / low R"]["adjusted_failure_prob"]
