import json

import numpy as np
import pandas as pd

from analysis import table4

SCALES = [1.0, 2.0, 4.0, 8.0, 16.0]


def make_records(n_q=40, n_sub=4, peak=4.0, seed=0, noise=0.05):
    """necessity: inverse-U in log size peaking at ``peak``; robustness: U."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        for sc in SCALES:
            s = np.log(sc) - np.log(peak)
            rows.append(dict(
                family="fam", scale_b=sc, benchmark="b0", subtask=f"t{q % n_sub}",
                question_id=f"q{q}",
                necessity=-s ** 2 + rng.normal(0, noise),
                robustness=s ** 2 + rng.normal(0, noise)))
    return pd.DataFrame(rows)


def test_intervals_inside_unit_interval_and_labelled():
    tab = table4.build_table(make_records(), n_boot=300, seed=1)
    assert (tab[["pct_lo", "pct_hi", "mean"]] >= 0).all().all()
    assert (tab[["pct_lo", "pct_hi", "mean"]] <= 1).all().all()
    assert (tab["pct_lo"] <= tab["mean"] + 1e-9).all()
    assert (tab["pct_hi"] >= tab["mean"] - 1e-9).all()
    md = table4.render_markdown(tab, 99.0)
    assert "percentile bootstrap" in md and "normal approximation" in md


def test_normal_interval_can_leave_unit_interval():
    # a mean at the boundary: every question has the maximum at one scale
    df = make_records(noise=0.0)
    tab = table4.build_table(df, n_boot=200, seed=1)
    top = tab[(tab.measure == "necessity") & (tab.scale_b == 4.0)].iloc[0]
    assert top["mean"] == 1.0
    assert top["pct_hi"] <= 1.0


def test_truncation_applied_before_normalization():
    df = make_records(noise=0.0)
    df.loc[0, "necessity"] = 1e6  # extreme value
    clipped = table4.truncate(df, "necessity")
    assert clipped.max() < 1e6


def test_bold_marks_peak_necessity_and_min_robustness():
    tab = table4.build_table(make_records(), n_boot=100, seed=2)
    nec = tab[tab.measure == "necessity"]
    rob = tab[tab.measure == "robustness"]
    assert nec[nec.bold].scale_b.tolist() == [4.0]
    assert rob[rob.bold].scale_b.tolist() == [4.0]


def test_bold_ties_at_displayed_precision():
    df = make_records(noise=0.0)
    tab = table4.build_table(df, n_boot=50, seed=0, decimals=0)
    nec = tab[tab.measure == "necessity"]
    assert nec.bold.sum() >= 1
    assert nec.loc[nec.bold, "mean"].round(0).nunique() == 1


def test_deterministic_given_seed():
    a = table4.build_table(make_records(), n_boot=100, seed=5)
    b = table4.build_table(make_records(), n_boot=100, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_cli(tmp_path):
    src = tmp_path / "r.csv"
    make_records(n_q=20).to_csv(src, index=False)
    out = tmp_path / "o"
    assert table4.main(["--input", str(src), "--out", str(out),
                        "--n-boot", "50"]) == 0
    payload = json.loads((out / "table4.json").read_text())
    assert payload["config"]["level"] == 99.0
    assert (out / "table4.md").exists()
