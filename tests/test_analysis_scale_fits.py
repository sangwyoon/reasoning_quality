import json

import numpy as np
import pandas as pd

from analysis import scale_fits as sf
from tests.test_analysis_table4 import SCALES, make_records


def test_quadratic_recovers_planted_inverse_u():
    sizes = np.array(SCALES)
    s = np.log(sizes)
    y = 0.8 - 0.1 * (s - np.log(4.0)) ** 2
    fit = sf.fit_quadratic(sizes, y)
    assert np.isclose(fit["b2"], -0.1)
    assert np.isclose(fit["turning_point_b"], 4.0)
    assert fit["interior"]
    assert fit["slope_smallest"] > 0 > fit["slope_largest"]


def test_monotone_curve_not_interior():
    sizes = np.array(SCALES)
    fit = sf.fit_quadratic(sizes, np.log(sizes) * 0.1 + 0.01 * np.log(sizes) ** 2)
    assert not fit["interior"]


def test_fewer_than_three_scales_returns_none():
    assert sf.fit_quadratic([1.0, 2.0], [0.1, 0.2]) is None
    assert sf.fit_quadratic([1.0, 1.0, 2.0], [0.1, 0.2, 0.3]) is None


def test_fits_from_means_flags_expected_signs():
    rep = sf.build_report(_means(), n_boot=10)
    pooled = {(r["measure"], r["benchmark"]): r["fit"] for r in rep["fits"]
              if r["family"] == "fam"}
    nec = pooled[("necessity", "pooled_equal_weight")]
    rob = pooled[("robustness", "pooled_equal_weight")]
    assert nec["expected_sign"] and rob["expected_sign"]
    assert nec["b2"] < 0 < rob["b2"]
    assert rep["subtask_bootstrap"] == []


def _means():
    rows = []
    for bench in ("b0", "b1"):
        for sc in SCALES:
            s = np.log(sc) - np.log(4.0)
            rows.append(dict(family="fam", benchmark=bench, scale_b=sc,
                             measure="necessity", mean=0.9 - 0.1 * s ** 2))
            rows.append(dict(family="fam", benchmark=bench, scale_b=sc,
                             measure="robustness", mean=0.1 + 0.1 * s ** 2))
    return pd.DataFrame(rows)


def test_records_input_turning_point_and_subtask_bootstrap():
    rep = sf.build_report(make_records(n_q=60, n_sub=6), n_boot=100, seed=1)
    nec = [r for r in rep["fits"] if r["measure"] == "necessity"
           and r["benchmark"] == "b0"][0]["fit"]
    assert 3.0 < nec["turning_point_b"] < 5.5
    boot = {b["measure"]: b for b in rep["subtask_bootstrap"]}
    assert boot["necessity"]["frac_expected_curvature"] > 0.9
    assert boot["necessity"]["frac_expected_curvature_interior"] > 0.9
    assert boot["robustness"]["frac_expected_curvature"] > 0.9
    assert boot["necessity"]["n_valid"] == 100


def test_bootstrap_reproducible():
    df = make_records(n_q=30, n_sub=5)
    a = sf.subtask_bootstrap(df, 50, seed=3)
    b = sf.subtask_bootstrap(df, 50, seed=3)
    assert a == b


def test_cli(tmp_path):
    src = tmp_path / "r.csv"
    make_records(n_q=20).to_csv(src, index=False)
    out = tmp_path / "o"
    assert sf.main(["--input", str(src), "--out", str(out), "--n-boot", "20"]) == 0
    assert json.loads((out / "scale_fits.json").read_text())["fits"]
    assert "Quadratic fits" in (out / "scale_fits.md").read_text()
