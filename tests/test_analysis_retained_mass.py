import json

import numpy as np
import pandas as pd

from analysis import retained_mass as rm


def make_df():
    return pd.DataFrame({
        "model": ["a"] * 5 + ["b"] * 4,
        "benchmark": ["mc"] * 4 + ["open"] + ["mc"] * 4,
        "condition": ["intact"] * 9,
        "choice_mass_observed": [1.0, 0.8, 0.6, 0.4, np.nan, 0.9, 0.95, 1.0, 0.5],
        "tail_mass": [np.nan] * 4 + [0.1] + [np.nan] * 4,
        "n_choices_missing": [0, 0, 1, 2, np.nan, 0, 0, 0, 1],
    })


def test_retained_mass_uses_choice_mass_then_tail():
    mass = rm.retained_mass(make_df())
    assert mass.iloc[1] == 0.8
    assert np.isclose(mass.iloc[4], 0.9)


def test_unknown_rows_excluded_from_mask_and_counted():
    df = make_df()
    df.loc[0, ["choice_mass_observed", "tail_mass"]] = np.nan
    assert not rm.retained_mask(df, 0.0).iloc[0]
    rep = rm.build_report(df, 0.5)
    assert sum(g["n_unknown"] for g in rep["groups"]) == 1


def test_percentiles_and_pct_all_observed():
    df = make_df()
    rep = rm.build_report(df, 0.7)
    g = [x for x in rep["groups"] if x["model"] == "a" and x["benchmark"] == "mc"][0]
    assert g["min"] == 0.4
    assert np.isclose(g["median"], 0.7)
    assert g["percentiles"]["50"] == g["median"]
    assert g["pct_mc_all_choices_observed"] == 50.0


def test_mask_and_dropped_counts():
    df = make_df()
    mask = rm.retained_mask(df, 0.9)
    assert mask.tolist() == [True, False, False, False, True, True, True, True, False]
    d = rm.dropped_counts(df, 0.9)
    row = d[(d["model"] == "b")].iloc[0]
    assert (row["n_rows"], row["n_kept"], row["n_dropped"]) == (4, 3, 1)
    assert d["n_dropped"].sum() == int((~mask).sum())


def test_cli_writes_outputs(tmp_path):
    src = tmp_path / "r.csv"
    make_df().to_csv(src, index=False)
    out = tmp_path / "o"
    assert rm.main(["--input", str(src), "--out", str(out), "--threshold", "0.8"]) == 0
    rep = json.loads((out / "retained_mass.json").read_text())
    assert rep["threshold"] == 0.8
    assert "Retained top-k mass" in (out / "retained_mass.md").read_text()
