import json

import pytest

try:
    import sklearn  # noqa: F401
except Exception as exc:  # missing or binary-incompatible install
    pytest.skip(f"scikit-learn unusable: {exc}", allow_module_level=True)

from analysis import predictive_value as pv
from tests.test_analysis_predictive_value import make_records


def test_cli_csv_roundtrip(tmp_path):
    src = tmp_path / "records.csv"
    make_records("r_indep", n_q=100).to_csv(src, index=False)
    out = tmp_path / "out"
    rc = pv.main(["--input", str(src), "--out", str(out), "--n-boot", "30",
                  "--n-splits", "2", "--c-grid", "0.1", "1", "--n-knots", "4",
                  "--inner-folds", "3"])
    assert rc == 0
    split = json.loads((out / "split.json").read_text())
    assert split["heldout_question_ids"]
    assert json.loads((out / "report.json").read_text())["config"]["n_knots"] == 4
