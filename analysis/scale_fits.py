"""Quadratic-in-log-size fits of necessity and robustness curves.

Input is either

* the per-scale means of the scale table (columns ``family``, ``benchmark``,
  ``scale_b``, ``measure``, ``mean``), e.g. rows of ``table4.json``; or
* the long records used by ``analysis.table4`` (columns include
  ``question_id`` and ``subtask``). Means are then computed with the same
  truncation and normalization, and the subtask bootstrap is available.

Fits (numpy least squares): ``value = b0 + b1 * s + b2 * s**2`` with
``s = log(size_B)``, for each (family, benchmark) and for the equal-weight
pool across benchmarks (the mean of benchmark means at each scale, over the
benchmarks present at that scale). A fit needs >= 3 distinct scales.

Reported per fit: beta2; turning point ``exp(-b1 / (2 b2))`` in billions of
parameters; ``interior`` (turning point strictly between the smallest and
largest scale); slopes (per unit of log size) at the smallest and largest
scale; ``expected_sign`` (necessity: beta2 < 0, robustness: beta2 > 0).

Subtask bootstrap (records input only): within each family, subtasks
(benchmark, subtask) are resampled with replacement; at each scale the
subtask means are averaged with equal weight; the quadratic is refit. Among
replicates with >= 3 distinct scales we report the fraction with the
expected curvature sign and the fraction that also have an interior turning
point. Seeded.

Outputs ``scale_fits.json`` and ``scale_fits.md``. These are descriptive fits
of a three-parameter curve to few scales; they are not a test of a
mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import table4

MEASURES = table4.MEASURES
EXPECTED_SIGN = {"necessity": -1, "robustness": 1}
DEFAULT_N_BOOT = 500
DEFAULT_SEED = 0
MIN_SCALES = 3


def fit_quadratic(sizes_b, values) -> dict | None:
    """OLS quadratic in log size. None when fewer than 3 distinct scales."""
    x = np.asarray(sizes_b, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if len(np.unique(x)) < MIN_SCALES:
        return None
    s = np.log(x)
    b2, b1, b0 = np.polyfit(s, y, 2)
    smin, smax = float(s.min()), float(s.max())
    res = {"b0": float(b0), "b1": float(b1), "b2": float(b2),
           "n_scales": int(len(np.unique(x))),
           "slope_smallest": float(b1 + 2 * b2 * smin),
           "slope_largest": float(b1 + 2 * b2 * smax),
           "turning_point_b": None, "interior": False}
    if abs(b2) > 1e-12:
        s_star = -b1 / (2 * b2)
        res["turning_point_b"] = float(np.exp(s_star))
        res["interior"] = bool(smin < s_star < smax)
    return res


def add_flags(fit: dict | None, measure: str) -> dict | None:
    if fit is None:
        return None
    fit = dict(fit)
    fit["expected_sign"] = bool(np.sign(fit["b2"]) == EXPECTED_SIGN[measure])
    return fit


def means_from_records(df: pd.DataFrame) -> pd.DataFrame:
    """Per (family, benchmark, scale, measure) mean of normalized values."""
    df = df.copy()
    df["question_id"] = df["question_id"].astype(str)
    rows = []
    for m in MEASURES:
        df[m + "_norm"] = table4.normalize(df, m)
        per_q = (df.groupby(["family", "benchmark", "scale_b", "question_id"])
                 [m + "_norm"].mean().dropna().reset_index())
        g = (per_q.groupby(["family", "benchmark", "scale_b"])[m + "_norm"]
             .mean().reset_index(name="mean"))
        g["measure"] = m
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def fits_from_means(means: pd.DataFrame) -> list:
    out = []
    for m in MEASURES:
        sub = means[means["measure"] == m]
        for fam, fsub in sub.groupby("family", sort=True):
            groups = [(b, bsub) for b, bsub in fsub.groupby("benchmark", sort=True)]
            for bench, bsub in groups:
                fit = add_flags(fit_quadratic(bsub["scale_b"], bsub["mean"]), m)
                out.append(dict(family=fam, benchmark=bench, measure=m, fit=fit))
            pooled = fsub.groupby("scale_b")["mean"].mean()
            fit = add_flags(fit_quadratic(pooled.index, pooled.to_numpy()), m)
            out.append(dict(family=fam, benchmark="pooled_equal_weight",
                            measure=m, fit=fit))
    return out


def _subtask_matrix(fam_df: pd.DataFrame, measure: str):
    """(subtask x scale) matrix of subtask means of normalized values."""
    col = measure + "_norm"
    key = fam_df["benchmark"].astype(str) + "/" + fam_df["subtask"].astype(str)
    tab = (fam_df.assign(_k=key).groupby(["_k", "scale_b"])[col].mean()
           .unstack("scale_b"))
    return tab.index.to_numpy(), tab.columns.to_numpy(dtype=float), tab.to_numpy()


def subtask_bootstrap(df: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT,
                      seed: int = DEFAULT_SEED) -> list:
    df = df.copy()
    df["question_id"] = df["question_id"].astype(str)
    rng = np.random.default_rng(seed)
    out = []
    for m in MEASURES:
        df[m + "_norm"] = table4.normalize(df, m)
    for m in MEASURES:
        for fam, fdf in df.groupby("family", sort=True):
            _, scales, mat = _subtask_matrix(fdf, m)
            n_sub = mat.shape[0]
            n_valid = n_curv = n_curv_int = 0
            for _ in range(n_boot):
                draw = mat[rng.integers(0, n_sub, size=n_sub)]
                with np.errstate(all="ignore"):
                    curve = np.nanmean(draw, axis=0)
                keep = np.isfinite(curve)
                fit = fit_quadratic(scales[keep], curve[keep])
                if fit is None:
                    continue
                n_valid += 1
                if np.sign(fit["b2"]) == EXPECTED_SIGN[m]:
                    n_curv += 1
                    n_curv_int += int(fit["interior"])
            out.append(dict(
                family=fam, measure=m, n_subtasks=int(n_sub), n_boot=n_boot,
                n_valid=n_valid,
                frac_expected_curvature=(n_curv / n_valid if n_valid else None),
                frac_expected_curvature_interior=(
                    n_curv_int / n_valid if n_valid else None)))
    return out


def load_input(path) -> pd.DataFrame:
    return table4.load_records(path)


def is_records(df: pd.DataFrame) -> bool:
    return {"question_id", "subtask", "necessity", "robustness"} <= set(df.columns)


def build_report(df: pd.DataFrame, n_boot: int = DEFAULT_N_BOOT,
                 seed: int = DEFAULT_SEED) -> dict:
    if is_records(df):
        means = means_from_records(df)
        boot = subtask_bootstrap(df, n_boot, seed)
    else:
        need = {"family", "benchmark", "scale_b", "measure", "mean"}
        if not need <= set(df.columns):
            raise ValueError(f"means input needs columns {sorted(need)}")
        means, boot = df, []
    return {"config": {"n_boot": n_boot, "seed": seed},
            "fits": fits_from_means(means), "subtask_bootstrap": boot}


def _f(x, nd=3) -> str:
    return "NA" if x is None else f"{x:.{nd}f}"


def render_markdown(report: dict) -> str:
    lines = ["# Quadratic fits in log size", "",
             "value = b0 + b1 s + b2 s^2, s = log(size in B). Expected sign: "
             "necessity b2 < 0, robustness b2 > 0.", "",
             "| measure | family | benchmark | beta2 | turning point (B) | "
             "interior | slope smallest | slope largest | expected sign |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in report["fits"]:
        f = r["fit"]
        if f is None:
            lines.append(f"| {r['measure']} | {r['family']} | {r['benchmark']} "
                         "| NA (<3 scales) | | | | | |")
            continue
        lines.append(
            f"| {r['measure']} | {r['family']} | {r['benchmark']} | "
            f"{_f(f['b2'])} | {_f(f['turning_point_b'], 2)} | {f['interior']} | "
            f"{_f(f['slope_smallest'])} | {_f(f['slope_largest'])} | "
            f"{f['expected_sign']} |")
    if report["subtask_bootstrap"]:
        lines += ["", "## Subtask bootstrap", "",
                  "| measure | family | subtasks | valid reps | frac expected "
                  "curvature | frac expected curvature and interior |",
                  "|---|---|---|---|---|---|"]
        for b in report["subtask_bootstrap"]:
            lines.append(
                f"| {b['measure']} | {b['family']} | {b['n_subtasks']} | "
                f"{b['n_valid']}/{b['n_boot']} | "
                f"{_f(b['frac_expected_curvature'])} | "
                f"{_f(b['frac_expected_curvature_interior'])} |")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale_fits.json").write_text(json.dumps(report, indent=2))
    (out / "scale_fits.md").write_text(render_markdown(report))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True,
                   help="table4 means (csv/parquet) or long records")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    write_outputs(build_report(load_input(a.input), a.n_boot, a.seed), Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
