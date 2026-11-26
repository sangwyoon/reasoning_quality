"""Retained probability mass of the observed top-k outputs, per checkpoint.

Input: one row per (item, condition) with columns ``model``, ``benchmark``,
``condition`` and at least one of

* ``choice_mass_observed``: for multiple-choice items, the probability mass
  that falls on the valid choice labels among the observed top-k outputs;
* ``tail_mass``: for open-ended items, the mass outside the observed top-k.

``n_choices_missing`` (multiple-choice only) is the number of valid choices
that did not appear in the observed top-k.

Retained mass is ``choice_mass_observed`` when it is present and
``1 - tail_mass`` otherwise. Rows with neither are reported as unknown and
are excluded from every statistic (their count is listed).

Outputs ``retained_mass.json`` and ``retained_mass.md``. Groups are
(model, benchmark, condition). Nothing here is random.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

GROUP_COLS = ["model", "benchmark", "condition"]
PERCENTILES = (5, 10, 25, 50, 75, 90)
DEFAULT_THRESHOLD = 0.9


def load_records(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input extension: {path.suffix}")


def retained_mass(df: pd.DataFrame) -> pd.Series:
    """Per-row retained mass (NaN when neither mass column is available)."""
    for c in ("choice_mass_observed", "tail_mass"):
        if c not in df.columns:
            df = df.assign(**{c: np.nan})
    mc = pd.to_numeric(df["choice_mass_observed"], errors="coerce")
    tail = pd.to_numeric(df["tail_mass"], errors="coerce")
    return mc.where(mc.notna(), 1.0 - tail).astype(float)


def retained_mask(df: pd.DataFrame, threshold: float) -> pd.Series:
    """True for rows whose retained mass is >= threshold (unknown -> False)."""
    return (retained_mass(df) >= threshold).fillna(False).astype(bool)


def dropped_counts(df: pd.DataFrame, threshold: float,
                   group_cols: Sequence[str] = GROUP_COLS) -> pd.DataFrame:
    """Per-group counts of rows kept / dropped at ``threshold``."""
    cols = [c for c in group_cols if c in df.columns]
    keep = retained_mask(df, threshold)
    known = retained_mass(df).notna()
    tmp = df[cols].copy()
    tmp["_keep"] = keep.to_numpy()
    tmp["_unknown"] = (~known).to_numpy()
    out = tmp.groupby(cols, dropna=False).agg(
        n_rows=("_keep", "size"), n_kept=("_keep", "sum"),
        n_unknown=("_unknown", "sum")).reset_index()
    out["n_dropped"] = out["n_rows"] - out["n_kept"]
    out["threshold"] = float(threshold)
    return out


def summarize_group(sub: pd.DataFrame) -> dict:
    vals = retained_mass(sub).dropna().to_numpy()
    res = {"n_rows": int(len(sub)), "n_known": int(len(vals)),
           "n_unknown": int(len(sub) - len(vals))}
    if len(vals):
        res["min"] = float(vals.min())
        res["mean"] = float(vals.mean())
        res["median"] = float(np.median(vals))
        res["percentiles"] = {str(p): float(np.percentile(vals, p))
                              for p in PERCENTILES}
    if "n_choices_missing" in sub.columns:
        mc = pd.to_numeric(sub["n_choices_missing"], errors="coerce").dropna()
        res["n_mc_items"] = int(len(mc))
        res["pct_mc_all_choices_observed"] = (
            float(100.0 * (mc == 0).mean()) if len(mc) else None)
    return res


def build_report(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict:
    cols = [c for c in GROUP_COLS if c in df.columns]
    groups = []
    for key, sub in df.groupby(cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(cols, [str(k) for k in key]))
        row.update(summarize_group(sub))
        groups.append(row)
    drops = dropped_counts(df, threshold, cols)
    return {"threshold": float(threshold), "group_cols": cols,
            "groups": groups,
            "dropped_at_threshold": json.loads(drops.to_json(orient="records"))}


def _fmt(x) -> str:
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def render_markdown(report: dict) -> str:
    cols = report["group_cols"]
    lines = ["# Retained top-k mass", "",
             "Retained mass = choice_mass_observed (multiple choice) or "
             "1 - tail_mass (open-ended).", "",
             "| " + " | ".join(cols) + " | n | min | p5 | p10 | median | "
             "% MC all choices observed |",
             "|" + "---|" * (len(cols) + 6)]
    for g in report["groups"]:
        p = g.get("percentiles", {})
        pct = g.get("pct_mc_all_choices_observed")
        pct_s = "NA" if pct is None else f"{pct:.1f}"
        lines.append("| " + " | ".join(g[c] for c in cols)
                     + f" | {g['n_known']} | {_fmt(g.get('min'))} | "
                       f"{_fmt(p.get('5'))} | {_fmt(p.get('10'))} | "
                       f"{_fmt(g.get('median'))} | {pct_s} |")
    t = report["threshold"]
    lines += ["", f"## Rows dropped at retained mass >= {t}", "",
              "| " + " | ".join(cols) + " | rows | kept | dropped | unknown |",
              "|" + "---|" * (len(cols) + 4)]
    for d in report["dropped_at_threshold"]:
        lines.append("| " + " | ".join(str(d[c]) for c in cols)
                     + f" | {d['n_rows']} | {d['n_kept']} | {d['n_dropped']} "
                       f"| {d['n_unknown']} |")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "retained_mass.json").write_text(json.dumps(report, indent=2))
    (out / "retained_mass.md").write_text(render_markdown(report))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True, help="records.parquet or records.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    df = load_records(a.input)
    if not ({"choice_mass_observed", "tail_mass"} & set(df.columns)):
        print("need choice_mass_observed and/or tail_mass", file=sys.stderr)
        return 2
    write_outputs(build_report(df, a.threshold), Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
