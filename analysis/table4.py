"""Scale table of necessity and robustness with cluster-bootstrap intervals.

Input: long records with columns ``family``, ``scale_b`` (size in billions of
parameters), ``benchmark``, ``subtask``, ``question_id``, ``necessity``,
``robustness``.

Procedure (per measure, applied independently):
1. Truncate at the 1st / 99th percentiles computed within (benchmark,
   family), before any normalization.
2. Min-max normalize within (benchmark, family, question) across scales.
   Questions with a constant value across scales get no normalized value
   (NaN) and are skipped.
3. For each (family, benchmark, scale): the mean of normalized values over
   questions, and two intervals at ``level`` percent:

   * ``pct_lo`` / ``pct_hi``: PERCENTILE cluster bootstrap (questions
     resampled with replacement, ``n_boot`` replicates). Each replicate is a
     mean of values in [0, 1], so these always lie in [0, 1].
   * ``norm_lo`` / ``norm_hi``: NORMAL-APPROXIMATION interval, mean +/- z *
     bootstrap SD. It can leave [0, 1]; it is given for comparison only.

Bold markers: per (family, benchmark), the scale with the largest necessity
mean and the one with the smallest robustness mean, compared at displayed
precision (``decimals``), so ties at that precision are all marked.

Outputs ``table4.json`` and ``table4.md``. Seeded; same seed, same output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

MEASURES = ("necessity", "robustness")
KEYS = ("benchmark", "family")
DEFAULT_LEVEL = 99.0
DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 0
DEFAULT_DECIMALS = 2
TRUNC_PCT = (1.0, 99.0)
REQUIRED = ("family", "scale_b", "benchmark", "subtask", "question_id",
            "necessity", "robustness")


def load_records(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported input extension: {path.suffix}")


def truncate(df: pd.DataFrame, measure: str) -> pd.Series:
    """Clip ``measure`` at the 1st/99th percentile within (benchmark, family)."""
    g = df.groupby(list(KEYS))[measure]
    lo = g.transform(lambda s: np.nanpercentile(s, TRUNC_PCT[0]))
    hi = g.transform(lambda s: np.nanpercentile(s, TRUNC_PCT[1]))
    return df[measure].clip(lower=lo, upper=hi)


def normalize(df: pd.DataFrame, measure: str) -> pd.Series:
    """Truncate, then min-max within (benchmark, family, question)."""
    tmp = df[list(KEYS) + ["question_id"]].copy()
    tmp["_v"] = truncate(df, measure)
    g = tmp.groupby(list(KEYS) + ["question_id"])["_v"]
    lo, hi = g.transform("min"), g.transform("max")
    span = (hi - lo).where(hi - lo > 0)
    return (tmp["_v"] - lo) / span


def bootstrap_means(values: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Cluster bootstrap of the mean: ``values`` has one entry per question."""
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def intervals(values: np.ndarray, level: float, n_boot: int,
              rng: np.random.Generator) -> dict:
    mean = float(values.mean())
    if len(values) < 2:
        nan = float("nan")
        return dict(mean=mean, n_questions=int(len(values)), pct_lo=nan,
                    pct_hi=nan, norm_lo=nan, norm_hi=nan)
    reps = bootstrap_means(values, n_boot, rng)
    a = (100.0 - level) / 2.0
    lo, hi = np.percentile(reps, [a, 100.0 - a])
    z = NormalDist().inv_cdf(0.5 + level / 200.0)
    sd = float(reps.std(ddof=1))
    return dict(mean=mean, n_questions=int(len(values)), pct_lo=float(lo),
                pct_hi=float(hi), norm_lo=mean - z * sd, norm_hi=mean + z * sd)


def build_table(df: pd.DataFrame, level: float = DEFAULT_LEVEL,
                n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                decimals: int = DEFAULT_DECIMALS) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df.copy()
    df["question_id"] = df["question_id"].astype(str)
    for m in MEASURES:
        df[m + "_norm"] = normalize(df, m)
    rng = np.random.default_rng(seed)
    rows = []
    for (fam, bench), sub in df.groupby(["family", "benchmark"], sort=True):
        for m in MEASURES:
            per_scale = []
            for scale, s in sub.groupby("scale_b", sort=True):
                per_q = s.groupby("question_id")[m + "_norm"].mean().dropna()
                if per_q.empty:
                    continue
                r = intervals(per_q.to_numpy(), level, n_boot, rng)
                per_scale.append(dict(family=fam, benchmark=bench,
                                      scale_b=float(scale), measure=m, **r))
            if not per_scale:
                continue
            shown = [round(r["mean"], decimals) for r in per_scale]
            best = max(shown) if m == "necessity" else min(shown)
            for r, sh in zip(per_scale, shown):
                r["bold"] = bool(sh == best)
            rows.extend(per_scale)
    return pd.DataFrame(rows)


def render_markdown(table: pd.DataFrame, level: float,
                    decimals: int = DEFAULT_DECIMALS) -> str:
    f = f"{{:.{decimals}f}}"
    lines = ["# Normalized necessity / robustness by scale", "",
             f"Interval columns: `pct interval` = percentile cluster-bootstrap "
             f"interval ({level:g}%, within [0,1]); `normal interval` = mean "
             f"+/- z*SD normal-approximation interval (may leave [0,1]). Bold "
             f"= largest necessity / smallest robustness mean in the family "
             f"and benchmark.", ""]
    for m in MEASURES:
        lines += [f"## {m}", "",
                  "| family | benchmark | scale_b | mean | n_q | "
                  "pct interval (percentile bootstrap) | "
                  "normal interval (normal approximation) |",
                  "|---|---|---|---|---|---|---|"]
        for r in table[table["measure"] == m].itertuples():
            mean = f.format(r.mean)
            mean = f"**{mean}**" if r.bold else mean
            lines.append(
                f"| {r.family} | {r.benchmark} | {r.scale_b:g} | {mean} | "
                f"{r.n_questions} | [{f.format(r.pct_lo)}, {f.format(r.pct_hi)}] | "
                f"[{f.format(r.norm_lo)}, {f.format(r.norm_hi)}] |")
        lines.append("")
    return "\n".join(lines)


def write_outputs(table: pd.DataFrame, cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    payload = {"config": cfg, "rows": json.loads(table.to_json(orient="records"))}
    (out / "table4.json").write_text(json.dumps(payload, indent=2))
    (out / "table4.md").write_text(
        render_markdown(table, cfg["level"], cfg["decimals"]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--level", type=float, default=DEFAULT_LEVEL)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    table = build_table(load_records(a.input), a.level, a.n_boot, a.seed,
                        a.decimals)
    write_outputs(table, dict(level=a.level, n_boot=a.n_boot, seed=a.seed,
                              decimals=a.decimals), Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
