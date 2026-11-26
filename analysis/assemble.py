"""Assemble the analysis input tables from runner outputs.

Writes ``<out>/{predictive_value,flagging,retained_mass,table4,scale_fits}.csv``,
one row per (model, benchmark, question) except ``retained_mass`` (one row per
item and condition). Records with an ``error`` are not used; for a question
stored more than once the last error-free record counts.

Column definitions (``tag`` = model name as used in the output directory):

* intact condition: the unedited baseline of the sentence ablation run of
  ``--validation-draw`` (``p_intact_max``, ``entropy_intact`` in nats,
  ``correct_intact``, ``logp_gold_intact``);
* question-only condition: the direct run (``p_q_max``, ``correct_qonly``,
  ``logp_gold_q``);
* ``N`` (``necessity``) = ``logp_gold_intact - logp_gold_q``; for open-ended
  benchmarks both are per-token means from teacher forcing;
* ``y``: for items correct with the intact trace, 1 if the meaning-breaking
  edit of ``--validation-draw`` gives a wrong answer; empty otherwise;
* ``R`` (``robustness``): chosen by ``--robustness``: ``mask10`` is
  ``js_cot_cotm`` of the mask-10% run; ``para``/``neg``/``del``/``mask`` is
  ``js_<name>`` of the sentence ablation run of ``--robustness-draw``, which
  must differ from the validation draw;
* ``log_trace_len``: natural log of the trace's whitespace-token count;
* ``family``, ``scale_b`` (billions of parameters) and ``log_size``
  (natural log of ``scale_b``): from ``--models``;
* ``agree_rate``: not defined by the runners; read from ``--agree-rate``
  (columns ``model, benchmark, question_id, agree_rate``);
* ``subtask``: the field of the CoT record named by ``--subtask-field``.

``predictive_value`` and ``flagging`` cover multiple-choice benchmarks only:
``p_intact_max``, ``entropy_intact``, ``p_q_max`` and ``correct_qonly`` are not
defined for open-ended benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

import measures
from analysis import predictive_value as pv
from analysis import table4
from llm_client import make_model_tag

MC_BENCHMARKS = ("gpqa", "mmlu", "agie", "mmlupro")
OPEN_DIRECT = {
    "gsm8k": ("kl_divergence_gsm8k_{tag}", "kl_onehot_direct"),
    "math_t": ("kl_direct_math_t_{tag}", "kl_oh_direct"),
    "omni": ("kl_direct_omni_{tag}", "kl_oh_direct"),
}
BENCHMARKS = MC_BENCHMARKS + tuple(OPEN_DIRECT)
CONFIDENCE_TABLES = ("predictive_value", "flagging")
SCALE_TABLES = ("table4", "scale_fits")
TABLES = CONFIDENCE_TABLES + ("retained_mass",) + SCALE_TABLES
VARIANTS = ("para", "neg", "del", "mask")
ROBUSTNESS = ("mask10",) + VARIANTS


def read_records(directory: Path) -> tuple[dict, set]:
    """(question_id -> last error-free record, ids that only have error records)."""
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no runner output in {directory}")
    good, failed = {}, set()
    for path in files:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("error") is None:
                    good[record["question_id"]] = record
                else:
                    failed.add(record["question_id"])
    return good, failed - set(good)


def ablation_dir(results: Path, benchmark: str, tag: str, draw: int) -> Path:
    suffix = f"_draw{draw}" if draw else ""
    return results / f"abl_sent_{benchmark}_{tag}{suffix}"


def load_models(path: str) -> dict:
    models = json.loads(Path(path).read_text())
    for name, meta in models.items():
        if not meta.get("family") or not float(meta.get("size_b", 0)) > 0:
            raise SystemExit(f"--models: {name!r} needs 'family' and a positive 'size_b'")
    return models


def question_only(results: Path, benchmark: str, tag: str) -> dict:
    """question_id -> question-only columns from the direct run."""
    if benchmark in OPEN_DIRECT:
        folder, key = OPEN_DIRECT[benchmark]
        records, _ = read_records(results / folder.format(tag=tag))
        return {q: {"logp_gold_q": -r[key]} for q, r in records.items()}
    records, _ = read_records(results / f"exp2_direct_{benchmark}_{tag}")
    return {q: {
        "ground_truth": r["ground_truth"],
        "p_q_max": max(r["top_probs"].values()),
        "correct_qonly": int(r["predicted"] == r["ground_truth"]),
        "logp_gold_q": -measures.kl_onehot_choice(r["top_probs"], r["ground_truth"]),
    } for q, r in records.items()}


def robustness_values(results: Path, benchmark: str, tag: str, args) -> dict:
    """question_id -> (R, selected sentence indices or None)."""
    if args.robustness == "mask10":
        records, _ = read_records(results / f"kl_mask10_{benchmark}_{tag}")
        return {q: (r["js_cot_cotm"], None) for q, r in records.items()}
    records, _ = read_records(ablation_dir(results, benchmark, tag, args.robustness_draw))
    return {q: (r[f"js_{args.robustness}"], r["sel_indices"]) for q, r in records.items()}


def intact_columns(record: dict, benchmark: str) -> dict:
    if "is_correct_orig" not in record:
        raise SystemExit("sentence ablation records lack 'is_correct_orig'; re-run the ablation")
    columns = {
        "log_trace_len": math.log(record["n_trace_tokens"]),
        "correct_intact": int(record["is_correct_orig"]),
        "logp_gold_intact": -record["kl_oh_orig"],
        "y": int(not record["is_correct_neg"]) if record["is_correct_orig"] else None,
    }
    if benchmark in MC_BENCHMARKS:
        probs = record["top_probs_orig"].values()
        columns["p_intact_max"] = max(probs)
        columns["entropy_intact"] = -sum(p * math.log(p) for p in probs if p > 0)
    return columns


def item_rows(results: Path, model: str, meta: dict, benchmark: str, args) -> list:
    """One row per question with an error-free validation-draw record."""
    tag = make_model_tag(model)
    validation, failed = read_records(
        ablation_dir(results, benchmark, tag, args.validation_draw))
    direct = question_only(results, benchmark, tag)
    robust = robustness_values(results, benchmark, tag, args)
    rows, same_indices = [], 0
    for qid, record in validation.items():
        q = dict(direct.get(qid, {}))
        gold = q.pop("ground_truth", record["ground_truth"])
        if gold != record["ground_truth"]:
            raise SystemExit(f"{model} {benchmark} {qid}: direct and ablation runs "
                             "have different ground truth")
        r_value, indices = robust.get(qid, (None, None))
        same = indices == record["sel_indices"]
        same_indices += same
        row = {"question_id": qid, "model": model, "family": meta["family"],
               "benchmark": benchmark, "scale_b": float(meta["size_b"]),
               "log_size": math.log(float(meta["size_b"])), "R": r_value,
               "same_selection": same,
               **intact_columns(record, benchmark), **q}
        row["N"] = (row["logp_gold_intact"] - row["logp_gold_q"]
                    if "logp_gold_q" in row else None)
        rows.append(row)
    print(f"{model} {benchmark}: {len(rows)} items; {len(failed)} with only error records; "
          f"{sum(q not in direct for q in validation)} without direct run; "
          f"{sum(q not in robust for q in validation)} without R; "
          f"{same_indices} where both draws selected the same sentences (R left empty in predictive tables)", file=sys.stderr)
    return rows


def parse_subtask_fields(pairs: list) -> dict:
    fields = {}
    for pair in pairs or []:
        benchmark, sep, field = pair.partition("=")
        if not sep or not field:
            raise SystemExit(f"--subtask-field expects BENCHMARK=FIELD, got {pair!r}")
        fields[benchmark] = field
    return fields


def add_subtasks(frame: pd.DataFrame, results: Path, fields: dict) -> pd.DataFrame:
    values = []
    cache = {}
    for model, benchmark, qid in zip(frame["model"], frame["benchmark"], frame["question_id"]):
        key = (model, benchmark)
        if key not in cache:
            folder = results / f"exp1_cot_{benchmark}_{make_model_tag(model)}"
            cache[key] = read_records(folder)[0]
        field = fields[benchmark]
        value = cache[key].get(qid, {}).get(field)
        if value in (None, ""):
            raise SystemExit(f"--subtask-field: CoT record {qid} of {model} has no {field!r}")
        values.append(value)
    return frame.assign(subtask=values)


def add_agree_rate(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    agree = pv.load_records(path)
    keys = ["model", "benchmark", "question_id"]
    missing = [c for c in keys + ["agree_rate"] if c not in agree.columns]
    if missing:
        raise SystemExit(f"--agree-rate is missing columns {missing}")
    agree = agree[keys + ["agree_rate"]].astype({"question_id": str})
    merged = frame.merge(agree, on=keys, how="left", validate="one_to_one")
    n_missing = int(merged["agree_rate"].isna().sum())
    if n_missing:
        raise SystemExit(f"--agree-rate has no value for {n_missing} rows")
    return merged


def _mass_rows(record: dict, prefix: str) -> list:
    """(condition, choice_mass_observed, tail_mass, n_choices_missing) of one record."""
    found = {}
    for column in ("choice_mass_observed", "tail_mass", "n_choices_missing"):
        for key, value in record.items():
            if key == column and isinstance(value, dict):
                for name, v in value.items():
                    found.setdefault(f"{prefix}_{name}", {})[column] = v
            elif key == column or key.startswith(column + "_"):
                name = key[len(column) + 1:]
                condition = f"{prefix}_{name}" if name else prefix
                if value is not None:
                    found.setdefault(condition, {})[column] = value
    if record.get("correct_tail_mass") is not None:
        found.setdefault(prefix, {})["tail_mass"] = record["correct_tail_mass"]
    return [{"condition": c, **v} for c, v in found.items()
            if "choice_mass_observed" in v or "tail_mass" in v]


def mass_table(results: Path, models: dict, benchmarks: list) -> pd.DataFrame:
    rows = []
    for model in models:
        tag = make_model_tag(model)
        for benchmark in benchmarks:
            sources = [(f"exp1_cot_{benchmark}_{tag}", "intact"),
                       (f"exp2_direct_{benchmark}_{tag}", "question_only"),
                       (f"kl_mask10_{benchmark}_{tag}", "mask10"),
                       (f"abl_sent_{benchmark}_{tag}", "sent")]
            sources += [(p.name, "sent") for p in
                        sorted(results.glob(f"abl_sent_{benchmark}_{tag}_draw*"))]
            for folder, prefix in sources:
                if not (results / folder).is_dir():
                    continue
                _, sep, draw = folder.rpartition(f"{tag}_draw")
                for qid, record in read_records(results / folder)[0].items():
                    for row in _mass_rows(record, prefix):
                        if sep:
                            row["condition"] += f"_draw{draw}"
                        rows.append({"model": model, "benchmark": benchmark,
                                     "question_id": qid, **row})
    if not rows:
        raise SystemExit(f"no retained-mass fields found under {results}")
    columns = ["model", "benchmark", "condition", "question_id",
               "choice_mass_observed", "tail_mass", "n_choices_missing"]
    return pd.DataFrame(rows).reindex(columns=columns)


def check_arguments(args) -> dict:
    """Fail before reading anything when a needed definition was not given."""
    features = [t for t in args.tables if t != "retained_mass"]
    confidence = [t for t in args.tables if t in CONFIDENCE_TABLES]
    fields = parse_subtask_fields(args.subtask_field)
    if features:
        if args.validation_draw is None or args.robustness is None:
            raise SystemExit(f"{features} need --validation-draw and --robustness")
        if args.robustness != "mask10":
            if args.robustness_draw is None:
                raise SystemExit(f"--robustness {args.robustness} needs --robustness-draw")
            if args.robustness_draw == args.validation_draw:
                raise SystemExit("--robustness-draw must be a different draw "
                                 "from --validation-draw")
    if confidence:
        open_ended = [b for b in args.benchmarks if b in OPEN_DIRECT]
        if open_ended:
            raise SystemExit(f"{confidence}: p_intact_max, entropy_intact, p_q_max and "
                             f"correct_qonly are not defined for {open_ended}")
        if not args.agree_rate:
            raise SystemExit(f"{confidence} need agree_rate, which the runners do not "
                             "define; pass --agree-rate")
    if any(t in SCALE_TABLES for t in args.tables):
        missing = [b for b in args.benchmarks if b not in fields]
        if missing:
            raise SystemExit(f"pass --subtask-field BENCHMARK=FIELD for {missing}")
    return fields


def build_tables(args) -> dict:
    fields = check_arguments(args)
    results, models = Path(args.results), load_models(args.models)
    tables = {}
    if "retained_mass" in args.tables:
        tables["retained_mass"] = mass_table(results, models, args.benchmarks)
    wanted = [t for t in args.tables if t != "retained_mass"]
    if not wanted:
        return tables
    frame = pd.DataFrame([row for model, meta in models.items() for benchmark in args.benchmarks
                          for row in item_rows(results, model, meta, benchmark, args)])
    if any(t in CONFIDENCE_TABLES for t in wanted):
        records = add_agree_rate(frame, args.agree_rate)
        records.loc[records["same_selection"], "R"] = float("nan")
        records = records.reindex(columns=list(pv.REQUIRED_COLUMNS))
        tables.update({t: records for t in wanted if t in CONFIDENCE_TABLES})
    if any(t in SCALE_TABLES for t in wanted):
        long = add_subtasks(frame, results, fields).rename(
            columns={"N": "necessity", "R": "robustness"})
        long = long.reindex(columns=list(table4.REQUIRED))
        tables.update({t: long for t in wanted if t in SCALE_TABLES})
    return tables


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results", help="directory of runner outputs")
    p.add_argument("--models", required=True,
                   help='JSON: {"<model>": {"family": str, "size_b": number}}')
    p.add_argument("--benchmarks", nargs="+", required=True, choices=BENCHMARKS)
    p.add_argument("--out", required=True)
    p.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    p.add_argument("--validation-draw", type=int, default=None,
                   help="sentence ablation draw that gives the intact baseline and y")
    p.add_argument("--robustness", choices=ROBUSTNESS, default=None,
                   help="which perturbation's JS is R")
    p.add_argument("--robustness-draw", type=int, default=None,
                   help="sentence ablation draw used for R (not for mask10)")
    p.add_argument("--agree-rate", default=None,
                   help="csv/parquet: model, benchmark, question_id, agree_rate")
    p.add_argument("--subtask-field", nargs="+", default=None, metavar="BENCHMARK=FIELD",
                   help="CoT record field used as subtask, per benchmark")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    tables = build_tables(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"{out / (name + '.csv')}: {len(frame)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
