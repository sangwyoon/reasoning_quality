"""Runner outputs (fake client) -> analysis.assemble -> every analysis CLI."""
import hashlib
import json

import pytest

try:
    import sklearn  # noqa: F401
except Exception as exc:  # missing or binary-incompatible install
    pytest.skip(f"scikit-learn unusable: {exc}", allow_module_level=True)

import pandas as pd

from runner_stubs import client, response
from analysis import assemble, flagging, predictive_value, retained_mass, scale_fits, table4

MODELS = {"org/fam-1b": 1.0, "org/fam-3b": 3.0, "org/fam-9b": 9.0}
N_QUESTIONS = 40


def _unit(*key):
    return int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _fake_model(model):
    def create(**kw):
        messages = kw["messages"]
        if not kw.get("logprobs"):
            return response(f"Fact one about {messages[0]['content'][-400:]}. Fact two.\n\n"
                            "Fact three.")
        key = [m["content"] for m in messages]
        p_gold = 0.3 + 0.65 * _unit(model, key)
        if len(messages) > 1 and "NOT " in messages[1]["content"] and _unit(key, model) < 0.5:
            p_gold = 0.05
        rest = 0.9 * (1 - p_gold)
        tokens = sorted((("A", p_gold), ("B", rest * 0.5), ("C", rest * 0.3), ("D", rest * 0.2)),
                        key=lambda t: -t[1])
        return response(tokens[0][0], tuple(tokens))
    return client(create)


def _editor(prompt):
    instruction, sentence = prompt.split("\n", 1)
    return ("NOT " if "invert" in instruction else "SAME ") + sentence


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    import llm_client
    import run_mmlu_full as M
    import run_sentence_ablation_all as A

    root = tmp_path_factory.mktemp("assemble")
    rows = [{"subject": f"s{i % 4}", "question": f"question {i}", "choices": list("abcd"),
             "answer": 0} for i in range(N_QUESTIONS)]
    previous_model, previous_path = (M.MODEL, A.MODEL), A.DATASETS["mmlu"]["load"]
    (root / "data").mkdir()
    (root / "data" / "mmlu_full.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        ablation_rows = previous_path()
    for model in MODELS:
        tag = llm_client.make_model_tag(model)
        fake = _fake_model(model)
        M.MODEL = A.MODEL = model
        base = root / "results"
        paths = {
            "cot": base / f"exp1_cot_mmlu_{tag}" / "mmlu.jsonl",
            "direct": base / f"exp2_direct_mmlu_{tag}" / "mmlu.jsonl",
            "kl": base / f"kl_mask10_mmlu_{tag}" / "kl_mask10.jsonl",
            0: base / f"abl_sent_mmlu_{tag}" / "abl_sent.jsonl",
            1: base / f"abl_sent_mmlu_{tag}_draw1" / "abl_sent.jsonl",
        }
        for path in paths.values():
            path.parent.mkdir(parents=True)
        for i, row in enumerate(rows):
            cot = M.process_cot(fake, row, i, paths["cot"])
            M.process_direct(fake, row, i, paths["direct"])
            M.process_kl(fake, row, i, paths["kl"], cot)
            for draw in (0, 1):
                A.process_item(fake, _editor, ablation_rows[i], cot, paths[draw], draw)
    M.MODEL, A.MODEL = previous_model
    (root / "models.json").write_text(json.dumps(
        {m: {"family": "fam", "size_b": size} for m, size in MODELS.items()}))
    agree = [{"model": m, "benchmark": "mmlu", "question_id": f"mmlu_{i}",
              "agree_rate": _unit(m, i)} for m in MODELS for i in range(N_QUESTIONS)]
    pd.DataFrame(agree).to_csv(root / "agree.csv", index=False)
    return root


def _args(results, out, *extra):
    return ["--results", str(results / "results"), "--models", str(results / "models.json"),
            "--benchmarks", "mmlu", "--out", str(out), *extra]


def _full(results, out, *extra):
    return _args(results, out, "--validation-draw", "1", "--robustness", "mask10",
                 "--subtask-field", "mmlu=subject",
                 "--agree-rate", str(results / "agree.csv"), *extra)


def test_assembled_tables_feed_every_analysis_cli(results, tmp_path):
    out = tmp_path / "tables"
    assert assemble.main(_full(results, out)) == 0
    records = pd.read_csv(out / "predictive_value.csv")
    assert set(predictive_value.REQUIRED_COLUMNS) <= set(records.columns)
    assert len(records) == len(MODELS) * N_QUESTIONS
    assert records["y"].nunique() == 2
    assert records.loc[records["correct_intact"] == 0, "y"].isna().all()
    assert records["N"].to_numpy() == pytest.approx(
        (records["logp_gold_intact"] - records["logp_gold_q"]).to_numpy())

    runs = [
        (predictive_value, "predictive_value", ["--n-boot", "20", "--n-splits", "2",
                                                "--c-grid", "1", "--inner-folds", "3"]),
        (flagging, "flagging", ["--n-boot", "20"]),
        (retained_mass, "retained_mass", []),
        (table4, "table4", ["--n-boot", "20"]),
        (scale_fits, "scale_fits", ["--n-boot", "20"]),
    ]
    for module, name, extra in runs:
        assert module.main(["--input", str(out / f"{name}.csv"),
                            "--out", str(tmp_path / name), *extra]) == 0
    fits = json.loads((tmp_path / "scale_fits" / "scale_fits.json").read_text())
    assert any(row["fit"] for row in fits["fits"]) and fits["subtask_bootstrap"]


def test_values_come_from_the_expected_records(results, tmp_path):
    out = tmp_path / "tables"
    assemble.main(_full(results, out))
    tag = "fam-1b"
    base = results / "results"
    val = json.loads((base / f"abl_sent_mmlu_{tag}_draw1" / "abl_sent.jsonl")
                     .read_text().splitlines()[0])
    mask = json.loads((base / f"kl_mask10_mmlu_{tag}" / "kl_mask10.jsonl")
                      .read_text().splitlines()[0])
    records = pd.read_csv(out / "predictive_value.csv")
    row = records[(records["model"] == "org/fam-1b")
                  & (records["question_id"] == val["question_id"])].iloc[0]
    assert row["R"] == pytest.approx(mask["js_cot_cotm"])
    assert row["p_intact_max"] == pytest.approx(max(val["top_probs_orig"].values()))
    if val["is_correct_orig"]:
        assert row["y"] == int(not val["is_correct_neg"])
    mass = pd.read_csv(out / "retained_mass.csv")
    assert {"intact", "question_only", "mask10_cot", "mask10_cot_m",
            "sent_neg", "sent_neg_draw1"} <= set(mass["condition"])


def test_ablation_robustness_comes_from_the_other_draw(results, tmp_path):
    out = tmp_path / "tables"
    args = _args(results, out, "--validation-draw", "1", "--robustness", "para",
                 "--robustness-draw", "0", "--subtask-field", "mmlu=subject",
                 "--tables", "table4")
    assert assemble.main(args) == 0
    table = pd.read_csv(out / "table4.csv")
    assert set(table.columns) >= set(table4.REQUIRED)


@pytest.mark.parametrize("extra,message", [
    (["--validation-draw", "1", "--robustness", "neg", "--robustness-draw", "1",
      "--tables", "table4", "--subtask-field", "mmlu=subject"], "different draw"),
    (["--validation-draw", "1", "--robustness", "neg",
      "--tables", "table4", "--subtask-field", "mmlu=subject"], "--robustness-draw"),
    (["--validation-draw", "1", "--robustness", "mask10",
      "--tables", "predictive_value"], "--agree-rate"),
    (["--validation-draw", "1", "--robustness", "mask10", "--tables", "table4"],
     "--subtask-field"),
    (["--tables", "table4", "--subtask-field", "mmlu=subject"], "--validation-draw"),
    (["--validation-draw", "7", "--robustness", "mask10", "--tables", "table4",
      "--subtask-field", "mmlu=subject"], "abl_sent_mmlu"),
])
def test_undetermined_inputs_are_errors(results, tmp_path, extra, message):
    with pytest.raises(SystemExit, match=message):
        assemble.main(_args(results, tmp_path / "t", *extra))


def test_open_ended_benchmark_has_no_confidence_columns(results, tmp_path):
    args = _full(results, tmp_path / "t", "--tables", "flagging")
    args[args.index("--benchmarks") + 1] = "gsm8k"
    with pytest.raises(SystemExit, match="p_intact_max"):
        assemble.main(args)


def test_retained_mass_needs_no_feature_definitions(results, tmp_path):
    assert assemble.main(_args(results, tmp_path / "t", "--tables", "retained_mass")) == 0


def test_identical_selections_leave_r_empty_in_predictive_tables(results, tmp_path, capsys):
    out = tmp_path / "tables"
    assemble.main(_args(results, out, "--validation-draw", "1", "--robustness", "neg",
                        "--robustness-draw", "0", "--agree-rate", str(results / "agree.csv"),
                        "--tables", "predictive_value"))
    records = pd.read_csv(out / "predictive_value.csv")
    base = results / "results"
    same = set()
    for model in MODELS:
        tag = assemble.make_model_tag(model)
        draws = [{r["question_id"]: r["sel_indices"] for r in map(json.loads, (
            base / f"abl_sent_mmlu_{tag}{suffix}" / "abl_sent.jsonl").read_text().splitlines())}
            for suffix in ("", "_draw1")]
        same |= {(model, q) for q in draws[0] if draws[0][q] == draws[1].get(q)}
    assert same
    empty = {(m, q) for m, q, r in zip(records["model"], records["question_id"], records["R"])
             if pd.isna(r)}
    assert empty == same
