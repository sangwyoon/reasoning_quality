"""Sentence ablation runners use the semantic_edits protocol (fake clients, no network)."""
import importlib
import json
import sys

import pytest

import measures
import semantic_edits
from runner_stubs import client, response

TRACE = "First fact holds. Second fact holds.\nThird fact holds. Fourth fact holds.\n\n" \
        "Fifth fact holds."
MC_ROW = {"question_id": "gpqa_0", "mode": "mcq", "letters": "ABCD", "prompt": "Q?",
          "followup": "letter:"}
MC_TOKENS = (("A", 0.7), ("B", 0.1), ("C", 0.1), ("D", 0.1))


def _editor(prompt):
    instruction, sentence = prompt.split("\n", 1)
    return ("NOT " if "invert" in instruction else "SAME ") + sentence


def _recording_client(calls, tokens=MC_TOKENS, content="A"):
    def create(**kw):
        calls.append(kw)
        return response(content, tokens)
    return client(create)


def _run(tmp_path, row=MC_ROW, cot=None, editor=_editor, draw=0, calls=None, **kw):
    import run_sentence_ablation_all as R
    calls = [] if calls is None else calls
    out = tmp_path / "abl.jsonl"
    cot = cot or {"reasoning": TRACE, "ground_truth": "A"}
    rec = R.process_item(_recording_client(calls, **kw), editor, row, cot, out, draw)
    return R, rec, calls, out


def test_variants_edit_the_same_sentences_selected_by_semantic_edits(tmp_path):
    import run_sentence_ablation_all as R
    variants, built = R.make_variants(_editor, TRACE, "gpqa_0", 0)
    expected = semantic_edits.build_draw(TRACE, R.item_seed("gpqa_0"), _editor, draw=0)
    assert built["indices"] == expected["indices"]
    assert len(built["indices"]) == semantic_edits.n_to_select(5)
    assert built["original"] == TRACE
    segments = semantic_edits.split_sentences(TRACE)
    (i,) = built["indices"]
    core = segments[i].strip()
    assert variants["para"] == TRACE.replace(core, "SAME " + core)
    assert variants["neg"] == TRACE.replace(core, "NOT " + core)
    assert variants["del"] == "".join(s for k, s in enumerate(segments) if k != i)
    assert variants["mask"] == R._mask_tokens(TRACE, "gpqa_0", 0)
    assert all(m in (o, R.UNK) for o, m in zip(TRACE.split(), variants["mask"].split()))
    assert len(variants["mask"].split()) == len(TRACE.split())


def test_draws_are_separate_random_selections():
    trace = " ".join(f"Sentence number {i} is here." for i in range(40))
    a = semantic_edits.build_draw(trace, 7, _editor, draw=0)
    b = semantic_edits.build_draw(trace, 7, _editor, draw=1)
    assert a["seed"] != b["seed"] and a["indices"] != b["indices"]
    assert a == semantic_edits.build_draw(trace, 7, _editor, draw=0)
    full = semantic_edits.build_perturbation_set(trace, 7, _editor, n_samples=2)
    assert [s["indices"] for s in full["samples"]] == [a["indices"], b["indices"]]


def test_mc_record_remeasures_baseline_and_stores_predictions(tmp_path):
    _, rec, calls, _ = _run(tmp_path, cot={"reasoning": TRACE, "ground_truth": "A",
                                           "top_probs": {"B": 1.0}})
    assert rec["error"] is None
    assert [c["messages"][1]["content"] for c in calls][0] == TRACE
    assert len(calls) == 5
    assert rec["predicted_orig"] == "A" and rec["is_correct_orig"] is True
    assert rec["top_probs_orig"]["A"] == pytest.approx(0.7)
    assert rec["draw"] == 0 and rec["n_trace_tokens"] == len(TRACE.split())
    for v in ("del", "mask", "para", "neg"):
        assert rec[f"predicted_{v}"] == "A" and rec[f"is_correct_{v}"] is True
        assert rec[f"js_{v}"] == pytest.approx(0.0)


def test_records_without_predictions_are_rerun(tmp_path):
    import run_sentence_ablation_all as R
    out = tmp_path / "abl.jsonl"
    out.write_text(json.dumps({"question_id": "gpqa_0", "js_neg": 0.1}) + "\n")
    assert R.load_completed_ids(out) == set()


def test_top_logprobs_applies_to_both_answer_kinds(monkeypatch):
    import argparse
    import run_sentence_ablation_all as R
    for attr in ("MODEL", "PROVIDER", "TOP_LOGPROBS"):
        monkeypatch.setattr(R, attr, getattr(R, attr))
    monkeypatch.setattr(measures, "MC_TOP_LOGPROBS", 200)
    parser = argparse.ArgumentParser()
    R.add_arguments(parser)
    R.configure(parser.parse_args(["--top_logprobs", "37"]))
    assert R.TOP_LOGPROBS == measures.MC_TOP_LOGPROBS == 37


def test_edit_failure_is_error_record_and_rerun(tmp_path):
    R, rec, calls, out = _run(tmp_path, editor=lambda prompt: "")
    assert "empty response" in rec["error"] and calls == []
    assert all(rec[k] is None for k in R.ITEM_METRIC_KEYS)
    assert R.load_completed_ids(out) == set()


def test_open_ended_record_stores_answer_and_correctness(tmp_path, monkeypatch):
    import run_sentence_ablation_all as R
    monkeypatch.setattr(R, "PROVIDER", "openai")
    monkeypatch.setattr(R, "VOCAB_SIZE", 1000)
    monkeypatch.setattr(R, "JUDGE", None)

    class Tok:
        def encode(self, text):
            return [0]

        def decode(self, ids):
            return "42"

    monkeypatch.setattr(R, "get_tokenizer", lambda model: Tok())
    row = {"question_id": "gsm_0", "mode": "open", "prompt": "p", "followup": "f",
           "answer_text": "42", "gen_max_tokens": 100}
    _, rec, calls, _ = _run(tmp_path, row=row, cot={"reasoning": TRACE, "ground_truth": "42"},
                            tokens=(("42", 0.5),), content="42")
    assert rec["error"] is None
    for v in ("orig", "del", "mask", "para", "neg"):
        assert rec[f"generated_answer_{v}"] == "42" and rec[f"is_correct_{v}"] is True
    generations = [c for c in calls if not c.get("logprobs")]
    assert len(generations) == 5 and {c["max_tokens"] for c in generations} == {100}


def test_empty_generated_answer_is_error_record(tmp_path, monkeypatch):
    import run_sentence_ablation_all as R
    monkeypatch.setattr(R, "PROVIDER", "openai")
    monkeypatch.setattr(R, "VOCAB_SIZE", 1000)

    class Tok:
        def encode(self, text):
            return [0]

        def decode(self, ids):
            return "42"

    monkeypatch.setattr(R, "get_tokenizer", lambda model: Tok())
    row = {"question_id": "gsm_0", "mode": "open", "prompt": "p", "followup": "f",
           "answer_text": "42", "gen_max_tokens": 100}
    _, rec, _, _ = _run(tmp_path, row=row, cot={"reasoning": TRACE, "ground_truth": "42"},
                        tokens=(("42", 0.5),), content="")
    assert "empty answer" in rec["error"]


def test_editor_is_opus_on_an_anthropic_client(monkeypatch):
    import sys as _sys
    import types
    import run_sentence_ablation_all as R
    seen = []

    def create(**kw):
        seen.append(kw)
        return types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="edited")])

    fake = types.SimpleNamespace(
        Anthropic=lambda: types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    monkeypatch.setitem(_sys.modules, "anthropic", fake)
    monkeypatch.setattr(R, "MODEL", "model-under-test")
    assert R.make_editor(R.DEFAULT_EDITOR_MODEL)("prompt") == "edited"
    assert seen[0]["model"] == "claude-opus-4-7" and seen[0]["max_tokens"] == 256
    assert "temperature" not in seen[0]


@pytest.mark.parametrize("name,argv", [
    ("run_sentence_ablation_all", ["--dataset", "gpqa"]),
    ("run_sentence_ablation", []),
])
def test_runner_main_writes_config_and_records(monkeypatch, tmp_path, name, argv):
    import run_sentence_ablation_all as A
    module = importlib.import_module(name)
    monkeypatch.chdir(tmp_path)
    for attr in ("MODEL", "PROVIDER", "VOCAB_SIZE", "JUDGE"):
        monkeypatch.setattr(A, attr, getattr(A, attr))
    monkeypatch.setattr(measures, "MC_TOP_LOGPROBS", measures.MC_TOP_LOGPROBS)
    monkeypatch.setitem(A.DATASETS["gpqa"], "load", lambda: [MC_ROW])
    monkeypatch.setattr(module, "create_client", lambda provider: _recording_client([]))
    monkeypatch.setattr(A, "make_editor", lambda model: _editor)
    cot = tmp_path / "results" / "exp1_cot_gpqa_m" / "gpqa.jsonl"
    cot.parent.mkdir(parents=True)
    cot.write_text(json.dumps({"question_id": "gpqa_0", "reasoning": TRACE,
                               "ground_truth": "A"}) + "\n")
    monkeypatch.setattr(sys, "argv", [name, "--model", "m", "--draw", "1",
                                      "--editor_model", "editor-x"] + argv)
    module.main()
    out = tmp_path / "results" / "abl_sent_gpqa_m_draw1"
    config = json.loads((out / "run_config.json").read_text())
    assert config["editor_model"] == "editor-x" and config["draw"] == 1
    record = json.loads((out / "abl_sent.jsonl").read_text())
    assert record["error"] is None and record["editor_model"] == "editor-x"
