"""Multiple-choice prediction is the most probable valid label, not the first generated token."""
import math
from types import SimpleNamespace

import pytest

import measures
from runner_stubs import client, response

LABELS = ("A", "B", "C", "D")
TOKENS = (("**", 0.1), ("B", 0.7875), ("A", 0.1125))


def _content(tokens, first):
    return response("", tokens, first).choices[0].logprobs.content


def test_predicted_is_argmax_over_labels():
    _, predicted, info = measures.extract_choice_probs(_content(TOKENS, "**"), LABELS)
    assert predicted == "B"
    assert info["first_token"] == "**"


def test_filled_label_is_never_predicted():
    content = _content((("x", 0.5), ("C", 1e-14), ("y", 1e-16)), "x")
    assert measures.extract_choice_probs(content, LABELS)[1] == "C"


def test_no_valid_label_still_raises():
    with pytest.raises(ValueError):
        measures.extract_choice_probs(_content((("**", 0.9),), "**"), LABELS)


def test_runner_record_has_label_and_raw_token(tmp_path):
    import run_mmlu_full as R
    row = {"subject": "x", "question": "q", "choices": list("abcd"), "answer": 1}
    rec = R.process_direct(client(lambda **kw: response("**", TOKENS, "**")), row, 0,
                           tmp_path / "out.jsonl")
    assert rec["predicted"] == rec["ground_truth"] == "B"
    assert rec["first_token"] == "**"
    assert math.isclose(rec["correct_token_prob"], rec["top_probs"]["B"])


RECORD_WRITERS = [
    "run_gpqa.py", "run_kl_mask10_gpqa.py", "run_mmlu_full.py", "run_mmlu_qwen3.py",
    "additional_experiments/benchmarks/run_agieval.py",
    "additional_experiments/benchmarks/run_mmlu_pro.py",
    "additional_experiments/semantic_and_structural/run_paraphrase.py",
    "additional_experiments/semantic_and_structural/run_paraphrase_gpqa.py",
    "additional_experiments/semantic_and_structural/run_full_paraphrase_gpqa.py",
]


@pytest.mark.parametrize("rel", RECORD_WRITERS)
def test_every_predicted_record_keeps_first_token(rel):
    from runner_stubs import ROOT
    source = (ROOT / rel).read_text()
    assert source.count('"predicted": predicted,') == source.count('"first_token": ') > 0
