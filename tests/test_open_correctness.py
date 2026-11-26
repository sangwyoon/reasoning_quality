"""Open-ended runners decide correctness with measures.judge_correct."""
import importlib
import re

import openai
import pytest

import measures
from runner_stubs import ROOT, client, response

GOLD = "1/2"


def _row(name):
    return {"run_math_test": {"problem": "p", "answer": GOLD},
            "run_math_qwen3": {"problem": "p", "answer": GOLD},
            "run_gsm8k": {"question": "p", "answer": f"#### {GOLD}"},
            "run_omnimath": {"problem": "p", "answer": GOLD},
            "run_omnimath_kl": {"problem": "p", "answer": GOLD},
            "run_paraphrase_math": {"problem": "p", "solution": f"\\boxed{{{GOLD}}}"},
            "run_paraphrase_omnimath": {"problem": "p", "answer": GOLD}}[name]


RUNNERS = ["run_math_test", "run_math_qwen3", "run_gsm8k", "run_omnimath", "run_omnimath_kl",
           "run_paraphrase_math", "run_paraphrase_omnimath"]


class _Tokenizer:
    def encode(self, text):
        return [0]

    def decode(self, ids):
        return GOLD


def _is_correct(monkeypatch, tmp_path, name, answer, judge_model, judge_create):
    """Process one item with a fake model; the judge client is ``openai.OpenAI``."""
    module = importlib.import_module(name)
    judge_calls = []

    def create(**kw):
        judge_calls.append(kw["model"])
        return judge_create(**kw)

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: client(create))
    monkeypatch.setattr(module, "JUDGE", measures.openai_judge(judge_model), raising=False)
    monkeypatch.setattr(module, "PROVIDER", "openai")
    monkeypatch.setattr(module, "VOCAB_SIZE", 1000, raising=False)
    monkeypatch.setattr(module, "THINKING_MODEL", False, raising=False)
    import llm_client
    monkeypatch.setattr(llm_client, "get_tokenizer", lambda model: _Tokenizer())
    monkeypatch.setattr(module, "get_tokenizer", lambda model: _Tokenizer(), raising=False)

    model_calls = []

    def generate(**kw):
        model_calls.append(kw["messages"][-1]["content"])
        return response(answer)

    out = tmp_path / f"{name}.jsonl"
    fake = client(generate)
    if name.startswith("run_paraphrase"):
        monkeypatch.setattr(module, "paraphraser", None)
        record = module.process_paraphrase(
            fake, "model-under-test", _row(name), 0, out, {"reasoning": "a\n\nb"},
            1, 42, "mask", mask_ratio=0.1)
    else:
        record = module.process_cot(fake, _row(name), 0, out)
    assert not any("equivalence checker" in m for m in model_calls)
    return record, judge_calls


def _says(reply):
    return lambda **kw: response(reply)


def _down(**kw):
    raise RuntimeError("judge down")


@pytest.mark.parametrize("name", RUNNERS)
def test_judge_decides_even_numeric_matches(monkeypatch, tmp_path, name):
    rec, calls = _is_correct(monkeypatch, tmp_path, name, "0.5", "gpt-4.1-nano", _says("no"))
    assert rec["is_correct"] is False and calls == ["gpt-4.1-nano"]


@pytest.mark.parametrize("name", RUNNERS)
def test_undecided_answer_goes_to_judge_model(monkeypatch, tmp_path, name):
    rec, calls = _is_correct(monkeypatch, tmp_path, name, "one half", "judge-x", _says("yes"))
    assert rec["is_correct"] is True and calls == ["judge-x"]


@pytest.mark.parametrize("name", RUNNERS)
def test_judge_none_disables_judge(monkeypatch, tmp_path, name):
    rec, calls = _is_correct(monkeypatch, tmp_path, name, "one half", "none", _down)
    assert rec["is_correct"] is False and rec["error"] is None and calls == []


@pytest.mark.parametrize("name", RUNNERS)
def test_judge_failure_is_error_record(monkeypatch, tmp_path, name):
    rec, _ = _is_correct(monkeypatch, tmp_path, name, "one half", "gpt-4.1-nano", _down)
    assert rec["is_correct"] is None and "judge down" in rec["error"]


def test_openai_judge_none_and_default():
    assert measures.openai_judge("none") is None
    assert callable(measures.openai_judge(measures.DEFAULT_JUDGE_MODEL))


def test_no_runner_bypasses_the_shared_judge():
    offenders = [p.name for p in ROOT.rglob("run_*.py")
                 if re.search(r"judge_equivalent|def judge_answer", p.read_text())]
    assert offenders == []
