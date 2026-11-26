"""Empty logprobs raise during teacher forcing and the item is stored as an errored record (no network)."""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for sub in ("additional_experiments/benchmarks", "additional_experiments/semantic_and_structural"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

import llm_client
import run_math_qwen3
import run_math_test

RUNNERS = [run_math_test, run_math_qwen3]
KL_KEYS = ("kl_oh_cot", "kl_oh_cot_m", "js_cot_cotm", "kl_oh_dup", "kl_oh_dup_m",
           "js_dup_dupm", "kl_oh_sm", "kl_oh_sm_m", "js_sm_smm", "tail_mass")
ROW = {"problem": "What is 1+1?", "answer": "42"}
MESSAGES = [{"role": "user", "content": "q"}]


class FakeTokenizer:
    def encode(self, text):
        return list(range(len(text)))

    def decode(self, ids):
        return "x"


class EmptyLogprobsClient:
    """Every request succeeds but carries no logprobs."""

    def __init__(self):
        message = SimpleNamespace(content="ans")
        choice = SimpleNamespace(logprobs=None, text="", message=message)
        response = SimpleNamespace(choices=[choice])
        create = lambda **kw: response
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.completions = SimpleNamespace(create=create)


@pytest.fixture(params=RUNNERS, ids=lambda m: m.__name__)
def runner(request, monkeypatch):
    module = request.param
    monkeypatch.setattr(module, "PROVIDER", "openai")
    monkeypatch.setattr(module, "THINKING_MODEL", False, raising=False)
    monkeypatch.setattr(llm_client, "get_tokenizer", lambda model: FakeTokenizer())
    monkeypatch.setattr(module, "get_tokenizer", lambda model: FakeTokenizer(), raising=False)
    return module


def test_teacher_force_raises_on_empty_logprobs(runner):
    with pytest.raises(RuntimeError, match="token 0"):
        runner.teacher_force(EmptyLogprobsClient(), MESSAGES, "42")


def test_extract_distribution_raises_on_empty_input(runner):
    with pytest.raises(ValueError):
        runner.extract_distribution([])


def test_direct_record_has_error_and_no_metric(runner, tmp_path, capsys):
    out = tmp_path / "direct.jsonl"
    rec = runner.process_direct(EmptyLogprobsClient(), ROW, 7, out)
    assert rec["error"] and "token 0" in rec["error"]
    assert rec["kl_oh_direct"] is None
    assert json.loads(out.read_text())["error"] == rec["error"]
    assert "[ERROR]" in capsys.readouterr().err


def test_kl_record_has_error_and_none_metrics(runner, tmp_path, capsys):
    out = tmp_path / "kl.jsonl"
    cot = {"reasoning": "step one\n\nstep two\n\nstep three", "is_correct": True}
    rec = runner.process_kl(EmptyLogprobsClient(), ROW, 3, out, cot)
    assert rec["error"] and "token 0" in rec["error"]
    assert all(rec[k] is None for k in KL_KEYS)
    stored = json.loads(out.read_text())
    assert stored["error"] == rec["error"]
    assert "[ERROR]" in capsys.readouterr().err
    assert runner.load_completed_ids(out) == set()


FILES = [
    "run_math_test.py",
    "run_math_qwen3.py",
    "additional_experiments/semantic_and_structural/run_paraphrase_math.py",
    "additional_experiments/semantic_and_structural/run_paraphrase_omnimath.py",
    "additional_experiments/benchmarks/run_omnimath.py",
]


@pytest.mark.parametrize("rel", FILES)
def test_no_empty_distribution_fallback(rel):
    source = (ROOT / rel).read_text()
    assert not re.search(r"\{\}\s*,\s*0\.0", source)
