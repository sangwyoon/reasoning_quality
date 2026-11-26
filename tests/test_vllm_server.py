"""vLLM base URL, tokenize / detokenize, and no tokenizer substitution under --provider vllm."""
import importlib
import re

import pytest
import requests

import llm_client
import vllm_server
from runner_stubs import ROOT, client, response


class FakeHTTP:
    def __init__(self, payload, status=200):
        self.payload, self.status, self.calls = payload, status, []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append((url, json))
        return self

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def json(self):
        return self.payload


@pytest.mark.parametrize("env,expected", [
    ("http://localhost:8001/v1", "http://localhost:8001"),
    ("http://localhost:8001/v1/", "http://localhost:8001"),
    ("http://localhost:8001", "http://localhost:8001"),
    ("http://host/api/v1", "http://host/api"),
    ("http://gpu-v1:11/v1", "http://gpu-v1:11"),
])
def test_base_url_removes_only_v1_suffix(monkeypatch, env, expected):
    monkeypatch.setenv("VLLM_API_BASE", env)
    assert vllm_server.base_url() == expected


def test_tokenize_never_adds_special_tokens(monkeypatch):
    monkeypatch.setenv("VLLM_API_BASE", "http://localhost:8001/v1")
    http = FakeHTTP({"tokens": [5, 6]})
    monkeypatch.setattr(requests, "post", http)
    assert vllm_server.tokenize("m", "42") == [5, 6]
    assert http.calls == [("http://localhost:8001/tokenize",
                           {"model": "m", "prompt": "42", "add_special_tokens": False})]


def test_detokenize_returns_one_string_per_id(monkeypatch):
    http = FakeHTTP({"prompt": "4"})
    monkeypatch.setattr(requests, "post", http)
    assert vllm_server.detokenize("m", [5, 6]) == ["4", "4"]
    assert [c[1] for c in http.calls] == [{"model": "m", "tokens": [5]}, {"model": "m", "tokens": [6]}]


@pytest.mark.parametrize("http", [FakeHTTP({}, 500), FakeHTTP({"tokens": []}), FakeHTTP({})])
def test_tokenize_failure_raises(monkeypatch, http):
    monkeypatch.setattr(requests, "post", http)
    with pytest.raises((requests.HTTPError, ValueError)):
        vllm_server.tokenize("m", "42")


def test_detokenize_failure_raises(monkeypatch):
    monkeypatch.setattr(requests, "post", FakeHTTP({}, 404))
    with pytest.raises(requests.HTTPError):
        vllm_server.detokenize("m", [1])
    monkeypatch.setattr(requests, "post", FakeHTTP({}))
    with pytest.raises(KeyError):
        vllm_server.detokenize("m", [1])
    with pytest.raises(ValueError):
        vllm_server.detokenize("m", [])


TEACHER_FORCERS = [
    ("run_math_test", "teacher_force"), ("run_math_qwen3", "teacher_force"),
    ("run_gsm8k", "teacher_force"), ("run_omnimath_kl", "teacher_force"),
    ("run_js_cot_direct", "teacher_force"), ("run_kl_paraphrase", "teacher_force"),
    ("run_kl_dup_para", "teacher_force"), ("run_kl_stepmask_para", "teacher_force"),
    ("run_kl_duplicate_step", "teacher_force_distribution"),
    ("run_sentence_ablation_all", "teacher_force"),
    ("run_omnimath", "teacher_force_correct_prob"),
    ("run_paraphrase_math", "teacher_force_correct_prob"),
]


@pytest.mark.parametrize("name,func", TEACHER_FORCERS)
def test_vllm_tokenizer_failure_is_not_replaced(monkeypatch, name, func):
    module = importlib.import_module(name)
    used = []
    fake_tokenizer = lambda model: used.append(model)
    monkeypatch.setattr(llm_client, "get_tokenizer", fake_tokenizer)
    monkeypatch.setattr(module, "get_tokenizer", fake_tokenizer, raising=False)
    monkeypatch.setattr(module, "PROVIDER", "vllm")
    monkeypatch.setattr(requests, "post", FakeHTTP({}, 500))
    fake = client(lambda **kw: response("42"))
    args = [fake, [{"role": "user", "content": "q"}], "42"]
    if func == "teacher_force_correct_prob":
        args.insert(1, "m")
    with pytest.raises(requests.HTTPError):
        getattr(module, func)(*args)
    assert used == []


def test_vllm_failure_gives_error_record_that_is_rerun(monkeypatch, tmp_path):
    import run_math_test as R
    monkeypatch.setattr(R, "PROVIDER", "vllm")
    monkeypatch.setattr(requests, "post", FakeHTTP({}, 500))
    out = tmp_path / "direct.jsonl"
    rec = R.process_direct(client(lambda **kw: response("42")),
                           {"problem": "p", "answer": "42"}, 0, out)
    assert "500" in rec["error"] and rec["kl_oh_direct"] is None
    assert R.load_completed_ids(out) == set()


def test_no_character_stripping_of_base_url():
    offenders = [p.name for p in ROOT.rglob("*.py")
                 if p.parent.name != "tests" and re.search(r"""rstrip\(["']/v1""", p.read_text())]
    assert offenders == []
