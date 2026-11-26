"""Vocabulary size: --vocab_size, else a known model family, else an error."""
import importlib
import sys

import pytest

import measures
from runner_stubs import ROOT  # noqa: F401  (adds runner directories to sys.path)

OPEN_VOCAB_RUNNERS = [
    "run_math_test", "run_math_qwen3", "run_gsm8k", "run_omnimath", "run_omnimath_kl",
    "run_paraphrase_math", "run_paraphrase_omnimath", "run_js_cot_direct",
    "run_kl_dup_para", "run_kl_paraphrase", "run_kl_stepmask_para", "run_kl_duplicate_step",
    "run_sentence_ablation_all",
]
JUDGED_RUNNERS = OPEN_VOCAB_RUNNERS[:7]
UNKNOWN = "acme/unknown-7b"


@pytest.mark.parametrize("model,size", [
    ("google/gemma-3-4b-it", 262144),
    ("google/gemma-2-9b-it", 256000),
    ("meta-llama/Llama-3.1-8B-Instruct", 128256),
    ("mistralai/Ministral-3-3B-Instruct-2512", 131072),
])
def test_known_families(model, size):
    assert measures.resolve_vocab_size(model) == size


def test_explicit_value_wins():
    assert measures.resolve_vocab_size("google/gemma-3-4b-it", 99) == 99
    assert measures.resolve_vocab_size(UNKNOWN, 99) == 99


@pytest.mark.parametrize("model", [UNKNOWN, "Qwen/Qwen3-8B", "gpt-4o-mini"])
def test_unknown_model_requires_flag(model):
    with pytest.raises(ValueError, match="--vocab_size"):
        measures.resolve_vocab_size(model)


def test_non_positive_value_rejected():
    with pytest.raises(ValueError):
        measures.resolve_vocab_size(UNKNOWN, 0)


class _Stop(Exception):
    pass


def _run_to_config(monkeypatch, tmp_path, name, argv):
    """Run ``main()`` up to the first ``run_config.json`` write and return its fields."""
    module = importlib.import_module(name)
    seen = {}

    def capture(path, **fields):
        seen.update(fields)
        raise _Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(measures, "write_run_config", capture)
    stubs = {
        "create_client": lambda *a: None,
        "load_dataset": lambda *a, **k: [],
        "load_data": lambda: [],
        "load_cot_results": lambda *a: {"q": {}},
        "load_paraphrase_results": lambda *a: {"q": {}},
        "get_cot_source_dir": lambda *a: tmp_path,
        "DATA_PATH": str(tmp_path / "data.jsonl"),
    }
    for attr, value in stubs.items():
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, value)
    (tmp_path / "data.jsonl").write_text("")
    if name == "run_sentence_ablation_all":
        tag = module.make_model_tag(argv[argv.index("--model") + 1])
        cot = tmp_path / "results" / f"exp1_cot_math_t_{tag}" / "math_test.jsonl"
        cot.parent.mkdir(parents=True)
        cot.write_text("")
        monkeypatch.setitem(module.DATASETS["math_t"], "load", lambda: [])
        argv = argv + ["--dataset", "math_t"]
    for attr in ("MODEL", "PROVIDER", "VOCAB_SIZE", "JUDGE", "OUTDIR"):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, getattr(module, attr))
    monkeypatch.setattr(sys, "argv", [name, "--provider", "litellm"] + argv)
    with pytest.raises(_Stop):
        module.main()
    return seen


@pytest.mark.parametrize("name", OPEN_VOCAB_RUNNERS)
def test_runner_records_vocab_size(monkeypatch, tmp_path, name):
    cfg = _run_to_config(monkeypatch, tmp_path, name, ["--model", UNKNOWN, "--vocab_size", "999"])
    assert cfg["vocab_size"] == 999


@pytest.mark.parametrize("name", OPEN_VOCAB_RUNNERS)
def test_runner_rejects_unknown_model(monkeypatch, tmp_path, name):
    with pytest.raises(ValueError, match="--vocab_size"):
        _run_to_config(monkeypatch, tmp_path, name, ["--model", UNKNOWN])


@pytest.mark.parametrize("name", JUDGED_RUNNERS)
@pytest.mark.parametrize("flags,expected", [([], "gpt-4.1-nano"), (["--judge_model", "none"], "none")])
def test_runner_records_judge_model(monkeypatch, tmp_path, name, flags, expected):
    cfg = _run_to_config(monkeypatch, tmp_path, name,
                         ["--model", UNKNOWN, "--vocab_size", "999"] + flags)
    assert cfg["judge_model"] == expected


@pytest.mark.parametrize("name", ["run_omnimath", "run_paraphrase_math", "run_paraphrase_omnimath"])
def test_openai_provider_keeps_tiktoken_size(monkeypatch, name):
    import llm_client
    from types import SimpleNamespace
    monkeypatch.setattr(llm_client, "get_tokenizer", lambda model: SimpleNamespace(n_vocab=1234))
    assert llm_client.resolve_tiktoken_vocab_size("openai", "gpt-4o-mini") == 1234
    assert llm_client.resolve_tiktoken_vocab_size("openai", "gpt-4o-mini", 7) == 7
    with pytest.raises(ValueError, match="--vocab_size"):
        llm_client.resolve_tiktoken_vocab_size("litellm", UNKNOWN)
