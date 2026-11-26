"""``--top_logprobs`` reaches ``measures.MC_TOP_LOGPROBS`` / ``TOP_LOGPROBS`` without any API call."""
import importlib
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import measures

ROOT = Path(__file__).resolve().parent.parent
for sub in ("additional_experiments/benchmarks", "additional_experiments/semantic_and_structural"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

MC_RUNNERS = [
    "run_gpqa", "run_kl_mask10_gpqa", "run_mmlu_full", "run_mmlu_qwen3",
    "run_agieval", "run_mmlu_pro", "run_sentence_ablation", "run_sentence_ablation_all",
    "run_paraphrase", "run_paraphrase_gpqa", "run_full_paraphrase_gpqa",
]
OPEN_RUNNERS = [
    "run_math_qwen3", "run_math_test", "run_gsm8k", "run_omnimath", "run_omnimath_kl",
    "run_paraphrase_math", "run_paraphrase_omnimath", "run_js_cot_direct",
    "run_kl_dup_para", "run_kl_paraphrase", "run_kl_stepmask_para", "run_kl_duplicate_step",
]


class _Stop(Exception):
    pass


def _stop(*args, **kwargs):
    raise _Stop


def _run_main(monkeypatch, module, argv):
    """Run ``module.main()`` until the first client/dataset/network access."""
    def no_network(*args, **kwargs):
        raise _Stop
    monkeypatch.setattr(socket.socket, "connect", no_network)
    for name in ("create_client", "load_dataset", "load_data",
                 "run_paraphrase_experiment", "run_experiment"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _stop)
    monkeypatch.setattr(sys, "argv", [module.__name__] + argv)
    with pytest.raises(_Stop):
        module.main()


def test_default_constant_matches_paper_setting():
    assert measures.MC_TOP_LOGPROBS == 200


@pytest.mark.parametrize("name", MC_RUNNERS)
def test_mc_runner_sets_measures_constant(monkeypatch, name):
    module = importlib.import_module(name)
    monkeypatch.setattr(measures, "MC_TOP_LOGPROBS", measures.MC_TOP_LOGPROBS)
    extra = ["--dataset", "gpqa"] if name == "run_sentence_ablation_all" else []
    _run_main(monkeypatch, module, ["--top_logprobs", "37"] + extra)
    assert measures.MC_TOP_LOGPROBS == 37


@pytest.mark.parametrize("name", MC_RUNNERS)
def test_mc_runner_default_equals_constant(monkeypatch, name):
    module = importlib.import_module(name)
    monkeypatch.setattr(measures, "MC_TOP_LOGPROBS", 200)
    extra = ["--dataset", "gpqa"] if name == "run_sentence_ablation_all" else []
    _run_main(monkeypatch, module, extra)
    assert measures.MC_TOP_LOGPROBS == 200


@pytest.mark.parametrize("name", OPEN_RUNNERS)
def test_open_ended_runner_sets_module_constant(monkeypatch, name):
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "TOP_LOGPROBS", module.TOP_LOGPROBS)
    assert module.TOP_LOGPROBS == 200
    _run_main(monkeypatch, module, ["--top_logprobs", "41", "--vocab_size", "1000"])
    assert module.TOP_LOGPROBS == 41


def test_request_sites_read_constant_at_call_time(monkeypatch):
    """The constant is read when a request is built, so a runtime change is honoured."""
    import run_mmlu_full as R
    seen = []

    def create(**kw):
        seen.append(kw["top_logprobs"])
        raise _Stop

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(measures, "MC_TOP_LOGPROBS", 53)
    R.get_answer_dist(client, [{"role": "user", "content": "q"}])
    assert seen == [53]


def test_run_config_records_requested(tmp_path):
    p = measures.write_run_config(tmp_path / "x.jsonl", model="m", top_logprobs=200)
    assert json.loads(p.read_text())["top_logprobs_requested"] == 200
