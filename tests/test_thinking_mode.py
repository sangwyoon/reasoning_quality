"""--thinking-model and --mask_ratio on the benchmark and ablation runners."""
import importlib
import math
import sys
import types

import pytest

import llm_client
import vllm_server

BENCH = ["run_gsm8k", "run_omnimath_kl", "run_agieval", "run_mmlu_pro"]
OPEN = ["run_gsm8k", "run_omnimath_kl", "run_sentence_ablation_all"]


def _load(name):
    return importlib.import_module(name)


@pytest.mark.parametrize("name", BENCH + ["run_sentence_ablation_all"])
def test_runner_exposes_the_switch(name):
    source = open(_load(name).__file__).read()
    assert '"--thinking-model"' in source
    assert "thinking_model=THINKING_MODEL" in source


@pytest.mark.parametrize("name", ["run_gsm8k", "run_omnimath_kl", "run_agieval"])
def test_runner_exposes_mask_ratio(name):
    source = open(_load(name).__file__).read()
    assert '"--mask_ratio"' in source and "MASK_RATIO = args.mask_ratio" in source


def test_thinking_kwargs_and_block():
    assert llm_client.thinking_kwargs(False, True) == {}
    assert llm_client.thinking_kwargs(True, True) == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    assert llm_client.extract_think_block("<think> a </think> b") == "a"
    assert llm_client.extract_think_block("plain") == "plain"


@pytest.mark.parametrize("name", OPEN)
def test_thinking_teacher_forcing_uses_raw_prompt(monkeypatch, name):
    module = _load(name)
    monkeypatch.setattr(module, "THINKING_MODEL", True)
    monkeypatch.setattr(module, "PROVIDER", "vllm")
    monkeypatch.setattr(vllm_server, "tokenize", lambda model, text: [1, 2])
    monkeypatch.setattr(vllm_server, "detokenize", lambda model, ids: ["4", "2"])
    prompts = []

    def create(**kw):
        prompts.append(kw["prompt"])
        logprobs = types.SimpleNamespace(top_logprobs=[{"4": math.log(0.5)}])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(logprobs=logprobs)])

    fake = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    result = module.teacher_force(fake, [{"role": "user", "content": "q"}], "42")
    assert [tok for tok, _, _ in result] == ["4", "2"]
    assert prompts[1].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n4")
    assert result[0][1] == {"4": pytest.approx(0.5)} and result[0][2] == pytest.approx(0.5)
