"""Failures are stored as error records or raise; no substitute values (no network)."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

import llm_client
import paraphrase
from runner_stubs import ROOT, client, response

MC_TOKENS = (("A", 0.7), ("B", 0.1), ("C", 0.1), ("D", 0.1))
TRACE = "para one\n\npara two\n\npara three"
SIMPLE_ROW = {"subject": "x", "question": "q", "choices": list("abcd"), "answer": 0}
GPQA_ROW = {"Question": "q", "High-level domain": "d"}


def _down(**kw):
    raise RuntimeError("server down")


def _kl(module, fake, out, cot):
    name = module.__name__
    if name == "run_kl_mask10_gpqa":
        return module.process_kl(fake, GPQA_ROW, 0, out, cot, list("abcd"), "A")
    if name == "run_agieval":
        row = {"question_id": "a0", "task": "t", "question": "q", "choices": list("abcd"),
               "answer": 0}
        return module.process_kl(fake, row, out, cot)
    if name == "run_mmlu_pro":
        row = {"question": "q", "options": list("abcd"), "answer": "A", "category": "c"}
        return module.process_kl(fake, row, 0, out, cot)
    return module.process_kl(fake, SIMPLE_ROW, 0, out, cot)


MC_KL_RUNNERS = ["run_mmlu_full", "run_mmlu_qwen3", "run_kl_mask10_gpqa", "run_agieval",
                 "run_mmlu_pro"]


@pytest.mark.parametrize("name", MC_KL_RUNNERS)
def test_mc_kl_failure_is_error_record_and_rerun(tmp_path, monkeypatch, name):
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "THINKING_MODEL", False, raising=False)
    out = tmp_path / "kl.jsonl"
    rec = _kl(module, client(_down), out, {"reasoning": TRACE})
    assert rec["error"] and rec["kl_oh_cot"] is None and rec["js_sm_smm"] is None
    assert json.loads(out.read_text())["error"] == rec["error"]
    assert module.load_completed_ids(out) == set()


@pytest.mark.parametrize("name", ["run_math_test", "run_math_qwen3", "run_gsm8k",
                                  "run_omnimath_kl"])
def test_open_kl_without_gold_is_error_record(tmp_path, name):
    module = importlib.import_module(name)
    row = {"problem": "p", "question": "p", "answer": "", "solution": ""}
    rec = module.process_kl(client(_down), row, 0, tmp_path / "kl.jsonl", {"reasoning": TRACE})
    assert "answer" in rec["error"] and rec["kl_oh_cot"] is None


@pytest.mark.parametrize("create", [_down, lambda **kw: response(""),
                                    lambda **kw: response("  "), lambda **kw: response(None)])
def test_llm_paraphraser_raises_instead_of_returning_original(create):
    with pytest.raises(RuntimeError):
        paraphrase.LLMParaphraser(client=client(create), model="p").paraphrase("original")


def _paraphrase_item(name, module, out, cot):
    if name == "run_paraphrase":
        return module.process_paraphrase(
            client(lambda **kw: response("A", MC_TOKENS)), "m", SIMPLE_ROW, 0, out, "x", cot,
            1, 42, "llm")
    return module.process_paraphrase(
        client(lambda **kw: response("A", MC_TOKENS)), "m", GPQA_ROW, 0, out, cot,
        list("abcd"), "A", 1, 42, "llm")


@pytest.mark.parametrize("name", ["run_paraphrase", "run_paraphrase_gpqa"])
@pytest.mark.parametrize("cot,message", [
    ({"reasoning": TRACE}, "server down"),
    ({"reasoning": "ERROR: boom", "error": "boom"}, "boom"),
])
def test_mc_paraphrase_failure_is_error_record(tmp_path, monkeypatch, name, cot, message):
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "paraphraser",
                        paraphrase.LLMParaphraser(client=client(_down), model="p"))
    out = tmp_path / "p.jsonl"
    rec = _paraphrase_item(name, module, out, cot)
    assert message in rec["error"]
    assert rec["correct_token_prob"] is None and rec["paraphrased_reasoning"] is None
    assert rec["original_correct_token_prob"] is None and rec["prob_delta"] is None
    assert module.load_completed_ids(out) == set()


def test_full_paraphrase_failure_is_error_record_and_success_is_stored(tmp_path):
    import run_full_paraphrase_gpqa as R
    out = tmp_path / "f.jsonl"
    evaluate = client(lambda **kw: response("A", MC_TOKENS))
    args = [GPQA_ROW, 0, out, {"reasoning": TRACE}, list("abcd"), "A", "cot", "reasoning"]
    failed = R.process_item(evaluate, client(lambda **kw: response("")), *args)
    assert "empty" in failed["error"] and failed["correct_token_prob"] is None
    assert R.load_completed_ids(out) == set()
    ok = R.process_item(evaluate, client(lambda **kw: response("reworded")), *args)
    assert ok["error"] is None and ok["paraphrased_reasoning"] == "reworded"


@pytest.mark.parametrize("name", ["run_kl_dup_para", "run_kl_stepmask_para"])
def test_self_paraphrase_empty_output_raises(name):
    module = importlib.import_module(name)
    with pytest.raises(ValueError, match="empty"):
        module.self_paraphrase(client(lambda **kw: response(" ")), "text")


def test_unknown_model_has_no_substitute_tokenizer(monkeypatch):
    def unknown(model):
        raise KeyError(model)

    used = []
    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(
        encoding_for_model=unknown, get_encoding=used.append))
    with pytest.raises(ValueError, match="tiktoken"):
        llm_client.get_tokenizer("together_ai/Qwen/Qwen3-32B")
    assert used == []


@pytest.mark.parametrize("name", ["run_math_test", "run_math_qwen3", "run_gsm8k", "run_omnimath",
                                  "run_omnimath_kl", "run_paraphrase_math",
                                  "run_paraphrase_omnimath"])
def test_average_token_probability_needs_logprobs(name):
    module = importlib.import_module(name)
    for empty in ([], None):
        with pytest.raises(ValueError):
            module.compute_avg_token_prob(empty)


@pytest.mark.parametrize("name", ["run_math_test", "run_math_qwen3", "run_gsm8k",
                                  "run_omnimath_kl", "run_js_cot_direct", "run_kl_paraphrase",
                                  "run_kl_dup_para", "run_kl_stepmask_para",
                                  "run_kl_duplicate_step", "run_sentence_ablation_all"])
def test_extract_distribution_needs_logprobs(name):
    module = importlib.import_module(name)
    func = getattr(module, "extract_distribution", None) or module.extract_distribution_tf
    with pytest.raises(ValueError):
        func([])


FORCERS = ["run_omnimath", "run_paraphrase_math", "run_paraphrase_omnimath"]


@pytest.mark.parametrize("name", FORCERS)
def test_forced_probability_needs_a_gold_answer(name):
    module = importlib.import_module(name)
    with pytest.raises(ValueError):
        module.teacher_force_correct_prob(client(_down), "m", [], "  ")


@pytest.mark.parametrize("name", FORCERS)
def test_vllm_teacher_forcing_continues_the_prefix(monkeypatch, name):
    import vllm_server
    module = importlib.import_module(name)
    monkeypatch.setattr(module, "PROVIDER", "vllm")
    monkeypatch.setattr(module, "VOCAB_SIZE", 1000)
    monkeypatch.setattr(vllm_server, "tokenize", lambda model, text: [1, 2])
    monkeypatch.setattr(vllm_server, "detokenize", lambda model, ids: ["4", "2"])
    calls = []

    def create(**kw):
        calls.append(kw)
        return response("4", (("4", 0.5), ("2", 0.4)))

    module.teacher_force_correct_prob(client(create), "m", [{"role": "user", "content": "q"}], "42")
    assert "extra_body" not in calls[0]
    assert calls[1]["messages"][-1] == {"role": "assistant", "content": "4"}
    assert calls[1]["extra_body"] == {"continue_final_message": True,
                                      "add_generation_prompt": False}


def test_exclusion_manifest_counts_error_records_as_missing(tmp_path):
    import subprocess
    for folder, rows in (
        ("exp1_cot_gpqa_m", [{"question_id": "a"}, {"question_id": "b"}]),
        ("kl_mask10_gpqa_m", [{"question_id": "a", "error": None},
                              {"question_id": "b", "error": "x"}]),
    ):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    script = ROOT / "additional_experiments/semantic_and_structural/make_exclusion_manifest.py"
    subprocess.run([sys.executable, str(script), "--base", str(tmp_path)], check=True,
                   capture_output=True)
    manifest = json.loads((tmp_path / "excluded_questions.json").read_text())
    assert manifest["kl/gpqa/m"]["excluded"] == ["b"]
