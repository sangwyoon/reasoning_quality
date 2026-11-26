import json
import math
from types import SimpleNamespace

import run_mmlu_qwen3 as R

ROW = {"subject": "x", "question": "q", "choices": ["a", "b", "c", "d"], "answer": 0}


class FakeClient:
    """Returns a constant top-k over A-D for every request."""

    def __init__(self):
        self.n_calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.n_calls += 1
        top = [SimpleNamespace(token=t, logprob=math.log(p)) for t, p in
               (("A", 0.5), ("B", 0.2), ("C", 0.2), ("D", 0.1))]
        content = [SimpleNamespace(token="A", logprob=math.log(0.5), top_logprobs=top)]
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])


def _run(tmp_path, reasoning, **extra):
    client = FakeClient()
    cot = {"reasoning": reasoning, "predicted": "A", "ground_truth": "A", **extra}
    rec = R.process_kl(client, ROW, 0, tmp_path / "kl.jsonl", cot)
    return rec, client


def test_sm_condition_skipped_for_single_step_trace(tmp_path):
    rec, client = _run(tmp_path, "just one paragraph of reasoning here")
    assert rec["kl_oh_sm"] is None and rec["kl_oh_sm_m"] is None and rec["js_sm_smm"] is None
    assert rec["kl_oh_cot"] is not None and rec["kl_oh_dup_m"] is not None
    assert client.n_calls == 4  # cot, cot_m, dup, dup_m


def test_sm_condition_present_for_multi_step_trace(tmp_path):
    rec, client = _run(tmp_path, "para one\n\npara two\n\npara three")
    assert rec["kl_oh_sm"] is not None and rec["js_sm_smm"] is not None
    assert client.n_calls == 6


def test_reasoning_containing_word_error_is_not_skipped(tmp_path):
    rec, _ = _run(tmp_path, "an ERROR term in the algebra\n\nsecond paragraph")
    assert rec is not None


def test_record_with_error_field_is_skipped(tmp_path):
    rec, client = _run(tmp_path, "some reasoning\n\nmore", error="boom")
    assert rec is None and client.n_calls == 0


def test_resume_does_not_count_error_records(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"question_id": "ok", "error": None},
        {"question_id": "ok2"},
        {"question_id": "bad", "error": "boom"},
    ]) + "\n")
    assert R.load_completed_ids(p) == {"ok", "ok2"}


def test_apply_mask10_probe_id_and_seed():
    text = " ".join(f"w{i}" for i in range(100))
    a = R.apply_mask10(text, "s", 0)
    assert a == R.apply_mask10(text, "s")
    assert a != R.apply_mask10(text, "s", 1)
    assert "<redacted_UNK>" in a
