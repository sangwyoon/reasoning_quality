import json
import math
from collections import Counter
from types import SimpleNamespace

import pytest

import controls
import run_protocol_controls as R

TRACE = "Step 1: the force is mass times acceleration. Step 2: so a = F / m and option B holds."


def test_trace_messages_structure():
    m = controls.trace_messages("P", "T", "F")
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert [x["content"] for x in m] == ["P", "T", "F"]


def test_matched_template_same_structure_without_trace():
    m = controls.matched_template_messages("P", "F")
    assert [x["role"] for x in m] == ["user", "assistant", "user"]
    assert m[1]["content"] == ""
    assert controls.matched_template_messages("P", "F", placeholder="...")[1]["content"] == "..."


@pytest.mark.parametrize("n", [0, 1, 7, 50, 333])
def test_filler_exact_token_count(n):
    assert controls.count_tokens(controls.uninformative_text(n, 3, "filler")) == n


def test_shuffled_keeps_multiset_and_length():
    s = controls.shuffled_trace(TRACE, 1)
    assert Counter(s.split()) == Counter(TRACE.split())
    assert controls.count_tokens(s) == controls.count_tokens(TRACE)
    assert s != TRACE
    n = controls.count_tokens(TRACE)
    assert controls.uninformative_text(n, 1, "shuffled", trace=TRACE) == s


def test_determinism_and_seed_dependence():
    assert controls.filler_text(40, 5) == controls.filler_text(40, 5)
    assert controls.shuffled_trace(TRACE, 5) == controls.shuffled_trace(TRACE, 5)
    assert controls.shuffled_trace(TRACE, 5) != controls.shuffled_trace(TRACE, 6)


def test_uninformative_text_errors():
    with pytest.raises(ValueError):
        controls.uninformative_text(5, 0, "shuffled")
    with pytest.raises(ValueError):
        controls.uninformative_text(5, 0, "shuffled", trace=TRACE)
    with pytest.raises(ValueError):
        controls.uninformative_text(5, 0, "other")


def test_necessity_vs_arithmetic():
    assert controls.necessity_vs(math.log(0.5), math.log(0.25)) == pytest.approx(math.log(2))
    assert controls.necessity_vs(-1.0, -0.5) == pytest.approx(-0.5)


class FakeClient:
    """Answer distribution depends on the dialogue; optionally fails on a trigger."""

    def __init__(self, fail_if=None):
        self.calls = []
        self.fail_if = fail_if
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        msgs = kw["messages"]
        self.calls.append(msgs)
        if self.fail_if and self.fail_if(msgs):
            raise RuntimeError("boom")
        pa = 0.7 if len(msgs) == 3 and msgs[1]["content"] == TRACE else 0.4
        rest = (1 - pa) / 3
        top = [SimpleNamespace(token=t, logprob=math.log(p)) for t, p in
               (("A", pa), ("B", rest), ("C", rest), ("D", rest))]
        content = [SimpleNamespace(token="A", logprob=math.log(pa), top_logprobs=top)]
        return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(content=content))])


def _items(with_direct=True):
    out = []
    for i in range(3):
        it = {"question_id": f"q{i}", "prompt": f"Q{i}?", "reasoning": TRACE if i != 1 else TRACE + " x",
              "ground_truth": "A"}
        if with_direct:
            it["direct_prompt"] = f"Q{i}? answer only"
        out.append(it)
    return out


def test_condition_messages_structure():
    item = _items()[0]
    msgs = R.condition_messages(item, 1, "")
    assert set(msgs) == {"trace", "matched_template", "filler", "shuffled", "direct"}
    for k in ("trace", "matched_template", "filler", "shuffled"):
        assert [m["role"] for m in msgs[k]] == ["user", "assistant", "user"]
    n = controls.count_tokens(item["reasoning"])
    assert controls.count_tokens(msgs["filler"][1]["content"]) == n
    assert Counter(msgs["shuffled"][1]["content"].split()) == Counter(item["reasoning"].split())
    assert len(msgs["direct"]) == 1
    assert "direct" not in R.condition_messages(_items(False)[0], 1, "")


def test_runner_end_to_end_with_failure_and_resume(tmp_path):
    out = tmp_path / "out.jsonl"
    items = _items()
    # q1 filler condition fails: its trace is TRACE + " x" so the shuffled/filler turns differ
    fail_state = {"on": True}

    def fail_if(msgs):
        return fail_state["on"] and msgs[0]["content"].startswith("Q1?") and len(msgs) == 1

    client = FakeClient(fail_if)
    recs = R.run(client, "m", items, out, seed=7, workers=2)
    assert len(recs) == 3
    by = {r["question_id"]: r for r in recs}
    assert by["q0"]["error"] is None and by["q2"]["error"] is None
    # trace beats every baseline for gold A
    for k in ("matched_template", "filler", "shuffled", "direct"):
        assert by["q0"]["necessity"][f"necessity_vs_{k}"] == pytest.approx(math.log(0.7 / 0.4), abs=1e-6)
    c = by["q0"]["conditions"]["trace"]
    assert set(c["info"]) == {"choice_mass_observed", "n_missing", "tail_mass", "first_token"}
    # injected failure: only the direct condition for q1, no fabricated numbers
    assert by["q1"]["error"] is not None
    d = by["q1"]["conditions"]["direct"]
    assert d["error"] == "boom" and d["logp_gold"] is None and d["dist"] is None
    assert by["q1"]["necessity"]["necessity_vs_direct"] is None
    assert by["q1"]["conditions"]["filler"]["error"] is None

    # resume: only the errored item is re-run
    fail_state["on"] = False
    client2 = FakeClient()
    recs2 = R.run(client2, "m", items, out, seed=7, workers=1)
    assert [r["question_id"] for r in recs2] == ["q1"]
    assert recs2[0]["error"] is None
    assert len(client2.calls) == 5
    assert R.load_completed_ids(out) == {"q0", "q1", "q2"}
    # nothing pending now
    assert R.run(FakeClient(), "m", items, out) == []


def test_empty_reasoning_is_an_error_record(tmp_path):
    out = tmp_path / "o.jsonl"
    recs = R.run(FakeClient(), "m", [{"question_id": "e", "prompt": "p", "reasoning": " ",
                                      "ground_truth": "A"}], out)
    assert recs[0]["error"] == "empty reasoning" and recs[0]["conditions"] == {}


def test_main_writes_run_config(tmp_path, monkeypatch):
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("\n".join(json.dumps(i) for i in _items()) + "\n")
    import llm_client
    monkeypatch.setattr(llm_client, "create_client", lambda provider: FakeClient())
    out = tmp_path / "res" / "out.jsonl"
    R.main(["--items", str(items_path), "--model", "m", "--provider", "vllm",
            "--out", str(out), "--workers", "1"])
    cfg = json.loads((tmp_path / "res" / "run_config.json").read_text())
    assert cfg["model"] == "m" and cfg["top_logprobs"] == R.measures.MC_TOP_LOGPROBS
    assert len(out.read_text().strip().splitlines()) == 3
