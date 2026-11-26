import json
from types import SimpleNamespace

import pytest

import measures as M


class FakeClient:
    def __init__(self, reply="yes", exc=None):
        self.calls = []
        self.reply, self.exc = reply, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        msg = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_judge_prompt_and_params():
    c = FakeClient("Yes")
    assert M.judge_equivalent(c, "gpt-4.1-nano", "5", "five") is True
    call = c.calls[0]
    assert call["temperature"] == 0 and call["max_tokens"] == 5
    assert call["model"] == "gpt-4.1-nano"
    assert call["messages"][0]["content"] == (
        "You are a math answer equivalence checker. Determine if the given answer is "
        "mathematically equivalent to the correct answer.\n\n"
        "Correct answer: 5\nGiven answer: five\n\n"
        'Are these two answers mathematically equivalent? Answer with ONLY "yes" or "no".'
    )


@pytest.mark.parametrize("reply,expected", [("yes", True), (" YES.", True), ("no", False), ("maybe", False)])
def test_judge_reply_parsing(reply, expected):
    assert M.judge_equivalent(FakeClient(reply), "m", "a", "b") is expected


def test_judge_raises_on_api_failure():
    with pytest.raises(RuntimeError):
        M.judge_equivalent(FakeClient(exc=RuntimeError("boom")), "m", "a", "b")


def test_judge_none_content_raises():
    with pytest.raises(ValueError):
        M.judge_equivalent(FakeClient(reply=None), "m", "a", "b")


def test_match_answers_judge_only_when_undecided():
    c = FakeClient("yes")
    judge = M.make_judge(c)
    assert M.match_answers("5", "5", judge) == (True, "exact")
    assert M.match_answers("1/2", "0.5", judge) == (True, "numeric")
    assert c.calls == []
    assert M.match_answers("\\sqrt{2}", "2^{1/2}", judge) == (True, "judge")
    assert len(c.calls) == 1
    assert c.calls[0]["model"] == M.DEFAULT_JUDGE_MODEL


def test_match_answers_without_judge_never_calls():
    assert M.match_answers("\\sqrt{2}", "2^{1/2}") == (False, "none")


def test_match_answers_judge_failure_propagates():
    judge = M.make_judge(FakeClient(exc=RuntimeError("down")))
    with pytest.raises(RuntimeError):
        M.match_answers("\\sqrt{2}", "2^{1/2}", judge)


def test_judge_no_for_inequivalent():
    judge = M.make_judge(FakeClient("no"))
    assert M.match_answers("3", "4", judge) == (False, "none")


def test_write_run_config(tmp_path):
    out = tmp_path / "res" / "kl.jsonl"
    p = M.write_run_config(out, model="m", dataset="d", seed=1, mask_ratio=0.1,
                           n_probes=1, top_logprobs=200)
    cfg = json.loads(p.read_text())
    assert p == tmp_path / "res" / "run_config.json"
    for key in ("model", "dataset", "seed", "mask_ratio", "n_probes", "top_logprobs",
                "top_logprobs_requested", "git_commit", "timestamp", "measures_version"):
        assert key in cfg
    assert cfg["top_logprobs_requested"] == 200
    assert cfg["measures_version"] == M.MEASURES_VERSION


def test_write_run_config_directory(tmp_path):
    p = M.write_run_config(tmp_path, model="m")
    assert p == tmp_path / "run_config.json"


def test_tail_mass_and_completed_prob():
    assert M.tail_mass(0.9) == pytest.approx(0.1)
    assert M.tail_mass(1.2) == 0.0
    dist = {"a": 0.5, "b": 0.4}
    assert M.completed_token_prob(dist, 0.9, "a", 1000) == 0.5
    missing = M.completed_token_prob(dist, 0.9, "z", 1002)
    assert missing == pytest.approx(0.1 / 1000)
    assert missing > 0
