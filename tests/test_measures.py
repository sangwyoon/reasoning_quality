import math
import random

import pytest

import measures as M

LABELS = ("A", "B", "C", "D")


def _items(probs):
    return [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]


def test_missing_choice_gets_tail_based_probability():
    # A choice absent from the returned top-k receives a nonzero tail-based probability.
    items = _items({"A": 0.55, "B": 0.25, "C": 0.1, "X": 0.0412})
    dist, info = M.choice_distribution(items, LABELS)
    assert info["n_missing"] == 1
    assert dist["D"] > 0
    nll = M.kl_onehot_choice(dist, "D")
    assert 2.5 < nll < 5.0


def test_choice_dist_sums_to_one_and_info():
    items = _items({"A": 0.6, " B": 0.2, "other": 0.1})
    dist, info = M.choice_distribution(items, LABELS)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert info["choice_mass_observed"] == pytest.approx(0.8)
    assert info["n_missing"] == 2
    assert info["tail_mass"] == pytest.approx(0.1)
    assert all(v > 0 for v in dist.values())


def test_choice_dist_all_observed_no_fill():
    items = _items({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
    dist, info = M.choice_distribution(items, LABELS)
    assert info["n_missing"] == 0
    assert dist["A"] == pytest.approx(0.4)


def test_choice_dist_duplicate_tokens_summed_and_dot():
    items = _items({"A": 0.3, " A": 0.2, "B.": 0.1})
    dist, info = M.choice_distribution(items, LABELS)
    assert info["choice_mass_observed"] == pytest.approx(0.6)
    assert dist["A"] > dist["B"]


def test_choice_dist_no_valid_choice_raises():
    with pytest.raises(ValueError):
        M.choice_distribution(_items({"The": 0.9, "I": 0.05}), LABELS)


def test_choice_dist_object_items():
    class T:
        def __init__(self, token, logprob):
            self.token, self.logprob = token, logprob

    dist, _ = M.choice_distribution([T("A", math.log(0.9)), T("B", math.log(0.1))], ("A", "B"))
    assert dist["A"] == pytest.approx(0.9)


def test_js_choice_zero_and_symmetric():
    p = {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
    q = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}
    assert M.js_choice(p, p) == pytest.approx(0.0, abs=1e-12)
    assert M.js_choice(p, q) == pytest.approx(M.js_choice(q, p))
    assert M.js_choice(p, q) > 0


VOCAB = 151936


def _acceptance_dists():
    toks = [f"t{i}" for i in range(200)]
    return ({t: 0.004 for t in toks}, 0.8, {t: 0.002 for t in toks}, 0.4)


def test_js_open_ended_acceptance_value():
    dp, mp, dq, mq = _acceptance_dists()
    assert M.js_open_ended(dp, mp, dq, mq, VOCAB) == pytest.approx(0.0863, abs=5e-4)


def test_js_open_ended_exceeds_truncated_sum_and_event_bound():
    dp, mp, dq, mq = _acceptance_dists()
    full = M.js_open_ended(dp, mp, dq, mq, VOCAB)
    # Without the tail term (n_rest = 0) we recover the old ~0.0340.
    truncated = M.js_open_ended(dp, mp, dq, mq, 200)
    assert truncated < full
    assert math.sqrt(2 * full) >= 0.4


def test_js_open_ended_identical_zero_and_symmetric():
    dp, mp, _, _ = _acceptance_dists()
    assert M.js_open_ended(dp, mp, dict(dp), mp, VOCAB) == pytest.approx(0.0, abs=1e-12)
    dq = {"t0": 0.3, "z": 0.2}
    a = M.js_open_ended(dp, mp, dq, 0.5, VOCAB)
    b = M.js_open_ended(dq, 0.5, dp, mp, VOCAB)
    assert a == pytest.approx(b)


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("1", "10", False),
        ("1", "1", True),
        ("1/2", "0.5", True),
        ("\\frac{1}{2}", "1/2", True),
        ("\\dfrac{1}{2}", "0.5", True),
        ("\\boxed{5}", "5", True),
        ("5", "x=5", True),
        ("5", "x = 6", False),
        ("5", "The answer is 5", False),
        ("1,000", "1000", True),
        ("5.", "5", True),
        ("$5$", "5", True),
        ("\\text{yes}", "Yes", True),
        ("3", "30", False),
        ("(1,2)", "(1,2)", True),
        ("(1,2)", "(1,3)", False),
        ("", "5", False),
        ("5", "", False),
        ("1/0", "1/0", True),  # identical strings are exact matches
        ("1/0", "5", False),
    ],
)
def test_answers_equivalent(gold, pred, expected):
    assert M.answers_equivalent(gold, pred) is expected


def test_match_methods_and_judge_hook():
    assert M.match_answers("5", "5") == (True, "exact")
    assert M.match_answers("1/2", "0.5") == (True, "numeric")
    assert M.match_answers("sqrt2", "root 2") == (False, "none")
    assert M.match_answers("sqrt2", "root 2", judge=lambda g, p: True) == (True, "judge")
    calls = []
    M.match_answers("5", "5", judge=lambda g, p: calls.append(1) or True)
    assert calls == []  # judge not consulted when exact


def test_prop1_bound_random_distributions():
    rng = random.Random(0)
    n = 30
    for _ in range(200):
        p = [rng.random() ** 3 + 1e-9 for _ in range(n)]
        q = [rng.random() ** 3 + 1e-9 for _ in range(n)]
        sp, sq = sum(p), sum(q)
        p = {i: v / sp for i, v in enumerate(p)}
        q = {i: v / sq for i, v in enumerate(q)}
        js = M.js_open_ended(p, 1.0, q, 1.0, n)
        event = {i for i in range(n) if rng.random() < 0.5}
        gap = abs(sum(p[i] for i in event) - sum(q[i] for i in event))
        assert gap <= math.sqrt(2 * js) + 1e-9


def test_extract_choice_probs_wrapper():
    class T:
        def __init__(self, token, logprob):
            self.token, self.logprob = token, logprob

    class First:
        token = " A"
        top_logprobs = [T(" A", math.log(0.7)), T("B", math.log(0.2))]

    dist, predicted, info = M.extract_choice_probs([First()], LABELS)
    assert predicted == "A"
    assert sum(dist.values()) == pytest.approx(1.0)
    assert info["n_missing"] == 2
    with pytest.raises(ValueError):
        M.extract_choice_probs([], LABELS)


def test_boxed_inside_sentence_extracted_but_not_substring():
    assert M.answers_equivalent("5", "So the result is \\boxed{5}.")
    assert M.answers_equivalent("\\frac{1}{2}", "\\boxed{\\dfrac{1}{2}}")
    assert not M.answers_equivalent("1", "\\boxed{10}")
