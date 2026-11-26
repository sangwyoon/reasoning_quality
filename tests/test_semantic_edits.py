import pytest

import semantic_edits as S

TRACE = (
    "First, compute 3.14 * 2 = 6.28. Then, e.g. use $x = 1.5. y$ here! "
    "Is it done? Yes.\n\n1. Check the sum.\n2. Verify the result.\n- Done now."
)


def _long_trace(n):
    return " ".join(f"Sentence number {i} is here." for i in range(n))


def _fake(calls=None, fn=lambda p: "EDITED"):
    def llm(prompt):
        if calls is not None:
            calls.append(prompt)
        return fn(prompt)

    return llm


# ---- split -------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        TRACE,
        "",
        "   leading and trailing   ",
        "No terminator at all",
        "a. b.\n\n\nc!  d?\n",
        "Cost is $5. Next $x.y$ line. Done.",
        "Line one\nLine two\n- bullet\n(a) item\nend.",
    ],
)
def test_split_is_lossless(text):
    assert "".join(S.split_sentences(text)) == text


def test_split_does_not_break_decimals_abbreviations_lists_or_math():
    segs = S.split_sentences(TRACE)
    stripped = [s.strip() for s in segs]
    assert stripped[0] == "First, compute 3.14 * 2 = 6.28."
    assert stripped[1] == "Then, e.g. use $x = 1.5. y$ here!"
    assert stripped[2] == "Is it done?"
    assert stripped[3] == "Yes."
    assert stripped[4] == "1. Check the sum."
    assert stripped[5] == "2. Verify the result."
    assert stripped[6] == "- Done now."


def test_split_empty_and_blank():
    assert S.split_sentences("") == []
    assert "".join(S.split_sentences("  \n ")) == "  \n "


def test_split_paragraph_break_is_boundary_without_terminator():
    segs = S.split_sentences("alpha\n\nbeta")
    assert [s.strip() for s in segs] == ["alpha", "beta"]


# ---- selection ---------------------------------------------------------
@pytest.mark.parametrize("n,expected", [(0, 0), (1, 1), (3, 1), (5, 1), (15, 2), (25, 3), (45, 5), (100, 10)])
def test_selection_count(n, expected):
    idx = S.select_sentences(n, seed=0)
    assert len(idx) == expected
    assert idx == sorted(set(idx))
    assert all(0 <= i < n for i in idx)


def test_selection_rounds_half_up_not_bankers():
    # round(0.5) == 0 and round(2.5) == 2 in Python; half-up gives 1 and 3.
    assert S.n_to_select(5, 0.1) == 1
    assert S.n_to_select(25, 0.1) == 3
    assert S.n_to_select(15, 0.1) == 2


def test_selection_empty_returns_empty_list():
    assert S.select_sentences(0, seed=1) == []


def test_selection_deterministic_and_seed_sensitive():
    assert S.select_sentences(50, seed=7) == S.select_sentences(50, seed=7)
    assert any(S.select_sentences(50, seed=s) != S.select_sentences(50, seed=7) for s in range(8))


def test_selection_rejects_rng_and_seed_together():
    import random

    with pytest.raises(ValueError):
        S.select_sentences(10, rng=random.Random(0), seed=0)


# ---- edits -------------------------------------------------------------
def test_delete_removes_exactly_selected_sentences():
    trace = _long_trace(15)
    segs = S.split_sentences(trace)
    idx = [2, 9]
    out = S.apply_edits(trace, idx, S.delete_edit)
    assert out == "".join(s for i, s in enumerate(segs) if i not in idx)
    assert len(S.split_sentences(out)) == 13
    assert "Sentence number 2 " not in out and "Sentence number 9 " not in out


def test_llm_edit_replaces_only_selected_and_keeps_whitespace():
    trace = "One is here.  Two is here.\nThree is here."
    out = S.apply_edits(trace, [1], lambda s: "X")
    assert out == "One is here.  X\nThree is here."


def test_apply_edits_validates_indices():
    with pytest.raises(IndexError):
        S.apply_edits("A b. C d.", [5], S.delete_edit)
    with pytest.raises(ValueError):
        S.apply_edits("A b. C d.", [0, 0], S.delete_edit)


def test_prompts_are_exact_and_sentence_follows_on_new_line():
    calls = []
    llm = _fake(calls)
    S.paraphrase_edit(llm)("Hello there.")
    S.meaning_breaking_edit(llm)("Hello there.")
    assert calls[0] == (
        "Please paraphrase the following sentence, while preserving the original semantics.\nHello there."
    )
    assert calls[1] == (
        "Please invert the semantics of the following sentence. Replace key adjectives or relational "
        "predicates with antonyms or contradiction. For mathematical expressions, perturb operators or "
        "quantities.\nHello there."
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"Quoted answer."', "Quoted answer."),
        ("Here is the paraphrased sentence:\nNew words.", "New words."),
        ("“Curly.”", "Curly."),
        ("  plain  ", "plain"),
    ],
)
def test_response_cleaning(raw, expected):
    assert S.clean_response(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", '""'])
def test_empty_response_raises(raw):
    with pytest.raises(S.EditError):
        S.paraphrase_edit(_fake(fn=lambda p: raw))("Some sentence.")


def test_llm_exception_propagates_as_edit_error_without_fallback():
    def boom(prompt):
        raise RuntimeError("api down")

    trace = _long_trace(5)
    with pytest.raises(S.EditError, match="api down"):
        S.build_perturbation_set(trace, seed=0, llm=boom)


def test_non_string_response_raises():
    with pytest.raises(S.EditError):
        S.paraphrase_edit(lambda p: None)("Some sentence.")


# ---- perturbation set --------------------------------------------------
def test_same_indices_for_all_three_edits_and_call_counts():
    trace = _long_trace(25)
    calls = []
    llm = _fake(calls, fn=lambda p: "REPLACED " + p.split("\n")[-1])
    out = S.build_perturbation_set(trace, seed=3, llm=llm)
    assert out["n_sentences"] == 25
    assert out["n_selected"] == 3 == len(out["indices"])
    # one call per selected sentence for each of the two LLM edits
    assert len(calls) == 2 * out["n_selected"]
    assert sum(p.startswith("Please paraphrase") for p in calls) == out["n_selected"]
    assert sum(p.startswith("Please invert") for p in calls) == out["n_selected"]

    segs = S.split_sentences(trace)
    idx = set(out["indices"])
    for key in ("paraphrase", "meaning_breaking"):
        edited = S.split_sentences(out[key])
        assert len(edited) == len(segs)
        changed = {i for i, (a, b) in enumerate(zip(segs, edited)) if a != b}
        assert changed == idx
    kept = [s for i, s in enumerate(segs) if i not in idx]
    assert out["delete"] == "".join(kept)


def test_single_short_trace_gets_one_edit():
    calls = []
    out = S.build_perturbation_set("Only one. Two here. Three now.", seed=1, llm=_fake(calls))
    assert out["n_selected"] == 1
    assert len(calls) == 2


def test_empty_trace_makes_no_calls_and_is_unchanged():
    calls = []
    out = S.build_perturbation_set("", seed=1, llm=_fake(calls))
    assert calls == []
    assert out["indices"] == [] and out["n_selected"] == 0
    assert out["paraphrase"] == out["meaning_breaking"] == out["delete"] == ""


def test_deterministic_given_seed_and_sample_id():
    trace = _long_trace(30)
    llm = _fake(fn=lambda p: "E:" + p.split("\n")[-1])
    a = S.build_perturbation_set(trace, seed=5, llm=llm, n_samples=3)
    b = S.build_perturbation_set(trace, seed=5, llm=llm, n_samples=3)
    c = S.build_perturbation_set(trace, seed=6, llm=llm, n_samples=3)
    assert a == b
    assert a["seeds"] == [S.sample_seed(5, i) for i in range(3)]
    assert len(set(a["seeds"])) == 3
    assert [s["indices"] for s in a["samples"]] != [s["indices"] for s in c["samples"]]
    assert a["samples"][0]["indices"] == a["indices"]


def test_token_counts_recorded():
    trace = _long_trace(10)
    out = S.build_perturbation_set(trace, seed=0, llm=_fake(fn=lambda p: "one two three four five six"))
    t = out["samples"][0]["n_tokens"]
    assert t["original"] == len(trace.split())
    assert t["delete"] < t["original"] < t["paraphrase"]
    assert set(t) == {"original", "paraphrase", "meaning_breaking", "delete"}


def test_n_samples_must_be_positive():
    with pytest.raises(ValueError):
        S.build_perturbation_set("A b.", seed=0, llm=_fake(), n_samples=0)
