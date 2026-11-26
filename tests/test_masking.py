import random
import re

import pytest

from paraphrase import TokenMasker, remove_one_step, split_reasoning_into_steps

TEXT = "Step 1: A is wrong.\n\nStep 2:  compute $x^2 + 1$ and B.\tDone"


def _masker(ratio, seed):
    return TokenMasker(mask_ratio=ratio, rng=random.Random(seed))


def test_unk_literal():
    assert TokenMasker.UNK == "<redacted_UNK>"


def test_mask_rate_matches_lambda():
    words = " ".join(f"w{i}" for i in range(200))
    masked = total = 0
    for seed in range(100):
        _, m, n = _masker(0.1, seed).mask(words)
        masked += m
        total += n
    assert masked / total == pytest.approx(0.1, abs=0.01)


def test_count_is_random_not_fixed():
    words = " ".join(f"w{i}" for i in range(50))
    counts = {_masker(0.1, s).mask(words)[1] for s in range(60)}
    assert len(counts) > 3


def test_no_forced_minimum_of_one():
    # lambda small and few tokens: some samples mask nothing
    counts = [_masker(0.05, s).mask("a b c")[1] for s in range(200)]
    assert 0 in counts


def test_whitespace_preserved_exactly():
    for seed in range(20):
        out, _, _ = _masker(0.5, seed).mask(TEXT)
        restored = re.sub(re.escape(TokenMasker.UNK), "X", out)
        assert re.findall(r"\s+", restored) == re.findall(r"\s+", TEXT)


def test_choice_labels_can_be_masked():
    seen = set()
    for seed in range(100):
        out, _, _ = _masker(0.5, seed).mask("A B")
        seen.add(out)
    assert any(o.startswith(TokenMasker.UNK) for o in seen)
    assert any(o.endswith(TokenMasker.UNK) for o in seen)


def test_latex_is_not_protected():
    out, m, n = _masker(1.0, 0).mask("$x^2$ + $y$")
    assert out == " ".join([TokenMasker.UNK] * 3)
    assert (m, n) == (3, 3)


def test_protect_latex_argument_is_ignored():
    a = TokenMasker(0.5, random.Random(3), protect_latex=True).mask(TEXT)
    b = TokenMasker(0.5, random.Random(3), protect_latex=False).mask(TEXT)
    assert a == b


def test_seed_reproducibility():
    assert _masker(0.3, "s").mask(TEXT) == _masker(0.3, "s").mask(TEXT)


def test_lambda_zero_and_one():
    assert _masker(0.0, 1).mask(TEXT) == (TEXT, 0, len(TEXT.split()))
    out, m, n = _masker(1.0, 1).mask(TEXT)
    assert m == n == len(TEXT.split())
    assert out.split() == [TokenMasker.UNK] * n


def test_empty_and_whitespace_only():
    assert _masker(1.0, 1).mask("") == ("", 0, 0)
    assert _masker(1.0, 1).mask("  \n ") == ("  \n ", 0, 0)


def test_invalid_ratio():
    with pytest.raises(ValueError):
        TokenMasker(mask_ratio=1.5)


TRACE = "Step 1: a\n\nStep 2: b\n\nStep 3: c\n\nStep 4: d"


def test_remove_one_step_removes_exactly_one():
    _, steps = split_reasoning_into_steps(TRACE)
    out, idx = remove_one_step(TRACE, "seed")
    assert 0 <= idx < len(steps)
    assert out == "\n\n".join(steps[:idx] + steps[idx + 1:])
    assert steps[idx] not in out
    assert len(split_reasoning_into_steps(out)[1]) == len(steps) - 1


def test_remove_one_step_reproducible_and_varies():
    assert remove_one_step(TRACE, "k") == remove_one_step(TRACE, "k")
    assert len({remove_one_step(TRACE, i)[1] for i in range(40)}) > 1


def test_remove_one_step_short_trace_returns_none():
    assert remove_one_step("only one paragraph", "k") == (None, -1)
    assert remove_one_step("", "k") == (None, -1)
