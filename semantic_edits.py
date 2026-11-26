"""Semantic perturbation protocol (paper appendix D.5) as a pure, testable module.

Three sentence-level edits are applied to the *same* randomly selected
sentences of a reasoning trace:

* ``paraphrase``       (LLM)  rewrite the sentence, preserving semantics
* ``meaning_breaking`` (LLM)  invert the semantics of the sentence
* ``delete``           (no LLM) remove the sentence

Unselected content (and the question, which is never passed in here) is left
byte-for-byte untouched. No API code runs at import time or in tests: LLM edits
take an injected ``llm(prompt: str) -> str`` callable.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Callable, Dict, List, Optional, Sequence

PARAPHRASE_PROMPT = (
    "Please paraphrase the following sentence, while preserving the original semantics."
)
MEANING_BREAKING_PROMPT = (
    "Please invert the semantics of the following sentence. Replace key adjectives or "
    "relational predicates with antonyms or contradiction. For mathematical expressions, "
    "perturb operators or quantities."
)

DEFAULT_RATE = 0.10

# Abbreviations whose trailing period never ends a sentence.
_ALWAYS_NONBREAK = {
    "e.g", "i.e", "vs", "cf", "fig", "eq", "eqs", "dr", "mr", "mrs", "ms", "prof",
    "approx", "sec", "viz", "resp", "ca",
}
# Abbreviations that only fail to end a sentence when a lowercase word follows.
_LOWER_ONLY_NONBREAK = {"etc", "al", "no", "st"}
_CLOSERS = "\"')]}”’*"
_LIST_MARKER = re.compile(r"\s*(?:[-*•]\s|\d+[.)]\s|\(?[A-Za-z]\)\s|#+\s)")
_LABEL_NUMBER = re.compile(r"(?i)\b(?:step|problem|question|part|case|answer|option)\s+\d+$")


class EditError(RuntimeError):
    """Raised when an LLM editor fails or returns an unusable (empty) response."""


# --------------------------------------------------------------------------- split
def _is_nonbreak_period(text: str, dot: int, next_char: str, same_line: bool) -> bool:
    """True if the '.' at ``dot`` (followed by whitespace then ``next_char``) does not end a sentence."""
    before = text[:dot]
    line_start = before.rfind("\n") + 1
    token = before[max(before.rfind(" ", line_start), before.rfind("\t", line_start), line_start - 1) + 1:]
    token = token.lstrip("(").lower()
    if token in _ALWAYS_NONBREAK:
        return True
    if token in _LOWER_ONLY_NONBREAK and next_char.islower():
        return True
    # Numbered/lettered list marker at line start: "1. foo", "a. foo".
    if before[line_start:].strip() == token.strip() and re.fullmatch(r"\d+|[a-z]", token):
        return True
    if _LABEL_NUMBER.search(before[line_start:]):
        return True
    # Same-line lowercase continuation ("... 3.14. then" is rare; "e.g. the" handled above).
    return same_line and next_char.islower()


def split_sentences(trace: str) -> List[str]:
    """Split ``trace`` into sentence segments, losslessly.

    ``"".join(split_sentences(t)) == t`` always holds: each segment is the
    sentence text plus the whitespace that follows it (leading whitespace of the
    trace stays on the first segment). Boundaries:

    * ``.``/``!``/``?`` (optionally followed by closing quotes/brackets) followed by
      whitespace, except decimals (``3.14`` has no whitespace), abbreviations
      (``e.g.``, ``i.e.``, ``Fig.``...), numbered/lettered list markers at line
      start (``1. ``), ``Step 3.``-style labels, and a same-line lowercase continuation;
    * a blank line (paragraph break);
    * a newline followed by a list marker (``- ``, ``1. ``, ``(a) ``).

    Nothing inside LaTeX math (``$..$``, ``$$..$$``, ``\\(..\\)``, ``\\[..\\]``)
    is split; an unbalanced ``$`` is reset at the next blank line.
    """
    n = len(trace)
    segments: List[str] = []
    start = 0
    closer: Optional[str] = None  # the closing token of the open math span, if any
    i = 0
    while i < n:
        ch = trace[i]
        if closer is None:
            if trace.startswith("\\$", i):
                i += 2
                continue
            if trace.startswith("$$", i):
                closer, i = "$$", i + 2
                continue
            if ch == "$":
                closer, i = "$", i + 1
                continue
            if trace.startswith("\\(", i):
                closer, i = "\\)", i + 2
                continue
            if trace.startswith("\\[", i):
                closer, i = "\\]", i + 2
                continue
        elif trace.startswith("\\$", i):
            i += 2
            continue
        elif trace.startswith(closer, i):
            i += len(closer)
            closer = None
            continue

        if not ch.isspace():
            i += 1
            continue

        # Whitespace run [i, m).
        m = i
        while m < n and trace[m].isspace():
            m += 1
        run = trace[i:m]
        blank_line = run.count("\n") >= 2
        if blank_line:
            closer = None
        prev_ok = i > start and m < n  # need a sentence before and text after
        if closer is None and prev_ok:
            j = i
            while j > start and trace[j - 1] in _CLOSERS:
                j -= 1
            term = trace[j - 1] if j > start else ""
            nxt = trace[m]
            is_break = False
            if blank_line:
                is_break = True
            elif term in ".!?":
                has_nl = "\n" in run
                if term == "." and _is_nonbreak_period(trace, j - 1, nxt, not has_nl):
                    is_break = False
                elif term != "." and not has_nl and nxt.islower():
                    is_break = False
                else:
                    is_break = True
            elif "\n" in run and _LIST_MARKER.match(trace, m):
                is_break = True
            if is_break and trace[start:i].strip():
                segments.append(trace[start:m])
                start = m
        i = m
    if start < n:
        segments.append(trace[start:])
    return segments


# ------------------------------------------------------------------------ selection
def n_to_select(n_sentences: int, rate: float = DEFAULT_RATE) -> int:
    """Number of sentences to edit: ``max(1, floor(rate*n + 0.5))`` for n >= 1, else 0.

    Rounding is half-up via ``floor(x + 0.5)`` (not Python's ``round``, which is
    banker's rounding: ``round(0.5) == 0``, ``round(2.5) == 2``). A tiny epsilon
    guards float noise (e.g. 0.1*45 == 4.5 must round to 5). Short traces always
    get at least one edited sentence; the count never exceeds ``n_sentences``.
    """
    if n_sentences <= 0:
        return 0
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    k = math.floor(rate * n_sentences + 0.5 + 1e-9)
    return min(n_sentences, max(1, k))


def select_sentences(
    n_sentences: int,
    rate: float = DEFAULT_RATE,
    rng: Optional[random.Random] = None,
    seed: Optional[int] = None,
) -> List[int]:
    """Sample ``n_to_select(n_sentences, rate)`` distinct indices, returned sorted.

    Count = ``max(1, floor(rate*n + 0.5))`` for ``n >= 1`` (half-up rounding, see
    :func:`n_to_select`); ``[]`` for ``n == 0``. Pass ``rng`` or ``seed`` (not both)
    for determinism.
    """
    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    k = n_to_select(n_sentences, rate)
    if k == 0:
        return []
    rng = rng if rng is not None else random.Random(seed)
    return sorted(rng.sample(range(n_sentences), k))


# --------------------------------------------------------------------------- edits
_PREAMBLE = re.compile(
    r"(?is)^\s*(?:sure|certainly|okay|ok|here(?:'s| is| are)|the (?:paraphrased|inverted|rewritten|revised)|"
    r"paraphrase[d]?|inverted|revised|rewritten)\b[^\n]*:\s*\n?"
)
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`"}


def clean_response(text: str) -> str:
    """Strip a leading preamble line ("Here is ...:") and wrapper quotes."""
    out = text.strip()
    stripped = _PREAMBLE.sub("", out, count=1).strip()
    if stripped:
        out = stripped
    while len(out) >= 2 and _QUOTE_PAIRS.get(out[0]) == out[-1]:
        out = out[1:-1].strip()
    return out


def make_llm_edit(llm: Callable[[str], str], instruction: str) -> Callable[[str], str]:
    """Build an edit function ``sentence -> edited sentence`` from an LLM callable.

    The prompt is ``instruction + "\\n" + sentence`` and nothing else. Errors and
    empty responses raise :class:`EditError`; there is no fallback to the original.
    """

    def edit(sentence: str) -> str:
        try:
            raw = llm(f"{instruction}\n{sentence}")
        except Exception as exc:
            raise EditError(f"LLM editor failed on sentence {sentence!r}: {exc}") from exc
        if not isinstance(raw, str):
            raise EditError(f"LLM editor returned {type(raw).__name__}, expected str")
        cleaned = clean_response(raw)
        if not cleaned:
            raise EditError(f"LLM editor returned an empty response for sentence {sentence!r}")
        return cleaned

    return edit


def paraphrase_edit(llm: Callable[[str], str]) -> Callable[[str], str]:
    return make_llm_edit(llm, PARAPHRASE_PROMPT)


def meaning_breaking_edit(llm: Callable[[str], str]) -> Callable[[str], str]:
    return make_llm_edit(llm, MEANING_BREAKING_PROMPT)


def delete_edit(sentence: str) -> None:
    """Edit function for deletion: ``None`` means "remove this sentence"."""
    return None


def apply_edits(
    trace: str,
    indices: Sequence[int],
    edit_fn: Callable[[str], Optional[str]],
    segments: Optional[List[str]] = None,
) -> str:
    """Apply ``edit_fn`` to the sentences at ``indices`` and rejoin.

    ``edit_fn`` receives the sentence text without its surrounding whitespace and
    returns the replacement, or ``None`` to delete the sentence (the sentence and
    its trailing whitespace are removed). The original leading/trailing whitespace
    of an edited sentence is preserved; unselected segments are untouched.
    """
    segs = list(segments) if segments is not None else split_sentences(trace)
    idx = sorted(indices)
    if len(set(idx)) != len(idx):
        raise ValueError(f"duplicate indices: {idx}")
    for i in idx:
        if not 0 <= i < len(segs):
            raise IndexError(f"sentence index {i} out of range (n={len(segs)})")
    selected = set(idx)
    out: List[str] = []
    for i, seg in enumerate(segs):
        if i not in selected:
            out.append(seg)
            continue
        core = seg.strip()
        lead = seg[: len(seg) - len(seg.lstrip())]
        trail = seg[len(seg.rstrip()):]
        new = edit_fn(core)
        if new is None:
            continue
        out.append(lead + new + trail)
    return "".join(out)


# ------------------------------------------------------------------- perturbation set
def sample_seed(seed: int, sample_id: int) -> int:
    """Deterministic per-sample seed from (seed, sample_id); independent of PYTHONHASHSEED."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def whitespace_token_count(text: str) -> int:
    """Default token proxy (whitespace words). Pass a real tokenizer's counter for the paper's budgets."""
    return len(text.split())


def build_perturbation_set(
    trace: str,
    seed: int,
    llm: Callable[[str], str],
    rate: float = DEFAULT_RATE,
    n_samples: int = 1,
    count_tokens: Callable[[str], int] = whitespace_token_count,
) -> Dict[str, object]:
    """Build paraphrase / meaning-breaking / delete variants of ``trace``.

    For each sample ``s`` in ``range(n_samples)`` one index set is drawn with seed
    ``sample_seed(seed, s)`` and used for *all three* edits. Deterministic given
    (trace, seed, sample_id) as long as ``llm`` is.

    Return value::

        {"n_sentences", "n_selected", "rate", "n_samples",
         "samples": [{"sample_id", "seed", "indices", "paraphrase",
                      "meaning_breaking", "delete", "n_tokens"}, ...],
         # convenience aliases for sample 0:
         "paraphrase", "meaning_breaking", "delete", "indices", "seeds"}

    ``n_tokens`` maps ``original`` / ``paraphrase`` / ``meaning_breaking`` /
    ``delete`` to token counts: unlike masking, these edits change the token budget.
    ``seeds`` lists the per-sample seeds. LLM failures raise :class:`EditError`.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    segments = split_sentences(trace)
    n = len(segments)
    k = n_to_select(n, rate)
    para = paraphrase_edit(llm)
    brk = meaning_breaking_edit(llm)
    original_tokens = count_tokens(trace)

    samples = []
    for sid in range(n_samples):
        s_seed = sample_seed(seed, sid)
        indices = select_sentences(n, rate, seed=s_seed)
        variants = {
            "paraphrase": apply_edits(trace, indices, para, segments),
            "meaning_breaking": apply_edits(trace, indices, brk, segments),
            "delete": apply_edits(trace, indices, delete_edit, segments),
        }
        samples.append(
            {
                "sample_id": sid,
                "seed": s_seed,
                "indices": indices,
                **variants,
                "n_tokens": {
                    "original": original_tokens,
                    **{name: count_tokens(text) for name, text in variants.items()},
                },
            }
        )
    first = samples[0]
    return {
        "n_sentences": n,
        "n_selected": k,
        "rate": rate,
        "n_samples": n_samples,
        "samples": samples,
        "paraphrase": first["paraphrase"],
        "meaning_breaking": first["meaning_breaking"],
        "delete": first["delete"],
        "indices": first["indices"],
        "seeds": [s["seed"] for s in samples],
    }


def make_openai_editor(client, model: str, **create_kwargs) -> Callable[[str], str]:
    """Wrap an OpenAI-style client as ``llm(prompt) -> str``. Never called in tests."""

    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], **create_kwargs
        )
        return resp.choices[0].message.content or ""

    return llm
