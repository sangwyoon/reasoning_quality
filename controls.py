"""Protocol-matched controls for multiple-choice necessity (pure functions, no I/O).

The with-trace condition is a three-message dialogue: user prompt, assistant trace,
user follow-up.  A question-only baseline is a single-turn prompt, so comparing the
two mixes dialogue structure with trace content.  The helpers here build baselines
that keep the same dialogue structure while changing only the assistant turn:

  * ``trace_messages``            assistant turn = the trace
  * ``matched_template_messages`` assistant turn = a placeholder (trace omitted)
  * ``uninformative_text``        neutral filler text of the trace's length
  * ``shuffled_trace``            the trace's own tokens in random order

Length is measured in whitespace-separated tokens.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

Message = Dict[str, str]

FILLER_KIND = "filler"
SHUFFLED_KIND = "shuffled"
KINDS = (FILLER_KIND, SHUFFLED_KIND)

# Neutral sentences unrelated to any question; cycled to the requested length.
FILLER_SENTENCES = (
    "The afternoon light settled over the quiet street while a few people walked by.",
    "A small cafe on the corner served warm bread and tea to its regular guests.",
    "Several birds gathered near the fountain and then flew toward the distant hills.",
    "The train arrived at the platform, paused for a moment, and departed again.",
    "A gentle breeze moved through the garden and turned the pages of an open book.",
    "Children played near the river while the water carried leaves slowly downstream.",
    "The old clock in the hallway ticked steadily as evening came to the house.",
)


def trace_messages(prompt: str, trace: str, followup: str) -> List[Message]:
    """Standard with-trace dialogue: [user prompt, assistant trace, user follow-up]."""
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": trace},
        {"role": "user", "content": followup},
    ]


def matched_template_messages(prompt: str, followup: str, placeholder: str = "") -> List[Message]:
    """Same three-message structure as ``trace_messages`` with the trace omitted.

    The assistant turn holds only ``placeholder``.  Some chat templates handle an
    empty assistant turn differently from a non-empty one, so the placeholder is
    configurable.
    """
    return trace_messages(prompt, placeholder, followup)


def count_tokens(text: str) -> int:
    """Number of whitespace-separated tokens."""
    return len(text.split())


def filler_text(n_tokens: int, seed: int) -> str:
    """Neutral filler of exactly ``n_tokens`` whitespace tokens.

    The filler sentences are concatenated cyclically starting at a seed-dependent
    sentence and cut at ``n_tokens`` tokens.
    """
    if n_tokens < 0:
        raise ValueError("n_tokens must be >= 0")
    start = random.Random(seed).randrange(len(FILLER_SENTENCES))
    tokens: List[str] = []
    i = start
    while len(tokens) < n_tokens:
        tokens.extend(FILLER_SENTENCES[i % len(FILLER_SENTENCES)].split())
        i += 1
    return " ".join(tokens[:n_tokens])


def shuffled_trace(trace: str, seed: int) -> str:
    """The trace's own whitespace tokens in a seeded random order.

    Same length and vocabulary as ``trace``; the token order is permuted.
    """
    tokens = trace.split()
    random.Random(seed).shuffle(tokens)
    return " ".join(tokens)


def uninformative_text(
    n_tokens: int,
    seed: int,
    kind: str = FILLER_KIND,
    trace: Optional[str] = None,
) -> str:
    """Text with exactly ``n_tokens`` whitespace tokens.

    ``kind="filler"``: neutral sentences unrelated to any question.
    ``kind="shuffled"``: requires ``trace`` whose token count equals ``n_tokens``;
    returns ``shuffled_trace(trace, seed)``.
    """
    if kind == FILLER_KIND:
        return filler_text(n_tokens, seed)
    if kind == SHUFFLED_KIND:
        if trace is None:
            raise ValueError("kind='shuffled' requires the trace")
        if count_tokens(trace) != n_tokens:
            raise ValueError(
                f"n_tokens={n_tokens} does not match the trace length {count_tokens(trace)}")
        return shuffled_trace(trace, seed)
    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


def necessity_vs(logp_trace: float, logp_baseline: float) -> float:
    """Necessity relative to a baseline: log P_trace(y) - log P_baseline(y)."""
    return logp_trace - logp_baseline
