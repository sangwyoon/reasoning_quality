"""Shared, pure measurement helpers (no I/O, no API calls).

Replaces the copy-pasted helpers in the run_*.py scripts:
  * multiple-choice probability extraction with tail-mass fill
  * one-hot NLL / JS for choice distributions
  * full (tail-completed) JS divergence for open-ended top-k distributions
  * answer equivalence for open-ended (MATH-style) answers, with an optional LLM judge
  * run_config.json writer
"""

from __future__ import annotations

import datetime
import json
import math
import re
import subprocess
from pathlib import Path
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

# Number of top logprobs requested for single-token multiple-choice answers.
MC_TOP_LOGPROBS = 200

# Floor used only to keep log() finite when a probability is exactly 0.
_PROB_FLOOR = 1e-300
# Fill used only in the degenerate case where the returned top-k covers all mass
# (tail_mass == 0) but a valid choice is still missing.
_MIN_FILL = 1e-10

NUMERIC_TOL = 1e-9

# Version of the measurement definitions implemented in this module.
MEASURES_VERSION = "1"

DEFAULT_JUDGE_MODEL = "gpt-4.1-nano"
JUDGE_PROMPT = (
    "You are a math answer equivalence checker. Determine if the given answer is "
    "mathematically equivalent to the correct answer.\n\n"
    "Correct answer: {correct_answer}\n"
    "Given answer: {given_answer}\n\n"
    "Are these two answers mathematically equivalent? Answer with ONLY \"yes\" or \"no\"."
)

CHOICE_LABELS_ABCD = ("A", "B", "C", "D")


def _item_fields(item: Any) -> tuple[str, float]:
    """Return (token, logprob) from an object, dict or 2-tuple."""
    if isinstance(item, Mapping):
        return item["token"], float(item["logprob"])
    if isinstance(item, (tuple, list)):
        return item[0], float(item[1])
    return item.token, float(item.logprob)


def normalize_choice_token(token: str) -> str:
    """Normalise a returned token to a candidate choice label (e.g. ' A.' -> 'A')."""
    return token.strip().rstrip(".")


def choice_distribution(
    top_logprobs_items: Iterable[Any],
    valid_labels: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Build a normalised distribution over ``valid_labels`` from top-k logprobs.

    Choices absent from the returned top-k get a nonzero fill
    ``min(tail_mass, smallest_returned_prob) / n_missing`` where
    ``tail_mass = max(0, 1 - sum(all returned top-k probs))``; the result is then
    renormalised over the valid labels.  Several tokens mapping to the same label
    (e.g. "A" and " A") are summed.

    Returns ``(dist, info)`` with ``info`` keys ``choice_mass_observed`` (sum of
    observed choice probs before fill), ``n_missing``, ``tail_mass``.

    Raises ValueError if no valid label is observed at all (no information; the
    caller must skip the item rather than invent a distribution).
    """
    labels = tuple(valid_labels)
    observed: dict[str, float] = {}
    all_probs: list[float] = []
    for item in top_logprobs_items:
        token, logprob = _item_fields(item)
        prob = math.exp(logprob)
        all_probs.append(prob)
        label = normalize_choice_token(token)
        if label in labels:
            observed[label] = observed.get(label, 0.0) + prob

    if not observed:
        raise ValueError("none of the valid choices appear in the returned top-k logprobs")

    tail_mass = max(0.0, 1.0 - sum(all_probs))
    smallest = min(all_probs)
    missing = [lab for lab in labels if lab not in observed]
    n_missing = len(missing)
    fill = 0.0
    if n_missing:
        fill = max(min(tail_mass, smallest) / n_missing, _MIN_FILL)

    raw = {lab: observed.get(lab, fill) for lab in labels}
    total = sum(raw.values())
    dist = {lab: v / total for lab, v in raw.items()}
    info = {
        "choice_mass_observed": sum(observed.values()),
        "n_missing": n_missing,
        "tail_mass": tail_mass,
    }
    return dist, info


def extract_choice_probs(
    logprobs_content: Sequence[Any],
    valid_labels: Sequence[str],
) -> tuple[dict[str, float], Optional[str], dict[str, Any]]:
    """Convenience wrapper over an API ``logprobs.content`` list (first token only).

    Returns ``(dist, predicted_token, info)``; raises ValueError on empty content
    or when no valid label is observed.
    """
    if not logprobs_content:
        raise ValueError("empty logprobs content")
    first = logprobs_content[0]
    dist, info = choice_distribution(first.top_logprobs, valid_labels)
    return dist, first.token.strip(), info


def kl_onehot_choice(dist: Mapping[str, float], correct_label: str) -> float:
    """NLL of the correct label (= KL(one-hot || dist))."""
    return -math.log(max(dist.get(correct_label, 0.0), _PROB_FLOOR))


def js_choice(
    dist_p: Mapping[str, float],
    dist_q: Mapping[str, float],
    labels: Optional[Sequence[str]] = None,
) -> float:
    """Jensen-Shannon divergence (nats) between two choice distributions."""
    keys = tuple(labels) if labels is not None else tuple(set(dist_p) | set(dist_q))
    js = 0.0
    for key in keys:
        pv = dist_p.get(key, 0.0)
        qv = dist_q.get(key, 0.0)
        mv = 0.5 * (pv + qv)
        if pv > 0:
            js += 0.5 * pv * math.log(pv / mv)
        if qv > 0:
            js += 0.5 * qv * math.log(qv / mv)
    return max(js, 0.0)


def js_open_ended(
    dist_p: Mapping[str, float],
    mass_p: float,
    dist_q: Mapping[str, float],
    mass_q: float,
    vocab_size: int,
) -> float:
    """JS divergence between two top-k distributions completed over the vocabulary.

    Each distribution's unobserved mass ``1 - mass`` is spread uniformly over the
    ``vocab_size - len(dist)`` tokens it did not return.  The sum covers the union
    of explicit tokens plus an aggregate term for the ``n_rest`` tokens in neither.
    """
    union = set(dist_p) | set(dist_q)
    p_def = max(0.0, 1.0 - mass_p) / max(1, vocab_size - len(dist_p))
    q_def = max(0.0, 1.0 - mass_q) / max(1, vocab_size - len(dist_q))

    def term(pv: float, qv: float) -> float:
        mv = 0.5 * (pv + qv)
        out = 0.0
        if pv > 0 and mv > 0:
            out += 0.5 * pv * math.log(pv / mv)
        if qv > 0 and mv > 0:
            out += 0.5 * qv * math.log(qv / mv)
        return out

    js = sum(term(dist_p.get(t, p_def), dist_q.get(t, q_def)) for t in union)
    n_rest = vocab_size - len(union)
    if n_rest > 0:
        js += n_rest * term(p_def, q_def)
    return max(js, 0.0)


def tail_mass(mass: float) -> float:
    """Probability mass not covered by a returned top-k distribution (1 - sum of top-k probs)."""
    return max(0.0, 1.0 - mass)


def mean_tail_mass(results: Sequence[Any]) -> Optional[float]:
    """Mean tail mass over teacher-forced ``(token, dist, mass)`` triples (None if empty)."""
    if not results:
        return None
    return sum(tail_mass(mass) for _, _, mass in results) / len(results)


def completed_token_prob(
    dist: Mapping[str, float],
    mass: float,
    token: str,
    vocab_size: int,
) -> float:
    """P(token) under a top-k distribution completed over the vocabulary.

    A token absent from the top-k receives the tail mass spread uniformly over the
    ``vocab_size - len(dist)`` tokens that were not returned (same completion as
    ``js_open_ended``); the result is never exactly 0.
    """
    if dist.get(token, 0.0) > 0:
        return dist[token]
    return max(tail_mass(mass), 1e-10) / max(1, vocab_size - len(dist))


def git_commit_hash() -> Optional[str]:
    """Current git commit hash, or None when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, cwd=str(Path(__file__).resolve().parent),
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def write_run_config(path: Any, **fields: Any) -> Path:
    """Write ``run_config.json`` into the directory of ``path`` (an output file or directory).

    Always records ``git_commit``, ``timestamp`` (UTC ISO-8601) and ``measures_version``
    in addition to the given fields (model, dataset, seed, mask_ratio, n_probes,
    top_logprobs, ...). Returns the written path.
    """
    p = Path(path)
    directory = p.parent if p.suffix in (".jsonl", ".json") and not p.is_dir() else p
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(fields)
    config.setdefault("top_logprobs_requested", config.get("top_logprobs", MC_TOP_LOGPROBS))
    config["git_commit"] = git_commit_hash()
    config["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    config["measures_version"] = MEASURES_VERSION
    out = directory / "run_config.json"
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n")
    return out


def judge_equivalent(client: Any, model: str, gold: str, pred: str) -> bool:
    """LLM equivalence judge: True iff the model's reply starts with "yes" (case-insensitive).

    Any API failure raises; there is no silent False.
    """
    prompt = JUDGE_PROMPT.format(correct_answer=gold, given_answer=pred)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise ValueError("judge returned no content")
    return content.strip().lower().startswith("yes")


def make_judge(client: Any, model: str = DEFAULT_JUDGE_MODEL) -> Callable[[str, str], bool]:
    """Bind ``judge_equivalent`` to a client/model for use as ``match_answers(judge=...)``."""
    return lambda gold, pred: judge_equivalent(client, model, gold, pred)

_THOUSANDS_RE = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_DECIMAL_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")
_SLASH_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$")
_FRAC_RE = re.compile(r"^([+-]?)\\frac\{([+-]?\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}$")
_LHS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*=(.+)$")


def _unwrap(text: str, command: str) -> str:
    """Replace ``\\command{...}`` (balanced braces) by its content."""
    marker = "\\" + command + "{"
    while True:
        start = text.find(marker)
        if start < 0:
            return text
        depth, i = 1, start + len(marker)
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        end = i if depth == 0 else len(text)
        inner = text[start + len(marker): end - 1 if depth == 0 else end]
        text = text[:start] + inner + text[end:]


def _last_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` (balanced braces) if present, else text."""
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text
    depth, i = 1, start + len(marker)
    begin = i
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[begin: i - 1] if depth == 0 else text[begin:]


def normalize_answer(text: str) -> str:
    """Normalise an answer string for comparison (see answers_equivalent)."""
    s = _last_boxed(text.strip())
    for cmd in ("text", "textbf", "mathrm"):
        s = _unwrap(s, cmd)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for junk in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$"):
        s = s.replace(junk, "")
    s = re.sub(r"\s+", "", s).lower()
    s = s.rstrip(".")
    if _THOUSANDS_RE.match(s):
        s = s.replace(",", "")
    return s


def _strip_lhs(s: str) -> str:
    match = _LHS_RE.match(s)
    return match.group(1) if match else s


def parse_number(s: str) -> Optional[Fraction]:
    """Parse int / decimal / a/b / \\frac{a}{b} (normalised input) into a Fraction."""
    try:
        if _DECIMAL_RE.match(s):
            return Fraction(s)
        match = _SLASH_RE.match(s)
        if match:
            den = Fraction(match.group(2))
            return Fraction(match.group(1)) / den if den else None
        match = _FRAC_RE.match(s)
        if match:
            den = Fraction(match.group(3))
            if not den:
                return None
            value = Fraction(match.group(2)) / den
            return -value if match.group(1) == "-" else value
    except (ValueError, ZeroDivisionError):
        return None
    return None


def match_answers(
    gold: str,
    pred: str,
    judge: Optional[Callable[[str, str], bool]] = None,
) -> tuple[bool, str]:
    """Compare answers; return (is_correct, method) with method in
    exact / numeric / judge / none.  ``judge`` is an optional caller-supplied
    fallback (e.g. an LLM judge); it is never called unless provided.
    """
    if not gold or not pred or not gold.strip() or not pred.strip():
        return False, "none"
    g, p = normalize_answer(gold), normalize_answer(pred)
    if g and g == p:
        return True, "exact"
    # "x=5" vs "5": drop a simple variable on the left of '=' only when gold has none.
    if "=" not in g:
        p = _strip_lhs(p)
        if g and g == p:
            return True, "exact"
    gn, pn = parse_number(g), parse_number(p)
    if gn is not None and pn is not None and abs(gn - pn) <= Fraction(NUMERIC_TOL):
        return True, "numeric"
    if judge is not None and judge(gold, pred):
        return True, "judge"
    return False, "none"


def answers_equivalent(
    gold: str,
    pred: str,
    judge: Optional[Callable[[str, str], bool]] = None,
) -> bool:
    """True if gold and pred are equivalent (normalised-exact or numeric); never substring."""
    return match_answers(gold, pred, judge)[0]
