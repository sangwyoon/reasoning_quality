"""Protocol-matched controls for multiple-choice necessity.

For every item the gold-label log-probability is measured under:

  trace             prompt + intact trace + follow-up (three messages)
  matched_template  same three messages, assistant turn = placeholder (trace omitted)
  filler            same structure, trace replaced by neutral filler of equal token count
  shuffled          same structure, trace replaced by its own tokens in random order
  direct            single-turn ``direct_prompt`` (only when the item has one)

Necessity relative to each baseline b is ``log P_trace(y) - log P_b(y)``
(``necessity_vs_<b>`` in the output), so dialogue structure is held constant for
every baseline except ``direct``.

Input (``--items``, JSONL), one object per line:
  {"question_id", "prompt", "reasoning", "ground_truth", "labels" (default A-D),
   "followup" (optional), "direct_prompt" (optional)}

The ``reasoning`` field may come from any source model.  Running the same items
file against different ``--model`` checkpoints therefore feeds one model's traces
to other checkpoints (cross-checkpoint trace experiment).

Output (``--out``, JSONL) gets one record per item with a ``conditions`` mapping;
each condition holds ``dist``, ``logp_gold``, ``info`` (choice_mass_observed,
n_missing, tail_mass, first_token) and ``error``.  A failed condition has ``error`` set and no
numbers; necessity values that depend on it are null.  The record-level ``error``
is null only when every condition succeeded; resume re-runs only items whose
record has a non-null error or is missing.  ``run_config.json`` is written next to
the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import controls
import measures

DEFAULT_LABELS = ("A", "B", "C", "D")
BASELINES = ("matched_template", "filler", "shuffled", "direct")

write_lock = threading.Lock()


def default_followup(labels: Sequence[str]) -> str:
    """Answer-extraction follow-up naming the valid labels."""
    names = ", ".join(labels[:-1]) + f", or {labels[-1]}" if len(labels) > 1 else labels[0]
    return f"Now answer with only a single letter ({names}). The answer is"


def condition_messages(item: Dict[str, Any], seed: int, placeholder: str) -> Dict[str, Optional[List[Dict[str, str]]]]:
    """Messages per condition; None marks a condition that cannot be built."""
    prompt, trace = item["prompt"], item["reasoning"]
    followup = item.get("followup") or default_followup(item.get("labels") or DEFAULT_LABELS)
    n = controls.count_tokens(trace)
    item_seed = f"{seed}_{item['question_id']}"
    # controls take int seeds; derive one deterministically from the string seed
    int_seed = int.from_bytes(hashlib.sha256(item_seed.encode()).digest()[:8], "big")
    msgs: Dict[str, Optional[List[Dict[str, str]]]] = {
        "trace": controls.trace_messages(prompt, trace, followup),
        "matched_template": controls.matched_template_messages(prompt, followup, placeholder),
        "filler": controls.trace_messages(
            prompt, controls.uninformative_text(n, int_seed, "filler"), followup),
        "shuffled": controls.trace_messages(
            prompt, controls.uninformative_text(n, int_seed, "shuffled", trace=trace), followup),
    }
    if item.get("direct_prompt"):
        msgs["direct"] = [{"role": "user", "content": item["direct_prompt"]}]
    return msgs


def query_condition(client: Any, model: str, messages: List[Dict[str, str]],
                    labels: Sequence[str], top_logprobs: int, gold: str) -> Dict[str, Any]:
    """One single-token call; returns a condition record (error set on failure)."""
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=1,
            logprobs=True, top_logprobs=top_logprobs)
        content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
        if not content:
            raise ValueError("empty logprobs content")
        dist, predicted, info = measures.extract_choice_probs(content, labels)
        return {"dist": dist, "predicted": predicted, "info": info,
                "logp_gold": -measures.kl_onehot_choice(dist, gold), "error": None}
    except Exception as e:  # recorded, never replaced by a value
        print(f"[ERROR] condition failed: {e}", file=sys.stderr)
        return {"dist": None, "predicted": None, "info": None, "logp_gold": None, "error": str(e)}


def process_item(client: Any, model: str, item: Dict[str, Any], seed: int,
                 top_logprobs: int, placeholder: str) -> Dict[str, Any]:
    """Compute every condition and the necessity values for one item."""
    qid = item["question_id"]
    labels = tuple(item.get("labels") or DEFAULT_LABELS)
    gold = item["ground_truth"]
    record: Dict[str, Any] = {
        "question_id": qid, "ground_truth": gold, "model": model,
        "n_trace_tokens": controls.count_tokens(item.get("reasoning") or ""),
        "conditions": {}, "necessity": {}, "error": None,
    }
    if not (item.get("reasoning") or "").strip():
        record["error"] = "empty reasoning"
        return record

    messages = condition_messages(item, seed, placeholder)
    for name, msgs in messages.items():
        record["conditions"][name] = query_condition(client, model, msgs, labels, top_logprobs, gold)

    trace_lp = record["conditions"]["trace"]["logp_gold"]
    for name in BASELINES:
        cond = record["conditions"].get(name)
        if cond is None:
            continue
        base_lp = cond["logp_gold"]
        record["necessity"][f"necessity_vs_{name}"] = (
            controls.necessity_vs(trace_lp, base_lp)
            if trace_lp is not None and base_lp is not None else None)

    failed = [n for n, c in record["conditions"].items() if c["error"] is not None]
    if failed:
        record["error"] = "failed conditions: " + ", ".join(failed)
    return record


def load_completed_ids(path: Path) -> set:
    """Question ids with an error-free record."""
    done: set = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None:
                        done.add(rec["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def load_items(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def run(client: Any, model: str, items: Sequence[Dict[str, Any]], out_path: Path,
        seed: int = 42, top_logprobs: int = measures.MC_TOP_LOGPROBS,
        workers: int = 8, placeholder: str = "") -> List[Dict[str, Any]]:
    """Process pending items, append records to ``out_path``, return the new records."""
    completed = load_completed_ids(out_path)
    pending = [it for it in items if it["question_id"] not in completed]
    results: List[Dict[str, Any]] = []

    def work(it: Dict[str, Any]) -> Dict[str, Any]:
        rec = process_item(client, model, it, seed, top_logprobs, placeholder)
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(work, it) for it in pending]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--items", required=True, help="items JSONL")
    p.add_argument("--model", required=True)
    p.add_argument("--provider", choices=["openai", "litellm", "vllm"], default="vllm")
    p.add_argument("--out", required=True, help="output JSONL")
    p.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--placeholder", default="",
                   help="assistant-turn text of the matched_template condition")
    args = p.parse_args(argv)

    from llm_client import create_client
    client = create_client(args.provider)
    items = load_items(Path(args.items))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    measures.write_run_config(
        out_path, model=args.model, provider=args.provider, items=args.items,
        n_items=len(items), seed=args.seed, top_logprobs=args.top_logprobs,
        placeholder=args.placeholder, conditions=["trace", *BASELINES])
    new = run(client, args.model, items, out_path, args.seed, args.top_logprobs,
              args.workers, args.placeholder)
    n_err = sum(1 for r in new if r["error"] is not None)
    print(f"processed {len(new)} items ({n_err} with errors) -> {out_path}")


if __name__ == "__main__":
    main()
