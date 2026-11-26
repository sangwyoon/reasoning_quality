"""MMLU experiment (Qwen3 thinking model): CoT + Direct + 10% masking KL/JS.
4-choice MCQ. Based on run_kl_mask10_gpqa.py structure.
"""

import json
import math
import random
import argparse
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import create_client, make_model_tag
from paraphrase import split_reasoning_into_steps, TokenMasker, remove_one_step

import measures

load_dotenv()

MODEL = "Qwen/Qwen3-0.6B"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
MASK_RATIO = 0.1
THINKING_MODEL = False
DATA_PATH = "data/mmlu_full.jsonl"

ANSWER_TOKENS = {"A", "B", "C", "D"}
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

COT_TEMPLATE = """The following is a multiple choice question about {subject}.

Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Think step by step, analyzing the logic behind each option.
Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~

RULES:
- You MAY reason about which options are more or less plausible, and why.
- You MAY state, in theory, what the correct concept or principle is.
- **STOP** before explicitly selecting or labeling any option as the final answer.
- Do NOT write phrases like "therefore the answer is", "the correct option is", "this matches option X", or any equivalent conclusion that names a letter as the answer.
- End your response as if handing off your reasoning to someone else to make the final selection."""

DIRECT_TEMPLATE = """The following is a multiple choice question about {subject}.

Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Do not explain or reason. Answer with only a single letter (A, B, C, or D).
The answer is"""

ANSWER_FOLLOWUP = 'Now answer with only a single letter (A, B, C, or D). The answer is'

write_lock = threading.Lock()

import re

def _extra_body(enable_thinking):
    if THINKING_MODEL:
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return None


def _extract_think_block(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text


def build_prompt(row, template):
    subject = row["subject"].replace("_", " ")
    choices = row["choices"]
    return template.format(
        subject=subject, question=row["question"],
        choice_A=choices[0], choice_B=choices[1],
        choice_C=choices[2], choice_D=choices[3],
    )


ANSWER_LABELS = tuple(sorted(ANSWER_TOKENS))


def extract_abcd(logprobs_content):
    """Return (probs, predicted, info); ValueError if no A-D token is observed.

    Choices missing from the top-k get a tail-mass fill (see measures.py).
    """
    return measures.extract_choice_probs(logprobs_content, ANSWER_LABELS)


def kl_onehot(correct_token, dist):
    return measures.kl_onehot_choice(dist, correct_token)


def js_div(dist_p, dist_q):
    return measures.js_choice(dist_p, dist_q, ANSWER_LABELS)


def get_answer_dist(client, messages, question_id="?"):
    try:
        eb = _extra_body(enable_thinking=False)
        kwargs = {}
        if eb:
            kwargs["extra_body"] = eb
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0,
            max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS, **kwargs)
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            probs, _, info = extract_abcd(resp.choices[0].logprobs.content)
            return probs, info
        raise ValueError("empty logprobs content")
    except Exception as e:
        print(f"[ERROR] {question_id}: get_answer_dist failed: {e}", file=sys.stderr)
        return None


def load_completed_ids(output_path):
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("error") is None:
                        completed.add(record["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def apply_mask10(text, seed_str, probe_id=0):
    """Mask the reasoning trace: each token independently with probability MASK_RATIO."""
    rng = random.Random(f"{seed_str}_p{probe_id}")
    masker = TokenMasker(mask_ratio=MASK_RATIO, rng=rng)
    masked, _, _ = masker.mask(text)
    return masked


def duplicate_step(reasoning, seed_str):
    _, steps = split_reasoning_into_steps(reasoning)
    if len(steps) < 2:
        return reasoning, -1
    rng = random.Random(seed_str)
    idx = rng.randint(0, len(steps) - 1)
    new_steps = list(steps)
    new_steps.insert(idx + 1, steps[idx])
    return "\n\n".join(new_steps), idx


def process_cot(client, row, idx, output_path):
    question_id = f"mmlu_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]

    error = None
    mc_info = {}
    try:
        eb1 = _extra_body(enable_thinking=True)
        kwargs1 = {}
        if eb1:
            kwargs1["extra_body"] = eb1
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=8192, **kwargs1)
        raw = resp1.choices[0].message.content
        reasoning = _extract_think_block(raw) if THINKING_MODEL else raw

        eb2 = _extra_body(enable_thinking=False)
        kwargs2 = {}
        if eb2:
            kwargs2["extra_body"] = eb2
        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS, **kwargs2)
        probs, predicted, mc_info = extract_abcd(resp2.choices[0].logprobs.content)

    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        reasoning = f"ERROR: {error}"
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "subject": row["subject"],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "first_token": mc_info.get("first_token"),
        "correct_token_prob": probs.get(ground_truth, 0.0) if probs else None,
        "top_probs": probs,
        "choice_mass_observed": mc_info.get("choice_mass_observed"),
        "n_choices_missing": mc_info.get("n_missing"),
        "error": error,
        "reasoning": reasoning,
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def process_direct(client, row, idx, output_path):
    question_id = f"mmlu_{idx}"
    prompt = build_prompt(row, DIRECT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]

    error = None
    mc_info = {}
    try:
        eb = _extra_body(enable_thinking=False)
        kwargs = {}
        if eb:
            kwargs["extra_body"] = eb
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS, **kwargs)
        probs, predicted, mc_info = extract_abcd(resp.choices[0].logprobs.content)

    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "subject": row["subject"],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "first_token": mc_info.get("first_token"),
        "correct_token_prob": probs.get(ground_truth, 0.0) if probs else None,
        "top_probs": probs,
        "choice_mass_observed": mc_info.get("choice_mass_observed"),
        "n_choices_missing": mc_info.get("n_missing"),
        "error": error,
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


KL_METRIC_KEYS = ("is_cot_correct", "kl_oh_cot", "kl_oh_cot_m", "js_cot_cotm", "kl_oh_dup",
                  "kl_oh_dup_m", "js_dup_dupm", "kl_oh_sm", "kl_oh_sm_m", "js_sm_smm")


def write_error_record(output_path, question_id, ground_truth, condition):
    """Store a record with ``error`` and empty metrics for an item whose request failed."""
    record = {"question_id": question_id, "ground_truth": ground_truth,
              "error": f"answer distribution failed for condition {condition!r}"}
    record.update({k: None for k in KL_METRIC_KEYS})
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def process_kl(client, row, idx, output_path, cot_rec):
    question_id = f"mmlu_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]
    seed_base = f"{SEED}_{question_id}"

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    cot_original = reasoning
    dup_original, _ = duplicate_step(reasoning, seed_base)
    sm_original, _ = remove_one_step(reasoning, seed_base)

    conditions = {
        "cot": cot_original, "cot_m": apply_mask10(cot_original, f"m10_{seed_base}_cot"),
        "dup": dup_original, "dup_m": apply_mask10(dup_original, f"m10_{seed_base}_dup"),
    }
    if sm_original is not None:
        conditions["sm"] = sm_original
        conditions["sm_m"] = apply_mask10(sm_original, f"m10_{seed_base}_sm")
    dists = {}
    infos = {}
    for name, reas in conditions.items():
        res = get_answer_dist(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ], question_id)
        if res is None:
            return write_error_record(output_path, question_id, ground_truth, name)
        dists[name], infos[name] = res

    def compute(o, m):
        return kl_onehot(ground_truth, dists[o]), kl_onehot(ground_truth, dists[m]), js_div(dists[o], dists[m])

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    if "sm" in dists:
        kl_s, kl_sm, js_s = compute("sm", "sm_m")
    else:
        kl_s = kl_sm = js_s = None

    record = {
        "question_id": question_id, "subject": row["subject"], "ground_truth": ground_truth,
        "is_cot_correct": cot_rec.get("predicted") == cot_rec.get("ground_truth"),
        "kl_oh_cot": kl_c, "kl_oh_cot_m": kl_cm, "js_cot_cotm": js_c,
        "kl_oh_dup": kl_d, "kl_oh_dup_m": kl_dm, "js_dup_dupm": js_d,
        "kl_oh_sm": kl_s, "kl_oh_sm_m": kl_sm, "js_sm_smm": js_s,
        "choice_mass_observed": {k: v["choice_mass_observed"] for k, v in infos.items()},
        "n_choices_missing": {k: v["n_missing"] for k, v in infos.items()},
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="vllm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--phase", choices=["cot", "direct", "kl", "all"], default="all")
    parser.add_argument("--thinking-model", action="store_true")
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    args = parser.parse_args()
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, THINKING_MODEL
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    THINKING_MODEL = args.thinking_model

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f]
    print(f"Loaded {len(data)} MMLU questions from {DATA_PATH} (thinking_model={THINKING_MODEL})")

    if args.phase in ("cot", "all"):
        d = Path("results") / f"exp1_cot_mmlu_{model_tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "mmlu.jsonl"
        measures.write_run_config(
            p, model=MODEL, dataset="mmlu", phase="cot", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        done = load_completed_ids(p)
        pending = [(i, r) for i, r in enumerate(data) if f"mmlu_{i}" not in done]
        print(f"Phase 1: CoT (pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_cot, client, r, i, p) for i, r in pending]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="CoT"):
                    pass

    if args.phase in ("direct", "all"):
        d = Path("results") / f"exp2_direct_mmlu_{model_tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "mmlu.jsonl"
        measures.write_run_config(
            p, model=MODEL, dataset="mmlu", phase="cot", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        done = load_completed_ids(p)
        pending = [(i, r) for i, r in enumerate(data) if f"mmlu_{i}" not in done]
        print(f"Phase 2: Direct (pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_direct, client, r, i, p) for i, r in pending]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="Direct"):
                    pass

    if args.phase in ("kl", "all"):
        cot_path = Path("results") / f"exp1_cot_mmlu_{model_tag}" / "mmlu.jsonl"
        if not cot_path.exists():
            print("CoT results not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        d = Path("results") / f"kl_mask10_mmlu_{model_tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "kl_mask10.jsonl"
        measures.write_run_config(
            p, model=MODEL, dataset="mmlu", phase="kl", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        done = load_completed_ids(p)
        pending = [(i, r, cot_map[f"mmlu_{i}"]) for i, r in enumerate(data)
                    if f"mmlu_{i}" not in done and f"mmlu_{i}" in cot_map]
        print(f"Phase 3: KL/JS (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_kl, client, r, i, p, c) for i, r, c in pending]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="KL_m10"):
                    pass

        records = []
        with open(p) as f:
            records = [r for r in map(json.loads, f) if r.get("error") is None]
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, MMLU)")
            print(f"{'='*70}")
            for label, oh, ohm, js in [("CoT","kl_oh_cot","kl_oh_cot_m","js_cot_cotm"),("Dup","kl_oh_dup","kl_oh_dup_m","js_dup_dupm"),("SM","kl_oh_sm","kl_oh_sm_m","js_sm_smm")]:
                rs = [r for r in records if r[oh] is not None]
                if not rs:
                    continue
                n = len(rs)
                o = sum(r[oh] for r in rs)/n
                m = sum(r[ohm] for r in rs)/n
                j = sum(r[js] for r in rs)/n
                print(f"  {label:<4}: KL(orig)={o:.4f}  KL(masked)={m:.4f}  D={m-o:+.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
