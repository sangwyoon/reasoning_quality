"""GPQA experiment: CoT + Direct + 10% masking KL/JS.
4-choice MCQ. Necessity + Instability integrated script.
Graduate-level science questions.
Qwen3 thinking model support: --thinking-model flag.
"""

import json
import math
import os
import re
import random
import argparse
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag
from paraphrase import split_reasoning_into_steps, TokenMasker, remove_one_step

import measures

load_dotenv()

MODEL = "Qwen/Qwen3-1.7B"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
MASK_RATIO = 0.1
HF_TOKEN = os.environ.get("HF_TOKEN", None)
THINKING_MODEL = False

ANSWER_TOKENS = {"A", "B", "C", "D"}

COT_TEMPLATE = """The following is a multiple choice question about {domain}.

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

DIRECT_TEMPLATE = """The following is a multiple choice question about {domain}.

Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Do not explain or reason. Answer with only a single letter (A, B, C, or D).
The answer is"""

ANSWER_FOLLOWUP = 'Now answer with only a single letter (A, B, C, or D). The answer is'

write_lock = threading.Lock()


def _extra_body(enable_thinking):
    if THINKING_MODEL:
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return None


def _extract_think_block(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def shuffle_choices(row, rng):
    correct = row["Correct Answer"]
    choices = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng.shuffle(choices)
    correct_label = chr(65 + choices.index(correct))
    return choices, correct_label


def build_prompt(row, choices, template=None):
    if template is None:
        template = COT_TEMPLATE
    domain = row.get("High-level domain", "science")
    return template.format(
        domain=domain, question=row["Question"],
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


# ---- Phase 1: CoT ----

def process_cot(client, row, idx, output_path, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, choices)

    error = None
    mc_info = {}
    try:
        # Step 1: Generate reasoning
        eb1 = _extra_body(enable_thinking=True)
        kwargs1 = {}
        if eb1:
            kwargs1["extra_body"] = eb1
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=8192, **kwargs1)
        raw_response = resp1.choices[0].message.content
        reasoning = _extract_think_block(raw_response) if THINKING_MODEL else raw_response

        # Step 2: Extract answer with thinking off
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
        correct_prob = probs.get(ground_truth, 0.0)

    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        reasoning = f"ERROR: {error}"
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "first_token": mc_info.get("first_token"),
        "correct_token_prob": correct_prob,
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


# ---- Phase 2: Direct ----

def process_direct(client, row, idx, output_path, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, choices, DIRECT_TEMPLATE)

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
        correct_prob = probs.get(ground_truth, 0.0)

    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "first_token": mc_info.get("first_token"),
        "correct_token_prob": correct_prob,
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


# ---- Phase 3: KL/JS mask-10% (Instability) ----

def process_kl(client, row, idx, output_path, cot_rec, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, choices)
    seed_base = f"{SEED}_{question_id}"

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    cot_original = reasoning
    dup_original, _ = duplicate_step(reasoning, seed_base)
    sm_original, _ = remove_one_step(reasoning, seed_base)

    cot_masked = apply_mask10(cot_original, f"m10_{seed_base}_cot")
    dup_masked = apply_mask10(dup_original, f"m10_{seed_base}_dup")
    sm_masked = apply_mask10(sm_original, f"m10_{seed_base}_sm") if sm_original is not None else None

    conditions = {
        "cot": cot_original, "cot_m": cot_masked,
        "dup": dup_original, "dup_m": dup_masked,
        "sm": sm_original, "sm_m": sm_masked,
    }

    dists = {}
    infos = {}
    for name, reas in conditions.items():
        if reas is None:
            continue
        res = get_answer_dist(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ], question_id)
        if res is None:
            return write_error_record(output_path, question_id, ground_truth, name)
        dists[name], infos[name] = res

    def compute(orig_key, masked_key):
        od = dists[orig_key]
        md = dists[masked_key]
        return kl_onehot(ground_truth, od), kl_onehot(ground_truth, md), js_div(od, md)

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    if "sm" in dists:
        kl_s, kl_sm, js_s = compute("sm", "sm_m")
    else:
        kl_s = kl_sm = js_s = None

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
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
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--thinking-model", action="store_true",
                        help="Enable thinking toggle for Qwen3 (enable_thinking=True for CoT, False for answer extraction)")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    args = parser.parse_args()
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, MASK_RATIO, HF_TOKEN, THINKING_MODEL
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    MASK_RATIO = args.mask_ratio
    THINKING_MODEL = args.thinking_model
    if args.hf_token:
        HF_TOKEN = args.hf_token

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", token=HF_TOKEN)
    data = list(dataset)

    shuffled = []
    for i, row in enumerate(data):
        choices, correct_label = shuffle_choices(row, random.Random(42 + i))
        shuffled.append((i, row, choices, correct_label))

    print(f"Loaded {len(shuffled)} GPQA questions (thinking_model={THINKING_MODEL})")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        cot_dir = Path("results") / f"exp1_cot_gpqa_{model_tag}"
        cot_dir.mkdir(parents=True, exist_ok=True)
        cot_path = cot_dir / "gpqa.jsonl"
        measures.write_run_config(
            cot_path, model=MODEL, dataset="gpqa", phase="cot", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(cot_path)
        pending = [(i, r, ch, gt) for i, r, ch, gt in shuffled if f"gpqa_{i}" not in completed]
        print(f"Phase 1: CoT (model: {MODEL}, pending: {len(pending)}/{len(shuffled)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, idx, cot_path, ch, gt)
                           for idx, row, ch, gt in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 2: Direct
    if args.phase in ("direct", "all"):
        dir_dir = Path("results") / f"exp2_direct_gpqa_{model_tag}"
        dir_dir.mkdir(parents=True, exist_ok=True)
        dir_path = dir_dir / "gpqa.jsonl"
        measures.write_run_config(
            dir_path, model=MODEL, dataset="gpqa", phase="direct", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(dir_path)
        pending = [(i, r, ch, gt) for i, r, ch, gt in shuffled if f"gpqa_{i}" not in completed]
        print(f"Phase 2: Direct (model: {MODEL}, pending: {len(pending)}/{len(shuffled)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, idx, dir_path, ch, gt)
                           for idx, row, ch, gt in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 3: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path("results") / f"exp1_cot_gpqa_{model_tag}" / "gpqa.jsonl"
        if not cot_path.exists():
            print("CoT results not found. Run with --phase cot first.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        kl_dir = Path("results") / f"kl_mask10_gpqa_{model_tag}"
        kl_dir.mkdir(parents=True, exist_ok=True)
        kl_path = kl_dir / "kl_mask10.jsonl"
        measures.write_run_config(
            kl_path, model=MODEL, dataset="gpqa", phase="kl", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(kl_path)
        pending = [(i, r, ch, gt, cot_map[f"gpqa_{i}"]) for i, r, ch, gt in shuffled
                    if f"gpqa_{i}" not in completed and f"gpqa_{i}" in cot_map]
        print(f"Phase 3: KL/JS (model: {MODEL}, pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kl, client, row, idx, kl_path, cot, ch, gt)
                           for idx, row, ch, gt, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_m10"):
                    pass

        records = []
        with open(kl_path) as f:
            for line in f:
                records.append(json.loads(line))
        records = [r for r in records if r.get("error") is None]
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, GPQA)")
            print(f"{'='*70}")
            for label, oh, ohm, js in [
                ("CoT", "kl_oh_cot", "kl_oh_cot_m", "js_cot_cotm"),
                ("Dup", "kl_oh_dup", "kl_oh_dup_m", "js_dup_dupm"),
                ("SM", "kl_oh_sm", "kl_oh_sm_m", "js_sm_smm"),
            ]:
                rs = [r for r in records if r[oh] is not None]
                if not rs:
                    continue
                o = sum(r[oh] for r in rs) / len(rs)
                m = sum(r[ohm] for r in rs) / len(rs)
                j = sum(r[js] for r in rs) / len(rs)
                print(f"  {label:<4}: KL(oh∥orig)={o:.4f}  KL(oh∥masked)={m:.4f}  Δ={m-o:+.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
