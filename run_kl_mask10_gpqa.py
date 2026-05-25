"""GPQA experiment: CoT + Direct + 10% masking KL/JS.
4-choice MCQ. Necessity + Instability integrated script.
448 questions (graduate-level science).
Qwen3 thinking model support: --thinking-model flag.
"""

import json
import math
import os
import re
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag
from paraphrase import split_reasoning_into_steps, TokenMasker

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

RULES:
- You MAY reason about which options are more or less plausible, and why.
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

ANSWER_FOLLOWUP = "Based on the reasoning above, answer with only a single letter (A, B, C, or D). Do NOT add any additional reasoning, explanation, or modification to the reasoning above. Simply output the final answer. The answer is"

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


def extract_abcd(logprobs_content):
    probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    predicted = None
    if logprobs_content:
        first = logprobs_content[0]
        predicted = first.token.strip()
        for top_lp in first.top_logprobs:
            token = top_lp.token.strip().rstrip(".")
            if token in probs:
                probs[token] = max(probs[token], math.exp(top_lp.logprob))
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    else:
        probs = {k: 0.25 for k in probs}
    return probs, predicted


def kl_onehot(correct_token, dist):
    return -math.log(max(dist.get(correct_token, 1e-10), 1e-10))


def js_div(dist_p, dist_q):
    js = 0.0
    for t in ANSWER_TOKENS:
        pv = max(dist_p.get(t, 1e-10), 1e-10)
        qv = max(dist_q.get(t, 1e-10), 1e-10)
        mv = 0.5 * (pv + qv)
        js += 0.5 * pv * math.log(pv / mv) + 0.5 * qv * math.log(qv / mv)
    return max(js, 0.0)


def get_answer_dist(client, messages):
    try:
        eb = _extra_body(enable_thinking=False)
        kwargs = {}
        if eb:
            kwargs["extra_body"] = eb
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0,
            max_tokens=1, logprobs=True, top_logprobs=5, **kwargs)
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            probs, _ = extract_abcd(resp.choices[0].logprobs.content)
            return probs
    except:
        pass
    return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


def load_completed_ids(output_path):
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["question_id"])
                except:
                    pass
    return completed


def apply_mask10(text, seed_str):
    rng = random.Random(seed_str)
    masker = TokenMasker(mask_ratio=MASK_RATIO, rng=rng, protect_latex=False)
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


def mask_one_step(reasoning, seed_str):
    _, steps = split_reasoning_into_steps(reasoning)
    if len(steps) < 2:
        return reasoning, -1
    rng = random.Random(seed_str)
    idx = rng.randint(0, len(steps) - 1)
    masker = TokenMasker(mask_ratio=1.0, rng=rng, protect_latex=False)
    masked_step, _, _ = masker.mask(steps[idx])
    new_steps = list(steps)
    new_steps[idx] = masked_step
    return "\n\n".join(new_steps), idx


# ---- Phase 1: CoT ----

def process_cot(client, row, idx, output_path, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, choices)

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
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=5, **kwargs2)

        probs, predicted = extract_abcd(resp2.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)

    except Exception as e:
        reasoning = f"ERROR: {str(e)}"
        probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        predicted = None
        correct_prob = 0.0

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct_token_prob": correct_prob,
        "top_probs": probs,
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

    try:
        eb = _extra_body(enable_thinking=False)
        kwargs = {}
        if eb:
            kwargs["extra_body"] = eb
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1, logprobs=True, top_logprobs=5, **kwargs)

        probs, predicted = extract_abcd(resp.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)

    except Exception as e:
        probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        predicted = None
        correct_prob = 0.0

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct_token_prob": correct_prob,
        "top_probs": probs,
    }

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
    if not reasoning or "ERROR" in reasoning:
        return None

    cot_original = reasoning
    dup_original, _ = duplicate_step(reasoning, seed_base)
    sm_original, _ = mask_one_step(reasoning, seed_base)

    cot_masked = apply_mask10(cot_original, f"m10_{seed_base}_cot")
    dup_masked = apply_mask10(dup_original, f"m10_{seed_base}_dup")
    sm_masked = apply_mask10(sm_original, f"m10_{seed_base}_sm")

    conditions = {
        "cot": cot_original, "cot_m": cot_masked,
        "dup": dup_original, "dup_m": dup_masked,
        "sm": sm_original, "sm_m": sm_masked,
    }

    dists = {}
    for name, reas in conditions.items():
        dists[name] = get_answer_dist(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ])

    def compute(orig_key, masked_key):
        od = dists[orig_key]
        md = dists[masked_key]
        return kl_onehot(ground_truth, od), kl_onehot(ground_truth, md), js_div(od, md)

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    kl_s, kl_sm, js_s = compute("sm", "sm_m")

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
        "is_cot_correct": cot_rec.get("predicted") == cot_rec.get("ground_truth"),
        "kl_oh_cot": kl_c, "kl_oh_cot_m": kl_cm, "js_cot_cotm": js_c,
        "kl_oh_dup": kl_d, "kl_oh_dup_m": kl_dm, "js_dup_dupm": js_d,
        "kl_oh_sm": kl_s, "kl_oh_sm_m": kl_sm, "js_sm_smm": js_s,
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
    args = parser.parse_args()

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
                o = sum(r[oh] for r in records) / n
                m = sum(r[ohm] for r in records) / n
                j = sum(r[js] for r in records) / n
                print(f"  {label:<4}: KL(oh∥orig)={o:.4f}  KL(oh∥masked)={m:.4f}  Δ={m-o:+.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
