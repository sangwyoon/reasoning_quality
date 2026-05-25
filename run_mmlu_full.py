"""MMLU 전체 (14,042문항) 실험: CoT + Direct + 10% masking KL/JS.
4지선다(A~D). Necessity + Instability 통합 스크립트.
"""

import json
import math
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import create_client, make_model_tag
from paraphrase import split_reasoning_into_steps, TokenMasker

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
MASK_RATIO = 0.1
DATA_PATH = "data/mmlu_full.jsonl"

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
ANSWER_TOKENS = {"A", "B", "C", "D"}

COT_TEMPLATE = """The following is a multiple choice question about {subject}.

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

DIRECT_TEMPLATE = """The following is a multiple choice question about {subject}.

Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Do not explain or reason. Answer with only a single letter (A, B, C, or D).
The answer is"""

ANSWER_FOLLOWUP = "Based on the reasoning above, answer with only a single letter (A, B, C, or D). Do NOT add any additional reasoning, explanation, or modification to the reasoning above. Simply output the final answer. The answer is"

write_lock = threading.Lock()


def build_prompt(row, template):
    subject = row["subject"].replace("_", " ")
    choices = row["choices"]
    return template.format(
        subject=subject, question=row["question"],
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


def load_data():
    with open(DATA_PATH) as f:
        return [json.loads(l) for l in f]


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


def get_answer_dist(client, messages):
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0,
            max_tokens=1, logprobs=True, top_logprobs=5)
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            probs, _ = extract_abcd(resp.choices[0].logprobs.content)
            return probs
    except:
        pass
    return {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}


# ---- Phase 1: CoT ----

def process_cot(client, row, idx, output_path):
    question_id = f"mmlu_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]

    try:
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1024)
        reasoning = resp1.choices[0].message.content

        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)

        probs, predicted = extract_abcd(resp2.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception as e:
        reasoning = f"ERROR: {str(e)}"
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        predicted = None
        correct_prob = 0.25

    record = {
        "question_id": question_id,
        "subject": row["subject"],
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

def process_direct(client, row, idx, output_path):
    question_id = f"mmlu_{idx}"
    prompt = build_prompt(row, DIRECT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]

    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)

        probs, predicted = extract_abcd(resp.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception as e:
        probs = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        predicted = None
        correct_prob = 0.25

    record = {
        "question_id": question_id,
        "subject": row["subject"],
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


def process_kl(client, row, idx, output_path, cot_rec):
    question_id = f"mmlu_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]
    seed_base = f"{SEED}_{question_id}"

    reasoning = cot_rec.get("reasoning", "")
    if not reasoning or "ERROR" in reasoning:
        return None

    cot_orig = reasoning
    dup_orig, _ = duplicate_step(reasoning, seed_base)
    sm_orig, _ = mask_one_step(reasoning, seed_base)

    cot_m = apply_mask10(cot_orig, f"m10_{seed_base}_cot")
    dup_m = apply_mask10(dup_orig, f"m10_{seed_base}_dup")
    sm_m = apply_mask10(sm_orig, f"m10_{seed_base}_sm")

    conditions = {"cot": cot_orig, "cot_m": cot_m,
                   "dup": dup_orig, "dup_m": dup_m,
                   "sm": sm_orig, "sm_m": sm_m}

    dists = {}
    for name, reas in conditions.items():
        dists[name] = get_answer_dist(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ])

    def compute(o, m):
        return kl_onehot(ground_truth, dists[o]), kl_onehot(ground_truth, dists[m]), js_div(dists[o], dists[m])

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    kl_s, kl_sm, js_s = compute("sm", "sm_m")

    record = {
        "question_id": question_id, "subject": row["subject"], "ground_truth": ground_truth,
        "is_cot_correct": cot_rec.get("predicted") == ground_truth,
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
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, MASK_RATIO
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    MASK_RATIO = args.mask_ratio

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)
    data = load_data()
    print(f"Loaded {len(data)} MMLU questions (full)")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        cot_dir = Path("results") / f"exp1_cot_mmlu_{model_tag}"
        cot_dir.mkdir(parents=True, exist_ok=True)
        cot_path = cot_dir / "mmlu.jsonl"
        completed = load_completed_ids(cot_path)
        pending = [(i, r) for i, r in enumerate(data) if f"mmlu_{i}" not in completed]
        print(f"Phase 1: CoT (model: {MODEL}, pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, idx, cot_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 2: Direct
    if args.phase in ("direct", "all"):
        dir_dir = Path("results") / f"exp2_direct_mmlu_{model_tag}"
        dir_dir.mkdir(parents=True, exist_ok=True)
        dir_path = dir_dir / "mmlu.jsonl"
        completed = load_completed_ids(dir_path)
        pending = [(i, r) for i, r in enumerate(data) if f"mmlu_{i}" not in completed]
        print(f"Phase 2: Direct (model: {MODEL}, pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, idx, dir_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 3: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path("results") / f"exp1_cot_mmlu_{model_tag}" / "mmlu.jsonl"
        if not cot_path.exists():
            print("CoT results not found. Run with --phase cot first.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        kl_dir = Path("results") / f"kl_mask10_mmlu_{model_tag}"
        kl_dir.mkdir(parents=True, exist_ok=True)
        kl_path = kl_dir / "kl_mask10.jsonl"
        completed = load_completed_ids(kl_path)
        pending = [(i, r, cot_map[f"mmlu_{i}"]) for i, r in enumerate(data)
                    if f"mmlu_{i}" not in completed and f"mmlu_{i}" in cot_map]
        print(f"Phase 3: KL/JS (model: {MODEL}, pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kl, client, row, idx, kl_path, cot) for idx, row, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_m10"):
                    pass

        # Summary
        records = []
        with open(kl_path) as f:
            for line in f:
                records.append(json.loads(line))
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, MMLU Full)")
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
