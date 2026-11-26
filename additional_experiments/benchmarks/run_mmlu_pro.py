
import json
import math
import random
import argparse
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag, extract_think_block, raw_teacher_force_step, thinking_kwargs, THINKING_COT_MAX_TOKENS
from paraphrase import split_reasoning_into_steps, TokenMasker, remove_one_step

import measures

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
MASK_RATIO = 0.1
THINKING_MODEL = False
OUTDIR = "ablation_0524"

ALL_LABELS = [chr(65 + i) for i in range(10)]  # A~J

write_lock = threading.Lock()


def get_labels(n_options):
    return ALL_LABELS[:n_options]


def build_cot_prompt(row):
    options = row["options"]
    n = len(options)
    labels = get_labels(n)
    options_str = "\n".join(f"{labels[i]}. {options[i]}" for i in range(n))
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"""The following is a multiple choice question about {row.get("category", "general knowledge")}.

Question: {row["question"]}
{options_str}

Think step by step, analyzing the logic behind each option.
Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~

RULES:
- You MAY reason about which options are more or less plausible, and why.
- You MAY state, in theory, what the correct concept or principle is.
- **STOP** before explicitly selecting or labeling any option as the final answer.
- Do NOT write phrases like "therefore the answer is", "the correct option is", "this matches option X", or any equivalent conclusion that names a letter as the answer.
- End your response as if handing off your reasoning to someone else to make the final selection."""


def build_direct_prompt(row):
    options = row["options"]
    n = len(options)
    labels = get_labels(n)
    options_str = "\n".join(f"{labels[i]}. {options[i]}" for i in range(n))
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"""The following is a multiple choice question about {row.get("category", "general knowledge")}.

Question: {row["question"]}
{options_str}

Do not explain or reason. Answer with only a single letter ({label_list}).
The answer is"""


def build_followup(row):
    n = len(row["options"])
    labels = get_labels(n)
    label_list = ", ".join(labels[:-1]) + f", or {labels[-1]}"
    return f"Now answer with only a single letter ({label_list}). The answer is"


def extract_distribution(logprobs_content, labels):
    """Return (probs, predicted, info); ValueError if no valid choice is observed.

    Choices missing from the top-k get a tail-mass fill (see measures.py).
    """
    return measures.extract_choice_probs(logprobs_content, labels)


def kl_onehot(correct_token, dist):
    return measures.kl_onehot_choice(dist, correct_token)


def js_div(dist_p, dist_q, labels):
    return measures.js_choice(dist_p, dist_q, labels)


def load_data():
    print("Loading MMLU-Pro from HuggingFace...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return list(dataset)


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


def get_answer_dist(client, messages, labels, question_id="?"):
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0,
            max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            probs, _, info = extract_distribution(resp.choices[0].logprobs.content, labels)
            return probs, info
        raise ValueError("empty logprobs content")
    except Exception as e:
        print(f"[ERROR] {question_id}: get_answer_dist failed: {e}", file=sys.stderr)
        return None


# ---- Phase 1: CoT ----

def process_cot(client, row, idx, output_path):
    question_id = f"mmlupro_{idx}"
    prompt = build_cot_prompt(row)
    followup = build_followup(row)
    ground_truth = row["answer"]
    labels = get_labels(len(row["options"]))

    error = None
    mc_info = {}
    try:
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=THINKING_COT_MAX_TOKENS if THINKING_MODEL else 2048,
            **thinking_kwargs(THINKING_MODEL, True))
        reasoning = resp1.choices[0].message.content
        if THINKING_MODEL:
            reasoning = extract_think_block(reasoning)

        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": followup},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)

        probs, predicted, mc_info = extract_distribution(resp2.choices[0].logprobs.content, labels)
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
        "category": row.get("category", ""),
        "n_options": len(row["options"]),
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

def process_direct(client, row, idx, output_path):
    question_id = f"mmlupro_{idx}"
    prompt = build_direct_prompt(row)
    ground_truth = row["answer"]
    labels = get_labels(len(row["options"]))

    error = None
    mc_info = {}
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)

        probs, predicted, mc_info = extract_distribution(resp.choices[0].logprobs.content, labels)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "category": row.get("category", ""),
        "n_options": len(row["options"]),
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


# ---- Phase 3: KL/JS mask-10% (Instability) ----

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
    question_id = f"mmlupro_{idx}"
    prompt = build_cot_prompt(row)
    followup = build_followup(row)
    ground_truth = row["answer"]
    labels = get_labels(len(row["options"]))
    seed_base = f"{SEED}_{question_id}"

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    cot_orig = reasoning
    dup_orig, _ = duplicate_step(reasoning, seed_base)
    sm_orig, _ = remove_one_step(reasoning, seed_base)

    cot_m = apply_mask10(cot_orig, f"m10_{seed_base}_cot")
    dup_m = apply_mask10(dup_orig, f"m10_{seed_base}_dup")
    sm_m = apply_mask10(sm_orig, f"m10_{seed_base}_sm") if sm_orig is not None else None

    conditions = {"cot": cot_orig, "cot_m": cot_m,
                   "dup": dup_orig, "dup_m": dup_m,
                   "sm": sm_orig, "sm_m": sm_m}

    dists = {}
    infos = {}
    for name, reas in conditions.items():
        if reas is None:
            continue
        res = get_answer_dist(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": followup},
        ], labels, question_id)
        if res is None:
            return write_error_record(output_path, question_id, ground_truth, name)
        dists[name], infos[name] = res

    def compute(o, m):
        return kl_onehot(ground_truth, dists[o]), kl_onehot(ground_truth, dists[m]), js_div(dists[o], dists[m], labels)

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    if "sm" in dists:
        kl_s, kl_sm, js_s = compute("sm", "sm_m")
    else:
        kl_s = kl_sm = js_s = None

    record = {
        "question_id": question_id, "category": row.get("category", ""), "ground_truth": ground_truth,
        "is_cot_correct": cot_rec.get("predicted") == ground_truth,
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
    parser.add_argument("--outdir", type=str, default="ablation_0524")
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    parser.add_argument("--thinking-model", action="store_true",
                        help="Thinking model: thinking on for CoT, off for all other requests")
    args = parser.parse_args()
    global THINKING_MODEL
    THINKING_MODEL = args.thinking_model
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, MASK_RATIO, OUTDIR
    OUTDIR = args.outdir
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    MASK_RATIO = args.mask_ratio

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)
    data = load_data()
    print(f"Loaded {len(data)} MMLU-Pro questions")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        cot_dir = Path(OUTDIR) / f"exp1_cot_mmlupro_{model_tag}"
        cot_dir.mkdir(parents=True, exist_ok=True)
        cot_path = cot_dir / "mmlu_pro.jsonl"
        measures.write_run_config(
            cot_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="mmlu_pro", phase="cot", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(cot_path)
        pending = [(i, r) for i, r in enumerate(data) if f"mmlupro_{i}" not in completed]
        print(f"Phase 1: CoT (model: {MODEL}, pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, idx, cot_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 2: Direct
    if args.phase in ("direct", "all"):
        dir_dir = Path(OUTDIR) / f"exp2_direct_mmlupro_{model_tag}"
        dir_dir.mkdir(parents=True, exist_ok=True)
        dir_path = dir_dir / "mmlu_pro.jsonl"
        measures.write_run_config(
            dir_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="mmlu_pro", phase="direct", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(dir_path)
        pending = [(i, r) for i, r in enumerate(data) if f"mmlupro_{i}" not in completed]
        print(f"Phase 2: Direct (model: {MODEL}, pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, idx, dir_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 3: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path(OUTDIR) / f"exp1_cot_mmlupro_{model_tag}" / "mmlu_pro.jsonl"
        if not cot_path.exists():
            print("CoT results not found. Run with --phase cot first.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        kl_dir = Path(OUTDIR) / f"kl_mask10_mmlupro_{model_tag}"
        kl_dir.mkdir(parents=True, exist_ok=True)
        kl_path = kl_dir / "kl_mask10.jsonl"
        measures.write_run_config(
            kl_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="mmlu_pro", phase="kl", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(kl_path)
        pending = [(i, r, cot_map[f"mmlupro_{i}"]) for i, r in enumerate(data)
                    if f"mmlupro_{i}" not in completed and f"mmlupro_{i}" in cot_map]
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
        records = [r for r in records if r.get("error") is None]
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, MMLU-Pro)")
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
