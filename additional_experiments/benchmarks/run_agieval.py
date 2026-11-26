
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
DATA_PATH = "data/agieval_english.jsonl"
ALL_LETTERS = "ABCDE"

COT_TEMPLATE = """The following is a multiple choice question ({task}).

Question: {question}
{options_block}

Think step by step, analyzing the logic behind each option.
Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~

RULES:
- You MAY reason about which options are more or less plausible, and why.
- You MAY state, in theory, what the correct concept or principle is.
- **STOP** before explicitly selecting or labeling any option as the final answer.
- Do NOT write phrases like "therefore the answer is", "the correct option is", "this matches option X", or any equivalent conclusion that names a letter as the answer.
- End your response as if handing off your reasoning to someone else to make the final selection."""

DIRECT_TEMPLATE = """The following is a multiple choice question ({task}).

Question: {question}
{options_block}

Do not explain or reason. Answer with only a single letter ({letters_list}).
The answer is"""

ANSWER_FOLLOWUP_TMPL = "Now answer with only a single letter ({letters_list}). The answer is"

write_lock = threading.Lock()


def letters_for(row):
    return ALL_LETTERS[:len(row["choices"])]


def letters_list_str(letters):
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def build_prompt(row, template):
    letters = letters_for(row)
    options_block = "\n".join(f"{l}. {c}" for l, c in zip(letters, row["choices"]))
    return template.format(
        task=row["task"].replace("_", "-"), question=row["question"],
        options_block=options_block, letters_list=letters_list_str(letters),
    )


def extract_choice_dist(logprobs_content, letters):
    """Return (probs, predicted, info); ValueError if no valid choice is observed.

    Choices missing from the top-k get a tail-mass fill (see measures.py).
    """
    return measures.extract_choice_probs(logprobs_content, letters)


def kl_onehot(correct_token, dist):
    return measures.kl_onehot_choice(dist, correct_token)


def js_div(dist_p, dist_q, letters):
    return measures.js_choice(dist_p, dist_q, letters)


def load_data():
    with open(DATA_PATH) as f:
        return [json.loads(l) for l in f]


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


# ---- Phase 1: CoT ----

def process_cot(client, row, output_path):
    question_id = row["question_id"]
    letters = letters_for(row)
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = letters[row["answer"]]
    followup = ANSWER_FOLLOWUP_TMPL.format(letters_list=letters_list_str(letters))

    error = None
    mc_info = {}
    try:
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=THINKING_COT_MAX_TOKENS if THINKING_MODEL else 1024,
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

        probs, predicted, mc_info = extract_choice_dist(resp2.choices[0].logprobs.content, letters)
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
        "task": row["task"],
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


# ---- Phase 1b: Direct ----

def process_direct(client, row, output_path):
    question_id = row["question_id"]
    letters = letters_for(row)
    prompt = build_prompt(row, DIRECT_TEMPLATE)
    ground_truth = letters[row["answer"]]

    error = None
    mc_info = {}
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)
        probs, predicted, mc_info = extract_choice_dist(resp.choices[0].logprobs.content, letters)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "task": row["task"],
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


# ---- Phase 2: KL/JS mask-10% ----

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


def get_answer_dist(client, row, reasoning_text, letters):
    prompt = build_prompt(row, COT_TEMPLATE)
    followup = ANSWER_FOLLOWUP_TMPL.format(letters_list=letters_list_str(letters))
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning_text},
                {"role": "user", "content": followup},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)
        probs, _, info = extract_choice_dist(resp.choices[0].logprobs.content, letters)
        return probs, info
    except Exception as e:
        print(f"[ERROR] {row['question_id']}: get_answer_dist failed: {e}", file=sys.stderr)
        return None


def process_kl(client, row, output_path, cot_rec):
    question_id = row["question_id"]
    letters = letters_for(row)
    ground_truth = letters[row["answer"]]

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    seed_base = f"{SEED}_{question_id}"
    cot_orig = reasoning
    dup_orig, dup_idx = duplicate_step(reasoning, seed_base)
    sm_orig, sm_idx = remove_one_step(reasoning, seed_base)

    conditions = {
        "cot": (cot_orig, apply_mask10(cot_orig, f"m10_{seed_base}_cot")),
        "dup": (dup_orig, apply_mask10(dup_orig, f"m10_{seed_base}_dup")),
    }
    if sm_orig is not None:
        conditions["sm"] = (sm_orig, apply_mask10(sm_orig, f"m10_{seed_base}_sm"))

    dists = {}
    infos = {}
    for name, (orig, masked) in conditions.items():
        r_o = get_answer_dist(client, row, orig, letters)
        r_m = get_answer_dist(client, row, masked, letters)
        if r_o is None or r_m is None:
            return write_error_record(output_path, question_id, ground_truth, name)
        dists[name] = (r_o[0], r_m[0])
        infos[name] = (r_o[1], r_m[1])

    record = {
        "question_id": question_id,
        "task": row["task"],
        "ground_truth": ground_truth,
        "is_cot_correct": cot_rec.get("predicted") == ground_truth,
        "dup_step_idx": dup_idx,
        "sm_step_idx": sm_idx,
    }
    for name in ("cot", "dup", "sm"):
        if name not in dists:
            record[f"kl_oh_{name}"] = record[f"kl_oh_{name}_m"] = record[f"js_{name}_{name}m"] = None
            continue
        d_o, d_m = dists[name]
        record[f"kl_oh_{name}"] = kl_onehot(ground_truth, d_o)
        record[f"kl_oh_{name}_m"] = kl_onehot(ground_truth, d_m)
        record[f"js_{name}_{name}m"] = js_div(d_o, d_m, letters)
        i_o, i_m = infos[name]
        record[f"choice_mass_observed_{name}"] = i_o["choice_mass_observed"]
        record[f"choice_mass_observed_{name}_m"] = i_m["choice_mass_observed"]
        record[f"n_choices_missing_{name}"] = i_o["n_missing"]
        record[f"n_choices_missing_{name}_m"] = i_m["n_missing"]

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
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    parser.add_argument("--thinking-model", action="store_true",
                        help="Thinking model: thinking on for CoT, off for all other requests")
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    args = parser.parse_args()
    global THINKING_MODEL
    THINKING_MODEL = args.thinking_model
    global MASK_RATIO
    MASK_RATIO = args.mask_ratio
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)
    data = load_data()
    print(f"AGIEval English: {len(data)} questions (model: {MODEL})")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        out_dir = Path("results") / f"exp1_cot_agie_{model_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "agieval.jsonl"
        measures.write_run_config(
            out_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="agieval_en", phase="cot", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(out_path)
        pending = [r for r in data if r["question_id"] not in completed]
        print(f"Phase 1: CoT (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, out_path) for row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 1b: Direct
    if args.phase in ("direct", "all"):
        out_dir = Path("results") / f"exp2_direct_agie_{model_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "agieval.jsonl"
        measures.write_run_config(
            out_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="agieval_en", phase="direct", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(out_path)
        pending = [r for r in data if r["question_id"] not in completed]
        print(f"Phase 1b: Direct (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, out_path) for row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 2: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path("results") / f"exp1_cot_agie_{model_tag}" / "agieval.jsonl"
        if not cot_path.exists():
            print("CoT results not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        out_dir = Path("results") / f"kl_mask10_agie_{model_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "kl_mask10.jsonl"
        measures.write_run_config(
            out_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="agieval_en", phase="kl", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=measures.MC_TOP_LOGPROBS)
        completed = load_completed_ids(out_path)
        pending = [(r, cot_map[r["question_id"]]) for r in data
                   if r["question_id"] not in completed and r["question_id"] in cot_map]
        print(f"Phase 2: KL/JS (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kl, client, row, out_path, cot) for row, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_m10"):
                    pass

        records = [r for r in map(json.loads, open(out_path)) if r.get("error") is None] \
            if out_path.exists() else []
        if records:
            n = len(records)
            print(f"\n{'='*70}\nResults (model: {MODEL}, n={n}, AGIEval-EN)\n{'='*70}")
            for label in ("cot", "dup", "sm"):
                rs = [r for r in records if r[f"kl_oh_{label}"] is not None]
                if not rs:
                    continue
                o = sum(r[f"kl_oh_{label}"] for r in rs) / len(rs)
                m = sum(r[f"kl_oh_{label}_m"] for r in rs) / len(rs)
                j = sum(r[f"js_{label}_{label}m"] for r in rs) / len(rs)
                print(f"  {label:<4}: KL(oh∥orig)={o:.4f}  KL(oh∥masked)={m:.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
