"""GPQA dataset experiment: CoT vs Direct."""

import json
import math
import os
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

import measures

load_dotenv()

MODEL = "gpt-4.1-nano"
PROVIDER = "openai"
MAX_WORKERS = 32
HF_TOKEN = None

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


def shuffle_choices(row, rng):
    """Shuffle choices and return correct label."""
    correct = row["Correct Answer"]
    choices = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng.shuffle(choices)
    correct_label = chr(65 + choices.index(correct))  # A, B, C, D
    return choices, correct_label


ANSWER_LABELS = ("A", "B", "C", "D")


def extract_answer_probs(logprobs_content):
    """Return (probs, predicted, info); ValueError if no A-D token is observed.

    Choices missing from the top-k get a tail-mass fill (see measures.py).
    """
    return measures.extract_choice_probs(logprobs_content, ANSWER_LABELS)


def load_completed_ids(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("error") is None:
                        completed.add(record["question_id"])
                except json.JSONDecodeError:
                    continue
    return completed


def build_prompt(row, template, choices):
    domain = row.get("High-level domain", "science")
    return template.format(
        domain=domain,
        question=row["Question"],
        choice_A=choices[0],
        choice_B=choices[1],
        choice_C=choices[2],
        choice_D=choices[3],
    )


def process_cot(client, row, idx, output_path, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE, choices)

    error = None
    mc_info = {}
    try:
        resp1 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2048,
        )
        reasoning = resp1.choices[0].message.content

        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=measures.MC_TOP_LOGPROBS,
        )

        probs, predicted, mc_info = extract_answer_probs(resp2.choices[0].logprobs.content)
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
        "subdomain": row.get("Subdomain", ""),
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


def process_direct(client, row, idx, output_path, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, DIRECT_TEMPLATE, choices)

    error = None
    mc_info = {}
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=measures.MC_TOP_LOGPROBS,
        )

        probs, predicted, mc_info = extract_answer_probs(resp.choices[0].logprobs.content)
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
        "subdomain": row.get("Subdomain", ""),
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


def run_experiment(exp_name):
    client = create_client(PROVIDER)
    is_cot = exp_name == "exp1_cot"
    model_tag = make_model_tag(MODEL)
    output_dir = Path("results") / f"{exp_name}_gpqa_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gpqa.jsonl"
    measures.write_run_config(
        output_path, model=MODEL, dataset="gpqa", phase=exp_name, seed=42,
        mask_ratio=None, n_probes=None, top_logprobs=measures.MC_TOP_LOGPROBS)

    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train",
                           token=HF_TOKEN)

    completed_ids = load_completed_ids(output_path)

    # Fix seed for choice shuffling
    rng = random.Random(42)
    shuffled = []
    for i, row in enumerate(dataset):
        choices, correct_label = shuffle_choices(row, random.Random(42 + i))
        shuffled.append((i, row, choices, correct_label))

    pending = []
    for idx, row, choices, correct_label in shuffled:
        qid = f"gpqa_{idx}"
        if qid in completed_ids:
            continue
        pending.append((idx, row, choices, correct_label))

    total = len(dataset)
    done = len(completed_ids)
    print(f"Running '{exp_name}' on GPQA (model: {MODEL}, workers: {MAX_WORKERS})")
    print(f"  Total: {total}, Already done: {done}, Pending: {len(pending)}")

    if not pending:
        print("All questions already processed.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row, choices, correct_label in pending:
            if is_cot:
                futures.append(executor.submit(
                    process_cot, client, row, idx, output_path, choices, correct_label))
            else:
                futures.append(executor.submit(
                    process_direct, client, row, idx, output_path, choices, correct_label))

        for _ in tqdm(as_completed(futures), total=len(futures), desc=exp_name):
            pass

    print(f"Done! Processed: {len(pending)}")


def main():
    parser = argparse.ArgumentParser(description="Run GPQA experiment")
    parser.add_argument("--exp", choices=["cot", "direct", "both"], default="both")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="openai")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    args = parser.parse_args()
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers

    global HF_TOKEN
    HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN", None)

    experiments = []
    if args.exp in ("cot", "both"):
        experiments.append("exp1_cot")
    if args.exp in ("direct", "both"):
        experiments.append("exp2_direct")

    for exp_name in experiments:
        run_experiment(exp_name)


if __name__ == "__main__":
    main()
