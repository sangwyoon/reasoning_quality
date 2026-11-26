
import json
import os
import math
import sys
import argparse
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from paraphrase import (
    split_reasoning_into_steps,
    RuleBasedParaphraser,
    LLMParaphraser,
    TokenMasker,
    select_steps_to_paraphrase,
)
from llm_client import create_client, make_model_tag
import measures

load_dotenv()

PROVIDER = "openai"
MAX_WORKERS = 32
PARAPHRASE_MODEL = "gpt-4.1-nano"
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

ANSWER_FOLLOWUP = 'Now answer with only a single letter (A, B, C, or D). The answer is'

write_lock = threading.Lock()
paraphraser = None


def shuffle_choices(row, rng):
    correct = row["Correct Answer"]
    choices = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng.shuffle(choices)
    correct_label = chr(65 + choices.index(correct))
    return choices, correct_label


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


def get_cot_source_dir(model: str) -> Path:
    model_tag = make_model_tag(model)
    candidates = [
        Path("results") / f"exp1_cot_gpqa_{model_tag}",
    ]
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError(
        f"GPQA CoT results directory not found: {[str(c) for c in candidates]}"
    )


def load_cot_results(cot_dir: Path) -> dict:
    cot_map = {}
    for jsonl_file in sorted(cot_dir.glob("*.jsonl")):
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cot_map[rec["question_id"]] = rec
                except json.JSONDecodeError:
                    continue
    return cot_map


def process_paraphrase(client, model, row, idx, output_path,
                       cot_record, choices, ground_truth,
                       num_paraphrase, seed, paraphrase_mode,
                       mask_ratio=0.0, protect_latex=True):
    global paraphraser
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE, choices)

    original_reasoning = cot_record["reasoning"]
    original_predicted = cot_record.get("predicted")
    original_correct_prob = cot_record.get("correct_token_prob")

    # 1. Reasoning split
    split_pattern, steps = split_reasoning_into_steps(original_reasoning)
    failure = None

    if paraphrase_mode == "mask":
        rng = random.Random(f"{seed}_{question_id}_mask_p0")
        masker = TokenMasker(mask_ratio=mask_ratio, rng=rng)

        mask_info = {}
        new_steps = []
        for si, step in enumerate(steps):
            masked_text, num_masked, num_total = masker.mask(step)
            new_steps.append(masked_text)
            if num_masked > 0:
                mask_info[str(si)] = f"{num_masked}/{num_total} tokens masked"

        paraphrased_reasoning = "\n\n".join(new_steps)
        selected_indices = list(range(len(steps)))
        transformations = mask_info
    else:
        rng = random.Random(f"{seed}_{question_id}")
        selected_indices = select_steps_to_paraphrase(steps, num_paraphrase, rng)

        transformations = {}
        new_steps = list(steps)
        try:
            for si in selected_indices:
                paraphrased_text, applied = paraphraser.paraphrase(steps[si])
                new_steps[si] = paraphrased_text
                if applied:
                    transformations[str(si)] = applied
            paraphrased_reasoning = "\n\n".join(new_steps)
        except Exception as e:
            failure, paraphrased_reasoning = e, None

    error = None
    try:
        if cot_record.get("error") is not None:
            raise ValueError(f"CoT record has an error: {cot_record['error']}")
        if failure is not None:
            raise failure
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": paraphrased_reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ],
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
        mc_info = {}
        probs = None
        predicted = None
        correct_prob = None

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "subdomain": row.get("Subdomain", ""),
        "ground_truth": ground_truth,
        "paraphrase_mode": paraphrase_mode,
        "mask_ratio": mask_ratio if paraphrase_mode == "mask" else None,
        "original_predicted": original_predicted,
        "original_correct_token_prob": original_correct_prob,
        "original_reasoning": original_reasoning,
        "split_pattern": split_pattern,
        "num_steps": len(steps),
        "paraphrased_step_indices": selected_indices,
        "transformations_applied": transformations,
        "paraphrased_reasoning": paraphrased_reasoning,
        "predicted": predicted,
        "first_token": mc_info.get("first_token"),
        "correct_token_prob": correct_prob,
        "top_probs": probs,
        "prob_delta": (None if correct_prob is None or original_correct_prob is None
                       else correct_prob - original_correct_prob),
        "choice_mass_observed": mc_info.get("choice_mass_observed"),
        "n_choices_missing": mc_info.get("n_missing"),
        "error": error,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def run_paraphrase_experiment(model, num_paraphrase, seed=42,
                              paraphrase_mode="rule", mask_ratio=0.1,
                              protect_latex=True):
    global paraphraser
    client = create_client(PROVIDER)

    latex_tag = "" if protect_latex else "_nolatex"

    if paraphrase_mode == "mask":
        paraphraser = None
        mode_label = f"mask_k{int(mask_ratio * 100)}{latex_tag}"
    elif paraphrase_mode == "llm":
        from openai import OpenAI
        paraphraser = LLMParaphraser(client=OpenAI(), model=PARAPHRASE_MODEL)
        mode_label = "llm"
    else:
        paraphraser = RuleBasedParaphraser()
        mode_label = "rule"

    cot_dir = get_cot_source_dir(model)
    print(f"Loading GPQA CoT results from: {cot_dir}")
    cot_map = load_cot_results(cot_dir)
    print(f"  Loaded {len(cot_map)} CoT records")

    mtag = make_model_tag(model)
    if paraphrase_mode == "mask":
        output_dir = Path("results") / f"exp3_mask_k{int(mask_ratio * 100)}{latex_tag}_gpqa_{mtag}"
    elif paraphrase_mode == "llm":
        output_dir = Path("results") / f"exp3_paraphrase_llm_n{num_paraphrase}_gpqa_{mtag}"
    else:
        output_dir = Path("results") / f"exp3_paraphrase_n{num_paraphrase}_gpqa_{mtag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gpqa.jsonl"
    measures.write_run_config(
        output_path, model=model, dataset="gpqa", phase=mode_label, seed=seed,
        mask_ratio=mask_ratio if paraphrase_mode == "mask" else None, n_probes=1,
        top_logprobs=measures.MC_TOP_LOGPROBS, num_paraphrase=num_paraphrase)

    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train",
                           token=HF_TOKEN)

    completed_ids = load_completed_ids(output_path)

    pending = []
    no_cot = 0
    for i, row in enumerate(dataset):
        qid = f"gpqa_{i}"
        if qid in completed_ids:
            continue
        if qid not in cot_map:
            no_cot += 1
            continue
        choices, correct_label = shuffle_choices(row, random.Random(42 + i))
        pending.append((i, row, choices, correct_label, cot_map[qid]))

    total = len(dataset)
    done = len(completed_ids)
    print(f"Running paraphrase experiment on GPQA "
          f"(model={model}, mode={mode_label}, mask_ratio={mask_ratio}, "
          f"seed={seed}, workers={MAX_WORKERS})")
    print(f"  Total: {total}, Already done: {done}, Pending: {len(pending)}, No CoT: {no_cot}")

    if not pending:
        print("All questions already processed.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row, choices, correct_label, cot_rec in pending:
            futures.append(executor.submit(
                process_paraphrase,
                client, model, row, idx, output_path,
                cot_rec, choices, correct_label,
                num_paraphrase, seed, paraphrase_mode,
                mask_ratio, protect_latex,
            ))

        for _ in tqdm(as_completed(futures), total=len(futures), desc=mode_label):
            pass

    print(f"\nDone! Processed: {len(pending)}")
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run paraphrase experiment on GPQA")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="openai")
    parser.add_argument("--num_paraphrase", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["rule", "llm", "mask"], default="rule")
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-protect-latex", action="store_true", default=False)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    args = parser.parse_args()
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    global PROVIDER, HF_TOKEN
    PROVIDER = args.provider
    HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN", None)
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers

    run_paraphrase_experiment(
        model=args.model,
        num_paraphrase=args.num_paraphrase,
        seed=args.seed,
        paraphrase_mode=args.mode,
        mask_ratio=args.mask_ratio,
        protect_latex=not args.no_protect_latex,
    )


if __name__ == "__main__":
    main()
