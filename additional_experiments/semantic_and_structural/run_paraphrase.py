
import json
import math
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

load_dotenv()

PROVIDER = "openai"
MAX_WORKERS = 128
LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
PARAPHRASE_MODEL = "gpt-4.1-nano"

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

ANSWER_FOLLOWUP = 'Now answer with only a single letter (A, B, C, or D). The answer is'

write_lock = threading.Lock()
paraphraser = None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_prompt(row, template):
    subject = row["subject"].replace("_", " ")
    choices = row["choices"]
    return template.format(
        subject=subject,
        question=row["question"],
        choice_A=choices[0],
        choice_B=choices[1],
        choice_C=choices[2],
        choice_D=choices[3],
    )


def extract_answer_probs(logprobs_content):
    probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    seen = set()
    predicted = None

    if logprobs_content:
        first_token = logprobs_content[0]
        predicted = first_token.token.strip()

        for top_lp in first_token.top_logprobs:
            token = top_lp.token.strip().rstrip(".")
            if token in probs and token not in seen:
                probs[token] = math.exp(top_lp.logprob)
                seen.add(token)

    return probs, predicted


def load_completed_ids(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    completed.add(record["question_id"])
                except json.JSONDecodeError:
                    continue
    return completed


def get_cot_source_dir(model: str) -> Path:
    model_tag = make_model_tag(model)
    candidates = [
        Path("results") / f"exp1_cot_{model_tag}",
        Path("results") / "exp1_cot",
    ]
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError(
        f"CoT results directory not found: {[str(c) for c in candidates]}"
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


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def process_paraphrase(client, model, row, idx, output_path, subject,
                       cot_record, num_paraphrase, seed, paraphrase_mode,
                       mask_ratio=0.0, protect_latex=True):
    global paraphraser
    question_id = f"{subject}_{idx}"
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = LABEL_MAP[row["answer"]]

    original_reasoning = cot_record["reasoning"]
    original_predicted = cot_record.get("predicted")
    original_correct_prob = cot_record.get("correct_token_prob", 0.0)

    # 1. Reasoning split
    split_pattern, steps = split_reasoning_into_steps(original_reasoning)

    if paraphrase_mode == "mask":
        rng = random.Random(f"{seed}_{question_id}")
        masker = TokenMasker(mask_ratio=mask_ratio, rng=rng, protect_latex=protect_latex)

        mask_info = {}
        new_steps = []
        total_masked = 0
        total_tokens = 0
        for si, step in enumerate(steps):
            masked_text, num_masked, num_total = masker.mask(step)
            new_steps.append(masked_text)
            total_masked += num_masked
            total_tokens += num_total
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
        for si in selected_indices:
            paraphrased_text, applied = paraphraser.paraphrase(steps[si])
            new_steps[si] = paraphrased_text
            if applied:
                transformations[str(si)] = applied

        paraphrased_reasoning = "\n\n".join(new_steps)

    try:
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
            top_logprobs=5,
        )

        probs, predicted = extract_answer_probs(resp.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)

    except Exception as e:
        probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        predicted = None
        correct_prob = 0.0

    record = {
        "subject": subject,
        "question_id": question_id,
        "ground_truth": ground_truth,
        "paraphrase_mode": paraphrase_mode,
        "mask_ratio": mask_ratio if paraphrase_mode == "mask" else None,
        "num_paraphrase": num_paraphrase if paraphrase_mode != "mask" else None,
        "original_predicted": original_predicted,
        "original_correct_token_prob": original_correct_prob,
        "original_reasoning": original_reasoning,
        "removed_last_step": "",
        "split_pattern": split_pattern,
        "num_steps": len(steps),
        "paraphrased_step_indices": selected_indices,
        "transformations_applied": transformations,
        "paraphrased_reasoning": paraphrased_reasoning,
        "predicted": predicted,
        "correct_token_prob": correct_prob,
        "top_probs": probs,
        "prob_delta": correct_prob - original_correct_prob,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

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
        paraphraser = LLMParaphraser(client=client, model=PARAPHRASE_MODEL)
        mode_label = "llm"
    else:
        paraphraser = RuleBasedParaphraser()
        mode_label = "rule"

    cot_dir = get_cot_source_dir(model)
    print(f"Loading CoT results from: {cot_dir}")
    cot_map = load_cot_results(cot_dir)
    print(f"  Loaded {len(cot_map)} CoT records")

    mtag = make_model_tag(model)
    if paraphrase_mode == "mask":
        output_dir = Path("results") / f"exp3_mask_k{int(mask_ratio * 100)}{latex_tag}_{mtag}"
    elif paraphrase_mode == "llm":
        output_dir = Path("results") / f"exp3_paraphrase_llm_n{num_paraphrase}_{mtag}"
    else:
        output_dir = Path("results") / f"exp3_paraphrase_n{num_paraphrase}_{mtag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MMLU dataset...")
    dataset = load_dataset("cais/mmlu", "all", split="test")

    subject_data = {}
    for i, row in enumerate(dataset):
        subj = row["subject"]
        if subj not in subject_data:
            subject_data[subj] = []
        subject_data[subj].append((i, row))

    total_subjects = len(subject_data)
    print(f"Running paraphrase experiment (model={model}, mode={mode_label}, "
          f"num_paraphrase={num_paraphrase}, mask_ratio={mask_ratio}, "
          f"subjects={total_subjects}, seed={seed}, workers={MAX_WORKERS})")

    total_processed = 0
    total_skipped = 0
    total_no_cot = 0

    for subj_idx, (subject, items) in enumerate(sorted(subject_data.items()), 1):
        output_path = output_dir / f"{subject}.jsonl"
        completed_ids = load_completed_ids(output_path)

        pending = []
        for idx, row in items:
            question_id = f"{subject}_{idx}"
            if question_id in completed_ids:
                total_skipped += 1
                continue
            if question_id not in cot_map:
                total_no_cot += 1
                continue
            pending.append((idx, row, cot_map[question_id]))

        if not pending:
            print(f"[{subj_idx}/{total_subjects}] {subject}: all {len(items)} done, skipping")
            continue

        print(f"[{subj_idx}/{total_subjects}] {subject}: {len(pending)} to process "
              f"({len(completed_ids)} already done)")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for idx, row, cot_rec in pending:
                futures.append(executor.submit(
                    process_paraphrase,
                    client, model, row, idx, output_path, subject,
                    cot_rec, num_paraphrase, seed, paraphrase_mode,
                    mask_ratio, protect_latex,
                ))

            for _ in tqdm(as_completed(futures), total=len(futures), desc=subject):
                pass

        total_processed += len(pending)

    print(f"\nDone! Processed: {total_processed}, Skipped: {total_skipped}, "
          f"No CoT: {total_no_cot}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run paraphrase experiment on MMLU CoT results")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name (gpt-4.1-mini or gpt-4o-mini)",
    )
    parser.add_argument(
        "--num_paraphrase",
        type=int,
        default=1,
        help="Number of steps to paraphrase (1, 2, 3, ...)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for step selection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rule", "llm", "mask"],
        default="rule",
        help="Paraphrase mode: rule (rule-based), llm (LLM-based), or mask (token masking)",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.1,
        help="Token masking ratio (0.0~1.0). Only used when --mode=mask. (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--no-protect-latex",
        action="store_true",
        default=False,
        help="also mask LaTeX/math (default: LaTeX protected)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "litellm"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    args = parser.parse_args()

    global PROVIDER
    PROVIDER = args.provider

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
