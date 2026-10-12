
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
    original_correct_prob = cot_record.get("correct_token_prob", 0.0)

    # 1. Reasoning split
    split_pattern, steps = split_reasoning_into_steps(original_reasoning)

    if paraphrase_mode == "mask":
        rng = random.Random(f"{seed}_{question_id}")
        masker = TokenMasker(mask_ratio=mask_ratio, rng=rng, protect_latex=protect_latex)

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
        "correct_token_prob": correct_prob,
        "top_probs": probs,
        "prob_delta": correct_prob - original_correct_prob,
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
    args = parser.parse_args()

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
