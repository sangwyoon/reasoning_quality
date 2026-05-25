"""GPQA 데이터셋 실험: CoT vs Direct."""

import json
import math
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag

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


def shuffle_choices(row, rng):
    """선택지를 셔플하고 정답 라벨을 반환한다."""
    correct = row["Correct Answer"]
    choices = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng.shuffle(choices)
    correct_label = chr(65 + choices.index(correct))  # A, B, C, D
    return choices, correct_label


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
            top_logprobs=5,
        )

        probs, predicted = extract_answer_probs(resp2.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)

    except Exception as e:
        reasoning = f"ERROR: {str(e)}"
        probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        predicted = None
        correct_prob = 0.0

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "subdomain": row.get("Subdomain", ""),
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


def process_direct(client, row, idx, output_path, choices, ground_truth):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, DIRECT_TEMPLATE, choices)

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
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
        "predicted": predicted,
        "correct_token_prob": correct_prob,
        "top_probs": probs,
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

    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train",
                           token=HF_TOKEN)

    completed_ids = load_completed_ids(output_path)

    # 선택지 셔플을 위한 시드 고정
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
    args = parser.parse_args()

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
