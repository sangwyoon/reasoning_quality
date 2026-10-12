import json
import math
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag, get_tokenizer

load_dotenv()

MODEL = "gpt-4o-mini"
PROVIDER = "openai"
MAX_WORKERS = 64

COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Think step by step, analyzing the problem carefully.

RULES:
- You MAY reason about the problem and show your work.
- **STOP** before stating your final answer.
- Do NOT write phrases like "therefore the answer is" or any equivalent conclusion.
- End your response as if handing off your reasoning to someone else to state the final answer.

Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~"""

DIRECT_TEMPLATE = """Solve the following math problem. Do not explain or show any work.
Answer with ONLY the final answer (a number, expression, or short mathematical notation). Nothing else.

Problem: {problem}

Answer:"""

ANSWER_FOLLOWUP = 'Now state ONLY the final answer\n(a number, expression, or short mathematical notation). Nothing else.\n\nAnswer:'

JUDGE_TEMPLATE = """You are a math answer equivalence checker. Determine if the given answer is mathematically equivalent to the correct answer.

Correct answer: {correct_answer}
Given answer: {given_answer}

Are these two answers mathematically equivalent? Answer with ONLY "yes" or "no"."""

write_lock = threading.Lock()


def compute_avg_token_prob(logprobs_content):
    if not logprobs_content:
        return 0.0
    probs = [math.exp(tok.logprob) for tok in logprobs_content]
    return sum(probs) / len(probs)


def teacher_force_correct_prob(client, model, messages_before_answer, correct_answer):
    if not correct_answer or not correct_answer.strip():
        return 0.0, []

    enc = get_tokenizer(model)
    correct_token_ids = enc.encode(correct_answer.strip())
    correct_tokens = [enc.decode([tid]) for tid in correct_token_ids]

    if not correct_tokens:
        return 0.0, []

    token_probs = []
    forced_prefix = ""

    for target_token in correct_tokens:
        msgs = list(messages_before_answer)
        if forced_prefix:
            msgs.append({"role": "assistant", "content": forced_prefix})

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
            )

            found_prob = 0.0
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                for top_lp in resp.choices[0].logprobs.content[0].top_logprobs:
                    if top_lp.token == target_token:
                        found_prob = math.exp(top_lp.logprob)
                        break
            token_probs.append(found_prob)

        except Exception:
            token_probs.append(0.0)

        forced_prefix += target_token

    avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
    return avg_prob, token_probs


def judge_answer(client, correct_answer, given_answer):
    if not given_answer or not given_answer.strip():
        return False
    prompt = JUDGE_TEMPLATE.format(
        correct_answer=correct_answer,
        given_answer=given_answer,
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        verdict = resp.choices[0].message.content.strip().lower()
        return verdict.startswith("yes")
    except Exception:
        return False


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


def process_cot(client, row, idx, output_path):
    question_id = f"omnimath_{idx}"
    problem = row["problem"]
    correct_answer = row["answer"]

    try:
        prompt = COT_TEMPLATE.format(problem=problem)

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
            max_tokens=100,
            logprobs=True,
            top_logprobs=20,
        )
        generated_answer = resp2.choices[0].message.content.strip()
        generated_prob = compute_avg_token_prob(resp2.choices[0].logprobs.content)

        is_correct = judge_answer(client, correct_answer, generated_answer)

        if is_correct:
            correct_prob = generated_prob
        else:
            tf_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ]
            correct_prob, _ = teacher_force_correct_prob(
                client, MODEL, tf_messages, correct_answer
            )

    except Exception as e:
        reasoning = f"ERROR: {str(e)}"
        generated_answer = ""
        is_correct = False
        correct_prob = 0.0
        generated_prob = 0.0

    record = {
        "question_id": question_id,
        "domain": row.get("domain", []),
        "difficulty": row.get("difficulty", 0),
        "ground_truth": correct_answer,
        "generated_answer": generated_answer,
        "is_correct": is_correct,
        "generated_prob": generated_prob,
        "correct_prob": correct_prob,
        "reasoning": reasoning,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def process_direct(client, row, idx, output_path):
    question_id = f"omnimath_{idx}"
    problem = row["problem"]
    correct_answer = row["answer"]

    try:
        prompt = DIRECT_TEMPLATE.format(problem=problem)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            logprobs=True,
            top_logprobs=20,
        )
        generated_answer = resp.choices[0].message.content.strip()
        generated_prob = compute_avg_token_prob(resp.choices[0].logprobs.content)

        is_correct = judge_answer(client, correct_answer, generated_answer)

        if is_correct:
            correct_prob = generated_prob
        else:
            tf_messages = [
                {"role": "user", "content": prompt},
            ]
            correct_prob, _ = teacher_force_correct_prob(
                client, MODEL, tf_messages, correct_answer
            )

    except Exception as e:
        generated_answer = ""
        is_correct = False
        correct_prob = 0.0
        generated_prob = 0.0

    record = {
        "question_id": question_id,
        "domain": row.get("domain", []),
        "difficulty": row.get("difficulty", 0),
        "ground_truth": correct_answer,
        "generated_answer": generated_answer,
        "is_correct": is_correct,
        "generated_prob": generated_prob,
        "correct_prob": correct_prob,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def run_experiment(exp_name):
    client = create_client(PROVIDER)
    is_cot = exp_name == "exp1_cot"
    model_tag = make_model_tag(MODEL)
    output_dir = Path("results") / f"{exp_name}_omnimath_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "omnimath.jsonl"

    print("Loading Omni-MATH dataset...")
    dataset = load_dataset("KbsdJames/Omni-MATH", split="test")

    completed_ids = load_completed_ids(output_path)

    pending = []
    for i, row in enumerate(dataset):
        qid = f"omnimath_{i}"
        if qid in completed_ids:
            continue
        pending.append((i, row))

    total = len(dataset)
    done = len(completed_ids)
    print(f"Running '{exp_name}' on Omni-MATH (model: {MODEL}, workers: {MAX_WORKERS})")
    print(f"  Total: {total}, Already done: {done}, Pending: {len(pending)}")

    if not pending:
        print("All questions already processed.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row in pending:
            if is_cot:
                futures.append(executor.submit(process_cot, client, row, idx, output_path))
            else:
                futures.append(executor.submit(process_direct, client, row, idx, output_path))

        for _ in tqdm(as_completed(futures), total=len(futures), desc=exp_name):
            pass

    print(f"Done! Processed: {len(pending)}")


def main():
    parser = argparse.ArgumentParser(description="Run Omni-MATH experiment")
    parser.add_argument(
        "--exp",
        choices=["cot", "direct", "both"],
        default="both",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "litellm", "vllm"],
        default="vllm",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers

    experiments = []
    if args.exp in ("cot", "both"):
        experiments.append("exp1_cot")
    if args.exp in ("direct", "both"):
        experiments.append("exp2_direct")

    for exp_name in experiments:
        run_experiment(exp_name)


if __name__ == "__main__":
    main()
