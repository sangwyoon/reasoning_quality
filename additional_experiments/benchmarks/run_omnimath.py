import json
import math
import sys
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag, get_tokenizer, resolve_tiktoken_vocab_size
import measures
import vllm_server

load_dotenv()

MODEL = "gpt-4o-mini"
PROVIDER = "openai"
TOP_LOGPROBS = 200
JUDGE_MODEL = measures.DEFAULT_JUDGE_MODEL
JUDGE = measures.openai_judge(JUDGE_MODEL)
VOCAB_SIZE = None
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

write_lock = threading.Lock()


def compute_avg_token_prob(logprobs_content):
    if not logprobs_content:
        raise ValueError("empty logprobs content")
    probs = [math.exp(tok.logprob) for tok in logprobs_content]
    return sum(probs) / len(probs)


def teacher_force_correct_prob(client, model, messages_before_answer, correct_answer):
    if not correct_answer or not correct_answer.strip():
        raise ValueError("empty gold answer; nothing to teacher-force")

    if PROVIDER == "vllm":
        correct_tokens = vllm_server.detokenize(model, vllm_server.tokenize(model, correct_answer.strip()))
    else:
        enc = get_tokenizer(model)
        correct_tokens = [enc.decode([tid]) for tid in enc.encode(correct_answer.strip())]

    if not correct_tokens:
        raise ValueError("no answer tokens to teacher-force")

    token_probs = []
    forced_prefix = ""

    for target_token in correct_tokens:
        msgs = list(messages_before_answer)
        extra = {}
        if forced_prefix:
            msgs.append({"role": "assistant", "content": forced_prefix})
            if PROVIDER == "vllm":
                extra["extra_body"] = {"continue_final_message": True, "add_generation_prompt": False}

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                **extra,
            )

            if not (resp.choices[0].logprobs and resp.choices[0].logprobs.content):
                raise ValueError("empty logprobs content")
            dist = {}
            mass = 0.0
            for top_lp in resp.choices[0].logprobs.content[0].top_logprobs:
                p = math.exp(top_lp.logprob)
                dist[top_lp.token] = p
                mass += p
            token_probs.append(measures.completed_token_prob(dist, mass, target_token, VOCAB_SIZE))

        except Exception as e:
            raise RuntimeError(
                f"teacher forcing failed at token {len(token_probs)}: {e}") from e

        forced_prefix += target_token

    avg_prob = sum(token_probs) / len(token_probs)
    return avg_prob, token_probs


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


def process_cot(client, row, idx, output_path):
    question_id = f"omnimath_{idx}"
    problem = row["problem"]
    correct_answer = row["answer"]

    error = None
    reasoning = ""
    generated_answer = ""
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
            top_logprobs=TOP_LOGPROBS,
        )
        generated_answer = resp2.choices[0].message.content.strip()
        generated_prob = compute_avg_token_prob(resp2.choices[0].logprobs.content)

        is_correct = measures.judge_correct(str(correct_answer), generated_answer, JUDGE)

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
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        is_correct = None
        correct_prob = None
        generated_prob = None

    record = {
        "question_id": question_id,
        "domain": row.get("domain", []),
        "difficulty": row.get("difficulty", 0),
        "ground_truth": correct_answer,
        "generated_answer": generated_answer,
        "is_correct": is_correct,
        "generated_prob": generated_prob,
        "correct_prob": correct_prob,
        "error": error,
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

    error = None
    generated_answer = ""
    try:
        prompt = DIRECT_TEMPLATE.format(problem=problem)

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
        )
        generated_answer = resp.choices[0].message.content.strip()
        generated_prob = compute_avg_token_prob(resp.choices[0].logprobs.content)

        is_correct = measures.judge_correct(str(correct_answer), generated_answer, JUDGE)

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
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        is_correct = None
        correct_prob = None
        generated_prob = None

    record = {
        "question_id": question_id,
        "domain": row.get("domain", []),
        "difficulty": row.get("difficulty", 0),
        "ground_truth": correct_answer,
        "generated_answer": generated_answer,
        "is_correct": is_correct,
        "generated_prob": generated_prob,
        "correct_prob": correct_prob,
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
    output_dir = Path("results") / f"{exp_name}_omnimath_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "omnimath.jsonl"
    measures.write_run_config(
        output_path, model=MODEL, dataset="omnimath", phase=exp_name, seed=None,
        mask_ratio=None, n_probes=None, top_logprobs=TOP_LOGPROBS, judge_model=JUDGE_MODEL, vocab_size=VOCAB_SIZE)

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
    global TOP_LOGPROBS, VOCAB_SIZE, JUDGE, JUDGE_MODEL
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
    parser.add_argument("--top_logprobs", type=int, default=TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    parser.add_argument("--vocab_size", type=int, default=None)
    measures.add_judge_argument(parser)
    args = parser.parse_args()
    TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    VOCAB_SIZE = resolve_tiktoken_vocab_size(PROVIDER, MODEL, args.vocab_size)
    JUDGE_MODEL = args.judge_model
    JUDGE = measures.openai_judge(JUDGE_MODEL)

    experiments = []
    if args.exp in ("cot", "both"):
        experiments.append("exp1_cot")
    if args.exp in ("direct", "both"):
        experiments.append("exp2_direct")

    for exp_name in experiments:
        run_experiment(exp_name)


if __name__ == "__main__":
    main()
