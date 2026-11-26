
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
from llm_client import create_client, make_model_tag, get_tokenizer, resolve_tiktoken_vocab_size
import measures
import vllm_server

load_dotenv()

PROVIDER = "openai"
TOP_LOGPROBS = 200
JUDGE_MODEL = measures.DEFAULT_JUDGE_MODEL
JUDGE = measures.openai_judge(JUDGE_MODEL)
VOCAB_SIZE = None
MAX_WORKERS = 64
PARAPHRASE_MODEL = "gpt-4.1-nano"

COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Think step by step, analyzing the problem carefully.

RULES:
- You MAY reason about the problem and show your work.
- **STOP** before stating your final answer.
- Do NOT write phrases like "therefore the answer is" or any equivalent conclusion.
- End your response as if handing off your reasoning to someone else to state the final answer.

Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~"""

ANSWER_FOLLOWUP = 'Now state ONLY the final answer\n(a number, expression, or short mathematical notation). Nothing else.\n\nAnswer:'

write_lock = threading.Lock()
paraphraser = None


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

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


def get_cot_source_dir(model: str) -> Path:
    model_tag = make_model_tag(model)
    candidates = [
        Path("results") / f"exp1_cot_omnimath_{model_tag}",
        Path("results") / f"exp1_cot_omnimath_{model}",
    ]
    for d in candidates:
        if d.exists():
            return d
    raise FileNotFoundError(
        f"OmniMath CoT results directory not found: {[str(c) for c in candidates]}"
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

def process_paraphrase(client, model, row, idx, output_path,
                       cot_record, num_paraphrase, seed, paraphrase_mode,
                       mask_ratio=0.0, protect_latex=True):
    global paraphraser
    question_id = f"omnimath_{idx}"

    try:
        problem = row["problem"]
        correct_answer = row["answer"]
        prompt = COT_TEMPLATE.format(problem=problem)
        if cot_record.get("error") is not None:
            raise ValueError(f"CoT record has an error: {cot_record['error']}")

        original_reasoning = cot_record["reasoning"]
        original_correct_prob = cot_record.get("correct_prob")
        original_is_correct = cot_record.get("is_correct", False)
        original_generated_answer = cot_record.get("generated_answer", "")

        # 1. Reasoning split
        split_pattern, steps = split_reasoning_into_steps(original_reasoning)

        if paraphrase_mode == "mask":
            rng = random.Random(f"{seed}_{question_id}_mask_p0")
            masker = TokenMasker(mask_ratio=mask_ratio, rng=rng)

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

        error = None
        generated_answer = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": paraphrased_reasoning},
                    {"role": "user", "content": ANSWER_FOLLOWUP},
                ],
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
                    {"role": "assistant", "content": paraphrased_reasoning},
                    {"role": "user", "content": ANSWER_FOLLOWUP},
                ]
                correct_prob, _ = teacher_force_correct_prob(
                    client, model, tf_messages, correct_answer
                )

        except Exception as e:
            print(f"[ERROR] {question_id}: {e}", flush=True)
            error = str(e)
            is_correct = None
            correct_prob = None
            generated_prob = None

        record = {
            "question_id": question_id,
            "domain": row.get("domain", []),
            "difficulty": row.get("difficulty", 0),
            "ground_truth": correct_answer,
            "paraphrase_mode": paraphrase_mode,
            "mask_ratio": mask_ratio if paraphrase_mode == "mask" else None,
            "num_paraphrase": num_paraphrase if paraphrase_mode != "mask" else None,
            "original_generated_answer": original_generated_answer,
            "original_is_correct": original_is_correct,
            "original_correct_prob": original_correct_prob,
            "original_reasoning": original_reasoning,
            "split_pattern": split_pattern,
            "num_steps": len(steps),
            "paraphrased_step_indices": selected_indices,
            "transformations_applied": transformations,
            "paraphrased_reasoning": paraphrased_reasoning,
            "generated_answer": generated_answer,
            "is_correct": is_correct,
            "generated_prob": generated_prob,
            "correct_prob": correct_prob,
            "prob_delta": (None if correct_prob is None or original_correct_prob is None
                           else correct_prob - original_correct_prob),
            "error": error,
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] {question_id}: {e}", flush=True)
        traceback.print_exc()
        record = {
            "question_id": question_id,
            "domain": row.get("domain", []),
            "difficulty": row.get("difficulty", 0),
            "ground_truth": row.get("answer", ""),
            "paraphrase_mode": paraphrase_mode,
            "mask_ratio": mask_ratio if paraphrase_mode == "mask" else None,
            "num_paraphrase": num_paraphrase if paraphrase_mode != "mask" else None,
            "original_generated_answer": cot_record.get("generated_answer", ""),
            "original_is_correct": cot_record.get("is_correct", False),
            "original_correct_prob": cot_record.get("correct_prob"),
            "original_reasoning": cot_record.get("reasoning", ""),
            "split_pattern": "error",
            "num_steps": 0,
            "paraphrased_step_indices": [],
            "transformations_applied": {"error": str(e)},
            "paraphrased_reasoning": "",
            "generated_answer": "",
            "is_correct": None,
            "generated_prob": None,
            "correct_prob": None,
            "prob_delta": None,
            "error": str(e),
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
    print(f"Loading OmniMath CoT results from: {cot_dir}")
    cot_map = load_cot_results(cot_dir)
    print(f"  Loaded {len(cot_map)} CoT records")

    mtag = make_model_tag(model)
    if paraphrase_mode == "mask":
        output_dir = Path("results") / f"exp3_mask_k{int(mask_ratio * 100)}{latex_tag}_omnimath_{mtag}"
    elif paraphrase_mode == "llm":
        output_dir = Path("results") / f"exp3_paraphrase_llm_n{num_paraphrase}_omnimath_{mtag}"
    else:
        output_dir = Path("results") / f"exp3_paraphrase_n{num_paraphrase}_omnimath_{mtag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "omnimath.jsonl"
    measures.write_run_config(
        output_path, model=model, dataset="omnimath", phase=mode_label, seed=seed,
        mask_ratio=mask_ratio if paraphrase_mode == "mask" else None, n_probes=1,
        top_logprobs=TOP_LOGPROBS, num_paraphrase=num_paraphrase, judge_model=JUDGE_MODEL, vocab_size=VOCAB_SIZE)

    print("Loading Omni-MATH dataset...")
    dataset = load_dataset("KbsdJames/Omni-MATH", split="test")

    completed_ids = load_completed_ids(output_path)

    pending = []
    no_cot = 0
    for i, row in enumerate(dataset):
        qid = f"omnimath_{i}"
        if qid in completed_ids:
            continue
        if qid not in cot_map:
            no_cot += 1
            continue
        pending.append((i, row, cot_map[qid]))

    total = len(dataset)
    done = len(completed_ids)
    print(f"Running paraphrase experiment on Omni-MATH "
          f"(model={model}, mode={mode_label}, num_paraphrase={num_paraphrase}, "
          f"mask_ratio={mask_ratio}, seed={seed}, workers={MAX_WORKERS})")
    print(f"  Total: {total}, Already done: {done}, Pending: {len(pending)}, No CoT: {no_cot}")

    if not pending:
        print("All questions already processed.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row, cot_rec in pending:
            futures.append(executor.submit(
                process_paraphrase,
                client, model, row, idx, output_path,
                cot_rec, num_paraphrase, seed, paraphrase_mode,
                mask_ratio, protect_latex,
            ))

        for _ in tqdm(as_completed(futures), total=len(futures), desc=mode_label):
            pass

    print(f"\nDone! Processed: {len(pending)}")
    print(f"Results saved to: {output_path}")


def main():
    global TOP_LOGPROBS, VOCAB_SIZE, JUDGE, JUDGE_MODEL
    parser = argparse.ArgumentParser(description="Run paraphrase experiment on OmniMath CoT results")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name (default: gpt-4o-mini)",
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
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
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
        choices=["openai", "litellm", "vllm"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument("--top_logprobs", type=int, default=TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    parser.add_argument("--vocab_size", type=int, default=None)
    measures.add_judge_argument(parser)
    args = parser.parse_args()
    TOP_LOGPROBS = args.top_logprobs

    global PROVIDER
    PROVIDER = args.provider

    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers

    VOCAB_SIZE = resolve_tiktoken_vocab_size(PROVIDER, args.model, args.vocab_size)
    JUDGE_MODEL = args.judge_model
    JUDGE = measures.openai_judge(JUDGE_MODEL)

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
