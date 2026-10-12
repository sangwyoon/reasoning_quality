
import json
import math
import re
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import create_client, make_model_tag
from paraphrase import split_reasoning_into_steps

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
TOP_LOGPROBS = 200
VOCAB_SIZE = 151936
DATA_PATH = "data/math_test.jsonl"
SEED = 42

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


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r'\\boxed\{([^}]+)\}', solution)
    return matches[-1] if matches else ""


def load_data():
    records = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


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


def load_cot_results(model_tag):
    cot_dir = Path("results") / f"exp1_cot_math_t_{model_tag}"
    cot_path = cot_dir / "math_test.jsonl"
    cot_map = {}
    if cot_path.exists():
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r
    return cot_map


def vllm_tokenize(text):
    import os, requests
    base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    base = base_url.rstrip("/v1").rstrip("/")
    resp = requests.post(f"{base}/tokenize", json={"prompt": text, "add_special_tokens": False}, timeout=10)
    return resp.json().get("tokens", [])


def vllm_detokenize(token_ids):
    import os, requests
    base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")
    base = base_url.rstrip("/v1").rstrip("/")
    results = []
    for tid in token_ids:
        resp = requests.post(f"{base}/detokenize", json={"tokens": [tid]}, timeout=10)
        results.append(resp.json().get("prompt", ""))
    return results


def extract_distribution(logprobs_content):
    if not logprobs_content:
        return {}, 0.0
    token_info = logprobs_content[0]
    dist = {}
    prob_mass = 0.0
    for top_lp in token_info.top_logprobs:
        prob = math.exp(top_lp.logprob)
        dist[top_lp.token] = prob
        prob_mass += prob
    return dist, prob_mass


def kl_onehot_vs_dist(correct_token, dist, prob_mass):
    if correct_token in dist:
        p = dist[correct_token]
        if p > 0:
            return -math.log(p)
    remaining_mass = max(1e-10, 1.0 - prob_mass)
    remaining_tokens = max(1, VOCAB_SIZE - len(dist))
    p_est = remaining_mass / remaining_tokens
    return -math.log(max(p_est, 1e-30))


def kl_dist_vs_dist(dist_p, mass_p, dist_q, mass_q):
    remaining_q = max(1e-10, 1.0 - mass_q)
    remaining_tokens_q = max(1, VOCAB_SIZE - len(dist_q))
    q_default = remaining_q / remaining_tokens_q

    kl = 0.0
    for token, p_val in dist_p.items():
        if p_val <= 0:
            continue
        q_val = dist_q.get(token, q_default)
        if q_val <= 0:
            q_val = 1e-30
        kl += p_val * math.log(p_val / q_val)
    return max(kl, 0.0)


def js_divergence(dist_p, mass_p, dist_q, mass_q):
    """Jensen-Shannon divergence: JS(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M=(P+Q)/2"""
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())

    remaining_p = max(1e-10, 1.0 - mass_p)
    remaining_q = max(1e-10, 1.0 - mass_q)
    p_default = remaining_p / max(1, VOCAB_SIZE - len(dist_p))
    q_default = remaining_q / max(1, VOCAB_SIZE - len(dist_q))

    dist_m = {}
    for token in all_tokens:
        p_val = dist_p.get(token, p_default)
        q_val = dist_q.get(token, q_default)
        dist_m[token] = 0.5 * (p_val + q_val)

    mass_m = sum(dist_m.values())

    kl_pm = kl_dist_vs_dist(dist_p, mass_p, dist_m, mass_m)
    kl_qm = kl_dist_vs_dist(dist_q, mass_q, dist_m, mass_m)

    return 0.5 * kl_pm + 0.5 * kl_qm


def teacher_force_distribution(client, messages, correct_answer):
    answer_text = correct_answer.strip()
    if not answer_text:
        return []

    if PROVIDER == "vllm":
        try:
            token_ids = vllm_tokenize(answer_text)
            correct_tokens = vllm_detokenize(token_ids)
        except Exception:
            from llm_client import get_tokenizer
            enc = get_tokenizer(MODEL)
            token_ids = enc.encode(answer_text)
            correct_tokens = [enc.decode([tid]) for tid in token_ids]
    else:
        from llm_client import get_tokenizer
        enc = get_tokenizer(MODEL)
        token_ids = enc.encode(answer_text)
        correct_tokens = [enc.decode([tid]) for tid in token_ids]

    if not correct_tokens:
        return []

    results = []
    forced_prefix = ""

    for target_token in correct_tokens:
        msgs = list(messages)
        extra = {}
        if forced_prefix:
            msgs.append({"role": "assistant", "content": forced_prefix})
            if PROVIDER == "vllm":
                extra["extra_body"] = {
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                }

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
                **extra,
            )
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                dist, mass = extract_distribution(resp.choices[0].logprobs.content)
            else:
                dist, mass = {}, 0.0
        except Exception:
            dist, mass = {}, 0.0

        results.append((target_token, dist, mass))
        forced_prefix += target_token

    return results


def duplicate_step(reasoning, seed_str):
    pattern, steps = split_reasoning_into_steps(reasoning)

    if len(steps) < 2:
        return reasoning, -1, pattern, len(steps)

    rng = random.Random(seed_str)
    dup_idx = rng.randint(0, len(steps) - 1)

    new_steps = list(steps)
    new_steps.insert(dup_idx + 1, steps[dup_idx])

    return "\n\n".join(new_steps), dup_idx, pattern, len(steps)


def process_item(client, row, idx, output_path, cot_record):
    question_id = f"math_t_{idx}"
    problem = row["problem"]
    correct_answer = extract_boxed_answer(row["solution"])

    if not correct_answer.strip():
        return None

    reasoning = cot_record.get("reasoning", "")
    if not reasoning or "ERROR" in reasoning:
        return None

    prompt_cot = COT_TEMPLATE.format(problem=problem)

    dup_reasoning, dup_idx, split_pattern, n_steps = duplicate_step(
        reasoning, f"{SEED}_{question_id}"
    )

    # --- 1. Original CoT ---
    cot_messages = [
        {"role": "user", "content": prompt_cot},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ]
    cot_results = teacher_force_distribution(client, cot_messages, correct_answer)

    # --- 2. Duplicated CoT ---
    dup_messages = [
        {"role": "user", "content": prompt_cot},
        {"role": "assistant", "content": dup_reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ]
    dup_results = teacher_force_distribution(client, dup_messages, correct_answer)

    # --- 3. Direct ---
    direct_prompt = DIRECT_TEMPLATE.format(problem=problem)
    direct_messages = [{"role": "user", "content": direct_prompt}]
    direct_results = teacher_force_distribution(client, direct_messages, correct_answer)

    if not cot_results or not dup_results or not direct_results:
        return None

    n_tokens = min(len(cot_results), len(dup_results), len(direct_results))

    kl_oh_cot_list = []
    kl_oh_dup_list = []
    kl_cot_direct_list = []
    kl_dup_direct_list = []
    js_cot_dup_list = []

    for i in range(n_tokens):
        ct_tok, ct_dist, ct_mass = cot_results[i]
        dp_tok, dp_dist, dp_mass = dup_results[i]
        dt_tok, dt_dist, dt_mass = direct_results[i]

        kl_oh_cot_list.append(kl_onehot_vs_dist(ct_tok, ct_dist, ct_mass))
        kl_oh_dup_list.append(kl_onehot_vs_dist(dp_tok, dp_dist, dp_mass))
        kl_cot_direct_list.append(kl_dist_vs_dist(ct_dist, ct_mass, dt_dist, dt_mass))
        kl_dup_direct_list.append(kl_dist_vs_dist(dp_dist, dp_mass, dt_dist, dt_mass))
        js_cot_dup_list.append(js_divergence(ct_dist, ct_mass, dp_dist, dp_mass))

    avg = lambda lst: sum(lst) / len(lst) if lst else 0

    record = {
        "question_id": question_id,
        "type": row.get("type", ""),
        "level": row.get("level", ""),
        "ground_truth": correct_answer,
        "n_tokens": n_tokens,
        "n_steps": n_steps,
        "duplicated_step_idx": dup_idx,
        "split_pattern": split_pattern,
        "kl_onehot_cot": avg(kl_oh_cot_list),
        "kl_onehot_dup": avg(kl_oh_dup_list),
        "kl_cot_direct": avg(kl_cot_direct_list),
        "kl_dup_direct": avg(kl_dup_direct_list),
        "js_cot_dup": avg(js_cot_dup_list),
        "kl_onehot_cot_per_token": kl_oh_cot_list,
        "kl_onehot_dup_per_token": kl_oh_dup_list,
        "js_cot_dup_per_token": js_cot_dup_list,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def main():
    parser = argparse.ArgumentParser(description="KL/JS experiment: CoT vs Duplicated-Step CoT")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="vllm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--top_logprobs", type=int, default=200)
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, TOP_LOGPROBS
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    TOP_LOGPROBS = args.top_logprobs

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    cot_map = load_cot_results(model_tag)
    if not cot_map:
        print(f"No CoT results: exp1_cot_math_t_{model_tag}")
        return
    print(f"Loaded {len(cot_map)} CoT records")

    output_dir = Path("results") / f"kl_dupstep_math_t_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kl_dupstep.jsonl"

    data = load_data()
    completed_ids = load_completed_ids(output_path)

    pending = []
    for idx, row in enumerate(data):
        qid = f"math_t_{idx}"
        if qid in completed_ids:
            continue
        if qid not in cot_map:
            continue
        pending.append((idx, row, cot_map[qid]))

    print(f"Running KL/JS dup-step experiment (model: {MODEL})")
    print(f"  Total: {len(data)}, Done: {len(completed_ids)}, Pending: {len(pending)}")

    if not pending:
        print("All done.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row, cot_rec in pending:
            futures.append(executor.submit(process_item, client, row, idx, output_path, cot_rec))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_dup"):
            pass

    records = []
    with open(output_path) as f:
        for line in f:
            records.append(json.loads(line))

    if records:
        n = len(records)
        print(f"\n{'='*70}")
        print(f"Results (model: {MODEL}, n={n})")
        print(f"{'='*70}")
        print(f"  KL(one-hot ∥ CoT):          {sum(r['kl_onehot_cot'] for r in records)/n:.4f}")
        print(f"  KL(one-hot ∥ Dup):           {sum(r['kl_onehot_dup'] for r in records)/n:.4f}")
        print(f"  KL(CoT ∥ Direct):            {sum(r['kl_cot_direct'] for r in records)/n:.4f}")
        print(f"  KL(Dup ∥ Direct):            {sum(r['kl_dup_direct'] for r in records)/n:.4f}")
        print(f"  JS(CoT ∥ Dup):               {sum(r['js_cot_dup'] for r in records)/n:.4f}")


if __name__ == "__main__":
    main()
