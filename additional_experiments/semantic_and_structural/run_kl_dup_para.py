
import sys
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
import measures
import vllm_server
from paraphrase import split_reasoning_into_steps

load_dotenv()

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
TOP_LOGPROBS = 200
VOCAB_SIZE = None
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

PARAPHRASE_SYSTEM = (
    "Paraphrase the following reasoning text. "
    "Preserve the logical structure and all mathematical content exactly. "
    "Change the wording and sentence structure while keeping the same meaning. "
    "Keep all LaTeX, formulas, and technical terms intact. "
    "Do NOT add any new reasoning or conclusions."
)

write_lock = threading.Lock()


def extract_boxed_answer(solution: str) -> str:
    matches = re.findall(r'\\boxed\{([^}]+)\}', solution)
    return matches[-1] if matches else ""


def load_data():
    with open(DATA_PATH) as f:
        return [json.loads(l) for l in f]


def load_completed_ids(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("error") is None:
                        completed.add(record["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def load_cot_results(model_tag):
    path = Path("results") / f"exp1_cot_math_t_{model_tag}" / "math_test.jsonl"
    m = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                m[r["question_id"]] = r
    return m


def extract_distribution(logprobs_content):
    if not logprobs_content:
        raise ValueError("empty logprobs content")
    dist, mass = {}, 0.0
    for top_lp in logprobs_content[0].top_logprobs:
        p = math.exp(top_lp.logprob)
        dist[top_lp.token] = p
        mass += p
    return dist, mass


def kl_onehot(correct_token, dist, mass):
    if correct_token in dist and dist[correct_token] > 0:
        return -math.log(dist[correct_token])
    remaining = max(1e-10, 1.0 - mass)
    return -math.log(max(remaining / max(1, VOCAB_SIZE - len(dist)), 1e-30))


def kl_pq(dist_p, mass_p, dist_q, mass_q):
    q_def = max(1e-10, 1.0 - mass_q) / max(1, VOCAB_SIZE - len(dist_q))
    kl = 0.0
    for tok, pv in dist_p.items():
        if pv <= 0:
            continue
        qv = dist_q.get(tok, q_def)
        kl += pv * math.log(pv / max(qv, 1e-30))
    return max(kl, 0.0)


def js_div(dist_p, mass_p, dist_q, mass_q):
    """Full JS over the vocabulary, including the residual-vocabulary term."""
    return measures.js_open_ended(dist_p, mass_p, dist_q, mass_q, VOCAB_SIZE)


def teacher_force(client, messages, correct_answer):
    answer_text = correct_answer.strip()
    if not answer_text:
        return []
    if PROVIDER == "vllm":
        tokens = vllm_server.detokenize(MODEL, vllm_server.tokenize(MODEL, answer_text))
    else:
        from llm_client import get_tokenizer
        enc = get_tokenizer(MODEL)
        tids = enc.encode(answer_text)
        tokens = [enc.decode([t]) for t in tids]
    if not tokens:
        return []

    results = []
    prefix = ""
    for tok in tokens:
        msgs = list(messages)
        extra = {}
        if prefix:
            msgs.append({"role": "assistant", "content": prefix})
            if PROVIDER == "vllm":
                extra["extra_body"] = {"continue_final_message": True, "add_generation_prompt": False}
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0,
                max_tokens=1, logprobs=True, top_logprobs=TOP_LOGPROBS, **extra)
            if not (resp.choices[0].logprobs and resp.choices[0].logprobs.content):
                raise ValueError("empty logprobs content")
            d, m = extract_distribution(resp.choices[0].logprobs.content)
        except Exception as e:
            raise RuntimeError(f"teacher forcing failed at token {len(results)}: {e}") from e
        results.append((tok, d, m))
        prefix += tok
    return results


def duplicate_step(reasoning, seed_str):
    pattern, steps = split_reasoning_into_steps(reasoning)
    if len(steps) < 2:
        return reasoning, -1, len(steps)
    rng = random.Random(seed_str)
    idx = rng.randint(0, len(steps) - 1)
    new_steps = list(steps)
    new_steps.insert(idx + 1, steps[idx])
    return "\n\n".join(new_steps), idx, len(steps)


def self_paraphrase(client, text):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PARAPHRASE_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=2048,
    )
    paraphrased = (resp.choices[0].message.content or "").strip()
    if not paraphrased:
        raise ValueError("paraphraser returned empty output")
    return paraphrased


ITEM_METRIC_KEYS = ('n_tokens', 'n_steps', 'dup_step_idx', 'kl_onehot_dup', 'kl_onehot_duppara', 'kl_dup_direct', 'kl_duppara_direct', 'js_dup_duppara', 'tail_mass',)


def process_item(client, row, idx, output_path, cot_rec):
    """Compute and store one item; a failure is reported and stored as an ``error`` record."""
    try:
        return _compute_item(client, row, idx, output_path, cot_rec)
    except Exception as e:
        question_id = f"math_t_{idx}"
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        record = {"question_id": question_id, "type": row.get("type", ""),
                  "ground_truth": extract_boxed_answer(row["solution"]), "error": str(e)}
        record.update({k: None for k in ITEM_METRIC_KEYS})
        with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def _compute_item(client, row, idx, output_path, cot_rec):
    question_id = f"math_t_{idx}"
    problem = row["problem"]
    correct_answer = extract_boxed_answer(row["solution"])
    if not correct_answer.strip():
        raise ValueError("empty gold answer")

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    prompt = COT_TEMPLATE.format(problem=problem)

    # 1. Duplicate step
    dup_reasoning, dup_idx, n_steps = duplicate_step(reasoning, f"{SEED}_{question_id}")

    # 2. Paraphrase the duplicated reasoning
    dup_para_reasoning = self_paraphrase(client, dup_reasoning)

    # 3. Teacher forcing: Dup
    dup_res = teacher_force(client, [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": dup_reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ], correct_answer)

    # 4. Teacher forcing: Dup+Para
    duppara_res = teacher_force(client, [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": dup_para_reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ], correct_answer)

    # 5. Teacher forcing: Direct
    direct_res = teacher_force(client, [
        {"role": "user", "content": DIRECT_TEMPLATE.format(problem=problem)},
    ], correct_answer)

    if not dup_res or not duppara_res or not direct_res:
        raise ValueError("no teacher-forcing tokens")

    nt = min(len(dup_res), len(duppara_res), len(direct_res))
    kl_oh_dup, kl_oh_dp, kl_dup_dir, kl_dp_dir, js_dup_dp = [], [], [], [], []

    for i in range(nt):
        _, dd, dm = dup_res[i]
        _, dpd, dpm = duppara_res[i]
        _, dtd, dtm = direct_res[i]
        tok = dup_res[i][0]

        kl_oh_dup.append(kl_onehot(tok, dd, dm))
        kl_oh_dp.append(kl_onehot(tok, dpd, dpm))
        kl_dup_dir.append(kl_pq(dd, dm, dtd, dtm))
        kl_dp_dir.append(kl_pq(dpd, dpm, dtd, dtm))
        js_dup_dp.append(js_div(dd, dm, dpd, dpm))

    avg = lambda l: sum(l) / len(l) if l else 0
    record = {
        "question_id": question_id,
        "type": row.get("type", ""),
        "ground_truth": correct_answer,
        "n_tokens": nt,
        "n_steps": n_steps,
        "dup_step_idx": dup_idx,
        "kl_onehot_dup": avg(kl_oh_dup),
        "kl_onehot_duppara": avg(kl_oh_dp),
        "kl_dup_direct": avg(kl_dup_dir),
        "kl_duppara_direct": avg(kl_dp_dir),
        "js_dup_duppara": avg(js_dup_dp),
        "tail_mass": {"dup": measures.mean_tail_mass(dup_res), "duppara": measures.mean_tail_mass(duppara_res), "direct": measures.mean_tail_mass(direct_res)},
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="vllm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--top_logprobs", type=int, default=200)
    parser.add_argument("--vocab_size", type=int, default=None)
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, TOP_LOGPROBS, VOCAB_SIZE
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    TOP_LOGPROBS = args.top_logprobs
    VOCAB_SIZE = measures.resolve_vocab_size(MODEL, args.vocab_size)

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    cot_map = load_cot_results(model_tag)
    if not cot_map:
        print(f"No CoT results: exp1_cot_math_t_{model_tag}")
        return
    print(f"Loaded {len(cot_map)} CoT records")

    output_dir = Path("results") / f"kl_duppara_math_t_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kl_duppara.jsonl"
    measures.write_run_config(
        output_path, model=MODEL, dataset="math_test", phase="kl_duppara", seed=SEED,
        mask_ratio=None, n_probes=1, top_logprobs=TOP_LOGPROBS, vocab_size=VOCAB_SIZE)

    data = load_data()
    completed = load_completed_ids(output_path)

    pending = []
    for idx, row in enumerate(data):
        qid = f"math_t_{idx}"
        if qid in completed or qid not in cot_map:
            continue
        pending.append((idx, row, cot_map[qid]))

    print(f"Running KL/JS dup+para experiment (model: {MODEL})")
    print(f"  Total: {len(data)}, Done: {len(completed)}, Pending: {len(pending)}")

    if not pending:
        print("All done.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_item, client, row, idx, output_path, cot)
                   for idx, row, cot in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_dp"):
            pass

    records = []
    with open(output_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("error") is None:
                records.append(r)
    if records:
        n = len(records)
        print(f"\n{'='*60}")
        print(f"Results (model: {MODEL}, n={n})")
        print(f"{'='*60}")
        print(f"  KL(one-hot ∥ Dup):         {sum(r['kl_onehot_dup'] for r in records)/n:.4f}")
        print(f"  KL(one-hot ∥ Dup+Para):    {sum(r['kl_onehot_duppara'] for r in records)/n:.4f}")
        print(f"  KL(Dup ∥ Direct):          {sum(r['kl_dup_direct'] for r in records)/n:.4f}")
        print(f"  KL(Dup+Para ∥ Direct):     {sum(r['kl_duppara_direct'] for r in records)/n:.4f}")
        print(f"  JS(Dup ∥ Dup+Para):        {sum(r['js_dup_duppara'] for r in records)/n:.4f}")


if __name__ == "__main__":
    main()
