
import json
import math
import re
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import create_client, make_model_tag

load_dotenv()

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
TOP_LOGPROBS = 200
VOCAB_SIZE = 151936
DATA_PATH = "data/math_test.jsonl"

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
                    completed.add(json.loads(line)["question_id"])
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


def load_paraphrase_results(model_tag):
    path = Path("results") / f"exp3_selfpara_math_t_{model_tag}" / "math_test.jsonl"
    m = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                m[r["question_id"]] = r
    return m


def vllm_tokenize(text):
    import os, requests
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/v1").rstrip("/")
    return requests.post(f"{base}/tokenize", json={"prompt": text, "add_special_tokens": False}, timeout=10).json().get("tokens", [])


def vllm_detokenize(token_ids):
    import os, requests
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/v1").rstrip("/")
    return [requests.post(f"{base}/detokenize", json={"tokens": [tid]}, timeout=10).json().get("prompt", "") for tid in token_ids]


def extract_distribution(logprobs_content):
    if not logprobs_content:
        return {}, 0.0
    dist = {}
    mass = 0.0
    for top_lp in logprobs_content[0].top_logprobs:
        p = math.exp(top_lp.logprob)
        dist[top_lp.token] = p
        mass += p
    return dist, mass


def kl_onehot(correct_token, dist, mass):
    if correct_token in dist and dist[correct_token] > 0:
        return -math.log(dist[correct_token])
    remaining = max(1e-10, 1.0 - mass)
    p_est = remaining / max(1, VOCAB_SIZE - len(dist))
    return -math.log(max(p_est, 1e-30))


def kl_pq(dist_p, mass_p, dist_q, mass_q):
    q_default = max(1e-10, 1.0 - mass_q) / max(1, VOCAB_SIZE - len(dist_q))
    kl = 0.0
    for tok, pv in dist_p.items():
        if pv <= 0:
            continue
        qv = dist_q.get(tok, q_default)
        if qv <= 0:
            qv = 1e-30
        kl += pv * math.log(pv / qv)
    return max(kl, 0.0)


def js_div(dist_p, mass_p, dist_q, mass_q):
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())
    p_def = max(1e-10, 1.0 - mass_p) / max(1, VOCAB_SIZE - len(dist_p))
    q_def = max(1e-10, 1.0 - mass_q) / max(1, VOCAB_SIZE - len(dist_q))
    dist_m = {}
    for t in all_tokens:
        dist_m[t] = 0.5 * (dist_p.get(t, p_def) + dist_q.get(t, q_def))
    mass_m = sum(dist_m.values())
    return 0.5 * kl_pq(dist_p, mass_p, dist_m, mass_m) + 0.5 * kl_pq(dist_q, mass_q, dist_m, mass_m)


def teacher_force(client, messages, correct_answer):
    answer_text = correct_answer.strip()
    if not answer_text:
        return []

    if PROVIDER == "vllm":
        try:
            tids = vllm_tokenize(answer_text)
            tokens = vllm_detokenize(tids)
        except:
            from llm_client import get_tokenizer
            enc = get_tokenizer(MODEL)
            tids = enc.encode(answer_text)
            tokens = [enc.decode([t]) for t in tids]
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
                max_tokens=1, logprobs=True, top_logprobs=TOP_LOGPROBS, **extra,
            )
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                d, m = extract_distribution(resp.choices[0].logprobs.content)
            else:
                d, m = {}, 0.0
        except:
            d, m = {}, 0.0
        results.append((tok, d, m))
        prefix += tok
    return results


def process_item(client, row, idx, output_path, cot_rec, para_rec):
    question_id = f"math_t_{idx}"
    problem = row["problem"]
    correct_answer = extract_boxed_answer(row["solution"])
    if not correct_answer.strip():
        return None

    reasoning = cot_rec.get("reasoning", "")
    para_reasoning = para_rec.get("paraphrased_reasoning", "")
    if not reasoning or "ERROR" in reasoning or not para_reasoning:
        return None

    prompt = COT_TEMPLATE.format(problem=problem)

    # 1. Original CoT
    cot_res = teacher_force(client, [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ], correct_answer)

    # 2. Paraphrased CoT
    para_res = teacher_force(client, [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": para_reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ], correct_answer)

    # 3. Direct
    direct_res = teacher_force(client, [
        {"role": "user", "content": DIRECT_TEMPLATE.format(problem=problem)},
    ], correct_answer)

    if not cot_res or not para_res or not direct_res:
        return None

    nt = min(len(cot_res), len(para_res), len(direct_res))
    kl_oh_cot, kl_oh_para, kl_cd, kl_pd, js_cp = [], [], [], [], []

    for i in range(nt):
        ct, cd, cm = cot_res[i]
        pt, pd, pm = para_res[i]
        dt, dd, dm = direct_res[i]
        kl_oh_cot.append(kl_onehot(ct, cd, cm))
        kl_oh_para.append(kl_onehot(pt, pd, pm))
        kl_cd.append(kl_pq(cd, cm, dd, dm))
        kl_pd.append(kl_pq(pd, pm, dd, dm))
        js_cp.append(js_div(cd, cm, pd, pm))

    avg = lambda l: sum(l)/len(l) if l else 0
    record = {
        "question_id": question_id,
        "type": row.get("type", ""),
        "ground_truth": correct_answer,
        "n_tokens": nt,
        "kl_onehot_cot": avg(kl_oh_cot),
        "kl_onehot_para": avg(kl_oh_para),
        "kl_cot_direct": avg(kl_cd),
        "kl_para_direct": avg(kl_pd),
        "js_cot_para": avg(js_cp),
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
    para_map = load_paraphrase_results(model_tag)
    if not cot_map:
        print(f"No CoT results: exp1_cot_math_t_{model_tag}")
        return
    if not para_map:
        print(f"No paraphrase results: exp3_paraphrase_math_t_{model_tag}")
        return
    print(f"Loaded {len(cot_map)} CoT, {len(para_map)} Paraphrase records")

    output_dir = Path("results") / f"kl_paraphrase_math_t_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kl_para.jsonl"

    data = load_data()
    completed = load_completed_ids(output_path)

    pending = []
    for idx, row in enumerate(data):
        qid = f"math_t_{idx}"
        if qid in completed or qid not in cot_map or qid not in para_map:
            continue
        pending.append((idx, row, cot_map[qid], para_map[qid]))

    print(f"Running KL/JS paraphrase experiment (model: {MODEL})")
    print(f"  Total: {len(data)}, Done: {len(completed)}, Pending: {len(pending)}")

    if not pending:
        print("All done.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_item, client, row, idx, output_path, cot, para)
                   for idx, row, cot, para in pending]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_para"):
            pass

    records = []
    with open(output_path) as f:
        for line in f:
            records.append(json.loads(line))
    if records:
        n = len(records)
        print(f"\n{'='*60}")
        print(f"Results (model: {MODEL}, n={n})")
        print(f"{'='*60}")
        print(f"  KL(one-hot ∥ CoT):      {sum(r['kl_onehot_cot'] for r in records)/n:.4f}")
        print(f"  KL(one-hot ∥ Para):      {sum(r['kl_onehot_para'] for r in records)/n:.4f}")
        print(f"  KL(CoT ∥ Direct):        {sum(r['kl_cot_direct'] for r in records)/n:.4f}")
        print(f"  KL(Para ∥ Direct):       {sum(r['kl_para_direct'] for r in records)/n:.4f}")
        print(f"  JS(CoT ∥ Para):          {sum(r['js_cot_para'] for r in records)/n:.4f}")


if __name__ == "__main__":
    main()
