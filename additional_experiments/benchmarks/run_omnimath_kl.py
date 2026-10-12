
import json
import math
import re
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag, get_tokenizer
from paraphrase import split_reasoning_into_steps, TokenMasker

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
JUDGE_MODEL = "gpt-4.1-nano"
MAX_WORKERS = 8
TOP_LOGPROBS = 200
VOCAB_SIZE = 151936
SEED = 42
MASK_RATIO = 0.1
OUTDIR = "ablation_0524"


def detect_vocab_size(model: str) -> int:
    m = model.lower()
    if "gemma-3" in m:
        return 262144
    if "gemma-2" in m:
        return 256000
    if "llama-3" in m or "llama3" in m:
        return 128256
    return 151936

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
    return sum(math.exp(t.logprob) for t in logprobs_content) / len(logprobs_content)


def judge_answer(correct_answer, given_answer):
    if not given_answer or not given_answer.strip():
        return False
    from openai import OpenAI
    try:
        resp = OpenAI().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(
                correct_answer=correct_answer, given_answer=given_answer)}],
            temperature=0, max_tokens=5)
        return resp.choices[0].message.content.strip().lower().startswith("yes")
    except:
        return False


def vllm_tokenize(text):
    import os, requests
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/v1").rstrip("/")
    return requests.post(f"{base}/tokenize", json={"model": MODEL, "prompt": text, "add_special_tokens": False}, timeout=10).json().get("tokens", [])


def vllm_detokenize(token_ids):
    import os, requests
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/v1").rstrip("/")
    return [requests.post(f"{base}/detokenize", json={"model": MODEL, "tokens": [tid]}, timeout=10).json().get("prompt", "") for tid in token_ids]


def extract_distribution(logprobs_content):
    if not logprobs_content:
        return {}, 0.0
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


def js_div(dist_p, mass_p, dist_q, mass_q):
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())
    p_def = max(1e-10, 1.0 - mass_p) / max(1, VOCAB_SIZE - len(dist_p))
    q_def = max(1e-10, 1.0 - mass_q) / max(1, VOCAB_SIZE - len(dist_q))
    dist_m = {t: 0.5 * (dist_p.get(t, p_def) + dist_q.get(t, q_def)) for t in all_tokens}
    mass_m = sum(dist_m.values())
    m_def = max(1e-10, 1.0 - mass_m) / max(1, VOCAB_SIZE - len(dist_m))

    kl_pm, kl_qm = 0.0, 0.0
    for t in all_tokens:
        pv = dist_p.get(t, p_def)
        qv = dist_q.get(t, q_def)
        mv = dist_m.get(t, m_def)
        if pv > 0 and mv > 0:
            kl_pm += pv * math.log(pv / mv)
        if qv > 0 and mv > 0:
            kl_qm += qv * math.log(qv / mv)
    return max(0.5 * kl_pm + 0.5 * kl_qm, 0.0)


def teacher_force(client, messages, correct_answer):
    answer_text = str(correct_answer).strip()
    if not answer_text:
        return []
    if PROVIDER == "vllm":
        try:
            tids = vllm_tokenize(answer_text)
            tokens = vllm_detokenize(tids)
        except:
            enc = get_tokenizer(MODEL)
            tids = enc.encode(answer_text)
            tokens = [enc.decode([t]) for t in tids]
    else:
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
            if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                d, m = extract_distribution(resp.choices[0].logprobs.content)
            else:
                d, m = {}, 0.0
        except:
            d, m = {}, 0.0
        results.append((tok, d, m))
        prefix += tok
    return results


def apply_mask10(text, seed_str):
    rng = random.Random(seed_str)
    masker = TokenMasker(mask_ratio=MASK_RATIO, rng=rng, protect_latex=False)
    masked, _, _ = masker.mask(text)
    return masked


def duplicate_step(reasoning, seed_str):
    _, steps = split_reasoning_into_steps(reasoning)
    if len(steps) < 2:
        return reasoning, -1
    rng = random.Random(seed_str)
    idx = rng.randint(0, len(steps) - 1)
    new_steps = list(steps)
    new_steps.insert(idx + 1, steps[idx])
    return "\n\n".join(new_steps), idx


def mask_one_step(reasoning, seed_str):
    _, steps = split_reasoning_into_steps(reasoning)
    if len(steps) < 2:
        return reasoning, -1
    rng = random.Random(seed_str)
    idx = rng.randint(0, len(steps) - 1)
    masker = TokenMasker(mask_ratio=1.0, rng=rng, protect_latex=False)
    masked_step, _, _ = masker.mask(steps[idx])
    new_steps = list(steps)
    new_steps[idx] = masked_step
    return "\n\n".join(new_steps), idx


def load_completed_ids(output_path):
    completed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    completed.add(json.loads(line)["question_id"])
                except:
                    pass
    return completed


def process_cot(client, row, idx, output_path):
    question_id = f"omni_{idx}"
    problem = row["problem"]
    correct_answer = str(row["answer"]).strip()

    try:
        prompt = COT_TEMPLATE.format(problem=problem)
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=2048)
        reasoning = resp1.choices[0].message.content

        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], temperature=0, max_tokens=256, logprobs=True, top_logprobs=5)
        generated_answer = resp2.choices[0].message.content.strip() if resp2.choices[0].message.content else ""
        generated_prob = compute_avg_token_prob(resp2.choices[0].logprobs.content) if resp2.choices[0].logprobs else 0.0

        tf_results = teacher_force(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ], correct_answer)

        if tf_results:
            correct_prob = sum(d.get(t, 0) for t, d, m in tf_results) / len(tf_results)
        else:
            correct_prob = 0.0

        is_correct = judge_answer(correct_answer, generated_answer)
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
    question_id = f"omni_{idx}"
    problem = row["problem"]
    correct_answer = str(row["answer"]).strip()

    try:
        prompt = DIRECT_TEMPLATE.format(problem=problem)
        tf_results = teacher_force(client, [
            {"role": "user", "content": prompt},
        ], correct_answer)

        if not tf_results or all(not d for _, d, _ in tf_results):
            return None
        kl_direct = sum(kl_onehot(t, d, m) for t, d, m in tf_results) / len(tf_results)
    except Exception:
        return None

    record = {
        "question_id": question_id,
        "kl_oh_direct": kl_direct,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def process_kl(client, row, idx, output_path, cot_rec):
    question_id = f"omni_{idx}"
    problem = row["problem"]
    correct_answer = str(row["answer"]).strip()
    if not correct_answer:
        return None

    reasoning = cot_rec.get("reasoning", "")
    if not reasoning or "ERROR" in reasoning:
        return None

    prompt = COT_TEMPLATE.format(problem=problem)
    seed_base = f"{SEED}_{question_id}"

    cot_orig = reasoning
    dup_orig, _ = duplicate_step(reasoning, seed_base)
    sm_orig, _ = mask_one_step(reasoning, seed_base)

    cot_m = apply_mask10(cot_orig, f"m10_{seed_base}_cot")
    dup_m = apply_mask10(dup_orig, f"m10_{seed_base}_dup")
    sm_m = apply_mask10(sm_orig, f"m10_{seed_base}_sm")

    conditions = {"cot": cot_orig, "cot_m": cot_m,
                   "dup": dup_orig, "dup_m": dup_m,
                   "sm": sm_orig, "sm_m": sm_m}

    tf_results = {}
    for name, reas in conditions.items():
        tf_results[name] = teacher_force(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ], correct_answer)

    min_tokens = min((len(r) for r in tf_results.values() if r), default=0)
    if min_tokens == 0:
        return None
    if any(all(not d for _, d, _ in res) for res in tf_results.values()):
        return None

    def compute(orig_key, masked_key):
        kl_oh_o, kl_oh_m, js_vals = [], [], []
        nt = min(len(tf_results[orig_key]), len(tf_results[masked_key]))
        for i in range(nt):
            tok, od, om = tf_results[orig_key][i]
            _, md, mm = tf_results[masked_key][i]
            kl_oh_o.append(kl_onehot(tok, od, om))
            kl_oh_m.append(kl_onehot(tok, md, mm))
            js_vals.append(js_div(od, om, md, mm))
        avg = lambda l: sum(l)/len(l) if l else 0
        return avg(kl_oh_o), avg(kl_oh_m), avg(js_vals)

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    kl_s, kl_sm, js_s = compute("sm", "sm_m")

    record = {
        "question_id": question_id,
        "ground_truth": correct_answer,
        "is_cot_correct": cot_rec.get("is_correct", False),
        "kl_oh_cot": kl_c, "kl_oh_cot_m": kl_cm, "js_cot_cotm": js_c,
        "kl_oh_dup": kl_d, "kl_oh_dup_m": kl_dm, "js_dup_dupm": js_d,
        "kl_oh_sm": kl_s, "kl_oh_sm_m": kl_sm, "js_sm_smm": js_s,
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
    parser.add_argument("--phase", choices=["cot", "direct", "kl", "all"], default="all")
    parser.add_argument("--top_logprobs", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="ablation_0524", help="output base directory (results | ablation_0524)")
    parser.add_argument("--vocab_size", type=int, default=None)
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, TOP_LOGPROBS, OUTDIR, VOCAB_SIZE
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    TOP_LOGPROBS = args.top_logprobs
    OUTDIR = args.outdir
    VOCAB_SIZE = args.vocab_size if args.vocab_size else detect_vocab_size(MODEL)
    print(f"Model: {MODEL} (vocab_size={VOCAB_SIZE}, outdir={OUTDIR})")

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)


    print("Loading Omni-MATH dataset...")
    dataset = load_dataset("KbsdJames/Omni-MATH", split="test")
    data = list(dataset)
    print(f"Total: {len(data)}")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        cot_dir = Path(OUTDIR) / f"exp1_cot_omni_{model_tag}"
        cot_dir.mkdir(parents=True, exist_ok=True)
        cot_path = cot_dir / "omni.jsonl"
        completed = load_completed_ids(cot_path)
        pending = [(i, r) for i, r in enumerate(data) if f"omni_{i}" not in completed]
        print(f"Phase 1: CoT (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, idx, cot_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 2: Direct (teacher forcing KL)
    if args.phase in ("direct", "all"):
        dir_dir = Path(OUTDIR) / f"kl_direct_omni_{model_tag}"
        dir_dir.mkdir(parents=True, exist_ok=True)
        dir_path = dir_dir / "kl_direct.jsonl"
        completed = load_completed_ids(dir_path)
        pending = [(i, r) for i, r in enumerate(data) if f"omni_{i}" not in completed]
        print(f"Phase 2: Direct KL (model: {MODEL}, pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, idx, dir_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 3: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path(OUTDIR) / f"exp1_cot_omni_{model_tag}" / "omni.jsonl"
        if not cot_path.exists():
            print("CoT not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        kl_dir = Path(OUTDIR) / f"kl_mask10_omni_{model_tag}"
        kl_dir.mkdir(parents=True, exist_ok=True)
        kl_path = kl_dir / "kl_mask10.jsonl"
        completed = load_completed_ids(kl_path)
        pending = [(i, r, cot_map[f"omni_{i}"]) for i, r in enumerate(data)
                    if f"omni_{i}" not in completed and f"omni_{i}" in cot_map]
        print(f"Phase 2: KL/JS (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kl, client, row, idx, kl_path, cot) for idx, row, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_m10"):
                    pass

        records = []
        with open(kl_path) as f:
            for line in f:
                records.append(json.loads(line))
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, Omni-MATH)")
            print(f"{'='*70}")
            for label, oh, ohm, js in [
                ("CoT", "kl_oh_cot", "kl_oh_cot_m", "js_cot_cotm"),
                ("Dup", "kl_oh_dup", "kl_oh_dup_m", "js_dup_dupm"),
                ("SM", "kl_oh_sm", "kl_oh_sm_m", "js_sm_smm"),
            ]:
                o = sum(r[oh] for r in records) / n
                m = sum(r[ohm] for r in records) / n
                j = sum(r[js] for r in records) / n
                print(f"  {label:<4}: KL(oh∥orig)={o:.4f}  KL(oh∥masked)={m:.4f}  Δ={m-o:+.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
