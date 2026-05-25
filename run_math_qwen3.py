"""MATH 실험 (Qwen3 thinking model): CoT + Direct + 10% masking KL/JS.
Open-ended, teacher forcing. run_math_test.py + GPQA thinking model 구조 기반.
"""

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

from llm_client import create_client, make_model_tag, get_tokenizer
from paraphrase import split_reasoning_into_steps, TokenMasker

load_dotenv()

MODEL = "Qwen/Qwen3-0.6B"
PROVIDER = "vllm"
MAX_WORKERS = 8
TOP_LOGPROBS = 200
VOCAB_SIZE = 151936
SEED = 42
MASK_RATIO = 0.1
THINKING_MODEL = False
DATA_PATH = "data/math_test_5000.jsonl"

COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Think step by step, analyzing the problem carefully.

RULES:
- You MAY reason about the problem and show your work.
- **STOP** before stating your final answer.
- Do NOT write phrases like "therefore the answer is" or any equivalent conclusion.
- End your response as if handing off your reasoning to someone else to state the final answer."""

DIRECT_TEMPLATE = """Solve the following math problem. Do not explain or show any work.
Answer with ONLY the final answer (a number, expression, or short mathematical notation). Nothing else.

Problem: {problem}

Answer:"""

ANSWER_FOLLOWUP = (
    "Now state ONLY the final answer "
    "(a number, expression, or short mathematical notation). Nothing else.\n"
    "Answer:"
)

write_lock = threading.Lock()


def _extra_body(enable_thinking):
    if THINKING_MODEL:
        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return None


def _extract_think_block(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else text


def extract_boxed_answer(solution):
    matches = re.findall(r'\\\\boxed\{([^}]+)\}', solution)
    if not matches:
        matches = re.findall(r'\\boxed\{([^}]+)\}', solution)
    return matches[-1] if matches else ""


def vllm_tokenize(text):
    import os, requests
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/v1").rstrip("/")
    return requests.post(f"{base}/tokenize", json={"model": MODEL, "prompt": text}, timeout=10).json().get("tokens", [])


def vllm_detokenize(token_ids):
    import os, requests
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/v1").rstrip("/")
    results = []
    for tid in token_ids:
        resp = requests.post(f"{base}/detokenize", json={"model": MODEL, "tokens": [tid]}, timeout=10).json()
        results.append(resp.get("prompt", ""))
    return results


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


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


THINK_BLOCK = "<think>\n\n</think>\n\n"


def _build_raw_prompt(messages, assistant_prefix=""):
    parts = []
    for msg in messages:
        parts.append(f"{IM_START}{msg['role']}\n{msg['content']}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n{THINK_BLOCK}{assistant_prefix}")
    return "".join(parts)


def teacher_force(client, messages, correct_answer):
    answer_text = correct_answer.strip()
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
        try:
            if THINKING_MODEL and PROVIDER == "vllm":
                raw = _build_raw_prompt(messages, assistant_prefix=prefix)
                resp = client.completions.create(
                    model=MODEL, prompt=raw, temperature=0,
                    max_tokens=1, logprobs=TOP_LOGPROBS)
                if resp.choices[0].logprobs and resp.choices[0].logprobs.top_logprobs:
                    d, mass = {}, 0.0
                    for t_str, lp in resp.choices[0].logprobs.top_logprobs[0].items():
                        p = math.exp(lp)
                        d[t_str] = p
                        mass += p
                    d_out, m_out = d, mass
                else:
                    d_out, m_out = {}, 0.0
            else:
                msgs = list(messages)
                extra = {}
                if prefix:
                    msgs.append({"role": "assistant", "content": prefix})
                    if PROVIDER == "vllm":
                        extra["extra_body"] = {"continue_final_message": True, "add_generation_prompt": False}
                resp = client.chat.completions.create(
                    model=MODEL, messages=msgs, temperature=0,
                    max_tokens=1, logprobs=True, top_logprobs=TOP_LOGPROBS, **extra)
                if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
                    d_out, m_out = extract_distribution(resp.choices[0].logprobs.content)
                else:
                    d_out, m_out = {}, 0.0
        except:
            d_out, m_out = {}, 0.0
        results.append((tok, d_out, m_out))
        prefix += tok
    return results


def compute_avg_token_prob(logprobs_content):
    if not logprobs_content:
        return 0.0
    return sum(math.exp(tok.logprob) for tok in logprobs_content) / len(logprobs_content)


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


# ---- Phase 1: CoT ----

def process_cot(client, row, idx, output_path):
    question_id = f"math_{idx}"
    problem = row["problem"]
    correct_answer = row.get("answer", extract_boxed_answer(row.get("solution", "")))

    try:
        prompt = COT_TEMPLATE.format(problem=problem)
        eb1 = _extra_body(enable_thinking=True)
        kwargs1 = {}
        if eb1:
            kwargs1["extra_body"] = eb1
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=8192, **kwargs1)
        raw = resp1.choices[0].message.content
        reasoning = _extract_think_block(raw) if THINKING_MODEL else raw

        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], temperature=0, max_tokens=256, logprobs=True, top_logprobs=5,
            **({} if not _extra_body(False) else {"extra_body": _extra_body(False)}))
        generated_answer = resp2.choices[0].message.content.strip() if resp2.choices[0].message.content else ""
        generated_prob = compute_avg_token_prob(resp2.choices[0].logprobs.content) if resp2.choices[0].logprobs else 0.0

        tf_results = teacher_force(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ], correct_answer)
        correct_prob = sum(d.get(t, 0) for t, d, m in tf_results) / len(tf_results) if tf_results else 0.0
        is_correct = correct_answer.strip().lower() in generated_answer.lower() if correct_answer else False

    except Exception as e:
        reasoning = f"ERROR: {str(e)}"
        generated_answer = ""
        generated_prob = 0.0
        correct_prob = 0.0
        is_correct = False

    record = {
        "question_id": question_id, "type": row.get("type", ""), "level": row.get("level", ""),
        "ground_truth": correct_answer, "generated_answer": generated_answer,
        "is_correct": is_correct, "generated_prob": generated_prob,
        "correct_prob": correct_prob, "reasoning": reasoning,
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ---- Phase 2: Direct ----

def process_direct(client, row, idx, output_path):
    question_id = f"math_{idx}"
    correct_answer = row.get("answer", extract_boxed_answer(row.get("solution", "")))

    try:
        prompt = DIRECT_TEMPLATE.format(problem=row["problem"])
        tf_results = teacher_force(client, [
            {"role": "user", "content": prompt},
        ], correct_answer)
        kl_direct = sum(kl_onehot(t, d, m) for t, d, m in tf_results) / len(tf_results) if tf_results else -math.log(1e-10)
    except:
        kl_direct = -math.log(1e-10)

    record = {"question_id": question_id, "kl_oh_direct": kl_direct}
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ---- Phase 3: KL/JS ----

def process_kl(client, row, idx, output_path, cot_rec):
    question_id = f"math_{idx}"
    correct_answer = row.get("answer", extract_boxed_answer(row.get("solution", "")))
    if not correct_answer.strip():
        return None
    reasoning = cot_rec.get("reasoning", "")
    if not reasoning or "ERROR" in reasoning:
        return None

    prompt = COT_TEMPLATE.format(problem=row["problem"])
    seed_base = f"{SEED}_{question_id}"

    cot_orig = reasoning
    dup_orig, _ = duplicate_step(reasoning, seed_base)
    sm_orig, _ = mask_one_step(reasoning, seed_base)
    cot_m = apply_mask10(cot_orig, f"m10_{seed_base}_cot")
    dup_m = apply_mask10(dup_orig, f"m10_{seed_base}_dup")
    sm_m = apply_mask10(sm_orig, f"m10_{seed_base}_sm")

    conditions = {"cot": cot_orig, "cot_m": cot_m, "dup": dup_orig, "dup_m": dup_m, "sm": sm_orig, "sm_m": sm_m}
    tf = {}
    for name, reas in conditions.items():
        tf[name] = teacher_force(client, [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reas},
            {"role": "user", "content": ANSWER_FOLLOWUP},
        ], correct_answer)

    min_tokens = min((len(r) for r in tf.values() if r), default=0)
    if min_tokens == 0:
        return None

    def compute(o, m):
        kl_o, kl_m, js_vals = [], [], []
        nt = min(len(tf[o]), len(tf[m]))
        for i in range(nt):
            tok, od, om = tf[o][i]
            _, md, mm = tf[m][i]
            kl_o.append(kl_onehot(tok, od, om))
            kl_m.append(kl_onehot(tok, md, mm))
            js_vals.append(js_div(od, om, md, mm))
        avg = lambda l: sum(l)/len(l) if l else 0
        return avg(kl_o), avg(kl_m), avg(js_vals)

    kl_c, kl_cm, js_c = compute("cot", "cot_m")
    kl_d, kl_dm, js_d = compute("dup", "dup_m")
    kl_s, kl_sm, js_s = compute("sm", "sm_m")

    record = {
        "question_id": question_id, "ground_truth": correct_answer,
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
    parser.add_argument("--thinking-model", action="store_true")
    parser.add_argument("--top_logprobs", type=int, default=200)
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, THINKING_MODEL, TOP_LOGPROBS
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    THINKING_MODEL = args.thinking_model
    TOP_LOGPROBS = args.top_logprobs

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f]
    print(f"Loaded {len(data)} MATH questions from {DATA_PATH} (thinking_model={THINKING_MODEL})")

    if args.phase in ("cot", "all"):
        d = Path("ablation_0524") / f"exp1_cot_math_{model_tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "math.jsonl"
        done = load_completed_ids(p)
        pending = [(i, r) for i, r in enumerate(data) if f"math_{i}" not in done]
        print(f"Phase 1: CoT (pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_cot, client, r, i, p) for i, r in pending]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="CoT"):
                    pass

    if args.phase in ("direct", "all"):
        d = Path("ablation_0524") / f"kl_direct_math_{model_tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "kl_direct.jsonl"
        done = load_completed_ids(p)
        pending = [(i, r) for i, r in enumerate(data) if f"math_{i}" not in done]
        print(f"Phase 2: Direct KL (pending: {len(pending)}/{len(data)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_direct, client, r, i, p) for i, r in pending]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="Direct"):
                    pass

    if args.phase in ("kl", "all"):
        cot_path = Path("ablation_0524") / f"exp1_cot_math_{model_tag}" / "math.jsonl"
        if not cot_path.exists():
            print("CoT results not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        d = Path("ablation_0524") / f"kl_mask10_math_{model_tag}"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "kl_mask10.jsonl"
        done = load_completed_ids(p)
        pending = [(i, r, cot_map[f"math_{i}"]) for i, r in enumerate(data)
                    if f"math_{i}" not in done and f"math_{i}" in cot_map]
        print(f"Phase 3: KL/JS (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = [ex.submit(process_kl, client, r, i, p, c) for i, r, c in pending]
                for _ in tqdm(as_completed(futs), total=len(futs), desc="KL_m10"):
                    pass

        records = []
        with open(p) as f:
            records = [json.loads(l) for l in f]
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, MATH)")
            print(f"{'='*70}")
            for label, oh, ohm, js in [("CoT","kl_oh_cot","kl_oh_cot_m","js_cot_cotm"),("Dup","kl_oh_dup","kl_oh_dup_m","js_dup_dupm"),("SM","kl_oh_sm","kl_oh_sm_m","js_sm_smm")]:
                o = sum(r[oh] for r in records)/n
                m = sum(r[ohm] for r in records)/n
                j = sum(r[js] for r in records)/n
                print(f"  {label:<4}: KL(orig)={o:.4f}  KL(masked)={m:.4f}  D={m-o:+.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
