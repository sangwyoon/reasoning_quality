
import json
import math
import re
import sys
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag, get_tokenizer, extract_think_block, raw_teacher_force_step, thinking_kwargs, THINKING_COT_MAX_TOKENS
import measures
import vllm_server
from paraphrase import split_reasoning_into_steps, TokenMasker, remove_one_step

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
JUDGE = measures.openai_judge(measures.DEFAULT_JUDGE_MODEL)
MAX_WORKERS = 8
TOP_LOGPROBS = 200
THINKING_MODEL = False
VOCAB_SIZE = None
SEED = 42
MASK_RATIO = 0.1

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
Answer with ONLY the final answer (a number). Nothing else.

Problem: {problem}

Answer:"""

ANSWER_FOLLOWUP = 'Now state ONLY the final answer\n(a number). Nothing else.\n\nAnswer:'

write_lock = threading.Lock()


def extract_gsm8k_answer(answer_text):
    match = re.search(r'####\s*(.+)', answer_text)
    if match:
        return match.group(1).strip()
    return ""


def compute_avg_token_prob(logprobs_content):
    if not logprobs_content:
        raise ValueError("empty logprobs content")
    probs = [math.exp(tok.logprob) for tok in logprobs_content]
    return sum(probs) / len(probs)


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


def kl_dist_vs_dist(dist_p, mass_p, dist_q, mass_q):
    remaining_q = max(1e-10, 1.0 - mass_q)
    q_default = remaining_q / max(1, VOCAB_SIZE - len(dist_q))
    kl = 0.0
    for token, p_val in dist_p.items():
        if p_val <= 0:
            continue
        q_val = dist_q.get(token, q_default)
        if q_val <= 0:
            q_val = 1e-30
        kl += p_val * math.log(p_val / q_val)
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
            if THINKING_MODEL and PROVIDER == "vllm":
                d, m = raw_teacher_force_step(client, MODEL, messages, prefix, TOP_LOGPROBS)
            else:
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


def apply_mask10(text, seed_str, probe_id=0):
    """Mask the reasoning trace: each token independently with probability MASK_RATIO."""
    rng = random.Random(f"{seed_str}_p{probe_id}")
    masker = TokenMasker(mask_ratio=MASK_RATIO, rng=rng)
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


def load_completed_ids(output_path):
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


# ---- Phase 1: CoT ----

def process_cot(client, row, idx, output_path):
    question_id = f"gsm_{idx}"
    problem = row["question"]
    correct_answer = extract_gsm8k_answer(row["answer"])

    error = None
    reasoning = ""
    generated_answer = ""
    try:
        prompt = COT_TEMPLATE.format(problem=problem)
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=THINKING_COT_MAX_TOKENS if THINKING_MODEL else 2048,
            **thinking_kwargs(THINKING_MODEL, True))
        reasoning = resp1.choices[0].message.content
        if THINKING_MODEL:
            reasoning = extract_think_block(reasoning)

        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], temperature=0, max_tokens=100, logprobs=True, top_logprobs=TOP_LOGPROBS)
        generated_answer = resp2.choices[0].message.content.strip()
        generated_prob = compute_avg_token_prob(resp2.choices[0].logprobs.content)
        is_correct = measures.judge_correct(correct_answer, generated_answer, JUDGE)
        correct_prob = generated_prob if is_correct else 0.0
    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        is_correct = None
        correct_prob = None
        generated_prob = None

    record = {
        "question_id": question_id,
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


# ---- Phase 1b: Direct ----

def process_direct(client, row, idx, output_path):
    question_id = f"gsm_{idx}"
    problem = row["question"]
    correct_answer = extract_gsm8k_answer(row["answer"])

    error = None
    generated_answer = ""
    try:
        prompt = DIRECT_TEMPLATE.format(problem=problem)
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=100, logprobs=True, top_logprobs=TOP_LOGPROBS)
        generated_answer = resp.choices[0].message.content.strip()
        generated_prob = compute_avg_token_prob(resp.choices[0].logprobs.content)
        is_correct = measures.judge_correct(correct_answer, generated_answer, JUDGE)
        correct_prob = generated_prob if is_correct else 0.0
    except Exception as e:
        print(f"[ERROR] {question_id}: {e}", file=sys.stderr)
        error = str(e)
        is_correct = None
        correct_prob = None
        generated_prob = None

    record = {
        "question_id": question_id,
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


def write_error_record(output_path, question_id, correct_answer, exc, metric_keys):
    """Report a failed item on stderr and store a record with ``error`` and empty metrics."""
    print(f"[ERROR] {question_id}: {exc}", file=sys.stderr)
    record = {"question_id": question_id, "ground_truth": correct_answer, "error": str(exc)}
    record.update({k: None for k in metric_keys})
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


KLDIV_METRIC_KEYS = ("n_tokens", "kl_onehot_cot", "kl_onehot_direct", "kl_cot_direct",
                     "kl_direct_cot", "kl_onehot_cot_per_token", "kl_onehot_direct_per_token",
                     "is_cot_correct")
KL_METRIC_KEYS = ("kl_oh_cot", "kl_oh_cot_m", "js_cot_cotm", "kl_oh_dup", "kl_oh_dup_m",
                  "js_dup_dupm", "kl_oh_sm", "kl_oh_sm_m", "js_sm_smm", "tail_mass",
                  "is_cot_correct")


# ---- Phase 1c: Necessity KL (CoT vs Direct, teacher forcing) ----

def process_kldiv(client, row, idx, output_path, cot_rec):
    question_id = f"gsm_{idx}"
    problem = row["question"]
    correct_answer = extract_gsm8k_answer(row["answer"])
    if not correct_answer.strip():
        return write_error_record(output_path, question_id, correct_answer,
                                  ValueError("empty gold answer"), KLDIV_METRIC_KEYS)

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    cot_messages = [
        {"role": "user", "content": COT_TEMPLATE.format(problem=problem)},
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": ANSWER_FOLLOWUP},
    ]
    direct_messages = [
        {"role": "user", "content": DIRECT_TEMPLATE.format(problem=problem)},
    ]
    try:
        cot_results = teacher_force(client, cot_messages, correct_answer)
        direct_results = teacher_force(client, direct_messages, correct_answer)
    except Exception as e:
        return write_error_record(output_path, question_id, correct_answer, e, KLDIV_METRIC_KEYS)

    if not cot_results or not direct_results:
        return write_error_record(output_path, question_id, correct_answer,
                                  ValueError("no teacher-forcing tokens"), KLDIV_METRIC_KEYS)

    kl_onehot_cot_list, kl_onehot_direct_list = [], []
    kl_cot_direct_list, kl_direct_cot_list = [], []
    n_tokens = min(len(cot_results), len(direct_results))
    for i in range(n_tokens):
        ct_token, ct_dist, ct_mass = cot_results[i]
        dt_token, dt_dist, dt_mass = direct_results[i]
        kl_onehot_cot_list.append(kl_onehot(ct_token, ct_dist, ct_mass))
        kl_onehot_direct_list.append(kl_onehot(dt_token, dt_dist, dt_mass))
        kl_cot_direct_list.append(kl_dist_vs_dist(ct_dist, ct_mass, dt_dist, dt_mass))
        kl_direct_cot_list.append(kl_dist_vs_dist(dt_dist, dt_mass, ct_dist, ct_mass))

    avg = lambda l: sum(l) / len(l) if l else 0.0
    record = {
        "question_id": question_id,
        "ground_truth": correct_answer,
        "n_tokens": n_tokens,
        "kl_onehot_cot": avg(kl_onehot_cot_list),
        "kl_onehot_direct": avg(kl_onehot_direct_list),
        "kl_cot_direct": avg(kl_cot_direct_list),
        "kl_direct_cot": avg(kl_direct_cot_list),
        "kl_onehot_cot_per_token": kl_onehot_cot_list,
        "kl_onehot_direct_per_token": kl_onehot_direct_list,
        "is_cot_correct": cot_rec.get("is_correct", False),
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ---- Phase 2: KL/JS ----

def process_kl(client, row, idx, output_path, cot_rec):
    question_id = f"gsm_{idx}"
    problem = row["question"]
    correct_answer = extract_gsm8k_answer(row["answer"])
    if not correct_answer.strip():
        return write_error_record(output_path, question_id, correct_answer,
                                  ValueError("empty gold answer"), KL_METRIC_KEYS)

    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    prompt = COT_TEMPLATE.format(problem=problem)
    seed_base = f"{SEED}_{question_id}"

    cot_orig = reasoning
    dup_orig, _ = duplicate_step(reasoning, seed_base)
    sm_orig, _ = remove_one_step(reasoning, seed_base)

    cot_m = apply_mask10(cot_orig, f"m10_{seed_base}_cot")
    dup_m = apply_mask10(dup_orig, f"m10_{seed_base}_dup")

    conditions = {"cot": cot_orig, "cot_m": cot_m,
                   "dup": dup_orig, "dup_m": dup_m}
    if sm_orig is not None:
        conditions["sm"] = sm_orig
        conditions["sm_m"] = apply_mask10(sm_orig, f"m10_{seed_base}_sm")

    tf_results = {}
    try:
        for name, reas in conditions.items():
            tf_results[name] = teacher_force(client, [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reas},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], correct_answer)
    except Exception as e:
        return write_error_record(output_path, question_id, correct_answer, e, KL_METRIC_KEYS)

    min_tokens = min((len(r) for r in tf_results.values() if r), default=0)
    if min_tokens == 0:
        return write_error_record(output_path, question_id, correct_answer,
                                  ValueError("no teacher-forcing tokens"), KL_METRIC_KEYS)

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
    if "sm" in tf_results:
        kl_s, kl_sm, js_s = compute("sm", "sm_m")
    else:
        kl_s = kl_sm = js_s = None
    tail_masses = {name: sum(measures.tail_mass(m) for _, _, m in res) / len(res)
                   for name, res in tf_results.items() if res}

    record = {
        "question_id": question_id, "ground_truth": correct_answer,
        "is_cot_correct": cot_rec.get("is_correct", False),
        "kl_oh_cot": kl_c, "kl_oh_cot_m": kl_cm, "js_cot_cotm": js_c,
        "kl_oh_dup": kl_d, "kl_oh_dup_m": kl_dm, "js_dup_dupm": js_d,
        "kl_oh_sm": kl_s, "kl_oh_sm_m": kl_sm, "js_sm_smm": js_s,
        "tail_mass": tail_masses,
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
    parser.add_argument("--phase", choices=["cot", "direct", "kldiv", "kl", "all"], default="all")
    parser.add_argument("--top_logprobs", type=int, default=200)
    parser.add_argument("--vocab_size", type=int, default=None)
    measures.add_judge_argument(parser)
    parser.add_argument("--thinking-model", action="store_true",
                        help="Thinking model: thinking on for CoT, off for all other requests")
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    args = parser.parse_args()
    global THINKING_MODEL
    THINKING_MODEL = args.thinking_model
    global MASK_RATIO
    MASK_RATIO = args.mask_ratio

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, TOP_LOGPROBS, VOCAB_SIZE, JUDGE
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    TOP_LOGPROBS = args.top_logprobs
    VOCAB_SIZE = measures.resolve_vocab_size(MODEL, args.vocab_size)
    JUDGE = measures.openai_judge(args.judge_model)
    print(f"Model: {MODEL} (vocab_size={VOCAB_SIZE})")

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    data = list(dataset)
    print(f"Total: {len(data)}")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        cot_dir = Path("results") / f"exp1_cot_gsm8k_{model_tag}"
        cot_dir.mkdir(parents=True, exist_ok=True)
        cot_path = cot_dir / "gsm8k.jsonl"
        measures.write_run_config(
            cot_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="gsm8k", phase="cot", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=TOP_LOGPROBS, vocab_size=VOCAB_SIZE,
        judge_model=args.judge_model)
        completed = load_completed_ids(cot_path)
        pending = [(i, r) for i, r in enumerate(data) if f"gsm_{i}" not in completed]
        print(f"Phase 1: CoT (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, idx, cot_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 1b: Direct
    if args.phase in ("direct", "all"):
        direct_dir = Path("results") / f"exp2_direct_gsm8k_{model_tag}"
        direct_dir.mkdir(parents=True, exist_ok=True)
        direct_path = direct_dir / "gsm8k.jsonl"
        measures.write_run_config(
            direct_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="gsm8k", phase="direct", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=TOP_LOGPROBS, vocab_size=VOCAB_SIZE,
        judge_model=args.judge_model)
        completed = load_completed_ids(direct_path)
        pending = [(i, r) for i, r in enumerate(data) if f"gsm_{i}" not in completed]
        print(f"Phase 1b: Direct (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, idx, direct_path) for idx, row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 1c: Necessity KL (CoT vs Direct)
    if args.phase in ("kldiv", "all"):
        cot_path = Path("results") / f"exp1_cot_gsm8k_{model_tag}" / "gsm8k.jsonl"
        if not cot_path.exists():
            print("CoT not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        kldiv_dir = Path("results") / f"kl_divergence_gsm8k_{model_tag}"
        kldiv_dir.mkdir(parents=True, exist_ok=True)
        kldiv_path = kldiv_dir / "kl_results.jsonl"
        measures.write_run_config(
            kldiv_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="gsm8k", phase="kldiv", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=TOP_LOGPROBS, vocab_size=VOCAB_SIZE,
        judge_model=args.judge_model)
        completed = load_completed_ids(kldiv_path)
        pending = [(i, r, cot_map[f"gsm_{i}"]) for i, r in enumerate(data)
                    if f"gsm_{i}" not in completed and f"gsm_{i}" in cot_map]
        print(f"Phase 1c: Necessity KL (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kldiv, client, row, idx, kldiv_path, cot) for idx, row, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_div"):
                    pass

        records = [r for r in map(json.loads, open(kldiv_path)) if r.get("error") is None]
        if records:
            n = len(records)
            kc = sum(r["kl_onehot_cot"] for r in records) / n
            kd = sum(r["kl_onehot_direct"] for r in records) / n
            print(f"\nNecessity (n={n}): KL(oh∥CoT)={kc:.4f}  KL(oh∥Dir)={kd:.4f}  Δ_KL={kd-kc:+.4f}")

    # Phase 2: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path("results") / f"exp1_cot_gsm8k_{model_tag}" / "gsm8k.jsonl"
        if not cot_path.exists():
            print("CoT not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        kl_dir = Path("results") / f"kl_mask10_gsm8k_{model_tag}"
        kl_dir.mkdir(parents=True, exist_ok=True)
        kl_path = kl_dir / "kl_mask10.jsonl"
        measures.write_run_config(
            kl_path, thinking_model=THINKING_MODEL, model=MODEL, dataset="gsm8k", phase="kl", seed=SEED,
            mask_ratio=MASK_RATIO, n_probes=1, top_logprobs=TOP_LOGPROBS, vocab_size=VOCAB_SIZE,
        judge_model=args.judge_model)
        completed = load_completed_ids(kl_path)
        pending = [(i, r, cot_map[f"gsm_{i}"]) for i, r in enumerate(data)
                    if f"gsm_{i}" not in completed and f"gsm_{i}" in cot_map]
        print(f"Phase 2: KL/JS (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kl, client, row, idx, kl_path, cot) for idx, row, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_m10"):
                    pass

        records = []
        with open(kl_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("error") is None:
                    records.append(r)
        if records:
            n = len(records)
            print(f"\n{'='*70}")
            print(f"Results (model: {MODEL}, n={n}, GSM8K)")
            print(f"{'='*70}")
            for label, oh, ohm, js in [
                ("CoT", "kl_oh_cot", "kl_oh_cot_m", "js_cot_cotm"),
                ("Dup", "kl_oh_dup", "kl_oh_dup_m", "js_dup_dupm"),
                ("SM", "kl_oh_sm", "kl_oh_sm_m", "js_sm_smm"),
            ]:
                rs = [r for r in records if r[oh] is not None]
                if not rs:
                    continue
                o = sum(r[oh] for r in rs) / len(rs)
                m = sum(r[ohm] for r in rs) / len(rs)
                j = sum(r[js] for r in rs) / len(rs)
                print(f"  {label:<4}: KL(oh∥orig)={o:.4f}  KL(oh∥masked)={m:.4f}  Δ={m-o:+.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
