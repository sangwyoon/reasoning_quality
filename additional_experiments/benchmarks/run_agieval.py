
import json
import math
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import create_client, make_model_tag
from paraphrase import split_reasoning_into_steps, TokenMasker

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
MASK_RATIO = 0.1
DATA_PATH = "data/agieval_english.jsonl"
ALL_LETTERS = "ABCDE"

COT_TEMPLATE = """The following is a multiple choice question ({task}).

Question: {question}
{options_block}

Think step by step, analyzing the logic behind each option.
Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~

RULES:
- You MAY reason about which options are more or less plausible, and why.
- You MAY state, in theory, what the correct concept or principle is.
- **STOP** before explicitly selecting or labeling any option as the final answer.
- Do NOT write phrases like "therefore the answer is", "the correct option is", "this matches option X", or any equivalent conclusion that names a letter as the answer.
- End your response as if handing off your reasoning to someone else to make the final selection."""

DIRECT_TEMPLATE = """The following is a multiple choice question ({task}).

Question: {question}
{options_block}

Do not explain or reason. Answer with only a single letter ({letters_list}).
The answer is"""

ANSWER_FOLLOWUP_TMPL = "Now answer with only a single letter ({letters_list}). The answer is"

write_lock = threading.Lock()


def letters_for(row):
    return ALL_LETTERS[:len(row["choices"])]


def letters_list_str(letters):
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def build_prompt(row, template):
    letters = letters_for(row)
    options_block = "\n".join(f"{l}. {c}" for l, c in zip(letters, row["choices"]))
    return template.format(
        task=row["task"].replace("_", "-"), question=row["question"],
        options_block=options_block, letters_list=letters_list_str(letters),
    )


def extract_choice_dist(logprobs_content, letters):
    probs = {l: 0.0 for l in letters}
    predicted = None
    if logprobs_content:
        first = logprobs_content[0]
        predicted = first.token.strip()
        for top_lp in first.top_logprobs:
            token = top_lp.token.strip().rstrip(".")
            if token in probs:
                probs[token] = max(probs[token], math.exp(top_lp.logprob))
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    else:
        probs = {k: 1.0 / len(letters) for k in probs}
    return probs, predicted


def kl_onehot(correct_token, dist):
    return -math.log(max(dist.get(correct_token, 1e-10), 1e-10))


def js_div(dist_p, dist_q, letters):
    js = 0.0
    for t in letters:
        pv = max(dist_p.get(t, 1e-10), 1e-10)
        qv = max(dist_q.get(t, 1e-10), 1e-10)
        mv = 0.5 * (pv + qv)
        js += 0.5 * pv * math.log(pv / mv) + 0.5 * qv * math.log(qv / mv)
    return max(js, 0.0)


def load_data():
    with open(DATA_PATH) as f:
        return [json.loads(l) for l in f]


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

def process_cot(client, row, output_path):
    question_id = row["question_id"]
    letters = letters_for(row)
    prompt = build_prompt(row, COT_TEMPLATE)
    ground_truth = letters[row["answer"]]
    followup = ANSWER_FOLLOWUP_TMPL.format(letters_list=letters_list_str(letters))

    try:
        resp1 = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1024)
        reasoning = resp1.choices[0].message.content

        resp2 = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning},
                {"role": "user", "content": followup},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)

        probs, predicted = extract_choice_dist(resp2.choices[0].logprobs.content, letters)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception as e:
        reasoning = f"ERROR: {str(e)}"
        probs = {l: 1.0 / len(letters) for l in letters}
        predicted = None
        correct_prob = 1.0 / len(letters)

    record = {
        "question_id": question_id,
        "task": row["task"],
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


# ---- Phase 1b: Direct ----

def process_direct(client, row, output_path):
    question_id = row["question_id"]
    letters = letters_for(row)
    prompt = build_prompt(row, DIRECT_TEMPLATE)
    ground_truth = letters[row["answer"]]

    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)
        probs, predicted = extract_choice_dist(resp.choices[0].logprobs.content, letters)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception:
        probs = {l: 1.0 / len(letters) for l in letters}
        predicted = None
        correct_prob = 1.0 / len(letters)

    record = {
        "question_id": question_id,
        "task": row["task"],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct_token_prob": correct_prob,
        "top_probs": probs,
    }
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ---- Phase 2: KL/JS mask-10% ----

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


def get_answer_dist(client, row, reasoning_text, letters):
    prompt = build_prompt(row, COT_TEMPLATE)
    followup = ANSWER_FOLLOWUP_TMPL.format(letters_list=letters_list_str(letters))
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning_text},
                {"role": "user", "content": followup},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)
        probs, _ = extract_choice_dist(resp.choices[0].logprobs.content, letters)
        return probs
    except Exception:
        return None


def process_kl(client, row, output_path, cot_rec):
    question_id = row["question_id"]
    letters = letters_for(row)
    ground_truth = letters[row["answer"]]

    reasoning = cot_rec.get("reasoning", "")
    if not reasoning or "ERROR" in reasoning:
        return None

    seed_base = f"{SEED}_{question_id}"
    cot_orig = reasoning
    dup_orig, dup_idx = duplicate_step(reasoning, seed_base)
    sm_orig, sm_idx = mask_one_step(reasoning, seed_base)

    conditions = {
        "cot": (cot_orig, apply_mask10(cot_orig, f"m10_{seed_base}_cot")),
        "dup": (dup_orig, apply_mask10(dup_orig, f"m10_{seed_base}_dup")),
        "sm": (sm_orig, apply_mask10(sm_orig, f"m10_{seed_base}_sm")),
    }

    dists = {}
    for name, (orig, masked) in conditions.items():
        d_o = get_answer_dist(client, row, orig, letters)
        d_m = get_answer_dist(client, row, masked, letters)
        if d_o is None or d_m is None:
            return None
        dists[name] = (d_o, d_m)

    record = {
        "question_id": question_id,
        "task": row["task"],
        "ground_truth": ground_truth,
        "is_cot_correct": cot_rec.get("predicted") == ground_truth,
        "dup_step_idx": dup_idx,
        "sm_step_idx": sm_idx,
    }
    for name in ("cot", "dup", "sm"):
        d_o, d_m = dists[name]
        record[f"kl_oh_{name}"] = kl_onehot(ground_truth, d_o)
        record[f"kl_oh_{name}_m"] = kl_onehot(ground_truth, d_m)
        record[f"js_{name}_{name}m"] = js_div(d_o, d_m, letters)

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
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)
    data = load_data()
    print(f"AGIEval English: {len(data)} questions (model: {MODEL})")

    # Phase 1: CoT
    if args.phase in ("cot", "all"):
        out_dir = Path("results") / f"exp1_cot_agie_{model_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "agieval.jsonl"
        completed = load_completed_ids(out_path)
        pending = [r for r in data if r["question_id"] not in completed]
        print(f"Phase 1: CoT (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_cot, client, row, out_path) for row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="CoT"):
                    pass

    # Phase 1b: Direct
    if args.phase in ("direct", "all"):
        out_dir = Path("results") / f"exp2_direct_agie_{model_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "agieval.jsonl"
        completed = load_completed_ids(out_path)
        pending = [r for r in data if r["question_id"] not in completed]
        print(f"Phase 1b: Direct (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_direct, client, row, out_path) for row in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="Direct"):
                    pass

    # Phase 2: KL/JS
    if args.phase in ("kl", "all"):
        cot_path = Path("results") / f"exp1_cot_agie_{model_tag}" / "agieval.jsonl"
        if not cot_path.exists():
            print("CoT results not found.")
            return
        cot_map = {}
        with open(cot_path) as f:
            for line in f:
                r = json.loads(line)
                cot_map[r["question_id"]] = r

        out_dir = Path("results") / f"kl_mask10_agie_{model_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "kl_mask10.jsonl"
        completed = load_completed_ids(out_path)
        pending = [(r, cot_map[r["question_id"]]) for r in data
                   if r["question_id"] not in completed and r["question_id"] in cot_map]
        print(f"Phase 2: KL/JS (pending: {len(pending)})")
        if pending:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(process_kl, client, row, out_path, cot) for row, cot in pending]
                for _ in tqdm(as_completed(futures), total=len(futures), desc="KL_m10"):
                    pass

        records = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
        if records:
            n = len(records)
            print(f"\n{'='*70}\nResults (model: {MODEL}, n={n}, AGIEval-EN)\n{'='*70}")
            for label in ("cot", "dup", "sm"):
                o = sum(r[f"kl_oh_{label}"] for r in records) / n
                m = sum(r[f"kl_oh_{label}_m"] for r in records) / n
                j = sum(r[f"js_{label}_{label}m"] for r in records) / n
                print(f"  {label:<4}: KL(oh∥orig)={o:.4f}  KL(oh∥masked)={m:.4f}  JS={j:.4f}")


if __name__ == "__main__":
    main()
