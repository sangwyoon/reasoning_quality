
import json
import math
import os
import re
import sys
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from llm_client import create_client, make_model_tag, get_tokenizer
import measures
from paraphrase import TokenMasker

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
SENT_RATIO = 0.1
UNK = "<redacted_UNK>"
TOP_LOGPROBS = 200
VOCAB_SIZE = 151936


def detect_vocab_size(model: str) -> int:
    m = model.lower()
    if "gemma-3" in m:
        return 262144
    if "gemma-2" in m:
        return 256000
    if "llama-3" in m or "llama3" in m:
        return 128256
    if "ministral" in m or "mistral" in m:
        return 131072
    return 151936


# ====================================================================
# ====================================================================

MCQ_RULES = """Think step by step, analyzing the logic behind each option.
Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~

RULES:
- You MAY reason about which options are more or less plausible, and why.
- You MAY state, in theory, what the correct concept or principle is.
- **STOP** before explicitly selecting or labeling any option as the final answer.
- Do NOT write phrases like "therefore the answer is", "the correct option is", "this matches option X", or any equivalent conclusion that names a letter as the answer.
- End your response as if handing off your reasoning to someone else to make the final selection."""

MATH_COT_TEMPLATE = """Solve the following math problem step by step.

Problem: {problem}

Think step by step, analyzing the problem carefully.

RULES:
- You MAY reason about the problem and show your work.
- **STOP** before stating your final answer.
- Do NOT write phrases like "therefore the answer is" or any equivalent conclusion.
- End your response as if handing off your reasoning to someone else to state the final answer.

Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~"""

FOLLOWUP_ABCD = 'Now answer with only a single letter (A, B, C, or D). The answer is'

GSM8K_FOLLOWUP = 'Now state ONLY the final answer\n(a number). Nothing else.\n\nAnswer:'

MATH_FOLLOWUP = 'Now state ONLY the final answer\n(a number, expression, or short mathematical notation). Nothing else.\n\nAnswer:'

PARAPHRASE_PROMPT = """Please paraphrase the following sentence, while preserving the original semantics.

Sentence: {sentence}"""

NEGATE_PROMPT = """Please invert the semantics of the following sentence. Replace key adjectives or relational predicates with antonyms or contradiction. For mathematical expressions, perturb operators or quantities.

Sentence: {sentence}"""

write_lock = threading.Lock()


def _mcq_prompt(header, question, letters, choices):
    options = "\n".join(f"{l}. {c}" for l, c in zip(letters, choices))
    return f"{header}\n\nQuestion: {question}\n{options}\n\n{MCQ_RULES}"


# ====================================================================
# ====================================================================

def load_gpqa():
    from datasets import load_dataset
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", token=token)
    rows = []
    for i, r in enumerate(ds):
        correct = r["Correct Answer"]
        choices = [correct, r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        random.Random(42 + i).shuffle(choices)
        domain = r.get("High-level domain", "science")
        header = f"The following is a multiple choice question about {domain}."
        rows.append({
            "question_id": f"gpqa_{i}", "mode": "mcq", "letters": "ABCD",
            "prompt": _mcq_prompt(header, r["Question"], "ABCD", choices),
            "followup": FOLLOWUP_ABCD,
        })
    return rows


def load_mmlu():
    rows = []
    with open("data/mmlu_full.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            subject = r["subject"].replace("_", " ")
            header = f"The following is a multiple choice question about {subject}."
            rows.append({
                "question_id": f"mmlu_{i}", "mode": "mcq", "letters": "ABCD",
                "prompt": _mcq_prompt(header, r["question"], "ABCD", r["choices"]),
                "followup": FOLLOWUP_ABCD,
            })
    return rows


def load_agie():
    rows = []
    with open("data/agieval_english.jsonl") as f:
        for line in f:
            r = json.loads(line)
            letters = "ABCDE"[:len(r["choices"])]
            ll = ", ".join(letters[:-1]) + f", or {letters[-1]}"
            task = r["task"].replace("_", "-")
            header = f"The following is a multiple choice question ({task})."
            followup = f"Now answer with only a single letter ({ll}). The answer is"
            rows.append({
                "question_id": r["question_id"], "mode": "mcq", "letters": letters,
                "prompt": _mcq_prompt(header, r["question"], letters, r["choices"]),
                "followup": followup,
            })
    return rows


def load_mmlupro():
    rows = []
    with open("data/mmlu_pro.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            n = len(r["options"])
            letters = "".join(chr(65 + k) for k in range(n))
            ll = ", ".join(letters[:-1]) + f", or {letters[-1]}"
            header = f'The following is a multiple choice question about {r.get("category", "general knowledge")}.'
            followup = f"Now answer with only a single letter ({ll}). The answer is"
            rows.append({
                "question_id": f"mmlupro_{i}", "mode": "mcq", "letters": letters,
                "prompt": _mcq_prompt(header, r["question"], letters, r["options"]),
                "followup": followup,
            })
    return rows


def load_gsm8k():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, r in enumerate(ds):
        m = re.search(r'####\s*(.+)', r["answer"])
        ans = m.group(1).strip() if m else ""
        rows.append({
            "question_id": f"gsm_{i}", "mode": "open",
            "prompt": MATH_COT_TEMPLATE.format(problem=r["question"]),
            "followup": GSM8K_FOLLOWUP, "answer_text": ans,
        })
    return rows


def load_math_t():
    rows = []
    with open("data/math_test.jsonl") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            matches = re.findall(r'\\\\boxed\{([^}]+)\}', r["solution"])
            if not matches:
                matches = re.findall(r'\\boxed\{([^}]+)\}', r["solution"])
            ans = matches[-1] if matches else ""
            rows.append({
                "question_id": f"math_t_{i}", "mode": "open",
                "prompt": MATH_COT_TEMPLATE.format(problem=r["problem"]),
                "followup": MATH_FOLLOWUP, "answer_text": ans,
            })
    return rows


def load_omni():
    from datasets import load_dataset
    ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    rows = []
    for i, r in enumerate(ds):
        qid = f"omni_{i}"
        rows.append({
            "question_id": qid, "mode": "open",
            "prompt": MATH_COT_TEMPLATE.format(problem=r["problem"]),
            "followup": MATH_FOLLOWUP, "answer_text": str(r["answer"]).strip(),
        })
    return rows


DATASETS = {
    "gpqa":    {"load": load_gpqa,    "cot_dir": "exp1_cot_gpqa_{tag}",    "cot_file": "gpqa.jsonl"},
    "mmlu":   {"load": load_mmlu,   "cot_dir": "exp1_cot_mmlu_{tag}",   "cot_file": "mmlu.jsonl"},
    "agie":    {"load": load_agie,    "cot_dir": "exp1_cot_agie_{tag}",    "cot_file": "agieval.jsonl"},
    "mmlupro": {"load": load_mmlupro, "cot_dir": "exp1_cot_mmlupro_{tag}", "cot_file": "mmlu_pro.jsonl"},
    "gsm8k":   {"load": load_gsm8k,   "cot_dir": "exp1_cot_gsm8k_{tag}",   "cot_file": "gsm8k.jsonl"},
    "math_t":   {"load": load_math_t,   "cot_dir": "exp1_cot_math_t_{tag}",   "cot_file": "math_test.jsonl"},
    "omni":    {"load": load_omni,    "cot_dir": "exp1_cot_omni_{tag}",    "cot_file": "omni.jsonl"},
}


# ====================================================================
# ====================================================================

def split_sentences(text):
    sents = []
    for para in text.split("\n"):
        if not para.strip():
            sents.append(("\n", None))
            continue
        parts = re.split(r'(?<=[.!?])\s+', para.strip())
        for p in parts:
            if p.strip():
                sents.append(("s", p.strip()))
    return sents


def rebuild(sents):
    out, line = [], []
    for kind, s in sents:
        if kind == "\n":
            if line:
                out.append(" ".join(line)); line = []
            out.append("")
        elif s is not None:
            line.append(s)
    if line:
        out.append(" ".join(line))
    return "\n".join(out)


def select_indices(sents, seed_str):
    idxs = [i for i, (k, s) in enumerate(sents) if k == "s"]
    if not idxs:
        return []
    n = len(idxs)
    k = 1 if n <= 10 else max(1, round(n * SENT_RATIO))
    rng = random.Random(seed_str)
    return sorted(rng.sample(idxs, min(k, n)))


def rewrite_sentence(client, prompt_tmpl, sentence):
    resp = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt_tmpl.format(sentence=sentence)}],
        temperature=0, max_tokens=256)
    out = resp.choices[0].message.content.strip().split("\n")[0].strip()
    return out if out else sentence


def make_variants(client, reasoning, seed_str):
    sents = split_sentences(reasoning)
    sel = select_indices(sents, seed_str)
    if not sel:
        return None, []

    def apply(fn):
        new = list(sents)
        for i in sel:
            new[i] = ("s", fn(sents[i][1]))
        return rebuild(new)

    def delete():
        return rebuild([x for i, x in enumerate(sents) if i not in sel])

    variants = {
        "del": delete(),
        "mask": apply(lambda s: " ".join(UNK for _ in s.split())),
        "para": apply(lambda s: rewrite_sentence(client, PARAPHRASE_PROMPT, s)),
        "neg": apply(lambda s: rewrite_sentence(client, NEGATE_PROMPT, s)),
    }
    return variants, sel


# ====================================================================
# ====================================================================

def extract_choice_dist(logprobs_content, letters):
    """Return (probs, predicted, info); ValueError if no valid choice is observed.

    Choices missing from the top-k get a tail-mass fill (see measures.py).
    """
    return measures.extract_choice_probs(logprobs_content, letters)


def mcq_kl_onehot(gt, dist):
    return measures.kl_onehot_choice(dist, gt)


def mcq_js(p, q, letters):
    return measures.js_choice(p, q, letters)


def mcq_answer_dist(client, prompt, reasoning_text, followup, letters):
    """Return (dist, info); raises if the request fails or no choice is observed."""
    resp = client.chat.completions.create(
        model=MODEL, messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_text},
            {"role": "user", "content": followup},
        ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)
    dist, _, info = extract_choice_dist(resp.choices[0].logprobs.content, letters)
    return dist, info


# ====================================================================
# ====================================================================

def _vllm_base():
    base = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3].rstrip("/")
    return base


def vllm_tokenize(text):
    import requests
    return requests.post(f"{_vllm_base()}/tokenize",
                         json={"model": MODEL, "prompt": text, "add_special_tokens": False},
                         timeout=10).json().get("tokens", [])


def vllm_detokenize(token_ids):
    import requests
    return [requests.post(f"{_vllm_base()}/detokenize", json={"model": MODEL, "tokens": [tid]},
                          timeout=10).json().get("prompt", "") for tid in token_ids]


def extract_distribution_tf(logprobs_content):
    if not logprobs_content:
        return {}, 0.0
    dist, mass = {}, 0.0
    for top_lp in logprobs_content[0].top_logprobs:
        p = math.exp(top_lp.logprob)
        dist[top_lp.token] = p
        mass += p
    return dist, mass


def tf_kl_onehot(correct_token, dist, mass):
    if correct_token in dist and dist[correct_token] > 0:
        return -math.log(dist[correct_token])
    remaining = max(1e-10, 1.0 - mass)
    return -math.log(max(remaining / max(1, VOCAB_SIZE - len(dist)), 1e-30))


def tf_js(dist_p, mass_p, dist_q, mass_q):
    """Full JS over the vocabulary, including the residual-vocabulary term."""
    return measures.js_open_ended(dist_p, mass_p, dist_q, mass_q, VOCAB_SIZE)


def teacher_force(client, messages, answer_text):
    if not answer_text:
        return []
    if PROVIDER == "vllm":
        try:
            tids = vllm_tokenize(answer_text)
            tokens = vllm_detokenize(tids)
        except Exception:
            enc = get_tokenizer(MODEL)
            tokens = [enc.decode([t]) for t in enc.encode(answer_text)]
    else:
        enc = get_tokenizer(MODEL)
        tokens = [enc.decode([t]) for t in enc.encode(answer_text)]
    if not tokens:
        return []
    results, prefix = [], ""
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
            d, m = extract_distribution_tf(resp.choices[0].logprobs.content)
        except Exception as e:
            raise RuntimeError(f"teacher forcing failed at token {len(results)}: {e}") from e
        results.append((tok, d, m))
        prefix += tok
    return results


def open_measure(client, prompt, followup, reasoning_text, answer_text):
    res = teacher_force(client, [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": reasoning_text},
        {"role": "user", "content": followup},
    ], answer_text)
    if not res:
        raise ValueError("no answer tokens to teacher-force")
    return res


# ====================================================================
# ====================================================================

ITEM_METRIC_KEYS = ("kl_oh_orig",) + tuple(
    f"{prefix}_{v}" for v in ("del", "mask", "para", "neg") for prefix in ("js", "kl_oh"))


def process_item(client, row, cot_rec, output_path):
    """Compute and store one item; a failure is reported and stored as an ``error`` record."""
    try:
        return _compute_item(client, row, cot_rec, output_path)
    except Exception as e:
        qid = row["question_id"]
        print(f"[ERROR] {qid}: {e}", file=sys.stderr)
        record = {"question_id": qid, "ground_truth": cot_rec.get("ground_truth"), "error": str(e)}
        record.update({k: None for k in ITEM_METRIC_KEYS})
        with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def _compute_item(client, row, cot_rec, output_path):
    qid = row["question_id"]
    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning:
        return None

    variants, sel = make_variants(client, reasoning, f"{SEED}_{qid}")
    if not variants:
        return None

    record = {
        "question_id": qid,
        "ground_truth": cot_rec.get("ground_truth"),
        "n_sent_perturbed": len(sel),
        "sel_indices": sel,
    }

    if row["mode"] == "mcq":
        letters = row["letters"]
        p_orig = cot_rec.get("top_probs")
        if not p_orig or sum(p_orig.values()) <= 0:
            return None
        record["kl_oh_orig"] = mcq_kl_onehot(cot_rec["ground_truth"], p_orig)
        for name, text in variants.items():
            res = mcq_answer_dist(client, row["prompt"], text, row["followup"], letters)
            dist, info = res
            record[f"js_{name}"] = mcq_js(p_orig, dist, letters)
            record[f"kl_oh_{name}"] = mcq_kl_onehot(cot_rec["ground_truth"], dist)
            record[f"choice_mass_observed_{name}"] = info["choice_mass_observed"]
            record[f"n_choices_missing_{name}"] = info["n_missing"]
    else:
        answer_text = (cot_rec.get("ground_truth") or row.get("answer_text") or "").strip()
        if not answer_text:
            return None
        orig = open_measure(client, row["prompt"], row["followup"], reasoning, answer_text)
        n0 = len(orig)
        record["kl_oh_orig"] = sum(tf_kl_onehot(t, d, m) for t, d, m in orig) / n0
        record["tail_mass_orig"] = measures.mean_tail_mass(orig)
        for name, text in variants.items():
            var = open_measure(client, row["prompt"], row["followup"], text, answer_text)
            nt = min(len(orig), len(var))
            js_vals = [tf_js(orig[i][1], orig[i][2], var[i][1], var[i][2]) for i in range(nt)]
            kl_vals = [tf_kl_onehot(var[i][0], var[i][1], var[i][2]) for i in range(nt)]
            record[f"js_{name}"] = sum(js_vals) / nt
            record[f"kl_oh_{name}"] = sum(kl_vals) / nt
            record[f"tail_mass_{name}"] = measures.mean_tail_mass(var)

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="vllm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")
    args = parser.parse_args()
    measures.MC_TOP_LOGPROBS = args.top_logprobs

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, MAX_WORKERS, VOCAB_SIZE
    PROVIDER = args.provider
    if args.workers:
        MAX_WORKERS = args.workers
    VOCAB_SIZE = args.vocab_size if args.vocab_size else detect_vocab_size(MODEL)

    cfg = DATASETS[args.dataset]
    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    cot_path = Path("results") / cfg["cot_dir"].format(tag=model_tag) / cfg["cot_file"]
    if not cot_path.exists():
        print(f"No CoT: {cot_path}")
        return
    cot_map = {}
    with open(cot_path) as f:
        for line in f:
            r = json.loads(line)
            cot_map[r["question_id"]] = r

    rows = cfg["load"]()
    row_map = {r["question_id"]: r for r in rows}

    out_dir = Path("results") / f"abl_sent_{args.dataset}_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "abl_sent.jsonl"
    measures.write_run_config(
        out_path, model=MODEL, dataset=args.dataset, phase="sentence_ablation", seed=SEED,
        mask_ratio=None, n_probes=1,
        top_logprobs={"mcq": measures.MC_TOP_LOGPROBS, "open_ended": TOP_LOGPROBS})
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("error") is None:
                        completed.add(record["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    pending = [(row_map[q], cot_map[q]) for q in cot_map
               if q not in completed and q in row_map]
    print(f"AblSent[{args.dataset}] model={MODEL} vocab={VOCAB_SIZE} pending: {len(pending)}/{len(cot_map)}")
    if pending:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_item, client, row, cot, out_path) for row, cot in pending]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Abl_{args.dataset}"):
                pass

    records = [r for r in map(json.loads, open(out_path)) if r.get("error") is None] \
        if out_path.exists() else []
    if records:
        n = len(records)
        print(f"\n{'='*60}\nSentence ablation ({args.dataset}, {MODEL}, n={n})\n{'='*60}")
        for v in ("del", "mask", "para", "neg"):
            print(f"  {v:<5}: JS={sum(r[f'js_{v}'] for r in records)/n:.4f}  "
                  f"KL(oh)={sum(r[f'kl_oh_{v}'] for r in records)/n:.4f}")
        print(f"  orig : KL(oh)={sum(r['kl_oh_orig'] for r in records)/n:.4f}")


if __name__ == "__main__":
    main()
