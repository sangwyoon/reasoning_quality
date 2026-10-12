
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

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
SENT_RATIO = 0.1
UNK = "[UNK]"

COT_TEMPLATE = """The following is a multiple choice question about {domain}.

Question: {question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Think step by step, analyzing the logic behind each option.
Write the reasoning process as Step 1: ~~, Step 2: ~~ ... Step n: ~~

RULES:
- You MAY reason about which options are more or less plausible, and why.
- You MAY state, in theory, what the correct concept or principle is.
- **STOP** before explicitly selecting or labeling any option as the final answer.
- Do NOT write phrases like "therefore the answer is", "the correct option is", "this matches option X", or any equivalent conclusion that names a letter as the answer.
- End your response as if handing off your reasoning to someone else to make the final selection."""

ANSWER_FOLLOWUP = 'Now answer with only a single letter (A, B, C, or D). The answer is'

PARAPHRASE_PROMPT = """Please paraphrase the following sentence, while preserving the original semantics.

Sentence: {sentence}"""

NEGATE_PROMPT = """Please invert the semantics of the following sentence. Replace key adjectives or relational predicates with antonyms or contradiction. For mathematical expressions, perturb operators or quantities.

Sentence: {sentence}"""

write_lock = threading.Lock()
ANSWER_TOKENS = ("A", "B", "C", "D")


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
        return rebuild([x for x in new if x[1] is not None or x[0] == "\n"])

    def delete():
        new = [x for i, x in enumerate(sents) if i not in sel]
        return rebuild(new)

    variants = {
        "del": delete(),
        "mask": apply(lambda s: " ".join(UNK for _ in s.split())),
        "para": apply(lambda s: rewrite_sentence(client, PARAPHRASE_PROMPT, s)),
        "neg": apply(lambda s: rewrite_sentence(client, NEGATE_PROMPT, s)),
    }
    return variants, sel


def extract_abcd(logprobs_content):
    probs = {t: 0.0 for t in ANSWER_TOKENS}
    if logprobs_content:
        for top_lp in logprobs_content[0].top_logprobs:
            token = top_lp.token.strip().rstrip(".")
            if token in probs:
                probs[token] = max(probs[token], math.exp(top_lp.logprob))
    total = sum(probs.values())
    if total > 0:
        return {k: v / total for k, v in probs.items()}
    return None


def get_answer_dist(client, prompt, reasoning_text):
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning_text},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=5)
        return extract_abcd(resp.choices[0].logprobs.content)
    except Exception:
        return None


def kl_onehot(gt, dist):
    return -math.log(max(dist.get(gt, 1e-10), 1e-10))


def js_div(p, q):
    js = 0.0
    for t in ANSWER_TOKENS:
        pv = max(p.get(t, 1e-10), 1e-10)
        qv = max(q.get(t, 1e-10), 1e-10)
        mv = 0.5 * (pv + qv)
        js += 0.5 * pv * math.log(pv / mv) + 0.5 * qv * math.log(qv / mv)
    return max(js, 0.0)


def build_prompt(row):
    return COT_TEMPLATE.format(
        domain=row.get("domain", "science"), question=row["question"],
        choice_A=row["choices"][0], choice_B=row["choices"][1],
        choice_C=row["choices"][2], choice_D=row["choices"][3])


def process_item(client, row, cot_rec, output_path):
    qid = cot_rec["question_id"]
    reasoning = cot_rec.get("reasoning", "")
    if not reasoning or "ERROR" in reasoning:
        return None
    p_orig = cot_rec.get("top_probs")
    if not p_orig or sum(p_orig.values()) <= 0:
        return None

    try:
        variants, sel = make_variants(client, reasoning, f"{SEED}_{qid}")
    except Exception:
        return None
    if not variants:
        return None

    prompt = build_prompt(row)
    record = {
        "question_id": qid,
        "ground_truth": cot_rec["ground_truth"],
        "n_sent_perturbed": len(sel),
        "sel_indices": sel,
        "kl_oh_orig": kl_onehot(cot_rec["ground_truth"], p_orig),
    }
    for name, text in variants.items():
        dist = get_answer_dist(client, prompt, text)
        if dist is None:
            return None
        record[f"js_{name}"] = js_div(p_orig, dist)
        record[f"kl_oh_{name}"] = kl_onehot(cot_rec["ground_truth"], dist)

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="vllm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER, MAX_WORKERS
    PROVIDER = args.provider
    if args.workers:
        MAX_WORKERS = args.workers

    model_tag = make_model_tag(MODEL)
    client = create_client(PROVIDER)

    import os
    from datasets import load_dataset
    token = args.hf_token or os.environ.get("HF_TOKEN")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train", token=token)
    rows = []
    for i, r in enumerate(ds):
        correct = r["Correct Answer"]
        choices = [correct, r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        random.Random(42 + i).shuffle(choices)
        rows.append({"question_id": f"gpqa_{i}", "question": r["Question"],
                     "choices": choices, "domain": r.get("High-level domain", "science")})

    cot_path = Path("results") / f"exp1_cot_gpqa_{model_tag}" / "gpqa.jsonl"
    if not cot_path.exists():
        print(f"No CoT: {cot_path}"); return
    cot_map = {}
    with open(cot_path) as f:
        for line in f:
            r = json.loads(line)
            cot_map[r["question_id"]] = r

    out_dir = Path("results") / f"abl_sent_gpqa_{model_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "abl_sent.jsonl"
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try: completed.add(json.loads(line)["question_id"])
                except: pass

    row_map = {r["question_id"]: r for r in rows}
    pending = [(row_map[qid], cot_map[qid]) for qid in cot_map
               if qid not in completed and qid in row_map]
    print(f"Sentence ablation (model: {MODEL}) pending: {len(pending)}/{len(cot_map)}")
    if pending:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_item, client, row, cot, out_path) for row, cot in pending]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="AblSent"):
                pass

    records = [json.loads(l) for l in open(out_path)] if out_path.exists() else []
    if records:
        n = len(records)
        print(f"\n{'='*60}\nSentence ablation (model: {MODEL}, n={n}, GPQA)\n{'='*60}")
        for v in ("del", "mask", "para", "neg"):
            js = sum(r[f"js_{v}"] for r in records) / n
            kl = sum(r[f"kl_oh_{v}"] for r in records) / n
            print(f"  {v:<5}: JS={js:.4f}  KL(oh)={kl:.4f}")
        print(f"  orig : KL(oh)={sum(r['kl_oh_orig'] for r in records)/n:.4f}")


if __name__ == "__main__":
    main()
