
import hashlib
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

from llm_client import create_client, make_model_tag, get_tokenizer, extract_think_block, raw_teacher_force_step, thinking_kwargs, THINKING_COT_MAX_TOKENS
import controls
import measures
from paraphrase import TokenMasker
import semantic_edits
import vllm_server

load_dotenv()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROVIDER = "vllm"
MAX_WORKERS = 8
SEED = 42
SENT_RATIO = semantic_edits.DEFAULT_RATE
DEFAULT_EDITOR_MODEL = "claude-opus-4-7"
EDITOR_MAX_TOKENS = 256
JUDGE_MODEL = measures.DEFAULT_JUDGE_MODEL
JUDGE = measures.openai_judge(JUDGE_MODEL)
UNK = TokenMasker.UNK
MASK_RATIO = 0.1
THINKING_MODEL = False
TOP_LOGPROBS = 200
VOCAB_SIZE = None


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
            "followup": GSM8K_FOLLOWUP, "answer_text": ans, "gen_max_tokens": 100,
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
                "followup": MATH_FOLLOWUP, "answer_text": ans, "gen_max_tokens": 100,
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
            "followup": MATH_FOLLOWUP, "answer_text": str(r["answer"]).strip(), "gen_max_tokens": 100,
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
OPEN_DATASETS = ("gsm8k", "math_t", "omni")


def item_seed(question_id):
    return int.from_bytes(hashlib.sha256(f"{SEED}_{question_id}".encode()).digest()[:8], "big")


def make_editor(model):
    """Sentence editor on an Anthropic client, independent of the evaluated model.

    No temperature is sent: the editor model does not accept sampling parameters.
    """
    from anthropic import Anthropic
    return semantic_edits.make_anthropic_editor(Anthropic(), model, max_tokens=EDITOR_MAX_TOKENS)


def _mask_tokens(reasoning, question_id, draw):
    rng = random.Random(f"{SEED}_{question_id}_{draw}_mask")
    return TokenMasker(mask_ratio=MASK_RATIO, rng=rng).mask(reasoning)[0]


def make_variants(editor, reasoning, question_id, draw):
    """Return (variants, draw info).

    del / para / neg edit the same selected sentences; mask is the token-level baseline.
    """
    built = semantic_edits.build_draw(
        reasoning, item_seed(question_id), editor, draw=draw, rate=SENT_RATIO)
    variants = {
        "del": built["delete"],
        "mask": _mask_tokens(reasoning, question_id, draw),
        "para": built["paraphrase"],
        "neg": built["meaning_breaking"],
    }
    return variants, built


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
    """Return (dist, predicted, info); raises if the request fails or no choice is observed."""
    resp = client.chat.completions.create(
        model=MODEL, messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning_text},
            {"role": "user", "content": followup},
        ], temperature=0, max_tokens=1, logprobs=True, top_logprobs=measures.MC_TOP_LOGPROBS)
    return extract_choice_dist(resp.choices[0].logprobs.content, letters)


# ====================================================================
# ====================================================================

def extract_distribution_tf(logprobs_content):
    if not logprobs_content:
        raise ValueError("empty logprobs content")
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
        tokens = vllm_server.detokenize(MODEL, vllm_server.tokenize(MODEL, answer_text))
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
            if THINKING_MODEL and PROVIDER == "vllm":
                d, m = raw_teacher_force_step(client, MODEL, messages, prefix, TOP_LOGPROBS)
            else:
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

VARIANTS = ("del", "mask", "para", "neg")
CONDITIONS = ("orig",) + VARIANTS
ITEM_METRIC_KEYS = tuple(f"{prefix}_{v}" for v in CONDITIONS
                         for prefix in ("kl_oh", "predicted", "is_correct")
                         ) + tuple(f"js_{v}" for v in VARIANTS)


def generate_answer(client, row, reasoning_text):
    resp = client.chat.completions.create(
        model=MODEL, messages=[
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": reasoning_text},
            {"role": "user", "content": row["followup"]},
        ], temperature=0, max_tokens=row["gen_max_tokens"])
    answer = (resp.choices[0].message.content or "").strip()
    if not answer:
        raise ValueError("empty answer from the evaluated model")
    return answer


def _write(output_path, record):
    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_item(client, editor, row, cot_rec, output_path, draw=0, editor_model=None):
    """Compute and store one item; a failure is reported and stored as an ``error`` record."""
    try:
        record = _compute_item(client, editor, row, cot_rec, draw)
    except Exception as e:
        qid = row["question_id"]
        print(f"[ERROR] {qid}: {e}", file=sys.stderr)
        record = {"question_id": qid, "ground_truth": cot_rec.get("ground_truth"),
                  "draw": draw, "error": str(e)}
        record.update({k: None for k in ITEM_METRIC_KEYS})
    if record is not None:
        record["editor_model"] = editor_model
        _write(output_path, record)
    return record


def _mcq_metrics(client, row, ground_truth, texts):
    metrics, p_orig = {}, None
    for name, text in texts.items():
        dist, predicted, info = mcq_answer_dist(
            client, row["prompt"], text, row["followup"], row["letters"])
        if name == "orig":
            p_orig = dist
        else:
            metrics[f"js_{name}"] = mcq_js(p_orig, dist, row["letters"])
        metrics[f"kl_oh_{name}"] = mcq_kl_onehot(ground_truth, dist)
        metrics[f"predicted_{name}"] = predicted
        metrics[f"is_correct_{name}"] = predicted == ground_truth
        metrics[f"top_probs_{name}"] = dist
        metrics[f"choice_mass_observed_{name}"] = info["choice_mass_observed"]
        metrics[f"n_choices_missing_{name}"] = info["n_missing"]
    return metrics


def _open_metrics(client, row, answer_text, texts):
    metrics, orig = {}, None
    for name, text in texts.items():
        res = open_measure(client, row["prompt"], row["followup"], text, answer_text)
        if name == "orig":
            orig = res
        else:
            nt = min(len(orig), len(res))
            metrics[f"js_{name}"] = sum(
                tf_js(orig[i][1], orig[i][2], res[i][1], res[i][2]) for i in range(nt)) / nt
        metrics[f"kl_oh_{name}"] = sum(tf_kl_onehot(t, d, m) for t, d, m in res) / len(res)
        metrics[f"tail_mass_{name}"] = measures.mean_tail_mass(res)
        generated = generate_answer(client, row, text)
        metrics[f"generated_answer_{name}"] = generated
        metrics[f"predicted_{name}"] = generated
        metrics[f"is_correct_{name}"] = measures.judge_correct(answer_text, generated, JUDGE)
    return metrics


def _compute_item(client, editor, row, cot_rec, draw):
    qid = row["question_id"]
    reasoning = cot_rec.get("reasoning", "")
    if cot_rec.get("error") is not None or not reasoning.strip():
        return None

    if row["mode"] == "mcq":
        ground_truth = cot_rec["ground_truth"]
    else:
        ground_truth = (cot_rec.get("ground_truth") or row.get("answer_text") or "").strip()
        if not ground_truth:
            raise ValueError("empty gold answer")

    variants, built = make_variants(editor, reasoning, qid, draw)
    texts = {"orig": built["original"], **variants}
    compute = _mcq_metrics if row["mode"] == "mcq" else _open_metrics
    return {
        "question_id": qid,
        "ground_truth": ground_truth,
        "draw": draw,
        "n_sentences": built["n_sentences"],
        "n_sent_perturbed": len(built["indices"]),
        "sel_indices": built["indices"],
        "n_trace_tokens": controls.count_tokens(reasoning),
        **compute(client, row, ground_truth, texts),
        "error": None,
    }


def output_dir(dataset, model_tag, draw):
    suffix = f"_draw{draw}" if draw else ""
    return Path("results") / f"abl_sent_{dataset}_{model_tag}{suffix}"


def add_arguments(parser):
    parser.add_argument("--thinking-model", action="store_true",
                        help="Thinking model: thinking on for CoT, off for all other requests")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="vllm")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--editor_model", type=str, default=DEFAULT_EDITOR_MODEL,
                        help="OpenAI model that paraphrases and inverts sentences")
    parser.add_argument("--draw", type=int, default=0,
                        help="index of the random sentence selection (separate output per draw)")
    parser.add_argument("--top_logprobs", type=int, default=measures.MC_TOP_LOGPROBS,
                        help="top-k log-probabilities requested per answer position")


def configure(args):
    global MODEL, PROVIDER, MAX_WORKERS, TOP_LOGPROBS, THINKING_MODEL
    THINKING_MODEL = args.thinking_model
    measures.MC_TOP_LOGPROBS = TOP_LOGPROBS = args.top_logprobs
    if args.model:
        MODEL = args.model
    PROVIDER = args.provider
    if args.workers:
        MAX_WORKERS = args.workers


def run(client, dataset, args):
    """Run one dataset's pending items for ``args.draw`` and print the summary."""
    cfg = DATASETS[dataset]
    model_tag = make_model_tag(MODEL)

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

    out_dir = output_dir(dataset, model_tag, args.draw)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "abl_sent.jsonl"
    measures.write_run_config(
        out_path, thinking_model=THINKING_MODEL, model=MODEL, dataset=dataset, phase="sentence_ablation", seed=SEED,
        mask_ratio=None, n_probes=1, draw=args.draw, sentence_rate=SENT_RATIO,
        editor_model=args.editor_model, judge_model=JUDGE_MODEL,
        top_logprobs={"mcq": measures.MC_TOP_LOGPROBS, "open_ended": TOP_LOGPROBS},
        vocab_size=VOCAB_SIZE)
    completed = load_completed_ids(out_path)

    pending = [(row_map[q], cot_map[q]) for q in cot_map
               if q not in completed and q in row_map]
    print(f"AblSent[{dataset}] model={MODEL} vocab={VOCAB_SIZE} draw={args.draw} "
          f"pending: {len(pending)}/{len(cot_map)}")
    if pending:
        editor = make_editor(args.editor_model)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_item, client, editor,
                                 row, cot, out_path, args.draw,
                                 args.editor_model) for row, cot in pending]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Abl_{dataset}"):
                pass
    print_summary(dataset, out_path)


def load_completed_ids(out_path):
    completed = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("error") is None and "is_correct_orig" in record:
                        completed.add(record["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def print_summary(dataset, out_path):
    if not out_path.exists():
        return
    with open(out_path) as f:
        records = [r for r in map(json.loads, f)
                   if r.get("error") is None and "is_correct_orig" in r]
    if not records:
        return
    n = len(records)
    print(f"\n{'='*60}\nSentence ablation ({dataset}, {MODEL}, n={n})\n{'='*60}")
    for v in VARIANTS:
        print(f"  {v:<5}: JS={sum(r[f'js_{v}'] for r in records)/n:.4f}  "
              f"KL(oh)={sum(r[f'kl_oh_{v}'] for r in records)/n:.4f}  "
              f"acc={sum(r[f'is_correct_{v}'] for r in records)/n:.4f}")
    print(f"  orig : KL(oh)={sum(r['kl_oh_orig'] for r in records)/n:.4f}  "
          f"acc={sum(r['is_correct_orig'] for r in records)/n:.4f}")


def main():
    global VOCAB_SIZE, JUDGE, JUDGE_MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--vocab_size", type=int, default=None)
    add_arguments(parser)
    measures.add_judge_argument(parser)
    args = parser.parse_args()
    configure(args)
    JUDGE_MODEL = args.judge_model
    JUDGE = measures.openai_judge(JUDGE_MODEL)
    if args.dataset in OPEN_DATASETS:
        VOCAB_SIZE = measures.resolve_vocab_size(MODEL, args.vocab_size)
    run(create_client(PROVIDER), args.dataset, args)


if __name__ == "__main__":
    main()
