
import json
import math
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from llm_client import create_client, make_model_tag

load_dotenv()

MODEL = "gpt-4.1-nano"
PROVIDER = "openai"
PARAPHRASE_MODEL = "gpt-4.1-nano"
MAX_WORKERS = 32
HF_TOKEN = None

PARAPHRASE_SYSTEM = (
    "Paraphrase the following reasoning text. "
    "Preserve the logical structure and all mathematical/scientific content exactly. "
    "Change the wording and sentence structure while keeping the same meaning. "
    "Keep all LaTeX, formulas, and technical terms intact. "
    "Do NOT add any new reasoning, conclusions, or answer selections. "
    "Do NOT state or imply which option is correct."
)

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

write_lock = threading.Lock()


def shuffle_choices(row, rng):
    correct = row["Correct Answer"]
    choices = [correct, row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    rng.shuffle(choices)
    correct_label = chr(65 + choices.index(correct))
    return choices, correct_label


def build_prompt(row, choices):
    domain = row.get("High-level domain", "science")
    return COT_TEMPLATE.format(
        domain=domain,
        question=row["Question"],
        choice_A=choices[0],
        choice_B=choices[1],
        choice_C=choices[2],
        choice_D=choices[3],
    )


def extract_answer_probs(logprobs_content):
    probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    seen = set()
    predicted = None
    if logprobs_content:
        first_token = logprobs_content[0]
        predicted = first_token.token.strip()
        for top_lp in first_token.top_logprobs:
            token = top_lp.token.strip().rstrip(".")
            if token in probs and token not in seen:
                probs[token] = math.exp(top_lp.logprob)
                seen.add(token)
    return probs, predicted


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


def load_source_results(source_dir: Path) -> dict:
    result_map = {}
    for f in sorted(source_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    result_map[rec["question_id"]] = rec
                except json.JSONDecodeError:
                    continue
    return result_map


def llm_paraphrase(client, text):
    try:
        resp = client.chat.completions.create(
            model=PARAPHRASE_MODEL,
            messages=[
                {"role": "system", "content": PARAPHRASE_SYSTEM},
                {"role": "user", "content": text},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return text


def process_item(exp_client, para_client, row, idx, output_path,
                 source_record, choices, ground_truth, source_type, reasoning_key):
    question_id = f"gpqa_{idx}"
    prompt = build_prompt(row, choices)

    if reasoning_key == "paraphrased_reasoning":
        original_reasoning = source_record.get("paraphrased_reasoning", "")
    else:
        original_reasoning = source_record.get("reasoning", "")

    if not original_reasoning:
        return None

    if source_type == "cot":
        orig_prob = source_record.get("correct_token_prob", 0.0)
        orig_predicted = source_record.get("predicted")
    else:
        orig_prob = source_record.get("correct_token_prob", 0.0)
        orig_predicted = source_record.get("predicted")

    paraphrased = llm_paraphrase(para_client, original_reasoning)

    try:
        resp = exp_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": paraphrased},
                {"role": "user", "content": ANSWER_FOLLOWUP},
            ],
            temperature=0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
        )
        probs, predicted = extract_answer_probs(resp.choices[0].logprobs.content)
        correct_prob = probs.get(ground_truth, 0.0)
    except Exception:
        probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        predicted = None
        correct_prob = 0.0

    record = {
        "question_id": question_id,
        "domain": row.get("High-level domain", ""),
        "ground_truth": ground_truth,
        "source_type": source_type,
        "original_predicted": orig_predicted,
        "original_correct_token_prob": orig_prob,
        "original_reasoning": original_reasoning,
        "paraphrased_reasoning": paraphrased,
        "predicted": predicted,
        "correct_token_prob": correct_prob,
        "top_probs": probs,
        "prob_delta": correct_prob - orig_prob,
    }

    with write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def run_experiment(source_type, mask_ratio=None):
    exp_client = create_client(PROVIDER)
    para_client = OpenAI()

    model_tag = make_model_tag(MODEL)

    if source_type == "cot":
        source_dir = Path("results") / f"exp1_cot_gpqa_{model_tag}"
        reasoning_key = "reasoning"
        label = "cot"
    else:
        k = int(mask_ratio * 100)
        source_dir = Path("results") / f"exp3_mask_k{k}_nolatex_gpqa_{model_tag}"
        reasoning_key = "paraphrased_reasoning"
        label = f"mask_k{k}"

    if not source_dir.exists():
        print(f"Source not found: {source_dir}")
        return

    output_dir = Path("results") / f"exp4_fullpara_{label}_gpqa_{model_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gpqa.jsonl"

    print(f"Loading source results from: {source_dir}")
    source_map = load_source_results(source_dir)
    print(f"  Loaded {len(source_map)} records")

    print("Loading GPQA dataset...")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train",
                           token=HF_TOKEN)

    completed_ids = load_completed_ids(output_path)

    pending = []
    for i, row in enumerate(dataset):
        qid = f"gpqa_{i}"
        if qid in completed_ids:
            continue
        if qid not in source_map:
            continue
        choices, correct_label = shuffle_choices(row, random.Random(42 + i))
        pending.append((i, row, choices, correct_label, source_map[qid]))

    print(f"Running full paraphrase experiment ({label}, model={MODEL})")
    print(f"  Total: {len(source_map)}, Done: {len(completed_ids)}, Pending: {len(pending)}")

    if not pending:
        print("All done.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, row, choices, correct_label, src_rec in pending:
            futures.append(executor.submit(
                process_item,
                exp_client, para_client, row, idx, output_path,
                src_rec, choices, correct_label, source_type, reasoning_key,
            ))
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"para_{label}"):
            pass

    print(f"Done! Results: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="all",
                        help="cot, mask_10, mask_20, ..., mask_90, or all")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--provider", type=str, choices=["openai", "litellm", "vllm"], default="openai")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    if args.model:
        global MODEL
        MODEL = args.model
    global PROVIDER
    PROVIDER = args.provider
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    global HF_TOKEN
    HF_TOKEN = args.hf_token or os.environ.get("HF_TOKEN", None)

    if args.source == "all":
        sources = ["cot"] + [f"mask_{k}" for k in range(10, 100, 10)]
    else:
        sources = [args.source]

    for src in sources:
        if src == "cot":
            run_experiment("cot")
        elif src.startswith("mask_"):
            k = int(src.split("_")[1])
            run_experiment("mask", mask_ratio=k / 100)


if __name__ == "__main__":
    main()
