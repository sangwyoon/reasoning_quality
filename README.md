# reasoning_quality
Reasoning Is Most Fragile When It Matters Most

## Setup

```
pip install -r requirements.txt            # experiment runners
pip install -r requirements-analysis.txt   # analysis/
pip install -r requirements-dev.txt        # tests
python -m pytest tests -q
```

`--provider litellm` additionally needs `litellm`. Set `OPENAI_API_KEY` for OpenAI models and the judge, `HF_TOKEN` for gated datasets (GPQA), and `VLLM_API_BASE` for a vLLM server (default `http://localhost:8000/v1`). A `.env` file is read.

## Running experiments

Run from the repository root; outputs go under `results/`.

```
vllm serve <model> --max-logprobs 200
python run_kl_mask10_gpqa.py --model <model> --provider vllm --phase all
python run_mmlu_full.py      --model <model> --provider vllm --phase all
python run_math_test.py      --model <model> --provider vllm --phase all --judge_model <judge>
PYTHONPATH=. python additional_experiments/benchmarks/run_gsm8k.py --model <model> --provider vllm --phase all
```

Other benchmarks and interventions are in `additional_experiments/` (see its README).

- Input files: some runners read local JSONL files (`data/math_test.jsonl`, `data/mmlu_full.jsonl`, `data/agieval_english.jsonl`, `data/mmlu_pro.jsonl`). These files and the scripts that build them are not included.
- Answer judging: `run_math_test.py` and `run_math_qwen3.py` use exact and numeric matching unless `--judge_model` is given.
- Masking: `--mask_ratio` (default 0.1) is a per-token probability; `run_math_test.py` records `n_probes=1` in its `run_config.json`.

## Analysis

```
python -m analysis.predictive_value --input records.parquet --out out/predictive
python -m analysis.flagging         --input records.parquet --out out/flagging
python -m analysis.retained_mass    --input mass.csv        --out out/mass
python -m analysis.table4           --input long.parquet    --out out/table4
python -m analysis.scale_fits       --input long.parquet    --out out/fits
```

Input columns are listed in each module (`REQUIRED_COLUMNS` in `predictive_value.py`, the module docstrings elsewhere). A script that assembles these tables from the runner outputs is not included.

## Semantic perturbations (`semantic_edits.py`)

Implements the sentence-level protocol of appendix D.5 as a pure module (LLM access is an injected `llm(prompt) -> str` callable; `make_openai_editor(client, model)` wraps an OpenAI-style client).

- `split_sentences(trace)` splits losslessly (`"".join(segments) == trace`); decimals, `e.g.`/`i.e.`, numbered lists, LaTeX math and newlines do not create spurious boundaries.
- `select_sentences(n, rate=0.10, seed=...)` returns sorted indices. Count = `max(1, floor(rate*n + 0.5))` for `n >= 1` (half-up rounding, not Python's banker's `round`; short traces always get at least one sentence), and `[]` for `n == 0`.
- The same indices feed all three edits: `paraphrase` (LLM), `meaning_breaking` (LLM) and `delete` (no LLM). The question and unselected sentences are untouched. An empty or failed LLM response raises `EditError`; the original sentence is never substituted silently.
- `build_perturbation_set(trace, seed, llm, rate=0.10, n_samples=1)` is deterministic given `seed` and sample id, and records `n_sentences`, `n_selected`, per-sample seeds/indices and token counts of the edited vs. original trace (the token budget differs from masking; pass a tokenizer's counter via `count_tokens`, default is whitespace words).

## Protocol-matched controls (`controls.py`, `run_protocol_controls.py`)

Necessity compares the with-trace condition (user prompt, assistant trace, user follow-up) with baselines. A single-turn question-only baseline differs from it in dialogue structure as well as in trace content, so the runner also measures baselines with the same three-message structure: `matched_template` (assistant turn holds only a placeholder, default empty), `filler` (neutral sentences cut to the trace's whitespace-token count) and `shuffled` (the trace's own tokens in random order). `direct` (single-turn) is added when an item has `direct_prompt`. For each baseline the output records `necessity_vs_<baseline> = log P_trace(y) - log P_baseline(y)`.

```
python run_protocol_controls.py --items items.jsonl --model M --provider vllm --out out.jsonl
```

Each items line: `{question_id, prompt, reasoning, ground_truth, labels (default A-D), followup?, direct_prompt?}`. The `reasoning` field can come from any source model, so one items file run with different `--model` values feeds one model's traces to other checkpoints (cross-checkpoint trace experiment). Chat templates may treat an empty assistant turn differently from a non-empty one; `--placeholder` sets the text used in the `matched_template` assistant turn. A condition whose call fails is stored with an `error` field and no values; resume re-runs items whose record has an error.

## Log-probability depth (`--top_logprobs`)

Answer distributions use the top-200 log-probabilities at every answer position, with the residual probability mass assigned uniformly (appendix A.5). The default is `measures.MC_TOP_LOGPROBS = 200` for multiple-choice runners and `TOP_LOGPROBS = 200` for open-ended runners; every runner accepts `--top_logprobs` to override it, and the requested value is stored as `top_logprobs_requested` in each `run_config.json`.

The serving stack needs a log-probability limit of at least the requested value (for vLLM, start the server with `--max-logprobs 200`). Hosted APIs may cap the number lower; in that case pass a smaller `--top_logprobs`. The probability mass retained by the returned top-k is recorded per item (`choice_mass_observed`, `tail_mass`).

Items whose teacher-forcing or API call fails are stored with an `error` field and empty metrics (or skipped), and are re-run on resume; no values are substituted.
