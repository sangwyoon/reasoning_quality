# reasoning_quality
Reasoning Is Most Fragile When It Matters Most

## Setup

```
pip install -r requirements.txt            # experiment runners
pip install -r requirements-analysis.txt   # analysis/
pip install -r requirements-dev.txt        # tests
python -m pytest tests -q
```

`--provider litellm` additionally needs `litellm`. Set `OPENAI_API_KEY` for OpenAI models and the judge, `ANTHROPIC_API_KEY` for the sentence editor, `HF_TOKEN` for gated datasets (GPQA), and `VLLM_API_BASE` for a vLLM server (default `http://localhost:8000/v1`). A `.env` file is read.

## Running experiments

Run from the repository root; outputs go under `results/`.

```
vllm serve <model> --max-logprobs 200
python run_kl_mask10_gpqa.py --model <model> --provider vllm --phase all
python run_mmlu_full.py      --model <model> --provider vllm --phase all
python run_math_test.py      --model <model> --provider vllm --phase all
PYTHONPATH=. python additional_experiments/benchmarks/run_gsm8k.py --model <model> --provider vllm --phase all
```

Other benchmarks and interventions are in `additional_experiments/` (see its README).

- Input files: some runners read local JSONL files (`data/math_test.jsonl`, `data/mmlu_full.jsonl`, `data/agieval_english.jsonl`, `data/mmlu_pro.jsonl`). These files and the scripts that build them are not included.
- Phase order: later phases and the sentence ablation read the CoT output of the same model. `run_mmlu_pro.py` and `run_omnimath_kl.py` need `--outdir results` for the ablation and assembly steps to find their outputs.
- Answer judging: open-ended correctness is decided by an OpenAI judge (`--judge_model`, default `gpt-4.1-nano`; `none` uses exact and numeric matching instead). The judge needs `OPENAI_API_KEY`; a failed judge call stores the item as an error.
- Vocabulary size: runners that complete open-ended top-k distributions over the vocabulary know the Gemma 2/3, Llama 3 and Mistral/Ministral families; for other models (including the Qwen defaults) pass `--vocab_size`. With `--provider openai`, `run_omnimath.py`, `run_paraphrase_math.py` and `run_paraphrase_omnimath.py` use the tiktoken size.
- Tokenizer: teacher forcing uses the vLLM server's tokenizer under `--provider vllm` and tiktoken otherwise; a model unknown to tiktoken is an error.
- Thinking models: pass `--thinking-model` to `run_kl_mask10_gpqa.py`, `run_math_qwen3.py`, `run_mmlu_qwen3.py`, `run_gsm8k.py`, `run_omnimath_kl.py`, `run_agieval.py`, `run_mmlu_pro.py` and the sentence ablation runners: thinking is on for CoT generation and off for all other requests. Output paths are the same with and without it; `run_config.json` of the last five records `thinking_model`.
- Masking: `--mask_ratio` (default 0.1) is a per-token probability; `run_math_test.py` records `n_probes=1` in its `run_config.json`.

## Sentence ablation

For each dataset (`gpqa`, `mmlu`, `agie`, `mmlupro`, `gsm8k`, `math_t`, `omni`), after its CoT phase:

```
PYTHONPATH=. python additional_experiments/semantic_and_structural/run_sentence_ablation_all.py --dataset mmlu --model <model> --provider vllm --draw 0
PYTHONPATH=. python additional_experiments/semantic_and_structural/run_sentence_ablation_all.py --dataset mmlu --model <model> --provider vllm --draw 1
```

Each run re-measures the unedited trace and four edited versions: `del`, `mask` (token-level masking of the whole trace with probability 0.1), `para` and `neg` (meaning-breaking), using `semantic_edits.py` (`del`, `para` and `neg` share the selected sentences). `para` and `neg` are written by `--editor_model` (default `claude-opus-4-7`) on the Anthropic API, so `ANTHROPIC_API_KEY` is needed whatever model is evaluated; no temperature is sent to it. `--draw` picks an independent random sentence selection; draw 0 writes to `results/abl_sent_<dataset>_<model>/`, draw d to `..._draw<d>/`. Records hold the predicted answer and correctness for every condition; open-ended datasets additionally generate an answer per condition (the token limit of the corresponding CoT runner) and judge it (`--judge_model`).

## Analysis

Assemble the input tables from `results/`:

```
python -m analysis.assemble --models models.json --benchmarks gpqa mmlu --validation-draw 1 \
    --robustness mask10 --subtask-field gpqa=domain mmlu=subject --agree-rate agree.csv --out tables
```

`models.json` maps each model name to `{"family": "...", "size_b": <billions of parameters>}`. `--robustness` selects R: `mask10`, or `para`/`neg`/`del`/`mask` together with a `--robustness-draw` different from `--validation-draw`; items where both draws selected the same sentences get an empty R in the predictive tables. `agree.csv` has columns `model, benchmark, question_id, agree_rate`; the runners do not produce answer agreement. `--tables` restricts what is written: `retained_mass` needs only `--models`, `--benchmarks` and `--out`; `table4` and `scale_fits` do not need `--agree-rate`; `predictive_value` and `flagging` cover multiple-choice benchmarks only. Column definitions are in the docstring of `analysis/assemble.py`.

```
python -m analysis.predictive_value --input tables/predictive_value.csv --out out/predictive
python -m analysis.flagging         --input tables/flagging.csv         --out out/flagging
python -m analysis.retained_mass    --input tables/retained_mass.csv    --out out/mass
python -m analysis.table4           --input tables/table4.csv           --out out/table4
python -m analysis.scale_fits       --input tables/scale_fits.csv       --out out/fits
```

Input columns are listed in each module (`REQUIRED_COLUMNS` in `predictive_value.py`, the module docstrings elsewhere); `.parquet` input needs a parquet engine such as `pyarrow`. `scale_fits` fits a curve only for families with at least three sizes.

## Semantic perturbations (`semantic_edits.py`)

Implements the sentence-level protocol of appendix D.5, used by the sentence ablation runners, as a pure module (LLM access is an injected `llm(prompt) -> str` callable; `make_anthropic_editor(client, model, max_tokens)` wraps an Anthropic client).

- `split_sentences(trace)` splits losslessly (`"".join(segments) == trace`); decimals, `e.g.`/`i.e.`, numbered lists, LaTeX math and newlines do not create spurious boundaries.
- `select_sentences(n, rate=0.10, seed=...)` returns sorted indices. Count = `max(1, floor(rate*n + 0.5))` for `n >= 1` (half-up rounding, not Python's banker's `round`; short traces always get at least one sentence), and `[]` for `n == 0`.
- The same indices feed all three edits: `paraphrase` (LLM), `meaning_breaking` (LLM) and `delete` (no LLM). The question and unselected sentences are untouched. An empty or failed LLM response raises `EditError`; the original sentence is never substituted silently.
- `build_draw(trace, seed, llm, draw=0)` returns the three edits of one random selection; the sentence ablation uses it with a per-question seed.
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

An item whose request, teacher forcing, sentence or step rewrite, or answer judging fails (in the direct, KL and ablation phases also one with no gold answer to teacher-force) is stored with an `error` field and empty metrics and is re-run on resume; no value stands in for a failed call. The KL and ablation phases write nothing for an item whose CoT record has an error (the paraphrase runners store an error record for it). The step-removal condition is undefined for single-step traces and left empty; the step-duplication condition leaves such a trace unchanged. In `run_gsm8k.py`, `correct_prob` is 0 for a wrong answer.

Output files are appended to, so a question can have an error record followed by a successful one; use the last error-free record. Do not resume into output directories written with a different version of the code.
