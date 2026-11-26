# Additional experiments

Scripts for the experiments added after the initial release. They import `llm_client.py` and `paraphrase.py` from the repository root, so run them from the root with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python additional_experiments/benchmarks/run_mmlu_pro.py ...`.

- `benchmarks/`: MMLU-Pro, AGIEval, GSM8K, Omni-MATH (CoT / Direct / mask-10% KL-JS).
- `semantic_and_structural/`: paraphrase, meaning-breaking and deletion interventions, step removal and duplication (completeness, redundancy), sentence ablation, exclusion manifest.
  `make_exclusion_manifest.py --base <dir>` (default `results`) lists questions present in the CoT outputs but missing from the ablation / KL outputs. `run_mmlu_pro.py` and `run_omnimath_kl.py` write to `ablation_0524/` unless `--outdir results` is given.
- `remote/`: vLLM launch and runner scripts for the added model families (Llama 3.1 70B, Ministral 3, Qwen 3).
  Paths are set with `REPO_DIR`, `PY` and `VLLM` (defaults: current directory, `python`, `vllm`). These are records of earlier server runs: some call `run_kl_experiment.py`, `run_kl_mask10.py` or an `--exp` option of `run_math_test.py` that the current code does not provide, so they do not run unchanged.
