# Additional experiments

Scripts for the experiments added after the initial release. They import `llm_client.py` and `paraphrase.py` from the repository root, so run them from the root with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python additional_experiments/benchmarks/run_mmlu_pro.py ...`.

- `benchmarks/`: MMLU-Pro, AGIEval, GSM8K, Omni-MATH (CoT / Direct / mask-10% KL-JS).
- `semantic_and_structural/`: paraphrase, meaning-breaking and deletion interventions, step removal and duplication (completeness, redundancy), sentence ablation, exclusion manifest.
- `remote/`: vLLM launch and runner scripts for the added model families (Llama 3.1 70B, Ministral 3, Qwen 3).
