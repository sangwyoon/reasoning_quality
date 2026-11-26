# Additional experiments

Scripts for the experiments added after the initial release. They import modules from the repository root, so run them from the root with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python additional_experiments/benchmarks/run_mmlu_pro.py ...`.

- `benchmarks/`: MMLU-Pro, AGIEval, GSM8K, Omni-MATH (CoT / Direct / mask-10% KL-JS).
- `semantic_and_structural/`: paraphrase, meaning-breaking and deletion interventions, step removal and duplication (completeness, redundancy), sentence ablation, exclusion manifest.
  `run_sentence_ablation_all.py` covers all datasets and `run_sentence_ablation.py` is its GPQA entry point (see the root README).
  `make_exclusion_manifest.py --base <dir>` (default `results`) lists questions present in the CoT outputs but without an error-free record in the ablation (draw 0) / KL outputs. `run_mmlu_pro.py` and `run_omnimath_kl.py` write to `ablation_0524/` unless `--outdir results` is given.
