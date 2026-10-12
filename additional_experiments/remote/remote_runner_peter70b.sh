#!/bin/bash
set -u
cd /data2/reasoning_paper
export VLLM_API_BASE=http://localhost:8000/v1
export HF_HOME=/root/.cache/huggingface
PY=/data2/venv8b/bin/python
M70=meta-llama/Llama-3.1-70B-Instruct
mkdir -p logs

$PY run_kl_experiment.py --model $M70 --provider vllm --workers 24 > logs/math_t_kldiv_llama70b.log 2>&1 || { echo "FATAL math_t_kldiv" > logs/REMOTE_STATUS; exit 1; }
$PY run_kl_mask10.py --model $M70 --provider vllm --workers 24 > logs/math_t_kl_llama70b.log 2>&1 || { echo "FATAL math_t_kl" > logs/REMOTE_STATUS; exit 1; }
echo "math_t done" >> logs/REMOTE_STATUS
$PY run_omnimath_kl.py --model $M70 --provider vllm --phase cot --workers 24 --outdir results > logs/omni_cot_llama70b.log 2>&1 || { echo "FATAL omni_cot" >> logs/REMOTE_STATUS; exit 1; }
$PY run_omnimath_kl.py --model $M70 --provider vllm --phase direct --workers 24 --outdir results > logs/omni_direct_llama70b.log 2>&1 || { echo "FATAL omni_direct" >> logs/REMOTE_STATUS; exit 1; }
$PY run_omnimath_kl.py --model $M70 --provider vllm --phase kl --workers 24 --outdir results > logs/omni_kl_llama70b.log 2>&1 || { echo "FATAL omni_kl" >> logs/REMOTE_STATUS; exit 1; }
echo "omni done" >> logs/REMOTE_STATUS
$PY run_kl_mask10_gpqa.py --model $M70 --provider vllm --phase all --workers 24 --outdir results > logs/gpqa_llama70b.log 2>&1 || { echo "FATAL gpqa" >> logs/REMOTE_STATUS; exit 1; }
$PY run_mmlu_full.py --model $M70 --provider vllm --phase all --workers 24 > logs/mmlu_llama70b.log 2>&1 || { echo "FATAL mmlu" >> logs/REMOTE_STATUS; exit 1; }
echo "SUITE_70B_COMPLETE" >> logs/REMOTE_STATUS
