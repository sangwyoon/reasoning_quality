#!/bin/bash
set -u
cd ~/reasoning_paper
export VLLM_API_BASE=http://localhost:8001/v1
PY=/home/cau/.conda/envs/sangwon/bin/python
MM=mistralai/Ministral-3-3B-Instruct-2512
mkdir -p logs

deadline=$(( $(date +%s) + 3600 ))
until $PY -c "
import urllib.request, json, sys
try:
    d = json.load(urllib.request.urlopen('http://localhost:8001/v1/models', timeout=4))
    sys.exit(0 if 'ministral' in d['data'][0]['id'].lower() else 1)
except Exception:
    sys.exit(1)
"; do
  [ $(date +%s) -gt $deadline ] && { echo "FATAL not_ready" > logs/REMOTE_STATUS; exit 1; }
  sleep 20
done
echo "server confirmed" > logs/REMOTE_STATUS

VOCAB=$($PY -c "
import json, glob
p = glob.glob('/home/cau/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/*/config.json')[0]
c = json.load(open(p))
print(c.get('vocab_size') or c.get('text_config', {}).get('vocab_size'))
")
echo "vocab=$VOCAB" >> logs/REMOTE_STATUS

$PY run_gsm8k.py --model $MM --provider vllm --phase all --workers 16 --vocab_size $VOCAB > logs/gsm8k_ministral3b.log 2>&1 || { echo "FATAL gsm8k" >> logs/REMOTE_STATUS; exit 1; }
echo "gsm8k done" >> logs/REMOTE_STATUS
$PY run_math_test.py --model $MM --provider vllm --exp cot --workers 16 > logs/math_t_cot_ministral3b.log 2>&1 || { echo "FATAL math_t_cot" >> logs/REMOTE_STATUS; exit 1; }
$PY run_math_test.py --model $MM --provider vllm --exp direct --workers 16 > logs/math_t_direct_ministral3b.log 2>&1 || { echo "FATAL math_t_direct" >> logs/REMOTE_STATUS; exit 1; }
$PY run_kl_experiment.py --model $MM --provider vllm --workers 16 --vocab_size $VOCAB > logs/math_t_kldiv_ministral3b.log 2>&1 || { echo "FATAL math_t_kldiv" >> logs/REMOTE_STATUS; exit 1; }
$PY run_kl_mask10.py --model $MM --provider vllm --workers 16 --vocab_size $VOCAB > logs/math_t_kl_ministral3b.log 2>&1 || { echo "FATAL math_t_kl" >> logs/REMOTE_STATUS; exit 1; }
echo "math_t done" >> logs/REMOTE_STATUS
$PY run_omnimath_kl.py --model $MM --provider vllm --phase cot --workers 16 --vocab_size $VOCAB --outdir results > logs/omni_cot_ministral3b.log 2>&1 || { echo "FATAL omni_cot" >> logs/REMOTE_STATUS; exit 1; }
$PY run_omnimath_kl.py --model $MM --provider vllm --phase direct --workers 16 --vocab_size $VOCAB --outdir results > logs/omni_direct_ministral3b.log 2>&1 || { echo "FATAL omni_direct" >> logs/REMOTE_STATUS; exit 1; }
$PY run_omnimath_kl.py --model $MM --provider vllm --phase kl --workers 16 --vocab_size $VOCAB --outdir results > logs/omni_kl_ministral3b.log 2>&1 || { echo "FATAL omni_kl" >> logs/REMOTE_STATUS; exit 1; }
echo "omni done" >> logs/REMOTE_STATUS
$PY run_kl_mask10_gpqa.py --model $MM --provider vllm --phase all --workers 16 --outdir results > logs/gpqa_ministral3b.log 2>&1 || { echo "FATAL gpqa" >> logs/REMOTE_STATUS; exit 1; }
$PY run_mmlu_full.py --model $MM --provider vllm --phase all --workers 16 > logs/mmlu_ministral3b.log 2>&1 || { echo "FATAL mmlu" >> logs/REMOTE_STATUS; exit 1; }
echo "MINISTRAL_ALL_DONE" >> logs/REMOTE_STATUS
