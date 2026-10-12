#!/bin/bash
set -u
cd ~/reasoning_paper
export VLLM_API_BASE=http://localhost:8001/v1
PY=/home/cau/.conda/envs/sangwon/bin/python
VLLM=/home/cau/.conda/envs/sangwon/bin/vllm
ENVV="CUDA_VISIBLE_DEVICES=1 PATH=/home/cau/.conda/envs/sangwon/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=/home/cau/.conda/envs/sangwon/lib/python3.11/site-packages/nvidia/cu13/lib"
ST=logs/MASTER_STATUS
mkdir -p logs
echo "start $(date)" > $ST

serve() {  # serve <model> <match>
  pkill -f "vllm serve" 2>/dev/null; sleep 5
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    case "$(ps -o cmd= -p $p 2>/dev/null)" in *EngineCore*) kill -9 $p 2>/dev/null;; esac
  done
  sleep 3
  env $ENVV nohup $VLLM serve "$1" --host 0.0.0.0 --port 8001 --max-logprobs 200 \
    --gpu-memory-utilization 0.6 --max-model-len 16384 > ~/vllm_master.log 2>&1 &
  local deadline=$(( $(date +%s) + 2400 ))
  until $PY -c "
import urllib.request, json, sys
try:
    d = json.load(urllib.request.urlopen('http://localhost:8001/v1/models', timeout=4))
    sys.exit(0 if '$2'.lower() in d['data'][0]['id'].lower() else 1)
except Exception: sys.exit(1)"; do
    grep -q "Engine core initialization failed" ~/vllm_master.log 2>/dev/null && { echo "FATAL serve $1" >> $ST; exit 1; }
    [ $(date +%s) -gt $deadline ] && { echo "FATAL timeout $1" >> $ST; exit 1; }
    sleep 20
  done
  echo "serving $1" >> $ST
}

gemma_suite() {  # gemma_suite <model> <tag>
  local M="$1"; local TAG="$2"
  $PY run_gsm8k.py --model $M --provider vllm --phase all --workers 16 > logs/gsm8k_${TAG}.log 2>&1 || { echo "FATAL gsm8k $M" >> $ST; exit 1; }
  $PY run_math_test.py --model $M --provider vllm --exp cot --workers 16 > logs/math_t_cot_${TAG}.log 2>&1 || { echo "FATAL math_t_cot $M" >> $ST; exit 1; }
  $PY run_math_test.py --model $M --provider vllm --exp direct --workers 16 > logs/math_t_direct_${TAG}.log 2>&1 || { echo "FATAL math_t_direct $M" >> $ST; exit 1; }
  $PY run_kl_experiment.py --model $M --provider vllm --workers 16 > logs/math_t_kldiv_${TAG}.log 2>&1 || { echo "FATAL math_t_kldiv $M" >> $ST; exit 1; }
  $PY run_kl_mask10.py --model $M --provider vllm --workers 16 > logs/math_t_kl_${TAG}.log 2>&1 || { echo "FATAL math_t_kl $M" >> $ST; exit 1; }
  $PY run_omnimath_kl.py --model $M --provider vllm --phase direct --workers 16 --outdir results > logs/omni_direct_${TAG}.log 2>&1 || { echo "FATAL omni_direct $M" >> $ST; exit 1; }
  $PY run_omnimath_kl.py --model $M --provider vllm --phase kl --workers 16 --outdir results > logs/omni_kl_${TAG}.log 2>&1 || { echo "FATAL omni_kl $M" >> $ST; exit 1; }
  $PY run_agieval.py --model $M --provider vllm --phase all --workers 16 > logs/agie_${TAG}.log 2>&1 || { echo "FATAL agie $M" >> $ST; exit 1; }
  $PY run_mmlu_pro.py --model $M --provider vllm --phase all --workers 16 --outdir results --data_file data/mmlu_pro.jsonl > logs/mmlupro_${TAG}.log 2>&1 || { echo "FATAL mmlupro $M" >> $ST; exit 1; }
  echo "gemma suite done $M" >> $ST
}

for entry in "google/gemma-3-1b-it:gemma-3-1b" "google/gemma-3-4b-it:gemma-3-4b" "google/gemma-3-12b-it:gemma-3-12b" "google/gemma-3-27b-it:gemma-3-27b"; do
  M="${entry%%:*}"; MATCH="${entry#*:}"
  TAG=$(echo "$M" | sed 's|.*/||')
  serve "$M" "$MATCH"
  gemma_suite "$M" "$TAG"
done
echo "GEMMA_ALL_DONE" >> $ST

# 2) Ministral + Gemma ablation
for entry in "mistralai/Ministral-3-3B-Instruct-2512:ministral-3-3b" "mistralai/Ministral-3-8B-Instruct-2512:ministral-3-8b" "mistralai/Ministral-3-14B-Instruct-2512:ministral-3-14b" "google/gemma-3-1b-it:gemma-3-1b" "google/gemma-3-4b-it:gemma-3-4b" "google/gemma-3-12b-it:gemma-3-12b" "google/gemma-3-27b-it:gemma-3-27b"; do
  M="${entry%%:*}"; MATCH="${entry#*:}"
  TAG=$(echo "$M" | sed 's|.*/||; s|\.||g')
  serve "$M" "$MATCH"
  $PY run_sentence_ablation.py --model "$M" --provider vllm --workers 16 > "logs/ablsent_${TAG}.log" 2>&1 || { echo "FATAL ablsent $M" >> $ST; exit 1; }
  echo "ablsent done $M" >> $ST
done

pkill -f "vllm serve" 2>/dev/null
echo "CAU_MASTER_COMPLETE" >> $ST
