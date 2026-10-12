#!/bin/bash
set -u
cd /data2/reasoning_paper
export VLLM_API_BASE=http://localhost:8000/v1
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
export QWEN3_THINKING=1
PY=/data2/venv8b/bin/python
ST=logs/QWEN3_STATUS
mkdir -p logs
echo "qwen3 start $(date)" > $ST

serve() {  # serve <model> <tp> <match>
  pkill -f "vllm serve" 2>/dev/null; sleep 8
  HF_HUB_OFFLINE=1 nohup /data2/venv8b/bin/vllm serve "$1" --port 8000 --host 0.0.0.0 \
    --max-logprobs 200 --tensor-parallel-size "$2" --max-model-len 16384 \
    --gpu-memory-utilization 0.92 > /data2/tmp/vllm_q3.log 2>&1 &
  local deadline=$(( $(date +%s) + 1800 ))
  until $PY -c "
import urllib.request, json, sys
try:
    d = json.load(urllib.request.urlopen('http://localhost:8000/v1/models', timeout=4))
    sys.exit(0 if '$3' in d['data'][0]['id'] else 1)
except Exception: sys.exit(1)"; do
    grep -q "Engine core initialization failed" /data2/tmp/vllm_q3.log 2>/dev/null && { echo "FATAL serve $1" >> $ST; exit 1; }
    [ $(date +%s) -gt $deadline ] && { echo "FATAL timeout $1" >> $ST; exit 1; }
    sleep 15
  done
  echo "serving $1" >> $ST
}

for entry in "Qwen/Qwen3-0.6B:1:0.6B" "Qwen/Qwen3-1.7B:1:1.7B" "Qwen/Qwen3-4B:1:Qwen3-4B" "Qwen/Qwen3-8B:1:Qwen3-8B" "Qwen/Qwen3-14B:1:14B" "Qwen/Qwen3-32B:2:32B"; do
  M="${entry%%:*}"; rest="${entry#*:}"; TP="${rest%%:*}"; MATCH="${rest#*:}"
  TAG=$(echo "$M" | sed 's|.*/||; s|\.||g')
  serve "$M" "$TP" "$MATCH"
  $PY run_gsm8k.py --model $M --provider vllm --phase all --workers 16 > logs/gsm8k_${TAG}.log 2>&1 || { echo "FATAL gsm8k $M" >> $ST; exit 1; }
  $PY run_math_test.py --model $M --provider vllm --exp cot --workers 16 > logs/math_t_cot_${TAG}.log 2>&1 || { echo "FATAL math_t_cot $M" >> $ST; exit 1; }
  $PY run_math_test.py --model $M --provider vllm --exp direct --workers 16 > logs/math_t_direct_${TAG}.log 2>&1 || { echo "FATAL math_t_direct $M" >> $ST; exit 1; }
  $PY run_kl_experiment.py --model $M --provider vllm --workers 16 > logs/math_t_kldiv_${TAG}.log 2>&1 || { echo "FATAL math_t_kldiv $M" >> $ST; exit 1; }
  $PY run_kl_mask10.py --model $M --provider vllm --workers 16 > logs/math_t_kl_${TAG}.log 2>&1 || { echo "FATAL math_t_kl $M" >> $ST; exit 1; }
  $PY run_omnimath_kl.py --model $M --provider vllm --phase cot --workers 16 --outdir results > logs/omni_cot_${TAG}.log 2>&1 || { echo "FATAL omni_cot $M" >> $ST; exit 1; }
  $PY run_omnimath_kl.py --model $M --provider vllm --phase direct --workers 16 --outdir results > logs/omni_direct_${TAG}.log 2>&1 || { echo "FATAL omni_direct $M" >> $ST; exit 1; }
  $PY run_omnimath_kl.py --model $M --provider vllm --phase kl --workers 16 --outdir results > logs/omni_kl_${TAG}.log 2>&1 || { echo "FATAL omni_kl $M" >> $ST; exit 1; }
  $PY run_mmlu_full.py --model $M --provider vllm --phase all --workers 16 > logs/mmlu_${TAG}.log 2>&1 || { echo "FATAL mmlu $M" >> $ST; exit 1; }
  $PY run_agieval.py --model $M --provider vllm --phase all --workers 16 > logs/agie_${TAG}.log 2>&1 || { echo "FATAL agie $M" >> $ST; exit 1; }
  $PY run_mmlu_pro.py --model $M --provider vllm --phase all --workers 16 --outdir results --data_file data/mmlu_pro.jsonl > logs/mmlupro_${TAG}.log 2>&1 || { echo "FATAL mmlupro $M" >> $ST; exit 1; }
  $PY run_sentence_ablation.py --model $M --provider vllm --workers 16 > logs/ablsent_${TAG}.log 2>&1 || { echo "FATAL ablsent $M" >> $ST; exit 1; }
  echo "qwen3 all done $M" >> $ST
done

pkill -f "vllm serve" 2>/dev/null
echo "QWEN3_MASTER_COMPLETE" >> $ST
