#!/bin/bash
set -u
cd /data2/reasoning_paper
export VLLM_API_BASE=http://localhost:8000/v1
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
PY=/data2/venv8b/bin/python
ST=logs/MASTER_STATUS
mkdir -p logs
echo "start $(date)" > $ST

serve() {  # serve <model> <tp> <match>
  pkill -f "vllm serve" 2>/dev/null; sleep 8
  HF_HUB_OFFLINE=1 nohup /data2/venv8b/bin/vllm serve "$1" --port 8000 --host 0.0.0.0 \
    --max-logprobs 200 --tensor-parallel-size "$2" --max-model-len 16384 \
    --gpu-memory-utilization 0.92 > /data2/tmp/vllm_master.log 2>&1 &
  local deadline=$(( $(date +%s) + 1800 ))
  until $PY -c "
import urllib.request, json, sys
try:
    d = json.load(urllib.request.urlopen('http://localhost:8000/v1/models', timeout=4))
    sys.exit(0 if '$3' in d['data'][0]['id'] else 1)
except Exception: sys.exit(1)"; do
    grep -q "Engine core initialization failed" /data2/tmp/vllm_master.log 2>/dev/null && { echo "FATAL serve $1" >> $ST; exit 1; }
    [ $(date +%s) -gt $deadline ] && { echo "FATAL timeout $1" >> $ST; exit 1; }
    sleep 15
  done
  echo "serving $1" >> $ST
}

$PY -c "
import urllib.request, json, sys
try:
    d = json.load(urllib.request.urlopen('http://localhost:8000/v1/models', timeout=4))
    sys.exit(0 if '32B' in d['data'][0]['id'] else 1)
except Exception: sys.exit(1)" || serve Qwen/Qwen2.5-32B-Instruct 2 "32B"
$PY run_mmlu_pro.py --model Qwen/Qwen2.5-32B-Instruct --provider vllm --phase all --workers 24 --outdir results --data_file data/mmlu_pro.jsonl > logs/mmlupro_Qwen25-32B.log 2>&1 || { echo "FATAL 32B mmlupro" >> $ST; exit 1; }
echo "32B bench done" >> $ST

# 2) Qwen ablation (0.5B~32B)
for entry in "Qwen/Qwen2.5-0.5B-Instruct:1:0.5B" "Qwen/Qwen2.5-1.5B-Instruct:1:1.5B" "Qwen/Qwen2.5-3B-Instruct:1:-3B" "Qwen/Qwen2.5-7B-Instruct:1:-7B" "Qwen/Qwen2.5-14B-Instruct:1:14B" "Qwen/Qwen2.5-32B-Instruct:2:32B"; do
  M="${entry%%:*}"; rest="${entry#*:}"; TP="${rest%%:*}"; MATCH="${rest#*:}"
  TAG=$(echo "$M" | sed 's|.*/||; s|\.||g')
  serve "$M" "$TP" "$MATCH"
  $PY run_sentence_ablation.py --model "$M" --provider vllm --workers 16 > "logs/ablsent_${TAG}.log" 2>&1 || { echo "FATAL ablsent $M" >> $ST; exit 1; }
  echo "ablsent done $M" >> $ST
done

# 3) Llama ablation (1B/3B/8B/70B)
for entry in "meta-llama/Llama-3.2-1B-Instruct:1:3.2-1B" "meta-llama/Llama-3.2-3B-Instruct:1:3.2-3B" "meta-llama/Llama-3.1-8B-Instruct:1:3.1-8B" "meta-llama/Llama-3.1-70B-Instruct:4:70B"; do
  M="${entry%%:*}"; rest="${entry#*:}"; TP="${rest%%:*}"; MATCH="${rest#*:}"
  TAG=$(echo "$M" | sed 's|.*/||; s|\.||g')
  serve "$M" "$TP" "$MATCH"
  $PY run_sentence_ablation.py --model "$M" --provider vllm --workers 16 > "logs/ablsent_${TAG}.log" 2>&1 || { echo "FATAL ablsent $M" >> $ST; exit 1; }
  echo "ablsent done $M" >> $ST
done

pkill -f "vllm serve" 2>/dev/null
echo "PETER_MASTER_COMPLETE" >> $ST
