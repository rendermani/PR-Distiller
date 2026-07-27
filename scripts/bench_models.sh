#!/usr/bin/env bash
# Benchmark different LLMs against the same cached PR-comment corpus.
#
# Pre-requisites: a dev-cache for the target repo must already exist
# (run the pipeline once with the crawler before invoking this script).
#
# Each model gets a clean run: wipe existing rules → switch model →
# start job → poll to completion → save rules + timing to results/.

set -euo pipefail

REPO="${REPO:-tucowsinc/tdp-apis}"
MONTHS="${MONTHS:-5}"
API="${API:-http://localhost:8923}"
TOKEN="${TOKEN:-$(docker exec pr-distiller-backend-1 cat /app/data/.api_token)}"
RESULTS_DIR="/Users/mlautenschlager/Documents/Development/PR-Distiller/results"
mkdir -p "$RESULTS_DIR"

# Models to benchmark — Ollama prefixed for LiteLLM provider routing.
MODELS=(
  "ollama/qwen2.5-coder:7b-instruct"
  "ollama/gemma4:e4b"
  "ollama/gpt-oss:20b"
)

slug() { echo "$1" | sed 's|ollama/||; s|[:/]|-|g'; }

wipe_rules() {
  docker exec pr-distiller-backend-1 python3 -c "
import sys; sys.path.insert(0, '/app')
from db.lightrag_manager import LightRAGManager
db = LightRAGManager()
print('Removed', db.delete_repo_rules('$REPO'), 'rules')
" 2>&1 | grep -v -E "^Warning|Loading weights"
}

set_model() {
  local model="$1"
  local current
  current=$(curl -s "$API/api/config" -H "Authorization: Bearer $TOKEN")
  # Replace only the llm_model field; preserve everything else.
  echo "$current" | python3 -c "
import sys, json, urllib.request
cfg = json.load(sys.stdin)
cfg['llm_model'] = '$model'
# Strip masked secrets before PUT — server preserves them when absent.
for k in list(cfg.keys()):
    if cfg[k] == '***':
        del cfg[k]
print(json.dumps(cfg))
" | curl -s -X PUT "$API/api/config" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    --data-binary @- > /dev/null
  echo "  Model set to: $model"
}

start_job() {
  curl -s -X POST "$API/api/jobs/start" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"repo\": \"$REPO\", \"months\": $MONTHS, \"use_cache\": true}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])"
}

poll_until_done() {
  local job="$1"
  local start_ts
  start_ts=$(date +%s)
  while true; do
    local s p
    s=$(curl -s "$API/api/jobs/status/$job" -H "Authorization: Bearer $TOKEN")
    p=$(echo "$s" | python3 -c "import sys,json; print(json.load(sys.stdin).get('progress',0))" 2>/dev/null || echo 0)
    if [ "$p" = "100" ] || [ "$p" = "-1" ]; then
      echo $(( $(date +%s) - start_ts ))
      return 0
    fi
    sleep 30
  done
}

save_results() {
  local model="$1"
  local secs="$2"
  local slug
  slug=$(slug "$model")
  local ts
  ts=$(date '+%Y%m%d-%H%M%S')
  local out="$RESULTS_DIR/${slug}-${ts}.json"

  local rules
  rules=$(curl -s "$API/api/rules" -H "Authorization: Bearer $TOKEN")
  echo "$rules" | python3 -c "
import sys, json
d = json.load(sys.stdin)
rules = d.get('rules', [])
result = {
    'model': '$model',
    'duration_seconds': $secs,
    'rule_count': len(rules),
    'rules': rules,
}
print(json.dumps(result, indent=2))
" > "$out"
  echo "  Saved: $out ($(python3 -c "print(round($secs/60, 1))") min, $(python3 -c "import json; print(len(json.load(open('$out'))['rules']))") rules)"
}

echo "=== Benchmark start: $(date) ==="
for model in "${MODELS[@]}"; do
  echo ""
  echo "=== $model ==="
  echo "[1/4] Wiping rules..."
  wipe_rules
  echo "[2/4] Switching model..."
  set_model "$model"
  echo "[3/4] Starting job..."
  job=$(start_job)
  echo "  Job: $job"
  echo "[4/4] Polling..."
  secs=$(poll_until_done "$job")
  echo "  Done in ${secs}s"
  save_results "$model" "$secs"
done

echo ""
echo "=== Benchmark complete: $(date) ==="
ls -la "$RESULTS_DIR"
