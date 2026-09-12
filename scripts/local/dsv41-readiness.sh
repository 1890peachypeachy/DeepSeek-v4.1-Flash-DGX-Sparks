#!/usr/bin/env bash
# Poll the DSV4.1 TP3 head health endpoint until ready, then print boot signature
# lines. Polls HTTP status code (not log text) per the porting-skill rule.
set -u
PORT="${1:-8000}"
WATCH_LIMIT="${2:-1500}"   # seconds cap (default 25 min)
start=$(date +%s)
echo "=== $(date) watching 127.0.0.1:$PORT/health (workers boot headless first) ==="
while true; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "HEALTH_READY http=$code at $(date) after $(( $(date +%s) - start ))s"
    # boot-signature evidence from logs
    L=$(ls -t ~/dsv41/logs/*.log 2>/dev/null | head -1)
    [ -n "$L" ] && { echo "== log tail ($L) =="; grep -aiE "world_size|NCCL world|engram|ready|running|listening|initialize the engine|SGLang" "$L" | tail -6; }
    exit 0
  fi
  now=$(date +%s)
  if [ $(( now - start )) -gt "$WATCH_LIMIT" ]; then
    echo "TIMEOUT after ${WATCH_LIMIT}s (last http=$code)"
    L=$(ls -t ~/dsv41/logs/*.log 2>/dev/null | head -1)
    [ -n "$L" ] && { echo "== log tail ($L) =="; tail -15 "$L"; }
    exit 1
  fi
  sleep 20
done