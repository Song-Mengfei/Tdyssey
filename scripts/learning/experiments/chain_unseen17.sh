#!/usr/bin/env bash
# D052: original vs ten-letter 96 px / 192 px students on the 17 untrained capitals (+ T), 200 fresh scenes each.
source "$(dirname "$0")/common.sh"
OUT=$WORK/runs/unseen17-d052; mkdir -p $OUT $WORK/logs/base-rollouts
SH="T A B C G H I K M N O Q R S V W X Z"
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
declare -A ST=([base]="" [main96]="--student $WORK/runs/dagger10m/round13/train/ema-step20000.pt" [side192]="--student $WORK/runs/dagger10h/round6/train/ema-step20000.pt")
for attempt in 1 2; do
  pids=""
  for k in 0 1 2 3; do for name in base main96 side192; do
    [ -f $OUT/$name-part$k.json ] && continue
    while true; do n=0; for q in $pids; do kill -0 $q 2>/dev/null && n=$((n+1)); done; [ $n -lt 3 ] && [ "$(free_gb)" -ge 20 ] && break; sleep 20; done
    nohup .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 11000000 --start $((50*k)) --scenes 50 --execute 8 \
        --shapes $SH ${ST[$name]} --out $OUT/$name-part$k.json > $WORK/logs/base-rollouts/d052-$name-part$k.log 2>&1 &
    pids="$pids $!"; echo "[d052 $(date +%T)] $name part $k started (attempt $attempt)"; sleep 30
  done; done
  for q in $pids; do while kill -0 $q 2>/dev/null; do sleep 20; done; done
  [ $(ls $OUT/*-part*.json 2>/dev/null | wc -l) -eq 12 ] && break
done
echo "[d052 $(date +%T)] done: $(ls $OUT/*-part*.json | wc -l) of 12 parts"
