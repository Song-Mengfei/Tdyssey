#!/usr/bin/env bash
# D049 candidate half: the main-line round-13 student on the same 1000 fresh scenes per letter as the original policy
# (seeds 9000000 + 20*i), 10 parts of 100 scenes, up to 3 at a time when >= 20 GB GPU memory is free, failed parts retried.
source "$(dirname "$0")/common.sh"
R=$WORK/runs; OUT=$R/fairtest-d049; CAND=$R/dagger10m/round13/train/ema-step20000.pt; NAME=${1:-main13}
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
for attempt in 1 2 3; do
  pids=""
  for k in 0 1 2 3 4 5 6 7 8 9; do
    [ -f $OUT/$NAME-part$k.json ] && continue
    while true; do
      n=0; for p in $pids; do kill -0 $p 2>/dev/null && n=$((n+1)); done
      [ $n -lt 3 ] && [ "$(free_gb)" -ge 20 ] && break; sleep 20
    done
    nohup .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 9000000 --start $((100*k)) --scenes 100 --execute 8 \
        --shapes T T_narrow Y D L E F J P U --student $CAND --out $OUT/$NAME-part$k.json > $WORK/logs/base-rollouts/d049-$NAME-part$k.log 2>&1 &
    pids="$pids $!"; echo "[d049-$NAME $(date +%T)] part $k started (attempt $attempt)"; sleep 45
  done
  for p in $pids; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
  [ $(ls $OUT/$NAME-part*.json 2>/dev/null | wc -l) -eq 10 ] && break
done
echo "[d049-$NAME $(date +%T)] done: $(ls $OUT/$NAME-part*.json | wc -l) of 10 parts"
REPORT_RUNS=$OUT python3 -I scripts/learning/final_test_report.py --candidate $NAME --scenes 1000 --out $OUT/report-$NAME.json
