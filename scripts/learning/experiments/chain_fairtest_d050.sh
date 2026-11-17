#!/usr/bin/env bash
# D050: at 15:00 stop the 192 px line, pick its best validation student, test it on the D049 scenes.
source "$(dirname "$0")/common.sh"
R=$WORK/runs; OUT=$R/fairtest-d049; LOG=$WORK/logs
until [ $(date +%H%M) -ge 1500 ]; do sleep 60; done
touch $WORK/stop_rounds6
best=$(grep -h "\[loop10h\] round [0-9]* student (before training)" $LOG/chain_dagger_loop10h.log | sed 's/.*round \([0-9]*\) student.*total \([0-9]*\)/\2 \1/' | sort -n -k1,1 -k2,2 | tail -1)
score=${best% *}; k=${best#* }; prev=$((k-1))
if [ $prev -eq 0 ]; then CAND=$R/dagger10h/round0/train/ema-step30000.pt; else CAND=$R/dagger10h/round$prev/train/ema-step20000.pt; fi
echo "[d050 $(date +%T)] candidate: test round $k ($score of 500) -> $CAND"
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
for attempt in 1 2 3; do
  pids=""
  for p in 0 1 2 3 4 5 6 7 8 9; do
    [ -f $OUT/side192-part$p.json ] && continue
    while true; do n=0; for q in $pids; do kill -0 $q 2>/dev/null && n=$((n+1)); done; [ $n -lt 3 ] && [ "$(free_gb)" -ge 20 ] && break; sleep 20; done
    nohup .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 9000000 --start $((100*p)) --scenes 100 --execute 8 \
        --shapes T T_narrow Y D L E F J P U --student $CAND --out $OUT/side192-part$p.json > $LOG/base-rollouts/d050-part$p.log 2>&1 &
    pids="$pids $!"; echo "[d050 $(date +%T)] part $p started (attempt $attempt)"; sleep 45
  done
  for q in $pids; do while kill -0 $q 2>/dev/null; do sleep 20; done; done
  [ $(ls $OUT/side192-part*.json 2>/dev/null | wc -l) -eq 10 ] && break
done
echo "[d050 $(date +%T)] done: $(ls $OUT/side192-part*.json | wc -l) of 10 parts"
REPORT_RUNS=$OUT python3 -I scripts/learning/final_test_report.py --candidate side192 --scenes 1000 --out $OUT/report-side192.json
