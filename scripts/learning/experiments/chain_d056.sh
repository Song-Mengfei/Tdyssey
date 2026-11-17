#!/usr/bin/env bash
# D056: round-50 96 px student executing 12 and 15 of its 16 planned steps on the D054 scenes (8 parts of 25 per
# setting, at most 3 jobs at once beside the 192 px line), then paired reports against its 8-step results (r50).
source "$(dirname "$0")/common.sh"
R=$WORK/runs; OUT=$R/fairtest-d054; L=$WORK/logs/base-rollouts; mkdir -p $L
N=$R/dagger26m/round50/train/ema-step54000.pt
SH="T T_narrow A B C D E F G H I J K L M N O P Q R S U V W X Y Z digit3 digit4 digit7 plus hash arrow"
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
echo "[d056 $(date +%T)] start"
for attempt in 1 2; do
  pids=""
  for e in 15 12; do for k in 0 1 2 3 4 5 6 7; do
    [ -f $OUT/e$e-part$k.json ] && continue
    while true; do n=0; for q in $pids; do kill -0 $q 2>/dev/null && n=$((n+1)); done; [ $n -lt 3 ] && [ "$(free_gb)" -ge 15 ] && break; sleep 20; done
    nohup .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 12000000 --start $((25*k)) --scenes 25 --execute $e \
        --shapes $SH --student $N --out $OUT/e$e-part$k.json > $L/d056-e$e-part$k.log 2>&1 &
    pids="$pids $!"; sleep 20
  done; done
  for q in $pids; do while kill -0 $q 2>/dev/null; do sleep 20; done; done
  [ $(ls $OUT/e12-part*.json $OUT/e15-part*.json 2>/dev/null | wc -l) -eq 16 ] && break
done
.venv/bin/python -I scripts/learning/report_pair.py $OUT r50 e12 e15 > $OUT/report-d056.txt 2>&1
echo "[d056 $(date +%T)] done"; cat $OUT/report-d056.txt
