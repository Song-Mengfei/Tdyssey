#!/usr/bin/env bash
# D054 check: after the 96 px line's round 50, the round-50 student vs the D051 pick (round 44) on 200 fresh scenes per
# shape, 33 shapes, seeds 12000000 + 20*i (8 parts of 25 scenes per policy, lockstep), paired. Usage: chain_d054.sh <loop PID>
source "$(dirname "$0")/common.sh"
R=$WORK/runs; OUT=$R/fairtest-d054; L=$WORK/logs; mkdir -p $OUT $L/base-rollouts
while kill -0 $1 2>/dev/null; do sleep 60; done
N=$R/dagger26m/round50/train/ema-step54000.pt; P=$R/dagger26m/round44/train/ema-step54000.pt
[ -f $N ] || { echo "[d054 $(date +%T)] no round-50 student"; exit 1; }
echo "[d054 $(date +%T)] start: r50 $N vs r44 $P"
SH="T T_narrow A B C D E F G H I J K L M N O P Q R S U V W X Y Z digit3 digit4 digit7 plus hash arrow"
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
for attempt in 1 2; do
  pids=""
  for k in 0 1 2 3 4 5 6 7; do for spec in "r44 $P" "r50 $N"; do
    set -- $spec; name=$1; cand=$2
    [ -f $OUT/$name-part$k.json ] && continue
    while true; do n=0; for q in $pids; do kill -0 $q 2>/dev/null && n=$((n+1)); done; [ $n -lt 4 ] && [ "$(free_gb)" -ge 15 ] && break; sleep 20; done
    nohup .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 12000000 --start $((25*k)) --scenes 25 --execute 8 \
        --shapes $SH --student $cand --out $OUT/$name-part$k.json > $L/base-rollouts/d054-$name-part$k.log 2>&1 &
    pids="$pids $!"; sleep 20
  done; done
  for q in $pids; do while kill -0 $q 2>/dev/null; do sleep 20; done; done
  [ $(ls $OUT/r44-part*.json $OUT/r50-part*.json 2>/dev/null | wc -l) -eq 16 ] && break
done
.venv/bin/python -I scripts/learning/report_pair.py $OUT r44 r50 > $OUT/report.txt 2>&1
echo "[d054 $(date +%T)] done"; cat $OUT/report.txt
