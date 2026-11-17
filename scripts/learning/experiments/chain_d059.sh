#!/usr/bin/env bash
# D059 test: the flow-matching policy's final checkpoint vs the 96 px round-50 student on 200 new scenes per shape,
# 33 shapes, seeds 13000000 + 20*i (8 parts of 25 scenes per policy), paired. Round 50 runs first; the flow policy
# runs once its training has written the final checkpoint. Usage: chain_d059.sh
source "$(dirname "$0")/common.sh"
R=$WORK/runs; OUT=$R/fairtest-d059; L=$WORK/logs; mkdir -p $OUT $L/base-rollouts
N=$R/dagger26m/round50/train/ema-step54000.pt; F=$R/flow26/v1/ema-step160000.pt
SH="T T_narrow A B C D E F G H I J K L M N O P Q R S U V W X Y Z digit3 digit4 digit7 plus hash arrow"
PY=${PY:-$REPO/.venv/bin/python}; [ -x "$PY" ] || PY=$WORK/repo/.venv/bin/python
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
run() {   # name checkpoint
  local pids="" name=$1 cand=$2
  for attempt in 1 2; do
    for k in 0 1 2 3 4 5 6 7; do
      [ -f $OUT/$name-part$k.json ] && continue
      while true; do n=0; for q in $pids; do kill -0 $q 2>/dev/null && n=$((n+1)); done; [ $n -lt 4 ] && [ "$(free_gb)" -ge 15 ] && break; sleep 20; done
      nohup $PY -B scripts/learning/base_rollouts.py eval --seed-base 13000000 --start $((25*k)) --scenes 25 --execute 8 \
          --shapes $SH --student $cand --out $OUT/$name-part$k.json > $L/base-rollouts/d059-$name-part$k.log 2>&1 &
      pids="$pids $!"; sleep 20
    done
    for q in $pids; do while kill -0 $q 2>/dev/null; do sleep 20; done; done
    [ $(ls $OUT/$name-part*.json 2>/dev/null | wc -l) -eq 8 ] && break
  done
}
echo "[d059 $(date +%T)] round 50: $N"; run r50 $N
until [ -f $F ] && ! pgrep -f "flow_train.py.*flow26/v1" >/dev/null; do sleep 60; done
echo "[d059 $(date +%T)] flow: $F"; run flow $F
$PY -I scripts/learning/report_pair.py $OUT r50 flow > $OUT/report.txt 2>&1
echo "[d059 $(date +%T)] done"; cat $OUT/report.txt
