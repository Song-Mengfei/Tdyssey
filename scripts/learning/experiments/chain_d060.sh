#!/usr/bin/env bash
# D060 test: the flow policy after its correction rounds (last completed round) against the flow policy without
# rounds (D059 final checkpoint) and the 96 px round-50 student, 200 new scenes per shape, 33 shapes, seeds
# 14000000 + 20*i (8 parts of 25 scenes per policy), paired. The two fixed policies run first (at most CONC parts at
# once, to leave the GPU to the rounds); the corrected policy runs after chain_flow_dagger.sh ends.
# Usage: [FLOW_RUN=flow26 POOL=14000000 OUTNAME=fairtest-d060] chain_d060.sh   (D061: flow26b, 15000000, fairtest-d061)
source "$(dirname "$0")/common.sh"
FLOW_RUN=${FLOW_RUN:-flow26}; POOL=${POOL:-14000000}; OUTNAME=${OUTNAME:-fairtest-d060}
R=$WORK/runs; OUT=$R/$OUTNAME; L=$WORK/logs; mkdir -p $OUT $L/base-rollouts
PY=${PY:-$REPO/.venv/bin/python}; [ -x "$PY" ] || PY=$WORK/repo/.venv/bin/python
SH="T T_narrow A B C D E F G H I J K L M N O P Q R S U V W X Y Z digit3 digit4 digit7 plus hash arrow"
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
run() {   # name checkpoint max-concurrent
  local pids="" name=$1 cand=$2 conc=$3
  for attempt in 1 2; do
    for k in 0 1 2 3 4 5 6 7; do
      [ -f $OUT/$name-part$k.json ] && continue
      while true; do n=0; for q in $pids; do kill -0 $q 2>/dev/null && n=$((n+1)); done; [ $n -lt $conc ] && [ "$(free_gb)" -ge 15 ] && break; sleep 20; done
      nohup $PY -B scripts/learning/base_rollouts.py eval --seed-base $POOL --start $((25*k)) --scenes 25 --execute 8 \
          --shapes $SH --student $cand --out $OUT/$name-part$k.json > $L/base-rollouts/$OUTNAME-$name-part$k.log 2>&1 &
      pids="$pids $!"; sleep 20
    done
    for q in $pids; do while kill -0 $q 2>/dev/null; do sleep 20; done; done
    [ $(ls $OUT/$name-part*.json 2>/dev/null | wc -l) -eq 8 ] && break
  done
}
CONC=${CONC:-2}
run r50 $R/dagger26m/round50/train/ema-step54000.pt $CONC
run flow0 $R/$FLOW_RUN/v1/ema-step160000.pt $CONC
until [ -f $R/$FLOW_RUN/last_round_checkpoint.txt ] && ! pgrep -f "chain_flow_dagger.sh" >/dev/null; do sleep 60; done
C=$(cat $R/$FLOW_RUN/last_round_checkpoint.txt); echo "[d060 $(date +%T)] corrected policy: $C"
run flowdagger $C 4
$PY -I scripts/learning/report_pair.py $OUT r50 flow0 flowdagger > $OUT/report.txt 2>&1
$PY -I scripts/learning/report_pair.py $OUT flow0 flowdagger > $OUT/report-rounds.txt 2>&1
echo "[d060 $(date +%T)] done"; cat $OUT/report.txt $OUT/report-rounds.txt
