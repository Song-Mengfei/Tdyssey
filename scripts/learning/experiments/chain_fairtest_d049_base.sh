#!/usr/bin/env bash
# D049, original policy's half, low priority: 1000 fresh scenes per letter (seeds 9000000 + 20*i), ten shapes, as
# 10 parts of 100 scenes (~7.5 GB each), ONE at a time, each started only with >= 25 GB GPU memory free, capped GPU share
# (MPS 35%). Parts not finished tonight run tomorrow beside the candidate. Output runs/fairtest-d049/base-part<k>.json.
source "$(dirname "$0")/common.sh"
R=$WORK/runs; OUT=$R/fairtest-d049; mkdir -p $OUT $WORK/logs/base-rollouts
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
for k in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $OUT/base-part$k.json ] && continue
  [ -f $WORK/stop_baseline ] && break
  until [ "$(free_gb)" -ge 25 ]; do sleep 30; done
  echo "[d049-base $(date +%T)] part $k (scenes $((100*k))-$((100*k+99))) started"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=35 .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 9000000 \
      --start $((100*k)) --scenes 100 --execute 8 --shapes T T_narrow Y D L E F J P U \
      --out $OUT/base-part$k.json > $WORK/logs/base-rollouts/d049-base-part$k.log 2>&1
  [ -f $OUT/base-part$k.json ] || { echo "[d049-base $(date +%T)] part $k failed"; sleep 300; }
done
echo "[d049-base $(date +%T)] stopped: $(ls $OUT/base-part*.json 2>/dev/null | wc -l) of 10 parts done"
