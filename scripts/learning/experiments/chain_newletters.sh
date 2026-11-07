#!/usr/bin/env bash
# Adding letters E, F, J, P, U (decided with the project lead 2026-10-31). Starts after the D046 fair test (memory).
#  1. dynamics data: the original policy with Gaussian action noise 0/10/25/50 px on T + the new letters
#     (value pool 600000 + 20*i, i = 1000-1399; same recipe as the first five letters' dyn-data)
#  2. world model wm2 v2 on all ten letters (old dyn-data + new), TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
#  3. teacher check: planner with 1 px / 1 deg wiggle on 50 validation scenes for E, F, J, P, U, and D, L (old letters)
#  4. original camera policy on the new letters (same 50 validation scenes), as the baseline
source "$(dirname "$0")/common.sh"
R=$WORK/runs; L=$WORK/logs; mkdir -p $L/newletters $R/dyn-data-new $R/dyn-data-10
# (On the cluster this first waited for the D046 fair test, chain_fairtest_fix.sh, to finish.)
echo "[new $(date +%T)] start"
# The earlier dynamics data (runs/dyn-data, runs/dyn-data-wide) must be in $WORK/runs. Set TDYSSEY_STORE to a Hugging
# Face repository holding a copy (owner/name) to fetch it first.
if [ -n "${TDYSSEY_STORE:-}" ]; then .venv/bin/python -I -c "
from huggingface_hub import snapshot_download
snapshot_download('$TDYSSEY_STORE', local_dir='$WORK', allow_patterns=['runs/dyn-data/*', 'runs/dyn-data-wide/*'])
print('old dyn-data downloaded')"; fi
k=0
for noise in 0 10 25 50; do for half in 0 1; do
  while [ "$(pgrep -fc 'base_rollouts.py collect')" -ge 4 ]; do sleep 20; done
  start=$((1000 + 50*k))
  .venv/bin/python -B scripts/learning/base_rollouts.py collect --start $start --scenes 50 --action-noise $noise \
      --shapes T E F J P U --out $R/dyn-data-new/part$((k+20))-noise$noise.npz > $L/newletters/collect-$k.log 2>&1 &
  k=$((k+1)); sleep 3
done; done
wait
echo "[new $(date +%T)] collected $(ls $R/dyn-data-new/*.npz | wc -l) parts"
for f in $R/dyn-data/*.npz $R/dyn-data-wide/*.npz $R/dyn-data-new/*.npz; do ln -sf $f $R/dyn-data-10/$(basename $f); done
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
.venv/bin/python -B scripts/learning/wm2.py train --data $R/dyn-data-10 --out $R/wm2/v2 > $L/newletters/wm2-v2.log 2>&1
[ -f $R/wm2/v2/model.pt ] || { echo "[new] wm2 v2 training failed"; exit 1; }
echo "[new $(date +%T)] wm2 v2 trained"
P="--wm $R/wm2/v2/model.pt --scenes 50 --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
for s in E F J P U D L; do
  .venv/bin/python -B scripts/learning/mpc_teacher.py $P --shape $s --state-noise 1 --angle-noise 1 \
      --out $R/newletters/teacher-v2-$s.json > $L/newletters/teacher-v2-$s.log 2>&1 &
  sleep 2
done
.venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --scenes 50 --shapes T E F J P U \
    --out $R/newletters/base-x8.json > $L/newletters/base-eval.log 2>&1 &
wait
for s in E F J P U D L; do python3 -c "import json,numpy as np; j=json.load(open('$R/newletters/teacher-v2-$s.json')); print('[new] teacher $s:', sum(j['success']), '/', len(j['success']), 'coverage', round(float(np.mean(j['final_coverage'])), 3))"; done
python3 -c "import json; s=json.load(open('$R/newletters/base-x8.json'))['summary']; print('[new] original policy:', '  '.join(f\"{k} {v['success']} ({v['mean_final_coverage']:.2f})\" for k,v in s.items()))"
echo "[new $(date +%T)] done"
