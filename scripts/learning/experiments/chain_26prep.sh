#!/usr/bin/env bash
# Preparation for the 26-letter run (agreed 2026-11-02): same recipe as chain_newletters.sh (E F J P U), for the other
# 17 letters A B C G H I K M N O Q R S V W X Z.
#  1. dynamics data: original policy with Gaussian action noise 0/10/25/50 px, each group = T + 5-6 new letters
#     (value pool 600000 + 20*i, i = 1400-1799; 3 groups x 4 noise levels x 2 halves of 50 scenes)
#  2. world model wm2 v3 on all 27 shapes (all earlier dyn-data + new)
#  3. teacher check (1 px / 1 deg wiggle) on 50 validation scenes for the 17 letters and T, D, L
#  4. teacher demonstrations with plans (angle-only wiggle, as plan demos) for every new letter whose teacher check is
#     >= 30/50: 150 training scenes per letter (seeds 800000 + 20*i, i = 50600-50749), then 96 px frames.
# GPU memory guard: each job starts only with >= 15 GB free (the fair test and the 192 px line share the GPU).
source "$(dirname "$0")/common.sh"
R=$WORK/runs; L=$WORK/logs/prep26; mkdir -p $L $R/dyn-data-26new $R/dyn-data-27 $R/prep26
NEW="A B C G H I K M N O Q R S V W X Z"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U,A,B,C,G,H,I,K,M,N,O,Q,R,S,V,W,X,Z
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
alive() { local n=0; for p in $pids; do kill -0 $p 2>/dev/null && n=$((n+1)); done; echo $n; }
echo "[prep26 $(date +%T)] start"
# 1. dynamics data
pids=""; k=0
for group in "A B C G H I" "K M N O Q R" "S V W X Z"; do
  for noise in 0 10 25 50; do for half in 0 1; do
    while [ "$(alive)" -ge 4 ] || [ "$(free_gb)" -lt 15 ]; do sleep 20; done
    start=$((1400 + 50*(k % 8)))
    .venv/bin/python -B scripts/learning/base_rollouts.py collect --start $start --scenes 50 --action-noise $noise \
        --shapes T $group --out $R/dyn-data-26new/part$((k+40))-noise$noise.npz > $L/collect-$k.log 2>&1 &
    pids="$pids $!"; k=$((k+1)); sleep 5
  done; done
done
for p in $pids; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
echo "[prep26 $(date +%T)] dynamics data: $(ls $R/dyn-data-26new/*.npz | wc -l) of 24 parts, $(grep -l Traceback $L/collect-*.log 2>/dev/null | wc -l) crashed"
# 2. world model on all 27 shapes
for f in $R/dyn-data-10/*.npz $R/dyn-data-26new/*.npz; do ln -sf $(readlink -f $f) $R/dyn-data-27/$(basename $f); done
.venv/bin/python -B scripts/learning/wm2.py train --data $R/dyn-data-27 --out $R/wm2/v3 > $L/wm2-v3.log 2>&1
[ -f $R/wm2/v3/model.pt ] || { echo "[prep26] wm2 v3 training failed"; exit 1; }
echo "[prep26 $(date +%T)] wm2 v3 trained"
# 3. teacher check
P="--wm $R/wm2/v3/model.pt --scenes 50 --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
pids=""
for s in $NEW T D L; do
  while [ "$(alive)" -ge 10 ] || [ "$(free_gb)" -lt 15 ]; do sleep 20; done
  .venv/bin/python -B scripts/learning/mpc_teacher.py $P --shape $s --state-noise 1 --angle-noise 1 \
      --out $R/prep26/teacher-v3-$s.json > $L/teacher-v3-$s.log 2>&1 &
  pids="$pids $!"; sleep 2
done
for p in $pids; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
GOOD=""
for s in $NEW T D L; do
  n=$(python3 -c "import json; print(sum(json.load(open('$R/prep26/teacher-v3-$s.json'))['success']))" 2>/dev/null || echo 0)
  echo "[prep26] teacher $s: $n / 50"
  case " $NEW " in *" $s "*) [ "$n" -ge 30 ] && GOOD="$GOOD $s";; esac
done
echo "[prep26 $(date +%T)] letters with a good teacher (>= 30/50):$GOOD"
# 4. teacher demonstrations with plans for those letters
OUT=$R/teacher-plan-demos26; mkdir -p $OUT/frames
W="--wm $R/wm2/v3/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
pids=""
for s in $GOOD; do for k in 0 50 100; do
  while [ "$(alive)" -ge 12 ] || [ "$(free_gb)" -lt 15 ]; do sleep 20; done
  .venv/bin/python -B scripts/learning/mpc_teacher.py $W --scenes 50 --seed-base 800000 --start $((50600+k)) --shape $s \
      --state-noise 0 --angle-noise 1 --noise-seed $((k+21)) --save-traj $OUT/$s-$((50600+k)).npz --out $OUT/$s-$((50600+k)).json \
      > $L/demos-$s-$k.log 2>&1 &
  pids="$pids $!"; sleep 2
done; done
for p in $pids; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
for f in $OUT/*.npz; do
  .venv/bin/python -B scripts/learning/image_wm_data/replay_frames.py $f $OUT/frames/$(basename $f .npz)_frames.npz --workers 6 > /dev/null 2>&1
done
echo "[prep26 $(date +%T)] done: $(ls $OUT/*.npz 2>/dev/null | wc -l) demo files, $(ls $OUT/frames 2>/dev/null | wc -l) with frames"
