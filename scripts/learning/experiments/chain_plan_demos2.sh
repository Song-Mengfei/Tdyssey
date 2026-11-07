#!/usr/bin/env bash
# Teacher demonstrations that also record the teacher's INTENDED 16-step plan at every step (mpc_teacher.py --save-traj
# now stores `plan`), for the label A/B (executed pushes vs intended plan). All ten letters, 50 new training scenes each
# (seeds 800000 + 20*i, i = 56100-56199), world model v2, angle-only wiggle (0 px, 1 deg: best on near-T in the sweep).
# Then exact 96x96 frames by replay. Jobs tracked by PID.
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
R=$WORK/runs; OUT=$R/teacher-plan-demos2; L=$WORK/logs/plan-demos; mkdir -p $OUT/frames $L
W="--wm $R/wm2/v2/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
pids=""
for s in T T_narrow Y D L E F J P U; do
  .venv/bin/python -B scripts/learning/mpc_teacher.py $W --scenes 50 --seed-base 800000 --start $((56100+k)) --shape $s \
      --state-noise 0 --angle-noise 1 --noise-seed 7 --save-traj $OUT/$s-56000.npz --out $OUT/$s-56000.json > $L/$s.log 2>&1 &
  pids="$pids $!"; sleep 2
done
wait $pids
for f in $OUT/*.npz; do
  .venv/bin/python -B scripts/learning/image_wm_data/replay_frames.py $f $OUT/frames/$(basename $f .npz)_frames.npz --workers 6 2>&1 | grep "verified\|Error"
done
for s in T T_narrow Y D L E F J P U; do python3 -c "import json; j=json.load(open('$OUT/$s-56000.json')); print('[plan-demos] $s teacher:', sum(j['success']), '/ 50')"; done
echo "[plan-demos $(date +%T)] done"
