#!/usr/bin/env bash
# Second batch of plan-recording teacher demonstrations (alongside chain_plan_demos2.sh, which covers i = 56100-56149):
# all ten letters, 50 new training scenes each (seeds 800000 + 20*i, i = 56150-56199), world model v2, angle-only wiggle,
# then exact 96 px frames by replay for this batch's files. Jobs tracked by PID.
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
R=$WORK/runs; OUT=$R/teacher-plan-demos2; L=$WORK/logs/plan-demos3; mkdir -p $OUT/frames $L
W="--wm $R/wm2/v2/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
pids=""
for s in T T_narrow Y D L E F J P U; do
  .venv/bin/python -B scripts/learning/mpc_teacher.py $W --scenes 50 --seed-base 800000 --start 56150 --shape $s \
      --state-noise 0 --angle-noise 1 --noise-seed 9 --save-traj $OUT/$s-56150.npz --out $OUT/$s-56150.json > $L/$s.log 2>&1 &
  pids="$pids $!"; sleep 2
done
wait $pids
for f in $OUT/*-56150.npz; do
  .venv/bin/python -B scripts/learning/image_wm_data/replay_frames.py $f $OUT/frames/$(basename $f .npz)_frames.npz --workers 6 2>&1 | grep "verified\|Error"
done
echo "[plan-demos3 $(date +%T)] done: $(ls $OUT/*-56150.npz | wc -l) files"
