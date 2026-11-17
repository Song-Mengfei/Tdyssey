#!/usr/bin/env bash
# Improved-teacher demonstrations for the new letters E, F, J, P, U (world model wm2 v2, 1 px / 1 deg wiggle):
# 100 training scenes per letter (seeds 800000 + 20*i, i = 46000-46099), recorded continuously (--save-traj), then the
# exact camera frames by replaying the physics (replay_frames.py, verified against the record).
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
R=$WORK/runs; L=$WORK/logs/newletters; OUT=$R/teacher-noisy-demos-new; mkdir -p $OUT/frames $L
W="--wm $R/wm2/v2/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
for s in E F J P U; do for k in 0 50; do
  .venv/bin/python -B scripts/learning/mpc_teacher.py $W --scenes 50 --seed-base 800000 --start $((46000+k)) --shape $s \
      --state-noise 1 --angle-noise 1 --noise-seed $((k+21)) --save-traj $OUT/$s-$((46000+k)).npz --out $OUT/$s-$((46000+k)).json \
      > $L/demos-$s-$k.log 2>&1 &
  sleep 2
done; done
wait
for f in $OUT/*.npz; do
  .venv/bin/python -B scripts/learning/image_wm_data/replay_frames.py $f $OUT/frames/$(basename $f .npz)_frames.npz --workers 6 2>&1 | grep "verified\|Error"
done
for s in E F J P U; do python3 -c "
import json; a = [json.load(open(f'$OUT/$s-{k}.json')) for k in (46000, 46050)]
print('[demos-new] $s teacher on training scenes:', sum(sum(x['success']) for x in a), '/ 100')"; done
echo "[demos-new $(date +%T)] done"
