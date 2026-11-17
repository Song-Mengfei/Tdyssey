#!/usr/bin/env bash
# Ten-letter correction rounds (T, near-T, Y, D, L + E, F, J, P, U), after the 10-letter student's first training.
# Round k: test the current student on 50 validation scenes per letter (all ten), let it drive (probability 0.7 per
# 8-control decision) on 30 new training scenes per letter while the improved teacher (world model wm2 v2,
# 1 px / 1 deg) labels every visited state, then fine-tune on all demonstrations + all correction rounds (20k steps).
# v2: the test runs in 5 parallel parts alongside training (no waiting). Usage: chain_dagger_loop10v2.sh <first student checkpoint> <first round number> <last round number>
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
R=$WORK/runs; L=$WORK/logs; mkdir -p $L/bc $L/mpc-data $R/dagger10p
LETTERS="T T_narrow Y D L E F J P U"
DEMOS="$R/teacher-plan-demos $R/teacher-plan-demos2"; HUMAN=$WORK/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
W="--wm $R/wm2/v2/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
student=$1; first=$2; last=$3
for k in $(seq $first $last); do
  out=$R/dagger10p/round$k; mkdir -p $out
  echo "[loop10p $(date +%T)] round $k: test + collect with $student"
  evals=""                                   # test in 2 parallel parts; training does NOT wait for it
  for part in 0 25; do
    .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --start $part --scenes 25 --execute 8 \
        --shapes $LETTERS --student $student --out $out/eval-in-x8-part$part.json > $L/bc/dagger10p-round$k-eval-$part.log 2>&1 &
    evals="$evals $!"
  done
  data=$R/mpc-data/dagger10p-r$k; coll=""
  for s in $LETTERS; do
    .venv/bin/python -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes 30 \
        --shape $s --start $((48000 + 100*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/dagger10p-r$k-$s.log 2>&1 &
    coll="$coll $!"; sleep 2
  done
  wait $coll
  ( while pgrep -f "dagger10p/round$k/eval-in" > /dev/null; do sleep 20; done
    python3 -c "
import json, glob
s = {}
for f in glob.glob('$out/eval-in-x8-part*.json'):
    for k2, v in json.load(open(f))['summary'].items():
        s.setdefault(k2, 0); s[k2] += v['success']
print('[loop10p] round $k student (before training):', '  '.join(f'{a} {b}' for a, b in s.items()), ' total', sum(s.values()))" ) &
  echo "[loop10p $(date +%T)] round $k collected $(ls $data/data 2>/dev/null | wc -l) scenes, $(grep -l Traceback $L/mpc-data/dagger10p-r$k-* 2>/dev/null | wc -l) crashed"
  RECS=$(ls -d $R/mpc-data/dagger-bc1/data $R/mpc-data/dagger-r*/data $R/mpc-data/dagger10-r*/data $R/mpc-data/dagger10p-r*/data $R/lastmile/lastmile-r*/data 2>/dev/null | tr '\n' ' ')
  [ -f $out/train/ema-step20000.pt ] || rm -rf $out/train          # a half-finished earlier attempt
  .venv/bin/python -B scripts/learning/bc_train.py --demos $DEMOS --human-t $HUMAN --records $RECS --init $student --label plan \
      --steps 20000 --save-every 10000 --out $out/train > $L/bc/dagger10p-round$k.log 2>&1
  [ -f $out/train/ema-step20000.pt ] || { echo "[loop10p] round $k training failed"; exit 1; }
  student=$out/train/ema-step20000.pt
done
tag=$(basename $(dirname $(dirname $student)))
for part in 0 10 20 30 40; do
  .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --start $part --scenes 10 --execute 8 --shapes $LETTERS \
      --student $student --out $R/dagger10p/final-$tag-x8-part$part.json > $L/bc/dagger10p-final-eval-$part.log 2>&1 &
done
wait
python3 -c "
import json, glob
s = {}
for f in glob.glob('$R/dagger10p/final-$tag-x8-part*.json'):
    for k2, v in json.load(open(f))['summary'].items():
        s.setdefault(k2, 0); s[k2] += v['success']
print('[loop10p] final student ($tag):', '  '.join(f'{a} {b}' for a, b in s.items()), ' total', sum(s.values()))"
echo "[loop10p $(date +%T)] done"
