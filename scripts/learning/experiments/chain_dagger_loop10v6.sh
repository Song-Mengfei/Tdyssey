#!/usr/bin/env bash
# Ten-letter correction rounds, one script for both lines of D048 (agreed 2026-11-01):
#   main line  TAG=m SIZE=96   (headline results; same camera as the official policy)
#   side line  TAG=h SIZE=192  (labelled side experiment: is the 96 px camera the limit?)
# Both lines start from the same 96 px student and follow the same schedule, so the main line is the side line's
# matched control:
#   round 0: retrain 30k steps on all teacher demos (intended-plan labels) + human T + all earlier corrections
#   round k>=1: test on 50 validation scenes per letter (2 parts, alongside), the student drives 30 new training scenes
#            per letter (teacher takes 30% of 8-control decisions, labels every visited state with its 8-control plan),
#            then fine-tune 20k steps on demos + human T + earlier corrections + this line's corrections.
# Data at 96 px (human T, earlier corrections) is enlarged for the 192 px line; its own corrections are native 192 px.
# v6 (2026-11-01 21:20): only one line collects at a time (flock on $WORK/collect.lock; two lines collecting
# together ran out of GPU memory at 20:24), and a letter whose collection crashed is collected again once.
# Usage: chain_dagger_loop10v6.sh TAG SIZE "DEMO DIRS" <start student> <first round> <last round>
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U
TAG=$1; SIZE=$2; DEMOS=$3; student=$4; first=$5; last=$6
case $TAG in m) IBASE=53000;; h) IBASE=57000;; *) echo "unknown line $TAG"; exit 1;; esac   # training seeds 800000 + 20*i, i < 60000
R=$WORK/runs; L=$WORK/logs; RUN=$R/dagger10$TAG; mkdir -p $L/bc $L/mpc-data $RUN
LETTERS="T T_narrow Y D L E F J P U"
HUMAN=$WORK/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
W="--wm $R/wm2/v2/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
OLD="$R/mpc-data/dagger-bc1/data $(ls -d $R/mpc-data/dagger-r*/data $R/mpc-data/dagger10-r*/data $R/mpc-data/dagger10p-r*/data $R/lastmile/lastmile-r*/data 2>/dev/null | tr '\n' ' ')"
train() {   # round, steps
  local out=$RUN/round$1
  local recs="$OLD $(ls -d $R/mpc-data/dagger10$TAG-r*/data 2>/dev/null | tr '\n' ' ')"
  [ -f $out/train/ema-step$2.pt ] || rm -rf $out/train
  local init="--init $student"; [ "$student" = official ] && init=""     # "official": start from the official checkpoint
  .venv/bin/python -B scripts/learning/bc_train.py --demos $DEMOS --human-t $HUMAN --records $recs $init --label plan \
      --image-size $SIZE --steps $2 --save-every 10000 --out $out/train > $L/bc/dagger10$TAG-round$1.log 2>&1
  [ -f $out/train/ema-step$2.pt ] || { echo "[loop10$TAG] round $1 training failed"; exit 1; }
  student=$out/train/ema-step$2.pt
}
for k in $(seq $first $last); do
  [ -f $WORK/stop_rounds6 ] && { echo "[loop10$TAG $(date +%T)] stop file found; no round $k"; break; }
  out=$RUN/round$k; mkdir -p $out
  if [ $k -eq 0 ]; then
    echo "[loop10$TAG $(date +%T)] round 0: retrain at $SIZE px from $student"
    if [ "$student" = official ]; then train 0 50000; else train 0 30000; fi; continue   # from scratch needs more steps
  fi
  echo "[loop10$TAG $(date +%T)] round $k: test + collect with $student"
  ev=""
  for part in 0 25; do
    .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --start $part --scenes 25 --execute 8 \
        --shapes $LETTERS --student $student --out $out/eval-in-x8-part$part.json > $L/bc/dagger10$TAG-round$k-eval-$part.log 2>&1 &
    ev="$ev $!"
  done
  data=$R/mpc-data/dagger10$TAG-r$k; coll=""
  exec 9>$WORK/collect.lock; flock 9                        # wait for the other line's collection
  for s in $LETTERS; do
    .venv/bin/python -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes 30 \
        --shape $s --start $((IBASE + 100*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/dagger10$TAG-r$k-$s.log 2>&1 &
    coll="$coll $!"; sleep 2
  done
  wait $coll
  for s in $LETTERS; do                                            # one retry for crashed letters
    if grep -q Traceback $L/mpc-data/dagger10$TAG-r$k-$s.log 2>/dev/null; then
      mv $L/mpc-data/dagger10$TAG-r$k-$s.log $L/mpc-data/dagger10$TAG-r$k-$s.crashed.log
      .venv/bin/python -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes 30 \
          --shape $s --start $((IBASE + 100*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/dagger10$TAG-r$k-$s.log 2>&1 &
      coll="$coll $!"
    fi
  done
  wait $coll; flock -u 9; exec 9>&-
  ( for p in $ev; do while kill -0 $p 2>/dev/null; do sleep 20; done; done
    python3 -c "
import json, glob
s = {}
for f in glob.glob('$out/eval-in-x8-part*.json'):
    for k2, v in json.load(open(f))['summary'].items():
        s.setdefault(k2, 0); s[k2] += v['success']
print('[loop10$TAG] round $k student (before training):', '  '.join(f'{a} {b}' for a, b in s.items()), ' total', sum(s.values()))" ) &
  echo "[loop10$TAG $(date +%T)] round $k collected $(ls $data/data 2>/dev/null | wc -l) scenes, $(grep -l Traceback $L/mpc-data/dagger10$TAG-r$k-* 2>/dev/null | wc -l) crashed"
  train $k 20000
done
wait
echo "[loop10$TAG $(date +%T)] done; last student $student"
