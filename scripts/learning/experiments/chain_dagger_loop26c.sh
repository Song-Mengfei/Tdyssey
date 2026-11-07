#!/usr/bin/env bash
# 26-letter correction rounds (D051): chain_dagger_loop10v6.sh generalised to the 27 trained shapes (26 capitals +
# near-T) with world model wm2 v3. Same schedule: round 0 retrains on all data (30k steps; 50k from "official"), then
# rounds: test on 50 validation scenes per shape (2 parts, alongside), collect 30 training scenes per shape with the
# teacher taking 30% of 8-control decisions (only one line collects at a time: flock; collection in waves of at most
# 9 shapes, ~4.9 GB GPU memory each; crashed shapes collected once more), fine-tune 20k steps on everything.
# v26c (2026-11-03, D053 budget): STEPS training steps per round (default 54000), NEWSCENES practice scenes per round
# for the 17 new letters (default 60; the ten earlier shapes keep 30), seeds i = IBASE + SPACING*k (default 70).
# Set TDYSSEY_FRAMES_MMAP=<local dir> where host memory is short (bc_train then streams its frames to disk).
# From v26b (2026-11-02, for GPUs smaller than the 93 GB H100, e.g. an A100): collection waves of WAVE jobs (default 4)
# and each job starts only with MIN_FREE_GB free (default 10); evaluations use 4 parts of 13 scenes.
# Usage: chain_dagger_loop26c.sh TAG SIZE "DEMO DIRS" <start student|official> <first round> <last round> <IBASE>
#   TAG m (96 px) or h (192 px); training seeds 800000 + 20*i, i = IBASE + 50*k (D051: m 58000, h 52100; keep i < 60000)
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U,A,B,C,G,H,I,K,M,N,O,Q,R,S,V,W,X,Z
TAG=$1; SIZE=$2; DEMOS=$3; student=$4; first=$5; last=$6; IBASE=$7
[ -n "$IBASE" ] || { echo "usage: TAG SIZE DEMOS student first last IBASE"; exit 1; }
R=$WORK/runs; L=$WORK/logs; RUN=$R/dagger26$TAG; mkdir -p $L/bc $L/mpc-data $RUN
WAVE=${WAVE:-4}; MIN_FREE_GB=${MIN_FREE_GB:-10}; STEPS=${STEPS:-54000}; NEWSCENES=${NEWSCENES:-60}; SPACING=${SPACING:-70}
nsc() { case " T T_narrow Y D L E F J P U " in *" $1 "*) echo 30;; *) echo $NEWSCENES;; esac; }
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
LETTERS="T T_narrow Y D L E F J P U A B C G H I K M N O Q R S V W X Z"
HUMAN=$WORK/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
W="--wm $R/wm2/v3/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
OLD="$R/mpc-data/dagger-bc1/data $(ls -d $R/mpc-data/dagger-r*/data $R/mpc-data/dagger10-r*/data $R/mpc-data/dagger10[pmh]-r*/data $R/lastmile/lastmile-r*/data 2>/dev/null | tr '\n' ' ')"
train() {   # round, steps
  local out=$RUN/round$1
  local recs="$OLD $(ls -d $R/mpc-data/dagger26$TAG-r*/data 2>/dev/null | tr '\n' ' ')"
  [ -f $out/train/ema-step$2.pt ] || rm -rf $out/train
  local init="--init $student"; [ "$student" = official ] && init=""     # "official": start from the official checkpoint
  .venv/bin/python -B scripts/learning/bc_train.py --demos $DEMOS --human-t $HUMAN --records $recs $init --label plan \
      --image-size $SIZE --steps $2 --save-every 10000 --out $out/train > $L/bc/dagger26$TAG-round$1.log 2>&1
  [ -f $out/train/ema-step$2.pt ] || { echo "[loop26$TAG] round $1 training failed"; exit 1; }
  student=$out/train/ema-step$2.pt
}
for k in $(seq $first $last); do
  [ -f $WORK/stop_rounds26 ] && { echo "[loop26$TAG $(date +%T)] stop file found; no round $k"; break; }
  out=$RUN/round$k; mkdir -p $out
  if [ $k -eq 0 ]; then
    echo "[loop26$TAG $(date +%T)] round 0: retrain at $SIZE px from $student"
    if [ "$student" = official ]; then train 0 50000; else train 0 30000; fi; continue   # from scratch needs more steps
  fi
  echo "[loop26$TAG $(date +%T)] round $k: test + collect with $student"
  ev=""
  for part in 0 13 26 39; do
    n=13; [ $part -eq 39 ] && n=11
    until [ "$(free_gb)" -ge $MIN_FREE_GB ]; do sleep 15; done
    .venv/bin/python -B scripts/learning/base_rollouts.py eval --seed-base 400000 --start $part --scenes $n --execute 8 \
        --shapes $LETTERS --student $student --out $out/eval-in-x8-part$part.json > $L/bc/dagger26$TAG-round$k-eval-$part.log 2>&1 &
    ev="$ev $!"
  done
  data=$R/mpc-data/dagger26$TAG-r$k; coll=""
  exec 9>$WORK/collect.lock; flock 9                        # wait for the other line's collection
  for s in $LETTERS; do
    while [ $(for p in $coll; do kill -0 $p 2>/dev/null && echo x; done | wc -l) -ge $WAVE ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 15; done   # waves of <= 9
    .venv/bin/python -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes $(nsc $s) \
        --shape $s --start $((IBASE + SPACING*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/dagger26$TAG-r$k-$s.log 2>&1 &
    coll="$coll $!"; sleep 2
  done
  wait $coll
  for s in $LETTERS; do                                            # one retry for crashed letters
    if grep -q Traceback $L/mpc-data/dagger26$TAG-r$k-$s.log 2>/dev/null; then
      mv $L/mpc-data/dagger26$TAG-r$k-$s.log $L/mpc-data/dagger26$TAG-r$k-$s.crashed.log
      .venv/bin/python -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes $(nsc $s) \
          --shape $s --start $((IBASE + SPACING*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/dagger26$TAG-r$k-$s.log 2>&1 &
      coll="$coll $!"
      while [ $(for p in $coll; do kill -0 $p 2>/dev/null && echo x; done | wc -l) -ge $WAVE ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 15; done
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
print('[loop26$TAG] round $k student (before training):', '  '.join(f'{a} {b}' for a, b in s.items()), ' total', sum(s.values()))" ) &
  echo "[loop26$TAG $(date +%T)] round $k collected $(ls $data/data 2>/dev/null | wc -l) scenes, $(grep -l Traceback $L/mpc-data/dagger26$TAG-r$k-* 2>/dev/null | wc -l) crashed"
  train $k $STEPS
done
wait
echo "[loop26$TAG $(date +%T)] done; last student $student"
