#!/usr/bin/env bash
# D060: correction rounds driven by the flow-matching policy (D059), same recipe as the 26-letter rounds (D053): the
# teacher (wm2 v3) takes 30% of 8-control decisions on practice scenes (30 per round for the ten earlier shapes, 60 for
# the 17 new letters), then the policy is fine-tuned from its latest checkpoint on all data (the round-50 data set plus
# every flow round's records). Each round also checks the previous checkpoint on the 50 validation scenes per shape
# (monitoring only). New rounds start only while DEADLINE minus ROUND_HOURS + RESERVE_HOURS is still ahead; then the
# deciding test runs on the last completed round (chain_d060.sh).
# Usage: DEADLINE="2026-11-09 20:15" chain_flow_dagger.sh [first round] [last round]
#   training seeds 800000 + 20*i, i = 30000 + 70*k (unused range 30000-43999)
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U,A,B,C,G,H,I,K,M,N,O,Q,R,S,V,W,X,Z
first=${1:-1}; last=${2:-8}
[ -n "$DEADLINE" ] || { echo "set DEADLINE (e.g. '2026-11-09 20:15')"; exit 1; }
end=$(date -d "$DEADLINE" +%s)
R=$WORK/runs; L=$WORK/logs/flow; RUN=$R/flow26; mkdir -p $L
PY=${PY:-$REPO/.venv/bin/python}; [ -x "$PY" ] || PY=$WORK/repo/.venv/bin/python
STEPS=${STEPS:-20000}; NEWSCENES=${NEWSCENES:-60}; WAVE=${WAVE:-4}; MIN_FREE_GB=${MIN_FREE_GB:-10}
ROUND_HOURS=${ROUND_HOURS:-4}; RESERVE_HOURS=${RESERVE_HOURS:-2.5}
LETTERS="T T_narrow Y D L E F J P U A B C G H I K M N O Q R S V W X Z"
nsc() { case " T T_narrow Y D L E F J P U " in *" $1 "*) echo 30;; *) echo $NEWSCENES;; esac; }
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
W="--wm $R/wm2/v3/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
DEMOS="$R/teacher-plan-demos $R/teacher-plan-demos2 $R/teacher-plan-demos3 $R/teacher-plan-demos26"
HUMAN=$WORK/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
OLD="$R/mpc-data/dagger-bc1/data $(ls -d $R/mpc-data/dagger-r*/data $R/mpc-data/dagger10-r*/data $R/mpc-data/dagger10[pmh]-r*/data $R/lastmile/lastmile-r*/data $R/mpc-data/dagger26m-r*/data 2>/dev/null | tr '\n' ' ')"
export TDYSSEY_FRAMES_MMAP=${TDYSSEY_FRAMES_MMAP:-$WORK/mmap}; mkdir -p $TDYSSEY_FRAMES_MMAP
student=$RUN/v1/ema-step160000.pt
for k in $(seq 1 $((first - 1))); do [ -f $RUN/round$k/train/ema-step$STEPS.pt ] && student=$RUN/round$k/train/ema-step$STEPS.pt; done
for k in $(seq $first $last); do
  left=$(( end - $(date +%s) )); need=$(python3 -c "print(int(($ROUND_HOURS + $RESERVE_HOURS) * 3600))")
  if [ $left -lt $need ]; then echo "[flowdagger $(date +%T)] stop before round $k: $((left / 60)) min left"; break; fi
  out=$RUN/round$k; mkdir -p $out; echo "[flowdagger $(date +%T)] round $k from $student"
  for part in 0 13 26 39; do n=13; [ $part -eq 39 ] && n=11                    # monitoring: previous checkpoint
    $PY -B scripts/learning/base_rollouts.py eval --seed-base 400000 --start $part --scenes $n --execute 8 \
        --shapes $LETTERS --student $student --out $out/eval-in-x8-part$part.json > $L/round$k-eval-$part.log 2>&1 &
  done
  data=$R/mpc-data/flow26-r$k; coll=""
  for s in $LETTERS; do
    while [ $(for p in $coll; do kill -0 $p 2>/dev/null && echo x; done | wc -l) -ge $WAVE ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 15; done
    $PY -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes $(nsc $s) \
        --shape $s --start $((30000 + 70*k)) --state-noise 1 --angle-noise 1 --out $data > $L/round$k-collect-$s.log 2>&1 &
    coll="$coll $!"; sleep 2
  done
  wait $coll
  for s in $LETTERS; do                                                        # one retry for crashed letters
    if grep -q Traceback $L/round$k-collect-$s.log 2>/dev/null; then
      mv $L/round$k-collect-$s.log $L/round$k-collect-$s.crashed.log
      $PY -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes $(nsc $s) \
          --shape $s --start $((30000 + 70*k)) --state-noise 1 --angle-noise 1 --out $data > $L/round$k-collect-$s.log 2>&1
    fi
  done
  wait
  python3 -c "
import json, glob
s = sum(v['success'] for f in glob.glob('$out/eval-in-x8-part*.json') for v in json.load(open(f))['summary'].values())
print('[flowdagger] round $k: previous checkpoint solved', s, 'of 1350 validation scenes')"
  echo "[flowdagger $(date +%T)] round $k collected $(ls $data/data 2>/dev/null | wc -l) scenes"
  recs="$OLD $(ls -d $R/mpc-data/flow26-r*/data | tr '\n' ' ')"
  $PY -B scripts/learning/flow_train.py --init $student --demos $DEMOS --human-t $HUMAN --records $recs --label plan \
      --batch 128 --steps $STEPS --save-every 10000 --resume --out $out/train > $L/round$k-train.log 2>&1
  [ -f $out/train/ema-step$STEPS.pt ] || { echo "[flowdagger $(date +%T)] round $k training failed"; break; }
  student=$out/train/ema-step$STEPS.pt; echo "$student" > $RUN/last_round_checkpoint.txt
done
echo "[flowdagger $(date +%T)] rounds done; last checkpoint $student"
echo "$student" > $RUN/last_round_checkpoint.txt
