#!/usr/bin/env bash
# D066 goal overlay: one arm's correction rounds, continuing the 96 px round-50 student with the D053 recipe
# (chain_dagger_loop26c.sh: teacher wm2 v3 takes 30% of 8-control decisions, intended-plan labels, 54k steps per round).
#   ARM=overlay  every camera image (teacher demos, human T, new practice records, validation, test) carries the goal
#                overlay with radius OVERLAY (default 9 simulator px); the 116 earlier correction folders have no block
#                poses and keep their official frames (--mixed-records), unless OLD_RECORDS=drop.
#   ARM=control  the same rounds, seeds, steps and data rules without the overlay.
# Round 0 adapts the start student to its arm's data (54k steps, no new practice data, for both arms). Rounds k >= 1:
# practice collection by the current student (30 scenes for the ten earlier shapes, 60 for the 17 new letters; seeds
# 800000 + 20*i, i = IBASE + SPACING*k, the same scenes for both arms; only one arm collects at a time: flock) + 54k
# steps on all data. Every round's student (0..last) is validated in the background as soon as it exists, while the
# next round goes on: the 27 trained shapes' selection scenes (seeds 400000 + 20*i, i < 50, 8 executed targets, the
# same scenes and policy-noise seed for both arms, so paired by scene); a progress curve, monitoring only, never used to
# pick a checkpoint (VALIDATE=0 turns it off). The arm ends after its last validation has finished.
# No round starts after ROUND_DEADLINE (epoch seconds) or when $WORK/stop_d066 exists.
# Usage: ARM=overlay|control chain_d066.sh <start student> <first round> <last round>
#   records: $WORK/runs/mpc-data/d066-$ARM-r<k>/data; students: $WORK/runs/d066/$ARM/round<k>/train/ema-step54000.pt;
#   validation: $WORK/runs/d066/$ARM/round<k>/val-part{0,13,26,39}.json
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U,A,B,C,G,H,I,K,M,N,O,Q,R,S,V,W,X,Z
student=$1; first=$2; last=$3
[ -n "$last" ] && [ -f "$student" ] || { echo "usage: ARM=overlay|control chain_d066.sh <start student> <first> <last>"; exit 1; }
case "${ARM:-}" in overlay) OV=${OVERLAY:-9};; control) OV=0;; *) echo "ARM must be overlay or control"; exit 1;; esac
R=$WORK/runs; L=$WORK/logs; RUN=$R/d066/$ARM; mkdir -p $L/bc $L/mpc-data $L/base-rollouts $RUN
WAVE=${WAVE:-8}; MIN_FREE_GB=${MIN_FREE_GB:-12}; STEPS=${STEPS:-54000}; NEWSCENES=${NEWSCENES:-60}
IBASE=${IBASE:-32000}; SPACING=${SPACING:-70}; ROUND_DEADLINE=${ROUND_DEADLINE:-9999999999}; OLD_RECORDS=${OLD_RECORDS:-keep}
PY=${PY:-.venv/bin/python}; VALIDATE=${VALIDATE:-1}
nsc() { case " T T_narrow Y D L E F J P U " in *" $1 "*) echo 30;; *) echo $NEWSCENES;; esac; }
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{print int($1/1024)}'; }
log() { echo "[d066-$ARM $(date '+%F %T')] $*"; }
LETTERS="T T_narrow Y D L E F J P U A B C G H I K M N O Q R S V W X Z"
DEMOS="$R/teacher-plan-demos $R/teacher-plan-demos2 $R/teacher-plan-demos3 $R/teacher-plan-demos26"
HUMAN=$WORK/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
W="--wm $R/wm2/v3/model.pt --select best --gate-margin 0 --pushes 128 --horizon 48 --knots 8 --approach 0 --kp area --tight 2"
OLD="$R/mpc-data/dagger-bc1/data $(ls -d $R/mpc-data/dagger-r*/data $R/mpc-data/dagger10-r*/data $R/mpc-data/dagger10[pmh]-r*/data $R/lastmile/lastmile-r*/data 2>/dev/null | tr '\n' ' ')"
OLD="$OLD $(ls -d $R/mpc-data/dagger26m-r*/data 2>/dev/null | tr '\n' ' ')"      # round 50's data set: 116 folders
[ "$OLD_RECORDS" = drop ] && OLD=""
CAM=""; [ $OV -gt 0 ] && CAM="--goal-overlay $OV"; [ $OV -gt 0 ] && [ -n "$OLD" ] && CAM="$CAM --mixed-records"
log "start $student, rounds $first-$last, overlay $OV, earlier correction folders $(echo $OLD | wc -w) ($OLD_RECORDS)"

train() {   # round
  local out=$RUN/round$1 recs="$OLD $(ls -d $R/mpc-data/d066-$ARM-r*/data 2>/dev/null | tr '\n' ' ')"
  [ -f $out/train/ema-step$STEPS.pt ] && { student=$out/train/ema-step$STEPS.pt; return; }    # resumed chain
  rm -rf $out/train; mkdir -p $out
  $PY -B scripts/learning/bc_train.py --demos $DEMOS --human-t $HUMAN --records $recs --init $student --label plan \
      --image-size 96 $CAM --steps $STEPS --save-every $STEPS --out $out/train > $L/bc/d066-$ARM-round$1.log 2>&1
  [ -f $out/train/ema-step$STEPS.pt ] || { log "round $1 training failed (see $L/bc/d066-$ARM-round$1.log)"; exit 1; }
  student=$out/train/ema-step$STEPS.pt
}

validate() {   # round, student: background; waits for free GPU memory per part, never blocks the next round
  local out=$RUN/round$1 k=$1 c=$2
  [ "$VALIDATE" = 1 ] || return 0
  [ $(ls $out/val-part*.json 2>/dev/null | wc -l) -eq 4 ] && return 0                     # resumed chain
  ( ev=""
    for part in 0 13 26 39; do
      n=13; [ $part -eq 39 ] && n=11
      until [ "$(free_gb)" -ge $MIN_FREE_GB ]; do sleep 15; done
      $PY -B scripts/learning/base_rollouts.py eval --seed-base 400000 --start $part --scenes $n --execute 8 --shapes $LETTERS \
          --student $c --out $out/val-part$part.json > $L/base-rollouts/d066-$ARM-round$k-val-$part.log 2>&1 &
      ev="$ev $!"; sleep 5
    done
    wait $ev
    python3 -c "
import json, glob
s = {}
for f in glob.glob('$out/val-part*.json'):
    for k2, v in json.load(open(f))['summary'].items():
        s[k2] = s.get(k2, 0) + v['success']
print('[d066-$ARM] round $k student validation:', '  '.join(f'{a} {b}' for a, b in s.items()), ' total', sum(s.values()), flush=True)" ) &
}

for k in $(seq $first $last); do
  [ -f $WORK/stop_d066 ] && { log "stop file found; no round $k"; break; }
  [ $(date +%s) -ge $ROUND_DEADLINE ] && { log "round deadline passed; no round $k"; break; }
  out=$RUN/round$k; mkdir -p $out
  if [ $k -eq 0 ]; then
    log "round 0: adapt to the arm's data ($STEPS steps)"; train 0; log "round 0 done"; validate 0 $student; continue
  fi
  log "round $k: practice with $student"
  data=$R/mpc-data/d066-$ARM-r$k
  if [ ! -f $data/COMPLETE ]; then
    rm -rf $data; coll=""
    exec 9>$WORK/collect.lock; flock 9                       # one arm collects at a time (GPU memory)
    for s in $LETTERS; do
      while [ $(for p in $coll; do kill -0 $p 2>/dev/null && echo x; done | wc -l) -ge $WAVE ] || [ "$(free_gb)" -lt $MIN_FREE_GB ]; do sleep 15; done
      $PY -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes $(nsc $s) \
          --shape $s --start $((IBASE + SPACING*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/d066-$ARM-r$k-$s.log 2>&1 &
      coll="$coll $!"; sleep 2
    done
    wait $coll
    for s in $LETTERS; do                                      # one retry for crashed letters
      if grep -q Traceback $L/mpc-data/d066-$ARM-r$k-$s.log 2>/dev/null; then
        mv $L/mpc-data/d066-$ARM-r$k-$s.log $L/mpc-data/d066-$ARM-r$k-$s.crashed.log
        $PY -B scripts/learning/mpc_dagger.py $W --execute 8 --every 8 --beta 0.3 --student $student --scenes $(nsc $s) \
            --shape $s --start $((IBASE + SPACING*k)) --state-noise 1 --angle-noise 1 --out $data > $L/mpc-data/d066-$ARM-r$k-$s.log 2>&1
      fi
    done
    flock -u 9; exec 9>&-
    n=$(ls $data/data 2>/dev/null | wc -l)
    [ $(grep -l Traceback $L/mpc-data/d066-$ARM-r$k-*.log 2>/dev/null | grep -vc crashed) -eq 0 ] || { log "round $k: a letter crashed twice"; exit 1; }
    echo "$n scene files, $(date '+%F %T')" > $data/COMPLETE
  fi
  log "round $k collected $(ls $data/data | wc -l) scenes"
  train $k
  log "round $k done: $student"
  validate $k $student
done
echo "$student" > $RUN/LAST_STUDENT
log "last student $student; waiting for the remaining validations"
wait
log "done"
