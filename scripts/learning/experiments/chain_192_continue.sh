#!/usr/bin/env bash
# D055: the 26-letter 192 px line continues on an H100 job from its newest student, D053 recipe, through four unused
# training-seed ranges (60 scenes per round, all below i = 60000):
#   round 12 i = 52100 + 70*k | rounds 13-19 i = 48590 + 70*k | rounds 20-29 i = 53900 + 70*k | rounds 30-39 i = 54200 + 70*k
# Each segment resumes from the newest finished 192 px student. Frames are streamed to local disk (TDYSSEY_FRAMES_MMAP),
# which keeps host memory near 10-30 GB; unset it on a node with >= 400 GB free to load frames in memory instead.
# Stops early if $WORK/stop_rounds26 exists. Usage: chain_192_continue.sh [WAVE] [MIN_FREE_GB]
source "$(dirname "$0")/common.sh"
WV=${1:-9}; MF=${2:-10}                                                   # before set -- overwrites $1 $2
R=$WORK/runs; D="$R/teacher-plan-demos $R/teacher-plan-demos2 $R/teacher-plan-demos3 $R/teacher-plan-demos26"
export TDYSSEY_FRAMES_MMAP=${TDYSSEY_FRAMES_MMAP-$WORK/mmap}
newest() { ls -d $R/dagger26h/round*/train/ema-step54000.pt 2>/dev/null | sed 's|.*round\([0-9]*\)/train.*|\1|' | sort -n | tail -1; }
for seg in "12 12 52100" "13 19 48590" "20 29 53900" "30 39 54200"; do
  set -- $seg; first=$1; last=$2; ib=$3
  [ -f $WORK/stop_rounds26 ] && { echo "[192 $(date +%T)] stop file: done"; exit 0; }
  k=$(newest); [ -n "$k" ] || { echo "[192] no 192 px student found"; exit 1; }
  [ $((k+1)) -gt $last ] && continue                                       # this segment is already done
  start=$(( k+1 > first ? k+1 : first ))
  echo "[192 $(date +%T)] rounds $start-$last on seeds i = $ib + 70*k, from round $k"
  WAVE=$WV MIN_FREE_GB=$MF STEPS=54000 NEWSCENES=60 SPACING=70 \
    "$REPO/scripts/learning/experiments/chain_dagger_loop26c.sh" h 192 "$D" $R/dagger26h/round$k/train/ema-step54000.pt $start $last $ib \
    >> $WORK/logs/chain_dagger_loop26h.log 2>&1 || { echo "[192] segment $first-$last failed"; exit 1; }
done
echo "[192 $(date +%T)] all segments done"
