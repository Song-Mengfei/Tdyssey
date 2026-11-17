#!/usr/bin/env bash
# Plan-label ten-letter correction rounds (agreed 2026-11-01 after the label A/B): start from the best ten-letter
# student (round 16), demos = plan-recording teacher demos with intended-plan labels, + all correction records (already
# plan-labelled) + human T. Rounds until about 2 hours before the job ends (the watchdog adds more if idle).
source "$(dirname "$0")/common.sh"
while kill -0 3968591 2>/dev/null || kill -0 3969585 2>/dev/null; do sleep 30; done
echo "[planlabel $(date +%T)] plan demos ready: $(ls $WORK/runs/teacher-plan-demos2/*.npz | wc -l) files"
"$REPO/scripts/learning/experiments/chain_dagger_loop10v3.sh" $WORK/runs/dagger10/round16/train/ema-step20000.pt 1 14 > $WORK/logs/chain_dagger_loop10v3.log 2>&1
echo "[planlabel $(date +%T)] done"
