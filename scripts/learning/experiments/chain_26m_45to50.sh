#!/usr/bin/env bash
# 96 px 26-letter line, rounds 45-50 on an A100 job (lead's decision 2026-11-06: continue to round 50, then
# stop). Round 45's practice data was collected on the H100 (dagger26m-r45) before the D051 pick stopped it, so round 45
# starts at its training; rounds 46-50 run the D053 recipe (54k steps, 60 scenes per new letter) on seeds
# i = 48440 + 70*k (51660-51999). Frames are streamed to local disk (TDYSSEY_FRAMES_MMAP): loading them in memory
# (~145 GB for 2.96M samples) ran a 251 GB node out of memory on 2026-11-06 and the job was requeued.
source "$(dirname "$0")/common.sh"
export TDYSSEY_SHAPES=T,T_narrow,Y,D,L,E,F,J,P,U,A,B,C,G,H,I,K,M,N,O,Q,R,S,V,W,X,Z
export TDYSSEY_FRAMES_MMAP=$WORK/mmap
R=$WORK/runs; L=$WORK/logs; RUN=$R/dagger26m
DEMOS="$R/teacher-plan-demos $R/teacher-plan-demos2 $R/teacher-plan-demos3 $R/teacher-plan-demos26"
HUMAN=$WORK/data/pusht-official/pusht/pusht_cchi_v7_replay.zarr
OLD="$R/mpc-data/dagger-bc1/data $(ls -d $R/mpc-data/dagger-r*/data $R/mpc-data/dagger10-r*/data $R/mpc-data/dagger10[pmh]-r*/data $R/lastmile/lastmile-r*/data 2>/dev/null | tr '\n' ' ')"
recs="$OLD $(ls -d $R/mpc-data/dagger26m-r*/data 2>/dev/null | tr '\n' ' ')"
echo "[loop26m $(date +%T)] round 45: training (its practice data came from the H100), $(echo $recs | wc -w) correction folders"
rm -rf $RUN/round45/train
.venv/bin/python -B scripts/learning/bc_train.py --demos $DEMOS --human-t $HUMAN --records $recs --init $RUN/round44/train/ema-step54000.pt \
    --label plan --image-size 96 --steps 54000 --save-every 10000 --out $RUN/round45/train > $L/bc/dagger26m-round45.log 2>&1
[ -f $RUN/round45/train/ema-step54000.pt ] || { echo "[loop26m] round 45 training failed"; exit 1; }
WAVE=5 MIN_FREE_GB=12 STEPS=54000 NEWSCENES=60 SPACING=70 exec "$REPO/scripts/learning/experiments/chain_dagger_loop26c.sh" m 96 "$DEMOS" \
    $RUN/round45/train/ema-step54000.pt 46 50 48440
