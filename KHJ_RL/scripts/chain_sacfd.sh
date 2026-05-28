#!/bin/bash
# Chain script: SACfD smoke -> full training, with close-hang killer.
#
# Usage:
#   ./scripts/chain_sacfd.sh smoke   # 50k step smoke (~25 min)
#   ./scripts/chain_sacfd.sh full    # 500k step full + video sampler
#
# Both phases assume BC ckpt at runs/stage01_bc_1step_v1/bc.pt.
# Operates on GPU 0 (CUDA_VISIBLE_DEVICES=0) — GPU 1 reserved for teammate's
# lerobot training (CLAUDE.md GPU policy override, troubleshooting #20).
#
# Close-hang killer follows troubleshooting #17/#22 pattern: marker grep
# on log file + sleep 10s + pkill. New python steps added here MUST add
# their summary marker to the killer's grep alternation.

set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-smoke}"
BC_CKPT="${BC_CKPT:-runs/stage01_bc_1step_v1/bc.pt}"
DEMO_DIRS="runs/demos/stage0 runs/demos/stage0_extra runs/demos/stage1 runs/demos/stage1_extra runs/demos/stage1_extra2"

if [ ! -f "$BC_CKPT" ]; then
    echo "[chain_sacfd] BC ckpt not found: $BC_CKPT" >&2
    exit 1
fi

if [ "$PHASE" = "smoke" ]; then
    RUN_NAME="stage0_sacfd_smoke50k"
    TOTAL_STEPS=50000
    VIDEO_EVERY=0
elif [ "$PHASE" = "full" ]; then
    RUN_NAME="stage0_sacfd_full500k"
    TOTAL_STEPS=500000
    VIDEO_EVERY=10000
else
    echo "[chain_sacfd] unknown phase '$PHASE' (smoke|full)" >&2
    exit 1
fi

OUT_DIR="runs/$RUN_NAME"
LOG_FILE="$OUT_DIR/train.log"
mkdir -p "$OUT_DIR"

echo "[chain_sacfd] phase=$PHASE run=$RUN_NAME total_steps=$TOTAL_STEPS video_every=$VIDEO_EVERY"
echo "[chain_sacfd] bc_ckpt=$BC_CKPT"
echo "[chain_sacfd] log=$LOG_FILE"

# Launch trainer in background, then start the killer watcher.
nohup env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt "$BC_CKPT" \
        --demo-dir $DEMO_DIRS \
        --run-name "$RUN_NAME" \
        --total-steps "$TOTAL_STEPS" \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 1.0 \
        --bc-anneal-steps 200000 \
        --demo-ratio-initial 0.5 \
        --demo-ratio-final 0.25 \
        --demo-ratio-switch-step 50000 \
        --video-every-steps "$VIDEO_EVERY" \
        --seed 0 \
        > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "[chain_sacfd] train PID=$TRAIN_PID"
disown $TRAIN_PID

# Close-hang killer: when "[train_sac] saved" appears, give 10s then SIGTERM.
# Marker grep alternation includes Traceback/Error so a crashed process is
# also caught.
(
    while ! grep -qE "\[train_sac\] saved|Traceback|RuntimeError" "$LOG_FILE" 2>/dev/null; do
        if ! kill -0 $TRAIN_PID 2>/dev/null; then
            echo "[killer] train PID dead before marker — exiting"
            exit 0
        fi
        sleep 30
    done
    echo "[killer] marker seen, sleeping 10s before SIGTERM"
    sleep 10
    pkill -TERM -P $TRAIN_PID 2>/dev/null || true
    kill -TERM $TRAIN_PID 2>/dev/null || true
    sleep 5
    pkill -KILL -P $TRAIN_PID 2>/dev/null || true
    kill -KILL $TRAIN_PID 2>/dev/null || true
    echo "[killer] done"
) > "$OUT_DIR/killer.log" 2>&1 &
disown

echo "[chain_sacfd] launched. tail -f $LOG_FILE to follow."
