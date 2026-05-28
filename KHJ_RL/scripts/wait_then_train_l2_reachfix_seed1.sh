#!/usr/bin/env bash
# Multi-seed companion to wait_then_train_l2_reachfix.sh.
# Same BC + demos + hyperparams as seed=0 launch, but seed=1 + GPU 2.
# Purpose: measure seed-noise in the reach-fix SACfD result so a single-seed
# 50%-vs-65% delta can be disambiguated from a 60%-±-8% noise band.
#
# Prerequisites (identical to seed=0 launch):
#   runs/g_plus_sacfd_L2_full/sac.pt              (current L2 600k baseline done)
#   runs/g_plus_stage2_bc_reachfix/bc.pt          (new BC pretrain)
#   runs/demos/g_plus_stage2_reachfix/stage0/     (new demo dir, non-empty)

set -u
cd /home/j-k14d101/KHJ_RL

L2_DONE_MARKER="runs/g_plus_sacfd_L2_full/sac.pt"
NEW_BC="runs/g_plus_stage2_bc_reachfix/bc.pt"
NEW_DEMO_DIR="runs/demos/g_plus_stage2_reachfix/stage0"
OUT_DIR="runs/g_plus_sacfd_L2_reachfix_seed1"
LOG="runs/g_plus_chain/step_l2_reachfix_seed1_full.log"

mkdir -p runs/g_plus_chain

if [ -f "$OUT_DIR/sac.pt" ]; then
    echo "[wait_then_train_l2_reachfix_seed1] $OUT_DIR/sac.pt already exists — exit"
    exit 0
fi

echo "[wait_then_train_l2_reachfix_seed1] waiting for L2 600k done: $L2_DONE_MARKER"
echo "[wait_then_train_l2_reachfix_seed1] waiting for new BC:        $NEW_BC"
echo "[wait_then_train_l2_reachfix_seed1] waiting for new demos:     $NEW_DEMO_DIR"

while true; do
    if [ -f "$L2_DONE_MARKER" ] && [ -f "$NEW_BC" ]; then
        n=$(find "$NEW_DEMO_DIR" -maxdepth 1 -name "*.npz" 2>/dev/null | wc -l)
        if [ "$n" -ge 100 ]; then
            echo "[wait_then_train_l2_reachfix_seed1] all prereqs ready (demos=$n). Sleeping 60s grace then launching."
            sleep 60
            break
        fi
    fi
    sleep 60
done

echo "[wait_then_train_l2_reachfix_seed1] launching SACfD reach-fix seed=1 on GPU 2 at $(date)"

env CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt "$NEW_BC" \
        --demo-dir "$NEW_DEMO_DIR" \
        --run-name "$(basename "$OUT_DIR")" \
        --total-steps 600000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 3.0 \
        --bc-anneal-steps 2000000 \
        --her --her-k-future 4 \
        --her-ratio-initial 0.4 --her-ratio-final 0.3 --her-ratio-switch-step 100000 \
        --demo-ratio-initial 0.4 --demo-ratio-final 0.3 --demo-ratio-switch-step 100000 \
        --target-entropy -6.0 --init-alpha 0.1 --actor-lr 3e-5 \
        --no-entropy-term --no-q-filter \
        --critic-warmup-steps 3000 \
        --task-level 2 --curriculum-stage 2 \
        --video-every-steps 40000 \
        --skip-distribution-check \
        --seed 1 2>&1 | tee "$LOG"

rc=$?
echo "[wait_then_train_l2_reachfix_seed1] SACfD exited with rc=$rc at $(date)"
exit "$rc"
