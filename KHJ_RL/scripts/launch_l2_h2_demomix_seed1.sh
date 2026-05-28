#!/usr/bin/env bash
# H2 seed=1: H2 동일 config + GPU 1 + seed=1 (재현성 검증용).
# launch_l2_h2_demomix.sh와 완전히 동일하되 seed/GPU/run-name/log만 변경.

set -u
cd /home/j-k14d101/KHJ_RL

OUT_DIR="runs/g_plus_sacfd_L2_H2_demomix_seed1"
LOG="runs/g_plus_chain/step_l2_h2_seed1_full.log"

if [ -f "$OUT_DIR/sac.pt" ]; then
    echo "[h2-seed1] $OUT_DIR/sac.pt already exists — exit"
    exit 0
fi

mkdir -p runs/g_plus_chain

echo "[h2-seed1] launching SACfD H2 seed=1 at $(date) -> $OUT_DIR"

env CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt runs/g_plus_stage0_bc_narrow/bc.pt \
        --demo-dir runs/demos/g_plus_stage0_narrow/stage0 runs/demos/g_plus_stage2_reachfix/stage0 \
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
echo "[h2-seed1] SACfD exited with rc=$rc at $(date)"
exit "$rc"
