#!/usr/bin/env bash
# H1 (C1 변형): baseline L2 full SACfD를 hyperparam 2개만 변경
#   - bc-loss-weight 3.0 → 5.0  (BC anchor 더 강하게)
#   - critic-warmup-steps 3000 → 10000  (lift signal 잃지 않도록 critic을 더 안정시킴)
# GPU 0, seed=0. baseline g_plus_stage0_bc_narrow/bc.pt + demo는 stage 0 narrow 그대로.

set -u
cd /home/j-k14d101/KHJ_RL

OUT_DIR="runs/g_plus_sacfd_L2_H1_bcw5_warm10k"
LOG="runs/g_plus_chain/step_l2_h1_full.log"

if [ -f "$OUT_DIR/sac.pt" ]; then
    echo "[h1] $OUT_DIR/sac.pt already exists — exit"
    exit 0
fi

mkdir -p runs/g_plus_chain

echo "[h1] launching SACfD H1 at $(date) -> $OUT_DIR"

env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt runs/g_plus_stage0_bc_narrow/bc.pt \
        --demo-dir runs/demos/g_plus_stage0_narrow/stage0 \
        --run-name "$(basename "$OUT_DIR")" \
        --total-steps 600000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 5.0 \
        --bc-anneal-steps 2000000 \
        --her --her-k-future 4 \
        --her-ratio-initial 0.4 --her-ratio-final 0.3 --her-ratio-switch-step 100000 \
        --demo-ratio-initial 0.4 --demo-ratio-final 0.3 --demo-ratio-switch-step 100000 \
        --target-entropy -6.0 --init-alpha 0.1 --actor-lr 3e-5 \
        --no-entropy-term --no-q-filter \
        --critic-warmup-steps 10000 \
        --task-level 2 --curriculum-stage 2 \
        --video-every-steps 40000 \
        --skip-distribution-check \
        --seed 0 2>&1 | tee "$LOG"

rc=$?
echo "[h1] SACfD exited with rc=$rc at $(date)"
exit "$rc"
