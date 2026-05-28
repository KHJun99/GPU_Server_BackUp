#!/usr/bin/env bash
# H2 (A3 변형): baseline L2 full SACfD를 demo curriculum mix만 변경
#   - demo-dir에 stage 0 narrow + stage 2 reachfix 두 dir 모두 (2000 NPZ total)
#   - 다른 hyperparam은 baseline 동일 (bc-loss-weight 3.0, critic-warmup-steps 3000)
# GPU 2, seed=0. baseline BC ckpt (stage 0 narrow에 학습한 것) 그대로 사용.
#   → critic이 wider demo 분포에 노출되어 generalization 가능한지 검증.

set -u
cd /home/j-k14d101/KHJ_RL

OUT_DIR="runs/g_plus_sacfd_L2_H2_demomix"
LOG="runs/g_plus_chain/step_l2_h2_full.log"

if [ -f "$OUT_DIR/sac.pt" ]; then
    echo "[h2] $OUT_DIR/sac.pt already exists — exit"
    exit 0
fi

mkdir -p runs/g_plus_chain

echo "[h2] launching SACfD H2 at $(date) -> $OUT_DIR"

env CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
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
        --seed 0 2>&1 | tee "$LOG"

rc=$?
echo "[h2] SACfD exited with rc=$rc at $(date)"
exit "$rc"
