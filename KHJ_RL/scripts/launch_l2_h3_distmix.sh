#!/usr/bin/env bash
# H3: H2 동일 hyperparam + demo dir 3개 균등 mix.
#   - narrow 1000 NPZ (BC 분포)
#   - reachfix 1000 NPZ (stage 2 reach-fix track)
#   - stage2_wider 1000 NPZ (NEW, stage 2 wide spawn)
# bc-ckpt는 기존 narrow BC 그대로 (B1).

set -u
cd /home/j-k14d101/KHJ_RL

OUT_DIR="runs/g_plus_sacfd_L2_H3_distmix"
LOG="runs/g_plus_chain/step_l2_h3_full.log"

if [ -f "$OUT_DIR/sac.pt" ]; then
    echo "[h3] $OUT_DIR/sac.pt already exists — exit"
    exit 0
fi

mkdir -p runs/g_plus_chain

echo "[h3] launching SACfD H3 at $(date) -> $OUT_DIR"

env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt runs/g_plus_stage0_bc_narrow/bc.pt \
        --demo-dir \
            runs/demos/g_plus_stage0_narrow/stage0 \
            runs/demos/g_plus_stage2_reachfix/stage0 \
            runs/demos/g_plus_stage2_wider/stage0 \
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
echo "[h3] SACfD exited with rc=$rc at $(date)"
exit "$rc"
