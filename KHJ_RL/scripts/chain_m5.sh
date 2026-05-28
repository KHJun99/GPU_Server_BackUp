#!/usr/bin/env bash
# Method B M5 — BC pretrain → SACfD 학습 → multi-seed eval. 모두 자동 chain.
# 1) train_bc.py 1-step BC pretrain (GPU 0, sim 없음, ~10~30분)
# 2) train_sac.py SACfD 600k (GPU 0, ~11h)
# 3) chain_reval.sh m5 0 sac.pt (multi-seed eval ~1h)
# close-hang 자동 처리 inline.

set -u
cd /home/j-k14d101/KHJ_RL

DEMO_DIR=runs/demos/method_b_stage0/stage0
BC_RUN=runs/method_b_stage0_bc
BC_PT=$BC_RUN/bc.pt
SAC_RUN=runs/method_b_l2_sacfd
SAC_PT=$SAC_RUN/sac.pt
SUMMARY=runs/g_plus_chain/m5_chain_summary.log
BC_LOG=runs/g_plus_chain/m5_bc.log
SAC_LOG=runs/g_plus_chain/m5_sac.log

mkdir -p runs/g_plus_chain
echo "[m5] start $(date)" | tee -a "$SUMMARY"
echo "[m5] demo: $DEMO_DIR" | tee -a "$SUMMARY"
echo "[m5] BC:   $BC_RUN" | tee -a "$SUMMARY"
echo "[m5] SAC:  $SAC_RUN" | tee -a "$SUMMARY"

# ============================================================
# Phase 1) BC pretrain (1-step BC, no chunking, SACfD 호환)
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[m5] Phase 1: BC pretrain start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_bc.py \
        --demo-dir $DEMO_DIR \
        --no-chunking \
        --run-name "$(basename $BC_RUN)" \
        --epochs 50 --batch-size 256 --lr 3e-4 \
        --actor-logstd-init -1.0 \
        --seed 0 \
    > "$BC_LOG" 2>&1
BC_RC=$?
echo "[m5] Phase 1 done rc=$BC_RC $(date)" | tee -a "$SUMMARY"
if [ ! -f "$BC_PT" ]; then
  echo "[m5] ERROR bc.pt not created — abort" | tee -a "$SUMMARY"
  exit 1
fi
echo "[m5] bc.pt: $(ls -la $BC_PT)" | tee -a "$SUMMARY"

# ============================================================
# Phase 2) SACfD 600k 학습
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[m5] Phase 2: SACfD start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt $BC_PT \
        --demo-dir $DEMO_DIR \
        --run-name "$(basename $SAC_RUN)" \
        --total-steps 600000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 3.0 \
        --bc-anneal-steps 2000000 \
        --her --her-k-future 4 \
        --her-ratio-initial 0.4 --her-ratio-final 0.3 --her-ratio-switch-step 100000 \
        --demo-ratio-initial 0.4 --demo-ratio-final 0.3 --demo-ratio-switch-step 100000 \
        --target-entropy -4.0 --init-alpha 0.1 --actor-lr 3e-5 \
        --no-entropy-term --no-q-filter \
        --critic-warmup-steps 3000 \
        --task-level 2 --curriculum-stage 2 \
        --video-every-steps 40000 \
        --skip-distribution-check \
        --seed 0 \
    > "$SAC_LOG" 2>&1 &
SAC_TEE=$!
echo "[m5] SAC launched (parent PID=$SAC_TEE)" | tee -a "$SUMMARY"

sleep 90
SAC_PY=""
for i in 1 2 3 4 5 6; do
  SAC_PY=$(pgrep -f "train_sac.py.*method_b_l2_sacfd" | head -1)
  [ -n "$SAC_PY" ] && break
  sleep 30
done
if [ -z "$SAC_PY" ]; then
  echo "[m5] ERROR SAC python PID not found — abort" | tee -a "$SUMMARY"
  exit 1
fi
echo "[m5] SAC python PID=$SAC_PY at $(date)" | tee -a "$SUMMARY"

# sac.pt 검출 대기 + close-hang
while true; do
  if [ -f "$SAC_PT" ]; then
    echo "[m5] sac.pt detected $(date), 30s grace + SIGTERM" | tee -a "$SUMMARY"
    sleep 30
    if kill -0 $SAC_PY 2>/dev/null; then kill $SAC_PY || true; fi
    sleep 10
    if kill -0 $SAC_PY 2>/dev/null; then kill -9 $SAC_PY || true; fi
    sleep 20
    break
  fi
  if ! kill -0 $SAC_PY 2>/dev/null; then
    if [ -f "$SAC_PT" ]; then
      echo "[m5] SAC died but sac.pt present, ok" | tee -a "$SUMMARY"
      break
    else
      echo "[m5] ERROR SAC died with no sac.pt — abort" | tee -a "$SUMMARY"
      exit 1
    fi
  fi
  sleep 60
done
wait $SAC_TEE 2>/dev/null

# ============================================================
# Phase 3) multi-seed eval (chain_reval)
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[m5] Phase 3: multi-seed eval start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
bash scripts/chain_reval.sh m5 0 "$SAC_PT" 2>&1 | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[m5] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  BC ckpt:    $BC_PT" | tee -a "$SUMMARY"
echo "  SAC ckpt:   $SAC_PT" | tee -a "$SUMMARY"
echo "  BC log:     $BC_LOG" | tee -a "$SUMMARY"
echo "  SAC log:    $SAC_LOG" | tee -a "$SUMMARY"
echo "  eval logs:  runs/g_plus_chain/eval_m5_st{2,1,0}_s{1,2,0}.log" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
