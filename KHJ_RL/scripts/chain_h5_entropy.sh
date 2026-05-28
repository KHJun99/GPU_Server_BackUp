#!/usr/bin/env bash
# H5 entropy: H2R ckpt에서 --no-entropy-term 제거하고 이어 학습.
# entropy regularization 재활성화로 exploration 확대 → lift plateau 돌파 시도.
# H2R 대비 유일한 차이: --no-entropy-term 제거.
# GPU 1. Close-hang sentinel inline.

set -u
cd /home/j-k14d101/KHJ_RL

TAG=h5e
GPU=1
RESUME_CKPT=runs/g_plus_sacfd_L2_H2_resume/sac.pt
OUT_DIR=runs/g_plus_sacfd_L2_H5_entropy
SAC_PT=$OUT_DIR/sac.pt

DEMO1=runs/demos/g_plus_stage0_narrow/stage0
DEMO2=runs/demos/g_plus_stage2_reachfix/stage0
DEMO3=runs/demos/g_plus_stage2_wider/stage0

SUMMARY=runs/g_plus_chain/h5e_chain_summary.log
SAC_LOG=runs/g_plus_chain/h5e_sac.log

mkdir -p runs/g_plus_chain

[ -f "$RESUME_CKPT" ] || { echo "[$TAG] ERROR ckpt not found: $RESUME_CKPT"; exit 1; }

echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] start $(date)" | tee -a "$SUMMARY"
echo "[$TAG] resume from: $RESUME_CKPT" | tee -a "$SUMMARY"
echo "[$TAG] change: --no-entropy-term REMOVED (entropy ON)" | tee -a "$SUMMARY"
echo "[$TAG] output: $OUT_DIR" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# ============================================================
# Phase 1: SACfD 600k (entropy ON)
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "[$TAG] Phase 1: SACfD start $(date)" | tee -a "$SUMMARY"

env CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --resume $RESUME_CKPT \
        --demo-dir $DEMO1 $DEMO2 $DEMO3 \
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
        --no-q-filter \
        --critic-warmup-steps 0 \
        --task-level 2 --curriculum-stage 2 \
        --video-every-steps 40000 \
        --skip-distribution-check \
        --seed 0 \
    > "$SAC_LOG" 2>&1 &
SAC_TEE=$!
echo "[$TAG] SAC launched (parent PID=$SAC_TEE)" | tee -a "$SUMMARY"

sleep 90
SAC_PY=""
for i in 1 2 3 4 5 6; do
  SAC_PY=$(pgrep -f "train_sac.py.*g_plus_sacfd_L2_H5_entropy" | head -1)
  [ -n "$SAC_PY" ] && break
  sleep 30
done
if [ -z "$SAC_PY" ]; then
  echo "[$TAG] ERROR SAC python PID not found — abort" | tee -a "$SUMMARY"
  exit 1
fi
echo "[$TAG] SAC python PID=$SAC_PY at $(date)" | tee -a "$SUMMARY"

while true; do
  if [ -f "$SAC_PT" ]; then
    echo "[$TAG] sac.pt detected $(date), 30s grace + SIGTERM" | tee -a "$SUMMARY"
    sleep 30
    if kill -0 $SAC_PY 2>/dev/null; then kill $SAC_PY || true; fi
    sleep 10
    if kill -0 $SAC_PY 2>/dev/null; then kill -9 $SAC_PY || true; fi
    sleep 20
    break
  fi
  if ! kill -0 $SAC_PY 2>/dev/null; then
    if [ -f "$SAC_PT" ]; then
      echo "[$TAG] SAC died but sac.pt present, ok" | tee -a "$SUMMARY"
      break
    else
      echo "[$TAG] ERROR SAC died with no sac.pt — abort" | tee -a "$SUMMARY"
      exit 1
    fi
  fi
  sleep 60
done
wait $SAC_TEE 2>/dev/null

[ -f "$SAC_PT" ] || { echo "[$TAG] ERROR sac.pt missing after training"; exit 1; }

# ============================================================
# Phase 2: multi-seed eval
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 2: multi-seed eval start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
bash scripts/chain_reval.sh $TAG $GPU "$SAC_PT" 2>&1 | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  SAC ckpt:  $SAC_PT" | tee -a "$SUMMARY"
echo "  SAC log:   $SAC_LOG" | tee -a "$SUMMARY"
echo "  eval logs: runs/g_plus_chain/eval_${TAG}_st{2,1,0}_s{1,2,0}.log" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
