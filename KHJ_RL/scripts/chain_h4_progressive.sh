#!/usr/bin/env bash
# H4 progressive: stage regression → expansion.
#   Phase 1: BC ckpt → SACfD stage 1 (5cm), task_level 2, 400k steps
#   Phase 2: stage 1 ckpt → SACfD stage 2 (10cm), task_level 2, 600k steps
#   Phase 3: multi-seed eval (chain_reval.sh h4p)
# GPU 1. Close-hang sentinel inline.

set -u
cd /home/j-k14d101/KHJ_RL

TAG=h4p
GPU=1
BC_CKPT=runs/g_plus_stage0_bc_narrow/bc.pt

STAGE1_DIR=runs/g_plus_sacfd_L2_H4_stage1
STAGE1_PT=$STAGE1_DIR/sac.pt

STAGE2_DIR=runs/g_plus_sacfd_L2_H4_stage2
STAGE2_PT=$STAGE2_DIR/sac.pt

DEMO_NARROW=runs/demos/g_plus_stage0_narrow/stage0
DEMO_REACHFIX=runs/demos/g_plus_stage2_reachfix/stage0
DEMO_WIDER=runs/demos/g_plus_stage2_wider/stage0

SUMMARY=runs/g_plus_chain/h4p_chain_summary.log
STAGE1_LOG=runs/g_plus_chain/h4p_stage1_sac.log
STAGE2_LOG=runs/g_plus_chain/h4p_stage2_sac.log

mkdir -p runs/g_plus_chain

[ -f "$BC_CKPT" ] || { echo "[$TAG] ERROR BC_CKPT not found: $BC_CKPT"; exit 1; }

echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] start $(date)" | tee -a "$SUMMARY"
echo "[$TAG] BC ckpt:  $BC_CKPT" | tee -a "$SUMMARY"
echo "[$TAG] Phase 1:  stage 1 (5cm) 400k -> $STAGE1_DIR" | tee -a "$SUMMARY"
echo "[$TAG] Phase 2:  stage 2 (10cm) 600k -> $STAGE2_DIR" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# ============================================================
# Helper: run SACfD with close-hang sentinel
# Args: $1=log_file $2=output_dir $3=sac_pt_path $4=pgrep_pattern $5..=extra train_sac args
# ============================================================
run_sacfd_with_sentinel() {
  local LOG="$1"; shift
  local OUT="$1"; shift
  local PT="$1"; shift
  local PATTERN="$1"; shift

  env CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
      python scripts/train_sac.py "$@" > "$LOG" 2>&1 &
  local TEE_PID=$!
  echo "[$TAG]   launched (parent PID=$TEE_PID)" | tee -a "$SUMMARY"

  sleep 90
  local PY_PID=""
  for i in 1 2 3 4 5 6; do
    PY_PID=$(pgrep -f "$PATTERN" | head -1)
    [ -n "$PY_PID" ] && break
    sleep 30
  done
  if [ -z "$PY_PID" ]; then
    echo "[$TAG]   ERROR python PID not found — abort" | tee -a "$SUMMARY"
    exit 1
  fi
  echo "[$TAG]   python PID=$PY_PID at $(date)" | tee -a "$SUMMARY"

  while true; do
    if [ -f "$PT" ]; then
      echo "[$TAG]   sac.pt detected $(date), 30s grace + SIGTERM" | tee -a "$SUMMARY"
      sleep 30
      if kill -0 $PY_PID 2>/dev/null; then kill $PY_PID || true; fi
      sleep 10
      if kill -0 $PY_PID 2>/dev/null; then kill -9 $PY_PID || true; fi
      sleep 20
      break
    fi
    if ! kill -0 $PY_PID 2>/dev/null; then
      if [ -f "$PT" ]; then
        echo "[$TAG]   died but sac.pt present, ok" | tee -a "$SUMMARY"
        break
      else
        echo "[$TAG]   ERROR died with no sac.pt — abort" | tee -a "$SUMMARY"
        exit 1
      fi
    fi
    sleep 60
  done
  wait $TEE_PID 2>/dev/null
}

# ============================================================
# Phase 1: SACfD stage 1 (5cm), from BC ckpt, 400k steps
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "[$TAG] Phase 1: SACfD stage 1 (5cm) start $(date)" | tee -a "$SUMMARY"

run_sacfd_with_sentinel "$STAGE1_LOG" "$STAGE1_DIR" "$STAGE1_PT" "train_sac.py.*g_plus_sacfd_L2_H4_stage1" \
    --bc-ckpt "$BC_CKPT" \
    --demo-dir "$DEMO_NARROW" \
    --run-name "$(basename "$STAGE1_DIR")" \
    --total-steps 400000 \
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
    --task-level 2 --curriculum-stage 1 \
    --video-every-steps 40000 \
    --skip-distribution-check \
    --seed 0

echo "[$TAG] Phase 1 done $(date)" | tee -a "$SUMMARY"
[ -f "$STAGE1_PT" ] || { echo "[$TAG] ERROR STAGE1 ckpt missing, abort"; exit 1; }

# ============================================================
# Phase 2: SACfD stage 2 (10cm), resume from stage 1, 600k steps
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 2: SACfD stage 2 (10cm) resume start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

run_sacfd_with_sentinel "$STAGE2_LOG" "$STAGE2_DIR" "$STAGE2_PT" "train_sac.py.*g_plus_sacfd_L2_H4_stage2" \
    --resume "$STAGE1_PT" \
    --demo-dir "$DEMO_NARROW" "$DEMO_REACHFIX" "$DEMO_WIDER" \
    --run-name "$(basename "$STAGE2_DIR")" \
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
    --critic-warmup-steps 0 \
    --task-level 2 --curriculum-stage 2 \
    --video-every-steps 40000 \
    --skip-distribution-check \
    --seed 0

echo "[$TAG] Phase 2 done $(date)" | tee -a "$SUMMARY"
[ -f "$STAGE2_PT" ] || { echo "[$TAG] ERROR STAGE2 ckpt missing after Phase 2"; exit 1; }

# ============================================================
# Phase 3: multi-seed eval
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 3: multi-seed eval start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
bash scripts/chain_reval.sh $TAG $GPU "$STAGE2_PT" 2>&1 | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  Stage 1 ckpt: $STAGE1_PT" | tee -a "$SUMMARY"
echo "  Stage 2 ckpt: $STAGE2_PT" | tee -a "$SUMMARY"
echo "  Stage 1 log:  $STAGE1_LOG" | tee -a "$SUMMARY"
echo "  Stage 2 log:  $STAGE2_LOG" | tee -a "$SUMMARY"
echo "  eval logs:    runs/g_plus_chain/eval_${TAG}_st{2,1,0}_s{1,2,0}.log" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
