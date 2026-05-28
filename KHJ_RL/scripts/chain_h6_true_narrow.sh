#!/usr/bin/env bash
# H6 true-narrow: 진짜 narrow (2cm, curriculum-stage 0) demo로 처음부터 재구축.
#   Phase 1: collect 1000 narrow demos (2cm side_length)
#   Phase 2: BC pretrain on narrow demos
#   Phase 3: SACfD stage 1 (5cm) 400k
#   Phase 4: SACfD stage 2 (10cm) 600k (resume from stage 1)
#   Phase 5: multi-seed eval
# GPU 1. Close-hang sentinel inline.

set -u
cd /home/j-k14d101/KHJ_RL

TAG=h6
GPU=1

DEMO_DIR=runs/demos/h6_true_narrow/stage0
BC_DIR=runs/h6_bc_narrow
BC_PT=$BC_DIR/bc.pt

STAGE1_DIR=runs/g_plus_sacfd_L2_H6_stage1
STAGE1_PT=$STAGE1_DIR/sac.pt

STAGE2_DIR=runs/g_plus_sacfd_L2_H6_stage2
STAGE2_PT=$STAGE2_DIR/sac.pt

SUMMARY=runs/g_plus_chain/h6_chain_summary.log
COLLECT_LOG=runs/g_plus_chain/h6_collect.log
BC_LOG=runs/g_plus_chain/h6_bc.log
STAGE1_LOG=runs/g_plus_chain/h6_stage1_sac.log
STAGE2_LOG=runs/g_plus_chain/h6_stage2_sac.log

mkdir -p runs/g_plus_chain

echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] start $(date)" | tee -a "$SUMMARY"
echo "[$TAG] TRUE narrow 2cm demo → BC → progressive SACfD" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# ============================================================
# Helper: run SACfD with close-hang sentinel
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
# Phase 1: Collect 1000 TRUE narrow (2cm) demos
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "[$TAG] Phase 1: collect 1000 narrow (2cm) demos start $(date)" | tee -a "$SUMMARY"

if [ -d "$DEMO_DIR" ] && [ "$(ls "$DEMO_DIR"/*.npz 2>/dev/null | wc -l)" -ge 1000 ]; then
  echo "[$TAG]   skip — $(ls "$DEMO_DIR"/*.npz | wc -l) NPZs already exist" | tee -a "$SUMMARY"
else
  env CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
      python scripts/collect_demos.py \
          --episodes 2000 \
          --stage stage0 \
          --out-root runs/demos/h6_true_narrow \
          --max-success 1000 \
          --curriculum-stage 0 \
      > "$COLLECT_LOG" 2>&1 &
  COLLECT_PID=$!
  echo "[$TAG]   collect PID=$COLLECT_PID" | tee -a "$SUMMARY"

  # Wait for collect to finish (oracle ~10-20s/ep, 2000 ep max ~11h, but 1000 success ~3-5h)
  while kill -0 $COLLECT_PID 2>/dev/null; do
    sleep 60
  done
  wait $COLLECT_PID 2>/dev/null

  # Close-hang cleanup
  COLLECT_PY=$(pgrep -f "collect_demos.py.*h6_true_narrow" | head -1 || true)
  if [ -n "$COLLECT_PY" ]; then
    echo "[$TAG]   collect close-hang, killing $COLLECT_PY" | tee -a "$SUMMARY"
    kill $COLLECT_PY 2>/dev/null || true; sleep 5
    kill -9 $COLLECT_PY 2>/dev/null || true; sleep 3
  fi

  N_DEMOS=$(ls "$DEMO_DIR"/*.npz 2>/dev/null | wc -l)
  echo "[$TAG]   collected $N_DEMOS demos at $(date)" | tee -a "$SUMMARY"
  if [ "$N_DEMOS" -lt 500 ]; then
    echo "[$TAG]   ERROR too few demos ($N_DEMOS < 500) — abort" | tee -a "$SUMMARY"
    exit 1
  fi
fi

echo "[$TAG] Phase 1 done $(date)" | tee -a "$SUMMARY"

# ============================================================
# Phase 2: BC pretrain on narrow demos (no sim needed)
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 2: BC pretrain start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

if [ -f "$BC_PT" ]; then
  echo "[$TAG]   skip — $BC_PT already exists" | tee -a "$SUMMARY"
else
  mkdir -p "$BC_DIR"
  env CUDA_VISIBLE_DEVICES=$GPU \
      python scripts/train_bc.py \
          --demo-dir "$DEMO_DIR" \
          --run-name "$(basename "$BC_DIR")" \
          --no-chunking \
          --epochs 50 \
          --batch-size 256 \
          --lr 3e-4 \
          --actor-logstd-init -1.0 \
          --seed 0 \
      > "$BC_LOG" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$TAG]   ERROR BC train failed (rc=$rc)" | tee -a "$SUMMARY"
    exit 1
  fi
  if [ ! -f "$BC_PT" ]; then
    echo "[$TAG]   ERROR bc.pt not created" | tee -a "$SUMMARY"
    exit 1
  fi
fi

echo "[$TAG]   bc.pt: $(ls -lh "$BC_PT")" | tee -a "$SUMMARY"
echo "[$TAG] Phase 2 done $(date)" | tee -a "$SUMMARY"

# ============================================================
# Phase 3: SACfD stage 1 (5cm), from BC ckpt, 400k steps
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 3: SACfD stage 1 (5cm) start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

run_sacfd_with_sentinel "$STAGE1_LOG" "$STAGE1_DIR" "$STAGE1_PT" "train_sac.py.*g_plus_sacfd_L2_H6_stage1" \
    --bc-ckpt "$BC_PT" \
    --demo-dir "$DEMO_DIR" \
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

echo "[$TAG] Phase 3 done $(date)" | tee -a "$SUMMARY"
[ -f "$STAGE1_PT" ] || { echo "[$TAG] ERROR stage1 ckpt missing"; exit 1; }

# ============================================================
# Phase 4: SACfD stage 2 (10cm), resume from stage 1, 600k steps
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 4: SACfD stage 2 (10cm) resume start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

run_sacfd_with_sentinel "$STAGE2_LOG" "$STAGE2_DIR" "$STAGE2_PT" "train_sac.py.*g_plus_sacfd_L2_H6_stage2" \
    --resume "$STAGE1_PT" \
    --demo-dir "$DEMO_DIR" \
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

echo "[$TAG] Phase 4 done $(date)" | tee -a "$SUMMARY"
[ -f "$STAGE2_PT" ] || { echo "[$TAG] ERROR stage2 ckpt missing"; exit 1; }

# ============================================================
# Phase 5: multi-seed eval
# ============================================================
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] Phase 5: multi-seed eval start $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
bash scripts/chain_reval.sh $TAG $GPU "$STAGE2_PT" 2>&1 | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[$TAG] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  Demos:      $DEMO_DIR" | tee -a "$SUMMARY"
echo "  BC ckpt:    $BC_PT" | tee -a "$SUMMARY"
echo "  Stage1 ckpt: $STAGE1_PT" | tee -a "$SUMMARY"
echo "  Stage2 ckpt: $STAGE2_PT" | tee -a "$SUMMARY"
echo "  eval logs:  runs/g_plus_chain/eval_${TAG}_st{2,1,0}_s{1,2,0}.log" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
