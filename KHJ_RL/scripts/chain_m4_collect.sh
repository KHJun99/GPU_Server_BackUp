#!/usr/bin/env bash
# Method B M4 — stage 0 narrow 1000 ep demo 재수집 (4-D action, 29-D obs schema).
# - 이전 옛 schema demo (runs/demos/g_plus_stage0_narrow/) 폐기, 새 path 생성
# - close-hang 자동 처리: '[SUMMARY] collected' sentinel detect 후 60s grace + SIGKILL

set -u
cd /home/j-k14d101/KHJ_RL

OUT_DIR=runs/demos/method_b_stage0
OUT_STAGE=$OUT_DIR/stage0
LOG=runs/g_plus_chain/m4_collect.log
SUMMARY=runs/g_plus_chain/m4_collect_summary.log

mkdir -p "$OUT_STAGE" runs/g_plus_chain
echo "[m4] start $(date)" | tee -a "$SUMMARY"
echo "[m4] target: 1000 success NPZ (4-D action, 29-D obs)" | tee -a "$SUMMARY"
echo "[m4] out:    $OUT_STAGE" | tee -a "$SUMMARY"

env CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/collect_demos.py \
        --curriculum-stage 0 \
        --episodes 3000 \
        --max-success 1000 \
        --stage stage0 \
        --out-root $OUT_DIR \
        --seed-offset 100000 \
    > "$LOG" 2>&1 &
TEE_PID=$!
echo "[m4] launched parent PID=$TEE_PID at $(date)" | tee -a "$SUMMARY"

sleep 60
PY_PID=""
for i in 1 2 3 4 5 6; do
  PY_PID=$(pgrep -f "scripts/collect_demos.py.*method_b_stage0" | head -1)
  [ -n "$PY_PID" ] && break
  sleep 15
done
if [ -z "$PY_PID" ]; then
  echo "[m4] ERROR python PID not found at $(date)" | tee -a "$SUMMARY"
  exit 1
fi
echo "[m4] python PID=$PY_PID at $(date)" | tee -a "$SUMMARY"

# sentinel: "[SUMMARY] collected" + 60s grace → SIGKILL
while kill -0 $PY_PID 2>/dev/null; do
  if grep -q '\[SUMMARY\] collected' "$LOG" 2>/dev/null; then
    echo "[m4] SUMMARY detected at $(date), 60s grace + SIGKILL" | tee -a "$SUMMARY"
    sleep 60
    if kill -0 $PY_PID 2>/dev/null; then
      echo "  SIGKILL $PY_PID" | tee -a "$SUMMARY"
      kill -9 $PY_PID 2>/dev/null || true
    fi
    break
  fi
  sleep 60
done
wait $TEE_PID 2>/dev/null

# 결과
n=$(ls "$OUT_STAGE"/*.npz 2>/dev/null | wc -l)
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[m4] FINAL $(date)" | tee -a "$SUMMARY"
echo "  collected: $n NPZ" | tee -a "$SUMMARY"
echo "  path:      $OUT_STAGE" | tee -a "$SUMMARY"
echo "  log:       $LOG" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
python3 - <<PY 2>/dev/null | tee -a "$SUMMARY"
import numpy as np, glob
files = sorted(glob.glob("$OUT_STAGE/*.npz"))
if files:
    z = np.load(files[0])
    print(f"  shape check: obs={z['obs'].shape} actions={z['actions'].shape}")
PY
echo "[m4] ALL DONE $(date)" | tee -a "$SUMMARY"
