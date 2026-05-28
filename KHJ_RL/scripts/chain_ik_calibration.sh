#!/usr/bin/env bash
# Method B M3.3 + M0.5 Step D/E chain — 자동 close-hang 처리 포함.
# 1) scripts/run_ik_calibration.py 100 cal + 100 val launch (GPU 1)
# 2) runs/calibration/ik_thresholds.json 생성 detect → 30s grace → SIGKILL
# 3) 결과 요약 (ik_calibration_report.txt) tail
# 4) ALL DONE → 사용자 기상 시 확인

set -u
cd /home/j-k14d101/KHJ_RL

LOG=runs/g_plus_chain/ik_calibration.log
SUMMARY=runs/g_plus_chain/ik_calibration_chain_summary.log
THRESHOLDS_JSON=runs/calibration/ik_thresholds.json
REPORT_TXT=runs/calibration/ik_calibration_report.txt

mkdir -p runs/g_plus_chain runs/calibration
echo "[chain-ik-cal] start $(date)" | tee -a "$SUMMARY"

# 1) launch
env CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/run_ik_calibration.py \
        --n-calibration 100 \
        --n-validation 100 \
        --out-dir runs/calibration \
        --curriculum-stage 0 \
        --task-level 2 \
    > "$LOG" 2>&1 &
TEE_PID=$!
echo "[chain-ik-cal] launched (parent PID=$TEE_PID) at $(date)" | tee -a "$SUMMARY"

# 2) python PID detect (Isaac Sim 부팅 30~90s)
sleep 60
PY_PID=""
for i in 1 2 3 4 5 6; do
  PY_PID=$(pgrep -f "scripts/run_ik_calibration.py" | head -1)
  [ -n "$PY_PID" ] && break
  sleep 15
done
if [ -z "$PY_PID" ]; then
  echo "[chain-ik-cal] ERROR python PID not found at $(date)" | tee -a "$SUMMARY"
  exit 1
fi
echo "[chain-ik-cal] python PID=$PY_PID at $(date)" | tee -a "$SUMMARY"

# 3) ik_thresholds.json 생성 detect (sentinel) → 60s grace → SIGKILL
while kill -0 $PY_PID 2>/dev/null; do
  if [ -f "$THRESHOLDS_JSON" ]; then
    echo "[chain-ik-cal] $THRESHOLDS_JSON detected at $(date), 60s grace then SIGKILL" | tee -a "$SUMMARY"
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

# 4) 결과 요약
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[chain-ik-cal] FINAL $(date)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
if [ -f "$THRESHOLDS_JSON" ]; then
  echo "" | tee -a "$SUMMARY"
  echo "--- ik_thresholds.json ---" | tee -a "$SUMMARY"
  cat "$THRESHOLDS_JSON" | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
fi
if [ -f "$REPORT_TXT" ]; then
  echo "--- ik_calibration_report.txt ---" | tee -a "$SUMMARY"
  cat "$REPORT_TXT" | tee -a "$SUMMARY"
fi
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[chain-ik-cal] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  log:        $LOG" | tee -a "$SUMMARY"
echo "  thresholds: $THRESHOLDS_JSON" | tee -a "$SUMMARY"
echo "  report:     $REPORT_TXT" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
