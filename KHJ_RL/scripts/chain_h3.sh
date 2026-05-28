#!/usr/bin/env bash
# H3 full pipeline (자동, 사용자 개입 X):
# 1) collect_demos (PID $COLLECT_PID) 완료 대기 → wider demo 검증
# 2) launch_l2_h3_distmix.sh background launch (GPU 0)
# 3) train_sac PID detect, sac.pt 검출 → close-hang 정리
# 4) measure_reach (GPU 0, inline sentinel killer)
# 5) chain_reval.sh h3 0 $SAC_PT (multi-seed eval: stage 2 seed=1,2 + stage 1/0)
# 6) ALL DONE

set -u
cd /home/j-k14d101/KHJ_RL

COLLECT_PID="${1:-1410245}"
WIDER_DIR=runs/demos/g_plus_stage2_wider/stage0
OUT_DIR=runs/g_plus_sacfd_L2_H3_distmix
SAC_PT=$OUT_DIR/sac.pt
LAUNCH_SCRIPT=scripts/launch_l2_h3_distmix.sh
SUMMARY=runs/g_plus_chain/h3_chain_summary.log
LAUNCH_LOG=runs/g_plus_chain/launch_l2_h3.log

mkdir -p runs/g_plus_chain
echo "[chain-h3] start $(date) collect_pid=$COLLECT_PID" | tee -a "$SUMMARY"

# 1) collect 완료 wait
echo "[chain-h3] waiting for collect PID=$COLLECT_PID" | tee -a "$SUMMARY"
while kill -0 $COLLECT_PID 2>/dev/null; do
  sleep 60
done
echo "[chain-h3] collect done at $(date)" | tee -a "$SUMMARY"

# 2) demo count 검증
n=$(ls "$WIDER_DIR"/*.npz 2>/dev/null | wc -l)
echo "[chain-h3] wider demo NPZ count: $n" | tee -a "$SUMMARY"
if [ "$n" -lt 600 ]; then
  echo "[chain-h3] WARN: only $n NPZ — continue anyway" | tee -a "$SUMMARY"
fi

# 3) H3 launcher background
nohup bash "$LAUNCH_SCRIPT" > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
echo "[chain-h3] H3 launcher PID=$LAUNCH_PID at $(date)" | tee -a "$SUMMARY"

# 4) train_sac PID detect (Isaac Sim 부팅 30~90s)
sleep 90
TRAIN_PID=""
for i in 1 2 3 4 5 6; do
  TRAIN_PID=$(pgrep -f "train_sac.py.*g_plus_sacfd_L2_H3_distmix" | head -1)
  [ -n "$TRAIN_PID" ] && break
  sleep 30
done
if [ -z "$TRAIN_PID" ]; then
  echo "[chain-h3] ERROR train_sac PID not found at $(date)" | tee -a "$SUMMARY"
  exit 1
fi
echo "[chain-h3] train_sac PID=$TRAIN_PID at $(date)" | tee -a "$SUMMARY"

# 5) sac.pt 검출 대기 + close-hang 정리
while true; do
  if [ -f "$SAC_PT" ]; then
    echo "[chain-h3] sac.pt detected at $(date), 30s grace then SIGTERM" | tee -a "$SUMMARY"
    sleep 30
    if kill -0 $TRAIN_PID 2>/dev/null; then echo "  SIGTERM $TRAIN_PID" | tee -a "$SUMMARY"; kill $TRAIN_PID || true; fi
    sleep 10
    if kill -0 $TRAIN_PID 2>/dev/null; then echo "  SIGKILL $TRAIN_PID" | tee -a "$SUMMARY"; kill -9 $TRAIN_PID || true; fi
    sleep 20
    break
  fi
  if ! kill -0 $TRAIN_PID 2>/dev/null; then
    if [ -f "$SAC_PT" ]; then
      echo "[chain-h3] train died but sac.pt present, ok" | tee -a "$SUMMARY"
      break
    else
      echo "[chain-h3] ERROR train died with no sac.pt at $(date)" | tee -a "$SUMMARY"
      exit 1
    fi
  fi
  sleep 60
done

# 6) measure_reach (GPU 0, inline sentinel killer)
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "POST-EVAL: measure_reach (GPU 0)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
MLOG=runs/g_plus_chain/measure_h3.log
echo "[chain-h3] measure_reach start $(date)" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/measure_reach.py --ckpt "$SAC_PT" 2>&1 | tee "$MLOG" &
MTEE=$!
sleep 30
MPY=$(pgrep -f "scripts/measure_reach.py.*$SAC_PT" | head -1)
if [ -n "$MPY" ]; then
  while kill -0 $MPY 2>/dev/null; do
    if grep -q '\[measure_reach\] OVERALL' "$MLOG" 2>/dev/null; then
      sleep 5
      if kill -0 $MPY 2>/dev/null; then kill -9 $MPY 2>/dev/null || true; fi
      break
    fi
    sleep 15
  done
fi
wait $MTEE 2>/dev/null
echo "[chain-h3] measure_reach done $(date)" | tee -a "$SUMMARY"

# 7) chain_reval h3 (multi-seed eval)
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "POST-EVAL: chain_reval h3 (multi-seed eval, GPU 0)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[chain-h3] chain_reval start $(date)" | tee -a "$SUMMARY"
bash scripts/chain_reval.sh h3 0 "$SAC_PT" 2>&1 | tee -a "$SUMMARY"
echo "[chain-h3] chain_reval done $(date)" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[chain-h3] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  H3 ckpt:        $SAC_PT" | tee -a "$SUMMARY"
echo "  step_l2_h3:     runs/g_plus_chain/step_l2_h3_full.log" | tee -a "$SUMMARY"
echo "  measure_h3:     $MLOG" | tee -a "$SUMMARY"
echo "  eval_h3_*:      runs/g_plus_chain/eval_h3_st{2,1,0}_s{1,2,0}.log" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
