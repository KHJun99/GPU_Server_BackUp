#!/usr/bin/env bash
# 1) launch_l2_h2_demomix_seed1.sh를 background로 시작 (GPU 1, seed=1)
# 2) sac.pt 검출 → 30s grace → SIGTERM → 10s → SIGKILL → 20s GPU 회수
# 3) H2 seed=1: measure_reach + eval_policy 100ep × stage 2/1/0 (GPU 1)
# 4) watcher v2 "ALL DONE" 대기
# 5) H1, H2(seed=0)에 stage 1, 0 추가 평가 (GPU 0)
# 6) 결과 요약 → runs/g_plus_chain/seed1_chain_summary.log

set -u
cd /home/j-k14d101/KHJ_RL

OUT_SEED1=runs/g_plus_sacfd_L2_H2_demomix_seed1
SAC_SEED1=$OUT_SEED1/sac.pt
SAC_H1=runs/g_plus_sacfd_L2_H1_bcw5_warm10k/sac.pt
SAC_H2=runs/g_plus_sacfd_L2_H2_demomix/sac.pt

SUMMARY=runs/g_plus_chain/seed1_chain_summary.log
LAUNCH_LOG=runs/g_plus_chain/launch_l2_h2_seed1.log

mkdir -p runs/g_plus_chain
echo "[chain-seed1] start $(date)" | tee -a "$SUMMARY"

# 1) launcher background
nohup bash scripts/launch_l2_h2_demomix_seed1.sh > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
echo "[chain-seed1] launcher PID=$LAUNCH_PID at $(date)" | tee -a "$SUMMARY"

# 2) train_sac PID 검출 (Isaac Sim 부팅 30~90초)
sleep 90
TRAIN_PID=$(pgrep -f "train_sac.py.*g_plus_sacfd_L2_H2_demomix_seed1" | head -1)
if [ -z "$TRAIN_PID" ]; then
  echo "[chain-seed1] ERROR train_sac PID not found after 90s at $(date)" | tee -a "$SUMMARY"
  echo "  see $LAUNCH_LOG and $LOG" | tee -a "$SUMMARY"
  exit 1
fi
echo "[chain-seed1] train_sac PID=$TRAIN_PID at $(date)" | tee -a "$SUMMARY"

# 3) sac.pt 검출 대기 + close-hang 정리
while true; do
  if [ -f "$SAC_SEED1" ]; then
    echo "[chain-seed1] sac.pt detected at $(date), 30s grace then SIGTERM" | tee -a "$SUMMARY"
    sleep 30
    if kill -0 $TRAIN_PID 2>/dev/null; then echo "  SIGTERM $TRAIN_PID" | tee -a "$SUMMARY"; kill $TRAIN_PID || true; fi
    sleep 10
    if kill -0 $TRAIN_PID 2>/dev/null; then echo "  SIGKILL $TRAIN_PID" | tee -a "$SUMMARY"; kill -9 $TRAIN_PID || true; fi
    sleep 20  # GPU 회수
    break
  fi
  if ! kill -0 $TRAIN_PID 2>/dev/null; then
    if [ -f "$SAC_SEED1" ]; then
      echo "[chain-seed1] train died but sac.pt present, ok" | tee -a "$SUMMARY"
      break
    else
      echo "[chain-seed1] ERROR train died with no sac.pt at $(date)" | tee -a "$SUMMARY"
      exit 1
    fi
  fi
  sleep 60
done

# 4) H2 seed=1: measure_reach + eval stage 2/1/0 (GPU 1; watcher v2가 GPU 0에서 eval 중이라 conflict 없음)
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "POST-EVAL phase A: H2 seed=1 (GPU 1, sequential)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

echo "[chain-seed1] measure_reach H2-seed1 (GPU 1) start $(date)" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/measure_reach.py --ckpt "$SAC_SEED1" 2>&1 \
    | tee runs/g_plus_chain/measure_h2_seed1.log
echo "[chain-seed1] measure_reach H2-seed1 done $(date)" | tee -a "$SUMMARY"

for STAGE in 2 1 0; do
  echo "[chain-seed1] eval_policy H2-seed1 stage=$STAGE (GPU 1) start $(date)" | tee -a "$SUMMARY"
  env CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
      python scripts/eval_policy.py --ckpt "$SAC_SEED1" --episodes 100 \
          --task-level 2 --curriculum-stage $STAGE --seed 0 2>&1 \
      | tee runs/g_plus_chain/eval_h2_seed1_stage${STAGE}.log
  echo "[chain-seed1] eval_policy H2-seed1 stage=$STAGE done $(date)" | tee -a "$SUMMARY"
done

# 5) watcher v2 "ALL DONE" 대기 (이미 끝났으면 즉시 통과)
echo "" | tee -a "$SUMMARY"
echo "[chain-seed1] waiting for watcher v2 ALL DONE at $(date)" | tee -a "$SUMMARY"
while true; do
  if grep -q "ALL DONE" runs/g_plus_chain/h1h2_summary.log 2>/dev/null; then
    echo "[chain-seed1] watcher v2 done at $(date)" | tee -a "$SUMMARY"
    break
  fi
  sleep 60
done

# 6) H1, H2(seed=0)에 stage 1, 0 추가 평가 (GPU 0; watcher v2가 비운 후)
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "POST-EVAL phase B: H1/H2(seed=0) stage 1/0 (GPU 0, sequential)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

for TAG in h1 h2; do
  if [ "$TAG" = "h1" ]; then CKPT=$SAC_H1; else CKPT=$SAC_H2; fi
  for STAGE in 1 0; do
    echo "[chain-seed1] eval_policy $TAG stage=$STAGE (GPU 0) start $(date)" | tee -a "$SUMMARY"
    env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
        python scripts/eval_policy.py --ckpt "$CKPT" --episodes 100 \
            --task-level 2 --curriculum-stage $STAGE --seed 0 2>&1 \
        | tee runs/g_plus_chain/eval_${TAG}_stage${STAGE}.log
    echo "[chain-seed1] eval_policy $TAG stage=$STAGE done $(date)" | tee -a "$SUMMARY"
  done
done

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[chain-seed1] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  H2-seed1 ckpt:       $SAC_SEED1" | tee -a "$SUMMARY"
echo "  measure_h2_seed1:    runs/g_plus_chain/measure_h2_seed1.log" | tee -a "$SUMMARY"
echo "  eval_h2_seed1_st0/1/2: runs/g_plus_chain/eval_h2_seed1_stage{0,1,2}.log" | tee -a "$SUMMARY"
echo "  eval_h1_st0/1:       runs/g_plus_chain/eval_h1_stage{0,1}.log" | tee -a "$SUMMARY"
echo "  eval_h2_st0/1:       runs/g_plus_chain/eval_h2_stage{0,1}.log" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
