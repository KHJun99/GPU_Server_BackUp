#!/usr/bin/env bash
# H1/H2 학습 완료 watcher + 자동 평가 파이프라인.
# 1) sac.pt 두 개 다 detect → 30s grace → SIGTERM → 10s → SIGKILL
# 2) measure_reach.py 두 ckpt에 sequential 실행
# 3) eval_policy.py 두 ckpt에 100 ep sequential 실행
# 4) 결과 요약을 runs/g_plus_chain/h1h2_summary.log에 기록
set -u
cd /home/j-k14d101/KHJ_RL

SAC0=runs/g_plus_sacfd_L2_H1_bcw5_warm10k/sac.pt
SAC1=runs/g_plus_sacfd_L2_H2_demomix/sac.pt
PID0=1087796
PID1=1088335

SUMMARY=runs/g_plus_chain/h1h2_summary.log
M0=runs/g_plus_chain/measure_h1.log
M1=runs/g_plus_chain/measure_h2.log
E0=runs/g_plus_chain/eval_h1.log
E1=runs/g_plus_chain/eval_h2.log

mkdir -p runs/g_plus_chain
echo "[h1h2-watcher-v2] start $(date)" | tee -a "$SUMMARY"

s0=0; s1=0
while true; do
  if [ "$s0" -eq 0 ] && [ -f "$SAC0" ]; then
    s0=1; echo "[h1h2-watcher-v2] $(date) H1 sac.pt detected" | tee -a "$SUMMARY"
  fi
  if [ "$s1" -eq 0 ] && [ -f "$SAC1" ]; then
    s1=1; echo "[h1h2-watcher-v2] $(date) H2 sac.pt detected" | tee -a "$SUMMARY"
  fi
  if [ "$s0" -eq 1 ] && [ "$s1" -eq 1 ]; then
    echo "[h1h2-watcher-v2] both sac.pt — 30s grace then SIGTERM" | tee -a "$SUMMARY"
    sleep 30
    if kill -0 $PID0 2>/dev/null; then echo "SIGTERM $PID0" | tee -a "$SUMMARY"; kill $PID0 || true; fi
    if kill -0 $PID1 2>/dev/null; then echo "SIGTERM $PID1" | tee -a "$SUMMARY"; kill $PID1 || true; fi
    sleep 10
    if kill -0 $PID0 2>/dev/null; then echo "SIGKILL $PID0" | tee -a "$SUMMARY"; kill -9 $PID0 || true; fi
    if kill -0 $PID1 2>/dev/null; then echo "SIGKILL $PID1" | tee -a "$SUMMARY"; kill -9 $PID1 || true; fi
    # GPU 메모리 회수 grace
    sleep 20
    break
  fi
  if ! kill -0 $PID0 2>/dev/null && ! kill -0 $PID1 2>/dev/null && [ "$s0" -eq 0 ] && [ "$s1" -eq 0 ]; then
    echo "[h1h2-watcher-v2] WARNING both PIDs died with no sac.pt at $(date)" | tee -a "$SUMMARY"
    exit 1
  fi
  sleep 60
done

echo "[h1h2-watcher-v2] cleanup done at $(date)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "POST-EVAL phase 1: measure_reach.py (GPU 0, sequential)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# H1 measure_reach
echo "[h1h2-watcher-v2] measure_reach H1 start $(date)" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/measure_reach.py --ckpt "$SAC0" 2>&1 | tee "$M0"
echo "[h1h2-watcher-v2] measure_reach H1 done $(date)" | tee -a "$SUMMARY"

# H2 measure_reach
echo "[h1h2-watcher-v2] measure_reach H2 start $(date)" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/measure_reach.py --ckpt "$SAC1" 2>&1 | tee "$M1"
echo "[h1h2-watcher-v2] measure_reach H2 done $(date)" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "POST-EVAL phase 2: eval_policy.py 100 ep (GPU 0, sequential)" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"

# H1 eval_policy
echo "[h1h2-watcher-v2] eval_policy H1 start $(date)" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/eval_policy.py --ckpt "$SAC0" --episodes 100 \
        --task-level 2 --curriculum-stage 2 --seed 0 2>&1 | tee "$E0"
echo "[h1h2-watcher-v2] eval_policy H1 done $(date)" | tee -a "$SUMMARY"

# H2 eval_policy
echo "[h1h2-watcher-v2] eval_policy H2 start $(date)" | tee -a "$SUMMARY"
env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/eval_policy.py --ckpt "$SAC1" --episodes 100 \
        --task-level 2 --curriculum-stage 2 --seed 0 2>&1 | tee "$E1"
echo "[h1h2-watcher-v2] eval_policy H2 done $(date)" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
echo "[h1h2-watcher-v2] ALL DONE $(date)" | tee -a "$SUMMARY"
echo "  H1 ckpt:      $SAC0" | tee -a "$SUMMARY"
echo "  H2 ckpt:      $SAC1" | tee -a "$SUMMARY"
echo "  measure_h1:   $M0" | tee -a "$SUMMARY"
echo "  measure_h2:   $M1" | tee -a "$SUMMARY"
echo "  eval_h1:      $E0" | tee -a "$SUMMARY"
echo "  eval_h2:      $E1" | tee -a "$SUMMARY"
echo "============================================================" | tee -a "$SUMMARY"
