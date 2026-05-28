#!/usr/bin/env bash
# Re-evaluation chain (inline close-hang handling — safe).
# 인자: $1 = tag (h1 | h2 | h2_seed1), $2 = GPU index, $3 = ckpt path
# 각 ckpt에 대해 4 eval sequential 실행:
#   - stage=2 seed=1 (H1 lift 83% 재현성 검증용)
#   - stage=2 seed=2 (또 다른 seed)
#   - stage=1 seed=0 (curriculum stage 1 — generalization)
#   - stage=0 seed=0 (curriculum stage 0 — most narrow)
# Inline killer: eval_policy 종료 sentinel ('[eval] ep_length mean=') detect 후
#                5s grace → SIGKILL. log mtime polling 없음 → 다른 PID 잘못 죽이는 버그 차단.

set -u
cd /home/j-k14d101/KHJ_RL

TAG=$1
GPU=$2
CKPT=$3

OUT_DIR=runs/g_plus_chain
SUMMARY=$OUT_DIR/reval_${TAG}_summary.log
mkdir -p $OUT_DIR

echo "[reval-$TAG] start $(date) GPU=$GPU ckpt=$CKPT" | tee -a "$SUMMARY"

run_eval() {
  local stage=$1
  local seed=$2
  local log=$OUT_DIR/eval_${TAG}_st${stage}_s${seed}.log
  echo "[reval-$TAG] eval stage=$stage seed=$seed start $(date) -> $log" | tee -a "$SUMMARY"

  env CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
      python scripts/eval_policy.py --ckpt "$CKPT" --episodes 100 \
          --task-level 2 --curriculum-stage $stage --seed $seed 2>&1 \
      | tee "$log" &
  local tee_pid=$!

  # python PID 검출 (Isaac Sim 부팅 30~60s)
  sleep 30
  local py_pid=""
  for i in 1 2 3 4 5 6; do
    py_pid=$(pgrep -f "scripts/eval_policy.py.*$CKPT.*--curriculum-stage $stage.*--seed $seed" 2>/dev/null | head -1)
    [ -n "$py_pid" ] && break
    sleep 10
  done
  if [ -z "$py_pid" ]; then
    echo "[reval-$TAG] WARN python PID not found, fall back to wait" | tee -a "$SUMMARY"
    wait $tee_pid 2>/dev/null
    return
  fi
  echo "[reval-$TAG]   python PID=$py_pid" | tee -a "$SUMMARY"

  # sentinel 대기 (close-hang 직전 마지막 출력)
  local sentinel='\[eval\] ep_length mean='
  while kill -0 $py_pid 2>/dev/null; do
    if grep -q "$sentinel" "$log" 2>/dev/null; then
      echo "[reval-$TAG]   sentinel detected, 5s grace then SIGKILL pid=$py_pid" | tee -a "$SUMMARY"
      sleep 5
      if kill -0 $py_pid 2>/dev/null; then
        kill -9 $py_pid 2>/dev/null || true
      fi
      break
    fi
    sleep 15
  done
  wait $tee_pid 2>/dev/null
  echo "[reval-$TAG] eval stage=$stage seed=$seed done $(date)" | tee -a "$SUMMARY"
}

run_eval 2 1
run_eval 2 2
run_eval 1 0
run_eval 0 0

echo "[reval-$TAG] ALL DONE $(date)" | tee -a "$SUMMARY"
