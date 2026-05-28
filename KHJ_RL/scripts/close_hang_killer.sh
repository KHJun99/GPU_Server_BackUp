#!/usr/bin/env bash
# Close-hang killer (watcher v3).
# runs/g_plus_chain/{measure_*,eval_*}.log 의 mtime이 300s 이상 idle이고
# 결과 sentinel이 찍혀 있으면 해당 ckpt에 매칭되는 python PID를 SIGKILL.
# chain_h2_seed1.sh + h1h2_watcher_v2.sh가 sequential 진행하는 동안 close-hang을
# 자동 해소하여 다음 단계 (eval) 로 넘어가게 함.

set -u
cd /home/j-k14d101/KHJ_RL

LOG=runs/g_plus_chain/close_hang_killer.log
mkdir -p runs/g_plus_chain
echo "[killer] start $(date) pid=$$" | tee -a "$LOG"

while true; do
  for f in runs/g_plus_chain/measure_h1.log \
           runs/g_plus_chain/measure_h2.log \
           runs/g_plus_chain/measure_h2_seed1.log \
           runs/g_plus_chain/eval_h1.log \
           runs/g_plus_chain/eval_h2.log \
           runs/g_plus_chain/eval_h2_seed1_stage0.log \
           runs/g_plus_chain/eval_h2_seed1_stage1.log \
           runs/g_plus_chain/eval_h2_seed1_stage2.log \
           runs/g_plus_chain/eval_h1_stage0.log \
           runs/g_plus_chain/eval_h1_stage1.log \
           runs/g_plus_chain/eval_h2_stage0.log \
           runs/g_plus_chain/eval_h2_stage1.log
  do
    [ -f "$f" ] || continue

    mtime=$(stat -c %Y "$f" 2>/dev/null) || continue
    idle=$(( $(date +%s) - mtime ))
    [ $idle -ge 300 ] || continue

    # sentinel 확인 (false positive 방지)
    if ! grep -q -E '\[measure_reach\] OVERALL|\[eval\] ep_length mean=' "$f" 2>/dev/null; then
      continue
    fi

    # ckpt path 추출 → 해당 PID 찾기
    ckpt=$(grep -oE 'runs/g_plus_sacfd_[^/]+/sac\.pt' "$f" 2>/dev/null | head -1)
    [ -z "$ckpt" ] && continue

    # 어느 script 인지 결정 (measure 또는 eval 만 잡기)
    case "$(basename "$f")" in
      measure_*) script_pat="scripts/measure_reach.py" ;;
      eval_*) script_pat="scripts/eval_policy.py" ;;
      *) continue ;;
    esac

    # 매칭 PID list (script + ckpt 둘 다 cmdline에 포함된 것)
    pids=""
    for pid in $(pgrep -f "$script_pat" 2>/dev/null); do
      if grep -q "$ckpt" "/proc/$pid/cmdline" 2>/dev/null; then
        pids="$pids $pid"
      fi
    done
    [ -z "$pids" ] && continue

    # SIGKILL + tee 자식 정리
    for pid in $pids; do
      echo "[killer] $(date) SIGKILL pid=$pid log=$f idle=${idle}s ckpt=$ckpt" | tee -a "$LOG"
      # python child 먼저
      for child in $(pgrep -P $pid 2>/dev/null); do
        kill -9 "$child" 2>/dev/null || true
      done
      kill -9 "$pid" 2>/dev/null || true
    done
    # 같은 log 파일에 쓰던 tee 도 정리
    pkill -9 -f "tee $f" 2>/dev/null || true

    sleep 5  # SIGKILL 후 GPU 회수 잠시 대기
  done
  sleep 30
done
