#!/bin/bash
# BC stochastic eval sweep — measures BC robustness at SAC-like noise levels.
#
# Hypothesis: BC ckpt deterministic eval = 90% lift_history. SAC online
# rollout uses actor.sample() which adds Gaussian noise. If BC's robust
# strength under noise is much lower than deterministic, that explains
# why F3 SACfD can't ignite (Q-anchored on BC manifold but actor's own
# stochastic sampling pushes it OOD).
#
# Sweep std ∈ {0.05, 0.10, 0.20} × 100 ep each at task_level=2 (full
# success criterion for cross-comparison with chain STEP 3 result).
#
# Each eval ~10 min (with close-hang marker killer). Total ~30 min.
#
# Usage:
#   bash scripts/eval_bc_stochastic_sweep.sh
# Logs/results land at runs/g_plus_chain/bc_stoch_*.

set -uo pipefail
cd "$(dirname "$0")/.."

CHAIN_ROOT="runs/g_plus_chain"
mkdir -p "$CHAIN_ROOT"
SUMMARY="$CHAIN_ROOT/bc_stoch_sweep_summary.txt"
SWEEP_LOG="$CHAIN_ROOT/bc_stoch_sweep.log"
BC_CKPT="runs/g_plus_stage0_bc_narrow/bc.pt"

if [ ! -f "$BC_CKPT" ]; then
    echo "BC ckpt not found at $BC_CKPT — aborting" | tee -a "$SWEEP_LOG"
    exit 1
fi

log() {
    local ts
    ts="$(date '+%F %T')"
    echo "[$ts] $*" | tee -a "$SWEEP_LOG"
}

# Marker-based close-hang killer (same pattern as chain_g_plus.sh).
run_eval_with_killer() {
    local std="$1"
    local logf="$2"
    log "  launching BC stochastic eval std=$std → $logf"
    env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
        python scripts/eval_policy.py \
            --ckpt "$BC_CKPT" \
            --episodes 100 \
            --curriculum-stage 0 \
            --task-level 2 \
            --policy-std-override "$std" \
            --seed 0 \
            > "$logf" 2>&1 &
    local pid=$!
    log "  PID=$pid waiting for SUCCESS RATE marker"
    local elapsed=0
    local timeout=1500
    while ! grep -qE "SUCCESS RATE:|Traceback|RuntimeError|FAILED|Killed|OOM" "$logf" 2>/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "  PID dead before marker"
            wait "$pid" 2>/dev/null || true
            return 1
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            log "  TIMEOUT after ${timeout}s — killing"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 5
            kill -KILL "$pid" 2>/dev/null || true
            return 124
        fi
        sleep 30
        elapsed=$((elapsed + 30))
    done
    if grep -qE "Traceback|RuntimeError|FAILED|Killed|OOM" "$logf"; then
        log "  ERROR detected"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 3
        kill -KILL "$pid" 2>/dev/null || true
        return 1
    fi
    log "  marker seen — giving 10s grace then SIGTERM"
    sleep 10
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    kill -KILL "$pid" 2>/dev/null || true
    return 0
}

extract_metric() {
    # extract_metric <logf> <pattern>
    # Returns the percentage float after the last match of pattern
    grep -oE "$2: [0-9]+/[0-9]+ = [0-9.]+%" "$1" 2>/dev/null | tail -1 \
        | sed -E 's|.*= ([0-9.]+)%|\1|'
}

extract_sr() {
    grep -oE "SUCCESS RATE: [0-9]+/[0-9]+ = [0-9.]+%" "$1" 2>/dev/null | tail -1 \
        | sed -E 's|.*= ([0-9.]+)%|\1|'
}

# -----------------------------------------------------------------------
log "============================================================"
log "BC stochastic eval sweep START"
log "  ckpt: $BC_CKPT"
log "  std values: 0.05 0.10 0.20"
log "  episodes per std: 100"
log "  task_level: 2 (full 5-cond AND, same as chain STEP 3)"
log "  curriculum-stage: 0 (narrow 2cm spawn, same as BC training)"
log "============================================================"

# Reference: deterministic eval result from chain STEP 3.
# SR = 35.0%, lift_history = 90.0%, stable_placement = 85%,
# velocity_stability = 85%, release_retreat = 35%, visual_agreement = 38%
echo "[reference, deterministic]" >> "$SUMMARY"
echo "  SR=35.0% lift_history=90.0% stable_placement=85.0% velocity_stability=85.0% release_retreat=35.0% visual_agreement=38.0%" >> "$SUMMARY"
echo "" >> "$SUMMARY"

for STD in 0.05 0.10 0.20; do
    LOGF="$CHAIN_ROOT/bc_stoch_std${STD}.log"
    log "STD = $STD"
    run_eval_with_killer "$STD" "$LOGF"
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
        log "  eval std=$STD CRASHED rc=$rc — recording in summary and continuing"
        echo "[std=$STD]  CRASHED (see $LOGF)" >> "$SUMMARY"
        continue
    fi
    sr=$(extract_sr "$LOGF")
    lift=$(extract_metric "$LOGF" "lift_history")
    place=$(extract_metric "$LOGF" "stable_placement")
    vel=$(extract_metric "$LOGF" "velocity_stability")
    rel=$(extract_metric "$LOGF" "release_retreat")
    vis=$(extract_metric "$LOGF" "visual_agreement")
    log "  result std=$STD: SR=${sr}% lift=${lift}% place=${place}% vel=${vel}% rel=${rel}% vis=${vis}%"
    echo "[std=$STD]" >> "$SUMMARY"
    echo "  SR=${sr}% lift_history=${lift}% stable_placement=${place}% velocity_stability=${vel}% release_retreat=${rel}% visual_agreement=${vis}%" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
done

log "============================================================"
log "BC stochastic eval sweep DONE"
log "summary -> $SUMMARY"
log "============================================================"
cat "$SUMMARY" | tee -a "$SWEEP_LOG"
exit 0
