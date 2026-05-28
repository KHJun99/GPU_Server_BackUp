#!/bin/bash
# G+ chain pipeline — runs steps 1..8 of the plan autonomously.
#
# Steps:
#   1. Wait for demo collect (assumed already running in tmux 'gplus_collect').
#   2. BC retrain (1-step, no chunking) on the narrow demos.
#   3. BC eval gate at task_level 2 — must reach >= 50% before proceeding.
#   4. SACfD smoke task_level 0 — 50k step, gate: ep_return > 0 within 50k.
#   5. SACfD full task_level 0 — 300k step + video sampler.
#   6. Eval task_level 0 — 100 ep, gate: lift_history rate >= 70%.
#   7. SACfD smoke task_level 1 — 50k step.
#   8. SACfD full task_level 1 — 300k step + video sampler.
#   9. SACfD smoke task_level 2 — 50k step (full PnP, 5-condition AND).
#  10. SACfD full task_level 2 — 300k step + video sampler.
#
# Each python step is launched with a marker-based killer (see
# troubleshooting #17 / #22) so close-hang doesn't stall the chain.
# Logs land at runs/g_plus_chain/step{N}_{name}.log. Final ckpts:
#   runs/g_plus_stage0_bc_narrow/bc.pt
#   runs/g_plus_sacfd_L0_full/sac.pt
#   runs/g_plus_sacfd_L1_full/sac.pt
#
# Usage:
#   ./scripts/chain_g_plus.sh
# Stop with: pkill -f chain_g_plus.sh && pkill -f train_sac && pkill -f eval_policy

set -uo pipefail
cd "$(dirname "$0")/.."

CHAIN_ROOT="runs/g_plus_chain"
mkdir -p "$CHAIN_ROOT"
CHAIN_LOG="$CHAIN_ROOT/chain.log"
STATUS_FILE="$CHAIN_ROOT/status.txt"

DEMO_DIR="runs/demos/g_plus_stage0_narrow/stage0"
BC_DIR="runs/g_plus_stage0_bc_narrow"
SAC_L0_SMOKE="runs/g_plus_sacfd_L0_smoke"
SAC_L0_FULL="runs/g_plus_sacfd_L0_full"
SAC_L1_SMOKE="runs/g_plus_sacfd_L1_smoke"
SAC_L1_FULL="runs/g_plus_sacfd_L1_full"
SAC_L2_SMOKE="runs/g_plus_sacfd_L2_smoke"
SAC_L2_FULL="runs/g_plus_sacfd_L2_full"

DEMO_DIRS_ARG=("$DEMO_DIR")

log() {
    local ts
    ts="$(date '+%F %T')"
    echo "[$ts] $*" | tee -a "$CHAIN_LOG"
}

set_status() {
    echo "$(date '+%F %T') | $*" >> "$STATUS_FILE"
}

run_python_with_killer() {
    # Args: <step name> <log file> <success marker regex> <wait timeout sec> <python cmd...>
    local step_name="$1"; shift
    local logf="$1"; shift
    local marker_re="$1"; shift
    local timeout_sec="$1"; shift
    log "  launching $step_name: $*"
    log "  log -> $logf"
    "$@" > "$logf" 2>&1 &
    local pid=$!
    log "  PID=$pid waiting for marker /$marker_re/ (timeout ${timeout_sec}s)"
    local elapsed=0
    while ! grep -qE "$marker_re|Traceback|RuntimeError|FAILED|Killed|OOM" "$logf" 2>/dev/null; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "  $step_name PID dead before marker"
            wait "$pid" 2>/dev/null || true
            local rc=$?
            return "$rc"
        fi
        if [ "$elapsed" -ge "$timeout_sec" ]; then
            log "  $step_name TIMEOUT after ${timeout_sec}s — killing"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 5
            kill -KILL "$pid" 2>/dev/null || true
            return 124
        fi
        sleep 30
        elapsed=$((elapsed + 30))
    done
    # Check for error markers
    if grep -qE "Traceback|RuntimeError|FAILED|Killed|OOM" "$logf"; then
        log "  $step_name ERROR detected in log"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 3
        kill -KILL "$pid" 2>/dev/null || true
        return 1
    fi
    log "  $step_name marker seen — giving 10s grace then SIGTERM (close-hang killer)"
    sleep 10
    kill -TERM "$pid" 2>/dev/null || true
    sleep 5
    kill -KILL "$pid" 2>/dev/null || true
    return 0
}

extract_success_rate() {
    # Last "SUCCESS RATE: N/M = P.P%" line → P.P (float, no %)
    local logf="$1"
    grep -oE "SUCCESS RATE: [0-9]+/[0-9]+ = [0-9.]+%" "$logf" | tail -1 \
        | sed -E 's|.*= ([0-9.]+)%|\1|'
}

extract_condition_rate() {
    # condition_rate "lift_history: N/M = P.P%" → P.P
    local logf="$1"
    local cond="$2"
    grep -oE "$cond: [0-9]+/[0-9]+ = [0-9.]+%" "$logf" | tail -1 \
        | sed -E "s|.*= ([0-9.]+)%|\\1|"
}

extract_ep_return_max() {
    # Max ep_return_mean across all [SACfD] step lines.
    local logf="$1"
    grep -oE "ep_return_mean=-?[0-9.]+" "$logf" \
        | sed -E 's|ep_return_mean=||' \
        | sort -gr | head -1
}

abort_chain() {
    set_status "ABORT: $*"
    log "============================================================"
    log "CHAIN ABORTED: $*"
    log "============================================================"
    exit 1
}

# -----------------------------------------------------------------------
log "============================================================"
log "G+ CHAIN START (steps 1..8)"
log "============================================================"
set_status "STARTED"

# -----------------------------------------------------------------------
# Step 1: wait for demo collect to finish (assumes tmux 'gplus_collect').
# Idempotent: skip if 1000 NPZs already exist (re-run safe).
n_existing=0
if [ -d "$DEMO_DIR" ]; then
    n_existing=$(ls "$DEMO_DIR"/*.npz 2>/dev/null | wc -l)
fi
if [ "$n_existing" -ge 1000 ]; then
    log "STEP 1: skip — $n_existing NPZs already at $DEMO_DIR"
    set_status "STEP1 skipped (already $n_existing NPZs)"
else
    log "STEP 1: wait for demo collect (target = 1000 NPZs at $DEMO_DIR)"
    set_status "STEP1 wait collect"
    while true; do
        if [ -d "$DEMO_DIR" ]; then
            n=$(ls "$DEMO_DIR"/*.npz 2>/dev/null | wc -l)
            if [ "$n" -ge 1000 ]; then
                log "  collect done: $n NPZs"
                break
            fi
            log "  progress: $n / 1000 NPZs"
        else
            log "  $DEMO_DIR does not exist yet — waiting"
        fi
        sleep 120
    done

    # Kill any lingering collect_demos (close-hang) so step 2 has GPU.
    collect_pid=$(pgrep -f "collect_demos.py" | head -1 || true)
    if [ -n "$collect_pid" ]; then
        log "  killing lingering collect_demos PID=$collect_pid (close-hang cleanup)"
        kill -TERM "$collect_pid" 2>/dev/null || true
        sleep 10
        kill -KILL "$collect_pid" 2>/dev/null || true
    fi
    set_status "STEP1 done"
fi

# -----------------------------------------------------------------------
# Step 2: BC retrain (1-step, no chunking) on G+ narrow demos.
# Idempotent: skip if bc.pt + norm.json already exist.
if [ -f "$BC_DIR/bc.pt" ] && [ -f "$BC_DIR/norm.json" ]; then
    log "STEP 2: skip — $BC_DIR/bc.pt already exists"
    set_status "STEP2 skipped"
else
    log "STEP 2: BC retrain (1-step) on $DEMO_DIR"
    set_status "STEP2 BC retrain"
    mkdir -p "$BC_DIR"
    BC_LOG="$CHAIN_ROOT/step2_bc.log"
    env CUDA_VISIBLE_DEVICES=0 \
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
        abort_chain "BC retrain failed (rc=$rc), see $BC_LOG"
    fi
    if [ ! -f "$BC_DIR/bc.pt" ]; then
        abort_chain "BC ckpt not produced at $BC_DIR/bc.pt"
    fi
    log "  BC done: $BC_DIR/bc.pt"
    set_status "STEP2 done"
fi

# -----------------------------------------------------------------------
# Step 3: BC eval gate (task_level 2, deterministic, 100 ep).
# Gate aligned with SACfD L0 entry: lift_history >= 70% (NOT SR).
# Rationale: SACfD trains task_level=0 first (lift-only), so BC's L2 SR
# is gated by release/visual conditions that L0 doesn't even evaluate.
# What matters for the L0 RL start is whether BC reliably lifts.
# Idempotent: skip if existing log already shows a passing lift_history.
EVAL_LOG="$CHAIN_ROOT/step3_bc_eval.log"
skip_step3=0
if [ -f "$EVAL_LOG" ] && grep -q "SUCCESS RATE:" "$EVAL_LOG"; then
    existing_lift=$(extract_condition_rate "$EVAL_LOG" "lift_history")
    if [ -n "$existing_lift" ]; then
        existing_ok=$(awk -v l="$existing_lift" 'BEGIN{print (l+0 >= 70.0) ? 1 : 0}')
        if [ "$existing_ok" -eq 1 ]; then
            existing_sr=$(extract_success_rate "$EVAL_LOG")
            log "STEP 3: skip — existing eval has lift_history=${existing_lift}% >= 70% (SR=${existing_sr}%)"
            set_status "STEP3 skipped lift=${existing_lift}%"
            skip_step3=1
        fi
    fi
fi
if [ "$skip_step3" -ne 1 ]; then
    log "STEP 3: BC eval gate (stage 0 narrow, task_level 2, 100 ep)"
    set_status "STEP3 BC eval"
    run_python_with_killer "bc_eval" "$EVAL_LOG" "SUCCESS RATE:" 1800 \
        env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
        python scripts/eval_policy.py \
            --ckpt "$BC_DIR/bc.pt" \
            --episodes 100 \
            --curriculum-stage 0 \
            --task-level 2 \
            --seed 0
    rc=$?
    if [ "$rc" -ne 0 ]; then
        abort_chain "BC eval failed (rc=$rc), see $EVAL_LOG"
    fi
    sr=$(extract_success_rate "$EVAL_LOG")
    lift=$(extract_condition_rate "$EVAL_LOG" "lift_history")
    log "  BC eval SR=${sr}% lift_history=${lift}%"
    gate_ok=$(awk -v l="$lift" 'BEGIN{print (l+0 >= 70.0) ? 1 : 0}')
    if [ "$gate_ok" -ne 1 ]; then
        abort_chain "BC eval gate FAIL: lift_history=${lift}% < 70% (BC cannot reliably lift — SACfD L0 won't ignite)"
    fi
    set_status "STEP3 done lift=${lift}% SR=${sr}%"
fi

# -----------------------------------------------------------------------
# Step 4: SACfD smoke task_level 0 (50k step, gate: max ep_return > 0).
# F4 escalation from F3 (per BC stochastic sweep: BC at SAC mid-noise
# std=0.10 has lift_history=61%, so the starting point is fine — F3 lift=0%
# was SAC actively breaking the BC actor, not BC brittleness). Fixes:
#   - bc_w 0.3 → 0.7 (anchor harder to BC manifold)
#   - bc_anneal 50k → 300k (smoke 50k effectively no anneal)
#   - target_entropy -1 → -0.5 (force more exploration)
#   - init_alpha 0.05 → 0.1
#   - actor_lr 3e-4 → 1e-4 (slower BC manifold departure)
#   - sac.py adds alpha hard clip (log_alpha ≥ log(0.02)) → no alpha collapse
#   - sac.py logs pi_std (actor's actual sampling noise) for diagnostic
# Idempotency: skip STEP 4 if smoke ckpt exists AND prior log shows passing
# gate (max ep_return > 0). Saves ~70 min on chain re-runs after reboot or
# transient stops. Re-run smoke from scratch by deleting $SAC_L0_SMOKE/sac.pt.
SMOKE_L0_LOG="$CHAIN_ROOT/step4_sacfd_L0_smoke.log"
mkdir -p "$SAC_L0_SMOKE"
skip_step4=0
if [ -f "$SAC_L0_SMOKE/sac.pt" ] && [ -f "$SMOKE_L0_LOG" ]; then
    existing_max=$(extract_ep_return_max "$SMOKE_L0_LOG")
    if [ -n "$existing_max" ]; then
        existing_ok=$(awk -v s="$existing_max" 'BEGIN{print (s+0 > 0.0) ? 1 : 0}')
        if [ "$existing_ok" -eq 1 ]; then
            log "STEP 4: skip — $SAC_L0_SMOKE/sac.pt exists with max ep_return=$existing_max"
            set_status "STEP4 skipped max_ret=$existing_max"
            ep_ret_max="$existing_max"
            skip_step4=1
        fi
    fi
fi
if [ "$skip_step4" -ne 1 ]; then
    log "STEP 4: SACfD smoke task_level 0 (50k step, F4)"
    set_status "STEP4 SACfD L0 smoke F4"
    run_python_with_killer "sacfd_L0_smoke" "$SMOKE_L0_LOG" "\[train_sac\] saved" 5400 \
        env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
        python scripts/train_sac.py \
            --bc-ckpt "$BC_DIR/bc.pt" \
            --demo-dir "$DEMO_DIR" \
            --run-name "$(basename "$SAC_L0_SMOKE")" \
            --total-steps 50000 \
            --learning-starts 5000 \
            --batch-size 256 \
            --bc-loss-weight 3.0 \
            --bc-anneal-steps 1000000 \
            --demo-ratio-initial 0.5 \
            --demo-ratio-final 0.4 \
            --demo-ratio-switch-step 25000 \
            --target-entropy -6.0 \
            --init-alpha 0.1 \
            --actor-lr 3e-5 \
            --no-entropy-term \
            --no-q-filter \
            --critic-warmup-steps 3000 \
            --task-level 0 \
            --curriculum-stage 0 \
            --video-every-steps 0 \
            --skip-distribution-check \
            --seed 0
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
        abort_chain "SACfD L0 smoke crashed (rc=$rc), see $SMOKE_L0_LOG"
    fi
    ep_ret_max=$(extract_ep_return_max "$SMOKE_L0_LOG")
    log "  SACfD L0 smoke max ep_return = $ep_ret_max"
    gate_ok=$(awk -v s="$ep_ret_max" 'BEGIN{print (s+0 > 0.0) ? 1 : 0}')
    if [ "$gate_ok" -ne 1 ]; then
        abort_chain "SACfD L0 smoke gate FAIL: max ep_return=$ep_ret_max <= 0 (BC anchor + no-HER not igniting either)"
    fi
    set_status "STEP4 done max_ret=$ep_ret_max"
fi

# -----------------------------------------------------------------------
# Step 5: SACfD full task_level 0 (300k step + video sampler).
# F3 same hyperparams as STEP 4 but 300k step + longer bc anneal.
# Idempotency: skip STEP 5 if full ckpt already exists. Re-run by deleting
# $SAC_L0_FULL/sac.pt.
mkdir -p "$SAC_L0_FULL"
FULL_L0_LOG="$CHAIN_ROOT/step5_sacfd_L0_full.log"
if [ -f "$SAC_L0_FULL/sac.pt" ]; then
    log "STEP 5: skip — $SAC_L0_FULL/sac.pt already exists"
    set_status "STEP5 skipped"
else
    log "STEP 5: SACfD full task_level 0 (300k step, F4)"
    set_status "STEP5 SACfD L0 full F4"
    run_python_with_killer "sacfd_L0_full" "$FULL_L0_LOG" "\[train_sac\] saved" 28800 \
        env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
        python scripts/train_sac.py \
            --bc-ckpt "$BC_DIR/bc.pt" \
            --demo-dir "$DEMO_DIR" \
            --run-name "$(basename "$SAC_L0_FULL")" \
            --total-steps 300000 \
            --learning-starts 5000 \
            --batch-size 256 \
            --bc-loss-weight 3.0 \
            --bc-anneal-steps 1000000 \
            --demo-ratio-initial 0.5 \
            --demo-ratio-final 0.4 \
            --demo-ratio-switch-step 50000 \
            --target-entropy -6.0 \
            --init-alpha 0.1 \
            --actor-lr 3e-5 \
            --no-entropy-term \
            --no-q-filter \
            --critic-warmup-steps 3000 \
            --task-level 0 \
            --curriculum-stage 0 \
            --video-every-steps 20000 \
            --skip-distribution-check \
            --seed 0
    rc=$?
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
        abort_chain "SACfD L0 full crashed (rc=$rc), see $FULL_L0_LOG"
    fi
    if [ ! -f "$SAC_L0_FULL/sac.pt" ]; then
        abort_chain "SAC L0 ckpt not produced at $SAC_L0_FULL/sac.pt"
    fi
    set_status "STEP5 done"
fi

# -----------------------------------------------------------------------
# Step 6: Eval task_level 0 (100 ep, deterministic).
# Gate: lift_history rate >= 70% (level-0 advance criterion).
# Idempotency: skip STEP 6 if existing eval log already shows passing
# lift_history (>= 70%). Re-run by deleting $EVAL_L0_LOG.
EVAL_L0_LOG="$CHAIN_ROOT/step6_eval_L0.log"
skip_step6=0
if [ -f "$EVAL_L0_LOG" ] && grep -q "SUCCESS RATE:" "$EVAL_L0_LOG"; then
    existing_lift=$(extract_condition_rate "$EVAL_L0_LOG" "lift_history")
    if [ -n "$existing_lift" ]; then
        existing_ok=$(awk -v l="$existing_lift" 'BEGIN{print (l+0 >= 70.0) ? 1 : 0}')
        if [ "$existing_ok" -eq 1 ]; then
            log "STEP 6: skip — existing eval has lift_history=${existing_lift}% >= 70%"
            set_status "STEP6 skipped lift=${existing_lift}%"
            lift_rate="$existing_lift"
            skip_step6=1
        fi
    fi
fi
if [ "$skip_step6" -ne 1 ]; then
    log "STEP 6: Eval SACfD L0 (task_level 0, 100 ep)"
    set_status "STEP6 eval L0"
    run_python_with_killer "eval_L0" "$EVAL_L0_LOG" "SUCCESS RATE:" 1800 \
        env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
        python scripts/eval_policy.py \
            --ckpt "$SAC_L0_FULL/sac.pt" \
            --episodes 100 \
            --curriculum-stage 0 \
            --task-level 0 \
            --seed 0
    rc=$?
    if [ "$rc" -ne 0 ]; then
        abort_chain "Eval L0 failed (rc=$rc), see $EVAL_L0_LOG"
    fi
    sr_l0=$(extract_success_rate "$EVAL_L0_LOG")
    lift_rate=$(extract_condition_rate "$EVAL_L0_LOG" "lift_history")
    log "  Eval L0 SR=${sr_l0}% lift_history=${lift_rate}%"
    gate_ok=$(awk -v s="$lift_rate" 'BEGIN{print (s+0 >= 70.0) ? 1 : 0}')
    if [ "$gate_ok" -ne 1 ]; then
        abort_chain "L0 advance gate FAIL: lift_history=${lift_rate}% < 70% (cannot advance to L1)"
    fi
    set_status "STEP6 done lift=${lift_rate}%"
fi

# -----------------------------------------------------------------------
# Step 7: SACfD smoke task_level 1 (50k step).
# Level transition guards (docs/g_plus_design.md §3):
#   - actor logstd implicitly reset by re-init from L0 ckpt; we load
#     the L0 sac.pt and the trainer's load_bc_actor path resets std to
#     -1.0 via the GaussianActor's logstd_head bias init. Demo ratio
#     temporarily up.
# Idempotency: skip STEP 7 if smoke ckpt + passing gate already exists.
mkdir -p "$SAC_L1_SMOKE"
SMOKE_L1_LOG="$CHAIN_ROOT/step7_sacfd_L1_smoke.log"
skip_step7=0
if [ -f "$SAC_L1_SMOKE/sac.pt" ] && [ -f "$SMOKE_L1_LOG" ]; then
    existing_max=$(extract_ep_return_max "$SMOKE_L1_LOG")
    if [ -n "$existing_max" ]; then
        existing_ok=$(awk -v s="$existing_max" 'BEGIN{print (s+0 > 0.0) ? 1 : 0}')
        if [ "$existing_ok" -eq 1 ]; then
            log "STEP 7: skip — $SAC_L1_SMOKE/sac.pt exists with max ep_return=$existing_max"
            set_status "STEP7 skipped max_ret=$existing_max"
            ep_ret_max_l1="$existing_max"
            skip_step7=1
        fi
    fi
fi
if [ "$skip_step7" -ne 1 ]; then
log "STEP 7: SACfD smoke task_level 1 (50k step, F4) — stage 2 (10cm spawn)"
set_status "STEP7 SACfD L1 smoke F4 stage2"
# F4 (L1): keep HER on — L1's stable_placement IS goal-dependent so HER
# relabel can produce success signal. Same BC-anchor / alpha-clip / slower
# actor_lr fixes as L0 to prevent Q overestimation + BC manifold drift.
# F15p (post-2026-05-20 trivial-spawn fix): stage 0 (2cm) put cube spawn
# already inside goal radius (3cm) → stable_placement trivially True at
# reset → L1 collapses to lift-only. Lifted to stage 1 (5cm spawn) so
# place IS goal-dependent for half the episodes; HER can produce signal.
run_python_with_killer "sacfd_L1_smoke" "$SMOKE_L1_LOG" "\[train_sac\] saved" 5400 \
    env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt "$BC_DIR/bc.pt" \
        --demo-dir "$DEMO_DIR" \
        --run-name "$(basename "$SAC_L1_SMOKE")" \
        --total-steps 50000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 3.0 \
        --bc-anneal-steps 1000000 \
        --her \
        --her-k-future 4 \
        --her-ratio-initial 0.4 \
        --her-ratio-final 0.3 \
        --her-ratio-switch-step 25000 \
        --demo-ratio-initial 0.4 \
        --demo-ratio-final 0.3 \
        --demo-ratio-switch-step 25000 \
        --target-entropy -6.0 \
        --init-alpha 0.1 \
        --actor-lr 3e-5 \
        --no-entropy-term \
        --no-q-filter \
        --critic-warmup-steps 3000 \
        --task-level 1 \
        --curriculum-stage 2 \
        --video-every-steps 0 \
        --skip-distribution-check \
        --seed 0
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    abort_chain "SACfD L1 smoke crashed (rc=$rc), see $SMOKE_L1_LOG"
fi
ep_ret_max_l1=$(extract_ep_return_max "$SMOKE_L1_LOG")
log "  SACfD L1 smoke max ep_return = $ep_ret_max_l1"
gate_ok=$(awk -v s="$ep_ret_max_l1" 'BEGIN{print (s+0 > 0.0) ? 1 : 0}')
if [ "$gate_ok" -ne 1 ]; then
    abort_chain "SACfD L1 smoke gate FAIL: max ep_return=$ep_ret_max_l1 <= 0"
fi
set_status "STEP7 done max_ret=$ep_ret_max_l1"
fi  # end skip_step7

# -----------------------------------------------------------------------
# Step 8: SACfD full task_level 1 (300k step + video sampler).
# Idempotency: skip STEP 8 if full L1 ckpt exists.
mkdir -p "$SAC_L1_FULL"
FULL_L1_LOG="$CHAIN_ROOT/step8_sacfd_L1_full.log"
if [ -f "$SAC_L1_FULL/sac.pt" ]; then
    log "STEP 8: skip — $SAC_L1_FULL/sac.pt already exists"
    set_status "STEP8 skipped"
else
log "STEP 8: SACfD full task_level 1 (300k step, F4) — stage 2 (10cm spawn)"
set_status "STEP8 SACfD L1 full F4 stage2"
run_python_with_killer "sacfd_L1_full" "$FULL_L1_LOG" "\[train_sac\] saved" 28800 \
    env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt "$BC_DIR/bc.pt" \
        --demo-dir "$DEMO_DIR" \
        --run-name "$(basename "$SAC_L1_FULL")" \
        --total-steps 300000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 3.0 \
        --bc-anneal-steps 1000000 \
        --her \
        --her-k-future 4 \
        --her-ratio-initial 0.4 \
        --her-ratio-final 0.3 \
        --her-ratio-switch-step 50000 \
        --demo-ratio-initial 0.4 \
        --demo-ratio-final 0.3 \
        --demo-ratio-switch-step 50000 \
        --target-entropy -6.0 \
        --init-alpha 0.1 \
        --actor-lr 3e-5 \
        --no-entropy-term \
        --no-q-filter \
        --critic-warmup-steps 3000 \
        --task-level 1 \
        --curriculum-stage 2 \
        --video-every-steps 20000 \
        --skip-distribution-check \
        --seed 0
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    abort_chain "SACfD L1 full crashed (rc=$rc), see $FULL_L1_LOG"
fi
if [ ! -f "$SAC_L1_FULL/sac.pt" ]; then
    abort_chain "SAC L1 ckpt not produced at $SAC_L1_FULL/sac.pt"
fi
set_status "STEP8 done"
fi  # end skip_step8

# -----------------------------------------------------------------------
# Step 9: SACfD smoke task_level 2 (50k step) — full PnP (5-condition AND).
# task_level=2 reward = report.success = lift_history AND stable_placement
# AND release_retreat AND velocity_stability AND visual_agreement.
# L1 baseline showed release_retreat 95% emerge naturally → L2 transition
# should work. visual_agreement uses GT-pose proxy (Phase 1 결정사항 6).
# Idempotency: skip STEP 9 if smoke ckpt exists with passing gate.
mkdir -p "$SAC_L2_SMOKE"
SMOKE_L2_LOG="$CHAIN_ROOT/step9_sacfd_L2_smoke.log"
skip_step9=0
if [ -f "$SAC_L2_SMOKE/sac.pt" ] && [ -f "$SMOKE_L2_LOG" ]; then
    existing_max=$(extract_ep_return_max "$SMOKE_L2_LOG")
    if [ -n "$existing_max" ]; then
        existing_ok=$(awk -v s="$existing_max" 'BEGIN{print (s+0 > 0.0) ? 1 : 0}')
        if [ "$existing_ok" -eq 1 ]; then
            log "STEP 9: skip — $SAC_L2_SMOKE/sac.pt exists with max ep_return=$existing_max"
            set_status "STEP9 skipped max_ret=$existing_max"
            ep_ret_max_l2="$existing_max"
            skip_step9=1
        fi
    fi
fi
if [ "$skip_step9" -ne 1 ]; then
log "STEP 9: SACfD smoke task_level 2 (50k step, F4) — stage 2 (10cm spawn)"
set_status "STEP9 SACfD L2 smoke F4 stage2"
run_python_with_killer "sacfd_L2_smoke" "$SMOKE_L2_LOG" "\[train_sac\] saved" 5400 \
    env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt "$BC_DIR/bc.pt" \
        --demo-dir "$DEMO_DIR" \
        --run-name "$(basename "$SAC_L2_SMOKE")" \
        --total-steps 50000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 3.0 \
        --bc-anneal-steps 1000000 \
        --her \
        --her-k-future 4 \
        --her-ratio-initial 0.4 \
        --her-ratio-final 0.3 \
        --her-ratio-switch-step 25000 \
        --demo-ratio-initial 0.4 \
        --demo-ratio-final 0.3 \
        --demo-ratio-switch-step 25000 \
        --target-entropy -6.0 \
        --init-alpha 0.1 \
        --actor-lr 3e-5 \
        --no-entropy-term \
        --no-q-filter \
        --critic-warmup-steps 3000 \
        --task-level 2 \
        --curriculum-stage 2 \
        --video-every-steps 0 \
        --skip-distribution-check \
        --seed 0
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    abort_chain "SACfD L2 smoke crashed (rc=$rc), see $SMOKE_L2_LOG"
fi
ep_ret_max_l2=$(extract_ep_return_max "$SMOKE_L2_LOG")
log "  SACfD L2 smoke max ep_return = $ep_ret_max_l2"
gate_ok=$(awk -v s="$ep_ret_max_l2" 'BEGIN{print (s+0 > 0.0) ? 1 : 0}')
if [ "$gate_ok" -ne 1 ]; then
    abort_chain "SACfD L2 smoke gate FAIL: max ep_return=$ep_ret_max_l2 <= 0 (5-condition AND too strict — consider warm-start from L1 sac.pt or relax visual_agreement proxy)"
fi
set_status "STEP9 done max_ret=$ep_ret_max_l2"
fi  # end skip_step9

# -----------------------------------------------------------------------
# Step 10: SACfD full task_level 2 (300k step + video sampler).
# Phase 1 final: 5-condition AND full PnP.
# Idempotency: skip STEP 10 if full L2 ckpt exists.
mkdir -p "$SAC_L2_FULL"
FULL_L2_LOG="$CHAIN_ROOT/step10_sacfd_L2_full.log"
if [ -f "$SAC_L2_FULL/sac.pt" ]; then
    log "STEP 10: skip — $SAC_L2_FULL/sac.pt already exists"
    set_status "STEP10 skipped"
else
log "STEP 10: SACfD full task_level 2 (600k step, F4) — stage 2, extended for lift-quality ceiling"
set_status "STEP10 SACfD L2 full F4 stage2 600k"
# 2026-05-21 evening reach diagnosis: L2 at 300k plateaus at lift_history 65% / lifted_rate 70%.
# BC reach mean 16mm vs L2 reach mean 18mm — reach is NOT the bottleneck.
# Real bottleneck is grasp/lift robustness, which SAC slowly improves via
# sparse signal (BC lift 49% → L2 lift 70%). Doubling to 600k gives the
# sparse-reward learner more time to climb past the plateau. Anneal also
# extended (1M → 2M) so bc anchor stays meaningful through the longer run.
# video-every-steps 40000 keeps 16 videos total instead of doubling.
run_python_with_killer "sacfd_L2_full" "$FULL_L2_LOG" "\[train_sac\] saved" 57600 \
    env CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y \
    python scripts/train_sac.py \
        --bc-ckpt "$BC_DIR/bc.pt" \
        --demo-dir "$DEMO_DIR" \
        --run-name "$(basename "$SAC_L2_FULL")" \
        --total-steps 600000 \
        --learning-starts 5000 \
        --batch-size 256 \
        --bc-loss-weight 3.0 \
        --bc-anneal-steps 2000000 \
        --her \
        --her-k-future 4 \
        --her-ratio-initial 0.4 \
        --her-ratio-final 0.3 \
        --her-ratio-switch-step 100000 \
        --demo-ratio-initial 0.4 \
        --demo-ratio-final 0.3 \
        --demo-ratio-switch-step 100000 \
        --target-entropy -6.0 \
        --init-alpha 0.1 \
        --actor-lr 3e-5 \
        --no-entropy-term \
        --no-q-filter \
        --critic-warmup-steps 3000 \
        --task-level 2 \
        --curriculum-stage 2 \
        --video-every-steps 40000 \
        --skip-distribution-check \
        --seed 0
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ]; then
    abort_chain "SACfD L2 full crashed (rc=$rc), see $FULL_L2_LOG"
fi
if [ ! -f "$SAC_L2_FULL/sac.pt" ]; then
    abort_chain "SAC L2 ckpt not produced at $SAC_L2_FULL/sac.pt"
fi
set_status "STEP10 done"
fi  # end skip_step10

# -----------------------------------------------------------------------
log "============================================================"
log "G+ CHAIN STEPS 1..10 COMPLETE"
log "  BC ckpt:  $BC_DIR/bc.pt"
log "  L0 ckpt:  $SAC_L0_FULL/sac.pt"
log "  L1 ckpt:  $SAC_L1_FULL/sac.pt"
log "  L2 ckpt:  $SAC_L2_FULL/sac.pt"
log "  Tail $STATUS_FILE for per-step gates"
log "============================================================"
set_status "FINISHED"
exit 0
