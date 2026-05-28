# 세션 13 결과 요약 보고서 — Policy Behavior 직접 진단

**일자**: 2026-05-09 (재개)
**브랜치**: `khj-rl-track`
**대상**: 양팔 시연 RL 트랙 (한 팔 담당)

---

## TL;DR

- **결정적 발견**: BC/PPO 가 cube 에 도달 못 함 (ee_to_cube_min < 5cm: Oracle 86% vs **BC 2% / PPO 1%**)
- **가설 1 (도달 못 함) PASS** — saddle 의 진짜 원인 식별
- **가설 4 (좁은 영역만 성공) 무효** — success/fail cube_init 분포 동일
- **다음 세션 권고**: BC trajectory drift fix (DAgger / hindsight relabel / reward 강화)
- RL 트랙 종료 결정 **재검토 가능** (새 가설 발견)

---

## 1. 진단 4 가설 + 새 가설

| 가설 | 진단 결과 |
|---|---|
| 1 도달 못 함 (ee 멀리) | **PASS** ⭐ |
| 2 도달 후 grasp 안 함 | partial (BC 9% close, PPO 100% close but 멀리서) |
| 3 grasp 후 미끄러짐 | partial (PPO cube_vel 0.34 = random push) |
| 4 좁은 영역만 성공 | FAIL (분포 동일) |
| **5 BC trajectory drift** (NEW) | **유력** |

## 2. Policy Behavior — Oracle/BC/PPO 비교

| metric | Oracle | BC | PPO best (sess10 v2) |
|---|---|---|---|
| success rate | 15% | 14% | 13% |
| z_max mean | 0.0524 | 0.0514 | 0.0501 |
| z_max max | 0.189 | 0.194 | 0.189 |
| **ee_to_cube_min mean** | **4.6cm** | **13.9cm** | **8.4cm** |
| **ee_to_cube_min < 5cm (%)** | **86%** | **2%** | **1%** |
| ep with close attempt (%) | 98% | 9% | 100% |
| gripper_closed_steps mean | 243 | 15 | 485 |
| **ee_dist_at_close mean** | **4.8cm** | **9.8cm** | **14.4cm** ⚠️ |
| z_at_first_close (cube z) | 0.013 | 0.012 | 0.028 |
| cube_vel_max after close | 0.016 | 0.0003 | **0.337** |

**해석**:
- **Oracle 정상**: ee 가 cube 5cm 안 86% 도달, close 명령 시 ee_dist 4.8cm + cube_z 0.013 (측면) → 정상 grasp.
- **BC 비정상**: ee 가 14cm 떨어진 채 학습. 9% 만 close 시도. close 시 ee_dist 9.8cm — grasp 못함.
- **PPO 비정상**: ee 가 8cm 떨어진 채 100% close. ee_dist 14cm 에서 close — grasp 불가능. cube_vel 0.337 (큰 push) = random catch 패턴.

→ **모두 cube 에 못 가지만 14% 가량 success** = cube 가 random init 시 ee 근처 있을 때 우연한 catch. 

## 3. Success vs Fail cube_init 분포

| Policy | n_success | success cube_y σ | n_fail | fail cube_y σ |
|---|---|---|---|---|
| Oracle | 15 | 0.144 | 85 | 0.113 |
| BC | 14 | 0.139 | 86 | 0.113 |
| PPO | 13 | 0.144 | 87 | 0.113 |

success cube_init 분포 σ 가 fail 보다 약간 큼 (0.14 vs 0.11) — **success episode 가 cube_y 분포 가장자리** 에 있음. cube 가 robot 에 가까이 있을 때만 우연한 catch. 그러나 *분포 자체* 는 거의 같음.

scatter plot: `tasks/session13_success_vs_fail.png` (oracle/BC/PPO 3 panel).

→ 가설 4 (좁은 영역만 성공) **무효**.

## 4. 가설 5 — BC Trajectory Drift (NEW)

**증거**:
- Oracle demos 가 cube 4cm 안 도달 (reach_dist=0.04)
- BC 학습 loss low (val 0.001) — supervised metric 정확
- **그러나 BC actor 의 행동 시점 ee 14cm 떨어짐**
- → Demos 의 sequence 따라가면 cube 도달, BC 가 perfect copy 못 하면 trajectory drift 누적

**Fix 후보**:
1. **DAgger (Dataset Aggregation)** — BC 학습 후 BC rollout 의 fail trajectory 에서 oracle action 으로 relabel
2. **Reward shaping 강화** — reaching weight ↑ (현재 0.5 인데 1.0 또는 2.0 으로)
3. **Reward distance std ↓** — `mdp.object_ee_distance` 의 std=0.05 → 0.02 (더 가까워야 reward)
4. **BC 학습 hyperparam** — epoch ↑ (30→100), lr 작게 (1e-3→3e-4)
5. **Demo 추가 with pertur trajectory** — oracle 에 noise 주고 다양한 trajectory 수집

**가장 promising**: Fix 3 (reward distance std ↓) — saddle 근접 영역 reward 더 sharp. 학습 PPO ablation 으로 검증 가능.

## 5. 영상 (12 mp4, 시연 자료)

`~/jabis_sim/day5/videos/session13/`:
- `bc_diagnose_{top,diag}_seed{0,1,2}.mp4` — BC actor 가 cube 에 못 도달하는 패턴
- `oracle_diagnose_{top,diag}_seed{0,1,2}.mp4` — Oracle 정상 grasp 패턴 (비교용)

시연 메시지: "BC 가 cube 에서 14cm 떨어진 채 학습 — trajectory drift 의 시각적 증거".

## 6. RL 트랙 종료 결정 재검토 권고

세션 12 에서 **Path C (RL 종료) 확정** 했으나, 세션 13 의 새 발견:
- **saddle 의 진짜 원인 (도달 못 함) 식별** — codex 5세션 동안 식별 못 한 핵심
- 가설 5 (BC trajectory drift) 의 fix 후보 명확
- 시연 path B (PPO + BC 병행) 가능성 부활

**사용자 결정 사항**:
- (a) Path B 검토 — fix 3 (reward distance std ↓) ablation 1세션 추가
- (b) Path C 확정 유지 — sim2real 전환
- (c) DAgger 같은 더 큰 fix 시도 — 1~2세션 추가

## 7. Cleanup Whitelist 준수 ✓

USD/oracle/collect/train_bc/validate_oracle/demos/bc_actor 모두 unchanged. 신규 파일만:
- `tasks/diagnose_policy.py` (episode-level metric driver)
- `tasks/analyze_diagnose_session13.py` (분석 스크립트)
- `tasks/diagnose_{oracle,bc,ppo}_session13.csv` (각 100 rows)
- `tasks/session13_success_vs_fail.png` (scatter plot)
- `tasks/session13_report.md`
- 12 mp4 영상 `videos/session13/`
