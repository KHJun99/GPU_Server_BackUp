# Jabis sim — Grip ablation (12cm lift 회복 시도)

날짜: 2026-05-13
스크립트: `scripts/eval_oracle_lift.py`
공통 조건: `SoArm101PickPlaceEnvCfg`, num_envs=64, n_trials=3 (총 192 env), steps=480 (full 8s episode), lift_dz=0.15, success_z=0.10
사전 컨텍스트: `tasks/oracle_lift_validation.md` (Case C — 환경 한계 확정)

## 1. 4-way ablation 결과

| Variant | finger PD | cube friction | close_steps | max_z med | max_z mean | >5cm | >7cm | >10cm | **>12cm** | LIFT 도달 | MOVE_TO_GOAL |
|---------|-----------|---------------|-------------|-----------|------------|------|------|-------|-----------|-----------|--------------|
| Baseline       | 50/10  | 3.0 | 60  | 0.050 | 0.058 | 9.9%  | 8.3% | 7.8% | **7.8%** | 28.6% | 7.8% |
| Ablation A (finger++) | 200/30 | 3.0 | 120 | 0.050 | 0.056 | 8.9%  | 5.7% | 5.7% | **5.7%** | 26.6% | 5.7% |
| Ablation B (friction++) | 50/10  | 5.0 | 60  | 0.050 | 0.059 | 13.5% | 8.3% | 7.3% | **7.3%** | 22.9% | 6.8% |
| Ablation C (A+B)      | 200/30 | 5.0 | 120 | 0.050 | 0.060 | 12.0% | 8.3% | 6.8% | **6.8%** | 20.3% | 6.8% |

**State 분포 (마지막 step)**: 4 variant 모두 APPROACH stuck **~70%** 동일 (68.8~72.9%).

원본 로그:
- `/tmp/oracle_baseline_480.log`
- `/tmp/oracle_ablation_A.log`
- `/tmp/oracle_ablation_B.log`
- `/tmp/oracle_ablation_C.log`

## 2. 결론

**Case C — 환경 한계 재확정. RL 트랙 종료, Oracle 기반 BC/diffusion 진행.**

근거:
- 4가지 variant 모두 12cm 성공률 **5.7~7.8% (delta ±2%pt)**. 모두 통계 노이즈 범위 안.
- 최선 변종 = **Baseline 7.8%** (finger 강화도, friction 증가도, 결합도 baseline을 뛰어넘지 못함).
- Ablation A에서 12cm가 오히려 떨어진 이유: close_steps 120으로 인한 시간 손실(LIFT 진입 시점 지연)이 finger 효과를 상쇄.
- friction 5.0 증가도 무효 — 애초에 cube를 잡는 데 실패한 케이스가 majority라 friction은 작용 기회 없음.
- 472 step (8s episode 끝)에서도 APPROACH stuck ~70%. steps=300 (5s)와 거의 동일 → IK reach 자체 한계지 시간 문제 아님.

## 3. 근본 원인 분석 (state machine 흐름)

전체 192 env 기준 (Baseline):
```
spawn (192) ─ APPROACH ──▶ DESCEND (31% pass)  # 132/192 stuck = IK reach 실패
                          ├──CLOSE (31% pass)
                          ├──LIFT (28.6% 도달)
                          └─12cm 성공 (7.8%)     # LIFT 도달자 중 27%만 실제 들어올림
```

병목 우선순위:
1. **IK reach 실패 (~70%)** — Oracle NullspaceIK가 cube 위 10cm waypoint에 도달 못함. cube spawn 범위 끝단(x=0.34, y=±0.10)에서 SO-ARM101 작업 영역 한계로 추정.
2. **Grip 실패 (~70% of LIFT)** — 41~55 env가 LIFT 도달했지만 그 중 27%만 cube 12cm 띄움. 나머지는 finger가 cube에서 미끄러짐. 이건 finger PD나 friction 변경으로 해결 안 됨 (gripper jaw 형상/접촉면 한계 = USD/물리 한계).

두 병목 모두 actuator/material 파라미터 영역 밖. **USD 수정 또는 robot 변경** 필요.

## 4. 다음 step 액션 (Case C 기준)

### 즉시 (오늘~내일)
1. **RL 트랙 공식 종료**. `train_pnp_*.py` 류 스크립트 archive. Tensorboard log 5/13 12:44, 15:11 dead policy 보존.
2. **Oracle policy를 fixed expert로 채택** — `OraclePolicy`를 production demo-collection 도구로 정리. 현재 LIFT 도달 28%, 12cm 도달 8%라도 단일 expert로는 충분.
3. **Oracle 성능 개선 ablation 한 번 더** — NullspaceIK damping(0.05→0.02)/max_dq(0.30→0.50) 튜닝으로 APPROACH stuck 70% 해소 가능한지 1시간 측정. 이건 환경 변경 없이 Oracle 내부만 손대는 거라 별개.

### 단기 (1주)
4. **BC/Diffusion policy 학습 파이프라인 구축** — Oracle 12cm 성공 케이스(~15/192)만 필터링해서 demo로 저장. `lerobot` 형식 (already on board) 또는 자체 dataset format. 200~500 demo 모으면 학습 시작 가능.
5. **Sim2real 우선 검증** — Oracle 12cm 성공 케이스를 real SO-ARM101에서 replay. sim 12cm 성공인데 real 실패면 friction/물성 mismatch 추가 조사.

### 보류
- USD 재변환은 큰 일정 영향, 대안(Oracle+BC) 우선 검증 후 결정.

## 5. 보조 진단

- steps 300 → 480 효과: Baseline 12cm 5.2% → 7.8% (+2.6%pt). 시간 효과 미미. 진짜 병목은 reach 정확도지 시간 아님.
- max_z median 0.050m = cube 초기 z(0.025+큐브 반 0.025=0.05) 그대로. 절반 이상 env에서 cube 1mm도 못 움직임. friction 증가의 의미가 없는 이유.
- final cube xy median (0.265, -0.00): 모든 variant 동일. cube가 spawn 분포에서 거의 정지 = Oracle이 옮기지 못함.

---

**핵심 결론**: grip ablation 효과 없음 — 12cm 성공률 5.7~7.8% (baseline 대비 ±2%pt 노이즈). RL 트랙 종료, Oracle expert 단독 + BC/diffusion 학습으로 전환.

파일: `/home/j-k14d101/jabis_sim_v2/tasks/oracle_grip_ablation.md`

## Memory 업데이트용 정보

```
Title: Grip ablation 무효, 환경 한계 재재확정
Type: project
Body: 2026-05-13 finger PD (50/10→200/30), cube friction (3.0→5.0), close_steps (60→120), 그리고 결합까지 4-way ablation 결과 모두 12cm 성공률 5.7~7.8% (baseline 7.8% 대비 delta ±2%pt). 근본 원인은 IK reach 실패 70% (cube 위 10cm waypoint 미도달) + LIFT 도달자 grip 실패 70% (gripper jaw 미끄러짐). actuator/material 파라미터로 해결 불가. RL 트랙 종료, Oracle expert + BC/diffusion 전환 결정. 보고서: tasks/oracle_grip_ablation.md.
```
