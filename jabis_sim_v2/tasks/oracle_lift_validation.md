# Jabis RL — Oracle policy lift validation

날짜: 2026-05-13
스크립트: `scripts/eval_oracle_lift.py`
환경: `SoArm101PickPlaceEnvCfg` (학습용과 동일), episode_length=8s
측정 조건: num_envs=64, n_trials=3 → 총 192 envs/조건, steps=300 (=5s sim)
gripper PD: stiffness=500, damping=80 (이미 강화된 값)

## 1. 4-way lift_dz sweep 결과

| lift_dz | max_z median | max_z mean | max_z p90 | >5cm | >7cm | >10cm | >12cm | LIFT+ 도달 | MOVE_TO_GOAL 도달 |
|---------|-------------:|-----------:|----------:|-----:|-----:|------:|------:|----------:|------------------:|
| 0.08    | 0.048        | 0.047      | 0.048     | 5.7% | 0.0% | 0.0%  | 0.0%  | 21.9%     | 0.0%              |
| 0.12    | 0.048        | 0.050      | 0.048     | 5.7% | 5.2% | 4.7%  | 0.0%  | 21.4%     | 4.7%              |
| 0.15    | 0.048        | 0.052      | 0.048     | 5.7% | 5.2% | 5.2%  | **5.2%** | 21.4%     | 5.2%              |
| 0.20    | 0.048        | 0.056      | 0.048     | 5.7% | 5.2% | 5.2%  | **5.2%** | 21.4%     | 5.2%              |

State machine 분포 (마지막 step, 4 조건 평균):
- APPROACH stuck: **~70%**
- DESCEND: ~4%
- CLOSE: ~4%
- LIFT: ~17%
- MOVE_TO_GOAL: 0~5% (lift_dz에 비례)

final cube xy median: (0.33~0.39, ~0) — cube가 spawn 분포 중앙에서 거의 안 움직임 (median이 spawn x mid 0.245~0.34 범위)

## 2. 결론

**Case C — 환경 한계 (12cm < 30%)**

- 가장 관대한 조건(lift_dz=0.20)에서도 12cm 도달은 **5.2%**.
- max_z median은 0.048m로 cube spawn z(0.025+큐브반 0.025)와 거의 동일 → 192 env 중 절반 이상이 cube를 1mm도 들지 못함.
- "Mode B (gripper jaw 물리 한계) USD 레벨 수정 불가" memory가 데이터로 재확인됨.

## 3. 병목 세분화 (state 분포 분석)

전체 192 env 기준 흐름 (lift_dz=0.12 기준):
```
spawn (192) ──APPROACH──▶ DESCEND (29% pass)  # 136/192 stuck
                         ├──CLOSE (17% pass)   # 8 stuck
                         └──LIFT (16% reach)   # 7 stuck
                            └─12cm 성공 (4.7%)  # LIFT 도달자 중 ~30%만 들어올림
```

병목 두 군데:
1. **APPROACH→DESCEND 진입 실패 (70% stuck)**: IK 정확도 또는 reach 시간 부족
   - 5초 sim (steps=300) 안에 cube 위 10cm 지점(reach_dist=2cm) 도달 못함
   - Episode는 8s까지인데 측정은 5s만 — 추가 측정 가치 있음
2. **LIFT 도달자 중 grip 실패 (~70%)**: LIFT 41개 도달 중 12cm 도달 10개. 나머지는 잡았다가 놓침
   - close_steps=60 후에도 cube 미끄러짐
   - finger_distal stiffness 50 / friction 3.0 도 부족

## 4. 다음 step 액션 (Case C 기준)

우선순위:

1. **steps=480 (전체 episode)로 재측정** — APPROACH stuck 중 일부는 시간 부족일 수 있음. 30분이면 검증.
2. **Reach 실패 분리 진단** — `scripts/eval_oracle_lift.py`에 episode 끝까지 측정 옵션 추가, APPROACH stuck env 의 cube xy 위치 vs 도달 성공 env 분포 비교. spawn x=0.34/y=±0.10 끝단에서 reach 실패율 높은지 확인. 1시간 분석.
3. **Grip 실패 ablation (Case B 시도)** — 별도 실험. close_steps 60→120, finger_distal stiffness 50→200/damping 10→40, friction 3.0→5.0. LIFT 도달자 중 12cm 비율이 30%→70%+로 오르면 RL 가능성 회복. 2시간.
4. **RL 재학습은 NO-GO** — 위 1~3 모두 호전 없으면 RL 자체 포기. Oracle 기반 demo collection + BC/diffusion policy 학습으로 트랙 전환.
5. **USD 재변환은 마지막 선택** — robot/cube SDF, friction, mimic joint 모두 ablation 후에도 안 되면 v1 그리퍼 모델 다시 처음부터 import. 일정 영향 큼.

## 5. 보조 분석

- LIFT state 도달 비율이 4개 lift_dz 모두 동일(21.4%) → state machine 진행은 lift_dz와 무관. lift_dz 효과는 LIFT 후 z 도달 한계만 변경 (0.08 → 8cm 한계 / 0.20 → 20cm 한계).
- lift_dz=0.12에서 12cm 0% / lift_dz=0.15에서 5.2% → 그립 성공한 case라도 lift_dz=0.12 target은 cube z=0.12에 ee=0.12로 맞춰지지만 gripper_link이 TCP+9cm 위라 실제 cube z는 0.12에 못 미침. lift_dz≥0.15가 진짜 12cm 큐브 lift에 필요한 target offset.
- MOVE_TO_GOAL 진입은 success_z=0.10 충족 필요. lift_dz=0.08에서 GOAL 0% = 8cm target으론 success_z 미달. lift_dz=0.12+ 에선 일부 진입.

---

**핵심 결론**: Oracle 12cm 성공률 5.2% (lift_dz=0.15). 환경 한계 확정. RL 재학습 전에 grip ablation (close_steps + finger PD + friction) 1회 시도 → 그래도 안 되면 RL 트랙 종료.

파일: `/home/j-k14d101/jabis_sim_v2/tasks/oracle_lift_validation.md`
