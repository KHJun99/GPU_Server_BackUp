# Day 2 (5/12) 마무리 — 내일 (5/13) 첫 작업 가이드

## 오늘 진짜 진척
- v1 12세션 saddle 풀음 (mimic 제거 + NullspaceIK + PD)
- v2 Oracle 31% 작동
- **BC pipeline 완성, 30ep BC = 51% (sim)**
- PPO warmstart 시도 → reward shaping 함정으로 실패
- BC 강화 시도 (500ep, MoG, ensemble, filter, state aug) 모두 30ep BC 못 넘김

## RL 트랙 결론
- ceiling: 30ep BC single = 51%
- 발표 자료로 진짜 좋은 메시지 (saddle 진단 + BC 한계 + PPO 함정)
- 추가 시간 ROI 낮음 → 메인 트랙 우선

## 내일 첫 작업 — 메인 트랙
**실물 Oracle IK orientation constraint 추가** (어제 5/11 6회 시도 실패):
- 매트 닿아서 DESCEND step 38+ stall 재현
- IK position-only 가 gripper 자세 제어 못함
- desired EE rotation: gripper z+ ↔ world z-
- 오른팔 rotation π 보정: x_real=-x_sim, y_real=-y_sim

## 진짜 우선순위 (5/13~5/14)
1. IK orientation constraint 코드 추가
2. 실물 cube grasp 시도 (1~2회)
3. **5/14 저녁 go/no-go**: cube only 시연 (T1) 또는 양팔 IL fallback

## 산출물 (commit 됨)
- `tasks/bc_init_v2.pt` (30ep BC, 51% sim)
- `tasks/demos_v2.pt` (30 oracle demos)
- `tasks/demos_v2_500ep.pt.bak` (보존, 미래 multimodal policy 학습용)
- `docs/lessons_from_v1.md` (모든 lessons)
- `src/jabis_sim_v2/tasks/cube_lift/mdp/observations.py` (ee_position_b, ee_to_cube 추가, env_cfg 는 미사용 상태)
