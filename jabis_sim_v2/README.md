# jabis_sim_v2 — RL track restart (5/12 시작)

## paradigm
BC warmstart + PPO fine-tuning

## env
- conda: `jabis-sim` (IsaacLab 0.34.9 + torch 2.5.1+cu121)
- package: `~/isaac_so_arm101/` (editable, --no-deps)
- USD asset: `~/jabis_sim/usd/so101_pincopen.usd` (v1 mimic-fixed, 참조만)
- GPU: 1번 고정 (CUDA_VISIBLE_DEVICES=1 ~/.bashrc)

## v1 분리
- shared: env, IsaacLab, isaac_so_arm101 package, USD asset
- new (clean): env_cfg, reward, obs, action, train script, BC actor

## 일정
- Day 1 (5/12): env 셋업 [95% 완료, baseline 검증 보류]
- Day 2 (5/13): hang 해결 + Oracle demo 설계
- Day 3 (5/14): BC 학습 + sim 검증 / Oracle 실물 deploy 결정
- Day 4~5 (5/15~16): PPO fine-tuning
- Day 6~7 (5/17~18): 실물 deploy
- Day 8~10 (5/19~21): 통합/리허설/D-day
