# Day 1 핸드오프 (5/12 02:20)

## ✅ 완료
- conda env jabis-sim 활성
- IsaacLab 0.34.9 + torch 2.5.1+cu121 + isaac_so_arm101 1.2.0 (--no-deps)
- flatdict install
- CUDA_VISIBLE_DEVICES=1 영구 (~/.bashrc)
- v2 디렉토리 + git init

## ⚠️ 보류 — Day 2 첫 작업
create_empty.py --headless hang 2회 (omni.physx.fabric 의심)

### Day 2 우회 순서
1. AppLauncher 단독 검증:
   python -c "from isaaclab.app import AppLauncher; import argparse; p=argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a=p.parse_args(['--headless']); l=AppLauncher(a); print('OK'); l.app.close()"

2. v1 진입점으로 검증:
   cd ~/isaac_so_arm101
   python scripts/list_envs.py 2>&1 | head -20

3. 위 hang이면 v1 isaaclab env에서 동일 시도 → 차이 분석

## fact
- 할당 GPU: 1번 (L40S 46GB)
- IsaacLab: ~/jabis_sim/IsaacLab (HEAD v2.3.0)
- v1 코드: ~/isaac_so_arm101 (khj-rl-track 브랜치)
- v1 USD: ~/jabis_sim/usd/so101_pincopen.usd

## 충돌 위험 (잊지 말기)
- 새 터미널 = isaaclab env로 떨어질 수 있음 (.bashrc 확인 안 함)
- v2 작업 = 무조건 `conda activate jabis-sim` 먼저

---

## 5/12 02:55 업데이트 — hang 해결됨

### 원인
- `CUDA_VISIBLE_DEVICES` 미설정 시 IsaacLab이 GPU 0번 자동 선택
- 다른 user 프로세스와 충돌 → hang
- 해결: `~/.bashrc` 에 `CUDA_VISIBLE_DEVICES=1` 영구 설정

### 검증 성공 명령
```bash
python -c "
from isaaclab.app import AppLauncher
import argparse
p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(['--headless'])
launcher = AppLauncher(a)
print('AppLauncher OK')
launcher.app.close()
"
```
→ `AppLauncher OK`, exit 0

### Day 2 첫 작업
v2 task 환경 설계 (env_cfg + cube spawn + reset)

---

## 5/12 03:20 — Day 2 첫 작업으로 이월

### v2 package 인프라 완성
- src/jabis_sim_v2/ editable install (--no-deps)
- env_cfg.py 첫 버전 작성 (CubeLiftEnvCfg, CubeLiftSceneCfg, ActionsCfg 등)
- mdp/terminations.py 더미

### 진행 막힌 지점
env_cfg 검증 시 AppLauncher 부팅 후 P2P 검증 뒤 **import 코드 도달 못 함**.
- 어제는 같은 명령에서 `AppLauncher OK` 보였음 (`tail -20`)
- 오늘은 같은 명령 + `tee` 에서 P2P 후 출력 0
- 동일 환경 재현성 불안정 — 새벽 추가 디버깅 ROI 낮음

### Day 2 진단 순서
1. AppLauncher 단독 다시 (어제 작동했던 명령 그대로)
2. 차이 시 카운트: 새벽 vs 낮 GPU 1번 점유 상태 다른가? (`nvidia-smi -i 1`)
3. fix 후 env_cfg import 검증 재시도

---

## 5/12 새벽 완료

### 완성 (commit 함)
- v2 env_cfg + mdp 완성
- joint_pos_env_cfg (SoArm101 + JointPos + BinaryJointPosition action)
- Oracle 5-state machine 작동 (state machine 통과 빠름, ~50 step LIFT 도달)
- URDF collision 추가 + USD reconvert 성공
- Mimic 의존 최소화 (left_proximal only)
- 단계적 close (0 → -1 over 8 steps)

### 미해결 (구조적 한계)
- Oracle cube lift success 0% (v19 saddle 그대로)
- 원인 3개 lessons_from_v1.md 에 정리

### 5/12 낮 결정
- (γ) v1 demo + BC v2 학습 OR
- 메인 트랙 (Oracle 실물 IK) 집중
- 새벽 시점은 RL/Oracle sim 한계 인정, IL 트랙이 시연 담보

---

## 5/12 morning — saddle 해결

### 새 성과
- NullspaceIK 직접 구현 (`src/.../oracle/ik/nullspace_ik.py`)
- PD stiffness override (env_cfg)
- URDF mimic 제거 + USD reconvert
- 4 finger explicit close

### 결과
**success rate: 12.5% (1/8 env, cube 11cm lift)**

### 다음
- success rate 더 올리기
- close timing / spawn range / lift trajectory tuning

---

## 5/12 BC 학습 진척

### 인프라
- scripts/collect_demos.py
- scripts/train_bc.py (v1 copy)
- scripts/eval_bc.py

### 산출물
- tasks/demos_v2.pt (30 episodes, 1.4MB)
- tasks/bc_init_v2.pt (BC actor state_dict)

### 결과
- BC success rate: 57.3% (sim)
- 다음: PPO warmstart (bc_init.pt 로드 후 PPO 학습)
