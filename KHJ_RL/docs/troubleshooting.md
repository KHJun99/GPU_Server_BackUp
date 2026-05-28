# Troubleshooting

KHJ_RL Phase 1 진행 중 발생한 문제와 해결 기록.
새 사고가 발생하면 카테고리에 맞춰 추가하고, 해결되면 commit hash로 cross-ref.

**Status 범례**:
- `RESOLVED` — 해결됨 + commit (또는 uncommitted patch)
- `IN-PROGRESS` — 가설 검증 / 실험 중
- `OPEN` — 발견됐지만 미해결 / 보류
- `RECURRING` — 해결됐지만 재발 가능 (운영상 주의)

**관련 설계 문서**:
- `docs/dense_reward_design.md` — A 옵션(dense shaping) 도입 완료 (2026-05-17), codex 검토 + weight 조정 반영
- `docs/dense_reward_weight_tuning.md` — A 결과별 weight 조정 matrix (시나리오 A/B/C/D)
- `docs/action_chunking_design.md` — ACT-style 사전 설계. A 결과 < 50% AND covariate shift 패턴 시 도입 검토
- `docs/dagger_design.md` — DAgger 사전 설계. A + ACT 둘 다 실패 시 마지막 카드
- `docs/phase2_migration_plan.md` — Phase 1 → Phase 2 vec env 이행 계획
- `docs/real_robot_deployment.md` — Phase 3 real robot 배포 계획 (Jetson + SO-ARM driver layer)

> 형식: **상태** → **증상** → **원인** → **해결** → `commit`

---

## Env / Sim 셋업

### 1. PincOpen 4-bar mimic 미동작
- **상태**: RESOLVED
- **증상**: gripper 닫아도 4-bar passive joint가 안 따라옴 (URDF에는 `<mimic>` 있었음)
- **원인**: URDF의 `<mimic>` 태그가 USD 변환에서 자동으로 넘어가지 않음
- **해결**: `scripts/usd_add_mimic_api.py`로 4개 passive joint에 `PhysxSchema.PhysxMimicJointAPI` 수동 stamp. reference=`gripper`, gearing=-1.0. mimic joint는 actuator에 넣지 않음 (drive 충돌)
- `322d54c`

### 2. `omni.physx.fabric` exact pin 충돌
- **상태**: RECURRING (Isaac Sim 설치 직후 매번 수동 패치 필요)
- **증상**: `use_fabric=True`로 부팅 시 raise. viewport가 env.reset 시점 USD에 멈춤
- **원인**: 번들된 `omni.physx.fabric-106.5.3`이 `omni.physx==106.5.3, exact=true`를 요구하는데 실제 코어는 106.5.7 → 옛 fabric으로 fallback → IPhysxPrivate ABI v0.2 vs v1.2 충돌
- **해결**: `extension.toml`에서 dependency 한 줄을 `exact=false`로 패치 (CLAUDE.md 설치 섹션의 `sed` 명령 참고). `.bak` 자동 백업

### 3. Pinocchio + Isaac Sim Assimp 심볼 충돌
- **상태**: RECURRING (새 entry-point script 추가할 때마다 같은 패턴 필요)
- **증상**: AppLauncher 후 `import pinocchio` 시 undefined symbol (`Assimp::IOSystem::CurrentDirectory`)
- **원인**: pinocchio의 `libhpp-fcl.so`가 시스템 Assimp에 링크. isaacsim이 먼저 로드되면 자체 Assimp 빌드가 바인딩됨 → C++ symbol mangling 충돌
- **해결**: 모든 entry-point script(`launch_viewer.py`, `collect_demos.py`, `train_ppo.py`, `eval_policy.py`)에서 **`AppLauncher` 전에 `import pinocchio` 먼저**. 시스템 Assimp가 먼저 바인딩되면 동거 가능
- `76f3520`

### 4. scene.py edit에서 `robot = robot_cfg` 줄 유실
- **상태**: RESOLVED
- **증상**: collect_demos 부팅 시 `KeyError: "Scene entity with key 'robot' not found"`
- **원인**: goal_marker 추가 Edit의 old_string에 `robot = robot_cfg`까지 포함됐고 new_string에서 빠뜨림 → robot이 scene cfg에서 사라짐
- **해결**: `robot = robot_cfg` 한 줄 복구. 디스크 파일 변경이라 이미 메모리에 로드된 PPO 학습은 영향 없음
- (uncommitted; 2026-05-17 fix)

---

## Demo collection

### 5. `oracle.done` 조기 종료
- **상태**: RESOLVED
- **증상**: 초기 sanity 수집에서 모든 demo가 fail로 기록됨
- **원인**: `collect_demos.py`가 `oracle.done`을 체크해 break했는데, oracle이 PnP 시퀀스 끝낸 직후 done=True로 flip → MultiConditionSuccess가 lift_history latch를 채우기 전에 episode 종료
- **해결**: `oracle.done` 기반 early break 제거. env가 `term`/`trunc` 신호로 자연스럽게 종료될 때까지 step
- `d88af6e`

### 6. Lift dwell 부족 → latch 미달
- **상태**: RESOLVED
- **증상**: oracle의 PnP는 잘 되는데 success rate 12-16% 정도
- **원인**: lift phase에서 cube가 z≥0.08에 머무는 시간이 12-16 step → `lift_history` latch buffer(20 step) 못 채움
- **해결**: oracle lift phase에 `min_steps=30` 추가. retreat target도 `[goal[0]-0.08, goal[1], _RETREAT_Z]`로 옆으로 빠지게 → top-cam occlusion proxy 회피
- `d4bc62d` (oracle success 100% 달성)

---

## BC training

### 7. `actions` vs `action` 단/복수 mismatch
- **상태**: RESOLVED
- **증상**: BC 첫 학습 시 `KeyError("action")`
- **원인**: writer는 `actions`(복수)로 NPZ에 저장하는데 `demo_buffer.py`가 `data["action"]`(단수)로 읽음
- **해결**: `demo_buffer.py`에서 `data["actions"]`로 통일. docstring schema도 일치시킴
- `a77b691`

### 8. BC 15% 성공률에서 PPO 0%로 후퇴
- **상태**: IN-PROGRESS (D 옵션 200k 정찰 학습 + B 옵션 demo 2000ep 추가 수집 병행 중)
- **증상**: 1k demo로 학습한 BC가 eval에서 15%, 그 ckpt로 PPO 500k 학습 후 eval 0/50
- **원인 (가설)**:
  - `ent_coef=0.01`이 BC actor_logstd를 키워 BC 행동 분포에서 표류 (entropy 2.51→2.78 상승 관찰)
  - `critic_warmup_iters=15` 동안 critic이 "value=0 상수" 학습 → warmup 끝난 후 actor가 노이즈 advantage로 업데이트
  - BC 15% 자체가 약함 — 1 PPO iter(~13 episode)에 success 1-2번이라 학습 신호 부족
- **D 옵션 검증 결과 (정정)**:
  - **`ent` 안정 하강 ≠ BC 보존**. iter=27 영상은 PnP 시도가 살아있었지만 **iter ~40 (step 81k) 부터 짓누르기로 변형**, iter ~53 (step 108k)에서도 짓누르기 지속. `ent` 하강은 actor가 한 행동(짓누르기)에 정착해 분포가 좁아진 것 — 행동 *방향*은 안 보호됨
  - 이전 학습(500k)은 iter ~150까지 PnP 살아있었는데 D는 iter ~40부터 변형 → **이전보다 더 빨리 망가졌을 가능성**
  - PPO D는 iter=65 (step 135k, 67.6%)에서 의도적 중단 — 짓누르기 attractor 강화만 남았음. ppo.pt 미저장
  - **결론**: D 단독으로 본질 sparse signal 부재 문제 해결 불가 (codex 예측 적중)
- **B 옵션 진행 중**: demo 2000ep 추가 수집 → BC retrain. collect_extra가 GPU 자유 후 페이스 2배 예상
- **후속 옵션 (B도 실패 시)**: A 옵션 = bounded·annealed dense shaping. 짓누르기 attractor 차단에 phase-gated 신호가 필요할 가능성 ↑. 코드 형태는 `docs/dense_reward_design.md`에 사전 설계됨 (도입 시 CLAUDE.md 결정사항 5 갱신 + codex 리뷰 필수)

---

## PPO training

### 9. Video sampler 미발화
- **상태**: RESOLVED
- **증상**: `--video-every-steps 5000`인데 mp4가 한 번도 안 떨어짐
- **원인**: `step % every_steps == 0` 체크가 num_steps(2048) 청크 단위와 일치하지 않음 → modulo가 영원히 0 안 됨
- **해결**: threshold-based `_last_capture_step` 사용. `step - _last >= every_steps`면 발화
- `8f28ca4`

### 10. obs_distribution degenerate dim 가짜 통과
- **상태**: RESOLVED
- **증상**: `normalized_t`처럼 reset 시 std=0인 dim에서 drift가 있어도 z-score 검사가 통과해버림
- **원인**: 분모가 ~0이라 z-score가 발산하거나 마스킹됨 → "constant" dim의 wrapper drift가 가장 진단적인데 그게 silent
- **해결**: `degenerate_std_threshold=1e-6`, `degenerate_abs_tol=1e-3`로 분기. degenerate dim은 절대 편차 검사
- `8f28ca4`

### 11. PPO eval policy에 `eval()` 컨텍스트 누락
- **상태**: RESOLVED
- **증상**: video_sampler가 호출하는 정책에서 dropout/batchnorm이 train 모드로 평가됨 (잠재적 문제)
- **원인**: `_make_eval_policy()`가 `net.eval()` 호출 안 함
- **해결**: `_make_eval_policy()` helper에 `eval()/train()` 컨텍스트 가드 추가
- `8f28ca4`

### 12. `enable_cameras` 누락
- **상태**: RESOLVED
- **증상**: `--video-every-steps>0`로 띄우면 `RuntimeError: A camera was spawned without the --enable_cameras flag`
- **원인**: `AppLauncher(headless=True)` 디폴트에서 카메라 sensor 생성 거부
- **해결**: `enable_cameras=args.video_every_steps > 0`로 조건부 활성화
- `5f11f95` 이후 train_ppo.py 수정

---

## Tooling / Infra

### 13. tmux libtinfo 충돌
- **상태**: RESOLVED (작업 시 매번 LD_LIBRARY_PATH 비우기 필요)
- **증상**: `tmux: error while loading shared libraries: libtinfo.so.6: version 'NCURSES6_TINFO_6.4.current' not found`
- **원인**: conda env의 LD_LIBRARY_PATH가 시스템 ncurses보다 우선 → tmux 바이너리가 conda ncurses 잘못 링크
- **해결**: `env LD_LIBRARY_PATH= /usr/bin/tmux ...`로 LD_LIBRARY_PATH 비우고 시스템 tmux 직접 호출

### 14. tmux send-keys Enter 타이밍
- **상태**: RESOLVED (운영 노하우)
- **증상**: PPO 학습 끝난 직후 명령어 입력은 됐는데 Enter가 안 먹혀서 명령 미실행
- **원인**: 학습 프로세스 종료 직후 bash로 떨어지는 타이밍 사이에 send-keys가 들어감
- **해결**: `Ctrl+C` 한 번 보내고 sleep 1s, 그다음 다시 명령 + Enter 보내기

### 15. video와 eval 결과 불일치 (진단 도구로 활용)
- **상태**: RESOLVED (mismatch는 의도된 동작, 진단 lens로 활용)
- **증상**: 학습 중간 mp4에서는 PnP 행동이 보이는데 마지막 ckpt eval은 0/50
- **원인**: video는 학습 진행 중 *그 시점의 weight*를 캡처. 학습 후반에 정책이 망가지면 초/중반 영상은 BC weight 살아있는 상태, eval은 망가진 마지막 ckpt
- **해결**: 시간순 영상(`rollout_step_00002048.mp4` → `..._00462848.mp4`)을 비교해 정책이 무너지는 지점 진단

---

### 16. `Path(list)` TypeError — train_bc.py CLI/main 불일치
- **상태**: RESOLVED (uncommitted; 2026-05-17 fix)
- **증상**: `TypeError: expected str, bytes or os.PathLike object, not list` at `train_bc.py:115`
- **원인**: 다중 demo-dir 지원으로 CLI를 `nargs="+"`로 바꿔서 `args.demo_dir`이 항상 `list[Path]`인데 `main()`은 여전히 `Path(args.demo_dir)`로 list 래핑 시도
- **해결**: `BCConfig(demo_dir=list(args.demo_dir), ...)`로 변경 (DemoBuffer.from_dir이 list/single 둘 다 받음)
- **교훈**: CLI 시그니처 변경 시 main()까지 grep 필수. 시그니처와 사용처가 분산된 경우 sanity test가 사전 catch에 효과적

---

## Tooling / Infra (계속)

### 17. Isaac Sim sim_app.close() hang (좀비 패턴)
- **상태**: RECURRING (Isaac Sim 4.5 좀비 종료 — process kill로 우회)
- **증상**: `scripts/collect_demos.py`, `scripts/eval_policy.py`가 `[SUMMARY]`/`SUCCESS RATE` 출력까지 정상 진행 후 `sim_app.close()`에서 hang. PID는 살아있지만 GPU 메모리 점유 + stdout 정지
- **원인 (가설)**: Isaac Sim 4.5의 carb singleton + omni.kit shutdown 순서가 깔끔하게 닫히지 않음. fabric / physx / kit 플러그인 unload 단계에서 멈춤. 4시간 25분 동안 좀비로 남은 사례 있음
- **해결 (운영 patten)**:
  - 작업 완료 표시(`[SUMMARY]` 또는 `SUCCESS RATE`)가 stdout에 찍힌 후에도 process 살아있으면 안전하게 SIGTERM 가능
  - pipeline의 `pgrep` 폴링 wait는 좀비 때문에 영원히 wait → 좀비 즉시 kill 필요
- **후속 대응 후보**:
  - eval_policy.py / collect_demos.py가 `sim_app.close()` 호출 후 timeout 기반 `os._exit(0)` fallback 추가
  - pipeline의 wait 로직을 process 죽음 + "[SUMMARY] / DONE 라인 stdout" OR 조건으로 변경

---

### 18. health_monitor zombie misclassification
- **상태**: RESOLVED (uncommitted; 2026-05-17 fix)
- **증상**: 자율화 패키지 launch 후 health_monitor 첫 폴링에서 진행 중인 train_ppo.py(PPO A) 두 개를 SIGTERM. 학습 ~30분 손실
- **원인**: `_kill_zombies`가 `runs/` 안 모든 ckpt를 글로벌하게 스캔. 이전 run인 `runs/stage0_ppo/ppo.pt`(9시간 old)을 기준으로 ckpt_age=32053s 계산 → 현재 살아있는 train_ppo.py 둘 다 zombie 오인 → kill
- **해결**: health_monitor의 글로벌 zombie kill 로직 비활성화. 좀비 처리는 *각 watcher가 자기 run dir 기준으로*만 — `_kill_zombies` 함수는 유지하되 main loop에서 호출 안 함
- **교훈**: 여러 run의 ckpt가 공존하는 디렉토리에서 "ckpt mtime 기반 zombie 판정"은 절대 글로벌하게 적용하지 말 것. scope를 명시적 run-name으로 한정

---

## 분류 안 된 사고

(새 사고 발생 시 위 카테고리에 못 넣으면 여기 임시 저장 후 분류)

---

## RL / Modeling (2026-05-18)

### 19. ACT inference trick이 정책 성능 95% 차지 (BC v2 90% → CB off 시 5%)
- **상태**: RESOLVED (진단)
- **증상**: BC ACT v2 ckpt가 `eval_policy.py` 기본 모드에서 stage 0 90% / stage 1 60% 도달. 같은 ckpt를 `--no-chunk-buffer`로 평가 시 stage 0 **5%** (-85%p), visual_agreement 91→12%
- **원인**: ACT의 inference 단 trick (temporal ensemble α=0.2 + gripper_use_latest)이 actor 자체보다 큰 기여. `actor_mean(chunk[0])` 단독은 5%, ChunkBuffer가 평균 + gripper 보정 적용 후 90%
- **결과**:
  - PPO/SAC fine-tune은 `actor_mean` 1-step interface만 사용 → 시작점 5% → stochastic noise → 0%
  - PPO + BC v2 시도 (logstd -1.0, -2.5 둘 다) 24 iter 모두 ep_return=0 → ABORT
  - "BC ACT 90%"는 deploy 시점에 ChunkBuffer 인터페이스 필수 (sim2real에서 real robot policy도 ChunkBuffer 호환되어야)
- **해결**:
  - **다음 RL은 1-step BC (no chunking) 새로 학습 → SACfD or PPO**. ChunkBuffer 의존 X
  - BC ckpt 성능 평가 시 항상 CB on/off 둘 다 측정 → 진짜 actor 강도 파악
- **코드 자산**: `eval_policy.py --no-chunk-buffer` flag (검증 + sim2real 진단용)

---

### 20. PPO 학습 GPU 1에서 lerobot 학습과 충돌로 sps 5배 저하
- **상태**: RESOLVED (운영 패턴)
- **증상**: PPO `train_ppo.py` sps=23 → **sps=4** (5배 느림). 50k step ETA 30min → 3.5h
- **원인**: 다른 팀원이 GPU 1에서 lerobot 학습 (`~/.conda/envs/lerobot/python`) 동시 실행. CLAUDE.md "GPU index 1만 사용" 정책 따라 우리 PPO도 GPU 1 → compute 경쟁
- **해결**:
  - 우리 PPO를 GPU 0로 옮김 (`CUDA_VISIBLE_DEVICES=0`). 사용자 명시 동의 후 진행
  - lerobot 학습은 보존
- **교훈**: CLAUDE.md GPU 정책은 default — 팀 협업 상황에서 다른 GPU 사용 필요 시 사용자에게 명시 요청 후 override OK. chain script에 `export CUDA_VISIBLE_DEVICES=0` 명시

---

### 21. ChunkBuffer 미적용으로 PPO ep_return=0 24 iter 지속
- **상태**: RESOLVED (#19와 동일 원인)
- **증상**: PPO + BC v2 + critic_warmup=5 + actor_logstd=-2.5로 50k step pilot 시도. iter 0~23 (frozen 5 + 풀린 후 18) 모두 ep_return_mean=0.000. pi_loss는 변동(±0.1)있지만 advantage가 0이라 의미 없는 update
- **원인**: BC v2 actor_mean(chunk[0]) 단독 성공률 5% (#19 참조). std=0.082 stochastic noise만으로도 success 0% 도달. Sparse reward signal 부재 + actor freeze 풀린 후 random drift
- **해결**: 시도 abort. PPO + ACT BC paradigm은 ChunkBuffer-aware learner 없이는 valid 아님
- **재발 방지**: BC ckpt를 RL pretrain으로 쓸 때 **반드시 actor 1-step 단독 성공률 측정**. 시작점이 < 50%면 sparse PPO는 fail 확정

---

### 22. Chain bash가 close-hang에 막혀 다음 step 못 감
- **상태**: PARTIAL FIX
- **증상**: v2 chain script에서 collect_demos.py가 max-success 도달 후 sim_app.close() hang. Chain bash `python ... > log` 라인에서 영원히 wait → 다음 step 진행 X
- **원인**: bash가 child python 종료를 wait. close-hang으로 python 안 죽으면 chain도 멈춤. v2 killer script는 eval/video만 다뤘고 collect는 안 잡았음 (#17 패턴 동일)
- **해결**:
  - v3 chain부터 killer가 `collect_demos`, `eval_policy`, `rollout_video` 모두 cover (log 마커 라인 감지 + 10초 후 pkill)
  - 또는 chain script에 `timeout NNN python ...` 적용 검토 (미적용)
- **운영 패턴**: 새 background python step 추가 시 killer script에 해당 log marker + pkill 한 줄 추가 필수

---

### 23. BCConfig dataclass field가 _json_safe에서 JSON 직렬화 실패
- **상태**: RESOLVED (2026-05-18 BC ACT 첫 학습 직후)
- **증상**: BC ACT k=12 50 epoch 완료 후 `bc.py.save()`의 `bc_meta.json` 작성 도중 `TypeError: Object of type ActionChunkingCfg is not JSON serializable`. `bc.pt` + `norm.json`은 정상 저장 (이전 라인) — meta JSON만 mid-write 실패
- **원인**: `_json_safe(v)`가 Path/list/dict만 처리. dataclass instance(ActionChunkingCfg)는 그대로 통과 → json.dump가 raise
- **해결**: `_json_safe`에 `dataclasses.is_dataclass` 분기 추가 → `asdict(v)`로 변환 후 recurse. bc.pt는 영향 없어 그대로 eval 가능했음
- **재발 방지**: 새 dataclass cfg를 BCConfig에 추가할 때 `_json_safe` 분기 자동 처리됨

---

## RL / Modeling (2026-05-18 밤 ~ 2026-05-19 새벽)

### 24. SACfD 5장 카드 연속 실패 (1-step BC 3% + sparse 5-cond AND가 RL 부트 한계)
- **상태**: RESOLVED (paradigm 전환 — G+ env 재설계)
- **증상**: 1-step BC (no chunking) actor 단독 성공률 3%에서 출발한 SACfD 시도 5장 모두 ep_return=0 정체:
  - **orig**: bc_w=1.0, q_filter, demo+online, target_entropy=auto → lock-in (BC 분포 못 벗어남)
  - **A**: bc_w=0.3, target_entropy=-3, anneal 50k → 여전히 0
  - **C'**: no BC transfer + no BC loss + HER + demo replay → random init부터라 HER 무용
  - **C''**: BC weight 유지 + no BC loss + HER → HER이 5-cond AND 못 채움 (relabel해도 success 신호 없음)
  - **D**: C'' + lift-only dense reward → dense 신호 약하고 굴리기 reward hack 우려
- **원인**: 세 가지 동시 실패 모드 — (1) 1-step BC가 sparse 5-cond AND를 거의 못 만지는 trajectory만 뽑음, (2) HER이 4/5 조건(stable_placement, release_retreat, velocity_stability, visual_agreement)이 goal-independent라 relabel 효과 없음, (3) horizon 160 step 짧고 spawn 분포 5cm로 넓어 random exploration도 도달 불가
- **해결**: G+ paradigm (`docs/g_plus_design.md`) — 환경 자체 재설계
  - Narrow curriculum: cube spawn 5cm → **2cm** (CurriculumCfg.side_length_m에 0.02 추가)
  - Sub-task decomposition: `task_level` 0=lift만 / 1=lift+place / 2=full PnP
  - Hover-attractor guard: task_level=0에서 lift latching 후 30 step 내 force_terminate (`cube xyz` drift 추적)
  - 매번 task_level 단계별로 졸업 (lift_history ≥ 70%, place_history ≥ 70%, full SR ≥ 70%)
- **결과 (2026-05-19 새벽)**: G+ 환경에서 BC 재학습 시 lift_history **3% → 90%** (30배). RL은 검증 진행 중 (STEP 4 SACfD L0 smoke)
- **재발 방지**: sparse RL은 환경 friendly 정도가 결정적 — 5조건 AND 같은 multi-condition success는 sub-task로 깨고, BC가 첫 조건은 거의 풀어주는 출발점에서 RL을 시작. "BC가 너무 약해" 진단은 BC만 손보지 말고 **환경의 step 의미**부터 다시 봐야 함

---

### 25. chain_g_plus.sh STEP 3 gate task-misalignment (BC SR=35%@L2로 abort, 실제는 lift 90%로 충분)
- **상태**: RESOLVED (chain script 수정)
- **증상**: G+ chain 첫 run에서 STEP 3 BC eval gate가 `SR=35.0% < 50%`로 chain abort. 그러나 per-condition 분석:
  - lift_history **90%**, stable_placement 85%, velocity_stability 85%
  - release_retreat 35%, visual_agreement 38% (이 둘이 SR 깎음)
- **원인**: SACfD는 task_level=0 (lift-only)부터 학습하는데 BC gate는 task_level=2 (full PnP) 기준. L0 success criterion은 lift_history 단독이라 BC의 L2 SR이 release/visual에 발목 잡혀도 L0 출발점으로는 90%가 이미 충분
- **해결**: chain_g_plus.sh STEP 3 gate를 `SR≥50%@L2` → `lift_history≥70%@L2`로 변경. 동시에 STEP 1-3 idempotency 추가 (산출물 있으면 skip)
- **재발 방지**: chain gate는 그 다음 step의 학습 task와 같은 criterion이어야 함 — sub-task curriculum 도입 시 gate도 sub-task 기준으로 분리. 한 줄 정리: **"gate metric은 다음 학습이 실제로 평가하는 metric과 같아야 한다"**

---

### 27. SACfD F7 STEP 5에서 짓누르기 reward hacking 발견 (jabis_sim_v2 v2)
- **상태**: RESOLVED (env success criterion 강화 + 4-bar mimic 충돌 fix)
- **증상**: G+ paradigm + F7 (LOG_STD_MAX=-3, no_entropy_term, BC anchor 0.7)로 STEP 5 full 300k 진행 중 step 18432 시점 `ep_return_mean = 0.188` (19% success), step 30720에 `0.375` (37.5%) 성장. 그러나 step 20001 video sample 확인 시 **policy가 cube를 grasp하지 않고 위에서 짓누르고 있음**. lift_history latch 통과는 press 후 cube z momentary spike로 trigger됨
- **원인 (이중 layer 문제)**:
  - **Layer A — success criterion 부실**: `success.py`의 lift_history check가 `cube_xyz[2] >= lift_z_m` 단독. grasp 여부 검증 없음. press → physics rebound → cube z momentary spike → latch가능
  - **Layer B — gripper 4-bar floppy**: `articulation.py`의 "mimic" ImplicitActuator entry가 PincOpen 4-bar의 4개 mimic joint에 PD 적용. USD에 이미 stamp된 `PhysxMimicJointAPI:rotZ` (hard constraint)와 충돌 → 외력 받았을 때 좌/우 finger 비대칭 변형. **그래서 진짜 grasp가 안 됨 → policy가 어차피 못 잡으니 press가 유일한 z 임계 통과 path가 됨**
- **해결**:
  - **Layer A fix (success.py + cfg.py)**: lift_history latch에 3-condition AND 적용:
    ```python
    cube_above = cube_z >= lift_z_m
    ee_close = ||EE - cube|| <= lift_ee_cube_max_dist_m   # default 0.08 m
    gripper_closed = gripper_opening <= lift_gripper_max_open  # default 0.7 (normalized 0..1)
    lifted_now = cube_above AND ee_close AND gripper_closed
    ```
  - **Layer B fix (articulation.py)**: 4-bar의 `"mimic"` ImplicitActuator entry 완전 제거 (`#28` 참조)
- **검증 후 진단 — guardrail 작동**:
  - CLAUDE.md #1 (video auto-sampling) 가드가 작동: ep_return 0.375가 의심됐다면 보지 않고 RL 성공이라 결론냈을 수 있음
  - **눈으로 영상 확인 = success metric 의심 = guardrail 핵심**. 사용자가 영상 보고 즉시 catching함
- **결과**: F7 STEP 5 결과 무효화. 새 env로 demo 재수집 + BC 재학습 + SAC chain 재시작 필요
- **재발 방지**:
  - sparse multi-condition success는 **각 조건이 독립적으로 hackable한지 별도 검토** 필수
  - "cube 위치 단독"은 절대 success criterion 아님 — gripper grasp 신호와 AND 결합
  - video sampler를 끄지 말 것. ep_return이 좋아 보여도 video를 5분이라도 보고 검증

---

### 28. PincOpen 4-bar mimic actuator vs PhysxMimicJointAPI drive conflict (흐물흐물)
- **상태**: RESOLVED (mimic actuator 제거)
- **증상**: gripper close 후 cube를 잡으려 하면 좌/우 finger가 **비대칭으로 꺽임** — 한쪽이 외력에 더 변형. 4-bar parallelogram이 mechanical structure로 평행 유지해야 함에도 floppy
- **원인**: `articulation.py`에서 4개 mimic joint(`left_proximal/distal`, `right_proximal/distal`)에 ImplicitActuatorCfg 적용 (stiffness=2000, effort=50, damping=100). 그러나 USD에 이미 stamp된 `PhysxMimicJointAPI:rotZ` (`scripts/usd_add_mimic_api.py` 결과)도 같은 joint에서 hard constraint enforce. **두 메커니즘이 같은 joint 권한 다툼** → solver 진동 → 외력 받았을 때 각 PD가 독립적으로 보정 시도 → 비대칭 변형
- **검증**: USD binary dump에 `pxMimicJointAPI:rotZ apiSchemas` + `gearing` + `targetPaths` 확인됨 — MimicAPI 정상 stamp
- **해결**: `articulation.py`의 `"mimic"` ImplicitActuatorCfg entry 완전 제거. PhysxMimicJointAPI가 4-bar coupling 단독으로 enforce. gripper main actuator의 torque가 constraint를 통해 4-bar에 전파 — 별도 PD 불필요
- **과거 우려 분석**: 코드 comment(L98-128)의 "force_mimic_coupling fought PhysX contact resolver" 우려는 **다른 메커니즘**(Python에서 매 tick joint_state 직접 write)이었음. PhysxMimicJointAPI는 USD-level constraint라 PhysX와 native 통합 → 진동 없음
- **CLAUDE.md 일치**: "ActuatorCfg에는 reference joint 1개(gripper)만 포함. mimic 조인트는 PhysX가 자동 추종 — 양쪽 다 actuator로 잡으면 drive 충돌" 명시 — 이번 fix가 정확히 그 원칙 복원
- **재발 방지**: 4-bar / 4-mimic linkage가 있는 robot에 actuator 추가할 때 **반드시 USD MimicJointAPI 적용 여부 확인**. 둘 다 적용은 절대 금지. 의심 시 USD에서 `PhysxMimicJointAPI:*` schema 검색

---

### 26. SACfD F2 (HER + no BC loss) Q overestimation 발산 + alpha collapse
- **상태**: RESOLVED (진단 — F3로 부분 fix 확정)
- **증상**: G+ 환경 (narrow 2cm spawn, L0 lift-only, BC lift_history=90%) 위에서 SACfD smoke 50k step 시도. step 16384 시점:
  - q1=8.918 (이론 max γ^160·1≈0.2의 44배 hallucinate)
  - alpha=0.0161 (init 0.2 → 거의 0 collapse)
  - ep_len_mean=160 (lift latch 0회, BC manifold 완전 이탈)
  - critic loss 0.009 (wrong fixpoint에 수렴)
  - ep_return=0 across 50k → 5400s timeout abort
- **원인 (textbook SAC sparse divergence 5단계)**:
  1. drift rare positive transition (BC 90% actor 출발인데 SAC stochastic sample이 OOD로)
  2. Bellman bootstrap hallucination (demo terminal +1이 모든 state에 propagate, HER transition은 r=0이라 grounding 없음)
  3. alpha auto-tuning이 target_entropy=−3 향해 빠르게 감소 → exploration 죽음
  4. BC anchor 없음 (`--no-bc-loss`)으로 actor가 critic의 hallucinated optimum 쫓아 BC manifold에서 이탈
  5. dead policy lock (deterministic + lift 못 함 + horizon hitting)
- **HER이 L0에서 dead weight인 추가 원인**: L0 success = lift_history 단독 (goal-independent). HER이 relabel하는 건 `target_delta` (goal), 그러나 lift_history는 goal과 무관 → relabel 후 success rate 항상 0. batch 40%(102/256)를 무의미한 신호로 채움
- **부분 해결 (F3)**: F3는 4가지 변경 동시 적용:
  - HER off (L0 무용 + Q bootstrap noise 제거)
  - BC anchor `--bc-loss-weight 0.3` + q_filter on (actor를 BC manifold에 잡아둠)
  - `--target-entropy −1` (−3 → −1, entropy 강제 완화)
  - `--init-alpha 0.05` (init 0.2 → 0.05)
- **F3 결과 (검증)**:
  - q1 발산 막힘: 2.84 stabilize (F2의 1/3 수준) ✅
  - critic loss 정상 수렴 ✅
  - BC manifold 유지 (bc loss 0.012, actor=demo 분포 위) ✅
  - **그러나 lift 0%, ep_return=0 동일** — gate fail
  - alpha 결국 collapse (0.0066) — `--target-entropy` 완화로 부족
- **잔존 가설 (F3 lift 안 됨)**: BC 90%는 **deterministic + chunkbuffer X 1-step actor_mean 모드** 평가. SAC stochastic rollout (actor.sample()) 모드에서는 BC의 실제 robust 강도가 훨씬 낮을 수 있음 — #19 ACT trick 발견의 1-step BC 버전. 검증 필요: BC actor를 stochastic sample mode로 100 ep 평가
- **다음 카드 (F4) — 더 보수적**:
  - `--bc-loss-weight 0.3 → 0.7` (anchor 강화, Grok 권고)
  - `--bc-anneal-steps 50000 → 300000` (anneal 거의 안 함)
  - `--target-entropy −1.0 → −0.5` (entropy 더 강제)
  - `--init-alpha 0.05 → 0.1`
  - `--actor-lr 3e-4 → 1e-4` (BC manifold 천천히 이탈)
  - **alpha hard clip (sac.py 1줄 추가): `self._log_alpha.data.clamp_(min=math.log(0.02))`** — auto-tuning이 collapse 못 함
- **재발 방지**:
  - SACfD에서 demo의 terminal +1만 있고 dense reward 없으면 critic이 grounding 부족 → BC anchor 필수 (BC anchor 0.3는 약하다, 0.5~0.7 권장)
  - alpha auto-tuning은 sparse + sub-task 조합에서 collapse 위험 → hard clip floor 0.02 권장
  - sub-task의 success criterion이 goal-independent면 HER 절대 켜지 말 것
  - BC ckpt 성능 평가는 deterministic + stochastic 둘 다 측정 — sim2real 및 RL bootstrap에서 진짜 강도는 stochastic 수치

---

## RL / Modeling (2026-05-19~05-20 SAC 카드 시퀀스)

### 29. SAC F2~F11 lift=0% — 5개 fundamental bug 누적 진단 (BC sweep 71% vs SAC 0% 미스터리)
- **상태**: RESOLVED (F12부터 점진 fix)
- **증상**: G+ env + 1-step BC (anti-pressing 통과 90%) 위에서 F2~F11 모든 SAC 카드 lift_history=0%. F12 critic warmup으로 19% 첫 진전 + F14 100% sustained 도달
- **근본 진단 (각 카드에서 발견)**:
  - **F2**: Q overestimation (HER + no BC loss + target_entropy -3 → q1 8.9 발산, alpha 0.006 collapse) — #26 별도 entry
  - **F3**: BC anchor 0.3 약함 — Q hallucination 시작 (q_o > q_d)
  - **F4**: target_entropy -0.5 + LOG_STD_MAX 너무 high → squashed Gaussian의 strict entropy로 pi_std=0.86 → BC robust 범위(<0.20) 4배 초과
  - **F5**: LOG_STD_MAX -1.6 cap (std≤0.20) 적용했지만 squashed Gaussian log π 양수로 entropy bonus가 Q를 음수로 끌어내림 (target_v = min_q - α·log π → log π > 0 → target_v 감소)
  - **F6**: --no-entropy-term (TD3-like Bellman) → entropy backfire 해소
  - **F7**: LOG_STD_MAX -3 (std≤0.05) cap. BC sweep 79% lift였지만 SAC online lift=0% — 다른 root cause 있음 진단
  - **F8/F8'**: clamp vs tanh 변경 시도 — bc_loss=0.21로 폭증 (mask=0 collapse 발견)
- **공통 lesson**:
  - BC sweep eval과 SAC online이 같은 weight+std임에도 다른 결과 보이는 경우, **SAC training loop의 implicit difference 찾아야**. 동일 input → 다른 output 시 코드 layer 모두 의심
  - Bellman target에서 entropy term이 Q에 미치는 영향은 squashed Gaussian에서 log π 부호에 따라 반전됨 — narrow std에서 log π > 0 → entropy가 Q 깎음
  - 5개 fundamental bug가 누적되어 있었던 상황 — 1개씩 fix해도 다른 bug가 가려져 lift=0% 유지

---

### 30. SAC GaussianActor.trunk ReLU vs BC ActorCritic.actor Tanh activation 불일치 (silent fundamental bug)
- **상태**: RESOLVED (sac_network.py L63, L65: ReLU → Tanh, F10에서 codex 발견)
- **증상**: F10까지 SAC lift_history=0%. load_bc_actor가 BC weight를 SAC GaussianActor.trunk + mean_head로 transfer하지만 trunk의 activation이 ReLU인 반면 BC ActorCritic.actor는 Tanh 사용. **동일 weight + 다른 activation = 완전히 다른 hidden representation.**
- **진단 방법**: codex가 두 actor의 forward pass에 같은 obs 넣고 max diff 측정 → ReLU 사용 시 mismatch, Tanh 통일 후 max diff = **0.00e+00** (수치적 완전 일치)
- **검증된 동작 (F10)**: Tanh 적용 후 bc_loss step 6144에 0.037 (F9의 0.251 대비 6.8배 작음). BC weight transfer가 진짜로 의미 있게 작동
- **그러나 F10 lift는 여전히 0%** — 활성화 수정만으로 부족. 다른 bug (#31, #32) 동시 존재
- **재발 방지**:
  - 두 모델 간 weight transfer 시 반드시 forward pass 비교 (max diff = 0인지 확인)
  - SAC GaussianActor와 BC ActorCritic이 아키텍처 일치 보장 — Linear/Tanh sequence 동일해야
  - "weight transfer 잘 되고 있다"는 가정은 코드 layer마다 검증 — activation이 silent하게 다를 수 있음
- **Codex의 발견 가치**: ReLU/Tanh 불일치는 코드 두 줄 차이지만 모든 F2~F9 카드의 base bug였음. 사람이 보기 어려운 silent bug — codex/외부 review의 가치

---

### 31. Q-filter mask=0 collapse → BC anchor silently disabled (F8' fake bc_loss=0.004)
- **상태**: RESOLVED (`--no-q-filter` 적용, codex 진단)
- **증상**: F8'에서 bc_loss가 step 16384에 0.004로 감소 → actor가 demo와 일치한다고 판단. 그러나 ep_len=160 / lift=0% 그대로
- **원인**: `sac.py _bc_loss`의 Q-filter mask가 거의 0으로 collapse:
  ```python
  mask = (q_demo > q_actor).float().unsqueeze(-1)  # Q-filter
  per_row = ((actor_mean - action_demo) ** 2).sum(dim=-1, keepdim=True)
  return (mask * per_row).mean()  # mask=0이면 loss=0
  ```
  - 학습 초기 Q는 random → q_demo vs q_actor 비교가 random → mask 거의 0 → BC loss가 silent하게 disabled됨
  - bc_loss = 0.004는 "actor=demo"가 아니라 "mask가 row 가려서 loss 자체가 사라짐"
- **검증 (F9)**: --no-q-filter 적용 후 bc_mask = 1.000 보장. bc_loss = 0.251 (진짜 drift)로 폭증 — 이게 실제 actor와 demo의 거리
- **재발 방지**:
  - bc_loss 값을 그대로 믿지 말고 mask.mean()을 logging해서 BC anchor가 진짜로 active한지 확인
  - Q-filter (Vecerik 2017)는 critic이 sane Q를 학습한 후에만 유의미. critic warmup 이전 / 초기 학습 단계에서는 unconditional BC anchor 권장

---

### 32. SAC critic warmup 누락 — actor 풀리자마자 random Q에 망가짐 (CLAUDE.md decision 7.3 미구현)
- **상태**: RESOLVED (`critic_warmup_steps=3000` 추가, F12부터)
- **증상**: F11에서 instrumented `[ep_end]` 추가하여 episode 종료 시 per_cond 확인. Pre-learning_starts random 31 episodes (random action) 모두 lift=0 (당연). Post-learning_starts step 5120 첫 actor episode lift=0, step 5280 actor가 cube 흩뜨림 (place=0 vel=0), step 5440-9538 lift=0% 12 episodes 연속
- **원인**: CLAUDE.md Phase 1 결정 7.3에 명시:
  > 3. Critic warm-up — 첫 10~20 iter는 actor freeze + critic만 학습 (garbage advantage가 actor 망치는 것 방지)
  - 그러나 `sac.py` 코드에 critic warmup 로직 미구현. learning_starts=5000 직후 첫 update부터 actor + critic 동시 train. Critic은 random init이라 wrong gradient 보냄 → actor가 BC behavior 망가짐
- **F11→F12 fix (sac.py)**:
  ```python
  in_critic_warmup = (global_step < cfg.learning_starts + cfg.critic_warmup_steps)
  if not in_critic_warmup:
      actor_optim.step()  # actor freeze during warmup
  if not cfg.no_entropy_term and not in_critic_warmup:
      alpha_optim.step()  # alpha tuning skip
  ```
  - SACConfig에 `critic_warmup_steps: int = 2_000` 추가 (F13부터 3000)
- **F12 결과**: ep_return 0→19% 도달 (warmup phase에서 BC behavior 보존 + critic이 BC trajectories에서 Q 학습)
- **재발 방지**:
  - **CLAUDE.md 결정사항을 코드에서 한 줄 한 줄 검증**. "기록되어 있으니 구현되어 있을 것"이라는 가정 위험
  - BC weight transfer + RL fine-tune 시퀀스에서 critic warmup은 표준 — random critic gradient가 BC actor를 corrupt하기 전에 critic이 reasonable Q estimate 갖춰야

---

### 33. F13 Q hallucination 재발 → F14 (bc_w 3.0 + target_v clamp [0, 1]) 정착
- **상태**: RESOLVED (F14가 finally working sustainable SAC RL)
- **증상**:
  - F12 (warmup 2000, bc_w 0.7): ep_return 19% → 37% (peak) → 0% (actor degradation)
  - F13 (warmup 3000, bc_w 1.5, actor_lr 3e-5): ep_return 68% peak → step 20480 q_o > q_d 시작 → step 24576 q_o=0.93 (2.3×q_d) hallucination → ep_return 0%로 collapse
- **원인**: warmup + BC anchor 강화로 buy time했지만 Q가 결국 online states를 overestimate. SAC 표준 actor_loss = -min_Q에 따라 actor가 hallucinated high-Q 방향 pursuit → BC manifold 이탈
- **F14 fix 3가지 동시 적용**:
  1. **bc_w 1.5 → 3.0** (anchor 2배 강화)
  2. **bc_anneal_steps 300k → 1M** (smoke 50k 동안 anneal 거의 없음)
  3. **target_v.clamp(0.0, 1.0)** (sac.py Bellman target에 hard cap — sparse reward {0, 1}이므로 realistic Q max ≈ γ^k ≤ 1, clamp가 over-estimation 차단)
- **F14 trajectory (50k smoke)**:
  - step 14336: 41% lift, q_d=0.28, q_o=0.07 (ratio 0.25)
  - step 20480: 63%, q_d=0.28, q_o=0.14 (ratio 0.50) — F13는 같은 시점 collapse 시작
  - step 24576: 81%, q_d=0.34, q_o=0.28 (ratio 0.82, healthy)
  - step 30720: 91%, q_d=0.55, q_o=0.39 (ratio 0.71)
  - step 45056: **100%** (rolling 32 perfect), q_d=0.48, q_o=0.41 (ratio 0.85)
  - 39 lift 연속 streak 여러 번 관찰
- **재발 방지**:
  - SAC + BC anchor + sparse reward 조합에서 target_v.clamp는 표준 권장. sparse {0, 1} reward는 Q ∈ [0, 1] 보장
  - bc_w 1.5도 충분치 않을 수 있음 — 3.0이 가장 효과적. anneal은 smoke 동안 거의 없게 (300k+ steps)
  - Q ratio q_o/q_d 모니터링 권장 — 1 초과하면 hallucination 시작 신호
- **F14 코드 변경 요약 (모든 fix 누적)**:
  - sac_network.py: `LOG_STD_MAX = -3.0`, trunk ReLU → Tanh, sample/mean_action: tanh → clamp (F8 잔여)
  - sac.py: `no_entropy_term` flag, `critic_warmup_steps=3000`, `target_v.clamp(0, 1)`, raw_mean bc_loss, alpha hard clip (skip if no_entropy_term)
  - chain_g_plus.sh STEP 4: `--no-entropy-term --no-q-filter --critic-warmup-steps 3000 --bc-loss-weight 3.0 --bc-anneal-steps 1000000 --target-entropy -6.0 --init-alpha 0.1 --actor-lr 3e-5`

---

### 34. F14 STEP 5 squash hack 발견 (영상 검증으로 catch — guardrail 작동)
- **상태**: RESOLVED (success.py F15 anti-press v2)
- **증상**:
  - F14 smoke 50k step ep_return 100% 도달 + 영상에서 진짜 grasp 확인 ✅
  - F14 STEP 5 (300k full) launch 후 step 20001 video sample에서 **새로운 reward hack 발견**:
    - actor가 cube에 접근 + 그리퍼 닫기 시도 (잡는척)
    - cube가 misaligned 4-bar finger 사이에 끼임 → squash (짓이김)
    - squashed cube가 jaws에서 튕겨 위로 → cube_z >= 8cm 잠깐 만족
    - lift_history 3-cond AND (cube_z + ee_close + gripper_closed)는 통과
  - 즉 success criterion이 squash 패턴을 차단 못함
- **원인 (anti-press v1 한계)**:
  - F15 이전 lift_history 조건: `cube_z >= 0.08 AND ee_close <= 0.15 AND gripper_opening <= 0.85`
  - Squash 시 모든 조건 만족 가능:
    - cube_z ≥ 0.08 (튕긴 cube 잠시 위로)
    - ee_cube_dist 작음 (gripper가 cube에 있음)
    - gripper_opening < 0.85 (눌리는 동안 partially closed)
  - **cube velocity stability를 보지 않아서** squash bouncing 통과됨
  - 4-bar parallelism issue (#28 cosmetic)이 reward hacking 경로를 만들어줌 — 좌/우 finger 비대칭 변형이 squash 메커니즘
- **F15 fix (success.py + cfg.py)**:
  ```python
  # lift_history latch 추가 조건:
  v_xy = ||cube_lin_vel[:2]||  # 수평 속도
  omega = ||cube_ang_vel||      # 각속도
  cube_stable_xy = v_xy <= lift_cube_v_xy_max       # default 0.20 m/s
  cube_not_tumbling = omega <= lift_cube_omega_max  # default 270°/s
  lifted_now = cube_above AND ee_close AND gripper_closed
               AND cube_stable_xy AND cube_not_tumbling
  ```
- **Threshold tuning (pilot 2026-05-20)**:
  - 초기 시도: v_xy ≤ 0.10, omega ≤ 60°/s → oracle 5/5 fail
  - 진단: temp debug print로 측정 — oracle's cube during lift has v_xy ≤ 0.04 m/s but omega ~2.2 rad/s (126°/s) **sustained** (cube가 헐거운 4-bar 안에서 회전)
  - 최종: v_xy ≤ 0.20 m/s + omega ≤ 270°/s (oracle 5/5 pass, squash 차단 기대)
- **F15 검증 (영상 비교)**:
  - F14 step 20001 (squash hack): cube 짓이김 + gripper 엉망
  - F15 step 20001 (cube velocity 차단 후): 진짜 grasp + lift ✅ (사용자 영상 확인)
  - F15 step 40001, 60001: 진짜 grasp 유지 — sustainable
  - F15 SAC trajectory: ep_return 100% sustained (F14와 같은 수치, 그러나 quality 다름)
- **Guardrail 작동 사례**:
  - CLAUDE.md Phase 1 결정사항 #1 (video auto-sampling)이 정확히 이 hack을 catch
  - 메트릭만 봤으면 F14를 "RL success"라고 발표할 뻔 — jabis_sim_v2 reward hacking 반복
  - F7 press hack도, F14 squash hack도 둘 다 **영상으로 catch**. 영상 없이 RL 결과 발표 금지
- **재발 방지**:
  - sparse multi-condition success criterion에서 각 조건이 **independent하게 hackable한지** 별도 검토 필수
  - cube position 조건은 cube velocity 조건과 함께 — position 단독은 항상 hackable (튕김으로 잠시 만족 가능)
  - 시뮬레이션 actuator의 cosmetic issue (#28 4-bar)가 reward hacking 경로 만들 수 있음 — purely cosmetic이라고 가정 X
  - **모든 새 success criterion 도입 후 영상 검증 필수** — 메트릭 통과 후에도 영상 안 보면 hack 놓침

### 35. F15 L1 chain — trivial spawn / throw hack / high-release hack / BC OOD 4종 동시 발견 + stage 2 BC로 정착

**날짜:** 2026-05-20 ~ 05-21. F15 anti-pressing v2 통과 후 L1 (lift+place) 학습으로 넘어가면서 4개 별개 문제가 sequential하게 발견.

**(a) Stage 0/1에서 stable_placement이 trivial:**
- `env.py:406`: `cube_xy = uniform(-side/2, side/2) + goal_xy[:2]`. cube spawn이 goal 중심의 정사각형.
- `cfg.py`: stage 0=2cm, stage 1=5cm, stage 2=10cm, stage 3=15cm, stage 4=20cm. goal radius = 3cm.
- 기하학: stage 0에서 100%, stage 1에서 **95%**, stage 2에서 28%, stage 3에서 7% 확률로 cube가 reset 시점부터 goal radius 안.
- 영향: L1 success = lift_history AND stable_placement인데 cube spawn 시점부터 placement=True면 정책은 lift만 학습 → L0와 동치.
- **fix:** chain script STEP 7/8을 `--curriculum-stage 2` 이상으로 변경.

**(b) Throw hack (사용자 영상으로 발견):**
- lift_history는 latching (1초 들고 있으면 영구 True).
- velocity_stability는 task_level=2 reward에만 들어가고 task_level=1 reward에는 빠져있음.
- 정책이 학습한 패턴: "1초 들었다 → cube를 던져서 → goal xy 안에 안착시키면 reward". 영상에서 "큐브가 던지듯 빠르게 떨어짐".
- **fix:** `env.py` task_level=1 sparse/terminated에 `velocity_stability` AND 조건 추가.

**(c) High-release hack (사용자 영상으로 발견 — "goal 위에서 그냥 그리퍼 open"):**
- stable_placement은 cube xy만 봄 (z 무관).
- 정책: cube를 goal 위 공중에서 hover하다가 gripper open → cube 자유낙하 → goal 안 착지.
- velocity_stability만으로 못 막음 — 떨어진 후 stop하면 velocity=0이라 OK.
- **fix:** `success.py` stable_placement에 `cube_z <= place_z_max_m` AND 추가.
  - cube center at rest = `table_top_z + cube/2 + clearance = 0.0 + 0.0225 + 0.001 = 0.0235m`
  - `place_z_max_m = 0.035m` (cube rest 통과 + lift_z 0.08m 차단)
  - **codex blocking review** 가 이 임계값 검증 — 초기 `place_z_tolerance_m = 0.025`는 cube resting state도 fail시킴.

**(d) BC stage 0→2 OOD (Codex가 예측 + 영상으로 확인):**
- BC pretrain demo는 stage 0 (2cm) 분포에서 수집. cube가 항상 goal ±1cm.
- stage 2 (10cm) spawn에서는 cube가 goal에서 ±5cm까지 떨어짐 → BC의 reach trajectory가 wrong xy로 향함.
- 증상: L1 SAC at stage 2에서 lift_history rate **16%만** (이전 stage 0에서 70%+). SAC가 BC anchor 못 활용. ep_return 0.03 정체.
- **fix:** oracle demo를 stage 2 분포에서 1000 success 재수집 (~2시간) → BC 재학습 (~30초) → SAC 재학습 (~5시간).

**(e) eval_policy.py가 SAC ckpt 호환 안 됨:**
- 기존 eval_policy.py는 BC/PPO ckpt의 `actor_critic` key 가정 → SAC ckpt는 `actor` key만 → `KeyError`.
- STEP 6 (L0 eval)에서 33초 만에 fail로 chain abort.
- **fix:** SAC ckpt detection (`"actor" in ckpt and "actor_critic" not in ckpt`) → `GaussianActor` 로드 fallback.

**최종 F15 chain (2026-05-21 07:21 완료, 13h 8m):**
- BC stage 2 demo로 재학습 → SR 35% → **55%** (L2 평가)
- L0 full 300k @ stage 0: ep_return 1.000, lift_history 99.0%
- L1 full 300k @ stage 2 + z-cap + velocity_stability: ep_return 0.69~0.75
- per_cond 분해 (final 500 ep): **진짜 PnP (lift+place) 69.0%**, trivial spawn 18.8%, lift만 4.4%, 완전 fail 7.8%
- 영상 검증: 사용자가 cube를 잡고 들어 올려 goal로 옮긴 후 안착시키는 진짜 PnP 동작 확인

**재발 방지:**
- Multi-condition success criterion에서 각 조건의 reward (task_level별) 정의가 일관된지 검토 — task_level=1 reward에 velocity_stability 빠진 게 throw hack의 원인.
- BC pretrain은 SAC가 학습할 spawn 분포와 같은 분포에서 수집 — stage 0 demo는 stage 2 학습에 OOD라 sample efficiency 박살.
- success criterion의 임계값은 simulation 상수 (table_top_z, cube_size, spawn_clearance)와 일치하는지 codex review.
- chain script idempotent skip은 dir 잔재 archive 필수 — 옛 sac.pt가 새 chain의 STEP 7을 silently skip시켜 학습 검증 누락.

### 36. Gripper rigidity fix journey — 4-bar mimic이 본질적 한계 (sim2real 영향 없음으로 정리)

**날짜:** 2026-05-21. F15 milestone 후 "그리퍼 양쪽이 평행하지 않고 외력에 한쪽이 더 꺽임" 문제 진단 + 시도 + 결정.

**진단 (`scripts/inspect_4bar_drive.py` 신규 작성):**
- USD 4-bar joint 4개에 `PhysxMimicJointAPI:rotZ`만 stamp돼 있고 drive 가는 URDF 변환 default (stiffness=1.7, damping=0.017, target=0) — 매우 weak
- `articulation.py`의 "mimic" `ImplicitActuatorCfg` (stiffness=2000) 도 stamp — PD가 4-bar joint를 target_position으로 끌어당김
- 두 source 동시 적용. 충돌의 진짜 원인은 troubleshooting #28에서 잘못 진단됐음 — 핵심은 **두 source의 target magnitude 불일치**:
  - `env.py` `_mimic_signs = [+0.5, -0.5, -0.5, +0.5]` (PincOpen URDF mimic multiplier 0.5)
  - `usd_add_mimic_api.py`가 stamp한 gearing = ±**1.0** ← mismatch
  - gripper=-1.8에서 PD target = ±0.9 vs USD mimic target = ±1.8 → 두 force 충돌 → 비대칭

**Option A (실패):** USD drive를 강하게 stamp + ImplicitActuator 제거.
- `usd_set_4bar_drive.py` 스크립트 작성 → stiffness 200, damping 10, maxForce 50 stamp.
- 결과: oracle **0/5 success**. cube z = 0.022 (table 그대로), mimic = [0,0,0,0] (anti-mimic).
- 원인: USD `PhysicsDriveAPI`가 `targetPosition=0`으로 mimic constraint를 override. PhysX 동작 — 같은 joint에 drive + mimic 동시는 drive 우세.
- 즉시 복원 (`so101_pincopen.usd.f15_chain_state_bak`에서 cp).

**Option B (적용, 부분 성공):** `usd_add_mimic_api.py`의 gearing magnitude 1.0 → 0.5로 변경 (env.py PD target과 일치).
- 결과: oracle **5/5 success**, mimic readouts (L_prox=-0.47, L_dist=+0.90, R_prox=+0.44, R_dist=-0.89) — distal 정확, proximal은 4-bar geometry constraint로 자연스럽게 절반.
- F15 L1 sac.pt 영상 검증: cube grasp + lift + place 모두 정상.
- ❌ 사용자 영상에서 finger 평행 X 여전히. gearing 일치만으로 충분치 않음.

**Option C (적용, 추가 개선 미미):** ImplicitActuator stiffness 2000→5000, damping 100→200.
- 외력 대항력 강화. oracle 3/3 success. F15 L1 sac.pt 영상 검증.
- ❌ 사용자 영상에서 여전히 그리퍼팁이 cube를 평행하게 잡지 못함. 짓눌릴 때 특히.

**근본 한계:**
- `PhysxMimicJointAPI` + `ImplicitActuator` 모두 4-bar를 **open chain의 4개 독립 joint + soft coupling**으로 모델링.
- 진짜 4-bar parallelogram은 **closed kinematic loop**으로 link 자체가 geometric으로 평행 강제. URDF 4-bar는 simplified로 closed loop 미구현.
- Stiffness 아무리 올려도 closed loop이 아니라 외력에 부분 위반 발생.

**Option D 후보 (미실행):** Closed-loop 4-bar URDF 수정 (Isaac Sim docs "Closed-Loop Structures") — 한 finger 끝에 D6 fixed joint로 4-bar 닫음. 2-3시간 작업 + URDF 변환 위험.

**최종 결정: Accept (sim2real 영향 없음 분석 후).**
- **실물 PincOpen은 캠 구조** (1 모터 → cam mechanism → 2 finger 동기화 강제 평행). 외력으로 비대칭 변형 불가능.
- **Sim 4-bar simplified는 캠의 근사**. Sim의 비대칭은 sim-side artifact.
- sim2real 방향: sim이 더 어려운 환경 → 학습 정책 over-robust → 실물 transfer 시 +. 비대칭 fix 시간 가치 적음.

**최종 적용 상태:**
- USD: gearing ±0.5 (env.py PD와 일치)
- articulation.py: mimic ImplicitActuator stiffness=5000, damping=200
- F15 L1 sac.pt: 새 USD에서 정상 작동 검증 완료 (videos_gearing_half_verify/, videos_optC_verify/)
- 학습 환경 backup: `assets/converted/usd/so101_pincopen.usd.f15_chain_state_bak` (gearing 1.0 + 옛 drive — F15 chain 학습 시점)

**재발 방지 / 다음 iteration:**
- 그리퍼 mechanism에 mimic이 적용된 경우, **PD target과 USD mimic gearing magnitude는 반드시 일치**시켜야 충돌 없음.
- PhysX 같은 joint에 `DriveAPI + MimicJointAPI` 동시 적용 시 drive 우세 — DriveAPI 사용 안 하려면 stiffness=0, maxForce=0으로 disable 필요 (그러나 그러면 외력 강성 부족).
- Sim 평행 미달은 sim2real에서 자동 해결되는 경우 많음 — 실물 mechanism이 더 rigid라면 sim 비대칭이 robustness 학습으로 작용.

### 37. Phase 1 완성 — L2 (5조건 AND) full PnP 학습 도달 + lift_history 65% ceiling 발견

**날짜:** 2026-05-21. F15 stack + 새 BC + 모든 가드 위에서 task_level=2 (lift_history AND stable_placement AND release_retreat AND velocity_stability AND visual_agreement) 학습. STEP 9 (smoke 50k) + STEP 10 (full 300k @ stage 2).

**결과:**
- STEP 9 smoke max ep_return = 0.469 (L1 smoke 0.594 대비 살짝 낮음 — 5조건 AND가 더 엄격)
- STEP 10 full 300k 끝 (16:33): ep_return_mean (last 30) 0.55, max 0.719
- per_cond (final 500 ep): lift 66.4%, **place 88.2%, release 97.0%, vel 99.6%, visual 85.4%**
- **진짜 5조건 동시 만족 (real PnP): 55.6%** (lift+place+release+vel+visual=1 episode)
- trivial spawn (lift=0+place=1) 23.2%, other 21.2%
- 사용자 영상 검증: 큐브 정상 grasp + lift + goal로 이동 + 안착. 진짜 PnP 동작 시각 확인.

**핵심 발견 — lift_history rate 65% ceiling:**
- 학습 step 117k → 246k → 283k 동안 lift_history는 62→66→65로 거의 정체
- 사용자 영상 관찰: "거리가 있는 cube의 경우 그리퍼가 cube center에 오지 않아 잡지 못함"
- 즉 **cube reach 정확도가 진짜 PnP의 bottleneck** — 일단 lift만 되면 나머지 4 condition은 거의 자동 (place 88%, release 97%, vel 99%, visual 85%)
- 이론 동시 만족률 = 0.664 × 0.882 × 0.970 × 0.996 × 0.854 ≈ 48% (실제 55.6%와 조건 상관관계 고려하면 일치)

**Reach 정확도 ceiling 원인 가설:**
1. BC reach manifold 한계 — oracle demo의 reach는 3-4mm 정확하지만 BC가 학습한 reach trajectory는 일부 imperfect
2. Observation noise — cube_xyz_base에 4mm σ xy noise가 멀리 spawn된 cube에서 누적 영향
3. Reach 동작 자체의 fundamental constraint — 5-DOF arm + 6-DOF 조작 task에서 IK redundancy/singularity

**다음 iteration 후보 (lift_history 80%+ 도달 위해):**
- Oracle reach 정확도 강화 (descend phase의 finger_to_target exit threshold 0.015 → 0.005m) + BC 재수집
- BC anchor 강화 (bc_w 3.0 → 5.0, anneal 더 길게) — SAC가 BC manifold 너무 빨리 벗어나지 않도록
- 또는 dense reach reward 도입 (CLAUDE.md "Phase 1 sparse-only" 원칙 이탈 — 마지막 수단)

**Phase 1 baseline ckpt 보존:**
- `runs/g_plus_sacfd_L0_full/sac.pt` — lift-only (stage 0)
- `runs/g_plus_sacfd_L1_full/sac.pt` — lift+place (stage 2)
- `runs/g_plus_sacfd_L2_full/sac.pt` — full PnP (stage 2, 5조건 AND)
- `runs/g_plus_stage0_bc_narrow/bc.pt` — BC pretrain (stage 2 demo로 재학습됨)

**Phase 1 종료 — 다음 단계는 사용자 결정:**
- Reach ceiling fix (Phase 1 최적화) / Trivial spawn fix (stage 3) / Sim2real / Phase 2-4 트랙


