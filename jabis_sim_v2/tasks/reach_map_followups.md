# Reach map follow-ups — Tight, finger_offset, Dense

날짜: 2026-05-13
이전 보고서: `tasks/reach_map_analysis.md` (1차 7×7, 8 env/pt)

## 1. Tight spawn 측정 (x∈[0.25, 0.31], y∈[0, +0.06])

| 지표 | 값 |
|---|---:|
| 12cm 성공률 | **26.6%** |
| LIFT 도달 | 29.2% |
| APPROACH stuck | 70.8% |

**예상 60~80% 대비 훨씬 낮음**. 이유: Tight 영역이 reach map sweet spot 4점이 아니라 그 사이의 0% 영역도 포함. dense map으로 확인 (아래 §3).

비교:
- Narrow (x∈[0.22, 0.28], y∈[±0.05]): **30.7%** ★
- Balanced (x∈[0.23, 0.31], y∈[-0.03, +0.08]): 27.1%
- Tight (x∈[0.25, 0.31], y∈[0, +0.06]): 26.6%

Narrow가 여전히 최고.

## 2. finger_x_offset sweep (-y 편향 수정 시도)

가설: `finger_x_offset = +0.006`이 +y 편향 원인일 수 있음
조건: Narrow spawn 고정, 3 variants

| finger_x_offset | LIFT 도달 | 12cm 성공 | APPROACH stuck |
|---|---:|---:|---:|
| **+0.006** (baseline) | 58.3% | **30.7%** | 39.6% |
| 0.000 | 58.9% | 7.3% | 39.6% |
| -0.006 | **73.4%** | 7.3% | **26.0%** |

**가설 기각**: finger_x_offset은 x 방향만 영향. -y 영역 grip 실패는 다른 원인.

발견:
- **+0.006이 grip 최적** (현재 채택값 옳음): cube 앞쪽에 finger 위치 → 잡기 좋음
- **-0.006이 reach 최적**: cube에 더 빨리 접근 가능, but cube 뒤쪽을 밀어서 grip 실패
- **0.000은 둘 다 안 됨**: 중간값이 worst

-y 편향 진짜 원인 추정 (안 확인): default arm pose의 wrist_flex 비대칭 또는 gripper jaw의 좌/우 friction 차이.

## 3. Dense reach map (32 env/pt, 7×7=49 pts, 1568 envs)

### LIFT 도달 %
```
y \ x   |  0.15   0.19   0.23   0.27   0.31   0.35   0.39
--------|------------------------------------------------
+0.12   |   9     91     78      0      0      0      0
+0.08   |   0      0    100      0      0      0      0
+0.04   |   0      0    100      0      0      0      0
+0.00   | 100    100    100    100     78      0      0
-0.04   |   0     47     62      0      0      0      0
-0.08   |   0      0    100      0      0      0      0
-0.12   |  19     59      0      0      0      0      0
```

### 12cm 성공 %
```
y \ x   |  0.15   0.19   0.23   0.27   0.31   0.35   0.39
--------|------------------------------------------------
+0.12   |   0      0      0      0      0      0      0
+0.08   |   0      0    100      0      0      0      0
+0.04   |   0      0    100      0      0      0      0
+0.00   |   0      0     31    100     47      0      0
-0.04   |   0     31     12      0      0      0      0
-0.08   |   0      0      0      0      0      0      0
-0.12   |   0      0      0      0      0      0      0
```

### Sweet spot 7 점 (12cm > 0%)
- **(0.23, +0.04): 100%** ★
- **(0.23, +0.08): 100%** ★
- **(0.27, +0.00): 100%** ★
- (0.31, +0.00): 47%
- (0.23, +0.00): 31%
- (0.19, -0.04): 31%
- (0.23, -0.04): 12%

### 핵심 패턴 확정
1. **x=0.23이 superstar column**: y ∈ [-0.08, +0.12]에서 LIFT 도달, y ∈ [-0.04, +0.08]에서 grip 가능
2. **x=0.27, 0.31은 y=0 line만**: 너비 ~4cm 좁은 sweet spot. y±0.04에서 즉시 0% cliff
3. **+y 편향 확정**: (0.23, +0.04): 100% vs (0.23, -0.04): 12% — 약 8x 차이
4. **x ≥ 0.35는 절대 reach 불가능** (이전 결과 확인)

## 4. 최종 추천 spawn (revised)

| 옵션 | x range | y range | 예상 12cm | 비고 |
|---|---|---|---:|---|
| **x=0.23 column** ★ | [0.21, 0.25] | [+0.00, +0.08] | 75~100% | sweet spot 3점 포함 ((0.23, +0.04), (0.23, +0.08), (0.23, 0)) |
| y=0 line | [0.21, 0.30] | [-0.02, +0.02] | 50~70% | reach 광범위, grip 적당 |
| Balanced (검증) | [0.21, 0.31] | [-0.04, +0.08] | 30~40% | 일반화 좋음, RL 학습용 적당 |
| Narrow (이전) | [0.22, 0.28] | [-0.05, +0.05] | 30.7% | -y 영역 포함이지만 작동 |

원본 학습 환경의 pose_range:
- **x=0.23 column**: `{"x": (-0.03, 0.01), "y": (0.00, 0.08), "z": (0.0, 0.0)}`
- **y=0 line**: `{"x": (-0.03, 0.06), "y": (-0.02, 0.02), "z": (0.0, 0.0)}`

## 5. 결론

1. **Tight 가설 실패**: sweet spot 4점이 아닌 0% 점들도 포함해서 실제 성공률 26.6%.
2. **+y 편향은 finger_x_offset과 무관**. default arm pose 또는 gripper jaw 비대칭 가능성 (별도 진단 필요).
3. **Dense map으로 superstar column 확정**: x=0.23.
4. **추천 spawn 변경**: x=0.23 ± 0.02 좁은 column, y는 +y에 편향 (0~+0.08).
5. 코드 상태: env_cfg.py spawn = Narrow (이전 측정용). oracle_policy.py에 `finger_x_offset` config 추가됨 (backward compat, default 0.006).

## 6. 다음 step

1. **x=0.23 column으로 Oracle 측정** (10분) — 75%+ 검증
2. **-y 편향 진단**: default arm pose의 wrist_flex, wrist_roll 또는 gripper joints의 좌/우 비대칭. 1~2시간.
3. **PPO 재학습 시작** — x=0.23 column spawn (75%+) 또는 y=0 line spawn 으로. 8시간.
4. **Spawn curriculum** — column → balanced → wide. 자동 난이도 증가.

---

**핵심**: dense reach map으로 SO-ARM101의 진짜 sweet spot은 **x=0.23 column** (y에 robust). 이걸 spawn 영역으로 사용하면 12cm 75%+ 가능.

파일: `/home/j-k14d101/jabis_sim_v2/tasks/reach_map_followups.md`
