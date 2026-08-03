# Exp 2 설계 — Resource Imbalance (P/D 노드 간 HBM 불균형 해소)

> 주장하고 싶은 것: **(A)** GPU 간 KV 포화도가 균일해졌다, **(B)** KV가 여유 있는 곳으로 갔다.
> 이 문서는 그 두 문장을 *반박 가능한* 측정으로 바꾸는 설계다.

---

## 0. 먼저 — 이 실험의 가장 큰 함정

**"포화도가 균일해졌다"는 그 자체로는 아무것도 증명하지 못한다.**

유휴 GPU에 KV를 무조건 쏟아붓는 정책이면 점유율은 *정의상* 균일해진다. 리뷰어의 질문은
"놀던 메모리를 채웠다는 건 알겠는데, 그래서 뭐가 좋아졌나?"이고, 균일성 그래프 하나로는
답이 안 된다.

따라서 균일성(M2)은 **반드시 유용성(M4)과 짝으로** 보고해야 한다:

| | 무작정 채우는 정책 | 우리 정책 |
|---|---|---|
| 균일성 (M2) | **좋음** | 좋음 |
| park→fetch 전환율 (M4) | **나쁨** | 좋음 |

→ 이 표를 실제로 만들기 위해 **`park_pd_blind` arm(압력 신호 끄기)을 반드시 넣는다.**
이 arm이 "균일하지만 쓸모없음"을 보여줘야 우리 arm의 균일성이 의미를 갖는다.

---

## 1. 코드 조사 결과 — 무엇이 되고 무엇이 안 되나

`python/sglang/srt/disaggregation/idle_kv_parking.py` 및 `start_2P_2D.sh` 확인 결과:

### ✅ 이미 되는 것 (코드 수정 불필요)

- **decode GPU를 park 대상으로 지정 가능.** `SGLANG_KV_PARK_GPUS`는 임의의 GPU 리스트를
  받고 `_init_park_gpu_pool()`이 그 GPU마다 pool을 만든다 (`idle_kv_parking.py:1174`).
  `start_2P_2D.sh:147`의 `PARK_GPUS_P0` / `PARK_GPUS_P1` 오버라이드가 이미 존재한다.
- **decode GPU의 점유율을 prefill이 읽을 수 있다.** `_publish_usage_loop`는
  *모든 role*에서 뜬다 (`:689` — "Runs for all roles"). D0/D1도 `usage_gpu2.txt`,
  `usage_gpu3.txt`를 0.5초마다 쓴다. 즉 **압력 인지 배치가 필요로 하는 신호가 이미 있다.**
- **GPU 간 KV 점유 시계열 샘플러 존재.** `benchmark/kv_occupancy_timeseries.py`가
  P0/P1/D0/D1의 `token_usage`, `num_used_tokens`, `num_running_reqs`를 긁는다. M1/M2의 원천 데이터.

### ⚠️ 막히는 것 (반드시 고쳐야 실험이 성립)

**(a) decode GPU에 park pool 놓을 HBM이 없다 — 치명적**

`MEMFRAC_P`(`--mem-fraction-static 0.70`)는 **prefill에만** 붙는다
(`start_2P_2D.sh:206,219`). Decode는 기본값(~0.87)으로 떠서 49 GB 중 42.6 GB를 이미 잡고
있다 (기존 exp2 table.md의 `peak gpu2/gpu3 HBM = 42.63`). **park pool을 만들 자리가 없어
OOM 난다.**
→ `PARK_MEM_FRACTION_D`(제안: 0.70)를 추가하고 D0/D1 launch에 붙일 것.

**(b) `BW_AWARE=1`이 decode GPU를 후보에서 강등시킬 수 있다 — 설계를 바꾸는 문제**

`_select_pool()`의 정렬 키는 `(slow_link, serving_usage, -headroom)`이고, `slow_link`는
"이 링크가 host DRAM보다 느린가"다. 측정치: NVLink 27–53 GB/s, **non-NVLink peer 3.3 GB/s**,
pinned host 26 GB/s. → **PCIe로만 붙은 GPU는 무조건 후순위로 밀린다.**

A6000 4장은 보통 브릿지가 **쌍**으로 붙는다(0-1, 2-3). 현재 배치가 P={0,1}, D={2,3}이면
**모든 P→D 링크가 PCIe-only**라서, `park_pd` arm이 `park_local`로 퇴화한다.

→ **해결: GPU 배치를 바꿔 P와 D를 브릿지 쌍으로 묶는다.**

| | 현재 | 제안 |
|---|---|---|
| GPU0 | P0 | P0 |
| GPU1 | P1 | **D0** |
| GPU2 | D0 | **P1** |
| GPU3 | D1 | D1 |

→ 브릿지가 (0,1),(2,3)이면 P0↔D0, P1↔D1이 각각 NVLink. `BW_AWARE=1`(정직한 기본값)을
유지한 채로 P→D 파킹이 27–53 GB/s로 동작한다.

> **선행 확인 필수** (서버17에서 1분): `nvidia-smi topo -m`
> `NV1`/`NV2`/`NV4`가 어느 쌍에 있는지 보고 위 매핑을 확정한다. 전부 `PHB`/`SYS`(브릿지 없음)로
> 나오면 P/D 파킹은 3.3 GB/s로만 가능하고, 그때는 "능력은 보이되 이 하드웨어에선 경제성이
> 없다 + 인터커넥트 의존성"으로 정직하게 서술해야 한다 (§6 참조).

**(c) 후보 GPU 수가 늘면 park HBM 총량이 늘어 arm 비교가 불공정**

`PARK_PEER=1` 분기는 `PARK_POOL_TOKENS`를 정확히 이 이유로 반으로 나눈다
(`start_2P_2D.sh:141-143`). 그런데 `PARK_GPUS_P0` 오버라이드 분기(`:147`)는 **나누지 않는다.**
`PARK_GPUS_P0=0,1,3`으로 주면 pool이 3개 × 30000 토큰 = **park HBM 3배**가 되고,
"우리 arm이 이긴 건 그냥 캐시가 3배라서"가 된다.
→ 후보 수로 `PARK_POOL_TOKENS`를 나누도록 일반화할 것 (PEER 분기와 동일 로직).

**(d) 배치 *결정* 로그가 없다 — (B) 주장을 측정할 수단이 없음**

`_select_pool()`은 고른 pool만 반환하고, **후보들의 그 순간 점유율은 버려진다.**
"여유 있는 곳으로 갔다"를 증명하려면 *선택된 곳*뿐 아니라 *선택되지 않은 곳*이 필요하다.
→ park 결정마다 한 줄씩 JSONL로 남길 것 (§4-M3).

---

## 2. 부하 설계 — 불균형을 "바라지" 말고 "만들어라"

기존 `run_exp2_imbalance.sh`는 P0/P1 사이 skew를 라우터 2개로 강제한다. 하지만 **이번 질문은
P↔P가 아니라 P↔D 불균형**이고, 이건 강제할 필요가 없다. 구조적으로 이미 존재하기 때문이다:

- `pd_hbm_occupancy.py` 관측: **"turn 종료 후 D는 token usage→0으로 복귀 (turn 사이 KV 미보존)"**
- 즉 **tool think-time 3초 동안 D 노드의 KV 풀은 거의 비어 있고, 같은 시각 P 노드는
  radix 캐시로 포화 상태**다.

→ **수요가 반(反)상관(anti-correlated)이다.** 이건 빌려쓰기에 이상적인 조건이고, 워크로드
트릭이 아니라 PD disaggregation의 구조적 성질이라서 논문에서 더 강하다.

- 워크로드: ShareGPT multi-turn, `TOOL_DELAY_SEC=3`, `CONCURRENCY=16`
- `PREFILL_MAX_TOTAL_TOKENS=60000` (§Exp1과 동일 — P측 축출이 실제로 일어나는 조건)
- `DECODE_MAX_TOTAL_TOKENS` 미설정 (D는 여유롭게 → 그게 빌려줄 헤드룸)
- 라우터는 **balanced 단일 라우터** (skew 불필요). 라우터 2개 안 쓰므로 예전 실패 원인
  하나가 자동 제거됨
- **`FLOOD_DURING_DELAY=0` 필수** (기존 스크립트 주석의 교훈 — 이걸로 sweep 하나를 날렸다)

---

## 3. Arm 구성 (5개)

전부 **동일 서버 설정 + 동일 park HBM 총량**, 차이는 오직 *배치가 어디로 갈 수 있는가*.

| arm | park 후보 GPU | 압력 인지 | 이 arm이 답하는 질문 |
|---|---|---|---|
| `hicache` | — (파킹 없음) | — | 불균형이 실재하고 낭비되는가? **(M1의 기준선)** |
| `park_local` | 자기 GPU만 | — | 파킹 자체의 효과 vs 배치의 효과 분리 |
| `park_pp` | P0+P1 | O | 기존 peer 주장 (P↔P) — *선택적, 시간 없으면 생략* |
| **`park_pd`** | 자기 P + D0 + D1 | **O** | **← 사용자 질문의 본 arm** |
| **`park_pd_blind`** | 자기 P + D0 + D1 | **X** (`PRESSURE_AWARE=0`) | **균일성이 gameable함을 보이는 대조군** |

`park_pd_blind`가 왜 핵심인가: 후보는 같지만 헤드룸만 보고 고른다(실시간 점유 무시).
→ 점유는 비슷하게 균일해지지만 **엉뚱한 시점에 바쁜 GPU를 때린다.** M2는 비슷하고 M4는
갈라져야 한다. 이게 갈라지지 않으면 압력 인지 배치는 논문에서 뺄 근거가 된다 (= 반증 가능).

**실행 시간 추정**: arm당 (기동 4분 + 벤치 20분 + 정지 1분) ≈ 25분 → **5 arm ≈ 2시간 10분.**

---

## 4. 측정 지표 — 4계층

### M1. 낭비된 헤드룸 (전제 확립, `hicache` arm에서 측정)

> "P가 축출하고 있던 바로 그 순간, 클러스터 어딘가에 놀고 있던 메모리가 얼마였나?"

각 샘플 t, GPU g에 대해 여유 토큰 `F_g(t) = (1 − use_g(t)) × cap_g`:

```
W(t) = 1[ max_g use_g(t) > 0.90 ] · Σ_{g : use_g(t) < 0.50} F_g(t)
stranded = ∫ W(t) dt        [GB·초]
```

- **이 숫자가 논문의 전제 그 자체다.** 작으면 실험 전체가 성립 안 하므로 **가장 먼저 확인**
- 데이터 원천: `occ_hicache.csv` (기존 `kv_occupancy_timeseries.py`, 신규 코드 0줄)
- 부수 지표: `hicache` arm의 D 노드 평균 점유율 (예상: 매우 낮음 → 빌릴 곳이 있다는 직접 증거)

### M2. 불균형 지수 (헤드라인 패널)

4개 GPU의 KV 포화도 `use_g(t)`에 대해 매 샘플:

- **CoV** = `std/mean` — 시간 평균. 낮을수록 균일. 막대 그래프에 가장 읽기 쉬움
- **spread P95** = `percentile_95( max_g use − min_g use )` — 최악 순간의 격차
- (선택) **Jain fairness** `J = (Σx)²/(n·Σx²)` — 자원 할당 문헌의 표준 지표라 리뷰어가 알아봄

> 단위 주의: **포화도(분율)** 로 계산한다. P 풀은 60k로 캡했고 D 풀은 캡이 없어 절대 바이트는
> 사과-오렌지 비교다. 사용자가 물은 "포화도"가 정확히 분율이므로 이게 맞다.
> 단, **절대 여유 GB도 같이 보고**한다 — 헤드룸이 실제로 쓸 만한 크기인지는 절대량이 결정.

### M3. 배치가 헤드룸을 따라가는가 (= 주장 (B)) — **신규 로깅 필요**

park 결정마다 JSONL 한 줄:

```json
{"t": 12.34, "src_gpu": 0, "chosen": 2, "tokens": 1876,
 "cand": {"0": {"use": 0.94, "head": 3200, "slow": 0},
          "2": {"use": 0.11, "head": 14800, "slow": 0},
          "3": {"use": 0.63, "head": 14800, "slow": 0}}}
```

여기서 나오는 지표:

- **선택 정확도** = `P(chosen == argmin use | fast-link 후보 중)`
- **점유율 격차** = `mean(use[chosen]) − mean(use[rejected])`. **음수여야 하고, 클수록 좋다.
  주장 (B)를 한 숫자로 요약하는 값.**
- **널 모델 대비**: 같은 로그에서 random / round-robin / always-local을 후향 계산.
  추가 실행 0회로 "우연이 아님"을 보이는 가장 싼 증거
- **P→D 비율**: 전체 파킹 바이트 중 decode GPU로 간 비율 (= P/D 불균형 해소의 직접 측정)

구현: `_select_pool()` 반환 직전에 후보별 `(use, headroom, slow_link)`를 이미 계산해 두므로
그 튜플을 파일에 append만 하면 된다. `SGLANG_KV_PARK_DECISION_LOG=<path>`로 게이팅해
기본 off (핫 패스). ~25줄.

### M4. 유용성 (M2의 gameability 방어 — **빼면 안 됨**)

- **park→fetch 전환율** = `fetched_tokens / parked_tokens` (`parked_gpu*.json`에 이미 있음)
- **fetch 출처 분해**: local / peer / host (`fetch_local_hits`, `fetch_peer_hits`, `fetch_host_hits` — 존재)
- **버려진 바이트** `dropped_bytes` (존재)
- **fetch 평균 지연** `fetch_ms_sum / fetch_hits` — P→D가 PCIe면 여기서 드러난다
- **성능**: cache hit rate, TTFT p50/p95, goodput

---

## 5. 그림 (2패널, 2-column)

**(a) 포화도 시계열** — `hicache` vs `park_pd`, 각 4개 선(P0/P1/D0/D1), 위아래 2단.
   기준선 arm에서 P는 천장에 붙어 있고 D는 바닥을 기는 그림 → 우리 arm에서 좁혀지는 그림.
   **이 그림 한 장이 (A)를 말한다.**

**(b) 배치 결정 산점도** — x축 = 선택 시점 후보 GPU의 점유율, y축 = 선택 확률(또는
   선택/거부 히스토그램 2개). 강한 음의 관계 → **(B)를 말한다.**
   `park_pd_blind`를 회색으로 겹쳐 그리면 대조가 한눈에 보인다.

M2/M4 숫자는 본문 표. 막대 그래프는 굳이 안 만든다 (패널 2개로 충분).

---

## 6. 예상되는 실패 모드와 사전 대응

| 위험 | 징후 | 대응 |
|---|---|---|
| P/D 간 NVLink 없음 | `nvidia-smi topo -m`이 전부 PHB/SYS | GPU 재배치 불가 → `BW_AWARE=0` arm 추가하고, "메커니즘은 동작하나 이 박스의 P→D 링크(3.3 GB/s)가 host DRAM(26)보다 느려 경제성 없음"으로 **정직하게 한계 서술**. 이건 실패가 아니라 인터커넥트 의존성이라는 발견 |
| M1이 작게 나옴 | `stranded` ≈ 0 | 전제 붕괴. `CONCURRENCY`↑ 또는 `PREFILL_MAX_TOTAL_TOKENS`↓로 P측 압력을 더 준다. **이건 M1을 먼저 재는 이유** |
| D 서버 OOM | 기동 실패 | `PARK_MEM_FRACTION_D` 더 낮추기 (0.70 → 0.62) |
| `park_pd_blind`가 안 갈라짐 | M2/M4 모두 동일 | 압력 인지가 무효라는 뜻. 논문에서 해당 주장 삭제 (반증됨) |
| 예전처럼 503 대량 실패 | 턴 실패율 >5% | `collect_arm_metrics.py`가 이미 거부함. `start_2P_2D.sh`의 라우터 준비 게이트는 수정 완료됨 |

---

## 7. 착수 순서

1. **[서버17, 1분]** `nvidia-smi topo -m` → 브릿지 쌍 확정 → GPU 매핑 결정
2. **[코드]** `start_2P_2D.sh`: `PARK_MEM_FRACTION_D` 추가 + 후보 수로 pool 토큰 나누기 + GPU 매핑 파라미터화
3. **[코드]** `idle_kv_parking.py`: `_select_pool()` 결정 로그 (~25줄, 기본 off)
4. **[코드]** `scripts/sglang/run_exp2_pd_imbalance.sh` (기존 스크립트 기반, 라우터 skew 제거)
5. **[코드]** `benchmark/collect_imbalance.py` — occ CSV + 결정 로그 → M1~M4 표
6. **[서버17, 25분]** `hicache` arm만 먼저 → **M1 확인**. 전제가 서면 나머지 진행
7. **[서버17, 2시간]** 전체 5 arm
8. **[코드]** `benchmark/plot_imbalance.py` → 2패널 그림

**6번에서 한 번 끊는 것이 중요하다.** M1이 작으면 나머지 2시간은 버리는 시간이고, 부하 조건을
먼저 고쳐야 한다.
