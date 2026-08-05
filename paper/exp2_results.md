# Exp 2 결과 — P/D 불균형

## ★★ 큰 park 풀 (`results/exp2/big_c32_r1`) — 유휴 메모리를 실제로 쓸 때

`PARK_POOL_TOKENS_PER_GPU=32000`, `PARK_MEM_FRACTION_D=0.88`, C=32, 288세션×12턴.
클램프가 prefill 자기 GPU만 15,344로 줄였고(여유 HBM 부족), decode GPU에는 32,000이
그대로 들어갔다. 8개 서버 전부 `ready to roll`, 두 prefill 모두 ~950초 완주, 에러 0.

| | `park_local` | **`park_pd` (Ours)** | Δ |
|---|---|---|---|
| prefill이 도달 가능한 캐시 | 15,344 tok (2.01 GB) | **79,344 tok (10.40 GB)** | **5.2×** |
| park 총량 | 4.02 GB | **20.79 GB** | 5.2× |
| — 유휴 decode GPU 위 | 0.00 | **16.78 GB** | — |
| **park-fetch miss** | 522 | **321** | **−38.5%** |
| **park-fetch hit rate** | 64.0% | **77.9%** | **+13.9%p** |
| TTFT p50 | 0.301 | **0.277** | −8.2% |
| TTFT p95 | 3.609 | 4.806 | +33.2% ⚠ |
| TTFT p99 | 11.372 | 11.780 | +3.6% ⚠ |
| throughput | 834.3 | 842.8 | +1.0% |

### 용량을 주니 효과가 비례해서 커졌다 — 메커니즘은 수요가 아니라 용량에 막혀 있었다

| park 크기 | park-fetch miss | hit rate |
|---|---|---|
| 10,000 tok/GPU (3회 반복) | −5.1 ~ −12.5% | +1.8 ~ +5.5%p |
| **32,000 tok/GPU** | **−38.5%** | **+13.9%p** |

작은 풀에서 park 풀이 항상 100% 포화였던 것이 수요 부족이 아니라 **용량 부족**이었음을
확인해 준다.

### 핵심: 로컬 GPU에는 자랄 자리가 없고, 유휴 GPU에는 있다

두 arm 모두 "GPU당 32,000 토큰"을 요청했지만 결과가 다르다.

- `park_local`은 **자기 prefill GPU 하나뿐**이고 그 GPU는 여유 HBM이 6.05 GB라
  15,344로 클램프됐다 → 2.01 GB
- `park_pd`는 **유휴 decode GPU 2개에 32,000씩 그대로** 받았다 → 10.40 GB

**같은 정책, 같은 요청 크기, 5.2배 차이.** 차이를 만든 것은 정책의 공격성이 아니라
**닿을 수 있는 GPU가 몇 개인가**다. 이것이 §7-2의 주장이다.

### 메모리는 여전히 만들어낸 것이 아니라 재분할이다

decode GPU의 KV 가용 메모리 총량:

| | serving 풀 | park 풀 | 합 |
|---|---|---|---|
| `park_local` | 26.41 GiB | 0 | **26.41** |
| `park_pd` | 18.84 GiB | 7.81 GiB | **26.65** |

decode serving 풀이 park 풀만큼 정확히 줄었다. decode 점유는 15.4% (p95 24.9%)로
여전히 여유가 크다 — 압박을 옮긴 것이 아니다.

유휴 활용률: **9% → 29%** (decode GPU의 idle-ish 26.6 GiB 중 7.81 GiB).

### ⚠ 꼬리는 1회 측정이다 — 인용 금지

p95 +33.2%, p99 +3.6%. 방향은 3회 반복(p99 3/3 악화)과 일치하지만 **이 run은 n=1이고,
기준선 p95는 단독으로 4.5배(1.45~6.58s) 흔들린 전력이 있다.** +33%는 그 변동폭 안이다.
꼬리를 논문에 쓰려면 이 설정으로 3회 반복해야 한다.

> **그림**: `results/exp2/fig_exp2_timeline_big.{pdf,png}`, `fig_exp2_stack_big.{pdf,png}`
> `python benchmark/plot_exp2_timeline.py --dirs results/exp2/big_c32_r1`

---

## ★ 최종 결과 — 3회 반복 (`results/exp2/repeat_nocap_c32_r{1,2,3}`)

C=32, 288세션×12턴, **`--max-total-tokens` 없음**, prefill mem-fraction 0.85,
park 풀은 **건드리는 모든 GPU에 10,000 토큰씩**. 반복마다 **서로 겹치지 않는 ShareGPT
슬라이스** (`ITEM_OFFSET` 0/288/576). 전 실행 에러 0.

| | `park_local` | **`park_pd` (Ours)** | 3회 판정 |
|---|---|---|---|
| park 총량 | 2.62 GB | **7.86 GB** | 3회 동일 — **설정값, 결과 아님** |
| — 유휴 GPU 위 | 0.00 | **5.24 GB (67%)** | 3회 동일 |
| **park-fetch miss** | 762 / 751 / 763 | **723 / 692 / 668** | ✅ **3/3 개선** (−5.1/−7.9/−12.5%) |
| **park-fetch hit rate** | 46.9 / 45.7 / 44.7% | **48.7 / 49.6 / 50.2%** | ✅ **3/3 개선** (+1.8/+3.8/+5.5%p) |
| TTFT p50 | 0.317 / 0.316 / 0.299 | 0.315 / 0.271 / 0.277 | ⚠ 3/3 개선이나 **r1은 −0.7%로 노이즈 내부** |
| **TTFT p95** | 1.82 / 3.08 / 1.45 | 4.33 / 1.97 / 2.39 | ❌ **SPLIT 1승 2패** |
| **TTFT p99** | 6.32 / 7.05 / 4.10 | 8.22 / 8.33 / 7.30 | ❌ **3/3 악화** (+30%) |
| throughput | 834 / 763 / 804 | 843 / 759 / 805 | ➖ 변화 없음 (median 0.5%) |

### ⛔ 단일 측정의 "p95 −70%"는 취소한다

이전 `final_nocap_c32` 1회 실행은 p95 **6.58 → 2.01 (−69.5%)**, p99 **13.65 → 7.75
(−43.2%)**를 보고했다. **둘 다 재현되지 않았고, 원인은 기준선의 이상치였다.**

결정적 증거 — **r1은 `ITEM_OFFSET=0`으로 원본과 동일한 슬라이스·동일한 설정이다**
(출력 토큰 798,564 vs 803,194):

| `park_local` p95 | 같은 워크로드, 같은 설정 |
|---|---|
| `final_nocap_c32` | **6.582 s** |
| `repeat_r1` | **1.824 s** |
| | **3.6× 차이 — 순수 run-to-run 변동** |

4회 관측 전체에서 기준선 p95는 **1.45 ~ 6.58 s (4.5×, CoV 0.72)** 범위다.
−70%는 **기준선이 4개 관측 중 최악값을 뽑은 결과**였지 효과가 아니다.
`park_local`이 포화된 자기 GPU에서 할당 경합을 일으킨다는 §"꼬리 역전 가설"도
함께 폐기한다 — 설명해야 할 현상이 존재하지 않는다.

### 꼬리는 비용이다 (개선이 아니라)

p99는 **3회 모두 악화**했다 (6.32→8.22, 7.05→8.33, 4.10→7.30, median +30%).
원격 fetch가 가장 느린 요청들에 지연을 더한다는 해석과 일치한다.

한 가지 관찰: `park_pd`의 p99는 4회 관측에서 **7.30~8.33 s (CoV 0.06)**로 극히 안정적인
반면 `park_local`은 **4.10~13.65 s (CoV 0.53)**로 요동친다. "꼬리를 낮추지는 않지만
꼬리에 천장을 씌운다"는 해석이 가능하다. **그러나 이 해석은 13.65 s 관측 하나에
전적으로 의존한다** — 그 하나를 빼면 기준선 p99는 4.10~7.05로 우리보다 항상 낫다.
**방금 −70%를 취소한 것과 정확히 같은 오류이므로, 반복을 더 하기 전에는 쓰지 않는다.**

### 남는 주장 (이건 견고하다)

**메모리 배치와 캐시 회수는 3회 모두 일관되게 개선됐고, 효과가 기준선 자체의 변동폭보다
크다** — 유일하게 그 조건을 만족하는 지표들이다:

| 지표 | 효과 (median) | 기준선 자체 변동폭 | |
|---|---|---|---|
| park-fetch miss | **7.9%** | 1.6% | ✅ 효과가 5배 크다 |
| park-fetch hit rate | **8.4%** | 4.8% | ✅ 효과가 크다 |
| TTFT p50 | 7.5% | 5.7% | ⚠ 아슬아슬 |
| TTFT p95 | 64.7% | **89.4%** | ❌ 노이즈에 묻힘 |
| TTFT p99 | 30.0% | **46.8%** | ❌ 노이즈에 묻힘 |
| throughput | 0.5% | 8.9% | ❌ 노이즈에 묻힘 |

### 인용 문장 (반복 반영)

> With no artificial cap on either pool (25.0 GB prefill against 22.6 GB decode), the
> prefill pools run at 82% occupancy for 70% of the run while the decode pools sit at
> 14%. GPU-first placement moves **5.24 GB of reusable KV — two thirds of the resident
> cache — onto those otherwise-idle GPUs**, tripling the cache from 2.62 to 7.86 GB.
> Across three repeats on disjoint workload slices, park-fetch misses fall by
> **5.1–12.5%** and the park-fetch hit rate rises by **1.8–5.5 pp**, in every repeat.
> Throughput is unchanged. **Tail latency is not improved: p99 is 30% worse in all three
> repeats, and p95 varies by more between repeats of the baseline alone (1.45–3.08 s)
> than it does between the two policies.**

> ⚠ **용어 주의 — Exp 1의 hit rate와 단위가 다르다. 절대 나란히 비교하지 말 것.**
>
> | | Exp 1 "cache hit rate" | Exp 2 "park-fetch hit rate" |
> |---|---|---|
> | 정의 | `cached_tokens / prompt_tokens` | `fetch_hits / (fetch_hits + fetch_miss)` |
> | 가중 | **토큰** 가중 | **요청** 가중 |
> | 분모 | 전체 요청 | radix가 **못** 준 요청만 |
>
> park 카운터는 prefill 요청을 4개 배타 버킷으로 나눈다 (final_nocap_c32, park_pd):
> `park-hit 715 + miss 713 + radix-already 56 + nospace 0 = 1484`.
> 도표의 50.1%는 `715/(715+713)`으로 **radix-already를 분모에서 뺀 값**이다.
> 전체 요청 기준으로는 `(715+56)/1484 = 52.0%`이며, 두 값의 차이는 3.7%p로 작지만
> **단위 자체가 Exp 1과 다르므로** "Exp 1은 55.7%인데 Exp 2는 50.1%밖에 안 된다"는
> 식의 독해가 가능한 라벨을 쓰면 안 된다.
>
> 특히 **v5의 park-fetch hit rate 55.7%는 Exp 1의 token-weighted hit rate 55.7%와
> 숫자가 우연히 일치한다** (아래 §v5, `exp2_design.md:196`). 서로 다른 두 지표가 같은
> 숫자로 같은 이름을 달고 있으므로, 논문에서는 반드시 *park-fetch hit rate*로 표기한다.
> 코드/그림 라벨은 수정 완료 (`plot_exp2_gpu_stack.py`, `plot_exp2_harvest.py`,
> `collect_repeats.py`).

---


> **그림**: `results/exp2/fig_exp2_stack.{pdf,png}`
> `python benchmark/plot_exp2_gpu_stack.py --dirs 'results/exp2/repeat_nocap_c32_r*'`
> — **3회 반복의 median**을 그리고, 막대 위에 각 반복 값을 점으로 찍는다. 단일 실행을
> 인용하던 버전은 "p95 3.3× 개선" 캡션을 달았고 그건 틀린 진술이었다.
>
> **최종 주장 (§7-2)**: GPU-first placement는 배치가 헤드룸을 정확히 따라가며
> (usage gap −0.31, 널 모델 대비 2배 이상), 유휴 decode GPU로 **재사용 가능 KV의 3배**를
> 옮겨 **park-fetch miss를 3회 반복 전부에서 5.1~12.5% 줄인다.**
> 처리량은 불변. **꼬리 지연은 개선하지 못하며, p99는 3회 모두 30% 악화한다.**

---

5 arm, Llama-3.1-8B, ShareGPT C=16, PD_LAYOUT=b, 전 arm 에러 0.

---

## 1. 결론 요약

| 주장 | 결과 |
|---|---|
| **(B) KV가 여유 있는 곳으로 간다** | ✅ **증명됨.** usage gap **−0.403**, 파킹의 98.5%가 decode GPU로 |
| **(A) 포화도가 균일해졌다** | ❌ **측정 안 됨.** M1/M2가 잘못된 지표였다 (§3) |
| 압력 인지가 필요한가 | ✅ **필요.** 끄면 98.4%가 자기 포화 GPU로 감 |
| bandwidth 강등이 옳았나 | ✅ 옳았다 (§5) |

---

## 2. M3 — 배치는 헤드룸을 따라간다 (핵심 결과)

| arm | usage gap | 선택된 GPU 점유 | 거부된 GPU 점유 | cross-GPU | 파킹 대상 |
|---|---|---|---|---|---|
| **`park_pd`** | **−0.403** | **0.063** | 0.466 | **98.5%** | gpu1 70.8% + gpu3 27.7% (**둘 다 decode**) |
| `park_pd_blind` | +0.799 | 0.876 | 0.077 | 1.6% | gpu2 98.4% (**자기 GPU, 포화 상태**) |
| `park_slowlink` | −0.029 | 0.826 | 0.855 | 3.2% | 자기 GPU |
| `park_local` | — | 0.836 | — | 0% | 후보가 1개라 gap 정의 불가 |

**널 모델 대비** (각 정책이 고를 GPU의 평균 점유, 낮을수록 좋음):

| arm | policy | random | round-robin | always-local |
|---|---|---|---|---|
| **`park_pd`** | **0.063** | 0.348 | 0.331 | 0.842 |
| `park_pd_blind` | 0.876 | 0.362 | 0.344 | **0.877** ← 정책과 일치 |

- `park_pd`는 **어떤 널 모델보다도 5배 이상 한가한 GPU를 고른다.** 우연이 아니다
- `park_pd_blind`는 **always-local 널과 소수점 4자리까지 일치**(0.8764 vs 0.8766). 즉
  헤드룸만 보는 선택은 *자기 포화 GPU를 고르는 것과 수학적으로 같은 행동*이다.
  → **압력 신호가 하는 일이 정확히 무엇인지를 보여주는 대조군.** 설계 의도대로 작동

> ⚠ `park_pd_blind`는 **P0의 결정 로그가 없다** (`decisions_park_pd_blind.gpu0.jsonl` 부재,
> `parked_*.csv`에서도 gpu0/gpu1 = 0). P0이 한 번도 파킹하지 않았다. 744 parks는 전부 P1의 것.
> 결론의 방향은 바뀌지 않지만 **원인 규명 전에는 이 arm을 "한쪽 prefill만의 관측"으로 서술해야 한다.**

---

## 3. M1/M2는 잘못된 지표였다 — 설계 오류

M1(stranded headroom)과 M2(CoV)는 **serving 풀 점유율**로 계산한다. 그런데 **park된 KV는
별도의 park 풀에 산다.** `/metrics`의 어떤 gauge도 park 풀을 보지 않는다.
→ **파킹이 아무리 잘 동작해도 M1/M2는 원리적으로 움직이지 않는다.**

수치가 이를 확정한다:

| arm | decode 풀 크기 | M1 평균 | ΔPool | ΔM1 |
|---|---|---|---|---|
| `hicache` | 45.28 GB | 42.42 GB | — | — |
| `park_local` | 45.28 | 42.38 | +0.00 | −0.04 |
| **`park_pd`** | **40.30** | **37.39** | **−4.98** | **−5.03** |
| `park_pd_blind` | 40.30 | 37.41 | −4.98 | −5.01 |
| `park_slowlink` | 45.28 | 42.37 | +0.00 | −0.05 |

**ΔM1(−5.03) = ΔPool(−4.98).** M1이 줄어든 유일한 이유는 park 풀을 할당하느라 decode의
serving 풀이 그만큼 작아진 것이다. **KV가 여유 공간으로 옮겨간 몫은 0 GB다.**

M2도 같다: CoV 0.846 → 0.829. 파킹은 serving 풀을 건드리지 않으므로 변할 수가 없다.

**→ Exp 2의 그림은 M1/M2 막대가 아니라 M3(배치 결정)과 residency로 그려야 한다.**
M1은 §Fig 0(동기)에서만 쓴다: "낭비가 이만큼 있다"는 baseline 진술로는 여전히 유효하다.

---

## 4. 불편한 결과 — `park_pd`가 `park_local`보다 KV를 *덜* 회수한다

| arm | park 총량 | fetch hits | fetched tokens | peer 비율 | TTFT p50 |
|---|---|---|---|---|---|
| `hicache` | — | — | — | — | 0.2903 |
| **`park_local`** | **7.86 GB** | **594** | **502,185** | 47.8% | **0.2359** ← 최고 |
| `park_pd` | 3.67 GB | 491 | 428,809 | **99.4%** | 0.2415 |
| `park_pd_blind` | 2.62 GB | 449 | 357,328 | 49.0% | 0.2554 |
| `park_slowlink` | 3.93 GB | 571 | 487,462 | 100% | 0.2392 |

**원인: `_select_pool()`의 정렬 키가 꽉 찬 풀을 빈 풀보다 먼저 고른다.**

```python
key = (slow_link(p), round(serving_usage, 2), -p.headroom())
```

점유율이 **헤드룸보다 먼저** 비교된다. 그래서 헤드룸이 0인 decode 풀(점유 0.06)이
헤드룸이 가득한 로컬 풀(점유 0.86)을 계속 이긴다. 결과:

- `park_pd` 실측 배치: gpu1 **1.31 GB (=10k 토큰, 꽉 참)**, gpu3 **1.31 GB (꽉 참)**,
  gpu0 1.05, **gpu2 0.00**
- 예산은 prefill당 30,000 토큰(≈3.93 GB)인데 **실제로 채운 건 3.67/7.86 GB**
- 즉 **park 용량의 절반이 자기 GPU에 놀고 있고**, 그동안 두 개의 작은 decode 풀이
  서로를 축출하며 thrash 중이다 → fetch hits가 `park_local` 대비 **−17%**

**고칠 지점 (둘 중 하나):**
1. `-p.headroom()`을 키의 **두 번째**로 올려 헤드룸 0인 풀을 배제 — 가장 작은 수정
2. 예산을 후보 수로 균등 분할하지 말고 유휴 GPU에 더 크게 배정

→ **이건 실험의 실패가 아니라 정책의 버그다.** 배치 방향은 옳고(§2), 용량 배분이 틀렸다.

---

## 5. `park_slowlink` — bandwidth 강등은 옳았다

`BW_AWARE=0`으로 PCIe 너머 배치를 강제한 arm. usage gap이 **−0.029**에 그친다
(`park_pd`의 −0.403 대비). peer prefill도 함께 포화 상태라 옮겨봐야 한가한 곳이 없다.
TTFT는 0.2392로 `park_local`(0.2359)보다 나쁘다.
→ **느린 링크 너머로 옮길 이유가 없다는 것을 실측으로 확인.** 강등 로직 유지가 맞다.

---

## 6. 성능 (결과 아님, 부작용 없음 확인용)

| arm | TTFT p50 | TTFT p95 | TPOT | throughput | errors |
|---|---|---|---|---|---|
| `hicache` | 0.2903 | 0.6784 | 0.0262 | 435.3 | 0 |
| `park_local` | **0.2359** | **0.5545** | 0.0265 | 430.6 | 0 |
| `park_pd` | 0.2415 | 0.5957 | 0.0266 | **438.4** | 0 |
| `park_pd_blind` | 0.2554 | 0.7409 | 0.0265 | 429.7 | 0 |
| `park_slowlink` | 0.2392 | 0.6814 | 0.0265 | 430.9 | 0 |

- 모든 park arm이 `hicache`보다 TTFT가 좋다 (−12% ~ −19%)
- TPOT은 0.0262~0.0266으로 **1.5% 이내** — 균형을 지연으로 산 게 아님
- **주의**: Layout B는 Mooncake P→D 전송 절반을 NVLink로 옮기므로 **Exp 1과 절대값 비교 금지**

---

## 7. `park_pd` 재측정 3회 — 배치는 고쳐졌으나 **회수는 `park_local`을 못 넘었다**

| | v1 (원본) | v2 (`full`>usage) | **v3 (`full`>`slow_link` + 샘플러 수정)** | `park_local` |
|---|---|---|---|---|
| park 총량 | 3.67* | 3.93* | **7.85** | 7.86 |
| fetch hits | 491 | 523 | **481** | **594** |
| fetched tokens | 428,809 | 428,840 | **404,394** | **502,185** |
| TTFT p50 | 0.2415 | 0.2338 | **0.2309** | 0.2359 |
| TTFT p95 | 0.5957 | 0.6562 | 0.6363 | **0.5545** |

> \* v1/v2의 park 총량은 샘플러 병합 버그로 **과소 보고**된 값이다 (§8). v3만 정확하다.

- **용량 문제는 해결됐다**: 7.85 GB로 `park_local`(7.86)과 동일. 6개 풀이 모두 사용됨
- **그런데 회수는 오히려 줄었다**: 481 hits / 404k tokens로 3회 중 최저
- **TTFT p50은 단조 개선**되어 `park_local`을 앞섰지만(0.2309 vs 0.2359), **p95는 뒤진다**(0.636 vs 0.554)
- 배치 자체는 계속 정확: usage gap **−0.370**, cross-GPU 96.6%, 널 모델 최선이 0.329

**왜 못 넘는가 — 예산 분할이 축출을 늘린다**

두 arm의 **총 park 용량은 60,000 토큰으로 동일**하고, shared index 덕에 양쪽 다 전체에 접근 가능하다. 차이는 **분할 방식**뿐이다:

| | 풀 구성 | 중앙값 park(1,698토큰) 기준 풀당 수용 |
|---|---|---|
| `park_local` | 30,000 × 2 | **17개** |
| `park_pd` | 10,000 × 6 | **5개** |

실행 중 총 파킹량은 1,450,337 토큰 = **용량의 24배**. 즉 풀은 계속 회전한다.
풀이 작을수록 **각 풀이 독립적으로 LRU 축출**하므로 유효 축출 지평이 짧아지고,
한 세션의 항목이 무관한 압력에 밀려난다. v1 로그의 `survival=100%→3%`이 그 증거다.
파킹 자체가 실패한 것은 아니다 (`dropped_gb=0`, 8,065토큰 최대 park도 10k 풀에 들어감).

**결론: 배치 메커니즘은 작동하지만, *총 예산 고정* 조건에서 decode GPU로 분산하는 것은
로컬 집중보다 회수가 나쁘다.**

**남은 선택지 (사용자 판단 필요)**

1. **공정성 기준을 바꾼다** — 논문의 전제가 "유휴 HBM은 공짜"라면, arm 간에 맞춰야 할 것은
   *총 park HBM*이 아니라 *풀당 크기*다. `park_pd`에 후보당 30,000 토큰(총 3배 HBM)을 주고
   재측정. **전제와는 일관되지만 §3의 공정성 논거를 다시 써야 한다**
2. **현 결과를 그대로 보고한다** — "배치는 옳게 동작하나(§2), 이 워크로드에서는 로컬 파킹으로
   충분하다"는 정직한 negative. Exp 2의 기여는 **(B)의 증명 + 분할의 대가**라는 발견이 된다

---

## 7-1. v4 `bigpool` — 후보당 30,000 토큰 (총 HBM 3배)

| | v3 (총량 동일) | **v4 bigpool (풀당 동일)** | `park_local` |
|---|---|---|---|
| park 총량 | 7.85 GB | **23.58 GB** | 7.86 GB |
| fetch hits | 481 | **596** | **594** |
| fetched tokens | 404,394 | **506,412** | 502,185 |
| **fetch miss** | 272 | **179** | **179** |
| TTFT p50 | 0.2309 | 0.2324 | 0.2359 |
| TTFT p95 | 0.6363 | **0.6997** | **0.5545** |
| throughput | 440.6 | 428.1 | 430.6 |

분할 페널티는 사라졌다. 그런데 **`park_local`을 넘지 못하고 정확히 따라잡기만 했다.**

**결정적 단서: `fetch_miss`가 179로 완전히 동일하다.** 두 arm이 같은 요청을 놓친다.
남은 miss는 용량 부족이 아니라 **구조적**이다 (첫 턴, 아직 park된 적 없는 prefix).
→ **이 워크로드의 재사용 가능한 working set은 이미 `park_local`의 30,000 토큰 로컬 풀에
들어간다.** 용량을 3배로 늘리든 GPU에 분산하든 **더 잡을 것이 남아 있지 않다.**

대가는 실재한다: HBM 3배, TTFT p95 0.554 → 0.700 (+26%), throughput −0.6%.

### Exp 2 최종 판정

| | |
|---|---|
| **(B) 배치가 헤드룸을 따라간다** | ✅ **증명됨** — usage gap −0.33~−0.40, cross-GPU 94~99%, 널 모델 전부 3배 이상 뒤짐 |
| **P→D 분산이 이득인가** | ❌ **이 구성에서는 아니다.** 총량 동일 시 열세(481 vs 594), HBM 3배 시 동률(596 vs 594) + 꼬리 악화 |

**왜인가 — 로컬 GPU가 부족하지 않았다.** 실측 유휴 HBM: gpu0 20.05 GB, gpu2 20.61 GB.
prefill GPU에 20 GB씩 남아 있으면 **굳이 남의 GPU로 건너갈 이유가 없다.**
P→D 차용은 **로컬 GPU가 제약일 때만** 값을 한다.

**이것이 Exp 2가 실제로 보여준 것이다**: 메커니즘은 정확하나 이 하드웨어·워크로드에서는
필요하지 않다. 필요해지는 조건(로컬 HBM 부족)을 만들어야 이득이 드러난다.

---

## 7-2. v5 — C=32 + **GPU당 할당 고정** (도달 범위만 다름)

`PARK_POOL_TOKENS_PER_GPU=10000`: 두 arm 모두 **건드리는 모든 GPU에 10,000 토큰씩**.
`park_local`은 자기 GPU 1개(=10k 도달), `park_pd`는 3개(=30k 도달). C=32, 288 세션, 12턴.

| | `park_local` | **`park_pd`** | Δ |
|---|---|---|---|
| park 용량 | 2.62 GB | **7.86 GB** | 3× 도달 (GPU당은 동일) |
| **fetch miss** | 763 | **629** | **−17.6%** |
| **fetch hits** | 680 | **791** | **+16.3%** |
| park-fetch hit rate | 47.1% | **55.7%** | **+8.6%p** |
| fetched tokens | 534,037 | **635,169** | **+18.9%** |
| TTFT p50 | 0.3400 | **0.3096** | **−8.9%** |
| **TTFT p95** | **2.4339** | 4.0619 | **+67%** ❌ |
| **TTFT p99** | **7.7242** | 11.3695 | **+47%** ❌ |
| throughput | **838.5** | 830.0 | −1.0% |
| errors | 0 | 0 | — |

**`fetch_miss`가 드디어 갈라졌다** — 사전에 정해둔 판정 기준이 충족됐다.
C=16에서 179 vs 179로 동일했던 것이 C=32에서 763 vs 629가 된다. 즉 **working set이
10,000 토큰 로컬 풀을 넘어섰고, 그 초과분을 `park_pd`가 실제로 잡아낸다.**
배치도 여전히 정확하다: usage gap **−0.382**, cross-GPU 97.7%, 대상의 97.7%가 decode GPU.

**교란 변수 없음 확인**: decode 점유는 0.13~0.16으로 여전히 낮아 decode 포화가 아니며,
양쪽 288/288 세션 완주, 에러 0.

### 그러나 꼬리 지연이 크게 나빠진다 — 숨기면 안 되는 대가

p95 **+67%**, p99 **+47%**. 파킹의 97.7%가 cross-GPU이므로 **거의 모든 fetch가 원격
읽기**가 되고, C=32의 경합 구간에서 그 큐가 쌓인다. 중앙값은 좋아지지만 **꼬리는 확실히
나빠진다.** 서빙 시스템에서 p95/p99가 p50보다 중요한 경우가 많으므로,
이 결과는 **"명확한 승리"가 아니라 "median–tail 트레이드오프"로 서술해야 한다.**

### Exp 2 최종 판정 (v1~v5 종합)

| 조건 | P→D 분산의 결과 |
|---|---|
| 총 예산 고정 (v1~v3) | ❌ 열세 — 작은 풀들이 독립 축출 |
| 총 예산 3배 (v4) | ➖ 동률 — `fetch_miss` 179로 동일, 더 잡을 게 없음 |
| **GPU당 고정 + C=32 (v5)** | ✅ **park-fetch hit rate +8.6%p, median TTFT −8.9%** / ❌ **p95 +67%** |

**operating regime: GPU당 park 용량이 binding constraint일 때만 값을 한다.**
로컬 GPU가 working set 전체를 담을 수 있으면(v4: prefill GPU에 20 GB 유휴) 이득이 없다.
그 조건은 모델이 크거나 serving 풀이 커서 **prefill GPU의 HBM이 이미 소진된 경우**다.

> ⚠ 리뷰어가 물을 것: "`park_local`에 왜 30,000을 안 주는가? prefill GPU에 20 GB 남는데."
> 정직한 답: 이 하드웨어에서는 줄 수 있고, 주면 v4처럼 동률이 된다. v5는 **GPU당 여유가
> 제한된 배치를 가정한 조건부 결과**이며, 그 가정을 논문에 명시해야 한다.

---

## 8. 정정 — 제가 틀렸던 두 가지

1. **"P0의 gpu1 풀 용량이 0, 파킹 97%가 버려짐"** → **틀렸다.** 서버 로그에 풀은
   `10000 tokens x 32 layers`로 정상 할당되었고 `g1:10000/10000`으로 꽉 차 있었다.
   `parked_*.csv`가 0으로 보고한 것은 **샘플러의 병합 버그**였다: 후보 목록이 겹치면
   (`P0=0,1,3` / `P1=2,3,1`) 같은 GPU에 두 prefill이 각각 풀을 가지는데
   `per_gpu[target] = bytes`가 **나중 파일로 덮어썼다**. 합산으로 수정.
   → v1/v2의 residency 수치는 전부 과소 보고이며 v3만 신뢰할 수 있다.
2. **M1/M2로 (A)를 측정하려 한 설계** → §3. serving 풀만 보는 지표로는 원리적으로 불가능했다.

## 9. Exp 2 그림 (`fig_exp2_stack`) — GPU별 KV 풀을 **자기 캐시 / 빌려준 캐시 / 빈 공간**으로 쌓기

GPU 4개 각각에 local-only vs Ours 막대 한 쌍. 세 층으로 쌓는다:

| 층 | 뜻 |
|---|---|
| **own cache** (회색) | 그 GPU가 자기 작업을 위해 들고 있는 캐시 |
| **parked** (초록) | 다른 노드가 여기에 올려둔 재사용 캐시 ← **메커니즘** |
| **unused** (점선 빈칸) | 아직 안 쓴 풀 용량 ← **낭비** |

**낭비와 그것을 채우는 것이 같은 막대, 같은 축척에 있다.** 비율이나 집계 막대로는
보여줄 수 없는 것이다:

- **prefill(GPU0/2)**: 회색이 천장까지 차 있고 빈칸이 거의 없다 — 더 담을 데가 없다
- **decode(GPU1/3)**: local-only는 3.0 GB만 쓰고 **19.7 GB가 점선 빈칸**.
  Ours에서 그 빈칸 안에 **2.6 GB 초록 블록이 생긴다**

막대 총높이 = serving 풀 + 그 GPU의 park 풀. 두 arm의 총높이가 거의 같다(22.64 vs 22.77 GB).
park 풀은 serving 풀이 가져갔을 HBM에서 잘라낸 것이므로 **없던 메모리를 만들어낸 게 아니라
재분할한 것**이며, 그림이 그 사실을 그대로 보여준다.

오른쪽: park-fetch hit rate 47.1 → 55.7%, TTFT p50 0.34 → 0.31, **p95 2.43 → 4.06 (빨강)**.

> 앞선 두 버전은 남겨둠. `plot_exp2.py`(결정 로그: usage gap·널 모델)는 **배치가 우연이
> 아님**의 증거로 부록에, `plot_exp2_harvest.py`(시계열 + 3.0× 캐시)는 대체안으로.

첫 버전(`plot_exp2.py`)은 *정책*을 그렸다: 선택된 GPU의 점유율 vs 거부된 GPU의 점유율.
그건 "선택기가 한가한 GPU를 선호하는가"에 답할 뿐이고 **selector 소스코드의 재진술에
가깝다.** 정작 이 실험의 주장인 **"유휴 GPU 용량을 얼마나 활용했는가"는 전혀 안 나온다.**
그래서 자원 기준으로 다시 그렸다. (`plot_exp2.py`는 부록용으로 남겨둠)

- **(a) 재사용 KV가 실제로 어디에 사는가** — 시간축, GPU별 stack.
  local-only는 decode GPU가 아무리 한가해도 **0에서 평평**하다(점선). Ours는 **decode
  GPU가 캐시의 대부분을 떠받친다.** 주장을 직접 그린 그림
- **(b) 빌려온 유휴 메모리** — 2.62 GB → **7.86 GB (3.0× 캐시)**, 그중 **5.24 GB(67%)가
  원래 놀던 decode GPU** 위에 있다. 점선 테두리는 **아직 안 쓴 유휴 용량**(34 GB)이며,
  이걸 그려야 "유휴 메모리를 다 썼다"로 오독되지 않는다
- **(c) 이득과 대가** — park-fetch hit rate 47.1 → 55.7%, TTFT p50 0.34 → 0.31,
  **p95 2.43 → 4.06 (빨간색)**

**(b)에서 인용할 숫자는 "유휴 메모리의 15%를 썼다"가 아니라 "캐시가 3배가 됐다"이다.**
전자는 우리가 설정한 per-GPU 상한(10,000토큰)이 정한 값이지 메커니즘의 한계가 아니다.
후자가 (c)의 hit rate 상승을 만든 원인이다.

## 10. 다음에 할 일

1. ~~`_select_pool()` 수정 후 재측정~~ → **완료** (v2/v3/v4, §7·§7-1). 두 정렬 버그 수정됨
2. **Exp 2 그림을 M3 기반으로 재설계** — M1/M2 막대는 버린다 (§3)
3. **P→D가 값을 하는 조건을 만든다** (§7-1): prefill GPU의 유휴 HBM을 없애야 한다.
   `PARK_MEM_FRACTION`을 올려 로컬 park 여유를 제거하거나, 컨텍스트/동시성을 키워
   working set이 로컬 풀을 넘게 한다. 그 구간이 이 메커니즘의 **operating regime**이고,
   지금 결과는 그 밖에서 측정한 것이다
4. `park_pd_blind`의 P0 파킹 부재 원인 규명 (§2 경고)

---

## Prefill-bound workload, hicache vs park_gpu (2026-08, `results/why/why_longctx_c16_p32000`)

**Result: park_gpu loses by ~30% on TTFT. Reproducible, effect far larger than noise.**

The sharegpt workload could never have answered this — TTFT was 0.5% of a request there
(49 s of decode against 0.25 s of TTFT), so a prefix cache had nothing to give back. The
longctx workload puts TTFT at 89% of a request, which is the regime the mechanism claims.
It is the right testbed, and the mechanism loses on it.

Interleaved arms (h1 p1 h2 p2 h3 p3), 512 turns each, 0 errors, matched output tokens.

| metric | r1 | r2 | r3 | median | |
|---|---|---|---|---|---|
| TTFT mean | +40.0% | +29.9% | +28.2% | **+29.9%** | all same sign |
| TTFT p50 | +45.5% | +65.4% | +49.3% | **+49.3%** | all same sign |
| TTFT p90 | +66.1% | +50.4% | +50.2% | **+50.4%** | all same sign |
| TTFT p95 | +47.6% | +38.7% | +38.9% | **+38.9%** | all same sign |
| TPOT | +15.6% | +11.3% | +4.5% | **+11.3%** | all same sign |
| throughput | −17.6% | −22.7% | −20.0% | **−20.0%** | all same sign |

(positive = park_gpu worse). hicache baseline spread is 7.2% on the mean and 3.8% on p95,
against a 29.9%/38.9% effect, so this is not run-to-run noise.

### A SINGLE SEQUENTIAL RUN SAID THE OPPOSITE, AND WAS WRONG

The first longctx run — one repeat, arms run back to back — reported park_gpu at 6.626 s
against hicache 7.848 s, i.e. **15.6% BETTER**. Against the repeat distributions:

    hicache   single 7.848    repeats 6.176 / 6.641 / 6.448
    park_gpu  single 6.626    repeats 8.646 / 8.624 / 8.266

Both arms landed outside their own repeat range, in opposite directions, in the same run.
park_gpu's single-run value is 20% below the minimum of its three repeats. That is not a
draw from the same distribution; the two arms ran under different machine conditions,
which is exactly what running them back to back cannot control for and what interleaving
does. **Do not quote the −15.6%.**

### Why it loses

| | r1 | r2 | r3 |
|---|---|---|---|
| fetch hit rate | 39.6% | 44.2% | 46.0% |
| tokens fetched back | 1.06 M | 1.18 M | 1.11 M |
| tokens parked | 4.38 M | 4.38 M | 4.38 M |
| **park : fetch ratio** | **4.1 : 1** | **3.7 : 1** | **4.0 : 1** |
| peer fetch cost | 80.1 ms | 81.4 ms | 79.2 ms |

Three compounding costs, all fixable in principle, none fixed here:

1. **It parks 4x what it ever reads back.** 4.38 M tokens at 128 KiB is ~574 GB copied per
   run to serve ~139 GB of hits. Three quarters of the park traffic is pure loss.
2. **A fetch costs ~80 ms on the scheduler main thread.** This is the ASYNC path, where the
   copy is only enqueued — an enqueue should be microseconds. 200+ fetches per run is
   ~16 s of scheduler stall, and the stall blocks every other request in the batch, not
   just the one being served. `maybe_fetch` also calls `match_prefix`, `alloc` and, when
   the pool is full (it is: 170 k working set against a 60 k pool), `tree_cache.evict` —
   all synchronous, all on that thread.
3. **Hit rate is only 40-46%,** so the majority of that parking never pays at all.

Cost 2 is the most suspicious: 80 ms for what is supposed to be a non-blocking enqueue
means the fetch path is not actually async. That is worth finding before any redesign,
because it is a bug-shaped number rather than a design limit.

### What this does not say

It does not say GPU-first placement is wrong. It says THIS implementation, at this hit
rate and this park:fetch ratio, costs more than it saves on a workload built to favour it.
The park_host control was not run here, so the medium question is still open on longctx.

### Where the cost actually is (phase instrumentation, 2026-08)

Chasing the fetch was chasing the wrong path. Full accounting of scheduler-main-thread
time on the longctx workload, 513 parks and 206 fetches:

| path | total | share of the 42.9 s |
|---|---|---|
| **park (write)** | **30.1 s** | **70%** |
| fetch (read) | 12.8 s | 30% |

and inside the park path:

| phase | total | share |
|---|---|---|
| **sync** | **26.9 s** | **89.5%** |
| select | 1.1 s | 3.6% |
| xfer | 0.9 s | 3.1% |
| write | 0.5 s | 1.6% |
| gather | 0.5 s | 1.5% |
| index | 0.2 s | 0.8% |

gather/xfer/write are asynchronous enqueues and return in 1.9 s combined; `sync` is where
the 535 GB is actually executed and waited for. So the achieved rate is 535 GB / 26.9 s =
**19.9 GB/s** — 76% of the PCIe link and 38% of NVLink, i.e. the copy is not especially
inefficient. The problems are that it is BLOCKING and that it is 5.7x larger than it needs
to be.

Scheduler-thread occupancy accounts for 42.9 s of the 70 s wall-clock gap (61%); the
remainder is plausibly GPU bandwidth contention with the forward pass, which parking also
consumes and which is not on this thread.

**The link was never the problem.** `p2p_matrix.py` confirms peer access on every pair,
NVLink pairs at 52.6-52.7 GB/s and PCIe pairs at 26.3 GB/s with no host staging anywhere.
The fetch path's 80 ms/fetch, which prompted this instrumentation, accounts for 1% of the
per-turn TTFT gap (21 ms/turn against 2.17 s).

### Two fixes, in order

1. **Stop blocking on the park copy.** `_gather_copy_peer_to_park` ends in
   `torch.cuda.synchronize(pool.gpu)` on the scheduler main thread, so all 26.9 s of
   transfer is serialised against request handling, while the FETCH path on the same
   module is already asynchronous. The synchronize is not gratuitous — the source is the
   decode node's IPC-mapped KV pool and the slots must not be recycled before the copy
   lands — so it cannot simply be deleted. It needs a CUDA event recorded at park time and
   checked on a later scheduler pass, which overlaps the copy with the forward pass
   instead of stalling on it.
2. **Park less.** 535 GB written to serve 94 GB of hits. Even at the current rate,
   parking only what is later fetched would cost 4.7 s instead of 26.9 s. This is the
   policy question (what deserves parking), and it is worth strictly less than fix 1 until
   fix 1 lands, because right now every avoided byte only saves blocking time.

### Async park + pinned index upload: the fixes work, the TTFT does not follow

Three interleaved repeats, all nine arms on one build (92185c97a), 512 turns each,
0 errors. `results/why/async_r3`.

**The engineering worked, unambiguously.** Park-path scheduler time, stable across repeats:

| | r1 | r2 | r3 |
|---|---|---|---|
| park_sync (blocking) | 27.3 s | 28.8 s | 29.2 s |
| park_gpu (async + pinned index) | **3.4 s** | **3.1 s** | **2.8 s** |

`sync` (24-26 s) and `index` (4.4 s) are both gone; the top phases are now `select` and
`xfer` at ~1 s each. Total scheduler occupancy roughly halved, 40-44 s to 17-21 s.

**TTFT does not follow, and the two are not even correlated:**

| rep | scheduler saved | wall saved |
|---|---|---|
| 1 | 19.4 s | 52 s |
| 2 | 23.6 s | 26 s |
| 3 | 27.2 s | **−4 s** |

park_sync -> park_gpu on TTFT mean: −17.4%, −7.2%, +2.9% — **sign flips**, so the change
cannot be claimed as a TTFT win despite a 90% cut in the cost it targeted. Only p90 and
p95 hold their sign, at −3.2% and −4.1%. And park_gpu degrades monotonically across
repeats (6.505 / 7.362 / 8.252) while its own scheduler cost FALLS (3.4 / 3.1 / 2.8 s), so
whatever is degrading is not the park machinery.

**Against hicache the shape is the finding:**

| | median of 3 | sign |
|---|---|---|
| p50 | **−25.0%** (better) | flips |
| p90 | **+45.1%** (worse) | all same |
| p95 | **+24.3%** (worse) | all same |
| throughput | −15.4% | all same |

Parking helps the typical request and badly hurts the tail. That is the signature of
CONTENTION rather than of a cache that does not work: the median request gets its hit,
while some requests wait behind park traffic.

### Why this points at volume, not at the copy path

Parking moves 535 GB per run to serve 94 GB of hits, and both directions cross the DECODE
GPUs -- the park writes into their HBM, the fetches read out of it -- while those same GPUs
are running forward passes. Blocking parking rate-limited that traffic as a side effect of
stalling the scheduler. Removing the stall let it run free, which is why halving scheduler
occupancy did not halve anything the user sees, and plausibly why the tail got worse.

So the remaining work is NOT more transfer-path optimisation. It is the 5.7:1 ratio: at
the current rate, parking only what is later fetched would cut that traffic 5.7x. That is
a policy question -- what deserves parking -- and it is now the only lever left that is
large enough to matter.
