# Evaluation 아웃라인 — ShareGPT multi-turn

> 근거 데이터: `results/models/{llama8b,llama13b,qwen14b}/table.json`
> 그림: `results/models/fig_models.pdf` (3패널) / `.pptx` (편집 가능)

---

## 1. 실험 설정

- **워크로드**: ShareGPT multi-turn 대화. 턴 사이에 tool-call think-time 3초 삽입
  - Llama-3.1-8B / Qwen3-14B: `MAX_TOKENS=1024`, 최대 10턴, 1440턴 측정
  - Llama-2-13B: `MAX_TOKENS=512`, 최대 4턴, 800턴 측정 — **position limit 4096 때문**
- **시스템**: SGLang 2P2D disaggregation, RTX A6000 ×4 (49GB), Mooncake transfer backend
- **부하**: closed-loop, concurrency 8
- **비교 대상 3종**
  - **Recompute** — `--disable-radix-cache`. 매 턴 전체 컨텍스트를 재계산하는 하한선
  - **SGLang HiCache** — 기존 계층 캐시. GPU에서 축출된 KV를 host DRAM / file backend로 offload
  - **Ours** — GPU-first unified placement. 유휴 peer GPU 풀에 KV를 park
- **측정 신뢰성**: 3개 모델 전 arm에서 infra 실패율 0.0

---

## 2. 성능

### 2-1. Cache Hit Rate — SGLang 대비 **1.38 ~ 1.72배**

| 모델 | Recompute | SGLang | Ours | **SGLang 대비** |
|---|---|---|---|---|
| Llama-3.1-8B | 0% | 55.7% | **95.7%** | **1.72×** |
| Llama-2-13B | 0% | 35.2% | **58.3%** | **1.66×** |
| Qwen3-14B | 0% | 56.5% | **78.1%** | **1.38×** |

- Recompute의 0%는 결측이 아니라 **정의상 0** (캐시가 없음)
- 세 모델 모두 일관되게 개선 — 특정 모델의 우연이 아님

**왜 오르는가 (핵심 메커니즘)**

- PD disaggregation에서 **생성 토큰의 KV는 decode 노드가 계산**하는데, SGLang의 radix/HiCache는 **prefill 노드**에 있음
- 따라서 prefill 노드는 그 KV를 본 적이 없고, 다음 턴에 이전 assistant 응답이 프롬프트에 포함되면 **전부 다시 prefill**해야 함
- 실측: 턴당 프롬프트 증가분 482토큰 중 **446토큰(93%)이 이전 assistant 응답**
- Ours는 decode 노드에서 `prompt_ids + output_ids`를 함께 park하므로 이 부분을 회수함

**구조만으로 hit rate를 예측하면 실측과 1%p 이내로 일치** (median context 1101, 증가분 482, 응답 446):

| | 예측식 | 예측 | 실측 |
|---|---|---|---|
| SGLang | (1101−482)/1101 | 56.2% | 55.7% |
| Ours | (1101−(482−446))/1101 | 96.7% | 95.7% |

> 반증 실험 가능: `SGLANG_KV_PARK_GEN=0` (prefix만 park) A/B로 생성-KV 기여분을 분리 측정 (~30분)

### 2-2. Normalized TTFT (Recompute = 1.0, 낮을수록 좋음)

| 모델 | Recompute | SGLang | Ours | SGLang 대비 |
|---|---|---|---|---|
| Llama-3.1-8B | 1.000 | 0.882 | **0.671** | **1.31×** |
| Llama-2-13B | 1.000 | 1.035 | 1.056 | 0.98× |
| Qwen3-14B | 1.000 | 0.831 | **0.706** | **1.18×** |

- 절대 TTFT는 **메커니즘보다 모델 크기가 좌우**하므로 정규화 — 그래야 모델 간 높이 비교라는 무의미한 읽기를 막음
- P95도 같은 방향: Llama-3.1-8B 0.955 / 0.765 / **0.620**, Qwen3-14B 1.589 / 1.195 / **0.913**

**Llama-2-13B가 예외인 이유 (반드시 명시)**

- position limit 4096 → 4턴, 컨텍스트 ~3k 토큰으로 제한
- **재사용으로 아낄 수 있는 상금 자체가 작아** fetch 비용을 넘지 못함
- 이 모델에서는 SGLang(1.035)도 Recompute보다 느림 — 즉 Ours만의 문제가 아니라 **짧은 컨텍스트 구간의 공통 특성**
- 메커니즘은 정상 작동함(hit rate 1.66×) — 이길 판이 작을 뿐

---

## 3. 메모리

### 3-1. Host DRAM 사용량 — SGLang 대비 **1.45 ~ 2.05배 절감**

| 모델 | Recompute | SGLang | Ours | 절감량 | **SGLang 대비** |
|---|---|---|---|---|---|
| Llama-3.1-8B | 16.6 GB | 34.2 GB | **16.6 GB** | −17.5 GB | **2.05×** |
| Llama-2-13B | 14.8 GB | 29.8 GB | **15.2 GB** | −14.6 GB | **1.96×** |
| Qwen3-14B | 16.5 GB | 23.4 GB | **16.2 GB** | −7.2 GB | **1.45×** |

- Ours는 host DRAM 사용량이 **Recompute 수준(캐시 없는 하한선)으로 복귀** — 캐시를 유지하면서 host 비용은 0에 수렴
- 비교 기준을 SGLang으로 잡은 이유: Recompute는 **host tier 자체가 없어** 정의상 이기므로 기준이 될 수 없음

**측정 방법 (본문 한 줄, 자세한 건 각주로)**

- `/proc/meminfo`의 anonymous 메모리를 1초 간격으로 샘플링한 peak 값
- 파일 캐시를 제외한 이유: 파일 캐시는 메모리가 부족하면 커널이 **버릴 수 있어서** 다른 프로세스를 실제로 막지 않음
- **어떤 지표를 써도 결론이 같음** — 절감량 (GB):

  | 모델 | anonymous | 프로세스 RSS | MemAvailable 감소분 |
  |---|---|---|---|
  | Llama-3.1-8B | 17.5 | 16.4 | 17.3 |
  | Llama-2-13B | 14.6 | 13.7 | 14.8 |
  | Qwen3-14B | 7.2 | 6.2 | 7.1 |

  → 지표 선택이 결과를 만든 것이 아님. 리뷰어의 "왜 하필 그 지표냐"에 대한 답이 이 표
- 절대값으로 anonymous를 고른 이유는 단순함: RSS는 모델 가중치 등 **비교와 무관한 ~65 GB 상수**를 포함해 비율을 1.24×로 희석시킴

### 3-2. HiCache의 host 사용량은 write policy와 무관한 **정적 예약**

- `write_through` / `write_through_selective` / `write_back` / L2-only 4개 구성에서 host DRAM 사용량 편차 **0.4 GB 이내**
- 즉 "실제로 얼마나 쓰느냐"가 아니라 **서버 기동 시 예약하고 반납하지 않는 양** → 워크로드를 조정해도 줄지 않음

### 3-3. Trade-off (숨기지 말 것)

- host DRAM 절감의 대가는 **GPU HBM 증가**: Llama-3.1-8B +8.9 GB, Qwen3-14B +14.0 GB, Llama-2-13B +18.0 GB
- 논지: **이미 놀고 있던 HBM**으로 옮긴 것 (prefill KV pool 55% 유휴 측정) → 새 자원을 요구하지 않음
- 정직한 한 줄: *"host DRAM을 GPU HBM으로 치환한다. 후자가 유휴일 때 이득이다."*

---

## 4. 한계 (Limitations)

- **부하가 높아지면 역전됨** — open-loop sweep (Llama-3.1-8B) median TTFT:
  | offered rate | SGLang | SGLang(동일 메모리 예산) | Ours |
  |---|---|---|---|
  | 0.05 sess/s | 0.341 | — | **0.214** |
  | 0.35 sess/s | 0.324 | 0.361 | 0.450 |
  | 0.75 sess/s | 0.922 | 1.279 | 2.735 |
  - 동일 메모리 예산으로 통제해도 격차의 **70~80%가 남음** → 메모리 예산이 아니라 **메커니즘 자체**
  - 해석: parking 작업이 prefill과 같은 자원을 두고 경쟁. 저부하에선 유휴 용량이 흡수하지만 고부하에선 순수 오버헤드
  - 적용 범위: **동시 세션 ~50 미만** 구간의 기법

- **Throughput은 개선되지 않음** — 서버가 **decode에서 포화**하는데 KV 배치는 decode를 건드리지 않음
  - peak: Recompute 1105 / SGLang 1249 / Ours 1135 tok/s
  - 그래서 그림에서 throughput 패널을 뺌

- **짧은 컨텍스트 워크로드에서는 이득 없음** — BFCL v3 기준 median context 415토큰, 턴당 증가분 52토큰
  - SGLang의 hit rate가 이미 87%라 여지가 12%p뿐 (Ours 최대 94%, **1.08×**)
  - full re-prefill 상금이 **67ms**인데 park 오버헤드는 **+562ms** → TTFT는 오히려 악화 (실측 2.106s vs Recompute 1.688s)
  - 턴의 75%가 tool call인데 템플릿 재직렬화로 생성-KV 경계가 자주 miss

---

## 5. 한 문장 요약

> GPU-first KV placement는 PD disaggregation에서 **decode가 생성한 KV를 prefill 측이 재사용 가능하게** 만들어, cache hit rate를 SGLang 대비 **1.38~1.72배**로 올리고 host DRAM을 **최대 2.05배** 줄인다. 대가는 유휴 GPU HBM이며, 이득은 **중저 동시성 구간**에 한정된다.
