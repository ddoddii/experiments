# LLM Serving Disaggregation Experiments

> **목적**: Prefill-Decode(PD) disaggregation이 agentic multi-turn 워크로드에서  
> TTFT / TPOT / Throughput에 미치는 영향을 여러 serving 프레임워크/설정으로 비교.

---

## 실험 환경

| 항목 | 내용 |
|------|------|
| **서버** | server17 |
| **GPU** | NVIDIA RTX A6000 × 4 (49 GB VRAM each) |
| **RAM** | 125 GB |
| **모델** | `Llama-3.1-8B-Instruct` (`/home/uhmturks/hf_models/`) |
| **conda 환경** | `vllm-source` (vLLM TP=4) / `vllm-ppd` (vllm-ppd) / `sglang` (SGLang) |

---

## 벤치마크 데이터셋

**BFCL v3 Multi-Turn Base** (Berkeley Function Calling Leaderboard)

| 항목 | 내용 |
|------|------|
| 파일 | `data/BFCL_v3_multi_turn_base.json` |
| 아이템 수 | 200개 |
| 구조 | 아이템당 1~7 turn (평균 3.7 turn), 각 turn마다 tool call 포함 |
| Tool 종류 | GorillaFileSystem, TicketAPI, MessageAPI, MathAPI, TradingBot, TwitterAPI, TravelAPI, VehicleControlAPI |
| 특징 | **Prefill-heavy agentic workload**: 매 turn마다 system prompt + tool 정의 + 누적 대화 이력이 context로 전달, turn이 늘어날수록 context 길어짐 |

Context 증가 패턴 (vLLM TP=4 측정):

| Turn | 평균 context 길이 |
|------|-----------------|
| 0 | ~1,553 chars |
| 1 | ~1,727 chars (+11%) |
| 2 | ~1,885 chars (+9%) |
| 3 | ~2,178 chars (+16%) |
| 4 | ~2,369 chars (+9%) |
| 5 | ~2,680 chars (+13%) |

→ 매 turn마다 이전 응답이 history에 누적되어 prefill 부하가 단조 증가함.

---

## 측정 메트릭

| 메트릭 | 정의 | 측정 위치 |
|--------|------|----------|
| **TTFT** | Time To First Token = `t_first_token − t_request` | 클라이언트 |
| **TPOT** | Time Per Output Token = `(t_last − t_first) / (n_tokens − 1)` | 클라이언트 |
| **Per-req throughput** | `(n_tokens − 1) / decode_time` (tok/s) | 클라이언트 (C=1만 의미 있음) |
| **Overall throughput** | `total_output_tokens / total_wall_time` (tok/s) | 실험 전체 |
| **KV cache usage %** | GPU KV 블록 풀 대비 현재 점유율 | Prometheus |
| **Context chars** | 매 turn 전송되는 전체 conversation 길이 | 클라이언트 |

---

## 실험 설정 (Configurations)

### Config 1: vLLM 4-GPU TP=4 — 기준선 (Baseline)

| 항목 | 내용 |
|------|------|
| **구성** | 단일 인스턴스, Tensor Parallelism 4 |
| **GPU** | GPU 0~3 전체 사용 |
| **엔드포인트** | `http://localhost:8000` |
| **시작** | `vllm serve ... --tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser llama3_json` |
| **벤치마크** | `benchmark/vllm_4gpu_BFCL_v3_multi_turn_base.py` |

**특징**: P/D 분리 없이 4-GPU를 하나의 큰 GPU처럼 사용. Prefill과 decode가 동일 GPU 풀을 공유해 KV cache가 항상 로컬에 있음 → 가장 낮은 TTFT. vLLM RadixAttention(prefix caching)이 활성화되어 turn 1부터 system prompt + tool 정의 KV를 GPU DRAM에서 재사용.

---

### Config 2: SGLang 2P2D — hicache 3-tier KV offload

| 항목 | 내용 |
|------|------|
| **구성** | Prefill 2 + Decode 2, KV 전송: Mooncake |
| **GPU** | P1=GPU0(30000), P2=GPU1(30001), D1=GPU2(30002), D2=GPU3(30003) |
| **Router** | port 8000 (SGLang router) |
| **KV 전송** | Mooncake metadata server (port 8080) |
| **hicache** | `--enable-hierarchical-cache --hicache-ratio 1.2 --hicache-storage-backend file` |
| **시작** | `bash scripts/sglang/start_2P_2D.sh` (또는 `HICACHE_RATIO=0.5 bash ...`) |
| **벤치마크** | `benchmark/sglang_BFCL_v3_multi_turn_base.py` / `sglang_BFCL_v3_multi_turn_concurrent.py` |

**SGLang hicache 작동 방식 (핵심)**

SGLang hicache는 KV cache를 3계층으로 관리한다:

```
┌─────────────────────────────────────────────────────┐
│                  SGLang hicache 구조                 │
│                                                     │
│  ┌─────────────┐    evict    ┌─────────────────┐   │
│  │ L1: GPU VRAM│ ──────────▶ │ L2: CPU DRAM    │   │
│  │  ~25.5 GB   │ (write-thru)│  ~30.6 GB pool  │   │
│  │  (동적 캐시) │             │  (ratio=1.2)    │   │
│  └─────────────┘             └────────┬────────┘   │
│                                       │ evict       │
│                              ┌────────▼────────┐   │
│                              │ L3: SSD         │   │
│                              │ /tmp/hicache/   │   │
│                              │ (content-addr.  │   │
│                              │  hash 파일)     │   │
│                              └─────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**`hicache_write_policy = 'write_through'` (중요)**

1. Prefill 완료 시 KV block이 GPU(L1)에 생성됨
2. **즉시** SSD(L3)에도 동시에 기록됨 (write-through)
3. GPU의 KV block은 해당 요청 처리 후 free됨
4. → **steady-state에서 GPU KV cache usage ≈ 0%** — 정상 동작

**Content-addressable SSD caching**

SSD에 저장되는 파일명은 token sequence prefix의 해시 값이다:
- `system prompt + tool 정의` → 고정 prefix → 동일 해시 → SSD 캐시 히트
- P1과 P2가 `/tmp/hicache/`를 **공유**하므로 cross-node 캐시 재사용 가능

**D 노드 hicache 제약 (SGLang 0.5.9)**

`--disaggregation-mode decode` 사용 시 내부적으로 chunk cache를 강제 설정하며,
이것이 `--enable-hierarchical-cache`와 상호 배타적. → **D 노드에 hicache 불가**.
현재 구현에서는 P 노드만 hicache를 적용할 수 있음.

---

### Config 3: vllm-ppd 2P2D — P2pNcclConnector

| 항목 | 내용 |
|------|------|
| **구성** | Prefill 2 + Decode 2, KV 전송: P2pNcclConnector (NCCL P2P) |
| **GPU** | P1=GPU0(8100), P2=GPU1(8101), D1=GPU2(8200), D2=GPU3(8201) |
| **Proxy** | `ppd/comprehensive_proxy.py` — HTTP port 10001, ZMQ port 30001 |
| **KV buffer** | P: 1 GB send / D: 10 GB receive (per node, 기본값) |
| **시작** | `bash scripts/vllm-ppd/start_2P_2D.sh` (또는 `KV_BUFFER_GB=2 bash ...`) |
| **벤치마크** | `benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py` |

**2P2D vs 2P2pD**:
- 2P2D: Decode 노드가 순수 decode만 담당
- 2P2pD: Decode 노드가 PPD(Partial Prefill-Decode) 모드 — short prefill을 decode 노드가 처리해 P 노드 부하 분산. TTFT 약 27% 개선 (0.590s → 0.429s)

---

## 실험 결과 요약

### 메인 비교 테이블

| Config | C | TTFT avg | TPOT avg | Per-req Tput† | Overall Tput | 성공 | Wall time |
|--------|---|---------|---------|--------------|-------------|------|----------|
| vLLM TP=4 | 1 | **0.130 s** | **0.022 s** | **50.5 tok/s** | 34.1 tok/s | 200/200 | 280 s |
| **vLLM TP=4** | **8** | 0.874 s | 0.038 s | — | **132.9 tok/s** | **200/200** | 198 s |
| SGLang 1P1D | 1 | 0.395 s | 0.023 s | 44.0 tok/s | 29.3 tok/s | 38/200 ⚠️ | 151 s |
| SGLang 2P2D (hicache) | 1 | 0.814 s | 0.023 s | 43.9 tok/s | 22.6 tok/s | 200/200 | 1,252 s |
| SGLang 2P2D | 4 | 1.500 s | 0.020 s | — | 47.8 tok/s | 153/200 ❌ | 306 s |
| SGLang 2P2D | 8 | 2.511 s | 0.020 s | — | 42.5 tok/s | 83/200 ❌ | 198 s |
| SGLang 2P2D | 12 | 3.445 s | 0.019 s | — | 37.1 tok/s | 67/200 ❌ | 181 s |
| vllm-ppd 2P2D | 1 | 0.590 s | 0.065 s | 16.7 tok/s | 11.0 tok/s | 200/200 | 732 s |
| vllm-ppd 2P2pD | 1 | 0.429 s | 0.063 s | 17.2 tok/s | 12.0 tok/s | 200/200 | 764 s |
| vllm-ppd 2P2D | 8 | 0.872 s | 0.072 s | — | 63.7 tok/s | 200/200 | 148 s |
| vllm-ppd 2P2D | 12 | 1.305 s | 0.076 s | — | 22.1 tok/s | 132/200 ❌ | 197 s |

> ⚠️ SGLang 1P1D: func_doc 타입 변환 미적용으로 162개 에러, 수치는 참고용.  
> ❌ SGLang C≥4: Mooncake KV 전송 타임아웃/실패로 에러 증가 (C=4: 23.5%, C=8: 58.5%, C=12: 66.5%).  
> ❌ vllm-ppd C=12: P 노드 1GB KV send buffer 포화로 68개 에러.  
> † Per-req Tput은 C=1 순차 실행에서만 의미 있음 (동시 실행 시 — 표시).  
> ※ vLLM TP=4 C=1과 C=8의 총 출력 토큰이 9,557 vs 26,351로 크게 다름 — 아래 주의사항 참조.

---

### 핵심 인사이트

**1. Decode 속도: SGLang이 vllm-ppd 대비 3–4× 빠름**

| | SGLang (C=1) | vllm-ppd C=1 | vllm-ppd C=8 |
|--|-------------|-------------|-------------|
| **TPOT** | **0.023 s/tok** | 0.065 s/tok | 0.072 s/tok |
| **Per-req Tput** | **43.9 tok/s** | 16.7 tok/s | — |

SGLang은 FlashInfer decode 커널과 continuous batching 최적화 덕분에 decode 속도가 일관되게 빠름. 반면 vllm-ppd는 D 노드가 KV 수신 후 attention 연산 시 추가 오버헤드 발생.

**2. 동시성 확장: vllm-ppd가 SGLang보다 안정적**

SGLang은 C=4부터 에러가 급증 (Mooncake KV 전송 실패/타임아웃):
```
SGLang C=1: 200/200 (100%)  vllm-ppd C=1: 200/200 (100%)
SGLang C=4: 153/200 (77%)   vllm-ppd C=8: 200/200 (100%)  ← 안정
SGLang C=8:  83/200 (42%)   vllm-ppd C=12: 132/200 (66%)
SGLang C=12: 67/200 (34%)
```

SGLang의 에러율이 높아질수록 **성공한 아이템이 easier 항목으로 편향**됨.  
따라서 SGLang C≥4의 TTFT/TPOT 개선은 일부 survivor bias 포함.

**3. Overall throughput 피크: vLLM TP=4 C=8이 압도적 1위**

| Config | Overall throughput | 성공률 |
|--------|-------------------|--------|
| **vLLM TP=4 C=8** | **132.9 tok/s** | 100% |
| vllm-ppd C=8 | 63.7 tok/s | 100% |
| SGLang C=4 | 47.8 tok/s | 77% |
| SGLang C=8 | 42.5 tok/s | 42% |
| vLLM TP=4 C=1 | 34.1 tok/s | 100% |

vLLM TP=4 C=8이 **132.9 tok/s**로 vllm-ppd C=8(63.7)의 **2.1×**, vLLM TP=4 C=1(34.1)의 **3.9×**. 4-GPU를 single instance로 공유하므로 KV transfer 병목 없이 C=8을 100% 성공으로 처리. PD disaggregation이 throughput에서도 TP=4에 뒤처지는 결과.

**4. SGLang TPOT의 concurrency 개선 (batching 효과)**

SGLang C=1→C=12로 갈수록 TPOT이 0.023s → 0.019s로 **17% 개선**됨. Decode 노드가 여러 요청을 배치 처리하면서 GPU 활용률이 올라가는 것. 에러 없이 동시성을 높일 수 있다면 SGLang decode throughput은 더 좋아질 여지가 있음.

**5. vllm-ppd C=8→C=12 절벽**

C=8: 200/200, 63.7 tok/s → C=12: 132/200, 22.1 tok/s. 에러율 34%로 급증하며 overall throughput도 **65% 폭락**. P 노드 1 GB KV send buffer가 C=12에서 포화 → buffer expansion 실험(Task D) 필요.

---

### Per-Turn TTFT 패턴 (Prefix Cache 효과)

Turn 0은 첫 prefill (캐시 없음), turn 1+는 system prompt + tool 정의 KV가 캐시에 남아 있을 때.

| Turn | vLLM TP=4 C=1 | vLLM TP=4 C=8 | vllm-ppd C=1 | vllm-ppd C=8 | vllm-ppd C=12 | SGLang C=4 | SGLang C=8 | SGLang C=12 |
|------|--------------|--------------|-------------|-------------|--------------|-----------|-----------|------------|
| 0 | 0.145 s | 1.577 s | 0.967 s | 1.712 s | 2.003 s | 2.840 s | 1.366 s | 1.413 s |
| 1 | 0.078 s (−46%) | **0.560 s (−64%)** | 0.321 s (−67%) | 0.669 s (−61%) | 1.400 s (−30%) | 1.023 s (−64%) | 0.802 s (−41%) | 7.652 s (+441%) ⚠️ |
| 2 | 0.076 s | **0.550 s** | 0.310 s | 0.658 s | — | — | — | — |
| 3 | 0.076 s | **0.592 s** | 0.313 s | 0.565 s | — | — | — | — |
| 4 | 0.093 s | **0.567 s** | — | — | — | — | — | — |

> ⚠️ SGLang C=12 turn 1 TTFT 7.652s: turn 0(1.413s)보다 **5× 높음**. 정상적인 prefix cache 히트라면 낮아져야 하는데 오히려 급등. 추정 원인: C=12 부하에서 `/tmp/hicache/` SSD I/O 경합으로 캐시 읽기 지연 발생.

**관찰**:
- 모든 config에서 **turn 0 → turn 1에 TTFT 감소** (SGLang C=12 제외) — prefix cache hit 확인
- **vLLM TP=4 C=8**: turn 0이 1.577s로 C=1(0.145s)보다 10.9× 높아짐 (동시 8개 큐 대기). 하지만 turn 1부터 0.550~0.592s로 안정 → **prefix cache가 동시성 환경에서도 안정적으로 동작**
- vLLM TP=4 C=8의 turn 1+ TTFT(0.56s)는 vllm-ppd C=8(0.67s)보다 낮음 — KV 전송 오버헤드 없이 로컬에서 처리하기 때문
- vLLM TP=4 C=1 → C=8 TTFT 변화: turn0 0.145s → 1.577s (10.9×), turn1+ 0.078s → 0.56s (7.2×)
- SGLang hicache의 SSD 기반 캐싱은 단일 요청에서는 효과적이나, **고동시성에서 SSD I/O 병목** 발생 가능

---

### KV Cache 사용 패턴 (vllm-ppd C=8 측정)

| Node | 역할 | Max KV | Mean KV |
|------|------|--------|---------|
| P1 (GPU0) | prefill | 3.1% | 0.75% |
| P2 (GPU1) | prefill | 2.3% | 0.58% |
| D1 (GPU2) | decode | **17.2%** | 5.9% |
| D2 (GPU3) | decode | **15.5%** | 5.4% |

- **P 노드**: KV를 즉시 D 노드로 전송하므로 거의 비어 있음 (turn-around buffer 역할)
- **D 노드**: 여러 요청의 KV가 누적되어 P 대비 ~7–8× 높은 점유율
- vllm-ppd C=12에서 D1 최대 26.3%까지 상승 — OOM 직전 고수위

---

### 출력 토큰 수 비교 (주의 사항)

| Config | Benchmark 스크립트 | Total Tokens | 성공 | 아이템당 평균 |
|--------|-----------------|-------------|------|------------|
| vLLM TP=4 C=1 | `*_base.py` | 9,557 | 200 | 47.8 tok |
| **vLLM TP=4 C=8** | `*_concurrent.py` | **26,351** | 200 | **131.8 tok** |
| vllm-ppd C=1 | `*_base.py` | 8,075 | 200 | 40.4 tok |
| vllm-ppd C=8 | `*_concurrent.py` | 9,424 | 200 | 47.1 tok |
| vllm-ppd C=12 | `*_concurrent.py` | 4,352 | 132 | 33.0 tok |
| SGLang C=1 | `*_base.py` | 28,323 | 200 | 141.6 tok |
| SGLang C=4 | `*_concurrent.py` | 14,633 | 153 | 95.6 tok |
| SGLang C=8 | `*_concurrent.py` | 8,417 | 83 | 101.4 tok |
| SGLang C=12 | `*_concurrent.py` | 6,713 | 67 | 100.2 tok |

**⚠️ vLLM TP=4 C=1 vs C=8 토큰 수 불일치 (9,557 vs 26,351, 2.75×)**

같은 모델·데이터셋인데 C=8에서 토큰이 훨씬 많다. 가능한 원인:
1. `*_base.py`와 `*_concurrent.py`의 토큰 카운팅 방식 차이 — base 스크립트가 `stream_options: include_usage` 미적용 시 실제보다 적게 카운팅할 수 있음
2. 동시성 환경에서 모델이 다른 응답 형식을 생성할 가능성 (미확인)

vllm-ppd는 C=1(40 tok) vs C=8(47 tok)으로 큰 차이 없음. SGLang C=1(142 tok)도 concurrent와 격차 있으나 이는 에러 아이템의 survivor bias로 설명 가능. **vLLM TP=4 C=1 결과는 재실행이 필요할 수 있음.**

**프레임워크 간 직접 throughput 비교 시 토큰 수 차이를 반드시 고려해야 함.** 특히 vLLM TP=4 C=8의 132.9 tok/s는 vllm-ppd(47 tok)보다 ~2.8× 많은 토큰을 출력한 결과임을 감안할 것.

---

## 추가 실험 제안 / 연구 주제

### 진행 중 (구현 완료)

**Task D: vllm-ppd KV buffer sweep**

P 노드 `kv_buffer_size` 1→2→4 GB 확장으로 C=8/12/16 성공 여부 확인.

```bash
cd ~/experiments
bash scripts/vllm-ppd/run_buffer_sweep.sh
```

예상: 버퍼 4GB에서 C=12 OOM 해소, C=16까지 안정적 처리 가능 여부 확인.

---

### 우선순위 높음

**E. SGLang no-hicache baseline**

현재 hicache ON(P-only)과 OFF의 TTFT/에러율 차이가 불명확함.

```bash
# 현재 start_2P_2D.sh에서 --enable-hierarchical-cache 제거한 버전 필요
# → hicache 오버헤드 vs 이득을 고동시성에서 분리
CONFIG=sglang_2p2d_nohicache CONCURRENCY=1 python ...
CONFIG=sglang_2p2d_nohicache CONCURRENCY=4 python ...
CONFIG=sglang_2p2d_nohicache CONCURRENCY=8 python ...
```

가설: hicache SSD I/O가 C=8+ 에러의 주 원인이라면 hicache OFF 시 에러율 대폭 감소.

**F. SGLang concurrent 에러 원인 분석**

C=4에서 이미 23.5% 에러. 에러 유형 분류:
- Mooncake KV transfer timeout
- SGLang router timeout
- P→D KV 전송 실패 (KVTransferError)
- SSD write 실패

로그 분석으로 병목 정확히 규명 → 파라미터 튜닝 방향 결정.

**G. vLLM TP=4 concurrent benchmark**

현재 TP=4는 C=1만 측정됨. C=4/8/12 동시성에서 throughput 피크 확인.

```bash
CONCURRENCY=8 CONFIG=vllm_tp4_c8 python benchmark/vllm_4gpu_BFCL_v3_multi_turn_concurrent.py
```

예상: TP=4는 모든 요청이 로컬 GPU이므로 KV transfer 병목 없음 → 높은 동시성에서도 안정적.

---

### 중간 우선순위

**H. SGLang hicache cross-session 재사용 측정**

벤치마크를 두 번 실행 (1st run: cold SSD, 2nd run: warm SSD, /tmp/hicache 미삭제).
TTFT 차이를 측정해 SSD 캐시의 실제 cross-session 이득 정량화.

```bash
bash scripts/sglang/start_2P_2D.sh
python benchmark/sglang_BFCL_v3_multi_turn_base.py  # cold
python benchmark/sglang_BFCL_v3_multi_turn_base.py  # warm
```

**I. SGLang chunked prefix cache 활성화**

현재 `disable_chunked_prefix_cache: True`로 전체 prefix를 atomic하게 캐싱.
`False`로 바꾸면 부분 매칭도 가능 → 캐시 히트율 상승 가능.

```python
# SGLang server arg
--disable-chunked-prefix-cache False  # (또는 해당 플래그 이름 확인 필요)
```

가설: 특히 multi-turn에서 prefix 길이가 가변적이므로 partial match로 TTFT 5–15% 개선 가능.

**J. vllm-ppd 2P2pD concurrent**

2P2pD(PPD decode 모드)는 C=1만 측정됨. C=8에서 2P2D와 비교.

```bash
CONCURRENCY=8 python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py  # 2P2pD config
```

PPD decode가 short prefill을 D 노드에서 처리하므로 P 노드 부하 분산 → C=8에서도 TTFT 개선 유지 여부 확인.

**K. 출력 토큰 정규화 실험**

SGLang 3× 토큰 원인 규명. 동일 프롬프트로 vllm-ppd vs SGLang 비교하여 실제 응답 내용 차이 분석. `max_tokens=100`으로 강제 제한 후 동일 조건 재실험.

---

### 탐색적 연구 주제

**L. PD disaggregation 이득이 유효한 조건 분석**

현재 결과에서 PD disaggregation의 이득은 throughput에서만 명확함 (vllm-ppd C=8: 63.7 vs TP=4 C=1: 34.1). TTFT는 오히려 나빠짐 (0.130s → 0.590s+).

연구 질문: PD disaggregation이 TTFT도 개선되는 조건은?
- Context length가 매우 긴 경우 (prefill-heavy)
- Decode가 매우 긴 경우 (decode-heavy)
- Batch size를 더 크게 잡을 수 있는 경우
→ BFCL 이외의 워크로드 (예: long-context summarization, code generation) 실험 필요

**M. Mooncake vs NCCL P2P KV 전송 성능 비교**

SGLang (Mooncake)과 vllm-ppd (NCCL P2P) 모두 P→D KV 전송을 수행하지만 메커니즘이 다름.
- Mooncake: 고수준 KV 전송 추상화, 타임아웃 관리 포함
- NCCL P2P: 저수준 GPU 메모리 직접 전송

동일 workload에서 두 방식의 KV 전송 대역폭/latency 직접 측정.

**N. SGLang hicache SSD I/O 모니터링**

C=8+ 에서 SSD read/write latency를 `iostat -x` 또는 `/proc/diskstats`로 모니터링.
캐시 히트 시 SSD read latency가 TTFT에 얼마나 기여하는지 정량화.

**O. 4P4D 또는 1P3D 비율 실험**

현재 실험은 2P2D 고정. 동일 4-GPU에서:
- 1P3D: prefill 병목 시 D 증설 효과
- 3P1D: decode 병목 시 P 증설 효과
vllm-ppd의 proxy는 이미 N:M 구성을 지원하므로 스크립트 변경만으로 실험 가능.

---

## 실험 재현 방법

### vLLM 4-GPU TP=4 (Baseline)
```bash
conda activate vllm-source
vllm serve /home/uhmturks/hf_models/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name meta-llama/Llama-3.1-8B-Instruct-FC \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --max-model-len 131072
python benchmark/vllm_4gpu_BFCL_v3_multi_turn_base.py
```

### SGLang 2P2D (hicache, 순차)
```bash
conda activate sglang
bash scripts/sglang/start_2P_2D.sh          # 기본 ratio=1.2
# 또는 HICACHE_RATIO=0.5 bash scripts/sglang/start_2P_2D.sh
python benchmark/sglang_BFCL_v3_multi_turn_base.py
```

### SGLang 2P2D (동시, C=4 권장 — C=8+는 에러율 높음)
```bash
CONCURRENCY=4 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
```

> **주의**: 실험 사이에 반드시 cleanup 실행하여 `/tmp/hicache/` 삭제.  
> 이전 실험의 SSD 캐시가 남아있으면 TTFT가 인위적으로 낮아짐.

### vllm-ppd 2P2D (C=8 권장)
```bash
conda activate vllm-ppd
bash scripts/vllm-ppd/start_2P_2D.sh         # 기본 KV_BUFFER_GB=1
# 또는 KV_BUFFER_GB=2 bash ...
CONCURRENCY=8 python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py
```

### vllm-ppd KV buffer sweep
```bash
conda activate vllm-ppd
cd ~/experiments
bash scripts/vllm-ppd/run_buffer_sweep.sh
# buffer {1,2,4}GB × concurrency {8,12,16} 자동 실행
# 결과: results/vllm-ppd/buffer_sweep_summary.txt
```

---

## 결과 파일 목록

| 파일 | Config | C | 성공 | 비고 |
|------|--------|---|------|------|
| `results/vllm/bfcl_multiturn_results_vllm_tp4.json` | vLLM TP=4 | 1 | 200/200 | 기준선, per-turn TTFT 있음 |
| `results/vllm/bfcl_multiturn_results_vllm_4gpu_c8.json` | vLLM TP=4 | 8 | 200/200 | **132.9 tok/s**, per-turn TTFT 있음 |
| `results/sglang_hicache/bfcl_multiturn_results_sglang_2p2d.json` | SGLang 2P2D hicache | 1 | 200/200 | 28,323 tok |
| `results/sglang_hicache/bfcl_multiturn_results_1P_1D.json` | SGLang 1P1D | 1 | 38/200 ⚠️ | 타입 변환 버그, 참고용 |
| `results/sglang_hicache/bfcl_multiturn_results_sglang_2p2d_c4.json` | SGLang 2P2D | 4 | 153/200 | per-turn TTFT 있음 |
| `results/sglang_hicache/bfcl_multiturn_results_sglang_2p2d_c8.json` | SGLang 2P2D | 8 | 83/200 | SSD 경합 의심 |
| `results/bfcl_multiturn_results_sglang_2p2d_c12.json` | SGLang 2P2D | 12 | 67/200 | turn1 TTFT 이상 (7.7s) |
| `results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_2p2d_c8.json` | vllm-ppd 2P2D | 8 | 200/200 | 최고 throughput (63.7 tok/s) |
| `results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_2p2d_c12.json` | vllm-ppd 2P2D | 12 | 132/200 | buffer OOM |
| `results/vllm-ppd/bfcl_multiturn_results_2P_2D_vllmppd.json` | vllm-ppd 2P2D | 1 | 200/200 | 순차 |
| `results/vllm-ppd/bfcl_multiturn_results_2P_2pD_vllmppd.json` | vllm-ppd 2P2pD | 1 | 200/200 | PPD decode, TTFT 개선 |

---

## 모니터링

```
monitoring/
├── docker-compose.yaml        # Prometheus(9090) + Grafana(3000) + Pushgateway(9091)
├── prometheus.yml             # scrape targets
└── grafana/
    └── dashboards/vllm_bfcl_agentic.json   # 커스텀 대시보드 (v5)
```

**대시보드 패널 (v5):**
- TTFT/TPOT percentile, Request queue, Prefix cache hit rate
- P vs D 노드별 KV cache 점유율
- SGLang hicache 3-tier: L1 GPU (≈0%), L2 CPU DRAM (VmRSS), L3 SSD (du)
- BFCL Per-Turn Metrics (Pushgateway): turn별 TTFT/TPOT/context 성장

```bash
# SSH 포트 포워딩
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 9091:localhost:9091 uhmturks@server17
```

---

## 디렉토리 구조

```
experiments/
│
├── index.md                          ← 이 파일
├── CLAUDE.md
│
├── benchmark/
│   ├── kv_cache_poller.py
│   ├── sglang_hicache_exporter.py    # SGLang Prometheus exporter (port 9199)
│   ├── vllm_4gpu_BFCL_v3_multi_turn_base.py
│   ├── vllm_4gpu_BFCL_v3_multi_turn_concurrent.py
│   ├── vllmppd_BFCL_v3_multi_turn_base.py
│   ├── vllmppd_BFCL_v3_multi_turn_concurrent.py
│   ├── sglang_BFCL_v3_multi_turn_base.py
│   └── sglang_BFCL_v3_multi_turn_concurrent.py
│
├── scripts/
│   ├── vllm/start_2P_2D.sh
│   ├── vllm-ppd/
│   │   ├── start_2P_2D.sh            # KV_BUFFER_GB=N 환경변수 지원
│   │   ├── start_2P_2pD.sh
│   │   ├── run_buffer_sweep.sh       # buffer(1/2/4GB) × C(8/12/16) 자동 스윕
│   │   └── cleanup_all.sh
│   └── sglang/
│       ├── start_2P_2D.sh            # HICACHE_RATIO=N 환경변수 지원
│       └── cleanup_all.sh
│
├── results/
│   ├── bfcl_multiturn_results_sglang_2p2d_c12.json
│   ├── vllm/
│   │   └── bfcl_multiturn_results_vllm_tp4.json
│   ├── vllm-ppd/
│   │   ├── bfcl_multiturn_results_2P_2D_vllmppd.json
│   │   ├── bfcl_multiturn_results_2P_2pD_vllmppd.json
│   │   ├── bfcl_multiturn_results_vllm_ppd_2p2d_c8.json
│   │   └── bfcl_multiturn_results_vllm_ppd_2p2d_c12.json
│   └── sglang_hicache/
│       ├── bfcl_multiturn_results_1P_1D.json
│       ├── bfcl_multiturn_results_sglang_2p2d.json
│       ├── bfcl_multiturn_results_sglang_2p2d_c4.json
│       └── bfcl_multiturn_results_sglang_2p2d_c8.json
│
├── logs/sglang/                      ← SGLang 서버 로그 (gitignore)
├── logs/vllm-ppd/                    ← vllm-ppd 서버 로그 (gitignore)
└── monitoring/
```
