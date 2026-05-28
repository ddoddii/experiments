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
| **conda 환경** | `vllm-source` (vLLM 2P2D) / `vllm-ppd` (vllm-ppd) / `sglang` (SGLang) |

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
| **Per-req throughput** | `(n_tokens − 1) / decode_time` (tok/s) | 클라이언트 |
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

**특징**: P/D 분리 없이 4-GPU를 하나의 큰 GPU처럼 사용. Prefill과 decode가 동일 GPU 풀을 공유하고 인접 메모리에 KV cache를 유지해 TTFT와 TPOT 모두 최소. vLLM의 RadixAttention(prefix caching)이 활성화되어 turn 1부터 system prompt + tool 정의 부분의 KV를 GPU DRAM에서 재사용함.

---

### Config 2: SGLang 2P2D — hicache 3-tier KV offload

| 항목 | 내용 |
|------|------|
| **구성** | Prefill 2 + Decode 2, KV 전송: Mooncake |
| **GPU** | P1=GPU0(30000), P2=GPU1(30001), D1=GPU2(30002), D2=GPU3(30003) |
| **Router** | port 8000 (SGLang router) |
| **KV 전송** | Mooncake metadata server (port 8080) |
| **hicache** | `--enable-hierarchical-cache --hicache-ratio 1.2 --hicache-storage-backend file` |
| **시작** | `scripts/sglang/start_2P_2D.sh` |
| **벤치마크** | `benchmark/sglang_BFCL_v3_multi_turn_base.py` |

#### SGLang hicache 작동 방식 (핵심)

SGLang hicache는 KV cache를 3계층으로 관리한다:

```
┌─────────────────────────────────────────────────────┐
│                  SGLang hicache 구조                 │
│                                                     │
│  ┌─────────────┐    evict    ┌─────────────────┐   │
│  │ L1: GPU VRAM│ ──────────▶ │ L2: CPU DRAM    │   │
│  │  ~25.5 GB   │ (write-thru)│  (Python 프로세스│   │
│  │  (동적 캐시) │             │   RSS 기반)     │   │
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

현재 실험 설정에서 P 노드는 `write_through` 정책을 사용한다. 이 동작을 이해하는 것이 핵심이다:

1. Prefill 완료 시 KV block이 GPU(L1)에 생성됨
2. **즉시** SSD(L3)에도 동시에 기록됨 (write-through)
3. GPU의 KV block은 해당 요청 처리 후 free됨
4. 따라서 **steady-state에서 GPU KV cache usage ≈ 0%** — 이것이 정상 동작

이 때문에 Grafana에서 L1 KV cache가 항상 0%로 보이는 것은 버그가 아니라 의도된 설계임.  
실제로 GPU에 할당된 KV 메모리는 25.52 GB (token capacity = 209,026 tokens)이지만, 이 공간은 단기 작업 버퍼로만 사용된다.

**Content-addressable SSD caching**

SSD에 저장되는 파일명은 token sequence prefix의 해시 값이다:
- `system prompt + tool 정의` → 고정 prefix → 동일 해시 → SSD 캐시 히트
- P1과 P2가 `/tmp/hicache/`를 **공유**하므로 cross-node 캐시 재사용 가능
- 다음 실험 시작 전 `/tmp/hicache/` 삭제 필요 (캐시 오염 방지)

**L3 SSD가 vllm-ppd 대비 무엇을 제공하는가?**

| 특성 | vllm-ppd (GPU-only KV) | SGLang hicache (SSD KV) |
|------|------------------------|-------------------------|
| KV 저장소 | GPU DRAM에 LRU 유지 | SSD에 영구 저장 |
| 용량 | ~25 GB (GPU 한계) | 수백 GB (디스크 한계) |
| 재시작 후 재사용 | ❌ GPU 메모리 소멸 | ✅ 디스크 파일 유지 (선택적) |
| 크로스 세션 재사용 | ❌ | ✅ 동일 prefix 해시면 재사용 |
| TTFT 이득 (캐시 히트 시) | GPU DRAM 읽기 (~μs) | SSD 읽기 (~ms, 의미 있는 절감) |
| Decode 속도 | P→D KV 전송 포함 | P→D KV 전송 후 decode |

**실제 관측 (SGLang 2P2D 실험 중)**:
- L3 SSD 사용량: 벤치마크 1,250초 동안 ~42 GB까지 성장
- L2 CPU DRAM: ~1 GB 안정 유지 (LRU 상위 계층)
- L1 GPU: ≈0% (write_through로 즉시 evict)
- TPOT: 0.023 s/tok — vllm-ppd (0.064 s/tok) 대비 **2.8× 빠름**

**현재 실험의 SSD 캐시 한계**:
`disable_chunked_prefix_cache: True` 설정으로 인해 전체 prefix를 atomic하게 캐시한다. Chunked prefix cache를 활성화하면 부분 매칭도 가능해져 캐시 히트율이 올라갈 수 있다.

---

### Config 3: vllm-ppd 2P2D — P2pNcclConnector

| 항목 | 내용 |
|------|------|
| **구성** | Prefill 2 + Decode 2, KV 전송: P2pNcclConnector (NCCL P2P) |
| **GPU** | P1=GPU0(8100), P2=GPU1(8101), D1=GPU2(8200), D2=GPU3(8201) |
| **Proxy** | `ppd/comprehensive_proxy.py` — HTTP port 10001, ZMQ port 30001 |
| **KV buffer** | P: 1 GB send / D: 10 GB receive (per node) |
| **시작** | `scripts/vllm-ppd/start_2P_2D.sh` |
| **벤치마크** | `benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py` |

**동시성 한계 분석**:
- Per-request KV transfer ≈ 0.64 GB (Llama 3.1 8B, avg context length)
- P node send buffer = 1 GB → **C=2** 이상이면 이미 버퍼 압박
- D node recv buffer = 10 GB → **C=15** 이하에서 안전
- 실험 결과: **C=8** 성공 (200/200), **C=12** 실패 (132/200, P node buffer OOM)

**2P2D vs 2P2pD**:
- 2P2D: Decode 노드가 순수 decode만 담당
- 2P2pD: Decode 노드가 PPD(Partial Prefill-Decode) 모드 — short prefill을 decode 노드가 처리해 P 노드 부하 분산. TTFT 약 14% 개선 (0.590s → 0.429s)

---

### Config 4: vLLM 2P2D (xpyd, non-ppd)

| 항목 | 내용 |
|------|------|
| **구성** | upstream vLLM의 실험적 P/D disaggregation |
| **GPU** | P1=GPU0(8100), P2=GPU1(8101), D1=GPU2(8200), D2=GPU3(8201) |
| **Proxy** | `disagg_proxy_p2p_nccl_xpyd.py` — port 10001 |
| **시작** | `scripts/vllm/start_2P_2D.sh` |

---

## 실험 결과 요약

### 메인 비교 테이블

| Config | TTFT avg | TPOT avg | Per-req Tput | Overall Tput | 성공 | Wall time |
|--------|---------|---------|-------------|-------------|------|----------|
| **vLLM TP=4** | **0.130 s** | **0.022 s** | **50.5 tok/s** | 34.1 tok/s | 200/200 | 280 s |
| SGLang 2P2D (C=1) | 0.825 s | **0.023 s** | **43.9 tok/s** | 23.0 tok/s | 200/200 | 1,250 s |
| SGLang 1P1D (C=1) | 0.395 s | 0.023 s | 44.0 tok/s | 29.3 tok/s | 38/200 ⚠️ | 151 s |
| vllm-ppd 2P2D (C=1) | 0.517 s | 0.064 s | 17.1 tok/s | 10.6 tok/s | 200/200 | 887 s |
| vllm-ppd 2P2pD (C=1) | 0.429 s | 0.063 s | 17.2 tok/s | 12.0 tok/s | 200/200 | 764 s |
| **vllm-ppd 2P2D (C=8)** | 0.872 s | 0.072 s | 15.2 tok/s | **63.7 tok/s** | 200/200 | 148 s |
| vllm-ppd 2P2D (C=12) | 1.305 s | 0.076 s | 14.3 tok/s | 22.1 tok/s | 132/200 ❌ | 197 s |

> ⚠️ SGLang 1P1D: func_doc 타입 변환 미적용(float→number 변환 누락)으로 162개 에러 발생, 수치는 참고용.  
> ❌ vllm-ppd C=12: P node KV send buffer OOM으로 68개 에러.

**핵심 인사이트**:
- **최저 TTFT**: vLLM TP=4 (0.130 s) — PD 분리 없이 GPU KV가 항상 인접
- **최고 단일 요청 decode 속도**: vLLM TP=4와 SGLang 동급 (~44–50 tok/s)
- **최고 시스템 throughput**: vllm-ppd C=8 (63.7 tok/s) — PD disaggregation의 병렬처리 이득
- **SGLang vs vllm-ppd decode 비교**: SGLang TPOT 0.023s vs vllm-ppd 0.064s — **2.8× 빠름**

---

### Per-Turn TTFT 패턴 (Prefix Cache 효과)

Agentic 워크로드에서 turn이 반복될수록 system prompt + tool 정의가 KV cache에 쌓인다. 이를 재사용하면 TTFT가 떨어져야 한다.

| Turn | vLLM TP=4 | vllm-ppd C=1 | vllm-ppd C=8 |
|------|-----------|-------------|-------------|
| 0 (첫 요청) | 0.221 s | 0.967 s | 1.327 s |
| 1 | **0.093 s** (−58%) | **0.321 s** (−67%) | **0.687 s** (−48%) |
| 2 | 0.090 s | 0.310 s | 0.658 s |
| 3 | 0.092 s | 0.313 s | 0.565 s |
| 4 | 0.093 s | 0.319 s | 0.601 s |
| 5 | 0.099 s | 0.321 s | 0.478 s |

**관찰**:
- 모든 config에서 **turn 0 → turn 1에 TTFT 급감** — prefix cache hit 확인
- vLLM TP=4: 0.221s → 0.093s (2.4× 개선). Turn 1부터 system prompt + tool 정의 KV 재사용
- vllm-ppd C=1: 0.967s → 0.321s (3.0× 개선). P 노드 KV cache에 prefix 유지
- vllm-ppd C=8: 1.327s → 0.687s (1.9× 개선). 동시 요청 경쟁으로 캐시 히트율 낮아짐
- SGLang 2P2D: 전체 요약만 있고 per-turn TTFT 데이터 부재 (0.825s avg는 turn 0~N 혼합)

**vllm-ppd turn 0 TTFT가 높은 이유**:  
P 노드가 prefill 후 KV를 D 노드로 전송(P2pNccl)하는 시간이 포함. Llama 3.1 8B 기준 KV transfer ≈ 0.6 GB/request, NVLink 없이 PCIe bus를 통해 전송.

---

### KV Cache 사용 패턴 (vllm-ppd C=8 측정)

| Node | 역할 | Max KV | Mean KV |
|------|------|--------|---------|
| P1 (GPU0) | prefill | 3.1% | 0.75% |
| P2 (GPU1) | prefill | 2.3% | 0.58% |
| D1 (GPU2) | decode | **17.2%** | 5.9% |
| D2 (GPU3) | decode | **15.5%** | 5.4% |

- **P 노드**: KV를 즉시 D 노드로 전송하므로 거의 비어 있음
- **D 노드**: 여러 요청의 KV가 누적되어 P 대비 ~7–8× 높은 점유율
- 이것이 PD disaggregation의 핵심 효과: D 노드가 KV를 장기 보유하며 decode에 집중

vllm-ppd C=12에서는 D1이 최대 26.3%까지 상승 — OOM 직전 고수위 기록.

---

## SGLang hicache vs vllm-ppd 상세 비교

### Decode 속도 차이 (가장 큰 차이점)

| | SGLang 2P2D | vllm-ppd 2P2D C=1 |
|--|-------------|------------------|
| **TPOT** | **0.023 s/tok** | 0.064 s/tok |
| **Per-req Tput** | **43.9 tok/s** | 17.1 tok/s |
| 격차 | — | SGLang이 **2.8× 빠름** |

SGLang이 decode에서 훨씬 빠른 이유:
1. **Continuous batching 최적화**: SGLang의 스케줄러가 decode step을 더 효율적으로 배치
2. **FlashInfer 커널**: SGLang은 decode-optimized CUDA 커널 사용
3. **vllm-ppd 오버헤드**: D 노드가 KV를 받은 후 자체 attention 연산 시 일부 추가 오버헤드

### TTFT 비교

| | SGLang 2P2D (avg) | vllm-ppd C=1 Turn 0 | vllm-ppd C=1 Turn 1+ |
|--|------------------|--------------------|---------------------|
| TTFT | 0.825 s | 0.967 s | ~0.315 s |

SGLang의 평균 TTFT(0.825s)는 vllm-ppd turn 0(0.967s)보다 낮지만, turn 1+의 캐시 히트 TTFT(0.315s)보다는 높다. SGLang도 per-turn TTFT를 수집하면 동일한 패턴이 보일 것으로 예상.

### 총 Output Token 수 차이 (주의 사항)

| Config | Total Output Tokens | Per-turn avg |
|--------|---------------------|-------------|
| vLLM TP=4 | 9,557 | ~12.9 tok |
| vllm-ppd C=1 | 9,440 | ~12.7 tok |
| vllm-ppd C=8 | 9,424 | ~12.7 tok |
| **SGLang 2P2D** | **28,791** | **~38.9 tok** |

SGLang 2P2D의 총 output token이 ~3× 많다. 응답 형식 차이 또는 MAX_TOKENS 설정 차이에 기인할 수 있으며, 직접적인 throughput 비교 시 이 점을 고려해야 함.

### 시스템 Throughput 비교

C=1 순차 처리 기준으로 SGLang이 vllm-ppd보다 overall throughput이 높은 이유는 decode가 2.8× 빠르기 때문이다. 동시성을 높이면 (SGLang C=8 미실험) 격차는 더 벌어질 것으로 예상.

vllm-ppd의 장점은 동시성 처리 효율성: C=8에서 63.7 tok/s로 baseline TP=4(34.1 tok/s)의 **1.87×** 달성.

---

## 결과 파일 목록

| 파일 | Config | 비고 |
|------|--------|------|
| `results/bfcl_multiturn_results_vllm_tp4.json` | vLLM TP=4 | 기준선, 200/200 성공 |
| `results/bfcl_multiturn_results_vllm_ppd_2p2d.json` | vllm-ppd C=1 | 순차, per-turn 데이터 있음 |
| `results/bfcl_multiturn_results_vllm_ppd_2p2d_c8.json` | vllm-ppd C=8 | 최고 throughput, KV 모니터링 포함 |
| `results/bfcl_multiturn_results_vllm_ppd_2p2d_c12.json` | vllm-ppd C=12 | 68 에러 (OOM), 참고용 |
| `results/sglang_hicache/bfcl_multiturn_results_2P_2D.json` | SGLang 2P2D | 요약만 (per-turn 미수집) |
| `results/sglang_hicache/bfcl_multiturn_results_1P_1D.json` | SGLang 1P1D | 162 에러 (타입 변환 버그), 참고용 |
| `results/vllm-ppd/bfcl_multiturn_results_2P_2D_vllmppd.json` | vllm-ppd 구버전 2P2D | 이전 설정 |
| `results/vllm-ppd/bfcl_multiturn_results_2P_2pD_vllmppd.json` | vllm-ppd 구버전 2P2pD | PPD decode 실험 |

---

## 모니터링

```
monitoring/
├── docker-compose.yaml        # Prometheus(9090) + Grafana(3000) + Pushgateway(9091)
├── prometheus.yml             # scrape targets: 모든 config P/D nodes
└── grafana/
    ├── dashboards/
    │   └── vllm_bfcl_agentic.json   # 커스텀 대시보드 (v5)
    └── provisioning/
        ├── datasources/prometheus.yml
        └── dashboards/dashboards.yml
```

**대시보드 패널 구성 (v5):**
- Summary Stats: TTFT P50/P99, TPOT P50, Generation Throughput, KV Cache, Prefix Cache Hit Rate
- Latency: TTFT/TPOT percentile timeseries
- Throughput & Queue: Token throughput (prompt vs generation), Request queue
- Memory: GPU KV cache per instance, Prefix cache hit rate
- P vs D Node KV Cache: P/D 노드별 KV cache 점유율 비교
- **SGLang hicache 3-tier panels** (v5):
  - L1 GPU KV Cache Usage (≈0 with write_through — 정상)
  - L2 CPU DRAM (VmRSS 기반)
  - L3 SSD (`/tmp/hicache/` du 기반)
  - Active Requests per Instance
  - Prefix Cache Hit Rate (scheduler log 파싱)
  - KV Cache Allocated GB (static config, 25.52 GB)
- BFCL Per-Turn Metrics (Pushgateway): 클라이언트 측 turn별 TTFT, TPOT, Context 성장

**SSH 포트 포워딩:**
```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 9091:localhost:9091 uhmturks@server17
```

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

### SGLang 2P2D (hicache)
```bash
# 이전 실험 잔여물 정리 (SSD 캐시 포함)
bash scripts/sglang/cleanup_all.sh

# 서버 시작 (Mooncake + 4 nodes + router + exporter 자동 시작)
conda activate sglang
bash scripts/sglang/start_2P_2D.sh

# 벤치마크 (순차)
python benchmark/sglang_BFCL_v3_multi_turn_base.py

# 벤치마크 (동시, C=4 권장)
CONCURRENCY=4 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
```

> **주의**: 실험 사이에 반드시 `cleanup_all.sh` 실행하여 `/tmp/hicache/` 삭제.  
> 이전 실험의 SSD 캐시가 남아있으면 다음 실험의 TTFT가 인위적으로 낮아짐.

### vllm-ppd 2P2D (순차)
```bash
conda activate vllm-ppd
bash scripts/vllm-ppd/start_2P_2D.sh
python benchmark/vllmppd_BFCL_v3_multi_turn_base.py
```

### vllm-ppd 2P2D (동시, C=8 권장)
```bash
conda activate vllm-ppd
bash scripts/vllm-ppd/start_2P_2D.sh
CONCURRENCY=8 python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py
```

---

## 디렉토리 구조

```
experiments/
│
├── index.md                          ← 이 파일
├── CLAUDE.md                         ← Claude Code 지시사항
│
├── benchmark/
│   ├── kv_cache_poller.py            # GPU KV cache 백그라운드 폴링 (vllm-ppd용)
│   ├── sglang_hicache_exporter.py    # SGLang Prometheus exporter (port 9199)
│   ├── vllm_4gpu_BFCL_v3_multi_turn_base.py
│   ├── vllm_2P2D_BFCL_v3_multi_turn_base.py
│   ├── vllmppd_BFCL_v3_multi_turn_base.py
│   ├── vllmppd_BFCL_v3_multi_turn_concurrent.py  # CONCURRENCY=N 환경변수
│   ├── sglang_BFCL_v3_multi_turn_base.py
│   └── sglang_BFCL_v3_multi_turn_concurrent.py   # CONCURRENCY=N 환경변수
│
├── data/
│   ├── BFCL_v3_multi_turn_base.json
│   └── multi_turn_func_doc/          # Tool 함수 정의 (8종)
│
├── possible_answer/
│   └── BFCL_v3_multi_turn_base.json
│
├── gorilla/                          # Berkeley 공식 BFCL evaluator
│
├── ppd/
│   ├── comprehensive_proxy.py
│   └── optimizer/ppd_decision_engine.py
│
├── scripts/
│   ├── vllm/start_2P_2D.sh
│   ├── vllm-ppd/
│   │   ├── start_2P_2D.sh
│   │   ├── start_2P_2pD.sh
│   │   └── cleanup_all.sh
│   └── sglang/
│       ├── start_2P_2D.sh            # hicache exporter 자동 시작 포함
│       └── cleanup_all.sh            # /tmp/hicache/ 포함 전체 정리
│
├── monitoring/
│   ├── docker-compose.yaml
│   ├── prometheus.yml
│   └── grafana/dashboards/vllm_bfcl_agentic.json
│
├── results/
│   ├── bfcl_multiturn_results_vllm_tp4.json
│   ├── bfcl_multiturn_results_vllm_ppd_2p2d.json
│   ├── bfcl_multiturn_results_vllm_ppd_2p2d_c8.json
│   ├── bfcl_multiturn_results_vllm_ppd_2p2d_c12.json
│   ├── vllm-ppd/
│   │   ├── bfcl_multiturn_results_2P_2D_vllmppd.json
│   │   └── bfcl_multiturn_results_2P_2pD_vllmppd.json
│   └── sglang_hicache/
│       ├── bfcl_multiturn_results_1P_1D.json
│       └── bfcl_multiturn_results_2P_2D.json
│
└── logs/                             ← 서버 로그 (gitignore)
```

---

## 결과 파일 구조

```json
{
  "summary": {
    "config": "vllm_ppd_2p2d_c8",
    "concurrency": 8,
    "total_items": 200,
    "success_items": 200,
    "total_output_tokens": 9424,
    "total_wall_time_s": 147.97,
    "overall_throughput_tok_per_s": 63.69,
    "avg_ttft_s": 0.8715,
    "avg_tpot_s": 0.0723,
    "avg_throughput_tok_per_s": 15.2,
    "kv_cache_per_gpu": {
      "p1": {"min": 0.0, "max": 0.031, "mean": 0.008},
      "d1": {"min": 0.0, "max": 0.172, "mean": 0.059}
    }
  },
  "results": [
    {
      "id": "multi_turn_base_0",
      "num_turns": 4,
      "avg_ttft_s": 0.83,
      "ttft_by_turn": {"0": 1.327, "1": 0.687, "2": 0.658, "3": 0.565},
      "turns": [
        {
          "turn": 0,
          "ttft_s": 1.327,
          "tpot_s": 0.072,
          "output_tokens": 18,
          "context_chars": 4821
        }
      ]
    }
  ]
}
```

`ttft_by_turn`: turn 인덱스별 TTFT → context가 길어질수록 TTFT 증가하는 agentic 패턴 관찰 가능.
