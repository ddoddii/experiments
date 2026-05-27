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
| 구조 | 아이템당 2~6 turn의 대화, 각 turn마다 tool call 포함 |
| Tool 종류 | GorillaFileSystem, TicketAPI, MessageAPI, MathAPI, TradingBot, TwitterAPI, TravelAPI, VehicleControlAPI |
| 특징 | **Agentic workload**: 매 turn마다 system prompt + tool 정의 + 누적 대화 이력이 context로 전달 → prefill-heavy |

Ground truth: `possible_answer/BFCL_v3_multi_turn_base.json`  
공식 평가기: `gorilla/` (Berkeley Gorilla 공식 repo)

---

## 측정 메트릭

| 메트릭 | 정의 | 측정 위치 |
|--------|------|----------|
| **TTFT** | Time To First Token = `t_first_token − t_request` | 클라이언트 (benchmark script) |
| **TPOT** | Time Per Output Token = `(t_last − t_first) / (n_tokens − 1)` | 클라이언트 |
| **Per-req throughput** | `(n_tokens − 1) / decode_time` (tok/s) | 클라이언트 |
| **Overall throughput** | `total_output_tokens / total_wall_time` (tok/s) | 실험 전체 |
| **KV cache usage %** | `vllm:kv_cache_usage_perc` — GPU KV 블록 풀 대비 사용 비율 | Prometheus (서버) |
| **Context chars** | 매 turn 전송되는 전체 conversation 길이 (chars) | 클라이언트 |

> **KV cache usage %** 기준:  
> vLLM이 시작 시 `gpu_memory_utilization(0.85) × VRAM − 모델 가중치`로 미리 확보한 **~25 GB KV 블록 풀** 대비 현재 사용 비율.  
> P2pNccl 전송 버퍼(별도 할당)는 포함되지 않음.

---

## 실험 설정 (Configurations)

### 1. vLLM 4-GPU TP=4 (Baseline)

| 항목 | 내용 |
|------|------|
| 구성 | 단일 인스턴스, Tensor Parallelism 4 |
| GPU | GPU 0~3 전체 사용 |
| 엔드포인트 | `http://localhost:8000` |
| served-model-name | `meta-llama/Llama-3.1-8B-Instruct-FC` |
| 시작 방법 | `vllm serve ... --tensor-parallel-size 4 --enable-auto-tool-choice --tool-call-parser llama3_json` |
| 벤치마크 | `benchmark/vllm_4gpu_BFCL_v3_multi_turn_base.py` |

---

### 2. SGLang 2P2D (hicache)

| 항목 | 내용 |
|------|------|
| 구성 | Prefill 2개 + Decode 2개, KV 전송: Mooncake |
| GPU | P1=GPU0(30000), P2=GPU1(30001), D1=GPU2(30002), D2=GPU3(30003) |
| Router | port 8000 (SGLang router) |
| KV 전송 | Mooncake metadata server (port 8080) |
| 특이사항 | `--enable-hierarchical-cache --hicache-ratio 1.2` (CPU RAM KV offload) |
| 시작 방법 | `scripts/sglang/start_2P_2D.sh` |
| 벤치마크 | `benchmark/sglang_BFCL_v3_multi_turn_base.py` |

SGLang 1P1D도 동일 스크립트에서 P/D 각 1개로 실험.

---

### 3. vllm-ppd 2P2D (P2pNccl)

| 항목 | 내용 |
|------|------|
| 구성 | Prefill 2개 + Decode 2개, KV 전송: P2pNcclConnector |
| GPU | P1=GPU0(8100), P2=GPU1(8101), D1=GPU2(8200), D2=GPU3(8201) |
| Proxy | `ppd/comprehensive_proxy.py` — HTTP port 10001, ZMQ port 30001 |
| served-model-name | 없음 → model ID = full path |
| KV buffer | P nodes: 1 GB send / D nodes: 10 GB receive |
| 시작 방법 | `scripts/vllm-ppd/start_2P_2D.sh` (2P2D) / `start_2P_2pD.sh` (2P2pD) |
| 벤치마크 (순차) | `benchmark/vllmppd_BFCL_v3_multi_turn_base.py` |
| 벤치마크 (병렬) | `benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py` |

**2P2D vs 2P2pD 차이**: 2P2pD는 Decode 노드가 PPD(Partial Prefill-Decode) 모드로 동작.

**동시성(Concurrency) 한계**:
- P node KV send buffer = 1 GB, per-request KV ≈ 0.64 GB
- D node KV recv buffer = 10 GB, C=8 → 5.1 GB 사용 (안전)
- **C=8**: 안전 ✅ / **C=12**: P node buffer OOM ❌ / **C=16**: D node buffer OOM ❌

---

### 4. vLLM 2P2D (xpyd, non-ppd)

| 항목 | 내용 |
|------|------|
| 구성 | Prefill 2개 + Decode 2개, KV 전송: P2pNcclConnector (upstream vLLM) |
| GPU | P1=GPU0(8100), P2=GPU1(8101), D1=GPU2(8200), D2=GPU3(8201) |
| Proxy | `xpyd` (disagg_proxy_p2p_nccl_xpyd.py) — port 10001 |
| served-model-name | `Llama` |
| 시작 방법 | `scripts/vllm/start_2P_2D.sh` |
| 벤치마크 | `benchmark/vllm_2P2D_BFCL_v3_multi_turn_base.py` |

---

## 실험 결과 요약

> 모든 결과: 벤치마크 200개 아이템, 순차 처리(C=1) 기준  
> vllm-ppd C=8은 동시 8개 처리

| Config | TTFT avg | TPOT avg | Overall Tput | 성공 | Wall time |
|--------|---------|---------|-------------|------|----------|
| **vLLM 4-GPU TP=4** | **0.130s** | **0.022s** | **34.1 tok/s** | 200/200 | 280s |
| SGLang 2P2D (hicache) | 0.825s | 0.023s | 23.0 tok/s | 200/200 | 1250s |
| SGLang 1P1D (hicache) | 0.395s | 0.023s | 29.3 tok/s | 38/200 ⚠️ | 151s |
| vllm-ppd 2P2D (C=1) | 0.590s | 0.065s | 11.0 tok/s | 200/200 | 732s |
| vllm-ppd 2P2pD (C=1) | 0.429s | 0.063s | 12.0 tok/s | 200/200 | 764s |
| **vllm-ppd 2P2D (C=8)** | 0.872s | 0.072s | **63.7 tok/s** | 200/200 | 148s |

> ⚠️ SGLang 1P1D: func_doc 타입 변환 오류(float→number 미처리)로 162개 에러, 수정 후 재실험 필요

### KV Cache 관측 (vllm-ppd C=8)

| Node | Max KV cache | Mean KV cache |
|------|-------------|--------------|
| P1 (prefill, GPU0) | 3.1% | 0.75% |
| P2 (prefill, GPU1) | 2.3% | 0.58% |
| D1 (decode, GPU2) | **17.2%** | 5.9% |
| D2 (decode, GPU3) | **15.5%** | 5.4% |

→ D node가 P node 대비 ~7배 높은 KV 점유 → disaggregation이 decode 노드에 KV를 집중시키는 것 확인.

---

## 모니터링

```
monitoring/
├── docker-compose.yaml        # Prometheus(9090) + Grafana(3000) + Pushgateway(9091)
├── prometheus.yml             # scrape targets: vllm-ppd P/D nodes (8100~8201)
└── grafana/
    ├── dashboards/
    │   └── vllm_bfcl_agentic.json   # 커스텀 대시보드 (v3)
    └── provisioning/
        ├── datasources/prometheus.yml
        └── dashboards/dashboards.yml
```

**대시보드 패널 구성:**
- Summary Stats: TTFT P50/P99, TPOT P50, Generation Throughput, KV Cache, Prefix Cache Hit Rate
- Latency: TTFT/TPOT percentile timeseries
- Throughput & Queue: Token throughput (prompt vs generation), Request queue
- Memory: GPU KV cache per instance, Prefix cache hit rate
- **P vs D Node KV Cache (vllm-ppd 전용)**: P/D 노드별 KV cache 비교 패널
- BFCL Per-Turn Metrics (Pushgateway): 클라이언트 측 turn별 TTFT, TPOT, Context 성장

**SSH 포트 포워딩:**
```bash
ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 9091:localhost:9091 uhmturks@server17
```

---

## 디렉토리 구조

```
experiments/
│
├── index.md                          ← 이 파일
├── CLAUDE.md                         ← Claude Code 지시사항
│
├── benchmark/                        ← 벤치마크 스크립트
│   ├── kv_cache_poller.py            # GPU KV cache 백그라운드 폴링 유틸
│   ├── vllm_4gpu_BFCL_v3_multi_turn_base.py      # vLLM TP=4
│   ├── vllm_2P2D_BFCL_v3_multi_turn_base.py      # vLLM 2P2D (xpyd)
│   ├── vllmppd_BFCL_v3_multi_turn_base.py        # vllm-ppd 2P2D (순차)
│   ├── vllmppd_BFCL_v3_multi_turn_concurrent.py  # vllm-ppd 2P2D (동시: CONCURRENCY=N)
│   ├── sglang_BFCL_v3_multi_turn_base.py         # SGLang
│   └── vllm_BFCL_v3_multi_turn_base.py           # (구버전)
│
├── data/                             ← 벤치마크 데이터
│   ├── BFCL_v3_multi_turn_base.json  # 200개 multi-turn 아이템
│   └── multi_turn_func_doc/          # Tool 함수 정의 (8종)
│       ├── gorilla_file_system.json
│       ├── math_api.json
│       ├── message_api.json
│       ├── posting_api.json          # TwitterAPI
│       ├── ticket_api.json
│       ├── trading_bot.json
│       ├── travel_booking.json       # TravelAPI
│       └── vehicle_control.json
│
├── possible_answer/
│   └── BFCL_v3_multi_turn_base.json  # Ground truth (정답)
│
├── gorilla/                          # Berkeley 공식 BFCL evaluator (submodule)
│
├── ppd/                              # vllm-ppd custom proxy 코드
│   ├── comprehensive_proxy.py        # HTTP↔ZMQ proxy (port 10001)
│   ├── config.py
│   └── optimizer/
│       └── ppd_decision_engine.py    # PPD 모드 결정 엔진
│
├── scripts/                          ← 서버 시작/종료 스크립트
│   ├── vllm/
│   │   ├── start_2P_2D.sh            # vLLM 2P2D (xpyd proxy)
│   │   └── run_benchmark_2p2d.sh
│   ├── vllm-ppd/
│   │   ├── config.sh                 # 공통 설정 (MODEL_PATH, 포트 등)
│   │   ├── common.sh                 # 공통 함수 (check_gpu, cleanup 등)
│   │   ├── start_2P_2D.sh            # vllm-ppd 2P+2D
│   │   ├── start_2P_2pD.sh           # vllm-ppd 2P+2pD (PPD decode)
│   │   └── cleanup_all.sh
│   └── sglang/
│       ├── start_2P_2D.sh            # SGLang 2P2D (Mooncake)
│       ├── stop.sh
│       └── cleanup_all.sh
│
├── monitoring/                       ← Prometheus + Grafana (Docker)
│   ├── docker-compose.yaml
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   └── vllm_bfcl_agentic.json
│       └── provisioning/
│           ├── datasources/prometheus.yml
│           └── dashboards/dashboards.yml
│
├── results/                          ← 실험 결과 JSON
│   ├── bfcl_multiturn_results_vllm_tp4.json        # vLLM TP=4
│   ├── bfcl_multiturn_results_vllm_ppd_2p2d.json   # vllm-ppd 2P2D (C=1)
│   ├── bfcl_multiturn_results_vllm_ppd_2p2d_c8.json  # vllm-ppd 2P2D (C=8)
│   ├── comparison_2P2D.png
│   ├── vllm-ppd/
│   │   ├── bfcl_multiturn_results_2P_2D_vllmppd.json   # 구버전 결과
│   │   └── bfcl_multiturn_results_2P_2pD_vllmppd.json
│   └── sglang_hicache/
│       ├── bfcl_multiturn_results_1P_1D.json        # SGLang 1P1D
│       └── bfcl_multiturn_results_2P_2D.json        # SGLang 2P2D
│
└── logs/                             ← 서버 로그 (gitignore)
    ├── vllm/2P_2D/                   # prefill1/2, decode1/2, proxy
    ├── vllm-ppd/2P_2D/
    ├── vllm-ppd/2P_2pD/
    └── sglang/                       # p1/p2, d1/d2, mooncake, router
```

---

## 실험 재현 방법

### vllm-ppd 2P2D (순차)
```bash
conda activate vllm-ppd
bash scripts/vllm-ppd/start_2P_2D.sh
python benchmark/vllmppd_BFCL_v3_multi_turn_base.py
# 결과: results/bfcl_multiturn_results_vllm_ppd_2p2d.json
```

### vllm-ppd 2P2D (동시, C=8 권장)
```bash
conda activate vllm-ppd
bash scripts/vllm-ppd/start_2P_2D.sh
CONCURRENCY=8 PUSHGATEWAY_URL=localhost:9091 \
  python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py
# 결과: results/bfcl_multiturn_results_vllm_ppd_2p2d_c8.json
```

### SGLang 2P2D
```bash
conda activate sglang
bash scripts/sglang/start_2P_2D.sh
python benchmark/sglang_BFCL_v3_multi_turn_base.py
# 결과: results/sglang_hicache/bfcl_multiturn_results_2P_2D.json
```

### vLLM 4-GPU TP=4
```bash
conda activate vllm-source
vllm serve /home/uhmturks/hf_models/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 4 \
  --served-model-name meta-llama/Llama-3.1-8B-Instruct-FC \
  --enable-auto-tool-choice --tool-call-parser llama3_json \
  --max-model-len 131072
python benchmark/vllm_4gpu_BFCL_v3_multi_turn_base.py
```

---

## 결과 파일 구조

```json
{
  "summary": {
    "config": "vllm_ppd_2p2d_c8",
    "model": "...",
    "concurrency": 8,
    "total_items": 200,
    "success_items": 200,
    "error_items": 0,
    "total_output_tokens": 9432,
    "total_wall_time_s": 147.97,
    "overall_throughput_tok_per_s": 63.69,
    "avg_ttft_s": 0.8715,
    "avg_tpot_s": 0.0723,
    "avg_throughput_tok_per_s": 86.4,
    "kv_cache_per_gpu": {
      "p1": {"min": 0.0, "max": 0.031, "mean": 0.008, "samples": 72},
      "d1": {"min": 0.0, "max": 0.172, "mean": 0.059, "samples": 72}
    }
  },
  "results": [
    {
      "id": "multi_turn_base_0",
      "num_turns": 4,
      "avg_ttft_s": 0.83,
      "avg_tpot_s": 0.067,
      "ttft_by_turn": {"0": 0.65, "1": 0.78, "2": 0.92, "3": 1.01},
      "turns": [
        {
          "turn": 0,
          "ttft_s": 0.65, "tpot_s": 0.067,
          "output_tokens": 18,
          "context_chars": 4821,
          "tool_calls": [{"function": {"name": "mv", "arguments": "{...}"}}]
        }
      ]
    }
  ]
}
```

`ttft_by_turn`: turn 인덱스별 TTFT → context가 길어질수록 TTFT 증가하는 agentic 패턴 관찰 가능.
