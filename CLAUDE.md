# SGLang PD Disaggregation Experiment

## Overview

SGLang의 Prefill-Decode (PD) disaggregation 성능을 BFCL v3 multi-turn 벤치마크로 측정하는 실험.
측정 지표: TTFT, TPOT, Throughput (tok/s)

## Environment

- **Server**: server17
- **GPUs**: NVIDIA RTX A6000 x4 (49GB VRAM each)
- **RAM**: 125GB (주의: hicache가 CPU RAM 사용)
- **Conda env**: `sglang`
- **Model**: `Llama-3.1-8B-Instruct`
- **Model path**: `/home/uhmturks/hf_models/Llama-3.1-8B-Instruct`

## Directory Structure

```
~/experiments/
├── benchmark/
│   └── BFCL_v3_multi_turn_base_sglang.py   # 메인 벤치마크 스크립트
├── data/
│   ├── BFCL_v3_multi_turn_base.json         # 벤치마크 데이터 (200개)
│   └── multi_turn_func_doc/                 # Tool function definitions
│       ├── gorilla_file_system.json
│       ├── math_api.json
│       ├── message_api.json
│       ├── posting_api.json      # TwitterAPI
│       ├── ticket_api.json
│       ├── trading_bot.json
│       ├── travel_booking.json   # TravelAPI
│       └── vehicle_control.json  # VehicleControlAPI
├── gorilla/                                  # 공식 BFCL evaluator repo
├── logs/                                     # 서버 로그 (p1/p2/d1/d2/mooncake/router)
├── possible_answer/
│   └── BFCL_v3_multi_turn_base.json         # Ground truth
├── results/
│   └── bfcl_multiturn_results_1P_1D.json    # 1P1D 실험 결과
└── scripts/
    ├── start_2P_2D.sh    # 2P2D 서버 시작
    ├── stop.sh           # 전체 서버 종료
    └── cleanup_all.sh    # 로그/캐시 정리
```

## Configurations

### 1P1D (완료)
- Prefill: GPU 0, port 30000
- Decode: GPU 1, port 30001
- Router: port 8000
- 결과: `results/bfcl_multiturn_results_1P_1D.json`

### 2P2D (진행 중)
- Prefill 1: GPU 0, port 30000
- Prefill 2: GPU 1, port 30001
- Decode 1: GPU 2, port 30002
- Decode 2: GPU 3, port 30003
- Router: port 8000

## How to Run

### 서버 시작
```bash
cd ~/experiments
./scripts/start_2P_2D.sh
```

스크립트가 자동으로:
1. Mooncake metadata server 시작 (port 8080)
2. Prefill/Decode 서버 4개 시작
3. 모든 서버 ready 대기
4. Router 시작 (port 8000)

### 서버 상태 확인
```bash
grep "ready to roll" logs/*.log
nvidia-smi
free -h
```

### 벤치마크 실행
```bash
cd ~/experiments
python benchmark/BFCL_v3_multi_turn_base_sglang.py
```

결과는 `results/` 에 저장됨. 파일명에 config 명시 권장:
```
results/bfcl_multiturn_results_2P_2D.json
```

### 서버 종료
```bash
./scripts/stop.sh
```

## Known Issues & Notes

### func_doc 타입 변환 필요
BFCL func doc의 파라미터 타입이 Python 스타일이라 OpenAI schema로 변환 필요.
벤치마크 스크립트 내 `dict_to_object()` + `TYPE_MAP`이 처리함:
- `dict` → `object`
- `float` → `number`
- `list` / `tuple` → `array`
- `str` → `string`
- `bool` → `boolean`

### RAM 제약
hicache는 KV cache를 CPU RAM으로 offload함.
현재 서버 RAM 여유가 적어서 `--hicache-ratio`를 낮게 설정:
- Prefill: `--hicache-ratio 1.2` (host > device 조건 충족 필요)
- Decode: hicache 미사용

### involved_classes → func_doc 매핑
```python
CLASS_TO_FILE = {
    "GorillaFileSystem": "multi_turn_func_doc/gorilla_file_system.json",
    "TicketAPI":         "multi_turn_func_doc/ticket_api.json",
    "MessageAPI":        "multi_turn_func_doc/message_api.json",
    "MathAPI":           "multi_turn_func_doc/math_api.json",
    "TradingBot":        "multi_turn_func_doc/trading_bot.json",
    "TwitterAPI":        "multi_turn_func_doc/posting_api.json",
    "TravelAPI":         "multi_turn_func_doc/travel_booking.json",
    "VehicleControlAPI": "multi_turn_func_doc/vehicle_control.json",
}
```

### Mooncake KVTransferError
prefill→decode KV 전송 실패 시 `KVTransferError: Aborted by AbortReq` 발생.
`MOONCAKE_MASTER_SERVER=127.0.0.1:8080` 환경변수 설정 확인.

## Results Summary

| Config | TTFT (avg) | TPOT (avg) | Throughput | Errors |
|--------|-----------|------------|------------|--------|
| 1P1D   | 0.395s    | 0.0227s    | 43.95 tok/s | 162/200 |
| 2P2D   | -         | -          | -          | -      |

> 에러 162개는 func_doc `float` 타입 문제 → TYPE_MAP 수정 후 재실행 필요

## Benchmark Script Key Parameters

```python
ROUTER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS = 512
TIMEOUT = 120  # seconds per request
STREAM = True  # TTFT 측정을 위해 필수
```

## Metrics Definition

- **TTFT** (Time to First Token): `t_first_token - t_request`
- **TPOT** (Time Per Output Token): `(t_last - t_first) / (output_tokens - 1)`
- **Per-request Throughput**: `(output_tokens - 1) / decode_time` (tok/s)
- **Overall Throughput**: `total_output_tokens / total_wall_time` (tok/s)