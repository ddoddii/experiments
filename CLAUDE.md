# SGLang PD Disaggregation Experiment

## Overview

SGLang의 Prefill-Decode (PD) disaggregation 성능을 BFCL v3 multi-turn 벤치마크로 측정하는 실험.
측정 지표: TTFT, TPOT, Throughput (tok/s)

## Environment

두 개의 실행 환경이 있고, **실행 방식이 다르다**.

### server17 (전용 머신, bash 직접 실행)

- **GPUs**: NVIDIA RTX A6000 x4 (49GB VRAM each)
- **RAM**: 125GB (주의: hicache가 CPU RAM 사용)
- **Conda env**: `sglang`
- **Model**: `Llama-3.1-8B-Instruct`
- **Model path**: `/home/uhmturks/hf_models/Llama-3.1-8B-Instruct`

### A100 클러스터 (SLURM, sbatch 필수)

bash 에서 직접 서버를 띄우지 않는다. `.sh` 로 만들어 `sbatch` 로 올린다.

- **Partition / QOS**: `-p suma_a100 -q a100_qos`
- **GPU**: 2장 → **1P1D 전용** (2P2D 는 4장 필요)
- **실행**: `./scripts/slurm/submit.sh` (또는 `sbatch ~/experiments/scripts/slurm/submit_bench.sh`)
- **대화형**: `srun -p suma_a100 -q a100_qos --gres=gpu:2 --pty bash -i`
- **도는 job 에 붙기**: `srun --jobid=<jobid> --pty bash`
- **문서**: `scripts/slurm/README.md`

**수정한 sglang 소스를 반영하려면** 한 번만:

```bash
conda activate sglang
cd ~/sglang-source/python && pip install -e . --no-deps
```

이후로는 `.py` 를 고치고 서버만 재시작하면 된다 (`run_why_faster.sh` 는 arm 마다
재시작하므로 자동). 이걸 안 하면 `import sglang` 이 conda env 의 **설치본**을 집어서
park 패치 없는 upstream 이 도는데, 서버도 로그도 정상이라 증상이 "park arm 이 baseline
과 비슷하다" 뿐이다. `preflight.sh` 와 `_use_source.sh` 가 그 상태면 실험을 막는다.
`sgl-kernel/` (C++/CUDA) 을 고쳤다면 재빌드가 필요하다.

노드를 남과 공유하므로 포트/pkill/GPU번호/공유디렉터리가 전부 job 단위로 격리된다
(`scripts/slurm/env.sh`). 이 격리를 우회해서 서버를 손으로 띄우면 같은 노드의 다른
job 을 죽이거나, 벤치가 남의 라우터에 요청을 쏜다.

**A6000 에서 잰 상수를 그대로 들고 오면 안 된다.** peer/host 대역폭, 카드 크기,
prefill ms/token, `DECODE_MAX_TOTAL_TOKENS` 가 분석에 박혀 있다. `preflight.sh` 가
앞의 둘을 자동 갱신하고, `DECODE_MAX_TOTAL_TOKENS` 는 `MODE=probe` 가 계산해 준다.
prefill ms/token 은 아직 수동 (`benchmark/ttft_ctx_sweep.py`). 자세한 건 위 README.

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
    ├── sglang/
    │   ├── start_1P_1D.sh    # 1P1D 서버 시작 (SLURM job 격리 지원)
    │   ├── start_2P_2D.sh    # 2P2D 서버 시작 (포트 하드코딩 → 노드 독점 필요)
    │   ├── stop_1P_1D.sh     # 1P1D 종료 (job 소속 프로세스만)
    │   ├── stop.sh           # 2P2D 종료 (광역 pkill)
    │   ├── _job_scope.sh     # job 단위 pkill 헬퍼
    │   └── run_why_faster.sh # 메인 실험 러너 (arm 별 비교)
    └── slurm/                # A100 클러스터 전용
        ├── env.sh            # 공통 환경 + job 단위 포트/GPU/디렉터리
        ├── preflight.sh      # 무효한 결과를 만들 조건 사전 차단
        ├── submit_bench.sh   # sbatch 엔트리 (probe / full)
        ├── submit.sh         # sbatch 래퍼
        └── interactive.sh    # srun --pty 헬퍼
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

### A100 (SLURM) — sbatch

```bash
cd ~/experiments
./scripts/slurm/submit.sh                                        # 1) probe
./scripts/slurm/submit.sh MODE=full DECODE_MAX_TOTAL_TOKENS=<값>  # 2) 본 실험

squeue -u $USER
tail -f logs/slurm/sglang-why-<jobid>.out
srun --jobid=<jobid> --pty bash     # 도는 job 안으로
```

probe 를 먼저 돌리는 이유: arm 마다 decode KV pool 크기가 달라지는데
(park pool 이 decode GPU 의 메모리를 먼저 먹는다), 그걸 고정하지 않으면
캐시 계층 차이와 용량 삭감이 한 숫자에 섞인다. 자세한 건 `scripts/slurm/README.md`.

### server17 (전용 머신) — bash 직접

```bash
cd ~/experiments
./scripts/sglang/start_2P_2D.sh
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