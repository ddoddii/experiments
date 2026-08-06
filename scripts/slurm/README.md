# A100 (SLURM) 실행

server17 (A6000 x4 전용 머신) 에서 A100 클러스터로 옮기면서 달라지는 건 GPU 세대만이
아니다. **노드를 남과 공유한다**는 게 더 크고, 기존 스크립트는 전부 전용 머신을 가정하고
있었다. 이 디렉터리가 그 간극을 메운다.

## 빠른 사용법

```bash
# 1. probe: 이 카드에서 arm 별 decode KV pool 이 얼마나 잡히는지 먼저 잰다
./scripts/slurm/submit.sh

# 2. probe 로그가 알려준 값으로 본 실험
./scripts/slurm/submit.sh MODE=full DECODE_MAX_TOTAL_TOKENS=<probe가 알려준 값>

# 상태 확인
squeue -u $USER
tail -f ~/experiments/logs/slurm/sglang-why-<jobid>.out

# 도는 job 안으로 들어가기
srun --jobid=<jobid> --pty bash
```

`sbatch` 를 직접 쓰고 싶으면:

```bash
sbatch ~/experiments/scripts/slurm/submit_bench.sh
sbatch --export=ALL,MODE=full,WORKLOAD=bfcl,REPEATS=3 \
       ~/experiments/scripts/slurm/submit_bench.sh
```

`submit.sh` 래퍼를 권하는 이유는 두 가지뿐이다: `logs/slurm` 디렉터리를 미리 만들고
(없으면 job 이 출력도 못 남기고 죽는다), `EXP_ROOT` 를 실제 리포 경로로 넘긴다
(sbatch 는 스크립트를 노드의 spool 로 **복사**해 실행하므로, batch job 안에서
`${BASH_SOURCE[0]}` 로 리포를 찾는 흔한 패턴이 통하지 않는다).

디버깅용 대화형 셸:

```bash
./scripts/slurm/interactive.sh          # = srun -p suma_a100 -q a100_qos --gres=gpu:2 --pty bash -i
```

## 파일

| 파일 | 역할 |
|---|---|
| `env.sh` | 모든 스크립트가 source 하는 공통 환경. conda, sglang 소스, **job 단위 포트/GPU/디렉터리** |
| `preflight.sh` | 결과를 무효로 만들 조건을 먼저 잡아낸다. 실패하면 실험을 시작하지 않는다 |
| `submit_bench.sh` | `#SBATCH` 지시어를 가진 배치 엔트리. probe / full 두 모드 |
| `submit.sh` | 위를 올리는 래퍼 (디렉터리 생성 + `EXP_ROOT` 전달) |
| `interactive.sh` | `srun --pty bash` 헬퍼 |

## 공유 노드에서 무엇이 달라지는가

전용 머신을 가정한 코드가 클러스터에서 만드는 실패는 전부 **조용하다**. 서버는 뜨고,
벤치는 돌고, 숫자는 그럴듯한데 측정 대상이 아니다. 네 가지를 job 단위로 격리했다.

**포트.** `30000/30001/8000/8998/8080` 이 하드코딩돼 있었다. 같은 노드에 두 job 이
뜨면 늦은 쪽이 bind 에서 죽거나 — 더 나쁘게는 — 벤치가 **남의 라우터**에 요청을 쏜다.
이제 `env.sh` 가 job id 에서 64포트 블록을 뽑는다 (`PORT_BASE`, 10240–29376).
32768 미만에 두는 건 의도한 것이다: 리눅스 기본 ephemeral 범위가 32768–60999 이라
그 위의 고정 포트는 아무 클라이언트 소켓에게나 선점당할 수 있다.

**pkill.** start/stop 스크립트가 `pkill -f sglang.launch_server` 를 했다. 내 job 두 개가
같은 노드에 뜨면, 나중에 시작한 job 의 정리 단계가 **몇 시간째 arm 을 돌던 앞 job 을
죽인다**. `_job_scope.sh` 의 `job_pkill` 은 `/proc/<pid>/environ` 의 `SLURM_JOB_ID` 로
내 job 소속만 골라낸다. SLURM 밖에서는 예전 그대로 동작한다.

**GPU 번호.** `PREFILL_GPU=0 DECODE_GPU=1` 이 하드코딩돼 있었다. cgroup 격리가 없는
사이트에서 SLURM 이 물리 GPU 4,5 를 줬다면 0,1 은 **남의 GPU** 다. 이제 논리 인덱스를
`CUDA_VISIBLE_DEVICES` 할당 목록을 통해 물리 번호로 매핑한다 (격리 O/X 양쪽에서 맞다).

**공유 경로.** `/dev/shm/sglang_kv_parking` 과 `/tmp/hicache` 를 job 이 공유했다.
앞 job 의 park telemetry 를 뒤 job 의 sampler 가 살아있는 점유율로 읽는다.
둘 다 `_${JOB_TAG}` 접미사를 붙였다.

## A100 으로 옮기면서 다시 재야 하는 상수

분석 스크립트에 A6000 값이 **박혀 있고**, 이것들은 장식이 아니라 결론을 만든다.

| 상수 | A6000 값 | 왜 중요한가 | 어떻게 갱신되나 |
|---|---|---|---|
| peer / host 대역폭 | 52.7 / 26.3 GB/s | "link budget" — 관측된 차이 중 매체가 설명할 수 있는 최대치 | `preflight.sh` 가 `hwprofile.py` 를 자동 실행 |
| 카드 크기 | 48 GB | park pool 크기, 메모리 예약 | 같음 |
| prefill ms/token | 0.132 | **캐시 히트 한 번의 가치.** A100 이 더 빨리 prefill 하므로 같은 히트율 상승이 더 적은 wall-clock 으로 돌아온다 | 서버가 떠야 잴 수 있다. `benchmark/ttft_ctx_sweep.py` 로 잰 뒤 `hwprofile.py --prefill-ms-per-tok` 로 넣어라 (**아직 수동**) |
| `DECODE_MAX_TOTAL_TOKENS` | 154304 | arm 간 decode 용량을 맞추는 대조군 | `MODE=probe` 가 계산해서 알려준다 |

`DECODE_MAX_TOTAL_TOKENS` 에 기본값이 없는 건 실수가 아니다. A6000 에서 잰 154304 를
A100 에 박으면 그 카드에서 아무 의미도 없는 값으로 decode 를 조이는 것 —
**대조군으로 위장한 오설정**이다. 그래서 `MODE=probe` 를 먼저 돌린다.

`MODE=probe` 는 각 arm 을 짧게 띄워 `max_total_num_tokens` 를 읽고, **작은 쪽**을
쓰라고 알려준다. 이걸 고정하지 않으면 park arm 의 decode pool 이 park pool 크기만큼
줄어든 채로 hicache 와 비교된다 — 캐시 계층 차이와 용량 삭감이 한 숫자에 섞인다.
(A6000 에서 실제로 216,374 → 154,304, 29% 삭감이었고, 카드의 총 HBM 은 같아서
오래 숨었다.)

## preflight 가 잡는 것

`preflight.sh` 는 환경 점검이 아니라 **무효한 결과 방지**용이다. 하드 실패면 실험을
시작하지 않는다.

- **peer access (하드 실패).** `cudaDeviceCanAccessPeer` 가 false 면 PyTorch 는 실패하지
  않고 복사를 **host 경유로 몰래 바꾼다**. `park_gpu` arm 이 사실상 `park_host` 가 되고,
  로그도 telemetry 도 여전히 "peer GPU hit" 이라고 말한다. 두 arm 의 차이가 사라진
  결과를 몇 시간 걸려 만드느니 여기서 죽는 게 낫다.
- **P/D 링크 종류 (경고).** A100 SXM (NVSwitch) 이면 모든 쌍이 `NV12`, PCIe A100 이면
  어느 쌍도 아니다. 후자에서도 "idle decode HBM 에 park" 라는 주장은 유효하지만 매체가
  PCIe 가 되므로, `nvidia_smi_topo.txt` 와 함께 기록하고 결과에 명시해야 한다.
- **MIG (하드 실패, `env.sh`).** parking 은 peer GPU 간 CUDA IPC 를 쓰는데 MIG 는 그걸
  막는다. `--gres=gpu:2` 로 온전한 카드를 받아라.
- 모델 경로, sglang 이 설치본이 아니라 소스 트리에서 import 되는지, mooncake 메타데이터
  서버가 `--port` 를 받는지, host RAM 이 `HICACHE_RATIO` 를 감당하는지, 포트 블록이 비었는지.

## 2P2D 는 아직 노드 독점이 필요하다

1P1D 는 완전히 job 격리된다. 2P2D 는 아니다 — `start_2P_2D.sh` 가 포트 30000–30003 을
하드코딩하고 (ready 대기 로직이 포트 번호로 로그 파일명을 고르는 `case` 문까지 포함),
`PD_LAYOUT` 으로 GPU 번호를 스스로 다시 계산해서 `env.sh` 가 넘긴 물리 번호를 덮어쓴다.

그래서 `A100_TOPOLOGY=2p2d` 는 할당 GPU 가 `0,1,2,3` 이 아니면 **거부한다**. 쓰려면:

```bash
GPUS=4 ./scripts/slurm/submit.sh MODE=full A100_TOPOLOGY=2p2d   # + sbatch --exclusive
```

애초에 2GPU 1P1D 가 이 주장을 가장 깨끗하게 보여주는 구성이다: park 대상이 NVLink 로
붙은 decode GPU 하나뿐이라, A6000 4장 실험을 괴롭혔던 혼동 — 각 prefill 의 후보 목록에
NVLink 대상과 PCIe 대상이 섞여 있고 둘 다 "peer" 로 보고되던 문제 — 이 사라진다.
