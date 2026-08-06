# A100 (SLURM) 실행

server17 (A6000 x4 전용 머신) 에서 A100 클러스터로 옮기면서 달라지는 건 GPU 세대만이
아니다. **노드를 남과 공유한다**는 게 더 크고, 기존 스크립트는 전부 전용 머신을 가정하고
있었다. 이 디렉터리가 그 간극을 메운다.

**GPU 2장 = 1P1D 전용.** 여기서는 그게 제약이 아니라 오히려 이 주장을 보여주기에 가장
깨끗한 구성이다. park 대상이 decode GPU 하나뿐이라, A6000 4장 실험을 괴롭혔던 혼동 —
각 prefill 의 후보 목록에 NVLink 대상과 PCIe 대상이 섞여 있고 둘 다 "peer" 로 보고되던
문제 — 이 구조적으로 사라진다. 아래 2P2D 관련 내용은 4장을 받게 될 때를 위한 것이다.

## 빠른 사용법

```bash
# 0. 클러스터 체크아웃을 먼저 최신으로. 이걸 빼먹으면 "고쳤는데 똑같은 에러" 가 난다
git pull origin claude/a100-experiment-setup-dtubqr

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

## 수정한 sglang 소스가 실제로 도는가

`~/sglang-source` 를 클론해 두는 것만으로는 반영되지 않는다. `import sglang` 은 conda
env 에 pip 로 설치된 복사본을 집고, 서버는 정상적으로 뜨고, 로그도 정상이고, 다만
**park 패치가 없는 upstream 이 돈다.** 증상은 "park arm 이 baseline 과 비슷하다" 하나뿐이다.

**한 번만 해두면 되는 것 (권장):**

```bash
conda activate sglang
cd ~/sglang-source/python
pip install -e . --no-deps
```

`--no-deps` 를 빼면 pip 가 torch/flashinfer 를 다시 해결하려 들면서 멀쩡한 env 를
깨뜨릴 수 있다. 이걸 해두면 이후로는 **`.py` 를 고치고 서버만 재시작하면 끝**이다.
(`env.sh` 가 `PYTHONPATH` 에도 소스 트리를 얹으므로 editable 설치 없이도 동작하지만,
editable 쪽이 도구들이 일관되게 같은 트리를 보게 해서 낫다.)

### `--no-build-isolation` 을 붙이면 안 된다

이 리포의 예전 안내(`_use_source.sh`, `which_sglang.sh`)는 `--no-build-isolation` 을
같이 쓰라고 했는데, 지금 sglang 트리에서는 그게 **실패한다**:

```
error: Failed to import grpc_tools: No module named 'grpc_tools'.
error: metadata-generation-failed
```

`python/pyproject.toml` 의 `build-system.requires` 에 `grpcio-tools==1.75.1` 가 있고,
`setup.py` 의 `egg_info` 훅이 `sglang_scheduler.proto` 에서 `*_pb2.py` 를 생성한다.
`--no-build-isolation` 은 바로 그 빌드 환경 구성을 끄는 플래그라, grpcio-tools 가
설치되지 않은 채로 생성 단계가 돌아 죽는다.

**`--no-deps` 와 `--no-build-isolation` 은 서로 다른 것을 끈다.** 앞은 *런타임* 의존성
재해결(torch/flashinfer — 이게 env 를 깨뜨린다), 뒤는 *빌드* 요구사항이다. 우리가 피하고
싶은 건 앞쪽뿐이므로, 빌드 격리는 켜둔 채로 두면 된다: pip 가 임시 환경에 grpcio-tools 만
받아서 proto 를 생성하고, conda env 의 런타임 패키지는 건드리지 않는다.

PyPI 접근이 안 되는 노드라면 grpcio-tools 를 먼저 넣고 예전 방식으로:

```bash
pip install "grpcio-tools==1.75.1"
cd ~/sglang-source/python && pip install -e . --no-deps --no-build-isolation
```

**둘 다 안 되면 editable 설치를 건너뛰어도 된다.** `env.sh` 의 `PYTHONPATH` 만으로 충분하다.
생성되는 `*_pb2.py` 는 `.gitignore` 대상이라 클론에 원래 없고, import 하는 곳은
`srt/entrypoints/grpc_server.py` 와 `srt/grpc/health_servicer.py` 뿐이다 — 이 실험은
`sglang.launch_server` (HTTP) 만 쓰므로 그 경로를 아예 타지 않는다.

**반영되는 범위:**

| 고친 것 | 필요한 조치 |
|---|---|
| `python/sglang/**/*.py` (park 코드 전부 여기 있다) | 서버 재시작만. `run_why_faster.sh` 는 arm 마다 재시작하므로 자동 |
| `sgl-kernel/` (C++/CUDA) | 재빌드 필요. editable 설치도 PYTHONPATH 도 커널은 못 건드린다 |
| 도는 서버에 실시간 반영 | 불가능. 파이썬은 import 시점에 모듈을 읽는다 |

**확인 방법:** `preflight.sh` 가 계산 노드에서 실제 import 경로를 찍고, 설치본을 집고
있으면 **실험을 시작하지 않는다.** 손으로 볼 때는:

```bash
python -c "import sglang; print(sglang.__file__)"     # ~/sglang-source/... 여야 한다
./scripts/sglang/which_sglang.sh                       # 도는 서버가 뭘 실행 중인지까지
```

각 arm 의 `meta_<arm>.json` 에 그 arm 이 돈 sglang 커밋 SHA 가 찍힌다. 한 OUTDIR 에
서로 다른 빌드의 arm 이 섞이면 (한 arm 만 재실행했을 때 실제로 있었던 일) 여기서 보인다.

주의: 클론과 conda env 가 **계산 노드에서 보이는 파일시스템**에 있어야 한다. 홈이 NFS 로
공유되는 보통의 클러스터면 문제없고, 아니면 preflight 의 import 검사에서 걸린다.

## 필요한 패키지

conda env `sglang` 에 이 셋이 있어야 한다. `preflight.sh` 가 전부 확인한다.

```bash
conda activate sglang
pip install mooncake-transfer-engine==0.3.8.post1   # KV 전송 백엔드 (sglang CI 핀)
pip install sglang-router                          # PD 라우터 (모듈명 sglang_router)
cd ~/sglang-source/python && pip install -e . --no-deps   # 수정한 sglang 소스
```

`sglang-router` 는 sglang 리포에서 `sgl-model-gateway/` 로 옮겨갔지만 pip 패키지
이름과 모듈 이름은 그대로다. 이게 없으면 prefill/decode 서버가 **둘 다 뜬 다음**
`[5/5]` 단계에서 죽는다 — 모델 로딩에 쓴 시간이 통째로 날아간다.

## mooncake 가 안 뜰 때

PD disaggregation 의 KV 전송 백엔드다. 없으면 서버가 아예 안 뜬다.

```bash
conda activate sglang
pip install mooncake-transfer-engine==0.3.8.post1   # sglang CI 가 쓰는 핀
```

설치했는데도 `p1.log` 가 이렇게 죽는 경우가 있다:

```
ImportError: /lib64/libldap.so.2: undefined symbol: EVP_md2, version OPENSSL_3.0.0
...
ImportError: Please install mooncake by following the instructions at ...
[..] Received sigquit from a child process.
```

**두 번째 메시지는 무시하라. 진짜 원인은 첫 번째다.** sglang 의 `transfer_engine.py`
가 `ImportError` 를 통째로 "설치하세요" 메시지로 갈아끼우기 때문에, 이미 설치돼 있는데도
설치하라는 말이 나온다.

실제 원인은 OpenSSL 이 섞인 것이다. 시스템 `libldap` 은 `EVP_md2` 를 요구하는데,
먼저 로드된 conda-forge `libcrypto` 는 MD2 를 빼고 빌드돼 있어 그 심볼이 없다.
체인을 어느 한쪽으로 통일하면 풀린다:

| `MOONCAKE_LD_FIX` | 하는 일 |
|---|---|
| `system` | 시스템 `libcrypto.so.3` 을 `LD_PRELOAD` 해서 `EVP_md2` 를 제공한다 |
| `conda` | `LD_LIBRARY_PATH` 에 conda 의 `lib` 을 먼저 놓는다 (libcurl/libldap 까지 conda 것으로) |
| `none` (기본) | 아무것도 안 한다 |

`system` 을 먼저 시도한다. `conda` 쪽은 부작용이 크다 — 이 클러스터에서 실측하면
mooncake 는 넘어가도 CUDA 런타임 해석이 깨진다 (`ImportError: libcudart.so.12`).
torch 가 번들 CUDA 라이브러리를 RPATH 로 찾는데 `conda/lib` 이 앞에서 가로채기
때문이다. `LD_PRELOAD` 는 심볼 하나를 채울 뿐이라 훨씬 국소적이다.

**어느 쪽인지는 preflight 가 실제로 돌려보고 알려준다.** 둘 다 시도해서 되는 쪽의
이름을 출력하므로, 그대로 넘기면 된다:

```bash
./scripts/slurm/submit.sh MOONCAKE_LD_FIX=conda     # preflight 가 알려준 값
```

기본값이 `none` 인 건 의도한 것이다. 멀쩡히 도는 환경에서 라이브러리 해석 순서를
말없이 바꾸는 건 그 자체로 새 실패를 만든다. 지금 이 노드에서 확인하려면:

```bash
python -c 'from mooncake.engine import TransferEngine' && echo OK
LD_PRELOAD=/lib64/libcrypto.so.3 python -c 'from mooncake.engine import TransferEngine' && echo "system 으로 됨"
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python -c 'from mooncake.engine import TransferEngine' && echo "conda 로 됨"
```

셋 다 실패하면 체인을 직접 봐야 한다:

```bash
ldd $(python -c 'import mooncake,os;print(os.path.dirname(mooncake.__file__))')/engine*.so \
  | grep -E 'ldap|curl|crypto|ssl'
conda install -c conda-forge libcurl openldap    # 전부 conda 쪽으로 통일
```

`preflight.sh` 는 이제 메타데이터 서버 import 와 **엔진 로드를 따로** 확인한다. 앞의
것만 보던 동안에는 preflight 를 통과한 뒤 8분 뒤 `p1.log` 에서 터졌다 — 파이썬 모듈은
멀쩡히 import 되고 C++ 확장만 로드에 실패하는 상태였기 때문이다.

## RDMA 는 있는데 GPUDirect 가 없을 때 (이 클러스터가 그렇다)

엔진이 로드된 뒤 다음 단계에서 이렇게 죽는다:

```
Topology discovery complete. Found 2 HCAs.
installTransport, type=rdma
E rdma_context.cpp:265] Failed to register memory 0x...: Bad address [14]
```

`[14]` 은 `ibv_reg_mr` 의 EFAULT 다. node43 에서 측정한 결과:

| | 결과 |
|---|---|
| HOST 버퍼 등록 | `rc=0` OK |
| DEVICE 버퍼 등록 | `rc=-202` FAIL |
| `nvidia_peermem` | 로드되지 않음 |
| `ulimit -l` | unlimited |

host 는 되고 device 만 안 되는 건 **GPUDirect RDMA 부재**의 정확한 지문이다.
`nvidia_peermem` 로드는 root 권한이라 job 안에서는 못 한다.

**중요 (측정으로 확인):** `SGLANG_MOONCAKE_PROTOCOL=tcp` 만으로는 안 된다.
mooncake 0.3.8 은 `initialize()` 의 protocol 인자를 무시하고, **topology
auto-discovery 가 HCA 를 찾았는지**로 transport 를 정한다. 그래서 tcp 를 넘겨도
로그는 여전히 `installTransport, type=rdma` 다.

실제로 통하는 설정은 `--sweep` 이 찾는다:

```bash
python scripts/slurm/probe_mooncake.py --sweep
```

후보별로 transport 와 GPU 등록 결과를 표로 뽑고, 전체 출력을
`results/a100/mooncake_sweep/` 에 저장한다. node43 에서 통한 것:

| 설정 | 비고 |
|---|---|
| `MC_FORCE_TCP=1` | **권장.** 명시적이고 mooncake 환경변수라 서버로 그대로 상속된다 |
| `IB_DEVICE=mlx5_99` | 없는 장치를 지정해 HCA 를 0개로 만드는 우회. 로그에 가짜 장치명이 남는다 |

```bash
./scripts/slurm/submit.sh MOONCAKE_LD_FIX=system MC_FORCE_TCP=1
```

`env_summary` 가 `MC_*` 변수를 전부 찍는다. **RDMA 로 돈 run 과 TCP 로 돈 run 은
비교하면 안 되므로** 어느 쪽이었는지가 결과에 남아야 한다.

등록이 성공했다는 것만으로 RDMA 를 안 쓴다고 단정하지는 말 것 —
`mooncake_sweep/` 의 저장된 출력에서 `installTransport` 줄이 없거나 `tcp` 인지
직접 확인하는 게 맞다.

## 결과 디렉터리 이름

```
results/a100/a100_<workload>_c<concurrency>/          # 본 실험
results/a100/a100_<workload>_c<concurrency>_probe/    # probe
results/a100/... .prev/                               # 직전 run
```

job id 는 이름에 넣지 않는다 — 제출할 때마다 새 디렉터리가 생겨 쌓인다. 다만 job id 는
장식이 아니라 **출처 정보**였으므로 (어느 job/노드/커밋/전송설정에서 나온 숫자인지),
이름에서 뺀 대신 디렉터리 안 두 파일에 남긴다:

| 파일 | 내용 |
|---|---|
| `RUN_INFO.txt` | job id, 노드, 시각, experiments/sglang 커밋, `MC_*` 환경변수, decode pool 고정값 |
| `preflight.json` | job, 노드, GPU, 카드명, 측정 peer 대역폭, P/D 링크 종류 |

**같은 이름을 재사용하면 새 위험이 생긴다**: 이번 run 에서 실패한 arm 자리에 지난 run 의
파일이 남아, 분석 스크립트가 그걸 이번 결과로 읽는다. 이 리포는 이미 그 함정에 빠진 적이
있다 (arm 하나만 재실행했더니 나머지는 옛 빌드 것이 남아 있었다). 그래서 run 시작 시
직전 디렉터리를 `.prev` 로 **옮긴다** (지우지 않는다). 설정당 최대 두 세대만 남는다.
`KEEP_OUTDIR=1` 이면 그냥 덮어쓴다.

`CONCURRENCY` 는 이름에 남긴다. c4 probe 와 c32 본 실험은 비교 대상이 아닌데 이름이
같으면 뒤엣것이 앞엣것을 덮어쓴다. 완전히 고정하려면 `TAG=...` 로 직접 넘겨라.

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

## 왜 hicache 가 압박 아래서 무너지는가 (측정된 메커니즘)

C=64, prefill pool 79,000 (99% 점유) 에서 실측:

| arm | reuse | 서버측 TTFT | 완주 |
|---|---|---|---|
| recompute | 0% | 32.6s | 200/200 |
| radix | 7.6% | 37.7s | 200/200 |
| hicache | 6.1% | **112.3s** | 44/200 (156개 120초 타임아웃) |

**radix 와 hicache 의 reuse 가 거의 같은데(7.6% vs 6.1%) TTFT 는 3배 차이난다.**
재사용량이 같으므로 그 3배는 캐시의 이득이 아니라 hicache 계층의 비용이다.

원인은 대역폭이 아니다. hicache 가 쓴 양은 ~1M 토큰 × 128 KiB ≈ 131 GB 이고 480초에
나눠도 273 MB/s 라 PCIe 근처도 못 간다. 진짜 원인은 `hiradix_cache.py` 에 있다:

```python
write_through_threshold = 1 if policy == "write_through" else 2   # 기본이 write_through
...
self.ongoing_write_through[node.id] = node
if not write_back:
    self.inc_lock_ref(node)      # <- 쓰는 동안 노드를 잠근다
```

기본 정책에서 **모든 노드가 첫 접근에 host 로 복사되고, 복사가 끝날 때까지 잠긴다.**
잠긴 노드는 축출 대상에서 빠진다. 풀이 99% 인 상황에서는:

1. 새 prefix 마다 host 로 write-through 가 걸리고
2. 그동안 그 노드는 축출 불가
3. 그런데 풀이 꽉 차서 할당하려면 축출해야 함
4. → 축출 가능한 집합이 in-flight write 만큼 줄어들고, 할당이 write 완료를 기다린다

즉 **write-through 가 자기가 비우려는 풀의 페이지를 붙잡는다.** radix 는 쓸 게 없으니
즉시 축출하고, 그래서 같은 재사용률에 3배 빠르다.

이게 GPU park 이 유리한 이유의 정확한 형태다. 락을 잡고 있는 시간이 곧 복사 시간이고,
peer GPU 270 GB/s vs host PCIe ~25 GB/s 이므로 **락 보유 시간이 ~10배 짧다.**
1,000 토큰 노드 기준 5.1 ms → 0.47 ms. 이득의 정체는 "매체가 빨라서 fetch 가 빠르다"
보다 "축출 경로가 막히지 않는다" 쪽이다.

**단, 압박이 과하면 아무도 못 이긴다.** 위 run 은 W(318k) / P(79k) = 4배
oversubscribe 라 reuse 가 6~7% 로 붕괴했다 -- 재사용 전에 전부 쫓겨난다. victim cache
가 값을 하려면 1.5배 근처여야 한다 (`check_pressure.py --plan-concurrency`).
