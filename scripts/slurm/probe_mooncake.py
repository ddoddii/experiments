#!/usr/bin/env python3
"""
mooncake 의 RDMA 메모리 등록이 왜 실패하는지 판별한다. 추측하지 않고 가른다.

증상 (p1.log):

    Topology discovery complete. Found 2 HCAs.
    installTransport, type=rdma
    E rdma_context.cpp:265] Failed to register memory 0x7f3160000000: Bad address [14]

[14] 은 ibv_reg_mr 의 EFAULT 다. 후보가 셋이고 대응이 전부 다르므로, 어느 것인지
먼저 확정해야 한다. 이 스크립트가 그걸 가른다.

  A. GPUDirect RDMA 없음      device 포인터를 등록할 때만 EFAULT. nvidia_peermem
                              커널 모듈이 없으면 이렇게 된다. 로드는 root 권한이 필요.
  B. locked memory 한도       보통 ENOMEM[12] 이지만 환경에 따라 다르게 나온다.
                              ulimit -l 로 확인.
  C. 그 외 (주소/정렬 문제)    host 등록까지 실패하면 이쪽.

판별 방법은 단순하다: HOST 버퍼와 DEVICE 버퍼를 각각 등록해 본다.

    host OK  + device FAIL  -> A (GPUDirect RDMA 부재). 가장 흔하다.
    host FAIL + device FAIL -> B 또는 C
    둘 다 OK                -> 등록 자체는 되는데 sglang 이 넘기는 특정 버퍼가 문제

왜 중요한가: 이 실험은 단일 노드 1P1D 이고 P->D 전송이 같은 호스트 안에서 일어난다.
RDMA 가 주는 게 없다 -- server17 의 A6000 결과도 전부 mooncake 의 TCP 폴백으로
측정됐다. 그러니 A 로 확정되면 "RDMA 를 고친다"가 아니라 "RDMA 를 쓰지 않게 한다"가
맞는 방향이고, 그건 mooncake 가 HCA 를 발견하지 못하게 만드는 문제가 된다
(sglang 은 protocol 을 "rdma" 로 하드코딩하므로 sglang 쪽 스위치는 없다).

사용:
    python scripts/slurm/probe_mooncake.py
    MC_ENV="MC_FORCE_MNNVL=True" python scripts/slurm/probe_mooncake.py   # 후보 검증
"""
import os
import subprocess
import sys

# ─── sweep 모드 ───────────────────────────────────────────────────────────────
# 측정된 사실: protocol='tcp' 를 넘겨도 mooncake 0.3.8 은 무시한다. 로그가
# "installTransport, type=rdma" 로 나오고 GPU 등록은 여전히 EFAULT 다. 즉 전송
# 방식은 initialize() 인자가 아니라 "topology auto-discovery 가 HCA 를 찾았는가"로
# 정해진다. 그러니 HCA 를 못 찾게 만들어야 하고, 그 방법은 mooncake 내부 환경변수라
# 문서가 없다. 이름을 추측해서 job 을 던지면 라운드마다 몇 분씩 날아가므로, 후보를
# 서브프로세스로 한 번에 돌려서 어느 것이 transport 를 tcp 로 바꾸는지 표로 뽑는다.
#
#   python scripts/slurm/probe_mooncake.py --sweep
SWEEP_CANDIDATES = [
    ("(baseline)", {}),
    # sglang 의 공식 플래그 경로: --disaggregation-ib-device 가 device_name 이 된다.
    # 로그상 mlx5_1 은 "port not active" 라 mooncake 가 스스로 disable 한다. 그것만
    # 지정하면 쓸 수 있는 장치가 0개가 된다.
    ("ib_device=mlx5_1(inactive)", {"PROBE_IB_DEVICE": "mlx5_1"}),
    ("ib_device=nonexistent", {"PROBE_IB_DEVICE": "mlx5_99"}),
    # mooncake 의 장치 필터 후보들. 이름이 문서화돼 있지 않아 전부 시도한다.
    ("MC_FILTERS=[]", {"MC_FILTERS": "[]"}),
    ("MC_FILTERS=[nomatch]", {"MC_FILTERS": '["nomatch"]'}),
    ("MC_MS_FILTERS=[]", {"MC_MS_FILTERS": "[]"}),
    ("MC_DISABLE_RDMA=1", {"MC_DISABLE_RDMA": "1"}),
    ("MC_USE_TCP=1", {"MC_USE_TCP": "1"}),
    ("MC_FORCE_TCP=1", {"MC_FORCE_TCP": "1"}),
    ("MC_TRANSPORT=tcp", {"MC_TRANSPORT": "tcp"}),
    ("MC_PROTOCOL=tcp", {"MC_PROTOCOL": "tcp"}),
    ("MC_TOPO_JSON={}", {"MC_CUSTOM_TOPO_JSON": "{}"}),
]


def run_sweep():
    print("mooncake transport sweep -- 목표: installTransport 를 tcp 로 바꾸고")
    print("GPU 메모리 등록을 성공시키는 설정 찾기.\n")
    print(f"  {'후보':<28} {'transport':<10} {'device reg':<12} rc")
    print("  " + "-" * 62)
    winners = []
    for label, extra in SWEEP_CANDIDATES:
        env = dict(os.environ)
        env.update(extra)
        env["PROBE_ONE"] = "1"
        env.pop("PROBE_PROTOCOL", None)
        try:
            p = subprocess.run(
                [sys.executable, __file__],
                env=env, capture_output=True, text=True, timeout=180,
            )
            out = p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            print(f"  {label:<28} {'<timeout>':<10}")
            continue
        transport = "?"
        for line in out.splitlines():
            if "installTransport, type=" in line:
                transport = line.split("installTransport, type=")[-1].strip()
        devreg, rc = "?", p.returncode
        for line in out.splitlines():
            if line.strip().startswith("DEVICE"):
                devreg = "OK" if " rc=0 " in line else "FAIL"
        mark = ""
        if devreg == "OK":
            winners.append((label, extra))
            mark = "  <<< 이걸 쓰면 된다"
        print(f"  {label:<28} {transport:<10} {devreg:<12} {rc}{mark}")

    print()
    if winners:
        label, extra = winners[0]
        print(f"  통하는 설정: {label}")
        kv = " ".join(f"{k}={v}" for k, v in extra.items())
        if "PROBE_IB_DEVICE" in extra:
            print("  sglang 서버에는 --disaggregation-ib-device 로 넘어간다:")
            print(f"    ./scripts/slurm/submit.sh IB_DEVICE={extra['PROBE_IB_DEVICE']}")
        else:
            print("  mooncake 환경변수이므로 서버 프로세스에 그대로 넘기면 된다:")
            print(f"    ./scripts/slurm/submit.sh {kv}")
    else:
        print("  통하는 설정 없음. 이 노드에서 mooncake 로 GPU KV 전송은 불가능하다.")
        print("  남는 선택지는 셋뿐이고, 전부 실험 설계에 영향을 준다:")
        print("    1. 관리자에게 nvidia_peermem 로드 요청 (가장 깨끗하다)")
        print("    2. --disaggregation-transfer-backend nixl 로 백엔드 교체")
        print("       -> A6000 결과는 전부 mooncake 였으므로 교차 비교가 깨진다")
        print("    3. RDMA 없는 노드/파티션을 쓴다 (server17 이 그랬듯 TCP 자동 폴백)")
    return 0 if winners else 5


def sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def section(t):
    print(f"\n=== {t} ===")


# sweep 은 자식 프로세스만 돌리고 끝낸다 (본체는 각 자식이 실행한다).
if "--sweep" in sys.argv:
    sys.exit(run_sweep())


section("환경")
print(f"  hostname       : {sh('hostname')}")
_ulimit = sh("bash -c 'ulimit -l'")
print(f"  ulimit -l      : {_ulimit}  (RDMA 는 memory locking 을 쓴다)")
print(f"  nvidia_peermem : {sh('lsmod | grep -i peermem') or '<로드되지 않음>'}")
print(f"  ib devices     : {sh('ls /sys/class/infiniband 2>/dev/null') or '<없음>'}")
for k, v in sorted(os.environ.items()):
    if k.startswith("MC_"):
        print(f"  {k:15s}: {v}")

section("mooncake import")
try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    print(f"  FAIL: {e}")
    print("  (LD_PRELOAD=/lib64/libcrypto.so.3 이 필요할 수 있다 -- MOONCAKE_LD_FIX=system)")
    sys.exit(1)
print("  OK")

section(f"TransferEngine.initialize(protocol={os.environ.get('PROBE_PROTOCOL', 'rdma')!r})")
# sglang 이 넘기는 것과 같은 인자. device_name 은 --disaggregation-ib-device 에서 온다.
#
# PROBE_PROTOCOL 로 'tcp' 를 넣어볼 수 있다. sglang 은 이 자리를 "rdma" 로 하드코딩하지만
# (transfer_engine.py:181) 우리는 sglang 을 fork 해서 쓰므로, mooncake 쪽이 tcp 를
# 받아주기만 하면 한 줄로 바꿀 수 있다. 이 실험은 단일 노드라 RDMA 가 주는 게 없고,
# A6000 결과도 전부 TCP 폴백으로 측정됐다.
protocol = os.environ.get("PROBE_PROTOCOL", "rdma")
device_name = os.environ.get("PROBE_IB_DEVICE", "")
engine = TransferEngine()
rc = engine.initialize("127.0.0.1", "P2PHANDSHAKE", protocol, device_name)
print(f"  protocol={protocol!r} device_name={device_name!r}  ->  rc={rc}")
if rc != 0:
    print("  initialize 자체가 실패했다. 아래 등록 테스트는 의미가 없다.")
    sys.exit(2)

section("메모리 등록: HOST vs DEVICE")
# 이 두 줄이 A 와 B/C 를 가른다.
import ctypes

N = 32 * 1024 * 1024  # 32 MiB

host_buf = ctypes.create_string_buffer(N)
host_ptr = ctypes.addressof(host_buf)
host_rc = engine.register_memory(host_ptr, N)
print(f"  HOST   ptr=0x{host_ptr:x}  rc={host_rc}   {'OK' if host_rc == 0 else 'FAIL'}")
if host_rc == 0:
    engine.unregister_memory(host_ptr)

dev_rc = None
try:
    import torch

    t = torch.empty(N // 2, dtype=torch.float16, device="cuda:0")
    dev_ptr = t.data_ptr()
    dev_rc = engine.register_memory(dev_ptr, N)
    print(f"  DEVICE ptr=0x{dev_ptr:x}  rc={dev_rc}   {'OK' if dev_rc == 0 else 'FAIL'}")
    if dev_rc == 0:
        engine.unregister_memory(dev_ptr)
except Exception as e:  # noqa: BLE001
    print(f"  DEVICE  <테스트 불가: {e}>")

section("판정")
# 종료 코드로도 결론을 낸다. preflight.sh 가 이걸 읽고 서버를 띄우기 전에 멈춘다.
#   0 정상 / 3 device 등록 실패(A) / 4 host 등록 실패(B,C) / 1,2 위에서 이미 exit
_rc_out = 0
if host_rc == 0 and dev_rc not in (0, None):
    _rc_out = 3
elif host_rc != 0:
    _rc_out = 4

if host_rc == 0 and dev_rc not in (0, None):
    print("  A. GPUDirect RDMA 부재. host 는 등록되고 device 만 EFAULT 다.")
    print("     nvidia_peermem 커널 모듈이 없으면 정확히 이 모양이 된다 (로드는 root 권한).")
    print()
    print("  이 실험에는 RDMA 가 필요 없다 -- 단일 노드 1P1D 이고 P->D 가 같은 호스트다.")
    print("  A6000 결과도 전부 mooncake TCP 폴백으로 측정됐다.")
    if protocol != "tcp":
        print()
        print("  다음: protocol='tcp' 가 되는지 본다. 되면 sglang fork 한 줄로 끝난다.")
        print("    PROBE_PROTOCOL=tcp python scripts/slurm/probe_mooncake.py")
        print("  (sglang 은 이 자리를 'rdma' 로 하드코딩하지만 우리는 fork 를 쓴다.")
        print("   SGLANG_MOONCAKE_PROTOCOL=tcp 로 바꿀 수 있게 패치해 두었다.)")
elif host_rc != 0:
    print("  B/C. host 등록부터 실패한다. GPUDirect 문제가 아니다.")
    print(f"     ulimit -l 이 {_ulimit} 이다. unlimited 가 아니면 그것부터.")
    print("     sbatch 에 --propagate=MEMLOCK 를 붙이거나 관리자에게 한도 상향을 요청하라.")
else:
    print("  둘 다 등록된다. 그렇다면 sglang 이 넘기는 특정 버퍼가 문제이고,")
    print("  p1.log 의 실패 주소를 sglang 의 KV pool 주소와 대조해 봐야 한다.")

sys.exit(_rc_out)
