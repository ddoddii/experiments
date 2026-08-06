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


def sh(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def section(t):
    print(f"\n=== {t} ===")


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
