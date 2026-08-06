#!/bin/bash
# Phase 0: 1P1D 서버 및 관련 프로세스 종료.
#
# SLURM 하위에서는 내 job 소속 프로세스만 죽인다 (_job_scope.sh 주석 참고). 같은 A100
# 노드에 내 job 이 두 개 뜨면, 광역 pkill 은 옆에서 몇 시간째 돌던 arm 을 조용히 죽인다.
# 포트도 job 마다 다르므로 env 에서 받은 값만 정리한다.
set +u
source "$(dirname "${BASH_SOURCE[0]}")/_job_scope.sh"

echo "Stopping SGLang 1P1D servers, router, mooncake${SLURM_JOB_ID:+ (job ${SLURM_JOB_ID} 한정)}..."
# SIGTERM 먼저: SIGKILL 당한 scheduler 는 GPU 컨텍스트를 몇 초 더 붙들고 있어서
# 다음 arm 의 서버가 같은 GPU 를 잡을 때 OOM 으로 보이는 실패를 낸다.
for pat in "sglang::scheduler" "sglang::detokenizer" "sglang.launch_server" \
           "sglang_router.launch_router" "mooncake.http_metadata_server"; do
  job_pkill TERM "$pat"
done
for i in $(seq 1 15); do
  left=0
  for pat in "sglang.launch_server" "sglang_router.launch_router" "mooncake.http_metadata_server"; do
    left=$((left + $(job_pgrep_count "$pat")))
  done
  [ "$left" -eq 0 ] && break
  sleep 1
done
for pat in "sglang::scheduler" "sglang::detokenizer" "sglang.launch_server" \
           "sglang_router.launch_router" "mooncake.http_metadata_server"; do
  job_pkill 9 "$pat"
done

for port in ${BOOTSTRAP_PORT:-8998} ${PREFILL_PORT:-30000} ${DECODE_PORT:-30001} \
            ${ROUTER_PORT:-8000} ${ROUTER_PROM_PORT:-29000} ${MOONCAKE_PORT:-8080}; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2

# park telemetry 는 GPU 번호로만 키가 잡혀 있어서, 남겨두면 다음 arm 의 sampler 가
# 죽은 서버의 파일을 살아있는 점유율로 읽는다. KEEP_PARK_DIR=1 로 보존 가능.
PARK_DIR=${SGLANG_KV_PARK_DIR:-/dev/shm/sglang_kv_parking}
if [ -d "$PARK_DIR" ] && [ -z "$KEEP_PARK_DIR" ]; then
  rm -f "$PARK_DIR"/parked_gpu*.json "$PARK_DIR"/gpu*_usage 2>/dev/null
fi
# hicache file backend 는 종료 시 스스로 지우지 않는다. hicache arm 만 KV 를 디스크에
# 쓰므로, 놔두면 그 arm 만 디스크를 채우다 죽는다 (이미 두 번 측정을 날렸다).
HICACHE_DIR=${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-/tmp/hicache}
if [ -d "$HICACHE_DIR" ] && [ -z "$KEEP_HICACHE_DIR" ]; then
  rm -rf "${HICACHE_DIR:?}"/* 2>/dev/null
fi
echo "Done."
