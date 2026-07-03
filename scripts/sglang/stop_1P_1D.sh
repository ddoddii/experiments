#!/bin/bash
# Phase 0: 1P1D 서버 및 관련 프로세스 종료.
echo "Stopping SGLang 1P1D servers, router, mooncake..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true
for port in ${BOOTSTRAP_PORT:-8998} ${PREFILL_PORT:-30000} ${DECODE_PORT:-30001} ${ROUTER_PORT:-8000} 8080; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 2
echo "Done."
