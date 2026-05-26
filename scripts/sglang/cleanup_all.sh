#!/bin/bash
# SGLang 관련 프로세스 및 캐시 전체 정리

echo "[1/4] Killing sglang::scheduler / sglang::detokenizer..."
pkill -9 -f "sglang::scheduler" 2>/dev/null || true
pkill -9 -f "sglang::detokenizer" 2>/dev/null || true

echo "[2/4] Killing sglang launch_server / router / mooncake..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true

echo "[3/4] Freeing ports (8998 8999 30000 30001 30002 30003 8000 8080)..."
for port in 8998 8999 30000 30001 30002 30003 8000 8080; do
    fuser -k ${port}/tcp 2>/dev/null || true
done

echo "[4/4] Cleaning /tmp/hicache..."
if [ -d /tmp/hicache ]; then
    SIZE=$(du -sh /tmp/hicache 2>/dev/null | cut -f1)
    rm -rf /tmp/hicache
    echo "  Removed /tmp/hicache ($SIZE)"
else
    echo "  /tmp/hicache not found, skipping"
fi

sleep 2

echo ""
echo "Done. Remaining sglang processes:"
ps aux | grep -E "sglang|mooncake" | grep -v grep || echo "  (none)"
echo ""
echo "Disk free: $(df -h / | awk 'NR==2{print $4}') available"
