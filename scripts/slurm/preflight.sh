#!/bin/bash
# A100 노드에서 실험을 돌리기 전에, 결과를 무효로 만들 수 있는 것들을 먼저 확인한다.
#
# 이 파일이 존재하는 이유는 "환경 점검"이 아니라 "무효한 결과 방지"다. 지금까지 이
# 디렉터리의 결과를 망친 실패들은 전부 조용했다: 서버는 뜨고, 벤치는 돌고, 숫자는
# 그럴듯하고, 다만 측정 대상이 아니었다. A100 으로 옮기면서 새로 생기는 조용한 실패:
#
#   peer access 불가   PyTorch 는 실패하지 않고 복사를 HOST 경유로 몰래 바꾼다.
#                      park_gpu arm 이 사실상 park_host 가 되고 로그도 telemetry 도
#                      여전히 "peer GPU hit" 이라고 말한다. (benchmark/p2p_matrix.py 참고)
#   A6000 상수 상속    peer/host 대역폭, prefill ms/token, 카드 크기가 분석 스크립트에
#                      박혀 있다. A100 에서 그대로 쓰면 "link budget" 계산이 통째로 틀린다.
#   모델 경로 부재     클러스터 홈이 다른 파일시스템이면 조용히 HF 에서 받으려 들거나
#                      몇 분 뒤에야 죽는다.
#
# 하드 실패(=exit 1)와 경고를 구분한다. 경고는 실험을 막지는 않지만 결과 해석을 바꾼다.
#
# 사용:
#   source scripts/slurm/env.sh && ./scripts/slurm/preflight.sh
#   OUTDIR=results/a100/j12345 ./scripts/slurm/preflight.sh
set -u

[ -n "${EXP_ROOT:-}" ] || { echo "먼저 scripts/slurm/env.sh 를 source 하라." >&2; exit 1; }
cd "$EXP_ROOT"

OUTDIR=${OUTDIR:-"results/a100/${JOB_TAG}"}
mkdir -p "$OUTDIR"
FAIL=0
WARN=0
say()  { echo "  $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }
warn() { echo "  [warn] $*"; WARN=$((WARN + 1)); }

echo "================================================================"
echo " PREFLIGHT  ${JOB_TAG}  on $(hostname)"
echo "================================================================"
env_summary
echo

# ─── 1. 모델 ──────────────────────────────────────────────────────────────────
echo "[1/7] model"
if [ -d "$MODEL_PATH" ] && [ -n "$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
  say "$MODEL_PATH  ($(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1))"
else
  bad "MODEL_PATH 에 모델 가중치가 없다: $MODEL_PATH"
  say "      MODEL_PATH=... 로 넘기거나 클러스터 공유 스토리지에 복사하라."
fi

# ─── 2. sglang 소스 트리 ──────────────────────────────────────────────────────
echo "[2/7] sglang"
_loaded=$(python -c 'import os,sglang;print(os.path.dirname(os.path.abspath(sglang.__file__)))' 2>/dev/null)
if [ -z "$_loaded" ]; then
  bad "sglang import 실패 (conda env=${SGLANG_CONDA_ENV})"
else
  say "import 위치: $_loaded"
  case "$_loaded" in
    "$SGLANG_SRC_DIR"/*)
      say "branch: $(git -C "$SGLANG_SRC_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null)" \
          "@ $(git -C "$SGLANG_SRC_DIR" rev-parse --short HEAD 2>/dev/null)" ;;
    *)
      bad "설치본을 import 하고 있다. idle-KV-parking 패치가 적용되지 않는다."
      say "      cd $SGLANG_SRC_DIR/python && pip install -e . --no-deps" ;;
  esac
fi

# mooncake 메타데이터 서버가 포트를 인자로 받는가. 공유 노드에서는 8080 고정이 곧 충돌이다.
_mc=$(python -c 'import inspect,mooncake.http_metadata_server as m;print(inspect.getsourcefile(m))' 2>/dev/null)
if [ -z "$_mc" ]; then
  bad "mooncake.http_metadata_server import 실패 (disaggregation transfer backend)"
  say "      PD disaggregation 의 KV 전송 백엔드다. 없으면 서버가 아예 안 뜬다."
  say "      conda activate ${SGLANG_CONDA_ENV} && pip install mooncake-transfer-engine==0.3.8.post1"
  say "      (버전은 sglang 자신의 CI 가 쓰는 핀: scripts/ci/ci_install_dependency.sh)"
  say "      A6000 결과가 전부 mooncake 로 측정됐으므로, 백엔드를 바꾸면 비교가 깨진다."
elif grep -qE '\-\-port|"port"' "$_mc" 2>/dev/null; then
  say "mooncake metadata server: --port 지원 O  (이 job 은 :${MOONCAKE_PORT})"
else
  warn "mooncake metadata server 가 --port 를 안 받는 것으로 보인다 ($_mc)."
  say "      MOONCAKE_PORT=8080 으로 두고, 노드를 독점(--exclusive)해서 돌려라."
fi

# ─── 3. GPU 와 배선 ───────────────────────────────────────────────────────────
echo "[3/7] GPU topology"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'
nvidia-smi topo -m > "$OUTDIR/nvidia_smi_topo.txt" 2>&1 || true
say "topo -m -> $OUTDIR/nvidia_smi_topo.txt"

# P/D 쌍이 NVLink 인지. A100 SXM(NVSwitch)이면 모든 쌍이 NV12, PCIe A100 이면 어느 쌍도
# 아니다. 후자에서 park_gpu 를 돌리면 "idle decode HBM 에 park" 라는 주장은 그대로지만
# 매체는 PCIe 가 되고, 결과의 의미가 달라진다 — 막지는 않되 반드시 기록에 남긴다.
# 데이터 행만 골라야 한다. 헤더 행도 첫 필드가 "GPU0" 이라 ($1 == "GPU0") 조건에
# 같이 걸리고, 그러면 링크 값 대신 헤더의 열 이름("GPU2")까지 두 줄로 나온다.
# 자기 자신 칸이 "X" 인 행이 데이터 행이라는 게 유일하게 안정적인 구분점이다.
_link=$(nvidia-smi topo -m 2>/dev/null | awk -v a="GPU${PREFILL_GPU}" -v d="${DECODE_GPU}" '
  $1 == a {
    for (i = 2; i <= NF; i++) if ($i == "X") { print $(d + 2); exit }
  }')
say "link P(gpu${PREFILL_GPU}) <-> D(gpu${DECODE_GPU}) : ${_link:-unknown}"
case "$_link" in
  NV*) : ;;
  "" ) warn "topo 행렬에서 링크를 못 읽었다. $OUTDIR/nvidia_smi_topo.txt 를 직접 확인하라." ;;
  *  ) warn "P/D 쌍이 NVLink 가 아니다 (${_link}). park_gpu 는 PCIe 를 타게 되고,"
       say "      'peer HBM over NVLink' 라는 주장의 매체 축이 바뀐다. 결과에 명시할 것." ;;
esac

# ─── 4. peer access: 능력이 아니라 실제로 IPC 가 되는가 ───────────────────────
# torch 는 CUDA_VISIBLE_DEVICES 안에서 0..N-1 로 다시 번호를 매기므로, 여기서는 논리
# 인덱스(P=0, D=1)로 물어야 한다. 물리 번호로 물으면 없는 장치를 짚는다.
echo "[4/7] peer access (CUDA IPC)"
python - "$OUTDIR/peer_check.json" <<'PY'
import json, sys, time
import torch

out = sys.argv[1]
n = torch.cuda.device_count()
res = {"n_visible": n, "pairs": {}}
if n < 2:
    res["error"] = f"need >= 2 visible GPUs, found {n}"
    print(f"  [FAIL] {res['error']}")
else:
    src, dst = 0, 1
    can = torch.cuda.can_device_access_peer(src, dst)
    n_elem = 256 * 2**20 // 2
    a = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{src}")
    b = torch.empty(n_elem, dtype=torch.float16, device=f"cuda:{dst}")
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    t0 = time.perf_counter()
    for _ in range(10):
        b.copy_(a)
    torch.cuda.synchronize(src); torch.cuda.synchronize(dst)
    gbs = (n_elem * 2) / ((time.perf_counter() - t0) / 10) / 1e9
    res["pairs"]["0->1"] = {"can_peer": bool(can), "gbps": round(gbs, 1)}
    res["card"] = torch.cuda.get_device_name(0)
    res["card_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    print(f"  {res['card']}  {res['card_gb']:.0f} GB")
    print(f"  P->D can_peer={can}  measured {gbs:.1f} GB/s")
json.dump(res, open(out, "w"), indent=1)
if res.get("pairs", {}).get("0->1", {}).get("can_peer") is False:
    print("  [FAIL] peer access 불가. PyTorch 가 복사를 host 경유로 바꾸므로")
    print("         park_gpu arm 이 사실상 park_host 가 된다 — 두 arm 의 차이가 사라진다.")
    sys.exit(3)
PY
_rc=$?
[ "$_rc" -eq 3 ] && bad "peer access 불가 (위 참고)"
[ "$_rc" -ne 0 ] && [ "$_rc" -ne 3 ] && bad "peer check 스크립트 실패 (rc=$_rc)"

# ─── 5. 이 머신의 상수 ────────────────────────────────────────────────────────
# 분석 스크립트가 A6000 값을 물려받지 않도록 여기서 새로 측정한다. 서버가 뜨기 전에
# 돌려야 한다 (모든 GPU 에 할당한다).
echo "[5/7] machine constants (hwprofile)"
# 파일이 "있다"는 것만으로 재사용하면 안 된다. 이 파일이 막으려는 실패가 정확히 그것 --
# A6000 에서 잰 hwprofile.json 이 리포에 남아 있으면 A100 에서 한 번도 재측정하지 않고
# peer/host 대역폭과 카드 크기를 그대로 물려받는다. 기록된 gpu_name 이 지금 카드와
# 다르면 무조건 다시 잰다.
_prof_gpu=$(python -c "import json;print(json.load(open('results/hwprofile.json'))['gpu_name'])" 2>/dev/null || echo "")
_here_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -n "$_prof_gpu" ] && [ "$_prof_gpu" != "$_here_gpu" ]; then
  say "기존 hwprofile 은 '${_prof_gpu}' 것이고 지금 카드는 '${_here_gpu}' -> 재측정한다."
  FORCE_HWPROFILE=1
fi
if [ ! -f "results/hwprofile.json" ] || [ "${FORCE_HWPROFILE:-0}" = "1" ]; then
  python benchmark/hwprofile.py --out "$OUTDIR/hwprofile.json" 2>&1 | sed 's/^/  /' || warn "hwprofile 실패"
  cp "$OUTDIR/hwprofile.json" results/hwprofile.json 2>/dev/null || true
else
  say "results/hwprofile.json 이 이미 있다 (FORCE_HWPROFILE=1 로 재측정)."
  cp results/hwprofile.json "$OUTDIR/hwprofile.json" 2>/dev/null || true
fi
# p2p_matrix.py 의 NVLINK_REF 기본값은 A6000 에서 잰 52.7 GB/s 다. A100 NV12 는 그
# 5배쯤 나오므로, 기본값을 두면 PCIe 경유 쌍까지 "full speed" 로 분류된다. 방금 잰
# 이 머신의 최고 peer 대역폭을 기준으로 넘긴다.
_ref=$(python -c "import json;print(json.load(open('$OUTDIR/hwprofile.json'))['peer_gpu_gbps'])" 2>/dev/null || echo "")
NVLINK_REF=${_ref:-52.7} python benchmark/p2p_matrix.py > "$OUTDIR/p2p_matrix.txt" 2>&1 \
  || warn "p2p_matrix 실패"
say "p2p matrix -> $OUTDIR/p2p_matrix.txt  (NVLINK_REF=${_ref:-52.7} GB/s 기준)"

# ─── 6. host RAM: hicache arm 이 여기서 죽는다 ────────────────────────────────
echo "[6/7] host memory"
# 노드의 MemAvailable 이 아니라 이 JOB 이 쓸 수 있는 양을 봐야 한다. cgroup 이 붙은
# 클러스터에서 두 값은 완전히 다르다 -- 실측 예: 노드 942 GB, job 상한 128 GB.
# MemAvailable 로 판단하면 "여유 충분"이라고 말한 다음 hicache arm 이 OOM 으로 죽는다.
_node_gb=$(awk '/MemAvailable/ {printf "%.0f", $2/1048576}' /proc/meminfo)
_avail_gb=$_node_gb
if [ -n "${SLURM_MEM_PER_NODE:-}" ] && [ "${SLURM_MEM_PER_NODE}" -gt 0 ] 2>/dev/null; then
  _slurm_gb=$((SLURM_MEM_PER_NODE / 1024))
  [ "$_slurm_gb" -lt "$_avail_gb" ] && _avail_gb=$_slurm_gb
fi
say "node MemAvailable: ${_node_gb} GB, SLURM 할당: ${SLURM_MEM_PER_NODE:-?} MB" \
    "-> 이 job 의 상한 ${_avail_gb} GB"
# hicache host pool ≈ device KV pool x HICACHE_RATIO. device pool 은 카드 크기에서
# 가중치를 뺀 값이므로 A100 80GB 에서는 A6000 보다 훨씬 크다 — 같은 ratio 가 훨씬 큰
# host 할당을 뜻한다는 게 A6000 에서 옮겨올 때의 함정이다.
_card_gb=$(python -c 'import json;print(json.load(open("'"$OUTDIR"'/hwprofile.json"))["card_gb"])' 2>/dev/null || echo 0)
if [ "${_card_gb%.*}" -gt 0 ] 2>/dev/null; then
  _need=$(python -c "print(int(($_card_gb - 17) * ${HICACHE_RATIO:-1.2}))" 2>/dev/null || echo 0)
  say "hicache host pool 추정: ~${_need} GB (카드 ${_card_gb}GB, ratio ${HICACHE_RATIO:-1.2})"
  # A100 80GB 는 A6000 48GB 보다 device KV pool 이 훨씬 크고, hicache host pool 은
  # 그 pool 의 ratio 배다. 즉 "A6000 에서 쓰던 ratio 1.2" 가 여기서는 훨씬 큰 host
  # 할당을 뜻한다 -- 카드만 바꿔도 hicache arm 만 OOM 나는 이유가 이것이다.
  if [ "$_need" -gt "${_avail_gb:-0}" ] 2>/dev/null; then
    warn "추정 host pool(${_need}GB) > job 상한(${_avail_gb}GB). hicache arm 이 OOM 난다."
    say "      HICACHE_RATIO 를 낮추거나 sbatch --mem 을 올려라."
  elif [ "$((_need * 10))" -gt "$((${_avail_gb:-0} * 7))" ] 2>/dev/null; then
    warn "추정 host pool(${_need}GB) 이 job 상한(${_avail_gb}GB) 의 70% 를 넘는다."
    say "      모델 로딩/활성화/page cache 가 나머지를 쓰므로 빠듯하다. --mem 을 올려두는 편이 낫다."
  fi
fi
_disk=$(df -BG --output=avail "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
say "hicache file backend 여유 디스크: ${_disk:-?} GB  ($SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR)"
[ -n "$_disk" ] && [ "$_disk" -lt 40 ] 2>/dev/null && \
  warn "40GB 미만. hicache file backend 는 종료 시까지 지우지 않아 arm 도중 디스크를 채운 전례가 있다."

# ─── 7. 포트 ──────────────────────────────────────────────────────────────────
echo "[7/7] ports"
for _p in $PREFILL_PORT $DECODE_PORT $ROUTER_PORT $ROUTER_PROM_PORT $BOOTSTRAP_PORT $MOONCAKE_PORT; do
  if command -v ss >/dev/null 2>&1 && [ -n "$(ss -Hltn "sport = :$_p" 2>/dev/null)" ]; then
    bad "포트 $_p 이미 사용 중 — PORT_BASE 를 바꿔서 다시 제출하라."
  fi
done
say "block ${PORT_BASE}..$((PORT_BASE + 30)) 사용 가능"

# ─── 기록 ─────────────────────────────────────────────────────────────────────
# 중첩 따옴표를 피해 먼저 변수로 뽑는다 (echo 안에서 $(python -c "...") 를 쓰면 바깥
# 문자열이 닫히는 지점이 모호해진다).
_peer_gbps=$(python -c "import json;print(json.load(open('$OUTDIR/peer_check.json'))['pairs']['0->1']['gbps'])" 2>/dev/null)
{
  echo "{"
  echo "  \"job\": \"${JOB_TAG}\", \"host\": \"$(hostname)\", \"ts\": $(date +%s),"
  echo "  \"partition\": \"${SLURM_JOB_PARTITION:-}\", \"gpus\": \"${A100_GPU_CSV}\","
  echo "  \"gpu_name\": \"${_here_gpu:-}\", \"peer_gbps\": ${_peer_gbps:-null},"
  echo "  \"topology\": \"${A100_TOPOLOGY}\", \"pd_link\": \"${_link:-unknown}\","
  echo "  \"port_base\": ${PORT_BASE}, \"model\": \"${MODEL_PATH}\","
  echo "  \"mem_available_gb\": ${_avail_gb:-0}, \"fails\": ${FAIL}, \"warns\": ${WARN}"
  echo "}"
} > "$OUTDIR/preflight.json"

echo
echo "================================================================"
if [ "$FAIL" -gt 0 ]; then
  echo " PREFLIGHT FAILED: ${FAIL} 개 (경고 ${WARN}개).  실행하지 않는다."
  echo "================================================================"
  exit 1
fi
echo " PREFLIGHT OK (경고 ${WARN}개) -> $OUTDIR/preflight.json"
echo "================================================================"
