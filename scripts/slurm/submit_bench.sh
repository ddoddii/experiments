#!/bin/bash
#SBATCH --job-name=sglang-why
#SBATCH --partition=suma_a100
#SBATCH --qos=a100_qos
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/uhmturks/experiments/logs/slurm/%x-%j.out
#SBATCH --error=/home/uhmturks/experiments/logs/slurm/%x-%j.err
#
# A100 배치 실행 엔트리.
#
#     sbatch ~/experiments/scripts/slurm/submit_bench.sh
#     sbatch --export=ALL,WORKLOAD=bfcl,REPEATS=3 ~/experiments/scripts/slurm/submit_bench.sh
#     ./scripts/slurm/submit.sh WORKLOAD=bfcl REPEATS=3      # 위와 같지만 디렉터리를 먼저 만든다
#
# 위 #SBATCH --output 경로는 리터럴이어야 한다 (SBATCH 지시어는 변수 확장이 안 되고,
# 그 디렉터리가 없으면 job 이 출력조차 못 쓰고 죽는다). 리포가 ~/experiments 가 아니면
# 두 줄을 고치거나 scripts/slurm/submit.sh 를 쓰라 -- 명령행 --output 이 지시어를 이긴다.
#
# GPU 장수는 토폴로지와 함께 움직인다. 기본은 1P1D (2장):
#     sbatch --gres=gpu:4 --export=ALL,A100_TOPOLOGY=2p2d ...   # 2P2D
#
# ─── A100 으로 넘어오면서 다시 재야 하는 것들 ─────────────────────────────────
# 이 실험의 분석 스크립트에는 A6000 에서 잰 상수가 박혀 있다. preflight 가 hwprofile 을
# 새로 돌려 카드 크기/대역폭은 갱신하지만, 서버를 띄워야만 얻는 값은 아직 A6000 값이다:
#
#   prefill ms/token       A6000 0.132. A100 은 훨씬 빠르므로 "캐시 히트 한 번의 가치"가
#                          줄고, 같은 히트율 상승이 더 적은 wall-clock 으로 돌아온다.
#                          benchmark/ttft_ctx_sweep.py 로 잰 뒤 hwprofile 에 넣어라.
#   DECODE_MAX_TOTAL_TOKENS  A6000 에서 잰 154304 을 A100 에 그대로 박으면, 그 카드에서는
#                          아무 의미도 없는 값으로 decode 를 조이는 셈이다 (= 대조군으로
#                          위장한 오설정). 그래서 기본값이 없다. 아래 PROBE 단계 참고.
#   PARK_POOL_TOKENS_PER_GPU  A6000 48GB 기준 32000. 80GB A100 이면 여유가 다르다.
#
# 그래서 이 스크립트는 두 모드를 가진다:
#   MODE=probe (기본)  arm 하나씩 짧게 돌려 각 arm 의 decode pool 용량을 뽑는다.
#                      결과로 "DECODE_MAX_TOTAL_TOKENS=<작은 쪽>" 을 출력한다.
#   MODE=full          그 값을 받아서 본 실험을 돌린다. 값이 없으면 경고하고 진행한다
#                      (run_why_faster.sh 가 이미 그 경고를 낸다).
set -e

EXP_ROOT=${EXP_ROOT:-"$HOME/experiments"}
source "$EXP_ROOT/scripts/slurm/env.sh"
cd "$EXP_ROOT"

MODE=${MODE:-probe}
export A100_TOPOLOGY=${A100_TOPOLOGY:-1p1d}
export NODES=$A100_TOPOLOGY            # run_why_faster.sh 가 읽는 이름
export WORKLOAD=${WORKLOAD:-bfcl}
export CONCURRENCY=${CONCURRENCY:-32}
export REPEATS=${REPEATS:-1}
# 디렉터리 이름에 job id 를 넣지 않는다. 제출할 때마다 새 디렉터리가 생겨서 쌓인다.
#
# 다만 job id 는 이름표가 아니라 출처 정보였다 -- 어느 job, 어느 노드, 어느 커밋,
# 어떤 전송 설정으로 나온 숫자인지가 이름에만 있었다. 그래서 이름에서 뺀 대신
# RUN_INFO.txt 로 디렉터리 안에 남긴다.
#
# CONCURRENCY 는 이름에 남긴다. c4 probe 와 c32 본 실험은 비교 대상이 아닌데 이름이
# 같으면 뒤엣것이 앞엣것을 덮어쓴다 (run_why_faster.sh 가 WORKLOAD 를 태그에 넣는
# 이유와 같다). 완전히 고정하고 싶으면 TAG=... 로 직접 넘겨라.
export TAG=${TAG:-"a100_${WORKLOAD}_c${CONCURRENCY}"}
export OUTDIR=${OUTDIR:-"results/a100/${TAG}"}

# 같은 이름을 재사용하면 새로 생기는 위험이 있다: 이번 run 에서 실패한 arm 의 자리에
# 지난 run 의 파일이 그대로 남아, 분석 스크립트가 그걸 이번 결과로 읽는다. 이 리포는
# 이미 그 함정에 빠진 적이 있다 (arm 하나만 재실행했더니 다른 arm 은 옛 빌드의 것이
# 남아 있었다). 그래서 시작할 때 직전 run 을 .prev 로 밀어둔다 -- 지우는 게 아니라
# 옮기는 것이고, 설정당 최대 두 세대만 남는다.
for _d in "$OUTDIR" "${OUTDIR}_probe"; do
  if [ -d "$_d" ] && [ -z "${KEEP_OUTDIR:-}" ]; then
    rm -rf "${_d}.prev"
    mv "$_d" "${_d}.prev"
    echo "  직전 run 을 ${_d}.prev 로 옮겼다 (KEEP_OUTDIR=1 이면 덮어쓰기)."
  fi
done
mkdir -p "$OUTDIR"

# 이름에서 뺀 출처 정보를 여기에 남긴다.
{
  echo "job        : ${SLURM_JOB_ID:-<none>}  (${JOB_TAG})"
  echo "node       : $(hostname)"
  echo "started    : $(date -Is)"
  echo "mode       : ${MODE}"
  echo "workload   : ${WORKLOAD}  concurrency=${CONCURRENCY}  repeats=${REPEATS}"
  echo "topology   : ${A100_TOPOLOGY}  gpus=[${A100_GPU_CSV}]"
  echo "experiments: $(git -C "$EXP_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$EXP_ROOT" rev-parse --short HEAD 2>/dev/null)"
  echo "sglang     : $(git -C "$SGLANG_SRC_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null) @ $(git -C "$SGLANG_SRC_DIR" rev-parse --short HEAD 2>/dev/null)"
  echo "mooncake   : protocol=${SGLANG_MOONCAKE_PROTOCOL} ld_fix=${MOONCAKE_LD_FIX:-none} ib_device=${IB_DEVICE:-<auto>}"
  echo "MC_* env   : $(env | grep '^MC_' | sort | tr '\n' ' ')"
  echo "decode pool: ${DECODE_MAX_TOTAL_TOKENS:-<unpinned>}"
} > "$OUTDIR/RUN_INFO.txt"

echo "################################################################"
echo " A100 ${MODE}  |  ${TAG}"
echo "################################################################"
env_summary
echo

# ─── 1. preflight ─────────────────────────────────────────────────────────────
# 실패하면 여기서 끝낸다. 무효한 결과를 몇 시간 걸려 만드는 것보다 낫다.
OUTDIR="$OUTDIR" ./scripts/slurm/preflight.sh

# ─── 2. probe: 이 카드에서 각 arm 의 decode pool 이 얼마나 되는가 ─────────────
if [ "$MODE" = "probe" ]; then
  echo
  echo "################################################################"
  echo " PROBE: arm 별 decode KV pool 용량 측정 (짧게)"
  echo "################################################################"
  # 짧게 도는 게 목적. 숫자는 로그의 max_total_num_tokens 에서 나오지 벤치 결과에서
  # 나오지 않으므로, 워크로드는 서버를 띄우고 한 번 때리는 용도면 충분하다.
  MAX_ITEMS=${PROBE_ITEMS:-8} \
  CONCURRENCY=${PROBE_CONCURRENCY:-4} \
  REPEATS=1 \
  ARMS="${PROBE_ARMS:-hicache park_gpu}" \
  TAG="${TAG}_probe" OUTDIR="${OUTDIR}_probe" \
    ./scripts/sglang/run_why_faster.sh || true

  echo
  echo "=== arm 별 decode pool (max_total_num_tokens) ==="
  _min=""
  for f in "${OUTDIR}_probe"/d1.*.digest.txt; do
    [ -e "$f" ] || continue
    _v=$(grep -oE "max_total_num_tokens=[0-9]+" "$f" | tail -1 | cut -d= -f2)
    echo "  $(basename "$f"): ${_v:-?}"
    # if 문으로 쓴다: `[ ... ] && x` 가 루프 본문의 마지막 명령이면 테스트가 거짓일 때
    # errexit 가 걸려서, probe 가 결론을 내기 직전에 조용히 끝난다.
    if [ -n "$_v" ]; then
      if [ -z "$_min" ] || [ "$_v" -lt "$_min" ]; then _min=$_v; fi
    fi
  done
  echo
  if [ -n "$_min" ]; then
    echo "  두 arm 을 같은 decode 용량으로 맞추려면 작은 쪽을 고정한다:"
    echo
    echo "      sbatch --export=ALL,MODE=full,DECODE_MAX_TOTAL_TOKENS=${_min} \\"
    echo "             $EXP_ROOT/scripts/slurm/submit_bench.sh"
    echo
    echo "  이걸 고정하지 않으면 park arm 의 decode pool 이 park pool 크기만큼 줄어든 채로"
    echo "  hicache 와 비교된다 -- 캐시 계층 차이와 용량 삭감이 한 숫자에 섞인다."
    echo "{\"decode_max_total_tokens_suggested\": ${_min}}" > "$OUTDIR/probe_suggestion.json"
  else
    echo "  [warn] digest 에서 max_total_num_tokens 를 못 읽었다."
    echo "         ${OUTDIR}_probe/d1.*.digest.txt 를 직접 확인하라."
  fi
  echo
  echo "probe 결과: ${OUTDIR}_probe/"
  exit 0
fi

# ─── 3. full run ──────────────────────────────────────────────────────────────
export DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS:-}
export ARMS=${ARMS:-"recompute hicache park_host park_gpu"}
export PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-32000}

# 압박 knob. hicache 도 park 도 victim cache 라서, prefill 풀이 밀어내지 않으면 잡을
# 것이 없고 비용만 남는다. A100 80GB 에서 BFCL C=32 는 풀의 38% 밖에 채우지 못했고,
# 그 run 에서 hicache 는 recompute 보다 TTFT 가 1.84배 나빴다 -- 캐시가 진 게 아니라
# 캐시가 이길 수 있는 상황이 아니었다. 모든 arm 에 똑같이 적용된다 (start_1P_1D.sh 가
# 읽는다). 값 선정은 scripts/slurm/check_pressure.py 가 도와준다.
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-}
if [ -z "$PREFILL_MAX_TOTAL_TOKENS" ]; then
  echo "  [note] PREFILL_MAX_TOTAL_TOKENS 미설정: 80GB 카드에서는 풀이 안 차서"
  echo "         victim cache 가 잡을 것이 없을 수 있다. run 이 끝나면 아래 압박"
  echo "         점검 결과를 반드시 확인하라."
fi

echo
echo "################################################################"
echo " FULL RUN  arms=[$ARMS]  workload=$WORKLOAD  C=$CONCURRENCY  repeats=$REPEATS"
echo " decode pool 고정: ${DECODE_MAX_TOTAL_TOKENS:-<없음 -- arm 마다 다른 용량으로 비교됨>}"
echo "################################################################"

./scripts/sglang/run_why_faster.sh

# 서버가 남아 있으면 다음 job 이 같은 노드에 배치됐을 때 GPU 를 물고 있다.
./scripts/sglang/stop_1P_1D.sh || true

# TTFT 를 읽기 전에 이 run 이 결과를 낼 수 있는 조건이었는지부터 본다. 풀이 차지
# 않았다면 TTFT 비교는 "victim cache 가 효과 없다" 가 아니라 "측정할 조건이
# 아니었다" 를 뜻하고, 그 구분이 없으면 잘못된 결론이 그대로 남는다.
echo
python scripts/slurm/check_pressure.py --dir "$OUTDIR" \
  | tee "$OUTDIR/pressure_check.txt" || true

echo
echo "결과: $OUTDIR/"
