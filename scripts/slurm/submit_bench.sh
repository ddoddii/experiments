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
export TAG=${TAG:-"a100_${WORKLOAD}_c${CONCURRENCY}_${JOB_TAG}"}
export OUTDIR=${OUTDIR:-"results/a100/${TAG}"}
mkdir -p "$OUTDIR"

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

echo
echo "################################################################"
echo " FULL RUN  arms=[$ARMS]  workload=$WORKLOAD  C=$CONCURRENCY  repeats=$REPEATS"
echo " decode pool 고정: ${DECODE_MAX_TOTAL_TOKENS:-<없음 -- arm 마다 다른 용량으로 비교됨>}"
echo "################################################################"

./scripts/sglang/run_why_faster.sh

# 서버가 남아 있으면 다음 job 이 같은 노드에 배치됐을 때 GPU 를 물고 있다.
./scripts/sglang/stop_1P_1D.sh || true
echo
echo "결과: $OUTDIR/"
