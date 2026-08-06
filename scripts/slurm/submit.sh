#!/bin/bash
# submit_bench.sh 를 sbatch 로 올리는 얇은 래퍼.
#
# 그냥 `sbatch scripts/slurm/submit_bench.sh` 로도 되지만, 이 래퍼는 두 가지를 대신한다:
#   1. logs/slurm 을 미리 만든다. #SBATCH --output 의 디렉터리가 없으면 job 은 아무것도
#      남기지 못하고 죽고, squeue 에는 그냥 사라진 것처럼 보인다.
#   2. EXP_ROOT 를 실제 리포 위치로 잡아 넘긴다. sbatch 는 스크립트를 노드의 spool 로
#      복사해 실행하므로 batch job 안에서는 스크립트 경로로 리포를 찾을 수 없다.
#
# 사용:
#   ./scripts/slurm/submit.sh                                # probe (기본)
#   ./scripts/slurm/submit.sh MODE=full DECODE_MAX_TOTAL_TOKENS=154304
#   ./scripts/slurm/submit.sh MODE=full WORKLOAD=bfcl REPEATS=3
#   GPUS=4 ./scripts/slurm/submit.sh MODE=full A100_TOPOLOGY=2p2d
#
# KEY=VALUE 인자는 그대로 job 환경으로 전달된다.
set -e

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$ROOT/logs/slurm" "$ROOT/results/a100"

PARTITION=${PARTITION:-suma_a100}
QOS=${QOS:-a100_qos}
GPUS=${GPUS:-2}
TIME=${TIME:-08:00:00}
MEM=${MEM:-128G}
CPUS=${CPUS:-16}
JOBNAME=${JOBNAME:-sglang-why}

EXPORTS="ALL,EXP_ROOT=$ROOT"
for kv in "$@"; do
  case "$kv" in
    *=*) EXPORTS="$EXPORTS,$kv" ;;
    *)   echo "인자는 KEY=VALUE 형태여야 한다: $kv" >&2; exit 1 ;;
  esac
done

echo "sbatch -p $PARTITION -q $QOS --gres=gpu:$GPUS --time=$TIME"
echo "  exports: $EXPORTS"
sbatch \
  --job-name="$JOBNAME" \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --gres=gpu:"$GPUS" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --output="$ROOT/logs/slurm/%x-%j.out" \
  --error="$ROOT/logs/slurm/%x-%j.err" \
  --export="$EXPORTS" \
  "$ROOT/scripts/slurm/submit_bench.sh"

echo
echo "확인:"
echo "  squeue -u \$USER"
echo "  tail -f $ROOT/logs/slurm/${JOBNAME}-<jobid>.out"
echo "  srun --jobid=<jobid> --pty bash        # 도는 job 안으로 들어가기"
