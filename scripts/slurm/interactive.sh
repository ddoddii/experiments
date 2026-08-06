#!/bin/bash
# A100 대화형 셸. 배치로 올리기 전에 손으로 확인할 때 쓴다.
#
#   ./scripts/slurm/interactive.sh              # GPU 2장 (1P1D), 2시간
#   GPUS=4 TIME=04:00:00 ./scripts/slurm/interactive.sh
#
# 붙고 나서 할 일:
#   source ~/experiments/scripts/slurm/env.sh
#   ./scripts/slurm/preflight.sh
#   NODES=1p1d ARMS=park_gpu ./scripts/sglang/run_why_faster.sh
#
# 이미 도는 배치 job 안으로 들어가려면 (같은 노드, 같은 할당, 같은 cgroup):
#   squeue -u $USER
#   srun --jobid=<jobid> --pty bash
# 이렇게 붙은 셸에는 SLURM_JOB_ID 가 그대로 있으므로, env.sh 를 source 하면 그 job 과
# 똑같은 포트/park 디렉터리를 잡는다 -- 즉 그 job 이 띄운 서버에 그대로 붙는다.
# 반대로 그 셸에서 stop 스크립트를 부르면 그 job 의 서버를 죽인다. 의도한 것만 하라.
set -e

PARTITION=${PARTITION:-suma_a100}
QOS=${QOS:-a100_qos}
GPUS=${GPUS:-2}
TIME=${TIME:-02:00:00}
MEM=${MEM:-128G}
CPUS=${CPUS:-16}

echo "srun -p $PARTITION -q $QOS --gres=gpu:$GPUS --time=$TIME --pty bash -i"
exec srun \
  --partition="$PARTITION" \
  --qos="$QOS" \
  --gres=gpu:"$GPUS" \
  --nodes=1 --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --pty bash -i
