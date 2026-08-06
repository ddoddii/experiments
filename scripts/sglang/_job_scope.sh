# shellcheck shell=bash
# 공유 노드에서 "내 job 의 프로세스만" 죽이기.
#
# start/stop 스크립트들은 전부 `pkill -f sglang.launch_server` 를 한다. 전용 머신
# (server17) 에서는 맞는 동작이었다 — 거기 도는 sglang 은 늘 내 것 하나였다.
# SLURM 클러스터에서는 아니다. 내 job 두 개가 같은 A100 노드에 배치되면, 나중에 뜬
# job 의 [0/5] 정리 단계가 먼저 뜬 job 의 서버를 죽인다. 그 job 은 몇 시간짜리 arm 을
# 돌던 중이고, 죽는 방식은 조용하다: 벤치는 connection refused 를 에러로 세고 계속 돈다.
#
# pkill 은 다른 사용자의 프로세스는 어차피 못 죽이므로 문제는 "내 다른 job" 뿐이고,
# 그건 /proc/<pid>/environ 의 SLURM_JOB_ID 로 정확히 가려낼 수 있다. sglang 의 scheduler
# / detokenizer 서브프로세스도 launcher 의 환경을 상속하므로 같은 방법으로 잡힌다.
#
# SLURM 밖(전용 머신)에서는 예전 그대로 광역 pkill 로 동작한다.

# job_pkill <signal> <pattern>
job_pkill() {
  local sig=$1 pat=$2 pid
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    pkill "-${sig}" -f "$pat" 2>/dev/null || true
    return 0
  fi
  for pid in $(pgrep -u "$(id -u)" -f "$pat" 2>/dev/null); do
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
       | grep -qx "SLURM_JOB_ID=${SLURM_JOB_ID}"; then
      kill "-${sig}" "$pid" 2>/dev/null || true
    fi
  done
  return 0
}

# job_pgrep_count <pattern> — 내 job 소속으로 살아있는 개수
job_pgrep_count() {
  local pat=$1 pid n=0
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    pgrep -f "$pat" 2>/dev/null | wc -l
    return 0
  fi
  for pid in $(pgrep -u "$(id -u)" -f "$pat" 2>/dev/null); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
      | grep -qx "SLURM_JOB_ID=${SLURM_JOB_ID}" && n=$((n + 1))
  done
  echo "$n"
}
