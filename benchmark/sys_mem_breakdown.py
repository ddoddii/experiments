#!/usr/bin/env python3
"""
Fine-grained memory breakdown sampler -- separates the categories a reviewer will
demand when you claim "HiCache uses 122GB of host RAM":

  GPU HBM ............ nvidia-smi memory.used (summed over GPUs)
  proc_anon .......... SGLang processes' ANONYMOUS host RSS (from per-pid
                       smaps_rollup) -- the real, NON-reclaimable allocation
                       (heap + the pinned/allocated L2 host KV pool). Per-process
                       so other tenants on a shared server don't pollute it.
  proc_locked ........ the mlock'd / pinned subset of the above (Locked in
                       smaps_rollup) -- if SGLang pins the L2 KV pool, this is it.
  proc_rss ........... total RSS of the SGLang processes (anon + file-mapped).
  file_cache ......... Active(file)+Inactive(file) from /proc/meminfo -- RECLAIMABLE
                       file-backed page cache. After a drop_caches at run start this
                       is dominated by the HiCache file backend (/tmp/hicache) -- i.e.
                       exactly the "is that 94GB really just page cache?" number.
  hicache_dir ........ on-disk size of the HiCache file backend dir (du) -- the L3
                       KV volume actually written (resident-or-not).
  reclaimable / nonreclaimable ... system-level split (meminfo).

The honest "host memory COST of HiCache" is proc_anon (+ any file_cache you count),
with file_cache flagged as reclaimable/disk-backed. write_back vs write_through vs
write_through_selective, and file-backend-on vs L2-only, move these very differently.

Output: CSV, one row per sample.

사용:
  python benchmark/sys_mem_breakdown.py --out results/mem/breakdown_write_back.csv \
    --interval 1 --hicache-dir /tmp/hicache
  # optionally pin which processes count as "SGLang":
  #   --pid-pattern "sglang.launch_server"
"""
import argparse
import csv
import os
import subprocess
import time


def read_meminfo():
    """/proc/meminfo -> dict of MB (float)."""
    out = {}
    want = {"MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached",
            "Active(file)", "Inactive(file)", "AnonPages", "Mapped", "Shmem",
            "SReclaimable", "SUnreclaim", "Unevictable", "Mlocked", "Dirty"}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            if k in want:
                out[k] = float(rest.strip().split()[0]) / 1024.0   # kB -> MB
    return out


def _ppid_map():
    """pid -> ppid for every process (from /proc/*/stat)."""
    m = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            m[int(name)] = int(parts[1])   # field after "(comm)": state, ppid
        except Exception:  # noqa: BLE001
            continue
    return m


def sglang_pids(pattern):
    """Roots matching `pattern` PLUS all their descendants. The launch_server main is
    a thin HTTP proc; the big host KV pool lives in the scheduler/TP-worker children,
    whose cmdline may NOT contain the pattern -- so we must walk the process tree, or
    we undercount the L2 host pool by tens of GB."""
    try:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        roots = set(int(x) for x in r.stdout.split())
    except Exception:  # noqa: BLE001
        return []
    if not roots:
        return []
    ppid = _ppid_map()
    children = {}
    for pid, par in ppid.items():
        children.setdefault(par, []).append(pid)
    seen, stack = set(roots), list(roots)
    while stack:
        p = stack.pop()
        for ch in children.get(p, []):
            if ch not in seen:
                seen.add(ch); stack.append(ch)
    return sorted(seen)


def proc_mem(pids):
    """Sum smaps_rollup fields over pids (MB). Returns (rss, anon, locked, pss, pss_anon).
    NOTE: summing raw Anonymous/Rss over sibling processes DOUBLE-COUNTS pages they share
    (CUDA IPC, pinned host buffers, /dev/shm) -- it can exceed the system-wide AnonPages,
    which is nonsense. Pss / Pss_Anon divide each shared page by its sharers, so summing
    THEM is the de-duplicated per-process attribution. Prefer pss_anon (or the global
    AnonPages minus a GPU-only baseline) for the real host-pool number."""
    rss = anon = locked = pss = pss_anon = 0.0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                fields = {}
                for line in f:
                    k, _, rest = line.partition(":")
                    if rest.strip().endswith("kB"):
                        fields[k.strip()] = float(rest.strip().split()[0]) / 1024.0
            rss += fields.get("Rss", 0.0)
            anon += fields.get("Anonymous", 0.0)
            locked += fields.get("Locked", 0.0)
            pss += fields.get("Pss", 0.0)
            pss_anon += fields.get("Pss_Anon", 0.0)
        except Exception:  # noqa: BLE001
            continue
    return rss, anon, locked, pss, pss_anon


def gpu_hbm_mb():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return sum(float(x) for x in r.stdout.split())
    except Exception:  # noqa: BLE001
        return 0.0


def dir_mb(path):
    if not path or not os.path.isdir(path):
        return 0.0
    try:
        r = subprocess.run(["du", "-sb", path], capture_output=True, text=True, timeout=30)
        return float(r.stdout.split()[0]) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--pid-pattern", default="sglang.launch_server")
    ap.add_argument("--hicache-dir", default="/tmp/hicache")
    ap.add_argument("--dir-interval", type=float, default=10.0,
                    help="how often to du the hicache dir (slower op; carried between)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["elapsed_s", "gpu_hbm_mb", "proc_rss_mb", "proc_anon_mb", "proc_pss_mb",
            "proc_pss_anon_mb", "proc_locked_mb", "file_cache_mb", "hicache_dir_mb",
            "anonpages_mb", "reclaimable_mb", "nonreclaimable_mb", "mem_used_mb",
            "mem_avail_mb", "n_pids"]
    t0 = time.time()
    last_dir = 0.0
    last_dir_t = -1e9
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        print(f"[breakdown] -> {args.out}  (pattern='{args.pid_pattern}', "
              f"hicache_dir={args.hicache_dir})  Ctrl-C to stop")
        while True:
            now = time.time()
            mi = read_meminfo()
            pids = sglang_pids(args.pid_pattern)
            rss, anon, locked, pss, pss_anon = proc_mem(pids)
            gpu = gpu_hbm_mb()
            if now - last_dir_t >= args.dir_interval:
                last_dir = dir_mb(args.hicache_dir); last_dir_t = now

            file_cache = mi.get("Active(file)", 0.0) + mi.get("Inactive(file)", 0.0)
            reclaimable = file_cache + mi.get("SReclaimable", 0.0)
            mem_used = mi.get("MemTotal", 0.0) - mi.get("MemFree", 0.0)
            nonreclaim = mem_used - reclaimable - mi.get("Buffers", 0.0)

            w.writerow([round(now - t0, 1), round(gpu, 1), round(rss, 1), round(anon, 1),
                        round(pss, 1), round(pss_anon, 1), round(locked, 1),
                        round(file_cache, 1), round(last_dir, 1),
                        round(mi.get("AnonPages", 0.0), 1), round(reclaimable, 1),
                        round(nonreclaim, 1), round(mem_used, 1),
                        round(mi.get("MemAvailable", 0.0), 1), len(pids)])
            fh.flush()
            time.sleep(max(0.05, args.interval - (time.time() - now)))


if __name__ == "__main__":
    main()
