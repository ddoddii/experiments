# INVALID — the park arm never fetched anything back

`fetch_hits = 0`, and the cache hit rate came out at 46.8% against 47.5% for the no-cache
baseline: the mechanism contributed nothing, and the run cannot be used to say anything
about it.

Not a crash — 1440 turns per arm, zero failures of any kind. The park pools filled
normally (1.97 GB peer, 1.97 GB local, zero drops, zero no-space) and then nothing was
ever read back.

Cause: **the park pool was too small to hold one conversation.** With `PARK_PEER=1` the
pool is split across two GPUs, so `PARK_POOL_TOKENS=12000` gives 6000 tokens per pool,
while a conversation at `MAX_TOKENS=1024 MAX_TURNS=10` reaches ~13,200 tokens. A prefix
that does not fit in a pool can be parked in part but never fetched back whole.

Llama-3.1-8B had survived only by luck: 15,000 tokens per pool against the same ~13,200
token conversation — a 1.13× margin. It scored 95.5% hit rate for that reason, not
because it was configured any more carefully.

Fixed by sizing the park pool from the conversation length rather than from free HBM
alone (`PARKPOOL_QWEN14B` 12000 → 45000, `MEMFRAC_QWEN14B` 0.80 → 0.72 to make room), and
by a `check_park_pool` guard in `run_models_sweep.sh` that refuses to start an arm whose
pool cannot hold a conversation. The guard rejects both of the old settings and accepts
all three current ones.

Re-run required. The `recompute` arm did not exist for this run either — the arm set is
now Recompute / SGLang / Ours.
