# INVALID RUN — do not use these numbers

Every arm failed 92–93% of its turns with **HTTP 503 `server_selection_failed`
"No available servers"** (and, in the skew arms, connection-refused on `:8001`).

Cause: `start_2P_2D.sh` launched the router with `&` and returned immediately, so the
benchmark began before the router had registered any worker. The handful of turns that
succeeded are the tail of the run, after the router finally came up — they are not a
random sample of the workload, so every percentile, ratio and residency figure computed
from this directory is a measurement of "whatever ran once the system recovered".

Fixed by the router readiness gate in `start_2P_2D.sh`, which issues a real
`/v1/chat/completions` request and refuses to return until it gets a non-503 (`/health`
is not sufficient: a router has previously answered `/health` 200 with every circuit
open). `collect_arm_metrics.py` now prints the turn/error counts first and refuses to
present a table with >5% failures as a result.

The one thing this run does establish, as a smoke test rather than a result: in
`results/exp1/.../parked_park.csv` the park arm placed 3.9 GB onto a **peer** GPU and all
14 fetch hits came from the peer, so `PARK_PEER=1` does place and fetch across GPUs. In
the Exp 2 arms peer bytes stayed 0, which is consistent with there being almost no load —
with both GPUs near-idle the selector has no pressure signal and falls back to the
first-listed (own) pool.

Also: three bench files from this run landed in `results/sglang_hicache/Qwen3-14B/`
despite Llama-3.1-8B being served, because `/get_model_info` was unreachable while the
router was down and `MODEL` fell back to its env default.
