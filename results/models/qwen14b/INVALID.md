# INVALID — the park arm never fetched anything back

Two Qwen3-14B runs are invalid. Both had 1440 turns per arm and zero failures; both had
`fetch_hits = 0`, so the park tier contributed nothing while the figure looked healthy.

| run | park pool | `fetch_hits` / `fetch_miss` | hit rate (Ours vs SGLang) |
|---|---|---|---|
| first | 12000 tok | 0 / — | 46.8% vs 46.6% |
| second | 45000 tok | 0 / 1623 | 45.1% vs 50.6% |

## Root cause (confirmed, not inferred)

**Qwen3's chat template was not append-only**, because the benchmark passed
`chat_template_kwargs={"enable_thinking": False}`. With that flag the template emits
`<|im_start|>assistant\n<think>\n\n</think>\n\n` as the *generation prompt* but renders
history as `<|im_start|>assistant\n<content>` — so turn N's prompt is four tokens longer
than the head of turn N+1's and is **not a prefix of it**. Measured at every turn:
31→27, 106→102, 181→177.

The parked-KV index hashes the exact token sequence, so every lookup missed. The radix
cache degrades gracefully to a shorter match and still scored ~45%, which is why nothing
in either run looked broken.

**Enlarging the park pool was my first hypothesis and it was wrong** — the second run
proves it, since the pool went 12000 → 45000 and `fetch_hits` stayed at 0. The pool
guard added at the time is still correct and still worth having, but it was not the bug.
**Echoing the think block back into the stored reply was my second hypothesis and it was
also wrong** — the template strips `<think>...</think>` from history whenever
`enable_thinking=False`.

## Fix

Do not pass `enable_thinking` at all (`DISABLE_THINKING=0`), and store the reply exactly
as generated (`STRIP_THINK=0`) — with the flag absent the template neither injects nor
strips, so the raw reply renders verbatim and the generated KV stays reusable too.

Verified end to end: `MODEL_KEY=qwen14b ./scripts/sglang/smoke_park.sh` →
`fetch_hits 4 (peer 4)`, PASS.

## Residual caveat for the paper

Thinking is now ON, so thinking tokens consume the `MAX_TOKENS` budget. Qwen3-14B yields
less visible content per turn than a non-thinking model at the same setting and its
conversation grows more slowly, which shrinks the reuse prize. Compare the median context
in `ttft_by_turn.txt` against Llama's before reading the TTFT panel across models.
