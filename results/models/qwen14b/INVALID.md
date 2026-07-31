# INVALID — the park arm still never fetched anything back

Second attempt, with the park pool enlarged from 12000 to 45000 tokens (22,500 per pool,
comfortably above the ~13,200-token conversation). **`fetch_hits` is still 0**, over
`fetch_miss = 1623` lookups.

So the pool size was not the cause — or not the only one. What the counters say:

| | | |
|---|---|---|
| park pools | 7.37 GB peer + 7.37 GB local | filled to capacity, so parking DID happen |
| `fetch_miss` | 1623 | every lookup ran and matched nothing |
| `fetch_nospace`, `dropped` | 0 | not a capacity or eviction problem |
| hit rate | Ours 45.1% vs SGLang 50.6% | Ours ≈ plain radix; the park tier added nothing |

The park index stores a block under a rolling hash of the **exact token sequence**, so
turn N's prompt must be a token-exact prefix of turn N+1's. A single token inserted or
removed anywhere in the history makes every later lookup miss. The radix cache degrades
gracefully in the same situation — it just matches a shorter prefix — which is why the
baselines still look healthy while the park arm reports nothing.

Prime suspect: Qwen3's chat template inserts an empty `<think>\n\n</think>` block into the
assistant slot when generating and strips thinking content when re-rendering history. If
those disagree, the prompt is not append-only across turns.

`benchmark/check_prefix_continuity.py` settles this offline against the tokenizer in
seconds, rather than by another two-hour sweep:

```bash
python benchmark/check_prefix_continuity.py --model /home/uhmturks/hf_models/Qwen3-14B
python benchmark/check_prefix_continuity.py --model /home/uhmturks/hf_models/Llama-3.1-8B-Instruct
```

The Llama-3.1-8B run in `results/exp1/sharegpt_p60000_c8_m1024` is unaffected: 1211 fetch
hits against 306 misses.

Also noted: the `parked_tokens` counter reads 0 in both the working and the failing run —
it is incremented only on the fetch-insert path, so it does not report what its name
suggests and cannot be used to check whether parking occurred. `occupancy()` (cumulative
writes, saturating) is the field that shows parking happened.
