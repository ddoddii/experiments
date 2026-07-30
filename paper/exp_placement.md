# Exp 1 / Exp 2 — placement 실험 설계

두 실험은 서로 다른 질문에 답한다. Exp 1은 **총량** 질문("unified placement가 CPU DRAM을 얼마나
줄이나"), Exp 2는 **메커니즘** 질문("peer GPU로 옮기는 것이 실제로 기여하나")이다. Exp 2 없이는
Exp 1의 win이 전부 local park pool 덕분일 수 있고, 논문의 주장인 peer 배치는 측정되지 않은
상태로 남는다.

공통: server17, A6000×4, Llama-3.1-8B, 2P2D, BFCL v3 multi-turn 200 items,
`C=16`, `TOOL_DELAY_SEC=3`, `PREFILL_MAX_TOTAL_TOKENS=60000`.

`TOOL_DELAY_SEC>0`은 옵션이 아니다 — tool-call idle window가 KV가 식어서 축출되는 구간이고,
0이면 파킹할 대상 자체가 없다. 풀 크기도 축출이 실제로 일어날 만큼 작아야 한다. 둘 중 하나라도
틀리면 세 arm이 동일해지고 실험은 아무것도 측정하지 않는다.

---

## Exp 1 — GPU-first unified placement vs incumbents

```bash
CONCURRENCY=16 TOOL_DELAY=3 PREFILL_MAX_TOTAL_TOKENS=60000 \
  ./scripts/sglang/run_exp1_placement.sh
# -> results/exp1/p60000_c16_d3/table.md
```

### Arms

| arm | 축출된 KV의 운명 | 서버 config |
|---|---|---|
| `radix` | **폐기** (local GPU pool만) | `PARK_NO_HICACHE=1 IDLE_KV_PARKING=0` |
| `hicache` | CPU DRAM(L2) → file(L3). **SGLang 기본 구성** | `HICACHE_WRITE_POLICY=write_through_selective HICACHE_STORAGE_BACKEND=file` |
| `hicache_wb` | CPU DRAM만 (L3 미사용) — page cache 기여를 분리하는 보수적 비교군 | `write_back` + `backend=none` |
| **`park`** | **여유 최대 peer GPU → 부족 시 CPU DRAM overflow** | `IDLE_KV_PARKING=1 PARK_PEER=1 SGLANG_KV_PARK_HOST_OVERFLOW=1` |

`hicache`를 SGLang 기본값으로 둔 이유: 이기려는 대상은 "출하되는 그대로의 incumbent"다.
`hicache_wb`는 "win의 일부가 page cache 아니냐"는 반론에 대한 답 (AnonPages는 여전히 76.0 GB).

**`PARK_PEER=1`이 제안의 본체다.** 이걸 안 켜면 각 prefill은 자기 GPU에만 파킹하므로
"peer GPU 우선"이 아예 실행되지 않는다. 켜면 각 P가 GPU0·GPU1 양쪽에 풀을 갖고, **풀 토큰 수는
절반으로 줄여** GPU당 park HBM 총량을 `park_local`과 동일하게 맞춘다 — 안 그러면 peer arm이 단순히
capacity가 2배라서 이기고, 그건 주장이 아니다.

### 지표 → 출처

| 지표 | 컬럼 | 수집 |
|---|---|---|
| Peak host RSS | `proc_rss_mb` peak | `sys_mem_breakdown.py` |
| Peak host page cache | `file_cache_mb` peak | 동일 |
| Peak total host footprint | `rss_plus_cache_mb` peak | 동일 (신규 컬럼) |
| GPU별 peak HBM | `gpu0..3_hbm_mb` peak | 동일 (신규 컬럼) |
| 전체 GPU HBM | `gpu_hbm_mb` peak | 동일 |
| Peer-GPU KV bytes | `park_peer_gb` | `park_location_sampler.py` (신규) |
| CPU overflow KV bytes | `host_gb` | 동일 |
| **KV residency 시계열** | `serving_gb` / `park_local_gb` / `park_peer_gb` / `host_gb` / `dropped_gb` | 동일 (신규) |
| Median / P95 / P99 TTFT | `ttft_p50_s` / `ttft_p95_s` / `ttft_p99_s` | bench summary (p95 신규) |
| Overall throughput | `overall_throughput_tok_s` | 동일 |
| Goodput | 에러/빈 응답 turn 제외 output tok/s | `collect_arm_metrics.py` (신규) |
| Prefix reuse ratio | `reuse_ratio` | `phase0_metrics_scraper.py --delta` |
| Recomputed tokens | `uncached_tokens` | 동일 |
| Host fetch 수 | `fetch_host_hits` | park 텔레메트리 (신규) |
| Peer-GPU fetch 수 | `fetch_peer_hits` | 동일 (신규) |

`collect_arm_metrics.py`가 arm별 4개 산출물(bench json / mem csv / park csv / metrics delta)을
한 행으로 축약하고 markdown 표를 바로 출력한다. 메모리는 **peak**을 쓴다 — 주장은 commitment,
즉 다른 프로세스가 쓸 수 없었던 용량이고 그건 high-water mark가 결정한다. mean도 같이 찍어서
spiky한 arm이 드러나게 한다.

### 읽는 법

- `park` vs `hicache`: **CPU DRAM commitment 제거**, TTFT 동률 이상
- `park` vs `radix`: 재사용이 recompute보다 싸다 (recomputed_tokens로 확인)
- `park`의 `park_peer_gb` ≫ `host_gb`: **GPU-first가 실제로 작동**
- 세 arm의 `prefix_reuse_ratio`가 비슷: **KV 총량이 유지됨** (목표 문장의 전반부)

---

## Exp 2 — 부하가 한 prefill에 집중될 때 peer 배치의 기여

```bash
SKEWS="0.5 0.9" CONCURRENCY=16 ./scripts/sglang/run_exp2_imbalance.sh
# -> results/exp2/skew0.9_p60000_c16/table.md
```

### skew를 강제하는 이유와 방법

균등 routing에서는 P0·P1이 같은 속도로 차므로 **P1에 빌릴 headroom이 없고**, 메커니즘은 구조적으로
보이지 않는다. 그 상태의 null 결과는 메커니즘에 대한 정보가 아니다.

그래서 `ROUTER_MODE=skew`로 **prefill마다 라우터를 하나씩** 띄우고(`:8000`→P0, `:8001`→P1)
클라이언트가 **세션 단위**로 가중 배분한다(`SKEW=0.9` → 90%가 P0). request 단위가 아니라 세션
단위인 이유: 한 대화의 turn들이 서로 다른 prefill에 흩어지면 어느 쪽에도 재사용할 prefix가 없다.
decode는 양쪽 라우터가 공유하므로 **prefill 쪽만** skew된다.

배분은 Bresenham으로 결정적이다 — 200개에서 요청한 비율이 정확히 나오고(0.9 → 180/200),
두 arm이 완전히 같은 세션→prefill 할당을 보므로 **paired 비교**가 된다. 해시/RNG는 클러스터링
때문에 실제 skew가 요청값과 달라진다.

### Arms

| arm | 무엇을 분리하나 |
|---|---|
| `radix` | 재사용 없음 → full re-prefill 비용 |
| `hicache` | CPU DRAM으로만 도피 → PCIe fetch + host commitment |
| **`park_local`** | **파킹은 켜되 peer 배치 OFF** — P0의 유일한 park 대상이 포화 중인 자기 GPU |
| **`park`** | peer 배치 ON — P0가 P1의 유휴 HBM으로 |

**`park_local`이 핵심 비교군이다.** 이 arm 없이는 `park > hicache`가 전부 local park pool 때문일
수 있고, 논문이 주장하는 peer 메커니즘의 기여는 측정되지 않는다. `park_local`도 host overflow는
켠 채로 둔다 — 안 켜면 "peer가 없어서" 지는 게 아니라 "탈 곳이 아예 없어서" 져서 비교가 오염된다.

### 이 실험이 답하는 것

| 질문 | 지표 |
|---|---|
| Local eviction 감소? | `dropped_gb` (cumulative) |
| CPU overflow 감소? | `host_gb` peak, `fetch_host_hits` |
| Re-prefill 감소? | `recomputed_tokens`, `prefix_reuse_ratio` |
| Tail TTFT 감소? | `ttft_p95_s`, `ttft_p99_s` |
| **skew가 실제로 발생했나?** | `occ_*.csv` — P0 포화 / P1 headroom |

마지막 줄이 없으면 null 결과를 해석할 수 없다. `kv_occupancy_timeseries.py`를 arm마다 같이
돌려서 P0가 정말 포화했고 P1에 여유가 있었음을 먼저 보인 다음에 arm 비교를 읽는다.

`SKEWS="0.5 0.9"`로 sweep하면 "균등에서는 차이 없고 불균형에서 벌어진다"가 한 그림에 들어간다 —
이게 메커니즘 주장으로서 단일 skew 결과보다 강하다.

---

## 실행 전 확인

- `./scripts/sglang/stop.sh`는 이번에 새로 작성했다. 이전에는 **0 바이트 빈 파일**이어서
  runbook이 호출해도 아무 일도 하지 않았고, 다음 arm이 이전 arm의 서버를 물려받을 수 있었다.
- `sys_mem_breakdown.py`가 시작 시 pid 수를 검사한다. 2P2D는 ~150 프로세스이므로 `n_pids < 8`이면
  경고한다. **기존 `results/mem/bd_park.csv`·`bd_wts.csv`·`bd_l2.csv`는 `n_pids=4`로 수집돼서
  per-process(`proc_rss_mb`, `proc_anon_mb`) 컬럼이 무효다** — system-wide `anonpages_mb`,
  `mem_avail_mb`만 유효하고, `paper/evaluation.md`의 A1 표는 그 컬럼만 인용하므로 성립한다.
  Peak host RSS는 Exp 1에서 새로 수집한다.
- c4 run이 죽은 원인(`SGLANG_KV_PARK_HOST_OVERFLOW=1`의 동기 cold pin 의심)이 미해결이다.
  Exp 1/2는 host overflow를 켜므로 **영향을 받는다.** 첫 arm이 KVTransferError로 죽으면
  `HOST_OVERFLOW=0`으로 재시도해서 원인을 확정할 것.
