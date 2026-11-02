# Qwen4 (Qwen3.8-Flash-Next mq6q8-pleq8) native MTP decode — autoresearch campaign (gfx1151) — 2026-09-28

**Lifecycle:** `historical`

**Disposition:** measured local deltas on `gfx1151`, landed on branch
`autoresearch/improve-autoregressive-decode-performance-for-qw-20260926`
(`30f30629a` → `1b8fc7f9a`). Not a G5 admission, not a retained-replay
certification, not a cross-architecture result, and not a product speed-floor
update. The GPU was shared with external `llama-server` processes and another
agent's jobs for the whole campaign; late runs drifted ~4% slower at unchanged
code (see Method).

## Fixture

- Host `halo`, gfx1151, 128 GB UMA, HIP 7.2.
- Model `~/.hipfire/models/qwen3.8-flash-next.mq6q8-pleq8.hfq` (the Flash-Next
  pin in `AGENTS.md`, size `125467331096`, sha256 `58fb4f58…4aed`).
- Prompts `benchmarks/prompts/lru_cache_pep8_strict.txt` (md5
  `df5dedc8040ce70ba55080c4548e6024`, 72 greedy tokens to EOS) and
  `benchmarks/prompts/prose_river_short.txt` (md5
  `07a7880965142971dbb3cc7493f8fb94`, 128 tokens) — the same pair as the
  2026-09-23 MTP measurement.

## Method

- `autoresearch.sh` on the campaign branch: build `hipfire-daemon`, one fresh
  daemon under the GPU lock, load `max_seq` 2048, KV q8, `mtp_mode` on,
  `mtp_k` 3, `HIPFIRE_GRAPH=0 HIPFIRE_AR_GRAPH=0 HIPFIRE_CASK_OFF=1`; per
  prompt one warmup and three greedy generates. Metric: geometric mean of the
  two median client-side decode rates (token arrival times).
- Gate (every run): each prompt's greedy MTP token ids equal the AR greedy ids
  recorded with `mtp_mode` off on the segment's first build.
- Few-row exactness: `qwen4_rows` (new `lab` example) runs one `forward_chunk`
  of N rows and N single-row `forward_token`s from the same prefix and compares
  all logits bitwise; every kept kernel change was checked at N = 2..8.
- Breadth: 9 more committed prompts (fiction, merge sort, humaneval, mixed
  code/prose, trains, code edit, bare factual, glimmer 1024, tool system) at
  200 tokens: MTP ids equal AR ids on all; MTP/AR speedup 1.12x-1.74x.
- Serve route: `scripts/serve_harness.py --mode battery --sampling greedy
  --mtp on --thinking off --max-tokens 400 --kv q8 --max-seq 2048`, daemon md5
  `db7d7b891e86cb55c50f3b9c72983bef`: 5/5 turns `finish=stop`, no
  runaway/empty/attractor; answers read and correct (merge function, 210
  miles, axial tilt, four-sentence story, five practices).
- Noise: late in the campaign the unchanged HEAD measured 43.2 against its
  own 45.3 an hour earlier while external processes kept the GPU at 100%
  busy; comparisons near the noise floor were repeated back-to-back.

## Result

| | start (run 303 rerun, `30f30629a`) | end (run 324, `1b8fc7f9a`) |
|---|---:|---:|
| MTP decode, code | 28.1 | **55.2** |
| MTP decode, prose | 29.3 | **37.2** |
| geomean | 28.7 | **45.3** (+58%) |
| AR decode on the same prompts | ~33 | ~33 |

## What changed

- **Few-row verify made exact and cheap.** A 2..8-row forward now computes
  every row bitwise as the single-row decode route does: HC reads use the Q8
  decode copies with row-looped staged kernels, the LM head uses the staged
  Q8 kernel over all rows, MQ6 few-row kernels cover 2..8 rows, the fused HC
  write+norm (`hyper_write_norm_f32`) takes a row grid, and the MoE down runs
  the decode kernel per slot (the grouped tile ran near-empty). 4-row forward
  82 → 52.8 ms (single-row decode ~31 ms).
- **Replay-free rollback.** The GDN recurrence writes every verify row's state
  into the free half of a per-layer ring (the live state is a slot view), the
  convolution input rows are captured, and QSA marks, PLE convolution and PLE
  token history are rebuilt for the kept prefix; the MTP head keeps its own
  draft steps (`truncate_retain`). A rejected suffix costs ~0.4 ms instead of
  a 32-60 ms re-forward; no snapshot copy of the 113 MB recurrent state.
- **Adaptive route and depth.** Per-depth decayed draft agreement picks the
  depth maximizing expected tokens per window cost, or the interleaved route.
- **Draft head.** MQ2 copy of the LM head ranks the vocabulary; its top 8 are
  re-scored against the Q8_0 rows, so drafts are the Q8_0 head's argmax.
- **SSD PLE rows.** `RowStore` reads a request's scattered pages concurrently
  and gains lease-less `warm` tickets; each MTP window warms its seed's and
  drafts' rows while drafting (cold-row prompt: verify PLE wait 453 → 25 ms
  per request). AR decode also benefits from the parallel reads.
- Grouped MoE prefill round trips cover live rows only.

## Rejected (measured)

MQ6 few-row kernel variants (2 rows per wave, float4 x, occupancy 16,
up-front group loads, LDS-shared x): none beat ~170 GB/s. MQ3 draft head
without re-scoring: code acceptance loss. Whole-MoE indexed route for few rows:
no gain over the grouped gate/up. Self-calibrating route costs and a
re-measured cost table: flat or worse (K=1 overvalued).
