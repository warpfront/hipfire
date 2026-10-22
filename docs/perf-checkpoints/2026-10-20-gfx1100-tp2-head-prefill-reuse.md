# Dense TP2 vocabulary-head sharding and prefill scratch reuse

**Lifecycle:** `historical`.
**Disposition:** measured, default-off developer experiments; no admission or
product-default claim. Neither route changes quantization or TP sharding of the
trunk. This is not TP1-to-TP2 scaling evidence.

## Identity and method

- Date: 2026-10-20; W7900 + W7900 Dual Slot, both exact gfx1100, HIP 7.15.
- PCI: `0000:53:00.0`, `0000:9e:00.0`; UUIDs
  `GPU-dee503dba9717041`, `GPU-ecb57a743c42951e`.
- Source: beta `66db8cb528444b28751bee98fba2e24cfc024acc` plus the experimental
  delta, dirty-diff MD5 `33484bb6c66e458daef64badb5df0ea2` in every timed arm.
- Model: canonical `qwen3.8-27b.mq4-xts`, 14,987,185,152 bytes, SHA-256
  `3e38ccbae3776470eb5a89344d300e9279d6b9ab6c31fd40ca1758c4f7c6f8ae`;
  checked with `scripts/check_fixture.sh`.
- CLI MD5 `8c42aa409b41ffc3fe907b75c5474b2e`;
  daemon MD5 `2048c34c515d18a7129963965b17d945`.
- Same candidate binary for both arms. A disables both
  `HIPFIRE_TP_SHARD_LM_HEAD` and `HIPFIRE_TP_PREFILL_REUSE`; B sets both to `1`.
- TP2, Q8/Q8 VMM, `max_seq=16384`, pure AR, thinking off; explicit
  `--backend noslots --workload stateless`. All six points per workload are
  fresh processes, ordered **A,B,B,A,A,B**, with one measured request each.
- `HIPFIRE_DPM_WARMUP_SECS=10`; three discarded forwards per short-prompt
  decode process, two per long-prefill process. Default trunk hipGraphs stay
  enabled. Tracing/profiling is off for every timed sample.
- Per-device locks held by the normal daemon. An external observer samples
  both PCI devices' `mem_info_vram_used` every 100 ms. Reported peaks are
  **sampled process-run peaks**, not exact allocator/live-tensor maxima or
  decode-only residency. They include system/runtime VRAM.

The timed implementation already keeps head gather outside capture. Later
changes harden only the numerical oracle (independent unsliced-head reference,
growth/shrink and finite checks) and correct trace-only segment labels;
the measured arithmetic and scheduling remain unchanged.

## Production-path measurements

Decode fixture: `benchmarks/prompts/mq4v2_k5120_long_ar.txt`, MD5
`9f64b7bc79c38aaa65cccc511b167390`, 126 actual prompt tokens, 512 generated
tokens per measured request. This short prompt is a decode fixture, not a
prefill-throughput claim.

| Arm | Decode samples (tok/s) | Median |
|---|---|---:|
| A: both off | 53.1, 52.9, 52.9 | 52.9 |
| B: both on | 53.9, 54.0, 53.8 | 53.9 |

Median decode delta: **+1.89%** under this fixture. The sample ranges do not
overlap, but this remains a small, workload-specific effect, not a broad floor.

Prefill fixture: `benchmarks/prompts/glimmer_prefill_4096.txt`, MD5
`243e2c64fa7db53686dc0a1d58099654`, **4,408 actual Qwen prompt tokens**,
64 generated tokens per request. The file's name is not its Qwen token count.

| Arm | Prefill samples (tok/s) | Median | First-token samples (ms) |
|---|---|---:|---|
| A | 2112.1, 2088.0, 2088.9 | 2088.9 | 2087.0, 2111.1, 2110.2 |
| B | 2104.9, 2091.9, 2078.3 | 2091.9 | 2094.1, 2107.2, 2121.0 |

**No material prefill-speed gain or regression is established.** The +0.14%
throughput median is smaller than the observed sample spread; first-token
timing does not establish a win either.

### VRAM: median sampled peak over three fresh processes

| Workload / device | A bytes | B bytes | B minus A |
|---|---:|---:|---:|
| Decode / W7900 | 9,666,056,192 | 9,666,060,288 | +4 KiB |
| Decode / W7900 DS | 9,440,419,840 | 9,440,464,896 | +44 KiB |
| 4,408-token prefill / W7900 | 10,024,136,704 | 9,927,663,616 | -92.00 MiB |
| 4,408-token prefill / W7900 DS | 9,985,875,968 | 9,889,443,840 | -91.96 MiB |

The head uses non-owning row views of **existing replicated weights**, plus
the existing per-rank logits buffers. It does not physically shard or duplicate
the head storage. Prefill caching retains PBS owners rather than returning
them to the allocator pool between calls; the measured footprint is lower on
this repeated-prefill fixture, not an estimate derived from logical tensor size.
No general memory-saving percentage is inferred.

Isolated exploratory controls, **one process each, not promotion evidence**:
cache-only prefill 2085.0 tok/s, sampled peaks 9,927,667,712 / 9,889,402,880
bytes; head-only decode 53.6 tok/s. These support attributing the footprint
reduction to reuse, but do not establish additive/synergistic speed gains.

## Correctness and rejected implementations

- Initial head-only synthetic check: 16 input vectors, every logit bit equal.
  Its 0.992 to 0.714 ms timing used the original kernel-copy gather and is
  **not** a throughput claim for the final product implementation.
- **Rejected:** a plain peer-read gather inside per-rank whole-token graphs
  retained stale tail-vocabulary logits from step 2 onward. The first half
  stayed exact while the second half reached max absolute error 14.72. All
  32 argmax ids still matched: token-only checks would have missed this.
  Replacing the captured gather with HIP peer-copy alone also failed the
  actual-state oracle. Neither implementation is retained as a selectable arm.
- Final route retains the 128-boundary trunk graph, then enqueues rank-local
  heads and a system-release event-ordered `hipMemcpyPeerAsync` outside capture.
  Segmented graphs capture local head compute only; gather is outside capture.
  A guard refuses gather if any rank is capturing.
- TP2 baseline versus candidate: **32 x 248,320 logits byte-identical**;
  dump MD5 `532b0a5a419e5c7567dc0f9850d0bc2e`. This held for the final
  whole-token route, the direct control, and segmented replay including
  grow-then-shrink prefill. The actual-state oracle additionally compares
  against the original, unsliced head WeightRef, without using the row adapter
  as its reference. Repeated short prefills after a larger PBS are exact.
- These numerical runs use Q8 **legacy KV** and **FP32 recurrent state**;
  do not substitute them for the Q8-VMM/Q8-state product checks below.
- Product `serve_harness.py --mode chain`, Q8 VMM, default recurrent state,
  greedy, thinking off, 1024-token cap: **5/5 stop**, 586 generated tokens,
  **5/5 byte-identical outputs and request hashes**, no empty/attractor/stream
  errors. The decoded answers were read. Concatenated assistant MD5:
  `6474e8cde9a30f6532e30a61ddfd7b37`. `cached=0` throughout: this is not a
  prefix-cache reuse claim. It does exercise successive prefill calls.
- **Existing TP1/TP2 oracle is NOT green:** both the disabled control and
  candidate exceed its unchanged `<3e-3` relative-logit limit with worst
  `1.029e-2`, despite 32/32 argmax matching. The metric is max absolute logit
  difference divided by maximum absolute reference logit, not relative L2.
  Exit 101 is retained; the threshold was not relaxed. Exact TP2-to-TP2 parity
  does not resolve that independent baseline discrepancy.
- Library tests: 388 passed, 24 ignored. GPU-free pre-commit checks passed,
  including a separately archived scratch-only staged snapshot. No TP2 PM4
  validation or cross-SKU promotion is claimed.

## Reproduction and local raw receipts

Set `HIPFIRE_DEVICES=0,1`, `HIPFIRE_DPM_WARMUP_SECS=10`, both experiment
knobs to the chosen arm, and use the same binary and byte-identical prompts:

```sh
hipfire bench ~/.hipfire/models/qwen3.8-27b.mq4-xts \
  --tp 2 --runs 1 --warmups 3 --max-tokens 512 --max-seq 16384 \
  --kv-mode q8 --kv-backend vmm --spec off \
  --backend noslots --workload stateless \
  --prompt-file benchmarks/prompts/mq4v2_k5120_long_ar.txt --json
```

For prefill change the prompt to `glimmer_prefill_4096.txt`, warmups to 2 and
output cap to 64. Repeat in the declared fresh-process order, not as six
resident requests. Raw JSON, full VRAM traces, identities and oracle logs are
retained locally under `tmp/tp2-head-prefill-evidence/` (`final-decode-*`,
`final-prefill-*`, `external-gather-chain`, `parity-*`). This is a **local
discovery pointer**, not a publicly uploaded artifact URL.
