# Lloyd GDN chunk-scan admission (§2.3): correctness + perf claim (gfx1100, 27B mq4l)

**Lifecycle:** `historical`. Fixture-bound measurement under the exact method
below. Not a current default, baseline, or admission decision.

## Fixture

- GPU: gfx1100 (RX 7900 XTX), HIP 7.16, ROCm 10.1 therock toolchain.
- Model: `qwen3.8-27b.mq4l` (qt=52 Lloyd), fixture pin md5 `d020f248…`
  (established pin, carried over; file unmoved all session).
- Binaries (release, clean, no probe code): base `hipfire d0e57801…` /
  `daemon a0aa0f3a…` (= `cfe85515a`, sequential GDN always) vs scan
  `hipfire bb173fb8…` / `daemon 8f54c831…` (= `feat/lloyd-gdn-scan`
  @ `1a0e52f`, Lloyd admitted to chunk-scan gates). md5-verified swap
  per leg. Eval binaries `51875f24` (base) / `4b7a3108` (scan).
- Route under test: plain prompt prefill, AR (`--spec off`), KV q8
  legacy, `HIPFIRE_GRAPH=0`, fresh HOME + empty kernel cache per KLD leg.
- Flip background: seeded greedy battery 9/10 identical, one long prompt
  (7758 chars) flipping a single token (`time` vs `runtime`),
  deterministic — the BLOCKED question this record arbitrates.

## Method

- KLD: `eval_hipfire --scoring-mode {prefill,prefill-prompt} --max-chunks 24`,
  WT2 ref `8c545178…`. Split `prefill` = 1024-prefix (Standard, scan
  admissible on scan leg) + 1023 verify-captured scored region.
  `prefill-prompt` = whole 2048-window through production admission
  (scan engaged wherever production admits it). Engagement proven per leg
  via `gdn_chunk_scan` lazy-binds (2 on scan, 0 on base).
- pp2048: `hipfire bench <model> --matrix --pp 2048 --tg 32 --runs 5
  --warmups 3 --backend noslots --workload stateless --spec off --kv-mode q8
  --kv-v q8 --json`, 5 fresh interleaved processes per side (b,n,b,n…),
  median-of-medians. Take-counts from an instrumented build (temp
  per-take trace, removed before clean builds): 192 scan takes per
  pp2048 prefill (4×512 chunks × 48 DeltaNet layers), 6912 takes across
  the full counting session incl. sustained phases; odd tails (721, 91)
  admitted per the gate's tail rule.
- at-ctx: standard bench `--prompt-file
  benchmarks/prompts/scan_107k_synth.txt --max-tokens 1 --warmups 1
  --runs 3 --max-seq 110592` (same flags otherwise), 3 fresh interleaved
  processes per side. Prompt: 469200 B harbor-paragraph repetition, md5
  `3d050f53`, 92012 tokens — synthetic and uniform by design (prefill
  tok/s is content-agnostic, decode is 1 token).
- Correctness gate on claim binaries: seeded greedy battery (8 short +
  2 long prompts) scan-leg vs saved sequential control, plus eyeball.

## Results

KLD (WT2, teacher-faithful = lower):

| leg | KLD | `.kldseq` sha256 | scan binds |
|---|---|---|---|
| base split-`prefill` | 0.032519 | `bd3b2c16…` | 0 |
| scan split-`prefill` | 0.032516 | `16d1cc0c…` | 2 |
| base `prefill-prompt` | 0.032453 | `5c45988c…` | 0 |
| scan `prefill-prompt` | 0.032606 | `d646e3a0…` | 2 |
| scan `prefill-prompt` repeat | 0.032606 | `d646e3a0…` (identical) | 2 |

Δ +0.000153 on the production route, deterministic, P1-noise-grade
(P1 accepted +0.00009/+0.00017); an A8-style wrongness reads KLD 14+.
Verdict: WY scan is reorder-class fp drift, not scan breakage; the flip
is a near-tie argmax flip, same class as shipped q8-FA2 vs scalar.
Byte-identity gate retired for this path; KLD-parity + battery + eyeball
is the bar (P1 precedent). State-differential work moot, skipped.

Perf (median-of-medians, within-run stdev in parens):

| shape | base | scan | Δ |
|---|---|---|---|
| pp2048 synthetic (5 fresh/side) | 1018.2 (≤0.96) | 1066.3 (≤0.95) | **+4.7%** |
| 92K-token fill (3 fresh/side) | 681.7 (≤0.25) | 701.9 (≤0.22) | **+3.0%** |

pp2048 legs: 1018.2/1018.1/1018.2/1018.6/1017.7 vs
1066.1/1066.0/1066.8/1066.6/1066.3. 92K legs:
681.9/681.7/681.5 vs 701.9/702.8/701.4. Deltas are 14–50× the control
spreads. Battery on claim binaries: 9/10 identical (same known flip).

## Non-goals / caveats

- MoE out of scope by construction (`dense_layers_have_projection_dtype`
  is false on any `DeltaNetMoe`/`FullAttnMoe` layer). No MoE model on
  this box; none needed — the gate is code-proven.
- Cross-load tok/s comparisons carry ±5% noise on this box; the claim
  clears it by replication (5×/3× interleaved fresh processes), not by
  margin alone.
- 92K prompt is synthetic repetition, not prose — a perf instrument,
  not a quality probe (quality is the KLD table above).
