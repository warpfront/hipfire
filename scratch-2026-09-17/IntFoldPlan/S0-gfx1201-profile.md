# S0 — gfx1201 per-kernel prefill split (measured)

Date: 2026-09-18. Worktree `/home/kaden/ClaudeCode/warpfront/wt-lloyd`, HEAD `684f6de45`.
Plan: `docs/plans/2026-09-18-gfx1201-gdn-chunked-scan.md` §1 (Slice S0) / §1.3 (Gate G0).
**No source edits.** Profiler binary built from HEAD (`cargo build --release -p saddle-lab
--example profile_prefill_qwen35 --features deltanet`); the tree's one uncommitted file
(xdna doc edit) was not touched.

## 0. G0 verdict: SCAN SLICE DEFERRED (run P and T first, then re-measure)

| G0 input | Measured | Bar | Met |
|---|---|---|---|
| `t_serial` = serial `gated_delta_net_q8_batch_seq` µs/launch at pp512 | **543.6 µs** (A/B/C mean 544.23/542.97/543.47; band 524–545 over 7 same-config base runs, 542–554 across fp8v2/iu4 arms; rocprof GPU timestamps 534.9) | ≥ 600 authorize / 400–600 defer / < 400 abandon | **DEFER band** |
| `f_gdn` = GDN-scan share of matched bench wall, fastest admitted GEMM arm | **7.4 % @pp512, 6.8 % @pp8192** (scan-only vs measured base-fp8 serve wall; iu4-arm wall unmeasured — see §7, bar clears on every arm regardless) | ≥ 3 % | **yes** |

Authorization needs **both**; `t_serial` sits 56 µs below the authorize line with the bar
missed by 10 %. Per the plan: slice P (exact preamble fusion) and slice T (exact knob sweep)
run first; slice S is authorized only if the residual still clears the same bar.
Abandon is off the table (543 ≫ 400). Scan upside if it hits the plan's derived low end:
(544−190) µs × 768 launches ≈ 272 ms @pp8192 ≈ 4.4 % of wall — worth pursuing after P+T.

Halo reference for scale (`halo-profile`, gfx1151, iu4 route, pp8192): GDN 990.77 µs / 6.3 %,
conv 251.50 / 1.6 %, FA2 7760.75 / 8.2 %. gfx1201 serial GDN is already **1.82× faster**
(544.2 µs), conv 1.43× (175.7 µs), FA2 2.93× (2647.9 µs).

## 1. Method and fixture (exact receipts)

- Fixture: `scripts/check_fixture.sh --sha /home/kaden/.hipfire/models/qwen3.8-27b.mq4-xt` →
  `OK (14987185152 B, sha256 verified)`, resolves to
  `/home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq`.
- Box: gfx **ordinal 2** only (`ROCR_VISIBLE_DEVICES=2`), `HOME=/home/kaden/.hipfire-homes/ab2`
  (whose `.hipfire/models` symlinks to `/home/kaden/.hipfire/models`),
  `HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab2/.hipfire_kernels`.
  Ordinals 0/1/3 untouched (verified idle peeks only; coord reply sent to Fp8StageGate4).
- Env every run: `HIPFIRE_GRAPH=0 HIPFIRE_NORMALIZE_PROMPT=0 HIPFIRE_LLOYD_GFX12=1`
  (the last is load-bearing: without it gfx1201 falls back to per-token `forward_scratch`).
- Command per cell: `profile_prefill_qwen35 <model> --prefill {512,8192} --warmup 1 --kv-mode q8`,
  three GEMM arms: (a) default fp8 s2bt8; (b) `HIPFIRE_GFX12_MQ4V2_FP8_V2=1
  HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM=128x128`; (c) `HIPFIRE_GFX11_MQ4V2_IU4=1`.
- `--warmup 1` kept per the plan. Correction to the plan's honesty note §1.1(2): this binary
  does **not** average warmup+profiled — the table aggregates only the single profiled pass
  (`Captured N profile entries`, N = exactly one pass's launches). Warmup host times of
  11–27 ms are enqueue-only (no host sync on the warmup path); warmups of seconds are the
  same path plus per-process kernel recompile (5 kernels print `no hash file, recompiling`
  in **every** process; warmup-only cost, profiled numbers unaffected).
- Raw evidence: `GdnPlan/slice0/{base,fp8v2,iu4}-pp{512,8192}.txt`,
  `base-pp512-{r2,A,B,C}.txt`, `{fp8v2,iu4}-pp512-r2.txt`, `rocprof-pp512/` (trace DB +
  the run's own profiler table), `bench-{iu4,base}.json`/`.stderr`,
  `bench-iu4-probe.*`, `bench-cache-probe/` (daemon kernel inventory), `bracket/tele-{A,B,C}.txt`.
- Model/config as profiled: dim 5120, 64 layers (48 LA + 16 FA), heads 24, kv_heads 4,
  `prefill_chunk: requested=512 admitted=512` (all prefill runs chunk at 512 rows).

## 2. Device identity

- `GPU dev 0: gfx1201 (34.2 GB VRAM, HIP 7.15)` (post-visibility index; ROCR ordinal 2 =
  rocm-smi GPU index 3, UUID `85289909a86cc63` — rocm-smi and ROCR enumerate differently,
  UUIDs are authoritative, all telemetry below is that card).
- rocminfo: **64 CUs**, engine max 2350 MHz level (+boost; 3126 MHz observed on a sibling
  card under load), 32 GiB GDDR6 (≈640 GB/s nominal product spec — **not** re-measured in S0).
- Auto clocks (`Performance Level: auto`); DPM sclk levels {500, 2350} MHz + boost, mclk
  levels {96 … 1124} MHz. Short-run clock ramp is the dominant variance source (§9).
- Junction temp during bracketed runs: 54–62 °C edge/mem in step — nowhere near throttle;
  heat is excluded as a correlate with measurements, not assumed away.

## 3. Per-kernel tables (isolated-launch µs; shares are of the isolated sum, not wall)

us/call = total/calls. `%` = share of that run's isolated total. Every kernel above 0.5 %
shown; the tail (rmsnorm/deinterleave/rope/sigmoid_mul/gemv/singletons) is folded into
`other` in the group table (§4). pp512 base column = mean of bracket runs A/B/C;
fp8v2/iu4 pp512 = mean of 2 runs; pp8192 = single runs of 768 calls each (self-averaged).

### pp512 (1 chunk; 48 LA + 16 FA layers)

| kernel | base (mean, n=3) | fp8v2 (mean, n=2) | iu4 (mean, n=2) |
|---|---|---|---|
| `gemm_gate_up…` (64) | 1854.21 | 1647.82 | — |
| `gemm_…_residual…` (128) | 708.94 | 614.97 | — |
| `gemm_mq4g256v2_residual_mmq_iu4_full_set` (368) | — | — | 290.35 |
| `gemm_mq4g256v2_residual_mmq_iu4_full_add` (128) | — | — | 381.75 |
| `quantize_int4_mmq_ds128` (256) | — | — | 103.12 |
| `gemm_qkvza…` (48) | 868.01 | 823.10 | — |
| **`gated_delta_net_q8_batch_seq` (48)** | **543.56** | **553.31** | **542.06** |
| `gemm_qkv…` (16) | 810.73 | 700.97 | — |
| `fused_silu_mul_mq_rotate_awq_batched` (64) | 184.56 | 183.95 | 186.43 |
| `pack_f32_to_fp8_mq4v2_gfx12` (256) | 40.97 | 40.63 | — |
| **`conv1d_silu_split_f32_n` (48)** | **177.09** | **177.97** | **177.92** |
| `fused_rmsnorm_mq_rotate_awq_batched` (128) | 32.22 | 32.55 | 32.20 |
| **`attention_q8_0_fa2_gqa_gfx1201` (16)** | **237.03** | **240.60** | **235.81** |
| `gated_norm_f32_batched` (48) | 60.41 | 60.03 | 58.92 |
| `fused_qk_l2_norm_scale_interleave_f32_batched` (48) | 36.56 | 37.13 | 36.41 |
| `rotate_x_mq_awq_batched` (64) | 23.61 | 23.79 | 23.96 |
| `fused_sigmoid_alpha_gate_f32_batched` (48) | 9.76 | 9.96 | 9.69 |
| isolated total | 339.3 ms | 310.7 ms | 246.9 ms |

Repeat deltas (2nd vs 1st) ≤0.5 % on every row above 0.5 % (fp8v2 GDN 553.76→552.85;
iu4 GDN 542.02→542.09); base A/B/C spread ±0.12 % (§9).

### pp8192 (16 chunks; every count below is exactly 16× its pp512 count)

| kernel | base | fp8v2 | iu4 |
|---|---|---|---|
| `gemm_gate_up…` (1024) | 1871.02 | 1660.42 | — |
| `gemm_…_residual…` (2048) | 716.05 | 619.16 | — |
| `…_iu4_full_set` (5888) | — | — | 294.87 |
| `…_iu4_full_add` (2048) | — | — | 386.36 |
| `quantize_int4_mmq_ds128` (4096) | — | — | 105.09 |
| **`attention_q8_0_fa2_gqa_gfx1201` (256)** | **2647.89** | **2719.28** | **2642.53** |
| `gemm_qkvza…` (768) | 868.31 | 825.95 | — |
| **`gated_delta_net_q8_batch_seq` (768)** | **544.16** | **555.92** | **546.23** |
| `gemm_qkv…` (256) | 815.21 | 712.05 | — |
| `fused_silu_mul…` (1024) | 184.99 | 183.47 | 187.27 |
| `pack_f32_to_fp8…` (4096) | 39.82 | 41.62 | — |
| **`conv1d_silu_split_f32_n` (768)** | **175.72** | **177.83** | **177.29** |
| `fused_rmsnorm_mq_rotate…` (2048) | 34.52 | 32.82 | 33.94 |
| `gated_norm_f32_batched` (768) | 61.15 | 60.20 | 60.37 |
| `fused_qk_l2_norm…` (768) | 37.04 | 38.10 | 41.30 |
| `fused_sigmoid_alpha_gate…` (768) | 10.38 | 10.03 | 9.69 |
| isolated total | 6082.3 ms | 5641.5 ms | 4624.3 ms |

GDN us/launch is chunk-count independent: 543.6 (pp512, n=48×6 runs) vs 544–556 (pp8192,
n=768×3 arms). The serial cost is a per-512-row-launch constant, as the kernel shape
predicts (one launch per LA layer per 512-row chunk, §3.3 of the plan).

## 4. Group shares and launch census per 512-row layer-chunk

Groups — GEMM: all `gemm_*` + `quantize_int4` + `pack_f32_to_fp8` + `gemv` tail;
GDN: scan + `conv1d` + the three preamble launches + post-GDN `gated_norm`;
Attention: FA2 (+ preconvert + KV-write from rocprof, see below);
Glue: norms/rotates/silu/sigmoid_mul/deinterleave/rope + head/tail singletons.
Shares are of the isolated sum (plan §1.1: attribution only, wall comes from §7).

| run | isolated | GEMM | GDN-5 | Attention (FA2-only) | Glue |
|---|---|---|---|---|---|
| base pp512 | 339.3 ms | 81.2 % | 11.7 % | 1.1 % | 5.9 % |
| fp8v2 pp512 | 310.7 ms | 79.3 % | 13.0 % | 1.2 % | 6.5 % |
| iu4 pp512 | 246.9 ms | 74.2 % | 16.0 % | 1.5 % | 8.2 % |
| base pp8192 | 6082.3 ms | 73.0 % | 10.5 % | 11.1 % | 5.4 % |
| fp8v2 pp8192 | 5641.5 ms | 70.4 % | 11.5 % | 12.3 % | 5.8 % |
| iu4 pp8192 | 4624.3 ms | 64.3 % | 13.9 % | 14.6 % | 7.2 % |

Launch census per 512-row chunk (exact 16× scaling verified at pp8192 in every row):
- **Per LA layer-chunk (×48): 5 non-GEMM** — scan 1, conv/split 1, l2norm+scale+interleave 1,
  sigmoid gate 1, gated_norm 1 (plan §4.1's launches 1–4 + the post-GDN norm).
  No separate cumsum launch exists in-tree; the gate's exp/cumsum-equivalent work is inside
  the scan kernel. GEMM side per LA layer: qkvza 1, gate_up 1, residual 2, pack 4,
  silu_mul 1, rmsnorm_rotate 2, rotate 1.
- **Per FA layer-chunk (×16):** FA2 1 (+ preconvert 2 + KV-write 2, timed only in rocprof),
  qkv 1, gate_up 1, residual 2, pack 4, silu_mul 1, rmsnorm_rotate 2, rotate 1,
  deinterleave 1, rope 1, rmsnorm 1, sigmoid_mul 1.
- **Model head/tail (1×/prefill):** gemv 1, rmsnorm_f32 1, mq_rotate_x 1.
- Scan + P combined launch delta per layer-chunk (plan §3.3): 5 → 4 (−48 per 512-chunk).

## 5. rocprofv3 cross-check (same command, base pp512)

`rocprofv3 --kernel-trace` on the base-pp512 command: 3168 dispatches (2× the timed
kernels = warmup + profiled pass, plus untimed fills/copies/KV-writes). Split by time
order, warmup-half vs profiled-half agree ≤1 % on **every** kernel ≥ 100 µs
(GDN 531.1→534.9, +0.7 %; FA2 212.2→212.5; conv 168.6→168.7; GEMMs +0.1…+0.9 %) —
clocks were stable across passes, and call counts match the profiler exactly (2× each).

| kernel | profiler (profiled pass) | rocprof (profiled half) | Δ |
|---|---|---|---|
| gate_up | 1847.6 | 1827.9 | −1.1 % |
| residual | 709.6 | 693.5 | −2.3 % |
| qkvza | 867.1 | 847.1 | −2.3 % |
| **GDN scan** | **544.0** | **534.9** | **−1.7 %** |
| qkv | 811.6 | 791.7 | −2.5 % |
| silu_mul | 187.0 | 176.5 | −5.6 % |
| pack | 43.2 | 33.9 | −21.5 % |
| conv | 179.1 | 168.7 | −5.8 % |
| FA2 | 239.7 | 212.5 | −11.3 % |
| gated_norm | 62.7 | 53.0 | −15.5 % |
| qknorm | 38.9 | 28.1 | −27.8 % |
| gate | 12.3 | 3.1 | −75 % |

Reading: the in-process profiler carries a roughly **constant +9–11 µs/launch tax**
(event record + stop-event sync), verified across six kernels from 3 to 187 µs.
After subtracting it, every kernel ≥ 0.5 % of wall agrees within ~3 % **except FA2**,
which keeps a systematic +27 µs profiler-side gap (~+17 µs beyond the tax; hypothesized
1 M-thread-grid launch/drain latency — [INFERENCE], unconfirmed, and immaterial to G0:
FA2 is 1.1 % of wall at pp512). Verdict: cross-check passes on all kernels that carry
weight (top-5 = 85 % of wall agree ≤ 2.5 %); small-kernel gaps are a characterized
profiler tax, not a measurement disagreement. Tax-adjusted `t_serial` ≈ 533–535 µs —
still mid-defer-band, verdict unaffected.

Untimed-but-real launches (rocprof only, per pp512 pass): `kv_cache_write_q8_0_batched`
32 (16 FA × K/V), `attention_fa2_q_preconvert_gfx1201` 32, plus ~440 fill/copy
(memset/memcpy-class) kernels from reset/alloc paths. Full attention cost per pass:
FA2 3.80 ms + preconvert 0.41 ms + KV-write ~1.09 ms ≈ **5.3 ms** (vs 3.8 FA2-only).

## 6. GDN us/launch + GB/s; conv1d geometry + GB/s; bandwidth reference

- **Scan kernel** `gated_delta_net_q8_fast`, grid `[n_heads=48, 32, 1]` block `[32,1,1]`
  = 1536 waves (`norm.rs:3142-3146`; n_v_heads 48, value HD 128 derived from byte math
  §1: 52,149,281 B/launch ÷ 1,086,464 B/head = 48.0).
  Bytes/launch: **52.15 MB** on `profile.rs` accounting, **55.30 MB** counting the f16 EF
  read+write it omits (plan §1.2). At 543.6 µs: **95.9 / 101.7 GB/s**, and
  2.81 GFLOP/launch → **5.17 TFLOP/s f32** (~27 % of nominal 19.2 TFLOPS @64 CU ×
  2.35 GHz dual-issue; ~20 % at 3.1 GHz boost).
- **conv1d** `conv1d_silu_split_f32`, grid **[40,1,1] block 256 = 10,240 threads**,
  one thread per channel (`2·k_dim+v_dim` = 2·2048+6144), each looping all N tokens
  through the global ring (`conv.hip:39-42,50-78`; wrapper `norm.rs:4625-4627`).
  Distinct traffic/launch @N=512: qkv in 20.97 MB + q/k/v out 4.19+4.19+12.58 MB =
  **41.93 MB** (+160 KB weights, L2-resident; ring state negligible). At 177.1 µs:
  **≈237 GB/s** — the highest defensible DRAM rate in any table, hence the S0
  **bandwidth reference point** (~37 % of the 640 GB/s nominal roof).
  The table's 1316–1430 GB/s column for conv is the known fictitious analytical
  counting (weight + ring bytes charged per token, plan §1.1(3)) — never quoted as
  achieved bandwidth. Same caveat applies to FA2's ~1150–1170 GB/s (KV cache is
  L2-resident across query rows; real DRAM is a fraction).
- **Preamble combined** (P's target): conv 177.1 + qknorm 36.6 + gate 9.8 =
  **223.5 µs/launch** on gfx1201 (vs 367.35 Halo). P's ≥40 % cut target ⇒ ≤ ~134 µs
  fused. T's ≥5 % on the incumbent ⇒ ≤ ~516 µs/launch to matter.

## 7. Matched bench wall (real denominator) and the iu4-wall caveat

`hipfire bench --matrix --pp 512,8192 --ctx 128 --tg 32 --runs 3 --warmups 2 --kv-mode q8
--backend noslots --workload stateless`, eager (`HIPFIRE_GRAPH=0` inherited), daemon on
ab2 port 11527, tight samples (stdev ≤ 6 tok/s):

| arm | pp512 wall | pp8192 wall |
|---|---|---|
| base fp8 | 1459.6 tok/s → **350.8 ms** (1457.0/1459.8/1459.6) | 1324.5 tok/s → **6185 ms** |
| iu4-flagged | 1485.2 tok/s → 344.8 ms | 1336.4 tok/s → 6130 ms |

The iu4-flagged rows are **not** an iu4 wall: a fresh-cache probe run proves the bench
daemon compiled `gemm_*_wmma_fp8_gfx12_s2bt8` (base arm) — `HIPFIRE_GFX11_MQ4V2_IU4=1`
does not propagate through bench's spawn+configure (`BTreeMap::new()` env,
`Engine::spawn`, `main.rs:4862/4784`, `hipfire-client/lib.rs:428-438`; `hipfire config
get kernel.gfx11_mq4v2_iu4` reads `true` in the CLI with the env set, yet the daemon
still compiled base kernels). Treat the iu4 bench rows as base-arm repeats. The iu4
serve wall is therefore **unmeasured**; `f_gdn` on the fastest arm is bounded instead:
scan-only GDN is 26.0 ms @pp512 / 419.5 ms @pp8192 isolated, and no serve wall can fall
far enough below the isolated sums to threaten the 3 % bar (needs wall < 867 ms @512 —
the measured base wall is 351 ms *above* the isolated 247 ms already… more directly:
even against the fastest conceivable wall, `f_gdn` ≥ 6.7 % on every arm/size — §4
shares clear 3 % by ≥2× everywhere).

`f_gdn` vs the measured base wall: pp512 scan 26.09/350.8 = **7.4 %**
(GDN-5: 39.7/350.8 = 11.3 %); pp8192 scan 417.9/6185 = **6.8 %** (GDN-5: 636.3/6185 =
10.3 %). Isolated shares (§4) run hotter (6.9–10.5 % scan-only) because the isolated
sum bakes in the §5 launch tax while the wall doesn't — the wall ratio is the honest one.
Decode context row (not gated, recorded): tg32@128 ≈ 35.9 tok/s both arms.

## 8. Attention share curve (512 → 8192)

FA2-only isolated share: base 1.0 % → 11.1 %; fp8v2 1.2 % → 12.3 %; iu4 1.5 % → 14.6 %.
With preconvert + KV-write (rocprof, §5): ≈1.6 % → ≈11.5 % (8192 end scaled: FA2 677.9
+ preconvert ~6.6 + KV-write ~17.4 ≈ 702 ms / 6082). FA2 us/call (chunk-averaged):
236–241 → 2643–2719 (**11.2× for 16× rows** — sublinear: per-chunk cost grows with KV
length while fixed costs amortize; the last of the 16 chunks dominates, not separated
by this binary). Attention is the share-gainer with context; GDN is flat per launch.

## 9. Reproducibility log (Main's bracketing rule)

- GDN @pp512 base: 524.17 (first-ever run) / 687.65 / 544.71 / 543.98 (rocprof run's
  profiler table) / **544.23 / 542.97 / 543.47 (A/B/C bracket)**. The 687.65 reading
  (one run, everything in it ~30 % slow incl. GEMMs) **did not reproduce** in A/B/C
  (±0.12 %) or in four other same-config runs; junction 54–62 °C throughout
  (`bracket/tele-{A,B,C}.txt`, GPU[3] = ROCR ordinal 2 by UUID). Recorded as an
  unreproduced excursion, not averaged, not attributed to heat. Published base-pp512
  column = A/B/C means.
- fp8v2/iu4 pp512 GDN: 553.76→552.85 (−0.16 %), 542.02→542.09 (+0.01 %) — reproduced.
- pp8192 cells are single runs but each mean rests on 768 calls; warmup-half vs
  profiled-half in the trace agree ≤1 % (§5), which brackets the clock-drift risk
  in-run. A same-binary A/B/A at pp8192 was not run (cost); the 544–556 band across
  three different binaries/arms corroborates the per-launch constant.
- "Thermal"/"noise" words appear here only with the samples attached or not at all.

## 10. What S0 hands to P / T / O

- P: fuse conv+qknorm+gate (223.5 µs combined) with the §4.2 exactness recipe; kill bar:
  fused ≥ sum-of-three and byte-identity, else kill. q_raw/k_raw stores kept in v1.
- T: knobs `HIPFIRE_GDN_TILE_ROWS/{4,8}`, `WAVES_PER_BLOCK/{1,2}`, block/min-blocks,
  DPP+prefetch; every pin must hold bit-for-bit; <5 % isolated win ⇒ leave defaults.
  Reference points: GDN 543.6 µs, conv 177.1 µs, FA2 237.0 µs @pp512.
- O: f64-reference arms compare scan vs incumbent deviation (plan §2.3c); conditioning
  guard `max Σ|T| ≤ 10` before any timing claim.
- S (deferred, not abandoned): re-measure `t_serial` after P+T on this fixture; authorize
  only if ≥600 µs AND residual `f_gdn` ≥ 3 %.
