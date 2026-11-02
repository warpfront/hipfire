# Qwen4 (Qwen3.8-Flash-Next mq6q8-pleq8) prefill — autoresearch campaign (gfx1151) — 2026-09-26

**Lifecycle:** `historical`

**Disposition:** measured local deltas on `gfx1151`, landed on branch
`autoresearch/session-20260924` (`ae96347b4` → `1dcf81c02`). Not a G5
admission, not a retained-replay certification, not a cross-architecture
result, and not a product speed-floor update.

## Fixture

- Host `halo`, gfx1151, 128 GB UMA, HIP 7.2.
- Model `~/.hipfire/models/qwen3.8-flash-next.mq6q8-pleq8.hfq`, size
  `125467331096`, sha256
  `58fb4f586403000b3394413c38f58b0ec0d8845675f81c3d3c0b5de2cdaa4aed`
  (the Flash-Next pin in `AGENTS.md`).
- Prompt `benchmarks/prompts/glimmer_prefill_1024.txt`, md5
  `0ee8f86ada3683eda452bc294ec824a9`, 1131 prompt tokens.
- Harness `autoresearch.sh` at `fe1f26f76` (the campaign branch head; not kept
  in the tree). Final daemon (`1dcf81c02`, rebuilt for this record) md5
  `274f17b51f32177a8329a12a1a8fa15f`.

## Method

- Every run: build `hipfire-daemon`, start a fresh daemon under the GPU lock,
  load with `max_seq` 2048, KV q8, MTP off, `HIPFIRE_GRAPH=0
  HIPFIRE_AR_GRAPH=0 HIPFIRE_CASK_OFF=1 HIPFIRE_DPM_WARMUP_SECS=10`, one warmup
  and three measured greedy generates (`max_tokens` 16). The metric is the
  median of the three daemon-reported prefill rates (prefill tokens /
  prefill ms).
- Guard: the first 16 greedy token ids must match `REF_IDS` (at least 12
  leading). All 173 completed runs matched 16/16 against the `REF_IDS` in
  force. `REF_IDS` changed with the segment-2 baseline (`91a0a356d`) and the
  four token-changing KLD-gated keeps (runs 107, 123, 131, 147), each in the
  same commit.
- Segment 1 (runs 1-104): bit-exact only. Each keep is bit-identical to its
  predecessor: a kernel-level test against the replaced kernel, or matching
  output hashes in a standalone microbench, plus identical tokens.
- Segment 2 (runs 105-180): non-bit-exact kernels admitted when KLD against
  the BF16-source teacher (wikitext-2, 32 × 512 tokens; `qwen4_kld eval`,
  built with `--features lab`, then `saddle-quant reduce`) is not
  separated-worse. The other segment-2 keeps are bit-exact.
- Several sub-1% keeps were confirmed with interleaved fresh-daemon A/B
  (9 + 9 samples) or rocprofv3 kernel deltas instead of the 3-sample median.
- The GPU was shared with external `llama-server` processes. A run whose
  decode rate fell to ~13 tok/s (normal ~20) was contended; it was re-run and
  not logged.
- Kernel attribution: rocprofv3 `--kernel-trace` of one generate after a
  JIT-warming run. Profiles and microbench harnesses are local only
  (`.codeinsight+research/qwen4/ar-pp-20260924/prof*`, `/tmp/mb`); run logs in
  `~/.omp/autoresearch/--home-bjoern-hipfire--/runs/NNNN/benchmark.log`.

## Result

| | start (run 1, `ae96347b4`) | segment-1 best (run 103) | segment-2 baseline (run 105, `91a0a356d`) | end (run 178, `1dcf81c02`) |
|---|---:|---:|---:|---:|
| prefill tok/s | 185.34 | 512.55 | 578.28 | **1300.75** |
| TTFT, 1131 tokens | 6102.4 ms | — | 1955.8 ms | 869.5 ms |
| decode tok/s | 19.8 | — | 20.3 | 20.3 |

153 logged runs: 100 keep, 51 discard, 2 crash. Prefill 7.0× over the
campaign; segment 1 alone 2.77× with every keep bit-exact.

KLD trail (segment 2; every arm reduced NOT SEPARATED; bit-exact
reference 0.074745): PR #775 gate/up WMMA 0.075248 → MoE down WMMA 0.074272 →
BF16 dense via F16 shadow 0.075047 → HC read WMMA 0.077476 → dense attention
WMMA 0.074541 → GDN column split 0.073285 → HC read from the F16 copy 0.073544
→ GDN chunked WMMA **0.073471** (final numerics).

End-state profile (prof67, the prefill forward of one 1131-token generate:
span 875.5 ms, busy 857.6 ms), top kernels:

| kernel | calls | us/call | ms |
|---|---:|---:|---:|
| `gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out` (routed gate/up) | 48 | 4792 | 230.0 |
| `gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out` (routed down) | 48 | 2856 | 137.1 |
| `gemm_mq6g256v2_wmma_gfx11_bt8_x4*` (MQ6 trunk, 6 shapes) | 228 | — | 164.7 |
| `gemm_wmma_lds_64_64_32_64_k64_p` (HC input_mix_down 320×10240) | 97 | 516 | 50.0 |
| `hyper_read_up_wmma_bf16` | 97 | 469 | 45.5 |
| `hyper_norm_f32` / `hyper_norm_gate_f32` / `hyper_write_bf16x2` | 97/96/96 | ~224 | 64.8 |
| `gated_delta_chunk_gate_wmma` | 36 | 618 | 22.2 |
| `indexed_attention_dense_wmma_f16` | 12 | 1591 | 19.1 |
| `moe_down_combine_grouped_top10_bf16in` | 48 | 365 | 17.5 |

The routed MoE GEMMs are still ~1 ms/layer above their load-only floors
(microbench at layer 24: gate/up 3.78 ms, down 2.07 ms).

## What worked — reusable levers

Commit hashes are on the branch; the per-run ledger is Appendix A. The
condensed catalog lives in
[`.agents/skills/hipfire-kernel-tuning/levers.md`](../../.agents/skills/hipfire-kernel-tuning/levers.md) §11.

### 1. Matrix engine on the prefill paths (segment 2, KLD-gated)

| path | commit | kernel-level effect |
|---|---|---|
| routed MoE gate/up F16 WMMA (PR #775, @nwoolmer) | `91a0a356d` | segment-2 baseline, 512.6 → 578.3 tok/s |
| routed MoE down F16 WMMA (G128 group) | `aa25c5be0` | 8.6 → 5.4 ms/call |
| BF16 dense projections via a model-lifetime F16 weight shadow + LDS WMMA GEMM | `2b4a5819c` | 2.5-5× per GEMM |
| HC read tail as BF16 WMMA | `91396ea6a` | 112 → 48 ms per prefill |
| QSA attention whose selection is the whole causal window → dense causal GQA flash attention on WMMA | `97f5d7f2a` | 5.35 → 1.65 ms/layer |
| GDN recurrence in chunked WY form (16-row chunks, `(I+L)(I+L²)(I+L⁴)(I+L⁸)` inverse) on WMMA | `85d8fa53a` | 1263 → 626 us/layer |

### 2. Store what the consumer reads (bit-exact)

- **Narrow the stored type to the consumer's first rounding.** When every
  reader rounds a value to BF16 before use, store RNE BF16 bits from the
  producer's epilogue: the values used are unchanged and the traffic halves.
  MoE `*_bf16out`/`*_bf16in` (`f825e8022`), HC residual streams as BF16
  (`d134b3e50`), MQ6 qkv output feeding the conv (`903adccad`), conv output
  (`a9f7cd556`), GDN output feeding the rotation (`c5d44cb2d`).
- **Producers write the next GEMM's F16 input.** Rotations and norms emit F16
  directly: `mq_rotate_x_f16` (`6ec3c5ee2`), `mq_rotate_x_128_v2_f16`
  (`1ebbecb86`), `hyper_norm_f16` (`5bf7f3d80`), rotate right before the
  gate/up GEMM (`6852aa844`), unscatter + FWHT → F16 (`9ece7d891`). Pin the F32
  product with `asm volatile("" : "+v"(x))` so fast-math cannot fold it into an
  F16 multiply; the bytes then equal rotate → convert.
- **Drop round trips the next kernel already applies** (`e0d75ca30`,
  `05d6b618b`).
- **Fuse elementwise epilogues into the GEMM by pairing rows.** The gate/up
  GEMM's 16-row A tile holds gate rows m..m+7 and up rows mi+m..mi+m+7, so one
  more `permlanex16` gives each lane (gate, up) pairs and the epilogue stores
  the SwiGLU activation: half the output bytes, the next kernel reads one value
  instead of two (`1dcf81c02`, +0.9% e2e).

### 3. WMMA kernel structure (bit-exact)

- **Packed epilogue.** In the gfx11 16×16 F32 C layout lane l < 16 holds rows
  2j and lane l+16 rows 2j+1 of column l. One `permlanex16` swap per row pair
  leaves each lane 8 contiguous rows: one 16-byte BF16 store (or two float4)
  instead of 8 scattered stores. MoE `941b3c606` (gate/up 5.62 → 5.28 ms,
  down 3.29 → 2.90 ms), MQ6 `90da6b370` (10240×2560 1.92 → 1.68 ms), dense
  attention `3ee04a14a`.
- **Stage weights one quant group at a time, double-buffered in LDS.** Group
  g+1's coalesced 8-byte global loads (nine per lane) stay in flight while
  group g decodes and multiplies; LDS stays at 2-4 KB, so occupancy is kept.
  Down `5279f0525` (2.95 → 2.83 ms/layer), gate/up `c7ca999c3` (4.92 → 4.81).
  Staging the whole 16-row tile first (`f67df582d`) lost occupancy in
  production and was reverted for down (`4ee60d768`).
- **Several same-expert tiles per wave.** Prefill routing is skewed (~376
  active experts per layer, hot experts own up to 70 slot tiles), so one wave
  computes NT ∈ {1, 2, 4} consecutive tiles of an expert (three-tile runs mask
  the fourth), sharing each A load and decode across up to 64 slots
  (pairs `3c4dc29a2`: 782.8 → 816.1 tok/s; up to four `908c1a1d3`:
  1214.0 → 1244.1 tok/s).
- **Unroll the K loop so later tiles' loads issue ahead** (`e94e078da`:
  gate/up 9.4 → 6.95 ms).
- **Nibble decode with `v_perm` + F16 magic bias.** One byte permute copies
  byte p into bytes 0 and 2, one AND-OR keeps the low nibble under 0x6400
  (1024 + q) and the high nibble at bits 4-7 under 0x5400 (64 + q); subtract
  {1024, 64} and one packed FMA gives sc·q + zp for two weights, ~4 VALU per
  pair instead of ~6 (`405469994`, A/B +1.3%).
- **LDS-only barriers on gfx11.** `s_waitcnt lgkmcnt(0); s_barrier` instead of
  `__syncthreads()`, whose WGP-mode fence also waits for the pipelined
  next-stage global loads (`cc650be3a`).
- **Split accumulators across waves.** Dense attention runs two waves per head;
  both compute scores/softmax, each owns half of O^T (64 instead of 128
  accumulator VGPRs), next K/V tile prefetched into registers (`3ee04a14a`:
  1722 → 1568 us/layer).
- **More workgroups for short-M, long-K GEMMs.** 64×64 tile for the HC
  input_mix_down 320×10240 (90 instead of 27 workgroups, `e9250084f`) and
  128×128 for 512 ≤ M < 1024 (`55fea2e4d`). A tile change that keeps each
  output's K order is bit-exact.

### 4. SIMT kernels (segment 1, bit-exact)

- Exact lane-0 tree reductions with `permlanex16` + `v_add_f32_dpp`
  (`bound_ctrl`) instead of `ds_bpermute` shuffles; two independent trees
  packed per wave (runs 27, 28, 40, 41, 65, 66).
- Buffer loads with uniform scalar offsets instead of 64-bit VGPR address math,
  `v_cvt_f32_ubyteN` nibble conversion (runs 55, 56, 103); scalar loads for
  block-uniform operands (runs 90, 91).
- Share X or W across waves through LDS: MQ6 X-LDS, four row-tile waves per
  64-wide X chunk (run 57: 1665 → 574 us at 2560×6144); r16w4, four waves per
  16×256 BF16 weight chunk (run 63).
- Issue many independent loads before an ordered reduction instead of one
  round trip per element (runs 42, 79, 96, 101).
- Fuse per-row chains that keep an intermediate out of global memory: HC
  norm + gate projection (run 77), up projection + read (run 78), GDN
  recurrence + gated RMSNorm (run 136).

### 5. Scheduling and host

- Prefill chunk cap = `max_seq` (2048): the prompt is one chunk and each
  expert streams once (run 69: 375.8 → 406.3). A 1024 cap was slower (run 24);
  it needed the copy-kernel grid fix first (run 23: one block per element,
  i32 index overflow past 8 M elements).
- Token-parallel chunks for scans whose carried state only chunk 0 touches:
  PLE depthwise conv 1921 → 832 us (`3acdb8809`), GDN conv (run 95).
- Resolve per-token expert rank order once (`fc94f16d3`: 496 → 289 us).
- Overlap host PLE staging with GPU work (`d7000c5aa`).
- Per-kernel `HIPFIRE_COMPILER_FLAGS: -mllvm -amdgpu-sched-strategy=max-ilp`
  (run 76, ~4% microbench).

### Method lessons

- WMMA accumulation is K-order dependent: a K permutation changed 74% of the
  outputs. Keep the natural K order when bit-exactness is the contract.
- MoE microbenches need captured real routing (slot tile ids dumped from a
  prefill). Uniform routing misranked candidates.
- Microbench wins on a warm operand did not transfer when production reads it
  cold: HC input_mix_down tiles (runs 114, 134, 143, 180), MoE down two row
  blocks per wave (run 162). Confirm with rocprof in the pipeline.

## What did not work

Full list in Appendix B. The recurring ones:

- MoE: two 16-row blocks per wave, 2-deep group prefetch, rolling/full X
  software pipelines (VGPR pressure), 8 tiles per wave, `waves_per_eu` 7/8,
  half-wave decode via `permlanex16`, stacked shared gate+up GEMM (401 us vs
  2 × 179 us). A tiled weight repack was not attempted: 13+ kernels read the
  MQ4 layout.
- MQ6: two row tiles per wave, BT16, packed-F16 decode.
- Attention: four waves per head, Q in registers, reversed block order.
- GDN: z as BF16 for the chunk kernel, 10-wave split, two blocks per head.
- HC: fusing the write into a norm pass (runs 80, 120, 145), K-stage 32 tile,
  64×32 tile, 384-token read blocks.

## Not claimed / not done

- **Other architectures.** Every new fast path is gated to gfx1151 or to the
  gfx11 WMMA ISA; nothing was measured off gfx1151.
- **Validation routes.** Evidence is the token guard, kernel-level bitwise
  tests and microbench hashes, and the KLD gate. `scripts/serve_harness.py` and
  `scripts/redline_daemon_harness.py` were not run, and retained-route
  certification was not re-proved.
- **Other prompt lengths.** One 1131-token prompt. The F16 WMMA routes start
  at 512 tokens (`QWEN4_F16_WMMA_MIN_TOKENS`); shorter prefills take the
  segment-1 paths.
- **Decode and load time** were not targets (decode ~20 tok/s throughout).

## Appendix A — kept runs

| run | prefill tok/s | commit | change |
|---:|---:|---|---|
| 1 | 185.34 | `ae96347b4 (start)` | baseline: HEAD ae96347b4 + harness |
| 2 | 221.04 | `fd84b9707` | QSA attention: grouped hg4 kernel (4 heads/WG, LDS key tiles, expf once per head-token), bitwise-equal to … |
| 3 | 223.26 | `2f84b96ab` | GDN output BF16 round trip: one launch over all rows (was per-row, 36*rows launches); hg4 attention only for … |
| 5 | 237.40 | `4e8fd8e93` | GDN persistent step: per-row q/k norms, decay, beta precomputed per 256-row block in parallel; next-row q/k/v … |
| 6 | 244.77 | `8fe193c05` | MQ6 residual WMMA BT8 (was BT4) for N>=384 on gfx1151; raw-bit parity via test_mqv2_bt_gfx11 (62 arms PASS) |
| 8 | 250.55 | `e0d75ca30` | Drop bf16 round trip over all padded grouped-down rows (12800x2560): grouped combine already RNE-rounds each … |
| 9 | 253.73 | `52edf6d2e` | GDN batched gated RMSNorm: stage rounded head in LDS (sum reads LDS not global), 128-thread block (was 256 … |
| 11 | 262.80 | `e03e8ba90` | hg4 QSA attention: float4 LDS reads (key tile stride 68, q), float4 global key loads, weights interleaved … |
| 17 | 264.35 | `9a55cba6d` | BF16 R16 route for (640,2560) shared gate/up and (512,2560) router: profile 110.8->58.0 ms and 43.3->25.5 ms … |
| 18 | 266.49 | `c17bcfa12` | BF16 R16 route admitted from N>=64 (was 128) so the 107-row tail chunk uses it too (same bitwise contract) |
| 23 | 268.39 | `7c5e32c0a` | Fix copy_f32_buffer / copy_f32_strided_slot_buffer grids: launched one 256-thread block per element (256x … |
| 27 | 270.19 | `ed5dc72e0` | MoE down O4xR16: lane-0 tree reduction via permlanex16 + DPP row_shl (exact, same adds) instead of … |
| 28 | 271.77 | `3e8cb90da` | MoE gate/up O4xR4: exact lane-0 tree reduction via permlanex16 + DPP instead of ds_bpermute shuffles; bitwise … |
| 32 | 276.80 | `9eeaa5df0` | hg4 QSA attention score phase: one selected row per thread x 4 heads, key read directly from global (no LDS … |
| 34 | 282.29 | `682f84751` | hg4 attention PV loop: branch-free (invalid rows add exact +0.0), 8 value loads issued before in-order … |
| 40 | 287.47 | `cdabef02b` | MoE down/gate_up tree reduction: update_dpp with bound_ctrl so row_shl steps fold into v_add_f32_dpp (240 of … |
| 41 | 289.35 | `d6e26f35b` | BF16 4-row (gfx11/gfx12) and R16 kernels: lane-0 shuffle tree via permlanex16 + combined v_add_f32_dpp … |
| 42 | 308.05 | `1d9d64c0e` | MoE gate/up O4xR4: software pipeline (next group's header/payload/X float4 loads issued before decoding the … |
| 45 | 310.96 | `a75c1b19d` | MQ{2,3,5,6}V2 residual WMMA BT kernels: each 16-lane half decodes its 8 of 16 replicated A weights, halves … |
| 51 | 313.50 | `36e7163a0` | GDN persistent step: separate kv/delta/out handoff buffers and next row's norms staged behind the out barrier … |
| 54 | 316.32 | `4038315ba` | fuse MoE gate/up unscatter + SwiGLU into one kernel (path2), skipping separate silu/roundtrip; bitwise test … |
| 55 | 321.91 | `035a62c20` | gate/up O4xR4 buffer loads (uniform soffset, no 64-bit VGPR address math) + opaque 0x0F0F0F0F mask so nibbles … |
| 56 | 325.17 | `ad6dd7b15` | MoE down O4xR16 vector loop uses buffer loads (soffset per row/slot/group), scalar header loads … |
| 57 | 344.87 | `9daf20397` | gfx1151 MQ6 residual BT8 -> new X-LDS kernel (4 row-tile waves share each 64-wide X chunk via LDS … |
| 58 | 350.62 | `611f0b9e4` | gfx1151 MQ6 residual uses BT8 (X-LDS) from N>=96 instead of BT4 (microbench N=107: 260->157us); bitwise … |
| 61 | 353.85 | `38eefadc1` | MoE gate/up and down prologues issue all slot-index scalar loads together with the expert ID (clamped … |
| 63 | 356.22 | `8eba7003a` | bf16 R16 shapes with K%256==0 -> new r16w4 kernel (4 waves share each 16x256 BF16 weight chunk via LDS, rows … |
| 64 | 365.05 | `f39aa58f0` | gfx1151 bf16 small-K (257..768, K%8==0) -> rows-loop kernels (wave keeps 4 tokens' X in registers, walks 16 … |
| 65 | 372.70 | `e8d39128e` | packed two-value lane-0 tree sum (value a in lanes 0-15, b in 16-31 after one permlanex16; shared DPP steps … |
| 66 | 375.78 | `0ec41b9d8` | packed two-value tree sums in MoE gate/up (row pairs) and bf16 r16w4 (stream pairs); dropped now-unused … |
| 69 | 406.29 | `a6a2de4ae` | Qwen4 prefill chunk cap 512 -> 2048 (= contract max_seq; 1131-token prompt is one chunk, MoE/expert weights … |
| 70 | 419.01 | `342797fe3` | MoE gate/up O4xR4 -> O4xR8 with X staged in LDS (4 waves x 4 rows per 128-thread block, 8-slot subtiles: half … |
| 72 | 425.27 | `c34146dbc` | gfx1151 MQ6 overwrite GEMM (gemm_mq6g256v2) uses a one-launch X-LDS overwrite kernel (Y = +0.0 + acc) instead … |
| 73 | 438.15 | `1274ca0ae` | bf16 GEMM 256<K<=512 (HC 10240x320) -> per-thread-output kernel (lane = row, W tile in LDS with odd stride … |
| 75 | 441.88 | `f4fd495a5` | MoE down O4xR16 -> O8xR16 (eight rows per wave share each X vector; 189 VGPR); renamed file/const/kernel/test … |
| 76 | 444.79 | `5eecc215f` | gate/up O4xR8 kernel compiled with -mllvm -amdgpu-sched-strategy=max-ilp via HIPFIRE_COMPILER_FLAGS … |
| 77 | 449.18 | `88a4b0036` | HC write path fuses hyper_norm + BF16 (4 x 10240) gate projection into one per-row kernel (normalized row in … |
| 78 | 467.55 | `04e964be8` | HC read tail fuses the BF16 up projection (per-thread GEMM, 32 rows = 8 hidden columns x 4 branches per … |
| 79 | 469.37 | `fa25b9b05` | fused HC norm-gate fetches the gate weights ten steps at a time (was one serialized load + waitcnt per step) … |
| 83 | 472.96 | `af61bf43a` | QSA hg4 attention LDS: drop the unused 64x68 key tile (only 4x256 partial maxes remain) and reserve for the … |
| 85 | 479.58 | `3fe727274` | batched GDN gated RMSNorm: one wave per head (4 channels/lane, float4 IO); the serial norm chain runs once … |
| 88 | 481.73 | `93ea0ef26` | bf16 r16w4 token-tile-fastest sibling (r16w4t) when weight bytes exceed activation bytes (PLE key proj … |
| 89 | 486.05 | `2c65640ad` | PLE grouped gate 256-thread block with LDS-staged rows (serial reductions on thread 0 from LDS, parallel … |
| 90 | 487.50 | `ae26fe84a` | hg4 attention PV loop reads block-uniform token indices as scalars (s_load_b256) and fetches values via … |
| 91 | 488.07 | `26c6b7f4b` | hg4 attention scores read block-uniform q via scalar loads (s_load_b512, SGPR FMA operand) instead of LDS … |
| 92 | 493.52 | `c30700500` | hyper_read_up_fused 4 tokens per wave (each LDS weight load + widen serves 4 tokens), 128 tokens/block … |
| 95 | 497.10 | `7e75709bf` | batched GDN K=4 conv row-parallel (thread = channel x 16 rows, ring unrolled to x(r-3..r-1), history … |
| 96 | 498.39 | `6a3e0de3c` | HC norms (hyper_norm, hyper_norm_gate) issue ten strided loads before the ordered sum-of-squares partial … |
| 101 | 504.51 | `f2cdcf371` | persistent GDN step per-row q/k norm parameters read with float4 loads, next pair issued before the current … |
| 102 | 508.15 | `5e99ecf74` | QSA batched select scores four index heads side by side (each pooled value loaded once, float4; per-head d … |
| 103 | 512.55 | `64d9de7d0` | gfx1151 MQ6 X-LDS WMMA kernel reads each 64-wide chunk's 48 payload bytes with six 8-byte buffer loads (row … |
| 105 | 578.28 | `91a0a356d` | segment-2 baseline: perf(qwen4) PR #775 (@nwoolmer) F16 WMMA grouped gate/up on gfx1151 for >=512 tokens … |
| 107 | 627.95 | `aa25c5be0` | F16 WMMA grouped MQ4G128V2 MoE down on gfx1151 for >=512 tokens (new gemm_mq4g128v2_moe_grouped_wmma_gfx1151 … |
| 108 | 685 | `2b4a5819c` | BF16 dense prefill projections (hyper input_mix_down 320x10240, shared gate/up 640x2560, router 512x2560 … |
| 109 | 688.33 | `5bf7f3d80` | hyper_norm writes the F16 input of the F16 WMMA input_mix_down GEMM itself (hyper_norm_f16, same RNE as … |
| 110 | 765.90 | `e94e078da` | MoE grouped WMMA gate/up (shared gemm_mq4g256v2_moe_grouped_wmma_k2) and qwen4 down: kt loop fully unrolled … |
| 112 | 778.82 | `6ec3c5ee2` | MQ6 projections sharing an input (GDN qkv/a/b/z, attention indexer/q/k/v) rotate once straight to F16 (new … |
| 113 | 782.75 | `1ebbecb86` | MoE down on the F16 WMMA route rotates the activation straight to F16 inside the down stage (new … |
| 115 | 816.08 | `3c4dc29a2` | MoE grouped WMMA gate/up + down pair consecutive slot tiles of one expert in one wave (even expert-local tile … |
| 116 | 827.96 | `f825e8022` | MoE WMMA gate/up and down store their grouped outputs as BF16 bits (RNE, new *_bf16out entries) and the … |
| 117 | 833.15 | `9ece7d891` | MoE down stage runs a fused unscatter + SiLU + 128-wide FWHT kernel writing the down GEMM's F16 input … |
| 118 | 849.74 | `9ebf1dfe7` | F16-path hyper_norm stores the BF16-rounded normalized rows as BF16 bits instead of F32 (half the write), and … |
| 123 | 876.34 | `91396ea6a` | HC read tail on the F16 route uses a BF16 WMMA kernel (new kernels/src/hyper_read_up_wmma.gfx1151.hip … |
| 125 | 899.69 | `05d6b618b` | drop BF16 round trips the next kernel already applies: shared down's round trip of `out` and of the residual … |
| 126 | 904.29 | `f67df582d` | MoE grouped WMMA kernels stage the 16-row expert weight tile through LDS with whole-line b128 loads instead … |
| 127 | 906.10 | `4ee60d768` | revert the MoE down kernel's LDS tile staging (keep gate/up's): in production the 5.4 KB/wave dynamic LDS cut … |
| 128 | 910.70 | `7c8f2851e` | BF16-input grouped MoE combine takes eight columns per thread: the block-uniform expert-ascending rank order … |
| 129 | 920.49 | `911a1ecfa` | HC norm-gate holds each row in registers (all loads up front), runs the four branch RMS trees side by side … |
| 130 | 926.21 | `915d84e3f` | batched GDN gated RMSNorm drops the re-round of LDS values that are already BF16 (idempotent) and reads them … |
| 131 | 963.87 | `97f5d7f2a` | QSA attention whose selection is every row's whole causal window (budget covers all visible blocks, capacity … |
| 132 | 1,014.08 | `ca6a3c413` | GDN prefill recurrence split by value column on the F16 route: gated_delta_qk_norm_bf16_batched stores each … |
| 133 | 1,019.01 | `e9250084f` | F16-shadow BF16 GEMMs with M < 512 and K >= 4096 (HC input_mix_down 320 x 10240) use a new 64x64 / 32x64 k64 … |
| 135 | 1,032.50 | `cd547700c` | the BF16 WMMA HC read takes hyper_norm_f16's F16 copy (the BF16-rounded values, exact in F16 above 6e-5), so … |
| 136 | 1,043.45 | `7492e86fb` | the column-split GDN recurrence (F16 prefill route) now owns one value head per block and applies the gated … |
| 137 | 1,048.78 | `12fe71656` | fused GDN recurrence+gate fetches each 32-row group's z values into registers during the group's staging … |
| 139 | 1,066.28 | `d134b3e50` | on the F16 prefill route the HC residual streams are stored as BF16 bits (decided once per forward … |
| 140 | 1,078.27 | `c67a91f10` | hyper_norm_f32 dispatches once on the stream layout to a templated body (hc_stream_value overloads for F32 / … |
| 141 | 1,095.08 | `c08c7f337` | HC norm-gate on BF16 streams loads the row with 16-byte loads into the LDS region the normalized row later … |
| 146 | 1,096.99 | `b4da89aae` | on the F16 route hc_activation_fused_f32 writes the BF16-exact HC low-rank activations as packed BF16 … |
| 147 | 1,107.41 | `85d8fa53a` | GDN prefill recurrence + gated RMSNorm on the F16 route runs in the chunked WY form with 16-row chunks on F16 … |
| 148 | 1,122.80 | `cc650be3a` | gfx11 LDS-tile GEMM (WLDS, 6 barriers) and gfx1151 MQ6 BT8 x4 GEMM (2 barriers) synchronize with s_waitcnt … |
| 149 | 1,142.31 | `a9f7cd556` | on the chunked GDN route the K=4 convolution stores its (BF16-rounded) output as packed BF16 (BF16-typed … |
| 154 | 1,143.81 | `a8d33a936` | chunked GDN kernel keeps the next chunk's BF16 v prefetch as the raw uint4 and widens it at staging; widening … |
| 155 | 1,154.91 | `9ca67db97` | MoE shared expert prefill: (1) selector/gate/up BF16 projections go through project_weights, which now groups … |
| 156 | 1,166.58 | `fc94f16d3` | BF16-input MoE combine resolves each token's ascending-expert rank order once (new moe_combine_order_top10 … |
| 157 | 1,176.04 | `6852aa844` | path-2 F16 WMMA MoE gate/up rotates the activation straight to F16 (rotate_x_mq_batched_f16) right before the … |
| 158 | 1,178.37 | `d7000c5aa` | prefill (n > 1) defers the PLE wait/stage/upload/gather until the steps before the PLE layer (layer 0) are … |
| 159 | 1,207.56 | `941b3c606` | MoE grouped WMMA gate/up + down store each column's 8 rows as one 16-byte BF16 run (permlanex16 swaps row … |
| 160 | 1,214.04 | `90da6b370` | MQ6 X-LDS WMMA epilogue swaps row halves with permlanex16 and stores each column's 8 rows as two float4 (RMW … |
| 163 | 1,244.09 | `908c1a1d3` | MoE grouped WMMA gate/up + down compute up to four same-expert tiles per wave (PAIR -> NT in {1,2,4}; 3-tile … |
| 165 | 1,242.99 | `3acdb8809` | PLE grouped depthwise conv runs tokens in parallel chunks (>= history rows, so only chunk 0 reads/rewrites … |
| 169 | 1,230.95 | `c5d44cb2d` | chunked GDN stores its BF16-rounded gated output as BF16 bits when the MQ6 out-proj rotates it straight to … |
| 170 | 1,248.07 | `903adccad` | on the chunked GDN route the MQ6 qkv GEMM stores its output as BF16 bits (new … |
| 171 | 1,251.38 | `3ee04a14a` | dense QSA attention runs two waves per head (block 256): both compute the same scores/softmax, each keeps … |
| 172 | 1,271.64 | `405469994` | MoE grouped WMMA nibble decode (gate/up MQ4G256V2 + down MQ4G128V2): one v_perm copies byte p into bytes 0 … |
| 173 | 1,267.23 | `55fea2e4d` | qwen4 F16-route BF16 GEMM uses the 128x128 LDS tile for 512 <= M < 1024 (router 512, shared gate/up 640 at … |
| 174 | 1,282.75 | `028bae391` | hyper_read_up_wmma blocks cover 512 tokens (was 256): each staged 64-row W slice serves twice the tokens. … |
| 175 | 1,292.13 | `5279f0525` | MoE grouped down stages the wave's 16 rows one 68-byte group at a time, double-buffered in LDS (group g+1's … |
| 177 | 1,289.33 | `c7ca999c3` | MoE grouped gate/up stages one 136-byte group of the 16 rows at a time (272 8-byte pieces, nine per lane) … |
| 178 | 1,300.75 | `1dcf81c02` | MoE grouped gate/up WMMA SILU entry pairs gate rows m..m+7 with up rows mi+m..+7 in each 16-row A tile and … |

## Appendix B — rejected runs

| run | prefill tok/s | status | candidate and reason |
|---:|---:|---|---|
| 4 | 220.92 | discard | gate/up O4xR4 w8: 8 waves share X via LDS (bitwise-equal) -> slower per call (12.29 vs 11.70 ms); X traffic not the bound |
| 7 | 245.71 | discard | MQ6 residual BT16 for N>=384: +0.4% vs BT8, within noise; not worth two new symbols |
| 10 | 253.44 | discard | gate/up O4xR4: one block walks 4 subtiles of a 16-slot tile (weights once per tile): no change -> not weight-traffic bound either; likely issue-bound |
| 15 | 254.45 | discard | bf16 lanes-replay kernel for (10240,320) (1.43 vs 1.50 ms/call, ~0.4%) + gate/up O4xR8 (12.94 vs 11.81 ms/call, slower, 230 VGPR). Both bitwise-exact … |
| 16 | 259.95 | discard | gate/up O4xR4 branch-free slot loads (dead slots alias row 0, float4 X loads batched): bitwise-equal but 259.95 vs 262.8 best (VGPR 144->244); no gain |
| 19 | 262.01 | discard | MoE down O4xR16: LDS-transposed exact tree reduction instead of 320 shfl/bpermutes: bitwise-equal (new test vs multirow) but slower (262.0 vs 266.5) … |
| 20 | 0 | crash | QWEN4_PREFILL_CHUNK_CAP 512->1024 (+r16 range to 1024): GPU memory access fault during first prefill; some kernel/buffer assumes <=512 rows. Not … |
| 22 | 266.81 | discard | DPP/permlanex16 lane-0 tree reduction (exact) in bf16 4-row and R16 kernels instead of ds_bpermute shuffles: bitwise-equal, flat (266.8 vs 266.5) -> … |
| 24 | 265.80 | discard | Prefill chunk cap 1024 (+r16 range 1024) now runs (copy grid fix) with identical 16 tokens, but slower 265.8 vs 268.4: MoE is not weight-traffic … |
| 26 | 264.32 | discard | GDN persistent step with interleaved out_r / kv_{r+1} chains, 2 barriers/row, 512-row param ring: bitwise-equal (test extended to rows 1,2,3,600) but … |
| 29 | 272.36 | discard | 4-row x 4-token BF16 tile (DPP reduction) for (10240,320)/(2560,640): bitwise-equal but (10240,320) per call 1.67 vs 1.49 ms (slower); overall +0.2% … |
| 31 | 266.41 | discard | MQ6 residual WMMA with two row tiles per wave (bitwise-equal test): RT2xBT8 (206 VGPR) 270.9, RT2xBT4 266.4 vs 271.8 best -> X fragment traffic not … |
| 33 | 275.56 | discard | hyper_norm: keep BF16-rounded inputs in registers between reduction and normalize (one input read): flat/slightly lower (275.6 vs 276.8) |
| 35 | 280.54 | discard | hg4 attention: score d4 unroll 16 + PV 16-load batches: flat/slightly lower (280.5 vs 282.3) |
| 36 | 279.76 | discard | MQ6 residual BT8 from N>=96 (107-row tail chunk): 279.8 vs 282.3, no gain |
| 37 | 281.78 | discard | GDN persistent step: 4-row-deep raw q/k/v prefetch ring: bitwise-equal, flat (281.8 vs 282.3) -> row latency not load-bound |
| 39 | 275.83 | discard | MQ WMMA decode_tile in packed f16 (0x6400/q - 1024, pk_fma): raw-bit parity PASS (62 arms) but slower 275.8 vs 282.3 (extra v_mov_b16 packing; pk ops) |
| 43 | 292.14 | discard | MoE down O4xR16 full software pipeline (next group's header/payload + 16 X float4, alias loads for dead slots): 234 VGPR (launch bounds 32,6), slower … |
| 44 | 303.81 | discard | MoE down: prefetch next group's header/payload only (126 VGPR): 303.8 vs 308.1, no gain |
| 46 | 272.99 | discard | MoE down rolling X prefetch (refill each 4-slot batch with next group's X after consuming; 173 VGPR): much slower 273 vs 311 |
| 47 | 307.65 | discard | R16 (now DPP reductions) for (10240,320)/(2560,640): 307.7 vs 311.0, still not better than 4-row |
| 48 | 305.92 | discard | gate/up: skip X prefetch for dead slots (branch per slot): 305.9 vs 311.0, unconditional alias loads are better |
| 49 | 310.10 | discard | hyper_norm: in-wave tree steps via DPP (4 fewer barriers): flat 310.1 vs 311.0 |
| 50 | 311.06 | discard | GDN gated RMSNorm: one wave computes the serial sum, broadcast via LDS: flat (311.06 vs 310.96) |
| 52 | 294.70 | discard | bf16 4-row kernel: load two K steps before accumulating: slower 294.7 vs 313.5 |
| 53 | 310.08 | discard | MoE down: issue all 16 slots' X loads per group before FMAs (167 VGPR): 310.1 vs 313.5 |
| 59 | 345.31 | discard | gate/up O4xR4 with 4 waves per workgroup (adjacent row blocks of same subtile share X via L0): slower (345 vs 350.6) |
| 60 | 350.90 | discard | gate/up weights prefetched two groups ahead (X one ahead): flat (350.9 vs 350.6) |
| 62 | 308.28 | discard | bf16 r16 with 4 waves per block (same 16 rows, adjacent token pairs) to share W via L0: much slower (308 vs 353.9) |
| 67 | 375.10 | discard | gate/up: unconditional (aliased) weight loads, dropping per-row uniform branches: flat (375.1 vs 375.8) |
| 68 | 374.70 | discard | gate/up with 4-wave blocks sharing subtile X via LDS (register-staged, 2 barriers/group, 164 VGPR): flat (374.7 vs 375.8) |
| 71 | 416.62 | discard | MoE down O4xR16 with 4-wave blocks sharing X via LDS (119 VGPR): flat/slightly worse (416.6 vs 419.0) |
| 74 | 434.47 | discard | GDN persistent kernel split into 2 blocks per value head (64 channels, 128 threads): slower (434.5 vs 438.2) |
| 80 | 467.01 | discard | HC write: fold hyper_write into the per-row norm-gate kernel (one launch): flat/slightly worse (467.0 vs 469.4; samples 466.7-473.5) |
| 81 | 469.90 | discard | HC norm-gate: normalized row kept as exact BF16 bits in LDS (half the LDS): flat (469.9 vs 469.4) |
| 82 | 456.01 | discard | GDN persistent: q/k staged as F32 in LDS (q pre-scaled) to avoid SALU conversions + readfirstlane: slower (456 vs 469) |
| 84 | 472.08 | discard | hg4 attention: longest rows first (reverse blockIdx.z): flat (472.1 vs 473.0) |
| 87 | 478.26 | discard | QSA norm/RoPE batched: one wave per (row,head) instead of 256-thread block each running the serial sum; bitwise-equal but kernel is only 14ms/2415ms … |
| 93 | 487.04 | discard | hyper_norm_gate: 4 branch LDS trees side by side (9 barriers instead of 36) but rms_sum[4][256] grows LDS 41->45KB (3->2 blocks/WGP): slower |
| 94 | 492.83 | discard | hyper_norm_gate: side-by-side branch trees with rms_sum aliased onto xs (LDS 40KB): flat (492.8 vs 493.5) -> barrier count is not the limiter |
| 99 | 438.46 | discard | GDN persistent param loop float4 1-ahead prefetch: measurement INVALID (kld_teacher.py GPU jobs running concurrently, samples 407/449/438; earlier … |
| 100 | 0 | crash | GDN param prefetch re-measure: model load OOM (552 MB free) because a concurrent qwen4_kld job (pid 2337732) holds ~90 GB GTT. Environmental, not the … |
| 104 | 511.42 | discard | project_weights: rotate the shared input once for GDN qkv/a/b/z and attention indexer/q/k/v (saves ~108 small mq_rotate_x launches, ~7 ms): flat … |
| 114 | 778.93 | discard | dense F16 route: 64-row pipelined LDS tiles for M<=320 and K<=640 (bit-identical, probe 320x10240 0.446->0.383 ms, 2560x640 0.259->0.206 ms): flat … |
| 120 | 849.23 | discard | HyperWrite: fuse hyper_norm_gate + hyper_write per row (row read once into registers, mixed preloaded, BF16 xs in LDS, launch_bounds(256,3)) … |
| 134 | 1,016.99 | discard | Qwen4 F16-shadow GEMM tile override also for M <= 640 / K >= 2048 (shared gate/up 640x2560, router 512x2560 -> 128x128 pipelined; microbench 215->175 … |
| 143 | 1,023.71 | discard | HC input_mix_down 64x64 LDS tile with KS=32 instead of 64 (bitwise; microbench 387->340 us): production 1023.7 vs best 1095 (two runs, 1022.8 / … |
| 145 | 1,090.96 | discard | HC write + next HC read norm fused into one pass (hyper_write_norm_bf16 via a write->read peephole in execute_validated_steps; bitwise, tokens … |
| 152 | 1,138.29 | discard | GDN qkv and z MQ6 projections stored as BF16 (new gemm_mq6g256v2_wmma_gfx11_bt8_x4_bf16out epilogue, hip_bfloat16 RNE), conv reads BF16 input, chunk … |
| 162 | 1,208.08 | discard | MoE down: two 16-row blocks per wave with next-block weight prefetch (microbench 2.89->2.73 ms, bitwise): flat e2e (1214.3, 1208.1 vs 1214.0) |
| 166 | 1,240.68 | discard | PLE prefetch moved from forward entry into stage_ple (removes ~0.5 ms host-sync gap between embedding and hc_streams_init): flat within noise (1240.7) |
| 179 | 1,293.61 | discard | hyper_read_up_wmma_bf16 TOKENS 512->384 (even 3 blocks for 1131 rows): flat/noise (1293.6 vs 1300.8) |
| 180 | 1,290.07 | discard | HC input_mix_down (M=320,K=10240) LDS tile 64x64/32x64 -> 64x32/32x32 (twice the workgroups; bitwise, microbench hash equal): 1290.1 vs 1300.8 … |
