# F1-lite: SwiGLU folded into the gfx11 IU4 gate/up GEMM v2 epilogue

Branch `gfx11-f1lite` on official `mq4-lloyd` `2afc4a294`. The hipx worktree is `/home/kaden/hipfire-f1lite`. Model: `qwen3.8-27b.mq4v2.xt.sym-a035.qat-r7s200.hfq`.

**Result.** The change is exact on both cards. The WT2 c24 q8/q8 KLD sequences are byte-identical: gfx1100 0.076879 and Halo 0.076901. In the pp8192 traces, the targeted FFN kernels save 38.0 ms on XTX and 167.3 ms on Halo. The Halo request wall drops by 1.76% (8,196.9 → 8,053.1 ms). Kill switch: `HIPFIRE_F1LITE=0`. The feature is on by default on both cards. No ABBA was run; Stacker runs it.

## Variant shipped: f32 h stream plus a slimmer rotate+quant pass

The IU4 producer applies FWHT-256 across K, so every 256-wide group of h needs the whole group in one place.

- A V2C M128 tile covers 64 h rows (64 gate + 64 up).
- A V2B M256 tile covers 128 h rows.
- Neither tile holds a full FWHT group, so the epilogue cannot emit the rotated IU4 input directly. That would need an M512 tile (256 sums per lane), which is out of reach at V2B's 246 VGPRs.

What ships is therefore the two-stage variant:
1. The gate/up GEMM stores `h = silu(g)*u` once, as FP32 `[N][17408]`, into `pbs.gate_ffn_batch`.
2. The existing AWQ IU4 producer reads that single stream in phase 1. Everything from the AWQ divide on is unchanged: FWHT, `block_i4_128` A4 search, and the sidecar.

This removes one of the two f32 gate/up streams on both the write and the read side, one SET launch per FFN, and the SwiGLU arithmetic in the producer. The artifact, weights and decode path are untouched.

## Production change

**Kernels**

- `kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip` (gfx1100) and `…_v2b.gfx11.hip` (gfx1151) each gain an entry: `gemm_mq4g256v2_gate_up_silu_iu4_{v2c,v2b}_gfx11(G, U, Xq, H, M, K, N)`, grid `(N/T, 2M/T)`.
- **Row interleave at load time, done by address, not by moving weights.** Tile fragment `2p` holds gate rows and fragment `2p+1` holds the up rows of the same 16 outputs:
  - A staging wave takes its base from G or U (wave-uniform, still SGPR).
  - The per-lane scale pointer selects G or U by `lr>>3`.
  - No weight copy and no extra memory (a physical interleave would not fit a 24 GB XTX while decode still needs separate gate/up). The artifact is unchanged and the decode GEMVs are untouched.
- **Fold and epilogue.** The main loop and fold are the SET entry's.
  - The epilogue computes `((g) / (1.0f + expf(-(g))) * (u))`, spelled as the producer's `SILU_MUL`, and stores h.
  - The existing SET/ADD entries are **instruction-identical** to the base objects, on both archs and in both modules (`logs/build.log` DISASM lines).
  - Resources:

    | Entry | VGPR | Spill / private | CTAs/WGP |
    |---|---:|---|---:|
    | V2C F1 | 180 (SET: 179) | 0 | 2 |
    | V2B F1 | 246 | 0 | 1 |

- `kernels/src/fused_silu_mul_mq_rotate_awq.hip` gains `HIPFIRE_SILU_HIN`, which replaces the phase-1 gate/up reads plus SwiGLU with one h read. It builds `fused_silu_mul_mq_rotate_awq_i4_hin` (76 VGPR; the production producer uses 55). The non-HIN build is instruction-identical to the base.

**Rust**

- `gemm.rs`: the V2C/V2B eligibility rule moved unchanged into `Gpu::iu4_v2_tile` and `Iu4V2Tile`, and the SET/ADD route uses it. `gemm_gate_up_silu_mq4g256v2_iu4_prepared` fires only when:
  - both projections would take that same GEMM v2 tile on their own;
  - gate_m == up_m;
  - `HIPFIRE_F1LITE` is not `0`.

  Otherwise it returns `Ok(false)` and the SET pair runs.
- `gemv.rs`: `fused_silu_hin_rotate_mq_i4_batched` is the h-producer launcher.
- `kernels.rs`: `FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_HIN_SRC`.
- `qwen35/prefill.rs`, both FFN orchestrators (DeltaNet and full attention):
  - `f1lite_ffn_eligible` is decided once per FFN. It requires MQ4G256V2 gate/up/down, a down AWQ scale, a `Residual` epilogue, no S4 route, and the gfx11 IU4 sidecar live (eager only, N ≥ 64).
  - The gate/up hook returns whether it emitted h.
  - The down hook then runs the h-producer instead of `try_iu4_silu_prepared`.
  - Every other route (capture/replay, partial tiles, TP partial, non-AWQ, other archs) is unchanged.

## Standalone (before integration; `logs/`)

The harness is `f1-host.cpp` with objects from `build.sh`: hipfire's JIT argv plus `-DIU4_A4_CANDIDATES=2`, built locally with the hipx toolchain. It uses the non-periodic oracle `oracle_inputs.hpp` (sha `e9e9a0c1…`). Gate and up are separate allocations with U < G.

**Arms**
- A = base SET(gate) + base SET(up) + production `fused_silu_mul_mq_rotate_awq_i4`.
- C = F1 GEMM + h-producer.

**Oracle.** Both cards `ORACLE_PASS`, with 0 differing everywhere. Each case checks:
- a CPU fold-DAG spot check of both GEMMs;
- branch SET vs base SET;
- the JIT production producer vs the local one;
- C vs A on the f32 rotated row (x_rot) and on the `block_i4_128` bytes, both with and without the x_rot store.

| Case | Me | K | N | Notes |
|---|---:|---:|---:|---|
| Full (XTX) | 17408 | 5120 | 4096 | |
| Full (Halo) | 17408 | 5120 | 8192 | 142,606,336 x_rot, 80,216,064 B of `block_i4_128` |
| silu specials | 1024 | 1024 | 512 | 170,343 outputs with \|g\|>88, the exp-overflow path |
| multi-tile | 512 | 512 | 768 | |
| odd tiles | 768 | 768 | 1280 | 241,173 with \|g\|>88 |

`logs/xtx-oracle-invalidcase.log` is a first run whose odd-tile case used producer K=384. That K is not a multiple of 256, so the harness diffed the unwritten fill bytes. The case was fixed to Me=768.

**Timing.** Fresh processes in F/R/R/F order, ≥0.6 s warm-up, median of 7 HIP-event reps, mean of the process medians:

| Card | N | A pair+producer ms | C F1-lite ms | A/C (min–max) | GEMM A → C | Producer A → C |
|---|---:|---:|---:|---:|---|---|
| XTX | 4096 (the pp8192 call shape) | 9.564 | 9.244 | **1.0346** (1.0321–1.0366) | 8.626 → 8.714 | 0.942 → 0.532 |
| XTX | 8192 | 19.046 | 18.497 | 1.0297 (1.0284–1.0317) | 17.285 → 17.444 | 1.759 → 1.053 |
| Halo | 8192 | 44.107 | 41.381 | **1.0659** (1.0587–1.0715) | 38.338 → 38.414 | 5.768 → 2.967 |

On Halo the silu epilogue costs only +0.08 ms per call on the GEMM, while the producer saves 2.8 ms (it is bandwidth-bound and now reads one stream instead of two). FusionScreen's earlier Halo figure of +0.48 ms came from a physically interleaved harness in a window that a peer's load may have perturbed.

## Production proof on hipx (Stacker-granted leases, rocm-smi pre/post clean)

**pp8192 rocprof traces** (`run_profile.py`, `trace_delta.py`, `trace/*-delta.txt`)

- Method:
  - one warm request, then an uncached 8,192-token traced request;
  - ROCR/HIP, arch and `KV cache: Q8 vmm` asserted;
  - daemon stopped with `hipfire stop` so the CSVs flush.
- Arms: **A** is the same binaries with `HIPFIRE_F1LITE=0`. Its kernel cache holds no h-producer module, which also proves the kill switch. **B** is the default.

| Card | Kernel | A calls / ms | B calls / ms |
|---|---|---:|---:|
| XTX | gate/up SET (grid 1024×1088) | 256 / 1,071.08 | — |
| | F1 GEMM `…gate_up_silu…v2c` | — | 128 / 1,086.58 |
| | `fused_silu_mul_mq_rotate_awq_i4` | 128 / 120.27 | — |
| | `…_i4_hin` | — | 128 / 66.74 |
| | **Targeted FFN** | 1,191.35 | 1,153.32 (**−38.0 ms**) |
| Halo | gate/up SET (grid 1024×1088) | 128 / 2,576.76 | — |
| | F1 GEMM `…gate_up_silu…v2b` | — | 64 / 2,583.19 |
| | `fused_silu_mul_mq_rotate_awq_i4` | 64 / 365.80 | — |
| | `…_i4_hin` | — | 64 / 192.06 |
| | **Targeted FFN** | 2,942.56 | 2,775.25 (**−167.3 ms**) |

Whole trace:

| Card | Kernels A → B | GPU ms A → B | Request wall ms A → B |
|---|---|---|---|
| XTX | 4,492 → 4,364 | 3,150.8 → 3,116.7 (−34.0) | 3,213.9 → 3,162.6 |
| Halo | 4,849 → 4,785 | 8,143.9 → 8,004.7 (−139.1) | 8,196.9 → 8,053.1 (**−1.76%**) |

The other SET shapes run identical instructions and moved by at most +1.3% (noise).

Projection `[INF]`:
- Halo: 991.8 × 8,196.9/8,053.1 ≈ **1,010 tok/s**.
- XTX: about +1.1%, i.e. ~2,593 tok/s.

Stacker's ABBA is the gate for both.

**WT2 c24 q8/q8** (`run_wt2.py`; `eval_hipfire`; `HIPFIRE_GRAPH=0`). The h-producer was JIT-compiled in both runs.
- gfx1100: 0.076879. The sequence is byte-identical to the recorded mq4-lloyd `B.kldseq` (sha `8f2b94bb…`).
- Halo: 0.076901. Byte-identical (sha `eefd7d25…`).

**JIT objects.** hipfire's runtime compiler (radiowave) produced the F1 GEMM objects and the SET/ADD entries instruction-identical to the locally timed objects. It schedules the h-producer differently from local hipcc, as it does the production producer. `run_jit_oracle.sh` therefore re-ran the full oracle with C set to the runtime-JIT F1 GEMM plus the JIT h-producer, cross-checked against the JIT production producer from the `F1LITE=0` trace. Result: `ORACLE_PASS`, 0 differing, on both cards (`logs/*-jit-oracle.log`).

## Provenance

- **Build.** Release `hipfire`, `daemon` and `eval_hipfire` were built locally (same toolchain as hipx) from `fb5aa2b96` (= this branch's production tree) and copied to `/home/kaden/hipfire-f1lite/target/release`. SHAs:
  - `hipfire` `ef6956c1…`
  - `daemon` `bda4cacb…`
  - `eval_hipfire` `578b02d2…`
- **Compiles on hipx.** No cargo or hipcc compile ran on hipx. Kernels JIT on first use inside the leased runs.
- **Clean room.** A case-insensitive grep of `git log -p 2afc4a294..HEAD` for the excluded token returns 0.
