# G12 A4C2: one-pass {5,7} activation search in the gfx1201 IU4 producers — KEEP

- **Branch:** `g12-a4c2`, worktree `wt-a4c2`, from `mq4-lloyd` @ `0f6cea0dd`.
- **Code commit:** `44ca584ad` (evidence commit follows it).
- **Kill switch:** `HIPFIRE_G12_A4C2=0` (`kernel.g12_a4c2`) restores RTN. Default on for exact gfx1201.
- **Clean-room:** a case-insensitive grep for the clean-room token over `git log -p 0f6cea0dd..HEAD` returns 0.

## Verdict

At pp8192 the four producers cost **+1.81 ms** in aggregate with c2 on. That is the same-binary rocprof mean over ABBA pairs on card-E. The generic `-DIU4_A4_CANDIDATES=2` build costs **+68.27 ms** measured the same way. The pass bar is +5 ms, so this passes.

WT2 c24 reproduces the c2 flag arm byte for byte: **0.076071**, kldseq md5 **`5ac9c345e4483b7c317f48e265c351f5`**.

### Per-producer ms at pp8192 (rocprof, card-E, one warmed uncached request, kernels inside the request window)

| Producer | Calls | RTN (new bin, off, mean of 2) | c2 flag (base bin + flag) | c2 new (new bin, on, mean of 2) | new − RTN | flag − RTN (base bin) |
|---|---:|---:|---:|---:|---:|---:|
| `fused_rmsnorm_mq_rotate_awq_i4_gfx12_v2` | 128 | 50.527 | 84.089 | 52.158 | **+1.631** | +33.505 |
| `fused_silu_mul_mq_rotate_awq_i4_hin_gfx12` | 64 | 73.137 | 105.616 | 73.251 | **+0.114** | +32.449 |
| `gated_norm_mq_rotate_awq_i4_gfx12_v2` | 48 | 35.864 | 37.673 | 35.915 | **+0.051** | +1.835 |
| `sigmoid_mul_rotate_x_mq_awq_i4_gfx12` | 16 | 11.294 | 11.798 | 11.306 | **+0.011** | +0.477 |
| **Four producers** | | 170.822 | 239.176 | 172.629 | **+1.807** | **+68.266** |

- **Raw arms** (`rocprof_summary.txt`):
  - Four producers, new binary: off1 170.696, on1 172.546, on2 172.712, off2 170.947 ms.
  - Four producers, base binary: RTN 170.910, c2 flag 239.176 ms.
  - GPU total: 2040.3 / 2040.0 / 2045.3 / 2050.6 ms for the new-binary arms; 2048.5 (RTN) / 2112.9 (flag) for the base binary.
  - Every arm ran 3,633 dispatches and emitted token `'A'`.
- **IXPlan's earlier trace:** it measured +65.7 ms for the flag. The base-binary rows above are that same build (daemon md5 `a2e12566…`, cli `094d4cf9…`).

### Standalone ms (card-E, N=8192, production argv)
Method: `standalone_timing.txt`. Each (variant, producer) pair runs in a fresh process, with 3 interleaved rounds. Each value is the median of 30×20 launches.

| Producer | RTN | c2 flag | c2 new | new − RTN (×calls) | flag − RTN (×calls) |
|---|---:|---:|---:|---:|---:|
| RMS AWQ v2 K5120 | 0.3352 | 0.5389 | 0.3346 | −0.08 ms | +26.07 ms |
| SwiGLU hin K17408 | 1.1473 | 1.4584 | 1.1459 | −0.09 ms | +19.91 ms |
| gated AWQ v2 K6144 | 0.7404 | 0.7505 | 0.7416 | +0.06 ms | +0.48 ms |
| sigmoid AWQ K6144 | 0.7431 | 0.7461 | 0.7419 | −0.02 ms | +0.05 ms |

- **Standalone:** all four producers sit at the RTN floor (570–583 GB/s against the 583 GB/s D2D copy).
- **Pipeline:** RMS keeps +1.6 ms (+3.2%). Standalone it is at parity. Inside the pipeline the kernel runs at 0.395 ms/call, not 0.335, so the extra ALU work shows there. `[INFERENCE]` The likely cause is card clock/power state during the GEMM-heavy prefill; I did not measure clocks.

## Why the flag build was slow

`quantize_block_i4_128_wave<true>` with `IU4_A4_CANDIDATES=2` costs, per 128-block and per lane:
- 8 IEEE divides for the two candidates, plus 4 more for the final codes (≈11 VALU each);
- two serialized 5-step `ds_bpermute` mse butterflies, on top of the remap (8 permutes) and the amax/s butterflies (10).

On gfx1201 the extra ≈130 VALU plus 10 LDS round trips per block made the RMS and SwiGLU producers ALU/latency bound (RMS 0.335 → 0.539 ms, SwiGLU 1.147 → 1.458 ms).

## The change (`kernels/src/block_i4_128_quant.hip`, gfx1201 + `IU4_A4_CANDIDATES == 2` only)

The gfx1201 `emit_iu4_sidecar_from_producer8` calls `emit_iu4_sidecar_c2_gfx12`, which quantizes in **producer ownership**:
- Lane p holds 8 consecutive values of half `p >> 4`, so each DPP row of 16 lanes is one 128-block.
- Both halves are quantized at once, with no remap and no LDS permutes.

**Bit-identity argument:**
- **amax:** the max of |x| does not depend on order. There is no −0 after `fabsf`, and `max_num` ignores NaN exactly as the butterfly does. It is reduced by `row_xmask` 8, 4, 2, 1.
- **mse, both candidates:**
  - Remap lane `l = 2j + b` owns `x[4b..4b+3]` of producer lane j, so its fma chain is this lane's chain `acc[b]`.
  - The remap's xor-16/8/4/2 butterfly steps are xor-8/4/2/1 on j, run as DPP `row_xmask` adds.
  - Its final xor-1 step adds the lane's two chains, and + is commutative.
  - Both candidates' chains run in one pass over the registers already loaded, and their reductions interleave.
- **s:** an exact integer sum, reduced with DPP.
- **qs:** bytes 4j..4j+3 are this lane's 8 codes in order, written as one u32 store.
- **codes:**
  - `r = rcp(d)` gets one Newton step. Then `|r·d − 1| ≤ 2^-23` even if the rcp were only 2^-12 accurate.
  - For amax in [2^-100, FLT_MAX], both d are normal and |x/d| < 8.17, because d0 = (6/7)·amax/7.
  - So `|RN(x·r) − x/d| < 2^-19.3`, including the case where the backend contracts the check's subtraction.
  - If every `|y − rint(y)| ≤ 0.5 − 2^-18`, then `rint(y) == rint(RN(x/d))`. Candidate-1 codes then lie in [−7, 7], so they need no clamp.
- **Redo path:** any wave with a value inside the window, or with amax that is zero-adjacent, tiny, Inf or NaN, redoes both candidates with the reference divides. It is rare and wave-uniform. The test runs after the mse pass, so the common path stays one basic block.
- **best_d = 1 cases:** these are amax == 0 (codes forced to 0), NaN elements, and mse ≥ 1e30. They reproduce the reference's `rint(x/1)`.

**Rejected variants** (standalone RMS, ms, RTN 0.335):

| Variant | ms | Why rejected |
|---|---:|---|
| Wave-level fallback to the reference remap | 0.341 | 113 VGPRs, occupancy 12 |
| Mid-function divergent per-lane divide fix-up | 0.343 | Splits the hot block |
| Recomputing the chosen codes | 0.344 | |
| Forcing 16 waves (`__launch_bounds__(256,16)`, 84 VGPRs) | 0.348 | |

The kept version runs at 101 VGPRs (occupancy 12) on RMS, and 76 / 58 / 81 VGPRs on SwiGLU / gated / sigmoid (occupancy 16).

**Host:**
- `feature_flags.rs` appends `-DIU4_A4_CANDIDATES=2` to the gfx1201 JIT flags unless `HIPFIRE_G12_A4C2=0`. This is the same mechanism gfx11 uses for `HIPFIRE_GFX11_A4_CANDIDATES`.
- The field is registered in `hipfire-config` (`kernel.g12_a4c2`) and documented in `docs/env-vars.md`.
- `check-env-docs.py` passes, and `hipfire-config` unit tests pass (77).

## Gates

1. **Bitwise oracle** against the flag build's codes and scales. The reference is 0f6cea0dd's `kernels.rs` concatenations with the production argv plus `-DIU4_A4_CANDIDATES=2`.
   - **Producer matrix** (`oracle_matrix.sh` → `oracle_matrix.txt`): `ORACLE_MATRIX_PASS`, 144 cases, 0 differing bytes.
     - Sources: all 23 gfx1201-compiled sources that call the emit: RMS v1/v2/generic/FOLD, gated v1/v2, sigmoid, rotate ×4, SwiGLU gfx12/generic/hin.
     - Shapes: production shapes, plus tails (N = 1, 17, 33, 64, 65, 77, 333, 513, 777, 1000; K = 512…17408, including odd group counts) and x_rot on/off.
     - Data: nonperiodic random data with channel outliers, and special rows (zero, −0, denormal, 1e17/1e20, Inf, NaN, spikes, powers of two). Every run reports "untouched blocks: 0".
   - **Direct emit oracle** (`qemit.hip` + `test_c2.cpp` → `qc2_oracle.txt`): 1,294,342 blocks, 0 differing bytes.
     - ±64/±200-ulp walks around every half-integer and multiple of both d0 and d1, for about 5,000 amax values per seed. These cover the admission edges 2^-100 and FLT_MAX, plus denormal and huge values.
     - 36k blocks sit entirely just outside the 2^-18 window.
     - Also covered: clamp edges ±7.5·d0 and −8.5·d0; few-valued blocks (mse ties); NaN, Inf, zero and −0 blocks.
   - **Coverage controls:**
     - With the redo disabled, the edge-only set still passes, so the reciprocal path is exact at the window.
     - With the redo disabled, the full mix fails in 3,259 blocks, so the redo path is exercised.
     - The RTN build fails, so the oracle discriminates.
2. **Standalone ms:** see the table above.
3. **rocprof pp8192, same binary, off vs on, card-E:** +1.807 ms across the four producers. See the table.
   - Runner: `run_trace.py` via `run_traces.sh`. The server is stopped with `hipfire stop` and the wrapped daemon is SIGTERMed only if still alive, so the CSVs flushed.
   - Fixture md5: `8e588120112ee22d494bc4900166676f`.
   - FP8 VMM was asserted in every serve log.
4. **WT2 c24, card-D** (evaluator from this commit, `run_wt2.py` → `wt2/`):
   - **On (default):** KLD **0.076071**, kldseq md5 **`5ac9c345e4483b7c317f48e265c351f5`**, identical to IXPlan's c2 flag arm. sha256 `e2242712…f773`.
   - **Off (`HIPFIRE_G12_A4C2=0`):** KLD **0.083278**, md5 `9d0e860f41db992820ebdc9483c0a041`, sha256 `1dd3caed…8711`. This is identical to the G12Norm2 RTN baseline.
5. **gfx11 unaffected** (`isa_identity.py` → `isa_identity.txt`, `ISA_IDENTITY_PASS`):
   - Scope: every one of the 34 `kernels.rs` sources that include the quant file, compared against 0f6cea0dd.
   - Instruction-identical on gfx1100 and gfx1151 with `-DIU4_A4_CANDIDATES=2` (the production gfx11 default) and with `=8`, and on gfx1201 without the flag (the kill-switch arm).
   - With the flag on gfx1201, only the 26 emit-calling producer sources differ. The 8 GEMM/standalone-quantizer sources stay identical.

## Notes
- **Cache keys:** the gfx1201 default extra flags change, so every gfx1201 JIT cache key changes once. This matches what gfx11's A4 default already does.
- **Packaged installs:** `scripts/compile-kernels.sh` hashes packaged blobs with the env `HIPFIRE_HIPCC_EXTRA_FLAGS` only. With the new default, packaged gfx1201 blobs will read as stale and be recompiled when hipcc is present. Without hipcc, the runtime falls back to the unvalidated old (RTN) blob. gfx11's A4 default has the same pre-existing property.
- **Card use:**
  - Card-E: timing and oracles.
  - Card-D: WT2, announced by hub and released afterwards.
  - After the traces no daemon of mine is left on any card (`trace/*/gpu-post.txt`).

## Files

| File | Contents |
|---|---|
| `build.sh`, `var.sh`, `gen_prod.py` | Exact `kernels.rs` concatenation plus production hipcc argv; variant builds |
| `harness.cpp`, `oracle_matrix.sh` / `.txt`, `syms.txt`, `consts.txt` | Producer oracle and timing harness, matrix and output |
| `qemit.hip`, `test_c2.cpp`, `qc2_oracle.txt` | Direct emit oracle |
| `tm.sh`, `standalone_timing.txt` / `_raw.txt` | Standalone timing |
| `isa_identity.py`, `isa_identity.txt` | ISA identity |
| `run_trace.py`, `run_traces.sh`, `trace_delta.py`, `trace/*/`, `rocprof_summary.txt` | rocprof runs and summary |
| `run_wt2.py`, `wt2/` | WT2 runs |
| `bin/SHA256SUMS` | Frozen binaries: new (this commit), base (IXPlan's 0f6cea0dd build) |
| `exp/q_va.hip`, `exp/q_vb.hip`, `exp/q_fastonly.hip` | Rejected variants (reference-remap fallback; mid-function fix-up) and the redo-disabled coverage control |
