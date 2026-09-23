# Probe A — KILL

The gfx1201 up-GEMM epilogue candidate is exact on the bounded device probe and meets the resource gate, but regresses the matched FFN chain. It is not eligible for slice E integration.

## Candidate

- Kernel: `kernels/src/gemm_up_silu_mq4g256v2_iu4.gfx1201.hip`
- ABI: `(Wup,Xin,Gate,Awq,S1,S2,Down,I,K,N)`
- Geometry: 256 features × 64 tokens, 256 threads / 8 waves
- Dynamic LDS: 25,600 bytes; dead compute planes are reused as an 8-token × 256-feature F32 producer overlay
- Producer arithmetic and `emit_iu4_sidecar_from_producer8` quantization preserve the incumbent expression and reduction order

## Resources

ROCm 10 gfx1201 compile (`-O3`, kernel-resource-usage remarks):

- 208 VGPR
- 48 SGPR
- 0 bytes scratch/lane
- 0 VGPR spills, 0 SGPR spills
- compiler occupancy 7 waves/SIMD
- runtime dynamic LDS 25,600 bytes
- runtime occupancy 2 blocks/MP

Evidence: `resource.log`, `probe-final.log`.

## Byte identity

The disposable driver creates Xin with the device `quantize_int4_mmq_ds128`, then creates Gate and the incumbent Up with the real device `gemm_mq4g256v2_residual_mmq_iu4_full_set`. It forks those device-produced values into old Up→`old_silu_mul_mq_rotate_awq_i4_gfx12` and the candidate. No host gate/up matrix is used. Inputs cover all-zero tokens/features, alternating half-step-like values, outliers, changing 128-K and 256-K group boundaries, constant/zero-scale weight rows, and differing split headers.

Device-resident byte comparison of every 72-byte Down block:

- N=128: 1,253,376 bytes, 0 mismatches
- N=4096: 40,108,032 bytes, 0 mismatches

This is a producer-created device oracle, but not a loaded-model live-prefill capture. Because the independent performance gate already kills A, no model interception or host integration was funded.

## Matched chain timing

Ordinal 1 only; required HOME/model/cache/graph/Lloyd environment; both arms warmed 20 times; 20 HIP-event samples per arm. Each arm includes gate, up/epilogue, and down:

- Old: gate + up + standalone producer + down = 12,462.078 us median at N=4096
- New: gate + fused up/producer + down = 14,092.228 us median at N=4096
- Delta: -1,630.150 us per layer-chain
- Per layer: -0.397986 us/token
- 64-layer network-equivalent: -25.471085 us/token
- Ship threshold: at least +6.4 us/token

Verdict: **KILL_PERFORMANCE**. The candidate is about 13.1% slower for the matched chain and misses the ship gate by 31.871 us/token on the 64-layer equivalent. No KLD downgrade and no slice E routing.

## Slice E host work that would have been required after a pass

No host change should be made for this killed candidate. If a future exact and fast replacement passes the same gates, slice E must make this complete cutover:

1. `crates/rdna-compute/src/kernels.rs`: register the new source with the shared block-int4 prelude and its exact gfx1201 symbol.
2. `crates/rdna-compute/src/gemm.rs`: add a dedicated wrapper that launches the candidate and then existing `gemm_mq4g256v2_mmq_add_prequant_iu4` on the same stream. It must borrow the new Raw Down arena directly, never forge `Int4MmqPrepared`, never reserve intermediate int4 scratch, and return only after both launches enqueue successfully.
3. `crates/rdna-compute/src/scratch.rs`: add a distinct preallocated `PrefillBatchScratch.ffn_epilogue_i4: Raw`, checked at `N * I / 128 * 72` bytes (40,108,032 bytes at N=4096). Xin's prepared reservation/generation remains live and may not alias Down.
4. `crates/hipfire-arch-qwen35/src/qwen35/batch.rs`: allocate/free the optional Raw Down arena in `PrefillBatchScratch::{new_opt,new_opt_with_alloc}` and `free`, controlled by one precomputed `cap_ffn_epilogue_i4` boolean.
5. `crates/rdna-compute/src/feature_flags.rs`: add the experiment admission/capability control, default off for probing; a reviewed winner must become flag-free on the target.
6. `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`: under exact-gfx1201 dense uniform MQ4V2, iu4+fp8-KV, ordinary sequential eager prefill, N>=128 divisible by 128, normal Residual down-transform admission, replace gate/up sites around 6628 and 8219 plus producer/down sites around 6915/6944 and 8498/8525. Keep the unchanged gate writer, route candidate instead of old up+producer, then consume Down once and update `pbs.x_batch` once with the old residual.
7. Allocation plumbing callers `prefill.rs`, `forward.rs`, `ep_batch.rs`, and `serve_engine.rs`: migrate constructor/projection signatures and pass the same immutable `cap_ffn_epilogue_i4` to checked `projected_allocation_bytes`; disabled means zero A allocation.
8. Failure/state contract: prepare kernels and capacity before conv/KV/residual mutation; Gate stays immutable while read; Xin and Down never alias; Down is fully written before consumption; after successful candidate mutation there is no fallback/retry. Unsupported tails and every non-gfx1201/other-mode route retain the incumbent path.

`gemv.rs` and `norm.rs` need no A-specific edits; they are in slice E's shared ownership list for the broader A/B/C integration only.
