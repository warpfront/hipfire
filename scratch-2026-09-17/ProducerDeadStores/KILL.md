# Producer dead-store probe B — KILL

Base: `mq4-lloyd` `0252b7f85`
Branch/worktree: `gfx1201-producer-dead-stores` / `wt-prodb`
Device: gfx1201 ordinal 2
Decision: **KILL the first variant**. Do not route it in slice E.

The matched warm pp8192 calculation saved **3.319498 us/token**, below the required **6.399 us/token (~6.4)** gate by **3.079502 us/token**. This is about **0.767%** of the 433 us/token wall, below the required 1.5%. The second strided-ownership variant was intentionally not mixed into this first-variant result.

## Exactness and poison-canary result

The disposable device driver (`probe_test.rs`) forked real producer-created tensors from a live Qwen3.8-27B prefix prefill (first DeltaNet input at layer 2 and first full-attention input at layer 3). It did not upload random host tensors. Incumbent and candidate ran device-resident; comparison buffers were copied D2D, and only the compared bytes were read back.

At N=4096, all full 72-byte `block_i4_128` streams were byte-identical:

| Site | K | Compared bytes | Result |
|---|---:|---:|---|
| LA RMS qkvza producer | 5120 | 11,796,480 | PASS |
| LA GDN wo producer | 6144 | 14,155,776 | PASS |
| FA wo producer | 6144 | 14,155,776 | PASS |

For each candidate run, the skipped F32 destination was first device-poisoned with `0xA5`. The downstream LA qkv/z/beta/alpha outputs and the GDN/FA residual wo outputs remained byte-identical to the incumbent outputs. A separate N=128 run exercised the incumbent gfx1201 qkvza fallback with the same poison canary and passed, confirming the current gfx11-only small-tail attempt does not consume the F32 plane on gfx1201.

Matched c2 evaluation was also identical:

- baseline KLD: `0.049313`; mean NLL: `2.259701`; PPL: `9.5802`
- candidate KLD: `0.049313`; mean NLL: `2.259701`; PPL: `9.5802`
- `baseline-c2.bin` and `candidate-c2.bin`: byte-identical, SHA-256 `0e0b0f9914f429f71840a64c5be60bf3692e496a38bf3abd1865439a4e619079`

## Matched timing

Same ordinal, warm arms, 20 HIP events per arm per site, N=4096; candidate RMS includes the 1024-byte dynamic-LDS launch change.

| Site | Old mean us | New mean us | Old median us | New median us | Old span us | New span us | pp8192 old us/token | pp8192 new us/token | Net us/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LA RMS qkvza chain (48 sites) | 3577.004 | 3469.417 | 3617.393 | 3469.371 | 358.082 | 323.043 | 41.918016 | 40.657230 | 1.260785 |
| LA GDN wo chain (48 sites) | 1864.053 | 1731.240 | 1866.917 | 1744.735 | 114.641 | 117.441 | 21.844371 | 20.287969 | 1.556402 |
| FA wo chain (16 sites) | 1616.203 | 1487.611 | 1619.555 | 1508.154 | 130.561 | 127.680 | 6.313293 | 5.810980 | 0.502313 |
| **Total** | | | | | | | **70.075680** | **66.756178** | **3.319498 measured** |

A matched run with both RMS arms at 1024 bytes dynamic LDS measured 2.991752 us/token for dead-store suppression alone. The combined result is the acceptance number; the cross-run difference is not treated as an isolated LDS timing claim.

## Resources

Compiled for gfx1201 with `hipcc -O3 -Rpass-analysis=kernel-resource-usage`; runtime blocks/MP came from `hipOccupancyMaxActiveBlocksPerMultiprocessor` on ordinal 2.

| Kernel | Block | SGPR | VGPR | Static LDS | Dynamic LDS | Scratch/spills | Compiler waves/SIMD | Runtime blocks/MP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `rotate_x_mq_awq_i4_gfx12` | 32 | 25 | 76 | 0 | 0 | 0 / 0 | 16 | 64 |
| `gated_norm_mq_rotate_awq_i4_gfx12` | 64 | 28 | 45 | 1024 B | 0 | 0 / 0 | 16 | 32 |
| `fused_rmsnorm_mq_rotate_awq_i4_gfx12` | 256 | 34 | 91 | 0 | 21504 B old | 0 / 0 | 16 | 3 |
| `fused_rmsnorm_mq_rotate_awq_i4_gfx12` | 256 | 34 | 91 | 0 | 1024 B candidate | 0 / 0 | 16 | 8 |

RMS source audit: the IU4/AWQ kernel declares `reduce = smem` as `[256]` and only accesses `reduce[warp_id]`, `reduce[tid]` for `tid < 32`, and `reduce[0]`. No kernel offset depends on K; 1024 bytes is sufficient for this entry point.

## Source-audit conclusions

- GDN and FA prepared consumers accept only prepared IU4 blocks plus residual; they do not receive the F32 rotate surface.
- LA qkv/z already consume prepared IU4. On gfx1201, beta/alpha currently enter `small_tail_set_iu4`, whose f32 MW kernel rejects any arch except gfx1100/gfx1151 before reading x, then falls back to IU4. The canary confirms this dynamically. A clean shipping route would make the IU4 selection explicit.
- Traffic removed by this variant would be 983,040 B/token (LA RMS), 1,179,648 B/token (GDN), and 393,216 B/token (FA), total 2,555,904 B/token.

## Exact slice-E host changes that would be required if revived

These were exercised only by the disposable driver/quality binary and are **not shipped**:

1. `crates/rdna-compute/src/gemv.rs`
   - At `fused_rmsnorm_rotate_mq_i4_gfx12_batched` (current launch line 3350), replace `((k + 256) * 4)` dynamic LDS with `256 * size_of::<f32>()` (1024 B) only for this IU4 gfx12 entry.
   - Change `gated_norm_rotate_mq_i4_gfx12_batched` argument `x_rot: &GpuTensor` to `Option<&GpuTensor>`; encode a null ABI pointer for `None`, and invalidate x caches only when non-null.
   - Change `rotate_x_mq_i4_gfx12_batched` argument `x_out: &GpuTensor` identically.
2. `kernels/src/gated_norm_mq_rotate_quant.gfx12.hip`
   - Permit nullable `x_rot`; preserve all IU4 calculations and guard only the coalesced F32 store.
3. `kernels/src/mq_rotate_x_i4.hip`
   - Permit nullable `x_out`; preserve all IU4 calculations and guard only the coalesced F32 store.
4. `crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
   - LA RMS call near current 5342: request `emit_f32=false` only on exact gfx1201 after item 5 below is in place.
   - `try_gfx12_gdn_quant_fused_prepared` near current 256/279: pass `None` only for its already-admitted exact-gfx1201, uniform MQ4G256V2, Residual, head_dim=128 prepared route. This covers dense LA GDN wo at producer 6081 / prepared consumer 6176 and its admitted sibling at 9258 / 9410.
   - `try_gfx12_rotate_quant_fused_prepared` near current 218/234: pass `None` only for its already-admitted exact-gfx1201 Residual prepared route. This covers the dense LA fallback at 6149 / 6176, FA wo at 7686 / 7712, and any already-admitted sibling prepared consumers (including 10045); every route with an F32 reader must keep `Some(...)`.
5. `crates/rdna-compute/src/gemm.rs`
   - In `gemm_qkvza_mq4g256v2_wmma_iu4_prepared` near current 29290, exact gfx1201 must call `gemm_mq4g256v2_mmq_set_prequant_iu4` directly for beta and alpha (as it already does for qkv and z), bypassing `gemm_mq4g256v2_small_tail_set_iu4`, its output memset, and its rejected gfx11 MW attempt. Keep gfx1100/gfx1151 behavior unchanged. This is the explicit proof that LA no longer needs x.
6. Preserve all public kernel entry symbols and parameter ordering. This probe used nullable pointer values in existing ABI slots; no new runtime flag or environment flag is needed.

The complete probe diff is preserved as `candidate-full-probe.patch`; `probe_test.rs` is the disposable real-input device driver. The tracked worktree has been restored to the base because the ship gate failed. No commit was created.
