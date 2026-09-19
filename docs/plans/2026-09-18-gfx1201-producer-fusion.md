# gfx1201 prefill producer → activation-quant fusion

**Branch/worktree:** `gfx1201-producer-fusion` in `/home/kaden/ClaudeCode/warpfront/wt-glue`, based on `d21c95096` (`mq4-lloyd`). All commands and evidence in this plan use that worktree. Never write to `wt-lloyd` or `hipfire-beta`; never stash. GPU work uses ordinal 1 after announcing it on hub. Ordinal 3 is off limits.

## 1. Measured ceiling and stop conditions

The denominator is the supplied full-stack pp8192/chunk4096 trace at `/home/kaden/ClaudeCode/warpfront/wt-lloyd/.codeinsight+research/scratch-2026-09-17/chunkwide/trace/prof-c4096/run_kernel_stats.csv`: two measured passes, so each row below is `total_duration_ns / 2 / 8192 / 1000`. Total wall is **487 us/token**.

| Trace bucket | Calls | us/token | Share of 487 | Interpretation |
|---|---:|---:|---:|---|
| `quantize_int4_mmq_ds128` | 1024 | 47.937 | 9.84% | All IU4 activation prepares; arithmetic remains after fusion, so 9.84% is a hard ceiling, not a forecast. |
| `fused_silu_mul_mq_rotate_awq` | 256 | 21.982 | 4.51% | FFN-down producer. |
| `fused_rmsnorm_mq_rotate_awq` | 512 | 8.652 | 1.78% | Projection-input producer. |
| `gated_norm_f32` | 192 | 5.747 | 1.18% | GDN output producer before rotate. |
| `rotate_x_mq_awq` | 256 | 4.868 | 1.00% | 192 GDN-output and 64 full-attention-output calls, subject to trace confirmation. |
| `deinterleave_q_rmsnorm` + `sigmoid_mul` | 64 + 64 | 4.661 | 0.96% | Full-attention row glue; only the producer-to-rotate portion is in scope. |
| `gdn_pre_batched_gfx1201` | 192 | 15.521 | 3.19% | Upstream GDN preparation, **not** the gated-norm→rotate→quant pair. Do not book it as a fusion saving. |

The six requested non-GEMM rows sum to **93.848 us/token**, so their absolute Amdahl ceiling is **19.27%** of 487 us/token. Most producer math and all quant arithmetic remain; the credible prize is removing launches plus the intermediate F32 write/read, not deleting 93.848 us of computation.

Because the single quant row is an aggregate, call-count allocation is an estimate until a fused/unfused trace attributes each site:

| Candidate chain | Allocated incumbent bucket | Hard Amdahl ceiling | First useful target |
|---|---:|---:|---:|
| RMSNorm+rotate (512 calls) + IU4 quant | 8.652 + 23.968 = **32.621 us/token** | 6.70% | Admit only if fused `TIME` ≤ this exact measured chain; 23.968 us is the quant-share upside estimate, not a promise. |
| silu·up+rotate (256) + IU4 quant | 21.982 + 11.984 = **33.966 us/token** | 6.97% | P0: admit only if fused `TIME` ≤ this chain in three fresh processes. |
| GDN gated norm (192) + 192/256 of rotate + IU4 quant | 5.747 + 3.651 + 8.988 = **18.386 us/token** | 3.78% | Admit only if fused `TIME` ≤ the attributed incumbent chain. |
| Remaining 64 rotate sites + IU4 quant | 1.217 + 2.996 = **4.213 us/token** before producer glue | 0.87% | First identify the 64 calls; do not implement if fusion is slower than the complete attributed chain. |

The allocation assumes uniform `quantize_int4_mmq_ds128` cost per call. That is **[INFERENCE]**; the first profiler gate must replace it with site-specific measurements.

### Gate P0 — cheapest experiment that can kill this design

Use the K=17408 AWQ silu·up producer as the first probe, but keep its incumbent `[K/256,N]`, block-32 ownership: each wave already owns one complete 256-element FWHT group and its final eight floats/lane. Feed those registers directly to the shared two-block IU4 emitter. This is cheaper and lower risk than serializing the row into one workgroup.

The P0 candidate contract is `fused_silu_mul_mq_rotate_awq_i4_gfx12`, grid `[K/256,N]`, block 32, exact-gfx1201 host admission. Its initial resource target is the measured 55 VGPR / 18 SGPR / 0 spill / 0 LDS for AWQ; the plain twin budget is 84 / 18 / 0 / 0. Reproduce those numbers on the final object before timing.

Abandon this producer-local fusion before copying it to the other producers if any is true on gfx1201:

1. `-Rpass-analysis=kernel-resource-usage` exceeds the stated register budget or reports any scratch/spill.
2. In three fresh-process pp8192 traces, fused-kernel `TIME` is greater than the sum of `fused_silu_mul_mq_rotate_awq` plus the 256 eliminated `quantize_int4_mmq_ds128` calls.
3. Any of the 72-byte IU4 blocks differs from the standalone producer→quant chain on real K=17408 activations.
4. The fused symbol is not replay-safe in the graph oracle.

**Current P0 screen, not a promotion result:** `/home/kaden/ClaudeCode/warpfront/wt-glue/scratch-2026-09-17/silu-quant-fused/gate3_compare.txt:1-9` shows the expected 256 quant calls disappear (1024→768), with the attributed down chain 46.3→30.9 us/token and total-kernel time 10.5 us/token lower in that one A/B. The exact resource reports at `/home/kaden/ClaudeCode/warpfront/wt-glue/scratch-2026-09-17/silu-quant-fused/metadata_awq_gfx12.txt:1-11` and `/home/kaden/ClaudeCode/warpfront/wt-glue/scratch-2026-09-17/silu-quant-fused/metadata_plain_gfx12.txt:1-11` establish 55/18 and 84/18, zero spills, zero LDS, and 16 waves/SIMD. This authorizes the oracle and three-fresh-process gate only; it lacks the required repeated-process, prompt/binary-md5, graph, and model-quality evidence, so it is not yet a claimed win.

Do not substitute an upstream competitor baseline. `RADIANCE_FP8_STREAM_TP1` reportedly reduced competitor decode ms/step by 5%; hipfire prefill magnitude is unmeasured, and its tuned FP8-v2/IU4 baselines are the only comparison.

## 2. Current source contracts

Line anchors name the `d21c95096` source region; implementers must re-read the symbol before editing because the branch may contain concurrent work.

### Quant and producer math

- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/block_i4_128_quant.hip:21-29`, `struct block_i4_128`: exactly 72 bytes: `float d`, `int s`, then 64 bytes of packed nibbles.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/block_i4_128_quant.hip:31-88`, `quantize_block_i4_128_wave`: lane owns four consecutive values; wave XOR reduction finds max; eight MSE-clip candidates use strict `<`; values use `rintf`, clamp, integer sum, and even-K-low/odd-K-high nibble packing. Every IU4 sink must call this recipe, not copy or alter it.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/block_i4_128_quant.hip:90-123`, `emit_iu4_sidecar_from_producer8`, remaps the producer's eight final floats/lane into the standalone four-consecutive-values/lane ownership, calls the one quant recipe twice, and stores `(2*group+h)*N+token`. This is the preferred IU4 fusion primitive.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/block_i4_128_quant.hip:130-150`, `quantize_int4_mmq_ds128`: `grid=(ceil(K/1024), N)`, block 256, output address `[kblock * N + token]`. This layout is consumed directly by `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx12.hip:282-285`.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/fused_rmsnorm_mq_rotate.hip:43-63`, `fused_rmsnorm_mq_rotate*`: row entry; `:64-138` fixes the F32 sum/reduction and RMS order; `:142-233` fixes AWQ/sign/FWHT order.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/fused_silu_mul_mq_rotate_awq.hip:29-45`, `fused_silu_mul_mq_rotate_awq`: `:66-88` fixes `g / (1 + exp(-g)) * up` followed by AWQ/sign; `:90-133` fixes the FWHT butterfly order.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gated_norm.hip:9-37`, `gated_norm_f32`, fixes the per-head gated RMS math. `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gated_norm_mq_rotate.gfx1100.hip:16-135` is the fused gated-norm/rotate structural reference, but must not be changed for a gfx1201-only experiment.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/rotate_x_mq_awq.hip:23-101`, `rotate_x_mq_awq`, fixes the standalone AWQ/sign/FWHT order.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gdn_pre_batched.gfx1201.hip:7-13,58-84` produces GDN q/k/v and beta/alpha planes. It is not the post-GDN gated-normalization producer in this project.

### Existing FP8-v2 contract

- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip:8-22,89-180,192-283` is one 256-thread workgroup per token row. It emits row-major E4M3 bytes, two F32 decoded-code half sums per K/256 group, and one F32 row scale.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/scratch.rs:1124-1312`, `prepare_mq4v2_fp8_x`, `prepare_mq4v2_fp8_x_f32`, and their implementation return `Mq4v2Fp8Prepared`.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip:328-381` fixes the consumer ABI `(X_fp8, X_half_sums, X_row_scales)`; `:1169-1200,1352-1357` shows the row-major loads, half-sum correction, and F32 scale use. Producer fusion must first match this contract exactly. A new F16-scale stream is a separate, explicitly lossy ABI.

### Rust routing and live buffers

- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/scratch.rs:1488-1582`, `ensure_int4_mmq_x`, owns the standalone IU4 quant launch and scratch generation.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/gemm.rs:19157-19369`, `gemm_mq4g256v2_mmq_prequant_iu4`, is the typed prequant consumer. Projection-specific prepared consumers are near `:28883-28924`, `:29446-29465`, `:30664-30682`, and `:32270-32282`.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5021-5169`, `batch_chunk_delta_net_input_projection`, supplies RMSNorm/rotate to QKVZA; `:5718-5849`, `batch_chunk_delta_net_output_projection`, supplies gated-norm/rotate to GDN `wo`; `:6149-6274`, `batch_chunk_delta_net_ffn_gate_up`, supplies RMSNorm/rotate to gate/up; `:6436-6536`, `batch_chunk_delta_net_ffn_down`, supplies silu·up/rotate to down; `:7155-7240`, `batch_chunk_full_attn_output_projection`, is the 64-call full-attention producer chain.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:235-482`, `dispatch_batched_gemm_epilogue`, performs a separate `add_inplace_f32` only on Q8/low-bit or partial/all-reduce routes. The hot single-GPU IU4 residual GEMM already adds into `x_batch`; there is no separate residual-add launch to claim in the supplied IU4 trace.
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/batch.rs:40-71` defines the authoritative F32 residual and producer buffers: `x_batch`, `x_rot_batch`, `dn_normed_batch`, `dn_normed_rot_batch`, and `ffn_hidden_batch`; `:87-94` defines the full-attention planes. `x_batch` remains authoritative F32.

## 3. Frozen device design

### 3.1 Producer-local IU4 topology and resource budgets

IU4 quantization is local to 128 values; the FWHT is local to 256. It has no whole-K dependency. Therefore the preferred design keeps each producer's incumbent ownership and emits two 72-byte blocks directly from its final registers:

1. Preserve the producer's current lane assignment and F32 expression order.
2. At the point where each wave holds the final eight rotated floats/lane, call `emit_iu4_sidecar_from_producer8`.
3. The helper shuffle-remaps those registers into the standalone quantizer's four-consecutive-values/lane ownership, invokes `quantize_block_i4_128_wave` for each 128 half, and stores `[kblock*N+token]`.
4. Drop the intermediate F32 store when no other consumer needs it. If another consumer needs it, retain the store but still do not reload it for quantization.

This requires no row gather, no added global traffic, no added LDS, and no barrier beyond the producer's incumbent synchronization:

| Producer | Incumbent/fused ownership | K handling | Admission resource budget |
|---|---|---|---|
| silu·up AWQ/plain | one wave per 256 group, grid `[K/256,N]`, block 32 | K=17408 is 68 independent groups; no half-row buffer | AWQ ≤55 VGPR/18 SGPR; plain ≤84/18; 0 LDS; 0 spill/scratch |
| RMSNorm+rotate | one 256-thread row workgroup; each wave loops whole 256 groups after the unchanged whole-row RMS reduction | K=5120 is 20 groups; quant stays wave-local | ≤64 VGPR/32 SGPR; only incumbent 1,024-B reduction LDS; 0 spill/scratch |
| GDN gated norm+rotate | waves own complete head pairs/256 groups after the unchanged per-head reduction | loop groups for the token row; no full-row buffer | ≤64 VGPR/32 SGPR; no new LDS; 0 spill/scratch |
| sigmoid/full-attention rotate | one wave per 256 group if the trace gate admits it | group count from runtime K; no full-row buffer | ≤64 VGPR/32 SGPR; no new LDS; 0 spill/scratch |

K=5120 and K=17408 therefore do not approach the 64-KiB LDS ceiling on the IU4 path. A row-workgroup/LDS variant is a fallback only if a future producer cannot expose a complete 256 group in one wave. That fallback may stage at most 5,120 F32 values (20,480 B) or two 8,704-F32 halves (34,816 B each); it must still compute and quantize in the same workgroup. A separate gather that writes/reloads an F32 row is rejected because it restores the traffic this project removes.

When the intermediate F32 row is dead after its GEMM consumer, IU4 fusion removes an `8K`-byte producer-store/quantizer-read round trip per token: 40,960 B at K=5120, 139,264 B at K=17408, and 32,768 B at K=4096. The IU4 output itself remains `72*(K/128)=0.5625K` bytes (2,880 B, 9,792 B, and 2,304 B respectively). If the F32 output remains live, only the `4K` quantizer read disappears; report that smaller traffic saving for the site.

Unit admission is simple: fused-kernel `TIME` must be no greater than the exact producer-plus-eliminated-quantizer `TIME` it replaces. A call-count reduction without that timing result is not a win.

### 3.2 Compile-time output template

Each producer body is a compile-time template, never a runtime branch:

```cpp
enum ProducerOutput : int {
    PRODUCER_F32 = 0,
    PRODUCER_IU4_BLOCK128 = 1,
    PRODUCER_MQ4V2_FP8 = 2,
    PRODUCER_FP8_STREAM_F16 = 3,
    PRODUCER_MQ4V2_FP8_F16_SCALE = 4,
};

template<int OUTPUT, bool EMIT_F32>
__device__ void producer_body(/* incumbent producer ownership, typed sink pointers */);
```

`EMIT_F32` is separate because QKVZA small tails or the authoritative residual may still need the F32 row while the matrix consumer takes a quantized sidecar. Every exported entry instantiates a fixed `OUTPUT`/`EMIT_F32`; do not pass a runtime format integer to a captured graph.

Frozen output contracts:

- `PRODUCER_F32`: incumbent row-major F32 bytes.
- `PRODUCER_IU4_BLOCK128`: `block_i4_128[(K/128) * N]` with `[kblock * N + token]` addressing; optional F32 output only when `EMIT_F32=true`.
- `PRODUCER_MQ4V2_FP8`: byte-identical to `prepare_mq4v2_fp8_x_f32`: E4M3 codes `[N,K]`, F32 half sums `[N,K/256,2]`, F32 row scales `[N]`. This is the exact drop-in for existing FP8-v2 GEMMs.
- `PRODUCER_FP8_STREAM_F16`: OCP E4M3 codes `[N,K]` plus a separate `_Float16 scales[N]` plane for consumers that need no MQ4 zero-point correction.
- `PRODUCER_MQ4V2_FP8_F16_SCALE`: the same codes/F16 scale plus F32 decoded-code half sums `[N,K/256,2]`, emitted while the row is resident. This is the only lossy F16-scale handle accepted by MQ4v2 GEMMs.

Both F16-scale sinks use the statement-identical scale/encode recipe at `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/kv_cache_write_q8_0.hip:121-156`: zero row → 1.0; otherwise cast `amax/448` to F16, floor underflow at F16 bits `0x0001`, bump one positive F16 ULP if `448 * float(scale) < amax`, and divide by the stored scale before `__builtin_amdgcn_cvt_pk_fp8_f32`. A consumer converts the stored F16 scale to F32 before multiplication. These routes are not bit-exact and remain default-OFF.

Do not change `Mq4v2Fp8Prepared` to carry an F16 scale. Add distinct private-field handles `Fp8StreamPrepared { codes, scales_f16, n, k, generation }` and `Mq4v2Fp8F16Prepared { codes, half_sums_f32, scales_f16, n, k, generation }`; otherwise a pointer with the wrong scale type can silently reach an existing GEMM. `Int4MmqPrepared` and `Mq4v2Fp8Prepared` remain the only arguments accepted by their existing prepared consumers. Reservation generation, `N`, `K`, and pointer identity are checked before launch.

#### Whole-K FP8 scale

FP8-v2 and the F16-scale stream differ from IU4: one scale covers the whole K row. Do not use atomics or cross-workgroup partials to recover that maximum.

- `scale_mode=0`: scale is 1, so a 256-group producer can encode and accumulate its two half sums directly from registers.
- `scale_mode=1`, K=5120: one 256-thread row workgroup may stage the 20,480-B row in LDS, reduce the whole-row finite absmax, publish the incumbent F32 scale, then encode. Compare this with deterministic producer recomputation and keep the faster exact variant.
- `scale_mode=1`, K=17408: a 69,632-B row does not fit in 64-KiB LDS. Use two deterministic producer passes. Pass A reproduces every rotated F32 value and reduces the finite whole-row absmax. Pass B reproduces the same expressions, stages at most one 8,704-F32 half (34,816 B), encodes against the published scale, and emits decoded-code half sums. No F32 value reaches global memory between producer and pack.
- The F16-scale stream uses the same topology with its distinct scale recipe.

Recomputation is bit-exact only when the producer is pure and its expression order is unchanged. Non-finite real activations reject the candidate. Silu recomputation repeats `expf`, so transfer loss is likely; abandon FP8 producer fusion for that site if fused `TIME` exceeds producer plus standalone FP8 prepare. Never keep a slower fusion merely because it removed a launch.

### 3.3 Exported symbol and flag contract

The exact-gfx1201 entries are:

- existing RMS symbols `fused_rmsnorm_mq_rotate_i4` and `fused_rmsnorm_mq_rotate_awq_i4`
- P0 silu symbols `fused_silu_mul_mq_rotate_i4_gfx12` and `fused_silu_mul_mq_rotate_awq_i4_gfx12`
- `gated_norm_mq_rotate_awq_i4_gfx12` for the exact-gfx1201 GDN post route
- `sigmoid_mul_mq_rotate_awq_i4_gfx12` for the measured 64-call full-attention chain, only after its gate passes
- matching suffixes `_mq4v2_fp8_gfx12` for the byte-identical FP8-v2 sink
- matching suffixes `_mq4v2_fp8_f16_gfx12` for the lossy MQ4v2 stream and `_fp8_stream_f16_gfx12` for non-MQ consumers, only at sites admitted by the lossy-stream gate

Default-OFF feature/kill switches:

- `HIPFIRE_GFX12_RMS_QUANT_FUSED`
- `HIPFIRE_GFX12_SILU_QUANT_FUSED`
- `HIPFIRE_GFX12_GDN_QUANT_FUSED`
- `HIPFIRE_GFX12_ROTATE_QUANT_FUSED`
- `HIPFIRE_GFX12_MQ4V2_FP8_PRODUCER_FUSED`
- `HIPFIRE_GFX12_FP8_STREAM`

`HIPFIRE_GFX12_SILU_QUANT_FUSED` is already the P0 spelling and maps to `kernel.gfx12_silu_quant_fused` (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/feature_flags.rs:236-242`; `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-config/src/lib.rs:2142-2149`). Reuse it; do not introduce a second silu flag.

A value other than `1` is off. Each flag additionally requires exact `gfx1201`, `HIPFIRE_IU4_PREFILL=1` for IU4 or the current FP8-v2 eligibility for FP8, the dense prefill route, supported K, and a typed reservation. Other arches and MoE continue through the incumbent producer and standalone quantizer. Add centralized config/schema mappings and `/home/kaden/ClaudeCode/warpfront/wt-glue/docs/env-vars.md` entries in the same host-integration slice; do not read these environment variables in `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`.

### 3.4 Exact arithmetic constraints

- RMSNorm: retain the exact thread-strided accumulation, shuffle/shared reduction, epsilon placement, `rsqrtf`, AWQ divide, sign multiplies, and FWHT butterfly sequence from `fused_rmsnorm_mq_rotate*`. An LDS store/load is bit preserving.
- silu·up: retain `(gate / (1.0f + expf(-gate))) * up`, then the incumbent AWQ/sign/FWHT sequence. Do not rewrite to `gate * sigmoid(gate)`, reciprocal multiply, or fused expressions.
- GDN: retain the per-head square-sum ownership and shuffle tree from `gated_norm_f32`, then the incumbent rotate sequence. Complete head pairs/256 groups stay wave-owned; no atomic or cross-workgroup reduction.
- IU4: call the single `quantize_block_i4_128_wave` helper. Preserve strict candidate tie handling, `rintf`, clamp, integer sum, nibble order, and `[kblock*N+token]` address.
- FP8-v2: preserve the current F32 producer value, caller-selected `scale_mode` (unit scale or the exact power-of-two-to-224 formula at `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip:20-22,78-86`), conversion ownership, decoded-code half-sum tree, and F32 scale. Direct F32 producer fusion may compare bitwise against the `_f32` pack path. It must not compare against the F16 pack path and call a last-ULP difference exact.
- Graph/Redline: no atomics, clocks, counters, or address-dependent decisions. Grow scratch, compile entries, and resolve all pointer generations before capture. A captured graph must not allocate, compile, or switch output format during replay. Redline tape integration is not part of this change; its route keeps the incumbent kernels until separately admitted.

## 4. Residual-add treatment

The requested `residual add + RMSNorm + rotate + quant` kernel is valid only where a separate residual-add pass exists. It is not present on the supplied single-GPU IU4 route because the residual GEMM writes `Y += W·X` directly.

For a measured Q8/low-bit or post-all-reduce route, the future row kernel contract is:

1. `sum = residual[i] + delta[i]` in the same operand order as `add_inplace_f32`.
2. Store that exact F32 sum back to authoritative `x_batch`.
3. Accumulate RMS from the rounded stored sum with the incumbent RMS reduction order.
4. Re-read `x_batch`, normalize/rotate, and emit IU4 or FP8 through the same compile-time sink.

The re-read is intentional: it preserves the materialized F32 residual contract. Never replace `x_batch` with FP8. First profile that route; abandon this subproject if `add_inplace_f32` plus the following producer/quant chain is below 2% of step time or the fused kernel saves below 1% of step time in three fresh processes.

## 5. FP8 stream screen: f32/q8 → E4M3 + per-token F16 scale

The exact FP8-v2 producer sink removes the standalone pack without adding quantization error. `PRODUCER_FP8_STREAM_F16` and `PRODUCER_MQ4V2_FP8_F16_SCALE` are separate, lossy levers intended for direct-reading consumers. They are default-OFF and KLD-gated at every site.

Raw F32→E4M3/F16 stream traffic for one row is `4K → K+2` resident bytes, saving `3K-2`; producer-store plus consumer-read saves `6K-4`. MQ4v2 additionally requires two F32 half sums per K/256 group, or `K/32` bytes per row. The net MQ4v2 stream savings are therefore `4K-(K+2+K/32)` resident bytes and `8K-2(K+2+K/32)` round-trip bytes:

| K | Raw stream resident / round-trip saving | MQ4v2 stream resident saving | MQ4v2 producer→consumer round-trip saving |
|---:|---:|---:|---:|
| 5120 | 15,358 B / 30,716 B | 15,198 B | **30,396 B** |
| 17408 | 52,222 B / 104,444 B | 51,678 B | **103,356 B** |
| 4096 example | 12,286 B / 24,572 B | 12,158 B | **24,316 B** |

These byte counts are ceilings. They are not time savings unless the trace shows memory stalls and the consumer can decode directly without another materialization.

| Site and source buffer | Direct consumer | Risk | Gate/decision |
|---|---|---|---|
| Post-RMSNorm/rotate `x_rot_batch`, K=5120, LA QKVZA and FFN gate/up (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5021-5169,6149-6274`) | Existing MQ4v2 FP8 QKVZA/gate-up ABI first; an F16-scale consumer variant only after exact fusion wins | Medium. It perturbs every projection input, but current FP8-v2 already quantizes the same row. | First prove `_mq4v2_fp8` byte-equal to standalone `_f32` pack. Screen F16 scale on real activations; require site KLD increment ≤0.0005 and a measured total win. |
| Post-silu/rotate `ffn_hidden_batch`, K=17408 (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:6436-6536`) | FP8-v2 residual/down GEMM | Medium-high; largest byte win, but the rounded result enters the residual stream every layer. | Highest-priority F16-scale experiment after exact FP8-v2 fusion. Abandon if it saves <2% of pp8192 wall or exceeds the KLD cap. |
| Post-GDN `dn_normed_rot_batch`, K=`v_dim` (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:5718-5849`) | FP8-v2 residual `wo` | High; the projection result rejoins recurrent residual state. | Require per-layer real-activation oracle plus KLD; use the MQ4v2 formula `5.9375K-4` rather than assuming K=4096. |
| Post-full-attention `fa_attn_out_rot_batch`, K=`q_dim` (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs:7155-7240`) | FP8-v2 residual `wo` | High; attention output feeds residual. | Identify the 64 calls in the trace first. No implementation if the complete chain is <1% of wall. |
| GDN-pre `dn_q_raw_batch`, `dn_k_raw_batch`, `dn_v_batch`, `dn_q_batch`, `dn_k_batch` (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/batch.rs:54-58`; `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gdn_pre_batched.gfx1201.hip:58-84`) | GDN recurrence/attention consumers, currently F32/q8 paths | Very high. Quantization enters a recurrent state transition, unlike a one-shot GEMM activation. | Probe-only slice. Capture consumer-read bytes and hidden-state error; abandon unless eligible traffic is ≥2% of step time and 24-chunk KLD passes. Do not infer a win from the 15.521-us GDN-pre bucket. |
| `x_norm_batch` on mixed Q8/MoE routes (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/batch.rs:43-47`) | Q8 router/shared-expert consumers | Very high; small changes may change top-k expert routing categorically. | Not in initial implementation. Require top-k agreement in addition to KLD and a separate MoE profile. |
| Authoritative `x_batch` | All later residual additions | Unacceptable for this project | Never replace with FP8. A producer may emit an FP8 sidecar while retaining exact F32 `x_batch`. |

The F16-scale consumer must read codes and the F16 scale directly. A decode-to-F32 staging kernel defeats the stream. MQ4v2 producer sinks must emit the required decoded-code half sums while the row is resident; never launch the incumbent pack merely to obtain metadata.

### FP8 abandon criteria

Per site, abandon `PRODUCER_FP8_STREAM_F16` if any is true:

1. The eligible producer-store/consumer-read traffic is less than 2% of measured step time.
2. The direct consumer needs a new global F32 materialization.
3. Three fresh-process pp8192 runs improve by less than 2% at that site/route or regress any pp512/2048/32768 row beyond 2%.
4. 24-chunk KLD exceeds incumbent by 0.0005, top-1/decoded output becomes suspicious, or a recurrent/router site changes its additional state/top-k oracle.
5. Graph capture/replay differs from eager.

The competitor's 5% decode result authorizes only a decode profile. Before adding a decode consumer, trace gfx1201 decode and abandon if eligible F32/q8 producer↔consumer streams are below 2% of decode step. No prefill claim transfers from that number.

## 6. Independently executable composer slices

Only the named owner edits a shared file. Kernel slices each own exactly one `.hip` file. They do not edit Rust routing, run formatters/linters, or run project-wide tests. The integration owner consumes the frozen symbols above after kernel slices land.

### S0 — Oracle and baseline ledger (no production changes)

**Owns:** a throwaway lab harness removed before commit; evidence only under `/home/kaden/ClaudeCode/warpfront/wt-glue/scratch-2026-09-17/producer-fusion-oracle/`.

**Work:** capture real early/middle/late-layer producer inputs at K=5120 and K=17408; run incumbent producer then standalone IU4/FP8 pack; save full producer bytes, full 72-byte IU4 records, FP8 codes/metadata, and downstream GEMM output. Include `N=1`, a non-tile multiple, and the production `N=4096`. Record model, prompt bytes/md5, binary md5, commit, ROCm, arch, flags, and exact command.

**Accept:** rerunning the incumbent chain reproduces identical bytes; decoded text is non-degenerate. This baseline must exist before candidate timing.

### S1 — Shared IU4 recipe only

**Owns:** `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/block_i4_128_quant.hip`.

**Work:** retain one device-only group emitter around `quantize_block_i4_128_wave`; do not add a second candidate search, packing recipe, or output layout. Keep `quantize_int4_mmq_ds128` exported for FP8, non-gfx1201, MoE, and fallback routes.

**Accept:** existing standalone IU4 oracle bytes remain identical; header size is 72 bytes; no producer-specific code is added here.

### S2 — K=17408 silu kill probe and final producer

**S2-I owns:** `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/fused_silu_mul_mq_rotate_awq.hip`.

**S2-I work:** keep grid `[K/256,N]`, block 32, and the incumbent expression/FWHT order. Instantiate `fused_silu_mul_mq_rotate_awq_i4_gfx12`; pass the final eight floats/lane directly to the shared register-sidecar helper. Do not add LDS or write the F32 row when the prepared IU4 GEMM is its sole consumer.

**S2-I accept:** Gate P0 passes; all IU4 blocks and optional F32 bytes match at K=17408; final metadata is ≤55 VGPR/18 SGPR, 0 LDS, and zero spill/scratch; three-run fused `TIME` is ≤ producer plus its 256 eliminated quant calls. Otherwise stop copying this topology.

**S2-P owns sequentially:** `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/fused_silu_mul_mq_rotate.hip`. Instantiate the plain twin under its separate 84-VGPR/18-SGPR budget; never edit both producer files concurrently.

**S2-F owns the AWQ file only after S2-I passes:** add the exact FP8-v2 sink using the K=17408 two-pass whole-scale algorithm in §3.2. It is rejected if codes/half sums/F32 scale differ from standalone `prepare_mq4v2_fp8_x_f32` or if fused `TIME` exceeds producer plus prepare.

### S3 — RMSNorm producer

**Owns:** `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/fused_rmsnorm_mq_rotate.hip`.

**IU4 work:** preserve the exact 256-thread row reduction. Each wave emits the 256 group it already holds through the register-sidecar helper; no row LDS is added. Support `EMIT_F32=true` for QKVZA tails.

**FP8-v2 work, sequential after IU4:** for K=5120, stage at most the 20,480-B final row in LDS, reduce the whole-K scale, and emit the incumbent FP8-v2 ABI. Keep unit-scale mode group-local.

**Accept:** F32 and IU4 byte identity across real layers/N boundaries; exact FP8-v2 code/half-sum/scale identity; IU4 ≤64 VGPR/32 SGPR with only incumbent 1,024-B reduction LDS; FP8 total LDS ≤21,504 B; zero spill. Each fused entry's `TIME` is ≤ the exact chain it replaces.

### S4 — GDN post-gated-norm producer

**Owns:** new gfx12-named, exact-gfx1201-routed `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gated_norm_mq_rotate_quant.gfx12.hip`.

**IU4 work:** assign complete head pairs/256-element groups to waves, preserve the gated-normalization shuffle tree and rotate order, then emit two IU4 blocks directly from registers. Do not modify the gfx1100 kernel or `gdn_pre_batched_gfx1201`.

**FP8-v2 work, sequential after IU4:** use a row workgroup only for the whole-K scale/metadata pass; no global F32 staging.

**Accept:** gated-normalized F32, rotated F32, IU4 bytes, and exact FP8-v2 metadata match the incumbent chain; no atomics, zero spill; every fused entry's `TIME` is ≤ the chain it replaces.

### S5 — Full-attention producer/rotate probe

**Owns:** new `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/sigmoid_mul_mq_rotate_quant.gfx12.hip`, exact-gfx1201-routed.

**Work:** first map the 64 trace calls to `batch_chunk_full_attn_output_projection`. If confirmed, fuse the current sigmoid-multiply producer with AWQ/sign/FWHT. Emit IU4 directly from group registers; add whole-row FP8-v2 only as a later sequential sub-slice. Do not absorb `deinterleave_q_rmsnorm` unless a separate profile proves the dataflow and benefit.

**Accept:** producer/rotated/IU4 bytes and exact FP8-v2 metadata match; zero spill; each fused entry's `TIME` is ≤ the measured chain it replaces. Otherwise record rejection and retain standalone `rotate_x_mq_awq`.

### S6 — Single Rust integration owner

**Owns:**

- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/kernels.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/gemv.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/scratch.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/gemm.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/dispatch.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/rdna-compute/src/feature_flags.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-config/src/lib.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/prefill.rs`
- `/home/kaden/ClaudeCode/warpfront/wt-glue/crates/hipfire-arch-qwen35/src/qwen35/batch.rs` only if a new sidecar allocation is unavoidable
- `/home/kaden/ClaudeCode/warpfront/wt-glue/docs/env-vars.md`

**Work:** register distinct modules/symbols, implement typed reservations/prepared handles, route only exact-gfx1201 eligible dense prefill calls, and pass prepared handles directly to prequant GEMMs. The P0 silu host entry is `Gpu::fused_silu_mul_rotate_mq_i4_gfx12_batched`; `Gpu::iu4_silu_quant_fused_active` must require IU4, gfx1201, `N>=128`, `N%128==0`, and `K%256==0`. Standalone `ensure_int4_mmq_x` and `prepare_mq4v2_fp8_x*` remain fallbacks. Pre-grow/compile before graph capture; captured producer and consumer pointers must be stable. Never infer readiness from a non-null scratch pointer.

**Accept:** with each flag off, call sequence and outputs are incumbent; with one flag on, the expected fused symbol replaces exactly its producer+quant chain, and no eligible consumer calls standalone quant again. Other gfx targets, MoE, unsupported K, partial/all-reduce, and Redline remain on fallback. The initial eager-only gate may be lifted only after the graph oracle passes; final graph capture/replay uses the fused route without allocation/JIT during capture.

### S7 — Lossy F16-scale stream probe

**S7a producer ownership:** one producer `.hip` file per sequential sub-slice, starting with `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/fused_silu_mul_mq_rotate_awq.hip`. Add `PRODUCER_MQ4V2_FP8_F16_SCALE` using the frozen KV-FP8 scale recipe and emit half sums while the row is resident.

**S7b consumer ownership:** `/home/kaden/ClaudeCode/warpfront/wt-glue/kernels/src/gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip` only. Add distinct F16-scale qkvza/gate-up/residual entries that accept codes, F32 half sums, and F16 scales directly. Do not alter the F32-scale entries. S7a and S7b may not run concurrently with any earlier slice owning the same file.

**S7c host ownership:** the S6 integration owner alone edits Rust routing after S7a/S7b freeze their symbols. Do not begin any S7 slice until exact `_mq4v2_fp8` fusion passes.

**Order:** FFN-down K=17408, RMS projection K=5120, GDN post, full-attention post. GDN-pre, MoE/router, and decode are probe-only independent follow-ons.

**Accept:** per-site oracle and KLD/top-k/state gates pass, consumer reads codes/F16 scale/half sums directly, no F32 restaging occurs, and the site-specific perf threshold is met. A failed site does not block exact IU4/FP8-v2 fusion.

### S8 — Separate residual-add probe

**Owns:** no production file until the Q8/low-bit or post-all-reduce trace passes the 2% gate; then one new gfx1201 kernel file. S6 owns later Rust routing.

**Work:** measure the route with a real workload, then implement the exact materialized-F32 residual algorithm in §4 only if admitted.

**Accept:** `x_batch` bytes remain exact, quant sidecar meets its format gate, and measured saving exceeds 1% of step time. No change to the single-GPU IU4 residual GEMM path.

## 7. Ordered build and integration sequence

1. Confirm `/home/kaden/ClaudeCode/warpfront/wt-glue` is branch `gfx1201-producer-fusion` at base `d21c95096`; do not clean or overwrite concurrent user changes.
2. S0 records the unfused real-activation oracle and current fresh-process trace.
3. S1 freezes the single IU4 helper contract.
4. S2 runs alone as Gate P0. If rejected, stop copying the producer-local sidecar topology and retain only its evidence.
5. After S2 passes, S3, S4, and the trace-identification part of S5 are independent and may run in parallel; each owns one kernel file and performs no shared validation.
6. S6 integrates the already-frozen symbols and prepared-handle ABI. This is the only production-Rust integration writer.
7. Build the profiler once after all exact-fusion slices land:

   ```sh
   cd /home/kaden/ClaudeCode/warpfront/wt-glue
   cargo build --release -p saddle-lab --example profile_prefill_qwen35
   ```

8. Run scoped metadata, oracle, graph, and profiler gates in §8. Do not run formatters, linters, or a project-wide test suite in worker slices.
9. Only after exact IU4 and exact FP8-v2 fusion pass, run S7 site by site. Never batch-admit lossy sites.
10. The caller runs the single shared validation/doc check after all slices land.

## 8. Measurement and correctness gates

### 8.1 Evidence identity

Every run writes only below `/home/kaden/ClaudeCode/warpfront/wt-glue/scratch-2026-09-17/<unit>/`; no evidence goes to `/tmp`. Record:

- `git rev-parse HEAD`, dirty diff hash, model path/hash, GPU/arch, ROCm/hipcc version;
- prompt bytes, byte count, and md5; compared prompts must be byte-identical;
- executable md5; rebuild or flag changes require a new identity row;
- complete environment and command;
- raw stdout/stderr, rocprof CSV, resource report, JSON result, and decoded text.

Use at least three **fresh processes** per A and B. Do not use repeated loops in one process as the three runs; DPM/thermal drift is 10–15%. Interleave A/B process order when possible. Eyeball every decoded output; an unusually tight spread or high throughput with a single-token attractor is a failure, not a win.

### 8.2 Metadata and byte oracle

Compile the actual JIT source/defines with:

```sh
HIPFIRE_HIPCC_EXTRA_FLAGS=-Rpass-analysis=kernel-resource-usage
```

Archive the generated source, object hash, complete compiler flags, VGPR, SGPR, LDS, and scratch/spill report. Admission requires zero spill and the §3.1 bounds.

For IU4, compare every byte of optional F32 producer output and every 72-byte `block_i4_128`, including `d`, `s`, and all codes, against incumbent producer→`quantize_int4_mmq_ds128`. Then compare downstream GEMM output bytes. For FP8-v2 exact fusion, compare E4M3 codes, both F32 half sums, F32 row scale, and GEMM output. Use real activations at early/middle/late layers and both K=5120/K=17408.

A minimal graph oracle captures candidate producer + prepared consumer, replays twice, copies all outputs to host, and compares them with eager execution and with each other. It also records the captured node symbols. Benchmark tok/s is not graph correctness evidence.

### 8.3 Profile command and attribution

Before any ordinal-1 command, announce ownership on hub. Use:

```sh
cd /home/kaden/ClaudeCode/warpfront/wt-glue
export HOME=/home/kaden/.hipfire-homes/ab1
export ROCR_VISIBLE_DEVICES=1
export HIPFIRE_MODELS_DIR=/home/kaden/.hipfire/models
export HIPFIRE_KERNEL_CACHE=/home/kaden/.hipfire-homes/ab1/.hipfire_kernels
export HIPFIRE_GRAPH=0
export HIPFIRE_LLOYD_GFX12=1
export HIPFIRE_GFX12_MQ4V2_FP8_V2=1
export HIPFIRE_GFX12_MQ4V2_FP8_V2_GEOM=128x128
export HIPFIRE_GFX12_MQ4V2_FP8_SLABS=2
export HIPFIRE_GFX12_GDN_PRE_FUSED=1
export HIPFIRE_GFX12_FA2_PREFILL=1
export HIPFIRE_IU4_PREFILL=1
export HIPFIRE_PREFILL_MAX_BATCH=4096
E=/home/kaden/ClaudeCode/warpfront/wt-glue/scratch-2026-09-17/profile-c4096
mkdir -p "$E"
cd "$E"
rocprofv3 --kernel-trace --stats -f csv -- \
  /home/kaden/ClaudeCode/warpfront/wt-glue/target/release/examples/profile_prefill_qwen35 \
  /home/kaden/qcal/ladder-v2/artifacts/qwen3.8-27b.mq4v2.xt.ctrl.hfq \
  --prefill 8192 --warmup 1 --kv-mode q8
```

Before each fresh process, materialize and hash the profiler's exact synthetic token stream, then hash the executable:

```sh
python3 -c 'import struct; open("prompt-u32le.bin","wb").write(b"".join(struct.pack("<I", i) for i in range(8192)))'
md5sum prompt-u32le.bin /home/kaden/ClaudeCode/warpfront/wt-glue/target/release/examples/profile_prefill_qwen35
```

The example takes the model path first (`/home/kaden/ClaudeCode/warpfront/wt-glue/crates/saddle-lab/examples/profile_prefill_qwen35.rs:23-25`) and uses deterministic token IDs 0..8191 (`:123-125`). Compare two-pass totals as two passes. For each site, the candidate trace must contain its fused symbol, reduce the expected standalone quant call count, and contain no hidden conversion/restaging kernel. Compare sum-of-chain time and full profiled-forward wall; a call-count reduction alone is not acceptance.

### 8.4 Model exactness and KLD

With `HIPFIRE_IU4_PREFILL=1`, the supplied route anchors are:

- 1 chunk md5: `032ebad84f2c1dd5e2980fa805215ac0`
- 2 chunks md5: `dc7e53181662271780374f0a85fd7732`
- 24 chunks incumbent KLD: `0.063410`

An exact IU4 fusion must retain both md5s. If a fusion necessarily changes a reduction order, it must stay default-OFF and its 24-chunk KLD must be ≤`0.063910` (incumbent +0.0005), using identical prompt/reference/model/binary identities. `PRODUCER_FP8_STREAM_F16` is always in this KLD-gated class. Recurrent and router sites additionally require state/top-k agreement as specified in §5.

Use the claim-scoped Qwen evaluator named by `/home/kaden/ClaudeCode/warpfront/wt-glue/docs/VALIDATION.md:245-266`; archive its per-sequence output. The retired `coherence-gate-*.sh` family is historical reproduction only (`/home/kaden/ClaudeCode/warpfront/wt-glue/docs/VALIDATION.md:273-282`) and is never promotion or acceptance evidence.

### 8.5 Product benchmark and graph route

Run the requested matrix in three fresh processes per arm with the same flags as §8.3 plus `HIPFIRE_GFX12_FA2_FP8=1` and `HIPFIRE_GRAPH=1`:

```sh
hipfire bench qwen3.8:27b-mq4-xt --matrix \
  --pp 512,2048,8192,32768 --ctx 128 --tg 64 --spec off \
  --runs 3 --warmups 1 --kv-mode fp8 --json
```

Pair rows by the trailing decode column/state (about 29 versus 36.5), never by row position alone. Preserve the supplied 4096-chunk reference rows `2204/2236/2060/1524` as historical context, not as a pass threshold. Report medians and all raw rows. Exact fusion must not regress any paired pp row by more than 2%, must improve pp8192 outside run-to-run noise, and must not regress trailing decode by more than 2%. A lossy FP8 stream must meet its stricter site gate in §5.

## 9. Completion contract

The change is ready for reviewer gating only when:

1. Every enabled IU4 producer writes bit-identical 72-byte blocks and exact expected F32 outputs on real activations.
2. Exact FP8-v2 producer variants match standalone `_f32` pack bytes and downstream GEMM bytes.
3. Every admitted kernel has zero spill, stays within the stated LDS/VGPR/SGPR budgets, and has three fresh-process profile comparisons.
4. The standalone quantizers remain available and are used by FP8/non-gfx1201/unsupported/MoE/partial/Redline fallbacks.
5. Graph capture/replay is independently byte-verified; no allocation or JIT occurs during capture.
6. Prompt md5, binary md5, raw profiler/benchmark output, and decoded text are archived under the required untracked evidence root.
7. IU4 pins and KLD gates pass; lossy F16-scale sites pass independently and remain default-OFF.
8. Environment/config documentation and the claim-scoped `/home/kaden/ClaudeCode/warpfront/wt-glue/docs/VALIDATION.md` checks pass. No retired coherence script is cited.

## 10. Explicit non-goals

- No rewrite of the IU4 GEMM math or 72-byte block format.
- No deletion of `quantize_int4_mmq_ds128` or FP8 pack kernels.
- No gfx906/gfx10xx/gfx11xx/gfx94x behavior change; new dispatch is exact-gfx1201 only.
- No TypeScript/JavaScript control plane; all routing/config remains Rust.
- No compression of authoritative F32 residual state.
- No GDN-pre, MoE/router, decode, Redline-tape, or separate residual-add implementation without its own measurement gate.
- No claim that the competitor's decode result predicts hipfire prefill.
- No formatter, linter, or project-wide test run inside a worker slice; the caller runs shared validation once.
