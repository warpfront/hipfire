# Fp8Stream status (ordinal 2, worktree wt-fp8stream, branch gfx1201-fp8stream)

## Committed (e6f3e4ba0): Lloyd-path producer→FP8 fusion, default OFF
Kernel `fused_rmsnorm_mq_rotate(+AWQ)` gains `HIPFIRE_FP8_STREAM` variants
(`fused_rmsnorm_mq_rotate{,_awq}_mq4v2_fp8_gfx12`); host
`fused_rmsnorm_rotate_mq_fp8_gfx12_batched` seals `Mq4v2Fp8Prepared`;
qkvza/gate_up/FA-qkv/FA-gate_up prefill sites consume the existing
`*_prepared_lloyd` GEMMs. Flag `HIPFIRE_GFX12_FP8_STREAM` /
`kernel.gfx12_fp8_stream`, default OFF.

## Green gates
- Device oracle `target/release/examples/fp8stream_producer_oracle`
  (source: `crates/hipfire-runtime/examples/fp8stream_producer_oracle.rs`,
  NOT committed; manifest entry in `crates/hipfire-runtime/Cargo.toml`
  unstaged): 24/24 cases byte-identical, 0 mismatches
  (`scratch-2026-09-17/fp8stream/oracle-fp8stream.txt`).
- Metadata (`metadata-fp8stream.txt` + `tu-fp8-*.hip`): plain 45/33,
  AWQ 89/38 VGPR/SGPR, 0 spill/scratch, 16 waves/SIMD (matches admitted
  IU4 twins at full occupancy).
- KLD WT2 ON c1/c2/c24 match pins exactly — BUT trivially (see below).

## Red finding (2026-09-19): fusion never fires on qwen3.8-27b.mq4-xt
ON profile (`profile-on/stdout-on.txt`): 512 `pack_f32_to_fp8` remain,
unfused `fused_rmsnorm_mq_rotate_awq_batched` x256, zero fused rows.
Gate trace proved why: `dtype=MQ4G256V2` on all 128 calls — the model is
UNIFORM MQ4v2, while the hooks gate on `MQ4G256V2Lloyd` (dormant qt52
family). Flag/env parsing verified working (`flag=true`).

## What remains (uniform-path enablement)
1. Widen `try_gfx12_fp8_stream_rmsnorm_prepared` dtype gate to accept
   `DType::MQ4G256V2` (keep Lloyd).
2. Uniform prepared consumers: gate_up has one —
   `gemm_gate_up_hfq4g256_wmma_gfx12_mq4v2_fp8_bt12_prepared`
   (gemm.rs:30020, selects staged-v2 internally). qkvza/qkv uniform
   wrappers (gemm.rs:9383/9582) prepare-then-launch inline: split out
   `_prepared` twins mirroring the `_lloyd` pattern.
3. Hook the 4 prefill sites' UNIFORM arms (currently via
   `run_fused_*_key` dispatch at prefill.rs:5676/6775/7336/8319 or
   direct uniform wrappers) to call the fused producer + prepared GEMM.
   The F32 `x_rot` store is kept, so dispatch bypass only affects the
   pack input.
4. Re-verify: oracle (unchanged kernel — still valid), ON profile
   (expect ~256 packs gone at RMS sites), KLD pins (now non-trivial),
   interleaved bench OFF/ON/OFF/ON, decode-only ≥36.4, serve battery.
5. Decide silu (K=17408, needs row-workgroup rewrite) / GDN-post
   (K=2048, needs grid change) per-site admission; flip default ON only
   after all gates.

## Baselines (this dir)
- `profile-base/`: flag-free pp8192 — 512 packs, 181148 us total (22.1
  us/token, 3.8%). RMS-site share ~1/3 (~7.6 us/token).
- `profile-on/`: flag ON — identical (fusion dormant).
- `kld-on/`: c1 9fd99341 / c2 1cadeca9 / c24 0.048028 (trivially exact).
- `gate-debug/`: gate-trace runs proving dtype=MQ4G256V2.
