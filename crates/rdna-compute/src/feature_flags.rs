// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed, immutable startup-policy resolution for rdna-compute.
//!
//! The native daemon installs a schema-validated process configuration before
//! `Gpu::init()`. Dispatching hot paths access `self.flags.*` instead of
//! parsing TOML or consulting ambient process state on every call.

use hipfire_config::ProcessConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mb4Mode {
    Pack1,
    Pack2,
    Pack4,
}

#[derive(Debug, Clone)]
pub struct FeatureFlags {
    // ── Arch identity ──────────────────────────────────────────────
    pub arch: String,

    // ── GEMV tuning ────────────────────────────────────────────────
    pub gemv_rows: Option<u32>,
    pub gemv_dp4a_default_on: bool,
    pub gemv_prefetch: Option<bool>,
    pub gemv_prefetch_default_on: bool,
    pub gfx942_lds_gemv: Option<bool>,
    pub gfx942_lds_gemv_default_on: bool,
    pub gemv_rows_default: u32,
    pub gemv_dp4a: Option<bool>,
    /// Override the gfx1151 temporal raw-buffer E8 U4 route. `None` selects
    /// the accepted DeepSeek4 MQ2R route default while retaining portable
    /// behavior for other models and architectures.
    pub gfx1151_e8_buffer: Option<bool>,
    /// gfx11 MMQ per-128 (X128) activation path (`HIPFIRE_GFX11_MMQ_X128`).
    /// Default on gfx1100/gfx1151 (WT2 KLD +0.000186 <= +0.0005 gate passed);
    /// `=0` restores the per-32 MMQ route. Other arches stay per-32.
    pub gfx11_mmq_x128: Option<bool>,
    /// iu4-direct MMQ prefill (W4A4, `HIPFIRE_IU4_PREFILL`,
    /// `kernel.iu4_prefill`). Default ON for gfx1100/gfx1151 (K16 iu4 kernel)
    /// and gfx1201 (K32 iu4 kernel, standalone `quantize_int4_mmq_ds128`
    /// activations); `=0`/`false` keeps the incumbent Q8_1 MMQ (gfx11) /
    /// fp8 (gfx1201) route. Other arches have no iu4 route and ignore it.
    /// Quality cost ~+0.016 WT2-24 KLD on the MQ4-XT speed rung
    /// (gfx1201: 0.048028 -> 0.063890).
    pub iu4_prefill: Option<bool>,
    /// Exact gfx1201 symmetric A4 prefill: builder-emitted `_b1` GEMM
    /// (`HIPFIRE_G12_IU4_ISA`). Default on after byte-exact G3 and +7.04%
    /// pp8192 G4; `=0` restores hipcc K1 when that route is selected.
    pub g12_iu4_isa: bool,
    /// Oracle-accepted decode twins (`kernel.pm_decode`, `HIPFIRE_PM_DECODE`):
    /// exact gfx1201 loads the embedded builder images of
    /// [`crate::pm_decode_twins`] in place of those HIP modules. `auto`
    /// (default) is on for exact gfx1201 only; `=0` restores the hipcc
    /// modules. Other arches never load the twins.
    pub pm_decode: bool,
    /// Exact gfx1201 MQ4v2 A8 prefill (int8 per K128); opt-in, default off.
    pub a8_prefill: bool,
    /// Exact gfx1100/gfx1151 MQ4v2 A8 prefill (`kernel.gfx11_a8_prefill`,
    /// `HIPFIRE_GFX11_A8_PREFILL`); opt-in, default off. Takes precedence over
    /// the IU4 W4A4 projections wherever it is admitted.
    pub gfx11_a8_prefill: bool,
    /// Fuse A8 sidecar emission into the four activation producers (default on).
    pub a8_fused_prod: bool,
    /// Qwen4 chunked GDN prefill (`HIPFIRE_QWEN4_GDN_CONV_QKNORM`): the
    /// convolution also writes the normalized Q/K, dropping the separate
    /// Q/K-norm launch. Bytes identical to the two-launch pair. Exact gfx1151
    /// only, where it defaults on; `0` is the kill switch for the new kernel.
    pub qwen4_gdn_conv_qknorm: bool,
    /// Qwen4 chunked GDN prefill on a Q8 GDN state
    /// (`HIPFIRE_QWEN4_GDN_Q8_INLINE`): the recurrence kernel decodes and
    /// requantizes the state itself instead of the two conversion launches.
    /// Bytes identical. Opt-in, default off (the kill switch for the new
    /// module); exact gfx1151 only.
    pub qwen4_gdn_q8_inline: bool,
    /// Qwen4 gfx1201 FN prefill (`HIPFIRE_QWEN4_GDN_I8_FUSE`): the dense GDN gate
    /// also rotates and quantizes the MQ6 out projection's A8 activation
    /// (`fn_gdn_dense_gate_i8`), so the F32 gate output and the separate
    /// `mq_rotate_x_i8_gfx12` launch disappear. Bytes identical to the two-launch
    /// pair. Exact gfx1201 only, where it defaults on; `0` is the kill switch.
    pub qwen4_gdn_i8_fuse: bool,
    /// Qwen4 gfx1201 FN prefill (`HIPFIRE_QWEN4_GDN_CHAIN_FUSE`): the dense-GDN chain
    /// runs the load-batched `fn_gdn_dense_producer_v2_{f32,bf16}` entries (and, with
    /// `HIPFIRE_QWEN4_GDN_QKV_BF16`, reads a BF16 qkv projection). Bytes identical to the
    /// v1 producer on the F32 projection. Exact gfx1201 only, where it defaults on; `0` is
    /// the kill switch restoring today's chain exactly.
    pub qwen4_gdn_chain_fuse: bool,
    /// Qwen4 gfx1201 FN prefill (`HIPFIRE_QWEN4_GDN_QKV_BF16`, opt-in, default off): with
    /// [`Self::qwen4_gdn_chain_fuse_enabled`], the dense route's qkv GEMM stores BF16 bits
    /// (halving the GEMM write and the producer read; byte-identical). Off by default because
    /// the PeaceMaker INT8 GEMM's BF16 entry is currently slower than its F32 entry at the qkv
    /// shape (M = 10240, K = 2048, 16384 rows: 5.53 vs 4.96 ms), which outweighs the 0.39 ms the
    /// producer gains.
    pub qwen4_gdn_qkv_bf16: bool,
    /// Qwen4 HC read/write fusion level (`HIPFIRE_QWEN4_HC_FUSE`: 0 off, 1
    /// read side, 2 + attention epilogue, 3 + shared-down epilogue). Default
    /// 3 on gfx1151, 1 on gfx1201, 0 elsewhere; see [`Self::qwen4_hc_fuse_level`].
    pub qwen4_hc_fuse: u8,
    /// Retiled HC up+mix read tail (`HIPFIRE_QWEN4_HC_UP_TILE`), default on
    /// for gfx1151 and gfx1201, off elsewhere.
    pub qwen4_hc_up_tile: bool,
    /// Persistent W-stationary HC up+mix read tail (`HIPFIRE_QWEN4_HC_UP_V2`):
    /// `hyper_read_up_wmma_bf16_gfx1201_v2` / `_gfx1151_v2`, bytewise the retiled
    /// `_t128` (gfx1201) / `_swap` (gfx1151) entry.  Exact gfx1201 and gfx1151
    /// (with the retiled read admitted), default on; `0` is the kill switch.
    pub qwen4_hc_up_v2: bool,
    /// MoE combine zero-init (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`), default on
    /// for gfx1151 and gfx1201, off elsewhere.
    pub qwen4_moe_combine_zinit: bool,
    /// HC row folds (`HIPFIRE_QWEN4_HC_ROW_FOLD`), default on for gfx1151 and
    /// gfx1201, off elsewhere.  gfx1151: combine + shared fold + HC write + the
    /// next HC read's norm/gate in one MoE row kernel.  gfx1201: the same for
    /// the MoE tail (its scaled-add + HC write + norm chain) and for the mixer
    /// tail (HC write + the next read's norm/gate); `=0` restores the unfused
    /// launches on both.
    pub qwen4_hc_row_fold: bool,
    /// Wave-per-token Qwen4 E512/top-10 prefill router
    /// (`HIPFIRE_QWEN4_ROUTER_FAST`), default on for gfx1151 and gfx1201, off
    /// elsewhere.
    pub qwen4_router_fast: bool,
    /// Multi-lane fn-lanes GDN decode core (`HIPFIRE_QWEN4_GDN_LANES`), default
    /// on for gfx1151, off elsewhere.
    pub qwen4_gdn_lanes: bool,
    /// Qwen4 MQ4G128V2 K=640 indexed MoE down with four weight rows per wave
    /// (`HIPFIRE_QWEN4_MOE_DOWN_R4`), default on for gfx1151.
    pub qwen4_moe_down_r4: bool,
    /// Single-token Q8 K=640 shared-expert down + BF16 scaled add in one
    /// launch (`HIPFIRE_QWEN4_SHARED_DOWN_FUSE`), default on for gfx1151.
    pub qwen4_shared_down_fuse: bool,
    /// Exact single-token GDN a/b/z MQ4 projection fusion
    /// (`HIPFIRE_QWEN4_DECODE_ZBA_X3`), default on only for gfx1151.
    pub qwen4_decode_zba_x3: bool,
    /// Exact single-token decode MoE head fusion (BF16 router/selector rows
    /// plus the Q8_0 shared gate/up pair, `HIPFIRE_QWEN4_DECODE_HEAD_FUSE`),
    /// default on only for gfx1151.
    pub qwen4_decode_head_fuse: bool,
    /// Fused PLE gate/norm/conv/stream-add tail on BF16 HC streams
    /// (`HIPFIRE_QWEN4_PLE_FUSE`), default on for gfx1151 and gfx1201, off
    /// elsewhere.
    pub qwen4_ple_fuse: bool,
    /// Split partial-N gfx11 IU4 grids into unchecked full-tile interior and
    /// one guarded tail launch (`kernel.gfx11_iu4_gridspec`, default on).
    pub gfx11_iu4_gridspec: bool,
    /// Use the 16-wave low-footprint IU4 full-tile entry on gfx1100
    /// (`kernel.gfx11_iu4_shape`, default on). gfx1151 keeps its shipped
    /// SET-only selection regardless of this setting.
    pub gfx11_iu4_shape: bool,
    /// Exploit MQ4V2's symmetric `zp = -8 * scale` contract by rebiasing
    /// packed weight codes once and using signed-weight IU4 WMMA
    /// (`kernel.gfx11_iu4_symfold`). Default on for symmetric gfx1100/gfx1151 only.
    pub gfx11_iu4_symfold: bool,

    // ── Quant / format toggles ────────────────────────────────────
    pub hfq3_dp4a: Option<bool>,
    pub hfq3_mmq: Option<bool>,
    pub hfq4_mmq_rdna2: Option<bool>,
    pub fp8_wmma: bool,
    pub dot2_gemv: bool,
    pub gcn5_wave64_hybrid: Option<bool>,
    /// Radiowave experiment: pack two independent HFQ4 QKV rows into one
    /// explicitly compiled wave64 on RDNA3. Default off until model-level
    /// correctness and throughput gates promote it.
    pub rdna3_hfq4_qkv_wave64: bool,
    /// Radiowave experiment: split each gfx1100 QKVZA output row across two
    /// lighter wave32s and join their partials through LDS.
    pub rdna3_hfq4_qkvza_2wave: bool,
    /// Radiowave experiment: package four independent gfx1100 QKVZA row
    /// waves in one workgroup without changing per-row arithmetic.
    pub rdna3_hfq4_qkvza_wavepack4: bool,
    /// Stage the K=2048 QKVZA activation once per eight exact row waves in a
    /// transposed LDS tile on gfx1100.
    pub rdna3_hfq4_qkvza_ldsx8: bool,
    /// Radiowave recovery of the exact gfx1100 autoresearch schedule: keep an
    /// explicit row stride and spell out the five shuffle-reduction stages so
    /// LLVM cannot re-form the reduction as a loop.
    pub rdna3_hfq4_qkvza_reduce_chain: bool,
    /// Radiowave experiment: port the gfx12 QKVZA float4 activation-hoist
    /// schedule to gfx1100 while retaining one independent wave per row.
    pub rdna3_hfq4_qkvza_hoist_x32: bool,
    /// Compile the exact A3B decode shape with eight HFQ4 groups fixed at
    /// compile time, deleting the general QKVZA loop and tail machinery.
    /// Certified default on gfx1100; set
    /// `HIPFIRE_RDNA3_HFQ4_QKVZA_K2048=0` to restore the general kernel.
    pub rdna3_hfq4_qkvza_k2048: bool,
    /// Stage one residual GEMV row's 32 activation values ahead of the
    /// independent packed-weight loads. Certified default on exact gfx1100;
    /// other RDNA3 targets remain explicit opt-in until separately measured.
    pub rdna3_hfq4_residual_stage_x32: bool,
    /// Combine the certified activation staging with a fixed eight-group
    /// K=2048 loop on gfx1100. Opt-in until exact and tg128 gates promote it.
    pub rdna3_hfq4_residual_k2048: bool,
    /// Launch only the active half of the two-row shared-expert down grid.
    /// The kernel maps row0 = blockIdx.x * 2; the legacy launcher submits M
    /// workgroups, leaving the upper half to return immediately.
    pub rdna3_hfq4_sigmoid_tight_grid: bool,
    /// Compile the shared-expert sigmoid-scaled down kernel with gfx1100
    /// temporal buffer-resource weight loads instead of global addressing.
    /// Default-on for gfx1100; set `HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER=0`
    /// to restore global addressing for comparison.
    pub rdna3_hfq4_sigmoid_buffer: bool,
    /// Schedule four shared-expert down rows per wave on gfx1100, reusing
    /// each activation load across four independent weight streams.
    pub rdna3_hfq4_sigmoid_rows4: bool,
    /// Compile the exact Qwen3.5-35B LM-head shape (M=248320, K=2048) with
    /// its eight HFQ4 groups fixed at compile time. gfx1100 experiment only.
    pub rdna3_hfq4_lm_head_k2048: bool,
    /// Compile the gfx1100 A3B MoE gate/up GEMV with its eight K groups fixed
    /// at compile time. Opt-in until exact shadow and tg128 gates promote it.
    pub rdna3_hfq4_moe_gate_up_k2048: bool,
    pub mmq_override: Option<bool>,
    /// Experimental PR #616 packed prefill for uniform MQ4 and MQ4V2.
    pub packed_mq4_prefill: bool,
    pub mmq_min_batch: Option<usize>,
    pub fp16_disabled: bool,
    pub fp16_layer_min: Option<usize>,
    pub fp16_layer_max: Option<usize>,
    pub wo_mmq: bool,
    pub lm_head_wmma_disabled: bool,
    pub lm_head_overwrite: bool,
    /// Eager-only gfx1201 MQ6 X-LDS overwrite and fused F16 rotation (U2):
    /// `Some` = set (`1` on, any other value off); unset = on inside the
    /// Qwen4 forward for prefill chunks of >= 512 rows
    /// (`Gpu::qwen4_mq6_x4_gfx1201`).  Read only on gfx1201.
    pub qwen4_mq6_x4_gfx1201: Option<bool>,
    /// Halo MQ6 X-LDS tile (U3): `Some(Some([BV, RW, prefetch]))`, or
    /// `[0, 0, 0]` for the measured table; `Some(None)` (`0` or an unknown
    /// value) the incumbent; unset = the table inside the Qwen4 forward
    /// (`Gpu::qwen4_mq6_x4_tile`).
    pub qwen4_mq6_x4_tile: Option<Option<[u8; 3]>>,
    /// Fold three MQ6 sibling projections into independent row regions (U3):
    /// `Some` = set (`1` on, any other value off); unset = on inside the
    /// Qwen4 forward on gfx1151 (`Gpu::qwen4_mq6_x4_regions`).
    pub qwen4_mq6_x4_regions: Option<bool>,
    /// gfx1201 MQ6G256V2 native-FP8 prefill GEMM (`HIPFIRE_GFX12_MQ6_FP8`,
    /// developer, opt-in research route): `Some(true)` (`1`) on for any scope
    /// (still exact gfx1201, eager, K % 256 == 0); anything else off
    /// (`Gpu::gemm_mq6g256v2_fp8_applies`).
    pub gfx12_mq6_fp8: Option<bool>,
    /// Weight-plane arm of the MQ6 FP8 GEMM (`HIPFIRE_GFX12_MQ6_FP8_ARM`,
    /// developer): `split` (default, exact hi/lo E4M3 planes) or `round`
    /// (single lossy E4M3_RNE(q-32) plane).
    pub gfx12_mq6_fp8_arm: crate::gemm_mq6_fp8::Mq6Fp8Arm,
    /// Double-buffered continuous-pipeline kernel variant of the MQ6 FP8 GEMM
    /// (`HIPFIRE_GFX12_MQ6_FP8_DB`, developer): `1` selects the `_db_` symbols
    /// (A/W/S/SZ double-buffered in LDS, one barrier per K64 slab); anything
    /// else (default) keeps the single-buffer kernel until measured.
    pub gfx12_mq6_fp8_db: bool,
    /// Exact-INT8 MQ6G256V2 prefill GEMM on gfx1201 / gfx1151
    /// (`HIPFIRE_MQ6_I8`, developer): `Some(true)` (`1`) on for any scope,
    /// `Some(false)` (`0`) off, unset = the per-arch default
    /// (`crate::gemm_mq6_i8::MQ6_I8_DEFAULT_ARCHS`, qwen4 scope)
    /// (`Gpu::gemm_mq6g256v2_i8_applies`).
    pub mq6_i8: Option<bool>,
    /// Certified-builder (PeaceMaker) twin of the exact-INT8 MQ6G256V2 GEMM on
    /// exact gfx1201 (`HIPFIRE_MQ6_I8_PM`, developer): `Some(false)` (`0`) keeps
    /// the hipcc kernel; `Some(true)` (`1`) and unset (default) run
    /// `kernels::GEMM_MQ6G256V2_MMQ_I8_PM_GFX1201` whenever the INT8 route is
    /// admitted and the launch meets its preconditions, else hipcc
    /// (`Gpu::gemm_mq6g256v2_i8_prequant`).
    pub mq6_i8_pm: Option<bool>,
    /// Calibration-only override keeping native-BF16 teachers in BF16
    /// (`HIPFIRE_CALIB_BF16=1`). Mirrors the `calib_force_bf16` snapshot
    /// read so fresh `Gpu` code can take the flag from `self.flags`
    /// instead of a process-global cache.
    pub calib_force_bf16: bool,

    // ── MMQ screening ─────────────────────────────────────────────
    pub mmq_screen: bool,
    pub mmq_screen_threshold: f32,
    pub mmq_diag_quantize_only: bool,

    // ── Kernel variant overrides ─────────────────────────────────
    pub lloyd_mb4: Option<Mb4Mode>,
    pub mq3_mb4: Option<Mb4Mode>,
    /// Symmetric qt49 (MQ3V2 `mq3v2.symmetric`) W3A4 prefill override
    /// (`HIPFIRE_MQ3V25_A4`). Unset follows the per-arch default of
    /// [`Self::mq3v25_a4_enabled`]; `0` is the kill switch.
    pub mq3v25_a4: Option<bool>,
    /// Symmetric qt49 W3A8 prefill on exact gfx1201 (`HIPFIRE_MQ3V25_A8=1`,
    /// developer, default off). Wins over A4 when both are requested.
    pub mq3v25_a8: bool,
    pub hfq4g128_mmq: bool,
    pub hfq3_mmq_layer_min: Option<usize>,
    pub hfq3_mmq_layer_max: Option<usize>,
    pub hfq4_mmq_gfx906_y64: bool,
    pub gate_up_variant: Option<String>,
    pub gate_up_nosync: bool,
    /// gfx12 LDS-staged HFQ4-G256 residual/gate_up WMMA path. Reorders FP32
    /// K accumulation across 8 waves (not bit-exact). Default OFF; opt in with
    /// `HIPFIRE_HFQ4G256_LDSSTAGE=1`. Requires K % 512 == 0 at the launch site.
    pub hfq4g256_ldsstage_wmma: bool,
    /// RDNA3 QKVZA prefill route that keeps the large QKV/Z projections on
    /// MMQ while sending the narrow beta/alpha tails through dot2.
    /// Opt in with HIPFIRE_QKVZA_SPLIT_TAIL=1.
    pub qkvza_split_tail: bool,
    pub gfx942_gemv_v2: Option<bool>,
    pub gfx942_gemv_v3: bool,
    pub gfx942_rmsnorm_split: bool,
    /// Reserve only the reduction scratch still used by the current fused
    /// RMSNorm+MQ kernel. The historical K-float LDS cache was removed when
    /// the prefetch schedule landed, but its launch reservation remained.
    pub rmsnorm_mq_tight_lds: bool,
    /// Radiowave experiment: split the K=2048 fused RMSNorm+MQ rotation over
    /// eight resident wave32 workgroups on gfx1100. The workgroups retain the
    /// baseline reduction order and rendezvous through persistent GPU scratch.
    pub rdna3_rmsnorm_wavegrid: bool,
    /// Radiowave experiment: replace the cross-workgroup rendezvous with one
    /// exact reduction dispatch followed by eight independent rotate waves.
    pub rdna3_rmsnorm_split: bool,
    /// Compute the K=2048 RMS sum from each FWHT wave's prefetched float4
    /// values, eliminating the baseline's second x load. Certified default on
    /// gfx1100; set `HIPFIRE_RDNA3_RMSNORM_VECSUM=0` to restore the baseline.
    pub rdna3_rmsnorm_vecsum: bool,
    /// Radiowave experiment: stage the two shared MQ sign tables once in LDS
    /// for the gfx1100 vecsum kernel instead of reloading them in eight waves.
    pub rdna3_rmsnorm_sign_lds: bool,
    /// Radiowave experiment: compile the deterministic MQ sign streams as
    /// packed constants and apply them with FP32 sign-bit XORs.
    pub rdna3_rmsnorm_sign_const: bool,
    pub gfx942_mfma_prefill: Option<String>,
    pub moe_grouped_i8: Option<bool>,
    pub moe_grouped_i8_k8: bool,
    pub moe_grouped_i8_k4: bool,
    pub moe_grouped_i8_k4_gfx12: bool,
    pub moe_grouped_m2: bool,
    pub moe_grouped_4w: bool,
    /// Radiowave experiment: fold four adjacent K=8 MoE down outputs per
    /// thread with vector loads and independent exact accumulators.
    pub moe_down_combine_vec4: bool,
    pub moe_hfq6_i8: bool,
    pub moe_hfq6_v2: bool,
    // ── MoE prefill (Ship 4.2) ────────────────────────────────────
    /// Grouped-GEMM MoE prefill gate (HIPFIRE_MOE_GROUPED_GEMM). Default ON.
    pub moe_grouped_gemm: bool,
    /// gfx1151 i8 MMQ opt-in for Paro grouped GEMM (HIPFIRE_MOE_PARO_I8).
    /// None = arch-default (gfx1151 → true, else false).
    pub moe_paro_i8: Option<bool>,
    /// gfx1151 i8 MMQ k8 opt-in for Paro grouped GEMM (HIPFIRE_MOE_PARO_I8_K8).
    /// None = arch-default (gfx1151 → true, else false).
    pub moe_paro_i8_k8: Option<bool>,

    // ── Graph / capture / deterministic ─────────────────────────────
    pub graph_forward: Option<bool>,
    /// Autoregressive-forward hipGraph capture (`experimental.graph.ar`,
    /// `HIPFIRE_AR_GRAPH`). Default on, except on Windows exact gfx1100, where
    /// it defaults off (#655); see [`graph_ar_default`].
    pub graph_ar: bool,
    pub graph_moe: bool,
    pub force_blob_path: bool,
    /// `HIPFIRE_RESIDUAL_KSPLIT_OFF=1` disables the exact-gfx1100 split-K LDS
    /// residual tier (N<=16 DFlash verify) and restores the historical base
    /// kernel on the policy path. Default OFF (tier live). Test harnesses use
    /// this to force the base oracle now that the tier is capture-safe.
    /// Disables BOTH the ksplit and ldsstage kernels.
    pub residual_ksplit_off: bool,
    /// `HIPFIRE_RESIDUAL_LDSSTAGE` selects the exact-gfx1100 N<=16 tier's
    /// ldsstage kernel wherever K % 512 == 0. Certified default on exact
    /// gfx1100; other arches default false; launchers remain exact-gfx1100-only. Set
    /// `HIPFIRE_RESIDUAL_LDSSTAGE=0` to restore the split-K table path.
    pub residual_ldsstage: bool,
    /// `HIPFIRE_GATEUP_LDSSTAGE` selects the exact-gfx1100 N<=16 MQ4V2 gate_up
    /// RAW-slab ldsstage on eligible eager HIP (1<=N<=16, K%512==0). Certified
    /// default on exact gfx1100; other arches default false; launchers remain
    /// exact-gfx1100-only. Capture/replay keep the historical base. Set
    /// `HIPFIRE_GATEUP_LDSSTAGE=0` to restore the historical base symbol/block32.
    pub gate_up_ldsstage: bool,
    /// `HIPFIRE_GFX1100_Q8X_VERIFY` routes exact-gfx1100 2..=5-row MQ4G256V2
    /// projections (speculative verify) to the int8-activation small-batch
    /// GEMV (`gemv_q8x.rs`) instead of the f16 WMMA residual tiers. Eager
    /// only (capture/recording keep the WMMA tiers). Not byte-identical
    /// (q8_1-class activation quantization, KLD-gated). Default on for exact
    /// gfx1100, false elsewhere; `HIPFIRE_GFX1100_Q8X_VERIFY=0` restores the
    /// WMMA tiers.
    pub gfx1100_q8x_verify: bool,
    /// `HIPFIRE_GFX1151_XLDS` selects the exact-gfx1151 1..=16-row MQ4V2 and
    /// HFQ4G256/MQ4G256 v1 WMMA GEMMs (`*_gfx1151_xlds`: 4 waves x 16 rows per
    /// block, X slab staged in LDS) for gate_up/qkvza/qkv/residual (residual also
    /// carries the 4-bit batched lm_head), the matching Q8_0
    /// `gemm_q8_0_wmma_gfx1151_xlds` (DFlash Q8 lm_head, Q8 gate_up) and the F16
    /// `gemm_f16_wmma_gfx1151_xlds` (DSpark drafter lm_head). Default on for exact
    /// gfx1151, false elsewhere; outputs are byte-identical to the base kernels.
    /// `HIPFIRE_GFX1151_XLDS=0` restores the historical base kernels.
    pub gfx1151_xlds: bool,
    /// Exact-K5120 MQ4V2 gate/up decode on gfx1100: default on (byte-identical
    /// to the generic kernel; `HIPFIRE_MQ4V2_GATEUP_K5120=0` opts out).
    pub mq4v2_gateup_k5120: bool,
    /// Default-off exact gfx1201 K5120 MQ4V2 gate/up decode twin.
    pub gfx12_mq4v2_gateup_k5120: bool,
    /// `HIPFIRE_GFX12_MQ4V2_FP8_GATEUP=0` opts out of the gfx1201 FP8-WMMA MQ4v2
    /// gate_up prefill candidate (N=384, eager HIP only). Default ON on exact
    /// gfx1201; `=1` forces it on other arches (launchers stay gfx1201-only).
    pub gfx12_mq4v2_fp8_gateup: bool,
    /// FP8-WMMA MQ4v2 residual (down/out-proj) prefill candidate. Default ON on
    /// exact gfx1201; `=0` opts out.
    pub gfx12_mq4v2_fp8_resid: bool,
    /// FP8-WMMA MQ4v2 4-way QKVZA prefill candidate. Default ON on exact
    /// gfx1201; `=0` opts out.
    pub gfx12_mq4v2_fp8_qkvza: bool,
    /// FP8-WMMA MQ4v2 3-way QKV (full-attention) prefill candidate. Default ON
    /// on exact gfx1201; `=0` opts out.
    pub gfx12_mq4v2_fp8_qkv: bool,
    /// `HIPFIRE_WMMA_BATCH_TILES=0` opts out of the exact-gfx1201 batch-tiled
    /// MQ4V2 WMMA GEMMs at 16 < N < 64 (qkvza, qkv, gate_up, residual/lm_head):
    /// one block loads and dequantizes each 16-row weight fragment once per K
    /// step and feeds up to 4 consecutive 16-row batch tiles through independent
    /// accumulators (the BT4 kernels), so the multi-request speculative verify
    /// trunk shares weight bandwidth across requests. Per-(row, batch) WMMA
    /// sequence, K order and epilogue are identical to the one-tile kernels, so
    /// outputs are byte-identical. N <= 16 never uses it. Default ON on exact
    /// gfx1201; `=0` restores the one-batch-tile-per-block launch.
    pub wmma_batch_tiles: bool,
    /// `HIPFIRE_CB_VERIFY_CHUNK128` (default ON on gfx1201, `=0` is the kill
    /// switch) admits exact 64..=128-row aggregated verify chunks for the
    /// continuous-batching speculative route on exact gfx1201 (see
    /// `Gpu::mq4_verify_chunk_rows`): up to 128 rows per chunk instead of 63,
    /// so 8 lanes of 8-row blocks (64 rows) run one weight pass instead of
    /// two. `=0` keeps the 63-row chunks and every existing route. Needs
    /// `wmma_batch_tiles`.
    pub cb_verify_chunk128: bool,
    /// `HIPFIRE_CB_VERIFY_PM` (default ON; only meaningful with the wide
    /// verify route): runs the 64..=128-row exact kernels from the embedded
    /// PeaceMaker bundle; `=0` runs their byte-identical hipcc twins. Never
    /// selects IU4/FP8, and a missing PM image is a load error.
    pub cb_verify_pm: bool,
    /// `HIPFIRE_CB_VERIFY_FP8=1` (developer, default OFF): the Qwen3.5
    /// hybrid target verify (DFlash and MTP; singleton and continuous
    /// batching) runs its dense MQ4G256V2 GEMMs, including the verify
    /// lm_head, through the PeaceMaker FP8 verify family
    /// (`crate::gemm_fp8_verify`): symmetric weights dequantized exactly to
    /// E4M3, per-row E4M3 activations, `v_wmma_f32_16x16x16_fp8_fp8`, and a
    /// per-output reduction order fixed for every row count 1..=128, so
    /// batched rows are byte-identical to singleton rows by construction.
    /// Needs exact gfx1201, symmetric MQ4G256V2 weights and an eligible
    /// forward; the scope entry fails closed otherwise. Not the PM F16
    /// family: `HIPFIRE_CB_VERIFY_PM` does not apply.
    pub cb_verify_fp8: bool,
    /// `HIPFIRE_CB_VERIFY_IU4` (developer): DFlash IU4 (A4-activation) verify.
    /// `None` (unset) = the per-load default (on when `speculation.dflash_block`
    /// resolves to 8, i.e. a batching serve, unless `HIPFIRE_CB_VERIFY_FP8` is
    /// on); `Some(false)` = the kill switch; `Some(true)` = forced on (with
    /// `HIPFIRE_CB_VERIFY_FP8=1` the DFlash load fails). The decision is taken
    /// in `load_dflash_state` (`Gpu::iu4_verify_load`).
    pub cb_verify_iu4: Option<bool>,
    /// Staged-tile v2 selector for the four gfx1201 FP8-WMMA MQ4v2 prefill
    /// routes (`HIPFIRE_GFX12_MQ4V2_FP8_V2`, `kernel.gfx12_mq4v2_fp8_v2`).
    /// Default ON on exact gfx1201; `=0` selects the s2bt8/BT symbols.
    /// N>=256 uses the `_v2_gfx1201` symbols (smaller batches keep s2bt8/BT).
    /// The four family flags remain prerequisites; the activation prelude is
    /// unchanged.
    pub gfx12_mq4v2_fp8_v2: bool,
    /// gfx1201 batched prefill GDN preamble fusion (slice P:
    /// `HIPFIRE_GFX12_GDN_PRE_FUSED`, `kernel.gfx12_gdn_pre_fused`).
    /// Default ON on exact gfx1201; `=0` restores the 3-launch sequence.
    /// Fuses sigmoid+conv+qknorm into `gdn_pre_batched_gfx1201` on the
    /// sequential dense prefill route. Byte-exact vs the 3-launch sequence;
    /// any byte difference kills it.
    pub gfx12_gdn_pre_fused: bool,
    /// fused chunked gated-delta-net prefill scan, WMMA-resident state (hipfire design; structure informed by public FLA/AITER chunk decomposition).
    /// Enabled by default on gfx1100/gfx1151/gfx1201; `HIPFIRE_GFX12_GDN_CHUNK_SCAN=0` opts out.
    pub gfx12_gdn_chunk_scan: bool,
    /// gfx1201 down-proj SwiGLU/FWHT + int4 quant fusion (slice 1:
    /// `HIPFIRE_GFX12_SILU_QUANT_FUSED`, `kernel.gfx12_silu_quant_fused`).
    /// Default ON on exact gfx1201; `=0` opts out to the standalone
    /// `quantize_int4_mmq_ds128` launch for the down-proj input.
    /// Bit-identical vs the silu+quantizer chain; any byte difference kills it.
    pub gfx12_silu_quant_fused: bool,
    /// gfx1201 RMSNorm/rotate/gated-norm + int4 quant fusions (slices 2-4:
    /// `HIPFIRE_GFX12_PRODUCER_QUANT_FUSED`, `kernel.gfx12_producer_quant_fused`).
    /// Default ON on exact gfx1201; `=0` opts out to the standalone
    /// `quantize_int4_mmq_ds128` launch at each admitted site.
    /// Bit-identical vs the producer+quantizer chain; any byte difference
    /// kills it.
    pub gfx12_producer_quant_fused: bool,
    /// gfx1201 `_v2` RMSNorm and gated-norm IU4 producers (`HIPFIRE_G12_NORM`,
    /// `kernel.g12_norm`). Default ON on exact gfx1201; `=0` restores the
    /// incumbent `_gfx12` symbols. The RMSNorm twin batches its
    /// sum-of-squares loads and forms RTN codes with one reciprocal per lane;
    /// the gated-norm twin runs one wave per 256-group. Bit-identical outputs;
    /// any byte difference kills it.
    pub g12_norm: bool,
    /// Decode norm grids (`HIPFIRE_G12_DEC_NORM`, `kernel.g12_dec_norm`).
    /// Default ON on exact gfx1201 and exact gfx1151; `=0` restores the
    /// single-workgroup launches. The f32 AWQ RMSNorm+FWHT runs K/256
    /// workgroups (each redoes the row's reduction, rotates one group), the
    /// out-of-place `rmsnorm_f32` n/256 workgroups per row, and the half-split
    /// partial RoPE one workgroup per head. Bit-identical outputs; any byte
    /// difference kills it.
    pub g12_dec_norm: bool,
    /// gfx1100 decode norm grids (`HIPFIRE_GFX1100_DEC_NORM`,
    /// `kernel.gfx1100_dec_norm`). Default ON on exact gfx1100; `=0` restores
    /// the single-workgroup launches. Same twins as `g12_dec_norm` built for
    /// gfx1100: the f32 AWQ RMSNorm+FWHT runs K/256 workgroups and the
    /// out-of-place `rmsnorm_f32` n/256 workgroups per row (RoPE is untouched:
    /// gfx1100 decode fuses it into its FA prep). Bit-identical outputs; any
    /// byte difference kills it.
    pub gfx1100_dec_norm: bool,
    /// gfx11 sigmoid/gated-norm + int4 quant fusions
    /// (`HIPFIRE_GFX11_PRODUCER_QUANT_FUSED`,
    /// `kernel.gfx11_producer_quant_fused`). Default ON on gfx1100/gfx1151;
    /// `=0` opts out to the standalone `quantize_int4_mmq_ds128` launch at
    /// each admitted site. Bit-identical vs the producer+quantizer chain;
    /// any byte difference kills it.
    pub gfx11_producer_quant_fused: bool,
    /// gfx1201 RMSNorm+rotate producer → MQ4v2 FP8 pre-pass fusion
    /// (`HIPFIRE_GFX12_FP8_STREAM`, `kernel.gfx12_fp8_stream`). Default ON on
    /// exact gfx1201; `=0` opts out. The `_mq4v2_fp8_gfx12` producer twins
    /// emit byte-identical `prepare_mq4v2_fp8_x_f32` (scale_mode=1) outputs
    /// for the qkvza/gate_up/qkv inputs, so the standalone
    /// `pack_f32_to_fp8_mq4v2_gfx12` launch disappears at each admitted site.
    /// Any byte difference kills it.
    pub gfx12_fp8_stream: bool,
    /// `HIPFIRE_GFX12_FA2_PREFILL=0` opts out of the gfx1201 GQA-fused FA2
    /// prefill attention candidate (Qwen NH24/NKV4/HD256, eager HIP only).
    /// Default ON on exact gfx1201; `=1` forces it on other arches
    /// (launchers stay gfx1201-only).
    pub gfx12_fa2_prefill: bool,
    /// `HIPFIRE_FA_PREFILL_PM` (`developer.fa_prefill_pm`): typed flag for
    /// A3B H16/KV2/D256 Q8_0 prefill. Unset follows route qualification
    /// (on only on exact gfx1201); explicit 1/0 wins, admission still applies.
    pub fa_prefill_pm: bool,
    /// The same `HIPFIRE_FA_PREFILL_PM` key resolved for the native fp8-E4M3 KV codec: an
    /// explicit 1/0 wins, unset follows `PrefillRoute::FaPrefillPm` fp8 qualification
    /// (`prefill_route_capabilities::FP8_KV_QUALIFIED`, on only on exact gfx1201).
    /// The Q8_0 qualification never enables fp8 KV.
    pub fa_prefill_pm_fp8_kv: bool,
    /// `HIPFIRE_MOE_IU4_WIDE_TILES` (`developer.moe_iu4_wide_tiles`): typed flag
    /// for the NT8 gate/up expert-run tile on VRAM-resident symmetric IU4 MoE experts
    /// (exact gfx1201 raw inventory, per-expert real slots >= 128; see
    /// `Gpu::qwen4_moe_sym_gemm_symbols`). The resident down entry stays NT4 whatever this
    /// says. Unset follows route qualification (on only on exact gfx1201); explicit 1 opts in
    /// (raw inventory is applied at selection), 0 kills.
    /// Host-mapped experts keep their existing NT8 gate/up and down whatever this says.
    pub moe_iu4_wide_tiles: bool,
    /// `HIPFIRE_FA_PREFILL_PM_QPREP` (`developer.fa_prefill_pm_qprep`): typed
    /// flag for the exact Q-prep fold that feeds the PM direct attention
    /// (A3B H16/KV2/D256 Q8_0 prefill). Unset follows route qualification (on only on exact
    /// gfx1201), explicit 1 opts in, 0 kills. Independent of `fa_prefill_pm` as a flag; the
    /// route nests beneath the *resolved* PM flag (an explicit PM 0 refuses the fold), and
    /// admission still applies.
    pub fa_prefill_pm_qprep: bool,
    /// The same `HIPFIRE_FA_PREFILL_PM_QPREP` key resolved for the native fp8-E4M3 KV codec;
    /// qualification as [`Self::fa_prefill_pm_fp8_kv`] (on only on exact gfx1201 when unset).
    pub fa_prefill_pm_qprep_fp8_kv: bool,
    /// `HIPFIRE_MOE_ROTATE_I4_PM` (`developer.moe_rotate_i4_pm`): typed flag for the exact
    /// PeaceMaker twin of the down A4 producer `qwen4_moe_rotate128_i4`, resolved through its
    /// qualification (on only on exact gfx1201 when unset; explicit 1 opts in, 0 kills). Raw
    /// inventory, the requested symmetric IU4 route, the C2 candidate set and the launch shape
    /// are applied at selection.
    pub moe_rotate_i4_pm: bool,
    /// `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` (`developer.prefill_iu4_producer_fused`): typed
    /// flag for the exact IU4 producer fusion, resolved through its qualification (on only on
    /// exact gfx1201 when unset; explicit 1 opts in, 0 kills). The raw inventory is a separate
    /// conjunct at the callers. Read here only so the exact C8 PM wrappers can nest beneath it.
    pub prefill_iu4_producer_fused: bool,
    /// `HIPFIRE_PREFILL_IU4_C8_PM` (`developer.prefill_iu4_c8_pm`): typed flag for the
    /// PeaceMaker C8 rotate / RMSNorm-rotate producer twins, resolved through its qualification
    /// (on only on exact gfx1201 when unset; explicit 1 opts in, 0 kills; raw inventory is
    /// applied at admission). Nests beneath the resolved `prefill_iu4_producer_fused` flag.
    pub prefill_iu4_c8_pm: bool,
    /// `HIPFIRE_GDN_CHUNK_PREP_PM` (`developer.gdn_chunk_prep_pm`): typed flag for the PM
    /// twin of `gdn_chunk_prep` (A3B 16/32-head GDN chunk-scan preamble), resolved through its
    /// qualification (on only on exact gfx1201 when unset; explicit 1 opts in, 0 kills).
    /// Independent of the chunk-scan flag as a flag; the MoE chunk-scan layer admission still
    /// applies in `hipfire-arch-qwen35`.
    pub gdn_chunk_prep_pm: bool,
    /// gfx1201 packet-minimal Q128 FA2 body (`HIPFIRE_GFX12_FA_PACKET`,
    /// `kernel.gfx12_fa_packet`). Default ON on exact gfx1201; `=0` opts out
    /// to the route-N body. Exact stage-b arithmetic; any raw-O bit difference
    /// kills it.
    pub gfx12_fa_packet: bool,
    /// Register-resident-Q wide gfx1201 FA2 prefill body
    /// (`HIPFIRE_ATTN_QRESIDENT`, `kernel.attn_qresident`). Default ON on
    /// exact gfx1201; `=0` opts out. Exact H24/KV4/D256 native-fp8-KV shapes only.
    pub attn_qresident: bool,
    /// Bit-exact reschedule of the Q-resident body
    /// (`HIPFIRE_ATTN_QRESIDENT_V2`, `kernel.attn_qresident_v2`). Default ON
    /// on exact gfx1201; `=0` restores the v1 Q-resident kernel. Only
    /// consulted where `attn_qresident` selects the Q-resident route.
    pub attn_qresident_v2: bool,
    /// VerifyAttn (`HIPFIRE_VERIFY_ATTN`, `kernel.verify_attn`): the
    /// GQA-shared split-K twin of the batched flash tile + reduce for
    /// speculative-verify-sized batches (1..=32 rows, non-tree), on gfx1100
    /// also of the multi-row R4/R8 Q8 tile, and on gfx1151 the
    /// context-parallel twin of the single-slot WMMA flash prefill. Default ON
    /// on exact gfx1201, gfx1100 and gfx1151; `=0` opts out to
    /// `attention_flash_*_tile_batched`, `attention_flash_q8_0_rows{4,8}_d8`
    /// and `attention_q8_0_flash_prefill_wmma`. Byte-identical output; any
    /// byte difference kills it.
    pub verify_attn: bool,
    /// Exact-gfx1100 FA2 split-KV implementation of VerifyAttn
    /// (`HIPFIRE_GFX1100_FA2_SPLIT_VERIFY`,
    /// `kernel.gfx1100_fa2_split_verify`). Default ON only on exact gfx1100
    /// inside its existing VMM-KV admission envelope. This switch and the
    /// stable [`Self::verify_attn`] parent are independent opt-outs.
    pub gfx1100_fa2_split_verify: bool,
    /// `HIPFIRE_GFX1100_WMMA_VERIFY_ATTN` routes the exact-gfx1100 admitted
    /// multi-row Q8 verify attention (1..=8 rows, H24/KV4/HD256, plain Q8 K
    /// and V) to the WMMA split-K kernel (`attention_wmma_verify.rs`) ahead of
    /// the FA2 split-KV verifier / VerifyAttn rows twin / R4-R8 tile. Eager
    /// only (capture/recording keep those routes). Not byte-identical (f16
    /// WMMA operands, KLD-gated). Default on for exact gfx1100, false
    /// elsewhere; `=0` restores the incumbent routes. The [`Self::verify_attn`]
    /// parent opts out too.
    pub gfx1100_wmma_verify_attn: bool,
    /// Exact gfx1201 FA deinterleave, Q/K norm and RoPE fusion.
    /// `HIPFIRE_GFX12_FA_PREP_FUSED=0` restores the original chain.
    pub gfx12_fa_prep_fused: bool,
    /// Emit preconverted fp8 Q for the v2 Q-resident attention twin.
    /// `HIPFIRE_GFX12_FA_PREP_FP8Q=0` retains F32 Q.
    pub gfx12_fa_prep_fp8q: bool,
    /// Whole-chunk Q8/Q8 FA2 on exact gfx1100/gfx1151
    /// (`HIPFIRE_GFX11_Q8_FA2_WIDE`, `kernel.gfx11_q8_fa2_wide`).
    /// Default ON on exact gfx1100 and gfx1151 (whose FA2 twin dispatches
    /// the heaviest q-tiles first); explicit false opts out. Requires the
    /// master gfx11 FA2 flag. B64..8192, above-512 batches aligned to 512,
    /// ctx64..32768.
    pub gfx11_q8_fa2_wide: bool,
    /// `HIPFIRE_GFX11_FA2_PREFILL=0` opts out of the gfx11 GQA-fused FA2
    /// prefill attention candidate (Qwen NH24/NKV4/HD256, eager HIP only).
    /// Default ON on gfx1100/gfx1151; `=1` forces it on other arches
    /// (launchers stay on the gfx11 allowlist).
    pub gfx11_fa2_prefill: bool,
    pub gemm_dump: bool,
    pub deterministic: bool,
    pub q8_batched_legacy: bool,
    /// Optional ABI-v3 CK runtime artifact. Resolved once at GPU startup;
    /// absence or load/capability failure leaves native dispatch unchanged.
    pub flash_attn_ck_lib: Option<String>,
    /// Caller-owned CK staging workspace, allocated once at GPU startup.
    pub flash_attn_ck_workspace_bytes: usize,
    /// Fuse Gemma 4 Q8 prefill projections on exact gfx1100. This remains an
    /// opt-in while the E-series path is validated across QKV-sharing shapes.
    pub gemma4_q8_fused_prefill: bool,
    /// Batch Gemma 4 main and PLE embedding lookup on validated gfx1100 and
    /// gfx1201 paths. The `auto` policy remains independently disableable.
    pub gemma4_batched_embedding_prefill: bool,
    /// Batch the E-series per-layer-input model projection on exact gfx1100
    /// instead of re-streaming the same matrix through one GEMV per row.
    pub gemma4_ple_batched_prefill: bool,
    /// Exact-arithmetic batched E-series PLE branch projections on validated
    /// gfx1100/gfx1201 paths. Remains independently disableable.
    pub gemma4_ple_branch_batched_prefill: bool,
    /// Fuse the E-series PLE GELU and strided per-layer multiply on validated
    /// gfx1100 and gfx1201 paths.
    pub gemma4_ple_activation_fused_prefill: bool,
    /// `HIPFIRE_DEEPSEEK4_Q8_WMMA=0` disables the Q8_0 dense WMMA prefill path
    /// (forces the scalar chunked fallback). Consumed by
    /// `gemm_q8_0_wmma_prefill_auto`.
    pub deepseek4_q8_wmma_off: bool,
    /// `HIPFIRE_DEEPSEEK4_Q8_4W=0` forces the single-warp 16×16 Q8_0 WMMA
    /// kernel instead of the 4-warp 64×64 tile. Consumed by
    /// `gemm_q8_0_wmma_prefill_auto`.
    pub deepseek4_q8_4w_off: bool,
    pub rope_interleaved_legacy: bool,
    pub wo_wmma_variant: Option<String>,

    // ── rocBLAS ────────────────────────────────────────────────────
    pub rocblas_all_archs: bool,
    pub rocblas_off: bool,
    pub rocblas_min_batch: Option<usize>,
    /// Batched-attention tile size (`HIPFIRE_ATTN_TILE_SIZE`). `None` = use the
    /// 128 default. gfx1151 is the dev box, gfx1201 the deployment target, so
    /// this must never be a baked-in `const` (spec §11).
    pub attn_tile_size: Option<usize>,
    /// Multi-slot attention flash-vs-scalar crossover in tokens
    /// (`HIPFIRE_SLOTS_ATTN_CROSSOVER`). `None` = the per-arch default. Same
    /// reasoning as `attn_tile_size`: gfx1151 is the dev box and gfx1201 the
    /// deployment target, so this must be overridable rather than baked in.
    pub slots_attn_crossover: Option<usize>,
    /// Capture a pure-decode multi-slot step into a hipGraph and replay it
    /// (`HIPFIRE_SLOTS_DECODE_GRAPH`). Off by default.
    pub slots_decode_graph: bool,
    /// Trace multi-slot session continuation matching (`HIPFIRE_SLOT_TRACE`).
    /// Diagnostic only.
    pub slot_trace: bool,

    // ── Kernels.rs env reads ───────────────────────────────────────
    pub lloyd_force_baseline: bool,
    pub rdna2_variant: Option<u32>,

    // ── Compiler.rs env reads ──────────────────────────────────────
    /// `HIPFIRE_HIPCC_EXTRA_FLAGS`: global debug override for all module
    /// compilations, not a way to ship flags. Production scheduler choices
    /// belong in each module's Radiowave profile registration.
    pub hipcc_extra_flags: String,

    // ── Interpreter Phase 2a ───────────────────────────────────────
    /// Force the discrete (un-fused) projection path where supported, for
    /// fused-vs-unfused validation. Env: HIPFIRE_FORCE_UNFUSED=1. Single-GPU
    /// decode projection fusions only (see Phase-2a spec §4b honest-scope).
    pub force_unfused: bool,

    // ── Speculative decode (DFlash/DDTree) ────────────────────────
    /// DDTree tree-SWOR verify arm. **Default OFF** — linear chain wins on
    /// every drafter measured (the DFlash drafter's independent per-position
    /// marginals give a tree branch no joint to exploit). Opt in with the CLI
    /// `--ddtree` flag (sets `HIPFIRE_DFLASH_TREE=1`). Tree-SWOR is still
    /// distribution-exact at any temperature and lossless (== AR) at temp 0 —
    /// it's just slower than chain — so the opt-in path stays correct.
    pub dflash_tree: bool,
    /// Override DDTree node budget (`HIPFIRE_DDTREE_BUDGET`). `None` → use the
    /// per-call-site default (`DEFAULT_TREE_BUDGET = 8`).
    pub ddtree_budget: Option<usize>,
    /// Override DDTree per-position top-K breadth (`HIPFIRE_DDTREE_TOPK`).
    /// `None` → use the per-call-site default (`DEFAULT_TREE_TOPK = 2`).
    pub ddtree_topk: Option<usize>,
    // D8: ddtree_verify_naive removed — SWOR is the only temp>0 verify path.
    // The naive fallback required a ~37 MB/cycle full-logits D2H and is
    // superseded by SWOR (distribution-exact, on-GPU). HIPFIRE_DDTREE_VERIFY
    // env var is no longer parsed; setting it has no effect.
    /// qwen35 ddtree tree-LA (linearized-ancestor) fast-tape path. **Default ON**;
    /// opt out with `HIPFIRE_DDTREE_TREE_LA=0`.
    pub ddtree_tree_la: bool,
    /// qwen35 DFlash GPU softmax/nucleus fast path on the temp>0 sampled verify.
    /// **Default ON**; opt out with `HIPFIRE_DFLASH_FAST_SAMPLE=0`.
    pub dflash_fast_sample: bool,
    /// qwen35 ddtree meta-verifier expansion cutoff (`HIPFIRE_DDTREE_LOGW_CUTOFF`).
    /// Stores the positive X the user set (stop tree expansion when the next
    /// candidate's cumulative logw < −X); `None`/0/unparseable → no cutoff.
    /// Use [`FeatureFlags::ddtree_logw_cutoff_value`] for the resolved threshold.
    pub ddtree_logw_cutoff: Option<f32>,
    /// qwen35 DFlash Q8 WMMA lm_head in verify. **Default ON**; opt out with
    /// `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` ∈ {0,false,off,no}.
    pub dflash_q8_lmhead_wmma: bool,

    /// Fold the 3 separate QKV `BiasAdd` launches (qwen2-family `attention_bias`)
    /// into the per-row fused-QKV decode kernel's lane-0 store. Eliminates 3 tiny
    /// overhead-dominated launches/layer. Env: HIPFIRE_FUSE_QKV_BIAS — default ON;
    /// set `=0` to opt out (for bisection). NULL bias pointers are a kernel no-op,
    /// so non-bias callers stay byte-identical regardless. Only fires for fused-QKV
    /// keys whose dispatch arm reads the bias (see `qkv_bias_fold_supported`).
    pub fuse_qkv_bias: bool,

    /// Diagnostic: log each QKV-bias fold to stderr. Env:
    /// HIPFIRE_FUSE_QKV_BIAS_DEBUG=1. Default off. Resolved once at init so the
    /// default-on fold hot path takes no per-launch `env::var` lock.
    pub fuse_qkv_bias_debug: bool,

    // ── DFlash launch-fusion kill switches (prescaffold, all no-ops) ────
    // Each `HIPFIRE_*_OFF=1` disables its slice's fast route and restores the
    // pre-change path. All default OFF (fast routes live once slices land);
    // nothing reads these fields yet — composers wire them in per slice.
    /// S1: `HIPFIRE_DN_SNAPSHOT_BULK_OFF=1` restores the memcpy-loop snapshot.
    pub dn_snapshot_bulk_off: bool,
    /// S2: `HIPFIRE_HIDDEN_SCATTER_FUSE_OFF=1` restores the row-copy loops.
    pub hidden_scatter_fuse_off: bool,
    /// S3: `HIPFIRE_MQ_F16_PROJECTION_OFF=1` restores F32 producers + convert.
    pub mq_f16_projection_off: bool,
    /// S4: `HIPFIRE_MQ_F16_RESIDUAL_OFF=1` restores F32 residual producers.
    pub mq_f16_residual_off: bool,
    /// S5: `HIPFIRE_GDN_PRE_FUSE_OFF=1` restores unfused GDN pre-kernels.
    pub gdn_pre_fuse_off: bool,
    /// S6: `HIPFIRE_FA_BATCH_FUSE_OFF=1` restores unbatched FA prep/KV writes.
    pub fa_batch_fuse_off: bool,
    /// S7: `HIPFIRE_DRAFT_COLLAPSE_OFF=1` restores scalar draft embeddings. Also
    /// vetoes the gfx1201 DFlash draft overwrite-GEMM route
    /// (`gemm_mq4g256v2_overwrite_gfx12_dflash`), restoring memset + residual.
    pub draft_collapse_off: bool,
    /// DFlash draft shared MQ4V2 input rotation (#838), one switch per
    /// projection group: noise Q/K/V (`HIPFIRE_DRAFT_SHARED_QKV_ROTATION`,
    /// `kernel.draft_shared_qkv_rotation`), FFN gate/up
    /// (`HIPFIRE_DRAFT_SHARED_FFN_ROTATION`, `kernel.draft_shared_ffn_rotation`)
    /// and incremental context K/V (`HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION`,
    /// `kernel.draft_shared_ctx_kv_rotation`). Default ON on exact gfx1100,
    /// off elsewhere; `=0` restores one rotation per projection. The runtime
    /// admission (hipfire-runtime `dflash.rs` `gemm_dispatch_shared`) also
    /// requires exact gfx1100, so the switch is ignored on every other arch.
    /// Same rotation bytes, same GEMMs; bit-identical outputs.
    pub draft_shared_qkv_rotation: bool,
    pub draft_shared_ffn_rotation: bool,
    pub draft_shared_ctx_kv_rotation: bool,
    /// S8: `HIPFIRE_DDTREE_TOPK_DIRECT_OFF=1` restores full-logits top-K.
    pub ddtree_topk_direct_off: bool,
    /// S9: `HIPFIRE_MQ_PROLOGUE_FUSE_OFF=1` restores producer+GEMM pairs.
    pub mq_prologue_fuse_off: bool,
    /// Railgun E0 / L6c: `HIPFIRE_GDN_REPLAY_ML_OFF=1` restores the per-layer
    /// GDN replay launches (the gfx1201 two-launch multi-layer replay is
    /// default-on there).
    pub gdn_replay_ml_off: bool,
    /// Railgun E0 / L6d: `HIPFIRE_SELECT_REGRID_OFF=1` restores the one-block-
    /// per-row `topk_values_batched_f32` / `argmax_f32_batched` launches (the
    /// gfx1201 multi-block re-grid is default-on there).
    pub select_regrid_off: bool,
    /// Railgun D8: exact gfx1201 DFlash rollback without the restore copy: the
    /// GDN tape replay reads the pre-verify state from the snapshot buffers and
    /// writes the live ones (the snapshot stays the pre-window state).
    /// Byte-identical; default on (the replay admission gates the arch);
    /// `HIPFIRE_DN_SNAPSHOT_FLIP=0` restores the copy.
    pub dn_snapshot_flip: bool,
    /// Default-on gfx1201 Flash-Next dense GDN prefill scan (architecture and
    /// admission are gated by the runtime); `HIPFIRE_FN_GDN_DENSE_SCAN=0` opts out.
    pub fn_gdn_dense_scan: bool,
    /// Flash-Next lane-step PeaceMaker row kernels (`HIPFIRE_FN_LANE_PM_ROWS`):
    /// multi-column GEMVs that keep every column byte-identical to the 1-row
    /// incumbent while sharing activation loads across weight rows. Default on
    /// for gfx1151 (`0` is the kill switch); exact gfx1151, the lane step and
    /// the Qwen4 speculative verify only.
    pub fn_lane_pm_rows: bool,
    /// Stream-split MQ6G256V2 few-row verify kernels (`HIPFIRE_MQ6_ROWS_SPLIT`),
    /// 3..=8 rows, byte-identical to the incumbent rows kernels. Default on for
    /// exact gfx1201 and gfx1100; `0` restores the incumbent kernels.
    pub mq6_rows_split: bool,
    /// Exact gfx1201: the MQ4G256V2 gate+up verify kernel launches a reduced row-tile grid and loops (`HIPFIRE_GFX12_GATE_UP_TILE_LOOP`).
    pub gfx12_gate_up_tile_loop: bool,
    /// Split QSA decode attention (`HIPFIRE_QWEN4_QSA_DECODE_SPLITK`): split-K
    /// for F32 K/V and split-KV for q8 K/V. Parsed on unless the value is
    /// exactly `0` (kill switch for both). KLD-gated, not bit-exact against
    /// the per-head kernel; the F32 split-K is acted on only for exact gfx1151
    /// (`qwen4_qsa_decode_splitk_enabled`) and the q8 split-KV for exact
    /// gfx1151 and gfx1100 (`qwen4_qsa_decode_splitkv_q8_enabled`).
    pub qwen4_qsa_decode_splitk: bool,
    /// Developer QSA mapped/allocation extent checks; exactly `1` or `2`.
    pub qwen4_qsa_extent_check: bool,
    /// Exactly `2` also synchronizes immediately after each checked launch.
    pub qwen4_qsa_extent_sync: bool,
    /// Developer CPU-visible loaded-code snapshot; exactly `1`, default off.
    pub qwen4_qsa_code_check: bool,
    /// `HIPFIRE_MQ6_RESID_PORT_GFX1100`: the exact-gfx1100 MQ6 residual prefill
    /// GEMM runs the card-proven hipcc "pk twin" ports
    /// (`gemm_mq6g256v2_residual_gfx1100_port_{w8n256,w4n128}`) wherever hipcc
    /// picks MW8. Parsed on unless the value is exactly `0` (kill switch);
    /// acted on for exact gfx1100 only (`mq6_resid_port_gfx1100_enabled`).
    pub mq6_resid_port_gfx1100: bool,
    /// `HIPFIRE_QWEN4_QSA_HG4_Q8_LITE`: the q8 grouped prefill attention
    /// (`indexed_attention_attention_q8_batched_hg4`) runs its bit-identical
    /// low-LDS twin `..._hg4_lite`. Parsed on unless the value is exactly `0`
    /// (kill switch); acted on for exact gfx1100 only
    /// (`qwen4_qsa_hg4_q8_lite_enabled`).
    pub qwen4_qsa_hg4_q8_lite: bool,
    /// `HIPFIRE_QWEN4_HC_MIX_FILL80_GFX1100`: the HC `input_mix_down` projection
    /// (M = 320) runs the bit-identical 80 x 32 fill tile
    /// `gemm_wmma_lds_80_32_80_16_k64_p` instead of the 64 x 64 tile. Parsed on
    /// unless the value is exactly `0` (kill switch); acted on for exact gfx1100
    /// only (`qwen4_hc_mix_fill80_gfx1100_enabled`).
    pub qwen4_hc_mix_fill80_gfx1100: bool,
}

/// Built-in default of `experimental.graph.ar` for an architecture on a host
/// platform. On, except Windows exact gfx1100 (#655): there AR-forward graph
/// capture/replay silently produces wrong output (token soup on an RX 7900 XTX
/// until the replay is fixed), so the default is off. Linux gfx1100 and every
/// other architecture on every platform keep the default on.
pub fn graph_ar_default(arch: &str, windows: bool) -> bool {
    !(windows && arch == "gfx1100")
}

/// Resolve `experimental.graph.ar` from the process-config compatibility value
/// of `HIPFIRE_AR_GRAPH`. Absence (the built-in default is never carried in
/// the process config) selects [`graph_ar_default`]; an explicit value from the
/// user, the environment or a one-shot argument always wins, and only the exact
/// value `0` (a rendered `false`) turns capture off.
pub fn resolve_graph_ar(raw: Option<&str>, arch: &str, windows: bool) -> bool {
    match raw {
        None => graph_ar_default(arch, windows),
        Some(raw) => raw != "0",
    }
}

impl FeatureFlags {
    pub fn from_process_config(arch: &str, config: &ProcessConfig) -> Self {
        Self::from_lookup(arch, |name| config.legacy_value(name).ok_or(()))
    }

    pub fn from_active_config(arch: &str) -> Self {
        Self::from_process_config(arch, hipfire_config::active_or_local_process_config())
    }

    fn from_lookup(arch: &str, value: impl Fn(&str) -> std::result::Result<String, ()>) -> Self {
        Self::from_lookup_on_host(arch, cfg!(windows), value)
    }

    /// [`Self::from_lookup`] with the host platform passed in, so CPU tests can
    /// exercise the Windows-only defaults on any build host.
    fn from_lookup_on_host(
        arch: &str,
        windows: bool,
        value: impl Fn(&str) -> std::result::Result<String, ()>,
    ) -> Self {
        let parse_bool = |name: &str| -> Option<bool> {
            match value(name).ok().as_deref() {
                Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON") => Some(true),
                Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => {
                    Some(false)
                }
                _ => None,
            }
        };

        let parse_usize =
            |name: &str| -> Option<usize> { value(name).ok().and_then(|s| s.parse().ok()) };

        let parse_mb4 = |name: &str| -> Option<Mb4Mode> {
            match value(name).ok().as_deref() {
                Some("1") => Some(Mb4Mode::Pack1),
                Some("2") => Some(Mb4Mode::Pack2),
                Some("4") => Some(Mb4Mode::Pack4),
                _ => None,
            }
        };

        let qsa_extent_mode = value("HIPFIRE_QWEN4_QSA_EXTENT_CHECK").ok();

        let is_gfx906 = arch == "gfx906";
        // The bytewise-identical Halo hyper units (H5 GDN conv+Q/K norm, H4
        // HC fusion level 3, H3 HC up tile, H8a combine zero-init) default on
        // for exact gfx1151 only; gfx1201 takes its own subset below.
        let halo_hyper_default = arch == "gfx1151";
        // The exact units that do not need the F16 WMMA route (H8a combine
        // zero-init, wave-per-token router, fused PLE tail) default on for
        // exact gfx1151 and gfx1201.
        let exact_unit_default = matches!(arch, "gfx1151" | "gfx1201");
        // gfx1201 takes the H1 pairing and the retiled HC up+mix on its
        // default F16 WMMA HC read (levels 2/3 measured slower there).
        let is_gfx1201 = arch == "gfx1201";

        let mmq_screen_default: bool = false;
        let mmq_screen_threshold_default: f32 = if is_gfx906 { 0.50 } else { 0.10 };

        let gemv_rows_default: u32 = match arch {
            "gfx1100" | "gfx1101" | "gfx1102" => 1,
            "gfx1030" | "gfx1031" => 1,
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
            _ => 2,
        };

        let mut hipcc_extra_flags = value("HIPFIRE_HIPCC_EXTRA_FLAGS").unwrap_or_default();
        let mut append_hipcc_flag = |flag: &str| {
            if !hipcc_extra_flags.is_empty() {
                hipcc_extra_flags.push(' ');
            }
            hipcc_extra_flags.push_str(flag);
        };
        if arch == "gfx1100" {
            match value("HIPFIRE_GFX11_WEIGHT_LOAD_POLICY").ok().as_deref() {
                None | Some("") | Some("buffer") => {}
                Some("global") => append_hipcc_flag("-DHIPFIRE_GFX11_WEIGHT_GLOBAL_LOADS=1"),
                Some("flat-buffer") => {
                    append_hipcc_flag("-DHIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN=1")
                }
                Some(other) => {
                    eprintln!("unknown HIPFIRE_GFX11_WEIGHT_LOAD_POLICY={other:?}; using buffer")
                }
            }
        }
        if matches!(arch, "gfx1100" | "gfx1151") {
            let candidates = match value("HIPFIRE_GFX11_A4_CANDIDATES").ok().as_deref() {
                Some(candidate @ ("1" | "2" | "4" | "8")) => candidate.to_string(),
                None | Some("") => "2".to_string(),
                Some(other) => {
                    eprintln!(
                        "unknown HIPFIRE_GFX11_A4_CANDIDATES={other:?}; using default value 2"
                    );
                    "2".to_string()
                }
            };
            append_hipcc_flag(&format!("-DIU4_A4_CANDIDATES={candidates}"));
        }
        if arch == "gfx1201" {
            let policy_flag = match value("HIPFIRE_GFX12_WEIGHT_LOAD_POLICY").ok().as_deref() {
                None | Some("") | Some("rt") => None,
                Some("global") => Some("-DHIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS=1"),
                Some("ht") => Some("-DHIPFIRE_GFX12_WEIGHT_CPOL_AUX=18"),
                Some("nt-rt") => Some("-DHIPFIRE_GFX12_WEIGHT_CPOL_AUX=20"),
                Some("nt-ht") => Some("-DHIPFIRE_GFX12_WEIGHT_CPOL_AUX=22"),
                Some(other) => {
                    eprintln!("unknown HIPFIRE_GFX12_WEIGHT_LOAD_POLICY={other:?}; using rt");
                    None
                }
            };
            if let Some(flag) = policy_flag {
                append_hipcc_flag(flag);
            }
            // gfx1201 IU4 producers search {5,7} like gfx11 (`HIPFIRE_G12_A4C2`,
            // default on): the gfx1201 emit takes the one-pass producer-layout
            // path in block_i4_128_quant.hip, bit-identical to this flag's
            // reference search. `=0` restores RTN (d = amax/7).
            if parse_bool("HIPFIRE_G12_A4C2").unwrap_or(true) {
                append_hipcc_flag("-DIU4_A4_CANDIDATES=2");
            }
        }

        Self {
            arch: arch.to_string(),

            // GEMV tuning
            gemv_rows: value("HIPFIRE_GEMV_ROWS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .map(|r| match r {
                    1 | 2 | 4 | 8 => r,
                    _ => 1,
                }),
            gemv_dp4a_default_on: is_gfx906,
            gemv_dp4a: parse_bool("HIPFIRE_GEMV_DP4A"),
            gfx1151_e8_buffer: parse_bool("HIPFIRE_GFX1151_E8_BUFFER"),
            gfx11_mmq_x128: parse_bool("HIPFIRE_GFX11_MMQ_X128"),
            iu4_prefill: parse_bool("HIPFIRE_IU4_PREFILL"),
            g12_iu4_isa: parse_bool("HIPFIRE_G12_IU4_ISA").unwrap_or(true),
            pm_decode: parse_bool("HIPFIRE_PM_DECODE").unwrap_or(arch == "gfx1201"),
            a8_prefill: parse_bool("HIPFIRE_A8_PREFILL").unwrap_or(false),
            gfx11_a8_prefill: parse_bool("HIPFIRE_GFX11_A8_PREFILL").unwrap_or(false),
            a8_fused_prod: parse_bool("HIPFIRE_A8_FUSED_PROD").unwrap_or(true),
            qwen4_gdn_conv_qknorm: parse_bool("HIPFIRE_QWEN4_GDN_CONV_QKNORM")
                .unwrap_or(halo_hyper_default),
            qwen4_gdn_q8_inline: parse_bool("HIPFIRE_QWEN4_GDN_Q8_INLINE").unwrap_or(false),
            qwen4_gdn_i8_fuse: parse_bool("HIPFIRE_QWEN4_GDN_I8_FUSE").unwrap_or(is_gfx1201),
            qwen4_gdn_chain_fuse: parse_bool("HIPFIRE_QWEN4_GDN_CHAIN_FUSE").unwrap_or(is_gfx1201),
            qwen4_gdn_qkv_bf16: parse_bool("HIPFIRE_QWEN4_GDN_QKV_BF16").unwrap_or(false),
            qwen4_hc_fuse: value("HIPFIRE_QWEN4_HC_FUSE")
                .ok()
                .and_then(|v| v.trim().parse::<u8>().ok())
                .unwrap_or(if halo_hyper_default {
                    3
                } else if is_gfx1201 {
                    1
                } else {
                    0
                }),
            qwen4_hc_up_tile: parse_bool("HIPFIRE_QWEN4_HC_UP_TILE")
                .unwrap_or(halo_hyper_default || is_gfx1201),
            qwen4_hc_up_v2: parse_bool("HIPFIRE_QWEN4_HC_UP_V2").unwrap_or(is_gfx1201 || arch == "gfx1151"),
            qwen4_moe_combine_zinit: parse_bool("HIPFIRE_QWEN4_MOE_COMBINE_ZINIT")
                .unwrap_or(exact_unit_default),
            qwen4_hc_row_fold: parse_bool("HIPFIRE_QWEN4_HC_ROW_FOLD")
                .unwrap_or(halo_hyper_default || is_gfx1201),
            qwen4_router_fast: parse_bool("HIPFIRE_QWEN4_ROUTER_FAST")
                .unwrap_or(exact_unit_default),
            qwen4_gdn_lanes: parse_bool("HIPFIRE_QWEN4_GDN_LANES").unwrap_or(arch == "gfx1151"),
            qwen4_moe_down_r4: parse_bool("HIPFIRE_QWEN4_MOE_DOWN_R4").unwrap_or(arch == "gfx1151"),
            qwen4_shared_down_fuse: parse_bool("HIPFIRE_QWEN4_SHARED_DOWN_FUSE")
                .unwrap_or(arch == "gfx1151"),
            qwen4_decode_zba_x3: parse_bool("HIPFIRE_QWEN4_DECODE_ZBA_X3")
                .unwrap_or(arch == "gfx1151"),
            qwen4_decode_head_fuse: parse_bool("HIPFIRE_QWEN4_DECODE_HEAD_FUSE")
                .unwrap_or(arch == "gfx1151"),
            qwen4_ple_fuse: parse_bool("HIPFIRE_QWEN4_PLE_FUSE").unwrap_or(exact_unit_default),
            gfx11_iu4_gridspec: parse_bool("HIPFIRE_GFX11_IU4_GRIDSPEC").unwrap_or(true),
            gfx11_iu4_shape: parse_bool("HIPFIRE_GFX11_IU4_SHAPE").unwrap_or(true),
            gfx11_iu4_symfold: parse_bool("HIPFIRE_IU4_SYMFOLD").unwrap_or(true),
            gemv_prefetch: parse_bool("HIPFIRE_GEMV_PREFETCH"),
            gemv_prefetch_default_on: is_gfx906,
            gfx942_lds_gemv: parse_bool("HIPFIRE_GFX942_LDS_GEMV"),
            gfx942_lds_gemv_default_on: false,
            gemv_rows_default,

            // Quant/format toggles
            hfq3_dp4a: parse_bool("HIPFIRE_HFQ3_DP4A"),
            hfq3_mmq: parse_bool("HIPFIRE_HFQ3_MMQ"),
            hfq4_mmq_rdna2: parse_bool("HIPFIRE_HFQ4_MMQ_RDNA2"),
            fp8_wmma: value("HIPFIRE_FP8_WMMA").map_or(false, |v| v == "1"),
            dot2_gemv: value("HIPFIRE_DOT2_GEMV").map_or(false, |v| v == "1"),
            gcn5_wave64_hybrid: parse_bool("HIPFIRE_GCN5_WAVE64_HYBRID"),
            rdna3_hfq4_qkv_wave64: value("HIPFIRE_RDNA3_HFQ4_QKV_WAVE64").as_deref() == Ok("1"),
            rdna3_hfq4_qkvza_2wave: value("HIPFIRE_RDNA3_HFQ4_QKVZA_2WAVE").as_deref() == Ok("1"),
            rdna3_hfq4_qkvza_wavepack4: value("HIPFIRE_RDNA3_HFQ4_QKVZA_WAVEPACK4").as_deref()
                == Ok("1"),
            rdna3_hfq4_qkvza_ldsx8: value("HIPFIRE_RDNA3_HFQ4_QKVZA_LDSX8").as_deref() == Ok("1"),
            rdna3_hfq4_qkvza_reduce_chain: value("HIPFIRE_RDNA3_HFQ4_QKVZA_REDUCE_CHAIN")
                .as_deref()
                == Ok("1"),
            rdna3_hfq4_qkvza_hoist_x32: value("HIPFIRE_RDNA3_HFQ4_QKVZA_HOIST_X32").as_deref()
                == Ok("1"),
            rdna3_hfq4_qkvza_k2048: parse_bool("HIPFIRE_RDNA3_HFQ4_QKVZA_K2048")
                .unwrap_or(arch == "gfx1100"),
            rdna3_hfq4_residual_stage_x32: parse_bool("HIPFIRE_RDNA3_HFQ4_RESIDUAL_STAGE_X32")
                .unwrap_or(arch == "gfx1100"),
            rdna3_hfq4_residual_k2048: value("HIPFIRE_RDNA3_HFQ4_RESIDUAL_K2048").as_deref()
                == Ok("1"),
            rdna3_hfq4_sigmoid_tight_grid: value("HIPFIRE_RDNA3_HFQ4_SIGMOID_TIGHT_GRID")
                .as_deref()
                == Ok("1"),
            rdna3_hfq4_sigmoid_buffer: parse_bool("HIPFIRE_RDNA3_HFQ4_SIGMOID_BUFFER")
                .unwrap_or(arch == "gfx1100"),
            rdna3_hfq4_sigmoid_rows4: value("HIPFIRE_RDNA3_HFQ4_SIGMOID_ROWS4").as_deref()
                == Ok("1"),
            rdna3_hfq4_lm_head_k2048: value("HIPFIRE_RDNA3_HFQ4_LM_HEAD_K2048").as_deref()
                == Ok("1"),
            rdna3_hfq4_moe_gate_up_k2048: value("HIPFIRE_RDNA3_HFQ4_MOE_GATE_UP_K2048").as_deref()
                == Ok("1"),
            mmq_override: match value("HIPFIRE_MMQ").ok().as_deref() {
                Some("0") | Some("off") => Some(false),
                Some("1") | Some("on") => Some(true),
                _ => None,
            },
            packed_mq4_prefill: value("HIPFIRE_GFX1100_PACKED_MQ4_PREFILL").as_deref() == Ok("1"),
            mmq_min_batch: parse_usize("HIPFIRE_MMQ_MIN_BATCH"),
            fp16_disabled: value("HIPFIRE_FP16").map_or(false, |v| v == "0"),
            fp16_layer_min: parse_usize("HIPFIRE_FP16_LAYER_MIN"),
            fp16_layer_max: parse_usize("HIPFIRE_FP16_LAYER_MAX"),
            wo_mmq: value("HIPFIRE_WO_MMQ").ok().as_deref() == Some("1"),
            lm_head_wmma_disabled: value("HIPFIRE_LM_HEAD_WMMA").map_or(false, |v| v == "0"),
            lm_head_overwrite: value("HIPFIRE_LM_HEAD_OVERWRITE").as_deref() == Ok("1"),
            qwen4_mq6_x4_gfx1201: match value("HIPFIRE_QWEN4_MQ6_X4_GFX1201").as_deref() {
                Ok("1") => Some(true),
                Ok(_) => Some(false),
                Err(()) => None,
            },
            qwen4_mq6_x4_tile: match value("HIPFIRE_QWEN4_MQ6_X4_TILE").ok().as_deref() {
                None => None,
                Some("auto") => Some(Some([0, 0, 0])),
                Some("8x4x1") => Some(Some([8, 4, 1])),
                Some("8x4x2") => Some(Some([8, 4, 2])),
                Some("8x8x1") => Some(Some([8, 8, 1])),
                Some("8x8x2") => Some(Some([8, 8, 2])),
                Some("12x4x1") => Some(Some([12, 4, 1])),
                Some("12x4x2") => Some(Some([12, 4, 2])),
                Some("12x8x1") => Some(Some([12, 8, 1])),
                Some("12x8x2") => Some(Some([12, 8, 2])),
                Some("16x4x1") => Some(Some([16, 4, 1])),
                Some("16x4x2") => Some(Some([16, 4, 2])),
                Some("16x8x1") => Some(Some([16, 8, 1])),
                Some("16x8x2") => Some(Some([16, 8, 2])),
                Some(_) => Some(None),
            },
            qwen4_mq6_x4_regions: match value("HIPFIRE_QWEN4_MQ6_X4_REGIONS").as_deref() {
                Ok("1") => Some(true),
                Ok(_) => Some(false),
                Err(()) => None,
            },
            gfx12_mq6_fp8: parse_bool("HIPFIRE_GFX12_MQ6_FP8"),
            gfx12_mq6_fp8_arm: match value("HIPFIRE_GFX12_MQ6_FP8_ARM")
                .ok()
                .map(|v| v.to_ascii_lowercase())
                .as_deref()
            {
                Some("round") => crate::gemm_mq6_fp8::Mq6Fp8Arm::Round,
                _ => crate::gemm_mq6_fp8::Mq6Fp8Arm::Split,
            },
            gfx12_mq6_fp8_db: value("HIPFIRE_GFX12_MQ6_FP8_DB").as_deref() == Ok("1"),
            mq6_i8: parse_bool("HIPFIRE_MQ6_I8"),
            mq6_i8_pm: parse_bool("HIPFIRE_MQ6_I8_PM"),
            calib_force_bf16: value("HIPFIRE_CALIB_BF16").as_deref() == Ok("1"),

            // MMQ screening
            mmq_screen: value("HIPFIRE_MMQ_SCREEN")
                .ok()
                .map(|v| v == "1")
                .unwrap_or(mmq_screen_default),
            mmq_screen_threshold: value("HIPFIRE_MMQ_SCREEN_THRESHOLD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(mmq_screen_threshold_default),
            mmq_diag_quantize_only: value("HIPFIRE_MMQ_DIAG_QUANTIZE_ONLY").ok().as_deref()
                == Some("1"),

            // Kernel variant overrides
            lloyd_mb4: parse_mb4("HIPFIRE_LLOYD_MB4"),
            mq3_mb4: parse_mb4("HIPFIRE_MQ3_MB4"),
            mq3v25_a4: parse_bool("HIPFIRE_MQ3V25_A4"),
            mq3v25_a8: parse_bool("HIPFIRE_MQ3V25_A8").unwrap_or(false),
            hfq4g128_mmq: value("HIPFIRE_HFQ4G128_MMQ").as_deref() != Ok("0"),
            hfq3_mmq_layer_min: parse_usize("HIPFIRE_HFQ3_MMQ_LAYER_MIN"),
            hfq3_mmq_layer_max: parse_usize("HIPFIRE_HFQ3_MMQ_LAYER_MAX"),
            hfq4_mmq_gfx906_y64: value("HIPFIRE_HFQ4_MMQ_GFX906_Y64").map_or(false, |v| v == "1"),
            gate_up_variant: value("HIPFIRE_GATE_UP_VARIANT").ok(),
            gate_up_nosync: value("HIPFIRE_GATE_UP_NOSYNC").as_deref() == Ok("1"),
            hfq4g256_ldsstage_wmma: value("HIPFIRE_HFQ4G256_LDSSTAGE").as_deref() == Ok("1"),
            qkvza_split_tail: parse_bool("HIPFIRE_QKVZA_SPLIT_TAIL").unwrap_or(false),
            gfx942_gemv_v2: parse_bool("HIPFIRE_GFX942_GEMV_V2"),
            gfx942_gemv_v3: value("HIPFIRE_GFX942_GEMV_V3").map_or(false, |v| v == "1"),
            gfx942_rmsnorm_split: matches!(arch, "gfx940" | "gfx941" | "gfx942")
                && value("HIPFIRE_GFX942_RMSNORM_SPLIT").as_deref() != Ok("0"),
            rmsnorm_mq_tight_lds: value("HIPFIRE_RMSNORM_MQ_TIGHT_LDS").as_deref() == Ok("1"),
            rdna3_rmsnorm_wavegrid: value("HIPFIRE_RDNA3_RMSNORM_WAVEGRID").as_deref() == Ok("1"),
            rdna3_rmsnorm_split: value("HIPFIRE_RDNA3_RMSNORM_SPLIT").as_deref() == Ok("1"),
            rdna3_rmsnorm_vecsum: parse_bool("HIPFIRE_RDNA3_RMSNORM_VECSUM")
                .unwrap_or(arch == "gfx1100"),
            rdna3_rmsnorm_sign_lds: value("HIPFIRE_RDNA3_RMSNORM_SIGN_LDS").as_deref() == Ok("1"),
            rdna3_rmsnorm_sign_const: value("HIPFIRE_RDNA3_RMSNORM_SIGN_CONST").as_deref()
                == Ok("1"),
            gfx942_mfma_prefill: value("HIPFIRE_GFX942_MFMA_PREFILL").ok(),
            moe_grouped_i8: match value("HIPFIRE_MOE_GROUPED_I8").ok().as_deref() {
                Some("1") => Some(true),
                Some("0") => Some(false),
                _ => None,
            },
            moe_grouped_i8_k8: value("HIPFIRE_MOE_GROUPED_I8_K8").as_deref() == Ok("1"),
            moe_grouped_i8_k4: value("HIPFIRE_MOE_GROUPED_I8_K4").as_deref() == Ok("1"),
            moe_grouped_i8_k4_gfx12: value("HIPFIRE_MOE_GROUPED_I8_K4_GFX12").as_deref() == Ok("1"),
            moe_grouped_m2: value("HIPFIRE_MOE_GROUPED_M2").as_deref() == Ok("1"),
            moe_grouped_4w: value("HIPFIRE_MOE_GROUPED_4W").as_deref() == Ok("1"),
            moe_down_combine_vec4: value("HIPFIRE_MOE_DOWN_COMBINE_VEC4").as_deref() == Ok("1"),
            moe_hfq6_i8: value("HIPFIRE_MOE_HFQ6_I8").as_deref() == Ok("1"),
            moe_hfq6_v2: value("HIPFIRE_MOE_HFQ6_V2").as_deref() == Ok("1"),
            // MoE prefill (Ship 4.2)
            moe_grouped_gemm: match value("HIPFIRE_MOE_GROUPED_GEMM").ok().as_deref() {
                Some("0") | Some("off") => false,
                _ => true,
            },
            moe_paro_i8: parse_bool("HIPFIRE_MOE_PARO_I8"),
            moe_paro_i8_k8: parse_bool("HIPFIRE_MOE_PARO_I8_K8"),

            // Graph / capture / deterministic
            graph_forward: parse_bool("HIPFIRE_GRAPH"),
            graph_ar: resolve_graph_ar(value("HIPFIRE_AR_GRAPH").ok().as_deref(), arch, windows),
            graph_moe: value("HIPFIRE_GRAPH_MOE").ok().as_deref() != Some("0"),
            force_blob_path: value("HIPFIRE_BLOB_FORCE").ok().as_deref() == Some("1"),
            residual_ksplit_off: value("HIPFIRE_RESIDUAL_KSPLIT_OFF").ok().as_deref() == Some("1"),
            residual_ldsstage: parse_bool("HIPFIRE_RESIDUAL_LDSSTAGE").unwrap_or(arch == "gfx1100"),
            gate_up_ldsstage: parse_bool("HIPFIRE_GATEUP_LDSSTAGE").unwrap_or(arch == "gfx1100"),
            gfx1100_q8x_verify: parse_bool("HIPFIRE_GFX1100_Q8X_VERIFY").unwrap_or(arch == "gfx1100"),
            gfx1151_xlds: parse_bool("HIPFIRE_GFX1151_XLDS").unwrap_or(arch == "gfx1151"),
            mq4v2_gateup_k5120: parse_bool("HIPFIRE_MQ4V2_GATEUP_K5120").unwrap_or(true),
            gfx12_mq4v2_gateup_k5120: parse_bool("HIPFIRE_GFX12_MQ4V2_GATEUP_K5120")
                .unwrap_or(false),
            gfx12_mq4v2_fp8_gateup: parse_bool("HIPFIRE_GFX12_MQ4V2_FP8_GATEUP")
                .unwrap_or(arch == "gfx1201"),
            gfx12_mq4v2_fp8_resid: parse_bool("HIPFIRE_GFX12_MQ4V2_FP8_RESID")
                .unwrap_or(arch == "gfx1201"),
            gfx12_mq4v2_fp8_qkvza: parse_bool("HIPFIRE_GFX12_MQ4V2_FP8_QKVZA")
                .unwrap_or(arch == "gfx1201"),
            gfx12_mq4v2_fp8_qkv: parse_bool("HIPFIRE_GFX12_MQ4V2_FP8_QKV")
                .unwrap_or(arch == "gfx1201"),
            wmma_batch_tiles: parse_bool("HIPFIRE_WMMA_BATCH_TILES").unwrap_or(arch == "gfx1201"),
            cb_verify_chunk128: parse_bool("HIPFIRE_CB_VERIFY_CHUNK128").unwrap_or(arch == "gfx1201"),
            cb_verify_pm: parse_bool("HIPFIRE_CB_VERIFY_PM").unwrap_or(true),
            cb_verify_fp8: parse_bool("HIPFIRE_CB_VERIFY_FP8").unwrap_or(false),
            cb_verify_iu4: parse_bool("HIPFIRE_CB_VERIFY_IU4"),
            gfx12_mq4v2_fp8_v2: parse_bool("HIPFIRE_GFX12_MQ4V2_FP8_V2")
                .unwrap_or(arch == "gfx1201"),
            gfx12_gdn_pre_fused: parse_bool("HIPFIRE_GFX12_GDN_PRE_FUSED")
                .unwrap_or(arch == "gfx1201"),
            gfx12_gdn_chunk_scan: parse_bool("HIPFIRE_GFX12_GDN_CHUNK_SCAN")
                .unwrap_or(matches!(arch, "gfx1100" | "gfx1151" | "gfx1201")),
            gfx12_silu_quant_fused: parse_bool("HIPFIRE_GFX12_SILU_QUANT_FUSED")
                .unwrap_or(arch == "gfx1201"),
            gfx12_producer_quant_fused: parse_bool("HIPFIRE_GFX12_PRODUCER_QUANT_FUSED")
                .unwrap_or(arch == "gfx1201"),
            g12_norm: parse_bool("HIPFIRE_G12_NORM").unwrap_or(arch == "gfx1201"),
            g12_dec_norm: parse_bool("HIPFIRE_G12_DEC_NORM")
                .unwrap_or(matches!(arch, "gfx1201" | "gfx1151")),
            gfx1100_dec_norm: parse_bool("HIPFIRE_GFX1100_DEC_NORM").unwrap_or(arch == "gfx1100"),
            gfx11_producer_quant_fused: parse_bool("HIPFIRE_GFX11_PRODUCER_QUANT_FUSED")
                .unwrap_or(matches!(arch, "gfx1100" | "gfx1151")),
            gfx12_fp8_stream: parse_bool("HIPFIRE_GFX12_FP8_STREAM")
                .unwrap_or(arch == "gfx1201"),
            gfx12_fa2_prefill: parse_bool("HIPFIRE_GFX12_FA2_PREFILL").unwrap_or(arch == "gfx1201"),
            fa_prefill_pm: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_FA_PREFILL_PM").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_FA_PREFILL_PM is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::FaPrefillPm,
            ),
            fa_prefill_pm_fp8_kv: crate::prefill_route_capabilities::enabled_fp8_kv(
                value("HIPFIRE_FA_PREFILL_PM").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_FA_PREFILL_PM is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::FaPrefillPm,
            ),
            moe_iu4_wide_tiles: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_MOE_IU4_WIDE_TILES").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_MOE_IU4_WIDE_TILES is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::MoeIu4WideTiles,
            ),
            fa_prefill_pm_qprep: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_FA_PREFILL_PM_QPREP").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_FA_PREFILL_PM_QPREP is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::FaPrefillPmQprep,
            ),
            fa_prefill_pm_qprep_fp8_kv: crate::prefill_route_capabilities::enabled_fp8_kv(
                value("HIPFIRE_FA_PREFILL_PM_QPREP").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_FA_PREFILL_PM_QPREP is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::FaPrefillPmQprep,
            ),
            moe_rotate_i4_pm: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_MOE_ROTATE_I4_PM").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_MOE_ROTATE_I4_PM is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::MoeRotateI4Pm,
            ),
            prefill_iu4_producer_fused: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_PREFILL_IU4_PRODUCER_FUSED").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_PREFILL_IU4_PRODUCER_FUSED is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::PrefillIu4ProducerFused,
            ),
            prefill_iu4_c8_pm: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_PREFILL_IU4_C8_PM").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_PREFILL_IU4_C8_PM is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::PrefillIu4C8Pm,
            ),
            gdn_chunk_prep_pm: crate::prefill_route_capabilities::enabled(
                value("HIPFIRE_GDN_CHUNK_PREP_PM").ok().map(|raw| {
                    hipfire_config::parse_typed_bool(&raw).expect(
                        "HIPFIRE_GDN_CHUNK_PREP_PM is not a boolean: expected true, false, 1 or 0",
                    )
                }),
                arch,
                crate::prefill_route_capabilities::PrefillRoute::GdnChunkPrepPm,
            ),
            gfx12_fa_packet: parse_bool("HIPFIRE_GFX12_FA_PACKET").unwrap_or(arch == "gfx1201"),
            attn_qresident: parse_bool("HIPFIRE_ATTN_QRESIDENT").unwrap_or(arch == "gfx1201"),
            attn_qresident_v2: parse_bool("HIPFIRE_ATTN_QRESIDENT_V2").unwrap_or(arch == "gfx1201"),
            verify_attn: parse_bool("HIPFIRE_VERIFY_ATTN")
                .unwrap_or(matches!(arch, "gfx1201" | "gfx1100" | "gfx1151")),
            gfx1100_fa2_split_verify: parse_bool("HIPFIRE_GFX1100_FA2_SPLIT_VERIFY")
                .unwrap_or(arch == "gfx1100"),
            gfx1100_wmma_verify_attn: parse_bool("HIPFIRE_GFX1100_WMMA_VERIFY_ATTN")
                .unwrap_or(arch == "gfx1100"),
            gfx12_fa_prep_fused: parse_bool("HIPFIRE_GFX12_FA_PREP_FUSED")
                .unwrap_or(arch == "gfx1201"),
            gfx12_fa_prep_fp8q: parse_bool("HIPFIRE_GFX12_FA_PREP_FP8Q")
                .unwrap_or(arch == "gfx1201"),
            gfx11_q8_fa2_wide: parse_bool("HIPFIRE_GFX11_Q8_FA2_WIDE")
                .unwrap_or(matches!(arch, "gfx1100" | "gfx1151")),
            gfx11_fa2_prefill: parse_bool("HIPFIRE_GFX11_FA2_PREFILL")
                .unwrap_or(matches!(arch, "gfx1100" | "gfx1151")),
            gemm_dump: value("HIPFIRE_GEMM_DUMP").ok().as_deref() == Some("1"),
            deterministic: value("HIPFIRE_DETERMINISTIC").ok().as_deref() == Some("1"),
            q8_batched_legacy: value("HIPFIRE_Q8_BATCHED_LEGACY").as_deref() == Ok("1"),
            flash_attn_ck_lib: value("HIPFIRE_FLASH_ATTN_CK_LIB")
                .ok()
                .filter(|path| !path.is_empty()),
            flash_attn_ck_workspace_bytes: value("HIPFIRE_FLASH_ATTN_CK_WORKSPACE_BYTES")
                .ok()
                .and_then(|bytes| bytes.parse().ok())
                .unwrap_or(0),
            gemma4_q8_fused_prefill: parse_bool("HIPFIRE_GEMMA4_Q8_FUSED_PREFILL").unwrap_or(false),
            gemma4_batched_embedding_prefill: parse_bool(
                "HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL",
            )
            .unwrap_or(matches!(arch, "gfx1100" | "gfx1201")),
            gemma4_ple_batched_prefill: parse_bool("HIPFIRE_GEMMA4_PLE_BATCHED_PREFILL")
                .unwrap_or(false),
            gemma4_ple_branch_batched_prefill: parse_bool(
                "HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL",
            )
            .unwrap_or(matches!(arch, "gfx1100" | "gfx1201")),
            gemma4_ple_activation_fused_prefill: parse_bool(
                "HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL",
            )
            .unwrap_or(matches!(arch, "gfx1100" | "gfx1201")),
            deepseek4_q8_wmma_off: value("HIPFIRE_DEEPSEEK4_Q8_WMMA").as_deref() == Ok("0"),
            deepseek4_q8_4w_off: value("HIPFIRE_DEEPSEEK4_Q8_4W").as_deref() == Ok("0"),
            rope_interleaved_legacy: value("HIPFIRE_ROPE_INTERLEAVED_LEGACY").ok().as_deref()
                == Some("1"),
            wo_wmma_variant: value("HIPFIRE_WO_WMMA_VARIANT").ok(),

            // rocBLAS
            rocblas_all_archs: value("HIPFIRE_ROCBLAS_ALL_ARCHS").ok().as_deref() == Some("1"),
            rocblas_off: value("HIPFIRE_ROCBLAS_OFF").ok().as_deref() == Some("1"),
            rocblas_min_batch: parse_usize("HIPFIRE_ROCBLAS_MIN_BATCH"),
            attn_tile_size: parse_usize("HIPFIRE_ATTN_TILE_SIZE"),
            slots_attn_crossover: parse_usize("HIPFIRE_SLOTS_ATTN_CROSSOVER"),
            slots_decode_graph: value("HIPFIRE_SLOTS_DECODE_GRAPH").ok().as_deref() == Some("1"),
            slot_trace: value("HIPFIRE_SLOT_TRACE").ok().as_deref() == Some("1"),

            // Kernels.rs
            lloyd_force_baseline: value("HIPFIRE_LLOYD_FORCE_BASELINE").ok().as_deref()
                == Some("1"),
            rdna2_variant: value("HIPFIRE_RDNA2_VARIANT")
                .ok()
                .and_then(|s| s.parse::<u32>().ok()),

            // Compiler.rs
            hipcc_extra_flags,

            // Interpreter Phase 2a
            force_unfused: value("HIPFIRE_FORCE_UNFUSED")
                .map(|v| v == "1")
                .unwrap_or(false),

            // Speculative decode (DFlash/DDTree). DEFAULT OFF (chain): ddtree
            // tree-verify loses to linear chain on every drafter measured
            // (DeltaNet + non-DeltaNet qwen3-8b/Bielik) because the DFlash
            // drafter emits independent per-position marginals — a tree branch
            // has no joint to exploit. Opt in with the CLI `--ddtree` flag (sets
            // HIPFIRE_DFLASH_TREE=1).
            dflash_tree: value("HIPFIRE_DFLASH_TREE").as_deref() == Ok("1"),
            ddtree_budget: parse_usize("HIPFIRE_DDTREE_BUDGET").filter(|&b| b > 0),
            ddtree_topk: parse_usize("HIPFIRE_DDTREE_TOPK").filter(|&k| k >= 1),
            // D8: ddtree_verify_naive removed; HIPFIRE_DDTREE_VERIFY env is no-op.
            ddtree_tree_la: value("HIPFIRE_DDTREE_TREE_LA").as_deref() != Ok("0"),
            dflash_fast_sample: value("HIPFIRE_DFLASH_FAST_SAMPLE").as_deref() != Ok("0"),
            ddtree_logw_cutoff: value("HIPFIRE_DDTREE_LOGW_CUTOFF")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .filter(|&x| x > 0.0),
            dflash_q8_lmhead_wmma: match value("HIPFIRE_DFLASH_Q8_LMHEAD_WMMA") {
                Ok(v) => {
                    let v = v.trim().to_ascii_lowercase();
                    !(v == "0" || v == "false" || v == "off" || v == "no")
                }
                Err(_) => true,
            },

            // QKV bias fold — default ON, opt out with HIPFIRE_FUSE_QKV_BIAS=0.
            fuse_qkv_bias: parse_bool("HIPFIRE_FUSE_QKV_BIAS").unwrap_or(true),
            fuse_qkv_bias_debug: value("HIPFIRE_FUSE_QKV_BIAS_DEBUG").as_deref() == Ok("1"),

            // DFlash launch-fusion kill switches: `_OFF=1` disables, all no-ops.
            dn_snapshot_bulk_off: value("HIPFIRE_DN_SNAPSHOT_BULK_OFF").ok().as_deref()
                == Some("1"),
            hidden_scatter_fuse_off: value("HIPFIRE_HIDDEN_SCATTER_FUSE_OFF").ok().as_deref()
                == Some("1"),
            mq_f16_projection_off: value("HIPFIRE_MQ_F16_PROJECTION_OFF").ok().as_deref()
                == Some("1"),
            mq_f16_residual_off: value("HIPFIRE_MQ_F16_RESIDUAL_OFF").ok().as_deref() == Some("1"),
            gdn_pre_fuse_off: value("HIPFIRE_GDN_PRE_FUSE_OFF").ok().as_deref() == Some("1"),
            fa_batch_fuse_off: value("HIPFIRE_FA_BATCH_FUSE_OFF").ok().as_deref() == Some("1"),
            draft_collapse_off: value("HIPFIRE_DRAFT_COLLAPSE_OFF").ok().as_deref() == Some("1"),
            draft_shared_qkv_rotation: parse_bool("HIPFIRE_DRAFT_SHARED_QKV_ROTATION")
                .unwrap_or(arch == "gfx1100"),
            draft_shared_ffn_rotation: parse_bool("HIPFIRE_DRAFT_SHARED_FFN_ROTATION")
                .unwrap_or(arch == "gfx1100"),
            draft_shared_ctx_kv_rotation: parse_bool("HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION")
                .unwrap_or(arch == "gfx1100"),
            ddtree_topk_direct_off: value("HIPFIRE_DDTREE_TOPK_DIRECT_OFF").ok().as_deref()
                == Some("1"),
            mq_prologue_fuse_off: value("HIPFIRE_MQ_PROLOGUE_FUSE_OFF").ok().as_deref()
                == Some("1"),
            gdn_replay_ml_off: value("HIPFIRE_GDN_REPLAY_ML_OFF").ok().as_deref() == Some("1"),
            select_regrid_off: value("HIPFIRE_SELECT_REGRID_OFF").ok().as_deref() == Some("1"),
            dn_snapshot_flip: value("HIPFIRE_DN_SNAPSHOT_FLIP").ok().as_deref() != Some("0"),
            fn_gdn_dense_scan: value("HIPFIRE_FN_GDN_DENSE_SCAN").ok().as_deref() != Some("0"),
            fn_lane_pm_rows: parse_bool("HIPFIRE_FN_LANE_PM_ROWS").unwrap_or(arch == "gfx1151"),
            mq6_rows_split: parse_bool("HIPFIRE_MQ6_ROWS_SPLIT")
                .unwrap_or(matches!(arch, "gfx1201" | "gfx1100")),
            gfx12_gate_up_tile_loop: parse_bool("HIPFIRE_GFX12_GATE_UP_TILE_LOOP")
                .unwrap_or(arch == "gfx1201"),
            qwen4_qsa_decode_splitk: value("HIPFIRE_QWEN4_QSA_DECODE_SPLITK").ok().as_deref()
                != Some("0"),
            qwen4_qsa_extent_check: matches!(qsa_extent_mode.as_deref(), Some("1" | "2")),
            qwen4_qsa_extent_sync: qsa_extent_mode.as_deref() == Some("2"),
            qwen4_qsa_code_check: value("HIPFIRE_QWEN4_QSA_CODE_CHECK").ok().as_deref()
                == Some("1"),
            mq6_resid_port_gfx1100: value("HIPFIRE_MQ6_RESID_PORT_GFX1100")
                .ok()
                .as_deref()
                != Some("0"),
            qwen4_qsa_hg4_q8_lite: value("HIPFIRE_QWEN4_QSA_HG4_Q8_LITE").ok().as_deref()
                != Some("0"),
            qwen4_hc_mix_fill80_gfx1100: value("HIPFIRE_QWEN4_HC_MIX_FILL80_GFX1100")
                .ok()
                .as_deref()
                != Some("0"),
        }
    }

    // ── Methods replacing free functions ─────────────────────────────

    pub fn gemv_dp4a_enabled(&self) -> bool {
        self.gemv_dp4a.unwrap_or(self.gemv_dp4a_default_on)
    }

    /// Resolved ddtree meta-verifier expansion threshold: `−X` for a user-set
    /// positive `X` (stop expanding when cumulative logw drops below it), else
    /// `f32::NEG_INFINITY` (expand to the full budget).
    pub fn ddtree_logw_cutoff_value(&self) -> f32 {
        match self.ddtree_logw_cutoff {
            Some(x) => -x,
            None => f32::NEG_INFINITY,
        }
    }

    pub fn gemv_prefetch_enabled(&self) -> bool {
        self.gemv_prefetch.unwrap_or(self.gemv_prefetch_default_on)
    }

    pub fn gfx942_lds_gemv_enabled(&self) -> bool {
        self.gfx942_lds_gemv
            .unwrap_or(self.gfx942_lds_gemv_default_on)
    }

    /// Resolved gfx11 MMQ X128 route: env override, else default on
    /// gfx1100/gfx1151, per-32 elsewhere.
    pub fn gfx11_mmq_x128_enabled(&self) -> bool {
        self.gfx11_mmq_x128
            .unwrap_or(matches!(self.arch.as_str(), "gfx1100" | "gfx1151"))
    }

    /// Resolved iu4-direct MMQ route: on unless explicitly disabled, and only
    /// on exact gfx1100/gfx1151 (K16 kernel) or gfx1201 (K32 kernel).
    /// `=0`/other arches keep the incumbent.
    pub fn iu4_prefill_enabled(&self) -> bool {
        self.iu4_prefill.unwrap_or(true)
            && matches!(self.arch.as_str(), "gfx1100" | "gfx1151" | "gfx1201")
    }

    /// Symmetric qt49 W3A4 prefill for an artifact whose loader validated the
    /// `mq3v2.symmetric` zero contract (`symmetric`). Shares the W4A4 opt-out
    /// (`kernel.iu4_prefill`). Default on exact gfx1201; exact gfx1100 runs
    /// only on `HIPFIRE_MQ3V25_A4=1`; every other arch keeps the F16 route.
    pub fn mq3v25_a4_enabled(&self, symmetric: bool) -> bool {
        symmetric
            && self.iu4_prefill_enabled()
            && match self.arch.as_str() {
                "gfx1201" => self.mq3v25_a4.unwrap_or(true),
                "gfx1100" => self.mq3v25_a4.unwrap_or(false),
                _ => false,
            }
    }

    /// Symmetric qt49 W3A8 prefill: explicit developer opt-in on exact
    /// gfx1201 only. Wins over [`Self::mq3v25_a4_enabled`].
    pub fn mq3v25_a8_enabled(&self, symmetric: bool) -> bool {
        self.mq3v25_a8 && symmetric && self.arch == "gfx1201"
    }

    /// A8 projection route, only when explicitly selected: `HIPFIRE_A8_PREFILL`
    /// on exact gfx1201 (replacing FP8), `kernel.gfx11_a8_prefill` on exact
    /// gfx1100/gfx1151 (replacing IU4).
    pub fn a8_prefill_enabled(&self) -> bool {
        (self.a8_prefill && self.arch == "gfx1201")
            || (self.gfx11_a8_prefill && matches!(self.arch.as_str(), "gfx1100" | "gfx1151"))
    }

    /// H5: the chunked GDN convolution also writes the normalized Q/K. Exact
    /// gfx1151 only; every other arch (gfx1100 included) keeps the plain
    /// convolution and the separate Q/K-norm launch.
    pub fn qwen4_gdn_conv_qknorm_enabled(&self) -> bool {
        self.qwen4_gdn_conv_qknorm && self.arch == "gfx1151"
    }

    /// H6: the chunked GDN recurrence decodes/requantizes a Q8 state inline.
    /// Exact gfx1151 only.
    pub fn qwen4_gdn_q8_inline_enabled(&self) -> bool {
        self.qwen4_gdn_q8_inline && self.arch == "gfx1151"
    }

    /// Dense GDN gate + out-projection A8 producer fusion. Exact gfx1201 only.
    pub fn qwen4_gdn_i8_fuse_enabled(&self) -> bool {
        self.qwen4_gdn_i8_fuse && self.arch == "gfx1201"
    }

    /// Dense-GDN chain: the load-batched producer. Exact gfx1201 only.
    pub fn qwen4_gdn_chain_fuse_enabled(&self) -> bool {
        self.qwen4_gdn_chain_fuse && self.arch == "gfx1201"
    }

    /// Dense-GDN chain with the BF16 qkv projection (opt-in; needs the chain fusion on).
    pub fn qwen4_gdn_qkv_bf16_enabled(&self) -> bool {
        self.qwen4_gdn_qkv_bf16 && self.qwen4_gdn_chain_fuse_enabled()
    }

    /// Flash-Next lane step: the gfx1151 PeaceMaker multi-column row kernels
    /// (`HIPFIRE_FN_LANE_PM_ROWS`, gfx1151: on unless `0`). Exact gfx1151 only; the
    /// runtime additionally requires the lane-step scope
    /// ([`crate::Gpu::lane_rows_pm_enabled`]).
    pub fn fn_lane_pm_rows_enabled(&self) -> bool {
        self.fn_lane_pm_rows && self.arch == "gfx1151"
    }

    /// MQ6 stream-split few-row verify kernels (`HIPFIRE_MQ6_ROWS_SPLIT`, exact
    /// gfx1201 / gfx1100: on unless `0`).
    pub fn mq6_rows_split_enabled(&self) -> bool {
        self.mq6_rows_split && matches!(self.arch.as_str(), "gfx1201" | "gfx1100")
    }

    /// gfx1201 MQ4G256V2 gate+up verify: the one-tile WMMA kernel loops over row
    /// tiles on a grid no larger than the resident wave slots
    /// (`HIPFIRE_GFX12_GATE_UP_TILE_LOOP`, gfx1201: on unless `0`). Exact gfx1201 only.
    pub fn gfx12_gate_up_tile_loop_enabled(&self) -> bool {
        self.gfx12_gate_up_tile_loop && self.arch == "gfx1201"
    }

    /// Split QSA decode attention (`HIPFIRE_QWEN4_QSA_DECODE_SPLITK`, default
    /// on; `=0` is the kill switch for both the F32 split-K and the q8
    /// split-KV routes). This gate covers the F32 split-K route: exact gfx1151
    /// only; the runtime additionally requires F32 K/V, head_dim 256, an
    /// admitted decode call (1..=`few_row_limit` rows) and an eager (unrecorded,
    /// uncaptured) launch. The q8 split-KV route uses
    /// [`Self::qwen4_qsa_decode_splitkv_q8_enabled`].
    pub fn qwen4_qsa_decode_splitk_enabled(&self) -> bool {
        self.qwen4_qsa_decode_splitk && self.arch == "gfx1151"
    }

    /// MQ6 residual hipcc "pk twin" ports
    /// (`gemm_mq6g256v2_residual_gfx1100_port_{w8n256,w4n128}`): default on,
    /// exact gfx1100 only (the only architecture they were carded on);
    /// `HIPFIRE_MQ6_RESID_PORT_GFX1100=0` restores the PM twin
    /// (`HIPFIRE_MQ6_RESID_PM_GFX1100`) or hipcc MW8. Bit-identical output.
    pub fn mq6_resid_port_gfx1100_enabled(&self) -> bool {
        self.mq6_resid_port_gfx1100 && self.arch == "gfx1100"
    }

    /// q8 grouped prefill attention low-LDS twin
    /// (`indexed_attention_attention_q8_batched_hg4_lite`): default on, exact
    /// gfx1100 only (the only architecture its win was measured on);
    /// `HIPFIRE_QWEN4_QSA_HG4_Q8_LITE=0` restores
    /// `indexed_attention_attention_q8_batched_hg4`. Bit-identical output.
    pub fn qwen4_qsa_hg4_q8_lite_enabled(&self) -> bool {
        self.qwen4_qsa_hg4_q8_lite && self.arch == "gfx1100"
    }

    /// HC `input_mix_down` fill tile (`gemm_wmma_lds_80_32_80_16_k64_p`, 80 x 32
    /// workgroup of two 80 x 16 waves, 192 waves instead of 120): default on,
    /// exact gfx1100 only (the only architecture it was measured on);
    /// `HIPFIRE_QWEN4_HC_MIX_FILL80_GFX1100=0` restores
    /// `gemm_wmma_lds_64_64_32_64_k64_p`. Bit-identical output.
    pub fn qwen4_hc_mix_fill80_gfx1100_enabled(&self) -> bool {
        self.qwen4_hc_mix_fill80_gfx1100 && self.arch == "gfx1100"
    }

    /// Q8 split-KV QSA decode attention: the same
    /// `HIPFIRE_QWEN4_QSA_DECODE_SPLITK` switch (default on; `=0` kills it),
    /// acted on for exact gfx1151 and exact gfx1100. The runtime additionally
    /// requires q8 K/V, head_dim 256, 24 query heads over 2 KV heads, an
    /// admitted decode call (1..=`few_row_limit` rows) and an eager
    /// (unrecorded, uncaptured) launch.
    pub fn qwen4_qsa_decode_splitkv_q8_enabled(&self) -> bool {
        self.qwen4_qsa_decode_splitk && matches!(self.arch.as_str(), "gfx1151" | "gfx1100")
    }

    /// H4: Qwen4 HC read/write fusion level (`HIPFIRE_QWEN4_HC_FUSE`, 0 = off):
    /// 1 = read side (the HC read also produces its paired write's gates),
    /// 2 = + the attention output projection writes the HC streams,
    /// 3 = + the shared-down fold writes them. Exact gfx1151 / gfx1201;
    /// each stage still needs the F16 WMMA prefill route it hooks into.
    pub fn qwen4_hc_fuse_level(&self) -> u8 {
        if matches!(self.arch.as_str(), "gfx1151" | "gfx1201") {
            self.qwen4_hc_fuse.min(3)
        } else {
            0
        }
    }

    /// H3: the retiled operand-swapped HC up+mix read tail
    /// (`HIPFIRE_QWEN4_HC_UP_TILE`). Exact gfx1151 / gfx1201.
    pub fn qwen4_hc_up_tile_enabled(&self) -> bool {
        self.qwen4_hc_up_tile && matches!(self.arch.as_str(), "gfx1151" | "gfx1201")
    }

    /// The persistent W-stationary HC up+mix entry (`HIPFIRE_QWEN4_HC_UP_V2`).
    /// Exact gfx1201 and gfx1151.
    pub fn qwen4_hc_up_v2_enabled(&self) -> bool {
        self.qwen4_hc_up_v2 && matches!(self.arch.as_str(), "gfx1201" | "gfx1151")
    }

    /// H8a: the MoE combine zero-initializes its own output instead of
    /// accumulating into a cleared tensor (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`).
    /// Exact gfx1151 / gfx1201.
    pub fn qwen4_moe_combine_zinit_enabled(&self) -> bool {
        self.qwen4_moe_combine_zinit && matches!(self.arch.as_str(), "gfx1151" | "gfx1201")
    }

    /// Row fold of the MoE tail (`HIPFIRE_QWEN4_HC_ROW_FOLD`): exact gfx1151,
    /// on top of the `HIPFIRE_QWEN4_HC_FUSE` >= 3 shared-down route and the
    /// zero-initialized combine.  gfx1201 has its own accessors
    /// ([`Self::qwen4_hc_row_fold_moe_gfx1201_enabled`],
    /// [`Self::qwen4_hc_row_fold_mixer_gfx1201_enabled`]); this one stays false there.
    pub fn qwen4_hc_row_fold_enabled(&self) -> bool {
        self.qwen4_hc_row_fold
            && self.arch == "gfx1151"
            && self.qwen4_hc_fuse_level() >= 3
            && self.qwen4_moe_combine_zinit_enabled()
    }

    /// gfx1201 mixer-tail row fold (`HIPFIRE_QWEN4_HC_ROW_FOLD`, default on, `=0`
    /// off): a paired HC write and the next HC read's norm/gate in one row
    /// kernel.  Needs the read-side pairing (`HIPFIRE_QWEN4_HC_FUSE` >= 1) that
    /// hands the write its gates; the out-projection epilogue levels (>= 2) are
    /// NOT required and stay off by default on gfx1201.
    pub fn qwen4_hc_row_fold_mixer_gfx1201_enabled(&self) -> bool {
        self.qwen4_hc_row_fold && self.arch == "gfx1201" && self.qwen4_hc_fuse_level() >= 1
    }

    /// gfx1201 MoE-tail row fold (`HIPFIRE_QWEN4_HC_ROW_FOLD`, default on, `=0`
    /// off): the mixer fold's conditions plus the zero-initialized combine
    /// (`HIPFIRE_QWEN4_MOE_COMBINE_ZINIT`).
    pub fn qwen4_hc_row_fold_moe_gfx1201_enabled(&self) -> bool {
        self.qwen4_hc_row_fold_mixer_gfx1201_enabled() && self.qwen4_moe_combine_zinit_enabled()
    }

    /// Wave-per-token E512/top-10 router for batched prefill
    /// (`HIPFIRE_QWEN4_ROUTER_FAST`). Exact gfx1151 / gfx1201.
    /// Multi-lane fn-lanes GDN decode core (`HIPFIRE_QWEN4_GDN_LANES`). Exact
    /// gfx1151 only.
    pub fn qwen4_gdn_lanes_enabled(&self) -> bool {
        self.qwen4_gdn_lanes && self.arch == "gfx1151"
    }

    pub fn qwen4_moe_down_r4_enabled(&self) -> bool {
        self.qwen4_moe_down_r4 && self.arch == "gfx1151"
    }

    pub fn qwen4_shared_down_fuse_enabled(&self) -> bool {
        self.qwen4_shared_down_fuse && self.arch == "gfx1151"
    }

    pub fn qwen4_router_fast_enabled(&self) -> bool {
        self.qwen4_router_fast && matches!(self.arch.as_str(), "gfx1151" | "gfx1201")
    }

    pub fn qwen4_decode_zba_x3_enabled(&self) -> bool {
        self.qwen4_decode_zba_x3 && self.arch == "gfx1151"
    }

    /// Exact single-token decode MoE head fusion (`HIPFIRE_QWEN4_DECODE_HEAD_FUSE`):
    /// the BF16 router/selector rows and the Q8_0 shared gate/up pair in one
    /// launch. Exact gfx1151 only.
    pub fn qwen4_decode_head_fuse_enabled(&self) -> bool {
        self.qwen4_decode_head_fuse && self.arch == "gfx1151"
    }

    /// The fused PLE gate/norm/conv/stream-add tail on BF16 HC streams
    /// (`HIPFIRE_QWEN4_PLE_FUSE`). Exact gfx1151 / gfx1201.
    pub fn qwen4_ple_fuse_enabled(&self) -> bool {
        self.qwen4_ple_fuse && matches!(self.arch.as_str(), "gfx1151" | "gfx1201")
    }
    /// Producer-emitted IU4 sidecar route on gfx1100/gfx1151 + IU4 opt-in.
    /// When live (and eager + batch/K admission), RMSNorm/FWHT and
    /// SwiGLU/FWHT emit `block_i4_128` in-register; otherwise consumers
    /// keep standalone `quantize_int4_mmq_ds128`.
    pub fn iu4_producer_sidecar_enabled(&self) -> bool {
        self.iu4_prefill.unwrap_or(true) && matches!(self.arch.as_str(), "gfx1100" | "gfx1151")
    }
    /// True only on exact gfx1201 with the opt-in set. The producer emits
    /// the shared `block_i4_128` recipe, so output is bit-identical to the
    /// standalone `quantize_int4_mmq_ds128` chain on the same f32 row.
    pub fn gfx12_silu_quant_fused_enabled(&self) -> bool {
        self.gfx12_silu_quant_fused && self.arch == "gfx1201"
    }
    /// True only on exact gfx1201 with the opt-in set. The `_gfx12`
    /// RMSNorm/rotate/gated-norm producers emit the shared `block_i4_128`
    /// recipe, so output is bit-identical to the standalone
    /// `quantize_int4_mmq_ds128` chain on the same f32 row.
    pub fn gfx12_producer_quant_fused_enabled(&self) -> bool {
        self.gfx12_producer_quant_fused && self.arch == "gfx1201"
    }
    /// True only on exact gfx1201 with `HIPFIRE_G12_NORM` on: the
    /// slices-2/4 producers launch their bit-identical `_v2` twins.
    pub fn g12_norm_enabled(&self) -> bool {
        self.g12_norm && self.arch == "gfx1201"
    }
    /// True only on exact gfx1201 or exact gfx1151 with `HIPFIRE_G12_DEC_NORM`
    /// on: the decode RMSNorm+FWHT, final RMSNorm and RoPE launch their
    /// bit-identical multi-workgroup twins.
    pub fn g12_dec_norm_enabled(&self) -> bool {
        self.g12_dec_norm && matches!(self.arch.as_str(), "gfx1201" | "gfx1151")
    }
    /// True only on exact gfx1100 with `HIPFIRE_GFX1100_DEC_NORM` on: the
    /// decode RMSNorm+FWHT and final RMSNorm launch their bit-identical
    /// multi-workgroup twins.
    pub fn gfx1100_dec_norm_enabled(&self) -> bool {
        self.gfx1100_dec_norm && self.arch == "gfx1100"
    }
    /// The decode RMSNorm+FWHT and final RMSNorm run as multi-workgroup
    /// grids: exact gfx1201 or gfx1151 (`g12_dec_norm`) or exact gfx1100
    /// (`gfx1100_dec_norm`).
    pub fn dec_norm_grids_enabled(&self) -> bool {
        self.g12_dec_norm_enabled() || self.gfx1100_dec_norm_enabled()
    }
    pub fn mq4v2_gateup_k5120_enabled(&self, gate_m: usize, up_m: usize, k: usize) -> bool {
        ((self.arch == "gfx1100" && self.mq4v2_gateup_k5120)
            || (self.arch == "gfx1201" && self.gfx12_mq4v2_gateup_k5120))
            && (gate_m, up_m, k) == (17408, 17408, 5120)
    }
    /// True only on gfx1100/gfx1151 with the opt-in set. The `_gfx11`
    /// sigmoid/gated-norm producers emit the shared `block_i4_128` recipe,
    /// so output is bit-identical to the standalone
    /// `quantize_int4_mmq_ds128` chain on the same f32 row.
    pub fn gfx11_producer_quant_fused_enabled(&self) -> bool {
        self.gfx11_producer_quant_fused && matches!(self.arch.as_str(), "gfx1100" | "gfx1151")
    }
    /// True only on exact gfx1201 with the opt-in set. The `_mq4v2_fp8_gfx12`
    /// RMSNorm/rotate producers emit byte-identical standalone-pack outputs,
    /// so the fused route is exact on finite rows.
    pub fn gfx12_fp8_stream_enabled(&self) -> bool {
        self.gfx12_fp8_stream && self.arch == "gfx1201"
    }

    pub fn hfq3_mmq_layer_gate_pass(&self) -> bool {
        let lo = self.hfq3_mmq_layer_min;
        let hi = self.hfq3_mmq_layer_max;
        if lo.is_none() && hi.is_none() {
            return true;
        }
        let layer = super::dispatch::MMQ_CURRENT_LAYER.load(std::sync::atomic::Ordering::Relaxed);
        if let Some(lo) = lo {
            if layer < lo {
                return false;
            }
        }
        if let Some(hi) = hi {
            if layer > hi {
                return false;
            }
        }
        true
    }

    pub fn fp16_disabled_for_current_layer(&self) -> bool {
        if self.fp16_disabled {
            return true;
        }
        let lo = self.fp16_layer_min;
        let hi = self.fp16_layer_max;
        if lo.is_none() && hi.is_none() {
            return false;
        }
        let layer = super::dispatch::MMQ_CURRENT_LAYER.load(std::sync::atomic::Ordering::Relaxed);
        let above_min = lo.map(|m| layer >= m).unwrap_or(true);
        let below_max = hi.map(|m| layer <= m).unwrap_or(true);
        above_min && below_max
    }

    pub fn hfq4_mmq_gfx906_y64_enabled(&self) -> bool {
        self.hfq4_mmq_gfx906_y64
    }

    /// Test-only constructor: uses defaults for the given arch.
    /// Provides deterministic FeatureFlags for unit tests regardless of the
    /// developer's env-var configuration.
    #[doc(hidden)]
    pub fn for_test(arch: &str) -> Self {
        let is_gfx906 = arch == "gfx906";

        let gemv_rows_default: u32 = match arch {
            "gfx1100" | "gfx1101" | "gfx1102" => 1,
            "gfx1030" | "gfx1031" => 1,
            "gfx906" | "gfx908" | "gfx940" | "gfx941" | "gfx942" => 1,
            _ => 2,
        };

        Self {
            arch: arch.to_string(),
            gemv_rows: None,
            gemv_dp4a_default_on: is_gfx906,
            gemv_dp4a: None,
            gfx1151_e8_buffer: None,
            gfx11_mmq_x128: None,
            // Deterministic unit-test baseline: the iu4 route stays off here
            // even though the process default is on.
            iu4_prefill: Some(false),
            g12_iu4_isa: false,
            pm_decode: false,
            a8_prefill: false,
            gfx11_a8_prefill: false,
            a8_fused_prod: false,
            qwen4_gdn_conv_qknorm: false,
            qwen4_gdn_q8_inline: false,
            qwen4_gdn_i8_fuse: false,
            qwen4_gdn_chain_fuse: false,
            qwen4_gdn_qkv_bf16: false,
            qwen4_hc_fuse: 0,
            qwen4_hc_up_tile: false,
            qwen4_hc_up_v2: false,
            qwen4_moe_combine_zinit: false,
            qwen4_hc_row_fold: false,
            qwen4_router_fast: false,
            qwen4_gdn_lanes: false,
            qwen4_moe_down_r4: false,
            qwen4_shared_down_fuse: false,
            qwen4_decode_zba_x3: false,
            qwen4_decode_head_fuse: false,
            qwen4_ple_fuse: false,
            gfx11_iu4_gridspec: false,
            gfx11_iu4_shape: false,
            gfx11_iu4_symfold: false,
            gemv_prefetch: None,
            gemv_prefetch_default_on: is_gfx906,
            gfx942_lds_gemv: None,
            gfx942_lds_gemv_default_on: false,
            gemv_rows_default,
            hfq3_dp4a: None,
            hfq3_mmq: None,
            hfq4_mmq_rdna2: None,
            fp8_wmma: false,
            dot2_gemv: false,
            gcn5_wave64_hybrid: None,
            rdna3_hfq4_qkv_wave64: false,
            rdna3_hfq4_qkvza_2wave: false,
            rdna3_hfq4_qkvza_wavepack4: false,
            rdna3_hfq4_qkvza_ldsx8: false,
            rdna3_hfq4_qkvza_reduce_chain: false,
            rdna3_hfq4_qkvza_hoist_x32: false,
            rdna3_hfq4_qkvza_k2048: false,
            rdna3_hfq4_residual_stage_x32: false,
            rdna3_hfq4_residual_k2048: false,
            rdna3_hfq4_sigmoid_tight_grid: false,
            rdna3_hfq4_sigmoid_buffer: false,
            rdna3_hfq4_sigmoid_rows4: false,
            rdna3_hfq4_lm_head_k2048: false,
            rdna3_hfq4_moe_gate_up_k2048: false,
            mmq_override: None,
            packed_mq4_prefill: false,
            mmq_min_batch: None,
            fp16_disabled: false,
            fp16_layer_min: None,
            fp16_layer_max: None,
            wo_mmq: false,
            lm_head_wmma_disabled: false,
            lm_head_overwrite: false,
            qwen4_mq6_x4_gfx1201: None,
            qwen4_mq6_x4_tile: None,
            qwen4_mq6_x4_regions: None,
            gfx12_mq6_fp8: None,
            gfx12_mq6_fp8_arm: crate::gemm_mq6_fp8::Mq6Fp8Arm::Split,
            gfx12_mq6_fp8_db: false,
            mq6_i8: None,
            mq6_i8_pm: None,
            calib_force_bf16: false,
            mmq_screen: false,
            mmq_screen_threshold: if is_gfx906 { 0.50 } else { 0.10 },
            mmq_diag_quantize_only: false,
            lloyd_mb4: None,
            mq3_mb4: None,
            mq3v25_a4: None,
            mq3v25_a8: false,
            hfq4g128_mmq: true,
            hfq3_mmq_layer_min: None,
            hfq3_mmq_layer_max: None,
            hfq4_mmq_gfx906_y64: false,
            gate_up_variant: None,
            gate_up_nosync: false,
            hfq4g256_ldsstage_wmma: false,
            qkvza_split_tail: false,
            gfx942_gemv_v2: None,
            gfx942_gemv_v3: false,
            gfx942_rmsnorm_split: matches!(arch, "gfx940" | "gfx941" | "gfx942"),
            rmsnorm_mq_tight_lds: false,
            rdna3_rmsnorm_wavegrid: false,
            rdna3_rmsnorm_split: false,
            rdna3_rmsnorm_vecsum: false,
            rdna3_rmsnorm_sign_lds: false,
            rdna3_rmsnorm_sign_const: false,
            gfx942_mfma_prefill: None,
            moe_grouped_i8: None,
            moe_grouped_i8_k8: false,
            moe_grouped_i8_k4: false,
            moe_grouped_i8_k4_gfx12: false,
            moe_grouped_m2: false,
            moe_grouped_4w: false,
            moe_down_combine_vec4: false,
            moe_hfq6_i8: false,
            moe_hfq6_v2: false,
            moe_grouped_gemm: true,
            moe_paro_i8: None,
            moe_paro_i8_k8: None,
            graph_forward: None,
            graph_ar: true,
            graph_moe: true,
            force_blob_path: false,
            residual_ksplit_off: false,
            gfx12_gdn_pre_fused: false,
            gfx12_gdn_chunk_scan: false,
            gfx12_silu_quant_fused: false,
            gfx12_producer_quant_fused: false,
            g12_norm: false,
            g12_dec_norm: false,
            gfx1100_dec_norm: false,
            gfx11_producer_quant_fused: false,
            gfx12_fp8_stream: false,
            residual_ldsstage: false,
            gate_up_ldsstage: false,
            gfx1100_q8x_verify: false,
            gfx1151_xlds: false,
            mq4v2_gateup_k5120: false,
            gfx12_mq4v2_gateup_k5120: false,
            gfx12_mq4v2_fp8_gateup: false,
            gfx12_mq4v2_fp8_resid: false,
            gfx12_mq4v2_fp8_qkvza: false,
            gfx12_mq4v2_fp8_qkv: false,
            wmma_batch_tiles: false,
            cb_verify_chunk128: false,
            cb_verify_pm: true,
            cb_verify_fp8: false,
            cb_verify_iu4: None,
            gfx12_mq4v2_fp8_v2: false,
            gfx12_fa2_prefill: false,
            fa_prefill_pm: false,
            fa_prefill_pm_fp8_kv: false,
            moe_iu4_wide_tiles: false,
            fa_prefill_pm_qprep: false,
            fa_prefill_pm_qprep_fp8_kv: false,
            moe_rotate_i4_pm: false,
            prefill_iu4_producer_fused: false,
            prefill_iu4_c8_pm: false,
            gdn_chunk_prep_pm: false,
            gfx12_fa_packet: false,
            attn_qresident: false,
            attn_qresident_v2: false,
            verify_attn: false,
            gfx1100_fa2_split_verify: false,
            gfx1100_wmma_verify_attn: false,
            gfx12_fa_prep_fused: false,
            gfx12_fa_prep_fp8q: false,
            gfx11_q8_fa2_wide: false,
            gfx11_fa2_prefill: false,
            gemm_dump: false,
            deterministic: false,
            q8_batched_legacy: false,
            flash_attn_ck_lib: None,
            flash_attn_ck_workspace_bytes: 0,
            gemma4_q8_fused_prefill: false,
            gemma4_batched_embedding_prefill: false,
            gemma4_ple_batched_prefill: false,
            gemma4_ple_branch_batched_prefill: false,
            gemma4_ple_activation_fused_prefill: false,
            deepseek4_q8_wmma_off: false,
            deepseek4_q8_4w_off: false,
            rope_interleaved_legacy: false,
            wo_wmma_variant: None,
            rocblas_all_archs: false,
            rocblas_off: false,
            rocblas_min_batch: None,
            attn_tile_size: None,
            slots_attn_crossover: None,
            slots_decode_graph: false,
            slot_trace: false,
            lloyd_force_baseline: false,
            rdna2_variant: None,
            hipcc_extra_flags: String::new(),
            force_unfused: false,
            dflash_tree: true,
            ddtree_budget: None,
            ddtree_topk: None,
            // D8: ddtree_verify_naive removed.
            ddtree_tree_la: true,
            dflash_fast_sample: true,
            ddtree_logw_cutoff: None,
            dflash_q8_lmhead_wmma: true,
            fuse_qkv_bias: true,
            fuse_qkv_bias_debug: false,
            dn_snapshot_bulk_off: false,
            hidden_scatter_fuse_off: false,
            mq_f16_projection_off: false,
            mq_f16_residual_off: false,
            gdn_pre_fuse_off: false,
            fa_batch_fuse_off: false,
            draft_collapse_off: false,
            draft_shared_qkv_rotation: false,
            draft_shared_ffn_rotation: false,
            draft_shared_ctx_kv_rotation: false,
            ddtree_topk_direct_off: false,
            mq_prologue_fuse_off: false,
            gdn_replay_ml_off: false,
            select_regrid_off: false,
            dn_snapshot_flip: false,
            fn_gdn_dense_scan: false,
            fn_lane_pm_rows: false,
            mq6_rows_split: false,
            gfx12_gate_up_tile_loop: false,
            qwen4_qsa_decode_splitk: false,
            qwen4_qsa_extent_check: false,
            qwen4_qsa_extent_sync: false,
            qwen4_qsa_code_check: false,
            mq6_resid_port_gfx1100: false,
            qwen4_qsa_hg4_q8_lite: false,
            qwen4_hc_mix_fill80_gfx1100: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::{resolve, ConfigLayer, ConfigSource, NamedLayer, ProcessConfig};

    #[test]
    fn qsa_extent_check_parses_only_exact_developer_modes() {
        for (value, check, sync) in [
            (None, false, false),
            (Some("1"), true, false),
            (Some("2"), true, true),
            (Some("0"), false, false),
            (Some("3"), false, false),
            (Some("true"), false, false),
            (Some("01"), false, false),
            (Some(" 2"), false, false),
            (Some(""), false, false),
        ] {
            let flags = FeatureFlags::from_lookup("gfx1151", |name| {
                if name == "HIPFIRE_QWEN4_QSA_EXTENT_CHECK" {
                    value.map(str::to_owned).ok_or(())
                } else {
                    Err(())
                }
            });
            assert_eq!(flags.qwen4_qsa_extent_check, check, "{value:?}");
            assert_eq!(flags.qwen4_qsa_extent_sync, sync, "{value:?}");
        }
    }

    #[test]
    fn qsa_code_check_parses_only_exact_developer_mode() {
        for (value, enabled) in [
            (None, false),
            (Some("1"), true),
            (Some("2"), false),
            (Some("0"), false),
            (Some("true"), false),
            (Some("01"), false),
            (Some(" 1"), false),
            (Some("1 "), false),
            (Some(""), false),
        ] {
            let flags = FeatureFlags::from_lookup("gfx1151", |name| {
                if name == "HIPFIRE_QWEN4_QSA_CODE_CHECK" {
                    value.map(str::to_owned).ok_or(())
                } else {
                    Err(())
                }
            });
            assert_eq!(flags.qwen4_qsa_code_check, enabled, "{value:?}");
        }
    }

    const GRAPH_AR_ARCHES: [&str; 9] = [
        "gfx906", "gfx1030", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        "gfx942",
    ];

    /// #655: the built-in `experimental.graph.ar` default is off for Windows
    /// exact gfx1100 only; Linux gfx1100 and every other arch/platform stay on.
    #[test]
    fn graph_ar_default_is_off_only_on_windows_exact_gfx1100() {
        for arch in GRAPH_AR_ARCHES {
            for windows in [false, true] {
                assert_eq!(
                    graph_ar_default(arch, windows),
                    !(windows && arch == "gfx1100"),
                    "{arch} windows={windows}"
                );
            }
        }
        // Exact match: a different gfx11 part or a suffixed spelling is not gfx1100.
        assert!(graph_ar_default("gfx1101", true));
        assert!(graph_ar_default("gfx1100:xnack-", true));
        assert!(graph_ar_default("", true));
    }

    /// An explicit value always wins over the Windows gfx1100 default, in both
    /// directions and on every platform; only the exact value `0` is off.
    #[test]
    fn graph_ar_explicit_value_overrides_the_windows_gfx1100_default() {
        for arch in GRAPH_AR_ARCHES {
            for windows in [false, true] {
                assert!(
                    !resolve_graph_ar(Some("0"), arch, windows),
                    "{arch} windows={windows}: 0"
                );
                for raw in ["1", "true", "on", ""] {
                    assert!(
                        resolve_graph_ar(Some(raw), arch, windows),
                        "{arch} windows={windows}: {raw:?}"
                    );
                }
                assert_eq!(
                    resolve_graph_ar(None, arch, windows),
                    graph_ar_default(arch, windows),
                    "{arch} windows={windows}: unset"
                );
            }
        }
    }

    /// The full `FeatureFlags` lookup path honours the platform matrix and the
    /// `HIPFIRE_AR_GRAPH` override, simulating Windows on any build host.
    #[test]
    fn graph_ar_flag_platform_matrix_through_feature_flags() {
        for arch in GRAPH_AR_ARCHES {
            for windows in [false, true] {
                let flags = |raw: Option<&str>| {
                    FeatureFlags::from_lookup_on_host(arch, windows, |name| match raw {
                        Some(v) if name == "HIPFIRE_AR_GRAPH" => Ok(v.into()),
                        _ => Err(()),
                    })
                };
                let default_on = !(windows && arch == "gfx1100");
                assert_eq!(
                    flags(None).graph_ar,
                    default_on,
                    "{arch} windows={windows}: unset"
                );
                assert!(!flags(Some("0")).graph_ar, "{arch} windows={windows}: 0");
                assert!(flags(Some("1")).graph_ar, "{arch} windows={windows}: 1");
                // The sibling graph switches are not touched by the platform.
                assert!(flags(None).graph_moe, "{arch} windows={windows}: graph_moe");
                assert_eq!(
                    flags(None).graph_forward,
                    None,
                    "{arch} windows={windows}: graph_forward"
                );
            }
        }
    }

    /// Through the typed process config: the built-in default is carried as
    /// absence (so the arch/platform default applies), and an explicit
    /// `experimental.graph.ar` — by TOML key or legacy key — wins on Windows gfx1100.
    #[test]
    fn graph_ar_process_config_default_is_absence_and_explicit_wins() {
        let defaults = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        assert_eq!(defaults.legacy_value("HIPFIRE_AR_GRAPH"), None);
        let from = |arch: &str, windows: bool, config: &ProcessConfig| {
            FeatureFlags::from_lookup_on_host(arch, windows, |name| {
                config.legacy_value(name).ok_or(())
            })
            .graph_ar
        };
        assert!(!from("gfx1100", true, &defaults));
        assert!(from("gfx1100", false, &defaults));
        assert!(from("gfx1201", true, &defaults));
        // The host build agrees with the simulated platform.
        assert_eq!(
            FeatureFlags::from_process_config("gfx1100", &defaults).graph_ar,
            !cfg!(windows)
        );
        for key in ["experimental.graph.ar", "graph_ar"] {
            for (value, enabled) in [("true", true), ("false", false)] {
                let mut layer = ConfigLayer::default();
                layer.set_cli(key, value).unwrap();
                let resolved = resolve([NamedLayer {
                    source: ConfigSource::GlobalUser {
                        path: "config.toml".into(),
                    },
                    layer,
                }])
                .unwrap();
                let config = ProcessConfig::from_resolved(&resolved).unwrap();
                let config: ProcessConfig =
                    serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
                assert_eq!(
                    config.legacy_value("HIPFIRE_AR_GRAPH"),
                    Some(if enabled { "1" } else { "0" }.to_string())
                );
                for arch in GRAPH_AR_ARCHES {
                    for windows in [false, true] {
                        assert_eq!(
                            from(arch, windows, &config),
                            enabled,
                            "{key}={value} {arch} windows={windows}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn gfx12_mq4v2_gateup_k5120_opt_in_roundtrip_and_shape() {
        let defaults = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        assert!(!FeatureFlags::from_process_config("gfx1201", &defaults)
            .mq4v2_gateup_k5120_enabled(17408, 17408, 5120));
        for key in [
            "kernel.gfx12_mq4v2_gateup_k5120",
            "gfx12_mq4v2_gateup_k5120",
        ] {
            for (value, enabled) in [("true", true), ("false", false)] {
                let mut layer = ConfigLayer::default();
                layer.set_cli(key, value).unwrap();
                let resolved = resolve([NamedLayer {
                    source: ConfigSource::GlobalUser {
                        path: "config.toml".into(),
                    },
                    layer,
                }])
                .unwrap();
                let config = ProcessConfig::from_resolved(&resolved).unwrap();
                let config: ProcessConfig =
                    serde_json::from_str(&serde_json::to_string(&config).unwrap()).unwrap();
                assert_eq!(
                    config.legacy_value("HIPFIRE_GFX12_MQ4V2_GATEUP_K5120"),
                    Some(if enabled { "1" } else { "0" }.to_string())
                );
                for arch in [
                    "gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201", "gfx906",
                ] {
                    let flags = FeatureFlags::from_process_config(arch, &config);
                    assert_eq!(
                        flags.mq4v2_gateup_k5120_enabled(17408, 17408, 5120),
                        arch == "gfx1100" || (arch == "gfx1201" && enabled)
                    );
                    for shape in [
                        (17408, 17408, 2816),
                        (17407, 17408, 5120),
                        (17408, 17407, 5120),
                    ] {
                        assert!(!flags.mq4v2_gateup_k5120_enabled(shape.0, shape.1, shape.2));
                    }
                }
            }
        }
    }

    #[test]
    fn dn_snapshot_flip_defaults_on_and_zero_opts_out() {
        assert!(FeatureFlags::from_lookup("gfx1201", |_| Err(())).dn_snapshot_flip);
        for (value, expected) in [("0", false), ("1", true)] {
            let flags = FeatureFlags::from_lookup("gfx1201", |name| {
                if name == "HIPFIRE_DN_SNAPSHOT_FLIP" {
                    Ok(value.into())
                } else {
                    Err(())
                }
            });
            assert_eq!(flags.dn_snapshot_flip, expected);
        }
    }

    #[test]
    fn fn_gdn_dense_scan_defaults_on_and_zero_opts_out() {
        assert!(FeatureFlags::from_lookup("gfx1201", |_| Err(())).fn_gdn_dense_scan);
        for (value, expected) in [("0", false), ("1", true)] {
            let flags = FeatureFlags::from_lookup("gfx1201", |name| {
                if name == "HIPFIRE_FN_GDN_DENSE_SCAN" {
                    Ok(value.into())
                } else {
                    Err(())
                }
            });
            assert_eq!(flags.fn_gdn_dense_scan, expected);
        }
    }

    #[test]
    fn gfx12_gate_up_tile_loop_defaults_on_exact_gfx1201_and_zero_kills() {
        for arch in ["gfx906", "gfx1100", "gfx1151", "gfx1200", "gfx1201"] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_GFX12_GATE_UP_TILE_LOOP" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            assert_eq!(
                flags(None).gfx12_gate_up_tile_loop_enabled(),
                arch == "gfx1201",
                "{arch}: unset"
            );
            assert!(
                !flags(Some("0")).gfx12_gate_up_tile_loop_enabled(),
                "{arch}: 0"
            );
            assert_eq!(
                flags(Some("1")).gfx12_gate_up_tile_loop_enabled(),
                arch == "gfx1201",
                "{arch}: 1"
            );
        }
        assert!(!FeatureFlags::for_test("gfx1201").gfx12_gate_up_tile_loop_enabled());
    }

    /// The lane-rows PM kernels default on for exact gfx1151 only: `0` is the
    /// kill switch, and no other arch admits them whatever the variable says.
    #[test]
    fn fn_lane_pm_rows_defaults_on_exact_gfx1151_and_zero_kills() {
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1152", "gfx1201",
        ] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_FN_LANE_PM_ROWS" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            assert_eq!(
                flags(None).fn_lane_pm_rows_enabled(),
                arch == "gfx1151",
                "{arch}: unset"
            );
            assert!(!flags(Some("0")).fn_lane_pm_rows_enabled(), "{arch}: 0");
            assert_eq!(
                flags(Some("1")).fn_lane_pm_rows_enabled(),
                arch == "gfx1151",
                "{arch}: 1"
            );
        }
        assert!(!FeatureFlags::for_test("gfx1151").fn_lane_pm_rows_enabled());
    }

    /// The MQ6 split rows kernels default on for exact gfx1201 / gfx1100: `0`
    /// is the kill switch and no other arch admits them.
    #[test]
    fn mq6_rows_split_defaults_on_gfx1201_gfx1100_and_zero_kills() {
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1152", "gfx1201",
        ] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_MQ6_ROWS_SPLIT" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            let on = matches!(arch, "gfx1201" | "gfx1100");
            assert_eq!(flags(None).mq6_rows_split_enabled(), on, "{arch}: unset");
            assert!(!flags(Some("0")).mq6_rows_split_enabled(), "{arch}: 0");
            assert_eq!(flags(Some("1")).mq6_rows_split_enabled(), on, "{arch}: 1");
        }
        assert!(!FeatureFlags::for_test("gfx1201").mq6_rows_split_enabled());
    }

    /// The QSA split decode knob is default on and only the exact value `0`
    /// turns it off; the F32 split-K arch gate admits gfx1151 alone and the
    /// q8 split-KV arch gate admits gfx1151 and gfx1100.
    #[test]
    fn qwen4_qsa_decode_splitk_defaults_on_and_zero_opts_out() {
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1152", "gfx1201",
        ] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_QWEN4_QSA_DECODE_SPLITK" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            let gfx1151 = arch == "gfx1151";
            let q8_arch = arch == "gfx1151" || arch == "gfx1100";
            assert!(flags(None).qwen4_qsa_decode_splitk, "{arch}: unset");
            assert_eq!(
                flags(None).qwen4_qsa_decode_splitk_enabled(),
                gfx1151,
                "{arch}: unset enabled"
            );
            assert_eq!(
                flags(None).qwen4_qsa_decode_splitkv_q8_enabled(),
                q8_arch,
                "{arch}: unset q8 enabled"
            );
            assert!(!flags(Some("0")).qwen4_qsa_decode_splitk, "{arch}: 0");
            assert!(
                !flags(Some("0")).qwen4_qsa_decode_splitk_enabled(),
                "{arch}: 0 enabled"
            );
            assert!(
                !flags(Some("0")).qwen4_qsa_decode_splitkv_q8_enabled(),
                "{arch}: 0 q8 enabled"
            );
            for raw in ["1", "", "true", "TRUE", "on", "yes", "2", " 0", "0 ", "00"] {
                assert!(flags(Some(raw)).qwen4_qsa_decode_splitk, "{arch}: {raw:?}");
                assert_eq!(
                    flags(Some(raw)).qwen4_qsa_decode_splitk_enabled(),
                    gfx1151,
                    "{arch}: {raw:?} enabled"
                );
                assert_eq!(
                    flags(Some(raw)).qwen4_qsa_decode_splitkv_q8_enabled(),
                    q8_arch,
                    "{arch}: {raw:?} q8 enabled"
                );
            }
        }
        // for_test zeroes every default-on flag (dn_snapshot_flip, fn_gdn_dense_scan, ...).
        assert!(!FeatureFlags::for_test("gfx1151").qwen4_qsa_decode_splitk_enabled());
        assert!(!FeatureFlags::for_test("gfx1151").qwen4_qsa_decode_splitkv_q8_enabled());
        assert!(!FeatureFlags::for_test("gfx1100").qwen4_qsa_decode_splitkv_q8_enabled());
    }

    /// The MQ6 residual ports are default on for exact gfx1100 only; exactly `0`
    /// is the kill switch and every other value keeps them on.
    #[test]
    fn mq6_resid_port_gfx1100_defaults_on_exact_gfx1100_and_zero_opts_out() {
        for arch in [
            "gfx906", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1201",
        ] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_MQ6_RESID_PORT_GFX1100" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            let exact = arch == "gfx1100";
            assert!(flags(None).mq6_resid_port_gfx1100, "{arch}: unset");
            assert_eq!(
                flags(None).mq6_resid_port_gfx1100_enabled(),
                exact,
                "{arch}"
            );
            assert!(!flags(Some("0")).mq6_resid_port_gfx1100, "{arch}: 0");
            assert!(
                !flags(Some("0")).mq6_resid_port_gfx1100_enabled(),
                "{arch}: 0"
            );
            for raw in ["1", "", "true", "on", "2", " 0", "0 ", "00"] {
                assert!(flags(Some(raw)).mq6_resid_port_gfx1100, "{arch}: {raw:?}");
                assert_eq!(
                    flags(Some(raw)).mq6_resid_port_gfx1100_enabled(),
                    exact,
                    "{arch}: {raw:?}"
                );
            }
        }
        // for_test zeroes every default-on flag.
        assert!(!FeatureFlags::for_test("gfx1100").mq6_resid_port_gfx1100_enabled());
    }

    /// The q8 hg4 lite twin is default on for exact gfx1100 only; exactly `0`
    /// is the kill switch and every other value keeps it on.
    #[test]
    fn qwen4_qsa_hg4_q8_lite_defaults_on_exact_gfx1100_and_zero_opts_out() {
        for arch in [
            "gfx906", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1201",
        ] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_QWEN4_QSA_HG4_Q8_LITE" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            let exact = arch == "gfx1100";
            assert!(flags(None).qwen4_qsa_hg4_q8_lite, "{arch}: unset");
            assert_eq!(flags(None).qwen4_qsa_hg4_q8_lite_enabled(), exact, "{arch}");
            assert!(!flags(Some("0")).qwen4_qsa_hg4_q8_lite, "{arch}: 0");
            assert!(
                !flags(Some("0")).qwen4_qsa_hg4_q8_lite_enabled(),
                "{arch}: 0"
            );
            for raw in ["1", "", "true", "on", "2", " 0", "0 ", "00"] {
                assert!(flags(Some(raw)).qwen4_qsa_hg4_q8_lite, "{arch}: {raw:?}");
                assert_eq!(
                    flags(Some(raw)).qwen4_qsa_hg4_q8_lite_enabled(),
                    exact,
                    "{arch}: {raw:?}"
                );
            }
        }
        // for_test zeroes every default-on flag.
        assert!(!FeatureFlags::for_test("gfx1100").qwen4_qsa_hg4_q8_lite_enabled());
    }

    /// The HC mix fill tile is default on for exact gfx1100 only; exactly `0`
    /// is the kill switch and every other value keeps it on.
    #[test]
    fn qwen4_hc_mix_fill80_gfx1100_defaults_on_exact_gfx1100_and_zero_opts_out() {
        for arch in [
            "gfx906", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1201",
        ] {
            let flags = |raw: Option<&str>| {
                FeatureFlags::from_lookup(arch, |name| match raw {
                    Some(v) if name == "HIPFIRE_QWEN4_HC_MIX_FILL80_GFX1100" => Ok(v.into()),
                    _ => Err(()),
                })
            };
            let exact = arch == "gfx1100";
            assert!(flags(None).qwen4_hc_mix_fill80_gfx1100, "{arch}: unset");
            assert_eq!(
                flags(None).qwen4_hc_mix_fill80_gfx1100_enabled(),
                exact,
                "{arch}"
            );
            assert!(!flags(Some("0")).qwen4_hc_mix_fill80_gfx1100, "{arch}: 0");
            assert!(
                !flags(Some("0")).qwen4_hc_mix_fill80_gfx1100_enabled(),
                "{arch}: 0"
            );
            for raw in ["1", "", "true", "on", "2", " 0", "0 ", "00"] {
                assert!(
                    flags(Some(raw)).qwen4_hc_mix_fill80_gfx1100,
                    "{arch}: {raw:?}"
                );
                assert_eq!(
                    flags(Some(raw)).qwen4_hc_mix_fill80_gfx1100_enabled(),
                    exact,
                    "{arch}: {raw:?}"
                );
            }
        }
        // for_test zeroes every default-on flag.
        assert!(!FeatureFlags::for_test("gfx1100").qwen4_hc_mix_fill80_gfx1100_enabled());
    }

    #[test]
    fn mq4v2_gateup_k5120_is_default_on_exact_arch_and_shape() {
        for arch in ["gfx1100", "gfx1101", "gfx1151", "gfx1201", "gfx906"] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(
                unset.mq4v2_gateup_k5120_enabled(17408, 17408, 5120),
                arch == "gfx1100"
            );
            let off = FeatureFlags::from_lookup(arch, |name| {
                if name == "HIPFIRE_MQ4V2_GATEUP_K5120" {
                    Ok("0".into())
                } else {
                    Err(())
                }
            });
            assert!(!off.mq4v2_gateup_k5120_enabled(17408, 17408, 5120));
            let on = FeatureFlags::from_lookup(arch, |name| {
                if name == "HIPFIRE_MQ4V2_GATEUP_K5120" {
                    Ok("1".into())
                } else {
                    Err(())
                }
            });
            assert_eq!(
                on.mq4v2_gateup_k5120_enabled(17408, 17408, 5120),
                arch == "gfx1100"
            );
            for (gate_m, up_m, k) in [
                (17408, 17408, 2816),
                (17407, 17408, 5120),
                (17408, 17409, 5120),
            ] {
                assert!(!on.mq4v2_gateup_k5120_enabled(gate_m, up_m, k));
            }
        }
    }

    #[test]
    fn a8_prefill_is_opt_in_and_scoped_per_arch() {
        let defaults = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        for arch in [
            "gfx906", "gfx908", "gfx90a", "gfx942", "gfx1010", "gfx1030", "gfx1100", "gfx1101",
            "gfx1102", "gfx1151", "gfx1200", "gfx1201",
        ] {
            // Default is false on every arch, through both resolution paths.
            let from_defaults = FeatureFlags::from_process_config(arch, &defaults);
            assert!(!from_defaults.gfx11_a8_prefill, "{arch}");
            assert!(!from_defaults.a8_prefill_enabled(), "{arch}");
            assert!(
                !FeatureFlags::from_lookup(arch, |_| Err(())).a8_prefill_enabled(),
                "{arch}"
            );
            assert!(!FeatureFlags::for_test(arch).a8_prefill_enabled(), "{arch}");
            let only =
                |env: &'static str| {
                    FeatureFlags::from_lookup(arch, move |name| {
                        if name == env {
                            Ok("1".into())
                        } else {
                            Err(())
                        }
                    })
                };
            assert_eq!(
                only("HIPFIRE_GFX11_A8_PREFILL").a8_prefill_enabled(),
                matches!(arch, "gfx1100" | "gfx1151")
            );
            assert_eq!(
                only("HIPFIRE_A8_PREFILL").a8_prefill_enabled(),
                arch == "gfx1201"
            );
        }
        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.gfx11_a8_prefill", "true").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        assert!(FeatureFlags::from_process_config("gfx1151", &process).a8_prefill_enabled());
        assert!(!FeatureFlags::from_process_config("gfx1201", &process).a8_prefill_enabled());
    }

    #[test]
    fn mq4v2_gateup_k5120_process_config_roundtrip() {
        let defaults = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        assert!(FeatureFlags::from_process_config("gfx1100", &defaults)
            .mq4v2_gateup_k5120_enabled(17408, 17408, 5120));

        for key in ["kernel.mq4v2_gateup_k5120", "mq4v2_gateup_k5120"] {
            for (value, expected) in [("true", true), ("false", false)] {
                let mut layer = ConfigLayer::default();
                layer.set_cli(key, value).unwrap();
                let resolved = resolve([NamedLayer {
                    source: ConfigSource::GlobalUser {
                        path: "config.toml".into(),
                    },
                    layer,
                }])
                .unwrap();
                let process = ProcessConfig::from_resolved(&resolved).unwrap();
                let wire = serde_json::to_string(&process).unwrap();
                let decoded: ProcessConfig = serde_json::from_str(&wire).unwrap();
                assert_eq!(
                    decoded.legacy_value("HIPFIRE_MQ4V2_GATEUP_K5120"),
                    Some(if expected { "1" } else { "0" }.to_string())
                );
                for arch in ["gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1201"] {
                    let flags = FeatureFlags::from_process_config(arch, &decoded);
                    assert_eq!(
                        flags.mq4v2_gateup_k5120_enabled(17408, 17408, 5120),
                        expected && arch == "gfx1100"
                    );
                }
            }
        }
    }

    #[test]
    fn force_unfused_defaults_false_in_test_ctor() {
        let f = FeatureFlags::for_test("gfx1151");
        assert!(!f.force_unfused);
    }

    #[test]
    fn fuse_qkv_bias_defaults_on_in_test_ctor() {
        let f = FeatureFlags::for_test("gfx1151");
        assert!(f.fuse_qkv_bias);
    }

    /// H4/H3/H8a flags are exact gfx1151 / gfx1201: unset, H4 is level 3 on
    /// gfx1151 and level 1 on gfx1201, H3 and H8a default on for both, `0`
    /// turns each off, and the fusion level is clamped to 3.
    #[test]
    fn qwen4_hc_flags_default_on_exact_gfx1151_gfx1201() {
        let with = |fuse: &'static str, rest: &'static str| {
            move |name: &str| -> std::result::Result<String, ()> {
                match name {
                    "HIPFIRE_QWEN4_HC_FUSE" => Ok(fuse.into()),
                    "HIPFIRE_QWEN4_HC_UP_TILE" | "HIPFIRE_QWEN4_MOE_COMBINE_ZINIT" => {
                        Ok(rest.into())
                    }
                    _ => Err(()),
                }
            }
        };
        for arch in [
            "gfx906", "gfx1030", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            let halo = arch == "gfx1151";
            let exact = matches!(arch, "gfx1151" | "gfx1201");
            let default_level = match arch {
                "gfx1151" => 3,
                "gfx1201" => 1,
                _ => 0,
            };
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_hc_fuse_level(), default_level, "{arch}");
            assert_eq!(unset.qwen4_hc_up_tile_enabled(), exact, "{arch}");
            assert_eq!(unset.qwen4_moe_combine_zinit_enabled(), exact, "{arch}");
            assert_eq!(unset.qwen4_hc_row_fold_enabled(), halo, "{arch}");
            let off = FeatureFlags::from_lookup(arch, with("0", "0"));
            assert_eq!(off.qwen4_hc_fuse_level(), 0, "{arch}");
            assert!(!off.qwen4_hc_up_tile_enabled() && !off.qwen4_moe_combine_zinit_enabled());
            let on = FeatureFlags::from_lookup(arch, with("2", "1"));
            assert_eq!(
                on.qwen4_hc_fuse_level(),
                if exact { 2 } else { 0 },
                "{arch}"
            );
            assert_eq!(on.qwen4_hc_up_tile_enabled(), exact, "{arch}");
            assert_eq!(on.qwen4_moe_combine_zinit_enabled(), exact, "{arch}");
            assert_eq!(
                FeatureFlags::from_lookup(arch, with("9", "1")).qwen4_hc_fuse_level(),
                if exact { 3 } else { 0 }
            );
            assert_eq!(
                FeatureFlags::from_lookup(arch, with("junk", "1")).qwen4_hc_fuse_level(),
                default_level
            );
        }
    }

    /// `HIPFIRE_QWEN4_HC_ROW_FOLD`: the gfx1151 accessor is unchanged (gfx1151
    /// only, needs level 3 + zinit); gfx1201 takes its own two accessors, default
    /// on at the gfx1201 default level 1, independent of levels >= 2/3, killed by
    /// `=0`; the mixer fold needs the read-side pairing (level >= 1), the MoE fold
    /// also the zero-initialized combine.
    #[test]
    fn qwen4_hc_row_fold_flag_gfx1151_unchanged_gfx1201_default_on() {
        let with = |entries: &'static [(&'static str, &'static str)]| {
            move |name: &str| -> std::result::Result<String, ()> {
                entries
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).to_string())
                    .ok_or(())
            }
        };
        for arch in [
            "gfx906", "gfx1030", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            let (halo, rdna4) = (arch == "gfx1151", arch == "gfx1201");
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_hc_row_fold_enabled(), halo, "{arch}");
            assert_eq!(
                unset.qwen4_hc_row_fold_mixer_gfx1201_enabled(),
                rdna4,
                "{arch}"
            );
            assert_eq!(
                unset.qwen4_hc_row_fold_moe_gfx1201_enabled(),
                rdna4,
                "{arch}"
            );
            let killed =
                FeatureFlags::from_lookup(arch, with(&[("HIPFIRE_QWEN4_HC_ROW_FOLD", "0")]));
            assert!(!killed.qwen4_hc_row_fold_enabled(), "{arch}");
            assert!(!killed.qwen4_hc_row_fold_mixer_gfx1201_enabled(), "{arch}");
            assert!(!killed.qwen4_hc_row_fold_moe_gfx1201_enabled(), "{arch}");
            let no_pairing =
                FeatureFlags::from_lookup(arch, with(&[("HIPFIRE_QWEN4_HC_FUSE", "0")]));
            assert!(
                !no_pairing.qwen4_hc_row_fold_mixer_gfx1201_enabled(),
                "{arch}"
            );
            assert!(
                !no_pairing.qwen4_hc_row_fold_moe_gfx1201_enabled(),
                "{arch}"
            );
            assert!(!no_pairing.qwen4_hc_row_fold_enabled(), "{arch}");
            let no_zinit =
                FeatureFlags::from_lookup(arch, with(&[("HIPFIRE_QWEN4_MOE_COMBINE_ZINIT", "0")]));
            assert_eq!(
                no_zinit.qwen4_hc_row_fold_mixer_gfx1201_enabled(),
                rdna4,
                "{arch}"
            );
            assert!(!no_zinit.qwen4_hc_row_fold_moe_gfx1201_enabled(), "{arch}");
            assert!(!no_zinit.qwen4_hc_row_fold_enabled(), "{arch}");
            // An explicit `=1` enables nothing off gfx1151 / gfx1201.
            let forced =
                FeatureFlags::from_lookup(arch, with(&[("HIPFIRE_QWEN4_HC_ROW_FOLD", "1")]));
            assert_eq!(
                forced.qwen4_hc_row_fold_mixer_gfx1201_enabled(),
                rdna4,
                "{arch}"
            );
            assert_eq!(forced.qwen4_hc_row_fold_enabled(), halo, "{arch}");
            // Level 3 on gfx1201 does not change which fold runs.
            let level3 = FeatureFlags::from_lookup(arch, with(&[("HIPFIRE_QWEN4_HC_FUSE", "3")]));
            assert!(!level3.qwen4_hc_row_fold_enabled() || halo, "{arch}");
        }
    }

    /// The fused PLE tail defaults on for exact gfx1151 / gfx1201; `0` turns it off.
    #[test]
    fn qwen4_ple_fuse_flag_defaults_on_exact_gfx1151_gfx1201() {
        let with = |value: &'static str| {
            move |name: &str| -> std::result::Result<String, ()> {
                if name == "HIPFIRE_QWEN4_PLE_FUSE" {
                    Ok(value.into())
                } else {
                    Err(())
                }
            }
        };
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            let exact = matches!(arch, "gfx1151" | "gfx1201");
            assert_eq!(
                FeatureFlags::from_lookup(arch, |_| Err(())).qwen4_ple_fuse_enabled(),
                exact,
                "{arch}"
            );
            assert!(!FeatureFlags::from_lookup(arch, with("0")).qwen4_ple_fuse_enabled());
            assert_eq!(
                FeatureFlags::from_lookup(arch, with("1")).qwen4_ple_fuse_enabled(),
                exact,
                "{arch}"
            );
        }
    }

    /// The wave-per-token router defaults on for exact gfx1151 / gfx1201; `0` turns it off.
    #[test]
    fn qwen4_router_fast_defaults_on_exact_gfx1151_gfx1201() {
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            let exact = matches!(arch, "gfx1151" | "gfx1201");
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_router_fast_enabled(), exact, "{arch}");
            let off = FeatureFlags::from_lookup(arch, |n| {
                if n == "HIPFIRE_QWEN4_ROUTER_FAST" {
                    Ok("0".into())
                } else {
                    Err(())
                }
            });
            assert!(!off.qwen4_router_fast_enabled(), "{arch}");
            let on = FeatureFlags::from_lookup(arch, |n| {
                if n == "HIPFIRE_QWEN4_ROUTER_FAST" {
                    Ok("1".into())
                } else {
                    Err(())
                }
            });
            assert_eq!(on.qwen4_router_fast_enabled(), exact, "{arch}");
        }
    }

    /// The decode MoE head fusion defaults on for exact gfx1151 only; `0` turns it off.
    #[test]
    fn qwen4_decode_head_fuse_defaults_on_exact_gfx1151() {
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            let exact = arch == "gfx1151";
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_decode_head_fuse_enabled(), exact, "{arch}");
            let off = FeatureFlags::from_lookup(arch, |n| {
                if n == "HIPFIRE_QWEN4_DECODE_HEAD_FUSE" {
                    Ok("0".into())
                } else {
                    Err(())
                }
            });
            assert!(!off.qwen4_decode_head_fuse_enabled(), "{arch}");
            let on = FeatureFlags::from_lookup(arch, |n| {
                if n == "HIPFIRE_QWEN4_DECODE_HEAD_FUSE" {
                    Ok("1".into())
                } else {
                    Err(())
                }
            });
            assert_eq!(on.qwen4_decode_head_fuse_enabled(), exact, "{arch}");
        }
    }

    #[test]
    fn qwen4_decode_zba_x3_defaults_on_exact_gfx1151() {
        for arch in [
            "gfx906", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
        ] {
            let exact = arch == "gfx1151";
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_decode_zba_x3_enabled(), exact, "{arch}");
            for (value, expected) in [("0", false), ("1", exact)] {
                let flags = FeatureFlags::from_lookup(arch, |name| {
                    if name == "HIPFIRE_QWEN4_DECODE_ZBA_X3" {
                        Ok(value.into())
                    } else {
                        Err(())
                    }
                });
                assert_eq!(flags.qwen4_decode_zba_x3_enabled(), expected, "{arch}");
            }
        }
    }

    /// H5 (GDN conv + Q/K norm) defaults on for exact gfx1151 only, `0` turns
    /// it off; H6 (inline Q8) stays opt-in. Neither applies off gfx1151.
    #[test]
    fn qwen4_gdn_hyper_flags_exact_gfx1151_h5_default_on_h6_opt_in() {
        let on = |name: &str| -> std::result::Result<String, ()> {
            if matches!(
                name,
                "HIPFIRE_QWEN4_GDN_CONV_QKNORM" | "HIPFIRE_QWEN4_GDN_Q8_INLINE"
            ) {
                Ok("1".into())
            } else {
                Err(())
            }
        };
        let off = |name: &str| -> std::result::Result<String, ()> {
            if matches!(
                name,
                "HIPFIRE_QWEN4_GDN_CONV_QKNORM" | "HIPFIRE_QWEN4_GDN_Q8_INLINE"
            ) {
                Ok("0".into())
            } else {
                Err(())
            }
        };
        for arch in [
            "gfx906", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1152",
            "gfx1200", "gfx1201",
        ] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(
                unset.qwen4_gdn_conv_qknorm_enabled(),
                arch == "gfx1151",
                "{arch}: unset"
            );
            assert!(!unset.qwen4_gdn_q8_inline_enabled(), "{arch}: unset");
            let zero = FeatureFlags::from_lookup(arch, off);
            assert!(!zero.qwen4_gdn_conv_qknorm_enabled(), "{arch}: 0");
            assert!(!zero.qwen4_gdn_q8_inline_enabled(), "{arch}: 0");
            let one = FeatureFlags::from_lookup(arch, on);
            assert_eq!(
                one.qwen4_gdn_conv_qknorm_enabled(),
                arch == "gfx1151",
                "{arch}: 1"
            );
            assert_eq!(
                one.qwen4_gdn_q8_inline_enabled(),
                arch == "gfx1151",
                "{arch}: 1"
            );
        }
        assert!(!FeatureFlags::for_test("gfx1151").qwen4_gdn_conv_qknorm_enabled());
        assert!(!FeatureFlags::for_test("gfx1151").qwen4_gdn_q8_inline_enabled());
    }

    /// The dense-GDN gate + out-projection A8 producer fusion: default on for
    /// exact gfx1201 only, `0` is the kill switch, `1` never reaches another arch.
    #[test]
    fn qwen4_gdn_i8_fuse_defaults_on_exact_gfx1201_with_kill_switch() {
        let with = |value: &'static str| {
            move |name: &str| -> std::result::Result<String, ()> {
                if name == "HIPFIRE_QWEN4_GDN_I8_FUSE" {
                    Ok(value.into())
                } else {
                    Err(())
                }
            }
        };
        for arch in [
            "gfx906", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1152",
            "gfx1200", "gfx1201",
        ] {
            let exact = arch == "gfx1201";
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_gdn_i8_fuse_enabled(), exact, "{arch}: unset");
            let zero = FeatureFlags::from_lookup(arch, with("0"));
            assert!(!zero.qwen4_gdn_i8_fuse_enabled(), "{arch}: 0");
            let one = FeatureFlags::from_lookup(arch, with("1"));
            assert_eq!(one.qwen4_gdn_i8_fuse_enabled(), exact, "{arch}: 1");
        }
        assert!(!FeatureFlags::for_test("gfx1201").qwen4_gdn_i8_fuse_enabled());
    }

    /// The persistent HC up+mix entry: default on for exact gfx1201 and gfx1151
    /// only, `0` is the kill switch, `1` never reaches another arch.
    #[test]
    fn qwen4_hc_up_v2_defaults_on_exact_gfx1201_gfx1151_with_kill_switch() {
        let with = |value: &'static str| {
            move |name: &str| -> std::result::Result<String, ()> {
                if name == "HIPFIRE_QWEN4_HC_UP_V2" {
                    Ok(value.into())
                } else {
                    Err(())
                }
            }
        };
        for arch in [
            "gfx906", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1152",
            "gfx1200", "gfx1201",
        ] {
            let exact = matches!(arch, "gfx1201" | "gfx1151");
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_hc_up_v2_enabled(), exact, "{arch}: unset");
            let zero = FeatureFlags::from_lookup(arch, with("0"));
            assert!(!zero.qwen4_hc_up_v2_enabled(), "{arch}: 0");
            let one = FeatureFlags::from_lookup(arch, with("1"));
            assert_eq!(one.qwen4_hc_up_v2_enabled(), exact, "{arch}: 1");
        }
        assert!(!FeatureFlags::for_test("gfx1201").qwen4_hc_up_v2_enabled());
    }

    /// The dense-GDN chain fusion: default on for exact gfx1201 only, `0` is the kill
    /// switch, `1` never reaches another arch.
    #[test]
    fn qwen4_gdn_chain_fuse_defaults_on_exact_gfx1201_with_kill_switch() {
        let with = |value: &'static str| {
            move |name: &str| -> std::result::Result<String, ()> {
                if name == "HIPFIRE_QWEN4_GDN_CHAIN_FUSE" {
                    Ok(value.into())
                } else {
                    Err(())
                }
            }
        };
        for arch in [
            "gfx906", "gfx942", "gfx1030", "gfx1100", "gfx1101", "gfx1150", "gfx1151", "gfx1152",
            "gfx1200", "gfx1201",
        ] {
            let exact = arch == "gfx1201";
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.qwen4_gdn_chain_fuse_enabled(), exact, "{arch}: unset");
            let zero = FeatureFlags::from_lookup(arch, with("0"));
            assert!(!zero.qwen4_gdn_chain_fuse_enabled(), "{arch}: 0");
            let one = FeatureFlags::from_lookup(arch, with("1"));
            assert_eq!(one.qwen4_gdn_chain_fuse_enabled(), exact, "{arch}: 1");
        }
        assert!(!FeatureFlags::for_test("gfx1201").qwen4_gdn_chain_fuse_enabled());
    }

    /// The BF16 qkv store is opt-in and needs the chain fusion: never on by default, never
    /// with `HIPFIRE_QWEN4_GDN_CHAIN_FUSE=0`, never off gfx1201.
    #[test]
    fn qwen4_gdn_qkv_bf16_is_opt_in_behind_the_chain_fusion() {
        let with = |chain: Option<&'static str>, qkv: Option<&'static str>| {
            move |name: &str| -> std::result::Result<String, ()> {
                match (name, chain, qkv) {
                    ("HIPFIRE_QWEN4_GDN_CHAIN_FUSE", Some(v), _) => Ok(v.into()),
                    ("HIPFIRE_QWEN4_GDN_QKV_BF16", _, Some(v)) => Ok(v.into()),
                    _ => Err(()),
                }
            }
        };
        let on = |arch: &str, chain, qkv| {
            FeatureFlags::from_lookup(arch, with(chain, qkv)).qwen4_gdn_qkv_bf16_enabled()
        };
        assert!(!on("gfx1201", None, None));
        assert!(on("gfx1201", None, Some("1")));
        assert!(!on("gfx1201", Some("0"), Some("1")));
        assert!(!on("gfx1201", None, Some("0")));
        assert!(!on("gfx1151", None, Some("1")));
        assert!(!FeatureFlags::for_test("gfx1201").qwen4_gdn_qkv_bf16_enabled());
    }

    #[test]
    fn gfx12_mq4v2_fp8_defaults_on_gfx1201_with_opt_out() {
        // Default process policy: all four FP8 prefill flags admit on exact
        // gfx1201 (chunk-512 path); explicit `=0` restores the F16 route.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gfx12_mq4v2_fp8_gateup);
        assert!(gfx1201.gfx12_mq4v2_fp8_resid);
        assert!(gfx1201.gfx12_mq4v2_fp8_qkvza);
        assert!(gfx1201.gfx12_mq4v2_fp8_qkv);

        let mut layer = ConfigLayer::default();
        for key in [
            "kernel.gfx12_mq4v2_fp8_gateup",
            "kernel.gfx12_mq4v2_fp8_resid",
            "kernel.gfx12_mq4v2_fp8_qkvza",
            "kernel.gfx12_mq4v2_fp8_qkv",
        ] {
            layer.set_cli(key, "false").unwrap();
        }
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_out = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(!opted_out.gfx12_mq4v2_fp8_gateup);
        assert!(!opted_out.gfx12_mq4v2_fp8_resid);
        assert!(!opted_out.gfx12_mq4v2_fp8_qkvza);
        assert!(!opted_out.gfx12_mq4v2_fp8_qkv);

        // The unit-test constructor stays fully off (deterministic baseline).
        let test_flags = FeatureFlags::for_test("gfx1201");
        assert!(!test_flags.gfx12_mq4v2_fp8_gateup);
        assert!(!test_flags.gfx12_mq4v2_fp8_resid);
        assert!(!test_flags.gfx12_mq4v2_fp8_qkvza);
        assert!(!test_flags.gfx12_mq4v2_fp8_qkv);
    }
    #[test]
    fn wmma_batch_tiles_defaults_on_gfx1201_with_opt_out() {
        let flag = |arch: &str, v: Option<&str>| {
            FeatureFlags::from_lookup(arch, |name| match (name, v) {
                ("HIPFIRE_WMMA_BATCH_TILES", Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
            .wmma_batch_tiles
        };
        assert!(flag("gfx1201", None));
        assert!(!flag("gfx1201", Some("0")));
        assert!(flag("gfx1201", Some("1")));
        assert!(!flag("gfx1100", None));
        // The unit-test constructor stays on the historical one-tile route.
        assert!(!FeatureFlags::for_test("gfx1201").wmma_batch_tiles);
    }

    #[test]
    fn cb_verify_chunk128_defaults_on_for_gfx1201_and_pm_defaults_on() {
        let flags = |chunk: Option<&str>, pm: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, chunk, pm) {
                ("HIPFIRE_CB_VERIFY_CHUNK128", Some(v), _) => Ok(v.to_owned()),
                ("HIPFIRE_CB_VERIFY_PM", _, Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
        };
        let default = flags(None, None);
        assert!(default.cb_verify_chunk128);
        assert!(default.cb_verify_pm);
        assert!(flags(Some("1"), None).cb_verify_chunk128);
        assert!(!flags(Some("0"), None).cb_verify_chunk128);
        // Other arches never default it on (their cap stays 63 regardless).
        let other = FeatureFlags::from_lookup("gfx1100", |_| Err(()));
        assert!(!other.cb_verify_chunk128);
        // The unit-test constructor stays on the historical 63-row route.
        assert!(!flags(None, Some("0")).cb_verify_pm);
        assert!(flags(None, Some("1")).cb_verify_pm);
        assert!(!FeatureFlags::for_test("gfx1201").cb_verify_chunk128);
    }

    #[test]
    fn cb_verify_fp8_defaults_off_and_parses() {
        let flags = |v: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, v) {
                ("HIPFIRE_CB_VERIFY_FP8", Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
        };
        assert!(!flags(None).cb_verify_fp8);
        assert!(flags(Some("1")).cb_verify_fp8);
        assert!(!flags(Some("0")).cb_verify_fp8);
        // Independent of the PM F16 family flag and off in the test constructor.
        assert!(flags(Some("1")).cb_verify_pm);
        assert!(!FeatureFlags::for_test("gfx1201").cb_verify_fp8);
        assert!(!FeatureFlags::for_test("gfx1100").cb_verify_fp8);
    }

    #[test]
    fn cb_verify_iu4_is_tri_state() {
        let flags = |v: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, v) {
                ("HIPFIRE_CB_VERIFY_IU4", Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
        };
        assert_eq!(flags(None).cb_verify_iu4, None);
        assert_eq!(flags(Some("1")).cb_verify_iu4, Some(true));
        assert_eq!(flags(Some("0")).cb_verify_iu4, Some(false));
        assert_eq!(FeatureFlags::for_test("gfx1201").cb_verify_iu4, None);
    }

    #[test]
    fn gfx12_mq6_fp8_is_tri_state_and_arm_defaults_split() {
        use crate::gemm_mq6_fp8::Mq6Fp8Arm;
        let flags = |fp8: Option<&str>, arm: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, fp8, arm) {
                ("HIPFIRE_GFX12_MQ6_FP8", Some(v), _) => Ok(v.to_owned()),
                ("HIPFIRE_GFX12_MQ6_FP8_ARM", _, Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
        };
        assert_eq!(flags(None, None).gfx12_mq6_fp8, None);
        assert_eq!(flags(Some("1"), None).gfx12_mq6_fp8, Some(true));
        assert_eq!(flags(Some("0"), None).gfx12_mq6_fp8, Some(false));
        assert_eq!(flags(Some("auto"), None).gfx12_mq6_fp8, None);
        assert_eq!(flags(None, None).gfx12_mq6_fp8_arm, Mq6Fp8Arm::Split);
        assert_eq!(
            flags(None, Some("split")).gfx12_mq6_fp8_arm,
            Mq6Fp8Arm::Split
        );
        assert_eq!(
            flags(None, Some("round")).gfx12_mq6_fp8_arm,
            Mq6Fp8Arm::Round
        );
        assert_eq!(
            flags(None, Some("ROUND")).gfx12_mq6_fp8_arm,
            Mq6Fp8Arm::Round
        );
        assert_eq!(
            flags(None, Some("bogus")).gfx12_mq6_fp8_arm,
            Mq6Fp8Arm::Split
        );
        // Arch-independent parse; the unit-test constructor stays auto + split.
        let t = FeatureFlags::for_test("gfx1201");
        assert_eq!(t.gfx12_mq6_fp8, None);
        assert_eq!(t.gfx12_mq6_fp8_arm, Mq6Fp8Arm::Split);
        // The DB variant flag: `1` on, anything else off; test constructor off.
        let db = |v: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, v) {
                ("HIPFIRE_GFX12_MQ6_FP8_DB", Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
            .gfx12_mq6_fp8_db
        };
        assert!(!db(None));
        assert!(db(Some("1")));
        assert!(!db(Some("0")));
        assert!(!db(Some("bogus")));
        assert!(!t.gfx12_mq6_fp8_db);
    }

    #[test]
    fn mq6_i8_is_tri_state() {
        let flag = |v: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, v) {
                ("HIPFIRE_MQ6_I8", Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
            .mq6_i8
        };
        assert_eq!(flag(None), None);
        assert_eq!(flag(Some("1")), Some(true));
        assert_eq!(flag(Some("0")), Some(false));
        assert_eq!(flag(Some("auto")), None);
        assert_eq!(FeatureFlags::for_test("gfx1201").mq6_i8, None);
        assert_eq!(FeatureFlags::for_test("gfx1151").mq6_i8, None);
    }

    #[test]
    fn mq6_i8_pm_is_tri_state_and_unset_is_the_default() {
        let flag = |v: Option<&str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match (name, v) {
                ("HIPFIRE_MQ6_I8_PM", Some(v)) => Ok(v.to_owned()),
                _ => Err(()),
            })
            .mq6_i8_pm
        };
        assert_eq!(flag(None), None);
        assert_eq!(flag(Some("1")), Some(true));
        assert_eq!(flag(Some("0")), Some(false));
        assert_eq!(flag(Some("auto")), None);
        assert_eq!(FeatureFlags::for_test("gfx1201").mq6_i8_pm, None);
    }

    #[test]
    fn cb_verify_iu4_load_decision() {
        let decide = |iu4: Option<bool>, fp8: bool, block: usize| {
            let mut f = FeatureFlags::for_test("gfx1201");
            f.cb_verify_iu4 = iu4;
            f.cb_verify_fp8 = fp8;
            crate::Gpu::iu4_verify_load_decision(&f, block)
        };
        // Default: on exactly at the batching width 8, and FP8 wins it.
        assert_eq!(decide(None, false, 8), Ok(true));
        assert_eq!(decide(None, false, 16), Ok(false));
        assert_eq!(decide(None, true, 8), Ok(false));
        // Kill switch and forced on, any width.
        assert_eq!(decide(Some(false), false, 8), Ok(false));
        assert_eq!(decide(Some(true), false, 16), Ok(true));
        // Both forced on: refused.
        assert!(decide(Some(true), true, 8).is_err());
        assert_eq!(decide(Some(false), true, 8), Ok(false));
    }

    #[test]
    fn gfx12_mq4v2_fp8_v2_and_gdn_pre_fused_default_on_gfx1201_with_opt_out() {
        // Default process policy: staged-tile v2 and the GDN preamble fusion
        // admit on exact gfx1201; explicit `=0` restores the prior route.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gfx12_mq4v2_fp8_v2);
        assert!(gfx1201.gfx12_gdn_pre_fused);
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx12_mq4v2_fp8_v2, "arch={arch}");
            assert!(!flags.gfx12_gdn_pre_fused, "arch={arch}");
        }

        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.gfx12_mq4v2_fp8_v2", "false").unwrap();
        layer
            .set_cli("kernel.gfx12_gdn_pre_fused", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_out = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(!opted_out.gfx12_mq4v2_fp8_v2);
        assert!(!opted_out.gfx12_gdn_pre_fused);

        // The unit-test constructor stays fully off (deterministic baseline).
        let test_flags = FeatureFlags::for_test("gfx1201");
        assert!(!test_flags.gfx12_mq4v2_fp8_v2);
        assert!(!test_flags.gfx12_gdn_pre_fused);
    }

    #[test]
    fn fa2_prefill_defaults_on_certified_arches_with_opt_out() {
        // Default process policy: gfx12 FA2 admits on exact gfx1201, gfx11
        // FA2 on gfx1100/gfx1151; explicit `=0` restores the incumbent.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gfx12_fa2_prefill);
        assert!(!gfx1201.gfx11_fa2_prefill);
        for arch in ["gfx1100", "gfx1151"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx12_fa2_prefill, "arch={arch}");
            assert!(flags.gfx11_fa2_prefill, "arch={arch}");
        }
        for arch in ["gfx1101", "gfx1102", "gfx1150", "gfx1200", "gfx942"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx12_fa2_prefill, "arch={arch}");
            assert!(!flags.gfx11_fa2_prefill, "arch={arch}");
        }

        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.gfx12_fa2_prefill", "false").unwrap();
        layer.set_cli("kernel.gfx11_fa2_prefill", "false").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_out_12 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(!opted_out_12.gfx12_fa2_prefill);
        let opted_out_11 = FeatureFlags::from_process_config("gfx1100", &process);
        assert!(!opted_out_11.gfx11_fa2_prefill);

        // The unit-test constructor stays fully off (deterministic baseline).
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            let test_flags = FeatureFlags::for_test(arch);
            assert!(!test_flags.gfx12_fa2_prefill, "arch={arch}");
            assert!(!test_flags.gfx11_fa2_prefill, "arch={arch}");
        }
    }

    #[test]
    fn fa_prefill_pm_defaults_to_qualification_with_explicit_override() {
        use crate::prefill_route_capabilities::{qualified, PrefillRoute};
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(
                unset.fa_prefill_pm,
                qualified(arch, PrefillRoute::FaPrefillPm)
            );
            assert_eq!(unset.fa_prefill_pm, arch == "gfx1201");
            assert!(!FeatureFlags::for_test(arch).fa_prefill_pm);
            assert_eq!(
                unset.fa_prefill_pm_fp8_kv,
                arch == "gfx1201",
                "{arch}: fp8 default"
            );
            assert!(!FeatureFlags::for_test(arch).fa_prefill_pm_fp8_kv);
            for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
                let flags = FeatureFlags::from_lookup(arch, |name| {
                    if name == "HIPFIRE_FA_PREFILL_PM" {
                        Ok(raw.into())
                    } else {
                        Err(())
                    }
                });
                assert_eq!(flags.fa_prefill_pm, want, "{arch}: {raw}");
                assert_eq!(flags.fa_prefill_pm_fp8_kv, want, "{arch}: fp8 {raw}");
            }
        }
    }

    #[test]
    fn moe_iu4_wide_tiles_defaults_to_qualification_with_explicit_override() {
        use crate::prefill_route_capabilities::{qualified, raw_inventory, PrefillRoute};
        const ROUTE: PrefillRoute = PrefillRoute::MoeIu4WideTiles;
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(unset.moe_iu4_wide_tiles, qualified(arch, ROUTE), "{arch}");
            assert_eq!(unset.moe_iu4_wide_tiles, arch == "gfx1201", "{arch}");
            assert!(!FeatureFlags::for_test(arch).moe_iu4_wide_tiles, "{arch}");
            for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
                let flags = FeatureFlags::from_lookup(arch, |name| {
                    if name == "HIPFIRE_MOE_IU4_WIDE_TILES" {
                        Ok(raw.into())
                    } else {
                        Err(())
                    }
                });
                // The flag is the typed request only; raw inventory is applied at selection.
                assert_eq!(flags.moe_iu4_wide_tiles, want, "{arch}: {raw}");
            }
            // No other flag moves with it: an explicit PM `1` leaves the wide-tile default
            // at its own qualification.
            let only_other = FeatureFlags::from_lookup(arch, |name| {
                if name == "HIPFIRE_FA_PREFILL_PM" {
                    Ok("1".into())
                } else {
                    Err(())
                }
            });
            assert_eq!(only_other.moe_iu4_wide_tiles, arch == "gfx1201", "{arch}");
            assert!(only_other.fa_prefill_pm, "{arch}");
        }
        assert!(raw_inventory("gfx1201", ROUTE) && !raw_inventory("gfx1151", ROUTE));
    }

    #[test]
    #[should_panic(expected = "HIPFIRE_MOE_IU4_WIDE_TILES is not a boolean")]
    fn moe_iu4_wide_tiles_rejects_a_non_boolean_spelling() {
        let _ = FeatureFlags::from_lookup("gfx1201", |name| {
            if name == "HIPFIRE_MOE_IU4_WIDE_TILES" {
                Ok("on".into())
            } else {
                Err(())
            }
        });
    }

    #[test]
    fn moe_iu4_wide_tiles_selection_is_independent_of_the_combine_v2_and_multirow_flags() {
        use crate::gemm::qwen4_moe_sym_nt8_pair;
        // A3B top-8, 256 experts, 8192 rows: past the wide-tile shape floor.
        const SLOTS: usize = 8192 * 8;
        const EXPERTS: usize = 256;
        let with = |wide: Option<&'static str>, combine: Option<&'static str>| {
            FeatureFlags::from_lookup("gfx1201", |name| match name {
                "HIPFIRE_MOE_IU4_WIDE_TILES" => wide.map(String::from).ok_or(()),
                "HIPFIRE_MOE_COMBINE_V2" => combine.map(String::from).ok_or(()),
                "HIPFIRE_MOE_ROUTER_TOPK_MULTIROW" => combine.map(String::from).ok_or(()),
                _ => Err(()),
            })
        };
        let pair = |flags: &FeatureFlags, host: bool| {
            qwen4_moe_sym_nt8_pair("gfx1201", host, flags.moe_iu4_wide_tiles, SLOTS, EXPERTS)
        };
        // The F1 / F4 variables are read by `hipfire-dispatch`, never by `FeatureFlags`; setting
        // them in the lookup proves nothing here reads them for the wide-tile flag.
        for combine in [None, Some("0"), Some("1")] {
            // F2 killed: resident stays the NT4 incumbent whatever F1 / F4 say.
            let ctx = format!("wide=Some(\"0\") combine={combine:?}");
            let flags = with(Some("0"), combine);
            assert!(!flags.moe_iu4_wide_tiles, "{ctx}");
            assert_eq!(pair(&flags, false), [false, false], "{ctx}");
            // Host-mapped keeps its NT8 pair whatever any of the flags say.
            assert_eq!(pair(&flags, true), [true, true], "{ctx}");
            // F2 unset (qualified on exact gfx1201) or explicit 1: resident gate/up widens,
            // down stays NT4, host-mapped unchanged.
            for wide in [None, Some("1")] {
                let ctx = format!("wide={wide:?} combine={combine:?}");
                let flags = with(wide, combine);
                assert!(flags.moe_iu4_wide_tiles, "{ctx}");
                assert_eq!(pair(&flags, false), [true, false], "{ctx}");
                assert_eq!(pair(&flags, true), [true, true], "{ctx}");
            }
        }
    }

    #[test]
    fn fa_prefill_pm_qprep_defaults_to_qualification_with_explicit_override() {
        use crate::prefill_route_capabilities::{qualified, PrefillRoute};
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(
                unset.fa_prefill_pm_qprep,
                qualified(arch, PrefillRoute::FaPrefillPmQprep)
            );
            assert_eq!(unset.fa_prefill_pm_qprep, arch == "gfx1201");
            assert!(!FeatureFlags::for_test(arch).fa_prefill_pm_qprep);
            assert_eq!(
                unset.fa_prefill_pm_qprep_fp8_kv,
                arch == "gfx1201",
                "{arch}: fp8 default"
            );
            assert!(!FeatureFlags::for_test(arch).fa_prefill_pm_qprep_fp8_kv);
            for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
                let flags = FeatureFlags::from_lookup(arch, |name| {
                    if name == "HIPFIRE_FA_PREFILL_PM_QPREP" {
                        Ok(raw.into())
                    } else {
                        Err(())
                    }
                });
                assert_eq!(flags.fa_prefill_pm_qprep, want, "{arch}: {raw}");
                assert_eq!(flags.fa_prefill_pm_qprep_fp8_kv, want, "{arch}: fp8 {raw}");
                // Independent of the PM attention flag: the PM flag stays the resolved
                // qualification (on only on exact gfx1201), whatever the Q-prep flag says.
                assert_eq!(
                    flags.fa_prefill_pm,
                    qualified(arch, PrefillRoute::FaPrefillPm),
                    "{arch}: {raw}"
                );
            }
        }
    }

    #[test]
    fn fa_prefill_pm_qprep_resolves_beside_the_resolved_pm_flag() {
        let with = |arch: &str, pm: Option<&'static str>, qprep: Option<&'static str>| {
            FeatureFlags::from_lookup(arch, |name| match name {
                "HIPFIRE_FA_PREFILL_PM" => pm.map(String::from).ok_or(()),
                "HIPFIRE_FA_PREFILL_PM_QPREP" => qprep.map(String::from).ok_or(()),
                _ => Err(()),
            })
        };
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let gfx1201 = arch == "gfx1201";
            // Both unset: each resolves to its own qualification (on only on exact gfx1201),
            // so the pair is admitted by default exactly there.
            let unset = with(arch, None, None);
            assert_eq!(unset.fa_prefill_pm, gfx1201, "{arch}");
            assert_eq!(unset.fa_prefill_pm_qprep, gfx1201, "{arch}");
            assert_eq!(unset.fa_prefill_pm_fp8_kv, gfx1201, "{arch}");
            assert_eq!(unset.fa_prefill_pm_qprep_fp8_kv, gfx1201, "{arch}");
            // PM unset + Q-prep 1: the resolved PM flag is on exactly where qualified.
            let pm_unset = with(arch, None, Some("1"));
            assert_eq!(pm_unset.fa_prefill_pm, gfx1201, "{arch}");
            assert!(pm_unset.fa_prefill_pm_qprep, "{arch}");
            assert!(pm_unset.fa_prefill_pm_qprep_fp8_kv, "{arch}");
            // PM 0 + Q-prep unset / 1: the PM route is off, so the fold has nothing to feed
            // (the admission refuses it with `PmDisabled`); the Q-prep flag itself is its own.
            let pm_off = with(arch, Some("0"), None);
            assert!(!pm_off.fa_prefill_pm, "{arch}");
            assert_eq!(pm_off.fa_prefill_pm_qprep, gfx1201, "{arch}");
            assert!(!pm_off.fa_prefill_pm_fp8_kv, "{arch}");
            assert_eq!(pm_off.fa_prefill_pm_qprep_fp8_kv, gfx1201, "{arch}");
            let pm_off_on = with(arch, Some("0"), Some("1"));
            assert!(!pm_off_on.fa_prefill_pm, "{arch}");
            assert!(!pm_off_on.fa_prefill_pm_fp8_kv, "{arch}");
            assert!(pm_off_on.fa_prefill_pm_qprep_fp8_kv, "{arch}");
            // Explicit PM and Q-prep values resolve independently for both KV codecs.
            for (pm, want_pm) in [("0", false), ("1", true)] {
                for (qprep, want_qprep) in [("0", false), ("1", true)] {
                    let flags = with(arch, Some(pm), Some(qprep));
                    assert_eq!(
                        flags.fa_prefill_pm, want_pm,
                        "{arch}: pm={pm} qprep={qprep}"
                    );
                    assert_eq!(
                        flags.fa_prefill_pm_fp8_kv, want_pm,
                        "{arch}: fp8 pm={pm} qprep={qprep}"
                    );
                    assert_eq!(
                        flags.fa_prefill_pm_qprep, want_qprep,
                        "{arch}: pm={pm} qprep={qprep}"
                    );
                    assert_eq!(
                        flags.fa_prefill_pm_qprep_fp8_kv, want_qprep,
                        "{arch}: fp8 pm={pm} qprep={qprep}"
                    );
                }
            }
            // Q-prep 0 kills the fold whatever the PM flag says.
            for pm in [None, Some("0"), Some("1")] {
                let flags = with(arch, pm, Some("0"));
                assert!(!flags.fa_prefill_pm_qprep, "{arch}: pm {pm:?}");
                assert!(!flags.fa_prefill_pm_qprep_fp8_kv, "{arch}: fp8 pm {pm:?}");
            }
        }
    }
    #[test]
    fn prefill_iu4_c8_pm_follows_its_qualification_beside_the_resolved_f3_flag() {
        use crate::prefill_route_capabilities::{qualified, PrefillRoute};
        let with = |arch: &str, f3: Option<&'static str>, c8: Option<&'static str>| {
            FeatureFlags::from_lookup(arch, |name| match name {
                "HIPFIRE_PREFILL_IU4_PRODUCER_FUSED" => f3.map(String::from).ok_or(()),
                "HIPFIRE_PREFILL_IU4_C8_PM" => c8.map(String::from).ok_or(()),
                _ => Err(()),
            })
        };
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let gfx1201 = arch == "gfx1201";
            let unset = with(arch, None, None);
            // F3 and the C8 PM route both follow their exact-gfx1201 qualification.
            assert_eq!(
                unset.prefill_iu4_producer_fused,
                qualified(arch, PrefillRoute::PrefillIu4ProducerFused),
                "{arch}"
            );
            assert_eq!(unset.prefill_iu4_producer_fused, gfx1201, "{arch}");
            assert_eq!(
                unset.prefill_iu4_c8_pm,
                qualified(arch, PrefillRoute::PrefillIu4C8Pm),
                "{arch}"
            );
            assert_eq!(unset.prefill_iu4_c8_pm, gfx1201, "{arch}");
            assert!(!FeatureFlags::for_test(arch).prefill_iu4_c8_pm, "{arch}");
            assert!(
                !FeatureFlags::for_test(arch).prefill_iu4_producer_fused,
                "{arch}"
            );
            for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
                let flags = with(arch, None, Some(raw));
                assert_eq!(flags.prefill_iu4_c8_pm, want, "{arch}: {raw}");
                // Independent flag: F3 stays its own resolved qualification.
                assert_eq!(flags.prefill_iu4_producer_fused, gfx1201, "{arch}: {raw}");
            }
            // An explicit F3 0 does not change the C8 PM flag itself (the wrapper nests).
            let f3_off = with(arch, Some("0"), Some("1"));
            assert!(!f3_off.prefill_iu4_producer_fused, "{arch}");
            assert!(f3_off.prefill_iu4_c8_pm, "{arch}");
        }
    }

    #[test]
    fn moe_rotate_i4_pm_follows_its_qualification_and_explicit_overrides() {
        use crate::prefill_route_capabilities::{qualified, PrefillRoute};
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(
                qualified(arch, PrefillRoute::MoeRotateI4Pm),
                arch == "gfx1201",
                "{arch}"
            );
            assert_eq!(unset.moe_rotate_i4_pm, arch == "gfx1201", "{arch}");
            assert!(!FeatureFlags::for_test(arch).moe_rotate_i4_pm, "{arch}");
            for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
                let flags = FeatureFlags::from_lookup(arch, |name| {
                    if name == "HIPFIRE_MOE_ROTATE_I4_PM" {
                        Ok(raw.into())
                    } else {
                        Err(())
                    }
                });
                // The flag is the typed request only; raw inventory is applied at selection.
                assert_eq!(flags.moe_rotate_i4_pm, want, "{arch}: {raw}");
                // No other route flag moves with it.
                assert_eq!(flags.fa_prefill_pm, unset.fa_prefill_pm, "{arch}: {raw}");
                assert_eq!(
                    flags.moe_iu4_wide_tiles, unset.moe_iu4_wide_tiles,
                    "{arch}: {raw}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "HIPFIRE_MOE_ROTATE_I4_PM is not a boolean")]
    fn moe_rotate_i4_pm_rejects_a_non_boolean_spelling() {
        let _ = FeatureFlags::from_lookup("gfx1201", |name| {
            if name == "HIPFIRE_MOE_ROTATE_I4_PM" {
                Ok("on".into())
            } else {
                Err(())
            }
        });
    }

    #[test]
    fn gdn_chunk_prep_pm_follows_its_qualification_with_explicit_override() {
        use crate::prefill_route_capabilities::{qualified, raw_inventory, PrefillRoute};
        for arch in [
            "gfx1201", "gfx1200", "gfx1151", "gfx1100", "gfx942", "unknown",
        ] {
            let unset = FeatureFlags::from_lookup(arch, |_| Err(()));
            assert_eq!(
                unset.gdn_chunk_prep_pm,
                qualified(arch, PrefillRoute::GdnChunkPrepPm)
            );
            assert_eq!(unset.gdn_chunk_prep_pm, arch == "gfx1201", "{arch}");
            assert!(!FeatureFlags::for_test(arch).gdn_chunk_prep_pm);
            for (raw, want) in [("1", true), ("true", true), ("0", false), ("false", false)] {
                let flags = FeatureFlags::from_lookup(arch, |name| {
                    if name == "HIPFIRE_GDN_CHUNK_PREP_PM" {
                        Ok(raw.into())
                    } else {
                        Err(())
                    }
                });
                assert_eq!(flags.gdn_chunk_prep_pm, want, "{arch}: {raw}");
                // Independent of every other route flag: they keep their own resolution.
                assert_eq!(flags.fa_prefill_pm, arch == "gfx1201", "{arch}");
                assert_eq!(flags.fa_prefill_pm_qprep, arch == "gfx1201", "{arch}");
            }
            // An explicit 1 resolves on whatever the architecture; the launcher admission
            // adds the raw inventory (exact gfx1201).
            assert_eq!(
                raw_inventory(arch, PrefillRoute::GdnChunkPrepPm),
                arch == "gfx1201"
            );
        }
    }

    #[test]
    fn gfx1100_split_verify_defaults_only_on_exact_arch_and_keeps_two_opt_outs() {
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let gfx1100 = FeatureFlags::from_process_config("gfx1100", &process);
        assert!(gfx1100.verify_attn);
        assert!(gfx1100.gfx1100_fa2_split_verify);
        for arch in ["gfx1101", "gfx1151", "gfx1200", "gfx1201", "gfx942"] {
            assert!(
                !FeatureFlags::from_process_config(arch, &process).gfx1100_fa2_split_verify,
                "arch={arch}"
            );
        }

        // The parent switch stays independent: dispatch requires both.
        let mut parent_off = ConfigLayer::default();
        parent_off.set_cli("kernel.verify_attn", "false").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer: parent_off,
        }])
        .unwrap();
        let flags = FeatureFlags::from_process_config(
            "gfx1100",
            &ProcessConfig::from_resolved(&resolved).unwrap(),
        );
        assert!(!flags.verify_attn);
        assert!(flags.gfx1100_fa2_split_verify);

        let mut split_off = ConfigLayer::default();
        split_off
            .set_cli("kernel.gfx1100_fa2_split_verify", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer: split_off,
        }])
        .unwrap();
        let flags = FeatureFlags::from_process_config(
            "gfx1100",
            &ProcessConfig::from_resolved(&resolved).unwrap(),
        );
        assert!(flags.verify_attn);
        assert!(!flags.gfx1100_fa2_split_verify);
    }

    #[test]
    fn gfx12_silu_quant_fused_default_on_gfx1201_with_opt_out() {
        // Default process policy: the exact-gfx1201 silu+quant fusion admits
        // on gfx1201 (bit-identical to the standalone quantizer chain);
        // explicit `=0` restores the incumbent producer+quantizer launches.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gfx12_silu_quant_fused);
        assert!(gfx1201.gfx12_silu_quant_fused_enabled());
        for arch in [
            "gfx1100", "gfx1151", "gfx1101", "gfx1102", "gfx1150", "gfx1200", "gfx942",
        ] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx12_silu_quant_fused, "arch={arch}");
            assert!(!flags.gfx12_silu_quant_fused_enabled(), "arch={arch}");
        }

        let mut layer = ConfigLayer::default();
        layer
            .set_cli("kernel.gfx12_silu_quant_fused", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_out = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(!opted_out.gfx12_silu_quant_fused);
        assert!(!opted_out.gfx12_silu_quant_fused_enabled());

        // The unit-test constructor stays fully off (deterministic baseline).
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            let test_flags = FeatureFlags::for_test(arch);
            assert!(!test_flags.gfx12_silu_quant_fused, "arch={arch}");
        }
    }
    #[test]
    fn gfx12_producer_quant_fused_default_on_gfx1201_with_opt_out() {
        // Default process policy: the exact-gfx1201 RMSNorm/rotate/gated-norm
        // +quant fusions admit on gfx1201 (bit-identical to the standalone
        // quantizer chains); explicit `=0` restores the incumbent
        // producer+quantizer launches.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gfx12_producer_quant_fused);
        assert!(gfx1201.gfx12_producer_quant_fused_enabled());
        for arch in [
            "gfx1100", "gfx1151", "gfx1101", "gfx1102", "gfx1150", "gfx1200", "gfx942",
        ] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx12_producer_quant_fused, "arch={arch}");
            assert!(!flags.gfx12_producer_quant_fused_enabled(), "arch={arch}");
        }

        let mut layer = ConfigLayer::default();
        layer
            .set_cli("kernel.gfx12_producer_quant_fused", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_out = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(!opted_out.gfx12_producer_quant_fused);
        assert!(!opted_out.gfx12_producer_quant_fused_enabled());

        // The unit-test constructor stays fully off (deterministic baseline).
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            let test_flags = FeatureFlags::for_test(arch);
            assert!(!test_flags.gfx12_producer_quant_fused, "arch={arch}");
        }
    }
    #[test]
    fn gfx11_producer_quant_fused_default_on_gfx11_with_opt_out() {
        // Default process policy: the gfx11 sigmoid/gated-norm +quant
        // fusions admit on gfx1100/gfx1151 (bit-identical to the standalone
        // quantizer chains); explicit `=0` restores the incumbent
        // producer+quantizer launches.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        for arch in ["gfx1100", "gfx1151"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(flags.gfx11_producer_quant_fused, "arch={arch}");
            assert!(flags.gfx11_producer_quant_fused_enabled(), "arch={arch}");
        }
        for arch in [
            "gfx1201", "gfx1101", "gfx1102", "gfx1150", "gfx1200", "gfx942",
        ] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx11_producer_quant_fused, "arch={arch}");
            assert!(!flags.gfx11_producer_quant_fused_enabled(), "arch={arch}");
        }

        let mut layer = ConfigLayer::default();
        layer
            .set_cli("kernel.gfx11_producer_quant_fused", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_out = FeatureFlags::from_process_config("gfx1100", &process);
        assert!(!opted_out.gfx11_producer_quant_fused);
        assert!(!opted_out.gfx11_producer_quant_fused_enabled());

        // The unit-test constructor stays fully off (deterministic baseline).
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            let test_flags = FeatureFlags::for_test(arch);
            assert!(!test_flags.gfx11_producer_quant_fused, "arch={arch}");
        }
    }
    #[test]
    fn gfx12_fp8_stream_default_on_gfx1201_with_opt_out() {
        // Default process policy: the exact-gfx1201 RMSNorm+rotate → FP8
        // pre-pass fusion is ON on exact gfx1201 only (bit-identical, prepare
        // launches removed); `kernel.gfx12_fp8_stream=false` (or
        // `HIPFIRE_GFX12_FP8_STREAM=0`) opts out; other arches stay off.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let on = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(on.gfx12_fp8_stream && on.gfx12_fp8_stream_enabled());
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gfx12_fp8_stream, "arch={arch}");
            assert!(!flags.gfx12_fp8_stream_enabled(), "arch={arch}");
        }
        let mut off = ConfigLayer::default();
        off.set_cli("kernel.gfx12_fp8_stream", "false").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer: off,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        assert!(!FeatureFlags::from_process_config("gfx1201", &process).gfx12_fp8_stream_enabled());
        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.gfx12_fp8_stream", "true").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let opted_in = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(opted_in.gfx12_fp8_stream);
        assert!(opted_in.gfx12_fp8_stream_enabled());
        let other = FeatureFlags::from_process_config("gfx1151", &process);
        assert!(other.gfx12_fp8_stream);
        assert!(!other.gfx12_fp8_stream_enabled());
        // The unit-test constructor stays fully off (deterministic baseline).
        for arch in ["gfx1201", "gfx1100", "gfx1151"] {
            let test_flags = FeatureFlags::for_test(arch);
            assert!(!test_flags.gfx12_fp8_stream, "arch={arch}");
        }
    }

    #[test]
    fn pm_decode_auto_is_exact_gfx1201_and_follows_typed_key() {
        let flags_for = |arch: &str, raw: Option<&str>| {
            let mut layers = Vec::new();
            if let Some(raw) = raw {
                let mut layer = ConfigLayer::default();
                layer.set_cli("kernel.pm_decode", raw)?;
                layers.push(NamedLayer {
                    source: ConfigSource::GlobalUser {
                        path: "config.toml".into(),
                    },
                    layer,
                });
            }
            let resolved = resolve(layers)?;
            let process = ProcessConfig::from_resolved(&resolved)?;
            Ok::<_, hipfire_config::ConfigError>(FeatureFlags::from_process_config(arch, &process))
        };
        // Unset and `auto`: on for exact gfx1201 only.
        for raw in [None, Some("auto")] {
            assert!(flags_for("gfx1201", raw).unwrap().pm_decode, "{raw:?}");
            for arch in ["gfx1200", "gfx12", "gfx1151", "gfx1100", "gfx942"] {
                assert!(!flags_for(arch, raw).unwrap().pm_decode, "{arch} {raw:?}");
            }
        }
        // `0`/`false` is the opt-out; `1`/`true` is explicit on.
        assert!(!flags_for("gfx1201", Some("0")).unwrap().pm_decode);
        assert!(!flags_for("gfx1201", Some("false")).unwrap().pm_decode);
        assert!(flags_for("gfx1201", Some("1")).unwrap().pm_decode);
        assert!(flags_for("gfx1201", Some("true")).unwrap().pm_decode);
        // Invalid values are refused by the typed schema, not read as a default.
        assert!(flags_for("gfx1201", Some("maybe")).is_err());
        assert!(flags_for("gfx1201", Some("2")).is_err());
        // The unit-test constructor stays off (deterministic baseline).
        assert!(!FeatureFlags::for_test("gfx1201").pm_decode);
    }

    #[test]
    fn iu4_prefill_default_on_gfx11_and_gfx1201() {
        // Default process policy: the iu4-direct MMQ prefill route is on for
        // exact gfx1100/gfx1151 (K16 kernel) and gfx1201 (K32 kernel) and
        // nowhere else; `kernel.iu4_prefill=false` (or
        // `HIPFIRE_IU4_PREFILL=0`) opts out.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(flags.iu4_prefill_enabled(), "arch={arch}");
        }
        for arch in ["gfx1101", "gfx1150", "gfx942"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.iu4_prefill_enabled(), "arch={arch}");
        }

        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.iu4_prefill", "false").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx942"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.iu4_prefill_enabled(), "arch={arch}");
        }

        // The unit-test constructor stays off (deterministic baseline).
        for arch in ["gfx1100", "gfx1151"] {
            let test_flags = FeatureFlags::for_test(arch);
            assert!(!test_flags.iu4_prefill_enabled(), "arch={arch}");
        }
    }

    /// `hipfire_config::effective("kernel.iu4_prefill")` is the Settings/CLI
    /// prediction of the route gate; it must agree with
    /// `FeatureFlags::iu4_prefill_enabled` under the default process policy.
    #[test]
    fn effective_iu4_prefill_matches_iu4_prefill_enabled() {
        use hipfire_config::effective::{effective, A8Prefill, ArchCtx};
        use hipfire_config::ConfigValue;

        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx1030", "gfx1101"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            let ctx = ArchCtx {
                gpu_arch: Some(arch.into()),
                gpu_count: 1,
                model_arch_id: None,
                registry_tag: None,
                a8_prefill: A8Prefill::NotLanded,
            };
            let predicted = effective("kernel.iu4_prefill", &resolved, &ctx);
            assert_eq!(predicted.value, ConfigValue::Bool(true), "arch={arch}");
            assert_eq!(
                predicted.used_on_gpu,
                flags.iu4_prefill_enabled(),
                "arch={arch}"
            );
        }
    }

    #[test]
    fn qkvza_split_tail_defaults_false_in_test_ctor() {
        let f = FeatureFlags::for_test("gfx1100");
        assert!(!f.qkvza_split_tail);
        assert!(!f.gemma4_q8_fused_prefill);
        assert!(!f.gemma4_batched_embedding_prefill);
        assert!(!f.gemma4_ple_batched_prefill);
        assert!(!f.gemma4_ple_branch_batched_prefill);
        assert!(!f.gemma4_ple_activation_fused_prefill);
        assert!(f.flash_attn_ck_lib.is_none());
    }

    #[test]
    fn draft_shared_rotations_default_on_exact_gfx1100_with_independent_opt_out() {
        let keys = [
            (
                "kernel.draft_shared_qkv_rotation",
                "HIPFIRE_DRAFT_SHARED_QKV_ROTATION",
            ),
            (
                "kernel.draft_shared_ffn_rotation",
                "HIPFIRE_DRAFT_SHARED_FFN_ROTATION",
            ),
            (
                "kernel.draft_shared_ctx_kv_rotation",
                "HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION",
            ),
        ];
        let arches = [
            "gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1201", "gfx942",
        ];
        let triple = |f: &FeatureFlags| {
            [
                f.draft_shared_qkv_rotation,
                f.draft_shared_ffn_rotation,
                f.draft_shared_ctx_kv_rotation,
            ]
        };

        // Built-in true values must stay absent across the CLI-to-daemon
        // handoff, so each architecture still selects its own fallback.
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let wire = serde_json::to_string(&process).unwrap();
        let decoded: ProcessConfig = serde_json::from_str(&wire).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, process);
        for (key, env) in keys {
            assert_eq!(
                resolved.get(key).unwrap().value,
                hipfire_config::ConfigValue::Bool(true)
            );
            assert!(process.values.get(key).is_none(), "{key}");
            assert!(decoded.values.get(key).is_none(), "{key}");
            assert_eq!(decoded.legacy_value(env), None, "{env}");
            assert!(!wire.contains(key), "{key} must not be serialized");
        }
        for arch in arches {
            let expected = [arch == "gfx1100"; 3];
            assert_eq!(
                triple(&FeatureFlags::from_process_config(arch, &process)),
                expected,
                "{arch}"
            );
            assert_eq!(
                triple(&FeatureFlags::from_process_config(arch, &decoded)),
                expected,
                "{arch}"
            );
            assert_eq!(
                triple(&FeatureFlags::from_lookup(arch, |_| Err(()))),
                expected,
                "{arch}"
            );
            assert_eq!(
                triple(&FeatureFlags::for_test(arch)),
                [false; 3],
                "{arch}: test baseline"
            );
        }

        // Each switch independently carries both explicit true and false,
        // including true equal to the schema default (which must not vanish).
        for (i, (key, env)) in keys.iter().enumerate() {
            let schema = hipfire_config::field(key).unwrap();
            assert_eq!(schema.env_compat, Some(*env));
            for (enabled, spellings) in [
                (false, ["0", "false", "off", "no", "FALSE", "Off", "NO"]),
                (true, ["1", "true", "on", "yes", "TRUE", "On", "YES"]),
            ] {
                // TOML booleans lower to typed ConfigValue::Bool before set;
                // use that pure API rather than a file or ambient environment.
                let mut toml = ConfigLayer::default();
                toml.set(key, hipfire_config::ConfigValue::Bool(enabled))
                    .unwrap();
                let toml_process = ProcessConfig::from_resolved(
                    &resolve([NamedLayer {
                        source: ConfigSource::GlobalUser {
                            path: "config.toml".into(),
                        },
                        layer: toml,
                    }])
                    .unwrap(),
                )
                .unwrap();

                for raw in spellings {
                    let mut cli = ConfigLayer::default();
                    cli.set_cli(key, raw).unwrap();
                    let mut legacy_cli = ConfigLayer::default();
                    legacy_cli.set_cli(schema.legacy_key, raw).unwrap();
                    let mut environment = ConfigLayer::default();
                    // load_env_layer uses this same schema parser for these
                    // stable aliases; no process-global env mutation needed.
                    environment
                        .set(key, schema.parse_cli(raw).unwrap())
                        .unwrap();
                    for (source, layer) in [
                        (
                            ConfigSource::OneShot {
                                argument: format!("{key}={raw}"),
                            },
                            cli,
                        ),
                        (
                            ConfigSource::OneShot {
                                argument: format!("{}={raw}", schema.legacy_key),
                            },
                            legacy_cli,
                        ),
                        (
                            ConfigSource::LegacyEnv {
                                name: (*env).into(),
                            },
                            environment,
                        ),
                    ] {
                        let resolved = resolve([NamedLayer { source, layer }]).unwrap();
                        let process = ProcessConfig::from_resolved(&resolved).unwrap();
                        assert_eq!(process, toml_process, "{key}={raw}");
                        let wire = serde_json::to_string(&process).unwrap();
                        let decoded: ProcessConfig = serde_json::from_str(&wire).unwrap();
                        decoded.validate().unwrap();
                        assert_eq!(decoded, process, "{key}={raw}");
                        let normalized = if enabled { "1" } else { "0" };
                        assert_eq!(
                            decoded.legacy_value(env).as_deref(),
                            Some(normalized),
                            "{key}={raw}"
                        );
                        for (j, (other_key, other_env)) in keys.iter().enumerate() {
                            if j != i {
                                assert!(
                                    decoded.values.get(other_key).is_none(),
                                    "{key} changed {other_key}"
                                );
                                assert_eq!(
                                    decoded.legacy_value(other_env),
                                    None,
                                    "{key} changed {other_env}"
                                );
                            }
                        }
                        for arch in arches {
                            let mut expected = [arch == "gfx1100"; 3];
                            expected[i] = enabled;
                            assert_eq!(
                                triple(&FeatureFlags::from_process_config(arch, &decoded)),
                                expected,
                                "{arch}: {key}={raw}"
                            );
                            let from_env = FeatureFlags::from_lookup(arch, |name| {
                                if name == *env {
                                    Ok(normalized.into())
                                } else {
                                    Err(())
                                }
                            });
                            assert_eq!(triple(&from_env), expected, "{arch}: {env}={normalized}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gemma4_prefill_auto_defaults_on_validated_arches() {
        let resolved = resolve([]).unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();

        let gfx1100 = FeatureFlags::from_process_config("gfx1100", &process);
        assert!(gfx1100.gemma4_batched_embedding_prefill);
        assert!(gfx1100.gemma4_ple_branch_batched_prefill);
        assert!(gfx1100.gemma4_ple_activation_fused_prefill);
        assert!(!gfx1100.gemma4_q8_fused_prefill);
        assert!(!gfx1100.gemma4_ple_batched_prefill);

        let gfx1201 = FeatureFlags::from_process_config("gfx1201", &process);
        assert!(gfx1201.gemma4_batched_embedding_prefill);
        assert!(gfx1201.gemma4_ple_branch_batched_prefill);
        assert!(gfx1201.gemma4_ple_activation_fused_prefill);
        assert!(!gfx1201.gemma4_q8_fused_prefill);
        assert!(!gfx1201.gemma4_ple_batched_prefill);

        for arch in ["gfx1101", "gfx1102", "gfx1151", "gfx1200"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gemma4_batched_embedding_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_branch_batched_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_activation_fused_prefill, "arch={arch}");
        }
    }

    #[test]
    fn gemma4_prefill_auto_accepts_explicit_opt_out_on_validated_arches() {
        let mut layer = ConfigLayer::default();
        layer
            .set_cli("kernel.gemma4_batched_embedding_prefill", "false")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_branch_batched_prefill", "false")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_activation_fused_prefill", "false")
            .unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        for arch in ["gfx1100", "gfx1201"] {
            let flags = FeatureFlags::from_process_config(arch, &process);
            assert!(!flags.gemma4_batched_embedding_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_branch_batched_prefill, "arch={arch}");
            assert!(!flags.gemma4_ple_activation_fused_prefill, "arch={arch}");
        }
    }

    #[test]
    fn process_config_preserves_explicit_and_arch_default_flags() {
        let mut layer = ConfigLayer::default();
        layer.set_cli("kernel.qkvza_split_tail", "true").unwrap();
        layer
            .set_cli("kernel.gemma4_q8_fused_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_batched_embedding_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_batched_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_branch_batched_prefill", "true")
            .unwrap();
        layer
            .set_cli("kernel.gemma4_ple_activation_fused_prefill", "true")
            .unwrap();
        layer.set_cli("diagnostic.kernel.gemv_rows", "4").unwrap();
        let resolved = resolve([NamedLayer {
            source: ConfigSource::GlobalUser {
                path: "config.toml".into(),
            },
            layer,
        }])
        .unwrap();
        let process = ProcessConfig::from_resolved(&resolved).unwrap();
        let flags = FeatureFlags::from_process_config("gfx1100", &process);

        assert!(flags.qkvza_split_tail);
        assert!(flags.gemma4_q8_fused_prefill);
        assert!(flags.gemma4_batched_embedding_prefill);
        assert!(flags.gemma4_ple_batched_prefill);
        assert!(flags.gemma4_ple_branch_batched_prefill);
        assert!(flags.gemma4_ple_activation_fused_prefill);
        assert_eq!(flags.gemv_rows, Some(4));
        assert!(flags.rdna3_hfq4_qkvza_k2048);
        assert!(flags.rdna3_hfq4_residual_stage_x32);
    }

    #[test]
    fn gfx11_q8_fa2_wide_auto_is_exact_gfx1100_and_gfx1151_with_explicit_overrides() {
        let process = ProcessConfig::from_resolved(&resolve([]).unwrap()).unwrap();
        assert!(process.legacy_value("HIPFIRE_GFX11_Q8_FA2_WIDE").is_none());
        for (arch, expected) in [
            ("gfx1100", true),
            ("gfx1101", false),
            ("gfx1151", true),
            ("gfx1201", false),
        ] {
            assert_eq!(
                FeatureFlags::from_process_config(arch, &process).gfx11_q8_fa2_wide,
                expected,
                "arch={arch}"
            );
        }
        for (value, expected) in [("false", false), ("true", true)] {
            let mut layer = ConfigLayer::default();
            layer.set_cli("kernel.gfx11_q8_fa2_wide", value).unwrap();
            let resolved = resolve([NamedLayer {
                source: ConfigSource::GlobalUser {
                    path: "config.toml".into(),
                },
                layer,
            }])
            .unwrap();
            let process = ProcessConfig::from_resolved(&resolved).unwrap();
            for arch in ["gfx1100", "gfx1151"] {
                assert_eq!(
                    FeatureFlags::from_process_config(arch, &process).gfx11_q8_fa2_wide,
                    expected,
                    "arch={arch} value={value}"
                );
            }
        }
    }
}
