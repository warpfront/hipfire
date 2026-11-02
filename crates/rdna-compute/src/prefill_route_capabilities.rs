// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The ONE capability table of the A3B-class (qwen35 MoE) prefill routes that default on by
//! qualification: [`PrefillRoute`] x architecture. Same two layers as
//! [`crate::expert_capabilities`], deliberately separate:
//!
//! 1. **Raw inventory** ([`raw_inventory`]): whether the route's kernels EXIST on an
//!    architecture. Pure, allocation-free predicates over the architecture string, each
//!    anchored in the real kernel table it mirrors (the arch blocks of
//!    [`crate::kernel_registry::entries`], the exact-architecture predicate the consumer
//!    itself checks, or the symbols of an embedded PeaceMaker bundle); a test cross-checks every
//!    registry anchor against the registry and every bundle anchor against the bundle bytes. It
//!    never consults a qualification, so a kernel that exists on an architecture where it has
//!    not been measured is reported present.
//! 2. **Qualification** ([`qualified`]): the raw inventory AND an explicit, named row of
//!    [`QUALIFIED`] (architecture, route, evidence). A route is on by default only where a row
//!    names it.
//!
//! # What a default means
//!
//! The developer flag of each route (`HIPFIRE_GDN_QK_INDEXED`, `HIPFIRE_GDN_MOE_PRE_FUSED`,
//! `HIPFIRE_GDN_MOE_CHUNK_SCAN`, `HIPFIRE_MOE_ROUTER_FUSED_PREFILL`,
//! `HIPFIRE_MOE_SHARED_PREPARED_IU4`, `HIPFIRE_FA_PREFILL_PM`,
//! `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW`, `HIPFIRE_MOE_COMBINE_V2`, `HIPFIRE_MOE_IU4_WIDE_TILES`,
//! `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`, `HIPFIRE_FA_PREFILL_PM_QPREP`,
//! `HIPFIRE_MOE_ROTATE_I4_PM`, `HIPFIRE_PREFILL_IU4_C8_PM`,
//! `HIPFIRE_GDN_CHUNK_PREP_PM`, `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI`) is a typed boolean.
//! An explicit `1` / `0` always wins; unset follows [`qualified`] of the running device's
//! architecture. The admission predicates (shape, state format, recording, capture, GPU
//! capability) are unchanged and still apply after the default is resolved. The default callers
//! use [`qualified`]; the exact-gfx1201 strings live only in the raw-inventory anchors and the
//! [`QUALIFIED`] rows below.
//!
//! Every route has one exact-gfx1201 [`QUALIFIED`] row, so with its flag unset it is on at
//! exact gfx1201 and off elsewhere; an explicit `0` kills a route on any architecture and an
//! explicit `1` opts in elsewhere only where the callers' raw-inventory conjunct holds.
//! Nested routes retain the parent's resolved flag and admission: `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW`
//! runs only inside the admitted `HIPFIRE_MOE_ROUTER_FUSED_PREFILL` branch,
//! `HIPFIRE_FA_PREFILL_PM_QPREP` needs the *resolved* `HIPFIRE_FA_PREFILL_PM` flag (an explicit
//! PM `0` refuses the fold), and `HIPFIRE_GDN_CHUNK_PREP_PM` runs only inside an admitted
//! `HIPFIRE_GDN_MOE_CHUNK_SCAN` layer (an explicit chunk-scan `0` refuses it). For
//! `HIPFIRE_MOE_IU4_WIDE_TILES` the flag only governs VRAM-resident gate/up (the resident
//! down stays NT4); host-mapped experts keep their existing NT8 selection for both GEMMs
//! whatever the flag says. `HIPFIRE_MOE_ROTATE_I4_PM` stays nested beneath the requested
//! symmetric IU4 route and its frozen C2 producer candidate set (`Gpu::qwen4_moe_rotate128_i4`),
//! and `HIPFIRE_PREFILL_IU4_C8_PM` nests beneath the *resolved*
//! `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` flag. `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI` still applies
//! only where the raw inventory and the GPU-side shape admission hold.

/// A prefill route whose unset default is decided by qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PrefillRoute {
    /// `HIPFIRE_GDN_QK_INDEXED`: the sequential Q8 GDN launch reads the raw key-head Q/K
    /// (`gated_delta_net_q8_batch_seq_indexed`), no repeat/copy.
    GdnQkIndexed,
    /// `HIPFIRE_GDN_MOE_PRE_FUSED`: the MoE GDN preamble runs the fused `gdn_pre_batched_gfx1201`
    /// launch in place of the four-launch sequence.
    GdnMoePreFused,
    /// `HIPFIRE_GDN_MOE_CHUNK_SCAN`: the MoE GDN layer runs the dense chunk-scan route
    /// (`gdn_chunk_prep`, batched KKT solve, BF16 chunk scan) on the MoE scratch.
    GdnMoeChunkScan,
    /// `HIPFIRE_MOE_ROUTER_FUSED_PREFILL`: one fused exact softmax + top-8 + renorm launch
    /// (`moe_topk_renorm_k8_batched`, fused-softmax mode).
    MoeRouterFusedPrefill,
    /// `HIPFIRE_MOE_SHARED_PREPARED_IU4`: the router and shared-expert selector/gate/up consume
    /// one IU4 prequantized activation.
    MoeSharedPreparedIu4,
    /// `HIPFIRE_FA_PREFILL_PM`: A3B H16/KV2/D256 Q8_0 prefill with the
    /// packet-minimal gfx1201 body. Unset follows its exact-gfx1201 qualification.
    FaPrefillPm,
    /// `HIPFIRE_MOE_ROUTER_TOPK_MULTIROW`: the fused router runs one wave32 per row, eight rows
    /// per workgroup (`moe_router_softmax_topk_k8_wave`, same registry module). Nested
    /// beneath [`PrefillRoute::MoeRouterFusedPrefill`]. Unset follows its exact-gfx1201
    /// qualification.
    MoeRouterTopkMultirow,
    /// `HIPFIRE_MOE_COMBINE_V2`: the IU4 grouped-expert combine runs the v2 BF16-input twin
    /// (`moe_down_combine_grouped_bf16in_v2_pm_gfx1201`, eight contiguous columns per lane) in
    /// place of v1. Unset follows its exact-gfx1201 qualification.
    MoeCombineV2,
    /// `HIPFIRE_MOE_IU4_WIDE_TILES`: VRAM-resident symmetric IU4 routed experts take the
    /// NT8 gate/up entry (`qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8`) instead of NT4 when
    /// the per-expert real-slot count is high enough; the resident down entry stays NT4.
    /// Unset follows its exact-gfx1201 qualification.
    MoeIu4WideTiles,
    /// `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED`: exact IU4 producer fusion (RMSNorm / rotate /
    /// SiLU gfx12 producers write the IU4 prequantized activation directly). Unset follows
    /// its exact-gfx1201 qualification.
    PrefillIu4ProducerFused,
    /// `HIPFIRE_FA_PREFILL_PM_QPREP`: the A3B exact Q-prep fold (interleaved projection ->
    /// gate + final f16 Q in one PM pass, consuming angles captured by the K rope launch)
    /// ahead of the PM direct attention. Nested beneath [`PrefillRoute::FaPrefillPm`]. Unset
    /// follows its exact-gfx1201 qualification.
    FaPrefillPmQprep,
    /// `HIPFIRE_MOE_ROTATE_I4_PM`: the down A4 producer `qwen4_moe_rotate128_i4` runs the exact
    /// PeaceMaker twin `moe_rotate_i4_pm_gfx1201` (grid `[P, K/128, 1]`, wave32) instead of the
    /// HIP incumbent. Unset follows its exact-gfx1201 qualification.
    MoeRotateI4Pm,
    /// `HIPFIRE_PREFILL_IU4_C8_PM`: the PeaceMaker C8 rotate / RMSNorm-rotate producer twins
    /// (`iu4_c8_rotate_pm_gfx1201`, `iu4_c8_rmsnorm_rotate_pm_gfx1201`) run in place of the HIP
    /// exact (`cand_mode = 8`) F3 producers. Nested beneath the resolved
    /// [`PrefillRoute::PrefillIu4ProducerFused`] flag. Unset follows its exact-gfx1201
    /// qualification.
    PrefillIu4C8Pm,
    /// `HIPFIRE_GDN_CHUNK_PREP_PM`: the MoE GDN chunk-scan layer runs the PM twin of
    /// `gdn_chunk_prep` (`gdn_chunk_prep_pm_gfx1201`, 16 key / 32 value heads, rows 1..=8192)
    /// in place of the incumbent preamble. Nested beneath [`PrefillRoute::GdnMoeChunkScan`].
    /// Unset follows its exact-gfx1201 qualification.
    GdnChunkPrepPm,
    /// `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI`: the A3B shared-expert MQ4G256V2 down GEMM folds
    /// `residual += sigmoid(gate) * down` into a gfx12 IU4 ADD epilogue (the builder `_b1` entry
    /// `gemm_mq4g256v2_residual_mmq_iu4_full_add_sigmoid_b1` when eligible, else `add == 2` of the
    /// hipcc `gemm_mq4g256v2_residual_mmq_iu4_full_add_v3`), with no intermediate temp, no output
    /// memset and no separate gated residual add. Unset follows its exact-gfx1201 qualification.
    MoeSharedDownSigmoidEpi,
}

impl PrefillRoute {
    /// Every route, in declaration order.
    pub const ALL: [PrefillRoute; 15] = [
        PrefillRoute::GdnQkIndexed,
        PrefillRoute::GdnMoePreFused,
        PrefillRoute::GdnMoeChunkScan,
        PrefillRoute::MoeRouterFusedPrefill,
        PrefillRoute::MoeSharedPreparedIu4,
        PrefillRoute::FaPrefillPm,
        PrefillRoute::MoeRouterTopkMultirow,
        PrefillRoute::MoeCombineV2,
        PrefillRoute::MoeIu4WideTiles,
        PrefillRoute::PrefillIu4ProducerFused,
        PrefillRoute::FaPrefillPmQprep,
        PrefillRoute::MoeRotateI4Pm,
        PrefillRoute::PrefillIu4C8Pm,
        PrefillRoute::GdnChunkPrepPm,
        PrefillRoute::MoeSharedDownSigmoidEpi,
    ];
}

/// One qualified cell: the architecture the route was measured on, and the receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualifiedCell {
    pub arch: &'static str,
    pub route: PrefillRoute,
    /// Byte-oracle, paired-KLD and hold receipts that qualify the cell.
    pub evidence: &'static str,
}

/// The explicit qualification rows. A route is on by default exactly where a row names its
/// architecture AND the raw inventory is present.
pub const QUALIFIED: [QualifiedCell; 15] = [
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::GdnQkIndexed,
        evidence: "a3b-pp-hold/hga-screens-run2 gqk 16/16 byte-identical; hp2-run1; hp2c-run1",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::GdnMoePreFused,
        evidence: "hga-oracles-run1 gpre; hp2k-run1 paired KLD -0.0025; hp2c-run1",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::GdnMoeChunkScan,
        evidence: "a3b-pp-hold/hgc-run1 oracle 792/792 planes byte-identical, pp4096 Ornith 8126->10619 / Qwen 8248->10156 tok/s; hgk-run1 prefill-prompt paired KLD -0.0058 [-0.0100,-0.0016]",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeRouterFusedPrefill,
        evidence: "hga-oracles-run1 rtopk 1272/1272; hp2k-run1 kldseq identical; hp2c-run1",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeSharedPreparedIu4,
        evidence: "hga-oracles-run1 pshared 6/6; hp2k-run1 kldseq identical; hp2c-run1",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::FaPrefillPm,
        evidence: "a3b-pp-hold/s4a-run2 production-launcher oracle 34/34 byte-identical to the certified bundle ee7e1c20; pp4096 Ornith 10565->13629 / Qwen 10084->14006 tok/s; s4b-run1 pp8192 7860->12917; s4k-run1 prefill-prompt paired KLD 64 windows +0.0045 [-0.0031,+0.0122]; s4s-run1 serve",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeRouterTopkMultirow,
        evidence: "a3b-pp-hold/p31-run1 (j1647): stacked prefill-prompt kldseq byte-equal to BASE a152cd40; moe_router_topk_multirow_oracle GPU byte oracle all_byte_identical (p3thin-run1/2 + p31 stack-tip oracles); stacked pp4096 Ornith 13649->15095 / Qwen 13651->15076 tok/s, pp8192 Ornith 12935->13957",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeCombineV2,
        evidence: "a3b-pp-hold/p31-run1 (j1647): stacked prefill-prompt kldseq byte-equal to BASE a152cd40; moe_combine_v2_oracle GPU byte oracle all_byte_identical (p3thin-run1/2 + p31 stack-tip oracles); stacked pp4096 Ornith 13649->15095 / Qwen 13651->15076 tok/s, pp8192 Ornith 12935->13957",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeIu4WideTiles,
        evidence: "a3b-pp-hold/p31-run1 (j1647): stacked prefill-prompt kldseq byte-equal to BASE a152cd40; moe_iu4_wide_tiles_oracle GPU byte oracle all_byte_identical (p3thin-run1/2 + p31 stack-tip oracles); stacked pp4096 Ornith 13649->15095 / Qwen 13651->15076 tok/s, pp8192 Ornith 12935->13957",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::PrefillIu4ProducerFused,
        evidence: "a3b-pp-hold/p31-run1 (j1647): stacked prefill-prompt kldseq byte-equal to BASE a152cd40; prefill_iu4_producer_oracle GPU byte oracle all_byte_identical (p3thin-run1/2 + p31 stack-tip oracles); stacked pp4096 Ornith 13649->15095 / Qwen 13651->15076 tok/s, pp8192 Ornith 12935->13957",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::FaPrefillPmQprep,
        evidence: "a3b-pp-hold/p31-run1 (j1647): stacked prefill-prompt kldseq byte-equal to BASE a152cd40; fa_pm_qprep_oracle GPU byte oracle all_byte_identical (p3thin-run1/2 + p31 stack-tip oracles); stacked pp4096 Ornith 13649->15095 / Qwen 13651->15076 tok/s, pp8192 Ornith 12935->13957",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeRotateI4Pm,
        evidence: "a3b-20k/oracles/L0-001 moe_rotate_i4_pm_oracle all_byte_identical (beta 272e16c5b4); a3b-20k/quality/L0-compare-001 kldseq wt2/agentic x prefill-prompt/per-token byte-equal stock vs four late-H; A3B MQ4RS pp16000 stock 16625.7 -> four late-H 17035.8 tok/s (a3b-20k/runs/beta272-default-001, beta272-fourfold-001)",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::PrefillIu4C8Pm,
        evidence: "a3b-20k/oracles/L0-001 iu4_c8_pm_oracle all_byte_identical (beta 272e16c5b4); a3b-20k/quality/L0-compare-001 kldseq wt2/agentic x prefill-prompt/per-token byte-equal stock vs four late-H; A3B MQ4RS pp16000 stock 16625.7 -> four late-H 17035.8 tok/s (a3b-20k/runs/beta272-default-001, beta272-fourfold-001)",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::GdnChunkPrepPm,
        evidence: "a3b-20k/oracles/L0-001 gdn_chunk_prep_pm_oracle all_byte_identical (beta 272e16c5b4); a3b-20k/quality/L0-compare-001 kldseq wt2/agentic x prefill-prompt/per-token byte-equal stock vs four late-H; A3B MQ4RS pp16000 stock 16625.7 -> four late-H 17035.8 tok/s (a3b-20k/runs/beta272-default-001, beta272-fourfold-001)",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::MoeSharedDownSigmoidEpi,
        evidence: "a3b-20k/oracles/L0-001 shared_down_sigmoid_epi_oracle all_byte_identical (beta 272e16c5b4); a3b-20k/quality/L0-compare-001 kldseq wt2/agentic x prefill-prompt/per-token byte-equal stock vs four late-H; A3B MQ4RS pp16000 stock 16625.7 -> four late-H 17035.8 tok/s (a3b-20k/runs/beta272-default-001, beta272-fourfold-001)",
    },
];

/// The raw kernel inventory of `route` on `arch`: whether its kernels exist, before any
/// qualification. No allocation, no registry build: safe on a hot path.
///
/// * [`PrefillRoute::GdnQkIndexed`]: the `gated_delta_net_q8` registry module (the indexed
///   launcher runs the `gated_delta_net_q8_fast` / `gated_delta_net_q8` symbols with a runtime
///   key-head ratio), in the unconditional section of the registry: every supported
///   architecture ([`crate::kernel_registry::SUPPORTED_ARCHES`]).
/// * [`PrefillRoute::GdnMoePreFused`]: the `gdn_pre_batched_gfx1201` registry module, in the
///   exact-gfx1201 block of the registry.
/// * [`PrefillRoute::GdnMoeChunkScan`]: the four gfx1201 chunk-scan registry modules
///   `gdn_chunk_prep`, `gdn_chunk_prep_fixup`, `gdn_chunk_kkt_solve_batched` and
///   `gdn_chunk_scan_bf16_mseg` (sources `gdn_chunk_scan_prep.gfx1201.hip`,
///   `gdn_chunk_prep_fixup.gfx1201.hip`, `gdn_chunk_scan_kkt_solve.gfx1201.hip`,
///   `gdn_chunk_scan.gfx1201.hip`), all in the exact-gfx1201 block of the registry.
/// * [`PrefillRoute::MoeRouterFusedPrefill`]: the `moe_topk_renorm_k8_batched` registry module,
///   in the registry block of the wave32 architectures (gfx1100, gfx1151, gfx1201), the same
///   wave32 requirement `Gpu::moe_router_fused_prefill_admitted` checks.
/// * [`PrefillRoute::MoeSharedPreparedIu4`]: the gfx1201 IU4 prequant consumers
///   (`Gpu::mq4g256v2_shared_iu4_supported` requires exact gfx1201 WMMA and IU4 prefill).
/// * [`PrefillRoute::FaPrefillPm`]: the exact-gfx1201 embedded PM images selected
///   by `Gpu::attention_q8_0_fa_prefill_pm_gfx1201` (outside the HIP source registry).
/// * [`PrefillRoute::MoeRouterTopkMultirow`]: the same `moe_topk_renorm_k8_batched` registry
///   module and wave32 architectures as [`PrefillRoute::MoeRouterFusedPrefill`] (the wave-per-row
///   symbol `moe_router_softmax_topk_k8_wave` lives in that module); `Gpu::moe_router_topk_multirow_admitted` adds the shape.
/// * [`PrefillRoute::MoeCombineV2`]: the exact-gfx1201 embedded v2 combine bundle
///   `kernels::MOE_SLOT_OPS_V2_PM_GFX1201` (symbol `moe_down_combine_grouped_bf16in_v2_pm_gfx1201`)
///   selected by `moe_slot_ops::moe_slot_ops_v2_kernel` (outside the HIP source registry); a test
///   cross-checks this predicate against that bundle table.
/// * [`PrefillRoute::MoeIu4WideTiles`]: the NT8 gate/up and down exports
///   (`qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8`, `qwen4_moe_down_iu4_sym_pm_gfx1201_nt8`) of
///   the embedded gfx1201 PeaceMaker bundle `kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201`
///   (`kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco`); the gfx1151 bundle has no NT8 export. A test
///   searches the bundle bytes for both symbols and their descriptors. Both are required: the
///   resident route widens gate/up only, the host-mapped route runs both NT8 entries.
/// * [`PrefillRoute::PrefillIu4ProducerFused`]: the exact-gfx1201 gfx12 IU4 producer launchers,
///   outside the standard HIP source registry (JIT on demand, no registry module): RMSNorm
///   (`FUSED_RMSNORM_MQ_ROTATE_I4_GFX12_SRC`, `FUSED_RMSNORM_MQ_ROTATE_AWQ_I4_GFX12_SRC`),
///   SiLU-mul (`FUSED_SILU_MUL_MQ_ROTATE_I4_GFX12_SRC`,
///   `FUSED_SILU_MUL_MQ_ROTATE_AWQ_I4_GFX12_SRC`) and the rotate pair
///   (`MQ_ROTATE_X_I4_GFX12_SRC`, `MQ_ROTATE_X_AWQ_I4_GFX12_SRC`), each admitted only on exact
///   gfx1201.
/// * [`PrefillRoute::FaPrefillPmQprep`]: the exact-gfx1201 embedded `fa_qprep_pm_gfx1201` image
///   run by `Gpu::fa_pm_qprep_exact` plus the angle-capturing mode of the registry
///   `rope_partial_halfsplit_batched` module (all architectures); the PM image alone is
///   exact-gfx1201 only.
/// * [`PrefillRoute::MoeRotateI4Pm`]: the exact-gfx1201 embedded `moe_rotate_i4_pm_gfx1201`
///   image (`kernels::MOE_ROTATE_I4_PM_GFX1201`, symbol `moe_rotate_i4_pm_gfx1201`) run by
///   `Gpu::qwen4_moe_rotate128_i4_pm` (outside the HIP source registry); the incumbent HIP
///   `qwen4_moe_rotate128_i4` module stays the control. A test cross-checks this predicate
///   against `moe_rotate_i4_pm::moe_rotate_i4_pm_kernel`.
/// * [`PrefillRoute::PrefillIu4C8Pm`]: the exact-gfx1201 embedded `iu4_c8_producer_pm_gfx1201`
///   image (`kernels/iu4_c8_producer_pm_gfx1201.hxaco`, exports
///   `iu4_c8_rotate_pm_gfx1201` and `iu4_c8_rmsnorm_rotate_pm_gfx1201`) run by
///   `Gpu::mq_rotate_x_i4_c8_pm` / `Gpu::fused_rmsnorm_mq_rotate_i4_c8_pm` (outside the HIP
///   source registry); a test searches the image bytes for both symbols.
/// * [`PrefillRoute::GdnChunkPrepPm`]: the exact-gfx1201 embedded `gdn_chunk_prep_pm_gfx1201`
///   image run by `Gpu::gdn_chunk_prep_pm` (outside the HIP source registry; a test searches
///   the image for its symbol).
/// * [`PrefillRoute::MoeSharedDownSigmoidEpi`]: the exact-gfx1201 gfx12 V3 IU4 module outside
///   the standard HIP source registry (JIT on demand, no registry module):
///   `kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_V3_SRC`, entry
///   `gemm_mq4g256v2_residual_mmq_iu4_full_add_v3` (runtime `add == 2` with the trailing
///   `shared_gate` pointer; the same fold's `_b1` bundle twin `..._full_add_sigmoid_b1` is the
///   default launch target when the builder bundle is eligible), launched by
///   `Gpu::try_gemm_mq4g256v2_shared_down_sigmoid_epi`,
///   whose consumer `Gpu::mq4g256v2_shared_iu4_supported` requires exact gfx1201. A test pins
///   the source digest and the symbol.
pub fn raw_inventory(arch: &str, route: PrefillRoute) -> bool {
    match route {
        PrefillRoute::GdnQkIndexed => crate::kernel_registry::SUPPORTED_ARCHES.contains(&arch),
        PrefillRoute::GdnMoePreFused => arch == "gfx1201",
        PrefillRoute::GdnMoeChunkScan => arch == "gfx1201",
        PrefillRoute::MoeRouterFusedPrefill => matches!(arch, "gfx1100" | "gfx1151" | "gfx1201"),
        PrefillRoute::MoeSharedPreparedIu4 => arch == "gfx1201",
        PrefillRoute::FaPrefillPm => arch == "gfx1201",
        PrefillRoute::MoeRouterTopkMultirow => matches!(arch, "gfx1100" | "gfx1151" | "gfx1201"),
        PrefillRoute::MoeCombineV2 => arch == "gfx1201",
        PrefillRoute::MoeIu4WideTiles => arch == "gfx1201",
        PrefillRoute::PrefillIu4ProducerFused => arch == "gfx1201",
        PrefillRoute::FaPrefillPmQprep => arch == "gfx1201",
        PrefillRoute::MoeRotateI4Pm => arch == "gfx1201",
        PrefillRoute::PrefillIu4C8Pm => arch == "gfx1201",
        PrefillRoute::GdnChunkPrepPm => arch == "gfx1201",
        PrefillRoute::MoeSharedDownSigmoidEpi => arch == "gfx1201",
    }
}

/// Whether `route` is qualified on `arch`: the raw inventory is present AND an explicit
/// [`QUALIFIED`] row names the cell. This is the unset default of the route's developer flag.
pub fn qualified(arch: &str, route: PrefillRoute) -> bool {
    for cell in &QUALIFIED {
        if cell.arch == arch && cell.route == route {
            return raw_inventory(arch, route);
        }
    }
    false
}

/// The resolved flag of `route` on `arch`: an `explicit` developer value (`Some(true)` for `1`,
/// `Some(false)` for `0`) wins on any architecture, unset (`None`) follows [`qualified`]. The
/// admission predicates of the route still apply after this default.
pub fn enabled(explicit: Option<bool>, arch: &str, route: PrefillRoute) -> bool {
    match explicit {
        Some(explicit) => explicit,
        None => qualified(arch, route),
    }
}

/// Qualification rows of the native fp8-E4M3 KV codec of the two PM attention routes
/// ([`PrefillRoute::FaPrefillPm`], [`PrefillRoute::FaPrefillPmQprep`]). The Q8_0 rows of
/// [`QUALIFIED`] never cover fp8 KV: the fp8 direct body is a different kernel with its own
/// byte oracle and KLD receipts, named below. With the developer flags unset the fp8 cache
/// takes the PM routes on exact gfx1201; an explicit `0` still kills, `1` still opts in
/// where the raw inventory exists.
pub const FP8_KV_QUALIFIED: [QualifiedCell; 2] = [
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::FaPrefillPm,
        evidence: "currente626-dispatch-A-001 terminal-summary all11 handle match GQA8 0 spills; currente626-timing-C-001 PASS (pp4096/pp8192 retained + 12 new decode cells, same-card fp8>=q8 medians); currente626-quality-001 4/4 paired KLD pass; currente626-serve-redline-B-001 j1826 fp8 5/5 PM4 + j1831 redline bit-exact logits/kv/recurrent/gdn; currente626-state-G-fp8-pair-compare001 660/660 PASS (540 raw)",
    },
    QualifiedCell {
        arch: "gfx1201",
        route: PrefillRoute::FaPrefillPmQprep,
        evidence: "same fp8-E4M3 direct body as FaPrefillPm (receipts above); currente626-state-G-fp8-pair-compare001 fp8 same-codec eager-B vs graph-C 660/660 PASS; currente626-quality-001 4/4 paired KLD pass; currente626-serve-redline-B-001 j1831 redline bit-exact kv/recurrent/gdn",
    },
];

/// Whether `route` is qualified for native fp8-E4M3 KV on `arch`: the raw inventory is present
/// AND an explicit [`FP8_KV_QUALIFIED`] row names the cell.
pub fn qualified_fp8_kv(arch: &str, route: PrefillRoute) -> bool {
    for cell in &FP8_KV_QUALIFIED {
        if cell.arch == arch && cell.route == route {
            return raw_inventory(arch, route);
        }
    }
    false
}

/// The resolved flag of `route` on `arch` for native fp8-E4M3 KV: an `explicit` developer value
/// wins on any architecture (admission still applies), unset follows [`qualified_fp8_kv`].
pub fn enabled_fp8_kv(explicit: Option<bool>, arch: &str, route: PrefillRoute) -> bool {
    match explicit {
        Some(explicit) => explicit,
        None => qualified_fp8_kv(arch, route),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCHES: [&str; 6] = [
        "gfx1201", "gfx1151", "gfx1100", "gfx942", "gfx906", "gfx1030",
    ];

    /// The expected raw inventory, written independently of [`raw_inventory`].
    fn expected_raw(arch: &str, route: PrefillRoute) -> bool {
        let gfx1201 = arch == "gfx1201";
        let rdna = matches!(arch, "gfx1151" | "gfx1100");
        match route {
            PrefillRoute::GdnQkIndexed => gfx1201 || rdna || matches!(arch, "gfx942" | "gfx906"),
            PrefillRoute::GdnMoePreFused => gfx1201,
            PrefillRoute::GdnMoeChunkScan => gfx1201,
            PrefillRoute::MoeRouterFusedPrefill => gfx1201 || rdna,
            PrefillRoute::MoeSharedPreparedIu4 => gfx1201,
            PrefillRoute::FaPrefillPm => gfx1201,
            PrefillRoute::MoeRouterTopkMultirow => gfx1201 || rdna,
            PrefillRoute::MoeCombineV2 => gfx1201,
            PrefillRoute::MoeIu4WideTiles => gfx1201,
            PrefillRoute::PrefillIu4ProducerFused => gfx1201,
            PrefillRoute::FaPrefillPmQprep => gfx1201,
            PrefillRoute::MoeRotateI4Pm => gfx1201,
            PrefillRoute::PrefillIu4C8Pm => gfx1201,
            PrefillRoute::GdnChunkPrepPm => gfx1201,
            PrefillRoute::MoeSharedDownSigmoidEpi => gfx1201,
        }
    }

    /// The registry modules a route's raw inventory is anchored in (none for a route without
    /// a registry module).
    fn registry_modules(route: PrefillRoute) -> &'static [&'static str] {
        match route {
            PrefillRoute::GdnQkIndexed => &["gated_delta_net_q8"],
            PrefillRoute::GdnMoePreFused => &["gdn_pre_batched_gfx1201"],
            PrefillRoute::GdnMoeChunkScan => &[
                "gdn_chunk_prep",
                "gdn_chunk_prep_fixup",
                "gdn_chunk_kkt_solve_batched",
                "gdn_chunk_scan_bf16_mseg",
            ],
            PrefillRoute::MoeRouterFusedPrefill => &["moe_topk_renorm_k8_batched"],
            PrefillRoute::MoeSharedPreparedIu4 => &[],
            PrefillRoute::FaPrefillPm => &[],
            PrefillRoute::MoeRouterTopkMultirow => &["moe_topk_renorm_k8_batched"],
            PrefillRoute::MoeCombineV2 => &[],
            PrefillRoute::MoeIu4WideTiles => &[],
            PrefillRoute::PrefillIu4ProducerFused => &[],
            PrefillRoute::FaPrefillPmQprep => &[],
            PrefillRoute::MoeRotateI4Pm => &[],
            PrefillRoute::PrefillIu4C8Pm => &[],
            PrefillRoute::GdnChunkPrepPm => &[],
            PrefillRoute::MoeSharedDownSigmoidEpi => &[],
        }
    }

    /// The four chunk-scan registry modules and the kernel source file each is built from (the
    /// registry source is the file, behind an optional `#define` preamble).
    const CHUNK_MODULE_SOURCES: [(&str, &str); 4] = [
        (
            "gdn_chunk_prep",
            include_str!("../../../kernels/src/gdn_chunk_scan_prep.gfx1201.hip"),
        ),
        (
            "gdn_chunk_prep_fixup",
            include_str!("../../../kernels/src/gdn_chunk_prep_fixup.gfx1201.hip"),
        ),
        (
            "gdn_chunk_kkt_solve_batched",
            include_str!("../../../kernels/src/gdn_chunk_scan_kkt_solve.gfx1201.hip"),
        ),
        (
            "gdn_chunk_scan_bf16_mseg",
            include_str!("../../../kernels/src/gdn_chunk_scan.gfx1201.hip"),
        ),
    ];

    fn registry_has(arch: &str, module: &str) -> bool {
        match crate::kernel_registry::entries(arch, "") {
            Ok(entries) => entries.iter().any(|entry| entry.module == module),
            Err(_) => false,
        }
    }

    #[test]
    fn raw_inventory_matches_the_expected_table_for_every_route_and_arch() {
        for route in PrefillRoute::ALL {
            for arch in ARCHES {
                assert_eq!(
                    raw_inventory(arch, route),
                    expected_raw(arch, route),
                    "raw inventory of {route:?} on {arch}"
                );
            }
        }
    }

    #[test]
    fn raw_inventory_agrees_with_the_kernel_registry_modules() {
        for route in PrefillRoute::ALL {
            for module in registry_modules(route) {
                for arch in ARCHES {
                    assert_eq!(
                        registry_has(arch, module),
                        raw_inventory(arch, route),
                        "registry module {module} vs raw inventory on {arch}"
                    );
                }
            }
        }
    }

    #[test]
    fn chunk_scan_inventory_anchors_all_four_modules_to_their_kernel_sources() {
        let modules = registry_modules(PrefillRoute::GdnMoeChunkScan);
        assert_eq!(modules.len(), CHUNK_MODULE_SOURCES.len());
        for (module, expected) in CHUNK_MODULE_SOURCES {
            assert!(
                modules.contains(&module),
                "{module} is not an anchored module"
            );
            for arch in ARCHES {
                let entries = crate::kernel_registry::entries(arch, "");
                let found = entries
                    .iter()
                    .flatten()
                    .find(|entry| entry.module == module);
                assert_eq!(
                    found.is_some(),
                    raw_inventory(arch, PrefillRoute::GdnMoeChunkScan),
                    "{module} registered vs raw inventory on {arch}"
                );
                if let Some(entry) = found {
                    assert!(
                        entry.source().ends_with(expected),
                        "{module} on {arch} is not built from its kernel source file"
                    );
                }
            }
        }
    }

    #[test]
    fn qualification_is_exact_gfx1201_today_for_every_route() {
        for route in PrefillRoute::ALL {
            for arch in ARCHES {
                assert_eq!(
                    qualified(arch, route),
                    arch == "gfx1201",
                    "qualification of {route:?} on {arch}"
                );
            }
        }
    }

    #[test]
    fn an_explicit_flag_wins_and_unset_follows_qualification() {
        for route in PrefillRoute::ALL {
            for arch in ARCHES {
                assert!(enabled(Some(true), arch, route));
                assert!(!enabled(Some(false), arch, route));
                assert_eq!(enabled(None, arch, route), qualified(arch, route));
            }
        }
    }

    #[test]
    fn every_route_has_one_qualified_row_and_every_row_has_inventory() {
        for (i, cell) in QUALIFIED.iter().enumerate() {
            assert!(
                raw_inventory(cell.arch, cell.route),
                "row {i} is qualified without raw inventory"
            );
            assert!(!cell.evidence.is_empty(), "row {i} has no evidence");
        }
        for route in PrefillRoute::ALL {
            let mut rows = 0;
            for cell in &QUALIFIED {
                if cell.route == route {
                    rows += 1;
                }
            }
            assert_eq!(rows, 1, "{route:?} qualified row count");
        }
    }

    #[test]
    fn raw_inventory_beyond_the_qualified_cell_is_not_qualified() {
        let router = PrefillRoute::MoeRouterFusedPrefill;
        let multirow = PrefillRoute::MoeRouterTopkMultirow;
        for route in [router, multirow] {
            for arch in ["gfx1151", "gfx1100"] {
                assert!(raw_inventory(arch, route), "{arch} {route:?}");
                assert!(!qualified(arch, route), "{arch} {route:?}");
                assert!(!enabled(None, arch, route), "{arch} {route:?}");
                assert!(enabled(Some(true), arch, route), "{arch} {route:?}");
            }
        }
        for arch in ["gfx1151", "gfx1100", "gfx942", "gfx906", "gfx1030"] {
            assert!(
                !raw_inventory(arch, PrefillRoute::PrefillIu4ProducerFused),
                "{arch}"
            );
            assert!(!raw_inventory(arch, PrefillRoute::MoeRotateI4Pm), "{arch}");
            assert!(!raw_inventory(arch, PrefillRoute::GdnChunkPrepPm), "{arch}");
        }
        let qk = PrefillRoute::GdnQkIndexed;
        assert!(raw_inventory("gfx1100", qk));
        assert!(!qualified("gfx1100", qk));
        assert!(raw_inventory("gfx942", qk));
        assert!(!qualified("gfx942", qk));
        assert!(!raw_inventory("gfx1030", qk));
        let chunk = PrefillRoute::GdnMoeChunkScan;
        assert!(!raw_inventory("gfx1100", chunk));
        assert!(!qualified("gfx1100", chunk));
    }

    #[test]
    fn chunk_prep_pm_is_qualified_on_exact_gfx1201_and_nests_beneath_the_chunk_scan_route() {
        let route = PrefillRoute::GdnChunkPrepPm;
        assert_eq!(
            PrefillRoute::ALL[13],
            route,
            "appended at the end, earlier order unchanged"
        );
        assert_eq!(
            QUALIFIED.iter().filter(|cell| cell.route == route).count(),
            1
        );
        for arch in ARCHES {
            assert_eq!(raw_inventory(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(qualified(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(enabled(None, arch, route), arch == "gfx1201", "{arch}");
            assert!(!enabled(Some(false), arch, route), "{arch}: 0 kills");
            assert!(enabled(Some(true), arch, route), "{arch}: 1 opts in");
            // The raw inventory is the embedded image and is exactly the chunk-scan cell it
            // nests beneath; the chunk scan itself stays qualified there.
            assert_eq!(
                raw_inventory(arch, route),
                raw_inventory(arch, PrefillRoute::GdnMoeChunkScan),
                "{arch}"
            );
            assert_eq!(
                qualified(arch, PrefillRoute::GdnMoeChunkScan),
                arch == "gfx1201",
                "{arch}"
            );
        }
        let image = crate::gdn_chunk_prep_pm::IMAGE;
        let symbol = crate::gdn_chunk_prep_pm::SYMBOL.as_bytes();
        assert!(
            image.windows(symbol.len()).any(|w| w == symbol),
            "image lacks its symbol"
        );
        assert!(
            !registry_has("gfx1201", crate::gdn_chunk_prep_pm::MODULE),
            "the PM twin is outside the HIP source registry"
        );
    }

    #[test]
    fn rotate_i4_pm_is_qualified_exactly_where_its_bundle_exists() {
        fn contains(haystack: &[u8], needle: &[u8]) -> bool {
            !needle.is_empty()
                && haystack
                    .windows(needle.len())
                    .any(|window| window == needle)
        }
        let route = PrefillRoute::MoeRotateI4Pm;
        assert_eq!(PrefillRoute::ALL[11], route);
        assert_eq!(
            QUALIFIED.iter().filter(|cell| cell.route == route).count(),
            1
        );
        for arch in ARCHES {
            // The raw inventory is established by the embedded object itself: the arch's bundle
            // exists, carries the exported symbol and its kernel descriptor.
            let bundle = crate::moe_rotate_i4_pm::moe_rotate_i4_pm_kernel(arch);
            let carried = bundle.is_some_and(|(image, symbol)| {
                contains(image, symbol.as_bytes())
                    && contains(image, format!("{symbol}.kd").as_bytes())
            });
            assert_eq!(raw_inventory(arch, route), carried, "{arch}");
            assert_eq!(raw_inventory(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(qualified(arch, route), carried, "{arch}");
            assert_eq!(enabled(None, arch, route), carried, "{arch}");
            assert!(enabled(Some(true), arch, route), "{arch}: 1 opts in");
            assert!(!enabled(Some(false), arch, route), "{arch}: 0 kills");
            if let Some((image, symbol)) = bundle {
                assert!(!image.is_empty(), "{arch}: empty rotate bundle");
                assert_eq!(image, crate::kernels::MOE_ROTATE_I4_PM_GFX1201);
                assert_eq!(
                    symbol,
                    crate::moe_rotate_i4_pm::MOE_ROTATE_I4_PM_GFX1201_SYMBOL
                );
            }
        }
    }

    #[test]
    fn iu4_c8_pm_is_qualified_on_exact_gfx1201() {
        let route = PrefillRoute::PrefillIu4C8Pm;
        assert_eq!(PrefillRoute::ALL[12], route, "appended at the end of ALL");
        assert_eq!(
            QUALIFIED.iter().filter(|cell| cell.route == route).count(),
            1
        );
        for arch in ARCHES {
            assert_eq!(raw_inventory(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(qualified(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(enabled(None, arch, route), arch == "gfx1201", "{arch}");
            assert!(enabled(Some(true), arch, route), "{arch}: 1 opts in");
            assert!(!enabled(Some(false), arch, route), "{arch}: 0 kills");
        }
    }

    #[test]
    fn shared_down_sigmoid_epi_is_qualified_on_exact_gfx1201() {
        let route = PrefillRoute::MoeSharedDownSigmoidEpi;
        assert_eq!(PrefillRoute::ALL[14], route, "appended at the end of ALL");
        assert_eq!(
            QUALIFIED.iter().filter(|cell| cell.route == route).count(),
            1
        );
        for arch in ARCHES {
            assert_eq!(
                raw_inventory(arch, route),
                arch == "gfx1201",
                "raw inventory on {arch}"
            );
            assert_eq!(qualified(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(enabled(None, arch, route), arch == "gfx1201", "{arch}");
            assert!(!enabled(Some(false), arch, route), "0 on {arch}");
            assert!(enabled(Some(true), arch, route), "1 on {arch}");
        }
    }

    #[test]
    fn shared_down_sigmoid_epi_anchors_the_v3_source_and_symbol() {
        use sha2::{Digest, Sha256};
        let source = crate::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_V3_SRC;
        assert!(
            source.contains("gemm_mq4g256v2_residual_mmq_iu4_full_add_v3"),
            "V3 module lacks the full_add_v3 entry"
        );
        assert!(
            source.contains("shared_gate"),
            "V3 module lacks the shared_gate argument"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(source.as_bytes())),
            "5a845a9eea866a6617ae4d59efe04f992c9685cce12382643e32469f6af332e9",
            "V3 module source changed: re-run the owner GPU byte oracle, then update this pin"
        );
    }

    /// The five phase-3 routes: position in `ALL` and the oracle named by the row evidence.
    const PHASE3: [(usize, PrefillRoute, &str); 5] = [
        (
            6,
            PrefillRoute::MoeRouterTopkMultirow,
            "moe_router_topk_multirow_oracle",
        ),
        (7, PrefillRoute::MoeCombineV2, "moe_combine_v2_oracle"),
        (
            8,
            PrefillRoute::MoeIu4WideTiles,
            "moe_iu4_wide_tiles_oracle",
        ),
        (
            9,
            PrefillRoute::PrefillIu4ProducerFused,
            "prefill_iu4_producer_oracle",
        ),
        (10, PrefillRoute::FaPrefillPmQprep, "fa_pm_qprep_oracle"),
    ];

    #[test]
    fn phase3_routes_are_qualified_on_exact_gfx1201_with_their_own_evidence() {
        assert_eq!(PrefillRoute::ALL.len(), 15);
        assert_eq!(
            QUALIFIED.len(),
            15,
            "six prior rows, the five phase-3 rows and the four late-H rows"
        );
        for (index, route, oracle) in PHASE3 {
            assert_eq!(PrefillRoute::ALL[index], route, "ALL order is unchanged");
            let mut rows = QUALIFIED.iter().filter(|cell| cell.route == route);
            let cell = rows.next().expect("one qualified row");
            assert!(rows.next().is_none(), "{route:?} has exactly one row");
            assert_eq!(cell.arch, "gfx1201", "{route:?}");
            for needle in [
                "a3b-pp-hold/p31-run1 (j1647)",
                "BASE a152cd40",
                oracle,
                "all_byte_identical",
                "p3thin-run1/2",
                "Ornith 13649->15095 / Qwen 13651->15076",
                "pp8192 Ornith 12935->13957",
            ] {
                assert!(
                    cell.evidence.contains(needle),
                    "{route:?} evidence lacks {needle}"
                );
            }
            for arch in ARCHES {
                assert_eq!(
                    qualified(arch, route),
                    arch == "gfx1201",
                    "{route:?} on {arch}"
                );
                assert_eq!(
                    enabled(None, arch, route),
                    arch == "gfx1201",
                    "{route:?} on {arch}"
                );
                assert!(
                    enabled(Some(true), arch, route),
                    "{route:?} {arch}: 1 opts in"
                );
                assert!(
                    !enabled(Some(false), arch, route),
                    "{route:?} {arch}: 0 kills"
                );
            }
        }
    }

    #[test]
    fn topk_multirow_shares_the_fused_router_inventory_and_qualifies_only_on_gfx1201() {
        let multirow = PrefillRoute::MoeRouterTopkMultirow;
        let fused = PrefillRoute::MoeRouterFusedPrefill;
        for arch in ARCHES {
            assert_eq!(
                raw_inventory(arch, multirow),
                raw_inventory(arch, fused),
                "{arch}"
            );
            assert_eq!(qualified(arch, multirow), arch == "gfx1201", "{arch}");
            assert_eq!(qualified(arch, fused), arch == "gfx1201", "{arch}");
        }
        assert!(raw_inventory("gfx1201", multirow));
        assert!(!raw_inventory("gfx942", multirow));
    }

    #[test]
    fn combine_v2_is_qualified_exactly_where_its_bundle_exists() {
        let route = PrefillRoute::MoeCombineV2;
        for arch in ARCHES {
            let bundle = crate::moe_slot_ops::moe_slot_ops_v2_kernel(arch);
            assert_eq!(raw_inventory(arch, route), bundle.is_some(), "{arch}");
            assert_eq!(raw_inventory(arch, route), arch == "gfx1201", "{arch}");
            assert_eq!(qualified(arch, route), bundle.is_some(), "{arch}");
            if let Some((image, symbol)) = bundle {
                assert!(!image.is_empty(), "{arch}: empty v2 bundle");
                assert_eq!(image, crate::kernels::MOE_SLOT_OPS_V2_PM_GFX1201);
                assert_eq!(
                    symbol,
                    crate::moe_slot_ops::MOE_DOWN_COMBINE_BF16IN_V2_PM_GFX1201_SYMBOL
                );
            }
        }
    }

    #[test]
    fn qprep_is_qualified_on_the_same_cell_as_the_pm_route_it_nests_beneath() {
        let qprep = PrefillRoute::FaPrefillPmQprep;
        let pm = PrefillRoute::FaPrefillPm;
        for arch in ARCHES {
            assert_eq!(raw_inventory(arch, qprep), arch == "gfx1201", "{arch}");
            assert_eq!(qualified(arch, qprep), qualified(arch, pm), "{arch}");
            // An unset PM flag resolves on exactly where the fold does; an explicit PM `0`
            // is resolved off here and the caller refuses the fold (`PmDisabled`).
            assert_eq!(enabled(None, arch, pm), arch == "gfx1201", "{arch}");
            assert!(!enabled(Some(false), arch, pm), "{arch}");
            assert_eq!(enabled(None, arch, qprep), arch == "gfx1201", "{arch}");
        }
    }

    #[test]
    fn fp8_kv_has_its_own_qualification_receipts() {
        for route in [PrefillRoute::FaPrefillPm, PrefillRoute::FaPrefillPmQprep] {
            for arch in ARCHES {
                assert_eq!(
                    qualified_fp8_kv(arch, route),
                    FP8_KV_QUALIFIED
                        .iter()
                        .any(|cell| cell.arch == arch && cell.route == route)
                        && raw_inventory(arch, route),
                    "{arch}: {route:?}"
                );
                // The Q8_0 qualification never leaks into fp8 KV; explicit values still win.
                assert_eq!(
                    enabled_fp8_kv(None, arch, route),
                    arch == "gfx1201",
                    "{arch}: {route:?}"
                );
                assert!(enabled_fp8_kv(Some(true), arch, route), "{arch}: {route:?}");
                assert!(
                    !enabled_fp8_kv(Some(false), arch, route),
                    "{arch}: {route:?}"
                );
            }
            assert!(qualified("gfx1201", route));
        }
    }
}

#[cfg(test)]
mod moe_iu4_wide_tiles_bundle_tests {
    use super::*;

    const ARCHES: [&str; 6] = [
        "gfx1201", "gfx1151", "gfx1100", "gfx942", "gfx906", "gfx1030",
    ];
    const ROUTE: PrefillRoute = PrefillRoute::MoeIu4WideTiles;

    fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        !needle.is_empty()
            && haystack
                .windows(needle.len())
                .any(|window| window == needle)
    }

    /// The embedded PeaceMaker IU4 symmetric MoE bundle of `arch` (None: no such bundle).
    fn bundle(arch: &str) -> Option<&'static [u8]> {
        match arch {
            "gfx1201" => Some(crate::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201),
            "gfx1151" => Some(crate::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151),
            _ => None,
        }
    }

    /// `[gate/up, down]` NT8 symbol names of the bundle of `arch`, by the bundle's own naming.
    fn nt8_symbols(arch: &str) -> [String; 2] {
        [
            format!("qwen4_moe_gate_up_silu_iu4_sym_pm_{arch}_nt8"),
            format!("qwen4_moe_down_iu4_sym_pm_{arch}_nt8"),
        ]
    }

    #[test]
    fn raw_inventory_is_exactly_the_gfx1201_bundle_carrying_both_nt8_entries() {
        for arch in ARCHES {
            let carried = bundle(arch).is_some_and(|image| {
                nt8_symbols(arch).iter().all(|symbol| {
                    contains(image, symbol.as_bytes())
                        && contains(image, format!("{symbol}.kd").as_bytes())
                })
            });
            assert_eq!(raw_inventory(arch, ROUTE), carried, "{arch}");
        }
        assert!(raw_inventory("gfx1201", ROUTE));
    }

    #[test]
    fn gfx1201_bundle_carries_the_nt4_and_nt8_entries_as_distinct_exports() {
        let image = crate::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201;
        let [gate_up_nt8, down_nt8] = nt8_symbols("gfx1201");
        for (nt4, nt8) in [
            ("qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4", gate_up_nt8),
            ("qwen4_moe_down_iu4_sym_pm_gfx1201_nt4", down_nt8),
        ] {
            assert_ne!(nt4, nt8);
            for symbol in [nt4, nt8.as_str()] {
                assert!(contains(image, symbol.as_bytes()), "{symbol} missing");
                assert!(
                    contains(image, format!("{symbol}.kd").as_bytes()),
                    "{symbol}.kd missing"
                );
            }
        }
    }

    #[test]
    fn gfx1151_bundle_has_no_nt8_entry_and_no_other_arch_is_inventoried() {
        let image = crate::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151;
        for symbol in nt8_symbols("gfx1151") {
            assert!(
                !contains(image, symbol.as_bytes()),
                "{symbol} unexpectedly present"
            );
        }
        assert!(!contains(image, b"_nt8"));
        for arch in ARCHES {
            assert_eq!(raw_inventory(arch, ROUTE), arch == "gfx1201", "{arch}");
        }
    }

    #[test]
    fn wide_tiles_are_qualified_exactly_where_the_nt8_bundle_is_inventoried() {
        assert_eq!(
            QUALIFIED.iter().filter(|cell| cell.route == ROUTE).count(),
            1
        );
        for arch in ARCHES {
            assert_eq!(qualified(arch, ROUTE), raw_inventory(arch, ROUTE), "{arch}");
            assert_eq!(enabled(None, arch, ROUTE), arch == "gfx1201", "{arch}");
            assert!(enabled(Some(true), arch, ROUTE), "{arch}");
            assert!(!enabled(Some(false), arch, ROUTE), "{arch}");
        }
        assert!(PrefillRoute::ALL.contains(&ROUTE));
        assert_eq!(PrefillRoute::ALL[8], ROUTE);
    }
}
