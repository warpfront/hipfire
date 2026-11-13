// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx942 (MI300X) specifics of the all-rows lane segment (PLAN-beat-sglang §3.L, L1).
//!
//! A child module of `gpu_forward`, so `gpu_forward_lanes` keeps only arch-gated call sites.
//!
//! A single-device stock-FP8 lane step of 9..=128 rows (more than one `LANE_MAX_ROWS` group) runs
//! its shared stages ONCE over every row instead of once per <= 8-row group:
//!
//! * the HC reads / writes, embedding, PLE rows, GDN / QSA in / out projections, final hyper
//!   read, LM head and top-1 on the all-rows segment (`HIPFIRE_QWEN4_LANE_HC_ALLROWS`,
//!   `HIPFIRE_QWEN4_LANE_ALLROWS`). The trunk projections are BF16 on the MFMA entry
//!   `gemm_bf16_mfma_gfx942` (the rows16 few-row entry up to 16 rows, the tile entry above, bitwise
//!   equal: CHANGELOG `gemm_bf16_mfma_gfx942_rows16`), whose per-element chain does not depend on
//!   the row count;
//! * the routed MoE as ONE call per layer over every row on its own all-rows segment
//!   (`HIPFIRE_QWEN4_LANE_ALLROWS_MOE`): the FP8 grouped MFMA route
//!   (`gemm_fp8b128_moe_grouped_bf16mfma`: fixed ascending K, 1..16-row tiles, the same kernels
//!   at every row count) with the router / scatter / combine glue per row. Unlike the gfx1201
//!   EP all-rows MoE there is no row gather and no gather-capacity or MoE-call-parity limit:
//!   the call is the singleton route at `n` rows.
//!
//! Everything else (stage proofs, placement, ledger) is the existing generic machinery; the only
//! gfx942 pieces are the admission of the MoE segment without an EP gather
//! ([`moe_refusal`]) and the row bound the receipt covers ([`G942_ALLROWS_MAX_ROWS`]).

use rdna_compute::Gpu;

/// G0 receipt behind the gfx942 all-rows segment. 9..=64 rows: `fn_lane_oracle --g0 --controls
/// --ks 16,32,64 --contexts 64,512,2048 --steps 64` (plus the wide negative controls at k = 16
/// and 64) on the stock-FP8 artifact, with the C0 HC write / read fix (`2028a9d12e`): every
/// lane's logits rows, ids and state checkpoints byte-identical to the singleton under the
/// combined production policy, the all-rows HC / stage / MoE segments engaged (step trace
/// `hc_allrows=true stages_allrows=true moe=allrows`). It proves the combined policy only, like
/// [`GFX942_STOCK_FP8_RECEIPT`](crate::gpu_forward_lanes::GFX942_STOCK_FP8_RECEIPT), and binds
/// the exact-gfx942 all-rows flags, which are always on there. 65..=128 rows (the string's tail):
/// the MTP lane oracle `fn_lane_oracle --spec mtp --k 3 --ks 1,2,8,16,32 --contexts 64,512,2048
/// --steps 8 --controls` and `--k 7 --ks 8,16` (32 x 4 and 16 x 8 = 128-row verify steps) with
/// `HIPFIRE_QWEN4_LANE_STEP_TRACE=1`: every lane's committed ids and trunk / head / draft
/// checkpoints byte-identical to the singleton replay, the trace showing `rows=128
/// hc_allrows=true moe=allrows`.
pub const GFX942_STOCK_FP8_ALLROWS_RECEIPT: &str = "g0 mi300x/beat/L/l1-out/g0-final+ctl-final 2026-11-10 gfx942 stock-fp8 single, all-rows k=16/32/64 ctx 64/512/2048 steps 64, combined policy only, fn_lane_oracle sha256 3699aa99dddfc788 rc=0, with C0 2028a9d12e; 65..=128 rows: mi300x beat/WC8/out/c32/f1-k3+f1-k7 2026-11-13 MTP lanes K=3 ks 1/2/8/16/32 + K=7 ks 8/16 ctx 64/512/2048 steps 8, trace rows=128 hc_allrows moe=allrows, controls PASS, fn_lane_oracle sha256 1b38c0dda74d4318 rc=0, source 2f57c10ecf";

/// Widest all-rows step the gfx942 single-device lane forward admits: 32 MTP lanes x (K = 3 + 1)
/// verify rows ([`rdna_compute::cdna::gfx942::LANE_ALLROWS_MAX_ROWS`]). Non-gfx942 all-rows steps
/// keep the wide-verify bound (`WIDE_VERIFY_MAX_ROWS`, 64).
pub const G942_ALLROWS_MAX_ROWS: usize = rdna_compute::cdna::gfx942::LANE_ALLROWS_MAX_ROWS;

/// The single-device gfx942 all-rows rule: a gfx942 lane step with no EP row gather. Its MoE
/// segment is admitted by [`moe_refusal`] instead of the EP gather checks.
pub(crate) fn single_device_rule(gpu: &Gpu, gather_cap: Option<usize>) -> bool {
    gpu.arch_caps.is_gfx942() && gather_cap.is_none()
}

/// Why a single-device gfx942 step that asked for `HIPFIRE_QWEN4_LANE_ALLROWS_MOE` keeps its MoE
/// on the <= 8-row groups (`Some(reason)`), or `None` when the one all-rows MoE call applies:
/// the stage segment must apply (`stages_apply`), the policy must place the MoE once
/// (`moe_once`), no lane is wide-class (`any_wide_lane`: wide lanes keep their classed group) and
/// `n` is within what the receipt covers.
pub(crate) fn moe_refusal(
    stages_apply: bool,
    moe_once: bool,
    n: usize,
    any_wide_lane: bool,
) -> Option<String> {
    if !stages_apply {
        return Some(
            "the step takes no all-rows stage segment (LANE_ALLROWS / LANE_HC_ALLROWS off, not an eager 9..=128-row multi-group AR / verify step, or a CPU splice / wide scope / GDN qkv BF16 is in force)".into(),
        );
    }
    if !moe_once {
        return Some("the stage policy places the MoE per lane".into());
    }
    if any_wide_lane {
        return Some("a wide-class lane keeps its classed group".into());
    }
    if n > G942_ALLROWS_MAX_ROWS {
        return Some(format!(
            "{n} rows exceed the {G942_ALLROWS_MAX_ROWS}-row bound of the gfx942 all-rows receipt"
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_device_moe_takes_every_row_within_the_receipt_bound() {
        assert_eq!(G942_ALLROWS_MAX_ROWS, 128);
        for n in 9..=G942_ALLROWS_MAX_ROWS {
            assert_eq!(moe_refusal(true, true, n, false), None, "{n}");
        }
        assert!(moe_refusal(true, true, G942_ALLROWS_MAX_ROWS + 1, false).is_some());
        assert!(moe_refusal(true, true, 129, false).is_some());
    }

    #[test]
    fn single_device_moe_keeps_its_groups_without_the_segment_or_the_policy() {
        assert!(moe_refusal(false, true, 16, false).is_some());
        assert!(moe_refusal(true, false, 16, false).is_some());
        assert!(moe_refusal(true, true, 16, true).is_some());
    }

    /// The gfx942 all-rows flags are on by arch only with a receipt that cites the oracle run.
    #[test]
    fn gfx942_all_rows_on_needs_the_receipt() {
        assert!(!GFX942_STOCK_FP8_ALLROWS_RECEIPT.is_empty());
        assert!(G942_ALLROWS_MAX_ROWS >= 128);
    }
}
