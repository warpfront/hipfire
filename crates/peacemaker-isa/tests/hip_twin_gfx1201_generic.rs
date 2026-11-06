// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-free gate of the committed gfx1201 hipcc twins of the generic gated-delta ops
//! (`kernels/gated_delta_vector_pm_gfx1201.hxaco`, sealed by `scripts/pm-hip-twin.sh`): the hxaco
//! re-verifies (lift identity through `emit::checked`, byte-for-byte re-bundle, static hazards,
//! gfx1201 user+system SGPR >= 16), each twin symbol's explicit kernarg layout equals the
//! incumbent's frozen layout, and each twin `.text` is the one the R9700 card judge timed and
//! proved bit-identical (`R2Rda-05-rs-tv8-prio`, `R1Conv-04-rpw32-k4`; `kernels/*.twin.json` carries
//! the same pins and `pm-hip-twin.sh` aborts if a rebuild moves them). The gfx1100 module is the
//! same two bodies; the images differ only by the target.

use peacemaker_isa::arch::Arch;
use peacemaker_isa::hip_twin::{verify_hxaco_many, Sealed};
use sha2::{Digest, Sha256};

const RDA_CONV: &[u8] = include_bytes!("../../../kernels/gated_delta_vector_pm_gfx1201.hxaco");
const RDA_CONV_SHA: &str =
    include_str!("../../../kernels/gated_delta_vector_pm_gfx1201.hxaco.sha256");

const RDA: &str = "gated_delta_net_vector_serial_f32_pm_gfx1201";
const CONV: &str = "causal_conv_depthwise_silu_rows_f32_pm_gfx1201";

/// `.text` sha256 of the candidates as compiled and card-judged on the R9700 (hip_loop, bit-identical
/// to the incumbent at n = 512 / 2048 / 8192, in place).
const RDA_JUDGED_TEXT: &str = "4b0f755c125e8024da75103a4c1c7d6e73bf329815b77ee15e164c67fd1485ec";
const CONV_JUDGED_TEXT: &str = "35985539865d2a7e5a1ea193f1ee2287722fa528f551f115bd4e55f8c8fb8e83";

/// `(offset, size, pointer)` of `n` consecutive 8-byte pointers from offset 0.
fn ptrs(n: u32) -> Vec<(u32, u32, bool)> {
    (0..n).map(|i| (8 * i, 8, true)).collect()
}

/// `(offset, size, pointer)` of `n` consecutive i32 values from `start`.
fn i32s(start: u32, n: u32) -> Vec<(u32, u32, bool)> {
    (0..n).map(|i| (start + 4 * i, 4, false)).collect()
}

/// Explicit (non-hidden) kernargs of the sealed report, in metadata order.
fn explicit(s: &Sealed) -> Vec<(u32, u32, bool)> {
    s.report["abi"]["args"]
        .as_array()
        .expect("abi args")
        .iter()
        .filter(|a| !a["value_kind"].as_str().unwrap().starts_with("hidden_"))
        .map(|a| {
            (
                a["offset"].as_u64().unwrap() as u32,
                a["size"].as_u64().unwrap() as u32,
                a["pointer"].as_bool().unwrap(),
            )
        })
        .collect()
}

fn assert_sha(bytes: &[u8], pinned: &str) {
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        pinned,
        "pinned .sha256 differs"
    );
    assert_eq!(pinned.len(), 64, "sha file must be bare lowercase hex");
}

fn assert_sealed_clean(s: &Sealed) {
    assert_eq!(s.scratch, 0, "{}: scratch", s.report["kernel"]);
    assert_eq!(s.report["hazards"]["checked"], true);
    assert_eq!(s.report["hazards"]["blocking"], 0);
    assert_eq!(s.report["hazards"]["pm_ir_obligations"], 0);
    // User + system SGPRs reach at least 16 (checked by `check_user_system_sgprs` in seal).
    assert!(s.user_sgprs >= 2);
}

#[test]
fn rda_and_conv_gfx1201_module_verifies_with_the_incumbent_kernarg_layouts() {
    assert_sha(RDA_CONV, RDA_CONV_SHA);
    let all = verify_hxaco_many(RDA_CONV, Arch::Gfx1201, &[RDA, CONV]).expect("verify");
    for s in &all {
        assert_sealed_clean(s);
        assert_eq!(s.hxaco, RDA_CONV);
    }
    // gated_delta_net_vector_serial_f32: nine pointers (query, key, value, log_decay, beta, state,
    // output, slot_idx, n_dev), fourteen i32 (n_tokens .. state_row_stride), one u64 slot stride.
    let mut rda = ptrs(9);
    rda.extend(i32s(72, 14));
    rda.push((128, 8, false));
    assert_eq!(explicit(&all[0]), rda);
    assert_eq!(all[0].report["abi"]["kernarg_segment_size"], 136);
    assert_eq!(all[0].lds, 0);
    // causal_conv_depthwise_silu_rows_f32: five pointers (out, in, weight, hist_in, hist_out) and
    // eleven i32 (n_rows .. round_bf16).
    let mut conv = ptrs(5);
    conv.extend(i32s(40, 11));
    assert_eq!(explicit(&all[1]), conv);
    assert_eq!(all[1].report["abi"]["kernarg_segment_size"], 344);
    assert_eq!(all[1].lds, 0);
    assert_ne!(all[0].text_sha256, all[1].text_sha256);
}

#[test]
fn gfx1201_twin_text_is_the_card_judged_candidate() {
    let all = verify_hxaco_many(RDA_CONV, Arch::Gfx1201, &[RDA, CONV]).expect("verify");
    assert_eq!(all[0].text_sha256, RDA_JUDGED_TEXT);
    assert_eq!(all[1].text_sha256, CONV_JUDGED_TEXT);
}

#[test]
fn gfx1201_module_is_refused_as_another_arch_or_symbol() {
    assert!(verify_hxaco_many(RDA_CONV, Arch::Gfx1100, &[RDA, CONV]).is_err());
    assert!(verify_hxaco_many(
        RDA_CONV,
        Arch::Gfx1201,
        &["gated_delta_net_vector_serial_f32_pm_gfx1100"]
    )
    .is_err());
}
