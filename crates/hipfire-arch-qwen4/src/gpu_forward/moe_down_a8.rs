// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-time verdict of the int8 routed-expert DOWN route (`HIPFIRE_QWEN4_MOE_DOWN_A8`, gfx1201).
//!
//! The A8 kernels read every qt53 down expert as `w = s * (q - 8)`, so the route is valid only when
//! every group header of every routed MoE layer the device serves satisfies `z == -8 * s`. The
//! verdict reuses the per-layer symmetric flag of [`super::Qwen4MoeLayerRuntime::verify_symmetric`]
//! (a layer is symmetric only when its gate/up AND down tables passed, drop-zero planar layers by
//! construction), which covers every routed trunk layer including the S1 host-mapped ones, whose
//! tables hold the device-visible aliases of the pinned payload. One layer that is not symmetric (or
//! was never checked) keeps the incumbent routed down on the whole `Gpu`, so `lane == singleton`
//! cannot split between a verified and an unverified layer.

use rdna_compute::Gpu;

/// Outcome of the per-device verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoeDownA8Verdict {
    /// Not requested: `killed` when `HIPFIRE_QWEN4_MOE_DOWN_A8=0` turned it off on a gfx1201
    /// card, else the card is not gfx1201.
    NotRequested { killed: bool },
    /// The device serves no routed MoE layer.
    NoLayers,
    /// The load-time header check did not run (`HIPFIRE_QWEN4_MOE_SYM_IU4=0`, an S1 `Aggregate`
    /// profile, stock FP8): no layer carries a verdict.
    NotChecked,
    /// All `n` layers verified symmetric.
    Verified { n: usize },
    /// Only `verified` of `total` layers verified symmetric.
    Failed { verified: usize, total: usize },
}

impl MoeDownA8Verdict {
    pub(crate) fn verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

/// Pure verdict from the load-time check: `requested` (flag + exact gfx1201), `killed` (the
/// device is gfx1201 and the flag is false), `checked` (the symmetric header check ran on the
/// layers), and the `(verified, total)` layer counts.
pub(crate) fn moe_down_a8_verdict(
    requested: bool,
    killed: bool,
    checked: bool,
    verified: usize,
    total: usize,
) -> MoeDownA8Verdict {
    if !requested {
        return MoeDownA8Verdict::NotRequested { killed };
    }
    if total == 0 {
        return MoeDownA8Verdict::NoLayers;
    }
    if !checked {
        return MoeDownA8Verdict::NotChecked;
    }
    if verified == total {
        MoeDownA8Verdict::Verified { n: total }
    } else {
        MoeDownA8Verdict::Failed { verified, total }
    }
}

/// The one INFO line of a verdict, `None` when none is printed. `head_pending`: the MTP head's
/// MoE layer verdict is still owed, so the route stays off until it settles.
pub(crate) fn moe_down_a8_info(
    verdict: &MoeDownA8Verdict,
    device: i32,
    head_pending: bool,
) -> Option<String> {
    match verdict {
        MoeDownA8Verdict::NotRequested { killed: true } => Some(format!(
            "qwen4 moe down A8: killed by HIPFIRE_QWEN4_MOE_DOWN_A8=0 on device {device}; \
             int8 routed-down route off (incumbent down)"
        )),
        MoeDownA8Verdict::NotRequested { killed: false } | MoeDownA8Verdict::NoLayers => None,
        MoeDownA8Verdict::NotChecked => Some(format!(
            "qwen4 moe down A8: routed expert headers were not checked on device {device} \
             (symmetric header check did not run); int8 routed-down route off"
        )),
        MoeDownA8Verdict::Verified { n } => Some(format!(
            "qwen4 moe down A8: {n}/{n} MoE layers verified symmetric on device {device}; \
             int8 routed-down route on{} (HIPFIRE_QWEN4_MOE_DOWN_A8=0 kills it)",
            if head_pending {
                " once the MTP head layer is verified"
            } else {
                ""
            }
        )),
        MoeDownA8Verdict::Failed { verified, total } => Some(format!(
            "qwen4 moe down A8: {verified}/{total} MoE layers verified symmetric on device \
             {device}; int8 routed-down route off (incumbent down)"
        )),
    }
}

/// The INFO line of the MTP head layer's verdict.
pub(crate) fn moe_down_a8_head_info(symmetric: bool, trunk_verified: bool, device: i32) -> String {
    match (symmetric, trunk_verified) {
        (true, true) => format!(
            "qwen4 moe down A8: MTP head MoE layer verified symmetric on device {device}; \
             int8 routed-down route on (HIPFIRE_QWEN4_MOE_DOWN_A8=0 kills it)"
        ),
        (true, false) => format!(
            "qwen4 moe down A8: MTP head MoE layer verified symmetric on device {device}; \
             int8 routed-down route stays off (trunk not verified)"
        ),
        (false, _) => format!(
            "qwen4 moe down A8: MTP head MoE layer is not symmetric on device {device}; \
             int8 routed-down route off (incumbent down)"
        ),
    }
}

/// Take the device's verdict from the per-layer symmetric flags (`verified` of `total`), set
/// [`Gpu::set_qwen4_moe_down_a8_verified`] and the head-pending state, print the INFO line.
/// `head_moe`: the MTP head carries a MoE layer whose verdict [`settle_head`] owes later.
pub(crate) fn settle_trunk(
    gpu: &mut Gpu,
    checked: bool,
    verified: usize,
    total: usize,
    head_moe: bool,
) -> MoeDownA8Verdict {
    let device = gpu.device_id;
    let verdict = moe_down_a8_verdict(
        gpu.qwen4_moe_down_a8_requested(),
        gpu.arch == "gfx1201" && gpu.qwen4_moe_down_a8_explicitly_off(),
        checked,
        verified,
        total,
    );
    gpu.set_qwen4_moe_down_a8_verified(verdict.verified());
    let head_pending = verdict.verified() && head_moe;
    gpu.set_qwen4_moe_down_a8_head_pending(head_pending);
    if let Some(line) = moe_down_a8_info(&verdict, device, head_pending) {
        eprintln!("[INFO] {line}");
    }
    verdict
}

/// Settle the MTP head's MoE layer verdict (called once after the head is built, before any
/// decode): verify the head layer's routed tables and fold the result into the device verdict.
/// A no-op unless the trunk verdict left the head pending. The head layer's own `symmetric` flag
/// (the IU4 prefill selector) is left untouched; `config` is the head geometry
/// (`Qwen4Config::mtp_geometry`).
pub(crate) fn settle_mtp_head(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    head: Option<&mut super::Qwen4MoeLayerRuntime>,
) -> Result<(), super::Qwen4GpuForwardError> {
    if !gpu.qwen4_moe_down_a8_head_pending() {
        return Ok(());
    }
    let symmetric = match head {
        Some(layer) => {
            let saved = layer.symmetric;
            let verdict = layer.verify_symmetric(gpu, config);
            layer.symmetric = saved;
            verdict?
        }
        None => true,
    };
    settle_head(gpu, symmetric);
    Ok(())
}

/// Take the MTP head MoE layer's verdict: a non-symmetric head turns the route off device-wide;
/// either way the pending state is settled.
pub(crate) fn settle_head(gpu: &mut Gpu, symmetric: bool) {
    let device = gpu.device_id;
    let trunk_verified = gpu.qwen4_moe_down_a8_is_verified();
    if !symmetric {
        gpu.set_qwen4_moe_down_a8_verified(false);
    }
    gpu.set_qwen4_moe_down_a8_head_pending(false);
    if gpu.qwen4_moe_down_a8_requested() && trunk_verified {
        eprintln!(
            "[INFO] {}",
            moe_down_a8_head_info(symmetric, trunk_verified, device)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_table() {
        assert_eq!(
            moe_down_a8_verdict(false, true, true, 40, 40),
            MoeDownA8Verdict::NotRequested { killed: true }
        );
        assert_eq!(
            moe_down_a8_verdict(false, false, true, 40, 40),
            MoeDownA8Verdict::NotRequested { killed: false }
        );
        assert_eq!(
            moe_down_a8_verdict(true, false, true, 0, 0),
            MoeDownA8Verdict::NoLayers
        );
        assert_eq!(
            moe_down_a8_verdict(true, false, false, 0, 40),
            MoeDownA8Verdict::NotChecked
        );
        assert_eq!(
            moe_down_a8_verdict(true, false, true, 40, 40),
            MoeDownA8Verdict::Verified { n: 40 }
        );
        assert_eq!(
            moe_down_a8_verdict(true, false, true, 39, 40),
            MoeDownA8Verdict::Failed {
                verified: 39,
                total: 40
            }
        );
    }

    #[test]
    fn only_all_layers_verified_is_verified() {
        assert!(MoeDownA8Verdict::Verified { n: 1 }.verified());
        assert!(!MoeDownA8Verdict::NoLayers.verified());
        assert!(!MoeDownA8Verdict::NotChecked.verified());
        assert!(!MoeDownA8Verdict::Failed {
            verified: 0,
            total: 1
        }
        .verified());
        assert!(!MoeDownA8Verdict::NotRequested { killed: true }.verified());
    }

    #[test]
    fn verified_line_names_count_device_and_kill_switch() {
        assert_eq!(
            moe_down_a8_info(&MoeDownA8Verdict::Verified { n: 40 }, 1, false).unwrap(),
            "qwen4 moe down A8: 40/40 MoE layers verified symmetric on device 1; \
             int8 routed-down route on (HIPFIRE_QWEN4_MOE_DOWN_A8=0 kills it)"
        );
        let pending = moe_down_a8_info(&MoeDownA8Verdict::Verified { n: 40 }, 0, true).unwrap();
        assert!(pending.contains("once the MTP head layer is verified"));
    }

    #[test]
    fn off_reasons_are_named_and_quiet_states_are_silent() {
        let failed = moe_down_a8_info(
            &MoeDownA8Verdict::Failed {
                verified: 3,
                total: 40,
            },
            2,
            false,
        )
        .unwrap();
        assert!(failed.contains("3/40 MoE layers verified symmetric on device 2"));
        assert!(failed.ends_with("route off (incumbent down)"));
        let killed =
            moe_down_a8_info(&MoeDownA8Verdict::NotRequested { killed: true }, 1, false).unwrap();
        assert!(killed.contains("HIPFIRE_QWEN4_MOE_DOWN_A8=0"));
        assert!(moe_down_a8_info(&MoeDownA8Verdict::NotChecked, 0, false)
            .unwrap()
            .contains("not checked"));
        assert_eq!(
            moe_down_a8_info(&MoeDownA8Verdict::NotRequested { killed: false }, 1, false),
            None
        );
        assert_eq!(
            moe_down_a8_info(&MoeDownA8Verdict::NoLayers, 1, false),
            None
        );
    }

    #[test]
    fn head_lines() {
        assert!(moe_down_a8_head_info(true, true, 0).contains("route on"));
        assert!(moe_down_a8_head_info(true, false, 0).contains("stays off"));
        assert!(moe_down_a8_head_info(false, true, 0).contains("not symmetric"));
    }
}
