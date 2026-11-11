// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Load-time verdict of the symmetric-MQ6 x int8-activation trunk route
//! (`HIPFIRE_QWEN4_TRUNK_A8`, gfx1201).
//!
//! The route reads every qt47 (MQ6G256V2) projection as `w = s * (q - 32)`, so
//! it is only valid when each half-group header satisfies `z == -32 * s`. The
//! verdict is taken once per device, before any capture: every MQ6G256V2
//! tensor the device's forward reads is checked on device
//! ([`Gpu::qwen4_mq6_sym_check`]); all pass -> the route is verified on for the
//! whole `Gpu`, any failure -> it stays off and the affine route runs.

use crate::gpu_forward::Qwen4GpuForwardError;
use crate::weights::{DenseFfnWeights, QsaWeights, Qwen4FfnWeights, Qwen4Weights, TensorRef};
use rdna_compute::{DType, Gpu};

/// Outcome of the per-device verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrunkA8Verdict {
    /// Not requested: `killed` when `HIPFIRE_QWEN4_TRUNK_A8=0` turned it off
    /// on a gfx1201 card, else the card is not gfx1201.
    NotRequested { killed: bool },
    /// The device holds no qt47 tensor.
    NoTensors,
    /// All `n` tensors passed.
    Verified { n: usize },
    /// `first` (name) was the first tensor to fail.
    Failed { first: String },
}

impl TrunkA8Verdict {
    pub(crate) fn verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

/// The one INFO line of a verdict, `None` when none is printed.
pub(crate) fn trunk_a8_info(verdict: &TrunkA8Verdict, device: i32) -> Option<String> {
    match verdict {
        TrunkA8Verdict::NotRequested { killed: true } => Some(format!(
            "qwen4 symmetric MQ6 trunk: killed by HIPFIRE_QWEN4_TRUNK_A8=0 on device {device}; \
             int8 trunk route off (affine qt47)"
        )),
        TrunkA8Verdict::NotRequested { killed: false } | TrunkA8Verdict::NoTensors => None,
        TrunkA8Verdict::Verified { n } => Some(format!(
            "qwen4 symmetric MQ6 trunk: {n}/{n} qt47 tensors verified z == -32*s on device \
             {device}; int8 trunk route on (HIPFIRE_QWEN4_TRUNK_A8=0 kills it)"
        )),
        TrunkA8Verdict::Failed { first } => Some(format!(
            "qwen4 symmetric MQ6 trunk: qt47 tensor {first} is not symmetric (z != -32*s) on \
             device {device}; int8 trunk route off (affine qt47)"
        )),
    }
}

fn qsa_refs<'a>(qsa: &'a QsaWeights, out: &mut Vec<&'a TensorRef>) {
    out.extend([&qsa.indexer_qk, &qsa.q, &qsa.k, &qsa.v, &qsa.output]);
}

fn ffn_refs<'a>(ffn: &'a Qwen4FfnWeights, out: &mut Vec<&'a TensorRef>) {
    if let Qwen4FfnWeights::Dense(DenseFfnWeights { gate, up, down }) = ffn {
        out.extend([gate, up, down]);
    }
}

/// Every matrix-projection reference the forward may route through
/// `project_weights`: GDN in_proj qkv/z/a/b and out_proj, QSA q/k/v/o and the
/// indexer qk, dense FFN gate/up/down, the main lm_head and, when owned, the
/// MTP head's attention / FFN / fc projections. Which of them are MQ6G256V2 on
/// this device is decided from the resident tensor.
fn projection_refs(weights: &Qwen4Weights) -> Vec<&TensorRef> {
    let mut refs = Vec::new();
    for layer in &weights.layer_refs {
        if let Some(gdn) = &layer.gdn {
            refs.extend([
                &gdn.qkv,
                &gdn.z,
                &gdn.in_proj_a,
                &gdn.in_proj_b,
                &gdn.output,
            ]);
        }
        if let Some(qsa) = &layer.attention {
            qsa_refs(qsa, &mut refs);
        }
        ffn_refs(&layer.ffn, &mut refs);
    }
    refs.push(&weights.root.lm_head);
    if let Some(mtp) = &weights.mtp {
        refs.extend([&mtp.fc_embedding, &mtp.fc_hidden]);
        qsa_refs(&mtp.attention, &mut refs);
        ffn_refs(&mtp.ffn, &mut refs);
    }
    refs
}

/// Take the device's verdict and set [`Gpu::set_qwen4_trunk_a8_verified`].
/// Load-time, before any capture / reservation finalization.
pub(crate) fn verify_trunk_a8(
    gpu: &mut Gpu,
    weights: &Qwen4Weights,
) -> Result<TrunkA8Verdict, Qwen4GpuForwardError> {
    let device = gpu.device_id;
    let verdict = if !gpu.qwen4_trunk_a8_requested() {
        TrunkA8Verdict::NotRequested {
            killed: gpu.arch == "gfx1201",
        }
    } else {
        let mut n = 0usize;
        let mut failed = None;
        for reference in projection_refs(weights) {
            // Not owned by this device / band: nothing of it is read here.
            let Ok(tensor) = weights.resident(reference) else {
                continue;
            };
            if tensor.dtype != DType::MQ6G256V2 {
                continue;
            }
            let (m, k) = match (tensor.shape.as_slice(), reference.shape.as_slice()) {
                (&[m, k], _) | (_, &[m, k]) => (m, k),
                _ => {
                    failed = Some(reference.name.clone());
                    break;
                }
            };
            n += 1;
            if !gpu.qwen4_mq6_sym_check(tensor, m, k)? {
                failed = Some(reference.name.clone());
                break;
            }
        }
        match (failed, n) {
            (Some(first), _) => TrunkA8Verdict::Failed { first },
            (None, 0) => TrunkA8Verdict::NoTensors,
            (None, n) => TrunkA8Verdict::Verified { n },
        }
    };
    gpu.set_qwen4_trunk_a8_verified(verdict.verified());
    if let Some(line) = trunk_a8_info(&verdict, device) {
        eprintln!("[INFO] {line}");
    }
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_line_names_count_and_device() {
        let line = trunk_a8_info(&TrunkA8Verdict::Verified { n: 321 }, 2).unwrap();
        assert_eq!(
            line,
            "qwen4 symmetric MQ6 trunk: 321/321 qt47 tensors verified z == -32*s on device 2; \
             int8 trunk route on (HIPFIRE_QWEN4_TRUNK_A8=0 kills it)"
        );
    }

    #[test]
    fn failure_line_names_first_tensor_and_turns_route_off() {
        let line = trunk_a8_info(
            &TrunkA8Verdict::Failed {
                first: "lm_head.weight".into(),
            },
            0,
        )
        .unwrap();
        assert!(line.contains("lm_head.weight"));
        assert!(line.ends_with("int8 trunk route off (affine qt47)"));
    }

    #[test]
    fn killed_says_so_and_other_off_states_are_silent() {
        let killed = trunk_a8_info(&TrunkA8Verdict::NotRequested { killed: true }, 1).unwrap();
        assert!(killed.contains("HIPFIRE_QWEN4_TRUNK_A8=0"));
        assert!(killed.contains("int8 trunk route off"));
        assert_eq!(
            trunk_a8_info(&TrunkA8Verdict::NotRequested { killed: false }, 1),
            None
        );
        assert_eq!(trunk_a8_info(&TrunkA8Verdict::NoTensors, 1), None);
    }

    #[test]
    fn only_all_pass_is_verified() {
        assert!(TrunkA8Verdict::Verified { n: 1 }.verified());
        assert!(!TrunkA8Verdict::NoTensors.verified());
        assert!(!TrunkA8Verdict::Failed { first: "x".into() }.verified());
        assert!(!TrunkA8Verdict::NotRequested { killed: true }.verified());
    }
}
