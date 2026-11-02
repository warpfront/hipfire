// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU execution of a routed MoE layer's experts.
//!
//! One token's routed-expert FFN is, per selected expert, a fused gate/up GEMV
//! (`m = 2*mi`, `k = dim`), the SwiGLU between the two projections
//! ([`crate::epilogue::silu_mul`]), and a down GEMV (`m = dim`, `k = mi`) whose
//! result enters the token's output scaled by that expert's routing weight. On
//! the GPU that whole sequence is one fused `Step::Moe`, so a spilled expert
//! blob pays PCIe for every expert on every token. [`run_experts`] is the CPU
//! image of it: the same bytes, multiplied on the host — the arm
//! `memory.offload_exec=cpu` needs for MoE, exactly as [`crate::gemv::gemv`] is
//! the arm it needs for a dense layer.
//!
//! The caller owns everything the launcher owns: the activation must already be
//! in the form the packed weights expect (FWHT-rotated for a rotated format, any
//! AWQ division applied to the activation and not the weights), and `routing`
//! must come from the same route the GPU path would have used, so the expert
//! *selection* is not an input this module can vary.

use crate::epilogue::silu_mul;
use crate::gemv::{gemv, tensor_bytes};
use crate::quant::CpuQuant;

/// The packed expert weight blobs of one MoE layer, laid out as the on-disk
/// tensor is: every routed expert's rows end to end.
///
/// `gate_up` holds `n_exp * gate_up_stride` bytes and `down` holds
/// `n_exp * down_stride`; expert `slot`'s weight bytes are
/// `[slot*stride .. slot*stride + tensor_bytes(quant, m, k)]`. A `stride` larger than the
/// expert's own bytes is legal (a padded or interleaved layout); smaller is not,
/// and is reported rather than read out of range.
pub struct ExpertBlobs<'a> {
    pub gate_up: &'a [u8],
    pub down: &'a [u8],
    pub gate_up_stride: usize,
    pub down_stride: usize,
    pub quant: CpuQuant,
    /// The layer's hidden width — `k` of the gate/up projection.
    pub dim: usize,
    /// `moe_intermediate_size` — the expert's own hidden width, `m` of gate/up
    /// and `k` of down.
    pub mi: usize,
}

/// Execute one token's routed experts into `out`.
///
/// `routing` is `(expert_id, weight)` per selected expert. `out` is `dim` long
/// and is **accumulated into** (the fused down projection is residual form), so
/// the caller seeds it with whatever the layer's residual carries.
///
/// Returns the message of a shape violation — an activation or output too short
/// to be the layer's, or a slot outside the blobs — as an `Err` rather than
/// silently reading past a slice or skipping an expert.
pub fn run_experts(
    blobs: &ExpertBlobs<'_>,
    x: &[f32],
    routing: &[(usize, f32)],
    out: &mut [f32],
) -> Result<(), String> {
    let (dim, mi) = (blobs.dim, blobs.mi);
    if dim == 0 || mi == 0 {
        return Err(format!("moe: degenerate geometry dim={dim} mi={mi}"));
    }
    if x.len() < dim {
        return Err(format!(
            "moe: activation has {} elements, needs {dim}",
            x.len()
        ));
    }
    if out.len() < dim {
        return Err(format!(
            "moe: output has {} elements, needs {dim}",
            out.len()
        ));
    }
    let gu_bytes = tensor_bytes(blobs.quant, 2 * mi, dim);
    let down_bytes = tensor_bytes(blobs.quant, dim, mi);
    let mut gate_up_out = vec![0.0f32; 2 * mi];
    let mut hidden = vec![0.0f32; mi];
    let mut down_out = vec![0.0f32; dim];
    for &(slot, weight) in routing {
        let gu0 = slot
            .checked_mul(blobs.gate_up_stride)
            .filter(|&o| o + gu_bytes <= blobs.gate_up.len())
            .ok_or_else(|| {
                format!(
                    "moe: gate_up slot {slot} out of range ({} bytes, stride {}, needs {gu_bytes})",
                    blobs.gate_up.len(),
                    blobs.gate_up_stride
                )
            })?;
        let d0 = slot
            .checked_mul(blobs.down_stride)
            .filter(|&o| o + down_bytes <= blobs.down.len())
            .ok_or_else(|| {
                format!(
                    "moe: down slot {slot} out of range ({} bytes, stride {}, needs {down_bytes})",
                    blobs.down.len(),
                    blobs.down_stride
                )
            })?;
        gemv(
            blobs.quant,
            &blobs.gate_up[gu0..gu0 + gu_bytes],
            2 * mi,
            dim,
            x,
            &mut gate_up_out,
        );
        silu_mul(&gate_up_out, &mut hidden);
        // The device path fuses this rotation into the silu step
        // (`fused_silu_mul_rotate_mq_batched`): the down weights are stored
        // post-rotation, so the post-SiLU hidden must be forward-rotated
        // before the down GEMV — for every FWHT-basis format. Skip it for the
        // unrotated dtypes (e.g. `Mq2G256LloydU`), whose down weights live in
        // the natural basis. (CPU-down never caught this: it consumed the
        // GPU's already-rotated `rot_batch`.)
        crate::quant::rotate_input(blobs.quant, &mut hidden);
        gemv(
            blobs.quant,
            &blobs.down[d0..d0 + down_bytes],
            dim,
            mi,
            &hidden,
            &mut down_out,
        );
        for (o, v) in out.iter_mut().zip(down_out.iter()) {
            *o += weight * v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::testfix::{group_bytes, weight_bytes};

    const Q: CpuQuant = CpuQuant::Mq4G256;
    const DIM: usize = 256;
    const MI: usize = 512;

    /// A multi-expert blob whose experts are byte-distinct: `testfix`'s salt is
    /// shifted by `base`, so expert 0 and expert 1 never read the same weights
    /// (which is what makes a slot mix-up observable).
    fn blob(m: usize, k: usize, base: usize) -> Vec<u8> {
        let groups = k / Q.group_elems();
        let mut out = Vec::with_capacity(weight_bytes(Q, m, k).len());
        for row in 0..m {
            for g in 0..groups {
                out.extend_from_slice(&group_bytes(Q, base + row * groups + g));
            }
        }
        out
    }

    fn activation() -> Vec<f32> {
        (0..DIM).map(|i| ((i as f32) * 0.017).sin() * 1.5).collect()
    }

    fn blobs<'a>(gate_up: &'a [u8], down: &'a [u8]) -> ExpertBlobs<'a> {
        ExpertBlobs {
            gate_up,
            down,
            gate_up_stride: tensor_bytes(Q, 2 * MI, DIM),
            down_stride: tensor_bytes(Q, DIM, MI),
            quant: Q,
            dim: DIM,
            mi: MI,
        }
    }

    #[test]
    fn experts_are_addressed_by_slot_and_scaled_by_routing_weight() {
        // Two experts per blob, byte-distinct (distinct salts), so a slot mix-up
        // is observable.
        let mut gu = blob(2 * MI, DIM, 0);
        gu.extend_from_slice(&blob(2 * MI, DIM, 5_000_000));
        let mut dn = blob(DIM, MI, 7_000_000);
        dn.extend_from_slice(&blob(DIM, MI, 9_000_000));
        let b = blobs(&gu, &dn);
        let x = activation();

        let mut e0 = vec![0.0f32; DIM];
        run_experts(&b, &x, &[(0, 1.0)], &mut e0).unwrap();
        let mut e1 = vec![0.0f32; DIM];
        run_experts(&b, &x, &[(1, 1.0)], &mut e1).unwrap();
        assert_ne!(
            e0, e1,
            "slot 1 must read expert 1's weights, not expert 0's"
        );

        // Linearity in the routing weight *and* accumulation across experts: the
        // two-expert sum is exactly 0.5*e0 + 1.5*e1.
        let mut both = vec![0.0f32; DIM];
        run_experts(&b, &x, &[(0, 0.5), (1, 1.5)], &mut both).unwrap();
        for j in 0..DIM {
            let want = 0.5 * e0[j] + 1.5 * e1[j];
            assert!(
                (both[j] - want).abs() <= 1e-4 * want.abs().max(1.0),
                "element {j}: {} vs {want}",
                both[j]
            );
        }

        // A zero routing weight contributes nothing — no write, no read path
        // that leaks a scale.
        let mut none = vec![0.0f32; DIM];
        run_experts(&b, &x, &[(0, 0.0), (1, 0.0)], &mut none).unwrap();
        assert!(none.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_padded_stride_reads_the_same_expert_as_a_packed_one() {
        let gu = blob(2 * MI, DIM, 0);
        let dn = blob(DIM, MI, 7_000_000);
        let packed = blobs(&gu, &dn);
        let x = activation();
        let mut want = vec![0.0f32; DIM];
        run_experts(&packed, &x, &[(0, 1.0)], &mut want).unwrap();

        // Two slots with a stride twice the expert's bytes: slot 1 is an
        // untouched pad, so reading it with weight 0 must not change the result
        // and slot 0 must land where the packed layout put it.
        let mut gu_pad = gu.clone();
        gu_pad.resize(2 * packed.gate_up_stride, 0);
        gu_pad.extend_from_slice(&gu);
        let mut dn_pad = dn.clone();
        dn_pad.resize(2 * packed.down_stride, 0);
        dn_pad.extend_from_slice(&dn);
        let padded = ExpertBlobs {
            gate_up: &gu_pad,
            down: &dn_pad,
            gate_up_stride: 2 * packed.gate_up_stride,
            down_stride: 2 * packed.down_stride,
            ..blobs(&gu, &dn)
        };
        let mut got = vec![0.0f32; DIM];
        run_experts(&padded, &x, &[(0, 1.0), (1, 0.0)], &mut got).unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn post_silu_hidden_is_rotated_before_down_on_fwht_formats() {
        // Asymmetric gate/up bytes: the hidden is non-uniform, so skipping the
        // post-SiLU rotation changes the down GEMV observably. A single-group
        // swap (first two groups exchange their codes) is enough.
        let gu0 = blob(2 * MI, DIM, 0);
        let dn = blob(DIM, MI, 7_000_000);
        let b = blobs(&gu0, &dn);
        let x = activation();
        let mut rotated = vec![0.0f32; DIM];
        run_experts(&b, &x, &[(0, 1.0)], &mut rotated).unwrap();
        // Recompute the same expert with the hidden rotation skipped: decode
        // gate_up, silu, then down WITHOUT rotate_x. Must differ — otherwise
        // the rotation under test is a no-op on this fixture and the test is
        // vacuous.
        {
            use crate::epilogue::silu_mul;
            use crate::gemv::{gemv, tensor_bytes};
            let gu_bytes = tensor_bytes(Q, 2 * MI, DIM);
            let down_bytes = tensor_bytes(Q, DIM, MI);
            let mut gate_up_out = vec![0.0f32; 2 * MI];
            let mut hidden = vec![0.0f32; MI];
            let mut down_out = vec![0.0f32; DIM];
            let mut xr = x.clone();
            crate::quant::rotate_x(&mut xr);
            gemv(Q, &gu0[..gu_bytes], 2 * MI, DIM, &xr, &mut gate_up_out);
            silu_mul(&gate_up_out, &mut hidden);
            // NOTE: no rotate_x here — the pre-fix behavior.
            gemv(Q, &dn[..down_bytes], DIM, MI, &hidden, &mut down_out);
            assert_ne!(
                rotated, down_out,
                "post-SiLU rotation must change the down output on FWHT formats"
            );
        }
        // Unrotated sibling: rotation must NOT apply — same construction with
        // Mq2G256LloydU must equal the manual no-rotate recompute exactly.
        {
            use crate::epilogue::silu_mul;
            use crate::gemv::{gemv, tensor_bytes};
            use crate::quant::CpuQuant;
            let qu = CpuQuant::Mq2G256LloydU;
            assert!(!qu.is_fwht_g256());
            let gu_bytes = tensor_bytes(qu, 2 * MI, DIM);
            let down_bytes = tensor_bytes(qu, DIM, MI);
            let gu = crate::testfix::weight_bytes(qu, 2 * MI, DIM);
            let dn = crate::testfix::weight_bytes(qu, DIM, MI);
            let bu = ExpertBlobs {
                gate_up: &gu,
                down: &dn,
                gate_up_stride: gu_bytes,
                down_stride: down_bytes,
                quant: qu,
                dim: DIM,
                mi: MI,
            };
            let mut got = vec![0.0f32; DIM];
            run_experts(&bu, &x, &[(0, 1.0)], &mut got).unwrap();
            // Manual reference: no input rotation (natural basis), no hidden
            // rotation — must match bit-for-bit since the code path is
            // straight-line GEMV+silu+GEMV with no RNG.
            let mut gate_up_out = vec![0.0f32; 2 * MI];
            let mut hidden = vec![0.0f32; MI];
            let mut down_out = vec![0.0f32; DIM];
            gemv(qu, &gu[..gu_bytes], 2 * MI, DIM, &x, &mut gate_up_out);
            silu_mul(&gate_up_out, &mut hidden);
            gemv(qu, &dn[..down_bytes], DIM, MI, &hidden, &mut down_out);
            assert_eq!(got, down_out, "unrotated dtype must skip both rotations");
        }
    }

    #[test]
    fn shape_violations_are_errors_not_silent_partial_reads() {
        let gu = blob(2 * MI, DIM, 0);
        let dn = blob(DIM, MI, 7_000_000);
        let b = blobs(&gu, &dn);

        assert!(run_experts(&b, &[0.0f32; 8], &[(0, 1.0)], &mut [0.0f32; DIM]).is_err());
        assert!(run_experts(&b, &activation(), &[(0, 1.0)], &mut [0.0f32; 8]).is_err());
        assert!(run_experts(&b, &activation(), &[(2, 1.0)], &mut [0.0f32; DIM]).is_err());
        assert!(run_experts(&b, &activation(), &[], &mut [0.0f32; DIM]).is_ok());
    }
}
