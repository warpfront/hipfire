// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Checked adapter from today's aggregate MQ4-XTS expert tensors (one
//! `gate_up` and one `down` tensor per trunk layer) to the per-expert
//! [`ExpertSlab`] list the tier planner consumes.
//!
//! Real artifact geometry, per layer:
//!
//! * gate_up `[512, 1280, 2560]`, qt 44, 891,289,600 B (1,740,800 B per expert);
//! * down `[512, 2560, 640]`, qt 53, 445,644,800 B (870,400 B per expert).
//!
//! The two projections are independent spans and need not be adjacent. After
//! the K4 handoff this becomes `ExpertSlabTable::from_aggregate_mq4` inside
//! `expert_slab_table.rs`. Until then, callers feed the result to
//! [`crate::expert_tier_plan::plan_expert_tiers_from_slabs`].

use crate::expert_slab_table::{ExpertSlab, ExpertSlabSpan};
use crate::expert_slabs::{ExpertFormat, EXPERTS_PER_LAYER, TRUNK_LAYERS};
use crate::expert_tier_plan::ExpertTierError;

/// One aggregate expert tensor of a layer, as indexed in the artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateProjection {
    /// Payload offset in the artifact's data region.
    pub offset: u64,
    /// Payload length in bytes.
    pub len: u64,
    pub quant_type: u8,
    pub shape: Vec<usize>,
}

/// The two aggregate expert tensors of one trunk layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AggregateLayer {
    pub gate_up: AggregateProjection,
    pub down: AggregateProjection,
}

/// Split every layer's aggregate tensors into per-expert spans. Returns
/// `48 * 512` slabs in `layer * 512 + expert` order.
pub fn slabs_from_aggregate_mq4(
    layers: &[AggregateLayer],
) -> Result<Vec<ExpertSlab>, ExpertTierError> {
    let experts = EXPERTS_PER_LAYER as usize;
    if layers.len() != TRUNK_LAYERS as usize {
        return Err(ExpertTierError::SlabCount {
            got: layers.len().saturating_mul(experts),
        });
    }
    let (gate_up_qt, down_qt) = ExpertFormat::Mq4Xts.pair();
    let mut slabs = Vec::with_capacity(layers.len() * experts);
    for (layer, agg) in layers.iter().enumerate() {
        let gate_up_stride = stride(
            layer,
            &agg.gate_up,
            gate_up_qt,
            &GATE_UP_SHAPE,
            GATE_UP_BYTES,
        )?;
        let down_stride = stride(layer, &agg.down, down_qt, &DOWN_SHAPE, DOWN_BYTES)?;
        for expert in 0..experts {
            slabs.push(ExpertSlab {
                format: ExpertFormat::Mq4Xts,
                gate_up: span(layer, &agg.gate_up, gate_up_stride, expert)?,
                down: span(layer, &agg.down, down_stride, expert)?,
            });
        }
    }
    Ok(slabs)
}

/// Real aggregate geometry (module docs): the only shapes and byte lengths
/// the adapter admits.
pub(crate) const GATE_UP_SHAPE: [usize; 3] = [512, 1280, 2560];
pub(crate) const DOWN_SHAPE: [usize; 3] = [512, 2560, 640];
const GATE_UP_BYTES: u64 = 891_289_600;
const DOWN_BYTES: u64 = 445_644_800;

/// Validate one aggregate projection against its exact quant type, shape and
/// byte length, and return its per-expert byte stride.
fn stride(
    layer: usize,
    proj: &AggregateProjection,
    expected_qt: u8,
    expected_shape: &[usize; 3],
    expected_bytes: u64,
) -> Result<u64, ExpertTierError> {
    let bad = |reason| ExpertTierError::Aggregate { layer, reason };
    if proj.quant_type != expected_qt {
        return Err(bad("quant type is not the MQ4-XTS (44,53) pair"));
    }
    let experts = u64::from(EXPERTS_PER_LAYER);
    if proj.shape.first().copied() != Some(experts as usize) {
        return Err(bad("leading shape dimension is not 512"));
    }
    if proj.shape != expected_shape {
        return Err(bad(
            "shape is not [512,1280,2560] (gate_up) / [512,2560,640] (down)",
        ));
    }
    if proj.len == 0 {
        return Err(bad("empty projection"));
    }
    if proj.len % experts != 0 {
        return Err(bad("length is not divisible by 512"));
    }
    if proj.len != expected_bytes {
        return Err(bad(
            "length is not 891289600 (gate_up) / 445644800 (down) bytes",
        ));
    }
    let end = proj
        .offset
        .checked_add(proj.len)
        .ok_or_else(|| bad("span overflows u64"))?;
    usize::try_from(end).map_err(|_| bad("span exceeds usize"))?;
    Ok(proj.len / experts)
}

fn span(
    layer: usize,
    proj: &AggregateProjection,
    stride: u64,
    expert: usize,
) -> Result<ExpertSlabSpan, ExpertTierError> {
    let bad = |reason| ExpertTierError::Aggregate { layer, reason };
    // `stride` validation guarantees `offset + len` fits u64/usize, and
    // `expert * stride < len`, so these cannot overflow; stay checked anyway.
    let offset = (expert as u64)
        .checked_mul(stride)
        .and_then(|rel| proj.offset.checked_add(rel))
        .ok_or_else(|| bad("expert span overflows u64"))?;
    Ok(ExpertSlabSpan {
        offset: usize::try_from(offset).map_err(|_| bad("expert span exceeds usize"))?,
        len: usize::try_from(stride).map_err(|_| bad("expert span exceeds usize"))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_tier_plan::{plan_expert_tiers_from_slabs, Placement};

    const GU_LEN: u64 = 891_289_600;
    const DN_LEN: u64 = 445_644_800;
    const GU_STRIDE: u64 = 1_740_800;
    const DN_STRIDE: u64 = 870_400;

    fn layer(i: usize) -> AggregateLayer {
        // Non-adjacent projections: down sits a gap after gate_up.
        let base = 4096 + i as u64 * (GU_LEN + DN_LEN + 8192);
        AggregateLayer {
            gate_up: AggregateProjection {
                offset: base,
                len: GU_LEN,
                quant_type: 44,
                shape: vec![512, 1280, 2560],
            },
            down: AggregateProjection {
                offset: base + GU_LEN + 4096,
                len: DN_LEN,
                quant_type: 53,
                shape: vec![512, 2560, 640],
            },
        }
    }

    fn table() -> Vec<AggregateLayer> {
        (0..48).map(layer).collect()
    }

    #[test]
    fn real_geometry_splits_into_per_expert_spans() {
        assert_eq!(GU_LEN / 512, GU_STRIDE);
        assert_eq!(DN_LEN / 512, DN_STRIDE);
        let slabs = slabs_from_aggregate_mq4(&table()).unwrap();
        assert_eq!(slabs.len(), 24_576);
        for l in [0usize, 1, 47] {
            let agg = layer(l);
            for e in [0usize, 1, 511] {
                let s = slabs[l * 512 + e];
                assert_eq!(s.format, ExpertFormat::Mq4Xts);
                assert_eq!(s.gate_up.len as u64, GU_STRIDE);
                assert_eq!(s.down.len as u64, DN_STRIDE);
                assert_eq!(
                    s.gate_up.offset as u64,
                    agg.gate_up.offset + e as u64 * GU_STRIDE
                );
                assert_eq!(s.down.offset as u64, agg.down.offset + e as u64 * DN_STRIDE);
            }
        }
        // Expert 511 ends exactly at the end of its aggregate tensor.
        let last = slabs[511];
        assert_eq!(
            (last.gate_up.offset + last.gate_up.len) as u64,
            layer(0).gate_up.offset + GU_LEN
        );
        assert_eq!(
            (last.down.offset + last.down.len) as u64,
            layer(0).down.offset + DN_LEN
        );
    }

    #[test]
    fn adapter_output_feeds_the_planner() {
        let slabs = slabs_from_aggregate_mq4(&table()).unwrap();
        let plan = plan_expert_tiers_from_slabs(&slabs, &[0, 1], u64::MAX / 2).unwrap();
        assert_eq!(plan.device_experts(), 2);
        let e = plan.entry(0, 1).unwrap();
        assert_eq!(e.tier, Placement::Device);
        assert_eq!(e.gate_up_len, GU_STRIDE);
        assert_eq!(e.down_len, DN_STRIDE);
        // Host owner is far beyond 4 GiB and offsets stay u64.
        assert!(plan.owner_bytes(Placement::HostMapped) > u64::from(u32::MAX));
    }

    #[test]
    fn wrong_quant_type_is_rejected() {
        let mut t = table();
        t[5].gate_up.quant_type = 45;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 5, .. })
        ));
        let mut t = table();
        t[9].down.quant_type = 44;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 9, .. })
        ));
    }

    #[test]
    fn wrong_layer_count_is_rejected() {
        let mut t = table();
        t.pop();
        assert_eq!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::SlabCount { got: 47 * 512 })
        );
        t.push(layer(47));
        t.push(layer(48));
        assert_eq!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::SlabCount { got: 49 * 512 })
        );
        assert_eq!(
            slabs_from_aggregate_mq4(&[]),
            Err(ExpertTierError::SlabCount { got: 0 })
        );
    }

    #[test]
    fn bad_shape_or_indivisible_len_is_rejected() {
        let mut t = table();
        t[2].gate_up.shape[0] = 256;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 2, .. })
        ));
        let mut t = table();
        t[3].down.shape.clear();
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 3, .. })
        ));
        let mut t = table();
        t[4].down.len += 1;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 4, .. })
        ));
        let mut t = table();
        t[6].gate_up.len = 0;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 6, .. })
        ));
        let mut t = table();
        t[7].gate_up.offset = u64::MAX - 10;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 7, .. })
        ));
    }

    #[test]
    fn full_shape_and_exact_byte_length_are_required() {
        // Trailing dimensions, rank, and a 512-divisible but wrong length
        // are all refused, for both projections.
        let mut t = table();
        t[1].gate_up.shape[1] = 1281;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 1, .. })
        ));
        let mut t = table();
        t[2].down.shape = vec![512, 640, 2560];
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 2, .. })
        ));
        let mut t = table();
        t[3].gate_up.shape.push(1);
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 3, .. })
        ));
        let mut t = table();
        t[4].gate_up.len += 512;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 4, .. })
        ));
        let mut t = table();
        t[5].down.len -= 512;
        assert!(matches!(
            slabs_from_aggregate_mq4(&t),
            Err(ExpertTierError::Aggregate { layer: 5, .. })
        ));
    }
}
