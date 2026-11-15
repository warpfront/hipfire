//! The stock-FP8 (qt=55) MTP head's MoE scratch.
//!
//! A stock-FP8 head has no indexed decode route: its routed experts are the
//! `FP8E4M3B128` tables of the `<model>.hfq.mtp` sidecar, and the gfx942 route
//! ([`hipfire_dispatch::families::moe::MoeRouteCapability::Fp8B128Grouped`])
//! always seals the grouped prefill call, one routing row included
//! (`program::seal_moe_for_layer`). The one-row head step therefore needs the
//! grouped route's scratch (stable-scatter grouping buffers and the F32 grouped
//! gate_up / down outputs) that the QT44/QT53 one-row [`MtpGpuScratch`] does not
//! carry. It is allocated once at head attach, only for a stock-FP8 head, sized
//! like the trunk's `Full` plan at one routing row
//! (`gpu_forward::check_fp8_grouped_scratch`); every other buffer of the call
//! (HC rows, the FFN output) is the one-row scratch's.

use super::*;
use crate::program::{Qwen4LayerScratch, Qwen4ProgramDims, Qwen4ScratchLayout};

/// Allocations of one [`MtpFp8MoeScratch`], in field order.
const FP8_HEAD_MOE_TENSORS: usize = 19;

/// Routing rows of the one-row head step.
const FP8_HEAD_MOE_ROWS: usize = 1;

/// Bytes the qt=55 grouped route needs of `rows` routing rows, the extents
/// `gpu_forward::check_fp8_grouped_scratch` proves of the trunk's `Full` plan:
/// the grouped gate_up output `[G, 2*moe]` F32, the grouped down output
/// `[G, hidden]` F32, `rot_batch` (`slots * moe` F32), the tile descriptor
/// (`G` bytes, in `gate_batch`) and the sorted slot index (`G` i32), with `G`
/// the padded-16 slot bound.
pub(crate) struct Fp8GroupedBytes {
    pub(crate) y_gate_up: usize,
    pub(crate) y_down: usize,
    pub(crate) rot_batch: usize,
    pub(crate) tile_descriptor: usize,
    pub(crate) sorted_slot_index: usize,
}

impl Fp8GroupedBytes {
    pub(crate) fn for_rows(dims: Qwen4ProgramDims, rows: usize) -> Result<Self, MtpGpuError> {
        let Qwen4FfnDims::Moe { intermediate, .. } = dims.ffn else {
            return Err(invalid("a dense head has no grouped MoE route"));
        };
        let layout =
            Qwen4ScratchLayout::for_rows(dims, rows).map_err(|error| invalid(error.to_string()))?;
        let f32_bytes = DType::F32.size();
        Ok(Self {
            y_gate_up: layout.grouped_rows * 2 * intermediate * f32_bytes,
            y_down: layout.grouped_rows * dims.hidden * f32_bytes,
            rot_batch: layout.routed_rows * f32_bytes,
            tile_descriptor: layout.grouped_rows
                / hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M
                * 16,
            sorted_slot_index: layout.grouped_rows * std::mem::size_of::<i32>(),
        })
    }
}

/// Indices of the five grouped-route extents in a scratch's `shapes`.
#[derive(Clone, Copy)]
pub(crate) struct Fp8GroupedFields {
    pub(crate) y_gate_up: usize,
    pub(crate) y_down: usize,
    pub(crate) rot_batch: usize,
    pub(crate) tile_descriptor: usize,
    pub(crate) sorted_slot_index: usize,
}

/// [`MtpFp8MoeScratch::shapes`] indices of the grouped-route extents.
const ONE_ROW_FIELDS: Fp8GroupedFields = Fp8GroupedFields {
    y_gate_up: 17,
    y_down: 18,
    rot_batch: 8,
    tile_descriptor: 6,
    sorted_slot_index: 14,
};

impl Fp8GroupedBytes {
    /// Refuse a scratch (`shapes`, extents at `fields`) that would not hold
    /// the grouped route before the first launch, as the trunk's
    /// `check_fp8_grouped_scratch` does of its `Full` plan.
    pub(crate) fn check_holds(
        &self,
        shapes: &[(usize, DType)],
        fields: Fp8GroupedFields,
    ) -> Result<(), MtpGpuError> {
        for (name, index, need) in [
            ("grouped gate_up output", fields.y_gate_up, self.y_gate_up),
            ("grouped down output", fields.y_down, self.y_down),
            ("rot_batch", fields.rot_batch, self.rot_batch),
            (
                "tile descriptor (gate_batch)",
                fields.tile_descriptor,
                self.tile_descriptor,
            ),
            (
                "sorted slot index",
                fields.sorted_slot_index,
                self.sorted_slot_index,
            ),
        ] {
            let (elements, dtype) = shapes[index];
            let held = elements * dtype.size();
            if held < need {
                return Err(invalid(format!(
                    "stock-FP8 head {name} scratch holds {held} bytes, the grouped route needs {need}"
                )));
            }
        }
        Ok(())
    }
}

/// The grouped-route buffers of a stock-FP8 head's one-row MoE step.
pub(crate) struct MtpFp8MoeScratch {
    router_logits: GpuTensor,
    moe_x_rot: GpuTensor,
    moe_gate: GpuTensor,
    moe_up: GpuTensor,
    moe_hidden: GpuTensor,
    moe_scalar: GpuTensor,
    moe_gate_batch: GpuTensor,
    moe_up_batch: GpuTensor,
    moe_rot_batch: GpuTensor,
    moe_topk_indices: GpuTensor,
    moe_topk_weights: GpuTensor,
    moe_down_expanded: GpuTensor,
    moe_expert_token_counts: GpuTensor,
    moe_expert_offsets: GpuTensor,
    moe_sorted_slot_index: GpuTensor,
    moe_expert_tile_ids: GpuTensor,
    moe_inverse_perm: GpuTensor,
    moe_y_gate_up_grouped: GpuTensor,
    moe_y_down_grouped: GpuTensor,
}

impl MtpFp8MoeScratch {
    /// Element count and dtype of each tensor, in field order; `None` unless
    /// the head is a stock-FP8 profile with a MoE FFN (every other head runs
    /// the indexed decode route out of [`MtpGpuScratch`]).
    pub(crate) fn shapes(
        config: &Qwen4Config,
    ) -> Result<Option<[(usize, DType); FP8_HEAD_MOE_TENSORS]>, MtpGpuError> {
        let config = config.mtp_geometry();
        if !config.profile.is_stock_fp8() {
            return Ok(None);
        }
        let dims = program_dims(config);
        let Qwen4FfnDims::Moe {
            num_experts,
            intermediate,
            ..
        } = dims.ffn
        else {
            return Ok(None);
        };
        let layout = Qwen4ScratchLayout::for_rows(dims, FP8_HEAD_MOE_ROWS)
            .map_err(|error| invalid(error.to_string()))?;
        let hidden = dims.hidden;
        let i32_bytes = std::mem::size_of::<i32>();
        // The tile descriptor lives in `gate_batch`: it holds the larger of
        // the routed rows and the descriptor's bytes.
        let descriptor = Fp8GroupedBytes::for_rows(dims, FP8_HEAD_MOE_ROWS)?.tile_descriptor;
        let gate_batch = layout
            .routed_rows
            .max(descriptor.div_ceil(DType::F32.size()));
        let mut shapes = [
            (num_experts, DType::F32),
            (hidden, DType::F32),
            (layout.shared_rows, DType::F32),
            (layout.shared_rows, DType::F32),
            (layout.shared_rows, DType::F32),
            (layout.shared_rows, DType::F32),
            (gate_batch, DType::F32),
            (layout.routed_rows, DType::F32),
            (layout.routed_rows, DType::F32),
            (layout.slots * i32_bytes, DType::Raw),
            (layout.slots, DType::F32),
            (layout.expanded_rows + hidden.div_ceil(4), DType::F32),
            (num_experts * i32_bytes, DType::Raw),
            ((num_experts + 1) * i32_bytes, DType::Raw),
            (layout.grouped_rows * i32_bytes, DType::Raw),
            (
                layout.grouped_rows / hipfire_dispatch::families::moe::MOE_GROUPED_BLOCK_M
                    * i32_bytes,
                DType::Raw,
            ),
            (layout.slots * i32_bytes, DType::Raw),
            (layout.grouped_rows * 2 * intermediate, DType::F32),
            (layout.grouped_rows * hidden, DType::F32),
        ];
        for shape in &mut shapes {
            shape.0 = shape.0.max(1);
        }
        Ok(Some(shapes))
    }

    /// Device bytes [`Self::new`] allocates: zero unless the head is a
    /// stock-FP8 MoE head. `None` on overflow.
    pub(crate) fn device_bytes(config: &Qwen4Config) -> Option<usize> {
        Some(
            Self::shapes(config)
                .ok()?
                .map_or(0, |shapes| shapes.iter().map(|&(n, d)| n * d.size()).sum()),
        )
    }

    /// [`Self::device_bytes`] per allocation, in field order.
    pub(crate) fn allocation_sizes(config: &Qwen4Config) -> Option<Vec<u64>> {
        Self::shapes(config)
            .ok()?
            .map_or(Some(Vec::new()), |shapes| {
                shapes
                    .iter()
                    .map(|&(n, d)| u64::try_from(n.checked_mul(d.size())?).ok())
                    .collect()
            })
    }

    /// The scratch of a stock-FP8 MoE head (`None` for any other head).
    /// Also reserves the stable-scatter grouping scratch for the head's widest
    /// batched round ([`draft_call_capacity`] rows: 32 for this stock-FP8 head, which exists on
    /// exact single-device gfx942 only) before any forward or
    /// capture; the trunk's own reservation (its prefill chunk) normally
    /// covers it, the call only grows past it.
    pub(crate) fn new(gpu: &mut Gpu, config: &Qwen4Config) -> Result<Option<Self>, MtpGpuError> {
        let Some(shapes) = Self::shapes(config)? else {
            return Ok(None);
        };
        let config = config.mtp_geometry();
        let need = Fp8GroupedBytes::for_rows(program_dims(config), FP8_HEAD_MOE_ROWS)?;
        need.check_holds(&shapes, ONE_ROW_FIELDS)?;
        gpu.try_gfx942()
            .ok_or_else(|| invalid("the qt=55 FP8 head route needs a gfx942 device"))?
            .reserve_fp8b128_moe_group(
                draft_call_capacity(config),
                config.num_experts_per_tok,
                config.num_experts,
            )?;
        let mut allocated = Vec::with_capacity(FP8_HEAD_MOE_TENSORS);
        let result = (|| {
            for (elements, dtype) in shapes {
                allocated.push(gpu.zeros(&[elements], dtype)?);
            }
            Ok::<(), MtpGpuError>(())
        })();
        if let Err(error) = result {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        let mut next = || allocated.remove(0);
        Ok(Some(Self {
            router_logits: next(),
            moe_x_rot: next(),
            moe_gate: next(),
            moe_up: next(),
            moe_hidden: next(),
            moe_scalar: next(),
            moe_gate_batch: next(),
            moe_up_batch: next(),
            moe_rot_batch: next(),
            moe_topk_indices: next(),
            moe_topk_weights: next(),
            moe_down_expanded: next(),
            moe_expert_token_counts: next(),
            moe_expert_offsets: next(),
            moe_sorted_slot_index: next(),
            moe_expert_tile_ids: next(),
            moe_inverse_perm: next(),
            moe_y_gate_up_grouped: next(),
            moe_y_down_grouped: next(),
        }))
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.moe_scalar,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_expert_token_counts,
            self.moe_expert_offsets,
            self.moe_sorted_slot_index,
            self.moe_expert_tile_ids,
            self.moe_inverse_perm,
            self.moe_y_gate_up_grouped,
            self.moe_y_down_grouped,
        ] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }

    /// `[1, router_columns]` view of the router logits: the grouped MoE seal
    /// checks the matrix shape.
    pub(crate) fn router_view(&self, config: &Qwen4Config) -> Result<GpuTensor, MtpGpuError> {
        Ok(crate::gpu_forward::matrix_view(
            &self.router_logits,
            FP8_HEAD_MOE_ROWS,
            crate::gpu_forward::router_columns(program_dims(config.mtp_geometry())),
        )?)
    }

    /// The MoE seal's scratch description of a one-row head step: the grouped
    /// buffers of this scratch, the HC rows and the FFN output of the one-row
    /// `base`. Fields the grouped route never reads alias `base.wide`.
    pub(crate) fn layer_scratch<'a>(
        &'a self,
        base: &'a MtpGpuScratch,
        router_logits: &'a GpuTensor,
    ) -> Qwen4LayerScratch<'a> {
        let unused = &base.wide;
        Qwen4LayerScratch {
            streams: unused,
            hc_normalized: &base.hc_normalized,
            hc_low: &base.hc_low,
            hc_up: &base.hc_up,
            hc_mixed: &base.hc_mixed,
            hc_gates: &base.hc_gates,
            rotation: &base.rotation,
            projection: unused,
            projection2: unused,
            gdn_a: unused,
            gdn_b: unused,
            gdn_gate: unused,
            gdn_beta: unused,
            gdn_recurrent_output: unused,
            gdn_z: unused,
            gdn_output: unused,
            qsa_index: unused,
            qsa_qgate: unused,
            qsa_k: unused,
            qsa_v: unused,
            qsa_output: unused,
            attention_output: unused,
            ffn_output: &base.ffn_output,
            moe_shared_output: unused,
            moe_router_logits: router_logits,
            moe_x_rot: &self.moe_x_rot,
            moe_gate_up: unused,
            moe_scalar: &self.moe_scalar,
            moe_gate: &self.moe_gate,
            moe_up: &self.moe_up,
            moe_hidden: &self.moe_hidden,
            moe_gate_batch: &self.moe_gate_batch,
            moe_up_batch: &self.moe_up_batch,
            moe_rot_batch: &self.moe_rot_batch,
            moe_topk_indices: &self.moe_topk_indices,
            moe_topk_weights: &self.moe_topk_weights,
            moe_down_expanded: &self.moe_down_expanded,
            moe_expert_token_counts: &self.moe_expert_token_counts,
            moe_expert_offsets: &self.moe_expert_offsets,
            moe_sorted_slot_index: &self.moe_sorted_slot_index,
            moe_expert_tile_ids: &self.moe_expert_tile_ids,
            moe_inverse_perm: &self.moe_inverse_perm,
            moe_y_gate_up_grouped: &self.moe_y_gate_up_grouped,
            moe_y_down_grouped: &self.moe_y_down_grouped,
        }
    }
}

/// Lab-only: the last one-row FP8 head MoE routing, read back from the device
/// ([`MtpFp8MoeScratch::lab_read_routing`]).
///
/// What the buffers hold after one head Predict step (the `seal_moe_rows`
/// call at one routing row, `mtp_gpu.rs` `head_block`: `seal_moe_rows(.., 1,
/// layer, None)`, `gpu_forward.rs:7720`), with the producing code:
///
/// * `router`: `[num_experts]` F32 RAW router logits `x @ W_router^T` of the
///   head's MoE input (`hc_mixed`). Not softmax probabilities, not rounded to
///   BF16. The `[1, num_experts]` view of this tensor is the seal's
///   `moe_router_logits` (`fp8_moe.rs` `layer_scratch`/`router_view`), bound as
///   both `router_logits` and `router_scores` (`program.rs:1268-1269`); the
///   router projection writes it (`qt44_qt53_prefill.rs:698-704`,
///   `batch_projection(router, x_norm_batch, router_logits)`; a fused x4 launch
///   with the shared expert's projections yields the same bytes,
///   `qt44_qt53_prefill.rs:707-720`). The route stage (`moe_program.rs:1484`,
///   `sealed_moe::produce_prefill_route`, `sealed_moe.rs:2470`) takes the
///   `grouped` branch (`Fp8B128Grouped` is sealed-grouped,
///   `families/moe.rs:494-500`): the in-place BF16 round trip is skipped for a
///   grouped route (`sealed_moe.rs:2517-2521`, "the grouped top-10 router
///   rounds each logit it reads itself"), and on gfx942 the fused
///   route+scatter launches are not admitted (`rdna-compute/src/moe.rs:2836-2837`
///   gfx1201 only; glue-fold needs 9..=64 rows, `moe.rs:2979`), so
///   `moe_router_softmax_top10_f32` runs (`sealed_moe.rs:2625-2634`). That
///   kernel takes `logits` as `const float*`
///   (`kernels/src/moe_router_softmax_top10_f32.hip:86`): it never writes the
///   router buffer back. The buffer stays the raw F32 projection output.
/// * The routing DECISION is made on the BF16-rounded logits: the kernel
///   reads each logit through `router_bf16_rt` when `round_logits`
///   (`.hip:113-116`), which the head's recipe sets
///   (`MoeRecipe::SoftmaxGatedShared { bf16_round_trip: true, .. }`,
///   `program.rs:1277-1279`, passed at `sealed_moe.rs:2625-2634`
///   `params.recipe.bf16_round_trip()`). So a consumer reproducing the top-k
///   from `router` must round to BF16 first (RNE, `.hip:18-22`) and break ties
///   by (value descending, index ascending) (`.hip:134-175`).
/// * `topk_ids`: `[num_experts_per_tok]` i32 expert ids in rank order (best
///   first). A row with a non-finite rounded logit, or a failed softmax
///   denominator, holds `-1` ids and `0` weights (`.hip:41-49,122-125,196-199`).
/// * `topk_weights`: `[num_experts_per_tok]` F32 in the same rank order: the
///   softmax over all 512 BF16-rounded logits (`.hip:182-203`), RENORMALISED
///   over the selected 10 (`.hip:205-216`; `norm_topk_prob`, the config
///   default `true`, `program.rs:1270`), each stored through the BF16 round
///   trip (`.hip:218-220`).
#[cfg(feature = "lab")]
pub struct LabHeadRouting {
    /// Raw F32 router logits `[num_experts]` (see above).
    pub router: Vec<f32>,
    /// Top-k expert ids `[num_experts_per_tok]`, best first.
    pub topk_ids: Vec<i32>,
    /// Top-k routing weights `[num_experts_per_tok]`, renormalised, BF16-valued.
    pub topk_weights: Vec<f32>,
}

#[cfg(feature = "lab")]
impl MtpFp8MoeScratch {
    /// Download the buffers of the head's last one-row MoE call (what they
    /// hold: [`LabHeadRouting`]). They are written by the head Predict step's
    /// launches on the launch stream, so the copies are enqueued on the
    /// active stream and waited once (the `lab_expert_trace` pattern); nothing
    /// but a head Predict step writes them (an Append step skips the FFN).
    pub(crate) fn lab_read_routing(
        &self,
        gpu: &Gpu,
        config: &Qwen4Config,
    ) -> Result<LabHeadRouting, MtpGpuError> {
        let geometry = config.mtp_geometry();
        let word = std::mem::size_of::<f32>();
        let slots = FP8_HEAD_MOE_ROWS * geometry.num_experts_per_tok;
        let mut router = vec![0u8; geometry.num_experts * word];
        let mut ids = vec![0u8; slots * std::mem::size_of::<i32>()];
        let mut weights = vec![0u8; slots * word];
        let mut parts: [(&GpuTensor, &mut Vec<u8>); 3] = [
            (&self.router_logits, &mut router),
            (&self.moe_topk_indices, &mut ids),
            (&self.moe_topk_weights, &mut weights),
        ];
        for (tensor, out) in parts.iter() {
            if out.len() > tensor.buf.size() {
                return Err(invalid(format!(
                    "stock-FP8 head routing readback wants {} bytes of a {}-byte buffer",
                    out.len(),
                    tensor.buf.size()
                )));
            }
        }
        match gpu.active_stream.as_ref() {
            Some(stream) => {
                for (tensor, out) in parts.iter_mut() {
                    gpu.hip
                        .memcpy_dtoh_async(out.as_mut_slice(), &tensor.buf, Some(stream))?;
                }
                gpu.hip.stream_synchronize(stream)?;
            }
            None => {
                for (tensor, out) in parts.iter_mut() {
                    gpu.hip.memcpy_dtoh(out.as_mut_slice(), &tensor.buf)?;
                }
            }
        }
        let words = |bytes: &[u8]| -> Vec<[u8; 4]> {
            bytes
                .chunks_exact(4)
                .map(|w| [w[0], w[1], w[2], w[3]])
                .collect()
        };
        Ok(LabHeadRouting {
            router: words(&router).into_iter().map(f32::from_le_bytes).collect(),
            topk_ids: words(&ids).into_iter().map(i32::from_le_bytes).collect(),
            topk_weights: words(&weights)
                .into_iter()
                .map(f32::from_le_bytes)
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;

    fn stock_head() -> Qwen4Config {
        compact_test_config()
            .into_stock_mtp()
            .expect("headed stock-FP8 form")
    }

    /// Only a stock-FP8 MoE head carries the grouped scratch; the packed
    /// Flash-Next head allocates and accounts nothing.
    #[test]
    fn only_the_stock_fp8_head_carries_the_grouped_scratch() {
        let packed = compact_test_config();
        assert!(MtpFp8MoeScratch::shapes(&packed).unwrap().is_none());
        assert_eq!(MtpFp8MoeScratch::device_bytes(&packed), Some(0));
        assert_eq!(MtpFp8MoeScratch::allocation_sizes(&packed), Some(vec![]));
        let stock = stock_head();
        assert!(MtpFp8MoeScratch::shapes(&stock).unwrap().is_some());
    }

    /// The per-allocation sizes sum to the device bytes, and the head's
    /// accounting carries them past the packed head's by exactly that sum.
    #[test]
    fn stock_fp8_head_scratch_is_counted_in_the_head_accounting() {
        let stock = stock_head();
        let sizes = MtpFp8MoeScratch::allocation_sizes(&stock).unwrap();
        assert_eq!(sizes.len(), FP8_HEAD_MOE_TENSORS);
        let bytes = MtpFp8MoeScratch::device_bytes(&stock).unwrap();
        assert_eq!(sizes.iter().sum::<u64>() as usize, bytes);
        assert!(bytes > 0);
        let context = Qwen4ContextCommit::legacy(4096);
        let (resident, _) =
            Qwen4MtpGpu::allocation_sizes(&stock, &context, DType::BF16, false).unwrap();
        let (device, _) = Qwen4MtpGpu::device_bytes(&stock, &context, DType::BF16, false).unwrap();
        assert_eq!(resident.iter().sum::<u64>() as usize, device);
        assert!(resident.ends_with(&sizes));
        let packed = compact_test_config();
        let (packed_resident, _) =
            Qwen4MtpGpu::allocation_sizes(&packed, &context, DType::BF16, false).unwrap();
        assert!(!packed_resident.ends_with(&sizes));
    }

    /// The head's one-row scratch holds what the trunk's `Full` plan proves of
    /// its own at one routing row, and the batched rounds' rows scratch holds
    /// the same at its widest round.
    #[test]
    fn stock_fp8_head_scratch_holds_the_grouped_route_extents() {
        let stock = stock_head();
        let dims = program_dims(stock.mtp_geometry());
        let one = MtpFp8MoeScratch::shapes(&stock).unwrap().unwrap();
        Fp8GroupedBytes::for_rows(dims, FP8_HEAD_MOE_ROWS)
            .unwrap()
            .check_holds(&one, ONE_ROW_FIELDS)
            .unwrap();
        // `MtpRowsScratch::shapes` order: gate_batch 24, rot_batch 26, sorted
        // slot index 32, grouped gate_up 35, grouped down 36.
        let rows = MtpRowsScratch::shapes(&stock).unwrap().unwrap();
        Fp8GroupedBytes::for_rows(dims, draft_call_capacity(&stock))
            .unwrap()
            .check_holds(
                &rows,
                Fp8GroupedFields {
                    y_gate_up: 35,
                    y_down: 36,
                    rot_batch: 26,
                    tile_descriptor: 24,
                    sorted_slot_index: 32,
                },
            )
            .unwrap();
        // A scratch one slot short of any extent is refused.
        let mut short = one;
        short[ONE_ROW_FIELDS.y_down].0 -= 1;
        assert!(Fp8GroupedBytes::for_rows(dims, FP8_HEAD_MOE_ROWS)
            .unwrap()
            .check_holds(&short, ONE_ROW_FIELDS)
            .is_err());
    }
}
