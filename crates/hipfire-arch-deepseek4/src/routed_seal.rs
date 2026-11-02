//! Sealed-MoE binding of one layer's routed MQ2-Lloyd experts.
//!
//! The loader packs every layer's experts into two blobs and uploads one
//! device address per expert (`expert_gate_up_ptrs`, `expert_w2_ptrs`); EP
//! shards point non-owned experts at a zero gate/up dummy and at expert 0's
//! down slot. The binding describes exactly those addresses as non-owning
//! views, so the sealed executor sees the tables the kernels read.

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::RoutedExpertWeights;
use hipfire_dispatch::pipeline::{
    ExpertBindingCache, ExpertMetadata, ExpertResource, ExpertResources, ExpertTable,
};
use hipfire_dispatch::types::dtype_rotation_plan;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::deepseek4::DeepseekV4LayerWeights;

const DTYPE: DType = DType::MQ2G256Lloyd;

pub(crate) struct RoutedSeal {
    /// Device address of the layer's gate/up pointer table.
    key: usize,
    pub(crate) table: ExpertTable,
    pub(crate) cache: ExpertBindingCache,
    pub(crate) experts: RoutedViews,
}

/// Per-expert `(gate_up [2*mi, hidden], down [hidden, mi])` views.
pub(crate) struct RoutedViews {
    views: Vec<(GpuTensor, GpuTensor)>,
    hidden: usize,
    mi: usize,
    gate_up_row: usize,
    down_row: usize,
}

impl RoutedExpertWeights for RoutedViews {
    fn len(&self) -> usize {
        self.views.len()
    }

    fn get(&self, expert: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        let weight = |buf, m, k, row_stride| WeightRef {
            buf,
            dtype: DTYPE,
            m,
            k,
            row_stride,
            rotation: None,
            awq_scale: None,
            lloyd_lut_e4m3: None,
            lloyd_lut_f16: None,
            lloyd_lut_c16: None,
        };
        self.views.get(expert).map(|(gate_up, down)| {
            (
                weight(gate_up, 2 * self.mi, self.hidden, self.gate_up_row),
                weight(down, self.hidden, self.mi, self.down_row),
            )
        })
    }
}

impl RoutedSeal {
    /// Build the seal of `layer` unless present. Host-only: no launch, so it
    /// is safe inside graph capture.
    pub(crate) fn ensure(
        seals: &mut Vec<RoutedSeal>,
        gpu: &Gpu,
        layer: &DeepseekV4LayerWeights,
        layer_idx: usize,
        hidden: usize,
        mi: usize,
    ) -> Result<(), String> {
        let key = table_key(layer)
            .ok_or_else(|| format!("deepseek4 l{layer_idx}: gate/up pointer table missing"))?;
        if !seals.iter().any(|s| s.key == key) {
            seals.push(Self::build(gpu, layer, layer_idx, key, hidden, mi)?);
        }
        Ok(())
    }

    /// The seal [`Self::ensure`] built for `layer`.
    pub(crate) fn find<'s>(
        seals: &'s [RoutedSeal],
        layer: &DeepseekV4LayerWeights,
    ) -> Option<&'s RoutedSeal> {
        let key = table_key(layer)?;
        seals.iter().find(|s| s.key == key)
    }

    fn build(
        gpu: &Gpu,
        layer: &DeepseekV4LayerWeights,
        layer_idx: usize,
        key: usize,
        hidden: usize,
        mi: usize,
    ) -> Result<Self, String> {
        let err = |what: String| format!("deepseek4 l{layer_idx}: routed seal: {what}");
        let gate_up_row = DTYPE
            .row_bytes(hidden)
            .ok_or_else(|| err(format!("no {DTYPE:?} row layout for k={hidden}")))?;
        let down_row = DTYPE
            .row_bytes(mi)
            .ok_or_else(|| err(format!("no {DTYPE:?} row layout for k={mi}")))?;
        let (gate_up_bytes, down_bytes) = (2 * mi * gate_up_row, hidden * down_row);
        if layer.expert_gate_up_stride != gate_up_bytes || layer.expert_w2_stride != down_bytes {
            return Err(err(format!(
                "expert strides ({}, {}) are not {DTYPE:?} [{}, {hidden}] / [{hidden}, {mi}] ({gate_up_bytes}, {down_bytes})",
                layer.expert_gate_up_stride,
                layer.expert_w2_stride,
                2 * mi
            )));
        }
        let (gate_up_entries, down_entries) = (
            &layer.expert_gate_up_ptr_entries,
            &layer.expert_w2_ptr_entries,
        );
        if gate_up_entries.is_empty() || gate_up_entries.len() != down_entries.len() {
            return Err(err(format!(
                "pointer entries cover {} gate/up and {} down experts",
                gate_up_entries.len(),
                down_entries.len()
            )));
        }
        // Quantized buffers are shaped by byte length (`shape × dtype.size()`
        // must equal the capacity); the logical [m, k] lives in the WeightRef.
        let view = |ptr: u64, bytes: usize| GpuTensor {
            // SAFETY: `ptr` is an address the loader uploaded into this layer's
            // pointer table; the layer owns that allocation for at least
            // `bytes` bytes. The view is non-owning and never freed.
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, bytes) },
            shape: vec![bytes / DTYPE.size()],
            dtype: DTYPE,
        };
        let mut metadata = Vec::with_capacity(gate_up_entries.len());
        let mut views = Vec::with_capacity(gate_up_entries.len());
        for (e, (&gate_up, &down)) in gate_up_entries.iter().zip(down_entries).enumerate() {
            let resource = |name: &str, shape: [usize; 2], bytes: usize, row: usize| {
                ExpertResource::new(
                    format!("layers.{layer_idx}.ffn.experts.{e}.{name}"),
                    "deepseek4-mq2r-blob-v1",
                    shape.to_vec(),
                    DTYPE,
                    bytes,
                    row,
                    16,
                    dtype_rotation_plan(DTYPE),
                )
                .map_err(|e2| err(format!("expert {e} {name}: {e2:?}")))
            };
            let resources = ExpertResources::fused(
                resource("gate_up", [2 * mi, hidden], gate_up_bytes, gate_up_row)?,
                resource("down", [hidden, mi], down_bytes, down_row)?,
            )
            .map_err(|e2| err(format!("expert {e}: {e2:?}")))?;
            metadata.push(
                ExpertMetadata::new(e, 0, e, resources)
                    .map_err(|e2| err(format!("expert {e}: {e2:?}")))?,
            );
            views.push((view(gate_up, gate_up_bytes), view(down, down_bytes)));
        }
        let experts = RoutedViews {
            views,
            hidden,
            mi,
            gate_up_row,
            down_row,
        };
        let table = ExpertTable::new(metadata).map_err(|e| err(format!("table: {e:?}")))?;
        let mut cache = table
            .prepare_binding(0, 1, gpu.device_id)
            .map_err(|e| err(format!("binding: {e:?}")))?;
        cache
            .bind_live(
                &table,
                &experts,
                layer.expert_gate_up_ptrs.as_ref().unwrap(),
                layer
                    .expert_w2_ptrs
                    .as_ref()
                    .ok_or_else(|| err("down pointer table missing".into()))?,
                None,
                None,
            )
            .map_err(|e| err(format!("bind live experts: {e:?}")))?;
        Ok(Self {
            key,
            table,
            cache,
            experts,
        })
    }
}

fn table_key(layer: &DeepseekV4LayerWeights) -> Option<usize> {
    Some(layer.expert_gate_up_ptrs.as_ref()?.buf.as_ptr() as usize)
}
