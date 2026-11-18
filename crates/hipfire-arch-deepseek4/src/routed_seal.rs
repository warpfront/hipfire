//! Sealed-MoE binding of one layer's routed MQ2-Lloyd experts.
//!
//! The loader packs every layer's experts into two blobs and uploads one
//! device address per expert (`expert_gate_up_ptrs`, `expert_w2_ptrs`); EP
//! shards point non-owned experts at a zero gate/up dummy and at expert 0's
//! down slot. The binding describes exactly those addresses as non-owning
//! views, so the sealed executor sees the tables the kernels read.
//!
//! Under a mapped owner (`HIPFIRE_DS4_EXPERT_SOURCE=mapped`, [`crate::expert_source`]) the
//! blobs are pinned host-mapped allocations and every layer also carries a
//! [`MutableExpertResidency`] capability keyed by the owner generation and the catalog extents
//! of its two pointer tables. The seal is keyed by that generation too. The dispatch-level
//! `RoutedExpertWeights::mutable_residency` hook is deliberately not set: it is the shared
//! QT44/QT53 gfx1201 path (an S3-owned dispatch surface) and would refuse MQ2-Lloyd, so the
//! capability stays DS4-local and each reader ([`ExpertConsumer`]) verifies it through
//! [`verify_binding`].

use std::sync::Arc;

use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{
    MoePointerEntries, MutableExpertResidency, MutableExpertResidencySpec, ResidencySnapshot,
    ResidencySpan, RoutedExpertWeights,
};
use hipfire_dispatch::pipeline::{
    ExpertBindingCache, ExpertMetadata, ExpertResource, ExpertResources, ExpertTable,
};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_runtime::expert_cache::catalog::TableExtent;
use rdna_compute::{DType, Gpu, GpuTensor};

use crate::deepseek4::{DeepseekV4LayerWeights, DeepseekV4Weights};
use crate::expert_source::{Ds4MappedOwner, TABLE_DOWN, TABLE_GATE_UP};

const DTYPE: DType = DType::MQ2G256Lloyd;

pub(crate) struct RoutedSeal {
    /// Owner generation and device address of the layer's gate/up pointer table.
    key: SealKey,
    pub(crate) table: ExpertTable,
    pub(crate) cache: ExpertBindingCache,
    pub(crate) experts: RoutedViews,
}

/// Identity of one sealed layer: the owner generation it was sealed under
/// ([`RESIDENT_GENERATION`] for the resident loader) and its pointer table's device address.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SealKey {
    generation: u64,
    table: usize,
}

/// The owner generation of a layer no mapped owner governs (the incumbent resident loader).
/// A real owner generation is never zero ([`fresh_owner_generation`]).
pub(crate) const RESIDENT_GENERATION: u64 = 0;

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
    /// Build the seal of `layer_idx` unless present. Host-only: no launch, so it
    /// is safe inside graph capture.
    ///
    /// Under a mapped owner the seal is keyed by the owner generation as well as the
    /// pointer-table address, and its construction first verifies the layer's
    /// mutable-residency binding.
    pub(crate) fn ensure(
        seals: &mut Vec<RoutedSeal>,
        gpu: &Gpu,
        weights: &DeepseekV4Weights,
        layer_idx: usize,
        hidden: usize,
        mi: usize,
    ) -> Result<(), String> {
        let layer = layer_of(weights, layer_idx)?;
        let table = table_key(layer)
            .ok_or_else(|| format!("deepseek4 l{layer_idx}: gate/up pointer table missing"))?;
        let consumer = score_consumer(weights, layer_idx);
        let generation = verify_binding(weights, layer_idx, consumer, None)?
            .map_or(RESIDENT_GENERATION, |image| image.generation);
        let key = SealKey { generation, table };
        if !seals.iter().any(|s| s.key == key) {
            seals.push(Self::build(gpu, layer, layer_idx, key, hidden, mi)?);
        }
        Ok(())
    }

    /// The seal [`Self::ensure`] built for `layer_idx` under the weights' current owner
    /// generation. A seal for the same table under another generation is a stale binding and
    /// refuses by name rather than reading as missing.
    pub(crate) fn find<'s>(
        seals: &'s [RoutedSeal],
        weights: &DeepseekV4Weights,
        layer_idx: usize,
    ) -> Result<&'s RoutedSeal, String> {
        let layer = layer_of(weights, layer_idx)?;
        let table = table_key(layer)
            .ok_or_else(|| format!("deepseek4 l{layer_idx}: gate/up pointer table missing"))?;
        let generation = weights
            .expert_owner
            .as_deref()
            .map_or(RESIDENT_GENERATION, MappedExperts::owner_generation);
        if let Some(seal) = seals
            .iter()
            .find(|s| s.key == SealKey { generation, table })
        {
            return Ok(seal);
        }
        match seals.iter().find(|s| s.key.table == table) {
            Some(stale) => Err(format!(
                "deepseek4 l{layer_idx}: sealed expert binding is stale: it was sealed under \
                 owner generation {}, the weights are owner generation {generation}",
                stale.key.generation
            )),
            None => Err(format!(
                "deepseek4 l{layer_idx}: sealed expert binding missing (ensure_routed_experts)"
            )),
        }
    }

    /// The owner generation this seal was built under ([`RESIDENT_GENERATION`] when resident).
    pub(crate) fn generation(&self) -> u64 {
        self.key.generation
    }

    fn build(
        gpu: &Gpu,
        layer: &DeepseekV4LayerWeights,
        layer_idx: usize,
        key: SealKey,
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

fn layer_of(
    weights: &DeepseekV4Weights,
    layer_idx: usize,
) -> Result<&DeepseekV4LayerWeights, String> {
    if let Some(layer) = weights.layers.get(layer_idx) {
        return Ok(layer);
    }
    if layer_idx == weights.layers.len() {
        if let Some(mtp) = weights.mtp_layer.as_ref() {
            return Ok(mtp);
        }
    }
    Err(format!("deepseek4 l{layer_idx}: no such routed layer"))
}

// ---------------------------------------------------------------------------------------------
// Mutable-residency capability of a mapped owner (expert cache slice S6b)
// ---------------------------------------------------------------------------------------------

/// The separate readers of a layer's routed pointer tables. Each one verifies the binding on
/// its own (a seal built by one reader never vouches for another): the sealed score decode, the
/// hash decode, the batched rows program and the MTP head's two entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertConsumer {
    ScoreDecode,
    HashDecode,
    RowsPrefill,
    MtpDecode,
    MtpRows,
}

impl ExpertConsumer {
    fn name(self) -> &'static str {
        match self {
            Self::ScoreDecode => "score decode",
            Self::HashDecode => "hash decode",
            Self::RowsPrefill => "rows prefill",
            Self::MtpDecode => "MTP decode",
            Self::MtpRows => "MTP rows",
        }
    }

    /// The sealed score decode re-checks the cheap identity every token (its seal validated the
    /// full spans at bind time); every other reader re-validates the complete binding.
    fn full(self) -> bool {
        self != Self::ScoreDecode
    }
}

/// The single-row consumer of `layer_idx`: the trunk's score decode or the MTP head's.
pub(crate) fn score_consumer(weights: &DeepseekV4Weights, layer_idx: usize) -> ExpertConsumer {
    if layer_idx >= weights.layers.len() {
        ExpertConsumer::MtpDecode
    } else {
        ExpertConsumer::ScoreDecode
    }
}

/// The batched-rows consumer of `layer_idx`: the trunk's rows program or the MTP head's.
pub(crate) fn rows_consumer(weights: &DeepseekV4Weights, layer_idx: usize) -> ExpertConsumer {
    if layer_idx >= weights.layers.len() {
        ExpertConsumer::MtpRows
    } else {
        ExpertConsumer::RowsPrefill
    }
}

/// The mutable-residency capability of one mapped layer: its owner generation, its two pointer
/// tables' catalog identity ([`TableExtent`]) and the canonical image the tables were filled
/// from. No slot is bound yet (the cache fill/evict engine is a later slice), so the tables
/// hold the canonical image for the whole generation and any publication is a refusal.
pub(crate) struct MappedLayer {
    group_id: u32,
    residency: MutableExpertResidency,
    gate_up_table: TableExtent,
    down_table: TableExtent,
    gate_up_stride: usize,
    down_stride: usize,
    /// The published mirror of the canonical tables (epoch 0): what `expert_ptrs_host` carries.
    image: ResidencySnapshot,
}

/// The mapped owner of a loaded model plus one capability per group.
pub(crate) struct MappedExperts {
    owner: Arc<Ds4MappedOwner>,
    layers: Vec<MappedLayer>,
}

/// What a verified binding authorizes a consumer to read: the owner generation and the host
/// image of the pointer tables the device holds.
pub(crate) struct VerifiedImage<'w> {
    pub(crate) generation: u64,
    pub(crate) entries: MoePointerEntries<'w>,
}

fn entries_match(device: &[u64], image: &[usize]) -> bool {
    device.len() == image.len() && device.iter().zip(image).all(|(a, b)| *a == *b as u64)
}

fn spans(entries: &[u64], bytes: usize) -> Vec<ResidencySpan> {
    entries
        .iter()
        .map(|&address| ResidencySpan {
            address: address as usize,
            bytes,
        })
        .collect()
}

impl MappedExperts {
    /// Bind every group of `owner` to the layer weights the loader just filled: the blobs must
    /// be the owner's, the pointer entries the canonical `base + expert * stride` addresses and
    /// the two tables the layer's own. Builds one [`MutableExpertResidency`] per group.
    pub(crate) fn seal(owner: Ds4MappedOwner, weights: &DeepseekV4Weights) -> Result<Self, String> {
        let owner = Arc::new(owner);
        let bytes = *owner.source().expert_bytes();
        let (gate_up_stride, down_stride) = (bytes.gate_up() as usize, bytes.down as usize);
        let n = owner.source().geometry().n_experts;
        let mut layers = Vec::with_capacity(owner.groups().len());
        for group in owner.groups() {
            let layer_idx = group.group_id as usize;
            let err = |what: String| format!("deepseek4 l{layer_idx}: mapped binding: {what}");
            let layer = layer_of(weights, layer_idx)?;
            let base = |blob: &Option<GpuTensor>| blob.as_ref().map(|t| t.buf.as_ptr() as u64);
            if base(&layer.expert_gate_up_blob) != Some(group.gate_up.device_address)
                || base(&layer.expert_w2_blob) != Some(group.down.device_address)
            {
                return Err(err("the layer's blobs are not the owner's blobs".into()));
            }
            if layer.expert_gate_up_stride != gate_up_stride
                || layer.expert_w2_stride != down_stride
            {
                return Err(err(format!(
                    "expert strides ({}, {}) are not the catalog's ({gate_up_stride}, {down_stride})",
                    layer.expert_gate_up_stride, layer.expert_w2_stride
                )));
            }
            let canonical = |entries: &[u64], base: u64, stride: usize| {
                entries.len() == n
                    && entries
                        .iter()
                        .enumerate()
                        .all(|(e, &a)| a == base + (e * stride) as u64)
            };
            if !canonical(
                &layer.expert_gate_up_ptr_entries,
                group.gate_up.device_address,
                gate_up_stride,
            ) || !canonical(
                &layer.expert_w2_ptr_entries,
                group.down.device_address,
                down_stride,
            ) {
                return Err(err(
                    "the pointer entries are not the canonical blob-base + expert * stride image"
                        .into(),
                ));
            }
            let (Some(gate_up_ptrs), Some(down_ptrs)) = (
                layer.expert_gate_up_ptrs.as_ref(),
                layer.expert_w2_ptrs.as_ref(),
            ) else {
                return Err(err("pointer table missing".into()));
            };
            let extent = |table: u32, tensor: &GpuTensor| {
                owner
                    .source()
                    .table_extent(group.group_id, table, tensor.buf.as_ptr() as u64)
                    .map_err(|e| err(e.to_string()))
            };
            let gate_up_table = extent(TABLE_GATE_UP, gate_up_ptrs)?;
            let down_table = extent(TABLE_DOWN, down_ptrs)?;
            let table_span = |extent: &TableExtent| ResidencySpan {
                address: extent.address as usize,
                bytes: extent.bytes as usize,
            };
            let spec = MutableExpertResidencySpec {
                owner_generation: owner.owner_generation(),
                canonical_gate_up: spans(&layer.expert_gate_up_ptr_entries, gate_up_stride),
                canonical_down: spans(&layer.expert_w2_ptr_entries, down_stride),
                gate_up_table: table_span(&gate_up_table),
                down_table: table_span(&down_table),
                slots: Vec::new(),
                keepalive: Arc::clone(&owner) as Arc<dyn std::any::Any + Send + Sync>,
            };
            // SAFETY: every span names memory the layer weights own for the life of the model
            // (the pinned blobs and the two device tables, freed only by `free_gpu`); the
            // tables were uploaded synchronously by the loader and already hold exactly the
            // canonical addresses; the canonical bytes are filled once and never rewritten;
            // there are no slots. The keepalive retains the owner's identity and ranges, not
            // the allocations: physical teardown stays the explicit unload.
            let residency = unsafe { MutableExpertResidency::new(spec) }.map_err(err)?;
            let image = residency.snapshot().map_err(err)?;
            layers.push(MappedLayer {
                group_id: group.group_id,
                residency,
                gate_up_table,
                down_table,
                gate_up_stride,
                down_stride,
                image,
            });
        }
        Ok(Self { owner, layers })
    }

    pub(crate) fn owner(&self) -> &Ds4MappedOwner {
        &self.owner
    }

    pub(crate) fn owner_generation(&self) -> u64 {
        self.owner.owner_generation()
    }

    fn verify(
        &self,
        layer_idx: usize,
        layer: &DeepseekV4LayerWeights,
        consumer: ExpertConsumer,
        expected_generation: Option<u64>,
    ) -> Result<VerifiedImage<'_>, String> {
        let refuse = |why: String| {
            format!(
                "deepseek4 l{layer_idx}: {} refuses the mapped expert binding: {why}",
                consumer.name()
            )
        };
        let generation = self.owner_generation();
        if let Some(expected) = expected_generation {
            if expected != generation {
                return Err(refuse(format!(
                    "stale owner generation: bound under generation {expected}, the weights' \
                     owner is generation {generation}"
                )));
            }
        }
        let capability = self
            .layers
            .iter()
            .find(|capability| capability.group_id as usize == layer_idx)
            .ok_or_else(|| refuse("the owner has no group for this layer".into()))?;
        capability
            .verify(&self.owner, layer, consumer)
            .map_err(refuse)?;
        Ok(VerifiedImage {
            generation,
            entries: MoePointerEntries {
                gate_up: &capability.image.gate_up[..],
                down: &capability.image.down[..],
            },
        })
    }
}

impl MappedLayer {
    fn verify(
        &self,
        owner: &Ds4MappedOwner,
        layer: &DeepseekV4LayerWeights,
        consumer: ExpertConsumer,
    ) -> Result<(), String> {
        let generation = owner.owner_generation();
        if self.residency.owner_generation() != generation {
            return Err(format!(
                "the capability is owner generation {}, the owner is generation {generation}",
                self.residency.owner_generation()
            ));
        }
        let (Some(gate_up_ptrs), Some(down_ptrs)) = (
            layer.expert_gate_up_ptrs.as_ref(),
            layer.expert_w2_ptrs.as_ref(),
        ) else {
            return Err("pointer table missing".into());
        };
        // Table identity: the catalog's table extent of the tables the kernels will read.
        let table = |table: u32, tensor: &GpuTensor| {
            owner
                .source()
                .table_extent(self.group_id, table, tensor.buf.as_ptr() as u64)
                .map_err(|e| e.to_string())
        };
        if table(TABLE_GATE_UP, gate_up_ptrs)? != self.gate_up_table
            || table(TABLE_DOWN, down_ptrs)? != self.down_table
        {
            return Err(
                "the pointer tables are not the tables this binding was made for \
                        (catalog table extent changed)"
                    .into(),
            );
        }
        if !entries_match(&layer.expert_gate_up_ptr_entries, &self.image.gate_up)
            || !entries_match(&layer.expert_w2_ptr_entries, &self.image.down)
        {
            return Err(
                "the layer's pointer entries differ from the canonical image they were bound to"
                    .into(),
            );
        }
        // Any publication since load moved the tables off the image this consumer reads.
        self.residency
            .validate_published_entries(&self.image.gate_up, &self.image.down)?;
        if consumer.full() {
            let span = |extent: &TableExtent| ResidencySpan {
                address: extent.address as usize,
                bytes: extent.bytes as usize,
            };
            self.residency.validate_binding(
                generation,
                &spans(&layer.expert_gate_up_ptr_entries, self.gate_up_stride),
                &spans(&layer.expert_w2_ptr_entries, self.down_stride),
                span(&self.gate_up_table),
                span(&self.down_table),
            )?;
        }
        Ok(())
    }
}

/// Verify `layer_idx`'s mapped binding for `consumer`. `Ok(None)` when no mapped owner governs
/// these weights (the resident loader: nothing to verify, behaviour unchanged). Under a mapped
/// owner it refuses a stale generation (`expected_generation` is the generation the consumer
/// bound earlier; `None` for a reader that binds nothing), a moved table, a changed pointer
/// image and any publication since load, and returns the host image of the pointer tables the
/// kernels read.
pub(crate) fn verify_binding<'w>(
    weights: &'w DeepseekV4Weights,
    layer_idx: usize,
    consumer: ExpertConsumer,
    expected_generation: Option<u64>,
) -> Result<Option<VerifiedImage<'w>>, String> {
    let Some(mapped) = weights.expert_owner.as_deref() else {
        return match expected_generation {
            Some(expected) if expected != RESIDENT_GENERATION => Err(format!(
                "deepseek4 l{layer_idx}: {} refuses the expert binding: stale owner generation: \
                 bound under generation {expected}, the weights have no mapped owner",
                consumer.name()
            )),
            _ => Ok(None),
        };
    };
    let layer = layer_of(weights, layer_idx)?;
    mapped
        .verify(layer_idx, layer, consumer, expected_generation)
        .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Mq2rBackend;
    use crate::expert_source::{
        Ds4ExpertSource, Ds4Geometry, HostSpan, MappedBlob, MappedGroup, MtpSource, TensorFacts,
        TensorIndex, ROUTED_QUANT_TYPE,
    };
    use std::collections::BTreeMap;

    const GEOMETRY: Ds4Geometry = Ds4Geometry {
        hidden: 256,
        moe_intermediate: 256,
        n_experts: 4,
        trunk_layers: 3,
    };
    const GENERATION: u64 = 11;
    const ALL: [ExpertConsumer; 5] = [
        ExpertConsumer::ScoreDecode,
        ExpertConsumer::HashDecode,
        ExpertConsumer::RowsPrefill,
        ExpertConsumer::MtpDecode,
        ExpertConsumer::MtpRows,
    ];

    struct FakeFile(BTreeMap<String, TensorFacts>, u64);

    impl TensorIndex for FakeFile {
        fn label(&self) -> String {
            "fake.hfq".into()
        }
        fn fingerprint(&self) -> String {
            "fp-fake".into()
        }
        fn file_len(&self) -> u64 {
            self.1
        }
        fn is_overlaid(&self) -> bool {
            false
        }
        fn tensor(&self, name: &str) -> Option<TensorFacts> {
            self.0.get(name).copied()
        }
    }

    fn fake_file() -> FakeFile {
        let bytes = GEOMETRY.expert_bytes().unwrap();
        let mut tensors = BTreeMap::new();
        let mut offset = 4096u64;
        let prefixes = (0..GEOMETRY.trunk_layers)
            .map(|l| format!("layers.{l}"))
            .chain(["mtp.0".to_string()]);
        for prefix in prefixes {
            for e in 0..GEOMETRY.n_experts {
                for (part, len) in [("w1", bytes.gate), ("w3", bytes.up), ("w2", bytes.down)] {
                    tensors.insert(
                        format!("{prefix}.ffn.experts.{e}.{part}.weight"),
                        TensorFacts {
                            offset,
                            len,
                            quant_type: ROUTED_QUANT_TYPE,
                        },
                    );
                    offset += len;
                }
            }
        }
        FakeFile(tensors, offset)
    }

    /// A device tensor over a fake address (never dereferenced).
    fn tensor(address: u64, bytes: usize) -> GpuTensor {
        GpuTensor {
            // SAFETY: the address is never dereferenced; the tensor is never freed.
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(address as *mut _, bytes) },
            shape: vec![bytes],
            dtype: DType::Raw,
        }
    }

    /// Weights whose routed layers are the canonical image of `owner`'s blobs, plus the owner.
    fn mapped_weights() -> DeepseekV4Weights {
        let source =
            Ds4ExpertSource::build(1, GENERATION, GEOMETRY, &fake_file(), MtpSource::InBand)
                .unwrap();
        let bytes = *source.expert_bytes();
        let n = GEOMETRY.n_experts;
        let (gate_up_stride, down_stride) = (bytes.gate_up() as usize, bytes.down as usize);
        let mut layers = Vec::new();
        let mut groups = Vec::new();
        for (i, group) in source.groups().iter().enumerate() {
            let blob = |slot: usize, len: usize| MappedBlob {
                device_address: 0x7000_0000_0000 + ((slot as u64) << 32),
                host: HostSpan {
                    addr: 0x1000_0000_0000 + (slot << 32),
                    len,
                },
            };
            let mapped = MappedGroup {
                group_id: group.group_id(),
                gate_up: blob(2 * i, n * gate_up_stride),
                down: blob(2 * i + 1, n * down_stride),
            };
            let table = |k: u64| 0x6000_0000_0000 + (i as u64) * 0x1000 + k * 0x800;
            let mut layer = DeepseekV4LayerWeights::new_empty(0);
            layer.expert_gate_up_blob = Some(tensor(mapped.gate_up.device_address, 0));
            layer.expert_w2_blob = Some(tensor(mapped.down.device_address, 0));
            layer.expert_gate_up_ptrs = Some(tensor(table(0), n * 8));
            layer.expert_w2_ptrs = Some(tensor(table(1), n * 8));
            layer.expert_gate_up_stride = gate_up_stride;
            layer.expert_w2_stride = down_stride;
            layer.expert_gate_up_ptr_entries = (0..n)
                .map(|e| mapped.gate_up.device_address + (e * gate_up_stride) as u64)
                .collect();
            layer.expert_w2_ptr_entries = (0..n)
                .map(|e| mapped.down.device_address + (e * down_stride) as u64)
                .collect();
            layers.push(layer);
            groups.push(mapped);
        }
        let mtp_layer = layers.pop();
        let mut weights = DeepseekV4Weights {
            mq2r_backend: Mq2rBackend::Portable,
            token_embd: None,
            output_norm: None,
            head: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: 1.0,
            layers,
            mtp_layer,
            dspark: None,
            expert_owner: None,
            _scaffold: (),
        };
        let owner = Ds4MappedOwner::new(source, groups).unwrap();
        let sealed = MappedExperts::seal(owner, &weights).unwrap();
        weights.expert_owner = Some(Arc::new(sealed));
        weights
    }

    #[test]
    fn every_consumer_verifies_the_canonical_image_of_every_layer() {
        let weights = mapped_weights();
        assert_eq!(weights.layers.len(), GEOMETRY.trunk_layers);
        for layer_idx in 0..=GEOMETRY.trunk_layers {
            for consumer in ALL {
                let image = verify_binding(&weights, layer_idx, consumer, None)
                    .unwrap_or_else(|e| panic!("l{layer_idx} {consumer:?}: {e}"))
                    .expect("a mapped owner authorizes an image");
                let layer = layer_of(&weights, layer_idx).unwrap();
                assert_eq!(image.generation, GENERATION);
                assert!(entries_match(
                    &layer.expert_gate_up_ptr_entries,
                    image.entries.gate_up
                ));
                assert!(entries_match(
                    &layer.expert_w2_ptr_entries,
                    image.entries.down
                ));
                // Binding to the generation the consumer sealed under passes too.
                assert!(verify_binding(&weights, layer_idx, consumer, Some(GENERATION)).is_ok());
            }
        }
        // The MTP head is its own group (index `trunk_layers`).
        assert_eq!(score_consumer(&weights, 0), ExpertConsumer::ScoreDecode);
        assert_eq!(
            score_consumer(&weights, GEOMETRY.trunk_layers),
            ExpertConsumer::MtpDecode
        );
        assert_eq!(rows_consumer(&weights, 0), ExpertConsumer::RowsPrefill);
        assert_eq!(
            rows_consumer(&weights, GEOMETRY.trunk_layers),
            ExpertConsumer::MtpRows
        );
    }

    #[test]
    fn a_stale_generation_refuses_in_every_consumer() {
        let weights = mapped_weights();
        for consumer in ALL {
            for stale in [GENERATION - 1, GENERATION + 1, RESIDENT_GENERATION] {
                let error = verify_binding(&weights, 1, consumer, Some(stale))
                    .err()
                    .unwrap_or_else(|| panic!("{consumer:?} accepted generation {stale}"));
                assert!(error.contains("stale owner generation"), "{error}");
                assert!(error.contains(consumer.name()), "{error}");
                assert!(
                    error.contains(&format!("bound under generation {stale}")),
                    "{error}"
                );
            }
        }
    }

    #[test]
    fn resident_weights_have_nothing_to_verify_and_refuse_a_mapped_seal() {
        let mut weights = mapped_weights();
        weights.expert_owner = None;
        for consumer in ALL {
            assert!(verify_binding(&weights, 0, consumer, None)
                .unwrap()
                .is_none());
            assert!(
                verify_binding(&weights, 0, consumer, Some(RESIDENT_GENERATION))
                    .unwrap()
                    .is_none()
            );
            // A seal made under a mapped generation cannot serve weights that lost the owner.
            let error = verify_binding(&weights, 0, consumer, Some(GENERATION))
                .err()
                .expect("stale");
            assert!(error.contains("stale owner generation"), "{error}");
        }
    }

    #[test]
    fn a_moved_table_or_changed_image_refuses_in_every_consumer() {
        for consumer in ALL {
            // The gate/up table is another allocation: the catalog table extent differs.
            let mut weights = mapped_weights();
            weights.layers[1].expert_gate_up_ptrs = Some(tensor(0x6fff_0000_0000, 4 * 8));
            let error = verify_binding(&weights, 1, consumer, None)
                .err()
                .unwrap_or_else(|| panic!("{consumer:?} accepted a moved table"));
            assert!(error.contains("catalog table extent changed"), "{error}");

            // The down table moved too.
            let mut weights = mapped_weights();
            weights.layers[1].expert_w2_ptrs = Some(tensor(0x6fff_0000_0000, 4 * 8));
            assert!(verify_binding(&weights, 1, consumer, None).is_err());

            // The host mirror of the entries no longer is the bound canonical image.
            let mut weights = mapped_weights();
            weights.layers[2].expert_w2_ptr_entries[3] += 8;
            let error = verify_binding(&weights, 2, consumer, None)
                .err()
                .unwrap_or_else(|| panic!("{consumer:?} accepted a changed image"));
            assert!(error.contains("canonical image"), "{error}");

            // The MTP head is verified on its own group.
            let mut weights = mapped_weights();
            weights
                .mtp_layer
                .as_mut()
                .unwrap()
                .expert_gate_up_ptr_entries[0] += 16;
            assert!(verify_binding(&weights, GEOMETRY.trunk_layers, consumer, None).is_err());
            assert!(verify_binding(&weights, 0, consumer, None).is_ok());
        }
    }

    #[test]
    fn the_capability_is_keyed_by_the_owner_generation_and_table_extents() {
        let weights = mapped_weights();
        let mapped = weights.expert_owner.as_deref().unwrap();
        assert_eq!(mapped.owner_generation(), GENERATION);
        let layer = &mapped.layers[1];
        assert_eq!(layer.residency.owner_generation(), GENERATION);
        let table = weights.layers[1]
            .expert_gate_up_ptrs
            .as_ref()
            .unwrap()
            .buf
            .as_ptr() as u64;
        assert_eq!(
            layer.gate_up_table,
            mapped
                .owner()
                .source()
                .table_extent(1, TABLE_GATE_UP, table)
                .unwrap()
        );
        // The canonical views are the blobs' per-expert spans; no slot is bound yet.
        assert!(layer.residency.slots().is_empty());
        assert_eq!(
            layer.residency.canonical_gate_up().len(),
            GEOMETRY.n_experts
        );
    }

    #[test]
    fn sealing_refuses_layers_that_are_not_the_owners_canonical_image() {
        let weights = mapped_weights();
        let owner = weights.expert_owner.as_deref().unwrap().owner().clone();

        let mut other_blob = mapped_weights();
        other_blob.expert_owner = None;
        other_blob.layers[0].expert_gate_up_blob = Some(tensor(0x1234_0000, 0));
        let error = MappedExperts::seal(owner.clone(), &other_blob)
            .err()
            .unwrap();
        assert!(error.contains("not the owner's blobs"), "{error}");

        let mut ragged = mapped_weights();
        ragged.expert_owner = None;
        ragged.layers[1].expert_gate_up_ptr_entries[2] += 4;
        let error = MappedExperts::seal(owner.clone(), &ragged).err().unwrap();
        assert!(error.contains("canonical blob-base"), "{error}");

        let mut strided = mapped_weights();
        strided.expert_owner = None;
        strided.layers[1].expert_w2_stride += 256;
        let error = MappedExperts::seal(owner, &strided).err().unwrap();
        assert!(error.contains("strides"), "{error}");
    }

    #[test]
    fn find_reports_a_missing_seal_by_name() {
        let weights = mapped_weights();
        let error = RoutedSeal::find(&[], &weights, 0).err().unwrap();
        assert!(error.contains("sealed expert binding missing"), "{error}");
        let error = RoutedSeal::find(&[], &weights, 99).err().unwrap();
        assert!(error.contains("no such routed layer"), "{error}");
    }
}
