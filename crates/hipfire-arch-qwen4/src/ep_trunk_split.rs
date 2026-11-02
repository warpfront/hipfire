// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Output-axis split of the replicated Flash-Next trunk projections across the
//! EP mesh (#862, `HIPFIRE_EP_TRUNK_SPLIT`).
//!
//! Every rank keeps the whole trunk resident and runs the unchanged program.
//! For the projections in [`split_census`] (QSA q/k/v/indexer/output, GDN
//! qkv/z/output, the Q8 decode copies of the HC input-mix down, the LM head,
//! and the same roles of the MTP head) rank `r` of `N` projects only output
//! rows `[r * m / N, (r + 1) * m / N)` of the weight, complete rows with their
//! unchanged reduction order, and the shared projection seam
//! (`hipfire_dispatch::pipeline::projection_split`) gathers every rank's band
//! into the full output with copies only before any consumer reads it. The
//! LM head is gathered whole, so the unchanged full-vector argmax runs on it.
//!
//! The split is gated per forward shape: `prefill` bands chunk forwards
//! (more rows than the few-row limit), `decode` bands decode and verify
//! forwards of at most eight rows; the row-exact wide verify (9..=64 rows)
//! always projects whole. A call whose bands do not fit the exchange buffer
//! (an all-rows LM head of a prefill chunk) projects whole too. Both are
//! functions of the call's shape, so every rank takes the same decision.
//!
//! Layout identity: the census (role, shape, dtype of every banded weight)
//! and the mode must be equal on every rank at attach, and every exchange
//! carries a geometry tag the transport compares across ranks.

use crate::bundle::Qwen4Bundle;
use crate::gpu_forward::DECODE_Q8_PER_LAYER;
use crate::weights::{QsaWeights, TensorRef};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::projection_split::{ProjectionSplit, SplitBand};
use hipfire_runtime::ep_row_gather::{EpBand, EpRowGather};
use rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// `HIPFIRE_EP_TRUNK_SPLIT`: unset, `0` or `off` keep the trunk replicated;
/// `prefill`, `decode` or `on` (both) split it across the EP ranks.
pub const QWEN4_EP_TRUNK_SPLIT_ENV: &str = "HIPFIRE_EP_TRUNK_SPLIT";

/// Exchange buffers per rank (alternating, see `EpRowGather`'s hazard rule).
const SPLIT_BUFFERS: usize = 2;

/// Which forward shapes split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrunkSplitMode {
    pub prefill: bool,
    pub decode: bool,
}

impl TrunkSplitMode {
    /// Parse the env value; `Ok(None)` is off.
    pub fn parse(value: Option<&str>) -> Result<Option<Self>, String> {
        let (prefill, decode) = match value.map(str::trim) {
            None | Some("") | Some("0") | Some("off") => return Ok(None),
            Some("prefill") => (true, false),
            Some("decode") => (false, true),
            Some("on") | Some("1") => (true, true),
            Some(other) => {
                return Err(format!(
                    "{QWEN4_EP_TRUNK_SPLIT_ENV}={other:?}: expected off, prefill, decode or on"
                ))
            }
        };
        Ok(Some(Self { prefill, decode }))
    }

    pub(crate) fn from_env() -> Result<Option<Self>, String> {
        Self::parse(
            hipfire_config::developer_var(QWEN4_EP_TRUNK_SPLIT_ENV)
                .ok()
                .as_deref(),
        )
    }

    /// Whether a call of `route` routing rows splits (`few_row_limit` is the
    /// forward's admitted exact few-row cap).
    fn admits(self, route: usize, few_row_limit: usize) -> bool {
        if route > few_row_limit {
            self.prefill
        } else {
            self.decode && route <= FEW_ROW_ROUTE_ROWS
        }
    }
}

/// One banded weight of a rank's census: its role, logical `[m, k]` and
/// stored dtype. Compared across ranks; the device address is not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitWeight {
    pub label: String,
    pub m: usize,
    pub k: usize,
    pub dtype: DType,
}

/// The rank's banded weights with their device addresses, in census order.
/// BF16 weights are left out (the seam never bands them; see
/// `projection_split`), so the HC input-mix down splits only through its Q8
/// decode copies.
pub(crate) fn split_census(bundle: &Qwen4Bundle) -> Result<Vec<(SplitWeight, usize)>, String> {
    let weights = &bundle.weights;
    let mut census = Vec::new();
    let mut push = |label: String, reference: &TensorRef| -> Result<(), String> {
        let tensor = weights
            .resident(reference)
            .map_err(|error| format!("trunk split census {label}: {error}"))?;
        let &[m, k] = &reference.shape[..] else {
            return Err(format!(
                "trunk split census {label}: shape {:?}",
                reference.shape
            ));
        };
        if tensor.dtype != DType::BF16 {
            census.push((
                SplitWeight {
                    label,
                    m,
                    k,
                    dtype: tensor.dtype,
                },
                tensor.buf.as_ptr() as usize,
            ));
        }
        Ok(())
    };
    let qsa = |push: &mut dyn FnMut(String, &TensorRef) -> Result<(), String>,
               prefix: &str,
               qsa: &QsaWeights|
     -> Result<(), String> {
        push(format!("{prefix}.indexer_qk"), &qsa.indexer_qk)?;
        push(format!("{prefix}.q"), &qsa.q)?;
        push(format!("{prefix}.k"), &qsa.k)?;
        push(format!("{prefix}.v"), &qsa.v)?;
        push(format!("{prefix}.o"), &qsa.output)
    };
    for layer in &weights.layer_refs {
        let prefix = format!("layer{}", layer.layer);
        if let Some(attention) = &layer.attention {
            qsa(&mut push, &prefix, attention)?;
        }
        if let Some(gdn) = &layer.gdn {
            push(format!("{prefix}.gdn_qkv"), &gdn.qkv)?;
            push(format!("{prefix}.gdn_z"), &gdn.z)?;
            push(format!("{prefix}.gdn_out"), &gdn.output)?;
        }
    }
    push("lm_head".into(), &weights.root.lm_head)?;
    if let Some(mtp) = &weights.mtp {
        qsa(&mut push, "mtp", &mtp.attention)?;
    }
    // The HC input-mix down's Q8 decode copies (attn = slot 0, mlp = slot 2).
    if let Some(forward) = bundle.execution.as_ref() {
        for (index, copy) in forward.decode_q8.iter().enumerate() {
            let slot = index % DECODE_Q8_PER_LAYER;
            if slot != 0 && slot != 2 {
                continue;
            }
            let layer = index / DECODE_Q8_PER_LAYER;
            let reference = &weights.layer_refs[layer];
            let source = if slot == 0 {
                &reference.attn_hyper.input_mix_down
            } else {
                &reference.mlp_hyper.input_mix_down
            };
            let &[m, k] = &source.shape[..] else {
                return Err(format!(
                    "trunk split census: HC down shape {:?}",
                    source.shape
                ));
            };
            let role = if slot == 0 { "attn" } else { "mlp" };
            census.push((
                SplitWeight {
                    label: format!("layer{layer}.{role}_hc_down_q8"),
                    m,
                    k,
                    dtype: copy.dtype,
                },
                copy.buf.as_ptr() as usize,
            ));
        }
    }
    Ok(census)
}

/// Bytes one exchange buffer holds: the widest layer's banded F32 outputs at
/// `chunk_rows` rows (each band padded to 256 bytes), or a 64-row LM head,
/// whichever is larger.
pub(crate) fn split_buffer_bytes(census: &[SplitWeight], ranks: usize, chunk_rows: usize) -> usize {
    let band = |w: &SplitWeight| (w.m / ranks * 4 * chunk_rows).next_multiple_of(256);
    let mut per_layer = std::collections::BTreeMap::<&str, usize>::new();
    let mut head = 0;
    for weight in census {
        if weight.label == "lm_head" {
            head = (weight.m / ranks * 4 * 64).next_multiple_of(256);
            continue;
        }
        let layer = weight.label.split('.').next().unwrap_or("");
        *per_layer.entry(layer).or_default() += band(weight);
    }
    per_layer.into_values().max().unwrap_or(0).max(head)
}

/// One rank's split: the exchange buffers it owns (allocated before the mesh
/// enables peers) and, once every rank registered them, the hook it installs.
pub(crate) struct TrunkSplitRank {
    rank: usize,
    buffers: Vec<GpuTensor>,
    hook: Option<Arc<Qwen4TrunkSplitHook>>,
}

impl TrunkSplitRank {
    /// Allocate rank `rank`'s exchange buffers of `bytes` each.
    pub(crate) fn allocate(rank: usize, gpu: &mut Gpu, bytes: usize) -> Result<Self, String> {
        let mut split = Self {
            rank,
            buffers: Vec::with_capacity(SPLIT_BUFFERS),
            hook: None,
        };
        for _ in 0..SPLIT_BUFFERS {
            match gpu.alloc_tensor(&[bytes.div_ceil(4)], DType::F32) {
                Ok(buffer) => split.buffers.push(buffer),
                Err(error) => {
                    let _ = split.release(gpu);
                    return Err(format!("trunk split exchange buffer: {error}"));
                }
            }
        }
        Ok(split)
    }

    pub(crate) fn buffers(&self) -> Vec<&GpuTensor> {
        self.buffers.iter().collect()
    }

    /// The installed hook, once [`Self::install`] ran.
    pub(crate) fn hook(&self) -> Option<Arc<Qwen4TrunkSplitHook>> {
        self.hook.clone()
    }

    /// Build the rank's hook over its `census` (this rank's device addresses)
    /// and the built `transport`.
    pub(crate) fn install(
        &mut self,
        mode: TrunkSplitMode,
        census: &[(SplitWeight, usize)],
        transport: Arc<EpRowGather>,
    ) {
        self.hook = Some(Arc::new(Qwen4TrunkSplitHook {
            rank: self.rank,
            ranks: transport.ranks(),
            mode,
            weights: census.iter().map(|(w, ptr)| (*ptr, w.m)).collect(),
            buffers: self
                .buffers
                .iter()
                .map(|buffer| {
                    let bytes = buffer.buf.size().min(transport.spec().buffer_bytes());
                    (buffer.buf.as_ptr() as usize, bytes)
                })
                .collect(),
            transport,
            next: AtomicUsize::new(0),
        }));
    }

    /// Release the rank's exchange events (when installed) and free its
    /// buffers, on the rank's thread after its device drained.
    pub(crate) fn release(self, gpu: &mut Gpu) -> Result<(), String> {
        let mut first = self.hook.and_then(|hook| {
            hook.transport
                .release(self.rank, gpu)
                .map_err(|error| format!("release trunk split events: {error}"))
                .err()
        });
        for buffer in self.buffers {
            if let Err(error) = gpu.free_tensor(buffer) {
                first.get_or_insert(format!("free trunk split buffer: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

/// Every rank's census (role, shape, dtype in order) and chunk rows must be
/// rank 0's: a rank with another layout would exchange mismatched bands.
pub(crate) fn agree_layouts(layouts: &[(Vec<SplitWeight>, usize)]) -> Result<(), String> {
    let Some((first, rest)) = layouts.split_first() else {
        return Ok(());
    };
    for (index, layout) in rest.iter().enumerate() {
        if layout != first {
            let rank = index + 1;
            let detail = first
                .0
                .iter()
                .zip(&layout.0)
                .find(|(a, b)| a != b)
                .map(|(a, b)| format!("{a:?} vs {b:?}"))
                .unwrap_or_else(|| {
                    format!(
                        "{} vs {} weights, chunk rows {} vs {}",
                        first.0.len(),
                        layout.0.len(),
                        first.1,
                        layout.1
                    )
                });
            return Err(format!(
                "trunk split layout of rank {rank} differs from rank 0: {detail}"
            ));
        }
    }
    Ok(())
}

/// The [`ProjectionSplit`] of one rank.
pub(crate) struct Qwen4TrunkSplitHook {
    rank: usize,
    ranks: usize,
    mode: TrunkSplitMode,
    /// Banded weights by (device address, logical rows): a row view of a
    /// banded weight never matches, so nested entry points never band twice.
    weights: HashSet<(usize, usize)>,
    /// Device address and bytes of the rank's exchange buffers
    /// ([`TrunkSplitRank`] owns them).
    buffers: Vec<(usize, usize)>,
    transport: Arc<EpRowGather>,
    next: AtomicUsize,
}

impl ProjectionSplit for Qwen4TrunkSplitHook {
    fn band(&self, gpu: &Gpu, weight: &WeightRef<'_>, rows: usize) -> Option<(usize, usize)> {
        if !self.mode.admits(gpu.route_rows(rows), gpu.few_row_limit())
            || !weight.m.is_multiple_of(self.ranks)
            || !self
                .weights
                .contains(&(weight.buf.buf.as_ptr() as usize, weight.m))
        {
            return None;
        }
        let width = weight.m / self.ranks;
        Some((self.rank * width, (self.rank + 1) * width))
    }

    fn next_buffer(&self, need: usize) -> Option<(usize, GpuTensor)> {
        if self.buffers.first().is_none_or(|&(_, bytes)| need > bytes) {
            return None;
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.buffers.len();
        let (ptr, bytes) = self.buffers[index];
        // SAFETY: a non-owning view of the rank's live exchange buffer.
        let buf = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut _, bytes) };
        let view = GpuTensor {
            buf,
            shape: vec![bytes / 4],
            dtype: DType::F32,
        };
        Some((index, view))
    }

    fn exchange(
        &self,
        gpu: &mut Gpu,
        buffer: usize,
        bands: &[SplitBand<'_>],
    ) -> Result<(), String> {
        let mut tag = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |value: usize| {
            tag = (tag ^ value as u64).wrapping_mul(0x0100_0000_01b3);
        };
        let gathered: Vec<EpBand<'_>> = bands
            .iter()
            .map(|band| {
                mix(band.offset);
                mix(band.band_bytes);
                mix(band.rows);
                EpBand {
                    offset: band.offset,
                    band_bytes: band.band_bytes,
                    rows: band.rows,
                    dst: band.dst,
                }
            })
            .collect();
        self.transport
            .exchange_bands(gpu, self.rank, buffer, tag, &gathered)
            .map_err(|error| format!("qwen4 EP trunk split rank {}: {error}", self.rank))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse() {
        assert_eq!(TrunkSplitMode::parse(None), Ok(None));
        assert_eq!(TrunkSplitMode::parse(Some("off")), Ok(None));
        assert_eq!(TrunkSplitMode::parse(Some("0")), Ok(None));
        assert_eq!(
            TrunkSplitMode::parse(Some("prefill")),
            Ok(Some(TrunkSplitMode {
                prefill: true,
                decode: false
            }))
        );
        assert_eq!(
            TrunkSplitMode::parse(Some("on")),
            Ok(Some(TrunkSplitMode {
                prefill: true,
                decode: true
            }))
        );
        assert!(TrunkSplitMode::parse(Some("yes")).is_err());
    }

    #[test]
    fn prefill_and_decode_gate_separately_and_wide_verify_stays_whole() {
        let prefill = TrunkSplitMode {
            prefill: true,
            decode: false,
        };
        let decode = TrunkSplitMode {
            prefill: false,
            decode: true,
        };
        // A chunk forward (route rows above the few-row limit).
        assert!(prefill.admits(2048, 8) && !decode.admits(2048, 8));
        // Decode and few-row verify.
        assert!(decode.admits(1, 8) && decode.admits(8, 8) && !prefill.admits(1, 8));
        // A row-exact wide verify (limit raised to 64): whole in every mode.
        assert!(!decode.admits(16, 64) && !prefill.admits(16, 64));
    }

    #[test]
    fn mismatched_layouts_are_rejected() {
        let weight = |label: &str, m: usize, dtype: DType| SplitWeight {
            label: label.into(),
            m,
            k: 2560,
            dtype,
        };
        let base = vec![
            weight("layer0.q", 12288, DType::MQ6G256V2),
            weight("lm_head", 248_320, DType::MQ6G256V2),
        ];
        let same = (base.clone(), 2048);
        assert!(agree_layouts(&[same.clone(), same.clone()]).is_ok());
        // Another stored dtype on rank 1 (e.g. a different trunk tier).
        let mut dtype = base.clone();
        dtype[0].dtype = DType::Q8_0;
        let error = agree_layouts(&[same.clone(), (dtype, 2048)]).unwrap_err();
        assert!(
            error.contains("rank 1") && error.contains("layer0.q"),
            "{error}"
        );
        // A missing weight, and another chunk size, on rank 2.
        assert!(agree_layouts(&[same.clone(), same.clone(), (base[..1].to_vec(), 2048)]).is_err());
        assert!(agree_layouts(&[same.clone(), (base, 4096)]).is_err());
    }
}
