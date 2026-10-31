// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Layer bands: a contiguous layer range of Raon-1-mini that loads, holds state
//! for and runs on its own device.
//!
//! A band is the unit of pipeline placement. The only tensor crossing a band
//! boundary is the mHC stream `[rows, streams, hidden]` F32 in
//! `scratch.stream_a` (rows contiguous from row 0): every layer has two mHC
//! sites (ping-pong), so `stream_a` holds the result at the end of every layer.
//!
//! * the embedding lives on the band containing layer 0 ([`RaonBand::is_first`]);
//! * `model.norm` and `lm_head` live on the band containing the last layer
//!   ([`RaonBand::is_last`]);
//! * RDA / MLA state slots and the absorbed MLA pack exist only for the band's
//!   layers, indexed band-locally.
//!
//! # Band-local view
//!
//! Everything that is indexed by layer (weights, RDA/MLA state slots, the MLA
//! work packing, the forward's layer loop) is band-local: layer `l` of the model
//! is element `l - band.layers.start` of [`crate::weights::RaonWeights::layers`],
//! and a bundle built for a band carries a *band view* of the configuration
//! ([`RaonConfig::band_view`]: `n_layers` is the band's layer count and
//! `layer_kinds` / `ffn_kinds` are the band's slices), so `rda_index` /
//! `mla_index` / `n_rda` / `n_mla` / `state_dims` yield the band-local slot
//! mapping with no further code. Only tensor names (loading) use global layer
//! numbers. The whole band's view is the configuration itself, so the
//! single-device code path is the same code with the same allocations.

use std::ops::Range;

use crate::config::RaonConfig;
use crate::state::RaonPrefillScratch;
use rdna_compute::{Gpu, GpuTensor};

/// A contiguous range of layers of an `total`-layer Raon model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaonBand {
    /// Global layer indices the band owns (non-empty, within `0..total`).
    pub layers: Range<usize>,
    /// Layer count of the whole model.
    pub total: usize,
}

impl RaonBand {
    /// Band `layers` of a `total`-layer model.
    pub fn new(layers: Range<usize>, total: usize) -> Result<Self, String> {
        if layers.start >= layers.end || layers.end > total {
            return Err(format!(
                "raon1_mini band: layer range {}..{} is empty or outside the {total} layers",
                layers.start, layers.end
            ));
        }
        Ok(Self { layers, total })
    }

    /// The band owning every layer of `cfg`.
    pub fn whole(cfg: &RaonConfig) -> Self {
        Self {
            layers: 0..cfg.n_layers,
            total: cfg.n_layers,
        }
    }

    /// Consecutive bands cut before each layer index of `cuts` (strictly
    /// ascending, each in `1..total`): `cuts = [1, 12]` of 25 layers gives
    /// `0..1`, `1..12`, `12..25`.
    pub fn from_cuts(total: usize, cuts: &[usize]) -> Result<Vec<Self>, String> {
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let mut start = 0;
        for &c in cuts.iter().chain(std::iter::once(&total)) {
            out.push(Self::new(start..c, total).map_err(|e| format!("{e} (cuts {cuts:?})"))?);
            start = c;
        }
        Ok(out)
    }

    /// Layers in the band.
    pub fn len(&self) -> usize {
        self.layers.end - self.layers.start
    }

    /// Always false: a band owns at least one layer ([`Self::new`]).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The band is the whole model.
    pub fn is_whole(&self) -> bool {
        self.layers.start == 0 && self.layers.end == self.total
    }

    /// The band holds layer 0 (and so the embedding).
    pub fn is_first(&self) -> bool {
        self.layers.start == 0
    }

    /// The band holds the last layer (and so `model.norm` and `lm_head`).
    pub fn is_last(&self) -> bool {
        self.layers.end == self.total
    }

    /// Whether global layer `layer` belongs to the band.
    pub fn contains(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// Band-local index of global layer `layer`.
    pub fn local(&self, layer: usize) -> Option<usize> {
        self.contains(layer).then(|| layer - self.layers.start)
    }

    /// Global index of band-local layer `local`.
    pub fn global(&self, local: usize) -> usize {
        self.layers.start + local
    }

    /// Whether the checkpoint tensor `name` is the band's: its layer for
    /// `model.layers.N.*`, the embedding on the first band, `model.norm` and
    /// `lm_head` on the last. Any other name is claimed by every band so the
    /// inventory's unconsumed-tensor check still reports it.
    pub fn owns_tensor(&self, name: &str) -> bool {
        if let Some(rest) = name.strip_prefix("model.layers.") {
            return match rest.split('.').next().and_then(|n| n.parse::<usize>().ok()) {
                Some(l) => self.contains(l),
                None => true,
            };
        }
        if name.starts_with("model.embed_tokens.") {
            return self.is_first();
        }
        if name.starts_with("model.norm.") || name.starts_with("lm_head.") {
            return self.is_last();
        }
        true
    }

    /// Short `a..b of N` label for logs.
    pub fn label(&self) -> String {
        format!(
            "layers {}..{} of {}",
            self.layers.start, self.layers.end, self.total
        )
    }
}

impl RaonConfig {
    /// The band-local view of this configuration (see the module docs): the
    /// band's layer count and layer schedule slices, every other field as is.
    /// The whole band's view equals `self`.
    pub fn band_view(&self, band: &RaonBand) -> Result<RaonConfig, String> {
        if band.total != self.n_layers {
            return Err(format!(
                "raon1_mini band: band of a {}-layer model applied to a {}-layer config",
                band.total, self.n_layers
            ));
        }
        let mut v = self.clone();
        v.n_layers = band.len();
        v.layer_kinds = self.layer_kinds[band.layers.clone()].to_vec();
        v.ffn_kinds = self.ffn_kinds[band.layers.clone()].to_vec();
        Ok(v)
    }
}

/// The edge view: the `[rows, streams, hidden]` F32 stream rows (from row 0) of
/// `scratch.stream_a`, the tensor a band hands to its successor at band end
/// (export) and receives from its predecessor at band start (import). A view,
/// not an owner: never free it.
pub fn edge_view(cfg: &RaonConfig, scratch: &RaonPrefillScratch, rows: usize) -> GpuTensor {
    scratch
        .stream_a
        .sub_offset(0, rows * cfg.mhc_streams * cfg.hidden)
}

/// Bytes of an edge of `rows` rows.
pub fn edge_bytes(cfg: &RaonConfig, rows: usize) -> usize {
    rows * cfg.mhc_streams * cfg.hidden * 4
}

/// Device-to-device copy of an edge between two bands on the SAME device
/// (`dst` and `src` from [`edge_view`]), ordered on the device's active stream
/// (blocking when none). A cross-device hop is the mesh executor's.
pub fn copy_edge_same_device(
    gpu: &Gpu,
    dst: &GpuTensor,
    src: &GpuTensor,
    bytes: usize,
) -> Result<(), String> {
    if bytes > dst.byte_size() || bytes > src.byte_size() {
        return Err(format!(
            "raon1_mini band: edge copy of {bytes} B exceeds the views ({} / {} B)",
            dst.byte_size(),
            src.byte_size()
        ));
    }
    gpu.memcpy_dtod_auto(&dst.buf, &src.buf, bytes)
        .map_err(|e| format!("raon1_mini band: edge copy: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerKind;

    fn released() -> RaonConfig {
        RaonConfig::from_json(&serde_json::json!({"model_type": "raon1_mini"})).unwrap()
    }

    #[test]
    fn band_constructor_rejects_bad_ranges() {
        assert!(RaonBand::new(0..25, 25).is_ok());
        assert!(RaonBand::new(3..3, 25).is_err());
        assert!(RaonBand::new(5..26, 25).is_err());
        #[allow(clippy::reversed_empty_ranges)]
        let backwards = RaonBand::new(7..2, 25);
        assert!(backwards.is_err());
    }

    #[test]
    fn whole_band_is_first_last_and_whole() {
        let cfg = released();
        let b = RaonBand::whole(&cfg);
        assert!(b.is_whole() && b.is_first() && b.is_last());
        assert_eq!(b.len(), 25);
        assert_eq!(b.local(24), Some(24));
        assert_eq!(b.local(25), None);
    }

    #[test]
    fn cuts_partition_the_layers() {
        let bands = RaonBand::from_cuts(25, &[1, 12]).unwrap();
        assert_eq!(
            bands.iter().map(|b| b.layers.clone()).collect::<Vec<_>>(),
            vec![0..1, 1..12, 12..25]
        );
        assert!(bands[0].is_first() && !bands[0].is_last());
        assert!(!bands[1].is_first() && !bands[1].is_last());
        assert!(!bands[2].is_first() && bands[2].is_last());
        assert!(bands.iter().all(|b| !b.is_whole()));
        assert_eq!(bands[1].local(12), None);
        assert_eq!(bands[1].local(11), Some(10));
        assert_eq!(bands[1].global(10), 11);
        // Whole model with no cuts; bad cuts refused.
        assert_eq!(RaonBand::from_cuts(25, &[]).unwrap().len(), 1);
        assert!(RaonBand::from_cuts(25, &[0]).is_err());
        assert!(RaonBand::from_cuts(25, &[12, 12]).is_err());
        assert!(RaonBand::from_cuts(25, &[12, 8]).is_err());
        assert!(RaonBand::from_cuts(25, &[25]).is_err());
    }

    #[test]
    fn whole_band_view_is_the_config() {
        let cfg = released();
        let v = cfg.band_view(&RaonBand::whole(&cfg)).unwrap();
        assert_eq!(v.n_layers, cfg.n_layers);
        assert_eq!(v.layer_kinds, cfg.layer_kinds);
        assert_eq!(v.ffn_kinds, cfg.ffn_kinds);
        assert_eq!(v.state_dims(), cfg.state_dims());
    }

    #[test]
    fn band_view_maps_global_layers_to_band_local_slots() {
        let cfg = released();
        // Released schedule (zero-based): MLA at 3,7,11,15,19,23,24; RDA elsewhere.
        let band = RaonBand::new(2..12, 25).unwrap();
        let v = cfg.band_view(&band).unwrap();
        assert_eq!(v.n_layers, 10);
        // RDA layers in 2..12: 2,4,5,6,8,9,10; MLA: 3,7,11.
        assert_eq!(v.n_rda(), 7);
        assert_eq!(v.n_mla(), 3);
        for l in band.layers.clone() {
            let loc = band.local(l).unwrap();
            assert_eq!(v.layer_kinds[loc], cfg.layer_kinds[l]);
            assert_eq!(v.ffn_kinds[loc], cfg.ffn_kinds[l]);
            // Slot = the global slot minus the band's first slot of that kind.
            let first_rda = cfg
                .rda_layers()
                .iter()
                .filter(|&&x| x < band.layers.start)
                .count();
            let first_mla = cfg
                .mla_layers()
                .iter()
                .filter(|&&x| x < band.layers.start)
                .count();
            match cfg.layer_kinds[l] {
                LayerKind::Rda => {
                    assert_eq!(
                        v.rda_index(loc),
                        Some(cfg.rda_index(l).unwrap() - first_rda)
                    );
                    assert_eq!(v.mla_index(loc), None);
                }
                LayerKind::Mla => {
                    assert_eq!(
                        v.mla_index(loc),
                        Some(cfg.mla_index(l).unwrap() - first_mla)
                    );
                    assert_eq!(v.rda_index(loc), None);
                }
            }
        }
        // Slots of the bands tile the model's slots exactly.
        let bands = RaonBand::from_cuts(25, &[1, 4, 12, 20]).unwrap();
        let (mut rda, mut mla) = (0, 0);
        for b in &bands {
            let v = cfg.band_view(b).unwrap();
            rda += v.n_rda();
            mla += v.n_mla();
            let d = v.state_dims();
            assert_eq!(
                (d.n_layers, d.n_rda, d.n_mla),
                (b.len(), v.n_rda(), v.n_mla())
            );
            d.validate().unwrap();
        }
        assert_eq!((rda, mla), (cfg.n_rda(), cfg.n_mla()));
        // A band without MLA / without RDA layers is a valid state shape.
        let rda_only = cfg.band_view(&RaonBand::new(0..3, 25).unwrap()).unwrap();
        assert_eq!((rda_only.n_rda(), rda_only.n_mla()), (3, 0));
        rda_only.state_dims().validate().unwrap();
        let mla_only = cfg.band_view(&RaonBand::new(3..4, 25).unwrap()).unwrap();
        assert_eq!((mla_only.n_rda(), mla_only.n_mla()), (0, 1));
        mla_only.state_dims().validate().unwrap();
    }

    #[test]
    fn band_view_refuses_a_band_of_another_model() {
        let cfg = released();
        let other = RaonBand::new(0..4, 30).unwrap();
        assert!(cfg.band_view(&other).is_err());
    }

    #[test]
    fn tensor_ownership_follows_the_layer_and_the_edges() {
        let first = RaonBand::new(0..1, 25).unwrap();
        let mid = RaonBand::new(1..12, 25).unwrap();
        let last = RaonBand::new(12..25, 25).unwrap();
        assert!(first.owns_tensor("model.embed_tokens.weight"));
        assert!(!mid.owns_tensor("model.embed_tokens.weight"));
        assert!(!last.owns_tensor("model.embed_tokens.weight"));
        assert!(last.owns_tensor("lm_head.weight") && last.owns_tensor("model.norm.weight"));
        assert!(!first.owns_tensor("lm_head.weight") && !mid.owns_tensor("model.norm.weight"));
        assert!(first.owns_tensor("model.layers.0.self_attn.q_proj.weight"));
        assert!(!first.owns_tensor("model.layers.1.self_attn.q_proj.weight"));
        assert!(mid.owns_tensor("model.layers.11.moe.experts.127.down_proj.weight"));
        assert!(!mid.owns_tensor("model.layers.12.moe.gate.weight"));
        assert!(last.owns_tensor("model.layers.24.mlp_raon_mhc.res.eta"));
        // Unknown names are everyone's (so the unconsumed check still sees them).
        assert!(mid.owns_tensor("mtp.layers.0.weight"));
        assert!(mid.owns_tensor("model.layers.x.weight"));
    }
}
