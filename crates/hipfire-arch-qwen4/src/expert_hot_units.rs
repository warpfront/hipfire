// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `<model>.ep.hot-units` sidecar of the Flash-Next EP2 hot-unit loan.
//!
//! A calibration of the host-read demand of the static placement: `counts[id]` (global id
//! `layer * experts_per_layer + expert`) is the number of calibration trunk forwards (complete
//! forwards over every trunk layer) on the OWNER rank (`expert % ranks`) in which that unit was
//! read from pinned host (skew-log class `T` or `H`, `HIPFIRE_EP_SKEW_LOG`). The loader selects
//! the hottest owned units from it ([`crate::expert_ep_units::ep_unit_apply_hot_loan`]) and keeps
//! them VRAM-resident on top of the ordinary placement, up to a loan of bytes per rank.
//!
//! The file is closed JSON (unknown keys are refused). It is bound to the base artifact exactly
//! like the EP hotset sidecar (`base_size` + `base_index_sha256`, [`HotsetBase`]), to the model
//! (`model_sha256`), to the geometry of the plan it loans into, and to the EP hotset sidecar the
//! calibration ran under (`base_ranking_sha256`, the sha256 of that sidecar's bytes, which is the
//! plan's `ranking_sha256`): a calibration measured under another ranking describes another
//! placement. Written by `scripts/ep_hot_units.py`.

use std::path::Path;

use serde::Deserialize;

use crate::expert_ep_units::EpUnitGeometry;
use crate::expert_hotset::HotsetBase;

/// `kind` of a hot-units sidecar.
pub const EP_HOT_UNITS_KIND: &str = "hipfire.qwen4.ep_hot_units";
/// The only `version` this build reads.
pub const EP_HOT_UNITS_VERSION: u64 = 1;

/// Wire form; unknown keys are refused so v1 stays closed.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHotUnits {
    kind: String,
    version: u64,
    model_sha256: String,
    base_size: u64,
    base_index_sha256: String,
    base_ranking_sha256: String,
    unit_bytes: u64,
    layers: u32,
    experts_per_layer: u32,
    ranks: u32,
    corpus: String,
    capture_forwards: Vec<u64>,
    counts: Vec<u64>,
}

/// What a sidecar must match to apply to a load: the plan's geometry, the base artifact and the
/// plan's walked ranking.
#[derive(Clone, Copy, Debug)]
pub struct HotUnitsBinding<'a> {
    pub geometry: &'a EpUnitGeometry,
    pub base: &'a HotsetBase,
    /// `EpUnitPlan::ranking_sha256` of the plan: sha256 of the EP hotset sidecar bytes.
    pub ranking_sha256: &'a str,
}

/// A validated hot-units sidecar.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HotUnitsSidecar {
    model_sha256: String,
    base_ranking_sha256: String,
    corpus: String,
    capture_forwards: Vec<u64>,
    counts: Vec<u64>,
}

impl HotUnitsSidecar {
    /// Read and validate the file at `path` against `binding`.
    pub fn load(path: impl AsRef<Path>, binding: &HotUnitsBinding<'_>) -> Result<Self, String> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::from_bytes(&bytes, binding)
    }

    /// Parse and validate `bytes` against `binding`: kind and version, the digests, the counts
    /// length, the geometry, the base binding and the ranking binding, in that order.
    pub fn from_bytes(bytes: &[u8], binding: &HotUnitsBinding<'_>) -> Result<Self, String> {
        let raw: RawHotUnits = serde_json::from_slice(bytes)
            .map_err(|e| format!("not a v{EP_HOT_UNITS_VERSION} hot-units JSON: {e}"))?;
        if raw.kind != EP_HOT_UNITS_KIND || raw.version != EP_HOT_UNITS_VERSION {
            return Err(format!(
                "kind {:?} version {}, expected {EP_HOT_UNITS_KIND:?} version {EP_HOT_UNITS_VERSION}",
                raw.kind, raw.version
            ));
        }
        for (name, value) in [
            ("model_sha256", &raw.model_sha256),
            ("base_index_sha256", &raw.base_index_sha256),
            ("base_ranking_sha256", &raw.base_ranking_sha256),
        ] {
            check_sha256(name, value)?;
        }
        let total = (raw.layers as usize)
            .checked_mul(raw.experts_per_layer as usize)
            .ok_or_else(|| "layers * experts_per_layer overflows".to_string())?;
        if raw.counts.len() != total {
            return Err(format!(
                "counts has {} entries, layers {} x experts_per_layer {} = {total}",
                raw.counts.len(),
                raw.layers,
                raw.experts_per_layer
            ));
        }
        if raw.capture_forwards.len() != raw.ranks as usize {
            return Err(format!(
                "capture_forwards has {} entries for {} ranks",
                raw.capture_forwards.len(),
                raw.ranks
            ));
        }
        let g = binding.geometry;
        if raw.unit_bytes != g.unit_bytes()
            || raw.layers as usize != g.layers
            || raw.experts_per_layer as usize != g.experts_per_layer
            || raw.ranks as usize != g.ranks
        {
            return Err(format!(
                "geometry unit_bytes {} layers {} experts {} ranks {} differs from the plan's \
                 unit_bytes {} layers {} experts {} ranks {}",
                raw.unit_bytes,
                raw.layers,
                raw.experts_per_layer,
                raw.ranks,
                g.unit_bytes(),
                g.layers,
                g.experts_per_layer,
                g.ranks
            ));
        }
        if raw.base_size != binding.base.size() {
            return Err(format!(
                "base is {} bytes, sidecar binds {}",
                binding.base.size(),
                raw.base_size
            ));
        }
        if raw.base_index_sha256 != binding.base.index_sha256() {
            return Err(format!(
                "base index sha256 {} differs from the sidecar's {}",
                short(binding.base.index_sha256()),
                short(&raw.base_index_sha256)
            ));
        }
        if raw.base_ranking_sha256 != binding.ranking_sha256 {
            return Err(format!(
                "calibrated under ranking {}, the plan walks ranking {}",
                short(&raw.base_ranking_sha256),
                short(binding.ranking_sha256)
            ));
        }
        Ok(Self {
            model_sha256: raw.model_sha256,
            base_ranking_sha256: raw.base_ranking_sha256,
            corpus: raw.corpus,
            capture_forwards: raw.capture_forwards,
            counts: raw.counts,
        })
    }

    /// Host-read forward counts per global unit id (`layer * experts_per_layer + expert`).
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Calibration forwards counted per rank.
    pub fn capture_forwards(&self) -> &[u64] {
        &self.capture_forwards
    }

    /// Provenance string of the calibration corpus.
    pub fn corpus(&self) -> &str {
        &self.corpus
    }

    pub fn model_sha256(&self) -> &str {
        &self.model_sha256
    }

    /// sha256 of the EP hotset sidecar the calibration ran under.
    pub fn base_ranking_sha256(&self) -> &str {
        &self.base_ranking_sha256
    }
}

/// The first 12 characters of a digest.
fn short(hex: &str) -> &str {
    hex.get(..12).unwrap_or(hex)
}

fn check_sha256(name: &str, value: &str) -> Result<(), String> {
    let full = value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
    if full {
        Ok(())
    } else {
        Err(format!(
            "{name} {value:?} is not a full lowercase 64-hex SHA-256"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    const SHA_A: &str = "8b15b6fede7d7c5bfed0db4720a8295bedda51bc93e545fa242bd50d0f200972";
    const SHA_RANKING: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const BASE_SIZE: u64 = 125_144_775_192;
    const LAYERS: usize = 2;
    const EXPERTS: usize = 4;

    fn geometry() -> EpUnitGeometry {
        EpUnitGeometry {
            layers: LAYERS,
            experts_per_layer: EXPERTS,
            ranks: 2,
            gate_up_bytes: 1_689_600,
            down_bytes: 844_800,
        }
    }

    fn base() -> HotsetBase {
        HotsetBase::new(BASE_SIZE, [0xab; 32])
    }

    fn base_index_hex() -> String {
        "ab".repeat(32)
    }

    fn doc() -> Value {
        json!({
            "kind": EP_HOT_UNITS_KIND,
            "version": 1,
            "model_sha256": SHA_A,
            "base_size": BASE_SIZE,
            "base_index_sha256": base_index_hex(),
            "base_ranking_sha256": SHA_RANKING,
            "unit_bytes": 2_534_400u64,
            "layers": LAYERS,
            "experts_per_layer": EXPERTS,
            "ranks": 2,
            "corpus": "held-out mixed prompts",
            "capture_forwards": [10, 12],
            "counts": [0, 3, 5, 0, 7, 0, 0, 2],
        })
    }

    fn parse(doc: &Value) -> Result<HotUnitsSidecar, String> {
        let g = geometry();
        let b = base();
        HotUnitsSidecar::from_bytes(
            doc.to_string().as_bytes(),
            &HotUnitsBinding {
                geometry: &g,
                base: &b,
                ranking_sha256: SHA_RANKING,
            },
        )
    }

    fn with(key: &str, value: Value) -> Value {
        let mut doc = doc();
        doc[key] = value;
        doc
    }

    #[test]
    fn a_bound_sidecar_parses_and_exposes_its_counts() {
        let sidecar = parse(&doc()).unwrap();
        assert_eq!(sidecar.counts(), &[0, 3, 5, 0, 7, 0, 0, 2]);
        assert_eq!(sidecar.capture_forwards(), &[10, 12]);
        assert_eq!(sidecar.corpus(), "held-out mixed prompts");
        assert_eq!(sidecar.model_sha256(), SHA_A);
        assert_eq!(sidecar.base_ranking_sha256(), SHA_RANKING);
    }

    #[test]
    fn kind_and_version_are_checked() {
        let error = parse(&with("kind", json!("hipfire.qwen4.expert_hotset"))).unwrap_err();
        assert!(error.contains("expected"), "{error}");
        let error = parse(&with("version", json!(2))).unwrap_err();
        assert!(error.contains("version 2"), "{error}");
    }

    #[test]
    fn the_schema_is_closed() {
        let error = parse(&with("extra", json!(1))).unwrap_err();
        assert!(error.contains("extra"), "{error}");
        let mut missing = doc();
        missing.as_object_mut().unwrap().remove("corpus");
        let error = parse(&missing).unwrap_err();
        assert!(error.contains("corpus"), "{error}");
        assert!(parse(&json!({})).is_err());
    }

    #[test]
    fn digests_must_be_full_lowercase_hex() {
        let upper = SHA_A.to_uppercase();
        for key in ["model_sha256", "base_index_sha256", "base_ranking_sha256"] {
            for bad in ["abc", upper.as_str(), &SHA_A[..63]] {
                let error = parse(&with(key, json!(bad))).unwrap_err();
                assert!(error.contains(key), "{key} {bad}: {error}");
            }
        }
    }

    #[test]
    fn counts_and_capture_forwards_lengths_are_checked() {
        let error = parse(&with("counts", json!([1, 2, 3]))).unwrap_err();
        assert!(error.contains("counts has 3"), "{error}");
        let error = parse(&with("counts", json!([]))).unwrap_err();
        assert!(error.contains("counts has 0"), "{error}");
        let error = parse(&with("capture_forwards", json!([10]))).unwrap_err();
        assert!(error.contains("capture_forwards"), "{error}");
    }

    #[test]
    fn geometry_must_equal_the_plans() {
        let error = parse(&with("unit_bytes", json!(2_611_200u64))).unwrap_err();
        assert!(error.contains("geometry"), "{error}");
        let error = parse(&with("ranks", json!(3))).unwrap_err();
        assert!(
            error.contains("capture_forwards") || error.contains("geometry"),
            "{error}"
        );
        // A self-consistent doc of another shape is still the wrong geometry.
        let mut other = doc();
        other["experts_per_layer"] = json!(2);
        other["counts"] = json!([0, 1, 2, 3]);
        let error = parse(&other).unwrap_err();
        assert!(error.contains("geometry"), "{error}");
    }

    #[test]
    fn the_base_binding_is_checked() {
        let error = parse(&with("base_size", json!(BASE_SIZE + 1))).unwrap_err();
        assert!(error.contains("bytes"), "{error}");
        let other = "cd".repeat(32);
        let error = parse(&with("base_index_sha256", json!(other))).unwrap_err();
        assert!(error.contains("base index sha256"), "{error}");
    }

    #[test]
    fn the_calibration_ranking_must_be_the_plans() {
        let other = "22".repeat(32);
        let error = parse(&with("base_ranking_sha256", json!(other))).unwrap_err();
        assert!(error.contains("ranking"), "{error}");
        let error = HotUnitsSidecar::from_bytes(
            doc().to_string().as_bytes(),
            &HotUnitsBinding {
                geometry: &geometry(),
                base: &base(),
                ranking_sha256: &other,
            },
        )
        .unwrap_err();
        assert!(error.contains("calibrated under ranking"), "{error}");
    }

    #[test]
    fn garbage_and_missing_files_are_named_errors() {
        let g = geometry();
        let b = base();
        let binding = HotUnitsBinding {
            geometry: &g,
            base: &b,
            ranking_sha256: SHA_RANKING,
        };
        assert!(HotUnitsSidecar::from_bytes(b"not json", &binding).is_err());
        let error =
            HotUnitsSidecar::load("/nonexistent/hipfire-hot-units/m.ep.hot-units", &binding)
                .unwrap_err();
        assert!(error.contains("m.ep.hot-units"), "{error}");
    }
}
