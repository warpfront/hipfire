// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Redline route SKU identity: which model files are the MQ4R, MQ4RS and DS4
//! MQ2R SKUs, decided from the file's contents and its registry card — never
//! from its file name.
//!
//! Before 0.5.0 `retained_redline_default` and the DeepSeek V4 loader chose
//! these SKUs by the file extension (`.mq4r`, `.mq4rs`, `.mq2r`). A name is not
//! a property of the bytes (a hard link, a rename or a copy changed the
//! route), so the extension is no longer read anywhere. The identity is, in
//! order:
//!
//! 1. the HFQ metadata key [`ROUTE_SKU_KEY`] (`mq4r` | `mq4rs` | `mq2r`), which
//!    `hipfire-quantize` stamps on new MQ4R/MQ4RS writes, or for DS4 the
//!    `hipfire_quant_recipe` identity `deepseek4-mq2r-e8-p3-v1`;
//! 2. the registry card's `quant` (`mq4r` | `mq4rs` | `mq2r`) of the installed
//!    registry artifact, which the CLI forwards as the load parameter
//!    `route_sku`. The published MQ4R/MQ4RS files carry no metadata key, so
//!    they route through their registry identity;
//! 3. DeepSeek V4 only: the structural MQ2R P3 tensor map in the tensor index
//!    (554 dense MFP4G32E8SOA tensors, Q8F16 embedding);
//! 4. otherwise no SKU, and no automatic Redline route.

use crate::hfq::HfqFile;

/// Top-level HFQ metadata key naming the route SKU of the artifact.
pub const ROUTE_SKU_KEY: &str = "hipfire_route_sku";
/// Existing DeepSeek V4 recipe identity key; the MQ2R P3 value is an MQ2R
/// header identity.
pub const QUANT_RECIPE_KEY: &str = "hipfire_quant_recipe";
/// `hipfire_quant_recipe` value of the DeepSeek V4 MQ2R P3 artifact.
pub const DEEPSEEK4_MQ2R_RECIPE: &str = "deepseek4-mq2r-e8-p3-v1";

const DEEPSEEK4_ARCH_ID: u32 = 9;
const QT_Q8F16: u8 = 3;
const QT_MFP4G32E8SOA: u8 = 35;
/// Dense-tier tensor count of the MQ2R P3 recipe (head plus the per-layer
/// attention, shared-expert, compressor, indexer and gate projections).
const MQ2R_DENSE_TENSORS: usize = 554;

/// A route SKU whose retained-Redline policy is decided by identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteSku {
    /// Uniform-MQ4 speed SKU (Qwen3.5-family and other MoE-class models).
    Mq4r,
    /// Symmetric-expert A3B SKU (`qwen3_5_moe`).
    Mq4rs,
    /// DeepSeek V4 MQ2R P3 artifact.
    Mq2r,
}

impl RouteSku {
    pub const ALL: [RouteSku; 3] = [RouteSku::Mq4r, RouteSku::Mq4rs, RouteSku::Mq2r];

    pub fn as_str(self) -> &'static str {
        match self {
            RouteSku::Mq4r => "mq4r",
            RouteSku::Mq4rs => "mq4rs",
            RouteSku::Mq2r => "mq2r",
        }
    }

    /// Exact lowercase spelling, shared by the HFQ metadata key, the registry
    /// card `quant` and the `route_sku` load parameter.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|sku| sku.as_str() == value)
    }
}

/// Where a resolved SKU came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteSource {
    /// [`ROUTE_SKU_KEY`] (or the DS4 recipe identity) in the HFQ metadata.
    Header,
    /// The registry card of the installed artifact.
    Registry,
    /// The DeepSeek V4 MQ2R tensor-index structure.
    Structure,
}

impl RouteSource {
    pub fn as_str(self) -> &'static str {
        match self {
            RouteSource::Header => "hfq-metadata",
            RouteSource::Registry => "registry",
            RouteSource::Structure => "tensor-structure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Route {
    pub sku: RouteSku,
    pub source: RouteSource,
}

/// The SKU a registry card names: its `quant` when that is exactly a SKU
/// spelling (`mq4r` | `mq4rs` | `mq2r`), else none. The CLI forwards this as
/// the load parameter `route_sku`.
pub fn registry_route(registry_quant: Option<&str>) -> Option<RouteSku> {
    registry_quant.and_then(RouteSku::parse)
}

/// The SKU the HFQ metadata declares: [`ROUTE_SKU_KEY`], else the DeepSeek V4
/// MQ2R recipe identity. A present key with an unknown value is an error, not
/// "no SKU".
pub fn header_route(metadata_json: &str) -> Result<Option<RouteSku>, String> {
    let metadata: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|error| format!("HFQ metadata is not valid JSON: {error}"))?;
    if let Some(value) = metadata.get(ROUTE_SKU_KEY) {
        return value
            .as_str()
            .and_then(RouteSku::parse)
            .map(Some)
            .ok_or_else(|| {
                format!(
                    "HFQ metadata `{ROUTE_SKU_KEY}` = {value}: expected one of \
                     \"mq4r\", \"mq4rs\", \"mq2r\""
                )
            });
    }
    if metadata
        .get(QUANT_RECIPE_KEY)
        .and_then(serde_json::Value::as_str)
        == Some(DEEPSEEK4_MQ2R_RECIPE)
    {
        return Ok(Some(RouteSku::Mq2r));
    }
    Ok(None)
}

/// The DeepSeek V4 MQ2R P3 identity in the tensor index. `None` for any other
/// architecture or a file without MFP4G32E8SOA tensors; a file that carries
/// E8 tensors but not the exact recipe is an error naming the requirement
/// (it is neither MQ2R nor a plain DS4 route).
pub fn structural_route(hfq: &HfqFile) -> Result<Option<RouteSku>, String> {
    if hfq.arch_id != DEEPSEEK4_ARCH_ID {
        return Ok(None);
    }
    let e8 = hfq
        .tensors()
        .iter()
        .filter(|tensor| tensor.quant_type == QT_MFP4G32E8SOA)
        .count();
    if e8 == 0 {
        return Ok(None);
    }
    let qt_of = |name: &str| hfq.find_tensor_info(name).map(|info| info.quant_type);
    if e8 == MQ2R_DENSE_TENSORS
        && qt_of("head.weight") == Some(QT_MFP4G32E8SOA)
        && qt_of("embed.weight") == Some(QT_Q8F16)
    {
        return Ok(Some(RouteSku::Mq2r));
    }
    Err(format!(
        "deepseek4: the file carries {e8} MFP4G32E8SOA (qt35) tensors but the MQ2R P3 recipe \
         needs exactly {MQ2R_DENSE_TENSORS} with `head.weight` qt35 and `embed.weight` Q8F16; \
         the route is not selected by file name. Declare `{ROUTE_SKU_KEY}` / \
         `{QUANT_RECIPE_KEY}` = {DEEPSEEK4_MQ2R_RECIPE} in the HFQ metadata or re-quantize"
    ))
}

/// The route of an opened HFQ file: HFQ metadata, then the registry card's
/// `quant`, then (DeepSeek V4 only) the tensor structure, else no SKU.
pub fn resolve(hfq: &HfqFile, registry_quant: Option<&str>) -> Result<Option<Route>, String> {
    if let Some(sku) = header_route(&hfq.metadata_json)? {
        return Ok(Some(Route {
            sku,
            source: RouteSource::Header,
        }));
    }
    if let Some(sku) = registry_route(registry_quant) {
        return Ok(Some(Route {
            sku,
            source: RouteSource::Registry,
        }));
    }
    Ok(structural_route(hfq)?.map(|sku| Route {
        sku,
        source: RouteSource::Structure,
    }))
}

/// [`resolve`] over the file at `path`. A path that is not an HFQ container
/// (an HF snapshot directory, GGUF) has no HFQ identity, so only the registry
/// card can name its SKU.
pub fn resolve_path(
    path: &std::path::Path,
    registry_quant: Option<&str>,
) -> Result<Option<Route>, String> {
    match HfqFile::open_at_offset(path, 0) {
        Ok(hfq) => resolve(&hfq, registry_quant),
        Err(_) => Ok(registry_route(registry_quant).map(|sku| Route {
            sku,
            source: RouteSource::Registry,
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_key_selects_every_sku_and_rejects_unknown_values() {
        for sku in RouteSku::ALL {
            let json = format!(
                r#"{{"architecture":"x","{ROUTE_SKU_KEY}":"{}"}}"#,
                sku.as_str()
            );
            assert_eq!(header_route(&json), Ok(Some(sku)));
        }
        let error = header_route(&format!(r#"{{"{ROUTE_SKU_KEY}":"MQ4R"}}"#)).unwrap_err();
        assert!(error.contains(ROUTE_SKU_KEY), "{error}");
        let error = header_route(&format!(r#"{{"{ROUTE_SKU_KEY}":7}}"#)).unwrap_err();
        assert!(error.contains("expected one of"), "{error}");
    }

    #[test]
    fn deepseek4_recipe_identity_is_an_mq2r_header_route() {
        let json = format!(r#"{{"{QUANT_RECIPE_KEY}":"{DEEPSEEK4_MQ2R_RECIPE}"}}"#);
        assert_eq!(header_route(&json), Ok(Some(RouteSku::Mq2r)));
        // MQ2RXT carries its own recipe; it is not the MQ2R route.
        let rxt = format!(r#"{{"{QUANT_RECIPE_KEY}":"deepseek4-mq2rxt-mq4-p3-v1"}}"#);
        assert_eq!(header_route(&rxt), Ok(None));
        assert_eq!(header_route("{}"), Ok(None));
        assert!(header_route("not json").is_err());
    }

    #[test]
    fn sku_spelling_is_exact_lowercase() {
        assert_eq!(RouteSku::parse("mq4r"), Some(RouteSku::Mq4r));
        assert_eq!(RouteSku::parse("mq4rs"), Some(RouteSku::Mq4rs));
        assert_eq!(RouteSku::parse("mq2r"), Some(RouteSku::Mq2r));
        for other in ["MQ4R", "mq4", "mq2rxt", "mq4r ", ""] {
            assert_eq!(RouteSku::parse(other), None, "{other:?}");
        }
    }

    /// A tensor-index-only HFQ container: header, metadata, index, no payload
    /// (every tensor declares zero data bytes).
    fn write_hfq(
        dir: &std::path::Path,
        name: &str,
        arch_id: u32,
        metadata: &str,
        tensors: &[(String, u8)],
    ) -> std::path::PathBuf {
        let mut index = Vec::new();
        index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (tensor, qt) in tensors {
            index.extend_from_slice(&(tensor.len() as u16).to_le_bytes());
            index.extend_from_slice(tensor.as_bytes());
            index.push(*qt);
            index.push(0); // no dims
            index.extend_from_slice(&0u32.to_le_bytes()); // group size
            index.extend_from_slice(&0u64.to_le_bytes()); // data size
        }
        let data_offset = 32 + metadata.len() + index.len();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HFQM");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&arch_id.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&32u64.to_le_bytes());
        bytes.extend_from_slice(&(data_offset as u64).to_le_bytes());
        bytes.extend_from_slice(metadata.as_bytes());
        bytes.extend_from_slice(&index);
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    fn ds4_tensors(e8: usize) -> Vec<(String, u8)> {
        let mut tensors = vec![
            ("embed.weight".to_string(), QT_Q8F16),
            ("head.weight".to_string(), QT_MFP4G32E8SOA),
        ];
        for i in 1..e8 {
            tensors.push((format!("layers.0.dense{i}.weight"), QT_MFP4G32E8SOA));
        }
        tensors
    }

    #[test]
    fn resolve_orders_header_then_registry_then_structure_and_never_the_name() {
        let dir = tempfile::tempdir().unwrap();
        let qwen = |name: &str, metadata: &str| {
            write_hfq(dir.path(), name, 6, metadata, &[("w".to_string(), 13)])
        };
        let open = |path: &std::path::Path| HfqFile::open_at_offset(path, 0).unwrap();

        // Header key first, even against a different registry card.
        let stamped = qwen(
            "a.hfq",
            r#"{"architecture":"qwen3_5_moe","hipfire_route_sku":"mq4r"}"#,
        );
        assert_eq!(
            resolve(&open(&stamped), Some("mq4rs")),
            Ok(Some(Route {
                sku: RouteSku::Mq4r,
                source: RouteSource::Header
            }))
        );
        // No key: the registry card names the SKU, for any file name.
        let bare = qwen("b.mq4", r#"{"architecture":"qwen3_5_moe"}"#);
        assert_eq!(
            resolve_path(&bare, Some("mq4r")),
            Ok(Some(Route {
                sku: RouteSku::Mq4r,
                source: RouteSource::Registry
            }))
        );
        // No key and no card: no SKU. A removed-extension name changes nothing.
        for name in ["c.mq4r", "c.mq4rs", "c.mq2r", "c.MQ4R"] {
            let named = qwen(name, r#"{"architecture":"qwen3_5_moe"}"#);
            assert_eq!(resolve_path(&named, None), Ok(None), "{name}");
            assert_eq!(resolve_path(&named, Some("mq4")), Ok(None), "{name}");
        }
        // A non-HFQ path has only the registry identity.
        assert_eq!(resolve_path(dir.path(), None), Ok(None));
        // A declared SKU the file cannot be is refused, naming the key.
        let bad = qwen("d.hfq", r#"{"hipfire_route_sku":"mq9"}"#);
        let error = resolve_path(&bad, Some("mq4r")).unwrap_err();
        assert!(error.contains(ROUTE_SKU_KEY), "{error}");
    }

    #[test]
    fn deepseek4_mq2r_structure_is_a_header_identity_and_partial_recipes_fail() {
        let dir = tempfile::tempdir().unwrap();
        let meta = r#"{"architecture":"deepseek4"}"#;
        let open = |path: &std::path::Path| HfqFile::open_at_offset(path, 0).unwrap();

        let exact = write_hfq(
            dir.path(),
            "x.bin",
            9,
            meta,
            &ds4_tensors(MQ2R_DENSE_TENSORS),
        );
        assert_eq!(
            resolve(&open(&exact), None),
            Ok(Some(Route {
                sku: RouteSku::Mq2r,
                source: RouteSource::Structure
            }))
        );
        // The registry card outranks the structure.
        assert_eq!(
            resolve(&open(&exact), Some("mq2r"))
                .unwrap()
                .unwrap()
                .source,
            RouteSource::Registry
        );
        // E8 tensors without the exact recipe: an error naming the requirement.
        let partial = write_hfq(dir.path(), "p.mq2r", 9, meta, &ds4_tensors(553));
        let error = resolve(&open(&partial), None).unwrap_err();
        assert!(
            error.contains("554") && error.contains(ROUTE_SKU_KEY),
            "{error}"
        );
        // No E8 tensors (mq2lloyd / mq2rxt) and other architectures: no SKU.
        let lloyd = write_hfq(
            dir.path(),
            "l.mq2lloyd",
            9,
            meta,
            &[("head.weight".to_string(), 3), ("e".to_string(), 19)],
        );
        assert_eq!(resolve(&open(&lloyd), None), Ok(None));
        let other_arch = write_hfq(dir.path(), "o.bin", 6, meta, &ds4_tensors(554));
        assert_eq!(resolve(&open(&other_arch), None), Ok(None));
    }
}
