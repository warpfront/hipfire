// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 expert-slab container schema, version 1 (DESIGN F2).
//!
//! Artifacts that carry per-expert trunk entries (instead of the two aggregate
//! trunk expert tensors per layer) declare it in the HFQM metadata under
//! [`EXPERT_SLABS_METADATA_KEY`]:
//!
//! ```json
//! {"version":1,"trunk_layers":48,"experts_per_layer":512,
//!  "payload_order":"layer,expert,gate_up,down","payload_alignment":256,
//!  "mtp_layout":"legacy_aggregate_unchanged"}
//! ```
//!
//! Version 1 fixes every value; [`parse`] rejects any deviation, unknown field,
//! missing field or wrong JSON type. Entry names, per-entry shapes and the
//! legal qtype pairs are frozen here too, so packer and loader share one
//! source of truth. Range checks on `layer` / `expert` indices are the
//! loader's job, not the name parser's.

use std::fmt;

use hipfire_runtime::mq_i4::{self, MqI4Profile};

/// HFQM metadata key holding the [`ExpertSlabsV1`] object.
pub const EXPERT_SLABS_METADATA_KEY: &str = "qwen4.expert_slabs";

/// v1 trunk layer count.
pub const TRUNK_LAYERS: u32 = 48;
/// v1 routed experts per trunk layer.
pub const EXPERTS_PER_LAYER: u32 = 512;
/// v1 payload order of the per-expert entries.
pub const PAYLOAD_ORDER: &str = "layer,expert,gate_up,down";
/// v1 alignment of the container's data offset (bytes).
pub const PAYLOAD_ALIGNMENT: u32 = 256;
/// v1 layout of the MTP experts: the legacy aggregate tensors, unchanged.
pub const MTP_LAYOUT: &str = "legacy_aggregate_unchanged";

/// Rank-2 shape `[rows, K]` of every expert's gate/up entry.
pub const GATE_UP_SHAPE: [u32; 2] = [1280, 2560];
/// Rank-2 shape `[rows, K]` of every expert's down entry.
pub const DOWN_SHAPE: [u32; 2] = [2560, 640];

/// Failure parsing or validating the `qwen4.expert_slabs` metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertSlabsError {
    /// The metadata string is not valid JSON.
    InvalidJson(String),
    /// The metadata JSON is not an object.
    MetadataNotObject,
    /// The `qwen4.expert_slabs` value is not a JSON object.
    NotAnObject,
    /// A required v1 field is absent.
    MissingField(&'static str),
    /// A field outside the v1 schema is present.
    UnknownField(String),
    /// A known field has the wrong JSON type.
    WrongType {
        field: &'static str,
        expected: &'static str,
    },
    /// A known field differs from its frozen v1 value.
    Mismatch {
        field: &'static str,
        expected: String,
        found: String,
    },
}

impl fmt::Display for ExpertSlabsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(e) => write!(f, "HFQM metadata is not valid JSON: {e}"),
            Self::MetadataNotObject => write!(f, "HFQM metadata JSON is not an object"),
            Self::NotAnObject => write!(f, "`{EXPERT_SLABS_METADATA_KEY}` is not a JSON object"),
            Self::MissingField(field) => {
                write!(
                    f,
                    "`{EXPERT_SLABS_METADATA_KEY}` is missing field `{field}`"
                )
            }
            Self::UnknownField(field) => {
                write!(
                    f,
                    "`{EXPERT_SLABS_METADATA_KEY}` has unknown field `{field}`"
                )
            }
            Self::WrongType { field, expected } => write!(
                f,
                "`{EXPERT_SLABS_METADATA_KEY}.{field}` must be {expected}"
            ),
            Self::Mismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "`{EXPERT_SLABS_METADATA_KEY}.{field}` is {found}, version 1 requires {expected}"
            ),
        }
    }
}

impl std::error::Error for ExpertSlabsError {}

/// The version-1 expert-slab declaration. Every field is frozen; see
/// [`ExpertSlabsV1::fn_v1`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertSlabsV1 {
    pub version: u32,
    pub trunk_layers: u32,
    pub experts_per_layer: u32,
    pub payload_order: String,
    pub payload_alignment: u32,
    pub mtp_layout: String,
}

impl ExpertSlabsV1 {
    /// The one legal version-1 value (Flash-Next geometry).
    pub fn fn_v1() -> Self {
        Self {
            version: 1,
            trunk_layers: TRUNK_LAYERS,
            experts_per_layer: EXPERTS_PER_LAYER,
            payload_order: PAYLOAD_ORDER.to_string(),
            payload_alignment: PAYLOAD_ALIGNMENT,
            mtp_layout: MTP_LAYOUT.to_string(),
        }
    }

    /// JSON object to store under [`EXPERT_SLABS_METADATA_KEY`].
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "version": self.version,
            "trunk_layers": self.trunk_layers,
            "experts_per_layer": self.experts_per_layer,
            "payload_order": self.payload_order,
            "payload_alignment": self.payload_alignment,
            "mtp_layout": self.mtp_layout,
        })
    }
}

const V1_FIELDS: [&str; 6] = [
    "version",
    "trunk_layers",
    "experts_per_layer",
    "payload_order",
    "payload_alignment",
    "mtp_layout",
];

fn expect_u64(
    obj: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    want: u64,
) -> Result<u32, ExpertSlabsError> {
    let value = obj
        .get(field)
        .ok_or(ExpertSlabsError::MissingField(field))?;
    let found = value.as_u64().ok_or(ExpertSlabsError::WrongType {
        field,
        expected: "a non-negative integer",
    })?;
    if found != want {
        return Err(ExpertSlabsError::Mismatch {
            field,
            expected: want.to_string(),
            found: found.to_string(),
        });
    }
    Ok(want as u32)
}

fn expect_str(
    obj: &serde_json::Map<String, serde_json::Value>,
    field: &'static str,
    want: &'static str,
) -> Result<String, ExpertSlabsError> {
    let value = obj
        .get(field)
        .ok_or(ExpertSlabsError::MissingField(field))?;
    let found = value.as_str().ok_or(ExpertSlabsError::WrongType {
        field,
        expected: "a string",
    })?;
    if found != want {
        return Err(ExpertSlabsError::Mismatch {
            field,
            expected: format!("{want:?}"),
            found: format!("{found:?}"),
        });
    }
    Ok(want.to_string())
}

/// Parse the HFQM metadata JSON. `Ok(None)` when [`EXPERT_SLABS_METADATA_KEY`]
/// is absent (a legacy artifact); `Err` when present but anything other than
/// the exact version-1 object: unknown or missing fields, wrong types, or any
/// value that differs from v1 (including `version != 1`).
pub fn parse(metadata_json: &str) -> Result<Option<ExpertSlabsV1>, ExpertSlabsError> {
    let root: serde_json::Value = serde_json::from_str(metadata_json)
        .map_err(|e| ExpertSlabsError::InvalidJson(e.to_string()))?;
    let root = root
        .as_object()
        .ok_or(ExpertSlabsError::MetadataNotObject)?;
    let Some(slabs) = root.get(EXPERT_SLABS_METADATA_KEY) else {
        return Ok(None);
    };
    let obj = slabs.as_object().ok_or(ExpertSlabsError::NotAnObject)?;
    if let Some(unknown) = obj.keys().find(|k| !V1_FIELDS.contains(&k.as_str())) {
        return Err(ExpertSlabsError::UnknownField(unknown.clone()));
    }
    // `version` first so a future-version object reports the version, not
    // whichever other field happens to differ.
    let version = expect_u64(obj, "version", 1)?;
    Ok(Some(ExpertSlabsV1 {
        version,
        trunk_layers: expect_u64(obj, "trunk_layers", u64::from(TRUNK_LAYERS))?,
        experts_per_layer: expect_u64(obj, "experts_per_layer", u64::from(EXPERTS_PER_LAYER))?,
        payload_order: expect_str(obj, "payload_order", PAYLOAD_ORDER)?,
        payload_alignment: expect_u64(obj, "payload_alignment", u64::from(PAYLOAD_ALIGNMENT))?,
        mtp_layout: expect_str(obj, "mtp_layout", MTP_LAYOUT)?,
    }))
}

/// The four legal per-expert qtype pairs `(gate_up_qt, down_qt)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertFormat {
    /// `(131, 132)`: W3 planar, `MQ3I4G128R256` gate_up + `MQ3I4G128R128` down.
    Mq3I4,
    /// `(133, 134)`: W2 planar, `MQ2I4G128R256` gate_up + `MQ2I4G128R128` down.
    Mq2I4,
    /// `(135, 136)`: E8P-I4 planar, R256 gate_up + R128 down (codec byte 3).
    E8pI4,
    /// `(44, 53)`: copied symmetric MQ4-XTS, `MQ4G256V2` gate_up + `MQ4G128V2` down.
    Mq4Xts,
}

impl ExpertFormat {
    /// The authoritative `(gate_up_qt, down_qt)` pair of this format.
    pub const fn pair(self) -> (u8, u8) {
        match self {
            Self::Mq3I4 => (131, 132),
            Self::Mq2I4 => (133, 134),
            Self::E8pI4 => (135, 136),
            Self::Mq4Xts => (44, 53),
        }
    }

    /// Inverse of [`pair`](Self::pair); `None` for any mixed or unknown pair.
    pub const fn from_pair(gate_up_qt: u8, down_qt: u8) -> Option<Self> {
        match (gate_up_qt, down_qt) {
            (131, 132) => Some(Self::Mq3I4),
            (133, 134) => Some(Self::Mq2I4),
            (135, 136) => Some(Self::E8pI4),
            (44, 53) => Some(Self::Mq4Xts),
            _ => None,
        }
    }

    /// Payload bytes of one expert's gate_up entry (`[1280, 2560]`).
    pub fn gate_up_bytes(self) -> usize {
        let [m, k] = GATE_UP_SHAPE.map(|d| d as usize);
        match self {
            Self::Mq3I4 => mq_i4_bytes(MqI4Profile::MQ3_R256, m, k),
            Self::Mq2I4 => mq_i4_bytes(MqI4Profile::MQ2_R256, m, k),
            Self::E8pI4 => m * k / 4 + m * (k / 128) * 2,
            // MQ4G256V2: 136 B per 256-wide group per row.
            Self::Mq4Xts => m * (k / 256) * 136,
        }
    }

    /// Payload bytes of one expert's down entry (`[2560, 640]`).
    pub fn down_bytes(self) -> usize {
        let [m, k] = DOWN_SHAPE.map(|d| d as usize);
        match self {
            Self::Mq3I4 => mq_i4_bytes(MqI4Profile::MQ3_R128, m, k),
            Self::Mq2I4 => mq_i4_bytes(MqI4Profile::MQ2_R128, m, k),
            Self::E8pI4 => m * k / 4 + m * (k / 128) * 2,
            // MQ4G128V2: 68 B per 128-wide group per row.
            Self::Mq4Xts => m * (k / 128) * 68,
        }
    }

    /// Payload bytes of one expert slab: gate_up tensor then down tensor.
    pub fn slab_bytes(self) -> usize {
        self.gate_up_bytes() + self.down_bytes()
    }

    /// The `(gate_up, down)` dtypes of this format's two tensors (the
    /// [`pair`](Self::pair) quant types). Which of them a device route serves
    /// is the route's decision, not this mapping's.
    pub const fn dtypes(self) -> (rdna_compute::DType, rdna_compute::DType) {
        use rdna_compute::DType;
        match self {
            Self::Mq3I4 => (DType::MQ3I4G128R256, DType::MQ3I4G128R128),
            Self::Mq2I4 => (DType::MQ2I4G128R256, DType::MQ2I4G128R128),
            Self::E8pI4 => (DType::E8PI4G128R256, DType::E8PI4G128R128),
            Self::Mq4Xts => (DType::MQ4G256V2, DType::MQ4G128V2),
        }
    }
}

/// `tensor_bytes` for a frozen FN shape, which is legal for the profile by
/// construction (K divisible by 128 and by the profile's rotation width).
fn mq_i4_bytes(profile: MqI4Profile, m: usize, k: usize) -> usize {
    mq_i4::tensor_bytes(profile, m, k).expect("FN expert shape is legal for its MQ-I4 profile")
}

/// Which projection of an expert an HFQM entry holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertProj {
    GateUp,
    Down,
}

impl ExpertProj {
    /// Entry-name suffix after the expert index (without the leading dot).
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::GateUp => "gate_up_proj",
            Self::Down => "down_proj",
        }
    }

    /// The frozen rank-2 shape of this projection.
    pub const fn shape(self) -> [u32; 2] {
        match self {
            Self::GateUp => GATE_UP_SHAPE,
            Self::Down => DOWN_SHAPE,
        }
    }
}

const NAME_PREFIX: &str = "model.language_model.layers.";
const NAME_MIDDLE: &str = ".mlp.experts.";

/// `model.language_model.layers.{layer}.mlp.experts.{expert}.gate_up_proj`
/// (or `.down_proj`), decimal indices without padding.
pub fn expert_entry_name(layer: usize, expert: usize, proj: ExpertProj) -> String {
    format!(
        "{NAME_PREFIX}{layer}{NAME_MIDDLE}{expert}.{}",
        proj.suffix()
    )
}

/// Strict canonical decimal: ASCII digits only, no sign, no leading zeros
/// (except `"0"` itself), must fit `usize`.
fn strict_decimal(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.is_empty() || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    if bytes.len() > 1 && bytes[0] == b'0' {
        return None;
    }
    s.parse().ok()
}

/// Inverse of [`expert_entry_name`]; `None` for any other name, including
/// indices with signs, padding or non-digits. Does **not** range-check the
/// indices against [`TRUNK_LAYERS`] / [`EXPERTS_PER_LAYER`] — that is the
/// loader's job.
pub fn parse_expert_entry_name(name: &str) -> Option<(usize, usize, ExpertProj)> {
    let rest = name.strip_prefix(NAME_PREFIX)?;
    let (layer, rest) = rest.split_once(NAME_MIDDLE)?;
    let (expert, proj) = rest.split_once('.')?;
    let proj = match proj {
        "gate_up_proj" => ExpertProj::GateUp,
        "down_proj" => ExpertProj::Down,
        _ => return None,
    };
    Some((strict_decimal(layer)?, strict_decimal(expert)?, proj))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn metadata_with(slabs: Value) -> String {
        json!({"arch": "qwen4", EXPERT_SLABS_METADATA_KEY: slabs, "other": [1, 2]}).to_string()
    }

    fn v1_json() -> Value {
        ExpertSlabsV1::fn_v1().to_json_value()
    }

    #[test]
    fn fn_v1_is_the_frozen_design_value() {
        let v = ExpertSlabsV1::fn_v1();
        assert_eq!(v.version, 1);
        assert_eq!(v.trunk_layers, 48);
        assert_eq!(v.experts_per_layer, 512);
        assert_eq!(v.payload_order, "layer,expert,gate_up,down");
        assert_eq!(v.payload_alignment, 256);
        assert_eq!(v.mtp_layout, "legacy_aggregate_unchanged");
        assert_eq!(EXPERT_SLABS_METADATA_KEY, "qwen4.expert_slabs");
        assert_eq!(
            v1_json(),
            json!({
                "version": 1,
                "trunk_layers": 48,
                "experts_per_layer": 512,
                "payload_order": "layer,expert,gate_up,down",
                "payload_alignment": 256,
                "mtp_layout": "legacy_aggregate_unchanged"
            })
        );
    }

    #[test]
    fn v1_json_round_trips_through_parse() {
        let parsed = parse(&metadata_with(v1_json())).unwrap();
        assert_eq!(parsed, Some(ExpertSlabsV1::fn_v1()));
        // The exact text from DESIGN F2 parses too.
        let text = r#"{"qwen4.expert_slabs":{"version":1,"trunk_layers":48,"experts_per_layer":512,"payload_order":"layer,expert,gate_up,down","payload_alignment":256,"mtp_layout":"legacy_aggregate_unchanged"}}"#;
        assert_eq!(parse(text).unwrap(), Some(ExpertSlabsV1::fn_v1()));
    }

    #[test]
    fn parse_is_none_without_the_key() {
        assert_eq!(parse("{}").unwrap(), None);
        assert_eq!(
            parse(r#"{"arch":"qwen4","qwen4.expert_slab":{}}"#).unwrap(),
            None
        );
    }

    #[test]
    fn parse_rejects_invalid_metadata_and_non_object_slabs() {
        assert!(matches!(parse("{"), Err(ExpertSlabsError::InvalidJson(_))));
        assert_eq!(parse("[]"), Err(ExpertSlabsError::MetadataNotObject));
        assert_eq!(parse("null"), Err(ExpertSlabsError::MetadataNotObject));
        for bad in [json!(null), json!(1), json!("v1"), json!([1])] {
            assert_eq!(
                parse(&metadata_with(bad)),
                Err(ExpertSlabsError::NotAnObject)
            );
        }
    }

    fn mutated(field: &str, value: Value) -> Value {
        let mut v = v1_json();
        v.as_object_mut().unwrap().insert(field.to_string(), value);
        v
    }

    #[test]
    fn parse_rejects_every_value_that_differs_from_v1() {
        for (field, value) in [
            ("version", json!(2)),
            ("version", json!(0)),
            ("trunk_layers", json!(47)),
            ("experts_per_layer", json!(256)),
            ("payload_order", json!("layer,expert,down,gate_up")),
            ("payload_alignment", json!(128)),
            ("mtp_layout", json!("split_per_expert")),
        ] {
            let err = parse(&metadata_with(mutated(field, value.clone()))).unwrap_err();
            assert!(
                matches!(&err, ExpertSlabsError::Mismatch { field: f, .. } if *f == field),
                "{field}={value}: {err:?}"
            );
        }
    }

    #[test]
    fn parse_rejects_wrong_types() {
        for (field, value) in [
            ("version", json!("1")),
            ("version", json!(1.0)),
            ("version", json!(-1)),
            ("version", json!(null)),
            ("trunk_layers", json!("48")),
            ("trunk_layers", json!(48.0)),
            ("experts_per_layer", json!([512])),
            ("payload_order", json!(5)),
            ("payload_order", json!(["layer"])),
            ("payload_alignment", json!("256")),
            ("mtp_layout", json!(true)),
        ] {
            let err = parse(&metadata_with(mutated(field, value.clone()))).unwrap_err();
            assert!(
                matches!(&err, ExpertSlabsError::WrongType { field: f, .. } if *f == field),
                "{field}={value}: {err:?}"
            );
        }
    }

    #[test]
    fn parse_rejects_unknown_and_missing_fields() {
        let err = parse(&metadata_with(mutated("extra", json!(1)))).unwrap_err();
        assert_eq!(err, ExpertSlabsError::UnknownField("extra".to_string()));

        // A near-miss spelling is unknown, not an alias.
        let err = parse(&metadata_with(mutated("payloadAlignment", json!(256)))).unwrap_err();
        assert_eq!(
            err,
            ExpertSlabsError::UnknownField("payloadAlignment".to_string())
        );

        for field in V1_FIELDS {
            let mut v = v1_json();
            v.as_object_mut().unwrap().remove(field);
            let err = parse(&metadata_with(v)).unwrap_err();
            assert_eq!(
                err,
                ExpertSlabsError::MissingField(field),
                "missing {field}"
            );
        }
        // Empty object: nothing from v1 is present.
        assert_eq!(
            parse(&metadata_with(json!({}))),
            Err(ExpertSlabsError::MissingField("version"))
        );
    }

    #[test]
    fn version_2_reports_the_version_first() {
        let mut v = mutated("version", json!(2));
        v.as_object_mut()
            .unwrap()
            .insert("trunk_layers".to_string(), json!(47));
        assert!(matches!(
            parse(&metadata_with(v)),
            Err(ExpertSlabsError::Mismatch {
                field: "version",
                ..
            })
        ));
    }

    #[test]
    fn expert_entry_names_round_trip() {
        assert_eq!(
            expert_entry_name(0, 0, ExpertProj::GateUp),
            "model.language_model.layers.0.mlp.experts.0.gate_up_proj"
        );
        assert_eq!(
            expert_entry_name(47, 511, ExpertProj::Down),
            "model.language_model.layers.47.mlp.experts.511.down_proj"
        );
        for (l, e) in [(0, 0), (1, 9), (10, 100), (47, 511)] {
            for proj in [ExpertProj::GateUp, ExpertProj::Down] {
                let name = expert_entry_name(l, e, proj);
                assert_eq!(parse_expert_entry_name(&name), Some((l, e, proj)), "{name}");
            }
        }
    }

    #[test]
    fn expert_entry_name_parse_is_strict_decimal_but_not_range_checked() {
        let p = |l: &str, e: &str, s: &str| {
            parse_expert_entry_name(&format!(
                "model.language_model.layers.{l}.mlp.experts.{e}.{s}"
            ))
        };
        // Range is the loader's job: out-of-range indices still parse.
        assert_eq!(
            p("48", "512", "gate_up_proj"),
            Some((48, 512, ExpertProj::GateUp))
        );
        assert_eq!(
            p("0", "1000000", "down_proj"),
            Some((0, 1_000_000, ExpertProj::Down))
        );
        // Leading zeros, signs, padding, empties and non-digits are not canonical.
        for (l, e) in [
            ("01", "0"),
            ("0", "01"),
            ("00", "0"),
            ("0", "00"),
            ("-1", "0"),
            ("0", "-1"),
            ("+1", "0"),
            ("0", "+1"),
            ("", "0"),
            ("0", ""),
            ("a", "0"),
            ("0", "x"),
            ("1_0", "0"),
            ("0", "1.5"),
            (" 1", "0"),
            ("99999999999999999999999999", "0"),
        ] {
            assert_eq!(p(l, e, "gate_up_proj"), None, "layer {l:?} expert {e:?}");
        }
        // Wrong suffixes, prefixes and trailing junk.
        assert_eq!(p("0", "0", "up_proj"), None);
        assert_eq!(p("0", "0", "gate_up_proj.weight"), None);
        assert_eq!(p("0", "0", "down_proj.0"), None);
        assert_eq!(
            parse_expert_entry_name("model.layers.0.mlp.experts.0.down_proj"),
            None
        );
        assert_eq!(
            parse_expert_entry_name("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            None
        );
        assert_eq!(
            parse_expert_entry_name("model.language_model.layers.0.mlp.experts.0.gate_up_proj "),
            None
        );
        // The legacy aggregate tensors are not per-expert entries.
        assert_eq!(
            parse_expert_entry_name("model.language_model.layers.0.mlp.experts.gate_up_proj"),
            None
        );
        assert_eq!(parse_expert_entry_name(""), None);
    }

    #[test]
    fn expert_format_pairs() {
        for (fmt, pair) in [
            (ExpertFormat::Mq3I4, (131u8, 132u8)),
            (ExpertFormat::Mq2I4, (133, 134)),
            (ExpertFormat::E8pI4, (135, 136)),
            (ExpertFormat::Mq4Xts, (44, 53)),
        ] {
            assert_eq!(fmt.pair(), pair);
            assert_eq!(ExpertFormat::from_pair(pair.0, pair.1), Some(fmt));
        }
        for (g, d) in [
            (131u8, 134u8),
            (133, 132),
            (131, 133),
            (132, 131),
            (44, 132),
            (131, 53),
            (44, 44),
            (53, 44),
            (53, 53),
            (0, 0),
        ] {
            assert_eq!(ExpertFormat::from_pair(g, d), None, "({g},{d})");
        }
    }

    #[test]
    fn expert_slab_sizes_match_the_design_and_are_256_aligned() {
        assert_eq!(ExpertFormat::Mq3I4.gate_up_bytes(), 1_280_000);
        assert_eq!(ExpertFormat::Mq3I4.down_bytes(), 640_000);
        assert_eq!(ExpertFormat::Mq3I4.slab_bytes(), 1_920_000);
        assert_eq!(ExpertFormat::Mq2I4.gate_up_bytes(), 870_400);
        assert_eq!(ExpertFormat::Mq2I4.down_bytes(), 435_200);
        assert_eq!(ExpertFormat::Mq2I4.slab_bytes(), 1_305_600);
        assert_eq!(ExpertFormat::E8pI4.gate_up_bytes(), 870_400);
        assert_eq!(ExpertFormat::E8pI4.down_bytes(), 435_200);
        assert_eq!(ExpertFormat::E8pI4.slab_bytes(), 1_305_600);
        assert_eq!(ExpertFormat::Mq4Xts.gate_up_bytes(), 1_740_800);
        assert_eq!(ExpertFormat::Mq4Xts.down_bytes(), 870_400);
        assert_eq!(ExpertFormat::Mq4Xts.slab_bytes(), 2_611_200);
        for fmt in [
            ExpertFormat::Mq3I4,
            ExpertFormat::Mq2I4,
            ExpertFormat::E8pI4,
            ExpertFormat::Mq4Xts,
        ] {
            assert_eq!(fmt.slab_bytes() % 256, 0, "{fmt:?}");
            assert_eq!(fmt.gate_up_bytes() % 256, 0, "{fmt:?}");
            assert_eq!(fmt.down_bytes() % 256, 0, "{fmt:?}");
        }
    }

    #[test]
    fn projection_shapes_and_suffixes() {
        assert_eq!(ExpertProj::GateUp.shape(), [1280, 2560]);
        assert_eq!(ExpertProj::Down.shape(), [2560, 640]);
        assert_eq!(GATE_UP_SHAPE, [1280, 2560]);
        assert_eq!(DOWN_SHAPE, [2560, 640]);
        assert_eq!(ExpertProj::GateUp.suffix(), "gate_up_proj");
        assert_eq!(ExpertProj::Down.suffix(), "down_proj");
    }
}
