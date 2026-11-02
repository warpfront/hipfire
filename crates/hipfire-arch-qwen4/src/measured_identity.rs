// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Immutable measured-load identity for one admitted Qwen4 load.
//!
//! Composed view of EXISTING verified artifact seals plus the resolved
//! placement / recipe / state / route descriptor, frozen as
//! `qwen4-measured-load.v1` flat string fields. The loader supplies ACTUAL
//! verified fields (never expected registry pins); construction validates the
//! full descriptor schema and caches the canonical JSON once. Consumers borrow
//! the cached string by reference — no per-window hashing or serialization.
//!
//! Canonical form matches exporter Python exactly:
//! `json.dumps(fields, sort_keys=True, separators=(',', ':'))`
//! over a flat `BTreeMap<String, String>`. Keys sort by byte order (all
//! descriptor keys are ASCII, so this equals Python `sort_keys`); string
//! escaping follows Python `ensure_ascii=True` conventions, including
//! `\uXXXX` (and surrogate pairs) for non-ASCII text.

use std::collections::BTreeMap;

/// Schema marker of the frozen measured-load descriptor.
const MEASURED_LOAD_SCHEMA: &str = "qwen4-measured-load.v1";

/// Route-policy selection key that is NEVER part of the numerical identity.
/// `pub` so the loader skips it explicitly before insertion: the validator
/// rejects it rather than stripping it.
pub const CHOOSER_POLICY_KEY: &str = "config.developer.qwen4_mtp_derived_k";

/// Exact non-`config.*` field set from `export_width_costs._measured_load_canonical`.
const KNOWN_FIELDS: &[&str] = &[
    "schema",
    "model.sha256",
    "model.bytes",
    "ple.sha256",
    "ple.bytes",
    "ple.encoding",
    "overlay",
    "overlay.sha256",
    "overlay.bytes",
    "hotset",
    "hotset.sha256",
    "hotset.bytes",
    "placement.digest",
    "recipe",
    "state",
    "kv_backend",
    "max_seq",
    "capture_rows",
    "gpu.arch",
    "gpu.uuid",
    "scope",
    "flags",
];

/// Required non-empty scalar fields (beyond schema/SHA/decimal/overlay laws).
const REQUIRED_NONEMPTY: &[&str] = &[
    "ple.encoding",
    "placement.digest",
    "recipe",
    "state",
    "kv_backend",
    "scope",
    "flags",
    "gpu.arch",
    "gpu.uuid",
];

/// Immutable, validated measured-load identity with privately cached canonical JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4MeasuredLoadIdentity {
    canonical: String,
}

impl Qwen4MeasuredLoadIdentity {
    /// Validate the full flat descriptor and cache its canonical JSON once.
    ///
    /// `Err` carries a named diagnostic reason; the caller keeps the real
    /// incumbent load and never substitutes an expected pin.
    pub fn from_fields(fields: BTreeMap<String, String>) -> Result<Self, String> {
        validate_fields(&fields)?;
        Ok(Self {
            canonical: canonical_json_fields(&fields),
        })
    }

    /// Borrow the cached canonical JSON. No per-window serialization or hashing.
    pub fn canonical_json(&self) -> &str {
        &self.canonical
    }
}

fn is_full_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn is_positive_decimal(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|b| b.is_ascii_digit())
        && value.bytes().any(|b| b != b'0')
}

fn check_sidecar(
    fields: &BTreeMap<String, String>,
    prefix: &str,
    bare: &str,
    digest: &str,
    bytes: &str,
) -> Result<(), String> {
    if fields.contains_key(digest) || fields.contains_key(bytes) {
        match (fields.get(digest), fields.get(bytes)) {
            (Some(d), Some(b))
                if is_full_sha(d) && is_positive_decimal(b) && !fields.contains_key(bare) =>
            {
                Ok(())
            }
            _ => Err(format!("{prefix} needs applied hashes or literal none")),
        }
    } else {
        match fields.get(bare) {
            Some(value) if value == "none" => Ok(()),
            _ => Err(format!("{prefix} must be applied hashes or none")),
        }
    }
}

fn validate_fields(fields: &BTreeMap<String, String>) -> Result<(), String> {
    if fields.contains_key(CHOOSER_POLICY_KEY) {
        return Err(format!(
            "{CHOOSER_POLICY_KEY} excludes chooser policy selection, not the numerical route"
        ));
    }
    for key in fields.keys() {
        if KNOWN_FIELDS.contains(&key.as_str()) || key.starts_with("config.") {
            continue;
        }
        return Err(format!("unknown field `{key}`"));
    }
    match fields.get("schema") {
        Some(schema) if schema == MEASURED_LOAD_SCHEMA => {}
        _ => {
            return Err(format!("schema must be {MEASURED_LOAD_SCHEMA}"));
        }
    }
    for field in ["model.sha256", "ple.sha256"] {
        match fields.get(field) {
            Some(value) if is_full_sha(value) => {}
            _ => return Err(format!("{field} SHA256 required")),
        }
    }
    for field in ["model.bytes", "ple.bytes", "max_seq", "capture_rows"] {
        match fields.get(field) {
            Some(value) if is_positive_decimal(value) => {}
            _ => return Err(format!("{field} positive decimal string required")),
        }
    }
    for field in REQUIRED_NONEMPTY {
        match fields.get(*field) {
            Some(value) if !value.is_empty() => {}
            _ => return Err(format!("{field} required")),
        }
    }
    check_sidecar(
        fields,
        "overlay",
        "overlay",
        "overlay.sha256",
        "overlay.bytes",
    )?;
    check_sidecar(fields, "hotset", "hotset", "hotset.sha256", "hotset.bytes")?;
    Ok(())
}

/// Python `ensure_ascii=True` string escaping over one scalar.
fn push_escaped(out: &mut String, text: &str) {
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\u{00}'..='\u{1F}' => {
                out.push_str(&format!("\\u{:04x}", ch as u32));
            }
            '\u{7F}' => out.push_str("\\u007f"),
            ch if (ch as u32) > 0x7F => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            ch => out.push(ch),
        }
    }
}

/// Byte-identical rendering of Python `canonical(fields)`: sorted keys, `,`/`:` separators.
fn canonical_json_fields(fields: &BTreeMap<String, String>) -> String {
    let mut out = String::new();
    out.push('{');
    for (index, (key, value)) in fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        push_escaped(&mut out, key);
        out.push_str("\":\"");
        push_escaped(&mut out, value);
        out.push('"');
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clearly synthetic valid descriptor: overlay applied, hotset none.
    fn synthetic_fields() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("schema".to_owned(), MEASURED_LOAD_SCHEMA.to_owned()),
            ("model.sha256".to_owned(), "ab".repeat(32)),
            ("model.bytes".to_owned(), "123456".to_owned()),
            ("ple.sha256".to_owned(), "cd".repeat(32)),
            ("ple.bytes".to_owned(), "7890".to_owned()),
            ("ple.encoding".to_owned(), "q8-synthetic".to_owned()),
            ("overlay.sha256".to_owned(), "ef".repeat(32)),
            ("overlay.bytes".to_owned(), "456".to_owned()),
            ("hotset".to_owned(), "none".to_owned()),
            (
                "placement.digest".to_owned(),
                "synthetic-placement-digest-0".to_owned(),
            ),
            (
                "recipe".to_owned(),
                "synthetic-recipe-trunk-fp8-mtp-test-only".to_owned(),
            ),
            (
                "state".to_owned(),
                "SyntheticTestState(TestOnly)".to_owned(),
            ),
            ("kv_backend".to_owned(), "synthetic-test-backend".to_owned()),
            ("max_seq".to_owned(), "2048".to_owned()),
            ("capture_rows".to_owned(), "64".to_owned()),
            ("gpu.arch".to_owned(), "gfx1201-synthetic".to_owned()),
            ("gpu.uuid".to_owned(), "SYNTHETIC-GPU-UUID-0".to_owned()),
            (
                "scope".to_owned(),
                "synthetic-scope/chunk0/state/test-only".to_owned(),
            ),
            (
                "flags".to_owned(),
                "SyntheticFlags(test_only:true)".to_owned(),
            ),
            ("config.kernel.tile_rows".to_owned(), "64".to_owned()),
            (
                "config.developer.some_route_flag".to_owned(),
                "test-only".to_owned(),
            ),
        ])
    }

    #[test]
    fn valid_fixture_matches_python_canonical() {
        let identity = Qwen4MeasuredLoadIdentity::from_fields(synthetic_fields())
            .expect("synthetic valid descriptor");
        let expected = format!(
            concat!(
                "{{\"capture_rows\":\"64\",",
                "\"config.developer.some_route_flag\":\"test-only\",",
                "\"config.kernel.tile_rows\":\"64\",",
                "\"flags\":\"SyntheticFlags(test_only:true)\",",
                "\"gpu.arch\":\"gfx1201-synthetic\",",
                "\"gpu.uuid\":\"SYNTHETIC-GPU-UUID-0\",",
                "\"hotset\":\"none\",",
                "\"kv_backend\":\"synthetic-test-backend\",",
                "\"max_seq\":\"2048\",",
                "\"model.bytes\":\"123456\",",
                "\"model.sha256\":\"{model}\",",
                "\"overlay.bytes\":\"456\",",
                "\"overlay.sha256\":\"{overlay}\",",
                "\"placement.digest\":\"synthetic-placement-digest-0\",",
                "\"ple.bytes\":\"7890\",",
                "\"ple.encoding\":\"q8-synthetic\",",
                "\"ple.sha256\":\"{ple}\",",
                "\"recipe\":\"synthetic-recipe-trunk-fp8-mtp-test-only\",",
                "\"schema\":\"qwen4-measured-load.v1\",",
                "\"scope\":\"synthetic-scope/chunk0/state/test-only\",",
                "\"state\":\"SyntheticTestState(TestOnly)\"}}",
            ),
            model = "ab".repeat(32),
            overlay = "ef".repeat(32),
            ple = "cd".repeat(32),
        );
        assert_eq!(identity.canonical_json(), expected);
        // Round-trips as the flat string map production.rs parses.
        let parsed: BTreeMap<String, String> =
            serde_json::from_str(identity.canonical_json()).expect("canonical parses");
        assert_eq!(parsed, synthetic_fields());
    }

    #[test]
    fn insertion_order_does_not_change_canonical() {
        let forward =
            Qwen4MeasuredLoadIdentity::from_fields(synthetic_fields()).expect("forward order");
        let reversed: BTreeMap<String, String> = synthetic_fields().into_iter().rev().collect();
        let backward = Qwen4MeasuredLoadIdentity::from_fields(reversed).expect("reverse order");
        assert_eq!(forward.canonical_json(), backward.canonical_json());
    }

    #[test]
    fn hotset_applied_variant_valid() {
        let mut fields = synthetic_fields();
        fields.remove("hotset");
        fields.insert("hotset.sha256".to_owned(), "01".repeat(32));
        fields.insert("hotset.bytes".to_owned(), "321".to_owned());
        fields.remove("overlay.sha256");
        fields.remove("overlay.bytes");
        fields.insert("overlay".to_owned(), "none".to_owned());
        let identity =
            Qwen4MeasuredLoadIdentity::from_fields(fields).expect("hotset applied valid");
        assert!(identity.canonical_json().contains("\"hotset.sha256\":\""));
        assert!(identity.canonical_json().contains("\"overlay\":\"none\""));
    }

    #[test]
    fn missing_fields_rejected() {
        for field in [
            "schema",
            "model.sha256",
            "model.bytes",
            "ple.sha256",
            "ple.bytes",
            "ple.encoding",
            "hotset",
            "placement.digest",
            "recipe",
            "state",
            "kv_backend",
            "max_seq",
            "capture_rows",
            "gpu.arch",
            "gpu.uuid",
            "scope",
            "flags",
        ] {
            let mut fields = synthetic_fields();
            fields.remove(field);
            assert!(
                Qwen4MeasuredLoadIdentity::from_fields(fields).is_err(),
                "missing {field} must fail"
            );
        }
    }

    #[test]
    fn ambiguous_overlay_hotset_rejected() {
        // Bare literal plus applied hashes at once.
        let mut fields = synthetic_fields();
        fields.insert("hotset.sha256".to_owned(), "01".repeat(32));
        fields.insert("hotset.bytes".to_owned(), "321".to_owned());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());

        // Digest without bytes.
        let mut fields = synthetic_fields();
        fields.remove("hotset");
        fields.insert("hotset.sha256".to_owned(), "01".repeat(32));
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());

        // Bare literal that is not `none`.
        let mut fields = synthetic_fields();
        fields.insert("hotset".to_owned(), "absent".to_owned());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());
    }

    #[test]
    fn invalid_scalars_and_bounds_rejected() {
        // Uppercase / short / empty SHAs.
        for bad in ["AB".repeat(32), "ab".repeat(31), String::new()] {
            let mut fields = synthetic_fields();
            fields.insert("model.sha256".to_owned(), bad);
            assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());
        }
        // Zero, empty, and non-numeric decimals; leading-zero all-zero bounds.
        for (field, bad) in [
            ("max_seq", "0"),
            ("capture_rows", ""),
            ("model.bytes", "12a"),
            ("ple.bytes", "00"),
            ("max_seq", "007x"),
        ] {
            let mut fields = synthetic_fields();
            fields.insert(field.to_owned(), bad.to_owned());
            assert!(
                Qwen4MeasuredLoadIdentity::from_fields(fields).is_err(),
                "{field}={bad} must fail"
            );
        }
        // Arbitrarily large positive decimals stay valid (no fixed-width trap).
        let mut fields = synthetic_fields();
        fields.insert("max_seq".to_owned(), "9".repeat(40));
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_ok());

        // Wrong schema and empty required scalars.
        let mut fields = synthetic_fields();
        fields.insert("schema".to_owned(), "qwen4-measured-load.v0".to_owned());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());
        let mut fields = synthetic_fields();
        fields.insert("ple.encoding".to_owned(), String::new());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());
        let mut fields = synthetic_fields();
        fields.insert("gpu.uuid".to_owned(), String::new());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());
    }

    #[test]
    fn unknown_and_chooser_keys_rejected() {
        let mut fields = synthetic_fields();
        fields.insert("tuning".to_owned(), "x".to_owned());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());

        let mut fields = synthetic_fields();
        fields.insert("config".to_owned(), "x".to_owned());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());

        let mut fields = synthetic_fields();
        fields.insert(CHOOSER_POLICY_KEY.to_owned(), "3".to_owned());
        assert!(Qwen4MeasuredLoadIdentity::from_fields(fields).is_err());
    }

    #[test]
    fn python_canonical_escaping_matches() {
        let mut fields = synthetic_fields();
        fields.insert(
            "config.kernel.note".to_owned(),
            "a\"b\\c\nd caf\u{00e9} \u{1d11e}".to_owned(),
        );
        let identity =
            Qwen4MeasuredLoadIdentity::from_fields(fields).expect("escapable value valid");
        assert!(
            identity
                .canonical_json()
                .contains(r#""config.kernel.note":"a\"b\\c\nd caf\u00e9 \ud834\udd1e""#),
            "canonical escapes like Python ensure_ascii: {}",
            identity.canonical_json()
        );
    }
}
