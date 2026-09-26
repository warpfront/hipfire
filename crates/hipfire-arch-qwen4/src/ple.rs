// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SSD-resident PLE row identity: stored metadata, signed wrapping hashes, and
//! the two-token EOS-aware history.  This module does not read a file or own a
//! scheduler.  It only computes checked physical row ids for a model-owned
//! reader.

use serde_json::Value;
use std::convert::TryFrom;

/// Qwen4 has eight bigram heads and eight trigram heads.
pub const PLE_HEAD_COUNT: usize = 16;
/// Three multipliers are stored even though each output head uses only the
/// prefix required by its n-gram size.
pub const PLE_MULTIPLIER_COUNT: usize = 3;
/// Physical row width after concatenating eight bigram and eight trigram heads.
pub const PLE_ROW_WIDTH: usize = 160;
/// Total valid rows across the sixteen stored head tables.
pub const PLE_VALID_ROWS: u64 = 320_001_446;
/// HFQM physical rows, rounded up to the Qwen4 padding multiple of 128.
pub const PLE_PADDED_ROWS: u64 = 320_001_536;
/// Physical padding alignment used by the checkpoint.
pub const PLE_PADDING_MULTIPLE: u64 = 128;

/// The exact signed-int64 hash multipliers stored by the checkpoint.
///
/// They are deliberately constants rather than regenerated from a seed.  A
/// future checkpoint with different metadata must provide its own stored
/// values through [`PleHashMetadata::from_stored`].
pub const PLE_MULTIPLIERS: [i64; PLE_MULTIPLIER_COUNT] =
    [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

/// The exact valid row count of each stored PLE head table.
pub const PLE_HEAD_VOCAB_SIZES: [u64; PLE_HEAD_COUNT] = [
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159, 20_000_161, 20_000_171,
];

/// The exact prefix-sum offsets of the stored PLE head tables.
pub const PLE_HEAD_OFFSETS: [u64; PLE_HEAD_COUNT] = [
    0,
    20_000_003,
    40_000_026,
    60_000_059,
    80_000_106,
    100_000_165,
    120_000_228,
    140_000_297,
    160_000_374,
    180_000_455,
    200_000_548,
    220_000_655,
    240_000_802,
    260_000_955,
    280_001_114,
    300_001_275,
];

/// Errors in stored PLE metadata.  No partially validated descriptor is
/// constructible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PleMetadataError {
    WrongMultiplierCount {
        got: usize,
    },
    WrongHeadCount {
        got: usize,
    },
    EmptyHead {
        head: usize,
    },
    HeadSizeTooLarge {
        head: usize,
        size: u64,
    },
    OffsetNotPrefixSum {
        head: usize,
        expected: u64,
        got: u64,
    },
    RowCountOverflow,
    PaddedRowsTooSmall {
        valid: u64,
        padded: u64,
    },
    PaddedRowsMisaligned {
        padded: u64,
        multiple: u64,
    },
    VersionMissing,
    UnsupportedVersion(u64),
    UnknownField(String),
    FieldMissing(&'static str),
    FieldWrongType(&'static str),
    NumberOutOfRange(&'static str),
}

impl std::fmt::Display for PleMetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongMultiplierCount { got } => {
                write!(f, "PLE metadata requires 3 multipliers, got {got}")
            }
            Self::WrongHeadCount { got } => write!(f, "PLE metadata requires 16 heads, got {got}"),
            Self::EmptyHead { head } => write!(f, "PLE head {head} has zero valid rows"),
            Self::HeadSizeTooLarge { head, size } => {
                write!(
                    f,
                    "PLE head {head} row count {size} does not fit signed i64"
                )
            }
            Self::OffsetNotPrefixSum {
                head,
                expected,
                got,
            } => write!(
                f,
                "PLE head {head} offset {got} is not prefix sum {expected}"
            ),
            Self::RowCountOverflow => f.write_str("PLE row count overflows u64"),
            Self::PaddedRowsTooSmall { valid, padded } => {
                write!(f, "PLE padded rows {padded} are below valid rows {valid}")
            }
            Self::PaddedRowsMisaligned { padded, multiple } => {
                write!(f, "PLE padded rows {padded} are not aligned to {multiple}")
            }
            Self::VersionMissing => f.write_str("PLE metadata is missing `version`"),
            Self::UnsupportedVersion(version) => {
                write!(f, "PLE metadata has unsupported version {version}")
            }
            Self::UnknownField(field) => write!(f, "PLE metadata has unknown field `{field}`"),
            Self::FieldMissing(field) => write!(f, "PLE metadata missing `{field}`"),
            Self::FieldWrongType(field) => write!(f, "PLE metadata `{field}` has the wrong type"),
            Self::NumberOutOfRange(field) => {
                write!(f, "PLE metadata `{field}` is outside its integer range")
            }
        }
    }
}

impl std::error::Error for PleMetadataError {}

/// Validated, immutable PLE hash metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHashMetadata {
    multipliers: [i64; PLE_MULTIPLIER_COUNT],
    head_vocab_sizes: [u64; PLE_HEAD_COUNT],
    head_offsets: [u64; PLE_HEAD_COUNT],
    valid_rows: u64,
    padded_rows: u64,
}

impl PleHashMetadata {
    /// Construct metadata from stored arrays.  Prefix offsets are checked
    /// against the supplied row counts; no prime or multiplier is generated.
    pub fn from_stored(
        multipliers: [i64; PLE_MULTIPLIER_COUNT],
        head_vocab_sizes: [u64; PLE_HEAD_COUNT],
        head_offsets: [u64; PLE_HEAD_COUNT],
        padded_rows: u64,
    ) -> Result<Self, PleMetadataError> {
        Self::from_stored_with_padding(
            multipliers,
            head_vocab_sizes,
            head_offsets,
            padded_rows,
            PLE_PADDING_MULTIPLE,
        )
    }

    /// Slice-based constructor useful at a source/container boundary.
    pub fn from_slices(
        multipliers: &[i64],
        head_vocab_sizes: &[u64],
        head_offsets: &[u64],
        padded_rows: u64,
    ) -> Result<Self, PleMetadataError> {
        if multipliers.len() != PLE_MULTIPLIER_COUNT {
            return Err(PleMetadataError::WrongMultiplierCount {
                got: multipliers.len(),
            });
        }
        if head_vocab_sizes.len() != PLE_HEAD_COUNT {
            return Err(PleMetadataError::WrongHeadCount {
                got: head_vocab_sizes.len(),
            });
        }
        if head_offsets.len() != PLE_HEAD_COUNT {
            return Err(PleMetadataError::WrongHeadCount {
                got: head_offsets.len(),
            });
        }
        let multipliers = [multipliers[0], multipliers[1], multipliers[2]];
        let mut sizes = [0u64; PLE_HEAD_COUNT];
        sizes.copy_from_slice(head_vocab_sizes);
        let mut offsets = [0u64; PLE_HEAD_COUNT];
        offsets.copy_from_slice(head_offsets);
        Self::from_stored(multipliers, sizes, offsets, padded_rows)
    }

    /// Constructor with an explicit physical padding alignment for fixtures
    /// and other containers.  The production Qwen4 metadata uses 128.
    pub fn from_stored_with_padding(
        multipliers: [i64; PLE_MULTIPLIER_COUNT],
        head_vocab_sizes: [u64; PLE_HEAD_COUNT],
        head_offsets: [u64; PLE_HEAD_COUNT],
        padded_rows: u64,
        padding_multiple: u64,
    ) -> Result<Self, PleMetadataError> {
        if padding_multiple == 0 {
            return Err(PleMetadataError::PaddedRowsMisaligned {
                padded: padded_rows,
                multiple: padding_multiple,
            });
        }
        let mut valid_rows = 0u64;
        for head in 0..PLE_HEAD_COUNT {
            let size = head_vocab_sizes[head];
            if size == 0 {
                return Err(PleMetadataError::EmptyHead { head });
            }
            if i64::try_from(size).is_err() {
                return Err(PleMetadataError::HeadSizeTooLarge { head, size });
            }
            if head_offsets[head] != valid_rows {
                return Err(PleMetadataError::OffsetNotPrefixSum {
                    head,
                    expected: valid_rows,
                    got: head_offsets[head],
                });
            }
            valid_rows = valid_rows
                .checked_add(size)
                .ok_or(PleMetadataError::RowCountOverflow)?;
        }
        if padded_rows < valid_rows {
            return Err(PleMetadataError::PaddedRowsTooSmall {
                valid: valid_rows,
                padded: padded_rows,
            });
        }
        if padded_rows % padding_multiple != 0 {
            return Err(PleMetadataError::PaddedRowsMisaligned {
                padded: padded_rows,
                multiple: padding_multiple,
            });
        }
        let expected_padded = valid_rows
            .checked_add(padding_multiple - 1)
            .ok_or(PleMetadataError::RowCountOverflow)?
            / padding_multiple;
        let expected_padded = expected_padded
            .checked_mul(padding_multiple)
            .ok_or(PleMetadataError::RowCountOverflow)?;
        if padded_rows != expected_padded {
            return Err(PleMetadataError::PaddedRowsMisaligned {
                padded: padded_rows,
                multiple: padding_multiple,
            });
        }
        Ok(Self {
            multipliers,
            head_vocab_sizes,
            head_offsets,
            valid_rows,
            padded_rows,
        })
    }

    /// Return the exact pinned Qwen4 metadata after checking it through the
    /// same constructor used for artifact-provided metadata.
    pub fn qwen4() -> Self {
        Self::from_stored(
            PLE_MULTIPLIERS,
            PLE_HEAD_VOCAB_SIZES,
            PLE_HEAD_OFFSETS,
            PLE_PADDED_ROWS,
        )
        .expect("pinned Qwen4 PLE metadata must satisfy its invariants")
    }

    /// Parse the canonical version-1 `qwen4_ple` object without converting
    /// typed integer arrays through floating point.
    ///
    /// The envelope is intentionally strict: version and the four canonical
    /// fields are the complete object schema.  Source tensor names and shapes
    /// remain typed I64/index records; they are not duplicated here.
    pub fn from_json_value(value: &Value) -> Result<Self, PleMetadataError> {
        const FIELDS: [&str; 5] = [
            "version",
            "multipliers",
            "head_vocab_sizes",
            "head_offsets",
            "padded_rows",
        ];
        let object = value
            .get("qwen4_ple")
            .ok_or(PleMetadataError::FieldMissing("qwen4_ple"))?
            .as_object()
            .ok_or(PleMetadataError::FieldWrongType("qwen4_ple"))?;
        for field in object.keys() {
            if !FIELDS.contains(&field.as_str()) {
                return Err(PleMetadataError::UnknownField(field.clone()));
            }
        }
        let version = object
            .get("version")
            .ok_or(PleMetadataError::VersionMissing)?;
        let version = parse_u64_number(version, "version")?;
        if version != 1 {
            return Err(PleMetadataError::UnsupportedVersion(version));
        }
        let multipliers = parse_i64_array(
            object
                .get("multipliers")
                .ok_or(PleMetadataError::FieldMissing("multipliers"))?,
            "multipliers",
        )?;
        let sizes = parse_u64_array(
            object
                .get("head_vocab_sizes")
                .ok_or(PleMetadataError::FieldMissing("head_vocab_sizes"))?,
            "head_vocab_sizes",
        )?;
        let offsets = parse_u64_array(
            object
                .get("head_offsets")
                .ok_or(PleMetadataError::FieldMissing("head_offsets"))?,
            "head_offsets",
        )?;
        let padded_rows = parse_u64_number(
            object
                .get("padded_rows")
                .ok_or(PleMetadataError::FieldMissing("padded_rows"))?,
            "padded_rows",
        )?;
        Self::from_slices(&multipliers, &sizes, &offsets, padded_rows)
    }

    /// Parse a JSON metadata envelope containing the canonical `qwen4_ple`
    /// object.
    pub fn from_json(json: &str) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| format!("qwen4 PLE metadata is not valid JSON: {error}"))?;
        Self::from_json_value(&value).map_err(|error| error.to_string())
    }

    pub fn multipliers(&self) -> &[i64; PLE_MULTIPLIER_COUNT] {
        &self.multipliers
    }

    pub fn head_vocab_sizes(&self) -> &[u64; PLE_HEAD_COUNT] {
        &self.head_vocab_sizes
    }

    pub fn head_offsets(&self) -> &[u64; PLE_HEAD_COUNT] {
        &self.head_offsets
    }

    pub const fn valid_rows(&self) -> u64 {
        self.valid_rows
    }

    pub const fn padded_rows(&self) -> u64 {
        self.padded_rows
    }

    /// Whether `row` addresses a real row, rather than HFQM padding.
    pub const fn is_valid_row(&self, row: u64) -> bool {
        row < self.valid_rows
    }

    /// Checked positive remainder used by the source hash oracle.
    pub fn positive_remainder(&self, value: i64, head: usize) -> Result<u64, PleMetadataError> {
        let modulus = *self
            .head_vocab_sizes
            .get(head)
            .ok_or(PleMetadataError::WrongHeadCount { got: head })?;
        let modulus = i64::try_from(modulus).map_err(|_| PleMetadataError::HeadSizeTooLarge {
            head,
            size: modulus,
        })?;
        Ok(value.rem_euclid(modulus) as u64)
    }

    /// Hash one n-gram's already normalized token inputs into one head table.
    /// Multiplication is explicitly wrapping signed-i64 arithmetic.
    pub fn hash_one(&self, tokens: &[u32], head: usize) -> Result<u64, PleMetadataError> {
        let ngram_len = if head < PLE_HEAD_COUNT / 2 { 2 } else { 3 };
        if tokens.len() != ngram_len {
            return Err(PleMetadataError::FieldWrongType("ngram token count"));
        }
        let mut mixed = (tokens[0] as i64).wrapping_mul(self.multipliers[0]);
        for (position, &token) in tokens.iter().enumerate().skip(1) {
            mixed ^= (token as i64).wrapping_mul(self.multipliers[position]);
        }
        let remainder = self.positive_remainder(mixed, head)?;
        Ok(self.head_offsets[head] + remainder)
    }

    /// Hash one current token using the supplied two-token history and EOS
    /// boundary.  The returned order is eight bigram ids followed by eight
    /// trigram ids, matching the concatenated PLE row.
    pub fn hash_token(
        &self,
        previous: [u32; 2],
        token: u32,
        eos_token_id: u32,
    ) -> [u64; PLE_HEAD_COUNT] {
        let mut output = [0u64; PLE_HEAD_COUNT];
        // The source implementation's shifted context is equivalent to using
        // the trailing non-EOS segment of the two-token history.  A shift that
        // crosses EOS is replaced with EOS, never with an older token.
        let trailing_non_eos = if previous[1] == eos_token_id {
            0
        } else if previous[0] == eos_token_id {
            1
        } else {
            2
        };
        let context = [previous[0], previous[1], token];
        for head in 0..PLE_HEAD_COUNT {
            let ngram_len = if head < PLE_HEAD_COUNT / 2 { 2 } else { 3 };
            let mut mixed = (context[2] as i64).wrapping_mul(self.multipliers[0]);
            for shift in 1..ngram_len {
                let source_index = 2 - shift;
                let source = if shift <= trailing_non_eos {
                    context[source_index]
                } else {
                    eos_token_id
                };
                mixed ^= (source as i64).wrapping_mul(self.multipliers[shift]);
            }
            let modulus = self.head_vocab_sizes[head] as i64;
            let remainder = mixed.rem_euclid(modulus) as u64;
            output[head] = self.head_offsets[head] + remainder;
        }
        output
    }
}

fn parse_u64_number(value: &Value, field: &'static str) -> Result<u64, PleMetadataError> {
    if !value.is_number() {
        return Err(PleMetadataError::FieldWrongType(field));
    }
    value
        .as_u64()
        .ok_or(PleMetadataError::NumberOutOfRange(field))
}

fn parse_i64_array(value: &Value, field: &'static str) -> Result<Vec<i64>, PleMetadataError> {
    value
        .as_array()
        .ok_or(PleMetadataError::FieldWrongType(field))?
        .iter()
        .map(|value| {
            value
                .as_i64()
                .ok_or(PleMetadataError::NumberOutOfRange(field))
        })
        .collect()
}

fn parse_u64_array(value: &Value, field: &'static str) -> Result<Vec<u64>, PleMetadataError> {
    value
        .as_array()
        .ok_or(PleMetadataError::FieldWrongType(field))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .ok_or(PleMetadataError::NumberOutOfRange(field))
        })
        .collect()
}

/// Mutable two-token PLE context.  `ids_for_token` computes ids before
/// advancing history, exactly like the model's causal embedding lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleHistory {
    previous: [u32; 2],
    eos_token_id: u32,
}

impl PleHistory {
    /// Start a new sequence with both context slots at EOS.
    pub const fn new(eos_token_id: u32) -> Self {
        Self {
            previous: [eos_token_id; 2],
            eos_token_id,
        }
    }

    /// Start from an already captured two-token context.
    pub const fn from_previous(eos_token_id: u32, previous: [u32; 2]) -> Self {
        Self {
            previous,
            eos_token_id,
        }
    }

    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub const fn previous(&self) -> [u32; 2] {
        self.previous
    }

    pub fn reset(&mut self) {
        self.previous = [self.eos_token_id; 2];
    }

    /// Compute the sixteen row ids without changing this history.
    pub fn ids_for_token(&self, metadata: &PleHashMetadata, token: u32) -> [u64; PLE_HEAD_COUNT] {
        metadata.hash_token(self.previous, token, self.eos_token_id)
    }

    /// Compute ids and then retain the current token as the newest history
    /// slot.  This is the only mutating operation needed by a sequential PLE
    /// reader.
    pub fn hash_token(&mut self, metadata: &PleHashMetadata, token: u32) -> [u64; PLE_HEAD_COUNT] {
        let ids = self.ids_for_token(metadata, token);
        self.previous = [self.previous[1], token];
        ids
    }

    pub fn push(&mut self, token: u32) {
        self.previous = [self.previous[1], token];
    }

    /// The token-major, head-minor row ids of `tokens`, hashed from this
    /// history without advancing it: callers commit history only after the
    /// surrounding request commits.
    pub fn row_ids(self, metadata: &PleHashMetadata, tokens: &[u32]) -> Vec<u64> {
        let mut history = self;
        tokens
            .iter()
            .flat_map(|&token| history.hash_token(metadata, token))
            .collect()
    }
}
pub type PleRowId = u64;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn canonical_value() -> Value {
        let mut offsets = Vec::new();
        for head in 0..PLE_HEAD_COUNT {
            offsets.push(head as u64 * 128);
        }
        json!({
            "qwen4_ple": {
                "version": 1,
                "multipliers": [-3, 5, 7],
                "head_vocab_sizes": [128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128],
                "head_offsets": offsets,
                "padded_rows": 2048
            }
        })
    }

    #[test]
    fn pinned_metadata_has_exact_prefix_and_padding() {
        let metadata = PleHashMetadata::qwen4();
        assert_eq!(metadata.multipliers(), &PLE_MULTIPLIERS);
        assert_eq!(metadata.head_vocab_sizes(), &PLE_HEAD_VOCAB_SIZES);
        assert_eq!(metadata.head_offsets(), &PLE_HEAD_OFFSETS);
        assert_eq!(metadata.valid_rows(), PLE_VALID_ROWS);
        assert_eq!(metadata.padded_rows(), PLE_PADDED_ROWS);
        assert!(metadata.is_valid_row(PLE_VALID_ROWS - 1));
        assert!(!metadata.is_valid_row(PLE_VALID_ROWS));
        assert!(!metadata.is_valid_row(PLE_PADDED_ROWS - 1));
    }

    #[test]
    fn rejects_wrong_lengths_prefixes_and_padding() {
        let sizes = [2u64; PLE_HEAD_COUNT];
        let mut offsets = [0u64; PLE_HEAD_COUNT];
        for head in 1..PLE_HEAD_COUNT {
            offsets[head] = offsets[head - 1] + sizes[head - 1];
        }
        let error = PleHashMetadata::from_slices(&[1, 2], &[1], &[0], 128).unwrap_err();
        assert_eq!(error, PleMetadataError::WrongMultiplierCount { got: 2 });
        offsets[3] += 1;
        let error = PleHashMetadata::from_stored([1, 2, 3], sizes, offsets, 128).unwrap_err();
        assert!(matches!(
            error,
            PleMetadataError::OffsetNotPrefixSum { head: 3, .. }
        ));
        offsets[3] -= 1;
        let error = PleHashMetadata::from_stored([1, 2, 3], sizes, offsets, 16).unwrap_err();
        assert!(matches!(error, PleMetadataError::PaddedRowsTooSmall { .. }));
    }

    #[test]
    fn eos_boundary_resets_bigrams_and_trigrams() {
        let metadata = PleHashMetadata::from_stored_with_padding(
            [3, 5, 7],
            [101; PLE_HEAD_COUNT],
            {
                let mut offsets = [0u64; PLE_HEAD_COUNT];
                let mut offset = 0;
                let mut head = 0;
                while head < PLE_HEAD_COUNT {
                    offsets[head] = offset;
                    offset += 101;
                    head += 1;
                }
                offsets
            },
            1664,
            128,
        )
        .unwrap();
        let eos = 99;
        let mut history = PleHistory::new(eos);
        let _ = history.hash_token(&metadata, 10);
        let _ = history.hash_token(&metadata, eos);
        let after_eos = history.hash_token(&metadata, 20);
        let fresh = PleHistory::new(eos).ids_for_token(&metadata, 20);
        assert_eq!(after_eos, fresh);
    }

    #[test]
    fn wrapping_signed_hash_matches_manual_i64_arithmetic() {
        let sizes = [17u64; PLE_HEAD_COUNT];
        let mut offsets = [0u64; PLE_HEAD_COUNT];
        for head in 1..PLE_HEAD_COUNT {
            offsets[head] = offsets[head - 1] + sizes[head - 1];
        }
        let metadata = PleHashMetadata::from_stored_with_padding(
            [i64::MAX, i64::MIN, -3],
            sizes,
            offsets,
            272,
            16,
        )
        .unwrap();
        let previous = [13, 15];
        let token = 16;
        let eos = 0;
        let ids = metadata.hash_token(previous, token, eos);
        let mixed_bigram = (token as i64).wrapping_mul(i64::MAX) ^ (15i64).wrapping_mul(i64::MIN);
        let expected_bigram = mixed_bigram.rem_euclid(17) as u64;
        assert_eq!(ids[0], expected_bigram);
        let mixed_trigram = mixed_bigram ^ (13i64).wrapping_mul(-3);
        let expected_trigram = mixed_trigram.rem_euclid(17) as u64;
        assert_eq!(ids[8], offsets[8] + expected_trigram);
    }

    #[test]
    fn json_parser_roundtrips_canonical_versioned_metadata() {
        let value = canonical_value();
        let metadata = PleHashMetadata::from_json_value(&value).unwrap();
        assert_eq!(metadata.multipliers(), &[-3, 5, 7]);
        assert_eq!(metadata.head_vocab_sizes(), &[128; PLE_HEAD_COUNT]);
        assert_eq!(metadata.head_offsets()[1], 128);
        assert_eq!(metadata.valid_rows(), 2048);
        assert_eq!(metadata.padded_rows(), 2048);
    }

    #[test]
    fn json_parser_rejects_missing_unknown_and_wrong_versions() {
        let mut missing = canonical_value();
        missing["qwen4_ple"]
            .as_object_mut()
            .unwrap()
            .remove("version");
        assert!(matches!(
            PleHashMetadata::from_json_value(&missing),
            Err(PleMetadataError::VersionMissing)
        ));

        let mut unknown_version = canonical_value();
        unknown_version["qwen4_ple"]["version"] = json!(2);
        assert!(matches!(
            PleHashMetadata::from_json_value(&unknown_version),
            Err(PleMetadataError::UnsupportedVersion(2))
        ));

        let mut float_version = canonical_value();
        float_version["qwen4_ple"]["version"] = json!(1.5);
        assert!(matches!(
            PleHashMetadata::from_json_value(&float_version),
            Err(PleMetadataError::NumberOutOfRange("version"))
        ));

        for field in [
            "schema",
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
            "physical_rows",
        ] {
            let mut value = canonical_value();
            value["qwen4_ple"][field] = json!(0);
            assert!(matches!(
                PleHashMetadata::from_json_value(&value),
                Err(PleMetadataError::UnknownField(got)) if got == field
            ));
        }
    }

    #[test]
    fn json_parser_rejects_float_out_of_range_prefix_and_padding_values() {
        let mut float_array = canonical_value();
        float_array["qwen4_ple"]["multipliers"][0] = json!(1.5);
        assert!(matches!(
            PleHashMetadata::from_json_value(&float_array),
            Err(PleMetadataError::NumberOutOfRange("multipliers"))
        ));

        let mut out_of_range = canonical_value();
        out_of_range["qwen4_ple"]["head_vocab_sizes"][0] = json!(u64::MAX);
        assert!(matches!(
            PleHashMetadata::from_json_value(&out_of_range),
            Err(PleMetadataError::HeadSizeTooLarge { head: 0, .. })
        ));

        let mut bad_prefix = canonical_value();
        bad_prefix["qwen4_ple"]["head_offsets"][1] = json!(1);
        assert!(matches!(
            PleHashMetadata::from_json_value(&bad_prefix),
            Err(PleMetadataError::OffsetNotPrefixSum { head: 1, .. })
        ));

        let mut too_small = canonical_value();
        too_small["qwen4_ple"]["padded_rows"] = json!(2047);
        assert!(matches!(
            PleHashMetadata::from_json_value(&too_small),
            Err(PleMetadataError::PaddedRowsTooSmall {
                valid: 2048,
                padded: 2047
            })
        ));

        let mut wrong_rounding = canonical_value();
        wrong_rounding["qwen4_ple"]["padded_rows"] = json!(2176);
        assert!(matches!(
            PleHashMetadata::from_json_value(&wrong_rounding),
            Err(PleMetadataError::PaddedRowsMisaligned {
                padded: 2176,
                multiple: 128
            })
        ));
    }
}
