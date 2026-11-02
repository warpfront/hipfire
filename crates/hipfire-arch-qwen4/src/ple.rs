// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! SSD-resident PLE row identity: stored metadata, signed wrapping hashes, and
//! the two-token EOS-aware history.  This module does not read a file or own a
//! scheduler.  It only computes checked physical row ids for a model-owned
//! reader.

use crate::config::Qwen4PleConfig;
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
/// Upper bound on stored head tables a generalized (developer-profile) layout
/// may carry: `(ngram_size - 1) * heads_per_ngram` with at most 64 heads per
/// n-gram order.
pub const PLE_MAX_HEAD_COUNT: usize = 128;

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
    /// Hash metadata disagrees with the typed PLE geometry it must serve.
    GeometryMismatch(String),
}

impl std::fmt::Display for PleMetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongMultiplierCount { got } => {
                write!(f, "PLE metadata requires 3 multipliers, got {got}")
            }
            Self::WrongHeadCount { got } => {
                write!(f, "PLE metadata head count {got} is not a supported layout")
            }
            Self::GeometryMismatch(reason) => {
                write!(
                    f,
                    "PLE metadata disagrees with the typed geometry: {reason}"
                )
            }
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
    head_vocab_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    valid_rows: u64,
    padded_rows: u64,
    padding_multiple: u64,
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
        Self::from_parts(
            multipliers,
            &head_vocab_sizes,
            &head_offsets,
            padded_rows,
            padding_multiple,
        )
    }

    /// General constructor for an explicit head count (even, `2..=`
    /// [`PLE_MAX_HEAD_COUNT`]: the first half are bigram heads, the second half
    /// trigram heads) and padding alignment.  Every invariant of the canonical
    /// constructor is enforced; nothing is inferred from the Qwen4 constants.
    pub fn from_parts(
        multipliers: [i64; PLE_MULTIPLIER_COUNT],
        head_vocab_sizes: &[u64],
        head_offsets: &[u64],
        padded_rows: u64,
        padding_multiple: u64,
    ) -> Result<Self, PleMetadataError> {
        if padding_multiple == 0 {
            return Err(PleMetadataError::PaddedRowsMisaligned {
                padded: padded_rows,
                multiple: padding_multiple,
            });
        }
        let head_count = head_vocab_sizes.len();
        if head_count < 2 || head_count % 2 != 0 || head_count > PLE_MAX_HEAD_COUNT {
            return Err(PleMetadataError::WrongHeadCount { got: head_count });
        }
        if head_offsets.len() != head_count {
            return Err(PleMetadataError::WrongHeadCount {
                got: head_offsets.len(),
            });
        }
        let mut valid_rows = 0u64;
        for head in 0..head_count {
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
            head_vocab_sizes: head_vocab_sizes.to_vec(),
            head_offsets: head_offsets.to_vec(),
            valid_rows,
            padded_rows,
            padding_multiple,
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

    /// Identity of the row layout: SHA-256 over the little-endian bytes of
    /// `padded_rows` (u64), each multiplier (i64), each head vocabulary size
    /// (u64), and each head offset (u64), in that order.  Sidecar files that
    /// are indexed by global row id (the PLE hot-row rank) record this value so
    /// a rank built for one hash layout is never applied to another.
    pub fn fingerprint(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.padded_rows.to_le_bytes());
        for multiplier in self.multipliers {
            hasher.update(multiplier.to_le_bytes());
        }
        for size in &self.head_vocab_sizes {
            hasher.update(size.to_le_bytes());
        }
        for offset in &self.head_offsets {
            hasher.update(offset.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Parse the canonical version-1 `qwen4_ple` object without converting
    /// typed integer arrays through floating point.
    ///
    /// The envelope is intentionally strict: version and the four canonical
    /// fields are the complete object schema.  Source tensor names and shapes
    /// remain typed I64/index records; they are not duplicated here.
    pub fn from_json_value(value: &Value) -> Result<Self, PleMetadataError> {
        let fields = Self::parse_object(value)?;
        Self::from_slices(
            &fields.multipliers,
            &fields.head_vocab_sizes,
            &fields.head_offsets,
            fields.padded_rows,
        )
    }

    /// Parse the same version-1 `qwen4_ple` object for an explicit typed
    /// profile: the head count and padding alignment come from `config`, and
    /// the result must agree with the config's row geometry
    /// ([`Qwen4PleConfig::check_hash_metadata`]).  For the Flash-Next config
    /// this is equivalent to [`Self::from_json_value`] plus the pinned check.
    pub fn from_json_value_for(
        value: &Value,
        config: &Qwen4PleConfig,
    ) -> Result<Self, PleMetadataError> {
        let fields = Self::parse_object(value)?;
        if fields.multipliers.len() != PLE_MULTIPLIER_COUNT {
            return Err(PleMetadataError::WrongMultiplierCount {
                got: fields.multipliers.len(),
            });
        }
        let multipliers = [
            fields.multipliers[0],
            fields.multipliers[1],
            fields.multipliers[2],
        ];
        let metadata = Self::from_parts(
            multipliers,
            &fields.head_vocab_sizes,
            &fields.head_offsets,
            fields.padded_rows,
            config.padding_multiple as u64,
        )?;
        config
            .check_hash_metadata(&metadata)
            .map_err(PleMetadataError::GeometryMismatch)?;
        Ok(metadata)
    }

    /// [`Self::from_json_value_for`] from JSON text.
    pub fn from_json_for(json: &str, config: &Qwen4PleConfig) -> Result<Self, String> {
        let value: Value = serde_json::from_str(json)
            .map_err(|error| format!("qwen4 PLE metadata is not valid JSON: {error}"))?;
        Self::from_json_value_for(&value, config).map_err(|error| error.to_string())
    }

    fn parse_object(value: &Value) -> Result<PleObjectFields, PleMetadataError> {
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
        let head_vocab_sizes = parse_u64_array(
            object
                .get("head_vocab_sizes")
                .ok_or(PleMetadataError::FieldMissing("head_vocab_sizes"))?,
            "head_vocab_sizes",
        )?;
        let head_offsets = parse_u64_array(
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
        Ok(PleObjectFields {
            multipliers,
            head_vocab_sizes,
            head_offsets,
            padded_rows,
        })
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

    /// Valid row count of each stored head table (length [`Self::head_count`]).
    pub fn head_vocab_sizes(&self) -> &[u64] {
        &self.head_vocab_sizes
    }

    /// Prefix-sum offset of each stored head table.
    pub fn head_offsets(&self) -> &[u64] {
        &self.head_offsets
    }

    /// Number of stored head tables: half bigram heads, half trigram heads.
    pub fn head_count(&self) -> usize {
        self.head_vocab_sizes.len()
    }

    /// Physical padding alignment of [`Self::padded_rows`].
    pub const fn padding_multiple(&self) -> u64 {
        self.padding_multiple
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
        let ngram_len = if head < self.head_count() / 2 { 2 } else { 3 };
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
    ///
    /// This is the canonical 16-head form; it panics for any other head
    /// count.  Generalized layouts use [`Self::hash_token_heads`].
    pub fn hash_token(
        &self,
        previous: [u32; 2],
        token: u32,
        eos_token_id: u32,
    ) -> [u64; PLE_HEAD_COUNT] {
        assert_eq!(
            self.head_count(),
            PLE_HEAD_COUNT,
            "hash_token requires the canonical 16-head layout; use hash_token_heads"
        );
        let mut output = [0u64; PLE_HEAD_COUNT];
        self.hash_token_heads(previous, token, eos_token_id, &mut output);
        output
    }

    /// Head-count-generic form of [`Self::hash_token`]: writes
    /// `head_count()` ids (first-half bigram heads, then trigram heads) into
    /// `output`, which must have exactly that length.
    pub fn hash_token_heads(
        &self,
        previous: [u32; 2],
        token: u32,
        eos_token_id: u32,
        output: &mut [u64],
    ) {
        let head_count = self.head_count();
        assert_eq!(output.len(), head_count, "PLE output slot count");
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
        for head in 0..head_count {
            let ngram_len = if head < head_count / 2 { 2 } else { 3 };
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
    }
}

struct PleObjectFields {
    multipliers: Vec<i64>,
    head_vocab_sizes: Vec<u64>,
    head_offsets: Vec<u64>,
    padded_rows: u64,
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
        let heads = metadata.head_count();
        let mut history = self;
        let mut ids = vec![0u64; tokens.len() * heads];
        for (slot, &token) in ids.chunks_exact_mut(heads).zip(tokens) {
            metadata.hash_token_heads(history.previous, token, history.eos_token_id, slot);
            history.push(token);
        }
        ids
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
    fn fingerprint_is_stable_and_layout_sensitive() {
        use sha2::{Digest, Sha256};
        let metadata = PleHashMetadata::qwen4();
        let mut hasher = Sha256::new();
        hasher.update(PLE_PADDED_ROWS.to_le_bytes());
        for multiplier in PLE_MULTIPLIERS {
            hasher.update(multiplier.to_le_bytes());
        }
        for size in PLE_HEAD_VOCAB_SIZES {
            hasher.update(size.to_le_bytes());
        }
        for offset in PLE_HEAD_OFFSETS {
            hasher.update(offset.to_le_bytes());
        }
        let expected: [u8; 32] = hasher.finalize().into();
        assert_eq!(metadata.fingerprint(), expected);
        assert_eq!(
            metadata.fingerprint(),
            PleHashMetadata::qwen4().fingerprint()
        );
        let mut multipliers = PLE_MULTIPLIERS;
        multipliers[2] += 1;
        let other = PleHashMetadata::from_stored(
            multipliers,
            PLE_HEAD_VOCAB_SIZES,
            PLE_HEAD_OFFSETS,
            PLE_PADDED_ROWS,
        )
        .unwrap();
        assert_ne!(metadata.fingerprint(), other.fingerprint());
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

    fn tiny_config() -> Qwen4PleConfig {
        // embed 16, one head per n-gram order, base 7, padding 8, one shard.
        Qwen4PleConfig::new(&[1], 16, 2, 1, 1, 3, 7, 8, None).unwrap()
    }

    fn tiny_value() -> Value {
        json!({"qwen4_ple": {
            "version": 1,
            "multipliers": [3, 5, 7],
            "head_vocab_sizes": [7, 11],
            "head_offsets": [0, 7],
            "padded_rows": 24
        }})
    }

    #[test]
    fn generalized_layout_accepts_two_heads_padding_eight() {
        let config = tiny_config();
        let metadata = PleHashMetadata::from_json_value_for(&tiny_value(), &config).unwrap();
        assert_eq!(metadata.head_count(), 2);
        assert_eq!(metadata.padding_multiple(), 8);
        assert_eq!((metadata.valid_rows(), metadata.padded_rows()), (18, 24));
        assert_eq!(metadata.head_vocab_sizes(), &[7, 11]);
        assert_eq!(metadata.head_offsets(), &[0, 7]);
        assert_eq!(
            PleHashMetadata::from_json_for(&tiny_value().to_string(), &config).unwrap(),
            metadata
        );
        // The canonical 16-head/128 parser refuses the same object.
        assert!(PleHashMetadata::from_json_value(&tiny_value()).is_err());
        // Distinct layouts have distinct fingerprints.
        assert_ne!(
            metadata.fingerprint(),
            PleHashMetadata::qwen4().fingerprint()
        );
    }

    #[test]
    fn generalized_layout_rejects_geometry_mismatch() {
        let config = tiny_config();
        let mut wrong_heads = tiny_value();
        wrong_heads["qwen4_ple"]["head_vocab_sizes"] = json!([7, 11, 13, 17]);
        wrong_heads["qwen4_ple"]["head_offsets"] = json!([0, 7, 18, 31]);
        wrong_heads["qwen4_ple"]["padded_rows"] = json!(48);
        assert!(matches!(
            PleHashMetadata::from_json_value_for(&wrong_heads, &config),
            Err(PleMetadataError::GeometryMismatch(_))
        ));
        let mut wrong_primes = tiny_value();
        wrong_primes["qwen4_ple"]["head_vocab_sizes"] = json!([8, 10]);
        wrong_primes["qwen4_ple"]["head_offsets"] = json!([0, 8]);
        assert!(matches!(
            PleHashMetadata::from_json_value_for(&wrong_primes, &config),
            Err(PleMetadataError::GeometryMismatch(_))
        ));
        // Odd head counts, empty heads, and a bad multiplier count fail
        // before any geometry comparison.
        let mut odd = tiny_value();
        odd["qwen4_ple"]["head_vocab_sizes"] = json!([7, 11, 13]);
        odd["qwen4_ple"]["head_offsets"] = json!([0, 7, 18]);
        assert!(matches!(
            PleHashMetadata::from_json_value_for(&odd, &config),
            Err(PleMetadataError::WrongHeadCount { got: 3 })
        ));
        let mut empty = tiny_value();
        empty["qwen4_ple"]["head_vocab_sizes"] = json!([]);
        empty["qwen4_ple"]["head_offsets"] = json!([]);
        assert!(matches!(
            PleHashMetadata::from_json_value_for(&empty, &config),
            Err(PleMetadataError::WrongHeadCount { got: 0 })
        ));
        let mut multipliers = tiny_value();
        multipliers["qwen4_ple"]["multipliers"] = json!([3, 5]);
        assert!(matches!(
            PleHashMetadata::from_json_value_for(&multipliers, &config),
            Err(PleMetadataError::WrongMultiplierCount { got: 2 })
        ));
        let mut unaligned = tiny_value();
        unaligned["qwen4_ple"]["padded_rows"] = json!(32);
        assert!(matches!(
            PleHashMetadata::from_json_value_for(&unaligned, &config),
            Err(PleMetadataError::PaddedRowsMisaligned { .. })
        ));
        // Row-count overflow and over-limit head counts.
        assert!(matches!(
            PleHashMetadata::from_parts(
                [1, 2, 3],
                &[u64::MAX / 2 + 1, 1],
                &[0, u64::MAX / 2 + 1],
                0,
                1
            ),
            Err(PleMetadataError::HeadSizeTooLarge { .. })
        ));
        let big = vec![(i64::MAX / 2) as u64; 4];
        let offsets: Vec<u64> = (0..4).map(|head| head * (i64::MAX / 2) as u64).collect();
        assert!(matches!(
            PleHashMetadata::from_parts([1, 2, 3], &big, &offsets, u64::MAX, 1),
            Err(PleMetadataError::RowCountOverflow | PleMetadataError::PaddedRowsMisaligned { .. })
        ));
        let too_many = vec![1u64; PLE_MAX_HEAD_COUNT + 2];
        let offsets: Vec<u64> = (0..too_many.len() as u64).collect();
        assert!(matches!(
            PleHashMetadata::from_parts([1, 2, 3], &too_many, &offsets, 256, 1),
            Err(PleMetadataError::WrongHeadCount { .. })
        ));
    }

    #[test]
    fn generalized_hash_token_heads_matches_manual_arithmetic() {
        let metadata = PleHashMetadata::from_json_value_for(&tiny_value(), &tiny_config()).unwrap();
        let eos = 3u32;
        let mut out = [0u64; 2];
        metadata.hash_token_heads([5, 9], 12, eos, &mut out);
        // Bigram head 0: 12*m0 ^ 9*m1 mod 7; trigram head 1: 12*m0 ^ 9*m1 ^ 5*m2 mod 11.
        let bigram = (12i64 * 3) ^ (9i64 * 5);
        let trigram = bigram ^ (5i64 * 7);
        assert_eq!(out[0], bigram.rem_euclid(7) as u64);
        assert_eq!(out[1], 7 + trigram.rem_euclid(11) as u64);
        // EOS truncates the context exactly as in the canonical form.
        metadata.hash_token_heads([5, eos], 12, eos, &mut out);
        let bigram = (12i64 * 3) ^ (eos as i64 * 5);
        let trigram = bigram ^ (eos as i64 * 7);
        assert_eq!(out[0], bigram.rem_euclid(7) as u64);
        assert_eq!(out[1], 7 + trigram.rem_euclid(11) as u64);
        // Every id addresses a valid row, token-major in `row_ids`.
        let ids = PleHistory::new(eos).row_ids(&metadata, &[1, 2, 3, 4]);
        assert_eq!(ids.len(), 8);
        assert!(ids.iter().all(|&row| metadata.is_valid_row(row)));
        assert!(ids.chunks(2).all(|pair| pair[0] < 7 && pair[1] >= 7));
        // The canonical array form panics for a non-16-head layout.
        assert!(std::panic::catch_unwind(|| {
            let metadata =
                PleHashMetadata::from_json_value_for(&tiny_value(), &tiny_config()).unwrap();
            metadata.hash_token([1, 2], 3, 4)
        })
        .is_err());
    }

    #[test]
    fn generalized_history_matches_canonical_for_sixteen_heads() {
        let metadata = PleHashMetadata::qwen4();
        let history = PleHistory::new(7);
        let tokens = [11u32, 7, 13, 17, 19];
        let expected: Vec<u64> = {
            let mut history = history;
            tokens
                .iter()
                .flat_map(|&token| history.hash_token(&metadata, token))
                .collect()
        };
        assert_eq!(history.row_ids(&metadata, &tokens), expected);
    }
}
