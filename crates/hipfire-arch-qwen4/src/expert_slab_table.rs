//! Immutable per-expert payload span table for version-1 expert-slab artifacts
//! (`qwen4.expert_slabs`, DESIGN F2).
//!
//! [`build_expert_slab_table`] validates the HFQM tensor index of an artifact
//! that declares [`ExpertSlabsV1`] metadata and produces one
//! [`ExpertSlab`] per `(layer, expert)`: the format and the *actual* gate_up
//! and down payload spans. Loaders derive pointers from these spans, never
//! from `expert_id * uniform_stride`.
//!
//! The table is only constructible through the validating builder, so a
//! `&ExpertSlabTable` guarantees:
//!
//! * exactly `48 * 512 = 49,152` trunk expert entries occupy the first index
//!   positions, ordered layer, expert, gate_up then down, each exactly once;
//! * each entry is rank 2 with the frozen shape, belongs to one legal
//!   [`ExpertFormat`] pair, carries the group size of its qtype, and has the
//!   exact payload size of its format;
//! * gate_up and down payloads are contiguous, slabs follow each other
//!   back-to-back in increasing offset order, and the first is aligned to the
//!   declared payload alignment;
//! * no trunk aggregate expert tensor (layers `< 48`) exists anywhere, and no
//!   MQ-I4 qtype (131..=134) appears outside the slab entries.
//!
//! Index names are only parsed (never allocated) on the success path, and the
//! build is a single O(n) pass.

use std::fmt;

use hipfire_runtime::hfq::HfqTensorInfo;
use hipfire_runtime::mq_i4::{self, MqI4Error, MqI4Profile};

use crate::expert_slabs::{
    parse_expert_entry_name, ExpertFormat, ExpertProj, ExpertSlabsV1, EXPERTS_PER_LAYER,
    MTP_LAYOUT, PAYLOAD_ALIGNMENT, PAYLOAD_ORDER, TRUNK_LAYERS,
};

const LAYERS: usize = TRUNK_LAYERS as usize;
const EXPERTS: usize = EXPERTS_PER_LAYER as usize;
/// Number of `(layer, expert)` slabs in a version-1 artifact.
const SLAB_COUNT: usize = LAYERS * EXPERTS;
/// Number of trunk expert index entries (gate_up and down per slab).
const ENTRY_COUNT: usize = SLAB_COUNT * 2;

const AGGREGATE_PREFIX: &str = "model.language_model.layers.";
const AGGREGATE_MIDDLE: &str = ".mlp.experts.";

/// A payload byte range inside the artifact's data region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertSlabSpan {
    pub offset: usize,
    pub len: usize,
}

impl ExpertSlabSpan {
    /// Exclusive end of the span, `None` on `usize` overflow.
    pub fn end(self) -> Option<usize> {
        self.offset.checked_add(self.len)
    }
}

/// One expert's format and its two payload spans (`down` starts where
/// `gate_up` ends).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSlab {
    pub format: ExpertFormat,
    pub gate_up: ExpertSlabSpan,
    pub down: ExpertSlabSpan,
}

/// Number of experts per format across the whole table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExpertSlabCounts {
    /// `(131, 132)` experts.
    pub w3: usize,
    /// `(133, 134)` experts.
    pub w2: usize,
    /// `(135, 136)` E8P-I4 experts.
    pub e8p: usize,
    /// `(44, 53)` copied symmetric MQ4-XTS experts.
    pub mq4: usize,
}

impl ExpertSlabCounts {
    fn add(&mut self, format: ExpertFormat) {
        match format {
            ExpertFormat::Mq3I4 => self.w3 += 1,
            ExpertFormat::Mq2I4 => self.w2 += 1,
            ExpertFormat::E8pI4 => self.e8p += 1,
            ExpertFormat::Mq4Xts => self.mq4 += 1,
        }
    }
}

/// Validated per-expert span table, indexed `layer * 512 + expert`.
#[derive(Clone, PartialEq, Eq)]
pub struct ExpertSlabTable {
    slabs: Vec<ExpertSlab>,
    counts: ExpertSlabCounts,
}

impl fmt::Debug for ExpertSlabTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertSlabTable")
            .field("slabs", &self.slabs.len())
            .field("counts", &self.counts)
            .finish()
    }
}

impl ExpertSlabTable {
    /// All slabs in payload order (`layer * 512 + expert`), 24,576 of them.
    pub fn slabs(&self) -> &[ExpertSlab] {
        &self.slabs
    }

    /// Experts per format (copied).
    pub fn counts(&self) -> ExpertSlabCounts {
        self.counts
    }

    /// The slab of `(layer, expert)`; `None` outside `48 x 512`.
    pub fn slab(&self, layer: usize, expert: usize) -> Option<&ExpertSlab> {
        if layer >= LAYERS || expert >= EXPERTS {
            return None;
        }
        self.slabs.get(layer * EXPERTS + expert)
    }
}

/// One expert index entry: `(layer, expert, projection)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertEntryId {
    pub layer: usize,
    pub expert: usize,
    pub proj: ExpertProj,
}

impl ExpertEntryId {
    /// Expected id at tensor-index position `index` (`index < ENTRY_COUNT`).
    fn at(index: usize) -> Self {
        let slab = index / 2;
        Self {
            layer: slab / EXPERTS,
            expert: slab % EXPERTS,
            proj: if index % 2 == 0 {
                ExpertProj::GateUp
            } else {
                ExpertProj::Down
            },
        }
    }

    /// Position in canonical payload/index order; inverse of [`Self::at`].
    fn ordinal(self) -> usize {
        (self.layer * EXPERTS + self.expert) * 2 + usize::from(self.proj == ExpertProj::Down)
    }
}

impl fmt::Display for ExpertEntryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "layer {} expert {} {}",
            self.layer,
            self.expert,
            self.proj.suffix()
        )
    }
}

/// Why an artifact's tensor index is not a valid version-1 expert-slab layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertSlabTableError {
    /// The supplied metadata is not exactly [`ExpertSlabsV1::fn_v1`].
    MetadataMismatch { found: ExpertSlabsV1 },
    /// The entry at `index` has no canonical expert name and is not a trunk
    /// aggregate.
    InvalidName { index: usize },
    /// A canonical expert name whose layer is `>= 48` or expert `>= 512`.
    IndexOutOfRange {
        index: usize,
        layer: usize,
        expert: usize,
    },
    /// The entry expected at `index` is absent (a later entry, or the end of
    /// the index, was found instead).
    MissingEntry {
        index: usize,
        expected: ExpertEntryId,
    },
    /// The entry at `index` repeats an entry that already occurred.
    DuplicateEntry { index: usize, found: ExpertEntryId },
    /// The entry at `index` is present but out of canonical order (for
    /// example gate_up and down swapped).
    OutOfOrder {
        index: usize,
        expected: ExpertEntryId,
        found: ExpertEntryId,
    },
    /// Not the frozen rank-2 shape of the projection.
    Shape {
        entry: ExpertEntryId,
        expected: [u32; 2],
        found: Vec<u32>,
    },
    /// The expert's `(gate_up, down)` qtypes are not a legal pair.
    InvalidPair {
        layer: usize,
        expert: usize,
        gate_up_qt: u8,
        down_qt: u8,
    },
    /// `group_size` differs from the qtype's frozen group size.
    GroupSize {
        entry: ExpertEntryId,
        quant_type: u8,
        expected: u32,
        found: u32,
    },
    /// `data_size` differs from the format's exact projection size.
    Size {
        entry: ExpertEntryId,
        expected: usize,
        found: usize,
    },
    /// `data_offset + data_size` overflows `usize`.
    Range {
        entry: ExpertEntryId,
        data_offset: usize,
        data_size: usize,
    },
    /// The down payload does not start where the gate_up payload ends.
    NoncontiguousGateDown {
        layer: usize,
        expert: usize,
        gate_up_end: usize,
        down_offset: usize,
    },
    /// A slab starts before the previous slab ended.
    SlabOverlap {
        layer: usize,
        expert: usize,
        previous_end: usize,
        offset: usize,
    },
    /// A slab starts after the previous slab ended (hole between slabs).
    SlabGap {
        layer: usize,
        expert: usize,
        previous_end: usize,
        offset: usize,
    },
    /// The first slab's offset is not a multiple of the payload alignment.
    FirstSlabUnaligned { offset: usize, alignment: usize },
    /// A legacy aggregate expert tensor of trunk layer `layer < 48` exists.
    AggregateTrunkExperts { index: usize, layer: usize },
    /// An MQ-I4 qtype (131..=134) on an entry outside the slab entries.
    StrayI4Qtype { index: usize, quant_type: u8 },
    /// The slab's total length overflows `usize`.
    SpanOverflow { layer: usize, expert: usize },
}

impl fmt::Display for ExpertSlabTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetadataMismatch { found } => write!(
                f,
                "expert-slab metadata is not the version-1 Flash-Next declaration: {found:?}"
            ),
            Self::InvalidName { index } => write!(
                f,
                "tensor #{index} is not the canonical expert entry expected at that position"
            ),
            Self::IndexOutOfRange {
                index,
                layer,
                expert,
            } => write!(
                f,
                "tensor #{index} names layer {layer} expert {expert}, outside 0..{LAYERS} x 0..{EXPERTS}"
            ),
            Self::MissingEntry { index, expected } => {
                write!(f, "tensor #{index}: missing expert entry ({expected})")
            }
            Self::DuplicateEntry { index, found } => {
                write!(f, "tensor #{index}: duplicate expert entry ({found})")
            }
            Self::OutOfOrder {
                index,
                expected,
                found,
            } => write!(
                f,
                "tensor #{index}: expert entries out of order (expected {expected}, found {found})"
            ),
            Self::Shape {
                entry,
                expected,
                found,
            } => write!(f, "{entry}: shape {found:?}, expected {expected:?}"),
            Self::InvalidPair {
                layer,
                expert,
                gate_up_qt,
                down_qt,
            } => write!(
                f,
                "layer {layer} expert {expert}: qtype pair ({gate_up_qt}, {down_qt}) is not one of (131,132), (133,134), (44,53)"
            ),
            Self::GroupSize {
                entry,
                quant_type,
                expected,
                found,
            } => write!(
                f,
                "{entry}: qtype {quant_type} requires group_size {expected}, found {found}"
            ),
            Self::Size {
                entry,
                expected,
                found,
            } => write!(f, "{entry}: payload is {found} bytes, expected {expected}"),
            Self::Range {
                entry,
                data_offset,
                data_size,
            } => write!(
                f,
                "{entry}: data_offset {data_offset} + data_size {data_size} overflows"
            ),
            Self::NoncontiguousGateDown {
                layer,
                expert,
                gate_up_end,
                down_offset,
            } => write!(
                f,
                "layer {layer} expert {expert}: down starts at {down_offset} but gate_up ends at {gate_up_end}"
            ),
            Self::SlabOverlap {
                layer,
                expert,
                previous_end,
                offset,
            } => write!(
                f,
                "layer {layer} expert {expert}: slab at {offset} overlaps the previous slab ending at {previous_end}"
            ),
            Self::SlabGap {
                layer,
                expert,
                previous_end,
                offset,
            } => write!(
                f,
                "layer {layer} expert {expert}: slab at {offset} leaves a gap after the previous slab ending at {previous_end}"
            ),
            Self::FirstSlabUnaligned { offset, alignment } => write!(
                f,
                "first expert slab at {offset} is not aligned to {alignment}"
            ),
            Self::AggregateTrunkExperts { index, layer } => write!(
                f,
                "tensor #{index}: aggregate expert tensor of trunk layer {layer} coexists with per-expert entries"
            ),
            Self::StrayI4Qtype { index, quant_type } => write!(
                f,
                "tensor #{index}: MQ-I4 qtype {quant_type} outside the expert slab entries"
            ),
            Self::SpanOverflow { layer, expert } => {
                write!(f, "layer {layer} expert {expert}: slab length overflows")
            }
        }
    }
}

impl std::error::Error for ExpertSlabTableError {}

/// Layer of a trunk aggregate expert tensor name
/// (`model.language_model.layers.{L}.mlp.experts.{gate_up,down}_proj`),
/// `None` for anything else (MTP names included) or `L >= 48`.
fn trunk_aggregate_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix(AGGREGATE_PREFIX)?;
    let (layer, tail) = rest.split_once(AGGREGATE_MIDDLE)?;
    if tail != "gate_up_proj" && tail != "down_proj" {
        return None;
    }
    if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    layer.parse::<usize>().ok().filter(|&l| l < LAYERS)
}

/// Frozen `group_size` of an expert qtype.
fn expected_group(quant_type: u8) -> u32 {
    match quant_type {
        44 => 256,
        _ => 128,
    }
}

/// Validate and return the entry that must sit at `index`.
fn expect_entry(
    tensors: &[HfqTensorInfo],
    index: usize,
) -> Result<&HfqTensorInfo, ExpertSlabTableError> {
    let expected = ExpertEntryId::at(index);
    let Some(info) = tensors.get(index) else {
        return Err(ExpertSlabTableError::MissingEntry { index, expected });
    };
    let Some((layer, expert, proj)) = parse_expert_entry_name(&info.name) else {
        return Err(match trunk_aggregate_layer(&info.name) {
            Some(layer) => ExpertSlabTableError::AggregateTrunkExperts { index, layer },
            None => ExpertSlabTableError::InvalidName { index },
        });
    };
    if layer >= LAYERS || expert >= EXPERTS {
        return Err(ExpertSlabTableError::IndexOutOfRange {
            index,
            layer,
            expert,
        });
    }
    let found = ExpertEntryId {
        layer,
        expert,
        proj,
    };
    let found_ordinal = found.ordinal();
    if found_ordinal == index {
        return Ok(info);
    }
    if found_ordinal < index {
        return Err(ExpertSlabTableError::DuplicateEntry { index, found });
    }
    // The expected entry exists later in the index: a permutation, not a
    // deletion. One scan, only on this error path.
    let expected_is_later = tensors[index + 1..].iter().any(|later| {
        parse_expert_entry_name(&later.name)
            == Some((expected.layer, expected.expert, expected.proj))
    });
    Err(if expected_is_later {
        ExpertSlabTableError::OutOfOrder {
            index,
            expected,
            found,
        }
    } else {
        ExpertSlabTableError::MissingEntry { index, expected }
    })
}

fn check_shape(entry: ExpertEntryId, info: &HfqTensorInfo) -> Result<(), ExpertSlabTableError> {
    let expected = entry.proj.shape();
    if info.shape.as_slice() != expected {
        return Err(ExpertSlabTableError::Shape {
            entry,
            expected,
            found: info.shape.clone(),
        });
    }
    Ok(())
}

/// Per-format projection sizes: `[gate_up, down]` for `Mq3I4`, `Mq2I4`,
/// `Mq4Xts`, `E8pI4`.
fn format_sizes() -> [[usize; 2]; 4] {
    [
        ExpertFormat::Mq3I4,
        ExpertFormat::Mq2I4,
        ExpertFormat::Mq4Xts,
        ExpertFormat::E8pI4,
    ]
    .map(|f| [f.gate_up_bytes(), f.down_bytes()])
}

fn format_slot(format: ExpertFormat) -> usize {
    match format {
        ExpertFormat::Mq3I4 => 0,
        ExpertFormat::Mq2I4 => 1,
        ExpertFormat::Mq4Xts => 2,
        ExpertFormat::E8pI4 => 3,
    }
}

/// Validate one projection's group size and exact size; returns its span.
fn check_projection(
    entry: ExpertEntryId,
    info: &HfqTensorInfo,
    quant_type: u8,
    expected_size: usize,
) -> Result<ExpertSlabSpan, ExpertSlabTableError> {
    let expected_group = expected_group(quant_type);
    if info.group_size != expected_group {
        return Err(ExpertSlabTableError::GroupSize {
            entry,
            quant_type,
            expected: expected_group,
            found: info.group_size,
        });
    }
    if info.data_size != expected_size {
        return Err(ExpertSlabTableError::Size {
            entry,
            expected: expected_size,
            found: info.data_size,
        });
    }
    if info.data_offset.checked_add(info.data_size).is_none() {
        return Err(ExpertSlabTableError::Range {
            entry,
            data_offset: info.data_offset,
            data_size: info.data_size,
        });
    }
    Ok(ExpertSlabSpan {
        offset: info.data_offset,
        len: info.data_size,
    })
}

/// Validate `tensors` (the HFQM index in file order) against the version-1
/// expert-slab layout declared by `metadata`, and build the span table.
pub fn build_expert_slab_table(
    metadata: &ExpertSlabsV1,
    tensors: &[HfqTensorInfo],
) -> Result<ExpertSlabTable, ExpertSlabTableError> {
    build_expert_slab_table_with_keep(metadata, tensors, None)
}

/// Build a fixed original-id table; excluded entries have zero-byte spans.
pub fn build_expert_slab_table_with_keep(
    metadata: &ExpertSlabsV1,
    tensors: &[HfqTensorInfo],
    keep: Option<&crate::ExpertKeep>,
) -> Result<ExpertSlabTable, ExpertSlabTableError> {
    if metadata.version != 1
        || metadata.trunk_layers != TRUNK_LAYERS
        || metadata.experts_per_layer != EXPERTS_PER_LAYER
        || metadata.payload_order != PAYLOAD_ORDER
        || metadata.payload_alignment != PAYLOAD_ALIGNMENT
        || metadata.mtp_layout != MTP_LAYOUT
        || keep.is_some_and(|keep| keep.n_experts() != EXPERTS || keep.n_layers() != LAYERS)
    {
        return Err(ExpertSlabTableError::MetadataMismatch {
            found: metadata.clone(),
        });
    }
    let alignment = PAYLOAD_ALIGNMENT as usize;
    let sizes = format_sizes();

    let mut slabs = Vec::with_capacity(SLAB_COUNT);
    let mut counts = ExpertSlabCounts::default();
    let mut previous_end = keep.map_or(0, |_| tensors.first().map_or(0, |info| info.data_offset));
    let mut index = 0;

    for slab_index in 0..SLAB_COUNT {
        let gate_entry = ExpertEntryId::at(slab_index * 2);
        let down_entry = ExpertEntryId::at(slab_index * 2 + 1);
        let (layer, expert) = (gate_entry.layer, gate_entry.expert);
        if keep.is_some_and(|keep| !keep.is_kept(layer, expert)) {
            let offset = if slabs.is_empty() {
                tensors.first().map_or(0, |info| info.data_offset)
            } else {
                previous_end
            };
            slabs.push(ExpertSlab {
                format: ExpertFormat::Mq3I4,
                gate_up: ExpertSlabSpan { offset, len: 0 },
                down: ExpertSlabSpan { offset, len: 0 },
            });
            continue;
        }

        let gate = if keep.is_none() {
            expect_entry(tensors, index)?
        } else {
            expect_kept_entry(tensors, index, gate_entry)?
        };
        check_shape(gate_entry, gate)?;
        let down = if keep.is_none() {
            expect_entry(tensors, index + 1)?
        } else {
            expect_kept_entry(tensors, index + 1, down_entry)?
        };
        check_shape(down_entry, down)?;
        index += 2;

        let format = ExpertFormat::from_pair(gate.quant_type, down.quant_type).ok_or(
            ExpertSlabTableError::InvalidPair {
                layer,
                expert,
                gate_up_qt: gate.quant_type,
                down_qt: down.quant_type,
            },
        )?;
        let [gate_size, down_size] = sizes[format_slot(format)];
        let gate_up = check_projection(gate_entry, gate, gate.quant_type, gate_size)?;
        let down_span = check_projection(down_entry, down, down.quant_type, down_size)?;

        if index == 2 && alignment != 0 && gate_up.offset % alignment != 0 {
            return Err(ExpertSlabTableError::FirstSlabUnaligned {
                offset: gate_up.offset,
                alignment,
            });
        }
        // `check_projection` proved both ends fit in `usize`.
        let gate_end = gate_up.offset + gate_up.len;
        if down_span.offset != gate_end {
            return Err(ExpertSlabTableError::NoncontiguousGateDown {
                layer,
                expert,
                gate_up_end: gate_end,
                down_offset: down_span.offset,
            });
        }
        let slab_end = gate_end + down_span.len;
        if gate_up.len.checked_add(down_span.len).is_none() {
            return Err(ExpertSlabTableError::SpanOverflow { layer, expert });
        }
        if index > 2 {
            if gate_up.offset < previous_end {
                return Err(ExpertSlabTableError::SlabOverlap {
                    layer,
                    expert,
                    previous_end,
                    offset: gate_up.offset,
                });
            }
            if gate_up.offset > previous_end {
                return Err(ExpertSlabTableError::SlabGap {
                    layer,
                    expert,
                    previous_end,
                    offset: gate_up.offset,
                });
            }
        }
        previous_end = slab_end;
        counts.add(format);
        slabs.push(ExpertSlab {
            format,
            gate_up,
            down: down_span,
        });
    }

    for (index, info) in tensors.iter().enumerate().skip(index) {
        if let Some(layer) = trunk_aggregate_layer(&info.name) {
            return Err(ExpertSlabTableError::AggregateTrunkExperts { index, layer });
        }
        if let Some((layer, expert, proj)) = parse_expert_entry_name(&info.name) {
            return Err(if layer < LAYERS && expert < EXPERTS {
                ExpertSlabTableError::DuplicateEntry {
                    index,
                    found: ExpertEntryId {
                        layer,
                        expert,
                        proj,
                    },
                }
            } else {
                ExpertSlabTableError::IndexOutOfRange {
                    index,
                    layer,
                    expert,
                }
            });
        }
        if (131..=134).contains(&info.quant_type) {
            return Err(ExpertSlabTableError::StrayI4Qtype {
                index,
                quant_type: info.quant_type,
            });
        }
    }

    Ok(ExpertSlabTable { slabs, counts })
}

fn expect_kept_entry(
    tensors: &[HfqTensorInfo],
    index: usize,
    expected: ExpertEntryId,
) -> Result<&HfqTensorInfo, ExpertSlabTableError> {
    let info = tensors
        .get(index)
        .ok_or(ExpertSlabTableError::MissingEntry { index, expected })?;
    if parse_expert_entry_name(&info.name) != Some((expected.layer, expected.expert, expected.proj))
    {
        return Err(ExpertSlabTableError::MissingEntry { index, expected });
    }
    Ok(info)
}

/// A W3/W2 projection failed scale/code validation, or its payload was not
/// supplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpertSlabScaleError {
    /// The payload callback returned `None` for the projection's span.
    MissingPayload {
        layer: usize,
        expert: usize,
        projection: ExpertProj,
        span: ExpertSlabSpan,
    },
    /// [`mq_i4::decode_tensor`] rejected the projection bytes.
    Codec {
        layer: usize,
        expert: usize,
        projection: ExpertProj,
        source: MqI4Error,
    },
}

impl fmt::Display for ExpertSlabScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPayload {
                layer,
                expert,
                projection,
                span,
            } => write!(
                f,
                "layer {layer} expert {expert} {}: no payload for span offset {} len {}",
                projection.suffix(),
                span.offset,
                span.len
            ),
            Self::Codec {
                layer,
                expert,
                projection,
                source,
            } => write!(
                f,
                "layer {layer} expert {expert} {}: {source}",
                projection.suffix()
            ),
        }
    }
}

impl std::error::Error for ExpertSlabScaleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MissingPayload { .. } => None,
            Self::Codec { source, .. } => Some(source),
        }
    }
}

/// Scalar MQ-I4 profiles; E8P records are guarded by their dedicated provider.
fn mq_i4_profiles(format: ExpertFormat) -> Option<(MqI4Profile, MqI4Profile)> {
    match format {
        ExpertFormat::Mq3I4 => Some((MqI4Profile::MQ3_R256, MqI4Profile::MQ3_R128)),
        ExpertFormat::Mq2I4 => Some((MqI4Profile::MQ2_R256, MqI4Profile::MQ2_R128)),
        ExpertFormat::Mq4Xts | ExpertFormat::E8pI4 => None,
    }
}

/// Run the full MQ-I4 import validation ([`mq_i4::decode_tensor`]: exact
/// length, codes `<= L`, scales finite / non-negative / no `-0.0`, zero scale
/// implies midpoint codes) over every W3/W2 projection of `table`. MQ4-XTS
/// experts are skipped and `payload` is never asked for their spans.
///
/// `payload` resolves a span to that projection's bytes (`None` when it is not
/// available). The first failing projection is reported with its
/// layer/expert/projection.
pub fn validate_expert_slab_scales<'a>(
    table: &ExpertSlabTable,
    mut payload: impl FnMut(ExpertSlabSpan) -> Option<&'a [u8]>,
) -> Result<(), ExpertSlabScaleError> {
    for (slab_index, slab) in table.slabs.iter().enumerate() {
        if slab.gate_up.len == 0 && slab.down.len == 0 {
            continue;
        }
        let Some((gate_profile, down_profile)) = mq_i4_profiles(slab.format) else {
            continue;
        };
        let (layer, expert) = (slab_index / EXPERTS, slab_index % EXPERTS);
        for (projection, span, profile) in [
            (ExpertProj::GateUp, slab.gate_up, gate_profile),
            (ExpertProj::Down, slab.down, down_profile),
        ] {
            let Some(bytes) = payload(span) else {
                return Err(ExpertSlabScaleError::MissingPayload {
                    layer,
                    expert,
                    projection,
                    span,
                });
            };
            let [m, k] = projection.shape().map(|d| d as usize);
            mq_i4::decode_tensor(profile, m, k, bytes).map_err(|source| {
                ExpertSlabScaleError::Codec {
                    layer,
                    expert,
                    projection,
                    source,
                }
            })?;
        }
    }
    Ok(())
}

/// Test fixture: a full 48x512 version-1 index with realistic back-to-back
/// spans starting at offset 4096. `formats(layer, expert)` picks each
/// expert's format; after the 49,152 slab entries come one non-expert tensor
/// and the legacy MTP aggregate expert tensors.
#[cfg(test)]
pub(crate) fn full_fn_headers(
    mut formats: impl FnMut(usize, usize) -> ExpertFormat,
) -> Vec<HfqTensorInfo> {
    let mut out = Vec::with_capacity(ENTRY_COUNT + 3);
    let mut cursor = 4096usize;
    for layer in 0..LAYERS {
        for expert in 0..EXPERTS {
            let format = formats(layer, expert);
            let (gate_qt, down_qt) = format.pair();
            for (proj, qt, bytes) in [
                (ExpertProj::GateUp, gate_qt, format.gate_up_bytes()),
                (ExpertProj::Down, down_qt, format.down_bytes()),
            ] {
                out.push(HfqTensorInfo {
                    name: crate::expert_slabs::expert_entry_name(layer, expert, proj),
                    quant_type: qt,
                    shape: proj.shape().to_vec(),
                    group_size: expected_group(qt),
                    data_offset: cursor,
                    data_size: bytes,
                });
                cursor += bytes;
            }
        }
    }
    for (name, qt, shape) in [
        ("model.embed_tokens.weight", 1u8, vec![16u32, 16]),
        (
            "mtp.layers.0.mlp.experts.gate_up_proj",
            44,
            vec![512, 1280, 2560],
        ),
        (
            "mtp.layers.0.mlp.experts.down_proj",
            53,
            vec![512, 2560, 640],
        ),
    ] {
        out.push(HfqTensorInfo {
            name: name.to_string(),
            quant_type: qt,
            shape,
            group_size: expected_group(qt),
            data_offset: cursor,
            data_size: 256,
        });
        cursor += 256;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_slabs::expert_entry_name;
    use hipfire_runtime::mq_i4::MqI4Codec;

    #[test]
    fn sparse_table_keeps_original_ids_without_payload() {
        let keep = crate::ExpertKeep::new(512, vec![(16..32).collect(); 48], None, 10).unwrap();
        let mut tensors = full_fn_headers(|_, _| ExpertFormat::Mq3I4);
        tensors.retain(|info| {
            parse_expert_entry_name(&info.name).map_or(true, |(l, e, _)| keep.is_kept(l, e))
        });
        let mut cursor = 4096;
        for info in &mut tensors {
            info.data_offset = cursor;
            cursor += info.data_size;
        }
        let metadata = crate::expert_slabs::ExpertSlabsV1::fn_v1();
        let table = build_expert_slab_table_with_keep(&metadata, &tensors, Some(&keep)).unwrap();
        assert_eq!(table.counts().w3, 48 * 16);
        assert_eq!(table.slab(0, 0).unwrap().gate_up.len, 0);
        assert_ne!(table.slab(0, 16).unwrap().gate_up.len, 0);
        let plan = crate::expert_arena::ExpertArenaPlan::from_table(&table).unwrap();
        assert_eq!(
            plan.physical_bytes(),
            (48 * 16 * ExpertFormat::Mq3I4.slab_bytes()) as u64
        );
    }

    fn all_w3() -> Vec<HfqTensorInfo> {
        full_fn_headers(|_, _| ExpertFormat::Mq3I4)
    }

    fn mixed_format(layer: usize, expert: usize) -> ExpertFormat {
        match (layer * 7 + expert * 13) % 5 {
            0 | 1 => ExpertFormat::Mq3I4,
            2 | 3 => ExpertFormat::Mq2I4,
            _ => ExpertFormat::Mq4Xts,
        }
    }

    fn mixed() -> Vec<HfqTensorInfo> {
        full_fn_headers(mixed_format)
    }

    fn ix(layer: usize, expert: usize, proj: ExpertProj) -> usize {
        ExpertEntryId {
            layer,
            expert,
            proj,
        }
        .ordinal()
    }

    fn id(layer: usize, expert: usize, proj: ExpertProj) -> ExpertEntryId {
        ExpertEntryId {
            layer,
            expert,
            proj,
        }
    }

    fn build(tensors: &[HfqTensorInfo]) -> Result<ExpertSlabTable, ExpertSlabTableError> {
        build_expert_slab_table(&ExpertSlabsV1::fn_v1(), tensors)
    }

    fn err(tensors: &[HfqTensorInfo]) -> ExpertSlabTableError {
        build(tensors).expect_err("layout must be rejected")
    }

    const G: ExpertProj = ExpertProj::GateUp;
    const D: ExpertProj = ExpertProj::Down;

    #[test]
    fn all_w3_full_table_has_back_to_back_realistic_spans() {
        let tensors = all_w3();
        let table = build(&tensors).unwrap();
        assert_eq!(table.slabs().len(), SLAB_COUNT);
        assert_eq!(
            table.counts(),
            ExpertSlabCounts {
                w3: SLAB_COUNT,
                w2: 0,
                e8p: 0,
                mq4: 0
            }
        );
        let first = table.slabs()[0];
        assert_eq!(first.format, ExpertFormat::Mq3I4);
        assert_eq!(
            first.gate_up,
            ExpertSlabSpan {
                offset: 4096,
                len: 1_280_000
            }
        );
        assert_eq!(
            first.down,
            ExpertSlabSpan {
                offset: 4096 + 1_280_000,
                len: 640_000
            }
        );
        let mut cursor = 4096;
        for (i, slab) in table.slabs().iter().enumerate() {
            assert_eq!(slab.gate_up.offset, cursor, "slab {i}");
            assert_eq!(slab.down.offset, cursor + 1_280_000);
            cursor += 1_920_000;
            assert_eq!(slab.down.end(), Some(cursor));
        }
        let last = table.slab(47, 511).unwrap();
        assert_eq!(last, &table.slabs()[SLAB_COUNT - 1]);
        // Entries and spans agree one-to-one.
        for (i, slab) in table.slabs().iter().enumerate() {
            assert_eq!(tensors[2 * i].data_offset, slab.gate_up.offset);
            assert_eq!(tensors[2 * i + 1].data_offset, slab.down.offset);
        }
    }

    #[test]
    fn mixed_full_table_follows_actual_spans_not_uniform_stride() {
        let tensors = mixed();
        let table = build(&tensors).unwrap();
        let mut expect = ExpertSlabCounts::default();
        let mut cursor = 4096;
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                let format = mixed_format(layer, expert);
                expect.add(format);
                let slab = table.slab(layer, expert).unwrap();
                assert_eq!(slab.format, format);
                assert_eq!(slab.gate_up.offset, cursor);
                assert_eq!(slab.gate_up.len, format.gate_up_bytes());
                assert_eq!(slab.down.offset, cursor + format.gate_up_bytes());
                assert_eq!(slab.down.len, format.down_bytes());
                cursor += format.slab_bytes();
            }
        }
        assert_eq!(table.counts(), expect);
        assert!(expect.w3 > 0 && expect.w2 > 0 && expect.mq4 > 0);
        assert_eq!(expect.w3 + expect.w2 + expect.mq4, SLAB_COUNT);
        // Slab strides differ, so a uniform stride cannot reproduce the table.
        assert_ne!(
            table.slabs()[1].gate_up.offset - table.slabs()[0].gate_up.offset,
            table.slabs()[2].gate_up.offset - table.slabs()[1].gate_up.offset
        );
    }

    #[test]
    fn accessors_are_bounds_checked_and_counts_are_copied() {
        let table = build(&all_w3()).unwrap();
        assert!(table.slab(48, 0).is_none());
        assert!(table.slab(0, 512).is_none());
        assert_eq!(table.slab(1, 2), Some(&table.slabs()[512 + 2]));
        let mut counts = table.counts();
        counts.w3 = 0;
        assert_eq!(counts.w3, 0);
        assert_eq!(table.counts().w3, SLAB_COUNT);
        assert_eq!(
            ExpertSlabSpan {
                offset: usize::MAX,
                len: 1
            }
            .end(),
            None
        );
    }

    #[test]
    fn metadata_must_be_fn_v1() {
        let mut meta = ExpertSlabsV1::fn_v1();
        meta.payload_alignment = 128;
        assert_eq!(
            build_expert_slab_table(&meta, &all_w3()).unwrap_err(),
            ExpertSlabTableError::MetadataMismatch { found: meta }
        );
    }

    #[test]
    fn empty_and_truncated_indexes_report_the_missing_entry() {
        assert_eq!(
            err(&[]),
            ExpertSlabTableError::MissingEntry {
                index: 0,
                expected: id(0, 0, G)
            }
        );
        let mut tensors = all_w3();
        tensors.truncate(ENTRY_COUNT - 1);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::MissingEntry {
                index: ENTRY_COUNT - 1,
                expected: id(47, 511, D)
            }
        );
    }

    #[test]
    fn missing_expert_is_missing_entry() {
        let mut tensors = all_w3();
        let at = ix(3, 7, G);
        tensors.drain(at..at + 2);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::MissingEntry {
                index: at,
                expected: id(3, 7, G)
            }
        );
    }

    #[test]
    fn missing_down_only_is_missing_entry() {
        let mut tensors = all_w3();
        tensors.remove(ix(0, 9, D));
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::MissingEntry {
                index: ix(0, 9, D),
                expected: id(0, 9, D)
            }
        );
    }

    #[test]
    fn duplicate_expert_is_duplicate_entry() {
        let mut tensors = all_w3();
        let at = ix(3, 7, G);
        let copy = [tensors[at].clone(), tensors[at + 1].clone()];
        tensors.splice(at + 2..at + 2, copy);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::DuplicateEntry {
                index: at + 2,
                found: id(3, 7, G)
            }
        );
    }

    #[test]
    fn trailing_per_expert_name_is_rejected() {
        let mut tensors = all_w3();
        let mut dup = tensors[ix(5, 5, D)].clone();
        dup.quant_type = 44;
        tensors.push(dup);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::DuplicateEntry {
                index: tensors.len() - 1,
                found: id(5, 5, D)
            }
        );
        let mut tensors = all_w3();
        let mut oob = tensors[0].clone();
        oob.name = expert_entry_name(60, 3, G);
        tensors.push(oob);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::IndexOutOfRange {
                index: tensors.len() - 1,
                layer: 60,
                expert: 3
            }
        );
    }

    #[test]
    fn expert_order_violation_is_out_of_order() {
        let mut tensors = all_w3();
        let at = ix(2, 5, D);
        tensors.swap(at, at + 1); // expert 5 down before expert 6 gate
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::OutOfOrder {
                index: at,
                expected: id(2, 5, D),
                found: id(2, 6, G)
            }
        );
    }

    #[test]
    fn nonadjacent_expert_permutation_is_out_of_order_not_missing() {
        let mut tensors = all_w3();
        tensors.swap(ix(0, 0, G), ix(0, 511, G));
        tensors.swap(ix(0, 0, D), ix(0, 511, D));
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::OutOfOrder {
                index: 0,
                expected: id(0, 0, G),
                found: id(0, 511, G)
            }
        );
    }

    #[test]
    fn nonadjacent_layer_permutation_is_out_of_order_not_missing() {
        let mut tensors = all_w3();
        let at = ix(0, 3, D);
        tensors.swap(at, ix(47, 3, D));
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::OutOfOrder {
                index: at,
                expected: id(0, 3, D),
                found: id(47, 3, D)
            }
        );
    }

    #[test]
    fn swapped_projection_is_out_of_order() {
        let mut tensors = all_w3();
        let at = ix(2, 5, G);
        tensors.swap(at, at + 1);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::OutOfOrder {
                index: at,
                expected: id(2, 5, G),
                found: id(2, 5, D)
            }
        );
    }

    #[test]
    fn invalid_and_out_of_range_names() {
        let mut tensors = all_w3();
        tensors[0].name = "model.language_model.layers.01.mlp.experts.0.gate_up_proj".into();
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::InvalidName { index: 0 }
        );

        let mut tensors = all_w3();
        let at = ix(47, 511, D);
        tensors[at].name = expert_entry_name(48, 511, D);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::IndexOutOfRange {
                index: at,
                layer: 48,
                expert: 511
            }
        );

        let mut tensors = all_w3();
        let at = ix(0, 4, G);
        tensors[at].name = expert_entry_name(0, 512, G);
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::IndexOutOfRange {
                index: at,
                layer: 0,
                expert: 512
            }
        );
    }

    #[test]
    fn wrong_shape_or_rank_is_rejected() {
        let mut tensors = all_w3();
        let at = ix(1, 1, G);
        tensors[at].shape = vec![2560, 1280];
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::Shape {
                entry: id(1, 1, G),
                expected: [1280, 2560],
                found: vec![2560, 1280]
            }
        );
        let mut tensors = all_w3();
        let at = ix(1, 1, D);
        tensors[at].shape = vec![1, 2560, 640];
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::Shape {
                entry: id(1, 1, D),
                expected: [2560, 640],
                found: vec![1, 2560, 640]
            }
        );
    }

    #[test]
    fn invalid_qtype_pairs_are_rejected() {
        // gate_up W3-R256 with a W2-R128 down (131/134) and the reverse.
        for (g, d) in [
            (131u8, 134u8),
            (133, 132),
            (132, 131),
            (44, 132),
            (131, 53),
            (0, 0),
        ] {
            let mut tensors = all_w3();
            tensors[ix(0, 0, G)].quant_type = g;
            tensors[ix(0, 0, D)].quant_type = d;
            assert_eq!(
                err(&tensors),
                ExpertSlabTableError::InvalidPair {
                    layer: 0,
                    expert: 0,
                    gate_up_qt: g,
                    down_qt: d
                }
            );
        }
    }

    #[test]
    fn wrong_group_size_is_rejected() {
        let mut tensors = all_w3();
        tensors[ix(0, 2, G)].group_size = 256;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::GroupSize {
                entry: id(0, 2, G),
                quant_type: 131,
                expected: 128,
                found: 256
            }
        );
        let mut tensors = mixed();
        // First MQ4-XTS expert: gate 44 wants 256, down 53 wants 128.
        let (layer, expert) = (0..EXPERTS)
            .map(|e| (0, e))
            .find(|&(l, e)| mixed_format(l, e) == ExpertFormat::Mq4Xts)
            .unwrap();
        tensors[ix(layer, expert, G)].group_size = 128;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::GroupSize {
                entry: id(layer, expert, G),
                quant_type: 44,
                expected: 256,
                found: 128
            }
        );
        let mut tensors = mixed();
        tensors[ix(layer, expert, D)].group_size = 256;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::GroupSize {
                entry: id(layer, expert, D),
                quant_type: 53,
                expected: 128,
                found: 256
            }
        );
    }

    #[test]
    fn wrong_payload_size_is_rejected() {
        let mut tensors = all_w3();
        tensors[ix(0, 3, D)].data_size += 1;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::Size {
                entry: id(0, 3, D),
                expected: 640_000,
                found: 640_001
            }
        );
        let mut tensors = all_w3();
        tensors[ix(0, 3, G)].data_size = 870_400; // W2 size under a W3 qtype
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::Size {
                entry: id(0, 3, G),
                expected: 1_280_000,
                found: 870_400
            }
        );
    }

    #[test]
    fn span_end_overflow_is_a_range_error() {
        let mut tensors = all_w3();
        let at = ix(0, 0, G);
        tensors[at].data_offset = usize::MAX - 5;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::Range {
                entry: id(0, 0, G),
                data_offset: usize::MAX - 5,
                data_size: 1_280_000
            }
        );
    }

    #[test]
    fn gate_down_gap_and_overlap_are_noncontiguous() {
        let mut tensors = all_w3();
        let down = &mut tensors[ix(2, 3, D)];
        down.data_offset += 256;
        let down_offset = down.data_offset;
        let gate_end = tensors[ix(2, 3, G)].data_offset + 1_280_000;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::NoncontiguousGateDown {
                layer: 2,
                expert: 3,
                gate_up_end: gate_end,
                down_offset
            }
        );
        let mut tensors = all_w3();
        tensors[ix(2, 3, D)].data_offset -= 256;
        assert!(matches!(
            err(&tensors),
            ExpertSlabTableError::NoncontiguousGateDown {
                layer: 2,
                expert: 3,
                ..
            }
        ));
    }

    #[test]
    fn slab_gap_and_overlap_are_distinct() {
        let mut tensors = all_w3();
        let mut previous_end = 0;
        for proj in [G, D] {
            tensors[ix(0, 5, proj)].data_offset += 256;
        }
        previous_end += tensors[ix(0, 4, D)].data_offset + 640_000;
        let offset = tensors[ix(0, 5, G)].data_offset;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::SlabGap {
                layer: 0,
                expert: 5,
                previous_end,
                offset
            }
        );

        let mut tensors = all_w3();
        for proj in [G, D] {
            tensors[ix(1, 0, proj)].data_offset -= 256;
        }
        let previous_end = tensors[ix(0, 511, D)].data_offset + 640_000;
        let offset = tensors[ix(1, 0, G)].data_offset;
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::SlabOverlap {
                layer: 1,
                expert: 0,
                previous_end,
                offset
            }
        );
    }

    #[test]
    fn first_slab_must_be_aligned() {
        let mut tensors = all_w3();
        for t in tensors.iter_mut().take(ENTRY_COUNT) {
            t.data_offset += 1;
        }
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::FirstSlabUnaligned {
                offset: 4097,
                alignment: 256
            }
        );
        // 256-aligned but not 4096-aligned is fine.
        let mut tensors = all_w3();
        for t in tensors.iter_mut().take(ENTRY_COUNT) {
            t.data_offset += 256;
        }
        assert!(build(&tensors).is_ok());
    }

    #[test]
    fn trunk_aggregates_are_rejected_but_mtp_is_allowed() {
        // Base entries already carry legacy MTP aggregates and build fine.
        assert!(all_w3()
            .iter()
            .any(|t| t.name.starts_with("mtp.layers.0.mlp.experts.")));
        assert!(build(&all_w3()).is_ok());

        for (layer, proj) in [
            (7usize, "gate_up_proj"),
            (47, "down_proj"),
            (0, "down_proj"),
        ] {
            let mut tensors = all_w3();
            let mut agg = tensors[0].clone();
            agg.name = format!("model.language_model.layers.{layer}.mlp.experts.{proj}");
            agg.quant_type = 44;
            agg.group_size = 256;
            tensors.push(agg);
            assert_eq!(
                err(&tensors),
                ExpertSlabTableError::AggregateTrunkExperts {
                    index: tensors.len() - 1,
                    layer
                }
            );
        }
        // Aggregate in place of the first expert entry.
        let mut tensors = all_w3();
        tensors[0].name = "model.language_model.layers.0.mlp.experts.gate_up_proj".into();
        assert_eq!(
            err(&tensors),
            ExpertSlabTableError::AggregateTrunkExperts { index: 0, layer: 0 }
        );
    }

    #[test]
    fn mq_i4_qtype_outside_slab_entries_is_stray() {
        for qt in 131u8..=134 {
            let mut tensors = all_w3();
            let mut stray = tensors.last().unwrap().clone();
            stray.name = "model.language_model.layers.3.mlp.shared_expert.up_proj".into();
            stray.quant_type = qt;
            tensors.push(stray);
            assert_eq!(
                err(&tensors),
                ExpertSlabTableError::StrayI4Qtype {
                    index: tensors.len() - 1,
                    quant_type: qt
                }
            );
        }
    }

    // ----- scale validation -------------------------------------------------

    fn good_tensor(profile: MqI4Profile, m: usize, k: usize) -> Vec<u8> {
        let levels = usize::from(profile.codec.levels()) + 1;
        let mid = profile.codec.midpoint();
        let mut codes: Vec<u8> = (0..m * k)
            .map(|i| ((i * 5 + i / 128) % levels) as u8)
            .collect();
        let mut scales: Vec<u16> = (0..mq_i4::scale_count(m, k))
            .map(|g| 0x3C00 + (g % 64) as u16)
            .collect();
        // One legitimate zero-scale group: +0 with all-midpoint codes.
        codes[3 * 128..4 * 128].fill(mid);
        scales[3] = 0;
        mq_i4::encode_tensor(profile, m, k, &codes, &scales).unwrap()
    }

    struct OneExpert {
        table: ExpertSlabTable,
        arena: Vec<u8>,
        gate_len: usize,
    }

    impl OneExpert {
        fn new(format: ExpertFormat) -> Self {
            let (gp, dp) = mq_i4_profiles(format).unwrap();
            let [gm, gk] = G.shape().map(|d| d as usize);
            let [dm, dk] = D.shape().map(|d| d as usize);
            let mut arena = good_tensor(gp, gm, gk);
            let gate_len = arena.len();
            arena.extend(good_tensor(dp, dm, dk));
            assert_eq!(gate_len, format.gate_up_bytes());
            assert_eq!(arena.len(), format.slab_bytes());
            Self {
                table: table_of(vec![slab_at(format, 0)]),
                arena,
                gate_len,
            }
        }

        fn validate(&self, arena: &[u8]) -> Result<(), ExpertSlabScaleError> {
            validate_expert_slab_scales(&self.table, |span| {
                arena.get(span.offset..span.offset + span.len)
            })
        }

        /// Set the scale bits of scale `index` in the projection starting at
        /// `base` (`profile`, shape `m x k`).
        fn with_scale(
            &self,
            base: usize,
            codec: MqI4Codec,
            m: usize,
            k: usize,
            index: usize,
            bits: u16,
        ) -> Vec<u8> {
            let mut arena = self.arena.clone();
            let at = base + mq_i4::codes_bytes(codec, m, k) + 2 * index;
            arena[at..at + 2].copy_from_slice(&bits.to_le_bytes());
            arena
        }
    }

    fn slab_at(format: ExpertFormat, offset: usize) -> ExpertSlab {
        ExpertSlab {
            format,
            gate_up: ExpertSlabSpan {
                offset,
                len: format.gate_up_bytes(),
            },
            down: ExpertSlabSpan {
                offset: offset + format.gate_up_bytes(),
                len: format.down_bytes(),
            },
        }
    }

    fn table_of(slabs: Vec<ExpertSlab>) -> ExpertSlabTable {
        let mut counts = ExpertSlabCounts::default();
        for s in &slabs {
            counts.add(s.format);
        }
        ExpertSlabTable { slabs, counts }
    }

    /// Expected `ZeroScaleNonMidpoint` when scale `index` of a fixture tensor
    /// (see [`good_tensor`]) is zeroed: the first non-midpoint code of that group.
    fn zero_scale_error(codec: MqI4Codec, k: usize, index: usize) -> MqI4Error {
        let levels = usize::from(codec.levels()) + 1;
        let midpoint = codec.midpoint();
        let groups_per_row = k / 128;
        let (row, group) = (index / groups_per_row, index % groups_per_row);
        let (off, code) = (0..128)
            .map(|c| {
                let i = row * k + group * 128 + c;
                (c, ((i * 5 + i / 128) % levels) as u8)
            })
            .find(|&(_, v)| v != midpoint)
            .expect("fixture group has non-midpoint codes");
        MqI4Error::ZeroScaleNonMidpoint {
            row,
            group,
            col: group * 128 + off,
            code,
            midpoint,
        }
    }

    fn scale_mutations(format: ExpertFormat) {
        let case = OneExpert::new(format);
        assert_eq!(case.validate(&case.arena), Ok(()));
        let (gp, dp) = mq_i4_profiles(format).unwrap();
        let [gm, gk] = G.shape().map(|d| d as usize);
        let [dm, dk] = D.shape().map(|d| d as usize);
        // Group 11 of each projection holds non-midpoint codes.
        let index = 11;
        for (proj, base, codec, m, k) in [
            (G, 0usize, gp.codec, gm, gk),
            (D, case.gate_len, dp.codec, dm, dk),
        ] {
            let cases: [(u16, MqI4Error); 6] = [
                (
                    0x7E00,
                    MqI4Error::ScaleNotFinite {
                        index,
                        bits: 0x7E00,
                    },
                ),
                (
                    0x7C00,
                    MqI4Error::ScaleNotFinite {
                        index,
                        bits: 0x7C00,
                    },
                ),
                (
                    0x8000,
                    MqI4Error::ScaleNegative {
                        index,
                        bits: 0x8000,
                    },
                ),
                (
                    0xBC00,
                    MqI4Error::ScaleNegative {
                        index,
                        bits: 0xBC00,
                    },
                ),
                (
                    0xFFFF,
                    MqI4Error::ScaleNegative {
                        index,
                        bits: 0xFFFF,
                    },
                ),
                (0x0000, zero_scale_error(codec, k, index)),
            ];
            for (bits, expected) in cases {
                let arena = case.with_scale(base, codec, m, k, index, bits);
                assert_eq!(
                    case.validate(&arena),
                    Err(ExpertSlabScaleError::Codec {
                        layer: 0,
                        expert: 0,
                        projection: proj,
                        source: expected
                    }),
                    "{format:?} {proj:?} bits {bits:#06x}"
                );
            }
        }
    }

    #[test]
    fn w3_scales_good_and_mutated_report_typed_codec_errors() {
        scale_mutations(ExpertFormat::Mq3I4);
    }

    #[test]
    fn w2_scales_good_and_mutated_report_typed_codec_errors() {
        scale_mutations(ExpertFormat::Mq2I4);
    }

    #[test]
    fn negative_zero_is_rejected_even_on_a_legit_zero_group() {
        let case = OneExpert::new(ExpertFormat::Mq3I4);
        let arena = case.with_scale(0, MqI4Codec::W3, 1280, 2560, 3, 0x8000);
        assert_eq!(
            case.validate(&arena),
            Err(ExpertSlabScaleError::Codec {
                layer: 0,
                expert: 0,
                projection: G,
                source: MqI4Error::ScaleNegative {
                    index: 3,
                    bits: 0x8000
                }
            })
        );
    }

    #[test]
    fn missing_or_short_payload_is_typed() {
        let case = OneExpert::new(ExpertFormat::Mq2I4);
        let gate = case.table.slabs()[0].gate_up;
        let down = case.table.slabs()[0].down;
        let arena = &case.arena;
        assert_eq!(
            validate_expert_slab_scales(&case.table, |_| None),
            Err(ExpertSlabScaleError::MissingPayload {
                layer: 0,
                expert: 0,
                projection: G,
                span: gate
            })
        );
        assert_eq!(
            validate_expert_slab_scales(&case.table, |span| {
                (span == gate).then(|| &arena[span.offset..span.offset + span.len])
            }),
            Err(ExpertSlabScaleError::MissingPayload {
                layer: 0,
                expert: 0,
                projection: D,
                span: down
            })
        );
        let err = validate_expert_slab_scales(&case.table, |span| {
            arena.get(span.offset..span.offset + span.len - 2)
        })
        .unwrap_err();
        assert!(matches!(
            err,
            ExpertSlabScaleError::Codec {
                projection: G,
                source: MqI4Error::LengthMismatch { what: "tensor", .. },
                ..
            }
        ));
        assert!(std::error::Error::source(&err).is_some());
        assert!(err.to_string().contains("gate_up_proj"));
    }

    #[test]
    fn mq4_slabs_are_skipped_and_layer_expert_come_from_the_slab_index() {
        let case = OneExpert::new(ExpertFormat::Mq3I4);
        // 515 MQ4 slabs (never queried), then the W3 slab at (layer 1, expert 3).
        let mut slabs = vec![slab_at(ExpertFormat::Mq4Xts, 0); 515];
        slabs.push(slab_at(ExpertFormat::Mq3I4, 0));
        let table = table_of(slabs);
        assert_eq!(
            table.counts(),
            ExpertSlabCounts {
                w3: 1,
                w2: 0,
                e8p: 0,
                mq4: 515
            }
        );

        let mut queried = Vec::new();
        let gate_span = table.slabs()[515].gate_up;
        let arena = case.with_scale(0, MqI4Codec::W3, 1280, 2560, 11, 0x8000);
        let result = validate_expert_slab_scales(&table, |span| {
            queried.push(span);
            arena.get(span.offset..span.offset + span.len)
        });
        assert_eq!(queried, vec![gate_span]);
        assert_eq!(
            result,
            Err(ExpertSlabScaleError::Codec {
                layer: 1,
                expert: 3,
                projection: G,
                source: MqI4Error::ScaleNegative {
                    index: 11,
                    bits: 0x8000
                }
            })
        );
        let mut queried = Vec::new();
        assert_eq!(
            validate_expert_slab_scales(&table, |span| {
                queried.push(span);
                case.arena.get(span.offset..span.offset + span.len)
            }),
            Ok(())
        );
        assert_eq!(
            queried,
            vec![table.slabs()[515].gate_up, table.slabs()[515].down]
        );
    }

    #[test]
    fn mq4_only_table_never_requests_payload() {
        let table = table_of(vec![slab_at(ExpertFormat::Mq4Xts, 0); 3]);
        assert_eq!(
            validate_expert_slab_scales(&table, |_| panic!("MQ4 payload must not be requested")),
            Ok(())
        );
    }
}
