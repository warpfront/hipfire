// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `I4ExpertFinalCodeRecord`: one expert's final MQ3-I4 / MQ2-I4 codes and f16
//! scale bits (DESIGN F2), plus the importer from the R1 calibration NPZ schema.
//!
//! The record holds *final* codes — the packer never re-quantises. Import
//! validates the NPZ identity (`bits`, `grid`, `scale_rule`, unit `sd`), converts
//! the f16-valued F32 scale arrays exactly (no rounding), range-checks codes and
//! then runs the canonical F1 encoder ([`mq_i4::encode_tensor`], which applies
//! the full F1 validation). Every record is decoded again and compared with the
//! imported codes/scales before it is returned; that check cannot be disabled.
//!
//! R1 NPZ schema (one file per expert, `L{L}/E{e}.npz`):
//! `gds f32[Mgu, Kgu/128]`, `gc u8[Mgu, Kgu]`, `dds f32[Md, Kd/128]`,
//! `dc u8[Md, Kd]`, `sd f32[Kd]` (must be exactly `1.0` everywhere — there is no
//! runtime alpha vector), `bits i64`, `grid <U`, `scale_rule <U`.
//!
//! E8P-I4 (qt135/qt136, codec 3) is a separate record kind ([`E8pExpertRecord`]):
//! the existing solved `vq-e8p` records repacked planar, see the E8P section
//! below. It has no codes/`scale_rule`/`grid`: the idx/sign words are carried
//! as solved and every scale goes through `e8p_i4_reference::writer_guard`.

use std::fmt;
use std::path::Path;

use hipfire_arch_qwen4::expert_slabs::{expert_entry_name, ExpertFormat, ExpertProj};
use hipfire_runtime::e8p_i4_reference as e8p;
use hipfire_runtime::mq_i4::{self, MqI4Error, MqI4Profile, GROUP};
use sha2::{Digest, Sha256};

use crate::npz::{NpzArchive, NpzError};

/// The only grid the approved MQ3-I4 / MQ2-I4 wire types may carry. The
/// `signed_27` challenger is calibration-only and is rejected at import.
pub const GRID_UNIFORM: &str = "uniform";

/// Explicit allow-list of recipe identifiers accepted at import.
pub const ALLOWED_SCALE_RULES: [&str; 3] = [
    "mqi4-canonical-v1-rne-tree128",
    "mqi4-recover-s1-v1",
    "mqi4-recover-s2-v1",
];

/// Lower-case hex of `bytes`.
pub fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(DIGITS[usize::from(b >> 4)] as char);
        out.push(DIGITS[usize::from(b & 15)] as char);
    }
    out
}

/// SHA-256 of `bytes`, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

/// `[M, K]` of the two projections of an expert. Tests use the tiny fixture
/// geometry; the FN CLI pins [`I4RecordShapes::FN`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I4RecordShapes {
    pub gate_up: [usize; 2],
    pub down: [usize; 2],
}

impl I4RecordShapes {
    /// Flash-Next: gate_up `[1280, 2560]` (R256), down `[2560, 640]` (R128).
    pub const FN: Self = Self {
        gate_up: [1280, 2560],
        down: [2560, 640],
    };
}

/// Failure importing, validating or encoding a record.
#[derive(Debug)]
pub enum I4RecordError {
    Npz(NpzError),
    Io(String),
    /// `bits` is neither 3 nor 2.
    UnsupportedBits(i64),
    /// The NPZ's `bits` does not match the format the caller expects.
    FormatMismatch {
        expected: ExpertFormat,
        bits: i64,
    },
    /// A record can only hold MQ3-I4 or MQ2-I4, never `Mq4Xts`.
    NotAnI4Format(ExpertFormat),
    GridNotUniform(String),
    ScaleRuleNotAllowed(String),
    /// `sd` has a value that is not exactly `1.0` (no runtime alpha exists).
    NonUnitSd {
        index: usize,
        bits: u32,
    },
    /// A scale is not an exact non-negative finite f16 (no rounding at import).
    Scale {
        projection: &'static str,
        index: usize,
        error: MqI4Error,
    },
    /// A code exceeds `L = 2^b - 1`.
    CodeOutOfRange {
        projection: &'static str,
        row: usize,
        col: usize,
        code: u8,
        max: u8,
    },
    /// F1 shape/length/zero-scale validation failed.
    Codec {
        projection: &'static str,
        error: MqI4Error,
    },
    /// decode(encode(x)) != x.
    RoundTrip {
        projection: &'static str,
    },
}

impl fmt::Display for I4RecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Npz(e) => write!(f, "{e}"),
            Self::Io(e) => write!(f, "record i/o error: {e}"),
            Self::UnsupportedBits(b) => {
                write!(
                    f,
                    "record bits={b}: only 3 (MQ3-I4) and 2 (MQ2-I4) are supported"
                )
            }
            Self::FormatMismatch { expected, bits } => write!(
                f,
                "record bits={bits} does not match the expected format {expected:?}"
            ),
            Self::NotAnI4Format(fmt_) => {
                write!(f, "{fmt_:?} is not an MQ3-I4/MQ2-I4 record format")
            }
            Self::GridNotUniform(g) => write!(
                f,
                "record grid `{g}` rejected: only `{GRID_UNIFORM}` may be emitted under qt131-134"
            ),
            Self::ScaleRuleNotAllowed(r) => {
                write!(
                    f,
                    "record scale_rule `{r}` is not in the allow-list {ALLOWED_SCALE_RULES:?}"
                )
            }
            Self::NonUnitSd { index, bits } => write!(
                f,
                "record sd[{index}] bits {bits:#010x} != 1.0: runtime alpha is not supported"
            ),
            Self::Scale {
                projection,
                index,
                error,
            } => write!(f, "{projection} scale[{index}]: {error}"),
            Self::CodeOutOfRange {
                projection,
                row,
                col,
                code,
                max,
            } => write!(f, "{projection} code {code} at [{row},{col}] exceeds {max}"),
            Self::Codec { projection, error } => write!(f, "{projection}: {error}"),
            Self::RoundTrip { projection } => {
                write!(
                    f,
                    "{projection}: decode(encode(record)) differs from the record"
                )
            }
        }
    }
}

impl std::error::Error for I4RecordError {}

impl From<NpzError> for I4RecordError {
    fn from(e: NpzError) -> Self {
        Self::Npz(e)
    }
}

/// One projection of an expert: final codes, f16 scale bits and the F1 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I4ProjectionRecord {
    /// Slab entry name, e.g. `model.language_model.layers.3.mlp.experts.17.gate_up_proj`.
    pub tensor_name: String,
    /// `[M, K]`.
    pub shape: [usize; 2],
    pub profile: MqI4Profile,
    /// One code per weight, row-major.
    pub codes: Vec<u8>,
    /// f16 bit patterns, `[M, K/128]` row-major.
    pub scale_bits: Vec<u16>,
    /// The encoded tensor (`codes plane ++ f16 scale plane`).
    pub bytes: Vec<u8>,
}

impl I4ProjectionRecord {
    fn encode(
        tensor_name: String,
        projection: &'static str,
        profile: MqI4Profile,
        shape: [usize; 2],
        codes: Vec<u8>,
        scale_bits: Vec<u16>,
    ) -> Result<Self, I4RecordError> {
        let [m, k] = shape;
        let codec = |error| I4RecordError::Codec { projection, error };
        // Shape legality (K % 128, K % R, size overflow) before any indexing.
        mq_i4::tensor_bytes(profile, m, k).map_err(codec)?;
        let max = profile.codec.levels();
        if let Some(pos) = codes.iter().position(|&c| c > max) {
            return Err(I4RecordError::CodeOutOfRange {
                projection,
                row: pos / k,
                col: pos % k,
                code: codes[pos],
                max,
            });
        }
        let bytes = mq_i4::encode_tensor(profile, m, k, &codes, &scale_bits).map_err(codec)?;
        let back = mq_i4::decode_tensor(profile, m, k, &bytes).map_err(codec)?;
        if back.codes != codes || back.scale_bits != scale_bits {
            return Err(I4RecordError::RoundTrip { projection });
        }
        Ok(Self {
            tensor_name,
            shape,
            profile,
            codes,
            scale_bits,
            bytes,
        })
    }
}

/// Final codes of one expert (gate_up + down) for MQ3-I4 or MQ2-I4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I4ExpertFinalCodeRecord {
    pub layer: usize,
    pub expert: usize,
    /// `Mq3I4` or `Mq2I4`; gives the qtype pair (131,132) / (133,134).
    pub format: ExpertFormat,
    pub gate_up: I4ProjectionRecord,
    pub down: I4ProjectionRecord,
    /// Recipe identifier (member of [`ALLOWED_SCALE_RULES`]).
    pub scale_rule: String,
    /// Always [`GRID_UNIFORM`].
    pub grid: String,
    /// SHA-256 (hex) of the source NPZ bytes (empty when built from parts).
    pub source_npz_sha256: String,
}

impl I4ExpertFinalCodeRecord {
    /// Build and fully validate a record from already-decoded codes and f16
    /// scale bits.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        layer: usize,
        expert: usize,
        format: ExpertFormat,
        shapes: I4RecordShapes,
        gate_up: (Vec<u8>, Vec<u16>),
        down: (Vec<u8>, Vec<u16>),
        scale_rule: &str,
        grid: &str,
        source_npz_sha256: String,
    ) -> Result<Self, I4RecordError> {
        if matches!(format, ExpertFormat::Mq4Xts | ExpertFormat::E8pI4) {
            return Err(I4RecordError::NotAnI4Format(format));
        }
        if grid != GRID_UNIFORM {
            return Err(I4RecordError::GridNotUniform(grid.to_string()));
        }
        if !ALLOWED_SCALE_RULES.contains(&scale_rule) {
            return Err(I4RecordError::ScaleRuleNotAllowed(scale_rule.to_string()));
        }
        let (gu_qt, down_qt) = format.pair();
        let gu_profile =
            MqI4Profile::from_quant_type(gu_qt).expect("I4 format has a gate_up profile");
        let down_profile =
            MqI4Profile::from_quant_type(down_qt).expect("I4 format has a down profile");
        let gate_up = I4ProjectionRecord::encode(
            expert_entry_name(layer, expert, ExpertProj::GateUp),
            "gate_up",
            gu_profile,
            shapes.gate_up,
            gate_up.0,
            gate_up.1,
        )?;
        let down = I4ProjectionRecord::encode(
            expert_entry_name(layer, expert, ExpertProj::Down),
            "down",
            down_profile,
            shapes.down,
            down.0,
            down.1,
        )?;
        Ok(Self {
            layer,
            expert,
            format,
            gate_up,
            down,
            scale_rule: scale_rule.to_string(),
            grid: grid.to_string(),
            source_npz_sha256,
        })
    }

    /// Payload bytes of the whole slab (gate_up ++ down).
    pub fn slab_len(&self) -> usize {
        self.gate_up.bytes.len() + self.down.bytes.len()
    }

    /// Import one expert from the bytes of an R1 NPZ. `expected` is the format
    /// the caller's record directory is supposed to hold (`bits` must agree).
    pub fn import_npz_bytes(
        npz_bytes: &[u8],
        layer: usize,
        expert: usize,
        expected: ExpertFormat,
        shapes: I4RecordShapes,
    ) -> Result<Self, I4RecordError> {
        let npz_sha = sha256_hex(npz_bytes);
        let z = NpzArchive::from_bytes(npz_bytes)?;
        let bits = z.i64_scalar("bits")?;
        let format = match bits {
            3 => ExpertFormat::Mq3I4,
            2 => ExpertFormat::Mq2I4,
            other => return Err(I4RecordError::UnsupportedBits(other)),
        };
        if format != expected {
            return Err(I4RecordError::FormatMismatch { expected, bits });
        }
        let grid = z.str_scalar("grid")?;
        let scale_rule = z.str_scalar("scale_rule")?;
        // Check identity before decoding any large array.
        if grid != GRID_UNIFORM {
            return Err(I4RecordError::GridNotUniform(grid.to_string()));
        }
        if !ALLOWED_SCALE_RULES.contains(&scale_rule) {
            return Err(I4RecordError::ScaleRuleNotAllowed(scale_rule.to_string()));
        }
        let [mg, kg] = shapes.gate_up;
        let [md, kd] = shapes.down;
        if kg % GROUP != 0 || kd % GROUP != 0 {
            return Err(I4RecordError::Codec {
                projection: "shape",
                error: MqI4Error::KNotGroupAligned {
                    k: if kg % GROUP != 0 { kg } else { kd },
                },
            });
        }
        let gds = z.f32_array("gds", &[mg, kg / GROUP])?;
        let gc = z.u8_array("gc", &[mg, kg])?;
        let dds = z.f32_array("dds", &[md, kd / GROUP])?;
        let dc = z.u8_array("dc", &[md, kd])?;
        let sd = z.f32_array("sd", &[kd])?;
        if let Some(index) = sd.iter().position(|v| v.to_bits() != 1.0f32.to_bits()) {
            return Err(I4RecordError::NonUnitSd {
                index,
                bits: sd[index].to_bits(),
            });
        }
        let gate_up_scales = scales_to_bits("gate_up", gds)?;
        let down_scales = scales_to_bits("down", dds)?;
        Self::from_parts(
            layer,
            expert,
            format,
            shapes,
            (gc.to_vec(), gate_up_scales),
            (dc.to_vec(), down_scales),
            scale_rule,
            grid,
            npz_sha,
        )
    }

    /// [`Self::import_npz_bytes`] from a file path.
    pub fn import_npz_file(
        path: &Path,
        layer: usize,
        expert: usize,
        expected: ExpertFormat,
        shapes: I4RecordShapes,
    ) -> Result<Self, I4RecordError> {
        let bytes = std::fs::read(path)
            .map_err(|e| I4RecordError::Io(format!("{}: {e}", path.display())))?;
        Self::import_npz_bytes(&bytes, layer, expert, expected, shapes)
    }
}

fn scales_to_bits(projection: &'static str, scales: &[f32]) -> Result<Vec<u16>, I4RecordError> {
    scales
        .iter()
        .enumerate()
        .map(|(index, &v)| {
            mq_i4::scale_bits_from_f16_valued_f32(v).map_err(|error| I4RecordError::Scale {
                projection,
                index,
                error,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// E8P-I4 (qt135 gate_up R256 / qt136 down R128, wire codec 3)
// ---------------------------------------------------------------------------
//
// Records are the *existing solved* `vq-e8p` records (`vq_books.py::Book.payload`,
// `halves == 1`), one per expert projection:
//
//   8-byte header `b"e8p\0\0\0\0\0"` | idx plane `m*k/8` B | signs plane `m*k/8` B |
//   FP16 scales `m*k/128` (little endian)
//
// where signs bits 0..6 are the signs of lanes 0..6 and bit 7 is the E8P `-1/4`
// shift. The slab payload (the PM E8P kernel ABI) is *planar*: `m*k/8` u16 wire
// words `idx | signs << 8` (`e8p_i4_reference::pack_word`) followed by the same
// FP16 scale plane. gate_up `[1280,2560]` = 870 400 B, down `[2560,640]` =
// 435 200 B, 1 305 600 B per expert (2.125 bpw). The idx byte is carried
// unchanged: the codebook is the pinned `vq_books.json`
// (`e8p_i4_reference::E8P_BOOKS_SHA256`), the one `E8P_ABS2` is derived from.
//
// The writer guard (`e8p_i4_reference::writer_guard`: finite FP16 `d >= 2^-14`)
// is applied to every scale; a tripped guard is never reinterpreted, it makes
// the expert an explicit MQ4 fallback at the slab writer (see
// `i4_slabs::E8pRecordProvider::screen`).

/// First 8 bytes of every solved E8P record.
pub const E8P_RECORD_HEADER: [u8; 8] = *b"e8p\0\0\0\0\0";

fn proj_label(proj: ExpertProj) -> &'static str {
    match proj {
        ExpertProj::GateUp => "gate_up",
        ExpertProj::Down => "down",
    }
}

/// Failure building an E8P record from the solved bytes.
#[derive(Debug)]
pub enum E8pRecordError {
    /// `[m, k]` is not legal for the projection (`k % 256` gate_up, `k % 128` down).
    Shape {
        projection: &'static str,
        shape: [usize; 2],
    },
    /// A plane/record has the wrong size.
    Length {
        projection: &'static str,
        what: &'static str,
        expected: usize,
        got: usize,
    },
    Header {
        projection: &'static str,
        got: [u8; 8],
    },
    /// The writer guard refused scale `group` (row-major `[m, k/128]`).
    Guard {
        projection: &'static str,
        group: usize,
        reject: e8p::Reject,
    },
    /// `unpack_word_fields(idx | signs << 8)` does not give the planes back.
    WordInverse {
        projection: &'static str,
        word: usize,
    },
    /// parse(encode(record)) != record.
    RoundTrip { projection: &'static str },
}

impl fmt::Display for E8pRecordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { projection, shape } => write!(
                f,
                "{projection}: shape {shape:?} is not legal for the E8P-I4 wire (K % 256 for gate_up, K % 128 for down)"
            ),
            Self::Length {
                projection,
                what,
                expected,
                got,
            } => write!(f, "{projection}: {what} is {got}, expected {expected}"),
            Self::Header { projection, got } => write!(
                f,
                "{projection}: record header {got:?} != {E8P_RECORD_HEADER:?}"
            ),
            Self::Guard {
                projection,
                group,
                reject,
            } => write!(f, "{projection} scale[{group}]: {reject}"),
            Self::WordInverse { projection, word } => write!(
                f,
                "{projection} word {word}: idx/signs planes do not survive the wire-word inverse"
            ),
            Self::RoundTrip { projection } => write!(
                f,
                "{projection}: planar bytes parsed back differ from the record"
            ),
        }
    }
}

impl std::error::Error for E8pRecordError {}

/// `(wire words, scales)` of a `[m, k]` projection.
fn e8p_dims(proj: ExpertProj, shape: [usize; 2]) -> Result<(usize, usize), E8pRecordError> {
    let [m, k] = shape;
    let width = match proj {
        ExpertProj::GateUp => 256,
        ExpertProj::Down => 128,
    };
    match m.checked_mul(k) {
        Some(n) if m > 0 && k > 0 && k % width == 0 && n <= usize::MAX / 4 => Ok((n / 8, n / 128)),
        _ => Err(E8pRecordError::Shape {
            projection: proj_label(proj),
            shape,
        }),
    }
}

/// Payload bytes of one planar E8P projection (`2 * words + 2 * scales`).
pub fn e8p_blob_bytes(proj: ExpertProj, shape: [usize; 2]) -> Result<usize, E8pRecordError> {
    let (words, scales) = e8p_dims(proj, shape)?;
    Ok(2 * words + 2 * scales)
}

/// Bytes of one solved source record: header + idx + signs + scales, i.e. the
/// planar payload plus the 8-byte header.
pub fn e8p_source_record_bytes(
    proj: ExpertProj,
    shape: [usize; 2],
) -> Result<usize, E8pRecordError> {
    Ok(E8P_RECORD_HEADER.len() + e8p_blob_bytes(proj, shape)?)
}

/// First scale of a little-endian FP16 plane the writer guard refuses.
pub fn e8p_first_guard_reject(scale_plane: &[u8]) -> Option<(usize, e8p::Reject)> {
    scale_plane.chunks_exact(2).enumerate().find_map(|(i, c)| {
        e8p::writer_guard(u16::from_le_bytes([c[0], c[1]]))
            .err()
            .map(|r| (i, r))
    })
}

/// One E8P projection: wire words, FP16 scale bits and the planar slab bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct E8pProjectionRecord {
    /// Slab entry name, e.g. `model.language_model.layers.3.mlp.experts.17.gate_up_proj`.
    pub tensor_name: String,
    /// `[M, K]`.
    pub shape: [usize; 2],
    /// `m*k/8` wire words `idx | sign7 << 8 | shift << 15`, row-major.
    pub words: Vec<u16>,
    /// FP16 scale bit patterns, `[M, K/128]` row-major.
    pub scale_bits: Vec<u16>,
    /// The slab payload (`words` LE plane ++ `scale_bits` LE plane).
    pub bytes: Vec<u8>,
    /// SHA-256 (hex) of the solved source record this was repacked from (empty
    /// when built from parts).
    pub source_record_sha256: String,
    /// Number of words carrying the `-1/4` shift.
    pub shift_words: u64,
}

impl E8pProjectionRecord {
    /// Repack one solved source record. Header, length, word inverse and the
    /// writer guard are all checked; none of it can be disabled.
    pub fn from_source_record(
        tensor_name: String,
        proj: ExpertProj,
        shape: [usize; 2],
        raw: &[u8],
    ) -> Result<Self, E8pRecordError> {
        let projection = proj_label(proj);
        let want = e8p_source_record_bytes(proj, shape)?;
        if raw.len() != want {
            return Err(E8pRecordError::Length {
                projection,
                what: "source record bytes",
                expected: want,
                got: raw.len(),
            });
        }
        let (nb, _) = e8p_dims(proj, shape)?;
        let header: [u8; 8] = raw[..8].try_into().expect("8-byte header");
        if header != E8P_RECORD_HEADER {
            return Err(E8pRecordError::Header {
                projection,
                got: header,
            });
        }
        let idx = &raw[8..8 + nb];
        let signs = &raw[8 + nb..8 + 2 * nb];
        let dbytes = &raw[8 + 2 * nb..];
        let mut words = Vec::with_capacity(nb);
        for i in 0..nb {
            let w = u16::from(idx[i]) | (u16::from(signs[i]) << 8);
            let (ix, s7, sh) = e8p::unpack_word_fields(w);
            if ix != idx[i]
                || (s7 | (u8::from(sh) << 7)) != signs[i]
                || e8p::pack_word(ix, s7, sh) != w
            {
                return Err(E8pRecordError::WordInverse {
                    projection,
                    word: i,
                });
            }
            words.push(w);
        }
        let scale_bits = dbytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Self::from_parts(tensor_name, proj, shape, words, scale_bits, sha256_hex(raw))
    }

    /// Build and fully validate a projection from wire words and scale bits.
    pub fn from_parts(
        tensor_name: String,
        proj: ExpertProj,
        shape: [usize; 2],
        words: Vec<u16>,
        scale_bits: Vec<u16>,
        source_record_sha256: String,
    ) -> Result<Self, E8pRecordError> {
        let projection = proj_label(proj);
        let (nb, ns) = e8p_dims(proj, shape)?;
        for (what, expected, got) in [("words", nb, words.len()), ("scales", ns, scale_bits.len())]
        {
            if expected != got {
                return Err(E8pRecordError::Length {
                    projection,
                    what,
                    expected,
                    got,
                });
            }
        }
        if let Some((group, reject)) = scale_bits
            .iter()
            .enumerate()
            .find_map(|(i, &d)| e8p::writer_guard(d).err().map(|r| (i, r)))
        {
            return Err(E8pRecordError::Guard {
                projection,
                group,
                reject,
            });
        }
        let mut bytes = Vec::with_capacity(2 * (nb + ns));
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        for d in &scale_bits {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        let le16 = |plane: &[u8]| -> Vec<u16> {
            plane
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        if le16(&bytes[..2 * nb]) != words || le16(&bytes[2 * nb..]) != scale_bits {
            return Err(E8pRecordError::RoundTrip { projection });
        }
        let shift_words = words.iter().filter(|&&w| w >> 15 == 1).count() as u64;
        Ok(Self {
            tensor_name,
            shape,
            words,
            scale_bits,
            bytes,
            source_record_sha256,
            shift_words,
        })
    }
}

/// Both projections of one expert in E8P-I4 (qt135 gate_up, qt136 down).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct E8pExpertRecord {
    pub layer: usize,
    pub expert: usize,
    pub gate_up: E8pProjectionRecord,
    pub down: E8pProjectionRecord,
}

impl E8pExpertRecord {
    /// Repack one expert from its two solved source records.
    pub fn from_source_records(
        layer: usize,
        expert: usize,
        shapes: I4RecordShapes,
        gate_up_raw: &[u8],
        down_raw: &[u8],
    ) -> Result<Self, E8pRecordError> {
        Ok(Self {
            layer,
            expert,
            gate_up: E8pProjectionRecord::from_source_record(
                expert_entry_name(layer, expert, ExpertProj::GateUp),
                ExpertProj::GateUp,
                shapes.gate_up,
                gate_up_raw,
            )?,
            down: E8pProjectionRecord::from_source_record(
                expert_entry_name(layer, expert, ExpertProj::Down),
                ExpertProj::Down,
                shapes.down,
                down_raw,
            )?,
        })
    }

    /// Payload bytes of the whole slab (gate_up ++ down).
    pub fn slab_len(&self) -> usize {
        self.gate_up.bytes.len() + self.down.bytes.len()
    }
}

// ---- CPU identity: slab -> E0 decode -> inverse rotation -> BF16 --------------
//
// This is the decode consumer24's `VQStore('vq2-e8p')` performs on the same solved
// record (`vq_books.Book.decode` then `fnemu_weights.transform(q, group, True)`
// then one BF16 RNE), written against the *slab* bytes so a qt135/qt136 slab can
// be compared with it bit for bit. The rotated-domain value
// `(f32(d) * 0.25) * v` equals the Python `codeword * d` exactly (both are an
// exact product of a <= 4-bit integer-quarter and an 11-bit significand).

/// Inverse-rotation group of a projection: 256 for gate_up (R256), 128 for down (R128).
pub const fn e8p_rotation_group(proj: ExpertProj) -> usize {
    match proj {
        ExpertProj::GateUp => 256,
        ExpertProj::Down => 128,
    }
}

/// `fnemu_weights.transform` sign vectors of `group`: `(s1, s2 * f32(1/sqrt(group)))`.
/// `s1` is the LCG stream seeded 42 (group 256) / 43 (group 128), `s2` seeded 1042 / 1043.
fn rotation_signs(group: usize) -> (Vec<f32>, Vec<f32>) {
    let lcg = |mut seed: u32| -> Vec<f32> {
        (0..group)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
                if (seed >> 16) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect()
    };
    let base = if group == 256 { 42 } else { 43 };
    let scale = (1.0f64 / (group as f64).sqrt()) as f32;
    (
        lcg(base),
        lcg(base + 1000).into_iter().map(|s| s * scale).collect(),
    )
}

/// In-place inverse rotation of every `group`-wide chunk of `x`, with the exact f32
/// operation order of `fnemu_weights.transform(x, group, inverse=True)`: multiply by
/// `s2`, Walsh-Hadamard butterflies `(a + b, a - b)` for `h = 1, 2, 4, ..`, multiply by `s1`.
pub fn e8p_inverse_rotate(x: &mut [f32], group: usize) {
    assert!(
        group == 128 || group == 256,
        "e8p_inverse_rotate: group {group}"
    );
    assert!(
        x.len() % group == 0,
        "e8p_inverse_rotate: length {} % {group}",
        x.len()
    );
    let (s1, s2) = rotation_signs(group);
    for chunk in x.chunks_exact_mut(group) {
        for (v, s) in chunk.iter_mut().zip(&s2) {
            *v *= *s;
        }
        let mut h = 1;
        while h < group {
            for base in (0..group).step_by(2 * h) {
                for j in base..base + h {
                    let (a, b) = (chunk[j], chunk[j + h]);
                    chunk[j] = a + b;
                    chunk[j + h] = a - b;
                }
            }
            h *= 2;
        }
        for (v, s) in chunk.iter_mut().zip(&s1) {
            *v *= *s;
        }
    }
}

/// Rotated-domain F32 weights `(f32(d) * 0.25) * v` (row-major `[m, k]`) of a planar
/// E8P slab payload. Shape/length are checked; no guard is re-evaluated here.
pub fn e8p_slab_weights_f32(
    proj: ExpertProj,
    shape: [usize; 2],
    slab: &[u8],
) -> Result<Vec<f32>, E8pRecordError> {
    let (nb, ns) = e8p_dims(proj, shape)?;
    if slab.len() != 2 * (nb + ns) {
        return Err(E8pRecordError::Length {
            projection: proj_label(proj),
            what: "slab bytes",
            expected: 2 * (nb + ns),
            got: slab.len(),
        });
    }
    let le16 = |p: &[u8]| -> Vec<u16> {
        p.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let (words, scales) = (le16(&slab[..2 * nb]), le16(&slab[2 * nb..]));
    Ok(e8p::decode_weights_f32(shape[0], shape[1], &words, &scales))
}

/// BF16 bits (row-major `[m, k]`) of a planar E8P slab payload: E0 decode, inverse
/// rotation of [`e8p_rotation_group`], one RNE BF16 rounding.
pub fn e8p_slab_decode_bf16(
    proj: ExpertProj,
    shape: [usize; 2],
    slab: &[u8],
) -> Result<Vec<u16>, E8pRecordError> {
    let mut w = e8p_slab_weights_f32(proj, shape, slab)?;
    e8p_inverse_rotate(&mut w, e8p_rotation_group(proj));
    Ok(w.into_iter()
        .map(|v| half::bf16::from_f32(v).to_bits())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::testutil::build_npz;
    use crate::npz::{NpyArray, NpyData};
    use std::collections::BTreeMap;

    const W3: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w3.npz");
    const W2: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w2.npz");
    const W3_EXPECTED: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w3.expected.bin");
    const W2_EXPECTED: &[u8] = include_bytes!("../tests/fixtures/subq4/tiny_w2.expected.bin");

    const TINY: I4RecordShapes = I4RecordShapes {
        gate_up: [4, 256],
        down: [8, 128],
    };

    fn import(
        bytes: &[u8],
        expected: ExpertFormat,
    ) -> Result<I4ExpertFinalCodeRecord, I4RecordError> {
        I4ExpertFinalCodeRecord::import_npz_bytes(bytes, 5, 9, expected, TINY)
    }

    /// Independent restatement of the Python transform for one group: the Sylvester
    /// Hadamard recursion `H_n = [[H, H], [H, -H]]` instead of the in-place stage loop.
    fn rotate_group_by_recursion(x: &[f32], s1: &[f32], s2: &[f32]) -> Vec<f32> {
        fn wht(v: &[f32]) -> Vec<f32> {
            if v.len() == 1 {
                return v.to_vec();
            }
            let h = v.len() / 2;
            let sum: Vec<f32> = (0..h).map(|i| v[i] + v[i + h]).collect();
            let diff: Vec<f32> = (0..h).map(|i| v[i] - v[i + h]).collect();
            let mut out = wht(&sum);
            out.extend(wht(&diff));
            out
        }
        let y: Vec<f32> = x.iter().zip(s2).map(|(a, s)| a * s).collect();
        wht(&y).iter().zip(s1).map(|(a, s)| a * s).collect()
    }

    #[test]
    fn e8p_inverse_rotation_matches_the_python_transform_definition() {
        for group in [128usize, 256] {
            let (s1, s2) = rotation_signs(group);
            assert!(s1.iter().all(|s| s.abs() == 1.0));
            let scale = (1.0f64 / (group as f64).sqrt()) as f32;
            assert!(s2.iter().all(|s| s.abs() == scale));
            // Different butterfly association: the results agree to f32 rounding.
            let mut x: Vec<f32> = (0..2 * group)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.0625)
                .collect();
            let want: Vec<f32> = x
                .chunks(group)
                .flat_map(|c| rotate_group_by_recursion(c, &s1, &s2))
                .collect();
            e8p_inverse_rotate(&mut x, group);
            // Same transform, different rounding order: agreement to far below one BF16 ulp.
            for (a, b) in x.iter().zip(&want) {
                assert!((a - b).abs() <= 1e-4, "group {group}: {a} vs {b}");
            }
        }
        // Rotation is (up to f32 rounding) orthogonal: it preserves the L2 norm.
        let mut v: Vec<f32> = (0..256).map(|i| (i as f32 * 0.37).sin()).collect();
        let n0: f32 = v.iter().map(|a| a * a).sum();
        e8p_inverse_rotate(&mut v, 256);
        let n1: f32 = v.iter().map(|a| a * a).sum();
        assert!((n0 - n1).abs() < 1e-3 * n0);
    }

    #[test]
    fn e8p_slab_decode_is_e0_weights_rotated_and_rounded_once() {
        let (m, k) = (8usize, 256usize);
        let words: Vec<u16> = (0..m * k / 8)
            .map(|i| e8p::pack_word((i * 5) as u8, ((i * 11) % 128) as u8, i % 2 == 1))
            .collect();
        let scales = e8p_legal_scales(m * k / 128);
        let raw = e8p_source_record(&words, &scales);
        let rec =
            E8pProjectionRecord::from_source_record("t".into(), ExpertProj::GateUp, [m, k], &raw)
                .unwrap();
        let f32w = e8p_slab_weights_f32(ExpertProj::GateUp, [m, k], &rec.bytes).unwrap();
        assert_eq!(f32w, e8p::decode_weights_f32(m, k, &words, &scales));
        let mut rot = f32w.clone();
        e8p_inverse_rotate(&mut rot, 256);
        let bf = e8p_slab_decode_bf16(ExpertProj::GateUp, [m, k], &rec.bytes).unwrap();
        assert_eq!(bf.len(), m * k);
        for (b, r) in bf.iter().zip(&rot) {
            assert_eq!(*b, half::bf16::from_f32(*r).to_bits());
        }
        // Wrong slab length / shape are rejected.
        assert!(e8p_slab_decode_bf16(ExpertProj::GateUp, [m, k], &rec.bytes[1..]).is_err());
        assert!(e8p_slab_decode_bf16(ExpertProj::GateUp, [m, 128], &rec.bytes).is_err());
    }

    fn mutated(bytes: &[u8], f: impl FnOnce(&mut BTreeMap<String, NpyArray>)) -> Vec<u8> {
        let mut arrays = NpzArchive::from_bytes(bytes).unwrap().into_arrays();
        f(&mut arrays);
        build_npz(&arrays)
    }

    fn f32s(a: &mut NpyArray) -> &mut Vec<f32> {
        match &mut a.data {
            NpyData::F32(v) => v,
            _ => panic!("not f32"),
        }
    }

    fn u8s(a: &mut NpyArray) -> &mut Vec<u8> {
        match &mut a.data {
            NpyData::U8(v) => v,
            _ => panic!("not u8"),
        }
    }

    #[test]
    fn fixtures_encode_to_the_independent_python_packer_bytes() {
        for (npz, expected, format, npz_sha) in [
            (
                W3,
                W3_EXPECTED,
                ExpertFormat::Mq3I4,
                "9917fec8f16447a0989649992e5eb94579e7d96fbbd3f5eedcc8270dd77ce1be",
            ),
            (
                W2,
                W2_EXPECTED,
                ExpertFormat::Mq2I4,
                "4fc8200ff3fafb4d73287b927a60b98f7a6ebde7beaf069f153bb622a778ee60",
            ),
        ] {
            let rec = import(npz, format).unwrap();
            assert_eq!(rec.source_npz_sha256, npz_sha);
            assert_eq!((rec.layer, rec.expert, rec.format), (5, 9, format));
            assert_eq!(rec.grid, "uniform");
            assert_eq!(rec.scale_rule, "mqi4-canonical-v1-rne-tree128");
            let mut got = rec.gate_up.bytes.clone();
            got.extend_from_slice(&rec.down.bytes);
            assert_eq!(got.len(), expected.len());
            assert_eq!(
                got, expected,
                "{format:?} bytes differ from the Python packer"
            );
            assert_eq!(rec.slab_len(), expected.len());
            assert_eq!(
                rec.gate_up.tensor_name,
                "model.language_model.layers.5.mlp.experts.9.gate_up_proj"
            );
            assert_eq!(
                rec.down.tensor_name,
                "model.language_model.layers.5.mlp.experts.9.down_proj"
            );
        }
        // README.json pins the expected-bin hashes.
        assert_eq!(
            sha256_hex(W3_EXPECTED),
            "83c69ed3929991c6ca08492dd7ffaa28cc0c74149d8d661a64702b3c7a2b8221"
        );
        assert_eq!(
            sha256_hex(W2_EXPECTED),
            "cc347c8feb98e99b94ce864896a4ff39e1b32ce15beb9009ef7f6b823c535502"
        );
    }

    #[test]
    fn decode_round_trips_to_the_npz_codes_and_scale_bits() {
        for (npz, format) in [(W3, ExpertFormat::Mq3I4), (W2, ExpertFormat::Mq2I4)] {
            let z = NpzArchive::from_bytes(npz).unwrap();
            let rec = import(npz, format).unwrap();
            for (proj, gk, dk, shape) in [
                (&rec.gate_up, "gc", "gds", [4usize, 256]),
                (&rec.down, "dc", "dds", [8usize, 128]),
            ] {
                let dec =
                    mq_i4::decode_tensor(proj.profile, shape[0], shape[1], &proj.bytes).unwrap();
                assert_eq!(dec.codes, z.u8_array(gk, &shape).unwrap());
                let want: Vec<u16> = z
                    .f32_array(dk, &[shape[0], shape[1] / 128])
                    .unwrap()
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect();
                assert_eq!(dec.scale_bits, want);
                assert_eq!(proj.codes, dec.codes);
                assert_eq!(proj.scale_bits, dec.scale_bits);
            }
        }
    }

    #[test]
    fn profiles_follow_the_format_pair() {
        let r = import(W3, ExpertFormat::Mq3I4).unwrap();
        assert_eq!(r.gate_up.profile, MqI4Profile::MQ3_R256);
        assert_eq!(r.down.profile, MqI4Profile::MQ3_R128);
        let r = import(W2, ExpertFormat::Mq2I4).unwrap();
        assert_eq!(r.gate_up.profile, MqI4Profile::MQ2_R256);
        assert_eq!(r.down.profile, MqI4Profile::MQ2_R128);
    }

    #[test]
    fn format_mismatch_and_foreign_bits_are_rejected() {
        assert!(matches!(
            import(W3, ExpertFormat::Mq2I4),
            Err(I4RecordError::FormatMismatch { .. })
        ));
        assert!(matches!(
            import(W2, ExpertFormat::Mq4Xts),
            Err(I4RecordError::FormatMismatch { .. })
        ));
        let bad = mutated(W3, |a| {
            a.get_mut("bits").unwrap().data = NpyData::I64(vec![4])
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::UnsupportedBits(4))
        ));
    }

    #[test]
    fn non_unit_sd_is_rejected() {
        for v in [0.5f32, 1.000_000_1, -1.0, 0.0, f32::NAN] {
            let bad = mutated(W3, |a| f32s(a.get_mut("sd").unwrap())[5] = v);
            assert!(
                matches!(
                    import(&bad, ExpertFormat::Mq3I4),
                    Err(I4RecordError::NonUnitSd { index: 5, .. })
                ),
                "sd={v}"
            );
        }
    }

    #[test]
    fn grid_signed_27_and_unknown_scale_rules_are_rejected() {
        let bad = mutated(W2, |a| {
            a.get_mut("grid").unwrap().data = NpyData::Str(vec!["signed_27".into()])
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq2I4),
            Err(I4RecordError::GridNotUniform(g)) if g == "signed_27"
        ));
        for rule in ["mqi4-legacy-rtz", "", "MQI4-CANONICAL-V1-RNE-TREE128"] {
            let bad = mutated(W3, |a| {
                a.get_mut("scale_rule").unwrap().data = NpyData::Str(vec![rule.into()])
            });
            assert!(
                matches!(
                    import(&bad, ExpertFormat::Mq3I4),
                    Err(I4RecordError::ScaleRuleNotAllowed(_))
                ),
                "rule {rule:?}"
            );
        }
    }

    #[test]
    fn every_allow_listed_scale_rule_imports() {
        for rule in ALLOWED_SCALE_RULES {
            let ok = mutated(W3, |a| {
                a.get_mut("scale_rule").unwrap().data = NpyData::Str(vec![rule.into()])
            });
            let rec = import(&ok, ExpertFormat::Mq3I4).unwrap();
            assert_eq!(rec.scale_rule, rule);
            assert_eq!(
                [rec.gate_up.bytes, rec.down.bytes].concat(),
                W3_EXPECTED,
                "the recipe label does not change the bytes"
            );
        }
    }

    #[test]
    fn code_above_l_is_rejected() {
        let bad = mutated(W3, |a| u8s(a.get_mut("gc").unwrap())[0] = 8);
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::CodeOutOfRange {
                projection: "gate_up",
                row: 0,
                col: 0,
                code: 8,
                max: 7
            })
        ));
        let bad = mutated(W2, |a| u8s(a.get_mut("dc").unwrap())[130] = 4);
        assert!(matches!(
            import(&bad, ExpertFormat::Mq2I4),
            Err(I4RecordError::CodeOutOfRange {
                projection: "down",
                row: 1,
                col: 2,
                code: 4,
                max: 3
            })
        ));
        // W3's 4 is legal for W3 but the same byte is illegal for W2 (> L = 3).
        let bad = mutated(W2, |a| u8s(a.get_mut("gc").unwrap())[7] = 4);
        assert!(import(&bad, ExpertFormat::Mq2I4).is_err());
    }

    #[test]
    fn non_f16_exact_negative_and_nonfinite_scales_are_rejected() {
        for (v, _label) in [
            (0.1f32, "not f16-exact"),
            (-0.5f32, "negative"),
            (-0.0f32, "negative zero"),
            (f32::INFINITY, "inf"),
            (f32::NAN, "nan"),
            (65520.0f32, "above f16 max"),
        ] {
            let bad = mutated(W3, |a| f32s(a.get_mut("gds").unwrap())[1] = v);
            assert!(
                matches!(
                    import(&bad, ExpertFormat::Mq3I4),
                    Err(I4RecordError::Scale {
                        projection: "gate_up",
                        index: 1,
                        ..
                    })
                ),
                "scale {v}"
            );
        }
        let bad = mutated(W2, |a| f32s(a.get_mut("dds").unwrap())[3] = 0.1);
        assert!(matches!(
            import(&bad, ExpertFormat::Mq2I4),
            Err(I4RecordError::Scale {
                projection: "down",
                index: 3,
                ..
            })
        ));
    }

    #[test]
    fn zero_scale_group_with_non_midpoint_code_is_rejected() {
        // gate_up row 1 group 1 (scale +0): columns 128..256 of row 1.
        for (npz, format, mid) in [
            (W3, ExpertFormat::Mq3I4, 4u8),
            (W2, ExpertFormat::Mq2I4, 2u8),
        ] {
            let z = NpzArchive::from_bytes(npz).unwrap();
            let gc = z.u8_array("gc", &[4, 256]).unwrap();
            assert!(gc[256 + 128..256 + 256].iter().all(|&c| c == mid));
            let bad = mutated(npz, |a| u8s(a.get_mut("gc").unwrap())[256 + 130] = mid - 1);
            assert!(matches!(
                import(&bad, format),
                Err(I4RecordError::Codec {
                    projection: "gate_up",
                    error: MqI4Error::ZeroScaleNonMidpoint {
                        row: 1,
                        group: 1,
                        ..
                    }
                })
            ));
        }
    }

    #[test]
    fn missing_and_misshapen_members_are_rejected() {
        let bad = mutated(W3, |a| {
            a.remove("sd");
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::Npz(NpzError::MissingEntry(_)))
        ));
        let bad = mutated(W3, |a| {
            let sd = a.get_mut("sd").unwrap();
            f32s(sd).pop();
            sd.shape = vec![127];
        });
        assert!(matches!(
            import(&bad, ExpertFormat::Mq3I4),
            Err(I4RecordError::Npz(NpzError::ShapeMismatch { .. }))
        ));
        // Wrong geometry parameters are caught by the shape check, too.
        let wrong = I4RecordShapes {
            gate_up: [4, 256],
            down: [8, 256],
        };
        assert!(
            I4ExpertFinalCodeRecord::import_npz_bytes(W3, 0, 0, ExpertFormat::Mq3I4, wrong)
                .is_err()
        );
    }

    #[test]
    fn from_parts_rejects_mq4_format_and_validates_directly() {
        let shapes = TINY;
        let ok = I4ExpertFinalCodeRecord::from_parts(
            0,
            0,
            ExpertFormat::Mq2I4,
            shapes,
            (vec![2; 4 * 256], vec![0x3C00; 8]),
            (vec![1; 8 * 128], vec![0x3C00; 8]),
            "mqi4-recover-s2-v1",
            "uniform",
            String::new(),
        );
        assert!(ok.is_ok());
        let err = I4ExpertFinalCodeRecord::from_parts(
            0,
            0,
            ExpertFormat::Mq4Xts,
            shapes,
            (vec![], vec![]),
            (vec![], vec![]),
            "mqi4-recover-s2-v1",
            "uniform",
            String::new(),
        );
        assert!(matches!(err, Err(I4RecordError::NotAnI4Format(_))));
    }

    // ---- E8P-I4 records ----------------------------------------------------

    /// Source record bytes (`vq_books.py::Book.payload` layout) of `words`/`scales`.
    fn e8p_source_record(words: &[u16], scales: &[u16]) -> Vec<u8> {
        let mut raw = E8P_RECORD_HEADER.to_vec();
        raw.extend(words.iter().map(|w| (w & 0xFF) as u8));
        raw.extend(words.iter().map(|w| (w >> 8) as u8));
        for d in scales {
            raw.extend_from_slice(&d.to_le_bytes());
        }
        raw
    }

    /// All 2^16 legal wire words: every idx row, every sign7 pattern, both shifts.
    fn e8p_all_words() -> Vec<u16> {
        let mut v = Vec::with_capacity(1 << 16);
        for shift in [false, true] {
            for s7 in 0..128u8 {
                for idx in 0..=255u8 {
                    v.push(e8p::pack_word(idx, s7, shift));
                }
            }
        }
        v
    }

    /// Guard-legal FP16 scales, led by the low-normal `2^-14` (0x0400) and the
    /// largest finite value (0x7BFF).
    fn e8p_legal_scales(n: usize) -> Vec<u16> {
        let mut s: Vec<u16> = (0..n)
            .map(|i| 0x0400 + ((i * 7919) % (0x7C00 - 0x0400)) as u16)
            .collect();
        s[..4].copy_from_slice(&[0x0400, 0x0401, 0x3C00, 0x7BFF]);
        s
    }

    #[test]
    fn e8p_record_decodes_every_wire_word_like_the_k_e0_reference() {
        use half::f16;
        let (m, k) = (2048usize, 256usize);
        let words = e8p_all_words();
        assert_eq!(words.len(), m * k / 8);
        let scales = e8p_legal_scales(m * k / 128);
        let raw = e8p_source_record(&words, &scales);
        let rec =
            E8pProjectionRecord::from_source_record("t".into(), ExpertProj::GateUp, [m, k], &raw)
                .unwrap();
        assert_eq!(rec.words, words);
        assert_eq!(rec.scale_bits, scales);
        assert_eq!(rec.shift_words, 1 << 15);
        assert_eq!(rec.source_record_sha256, sha256_hex(&raw));
        assert_eq!(
            rec.bytes.len(),
            e8p_blob_bytes(ExpertProj::GateUp, [m, k]).unwrap()
        );
        assert_eq!(raw.len(), rec.bytes.len() + 8);

        // Everything below uses the slab bytes only.
        let nb = words.len();
        let le16 = |p: &[u8]| -> Vec<u16> {
            p.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        let slab_words = le16(&rec.bytes[..2 * nb]);
        let slab_scales = le16(&rec.bytes[2 * nb..]);
        assert_eq!(slab_words, words);
        assert_eq!(slab_scales, scales);
        let q = e8p::decode_weights_q(m, k, &slab_words);
        let w = e8p::decode_weights_f32(m, k, &slab_words, &slab_scales);
        for (i, &word) in slab_words.iter().enumerate() {
            let (idx, s7, sh) = e8p::unpack_word_fields(word);
            let v = e8p::decode8(idx, s7, sh);
            let nib = e8p::unpack_word_ref(idx, s7, sh);
            for l in 0..8 {
                let flat = 8 * i + l;
                assert_eq!(q[flat], v[l], "word {i} ({word:#06x}) lane {l}");
                // Operand nibble o = (v - 1) / 2, two's complement.
                let o = (((nib >> (4 * l)) & 0xF) as i8) << 4 >> 4;
                assert_eq!(2 * o + 1, v[l], "word {i} lane {l} nibble");
                assert!(
                    v[l] % 2 != 0 && v[l].abs() <= 11,
                    "word {i} lane {l} v={}",
                    v[l]
                );
                let want = (f16::from_bits(scales[flat / 128]).to_f32() * 0.25) * f32::from(v[l]);
                assert_eq!(w[flat].to_bits(), want.to_bits(), "word {i} lane {l}");
            }
        }
    }

    #[test]
    fn e8p_writer_guard_trips_are_exact_and_the_low_normal_is_accepted() {
        let shape = [4usize, 256];
        let words = vec![0x1234u16 & 0x7FFF; 4 * 256 / 8];
        let good = vec![0x3C00u16; 4 * 256 / 128];
        let build = |scales: Vec<u16>| {
            E8pProjectionRecord::from_source_record(
                "t".into(),
                ExpertProj::GateUp,
                shape,
                &e8p_source_record(&words, &scales),
            )
        };
        assert!(build(good.clone()).is_ok());
        for ok in [0x0400u16, 0x0401, 0x7BFF] {
            let mut s = good.clone();
            s[5] = ok;
            assert!(build(s).is_ok(), "d={ok:#06x} must pass the guard");
        }
        for (bad, nonfinite) in [
            (0x0000u16, false),
            (0x0001, false),
            (0x03FF, false),
            (0x8000, false),
            (0x8400, false),
            (0x7C00, true),
            (0x7E00, true),
            (0xFC00, true),
        ] {
            let mut s = good.clone();
            s[5] = bad;
            match build(s.clone()) {
                Err(E8pRecordError::Guard {
                    group: 5, reject, ..
                }) => {
                    let want = if nonfinite {
                        e8p::Reject::NonFinite { d: bad }
                    } else {
                        e8p::Reject::BelowMin { d: bad }
                    };
                    assert_eq!(reject, want, "d={bad:#06x}");
                }
                other => panic!("d={bad:#06x}: {other:?}"),
            }
            let plane: Vec<u8> = s.iter().flat_map(|d| d.to_le_bytes()).collect();
            assert_eq!(
                e8p_first_guard_reject(&plane).map(|(g, _)| g),
                Some(5),
                "d={bad:#06x}"
            );
        }
        let plane: Vec<u8> = good.iter().flat_map(|d| d.to_le_bytes()).collect();
        assert_eq!(e8p_first_guard_reject(&plane), None);
    }

    #[test]
    fn e8p_header_length_and_shape_are_rejected() {
        let (m, k) = (4usize, 256usize);
        let words: Vec<u16> = (0..m * k / 8)
            .map(|i| e8p::pack_word(i as u8, (i % 128) as u8, i % 3 == 0))
            .collect();
        let scales = vec![0x3C00u16; m * k / 128];
        let raw = e8p_source_record(&words, &scales);
        let go = |b: &[u8], shape: [usize; 2]| {
            E8pProjectionRecord::from_source_record("t".into(), ExpertProj::GateUp, shape, b)
        };
        assert!(go(&raw, [m, k]).is_ok());
        let mut bad = raw.clone();
        bad[..3].copy_from_slice(b"e8h");
        assert!(matches!(
            go(&bad, [m, k]),
            Err(E8pRecordError::Header { .. })
        ));
        assert!(matches!(
            go(&raw[..raw.len() - 1], [m, k]),
            Err(E8pRecordError::Length { .. })
        ));
        let mut long = raw.clone();
        long.push(0);
        assert!(matches!(
            go(&long, [m, k]),
            Err(E8pRecordError::Length { .. })
        ));
        // gate_up is R256: K = 128 is not legal; K = 0 / M = 0 never are.
        assert!(matches!(
            go(&raw, [8, 128]),
            Err(E8pRecordError::Shape { .. })
        ));
        assert!(matches!(
            go(&raw, [0, 256]),
            Err(E8pRecordError::Shape { .. })
        ));
        // down is R128.
        let dw = vec![0u16; 8 * 128 / 8];
        let ds = vec![0x3C00u16; 8];
        let draw = e8p_source_record(&dw, &ds);
        assert!(E8pProjectionRecord::from_source_record(
            "t".into(),
            ExpertProj::Down,
            [8, 128],
            &draw
        )
        .is_ok());
    }

    #[test]
    fn e8p_flash_next_record_sizes_are_the_published_contract() {
        let s = I4RecordShapes::FN;
        assert_eq!(
            e8p_blob_bytes(ExpertProj::GateUp, s.gate_up).unwrap(),
            870_400
        );
        assert_eq!(e8p_blob_bytes(ExpertProj::Down, s.down).unwrap(), 435_200);
        assert_eq!(
            e8p_source_record_bytes(ExpertProj::GateUp, s.gate_up).unwrap(),
            870_408
        );
        assert_eq!(
            e8p_source_record_bytes(ExpertProj::Down, s.down).unwrap(),
            435_208
        );
        assert_eq!(870_400 + 435_200, 1_305_600);
        // 2.125 bits per weight over gate_up + down.
        assert_eq!(1_305_600 * 8 * 8, (1280 * 2560 + 2560 * 640) * 17);
        assert_eq!(ExpertFormat::E8pI4.pair(), (135, 136));
        assert_eq!(ExpertFormat::E8pI4.gate_up_bytes(), 870_400);
        assert_eq!(ExpertFormat::E8pI4.down_bytes(), 435_200);
    }

    #[test]
    fn e8p_expert_record_repacks_both_projections() {
        let shapes = TINY; // gate_up [4,256] R256, down [8,128] R128
        let mk = |m: usize, k: usize, salt: u16| {
            let words: Vec<u16> = (0..m * k / 8)
                .map(|i| {
                    e8p::pack_word(
                        (i as u16).wrapping_mul(salt) as u8,
                        (i % 128) as u8,
                        i % 5 == 0,
                    )
                })
                .collect();
            let scales = e8p_legal_scales(m * k / 128);
            e8p_source_record(&words, &scales)
        };
        let (g, d) = (mk(4, 256, 7), mk(8, 128, 11));
        let rec = E8pExpertRecord::from_source_records(5, 9, shapes, &g, &d).unwrap();
        assert_eq!(
            rec.gate_up.tensor_name,
            "model.language_model.layers.5.mlp.experts.9.gate_up_proj"
        );
        assert_eq!(
            rec.down.tensor_name,
            "model.language_model.layers.5.mlp.experts.9.down_proj"
        );
        assert_eq!(rec.slab_len(), g.len() - 8 + d.len() - 8);
        assert_eq!(rec.gate_up.source_record_sha256, sha256_hex(&g));
        assert_eq!(rec.down.source_record_sha256, sha256_hex(&d));
    }
}
