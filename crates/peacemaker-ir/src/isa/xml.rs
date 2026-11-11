// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Runtime model of the pinned AMD RDNA XML encoding catalogs
//! (`isa/{gfx1100,gfx1151,gfx12}.xml.jsonl`) and the SOP1 predicate evidence
//! (`isa/sop1-overrides.json`).
//!
//! The catalogs are deserialised once per architecture into borrowed
//! `&'static str` names (the text comes from `include_str!`; no allocation per
//! name, no `unsafe`, no leaking). Nothing is silently repaired:
//!
//! * a row key is `(name, encoding, condition, opcode, variant)`; `variant` is
//!   the instruction-local XML encoding ordinal, so the X and Y halves of VOPD
//!   and the literal/DPP siblings keep distinct keys;
//! * encoding bit ranges are listed least-significant chunk first
//!   (`Order="0"` first; `OP_SEL_HI`/`OPSEL_HI` = `[[59,2],[14,1]]` is bits 59-60
//!   then bit 14, confirmed with `llvm-mc`: `op_sel_hi[0]` sets bit 59);
//! * an XML condition that is absent (`null`, e.g. the dangling
//!   `Nothas_lit_0_Nothas_lit_1` SOP1 reference) or uses an unsupported
//!   construct never matches. The only way such a row becomes selectable is
//!   evidence from `sop1-overrides.json` (`Resolution::Corrected`), which also
//!   constrains the sibling `SOP1_INST_LITERAL` record of the same instruction;
//!   evidence that says "unavailable" turns into a named
//!   [`Refusal::NoCorrection`]. [`Catalog::validate_sop1_source`] applies the
//!   same per-instruction domain to callers that do not go through `select`;
//! * graphics encodings stay in the catalog and are reported by name
//!   ([`Refusal::UnsupportedGraphics`], [`SelectError::UnsupportedGraphics`]);
//! * selection never takes the first of several candidate rows. Several live
//!   rows are an explicit [`SelectError::Ambiguous`] unless every non-key field
//!   is identical ([`Row::equivalent_to`]), in which case the lowest `variant`
//!   is canonical and the other keys stay reachable through
//!   [`Catalog::equivalents`];
//! * the success paths of [`Catalog::select`] and [`Catalog::encode_row`] do
//!   not allocate: candidates are folded into stack accumulators and the
//!   detailed error vectors are built by a second pass that only runs when
//!   selection fails.
//!
//! ## XML defect: VOPD identifier overflow
//!
//! `VOPDXY`/`VOPDXY_INST_LITERAL` list an identifier for every opcode in *both*
//! the `OPX` (4 bits) and the `OPY` (5 bits) placement. Y-only opcodes 16, 17
//! and 18 do not fit `OPX`, so their X-placement `id + o * (2^OPX.offset -
//! 2^OPY.offset)` carries into the `ENCODING` field and produces
//! `0xcc000000`, `0xcc400000`, `0xcc800000`: patterns that belong to VOP3P.
//! `llvm-mc --mcpu=gfx1201 --disassemble` decodes first dword `0xcc000000` as
//! `v_pk_mad_i16` and rejects `0xcc400000`/`0xcc800000` as invalid encodings,
//! while a genuine VOPD first dword (`0xc8c00000`) decodes as `v_dual_*`.
//! The structural rule applied ([`Encoding::quarantined`]) is exactly this: for
//! a pair encoding, for every row whose opcode `o` does not fit the *other*
//! slot's opcode field, the identifier obtained by relocating the unique
//! identifier carrying `o` in its own slot to the other slot is the overflowed
//! one and is excluded from matching (kept verbatim in
//! [`Encoding::identifiers`]). No identifier is quarantined for any other
//! reason, and loading fails if the remaining identifiers do not agree on the
//! non-opcode mask bits.

use crate::inst::Arch;
use serde::Deserialize;
use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap, HashSet},
    sync::LazyLock,
};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure to load a catalog or the SOP1 evidence.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum CatalogError {
    #[error("line {line}, column {column}: invalid JSON: {message}")]
    Json {
        line: usize,
        column: usize,
        message: String,
    },
    #[error("line {line}: {reason}")]
    Malformed { line: usize, reason: String },
    #[error("inconsistent catalog: {0}")]
    Inconsistent(String),
    #[error("SOP1 evidence: {0}")]
    Evidence(String),
}

fn json_error(line: usize, error: &serde_json::Error) -> CatalogError {
    CatalogError::Json {
        line,
        column: error.column(),
        message: error.to_string(),
    }
}

/// A condition expression could not be evaluated.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum EvalError {
    #[error("identifier {0} is not bound in a condition")]
    Unbound(&'static str),
    #[error("field {field} is not in the encoding layout")]
    UnknownField { field: &'static str },
    #[error("field {field} lies beyond the {available_bits} available bits")]
    FieldUnavailable {
        field: &'static str,
        available_bits: u32,
    },
    #[error("operator {op}: {reason}")]
    Type {
        op: &'static str,
        reason: &'static str,
    },
}

/// A named condition could not produce a boolean.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum ConditionError {
    #[error("condition {condition} is not defined in the XML")]
    Undefined { condition: String },
    #[error("condition {condition} uses an unsupported construct: {reason}")]
    Unsupported { condition: String, reason: String },
    #[error(transparent)]
    Eval(#[from] EvalError),
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum LayoutError {
    #[error("field {0} is not in the encoding layout")]
    UnknownField(String),
    #[error("field {0} is assigned twice")]
    DuplicateField(String),
    #[error("value {value:#x} does not fit the {width}-bit field {field}")]
    ValueTooWide {
        field: &'static str,
        width: u32,
        value: u128,
    },
}

/// Why a row can never be selected or encoded. Always names the instruction.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum Refusal {
    #[error("{instruction}: encoding {encoding} is a graphics encoding and unsupported by the compute codec")]
    UnsupportedGraphics {
        instruction: &'static str,
        encoding: &'static str,
    },
    #[error("{instruction}: XML condition {condition} of {encoding} is not defined in the XML")]
    UndefinedCondition {
        instruction: &'static str,
        encoding: &'static str,
        condition: &'static str,
    },
    #[error("{instruction}: XML condition {condition} of {encoding} uses an unsupported construct: {reason}")]
    UnsupportedCondition {
        instruction: &'static str,
        encoding: &'static str,
        condition: &'static str,
        reason: String,
    },
    #[error("{instruction}: {encoding}/{condition} has no verified correction: {reason}")]
    NoCorrection {
        instruction: &'static str,
        encoding: &'static str,
        condition: &'static str,
        reason: &'static str,
    },
    #[error("{instruction}: condition {condition} of {encoding} failed to evaluate: {error}")]
    Evaluation {
        instruction: &'static str,
        encoding: &'static str,
        condition: &'static str,
        error: EvalError,
    },
    #[error(transparent)]
    SourceDomain(#[from] Sop1SourceError),
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum EncodeError {
    #[error(transparent)]
    Refused(#[from] Refusal),
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error("{instruction}: no unique XML identifier for opcode {opcode:#x} in {encoding}")]
    Identifier {
        instruction: &'static str,
        encoding: &'static str,
        opcode: u32,
    },
    #[error("{instruction}: field {field} is the opcode and must equal {opcode:#x}")]
    OpcodeField {
        instruction: &'static str,
        field: &'static str,
        opcode: u32,
    },
    #[error("{instruction}: assembled word {word:#x} does not match the {encoding} identifier")]
    IdentifierMismatch {
        instruction: &'static str,
        encoding: &'static str,
        word: u128,
    },
    #[error("{instruction}: condition {condition} of {encoding} is false for the assembled word")]
    PredicateFalse {
        instruction: &'static str,
        encoding: &'static str,
        condition: &'static str,
    },
    #[error("{instruction}: condition evaluation failed: {error}")]
    Eval {
        instruction: &'static str,
        error: EvalError,
    },
}

/// A SOP1 source selector outside what llvm-mc proved for the instruction.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum Sop1SourceError {
    #[error("{instruction}: no SOP1 source evidence in this catalog")]
    NoEvidence { instruction: String },
    #[error("{instruction}: SOP1 source rows are unavailable: {reason}")]
    Unavailable {
        instruction: &'static str,
        reason: &'static str,
    },
    #[error("{instruction}: SSRC0 {selector} is not an 8-bit selector")]
    OutOfRange {
        instruction: &'static str,
        selector: u32,
    },
    #[error("{instruction}: SSRC0 {selector} is outside the llvm-mc-proven domain ({predicate})")]
    OutsideDomain {
        instruction: &'static str,
        selector: u32,
        predicate: &'static str,
    },
    #[error("{instruction}: the literal form (SSRC0 255) is not proven or has no literal record")]
    LiteralUnavailable { instruction: &'static str },
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

fn low_mask(width: u32) -> u128 {
    if width >= 128 {
        u128::MAX
    } else {
        (1u128 << width) - 1
    }
}

/// One contiguous run of bits of a field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BitRange {
    pub offset: u32,
    pub width: u32,
}

/// A named field made of one or more bit ranges. `ranges[0]` holds the least
/// significant bits of the field value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Field {
    pub name: &'static str,
    pub ranges: Vec<BitRange>,
}

impl Field {
    pub fn width(&self) -> u32 {
        self.ranges.iter().map(|r| r.width).sum()
    }
    /// One past the highest bit the field occupies.
    pub fn end_bit(&self) -> u32 {
        self.ranges
            .iter()
            .map(|r| r.offset + r.width)
            .max()
            .unwrap_or(0)
    }
    /// The bits of a word this field occupies.
    pub fn mask(&self) -> u128 {
        self.ranges
            .iter()
            .fold(0, |mask, r| mask | (low_mask(r.width) << r.offset))
    }
    pub fn extract(&self, word: u128) -> u128 {
        let mut value = 0u128;
        let mut shift = 0u32;
        for r in &self.ranges {
            value |= ((word >> r.offset) & low_mask(r.width)) << shift;
            shift += r.width;
        }
        value
    }
    pub fn insert(&self, word: u128, value: u128) -> Result<u128, LayoutError> {
        let width = self.width();
        if width < 128 && value >> width != 0 {
            return Err(LayoutError::ValueTooWide {
                field: self.name,
                width,
                value,
            });
        }
        let mut word = word;
        let mut shift = 0u32;
        for r in &self.ranges {
            let mask = low_mask(r.width) << r.offset;
            word = (word & !mask) | (((value >> shift) << r.offset) & mask);
            shift += r.width;
        }
        Ok(word)
    }
}

/// Which opcode field a row's `opcode` lives in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Slot {
    /// `OP` (or no opcode field at all, opcode 0).
    Whole,
    /// `OPX` of a VOPD pair.
    X,
    /// `OPY` of a VOPD pair.
    Y,
}

impl Slot {
    fn index(self) -> usize {
        match self {
            Slot::Whole => 0,
            Slot::X => 1,
            Slot::Y => 2,
        }
    }
}

/// Pack up to four little-endian dwords into one word.
pub fn pack_words(words: &[u32]) -> u128 {
    words
        .iter()
        .take(4)
        .enumerate()
        .fold(0u128, |word, (i, w)| word | (u128::from(*w) << (32 * i)))
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operator {
    FieldDeref,
    ConsArray,
    And,
    Eq,
    Ne,
    Within,
    NotWithin,
}

impl Operator {
    pub fn as_str(self) -> &'static str {
        match self {
            Operator::FieldDeref => ".fieldderef",
            Operator::ConsArray => ".cons_array",
            Operator::And => "&&",
            Operator::Eq => "==",
            Operator::Ne => "!=",
            Operator::Within => ".within",
            Operator::NotWithin => ".notwithin",
        }
    }
    fn parse(text: &str) -> Option<Operator> {
        Some(match text {
            ".fieldderef" => Operator::FieldDeref,
            ".cons_array" => Operator::ConsArray,
            "&&" => Operator::And,
            "==" => Operator::Eq,
            "!=" => Operator::Ne,
            ".within" => Operator::Within,
            ".notwithin" => Operator::NotWithin,
            _ => return None,
        })
    }
    fn arity_ok(self, count: usize) -> bool {
        match self {
            Operator::FieldDeref
            | Operator::Eq
            | Operator::Ne
            | Operator::Within
            | Operator::NotWithin => count == 2,
            Operator::And => count >= 1,
            Operator::ConsArray => true,
        }
    }
}

/// The `<ty><t base=".." size=".."/></ty>` annotation of an operator node.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TypeTag {
    pub base: &'static str,
    pub size: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expr {
    Lit(u128),
    Id(&'static str),
    Op {
        op: Operator,
        args: Vec<Expr>,
        ty: Option<TypeTag>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    Int(u128),
    Array(Vec<u128>),
    /// The `INST` root.
    Instruction,
    /// `INST.<format>`.
    Format(&'static str),
}

impl Value {
    fn int(self, op: &'static str) -> Result<u128, EvalError> {
        match self {
            Value::Int(v) => Ok(v),
            _ => Err(EvalError::Type {
                op,
                reason: "operand is not a scalar integer",
            }),
        }
    }
}

fn operands<'a>(op: Operator, args: &'a [Expr]) -> Result<(&'a Expr, &'a Expr), EvalError> {
    match args {
        [a, b] => Ok((a, b)),
        _ => Err(EvalError::Type {
            op: op.as_str(),
            reason: "needs exactly two operands",
        }),
    }
}

impl Expr {
    /// Evaluate with `fields` resolving `INST.<format>.<field>` reads. All
    /// operands of every operator are evaluated (no short circuit), so an
    /// undefined reference can never be masked.
    pub fn eval(
        &self,
        fields: &dyn Fn(&'static str) -> Result<u128, EvalError>,
    ) -> Result<Value, EvalError> {
        match self {
            Expr::Lit(v) => Ok(Value::Int(*v)),
            Expr::Id("INST") => Ok(Value::Instruction),
            Expr::Id(name) => Err(EvalError::Unbound(*name)),
            Expr::Op { op, args, .. } => {
                let name = op.as_str();
                match op {
                    Operator::FieldDeref => {
                        let (base, field) = operands(*op, args)?;
                        let Expr::Id(field) = field else {
                            return Err(EvalError::Type {
                                op: name,
                                reason: "field name must be an identifier",
                            });
                        };
                        let field: &'static str = *field;
                        match base.eval(fields)? {
                            Value::Instruction => Ok(Value::Format(field)),
                            Value::Format(_) => fields(field).map(Value::Int),
                            _ => Err(EvalError::Type {
                                op: name,
                                reason: "base must be INST or an INST format",
                            }),
                        }
                    }
                    Operator::ConsArray => {
                        let mut items = Vec::with_capacity(args.len());
                        for arg in args {
                            items.push(arg.eval(fields)?.int(name)?);
                        }
                        Ok(Value::Array(items))
                    }
                    Operator::And => {
                        let mut all = true;
                        for arg in args {
                            all &= arg.eval(fields)?.int(name)? != 0;
                        }
                        Ok(Value::Int(u128::from(all)))
                    }
                    Operator::Eq | Operator::Ne => {
                        let (a, b) = operands(*op, args)?;
                        let a = a.eval(fields)?.int(name)?;
                        let b = b.eval(fields)?.int(name)?;
                        Ok(Value::Int(u128::from((a == b) == (*op == Operator::Eq))))
                    }
                    Operator::Within | Operator::NotWithin => {
                        let (needle, haystack) = operands(*op, args)?;
                        let needle = needle.eval(fields)?.int(name)?;
                        let inside = match haystack {
                            // The usual shape: compare against each element
                            // in place, without materialising the array.
                            Expr::Op {
                                op: Operator::ConsArray,
                                args: items,
                                ..
                            } => {
                                let mut found = false;
                                for item in items {
                                    found |=
                                        item.eval(fields)?.int(Operator::ConsArray.as_str())?
                                            == needle;
                                }
                                found
                            }
                            other => match other.eval(fields)? {
                                Value::Array(items) => items.contains(&needle),
                                _ => {
                                    return Err(EvalError::Type {
                                        op: name,
                                        reason: "right operand is not an array",
                                    })
                                }
                            },
                        };
                        Ok(Value::Int(u128::from(inside == (*op == Operator::Within))))
                    }
                }
            }
        }
    }

    /// Evaluate as a condition: a non-zero scalar is true.
    pub fn holds(
        &self,
        fields: &dyn Fn(&'static str) -> Result<u128, EvalError>,
    ) -> Result<bool, EvalError> {
        match self.eval(fields)? {
            Value::Int(v) => Ok(v != 0),
            _ => Err(EvalError::Type {
                op: "condition",
                reason: "does not evaluate to a boolean",
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConditionBody {
    Expr(Expr),
    /// The XML names the condition but never defines it.
    Undefined,
    /// The expression uses an operator or node this model does not implement.
    Unsupported(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Condition {
    pub name: &'static str,
    pub body: ConditionBody,
}

// ---------------------------------------------------------------------------
// Raw (serde) schema. Strings are borrowed from the `include_str!` text, so a
// JSON escape in any of them is a load error rather than an allocation.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNode<'a> {
    tag: &'a str,
    #[serde(borrow)]
    attributes: BTreeMap<&'a str, &'a str>,
    text: &'a str,
    #[serde(borrow)]
    children: Vec<RawNode<'a>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEncoding<'a> {
    bits: u32,
    #[serde(borrow)]
    conditions: BTreeMap<&'a str, Option<RawNode<'a>>>,
    #[serde(borrow)]
    fields: BTreeMap<&'a str, Vec<[u32; 2]>>,
    identifier_mask: &'a str,
    #[serde(borrow)]
    identifiers: Vec<&'a str>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHeader<'a> {
    #[serde(borrow)]
    encodings: BTreeMap<&'a str, RawEncoding<'a>>,
    instructions: usize,
    #[serde(borrow)]
    key: Vec<&'a str>,
    llvm: &'a str,
    llvm_sha256: &'a str,
    rows: usize,
    schema: u32,
    sha256: &'a str,
    target: &'a str,
    xml: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawOperand<'a> {
    #[serde(rename = "DataFormatName")]
    data_format: &'a str,
    #[serde(rename = "FieldName", borrow, default)]
    field_name: Option<&'a str>,
    #[serde(rename = "Input")]
    input: &'a str,
    #[serde(rename = "IsBinaryMicrocodeRequired")]
    binary_required: &'a str,
    #[serde(rename = "IsImplicit")]
    implicit: &'a str,
    #[serde(rename = "OperandSize")]
    size: &'a str,
    #[serde(rename = "OperandType")]
    operand_type: &'a str,
    #[serde(rename = "Order")]
    order: &'a str,
    #[serde(rename = "Output")]
    output: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFlags {
    #[serde(rename = "IsBranch")]
    is_branch: bool,
    #[serde(rename = "IsConditionalBranch")]
    is_conditional_branch: bool,
    #[serde(rename = "IsImmediatelyExecuted")]
    is_immediately_executed: bool,
    #[serde(rename = "IsIndirectBranch")]
    is_indirect_branch: bool,
    #[serde(rename = "IsProgramTerminator")]
    is_program_terminator: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProbe<'a> {
    status: &'a str,
    #[serde(borrow, default)]
    candidate: Option<&'a str>,
    #[serde(default)]
    decoded: Option<Vec<String>>,
    #[serde(default)]
    diagnostic: Option<String>,
    #[serde(borrow, default)]
    reencoded: Option<&'a str>,
    /// Present only in SOP1 evidence proofs.
    #[serde(default)]
    ssrc0: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRow<'a> {
    #[serde(borrow)]
    aliases: Vec<&'a str>,
    codec_status: &'a str,
    condition: &'a str,
    defs: Vec<usize>,
    encoding: &'a str,
    flags: RawFlags,
    group: &'a str,
    implicit: Vec<usize>,
    #[serde(borrow)]
    memory: Vec<RawOperand<'a>>,
    name: &'a str,
    opcode: u32,
    #[serde(borrow)]
    operands: Vec<RawOperand<'a>>,
    #[serde(borrow)]
    oracle: RawProbe<'a>,
    #[serde(borrow)]
    subgroups: Vec<&'a str>,
    uses: Vec<usize>,
    variant: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawKey<'a> {
    condition: &'a str,
    encoding: &'a str,
    name: &'a str,
    opcode: u32,
    variant: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPredicate<'a> {
    field: &'a str,
    op: &'a str,
    #[serde(default)]
    value: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFormDomain<'a> {
    #[serde(borrow)]
    refused: BTreeMap<&'a str, Vec<[u32; 2]>>,
    selectors: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDomain<'a> {
    #[serde(borrow)]
    by_form: BTreeMap<&'a str, RawFormDomain<'a>>,
    probed: u32,
    selectors: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEvidenceRow<'a> {
    #[serde(borrow)]
    corrected_predicate: Option<&'a str>,
    #[serde(borrow, default)]
    domain: Option<RawDomain<'a>>,
    #[serde(default)]
    domain_ranges: Option<Vec<[u32; 2]>>,
    #[serde(borrow)]
    key: RawKey<'a>,
    name: &'a str,
    #[serde(borrow)]
    proofs: BTreeMap<&'a str, RawProbe<'a>>,
    status: &'a str,
    #[serde(borrow)]
    unavailable_reason: Option<&'a str>,
    xml_condition: &'a str,
    #[serde(borrow)]
    xml_predicate: Option<RawNode<'a>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTarget<'a> {
    cpu: &'a str,
    #[serde(borrow)]
    rows: Vec<RawEvidenceRow<'a>>,
    xml: &'a str,
    xml_sha256: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEvidence<'a> {
    #[serde(borrow)]
    forms: BTreeMap<&'a str, u32>,
    literal: u32,
    llvm: &'a str,
    llvm_sha256: &'a str,
    #[serde(borrow)]
    predicates: BTreeMap<&'a str, RawPredicate<'a>>,
    schema: u32,
    #[serde(borrow)]
    targets: BTreeMap<&'a str, RawTarget<'a>>,
    xml_condition: &'a str,
}

// ---------------------------------------------------------------------------
// Expression parsing
// ---------------------------------------------------------------------------

enum ExprError {
    /// The tree is not shaped like the XML expression schema: a load error.
    Shape(String),
    /// A well-formed construct this model does not implement: the condition
    /// refuses to match.
    Unsupported(String),
}

fn parse_literal(text: &str) -> Result<u128, String> {
    let (digits, radix) =
        if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
            (hex, 16)
        } else if let Some(bin) = text.strip_prefix("0b") {
            (bin, 2)
        } else {
            (text, 10)
        };
    u128::from_str_radix(digits, radix).map_err(|e| format!("literal {text:?}: {e}"))
}

fn parse_hex(text: &str) -> Result<u128, String> {
    let digits = text
        .strip_prefix("0x")
        .ok_or_else(|| format!("expected a 0x-prefixed hex value, found {text:?}"))?;
    u128::from_str_radix(digits, 16).map_err(|e| format!("invalid hex value {text:?}: {e}"))
}

fn parse_type(node: &RawNode<'static>) -> Result<TypeTag, ExprError> {
    let shape = |reason: &str| ExprError::Shape(format!("<ty>: {reason}"));
    let [child] = node.children.as_slice() else {
        return Err(shape("expected exactly one <t> child"));
    };
    if child.tag != "t" || !child.children.is_empty() {
        return Err(shape("child is not a leaf <t>"));
    }
    let base = child
        .attributes
        .get("base")
        .copied()
        .ok_or_else(|| shape("missing base"))?;
    let size = child
        .attributes
        .get("size")
        .ok_or_else(|| shape("missing size"))?
        .parse::<u32>()
        .map_err(|e| shape(&format!("size: {e}")))?;
    Ok(TypeTag { base, size })
}

fn parse_expr(node: &RawNode<'static>) -> Result<Expr, ExprError> {
    if !node.text.trim().is_empty() {
        return Err(ExprError::Shape(format!("<{}> carries text", node.tag)));
    }
    let only_attribute = |key: &str| -> Result<&'static str, ExprError> {
        if node.attributes.len() != 1 {
            return Err(ExprError::Unsupported(format!(
                "<{}> carries attributes other than {key}",
                node.tag
            )));
        }
        node.attributes
            .get(key)
            .copied()
            .ok_or_else(|| ExprError::Shape(format!("<{}> has no {key} attribute", node.tag)))
    };
    match node.tag {
        "lit" => {
            if !node.children.is_empty() {
                return Err(ExprError::Shape("<lit> has children".to_owned()));
            }
            let value = parse_literal(only_attribute("val")?).map_err(ExprError::Unsupported)?;
            Ok(Expr::Lit(value))
        }
        "id" => {
            if !node.children.is_empty() {
                return Err(ExprError::Shape("<id> has children".to_owned()));
            }
            Ok(Expr::Id(only_attribute("val")?))
        }
        "op" => {
            let name = only_attribute("type")?;
            let mut args = Vec::new();
            let mut ty = None;
            for child in &node.children {
                if child.tag == "ty" {
                    if ty.is_some() {
                        return Err(ExprError::Shape("<op> has two <ty> children".to_owned()));
                    }
                    ty = Some(parse_type(child)?);
                } else {
                    args.push(parse_expr(child)?);
                }
            }
            let op = Operator::parse(name)
                .ok_or_else(|| ExprError::Unsupported(format!("operator {name}")))?;
            if !op.arity_ok(args.len()) {
                return Err(ExprError::Unsupported(format!(
                    "operator {name} with {} operands",
                    args.len()
                )));
            }
            Ok(Expr::Op { op, args, ty })
        }
        other => Err(ExprError::Unsupported(format!("expression node <{other}>"))),
    }
}

fn condition_body(raw: Option<RawNode<'static>>) -> Result<ConditionBody, String> {
    let Some(root) = raw else {
        return Ok(ConditionBody::Undefined);
    };
    if !matches!(root.tag, "CondtionExpression" | "ConditionExpression")
        || !root.attributes.is_empty()
        || !root.text.trim().is_empty()
    {
        return Err(format!(
            "condition root <{}> is not a condition expression",
            root.tag
        ));
    }
    let [child] = root.children.as_slice() else {
        return Err("condition expression needs exactly one child".to_owned());
    };
    match parse_expr(child) {
        Ok(expr) => Ok(ConditionBody::Expr(expr)),
        Err(ExprError::Unsupported(reason)) => Ok(ConditionBody::Unsupported(reason)),
        Err(ExprError::Shape(reason)) => Err(reason),
    }
}

/// Parse one `{"tag":"CondtionExpression",...}` tree (or `null`).
pub fn parse_condition_json(json: &'static str) -> Result<ConditionBody, CatalogError> {
    let raw: Option<RawNode<'static>> =
        serde_json::from_str(json).map_err(|e| json_error(e.line(), &e))?;
    condition_body(raw).map_err(|reason| CatalogError::Malformed { line: 1, reason })
}

// ---------------------------------------------------------------------------
// Encodings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CodecStatus {
    /// Metadata only: no numeric semantics are inferred from the catalog.
    MetadataOnly,
    /// A graphics encoding: reported by name, never decoded.
    UnsupportedGraphics,
}

#[derive(Clone, Debug)]
pub struct Encoding {
    pub name: &'static str,
    pub bits: u32,
    pub identifier_mask: u128,
    /// Every identifier exactly as the XML lists it.
    pub identifiers: Vec<u128>,
    /// Identifiers excluded from matching (see the module docs).
    pub quarantined: Vec<u128>,
    pub fields: Vec<Field>,
    pub conditions: Vec<Condition>,
    /// Shared status of the rows using this encoding, if it has any.
    pub status: Option<CodecStatus>,
    /// Indices into `fields` of `OP`, `OPX`, `OPY` (see [`Slot`]).
    opcode_fields: [Option<usize>; 3],
    pair: bool,
    /// For pair (VOPD) encodings: the value of the mask bits outside `OPX`/`OPY`.
    pair_constant: Option<u128>,
    masked: Vec<u128>,
    by_opcode: HashMap<(Slot, u32), Vec<usize>>,
    rows: Vec<usize>,
}

impl Encoding {
    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }
    pub fn condition(&self, name: &str) -> Option<&Condition> {
        self.conditions.iter().find(|c| c.name == name)
    }
    /// Both `OPX` and `OPY` and no `OP`: the two halves of one VOPD word.
    pub fn is_pair(&self) -> bool {
        self.pair
    }
    pub fn opcode_field(&self, slot: Slot) -> Option<&Field> {
        self.opcode_fields[slot.index()].map(|i| &self.fields[i])
    }
    /// Number of bits the identifier mask reads, rounded up to whole dwords.
    pub fn identifier_bits(&self) -> u32 {
        (128 - self.identifier_mask.leading_zeros()).div_ceil(32) * 32
    }
    /// `word & mask` equals a (non-quarantined) identifier.
    pub fn identifier_matches(&self, word: u128) -> bool {
        self.masked
            .binary_search(&(word & self.identifier_mask))
            .is_ok()
    }
    pub fn decompose(&self, word: u128) -> Vec<(&'static str, u128)> {
        self.fields
            .iter()
            .map(|f| (f.name, f.extract(word)))
            .collect()
    }
    /// Assemble a word from named fields; unlisted fields and unowned bits are 0.
    pub fn compose(&self, values: &[(&str, u128)]) -> Result<u128, LayoutError> {
        let mut word = 0u128;
        let mut seen: Vec<&str> = Vec::with_capacity(values.len());
        for &(name, value) in values {
            if seen.contains(&name) {
                return Err(LayoutError::DuplicateField(name.to_owned()));
            }
            let field = self
                .field(name)
                .ok_or_else(|| LayoutError::UnknownField(name.to_owned()))?;
            word = field.insert(word, value)?;
            seen.push(name);
        }
        Ok(word)
    }
    pub fn to_dwords(&self, word: u128) -> Vec<u32> {
        (0..self.bits / 32)
            .map(|i| (word >> (32 * i)) as u32)
            .collect()
    }
    fn eval_expr(&self, expr: &Expr, word: u128, available: u32) -> Result<bool, EvalError> {
        expr.holds(&|name: &'static str| {
            let field = self
                .field(name)
                .ok_or(EvalError::UnknownField { field: name })?;
            if field.end_bit() > available {
                return Err(EvalError::FieldUnavailable {
                    field: name,
                    available_bits: available,
                });
            }
            Ok(field.extract(word))
        })
    }
    /// Evaluate a named condition on `word`, of which `available_bits` are
    /// real. Undefined and unsupported conditions refuse; they never hold.
    pub fn holds(
        &self,
        condition: &str,
        word: u128,
        available_bits: u32,
    ) -> Result<bool, ConditionError> {
        match self.condition(condition).map(|c| &c.body) {
            None | Some(ConditionBody::Undefined) => Err(ConditionError::Undefined {
                condition: condition.to_owned(),
            }),
            Some(ConditionBody::Unsupported(reason)) => Err(ConditionError::Unsupported {
                condition: condition.to_owned(),
                reason: reason.clone(),
            }),
            Some(ConditionBody::Expr(expr)) => Ok(self.eval_expr(expr, word, available_bits)?),
        }
    }
    fn pair_side(&self, name: &str) -> Option<Slot> {
        for (from, to, slot) in [('X', 'Y', Slot::X), ('Y', 'X', Slot::Y)] {
            if let Some(at) = name.find(from) {
                let mut other = name.to_owned();
                other.replace_range(at..=at, &to.to_string());
                if other != name && self.field(&other).is_some() {
                    return Some(slot);
                }
            }
        }
        None
    }
    fn row_slot(&self, row: &Row) -> Result<Slot, String> {
        let has = |name: &str| self.field(name).is_some();
        match (has("OP"), has("OPX"), has("OPY")) {
            (true, false, false) | (false, false, false) => Ok(Slot::Whole),
            (false, true, true) => {
                let mut side: Option<Slot> = None;
                for name in row.operands.iter().filter_map(|o| o.field_name) {
                    if let Some(found) = self.pair_side(name) {
                        if side.is_some_and(|previous| previous != found) {
                            return Err(format!("operands of {} mix the X and Y halves", row.name));
                        }
                        side = Some(found);
                    }
                }
                side.ok_or_else(|| format!("no operand of {} names an X or Y half", row.name))
            }
            _ => Err(format!("encoding {} mixes OP/OPX/OPY fields", self.name)),
        }
    }
    fn rebuild_masked(&mut self) {
        let mut masked: Vec<u128> = self
            .identifiers
            .iter()
            .filter(|id| !self.quarantined.contains(id))
            .map(|id| id & self.identifier_mask)
            .collect();
        masked.sort_unstable();
        masked.dedup();
        self.masked = masked;
    }
    /// Apply the VOPD overflow rule from the module docs. `rows` are the
    /// `(slot, opcode)` pairs of this encoding's catalog rows.
    fn quarantine_overflow(&mut self, rows: &[(Slot, u32)]) -> Result<(), String> {
        if !self.pair {
            return Ok(());
        }
        let x = self
            .opcode_field(Slot::X)
            .expect("pair encodings have OPX")
            .clone();
        let y = self
            .opcode_field(Slot::Y)
            .expect("pair encodings have OPY")
            .clone();
        let single = |field: &Field| -> Result<BitRange, String> {
            match field.ranges.as_slice() {
                [range] => Ok(*range),
                _ => Err(format!(
                    "{}: pair opcode field {} must be one contiguous range",
                    self.name, field.name
                )),
            }
        };
        let mut overflowed: Vec<u128> = Vec::new();
        for (slot, placed_field, other_field) in [(Slot::X, &x, &y), (Slot::Y, &y, &x)] {
            let placed = single(placed_field)?;
            let other = single(other_field)?;
            let mut opcodes: Vec<u32> = rows
                .iter()
                .filter(|(s, o)| *s == slot && other.width < 32 && (*o >> other.width) != 0)
                .map(|(_, o)| *o)
                .collect();
            opcodes.sort_unstable();
            opcodes.dedup();
            for opcode in opcodes {
                let own: Vec<u128> = self
                    .identifiers
                    .iter()
                    .copied()
                    .filter(|id| placed_field.extract(*id) == u128::from(opcode))
                    .collect();
                let [defined] = own.as_slice() else {
                    return Err(format!(
                        "{}: opcode {opcode} needs exactly one identifier in its own slot, found {}",
                        self.name,
                        own.len()
                    ));
                };
                let moved = defined.wrapping_add(
                    (u128::from(opcode) << other.offset)
                        .wrapping_sub(u128::from(opcode) << placed.offset),
                );
                if self.identifiers.contains(&moved) {
                    overflowed.push(moved);
                }
            }
        }
        overflowed.sort_unstable();
        overflowed.dedup();
        self.quarantined = overflowed;
        self.rebuild_masked();
        let constant_mask = self.identifier_mask & !(x.mask() | y.mask());
        let mut constants: Vec<u128> = self
            .identifiers
            .iter()
            .filter(|id| !self.quarantined.contains(id))
            .map(|id| id & constant_mask)
            .collect();
        constants.sort_unstable();
        constants.dedup();
        match constants.as_slice() {
            [constant] => {
                self.pair_constant = Some(*constant);
                Ok(())
            }
            _ => Err(format!(
                "{}: identifiers disagree on the non-opcode mask bits after the overflow rule ({} distinct values)",
                self.name,
                constants.len()
            )),
        }
    }
    /// The identifier bits plus opcode for `row`, if they are unique.
    fn seed(&self, row: &Row) -> Option<u128> {
        let opcode = u128::from(row.opcode);
        let field = self.opcode_field(row.slot);
        match (row.slot, field) {
            (Slot::Y, Some(field)) => field.insert(self.pair_constant?, opcode).ok(),
            (_, field) => {
                // An opcode field outside the identifier mask cannot select an
                // identifier; then the identifier must be unique on its own.
                let wanted = |id: &u128| match field {
                    Some(f) if f.mask() & !self.identifier_mask == 0 => f.extract(*id) == opcode,
                    _ => true,
                };
                let mut hits = self.masked.iter().copied().filter(wanted);
                let first = hits.next()?;
                if hits.next().is_some() {
                    return None;
                }
                match field {
                    Some(field) => field.insert(first, opcode).ok(),
                    None => Some(first),
                }
            }
        }
    }
}

fn build_encoding(name: &'static str, raw: RawEncoding<'static>) -> Result<Encoding, String> {
    let RawEncoding {
        bits,
        conditions: raw_conditions,
        fields: raw_fields,
        identifier_mask,
        identifiers: raw_identifiers,
    } = raw;
    let fail = |reason: String| format!("encoding {name}: {reason}");
    if bits == 0 || bits > 128 || bits % 32 != 0 {
        return Err(fail(format!("{bits} bits is not 1-4 whole dwords")));
    }
    let mut fields: Vec<Field> = Vec::with_capacity(raw_fields.len());
    let mut occupied = 0u128;
    for (field_name, raw_ranges) in &raw_fields {
        let field_name: &'static str = *field_name;
        let mut ranges = Vec::with_capacity(raw_ranges.len());
        for [offset, width] in raw_ranges {
            let (offset, width) = (*offset, *width);
            match offset.checked_add(width) {
                Some(end) if width > 0 && end <= bits => {}
                _ => {
                    return Err(fail(format!(
                        "field {field_name} range [{offset},{width}] is outside the {bits}-bit word"
                    )))
                }
            }
            let mask = low_mask(width) << offset;
            if occupied & mask != 0 {
                return Err(fail(format!("field {field_name} overlaps another field")));
            }
            occupied |= mask;
            ranges.push(BitRange { offset, width });
        }
        let field = Field {
            name: field_name,
            ranges,
        };
        if field.ranges.is_empty() || field.width() > 128 {
            return Err(fail(format!("field {field_name} has no usable bit range")));
        }
        fields.push(field);
    }
    let mask = parse_hex(identifier_mask).map_err(&fail)?;
    let mut identifiers = Vec::with_capacity(raw_identifiers.len());
    for text in &raw_identifiers {
        identifiers.push(parse_hex(text).map_err(&fail)?);
    }
    if bits < 128 && (mask >> bits != 0 || identifiers.iter().any(|id| id >> bits != 0)) {
        return Err(fail("identifier or mask lies outside the word".to_owned()));
    }
    if identifiers.is_empty() {
        return Err(fail("no identifiers".to_owned()));
    }
    let mut conditions = Vec::with_capacity(raw_conditions.len());
    for (condition_name, node) in raw_conditions {
        conditions.push(Condition {
            name: condition_name,
            body: condition_body(node)
                .map_err(|reason| fail(format!("condition {condition_name}: {reason}")))?,
        });
    }
    let index_of = |wanted: &str| fields.iter().position(|f| f.name == wanted);
    let opcode_fields = [index_of("OP"), index_of("OPX"), index_of("OPY")];
    let pair =
        opcode_fields[0].is_none() && opcode_fields[1].is_some() && opcode_fields[2].is_some();
    let mut encoding = Encoding {
        name,
        bits,
        identifier_mask: mask,
        identifiers,
        quarantined: Vec::new(),
        fields,
        conditions,
        status: None,
        opcode_fields,
        pair,
        pair_constant: None,
        masked: Vec::new(),
        by_opcode: HashMap::new(),
        rows: Vec::new(),
    };
    encoding.rebuild_masked();
    Ok(encoding)
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// The full catalog key of a row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct RowKey {
    pub name: &'static str,
    pub encoding: &'static str,
    pub condition: &'static str,
    pub opcode: u32,
    pub variant: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Operand {
    pub field_name: Option<&'static str>,
    pub operand_type: &'static str,
    pub data_format: &'static str,
    pub size: u32,
    pub order: u32,
    pub input: bool,
    pub output: bool,
    pub implicit: bool,
    pub binary_required: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Flags {
    pub is_branch: bool,
    pub is_conditional_branch: bool,
    pub is_immediately_executed: bool,
    pub is_indirect_branch: bool,
    pub is_program_terminator: bool,
}

/// One LLVM oracle probe (catalog `oracle`, or an evidence proof).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Probe {
    pub status: &'static str,
    pub candidate: Option<&'static str>,
    pub reencoded: Option<&'static str>,
    pub decoded: Vec<String>,
    pub diagnostic: Option<String>,
}

impl Probe {
    pub fn verified(&self) -> bool {
        self.status == "verified"
    }
}

fn probe(raw: RawProbe<'static>) -> (Probe, Option<u32>) {
    (
        Probe {
            status: raw.status,
            candidate: raw.candidate,
            reencoded: raw.reencoded,
            decoded: raw.decoded.unwrap_or_default(),
            diagnostic: raw.diagnostic,
        },
        raw.ssrc0,
    )
}

/// How a row's condition resolves.
#[derive(Clone, Debug)]
pub enum Resolution {
    /// The XML condition is defined and is evaluated as written.
    Xml,
    /// The XML condition is dangling; evidence supplies the predicate.
    Corrected(Correction),
    /// Never matches.
    Refused(Refusal),
}

#[derive(Clone, Debug)]
pub struct Correction {
    pub predicate: &'static str,
    pub expr: Expr,
    pub proofs: Vec<Sop1Proof>,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub name: &'static str,
    pub aliases: Vec<&'static str>,
    pub encoding: &'static str,
    pub condition: &'static str,
    /// Instruction-local XML encoding ordinal.
    pub variant: u32,
    pub status: CodecStatus,
    pub opcode: u32,
    pub operands: Vec<Operand>,
    pub defs: Vec<usize>,
    pub uses: Vec<usize>,
    pub implicit: Vec<usize>,
    pub memory: Vec<Operand>,
    pub group: &'static str,
    pub subgroups: Vec<&'static str>,
    pub flags: Flags,
    pub oracle: Probe,
    slot: Slot,
    encoding_index: usize,
    condition_index: Option<usize>,
    resolution: Resolution,
}

impl Row {
    pub fn key(&self) -> RowKey {
        RowKey {
            name: self.name,
            encoding: self.encoding,
            condition: self.condition,
            opcode: self.opcode,
            variant: self.variant,
        }
    }
    pub fn slot(&self) -> Slot {
        self.slot
    }
    pub fn resolution(&self) -> &Resolution {
        &self.resolution
    }
    /// The layout field holding `opcode`, if the encoding has one.
    pub fn opcode_field<'c>(&self, catalog: &'c Catalog) -> Option<&'c Field> {
        catalog.encoding_of(self).opcode_field(self.slot)
    }
    /// Why this row can never be selected or encoded.
    pub fn refusal(&self) -> Option<Refusal> {
        if self.status == CodecStatus::UnsupportedGraphics {
            return Some(Refusal::UnsupportedGraphics {
                instruction: self.name,
                encoding: self.encoding,
            });
        }
        match &self.resolution {
            Resolution::Refused(refusal) => Some(refusal.clone()),
            _ => None,
        }
    }
    /// Same instruction, encoding, condition and opcode with identical
    /// operands, flags, group and every other non-key field. `variant` and the
    /// generator's `oracle` diagnostics are the only fields ignored: duplicate
    /// XML records (e.g. gfx1100 `global_*`/`scratch_*`) differ in nothing else.
    pub fn equivalent_to(&self, other: &Row) -> bool {
        self.name == other.name
            && self.aliases == other.aliases
            && self.encoding == other.encoding
            && self.condition == other.condition
            && self.opcode == other.opcode
            && self.status == other.status
            && self.slot == other.slot
            && self.operands == other.operands
            && self.defs == other.defs
            && self.uses == other.uses
            && self.implicit == other.implicit
            && self.memory == other.memory
            && self.group == other.group
            && self.subgroups == other.subgroups
            && self.flags == other.flags
    }
}

fn flag(text: &str) -> Result<bool, String> {
    match text {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(format!("expected \"true\" or \"false\", found {other:?}")),
    }
}

fn number(text: &str) -> Result<u32, String> {
    text.parse::<u32>().map_err(|e| format!("{text:?}: {e}"))
}

fn build_operand(raw: RawOperand<'static>) -> Result<Operand, String> {
    Ok(Operand {
        field_name: raw.field_name,
        operand_type: raw.operand_type,
        data_format: raw.data_format,
        size: number(raw.size)?,
        order: number(raw.order)?,
        input: flag(raw.input)?,
        output: flag(raw.output)?,
        implicit: flag(raw.implicit)?,
        binary_required: flag(raw.binary_required)?,
    })
}

fn build_row(raw: RawRow<'static>) -> Result<Row, String> {
    let status = match raw.codec_status {
        "metadata-only" => CodecStatus::MetadataOnly,
        "unsupported-graphics" => CodecStatus::UnsupportedGraphics,
        other => return Err(format!("unknown codec_status {other:?}")),
    };
    let operands = raw
        .operands
        .into_iter()
        .map(build_operand)
        .collect::<Result<Vec<_>, _>>()?;
    let memory = raw
        .memory
        .into_iter()
        .map(build_operand)
        .collect::<Result<Vec<_>, _>>()?;
    for index in raw.defs.iter().chain(&raw.uses).chain(&raw.implicit) {
        if *index >= operands.len() {
            return Err(format!("operand index {index} is out of range"));
        }
    }
    let (oracle, ssrc0) = probe(raw.oracle);
    if ssrc0.is_some() {
        return Err("oracle carries an evidence-only ssrc0".to_owned());
    }
    Ok(Row {
        name: raw.name,
        aliases: raw.aliases,
        encoding: raw.encoding,
        condition: raw.condition,
        variant: raw.variant,
        status,
        opcode: raw.opcode,
        operands,
        defs: raw.defs,
        uses: raw.uses,
        implicit: raw.implicit,
        memory,
        group: raw.group,
        subgroups: raw.subgroups,
        flags: Flags {
            is_branch: raw.flags.is_branch,
            is_conditional_branch: raw.flags.is_conditional_branch,
            is_immediately_executed: raw.flags.is_immediately_executed,
            is_indirect_branch: raw.flags.is_indirect_branch,
            is_program_terminator: raw.flags.is_program_terminator,
        },
        oracle,
        slot: Slot::Whole,
        encoding_index: 0,
        condition_index: None,
        resolution: Resolution::Xml,
    })
}

fn link_row(
    row: &mut Row,
    position: usize,
    encodings: &mut [Encoding],
    by_encoding: &HashMap<&'static str, usize>,
) -> Result<(), String> {
    let index = *by_encoding
        .get(row.encoding)
        .ok_or_else(|| format!("encoding {} is not in the header", row.encoding))?;
    let encoding = &mut encodings[index];
    row.encoding_index = index;
    row.condition_index = encoding
        .conditions
        .iter()
        .position(|c| c.name == row.condition);
    row.slot = encoding.row_slot(row)?;
    // Y-only VOPD opcodes never fit OPX but sit in the Y slot, whose field is
    // wide enough; any other opcode that does not fit is a catalog defect.
    match encoding.opcode_field(row.slot).map(Field::width) {
        Some(width) if width < 32 && row.opcode >> width != 0 => {
            return Err(format!(
                "opcode {} does not fit the {width}-bit opcode field of {}",
                row.opcode, row.encoding
            ));
        }
        None if row.opcode != 0 => {
            return Err(format!(
                "{} has no opcode field but opcode {}",
                row.encoding, row.opcode
            ));
        }
        _ => {}
    }
    match encoding.status {
        None => encoding.status = Some(row.status),
        Some(status) if status != row.status => {
            return Err(format!("rows of {} disagree on codec_status", row.encoding));
        }
        Some(_) => {}
    }
    row.resolution = match row.condition_index.map(|i| &encoding.conditions[i].body) {
        Some(ConditionBody::Expr(_)) => Resolution::Xml,
        Some(ConditionBody::Unsupported(reason)) => {
            Resolution::Refused(Refusal::UnsupportedCondition {
                instruction: row.name,
                encoding: row.encoding,
                condition: row.condition,
                reason: reason.clone(),
            })
        }
        Some(ConditionBody::Undefined) | None => Resolution::Refused(Refusal::UndefinedCondition {
            instruction: row.name,
            encoding: row.encoding,
            condition: row.condition,
        }),
    };
    encoding
        .by_opcode
        .entry((row.slot, row.opcode))
        .or_default()
        .push(position);
    encoding.rows.push(position);
    Ok(())
}

// ---------------------------------------------------------------------------
// SOP1 evidence
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompareOp {
    Eq,
    Ne,
    /// Membership in the row's `domain_ranges`.
    InRanges,
}

/// A named predicate shape from the evidence `predicates` table.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidencePredicate {
    pub name: &'static str,
    pub field: &'static str,
    pub op: CompareOp,
    /// Present for `==` / `!=`, absent for `in-ranges`.
    pub value: Option<u32>,
}

/// A predicate bound to one row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceCheck {
    Eq(u32),
    Ne(u32),
    /// Inclusive `(lo, hi)` selector ranges, sorted and disjoint.
    InRanges(Vec<(u32, u32)>),
}

impl EvidenceCheck {
    pub fn admits(&self, selector: u32) -> bool {
        match self {
            EvidenceCheck::Eq(v) => selector == *v,
            EvidenceCheck::Ne(v) => selector != *v,
            EvidenceCheck::InRanges(ranges) => {
                ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&selector))
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sop1Proof {
    /// `register`, `inline` or `literal`.
    pub form: &'static str,
    pub ssrc0: u32,
    pub probe: Probe,
}

/// Per-form result of the exhaustive selector probe of a domain row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormDomain {
    pub form: &'static str,
    pub selectors: Vec<(u32, u32)>,
    /// Refusal reason and the selector ranges it covers.
    pub refused: Vec<(&'static str, Vec<(u32, u32)>)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomainEvidence {
    pub probed: u32,
    pub selectors: Vec<(u32, u32)>,
    pub forms: Vec<FormDomain>,
}

#[derive(Clone, Debug)]
pub enum EvidenceOutcome {
    Corrected {
        predicate: &'static str,
        field: &'static str,
        check: EvidenceCheck,
    },
    Unavailable {
        reason: &'static str,
    },
}

#[derive(Clone, Debug)]
pub struct RowEvidence {
    pub key: RowKey,
    pub outcome: EvidenceOutcome,
    pub proofs: Vec<Sop1Proof>,
    pub domain: Option<DomainEvidence>,
}

#[derive(Clone, Debug)]
pub struct TargetEvidence {
    pub target: &'static str,
    pub cpu: &'static str,
    pub xml: &'static str,
    pub xml_sha256: &'static str,
    pub rows: Vec<RowEvidence>,
}

/// `isa/sop1-overrides.json`: llvm-mc-proved predicates for the ENC_SOP1 rows
/// whose XML condition is dangling.
#[derive(Clone, Debug, Default)]
pub struct Sop1Evidence {
    pub llvm: &'static str,
    pub llvm_sha256: &'static str,
    pub xml_condition: &'static str,
    /// The literal dword the probes appended.
    pub literal: u32,
    pub forms: Vec<(&'static str, u32)>,
    pub predicates: Vec<EvidencePredicate>,
    pub targets: Vec<TargetEvidence>,
}

/// Inclusive selector ranges: `lo <= hi <= 511`, ascending and disjoint.
fn ranges(raw: &[[u32; 2]], what: &str) -> Result<Vec<(u32, u32)>, String> {
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(raw.len());
    for &[lo, hi] in raw {
        if lo > hi || hi > 511 || out.last().is_some_and(|&(_, previous)| lo <= previous) {
            return Err(format!(
                "{what}: range [{lo},{hi}] is invalid or out of order"
            ));
        }
        out.push((lo, hi));
    }
    Ok(out)
}

impl Sop1Evidence {
    /// No evidence: every dangling condition stays refused.
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn shipped() -> &'static Sop1Evidence {
        static EVIDENCE: LazyLock<Sop1Evidence> = LazyLock::new(|| {
            Sop1Evidence::parse(include_str!("../../isa/sop1-overrides.json"))
                .expect("shipped SOP1 evidence must parse")
        });
        &EVIDENCE
    }
    pub fn parse(text: &'static str) -> Result<Self, CatalogError> {
        let raw: RawEvidence<'static> =
            serde_json::from_str(text).map_err(|e| json_error(e.line(), &e))?;
        Self::from_raw(raw).map_err(CatalogError::Evidence)
    }
    fn from_raw(raw: RawEvidence<'static>) -> Result<Self, String> {
        if raw.schema != 1 {
            return Err(format!("unsupported schema {}", raw.schema));
        }
        let mut predicates = Vec::new();
        for (name, p) in raw.predicates {
            let (op, value) = match (p.op, p.value) {
                ("==", Some(v)) => (CompareOp::Eq, Some(v)),
                ("!=", Some(v)) => (CompareOp::Ne, Some(v)),
                ("in-ranges", None) => (CompareOp::InRanges, None),
                (other, _) => {
                    return Err(format!(
                        "predicate {name}: unsupported operator {other:?} or value shape"
                    ))
                }
            };
            predicates.push(EvidencePredicate {
                name,
                field: p.field,
                op,
                value,
            });
        }
        let mut targets = Vec::new();
        for (target, raw_target) in raw.targets {
            let mut rows = Vec::new();
            for row in raw_target.rows {
                rows.push(Self::row(row, &predicates).map_err(|e| format!("{target}: {e}"))?);
            }
            targets.push(TargetEvidence {
                target,
                cpu: raw_target.cpu,
                xml: raw_target.xml,
                xml_sha256: raw_target.xml_sha256,
                rows,
            });
        }
        Ok(Self {
            llvm: raw.llvm,
            llvm_sha256: raw.llvm_sha256,
            xml_condition: raw.xml_condition,
            literal: raw.literal,
            forms: raw.forms.into_iter().collect(),
            predicates,
            targets,
        })
    }
    fn row(
        raw: RawEvidenceRow<'static>,
        predicates: &[EvidencePredicate],
    ) -> Result<RowEvidence, String> {
        let key = RowKey {
            name: raw.key.name,
            encoding: raw.key.encoding,
            condition: raw.key.condition,
            opcode: raw.key.opcode,
            variant: raw.key.variant,
        };
        let fail = |reason: &str| format!("{key:?}: {reason}");
        if raw.name != key.name || raw.xml_condition != key.condition {
            return Err(fail("name/xml_condition disagree with the key"));
        }
        if raw.xml_predicate.is_some() {
            return Err(fail("xml_predicate must be null for a dangling condition"));
        }
        let domain_ranges = raw
            .domain_ranges
            .as_deref()
            .map(|ranges_| ranges(ranges_, "domain_ranges"))
            .transpose()
            .map_err(|e| fail(&e))?;
        let outcome =
            match (raw.status, raw.corrected_predicate, raw.unavailable_reason) {
                ("corrected", Some(name), None) => {
                    let def = predicates
                        .iter()
                        .find(|p| p.name == name)
                        .ok_or_else(|| fail(&format!("unknown predicate {name}")))?;
                    let check = match (def.op, def.value, domain_ranges.clone()) {
                        (CompareOp::Eq, Some(v), None) => EvidenceCheck::Eq(v),
                        (CompareOp::Ne, Some(v), None) => EvidenceCheck::Ne(v),
                        (CompareOp::InRanges, None, Some(r)) if !r.is_empty() => {
                            EvidenceCheck::InRanges(r)
                        }
                        _ => return Err(fail("predicate shape and domain_ranges disagree")),
                    };
                    EvidenceOutcome::Corrected {
                        predicate: def.name,
                        field: def.field,
                        check,
                    }
                }
                ("unavailable", None, Some(reason)) if domain_ranges.is_none() => {
                    EvidenceOutcome::Unavailable { reason }
                }
                _ => return Err(fail(
                    "status, corrected_predicate, unavailable_reason and domain_ranges disagree",
                )),
            };
        let domain = match raw.domain {
            None => None,
            Some(raw_domain) => {
                let selectors =
                    ranges(&raw_domain.selectors, "domain.selectors").map_err(|e| fail(&e))?;
                if domain_ranges.as_ref() != Some(&selectors) {
                    return Err(fail("domain.selectors differ from domain_ranges"));
                }
                let mut forms = Vec::new();
                for (form, raw_form) in raw_domain.by_form {
                    let mut refused = Vec::new();
                    for (reason, raw_ranges) in raw_form.refused {
                        refused.push((
                            reason,
                            ranges(&raw_ranges, "domain.refused").map_err(|e| fail(&e))?,
                        ));
                    }
                    forms.push(FormDomain {
                        form,
                        selectors: ranges(&raw_form.selectors, "domain.by_form")
                            .map_err(|e| fail(&e))?,
                        refused,
                    });
                }
                Some(DomainEvidence {
                    probed: raw_domain.probed,
                    selectors,
                    forms,
                })
            }
        };
        if domain.is_some()
            != matches!(
                outcome,
                EvidenceOutcome::Corrected {
                    check: EvidenceCheck::InRanges(_),
                    ..
                }
            )
        {
            return Err(fail("only in-ranges rows carry a domain"));
        }
        let mut proofs = Vec::new();
        for (form, raw_probe) in raw.proofs {
            let (measured, ssrc0) = probe(raw_probe);
            let ssrc0 = ssrc0.ok_or_else(|| fail(&format!("proof {form} has no ssrc0")))?;
            proofs.push(Sop1Proof {
                form,
                ssrc0,
                probe: measured,
            });
        }
        Ok(RowEvidence {
            key,
            outcome,
            proofs,
            domain,
        })
    }
}

/// The per-instruction rule [`Catalog::validate_sop1_source`] enforces.
#[derive(Clone, Debug)]
struct Sop1Rule {
    instruction: &'static str,
    outcome: EvidenceOutcome,
    /// SSRC0 255 (a literal dword follows) is proven and has a literal record.
    literal_available: bool,
}

fn check_expr(encoding: &'static str, field: &'static str, check: &EvidenceCheck) -> Expr {
    let read = Expr::Op {
        op: Operator::FieldDeref,
        args: vec![
            Expr::Op {
                op: Operator::FieldDeref,
                args: vec![Expr::Id("INST"), Expr::Id(encoding)],
                ty: None,
            },
            Expr::Id(field),
        ],
        ty: None,
    };
    let compare = |op: Operator, value: u32| Expr::Op {
        op,
        args: vec![read.clone(), Expr::Lit(u128::from(value))],
        ty: None,
    };
    match check {
        EvidenceCheck::Eq(v) => compare(Operator::Eq, *v),
        EvidenceCheck::Ne(v) => compare(Operator::Ne, *v),
        EvidenceCheck::InRanges(selectors) => Expr::Op {
            op: Operator::Within,
            args: vec![
                read,
                Expr::Op {
                    op: Operator::ConsArray,
                    args: selectors
                        .iter()
                        .flat_map(|&(lo, hi)| lo..=hi)
                        .map(|v| Expr::Lit(u128::from(v)))
                        .collect(),
                    ty: None,
                },
            ],
            ty: None,
        },
    }
}

fn apply_evidence(
    rows: &mut [Row],
    encodings: &[Encoding],
    evidence: &RowEvidence,
    rules: &mut HashMap<&'static str, Sop1Rule>,
) -> Result<(), CatalogError> {
    let fail = |reason: &str| CatalogError::Evidence(format!("{:?}: {reason}", evidence.key));
    let index = rows
        .iter()
        .position(|row| row.key() == evidence.key)
        .ok_or_else(|| fail("row is not in the catalog"))?;
    if !matches!(
        rows[index].resolution,
        Resolution::Refused(Refusal::UndefinedCondition { .. })
    ) {
        return Err(fail(
            "the XML condition is defined or already resolved; evidence cannot override it",
        ));
    }
    let (name, opcode, encoding_name, encoding_index, condition) = {
        let row = &rows[index];
        (
            row.name,
            row.opcode,
            row.encoding,
            row.encoding_index,
            row.condition,
        )
    };
    let encoding = &encodings[encoding_index];
    let literal_proven = evidence
        .proofs
        .iter()
        .any(|p| p.form == "literal" && p.probe.verified());
    // The sibling literal record is reachable only when SSRC0 == 255 is proven
    // for the instruction; otherwise it is refused by name.
    let (resolution, literal_refusal): (Resolution, Option<&'static str>) = match &evidence.outcome
    {
        EvidenceOutcome::Corrected {
            predicate,
            field,
            check,
        } => {
            if encoding.field(field).is_none() {
                return Err(fail("corrected predicate reads a field the layout lacks"));
            }
            let literal_ok = match check {
                EvidenceCheck::Ne(_) => literal_proven,
                other => other.admits(255),
            };
            let refusal = if literal_ok {
                None
            } else if matches!(check, EvidenceCheck::Ne(_)) {
                Some("literal-form-not-verified")
            } else {
                Some("literal-selector-outside-proven-domain")
            };
            (
                Resolution::Corrected(Correction {
                    predicate: *predicate,
                    expr: check_expr(encoding.name, *field, check),
                    proofs: evidence.proofs.clone(),
                }),
                refusal,
            )
        }
        EvidenceOutcome::Unavailable { reason } => (
            Resolution::Refused(Refusal::NoCorrection {
                instruction: name,
                encoding: encoding_name,
                condition,
                reason: *reason,
            }),
            Some(*reason),
        ),
    };
    rows[index].resolution = resolution;
    let sibling = format!(
        "{}_INST_LITERAL",
        encoding_name.strip_prefix("ENC_").unwrap_or(encoding_name)
    );
    let mut literal_row_exists = false;
    for row in rows
        .iter_mut()
        .filter(|r| r.name == name && r.opcode == opcode && r.encoding == sibling)
    {
        match literal_refusal {
            Some(reason) => {
                row.resolution = Resolution::Refused(Refusal::NoCorrection {
                    instruction: name,
                    encoding: row.encoding,
                    condition: row.condition,
                    reason,
                });
            }
            None => literal_row_exists = true,
        }
    }
    let rule = Sop1Rule {
        instruction: name,
        outcome: evidence.outcome.clone(),
        literal_available: literal_refusal.is_none() && literal_row_exists,
    };
    if rules.insert(name, rule).is_some() {
        return Err(fail("a second evidence row for the same instruction"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct Catalog {
    target: &'static str,
    xml: &'static str,
    xml_sha256: &'static str,
    llvm: &'static str,
    llvm_sha256: &'static str,
    instructions: usize,
    encodings: Vec<Encoding>,
    rows: Vec<Row>,
    names: HashMap<&'static str, Vec<usize>>,
    sop1: HashMap<&'static str, Sop1Rule>,
}

/// A row a word could not select, with the reason.
#[derive(Clone, Debug)]
pub struct RowRefusal<'c> {
    pub row: &'c Row,
    pub refusal: Cow<'c, Refusal>,
}

/// Both halves of a VOPD word.
#[derive(Clone, Copy, Debug)]
pub struct VopdPair<'c> {
    pub encoding: &'c Encoding,
    pub x: &'c Row,
    pub y: &'c Row,
}

#[derive(Clone, Copy, Debug)]
pub enum Selection<'c> {
    Single(&'c Row),
    Vopd(VopdPair<'c>),
}

fn describe_rows(rows: &[&Row]) -> String {
    rows.iter()
        .map(|r| format!("{}[{}/{}#{}]", r.name, r.encoding, r.condition, r.variant))
        .collect::<Vec<_>>()
        .join(", ")
}

fn describe_refusals(refusals: &[RowRefusal<'_>]) -> String {
    refusals
        .iter()
        .map(|r| r.refusal.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Why a word selected no row, or more than one.
#[derive(Debug, Error)]
pub enum SelectError<'c> {
    #[error("no XML encoding identifier matches word {word:#x}")]
    NoEncoding { word: u128 },
    #[error("encoding(s) {encodings:?} match the identifier but no row matches the opcode and condition")]
    NoRow { encodings: Vec<&'static str> },
    #[error("instruction needs {needed_bits} bits, only {available_bits} available")]
    Truncated {
        needed_bits: u32,
        available_bits: u32,
    },
    #[error("ambiguous XML selection: {}", describe_rows(.candidates))]
    Ambiguous { candidates: Vec<&'c Row> },
    #[error("every matching row is refused: {}", describe_refusals(.refusals))]
    Refused { refusals: Vec<RowRefusal<'c>> },
    #[error("graphics encoding(s) {encodings:?} are unsupported by the compute codec: {}", describe_rows(.rows))]
    UnsupportedGraphics {
        encodings: Vec<&'static str>,
        rows: Vec<&'c Row>,
    },
    #[error("the VOPD X and Y halves resolve in different encodings (x: {}; y: {})", describe_rows(.x), describe_rows(.y))]
    VopdSlots { x: Vec<&'c Row>, y: Vec<&'c Row> },
    #[error("{encoding}: {reason}")]
    Unmodeled {
        encoding: &'static str,
        reason: &'static str,
    },
}

impl SelectError<'_> {
    /// The variants whose vectors are only filled by a collecting pass.
    fn needs_detail(&self) -> bool {
        matches!(
            self,
            SelectError::NoRow { .. }
                | SelectError::Ambiguous { .. }
                | SelectError::Refused { .. }
                | SelectError::UnsupportedGraphics { .. }
                | SelectError::VopdSlots { .. }
        )
    }
}

/// Folds live candidates without allocating. Identical duplicate records
/// collapse to the lowest variant; `all` is filled only by a collecting pass.
#[derive(Default)]
struct Accum<'c> {
    best: Option<&'c Row>,
    conflict: bool,
    any_compute: bool,
    all: Vec<&'c Row>,
}

impl<'c> Accum<'c> {
    fn push(&mut self, row: &'c Row, collect: bool) {
        if collect {
            self.all.push(row);
        }
        self.any_compute |= row.status != CodecStatus::UnsupportedGraphics;
        let best = self.best;
        match best {
            None => self.best = Some(row),
            Some(current) if current.equivalent_to(row) => {
                if row.variant < current.variant {
                    self.best = Some(row);
                }
            }
            Some(_) => self.conflict = true,
        }
    }
}

#[derive(Default)]
struct Noise<'c> {
    collect: bool,
    refused_any: bool,
    refused: Vec<RowRefusal<'c>>,
    truncated: Option<u32>,
}

impl<'c> Noise<'c> {
    fn note_truncated(&mut self, bits: u32) {
        self.truncated = Some(self.truncated.map_or(bits, |t| t.max(bits)));
    }
    fn refuse(&mut self, row: &'c Row, refusal: impl FnOnce() -> Cow<'c, Refusal>) {
        self.refused_any = true;
        if self.collect {
            self.refused.push(RowRefusal {
                row,
                refusal: refusal(),
            });
        }
    }
}

fn opcode_in(encoding: &Encoding, slot: Slot, word: u128, available: u32) -> Option<u32> {
    match encoding.opcode_field(slot) {
        None => Some(0),
        Some(field) if field.end_bit() > available => None,
        Some(field) => Some(u32::try_from(field.extract(word)).unwrap_or(u32::MAX)),
    }
}

fn candidates(encoding: &Encoding, slot: Slot, opcode: u32) -> &[usize] {
    const NONE: &[usize] = &[];
    encoding
        .by_opcode
        .get(&(slot, opcode))
        .map_or(NONE, |rows| rows.as_slice())
}

impl Catalog {
    pub fn target(&self) -> &'static str {
        self.target
    }
    pub fn xml(&self) -> &'static str {
        self.xml
    }
    pub fn xml_sha256(&self) -> &'static str {
        self.xml_sha256
    }
    pub fn llvm(&self) -> &'static str {
        self.llvm
    }
    pub fn llvm_sha256(&self) -> &'static str {
        self.llvm_sha256
    }
    /// Distinct instructions in the XML (rows are encodings of them).
    pub fn instructions(&self) -> usize {
        self.instructions
    }
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }
    pub fn encodings(&self) -> &[Encoding] {
        &self.encodings
    }
    pub fn encoding(&self, name: &str) -> Option<&Encoding> {
        self.encodings.iter().find(|e| e.name == name)
    }
    pub fn encoding_of(&self, row: &Row) -> &Encoding {
        &self.encodings[row.encoding_index]
    }
    pub fn rows_of<'c>(&'c self, encoding: &'c Encoding) -> impl Iterator<Item = &'c Row> + 'c {
        encoding.rows.iter().map(move |&i| &self.rows[i])
    }
    pub fn rows_named(&self, name: &str) -> impl Iterator<Item = &Row> + '_ {
        self.names
            .get(name)
            .into_iter()
            .flatten()
            .map(|&i| &self.rows[i])
    }
    pub fn row_by_key(
        &self,
        name: &str,
        encoding: &str,
        condition: &str,
        opcode: u32,
        variant: u32,
    ) -> Option<&Row> {
        self.rows_named(name).find(|r| {
            r.encoding == encoding
                && r.condition == condition
                && r.opcode == opcode
                && r.variant == variant
        })
    }
    /// Every record equivalent to `row` (including itself): see
    /// [`Row::equivalent_to`].
    pub fn equivalents<'c>(&'c self, row: &'c Row) -> impl Iterator<Item = &'c Row> + 'c {
        self.rows_named(row.name)
            .filter(move |r| r.equivalent_to(row))
    }
    /// Names of the encodings reported as unsupported graphics.
    pub fn unsupported_graphics_encodings(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self
            .encodings
            .iter()
            .filter(|e| e.status == Some(CodecStatus::UnsupportedGraphics))
            .map(|e| e.name)
            .collect();
        names.sort_unstable();
        names
    }

    pub fn has_sop1_source_rule(&self, name: &str) -> bool {
        self.sop1.contains_key(name)
    }

    /// Whether `selector` may appear in SSRC0 of SOP1 instruction `name`.
    ///
    /// The domain is the llvm-mc-proven one of the instruction and applies to
    /// both physical records: the base `ENC_SOP1` row and the
    /// `SOP1_INST_LITERAL` row (selector 255). Under the normal
    /// `ssrc0-not-literal` split, selector 255 is allowed only when the literal
    /// form has a verified proof and a literal record exists.
    pub fn validate_sop1_source(&self, name: &str, selector: u32) -> Result<(), Sop1SourceError> {
        let rule = self
            .sop1
            .get(name)
            .ok_or_else(|| Sop1SourceError::NoEvidence {
                instruction: name.to_owned(),
            })?;
        let instruction = rule.instruction;
        if selector > 255 {
            return Err(Sop1SourceError::OutOfRange {
                instruction,
                selector,
            });
        }
        match &rule.outcome {
            EvidenceOutcome::Unavailable { reason } => Err(Sop1SourceError::Unavailable {
                instruction,
                reason: *reason,
            }),
            EvidenceOutcome::Corrected {
                predicate, check, ..
            } => {
                // Under the normal `!= 255` split the literal marker is not a
                // domain member: it is governed by the literal-form proof.
                let not_literal_split = matches!(check, EvidenceCheck::Ne(255));
                if (selector != 255 || !not_literal_split) && !check.admits(selector) {
                    return Err(Sop1SourceError::OutsideDomain {
                        instruction,
                        selector,
                        predicate: *predicate,
                    });
                }
                if selector == 255 && !rule.literal_available {
                    return Err(Sop1SourceError::LiteralUnavailable { instruction });
                }
                Ok(())
            }
        }
    }

    pub fn parse(text: &'static str, evidence: &Sop1Evidence) -> Result<Catalog, CatalogError> {
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty());
        let (header_index, header_text) = lines
            .next()
            .ok_or_else(|| CatalogError::Inconsistent("catalog is empty".to_owned()))?;
        let header_line = header_index + 1;
        let header: RawHeader<'static> =
            serde_json::from_str(header_text).map_err(|e| json_error(header_line, &e))?;
        let malformed =
            |line: usize| move |reason: String| CatalogError::Malformed { line, reason };
        if header.schema != 1 {
            return Err(CatalogError::Inconsistent(format!(
                "unsupported catalog schema {}",
                header.schema
            )));
        }
        if header.key != ["name", "encoding", "condition", "opcode", "variant"] {
            return Err(CatalogError::Inconsistent(format!(
                "unexpected row key {:?}",
                header.key
            )));
        }
        let mut encodings = Vec::with_capacity(header.encodings.len());
        for (name, raw) in header.encodings {
            encodings.push(build_encoding(name, raw).map_err(malformed(header_line))?);
        }
        let by_encoding: HashMap<&'static str, usize> = encodings
            .iter()
            .enumerate()
            .map(|(i, e)| (e.name, i))
            .collect();
        let mut rows: Vec<Row> = Vec::with_capacity(header.rows);
        let mut seen: HashSet<RowKey> = HashSet::with_capacity(header.rows);
        for (index, line_text) in lines {
            let line = index + 1;
            let raw: RawRow<'static> =
                serde_json::from_str(line_text).map_err(|e| json_error(line, &e))?;
            let mut row = build_row(raw).map_err(malformed(line))?;
            link_row(&mut row, rows.len(), &mut encodings, &by_encoding)
                .map_err(malformed(line))?;
            if !seen.insert(row.key()) {
                return Err(CatalogError::Malformed {
                    line,
                    reason: format!("duplicate full XML key {:?}", row.key()),
                });
            }
            rows.push(row);
        }
        if rows.len() != header.rows {
            return Err(CatalogError::Inconsistent(format!(
                "header declares {} rows, found {}",
                header.rows,
                rows.len()
            )));
        }
        for encoding in &mut encodings {
            let placed: Vec<(Slot, u32)> = encoding
                .rows
                .iter()
                .map(|&i| (rows[i].slot, rows[i].opcode))
                .collect();
            encoding
                .quarantine_overflow(&placed)
                .map_err(CatalogError::Inconsistent)?;
        }
        let mut sop1: HashMap<&'static str, Sop1Rule> = HashMap::new();
        if let Some(target) = evidence.targets.iter().find(|t| t.target == header.target) {
            if target.xml != header.xml || target.xml_sha256 != header.sha256 {
                return Err(CatalogError::Evidence(format!(
                    "evidence for {} was generated from a different XML ({} {})",
                    header.target, target.xml, target.xml_sha256
                )));
            }
            for row in &target.rows {
                apply_evidence(&mut rows, &encodings, row, &mut sop1)?;
            }
        }
        let mut names: HashMap<&'static str, Vec<usize>> = HashMap::new();
        for (i, row) in rows.iter().enumerate() {
            names.entry(row.name).or_default().push(i);
        }
        Ok(Catalog {
            target: header.target,
            xml: header.xml,
            xml_sha256: header.sha256,
            llvm: header.llvm,
            llvm_sha256: header.llvm_sha256,
            instructions: header.instructions,
            encodings,
            rows,
            names,
            sop1,
        })
    }

    /// The XML condition of `row` as written, if it is a defined expression.
    fn xml_expr(&self, row: &Row) -> Option<&Expr> {
        let condition = self.encodings[row.encoding_index]
            .conditions
            .get(row.condition_index?)?;
        match &condition.body {
            ConditionBody::Expr(expr) => Some(expr),
            _ => None,
        }
    }

    fn predicate_of<'a>(&'a self, row: &'a Row) -> Option<&'a Expr> {
        match &row.resolution {
            Resolution::Xml => self.xml_expr(row),
            Resolution::Corrected(correction) => Some(&correction.expr),
            Resolution::Refused(_) => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate<'c>(
        &'c self,
        encoding: &'c Encoding,
        indices: &[usize],
        word: u128,
        available: u32,
        acc: &mut Accum<'c>,
        noise: &mut Noise<'c>,
    ) {
        for &i in indices {
            let row = &self.rows[i];
            if let Some(field) = encoding
                .field("SSRC0")
                .filter(|_| self.sop1.contains_key(row.name))
            {
                if let Err(error) = self.validate_sop1_source(row.name, field.extract(word) as u32)
                {
                    noise.refuse(row, || Cow::Owned(Refusal::SourceDomain(error)));
                    continue;
                }
            }
            if let Resolution::Refused(refusal) = &row.resolution {
                // A refused row with a defined XML condition only matters when
                // that condition holds for this word (the literal sibling of a
                // refused instruction); a dangling one can never be ruled out.
                let relevant = match self.xml_expr(row) {
                    Some(expr) => !matches!(encoding.eval_expr(expr, word, available), Ok(false)),
                    None => true,
                };
                if relevant {
                    noise.refuse(row, || Cow::Borrowed(refusal));
                }
                continue;
            }
            let expr = self
                .predicate_of(row)
                .expect("a row that is not refused has a predicate");
            match encoding.eval_expr(expr, word, available) {
                Ok(true) if encoding.bits > available => noise.note_truncated(encoding.bits),
                Ok(true) => acc.push(row, noise.collect),
                Ok(false) => {}
                Err(EvalError::FieldUnavailable { .. }) => noise.note_truncated(encoding.bits),
                Err(error) => noise.refuse(row, || {
                    Cow::Owned(Refusal::Evaluation {
                        instruction: row.name,
                        encoding: row.encoding,
                        condition: row.condition,
                        error,
                    })
                }),
            }
        }
    }

    /// Select the single row (or VOPD row pair) encoded by the instruction
    /// starting at `words[0]`; at most four dwords are read. A successful
    /// selection does not allocate.
    pub fn select<'c>(&'c self, words: &[u32]) -> Result<Selection<'c>, SelectError<'c>> {
        match self.select_pass(words, false) {
            Err(error) if error.needs_detail() => self.select_pass(words, true),
            result => result,
        }
    }

    fn select_pass<'c>(
        &'c self,
        words: &[u32],
        collect: bool,
    ) -> Result<Selection<'c>, SelectError<'c>> {
        if words.is_empty() {
            return Err(SelectError::Truncated {
                needed_bits: 32,
                available_bits: 0,
            });
        }
        let available = 32 * u32::try_from(words.len().min(4)).expect("at most four dwords");
        let word = pack_words(words);
        let mut singles = Accum::default();
        let mut noise = Noise {
            collect,
            ..Noise::default()
        };
        let mut pair_count = 0usize;
        let mut first_pair: Option<(&'c Encoding, &'c Row, &'c Row)> = None;
        let mut pair_rows: Vec<&'c Row> = Vec::new();
        let mut ambiguous_any = false;
        let mut ambiguous: Vec<&'c Row> = Vec::new();
        let mut partial_any = false;
        let mut partial_x = Accum::default();
        let mut partial_y = Accum::default();
        let mut matched_any = false;
        let mut matched: Vec<&'static str> = Vec::new();
        for encoding in &self.encodings {
            let needed = encoding.identifier_bits();
            if needed > available {
                noise.note_truncated(needed);
                continue;
            }
            if !encoding.identifier_matches(word) {
                continue;
            }
            matched_any = true;
            if collect {
                matched.push(encoding.name);
            }
            if encoding.is_pair() {
                let (Some(opx), Some(opy)) = (
                    opcode_in(encoding, Slot::X, word, available),
                    opcode_in(encoding, Slot::Y, word, available),
                ) else {
                    noise.note_truncated(encoding.bits);
                    continue;
                };
                let mut xs = Accum::default();
                let mut ys = Accum::default();
                self.evaluate(
                    encoding,
                    candidates(encoding, Slot::X, opx),
                    word,
                    available,
                    &mut xs,
                    &mut noise,
                );
                self.evaluate(
                    encoding,
                    candidates(encoding, Slot::Y, opy),
                    word,
                    available,
                    &mut ys,
                    &mut noise,
                );
                // A half with an inherent literal names the extended layout,
                // while its peer can name the base layout. The XML identifiers
                // enumerate individual opcodes, so the base identifier need not
                // match the extended half's opcode. Resolve the peer from a
                // physically identical prefix, never from a different layout.
                let needs_peer = xs.best.is_some() || ys.best.is_some();
                for base in self.encodings.iter().filter(|base| {
                    needs_peer
                        && base.is_pair()
                        && base.bits < encoding.bits
                        && base
                            .fields
                            .iter()
                            .all(|f| encoding.field(f.name) == Some(f))
                }) {
                    self.evaluate(
                        base,
                        candidates(base, Slot::X, opx),
                        word,
                        available,
                        &mut xs,
                        &mut noise,
                    );
                    self.evaluate(
                        base,
                        candidates(base, Slot::Y, opy),
                        word,
                        available,
                        &mut ys,
                        &mut noise,
                    );
                }
                if xs.conflict || ys.conflict {
                    ambiguous_any = true;
                    ambiguous.extend(xs.all);
                    ambiguous.extend(ys.all);
                } else if let (Some(x), Some(y)) = (xs.best, ys.best) {
                    pair_count += 1;
                    if first_pair.is_none() {
                        first_pair = Some((encoding, x, y));
                    }
                    if collect {
                        pair_rows.push(x);
                        pair_rows.push(y);
                    }
                } else if xs.best.is_some() || ys.best.is_some() {
                    partial_any = true;
                    if let Some(x) = xs.best {
                        partial_x.push(x, collect);
                    }
                    if let Some(y) = ys.best {
                        partial_y.push(y, collect);
                    }
                }
            } else {
                let Some(opcode) = opcode_in(encoding, Slot::Whole, word, available) else {
                    noise.note_truncated(encoding.bits);
                    continue;
                };
                self.evaluate(
                    encoding,
                    candidates(encoding, Slot::Whole, opcode),
                    word,
                    available,
                    &mut singles,
                    &mut noise,
                );
            }
        }
        if let Some(needed_bits) = noise.truncated {
            return Err(SelectError::Truncated {
                needed_bits,
                available_bits: available,
            });
        }
        if ambiguous_any {
            return Err(SelectError::Ambiguous {
                candidates: ambiguous,
            });
        }
        if let Some(best) = singles.best {
            if pair_count > 0 {
                let mut candidates = singles.all;
                candidates.extend(pair_rows);
                return Err(SelectError::Ambiguous { candidates });
            }
            if !singles.any_compute {
                let mut encodings: Vec<&'static str> =
                    singles.all.iter().map(|r| r.encoding).collect();
                encodings.sort_unstable();
                encodings.dedup();
                return Err(SelectError::UnsupportedGraphics {
                    encodings,
                    rows: singles.all,
                });
            }
            if singles.conflict {
                return Err(SelectError::Ambiguous {
                    candidates: singles.all,
                });
            }
            return Ok(Selection::Single(best));
        }
        match (pair_count, first_pair) {
            (1, Some((encoding, x, y))) => {
                let literal_marker = encoding
                    .field("SRCY0")
                    .is_some_and(|f| f.end_bit() <= available && f.extract(word) == 255);
                if literal_marker && encoding.field("LITERAL").is_none() {
                    return Err(SelectError::Unmodeled {
                        encoding: encoding.name,
                        reason: "VOPD SRCY0 is the literal marker, but the XML conditions only inspect SRCX0 and select an encoding without a literal dword",
                    });
                }
                return Ok(Selection::Vopd(VopdPair { encoding, x, y }));
            }
            (0, _) => {}
            _ => {
                return Err(SelectError::Ambiguous {
                    candidates: pair_rows,
                });
            }
        }
        if partial_any {
            return Err(SelectError::VopdSlots {
                x: partial_x.all,
                y: partial_y.all,
            });
        }
        if noise.refused_any {
            return Err(SelectError::Refused {
                refusals: noise.refused,
            });
        }
        if matched_any {
            Err(SelectError::NoRow { encodings: matched })
        } else {
            Err(SelectError::NoEncoding { word })
        }
    }

    /// Assemble the word of `row` from named fields. The identifier and opcode
    /// are seeded from the catalog; the assembled word must still match the
    /// encoding identifier and satisfy the row's condition. A successful
    /// encode does not allocate.
    pub fn encode_row(&self, row: &Row, fields: &[(&str, u128)]) -> Result<u128, EncodeError> {
        if let Some(refusal) = row.refusal() {
            return Err(EncodeError::Refused(refusal));
        }
        let encoding = self.encoding_of(row);
        let mut word = encoding.seed(row).ok_or(EncodeError::Identifier {
            instruction: row.name,
            encoding: row.encoding,
            opcode: row.opcode,
        })?;
        let opcode_field = encoding.opcode_field(row.slot).map(|f| f.name);
        for &(name, value) in fields {
            if let Some(opcode_name) = opcode_field {
                if name == opcode_name && value != u128::from(row.opcode) {
                    return Err(EncodeError::OpcodeField {
                        instruction: row.name,
                        field: opcode_name,
                        opcode: row.opcode,
                    });
                }
            }
            let field = encoding
                .field(name)
                .ok_or_else(|| LayoutError::UnknownField(name.to_owned()))?;
            word = field.insert(word, value)?;
        }
        if !encoding.identifier_matches(word) {
            return Err(EncodeError::IdentifierMismatch {
                instruction: row.name,
                encoding: row.encoding,
                word,
            });
        }
        let expr = self
            .predicate_of(row)
            .expect("a row that is not refused has a predicate");
        match encoding.eval_expr(expr, word, encoding.bits) {
            Ok(true) => Ok(word),
            Ok(false) => Err(EncodeError::PredicateFalse {
                instruction: row.name,
                encoding: row.encoding,
                condition: row.condition,
            }),
            Err(error) => Err(EncodeError::Eval {
                instruction: row.name,
                error,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Shipped catalogs
// ---------------------------------------------------------------------------

fn load(text: &'static str) -> Catalog {
    Catalog::parse(text, Sop1Evidence::shipped()).expect("shipped RDNA XML catalog must parse")
}

/// The XML catalog of `arch` (`gfx1200` shares the gfx12 catalog with
/// `gfx1201`); `None` for architectures without one.
pub fn catalog(arch: Arch) -> Option<&'static Catalog> {
    static GFX1100: LazyLock<Catalog> =
        LazyLock::new(|| load(include_str!("../../isa/gfx1100.xml.jsonl")));
    static GFX1151: LazyLock<Catalog> =
        LazyLock::new(|| load(include_str!("../../isa/gfx1151.xml.jsonl")));
    static GFX12: LazyLock<Catalog> =
        LazyLock::new(|| load(include_str!("../../isa/gfx12.xml.jsonl")));
    match arch {
        Arch::Gfx1100 => Some(&GFX1100),
        Arch::Gfx1151 => Some(&GFX1151),
        Arch::Gfx1200 | Arch::Gfx1201 => Some(&GFX12),
        Arch::Gfx1010 | Arch::Gfx1030 => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! lit {
        ($v:literal) => {
            concat!(
                r#"{"attributes":{"val":""#,
                $v,
                r#""},"children":[],"tag":"lit","text":""}"#
            )
        };
    }
    macro_rules! id {
        ($v:literal) => {
            concat!(
                r#"{"attributes":{"val":""#,
                $v,
                r#""},"children":[],"tag":"id","text":""}"#
            )
        };
    }
    macro_rules! op {
        ($t:literal, $a:expr, $b:expr) => {
            concat!(
                r#"{"attributes":{"type":""#,
                $t,
                r#""},"children":["#,
                $a,
                ",",
                $b,
                r#"],"tag":"op","text":""}"#
            )
        };
    }
    macro_rules! read {
        ($f:literal) => {
            op!(
                ".fieldderef",
                op!(".fieldderef", id!("INST"), id!("FMT")),
                id!($f)
            )
        };
    }
    macro_rules! array {
        () => {
            op!(".cons_array", lit!("250"), lit!("255"))
        };
    }
    macro_rules! cond {
        ($e:expr) => {
            concat!(
                r#"{"attributes":{},"children":["#,
                $e,
                r#"],"tag":"CondtionExpression","text":""}"#
            )
        };
    }

    /// Fields: A=3, B=255, C=16.
    fn holds(json: &'static str) -> Result<bool, EvalError> {
        let ConditionBody::Expr(expr) = parse_condition_json(json).unwrap() else {
            panic!("expected an expression");
        };
        expr.holds(&|name| match name {
            "A" => Ok(3),
            "B" => Ok(255),
            "C" => Ok(16),
            other => Err(EvalError::UnknownField { field: other }),
        })
    }

    #[test]
    fn expression_operators_follow_the_catalog_inventory() {
        assert_eq!(holds(cond!(lit!("1"))), Ok(true));
        assert_eq!(holds(cond!(lit!("0x0"))), Ok(false));
        assert_eq!(holds(cond!(op!("==", read!("B"), lit!("255")))), Ok(true));
        assert_eq!(holds(cond!(op!("!=", read!("B"), lit!("255")))), Ok(false));
        assert_eq!(
            holds(cond!(op!(
                "&&",
                op!("==", read!("A"), lit!("3")),
                op!("!=", read!("C"), lit!("0"))
            ))),
            Ok(true)
        );
        assert_eq!(
            holds(cond!(op!(
                "&&",
                op!("==", read!("A"), lit!("3")),
                op!("==", read!("C"), lit!("0"))
            ))),
            Ok(false)
        );
        assert_eq!(holds(cond!(op!(".within", read!("B"), array!()))), Ok(true));
        assert_eq!(
            holds(cond!(op!(".notwithin", read!("B"), array!()))),
            Ok(false)
        );
        assert_eq!(
            holds(cond!(op!(".notwithin", read!("A"), array!()))),
            Ok(true)
        );
    }

    #[test]
    fn expressions_never_mask_undefined_references() {
        // `&&` evaluates every operand, so a false first operand cannot hide
        // an unknown field in the second.
        assert_eq!(
            holds(cond!(op!(
                "&&",
                op!("==", read!("A"), lit!("9")),
                op!("==", read!("MISSING"), lit!("0"))
            ))),
            Err(EvalError::UnknownField { field: "MISSING" })
        );
        assert!(matches!(
            holds(cond!(op!("==", id!("OTHER"), lit!("0")))),
            Err(EvalError::Unbound("OTHER"))
        ));
        assert!(matches!(
            holds(cond!(op!(".within", read!("A"), lit!("3")))),
            Err(EvalError::Type { op: ".within", .. })
        ));
    }

    #[test]
    fn unknown_operators_and_undefined_conditions_refuse() {
        let unsupported = parse_condition_json(cond!(op!("<", read!("A"), lit!("3")))).unwrap();
        assert!(matches!(&unsupported, ConditionBody::Unsupported(r) if r.contains('<')));
        assert_eq!(
            parse_condition_json("null").unwrap(),
            ConditionBody::Undefined
        );
        let negative = parse_condition_json(cond!(lit!("-1"))).unwrap();
        assert!(matches!(negative, ConditionBody::Unsupported(_)));
    }

    macro_rules! row {
        ($name:literal, $enc:literal, $cond:literal, $op:literal, $var:literal, $status:literal) => {
            concat!(
                r#"{"aliases":[],"codec_status":""#,
                $status,
                r#"","condition":""#,
                $cond,
                r#"","defs":[],"encoding":""#,
                $enc,
                r#"","flags":{"IsBranch":false,"IsConditionalBranch":false,"IsImmediatelyExecuted":false,"IsIndirectBranch":false,"IsProgramTerminator":false},"group":"G","implicit":[],"memory":[],"name":""#,
                $name,
                r#"","opcode":"#,
                $op,
                r#","operands":[],"oracle":{"status":"verified"},"subgroups":[],"uses":[],"variant":"#,
                $var,
                "}"
            )
        };
    }

    macro_rules! layout32 {
        ($conditions:expr, $ids:literal) => {
            concat!(
                r#"{"bits":32,"conditions":{"#,
                $conditions,
                r#"},"fields":{"ENCODING":[[24,8]],"OP":[[16,8]],"SRC0":[[0,8]]},"identifier_mask":"0xffff0000","identifiers":["#,
                $ids,
                "]}"
            )
        };
    }

    macro_rules! synthetic {
        () => {
            concat!(
                r#"{"encodings":{"#,
                r#""ENC_A":"#,
                layout32!(
                    concat!(
                        r#""default":"#,
                        cond!(op!("!=", read!("SRC0"), lit!("255"))),
                        r#","dangling":null"#
                    ),
                    r#""0xa0010000","0xa0020000","0xa0030000""#
                ),
                r#","ENC_A_LIT":{"bits":64,"conditions":{"has_lit":"#,
                cond!(op!("==", read!("SRC0"), lit!("255"))),
                r#"},"fields":{"ENCODING":[[24,8]],"LITERAL":[[32,32]],"OP":[[16,8]],"SRC0":[[0,8]]},"identifier_mask":"0xffff0000","identifiers":["0xa0010000","0xa0020000","0xa0030000"]}"#,
                r#","ENC_G":"#,
                layout32!(
                    concat!(r#""default":"#, cond!(lit!("1"))),
                    r#""0xb0000000""#
                ),
                r#"},"instructions":5,"key":["name","encoding","condition","opcode","variant"],"llvm":"l","llvm_sha256":"0","rows":6,"schema":1,"sha256":"abc","target":"tgt","xml":"x.xml"}"#,
                "\n",
                row!("t_add", "ENC_A", "default", 1, 0, "metadata-only"),
                "\n",
                row!("t_add", "ENC_A_LIT", "has_lit", 1, 1, "metadata-only"),
                "\n",
                row!("t_alias_a", "ENC_A", "default", 2, 0, "metadata-only"),
                "\n",
                row!("t_alias_b", "ENC_A", "default", 2, 0, "metadata-only"),
                "\n",
                row!("t_dangling", "ENC_A", "dangling", 3, 0, "metadata-only"),
                "\n",
                row!("t_img", "ENC_G", "default", 0, 0, "unsupported-graphics"),
                "\n"
            )
        };
    }

    macro_rules! evidence {
        ($status:literal, $predicate:literal, $reason:literal) => {
            concat!(
                r#"{"forms":{"register":0},"literal":1,"llvm":"l","llvm_sha256":"0","predicates":{"ne255":{"field":"SRC0","op":"!=","value":255}},"schema":1,"targets":{"tgt":{"cpu":"c","rows":[{"corrected_predicate":"#,
                $predicate,
                r#","key":{"condition":"dangling","encoding":"ENC_A","name":"t_dangling","opcode":3,"variant":0},"name":"t_dangling","proofs":{"register":{"candidate":"00","decoded":["t"],"ssrc0":0,"status":"verified"}},"status":""#,
                $status,
                r#"","unavailable_reason":"#,
                $reason,
                r#","xml_condition":"dangling","xml_predicate":null}],"xml":"x.xml","xml_sha256":"abc"}},"xml_condition":"dangling"}"#
            )
        };
    }

    fn synthetic_catalog(evidence: &Sop1Evidence) -> Catalog {
        Catalog::parse(synthetic!(), evidence).expect("synthetic catalog parses")
    }

    fn selected<'c>(catalog: &'c Catalog, words: &[u32]) -> &'c Row {
        match catalog.select(words) {
            Ok(Selection::Single(row)) => row,
            other => panic!("expected a single row, got {other:?}"),
        }
    }

    #[test]
    fn synthetic_catalog_keeps_variant_keys_and_selects_by_condition() {
        let catalog = synthetic_catalog(&Sop1Evidence::empty());
        assert_eq!(catalog.rows().len(), 6);
        assert_eq!(catalog.instructions(), 5);
        let register = catalog
            .row_by_key("t_add", "ENC_A", "default", 1, 0)
            .unwrap();
        let literal = catalog
            .row_by_key("t_add", "ENC_A_LIT", "has_lit", 1, 1)
            .unwrap();
        assert_eq!((register.variant, literal.variant), (0, 1));
        assert!(catalog
            .row_by_key("t_add", "ENC_A", "default", 1, 1)
            .is_none());
        assert_eq!(selected(&catalog, &[0xa001_0005]).key(), register.key());
        assert_eq!(
            selected(&catalog, &[0xa001_00ff, 0x1234_5678]).key(),
            literal.key()
        );
        assert!(matches!(
            catalog.select(&[0xa001_00ff]),
            Err(SelectError::Truncated {
                needed_bits: 64,
                available_bits: 32
            })
        ));
        assert!(matches!(
            catalog.select(&[0x1234_5678]),
            Err(SelectError::NoEncoding { .. })
        ));
        assert!(matches!(
            catalog.select(&[0xa009_0005]),
            Err(SelectError::NoEncoding { .. })
        ));
    }

    #[test]
    fn ambiguous_rows_are_reported_not_first_matched() {
        let catalog = synthetic_catalog(&Sop1Evidence::empty());
        match catalog.select(&[0xa002_0005]) {
            Err(SelectError::Ambiguous { candidates }) => {
                let mut names: Vec<_> = candidates.iter().map(|r| r.name).collect();
                names.sort_unstable();
                assert_eq!(names, ["t_alias_a", "t_alias_b"]);
            }
            other => panic!("expected an ambiguity, got {other:?}"),
        }
    }

    #[test]
    fn graphics_rows_are_unsupported_by_name() {
        let catalog = synthetic_catalog(&Sop1Evidence::empty());
        assert_eq!(catalog.unsupported_graphics_encodings(), ["ENC_G"]);
        let row = catalog.rows_named("t_img").next().unwrap();
        assert_eq!(
            row.refusal(),
            Some(Refusal::UnsupportedGraphics {
                instruction: "t_img",
                encoding: "ENC_G"
            })
        );
        match catalog.select(&[0xb000_0000]) {
            Err(SelectError::UnsupportedGraphics { encodings, rows }) => {
                assert_eq!(encodings, ["ENC_G"]);
                assert_eq!(rows[0].name, "t_img");
            }
            other => panic!("expected a graphics refusal, got {other:?}"),
        }
        assert!(matches!(
            catalog.encode_row(row, &[]),
            Err(EncodeError::Refused(Refusal::UnsupportedGraphics { .. }))
        ));
    }

    #[test]
    fn dangling_conditions_refuse_without_evidence() {
        let catalog = synthetic_catalog(&Sop1Evidence::empty());
        let row = catalog.rows_named("t_dangling").next().unwrap();
        assert!(matches!(
            row.refusal(),
            Some(Refusal::UndefinedCondition {
                condition: "dangling",
                ..
            })
        ));
        match catalog.select(&[0xa003_0005]) {
            Err(SelectError::Refused { refusals }) => {
                assert_eq!(refusals.len(), 1);
                assert!(matches!(
                    *refusals[0].refusal,
                    Refusal::UndefinedCondition { .. }
                ));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        assert_eq!(
            catalog.encoding("ENC_A").unwrap().holds("dangling", 0, 32),
            Err(ConditionError::Undefined {
                condition: "dangling".to_owned()
            })
        );
        assert!(matches!(
            catalog.validate_sop1_source("t_dangling", 5),
            Err(Sop1SourceError::NoEvidence { .. })
        ));
    }

    #[test]
    fn evidence_corrects_or_names_the_unavailable_reason() {
        let corrected = Sop1Evidence::parse(evidence!("corrected", r#""ne255""#, "null")).unwrap();
        let catalog = synthetic_catalog(&corrected);
        let row = selected(&catalog, &[0xa003_0005]);
        assert_eq!(row.name, "t_dangling");
        assert!(matches!(row.resolution(), Resolution::Corrected(c) if c.predicate == "ne255"));
        // The corrected predicate is SRC0 != 255, so the literal marker selects nothing.
        assert!(matches!(
            catalog.select(&[0xa003_00ff, 0]),
            Err(SelectError::NoRow { .. })
        ));
        // The catalog still records the original dangling XML condition.
        assert_eq!(row.condition, "dangling");
        assert_eq!(
            catalog
                .encoding("ENC_A")
                .unwrap()
                .condition("dangling")
                .unwrap()
                .body,
            ConditionBody::Undefined
        );
        // Source validation: normal split, but the literal form has no proof here.
        assert_eq!(catalog.validate_sop1_source("t_dangling", 5), Ok(()));
        assert_eq!(
            catalog.validate_sop1_source("t_dangling", 255),
            Err(Sop1SourceError::LiteralUnavailable {
                instruction: "t_dangling"
            })
        );
        assert!(matches!(
            catalog.validate_sop1_source("t_dangling", 256),
            Err(Sop1SourceError::OutOfRange { .. })
        ));

        let unavailable =
            Sop1Evidence::parse(evidence!("unavailable", "null", r#""llvm-unknown""#)).unwrap();
        let catalog = synthetic_catalog(&unavailable);
        match catalog.select(&[0xa003_0005]) {
            Err(SelectError::Refused { refusals }) => assert!(matches!(
                *refusals[0].refusal,
                Refusal::NoCorrection {
                    reason: "llvm-unknown",
                    ..
                }
            )),
            other => panic!("expected a named refusal, got {other:?}"),
        }
        assert_eq!(
            catalog.validate_sop1_source("t_dangling", 5),
            Err(Sop1SourceError::Unavailable {
                instruction: "t_dangling",
                reason: "llvm-unknown"
            })
        );
    }

    #[test]
    fn evidence_for_a_defined_condition_is_rejected() {
        let evidence = Sop1Evidence {
            targets: vec![TargetEvidence {
                target: "tgt",
                cpu: "c",
                xml: "x.xml",
                xml_sha256: "abc",
                rows: vec![RowEvidence {
                    key: RowKey {
                        name: "t_add",
                        encoding: "ENC_A",
                        condition: "default",
                        opcode: 1,
                        variant: 0,
                    },
                    outcome: EvidenceOutcome::Unavailable { reason: "x" },
                    proofs: Vec::new(),
                    domain: None,
                }],
            }],
            ..Sop1Evidence::default()
        };
        assert!(matches!(
            Catalog::parse(synthetic!(), &evidence),
            Err(CatalogError::Evidence(_))
        ));
    }

    #[test]
    fn evidence_checks_admit_selectors_by_domain() {
        let domain = EvidenceCheck::InRanges(vec![(0, 3), (125, 125), (128, 130)]);
        assert!(domain.admits(0) && domain.admits(125) && domain.admits(130));
        assert!(!domain.admits(4) && !domain.admits(131) && !domain.admits(255));
        assert!(EvidenceCheck::Ne(255).admits(0) && !EvidenceCheck::Ne(255).admits(255));
        assert!(EvidenceCheck::Eq(0).admits(0) && !EvidenceCheck::Eq(0).admits(1));
        assert!(ranges(&[[5, 4]], "t").is_err());
        assert!(ranges(&[[0, 3], [3, 4]], "t").is_err());
        assert!(ranges(&[[0, 512]], "t").is_err());
        assert_eq!(ranges(&[[0, 3], [5, 5]], "t"), Ok(vec![(0, 3), (5, 5)]));
    }

    #[test]
    fn fields_split_across_ranges_round_trip() {
        let field = Field {
            name: "OPSEL_HI",
            ranges: vec![
                BitRange {
                    offset: 59,
                    width: 2,
                },
                BitRange {
                    offset: 14,
                    width: 1,
                },
            ],
        };
        // Bit 59 is the least significant bit of the value, bit 14 the most.
        assert_eq!(field.insert(0, 0b001).unwrap(), 1u128 << 59);
        assert_eq!(field.insert(0, 0b010).unwrap(), 1u128 << 60);
        assert_eq!(field.insert(0, 0b100).unwrap(), 1u128 << 14);
        for value in 0..8u128 {
            let word = field.insert(u128::MAX >> 40, value).unwrap();
            assert_eq!(field.extract(word), value);
        }
        assert!(matches!(
            field.insert(0, 8),
            Err(LayoutError::ValueTooWide { width: 3, .. })
        ));
    }

    #[test]
    fn layouts_reach_128_bits() {
        let field = Field {
            name: "WIDE",
            ranges: vec![BitRange {
                offset: 0,
                width: 128,
            }],
        };
        let value = 0x0123_4567_89ab_cdef_0011_2233_4455_6677u128;
        assert_eq!(field.extract(field.insert(0, value).unwrap()), value);
        assert_eq!(
            pack_words(&[1, 2, 3, 4, 5]),
            1 | 2 << 32 | 3 << 64 | 4 << 96
        );
    }

    #[test]
    fn encode_row_reconstructs_words() {
        let catalog = synthetic_catalog(&Sop1Evidence::empty());
        let row = catalog
            .row_by_key("t_add", "ENC_A", "default", 1, 0)
            .unwrap();
        assert_eq!(
            catalog.encode_row(row, &[("SRC0", 5)]).unwrap(),
            0xa001_0005
        );
        // The condition is SRC0 != 255.
        assert!(matches!(
            catalog.encode_row(row, &[("SRC0", 255)]),
            Err(EncodeError::PredicateFalse { .. })
        ));
        assert!(matches!(
            catalog.encode_row(row, &[("OP", 9)]),
            Err(EncodeError::OpcodeField { .. })
        ));
        assert!(matches!(
            catalog.encode_row(row, &[("NOPE", 1)]),
            Err(EncodeError::Layout(LayoutError::UnknownField(_)))
        ));
        let literal = catalog
            .row_by_key("t_add", "ENC_A_LIT", "has_lit", 1, 1)
            .unwrap();
        let word = catalog
            .encode_row(literal, &[("SRC0", 255), ("LITERAL", 0xdead_beef)])
            .unwrap();
        let encoding = catalog.encoding_of(literal);
        assert_eq!(encoding.to_dwords(word), [0xa001_00ff, 0xdead_beef]);
        let fields = encoding.decompose(word);
        assert_eq!(encoding.compose(&fields).unwrap(), word);
    }

    // ----- shipped catalogs -------------------------------------------------

    #[test]
    fn all_three_shipped_catalogs_parse_with_unique_full_keys() {
        for (arch, rows) in [
            (Arch::Gfx1100, 4929),
            (Arch::Gfx1151, 5056),
            (Arch::Gfx1201, 5031),
        ] {
            let catalog = catalog(arch).unwrap();
            assert_eq!(catalog.rows().len(), rows);
            for row in catalog.rows() {
                let found = catalog
                    .row_by_key(
                        row.name,
                        row.encoding,
                        row.condition,
                        row.opcode,
                        row.variant,
                    )
                    .unwrap();
                assert!(std::ptr::eq(found, row), "{:?}", row.key());
            }
        }
        assert!(std::ptr::eq(
            catalog(Arch::Gfx1200).unwrap(),
            catalog(Arch::Gfx1201).unwrap()
        ));
        assert!(catalog(Arch::Gfx1010).is_none());
    }

    #[test]
    fn vopd_variants_keep_their_x_and_y_halves() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        let key = |encoding, condition, variant| {
            catalog
                .row_by_key("v_dual_fmac_f32", encoding, condition, 0, variant)
                .unwrap()
        };
        assert_eq!(key("VOPDXY", "default", 0).slot(), Slot::X);
        assert_eq!(key("VOPDXY_INST_LITERAL", "has_lit", 1).slot(), Slot::X);
        assert_eq!(key("VOPDXY", "default", 2).slot(), Slot::Y);
        assert_eq!(key("VOPDXY_INST_LITERAL", "has_lit", 3).slot(), Slot::Y);
        // Y-only opcodes exceed the 4-bit OPX field but fit OPY.
        let lshlrev = catalog
            .row_by_key("v_dual_lshlrev_b32", "VOPDXY", "default", 17, 0)
            .unwrap();
        assert_eq!(lshlrev.slot(), Slot::Y);
        assert_eq!(lshlrev.opcode_field(catalog).unwrap().name, "OPY");
    }

    #[test]
    fn vopd_overflow_identifiers_are_quarantined_by_the_structural_rule() {
        for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201] {
            let catalog = catalog(arch).unwrap();
            for name in ["VOPDXY", "VOPDXY_INST_LITERAL"] {
                let encoding = catalog.encoding(name).unwrap();
                assert_eq!(
                    encoding.quarantined,
                    [0xcc00_0000, 0xcc40_0000, 0xcc80_0000]
                );
                assert!(encoding.identifiers.contains(&0xcc00_0000));
                assert!(!encoding.identifier_matches(0xcc00_0000));
                assert!(encoding.identifier_matches(0xc8c0_0000));
            }
            // No other encoding quarantines anything.
            assert!(catalog
                .encodings()
                .iter()
                .filter(|e| !e.name.starts_with("VOPDXY"))
                .all(|e| e.quarantined.is_empty()));
        }
    }

    fn vopd_words(catalog: &Catalog, fields: &[(&str, u128)], dwords: usize) -> Vec<u32> {
        let encoding = catalog.encoding("VOPDXY").unwrap();
        let mut words = encoding.to_dwords(encoding.compose(fields).unwrap());
        words.resize(dwords, 0);
        words
    }

    const VOPD_BASE: [(&str, u128); 9] = [
        ("ENCODING", 0x32),
        ("OPX", 3),
        ("OPY", 17),
        ("SRCX0", 256),
        ("SRCY0", 257),
        ("VSRCX1", 1),
        ("VSRCY1", 2),
        ("VDSTX", 0),
        ("VDSTY", 1),
    ];

    fn with<'a>(fields: &[(&'a str, u128)], name: &'a str, value: u128) -> Vec<(&'a str, u128)> {
        fields
            .iter()
            .map(|&(n, v)| if n == name { (n, value) } else { (n, v) })
            .collect()
    }

    #[test]
    fn vopd_pairs_select_both_halves() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        let words = vopd_words(catalog, &VOPD_BASE, 2);
        match catalog.select(&words) {
            Ok(Selection::Vopd(pair)) => {
                assert_eq!(pair.encoding.name, "VOPDXY");
                assert_eq!((pair.x.name, pair.x.slot()), ("v_dual_mul_f32", Slot::X));
                assert_eq!(
                    (pair.y.name, pair.y.slot()),
                    ("v_dual_lshlrev_b32", Slot::Y)
                );
            }
            other => panic!("expected a VOPD pair, got {other:?}"),
        }
    }

    #[test]
    fn vopd_halves_share_a_compatible_extended_packet() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        // X = mul uses the base layout; Y = fmaak requires a literal.
        // The shared fields coincide physically, so the packet uses 96 bits.
        let words = vopd_words(catalog, &with(&VOPD_BASE, "OPY", 1), 3);
        match catalog.select(&words) {
            Ok(Selection::Vopd(pair)) => {
                assert_eq!(pair.x.name, "v_dual_mul_f32");
                assert_eq!(pair.y.name, "v_dual_fmaak_f32");
                assert_eq!(pair.encoding.bits, 96);
            }
            other => panic!("expected compatible split slots, got {other:?}"),
        }
    }

    #[test]
    fn vopd_y_literal_marker_without_literal_dword_is_unmodeled() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        let fields = with(&with(&VOPD_BASE, "OPY", 3), "SRCY0", 255);
        let words = vopd_words(catalog, &fields, 2);
        assert!(matches!(
            catalog.select(&words),
            Err(SelectError::Unmodeled {
                encoding: "VOPDXY",
                ..
            })
        ));
    }

    #[test]
    fn sop1_evidence_makes_dangling_rows_selectable() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        // s_mov_b32 s0, s0 / s_mov_b32 s0, 0x41234567 (the evidence probes).
        let register = selected(catalog, &[0xbe80_0000]);
        assert_eq!(
            (register.name, register.encoding),
            ("s_mov_b32", "ENC_SOP1")
        );
        assert!(matches!(
            register.resolution(),
            Resolution::Corrected(c) if c.predicate == "ssrc0-not-literal"
        ));
        let literal = selected(catalog, &[0xbe80_00ff, 0x4123_4567]);
        assert_eq!(
            (literal.name, literal.encoding),
            ("s_mov_b32", "SOP1_INST_LITERAL")
        );
        // s_getpc_b64 reads no source: only SSRC0 == 0 is proven.
        let getpc = selected(catalog, &[0xbe80_4700]);
        assert_eq!(getpc.name, "s_getpc_b64");
        assert!(matches!(
            catalog.select(&[0xbe80_4705]),
            Err(SelectError::Refused { .. })
        ));
        // s_movrels_b32 has a proven selector domain (SGPRs 0-127 only).
        assert_eq!(selected(catalog, &[0xbe80_4003]).name, "s_movrels_b32");
        assert!(catalog.select(&[0xbe80_4082]).is_err());
        assert!(catalog.select(&[0xbe80_40ff, 0]).is_err());
        // No dangling ENC_SOP1 row is left undefined.
        for row in catalog
            .rows()
            .iter()
            .filter(|r| r.encoding == "ENC_SOP1" && r.condition == "Nothas_lit_0_Nothas_lit_1")
        {
            assert!(
                matches!(row.resolution(), Resolution::Corrected(_)),
                "{:?}",
                row.key()
            );
        }
        // The catalog still carries the original dangling condition.
        assert_eq!(
            catalog
                .encoding("ENC_SOP1")
                .unwrap()
                .condition("Nothas_lit_0_Nothas_lit_1")
                .unwrap()
                .body,
            ConditionBody::Undefined
        );
    }

    #[test]
    fn sop1_source_validation_applies_the_instruction_domain() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        // Normal split: any selector, including the literal marker (proven).
        for selector in [0, 5, 128, 255] {
            assert_eq!(catalog.validate_sop1_source("s_mov_b32", selector), Ok(()));
        }
        assert!(matches!(
            catalog.validate_sop1_source("s_mov_b32", 256),
            Err(Sop1SourceError::OutOfRange { .. })
        ));
        // Exceptional rows: only the proven selectors, on both records.
        assert_eq!(catalog.validate_sop1_source("s_getpc_b64", 0), Ok(()));
        for selector in [5, 255] {
            assert!(matches!(
                catalog.validate_sop1_source("s_getpc_b64", selector),
                Err(Sop1SourceError::OutsideDomain { .. })
            ));
        }
        assert_eq!(catalog.validate_sop1_source("s_movrels_b32", 127), Ok(()));
        assert!(matches!(
            catalog.validate_sop1_source("s_movrels_b32", 128),
            Err(Sop1SourceError::OutsideDomain { .. })
        ));
        assert!(matches!(
            catalog.validate_sop1_source("not_an_instruction", 0),
            Err(Sop1SourceError::NoEvidence { .. })
        ));
    }

    #[test]
    fn shipped_graphics_encodings_are_named() {
        let gfx12 = catalog(Arch::Gfx1201).unwrap();
        assert_eq!(
            gfx12.unsupported_graphics_encodings(),
            ["ENC_VEXPORT", "ENC_VIMAGE", "ENC_VINTERP", "ENC_VSAMPLE"]
        );
        assert_eq!(
            gfx12.rows_named("export").next().unwrap().refusal(),
            Some(Refusal::UnsupportedGraphics {
                instruction: "export",
                encoding: "ENC_VEXPORT"
            })
        );
        assert_eq!(
            catalog(Arch::Gfx1100)
                .unwrap()
                .unsupported_graphics_encodings(),
            ["ENC_EXP", "ENC_MIMG", "ENC_VINTERP", "MIMG_NSA1"]
        );
    }

    #[test]
    fn identical_duplicate_records_collapse_to_the_lowest_variant() {
        let catalog = catalog(Arch::Gfx1100).unwrap();
        let first = catalog
            .row_by_key("global_load_u8", "ENC_FLAT_GLOBAL", "default", 16, 0)
            .unwrap();
        let second = catalog
            .row_by_key("global_load_u8", "ENC_FLAT_GLOBAL", "default", 16, 1)
            .unwrap();
        assert!(first.equivalent_to(second));
        assert_eq!(catalog.equivalents(first).count(), 2);
        let word = catalog.encode_row(first, &[]).unwrap();
        let encoding = catalog.encoding_of(first);
        assert_eq!(
            selected(catalog, &encoding.to_dwords(word)).key(),
            first.key()
        );
    }

    #[test]
    fn distinct_operands_are_never_equivalent() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        let x = catalog
            .row_by_key("v_dual_fmac_f32", "VOPDXY", "default", 0, 0)
            .unwrap();
        let y = catalog
            .row_by_key("v_dual_fmac_f32", "VOPDXY", "default", 0, 2)
            .unwrap();
        assert!(!x.equivalent_to(y));
    }

    #[test]
    fn every_verified_candidate_reconstructs_and_selects_its_own_row() {
        let catalog = catalog(Arch::Gfx1201).unwrap();
        for row in catalog.rows() {
            let Some(hex) = row.oracle.candidate.filter(|_| row.oracle.verified()) else {
                continue;
            };
            let encoding = catalog.encoding_of(row);
            if encoding.is_pair() {
                continue;
            }
            let bytes: Vec<u8> = (0..hex.len() / 2)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
                .collect();
            let words: Vec<u32> = bytes
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(words.len() as u32 * 32, encoding.bits, "{:?}", row.key());
            let word = pack_words(&words);
            assert_eq!(
                encoding.compose(&encoding.decompose(word)).unwrap(),
                word,
                "{:?}",
                row.key()
            );
            match catalog.select(&words) {
                Ok(Selection::Single(found)) => {
                    assert!(found.equivalent_to(row), "{:?}", row.key())
                }
                Err(SelectError::UnsupportedGraphics { rows, .. }) => {
                    assert!(rows.iter().any(|r| r.key() == row.key()), "{:?}", row.key())
                }
                other => panic!("{:?}: {other:?}", row.key()),
            }
        }
    }
}
