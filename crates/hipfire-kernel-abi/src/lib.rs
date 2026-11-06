// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `KernelAbiV1`: one typed contract per kernel family that the PeaceMaker
//! generator metadata, the embedded code object, the `rdna-compute` launcher
//! (ordinary `kernelParams` and `KernargBlob` paths), `pm_loop` and railgun
//! nodes all agree on (PM/railgun bedrock R1).
//!
//! A contract states, for every export of the family:
//!
//! * the explicit kernarg fields `(name, offset, width, kind)` and the hidden
//!   block `(offset, width, kind)`, the logical explicit bytes and the
//!   object's kernarg segment size/alignment;
//! * the operands each pointer field carries: storage [`FormatId`], access,
//!   base alignment and byte extent as an [`Expr`] over the launch params;
//! * the admitted domain ([`DomainRuleV1`]) and the [`NumericContractId`];
//! * per export: arch, object identity, exact block, dynamic and static LDS
//!   and the grid as expressions.
//!
//! Two operations are frozen for consumers:
//!
//! * [`validate_invocation`]`(&KernelAbiV1, &InvocationV1) -> Result<ValidatedInvocationV1, AbiError>`
//!   rejects a wrong arch/export/dtype/format/alignment/extent/domain/grid/block/LDS
//!   **before enqueue** and encodes the explicit image once;
//! * [`encode_into`]`(&ValidatedInvocationV1, &mut [u8]) -> Result<(), AbiError>`
//!   copies that explicit image out (hidden words are the runtime's or, for
//!   railgun, `KernelProgram::encode_kernarg_into`'s; this crate never builds
//!   a second full railgun image).
//!
//! [`check_export`] compares a contract with plain-data evidence gathered from
//! a generator spec, an object's AMDGPU metadata or a railgun receipt and
//! returns every [`Drift`]; the evidence adapters live in dev-only tests of
//! crates that may depend on peacemaker. This crate depends on nothing GPU- or
//! peacemaker-side.
//!
//! Family contract data lives in [`families`].

mod check;
mod expr;
pub mod families;
mod validate;
pub mod work;

pub use check::{check_export, ArgEvidence, Drift, EvidenceSource, ExportEvidence};
pub use expr::eval;
pub use validate::{
    decode_params, encode_into, validate_invocation, DecodedV1, ValidatedInvocationV1,
    MAX_EXPLICIT_BYTES, MAX_FIELDS,
};

use serde::Serialize;

/// Exact target architecture. Near neighbours (gfx1200, gfx1150, ...) are
/// never aliases.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Arch {
    Gfx1100,
    Gfx1151,
    Gfx1201,
}

impl Arch {
    pub const ALL: [Arch; 3] = [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1201];

    pub fn name(self) -> &'static str {
        match self {
            Arch::Gfx1100 => "gfx1100",
            Arch::Gfx1151 => "gfx1151",
            Arch::Gfx1201 => "gfx1201",
        }
    }

    /// Exact-name parse; anything else (including `gfx1201:xnack-`) is `None`.
    pub fn parse(name: &str) -> Option<Self> {
        Arch::ALL.into_iter().find(|a| a.name() == name)
    }
}

/// Arithmetic/rounding/storage recipe an export implements (e.g. A8 x MQ6
/// integer dot, f32 fold, RNE-BF16 store). Exports of different numeric
/// contracts never share a [`KernelAbiV1`] even when their bytes agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct NumericContractId(pub &'static str);

/// Storage format and layout of one operand (codec, record size, row order).
/// Formats are compared by identity only: a same-sized buffer of another
/// format is rejected. Each family module defines its formats as `pub const`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct FormatId(pub &'static str);

/// Width/signedness of a scalar kernarg or launch param.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ScalarTy {
    /// Encoded little-endian in 4 bytes; value must be `<= i32::MAX`.
    I32,
    /// Encoded little-endian in 4 bytes; value must be `<= u32::MAX`.
    U32,
    /// Encoded little-endian in 8 bytes.
    U64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl Access {
    /// The AMDGPU metadata `.actual_access` spelling.
    pub fn metadata_name(self) -> &'static str {
        match self {
            Access::ReadOnly => "read_only",
            Access::WriteOnly => "write_only",
            Access::ReadWrite => "read_write",
        }
    }
}

/// Integer expression over the invocation params and the export constants.
/// Evaluation is checked: overflow, division by zero and an inexact
/// [`Expr::DivExact`] are errors, never wrapped values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Expr {
    Const(u64),
    /// A launch param ([`KernelAbiV1::params`]) or, when no param has the
    /// name, an export constant ([`ExportV1::consts`]).
    Param(&'static str),
    Add(&'static [Expr]),
    Mul(&'static [Expr]),
    /// `a - b`; negative is an error.
    Sub(&'static Expr, &'static Expr),
    /// `a / d`; `a % d != 0` is an error.
    DivExact(&'static Expr, u64),
    /// `ceil(a / d)`.
    DivCeil(&'static Expr, u64),
    /// `floor(a / d)`.
    Div(&'static Expr, u64),
}

/// One launch param (a logical dimension the caller supplies).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ParamV1 {
    pub name: &'static str,
    pub ty: ScalarTy,
    pub doc: &'static str,
}

/// One pointer operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct OperandV1 {
    pub name: &'static str,
    pub format: FormatId,
    pub access: Access,
    /// Required base-address alignment in bytes (power of two).
    pub align: u32,
    /// Bytes the kernel may touch from the base, over params and export
    /// constants. `None` only for operands whose reach is not a function of
    /// the launch params; [`OperandV1::note`] must then say what bounds it.
    pub extent: Option<Expr>,
    /// A null address is admitted (the kernel provably skips the operand).
    pub nullable: bool,
    /// Contract caveats, e.g. "entries point at expert slabs; their extent is
    /// the loader's contract, not checked here".
    pub note: &'static str,
}

/// What an explicit kernarg field carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldKind {
    /// 8-byte global pointer: the address of the named operand.
    Pointer { operand: &'static str },
    /// By-value scalar computed from params/constants (usually
    /// `Expr::Param(name)`; `Const(0)` for a reserved word).
    Scalar { ty: ScalarTy, value: Expr },
}

/// One explicit kernarg field, as the object's metadata lays it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct FieldV1 {
    /// The metadata `.name` of the argument.
    pub name: &'static str,
    pub offset: u32,
    pub width: u32,
    pub kind: FieldKind,
}

/// Launch axis of a hidden grid word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Hidden argument kinds (AMDGPU `.value_kind` `hidden_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenKind {
    BlockCount(Axis),
    GroupSize(Axis),
    Remainder(Axis),
    GlobalOffset(Axis),
    GridDims,
    DynamicLdsSize,
    /// Any other hidden kind by its exact metadata `.value_kind`
    /// (e.g. `hidden_multigrid_sync_arg`); its value is the runtime's.
    Other(&'static str),
}

impl HiddenKind {
    /// The metadata `.value_kind` spelling (`hidden_block_count_x`, ...).
    pub fn value_kind(self) -> String {
        let axis = |a: Axis| match a {
            Axis::X => "x",
            Axis::Y => "y",
            Axis::Z => "z",
        };
        match self {
            HiddenKind::BlockCount(a) => format!("hidden_block_count_{}", axis(a)),
            HiddenKind::GroupSize(a) => format!("hidden_group_size_{}", axis(a)),
            HiddenKind::Remainder(a) => format!("hidden_remainder_{}", axis(a)),
            HiddenKind::GlobalOffset(a) => format!("hidden_global_offset_{}", axis(a)),
            HiddenKind::GridDims => "hidden_grid_dims".into(),
            HiddenKind::DynamicLdsSize => "hidden_dynamic_lds_size".into(),
            HiddenKind::Other(kind) => kind.into(),
        }
    }
}

/// One hidden kernarg word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct HiddenV1 {
    pub offset: u32,
    pub width: u32,
    pub kind: HiddenKind,
}

/// A predicate over params/constants every admitted invocation satisfies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    /// `value > 0`.
    Positive(Expr),
    /// `value % m == 0`.
    MultipleOf(Expr, u64),
    /// `value <= limit`.
    AtMost(Expr, u64),
    /// `value >= limit`.
    AtLeast(Expr, u64),
    Equal(Expr, Expr),
    /// `value` is one of the listed constants.
    OneOf(Expr, &'static [u64]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DomainRuleV1 {
    pub rule: Rule,
    /// Why the kernel needs it (shown in the rejection).
    pub why: &'static str,
}

/// Identity of the object an export lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectRefV1 {
    /// A committed object `rdna-compute` embeds (`include_bytes!`).
    Embedded {
        /// Module name the launcher loads it under.
        module: &'static str,
        /// Repo-relative path of the committed object.
        artifact: &'static str,
        /// SHA-256 of the committed object bytes; a rebuilt object fails the
        /// drift check until the contract is reviewed and re-pinned.
        sha256: &'static str,
    },
    /// A hipcc-compiled module built from source at load. No offline object
    /// exists, so metadata evidence is unavailable; the contract still
    /// governs the launch encoding.
    HipccSource {
        module: &'static str,
        /// Repo-relative path of the `.hip` source.
        source: &'static str,
    },
}

/// One export (entry symbol) of a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ExportV1 {
    pub symbol: &'static str,
    pub arch: Arch,
    pub object: ObjectRefV1,
    /// Exact launch block.
    pub block: [u32; 3],
    /// Dynamic LDS bytes the launch passes.
    pub dynamic_lds: u32,
    /// Static LDS (`group_segment_fixed_size`) of the object.
    pub group_segment_fixed: u32,
    /// Grid in workgroups, over params and [`ExportV1::consts`].
    pub grid: [Expr; 3],
    /// Export-specific constants usable by name in every expression of the
    /// contract (tile rows/tokens, row blocks per CTA, ...).
    pub consts: &'static [(&'static str, u64)],
}

/// The typed ABI/numeric/extent contract of one kernel family layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct KernelAbiV1 {
    /// Stable family/layout identity, e.g. `mq6_i8_gemm.pm.set_f32`.
    pub id: &'static str,
    /// Bumped on any semantic change of this contract.
    pub version: u32,
    /// Wavefront size every export is built for.
    pub wave: u32,
    pub numeric: NumericContractId,
    pub params: &'static [ParamV1],
    pub operands: &'static [OperandV1],
    /// Explicit fields in offset order.
    pub fields: &'static [FieldV1],
    /// Hidden words in offset order.
    pub hidden: &'static [HiddenV1],
    /// End of the last explicit field (what a `KernargBlob` carries before
    /// its 16-byte tail pad; fields past the hidden block extend it).
    pub explicit_bytes: u32,
    /// The object's `kernarg_segment_size`.
    pub segment_bytes: u32,
    /// The object's `kernarg_segment_align`.
    pub segment_align: u32,
    pub domain: &'static [DomainRuleV1],
    pub exports: &'static [ExportV1],
}

impl KernelAbiV1 {
    /// The export `symbol` built for `arch`.
    pub fn export(&self, arch: Arch, symbol: &str) -> Option<&ExportV1> {
        self.exports
            .iter()
            .find(|e| e.arch == arch && e.symbol == symbol)
    }

    /// Position of the param `name` in [`KernelAbiV1::params`].
    pub fn param_index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// Position of the operand `name` in [`KernelAbiV1::operands`].
    pub fn operand_index(&self, name: &str) -> Option<usize> {
        self.operands.iter().position(|o| o.name == name)
    }

    /// Internal consistency, independent of any evidence: fields/hidden words
    /// sorted, disjoint, naturally aligned and inside the segment; explicit
    /// bytes = end of the last field; every pointer field names a declared
    /// operand and every operand is carried by exactly one pointer field;
    /// every expression names a declared param or a constant of every export;
    /// distinct export `(arch, symbol)` pairs; power-of-two alignments.
    pub fn self_check(&self) -> Result<(), Vec<AbiError>> {
        validate::self_check(self)
    }

    /// Lowercase hex SHA-256 of the contract's canonical JSON
    /// (`serde_json::to_vec(self)`): the identity R2/R5 records carry.
    pub fn digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let bytes = serde_json::to_vec(self).expect("KernelAbiV1 serializes");
        Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}

/// One operand as the caller binds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandBinding {
    pub address: u64,
    /// Bytes available from `address` to the end of the live allocation (or
    /// the tensor view), when the call site knows them. `None` is recorded in
    /// [`ValidatedInvocationV1::unchecked_extents`], never treated as proven.
    pub bytes: Option<u64>,
    /// The format the caller asserts the buffer holds.
    pub format: FormatId,
}

/// One launch request against a contract.
#[derive(Clone, Copy, Debug)]
pub struct InvocationV1<'a> {
    pub arch: Arch,
    pub symbol: &'a str,
    /// Values in [`KernelAbiV1::params`] order.
    pub params: &'a [u64],
    /// Bindings in [`KernelAbiV1::operands`] order.
    pub operands: &'a [OperandBinding],
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_lds: u32,
}

/// Why a contract or an invocation was rejected.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AbiError {
    #[error("{abi}: no export {symbol} for {arch}")]
    UnknownExport {
        abi: &'static str,
        arch: &'static str,
        symbol: String,
    },
    #[error("{abi}: {count} params supplied, the contract declares {expected}")]
    ParamCount {
        abi: &'static str,
        count: usize,
        expected: usize,
    },
    #[error("{abi}: {count} operands supplied, the contract declares {expected}")]
    OperandCount {
        abi: &'static str,
        count: usize,
        expected: usize,
    },
    #[error("{abi}: param {name} = {value} does not fit {ty:?}")]
    ParamRange {
        abi: &'static str,
        name: &'static str,
        value: u64,
        ty: ScalarTy,
    },
    #[error("{abi}: domain rule {rule:?} failed: {why}")]
    Domain {
        abi: &'static str,
        rule: Rule,
        why: &'static str,
    },
    #[error("{abi}: operand {operand} is {got:?}, the contract requires {expected:?}")]
    Format {
        abi: &'static str,
        operand: &'static str,
        expected: FormatId,
        got: FormatId,
    },
    #[error("{abi}: operand {operand} address {address:#x} is not {align}-byte aligned")]
    Misaligned {
        abi: &'static str,
        operand: &'static str,
        address: u64,
        align: u32,
    },
    #[error("{abi}: operand {operand} is null")]
    Null {
        abi: &'static str,
        operand: &'static str,
    },
    #[error("{abi}: operand {operand} needs {need} bytes, {have} are bound")]
    Extent {
        abi: &'static str,
        operand: &'static str,
        need: u64,
        have: u64,
    },
    #[error("{abi}: launch {what} is {got:?}, the contract requires {expected:?}")]
    Launch {
        abi: &'static str,
        what: &'static str,
        expected: [u64; 3],
        got: [u64; 3],
    },
    #[error("{abi}: expression {expr}: {reason}")]
    Expr {
        abi: &'static str,
        expr: String,
        reason: String,
    },
    #[error("{abi}: output buffer of {got} bytes, the explicit image needs {need}")]
    Buffer {
        abi: &'static str,
        need: usize,
        got: usize,
    },
    #[error("{abi}: malformed contract: {reason}")]
    Contract { abi: &'static str, reason: String },
}
