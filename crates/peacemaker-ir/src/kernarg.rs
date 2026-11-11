// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Kernarg facts (railgun A1/A3/A4, design §1.5): per-argument access
//! summaries from pointer provenance, read-cache classes, and scalar kernarg
//! roles including the implicit (hidden) suffix. Produced by
//! [`crate::passes::kernargs::analyze`]; carried on the `Analyzed` revision
//! as `Facts::kernargs`.
//!
//! Extents are not part of A1: every recognised pointer effect is
//! allocation-wide (`RegionBound::AllocationWide`) until A2 (M0b) bounds it.
use crate::cfg::InstId;

/// Whole-kernel A1 verdict. `Unknown` means at least one memory access has a
/// base that is not derived from kernarg pointer provenance (or the kernel
/// uses a construct the analysis does not model); the argument summaries of an
/// `Unknown` kernel are incomplete and must not be used to elide anything.
///
/// `Conditional` (railgun design A, amendment 2026-10-25 §A9.2) means every
/// access resolves, but the proof rests on [`KernargFacts::guards`] and/or on
/// [`IndexedArgEffect`]s (a direct pointer argument plus an integer
/// displacement that includes a loaded `i32`). It is neither `Proven` nor
/// usable alone: a consumer needs matching bind witnesses for every guard and
/// indexed effect, and a check like `status != Unknown` must not admit it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KernelStatus {
    Proven,
    Conditional,
    #[default]
    Unknown,
}

/// `RegionEffect.bound` (design §1.1). A1 never proves extents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionBound {
    AllocationWide,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}
impl AccessMode {
    pub fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }
    pub fn writes(self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
    pub fn join(self, other: Self) -> Self {
        if self == other {
            self
        } else {
            Self::ReadWrite
        }
    }
}

/// `RegionEffect.class` (design §1.1).
///
/// `AtomicRelease` and `LoadAcquire` (gfx12 only) refine `Atomic {
/// commutative: true }` and `VmemLoad` where the instruction stream proves the
/// shape LLVM's gfx12 HIP lowering gives a release atomic / an acquire load
/// (`passes::kernargs::sync_classes`): the atomic is device- or system-scope
/// and directly preceded by `s_wait_loadcnt 0` + `s_wait_storecnt 0` (and, at
/// system scope, `global_wb scope:SCOPE_SYS`); the load is device- or
/// system-scope and, on every path to the next memory read, is followed by
/// `s_wait_loadcnt 0` and a `global_inv` of at least its scope. Anything less
/// stays the relaxed class.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AccessClass {
    VmemLoad,
    VmemStore,
    SmemLoad,
    Atomic {
        commutative: bool,
    },
    LdsDma,
    /// A commutative atomic RMW issued with release ordering.
    AtomicRelease,
    /// A load whose observed value is followed by an acquire invalidate.
    LoadAcquire,
}
impl AccessClass {
    pub fn mode(self) -> AccessMode {
        match self {
            Self::VmemLoad | Self::SmemLoad | Self::LdsDma | Self::LoadAcquire => AccessMode::Read,
            Self::VmemStore => AccessMode::Write,
            Self::Atomic { .. } | Self::AtomicRelease => AccessMode::ReadWrite,
        }
    }
}

/// The instruction family that issued the access.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MemPath {
    Global,
    Buffer,
    Flat,
    Smem,
    SmemBuffer,
}

/// Raw cache-policy bits of the access (gfx12: `th`/`scope`; gfx10/11:
/// `glc`/`slc`/`dlc`). Recorded, not interpreted: the per-arch cache table
/// (design §1.3, G4) maps them to visibility rungs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CacheBits {
    pub th: u8,
    pub scope: u8,
    pub glc: bool,
    pub slc: bool,
    pub dlc: bool,
    pub nv: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgAccess {
    pub inst: InstId,
    pub class: AccessClass,
    pub path: MemPath,
    pub cache: CacheBits,
    /// The base also resolved to other arguments (e.g. `s_cselect_b64` of two
    /// kernarg pointers); every candidate gets the access.
    pub shared: bool,
    /// The mnemonic of a non-commutative atomic (`Atomic { commutative: false }`,
    /// e.g. `global_atomic_swap_b32`); `None` for every other access.
    pub atomic_op: Option<String>,
}

/// A3: which read caches can hold data this kernel read from an argument.
/// `Unknown` only at kernel level, when A1 is `Unknown`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReadCacheClass {
    #[default]
    NotRead,
    VmemOnly,
    ScalarOnly,
    ScalarAndVmem,
    Unknown,
}
impl ReadCacheClass {
    pub fn join(self, other: Self) -> Self {
        use ReadCacheClass::*;
        match (self, other) {
            (Unknown, _) | (_, Unknown) => Unknown,
            (NotRead, x) | (x, NotRead) => x,
            (a, b) if a == b => a,
            _ => ScalarAndVmem,
        }
    }
}

/// A4: where a kernarg dword may flow. A may-analysis (taint): a set bit
/// means some path may carry the dword's value into that use.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Sinks(pub u8);
impl Sinks {
    /// Part of a resolved pointer base.
    pub const ADDRESS_BASE: u8 = 1 << 0;
    /// Part of an address offset/index (not the base).
    pub const ADDRESS_OFFSET: u8 = 1 << 1;
    /// Operand of a compare inside a CFG cycle.
    pub const LOOP_BOUND: u8 = 1 << 2;
    /// Operand of a compare outside every CFG cycle.
    pub const GUARD: u8 = 1 << 3;
    /// Combined with a work-group or work-item id in arithmetic.
    pub const GRID_TERM: u8 = 1 << 4;
    /// Stored to memory as data.
    pub const STORED: u8 = 1 << 5;
    /// Part of an LDS address.
    pub const LDS_ADDRESS: u8 = 1 << 6;
    pub fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }
    pub fn names(self) -> Vec<&'static str> {
        [
            (Self::ADDRESS_BASE, "address_base"),
            (Self::ADDRESS_OFFSET, "address_offset"),
            (Self::LOOP_BOUND, "loop_bound"),
            (Self::GUARD, "guard"),
            (Self::GRID_TERM, "grid_term"),
            (Self::STORED, "stored"),
            (Self::LDS_ADDRESS, "lds_address"),
        ]
        .into_iter()
        .filter(|(bit, _)| self.has(*bit))
        .map(|(_, name)| name)
        .collect()
    }
}

/// A4 argument role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgRole {
    /// Some dword pair of the argument is a resolved pointer base.
    Pointer,
    /// Loaded, never a pointer base.
    Integer,
    /// Never loaded by the kernel (its bytes cannot influence execution
    /// unless `KernargFacts::dynamic_kernarg_reads` is set).
    NotLoaded,
}

/// One kernarg segment slot: a metadata `.args[]` entry, or a synthetic
/// 8-byte slot when an access base resolves inside no declared argument.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgFacts {
    pub offset: u32,
    pub size: u32,
    /// Index into `.args[]`, `None` for a synthetic slot.
    pub index: Option<usize>,
    pub name: String,
    pub value_kind: String,
    pub role: ArgRole,
    /// A1: `None` when the argument is never accessed as a pointer.
    pub mode: Option<AccessMode>,
    pub bound: Option<RegionBound>,
    pub accesses: Vec<ArgAccess>,
    /// A3 for this argument on this object's arch.
    pub read_cache: ReadCacheClass,
    /// A4: union over the argument's dwords.
    pub sinks: Sinks,
    pub loaded: bool,
    /// Every access attributed to this argument executes only on a path where
    /// the argument's 64-bit value was tested nonzero (a NULL argument reaches
    /// none of them). False when the argument has no access.
    pub null_guarded: bool,
}
impl ArgFacts {
    pub fn classes(&self) -> Vec<AccessClass> {
        let mut out: Vec<AccessClass> = self.accesses.iter().map(|a| a.class).collect();
        out.sort();
        out.dedup();
        out
    }
    pub fn is_hidden(&self) -> bool {
        self.value_kind.starts_with("hidden_")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DwordFacts {
    pub offset: u32,
    pub loaded: bool,
    pub sinks: Sinks,
}

/// Why a kernel is `Unknown`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unresolved {
    pub inst: Option<InstId>,
    pub rule: &'static str,
    pub detail: String,
}

/// Reads that are not argument effects: they cannot participate in an
/// inter-kernel hazard (the kernel may not write them; a write is `Unknown`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SideReads {
    /// VMEM reads of the kernarg segment itself.
    pub kernarg_segment: u32,
    /// Reads through a PC-relative (module global / constant table) base.
    pub module_global: u32,
    /// Reads through the dispatch/queue packet pointers.
    pub abi_packet: u32,
    /// FLAT accesses whose base is the LDS/private aperture.
    pub local_flat: u32,
    /// Private (scratch) accesses; not argument effects.
    pub scratch: u32,
}

/// Implicit-suffix use (A4). `hidden_grid` names the hidden args whose bytes
/// the kernel may load that encode launch geometry: block counts, group
/// sizes, remainders, grid dims — the words a grid repatch must also patch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ImplicitUse {
    pub hidden_loaded: Vec<String>,
    pub hidden_grid: Vec<String>,
    /// Hidden geometry args whose value may reach a compare or address.
    pub hidden_grid_live: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernargFacts {
    pub status: KernelStatus,
    pub unresolved: Vec<Unresolved>,
    pub kernarg_size: u32,
    pub args: Vec<ArgFacts>,
    pub dwords: Vec<DwordFacts>,
    pub implicit: ImplicitUse,
    /// A kernarg load at a non-constant offset: any dword may be loaded.
    pub dynamic_kernarg_reads: bool,
    pub side: SideReads,
    /// A3 at kernel level: join over arguments (`Unknown` if A1 is).
    pub read_cache: ReadCacheClass,
    /// Some A4 taint reached a kernarg dword past the tracked 512 bytes, or
    /// came from a dynamic kernarg read: the per-dword sinks are incomplete.
    pub taint_overflow: bool,
    /// Guarded analysis only: the launch/argument assumptions the facts rest
    /// on (empty for the unguarded analysis). A `Conditional` status is void
    /// without a matching discharge of every one.
    pub guards: Vec<Guard>,
    /// Guarded analysis only: accesses resolved through the integer/address
    /// witness. Each is also credited, allocation-wide, to its base
    /// argument's `accesses`; the index source is credited as an ordinary
    /// read of its pointer argument.
    pub indexed: Vec<IndexedArgEffect>,
    /// Sealed-table analysis only ([`crate::passes::kernargs::analyze_sealed`]):
    /// the pointer tables the kernel reads entries from, with every access
    /// whose base derives from such an entry. Not credited to any argument
    /// (the pointees are not arguments); a `Conditional` status is void without
    /// a matching `sealed_table` runtime witness.
    pub sealed: Vec<SealedTableUse>,
}
impl KernargFacts {
    pub fn arg_at(&self, offset: u32) -> Option<&ArgFacts> {
        self.args.iter().find(|a| a.offset == offset)
    }
}

// ---------------------------------------------------------------------------
// Sealed pointer tables: a 64-bit pointer loaded from `table[i]`, `i` guarded
// by `i < n`, then only read through.
// ---------------------------------------------------------------------------

/// The bound a sealed-table index is compared against (unsigned).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SealedBound {
    /// The 4-byte kernarg scalar at this byte offset.
    Kernarg {
        offset: u32,
    },
    Const(u32),
}

/// What the table index of an entry load is. Canonical text: `load@<pc>`,
/// `karg32:<offset>`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum SealedIndex {
    /// The 32-bit value loaded by the instruction at layout position `pc`.
    Load { pc: u32 },
    /// The kernarg scalar at this byte offset.
    Kernarg { offset: u32 },
}
impl std::fmt::Display for SealedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load { pc } => write!(f, "load@{pc}"),
            Self::Kernarg { offset } => write!(f, "karg32:{offset}"),
        }
    }
}

/// A 64-bit load of `table[index]` that executes only when `index < bound`
/// (unsigned), on every path from the kernel entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedEntryLoad {
    pub inst: InstId,
    pub pc: u32,
    pub class: AccessClass,
    pub path: MemPath,
    pub cache: CacheBits,
    pub width_bytes: u8,
    pub index: SealedIndex,
    /// Layout positions of the unsigned compares that establish
    /// `index < bound` on the paths reaching the load (sorted).
    pub guard_pcs: Vec<u32>,
}

/// A read whose base derives from sealed-table entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedAccess {
    pub inst: InstId,
    pub pc: u32,
    pub class: AccessClass,
    pub path: MemPath,
    pub cache: CacheBits,
    pub width_bytes: u8,
    /// `pc`s of this table's entry loads the base may derive from (sorted).
    pub entry_pcs: Vec<u32>,
}

/// One sealed table of a kernel: argument, bound, entry loads and the reads
/// through its entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedTableUse {
    /// Kernarg byte offset of the pointer argument holding the table base.
    pub table_arg_offset: u32,
    pub bound: SealedBound,
    pub entry_bytes: u8,
    pub entries: Vec<SealedEntryLoad>,
    pub accesses: Vec<SealedAccess>,
    /// Every read through an entry of this table is reachable only with that
    /// entry nonzero (proved on every path): a table holding null entries is
    /// safe for this kernel.
    pub null_guarded: bool,
}

// ---------------------------------------------------------------------------
// Amendment 2026-10-25 §A9.2: guarded cases and indexed argument effects.
// ---------------------------------------------------------------------------

/// A launch/argument assumption a [`KernelStatus::Conditional`] summary rests
/// on. Kernarg guards are *substituted* into the lifted ISA by the guarded
/// analysis (an argument known null is a guarded constant, never an
/// allocation at address 0); grid/block guards pin the work-group count and
/// size on an axis. The bind must discharge every guard against the real
/// launch before the summary may be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Guard {
    /// Kernarg bytes `[offset, offset + size)` (little-endian, `size` 4 or 8)
    /// equal `value`.
    KernargEq { offset: u32, size: u32, value: u64 },
    /// Work-group count on `axis` (0=x, 1=y, 2=z) equals `value`.
    GridDim { axis: u8, value: u32 },
    /// Work-group size on `axis` equals `value`.
    BlockDim { axis: u8, value: u32 },
}

/// Machine entry registers an address expression may start from (gfx12).
/// `WorkgroupIdYZ` is TTMP7 as held: `y | z << 16`; `WorkitemIdPacked` is v0
/// as held: `x | y << 10 | z << 20`. The grid/block guards make the packed
/// forms equal to the plain ids; the expression itself stays as observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EntryReg {
    WorkgroupIdX,
    WorkgroupIdYZ,
    WorkitemIdPacked,
}
impl EntryReg {
    pub fn name(self) -> &'static str {
        match self {
            Self::WorkgroupIdX => "wg_x",
            Self::WorkgroupIdYZ => "wg_yz",
            Self::WorkitemIdPacked => "wi",
        }
    }
}

/// A leaf an [`IdxExpr`] evaluator must supply.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdxLeaf {
    /// `bits` (32 or 64) of the kernarg segment at `offset`, little-endian.
    Kernarg {
        offset: u32,
        bits: u8,
    },
    Entry(EntryReg),
    /// The 32-bit value loaded by `IndexedArgEffect::index_loads[i]`.
    Load(u16),
}

/// Why [`IdxExpr::eval_checked`] refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IdxEvalError {
    /// The expression is not well formed (widths, shift amounts).
    Malformed,
    /// The environment supplied no value for a leaf.
    MissingLeaf,
    /// A node's exact (wide) value is outside its machine-width range: the
    /// machine result would wrap or change sign.
    Wrap,
}

/// An exact bit-vector expression over machine widths (32 or 64), as the
/// lifted ISA computes it. 32-bit nodes wrap at 32 bits, 64-bit nodes at 64;
/// `Ext` is the only width change (32 -> 64) and carries its signedness.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum IdxExpr {
    Const {
        value: u64,
        bits: u8,
    },
    Kernarg {
        offset: u32,
        bits: u8,
    },
    Entry(EntryReg),
    /// Result of the `i`th [`IndexLoad`] of the owning effect (32-bit).
    Load(u16),
    /// Sign (`signed`) or zero extension of a 32-bit operand to 64 bits.
    Ext {
        signed: bool,
        a: Box<IdxExpr>,
    },
    /// Left shift by a constant `amount < bits`.
    Shl {
        a: Box<IdxExpr>,
        amount: u8,
    },
    Add(Box<IdxExpr>, Box<IdxExpr>),
    Sub(Box<IdxExpr>, Box<IdxExpr>),
    Mul(Box<IdxExpr>, Box<IdxExpr>),
}

impl IdxExpr {
    /// Machine width of the value this node produces.
    pub fn bits(&self) -> u8 {
        match self {
            Self::Const { bits, .. } | Self::Kernarg { bits, .. } => *bits,
            Self::Entry(_) | Self::Load(_) => 32,
            Self::Ext { .. } => 64,
            Self::Shl { a, .. } => a.bits(),
            Self::Add(a, _) | Self::Sub(a, _) | Self::Mul(a, _) => a.bits(),
        }
    }

    /// Operand widths agree, only 32/64-bit widths occur, `Ext` takes a
    /// 32-bit operand, shift amounts are below the width.
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::Const { value, bits } => {
                matches!(bits, 32 | 64) && (*bits == 64 || *value <= u64::from(u32::MAX))
            }
            Self::Kernarg { bits, .. } => matches!(bits, 32 | 64),
            Self::Entry(_) | Self::Load(_) => true,
            Self::Ext { a, .. } => a.bits() == 32 && a.is_well_formed(),
            Self::Shl { a, amount } => {
                a.is_well_formed() && u32::from(*amount) < u32::from(a.bits())
            }
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.bits() == b.bits() && a.is_well_formed() && b.is_well_formed()
            }
        }
    }

    /// Every `Load(i)` index in the expression, in first-use order.
    pub fn load_indices(&self) -> Vec<u16> {
        let mut out = Vec::new();
        self.visit(&mut |e| {
            if let Self::Load(i) = e {
                if !out.contains(i) {
                    out.push(*i);
                }
            }
        });
        out
    }

    /// Pre-order walk.
    pub fn visit(&self, f: &mut impl FnMut(&IdxExpr)) {
        f(self);
        match self {
            Self::Ext { a, .. } | Self::Shl { a, .. } => a.visit(f),
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.visit(f);
                b.visit(f);
            }
            _ => {}
        }
    }

    /// Evaluate with checked wide arithmetic and compare with the machine
    /// semantics (§A9.3, last paragraph). Leaves and constants are unsigned
    /// machine values; `Ext { signed: true }` reinterprets its operand as two's
    /// complement. Every node other than a signed `Ext` must have an exact wide
    /// value inside `[0, 2^bits)` and the root must too: a wrap or a sign flip
    /// (for example a negative index reaching an unsigned address) is `Wrap`.
    pub fn eval_checked(&self, leaf: &dyn Fn(IdxLeaf) -> Option<u64>) -> Result<u64, IdxEvalError> {
        if !self.is_well_formed() {
            return Err(IdxEvalError::Malformed);
        }
        let wide = self.wide(leaf)?;
        let limit = 1i128 << self.bits();
        if (0..limit).contains(&wide) {
            Ok(wide as u64)
        } else {
            Err(IdxEvalError::Wrap)
        }
    }

    fn wide(&self, leaf: &dyn Fn(IdxLeaf) -> Option<u64>) -> Result<i128, IdxEvalError> {
        let mask = |bits: u8, v: u64| if bits == 64 { v } else { v & 0xffff_ffff };
        let fit = |bits: u8, v: i128| -> Result<i128, IdxEvalError> {
            if (0..1i128 << bits).contains(&v) {
                Ok(v)
            } else {
                Err(IdxEvalError::Wrap)
            }
        };
        Ok(match self {
            Self::Const { value, .. } => i128::from(*value),
            Self::Kernarg { offset, bits } => i128::from(mask(
                *bits,
                leaf(IdxLeaf::Kernarg {
                    offset: *offset,
                    bits: *bits,
                })
                .ok_or(IdxEvalError::MissingLeaf)?,
            )),
            Self::Entry(r) => i128::from(mask(
                32,
                leaf(IdxLeaf::Entry(*r)).ok_or(IdxEvalError::MissingLeaf)?,
            )),
            Self::Load(i) => i128::from(mask(
                32,
                leaf(IdxLeaf::Load(*i)).ok_or(IdxEvalError::MissingLeaf)?,
            )),
            Self::Ext { signed, a } => {
                let v = a.wide(leaf)?;
                if *signed && v >= 1 << 31 {
                    v - (1 << 32)
                } else {
                    v
                }
            }
            Self::Shl { a, amount } => fit(self.bits(), a.wide(leaf)? << *amount)?,
            Self::Add(a, b) => fit(self.bits(), a.wide(leaf)? + b.wide(leaf)?)?,
            Self::Sub(a, b) => fit(self.bits(), a.wide(leaf)? - b.wide(leaf)?)?,
            Self::Mul(a, b) => fit(
                self.bits(),
                a.wide(leaf)?
                    .checked_mul(b.wide(leaf)?)
                    .ok_or(IdxEvalError::Wrap)?,
            )?,
        })
    }
}

/// Canonical s-expression form (hash-bound in receipts; `FromStr` inverts):
/// `c32:256`, `c64:4096`, `karg32:24`, `karg64:16`, `wg_x`, `wg_yz`, `wi`,
/// `load:0`, `(sext e)`, `(zext e)`, `(shl e 3)`, `(add a b)`, `(sub a b)`,
/// `(mul a b)`. Widths follow from the leaves.
impl std::fmt::Display for IdxExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Const { value, bits } => write!(f, "c{bits}:{value}"),
            Self::Kernarg { offset, bits } => write!(f, "karg{bits}:{offset}"),
            Self::Entry(r) => f.write_str(r.name()),
            Self::Load(i) => write!(f, "load:{i}"),
            Self::Ext { signed, a } => write!(f, "({} {a})", if *signed { "sext" } else { "zext" }),
            Self::Shl { a, amount } => write!(f, "(shl {a} {amount})"),
            Self::Add(a, b) => write!(f, "(add {a} {b})"),
            Self::Sub(a, b) => write!(f, "(sub {a} {b})"),
            Self::Mul(a, b) => write!(f, "(mul {a} {b})"),
        }
    }
}

impl std::str::FromStr for IdxExpr {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        fn parse(
            toks: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
        ) -> Result<IdxExpr, String> {
            let tok = toks.next().ok_or("unexpected end")?;
            if tok == "(" {
                let op = toks.next().ok_or("unexpected end")?;
                let e = match op.as_str() {
                    "sext" | "zext" => IdxExpr::Ext {
                        signed: op == "sext",
                        a: Box::new(parse(toks)?),
                    },
                    "shl" => {
                        let a = Box::new(parse(toks)?);
                        let n = toks.next().ok_or("unexpected end")?;
                        IdxExpr::Shl {
                            a,
                            amount: n.parse().map_err(|_| format!("bad shift {n}"))?,
                        }
                    }
                    "add" | "sub" | "mul" => {
                        let (a, b) = (Box::new(parse(toks)?), Box::new(parse(toks)?));
                        match op.as_str() {
                            "add" => IdxExpr::Add(a, b),
                            "sub" => IdxExpr::Sub(a, b),
                            _ => IdxExpr::Mul(a, b),
                        }
                    }
                    other => return Err(format!("unknown operator {other}")),
                };
                return match toks.next().as_deref() {
                    Some(")") => Ok(e),
                    _ => Err("missing )".into()),
                };
            }
            let num = |t: &str| t.parse::<u64>().map_err(|_| format!("bad number in {tok}"));
            Ok(match tok.as_str() {
                "wg_x" => IdxExpr::Entry(EntryReg::WorkgroupIdX),
                "wg_yz" => IdxExpr::Entry(EntryReg::WorkgroupIdYZ),
                "wi" => IdxExpr::Entry(EntryReg::WorkitemIdPacked),
                t => match t.split_once(':') {
                    Some(("c32", v)) => IdxExpr::Const {
                        value: num(v)?,
                        bits: 32,
                    },
                    Some(("c64", v)) => IdxExpr::Const {
                        value: num(v)?,
                        bits: 64,
                    },
                    Some(("karg32", v)) => IdxExpr::Kernarg {
                        offset: num(v)? as u32,
                        bits: 32,
                    },
                    Some(("karg64", v)) => IdxExpr::Kernarg {
                        offset: num(v)? as u32,
                        bits: 64,
                    },
                    Some(("load", v)) => IdxExpr::Load(num(v)? as u16),
                    _ => return Err(format!("unknown token {t}")),
                },
            })
        }
        let mut toks = Vec::new();
        let mut cur = String::new();
        for c in s.chars() {
            match c {
                '(' | ')' => {
                    if !cur.is_empty() {
                        toks.push(std::mem::take(&mut cur));
                    }
                    toks.push(c.to_string());
                }
                c if c.is_whitespace() => {
                    if !cur.is_empty() {
                        toks.push(std::mem::take(&mut cur));
                    }
                }
                c => cur.push(c),
            }
        }
        if !cur.is_empty() {
            toks.push(cur);
        }
        let mut it = toks.into_iter().peekable();
        let e = parse(&mut it)?;
        if it.next().is_some() {
            return Err("trailing tokens".into());
        }
        if !e.is_well_formed() {
            return Err("ill-formed widths".into());
        }
        Ok(e)
    }
}

/// Element type of an [`IndexLoad`]. Only 32-bit integers qualify: a raw
/// 64-bit load is never an index leaf.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum IndexElem {
    I32,
}

/// How an indexed address consumes a loaded index.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum IndexExt {
    Sext64,
    Zext64,
    /// Used at 32 bits (inside 32-bit arithmetic, no extension).
    Raw32,
}

/// A qualifying ordinary 32-bit load from a directly rooted argument whose
/// value feeds an indexed address.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexLoad {
    pub inst: InstId,
    /// Position in the kernel's instruction layout (the instruction's index).
    pub pc: u32,
    /// `offset` of the pointer argument read.
    pub src_arg_offset: u32,
    pub elem: IndexElem,
    pub class: AccessClass,
    pub path: MemPath,
    pub cache: CacheBits,
    /// Byte offset from the source argument's pointer, including the
    /// instruction's immediate offset (here `4 * workgroup_id_y`).
    pub byte_offset: IdxExpr,
    /// How the consuming address uses the value, sorted, deduplicated.
    pub ext: Vec<IndexExt>,
}

/// One memory access whose address is `direct pointer argument + bounded
/// integer displacement`, where the displacement includes loaded `i32`
/// indices. The base is identified conditionally; a consumed bind witness
/// discharges bounds and stability (§A9.4). The effect set stays allocation
/// wide.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedArgEffect {
    pub inst: InstId,
    /// Position in the kernel's instruction layout (the instruction's index).
    pub pc: u32,
    /// `offset` of the direct base pointer argument.
    pub base_arg_offset: u32,
    pub mode: AccessMode,
    pub class: AccessClass,
    pub path: MemPath,
    pub cache: CacheBits,
    /// Bytes this access moves per active lane (store/load data width).
    pub width_bytes: u8,
    /// 64-bit byte displacement from the base pointer (wrapping 64-bit),
    /// including the instruction's immediate offset. `Load(i)` refers to
    /// `index_loads[i]`.
    pub displacement: IdxExpr,
    pub index_loads: Vec<IndexLoad>,
    /// The guards this effect was proved under (the case's guard set).
    pub guards: Vec<Guard>,
}

/// Hidden argument kinds that carry launch geometry (code object v5).
pub const HIDDEN_GRID_KINDS: &[&str] = &[
    "hidden_block_count_x",
    "hidden_block_count_y",
    "hidden_block_count_z",
    "hidden_group_size_x",
    "hidden_group_size_y",
    "hidden_group_size_z",
    "hidden_remainder_x",
    "hidden_remainder_y",
    "hidden_remainder_z",
    "hidden_grid_dims",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(load: u64, h: u64) -> impl Fn(IdxLeaf) -> Option<u64> {
        move |l| match l {
            IdxLeaf::Load(0) => Some(load),
            IdxLeaf::Kernarg { offset: 24, .. } => Some(h),
            IdxLeaf::Entry(EntryReg::WorkitemIdPacked) => Some(31),
            IdxLeaf::Entry(EntryReg::WorkgroupIdX) => Some(3),
            _ => None,
        }
    }
    const WRITER: &str = "(add (add (mul c64:256 (sext wg_x)) (mul (sext load:0) (sext (add (mul karg32:24 c32:256) (shl karg32:24 1))))) (zext (shl wi 3)))";

    #[test]
    fn canonical_form_round_trips_and_checks_widths() {
        let e: IdxExpr = WRITER.parse().expect("parses");
        assert_eq!(e.to_string(), WRITER);
        assert_eq!((e.bits(), e.load_indices()), (64, vec![0]));
        for bad in [
            "(add c32:1 c64:1)",   // mixed widths
            "(sext c64:1)",        // extension of a 64-bit operand
            "(shl c32:1 32)",      // shift out of range
            "(mul c32:1)",         // arity
            "(add c32:1 c32:1) x", // trailing tokens
            "load:x",
        ] {
            assert!(bad.parse::<IdxExpr>().is_err(), "{bad}");
        }
    }

    #[test]
    fn checked_evaluation_refuses_wrap_and_sign_changes() {
        let e: IdxExpr = WRITER.parse().unwrap();
        let h = 8u64;
        let r = h * 258;
        // head 3, position 5, lane 31: 3*256 + 5*R + 31*8.
        assert_eq!(e.eval_checked(&leaf(5, h)), Ok(3 * 256 + 5 * r + 31 * 8));
        // A negative position would flip the sign of the 64-bit product.
        assert_eq!(
            e.eval_checked(&leaf(u64::from(-1i32 as u32), h)),
            Err(IdxEvalError::Wrap)
        );
        // A stride whose 32-bit form wraps (H*258 >= 2^32) is refused, not wrapped.
        assert_eq!(
            e.eval_checked(&leaf(5, 0x4000_0000)),
            Err(IdxEvalError::Wrap)
        );
        // Missing leaves and ill-formed expressions are distinguished.
        assert_eq!(e.eval_checked(&|_| None), Err(IdxEvalError::MissingLeaf));
        let bad = IdxExpr::Add(
            Box::new(IdxExpr::Const { value: 1, bits: 32 }),
            Box::new(IdxExpr::Const { value: 1, bits: 64 }),
        );
        assert_eq!(bad.eval_checked(&|_| None), Err(IdxEvalError::Malformed));
        // A carry across the low 32 bits is plain 64-bit arithmetic.
        let carry: IdxExpr = "(add c64:4294967295 c64:1)".parse().unwrap();
        assert_eq!(carry.eval_checked(&|_| None), Ok(1 << 32));
    }
}
