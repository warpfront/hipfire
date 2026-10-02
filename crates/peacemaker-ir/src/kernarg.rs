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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KernelStatus {
    Proven,
    #[default]
    Unknown,
}

/// `RegionEffect.bound` (design §1.1). A1 never proves extents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegionBound { AllocationWide }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AccessMode { Read, Write, ReadWrite }
impl AccessMode {
    pub fn reads(self) -> bool { matches!(self, Self::Read | Self::ReadWrite) }
    pub fn writes(self) -> bool { matches!(self, Self::Write | Self::ReadWrite) }
    pub fn join(self, other: Self) -> Self {
        if self == other { self } else { Self::ReadWrite }
    }
}

/// `RegionEffect.class` (design §1.1).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AccessClass { VmemLoad, VmemStore, SmemLoad, Atomic { commutative: bool }, LdsDma }
impl AccessClass {
    pub fn mode(self) -> AccessMode {
        match self {
            Self::VmemLoad | Self::SmemLoad | Self::LdsDma => AccessMode::Read,
            Self::VmemStore => AccessMode::Write,
            Self::Atomic { .. } => AccessMode::ReadWrite,
        }
    }
}

/// The instruction family that issued the access.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum MemPath { Global, Buffer, Flat, Smem, SmemBuffer }

/// Raw cache-policy bits of the access (gfx12: `th`/`scope`; gfx10/11:
/// `glc`/`slc`/`dlc`). Recorded, not interpreted: the per-arch cache table
/// (design §1.3, G4) maps them to visibility rungs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CacheBits { pub th: u8, pub scope: u8, pub glc: bool, pub slc: bool, pub dlc: bool, pub nv: bool }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgAccess {
    pub inst: InstId,
    pub class: AccessClass,
    pub path: MemPath,
    pub cache: CacheBits,
    /// The base also resolved to other arguments (e.g. `s_cselect_b64` of two
    /// kernarg pointers); every candidate gets the access.
    pub shared: bool,
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
    pub fn has(self, bit: u8) -> bool { self.0 & bit != 0 }
    pub fn names(self) -> Vec<&'static str> {
        [(Self::ADDRESS_BASE, "address_base"), (Self::ADDRESS_OFFSET, "address_offset"), (Self::LOOP_BOUND, "loop_bound"),
         (Self::GUARD, "guard"), (Self::GRID_TERM, "grid_term"), (Self::STORED, "stored"), (Self::LDS_ADDRESS, "lds_address")]
            .into_iter().filter(|(bit, _)| self.has(*bit)).map(|(_, name)| name).collect()
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
}
impl ArgFacts {
    pub fn classes(&self) -> Vec<AccessClass> {
        let mut out: Vec<AccessClass> = self.accesses.iter().map(|a| a.class).collect();
        out.sort();
        out.dedup();
        out
    }
    pub fn is_hidden(&self) -> bool { self.value_kind.starts_with("hidden_") }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DwordFacts { pub offset: u32, pub loaded: bool, pub sinks: Sinks }

/// Why a kernel is `Unknown`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Unresolved { pub inst: Option<InstId>, pub rule: &'static str, pub detail: String }

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
}
impl KernargFacts {
    pub fn arg_at(&self, offset: u32) -> Option<&ArgFacts> { self.args.iter().find(|a| a.offset == offset) }
}

/// Hidden argument kinds that carry launch geometry (code object v5).
pub const HIDDEN_GRID_KINDS: &[&str] = &[
    "hidden_block_count_x", "hidden_block_count_y", "hidden_block_count_z",
    "hidden_group_size_x", "hidden_group_size_y", "hidden_group_size_z",
    "hidden_remainder_x", "hidden_remainder_y", "hidden_remainder_z",
    "hidden_grid_dims",
];
