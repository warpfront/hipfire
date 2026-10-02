//! M1: `Node::Kernel` (design §1.1) authored from certified facts.
//!
//! A kernel node is authored from one recorded launch — the recorder is the
//! discovery step (§6) — and from the fact record of the exact object that
//! launch ran: the corpus receipt keyed by the object's SHA-256 (§2.5).
//! Nothing here is keyed by a kernel name. Per explicit argument the receipt
//! carries A1 (pointer role, access mode, memory classes), A3 (read-cache
//! class) and A4 (scalar roles). The launch site declares its dynamic words
//! ([`DeclaredWord`]); every other explicit kernarg byte is a constant of the
//! authoring. A pointer is bound to the allocation the runtime reports for it
//! (`hipMemGetAddressRange`), so every effect is allocation-wide and the
//! program carries the `InBoundsNoAlias` obligation (§1.2).
//!
//! A launch whose object has no record, whose facts are `Unknown`, whose G8
//! differential failed, or whose pointers or words cannot be typed is not
//! PM4-eligible (§2.1): it becomes a `HipDirect` node, a full barrier.

use std::collections::BTreeMap;
use std::sync::Arc;

use railgun_corpus::{CorpusRecord, KernelRecord, Receipt};
use serde::Serialize;

use crate::{Binding, ResourceId, WordId};

/// A1 memory class of one access (receipt `classes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemClass {
    VmemLoad,
    VmemStore,
    SmemLoad,
    /// Non-commutative atomic (swap, cmpswap).
    Atomic,
    AtomicCommutative,
    LdsDma,
}

impl MemClass {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "vmem_load" => Self::VmemLoad,
            "vmem_store" => Self::VmemStore,
            "smem_load" => Self::SmemLoad,
            "atomic" => Self::Atomic,
            "atomic_commutative" => Self::AtomicCommutative,
            "lds_dma" => Self::LdsDma,
            other => return Err(format!("unknown memory class {other:?}")),
        })
    }

    pub fn reads(self) -> bool {
        !matches!(self, Self::VmemStore)
    }

    pub fn writes(self) -> bool {
        matches!(self, Self::VmemStore | Self::Atomic | Self::AtomicCommutative)
    }
}

/// A3 read-cache class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadCache {
    NotRead,
    VmemOnly,
    ScalarOnly,
    ScalarAndVmem,
    Unknown,
}

impl ReadCache {
    fn parse(name: &str) -> Result<Self, String> {
        Ok(match name {
            "not_read" => Self::NotRead,
            "vmem_only" => Self::VmemOnly,
            "scalar_only" => Self::ScalarOnly,
            "scalar_and_vmem" => Self::ScalarAndVmem,
            "unknown" => Self::Unknown,
            other => return Err(format!("unknown read-cache class {other:?}")),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArgRole {
    /// A1: a resolved access base. `classes` is empty when the kernel never
    /// dereferences it.
    Pointer { classes: Vec<MemClass>, read_cache: ReadCache },
    /// A4: loaded, never an access base.
    Integer,
    NotLoaded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgFact {
    pub offset: u32,
    pub size: u32,
    pub role: ArgRole,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactStatus {
    Proven,
    /// Rules of the accesses A1 could not attribute.
    Unknown(Vec<String>),
}

/// A hidden (implicit) argument the code object's metadata declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HiddenKind {
    BlockCount(u8),
    GroupSize(u8),
    Remainder(u8),
    GlobalOffset(u8),
    GridDims,
    DynamicLdsSize,
    /// Any other hidden kind (queue pointer, heap, hostcall, …): railgun
    /// supplies zero, so a kernel that loads one is not PM4-eligible.
    Other,
}

impl HiddenKind {
    fn parse(value_kind: &str) -> Option<Self> {
        let name = value_kind.strip_prefix("hidden_")?;
        let axis = |suffix: &str| match suffix {
            "x" => Some(0),
            "y" => Some(1),
            "z" => Some(2),
            _ => None,
        };
        let split = name.rsplit_once('_');
        Some(match (name, split) {
            ("grid_dims", _) => Self::GridDims,
            ("dynamic_lds_size", _) => Self::DynamicLdsSize,
            (_, Some(("block_count", a))) => axis(a).map_or(Self::Other, Self::BlockCount),
            (_, Some(("group_size", a))) => axis(a).map_or(Self::Other, Self::GroupSize),
            (_, Some(("remainder", a))) => axis(a).map_or(Self::Other, Self::Remainder),
            (_, Some(("global_offset", a))) => axis(a).map_or(Self::Other, Self::GlobalOffset),
            _ => Self::Other,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HiddenArg {
    pub offset: u32,
    pub size: u32,
    pub kind: HiddenKind,
    /// The metadata's `value_kind` spelling.
    pub value_kind: String,
    /// A4: the kernel loads it.
    pub loaded: bool,
}

/// One kernel's receipt facts (railgun section v1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelFacts {
    pub status: FactStatus,
    /// End of the last explicit argument.
    pub explicit_bytes: u32,
    /// The code object's whole kernarg segment (explicit + hidden).
    pub segment_bytes: u32,
    pub read_cache: ReadCache,
    /// Explicit arguments, sorted by offset, non-overlapping.
    pub args: Vec<ArgFact>,
    /// Hidden arguments, at their metadata offsets.
    pub hidden: Vec<HiddenArg>,
    /// Hidden grid dwords that reach a live grid term (I4b).
    pub hidden_grid_live: Vec<String>,
    /// G8: A1 ⊆ metadata-conservative (`pass`, `fail`, `not_applicable`).
    pub differential: String,
}

impl KernelFacts {
    pub fn from_record(record: &KernelRecord) -> Result<Self, String> {
        let status = match record.status.as_str() {
            "proven" => FactStatus::Proven,
            "unknown" => {
                let mut rules: Vec<String> = record.unresolved.iter().map(|u| u.rule.clone()).collect();
                rules.sort();
                rules.dedup();
                FactStatus::Unknown(rules)
            }
            other => return Err(format!("{}: unknown A1 status {other:?}", record.symbol)),
        };
        let mut args = Vec::with_capacity(record.args.len());
        let mut hidden = Vec::new();
        for arg in &record.args {
            if let Some(kind) = HiddenKind::parse(&arg.value_kind) {
                hidden.push(HiddenArg { offset: arg.offset, size: arg.size, kind, value_kind: arg.value_kind.clone(), loaded: arg.loaded });
                continue;
            }
            let role = match arg.role.as_str() {
                "pointer" => {
                    let classes = arg.classes.iter().map(|c| MemClass::parse(c)).collect::<Result<Vec<_>, _>>()?;
                    ArgRole::Pointer { classes, read_cache: ReadCache::parse(&arg.read_cache)? }
                }
                "integer" => ArgRole::Integer,
                "not_loaded" => ArgRole::NotLoaded,
                other => return Err(format!("{} +{}: unknown A4 role {other:?}", record.symbol, arg.offset)),
            };
            args.push(ArgFact { offset: arg.offset, size: arg.size, role });
        }
        args.sort_by_key(|a| a.offset);
        hidden.sort_by_key(|a| a.offset);
        let explicit_bytes = args.last().map_or(0, |a| a.offset + a.size);
        for pair in args.windows(2) {
            if pair[0].offset + pair[0].size > pair[1].offset {
                return Err(format!("{}: arguments +{} and +{} overlap", record.symbol, pair[0].offset, pair[1].offset));
            }
        }
        for h in &hidden {
            if h.offset < explicit_bytes || h.offset + h.size > record.kernarg_size {
                return Err(format!("{}: hidden {} at +{} lies outside the hidden block", record.symbol, h.value_kind, h.offset));
            }
        }
        if explicit_bytes > record.kernarg_size {
            return Err(format!("{}: explicit arguments end past the {}-byte segment", record.symbol, record.kernarg_size));
        }
        Ok(Self {
            status,
            explicit_bytes,
            segment_bytes: record.kernarg_size,
            read_cache: ReadCache::parse(&record.read_cache)?,
            args,
            hidden,
            hidden_grid_live: record.implicit.hidden_grid_live.clone(),
            differential: record.differential.status.clone(),
        })
    }

    /// Offsets of the A1 pointer arguments: the ones a recorder must bind to
    /// live allocations.
    pub fn pointer_offsets(&self) -> impl Iterator<Item = u32> + '_ {
        self.args.iter().filter(|a| matches!(a.role, ArgRole::Pointer { .. })).map(|a| a.offset)
    }
}

/// §1.1 `KernelRef`: which bytes ran and which record certifies them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct KernelRef {
    pub module: String,
    pub symbol: String,
    pub artifact_sha256: String,
    pub kd_sha256: String,
    pub receipt_sha256: String,
    /// D13: `D` (diagnostic lift + byte identity); no strict receipt exists.
    pub tier: String,
    /// Receipt-level obligations (e.g. `AddressArithmetic`, `InBoundsNoAlias`).
    pub obligations: Vec<String>,
}

/// A node's identity plus its facts.
#[derive(Clone, Debug)]
pub struct NodeFacts {
    pub kref: KernelRef,
    pub facts: KernelFacts,
}

impl NodeFacts {
    /// The facts for `symbol` from a verified corpus hit.
    pub fn from_hit(record: &CorpusRecord, receipt: &Receipt, symbol: &str) -> Result<Self, String> {
        let kernel = receipt
            .railgun
            .kernels
            .iter()
            .find(|k| k.symbol == symbol)
            .ok_or_else(|| format!("receipt {} has no railgun record for {symbol}", receipt.object_sha256))?;
        let identity = receipt
            .kernels
            .iter()
            .find(|k| k.symbol == symbol)
            .ok_or_else(|| format!("receipt {} has no identity for {symbol}", receipt.object_sha256))?;
        Ok(Self {
            kref: KernelRef {
                module: record.module.clone(),
                symbol: symbol.to_owned(),
                artifact_sha256: receipt.object_sha256.clone(),
                kd_sha256: identity.kd_sha256.clone(),
                receipt_sha256: record.receipt_sha256.clone(),
                tier: receipt.tier.clone(),
                obligations: receipt.obligations.clone(),
            },
            facts: KernelFacts::from_record(kernel)?,
        })
    }
}

/// §1.1 `Word.role` for the words a launch site declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "role")]
pub enum WordRole {
    /// GatedDeltaNet stochastic-rounding frame: a `u32` the host advances by
    /// `frames` per submit (design I6 wants a device counter; M1 keeps
    /// today's host patch at submit).
    GdnFrame { frames: u32 },
}

/// A dynamic kernarg dword declared by the composer at the launch site
/// (design §1.6: "words are declared, not discovered").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DeclaredWord {
    /// Byte offset in the explicit kernarg segment.
    pub offset: u32,
    #[serde(flatten)]
    pub role: WordRole,
}

/// What the recorder saw for one launch, plus the facts and live
/// allocations railgun needs to author it.
#[derive(Clone, Debug)]
pub struct LaunchRecord {
    pub symbol: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_lds: u32,
    /// The launch's kernarg bytes (HIP `extra` ABI, tail-padded).
    pub kernarg: Vec<u8>,
    pub declared: Vec<DeclaredWord>,
    /// The record for the object the launch ran, or why there is none
    /// (`receipt_missing`, §2.3).
    pub facts: Result<Arc<NodeFacts>, String>,
    /// The allocation behind every nonzero A1 pointer argument, probed while
    /// it was live: `(offset, allocation)`.
    pub allocations: Vec<(u32, Result<Binding, String>)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    Const(Vec<u8>),
    /// `base(resource) + interior`, re-encoded from the binding.
    Resource { resource: ResourceId, interior: u64 },
    Word(WordId),
}

/// §1.1 `RegionEffect` at allocation granularity (`bound: AllocationWide`;
/// A2 is M0b).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionEffect {
    pub arg_offset: u32,
    pub resource: ResourceId,
    pub classes: Vec<MemClass>,
}

impl RegionEffect {
    pub fn reads(&self) -> bool {
        self.classes.iter().any(|c| c.reads())
    }

    pub fn writes(&self) -> bool {
        self.classes.iter().any(|c| c.writes())
    }

    pub fn has(&self, class: MemClass) -> bool {
        self.classes.contains(&class)
    }
}

/// §2.1 per-launch lowering.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Lowering {
    Pm4,
    /// Certified but not PM4-admissible (e.g. a non-commutative atomic).
    Graph { reason: String },
    /// No usable facts: a full barrier on both sides (§1.4 rule 4).
    HipDirect { reason: String },
}

impl Lowering {
    pub fn is_pm4(&self) -> bool {
        matches!(self, Self::Pm4)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelNode {
    pub symbol: String,
    pub kref: Option<KernelRef>,
    /// `GridFormula` in M1 is a constant: the tape's grids are static (the
    /// attention tile grid is the recorded superset, MR
    /// `attention-tile-grid-superset`, byte-exact).
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub dynamic_lds: u32,
    /// End of the explicit arguments.
    pub explicit_bytes: u32,
    /// The certified kernarg segment (explicit + hidden); `None` without facts.
    pub segment_bytes: Option<u32>,
    /// Every explicit byte, in offset order, exactly once.
    pub args: Vec<(u32, Arg)>,
    /// The hidden arguments the metadata declares: the `ImplicitGrid` words
    /// bound to `grid` (I4b) and the dynamic LDS size, written by
    /// [`KernelProgram::kernarg_image`]; every other hidden byte is zero.
    pub hidden: Vec<HiddenArg>,
    pub effects: Vec<RegionEffect>,
    pub read_cache: ReadCache,
    pub lowering: Lowering,
}

/// A word of the program and where it is patched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProgramWord {
    pub node: usize,
    pub offset: u32,
    #[serde(flatten)]
    pub role: WordRole,
    /// Value at authoring (the recorded dword).
    pub authored: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelProgram {
    pub arch: String,
    /// `ResourceId` → the allocation it was authored against.
    pub resources: Vec<Binding>,
    pub words: Vec<ProgramWord>,
    pub nodes: Vec<KernelNode>,
}

/// Author the program for `launches`, in order (§1.1, §2.1).
pub fn author(arch: &str, launches: &[LaunchRecord]) -> KernelProgram {
    let mut resources: Vec<Binding> = Vec::new();
    let mut by_allocation: BTreeMap<(u64, u64), ResourceId> = BTreeMap::new();
    let mut words = Vec::new();
    let mut nodes = Vec::with_capacity(launches.len());
    for (index, launch) in launches.iter().enumerate() {
        let mut refusals: Vec<String> = Vec::new();
        let facts = match &launch.facts {
            Ok(f) => Some(f.clone()),
            Err(reason) => {
                refusals.push(format!("receipt_missing: {reason}"));
                None
            }
        };
        let Some(facts) = facts else {
            nodes.push(uncertified_node(launch, refusals.join("; ")));
            continue;
        };
        let f = &facts.facts;
        if let FactStatus::Unknown(rules) = &f.status {
            refusals.push(format!("A1 Unknown ({})", rules.join(", ")));
        }
        if f.differential == "fail" {
            refusals.push("G8 differential fails".to_owned());
        }
        let explicit = f.explicit_bytes as usize;
        if launch.kernarg.len() < explicit {
            refusals.push(format!("recorded kernarg is {} bytes, the object's explicit segment {explicit}", launch.kernarg.len()));
            nodes.push(uncertified_node(launch, refusals.join("; ")));
            continue;
        }
        if launch.kernarg[explicit..].iter().any(|b| *b != 0) {
            refusals.push(format!("recorded kernarg carries nonzero bytes past the {explicit}-byte explicit segment"));
        }
        for h in f.hidden.iter().filter(|h| h.loaded && h.kind == HiddenKind::Other) {
            refusals.push(format!("loads hidden {} which railgun does not supply", h.value_kind));
        }
        let declared: BTreeMap<u32, WordRole> = launch.declared.iter().map(|w| (w.offset, w.role)).collect();
        let mut used_words = 0usize;
        let mut args = Vec::with_capacity(f.args.len() * 2);
        let mut effects = Vec::new();
        let mut cursor = 0usize;
        let mut non_commutative_atomic = false;
        for arg in &f.args {
            let (start, end) = (arg.offset as usize, (arg.offset + arg.size) as usize);
            if start > cursor {
                args.push((cursor as u32, Arg::Const(launch.kernarg[cursor..start].to_vec())));
            }
            let bytes = &launch.kernarg[start..end];
            cursor = end;
            if let Some(role) = declared.get(&arg.offset) {
                used_words += 1;
                if arg.role == ArgRole::Integer && arg.size == 4 {
                    let authored = u32::from_le_bytes(bytes.try_into().expect("4-byte argument"));
                    args.push((arg.offset, Arg::Word(WordId(words.len() as u32))));
                    words.push(ProgramWord { node: index, offset: arg.offset, role: *role, authored });
                } else {
                    refusals.push(format!("declared word at +{} is not a loaded 4-byte integer argument (A4)", arg.offset));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                }
                continue;
            }
            let ArgRole::Pointer { classes, .. } = &arg.role else {
                args.push((arg.offset, Arg::Const(bytes.to_vec())));
                continue;
            };
            if arg.size != 8 {
                refusals.push(format!("pointer argument +{} is {} bytes", arg.offset, arg.size));
                args.push((arg.offset, Arg::Const(bytes.to_vec())));
                continue;
            }
            let value = u64::from_le_bytes(bytes.try_into().expect("8-byte argument"));
            if value == 0 {
                // A null pointer is a constant with no effect.
                args.push((arg.offset, Arg::Const(bytes.to_vec())));
                continue;
            }
            let allocation = launch.allocations.iter().find(|(offset, _)| *offset == arg.offset).map(|(_, a)| a);
            let binding = match allocation {
                Some(Ok(binding)) if value >= binding.base && value - binding.base < binding.bytes.max(1) => *binding,
                Some(Ok(binding)) => {
                    refusals.push(format!("pointer +{} = {value:#x} lies outside its reported allocation {:#x}+{}", arg.offset, binding.base, binding.bytes));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                    continue;
                }
                Some(Err(reason)) => {
                    refusals.push(format!("pointer +{} = {value:#x} has no live allocation: {reason}", arg.offset));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                    continue;
                }
                None => {
                    refusals.push(format!("pointer +{} = {value:#x} was not probed", arg.offset));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                    continue;
                }
            };
            let resource = *by_allocation.entry((binding.base, binding.bytes)).or_insert_with(|| {
                resources.push(binding);
                ResourceId(resources.len() as u32 - 1)
            });
            args.push((arg.offset, Arg::Resource { resource, interior: value - binding.base }));
            if !classes.is_empty() {
                non_commutative_atomic |= classes.contains(&MemClass::Atomic);
                effects.push(RegionEffect { arg_offset: arg.offset, resource, classes: classes.clone() });
            }
        }
        if cursor < explicit {
            args.push((cursor as u32, Arg::Const(launch.kernarg[cursor..explicit].to_vec())));
        }
        if used_words != declared.len() {
            refusals.push(format!("{} declared word(s) name no argument of the object", declared.len() - used_words));
        }
        let lowering = if !refusals.is_empty() {
            Lowering::HipDirect { reason: refusals.join("; ") }
        } else if non_commutative_atomic {
            Lowering::Graph { reason: "atomic without a commutativity certificate".to_owned() }
        } else {
            Lowering::Pm4
        };
        nodes.push(KernelNode {
            symbol: launch.symbol.clone(),
            kref: Some(facts.kref.clone()),
            grid: launch.grid,
            block: launch.block,
            dynamic_lds: launch.dynamic_lds,
            explicit_bytes: f.explicit_bytes,
            segment_bytes: Some(f.segment_bytes),
            args,
            hidden: f.hidden.clone(),
            effects,
            read_cache: f.read_cache,
            lowering,
        });
    }
    KernelProgram { arch: arch.to_owned(), resources, words, nodes }
}

fn uncertified_node(launch: &LaunchRecord, reason: String) -> KernelNode {
    KernelNode {
        symbol: launch.symbol.clone(),
        kref: None,
        grid: launch.grid,
        block: launch.block,
        dynamic_lds: launch.dynamic_lds,
        explicit_bytes: launch.kernarg.len() as u32,
        segment_bytes: None,
        args: vec![(0, Arg::Const(launch.kernarg.clone()))],
        hidden: Vec::new(),
        effects: Vec::new(),
        read_cache: ReadCache::Unknown,
        lowering: Lowering::HipDirect { reason },
    }
}

impl KernelProgram {
    /// The node's full kernarg segment as the loaded object lays it out
    /// (`loader_bytes`, which must equal the certified segment): explicit
    /// arguments re-encoded from the program (resources from their bindings,
    /// words from `words`), then the declared hidden arguments.
    pub fn kernarg_image(&self, node: usize, loader_bytes: usize, words: &[u32]) -> Result<Vec<u8>, String> {
        let n = self.nodes.get(node).ok_or_else(|| format!("no node {node}"))?;
        let segment = n.segment_bytes.ok_or_else(|| format!("{}: no certified kernarg segment", n.symbol))? as usize;
        if loader_bytes != segment {
            return Err(format!("{}: the loaded object's kernarg segment is {loader_bytes} B, the certified one {segment} B", n.symbol));
        }
        let mut image = vec![0u8; loader_bytes];
        for (offset, arg) in &n.args {
            let at = *offset as usize;
            match arg {
                Arg::Const(bytes) => image[at..at + bytes.len()].copy_from_slice(bytes),
                Arg::Resource { resource, interior } => {
                    let binding = self.resources.get(resource.0 as usize).ok_or_else(|| format!("unknown resource {}", resource.0))?;
                    let address = binding.base.checked_add(*interior).ok_or_else(|| "pointer overflows".to_owned())?;
                    image[at..at + 8].copy_from_slice(&address.to_le_bytes());
                }
                Arg::Word(id) => {
                    let value = words.get(id.0 as usize).ok_or_else(|| format!("no value for word {}", id.0))?;
                    image[at..at + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        let dims: u64 = if n.grid[2] != 1 || n.block[2] != 1 {
            3
        } else if n.grid[1] != 1 || n.block[1] != 1 {
            2
        } else {
            1
        };
        for h in &n.hidden {
            let value: u64 = match h.kind {
                HiddenKind::BlockCount(axis) => u64::from(n.grid[axis as usize]),
                HiddenKind::GroupSize(axis) => u64::from(n.block[axis as usize]),
                // HIP grids count workgroups: no partial groups, no offset.
                HiddenKind::Remainder(_) | HiddenKind::GlobalOffset(_) | HiddenKind::Other => 0,
                HiddenKind::GridDims => dims,
                HiddenKind::DynamicLdsSize => u64::from(n.dynamic_lds),
            };
            let size = h.size as usize;
            if size > 8 || (size < 8 && value >> (8 * size) != 0) {
                return Err(format!("{}: {} = {value} does not fit {size} bytes", n.symbol, h.value_kind));
            }
            let at = h.offset as usize;
            image[at..at + size].copy_from_slice(&value.to_le_bytes()[..size]);
        }
        Ok(image)
    }

    /// Authored value of every word, indexed by `WordId`.
    pub fn authored_words(&self) -> Vec<u32> {
        self.words.iter().map(|w| w.authored).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Facts for a kernel whose explicit arguments are `args` and whose
    /// kernarg segment is `segment` bytes (no hidden arguments).
    pub(crate) fn facts(args: Vec<ArgFact>, segment: u32) -> Arc<NodeFacts> {
        Arc::new(NodeFacts {
            kref: KernelRef {
                module: "m".into(),
                symbol: "k".into(),
                artifact_sha256: "00".repeat(32),
                kd_sha256: String::new(),
                receipt_sha256: String::new(),
                tier: "D".into(),
                obligations: vec!["InBoundsNoAlias".into()],
            },
            facts: KernelFacts {
                status: FactStatus::Proven,
                explicit_bytes: args.last().map_or(0, |a| a.offset + a.size),
                segment_bytes: segment,
                read_cache: ReadCache::VmemOnly,
                args,
                hidden: Vec::new(),
                hidden_grid_live: Vec::new(),
                differential: "pass".into(),
            },
        })
    }

    pub(crate) fn ptr(offset: u32, classes: &[MemClass]) -> ArgFact {
        ArgFact { offset, size: 8, role: ArgRole::Pointer { classes: classes.to_vec(), read_cache: ReadCache::VmemOnly } }
    }

    pub(crate) fn launch(pointers: &[(u32, u64)], explicit: u32, args: Vec<ArgFact>) -> LaunchRecord {
        let mut kernarg = vec![0u8; explicit.next_multiple_of(16) as usize];
        let mut allocations = Vec::new();
        for &(offset, value) in pointers {
            kernarg[offset as usize..offset as usize + 8].copy_from_slice(&value.to_le_bytes());
            allocations.push((offset, Ok(Binding { base: value & !0xfff, bytes: 0x10000 })));
        }
        LaunchRecord {
            symbol: "k".into(),
            grid: [4, 1, 1],
            block: [64, 1, 1],
            dynamic_lds: 0,
            kernarg,
            declared: Vec::new(),
            facts: Ok(facts(args, explicit)),
            allocations,
        }
    }

    #[test]
    fn declared_word_must_be_a_loaded_integer_and_image_reencodes_it() {
        let args = vec![ptr(0, &[MemClass::VmemLoad]), ArgFact { offset: 8, size: 4, role: ArgRole::Integer }];
        let mut good = launch(&[(0, 0x7000_0100)], 12, args.clone());
        good.kernarg[8..12].copy_from_slice(&41u32.to_le_bytes());
        good.declared = vec![DeclaredWord { offset: 8, role: WordRole::GdnFrame { frames: 1 } }];
        let program = author("gfx1201", &[good.clone()]);
        assert_eq!(program.nodes[0].lowering, Lowering::Pm4);
        assert_eq!(program.words[0].authored, 41);
        let image = program.kernarg_image(0, 12, &[99]).unwrap();
        assert_eq!(&image[0..8], &0x7000_0100u64.to_le_bytes());
        assert_eq!(&image[8..12], &99u32.to_le_bytes(), "the word is patched, not the recorded constant");

        let mut bad = good;
        bad.declared = vec![DeclaredWord { offset: 0, role: WordRole::GdnFrame { frames: 1 } }];
        let program = author("gfx1201", &[bad]);
        assert!(matches!(&program.nodes[0].lowering, Lowering::HipDirect { reason } if reason.contains("A4")));
    }

    #[test]
    fn missing_record_unknown_facts_and_unprobed_pointers_are_hip_direct() {
        let args = vec![ptr(0, &[MemClass::VmemStore])];
        let mut missing = launch(&[(0, 0x7000_0000)], 8, args.clone());
        missing.facts = Err("no record".into());
        let mut unknown = launch(&[(0, 0x7000_0000)], 8, args.clone());
        let mut f = (*unknown.facts.clone().unwrap()).clone();
        f.facts.status = FactStatus::Unknown(vec!["base-memory-derived".into()]);
        unknown.facts = Ok(Arc::new(f));
        let mut unprobed = launch(&[(0, 0x7000_0000)], 8, args);
        unprobed.allocations.clear();
        let program = author("gfx1201", &[missing, unknown, unprobed]);
        for (node, needle) in program.nodes.iter().zip(["receipt_missing", "A1 Unknown", "not probed"]) {
            assert!(matches!(&node.lowering, Lowering::HipDirect { reason } if reason.contains(needle)), "{:?}", node.lowering);
        }
    }

    fn hidden(offset: u32, size: u32, value_kind: &str, loaded: bool) -> HiddenArg {
        HiddenArg { offset, size, kind: HiddenKind::parse(value_kind).unwrap(), value_kind: value_kind.into(), loaded }
    }

    #[test]
    fn hidden_arguments_are_written_where_the_metadata_declares_them() {
        let mut l = launch(&[(0, 0x7000_0000)], 8, vec![ptr(0, &[MemClass::VmemLoad])]);
        l.grid = [3, 5, 1];
        l.block = [256, 1, 1];
        l.dynamic_lds = 1024;
        let mut f = (*l.facts.clone().unwrap()).clone();
        f.facts.segment_bytes = 8 + 256;
        f.facts.hidden = vec![
            hidden(8, 4, "hidden_block_count_x", false),
            hidden(12, 4, "hidden_block_count_y", true),
            hidden(20, 2, "hidden_group_size_x", false),
            hidden(72, 2, "hidden_grid_dims", false),
            hidden(128, 4, "hidden_dynamic_lds_size", false),
        ];
        l.facts = Ok(Arc::new(f.clone()));
        let program = author("gfx1201", &[l.clone()]);
        assert_eq!(program.nodes[0].lowering, Lowering::Pm4);
        let image = program.kernarg_image(0, 264, &[]).unwrap();
        assert_eq!(&image[8..16], &[3, 0, 0, 0, 5, 0, 0, 0]);
        assert_eq!(&image[20..22], &[0, 1]);
        assert_eq!(&image[72..74], &[2, 0]);
        assert_eq!(u32::from_le_bytes(image[128..132].try_into().unwrap()), 1024);
        // One nonzero byte each: the pointer, both block counts, the group size, the dims, the LDS size.
        assert_eq!(image.iter().filter(|b| **b != 0).count(), 6, "nothing undeclared is written");
        assert!(program.kernarg_image(0, 272, &[]).is_err(), "a loaded object with another segment size is refused");

        f.facts.hidden.push(hidden(200, 8, "hidden_queue_ptr", true));
        l.facts = Ok(Arc::new(f));
        let program = author("gfx1201", &[l]);
        assert!(matches!(&program.nodes[0].lowering, Lowering::HipDirect { reason } if reason.contains("hidden_queue_ptr")));
    }
}
