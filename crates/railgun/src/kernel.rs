// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

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

use railgun_corpus::{CorpusRecord, KernelRecord, PinnedReceipt, Receipt};
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
    /// A commutative atomic RMW issued with release ordering: the issuing
    /// workgroup's earlier writes are visible to whoever acquires the cell
    /// (an event-cell publication).
    AtomicRelease,
    /// A load whose observed value is followed by an acquire invalidate: what
    /// it observes happens-before the loads that follow (an event-cell wait).
    LoadAcquire,
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
            "atomic_release" => Self::AtomicRelease,
            "load_acquire" => Self::LoadAcquire,
            other => return Err(format!("unknown memory class {other:?}")),
        })
    }

    pub fn reads(self) -> bool {
        !matches!(self, Self::VmemStore)
    }

    pub fn writes(self) -> bool {
        matches!(
            self,
            Self::VmemStore | Self::Atomic | Self::AtomicCommutative | Self::AtomicRelease
        )
    }

    /// The classes an event cell may be accessed with: atomics (commutative,
    /// release) and loads (plain, scalar, acquire). A plain store or a
    /// non-commutative atomic is not one.
    pub fn is_event_class(self) -> bool {
        matches!(
            self,
            Self::AtomicCommutative
                | Self::AtomicRelease
                | Self::LoadAcquire
                | Self::VmemLoad
                | Self::SmemLoad
        )
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
    Pointer {
        classes: Vec<MemClass>,
        read_cache: ReadCache,
    },
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

/// Witness kind the receipt names for the loaded-index case (§A9.2).
pub const WITNESS_INDEXED_BIND: &str = "indexed_bind";

/// Witness kind the receipt names for the sealed pointer-table case: the
/// table argument resolves to a loader-sealed immutable region whose every
/// entry points into an immutable allocation (`crate::sealed_table`).
pub const WITNESS_SEALED_TABLE: &str = railgun_corpus::receipt::WITNESS_SEALED_TABLE;

/// `IndexLoad` (§A9.2): an immutable `I32` index loaded from a pointer
/// argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexLoad {
    pub pc: u32,
    pub source_arg_offset: u32,
    pub element: String,
    pub offset_expr: String,
    pub extension: String,
    pub class: MemClass,
    pub path: String,
    pub cache: String,
    pub width: u32,
}

/// `IndexedArgEffect` (§A9.2): one access whose address is `direct base +
/// bounded integer expression`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedArgEffect {
    pub pc: u32,
    pub base_arg_offset: u32,
    pub mode: String,
    pub class: MemClass,
    pub path: String,
    pub cache: String,
    pub width: u32,
    pub address_expr: String,
    pub index_loads: Vec<IndexLoad>,
    pub guards: Vec<String>,
}

/// One launch/argument guard of the guarded case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgGuard {
    pub kind: String,
    pub arg_offset: u32,
    pub size: u32,
    pub value: String,
}

/// Where a sealed table's guard bound comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TableBound {
    /// The 4-byte unsigned kernarg scalar at this byte offset.
    Kernarg(u32),
    /// A constant.
    Const(u32),
}

/// One entry load of a sealed table: a 64-bit load of `table[index]` whose
/// index is, on every path from the kernel entry, strictly below the bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableEntryLoad {
    pub pc: u32,
    pub class: MemClass,
    pub path: String,
    pub cache: String,
    pub width: u32,
    pub index_expr: String,
    pub guard_pcs: Vec<u32>,
}

/// One read whose base derives from sealed-table entries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableAccess {
    pub pc: u32,
    pub class: MemClass,
    pub path: String,
    pub cache: String,
    pub width: u32,
    pub entry_pcs: Vec<u32>,
}

/// One sealed pointer table of a `Conditional` kernel (receipt
/// `sealed_tables`, kept apart from the indexed effects): the pointer
/// argument that holds the table, the bound its index is checked against and
/// what the kernel does with the loaded entries (only reads).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedTableFact {
    /// Kernarg byte offset of the table's base pointer.
    pub table_arg_offset: u32,
    pub bound: TableBound,
    /// Bytes per entry (a pointer: 8).
    pub entry_bytes: u32,
    pub entries: Vec<TableEntryLoad>,
    pub accesses: Vec<TableAccess>,
    /// Every read through an entry is reachable only with that entry nonzero
    /// (the receipt's `null_guarded`): the sealed table may hold null entries.
    pub null_guarded: bool,
}

/// One pointer a kernel loads from a descriptor-arena record of some
/// template, and what it does through the loaded address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaPointerLoad {
    /// Record-relative byte offset of the 8-byte pointer (8-aligned).
    pub record_offset: u32,
    /// Sorted, deduplicated; empty when the pointer is loaded and never
    /// dereferenced.
    pub classes: Vec<MemClass>,
}

/// Writes through the pointer at `record_offset` by two workgroups `w != w'`
/// are disjoint whenever they differ on some axis in `axes` (a mask: bit 0 x,
/// bit 1 y, bit 2 z; never zero).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgFootprint {
    pub record_offset: u32,
    pub axes: u8,
    /// Work-items per tile on x the proof assumes (0: none).
    pub block_x: u32,
    /// Exclusive per-axis tile-id bounds (`None`: unbounded); the host
    /// requires the virtual grid extents `gx, gy, gz` within them.
    pub bounds: [Option<u64>; 3],
    /// Record dwords the proof used: `(offset, mask, value)`.
    pub requires: Vec<(u32, u32, u32)>,
}

impl WgFootprint {
    /// A footprint over the axis mask with no assumptions.
    pub fn plain(record_offset: u32, axes: u8) -> Self {
        Self {
            record_offset,
            axes,
            block_x: 0,
            bounds: [None; 3],
            requires: Vec::new(),
        }
    }
}

/// One record template of a descriptor arena: the pointer fields a kernel
/// loads from a record of this template. A field the facts do not list is
/// never loaded as an address base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaTemplate {
    pub template: u32,
    pub pointer_loads: Vec<ArenaPointerLoad>,
    /// The sealed pointer tables the body reads through (receipt
    /// `ArenaTemplateRecord::sealed_tables`). `table_arg_offset` is the
    /// record offset of the table pointer field; a [`TableBound::Kernarg`]
    /// bound is the record offset of a declared `u32` scalar
    /// ([`crate::BindingRecord::scalars`]). Discharged by
    /// [`KernelProgram::bind_records`] for every record of the template.
    pub sealed: Vec<SealedTableFact>,
    /// Certified per-workgroup write footprints (receipt
    /// `ArenaTemplateRecord::wg_footprints`).
    pub wg_footprints: Vec<WgFootprint>,
}

/// One descriptor arena of a `Conditional` kernel (receipt
/// `descriptor_arenas`): the pointer argument holding the base of an array of
/// binding records and, per record template, the pointers loaded from a
/// record. The certificate says "every address this kernel accesses through
/// the arena is one of these loaded fields"; the program supplies the
/// records ([`KernelProgram::bind_records`]), which turns each field into an
/// effect on the resource it names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaFact {
    /// Kernarg byte offset of the arena's base pointer.
    pub arena_arg_offset: u32,
    pub templates: Vec<ArenaTemplate>,
}

/// The guarded case of a `Conditional` kernel: valid only while every guard
/// holds and every `witnesses` kind has a matching runtime bind witness.
/// `indexed` and `tables` are the two independent case families (the
/// `indexed_bind` and `sealed_table` witnesses); a record has either or both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalFacts {
    pub guards: Vec<ArgGuard>,
    pub indexed: Vec<IndexedArgEffect>,
    pub tables: Vec<SealedTableFact>,
    /// Descriptor arenas (the `descriptor_arena` case): discharged by
    /// [`KernelProgram::bind_records`], not by a window witness.
    pub arenas: Vec<ArenaFact>,
    /// The dispatcher protocol the task-graph certificate proved (see
    /// [`crate::task_graph`]); only with `arenas`.
    pub task_protocol: Option<crate::task_graph::TaskProtocolFact>,
    pub witnesses: Vec<String>,
    pub case_digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactStatus {
    Proven,
    /// Rules of the accesses A1 could not attribute.
    Unknown(Vec<String>),
    /// `KernelStatus::Conditional`: neither `Proven` nor `Unknown`. Not
    /// PM4-eligible until a matching [`crate::index_bind::IndexedBindWitness`]
    /// is consumed by the window; the cached fact is never mutated.
    Conditional(ConditionalFacts),
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
    /// #867: kernarg offset of the ControlWord pointer whose clear flag makes
    /// the launch a certified no-op (`railgun_cert::guard_flag`).
    pub guard_flag: Option<u32>,
}

/// Parse a conditional record with `KernelRecord::validate_case` semantics
/// (the object-bound digest is verified at admission; this is the fail-closed
/// structural half): an indexed case needs guards and indexed effects, a
/// sealed-only case needs sealed tables and **no** guards, and the required
/// witnesses are exactly the kinds the case implies. An unknown kind is an
/// error.
fn conditional_from_record(record: &KernelRecord) -> Result<ConditionalFacts, String> {
    let sym = &record.symbol;
    let has_indexed = !record.indexed_effects.is_empty();
    let has_sealed = !record.sealed_tables.is_empty();
    let has_arena = !record.descriptor_arenas.is_empty();
    if (!has_indexed && !has_sealed && !has_arena)
        || record.required_witnesses.is_empty()
        || record.case_digest.is_empty()
    {
        return Err(format!(
            "{sym}: conditional record lacks effects (indexed, sealed tables or descriptor arenas), witnesses or case digest"
        ));
    }
    if has_indexed && record.guards.is_empty() {
        return Err(format!("{sym}: indexed conditional record lacks guards"));
    }
    if !has_indexed && !record.guards.is_empty() {
        return Err(format!(
            "{sym}: sealed-table-only conditional record carries guards"
        ));
    }
    for w in &record.required_witnesses {
        if !railgun_corpus::receipt::KNOWN_WITNESSES.contains(&w.as_str()) {
            return Err(format!("{sym}: unknown required witness kind {w:?}"));
        }
    }
    if record.required_witnesses != record.implied_witnesses() {
        return Err(format!(
            "{sym}: required witnesses {:?} are not the kinds the case implies {:?}",
            record.required_witnesses,
            record.implied_witnesses()
        ));
    }
    let mut tables = Vec::with_capacity(record.sealed_tables.len());
    for (i, t) in record.sealed_tables.iter().enumerate() {
        t.validate(sym)?;
        if record.sealed_tables[..i].iter().any(|o| {
            o.table_arg_offset == t.table_arg_offset
                && o.bound_kind == t.bound_kind
                && o.bound_value == t.bound_value
        }) {
            return Err(format!("{sym}: duplicate sealed table"));
        }
        let table_arg = record.args.iter().find(|a| a.offset == t.table_arg_offset);
        if !table_arg.is_some_and(|a| a.size == 8 && a.role == "pointer") {
            return Err(format!(
                "{sym}: sealed table argument +{} is not an 8-byte pointer argument",
                t.table_arg_offset
            ));
        }
        tables.push(sealed_fact(sym, t, Some(record.kernarg_size))?);
    }
    let mut indexed = Vec::with_capacity(record.indexed_effects.len());
    for e in &record.indexed_effects {
        let mut loads = Vec::with_capacity(e.index_loads.len());
        for l in &e.index_loads {
            loads.push(IndexLoad {
                pc: l.pc,
                source_arg_offset: l.source_arg_offset,
                element: l.element.clone(),
                offset_expr: l.offset_expr.clone(),
                extension: l.extension.clone(),
                class: MemClass::parse(&l.class).map_err(|m| format!("{sym}: {m}"))?,
                path: l.path.clone(),
                cache: l.cache.clone(),
                width: l.width,
            });
        }
        indexed.push(IndexedArgEffect {
            pc: e.pc,
            base_arg_offset: e.base_arg_offset,
            mode: e.mode.clone(),
            class: MemClass::parse(&e.class).map_err(|m| format!("{sym}: {m}"))?,
            path: e.path.clone(),
            cache: e.cache.clone(),
            width: e.width,
            address_expr: e.address_expr.clone(),
            index_loads: loads,
            guards: e.guards.clone(),
        });
    }
    let mut arenas = Vec::with_capacity(record.descriptor_arenas.len());
    for (i, a) in record.descriptor_arenas.iter().enumerate() {
        a.validate(sym)?;
        if record.descriptor_arenas[..i]
            .iter()
            .any(|o| o.arena_arg_offset == a.arena_arg_offset)
        {
            return Err(format!("{sym}: duplicate descriptor arena"));
        }
        let arena_arg = record.args.iter().find(|x| x.offset == a.arena_arg_offset);
        if !arena_arg.is_some_and(|x| x.size == 8 && x.role == "pointer") {
            return Err(format!(
                "{sym}: descriptor arena argument +{} is not an 8-byte pointer argument",
                a.arena_arg_offset
            ));
        }
        let mut templates = Vec::with_capacity(a.templates.len());
        for t in &a.templates {
            let sealed = t
                .sealed_tables
                .iter()
                .map(|s| sealed_fact(sym, s, None))
                .collect::<Result<Vec<_>, _>>()?;
            let mut pointer_loads = Vec::with_capacity(t.pointer_loads.len());
            for p in &t.pointer_loads {
                let mut classes = p
                    .classes
                    .iter()
                    .map(|c| MemClass::parse(c).map_err(|m| format!("{sym}: {m}")))
                    .collect::<Result<Vec<_>, _>>()?;
                classes.sort();
                classes.dedup();
                pointer_loads.push(ArenaPointerLoad {
                    record_offset: p.record_offset,
                    classes,
                });
            }
            templates.push(ArenaTemplate {
                template: t.template,
                pointer_loads,
                sealed,
                wg_footprints: t
                    .wg_footprints
                    .iter()
                    .map(|f| {
                        let mut axes = 0u8;
                        for a in &f.axes {
                            axes |= match a.as_str() {
                                "x" => 1,
                                "y" => 2,
                                "z" => 4,
                                other => {
                                    return Err(format!("{sym}: unknown workgroup axis {other:?}"))
                                }
                            };
                        }
                        if axes == 0 {
                            return Err(format!("{sym}: workgroup footprint has no axes"));
                        }
                        let mut bounds = [None; 3];
                        for b in &f.bounds {
                            let i = match b.axis.as_str() {
                                "x" => 0,
                                "y" => 1,
                                "z" => 2,
                                other => {
                                    return Err(format!("{sym}: unknown workgroup axis {other:?}"))
                                }
                            };
                            if b.bound == 0 || bounds[i].replace(b.bound).is_some() {
                                return Err(format!("{sym}: bad workgroup bound on {}", b.axis));
                            }
                        }
                        Ok(WgFootprint {
                            record_offset: f.record_offset,
                            axes,
                            block_x: f.block_x,
                            bounds,
                            requires: f
                                .requires
                                .iter()
                                .map(|r| (r.offset, r.mask, r.value))
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?,
            });
        }
        arenas.push(ArenaFact {
            arena_arg_offset: a.arena_arg_offset,
            templates,
        });
    }
    let task_protocol = record
        .task_protocol
        .as_ref()
        .map(|p| crate::task_graph::TaskProtocolFact::from_record(p, &record.args, sym))
        .transpose()?;
    if task_protocol.is_some() && arenas.is_empty() {
        return Err(format!(
            "{sym}: a task protocol needs the descriptor arena its tasks' records live in"
        ));
    }
    Ok(ConditionalFacts {
        guards: record
            .guards
            .iter()
            .map(|g| ArgGuard {
                kind: g.kind.clone(),
                arg_offset: g.arg_offset,
                size: g.size,
                value: g.value.clone(),
            })
            .collect(),
        indexed,
        tables,
        arenas,
        task_protocol,
        witnesses: record.required_witnesses.clone(),
        case_digest: record.case_digest.clone(),
    })
}

/// A receipt sealed table as a fact. `kernarg_size` bounds a kernarg-bound
/// offset (a standalone record's own segment); an arena template's bound is a
/// record offset, checked against the records when they are bound.
fn sealed_fact(
    sym: &str,
    t: &railgun_corpus::receipt::SealedTableRecord,
    kernarg_size: Option<u32>,
) -> Result<SealedTableFact, String> {
    let bound = match t.bound_kind.as_str() {
        railgun_corpus::receipt::SEALED_BOUND_KERNARG => {
            let at = u32::try_from(t.bound_value)
                .ok()
                .filter(|at| {
                    at.checked_add(4)
                        .is_some_and(|end| kernarg_size.is_none_or(|size| end <= size))
                })
                .ok_or_else(|| {
                    format!(
                        "{sym}: sealed table bound +{} lies outside the {:?}-byte kernarg segment",
                        t.bound_value, kernarg_size
                    )
                })?;
            TableBound::Kernarg(at)
        }
        railgun_corpus::receipt::SEALED_BOUND_CONST => TableBound::Const(
            u32::try_from(t.bound_value)
                .map_err(|_| format!("{sym}: sealed table constant bound exceeds u32"))?,
        ),
        other => return Err(format!("{sym}: unknown sealed table bound kind {other:?}")),
    };
    let mut entries = Vec::with_capacity(t.entries.len());
    for e in &t.entries {
        entries.push(TableEntryLoad {
            pc: e.pc,
            class: MemClass::parse(&e.class).map_err(|m| format!("{sym}: {m}"))?,
            path: e.path.clone(),
            cache: e.cache.clone(),
            width: e.width,
            index_expr: e.index_expr.clone(),
            guard_pcs: e.guard_pcs.clone(),
        });
    }
    let mut accesses = Vec::with_capacity(t.accesses.len());
    for a in &t.accesses {
        accesses.push(TableAccess {
            pc: a.pc,
            class: MemClass::parse(&a.class).map_err(|m| format!("{sym}: {m}"))?,
            path: a.path.clone(),
            cache: a.cache.clone(),
            width: a.width,
            entry_pcs: a.entry_pcs.clone(),
        });
    }
    Ok(SealedTableFact {
        table_arg_offset: t.table_arg_offset,
        bound,
        entry_bytes: t.entry_bytes,
        entries,
        accesses,
        null_guarded: t.null_guarded,
    })
}

impl KernelFacts {
    pub fn from_record(record: &KernelRecord) -> Result<Self, String> {
        let status = match record.status.as_str() {
            "proven" => FactStatus::Proven,
            "unknown" => {
                let mut rules: Vec<String> =
                    record.unresolved.iter().map(|u| u.rule.clone()).collect();
                rules.sort();
                rules.dedup();
                FactStatus::Unknown(rules)
            }
            railgun_corpus::receipt::STATUS_CONDITIONAL => {
                FactStatus::Conditional(conditional_from_record(record)?)
            }
            other => return Err(format!("{}: unknown A1 status {other:?}", record.symbol)),
        };
        let mut args = Vec::with_capacity(record.args.len());
        let mut hidden = Vec::new();
        for arg in &record.args {
            if let Some(kind) = HiddenKind::parse(&arg.value_kind) {
                hidden.push(HiddenArg {
                    offset: arg.offset,
                    size: arg.size,
                    kind,
                    value_kind: arg.value_kind.clone(),
                    loaded: arg.loaded,
                });
                continue;
            }
            let role = match arg.role.as_str() {
                "pointer" => {
                    let classes = arg
                        .classes
                        .iter()
                        .map(|c| MemClass::parse(c))
                        .collect::<Result<Vec<_>, _>>()?;
                    ArgRole::Pointer {
                        classes,
                        read_cache: ReadCache::parse(&arg.read_cache)?,
                    }
                }
                "integer" => ArgRole::Integer,
                "not_loaded" => ArgRole::NotLoaded,
                other => {
                    return Err(format!(
                        "{} +{}: unknown A4 role {other:?}",
                        record.symbol, arg.offset
                    ))
                }
            };
            args.push(ArgFact {
                offset: arg.offset,
                size: arg.size,
                role,
            });
        }
        args.sort_by_key(|a| a.offset);
        hidden.sort_by_key(|a| a.offset);
        let explicit_bytes = args.last().map_or(0, |a| a.offset + a.size);
        for pair in args.windows(2) {
            if pair[0].offset + pair[0].size > pair[1].offset {
                return Err(format!(
                    "{}: arguments +{} and +{} overlap",
                    record.symbol, pair[0].offset, pair[1].offset
                ));
            }
        }
        for h in &hidden {
            if h.offset < explicit_bytes || h.offset + h.size > record.kernarg_size {
                return Err(format!(
                    "{}: hidden {} at +{} lies outside the hidden block",
                    record.symbol, h.value_kind, h.offset
                ));
            }
        }
        if explicit_bytes > record.kernarg_size {
            return Err(format!(
                "{}: explicit arguments end past the {}-byte segment",
                record.symbol, record.kernarg_size
            ));
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
            guard_flag: record.guard_flag,
        })
    }

    /// Offsets of the A1 pointer arguments: the ones a recorder must bind to
    /// live allocations.
    pub fn pointer_offsets(&self) -> impl Iterator<Item = u32> + '_ {
        self.args
            .iter()
            .filter(|a| matches!(a.role, ArgRole::Pointer { .. }))
            .map(|a| a.offset)
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
    pub fn from_hit(
        record: &CorpusRecord,
        receipt: &Receipt,
        symbol: &str,
    ) -> Result<Self, String> {
        Self::from_receipt(&record.module, &record.receipt_sha256, receipt, symbol)
    }

    /// The facts for `symbol` from a pinned receipt of an externally emitted
    /// object (`railgun_corpus::PinnedReceipt`): the same record shape as a
    /// corpus hit, keyed by the same object SHA-256.
    pub fn from_pinned(pinned: &PinnedReceipt, symbol: &str) -> Result<Self, String> {
        Self::from_receipt(
            &pinned.module,
            &pinned.receipt_sha256,
            &pinned.receipt,
            symbol,
        )
    }

    fn from_receipt(
        module: &str,
        receipt_sha256: &str,
        receipt: &Receipt,
        symbol: &str,
    ) -> Result<Self, String> {
        let kernel = receipt
            .railgun
            .kernels
            .iter()
            .find(|k| k.symbol == symbol)
            .ok_or_else(|| {
                format!(
                    "receipt {} has no railgun record for {symbol}",
                    receipt.object_sha256
                )
            })?;
        let identity = receipt
            .kernels
            .iter()
            .find(|k| k.symbol == symbol)
            .ok_or_else(|| {
                format!(
                    "receipt {} has no identity for {symbol}",
                    receipt.object_sha256
                )
            })?;
        Ok(Self {
            kref: KernelRef {
                module: module.to_owned(),
                symbol: symbol.to_owned(),
                artifact_sha256: receipt.object_sha256.clone(),
                kd_sha256: identity.kd_sha256.clone(),
                receipt_sha256: receipt_sha256.to_owned(),
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
    /// A loaded 4-byte integer argument the window plan patches per window
    /// ([`PatchKind::Scalar32`], design A §3). The value and its domain live
    /// in the plan's patch site, not in the program.
    Window,
}

/// Typed overrides [`KernelProgram::encode_kernarg_into`] applies on top of
/// the authored program: the only ways a window changes a kernarg image
/// (design A §3: no untyped byte patches). `Default` reproduces
/// [`KernelProgram::kernarg_image`] exactly.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImageOverrides<'a> {
    /// [`PatchKind::GridAxis`]: the node's bound HIP grid; every hidden grid
    /// word (block counts, grid dims) is rebuilt from it.
    pub grid: Option<[u32; 3]>,
    /// The live allocation of every resource, indexed by `ResourceId`
    /// (replaces the authoring-time bindings).
    pub resources: Option<&'a [Binding]>,
    /// [`PatchKind::Interior64`]: `(explicit argument offset, interior byte
    /// offset)` replacing the authored interior of that pointer.
    pub interiors: &'a [(u32, u64)],
}

/// An input name an [`Expr`] resolved no value for, or an arithmetic fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprError {
    UnknownInput(String),
    Overflow,
}

impl std::fmt::Display for ExprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownInput(name) => write!(f, "unknown window input {name:?}"),
            Self::Overflow => f.write_str("checked arithmetic overflow"),
        }
    }
}

/// The small pure expression AST of design A §3: constant, named input,
/// checked add/multiply, saturating subtraction, min/max, division/modulo by
/// a nonzero load-time constant, ceil-div. All arithmetic is `u64`; the
/// conversion to a kernel `u32`/`i32` is a separate checked step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Const(u64),
    /// A named window input (a `WindowShape` field or a `WindowInputs`
    /// scalar).
    Input(String),
    /// Checked `a + b`.
    Add(Box<Expr>, Box<Expr>),
    /// Checked `a * b`.
    Mul(Box<Expr>, Box<Expr>),
    /// `a.saturating_sub(b)`, only where the source does.
    SubSat(Box<Expr>, Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    /// `a / c`, `c` a nonzero load-time constant.
    Div(Box<Expr>, std::num::NonZeroU64),
    /// `a % c`, `c` a nonzero load-time constant.
    Rem(Box<Expr>, std::num::NonZeroU64),
    /// `ceil(a / c)`, `c` a nonzero load-time constant (checked: no
    /// `a + c - 1` wrap).
    CeilDiv(Box<Expr>, std::num::NonZeroU64),
}

impl Expr {
    pub fn input(name: &str) -> Self {
        Self::Input(name.to_owned())
    }

    pub fn add(a: Expr, b: Expr) -> Self {
        Self::Add(Box::new(a), Box::new(b))
    }

    pub fn mul(a: Expr, b: Expr) -> Self {
        Self::Mul(Box::new(a), Box::new(b))
    }

    pub fn sub_sat(a: Expr, b: Expr) -> Self {
        Self::SubSat(Box::new(a), Box::new(b))
    }

    pub fn min(a: Expr, b: Expr) -> Self {
        Self::Min(Box::new(a), Box::new(b))
    }

    pub fn max(a: Expr, b: Expr) -> Self {
        Self::Max(Box::new(a), Box::new(b))
    }

    /// `None` for a zero divisor: it cannot be expressed.
    pub fn div(a: Expr, c: u64) -> Option<Self> {
        Some(Self::Div(Box::new(a), std::num::NonZeroU64::new(c)?))
    }

    pub fn rem(a: Expr, c: u64) -> Option<Self> {
        Some(Self::Rem(Box::new(a), std::num::NonZeroU64::new(c)?))
    }

    pub fn ceil_div(a: Expr, c: u64) -> Option<Self> {
        Some(Self::CeilDiv(Box::new(a), std::num::NonZeroU64::new(c)?))
    }

    /// Evaluate with `input` resolving the named inputs.
    pub fn eval(&self, input: &dyn Fn(&str) -> Option<u64>) -> Result<u64, ExprError> {
        Ok(match self {
            Self::Const(c) => *c,
            Self::Input(name) => {
                input(name).ok_or_else(|| ExprError::UnknownInput(name.clone()))?
            }
            Self::Add(a, b) => a
                .eval(input)?
                .checked_add(b.eval(input)?)
                .ok_or(ExprError::Overflow)?,
            Self::Mul(a, b) => a
                .eval(input)?
                .checked_mul(b.eval(input)?)
                .ok_or(ExprError::Overflow)?,
            Self::SubSat(a, b) => a.eval(input)?.saturating_sub(b.eval(input)?),
            Self::Min(a, b) => a.eval(input)?.min(b.eval(input)?),
            Self::Max(a, b) => a.eval(input)?.max(b.eval(input)?),
            Self::Div(a, c) => a.eval(input)? / c.get(),
            Self::Rem(a, c) => a.eval(input)? % c.get(),
            Self::CeilDiv(a, c) => a.eval(input)?.div_ceil(c.get()),
        })
    }

    /// Every input name the expression reads, in first-use order.
    pub fn inputs<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Self::Const(_) => {}
            Self::Input(name) => {
                if !out.contains(&name.as_str()) {
                    out.push(name);
                }
            }
            Self::Add(a, b)
            | Self::Mul(a, b)
            | Self::SubSat(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.inputs(out);
                b.inputs(out);
            }
            Self::Div(a, _) | Self::Rem(a, _) | Self::CeilDiv(a, _) => a.inputs(out),
        }
    }
}

/// A grid axis of a [`PatchKind::GridAxis`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    pub fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// One loaded 4-byte integer argument of a program node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarField {
    pub node: usize,
    /// Explicit-argument byte offset.
    pub offset: u32,
}

/// The declared patch kinds of design A §3 (the only ones; `GdnFrame` stays
/// a [`WordRole`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatchKind {
    /// `Scalar32 {field, checked expression, domain}`: a loaded scalar
    /// argument (declared in the program as a [`WordRole::Window`] word).
    /// The value must lie in `domain` and fit `u32`.
    Scalar32 {
        field: ScalarField,
        expr: Expr,
        domain: std::ops::RangeInclusive<u64>,
    },
    /// `Interior64 {resource, checked byte_offset expression,
    /// required_extent}`: the pointer argument at `arg_offset` (authored as
    /// an `Arg::Resource` of `resource`) becomes `base(resource) +
    /// byte_offset`; `byte_offset + required_extent` must lie inside the
    /// bound allocation.
    Interior64 {
        node: usize,
        arg_offset: u32,
        resource: ResourceId,
        byte_offset: Expr,
        required_extent: Expr,
    },
    /// `GridAxis {axis, checked expression, domain}`: the HIP grid of `node`
    /// on `axis` (in workgroups; the value must be nonzero, in `domain`, and
    /// fit `u32`). Every live implicit grid word is rebuilt from the
    /// resulting geometry.
    GridAxis {
        node: usize,
        axis: Axis,
        expr: Expr,
        domain: std::ops::RangeInclusive<u64>,
    },
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
    Resource {
        resource: ResourceId,
        interior: u64,
    },
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
    Graph {
        reason: String,
    },
    /// No usable facts: a full barrier on both sides (§1.4 rule 4).
    HipDirect {
        reason: String,
    },
    /// A `Conditional` kernel (§A9.2) whose bind witness has not been
    /// consumed: PM4-ineligible, a full barrier like `HipDirect`.
    Conditional {
        reason: String,
    },
    /// Lab only (`railgun::admission`): differential-verified, certificate
    /// pending. `pending` is the exact certificate refusal the node would
    /// otherwise carry (its `HipDirect`/`Conditional` reason). Built only by
    /// `KernelProgram::admit_lab_pending` (`feature = "lab"`); **not** PM4:
    /// every product route refuses it through [`Lowering::is_pm4`], and only
    /// the plan and the co-resident groups see it through
    /// [`Lowering::is_pm4_lowerable`].
    LabPending {
        pending: String,
    },
}

impl Lowering {
    /// A proven PM4 node. Strict: `LabPending` is not PM4.
    pub fn is_pm4(&self) -> bool {
        matches!(self, Self::Pm4)
    }

    /// PM4 can be emitted for the node: proven PM4, or (lab builds only)
    /// `LabPending`. Without `feature = "lab"` this is [`Lowering::is_pm4`],
    /// so a product plan is unchanged even for a hand-built `LabPending`.
    pub fn is_pm4_lowerable(&self) -> bool {
        #[cfg(feature = "lab")]
        {
            matches!(self, Self::Pm4 | Self::LabPending { .. })
        }
        #[cfg(not(feature = "lab"))]
        {
            self.is_pm4()
        }
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
    /// The guarded case of a `Conditional` kernel (a copy of the receipt's
    /// facts, not a mutation of them). A node carrying it is admitted to a
    /// window only while the window holds a matching
    /// [`crate::index_bind::IndexedBindWitness`].
    pub indexed: Option<IndexedNode>,
    /// The sealed-pointer-table case of a `Conditional` kernel (a copy of the
    /// receipt's facts; kept apart from `indexed`). A node carrying it is
    /// admitted to a window only while the window holds a matching
    /// [`crate::sealed_table::SealedTableWitness`].
    pub table: Option<TableNode>,
    /// #867: the ControlWord resource whose clear flag makes this launch a
    /// no-op, from the object's guard certificate (`KernelFacts::guard_flag`)
    /// with its pointer bound to the start of that resource.
    pub guard: Option<ResourceId>,
    /// The descriptor-arena case of a `Conditional` kernel (a copy of the
    /// receipt's facts): the node is PM4 only once
    /// [`KernelProgram::bind_records`] has bound every arena.
    pub arena: Option<ArenaNode>,
}

/// What a window needs to bind a `Conditional` node: the explicit arguments
/// of the loaded object, the guarded case and its receipt identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedNode {
    pub args: Vec<ArgFact>,
    pub effects: Vec<IndexedArgEffect>,
    pub guards: Vec<ArgGuard>,
    pub witnesses: Vec<String>,
    pub case_digest: String,
}

/// What a window needs to bind a sealed-table `Conditional` node: the explicit
/// arguments of the loaded object, the sealed tables and the case's receipt
/// identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableNode {
    pub args: Vec<ArgFact>,
    pub tables: Vec<SealedTableFact>,
    pub witnesses: Vec<String>,
    pub case_digest: String,
}

/// The lowering reason of a node whose descriptor arenas are not yet bound.
pub(crate) const ARENA_UNBOUND: &str = "descriptor_arena records not bound";

/// What [`KernelProgram::bind_records`] needs to discharge a
/// `descriptor_arena` `Conditional` node: the arena facts of the loaded
/// object, and which of them have been bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaNode {
    pub arenas: Vec<ArenaFact>,
    /// The proven dispatcher protocol of a persistent pool kernel; the node
    /// is only admitted to a group together with a task-graph proof
    /// ([`crate::task_graph::TaskGraph::check`]).
    pub task_protocol: Option<crate::task_graph::TaskProtocolFact>,
    /// Parallel to `arenas`: the arena's records were bound and the
    /// resulting effects added to the node.
    pub bound: Vec<bool>,
    /// The sealed tables `bind_records` discharged for the records of the
    /// arena templates (empty until bound, and for a template whose records
    /// all carry a zero bound). A node with any needs the `sealed_table`
    /// witness of a window bind ([`KernelNode::needs_sealed_table_witness`]).
    pub sealed: Vec<ArenaSealedBinding>,
    /// The receipt's required witness kinds (digest-bound).
    pub witnesses: Vec<String>,
    pub case_digest: String,
}

impl ArenaNode {
    pub fn all_bound(&self) -> bool {
        self.bound.iter().all(|b| *b)
    }
}

/// One sealed pointer table of an arena template, as
/// [`KernelProgram::bind_records`] discharged it: the table resource the
/// records' table field names, the sealed identity the registry held for it
/// and the program resources its entries' pointees are. The window bind
/// re-verifies the identity against the live allocation, generation and
/// content ([`crate::sealed_table::SealedTableRegistry::bind_sealed_table`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaSealedBinding {
    /// Index into [`ArenaNode::arenas`].
    pub arena: usize,
    pub template: u32,
    /// Record offset of the table pointer field.
    pub table_arg_offset: u32,
    /// The program resource the table field names (interior 0).
    pub resource: ResourceId,
    /// The largest nonzero declared bound over the records naming `resource`
    /// (each is `<= entries`).
    pub bound: u32,
    pub null_guarded: bool,
    /// The sealed table's identity at bind time.
    pub allocation: Binding,
    pub entries: u32,
    pub generation: u64,
    pub content_sha256: [u8; 32],
    /// Program resources equal to a pointee allocation of the sealed table
    /// (a pointee that is no program resource is read by no node's effects
    /// and cannot be written by any program node).
    pub pointees: Vec<ResourceId>,
    /// The read classes the certified body applies through an entry.
    pub accesses: Vec<MemClass>,
}

impl KernelNode {
    /// The node carries sealed tables a window must discharge with a
    /// [`crate::sealed_table::SealedTableWitness`]: its own `table` case, or
    /// tables of its bound arena templates.
    pub fn needs_sealed_table_witness(&self) -> bool {
        self.table.is_some() || self.arena.as_ref().is_some_and(|a| !a.sealed.is_empty())
    }
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
    /// #867 IF bodies, disjoint and in node order
    /// ([`KernelProgram::add_conditional`]).
    pub conditionals: Vec<Conditional>,
    /// What a resource is beyond "a plain allocation" (a resource absent
    /// here is one, which is today's semantics). Written only by
    /// [`KernelProgram::declare_event_cells`],
    /// [`KernelProgram::declare_task_arena`] and
    /// [`KernelProgram::bind_records`], each of which validates the nodes'
    /// effects against the kind.
    pub kinds: BTreeMap<ResourceId, crate::ResourceKind>,
    /// Persistent-kernel groups that run concurrently, disjoint and in node
    /// order ([`KernelProgram::add_coresident_group`]).
    pub groups: Vec<crate::coresident::CoResidentGroup>,
}

/// #867: how a [`Conditional`] body is skipped when its flag is clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CondLowering {
    /// The tape is unchanged: every body node is a guarded kernel whose
    /// certified no-op path the clear flag takes (hot path).
    KernelGuard,
    /// `COND_EXEC` discards the body's command dwords, behind a
    /// `RELEASE_MEM`/`WAIT_REG_MEM` fence that makes the flag fresh (coarse
    /// segments only: the fence drains the queue).
    CpSkip,
}

/// #867 IF on a device-written ControlWord: nodes `body` take effect only
/// when the first `u32` of `flag` is nonzero when the body starts. A clear
/// flag leaves every byte the body would write unchanged. The planner
/// treats the body as executed either way (its effects stay in every edge).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Conditional {
    pub flag: ResourceId,
    pub body: std::ops::Range<usize>,
    pub lowering: CondLowering,
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
        let conditional = match &f.status {
            FactStatus::Conditional(c) => {
                for g in &c.guards {
                    if let Err(why) = guard_holds(g, launch) {
                        refusals.push(format!(
                            "conditional guard {}@{} fails: {why}",
                            g.kind, g.arg_offset
                        ));
                    }
                }
                Some(c)
            }
            _ => None,
        };
        if f.differential == "fail" {
            refusals.push("G8 differential fails".to_owned());
        }
        let explicit = f.explicit_bytes as usize;
        if launch.kernarg.len() < explicit {
            refusals.push(format!(
                "recorded kernarg is {} bytes, the object's explicit segment {explicit}",
                launch.kernarg.len()
            ));
            nodes.push(uncertified_node(launch, refusals.join("; ")));
            continue;
        }
        if launch.kernarg[explicit..].iter().any(|b| *b != 0) {
            refusals.push(format!(
                "recorded kernarg carries nonzero bytes past the {explicit}-byte explicit segment"
            ));
        }
        for h in f
            .hidden
            .iter()
            .filter(|h| h.loaded && h.kind == HiddenKind::Other)
        {
            refusals.push(format!(
                "loads hidden {} which railgun does not supply",
                h.value_kind
            ));
        }
        let declared: BTreeMap<u32, WordRole> =
            launch.declared.iter().map(|w| (w.offset, w.role)).collect();
        let mut used_words = 0usize;
        let mut args = Vec::with_capacity(f.args.len() * 2);
        let mut effects = Vec::new();
        let mut cursor = 0usize;
        let mut non_commutative_atomic = false;
        for arg in &f.args {
            let (start, end) = (arg.offset as usize, (arg.offset + arg.size) as usize);
            if start > cursor {
                args.push((
                    cursor as u32,
                    Arg::Const(launch.kernarg[cursor..start].to_vec()),
                ));
            }
            let bytes = &launch.kernarg[start..end];
            cursor = end;
            if let Some(role) = declared.get(&arg.offset) {
                used_words += 1;
                if arg.role == ArgRole::Integer && arg.size == 4 {
                    let authored = u32::from_le_bytes(bytes.try_into().expect("4-byte argument"));
                    args.push((arg.offset, Arg::Word(WordId(words.len() as u32))));
                    words.push(ProgramWord {
                        node: index,
                        offset: arg.offset,
                        role: *role,
                        authored,
                    });
                } else {
                    refusals.push(format!(
                        "declared word at +{} is not a loaded 4-byte integer argument (A4)",
                        arg.offset
                    ));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                }
                continue;
            }
            let ArgRole::Pointer { classes, .. } = &arg.role else {
                args.push((arg.offset, Arg::Const(bytes.to_vec())));
                continue;
            };
            // A conditional kernel's indexed effects add their base access
            // and index-source reads to the argument's classes (still
            // allocation-wide; no narrower hazard region).
            let mut classes = classes.clone();
            if let Some(c) = conditional {
                for e in &c.indexed {
                    if e.base_arg_offset == arg.offset && !classes.contains(&e.class) {
                        classes.push(e.class);
                    }
                    for l in &e.index_loads {
                        if l.source_arg_offset == arg.offset && !classes.contains(&l.class) {
                            classes.push(l.class);
                        }
                    }
                }
                // A sealed table's own entry loads are reads of the table
                // argument; the table is immutable, so a write class on it
                // (from any source) refuses the node.
                for t in c.tables.iter().filter(|t| t.table_arg_offset == arg.offset) {
                    for e in &t.entries {
                        if !classes.contains(&e.class) {
                            classes.push(e.class);
                        }
                    }
                    if classes.iter().any(|k| k.writes()) {
                        refusals.push(format!(
                            "sealed table argument +{} has write effects",
                            arg.offset
                        ));
                    }
                }
            }
            if arg.size != 8 {
                refusals.push(format!(
                    "pointer argument +{} is {} bytes",
                    arg.offset, arg.size
                ));
                args.push((arg.offset, Arg::Const(bytes.to_vec())));
                continue;
            }
            let value = u64::from_le_bytes(bytes.try_into().expect("8-byte argument"));
            if value == 0 {
                // A null pointer is a constant with no effect.
                args.push((arg.offset, Arg::Const(bytes.to_vec())));
                continue;
            }
            let allocation = launch
                .allocations
                .iter()
                .find(|(offset, _)| *offset == arg.offset)
                .map(|(_, a)| a);
            let binding = match allocation {
                Some(Ok(binding))
                    if value >= binding.base && value - binding.base < binding.bytes.max(1) =>
                {
                    *binding
                }
                Some(Ok(binding)) => {
                    refusals.push(format!(
                        "pointer +{} = {value:#x} lies outside its reported allocation {:#x}+{}",
                        arg.offset, binding.base, binding.bytes
                    ));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                    continue;
                }
                Some(Err(reason)) => {
                    refusals.push(format!(
                        "pointer +{} = {value:#x} has no live allocation: {reason}",
                        arg.offset
                    ));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                    continue;
                }
                None => {
                    refusals.push(format!(
                        "pointer +{} = {value:#x} was not probed",
                        arg.offset
                    ));
                    args.push((arg.offset, Arg::Const(bytes.to_vec())));
                    continue;
                }
            };
            let resource = *by_allocation
                .entry((binding.base, binding.bytes))
                .or_insert_with(|| {
                    resources.push(binding);
                    ResourceId(resources.len() as u32 - 1)
                });
            args.push((
                arg.offset,
                Arg::Resource {
                    resource,
                    interior: value - binding.base,
                },
            ));
            if !classes.is_empty() {
                non_commutative_atomic |= classes.contains(&MemClass::Atomic);
                effects.push(RegionEffect {
                    arg_offset: arg.offset,
                    resource,
                    classes,
                });
            }
        }
        if cursor < explicit {
            args.push((
                cursor as u32,
                Arg::Const(launch.kernarg[cursor..explicit].to_vec()),
            ));
        }
        if used_words != declared.len() {
            refusals.push(format!(
                "{} declared word(s) name no argument of the object",
                declared.len() - used_words
            ));
        }
        if let Some(c) = conditional {
            for t in &c.tables {
                // A launch whose bound is zero never reads the table (the sealed-table
                // bind discharges it as vacuous without an allocation), so an argument
                // that is no allocation - a null optional pointer - is fine there.
                let zero_bound = match t.bound {
                    TableBound::Const(c) => c == 0,
                    TableBound::Kernarg(at) => launch
                        .kernarg
                        .get(at as usize..at as usize + 4)
                        .is_some_and(|b| b == [0u8; 4]),
                };
                let is_resource = args.iter().any(|(offset, arg)| {
                    *offset == t.table_arg_offset && matches!(arg, Arg::Resource { .. })
                });
                if !is_resource && !zero_bound {
                    refusals.push(format!(
                        "sealed table argument +{} is not a bound allocation",
                        t.table_arg_offset
                    ));
                }
                if let TableBound::Kernarg(at) = t.bound {
                    if at as usize + 4 > explicit {
                        refusals.push(format!(
                            "sealed table bound +{at} lies outside the {explicit}-byte explicit kernarg segment"
                        ));
                    }
                }
            }
            for a in &c.arenas {
                let is_base = args.iter().any(|(offset, arg)| {
                    *offset == a.arena_arg_offset
                        && matches!(arg, Arg::Resource { interior: 0, .. })
                });
                if !is_base {
                    refusals.push(format!(
                        "descriptor arena argument +{} is not a bound allocation base (interior 0)",
                        a.arena_arg_offset
                    ));
                }
            }
        }
        let lowering = if !refusals.is_empty() {
            Lowering::HipDirect {
                reason: refusals.join("; "),
            }
        } else if let Some(c) = conditional {
            let mut reasons = Vec::new();
            match (has_indexed_case(c), !c.tables.is_empty()) {
                (true, false) => reasons.push("indexed_bind witness not consumed"),
                (false, true) => reasons.push("sealed_table witness not consumed"),
                (true, true) => {
                    reasons.push("indexed_bind and sealed_table witnesses not consumed")
                }
                (false, false) => {}
            }
            if !c.arenas.is_empty() {
                reasons.push(ARENA_UNBOUND);
            }
            Lowering::Conditional {
                reason: reasons.join("; "),
            }
        } else if non_commutative_atomic {
            Lowering::Graph {
                reason: "atomic without a commutativity certificate".to_owned(),
            }
        } else {
            Lowering::Pm4
        };
        let guard = f.guard_flag.and_then(|at| {
            args.iter().find_map(|(offset, arg)| match arg {
                Arg::Resource {
                    resource,
                    interior: 0,
                } if *offset == at => Some(*resource),
                _ => None,
            })
        });
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
            indexed: conditional
                .filter(|c| has_indexed_case(c))
                .map(|c| IndexedNode {
                    args: f.args.clone(),
                    effects: c.indexed.clone(),
                    guards: c.guards.clone(),
                    witnesses: c.witnesses.clone(),
                    case_digest: c.case_digest.clone(),
                }),
            table: conditional
                .filter(|c| !c.tables.is_empty())
                .map(|c| TableNode {
                    args: f.args.clone(),
                    tables: c.tables.clone(),
                    witnesses: c.witnesses.clone(),
                    case_digest: c.case_digest.clone(),
                }),
            arena: conditional
                .filter(|c| !c.arenas.is_empty())
                .map(|c| ArenaNode {
                    arenas: c.arenas.clone(),
                    task_protocol: c.task_protocol.clone(),
                    bound: vec![false; c.arenas.len()],
                    sealed: Vec::new(),
                    witnesses: c.witnesses.clone(),
                    case_digest: c.case_digest.clone(),
                }),
            guard,
        });
    }
    KernelProgram {
        arch: arch.to_owned(),
        resources,
        words,
        nodes,
        conditionals: Vec::new(),
        kinds: BTreeMap::new(),
        groups: Vec::new(),
    }
}

/// Evaluate one receipt guard against the recorded launch. `grid`/`block`
/// guards whose value is symbolic (not `x,y,z` integers) are enforced by the
/// window bind, which recomputes them from the bound arguments.
fn guard_holds(g: &ArgGuard, launch: &LaunchRecord) -> Result<(), String> {
    let at = g.arg_offset as usize;
    let size = g.size as usize;
    match g.kind.as_str() {
        "ptr_null" => {
            let bytes = launch
                .kernarg
                .get(at..at + size)
                .ok_or("outside the kernarg")?;
            if bytes.iter().all(|b| *b == 0) {
                Ok(())
            } else {
                Err("pointer is not null".into())
            }
        }
        // `kernarg_eq` is railgun-cert's serialization of the analyzer's
        // `Guard::KernargEq` (a null optional pointer is `size 8, value 0`).
        "scalar_eq" | "kernarg_eq" => {
            let want: u64 = g
                .value
                .parse()
                .map_err(|_| format!("bad value {:?}", g.value))?;
            if size == 0 || size > 8 {
                return Err(format!("bad size {size}"));
            }
            let bytes = launch
                .kernarg
                .get(at..at + size)
                .ok_or("outside the kernarg")?;
            let mut le = [0u8; 8];
            le[..size].copy_from_slice(bytes);
            let got = u64::from_le_bytes(le);
            if got == want {
                Ok(())
            } else {
                Err(format!("kernarg is {got}, guard {want}"))
            }
        }
        kind @ ("grid" | "block") => {
            let dims: Option<Vec<u32>> =
                g.value.split(',').map(|v| v.trim().parse().ok()).collect();
            match dims.as_deref() {
                Some(&[x, y, z]) => {
                    let have = if kind == "grid" {
                        launch.grid
                    } else {
                        launch.block
                    };
                    if have == [x, y, z] {
                        Ok(())
                    } else {
                        Err(format!("{have:?} != [{x}, {y}, {z}]"))
                    }
                }
                _ => Ok(()),
            }
        }
        // railgun-cert's `Guard::GridDim` / `Guard::BlockDim`: `arg_offset`
        // carries the axis, `value` the workgroup count / block extent.
        kind @ ("grid_dim" | "block_dim") => {
            let axis = g.arg_offset as usize;
            let want: u32 = g
                .value
                .parse()
                .map_err(|_| format!("bad value {:?}", g.value))?;
            let have = if kind == "grid_dim" {
                launch.grid
            } else {
                launch.block
            };
            match have.get(axis) {
                Some(&v) if v == want => Ok(()),
                Some(&v) => Err(format!("{kind}[{axis}] is {v}, guard {want}")),
                None => Err(format!("bad axis {axis}")),
            }
        }
        other => Err(format!("unknown guard kind {other:?}")),
    }
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
        indexed: None,
        table: None,
        arena: None,
        guard: None,
    }
}

impl KernelProgram {
    /// The resource of allocation `binding`, added when the program does not
    /// name it yet: a binding-record field may point into an allocation no
    /// explicit kernarg pointer reaches ([`Self::bind_records`]). `binding`
    /// must be the whole live allocation the runtime reports
    /// (`hipMemGetAddressRange`), as [`author`] binds every pointer; the same
    /// `(base, bytes)` yields the same id, and an allocation overlapping a
    /// different one already in the program is refused.
    pub fn resource_for(&mut self, binding: Binding) -> Result<ResourceId, String> {
        if binding.bytes == 0 || binding.base.checked_add(binding.bytes).is_none() {
            return Err(format!(
                "allocation {:#x}+{} is empty or wraps",
                binding.base, binding.bytes
            ));
        }
        if let Some(i) = self.resources.iter().position(|r| *r == binding) {
            return Ok(ResourceId(i as u32));
        }
        if let Some((i, r)) = self.resources.iter().enumerate().find(|(_, r)| {
            binding.base < r.base.saturating_add(r.bytes) && r.base < binding.base + binding.bytes
        }) {
            return Err(format!(
                "allocation {:#x}+{} overlaps resource {i} ({:#x}+{})",
                binding.base, binding.bytes, r.base, r.bytes
            ));
        }
        self.resources.push(binding);
        Ok(ResourceId(self.resources.len() as u32 - 1))
    }

    /// Make `c.body` an IF on `c.flag` (#867). Refused unless: the body is
    /// non-empty, in range and disjoint from every other conditional (no
    /// nesting); `flag` is a resource of at least 4 bytes; every body node
    /// is PM4. `KernelGuard` also needs every body node's certified guard on
    /// `flag`, and a node writing `flag` must be the body's last and a single
    /// wave (`grid × block ≤ 32`): the guard runs once per wave, so an earlier
    /// node's write, or a later wave of the writer, would split the decision.
    /// `CpSkip` needs [`crate::plan::cp_skip_admitted`] for the arch.
    pub fn add_conditional(&mut self, c: Conditional) -> Result<(), String> {
        let Conditional {
            flag,
            ref body,
            lowering,
        } = c;
        if body.is_empty() || body.end > self.nodes.len() {
            return Err(format!(
                "conditional body {body:?} is empty or past {} nodes",
                self.nodes.len()
            ));
        }
        if let Some(other) = self
            .conditionals
            .iter()
            .find(|o| o.body.start < body.end && body.start < o.body.end)
        {
            return Err(format!(
                "conditional body {body:?} overlaps {:?} (nesting is not supported)",
                other.body
            ));
        }
        if let Some(other) = self
            .groups
            .iter()
            .find(|g| g.nodes.start < body.end && body.start < g.nodes.end)
        {
            return Err(format!(
                "conditional body {body:?} overlaps co-resident group {:?}",
                other.nodes
            ));
        }
        if self
            .resources
            .get(flag.0 as usize)
            .is_none_or(|b| b.bytes < 4)
        {
            return Err(format!(
                "flag resource {} is not a 4-byte ControlWord",
                flag.0
            ));
        }
        for (i, node) in self.nodes[body.clone()].iter().enumerate() {
            let j = body.start + i;
            if !node.lowering.is_pm4() {
                return Err(format!(
                    "node {j} ({}) in a conditional body is not PM4",
                    node.symbol
                ));
            }
            if lowering == CondLowering::KernelGuard {
                if node.guard != Some(flag) {
                    return Err(format!(
                        "node {j} ({}) has no certified guard on flag resource {}",
                        node.symbol, flag.0
                    ));
                }
                let threads: u64 = node
                    .grid
                    .iter()
                    .chain(&node.block)
                    .map(|&d| d as u64)
                    .product();
                if (j + 1 < body.end || threads > 32)
                    && node
                        .effects
                        .iter()
                        .any(|e| e.resource == flag && e.writes())
                {
                    return Err(format!(
                        "node {j} ({}) writes the flag but is not the body's last \
                         single-wave node ({threads} threads)",
                        node.symbol
                    ));
                }
            }
        }
        if lowering == CondLowering::CpSkip && !crate::plan::cp_skip_admitted(&self.arch) {
            return Err(format!("COND_EXEC skip is not verified on {}", self.arch));
        }
        let at = self
            .conditionals
            .partition_point(|o| o.body.start < body.start);
        self.conditionals.insert(at, c);
        Ok(())
    }

    /// #867 bounded WHILE: `iterations` copies of a `len`-node body laid out
    /// back to back from node `first`, each an IF on `flag`. The body's last
    /// node writes the continue decision into `flag`; once it is clear every
    /// later copy is skipped, so the copies run while the flag stays set, at
    /// most `iterations` times.
    pub fn bounded_while(
        &mut self,
        flag: ResourceId,
        first: usize,
        len: usize,
        iterations: usize,
        lowering: CondLowering,
    ) -> Result<(), String> {
        let before = self.conditionals.clone();
        for k in 0..iterations {
            let start = first + k * len;
            if let Err(e) = self.add_conditional(Conditional {
                flag,
                body: start..start + len,
                lowering,
            }) {
                self.conditionals = before;
                return Err(e);
            }
        }
        Ok(())
    }

    /// The node's full kernarg segment as the loaded object lays it out
    /// (`loader_bytes`, which must equal the certified segment): explicit
    /// arguments re-encoded from the program (resources from their bindings,
    /// words from `words`), then the declared hidden arguments.
    pub fn kernarg_image(
        &self,
        node: usize,
        loader_bytes: usize,
        words: &[u32],
    ) -> Result<Vec<u8>, String> {
        let mut image = vec![0u8; loader_bytes];
        self.encode_kernarg_into(node, words, &ImageOverrides::default(), &mut image)?;
        Ok(image)
    }

    /// The single full-image encoder behind [`Self::kernarg_image`] (design A
    /// §3: "`KernelProgram::kernarg_image` remains the only full-image
    /// encoder"). `out` is the loaded object's kernarg segment (`out.len()`
    /// is the loader's size and must equal the certified one); it is zeroed
    /// and rewritten in place, so a retired buffer is re-encoded without an
    /// allocation. `overrides` supplies the typed window patches of §3:
    /// resource bindings, `Interior64` offsets and the `GridAxis` geometry,
    /// from which every hidden grid word is rebuilt. Scalar words arrive in
    /// `words`. Nothing else is writable.
    pub fn encode_kernarg_into(
        &self,
        node: usize,
        words: &[u32],
        overrides: &ImageOverrides<'_>,
        out: &mut [u8],
    ) -> Result<(), String> {
        let n = self
            .nodes
            .get(node)
            .ok_or_else(|| format!("no node {node}"))?;
        let segment = n
            .segment_bytes
            .ok_or_else(|| format!("{}: no certified kernarg segment", n.symbol))?
            as usize;
        let loader_bytes = out.len();
        if loader_bytes != segment {
            return Err(format!("{}: the loaded object's kernarg segment is {loader_bytes} B, the certified one {segment} B", n.symbol));
        }
        let resources = overrides.resources.unwrap_or(&self.resources);
        let grid = overrides.grid.unwrap_or(n.grid);
        if overrides.grid.is_some() && grid.contains(&0) {
            return Err(format!("{}: grid {grid:?} has an empty axis", n.symbol));
        }
        let image = out;
        image.fill(0);
        for (offset, arg) in &n.args {
            let at = *offset as usize;
            match arg {
                Arg::Const(bytes) => image[at..at + bytes.len()].copy_from_slice(bytes),
                Arg::Resource { resource, interior } => {
                    let binding = resources
                        .get(resource.0 as usize)
                        .ok_or_else(|| format!("unknown resource {}", resource.0))?;
                    let interior = overrides
                        .interiors
                        .iter()
                        .find(|(arg_offset, _)| arg_offset == offset)
                        .map_or(*interior, |(_, interior)| *interior);
                    let address = binding
                        .base
                        .checked_add(interior)
                        .ok_or_else(|| "pointer overflows".to_owned())?;
                    image[at..at + 8].copy_from_slice(&address.to_le_bytes());
                }
                Arg::Word(id) => {
                    let value = words
                        .get(id.0 as usize)
                        .ok_or_else(|| format!("no value for word {}", id.0))?;
                    image[at..at + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        let dims: u64 = if grid[2] != 1 || n.block[2] != 1 {
            3
        } else if grid[1] != 1 || n.block[1] != 1 {
            2
        } else {
            1
        };
        for h in &n.hidden {
            let value: u64 = match h.kind {
                HiddenKind::BlockCount(axis) => u64::from(grid[axis as usize]),
                HiddenKind::GroupSize(axis) => u64::from(n.block[axis as usize]),
                // HIP grids count workgroups: no partial groups, no offset.
                HiddenKind::Remainder(_) | HiddenKind::GlobalOffset(_) | HiddenKind::Other => 0,
                HiddenKind::GridDims => dims,
                HiddenKind::DynamicLdsSize => u64::from(n.dynamic_lds),
            };
            let size = h.size as usize;
            if size > 8 || (size < 8 && value >> (8 * size) != 0) {
                return Err(format!(
                    "{}: {} = {value} does not fit {size} bytes",
                    n.symbol, h.value_kind
                ));
            }
            let at = h.offset as usize;
            image[at..at + size].copy_from_slice(&value.to_le_bytes()[..size]);
        }
        Ok(())
    }

    /// Authored value of every word, indexed by `WordId`.
    pub fn authored_words(&self) -> Vec<u32> {
        self.words.iter().map(|w| w.authored).collect()
    }

    /// Indices of the nodes that need an
    /// [`crate::index_bind::IndexedBindWitness`] (receipt `Conditional` with
    /// an `indexed_bind` case). Sealed-table nodes are
    /// [`Self::sealed_table_nodes`].
    pub fn conditional_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.indexed.is_some())
            .map(|(i, _)| i)
            .collect()
    }

    /// Indices of the nodes that need a
    /// [`crate::sealed_table::SealedTableWitness`] (receipt `Conditional`
    /// with sealed tables).
    pub fn sealed_table_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.needs_sealed_table_witness())
            .map(|(i, _)| i)
            .collect()
    }

    /// The program a window plan is laid out against *assuming* the bind
    /// witnesses of `kinds`: every `Conditional` node whose required witness
    /// kinds (`indexed_bind` for an `indexed` case, `sealed_table` for a
    /// `table` case) are all in `kinds` is lowered as PM4. A node that still
    /// needs another kind stays `Conditional`. This is a derived value; the
    /// cached node facts and this program are never promoted to `Proven`. The
    /// resulting plan keeps the nodes' cases, and `PreparedWindow` refuses to
    /// submit without a witness for every such node.
    pub fn assume_witnesses(&self, kinds: &[&str]) -> KernelProgram {
        let mut p = self.clone();
        for n in &mut p.nodes {
            if (n.indexed.is_none() && n.table.is_none() && !n.needs_sealed_table_witness())
                || !matches!(n.lowering, Lowering::Conditional { .. })
            {
                continue;
            }
            // A descriptor arena is discharged by `bind_records` alone; no
            // window witness stands in for it. The sealed tables its records
            // carry are the `sealed_table` kind.
            let covered = (n.indexed.is_none() || kinds.contains(&WITNESS_INDEXED_BIND))
                && ((n.table.is_none() && !n.needs_sealed_table_witness())
                    || kinds.contains(&WITNESS_SEALED_TABLE))
                && n.arena.as_ref().is_none_or(ArenaNode::all_bound);
            if covered {
                n.lowering = Lowering::Pm4;
            }
        }
        p
    }

    /// [`Self::assume_witnesses`] for the `indexed_bind` kind: every
    /// `Conditional` node that carries only its indexed case is lowered as
    /// PM4. A node that also needs a sealed table stays `Conditional`.
    pub fn assume_indexed_witness(&self) -> KernelProgram {
        self.assume_witnesses(&[WITNESS_INDEXED_BIND])
    }

    /// [`Self::assume_witnesses`] for the `sealed_table` kind: every
    /// `Conditional` node that carries only its sealed-table case is lowered
    /// as PM4. A node that also needs an indexed bind stays `Conditional`.
    pub fn assume_sealed_table_witness(&self) -> KernelProgram {
        self.assume_witnesses(&[WITNESS_SEALED_TABLE])
    }
}

/// Whether `c` carries an `indexed_bind` case. A conditional fact with
/// none of the three families is the legacy indexed shape (hand-built
/// facts).
fn has_indexed_case(c: &ConditionalFacts) -> bool {
    !c.indexed.is_empty() || (c.tables.is_empty() && c.arenas.is_empty())
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
                guard_flag: None,
            },
        })
    }

    pub(crate) fn ptr(offset: u32, classes: &[MemClass]) -> ArgFact {
        ArgFact {
            offset,
            size: 8,
            role: ArgRole::Pointer {
                classes: classes.to_vec(),
                read_cache: ReadCache::VmemOnly,
            },
        }
    }

    pub(crate) fn launch(
        pointers: &[(u32, u64)],
        explicit: u32,
        args: Vec<ArgFact>,
    ) -> LaunchRecord {
        let mut kernarg = vec![0u8; explicit.next_multiple_of(16) as usize];
        let mut allocations = Vec::new();
        for &(offset, value) in pointers {
            kernarg[offset as usize..offset as usize + 8].copy_from_slice(&value.to_le_bytes());
            allocations.push((
                offset,
                Ok(Binding {
                    base: value & !0xfff,
                    bytes: 0x10000,
                }),
            ));
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
        let args = vec![
            ptr(0, &[MemClass::VmemLoad]),
            ArgFact {
                offset: 8,
                size: 4,
                role: ArgRole::Integer,
            },
        ];
        let mut good = launch(&[(0, 0x7000_0100)], 12, args.clone());
        good.kernarg[8..12].copy_from_slice(&41u32.to_le_bytes());
        good.declared = vec![DeclaredWord {
            offset: 8,
            role: WordRole::GdnFrame { frames: 1 },
        }];
        let program = author("gfx1201", &[good.clone()]);
        assert_eq!(program.nodes[0].lowering, Lowering::Pm4);
        assert_eq!(program.words[0].authored, 41);
        let image = program.kernarg_image(0, 12, &[99]).unwrap();
        assert_eq!(&image[0..8], &0x7000_0100u64.to_le_bytes());
        assert_eq!(
            &image[8..12],
            &99u32.to_le_bytes(),
            "the word is patched, not the recorded constant"
        );

        let mut bad = good;
        bad.declared = vec![DeclaredWord {
            offset: 0,
            role: WordRole::GdnFrame { frames: 1 },
        }];
        let program = author("gfx1201", &[bad]);
        assert!(
            matches!(&program.nodes[0].lowering, Lowering::HipDirect { reason } if reason.contains("A4"))
        );
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
        for (node, needle) in
            program
                .nodes
                .iter()
                .zip(["receipt_missing", "A1 Unknown", "not probed"])
        {
            assert!(
                matches!(&node.lowering, Lowering::HipDirect { reason } if reason.contains(needle)),
                "{:?}",
                node.lowering
            );
        }
    }

    fn hidden(offset: u32, size: u32, value_kind: &str, loaded: bool) -> HiddenArg {
        HiddenArg {
            offset,
            size,
            kind: HiddenKind::parse(value_kind).unwrap(),
            value_kind: value_kind.into(),
            loaded,
        }
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
        assert_eq!(
            u32::from_le_bytes(image[128..132].try_into().unwrap()),
            1024
        );
        // One nonzero byte each: the pointer, both block counts, the group size, the dims, the LDS size.
        assert_eq!(
            image.iter().filter(|b| **b != 0).count(),
            6,
            "nothing undeclared is written"
        );
        assert!(
            program.kernarg_image(0, 272, &[]).is_err(),
            "a loaded object with another segment size is refused"
        );

        f.facts
            .hidden
            .push(hidden(200, 8, "hidden_queue_ptr", true));
        l.facts = Ok(Arc::new(f));
        let program = author("gfx1201", &[l]);
        assert!(
            matches!(&program.nodes[0].lowering, Lowering::HipDirect { reason } if reason.contains("hidden_queue_ptr"))
        );
    }

    /// `l` with a guard certificate on its pointer at +0.
    pub(crate) fn guarded(mut l: LaunchRecord) -> LaunchRecord {
        let mut f = (*l.facts.clone().unwrap()).clone();
        f.facts.guard_flag = Some(0);
        l.facts = Ok(Arc::new(f));
        l
    }

    #[test]
    fn kernel_guard_needs_the_certificate_on_the_flag_and_a_single_wave_last_writer() {
        let (flag, x, y) = (0x7000_0000u64, 0x7100_0000u64, 0x7200_0000u64);
        let body = vec![
            ptr(0, &[MemClass::SmemLoad]),
            ptr(8, &[MemClass::VmemStore]),
        ];
        let step = vec![ptr(0, &[MemClass::SmemLoad, MemClass::VmemStore])];
        let mut writer = guarded(launch(&[(0, flag)], 8, step.clone()));
        (writer.grid, writer.block) = ([1, 1, 1], [32, 1, 1]);
        let launches = vec![
            guarded(launch(&[(0, flag), (8, y)], 16, body.clone())),
            writer,
            launch(&[(0, flag), (8, x)], 16, body.clone()),
            guarded(launch(&[(0, x), (8, y)], 16, body)),
            // The guard runs per wave: a 4-wave writer's later waves could
            // read the flag after its first wave cleared it.
            guarded(launch(&[(0, flag)], 8, step.clone())),
        ];
        let mut p = author("gfx1151", &launches);
        let f = p.nodes[0].guard.unwrap();
        assert_eq!(p.nodes[1].guard, Some(f));
        assert_eq!(p.nodes[2].guard, None, "no certificate");
        assert_ne!(p.nodes[3].guard, Some(f), "certified on another resource");
        let c = |body: std::ops::Range<usize>, lowering| Conditional {
            flag: f,
            body,
            lowering,
        };
        for refused in [2..3, 3..4, 1..3, 0..0, 3..6, 4..5] {
            assert!(
                p.add_conditional(c(refused.clone(), CondLowering::KernelGuard))
                    .is_err(),
                "{refused:?}"
            );
        }
        // The flag writer may end the body, not open it.
        let mut swapped = author("gfx1151", &[launches[1].clone(), launches[0].clone()]);
        assert!(swapped
            .add_conditional(c(0..2, CondLowering::KernelGuard))
            .is_err());
        p.add_conditional(c(0..2, CondLowering::KernelGuard))
            .unwrap();
        assert!(
            p.add_conditional(c(1..3, CondLowering::CpSkip)).is_err(),
            "overlap"
        );
        p.add_conditional(c(2..4, CondLowering::CpSkip)).unwrap();
        assert_eq!(p.conditionals.len(), 2);
        // A refused copy leaves no earlier copy behind.
        let mut w = author(
            "gfx1151",
            &[
                launches[0].clone(),
                launches[1].clone(),
                launches[0].clone(),
                launches[4].clone(),
            ],
        );
        assert!(w
            .bounded_while(f, 0, 2, 2, CondLowering::KernelGuard)
            .is_err());
        assert!(w.conditionals.is_empty(), "{:?}", w.conditionals);

        let mut gfx12 = author("gfx1201", &launches);
        assert!(gfx12
            .add_conditional(c(2..4, CondLowering::CpSkip))
            .is_err());
    }
}
