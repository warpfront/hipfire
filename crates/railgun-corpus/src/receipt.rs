// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `peacemaker-receipt` schema (receipt v1 = radiowave `PeacemakerRecord`
//! schema 6; the `railgun` section makes it 7; the conditional indexed-argument
//! amendment makes it 8 / section 2; the sealed-table amendment (pointer tables
//! loaded from a loader-sealed immutable region) makes it 9 / section 3).
//! `railgun-cert` writes these
//! records offline; the runtime reads them through this crate, which depends
//! on no peacemaker crate (railgun design D12).
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA: &str = "peacemaker-receipt";
/// Receipt v1 (architecture.md §3) is schema 6; the `railgun` section is 7;
/// guarded/indexed facts (design A §A9.2) are 8; sealed-table facts (the
/// `sealed_table` witness) are 9. A v7/v8 receipt is never reinterpreted:
/// unknown versions fail admission.
pub const RECEIPT_SCHEMA_VERSION: u32 = 9;
pub const RAILGUN_SECTION_VERSION: u32 = 3;

/// Witness kind of a conditional indexed-argument case (design A §A9.2).
pub const WITNESS_INDEXED_BIND: &str = "indexed_bind";
/// Witness kind of a sealed-table case: a loader-sealed immutable pointer
/// table whose pointees are immutable (`railgun::sealed_table`).
pub const WITNESS_SEALED_TABLE: &str = "sealed_table";
/// Witness kind of a descriptor-arena case: the program binds the arena's
/// records from its own resources (`railgun::KernelProgram::bind_records`).
pub const WITNESS_DESCRIPTOR_ARENA: &str = "descriptor_arena";
/// Every witness kind a receipt may name.
pub const KNOWN_WITNESSES: &[&str] = &[
    WITNESS_INDEXED_BIND,
    WITNESS_SEALED_TABLE,
    WITNESS_DESCRIPTOR_ARENA,
];

/// `SealedTableRecord.bound_kind`: the guard bound is the 4-byte kernarg
/// scalar at byte offset `bound_value` (unsigned compare).
pub const SEALED_BOUND_KERNARG: &str = "kernarg_u32";
/// The guard bound is the constant `bound_value` (`<= u32::MAX`).
pub const SEALED_BOUND_CONST: &str = "const_u32";

/// `KernelRecord.status` values.
pub const STATUS_PROVEN: &str = "proven";
pub const STATUS_UNKNOWN: &str = "unknown";
/// Conditional: valid only for the guarded case, and only once every
/// `required_witnesses` kind has a matching runtime bind witness.
pub const STATUS_CONDITIONAL: &str = "conditional";

/// One launch/argument guard of a guarded case (§A9.1). `kind` is one of
/// `kernarg_eq` (kernarg bytes `[arg_offset, arg_offset+size)` == `value`,
/// decimal; a null pointer is `size 8 value 0`), `grid_dim` / `block_dim`
/// (`arg_offset` = axis 0/1/2, `size` 0, `value` = work-group count / size).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct GuardRecord {
    pub kind: String,
    pub arg_offset: u32,
    pub size: u32,
    pub value: String,
}

/// `IndexLoad`: an immutable `I32` index loaded from a pointer argument.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct IndexLoadRecord {
    pub pc: u32,
    pub source_arg_offset: u32,
    /// `i32`.
    pub element: String,
    /// Byte-offset expression, e.g. `4*workgroup_id_y`.
    pub offset_expr: String,
    /// Extension/arithmetic semantics of the consuming address
    /// (`sext32`/`zext32`/...).
    pub extension: String,
    pub class: String,
    pub path: String,
    pub cache: String,
    pub width: u32,
}

/// `IndexedArgEffect`: one access whose address is `direct base + expr`.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct IndexedEffectRecord {
    pub pc: u32,
    pub base_arg_offset: u32,
    /// `read`, `write`, `read_write`.
    pub mode: String,
    pub class: String,
    pub path: String,
    /// Raw cache bits, `th=.. scope=.. glc=.. slc=.. dlc=.. nv=..`.
    pub cache: String,
    /// Access width in bytes.
    pub width: u32,
    /// Checked 64-bit address expression, canonical text.
    pub address_expr: String,
    pub index_loads: Vec<IndexLoadRecord>,
    /// Guard `kind@arg_offset` names this effect depends on.
    pub guards: Vec<String>,
}

/// One entry load of a sealed table (design: `SealedTable` certificate): a
/// 64-bit load of `table[index]` whose index is, on every path from the kernel
/// entry, strictly below the table's bound (unsigned).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct SealedEntryRecord {
    /// Layout position of the entry load.
    pub pc: u32,
    /// `smem_load` or `vmem_load`.
    pub class: String,
    pub path: String,
    /// Raw cache bits, `th=.. scope=.. glc=.. slc=.. dlc=.. nv=..`.
    pub cache: String,
    /// Bytes loaded (the table's entry width).
    pub width: u32,
    /// Canonical text of the table index (`load@<pc>`, `karg32:<offset>`).
    pub index_expr: String,
    /// Layout positions of the unsigned compares of the index against the
    /// bound that establish `index < bound` on the paths reaching the load.
    pub guard_pcs: Vec<u32>,
}

/// One access whose base derives from sealed-table entries and that only
/// reads (`vmem_load`/`smem_load`).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct SealedAccessRecord {
    pub pc: u32,
    pub class: String,
    pub path: String,
    pub cache: String,
    pub width: u32,
    /// `pc`s of the table's entry loads this access's base may derive from.
    pub entry_pcs: Vec<u32>,
}

/// One sealed table of a kernel: the pointer argument that holds the table, the
/// bound its index is checked against, and what the kernel does with the
/// loaded entries. Valid only with a matching `sealed_table` runtime witness:
/// the argument resolves to a loader-sealed immutable region holding at least
/// `bound` entries, every entry points into an immutable allocation, and
/// nothing in the program writes either.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct SealedTableRecord {
    /// Kernarg byte offset of the (pointer) argument holding the table base.
    pub table_arg_offset: u32,
    /// [`SEALED_BOUND_KERNARG`] or [`SEALED_BOUND_CONST`].
    pub bound_kind: String,
    /// Kernarg byte offset of the 4-byte bound, or the constant.
    pub bound_value: u64,
    /// Bytes per table entry (a pointer: 8).
    pub entry_bytes: u32,
    pub entries: Vec<SealedEntryRecord>,
    pub accesses: Vec<SealedAccessRecord>,
    /// Every read through an entry is reachable only with that entry nonzero
    /// (proved on every path): the sealed table may then hold null entries.
    /// Absent (false) for a table whose entries must all be non-null.
    #[serde(default, skip_serializing_if = "is_false")]
    pub null_guarded: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl SealedTableRecord {
    /// Fail-closed structure check (the digest binds the content; this rejects
    /// shapes no certifier emits).
    pub fn validate(&self, symbol: &str) -> Result<(), String> {
        let bad = |what: &str| {
            Err(format!(
                "{symbol}: sealed table @{}: {what}",
                self.table_arg_offset
            ))
        };
        match self.bound_kind.as_str() {
            SEALED_BOUND_KERNARG => {
                if self.bound_value % 4 != 0 || self.bound_value > u64::from(u32::MAX) {
                    return bad("kernarg bound offset is not a 4-aligned u32");
                }
            }
            SEALED_BOUND_CONST => {
                if self.bound_value > u64::from(u32::MAX) {
                    return bad("constant bound exceeds u32");
                }
            }
            _ => return bad("unknown bound kind"),
        }
        if self.entry_bytes != 8 {
            return bad("entry width is not 8 bytes");
        }
        if self.entries.is_empty() {
            return bad("no entry load");
        }
        for e in &self.entries {
            if e.width != self.entry_bytes
                || !matches!(e.class.as_str(), "smem_load" | "vmem_load")
                || e.guard_pcs.is_empty()
                || e.index_expr.is_empty()
            {
                return bad("malformed entry load");
            }
        }
        let entry_pcs: Vec<u32> = self.entries.iter().map(|e| e.pc).collect();
        for a in &self.accesses {
            if !matches!(a.class.as_str(), "smem_load" | "vmem_load")
                || a.entry_pcs.is_empty()
                || !a.entry_pcs.iter().all(|p| entry_pcs.contains(p))
            {
                return bad("an access is not a read derived from this table's entries");
            }
        }
        Ok(())
    }
}

/// One pointer a descriptor arena's template record carries: the kernel loads
/// 8 bytes at `record_offset` from a record of the template and dereferences
/// the loaded value with `classes` (A1 memory class names).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArenaPointerRecord {
    /// Record-relative byte offset of the pointer (8-aligned).
    pub record_offset: u32,
    pub classes: Vec<String>,
}

/// The write footprint of one pointer field `record_offset` of a template:
/// writes through that pointer by two workgroups `w != w'` of the template's
/// standalone grid are disjoint whenever `w` and `w'` differ on some axis in
/// `axes` (`"x"`, `"y"`, `"z"`; sorted, no duplicates, non-empty), under the
/// assumptions below (each skipped when empty, so plain records are unchanged).
#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, PartialEq)]
pub struct WgFootprintRecord {
    pub record_offset: u32,
    pub axes: Vec<String>,
    /// Work-items per tile on x the proof assumes (the region block); 0: none.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub block_x: u32,
    /// Per-axis exclusive tile-id bounds the proof needs (32-bit address math
    /// wraps beyond): the host checks the tile grid extents against them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bounds: Vec<WgBoundRecord>,
    /// Record dword relations the proof used.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<WgRequirementRecord>,
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

/// Workgroup ids on `axis` are below `bound` (exclusive).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct WgBoundRecord {
    pub axis: String,
    pub bound: u64,
}

/// The record dword at `offset`, masked with `mask`, equals `value`.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct WgRequirementRecord {
    pub offset: u32,
    pub mask: u32,
    pub value: u32,
}

/// One record template of a descriptor arena: which pointer fields of a record
/// of this template the kernel loads (and nothing else of the record is a
/// pointer).
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArenaTemplateRecord {
    pub template: u32,
    pub pointer_loads: Vec<ArenaPointerRecord>,
    /// The sealed pointer tables the template's body reads through (the
    /// standalone body's `sealed_tables`, whose kernarg offsets are record
    /// offsets: inside the pool region the record is the body's kernarg
    /// block). `table_arg_offset` names a record pointer field holding the
    /// table base; a [`SEALED_BOUND_KERNARG`] bound is the record offset of
    /// the 4-byte entry-count scalar. Valid only once the program binds
    /// every record of the template with a sealed table the runtime holds
    /// (`railgun::KernelProgram::bind_records` with a
    /// `railgun::sealed_table::SealedTableRegistry`). Absent from a template
    /// that has none, so such receipts digest and serialize as before.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sealed_tables: Vec<SealedTableRecord>,
    /// Certified per-workgroup write footprints of pointer fields of this
    /// template (see [`WgFootprintRecord`]). Absent from a template without
    /// them, so such receipts digest and serialize as before.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wg_footprints: Vec<WgFootprintRecord>,
}

/// A descriptor arena of a kernel: the pointer argument that holds an array
/// of binding records, and per record template the pointer fields the kernel
/// loads from a record and what it does through each. Valid only once the
/// program binds the arena's records (`railgun::KernelProgram::bind_records`):
/// every pointer the kernel can load is then a pointer railgun wrote from the
/// program's own bindings, so each access lands in a named resource.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct DescriptorArenaRecord {
    /// Kernarg byte offset of the (pointer) argument holding the arena base.
    pub arena_arg_offset: u32,
    pub templates: Vec<ArenaTemplateRecord>,
}

impl DescriptorArenaRecord {
    /// Fail-closed structure check (the digest binds the content).
    pub fn validate(&self, symbol: &str) -> Result<(), String> {
        let bad = |what: &str| {
            Err(format!(
                "{symbol}: descriptor arena @{}: {what}",
                self.arena_arg_offset
            ))
        };
        if self.templates.is_empty() {
            return bad("no template");
        }
        for (i, t) in self.templates.iter().enumerate() {
            if self.templates[..i].iter().any(|o| o.template == t.template) {
                return bad("duplicate template");
            }
            for (j, p) in t.pointer_loads.iter().enumerate() {
                if p.record_offset % 8 != 0 {
                    return bad("pointer offset is not 8-aligned");
                }
                if t.pointer_loads[..j]
                    .iter()
                    .any(|o| o.record_offset == p.record_offset)
                {
                    return bad("duplicate pointer offset in a template");
                }
            }
            for (j, f) in t.wg_footprints.iter().enumerate() {
                if !t
                    .pointer_loads
                    .iter()
                    .any(|p| p.record_offset == f.record_offset)
                {
                    return bad("a workgroup footprint names a field that is not a pointer load");
                }
                if t.wg_footprints[..j]
                    .iter()
                    .any(|o| o.record_offset == f.record_offset)
                {
                    return bad("duplicate workgroup footprint for a field");
                }
                if f.axes.is_empty()
                    || !f.axes.windows(2).all(|w| w[0] < w[1])
                    || f.axes
                        .iter()
                        .any(|a| !matches!(a.as_str(), "x" | "y" | "z"))
                {
                    return bad("workgroup footprint axes are empty, unsorted or not in x/y/z");
                }
                for (k, b) in f.bounds.iter().enumerate() {
                    if !matches!(b.axis.as_str(), "x" | "y" | "z")
                        || b.bound < 1
                        || f.bounds[..k].iter().any(|o| o.axis == b.axis)
                    {
                        return bad("a workgroup bound has a bad axis or a zero or repeated bound");
                    }
                }
                if f.requires.iter().any(|r| r.offset % 4 != 0) {
                    return bad("a workgroup requirement offset is not 4-aligned");
                }
            }
            for (j, s) in t.sealed_tables.iter().enumerate() {
                s.validate(symbol)?;
                if s.table_arg_offset % 8 != 0 {
                    return bad("sealed table offset is not 8-aligned");
                }
                if t.sealed_tables[..j].iter().any(|o| {
                    o.table_arg_offset == s.table_arg_offset
                        && o.bound_kind == s.bound_kind
                        && o.bound_value == s.bound_value
                }) {
                    return bad("duplicate sealed table in a template");
                }
            }
        }
        Ok(())
    }
}

/// A persistent pool kernel's dispatcher protocol property (see
/// [`TaskProtocolRecord::proved`]): one claim is a single `fetch_add(1)` on
/// the ticket, and the task header is read from `tasks[claim * desc_stride]`.
pub const TASK_PROTO_CLAIM: &str = "claim_single_fetch_add";
/// The claim is compared against the `n_tasks` kernarg: a claim at or past it
/// runs no body, and every claim below it runs exactly one task.
pub const TASK_PROTO_BOUNDED: &str = "claim_bounded_by_n_tasks";
/// The body is reachable only after an acquire load of `events[pred]` has
/// observed a value `>= expected`, when `pred != none_event`.
pub const TASK_PROTO_ACQUIRE: &str = "acquire_before_body";
/// The only release increment of `events[done]` follows the body on every
/// path, and happens only when `done != none_event`.
pub const TASK_PROTO_RELEASE: &str = "body_before_release";
/// Every memory access of the body is complete (memory counters waited,
/// workgroup barrier passed) before that release.
pub const TASK_PROTO_COMPLETE: &str = "body_complete_before_release";
/// A claimed task increments `events[done]` exactly once (never zero times
/// or twice), by `+1`.
pub const TASK_PROTO_ONCE: &str = "one_release_per_task";
/// The dispatcher runs a task's region at virtual workgroup
/// `(vx + step, vyz)` for every `step < max(vcount, 1)`, where `vx, vy, vz,
/// vcount` are read from the task header at [`TaskGridOffsets`] (so the task's
/// virtual workgroup set is determined by those header fields alone). Optional:
/// proving it is what lets a host treat one launch's tile claims as the
/// intra-grid concurrency of a single launch.
pub const TASK_PROTO_REGION_WG: &str = "region_wg_from_header";
/// Every property a consumer may find in [`TaskProtocolRecord::proved`].
pub const TASK_PROTO_KNOWN: [&str; 7] = [
    TASK_PROTO_ACQUIRE,
    TASK_PROTO_BOUNDED,
    TASK_PROTO_CLAIM,
    TASK_PROTO_COMPLETE,
    TASK_PROTO_ONCE,
    TASK_PROTO_RELEASE,
    TASK_PROTO_REGION_WG,
];
/// Every property a task-graph proof needs, sorted.
pub const TASK_PROTO_REQUIRED: [&str; 6] = [
    TASK_PROTO_ACQUIRE,
    TASK_PROTO_RELEASE,
    TASK_PROTO_COMPLETE,
    TASK_PROTO_BOUNDED,
    TASK_PROTO_CLAIM,
    TASK_PROTO_ONCE,
];

/// What `railgun-cert`'s task-graph certificate proved about a persistent
/// pool kernel's dispatcher, on the emitted object (`railgun::task_graph`
/// consumes it: its proof of a co-resident group's ordering and progress is
/// only as good as this record). Present only on a `conditional` record, and
/// bound by the case digest (`railgun-case-v4`). A kernel without a proven
/// protocol carries none.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct TaskProtocolRecord {
    /// Kernarg byte offsets of the dispatcher's pointer arguments: the task
    /// headers, the claim ticket (one `u32` cell), the event cells; and of
    /// the `u32` task count.
    pub tasks_arg_offset: u32,
    pub ticket_arg_offset: u32,
    pub events_arg_offset: u32,
    pub n_tasks_arg_offset: u32,
    /// Bytes per task header, and the byte offsets inside one of the `u32`
    /// fields the dispatcher reads.
    pub desc_stride: u32,
    pub template_offset: u32,
    pub ka_off_offset: u32,
    pub pred_event_offset: u32,
    pub expected_offset: u32,
    pub done_event_offset: u32,
    /// The `pred`/`done` value meaning "no event".
    pub none_event: u32,
    /// Bytes between event cells: event id `i` is at `events + i * stride`.
    pub event_stride: u32,
    /// The properties the certificate proved on the object (sorted, no
    /// duplicates; names are the `TASK_PROTO_*` constants). A consumer
    /// requires every one of [`TASK_PROTO_REQUIRED`].
    pub proved: Vec<String>,
    /// Header offsets of the virtual-grid fields the dispatcher derives the
    /// region's workgroup from (needed by [`TASK_PROTO_REGION_WG`]). Absent
    /// (and skipped when serializing) on receipts that do not prove it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<TaskGridOffsets>,
}

/// Byte offsets, inside one task header, of the `u32` virtual-grid fields.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct TaskGridOffsets {
    pub vx_offset: u32,
    pub vy_offset: u32,
    pub vz_offset: u32,
    pub gx_offset: u32,
    pub gy_offset: u32,
    pub gz_offset: u32,
    pub vcount_offset: u32,
}

impl TaskProtocolRecord {
    /// Fail-closed structure check (the digest binds the content).
    pub fn validate(&self, symbol: &str) -> Result<(), String> {
        let bad = |what: &str| Err(format!("{symbol}: task protocol: {what}"));
        if self.desc_stride == 0 || self.desc_stride % 4 != 0 {
            return bad("descriptor stride is not a positive multiple of 4");
        }
        let fields = [
            self.template_offset,
            self.ka_off_offset,
            self.pred_event_offset,
            self.expected_offset,
            self.done_event_offset,
        ];
        for (i, &at) in fields.iter().enumerate() {
            if at % 4 != 0 || at.checked_add(4).is_none_or(|end| end > self.desc_stride) {
                return bad("a header field is misaligned or outside the descriptor");
            }
            if fields[..i].contains(&at) {
                return bad("two header fields share an offset");
            }
        }
        let args = [
            self.tasks_arg_offset,
            self.ticket_arg_offset,
            self.events_arg_offset,
            self.n_tasks_arg_offset,
        ];
        if args.iter().enumerate().any(|(i, a)| args[..i].contains(a)) {
            return bad("two arguments share an offset");
        }
        if self.tasks_arg_offset % 8 != 0
            || self.ticket_arg_offset % 8 != 0
            || self.events_arg_offset % 8 != 0
        {
            return bad("a pointer argument is not 8-aligned");
        }
        if self.event_stride == 0 {
            return bad("event stride is zero");
        }
        if !self.proved.windows(2).all(|w| w[0] < w[1]) {
            return bad("proved properties are not sorted and unique");
        }
        if let Some(p) = self
            .proved
            .iter()
            .find(|p| !TASK_PROTO_KNOWN.contains(&p.as_str()))
        {
            return bad(&format!("unknown property {p:?}"));
        }
        if let Some(g) = &self.grid {
            let all = [
                g.vx_offset,
                g.vy_offset,
                g.vz_offset,
                g.gx_offset,
                g.gy_offset,
                g.gz_offset,
                g.vcount_offset,
            ];
            for (i, &at) in all.iter().enumerate() {
                if at % 4 != 0 || at.checked_add(4).is_none_or(|end| end > self.desc_stride) {
                    return bad("a grid field is misaligned or outside the descriptor");
                }
                if all[..i].contains(&at) || fields.contains(&at) {
                    return bad("a grid field shares an offset with another header field");
                }
            }
        }
        if self.proved.iter().any(|p| p == TASK_PROTO_REGION_WG) && self.grid.is_none() {
            return bad("region_wg_from_header is proved without the grid field offsets");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub crate_version: String,
    pub git_sha: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelIdentity {
    pub symbol: String,
    /// `[entry_va, size]` of the kernel's `.text` range.
    pub text_range: [u64; 2],
    pub text_sha256: String,
    pub kd_sha256: String,
    pub metadata_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Check {
    pub id: String,
    pub status: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub schema: String,
    pub schema_version: u32,
    /// SHA-256 of the input file (HIP bundle or ELF): the lookup key.
    pub object_sha256: String,
    pub device_elf_sha256: String,
    pub arch: String,
    /// Tier D (design D13): diagnostic lift + byte identity, never strict.
    pub strict: bool,
    pub tier: String,
    pub lifter: Tool,
    pub certifier: Tool,
    pub round_trip_proof_sha256: String,
    pub kernels: Vec<KernelIdentity>,
    pub checks: Vec<Check>,
    pub obligations: Vec<String>,
    pub railgun: RailgunSection,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RailgunSection {
    pub section_version: u32,
    pub analysis: String,
    pub assumptions: Vec<String>,
    pub kernels: Vec<KernelRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnresolvedRecord {
    pub pc: Option<u32>,
    pub rule: String,
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SideRecord {
    pub kernarg_segment: u32,
    pub module_global: u32,
    pub abi_packet: u32,
    pub local_flat: u32,
    pub scratch: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImplicitRecord {
    pub hidden_loaded: Vec<String>,
    pub hidden_grid: Vec<String>,
    pub hidden_grid_live: Vec<String>,
    /// The kernel loads a `hidden_block_count_{x,y,z}` dword (the words
    /// `replay.rs` writes from the recorded grid and never repatches).
    pub loads_block_count: bool,
    pub loads_group_size: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArgRecord {
    pub offset: u32,
    pub size: u32,
    pub index: Option<usize>,
    pub name: String,
    pub value_kind: String,
    /// A4: `pointer` (a resolved access base), `integer` (loaded, never a
    /// base) or `not_loaded`.
    pub role: String,
    /// A1: `read`, `write`, `read_write`, or absent when never accessed.
    pub mode: Option<String>,
    pub bound: Option<String>,
    pub classes: Vec<String>,
    pub paths: Vec<String>,
    pub cache_bits: Vec<String>,
    pub accesses: usize,
    /// Some access resolved to more than one argument (e.g. `s_cselect_b64`).
    pub shared: bool,
    /// A3: read caches that can hold data this kernel read from the argument.
    pub read_cache: String,
    /// A4 sinks over the argument's dwords.
    pub sinks: Vec<String>,
    pub loaded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DwordRecord {
    pub offset: u32,
    pub loaded: bool,
    pub sinks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct ArgCheck {
    pub offset: u32,
    pub name: String,
    pub a1: Option<String>,
    /// radiowave-conservative access (`read_only`/`write_only`/`read_write`).
    pub metadata: String,
    /// `.actual_access` was present (not the conservative default).
    pub explicit: bool,
    pub status: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Differential {
    /// `sidecar` (runtime `*.radiowave.json`), `object-metadata` (the code
    /// object's own note, through the same radiowave rule) or `none`.
    pub source: String,
    pub status: String,
    pub args: Vec<ArgCheck>,
    /// Radiowave's text-derived read-cache class (sidecar only), next to A3.
    pub radiowave_mutable_read_cache: Option<String>,
    /// The guarded case this differential was computed under (empty for an
    /// unconditional summary). A passing differential with guards is a check
    /// of the guarded summary only, never an unconditional proof.
    pub guards: Vec<GuardRecord>,
    /// The summary compared was the guarded (Conditional) one.
    pub guarded: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelRecord {
    pub symbol: String,
    pub status: String,
    pub unresolved: Vec<UnresolvedRecord>,
    pub kernarg_size: u32,
    pub read_cache: String,
    pub dynamic_kernarg_reads: bool,
    pub taint_overflow: bool,
    pub side: SideRecord,
    pub implicit: ImplicitRecord,
    pub args: Vec<ArgRecord>,
    pub dwords: Vec<DwordRecord>,
    pub differential: Differential,
    /// Open peacemaker analysis obligations by rule id (diagnostic tier).
    pub pm_obligations: BTreeMap<String, usize>,
    /// `edit::analyze` failed; facts come from the kernargs pass alone.
    pub pm_analyze_error: Option<String>,
    /// Diagnostic only, conditional records: the unguarded analysis' refusals
    /// (what the guarded case replaced). Empty otherwise; never a fact source.
    pub unguarded_unresolved: Vec<UnresolvedRecord>,
    /// Guards of the guarded case; non-empty iff `status == "conditional"`.
    pub guards: Vec<GuardRecord>,
    /// Indexed argument effects of the guarded case.
    pub indexed_effects: Vec<IndexedEffectRecord>,
    /// Sealed pointer tables of the case (`sealed_table` witness). Absent from
    /// kernels without one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sealed_tables: Vec<SealedTableRecord>,
    /// Descriptor arenas of the case (`descriptor_arena` witness: discharged
    /// when the program binds the arena's records). Absent from kernels
    /// without one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub descriptor_arenas: Vec<DescriptorArenaRecord>,
    /// Witness kinds the runtime must hold before the executable may use the
    /// effects (`indexed_bind` for `indexed_effects`, `sealed_table` for
    /// `sealed_tables`); exactly the kinds those fields imply, sorted;
    /// non-empty iff conditional.
    pub required_witnesses: Vec<String>,
    /// SHA-256 binding object, kernel, status, guards, effects and witnesses
    /// ([`case_digest`]); empty unless conditional.
    pub case_digest: String,
    /// #867 kernel guard: the kernarg offset of the ControlWord pointer whose
    /// clear flag (`u32` zero) ends every wave before any other instruction
    /// that touches memory (`railgun_cert::guard_flag`). Absent from receipts
    /// written before the field and from unguarded kernels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_flag: Option<u32>,
    /// The dispatcher protocol of a persistent pool kernel that the
    /// task-graph certificate proved on the emitted object (see
    /// [`TaskProtocolRecord`]). Absent from every other kernel, so receipts
    /// written before the field are unchanged (and digest as before).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_protocol: Option<TaskProtocolRecord>,
    /// Workgroup write footprints of the kernel's pointer arguments, proven from
    /// the object's bytes (`peacemaker_ir::passes::wg_footprint`), for the tile
    /// geometry `block_x`; `record_offset` is the kernarg offset of the pointer
    /// argument. Absent when nothing is proven, so other receipts are unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wg_footprints: Vec<WgFootprintRecord>,
}

/// The guarded-case digest under the exact object receipt (§A9.2; v2 binds
/// `sealed_tables` as well). A record with `descriptor_arenas` digests under
/// `railgun-case-v3`, which binds them too, and one with a `task_protocol`
/// under `railgun-case-v4`, which binds that as well; a record without any
/// keeps the v2 input byte for byte, so every receipt written before arenas
/// existed digests identically.
pub fn case_digest(
    object_sha256: &str,
    device_elf_sha256: &str,
    arch: &str,
    k: &KernelRecord,
) -> String {
    let bytes = if let Some(protocol) = &k.task_protocol {
        serde_json::to_vec(&(
            "railgun-case-v4",
            object_sha256,
            device_elf_sha256,
            arch,
            &k.symbol,
            &k.status,
            k.kernarg_size,
            &k.guards,
            &k.indexed_effects,
            &k.sealed_tables,
            &k.descriptor_arenas,
            protocol,
            &k.required_witnesses,
        ))
    } else if k.descriptor_arenas.is_empty() {
        serde_json::to_vec(&(
            "railgun-case-v2",
            object_sha256,
            device_elf_sha256,
            arch,
            &k.symbol,
            &k.status,
            k.kernarg_size,
            &k.guards,
            &k.indexed_effects,
            &k.sealed_tables,
            &k.required_witnesses,
        ))
    } else {
        serde_json::to_vec(&(
            "railgun-case-v3",
            object_sha256,
            device_elf_sha256,
            arch,
            &k.symbol,
            &k.status,
            k.kernarg_size,
            &k.guards,
            &k.indexed_effects,
            &k.sealed_tables,
            &k.descriptor_arenas,
            &k.required_witnesses,
        ))
    }
    .expect("case digest input serialises");
    crate::sha256_hex(&bytes)
}

impl KernelRecord {
    /// The witness kinds this record's case fields imply, sorted.
    pub fn implied_witnesses(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.indexed_effects.is_empty() {
            out.push(WITNESS_INDEXED_BIND.to_owned());
        }
        if !self.sealed_tables.is_empty() {
            out.push(WITNESS_SEALED_TABLE.to_owned());
        }
        let arena_sealed = self
            .descriptor_arenas
            .iter()
            .any(|a| a.templates.iter().any(|t| !t.sealed_tables.is_empty()));
        if arena_sealed && self.sealed_tables.is_empty() {
            out.push(WITNESS_SEALED_TABLE.to_owned());
        }
        if !self.descriptor_arenas.is_empty() {
            out.push(WITNESS_DESCRIPTOR_ARENA.to_owned());
        }
        out.sort();
        out
    }

    /// Fail-closed structural check of the guarded fields against `status`.
    /// A conditional record needs its effects (indexed effects with guards,
    /// and/or sealed tables), exactly the witnesses they imply and a matching
    /// digest; a proven/unknown record must carry none of them (a v1 row is
    /// never promoted); any other status is rejected.
    pub fn validate_case(
        &self,
        object_sha256: &str,
        device_elf_sha256: &str,
        arch: &str,
    ) -> Result<(), String> {
        let empty = self.guards.is_empty()
            && self.indexed_effects.is_empty()
            && self.sealed_tables.is_empty()
            && self.descriptor_arenas.is_empty()
            && self.task_protocol.is_none()
            && self.required_witnesses.is_empty()
            && self.case_digest.is_empty();
        match self.status.as_str() {
            STATUS_PROVEN | STATUS_UNKNOWN => {
                if !empty {
                    return Err(format!(
                        "{}: status {} carries guarded-case fields",
                        self.symbol, self.status
                    ));
                }
                Ok(())
            }
            STATUS_CONDITIONAL => {
                let indexed = !self.indexed_effects.is_empty();
                if (!indexed && self.sealed_tables.is_empty() && self.descriptor_arenas.is_empty())
                    || (indexed && self.guards.is_empty())
                    || (!indexed && !self.guards.is_empty())
                    || self.required_witnesses.is_empty()
                {
                    return Err(format!(
                        "{}: conditional record lacks guards, effects or required witnesses",
                        self.symbol
                    ));
                }
                if self.required_witnesses != self.implied_witnesses() {
                    return Err(format!(
                        "{}: required witnesses {:?} are not the kinds the case implies {:?}",
                        self.symbol,
                        self.required_witnesses,
                        self.implied_witnesses()
                    ));
                }
                for (i, t) in self.sealed_tables.iter().enumerate() {
                    t.validate(&self.symbol)?;
                    if self.sealed_tables[..i].iter().any(|o| {
                        o.table_arg_offset == t.table_arg_offset
                            && o.bound_kind == t.bound_kind
                            && o.bound_value == t.bound_value
                    }) {
                        return Err(format!("{}: duplicate sealed table", self.symbol));
                    }
                }
                for (i, a) in self.descriptor_arenas.iter().enumerate() {
                    a.validate(&self.symbol)?;
                    if self.descriptor_arenas[..i]
                        .iter()
                        .any(|o| o.arena_arg_offset == a.arena_arg_offset)
                    {
                        return Err(format!("{}: duplicate descriptor arena", self.symbol));
                    }
                }
                if let Some(protocol) = &self.task_protocol {
                    protocol.validate(&self.symbol)?;
                    if self.descriptor_arenas.is_empty() {
                        return Err(format!(
                            "{}: a task protocol needs the descriptor arena its tasks' records live in",
                            self.symbol
                        ));
                    }
                }
                if !self.differential.guarded || self.differential.guards != self.guards {
                    return Err(format!(
                        "{}: differential does not report the record's guards",
                        self.symbol
                    ));
                }
                if self.case_digest != case_digest(object_sha256, device_elf_sha256, arch, self) {
                    return Err(format!("{}: case digest does not match", self.symbol));
                }
                Ok(())
            }
            other => Err(format!("{}: unknown A1 status {other:?}", self.symbol)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> KernelRecord {
        KernelRecord {
            symbol: "k".into(),
            status: STATUS_PROVEN.into(),
            unresolved: vec![],
            kernarg_size: 8,
            read_cache: "not_read".into(),
            dynamic_kernarg_reads: false,
            taint_overflow: false,
            side: SideRecord::default(),
            implicit: ImplicitRecord::default(),
            args: vec![],
            dwords: vec![],
            differential: Differential {
                source: "none".into(),
                status: "pass".into(),
                args: vec![],
                radiowave_mutable_read_cache: None,
                guards: vec![],
                guarded: false,
            },
            pm_obligations: BTreeMap::new(),
            pm_analyze_error: None,
            unguarded_unresolved: vec![],
            guards: vec![],
            indexed_effects: vec![],
            sealed_tables: vec![],
            descriptor_arenas: vec![],
            required_witnesses: vec![],
            case_digest: String::new(),
            guard_flag: None,
            task_protocol: None,
            wg_footprints: Vec::new(),
        }
    }

    fn arena() -> DescriptorArenaRecord {
        DescriptorArenaRecord {
            arena_arg_offset: 8,
            templates: vec![ArenaTemplateRecord {
                template: 3,
                pointer_loads: vec![ArenaPointerRecord {
                    record_offset: 16,
                    classes: vec!["vmem_load".into()],
                }],
                sealed_tables: vec![],
                wg_footprints: vec![],
            }],
        }
    }

    #[test]
    fn wg_footprints_validate_and_are_skipped_when_empty() {
        let mut a = arena();
        assert!(serde_json::to_value(&a.templates[0])
            .unwrap()
            .get("wg_footprints")
            .is_none());
        a.templates[0].wg_footprints = vec![WgFootprintRecord {
            record_offset: 16,
            axes: vec!["x".into(), "y".into()],
            block_x: 128,
            bounds: vec![WgBoundRecord {
                axis: "x".into(),
                bound: 1 << 24,
            }],
            requires: vec![WgRequirementRecord {
                offset: 28,
                mask: 0xffff,
                value: 128,
            }],
        }];
        a.validate("k").unwrap();
        let mut b = a.clone();
        b.templates[0].wg_footprints[0].record_offset = 24;
        assert!(b.validate("k").is_err());
        let mut b = a.clone();
        b.templates[0].wg_footprints[0].axes = vec!["y".into(), "x".into()];
        assert!(b.validate("k").is_err());
        let mut b = a.clone();
        b.templates[0].wg_footprints[0].axes.clear();
        assert!(b.validate("k").is_err());
    }

    fn sealed_table() -> SealedTableRecord {
        SealedTableRecord {
            table_arg_offset: 0,
            bound_kind: SEALED_BOUND_KERNARG.into(),
            bound_value: 24,
            entry_bytes: 8,
            entries: vec![SealedEntryRecord {
                pc: 0x40,
                class: "smem_load".into(),
                path: "scalar".into(),
                cache: "th=0 scope=0".into(),
                width: 8,
                index_expr: "load@0x20".into(),
                guard_pcs: vec![0x30],
            }],
            accesses: vec![SealedAccessRecord {
                pc: 0x50,
                class: "vmem_load".into(),
                path: "vector".into(),
                cache: "th=0 scope=0".into(),
                width: 16,
                entry_pcs: vec![0x40],
            }],
            null_guarded: false,
        }
    }

    #[test]
    fn arena_template_sealed_tables_are_optional_validated_and_digest_bound() {
        // Absent: the serialized template is byte-identical to before the field.
        let plain = arena();
        let json = serde_json::to_value(&plain).unwrap();
        assert!(json["templates"][0].get("sealed_tables").is_none());
        let back: DescriptorArenaRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back, plain);
        let mut k = record();
        k.status = STATUS_CONDITIONAL.into();
        k.descriptor_arenas = vec![plain];
        k.required_witnesses = k.implied_witnesses();
        assert_eq!(k.required_witnesses, vec![WITNESS_DESCRIPTOR_ARENA]);
        k.differential.guarded = true;
        k.case_digest = case_digest("obj", "elf", "gfx1201", &k);
        k.validate_case("obj", "elf", "gfx1201").unwrap();

        // Present: round-trips, implies the sealed_table witness, binds the digest.
        let mut sealed = k.clone();
        sealed.descriptor_arenas[0].templates[0].sealed_tables = vec![sealed_table()];
        let json = serde_json::to_value(&sealed).unwrap();
        assert!(json["descriptor_arenas"][0]["templates"][0]
            .get("sealed_tables")
            .is_some());
        let back: KernelRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back.descriptor_arenas, sealed.descriptor_arenas);
        assert_eq!(
            sealed.implied_witnesses(),
            vec![WITNESS_DESCRIPTOR_ARENA, WITNESS_SEALED_TABLE]
        );
        // The old witness set and digest no longer validate.
        assert!(sealed.validate_case("obj", "elf", "gfx1201").is_err());
        sealed.required_witnesses = sealed.implied_witnesses();
        sealed.case_digest = case_digest("obj", "elf", "gfx1201", &sealed);
        assert_ne!(sealed.case_digest, k.case_digest);
        sealed.validate_case("obj", "elf", "gfx1201").unwrap();

        // Fail-closed shapes.
        let mut bad = sealed.clone();
        bad.descriptor_arenas[0].templates[0].sealed_tables[0].entry_bytes = 4;
        assert!(bad.descriptor_arenas[0].validate("k").is_err());
        let mut dup = sealed.descriptor_arenas[0].clone();
        let t = dup.templates[0].sealed_tables[0].clone();
        dup.templates[0].sealed_tables.push(t);
        assert!(dup.validate("k").is_err());
        let mut unaligned = sealed.descriptor_arenas[0].clone();
        unaligned.templates[0].sealed_tables[0].table_arg_offset = 4;
        assert!(unaligned.validate("k").is_err());
    }

    #[test]
    fn record_without_arenas_serializes_and_digests_as_before() {
        let k = record();
        let json = serde_json::to_value(&k).unwrap();
        assert!(
            json.get("descriptor_arenas").is_none(),
            "the new optional record is absent from a record that has none"
        );
        // A receipt written before the field existed parses to an empty list.
        let back: KernelRecord = serde_json::from_value(json).unwrap();
        assert!(back.descriptor_arenas.is_empty());
        // The pre-arena digest input, spelled out: unchanged for such a record.
        let old = crate::sha256_hex(
            &serde_json::to_vec(&(
                "railgun-case-v2",
                "obj",
                "elf",
                "gfx1201",
                &k.symbol,
                &k.status,
                k.kernarg_size,
                &k.guards,
                &k.indexed_effects,
                &k.sealed_tables,
                &k.required_witnesses,
            ))
            .unwrap(),
        );
        assert_eq!(case_digest("obj", "elf", "gfx1201", &k), old);
        assert!(k.implied_witnesses().is_empty());
    }

    #[test]
    fn arena_record_round_trips_binds_the_digest_and_implies_its_witness() {
        let mut k = record();
        k.status = STATUS_CONDITIONAL.into();
        k.descriptor_arenas = vec![arena()];
        k.required_witnesses = k.implied_witnesses();
        assert_eq!(k.required_witnesses, vec![WITNESS_DESCRIPTOR_ARENA]);
        k.differential.guarded = true;
        k.case_digest = case_digest("obj", "elf", "gfx1201", &k);
        k.validate_case("obj", "elf", "gfx1201").unwrap();

        let json = serde_json::to_value(&k).unwrap();
        assert!(json.get("descriptor_arenas").is_some());
        let back: KernelRecord = serde_json::from_value(json).unwrap();
        assert_eq!(back.descriptor_arenas, k.descriptor_arenas);

        let mut changed = k.clone();
        changed.descriptor_arenas[0].templates[0].pointer_loads[0].classes =
            vec!["vmem_store".into()];
        assert!(
            changed.validate_case("obj", "elf", "gfx1201").is_err(),
            "the digest binds the arena facts"
        );
        let mut stale = k.clone();
        stale.required_witnesses.clear();
        assert!(stale.validate_case("obj", "elf", "gfx1201").is_err());
        let mut proven = record();
        proven.descriptor_arenas = vec![arena()];
        assert!(
            proven.validate_case("obj", "elf", "gfx1201").is_err(),
            "a proven record may not carry an arena"
        );
    }

    #[test]
    fn malformed_arena_is_rejected() {
        let mut a = arena();
        a.templates[0].pointer_loads[0].record_offset = 12;
        assert!(a.validate("k").is_err());
        let mut a = arena();
        let t = a.templates[0].clone();
        a.templates.push(t);
        assert!(a.validate("k").is_err());
        let mut a = arena();
        let load = a.templates[0].pointer_loads[0].clone();
        a.templates[0].pointer_loads.push(load);
        assert!(a.validate("k").is_err());
        assert!(DescriptorArenaRecord {
            arena_arg_offset: 0,
            templates: vec![]
        }
        .validate("k")
        .is_err());
    }

    fn protocol() -> TaskProtocolRecord {
        TaskProtocolRecord {
            tasks_arg_offset: 0,
            ticket_arg_offset: 8,
            events_arg_offset: 16,
            n_tasks_arg_offset: 56,
            desc_stride: 128,
            template_offset: 0,
            ka_off_offset: 28,
            pred_event_offset: 32,
            expected_offset: 36,
            done_event_offset: 40,
            none_event: u32::MAX,
            event_stride: 4,
            proved: TASK_PROTO_REQUIRED
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            grid: None,
        }
    }

    #[test]
    fn region_wg_property_needs_grid_offsets_and_is_skipped_when_absent() {
        let p = protocol();
        assert!(serde_json::to_value(&p).unwrap().get("grid").is_none());
        let mut q = p.clone();
        q.proved.push(TASK_PROTO_REGION_WG.into());
        assert!(q.validate("k").is_err(), "property without offsets");
        q.grid = Some(TaskGridOffsets {
            vx_offset: 4,
            vy_offset: 8,
            vz_offset: 12,
            gx_offset: 16,
            gy_offset: 20,
            gz_offset: 24,
            vcount_offset: 44,
        });
        q.validate("k").unwrap();
        let back: TaskProtocolRecord =
            serde_json::from_value(serde_json::to_value(&q).unwrap()).unwrap();
        assert_eq!(back, q);
        q.grid.as_mut().unwrap().vx_offset = 28;
        assert!(q.validate("k").is_err(), "grid field overlaps ka_off");
    }

    #[test]
    fn task_protocol_is_optional_binds_the_digest_and_needs_an_arena() {
        let k = record();
        assert!(serde_json::to_value(&k)
            .unwrap()
            .get("task_protocol")
            .is_none());

        let mut k = record();
        k.status = STATUS_CONDITIONAL.into();
        k.descriptor_arenas = vec![arena()];
        k.task_protocol = Some(protocol());
        k.required_witnesses = k.implied_witnesses();
        k.differential.guarded = true;
        let v3 = {
            let mut no_protocol = k.clone();
            no_protocol.task_protocol = None;
            case_digest("obj", "elf", "gfx1201", &no_protocol)
        };
        k.case_digest = case_digest("obj", "elf", "gfx1201", &k);
        assert_ne!(k.case_digest, v3, "the digest binds the protocol");
        k.validate_case("obj", "elf", "gfx1201").unwrap();
        let back: KernelRecord = serde_json::from_value(serde_json::to_value(&k).unwrap()).unwrap();
        assert_eq!(back.task_protocol, k.task_protocol);

        let mut changed = k.clone();
        changed.task_protocol.as_mut().unwrap().proved.pop();
        assert!(changed.validate_case("obj", "elf", "gfx1201").is_err());
        let mut no_arena = k.clone();
        no_arena.descriptor_arenas.clear();
        no_arena.required_witnesses.clear();
        assert!(no_arena.validate_case("obj", "elf", "gfx1201").is_err());
        let mut proven = record();
        proven.task_protocol = Some(protocol());
        assert!(
            proven.validate_case("obj", "elf", "gfx1201").is_err(),
            "a proven record may not carry a task protocol"
        );
    }

    #[test]
    fn malformed_task_protocol_is_rejected() {
        assert!(protocol().validate("k").is_ok());
        let mut p = protocol();
        p.proved.push("zzz".into());
        assert!(p.validate("k").is_err(), "unknown property");
        let mut p = protocol();
        p.proved.reverse();
        assert!(p.validate("k").is_err(), "unsorted");
        let mut p = protocol();
        p.done_event_offset = p.expected_offset;
        assert!(p.validate("k").is_err(), "shared field offset");
        let mut p = protocol();
        p.ka_off_offset = 126;
        assert!(p.validate("k").is_err(), "field outside the descriptor");
        let mut p = protocol();
        p.events_arg_offset = p.tasks_arg_offset;
        assert!(p.validate("k").is_err(), "shared argument offset");
        let mut p = protocol();
        p.event_stride = 0;
        assert!(p.validate("k").is_err());
    }
}
