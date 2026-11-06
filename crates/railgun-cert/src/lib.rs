// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun-cert: the offline railgun certifier (railgun design §1.5 delivery,
//! §2.5, gate G8).
//!
//! For one code object it lifts every kernel with `peacemaker-lift` (byte-exact
//! round trip), runs the peacemaker analysis (A1/A3/A4 are
//! `Facts::kernargs`), and serialises the facts into the `railgun` section of a
//! receipt (receipt v1 = radiowave `PeacemakerRecord` schema 6, architecture.md
//! §3; the `railgun` section bumps it to 7). The runtime never links this
//! crate: records reach it as `receipts/<arch>/<object_sha256>.receipt.json`.
//!
//! Differential (G8, non-authoritative): radiowave's `argument_effects`
//! (AMDGPU `.actual_access`, conservative `ReadWrite` where absent) is the
//! second source. The per-argument rule is A1 ⊆ metadata-conservative: an
//! argument A1 writes that the metadata calls read-only, or reads that it
//! calls write-only, fails; and an argument the metadata explicitly says is
//! written that a `Proven` A1 summary never writes fails (A1 would be missing
//! a write).
//!
//! [`whole16`] builds and verifies the DFlash Whole16 window manifest (the
//! complete node sequence with exact-object identities, the FP8 attention
//! explicit-ABI table, refusals for non-railgun or fact-less nodes).
pub mod recording;
pub mod task_graph;
pub mod whole16;

use std::collections::BTreeMap;

use peacemaker_ir::edit::analyze;
use peacemaker_ir::inst::{Abi, Arch, Frontend, Kernel, KernelOrigin, Program};
use peacemaker_ir::kernarg::{
    AccessClass, AccessMode, ArgFacts, ArgRole, Guard, IndexExt, IndexLoad, IndexedArgEffect,
    KernargFacts, KernelStatus, MemPath, ReadCacheClass, RegionBound, SealedAccess, SealedBound,
    SealedEntryLoad, SealedTableUse,
};
use peacemaker_ir::passes::kernargs;
use peacemaker_lift::kd::DescriptorCodec;
use peacemaker_lift::{lift_object, Options};
use radiowave::{
    CodeObjectCertification, CodeObjectInspection, CompileManifest, KernelArgumentAccess,
    KernelArgumentReport, KernelReport, MutableReadCache, Wavefront,
};
use sha2::{Digest, Sha256};

pub use railgun_corpus::receipt::{
    case_digest, ArgCheck, ArgRecord, Check, Differential, DwordRecord, GuardRecord,
    ImplicitRecord, IndexLoadRecord, IndexedEffectRecord, KernelIdentity, KernelRecord,
    RailgunSection, Receipt, SealedAccessRecord, SealedEntryRecord, SealedTableRecord, SideRecord,
    Tool, UnresolvedRecord, WgBoundRecord, WgFootprintRecord, WgRequirementRecord, KNOWN_WITNESSES,
    RAILGUN_SECTION_VERSION, RECEIPT_SCHEMA, RECEIPT_SCHEMA_VERSION, SEALED_BOUND_CONST,
    SEALED_BOUND_KERNARG, STATUS_CONDITIONAL, STATUS_PROVEN, STATUS_UNKNOWN, WITNESS_INDEXED_BIND,
    WITNESS_SEALED_TABLE,
};

/// Stated in every receipt: what a `Proven` A1 verdict rests on.
pub const ASSUMPTIONS: &[&str] = &[
    "AddressArithmetic: a pointer is never rebuilt by multiplication, shifting or bit-field extraction of a value loaded from memory",
    "AllocationWide: every recognised pointer effect covers its whole allocation (A2 bounds are M0b)",
    "SealedTable: a value loaded from a sealed table is only used as a whole 64-bit pointer base of reads that stay inside the pointee allocation the entry names (halves are never recombined across entries, an entry is never added to another pointer, and a read whose base may be an entry or an argument pointer reads whichever it names; a table whose certificate records `null_guarded` may hold null entries, every other table holds none; the loader keeps the table and every pointee immutable)",
];

#[derive(Debug, thiserror::Error)]
pub enum CertError {
    #[error("lift: {0}")]
    Lift(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn arch_name(arch: Arch) -> &'static str {
    match arch {
        Arch::Gfx1010 => "gfx1010",
        Arch::Gfx1030 => "gfx1030",
        Arch::Gfx1100 => "gfx1100",
        Arch::Gfx1151 => "gfx1151",
        Arch::Gfx1201 => "gfx1201",
    }
}

fn mode_name(mode: AccessMode) -> &'static str {
    match mode {
        AccessMode::Read => "read",
        AccessMode::Write => "write",
        AccessMode::ReadWrite => "read_write",
    }
}
fn access_name(access: KernelArgumentAccess) -> &'static str {
    match access {
        KernelArgumentAccess::ReadOnly => "read_only",
        KernelArgumentAccess::WriteOnly => "write_only",
        KernelArgumentAccess::ReadWrite => "read_write",
    }
}
pub fn read_cache_name(class: ReadCacheClass) -> &'static str {
    match class {
        ReadCacheClass::NotRead => "not_read",
        ReadCacheClass::VmemOnly => "vmem_only",
        ReadCacheClass::ScalarOnly => "scalar_only",
        ReadCacheClass::ScalarAndVmem => "scalar_and_vmem",
        ReadCacheClass::Unknown => "unknown",
    }
}
fn class_name(class: AccessClass) -> String {
    match class {
        AccessClass::VmemLoad => "vmem_load".into(),
        AccessClass::VmemStore => "vmem_store".into(),
        AccessClass::SmemLoad => "smem_load".into(),
        AccessClass::Atomic { commutative } => {
            format!("atomic{}", if commutative { "_commutative" } else { "" })
        }
        AccessClass::LdsDma => "lds_dma".into(),
        AccessClass::AtomicRelease => "atomic_release".into(),
        AccessClass::LoadAcquire => "load_acquire".into(),
    }
}
fn path_name(path: MemPath) -> &'static str {
    match path {
        MemPath::Global => "global",
        MemPath::Buffer => "buffer",
        MemPath::Flat => "flat",
        MemPath::Smem => "smem",
        MemPath::SmemBuffer => "smem_buffer",
    }
}

/// Metadata effects of one kernel: `(offset, conservative access, explicit)`
/// for each `global_buffer` argument, from radiowave's `argument_effects`.
#[derive(Clone, Debug, Default)]
pub struct MetadataEffects {
    pub source: String,
    pub effects: Option<Vec<(u32, KernelArgumentAccess, bool)>>,
    pub mutable_read_cache: Option<MutableReadCache>,
}

/// The A1 ⊆ metadata-conservative differential for one kernel.
pub fn cross_check(facts: &KernargFacts, meta: &MetadataEffects) -> Differential {
    let guarded = facts.status == KernelStatus::Conditional;
    let mut out = Differential {
        source: meta.source.clone(),
        status: "not_applicable".into(),
        args: Vec::new(),
        radiowave_mutable_read_cache: meta.mutable_read_cache.map(|c| match c {
            MutableReadCache::VmemOnly => "vmem_only".into(),
            MutableReadCache::ScalarOrUnknown => "scalar_or_unknown".into(),
        }),
        guards: if guarded {
            guard_records(&facts.guards)
        } else {
            Vec::new()
        },
        guarded,
    };
    let Some(effects) = &meta.effects else {
        return out;
    };
    // A Conditional summary is complete *under its guards* (indexed effects
    // are credited to their arguments), so the G8 comparison applies to it in
    // full; the result is reported with `guards`, not as an unconditional proof.
    let proven = matches!(
        facts.status,
        KernelStatus::Proven | KernelStatus::Conditional
    );
    let mut failed = false;
    for &(offset, access, explicit) in effects {
        let arg = facts.arg_at(offset);
        let a1 = arg.and_then(|a| a.mode);
        let name = arg.map(|a| a.name.clone()).unwrap_or_default();
        let (status, reason) = match (a1, access) {
            (Some(m), KernelArgumentAccess::ReadOnly) if m.writes() => (
                "fail",
                "A1 writes an argument the metadata says is read_only".to_owned(),
            ),
            (Some(m), KernelArgumentAccess::WriteOnly) if m.reads() => (
                "fail",
                "A1 reads an argument the metadata says is write_only".to_owned(),
            ),
            (m, KernelArgumentAccess::WriteOnly | KernelArgumentAccess::ReadWrite)
                if explicit && !m.is_some_and(AccessMode::writes) =>
            {
                if proven {
                    (
                        "fail",
                        format!(
                            "metadata says {} but the Proven A1 summary never writes it",
                            access_name(access)
                        ),
                    )
                } else {
                    (
                        "not_applicable",
                        "A1 is Unknown: a missing write may sit in an unresolved access".to_owned(),
                    )
                }
            }
            _ => ("pass", String::new()),
        };
        failed |= status == "fail";
        out.args.push(ArgCheck {
            offset,
            name,
            a1: a1.map(|m| mode_name(m).to_owned()),
            metadata: access_name(access).into(),
            explicit,
            status: status.into(),
            reason,
        });
    }
    out.status = if failed { "fail" } else { "pass" }.into();
    out
}

fn arg_record(a: &ArgFacts) -> ArgRecord {
    let mut classes: Vec<String> = a.classes().into_iter().map(class_name).collect();
    classes.dedup();
    let mut paths: Vec<String> = a
        .accesses
        .iter()
        .map(|x| path_name(x.path).to_owned())
        .collect();
    paths.sort();
    paths.dedup();
    let mut cache: Vec<String> = a
        .accesses
        .iter()
        .map(|x| {
            let c = x.cache;
            format!(
                "th={} scope={} glc={} slc={} dlc={} nv={}",
                c.th,
                c.scope,
                u8::from(c.glc),
                u8::from(c.slc),
                u8::from(c.dlc),
                u8::from(c.nv)
            )
        })
        .collect();
    cache.sort();
    cache.dedup();
    ArgRecord {
        offset: a.offset,
        size: a.size,
        index: a.index,
        name: a.name.clone(),
        value_kind: a.value_kind.clone(),
        role: match a.role {
            ArgRole::Pointer => "pointer",
            ArgRole::Integer => "integer",
            ArgRole::NotLoaded => "not_loaded",
        }
        .into(),
        mode: a.mode.map(|m| mode_name(m).to_owned()),
        bound: a
            .bound
            .map(|RegionBound::AllocationWide| "allocation_wide".to_owned()),
        classes,
        paths,
        cache_bits: cache,
        accesses: a.accesses.len(),
        shared: a.accesses.iter().any(|x| x.shared),
        read_cache: read_cache_name(a.read_cache).into(),
        sinks: a.sinks.names().into_iter().map(str::to_owned).collect(),
        loaded: a.loaded,
    }
}

/// The workgroup write footprints (`peacemaker_ir::passes::wg_footprint`) of
/// `kernel`'s pointer arguments as receipt records (`record_offset` = the
/// argument's kernarg offset), for tiles of `block_x` work-items on x (`None`:
/// the kernel's declared maximum flat workgroup size, else 1024). `facts` is the
/// A1 result the record carries: a `Conditional` one is admitted only as the
/// sealed-table case. Empty when nothing is proven (any other arch, geometry or
/// A1 status included); an argument with no proof has no record.
pub fn wg_footprint_records(
    kernel: &Kernel,
    arch: Arch,
    facts: &KernargFacts,
    block_x: Option<u32>,
) -> Vec<WgFootprintRecord> {
    use peacemaker_ir::passes::wg_footprint::{analyze, ArgWgClass, WgLaunch, HW_BOUNDS};
    let max_flat = match &kernel.abi {
        peacemaker_ir::inst::Abi::Hsa { metadata, .. } => metadata.parsed.max_flat_workgroup_size,
        _ => 0,
    };
    let block = block_x.unwrap_or(if max_flat == 0 { 1024 } else { max_flat });
    let sealed = facts.status == KernelStatus::Conditional;
    let result = analyze(kernel, arch, sealed, WgLaunch { block_x: block });
    let axis_name = |a: u8| ["x", "y", "z"][usize::from(a)].to_owned();
    result
        .args
        .iter()
        .filter_map(|a| match &a.class {
            ArgWgClass::Disjoint(d) => Some(WgFootprintRecord {
                record_offset: a.arg_offset,
                axes: d.axes.iter().map(|&x| axis_name(x)).collect(),
                block_x: d.block_x,
                bounds: d
                    .axes
                    .iter()
                    .filter(|&&x| d.bounds[usize::from(x)] < HW_BOUNDS[usize::from(x)])
                    .map(|&x| WgBoundRecord {
                        axis: axis_name(x),
                        bound: d.bounds[usize::from(x)],
                    })
                    .collect(),
                requires: d
                    .requires
                    .iter()
                    .map(|r| WgRequirementRecord {
                        offset: r.offset,
                        mask: r.mask,
                        value: r.value,
                    })
                    .collect(),
            }),
            _ => None,
        })
        .collect()
}

/// Serialise one kernel's facts (plus its differential) for the receipt.
pub fn kernel_record(
    kernel: &Kernel,
    facts: &KernargFacts,
    meta: &MetadataEffects,
) -> KernelRecord {
    let pc_of = |id| kernel.body.insts.get(id).and_then(|i| i.prov.pc);
    let loaded = |kind: &str| {
        facts
            .implicit
            .hidden_loaded
            .iter()
            .any(|k| k.starts_with(kind))
    };
    let conditional = facts.status == KernelStatus::Conditional;
    KernelRecord {
        symbol: kernel.symbol.0.clone(),
        status: match facts.status {
            KernelStatus::Proven => STATUS_PROVEN,
            KernelStatus::Conditional => STATUS_CONDITIONAL,
            KernelStatus::Unknown => STATUS_UNKNOWN,
        }
        .into(),
        unresolved: facts
            .unresolved
            .iter()
            .map(|u| UnresolvedRecord {
                pc: u.inst.and_then(pc_of),
                rule: u.rule.into(),
                detail: u.detail.clone(),
            })
            .collect(),
        kernarg_size: facts.kernarg_size,
        read_cache: read_cache_name(facts.read_cache).into(),
        dynamic_kernarg_reads: facts.dynamic_kernarg_reads,
        taint_overflow: facts.taint_overflow,
        side: SideRecord {
            kernarg_segment: facts.side.kernarg_segment,
            module_global: facts.side.module_global,
            abi_packet: facts.side.abi_packet,
            local_flat: facts.side.local_flat,
            scratch: facts.side.scratch,
        },
        implicit: ImplicitRecord {
            hidden_loaded: facts.implicit.hidden_loaded.clone(),
            hidden_grid: facts.implicit.hidden_grid.clone(),
            hidden_grid_live: facts.implicit.hidden_grid_live.clone(),
            loads_block_count: loaded("hidden_block_count_"),
            loads_group_size: loaded("hidden_group_size_"),
        },
        args: facts.args.iter().map(arg_record).collect(),
        dwords: facts
            .dwords
            .iter()
            .map(|d| DwordRecord {
                offset: d.offset,
                loaded: d.loaded,
                sinks: d.sinks.names().into_iter().map(str::to_owned).collect(),
            })
            .collect(),
        differential: cross_check(facts, meta),
        pm_obligations: BTreeMap::new(),
        pm_analyze_error: None,
        unguarded_unresolved: Vec::new(),
        guards: if conditional {
            guard_records(&facts.guards)
        } else {
            Vec::new()
        },
        indexed_effects: if conditional {
            facts.indexed.iter().map(indexed_record).collect()
        } else {
            Vec::new()
        },
        sealed_tables: if conditional {
            facts.sealed.iter().map(sealed_record).collect()
        } else {
            Vec::new()
        },
        descriptor_arenas: Vec::new(),
        required_witnesses: {
            let mut w = Vec::new();
            if conditional && !facts.indexed.is_empty() {
                w.push(WITNESS_INDEXED_BIND.to_owned());
            }
            if conditional && !facts.sealed.is_empty() {
                w.push(WITNESS_SEALED_TABLE.to_owned());
            }
            w.sort();
            w
        },
        case_digest: String::new(),
        guard_flag: None,
        task_protocol: None,
        wg_footprints: Vec::new(),
    }
}

/// Canonical kernel record of one sealed-table entry load.
fn sealed_entry_record(e: &SealedEntryLoad) -> SealedEntryRecord {
    SealedEntryRecord {
        pc: e.pc,
        class: class_name(e.class),
        path: path_name(e.path).into(),
        cache: cache_text(e.cache),
        width: u32::from(e.width_bytes),
        index_expr: e.index.to_string(),
        guard_pcs: e.guard_pcs.clone(),
    }
}
fn sealed_access_record(a: &SealedAccess) -> SealedAccessRecord {
    SealedAccessRecord {
        pc: a.pc,
        class: class_name(a.class),
        path: path_name(a.path).into(),
        cache: cache_text(a.cache),
        width: u32::from(a.width_bytes),
        entry_pcs: a.entry_pcs.clone(),
    }
}
fn sealed_record(t: &SealedTableUse) -> SealedTableRecord {
    let (bound_kind, bound_value) = match t.bound {
        SealedBound::Kernarg { offset } => (SEALED_BOUND_KERNARG, u64::from(offset)),
        SealedBound::Const(c) => (SEALED_BOUND_CONST, u64::from(c)),
    };
    SealedTableRecord {
        table_arg_offset: t.table_arg_offset,
        bound_kind: bound_kind.into(),
        bound_value,
        entry_bytes: u32::from(t.entry_bytes),
        entries: t.entries.iter().map(sealed_entry_record).collect(),
        accesses: t.accesses.iter().map(sealed_access_record).collect(),
        null_guarded: t.null_guarded,
    }
}

/// The sealed-table case of `kernel` (design: `SealedTable` certificate) when
/// its unguarded analysis is `Unknown` and the sealed analysis resolves every
/// access (`Conditional`, with the `sealed_table` witness required). When the
/// sealed analysis finds a table-shaped entry load but cannot prove it, the
/// unguarded refusals are replaced by its exact `sealed-*` rules (still
/// `Unknown`); a kernel with no table-shaped load keeps the unguarded result.
fn sealed_facts(kernel: &Kernel, arch: Arch, unguarded: &mut KernargFacts) -> Option<KernargFacts> {
    if unguarded.status != KernelStatus::Unknown {
        return None;
    }
    let sealed = kernargs::analyze_sealed(kernel, arch);
    if sealed.status == KernelStatus::Conditional && !sealed.sealed.is_empty() {
        return Some(sealed);
    }
    if sealed
        .unresolved
        .iter()
        .any(|u| u.rule.starts_with("sealed-"))
    {
        unguarded.unresolved = sealed.unresolved;
    }
    None
}

fn guard_record(g: &Guard) -> GuardRecord {
    match *g {
        Guard::KernargEq {
            offset,
            size,
            value,
        } => GuardRecord {
            kind: "kernarg_eq".into(),
            arg_offset: offset,
            size,
            value: value.to_string(),
        },
        Guard::GridDim { axis, value } => GuardRecord {
            kind: "grid_dim".into(),
            arg_offset: u32::from(axis),
            size: 0,
            value: value.to_string(),
        },
        Guard::BlockDim { axis, value } => GuardRecord {
            kind: "block_dim".into(),
            arg_offset: u32::from(axis),
            size: 0,
            value: value.to_string(),
        },
    }
}
fn guard_records(guards: &[Guard]) -> Vec<GuardRecord> {
    let mut out: Vec<GuardRecord> = guards.iter().map(guard_record).collect();
    out.sort_by(|a, b| {
        (&a.kind, a.arg_offset, a.size, &a.value).cmp(&(&b.kind, b.arg_offset, b.size, &b.value))
    });
    out.dedup();
    out
}
fn guard_name(g: &Guard) -> String {
    let r = guard_record(g);
    format!("{}@{}:{}={}", r.kind, r.arg_offset, r.size, r.value)
}
fn cache_text(c: peacemaker_ir::kernarg::CacheBits) -> String {
    format!(
        "th={} scope={} glc={} slc={} dlc={} nv={}",
        c.th,
        c.scope,
        u8::from(c.glc),
        u8::from(c.slc),
        u8::from(c.dlc),
        u8::from(c.nv)
    )
}
fn ext_name(e: IndexExt) -> &'static str {
    match e {
        IndexExt::Sext64 => "sext64",
        IndexExt::Zext64 => "zext64",
        IndexExt::Raw32 => "raw32",
    }
}
fn index_load_record(l: &IndexLoad) -> IndexLoadRecord {
    IndexLoadRecord {
        pc: l.pc,
        source_arg_offset: l.src_arg_offset,
        element: match l.elem {
            peacemaker_ir::kernarg::IndexElem::I32 => "i32",
        }
        .into(),
        offset_expr: l.byte_offset.to_string(),
        extension: l
            .ext
            .iter()
            .map(|e| ext_name(*e))
            .collect::<Vec<_>>()
            .join(","),
        class: class_name(l.class),
        path: path_name(l.path).into(),
        cache: cache_text(l.cache),
        width: 4,
    }
}
fn indexed_record(e: &IndexedArgEffect) -> IndexedEffectRecord {
    IndexedEffectRecord {
        pc: e.pc,
        base_arg_offset: e.base_arg_offset,
        mode: mode_name(e.mode).into(),
        class: class_name(e.class),
        path: path_name(e.path).into(),
        cache: cache_text(e.cache),
        width: u32::from(e.width_bytes),
        address_expr: format!("(add arg:{} {})", e.base_arg_offset, e.displacement),
        index_loads: e.index_loads.iter().map(index_load_record).collect(),
        guards: e.guards.iter().map(guard_name).collect(),
    }
}

/// A guarded case the certifier tries on an object whose unguarded analysis is
/// `Unknown` (design A §A9.1). The table only *selects* which guards to
/// substitute; it grants nothing: the case is attempted only when the
/// object's own metadata has exactly these explicit arguments, and the result
/// is Conditional only if the guarded analysis of the lifted ISA says so.
struct GuardedCase {
    symbol: &'static str,
    /// `(offset, size, value_kind)` of every explicit argument, asserted
    /// against the loaded object's metadata.
    args: &'static [(u32, u32, &'static str)],
    guards: &'static [Guard],
}

/// `kv_cache_write_fp8_e4m3_batched`: dst +0, src +8, positions +16,
/// n_kv_heads +24, head_dim +28, batch_size +32, slot_descs +40, row_slot +48;
/// HD256, B16, both optional slot tables null, grid `[n_kv_heads,16,1]`,
/// block `[32,1,1]` (the x extent `n_kv_heads` is a bind-time check).
const GUARDED_CASES: &[GuardedCase] = &[GuardedCase {
    symbol: "kv_cache_write_fp8_e4m3_batched",
    args: &[
        (0, 8, "global_buffer"),
        (8, 8, "global_buffer"),
        (16, 8, "global_buffer"),
        (24, 4, "by_value"),
        (28, 4, "by_value"),
        (32, 4, "by_value"),
        (40, 8, "global_buffer"),
        (48, 8, "global_buffer"),
    ],
    guards: &[
        Guard::KernargEq {
            offset: 28,
            size: 4,
            value: 256,
        },
        Guard::KernargEq {
            offset: 32,
            size: 4,
            value: 16,
        },
        Guard::KernargEq {
            offset: 40,
            size: 8,
            value: 0,
        },
        Guard::KernargEq {
            offset: 48,
            size: 8,
            value: 0,
        },
        Guard::GridDim { axis: 1, value: 16 },
        Guard::GridDim { axis: 2, value: 1 },
        Guard::BlockDim { axis: 0, value: 32 },
        Guard::BlockDim { axis: 1, value: 1 },
        Guard::BlockDim { axis: 2, value: 1 },
    ],
}];

/// The guarded facts for `kernel`, when its unguarded analysis is Unknown, a
/// guarded case names it, the metadata ABI matches that case exactly, and the
/// guarded analysis of the original ISA returns Conditional with indexed
/// effects under every one of the case's guards. Otherwise `None` and the
/// unguarded (Unknown) result stands.
fn guarded_facts(kernel: &Kernel, arch: Arch, unguarded: &KernargFacts) -> Option<KernargFacts> {
    if unguarded.status != KernelStatus::Unknown {
        return None;
    }
    let case = GUARDED_CASES.iter().find(|c| c.symbol == kernel.symbol.0)?;
    let explicit: Vec<&ArgFacts> = unguarded.args.iter().filter(|a| !a.is_hidden()).collect();
    if explicit.len() != case.args.len()
        || !case.args.iter().all(|&(offset, size, kind)| {
            explicit
                .iter()
                .any(|a| a.offset == offset && a.size == size && a.value_kind == kind)
        })
    {
        return None;
    }
    let facts = kernargs::analyze_guarded(kernel, arch, case.guards);
    let mut want = case.guards.to_vec();
    want.sort();
    let mut have = facts.guards.clone();
    have.sort();
    (facts.status == KernelStatus::Conditional && !facts.indexed.is_empty() && have == want)
        .then_some(facts)
}

/// #867 kernel-guard certificate for one lifted kernel: the kernarg offset
/// of its ControlWord pointer, or `None` when the kernel does not open with
/// the certified guard prologue. The entry block must be exactly
///
/// ```text
/// s_load_b64 s[P:P+1], s[0:1], <flag offset>   ; s[0:1] = kernarg pointer
/// s_waitcnt lgkmcnt(0)
/// s_load_b32 sF, s[P:P+1], 0
/// s_waitcnt lgkmcnt(0)
/// s_cmp_eq_u32 sF, 0
/// s_cbranch_scc0 <body> | s_cbranch_scc1 <exit>
/// ```
///
/// (plus `s_delay_alu` hints), with the SCC-set successor a block holding
/// only `s_endpgm`. The block is scalar, so every wave of a launch takes the
/// same edge, and its only memory accesses are those two loads: a clear flag
/// ends the launch with no store, atomic or other load. gfx11 only (the
/// `s_waitcnt` encoding); gfx12 objects get no certificate.
pub fn guard_flag(kernel: &Kernel, arch: Arch) -> Option<u32> {
    use peacemaker_ir::cfg::{Cond, Terminator};
    let (Arch::Gfx1100 | Arch::Gfx1151) = arch else {
        return None;
    };
    let Abi::Hsa { descriptor, .. } = &kernel.abi else {
        return None;
    };
    // Only the kernarg-segment-pointer user SGPR is enabled ahead of the
    // others, so it is s[0:1].
    if descriptor.kernel_code_properties.0 & 0xf != 0x8 {
        return None;
    }
    let body = &kernel.body;
    let text = |block: usize| -> Option<Vec<String>> {
        let (start, end) = body.blocks.get(block)?.range;
        body.layout[start..end]
            .iter()
            .map(|&id| body.insts.get(id)?.text(arch).ok())
            .filter(|t| !t.as_deref().is_some_and(|t| t.starts_with("s_delay_alu")))
            .collect()
    };
    let entry = text(0)?;
    // gfx11 `s_waitcnt` simm16: lgkmcnt is bits 9:4 (gfx12 has `s_wait_kmcnt`).
    let lgkm_drained = |t: &str| {
        t.strip_prefix("s_waitcnt ")
            .and_then(|n| n.parse::<i16>().ok())
            .is_some_and(|n| (n as u16 >> 4) & 0x3f == 0)
    };
    let Terminator::Branch {
        cond,
        taken,
        fallthrough,
    } = body.blocks[0].term
    else {
        return None;
    };
    let exit = match cond {
        Cond::Scc0 => fallthrough,
        Cond::Scc1 => taken,
        _ => return None,
    };
    if text(exit.0)? != ["s_endpgm"] || body.blocks[exit.0].term != Terminator::EndPgm {
        return None;
    }
    let [load_ptr, wait0, load_flag, wait1, cmp, _branch] = &entry[..] else {
        return None;
    };
    let offset = |t: &str| match t {
        "null" | "0" | "0x0" => Some(0),
        t => match t.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => t.parse().ok(),
        },
    };
    let (ptr, flag_offset) = load_ptr
        .strip_prefix("s_load_b64 ")?
        .split_once(", s[0:1], ")?;
    let flag_offset = offset(flag_offset)?;
    let (flag, zero) = load_flag
        .strip_prefix("s_load_b32 ")?
        .split_once(&format!(", {ptr}, "))?;
    let ptr_ok = ptr
        .strip_prefix("s[")
        .and_then(|p| p.strip_suffix(']'))
        .and_then(|p| p.split_once(':'))
        .and_then(|(lo, hi)| Some((lo.parse::<u32>().ok()?, hi.parse::<u32>().ok()?)))
        .is_some_and(|(lo, hi)| hi == lo + 1);
    let flag_ok = flag
        .strip_prefix('s')
        .is_some_and(|n| n.parse::<u32>().is_ok());
    (ptr_ok
        && flag_ok
        && offset(zero) == Some(0)
        && lgkm_drained(wait0)
        && lgkm_drained(wait1)
        && *cmp == format!("s_cmp_eq_u32 {flag}, 0"))
    .then_some(flag_offset)
}

/// `.actual_access` per argument from the kernel's own metadata note.
fn metadata_arguments(raw: &[u8]) -> Vec<KernelArgumentReport> {
    let Ok(value) = rmpv::decode::read_value(&mut &raw[..]) else {
        return Vec::new();
    };
    let get = |map: &rmpv::Value, key: &str| -> Option<rmpv::Value> {
        map.as_map()?
            .iter()
            .find(|(k, _)| k.as_str() == Some(key))
            .map(|(_, v)| v.clone())
    };
    let Some(args) = get(&value, ".args") else {
        return Vec::new();
    };
    let Some(args) = args.as_array() else {
        return Vec::new();
    };
    args.iter()
        .map(|a| KernelArgumentReport {
            offset: get(a, ".offset")
                .and_then(|v| v.as_u64())
                .unwrap_or_default() as u32,
            size: get(a, ".size").and_then(|v| v.as_u64()).unwrap_or_default() as u32,
            value_kind: get(a, ".value_kind")
                .and_then(|v| v.as_str().map(str::to_owned))
                .unwrap_or_default(),
            address_space: get(a, ".address_space").and_then(|v| v.as_str().map(str::to_owned)),
            actual_access: get(a, ".actual_access").and_then(|v| match v.as_str() {
                Some("read_only") => Some(KernelArgumentAccess::ReadOnly),
                Some("write_only") => Some(KernelArgumentAccess::WriteOnly),
                Some("read_write") => Some(KernelArgumentAccess::ReadWrite),
                _ => None,
            }),
        })
        .collect()
}

/// A radiowave certification for `bytes`: the runtime sidecar when given,
/// else a schema-5 manifest built from the object's own metadata notes, so
/// radiowave's `argument_effects` rule is applied either way.
fn radiowave_cert(
    bytes: &[u8],
    program: &Program,
    sidecar: Option<&str>,
) -> (String, Option<CodeObjectCertification>) {
    if let Some(json) = sidecar {
        if let Ok(cert) = CodeObjectCertification::from_json(bytes, json) {
            return ("sidecar".into(), Some(cert));
        }
    }
    let kernels = program
        .kernels
        .iter()
        .filter_map(|k| match &k.abi {
            Abi::Hsa { metadata, .. } => Some(KernelReport {
                name: k.symbol.0.clone(),
                arguments: metadata_arguments(&metadata.raw_msgpack),
                ..KernelReport::default()
            }),
            Abi::Raw { .. } => None,
        })
        .collect();
    let manifest = CompileManifest {
        schema_version: 5,
        compiler: "radiowave".into(),
        generated_unix_seconds: 0,
        source: Default::default(),
        output: Default::default(),
        arch: arch_name(program.target.arch).into(),
        wavefront: Wavefront::Wave32,
        scheduler_profile: Default::default(),
        hipcc: Default::default(),
        hipcc_version: String::new(),
        command: Vec::new(),
        source_sha256: String::new(),
        support_header_sha256: String::new(),
        output_sha256: sha256_hex(bytes),
        inspection: Some(CodeObjectInspection {
            bundle_target: String::new(),
            identity: None,
            kernels,
        }),
        peacemaker: None,
    };
    let cert = serde_json::to_string(&manifest)
        .ok()
        .and_then(|json| CodeObjectCertification::from_json(bytes, &json).ok());
    ("object-metadata".into(), cert)
}

fn metadata_effects(
    source: &str,
    cert: Option<&CodeObjectCertification>,
    symbol: &str,
) -> MetadataEffects {
    let Some(cert) = cert else {
        return MetadataEffects {
            source: "none".into(),
            ..MetadataEffects::default()
        };
    };
    let explicit: BTreeMap<u32, bool> = cert
        .kernel(symbol)
        .map(|k| {
            k.arguments
                .iter()
                .map(|a| (a.offset, a.actual_access.is_some()))
                .collect()
        })
        .unwrap_or_default();
    let effects = cert.argument_effects(symbol).map(|v| {
        v.into_iter()
            .map(|(offset, access)| {
                let offset = offset as u32;
                (
                    offset,
                    access,
                    explicit.get(&offset).copied().unwrap_or(false),
                )
            })
            .collect()
    });
    MetadataEffects {
        source: source.into(),
        effects,
        mutable_read_cache: (source == "sidecar").then(|| cert.mutable_read_cache(symbol)),
    }
}

fn git_sha() -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    let run = |args: &[&str]| {
        std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let head = run(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = run(&[
        "status",
        "--porcelain",
        "--",
        "crates/peacemaker-ir",
        "crates/peacemaker-lift",
        "crates/railgun-cert",
    ])
    .is_some_and(|s| !s.is_empty());
    if dirty {
        format!("{head}-dirty")
    } else {
        head
    }
}

/// Facts for one kernel of a lifted program: `edit::analyze` (the `Analyzed`
/// revision) when it succeeds, else the kernargs pass alone.
pub fn kernel_facts(
    program: &Program,
    index: usize,
) -> (KernargFacts, BTreeMap<String, usize>, Option<String>) {
    let kernel = &program.kernels[index];
    match analyze(program.clone(), &kernel.symbol) {
        Ok(analyzed) => {
            let mut obligations = BTreeMap::new();
            for o in &analyzed.obligations {
                *obligations.entry(o.rule_id.clone()).or_insert(0) += 1;
            }
            (analyzed.facts.kernargs, obligations, None)
        }
        Err(e) => (
            kernargs::analyze(kernel, program.target.arch),
            BTreeMap::new(),
            Some(e.to_string()),
        ),
    }
}

/// A kernel's code identity (text stream, descriptor and metadata digests)
/// and the SHA-256 of the device ELF it came from.
fn kernel_identity(
    kernel: &Kernel,
    proof: &peacemaker_ir::state::RoundTripProof,
) -> (KernelIdentity, String) {
    let (kd_sha256, metadata_sha256) = match &kernel.abi {
        Abi::Hsa {
            descriptor,
            metadata,
        } => (
            sha256_hex(&descriptor.to_bytes()),
            sha256_hex(&metadata.raw_msgpack),
        ),
        Abi::Raw { .. } => (String::new(), String::new()),
    };
    let (text_range, device) = match &kernel.origin {
        KernelOrigin::Frontend {
            entry_va,
            size,
            object_sha256,
            ..
        } => ([*entry_va, *size], hex(object_sha256)),
        KernelOrigin::Authored { .. } => ([0, 0], String::new()),
    };
    let text_sha256 = proof
        .streams
        .iter()
        .find(|s| s.entry == text_range[0])
        .map(|s| hex(&s.stream_sha256))
        .unwrap_or_default();
    (
        KernelIdentity {
            symbol: kernel.symbol.0.clone(),
            text_range,
            text_sha256,
            kd_sha256,
            metadata_sha256,
        },
        device,
    )
}

/// The code identity of every kernel in a code object, without analysis:
/// two objects whose identities agree carry the same kernel code, descriptors
/// and metadata even when their bytes differ elsewhere.
pub fn code_identity(bytes: &[u8]) -> Result<Vec<KernelIdentity>, CertError> {
    let lifted = lift_object(
        bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .map_err(|e| CertError::Lift(e.to_string()))?;
    Ok(lifted
        .program
        .kernels
        .iter()
        .map(|k| kernel_identity(k, &lifted.proof).0)
        .collect())
}

/// Lift, analyse and serialise one code object. `sidecar` is the runtime's
/// `*.radiowave.json` for these exact bytes, when one exists.
pub fn certify(bytes: &[u8], sidecar: Option<&str>) -> Result<Receipt, CertError> {
    let lifted = lift_object(
        bytes,
        Options {
            frontend: Frontend::Hipcc,
        },
    )
    .map_err(|e| CertError::Lift(e.to_string()))?;
    let program = &lifted.program;
    let arch = program.target.arch;
    let (source, cert) = radiowave_cert(bytes, program, sidecar);
    let mut identities = Vec::new();
    let mut records = Vec::new();
    let mut obligations = std::collections::BTreeSet::new();
    let mut device_sha = String::new();
    for (index, kernel) in program.kernels.iter().enumerate() {
        let (identity, device) = kernel_identity(kernel, &lifted.proof);
        device_sha = device;
        identities.push(identity);
        let (mut facts, pm, pm_error) = kernel_facts(program, index);
        let meta = metadata_effects(&source, cert.as_ref(), &kernel.symbol.0);
        let guarded =
            guarded_facts(kernel, arch, &facts).or_else(|| sealed_facts(kernel, arch, &mut facts));
        let mut record = kernel_record(kernel, guarded.as_ref().unwrap_or(&facts), &meta);
        if guarded.is_some() {
            record.unguarded_unresolved = kernel_record(kernel, &facts, &meta).unresolved;
            record.case_digest =
                case_digest(&sha256_hex(bytes), &device_sha, arch_name(arch), &record);
        }
        record.pm_obligations = pm;
        record.pm_analyze_error = pm_error;
        let facts = guarded.unwrap_or(facts);
        record.guard_flag = guard_flag(kernel, arch);
        record.wg_footprints = wg_footprint_records(kernel, arch, &facts, None);
        if facts.args.iter().any(|a| a.bound.is_some()) {
            obligations.insert("InBoundsNoAlias".to_owned());
        }
        match facts.status {
            KernelStatus::Proven => {
                obligations.insert("AddressArithmetic".to_owned());
            }
            KernelStatus::Conditional => {
                if !facts.indexed.is_empty() {
                    obligations.insert("IndexedBindWitness".to_owned());
                }
                if !facts.sealed.is_empty() {
                    obligations.insert("SealedTableWitness".to_owned());
                }
            }
            KernelStatus::Unknown => {}
        }
        records.push(record);
    }
    let proven = records.iter().filter(|r| r.status == STATUS_PROVEN).count();
    let conditional = records
        .iter()
        .filter(|r| r.status == STATUS_CONDITIONAL)
        .count();
    let failed: Vec<&str> = records
        .iter()
        .filter(|r| r.differential.status == "fail")
        .map(|r| r.symbol.as_str())
        .collect();
    let checks = vec![
        Check {
            id: "roundtrip".into(),
            status: "pass".into(),
            detail: format!(
                "{} kernels re-encode byte-identically; module re-emits identically",
                program.kernels.len()
            ),
        },
        Check {
            id: "railgun-a1".into(),
            status: if proven == records.len() {
                "proven"
            } else if proven + conditional == records.len() {
                "conditional"
            } else {
                "unknown"
            }
            .into(),
            detail: format!(
                "{proven}/{} kernels Proven, {conditional} Conditional (witness required)",
                records.len()
            ),
        },
        Check {
            id: "g8-metadata-differential".into(),
            status: if failed.is_empty() { "pass" } else { "fail" }.into(),
            detail: if failed.is_empty() {
                format!("source {source}")
            } else {
                format!("source {source}; failing: {}", failed.join(", "))
            },
        },
    ];
    let proof = &lifted.proof;
    let mut proof_bytes = proof.input_sha256.to_vec();
    for s in &proof.streams {
        proof_bytes.extend(s.entry.to_le_bytes());
        proof_bytes.extend(s.size.to_le_bytes());
        proof_bytes.extend(s.stream_sha256);
    }
    let git = git_sha();
    Ok(Receipt {
        schema: RECEIPT_SCHEMA.into(),
        schema_version: RECEIPT_SCHEMA_VERSION,
        object_sha256: sha256_hex(bytes),
        device_elf_sha256: device_sha,
        arch: arch_name(arch).into(),
        strict: false,
        tier: "D".into(),
        lifter: Tool {
            name: "peacemaker-lift".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            git_sha: git.clone(),
        },
        certifier: Tool {
            name: "railgun-cert".into(),
            crate_version: env!("CARGO_PKG_VERSION").into(),
            git_sha: git,
        },
        round_trip_proof_sha256: sha256_hex(&proof_bytes),
        kernels: identities,
        checks,
        obligations: obligations.into_iter().collect(),
        railgun: RailgunSection {
            section_version: RAILGUN_SECTION_VERSION,
            analysis: "peacemaker_ir::passes::kernargs (A1/A3/A4)".into(),
            assumptions: ASSUMPTIONS.iter().map(|s| (*s).to_owned()).collect(),
            kernels: records,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::cfg::Body;
    use peacemaker_ir::descriptor::{
        KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3,
    };
    use peacemaker_ir::inst::{SymbolId, Wave};
    use peacemaker_ir::metadata::{HsaKernelMetadata, Kernarg, KernelMeta};
    use KernelArgumentAccess::{ReadOnly, ReadWrite, WriteOnly};

    /// A gfx1201 kernel from llvm-mc words with two pointer arguments at 0 and 8.
    fn facts(words: &[u32]) -> KernargFacts {
        let mut body = Body::default();
        let mut at = 0;
        while at < words.len() {
            let (inst, used) = peacemaker_ir::codec::gfx12::decode(&words[at..]).expect("decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            at += used;
        }
        peacemaker_ir::passes::cfg::build_blocks(&mut body, Arch::Gfx1201).expect("blocks");
        let descriptor = KernelDescriptor {
            group_segment_fixed_size: 0,
            private_segment_fixed_size: 0,
            kernarg_size: 16,
            kernel_code_entry_byte_offset: 0,
            compute_pgm_rsrc3: Rsrc3(0),
            compute_pgm_rsrc1: Rsrc1(0x1f),
            compute_pgm_rsrc2: Rsrc2(0x384),
            kernel_code_properties: KernelCodeProperties(0x408),
            kernarg_preload: KernargPreload(0),
            reserved: [0; 28],
        };
        let arg = |name: &str, offset| Kernarg {
            name: name.into(),
            size: 8,
            offset,
            value_kind: "global_buffer".into(),
            address_space: Some("global".into()),
        };
        let parsed = KernelMeta {
            name: "k".into(),
            symbol: "k.kd".into(),
            kernarg_segment_size: 16,
            args: vec![arg("a", 0), arg("b", 8)],
            ..Default::default()
        };
        let kernel = Kernel {
            symbol: SymbolId("k".into()),
            wave: Wave::Wave32,
            abi: Abi::Hsa {
                descriptor,
                metadata: HsaKernelMetadata {
                    raw_msgpack: Vec::new(),
                    parsed,
                },
            },
            body,
            origin: KernelOrigin::Authored {
                builder_crate: "test".into(),
                version: "0".into(),
                git: "0".into(),
            },
        };
        kernargs::analyze(&kernel, Arch::Gfx1201)
    }
    fn meta(effects: &[(u32, KernelArgumentAccess, bool)]) -> MetadataEffects {
        MetadataEffects {
            source: "sidecar".into(),
            effects: Some(effects.to_vec()),
            mutable_read_cache: None,
        }
    }
    fn failing(d: &Differential) -> Vec<u32> {
        d.args
            .iter()
            .filter(|a| a.status == "fail")
            .map(|a| a.offset)
            .collect()
    }

    /// Reads a (s[4:5]) and writes b after a v_writelane/v_readlane round trip.
    const READ_A_WRITE_B: &[u32] = &[
        0xf4004100, 0xf8000000, 0xbfc70000, 0xd7610028, 0x02010006, 0xd7610028, 0x02010207,
        0xbe860180, 0xd760000a, 0x02010128, 0xd760000b, 0x02010328, 0x7e000280, 0xee050004,
        0x00000001, 0x00000000, 0xbfc00000, 0xee06800a, 0x00800000, 0x00000000, 0xbfb00000,
    ];
    /// if (v0 < 16) p = a else p = b; load p. Reads both, writes neither.
    const READ_A_OR_B: &[u32] = &[
        0xf4004100, 0xf8000000, 0xbfc70000, 0x7c980090, 0xbe88206a, 0x7e040204, 0x7e060205,
        0x8d7e087e, 0x7e040206, 0x7e060207, 0x8c7e087e, 0xee05007c, 0x00000001, 0x00000002,
        0xbfb00000,
    ];
    /// Loads a pointer from a, stores through it: Unknown.
    const STORE_VIA_LOADED_POINTER: &[u32] = &[
        0xf4002100, 0xf8000000, 0xbfc70000, 0xf4002182, 0xf8000000, 0xbfc70000, 0x7e000280,
        0xee068006, 0x00000000, 0x00000000, 0xbfb00000,
    ];

    #[test]
    fn written_argument_the_metadata_calls_read_only_fails() {
        let f = facts(READ_A_WRITE_B);
        let d = cross_check(&f, &meta(&[(0, ReadOnly, true), (8, ReadOnly, true)]));
        assert_eq!((d.status.as_str(), failing(&d)), ("fail", vec![8]));
        assert!(d.args[1].reason.contains("read_only"));
        // The conservative default (no `.actual_access`) admits the write.
        let d = cross_check(&f, &meta(&[(0, ReadOnly, true), (8, ReadWrite, false)]));
        assert_eq!(d.status, "pass");
    }

    #[test]
    fn metadata_write_the_proven_summary_misses_fails_but_not_when_unknown() {
        let f = facts(READ_A_OR_B);
        assert_eq!(f.status, KernelStatus::Proven);
        let d = cross_check(&f, &meta(&[(0, ReadOnly, true), (8, ReadWrite, true)]));
        assert_eq!(
            failing(&d),
            vec![8],
            "explicit read_write that A1 never writes"
        );
        let d = cross_check(&f, &meta(&[(0, ReadOnly, true), (8, WriteOnly, true)]));
        assert_eq!(failing(&d), vec![8], "A1 reads a write_only argument");
        let unknown = facts(STORE_VIA_LOADED_POINTER);
        assert_eq!(unknown.status, KernelStatus::Unknown);
        let d = cross_check(
            &unknown,
            &meta(&[(0, ReadOnly, true), (8, WriteOnly, true)]),
        );
        assert_eq!(
            (d.status.as_str(), d.args[1].status.as_str()),
            ("pass", "not_applicable")
        );
        assert_eq!(
            cross_check(
                &unknown,
                &MetadataEffects {
                    source: "none".into(),
                    ..Default::default()
                }
            )
            .status,
            "not_applicable"
        );
    }

    #[test]
    fn receipt_for_a_lifted_object_binds_identities_and_facts() {
        let bytes = include_bytes!("../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        let receipt = certify(bytes, None).expect("certifies");
        assert_eq!(
            (
                receipt.schema_version,
                receipt.arch.as_str(),
                receipt.strict
            ),
            (RECEIPT_SCHEMA_VERSION, "gfx1201", false)
        );
        assert_eq!(receipt.object_sha256, sha256_hex(bytes));
        assert_eq!(receipt.kernels.len(), receipt.railgun.kernels.len());
        let lifted = lift_object(
            bytes,
            Options {
                frontend: Frontend::Hipcc,
            },
        )
        .unwrap();
        for (identity, stream) in receipt.kernels.iter().zip(&lifted.proof.streams) {
            assert_eq!(identity.text_sha256, hex(&stream.stream_sha256));
            assert_eq!(identity.text_range, [stream.entry, stream.size]);
        }
        for k in &receipt.railgun.kernels {
            assert_eq!(k.differential.source, "object-metadata", "{}", k.symbol);
            assert_ne!(
                k.differential.status, "fail",
                "{}: {:?}",
                k.symbol, k.differential.args
            );
        }
        let back: Receipt =
            serde_json::from_str(&serde_json::to_string(&receipt).unwrap()).unwrap();
        assert_eq!(back.railgun.kernels.len(), receipt.railgun.kernels.len());
    }
}
