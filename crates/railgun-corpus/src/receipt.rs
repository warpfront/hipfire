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
/// Every witness kind a receipt may name.
pub const KNOWN_WITNESSES: &[&str] = &[WITNESS_INDEXED_BIND, WITNESS_SEALED_TABLE];

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
}

/// The guarded-case digest under the exact object receipt (§A9.2; v2 binds
/// `sealed_tables` as well).
pub fn case_digest(
    object_sha256: &str,
    device_elf_sha256: &str,
    arch: &str,
    k: &KernelRecord,
) -> String {
    let bytes = serde_json::to_vec(&(
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
                if (!indexed && self.sealed_tables.is_empty())
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
