//! The `peacemaker-receipt` schema (receipt v1 = radiowave `PeacemakerRecord`
//! schema 6; the `railgun` section makes it 7). `railgun-cert` writes these
//! records offline; the runtime reads them through this crate, which depends
//! on no peacemaker crate (railgun design D12).
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA: &str = "peacemaker-receipt";
/// Receipt v1 (architecture.md §3) is schema 6; the `railgun` section is 7.
pub const RECEIPT_SCHEMA_VERSION: u32 = 7;
pub const RAILGUN_SECTION_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool { pub name: String, pub crate_version: String, pub git_sha: String }

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
pub struct Check { pub id: String, pub status: String, pub detail: String }

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
pub struct UnresolvedRecord { pub pc: Option<u32>, pub rule: String, pub detail: String }

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SideRecord { pub kernarg_segment: u32, pub module_global: u32, pub abi_packet: u32, pub local_flat: u32, pub scratch: u32 }

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
pub struct DwordRecord { pub offset: u32, pub loaded: bool, pub sinks: Vec<String> }

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
}
