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
pub mod recording;

use std::collections::BTreeMap;

use peacemaker_ir::edit::analyze;
use peacemaker_ir::inst::{Abi, Arch, Frontend, Kernel, KernelOrigin, Program};
use peacemaker_ir::kernarg::{AccessClass, AccessMode, ArgFacts, ArgRole, KernargFacts, KernelStatus, MemPath, ReadCacheClass, RegionBound};
use peacemaker_ir::passes::kernargs;
use peacemaker_lift::kd::DescriptorCodec;
use peacemaker_lift::{lift_object, Options};
use radiowave::{CodeObjectCertification, CodeObjectInspection, CompileManifest, KernelArgumentAccess, KernelArgumentReport, KernelReport, MutableReadCache, Wavefront};
use sha2::{Digest, Sha256};

pub use railgun_corpus::receipt::{
    ArgCheck, ArgRecord, Check, Differential, DwordRecord, ImplicitRecord, KernelIdentity, KernelRecord, RailgunSection, Receipt,
    SideRecord, Tool, UnresolvedRecord, RAILGUN_SECTION_VERSION, RECEIPT_SCHEMA, RECEIPT_SCHEMA_VERSION,
};

/// Stated in every receipt: what a `Proven` A1 verdict rests on.
pub const ASSUMPTIONS: &[&str] = &[
    "AddressArithmetic: a pointer is never rebuilt by multiplication, shifting or bit-field extraction of a value loaded from memory",
    "AllocationWide: every recognised pointer effect covers its whole allocation (A2 bounds are M0b)",
];

#[derive(Debug, thiserror::Error)]
pub enum CertError {
    #[error("lift: {0}")]
    Lift(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn sha256_hex(bytes: &[u8]) -> String { hex(&Sha256::digest(bytes)) }
fn hex(bytes: &[u8]) -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() }

pub fn arch_name(arch: Arch) -> &'static str {
    match arch { Arch::Gfx1010 => "gfx1010", Arch::Gfx1030 => "gfx1030", Arch::Gfx1100 => "gfx1100", Arch::Gfx1151 => "gfx1151", Arch::Gfx1201 => "gfx1201" }
}

fn mode_name(mode: AccessMode) -> &'static str {
    match mode { AccessMode::Read => "read", AccessMode::Write => "write", AccessMode::ReadWrite => "read_write" }
}
fn access_name(access: KernelArgumentAccess) -> &'static str {
    match access { KernelArgumentAccess::ReadOnly => "read_only", KernelArgumentAccess::WriteOnly => "write_only", KernelArgumentAccess::ReadWrite => "read_write" }
}
pub fn read_cache_name(class: ReadCacheClass) -> &'static str {
    match class {
        ReadCacheClass::NotRead => "not_read", ReadCacheClass::VmemOnly => "vmem_only", ReadCacheClass::ScalarOnly => "scalar_only",
        ReadCacheClass::ScalarAndVmem => "scalar_and_vmem", ReadCacheClass::Unknown => "unknown",
    }
}
fn class_name(class: AccessClass) -> String {
    match class {
        AccessClass::VmemLoad => "vmem_load".into(), AccessClass::VmemStore => "vmem_store".into(), AccessClass::SmemLoad => "smem_load".into(),
        AccessClass::Atomic { commutative } => format!("atomic{}", if commutative { "_commutative" } else { "" }),
        AccessClass::LdsDma => "lds_dma".into(),
    }
}
fn path_name(path: MemPath) -> &'static str {
    match path { MemPath::Global => "global", MemPath::Buffer => "buffer", MemPath::Flat => "flat", MemPath::Smem => "smem", MemPath::SmemBuffer => "smem_buffer" }
}

/// Metadata effects of one kernel: `(offset, conservative access, explicit)`
/// for each `global_buffer` argument, from radiowave's `argument_effects`.
#[derive(Clone, Debug, Default)]
pub struct MetadataEffects { pub source: String, pub effects: Option<Vec<(u32, KernelArgumentAccess, bool)>>, pub mutable_read_cache: Option<MutableReadCache> }

/// The A1 ⊆ metadata-conservative differential for one kernel.
pub fn cross_check(facts: &KernargFacts, meta: &MetadataEffects) -> Differential {
    let mut out = Differential {
        source: meta.source.clone(), status: "not_applicable".into(), args: Vec::new(),
        radiowave_mutable_read_cache: meta.mutable_read_cache.map(|c| match c { MutableReadCache::VmemOnly => "vmem_only".into(), MutableReadCache::ScalarOrUnknown => "scalar_or_unknown".into() }),
    };
    let Some(effects) = &meta.effects else { return out };
    let proven = facts.status == KernelStatus::Proven;
    let mut failed = false;
    for &(offset, access, explicit) in effects {
        let arg = facts.arg_at(offset);
        let a1 = arg.and_then(|a| a.mode);
        let name = arg.map(|a| a.name.clone()).unwrap_or_default();
        let (status, reason) = match (a1, access) {
            (Some(m), KernelArgumentAccess::ReadOnly) if m.writes() => ("fail", "A1 writes an argument the metadata says is read_only".to_owned()),
            (Some(m), KernelArgumentAccess::WriteOnly) if m.reads() => ("fail", "A1 reads an argument the metadata says is write_only".to_owned()),
            (m, KernelArgumentAccess::WriteOnly | KernelArgumentAccess::ReadWrite) if explicit && !m.is_some_and(AccessMode::writes) => {
                if proven { ("fail", format!("metadata says {} but the Proven A1 summary never writes it", access_name(access))) }
                else { ("not_applicable", "A1 is Unknown: a missing write may sit in an unresolved access".to_owned()) }
            }
            _ => ("pass", String::new()),
        };
        failed |= status == "fail";
        out.args.push(ArgCheck { offset, name, a1: a1.map(|m| mode_name(m).to_owned()), metadata: access_name(access).into(), explicit, status: status.into(), reason });
    }
    out.status = if failed { "fail" } else { "pass" }.into();
    out
}

fn arg_record(a: &ArgFacts) -> ArgRecord {
    let mut classes: Vec<String> = a.classes().into_iter().map(class_name).collect();
    classes.dedup();
    let mut paths: Vec<String> = a.accesses.iter().map(|x| path_name(x.path).to_owned()).collect();
    paths.sort();
    paths.dedup();
    let mut cache: Vec<String> = a.accesses.iter().map(|x| {
        let c = x.cache;
        format!("th={} scope={} glc={} slc={} dlc={} nv={}", c.th, c.scope, u8::from(c.glc), u8::from(c.slc), u8::from(c.dlc), u8::from(c.nv))
    }).collect();
    cache.sort();
    cache.dedup();
    ArgRecord {
        offset: a.offset, size: a.size, index: a.index, name: a.name.clone(), value_kind: a.value_kind.clone(),
        role: match a.role { ArgRole::Pointer => "pointer", ArgRole::Integer => "integer", ArgRole::NotLoaded => "not_loaded" }.into(),
        mode: a.mode.map(|m| mode_name(m).to_owned()),
        bound: a.bound.map(|RegionBound::AllocationWide| "allocation_wide".to_owned()),
        classes, paths, cache_bits: cache, accesses: a.accesses.len(), shared: a.accesses.iter().any(|x| x.shared),
        read_cache: read_cache_name(a.read_cache).into(),
        sinks: a.sinks.names().into_iter().map(str::to_owned).collect(), loaded: a.loaded,
    }
}

/// Serialise one kernel's facts (plus its differential) for the receipt.
pub fn kernel_record(kernel: &Kernel, facts: &KernargFacts, meta: &MetadataEffects) -> KernelRecord {
    let pc_of = |id| kernel.body.insts.get(id).and_then(|i| i.prov.pc);
    let loaded = |kind: &str| facts.implicit.hidden_loaded.iter().any(|k| k.starts_with(kind));
    KernelRecord {
        symbol: kernel.symbol.0.clone(),
        status: match facts.status { KernelStatus::Proven => "proven", KernelStatus::Unknown => "unknown" }.into(),
        unresolved: facts.unresolved.iter().map(|u| UnresolvedRecord { pc: u.inst.and_then(pc_of), rule: u.rule.into(), detail: u.detail.clone() }).collect(),
        kernarg_size: facts.kernarg_size,
        read_cache: read_cache_name(facts.read_cache).into(),
        dynamic_kernarg_reads: facts.dynamic_kernarg_reads,
        taint_overflow: facts.taint_overflow,
        side: SideRecord { kernarg_segment: facts.side.kernarg_segment, module_global: facts.side.module_global, abi_packet: facts.side.abi_packet, local_flat: facts.side.local_flat, scratch: facts.side.scratch },
        implicit: ImplicitRecord {
            hidden_loaded: facts.implicit.hidden_loaded.clone(), hidden_grid: facts.implicit.hidden_grid.clone(),
            hidden_grid_live: facts.implicit.hidden_grid_live.clone(),
            loads_block_count: loaded("hidden_block_count_"), loads_group_size: loaded("hidden_group_size_"),
        },
        args: facts.args.iter().map(arg_record).collect(),
        dwords: facts.dwords.iter().map(|d| DwordRecord { offset: d.offset, loaded: d.loaded, sinks: d.sinks.names().into_iter().map(str::to_owned).collect() }).collect(),
        differential: cross_check(facts, meta),
        pm_obligations: BTreeMap::new(),
        pm_analyze_error: None,
    }
}

/// `.actual_access` per argument from the kernel's own metadata note.
fn metadata_arguments(raw: &[u8]) -> Vec<KernelArgumentReport> {
    let Ok(value) = rmpv::decode::read_value(&mut &raw[..]) else { return Vec::new() };
    let get = |map: &rmpv::Value, key: &str| -> Option<rmpv::Value> {
        map.as_map()?.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v.clone())
    };
    let Some(args) = get(&value, ".args") else { return Vec::new() };
    let Some(args) = args.as_array() else { return Vec::new() };
    args.iter().map(|a| KernelArgumentReport {
        offset: get(a, ".offset").and_then(|v| v.as_u64()).unwrap_or_default() as u32,
        size: get(a, ".size").and_then(|v| v.as_u64()).unwrap_or_default() as u32,
        value_kind: get(a, ".value_kind").and_then(|v| v.as_str().map(str::to_owned)).unwrap_or_default(),
        address_space: get(a, ".address_space").and_then(|v| v.as_str().map(str::to_owned)),
        actual_access: get(a, ".actual_access").and_then(|v| match v.as_str() {
            Some("read_only") => Some(KernelArgumentAccess::ReadOnly),
            Some("write_only") => Some(KernelArgumentAccess::WriteOnly),
            Some("read_write") => Some(KernelArgumentAccess::ReadWrite),
            _ => None,
        }),
    }).collect()
}

/// A radiowave certification for `bytes`: the runtime sidecar when given,
/// else a schema-5 manifest built from the object's own metadata notes, so
/// radiowave's `argument_effects` rule is applied either way.
fn radiowave_cert(bytes: &[u8], program: &Program, sidecar: Option<&str>) -> (String, Option<CodeObjectCertification>) {
    if let Some(json) = sidecar {
        if let Ok(cert) = CodeObjectCertification::from_json(bytes, json) { return ("sidecar".into(), Some(cert)); }
    }
    let kernels = program.kernels.iter().filter_map(|k| match &k.abi {
        Abi::Hsa { metadata, .. } => Some(KernelReport { name: k.symbol.0.clone(), arguments: metadata_arguments(&metadata.raw_msgpack), ..KernelReport::default() }),
        Abi::Raw { .. } => None,
    }).collect();
    let manifest = CompileManifest {
        schema_version: 5, compiler: "radiowave".into(), generated_unix_seconds: 0, source: Default::default(), output: Default::default(),
        arch: arch_name(program.target.arch).into(), wavefront: Wavefront::Wave32, scheduler_profile: Default::default(),
        hipcc: Default::default(), hipcc_version: String::new(), command: Vec::new(), source_sha256: String::new(),
        support_header_sha256: String::new(), output_sha256: sha256_hex(bytes),
        inspection: Some(CodeObjectInspection { bundle_target: String::new(), identity: None, kernels }), peacemaker: None,
    };
    let cert = serde_json::to_string(&manifest).ok().and_then(|json| CodeObjectCertification::from_json(bytes, &json).ok());
    ("object-metadata".into(), cert)
}

fn metadata_effects(source: &str, cert: Option<&CodeObjectCertification>, symbol: &str) -> MetadataEffects {
    let Some(cert) = cert else { return MetadataEffects { source: "none".into(), ..MetadataEffects::default() } };
    let explicit: BTreeMap<u32, bool> = cert.kernel(symbol).map(|k| k.arguments.iter().map(|a| (a.offset, a.actual_access.is_some())).collect()).unwrap_or_default();
    let effects = cert.argument_effects(symbol).map(|v| v.into_iter().map(|(offset, access)| {
        let offset = offset as u32;
        (offset, access, explicit.get(&offset).copied().unwrap_or(false))
    }).collect());
    MetadataEffects {
        source: source.into(), effects,
        mutable_read_cache: (source == "sidecar").then(|| cert.mutable_read_cache(symbol)),
    }
}

fn git_sha() -> String {
    let dir = env!("CARGO_MANIFEST_DIR");
    let run = |args: &[&str]| std::process::Command::new("git").arg("-C").arg(dir).args(args).output().ok()
        .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    let head = run(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = run(&["status", "--porcelain", "--", "crates/peacemaker-ir", "crates/peacemaker-lift", "crates/railgun-cert"]).is_some_and(|s| !s.is_empty());
    if dirty { format!("{head}-dirty") } else { head }
}

/// Facts for one kernel of a lifted program: `edit::analyze` (the `Analyzed`
/// revision) when it succeeds, else the kernargs pass alone.
pub fn kernel_facts(program: &Program, index: usize) -> (KernargFacts, BTreeMap<String, usize>, Option<String>) {
    let kernel = &program.kernels[index];
    match analyze(program.clone(), &kernel.symbol) {
        Ok(analyzed) => {
            let mut obligations = BTreeMap::new();
            for o in &analyzed.obligations { *obligations.entry(o.rule_id.clone()).or_insert(0) += 1; }
            (analyzed.facts.kernargs, obligations, None)
        }
        Err(e) => (kernargs::analyze(kernel, program.target.arch), BTreeMap::new(), Some(e.to_string())),
    }
}

/// A kernel's code identity (text stream, descriptor and metadata digests)
/// and the SHA-256 of the device ELF it came from.
fn kernel_identity(kernel: &Kernel, proof: &peacemaker_ir::state::RoundTripProof) -> (KernelIdentity, String) {
    let (kd_sha256, metadata_sha256) = match &kernel.abi {
        Abi::Hsa { descriptor, metadata } => (sha256_hex(&descriptor.to_bytes()), sha256_hex(&metadata.raw_msgpack)),
        Abi::Raw { .. } => (String::new(), String::new()),
    };
    let (text_range, device) = match &kernel.origin {
        KernelOrigin::Frontend { entry_va, size, object_sha256, .. } => ([*entry_va, *size], hex(object_sha256)),
        KernelOrigin::Authored { .. } => ([0, 0], String::new()),
    };
    let text_sha256 = proof.streams.iter().find(|s| s.entry == text_range[0]).map(|s| hex(&s.stream_sha256)).unwrap_or_default();
    (KernelIdentity { symbol: kernel.symbol.0.clone(), text_range, text_sha256, kd_sha256, metadata_sha256 }, device)
}

/// The code identity of every kernel in a code object, without analysis:
/// two objects whose identities agree carry the same kernel code, descriptors
/// and metadata even when their bytes differ elsewhere.
pub fn code_identity(bytes: &[u8]) -> Result<Vec<KernelIdentity>, CertError> {
    let lifted = lift_object(bytes, Options { frontend: Frontend::Hipcc }).map_err(|e| CertError::Lift(e.to_string()))?;
    Ok(lifted.program.kernels.iter().map(|k| kernel_identity(k, &lifted.proof).0).collect())
}

/// Lift, analyse and serialise one code object. `sidecar` is the runtime's
/// `*.radiowave.json` for these exact bytes, when one exists.
pub fn certify(bytes: &[u8], sidecar: Option<&str>) -> Result<Receipt, CertError> {
    let lifted = lift_object(bytes, Options { frontend: Frontend::Hipcc }).map_err(|e| CertError::Lift(e.to_string()))?;
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
        let (facts, pm, pm_error) = kernel_facts(program, index);
        let meta = metadata_effects(&source, cert.as_ref(), &kernel.symbol.0);
        let mut record = kernel_record(kernel, &facts, &meta);
        record.pm_obligations = pm;
        record.pm_analyze_error = pm_error;
        if facts.args.iter().any(|a| a.bound.is_some()) { obligations.insert("InBoundsNoAlias".to_owned()); }
        if facts.status == KernelStatus::Proven { obligations.insert("AddressArithmetic".to_owned()); }
        records.push(record);
    }
    let proven = records.iter().filter(|r| r.status == "proven").count();
    let failed: Vec<&str> = records.iter().filter(|r| r.differential.status == "fail").map(|r| r.symbol.as_str()).collect();
    let checks = vec![
        Check { id: "roundtrip".into(), status: "pass".into(), detail: format!("{} kernels re-encode byte-identically; module re-emits identically", program.kernels.len()) },
        Check { id: "railgun-a1".into(), status: if proven == records.len() { "proven" } else { "unknown" }.into(), detail: format!("{proven}/{} kernels Proven", records.len()) },
        Check { id: "g8-metadata-differential".into(), status: if failed.is_empty() { "pass" } else { "fail" }.into(), detail: if failed.is_empty() { format!("source {source}") } else { format!("source {source}; failing: {}", failed.join(", ")) } },
    ];
    let proof = &lifted.proof;
    let mut proof_bytes = proof.input_sha256.to_vec();
    for s in &proof.streams { proof_bytes.extend(s.entry.to_le_bytes()); proof_bytes.extend(s.size.to_le_bytes()); proof_bytes.extend(s.stream_sha256); }
    let git = git_sha();
    Ok(Receipt {
        schema: RECEIPT_SCHEMA.into(), schema_version: RECEIPT_SCHEMA_VERSION,
        object_sha256: sha256_hex(bytes), device_elf_sha256: device_sha, arch: arch_name(arch).into(),
        strict: false, tier: "D".into(),
        lifter: Tool { name: "peacemaker-lift".into(), crate_version: env!("CARGO_PKG_VERSION").into(), git_sha: git.clone() },
        certifier: Tool { name: "railgun-cert".into(), crate_version: env!("CARGO_PKG_VERSION").into(), git_sha: git },
        round_trip_proof_sha256: sha256_hex(&proof_bytes),
        kernels: identities, checks, obligations: obligations.into_iter().collect(),
        railgun: RailgunSection {
            section_version: RAILGUN_SECTION_VERSION, analysis: "peacemaker_ir::passes::kernargs (A1/A3/A4)".into(),
            assumptions: ASSUMPTIONS.iter().map(|s| (*s).to_owned()).collect(), kernels: records,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use peacemaker_ir::cfg::Body;
    use peacemaker_ir::descriptor::{KernargPreload, KernelCodeProperties, KernelDescriptor, Rsrc1, Rsrc2, Rsrc3};
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
        let descriptor = KernelDescriptor { group_segment_fixed_size: 0, private_segment_fixed_size: 0, kernarg_size: 16,
            kernel_code_entry_byte_offset: 0, compute_pgm_rsrc3: Rsrc3(0), compute_pgm_rsrc1: Rsrc1(0x1f), compute_pgm_rsrc2: Rsrc2(0x384),
            kernel_code_properties: KernelCodeProperties(0x408), kernarg_preload: KernargPreload(0), reserved: [0; 28] };
        let arg = |name: &str, offset| Kernarg { name: name.into(), size: 8, offset, value_kind: "global_buffer".into(), address_space: Some("global".into()) };
        let parsed = KernelMeta { name: "k".into(), symbol: "k.kd".into(), kernarg_segment_size: 16, args: vec![arg("a", 0), arg("b", 8)], ..Default::default() };
        let kernel = Kernel { symbol: SymbolId("k".into()), wave: Wave::Wave32,
            abi: Abi::Hsa { descriptor, metadata: HsaKernelMetadata { raw_msgpack: Vec::new(), parsed } }, body,
            origin: KernelOrigin::Authored { builder_crate: "test".into(), version: "0".into(), git: "0".into() } };
        kernargs::analyze(&kernel, Arch::Gfx1201)
    }
    fn meta(effects: &[(u32, KernelArgumentAccess, bool)]) -> MetadataEffects {
        MetadataEffects { source: "sidecar".into(), effects: Some(effects.to_vec()), mutable_read_cache: None }
    }
    fn failing(d: &Differential) -> Vec<u32> { d.args.iter().filter(|a| a.status == "fail").map(|a| a.offset).collect() }

    /// Reads a (s[4:5]) and writes b after a v_writelane/v_readlane round trip.
    const READ_A_WRITE_B: &[u32] = &[0xf4004100, 0xf8000000, 0xbfc70000, 0xd7610028, 0x02010006, 0xd7610028, 0x02010207,
        0xbe860180, 0xd760000a, 0x02010128, 0xd760000b, 0x02010328, 0x7e000280, 0xee050004, 0x00000001, 0x00000000,
        0xbfc00000, 0xee06800a, 0x00800000, 0x00000000, 0xbfb00000];
    /// if (v0 < 16) p = a else p = b; load p. Reads both, writes neither.
    const READ_A_OR_B: &[u32] = &[0xf4004100, 0xf8000000, 0xbfc70000, 0x7c980090, 0xbe88206a, 0x7e040204, 0x7e060205,
        0x8d7e087e, 0x7e040206, 0x7e060207, 0x8c7e087e, 0xee05007c, 0x00000001, 0x00000002, 0xbfb00000];
    /// Loads a pointer from a, stores through it: Unknown.
    const STORE_VIA_LOADED_POINTER: &[u32] = &[0xf4002100, 0xf8000000, 0xbfc70000, 0xf4002182, 0xf8000000, 0xbfc70000,
        0x7e000280, 0xee068006, 0x00000000, 0x00000000, 0xbfb00000];

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
        assert_eq!(failing(&d), vec![8], "explicit read_write that A1 never writes");
        let d = cross_check(&f, &meta(&[(0, ReadOnly, true), (8, WriteOnly, true)]));
        assert_eq!(failing(&d), vec![8], "A1 reads a write_only argument");
        let unknown = facts(STORE_VIA_LOADED_POINTER);
        assert_eq!(unknown.status, KernelStatus::Unknown);
        let d = cross_check(&unknown, &meta(&[(0, ReadOnly, true), (8, WriteOnly, true)]));
        assert_eq!((d.status.as_str(), d.args[1].status.as_str()), ("pass", "not_applicable"));
        assert_eq!(cross_check(&unknown, &MetadataEffects { source: "none".into(), ..Default::default() }).status, "not_applicable");
    }

    #[test]
    fn receipt_for_a_lifted_object_binds_identities_and_facts() {
        let bytes = include_bytes!("../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        let receipt = certify(bytes, None).expect("certifies");
        assert_eq!((receipt.schema_version, receipt.arch.as_str(), receipt.strict), (RECEIPT_SCHEMA_VERSION, "gfx1201", false));
        assert_eq!(receipt.object_sha256, sha256_hex(bytes));
        assert_eq!(receipt.kernels.len(), receipt.railgun.kernels.len());
        let lifted = lift_object(bytes, Options { frontend: Frontend::Hipcc }).unwrap();
        for (identity, stream) in receipt.kernels.iter().zip(&lifted.proof.streams) {
            assert_eq!(identity.text_sha256, hex(&stream.stream_sha256));
            assert_eq!(identity.text_range, [stream.entry, stream.size]);
        }
        for k in &receipt.railgun.kernels {
            assert_eq!(k.differential.source, "object-metadata", "{}", k.symbol);
            assert_ne!(k.differential.status, "fail", "{}: {:?}", k.symbol, k.differential.args);
        }
        let back: Receipt = serde_json::from_str(&serde_json::to_string(&receipt).unwrap()).unwrap();
        assert_eq!(back.railgun.kernels.len(), receipt.railgun.kernels.len());
    }
}
