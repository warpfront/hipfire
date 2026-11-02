// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! F3 for the ACO pilot: the finite HSA entry adapter and its native packaging.
//!
//! A Mesa/ACO compute shader expects the RADV entry state: user SGPRs holding inline push
//! constants (here full 64-bit buffer-device addresses plus scalars) and the hardware
//! work-group / local-id registers. HIP launches an HSA kernel instead: the kernarg segment
//! pointer in `s[0:1]` and the same architected `ttmp9`/`ttmp7`/`v0` hardware state on
//! gfx12. [`adapt_to_hsa`] bridges the two with a **finite prologue** and nothing else:
//!
//! 1. optionally copy the kernarg segment pointer to a temporary SGPR pair, chosen from the
//!    registers the shared liveness analysis proves free at the body entry (the pair is
//!    never one the contract binds);
//! 2. load every bound push-constant dword from the HIP kernarg segment, 64-bit pointers as
//!    whole 64-bit loads (no truncation, no address-high rule);
//! 3. derive any bound work-group-id / constant SGPRs;
//! 4. `s_wait_kmcnt 0`, then fall through into the untouched ACO body.
//!
//! The body is the producer's executable range: its typed instructions re-encoded by the
//! shared codec (`peacemaker_lift::emit::words`, never `prov.bytes`) and compared with the
//! certified stream hash and the contract's raw executable hash. Branch offsets are
//! body-relative, so they are unchanged by construction; the relift of the emitted object
//! additionally proves the CFG block boundaries are the raw ones shifted by the prologue.
//!
//! Packaging reuses the native descriptor / metadata / ELF writer through
//! `native::link_encoded` (typed encoded words in, no text front end for the body). The
//! prologue is written as instruction text lowered by `peacemaker_lift::text::parse_line`
//! and the gfx12 codec, the same typed path the native writer uses for builder text.
//!
//! The emitted object is lifted back (`lift_object`, `Frontend::Aco`), its words are
//! compared with prologue ++ body, and the HSA program goes through the same shared
//! certifier as the raw one (`aco_cert::certify_program`): every obligation must be closed.
//!
//! What the adapter does not do, and therefore rejects: any entry fact it has no
//! HSA-provided or contract-bound source for; scratch, relocations, `s_getpc`-style
//! PC-relative bodies, dynamic LDS, wave64, FP/mode state the gfx12 descriptor cannot
//! express, and 32-bit or otherwise truncated pointer apertures. There is no whole-kernel
//! register allocation, no descriptor truncation, no MODE reset to builder defaults.
use crate::aco_cert::{
    certify_program, CertError, CertificationReport, IdentityDomain, LaunchContract,
    StaticCertificate,
};
use crate::native::{self, EncodedKernel};
use crate::plan::Access;
use peacemaker_ir::cfg::BlockId;
use peacemaker_ir::codec::gfx12;
use peacemaker_ir::inst::{
    Abi, Arch as IrArch, Axis, EntryCell, EntryValue, FpState, Frontend, KernelOrigin,
    RawResources, Setting, SymbolId, Target, Wave,
};
use peacemaker_ir::passes::liveness::{LivePoint, Liveness, Loc};
use peacemaker_ir::passes::resources::summarize;
use peacemaker_ir::reg::{Kind, RegRef};
use peacemaker_lift::text::parse_line;
use peacemaker_lift::{emit, lift_object, Options};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// Adapter identity recorded in the emission proof and the candidate manifest.
pub const ADAPTER_VERSION: &str = concat!(
    "peacemaker-isa ",
    env!("CARGO_PKG_VERSION"),
    " aco-hsa adapter v1"
);
/// Schema of [`AcoHipContract::from_json`].
pub const CONTRACT_SCHEMA: &str = "pm-aco-hip-contract-v1";
/// Schema of [`AcoHsaEmission::candidate_manifest`].
pub const CANDIDATE_SCHEMA: &str = "aco-pm-candidate-v1";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a contract cell cannot be adapted. Every variant is a causal rejection; none is
/// recoverable by guessing.
#[derive(Debug)]
pub enum AcoError {
    /// The contract is self-inconsistent or malformed.
    Contract(String),
    /// The raw certificate is not an acceptable adapter input.
    Certificate(String),
    /// Contract and raw program disagree.
    Mismatch(String),
    /// The raw program needs something HSA/HIP cannot provide (or this adapter does not).
    Unsupported(String),
    /// An entry fact has no HSA-provided or contract-bound source.
    Entry(String),
    /// The raw body is not preserved or cannot be preserved.
    Body(String),
    /// No valid prologue exists (no free temporary, bad load shape, encode failure).
    Prologue(String),
    /// Native packaging failed.
    Emit(String),
    /// The emitted object does not lift back to prologue ++ body.
    Relift(String),
    /// The emitted HSA program is not statically certified.
    Certify(CertError),
    /// The producer's artifact directory is inconsistent with its `artifact.json`.
    Artifact(String),
}

impl fmt::Display for AcoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, s) = match self {
            Self::Contract(s) => ("contract", s.as_str()),
            Self::Certificate(s) => ("certificate", s.as_str()),
            Self::Mismatch(s) => ("contract/raw mismatch", s.as_str()),
            Self::Unsupported(s) => ("unsupported", s.as_str()),
            Self::Entry(s) => ("entry", s.as_str()),
            Self::Body(s) => ("body", s.as_str()),
            Self::Prologue(s) => ("prologue", s.as_str()),
            Self::Emit(s) => ("emit", s.as_str()),
            Self::Relift(s) => ("relift", s.as_str()),
            Self::Artifact(s) => ("artifact", s.as_str()),
            Self::Certify(e) => return write!(f, "emitted HSA program not certified: {e}"),
        };
        write!(f, "{kind}: {s}")
    }
}
impl std::error::Error for AcoError {}

// ---------------------------------------------------------------------------
// Contract
// ---------------------------------------------------------------------------

/// The address space the shader's pointers live in. Only full 64-bit global addresses are
/// representable on HIP VMM; the other cells are named so they can be refused by name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Aperture {
    /// 64-bit global (HIP VMM / flat global) addresses carried whole.
    Global64,
    /// A 32-bit pointer plus a fixed address-high dword (RADV `32bit_pointers`).
    Pointer32 { high_dword: u32 },
    /// Architected flat scratch.
    FlatScratch,
}

/// A HIP kernel argument.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HipArgKind {
    /// An 8-byte global pointer. `extent_bytes` is the byte extent the fixed shape domain
    /// accesses through it (the oracle's guard bound).
    GlobalPointer {
        access: Option<Access>,
        extent_bytes: u64,
    },
    /// A by-value scalar of the arg's `size`.
    ByValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HipArg {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub kind: HipArgKind,
}

/// Where a RADV entry SGPR run gets its value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindingSource {
    /// A 64-bit pointer kernarg, loaded whole into an even SGPR pair.
    KernargPointer { arg: usize },
    /// A by-value kernarg of 4 or 8 bytes (an inline push constant).
    KernargScalar { arg: usize },
    /// The HSA work-group id (`ttmp9` / `ttmp7`), copied into an SGPR.
    WorkgroupId(Axis),
    /// A fixed dword.
    Constant(u32),
}

/// One run of RADV entry SGPRs and what initialises it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntryBinding {
    pub dst: RegRef,
    pub source: BindingSource,
}

/// The finite launch domain the cell is claimed for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeDomain {
    pub block: [u32; 3],
    pub grid: [u32; 3],
    /// Fixed value of every by-value arg `(arg index, value)`.
    pub scalars: Vec<(usize, u64)>,
    /// Evidence hash (producer's exact-algorithm attestation or an oracle receipt) that the
    /// shader is correct on odd-tail shapes; `None` = the cell is the fixed fixture only.
    pub odd_tail_evidence_sha256: Option<[u8; 32]>,
}

/// Everything the adapter is told about the HIP side. Nothing here is inferred from the
/// binary; `adapt_to_hsa` cross-checks it against the certified raw program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcoHipContract {
    /// The raw program's target (exact).
    pub target: Target,
    /// The raw kernel's symbol inside the lifted program.
    pub raw_symbol: SymbolId,
    /// The exported HIP kernel symbol.
    pub hip_symbol: String,
    pub wave: Wave,
    pub args: Vec<HipArg>,
    pub kernarg_size: u32,
    pub bindings: Vec<EntryBinding>,
    /// Must equal the raw shader config's FP/mode state exactly.
    pub fp: FpState,
    /// Must equal the raw resources exactly (`lds_bytes` is the fixed LDS).
    pub resources: RawResources,
    /// Dynamic LDS maximum; only 0 is certifiable for a lifted kernel.
    pub dynamic_lds_max: u32,
    pub shape: ShapeDomain,
    pub aperture: Aperture,
    /// F1 artifact manifest hash, SPIR-V hash, the raw executable hash and sidecar hash.
    pub artifact_sha256: [u8; 32],
    /// SHA-256 of the exact `shader.rawspec.json` bytes (entry facts, FP, resources) the
    /// raw program was lifted under.
    pub rawspec_sha256: [u8; 32],
    pub spirv_sha256: [u8; 32],
    pub raw_binary_sha256: [u8; 32],
    pub raw_exec_sha256: [u8; 32],
    pub sidecar_sha256: [u8; 32],
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(2 * bytes.len());
    for b in bytes {
        write!(s, "{b:02x}").expect("String write");
    }
    s
}

fn ident_ok(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Canonical, length-prefixed serialisation hashed into the contract identity.
#[derive(Default)]
struct Canon(Vec<u8>);
impl Canon {
    fn u(&mut self, v: u64) {
        self.0.extend(v.to_le_bytes())
    }
    fn s(&mut self, s: &str) {
        self.u(s.len() as u64);
        self.0.extend(s.as_bytes())
    }
    fn b(&mut self, b: &[u8]) {
        self.u(b.len() as u64);
        self.0.extend(b)
    }
}

impl AcoHipContract {
    /// Self-consistency of the contract alone (no raw program involved).
    pub fn validate(&self) -> Result<(), AcoError> {
        let bad = |m: String| Err(AcoError::Contract(m));
        if !ident_ok(&self.hip_symbol) {
            return bad(format!(
                "hip symbol {:?} is not a plain identifier",
                self.hip_symbol
            ));
        }
        if self.raw_symbol.0.is_empty() {
            return bad("empty raw symbol".into());
        }
        if self.kernarg_size == 0 || self.kernarg_size > 4096 {
            return bad(format!(
                "kernarg size {} is outside 1..=4096",
                self.kernarg_size
            ));
        }
        let mut spans: Vec<(u32, u32, &str)> = Vec::new();
        for arg in &self.args {
            if !ident_ok(&arg.name) {
                return bad(format!("arg name {:?} is not a plain identifier", arg.name));
            }
            if !matches!(arg.size, 1 | 2 | 4 | 8) {
                return bad(format!("arg {} has size {}", arg.name, arg.size));
            }
            if arg.offset % arg.size != 0 {
                return bad(format!(
                    "arg {} offset {} is not aligned to its size {}",
                    arg.name, arg.offset, arg.size
                ));
            }
            let end = arg
                .offset
                .checked_add(arg.size)
                .ok_or_else(|| AcoError::Contract(format!("arg {} overflows", arg.name)))?;
            if end > self.kernarg_size {
                return bad(format!(
                    "arg {} ends at {end}, past kernarg size {}",
                    arg.name, self.kernarg_size
                ));
            }
            if let HipArgKind::GlobalPointer { .. } = arg.kind {
                if arg.size != 8 {
                    return bad(format!(
                        "pointer arg {} is {} bytes: pointers are carried as full 64 bits",
                        arg.name, arg.size
                    ));
                }
            }
            if let Some((_, _, other)) = spans.iter().find(|(o, e, _)| arg.offset < *e && *o < end)
            {
                return bad(format!("arg {} overlaps arg {other}", arg.name));
            }
            spans.push((arg.offset, end, arg.name.as_str()));
        }
        for dims in [self.shape.block, self.shape.grid] {
            if dims.contains(&0) {
                return bad(format!("empty launch dimension {dims:?}"));
            }
        }
        let threads: u64 = self.shape.block.iter().map(|&d| u64::from(d)).product();
        if threads > 1024 {
            return bad(format!(
                "block {:?} has {threads} work-items",
                self.shape.block
            ));
        }
        for (arg, value) in &self.shape.scalars {
            let a = self.args.get(*arg).ok_or_else(|| {
                AcoError::Contract(format!("shape scalar names missing arg {arg}"))
            })?;
            if a.kind != HipArgKind::ByValue {
                return bad(format!("shape scalar names pointer arg {}", a.name));
            }
            if a.size < 8 && value >> (8 * a.size) != 0 {
                return bad(format!(
                    "shape scalar {} = {value} exceeds its {} bytes",
                    a.name, a.size
                ));
            }
        }
        for (i, a) in self.args.iter().enumerate() {
            if a.kind == HipArgKind::ByValue
                && self.shape.scalars.iter().filter(|(x, _)| *x == i).count() != 1
            {
                return bad(format!(
                    "by-value arg {} needs exactly one fixed shape value",
                    a.name
                ));
            }
        }
        let mut used: BTreeSet<u16> = BTreeSet::new();
        for b in &self.bindings {
            b.dst.validate().map_err(AcoError::Contract)?;
            if b.dst.kind != Kind::S {
                return bad(format!("binding destination {} is not an SGPR run", b.dst));
            }
            let want = match b.source {
                BindingSource::KernargPointer { arg } => {
                    let a = self.args.get(arg).ok_or_else(|| {
                        AcoError::Contract(format!("binding names missing arg {arg}"))
                    })?;
                    if !matches!(a.kind, HipArgKind::GlobalPointer { .. }) {
                        return bad(format!("pointer binding names by-value arg {}", a.name));
                    }
                    if b.dst.base % 2 != 0 {
                        return bad(format!(
                            "pointer {} needs an even SGPR pair, got {}",
                            a.name, b.dst
                        ));
                    }
                    2
                }
                BindingSource::KernargScalar { arg } => {
                    let a = self.args.get(arg).ok_or_else(|| {
                        AcoError::Contract(format!("binding names missing arg {arg}"))
                    })?;
                    if a.kind != HipArgKind::ByValue || !matches!(a.size, 4 | 8) {
                        return bad(format!(
                            "scalar binding of arg {} (must be a 4 or 8 byte by-value arg)",
                            a.name
                        ));
                    }
                    (a.size / 4) as u8
                }
                BindingSource::WorkgroupId(_) | BindingSource::Constant(_) => 1,
            };
            if b.dst.len != want {
                return bad(format!(
                    "binding {:?} needs {want} dword(s), destination {} has {}",
                    b.source, b.dst, b.dst.len
                ));
            }
            for n in b.dst.base..b.dst.base + u16::from(b.dst.len) {
                if !used.insert(n) {
                    return bad(format!("two bindings write s{n}"));
                }
            }
        }
        Ok(())
    }

    /// SHA-256 of the canonical contract serialisation (every field, length-prefixed).
    pub fn sha256(&self) -> [u8; 32] {
        let mut c = Canon::default();
        c.s(CONTRACT_SCHEMA);
        c.s(&format!("{:?}", self.target));
        c.s(&self.raw_symbol.0);
        c.s(&self.hip_symbol);
        c.s(&format!("{:?}", self.wave));
        c.u(self.args.len() as u64);
        for a in &self.args {
            c.s(&a.name);
            c.u(u64::from(a.offset));
            c.u(u64::from(a.size));
            match &a.kind {
                HipArgKind::GlobalPointer {
                    access,
                    extent_bytes,
                } => {
                    c.s("ptr");
                    c.s(access.map_or("none", Access::name));
                    c.u(*extent_bytes)
                }
                HipArgKind::ByValue => c.s("value"),
            }
        }
        c.u(u64::from(self.kernarg_size));
        c.u(self.bindings.len() as u64);
        for b in &self.bindings {
            c.s(&format!("{:?}", b))
        }
        c.s(&format!("{:?}", self.fp));
        c.s(&format!("{:?}", self.resources));
        c.u(u64::from(self.dynamic_lds_max));
        c.s(&format!("{:?}", self.shape));
        c.s(&format!("{:?}", self.aperture));
        for h in [
            &self.artifact_sha256,
            &self.rawspec_sha256,
            &self.spirv_sha256,
            &self.raw_binary_sha256,
            &self.raw_exec_sha256,
            &self.sidecar_sha256,
        ] {
            c.b(h)
        }
        sha256(&c.0)
    }

    /// The kernarg template the oracle hashes: `kernarg_size` bytes with every pointer slot
    /// zeroed and every by-value slot holding its fixed shape value (little endian), i.e. the
    /// captured launch argument ABI of the base fixture.
    pub fn kernarg_template(&self) -> Vec<u8> {
        let mut t = vec![0u8; self.kernarg_size as usize];
        for (arg, value) in &self.shape.scalars {
            let a = &self.args[*arg];
            let (o, n) = (a.offset as usize, a.size as usize);
            t[o..o + n].copy_from_slice(&value.to_le_bytes()[..n]);
        }
        t
    }

    /// Parse a `pm-aco-hip-contract-v1` document. All keys are required; unknown keys reject.
    pub fn from_json(text: &str) -> Result<Self, AcoError> {
        let doc: Value = serde_json::from_str(text)
            .map_err(|e| AcoError::Contract(format!("contract JSON: {e}")))?;
        let m = object(
            &doc,
            "contract",
            &[
                "schema",
                "target",
                "raw_symbol",
                "hip_symbol",
                "wave",
                "kernarg_size",
                "args",
                "bindings",
                "fp",
                "resources",
                "dynamic_lds_max",
                "shape",
                "aperture",
                "artifact_sha256",
                "rawspec_sha256",
                "spirv_sha256",
                "raw_binary_sha256",
                "raw_exec_sha256",
                "sidecar_sha256",
            ],
        )?;
        if str_of(m, "schema")? != CONTRACT_SCHEMA {
            return Err(AcoError::Contract(format!(
                "schema is not {CONTRACT_SCHEMA}"
            )));
        }
        let t = object(
            &m["target"],
            "target",
            &["arch", "xnack", "sramecc", "abi_version"],
        )?;
        let setting = |k: &str| -> Result<Setting, AcoError> {
            Ok(match str_of(t, k)? {
                "any" => Setting::Any,
                "on" => Setting::On,
                "off" => Setting::Off,
                o => return Err(AcoError::Contract(format!("target.{k} {o:?}"))),
            })
        };
        let arch = match str_of(t, "arch")? {
            "gfx1100" => IrArch::Gfx1100,
            "gfx1151" => IrArch::Gfx1151,
            "gfx1201" => IrArch::Gfx1201,
            o => return Err(AcoError::Contract(format!("target.arch {o:?}"))),
        };
        let target = Target {
            arch,
            xnack: setting("xnack")?,
            sramecc: setting("sramecc")?,
            abi_version: u8::try_from(uint(t, "abi_version")?)
                .map_err(|_| AcoError::Contract("target.abi_version".into()))?,
        };
        let wave = match uint(m, "wave")? {
            32 => Wave::Wave32,
            64 => Wave::Wave64,
            o => return Err(AcoError::Contract(format!("wave {o}"))),
        };
        let mut args = Vec::new();
        for (i, a) in array(m, "args")?.iter().enumerate() {
            let am = object(
                a,
                &format!("args[{i}]"),
                &["name", "offset", "size", "kind", "access", "extent_bytes"],
            )?;
            let kind = match str_of(am, "kind")? {
                "global_pointer" => {
                    let access = match &am["access"] {
                        Value::Null => None,
                        Value::String(s) => Some(match s.as_str() {
                            "read_only" => Access::ReadOnly,
                            "write_only" => Access::WriteOnly,
                            "read_write" => Access::ReadWrite,
                            o => return Err(AcoError::Contract(format!("args[{i}].access {o:?}"))),
                        }),
                        _ => {
                            return Err(AcoError::Contract(format!(
                                "args[{i}].access must be a string or null"
                            )))
                        }
                    };
                    HipArgKind::GlobalPointer {
                        access,
                        extent_bytes: uint(am, "extent_bytes")?,
                    }
                }
                "by_value" => {
                    if !am["access"].is_null() || !am["extent_bytes"].is_null() {
                        return Err(AcoError::Contract(format!(
                            "args[{i}]: by_value takes null access and extent_bytes"
                        )));
                    }
                    HipArgKind::ByValue
                }
                o => return Err(AcoError::Contract(format!("args[{i}].kind {o:?}"))),
            };
            args.push(HipArg {
                name: str_of(am, "name")?.to_owned(),
                offset: u32_of(am, "offset")?,
                size: u32_of(am, "size")?,
                kind,
            });
        }
        let mut bindings = Vec::new();
        for (i, b) in array(m, "bindings")?.iter().enumerate() {
            let bm = object(b, &format!("bindings[{i}]"), &["dst", "source"])?;
            let d = object(&bm["dst"], "binding dst", &["kind", "base", "len"])?;
            if str_of(d, "kind")? != "sgpr" {
                return Err(AcoError::Contract(format!(
                    "bindings[{i}].dst.kind must be sgpr"
                )));
            }
            let dst = RegRef {
                kind: Kind::S,
                base: u16::try_from(uint(d, "base")?)
                    .map_err(|_| AcoError::Contract("dst.base".into()))?,
                len: u8::try_from(uint(d, "len")?)
                    .map_err(|_| AcoError::Contract("dst.len".into()))?,
            };
            let sm = bm["source"].as_object().ok_or_else(|| {
                AcoError::Contract(format!("bindings[{i}].source must be an object"))
            })?;
            let source = match str_of(sm, "kind")? {
                "kernarg_pointer" => {
                    only(sm, &["kind", "arg"])?;
                    BindingSource::KernargPointer {
                        arg: uint(sm, "arg")? as usize,
                    }
                }
                "kernarg_scalar" => {
                    only(sm, &["kind", "arg"])?;
                    BindingSource::KernargScalar {
                        arg: uint(sm, "arg")? as usize,
                    }
                }
                "workgroup_id" => {
                    only(sm, &["kind", "axis"])?;
                    BindingSource::WorkgroupId(axis_of(str_of(sm, "axis")?)?)
                }
                "constant" => {
                    only(sm, &["kind", "value"])?;
                    BindingSource::Constant(
                        u32::try_from(uint(sm, "value")?)
                            .map_err(|_| AcoError::Contract("constant exceeds a dword".into()))?,
                    )
                }
                o => {
                    return Err(AcoError::Contract(format!(
                        "bindings[{i}].source.kind {o:?}"
                    )))
                }
            };
            bindings.push(EntryBinding { dst, source });
        }
        let f = object(
            &m["fp"],
            "fp",
            &[
                "round32",
                "round16_64",
                "denorm32",
                "denorm16_64",
                "dx10_clamp",
                "ieee_mode",
                "fp16_overflow",
                "wgp_mode",
                "mem_ordered",
                "forward_progress",
                "wg_round_robin",
                "exceptions",
            ],
        )?;
        let fp = FpState {
            round32: u8_of(f, "round32")?,
            round16_64: u8_of(f, "round16_64")?,
            denorm32: u8_of(f, "denorm32")?,
            denorm16_64: u8_of(f, "denorm16_64")?,
            dx10_clamp: bool_of(f, "dx10_clamp")?,
            ieee_mode: bool_of(f, "ieee_mode")?,
            fp16_overflow: bool_of(f, "fp16_overflow")?,
            wgp_mode: bool_of(f, "wgp_mode")?,
            mem_ordered: bool_of(f, "mem_ordered")?,
            forward_progress: bool_of(f, "forward_progress")?,
            wg_round_robin: bool_of(f, "wg_round_robin")?,
            exceptions: u16::try_from(uint(f, "exceptions")?)
                .map_err(|_| AcoError::Contract("fp.exceptions".into()))?,
        };
        let r = object(
            &m["resources"],
            "resources",
            &[
                "lds_bytes",
                "scratch_bytes_per_lane",
                "vgprs_actual",
                "vgprs_allocated",
                "sgprs_actual",
                "sgprs_allocated",
            ],
        )?;
        let resources = RawResources {
            lds_bytes: u32_of(r, "lds_bytes")?,
            scratch_bytes_per_lane: u32_of(r, "scratch_bytes_per_lane")?,
            vgprs_actual: u16::try_from(uint(r, "vgprs_actual")?)
                .map_err(|_| AcoError::Contract("vgprs_actual".into()))?,
            vgprs_allocated: u16::try_from(uint(r, "vgprs_allocated")?)
                .map_err(|_| AcoError::Contract("vgprs_allocated".into()))?,
            sgprs_actual: u16::try_from(uint(r, "sgprs_actual")?)
                .map_err(|_| AcoError::Contract("sgprs_actual".into()))?,
            sgprs_allocated: u16::try_from(uint(r, "sgprs_allocated")?)
                .map_err(|_| AcoError::Contract("sgprs_allocated".into()))?,
        };
        let s = object(
            &m["shape"],
            "shape",
            &["block", "grid", "scalars", "odd_tail_evidence_sha256"],
        )?;
        let dims = |k: &str| -> Result<[u32; 3], AcoError> {
            let a = array(s, k)?;
            if a.len() != 3 {
                return Err(AcoError::Contract(format!(
                    "shape.{k} needs three dimensions"
                )));
            }
            let mut out = [0u32; 3];
            for (o, v) in out.iter_mut().zip(a) {
                *o = v
                    .as_u64()
                    .and_then(|x| u32::try_from(x).ok())
                    .ok_or_else(|| AcoError::Contract(format!("shape.{k} entry is not a u32")))?;
            }
            Ok(out)
        };
        let mut scalars = Vec::new();
        for (i, e) in array(s, "scalars")?.iter().enumerate() {
            let em = object(e, &format!("shape.scalars[{i}]"), &["arg", "value"])?;
            scalars.push((uint(em, "arg")? as usize, uint(em, "value")?));
        }
        let aperture = match &m["aperture"] {
            Value::String(a) if a == "global64" => Aperture::Global64,
            Value::String(a) if a == "flat_scratch" => Aperture::FlatScratch,
            Value::Object(o)
                if o.len() == 2 && o.get("kind").and_then(Value::as_str) == Some("pointer32") =>
            {
                Aperture::Pointer32 {
                    high_dword: o
                        .get("high_dword")
                        .and_then(Value::as_u64)
                        .and_then(|x| u32::try_from(x).ok())
                        .ok_or_else(|| AcoError::Contract("aperture.high_dword".into()))?,
                }
            }
            o => return Err(AcoError::Contract(format!("aperture {o}"))),
        };
        Ok(Self {
            target,
            raw_symbol: SymbolId(str_of(m, "raw_symbol")?.to_owned()),
            hip_symbol: str_of(m, "hip_symbol")?.to_owned(),
            wave,
            args,
            kernarg_size: u32_of(m, "kernarg_size")?,
            bindings,
            fp,
            resources,
            dynamic_lds_max: u32_of(m, "dynamic_lds_max")?,
            shape: ShapeDomain {
                block: dims("block")?,
                grid: dims("grid")?,
                scalars,
                odd_tail_evidence_sha256: match &s["odd_tail_evidence_sha256"] {
                    Value::Null => None,
                    _ => Some(hash_of(s, "odd_tail_evidence_sha256")?),
                },
            },
            aperture,
            artifact_sha256: hash_of(m, "artifact_sha256")?,
            rawspec_sha256: hash_of(m, "rawspec_sha256")?,
            spirv_sha256: hash_of(m, "spirv_sha256")?,
            raw_binary_sha256: hash_of(m, "raw_binary_sha256")?,
            raw_exec_sha256: hash_of(m, "raw_exec_sha256")?,
            sidecar_sha256: hash_of(m, "sidecar_sha256")?,
        })
    }
}

fn object<'a>(v: &'a Value, what: &str, keys: &[&str]) -> Result<&'a Map<String, Value>, AcoError> {
    let m = v
        .as_object()
        .ok_or_else(|| AcoError::Contract(format!("{what} must be an object")))?;
    only(m, keys).map_err(|e| AcoError::Contract(format!("{what}: {e}")))?;
    Ok(m)
}
/// Exactly `keys`, no more, no fewer.
fn only(m: &Map<String, Value>, keys: &[&str]) -> Result<(), AcoError> {
    if let Some(k) = m.keys().find(|k| !keys.contains(&k.as_str())) {
        return Err(AcoError::Contract(format!("unknown key {k:?}")));
    }
    if let Some(k) = keys.iter().find(|k| !m.contains_key(**k)) {
        return Err(AcoError::Contract(format!("missing key {k:?}")));
    }
    Ok(())
}
fn str_of<'a>(m: &'a Map<String, Value>, k: &str) -> Result<&'a str, AcoError> {
    m.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| AcoError::Contract(format!("{k} must be a string")))
}
fn uint(m: &Map<String, Value>, k: &str) -> Result<u64, AcoError> {
    m.get(k)
        .and_then(Value::as_u64)
        .ok_or_else(|| AcoError::Contract(format!("{k} must be an unsigned integer")))
}
fn u32_of(m: &Map<String, Value>, k: &str) -> Result<u32, AcoError> {
    u32::try_from(uint(m, k)?).map_err(|_| AcoError::Contract(format!("{k} exceeds u32")))
}
fn u8_of(m: &Map<String, Value>, k: &str) -> Result<u8, AcoError> {
    u8::try_from(uint(m, k)?).map_err(|_| AcoError::Contract(format!("{k} exceeds u8")))
}
fn bool_of(m: &Map<String, Value>, k: &str) -> Result<bool, AcoError> {
    m.get(k)
        .and_then(Value::as_bool)
        .ok_or_else(|| AcoError::Contract(format!("{k} must be a boolean")))
}
fn array<'a>(m: &'a Map<String, Value>, k: &str) -> Result<&'a Vec<Value>, AcoError> {
    m.get(k)
        .and_then(Value::as_array)
        .ok_or_else(|| AcoError::Contract(format!("{k} must be an array")))
}
fn axis_of(s: &str) -> Result<Axis, AcoError> {
    Ok(match s {
        "x" => Axis::X,
        "y" => Axis::Y,
        "z" => Axis::Z,
        o => return Err(AcoError::Contract(format!("axis {o:?}"))),
    })
}
fn hash_of(m: &Map<String, Value>, k: &str) -> Result<[u8; 32], AcoError> {
    let s = str_of(m, k)?;
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(AcoError::Contract(format!("{k} must be 64 hex digits")));
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex digits checked");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// A byte range of the emitted ELF and the SHA-256 of its bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodeRange {
    /// File offset in the ELF.
    pub offset: u64,
    pub len: u64,
    pub sha256: [u8; 32],
}

/// What the adapter established; each item was checked against the emitted object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdapterProof {
    pub adapter_version: &'static str,
    /// The pair the prologue loads kernargs through (`s[0:1]`, or a temporary copy).
    pub kernarg_base: RegRef,
    pub kernarg_base_copied: bool,
    /// Registers chosen from the liveness-free set (empty when no copy was needed).
    pub temps: Vec<RegRef>,
    /// Every SGPR run the prologue writes (temps and bound entry destinations).
    pub prologue_writes: Vec<RegRef>,
    /// The prologue's instruction text, in order.
    pub prologue_text: Vec<String>,
    /// Entry cells the HSA dispatch itself provides identically (no instruction emitted).
    pub hardware_entry: Vec<String>,
    pub prologue_words: usize,
    pub prologue_insts: usize,
    pub body_words: usize,
    /// The shared-analysis certificate of the emitted HSA program (no open obligation).
    pub hsa_report: CertificationReport,
}

/// The adapter's output: a native HSA code object around the untouched ACO body.
#[derive(Clone, Debug)]
pub struct AcoHsaEmission {
    pub symbol: String,
    pub elf: Vec<u8>,
    /// HIP offload bundle of `elf` (host entry `DEFAULT_HOST_TARGET`).
    pub bundle: Vec<u8>,
    pub prologue: CodeRange,
    pub body: CodeRange,
    pub proof: AdapterProof,
    pub hsa_certificate: StaticCertificate,
    pub raw_binary_sha256: [u8; 32],
    pub odd_tail_evidence_sha256: Option<[u8; 32]>,
    pub raw_exec_sha256: [u8; 32],
    pub sidecar_sha256: [u8; 32],
    pub artifact_sha256: [u8; 32],
    /// SHA-256 of the exact `shader.rawspec.json` bytes the raw spec was parsed from.
    pub rawspec_sha256: [u8; 32],
    pub spirv_sha256: [u8; 32],
    pub contract_sha256: [u8; 32],
    pub elf_sha256: [u8; 32],
    pub kernarg_size: u32,
    pub kernarg_abi_sha256: [u8; 32],
    pub block: [u32; 3],
    pub grid: [u32; 3],
    pub resources: RawResources,
}

impl AcoHsaEmission {
    /// The `aco-pm-candidate-v1` manifest the GPU oracle requires, for the loaded `image`
    /// (the ELF or the bundle). Adapter-specific bindings live under `adapter`.
    pub fn candidate_manifest(&self, image: &[u8]) -> Value {
        json!({
            "schema": CANDIDATE_SCHEMA,
            "symbol": self.symbol,
            "object_sha256": hex(&sha256(image)),
            "spirv_sha256": hex(&self.spirv_sha256),
            "aco_binary_sha256": hex(&self.raw_binary_sha256),
            "aco_exec_sha256": hex(&self.raw_exec_sha256),
            "aco_sidecar_sha256": hex(&self.sidecar_sha256),
            "body_sha256": hex(&self.body.sha256),
            "adapter_version": ADAPTER_VERSION,
            "wave_size": 32,
            "shape_support": {"fixture": true, "odd_tail": self.odd_tail_evidence_sha256.is_some(), "odd_tail_evidence_sha256": self.odd_tail_evidence_sha256.map(|h| hex(&h))},
            "block": self.block,
            "grid": self.grid,
            "kernarg_size": self.kernarg_size,
            "kernarg_abi_sha256": hex(&self.kernarg_abi_sha256),
            "vgpr_count": self.resources.vgprs_allocated,
            "sgpr_count": self.resources.sgprs_allocated,
            "lds_bytes": self.resources.lds_bytes,
            "scratch_bytes": self.resources.scratch_bytes_per_lane,
            "adapter": {
                "contract_sha256": hex(&self.contract_sha256),
                "artifact_sha256": hex(&self.artifact_sha256),
                "rawspec_sha256": hex(&self.rawspec_sha256),
                "elf_sha256": hex(&self.elf_sha256),
                "prologue_sha256": hex(&self.prologue.sha256),
                "prologue_range": {"offset": self.prologue.offset, "len": self.prologue.len},
                "body_range": {"offset": self.body.offset, "len": self.body.len},
                "kernarg_base": format!("{}", self.proof.kernarg_base),
                "prologue_text": self.proof.prologue_text,
            },
        })
    }
}

fn is_free_sgpr(free: &[RegRef], n: u16) -> bool {
    free.iter()
        .any(|r| r.kind == Kind::S && r.base <= n && n < r.base + u16::from(r.len))
}

fn sgprs(base: u16, n: u16) -> String {
    if n == 1 {
        format!("s{base}")
    } else {
        format!("s[{base}:{}]", base + n - 1)
    }
}

/// Greedy cover of the loaded dwords `(dst SGPR, kernarg byte offset)` by SMEM loads of
/// 16, 8, 4, 2 or 1 dwords: both sides contiguous, destination aligned as the ISA needs.
fn load_runs(dwords: &[(u16, u32)]) -> Vec<(u16, u32, u16)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < dwords.len() {
        let mut taken = 1usize;
        for n in [16usize, 8, 4, 2] {
            let fits = i + n <= dwords.len()
                && (0..n).all(|j| {
                    dwords[i + j].0 == dwords[i].0 + j as u16
                        && dwords[i + j].1 == dwords[i].1 + 4 * j as u32
                })
                && dwords[i].0 % (n.min(4) as u16) == 0;
            if fits {
                taken = n;
                break;
            }
        }
        runs.push((dwords[i].0, dwords[i].1, taken as u16));
        i += taken;
    }
    runs
}

/// `(file offset, size)` of the `.text` section of an ELF64 little-endian object.
fn text_section(elf: &[u8]) -> Result<(usize, usize), String> {
    let u16_at = |o: usize| elf.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |o: usize| {
        elf.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
    };
    let u64_at = |o: usize| {
        elf.get(o..o + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
    };
    let short = || "truncated ELF header".to_owned();
    let shoff = u64_at(0x28).ok_or_else(short)? as usize;
    let shentsize = u16_at(0x3a).ok_or_else(short)? as usize;
    let shnum = u16_at(0x3c).ok_or_else(short)? as usize;
    let shstrndx = u16_at(0x3e).ok_or_else(short)? as usize;
    let header = |i: usize| -> Result<(u32, usize, usize), String> {
        let at = shoff + i * shentsize;
        Ok((
            u32_at(at).ok_or_else(short)?,
            u64_at(at + 24).ok_or_else(short)? as usize,
            u64_at(at + 32).ok_or_else(short)? as usize,
        ))
    };
    let (_, str_off, str_size) = header(shstrndx)?;
    let names = elf
        .get(str_off..str_off + str_size)
        .ok_or_else(|| "section name table outside the file".to_owned())?;
    for i in 0..shnum {
        let (name, offset, size) = header(i)?;
        let tail = names
            .get(name as usize..)
            .ok_or_else(|| "section name outside the table".to_owned())?;
        if tail.starts_with(b".text\0") {
            return Ok((offset, size));
        }
    }
    Err("no .text section".into())
}

/// Adapt the certified raw kernel to an HSA code object under `contract`.
///
/// `raw` must be a [`StaticCertificate`] of the raw ACO program (domain `Raw`, frontend
/// `Aco`, no open obligation). Every contract/raw disagreement, unprovided entry fact or
/// unsupported feature is an `Err`; success means the emitted object lifts back, preserves
/// the body words exactly, and passes the shared certifier as an HSA program.
pub fn adapt_to_hsa(
    raw: &StaticCertificate,
    contract: &AcoHipContract,
) -> Result<AcoHsaEmission, AcoError> {
    contract.validate()?;
    if raw.report.domain != IdentityDomain::Raw || raw.report.frontend != Frontend::Aco {
        return Err(AcoError::Certificate(
            "the input certificate is not a raw ACO certificate".into(),
        ));
    }
    if !raw.analyzed.obligations.is_empty() || !raw.report.obligations.is_empty() {
        return Err(AcoError::Certificate(
            "the raw certificate carries open obligations".into(),
        ));
    }
    let program = &raw.analyzed.program;
    let arch = program.target.arch;
    if arch != IrArch::Gfx1201 {
        return Err(AcoError::Unsupported(format!("{arch:?}: the adapter's entry model (ttmp work-group ids, packed v0, SMEM kernarg loads) is the gfx1201 one")));
    }
    if program.target != contract.target {
        return Err(AcoError::Mismatch(format!(
            "contract target {:?} but the raw program is {:?}",
            contract.target, program.target
        )));
    }
    if contract.target.xnack != Setting::Any || contract.target.sramecc != Setting::Any {
        return Err(AcoError::Unsupported(
            "the native writer emits an object with no xnack/sramecc feature selection".into(),
        ));
    }
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol == contract.raw_symbol)
        .ok_or_else(|| {
            AcoError::Mismatch(format!(
                "raw program has no kernel {}",
                contract.raw_symbol.0
            ))
        })?;
    match &kernel.origin {
        KernelOrigin::Frontend {
            kind: Frontend::Aco,
            object_sha256,
            ..
        } => {
            if *object_sha256 != contract.raw_binary_sha256
                || raw.proof.input_sha256 != contract.raw_binary_sha256
            {
                return Err(AcoError::Mismatch(
                    "the contract's whole-binary hash is not the lifted shader.bin hash".into(),
                ));
            }
        }
        other => {
            return Err(AcoError::Certificate(format!(
                "kernel origin {other:?} is not an ACO frontend import"
            )))
        }
    }
    let Abi::Raw(raw_abi) = &kernel.abi else {
        return Err(AcoError::Certificate(
            "the kernel is not a raw (non-HSA) kernel".into(),
        ));
    };
    if kernel.wave != Wave::Wave32 || raw_abi.wave != Wave::Wave32 || contract.wave != Wave::Wave32
    {
        return Err(AcoError::Unsupported(
            "wave64: the native descriptor writer emits wave32 kernels only".into(),
        ));
    }

    // ---- contract vs raw shader config ------------------------------------------------
    if raw_abi.fp != contract.fp {
        return Err(AcoError::Mismatch(format!(
            "FP/mode state: contract {:?}, raw {:?}",
            contract.fp, raw_abi.fp
        )));
    }
    if raw_abi.resources != contract.resources {
        return Err(AcoError::Mismatch(format!(
            "resources: contract {:?}, raw {:?}",
            contract.resources, raw_abi.resources
        )));
    }
    if raw_abi.sidecar_sha256 != contract.sidecar_sha256 {
        return Err(AcoError::Mismatch(
            "sidecar hash differs from the contract".into(),
        ));
    }
    if contract.aperture != Aperture::Global64 {
        return Err(AcoError::Unsupported(format!("aperture {:?} is not representable on HIP VMM; supporting it needs a byte-oracle-gated address-lowering change, not a pointer fallback", contract.aperture)));
    }
    if contract.dynamic_lds_max != 0 {
        return Err(AcoError::Unsupported(
            "dynamic LDS: a lifted kernel carries no dynamic-LDS bound".into(),
        ));
    }
    if raw_abi.resources.scratch_bytes_per_lane != 0 {
        return Err(AcoError::Unsupported(format!(
            "scratch {} bytes per lane",
            raw_abi.resources.scratch_bytes_per_lane
        )));
    }
    if !raw_abi.relocations.is_empty() {
        return Err(AcoError::Unsupported(format!(
            "{} relocation(s): none are resolved by the adapter",
            raw_abi.relocations.len()
        )));
    }
    if raw_abi.fp.dx10_clamp || raw_abi.fp.ieee_mode {
        return Err(AcoError::Unsupported(
            "DX10_CLAMP/IEEE_MODE: the gfx12 kernel descriptor has no field that sets them".into(),
        ));
    }
    if raw_abi.fp.exceptions & !0x7f != 0 {
        return Err(AcoError::Unsupported(format!("exception enables {:#x}: bits 7-8 (address watch, memory violation) have no HSA descriptor directive", raw_abi.fp.exceptions)));
    }

    // ---- the body: typed re-encode, hashed, compared ---------------------------------
    let body_words = emit::words(kernel, arch)
        .map_err(|e| AcoError::Body(format!("typed re-encode of the raw body: {e}")))?;
    let body_bytes: Vec<u8> = body_words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let body_sha256 = sha256(&body_bytes);
    if body_sha256 != contract.raw_exec_sha256 {
        return Err(AcoError::Body(format!(
            "re-encoded body {} is not the contract's raw executable {}",
            hex(&body_sha256),
            hex(&contract.raw_exec_sha256)
        )));
    }
    match raw.proof.streams.as_slice() {
        [s] if s.stream_sha256 == body_sha256 && s.size == body_bytes.len() as u64 => {}
        other => {
            return Err(AcoError::Body(format!(
                "the round-trip proof does not bind exactly this body stream: {other:?}"
            )))
        }
    }
    for id in &kernel.body.layout {
        let inst = kernel
            .body
            .insts
            .get(*id)
            .ok_or_else(|| AcoError::Body("layout names a tombstone".into()))?;
        match inst.op.name(arch) {
            Some(name @ ("s_getpc_b64" | "s_setpc_b64" | "s_swappc_b64" | "s_call_b64")) =>
                return Err(AcoError::Unsupported(format!("{name}: PC-relative or indirect control flow needs exact symbol resolution the adapter does not do"))),
            Some(name) if name.starts_with("scratch_") => return Err(AcoError::Unsupported(format!("{name}: scratch"))),
            Some(_) => {}
            None => return Err(AcoError::Body("an instruction has no opcode name".into())),
        }
    }
    let summary = summarize(&kernel.body, arch, kernel.wave, raw_abi.resources.lds_bytes);
    if summary.uses_flat_scratch {
        return Err(AcoError::Unsupported("the body names flat scratch".into()));
    }
    if u32::from(summary.max_vgpr) > u32::from(raw_abi.resources.vgprs_allocated) {
        return Err(AcoError::Mismatch(format!(
            "body uses {} VGPRs, the shader config allocates {}",
            summary.max_vgpr, raw_abi.resources.vgprs_allocated
        )));
    }
    if summary.max_vgpr > raw_abi.resources.vgprs_actual {
        return Err(AcoError::Mismatch(format!(
            "body uses {} VGPRs, the shader config reports {} actual",
            summary.max_vgpr, raw_abi.resources.vgprs_actual
        )));
    }
    if summary.max_sgpr > raw_abi.resources.ordinary_sgprs_allocated() {
        return Err(AcoError::Mismatch(format!(
            "body uses {} ordinary SGPRs, the shader config allocates {} (physical {})",
            summary.max_sgpr,
            raw_abi.resources.ordinary_sgprs_allocated(),
            raw_abi.resources.sgprs_allocated
        )));
    }

    // ---- entry facts: every one is hardware-identical or contract-bound ---------------
    let mut hardware: Vec<String> = Vec::new();
    let mut s_facts: BTreeMap<u16, &EntryValue> = BTreeMap::new();
    let mut wg_ttmp = [false; 3];
    let mut local_ids = None;
    for (cell, fact) in raw_abi
        .entry_cells()
        .map_err(|e| AcoError::Entry(format!("malformed entry facts: {e}")))?
    {
        match (cell, &fact.value) {
            (EntryCell::Sgpr(n), v) => { s_facts.insert(n, v); }
            (EntryCell::Ttmp(9), EntryValue::WorkgroupId(Axis::X)) => { wg_ttmp[0] = true; hardware.push("ttmp9 = work-group id x".into()) }
            (EntryCell::Ttmp(7), EntryValue::WorkgroupId(Axis::Y)) => { wg_ttmp[1] = true; hardware.push("ttmp7 = work-group id y/z".into()) }
            (EntryCell::Vgpr(0), EntryValue::LocalId(fields)) => { local_ids = Some(fields); hardware.push("v0 = packed local ids".into()) }
            (EntryCell::ExecLo, EntryValue::Defined) => hardware.push("exec_lo = launch mask".into()),
            (EntryCell::Hwreg, EntryValue::Defined) => hardware.push("hwreg state".into()),
            (cell, value) => return Err(AcoError::Entry(format!("{cell:?} = {value:?}: no HSA entry state provides this and the adapter does not synthesise it"))),
        }
    }
    let mut workitem_id = 0u32;
    if let Some(fields) = local_ids {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let i = axis as usize;
            match fields.iter().find(|f| f.axis == axis) {
                Some(f) => {
                    if u32::from(f.shift) != 10 * i as u32 || f.bits != 10 {
                        return Err(AcoError::Entry(format!("local id {axis:?} is packed at shift {} width {}; HSA gfx12 v0 packs 10 bits at {}", f.shift, f.bits, 10 * i)));
                    }
                    if contract.shape.block[i] - 1 > f.max {
                        return Err(AcoError::Entry(format!(
                            "block {:?} exceeds the declared local id {axis:?} bound {}",
                            contract.shape.block, f.max
                        )));
                    }
                    workitem_id = workitem_id.max(i as u32);
                }
                None if contract.shape.block[i] != 1 => {
                    return Err(AcoError::Entry(format!(
                        "local id {axis:?} is not packed but the block has {} items along it",
                        contract.shape.block[i]
                    )));
                }
                None => {}
            }
        }
    }
    let scalar_value = |arg: usize| {
        contract
            .shape
            .scalars
            .iter()
            .find(|(a, _)| *a == arg)
            .map(|(_, v)| *v)
    };
    let mut covered: BTreeMap<u16, usize> = BTreeMap::new();
    for (i, b) in contract.bindings.iter().enumerate() {
        for n in b.dst.base..b.dst.base + u16::from(b.dst.len) {
            covered.insert(n, i);
        }
    }
    let live = Liveness::analyze(&kernel.body);
    let first = *kernel
        .body
        .layout
        .first()
        .ok_or_else(|| AcoError::Body("empty body".into()))?;
    let live_at_entry = live
        .live_before(first)
        .ok_or_else(|| AcoError::Body("no liveness at the first instruction".into()))?
        .clone();
    let mut unbound_dead: Vec<u16> = Vec::new();
    let mut wg_sgpr = [false; 3];
    for (n, value) in &s_facts {
        let Some(&bi) = covered.get(n) else {
            // An entry location the contract does not initialise (e.g. RADV's ring_offsets in s0:1): HSA leaves whatever it
            // has there (the kernarg pointer), which is sound only if no path reads it before the body overwrites it.
            // Typed backward liveness at the first instruction decides; no fabricated value is written.
            if live_at_entry.contains(&Loc::S(*n)) {
                return Err(AcoError::Entry(format!("s{n} is defined at entry ({value:?}) and read by the body before any write, but no contract binding initialises it")));
            }
            unbound_dead.push(*n);
            continue;
        };
        let source = contract.bindings[bi].source;
        let ok = match (*value, source) {
            (
                EntryValue::Defined,
                BindingSource::KernargPointer { .. } | BindingSource::KernargScalar { .. },
            ) => true,
            (EntryValue::Bounded { min, max }, BindingSource::KernargScalar { arg }) => {
                scalar_value(arg).is_some_and(|v| u64::from(*min) <= v && v <= u64::from(*max))
            }
            (EntryValue::Exact(x), BindingSource::KernargScalar { arg }) => {
                scalar_value(arg) == Some(u64::from(*x))
            }
            (EntryValue::WorkgroupId(a), BindingSource::WorkgroupId(b)) if *a == b => {
                wg_sgpr[b as usize] = true;
                true
            }
            (EntryValue::Exact(x), BindingSource::Constant(c)) => *x == c,
            _ => false,
        };
        if !ok {
            return Err(AcoError::Entry(format!(
                "s{n}: entry fact {value:?} is not satisfied by binding {source:?}"
            )));
        }
    }
    for n in &unbound_dead {
        hardware.push(format!(
            "s{n} = declared entry value, unbound and dead (overwritten before any read)"
        ))
    }
    if let Some(n) = covered.keys().find(|n| !s_facts.contains_key(*n)) {
        return Err(AcoError::Entry(format!(
            "a binding writes s{n}, which the raw entry facts do not define"
        )));
    }
    let mut need_wg = [
        wg_ttmp[0] || wg_sgpr[0],
        wg_ttmp[1] || wg_sgpr[1],
        wg_sgpr[2],
    ];
    if wg_ttmp[1] && contract.shape.grid[2] > 1 {
        need_wg[2] = true
    }

    // ---- prologue plan, temporaries from liveness ------------------------------------
    let free = live
        .free_at(
            &kernel.body,
            LivePoint {
                block: BlockId(0),
                before: Some(first),
            },
        )
        .0;
    let copy_needed = covered.contains_key(&0) || covered.contains_key(&1);
    let (kernarg_base, temps) = if copy_needed {
        let t = (2u16..105).step_by(2)
            .find(|&t| !covered.contains_key(&t) && !covered.contains_key(&(t + 1)) && is_free_sgpr(&free, t) && is_free_sgpr(&free, t + 1))
            .ok_or_else(|| AcoError::Prologue("no even SGPR pair is both free at the body entry (liveness) and unbound by the contract to hold the kernarg pointer".into()))?;
        let pair = RegRef {
            kind: Kind::S,
            base: t,
            len: 2,
        };
        (pair, vec![pair])
    } else {
        (
            RegRef {
                kind: Kind::S,
                base: 0,
                len: 2,
            },
            Vec::new(),
        )
    };
    let mut lines: Vec<String> = Vec::new();
    let mut writes: Vec<RegRef> = temps.clone();
    if copy_needed {
        lines.push(format!("s_mov_b64 {}, s[0:1]", sgprs(kernarg_base.base, 2)))
    }
    let mut dwords: Vec<(u16, u32)> = Vec::new();
    for b in &contract.bindings {
        let arg = match b.source {
            BindingSource::KernargPointer { arg } | BindingSource::KernargScalar { arg } => arg,
            _ => continue,
        };
        let a = &contract.args[arg];
        for j in 0..u32::from(b.dst.len) {
            dwords.push((b.dst.base + j as u16, a.offset + 4 * j))
        }
    }
    dwords.sort_unstable();
    let loads = !dwords.is_empty();
    for (dst, offset, n) in load_runs(&dwords) {
        lines.push(format!(
            "s_load_b{} {}, s[{}:{}], {:#x}",
            32 * u32::from(n),
            sgprs(dst, n),
            kernarg_base.base,
            kernarg_base.base + 1,
            offset
        ));
    }
    for b in &contract.bindings {
        let s = b.dst.base;
        match b.source {
            BindingSource::WorkgroupId(Axis::X) => lines.push(format!("s_mov_b32 s{s}, ttmp9")),
            BindingSource::WorkgroupId(Axis::Y) => {
                lines.push(format!("s_and_b32 s{s}, ttmp7, 0xffff"))
            }
            BindingSource::WorkgroupId(Axis::Z) => {
                lines.push(format!("s_lshr_b32 s{s}, ttmp7, 16"))
            }
            BindingSource::Constant(c) => lines.push(format!("s_mov_b32 s{s}, {c:#x}")),
            BindingSource::KernargPointer { .. } | BindingSource::KernargScalar { .. } => {}
        }
        writes.push(b.dst);
    }
    if loads {
        lines.push("s_wait_kmcnt 0x0".into())
    }
    let mut prologue: Vec<u32> = Vec::new();
    for line in &lines {
        let inst = parse_line(line, IrArch::Gfx1201)
            .map_err(|e| AcoError::Prologue(format!("{line}: {e}")))?;
        let words = gfx12::encode_for(IrArch::Gfx1201, &inst)
            .map_err(|e| AcoError::Prologue(format!("{line}: {e}")))?;
        prologue.extend_from_slice(&words);
    }
    let prologue_sgpr_hi = writes
        .iter()
        .map(|r| r.base + u16::from(r.len))
        .max()
        .unwrap_or(0);

    // ---- descriptor and metadata, from the shader config and the contract -------------
    // `ns` counts the ordinary file (s0..s105) only; the driver's physical 107/108 (VCC_LO/HI) stays in metadata sgpr_count.
    let ns = summary
        .max_sgpr
        .max(prologue_sgpr_hi)
        .max(raw_abi.resources.ordinary_sgprs_actual());
    if ns > RawResources::ORDINARY_SGPRS {
        return Err(AcoError::Mismatch(format!(
            "{ns} ordinary SGPRs exceed the {} a wave can address",
            RawResources::ORDINARY_SGPRS
        )));
    }
    let nv = raw_abi.resources.vgprs_allocated.max(1);
    let fp = &raw_abi.fp;
    let mut d: Vec<(&str, u64)> = vec![
        (
            "group_segment_fixed_size",
            u64::from(raw_abi.resources.lds_bytes),
        ),
        ("private_segment_fixed_size", 0),
        ("kernarg_size", u64::from(contract.kernarg_size)),
        ("user_sgpr_count", 2),
        ("user_sgpr_dispatch_ptr", 0),
        ("user_sgpr_queue_ptr", 0),
        ("user_sgpr_kernarg_segment_ptr", 1),
        ("user_sgpr_dispatch_id", 0),
        ("user_sgpr_private_segment_size", 0),
        ("wavefront_size32", 1),
        ("uses_dynamic_stack", 0),
        ("enable_private_segment", 0),
        ("system_sgpr_workgroup_id_x", u64::from(need_wg[0])),
        ("system_sgpr_workgroup_id_y", u64::from(need_wg[1])),
        ("system_sgpr_workgroup_id_z", u64::from(need_wg[2])),
        ("system_sgpr_workgroup_info", 0),
        ("system_vgpr_workitem_id", u64::from(workitem_id)),
        ("next_free_vgpr", u64::from(nv)),
        ("next_free_sgpr", u64::from(ns)),
        ("reserve_vcc", u64::from(summary.uses_vcc)),
        ("float_round_mode_32", u64::from(fp.round32)),
        ("float_round_mode_16_64", u64::from(fp.round16_64)),
        ("float_denorm_mode_32", u64::from(fp.denorm32)),
        ("float_denorm_mode_16_64", u64::from(fp.denorm16_64)),
        ("fp16_overflow", u64::from(fp.fp16_overflow)),
        ("workgroup_processor_mode", u64::from(fp.wgp_mode)),
        ("memory_ordered", u64::from(fp.mem_ordered)),
        ("forward_progress", u64::from(fp.forward_progress)),
        ("round_robin_scheduling", u64::from(fp.wg_round_robin)),
    ];
    for (bit, name) in [
        "fp_ieee_invalid_op",
        "fp_denorm_src",
        "fp_ieee_div_zero",
        "fp_ieee_overflow",
        "fp_ieee_underflow",
        "fp_ieee_inexact",
        "int_div_zero",
    ]
    .iter()
    .enumerate()
    {
        d.push((*name, u64::from(fp.exceptions >> bit & 1)));
    }
    let directives: Vec<(String, String)> = d
        .iter()
        .map(|(n, v)| {
            let name = if n.starts_with("fp_") || *n == "int_div_zero" {
                format!(".amdhsa_exception_{n}")
            } else {
                format!(".amdhsa_{n}")
            };
            (name, v.to_string())
        })
        .collect();
    let symbol = contract.hip_symbol.as_str();
    let mut yaml = String::from("amdhsa.kernels:\n  - .args:\n");
    for a in &contract.args {
        let (access, space, value_kind) = match &a.kind {
            HipArgKind::GlobalPointer { access, .. } => {
                (access.map(Access::name), Some("global"), "global_buffer")
            }
            HipArgKind::ByValue => (None, None, "by_value"),
        };
        let mut lead = "      - ";
        for (key, value) in [(".actual_access", access), (".address_space", space)] {
            if let Some(v) = value {
                yaml.push_str(&format!("{lead}{key}: {v}\n"));
                lead = "        "
            }
        }
        if lead == "      - " {
            yaml.push_str("      - \n")
        }
        yaml.push_str(&format!("        .name: {}\n        .offset: {}\n        .size: {}\n        .value_kind: {value_kind}\n", a.name, a.offset, a.size));
    }
    let threads: u32 = contract.shape.block.iter().product();
    yaml.push_str(&format!(
        "    .group_segment_fixed_size: {}\n    .kernarg_segment_align: 8\n    .kernarg_segment_size: {}\n    .max_flat_workgroup_size: {threads}\n    .name: {symbol}\n    .private_segment_fixed_size: 0\n    .sgpr_count: {}\n    .sgpr_spill_count: 0\n    .symbol: {symbol}.kd\n    .uniform_work_group_size: 1\n    .uses_dynamic_stack: false\n    .vgpr_count: {nv}\n    .vgpr_spill_count: 0\n    .wavefront_size: 32\n    .workgroup_processor_mode: {}\namdhsa.target: amdgcn-amd-amdhsa--gfx1201\namdhsa.version: [1, 2]\n",
        raw_abi.resources.lds_bytes, contract.kernarg_size,
        // Physical basis: the driver's own count (ordinary file plus VCC_LO/HI) is the floor; a prologue or VCC use above it raises it.
        u32::from(raw_abi.resources.sgprs_actual).max(u32::from(ns) + if summary.uses_vcc { 2 } else { 0 }),
        u8::from(fp.wgp_mode)));

    // ---- link, find the ranges, lift back, certify -----------------------------------
    let mut words = prologue.clone();
    words.extend_from_slice(&body_words);
    let elf = native::link_encoded(
        &[EncodedKernel {
            symbol,
            words: &words,
            directives: &directives,
        }],
        &yaml,
        crate::Arch::Gfx1201,
    )
    .map_err(AcoError::Emit)?;
    let (text_offset, text_size) =
        text_section(&elf).map_err(|e| AcoError::Emit(format!("locating the code: {e}")))?;
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if text_size < code.len()
        || elf.get(text_offset..text_offset + code.len()) != Some(code.as_slice())
    {
        return Err(AcoError::Emit(
            "the object's .text does not begin with prologue ++ body".into(),
        ));
    }
    let prologue_len = 4 * prologue.len();
    let prologue_range = CodeRange {
        offset: text_offset as u64,
        len: prologue_len as u64,
        sha256: sha256(&code[..prologue_len]),
    };
    let body_range = CodeRange {
        offset: (text_offset + prologue_len) as u64,
        len: body_bytes.len() as u64,
        sha256: sha256(&elf[text_offset + prologue_len..text_offset + code.len()]),
    };
    if body_range.sha256 != body_sha256 {
        return Err(AcoError::Emit(
            "the extracted body differs from the raw body".into(),
        ));
    }

    let relifted = lift_object(
        &elf,
        Options {
            frontend: Frontend::Aco,
        },
    )
    .map_err(|e| AcoError::Relift(e.to_string()))?;
    let [back] = relifted.program.kernels.as_slice() else {
        return Err(AcoError::Relift(
            "the emitted object does not hold exactly one kernel".into(),
        ));
    };
    if back.symbol.0 != symbol {
        return Err(AcoError::Relift(format!(
            "relifted symbol {} is not {symbol}",
            back.symbol.0
        )));
    }
    // The HSA identity domain: same arch/xnack/sramecc as the raw target, code object v6.
    let hsa_target = Target {
        arch: contract.target.arch,
        xnack: contract.target.xnack,
        sramecc: contract.target.sramecc,
        abi_version: 6,
    };
    if relifted.program.target != hsa_target {
        return Err(AcoError::Relift(format!(
            "the emitted object's target {:?} is not the HSA target {hsa_target:?}",
            relifted.program.target
        )));
    }
    let back_words = emit::words(back, arch)
        .map_err(|e| AcoError::Relift(format!("re-encode of the relifted kernel: {e}")))?;
    if back_words != words {
        return Err(AcoError::Relift(
            "the relifted stream is not prologue ++ body".into(),
        ));
    }
    let shift = lines.len();
    let raw_ends: BTreeSet<usize> = kernel
        .body
        .blocks
        .iter()
        .map(|b| b.range.1 + shift)
        .collect();
    let mut back_ends: BTreeSet<usize> = back.body.blocks.iter().map(|b| b.range.1).collect();
    if shift > 0 {
        back_ends.remove(&shift);
    }
    if raw_ends != back_ends {
        return Err(AcoError::Relift(format!(
            "block boundaries moved: raw {raw_ends:?} (shifted by {shift}), emitted {back_ends:?}"
        )));
    }
    let launch = LaunchContract {
        grid: contract.shape.grid,
        block: contract.shape.block,
        dynamic_lds_max: 0,
        bound_inputs: contract
            .args
            .iter()
            .filter_map(|a| match &a.kind {
                HipArgKind::GlobalPointer { extent_bytes, .. } => {
                    Some((a.name.clone(), *extent_bytes))
                }
                HipArgKind::ByValue => None,
            })
            .collect(),
    };
    let hsa_certificate = certify_program(relifted, &SymbolId(symbol.to_owned()), &launch)
        .map_err(AcoError::Certify)?;

    let proof = AdapterProof {
        adapter_version: ADAPTER_VERSION,
        kernarg_base,
        kernarg_base_copied: copy_needed,
        temps,
        prologue_writes: writes,
        prologue_text: lines,
        hardware_entry: hardware,
        prologue_words: prologue.len(),
        prologue_insts: shift,
        body_words: body_words.len(),
        hsa_report: hsa_certificate.report.clone(),
    };
    let bundle = native::bundle(&elf, crate::Arch::Gfx1201, native::DEFAULT_HOST_TARGET);
    Ok(AcoHsaEmission {
        symbol: symbol.to_owned(),
        elf_sha256: sha256(&elf),
        elf,
        bundle,
        prologue: prologue_range,
        body: body_range,
        proof,
        hsa_certificate,
        raw_exec_sha256: contract.raw_exec_sha256,
        sidecar_sha256: contract.sidecar_sha256,
        artifact_sha256: contract.artifact_sha256,
        rawspec_sha256: contract.rawspec_sha256,
        spirv_sha256: contract.spirv_sha256,
        contract_sha256: contract.sha256(),
        kernarg_size: contract.kernarg_size,
        kernarg_abi_sha256: sha256(&contract.kernarg_template()),
        raw_binary_sha256: contract.raw_binary_sha256,
        odd_tail_evidence_sha256: contract.shape.odd_tail_evidence_sha256,
        block: contract.shape.block,
        grid: contract.shape.grid,
        resources: raw_abi.resources,
    })
}

// ---------------------------------------------------------------------------
// Static import chain: artifact.json -> every file -> contract
// ---------------------------------------------------------------------------

/// Schema of the producer's `artifact.json`.
pub const ARTIFACT_SCHEMA: &str = "pm-aco-artifact-v1";
/// Files an artifact must list and carry (all others it lists are hashed too).
const REQUIRED_FILES: [&str; 5] = [
    "shader.spv",
    "shader.bin",
    "shader.abi.json",
    "shader.rawspec.json",
    "shader.aco.dump",
];

/// What `verify_artifact_dir` proved: every hash below was recomputed from the bytes on disk and
/// equals the one `artifact.json` records.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactBinding {
    /// SHA-256 of the exact `artifact.json` bytes.
    pub artifact_sha256: [u8; 32],
    pub spirv_sha256: [u8; 32],
    /// The whole `shader.bin` (executable range plus trailing data).
    pub binary_sha256: [u8; 32],
    pub sidecar_sha256: [u8; 32],
    pub rawspec_sha256: [u8; 32],
    pub dump_sha256: [u8; 32],
    pub exec_offset: u64,
    pub exec_len: u64,
    pub exec_sha256: [u8; 32],
    pub compiler_sha256: [u8; 32],
    pub mesa_patch_sha256: [u8; 32],
    pub mesa_commit: String,
    /// Number of files `artifact.json` lists (all verified).
    pub files: usize,
}

fn parse_hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

fn list_files(
    root: &std::path::Path,
    dir: &std::path::Path,
    out: &mut BTreeSet<String>,
) -> Result<(), AcoError> {
    let bad = |m: String| AcoError::Artifact(m);
    for entry in std::fs::read_dir(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))? {
        let entry = entry.map_err(|e| bad(format!("{}: {e}", dir.display())))?;
        let kind = entry
            .file_type()
            .map_err(|e| bad(format!("{}: {e}", entry.path().display())))?;
        let path = entry.path();
        if kind.is_dir() {
            list_files(root, &path, out)?;
        } else if kind.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|_| bad("file outside the artifact directory".into()))?;
            out.insert(
                rel.components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/"),
            );
        } else {
            return Err(bad(format!(
                "{} is neither a regular file nor a directory",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Verify a producer artifact directory against its `artifact.json`.
///
/// The recorded file set must equal the files on disk (no listed file missing or altered, no
/// unlisted file present), the five mandatory files must be listed, the provenance hashes must be
/// present, and the recorded executable range must be the hashed range of `shader.bin`. So an
/// edited rawspec (entry facts, FP, resources) cannot ride on an unchanged sidecar hash.
pub fn verify_artifact_dir(dir: &std::path::Path) -> Result<ArtifactBinding, AcoError> {
    let bad = |m: String| AcoError::Artifact(m);
    let artifact_bytes =
        std::fs::read(dir.join("artifact.json")).map_err(|e| bad(format!("artifact.json: {e}")))?;
    let doc: Value =
        serde_json::from_slice(&artifact_bytes).map_err(|e| bad(format!("artifact.json: {e}")))?;
    let top = doc
        .as_object()
        .ok_or_else(|| bad("artifact.json is not an object".into()))?;
    if top.get("schema").and_then(Value::as_str) != Some(ARTIFACT_SCHEMA) {
        return Err(bad(format!("schema is not {ARTIFACT_SCHEMA}")));
    }
    let files = top
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("artifact.json has no files object".into()))?;

    let mut on_disk = BTreeSet::new();
    list_files(dir, dir, &mut on_disk)?;
    on_disk.remove("artifact.json");
    if let Some(name) = on_disk.iter().find(|n| !files.contains_key(*n)) {
        return Err(bad(format!(
            "{name} is in the artifact directory but artifact.json does not list it"
        )));
    }
    let mut hashes: BTreeMap<&str, [u8; 32]> = BTreeMap::new();
    for (name, entry) in files {
        if name.is_empty()
            || name.starts_with('/')
            || name
                .split('/')
                .any(|c| c.is_empty() || c == "." || c == "..")
        {
            return Err(bad(format!(
                "{name:?} is not a relative path inside the artifact directory"
            )));
        }
        let want = entry
            .get("sha256")
            .and_then(Value::as_str)
            .and_then(parse_hex32)
            .ok_or_else(|| bad(format!("{name}: artifact.json records no 64-hex sha256")))?;
        let bytes = std::fs::read(dir.join(name)).map_err(|e| bad(format!("{name}: {e}")))?;
        let actual = sha256(&bytes);
        if actual != want {
            return Err(bad(format!(
                "{name}: SHA-256 {} differs from artifact.json's {}",
                hex(&actual),
                hex(&want)
            )));
        }
        hashes.insert(name.as_str(), want);
    }
    for name in REQUIRED_FILES {
        if !hashes.contains_key(name) {
            return Err(bad(format!("artifact.json lists no {name}")));
        }
    }
    let prov = top
        .get("provenance")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("artifact.json has no provenance object".into()))?;
    let prov_hash = |key: &str| {
        prov.get(key)
            .and_then(Value::as_str)
            .and_then(parse_hex32)
            .ok_or_else(|| bad(format!("provenance.{key} is missing or not 64 hex digits")))
    };
    let mesa_commit = prov
        .get("mesa_commit")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad("provenance.mesa_commit is missing".into()))?
        .to_owned();

    let binary = top
        .get("binary")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("artifact.json has no binary object".into()))?;
    let binary_sha256 = hashes["shader.bin"];
    if binary
        .get("sha256")
        .and_then(Value::as_str)
        .and_then(parse_hex32)
        != Some(binary_sha256)
    {
        return Err(bad("binary.sha256 is not the shader.bin hash".into()));
    }
    let exec = binary
        .get("executable")
        .and_then(Value::as_object)
        .ok_or_else(|| bad("binary.executable is missing".into()))?;
    let (Some(offset), Some(len), Some(exec_sha256)) = (
        exec.get("offset").and_then(Value::as_u64),
        exec.get("len").and_then(Value::as_u64),
        exec.get("sha256")
            .and_then(Value::as_str)
            .and_then(parse_hex32),
    ) else {
        return Err(bad("binary.executable needs offset, len and sha256".into()));
    };
    let bin = std::fs::read(dir.join("shader.bin")).map_err(|e| bad(format!("shader.bin: {e}")))?;
    let range = offset
        .checked_add(len)
        .and_then(|end| Some((usize::try_from(offset).ok()?, usize::try_from(end).ok()?)))
        .filter(|&(_, end)| end <= bin.len())
        .ok_or_else(|| {
            bad(format!(
                "binary.executable {offset:#x}+{len:#x} is outside the {}-byte shader.bin",
                bin.len()
            ))
        })?;
    if sha256(&bin[range.0..range.1]) != exec_sha256 {
        return Err(bad(
            "binary.executable.sha256 is not the hash of that range of shader.bin".into(),
        ));
    }

    Ok(ArtifactBinding {
        artifact_sha256: sha256(&artifact_bytes),
        spirv_sha256: hashes["shader.spv"],
        binary_sha256,
        sidecar_sha256: hashes["shader.abi.json"],
        rawspec_sha256: hashes["shader.rawspec.json"],
        dump_sha256: hashes["shader.aco.dump"],
        exec_offset: offset,
        exec_len: len,
        exec_sha256,
        compiler_sha256: prov_hash("compiler_sha256")?,
        mesa_patch_sha256: prov_hash("mesa_patch_sha256")?,
        mesa_commit,
        files: hashes.len(),
    })
}

impl ArtifactBinding {
    /// The raw spec the lifter was given must be the artifact's: same whole binary, executable
    /// range and hash, and sidecar.
    pub fn check_raw_spec(
        &self,
        spec: &peacemaker_ir::inst::RawKernelSpec,
    ) -> Result<(), AcoError> {
        let bad = |m: String| Err(AcoError::Artifact(m));
        if spec.binary_sha256 != self.binary_sha256 {
            return bad("the raw spec's binary hash is not the artifact's shader.bin".into());
        }
        if spec.extent.offset != self.exec_offset || spec.extent.len != self.exec_len {
            return bad("the raw spec's executable extent is not the artifact's".into());
        }
        if spec.exec_sha256 != self.exec_sha256 {
            return bad("the raw spec's executable hash is not the artifact's".into());
        }
        if spec.sidecar_sha256 != self.sidecar_sha256 {
            return bad("the raw spec's sidecar hash is not the artifact's shader.abi.json".into());
        }
        Ok(())
    }

    /// The contract's identity fields must be the artifact's.
    pub fn check_contract(&self, contract: &AcoHipContract) -> Result<(), AcoError> {
        let bad = |m: &str| {
            Err(AcoError::Artifact(format!(
                "the contract's {m} is not the artifact's"
            )))
        };
        if contract.artifact_sha256 != self.artifact_sha256 {
            return bad("artifact.json hash");
        }
        if contract.rawspec_sha256 != self.rawspec_sha256 {
            return bad("shader.rawspec.json hash");
        }
        if contract.spirv_sha256 != self.spirv_sha256 {
            return bad("shader.spv hash");
        }
        if contract.raw_binary_sha256 != self.binary_sha256 {
            return bad("shader.bin hash");
        }
        if contract.raw_exec_sha256 != self.exec_sha256 {
            return bad("executable hash");
        }
        if contract.sidecar_sha256 != self.sidecar_sha256 {
            return bad("shader.abi.json hash");
        }
        Ok(())
    }
}
