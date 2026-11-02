// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raw (non-HSA) executable lift: a Mesa/ACO shader binary into the shared typed core.
//!
//! [`lift_raw`] takes the producer's binary and a core-owned [`RawKernelSpec`] (target, symbol,
//! wave, executable extent, resources, FP state, relocations, explicit entry facts, binary /
//! executable / sidecar hashes). The executable range goes through the same decode as
//! [`crate::lift_object`] (table-driven gfx11/gfx12 codec, word-for-word re-encode from typed
//! fields, CFG from encoded branch offsets, clause/delay window checks); there is no opaque
//! word passthrough and no text path. The result is a `Lifted<Program>` with `Abi::Raw`, no
//! source envelope (so [`crate::emit::module`] refuses it until an HSA adapter exists) and a
//! `RoundTripProof` bound to the binary, the executable stream and the sidecar. Anything the
//! contract or the code does not cover is a `LiftError`; there is no partial success.

use std::collections::BTreeSet;

use peacemaker_ir::codec::{gfx11, gfx12};
use peacemaker_ir::edit::entry_undefined;
use peacemaker_ir::inst::{
    Abi, Arch, Axis, EntryFact, EntryLoc, EntryValue, ExecExtent, FpState, Frontend, Kernel,
    KernelOrigin, LocalIdField, Program, RawKernelSpec, RawResources, Relocation, Setting,
    SymbolId, Target, Wave,
};
use peacemaker_ir::operand::Special;
use peacemaker_ir::passes::{cfg::dwords_of, resources::summarize};
use peacemaker_ir::reg::{Kind, RegRef};
use peacemaker_ir::state::{LiftReport, Lifted, RoundTripProof, StreamProof};
use serde_json::{Map, Value};

use crate::{arch_of_name, decode_body, emit, rejected, sha256, LiftError, Options, Rule};

/// Lift `spec`'s executable range of `bytes` into a one-kernel raw `Program`.
///
/// Rejects (never partially succeeds) when: the frontend is not [`Frontend::Aco`]; the
/// declaration is inconsistent ([`Rule::Spec`]: malformed or overlapping entry facts, register
/// counts, extent outside the binary, binary or executable hash mismatch, code using more
/// registers than declared, a relocation not on an instruction literal); a word does not decode
/// or does not re-encode identically from typed fields ([`Rule::Decode`],
/// [`Rule::StreamIdentity`]); a branch leaves the stream or lands mid-instruction
/// ([`Rule::BranchTarget`]); a path lacks `s_endpgm` ([`Rule::ControlFlow`]); a clause/delay
/// window is malformed ([`Rule::Window`]); or the code reads a location the entry facts leave
/// undefined ([`Rule::Entry`], e.g. EXEC, VCC, SCC, M0 never declared).
///
/// Instruction provenance carries the binary hash, the byte offset of each instruction in the
/// binary (`prov.pc`) and its words; neither is read back: `emit::bytes(kernel, arch)` reproduces
/// the executable range from typed fields alone.
pub fn lift_raw(
    bytes: &[u8],
    spec: &RawKernelSpec,
    options: Options,
) -> Result<Lifted<Program>, LiftError> {
    let name = spec.symbol.0.as_str();
    let spec_error = |reason: String| rejected(name, spec.extent.offset, Rule::Spec, reason);
    if options.frontend != Frontend::Aco {
        return Err(spec_error(format!(
            "the raw importer is the ACO frontend, not {:?}",
            options.frontend
        )));
    }
    spec.validate().map_err(|e| spec_error(e.to_string()))?;
    let arch = spec.target.arch;
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(rejected(
            name,
            spec.extent.offset,
            Rule::UnsupportedArch,
            format!("{arch:?} has no opcode table"),
        ));
    }
    if sha256(bytes) != spec.binary_sha256 {
        return Err(spec_error(format!(
            "binary SHA-256 {} differs from the declared {}",
            hex(&sha256(bytes)),
            hex(&spec.binary_sha256)
        )));
    }
    let range = spec
        .extent
        .offset
        .checked_add(spec.extent.len)
        .and_then(|end| {
            Some((
                usize::try_from(spec.extent.offset).ok()?,
                usize::try_from(end).ok()?,
            ))
        })
        .filter(|&(_, end)| end <= bytes.len())
        .ok_or_else(|| {
            spec_error(format!(
                "executable extent {:#x}+{:#x} is outside the {}-byte binary",
                spec.extent.offset,
                spec.extent.len,
                bytes.len()
            ))
        })?;
    let code = &bytes[range.0..range.1];
    if sha256(code) != spec.exec_sha256 {
        return Err(spec_error(format!(
            "executable SHA-256 {} differs from the declared {}",
            hex(&sha256(code)),
            hex(&spec.exec_sha256)
        )));
    }
    let body = decode_body(
        name,
        code,
        spec.extent.offset,
        arch,
        spec.binary_sha256,
        Frontend::Aco,
    )?;

    // Relocations may only name an instruction literal dword.
    let mut literals = BTreeSet::new();
    for &id in &body.layout {
        let inst = body
            .insts
            .get(id)
            .expect("decoded layout references live instructions");
        let pc =
            u64::from(inst.prov.pc.expect("decode records the byte offset")) - spec.extent.offset;
        if inst.literal.is_some() {
            literals.insert(pc + 4 * (dwords_of(inst, arch) as u64 - 1));
        }
    }
    if let Some(reloc) = spec
        .relocations
        .iter()
        .find(|r| !literals.contains(&r.offset))
    {
        return Err(rejected(name, spec.extent.offset + reloc.offset, Rule::Spec,
            format!("relocation against {:?} at executable offset {:#x} is not an instruction literal dword", reloc.symbol, reloc.offset)));
    }

    // The code must fit the declared register counts (a count the code outruns is a false declaration).
    let used = summarize(&body, arch, spec.wave, spec.resources.lds_bytes);
    if used.max_vgpr > spec.resources.vgprs_actual {
        return Err(spec_error(format!(
            "code uses {} VGPRs, the declaration says {} actual",
            used.max_vgpr, spec.resources.vgprs_actual
        )));
    }
    // Ordinary SGPRs are bound by the ordinary part of the count. A producer may report the
    // driver's physical count (VCC_LO/VCC_HI as 106/107) or the ordinary count; either way s106
    // never becomes an ordinary register. VCC itself is a special with its own entry facts.
    if used.max_sgpr > spec.resources.ordinary_sgprs_actual() {
        return Err(spec_error(format!(
            "code uses {} ordinary SGPRs, the declaration allows {} (of {} counted)",
            used.max_sgpr,
            spec.resources.ordinary_sgprs_actual(),
            spec.resources.sgprs_actual
        )));
    }

    let kernel = Kernel {
        symbol: spec.symbol.clone(),
        wave: spec.wave,
        abi: Abi::Raw(spec.abi()),
        body,
        origin: KernelOrigin::Frontend {
            kind: Frontend::Aco,
            object_sha256: spec.binary_sha256,
            entry_va: spec.extent.offset,
            size: spec.extent.len,
        },
    };
    let encoded = emit::bytes(&kernel, arch)
        .map_err(|e| rejected(name, spec.extent.offset, Rule::StreamIdentity, e))?;
    if encoded != code {
        let at = encoded
            .iter()
            .zip(code)
            .position(|(a, b)| a != b)
            .unwrap_or(encoded.len().min(code.len()));
        return Err(rejected(
            name,
            spec.extent.offset + at as u64,
            Rule::StreamIdentity,
            "label-lowered stream differs from the input",
        ));
    }
    let program = Program {
        target: spec.target,
        kernels: vec![kernel],
        source: None,
    };
    program
        .validate()
        .map_err(|e| rejected(name, spec.extent.offset, Rule::Program, e))?;

    let undefined = entry_undefined(&program, &spec.symbol)
        .map_err(|e| rejected(name, spec.extent.offset, Rule::Entry, e))?;
    if let Some(first) = undefined.first() {
        let kernel = &program.kernels[0];
        let at = kernel
            .body
            .insts
            .get(first.insts[0])
            .and_then(|inst| inst.prov.pc)
            .map_or(spec.extent.offset, u64::from);
        let all: Vec<&str> = undefined.iter().map(|o| o.text.as_str()).collect();
        return Err(rejected(
            name,
            at,
            Rule::Entry,
            format!(
                "{} instruction(s) read undeclared entry state; first: {}",
                all.len(),
                all[0]
            ),
        ));
    }
    Ok(Lifted {
        program,
        proof: RoundTripProof {
            input_sha256: spec.binary_sha256,
            tools: Vec::new(),
            streams: vec![StreamProof {
                entry: spec.extent.offset,
                size: spec.extent.len,
                stream_sha256: spec.exec_sha256,
            }],
            sidecar_sha256: Some(spec.sidecar_sha256),
        },
        report: LiftReport::default(),
    })
}

/// The producer sidecar bytes must hash to the declared `sidecar_sha256` (a tampered or
/// substituted sidecar is a [`Rule::Spec`] rejection).
pub fn verify_sidecar(sidecar: &[u8], spec: &RawKernelSpec) -> Result<(), LiftError> {
    let actual = sha256(sidecar);
    if actual != spec.sidecar_sha256 {
        return Err(rejected(
            &spec.symbol.0,
            0,
            Rule::Spec,
            format!(
                "sidecar SHA-256 {} differs from the declared {}",
                hex(&actual),
                hex(&spec.sidecar_sha256)
            ),
        ));
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One decoded instruction of a [`scan_stream`] census.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanInst {
    pub offset: u64,
    pub words: usize,
    pub opcode: String,
    pub form: String,
}

/// One undecodable or non-identical word of a [`scan_stream`] census.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanFailure {
    pub offset: u64,
    pub word: u32,
    pub rule: Rule,
    pub reason: String,
}

/// Result of decoding a stream to the end without stopping at the first failure.
/// `failures[0]` is authoritative: it is the first word the codec cannot decode or re-encode,
/// exactly what [`lift_raw`] reports. Later failures are best effort: instruction length is
/// unknown after a failure, so the scan resynchronises one dword later and a later entry can
/// be a literal or trailing word of an undecoded instruction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StreamScan {
    pub words: usize,
    pub decoded: Vec<ScanInst>,
    pub failures: Vec<ScanFailure>,
}

/// Census of `code` under `arch`'s codec: instructions decoded and re-encoded word for word,
/// and every failing word. Offsets are byte offsets inside `code`.
pub fn scan_stream(code: &[u8], arch: Arch) -> StreamScan {
    let words: Vec<u32> = code
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("dword")))
        .collect();
    let mut scan = StreamScan {
        words: words.len(),
        ..StreamScan::default()
    };
    let mut at = 0;
    while at < words.len() {
        let offset = 4 * at as u64;
        let decoded = match arch {
            Arch::Gfx1100 | Arch::Gfx1151 => gfx11::decode(arch, &words[at..]),
            _ => gfx12::decode(&words[at..]),
        };
        match decoded {
            Err(e) => {
                scan.failures.push(ScanFailure {
                    offset,
                    word: words[at],
                    rule: Rule::Decode,
                    reason: e.to_string(),
                });
                at += 1;
            }
            Ok((inst, n)) => match gfx12::encode_for(arch, &inst) {
                Ok(encoded) if encoded.as_slice() == &words[at..at + n] => {
                    scan.decoded.push(ScanInst {
                        offset,
                        words: n,
                        opcode: inst.op.name(arch).unwrap_or("?").to_owned(),
                        form: format!("{:?}", inst.form),
                    });
                    at += n;
                }
                other => {
                    let reason = match other {
                        Ok(encoded) => format!(
                            "re-encoded {encoded:08x?} != input {:08x?}",
                            &words[at..at + n]
                        ),
                        Err(e) => e.to_string(),
                    };
                    scan.failures.push(ScanFailure {
                        offset,
                        word: words[at],
                        rule: Rule::StreamIdentity,
                        reason,
                    });
                    at += n;
                }
            },
        }
    }
    if code.len() % 4 != 0 {
        scan.failures.push(ScanFailure {
            offset: 4 * words.len() as u64,
            word: 0,
            rule: Rule::Decode,
            reason: format!("stream size {} is not a dword multiple", code.len()),
        });
    }
    scan
}

/// A parsed `shader.rawspec.json` (schema `pm-raw-kernel-spec-v1`): the core spec plus the
/// relative names of the binary and sidecar files it describes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpecFile {
    pub spec: RawKernelSpec,
    pub binary: String,
    pub sidecar: String,
}

pub const SPEC_SCHEMA: &str = "pm-raw-kernel-spec-v1";

type Obj = Map<String, Value>;

/// Parse the frozen JSON schema. Every key is required and unknown keys are rejected.
pub fn parse_spec_file(value: &Value) -> Result<SpecFile, String> {
    let m = exact(
        value,
        "spec",
        &[
            "schema",
            "target",
            "symbol",
            "wave",
            "binary",
            "binary_sha256",
            "exec",
            "exec_sha256",
            "sidecar",
            "sidecar_sha256",
            "resources",
            "fp",
            "relocations",
            "entry",
        ],
    )?;
    if string(m, "schema")? != SPEC_SCHEMA {
        return Err(format!("schema is not {SPEC_SCHEMA}"));
    }
    let target = {
        let t = exact(
            &m["target"],
            "target",
            &["arch", "xnack", "sramecc", "abi_version"],
        )
        .map_err(|e| format!("target: {e}"))?;
        let arch = string(t, "arch")?;
        Target {
            arch: arch_of_name(arch).ok_or_else(|| format!("target: unknown arch {arch}"))?,
            xnack: setting(string(t, "xnack")?)?,
            sramecc: setting(string(t, "sramecc")?)?,
            abi_version: num(t, "abi_version")?,
        }
    };
    let wave = match uint(m, "wave")? {
        32 => Wave::Wave32,
        64 => Wave::Wave64,
        other => return Err(format!("wave {other} is not 32 or 64")),
    };
    let extent = {
        let e = exact(&m["exec"], "exec", &["offset", "len"]).map_err(|e| format!("exec: {e}"))?;
        ExecExtent {
            offset: uint(e, "offset")?,
            len: uint(e, "len")?,
        }
    };
    let resources = {
        let r = exact(
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
        RawResources {
            lds_bytes: num(r, "lds_bytes")?,
            scratch_bytes_per_lane: num(r, "scratch_bytes_per_lane")?,
            vgprs_actual: num(r, "vgprs_actual")?,
            vgprs_allocated: num(r, "vgprs_allocated")?,
            sgprs_actual: num(r, "sgprs_actual")?,
            sgprs_allocated: num(r, "sgprs_allocated")?,
        }
    };
    let fp = {
        let f = exact(
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
        FpState {
            round32: num(f, "round32")?,
            round16_64: num(f, "round16_64")?,
            denorm32: num(f, "denorm32")?,
            denorm16_64: num(f, "denorm16_64")?,
            dx10_clamp: boolean(f, "dx10_clamp")?,
            ieee_mode: boolean(f, "ieee_mode")?,
            fp16_overflow: boolean(f, "fp16_overflow")?,
            wgp_mode: boolean(f, "wgp_mode")?,
            mem_ordered: boolean(f, "mem_ordered")?,
            forward_progress: boolean(f, "forward_progress")?,
            wg_round_robin: boolean(f, "wg_round_robin")?,
            exceptions: num(f, "exceptions")?,
        }
    };
    let relocations = array(m, "relocations")?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let r = exact(v, "relocation", &["offset", "r_type", "symbol", "addend"])
                .map_err(|e| format!("relocations[{i}]: {e}"))?;
            Ok(Relocation {
                offset: uint(r, "offset")?,
                r_type: num(r, "r_type")?,
                symbol: string(r, "symbol")?.to_owned(),
                addend: int(r, "addend")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let entry = array(m, "entry")?
        .iter()
        .enumerate()
        .map(|(i, v)| parse_fact(v).map_err(|e| format!("entry[{i}]: {e}")))
        .collect::<Result<Vec<_>, String>>()?;
    let spec = RawKernelSpec {
        target,
        symbol: SymbolId(string(m, "symbol")?.to_owned()),
        wave,
        extent,
        binary_sha256: hex32(m, "binary_sha256")?,
        exec_sha256: hex32(m, "exec_sha256")?,
        resources,
        fp,
        relocations,
        entry,
        sidecar_sha256: hex32(m, "sidecar_sha256")?,
    };
    Ok(SpecFile {
        spec,
        binary: string(m, "binary")?.to_owned(),
        sidecar: string(m, "sidecar")?.to_owned(),
    })
}

fn parse_fact(value: &Value) -> Result<EntryFact, String> {
    let m = exact(value, "fact", &["loc", "value"])?;
    let loc = {
        let l = value_object(&m["loc"], "loc")?;
        match string(l, "kind")? {
            kind @ ("sgpr" | "vgpr" | "ttmp") => {
                let l = exact(&m["loc"], "loc", &["kind", "base", "len"])?;
                let kind = match kind {
                    "sgpr" => Kind::S,
                    "vgpr" => Kind::V,
                    _ => Kind::Ttmp,
                };
                EntryLoc::Reg(RegRef {
                    kind,
                    base: num(l, "base")?,
                    len: num(l, "len")?,
                })
            }
            "special" => {
                let l = exact(&m["loc"], "loc", &["kind", "name"])?;
                EntryLoc::Special(special(string(l, "name")?)?)
            }
            "hwreg" => {
                exact(&m["loc"], "loc", &["kind"])?;
                EntryLoc::Hwreg
            }
            other => return Err(format!("loc kind {other:?} is unknown")),
        }
    };
    let value = {
        let v = value_object(&m["value"], "value")?;
        match string(v, "kind")? {
            "defined" => {
                exact(&m["value"], "value", &["kind"])?;
                EntryValue::Defined
            }
            "workgroup_id" => {
                let v = exact(&m["value"], "value", &["kind", "axis"])?;
                EntryValue::WorkgroupId(axis(string(v, "axis")?)?)
            }
            "exact" => {
                let v = exact(&m["value"], "value", &["kind", "value"])?;
                EntryValue::Exact(num(v, "value")?)
            }
            "bounded" => {
                let v = exact(&m["value"], "value", &["kind", "min", "max"])?;
                EntryValue::Bounded {
                    min: num(v, "min")?,
                    max: num(v, "max")?,
                }
            }
            "local_id" => {
                let v = exact(&m["value"], "value", &["kind", "fields"])?;
                let fields = array(v, "fields")?
                    .iter()
                    .map(|f| {
                        let f = exact(f, "local-id field", &["axis", "shift", "bits", "max"])?;
                        Ok(LocalIdField {
                            axis: axis(string(f, "axis")?)?,
                            shift: num(f, "shift")?,
                            bits: num(f, "bits")?,
                            max: num(f, "max")?,
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                EntryValue::LocalId(fields)
            }
            other => return Err(format!("value kind {other:?} is unknown")),
        }
    };
    Ok(EntryFact { loc, value })
}

fn value_object<'a>(value: &'a Value, what: &str) -> Result<&'a Obj, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{what}: expected an object"))
}

/// `value` is an object whose keys are exactly `keys`.
fn exact<'a>(value: &'a Value, what: &str, keys: &[&str]) -> Result<&'a Obj, String> {
    let m = value_object(value, what)?;
    if let Some(extra) = m.keys().find(|k| !keys.contains(&k.as_str())) {
        return Err(format!("{what}: unknown key {extra:?}"));
    }
    if let Some(missing) = keys.iter().find(|k| !m.contains_key(**k)) {
        return Err(format!("{what}: missing key {missing:?}"));
    }
    Ok(m)
}

fn get<'a>(m: &'a Obj, key: &str) -> Result<&'a Value, String> {
    m.get(key).ok_or_else(|| format!("missing key {key:?}"))
}
fn uint(m: &Obj, key: &str) -> Result<u64, String> {
    get(m, key)?
        .as_u64()
        .ok_or_else(|| format!("{key}: expected an unsigned integer"))
}
fn int(m: &Obj, key: &str) -> Result<i64, String> {
    get(m, key)?
        .as_i64()
        .ok_or_else(|| format!("{key}: expected an integer"))
}
fn num<T: TryFrom<u64>>(m: &Obj, key: &str) -> Result<T, String> {
    T::try_from(uint(m, key)?).map_err(|_| format!("{key}: out of range"))
}
fn boolean(m: &Obj, key: &str) -> Result<bool, String> {
    get(m, key)?
        .as_bool()
        .ok_or_else(|| format!("{key}: expected a boolean"))
}
fn string<'a>(m: &'a Obj, key: &str) -> Result<&'a str, String> {
    get(m, key)?
        .as_str()
        .ok_or_else(|| format!("{key}: expected a string"))
}
fn array<'a>(m: &'a Obj, key: &str) -> Result<&'a Vec<Value>, String> {
    get(m, key)?
        .as_array()
        .ok_or_else(|| format!("{key}: expected an array"))
}
fn hex32(m: &Obj, key: &str) -> Result<[u8; 32], String> {
    let text = string(m, key)?;
    if text.len() != 64 || !text.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{key}: expected 64 hex digits"));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte =
            u8::from_str_radix(&text[2 * i..2 * i + 2], 16).map_err(|e| format!("{key}: {e}"))?;
    }
    Ok(out)
}
fn setting(text: &str) -> Result<Setting, String> {
    match text {
        "any" => Ok(Setting::Any),
        "on" => Ok(Setting::On),
        "off" => Ok(Setting::Off),
        other => Err(format!("feature setting {other:?} is not any, on or off")),
    }
}
fn axis(text: &str) -> Result<Axis, String> {
    match text {
        "x" => Ok(Axis::X),
        "y" => Ok(Axis::Y),
        "z" => Ok(Axis::Z),
        other => Err(format!("axis {other:?} is not x, y or z")),
    }
}
fn special(name: &str) -> Result<Special, String> {
    Ok(match name {
        "vcc" => Special::Vcc,
        "vcc_lo" => Special::VccLo,
        "vcc_hi" => Special::VccHi,
        "exec" => Special::Exec,
        "exec_lo" => Special::ExecLo,
        "exec_hi" => Special::ExecHi,
        "scc" => Special::Scc,
        "m0" => Special::M0,
        "flat_scratch" => Special::FlatScratch,
        "src_shared_base" => Special::SrcSharedBase,
        "src_shared_limit" => Special::SrcSharedLimit,
        "src_private_base" => Special::SrcPrivateBase,
        "src_private_limit" => Special::SrcPrivateLimit,
        "pc" => Special::Pc,
        other => match other
            .strip_prefix("ttmp")
            .and_then(|n| n.parse::<u8>().ok())
        {
            Some(n) if n < 16 => Special::Ttmp(n),
            _ => return Err(format!("special register {other:?} is unknown")),
        },
    })
}
