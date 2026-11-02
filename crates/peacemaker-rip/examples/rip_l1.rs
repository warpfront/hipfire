// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! RIP-L1: the edit -> offline-ready latency probe (`cargo build --release -p peacemaker-rip --features l1
//! --example rip_l1`; build once, then run the binary for every edit: a script-body edit needs no rebuild).
//!
//! One attempt takes one `iu4_gemm` variant authored either as a script-native `.rip` body (the medium that
//! never recompiles Rust) or as the Rust generator compiled into this binary, and runs the offline pipeline
//! stage by stage, each stage timed with a UTC epoch-ns timestamp and a monotonic duration:
//!
//! `parse_eval` (emit) -> `native_link` (assemble + write the code object) -> `m7` (certification receipt)
//! -> `twin` (compare against the pinned unmodified Rust baseline) -> `emu_reference` (run the whole launch
//! grid on the PeaceMaker emulator and compare bitwise to the exact CPU fold) -> `offline_ready`.
//!
//! The first refusing stage ends the attempt; a refusal is a recorded outcome (`outcome: "rejected"`), not a
//! tool failure. `--write-pin` instead emits the unmodified Rust baseline and writes `PIN.json`.
//!
//! The emulator harness is the one the `peacemaker-isa` integration tests use, included by path.

#[allow(dead_code)]
#[path = "../../peacemaker-isa/tests/support/iu4_emu.rs"]
mod iu4_emu;

use peacemaker_isa::kernels::iu4_gemm::{self, ALayout, Cacc, Epi, Fold, Mma, Spec, Tile, Tune};
use peacemaker_isa::pm_check::{self, M7Cache};
use peacemaker_isa::{Arch, Emitted};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const USAGE: &str = "usage: rip_l1 --medium rip|rust --variant V [--rip FILE.rip] \
--shape M=136,N=40,K=768 [--seed 0x9e3779b97f4a7c15] --pin PIN.json --out-dir DIR --attempt ID \
[--edit-ready-ns N] [--host-build-ms MS]\n       rip_l1 --write-pin --variant V --pin PIN.json";

const DEFAULT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const ARCH: Arch = Arch::Gfx1201;
const ARCH_NAME: &str = "gfx1201";

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Every dense and 2:4-sparse `iu4_gemm` variant the generator publishes (default tune), as
/// `crates/peacemaker-rip/tests/iu4_gemm_fine_twin.rs` enumerates them.
fn all_specs() -> Vec<Spec> {
    let mut out = Vec::new();
    for mma in [Mma::Dense, Mma::S24X64, Mma::S24X32] {
        for tile in [Tile::T128x128x8, Tile::T256x128x16] {
            for epi in [
                Epi::Set,
                Epi::Add,
                Epi::AddSigmoid,
                Epi::GateUpSilu,
                Epi::GateUpSiluBf16,
                Epi::QkvzaGdn,
            ] {
                for act in [ALayout::Token, ALayout::Slab] {
                    let spec = Spec {
                        fold: Fold::K128,
                        tile,
                        cacc: Cacc::One,
                        epi,
                        act,
                        arch: ARCH,
                        mma,
                        tune: Tune::default(),
                    };
                    if spec.validate().is_ok() {
                        out.push(spec);
                    }
                }
            }
        }
    }
    out
}

fn spec_of(variant: &str) -> Result<Spec, String> {
    let specs = all_specs();
    specs
        .iter()
        .copied()
        .find(|s| s.variant_name() == variant)
        .ok_or_else(|| {
            format!(
                "unknown variant `{variant}`; the {} valid specs are: {}",
                specs.len(),
                specs
                    .iter()
                    .map(|s| s.variant_name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// sha256 of the proof's JSON with the `variant` field normalized (the variant name carries the tune suffix on
/// the generator's side only, as in `iu4_gemm_fine_twin.rs::assert_same`).
fn proof_sha256(e: &Emitted) -> Result<String, String> {
    let mut v = serde_json::to_value(&e.proof).map_err(|x| format!("proof json: {x}"))?;
    if let Some(o) = v.as_object_mut() {
        o.insert("variant".into(), Value::String(String::new()));
    }
    let s = serde_json::to_string(&v).map_err(|x| format!("proof json: {x}"))?;
    Ok(sha256_hex(s.as_bytes()))
}

/// Byte size of the `.text` section of an ELF64 little-endian object, if it has one.
fn text_bytes(elf: &[u8]) -> Option<u64> {
    let u16_at = |o: usize| {
        elf.get(o..o + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]) as usize)
    };
    let u32_at = |o: usize| {
        elf.get(o..o + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
    };
    let u64_at = |o: usize| {
        elf.get(o..o + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    };
    if elf.get(..4)? != b"\x7fELF" || *elf.get(4)? != 2 || *elf.get(5)? != 1 {
        return None;
    }
    let shoff = u64_at(0x28)? as usize;
    let shentsize = u16_at(0x3A)?;
    let shnum = u16_at(0x3C)?;
    let shstrndx = u16_at(0x3E)?;
    let sh = |i: usize| shoff + i * shentsize;
    let strtab_off = u64_at(sh(shstrndx) + 0x18)? as usize;
    for i in 0..shnum {
        let name_off = strtab_off + u32_at(sh(i))?;
        let name = elf.get(name_off..)?.split(|&b| b == 0).next()?;
        if name == b".text" {
            return u64_at(sh(i) + 0x20);
        }
    }
    None
}

#[derive(Default)]
struct Args {
    medium: Option<String>,
    variant: Option<String>,
    rip: Option<PathBuf>,
    shape: Option<String>,
    seed: Option<String>,
    pin: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    attempt: Option<String>,
    edit_ready_ns: Option<String>,
    host_build_ms: Option<String>,
    write_pin: bool,
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        if k == "--write-pin" {
            a.write_pin = true;
            continue;
        }
        if !k.starts_with("--") {
            return Err(format!("unexpected argument `{k}`\n{USAGE}"));
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{k} needs a value\n{USAGE}"))?
            .clone();
        match k.as_str() {
            "--medium" => a.medium = Some(v),
            "--variant" => a.variant = Some(v),
            "--rip" => a.rip = Some(v.into()),
            "--shape" => a.shape = Some(v),
            "--seed" => a.seed = Some(v),
            "--pin" => a.pin = Some(v.into()),
            "--out-dir" => a.out_dir = Some(v.into()),
            "--attempt" => a.attempt = Some(v),
            "--edit-ready-ns" => a.edit_ready_ns = Some(v),
            "--host-build-ms" => a.host_build_ms = Some(v),
            _ => return Err(format!("unknown flag `{k}`\n{USAGE}")),
        }
    }
    Ok(a)
}

fn parse_u64(s: &str, what: &str) -> Result<u64, String> {
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(&h.replace('_', ""), 16),
        None => s.replace('_', "").parse(),
    };
    r.map_err(|e| format!("{what} `{s}`: {e}"))
}

fn parse_shape(s: &str) -> Result<(usize, usize, usize), String> {
    let (mut m, mut n, mut k) = (None, None, None);
    for part in s.split(',') {
        let (key, val) = part
            .split_once('=')
            .ok_or_else(|| format!("--shape `{s}`: expected M=..,N=..,K=.."))?;
        let v = val
            .parse::<usize>()
            .map_err(|e| format!("--shape {key}=`{val}`: {e}"))?;
        match key {
            "M" => m = Some(v),
            "N" => n = Some(v),
            "K" => k = Some(v),
            _ => return Err(format!("--shape: unknown dimension `{key}`")),
        }
    }
    match (m, n, k) {
        (Some(m), Some(n), Some(k)) if m > 0 && n > 0 && k > 0 => {
            if k % 256 != 0 {
                return Err(format!("--shape K={k} must be a multiple of 256"));
            }
            Ok((m, n, k))
        }
        _ => Err(format!("--shape `{s}` must give positive M, N and K")),
    }
}

fn need<'a, T>(o: &'a Option<T>, flag: &str) -> Result<&'a T, String> {
    o.as_ref()
        .ok_or_else(|| format!("{flag} is required\n{USAGE}"))
}

fn write_json(path: &Path, v: &Value) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
    }
    let s = serde_json::to_string_pretty(v).map_err(|e| e.to_string())?;
    std::fs::write(path, s + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// `--write-pin`: emit the unmodified Rust baseline of `variant`, assemble it, write PIN.json.
fn write_pin(variant: &str, pin: &Path) -> Result<(), String> {
    let spec = spec_of(variant)?;
    let emitted = iu4_gemm::emit(spec)?;
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, ARCH)?;
    let doc = json!({
        "variant": variant,
        "arch": ARCH_NAME,
        "symbol": spec.symbol(),
        "s_text_sha256": sha256_hex(emitted.s_text.as_bytes()),
        "proof_sha256": proof_sha256(&emitted)?,
        "object_sha256": sha256_hex(&elf),
        "object_bytes": elf.len(),
        "text_lines": emitted.s_text.lines().count(),
        "code_bytes": text_bytes(&elf),
        "s_text": emitted.s_text,
    });
    write_json(pin, &doc)?;
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
    Ok(())
}

struct Stage {
    name: &'static str,
    start_ns: u64,
    end_ns: u64,
    ms: f64,
}

/// A refusing stage and why.
struct Refusal {
    stage: &'static str,
    error: String,
}

/// The attempt's accumulated record.
struct Rec {
    stages: Vec<Stage>,
    object_sha256: Option<String>,
    code_size: Option<u64>,
    object_bytes: Option<usize>,
    text_lines: Option<usize>,
    m7: Value,
    twin: Value,
    emu: Value,
}

impl Rec {
    fn stage<T>(
        &mut self,
        name: &'static str,
        f: impl FnOnce(&mut Rec) -> Result<T, String>,
    ) -> Result<T, Refusal> {
        let start_ns = now_ns();
        let t = Instant::now();
        let r = f(self);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.stages.push(Stage {
            name,
            start_ns,
            end_ns: now_ns(),
            ms,
        });
        r.map_err(|error| Refusal { stage: name, error })
    }
}

fn first_diff(candidate: &str, pin: &str) -> Value {
    let (mut c, mut p) = (candidate.lines(), pin.lines());
    let mut line = 0usize;
    loop {
        line += 1;
        match (c.next(), p.next()) {
            (None, None) => return Value::Null,
            (a, b) if a == b => {}
            (a, b) => {
                return json!({
                    "line": line,
                    "candidate": a,
                    "pin": b,
                    "candidate_lines": candidate.lines().count(),
                    "pin_lines": pin.lines().count(),
                })
            }
        }
    }
}

struct Inputs<'a> {
    a: &'a Args,
    medium: &'a str,
    variant: &'a str,
    spec: Spec,
    shape: (usize, usize, usize),
    seed: u64,
    pin: &'a Value,
    obj_path: &'a Path,
}

fn run_stages(rec: &mut Rec, inp: &Inputs) -> Result<(), Refusal> {
    let spec = inp.spec;
    let emitted = rec.stage("parse_eval", |rec| {
        let e = match inp.medium {
            "rip" => {
                let rip = need(&inp.a.rip, "--rip")?;
                let src =
                    std::fs::read_to_string(rip).map_err(|e| format!("{}: {e}", rip.display()))?;
                let base = rip
                    .file_name()
                    .map(|b| b.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "rip".into());
                peacemaker_rip::compile_named(&src, &base, ARCH, inp.variant)?
            }
            _ => iu4_gemm::emit(spec)?,
        };
        rec.text_lines = Some(e.s_text.lines().count());
        Ok(e)
    })?;

    let symbol = spec.symbol();
    rec.stage("native_link", |rec| {
        let elf = peacemaker_isa::native::assemble(&emitted.s_text, ARCH)?;
        std::fs::write(inp.obj_path, &elf)
            .map_err(|e| format!("{}: {e}", inp.obj_path.display()))?;
        rec.object_sha256 = Some(sha256_hex(&elf));
        rec.object_bytes = Some(elf.len());
        rec.code_size = text_bytes(&elf);
        Ok(())
    })?;

    rec.stage("m7", |rec| {
        let (receipt, cache) = pm_check::m7_status(inp.obj_path, ARCH_NAME, &symbol)?;
        rec.m7 = json!({
            "status": "pass",
            "cache": cache.as_str(),
            "cache_hit": cache == M7Cache::Hit,
            "receipt": receipt,
        });
        Ok(())
    })
    .map_err(|r| {
        rec.m7 = json!({"status": "fail", "error": r.error});
        r
    })?;

    rec.stage("twin", |rec| {
        let pin_s = inp.pin["s_text"].as_str().unwrap_or("");
        let cand_s_sha = sha256_hex(emitted.s_text.as_bytes());
        let cand_proof = proof_sha256(&emitted)?;
        let same_s = inp.pin["s_text_sha256"].as_str() == Some(cand_s_sha.as_str());
        let same_proof = inp.pin["proof_sha256"].as_str() == Some(cand_proof.as_str());
        let same_obj = inp.pin["object_sha256"].as_str() == rec.object_sha256.as_deref();
        rec.twin = json!({
            "identical_to_pin": same_s && same_proof && same_obj,
            "s_text_identical": same_s,
            "proof_identical": same_proof,
            "object_identical": same_obj,
            "s_text_sha256": cand_s_sha,
            "proof_sha256": cand_proof,
            "pin_s_text_sha256": inp.pin["s_text_sha256"],
            "first_diff": if same_s { Value::Null } else { first_diff(&emitted.s_text, pin_s) },
        });
        Ok(())
    })?;

    let add = match spec.epi {
        Epi::Set => false,
        Epi::Add => true,
        e => {
            return rec.stage("emu_reference", |_| {
                Err::<(), _>(format!(
                    "the emulator harness drives only SET/ADD variants, not {}",
                    e.name()
                ))
            })
        }
    };
    rec.stage("emu_reference", |rec| {
        let (m, n, k) = inp.shape;
        let p = iu4_emu::Problem::synth(m, n, k, inp.seed);
        let got = iu4_emu::run_emitted(spec, &emitted, &p)?;
        let want = iu4_emu::reference(&p, add);
        if got.len() != want.len() {
            return Err(format!(
                "emulator returned {} values, want {}",
                got.len(),
                want.len()
            ));
        }
        let mut mismatches = 0usize;
        let mut first = Value::Null;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            if g.to_bits() != w.to_bits() {
                if mismatches == 0 {
                    first = json!({
                        "index": i,
                        "row": i % m,
                        "col": i / m,
                        "got": g,
                        "got_bits": format!("{:#010x}", g.to_bits()),
                        "want": w,
                        "want_bits": format!("{:#010x}", w.to_bits()),
                    });
                }
                mismatches += 1;
            }
        }
        rec.emu = json!({
            "elements": got.len(),
            "mismatches": mismatches,
            "first_mismatch": first,
            "add": add,
        });
        if mismatches != 0 {
            return Err(format!(
                "{mismatches} of {} outputs differ bitwise from the CPU reference",
                got.len()
            ));
        }
        Ok(())
    })
}

fn run(args: &[String]) -> Result<(), String> {
    let t0 = Instant::now();
    let process_start_ns = now_ns();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        return Err(USAGE.to_owned());
    }
    let a = parse_args(args)?;
    let variant = need(&a.variant, "--variant")?.clone();
    if a.write_pin {
        return write_pin(&variant, need(&a.pin, "--pin")?);
    }
    let medium = need(&a.medium, "--medium")?.clone();
    if medium != "rip" && medium != "rust" {
        return Err(format!("--medium `{medium}`: expected rip or rust"));
    }
    if medium == "rip" {
        need(&a.rip, "--rip")?;
    }
    let spec = spec_of(&variant)?;
    let shape = parse_shape(need(&a.shape, "--shape")?)?;
    let seed = match &a.seed {
        Some(s) => parse_u64(s, "--seed")?,
        None => DEFAULT_SEED,
    };
    let out_dir = need(&a.out_dir, "--out-dir")?.clone();
    let attempt = need(&a.attempt, "--attempt")?.clone();
    let pin_path = need(&a.pin, "--pin")?;
    let pin: Value = serde_json::from_str(
        &std::fs::read_to_string(pin_path).map_err(|e| format!("{}: {e}", pin_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", pin_path.display()))?;
    if pin["variant"].as_str() != Some(variant.as_str()) {
        return Err(format!(
            "{}: pins variant {}, not `{variant}`",
            pin_path.display(),
            pin["variant"]
        ));
    }
    let edit_ready_ns = a
        .edit_ready_ns
        .as_deref()
        .map(|s| parse_u64(s, "--edit-ready-ns"))
        .transpose()?;
    let host_build_ms = a
        .host_build_ms
        .as_deref()
        .map(|s| {
            s.parse::<f64>()
                .map_err(|e| format!("--host-build-ms `{s}`: {e}"))
        })
        .transpose()?;
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let obj_path = out_dir.join(format!("{attempt}.co"));

    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let exe_sha = std::fs::read(&exe)
        .map(|b| sha256_hex(&b))
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    let rip_sha = match (&medium[..], &a.rip) {
        ("rip", Some(p)) => Some(
            std::fs::read(p)
                .map(|b| sha256_hex(&b))
                .map_err(|e| format!("{}: {e}", p.display()))?,
        ),
        _ => None,
    };

    let mut rec = Rec {
        stages: Vec::new(),
        object_sha256: None,
        code_size: None,
        object_bytes: None,
        text_lines: None,
        m7: Value::Null,
        twin: Value::Null,
        emu: Value::Null,
    };
    let inp = Inputs {
        a: &a,
        medium: &medium,
        variant: &variant,
        spec,
        shape,
        seed,
        pin: &pin,
        obj_path: &obj_path,
    };
    let refusal = run_stages(&mut rec, &inp).err();
    let (start, t) = (now_ns(), Instant::now());
    rec.stages.push(Stage {
        name: "offline_ready",
        start_ns: start,
        end_ns: start,
        ms: t.elapsed().as_secs_f64() * 1e3,
    });
    let offline_ready_ns = start;
    let total_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut doc = json!({
        "attempt": attempt,
        "medium": medium,
        "variant": variant,
        "symbol": spec.symbol(),
        "shape": {"M": shape.0, "N": shape.1, "K": shape.2},
        "seed": format!("{seed:#018x}"),
        "tool": {"exe": exe.display().to_string(), "exe_sha256": exe_sha},
        "rip_path": a.rip.as_ref().map(|p| p.display().to_string()),
        "rip_sha256": rip_sha,
        "edit_ready_ns": edit_ready_ns,
        "host_build_ms": host_build_ms,
        "process_start_ns": process_start_ns,
        "stages": rec.stages.iter().map(|s| json!({
            "name": s.name, "start_ns": s.start_ns, "end_ns": s.end_ns, "ms": s.ms,
        })).collect::<Vec<_>>(),
        "outcome": if refusal.is_some() { "rejected" } else { "survivor" },
        "refused_stage": refusal.as_ref().map(|r| r.stage),
        "error": refusal.as_ref().map(|r| r.error.clone()),
        "object_path": obj_path.display().to_string(),
        "object_sha256": rec.object_sha256,
        "object_bytes": rec.object_bytes,
        "code_size": rec.code_size,
        "text_lines": rec.text_lines,
        "m7": rec.m7,
        "twin": rec.twin,
        "emu": rec.emu,
        "offline_ready_ns": offline_ready_ns,
        "total_ms": total_ms,
    });
    if let Some(e) = edit_ready_ns {
        doc["edit_to_offline_ms"] = json!(offline_ready_ns.saturating_sub(e) as f64 / 1e6);
    }
    write_json(&out_dir.join(format!("{attempt}.json")), &doc)?;
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("rip_l1: {e}");
        std::process::exit(1);
    }
}
