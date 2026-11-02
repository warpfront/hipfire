// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Shared builders of the `pm_loop` workloads: hashing, the deterministic PRNG
//! and input generators, kernarg packing, shape-string parsing (`fn:`/`a3b:`
//! presets plus `key=value` overrides), shipped-object selection, native-writer
//! candidate objects, dtype-aware comparison, and the emulator launcher the
//! producer workloads share.
//!
//! Included with `#[path = "pm_loop/common.rs"] mod common;`.
#![allow(dead_code)]

use crate::workload::{Object, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// `needle` occurs in `hay` (symbol names inside an embedded bundle).
pub fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

pub fn dev_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub fn put_ptr(b: &mut Vec<u8>, p: u64) {
    b.extend_from_slice(&p.to_le_bytes());
}

pub fn put_i32(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}

pub fn put_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

pub fn put_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Deterministic inputs

/// xorshift64*. The one PRNG of the producer workloads (`seed` 0 becomes 1).
#[derive(Clone, Debug)]
pub struct Xs64(u64);

impl Xs64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }
    /// Standard normal (Box-Muller, one variate per call), as f32.
    pub fn normal(&mut self) -> f32 {
        let u = |r: &mut Self| ((r.next() >> 11) + 1) as f64 / (1u64 << 53) as f64;
        let (a, b) = (u(self), u(self));
        ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32
    }
}

/// `n` f32 values `unit() * amp`, little-endian bytes.
pub fn f32_unit_bytes(seed: u64, n: usize, amp: f32) -> Vec<u8> {
    let mut r = Xs64::new(seed);
    (0..n)
        .flat_map(|_| (r.unit() * amp).to_le_bytes())
        .collect()
}

/// `n` bf16 values (RNE of `normal() * amp`, with 0.1% x30 outliers), as bits.
pub fn bf16_normal_bytes(seed: u64, n: usize, amp: f32) -> Vec<u8> {
    let mut r = Xs64::new(seed);
    (0..n)
        .flat_map(|_| {
            let mut v = r.normal() * amp;
            if r.next() % 1000 == 0 {
                v *= 30.0;
            }
            bf16_bits(v).to_le_bytes()
        })
        .collect()
}

pub fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

pub fn f16_val(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

/// f32 -> bf16 bits, round to nearest even (finite and Inf; NaN keeps a quiet payload).
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// OCP E4M3 (finite, bias 7) code of `x`, round-to-nearest-even, saturating at 448 (NaN and zero are 0).
pub fn e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() || a == 0.0 {
        return 0;
    }
    let code = if a < 2f32.powi(-6) {
        // Subnormal: value = m * 2^-9 (m == 8 rounds up into the smallest normal, 0x08).
        (a * 512.0).round_ties_even() as u32
    } else {
        let mut e = (a.to_bits() >> 23) as i32 - 127;
        let mut m = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
        if m == 8 {
            e += 1;
            m = 0;
        }
        let biased = e + 7;
        if biased > 15 || (biased == 15 && m > 6) {
            0x7e
        } else {
            ((biased as u32) << 3) | m as u32
        }
    };
    sign | code as u8
}

/// The f32 value of an OCP E4M3FN code (`0x7f`/`0xff` are NaN).
pub fn e4m3_val(c: u8) -> f32 {
    let (s, e, m) = (c >> 7, i32::from((c >> 3) & 15), f32::from(c & 7));
    let v = if e == 0 {
        m * 2f32.powi(-9)
    } else if e == 15 && c & 7 == 7 {
        f32::NAN
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

/// One `block_i4_128` record (72 B: f32 d, i32 sum(q), 64 nibble bytes, low
/// nibble = even K) from 128 signed codes.
pub fn i4_record(d: f32, q: &[i32; 128]) -> [u8; 72] {
    let mut rec = [0u8; 72];
    rec[0..4].copy_from_slice(&d.to_le_bytes());
    rec[4..8].copy_from_slice(&q.iter().sum::<i32>().to_le_bytes());
    for j in 0..64 {
        rec[8 + j] = (q[2 * j] as u8 & 0xf) | ((q[2 * j + 1] as u8 & 0xf) << 4);
    }
    rec
}

/// Per-block symmetric A4 quantization: `d = max|x| / 7`, `q = clamp(round(x/d), -8, 7)`.
pub fn quantize_i4_block(x: &[f32; 128]) -> [u8; 72] {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 7.0;
    let mut q = [0i32; 128];
    if d > 0.0 {
        for (qi, v) in q.iter_mut().zip(x) {
            *qi = ((v / d).round() as i32).clamp(-8, 7);
        }
    }
    i4_record(d, &q)
}

// ---------------------------------------------------------------------------
// Shapes

/// A parsed `--shape`: an optional `fn:`/`a3b:` preset and `key=value` pairs.
/// Grammar: `[fn:NAME|a3b:NAME][,key=value]...` (items in any order; pairs
/// override the preset's values).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShapeSpec {
    /// `("fn", NAME)` or `("a3b", NAME)`.
    pub preset: Option<(String, String)>,
    pub kv: BTreeMap<String, String>,
}

impl ShapeSpec {
    pub fn parse(s: &str) -> Result<Self> {
        let mut out = Self::default();
        for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if let Some((fam, name)) = ["fn:", "a3b:"]
                .iter()
                .find_map(|p| part.strip_prefix(p).map(|n| (p.trim_end_matches(':'), n)))
            {
                if out.preset.replace((fam.into(), name.into())).is_some() {
                    return Err(format!("shape `{s}`: two presets"));
                }
            } else {
                let (k, v) = part.split_once('=').ok_or_else(|| {
                    format!("shape `{s}`: expected fn:NAME, a3b:NAME or key=value, got `{part}`")
                })?;
                if out.kv.insert(k.trim().into(), v.trim().into()).is_some() {
                    return Err(format!("shape `{s}`: duplicate {k}"));
                }
            }
        }
        Ok(out)
    }

    /// Merge: the preset's `defaults` first, then this shape's pairs on top.
    pub fn resolve(&self, defaults: &[(&str, usize)]) -> BTreeMap<String, String> {
        let mut m: BTreeMap<String, String> = defaults
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.to_string()))
            .collect();
        m.extend(self.kv.clone());
        m
    }
}

/// Take a required positive-or-zero integer dimension out of a resolved map.
pub fn take_dim(m: &mut BTreeMap<String, String>, key: &str) -> Result<usize> {
    let v = m
        .remove(key)
        .ok_or_else(|| format!("shape: {key} is required"))?;
    v.parse::<usize>()
        .map_err(|e| format!("shape: {key}={v}: {e}"))
}

/// Take an optional dimension.
pub fn take_dim_or(m: &mut BTreeMap<String, String>, key: &str, default: usize) -> Result<usize> {
    match m.remove(key) {
        Some(v) => v.parse().map_err(|e| format!("shape: {key}={v}: {e}")),
        None => Ok(default),
    }
}

/// Error naming every shape key nobody consumed.
pub fn no_leftover(m: &BTreeMap<String, String>) -> Result<()> {
    if m.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "shape: unknown key(s): {}",
            m.keys().cloned().collect::<Vec<_>>().join(", ")
        ))
    }
}

// ---------------------------------------------------------------------------
// Objects

/// The shipped baseline: an embedded `.hxaco`, with the launched symbol (and
/// every symbol in `exports`) checked to occur in it. `selection` names the
/// file:line where production picks it.
pub fn shipped_object(
    bytes: &[u8],
    artifact: &str,
    selection: &str,
    launch_symbol: &str,
    exports: &[&str],
    knobs: Value,
) -> Result<Object> {
    for s in std::iter::once(&launch_symbol).chain(exports) {
        if !contains(bytes, s.as_bytes()) {
            return Err(format!(
                "symbol {s} is not in the shipped bundle {artifact}"
            ));
        }
    }
    Ok(Object {
        bytes: bytes.to_vec(),
        launch_symbol: launch_symbol.into(),
        exports: exports.iter().map(|s| (*s).to_owned()).collect(),
        desc: json!({"artifact": artifact, "knobs": knobs}),
        source: json!({"source": selection, "artifact": artifact, "sha256": sha256_hex(bytes)}),
    })
}

/// The native-writer link of emitted assembly (`peacemaker-isa emit --co`).
pub fn emitted_object(
    s_text: &str,
    arch: peacemaker_isa::Arch,
    launch_symbol: &str,
    exports: &[String],
    what: &str,
    knobs: Value,
) -> Result<Object> {
    let elf = peacemaker_isa::native::assemble(s_text, arch)?;
    Ok(Object {
        source: json!({"source": what, "sha256": sha256_hex(&elf)}),
        desc: json!({"knobs": knobs}),
        exports: exports.to_vec(),
        launch_symbol: launch_symbol.into(),
        bytes: elf,
    })
}

/// `s_text` (one emitted kernel, `peacemaker-isa emit` layout) with the body of `symbol` replaced by a
/// single `s_endpgm`: the kernel descriptor (registers, LDS, kernarg) and the symbol are unchanged.
pub fn null_kernel_text(s_text: &str, symbol: &str) -> Result<String> {
    let (head, tail) = (format!("\n{symbol}:\n"), format!(".L{symbol}_end:\n"));
    let a = s_text
        .find(&head)
        .ok_or_else(|| format!("emitted text has no `{symbol}:` label"))?
        + head.len();
    let b = s_text[a..]
        .find(&tail)
        .ok_or_else(|| format!("emitted text has no `.L{symbol}_end:` label"))?
        + a;
    Ok(format!("{}\ts_endpgm\n{}", &s_text[..a], &s_text[b..]))
}

/// The null object of an emitted kernel: [`null_kernel_text`] linked with the native writer.
pub fn null_object(
    s_text: &str,
    arch: peacemaker_isa::Arch,
    symbol: &str,
    what: &str,
) -> Result<Object> {
    let elf = peacemaker_isa::native::assemble(&null_kernel_text(s_text, symbol)?, arch)?;
    Ok(Object {
        source: json!({"source": what, "body": "s_endpgm", "sha256": sha256_hex(&elf)}),
        desc: json!({"null_of": symbol}),
        exports: vec![symbol.to_owned()],
        launch_symbol: symbol.to_owned(),
        bytes: elf,
    })
}

/// `--rip FILE`: the script's path and text.
pub type RipSource = Option<(String, String)>;

/// Read the `--rip` script, if any.
pub fn read_rip(rip: Option<&std::path::Path>) -> Result<RipSource> {
    match rip {
        Some(p) => Ok(Some((
            p.display().to_string(),
            std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        ))),
        None => Ok(None),
    }
}

/// The candidate of a producer workload compiled from a `.rip` script: `peacemaker_rip::compile(source, arch,
/// variant)`, refused (exit 3, stage `rip`) unless its kernel `symbol` and ABI (kernarg layout, workgroup size,
/// LDS) equal those of the Rust emit of the same entry (`generator_s_text`), then linked with the native writer.
/// `desc.rip` = `{path, sha256, variant, identical_to_generator}`.
pub fn rip_object(
    rip: &(String, String),
    arch: peacemaker_isa::Arch,
    variant: &str,
    symbol: &str,
    generator_s_text: &str,
    knobs: Value,
) -> Result<Object> {
    let (path, source) = rip;
    let compiled = peacemaker_rip::compile(source, arch, variant)?;
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, arch)?;
    let gen_elf = peacemaker_isa::native::assemble(generator_s_text, arch)?;
    let got = crate::iu4::abi_signature(&elf, symbol)
        .map_err(|e| format!("compiled kernel does not expose {symbol}: {e}"))?;
    let want = crate::iu4::abi_signature(&gen_elf, symbol)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with the Rust emit of {variant}:\n  .rip:      {got}\n  generator: {want}"
        ));
    }
    Ok(Object {
        source: json!({
            "source": format!("peacemaker_rip::compile({path}) + native::assemble"),
            "variant": variant,
            "sha256": sha256_hex(&elf),
        }),
        desc: json!({
            "knobs": knobs,
            "rip": {
                "path": path,
                "sha256": sha256_hex(source.as_bytes()),
                "variant": variant,
                "identical_to_generator": compiled.s_text == generator_s_text,
            },
        }),
        exports: vec![symbol.to_owned()],
        launch_symbol: symbol.to_owned(),
        bytes: elf,
    })
}

// ---------------------------------------------------------------------------
// Comparison

/// Element type of an output buffer, for max-abs-diff and for reading mismatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Elem {
    F32,
    F16,
    Bf16,
    I32,
    U8,
}

impl Elem {
    pub fn width(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::U8 => 1,
        }
    }
    /// The value of one element as f64; `None` for a NaN.
    pub fn value(self, c: &[u8]) -> Option<f64> {
        let v = match self {
            Self::F32 => f64::from(f32::from_le_bytes(c.try_into().unwrap())),
            Self::F16 => f64::from(f16_val(u16::from_le_bytes([c[0], c[1]]))),
            Self::Bf16 => f64::from(bf16_val(u16::from_le_bytes([c[0], c[1]]))),
            Self::I32 => f64::from(i32::from_le_bytes(c.try_into().unwrap())),
            Self::U8 => f64::from(c[0]),
        };
        (!v.is_nan()).then_some(v)
    }
}

/// `|x - y|` of two elements: 0 when bitwise equal, infinite when only one is
/// NaN or when the NaN payloads differ.
pub fn elem_diff(elem: Elem, x: &[u8], y: &[u8]) -> f64 {
    if x == y {
        return 0.0;
    }
    match (elem.value(x), elem.value(y)) {
        (Some(a), Some(b)) => (a - b).abs(),
        _ => f64::INFINITY,
    }
}

/// Max element difference of two equal-length byte slices of `elem`.
pub fn max_diff(elem: Elem, a: &[u8], b: &[u8]) -> f64 {
    let w = elem.width();
    a.chunks_exact(w)
        .zip(b.chunks_exact(w))
        .map(|(x, y)| elem_diff(elem, x, y))
        .fold(0f64, f64::max)
}

// ---------------------------------------------------------------------------
// Emulator

/// One buffer of an emulated launch.
pub struct EmuBuf {
    pub name: &'static str,
    pub va: u64,
    pub bytes: Vec<u8>,
}

/// Run every workgroup of `grid` of `symbol` (in the linked object `elf`) on
/// the peacemaker emulator over real kernargs and mapped buffers, each followed
/// by a zeroed 4 KiB tail so a store past its end is caught. The hidden block
/// counts and group sizes of the kernarg segment are filled from the launch,
/// as ROCm does. Returns the memory (read outputs with `mem.read`) or the
/// emulator's honest refusal.
pub fn emu_run(
    elf: &[u8],
    symbol: &str,
    mut kernarg: Vec<u8>,
    bufs: Vec<EmuBuf>,
    grid: [u32; 3],
    block: [u32; 3],
    lds: u32,
) -> Result<peacemaker_ir::emu::Memory> {
    use peacemaker_ir::emu::{Emulator, Launch, Memory};
    use peacemaker_ir::inst::Abi;
    let lifted = peacemaker_lift::lift_object(
        elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))?;
    // Reports the first opcode without semantics, if any.
    let emu = Emulator::new(&lifted.program, symbol)?.hw_model();
    let kernel = lifted
        .program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("object has no kernel {symbol}"))?;
    if let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    {
        if kernarg.len() < descriptor.kernarg_size as usize {
            kernarg.resize(descriptor.kernarg_size as usize, 0);
        }
        for a in &metadata.parsed.args {
            let at = a.offset as usize;
            let v: Option<(u32, usize)> = match a.value_kind.as_str() {
                "hidden_block_count_x" => Some((grid[0], 4)),
                "hidden_block_count_y" => Some((grid[1], 4)),
                "hidden_block_count_z" => Some((grid[2], 4)),
                "hidden_group_size_x" => Some((block[0], 2)),
                "hidden_group_size_y" => Some((block[1], 2)),
                "hidden_group_size_z" => Some((block[2], 2)),
                _ => None,
            };
            if let Some((v, w)) = v {
                if at + w <= kernarg.len() {
                    kernarg[at..at + w].copy_from_slice(&v.to_le_bytes()[..w]);
                }
            }
        }
    }
    let kva = 0x1000_0000u64;
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kva, kernarg)?;
    let names: Vec<&'static str> = bufs.iter().map(|b| b.name).collect();
    for b in bufs {
        let len = b.bytes.len() as u64;
        mem.map(b.name.into(), b.va, b.bytes)?;
        mem.map(format!("{}_tail", b.name), b.va + len, vec![0; 4096])?;
    }
    for z in 0..grid[2] {
        for y in 0..grid[1] {
            for x in 0..grid[0] {
                emu.run_workgroup(
                    &Launch {
                        kernarg_ptr: kva,
                        workgroup: [x, y, z],
                        block,
                        dynamic_lds: lds,
                    },
                    &mut mem,
                )
                .map_err(|e| format!("workgroup ({x},{y},{z}): {e}"))?;
            }
        }
    }
    for n in names {
        let tail = format!("{n}_tail");
        let r = mem.regions.iter().find(|r| r.name == tail);
        if r.is_some_and(|r| r.bytes.iter().any(|&b| b != 0)) {
            return Err(format!("kernel wrote past the end of the {n} buffer"));
        }
    }
    Ok(mem)
}

/// An emulator refusal as the record's `emu` value: `refused (<reason>)`.
pub fn refused(reason: impl std::fmt::Display) -> Value {
    Value::String(format!("refused ({reason})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_spec_grammar() {
        let s = ShapeSpec::parse("fn:t4096,rows=3, K=2048").unwrap();
        assert_eq!(s.preset, Some(("fn".into(), "t4096".into())));
        assert_eq!(s.kv.get("rows").map(String::as_str), Some("3"));
        assert_eq!(s.kv.get("K").map(String::as_str), Some("2048"));
        let a = ShapeSpec::parse("a3b:x").unwrap();
        assert_eq!(a.preset, Some(("a3b".into(), "x".into())));
        assert_eq!(ShapeSpec::parse("").unwrap(), ShapeSpec::default());
        for bad in ["fn:a,a3b:b", "rows", "rows=1,rows=2"] {
            assert!(ShapeSpec::parse(bad).is_err(), "{bad}");
        }
        let mut m = ShapeSpec::parse("rows=3")
            .unwrap()
            .resolve(&[("rows", 9), ("K", 2048)]);
        assert_eq!(
            take_dim(&mut m, "rows").unwrap(),
            3,
            "pairs override the preset"
        );
        assert_eq!(take_dim(&mut m, "K").unwrap(), 2048);
        assert_eq!(take_dim_or(&mut m, "x", 7).unwrap(), 7);
        assert!(take_dim(&mut m, "missing").is_err());
        m.insert("zzz".into(), "1".into());
        assert!(no_leftover(&m).unwrap_err().contains("zzz"));
    }

    #[test]
    fn rounding_and_diff_helpers() {
        // bf16 RNE: 1 + 2^-8 is a tie between 1.0 (even) and 1 + 2^-7.
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        assert_eq!(bf16_val(bf16_bits(-2.5)), -2.5);
        assert!(bf16_val(bf16_bits(f32::NAN)).is_nan());
        let a = 1.0f32.to_le_bytes();
        let b = 3.0f32.to_le_bytes();
        assert_eq!(max_diff(Elem::F32, &a, &b), 2.0);
        assert_eq!(max_diff(Elem::F32, &a, &a), 0.0);
        assert_eq!(
            max_diff(Elem::F32, &a, &f32::NAN.to_le_bytes()),
            f64::INFINITY
        );
        assert_eq!(max_diff(Elem::U8, &[1, 5], &[1, 9]), 4.0);
        assert_eq!(
            elem_diff(Elem::Bf16, &[0x80, 0x3f], &[0x82, 0x3f]),
            2f64.powi(-6)
        );
    }

    #[test]
    fn generators_are_deterministic() {
        assert_eq!(f32_unit_bytes(5, 16, 2.0), f32_unit_bytes(5, 16, 2.0));
        assert_ne!(f32_unit_bytes(5, 16, 2.0), f32_unit_bytes(6, 16, 2.0));
        assert_eq!(bf16_normal_bytes(3, 64, 1.0).len(), 128);
        let mut x = [0f32; 128];
        x[0] = 7.0;
        x[1] = -3.5;
        let r = quantize_i4_block(&x);
        assert_eq!(f32::from_le_bytes(r[0..4].try_into().unwrap()), 1.0);
        assert_eq!(i32::from_le_bytes(r[4..8].try_into().unwrap()), 7 - 4);
        assert_eq!(r[8], 0x7 | (0xc << 4), "7 and round(-3.5) = -4 as nibbles");
    }

    #[test]
    fn shipped_object_requires_the_symbols() {
        let bytes = b"xx\0sym_a\0yy\0sym_b\0";
        let o = shipped_object(
            bytes,
            "k/a.hxaco",
            "f.rs:1",
            "sym_a",
            &["sym_a", "sym_b"],
            json!({}),
        )
        .unwrap();
        assert_eq!(o.exports, vec!["sym_a", "sym_b"]);
        assert_eq!(o.source["sha256"], sha256_hex(bytes));
        assert!(shipped_object(bytes, "k/a.hxaco", "f.rs:1", "sym_c", &[], json!({})).is_err());
    }
}
