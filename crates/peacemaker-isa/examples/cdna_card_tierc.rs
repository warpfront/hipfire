// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Card Tier C for the gfx942 PeaceMaker kernels: the FP8 activation producer, the FP8 x FP8 grouped MoE
//! gate/up tile and the QSA FP8 K/V row codec (encode/decode, Hkv 2 and 4).
//!
//! Every case runs the same inputs through up to five executions and compares **every output buffer
//! separately**, by bits:
//!
//! * `cardPM`  - the PeaceMaker code object (emitted as a v6 code object by `cdna::emit`) on the MI300X card;
//! * `cardHIP` - the HIP twin (`kernels/src/pm_cdna/<name>.gfx942.hip`, compiled to `<name>.hsaco`) on the card;
//! * `ref`     - the CPU reference of the kernel module, exact inputs;
//! * `simPM`   - the PeaceMaker bytes on the CPU wave64 emulator (`peacemaker_cdna::sim`);
//! * `simHIP`  - the HIP twin's code object on the emulator, when the emulator can decode and run it
//!               (otherwise an explicit "unavailable" reason is recorded, never a fake result).
//!
//! Comparisons: (a) cardPM/cardHIP byte identity, (b) cardPM/ref, (c) cardPM/simPM, (d) cardHIP/simHIP.
//! For the MoE tile (b) and (c) are evaluated under every `AccumModel` (Exact, Serial, Cdna3Fused, Grouped
//! 4/8/16/32) and the models that match bit for bit are listed; only the default model (`Cdna3Fused`,
//! `Config::default()`) decides the verdict, so a matching non-default model never masks a default failure.
//!
//! Modes:
//! * default (card): needs the HIP runtime and a gfx942 device 0 (any other arch is refused).
//! * `--dry-run`: no HIP library, no HIP call. The "card" columns are produced by a second, independent
//!   emulator run of the same code objects (PM, and HIP twin when emulable).
//! * `--quick`: small shapes, every kernel. Default is the full set (MoE R=1280 K=2560, act K=2560).
//!
//! ```text
//! cargo run --release -p peacemaker-isa --features lab --example cdna_card_tierc -- [--dry-run] [--quick] [--receipt PATH] [--hsaco-dir DIR]
//! ```
//!
//! Exit status: nonzero when any case has an (a), (b-default) or (c-default) mismatch, an execution error,
//! or the setup failed. (d) and non-default accumulation models are reported only. The receipt JSON
//! (`./cdna_card_tierc.json` unless `--receipt`) carries every comparison, seed, shape, hash, host and
//! runtime version, wall times and errors, and is rewritten after every case.

use hip_bridge::{DeviceBuffer, Function, HipRuntime, Module};
use peacemaker_cdna::sim::mfma::AccumModel;
use peacemaker_cdna::sim::{self, Config, KernelImage, Launch, Memory};
use peacemaker_isa::cdna::emit::{code_object, CodeObjectVersion};
use peacemaker_isa::cdna::Kernel;
use peacemaker_isa::kernels::cdna::{
    fp8_act_producer as act, fp8_moe_gate_up as moe, qsa_fp8_kv as qsa,
};
use peacemaker_isa::Arch;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const USAGE: &str = "usage: cdna_card_tierc [--dry-run] [--quick] [--receipt PATH] [--hsaco-dir DIR]\n\
  --dry-run       no HIP library/calls; the card columns are a second emulator run\n\
  --quick         small shapes (all kernels); default is the full set\n\
  --receipt PATH  receipt JSON (default ./cdna_card_tierc.json)\n\
  --hsaco-dir DIR directory of fp8_act_producer.hsaco, fp8_moe_gate_up.hsaco, qsa_fp8_kv.hsaco (default: beside the executable)";

/// Output buffers start with this byte (every byte a kernel must overwrite is visible).
const POISON: u8 = 0xee;
const KARG_ADDR: u64 = 0x100;
const BUF_BASE: u64 = 0x1000_0000;
/// `Config::default().accum`.
const DEFAULT_MODEL: AccumModel = AccumModel::Cdna3Fused;
const EXPECTED_ARCH: &str = "gfx942";

const MOD_ACT: &str = "fp8_act_producer";
const MOD_MOE: &str = "fp8_moe_gate_up";
const MOD_QSA: &str = "qsa_fp8_kv";

const K_ACT: usize = 0;
const K_MOE: usize = 1;
fn k_enc(h: usize) -> usize {
    if h == 2 {
        2
    } else {
        3
    }
}
fn k_dec(h: usize) -> usize {
    if h == 2 {
        4
    } else {
        5
    }
}

// ----------------------------------------------------------------------------
// Small helpers

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b)[..])
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}
fn flat<T>(r: std::thread::Result<Result<T, String>>) -> Result<T, String> {
    match r {
        Ok(v) => v,
        Err(p) => Err(format!("panic: {}", panic_msg(p))),
    }
}
fn model_name(m: AccumModel) -> String {
    format!("{m:?}")
}
fn one_line(s: &str) -> String {
    s.replace('\n', " ")
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn idx(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn mantissa(&mut self) -> f64 {
        1.0 + (self.next() >> 41) as f64 / (1u64 << 23) as f64
    }
    fn sign(&mut self) -> f64 {
        if self.next() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in (-1, 1).
    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
    /// Roughly normal, std ~1.15.
    fn normal(&mut self) -> f32 {
        self.signed() + self.signed() + self.signed() + self.signed()
    }
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}
fn pow2_f64(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}
fn pow2_f32(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}
fn up(f: f32) -> f32 {
    f32::from_bits(f.to_bits() + 1)
}
fn down(f: f32) -> f32 {
    f32::from_bits(f.to_bits() - 1)
}

// ----------------------------------------------------------------------------
// Launch description shared by every backend

#[derive(Clone)]
struct Buf {
    name: &'static str,
    data: Vec<u8>,
    output: bool,
}

#[derive(Clone, Copy)]
enum Val {
    Ptr(usize),
    U32(u32),
    I32(i32),
    U64(u64),
}

/// Buffers, kernarg layout and geometry of one launch; the same description is bound for every backend.
#[derive(Clone)]
struct Spec {
    grid: [u32; 3],
    block: [u32; 3],
    bufs: Vec<Buf>,
    args: Vec<(usize, Val)>,
    kernarg_bytes: usize,
}

impl Spec {
    /// Kernarg image with pointer arguments taken from `addrs` (one address per buffer).
    fn kernarg(&self, addrs: &[u64]) -> Vec<u8> {
        let mut k = vec![0u8; self.kernarg_bytes];
        for &(off, v) in &self.args {
            match v {
                Val::Ptr(i) => k[off..off + 8].copy_from_slice(&addrs[i].to_le_bytes()),
                Val::U64(x) => k[off..off + 8].copy_from_slice(&x.to_le_bytes()),
                Val::U32(x) => k[off..off + 4].copy_from_slice(&x.to_le_bytes()),
                Val::I32(x) => k[off..off + 4].copy_from_slice(&x.to_le_bytes()),
            }
        }
        k
    }
    /// Emulator addresses: page aligned with an unmapped page between buffers so an overrun faults.
    fn sim_addrs(&self) -> Vec<u64> {
        let mut a = BUF_BASE;
        let mut v = Vec::new();
        for b in &self.bufs {
            v.push(a);
            a += (b.data.len() as u64).div_ceil(0x1000) * 0x1000 + 0x1000;
        }
        v
    }
}

fn out_buf(name: &'static str, len: usize) -> Buf {
    Buf {
        name,
        data: vec![POISON; len],
        output: true,
    }
}
fn in_buf(name: &'static str, data: Vec<u8>) -> Buf {
    Buf {
        name,
        data,
        output: false,
    }
}

/// Output buffers' contents in buffer order.
type Outs = Vec<Vec<u8>>;

#[derive(Clone, Copy)]
enum Kind {
    F32 { cols: usize },
    Ocp8 { cols: usize },
    QsaRow { heads: usize },
}
impl Kind {
    fn esz(self) -> usize {
        match self {
            Kind::F32 { .. } => 4,
            _ => 1,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Kind::F32 { .. } => "f32",
            Kind::Ocp8 { .. } => "ocp-e4m3 codes",
            Kind::QsaRow { .. } => "qsa row (codes + f16 scales)",
        }
    }
}

struct OutSpec {
    name: &'static str,
    kind: Kind,
}

enum Refs {
    Fixed(Vec<Vec<u8>>),
    PerModel(Box<dyn Fn(AccumModel) -> Result<Vec<Vec<u8>>, String> + Send + Sync>),
}

struct Case {
    kernel: usize,
    name: String,
    seed: u64,
    shape: Value,
    spec: Spec,
    outs: Vec<OutSpec>,
    /// `models[0]` is the default model; non-MoE cases have exactly one.
    models: Vec<AccumModel>,
    refs: Refs,
}
impl Case {
    fn reference(&self, m: AccumModel) -> Result<Vec<Vec<u8>>, String> {
        match &self.refs {
            Refs::Fixed(v) => Ok(v.clone()),
            Refs::PerModel(f) => f(m),
        }
    }
}

struct Maker {
    name: String,
    make: Box<dyn Fn() -> Result<Case, String>>,
}
fn mk(name: impl Into<String>, f: impl Fn() -> Result<Case, String> + 'static) -> Maker {
    Maker {
        name: name.into(),
        make: Box::new(f),
    }
}

// ----------------------------------------------------------------------------
// Bit comparison

fn ocp8(b: u8) -> String {
    if b & 0x7f == 0x7f {
        return "NaN".into();
    }
    let (e, m) = (i32::from((b >> 3) & 15), f64::from(b & 7));
    let mag = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    format!("{:e}", if b & 0x80 != 0 { -mag } else { mag })
}

fn ulp(a: f32, b: f32) -> u64 {
    let key = |v: f32| {
        let x = v.to_bits();
        if x >> 31 == 1 {
            !x
        } else {
            x | 0x8000_0000
        }
    };
    (i64::from(key(a)) - i64::from(key(b))).unsigned_abs()
}

/// One mismatching element: index in hex, both raw values and their decoded meaning.
fn elem_json(kind: Kind, i: usize, a: Option<&[u8]>, b: Option<&[u8]>) -> Value {
    let show = |e: Option<&[u8]>| -> (String, String) {
        match e {
            None => ("missing".to_string(), "missing".to_string()),
            Some(s) => match kind {
                Kind::F32 { .. } => {
                    let bits = u32::from_le_bytes(s.try_into().unwrap());
                    (
                        format!("{bits:#010x}"),
                        format!("{:?}", f32::from_bits(bits)),
                    )
                }
                Kind::Ocp8 { .. } => (format!("{:#04x}", s[0]), ocp8(s[0])),
                Kind::QsaRow { heads } => {
                    let at = i % (heads * (qsa::HEAD_DIM + 2));
                    (
                        format!("{:#04x}", s[0]),
                        if at < heads * qsa::HEAD_DIM {
                            ocp8(s[0])
                        } else {
                            "f16 scale byte".to_string()
                        },
                    )
                }
            },
        }
    };
    let (ah, av) = show(a);
    let (bh, bv) = show(b);
    let mut m = Map::new();
    m.insert("index".into(), json!(format!("{i:#x}")));
    match kind {
        Kind::F32 { cols } | Kind::Ocp8 { cols } => {
            m.insert("row".into(), json!(i / cols));
            m.insert("col".into(), json!(i % cols));
        }
        Kind::QsaRow { heads } => {
            let rb = heads * (qsa::HEAD_DIM + 2);
            let at = i % rb;
            m.insert("row".into(), json!(i / rb));
            m.insert(
                "field".into(),
                json!(if at < heads * qsa::HEAD_DIM {
                    format!("head {} channel {}", at / qsa::HEAD_DIM, at % qsa::HEAD_DIM)
                } else {
                    format!("scale byte {}", at - heads * qsa::HEAD_DIM)
                }),
            );
        }
    }
    m.insert("lhs_hex".into(), json!(ah));
    m.insert("rhs_hex".into(), json!(bh));
    m.insert("lhs".into(), json!(av));
    m.insert("rhs".into(), json!(bv));
    Value::Object(m)
}

/// Bitwise comparison of two buffers. Floats are compared by bits; the max ULP is over finite pairs and
/// pairs involving NaN / Inf are counted separately.
fn compare(kind: Kind, a: &[u8], b: &[u8]) -> (bool, Value) {
    let esz = kind.esz();
    let total = a.len().max(b.len());
    let n = total.div_ceil(esz);
    let common = a.len().min(b.len());
    let mut mismatch_bytes = total - common;
    for i in 0..common {
        if a[i] != b[i] {
            mismatch_bytes += 1
        }
    }
    let (mut mismatch_elems, mut max_ulp, mut nan, mut inf) = (0usize, 0u64, 0usize, 0usize);
    let mut first = Vec::new();
    for i in 0..n {
        let ea = a.get(i * esz..(i + 1) * esz);
        let eb = b.get(i * esz..(i + 1) * esz);
        if ea.is_some() && ea == eb {
            continue;
        }
        mismatch_elems += 1;
        if first.len() < 8 {
            first.push(elem_json(kind, i, ea, eb))
        }
        if let (Kind::F32 { .. }, Some(x), Some(y)) = (kind, ea, eb) {
            let xf = f32::from_bits(u32::from_le_bytes(x.try_into().unwrap()));
            let yf = f32::from_bits(u32::from_le_bytes(y.try_into().unwrap()));
            if xf.is_nan() || yf.is_nan() {
                nan += 1
            } else if xf.is_infinite() || yf.is_infinite() {
                inf += 1
            } else {
                max_ulp = max_ulp.max(ulp(xf, yf))
            }
        }
    }
    let identical = mismatch_elems == 0 && mismatch_bytes == 0;
    let is_f32 = matches!(kind, Kind::F32 { .. });
    let v = json!({
        "status": "compared",
        "identical": identical,
        "kind": kind.label(),
        "elements": n,
        "bytes": total,
        "length_lhs": a.len(),
        "length_rhs": b.len(),
        "mismatch_elements": mismatch_elems,
        "mismatch_bytes": mismatch_bytes,
        "first_mismatches": first,
        "max_ulp_finite": if is_f32 { json!(max_ulp) } else { Value::Null },
        "nan_mismatch_elements": if is_f32 { json!(nan) } else { Value::Null },
        "inf_mismatch_elements": if is_f32 { json!(inf) } else { Value::Null },
    });
    (identical, v)
}

fn pick(r: &Result<Outs, String>, j: usize) -> Result<&[u8], String> {
    match r {
        Ok(o) => o
            .get(j)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("output {j} missing")),
        Err(e) => Err(e.clone()),
    }
}

/// `(Some(identical), details)` or `(None, unavailable + reason)`.
fn cmp(kind: Kind, x: Result<&[u8], String>, y: Result<&[u8], String>) -> (Option<bool>, Value) {
    match (x, y) {
        (Ok(a), Ok(b)) => {
            let (id, v) = compare(kind, a, b);
            (Some(id), v)
        }
        (x, y) => {
            let why: Vec<String> = [x.err(), y.err()].into_iter().flatten().collect();
            (
                None,
                json!({ "status": "unavailable", "reason": why.join("; ") }),
            )
        }
    }
}

fn agg(v: &[Option<bool>]) -> Option<bool> {
    if v.iter().any(|x| x.is_none()) {
        return None;
    }
    Some(v.iter().all(|x| *x == Some(true)))
}
fn st(o: Option<bool>) -> &'static str {
    match o {
        Some(true) => "match",
        Some(false) => "MISMATCH",
        None => "n/a",
    }
}
fn opt_json(o: Option<bool>) -> Value {
    match o {
        Some(b) => json!(b),
        None => Value::Null,
    }
}

// ----------------------------------------------------------------------------
// Backends

fn run_sim(image: &KernelImage, spec: &Spec, accum: AccumModel) -> Result<Outs, String> {
    let addrs = spec.sim_addrs();
    let mut mem = Memory::default();
    for (b, &a) in spec.bufs.iter().zip(&addrs) {
        let r = if b.output {
            mem.map(b.name, a, b.data.clone())
        } else {
            mem.map_readonly(b.name, a, b.data.clone())
        };
        r.map_err(|e| format!("map {}: {e}", b.name))?;
    }
    let launch = Launch {
        grid: spec.grid,
        block: spec.block,
        kernarg: spec.kernarg(&addrs),
        kernarg_addr: KARG_ADDR,
        dynamic_lds: 0,
    };
    let cfg = Config {
        accum,
        ..Config::default()
    };
    sim::run(image, &launch, &mut mem, &cfg).map_err(|e| e.to_string())?;
    spec.bufs
        .iter()
        .filter(|b| b.output)
        .map(|b| {
            mem.region(b.name)
                .map(|r| r.bytes.clone())
                .ok_or_else(|| format!("region {} missing", b.name))
        })
        .collect()
}

/// Fresh device allocations (poisoned outputs), one launch, readback, free.
fn run_card(rt: &HipRuntime, f: &Function, spec: &Spec) -> Result<Outs, String> {
    let mut bufs: Vec<DeviceBuffer> = Vec::new();
    let result = (|| -> Result<Outs, String> {
        for b in &spec.bufs {
            let d = rt
                .malloc(b.data.len().max(1))
                .map_err(|e| format!("hipMalloc({}, {} B): {e}", b.name, b.data.len()))?;
            bufs.push(d);
            rt.memcpy_htod(&bufs[bufs.len() - 1], &b.data)
                .map_err(|e| format!("upload {}: {e}", b.name))?;
        }
        let addrs: Vec<u64> = bufs.iter().map(|d| d.as_ptr() as usize as u64).collect();
        let mut blob = spec.kernarg(&addrs);
        // SAFETY: the blob is the kernel's full explicit kernarg image (layout asserted at kernel setup) and
        // every pointer in it is a live hipMalloc allocation of this call.
        let launched =
            unsafe { rt.launch_kernel_blob(f, spec.grid, spec.block, 0, None, &mut blob) };
        launched.map_err(|e| format!("launch: {e}"))?;
        rt.device_synchronize()
            .map_err(|e| format!("synchronize: {e}"))?;
        let mut outs = Vec::new();
        for (b, d) in spec.bufs.iter().zip(&bufs) {
            if b.output {
                let mut v = vec![0u8; b.data.len()];
                rt.memcpy_dtoh(&mut v, d)
                    .map_err(|e| format!("readback {}: {e}", b.name))?;
                outs.push(v);
            }
        }
        Ok(outs)
    })();
    let mut free_err = None;
    for d in bufs {
        if let Err(e) = rt.free(d) {
            free_err = Some(format!("hipFree: {e}"))
        }
    }
    match (result, free_err) {
        (Ok(_), Some(e)) => Err(e),
        (r, _) => r,
    }
}

// ----------------------------------------------------------------------------
// Kernel table

struct PmImage {
    elf: Vec<u8>,
    image: KernelImage,
    words: usize,
}
struct KernelEntry {
    module: &'static str,
    label: String,
    pm_symbol: String,
    hip_symbol: String,
    pm: Result<PmImage, String>,
    /// The HIP twin decoded for the emulator (Err: reason it is unavailable).
    hip: Result<KernelImage, String>,
}
struct Hsaco {
    name: &'static str,
    path: PathBuf,
    bytes: Result<Vec<u8>, String>,
}

fn build_kernels(hsaco_dir: &std::path::Path) -> (Vec<KernelEntry>, Vec<Hsaco>) {
    let hsacos: Vec<Hsaco> = [MOD_ACT, MOD_MOE, MOD_QSA]
        .iter()
        .map(|&name| {
            let path = hsaco_dir.join(format!("{name}.hsaco"));
            let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()));
            Hsaco { name, path, bytes }
        })
        .collect();
    // (module, label, PM symbol, HIP symbol, built kernel, expected kernarg bytes)
    let hip_sym = |pm: &str| pm.replacen("_pm_gfx942", "_gfx942", 1);
    let mut defs: Vec<(
        &'static str,
        String,
        String,
        String,
        Result<Kernel, String>,
        u32,
    )> = vec![
        (
            MOD_ACT,
            "act_producer".into(),
            act::SYMBOL.into(),
            act::SYMBOL.into(),
            act::build(Arch::Gfx942),
            48,
        ),
        (
            MOD_MOE,
            "moe_gate_up".into(),
            moe::SYMBOL.into(),
            moe::HIP_SYMBOL.into(),
            moe::build(Arch::Gfx942, moe::Shape { r: 1280, k: 2560 }),
            moe::KERNARG_BYTES,
        ),
    ];
    for h in [2usize, 4] {
        let s = qsa::encode_symbol(h);
        defs.push((
            MOD_QSA,
            format!("qsa_encode_h{h}"),
            s.clone(),
            hip_sym(&s),
            qsa::build_encode(Arch::Gfx942, h),
            16,
        ));
    }
    for h in [2usize, 4] {
        let s = qsa::decode_symbol(h);
        defs.push((
            MOD_QSA,
            format!("qsa_decode_h{h}"),
            s.clone(),
            hip_sym(&s),
            qsa::build_decode(Arch::Gfx942, h),
            16,
        ));
    }
    let entries = defs
        .into_iter()
        .map(|(module, label, pm_symbol, hip_symbol, built, kernarg)| {
            let pm = built.and_then(|k| {
                if k.resources.kernarg_size != kernarg {
                    return Err(format!(
                        "{}: kernarg size {} != the harness layout {kernarg}",
                        k.name, k.resources.kernarg_size
                    ));
                }
                let elf = code_object(&[&k], CodeObjectVersion::V6)?;
                let image = sim::load_code_object(&elf, &pm_symbol)
                    .map_err(|e| format!("load_code_object({pm_symbol}): {e}"))?;
                Ok(PmImage {
                    elf,
                    image,
                    words: k.words.len(),
                })
            });
            let hs = hsacos
                .iter()
                .find(|h| h.name == module)
                .expect("hsaco table");
            let hip = match &hs.bytes {
                Err(e) => Err(e.clone()),
                Ok(b) => sim::load_code_object(b, &hip_symbol)
                    .map_err(|e| format!("emulator cannot load the HIP twin `{hip_symbol}`: {e}")),
            };
            KernelEntry {
                module,
                label,
                pm_symbol,
                hip_symbol,
                pm,
                hip,
            }
        })
        .collect();
    (entries, hsacos)
}

struct CardKernel {
    pm: Result<Function, String>,
    hip: Result<Function, String>,
}
struct Card {
    rt: HipRuntime,
    kernels: Vec<CardKernel>,
    _modules: Vec<Module>,
}

fn setup_card(
    kernels: &[KernelEntry],
    hsacos: &[Hsaco],
    rec: &mut Receipt,
) -> Result<Card, String> {
    let rt = HipRuntime::load().map_err(|e| format!("HipRuntime::load: {e}"))?;
    let count = rt
        .device_count()
        .map_err(|e| format!("hipGetDeviceCount: {e}"))?;
    let mut rt_info = Map::new();
    rt_info.insert("device_count".into(), json!(count));
    if count < 1 {
        rec.top.insert("runtime".into(), Value::Object(rt_info));
        return Err("no HIP device".into());
    }
    rt.set_device(0)
        .map_err(|e| format!("hipSetDevice(0): {e}"))?;
    let arch = rt
        .get_arch(0)
        .map_err(|e| format!("hipGetDeviceProperties(0): {e}"))?;
    rt_info.insert("device".into(), json!(0));
    rt_info.insert("device_arch".into(), json!(arch));
    let version = rt.runtime_version();
    match &version {
        Ok((maj, min)) => {
            rt_info.insert("hip_runtime_version".into(), json!(format!("{maj}.{min}")));
            println!("HIP runtime version {maj}.{min}, device 0 arch {arch}");
        }
        Err(e) => {
            rt_info.insert("hip_runtime_version_error".into(), json!(e.to_string()));
            println!("HIP runtime version unavailable ({e}), device 0 arch {arch}");
        }
    }
    rec.top.insert("runtime".into(), Value::Object(rt_info));
    if arch != EXPECTED_ARCH {
        return Err(format!(
            "device 0 reports `{arch}`, this harness runs only on {EXPECTED_ARCH}: refusing"
        ));
    }
    version.map_err(|e| format!("hipRuntimeGetVersion: {e}"))?;

    let mut modules: Vec<Module> = Vec::new();
    let mut cache: HashMap<&'static str, Result<usize, String>> = HashMap::new();
    let mut ck = Vec::new();
    for k in kernels {
        let pm = match &k.pm {
            Err(e) => Err(format!("PM kernel not built: {e}")),
            Ok(p) => rt
                .module_load_data(&p.elf)
                .map_err(|e| format!("hipModuleLoadData(PM {}): {e}", k.pm_symbol))
                .and_then(|m| {
                    let f = rt
                        .module_get_function(&m, &k.pm_symbol)
                        .map_err(|e| format!("hipModuleGetFunction({}): {e}", k.pm_symbol));
                    modules.push(m);
                    f
                }),
        };
        let slot = cache
            .entry(k.module)
            .or_insert_with(|| {
                let hs = hsacos
                    .iter()
                    .find(|h| h.name == k.module)
                    .expect("hsaco table");
                match &hs.bytes {
                    Err(e) => Err(e.clone()),
                    Ok(b) => rt
                        .module_load_data(b)
                        .map(|m| {
                            modules.push(m);
                            modules.len() - 1
                        })
                        .map_err(|e| format!("hipModuleLoadData({}): {e}", hs.path.display())),
                }
            })
            .clone();
        let hip = match slot {
            Err(e) => Err(e),
            Ok(i) => rt
                .module_get_function(&modules[i], &k.hip_symbol)
                .map_err(|e| format!("hipModuleGetFunction({}): {e}", k.hip_symbol)),
        };
        ck.push(CardKernel { pm, hip });
    }
    Ok(Card {
        rt,
        kernels: ck,
        _modules: modules,
    })
}

// ----------------------------------------------------------------------------
// Receipt

struct Receipt {
    path: PathBuf,
    t0: Instant,
    top: Map<String, Value>,
    cases: Vec<Value>,
    errors: Vec<String>,
    pass: usize,
    fail: usize,
}
impl Receipt {
    fn flush(&self, complete: bool) {
        let mut m = self.top.clone();
        m.insert("cases".into(), Value::Array(self.cases.clone()));
        m.insert("errors".into(), json!(self.errors));
        m.insert("summary".into(), json!({ "cases": self.pass + self.fail, "pass": self.pass, "fail": self.fail, "setup_or_run_errors": self.errors.len() }));
        m.insert(
            "wall_seconds".into(),
            json!(self.t0.elapsed().as_secs_f64()),
        );
        m.insert("complete".into(), json!(complete));
        m.insert(
            "exit_ok".into(),
            json!(complete && self.fail == 0 && self.errors.is_empty()),
        );
        match serde_json::to_string_pretty(&Value::Object(m)) {
            Ok(s) => {
                if let Err(e) = std::fs::write(&self.path, s) {
                    eprintln!("error: writing receipt {}: {e}", self.path.display())
                }
            }
            Err(e) => eprintln!("error: serialising receipt: {e}"),
        }
    }
    fn error(&mut self, e: String) {
        eprintln!("error: {e}");
        self.errors.push(e);
    }
}

fn tool_version(path: &str) -> Value {
    match Command::new(path).arg("--version").output() {
        Ok(o) => json!({
            "command": format!("{path} --version"), "ok": o.status.success(), "exit_code": o.status.code(),
            "stdout": String::from_utf8_lossy(&o.stdout), "stderr": String::from_utf8_lossy(&o.stderr),
        }),
        Err(e) => {
            json!({ "command": format!("{path} --version"), "ok": false, "error": e.to_string() })
        }
    }
}

fn host_info() -> Value {
    let read = |p: &str| {
        std::fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string())
    };
    let cpu = read("/proc/cpuinfo").and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split_once(':'))
            .map(|(_, v)| v.trim().to_string())
    });
    json!({
        "hostname": read("/proc/sys/kernel/hostname").or_else(|| std::env::var("HOSTNAME").ok()),
        "kernel": read("/proc/sys/kernel/osrelease"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu_model": cpu,
        "logical_cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
        "executable": std::env::current_exe().ok().map(|p| p.display().to_string()),
        "cwd": std::env::current_dir().ok().map(|p| p.display().to_string()),
        "user": std::env::var("USER").ok(),
        "env": {
            "HIP_VISIBLE_DEVICES": std::env::var("HIP_VISIBLE_DEVICES").ok(),
            "ROCR_VISIBLE_DEVICES": std::env::var("ROCR_VISIBLE_DEVICES").ok(),
            "HSA_OVERRIDE_GFX_VERSION": std::env::var("HSA_OVERRIDE_GFX_VERSION").ok(),
        },
        "unix_time": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).ok(),
    })
}

// ----------------------------------------------------------------------------
// Case execution and verdict

struct Env<'a> {
    dry: bool,
    kernels: &'a [KernelEntry],
    card: Option<&'a Card>,
    /// Kernels whose HIP twin failed to emulate: the reason (skipped afterwards).
    hip_off: RefCell<Vec<Option<String>>>,
}

fn base_json(env: &Env, case: &Case) -> Map<String, Value> {
    let ke = &env.kernels[case.kernel];
    let mut m = Map::new();
    m.insert("kernel".into(), json!(ke.label));
    m.insert("module".into(), json!(ke.module));
    m.insert("pm_symbol".into(), json!(ke.pm_symbol));
    m.insert("hip_symbol".into(), json!(ke.hip_symbol));
    m.insert("case".into(), json!(case.name));
    m.insert("seed".into(), json!(format!("{:#x}", case.seed)));
    m.insert("shape".into(), case.shape.clone());
    m.insert("grid".into(), json!(case.spec.grid));
    m.insert("block".into(), json!(case.spec.block));
    m.insert("kernarg_bytes".into(), json!(case.spec.kernarg_bytes));
    m.insert("poison_byte".into(), json!(format!("{POISON:#04x}")));
    let bufs: Vec<Value> = case.spec.bufs.iter().map(|b| json!({
        "name": b.name, "bytes": b.data.len(), "output": b.output, "sha256": if b.output { Value::Null } else { json!(sha(&b.data)) },
    })).collect();
    m.insert("buffers".into(), Value::Array(bufs));
    m.insert(
        "models".into(),
        json!(case
            .models
            .iter()
            .map(|&x| model_name(x))
            .collect::<Vec<_>>()),
    );
    m
}

fn fail_case(env: &Env, case: &Case, errors: Vec<String>, t0: Instant) -> (bool, String, Value) {
    let mut m = base_json(env, case);
    m.insert("status".into(), json!("FAIL"));
    m.insert("errors".into(), json!(errors));
    m.insert("wall_seconds".into(), json!(t0.elapsed().as_secs_f64()));
    let line = format!(
        "FAIL {} [{}]: {}",
        env.kernels[case.kernel].label,
        case.name,
        one_line(&errors.join(" | "))
    );
    (false, line, Value::Object(m))
}

type RunRes = Result<Outs, String>;

fn run_case(env: &Env, case: &Case) -> (bool, String, Value) {
    let t0 = Instant::now();
    let ke = &env.kernels[case.kernel];
    let dry = env.dry;
    let pmimg = match &ke.pm {
        Ok(p) => p,
        Err(e) => return fail_case(env, case, vec![format!("PM kernel unavailable: {e}")], t0),
    };
    let spec = &case.spec;
    let mut errors: Vec<String> = Vec::new();

    let hip_reason: Option<String> = match &ke.hip {
        Err(e) => Some(format!("HIP twin unavailable in the emulator: {e}")),
        Ok(_) => env.hip_off.borrow()[case.kernel].clone(),
    };
    let hip_img: Option<&KernelImage> = if hip_reason.is_none() {
        ke.hip.as_ref().ok()
    } else {
        None
    };

    let (model_runs, p2, h1, h2, cpm, chip) = std::thread::scope(|s| {
        let mh: Vec<_> = case
            .models
            .iter()
            .map(|&m| s.spawn(move || (case.reference(m), run_sim(&pmimg.image, spec, m))))
            .collect();
        let h1 = hip_img.map(|img| s.spawn(move || run_sim(img, spec, DEFAULT_MODEL)));
        let h2 = if dry {
            hip_img.map(|img| s.spawn(move || run_sim(img, spec, DEFAULT_MODEL)))
        } else {
            None
        };
        let p2 = if dry {
            Some(s.spawn(move || run_sim(&pmimg.image, spec, DEFAULT_MODEL)))
        } else {
            None
        };
        // The card runs on this thread while the emulator threads work.
        let (cpm, chip): (Option<RunRes>, Option<RunRes>) = match env.card {
            Some(card) => {
                let ck = &card.kernels[case.kernel];
                let pm = match &ck.pm {
                    Ok(f) => run_card(&card.rt, f, spec),
                    Err(e) => Err(e.clone()),
                };
                let hp = match &ck.hip {
                    Ok(f) => run_card(&card.rt, f, spec),
                    Err(e) => Err(e.clone()),
                };
                (Some(pm), Some(hp))
            }
            None => (None, None),
        };
        let model_runs: Vec<(Result<Vec<Vec<u8>>, String>, RunRes)> = mh
            .into_iter()
            .map(|h| match h.join() {
                Ok(v) => v,
                Err(p) => {
                    let e = format!("panic: {}", panic_msg(p));
                    (Err(e.clone()), Err(e))
                }
            })
            .collect();
        (
            model_runs,
            p2.map(|h| flat(h.join())),
            h1.map(|h| flat(h.join())),
            h2.map(|h| flat(h.join())),
            cpm,
            chip,
        )
    });

    if let Some(Err(e)) = &h1 {
        env.hip_off.borrow_mut()[case.kernel] =
            Some(format!("HIP twin emulation failed on `{}`: {e}", case.name));
    }
    let sim_hip: RunRes = match h1 {
        Some(Ok(o)) => Ok(o),
        Some(Err(e)) => Err(format!("HIP twin emulation failed: {e}")),
        None => Err(hip_reason
            .clone()
            .unwrap_or_else(|| "HIP twin emulation not attempted".into())),
    };
    // In dry-run the synthetic card columns must exercise comparison (a)
    // even when no HIP object is present/decodable. This is explicitly PM
    // emulation, never evidence that the HIP twin ran.
    let synthetic_hip = dry && !matches!(&h2, Some(Ok(_)));
    let (card_pm, card_hip): (RunRes, RunRes) = if dry {
        let pm = p2.unwrap_or_else(|| Err("second emulator run missing".into()));
        let hip = match h2 {
            Some(Ok(o)) => Ok(o),
            _ => pm.clone(),
        };
        (pm, hip)
    } else {
        (
            cpm.unwrap_or_else(|| Err("card run missing".into())),
            chip.unwrap_or_else(|| Err("card run missing".into())),
        )
    };
    let (pm_what, hip_what) = if dry {
        (
            "second emulator run of PM (dry-run card column)",
            "second emulator run of the HIP twin (dry-run card column)",
        )
    } else {
        ("cardPM", "cardHIP")
    };
    if let Err(e) = &card_pm {
        errors.push(format!("{pm_what}: {e}"))
    }
    if !dry {
        if let Err(e) = &card_hip {
            errors.push(format!("{hip_what}: {e}"))
        }
    }

    let (refs, sims): (Vec<Result<Vec<Vec<u8>>, String>>, Vec<RunRes>) =
        model_runs.into_iter().unzip();
    if let Err(e) = &refs[0] {
        errors.push(format!(
            "CPU reference ({}): {e}",
            model_name(case.models[0])
        ))
    }
    if let Err(e) = &sims[0] {
        errors.push(format!("simPM ({}): {e}", model_name(case.models[0])))
    }

    let nm = case.models.len();
    let mut a_vec: Vec<Option<bool>> = Vec::new();
    let mut d_vec: Vec<Option<bool>> = Vec::new();
    let mut b_vec: Vec<Vec<Option<bool>>> = vec![Vec::new(); nm];
    let mut c_vec: Vec<Vec<Option<bool>>> = vec![Vec::new(); nm];
    let mut outs_json = Vec::new();
    for (j, o) in case.outs.iter().enumerate() {
        let kind = o.kind;
        let (a, a_j) = cmp(kind, pick(&card_pm, j), pick(&card_hip, j));
        let (d, d_j) = cmp(kind, pick(&card_hip, j), pick(&sim_hip, j));
        a_vec.push(a);
        d_vec.push(d);
        let (mut b_js, mut c_js) = (Vec::new(), Vec::new());
        for (mi, &m) in case.models.iter().enumerate() {
            let want: Result<&[u8], String> = match &refs[mi] {
                Ok(v) => v
                    .get(j)
                    .map(|x| x.as_slice())
                    .ok_or_else(|| format!("reference output {j} missing")),
                Err(e) => Err(e.clone()),
            };
            let (b, mut b_j) = cmp(kind, pick(&card_pm, j), want);
            let (c, mut c_j) = cmp(kind, pick(&card_pm, j), pick(&sims[mi], j));
            for (v, jj) in [(&mut b_j, &b), (&mut c_j, &c)] {
                if let Value::Object(mm) = v {
                    mm.insert("model".into(), json!(model_name(m)));
                    mm.insert("default_model".into(), json!(mi == 0));
                    mm.insert("verdict".into(), json!(st(*jj)));
                }
            }
            b_vec[mi].push(b);
            c_vec[mi].push(c);
            b_js.push(b_j);
            c_js.push(c_j);
        }
        let hash = |r: &RunRes| pick(r, j).ok().map(sha);
        outs_json.push(json!({
            "output": o.name,
            "kind": kind.label(),
            "a_cardPM_vs_cardHIP": a_j,
            "b_cardPM_vs_cpu_ref": b_js,
            "c_cardPM_vs_simPM": c_js,
            "d_cardHIP_vs_simHIP": d_j,
            "sha256": {
                "cardPM": hash(&card_pm), "cardHIP": hash(&card_hip), "simPM_default": hash(&sims[0]), "simHIP": hash(&sim_hip),
                "cpu_ref_default": refs[0].as_ref().ok().and_then(|v| v.get(j)).map(|x| sha(x)),
            },
        }));
    }
    let (a, d) = (agg(&a_vec), agg(&d_vec));
    let b_all: Vec<Option<bool>> = b_vec.iter().map(|v| agg(v)).collect();
    let c_all: Vec<Option<bool>> = c_vec.iter().map(|v| agg(v)).collect();
    let (b0, c0) = (b_all[0], c_all[0]);
    let pass = errors.is_empty() && a != Some(false) && b0 == Some(true) && c0 == Some(true);

    let models_json: Vec<Value> = case.models.iter().enumerate().map(|(mi, &m)| json!({
        "model": model_name(m), "default": mi == 0,
        "b_cardPM_matches_cpu_ref": opt_json(b_all[mi]), "c_cardPM_matches_simPM": opt_json(c_all[mi]),
        "reference_error": refs[mi].as_ref().err(), "simPM_error": sims[mi].as_ref().err(),
    })).collect();
    let exact = |v: &[Option<bool>]| -> String {
        case.models
            .iter()
            .zip(v)
            .filter(|(_, r)| **r == Some(true))
            .map(|(m, _)| model_name(*m))
            .collect::<Vec<_>>()
            .join(",")
    };

    let mut line = format!(
        "{} {} [{}]: a={} b={} c={} d={}",
        if pass { "PASS" } else { "FAIL" },
        ke.label,
        case.name,
        st(a),
        st(b0),
        st(c0),
        st(d)
    );
    if nm > 1 {
        line.push_str(&format!(
            " b_exact_models=[{}] c_exact_models=[{}]",
            exact(&b_all),
            exact(&c_all)
        ))
    }
    if let Some(e) = errors.first() {
        line.push_str(&format!(
            " error: {}{}",
            one_line(e),
            if errors.len() > 1 {
                format!(" (+{} more)", errors.len() - 1)
            } else {
                String::new()
            }
        ))
    }

    let mut m = base_json(env, case);
    m.insert("status".into(), json!(if pass { "PASS" } else { "FAIL" }));
    m.insert("a_cardPM_vs_cardHIP".into(), opt_json(a));
    m.insert("b_default_cardPM_vs_cpu_ref".into(), opt_json(b0));
    m.insert("c_default_cardPM_vs_simPM".into(), opt_json(c0));
    m.insert("d_cardHIP_vs_simHIP".into(), opt_json(d));
    m.insert(
        "hip_emulation".into(),
        match &sim_hip {
            Ok(_) => json!({ "available": true }),
            Err(e) => json!({ "available": false, "reason": e }),
        },
    );
    m.insert(
        "card_columns".into(),
        json!(if dry {
            "dry-run: second emulator run"
        } else {
            "hardware"
        }),
    );
    m.insert(
        "cardHIP_source".into(),
        json!(if synthetic_hip {
            "synthetic second PM emulator run (dry-run only; HIP twin unavailable)"
        } else if dry {
            "second HIP twin emulator run"
        } else {
            "hardware HIP twin"
        }),
    );
    m.insert("model_matches".into(), Value::Array(models_json));
    m.insert("outputs".into(), Value::Array(outs_json));
    m.insert("errors".into(), json!(errors));
    m.insert("wall_seconds".into(), json!(t0.elapsed().as_secs_f64()));
    (pass, line, Value::Object(m))
}

// ----------------------------------------------------------------------------
// Activation producer cases

fn act_fits(row: &[f32], k: usize, mode: u32, shift: i32) -> bool {
    match act::reference(row, k, mode, shift) {
        Ok(r) => {
            let a = f64::from(r.row_scales[0]);
            row.iter().all(|v| f64::from(v.abs()) / a <= 448.0)
        }
        Err(_) => false,
    }
}

fn ocp_value(b: u8) -> f64 {
    let (e, m) = (i32::from((b >> 3) & 15), f64::from(b & 7));
    if e == 0 {
        m * pow2_f64(-9)
    } else {
        (1.0 + m / 8.0) * pow2_f64(e - 7)
    }
}

/// Midpoints between adjacent OCP codes (every rounding tie) and the f32 neighbours of each.
fn tie_magnitudes() -> Vec<f32> {
    let mut v = Vec::new();
    for i in 0..0x7e {
        let mid = ((ocp_value(i) + ocp_value(i + 1)) / 2.0) as f32;
        v.extend([mid, up(mid), down(mid)]);
    }
    v
}

/// Rounding ties of both signs scaled by `2^ef` with `x[0] = cap * 2^ef` and no larger magnitude.
fn ties_row(k: usize, ef: i32, cap: f64) -> Vec<f32> {
    let mags: Vec<f32> = tie_magnitudes()
        .into_iter()
        .filter(|&m| f64::from(m) <= cap)
        .collect();
    let mut row = vec![(cap * pow2_f64(ef)) as f32];
    let mut i = 0usize;
    while row.len() < k {
        let m = f64::from(mags[(i / 2) % mags.len()]);
        row.push(((if i % 2 == 0 { m } else { -m }) * pow2_f64(ef)) as f32);
        i += 1;
    }
    row
}

/// A row whose largest magnitude lies in `[2^b, 2^(b+1))`: values spread over 24 binades below it, some zeros.
fn random_row(rng: &mut Rng, k: usize, b: i32) -> Vec<f32> {
    let mut row: Vec<f32> = (0..k)
        .map(|_| {
            if rng.below(16) == 0 {
                return if rng.below(2) == 0 { 0.0 } else { -0.0 };
            }
            (rng.sign() * rng.mantissa() * pow2_f64(b - rng.below(24) as i32 - 1)) as f32
        })
        .collect();
    let i = rng.idx(k);
    row[i] = (rng.sign() * rng.mantissa() * pow2_f64(b)) as f32;
    row
}

fn act_random_rows(
    rng: &mut Rng,
    n: usize,
    k: usize,
    mode: u32,
    shift: i32,
) -> Result<Vec<Vec<f32>>, String> {
    let mut rows = Vec::new();
    let mut tries = 0;
    while rows.len() < n {
        tries += 1;
        if tries > 100_000 {
            return Err(format!(
                "cannot generate fitting random rows for mode {mode} shift {shift}"
            ));
        }
        let b = if mode != 1 {
            rng.below(24) as i32 - 14
        } else {
            rng.below(78) as i32 - 60
        };
        let row = random_row(rng, k, b);
        if act_fits(&row, k, mode, shift) {
            rows.push(row)
        }
    }
    Ok(rows)
}

/// Zeros of both signs, single nonzeros, f32 subnormals, scale-exponent boundaries `224*2^e +- 1 ulp`, both
/// clamps, every OCP rounding tie, powers of two. Rows that leave the OCP finite range after scaling are
/// dropped by the caller (the OCP/FNUZ gray zone (448, 480] and overflow are never generated).
fn act_adversarial(rng: &mut Rng, k: usize, mode: u32, shift: i32) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<f32>> = vec![
        vec![0.0; k],
        vec![-0.0; k],
        (0..k)
            .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
            .collect(),
    ];
    for (j, &i) in [0usize, 1, 127, 128, k / 2, k - 1].iter().enumerate() {
        let mut r = vec![if j % 2 == 0 { 0.0f32 } else { -0.0 }; k];
        r[i] = if j % 2 == 0 { 1.0 } else { -3.5 };
        rows.push(r);
    }
    let ms = f32::from_bits(1);
    rows.push({
        let mut r = vec![0.0f32; k];
        r[7] = ms;
        r
    });
    rows.push((0..k).map(|i| if i % 2 == 0 { ms } else { -ms }).collect());
    rows.push(
        (0..k)
            .map(|i| f32::from_bits(i as u32 + 1) * if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect(),
    );
    rows.push(
        (0..k)
            .map(|i| {
                if i == 0 {
                    f32::MIN_POSITIVE
                } else {
                    f32::from_bits(i as u32 * 31)
                }
            })
            .collect(),
    );
    rows.push(
        (0..k)
            .map(|i| {
                if i == 3 {
                    1.0e-8
                } else {
                    f32::from_bits(0x3200_0000 + (i as u32) * 0x1_0003)
                        * if i % 2 == 0 { 1.0 } else { -1.0 }
                }
            })
            .collect(),
    );
    for kexp in [-33i32, -31, -30, -1, 0, 1, 8, 9, 10] {
        let base = (224.0 * pow2_f64(kexp)) as f32;
        for d in -1i32..=1 {
            let amax = f32::from_bits((base.to_bits() as i32 + d) as u32);
            let mut row: Vec<f32> = (0..k)
                .map(|_| {
                    (rng.sign() * (rng.next() >> 11) as f64 / (1u64 << 53) as f64 * f64::from(amax))
                        as f32
                })
                .collect();
            let at = rng.idx(k);
            row[at] = if (kexp + d) % 2 == 0 { amax } else { -amax };
            rows.push(row);
        }
    }
    if mode == 1 {
        for e in [0i32, -5, 5] {
            rows.push(ties_row(k, e + shift, 224.0 * pow2_f64(-shift)))
        }
    } else {
        rows.push(ties_row(k, 0, 448.0));
        rows.push(ties_row(k, 0, 224.0));
        rows.push(
            (0..k)
                .map(|i| [448.0f32, 432.0, 416.0, 400.0, 384.0, -448.0, -432.0, -440.0][i % 8])
                .collect(),
        );
        rows.push(
            (0..k)
                .map(|i| {
                    let base = [
                        pow2_f32(-9),
                        pow2_f32(-10),
                        pow2_f32(-11),
                        1.5 * pow2_f32(-9),
                        2.5 * pow2_f32(-9),
                        7.0 * pow2_f32(-9),
                    ][i % 6];
                    let v = match (i / 6) % 3 {
                        0 => base,
                        1 => up(base),
                        _ => down(base),
                    };
                    if (i / 18) % 2 == 0 {
                        v
                    } else {
                        -v
                    }
                })
                .collect(),
        );
    }
    rows.push(
        (0..k)
            .map(|i| {
                let v = pow2_f32(((i + 5) % 34) as i32 - 26);
                if i % 3 == 0 {
                    -v
                } else {
                    v
                }
            })
            .collect(),
    );
    for amax in [
        114_688.0f32,
        up(114_688.0),
        down(114_688.0),
        114_688.0 * 1.5,
        200_000.0,
    ] {
        rows.push(
            (0..k)
                .map(|i| {
                    if i == 17 % k {
                        amax
                    } else {
                        (rng.sign() * rng.mantissa() / 2.0 * f64::from(amax)) as f32
                    }
                })
                .collect(),
        );
    }
    let base = (224.0 * pow2_f64(-31)) as f32;
    for amax in [
        base,
        up(base),
        down(base),
        base * 0.5,
        1.0e-38,
        1.0e-44,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        3.0e-12,
    ] {
        rows.push(
            (0..k)
                .map(|i| {
                    if i == 31 % k {
                        -amax
                    } else {
                        (rng.sign() * rng.mantissa() / 2.0 * f64::from(amax)) as f32
                    }
                })
                .collect(),
        );
    }
    rows
}

fn act_case(
    k: usize,
    mode: u32,
    shift: i32,
    adversarial: bool,
    n: usize,
    seed: u64,
) -> Result<Case, String> {
    let mut rng = Rng(seed);
    let rows: Vec<Vec<f32>> = if adversarial {
        act_adversarial(&mut rng, k, mode, shift)
            .into_iter()
            .filter(|r| act_fits(r, k, mode, shift))
            .collect()
    } else {
        act_random_rows(&mut rng, n, k, mode, shift)?
    };
    if rows.is_empty() {
        return Err("no legal rows generated".into());
    }
    let nrows = rows.len();
    let x: Vec<f32> = rows.concat();
    let r = act::reference(&x, k, mode, shift)?;
    let spec = Spec {
        grid: [nrows as u32, 1, 1],
        block: [64, 1, 1],
        bufs: vec![
            in_buf("x_f32", f32b(&x)),
            out_buf("x_fp8", nrows * k),
            out_buf("half_sums", nrows * (k / 256) * 2 * 4),
            out_buf("row_scales", nrows * 4),
        ],
        args: vec![
            (0, Val::Ptr(0)),
            (8, Val::Ptr(1)),
            (16, Val::Ptr(2)),
            (24, Val::Ptr(3)),
            (32, Val::U32(k as u32)),
            (36, Val::U32(nrows as u32)),
            (40, Val::U32(mode)),
            (44, Val::I32(shift)),
        ],
        kernarg_bytes: 48,
    };
    Ok(Case {
        kernel: K_ACT,
        name: format!(
            "K{k} mode{mode} shift{shift} {} ({nrows} rows)",
            if adversarial { "adversarial" } else { "random" }
        ),
        seed,
        shape: json!({ "k": k, "rows": nrows, "scale_mode": mode, "row_scale_shift": shift, "adversarial": adversarial }),
        spec,
        outs: vec![
            OutSpec {
                name: "x_fp8",
                kind: Kind::Ocp8 { cols: k },
            },
            OutSpec {
                name: "half_sums",
                kind: Kind::F32 { cols: k / 128 },
            },
            OutSpec {
                name: "row_scales",
                kind: Kind::F32 { cols: 1 },
            },
        ],
        models: vec![DEFAULT_MODEL],
        refs: Refs::Fixed(vec![
            r.codes.clone(),
            f32b(&r.half_sums),
            f32b(&r.row_scales),
        ]),
    })
}

fn act_maker(k: usize, mode: u32, shift: i32, adversarial: bool) -> Maker {
    let seed = 0x1234_5678_9abc_def1u64
        ^ (k as u64) << 20
        ^ u64::from(mode) << 12
        ^ ((shift + 8) as u64) << 4
        ^ u64::from(adversarial);
    let n = if k >= 2048 { 3 } else { 6 };
    mk(
        format!(
            "act K{k} mode{mode} shift{shift} {}",
            if adversarial { "adversarial" } else { "random" }
        ),
        move || act_case(k, mode, shift, adversarial, n, seed),
    )
}

// ----------------------------------------------------------------------------
// MoE gate/up cases

struct MoeData {
    r: usize,
    k: usize,
    div: usize,
    w: Vec<u8>,
    stride: usize,
    tile_desc: Vec<[i32; 4]>,
    sorted: Vec<i32>,
    x: Vec<u8>,
    xs: Vec<f32>,
}

fn non_nan_weight(rng: &mut Rng, neg_zero_share: u64) -> u8 {
    loop {
        let b = if rng.below(neg_zero_share) == 0 {
            0x80
        } else {
            rng.next() as u8
        };
        if b & 0x7f != 0x7f {
            return b;
        }
    }
}

/// `runs`: per expert run `(expert, rows, pad16)`: `pad16` pads the run's last tile to 16 rows with `-1`,
/// otherwise the last tile is partial (`nrows < 16`). `skip_tile` appends a `-1` tile.
/// `adv`: weight rows cycle through extreme/cancelling/subnormal/0x80 patterns, activation rows through
/// +-max, alternating, smallest and zero codes, and every other expert's block scales are 2^7.
fn moe_data(
    seed: u64,
    r: usize,
    k: usize,
    div: usize,
    n_experts: usize,
    runs: &[(i32, usize, bool)],
    skip_tile: bool,
    adv: bool,
) -> MoeData {
    let mut rng = Rng(seed);
    let stride = moe::expert_stride(r, k);
    let mut w = vec![0u8; n_experts * stride];
    let (kb, sr) = (k / 128, r.div_ceil(128));
    for e in 0..n_experts {
        let blk = &mut w[e * stride..(e + 1) * stride];
        for c in 0..r {
            let row = &mut blk[c * k..(c + 1) * k];
            if !adv {
                for b in row.iter_mut() {
                    *b = non_nan_weight(&mut rng, 8)
                }
                continue;
            }
            match (c + e) % 8 {
                0 => row.fill(0x7e),
                1 => row.fill(0xfe),
                2 => row.fill(0x80),
                3 => row.fill(0x00),
                4 => {
                    for (i, b) in row.iter_mut().enumerate() {
                        *b = if i % 2 == 0 { 0x7e } else { 0xfe }
                    }
                }
                5 => {
                    for (i, b) in row.iter_mut().enumerate() {
                        *b = if (i / 3) % 2 == 0 { 0x01 } else { 0x81 }
                    }
                }
                6 => {
                    for b in row.iter_mut() {
                        *b = non_nan_weight(&mut rng, 2)
                    }
                }
                _ => {
                    for (i, b) in row.iter_mut().enumerate() {
                        let v = (i as u32).wrapping_mul(37).wrapping_add(c as u32) as u8;
                        *b = if v & 0x7f == 0x7f { 0x38 } else { v };
                    }
                }
            }
        }
        for i in 0..sr * kb {
            // BF16 block scales: positive 2^-8 .. 2^7 with a random mantissa (adv: every other expert exactly 2^7).
            let bits: u16 = if adv && e % 2 == 1 {
                0x4300
            } else {
                ((119 + rng.below(16) as u16) << 7) | rng.below(128) as u16
            };
            blk[r * k + 2 * i..r * k + 2 * i + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    // Tokens and flat slots token * x_row_div + rank.
    let total: usize = runs.iter().map(|r| r.1).sum();
    let tokens = total.div_ceil(div.min(4)).max(1) + 1;
    let mut tile_desc = Vec::new();
    let mut sorted = Vec::new();
    for &(expert, rows, pad16) in runs {
        let mut left = rows;
        while left > 0 {
            let n = left.min(16);
            tile_desc.push([
                expert,
                sorted.len() as i32,
                if pad16 { 16 } else { n as i32 },
                0,
            ]);
            for _ in 0..n {
                let tok = rng.idx(tokens);
                let rank = rng.idx(div);
                sorted.push((tok * div + rank) as i32);
            }
            if pad16 {
                sorted.extend(std::iter::repeat_n(-1, 16 - n))
            }
            left -= n;
        }
    }
    if skip_tile {
        tile_desc.push([-1, sorted.len() as i32, 16, 0]);
        sorted.extend(std::iter::repeat_n(-1, 16));
    }
    let mut x = vec![0u8; tokens * k];
    for t in 0..tokens {
        let row = &mut x[t * k..(t + 1) * k];
        let random = |rng: &mut Rng, row: &mut [u8]| {
            for b in row.iter_mut() {
                *b = loop {
                    let v = rng.next() as u8;
                    if v != 0x80 {
                        break v;
                    }
                }
            }
        };
        if !adv {
            random(&mut rng, row);
            continue;
        }
        match t % 6 {
            0 => row.fill(0x7f),
            1 => row.fill(0xff),
            2 => {
                for (i, b) in row.iter_mut().enumerate() {
                    *b = if i % 2 == 0 { 0x7f } else { 0xff }
                }
            }
            3 => row.fill(0x01),
            4 => row.fill(0x00),
            _ => random(&mut rng, row),
        }
    }
    let xs = (0..tokens)
        .map(|_| 2f32.powi(rng.below(10) as i32 - 6))
        .collect();
    MoeData {
        r,
        k,
        div,
        w,
        stride,
        tile_desc,
        sorted,
        x,
        xs,
    }
}

fn moe_models() -> Vec<AccumModel> {
    vec![
        DEFAULT_MODEL,
        AccumModel::Exact,
        AccumModel::Serial,
        AccumModel::Grouped(4),
        AccumModel::Grouped(8),
        AccumModel::Grouped(16),
        AccumModel::Grouped(32),
    ]
}

fn moe_case(
    name: String,
    seed: u64,
    r: usize,
    k: usize,
    div: usize,
    n_experts: usize,
    runs: Vec<(i32, usize, bool)>,
    skip_tile: bool,
    adv: bool,
) -> Result<Case, String> {
    if moe::KERNARG_OFFSETS != [0, 8, 12, 16, 24, 32, 40, 48, 56, 60, 64, 68] {
        return Err(format!(
            "unexpected MoE kernarg offsets {:?}",
            moe::KERNARG_OFFSETS
        ));
    }
    let d = moe_data(seed, r, k, div, n_experts, &runs, skip_tile, adv);
    let n_tiles = d.tile_desc.len();
    let n_sorted = d.sorted.len();
    let o = moe::KERNARG_OFFSETS.map(|v| v as usize);
    let i32b = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let spec = Spec {
        grid: moe::grid(r as u32, n_tiles as u32),
        block: [moe::WORKGROUP, 1, 1],
        bufs: vec![
            in_buf("W", d.w.clone()),
            in_buf(
                "tile_desc",
                d.tile_desc
                    .iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ),
            in_buf("sorted_rows", i32b(&d.sorted)),
            in_buf("X", d.x.clone()),
            in_buf("x_scale", f32b(&d.xs)),
            out_buf("Y", n_sorted * r * 4),
        ],
        // The u64 expert_stride at offset 8 is the two u32 halves the PM metadata declares at 8 and 12.
        args: vec![
            (o[0], Val::Ptr(0)),
            (o[1], Val::U64(d.stride as u64)),
            (o[3], Val::Ptr(1)),
            (o[4], Val::Ptr(2)),
            (o[5], Val::Ptr(3)),
            (o[6], Val::Ptr(4)),
            (o[7], Val::Ptr(5)),
            (o[8], Val::U32(r as u32)),
            (o[9], Val::U32(k as u32)),
            (o[10], Val::U32(div as u32)),
            (o[11], Val::U32(n_experts as u32)),
        ],
        kernarg_bytes: moe::KERNARG_BYTES as usize,
    };
    let live_rows: usize = d
        .tile_desc
        .iter()
        .filter(|t| t[0] >= 0)
        .map(|t| t[2] as usize)
        .sum();
    let shape = json!({
        "r": r, "k": k, "x_row_div": div, "experts": n_experts, "expert_stride": d.stride, "tiles": n_tiles,
        "sorted_rows": n_sorted, "live_rows": live_rows, "tile_desc": d.tile_desc, "adversarial": adv,
    });
    let refs = Refs::PerModel(Box::new(move |m| {
        let pb = moe::Problem {
            w: &d.w,
            expert_stride: d.stride,
            tile_desc: &d.tile_desc,
            sorted_rows: &d.sorted,
            x: &d.x,
            x_scale: &d.xs,
            r: d.r,
            k: d.k,
            x_row_div: d.div,
            n_experts,
        };
        let mut y = vec![f32::from_bits(u32::from_le_bytes([POISON; 4])); d.sorted.len() * d.r];
        moe::reference(&pb, m, &mut y)?;
        Ok(vec![f32b(&y)])
    }));
    Ok(Case {
        kernel: K_MOE,
        name,
        seed,
        shape,
        spec,
        outs: vec![OutSpec {
            name: "Y",
            kind: Kind::F32 { cols: r },
        }],
        models: moe_models(),
        refs,
    })
}

fn moe_maker(
    name: &str,
    seed: u64,
    r: usize,
    k: usize,
    div: usize,
    n_experts: usize,
    runs: Vec<(i32, usize, bool)>,
    skip_tile: bool,
    adv: bool,
) -> Maker {
    let name = format!("moe {name} R{r} K{k}");
    mk(name.clone(), move || {
        moe_case(
            name.clone(),
            seed,
            r,
            k,
            div,
            n_experts,
            runs.clone(),
            skip_tile,
            adv,
        )
    })
}

// ----------------------------------------------------------------------------
// QSA FP8 K/V codec cases (generators follow tests/cdna_qsa_fp8_kv.rs)

const D: usize = qsa::HEAD_DIM;

struct QCase {
    heads: usize,
    src: Vec<f32>,
}
impl QCase {
    fn new(heads: usize, rows: usize) -> Self {
        QCase {
            heads,
            src: vec![0.0; rows * heads * D],
        }
    }
    fn rows(&self) -> usize {
        self.src.len() / (self.heads * D)
    }
    fn head_mut(&mut self, row: usize, head: usize) -> &mut [f32] {
        let at = (row * self.heads + head) * D;
        &mut self.src[at..at + D]
    }
}

fn f16_value(bits: u16) -> f64 {
    let e = i32::from(bits >> 10);
    let m = f64::from(bits & 0x3ff);
    if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}

fn q_random(heads: usize, rng: &mut Rng) -> QCase {
    const MAGS: [f32; 8] = [1e-7, 1e-3, 0.02, 1.0, 37.0, 4096.0, 2e5, 3e5];
    let mut c = QCase::new(heads, 6);
    for row in 0..c.rows() {
        for head in 0..heads {
            let mag = MAGS[rng.idx(MAGS.len())];
            let exp = 2f32.powi(rng.below(7) as i32 - 3);
            for v in c.head_mut(row, head).iter_mut() {
                *v = mag * rng.normal() * if rng.below(5) == 0 { exp } else { 1.0 }
            }
        }
    }
    c
}

fn q_zero(heads: usize, rng: &mut Rng) -> QCase {
    let mut c = QCase::new(heads, 4);
    for head in 1..heads {
        for v in c.head_mut(1, head) {
            *v = rng.signed()
        }
    }
    for head in 0..heads {
        for v in c.head_mut(2, head) {
            *v = -0.0
        }
    }
    for head in 0..heads {
        for (i, v) in c.head_mut(3, head).iter_mut().enumerate() {
            *v = if i % 2 == 0 { 0.0 } else { -0.0 }
        }
    }
    c.head_mut(3, heads - 1)[D - 1] = 1e-30;
    c
}

fn q_tiny(heads: usize, rng: &mut Rng) -> QCase {
    let mut c = QCase::new(heads, 5);
    for head in 0..heads {
        for v in c.head_mut(0, head) {
            *v = 1e-9 * rng.signed()
        }
        let h = c.head_mut(1, head);
        for v in h.iter_mut() {
            *v = 448.0 * 2f32.powi(-26) * rng.signed()
        }
        h[7] = -448.0 * 2f32.powi(-26);
        for v in c.head_mut(2, head) {
            *v = 448.0 * 2f32.powi(-20) * 1.3 * rng.signed()
        }
        let h = c.head_mut(3, head);
        for (i, v) in h.iter_mut().enumerate() {
            *v = f32::from_bits(1 + (i as u32) * 977) * if i % 2 == 0 { 1.0 } else { -1.0 }
        }
        h[0] = 448.0;
        for v in c.head_mut(4, head) {
            *v = 2f32.powi(-24) * 448.0 * rng.signed() * rng.unit()
        }
    }
    c
}

/// Head scales on / next to F16 grid points so the one-ulp bump fires (`next_up(448*g)`) or does not.
fn q_bump(heads: usize, rng: &mut Rng, extra_random: usize) -> QCase {
    let mut gs: Vec<u16> = vec![
        0x0001, 0x0002, 0x0003, 0x03ff, 0x0400, 0x0401, 0x1234, 0x3bff, 0x3c00, 0x3c01, 0x5555,
        0x7bfd, 0x7bfe, 0x7bff,
    ];
    for _ in 0..extra_random {
        gs.push(1 + rng.below(0x7bfd) as u16)
    }
    let mut amaxes = Vec::new();
    for &g in &gs {
        let exact = 448.0f32 * f16_value(g) as f32;
        amaxes.push(exact);
        amaxes.push(down(exact));
        if g < 0x7bff {
            amaxes.push(up(exact))
        }
    }
    let mut c = QCase::new(heads, amaxes.len().div_ceil(heads));
    for (i, &amax) in amaxes.iter().enumerate() {
        let h = c.head_mut(i / heads, i % heads);
        for v in h.iter_mut() {
            *v = amax * rng.signed()
        }
        h[(i * 37) % D] = if i % 2 == 0 { amax } else { -amax };
        h[(i * 37 + 101) % D] = amax / 2.0;
    }
    c
}

/// Heads whose largest value is exactly +-448*s, every adjacent-code RNE midpoint (both signs) and both zeros.
fn q_pm448(heads: usize, rng: &mut Rng) -> QCase {
    const SCALES: [u16; 8] = [
        0x3c00, 0x3800, 0x0400, 0x4200, 0x7bff, 0x0001, 0x2e66, 0x5a3c,
    ];
    let mags: Vec<f64> = (0u8..=0x7e).map(ocp_value).collect();
    let mut c = QCase::new(heads, SCALES.len());
    for row in 0..SCALES.len() {
        for head in 0..heads {
            let g = f16_value(SCALES[(row + head) % SCALES.len()]) as f32;
            let mut vals: Vec<f32> = vec![448.0 * g, -448.0 * g, 0.0, -0.0];
            for w in mags.windows(2) {
                let mid = ((w[0] + w[1]) / 2.0) as f32 * g;
                vals.push(mid);
                vals.push(-mid);
            }
            for i in (1..D).rev() {
                vals.swap(i, rng.idx(i + 1))
            }
            c.head_mut(row, head).copy_from_slice(&vals);
        }
    }
    c
}

/// Negative zero and negative values that round to zero or sit on a tie next to it (amax 448: scale 1).
fn q_neg0(heads: usize) -> QCase {
    let mut c = QCase::new(heads, 2);
    let half_sub = 2f32.powi(-10);
    let vals = [
        -0.0,
        0.0,
        -1e-30,
        1e-30,
        -half_sub,
        half_sub,
        -3.0 * half_sub,
        3.0 * half_sub,
        -up(half_sub),
        up(half_sub),
        -down(half_sub),
        down(half_sub),
        -2f32.powi(-9),
        2f32.powi(-9),
        -f32::from_bits(1),
        f32::from_bits(1),
    ];
    for row in 0..2 {
        for head in 0..heads {
            let h = c.head_mut(row, head);
            for (i, v) in h.iter_mut().enumerate() {
                *v = vals[(i + row + head) % vals.len()]
            }
            h[(head * 61 + row * 17) % D] = if row == 0 { 448.0 } else { -448.0 };
        }
    }
    c
}

/// One large value sweeping through every channel of every head: the max reduction must see it.
fn q_sweep(heads: usize, rng: &mut Rng) -> QCase {
    let mut c = QCase::new(heads, D);
    for row in 0..D {
        for head in 0..heads {
            for v in c.head_mut(row, head) {
                *v = 0.003 * rng.normal()
            }
        }
        let hot = row % heads;
        c.head_mut(row, hot)[row] =
            if row % 3 == 0 { -100.0 } else { 100.0 } * (1.0 + 0.01 * (row % 7) as f32);
    }
    c
}

fn q_src(heads: usize, which: &str, quick: bool, seed: u64) -> Result<QCase, String> {
    let mut rng = Rng(seed);
    Ok(match which {
        "random" => q_random(heads, &mut rng),
        "zero" => q_zero(heads, &mut rng),
        "tiny" => q_tiny(heads, &mut rng),
        "bump" => q_bump(heads, &mut rng, if quick { 0 } else { 40 }),
        "pm448" => q_pm448(heads, &mut rng),
        "neg0" => q_neg0(heads),
        "sweep" => q_sweep(heads, &mut rng),
        other => return Err(format!("unknown QSA generator `{other}`")),
    })
}

/// Every legal byte `0x00..=0xff` except `0x80` (OCP -0), `0x7f` and `0xff` (OCP NaN) under adversarial F16 scales,
/// seeded independently of the encoder.
fn q_decode_stored(heads: usize, seed: u64) -> Vec<u8> {
    const SCALES: [u16; 12] = [
        0x0001, 0x0003, 0x03ff, 0x0400, 0x3800, 0x3c00, 0x4a3d, 0x5a3c, 0x7800, 0x7bfe, 0x7bff,
        0x0155,
    ];
    let codes: Vec<u8> = (0..=255u8)
        .filter(|&c| !matches!(c, 0x80 | 0x7f | 0xff))
        .collect();
    let rows = 2 * SCALES.len();
    let mut rng = Rng(seed);
    let mut out = Vec::new();
    for row in 0..rows {
        let mut scales = Vec::new();
        for head in 0..heads {
            let start = rng.idx(codes.len());
            out.extend((0..D).map(|i| codes[(start + i) % codes.len()]));
            scales.push(SCALES[(row + head * 5) % SCALES.len()]);
        }
        for s in scales {
            out.extend(s.to_le_bytes())
        }
    }
    out
}

fn qsa_spec(heads: usize, rows: usize, dst_len: usize, src: Vec<u8>) -> Spec {
    Spec {
        grid: qsa::grid(heads, rows),
        block: [qsa::BLOCK, 1, 1],
        bufs: vec![out_buf("dst", dst_len), in_buf("src", src)],
        args: vec![(0, Val::Ptr(0)), (8, Val::Ptr(1))],
        kernarg_bytes: 16,
    }
}

fn qsa_enc_case(heads: usize, which: &'static str, quick: bool, seed: u64) -> Result<Case, String> {
    let c = q_src(heads, which, quick, seed)?;
    let rows = c.rows();
    let want = qsa::reference_encode(heads, &c.src)?;
    Ok(Case {
        kernel: k_enc(heads),
        name: format!("H{heads} encode {which} ({rows} rows)"),
        seed,
        shape: json!({ "kv_heads": heads, "head_dim": D, "rows": rows, "generator": which, "row_bytes": qsa::row_bytes(heads) }),
        spec: qsa_spec(heads, rows, rows * qsa::row_bytes(heads), f32b(&c.src)),
        outs: vec![OutSpec {
            name: "dst",
            kind: Kind::QsaRow { heads },
        }],
        models: vec![DEFAULT_MODEL],
        refs: Refs::Fixed(vec![want]),
    })
}

fn qsa_dec_case(heads: usize, which: &'static str, quick: bool, seed: u64) -> Result<Case, String> {
    let stored = match which {
        "all-codes" => q_decode_stored(heads, seed),
        "encoded-random" | "encoded-pm448" | "encoded-bump" => {
            let g = &which["encoded-".len()..];
            qsa::reference_encode(heads, &q_src(heads, g, quick, seed)?.src)?
        }
        other => return Err(format!("unknown QSA decode input `{other}`")),
    };
    let rb = qsa::row_bytes(heads);
    if stored.len() % rb != 0 {
        return Err("stored rows are not a whole number of rows".into());
    }
    let rows = stored.len() / rb;
    let want = qsa::reference_decode(heads, &stored)?;
    Ok(Case {
        kernel: k_dec(heads),
        name: format!("H{heads} decode {which} ({rows} rows)"),
        seed,
        shape: json!({ "kv_heads": heads, "head_dim": D, "rows": rows, "input": which, "row_bytes": rb }),
        spec: qsa_spec(heads, rows, want.len() * 4, stored),
        outs: vec![OutSpec {
            name: "dst",
            kind: Kind::F32 { cols: heads * D },
        }],
        models: vec![DEFAULT_MODEL],
        refs: Refs::Fixed(vec![f32b(&want)]),
    })
}

fn qsa_seed(heads: usize, which: &str) -> u64 {
    0x9e37_79b9_7f4a_7c15 ^ ((heads as u64) << 8) ^ fnv(which)
}

fn qsa_enc_maker(heads: usize, which: &'static str, quick: bool) -> Maker {
    let seed = qsa_seed(heads, which);
    mk(format!("qsa H{heads} encode {which}"), move || {
        qsa_enc_case(heads, which, quick, seed)
    })
}
fn qsa_dec_maker(heads: usize, which: &'static str, quick: bool) -> Maker {
    let seed = qsa_seed(heads, which) ^ 0xdec0de;
    mk(format!("qsa H{heads} decode {which}"), move || {
        qsa_dec_case(heads, which, quick, seed)
    })
}

// ----------------------------------------------------------------------------
// Case list

fn makers(quick: bool) -> Vec<Maker> {
    let mut v = Vec::new();
    // Activation producer: K=256 always; K=2560 in full mode.
    for (mode, shift) in [(1u32, 0i32), (0, 0)] {
        v.push(act_maker(256, mode, shift, false));
        v.push(act_maker(256, mode, shift, true));
    }
    // MoE: partial/padded multi-expert tiles at a small shape.
    v.push(moe_maker(
        "random partial+padded+skipped tiles",
        0x6d6f_6501,
        64,
        128,
        2,
        3,
        vec![(2, 20, false), (0, 5, true)],
        true,
        false,
    ));
    v.push(moe_maker(
        "adversarial",
        0x6d6f_6502,
        64,
        128,
        1,
        3,
        vec![(1, 16, false), (0, 3, true)],
        false,
        true,
    ));
    // QSA.
    for h in [2usize, 4] {
        for which in ["random", "zero", "bump", "neg0"] {
            v.push(qsa_enc_maker(h, which, quick))
        }
        for which in ["all-codes", "encoded-random"] {
            v.push(qsa_dec_maker(h, which, quick))
        }
    }
    if !quick {
        for (mode, shift) in [(1u32, -1i32), (1, 2), (1, 3)] {
            v.push(act_maker(256, mode, shift, false));
            v.push(act_maker(256, mode, shift, true));
        }
        for (mode, shift) in [(1u32, 0i32), (1, -1), (1, 2), (1, 3), (0, 0)] {
            v.push(act_maker(2560, mode, shift, false));
            v.push(act_maker(2560, mode, shift, true));
        }
        // Production gate/up shape: one padded tile (13 live rows + 3 padding rows) plus a skipped tile,
        // and one partial tile (9 rows) of extreme weights/activations; then the down-style launch.
        v.push(moe_maker(
            "random padded tile + skipped tile",
            0x6d6f_6511,
            1280,
            2560,
            10,
            3,
            vec![(2, 13, true)],
            true,
            false,
        ));
        v.push(moe_maker(
            "adversarial partial tile",
            0x6d6f_6512,
            1280,
            2560,
            10,
            2,
            vec![(1, 9, false)],
            false,
            true,
        ));
        v.push(moe_maker(
            "down-style random partial tile",
            0x6d6f_6513,
            2560,
            640,
            1,
            2,
            vec![(1, 7, false)],
            false,
            false,
        ));
        for h in [2usize, 4] {
            for which in ["tiny", "pm448", "sweep"] {
                v.push(qsa_enc_maker(h, which, quick))
            }
            for which in ["encoded-pm448", "encoded-bump"] {
                v.push(qsa_dec_maker(h, which, quick))
            }
        }
    }
    v
}

// ----------------------------------------------------------------------------
// Driver

struct Opts {
    dry_run: bool,
    quick: bool,
    receipt: PathBuf,
    hsaco_dir: PathBuf,
}

fn parse_args() -> Result<Option<Opts>, String> {
    let mut o = Opts {
        dry_run: false,
        quick: false,
        receipt: PathBuf::from("./cdna_card_tierc.json"),
        hsaco_dir: PathBuf::new(),
    };
    let mut hsaco: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--dry-run" => o.dry_run = true,
            "--quick" => o.quick = true,
            "--receipt" => o.receipt = PathBuf::from(it.next().ok_or("--receipt needs a path")?),
            "--hsaco-dir" => {
                hsaco = Some(PathBuf::from(
                    it.next().ok_or("--hsaco-dir needs a directory")?,
                ))
            }
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    o.hsaco_dir = match hsaco {
        Some(p) => p,
        None => std::env::current_exe()
            .map_err(|e| format!("current_exe: {e}"))?
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or("executable has no parent directory")?,
    };
    Ok(Some(o))
}

const ASSUMPTIONS: [&str; 7] = [
    "[UNVERIFIED: confirm on MI300X] the installed HIP runtime accepts the PeaceMaker-emitted v6 code object through hipModuleLoadData",
    "[UNVERIFIED: confirm on MI300X] hipModuleLaunchKernel with the explicit-kernarg blob (HIP_LAUNCH_PARAM_BUFFER_*) binds the identical bytes for the PM and the HIP twin kernels (PM kernarg sizes 48 / 72 / 16, no hidden arguments)",
    "[UNVERIFIED: confirm on MI300X] the MFMA FP8 accumulation order of the hardware equals one of the emulator AccumModels (default Cdna3Fused); the MoE receipt lists which models match bit for bit",
    "[UNVERIFIED: confirm on MI300X] hipcc-built <name>.hsaco are raw (unbundled) ELF code objects: both hipModuleLoadData and the emulator loader read them as-is",
    "[UNVERIFIED: confirm on MI300X] hardware FP8 conversion overflow / NaN behaviour matches the emulator (inputs are generated inside the OCP finite range after scaling, so overflow is not exercised)",
    "[UNVERIFIED: confirm on MI300X] hipGetDeviceProperties reports gcnArchName `gfx942` for MI300X (the harness refuses any other string)",
    "no module unload / device reset between cases: a faulting kernel can poison the HIP context, after which later cases report errors",
];

fn run(opts: &Opts, rec: &mut Receipt) -> bool {
    rec.top.insert("schema".into(), json!("cdna_card_tierc/1"));
    rec.top.insert(
        "mode".into(),
        json!(if opts.dry_run { "dry-run" } else { "card" }),
    );
    rec.top.insert("quick".into(), json!(opts.quick));
    rec.top
        .insert("args".into(), json!(std::env::args().collect::<Vec<_>>()));
    rec.top.insert("code_object_version".into(), json!("v6"));
    rec.top.insert(
        "default_accum_model".into(),
        json!(model_name(DEFAULT_MODEL)),
    );
    rec.top
        .insert("poison_byte".into(), json!(format!("{POISON:#04x}")));
    rec.top.insert("host".into(), host_info());
    rec.top.insert(
        "tools".into(),
        json!({
            "hipcc": tool_version("/opt/rocm/bin/hipcc"),
            "clang": tool_version("/opt/rocm/lib/llvm/bin/clang"),
        }),
    );
    rec.top.insert("assumptions".into(), json!(ASSUMPTIONS));
    rec.top.insert(
        "hsaco_dir".into(),
        json!(opts.hsaco_dir.display().to_string()),
    );
    for t in ["hipcc", "clang"] {
        if rec.top["tools"][t]["ok"] != json!(true) {
            eprintln!("warning: could not query `{t} --version` (recorded in the receipt)");
        }
    }

    let (kernels, hsacos) = build_kernels(&opts.hsaco_dir);
    rec.top.insert("kernels".into(), Value::Array(kernels.iter().map(|k| json!({
        "kernel": k.label, "module": k.module, "pm_symbol": k.pm_symbol, "hip_symbol": k.hip_symbol,
        "pm_code_object_bytes": k.pm.as_ref().ok().map(|p| p.elf.len()),
        "pm_code_object_sha256": k.pm.as_ref().ok().map(|p| sha(&p.elf)),
        "pm_words": k.pm.as_ref().ok().map(|p| p.words),
        "pm_error": k.pm.as_ref().err(),
        "hip_emulator_load": match &k.hip { Ok(_) => json!({ "available": true }), Err(e) => json!({ "available": false, "reason": e }) },
    })).collect()));
    rec.top.insert("hsaco".into(), Value::Array(hsacos.iter().map(|h| json!({
        "name": h.name, "path": h.path.display().to_string(),
        "bytes": h.bytes.as_ref().ok().map(|b| b.len()), "sha256": h.bytes.as_ref().ok().map(|b| sha(b)), "error": h.bytes.as_ref().err(),
    })).collect()));
    let mut ok = true;
    for k in &kernels {
        if let Err(e) = &k.pm {
            rec.error(format!("{}: {e}", k.label));
            ok = false
        }
    }

    let card = if opts.dry_run {
        println!("mode: dry-run (no HIP library; card columns are a second emulator run)");
        None
    } else {
        match setup_card(&kernels, &hsacos, rec) {
            Ok(c) => Some(c),
            Err(e) => {
                rec.error(e);
                rec.flush(false);
                return false;
            }
        }
    };
    println!(
        "mode: {}{}; receipt {}",
        if opts.dry_run { "dry-run" } else { "card" },
        if opts.quick { " quick" } else { " full" },
        rec.path.display()
    );

    let env = Env {
        dry: opts.dry_run,
        kernels: &kernels,
        card: card.as_ref(),
        hip_off: RefCell::new(vec![None; kernels.len()]),
    };
    rec.flush(false);
    for m in makers(opts.quick) {
        let t0 = Instant::now();
        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (m.make)()));
        let (pass, line, json) = match built {
            Ok(Ok(case)) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_case(&env, &case)
            })) {
                Ok(r) => r,
                Err(p) => fail_case(
                    &env,
                    &case,
                    vec![format!("panic while running: {}", panic_msg(p))],
                    t0,
                ),
            },
            Ok(Err(e)) => (
                false,
                format!("FAIL {} : case generation failed: {}", m.name, one_line(&e)),
                json!({ "case": m.name, "status": "FAIL", "errors": [e] }),
            ),
            Err(p) => {
                let e = format!("panic while generating: {}", panic_msg(p));
                (
                    false,
                    format!("FAIL {} : {}", m.name, one_line(&e)),
                    json!({ "case": m.name, "status": "FAIL", "errors": [e] }),
                )
            }
        };
        println!("{line}  ({:.1}s)", t0.elapsed().as_secs_f64());
        if pass {
            rec.pass += 1
        } else {
            rec.fail += 1
        }
        rec.cases.push(json);
        rec.flush(false);
    }
    ok && rec.fail == 0
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(Some(o)) => o,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut rec = Receipt {
        path: opts.receipt.clone(),
        t0: Instant::now(),
        top: Map::new(),
        cases: Vec::new(),
        errors: Vec::new(),
        pass: 0,
        fail: 0,
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&opts, &mut rec)));
    let ok = match r {
        Ok(ok) => ok,
        Err(p) => {
            rec.error(format!("harness panic: {}", panic_msg(p)));
            false
        }
    };
    let ok = ok && rec.errors.is_empty();
    rec.flush(true);
    println!(
        "SUMMARY: {} cases, {} pass, {} fail, {} setup/run errors, {:.1}s wall -> {}",
        rec.pass + rec.fail,
        rec.pass,
        rec.fail,
        rec.errors.len(),
        rec.t0.elapsed().as_secs_f64(),
        if ok { "OK" } else { "FAILED" }
    );
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
