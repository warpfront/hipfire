// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! rip-search phase-2 hardware oracle for the HC8 mixer kernels on gfx1201 (S5 byte identity and
//! S6 interleaved timing of candidate objects against the SHIPPED `rdna_compute::hc8::IMAGE`).
//!
//! Variants (the shipped entry of each is the reference):
//!   down_n{1,2,3}  `hc8_gemm_e4m3_mw1_kw16_n{N}_act`, weight [320, 10240], block 512, grid M/16
//!   up_n{1,2,3}    `hc8_read_up_e4m3_n{N}`, hidden 2560, weight [4*hidden, 320], block 128, grid hidden/16
//! Kernarg, grid, block and LDS (0 dynamic) are exactly those of `hc8::hc8_gemm` / `hc8::hc8_read_up`.
//!
//! ```text
//! hc8_rip_oracle (--s5 | --s6 --interleave | --s5 --s6 --interleave)
//!     [--manifest PATH:VARIANT]...   rip-search manifest.tsv (cols id, axes, elf_sha256, ...; objects
//!                                    at <manifest dir>/objects/<id>.co); VARIANT is the run's variant
//!     [--variants V,V,...]           shipped-only variants (reference A/A runs, no candidates)
//!     [--rows-list R,R,...]          S6 rows override (rows outside 1..=16*NT of a variant are skipped
//!                                    and listed in `rows_skipped`)
//!     [--reps N=15] [--warm N=3] [--seed N] --out FILE.json
//! ```
//!
//! Every candidate's bytes are checked against the manifest `elf_sha256` before loading; a bare ELF is
//! retried inside the `__CLANG_OFFLOAD_BUNDLE__` wrapper if the loader refuses it (`load_wrap`).
//! Duplicate shas inside one manifest are skipped (`manifest_skipped`).
//!
//! S5 (`--s5`): per variant the cases are shapes (served; two small) x data modes x rows
//! ({1,2,3,7,8,15,16} + NT>=2 {17,24,31,32} + NT=3 {33,47,48} + 16*NT+1 which must store nothing).
//! Modes: `rand` (finite weights, BF16-exact x); `edge_w` (E4M3 NaN codes 0x7f/0xff, -0, row scales
//! finite); `edge_x` (x NaN/Inf/-0/subnormal/max rows); `edge_s` / `edge_s_act0` / `edge_s_actinf` /
//! `edge_s_actnan` (row-scale edges 0,-0,subnormal,huge,Inf,NaN and act_scale -2/0/Inf/NaN; down uses
//! act_scale, up ignores it); `edge_n` (up only: `normalized` NaN/Inf/-0/subnormal). Output buffers carry
//! 256-B 0x7f guards on both sides and are filled 0x7f before every launch; the whole guarded buffer of
//! the candidate must equal the shipped entry's bytes. Candidates are loaded one at a time (the shipped
//! reference is computed once per case and launched twice to prove it deterministic); inputs are
//! uploaded once per variant and freed (then `drain_pool`) after it.
//! Exit 1 on any mismatching byte or candidate load failure, 2 on error.
//!
//! S6 (`--s6 --interleave`): per variant at the served shape, per rows cell (down_n1/up_n1: 7;
//! n2: 24 and 32; n3: 48; `--rows-list`), arms ref, c1, ref, c2, ..., ref (ref = shipped entry). A
//! candidate whose whole guarded output differs from the shipped one on weight copy 0 is not timed.
//! One replay is a HIP graph of `calls_per_replay` back-to-back launches over `weight_copies`
//! rotating weight copies (copies = max(3, ceil(200 MB / copy bytes)); calls = copies * passes, passes
//! = ceil(120 / copies)); x/low/normalized/y stay hot. Per arm: `--warm` untimed replays then `--reps`
//! timed ones (per-call us = replay wall time / calls). median = sorted[len/2].
//!
//! JSON (schema `hc8-rip-oracle-v1`):
//! ```text
//! { schema, modes:[..], arch, mismatch_bytes_total,
//!   inputs:{ manifests:[{path,variant,sha256,objects,skipped}], baseline:{module,image_sha256,
//!            pinned_sha256,symbols:{variant:symbol}}, reps, warm, seed, calls_per_replay_rule,
//!            env:{FABRIC_JOB,FABRIC_CARDS,HIP_VISIBLE_DEVICES,ROCR_VISIBLE_DEVICES},
//!            device:{arch,device_id,pci_bus_id,pci_vendor,pci_device,pci_revision,product_name} },
//!   s5:{ variants:[{ variant, symbol, cases_per_object, ref_checks, objects:[{ id, sha256, axes,
//!        load_wrap, hw_bitwise, mismatch_bytes, guard_overwritten_bytes, cases_failed,
//!        by_mode:{mode:bytes}, first_diffs:[{case,word,region,ref,got}] }] | load_error }] },
//!   s6:{ order, reps, warm, median, variants:[{ variant, symbol, shape:{m,k,hidden?}, weight_copies,
//!        weight_copy_bytes, calls_per_replay, cells:[{ rows, ref:{arms,arm_medians_us,aa_spread_us,
//!        pooled_median_us,pooled_p10_us,pooled_p90_us,arm_replays_us,guard_overwritten_bytes},
//!        objects:[{ id, sha256, axes, hw_bitwise, mismatch_bytes, timed, median_us, p10_us, p90_us,
//!        reps, ref_before_median_us, ref_after_median_us, replays_us }] }] }] } }
//! ```

use peacemaker_isa::{arch::Arch, native};
use rdna_compute::hc8::{self, Hc8Fmt, Hc8Geom};
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use std::path::Path;
use std::time::Instant;

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

const GUARD: usize = 256;
const GUARD_BYTE: u8 = 0x7f;
const SERVED_COPY_TARGET: usize = 200_000_000;
const MIN_CALLS_PER_REPLAY: usize = 120;

/// xorshift64* stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform in [-1, 1).
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---- variants -------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Down,
    Up,
}

#[derive(Clone, Copy, Debug)]
struct Variant {
    name: &'static str,
    kind: Kind,
    nt: usize,
}

const VARIANTS: [Variant; 6] = [
    Variant {
        name: "down_n1",
        kind: Kind::Down,
        nt: 1,
    },
    Variant {
        name: "down_n2",
        kind: Kind::Down,
        nt: 2,
    },
    Variant {
        name: "down_n3",
        kind: Kind::Down,
        nt: 3,
    },
    Variant {
        name: "up_n1",
        kind: Kind::Up,
        nt: 1,
    },
    Variant {
        name: "up_n2",
        kind: Kind::Up,
        nt: 2,
    },
    Variant {
        name: "up_n3",
        kind: Kind::Up,
        nt: 3,
    },
];

/// Weight `[w_rows, k]` (down: `m = w_rows`; up: `w_rows = 4 * hidden`).
#[derive(Clone, Copy, Debug)]
struct Shape {
    w_rows: usize,
    k: usize,
}

impl Variant {
    fn by_name(name: &str) -> Result<Variant> {
        VARIANTS
            .iter()
            .copied()
            .find(|v| v.name == name)
            .ok_or_else(|| format!("unknown variant {name:?} (want one of down_n1..3, up_n1..3)"))
    }
    fn symbol(&self) -> String {
        match self.kind {
            Kind::Down => hc8::entry_name(Hc8Fmt::E4m3, Hc8Geom { mw: 1, kw: 16 }, self.nt, true),
            Kind::Up => hc8::read_up_symbol(self.nt as u32),
        }
    }
    fn block(&self) -> u32 {
        match self.kind {
            Kind::Down => 32 * 16,
            Kind::Up => hc8::READ_UP_BLOCK,
        }
    }
    fn max_rows(&self) -> usize {
        16 * self.nt
    }
    fn served(&self) -> Shape {
        match self.kind {
            Kind::Down => Shape {
                w_rows: 320,
                k: 10240,
            },
            Kind::Up => Shape {
                w_rows: 4 * 2560,
                k: 320,
            },
        }
    }
    fn small_shapes(&self) -> [Shape; 2] {
        match self.kind {
            Kind::Down => [
                Shape { w_rows: 64, k: 512 },
                Shape {
                    w_rows: 48,
                    k: 1536,
                },
            ],
            Kind::Up => [
                Shape {
                    w_rows: 4 * 32,
                    k: 64,
                },
                Shape {
                    w_rows: 4 * 48,
                    k: 96,
                },
            ],
        }
    }
    fn served_rows(&self) -> Vec<usize> {
        match self.nt {
            1 => vec![7],
            2 => vec![24, 32],
            _ => vec![48],
        }
    }
    fn s5_rows(&self) -> Vec<usize> {
        let mut r = vec![1, 2, 3, 7, 8, 15, 16];
        if self.nt >= 2 {
            r.extend([17, 24, 31, 32]);
        }
        if self.nt >= 3 {
            r.extend([33, 47, 48]);
        }
        r.push(16 * self.nt + 1);
        r
    }
}

impl Shape {
    /// Output columns per row: down `m`, up `hidden`.
    fn out_dim(&self, kind: Kind) -> usize {
        match kind {
            Kind::Down => self.w_rows,
            Kind::Up => self.w_rows / 4,
        }
    }
    fn label(&self, kind: Kind) -> String {
        match kind {
            Kind::Down => format!("M{}xK{}", self.w_rows, self.k),
            Kind::Up => format!("hidden{}xK{}", self.w_rows / 4, self.k),
        }
    }
}

// ---- launch ---------------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Ptrs {
    q: u64,
    s: u64,
    x: u64,
    nrm: u64,
    y: u64,
}

fn kernarg(v: &Variant, sh: Shape, p: &Ptrs, rows: usize, act: f32) -> Vec<u8> {
    match v.kind {
        Kind::Down => {
            let mut a = vec![0u8; hc8::KERNARG_BYTES];
            for (i, q) in [p.q, p.s, p.x, p.y].into_iter().enumerate() {
                a[i * 8..i * 8 + 8].copy_from_slice(&q.to_le_bytes());
            }
            a[32..36].copy_from_slice(&(sh.w_rows as u32).to_le_bytes());
            a[36..40].copy_from_slice(&(sh.k as u32).to_le_bytes());
            a[40..44].copy_from_slice(&(rows as u32).to_le_bytes());
            a[44..48].copy_from_slice(&act.to_le_bytes());
            a
        }
        Kind::Up => {
            let mut a = vec![0u8; hc8::READ_UP_KERNARG_BYTES as usize];
            for (i, q) in [p.q, p.s, p.x, p.nrm, p.y].into_iter().enumerate() {
                a[i * 8..i * 8 + 8].copy_from_slice(&q.to_le_bytes());
            }
            a[40..44].copy_from_slice(&((sh.w_rows / 4) as u32).to_le_bytes());
            a[44..48].copy_from_slice(&(sh.k as u32).to_le_bytes());
            a[48..52].copy_from_slice(&(rows as u32).to_le_bytes());
            a
        }
    }
}

/// One launch on the active stream (captured by the graph arms).
fn launch(
    gpu: &Gpu,
    func: &hip_bridge::Function,
    v: &Variant,
    sh: Shape,
    p: &Ptrs,
    rows: usize,
    act: f32,
) -> Result<()> {
    let mut ka = kernarg(v, sh, p, rows, act);
    let grid = (sh.out_dim(v.kind) / 16) as u32;
    // SAFETY: the kernarg follows the shipped hc8 launcher ABI for this entry and the pointers
    // address live allocations sized for `rows <= 16 * NT` rows.
    unsafe {
        gpu.hip.launch_kernel_blob(
            func,
            [grid, 1, 1],
            [v.block(), 1, 1],
            0,
            gpu.active_stream.as_ref(),
            &mut ka,
        )
    }
    .map_err(err)
}

// ---- objects --------------------------------------------------------------------------------

struct Loaded {
    wrap: &'static str,
    _module: Option<hip_bridge::Module>,
    func: hip_bridge::Function,
}

fn load_symbol(gpu: &Gpu, bytes: &[u8], symbol: &str, label: &str) -> Result<Loaded> {
    let attempt = |image: &[u8]| -> Result<(hip_bridge::Module, hip_bridge::Function)> {
        let module = gpu.hip.module_load_data(image).map_err(err)?;
        let func = gpu.hip.module_get_function(&module, symbol).map_err(err)?;
        Ok((module, func))
    };
    match attempt(bytes) {
        Ok((m, f)) => Ok(Loaded {
            wrap: "raw",
            _module: Some(m),
            func: f,
        }),
        Err(raw_err) if gpu.arch == "gfx1201" => {
            let wrapped = native::bundle(bytes, Arch::Gfx1201, native::DEFAULT_HOST_TARGET);
            let (m, f) = attempt(&wrapped)
                .map_err(|e| format!("{label}: raw load: {raw_err}; bundled load: {e}"))?;
            Ok(Loaded {
                wrap: "bundle",
                _module: Some(m),
                func: f,
            })
        }
        Err(e) => Err(e),
    }
}

struct Cand {
    id: String,
    sha: String,
    axes: Value,
    bytes: Vec<u8>,
}

struct ManifestRun {
    path: String,
    variant: Variant,
    sha: String,
    cands: Vec<Cand>,
    skipped: Vec<Value>,
}

/// rip-search manifest.tsv: tab separated, header; `id`, `axes`, `elf_sha256`. Objects:
/// `<manifest dir>/objects/<id>.co`, sha256-verified against `elf_sha256` (refuse on mismatch).
fn read_manifest(spec: &str) -> Result<ManifestRun> {
    let (path, vname) = spec
        .rsplit_once(':')
        .ok_or_else(|| format!("--manifest {spec}: want PATH:VARIANT"))?;
    let variant = Variant::by_name(vname)?;
    let raw = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let sha = sha256_hex(&raw);
    let text = String::from_utf8(raw).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("{path}: empty manifest"))?
        .split('\t')
        .map(str::trim)
        .collect();
    let col = |n: &str| {
        header
            .iter()
            .position(|h| *h == n)
            .ok_or_else(|| format!("{path}: no `{n}` column"))
    };
    let (c_id, c_axes, c_sha) = (col("id")?, col("axes")?, col("elf_sha256")?);
    let dir = Path::new(path).parent().unwrap_or(Path::new("."));
    let mut seen: std::collections::HashMap<String, String> = Default::default();
    let mut cands = Vec::new();
    let mut skipped = Vec::new();
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        let get = |c: usize| f.get(c).copied().unwrap_or("").trim();
        let (id, want) = (get(c_id), get(c_sha));
        if id.is_empty() || id.contains('/') || id.contains("..") {
            return Err(format!("{path}: row {}: bad id {id:?}", n + 1));
        }
        if want.is_empty() {
            skipped.push(json!({"id": id, "reason": "empty elf_sha256"}));
            continue;
        }
        if let Some(first) = seen.get(want) {
            skipped.push(json!({"id": id, "reason": "dedup", "dedup_of": first}));
            continue;
        }
        let axes: Value = serde_json::from_str(get(c_axes))
            .map_err(|e| format!("{path}: {id}: axes: {e}"))?;
        let obj = dir.join("objects").join(format!("{id}.co"));
        let bytes = std::fs::read(&obj).map_err(|e| format!("{}: {e}", obj.display()))?;
        let got = sha256_hex(&bytes);
        if got != want {
            return Err(format!(
                "{}: sha256 {got} != manifest elf_sha256 {want}; refusing to load",
                obj.display()
            ));
        }
        seen.insert(want.to_string(), id.to_string());
        cands.push(Cand {
            id: id.to_string(),
            sha: got,
            axes,
            bytes,
        });
    }
    Ok(ManifestRun {
        path: path.to_string(),
        variant,
        sha,
        cands,
        skipped,
    })
}

// ---- data -----------------------------------------------------------------------------------

struct Mode {
    name: &'static str,
    act: f32,
    w_edge: bool,
    x_edge: bool,
    s_edge: bool,
    n_edge: bool,
    up_only: bool,
}

const MODES: [Mode; 8] = [
    Mode {
        name: "rand",
        act: 0.8,
        w_edge: false,
        x_edge: false,
        s_edge: false,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_w",
        act: 1.0,
        w_edge: true,
        x_edge: false,
        s_edge: false,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_x",
        act: 0.5,
        w_edge: false,
        x_edge: true,
        s_edge: false,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_s",
        act: -2.0,
        w_edge: false,
        x_edge: false,
        s_edge: true,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_s_act0",
        act: 0.0,
        w_edge: false,
        x_edge: false,
        s_edge: true,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_s_actinf",
        act: f32::INFINITY,
        w_edge: false,
        x_edge: false,
        s_edge: true,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_s_actnan",
        act: f32::NAN,
        w_edge: false,
        x_edge: false,
        s_edge: true,
        n_edge: false,
        up_only: false,
    },
    Mode {
        name: "edge_n",
        act: 1.0,
        w_edge: false,
        x_edge: false,
        s_edge: false,
        n_edge: true,
        up_only: true,
    },
];

/// E4M3 codes: NaN (both signs), -0, +0, max, smallest subnormals.
const EDGE_Q: [u8; 9] = [0x7f, 0xff, 0x80, 0x00, 0x7e, 0xfe, 0x01, 0x81, 0x07];
/// f32 bit patterns of row scales: +-0, subnormals, min normal, huge, +-Inf, NaN payloads, +-1, 2^+-100.
const EDGE_S: [u32; 14] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x7f7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_1234,
    0x3f80_0000,
    0xbf80_0000,
    0x7180_0000,
    0x0e80_0000,
];
/// BF16 bit patterns for x edges.
const EDGE_X_NAN: [u16; 3] = [0x7fc1, 0xffc1, 0x7f81];
const EDGE_X_INF: [u16; 2] = [0x7f80, 0xff80];
const EDGE_X_MISC: [u16; 6] = [0x8000, 0x0001, 0x8001, 0x007f, 0x7f7f, 0xff7f];
/// f32 bit patterns of `normalized` edges.
const EDGE_N: [u32; 8] = [
    0x7fc0_0001,
    0xffc0_1234,
    0x7f80_0000,
    0xff80_0000,
    0x8000_0000,
    0x0000_0001,
    0x807f_ffff,
    0x7f7f_ffff,
];

struct HostData {
    q: Vec<u8>,
    s: Vec<f32>,
    x: Vec<u32>,
    nrm: Vec<u32>,
}

fn bf16_exact(v: f32) -> u32 {
    v.to_bits() & 0xffff_0000
}

fn rand_q(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let b = (rng.next() >> 24) as u8;
            if b & 0x7f == 0x7f {
                b ^ 0x08
            } else {
                b
            }
        })
        .collect()
}

fn rand_s(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| f32::from_bits(((127 - rng.below(9)) as u32) << 23))
        .collect()
}

fn gen_data(rng: &mut Rng, v: &Variant, sh: Shape, mode: &Mode) -> HostData {
    let rows = v.max_rows();
    let mut q = rand_q(rng, sh.w_rows * sh.k);
    if mode.w_edge {
        for b in q.iter_mut() {
            if rng.below(64) == 0 {
                *b = EDGE_Q[rng.below(EDGE_Q.len())];
            }
        }
    }
    let mut s = rand_s(rng, sh.w_rows);
    if mode.s_edge {
        for e in s.iter_mut() {
            if rng.below(4) != 0 {
                *e = f32::from_bits(EDGE_S[rng.below(EDGE_S.len())]);
            }
        }
    }
    let mut x: Vec<u32> = (0..rows * sh.k).map(|_| bf16_exact(rng.f32() * 3.0)).collect();
    if mode.x_edge {
        for r in 0..rows {
            let pool: &[u16] = match r % 4 {
                0 => &[],
                1 => &EDGE_X_NAN,
                2 => &EDGE_X_INF,
                _ => &EDGE_X_MISC,
            };
            if pool.is_empty() {
                continue;
            }
            for c in 0..sh.k {
                if rng.below(128) == 0 {
                    x[r * sh.k + c] = u32::from(pool[rng.below(pool.len())]) << 16;
                }
            }
        }
    }
    let mut nrm: Vec<u32> = Vec::new();
    if v.kind == Kind::Up {
        nrm = (0..rows * sh.w_rows)
            .map(|_| (rng.f32() * 3.0).to_bits())
            .collect();
        if mode.n_edge {
            for r in 0..rows {
                if r % 4 == 0 {
                    continue;
                }
                for c in 0..sh.w_rows {
                    if rng.below(64) == 0 {
                        nrm[r * sh.w_rows + c] = EDGE_N[rng.below(EDGE_N.len())];
                    }
                }
            }
        }
    }
    HostData { q, s, x, nrm }
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(gpu: &Gpu, bytes: &[u8]) -> Result<GpuTensor> {
    gpu.upload_raw(bytes, &[bytes.len()]).map_err(err)
}

struct ModeDev {
    mode: usize,
    q: GpuTensor,
    s: GpuTensor,
    x: GpuTensor,
    nrm: Option<GpuTensor>,
}

struct YBuf {
    t: GpuTensor,
    out_cap: usize,
}

impl YBuf {
    fn new(gpu: &Gpu, out_cap: usize) -> Result<Self> {
        let t = upload(gpu, &vec![GUARD_BYTE; GUARD + out_cap + GUARD])?;
        Ok(Self { t, out_cap })
    }
    fn reset(&self, gpu: &Gpu) -> Result<()> {
        gpu.hip
            .memset(&self.t.buf, GUARD_BYTE as i32, self.t.byte_size())
            .map_err(err)
    }
    fn ptr(&self) -> u64 {
        self.t.buf.as_ptr() as u64 + GUARD as u64
    }
    /// Non-0x7f bytes in the two guard bands of `buf`.
    fn guard_bad(&self, buf: &[u8]) -> usize {
        buf[..GUARD]
            .iter()
            .chain(&buf[GUARD + self.out_cap..])
            .filter(|&&b| b != GUARD_BYTE)
            .count()
    }
}

fn diff_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn first_diffs(y: &YBuf, r: &[u8], g: &[u8], case: &str, limit: usize) -> Vec<Value> {
    let mut out = Vec::new();
    for (i, (a, b)) in r.chunks(4).zip(g.chunks(4)).enumerate() {
        if a != b {
            let off = i * 4;
            let region = if off < GUARD {
                "guard_lo"
            } else if off >= GUARD + y.out_cap {
                "guard_hi"
            } else {
                "out"
            };
            out.push(json!({
                "case": case, "word": i, "region": region,
                "ref": format!("{:08x}", u32::from_le_bytes(a.try_into().unwrap_or([0; 4]))),
                "got": format!("{:08x}", u32::from_le_bytes(b.try_into().unwrap_or([0; 4]))),
            }));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn run_case(
    gpu: &Gpu,
    func: &hip_bridge::Function,
    v: &Variant,
    sh: Shape,
    d: &ModeDev,
    y: &YBuf,
    rows: usize,
) -> Result<Vec<u8>> {
    y.reset(gpu)?;
    let p = Ptrs {
        q: d.q.buf.as_ptr() as u64,
        s: d.s.buf.as_ptr() as u64,
        x: d.x.buf.as_ptr() as u64,
        nrm: d.nrm.as_ref().map_or(0, |n| n.buf.as_ptr() as u64),
        y: y.ptr(),
    };
    launch(gpu, func, v, sh, &p, rows, MODES[d.mode].act)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.download_raw_bytes(&y.t).map_err(err)
}

// ---- S5 -------------------------------------------------------------------------------------

struct S5Shape {
    sh: Shape,
    y: YBuf,
    modes: Vec<ModeDev>,
}

struct S5Ref {
    shape: usize,
    mode: usize,
    rows: usize,
    label: String,
    bytes: Vec<u8>,
}

fn s5_variant(
    gpu: &mut Gpu,
    v: &Variant,
    base: &Loaded,
    runs: &[&ManifestRun],
    seed: u64,
) -> Result<(Value, usize)> {
    let mut rng = Rng(seed ^ (v.nt as u64 * 0x9E37_79B9) ^ ((v.kind as u64) << 40));
    let shapes_list: Vec<Shape> = std::iter::once(v.served())
        .chain(v.small_shapes())
        .collect();
    let max_rows = v.max_rows();
    let mut shapes: Vec<S5Shape> = Vec::new();
    for &sh in &shapes_list {
        let mut modes = Vec::new();
        for (mi, mode) in MODES.iter().enumerate() {
            if mode.up_only && v.kind != Kind::Up {
                continue;
            }
            let h = gen_data(&mut rng, v, sh, mode);
            modes.push(ModeDev {
                mode: mi,
                q: upload(gpu, &h.q)?,
                s: upload(gpu, &words(&h.s.iter().map(|f| f.to_bits()).collect::<Vec<_>>()))?,
                x: upload(gpu, &words(&h.x))?,
                nrm: if v.kind == Kind::Up {
                    Some(upload(gpu, &words(&h.nrm))?)
                } else {
                    None
                },
            });
        }
        let y = YBuf::new(gpu, max_rows * sh.out_dim(v.kind) * 4)?;
        shapes.push(S5Shape { sh, y, modes });
    }
    // Shipped reference bytes, once per case, run twice (determinism) and sanity-checked.
    let rows_list = v.s5_rows();
    let mut refs: Vec<S5Ref> = Vec::new();
    let mut ref_nondeterministic = 0usize;
    for (si, s) in shapes.iter().enumerate() {
        for d in &s.modes {
            for &rows in &rows_list {
                let a = run_case(gpu, &base.func, v, s.sh, d, &s.y, rows)?;
                let b = run_case(gpu, &base.func, v, s.sh, d, &s.y, rows)?;
                if a != b {
                    ref_nondeterministic += 1;
                }
                if rows > max_rows {
                    if a.iter().any(|&q| q != GUARD_BYTE) {
                        return Err(format!(
                            "{}: shipped entry stored bytes for rows={rows} > 16*NT (case {} {})",
                            v.name,
                            s.sh.label(v.kind),
                            MODES[d.mode].name
                        ));
                    }
                } else if MODES[d.mode].name == "rand" && si == 0 && rows == max_rows {
                    let out = &a[GUARD..GUARD + rows * s.sh.out_dim(v.kind) * 4];
                    let finite = out.chunks(4).all(|w| {
                        f32::from_le_bytes(w.try_into().unwrap()).is_finite()
                    });
                    if out.iter().all(|&q| q == GUARD_BYTE) || !finite {
                        return Err(format!(
                            "{}: shipped rand-mode output at served shape is empty or non-finite",
                            v.name
                        ));
                    }
                }
                refs.push(S5Ref {
                    shape: si,
                    mode: d.mode,
                    rows,
                    label: format!(
                        "{} {} rows {rows}",
                        s.sh.label(v.kind),
                        MODES[d.mode].name
                    ),
                    bytes: a,
                });
            }
        }
    }
    println!(
        "s5 {}: {} cases/object, {} objects",
        v.name,
        refs.len(),
        runs.iter().map(|r| r.cands.len()).sum::<usize>()
    );
    let mut bad_total = 0usize;
    let mut objects_json = Vec::new();
    let mut n = 0usize;
    for run in runs {
        for cand in &run.cands {
            n += 1;
            let mut e = json!({"id": cand.id, "sha256": cand.sha, "axes": cand.axes,
                               "manifest": run.path});
            match load_symbol(gpu, &cand.bytes, &v.symbol(), &cand.id) {
                Err(le) => {
                    eprintln!("load failed {}: {le}", cand.id);
                    bad_total += 1;
                    e["hw_bitwise"] = json!(false);
                    e["load_error"] = json!(le);
                    objects_json.push(e);
                    continue;
                }
                Ok(obj) => {
                    let mut mism = 0usize;
                    let mut guard_bad = 0usize;
                    let mut failed = 0usize;
                    let mut by_mode = serde_json::Map::new();
                    let mut firsts: Vec<Value> = Vec::new();
                    for r in &refs {
                        let s = &shapes[r.shape];
                        let d = s.modes.iter().find(|d| d.mode == r.mode).unwrap();
                        let got = run_case(gpu, &obj.func, v, s.sh, d, &s.y, r.rows)?;
                        let bytes = diff_bytes(&got, &r.bytes);
                        guard_bad += s.y.guard_bad(&got);
                        if bytes != 0 {
                            mism += bytes;
                            failed += 1;
                            let slot = by_mode
                                .entry(MODES[r.mode].name.to_string())
                                .or_insert(json!(0));
                            *slot = json!(slot.as_u64().unwrap_or(0) + bytes as u64);
                            if firsts.len() < 8 {
                                let want = 8 - firsts.len();
                                firsts.extend(first_diffs(&s.y, &r.bytes, &got, &r.label, want));
                            }
                        }
                    }
                    bad_total += mism;
                    e["load_wrap"] = json!(obj.wrap);
                    e["hw_bitwise"] = json!(mism == 0);
                    e["mismatch_bytes"] = json!(mism);
                    e["guard_overwritten_bytes"] = json!(guard_bad);
                    e["cases_failed"] = json!(failed);
                    e["by_mode"] = Value::Object(by_mode);
                    e["first_diffs"] = json!(firsts);
                    if mism != 0 {
                        println!("  {} {}: {mism} mismatching bytes in {failed} cases", v.name, cand.id);
                    }
                    objects_json.push(e);
                }
            }
            if n % 50 == 0 {
                println!("  {} {n} objects", v.name);
            }
        }
    }
    for s in shapes {
        for d in s.modes {
            gpu.free_tensor(d.q).map_err(err)?;
            gpu.free_tensor(d.s).map_err(err)?;
            gpu.free_tensor(d.x).map_err(err)?;
            if let Some(t) = d.nrm {
                gpu.free_tensor(t).map_err(err)?;
            }
        }
        gpu.free_tensor(s.y.t).map_err(err)?;
    }
    gpu.drain_pool();
    Ok((
        json!({
            "variant": v.name, "symbol": v.symbol(), "cases_per_object": refs.len(),
            "shapes": shapes_list.iter().map(|s| s.label(v.kind)).collect::<Vec<_>>(),
            "rows": rows_list,
            "modes": MODES.iter().filter(|m| !m.up_only || v.kind == Kind::Up)
                .map(|m| m.name).collect::<Vec<_>>(),
            "ref_checks": {"nondeterministic_cases": ref_nondeterministic,
                           "overflow_rows_store_nothing": true},
            "objects": objects_json,
        }),
        bad_total + ref_nondeterministic,
    ))
}

// ---- S6 -------------------------------------------------------------------------------------

/// (formula copies, passes, calls per replay) for one shape.
fn copies_for(sh: Shape) -> (usize, usize, usize) {
    let copy_bytes = (sh.w_rows * sh.k + 4 * sh.w_rows).max(1);
    let copies = SERVED_COPY_TARGET.div_ceil(copy_bytes).max(3);
    let passes = MIN_CALLS_PER_REPLAY.div_ceil(copies);
    (copies, passes, copies * passes)
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// (median = `s[len/2]`, p10, p90) of per-call us replays.
fn stats(replays: &[f64]) -> (f64, f64, f64) {
    let mut s = replays.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], pct(&s, 0.10), pct(&s, 0.90))
}

/// Per-call us of each timed replay of `calls` launches captured in one graph.
fn graph_samples(
    gpu: &mut Gpu,
    reps: usize,
    warm: usize,
    calls: usize,
    copies: usize,
    mut f: impl FnMut(&mut Gpu, usize) -> Result<()>,
) -> Result<Vec<f64>> {
    for i in 0..copies.min(calls) {
        f(gpu, i)?;
    }
    gpu.ensure_capture_stream().map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.begin_stream_capture().map_err(err)?;
    for i in 0..calls {
        f(gpu, i)?;
    }
    let graph = gpu.end_stream_capture().map_err(err)?;
    let exec = gpu.hip.graph_instantiate(&graph).map_err(err)?;
    for _ in 0..warm {
        gpu.launch_graph(&exec).map_err(err)?;
    }
    let mut s = Vec::with_capacity(reps);
    for _ in 0..reps {
        gpu.hip.device_synchronize().map_err(err)?;
        let t = Instant::now();
        gpu.launch_graph(&exec).map_err(err)?;
        gpu.hip.device_synchronize().map_err(err)?;
        s.push(t.elapsed().as_secs_f64() * 1e6 / calls as f64);
    }
    gpu.hip.graph_exec_destroy(exec).map_err(err)?;
    gpu.hip.graph_destroy(graph).map_err(err)?;
    Ok(s)
}

struct S6Opts {
    reps: usize,
    warm: usize,
    seed: u64,
    rows_override: Option<Vec<usize>>,
}

fn s6_variant(
    gpu: &mut Gpu,
    v: &Variant,
    base: &Loaded,
    runs: &[&ManifestRun],
    o: &S6Opts,
) -> Result<(Value, usize)> {
    let sh = v.served();
    let (copies, passes, calls) = copies_for(sh);
    let copy_bytes = sh.w_rows * sh.k + 4 * sh.w_rows;
    let act = MODES[0].act;
    let mut rng = Rng(o.seed ^ (v.nt as u64 * 0x9E37_79B9) ^ ((v.kind as u64) << 40));
    let mut qs = Vec::new();
    let mut ss = Vec::new();
    for _ in 0..copies {
        qs.push(upload(gpu, &rand_q(&mut rng, sh.w_rows * sh.k))?);
        let s = rand_s(&mut rng, sh.w_rows);
        ss.push(upload(gpu, &words(&s.iter().map(|f| f.to_bits()).collect::<Vec<_>>()))?);
    }
    let h = gen_data(&mut rng, v, sh, &MODES[0]);
    let x = upload(gpu, &words(&h.x))?;
    let nrm = if v.kind == Kind::Up {
        Some(upload(gpu, &words(&h.nrm))?)
    } else {
        None
    };
    let out_cap = v.max_rows() * sh.out_dim(v.kind) * 4;
    let y_ref = YBuf::new(gpu, out_cap)?;
    let y_c = YBuf::new(gpu, out_cap)?;
    let ptrs = |i: usize, y: u64| Ptrs {
        q: qs[i % copies].buf.as_ptr() as u64,
        s: ss[i % copies].buf.as_ptr() as u64,
        x: x.buf.as_ptr() as u64,
        nrm: nrm.as_ref().map_or(0, |n| n.buf.as_ptr() as u64),
        y,
    };
    // Candidates are loaded once for the variant (a manifest of tens of objects).
    let mut objs: Vec<(&Cand, Result<Loaded>)> = Vec::new();
    for run in runs {
        for c in &run.cands {
            objs.push((c, load_symbol(gpu, &c.bytes, &v.symbol(), &c.id)));
        }
    }
    let rows_all: Vec<usize> = match &o.rows_override {
        Some(r) => r.clone(),
        None => v.served_rows(),
    };
    let (rows_list, rows_skipped): (Vec<usize>, Vec<usize>) = rows_all
        .into_iter()
        .partition(|&r| (1..=v.max_rows()).contains(&r));
    println!(
        "s6 {}: {} M={} K={} copies={copies} calls/replay={calls} rows={rows_list:?} objects={}",
        v.name,
        v.symbol(),
        sh.w_rows,
        sh.k,
        objs.len()
    );
    let mut bad = 0usize;
    let mut cells = Vec::new();
    for &rows in &rows_list {
        y_ref.reset(gpu)?;
        launch(gpu, &base.func, v, sh, &ptrs(0, y_ref.ptr()), rows, act)?;
        gpu.hip.device_synchronize().map_err(err)?;
        let ref_bytes = gpu.download_raw_bytes(&y_ref.t).map_err(err)?;
        let ref_guard_bad = y_ref.guard_bad(&ref_bytes);
        bad += ref_guard_bad;
        let mut mism: Vec<Option<usize>> = Vec::new();
        let mut timed: Vec<usize> = Vec::new();
        for (oi, (_, l)) in objs.iter().enumerate() {
            match l {
                Err(_) => {
                    mism.push(None);
                    bad += 1;
                }
                Ok(l) => {
                    y_c.reset(gpu)?;
                    launch(gpu, &l.func, v, sh, &ptrs(0, y_c.ptr()), rows, act)?;
                    gpu.hip.device_synchronize().map_err(err)?;
                    let got = gpu.download_raw_bytes(&y_c.t).map_err(err)?;
                    let d = diff_bytes(&got, &ref_bytes);
                    bad += d;
                    mism.push(Some(d));
                    if d == 0 {
                        timed.push(oi);
                    }
                }
            }
        }
        let ref_arm = |gpu: &mut Gpu| -> Result<Vec<f64>> {
            graph_samples(gpu, o.reps, o.warm, calls, copies, |g, i| {
                launch(g, &base.func, v, sh, &ptrs(i, y_ref.ptr()), rows, act)
            })
        };
        let mut ref_arms: Vec<Vec<f64>> = Vec::new();
        let mut obj_arms: Vec<Vec<f64>> = Vec::new();
        if !timed.is_empty() {
            ref_arms.push(ref_arm(gpu)?);
            for &oi in &timed {
                let l = objs[oi].1.as_ref().map_err(|e| e.clone())?;
                obj_arms.push(graph_samples(
                    gpu, o.reps, o.warm, calls, copies, |g, i| {
                        launch(g, &l.func, v, sh, &ptrs(i, y_c.ptr()), rows, act)
                    },
                )?);
                ref_arms.push(ref_arm(gpu)?);
            }
        } else {
            // No candidate to time: still record the shipped entry's A/A (two arms).
            ref_arms.push(ref_arm(gpu)?);
            ref_arms.push(ref_arm(gpu)?);
        }
        let ref_meds: Vec<f64> = ref_arms.iter().map(|a| stats(a).0).collect();
        let pooled: Vec<f64> = ref_arms.iter().flatten().copied().collect();
        let (pm, p10, p90) = stats(&pooled);
        let spread = ref_meds.iter().cloned().fold(f64::MIN, f64::max)
            - ref_meds.iter().cloned().fold(f64::MAX, f64::min);
        let ref_json = json!({
            "arms": ref_arms.len(), "reps": o.reps, "arm_medians_us": ref_meds,
            "aa_spread_us": spread, "pooled_median_us": pm, "pooled_p10_us": p10,
            "pooled_p90_us": p90, "guard_overwritten_bytes": ref_guard_bad,
            "arm_replays_us": ref_arms,
        });
        println!(
            "  {} rows={rows}: ref median {pm:.3} us/call (p10 {p10:.3} p90 {p90:.3}) A/A {spread:.3} us",
            v.name
        );
        let mut objs_json = Vec::new();
        for (oi, (c, l)) in objs.iter().enumerate() {
            let mut e = json!({"id": c.id, "sha256": c.sha, "axes": c.axes});
            match (l, mism[oi]) {
                (Err(le), _) => {
                    e["hw_bitwise"] = json!(false);
                    e["timed"] = json!(false);
                    e["load_error"] = json!(le);
                }
                (Ok(_), Some(d)) => {
                    e["hw_bitwise"] = json!(d == 0);
                    e["mismatch_bytes"] = json!(d);
                    e["timed"] = json!(d == 0);
                    if let Some(j) = timed.iter().position(|&t| t == oi) {
                        let (med, p10, p90) = stats(&obj_arms[j]);
                        e["median_us"] = json!(med);
                        e["p10_us"] = json!(p10);
                        e["p90_us"] = json!(p90);
                        e["reps"] = json!(o.reps);
                        e["ref_before_median_us"] = json!(ref_meds[j]);
                        e["ref_after_median_us"] = json!(ref_meds[j + 1]);
                        e["replays_us"] = json!(obj_arms[j]);
                        println!(
                            "    {}: {med:.3} us (p10 {p10:.3} p90 {p90:.3}) ref {:.3}/{:.3}",
                            c.id,
                            ref_meds[j],
                            ref_meds[j + 1]
                        );
                    } else {
                        println!("    {}: NOT TIMED, {d} mismatching bytes", c.id);
                    }
                }
                (Ok(_), None) => unreachable!(),
            }
            objs_json.push(e);
        }
        cells.push(json!({"rows": rows, "ref": ref_json, "objects": objs_json}));
    }
    drop(objs);
    for t in qs.into_iter().chain(ss) {
        gpu.free_tensor(t).map_err(err)?;
    }
    gpu.free_tensor(x).map_err(err)?;
    if let Some(t) = nrm {
        gpu.free_tensor(t).map_err(err)?;
    }
    gpu.free_tensor(y_ref.t).map_err(err)?;
    gpu.free_tensor(y_c.t).map_err(err)?;
    gpu.drain_pool();
    let mut shape_json = json!({"m": sh.w_rows, "k": sh.k});
    if v.kind == Kind::Up {
        shape_json["hidden"] = json!(sh.w_rows / 4);
    }
    Ok((
        json!({
            "variant": v.name, "symbol": v.symbol(), "shape": shape_json,
            "weight_copies": copies, "passes": passes, "weight_copy_bytes": copy_bytes,
            "calls_per_replay": calls, "rows_skipped": rows_skipped, "cells": cells,
        }),
        bad,
    ))
}

// ---- main -----------------------------------------------------------------------------------

fn parse_u64(s: &str) -> u64 {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    }
    .unwrap_or_else(|_| panic!("bad integer {s}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let all = |name: &str| -> Vec<String> {
        args.iter()
            .enumerate()
            .filter(|(_, a)| *a == name)
            .filter_map(|(i, _)| args.get(i + 1).cloned())
            .collect()
    };
    let (s5, s6, interleave) = (has("--s5"), has("--s6"), has("--interleave"));
    if !(s5 || s6) || (s6 && !interleave) {
        eprintln!("usage: hc8_rip_oracle (--s5 | --s6 --interleave) [--manifest PATH:VARIANT]... [--variants V,..] [--rows-list R,..] [--reps N] [--warm N] [--seed N] --out FILE");
        std::process::exit(2);
    }
    let out = flag("--out");
    let reps: usize = flag("--reps").map_or(15, |v| v.parse().expect("--reps N"));
    let warm: usize = flag("--warm").map_or(3, |v| v.parse().expect("--warm N"));
    let seed = flag("--seed").map_or(0x9E3779B97F4A7C15, |v| parse_u64(&v));
    let rows_override: Option<Vec<usize>> = flag("--rows-list").map(|l| {
        l.split(',')
            .map(|r| r.trim().parse().expect("--rows-list R,R"))
            .collect()
    });
    let manifest_specs = all("--manifest");
    let extra_variants: Vec<String> = flag("--variants")
        .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    let run = || -> Result<(Value, usize)> {
        if reps == 0 {
            return Err("--reps must be >= 1".into());
        }
        let mut manifests = Vec::new();
        for spec in &manifest_specs {
            manifests.push(read_manifest(spec)?);
        }
        let mut wanted: Vec<Variant> = Vec::new();
        for name in extra_variants
            .iter()
            .map(String::as_str)
            .chain(manifests.iter().map(|m| m.variant.name))
        {
            let v = Variant::by_name(name)?;
            if !wanted.iter().any(|w| w.name == v.name) {
                wanted.push(v);
            }
        }
        wanted.sort_by_key(|v| VARIANTS.iter().position(|w| w.name == v.name));
        if wanted.is_empty() {
            return Err("nothing to run: give --manifest PATH:VARIANT or --variants".into());
        }
        let mut gpu = Gpu::init().map_err(err)?;
        if gpu.arch != "gfx1201" {
            return Err(format!("requires exact gfx1201, got {}", gpu.arch));
        }
        let image_sha = sha256_hex(hc8::IMAGE);
        if image_sha != hc8::IMAGE_SHA256.trim() {
            return Err(format!(
                "shipped hc8 image sha256 {image_sha} != pinned {}",
                hc8::IMAGE_SHA256.trim()
            ));
        }
        let base_module = gpu.hip.module_load_data(hc8::IMAGE).map_err(err)?;
        let mut baselines = Vec::new();
        for v in &wanted {
            let func = gpu
                .hip
                .module_get_function(&base_module, &v.symbol())
                .map_err(err)?;
            baselines.push(Loaded {
                wrap: "shipped",
                _module: None,
                func,
            });
        }
        let mut bad_total = 0usize;
        let mut s5_json = Value::Null;
        let mut s6_json = Value::Null;
        if s5 {
            let mut vs = Vec::new();
            for (v, base) in wanted.iter().zip(&baselines) {
                let runs: Vec<&ManifestRun> =
                    manifests.iter().filter(|m| m.variant.name == v.name).collect();
                let (j, bad) = s5_variant(&mut gpu, v, base, &runs, seed)?;
                bad_total += bad;
                vs.push(j);
            }
            s5_json = json!({"variants": vs});
        }
        if s6 {
            let o = S6Opts {
                reps,
                warm,
                seed,
                rows_override: rows_override.clone(),
            };
            let mut vs = Vec::new();
            for (v, base) in wanted.iter().zip(&baselines) {
                let runs: Vec<&ManifestRun> =
                    manifests.iter().filter(|m| m.variant.name == v.name).collect();
                let (j, bad) = s6_variant(&mut gpu, v, base, &runs, &o)?;
                bad_total += bad;
                vs.push(j);
            }
            s6_json = json!({
                "order": "ref,c1,ref,c2,...,ref", "reps": reps, "warm": warm,
                "median": "sorted[len/2]", "p10_p90": "linear-interpolated quantiles",
                "per_call_us_of_replay": true, "variants": vs,
            });
        }
        let pci = gpu.hip.device_pci_bus_id(gpu.device_id).ok();
        let sysfs = |leaf: &str| {
            pci.as_ref().and_then(|p| {
                std::fs::read_to_string(format!(
                    "/sys/bus/pci/devices/{}/{leaf}",
                    p.to_lowercase()
                ))
                .ok()
                .map(|s| s.trim().to_string())
            })
        };
        let envv = |k: &str| std::env::var(k).ok();
        let mut modes = Vec::new();
        if s5 {
            modes.push("s5");
        }
        if s6 {
            modes.push("s6");
        }
        let result = json!({
            "schema": "hc8-rip-oracle-v1", "modes": modes, "arch": gpu.arch,
            "mismatch_bytes_total": bad_total,
            "inputs": {
                "manifests": manifests.iter().map(|m| json!({
                    "path": m.path, "variant": m.variant.name, "sha256": m.sha,
                    "objects": m.cands.len(), "skipped": m.skipped,
                })).collect::<Vec<_>>(),
                "baseline": {
                    "module": hc8::MODULE,
                    "image_sha256": sha256_hex(hc8::IMAGE),
                    "pinned_sha256": hc8::IMAGE_SHA256.trim(),
                    "symbols": wanted.iter().map(|v| (v.name.to_string(), json!(v.symbol())))
                        .collect::<serde_json::Map<_, _>>(),
                },
                "reps": reps, "warm": warm, "seed": seed,
                "calls_per_replay_rule": format!(
                    "copies = max(3, ceil({SERVED_COPY_TARGET} / copy_bytes)); calls = copies * ceil({MIN_CALLS_PER_REPLAY} / copies)"),
                "env": {
                    "FABRIC_JOB": envv("FABRIC_JOB"), "FABRIC_CARDS": envv("FABRIC_CARDS"),
                    "HIP_VISIBLE_DEVICES": envv("HIP_VISIBLE_DEVICES"),
                    "ROCR_VISIBLE_DEVICES": envv("ROCR_VISIBLE_DEVICES"),
                },
                "device": {
                    "arch": gpu.arch, "device_id": gpu.device_id, "pci_bus_id": pci,
                    "pci_vendor": sysfs("vendor"), "pci_device": sysfs("device"),
                    "pci_revision": sysfs("revision"), "product_name": sysfs("product_name"),
                },
            },
            "s5": s5_json, "s6": s6_json,
        });
        Ok((result, bad_total))
    };
    match run() {
        Ok((value, bad)) => {
            if let Some(out) = out {
                std::fs::write(&out, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
                eprintln!("wrote {out}");
            }
            eprintln!("hc8 rip oracle: {bad} mismatching bytes / failures");
            std::process::exit(i32::from(bad != 0));
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }
}
