// SPDX-License-Identifier: Apache-2.0
//! LAB-only GPU byte oracle and timing harness for the A3B FA2-GQA PeaceMaker twin
//! (`fa_prefill_pm_gfx1201`, H16/KV2/D256, Q8_0 K/V). No production dispatch.
//!
//! ```text
//! CPU only (no HIP init):
//!   fa_pm_oracle --plan            cell list, launch configs, allocation census
//!   fa_pm_oracle --self-check      CPU invariants (ABI layout, domain, fixtures, compare)
//! GPU (hold only):
//!   fa_pm_oracle --pm-bundle kernels/fa_prefill_pm_gfx1201.hxaco            byte oracle
//!   fa_pm_oracle --pm-bundle kernels/fa_prefill_pm_gfx1201.hxaco --time     timing
//!   fa_pm_oracle --launcher production   embedded launchers + owned-scratch parity
//! `--launcher raw` (default) retains the original external-bundle baseline.
//!   [--lab-hsaco PATH]  pinned lab object (default: the pinned a0-oracle-object copy)
//! ```
//!
//! Raw PM loads external bundle bytes with `hipModuleLoadData` and resolves the
//! frozen symbols. Production PM uses `Gpu`'s embedded loader and exact launcher
//! primitives for all probes, diagnostics and timing, plus an untimed admitted
//! owned-scratch wrapper check. Neither PM arm compiles a HIP fallback.
//! The lab arm loads the pinned lab object bytes (sha256
//! verified) and is never recompiled in this tool. Both arms share the same
//! runtime-written Q8_0 K/V bytes (`kv_cache_write_q8_0_batched`).
//!
//! Output: one JSON line per cell (oracle) or per timing scope, then a summary
//! line. Exit status 1 on any byte failure or error.
//!
//! On the A0 cells (standard positions, round 0) each PM cell also carries
//! independent diagnostics ported from the screen (`causal_probe`,
//! `causal_reference`): K/V positions beyond a probe row's causal horizon are
//! replaced (zeroed, or finite pattern blocks) and every PM output row at or
//! below that horizon must stay byte-identical; and a sampled f64 CPU causal
//! reference is reported for PM with lab alongside. Only the perturbation and
//! integrity checks gate their own `pass`; the byte gate stays PM vs lab.
use hip_bridge::{Function, KernargBlob, Module};
use rdna_compute::{fa_prefill_pm, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, ffi::c_void};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const H: usize = 16;
const KV: usize = 2;
const D: usize = 256;
/// f32/f16 elements of one query row (`H * D`).
const ROW: usize = H * D;
/// Q8_0 bytes of one cache position: KV heads x (D/32) blocks x 34 bytes.
const POS_BYTES: usize = KV * (D / 32) * 34;
const MAX_ROWS: usize = 8192;
const LAYERS: usize = 10;
const WARMUPS: usize = 3;
const SAMPLES: usize = 10;
const SCALE: f32 = 1.0 / 16.0;
const REL_FLOOR: f64 = 1e-6;
/// Canary bytes on each side of every device buffer.
const GUARD: usize = 4096;
const POISON: i32 = 0xFF;
const STOP_RULE_MS_PER_4096: f64 = 25.0;

const LAB_SRC: &str = include_str!("lab_kernels/fa_a3b_fa2.hip");
const LAB_SRC_SHA256: &str = "2cd432aba20ce61e28ffb9eb05f397b8be67647ee005e8a39bed7ee05d2e03fa";
const LAB_HSACO_DEFAULT: &str = "/home/kaden/qcal/release-0.4.3/a3b-pp-hold/a0-oracle-object/lab_fa_a3b_h16_kv2.ab74a5edc79cd675.hsaco";
const LAB_HSACO_SHA256: &str = "8b023c9701a053eb4491808a05c4e2443b3c4513e5bb8a528aa1b94e471ce6cf";
const LAB_PRE: &str = "attention_fa2_q_preconvert_gfx1201";
const LAB_DIRECT: &str = "attention_q8_0_fa2_gqa_gfx1201";
const PM_MODULE_KEY: &str = "fa_prefill_pm_gfx1201";
const PM_PRE: &str = "attention_fa2_q_preconvert_pm_gfx1201";
const PM_DIRECT: &str = "attention_q8_0_fa2_gqa_pm_gfx1201";
const PRE_KERNARG_BYTES: usize = 40;
const DIRECT_KERNARG_BYTES: usize = 60;
const PRE_BLOCK: u32 = 128;
const DIRECT_BLOCK: u32 = 160;
const DIRECT_LDS: u32 = 65536;

fn sha_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn bytes<T>(x: &[T]) -> &[u8] {
    // SAFETY: only initialized f32/i32/u8 slices are passed; no padding.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}
fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ---------------------------------------------------------------- domain

/// Pure host admission, run before any launch. `capacity` is the cache position
/// capacity of the cell; positions are the per-row causal positions.
fn validate_domain(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    batch: usize,
    positions: &[i32],
    capacity: usize,
) -> std::result::Result<(), String> {
    if (heads, kv_heads, head_dim) != (H, KV, D) {
        return Err(format!(
            "domain is H16/KV2/D256, got {heads}/{kv_heads}/{head_dim}"
        ));
    }
    if batch < 1 || batch > MAX_ROWS {
        return Err(format!("batch {batch} outside 1..={MAX_ROWS}"));
    }
    if positions.len() != batch {
        return Err(format!(
            "positions length {} != batch {batch}",
            positions.len()
        ));
    }
    for (row, &p) in positions.iter().enumerate() {
        if p < 0 || p == i32::MAX {
            return Err(format!("row {row}: position {p} outside 0..i32::MAX-1"));
        }
        if p as usize >= capacity {
            return Err(format!(
                "row {row}: position {p} >= cache capacity {capacity}"
            ));
        }
    }
    fa_prefill_pm::validate_positions(positions, capacity).map_err(|e| e.to_string())?;
    Ok(())
}

// -------------------------------------------------------------- fixtures

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fixture {
    Std,
    Reverse8,
    DupPairs8,
    Rotate8,
    SameGroup8,
}
impl Fixture {
    fn name(self) -> &'static str {
        match self {
            Fixture::Std => "std",
            Fixture::Reverse8 => "reverse8",
            Fixture::DupPairs8 => "dup_pairs8",
            Fixture::Rotate8 => "rotate8",
            Fixture::SameGroup8 => "same_group8",
        }
    }
    /// Offset inside the 8-row group `g` (live size `m`) of group-row `r`.
    fn offset(self, r: usize, g: usize, m: usize) -> usize {
        match self {
            Fixture::Std => r,
            Fixture::Reverse8 => m - 1 - r,
            Fixture::DupPairs8 => (r & !1) % m,
            Fixture::Rotate8 => (r * 5 + g) % 8 % m,
            Fixture::SameGroup8 => m - 1,
        }
    }
    fn positions(self, rows: usize, start: usize) -> Vec<i32> {
        (0..rows)
            .map(|row| {
                let (g, r) = (row / 8, row % 8);
                let m = (rows - g * 8).min(8);
                (start + g * 8 + self.offset(r, g, m)) as i32
            })
            .collect()
    }
    /// Round 0 uses the cell fixture; round 1 changes positions on the same allocations.
    fn for_round(self, round: usize) -> Fixture {
        match (round, self) {
            (0, f) => f,
            (_, Fixture::Std) => Fixture::Reverse8,
            _ => Fixture::Std,
        }
    }
}

// ----------------------------------------------------------------- cells

#[derive(Clone, Copy)]
struct Cell {
    rows: usize,
    start: usize,
    segment: usize,
    fixture: Fixture,
}
impl Cell {
    fn ctx(&self) -> usize {
        self.start + self.rows
    }
    fn name(&self) -> String {
        format!(
            "n{}_s{}_seg{}_{}",
            self.rows,
            self.start,
            self.segment,
            self.fixture.name()
        )
    }
    fn segments(&self) -> Vec<(usize, usize)> {
        (0..self.rows)
            .step_by(self.segment)
            .map(|off| (off, (self.rows - off).min(self.segment)))
            .collect()
    }
    /// Independent diagnostics (future-cache perturbation + sampled CPU causal
    /// reference) run on the A0 shapes at standard positions, round 0.
    fn diagnostics(&self) -> bool {
        self.fixture == Fixture::Std && A0_SHAPES.contains(&(self.rows, self.start))
    }
}

/// A0 `SHAPES`.
const A0_SHAPES: [(usize, usize); 5] = [(512, 0), (4096, 0), (8192, 0), (512, 3584), (4096, 4096)];
const BOUNDARY_SHAPES: [(usize, usize); 3] = [(64, 0), (65, 0), (8191, 0)];
const UNALIGNED_SHAPES: [(usize, usize); 2] = [(512, 1), (512, 63)];
const FIXTURE_SHAPES: [(usize, usize); 4] = [(512, 0), (512, 3584), (65, 63), (4096, 0)];
const FIXTURES: [Fixture; 4] = [
    Fixture::Reverse8,
    Fixture::DupPairs8,
    Fixture::Rotate8,
    Fixture::SameGroup8,
];
const TIME_SHAPES: [(usize, usize); 5] =
    [(4096, 0), (8192, 0), (4096, 4096), (512, 0), (512, 3584)];

/// Full-cell segment and (for rows > 512) segment 512.
fn push_modes(v: &mut Vec<Cell>, rows: usize, start: usize, fixture: Fixture) {
    v.push(Cell {
        rows,
        start,
        segment: rows,
        fixture,
    });
    if rows > 512 {
        v.push(Cell {
            rows,
            start,
            segment: 512,
            fixture,
        });
    }
}
fn oracle_cells() -> Vec<Cell> {
    let mut v = Vec::new();
    for (rows, start) in A0_SHAPES
        .into_iter()
        .chain(BOUNDARY_SHAPES)
        .chain(UNALIGNED_SHAPES)
    {
        push_modes(&mut v, rows, start, Fixture::Std);
    }
    for (rows, start) in FIXTURE_SHAPES {
        for f in FIXTURES {
            push_modes(&mut v, rows, start, f);
        }
    }
    v
}
fn time_cells() -> Vec<Cell> {
    TIME_SHAPES
        .into_iter()
        .map(|(rows, start)| Cell {
            rows,
            start,
            segment: rows,
            fixture: Fixture::Std,
        })
        .collect()
}

// ---------------------------------------------------------- memory census

/// Used bytes of every buffer for one cell (excluding guards).
#[derive(Clone, Copy, Default)]
struct Used {
    q32: usize,
    k32: usize,
    v32: usize,
    allpos: usize,
    kc: usize,
    vc: usize,
    pos: usize,
    q16: usize,
    out: usize,
}
fn used(rows: usize, ctx: usize) -> Used {
    Used {
        q32: rows * ROW * 4,
        k32: ctx * KV * D * 4,
        v32: ctx * KV * D * 4,
        allpos: ctx * 4,
        kc: ctx * POS_BYTES,
        vc: ctx * POS_BYTES,
        pos: rows * 4,
        q16: rows * ROW * 2,
        out: rows * ROW * 4,
    }
}
fn used_of(c: &Cell) -> Used {
    used(c.rows, c.ctx())
}
const BUFFER_NAMES: [&str; 12] = [
    "q32",
    "k32_src",
    "v32_src",
    "allpos",
    "k_cache_q8_0",
    "v_cache_q8_0",
    "positions",
    "q16_lab",
    "q16_pm",
    "out_lab",
    "out_pm_direct",
    "out_pm_full",
];
fn used_list(u: &Used) -> [usize; 12] {
    [
        u.q32, u.k32, u.v32, u.allpos, u.kc, u.vc, u.pos, u.q16, u.q16, u.out, u.out, u.out,
    ]
}
/// Arena = one allocation per buffer at the maximum size over `cells`, plus
/// two guards. Nothing else is allocated on the device by this tool.
fn arena_sizes(cells: &[Cell]) -> [usize; 12] {
    let mut m = [0usize; 12];
    for c in cells {
        for (slot, b) in m.iter_mut().zip(used_list(&used_of(c))) {
            *slot = (*slot).max(b);
        }
    }
    m
}
fn arena_total(cells: &[Cell]) -> usize {
    arena_sizes(cells).iter().map(|b| b + 2 * GUARD).sum()
}
/// Host phases of one oracle/timing round; each entry is the live host bytes.
fn host_phases(c: &Cell, diag: bool) -> Vec<(&'static str, usize)> {
    let u = used_of(c);
    // qh + k/v f32 generation + positions + allpos while preparing/uploading
    let prep = u.q32 + u.k32 + u.v32 + u.pos + u.allpos;
    // persistent copies for the integrity/causal comparison: q (immutable
    // snapshot), runtime-written K/V cache snapshots, positions.
    let snap = u.q32 + u.kc + u.vc + u.pos;
    let mut v = vec![
        ("prepare_generate_upload", prep),
        ("compare_q16_pair", snap + 2 * u.q16),
        ("compare_outputs_lab_plus_one", snap + 2 * u.out),
        ("integrity_q32_download", snap + u.q32),
    ];
    if diag {
        // baseline + changed output prefixes (rows 0..=rows/2) and the two
        // perturbed future-cache clones; sampled reference rows are tiny.
        let prefix = (c.rows / 2 + 1) * ROW * 4;
        v.push(("diagnostics_perturbation", snap + 2 * prefix + u.kc + u.vc));
    }
    v
}
fn host_peak(c: &Cell, mode_time: bool) -> usize {
    host_phases(c, !mode_time && c.diagnostics())
        .iter()
        .map(|p| p.1)
        .max()
        .unwrap()
}
fn launch_count(c: &Cell, mode_time: bool) -> usize {
    let s = c.segments().len();
    if mode_time {
        // untimed precheck (lab seq + pm seq) outside the timed intervals.
        4 * s
    } else {
        // lab pre+direct, pm pre+direct, pm direct from lab Q16; x2 rounds;
        // diagnostics (round 0): 6 PM pre+direct sequences.
        2 * (5 * s) + if c.diagnostics() { 12 * s } else { 0 }
    }
}

// ----------------------------------------------------------- ABI helpers

fn pre_kernargs(q: *mut c_void, q16: *mut c_void, n: usize) -> KernargBlob {
    let mut b = KernargBlob::new();
    b.push_ptr(q);
    b.push_ptr(q16);
    b.push_ptr(std::ptr::null::<c_void>()); // signs1: ignored by the ABI
    b.push_ptr(std::ptr::null::<c_void>()); // signs2: ignored by the ABI
    b.push_i32(n as i32);
    b.push_i32(0); // do_rotate: ignored by the ABI
    b
}
fn direct_kernargs(
    q16: *mut c_void,
    kc: *mut c_void,
    vc: *mut c_void,
    out: *mut c_void,
    pos: *mut c_void,
    n: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [q16, kc, vc, out, pos] {
        b.push_ptr(p);
    }
    for x in [H, KV, D, n] {
        b.push_i32(x as i32);
    }
    b.push_f32(SCALE);
    b
}
fn pre_grid(n: usize) -> [u32; 3] {
    [(n * H).div_ceil(4) as u32, 1, 1]
}
fn direct_grid(n: usize) -> [u32; 3] {
    [n.div_ceil(8) as u32, KV as u32, 1]
}
fn launch(
    gpu: &Gpu,
    f: &Function,
    grid: [u32; 3],
    block: u32,
    lds: u32,
    mut args: KernargBlob,
) -> Result<()> {
    // SAFETY: blobs match the 40/60-byte ABI asserted by --self-check; all
    // allocations outlive the launches (same-stream, synchronized before reuse).
    unsafe {
        gpu.hip
            .launch_kernel_blob(f, grid, [block, 1, 1], lds, None, args.as_mut_slice())
    }?;
    Ok(())
}

// ------------------------------------------------------------- comparison

#[derive(Clone, Copy)]
enum Elem {
    F32,
    F16,
}
impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::F16 => 2,
        }
    }
    /// (raw bits, value) of element `i`.
    fn at(self, b: &[u8], i: usize) -> (u32, f64) {
        match self {
            Elem::F32 => {
                let bits = u32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap());
                (bits, f32::from_bits(bits) as f64)
            }
            Elem::F16 => {
                let bits = u16::from_ne_bytes(b[i * 2..i * 2 + 2].try_into().unwrap());
                (bits as u32, half::f16::from_bits(bits).to_f32() as f64)
            }
        }
    }
    /// Monotone integer key for ULP distance.
    fn key(self, bits: u32) -> u64 {
        let (sign, mask) = match self {
            Elem::F32 => (0x8000_0000u32, 0xFFFF_FFFFu32),
            Elem::F16 => (0x8000u32, 0xFFFFu32),
        };
        if bits & sign != 0 {
            ((!bits) & mask) as u64
        } else {
            (bits | sign) as u64
        }
    }
}
fn coords(idx: usize) -> Value {
    json!({"index":idx,"row":idx / ROW,"head":(idx / D) % H,"dim":idx % D})
}
/// Compare `a` (lab reference) with `b` (PM) as `elem` arrays of `[row, H, D]`.
fn compare(a: &[u8], b: &[u8], elem: Elem) -> Value {
    if a.len() != b.len() || a.len() % elem.size() != 0 {
        return json!({"pass":false,"byte_identical":false,"error":"length mismatch",
            "len_a":a.len(),"len_b":b.len()});
    }
    let n = a.len() / elem.size();
    let identical = a == b;
    let (mut nf_a, mut nf_b) = (0usize, 0usize);
    let (mut count, mut first) = (0usize, None::<usize>);
    let (mut max_abs, mut max_rel, mut max_ulp) = (0f64, 0f64, 0u64);
    let (mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let (ba, va) = elem.at(a, i);
        let (bb_bits, vb) = elem.at(b, i);
        let (fa, fb) = (va.is_finite(), vb.is_finite());
        nf_a += !fa as usize;
        nf_b += !fb as usize;
        if identical {
            continue;
        }
        if ba != bb_bits {
            count += 1;
            first.get_or_insert(i);
            max_ulp = max_ulp.max(if fa && fb {
                elem.key(ba).abs_diff(elem.key(bb_bits))
            } else {
                u64::MAX
            });
        }
        if fa && fb {
            let e = (va - vb).abs();
            max_abs = max_abs.max(e);
            max_rel = max_rel.max(e / va.abs().max(REL_FLOOR));
            dot += va * vb;
            aa += va * va;
            bb += vb * vb;
        }
    }
    let finite = nf_a == 0 && nf_b == 0;
    let cosine = if identical {
        json!(1.0)
    } else if aa == 0.0 || bb == 0.0 {
        Value::Null
    } else {
        json!(dot / (aa * bb).sqrt())
    };
    let first_mismatch = first.map(|i| {
        let mut c = coords(i);
        let (ba, _) = elem.at(a, i);
        let (bb_bits, _) = elem.at(b, i);
        c["lab_bits"] = json!(format!("{ba:#x}"));
        c["pm_bits"] = json!(format!("{bb_bits:#x}"));
        c
    });
    json!({"elements":n,"element_type":match elem {Elem::F32=>"f32",Elem::F16=>"f16"},
        "byte_identical":identical,"mismatched_elements":count,"first_mismatch":first_mismatch,
        "max_abs":max_abs,"max_rel":max_rel,"relative_denominator_floor":REL_FLOOR,
        "max_ulp":max_ulp,"cosine":cosine,"nonfinite_lab":nf_a,"nonfinite_pm":nf_b,
        "all_finite":finite,"pass":identical && finite})
}
fn pass(v: &Value) -> bool {
    v["pass"] == true
}

// ------------------------------------------------------------------ rng

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
    fn normal(&mut self) -> f32 {
        ((-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos())
            as f32
    }
}

// ----------------------------------------------------------- GPU buffers

struct Buf {
    t: GpuTensor,
    es: usize,
    name: &'static str,
}
impl Buf {
    fn new(gpu: &mut Gpu, name: &'static str, bytes: usize, dtype: DType) -> Result<Self> {
        let es = dtype.size();
        assert_eq!(bytes % es, 0);
        let t = gpu.alloc_tensor(&[(bytes + 2 * GUARD) / es], dtype)?;
        Ok(Self { t, es, name })
    }
    /// Non-owning tensor view of `[off, off + len)` bytes of the payload.
    fn sub(&self, off: usize, len: usize) -> GpuTensor {
        self.t.sub_offset((GUARD + off) / self.es, len / self.es)
    }
    fn view(&self, len: usize) -> GpuTensor {
        self.sub(0, len)
    }
    fn ptr(&self, off: usize) -> *mut c_void {
        // SAFETY: offset stays inside the allocation (checked by cell sizing).
        unsafe {
            self.t
                .buf
                .as_ptr()
                .cast::<u8>()
                .add(GUARD + off)
                .cast::<c_void>() as *mut c_void
        }
    }
    fn upload(&self, gpu: &Gpu, data: &[u8]) -> Result<()> {
        gpu.hip.memcpy_htod_offset(&self.t.buf, GUARD, data)?;
        Ok(())
    }
    fn download_at(&self, gpu: &Gpu, off: usize, len: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; len];
        gpu.hip.memcpy_dtoh_at(&mut v, &self.t.buf, GUARD + off)?;
        Ok(v)
    }
    fn download(&self, gpu: &Gpu, len: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; len];
        gpu.hip.memcpy_dtoh_at(&mut v, &self.t.buf, GUARD)?;
        Ok(v)
    }
    fn poison(&self, gpu: &Gpu, len: usize) -> Result<()> {
        let v = self.view(len);
        gpu.hip.memset(&v.buf, POISON, len)?;
        Ok(())
    }
    fn pattern(&self, side: u8) -> Vec<u8> {
        (0..GUARD)
            .map(|i| (i as u8).wrapping_mul(131) ^ 0xC3 ^ side ^ (self.name.len() as u8))
            .collect()
    }
    fn arm_guards(&self, gpu: &Gpu, used: usize) -> Result<()> {
        gpu.hip
            .memcpy_htod_offset(&self.t.buf, 0, &self.pattern(1))?;
        gpu.hip
            .memcpy_htod_offset(&self.t.buf, GUARD + used, &self.pattern(2))?;
        Ok(())
    }
    fn guards_ok(&self, gpu: &Gpu, used: usize) -> Result<bool> {
        let mut lo = vec![0u8; GUARD];
        let mut hi = vec![0u8; GUARD];
        gpu.hip.memcpy_dtoh_at(&mut lo, &self.t.buf, 0)?;
        gpu.hip.memcpy_dtoh_at(&mut hi, &self.t.buf, GUARD + used)?;
        Ok(lo == self.pattern(1) && hi == self.pattern(2))
    }
}
struct Arena {
    b: Vec<Buf>,
}
impl Arena {
    fn new(gpu: &mut Gpu, cells: &[Cell]) -> Result<Self> {
        let sizes = arena_sizes(cells);
        let dtypes = [
            DType::F32,
            DType::F32,
            DType::F32,
            DType::F32,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::Raw,
            DType::Raw,
            DType::F32,
            DType::F32,
            DType::F32,
        ];
        let mut b = Vec::new();
        for i in 0..12 {
            b.push(Buf::new(gpu, BUFFER_NAMES[i], sizes[i], dtypes[i])?);
        }
        Ok(Self { b })
    }
    fn q32(&self) -> &Buf {
        &self.b[0]
    }
    fn k32(&self) -> &Buf {
        &self.b[1]
    }
    fn v32(&self) -> &Buf {
        &self.b[2]
    }
    fn allpos(&self) -> &Buf {
        &self.b[3]
    }
    fn kc(&self) -> &Buf {
        &self.b[4]
    }
    fn vc(&self) -> &Buf {
        &self.b[5]
    }
    fn pos(&self) -> &Buf {
        &self.b[6]
    }
    fn q16_lab(&self) -> &Buf {
        &self.b[7]
    }
    fn q16_pm(&self) -> &Buf {
        &self.b[8]
    }
    fn out_lab(&self) -> &Buf {
        &self.b[9]
    }
    /// Also the incumbent output scratch in timing mode.
    fn out_pm_direct(&self) -> &Buf {
        &self.b[10]
    }
    fn out_pm_full(&self) -> &Buf {
        &self.b[11]
    }
}

// ----------------------------------------------------------------- arms

struct RawArm {
    _module: Module,
    _image: Vec<u8>,
    pre: Function,
    direct: Function,
}
enum Arm {
    Raw(RawArm),
    Production,
}
fn load_arm(gpu: &Gpu, image: Vec<u8>, pre: &str, direct: &str, who: &str) -> Result<Arm> {
    let module = gpu.hip.module_load_data(&image)?;
    let pre_f = gpu
        .hip
        .module_get_function(&module, pre)
        .map_err(|e| format!("{who}: code object does not export {pre}: {e}"))?;
    let direct_f = gpu
        .hip
        .module_get_function(&module, direct)
        .map_err(|e| format!("{who}: code object does not export {direct}: {e}"))?;
    Ok(Arm::Raw(RawArm {
        _module: module,
        _image: image,
        pre: pre_f,
        direct: direct_f,
    }))
}
fn pre_seg(gpu: &mut Gpu, arm: &Arm, a: &Arena, q16: &Buf, off: usize, n: usize) -> Result<()> {
    let Arm::Raw(raw) = arm else {
        gpu.fa_prefill_pm_preconvert(
            &a.q32().sub(off * ROW * 4, n * ROW * 4),
            &q16.sub(off * ROW * 2, n * ROW * 2),
            n,
        )?;
        return Ok(());
    };
    raw_pre_seg(gpu, raw, a, q16, off, n)
}
// Keep raw handle provenance in a plain struct field, as in the frozen oracle:
// the source ABI gate traces arm.pre back to load_arm's module_get_function.
fn raw_pre_seg(gpu: &Gpu, arm: &RawArm, a: &Arena, q16: &Buf, off: usize, n: usize) -> Result<()> {
    launch(
        gpu,
        &arm.pre,
        pre_grid(n),
        PRE_BLOCK,
        0,
        pre_kernargs(a.q32().ptr(off * ROW * 4), q16.ptr(off * ROW * 2), n),
    )
}
fn direct_seg(
    gpu: &mut Gpu,
    arm: &Arm,
    a: &Arena,
    q16: &Buf,
    out: &Buf,
    off: usize,
    n: usize,
) -> Result<()> {
    let Arm::Raw(raw) = arm else {
        // Cache views exclude guards; payload is position-major Q8_0 bytes.
        let cache_bytes = a.kc().t.buf.size() - 2 * GUARD;
        gpu.fa_prefill_pm_direct(
            &q16.sub(off * ROW * 2, n * ROW * 2),
            &a.kc().view(cache_bytes),
            &a.vc().view(cache_bytes),
            &out.sub(off * ROW * 4, n * ROW * 4),
            &a.pos().sub(off * 4, n * 4),
            cache_bytes / POS_BYTES,
            n,
        )?;
        return Ok(());
    };
    raw_direct_seg(gpu, raw, a, q16, out, off, n)
}
fn raw_direct_seg(
    gpu: &Gpu,
    arm: &RawArm,
    a: &Arena,
    q16: &Buf,
    out: &Buf,
    off: usize,
    n: usize,
) -> Result<()> {
    launch(
        gpu,
        &arm.direct,
        direct_grid(n),
        DIRECT_BLOCK,
        DIRECT_LDS,
        direct_kernargs(
            q16.ptr(off * ROW * 2),
            a.kc().ptr(0),
            a.vc().ptr(0),
            out.ptr(off * ROW * 4),
            a.pos().ptr(off * 4),
            n,
        ),
    )
}
/// Complete per-segment preconvert -> direct sequence (no merge, no partials).
fn run_sequence(gpu: &mut Gpu, arm: &Arm, a: &Arena, c: &Cell, q16: &Buf, out: &Buf) -> Result<()> {
    for (off, n) in c.segments() {
        pre_seg(gpu, arm, a, q16, off, n)?;
        direct_seg(gpu, arm, a, q16, out, off, n)?;
    }
    Ok(())
}
fn run_pre(gpu: &mut Gpu, arm: &Arm, a: &Arena, c: &Cell, q16: &Buf) -> Result<()> {
    for (off, n) in c.segments() {
        pre_seg(gpu, arm, a, q16, off, n)?;
    }
    Ok(())
}
fn run_direct(gpu: &mut Gpu, arm: &Arm, a: &Arena, c: &Cell, q16: &Buf, out: &Buf) -> Result<()> {
    for (off, n) in c.segments() {
        direct_seg(gpu, arm, a, q16, out, off, n)?;
    }
    Ok(())
}

/// Untimed proof of the admitted wrapper, including Gpu-owned F4b scratch.
/// Guarded probes remain intact; timed sequences still have two launches/segment.
fn wrapper_parity(gpu: &mut Gpu, pm: &Arm, a: &Arena, c: &Cell) -> Result<Option<Value>> {
    if !matches!(pm, Arm::Production) {
        return Ok(None);
    }
    let u = used_of(c);
    gpu.attention_q8_0_fa_prefill_pm_gfx1201(
        &a.q32().view(u.q32),
        &a.kc().view(u.kc),
        &a.vc().view(u.vc),
        &a.out_pm_direct().view(u.out),
        &a.pos().view(u.pos),
        H,
        KV,
        D,
        c.ctx(),
        c.rows,
    )?;
    gpu.hip.device_synchronize()?;
    Ok(Some(compare(
        &a.out_lab().download(gpu, u.out)?,
        &a.out_pm_direct().download(gpu, u.out)?,
        Elem::F32,
    )))
}

// ------------------------------------------------------------ cell data

struct Prepared {
    qh: Vec<f32>,
    pos: Vec<i32>,
    kc: Vec<u8>,
    vc: Vec<u8>,
    used: Used,
    fixture: Fixture,
    seed: u64,
}
/// Generate inputs, arm canaries, upload, write the runtime Q8_0 caches on the
/// device and snapshot them. Everything but the kernels under test.
fn prepare(gpu: &mut Gpu, a: &Arena, c: &Cell, round: usize) -> Result<Prepared> {
    let u = used_of(c);
    let ctx = c.ctx();
    let fixture = c.fixture.for_round(round);
    let pos = fixture.positions(c.rows, c.start);
    validate_domain(H, KV, D, c.rows, &pos, ctx)?;
    let seed = fnv(&c.name()) ^ (0x41a3b060u64 + (round as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let mut rng = Rng(seed);
    let qh: Vec<f32> = (0..c.rows * ROW).map(|_| rng.normal()).collect();
    let kf: Vec<f32> = (0..ctx * KV * D).map(|_| rng.normal()).collect();
    let vf: Vec<f32> = (0..ctx * KV * D).map(|_| rng.normal() * 0.7).collect();
    let allpos: Vec<i32> = (0..ctx as i32).collect();
    for (buf, bytes_used) in a.b.iter().zip(used_list(&u)) {
        buf.arm_guards(gpu, bytes_used)?;
    }
    a.q32().upload(gpu, bytes(&qh))?;
    a.k32().upload(gpu, bytes(&kf))?;
    a.v32().upload(gpu, bytes(&vf))?;
    a.allpos().upload(gpu, bytes(&allpos))?;
    a.pos().upload(gpu, bytes(&pos))?;
    gpu.kv_cache_write_q8_0_batched(
        &a.kc().view(u.kc),
        &a.k32().view(u.k32),
        &a.allpos().view(u.allpos),
        KV,
        D,
        ctx,
    )?;
    gpu.kv_cache_write_q8_0_batched(
        &a.vc().view(u.vc),
        &a.v32().view(u.v32),
        &a.allpos().view(u.allpos),
        KV,
        D,
        ctx,
    )?;
    gpu.hip.device_synchronize()?;
    drop((kf, vf, allpos));
    let kc = a.kc().download(gpu, u.kc)?;
    let vc = a.vc().download(gpu, u.vc)?;
    for b in [
        a.q16_lab(),
        a.q16_pm(),
        a.out_lab(),
        a.out_pm_direct(),
        a.out_pm_full(),
    ] {
        let len = if b.es == 4 { u.out } else { u.q16 };
        b.poison(gpu, len)?;
    }
    Ok(Prepared {
        qh,
        pos,
        kc,
        vc,
        used: u,
        fixture,
        seed,
    })
}
fn integrity(gpu: &Gpu, a: &Arena, p: &Prepared) -> Result<Value> {
    let u = &p.used;
    let inputs = json!({
        "q32": a.q32().download(gpu, u.q32)? == bytes(&p.qh),
        "k_cache": a.kc().download(gpu, u.kc)? == p.kc,
        "v_cache": a.vc().download(gpu, u.vc)? == p.vc,
        "positions": a.pos().download(gpu, u.pos)? == bytes(&p.pos),
    });
    let mut guards = serde_json::Map::new();
    for (b, bytes_used) in a.b.iter().zip(used_list(u)) {
        guards.insert(b.name.into(), json!(b.guards_ok(gpu, bytes_used)?));
    }
    let ok = inputs.as_object().unwrap().values().all(|v| v == true)
        && guards.values().all(|v| v == true);
    Ok(
        json!({"inputs_unchanged":inputs,"guards_intact":guards,"guard_bytes_each_side":GUARD,"pass":ok}),
    )
}

// ------------------------------------------------------------ diagnostics
// Independent of the byte gate (PM vs lab). Ported from the imported screen's
// `decode`, `causal_reference` and `causal_probe` (fa_a3b_screen.rs).

const SAMPLE_HEADS: [usize; 2] = [0, H - 1];
/// The screen's sampled rows.
fn sample_rows(n: usize) -> [usize; 3] {
    [0, n / 2, n - 1]
}
/// Rows whose causal horizon splits the future cache; row 3 sits inside the
/// first 8-row group, so the horizon bisects a query group.
fn probe_rows(n: usize) -> [usize; 3] {
    [0, 3, n / 2]
}
const PERTURBATIONS: [&str; 2] = ["zero", "pattern"];
fn decode_q8(cache: &[u8], pos: usize, head: usize, d: usize) -> f64 {
    let at = ((pos * KV + head) * (D / 32) + d / 32) * 34;
    let scale = half::f16::from_bits(u16::from_ne_bytes([cache[at], cache[at + 1]])).to_f32();
    (scale * (cache[at + 2 + d % 32] as i8 as f32)) as f64
}
/// CPU f64 causal reference for the sampled rows/heads over the decoded
/// runtime Q8_0 bytes (positions are `start + row`, the Std fixture).
fn cpu_reference(p: &Prepared, c: &Cell) -> Vec<f32> {
    let mut expected = Vec::new();
    for row in sample_rows(c.rows) {
        for head in SAMPLE_HEADS {
            let kh = head / (H / KV);
            let end = c.start + row + 1;
            let mut scores = Vec::with_capacity(end);
            for pos in 0..end {
                let mut s = 0.0;
                for d in 0..D {
                    s += p.qh[(row * H + head) * D + d] as f64 * decode_q8(&p.kc, pos, kh, d);
                }
                scores.push(s / 16.0);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let denom: f64 = scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max).exp();
                    *s
                })
                .sum();
            for d in 0..D {
                let v: f64 = scores
                    .iter()
                    .enumerate()
                    .map(|(pos, s)| s * decode_q8(&p.vc, pos, kh, d))
                    .sum();
                expected.push((v / denom) as f32);
            }
        }
    }
    expected
}
/// The sampled rows/heads of one device output buffer, in `cpu_reference` order.
fn sampled_output(gpu: &Gpu, buf: &Buf, c: &Cell) -> Result<Vec<f32>> {
    let mut v = Vec::new();
    for row in sample_rows(c.rows) {
        let b = buf.download_at(gpu, row * ROW * 4, ROW * 4)?;
        let f: Vec<f32> = b
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
            .collect();
        for head in SAMPLE_HEADS {
            v.extend_from_slice(&f[head * D..(head + 1) * D]);
        }
    }
    Ok(v)
}
/// Screen `compare` fields, as a diagnostic (never an error, never the byte gate).
fn ref_compare(expected: &[f32], actual: &[f32]) -> Value {
    let (mut ma, mut mr, mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let (mut mismatch, mut nonfinite) = (0usize, 0usize);
    for (&e, &a) in expected.iter().zip(actual) {
        if !(e.is_finite() && a.is_finite()) {
            nonfinite += 1;
            mismatch += 1;
            continue;
        }
        let (e64, a64) = (e as f64, a as f64);
        let err = (e64 - a64).abs();
        ma = ma.max(err);
        mr = mr.max(err / e64.abs().max(REL_FLOOR));
        dot += e64 * a64;
        aa += e64 * e64;
        bb += a64 * a64;
        mismatch += (e.to_bits() != a.to_bits()) as usize;
    }
    json!({"elements":expected.len(),"length_matches":expected.len()==actual.len(),
        "max_abs":ma,"max_rel":mr,"relative_denominator_floor":REL_FLOOR,
        "cosine":if aa == 0.0 || bb == 0.0 { Value::Null } else { json!(dot / (aa * bb).sqrt()) },
        "mismatched_f32":mismatch,"nonfinite":nonfinite})
}
/// Replace whole Q8_0 blocks (34 bytes) with finite data: `zero` clears them
/// (as the screen's probe); `pattern` writes scale 1.0 and nonzero codes.
fn perturb_future(region: &mut [u8], kind: &str, salt: usize) {
    assert_eq!(region.len() % 34, 0);
    match kind {
        "zero" => region.fill(0),
        _ => {
            for blk in region.chunks_exact_mut(34) {
                blk[0] = 0x00;
                blk[1] = 0x3C; // f16 1.0
                for (i, b) in blk[2..].iter_mut().enumerate() {
                    *b = (i * 7 + salt * 13 + 13) as u8;
                }
            }
        }
    }
}
/// Full-cache image for one probe: the pristine bytes up to `fs`, perturbed
/// future blocks from `fs` on. Built from the pristine cache for EVERY probe,
/// so no probe inherits an earlier probe's perturbation of its causal prefix.
fn probe_cache_image(pristine: &[u8], fs: usize, kind: &str, salt: usize) -> Vec<u8> {
    let mut img = pristine.to_vec();
    perturb_future(&mut img[fs..], kind, salt);
    img
}
/// PM-arm diagnostics on the output left by the byte-gate round (round 0, Std
/// positions). Restores the caches and re-checks integrity before returning.
fn diagnostics(gpu: &mut Gpu, a: &Arena, pm: &Arm, c: &Cell, p: &Prepared) -> Result<Value> {
    let u = p.used;
    // Sampled CPU causal reference: PM and, for context, lab.
    let expected = cpu_reference(p, c);
    let pm_out = sampled_output(gpu, a.out_pm_full(), c)?;
    let lab_out = sampled_output(gpu, a.out_lab(), c)?;
    let cpu = json!({"sample_rows":sample_rows(c.rows),"sample_heads":SAMPLE_HEADS,
        "causal_prefix_only":true,"nonzero_start_pos":c.start > 0,
        "reference":"f64 over decoded runtime Q8_0 K/V bytes, scale 1/16, as the screen",
        "pm":ref_compare(&expected, &pm_out),"lab":ref_compare(&expected, &lab_out)});
    // Future-cache perturbation: rows with position <= P (rows 0..=r for the
    // Std fixture) must be byte-identical after every K/V position > P changes.
    let probes = probe_rows(c.rows);
    let max_len = (probes.iter().max().unwrap() + 1) * ROW * 4;
    let base = a.out_pm_full().download(gpu, max_len)?;
    let mut list = Vec::new();
    let mut all_ok = true;
    for kind in PERTURBATIONS {
        for &r in &probes {
            let horizon = c.start + r;
            let fs = (horizon + 1) * POS_BYTES;
            // Whole-cache upload per probe: positions <= P are the pristine
            // bytes again, whatever the previous probe changed.
            let kimg = probe_cache_image(&p.kc, fs, kind, 1);
            let vimg = probe_cache_image(&p.vc, fs, kind, 2);
            let changed_bytes = kimg.iter().zip(&p.kc).filter(|(x, y)| x != y).count()
                + vimg.iter().zip(&p.vc).filter(|(x, y)| x != y).count();
            a.kc().upload(gpu, &kimg)?;
            a.vc().upload(gpu, &vimg)?;
            let prefix_pristine =
                a.kc().download(gpu, fs)? == p.kc[..fs] && a.vc().download(gpu, fs)? == p.vc[..fs];
            a.out_pm_full().poison(gpu, u.out)?;
            run_sequence(gpu, pm, a, c, a.q16_pm(), a.out_pm_full())?;
            gpu.hip.device_synchronize()?;
            let len = (r + 1) * ROW * 4;
            let changed = a.out_pm_full().download(gpu, len)?;
            let cmp = compare(&base[..len], &changed, Elem::F32);
            let ok = pass(&cmp) && changed_bytes > 0 && prefix_pristine;
            all_ok &= ok;
            list.push(json!({"perturbation":kind,"probe_row":r,"probe_position":horizon,
                "future_positions_changed":[horizon + 1, c.ctx()],"cache_bytes_changed":changed_bytes,
                "causal_prefix_cache_pristine":prefix_pristine,
                "rows_compared":r + 1,"comparison":cmp,"pass":ok}));
        }
    }
    a.kc().upload(gpu, &p.kc)?;
    a.vc().upload(gpu, &p.vc)?;
    let restored = a.kc().download(gpu, u.kc)? == p.kc && a.vc().download(gpu, u.vc)? == p.vc;
    let integ = integrity(gpu, a, p)?;
    let ok = all_ok && restored && pass(&integ);
    Ok(
        json!({"scope":"independent diagnostics on PM; not the byte gate (byte gate = PM vs lab)",
        "cpu_causal_reference":cpu,
        "future_cache_perturbation":{"perturbations":list,"caches_restored":restored,"pass":all_ok && restored},
        "integrity_after":integ,"pass":ok}),
    )
}

fn oracle_round(
    gpu: &mut Gpu,
    a: &Arena,
    lab: &Arm,
    pm: &Arm,
    c: &Cell,
    round: usize,
) -> Result<Value> {
    let p = prepare(gpu, a, c, round)?;
    let u = p.used;
    // Lab sequence, PM sequence (independent Q16 + output), PM direct from lab Q16.
    run_sequence(gpu, lab, a, c, a.q16_lab(), a.out_lab())?;
    run_sequence(gpu, pm, a, c, a.q16_pm(), a.out_pm_full())?;
    run_direct(gpu, pm, a, c, a.q16_lab(), a.out_pm_direct())?;
    gpu.hip.device_synchronize()?;
    // (a) preconvert, every Q16 byte
    let pre = {
        let lab_q = a.q16_lab().download(gpu, u.q16)?;
        let pm_q = a.q16_pm().download(gpu, u.q16)?;
        compare(&lab_q, &pm_q, Elem::F16)
    };
    // (b) direct from identical Q16, (c) full sequences; lab output held once.
    let out_lab = a.out_lab().download(gpu, u.out)?;
    let direct = compare(
        &out_lab,
        &a.out_pm_direct().download(gpu, u.out)?,
        Elem::F32,
    );
    let full = compare(&out_lab, &a.out_pm_full().download(gpu, u.out)?, Elem::F32);
    drop(out_lab);
    let wrapper = wrapper_parity(gpu, pm, a, c)?;
    let integ = integrity(gpu, a, &p)?;
    let diag = if round == 0 && c.diagnostics() {
        Some(diagnostics(gpu, a, pm, c, &p)?)
    } else {
        None
    };
    let ok = pass(&pre)
        && pass(&direct)
        && pass(&full)
        && pass(&integ)
        && diag.as_ref().is_none_or(pass);
    let ok = ok && wrapper.as_ref().is_none_or(pass);
    Ok(
        json!({"round":round,"seed":p.seed,"positions_fixture":p.fixture.name(),
        "preconvert_q16":pre,"direct_same_q16":direct,"full_sequence":full,"production_wrapper":wrapper,
        "integrity":integ,"diagnostics":diag,"pass":ok}),
    )
}

// --------------------------------------------------------------- timing

fn median(x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) * 0.5
    }
}
type ArmFn<'a> = &'a mut dyn FnMut(&mut Gpu) -> Result<()>;
/// 3 warmups per arm, then 10 event-timed samples per arm; arm order rotates
/// with every warmup/sample so no arm is always first.
fn time_arms(gpu: &mut Gpu, arms: &mut [(&'static str, ArmFn)]) -> Result<Vec<Vec<f64>>> {
    let n = arms.len();
    for w in 0..WARMUPS {
        for k in 0..n {
            (arms[(w + k) % n].1)(gpu)?;
        }
    }
    gpu.hip.device_synchronize()?;
    let (ea, eb) = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let mut samples = vec![Vec::with_capacity(SAMPLES); n];
    for s in 0..SAMPLES {
        for k in 0..n {
            let i = (s + k) % n;
            gpu.hip.event_record(&ea, None)?;
            (arms[i].1)(gpu)?;
            gpu.hip.event_record(&eb, None)?;
            gpu.hip.event_synchronize(&eb)?;
            samples[i].push(gpu.hip.event_elapsed_ms(&ea, &eb)? as f64);
        }
    }
    gpu.hip.event_destroy(ea)?;
    gpu.hip.event_destroy(eb)?;
    Ok(samples)
}
fn scope_line(
    c: &Cell,
    scope: &str,
    note: &str,
    names: &[&'static str],
    samples: &[Vec<f64>],
    compare_to_incumbent: bool,
) -> Value {
    let norm = 4096.0 / c.rows as f64;
    let inc = names
        .iter()
        .position(|n| *n == "incumbent")
        .map(|i| median(&samples[i]));
    let mut arms = serde_json::Map::new();
    for (name, s) in names.iter().zip(samples) {
        let m = median(s);
        let mut v = json!({"per_layer_ms_all":s,"per_layer_ms_median":m,
            "envelope_ms_total":m * LAYERS as f64,
            "envelope_ms_per_4096_rows":m * LAYERS as f64 * norm});
        if let (true, Some(i)) = (compare_to_incumbent, inc) {
            let saved = (i - m) * LAYERS as f64 * norm;
            v["saving_vs_incumbent_envelope_ms_per_4096_rows"] = json!(saved);
            v["meets_25ms_per_4096_stop_rule"] = json!(saved >= STOP_RULE_MS_PER_4096);
        }
        arms.insert((*name).into(), v);
    }
    json!({"kind":"timing","cell":c.name(),"rows":c.rows,"start_pos":c.start,
        "segment_rows":c.segment,"segments":c.segments().len(),"scope":scope,"scope_note":note,
        "layers":LAYERS,"warmups_per_arm":WARMUPS,"samples_per_arm":SAMPLES,
        "arm_order":"rotated: warmup w and sample s start at arm (index mod arms)",
        "normalization":format!("envelope_ms_per_4096_rows = median*{LAYERS}*4096/{}", c.rows),
        "arms":arms})
}
fn time_cell(
    gpu: &mut Gpu,
    a: &Arena,
    lab: &Arm,
    pm: &Arm,
    c: &Cell,
    lines: &mut Vec<Value>,
) -> Result<bool> {
    let p = prepare(gpu, a, c, 0)?;
    let u = p.used;
    let ctx = c.ctx();
    // Untimed byte precheck of the exact sequences about to be timed.
    run_sequence(gpu, lab, a, c, a.q16_lab(), a.out_lab())?;
    run_sequence(gpu, pm, a, c, a.q16_pm(), a.out_pm_full())?;
    gpu.hip.device_synchronize()?;
    let q16_cmp = compare(
        &a.q16_lab().download(gpu, u.q16)?,
        &a.q16_pm().download(gpu, u.q16)?,
        Elem::F16,
    );
    let out_cmp = compare(
        &a.out_lab().download(gpu, u.out)?,
        &a.out_pm_full().download(gpu, u.out)?,
        Elem::F32,
    );
    let wrapper = wrapper_parity(gpu, pm, a, c)?;
    let ok = pass(&q16_cmp) && pass(&out_cmp) && wrapper.as_ref().is_none_or(pass);
    lines.push(json!({"kind":"timing_precheck","cell":c.name(),
        "preconvert_q16":q16_cmp,"full_sequence":out_cmp,"production_wrapper":wrapper,"pass":ok}));
    let (qv, kcv, vcv, incout, posv) = (
        a.q32().view(u.q32),
        a.kc().view(u.kc),
        a.vc().view(u.vc),
        a.out_pm_direct().view(u.out),
        a.pos().view(u.pos),
    );
    let ksrc = a.k32().sub(c.start * KV * D * 4, c.rows * KV * D * 4);
    let vsrc = a.v32().sub(c.start * KV * D * 4, c.rows * KV * D * 4);
    let rows = c.rows;
    let mut inc = |g: &mut Gpu| -> Result<()> {
        g.attention_q8_0_flash_prefill_wmma_slots(
            &qv, &kcv, &vcv, &incout, &posv, H, KV, D, ctx, rows, None, None, None, None,
        )?;
        Ok(())
    };
    let mut lab_direct = |g: &mut Gpu| run_direct(g, lab, a, c, a.q16_lab(), a.out_lab());
    let mut pm_direct = |g: &mut Gpu| run_direct(g, pm, a, c, a.q16_pm(), a.out_pm_full());
    let mut lab_pre = |g: &mut Gpu| run_pre(g, lab, a, c, a.q16_lab());
    let mut pm_pre = |g: &mut Gpu| run_pre(g, pm, a, c, a.q16_pm());
    let mut lab_seq = |g: &mut Gpu| run_sequence(g, lab, a, c, a.q16_lab(), a.out_lab());
    let mut pm_seq = |g: &mut Gpu| run_sequence(g, pm, a, c, a.q16_pm(), a.out_pm_full());
    // Scope 1: direct core on prepared Q16 (Q16 prepared above, untimed).
    let names = ["incumbent", "lab_direct", "pm_direct"];
    let s = time_arms(
        gpu,
        &mut [
            ("incumbent", &mut inc),
            ("lab_direct", &mut lab_direct),
            ("pm_direct", &mut pm_direct),
        ],
    )?;
    lines.push(scope_line(
        c,
        "direct_core_prepared_q16",
        "lab/pm: direct kernel only on prepared Q16; incumbent has no Q16, so its value is its complete attention kernel on F32 Q (not comparable as a saving)",
        &names,
        &s,
        false,
    ));
    // Scope 2: preconvert alone.
    let names = ["lab_preconvert", "pm_preconvert"];
    let s = time_arms(
        gpu,
        &mut [
            ("lab_preconvert", &mut lab_pre),
            ("pm_preconvert", &mut pm_pre),
        ],
    )?;
    lines.push(scope_line(
        c,
        "preconvert_alone",
        "Q32->Q16 only; incumbent has no preconvert",
        &names,
        &s,
        false,
    ));
    // Scope 3: preconvert + direct.
    let names = ["incumbent", "lab_pre_direct", "pm_pre_direct"];
    let s = time_arms(
        gpu,
        &mut [
            ("incumbent", &mut inc),
            ("lab_pre_direct", &mut lab_seq),
            ("pm_pre_direct", &mut pm_seq),
        ],
    )?;
    lines.push(scope_line(
        c,
        "preconvert_plus_direct",
        "complete attention sequence without KV writes",
        &names,
        &s,
        true,
    ));
    // Scope 4: two KV writes + complete attention.
    let writes = |g: &mut Gpu| -> Result<()> {
        g.kv_cache_write_q8_0_batched(&kcv, &ksrc, &posv, KV, D, rows)?;
        g.kv_cache_write_q8_0_batched(&vcv, &vsrc, &posv, KV, D, rows)?;
        Ok(())
    };
    let mut inc_w = |g: &mut Gpu| -> Result<()> {
        writes(g)?;
        g.attention_q8_0_flash_prefill_wmma_slots(
            &qv, &kcv, &vcv, &incout, &posv, H, KV, D, ctx, rows, None, None, None, None,
        )?;
        Ok(())
    };
    let mut lab_w = |g: &mut Gpu| -> Result<()> {
        g.kv_cache_write_q8_0_batched(&kcv, &ksrc, &posv, KV, D, rows)?;
        g.kv_cache_write_q8_0_batched(&vcv, &vsrc, &posv, KV, D, rows)?;
        run_sequence(g, lab, a, c, a.q16_lab(), a.out_lab())
    };
    let mut pm_w = |g: &mut Gpu| -> Result<()> {
        g.kv_cache_write_q8_0_batched(&kcv, &ksrc, &posv, KV, D, rows)?;
        g.kv_cache_write_q8_0_batched(&vcv, &vsrc, &posv, KV, D, rows)?;
        run_sequence(g, pm, a, c, a.q16_pm(), a.out_pm_full())
    };
    let names = ["incumbent", "lab_kv_attention", "pm_kv_attention"];
    let s = time_arms(
        gpu,
        &mut [
            ("incumbent", &mut inc_w),
            ("lab_kv_attention", &mut lab_w),
            ("pm_kv_attention", &mut pm_w),
        ],
    )?;
    lines.push(scope_line(
        c,
        "kv_writes_plus_attention",
        "two runtime Q8_0 KV writes + complete attention sequence (the screen's headline scope)",
        &names,
        &s,
        true,
    ));
    Ok(ok)
}

// ------------------------------------------------------------------ plan

fn memory_census(cells: &[Cell], mode_time: bool, production: bool) -> Value {
    let sizes = arena_sizes(cells);
    let buffers: Vec<Value> = BUFFER_NAMES
        .iter()
        .zip(sizes)
        .map(|(n, b)| json!({"name":n,"payload_bytes":b,"guards_bytes":2*GUARD,"allocated_bytes":b+2*GUARD}))
        .collect();
    let device = arena_total(cells);
    let owned_q16 = if production {
        cells.iter().map(|c| c.rows * ROW * 2).max().unwrap_or(0)
    } else {
        0
    };
    let peak_host = cells
        .iter()
        .map(|c| host_peak(c, mode_time))
        .max()
        .unwrap_or(0);
    let per_cell: Vec<Value> = cells
        .iter()
        .map(|c| {
            let u = used_of(c);
            let dev_used: usize = used_list(&u).iter().sum();
            json!({"cell":c.name(),
                "device_used_payload_bytes":{"q32":u.q32,"k_src_f32":u.k32,"v_src_f32":u.v32,"allpos":u.allpos,
                    "k_cache":u.kc,"v_cache":u.vc,"positions":u.pos,"q16_lab":u.q16,"q16_pm":u.q16,
                    "out_lab":u.out,"out_pm_direct":u.out,"out_pm_full":u.out,"total":dev_used},
                "device_guard_bytes":12*2*GUARD,
                "device_split_partials_bytes":0,
                "device_snapshots_bytes":0,
                "host_phases_bytes":host_phases(c, !mode_time && c.diagnostics()).iter().map(|(n,b)| json!({"phase":n,"bytes":b})).collect::<Vec<_>>(),
                "host_peak_bytes":host_peak(c, mode_time),
                "launches":launch_count(c, mode_time) + if production { if mode_time { 2 } else { 4 } } else { 0 }})
        })
        .collect();
    json!({"mode":if mode_time {"time"} else {"oracle"},
        "device_arena":buffers,"device_arena_total_bytes":device,
        "device_owned_fa2_q16_scratch_bytes":owned_q16,
        "device_peak_bytes":device + owned_q16,
        "device_split_partials":"none: the PM and lab-direct arms never use the split partial/merge scratch the original screen allocates (n*H*8*(D+1)*4 = up to 8192*16*8*257*4 B); it is not allocated by this tool",
        "device_immutable_input_snapshots":"host-side only (see host_phases); no device snapshot buffers",
        "device_not_counted":["HIP context","lab code object 35312 B","PM bundle (size from the file)","incumbent flash-prefill module JIT and its code","kernarg blobs"],
        "host_peak_bytes":peak_host,
        "host_copies":"q32 snapshot, runtime-written K/V Q8_0 snapshots, positions, plus two same-size downloads at the compare step (lab output held once, PM outputs compared one at a time); K/V/allpos F32 sources dropped after upload",
        "per_cell":per_cell})
}
fn plan(production: bool) -> Result<()> {
    let oracle = oracle_cells();
    let time = time_cells();
    for c in &oracle {
        let segs = c.segments();
        let last = *segs.last().unwrap();
        println!(
            "{}",
            json!({"plan_cell":c.name(),"mode":"oracle","rows":c.rows,"start_pos":c.start,
                "segment_rows":c.segment,"segments":segs.len(),"last_segment_rows":last.1,
                "positions_fixture_round0":c.fixture.name(),
                "positions_fixture_round1":c.fixture.for_round(1).name(),
                "rounds":2,
                "absolute_positions_round0":[c.fixture.positions(c.rows,c.start).iter().min(),
                    c.fixture.positions(c.rows,c.start).iter().max()],
                "cache_positions":c.ctx(),
                "preconvert":{"symbol":PM_PRE,"lab_symbol":LAB_PRE,"grid_first_segment":pre_grid(segs[0].1),
                    "block":PRE_BLOCK,"lds_bytes":0,"kernarg_bytes":PRE_KERNARG_BYTES},
                "direct":{"symbol":PM_DIRECT,"lab_symbol":LAB_DIRECT,"grid_first_segment":direct_grid(segs[0].1),
                    "block":DIRECT_BLOCK,"dynamic_lds_bytes":DIRECT_LDS,"kernarg_bytes":DIRECT_KERNARG_BYTES},
                "checks":["a: lab vs PM preconvert, every Q16 byte","b: PM direct vs lab direct from identical lab Q16",
                    "c: full PM pre->direct vs full lab pre->direct, independent outputs","e: inputs/caches/positions unchanged, guards intact, round 1 changed inputs+positions"],
                "diagnostics":if c.diagnostics() {json!({"round":0,
                    "future_cache_perturbation":{"probe_rows":probe_rows(c.rows),"kinds":PERTURBATIONS,
                        "rule":"K/V positions > start+probe_row replaced; PM output rows 0..=probe_row must stay byte-identical",
                        "pm_sequences":PERTURBATIONS.len()*3},
                    "cpu_causal_reference":{"sample_rows":sample_rows(c.rows),"sample_heads":SAMPLE_HEADS,"arms":["pm","lab"]}})} else {Value::Null}})
        );
    }
    for c in &time {
        println!(
            "{}",
            json!({"plan_cell":c.name(),"mode":"time","rows":c.rows,"start_pos":c.start,
                "segment_rows":c.segment,"warmups_per_arm":WARMUPS,"samples_per_arm":SAMPLES,"layers":LAYERS,
                "envelope_normalization_per_4096_rows":4096.0 / c.rows as f64,
                "scopes":["direct_core_prepared_q16","preconvert_alone","preconvert_plus_direct","kv_writes_plus_attention"],
                "arms":["incumbent (attention_q8_0_flash_prefill_wmma via _slots, as the screen runs it)","lab (pinned object)","pm (bundle)"]})
        );
    }
    let mo = memory_census(&oracle, false, production);
    let mt = memory_census(&time, true, production);
    println!(
        "{}",
        json!({"plan":true,"gpu_launched":false,"module_key":PM_MODULE_KEY,
        "launcher":if production {"production"} else {"raw"},
        "production_wrapper_parity":production,
        "production_wrapper_scope":"untimed full-cell owned-Q16 preconvert+direct; guarded primitive sequence remains the timed PM arm",
        "pm_symbols":[PM_PRE,PM_DIRECT],"lab_source_sha256":LAB_SRC_SHA256,
        "lab_hsaco_default":LAB_HSACO_DEFAULT,"lab_hsaco_sha256":LAB_HSACO_SHA256,
        "oracle_cells":oracle.len(),"time_cells":time.len(),"diagnostic_cells":oracle.iter().filter(|c| c.diagnostics()).count(),
        "output_schema":{
            "oracle_cell":"{kind:oracle_cell,cell,rows,start_pos,segment_rows,segments,fixture,module_key,rounds:[round],byte_identical}",
            "round":"{round,seed,positions_fixture,preconvert_q16,direct_same_q16,full_sequence,production_wrapper,integrity,diagnostics,pass}; production_wrapper is null for raw, else a comparison of the admitted owned-scratch wrapper; compare objects: {elements,element_type,byte_identical,mismatched_elements,first_mismatch{index,row,head,dim,lab_bits,pm_bits},max_abs,max_rel,relative_denominator_floor,max_ulp,cosine,nonfinite_lab,nonfinite_pm,all_finite,pass}",
            "diagnostics":"round 0 of A0 Std cells, else null: {scope,cpu_causal_reference{sample_rows,sample_heads,causal_prefix_only,nonzero_start_pos,reference,pm{elements,max_abs,max_rel,relative_denominator_floor,cosine,mismatched_f32,nonfinite},lab{...}},future_cache_perturbation{perturbations:[{perturbation,probe_row,probe_position,future_positions_changed,cache_bytes_changed,rows_compared,comparison,pass}],caches_restored,pass},integrity_after,pass}; diagnostics gate only their own pass, the byte gate stays PM vs lab",
            "timing":"{kind:timing,cell,rows,start_pos,segment_rows,scope,arms{name:{per_layer_ms_all,per_layer_ms_median,envelope_ms_total,envelope_ms_per_4096_rows,saving_vs_incumbent_envelope_ms_per_4096_rows?,meets_25ms_per_4096_stop_rule?}}}",
            "summary":"{summary:true,mode,launcher,pm_bundle_sha256,lab_object_sha256,cells,byte_identical,failures}"},
        "memory_oracle":mo,"memory_time":mt})
    );
    let dev = arena_total(&oracle).max(arena_total(&time))
        + if production { MAX_ROWS * ROW * 2 } else { 0 };
    let host = oracle
        .iter()
        .map(|c| host_peak(c, false))
        .chain(time.iter().map(|c| host_peak(c, true)))
        .max()
        .unwrap();
    println!(
        "{}",
        json!({"census":true,"peak_device_bytes":dev,"peak_host_bytes":host})
    );
    Ok(())
}

// ------------------------------------------------------------ self-check

fn self_check() -> usize {
    let mut n = 0usize;
    let mut ok = |cond: bool, what: &str| {
        assert!(cond, "self-check failed: {what}");
        n += 1;
    };
    ok(
        sha_hex(LAB_SRC.as_bytes()) == LAB_SRC_SHA256,
        "lab source sha256 pinned",
    );
    ok(
        DType::Raw.size() == 1 && DType::F32.size() == 4,
        "dtype sizes used by Buf",
    );
    // Kernarg layouts.
    let p = |x: usize| x as *mut c_void;
    let b = pre_kernargs(p(0x1000), p(0x2000), 77);
    let s = b.as_bytes();
    ok(s.len() == PRE_KERNARG_BYTES, "preconvert kernarg size 40");
    ok(
        usize::from_ne_bytes(s[0..8].try_into().unwrap()) == 0x1000,
        "pre q @0",
    );
    ok(
        usize::from_ne_bytes(s[8..16].try_into().unwrap()) == 0x2000,
        "pre q16 @8",
    );
    ok(s[16..32].iter().all(|&x| x == 0), "pre signs @16/@24 null");
    ok(
        i32::from_ne_bytes(s[32..36].try_into().unwrap()) == 77,
        "pre batch @32",
    );
    ok(
        i32::from_ne_bytes(s[36..40].try_into().unwrap()) == 0,
        "pre do_rotate @36",
    );
    let b = direct_kernargs(p(1), p(2), p(3), p(4), p(5), 99);
    let s = b.as_bytes();
    ok(s.len() == DIRECT_KERNARG_BYTES, "direct kernarg size 60");
    for i in 0..5 {
        ok(
            usize::from_ne_bytes(s[i * 8..i * 8 + 8].try_into().unwrap()) == i + 1,
            "direct pointer order q16,k,v,out,pos",
        );
    }
    for (i, want) in [16, 2, 256, 99].into_iter().enumerate() {
        let o = 40 + i * 4;
        ok(
            i32::from_ne_bytes(s[o..o + 4].try_into().unwrap()) == want,
            "direct i32 args @40..52",
        );
    }
    ok(
        f32::from_ne_bytes(s[56..60].try_into().unwrap()) == 1.0 / 16.0,
        "direct scale @56",
    );
    ok(
        pre_grid(8192) == [32768, 1, 1] && direct_grid(8191) == [1024, 2, 1],
        "grids",
    );
    // Domain truth table.
    let std = Fixture::Std.positions(64, 0);
    ok(
        validate_domain(16, 2, 256, 64, &std, 64).is_ok(),
        "domain accepts 64 rows",
    );
    ok(
        validate_domain(24, 4, 256, 64, &std, 64).is_err(),
        "domain refuses H24/KV4",
    );
    ok(
        validate_domain(16, 4, 256, 64, &std, 64).is_err(),
        "domain refuses KV4",
    );
    ok(
        validate_domain(16, 2, 128, 64, &std, 64).is_err(),
        "domain refuses D128",
    );
    ok(
        validate_domain(16, 2, 256, 0, &[], 64).is_err(),
        "domain refuses batch 0",
    );
    ok(
        validate_domain(16, 2, 256, 8193, &vec![0; 8193], 9000).is_err(),
        "domain refuses batch 8193",
    );
    ok(
        validate_domain(16, 2, 256, 63, &std, 64).is_err(),
        "domain refuses length mismatch",
    );
    ok(
        validate_domain(16, 2, 256, 1, &[-1], 64).is_err(),
        "domain refuses negative position",
    );
    ok(
        validate_domain(16, 2, 256, 1, &[64], 64).is_err(),
        "domain refuses position >= capacity",
    );
    ok(
        validate_domain(16, 2, 256, 1, &[i32::MAX], usize::MAX).is_err(),
        "domain refuses i32::MAX position",
    );
    ok(
        validate_domain(16, 2, 256, 1, &[1], 2).is_ok(),
        "domain accepts tiny valid",
    );
    // Cells, fixtures, segmentation, sizes.
    let cells = oracle_cells();
    ok(cells.len() == 34, "oracle cell count 34");
    ok(time_cells().len() == 5, "timing cell count 5");
    ok(
        cells.iter().filter(|c| c.diagnostics()).count() == 8,
        "diagnostic cell count 8 (A0 shapes x full/seg512)",
    );
    for c in cells.iter().filter(|c| c.diagnostics()) {
        for &r in probe_rows(c.rows).iter().chain(&sample_rows(c.rows)) {
            ok(r < c.rows, "diagnostic rows in range");
        }
        for &r in &probe_rows(c.rows) {
            ok(
                c.start + r + 1 < c.ctx(),
                "probe horizon leaves future positions",
            );
        }
    }
    let mut region = vec![7u8; 34 * 3];
    perturb_future(&mut region, "pattern", 1);
    ok(
        region
            .chunks_exact(34)
            .all(|b| half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32() == 1.0)
            && region[2..34].iter().any(|&x| x != 0),
        "pattern perturbation is finite scale 1.0 with nonzero codes",
    );
    perturb_future(&mut region, "zero", 1);
    ok(
        region.iter().all(|&x| x == 0),
        "zero perturbation clears blocks",
    );
    // Probe images: every probe starts from the pristine cache, so the causal
    // prefix is unchanged for each probe in sequence (the j1444 bug was probe
    // P=3 inheriting P=0's perturbation of positions 1..3).
    let pristine: Vec<u8> = (0..POS_BYTES * 16).map(|i| (i * 31 + 5) as u8).collect();
    for kind in PERTURBATIONS {
        for &r in &[0usize, 3, 8] {
            let fs = (r + 1) * POS_BYTES;
            let img = probe_cache_image(&pristine, fs, kind, 1);
            ok(
                img.len() == pristine.len() && img[..fs] == pristine[..fs],
                "probe image keeps the causal prefix pristine",
            );
            ok(
                img[fs..] != pristine[fs..],
                "probe image changes the future",
            );
        }
    }
    let mut blk = vec![0u8; 34];
    blk[0..2].copy_from_slice(&0x3C00u16.to_ne_bytes());
    blk[2 + 5] = 0xFE; // -2
    ok(decode_q8(&blk, 0, 0, 5) == -2.0, "decode_q8 block layout");
    ok(
        ref_compare(&[1.0, 2.0], &[1.0, 2.0])["mismatched_f32"] == 0
            && ref_compare(&[1.0, 2.0], &[1.0, 2.5])["mismatched_f32"] == 1,
        "ref_compare counts",
    );
    let (max_rows, max_ctx) = cells
        .iter()
        .chain(time_cells().iter())
        .fold((0, 0), |(r, c), x| (r.max(x.rows), c.max(x.ctx())));
    ok(max_rows <= MAX_ROWS && max_ctx == 8192, "cell extents");
    for c in cells.iter().chain(time_cells().iter()) {
        let u = used_of(c);
        let mut seen = 0;
        for (off, rows) in c.segments() {
            ok(
                off == seen && rows >= 1 && rows <= c.segment,
                "segments tile rows",
            );
            ok(
                (off + rows) * ROW * 4 <= u.out && (off + rows) * ROW * 2 <= u.q16,
                "segment bounds",
            );
            seen += rows;
        }
        ok(seen == c.rows, "segments cover rows");
        for round in 0..2 {
            let f = c.fixture.for_round(round);
            let pos = f.positions(c.rows, c.start);
            ok(
                validate_domain(16, 2, 256, c.rows, &pos, c.ctx()).is_ok(),
                "fixture positions valid",
            );
            // Group-local: every position lies in [start+8g, start+8g+m).
            for (row, &x) in pos.iter().enumerate() {
                let g = row / 8;
                let m = (c.rows - g * 8).min(8);
                let lo = (c.start + g * 8) as i32;
                ok(
                    x >= lo && x < lo + m as i32,
                    "fixture position in group window",
                );
            }
        }
        ok(
            used_list(&u).iter().all(|&b| b % 4 == 0),
            "buffer bytes 4-aligned",
        );
    }
    let f = |fx: Fixture| fx.positions(64, 100);
    let non_monotone = |v: &[i32]| v.chunks(8).any(|g| g.windows(2).any(|w| w[1] < w[0]));
    let dup = |v: &[i32]| v.chunks(8).any(|g| g.windows(2).any(|w| w[1] == w[0]));
    ok(non_monotone(&f(Fixture::Reverse8)), "reverse8 non-monotone");
    ok(dup(&f(Fixture::DupPairs8)), "dup_pairs8 duplicates");
    ok(non_monotone(&f(Fixture::Rotate8)), "rotate8 non-monotone");
    ok(dup(&f(Fixture::SameGroup8)), "same_group8 duplicates");
    ok(
        !non_monotone(&f(Fixture::Std)) && !dup(&f(Fixture::Std)),
        "std monotone",
    );
    // Compare.
    let a32: Vec<f32> = (0..2 * ROW).map(|i| (i as f32) * 0.25 + 1.0).collect();
    let mut b32 = a32.clone();
    ok(
        pass(&compare(bytes(&a32), bytes(&b32), Elem::F32)),
        "identical passes",
    );
    b32[ROW + 3 * D + 7] = f32::from_bits(b32[ROW + 3 * D + 7].to_bits() + 3);
    let r = compare(bytes(&a32), bytes(&b32), Elem::F32);
    ok(
        !pass(&r) && r["mismatched_elements"] == 1,
        "single flip fails with count 1",
    );
    ok(
        r["first_mismatch"]["row"] == 1
            && r["first_mismatch"]["head"] == 3
            && r["first_mismatch"]["dim"] == 7,
        "first mismatch coordinates",
    );
    ok(r["max_ulp"] == 3, "max ulp 3");
    b32[0] = f32::NAN;
    let r = compare(bytes(&a32), bytes(&b32), Elem::F32);
    ok(!pass(&r) && r["nonfinite_pm"] == 1, "nonfinite detected");
    let h16: Vec<u16> = vec![0x3c00, 0x0000, 0x8000];
    let mut g16 = h16.clone();
    g16[1] = 0x8000;
    let r = compare(bytes(&h16), bytes(&g16), Elem::F16);
    ok(
        !pass(&r) && r["max_ulp"] == 1,
        "signed zero differs by one ulp key",
    );
    ok(
        median(&[1.0, 3.0, 2.0, 4.0]) == 2.5 && median(&[3.0, 1.0, 2.0]) == 2.0,
        "median",
    );
    // Census sanity: peak device equals the arena for the largest cell.
    let total = arena_total(&cells);
    let u = used(8192, 8192);
    ok(
        total == used_list(&u).iter().map(|b| b + 2 * GUARD).sum::<usize>(),
        "arena total matches the 8192x8192 used sizes",
    );
    n
}

// ------------------------------------------------------------------ main

struct Opts {
    plan: bool,
    check: bool,
    time: bool,
    /// Only the 8 A0 diagnostic cells, round 0 (baseline byte gate, then the
    /// diagnostics); no 34-cell suite, no second round, no timing.
    diag_only: bool,
    production: bool,
    pm_bundle: Option<String>,
    lab_hsaco: String,
}
fn parse() -> Result<Opts> {
    let mut o = Opts {
        plan: false,
        check: false,
        time: false,
        diag_only: false,
        production: false,
        pm_bundle: None,
        lab_hsaco: LAB_HSACO_DEFAULT.into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => o.plan = true,
            "--self-check" => o.check = true,
            "--time" => o.time = true,
            "--diagnostics-only" => o.diag_only = true,
            "--launcher" => {
                o.production = match it.next().ok_or("--launcher needs raw or production")?.as_str() {
                    "raw" => false,
                    "production" => true,
                    value => return Err(format!("unknown launcher {value}; use raw or production").into()),
                };
            }
            "--pm-bundle" => o.pm_bundle = Some(it.next().ok_or("--pm-bundle needs a PATH")?),
            "--lab-hsaco" => o.lab_hsaco = it.next().ok_or("--lab-hsaco needs a PATH")?,
            _ => {
                return Err(format!(
                    "unknown argument {a}; usage: fa_pm_oracle [--plan|--self-check] | [--launcher raw|production] [--pm-bundle PATH] [--time | --diagnostics-only] [--lab-hsaco PATH]"
                )
                .into())
            }
        }
    }
    if o.time && o.diag_only {
        return Err("--time and --diagnostics-only are exclusive".into());
    }
    Ok(o)
}
fn is_code_object(b: &[u8]) -> bool {
    b.starts_with(b"\x7fELF") || b.starts_with(b"__CLANG_OFFLOAD_BUNDLE__")
}
fn run() -> Result<bool> {
    let n = self_check();
    let o = parse()?;
    if o.plan {
        plan(o.production)?;
        return Ok(true);
    }
    if o.check {
        println!(
            "{}",
            json!({"self_check":"passed","checks":n,"gpu_launched":false})
        );
        return Ok(true);
    }
    // Production needs no working-tree-relative artifact at runtime.
    let (pm_path, pm_image) = if o.production {
        if let Some(path) = &o.pm_bundle {
            let supplied = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
            if supplied != fa_prefill_pm::IMAGE {
                return Err(
                    "production --pm-bundle differs from the compiled-in certified image".into(),
                );
            }
        }
        (
            "embedded:fa_prefill_pm_gfx1201".to_owned(),
            fa_prefill_pm::IMAGE.to_vec(),
        )
    } else {
        let path = o.pm_bundle.clone().ok_or(
            "--pm-bundle PATH is required for --launcher raw; no HIP fallback is compiled",
        )?;
        let image = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
        (path, image)
    };
    if !is_code_object(&pm_image) {
        return Err(
            format!("{pm_path}: neither an ELF code object nor a clang offload bundle").into(),
        );
    }
    let pm_sha = sha_hex(&pm_image);
    println!(
        "{}",
        json!({"kind":"pm_bundle","path":pm_path,"bytes":pm_image.len(),"sha256":pm_sha,
            "module_key":PM_MODULE_KEY,"symbols":[PM_PRE,PM_DIRECT],"launcher":if o.production {"production"} else {"raw"}})
    );
    let lab_image =
        std::fs::read(&o.lab_hsaco).map_err(|e| format!("read {}: {e}", o.lab_hsaco))?;
    let lab_sha = sha_hex(&lab_image);
    if lab_sha != LAB_HSACO_SHA256 {
        return Err(format!(
            "{}: sha256 {lab_sha} is not the pinned lab object {LAB_HSACO_SHA256}; refusing (the lab arm is never recompiled)",
            o.lab_hsaco
        )
        .into());
    }
    println!(
        "{}",
        json!({"kind":"lab_object","path":o.lab_hsaco,"bytes":lab_image.len(),"sha256":lab_sha,
            "lab_source_sha256":LAB_SRC_SHA256,"compiled_by_this_tool":false})
    );
    if o.time {
        // Reject alternate incumbent tuning rather than silently mislabel a timing.
        for (name, allowed) in [
            ("HIPFIRE_FLASH_PREFILL_SPLITQ", "0"),
            ("HIPFIRE_FLASH_PREFILL_FIXED_HD", "1"),
            ("HIPFIRE_FLASH_PREFILL_PREFETCH_V", "1"),
        ] {
            if let Ok(v) = std::env::var(name) {
                if v != allowed {
                    return Err(
                        format!("unset {name}; timing requires default incumbent tuning").into(),
                    );
                }
            }
        }
    }
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    let lab = load_arm(&gpu, lab_image, LAB_PRE, LAB_DIRECT, "lab")?;
    let pm = if o.production {
        Arm::Production
    } else {
        load_arm(&gpu, pm_image, PM_PRE, PM_DIRECT, "pm")?
    };
    let cells = if o.time {
        time_cells()
    } else if o.diag_only {
        oracle_cells()
            .into_iter()
            .filter(|c| c.diagnostics())
            .collect()
    } else {
        oracle_cells()
    };
    let arena = Arena::new(&mut gpu, &cells)?;
    let (mut identical, mut failures) = (0usize, 0usize);
    if o.time {
        for c in &cells {
            let mut lines = Vec::new();
            let ok = time_cell(&mut gpu, &arena, &lab, &pm, c, &mut lines)?;
            for l in lines {
                println!("{l}");
            }
            identical += ok as usize;
            failures += !ok as usize;
        }
    } else {
        for c in &cells {
            let mut rounds = Vec::new();
            let mut ok = true;
            for round in 0..if o.diag_only { 1 } else { 2 } {
                let r = oracle_round(&mut gpu, &arena, &lab, &pm, c, round)?;
                ok &= r["pass"] == true;
                rounds.push(r);
            }
            println!(
                "{}",
                json!({"kind":"oracle_cell","cell":c.name(),"rows":c.rows,"start_pos":c.start,
                    "segment_rows":c.segment,"segments":c.segments().len(),
                    "fixture":c.fixture.name(),"module_key":PM_MODULE_KEY,
                    "rounds":rounds,"byte_identical":ok})
            );
            identical += ok as usize;
            failures += !ok as usize;
        }
    }
    println!(
        "{}",
        json!({"summary":true,"mode":if o.time {"time"} else if o.diag_only {"diagnostics"} else {"oracle"},"pm_bundle_sha256":pm_sha,
            "launcher":if o.production {"production"} else {"raw"},
            "lab_object_sha256":lab_sha,"cells":cells.len(),"byte_identical":identical,"failures":failures})
    );
    Ok(failures == 0)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("fa_pm_oracle: {e}");
            std::process::exit(1);
        }
    }
}
