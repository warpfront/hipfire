// SPDX-License-Identifier: Apache-2.0
//! LAB-only byte oracle and timing harness for the A3B exact GDN chunk-scan preparation
//! (`gdn_chunk_prep_pm_gfx1201`, fixed HK16/HV32/HD128, rows 1..=8192). No production
//! dispatch.
//!
//! ```text
//! CPU only (no HIP init):
//!   gdn_chunk_prep_pm_oracle --plan          cells, ABI, launch configs, admission, census
//!   gdn_chunk_prep_pm_oracle --self-check    CPU invariants (ABI, mapping, gate scan, ring, tree)
//! GPU (single process, bounded by --budget-s, default 120):
//!   gdn_chunk_prep_pm_oracle                 byte oracle (all cells) + timing (>=20 iterations)
//!   gdn_chunk_prep_pm_oracle --no-time       byte oracle only
//!   gdn_chunk_prep_pm_oracle --iters N       timing iterations per path (default 25, minimum 20)
//! ```
//!
//! Both arms launch the same 100-byte kernarg ABI (x, conv_w, conv_state, a, b, dt_bias,
//! a_log, q, k, v, T, q_scale, eps, hk, hv) on identical, separately uploaded clones of
//! every input, so an in-place write by either arm can never contaminate the other:
//!
//! * incumbent: `Gpu::gdn_chunk_prep` (the production runtime-compiled hipcc kernel
//!   `kernels/src/gdn_chunk_scan_prep.gfx1201.hip`, `n_key_heads=16, n_value_heads=32`).
//! * new: `Gpu::gdn_chunk_prep_pm` (the PM kernel `gdn_chunk_prep_pm_gfx1201`).
//!
//! Compared byte for byte, per cell: the FP16 `q` `[rows,16,128]`, `k` `[rows,16,128]` and
//! `v` `[rows,32,128]` planes, the in-place f32 `A` (`a`, C64 gate prefix) and `B` (`b`,
//! sigmoid beta) `[rows,32]` planes, and the f32 conv state ring `[8192,3]`. Cells are
//! rows 512 / 4096 / 8192 x zero / random input x zero / nonzero initial conv state, plus
//! bounded extras (ring-update branches at rows 1..=4, ragged C64 tails, wide-range gates
//! and activations). Output: one JSON line per cell/timing, a summary, a
//! `first_mismatch` line and a final `all_byte_identical true|false` line. The first
//! mismatch of every failing plane carries row/head/dim/index and both bit patterns. Exit
//! status 1 on any failure.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use hip_bridge::KernargBlob;
use rdna_compute::{gdn_chunk_prep_pm, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, ffi::c_void, time::Instant};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const HK: usize = 16;
const HV: usize = 32;
const HD: usize = 128;
/// q + k + v 128-channel heads; one wave owns one unit.
const UNITS: usize = 2 * HK + HV;
/// f32 elements of one `x` row.
const XDIM: usize = UNITS * HD;
const TAPS: usize = 4;
const RING: usize = 3;
/// Rows of one C64 chunk (grid x) and units handled per block (grid y).
const CHUNK: usize = 64;
const UNITS_PER_BLOCK: usize = 8;
const BLOCK: u32 = 256;
const MAX_ROWS: usize = 8192;
const EPS: f32 = 1e-6;
/// Canary bytes on each side of every device buffer.
const GUARD: usize = 4096;
const POISON: i32 = 0xFF;
const WARMUPS: usize = 3;
const DEFAULT_ITERS: usize = 25;
const MIN_ITERS: usize = 20;
const DEFAULT_BUDGET_S: f64 = 120.0;
/// Kernarg ABI of `gdn_chunk_prep` (bytes): 10 pointers, T, q_scale, eps, hk, hv.
const KERNARG_BYTES: usize = 100;

/// `q_scale` of the Qwen GDN head: `1 / sqrt(HD)`.
fn q_scale() -> f32 {
    1.0 / (HD as f32).sqrt()
}

fn sha_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn bytes<T>(x: &[T]) -> &[u8] {
    // SAFETY: only initialized f32 slices are passed; no padding.
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

// ------------------------------------------------------------------ rng

const SQRT3: f32 = 1.732_050_8;

/// Deterministic uniform f32 in `[lo, hi)` (same generator as the GDN lab oracles).
fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 40) as u32) as f32 / (1u32 << 24) as f32;
            lo + (hi - lo) * u
        })
        .collect()
}

// ---------------------------------------------------------------- cells

/// Content of `x` and the raw `a` / `b` gate inputs (weights are always random).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Data {
    Zeros,
    Random,
    /// Large-range activations and gates: conv outputs reach `|y| ~ 100` (SiLU `expf`
    /// overflow), `a + dt_bias` crosses both softplus clamps, `-b` overflows `expf`.
    Wide,
}
impl Data {
    fn name(self) -> &'static str {
        match self {
            Data::Zeros => "zeros",
            Data::Random => "random",
            Data::Wide => "wide",
        }
    }
    /// Uniform half-width multiplier of `x` (unit variance at 1).
    fn x_scale(self) -> f32 {
        match self {
            Data::Zeros => 0.0,
            Data::Random => 1.0,
            Data::Wide => 30.0,
        }
    }
    fn a_scale(self) -> f32 {
        match self {
            Data::Zeros => 0.0,
            Data::Random => 1.0,
            Data::Wide => 15.0,
        }
    }
    fn b_scale(self) -> f32 {
        match self {
            Data::Zeros => 0.0,
            Data::Random => 2.0 / SQRT3,
            Data::Wide => 60.0,
        }
    }
}

/// Initial conv state ring.
#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    Zero,
    Nonzero,
}
impl State {
    fn name(self) -> &'static str {
        match self {
            State::Zero => "zero",
            State::Nonzero => "nonzero",
        }
    }
}

#[derive(Clone, Copy)]
struct Cell {
    rows: usize,
    data: Data,
    state: State,
    timed: bool,
}
impl Cell {
    fn name(&self) -> String {
        format!(
            "rows{}_{}_state-{}",
            self.rows,
            self.data.name(),
            self.state.name()
        )
    }
    /// Wide cells may legitimately saturate; every other cell must stay finite.
    fn require_finite(&self) -> bool {
        self.data != Data::Wide
    }
}
const fn cell(rows: usize, data: Data, state: State, timed: bool) -> Cell {
    Cell {
        rows,
        data,
        state,
        timed,
    }
}
/// Required A3B shapes (rows 512/4096/8192 x zeros/random x zero/nonzero state, all
/// timed), then bounded untimed extras: rows 1..=4 (the T < 3 ring-update branches),
/// ragged C64 tails (63/65/129/8191) and the wide-range regimes.
fn cells() -> Vec<Cell> {
    let mut v = Vec::new();
    for rows in [512, 4096, 8192] {
        for data in [Data::Zeros, Data::Random] {
            for state in [State::Zero, State::Nonzero] {
                v.push(cell(rows, data, state, true));
            }
        }
    }
    for rows in [1, 2, 3, 4, 63, 65, 129, 8191] {
        v.push(cell(rows, Data::Random, State::Nonzero, false));
    }
    v.push(cell(1, Data::Random, State::Zero, false));
    v.push(cell(2, Data::Random, State::Zero, false));
    for rows in [1, 65, 4096] {
        v.push(cell(rows, Data::Wide, State::Nonzero, false));
    }
    v
}

// ----------------------------------------------------------- ABI / launch

/// Launch grid of the incumbent (and the PM kernel): C64 chunks x 8 unit blocks.
fn grid(rows: usize) -> [u32; 3] {
    [
        rows.div_ceil(CHUNK) as u32,
        UNITS.div_ceil(UNITS_PER_BLOCK) as u32,
        1,
    ]
}
/// The 100-byte kernarg blob both launchers build (reference layout for `--plan` and
/// `--self-check`; the GPU arms go through the `Gpu` launch methods).
fn kernargs(ptrs: [usize; 10], rows: usize, q_scale: f32, eps: f32) -> KernargBlob {
    let mut b = KernargBlob::with_capacity(KERNARG_BYTES);
    for p in ptrs {
        b.push_ptr(p as *const c_void);
    }
    b.push_i32(rows as i32);
    b.push_f32(q_scale);
    b.push_f32(eps);
    b.push_u32(HK as u32);
    b.push_u32(HV as u32);
    b
}

// ------------------------------------------------------------ comparison

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
    fn name(self) -> &'static str {
        match self {
            Elem::F32 => "f32",
            Elem::F16 => "f16",
        }
    }
    fn bits(self, b: &[u8], i: usize) -> u32 {
        match self {
            Elem::F32 => u32::from_ne_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()),
            Elem::F16 => u16::from_ne_bytes(b[i * 2..i * 2 + 2].try_into().unwrap()) as u32,
        }
    }
    fn finite(self, bits: u32) -> bool {
        match self {
            Elem::F32 => bits & 0x7F80_0000 != 0x7F80_0000,
            Elem::F16 => bits & 0x7C00 != 0x7C00,
        }
    }
    /// Any element whose magnitude bits are nonzero (+-0 are zero).
    fn any_nonzero(self, b: &[u8]) -> bool {
        (0..b.len() / self.size()).any(|i| {
            let bits = self.bits(b, i);
            match self {
                Elem::F32 => bits & 0x7FFF_FFFF != 0,
                Elem::F16 => bits & 0x7FFF != 0,
            }
        })
    }
}
/// Row-major coordinates of flat element `i` over `dims` (outermost first).
fn coords(mut i: usize, dims: &[(&str, usize)]) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    m.insert("index".into(), json!(i));
    for (name, size) in dims.iter().rev() {
        m.insert((*name).into(), json!(i % size));
        i /= size;
    }
    m
}
/// Compare incumbent vs new as `elem` arrays shaped `dims`. `require_finite` adds the
/// all-finite requirement to `pass`; `byte_identical` never depends on it.
fn compare(
    inc: &[u8],
    new: &[u8],
    elem: Elem,
    dims: &[(&str, usize)],
    require_finite: bool,
) -> Value {
    if inc.len() != new.len() || inc.len() % elem.size() != 0 {
        return json!({"pass":false,"byte_identical":false,"error":"length mismatch",
            "len_incumbent":inc.len(),"len_new":new.len()});
    }
    let n = inc.len() / elem.size();
    let identical = inc == new;
    let (mut nf_i, mut nf_n) = (0usize, 0usize);
    let (mut count, mut first) = (0usize, None::<usize>);
    for i in 0..n {
        let (a, b) = (elem.bits(inc, i), elem.bits(new, i));
        nf_i += !elem.finite(a) as usize;
        nf_n += !elem.finite(b) as usize;
        if a != b {
            count += 1;
            first.get_or_insert(i);
        }
    }
    let first_mismatch = first.map(|i| {
        let mut m = coords(i, dims);
        m.insert(
            "incumbent_bits".into(),
            json!(format!("{:#x}", elem.bits(inc, i))),
        );
        m.insert(
            "new_bits".into(),
            json!(format!("{:#x}", elem.bits(new, i))),
        );
        Value::Object(m)
    });
    let finite = nf_i == 0 && nf_n == 0;
    json!({"elements":n,"element_type":elem.name(),"byte_identical":identical,
        "mismatched_elements":count,"first_mismatch":first_mismatch,
        "nonfinite_incumbent":nf_i,"nonfinite_new":nf_n,"all_finite":finite,
        "finite_required":require_finite,"pass":identical && (finite || !require_finite)})
}
fn pass(v: &Value) -> bool {
    v["pass"] == true
}

// ------------------------------------------------------- memory census

/// Per-arm buffer slots (the arm index selects incumbent or PM).
const X: usize = 0;
const W: usize = 1;
const DT: usize = 2;
const AL: usize = 3;
const CONV: usize = 4;
const GA: usize = 5;
const GB: usize = 6;
const Q: usize = 7;
const K: usize = 8;
const V: usize = 9;
const PER_SIDE: usize = 10;
/// Pristine initial `a`, `b`, conv ring (device copies the timing loop restores from).
const P_A0: usize = 2 * PER_SIDE;
const P_B0: usize = P_A0 + 1;
const P_CONV0: usize = P_A0 + 2;
const N_BUF: usize = 2 * PER_SIDE + 3;
const SLOT_NAMES: [&str; PER_SIDE] = [
    "x",
    "conv_w",
    "dt_bias",
    "a_log",
    "conv_state",
    "a",
    "b",
    "q",
    "k",
    "v",
];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Arm {
    Incumbent,
    Pm,
}
const ARMS: [Arm; 2] = [Arm::Incumbent, Arm::Pm];
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Incumbent => "incumbent",
            Arm::Pm => "pm",
        }
    }
    fn base(self) -> usize {
        match self {
            Arm::Incumbent => 0,
            Arm::Pm => PER_SIDE,
        }
    }
}

fn buffer_name(i: usize) -> String {
    match i {
        P_A0 => "pristine_a".into(),
        P_B0 => "pristine_b".into(),
        P_CONV0 => "pristine_conv_state".into(),
        _ => format!("{}_{}", ARMS[i / PER_SIDE].name(), SLOT_NAMES[i % PER_SIDE]),
    }
}
fn buffer_dtype(i: usize) -> DType {
    if i >= 2 * PER_SIDE || i % PER_SIDE < Q {
        DType::F32
    } else {
        DType::F16
    }
}
/// Used payload bytes of every device buffer of one cell (excluding guards).
fn used(rows: usize) -> [usize; N_BUF] {
    let side = [
        rows * XDIM * 4,
        XDIM * TAPS * 4,
        HV * 4,
        HV * 4,
        XDIM * RING * 4,
        rows * HV * 4,
        rows * HV * 4,
        rows * HK * HD * 2,
        rows * HK * HD * 2,
        rows * HV * HD * 2,
    ];
    let mut u = [0usize; N_BUF];
    for i in 0..2 * PER_SIDE {
        u[i] = side[i % PER_SIDE];
    }
    u[P_A0] = side[GA];
    u[P_B0] = side[GB];
    u[P_CONV0] = side[CONV];
    u
}
/// Arena = one allocation per buffer at the maximum size over `cs`, plus two guards.
fn arena_sizes(cs: &[Cell]) -> [usize; N_BUF] {
    let mut m = [0usize; N_BUF];
    for c in cs {
        for (slot, n) in m.iter_mut().zip(used(c.rows)) {
            *slot = (*slot).max(n);
        }
    }
    m
}
fn arena_total(cs: &[Cell]) -> usize {
    arena_sizes(cs).iter().map(|b| b + 2 * GUARD).sum()
}
/// Peak host bytes of one cell: the f32 `x` snapshot, one `x` download for the input
/// integrity check, and the output downloads of both arms.
fn host_peak(c: &Cell) -> usize {
    let u = used(c.rows);
    let outputs: usize = [GA, GB, CONV, Q, K, V].iter().map(|&s| u[s]).sum();
    2 * u[X] + 2 * outputs
}

// ----------------------------------------------------------- GPU buffers

struct Buf {
    t: GpuTensor,
    es: usize,
    name: String,
}
impl Buf {
    fn new(gpu: &mut Gpu, name: String, nbytes: usize, dtype: DType) -> Result<Self> {
        let es = dtype.size();
        assert_eq!(nbytes % es, 0);
        let t = gpu.alloc_tensor(&[(nbytes + 2 * GUARD) / es], dtype)?;
        Ok(Self { t, es, name })
    }
    fn view(&self, len: usize) -> GpuTensor {
        self.t.sub_offset(GUARD / self.es, len / self.es)
    }
    fn upload(&self, gpu: &Gpu, data: &[u8]) -> Result<()> {
        gpu.hip.memcpy_htod_offset(&self.t.buf, GUARD, data)?;
        Ok(())
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
    fn new(gpu: &mut Gpu, cs: &[Cell]) -> Result<Self> {
        let sizes = arena_sizes(cs);
        let mut b = Vec::with_capacity(N_BUF);
        for i in 0..N_BUF {
            b.push(Buf::new(gpu, buffer_name(i), sizes[i], buffer_dtype(i))?);
        }
        Ok(Self { b })
    }
    fn slot(&self, arm: Arm, slot: usize) -> &Buf {
        &self.b[arm.base() + slot]
    }
}

// ----------------------------------------------------------------- arms

/// Launch one arm over its own cloned buffers (no synchronization).
fn launch(gpu: &mut Gpu, a: &Arena, c: &Cell, arm: Arm) -> Result<()> {
    let u = used(c.rows);
    let t = |slot: usize| a.slot(arm, slot).view(u[arm.base() + slot]);
    let (x, w, dt, al, conv) = (t(X), t(W), t(DT), t(AL), t(CONV));
    let (ga, gb, q, k, v) = (t(GA), t(GB), t(Q), t(K), t(V));
    match arm {
        Arm::Incumbent => gpu.gdn_chunk_prep(
            &x,
            &w,
            &conv,
            &ga,
            &gb,
            &dt,
            &al,
            &q,
            &k,
            &v,
            c.rows,
            q_scale(),
            EPS,
            HK,
            HV,
        )?,
        Arm::Pm => gpu.gdn_chunk_prep_pm(
            &x,
            &w,
            &conv,
            &ga,
            &gb,
            &dt,
            &al,
            &q,
            &k,
            &v,
            c.rows,
            q_scale(),
            EPS,
            HK,
            HV,
        )?,
    }
    Ok(())
}
/// Restore the arm's in-place state (conv ring, `a`, `b`) from the pristine device copies.
fn restore(gpu: &Gpu, a: &Arena, c: &Cell, arm: Arm) -> Result<()> {
    let u = used(c.rows);
    for (dst, src) in [(CONV, P_CONV0), (GA, P_A0), (GB, P_B0)] {
        let d = a.slot(arm, dst);
        gpu.hip
            .memcpy_dtod_at(&d.t.buf, GUARD, &a.b[src].t.buf, GUARD, u[arm.base() + dst])?;
    }
    Ok(())
}

// ------------------------------------------------------------ cell data

struct Prepared {
    x: Vec<f32>,
    w: Vec<f32>,
    dt: Vec<f32>,
    al: Vec<f32>,
    conv0: Vec<f32>,
    a0: Vec<f32>,
    b0: Vec<f32>,
    seed: u64,
}
/// Generate inputs, arm canaries and the pristine device copies. Nothing but the host
/// snapshots and the pristine buffers is written here; each arm clones its own inputs.
fn prepare(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<Prepared> {
    gdn_chunk_prep_pm::admit(c.rows, HK, HV)?;
    let seed = fnv(&c.name()) ^ 0x6a3b_0d2c_91e7_4f15;
    let rows = c.rows;
    let sx = c.data.x_scale() * SQRT3;
    let (sa, sb) = (c.data.a_scale() * SQRT3, c.data.b_scale() * SQRT3);
    let x = if c.data == Data::Zeros {
        vec![0.0; rows * XDIM]
    } else {
        lcg(seed ^ 1, rows * XDIM, -sx, sx)
    };
    let a0 = if c.data == Data::Zeros {
        vec![0.0; rows * HV]
    } else {
        lcg(seed ^ 4, rows * HV, -sa, sa)
    };
    let b0 = if c.data == Data::Zeros {
        vec![0.0; rows * HV]
    } else {
        lcg(seed ^ 3, rows * HV, -sb, sb)
    };
    let w = lcg(seed ^ 5, TAPS * XDIM, -0.5, 0.5);
    let dt = lcg(seed ^ 6, HV, -4.0, -2.0);
    let al = lcg(seed ^ 7, HV, 0.0, 2.0);
    let conv0 = match c.state {
        State::Zero => vec![0.0; RING * XDIM],
        State::Nonzero => {
            let s = if c.data == Data::Wide { 15.0 } else { 0.5 };
            lcg(seed ^ 9, RING * XDIM, -s, s)
        }
    };
    let u = used(rows);
    for (buf, n) in a.b.iter().zip(u) {
        buf.arm_guards(gpu, n)?;
    }
    a.b[P_A0].upload(gpu, bytes(&a0))?;
    a.b[P_B0].upload(gpu, bytes(&b0))?;
    a.b[P_CONV0].upload(gpu, bytes(&conv0))?;
    Ok(Prepared {
        x,
        w,
        dt,
        al,
        conv0,
        a0,
        b0,
        seed,
    })
}

/// One arm's downloaded results plus its input-integrity and guard verdicts.
struct Outs {
    q: Vec<u8>,
    k: Vec<u8>,
    v: Vec<u8>,
    a: Vec<u8>,
    b: Vec<u8>,
    conv: Vec<u8>,
    integrity: Value,
    integrity_ok: bool,
}
/// Clone every input onto the arm's buffers, poison q/k/v, run, download, and verify the
/// read-only inputs and all guards afterwards.
fn run_arm(gpu: &mut Gpu, a: &Arena, c: &Cell, p: &Prepared, arm: Arm) -> Result<Outs> {
    let u = used(c.rows);
    let s = |slot: usize| a.slot(arm, slot);
    let n = |slot: usize| u[arm.base() + slot];
    s(X).upload(gpu, bytes(&p.x))?;
    s(W).upload(gpu, bytes(&p.w))?;
    s(DT).upload(gpu, bytes(&p.dt))?;
    s(AL).upload(gpu, bytes(&p.al))?;
    s(CONV).upload(gpu, bytes(&p.conv0))?;
    s(GA).upload(gpu, bytes(&p.a0))?;
    s(GB).upload(gpu, bytes(&p.b0))?;
    for slot in [Q, K, V] {
        s(slot).poison(gpu, n(slot))?;
    }
    launch(gpu, a, c, arm)?;
    gpu.hip.device_synchronize()?;
    let o = Outs {
        q: s(Q).download(gpu, n(Q))?,
        k: s(K).download(gpu, n(K))?,
        v: s(V).download(gpu, n(V))?,
        a: s(GA).download(gpu, n(GA))?,
        b: s(GB).download(gpu, n(GB))?,
        conv: s(CONV).download(gpu, n(CONV))?,
        integrity: Value::Null,
        integrity_ok: false,
    };
    let inputs = json!({
        "x": s(X).download(gpu, n(X))? == bytes(&p.x),
        "conv_w": s(W).download(gpu, n(W))? == bytes(&p.w),
        "dt_bias": s(DT).download(gpu, n(DT))? == bytes(&p.dt),
        "a_log": s(AL).download(gpu, n(AL))? == bytes(&p.al),
    });
    let mut guards = serde_json::Map::new();
    for slot in 0..PER_SIDE {
        guards.insert(
            SLOT_NAMES[slot].into(),
            json!(s(slot).guards_ok(gpu, n(slot))?),
        );
    }
    let ok = inputs.as_object().unwrap().values().all(|v| v == true)
        && guards.values().all(|v| v == true);
    let integrity = json!({"inputs_unchanged":inputs,"guards_intact":guards,
        "guard_bytes_each_side":GUARD,"pass":ok});
    Ok(Outs {
        integrity,
        integrity_ok: ok,
        ..o
    })
}

/// Elements of a q/k/v plane still holding the 0xFF poison (f16 0xFFFF).
fn poison_left(plane: &[u8]) -> usize {
    plane
        .chunks_exact(2)
        .filter(|h| u16::from_ne_bytes([h[0], h[1]]) == 0xFFFF)
        .count()
}

/// Which planes must be nonzero, per cell: q/k/v unless zero input meets a zero ring;
/// the ring only when the input is nonzero or fewer than 3 rows leave old ring entries.
fn expected_nonzero(c: &Cell) -> (bool, bool) {
    let qkv = !(c.data == Data::Zeros && c.state == State::Zero);
    let ring = c.data != Data::Zeros || (c.state == State::Nonzero && c.rows < RING);
    (qkv, ring)
}

/// Byte oracle for one cell.
struct CellResult {
    line: Value,
    identical: bool,
    pass: bool,
    first: Option<Value>,
}
fn oracle_cell(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<CellResult> {
    let p = prepare(gpu, a, c)?;
    let inc = run_arm(gpu, a, c, &p, Arm::Incumbent)?;
    let new = run_arm(gpu, a, c, &p, Arm::Pm)?;
    let u = used(c.rows);
    let pristine_ok = a.b[P_A0].guards_ok(gpu, u[P_A0])?
        && a.b[P_B0].guards_ok(gpu, u[P_B0])?
        && a.b[P_CONV0].guards_ok(gpu, u[P_CONV0])?;
    let (rows, fin) = (c.rows, c.require_finite());
    let planes: Vec<(&str, Value)> = vec![
        (
            "q",
            compare(
                &inc.q,
                &new.q,
                Elem::F16,
                &[("row", rows), ("head", HK), ("dim", HD)],
                fin,
            ),
        ),
        (
            "k",
            compare(
                &inc.k,
                &new.k,
                Elem::F16,
                &[("row", rows), ("head", HK), ("dim", HD)],
                fin,
            ),
        ),
        (
            "v",
            compare(
                &inc.v,
                &new.v,
                Elem::F16,
                &[("row", rows), ("head", HV), ("dim", HD)],
                fin,
            ),
        ),
        (
            "a",
            compare(
                &inc.a,
                &new.a,
                Elem::F32,
                &[("row", rows), ("head", HV)],
                fin,
            ),
        ),
        (
            "b",
            compare(
                &inc.b,
                &new.b,
                Elem::F32,
                &[("row", rows), ("head", HV)],
                fin,
            ),
        ),
        (
            "conv_state",
            compare(
                &inc.conv,
                &new.conv,
                Elem::F32,
                &[("channel", XDIM), ("tap", RING)],
                fin,
            ),
        ),
    ];
    let byte_identical = planes.iter().all(|(_, v)| v["byte_identical"] == true);
    let planes_ok = planes.iter().all(|(_, v)| pass(v));
    // Identical poison/zero output must not pass: no arm may leave poison in q/k/v, and
    // the nonzero pattern must match what the inputs imply.
    let (exp_qkv, exp_ring) = expected_nonzero(c);
    let mut content = serde_json::Map::new();
    let mut content_ok = true;
    for (arm, o) in [("incumbent", &inc), ("pm", &new)] {
        let left = poison_left(&o.q) + poison_left(&o.k) + poison_left(&o.v);
        let qkv_nz = Elem::F16.any_nonzero(&o.q)
            && Elem::F16.any_nonzero(&o.k)
            && Elem::F16.any_nonzero(&o.v);
        let qkv_zero = !(Elem::F16.any_nonzero(&o.q)
            || Elem::F16.any_nonzero(&o.k)
            || Elem::F16.any_nonzero(&o.v));
        let ring_nz = Elem::F32.any_nonzero(&o.conv);
        let a_nz = Elem::F32.any_nonzero(&o.a);
        let b_nz = Elem::F32.any_nonzero(&o.b);
        let qkv_ok = if exp_qkv { qkv_nz } else { qkv_zero };
        let ok = left == 0 && qkv_ok && ring_nz == exp_ring && a_nz && b_nz;
        content_ok &= ok;
        content.insert(
            arm.into(),
            json!({"poison_elements_left":left,"qkv_nonzero":qkv_nz,"ring_nonzero":ring_nz,
                "a_nonzero":a_nz,"b_nonzero":b_nz,"pass":ok}),
        );
    }
    let ok = planes_ok && inc.integrity_ok && new.integrity_ok && pristine_ok && content_ok;
    let first = planes
        .iter()
        .find(|(_, v)| !pass(v) || v["byte_identical"] != true)
        .map(|(n, v)| {
            json!({"cell":c.name(),"plane":n,"first_mismatch":v["first_mismatch"],"detail":v})
        });
    let planes_json: serde_json::Map<String, Value> = planes
        .iter()
        .map(|(n, v)| (n.to_string(), v.clone()))
        .collect();
    let line = json!({"kind":"oracle_cell","cell":c.name(),"rows":c.rows,
        "data":c.data.name(),"initial_conv_state":c.state.name(),"seed":p.seed,
        "grid":grid(c.rows),"planes":planes_json,
        "integrity":{"incumbent":inc.integrity,"pm":new.integrity,"pristine_guards":pristine_ok},
        "content_sanity":content,"content_sanity_ok":content_ok,
        "byte_identical":byte_identical,"first_failure":first,"pass":ok});
    Ok(CellResult {
        line,
        identical: byte_identical,
        pass: ok,
        first,
    })
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
/// `WARMUPS` rotated warmups per arm, then `iters` event-timed samples per arm in
/// microseconds. Every sample first restores that arm's in-place state (outside the
/// event interval, same stream), so each launch sees the cell's true initial `a`, `b`
/// and conv ring; arm order rotates with every warmup/sample.
fn time_cell(gpu: &mut Gpu, a: &Arena, c: &Cell, iters: usize) -> Result<Value> {
    for w in 0..WARMUPS {
        for k in 0..2 {
            let arm = ARMS[(w + k) % 2];
            restore(gpu, a, c, arm)?;
            launch(gpu, a, c, arm)?;
        }
    }
    gpu.hip.device_synchronize()?;
    let (ea, eb) = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let mut samples = vec![Vec::with_capacity(iters); 2];
    for s in 0..iters {
        for k in 0..2 {
            let i = (s + k) % 2;
            restore(gpu, a, c, ARMS[i])?;
            gpu.hip.event_record(&ea, None)?;
            launch(gpu, a, c, ARMS[i])?;
            gpu.hip.event_record(&eb, None)?;
            gpu.hip.event_synchronize(&eb)?;
            samples[i].push(gpu.hip.event_elapsed_ms(&ea, &eb)? as f64 * 1000.0);
        }
    }
    gpu.hip.event_destroy(ea)?;
    gpu.hip.event_destroy(eb)?;
    let (mi, mn) = (median(&samples[0]), median(&samples[1]));
    Ok(json!({"kind":"timing","cell":c.name(),"rows":c.rows,
        "scope":"one gdn_chunk_prep launch per sample (state restored outside the interval)",
        "incumbent_api":"Gpu::gdn_chunk_prep","new_api":"Gpu::gdn_chunk_prep_pm",
        "warmups_per_path":WARMUPS,"iterations_per_path":iters,
        "arm_order":"rotated: warmup w and sample s start at path (index mod 2)",
        "incumbent_us_all":samples[0],"new_us_all":samples[1],
        "incumbent_us_median":mi,"new_us_median":mn,
        "saving_us_median":mi - mn,
        "speedup_median":if mn > 0.0 { json!(mi / mn) } else { Value::Null }}))
}

// ------------------------------------------------------------------ plan

fn plan() {
    let cs = cells();
    let kernarg_offsets = json!({"x_ptr":0,"conv_w_ptr":8,"conv_state_ptr":16,"a_ptr":24,
        "b_ptr":32,"dt_bias_ptr":40,"a_log_ptr":48,"q_ptr":56,"k_ptr":64,"v_ptr":72,
        "T_i32":80,"q_scale_f32":84,"eps_f32":88,"hk_u32":92,"hv_u32":96});
    for c in &cs {
        let u = used(c.rows);
        let admission = gdn_chunk_prep_pm::admit(c.rows, HK, HV);
        println!(
            "{}",
            json!({"plan_cell":c.name(),"rows":c.rows,"data":c.data.name(),
                "initial_conv_state":c.state.name(),"timed":c.timed,
                "admission":match &admission {Ok(()) => json!("ok"), Err(e) => json!(e)},
                "incumbent_launch":{"api":"Gpu::gdn_chunk_prep","symbol":"gdn_chunk_prep",
                    "grid":grid(c.rows),"block":BLOCK,"lds_bytes":0,"kernarg_bytes":KERNARG_BYTES},
                "new_launch":{"api":"Gpu::gdn_chunk_prep_pm","symbol":gdn_chunk_prep_pm::SYMBOL,
                    "grid":gdn_chunk_prep_pm::grid(c.rows),"block":gdn_chunk_prep_pm::BLOCK,
                    "lds_bytes":0,"kernarg_bytes":gdn_chunk_prep_pm::KERNARG_BYTES},
                "compared":[
                    format!("q f16 [{},{HK},{HD}]", c.rows),
                    format!("k f16 [{},{HK},{HD}]", c.rows),
                    format!("v f16 [{},{HV},{HD}]", c.rows),
                    format!("a f32 [{},{HV}] (in place, C64 gate prefix)", c.rows),
                    format!("b f32 [{},{HV}] (in place, sigmoid beta)", c.rows),
                    format!("conv_state f32 [{XDIM},{RING}]")],
                "diagnostics":["inputs unchanged per arm","guards intact","poison overwritten",
                    "nonzero/zero content matches the inputs","finite outputs (non-wide cells)"],
                "device_used_payload_bytes":u.iter().sum::<usize>(),
                "host_peak_bytes":host_peak(c)})
        );
    }
    let timed: Vec<&Cell> = cs.iter().filter(|c| c.timed).collect();
    println!(
        "{}",
        json!({"plan_timing":true,"cells":timed.iter().map(|c| c.name()).collect::<Vec<_>>(),
            "warmups_per_path":WARMUPS,"default_iterations_per_path":DEFAULT_ITERS,
            "minimum_iterations_per_path":MIN_ITERS,"unit":"microseconds, median of event-timed iterations",
            "paths":["Gpu::gdn_chunk_prep","Gpu::gdn_chunk_prep_pm"],
            "state_restore":"outside the event interval, before every sample"})
    );
    let buffers: Vec<Value> = arena_sizes(&cs)
        .iter()
        .enumerate()
        .map(|(i, b)| {
            json!({"name":buffer_name(i),"dtype":format!("{:?}", buffer_dtype(i)),
                "payload_bytes":b,"allocated_bytes":b + 2 * GUARD})
        })
        .collect();
    println!(
        "{}",
        json!({"plan":true,"gpu_launched":false,"module_key":gdn_chunk_prep_pm::MODULE,
            "symbol":gdn_chunk_prep_pm::SYMBOL,
            "embedded_bytes":gdn_chunk_prep_pm::IMAGE.len(),
            "embedded_sha256":sha_hex(gdn_chunk_prep_pm::IMAGE),
            "contract":{"key_heads":HK,"value_heads":HV,"head_dim":HD,"units":UNITS,
                "x_row_floats":XDIM,"conv_taps":TAPS,"ring_per_channel":RING,"eps":EPS,
                "q_scale":q_scale(),"kernarg_bytes":KERNARG_BYTES,"kernarg_offsets":kernarg_offsets,
                "block":BLOCK,"grid":"[ceil(rows/64), 8, 1]","exact_rows":"1..=8192",
                "planes":"q/k f16 [rows,16,128], v f16 [rows,32,128]; a/b f32 [rows,32] in place; conv ring f32 [8192,3]"},
            "oracle_cells":cs.len(),"timed_cells":timed.len(),
            "required_shapes":[512,4096,8192],"required_data":["zeros","random"],
            "required_initial_state":["zero","nonzero"],
            "extras":["rows 1,2,3,4 (ring-update branches)","rows 63,65,129,8191 (ragged C64 tail)",
                "wide-range gates/activations (rows 1, 65, 4096)"],
            "device_arena":buffers,"device_arena_total_bytes":arena_total(&cs),
            "host_peak_bytes":cs.iter().map(host_peak).max().unwrap(),
            "runtime_budget_s":DEFAULT_BUDGET_S,
            "output":"one JSON line per cell and per timing, then {summary:true,...}, a first_mismatch line and `all_byte_identical true|false`",
            "run":"cargo run --release -p rdna-compute --features lab,deltanet --example gdn_chunk_prep_pm_oracle"})
    );
}

// ------------------------------------------------------------ self-check

macro_rules! ck {
    ($n:ident, $cond:expr, $($msg:tt)+) => {{
        $n += 1;
        assert!($cond, $($msg)+);
    }};
}

/// Lane model of the per-token head norm sum: four sequential squares per lane
/// (`sum = 0; sum += o*o`, no contraction), then the XOR-shuffle butterfly 16, 8, 4, 2, 1.
/// Returns lane 0 and whether all 32 lanes agree bit for bit.
fn norm_butterfly(out: &[[f32; 4]; 32]) -> (f32, bool) {
    let mut sum = [0f32; 32];
    for l in 0..32 {
        for c in 0..4 {
            sum[l] += out[l][c] * out[l][c];
        }
    }
    let mut d = 16;
    while d > 0 {
        let prev = sum;
        for l in 0..32 {
            sum[l] += prev[l ^ d];
        }
        d >>= 1;
    }
    (sum[0], sum.iter().all(|s| s.to_bits() == sum[0].to_bits()))
}
/// Independent formulation of the same tree: halve the lane vector, adding lane `l` to
/// lane `l + half`, until one value remains.
fn norm_halving(out: &[[f32; 4]; 32]) -> f32 {
    let mut v: Vec<f32> = (0..32)
        .map(|l| {
            let mut s = 0f32;
            for c in 0..4 {
                s += out[l][c] * out[l][c];
            }
            s
        })
        .collect();
    while v.len() > 1 {
        let half = v.len() / 2;
        v = (0..half).map(|l| v[l] + v[l + half]).collect();
    }
    v[0]
}
/// Wave-local C64 gate prefix of the incumbent: lane `l` owns rows `2l`, `2l + 1`
/// (zero past `rows`), Hillis-Steele `__shfl_up` over the pair sums, then
/// `before = pair - g0 - g1`. Returns the written rows `0..rows`.
fn gate_scan_model(g: &[f32; CHUNK], rows: usize) -> Vec<f32> {
    let (mut g0, mut g1, mut pair) = ([0f32; 32], [0f32; 32], [0f32; 32]);
    for l in 0..32 {
        let (r0, r1) = (2 * l, 2 * l + 1);
        g0[l] = if r0 < rows { g[r0] } else { 0.0 };
        g1[l] = if r1 < rows { g[r1] } else { 0.0 };
        pair[l] = g0[l] + g1[l];
    }
    let mut d = 1;
    while d < 32 {
        let prev = pair;
        for l in d..32 {
            pair[l] += prev[l - d];
        }
        d <<= 1;
    }
    let mut out = vec![0f32; rows];
    for l in 0..32 {
        let before = pair[l] - g0[l] - g1[l];
        if 2 * l < rows {
            out[2 * l] = before + g0[l];
        }
        if 2 * l + 1 < rows {
            out[2 * l + 1] = before + g0[l] + g1[l];
        }
    }
    out
}
/// The incumbent's end-of-chunk-0 ring update for one channel (`old = [old0, old1, _]`,
/// `x[t]` the token values).
fn ring_update_kernel(old: [f32; 3], x: &[f32]) -> [f32; 3] {
    let t = x.len();
    [
        x[t - 1],
        if t >= 2 { x[t - 2] } else { old[0] },
        if t >= 3 {
            x[t - 3]
        } else if t == 2 {
            old[0]
        } else {
            old[1]
        },
    ]
}

fn self_check() -> usize {
    let mut n = 0usize;
    // --- ABI mirror: 100 bytes, field offsets.
    let fake: [usize; 10] = std::array::from_fn(|i| 0x1000_0000 + 0x100 * (i + 1));
    let blob = kernargs(fake, 8192, q_scale(), EPS);
    let kb = blob.as_bytes();
    ck!(
        n,
        kb.len() == KERNARG_BYTES,
        "kernarg blob is {} bytes",
        kb.len()
    );
    for i in 0..10 {
        let got = u64::from_ne_bytes(kb[8 * i..8 * i + 8].try_into().unwrap());
        ck!(
            n,
            got == fake[i] as u64,
            "kernarg pointer {i} at offset {}",
            8 * i
        );
    }
    let w32 = |o: usize| u32::from_ne_bytes(kb[o..o + 4].try_into().unwrap());
    ck!(n, w32(80) == 8192, "T at 80");
    ck!(n, w32(84) == q_scale().to_bits(), "q_scale at 84");
    ck!(n, w32(88) == EPS.to_bits(), "eps at 88");
    ck!(
        n,
        w32(92) == HK as u32 && w32(96) == HV as u32,
        "hk/hv at 92/96"
    );
    ck!(
        n,
        q_scale() == 1.0 / 128f32.sqrt(),
        "q_scale is 1/sqrt(128)"
    );

    // --- PM module constants and pure admission agree with this harness.
    ck!(
        n,
        gdn_chunk_prep_pm::KERNARG_BYTES == KERNARG_BYTES,
        "pm kernarg bytes"
    );
    ck!(n, gdn_chunk_prep_pm::BLOCK == [BLOCK, 1, 1], "pm block");
    ck!(
        n,
        gdn_chunk_prep_pm::KEY_HEADS == HK
            && gdn_chunk_prep_pm::VALUE_HEADS == HV
            && gdn_chunk_prep_pm::HEAD_DIM == HD,
        "pm geometry"
    );
    ck!(
        n,
        gdn_chunk_prep_pm::MIN_ROWS == 1 && gdn_chunk_prep_pm::MAX_ROWS == MAX_ROWS,
        "pm row range"
    );
    for rows in [1, 2, 3, 63, 64, 65, 127, 128, 129, 512, 4096, 8191, 8192] {
        ck!(
            n,
            gdn_chunk_prep_pm::grid(rows) == grid(rows),
            "pm grid at rows {rows}"
        );
        ck!(
            n,
            gdn_chunk_prep_pm::admit(rows, HK, HV).is_ok(),
            "pm admits rows {rows}"
        );
    }
    ck!(
        n,
        gdn_chunk_prep_pm::admit(0, HK, HV).is_err(),
        "pm refuses rows 0"
    );
    ck!(
        n,
        gdn_chunk_prep_pm::admit(MAX_ROWS + 1, HK, HV).is_err(),
        "pm refuses rows 8193"
    );
    ck!(
        n,
        gdn_chunk_prep_pm::admit(512, 16, 48).is_err(),
        "pm refuses HV 48"
    );
    ck!(
        n,
        gdn_chunk_prep_pm::admit(512, 8, 32).is_err(),
        "pm refuses HK 8"
    );
    ck!(
        n,
        !gdn_chunk_prep_pm::IMAGE.is_empty(),
        "pm bundle embedded"
    );

    // --- Launch/unit mapping: every x channel and every q/k/v element exactly once.
    ck!(
        n,
        grid(512) == [8, 8, 1] && grid(8192) == [128, 8, 1] && grid(1) == [1, 8, 1],
        "grid"
    );
    ck!(n, grid(65)[0] == 2 && grid(64)[0] == 1, "chunk count");
    ck!(n, UNITS == 64 && XDIM == 8192, "unit geometry");
    let mut unit_seen = [0u8; UNITS];
    for by in 0..UNITS.div_ceil(UNITS_PER_BLOCK) {
        for wave in 0..(BLOCK as usize / 32) {
            let unit = by * UNITS_PER_BLOCK + wave;
            ck!(n, unit < UNITS, "unit {unit} in range");
            unit_seen[unit] += 1;
        }
    }
    ck!(n, unit_seen.iter().all(|&s| s == 1), "units covered once");
    let (mut col_seen, mut out_seen) = (vec![0u8; XDIM], vec![0u8; XDIM]);
    for unit in 0..UNITS {
        let (region, head) = if unit < HK {
            (0, unit)
        } else if unit < 2 * HK {
            (1, unit - HK)
        } else {
            (2, unit - 2 * HK)
        };
        let base = [0, HK * HD, 2 * HK * HD][region];
        for lane in 0..32 {
            for c in 0..4 {
                let col = unit * HD + lane * 4 + c;
                let out = base + head * HD + lane * 4 + c;
                ck!(n, col == out, "x channel {col} is plane element {out}");
                col_seen[col] += 1;
                out_seen[out] += 1;
            }
        }
    }
    ck!(
        n,
        col_seen.iter().all(|&s| s == 1) && out_seen.iter().all(|&s| s == 1),
        "channel cover"
    );
    // Gate owner (blockIdx.y == 0): heads h = wave, wave + 8, ...; rows 2*lane, 2*lane+1.
    let mut head_seen = [0u8; HV];
    for wave in 0..8 {
        let mut h = wave;
        while h < HV {
            head_seen[h] += 1;
            h += 8;
        }
    }
    ck!(
        n,
        head_seen.iter().all(|&s| s == 1),
        "gate heads covered once"
    );
    let mut row_seen = [0u8; CHUNK];
    for lane in 0..32 {
        row_seen[2 * lane] += 1;
        row_seen[2 * lane + 1] += 1;
    }
    ck!(
        n,
        row_seen.iter().all(|&s| s == 1),
        "gate rows covered once"
    );

    // --- Gate prefix: with integer gates the f32 arithmetic is exact, so the lane/shuffle
    // scan must equal the serial inclusive prefix for every tail length.
    let ints = lcg(0x6a, CHUNK, -8.0, 8.0);
    let mut g = [0f32; CHUNK];
    for i in 0..CHUNK {
        g[i] = ints[i].floor();
    }
    for rows in [1usize, 2, 3, 31, 32, 33, 62, 63, 64] {
        let got = gate_scan_model(&g, rows);
        let mut s = 0f32;
        for r in 0..rows {
            s += g[r];
            ck!(
                n,
                got[r] == s,
                "gate prefix row {r} of {rows}: {} vs {s}",
                got[r]
            );
        }
    }

    // --- Conv ring: kernel branch expressions vs the history rule for T = 1..=9.
    let old = [-1.0f32, -2.0, -3.0]; // state[0] newest .. state[2] oldest
    for t in 1..=9usize {
        let x: Vec<f32> = (0..t).map(|i| i as f32).collect();
        let mut hist = vec![old[2], old[1], old[0]];
        hist.extend_from_slice(&x);
        let want: [f32; 3] = std::array::from_fn(|j| hist[hist.len() - 1 - j]);
        ck!(
            n,
            ring_update_kernel(old, &x) == want,
            "ring update at T={t}"
        );
    }
    // Window halo: per chunk the window at row r is x[t-3..t-1] of the history
    // [old2, old1, old0, x0, ...]; chunk 0 starts from the ring, later chunks from x.
    let total = 4 * CHUNK + 5;
    let x: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let mut hist = vec![old[2], old[1], old[0]];
    hist.extend_from_slice(&x);
    for chunk in 0..total.div_ceil(CHUNK) {
        let chunk0 = chunk * CHUNK;
        let mut win = if chunk0 == 0 {
            [old[2], old[1], old[0]]
        } else {
            [x[chunk0 - 3], x[chunk0 - 2], x[chunk0 - 1]]
        };
        for r in 0..CHUNK.min(total - chunk0) {
            let t = chunk0 + r;
            ck!(
                n,
                win == [hist[t], hist[t + 1], hist[t + 2]],
                "window at token {t}"
            );
            win = [win[1], win[2], x[t]];
        }
    }

    // --- Head-norm reduction tree: lane butterfly == halving tree, all lanes agree.
    for trial in 0..64u64 {
        let v = lcg(0x77 + trial, 32 * 4, -50.0, 50.0);
        let out: [[f32; 4]; 32] = std::array::from_fn(|l| std::array::from_fn(|c| v[4 * l + c]));
        let (lane0, agree) = norm_butterfly(&out);
        ck!(n, agree, "butterfly lanes disagree (trial {trial})");
        ck!(
            n,
            lane0.to_bits() == norm_halving(&out).to_bits(),
            "butterfly != halving tree (trial {trial})"
        );
    }

    // --- Oracle machinery: comparison, coordinates, statistics, cells, census.
    let dims = [("row", 3usize), ("head", 4), ("dim", 5)];
    let base: Vec<u8> = (0..60u16)
        .flat_map(|i| (0x3C00 + i).to_ne_bytes())
        .collect();
    ck!(
        n,
        pass(&compare(&base, &base, Elem::F16, &dims, true)),
        "identical planes pass"
    );
    let mut flipped = base.clone();
    flipped[37 * 2] ^= 1;
    let r = compare(&base, &flipped, Elem::F16, &dims, true);
    ck!(
        n,
        !pass(&r) && r["byte_identical"] == false,
        "flip is detected"
    );
    let fm = &r["first_mismatch"];
    ck!(
        n,
        fm["index"] == 37 && fm["row"] == 1 && fm["head"] == 3 && fm["dim"] == 2,
        "first mismatch coordinates {fm}"
    );
    ck!(
        n,
        fm["incumbent_bits"] == format!("{:#x}", 0x3C00 + 37)
            && fm["new_bits"] == format!("{:#x}", (0x3C00 + 37) ^ 1),
        "first mismatch bits {fm}"
    );
    ck!(n, r["mismatched_elements"] == 1, "one mismatched element");
    let nan16: Vec<u8> = (0..4).flat_map(|_| 0x7E00u16.to_ne_bytes()).collect();
    let d1 = [("row", 4usize)];
    ck!(
        n,
        !pass(&compare(&nan16, &nan16, Elem::F16, &d1, true)),
        "nonfinite fails when required"
    );
    let waived = compare(&nan16, &nan16, Elem::F16, &d1, false);
    ck!(
        n,
        pass(&waived) && waived["byte_identical"] == true,
        "nonfinite waived (wide cells)"
    );
    let f32b: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect();
    let mut f32c = f32b.clone();
    f32c[4 * 4 + 3] ^= 0x80;
    let r = compare(&f32b, &f32c, Elem::F32, &[("channel", 3), ("tap", 2)], true);
    ck!(
        n,
        r["first_mismatch"]["index"] == 4
            && r["first_mismatch"]["channel"] == 2
            && r["first_mismatch"]["tap"] == 0,
        "f32 coordinates"
    );
    ck!(
        n,
        !pass(&compare(&f32b, &f32c[..8], Elem::F32, &d1, true)),
        "length mismatch fails"
    );
    ck!(
        n,
        !Elem::F16.any_nonzero(&[0, 0, 0, 0x80]) && Elem::F16.any_nonzero(&[1, 0]),
        "f16 nonzero"
    );
    ck!(
        n,
        !Elem::F32.any_nonzero(&[0, 0, 0, 0x80]) && Elem::F32.any_nonzero(&[0, 0, 0x80, 0x3F]),
        "f32 nonzero"
    );
    ck!(
        n,
        poison_left(&[0xFF, 0xFF, 0xFF, 0x7F, 0xFF, 0xFF]) == 2,
        "poison counter"
    );
    ck!(
        n,
        median(&[3.0, 1.0, 2.0]) == 2.0 && median(&[4.0, 1.0, 3.0, 2.0]) == 2.5,
        "median"
    );
    let (l1, l2) = (lcg(5, 64, -1.0, 1.0), lcg(5, 64, -1.0, 1.0));
    ck!(
        n,
        l1 == l2 && l1.iter().all(|v| (-1.0..1.0).contains(v)),
        "lcg deterministic/in range"
    );
    ck!(
        n,
        lcg(5, 8, 0.0, 1.0) != lcg(6, 8, 0.0, 1.0),
        "lcg seed sensitivity"
    );
    let cs = cells();
    let mut names: Vec<String> = cs.iter().map(Cell::name).collect();
    names.sort();
    names.dedup();
    ck!(n, names.len() == cs.len(), "cell names unique");
    for rows in [512, 4096, 8192] {
        for data in [Data::Zeros, Data::Random] {
            for state in [State::Zero, State::Nonzero] {
                ck!(
                    n,
                    cs.iter().any(|c| {
                        c.rows == rows && c.data == data && c.state == state && c.timed
                    }),
                    "required timed cell rows {rows} {data:?} {state:?}"
                );
            }
        }
    }
    ck!(
        n,
        cs.iter().all(|c| (1..=MAX_ROWS).contains(&c.rows)),
        "cell rows in range"
    );
    ck!(
        n,
        cs.iter().any(|c| c.rows < RING),
        "ring-update branch cells present"
    );
    ck!(
        n,
        cs.iter().any(|c| c.rows % CHUNK != 0 && c.rows > CHUNK),
        "ragged tail cells present"
    );
    let u = used(8192);
    ck!(
        n,
        u[X] == 8192 * 8192 * 4
            && u[Q] == 8192 * 16 * 128 * 2
            && u[K] == u[Q]
            && u[V] == 8192 * 32 * 128 * 2
            && u[GA] == 8192 * 32 * 4
            && u[CONV] == 8192 * 3 * 4
            && u[W] == 8192 * 4 * 4,
        "plane byte counts"
    );
    ck!(
        n,
        (0..N_BUF).all(|i| u[i] % buffer_dtype(i).size() == 0) && arena_total(&cs) < (2usize << 30),
        "buffers aligned, arena < 2 GiB"
    );
    ck!(
        n,
        arena_sizes(&cs)[X] == u[X] && arena_sizes(&cs)[P_CONV0] == u[P_CONV0],
        "arena maxima"
    );
    for (c, want) in [
        (cell(512, Data::Zeros, State::Zero, true), (false, false)),
        (cell(512, Data::Zeros, State::Nonzero, true), (true, false)),
        (cell(2, Data::Zeros, State::Nonzero, false), (true, true)),
        (cell(512, Data::Random, State::Zero, true), (true, true)),
        (cell(1, Data::Random, State::Zero, false), (true, true)),
        (cell(4096, Data::Wide, State::Nonzero, false), (true, true)),
    ] {
        ck!(
            n,
            expected_nonzero(&c) == want,
            "content expectation for {}",
            c.name()
        );
    }
    let mut names = std::collections::BTreeSet::new();
    for i in 0..N_BUF {
        ck!(n, names.insert(buffer_name(i)), "buffer name {i} unique");
    }
    n
}

// ------------------------------------------------------------------ main

struct Opts {
    plan: bool,
    check: bool,
    time: bool,
    iters: usize,
    budget_s: f64,
}
fn parse() -> Result<Opts> {
    let mut o = Opts {
        plan: false,
        check: false,
        time: true,
        iters: DEFAULT_ITERS,
        budget_s: DEFAULT_BUDGET_S,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => o.plan = true,
            "--self-check" => o.check = true,
            "--no-time" => o.time = false,
            "--iters" => {
                o.iters = it.next().ok_or("--iters needs N")?.parse()?;
                if o.iters < MIN_ITERS {
                    return Err(format!("--iters must be >= {MIN_ITERS}").into());
                }
            }
            "--budget-s" => o.budget_s = it.next().ok_or("--budget-s needs SECONDS")?.parse()?,
            _ => {
                return Err(format!(
                    "unknown argument {a}; usage: gdn_chunk_prep_pm_oracle [--plan | --self-check | [--no-time] [--iters N] [--budget-s S]]"
                )
                .into())
            }
        }
    }
    Ok(o)
}
fn run() -> Result<bool> {
    let n = self_check();
    let o = parse()?;
    if o.plan {
        plan();
        return Ok(true);
    }
    if o.check {
        println!(
            "{}",
            json!({"self_check":"passed","checks":n,"gpu_launched":false})
        );
        return Ok(true);
    }
    let t0 = Instant::now();
    println!(
        "{}",
        json!({"kind":"prep_bundle","module_key":gdn_chunk_prep_pm::MODULE,
            "symbol":gdn_chunk_prep_pm::SYMBOL,"embedded_bytes":gdn_chunk_prep_pm::IMAGE.len(),
            "embedded_sha256":sha_hex(gdn_chunk_prep_pm::IMAGE),"budget_s":o.budget_s})
    );
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    let cs = cells();
    let arena = Arena::new(&mut gpu, &cs)?;
    let (mut identical, mut failures, mut all_identical) = (0usize, 0usize, true);
    let mut first_mismatch: Option<Value> = None;
    let mut timings = Vec::new();
    for c in &cs {
        let r = oracle_cell(&mut gpu, &arena, c)?;
        println!("{}", r.line);
        identical += r.identical as usize;
        failures += !r.pass as usize;
        all_identical &= r.identical;
        if first_mismatch.is_none() {
            first_mismatch = r.first;
        }
        if o.time && c.timed {
            let t = time_cell(&mut gpu, &arena, c, o.iters)?;
            println!("{t}");
            timings.push(
                json!({"cell":c.name(),"incumbent_us_median":t["incumbent_us_median"],
                "new_us_median":t["new_us_median"]}),
            );
        }
        let elapsed = t0.elapsed().as_secs_f64();
        if elapsed > o.budget_s {
            return Err(format!(
                "budget exceeded: {elapsed:.1}s > {}s after cell {}",
                o.budget_s,
                c.name()
            )
            .into());
        }
    }
    println!(
        "{}",
        json!({"summary":true,"cells":cs.len(),"byte_identical_cells":identical,
            "failures":failures,"all_byte_identical":all_identical,"timings":timings,
            "elapsed_s":t0.elapsed().as_secs_f64()})
    );
    match &first_mismatch {
        Some(m) => println!("first_mismatch {m}"),
        None => println!("first_mismatch none"),
    }
    println!("all_byte_identical {all_identical}");
    Ok(failures == 0 && all_identical)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("gdn_chunk_prep_pm_oracle: {e}");
            std::process::exit(1);
        }
    }
}
