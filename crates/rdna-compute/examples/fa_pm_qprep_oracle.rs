// SPDX-License-Identifier: Apache-2.0
//! LAB-only byte oracle and timing harness for the A3B F5 exact FA Q-prep fold
//! (`fa_qprep_pm_gfx1201`, H16/KV2/D256, rotary 64). No production dispatch.
//!
//! ```text
//! CPU only (no HIP init):
//!   fa_pm_qprep_oracle --plan          cells, ABI, launch configs, admission, census
//!   fa_pm_qprep_oracle --self-check    CPU invariants (ABI, mapping, reduction-tree proof)
//! GPU (single process, bounded by --budget-s, default 120):
//!   fa_pm_qprep_oracle                 byte oracle (all cells) + timing (>=20 iterations)
//!   fa_pm_qprep_oracle --no-time       byte oracle only
//!   fa_pm_qprep_oracle --iters N       timing iterations per path (default 25, minimum 20)
//!   fa_pm_qprep_oracle --kv-codec q8|fp8|all  select cache codec (default all)
//! ```
//!
//! Both arms read the same interleaved `[rows, 16 * 2 * 256]` projection, the same Q/K
//! norm weights, the same K source, the same positions / `pos_offset` / base / `n_rot`,
//! and the same runtime-written K/V caches for each codec: Q8_0 or native OCP
//! E4M3FN (516-byte rows: 512 head-major codes followed by two f16 scales).
//! Both use F32-tagged byte planes accepted by the typed PM launcher.
//!
//! * incumbent: `deinterleave_q_rmsnorm_f32_batched` -> K `rmsnorm_batched` ->
//!   original Q+K `rope_partial_interleaved_f32_batched` (null angle capture) ->
//!   `fa_prefill_pm_preconvert` -> `fa_prefill_pm_direct_codec`.
//! * new: K `rmsnorm_batched` -> K-only `rope_partial_halfsplit_capture_f32_batched`
//!   (angle table captured) -> `fa_pm_qprep_exact` (gate + final Q16 in one pass) ->
//!   the same `fa_prefill_pm_direct_codec` with the same codec and K/V cache bytes.
//!
//! Compared byte for byte, per cell: post-RoPE K (null capture vs captured K-only),
//! Q16, gate f32, final PM direct attention output; the captured angle table is also
//! checked against an f64 reference (sanity, not a byte claim: the table is the same
//! device `powf/cosf/sinf` array the incumbent rotation reads). Output: one JSON line
//! per cell/timing, a summary, and a final `all_byte_identical true|false` line. The
//! first mismatch of every failing stage carries row/head/dim/index and both bit
//! patterns. Exit status 1 on any failure.
//! The ten original Q8 cells retain their names, input seeds, writers and byte
//! gates. Each is repeated for FP8 with the same f32 inputs, using the production
//! batched native-FP8 writer for both K and V. No feature flag is enabled here.
//! All-codec mode reuses one guarded arena (Q8 is the larger cache stride);
//! FP8-only mode sizes cache payloads to native 516-byte rows. Timing remains
//! prep-only and is reported separately for each codec, not an FP8 speedup claim.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
use hip_bridge::KernargBlob;
use rdna_compute::fa_prefill_pm::FaPrefillKvCodec;
use rdna_compute::{fa_prefill_pm, fa_qprep_pm, DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, ffi::c_void, time::Instant};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const H: usize = 16;
const KV: usize = 2;
const D: usize = 256;
const NROT: usize = 64;
const HALF: usize = NROT / 2;
/// f32/f16 elements of one Q / gate / attention row (`H * D`).
const ROW: usize = H * D;
/// f32 elements of one interleaved projection row (`H * 2 * D`).
const SRC_ROW: usize = 2 * ROW;
/// f32 elements of one K row (`KV * D`).
const KROW: usize = KV * D;
const CODECS: [FaPrefillKvCodec; 2] = [FaPrefillKvCodec::Q8_0, FaPrefillKvCodec::Fp8E4M3];

fn codec_name(codec: FaPrefillKvCodec) -> &'static str {
    match codec {
        FaPrefillKvCodec::Q8_0 => "q8_0",
        FaPrefillKvCodec::Fp8E4M3 => "fp8_e4m3",
    }
}
fn writer_name(codec: FaPrefillKvCodec) -> &'static str {
    match codec {
        FaPrefillKvCodec::Q8_0 => "kv_cache_write_q8_0_batched",
        FaPrefillKvCodec::Fp8E4M3 => "kv_cache_write_fp8_e4m3_batched",
    }
}
const MAX_ROWS: usize = 8192;
const THETA: f32 = 10_000_000.0;
const EPS: f32 = 1e-6;
/// Canary bytes on each side of every device buffer.
const GUARD: usize = 4096;
const POISON: i32 = 0xFF;
const WARMUPS: usize = 3;
const DEFAULT_ITERS: usize = 25;
const MIN_ITERS: usize = 20;
const DEFAULT_BUDGET_S: f64 = 120.0;
const LAYERS: usize = 10;
/// Kernarg ABI of `attention_fa2_qprep_pm_gfx1201` (offsets, bytes).
const QPREP_KERNARG_BYTES: usize = 48;
const QPREP_BLOCK: u32 = 128;
const HEADS_PER_BLOCK: usize = 4;

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
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_ne_bytes(x.try_into().unwrap()))
        .collect()
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

// ---------------------------------------------------------------- cells

/// Content of the interleaved projection (K/V sources are always random).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Data {
    Zeros,
    Random,
    /// ~1e20 values: every square overflows to +inf (mean inf, rms 0).
    Wide,
    /// ~1e-30 values: every square underflows to zero (rms from eps alone).
    Tiny,
}
impl Data {
    fn name(self) -> &'static str {
        match self {
            Data::Zeros => "zeros",
            Data::Random => "random",
            Data::Wide => "wide",
            Data::Tiny => "tiny",
        }
    }
    fn scale(self) -> f32 {
        match self {
            Data::Zeros => 0.0,
            Data::Random => 1.0,
            Data::Wide => 1e20,
            Data::Tiny => 1e-30,
        }
    }
}

/// Positions are the packed per-row i32 the FA direct kernel consumes.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Fixture {
    /// `pos[row] = row`.
    Std,
    /// Group-local (8 rows) nonmonotone positions with duplicate pairs.
    Scrambled,
}
impl Fixture {
    fn name(self) -> &'static str {
        match self {
            Fixture::Std => "std",
            Fixture::Scrambled => "scrambled8",
        }
    }
    fn positions(self, rows: usize) -> Vec<i32> {
        (0..rows)
            .map(|row| {
                let (g, r) = (row / 8, row % 8);
                let m = (rows - g * 8).min(8);
                let off = match self {
                    Fixture::Std => r,
                    Fixture::Scrambled => ((r & !1) * 3 + g) % m,
                };
                (g * 8 + off) as i32
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
struct Cell {
    codec: FaPrefillKvCodec,
    rows: usize,
    data: Data,
    fixture: Fixture,
    /// Added to every position for the RoPE angle only (the cache `compact_offset`).
    pos_offset: i32,
    /// Also timed (qprep sequences) after its byte oracle.
    timed: bool,
}
impl Cell {
    /// Cache positions: every position lies in `0..rows`.
    fn ctx(&self) -> usize {
        self.rows
    }
    /// Codec-independent fixture identity also preserves every original Q8 seed.
    fn input_name(&self) -> String {
        format!(
            "n{}_{}_{}_off{}",
            self.rows,
            self.data.name(),
            self.fixture.name(),
            self.pos_offset
        )
    }
    fn name(&self) -> String {
        match self.codec {
            FaPrefillKvCodec::Q8_0 => self.input_name(),
            FaPrefillKvCodec::Fp8E4M3 => {
                format!("{}_{}", self.input_name(), codec_name(self.codec))
            }
        }
    }
}
const fn cell(rows: usize, data: Data, fixture: Fixture, pos_offset: i32, timed: bool) -> Cell {
    Cell {
        codec: FaPrefillKvCodec::Q8_0,
        rows,
        data,
        fixture,
        pos_offset,
        timed,
    }
}
/// Required A3B shapes (rows 512/4096/8192 x zeros/random), then bounded extras:
/// nonzero `pos_offset` with nonmonotone/duplicate positions, an odd row count,
/// and the overflow/underflow norm regimes. Repeat every original cell for FP8.
fn cells() -> Vec<Cell> {
    use Data::*;
    use Fixture::*;
    let q8 = vec![
        cell(512, Zeros, Std, 0, false),
        cell(512, Random, Std, 0, true),
        cell(4096, Zeros, Std, 0, false),
        cell(4096, Random, Std, 0, true),
        cell(8192, Zeros, Std, 0, false),
        cell(8192, Random, Std, 0, true),
        cell(512, Random, Scrambled, 37, false),
        cell(65, Random, Std, 0, false),
        cell(512, Wide, Std, 0, false),
        cell(512, Tiny, Std, 0, false),
    ];
    let mut cs = q8.clone();
    cs.extend(q8.into_iter().map(|mut c| {
        c.codec = FaPrefillKvCodec::Fp8E4M3;
        c
    }));
    cs
}

fn parse_codec(value: &str) -> Result<Option<FaPrefillKvCodec>> {
    match value {
        "q8" => Ok(Some(FaPrefillKvCodec::Q8_0)),
        "fp8" => Ok(Some(FaPrefillKvCodec::Fp8E4M3)),
        "all" => Ok(None),
        _ => Err(format!("--kv-codec expects q8|fp8|all, got {value}").into()),
    }
}
fn selected_cells(codec: Option<FaPrefillKvCodec>) -> Vec<Cell> {
    let mut cs = cells();
    cs.retain(|c| codec.is_none_or(|codec| c.codec == codec));
    cs
}
fn codec_census(cs: &[Cell]) -> Vec<Value> {
    CODECS
        .into_iter()
        .filter_map(|codec| {
            let count = cs.iter().filter(|c| c.codec == codec).count();
            let rows = cs
                .iter()
                .filter(|c| c.codec == codec)
                .map(|c| c.rows)
                .max()?;
            Some(json!({"codec":codec_name(codec),"cells":count,
            "timed_cells":cs.iter().filter(|c| c.codec == codec && c.timed).count(),
            "cache_row_bytes":codec.cache_row_bytes(),
            "max_cache_bytes_per_side":rows * codec.cache_row_bytes(),
            "standalone_device_arena_bytes":used_list(&used(rows, codec))
                .iter().map(|b| b + 2 * GUARD).sum::<usize>()}))
        })
        .collect()
}

// ----------------------------------------------------------- ABI / launch

/// Pure host admission for the exact Q-prep primitive and the PM direct kernel.
fn validate_cell(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    n_rot: usize,
    rows: usize,
    positions: &[i32],
    ctx: usize,
    pos_offset: i32,
) -> std::result::Result<(), String> {
    if (heads, kv_heads, head_dim, n_rot) != (H, KV, D, NROT) {
        return Err(format!(
            "domain is H16/KV2/D256/rot64, got {heads}/{kv_heads}/{head_dim}/{n_rot}"
        ));
    }
    if !(1..=MAX_ROWS).contains(&rows) {
        return Err(format!("rows {rows} outside 1..={MAX_ROWS}"));
    }
    if positions.len() != rows {
        return Err(format!(
            "positions length {} != rows {rows}",
            positions.len()
        ));
    }
    if pos_offset < 0 {
        return Err(format!("pos_offset {pos_offset} is negative"));
    }
    for (row, &p) in positions.iter().enumerate() {
        if p < 0 || p as usize >= ctx {
            return Err(format!("row {row}: position {p} outside 0..{ctx}"));
        }
        if (p as i64 + pos_offset as i64) > i32::MAX as i64 {
            return Err(format!("row {row}: position + pos_offset overflows i32"));
        }
    }
    fa_prefill_pm::validate_positions(positions, ctx).map_err(|e| e.to_string())?;
    // Production FA admission (64..=8192 rows) for the same shape.
    fa_prefill_pm::validate_domain(heads, kv_heads, head_dim, rows, ctx)
        .map_err(|e| e.to_string())?;
    Ok(())
}
fn qprep_grid(rows: usize) -> [u32; 3] {
    [(rows * H / HEADS_PER_BLOCK) as u32, 1, 1]
}
fn qprep_kernargs(
    inter: *mut c_void,
    q16: *mut c_void,
    gate: *mut c_void,
    qnorm: *mut c_void,
    angles: *mut c_void,
    rows: usize,
    eps: f32,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    for p in [inter, q16, gate, qnorm, angles] {
        b.push_ptr(p);
    }
    b.push_i32(rows as i32);
    b.push_f32(eps);
    b
}
/// Halfsplit RoPE launch used by both arms (existing kernel, `1 x rows` grid, 256
/// threads, `2 * n_rot/2` floats of dynamic LDS).
fn rope_grid(rows: usize) -> [u32; 3] {
    [1, rows as u32, 1]
}
const ROPE_BLOCK: u32 = 256;
const ROPE_LDS: u32 = (2 * HALF * 4) as u32;

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
}
/// Compare incumbent vs new as `elem` arrays of `[row, heads, dim]`.
fn compare(inc: &[u8], new: &[u8], elem: Elem, heads: usize, dim: usize) -> Value {
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
        json!({"index":i,"row":i / (heads * dim),"head":(i / dim) % heads,"dim":i % dim,
            "incumbent_bits":format!("{:#x}", elem.bits(inc, i)),
            "new_bits":format!("{:#x}", elem.bits(new, i))})
    });
    let finite = nf_i == 0 && nf_n == 0;
    json!({"elements":n,"element_type":elem.name(),"byte_identical":identical,
        "mismatched_elements":count,"first_mismatch":first_mismatch,
        "nonfinite_incumbent":nf_i,"nonfinite_new":nf_n,"all_finite":finite,
        "pass":identical && finite})
}
fn pass(v: &Value) -> bool {
    v["pass"] == true
}

// ----------------------------------------------------- CPU angle reference

/// f64 reference table for sanity only (the device table is `powf/cosf/sinf`).
fn angle_reference(pos: i64, i: usize) -> (f64, f64) {
    let freq = 1.0 / (THETA as f64).powf((2 * i) as f64 / NROT as f64);
    let angle = pos as f64 * freq;
    (angle.cos(), angle.sin())
}
/// Max abs error of the captured `[rows, 64]` table (cos[0..32], sin[32..64]) vs the
/// f64 reference at `pos + pos_offset`, plus unwritten (nonfinite) entries.
fn angle_sanity(table: &[f32], positions: &[i32], pos_offset: i32) -> Value {
    let (mut max_err, mut nonfinite, mut first_bad) = (0f64, 0usize, None::<(usize, usize)>);
    for (row, &p) in positions.iter().enumerate() {
        for i in 0..HALF {
            let (c, s) = angle_reference(p as i64 + pos_offset as i64, i);
            for (idx, want) in [(i, c), (HALF + i, s)] {
                let got = table[row * NROT + idx];
                if !got.is_finite() {
                    nonfinite += 1;
                    first_bad.get_or_insert((row, idx));
                    continue;
                }
                let e = (got as f64 - want).abs();
                if e > max_err {
                    max_err = e;
                }
            }
        }
    }
    json!({"rows":positions.len(),"max_abs_err_vs_f64":max_err,"tolerance":ANGLE_TOL,
        "nonfinite":nonfinite,"first_nonfinite_row_index":first_bad,
        "pass":nonfinite == 0 && max_err <= ANGLE_TOL})
}
const ANGLE_TOL: f64 = 2e-3;

// ------------------------------------------------------- memory census

/// Used payload bytes of every device buffer of one cell (excluding guards).
#[derive(Clone, Copy)]
struct Used {
    inter: usize,
    q_inc: usize,
    gate_inc: usize,
    gate_new: usize,
    q16_inc: usize,
    q16_new: usize,
    k_inc: usize,
    k_new: usize,
    angles: usize,
    pos: usize,
    allpos: usize,
    k_src: usize,
    v_src: usize,
    kc: usize,
    vc: usize,
    qn: usize,
    kn: usize,
    out_inc: usize,
    out_new: usize,
}
fn used(rows: usize, codec: FaPrefillKvCodec) -> Used {
    let ctx = rows;
    Used {
        inter: rows * SRC_ROW * 4,
        q_inc: rows * ROW * 4,
        gate_inc: rows * ROW * 4,
        gate_new: rows * ROW * 4,
        q16_inc: rows * ROW * 2,
        q16_new: rows * ROW * 2,
        k_inc: rows * KROW * 4,
        k_new: rows * KROW * 4,
        angles: rows * NROT * 4,
        pos: rows * 4,
        allpos: ctx * 4,
        k_src: ctx * KROW * 4,
        v_src: ctx * KROW * 4,
        kc: ctx * codec.cache_row_bytes(),
        vc: ctx * codec.cache_row_bytes(),
        qn: D * 4,
        kn: D * 4,
        out_inc: rows * ROW * 4,
        out_new: rows * ROW * 4,
    }
}
const BUFFER_NAMES: [&str; 19] = [
    "interleaved",
    "q_f32_incumbent",
    "gate_incumbent",
    "gate_new",
    "q16_incumbent",
    "q16_new",
    "k_incumbent",
    "k_new",
    "angles",
    "positions",
    "allpos",
    "k_src",
    "v_src",
    // Preserve the original cache-label lengths used in the guard pattern seed.
    "k_cache_wire",
    "v_cache_wire",
    "q_norm_weight",
    "k_norm_weight",
    "out_incumbent",
    "out_new",
];
const BUFFER_DTYPES: [DType; 19] = [
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::Raw,
    DType::Raw,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    // Production Q8 planes are F32-tagged byte allocations. The native-FP8
    // launcher also accepts F32 byte planes; codec-specific extents, not dtype,
    // select the wire format (544-byte Q8 rows vs 516-byte FP8 rows).
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
];
fn used_list(u: &Used) -> [usize; 19] {
    [
        u.inter, u.q_inc, u.gate_inc, u.gate_new, u.q16_inc, u.q16_new, u.k_inc, u.k_new, u.angles,
        u.pos, u.allpos, u.k_src, u.v_src, u.kc, u.vc, u.qn, u.kn, u.out_inc, u.out_new,
    ]
}
/// Arena = one allocation per buffer at the maximum size over `cs`, plus two guards.
fn arena_sizes(cs: &[Cell]) -> [usize; 19] {
    let mut m = [0usize; 19];
    for c in cs {
        for (slot, b) in m.iter_mut().zip(used_list(&used(c.rows, c.codec))) {
            *slot = (*slot).max(b);
        }
    }
    m
}
fn arena_total(cs: &[Cell]) -> usize {
    arena_sizes(cs).iter().map(|b| b + 2 * GUARD).sum()
}
/// Peak host bytes of one cell: interleaved + K/V f32 sources + caches during
/// preparation, then the largest compare pair (one download per arm).
fn host_peak(c: &Cell) -> usize {
    let u = used(c.rows, c.codec);
    let weights = u.qn + u.kn;
    let prep = u.inter + u.k_inc + u.k_src + u.v_src + u.pos + u.allpos + weights;
    let snap = u.inter + u.k_inc + u.kc + u.vc + u.pos + weights;
    prep.max(snap + 2 * u.out_inc).max(snap + 2 * u.gate_inc)
}
/// Launch counts of one cell's oracle: 4 incumbent + 3 new prep launches, 2 direct,
/// 2 KV writes.
fn oracle_launches() -> usize {
    4 + 3 + 2 + 2
}

// ----------------------------------------------------------- GPU buffers

struct Buf {
    t: GpuTensor,
    es: usize,
    name: &'static str,
}
impl Buf {
    fn new(gpu: &mut Gpu, name: &'static str, nbytes: usize, dtype: DType) -> Result<Self> {
        let es = dtype.size();
        assert_eq!(nbytes % es, 0);
        let t = gpu.alloc_tensor(&[(nbytes + 2 * GUARD) / es], dtype)?;
        Ok(Self { t, es, name })
    }
    fn sub(&self, off: usize, len: usize) -> GpuTensor {
        self.t.sub_offset((GUARD + off) / self.es, len / self.es)
    }
    fn view(&self, len: usize) -> GpuTensor {
        self.sub(0, len)
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
        let mut b = Vec::new();
        for i in 0..BUFFER_NAMES.len() {
            b.push(Buf::new(gpu, BUFFER_NAMES[i], sizes[i], BUFFER_DTYPES[i])?);
        }
        Ok(Self { b })
    }
    fn inter(&self) -> &Buf {
        &self.b[0]
    }
    fn q_inc(&self) -> &Buf {
        &self.b[1]
    }
    fn gate_inc(&self) -> &Buf {
        &self.b[2]
    }
    fn gate_new(&self) -> &Buf {
        &self.b[3]
    }
    fn q16_inc(&self) -> &Buf {
        &self.b[4]
    }
    fn q16_new(&self) -> &Buf {
        &self.b[5]
    }
    fn k_inc(&self) -> &Buf {
        &self.b[6]
    }
    fn k_new(&self) -> &Buf {
        &self.b[7]
    }
    fn angles(&self) -> &Buf {
        &self.b[8]
    }
    fn pos(&self) -> &Buf {
        &self.b[9]
    }
    fn allpos(&self) -> &Buf {
        &self.b[10]
    }
    fn k_src(&self) -> &Buf {
        &self.b[11]
    }
    fn v_src(&self) -> &Buf {
        &self.b[12]
    }
    fn kc(&self) -> &Buf {
        &self.b[13]
    }
    fn vc(&self) -> &Buf {
        &self.b[14]
    }
    fn qn(&self) -> &Buf {
        &self.b[15]
    }
    fn kn(&self) -> &Buf {
        &self.b[16]
    }
    fn out_inc(&self) -> &Buf {
        &self.b[17]
    }
    fn out_new(&self) -> &Buf {
        &self.b[18]
    }
}

// ----------------------------------------------------------------- arms

/// Incumbent Q/K prep: the unfused FullAttention prep stages, then the PM
/// preconvert (null angle capture; Q and K rotated by the original launch).
fn run_incumbent_prep(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<()> {
    let (rows, u) = (c.rows, used(c.rows, c.codec));
    let (qn, kn, pos) = (a.qn().view(u.qn), a.kn().view(u.kn), a.pos().view(u.pos));
    let (k, q) = (a.k_inc().view(u.k_inc), a.q_inc().view(u.q_inc));
    gpu.deinterleave_q_rmsnorm_f32_batched(
        &a.inter().view(u.inter),
        &q,
        &a.gate_inc().view(u.gate_inc),
        &qn,
        H,
        D,
        rows,
        EPS,
    )?;
    gpu.rmsnorm_batched(&k, &kn, &k, rows * KV, D, EPS)?;
    gpu.rope_partial_interleaved_f32_batched(
        &q,
        &k,
        &pos,
        H,
        KV,
        D,
        NROT,
        THETA,
        rows,
        c.pos_offset,
    )?;
    gpu.fa_prefill_pm_preconvert(&q, &a.q16_inc().view(u.q16_inc), rows)?;
    Ok(())
}
/// New Q/K prep: K norm, K-only RoPE with angle capture, then the fused exact Q prep.
fn run_new_prep(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<()> {
    let (rows, u) = (c.rows, used(c.rows, c.codec));
    let (qn, kn, pos) = (a.qn().view(u.qn), a.kn().view(u.kn), a.pos().view(u.pos));
    let (k, angles) = (a.k_new().view(u.k_new), a.angles().view(u.angles));
    gpu.rmsnorm_batched(&k, &kn, &k, rows * KV, D, EPS)?;
    gpu.rope_partial_halfsplit_capture_f32_batched(
        None,
        &k,
        &pos,
        0,
        KV,
        D,
        NROT,
        THETA,
        rows,
        c.pos_offset,
        Some(&angles),
    )?;
    gpu.fa_pm_qprep_exact(
        &a.inter().view(u.inter),
        &a.q16_new().view(u.q16_new),
        &a.gate_new().view(u.gate_new),
        &qn,
        &angles,
        rows,
        EPS,
    )?;
    Ok(())
}
/// Both fold arms share the typed PM direct body and runtime cache bytes.
fn run_direct(gpu: &mut Gpu, a: &Arena, c: &Cell, q16: &Buf, out: &Buf) -> Result<()> {
    let (rows, u) = (c.rows, used(c.rows, c.codec));
    gpu.fa_prefill_pm_direct_codec(
        c.codec,
        &q16.view(rows * ROW * 2),
        &a.kc().view(u.kc),
        &a.vc().view(u.vc),
        &out.view(rows * ROW * 4),
        &a.pos().view(u.pos),
        c.ctx(),
        rows,
    )?;
    Ok(())
}

// ------------------------------------------------------------ cell data

struct Prepared {
    inter: Vec<f32>,
    k0: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    pos: Vec<i32>,
    kc: Vec<u8>,
    vc: Vec<u8>,
    seed: u64,
}
/// Generate inputs, arm canaries, upload, write the runtime codec's caches on the
/// device and snapshot them. Everything but the kernels under test.
fn prepare(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<Prepared> {
    let u = used(c.rows, c.codec);
    let ctx = c.ctx();
    let pos = c.fixture.positions(c.rows);
    validate_cell(H, KV, D, NROT, c.rows, &pos, ctx, c.pos_offset)?;
    let seed = fnv(&c.input_name()) ^ 0x5f5a3b0c9d1e7a41;
    let mut rng = Rng(seed);
    let scale = c.data.scale();
    let inter: Vec<f32> = if c.data == Data::Zeros {
        vec![0.0; c.rows * SRC_ROW]
    } else {
        (0..c.rows * SRC_ROW)
            .map(|_| rng.normal() * scale)
            .collect()
    };
    let qn: Vec<f32> = (0..D).map(|_| 1.0 + 0.25 * rng.normal()).collect();
    let kn: Vec<f32> = (0..D).map(|_| 1.0 + 0.25 * rng.normal()).collect();
    let k0: Vec<f32> = (0..c.rows * KROW).map(|_| rng.normal()).collect();
    let ksrc: Vec<f32> = (0..ctx * KROW).map(|_| rng.normal()).collect();
    let vsrc: Vec<f32> = (0..ctx * KROW).map(|_| rng.normal() * 0.7).collect();
    let allpos: Vec<i32> = (0..ctx as i32).collect();
    for (buf, n) in a.b.iter().zip(used_list(&u)) {
        buf.arm_guards(gpu, n)?;
    }
    a.inter().upload(gpu, bytes(&inter))?;
    a.qn().upload(gpu, bytes(&qn))?;
    a.kn().upload(gpu, bytes(&kn))?;
    a.k_inc().upload(gpu, bytes(&k0))?;
    a.k_new().upload(gpu, bytes(&k0))?;
    a.pos().upload(gpu, bytes(&pos))?;
    a.allpos().upload(gpu, bytes(&allpos))?;
    a.k_src().upload(gpu, bytes(&ksrc))?;
    a.v_src().upload(gpu, bytes(&vsrc))?;
    // Both K and V use the production writer for this codec. Attention arms
    // subsequently consume these exact same snapshots; no host conversion.
    for (cache, src, cache_len, src_len) in [
        (a.kc(), a.k_src(), u.kc, u.k_src),
        (a.vc(), a.v_src(), u.vc, u.v_src),
    ] {
        match c.codec {
            FaPrefillKvCodec::Q8_0 => gpu.kv_cache_write_q8_0_batched(
                &cache.view(cache_len),
                &src.view(src_len),
                &a.allpos().view(u.allpos),
                KV,
                D,
                ctx,
            )?,
            FaPrefillKvCodec::Fp8E4M3 => gpu.kv_cache_write_fp8_e4m3_batched(
                &cache.view(cache_len),
                &src.view(src_len),
                &a.allpos().view(u.allpos),
                KV,
                D,
                ctx,
            )?,
        }
    }
    gpu.hip.device_synchronize()?;
    drop((ksrc, vsrc, allpos));
    let kc = a.kc().download(gpu, u.kc)?;
    let vc = a.vc().download(gpu, u.vc)?;
    for (b, len) in [
        (a.q_inc(), u.q_inc),
        (a.gate_inc(), u.gate_inc),
        (a.gate_new(), u.gate_new),
        (a.q16_inc(), u.q16_inc),
        (a.q16_new(), u.q16_new),
        (a.angles(), u.angles),
        (a.out_inc(), u.out_inc),
        (a.out_new(), u.out_new),
    ] {
        b.poison(gpu, len)?;
    }
    Ok(Prepared {
        inter,
        k0,
        qn,
        kn,
        pos,
        kc,
        vc,
        seed,
    })
}
/// Inputs still bit-identical to the host snapshots; every guard intact.
fn integrity(gpu: &Gpu, a: &Arena, c: &Cell, p: &Prepared) -> Result<Value> {
    let u = used(c.rows, c.codec);
    let inputs = json!({
        "interleaved": a.inter().download(gpu, u.inter)? == bytes(&p.inter),
        "q_norm_weight": a.qn().download(gpu, u.qn)? == bytes(&p.qn),
        "k_norm_weight": a.kn().download(gpu, u.kn)? == bytes(&p.kn),
        "positions": a.pos().download(gpu, u.pos)? == bytes(&p.pos),
        "k_cache": a.kc().download(gpu, u.kc)? == p.kc,
        "v_cache": a.vc().download(gpu, u.vc)? == p.vc,
    });
    let mut guards = serde_json::Map::new();
    for (b, n) in a.b.iter().zip(used_list(&u)) {
        guards.insert(b.name.into(), json!(b.guards_ok(gpu, n)?));
    }
    let ok = inputs.as_object().unwrap().values().all(|v| v == true)
        && guards.values().all(|v| v == true);
    Ok(json!({"inputs_unchanged":inputs,"guards_intact":guards,
        "guard_bytes_each_side":GUARD,"pass":ok}))
}

/// Byte oracle for one cell. Returns the cell JSON and its pass flag.
fn oracle_cell(gpu: &mut Gpu, a: &Arena, c: &Cell) -> Result<(Value, Prepared)> {
    let p = prepare(gpu, a, c)?;
    let u = used(c.rows, c.codec);
    run_incumbent_prep(gpu, a, c)?;
    run_new_prep(gpu, a, c)?;
    run_direct(gpu, a, c, a.q16_inc(), a.out_inc())?;
    run_direct(gpu, a, c, a.q16_new(), a.out_new())?;
    gpu.hip.device_synchronize()?;
    // K after norm + RoPE: null capture + Q&K rotation vs captured K-only rotation.
    let k = compare(
        &a.k_inc().download(gpu, u.k_inc)?,
        &a.k_new().download(gpu, u.k_new)?,
        Elem::F32,
        KV,
        D,
    );
    let q16 = compare(
        &a.q16_inc().download(gpu, u.q16_inc)?,
        &a.q16_new().download(gpu, u.q16_new)?,
        Elem::F16,
        H,
        D,
    );
    let gate = compare(
        &a.gate_inc().download(gpu, u.gate_inc)?,
        &a.gate_new().download(gpu, u.gate_new)?,
        Elem::F32,
        H,
        D,
    );
    let attention = compare(
        &a.out_inc().download(gpu, u.out_inc)?,
        &a.out_new().download(gpu, u.out_new)?,
        Elem::F32,
        H,
        D,
    );
    let angles = angle_sanity(
        &f32s(&a.angles().download(gpu, u.angles)?),
        &p.pos,
        c.pos_offset,
    );
    let integ = integrity(gpu, a, c, &p)?;
    // Random-data cells must produce nonzero Q16/gate so identical poison/zero
    // output cannot pass; zero cells must produce an all-zero-magnitude Q16.
    let q16_bytes = a.q16_new().download(gpu, u.q16_new)?;
    let nonzero_q16 = q16_bytes
        .chunks_exact(2)
        .any(|x| u16::from_ne_bytes([x[0], x[1]]) & 0x7FFF != 0);
    let content_ok = match c.data {
        Data::Zeros => !nonzero_q16,
        Data::Random => nonzero_q16,
        // Overflow / underflow regimes: finite, byte-identical output is the gate.
        Data::Wide | Data::Tiny => true,
    };
    let byte_identical = pass(&k) && pass(&q16) && pass(&gate) && pass(&attention);
    let ok = byte_identical && pass(&angles) && pass(&integ) && content_ok;
    let first = [
        ("k_post_rope", &k),
        ("q16", &q16),
        ("gate", &gate),
        ("attention", &attention),
    ]
    .into_iter()
    .find(|(_, v)| !pass(v))
    .map(|(n, v)| json!({"stage":n,"first_mismatch":v["first_mismatch"],"detail":v}));
    let line = json!({"kind":"oracle_cell","cell":c.name(),"rows":c.rows,
        "codec":codec_name(c.codec),"cache_row_bytes":c.codec.cache_row_bytes(),
        "cache_writer":writer_name(c.codec),"direct_module":c.codec.direct_module(),
        "direct_symbol":c.codec.direct_symbol(),
        "data":c.data.name(),"positions_fixture":c.fixture.name(),"pos_offset":c.pos_offset,
        "context_rows":c.ctx(),"seed":p.seed,
        "k_post_rope":k,"q16":q16,"gate":gate,"attention":attention,
        "captured_angles":angles,"integrity":integ,"content_sanity_ok":content_ok,
        "byte_identical":byte_identical,"first_failure":first,"pass":ok});
    Ok((line, p))
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
/// `WARMUPS` rotated warmups per arm, then `iters` event-timed samples per arm in
/// microseconds; arm order rotates with every warmup/sample.
fn time_arms(gpu: &mut Gpu, arms: &mut [ArmFn], iters: usize) -> Result<Vec<Vec<f64>>> {
    let n = arms.len();
    for w in 0..WARMUPS {
        for k in 0..n {
            (arms[(w + k) % n])(gpu)?;
        }
    }
    gpu.hip.device_synchronize()?;
    let (ea, eb) = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let mut samples = vec![Vec::with_capacity(iters); n];
    for s in 0..iters {
        for k in 0..n {
            let i = (s + k) % n;
            gpu.hip.event_record(&ea, None)?;
            (arms[i])(gpu)?;
            gpu.hip.event_record(&eb, None)?;
            gpu.hip.event_synchronize(&eb)?;
            samples[i].push(gpu.hip.event_elapsed_ms(&ea, &eb)? as f64 * 1000.0);
        }
    }
    gpu.hip.event_destroy(ea)?;
    gpu.hip.event_destroy(eb)?;
    Ok(samples)
}
/// Timing of the two Q-prep sequences (attention excluded; it is the shared kernel).
/// K is normalized and rotated in place repeatedly; values stay finite.
fn time_cell(gpu: &mut Gpu, a: &Arena, c: &Cell, iters: usize) -> Result<Value> {
    let mut inc = |g: &mut Gpu| run_incumbent_prep(g, a, c);
    let mut new = |g: &mut Gpu| run_new_prep(g, a, c);
    let s = time_arms(gpu, &mut [&mut inc as ArmFn, &mut new as ArmFn], iters)?;
    let (mi, mn) = (median(&s[0]), median(&s[1]));
    Ok(json!({"kind":"timing","cell":c.name(),"rows":c.rows,
        "codec":codec_name(c.codec),
        "scope":"qprep sequences (attention excluded: identical shared PM direct kernel)",
        "incumbent_sequence":["deinterleave_q_rmsnorm_f32_batched","rmsnorm_batched(k)",
            "rope_partial_interleaved_f32_batched(q+k, null capture)","fa_prefill_pm_preconvert"],
        "new_sequence":["rmsnorm_batched(k)","rope_partial_halfsplit_capture_f32_batched(k-only, capture)",
            "fa_pm_qprep_exact"],
        "warmups_per_path":WARMUPS,"iterations_per_path":iters,
        "arm_order":"rotated: warmup w and sample s start at path (index mod 2)",
        "incumbent_us_all":s[0],"new_us_all":s[1],
        "incumbent_us_median":mi,"new_us_median":mn,
        "saving_us_per_layer_median":mi - mn,
        "speedup_median":if mn > 0.0 { json!(mi / mn) } else { Value::Null },
        "envelope_saving_ms_all_layers":(mi - mn) * LAYERS as f64 / 1000.0,"layers":LAYERS}))
}

// ------------------------------------------------------------------ plan

fn plan(cs: &[Cell]) {
    let kernarg_offsets = json!({"interleaved_f32_ptr":0,"q16_ptr":8,"gate_f32_ptr":16,
        "q_norm_f32_256_ptr":24,"angles_f32_rows_x_64_ptr":32,"rows_i32":40,"eps_f32":44});
    for c in cs {
        let u = used(c.rows, c.codec);
        let pos = c.fixture.positions(c.rows);
        let admission = validate_cell(H, KV, D, NROT, c.rows, &pos, c.ctx(), c.pos_offset);
        println!(
            "{}",
            json!({"plan_cell":c.name(),"rows":c.rows,"data":c.data.name(),
                "codec":codec_name(c.codec),"cache_row_bytes":c.codec.cache_row_bytes(),
                "positions_fixture":c.fixture.name(),"pos_offset":c.pos_offset,
                "context_rows":c.ctx(),"cache_positions_bytes":u.kc,"timed":c.timed,
                "admission":match &admission {Ok(()) => json!("ok"), Err(e) => json!(e)},
                "incumbent_launches":[
                    {"api":"deinterleave_q_rmsnorm_f32_batched","grid":[H, c.rows, 1],"block":256,"lds_bytes":256*4},
                    {"api":"rmsnorm_batched(k)","rows":c.rows*KV,"dim":D},
                    {"api":"rope_partial_interleaved_f32_batched (halfsplit, q+k, null capture)",
                        "grid":rope_grid(c.rows),"block":ROPE_BLOCK,"lds_bytes":ROPE_LDS},
                    {"api":"fa_prefill_pm_preconvert","grid":[(c.rows * H).div_ceil(4),1,1],"block":128}],
                "new_launches":[
                    {"api":"rmsnorm_batched(k)","rows":c.rows*KV,"dim":D},
                    {"api":"rope_partial_halfsplit_capture_f32_batched (k-only, angles captured)",
                        "grid":rope_grid(c.rows),"block":ROPE_BLOCK,"lds_bytes":ROPE_LDS},
                    {"api":"fa_pm_qprep_exact","symbol":fa_qprep_pm::SYMBOL,"grid":qprep_grid(c.rows),
                        "block":QPREP_BLOCK,"lds_bytes":0,"kernarg_bytes":QPREP_KERNARG_BYTES}],
                "shared":{"cache_writer":writer_name(c.codec),"writer_launches":2,
                    "cache_inputs":"same runtime-written K/V bytes for fold off/on",
                    "direct_api":"fa_prefill_pm_direct_codec","direct_module":c.codec.direct_module(),
                    "direct_symbol":c.codec.direct_symbol(),"direct_grid":[c.rows.div_ceil(8),KV,1],
                    "direct_block":fa_prefill_pm::DIRECT_BLOCK,"direct_lds_bytes":fa_prefill_pm::DIRECT_LDS,
                    "direct_kernarg_bytes":fa_prefill_pm::DIRECT_KERNARG_BYTES},
                "compared":["k_post_rope f32 [rows,2,256]","q16 f16 [rows,16,256]","gate f32 [rows,16,256]",
                    "attention f32 [rows,16,256]"],
                "diagnostics":["captured angle table [rows,64] vs f64 reference","inputs unchanged","guards intact","finite outputs"],
                "device_used_payload_bytes":used_list(&u).iter().sum::<usize>(),
                "host_peak_bytes":host_peak(c),"launches":oracle_launches()})
        );
    }
    let timed: Vec<&Cell> = cs.iter().filter(|c| c.timed).collect();
    println!(
        "{}",
        json!({"plan_timing":true,"cells":timed.iter().map(|c| c.name()).collect::<Vec<_>>(),
            "warmups_per_path":WARMUPS,"default_iterations_per_path":DEFAULT_ITERS,
            "minimum_iterations_per_path":MIN_ITERS,"unit":"microseconds, median of event-timed iterations",
            "paths":["incumbent qprep sequence","new qprep sequence"],
            "attention_timed":false,"layers_envelope":LAYERS})
    );
    let buffers: Vec<Value> = BUFFER_NAMES
        .iter()
        .zip(arena_sizes(&cs))
        .map(|(n, b)| json!({"name":n,"payload_bytes":b,"allocated_bytes":b + 2 * GUARD}))
        .collect();
    println!(
        "{}",
        json!({"plan":true,"gpu_launched":false,"module_key":fa_qprep_pm::MODULE,
            "symbol":fa_qprep_pm::SYMBOL,
            "contract":{"heads":H,"kv_heads":KV,"head_dim":D,"n_rot":NROT,"rope_theta":THETA,"eps":EPS,
                "kernarg_bytes":QPREP_KERNARG_BYTES,"kernarg_offsets":kernarg_offsets,
                "block":QPREP_BLOCK,"heads_per_block":HEADS_PER_BLOCK,"wave_per_head":true,
                "grid":"[rows*16/4,1,1]","interleaved_row_stride_floats":SRC_ROW,
                "head_layout":"first 256 q, then 256 gate","q16_layout":"[rows,16,256] f16",
                "gate_layout":"[rows,16,256] f32","angle_row":"cos[0..32], sin[32..64]",
                "exact_rows":"1..=8192","production_fa_rows":"64..=8192"},
            "oracle_cells":cs.len(),"timed_cells":timed.len(),
            "codec_census":codec_census(cs),"arena_shared_between_cells":true,
            "required_shapes":[512,4096,8192],"required_data":["zeros","random"],
            "extras":["scrambled8 positions + pos_offset 37 (rows 512)","rows 65","wide (1e20)","tiny (1e-30)"],
            "device_arena":buffers,"device_arena_total_bytes":arena_total(&cs),
            "host_peak_bytes":cs.iter().map(host_peak).max().unwrap(),
            "runtime_budget_s":DEFAULT_BUDGET_S,
            "output":"one JSON line per cell and per timing, then {summary:true,...} and `all_byte_identical true|false`",
            "run":"cargo run --release -p rdna-compute --features lab,deltanet --example fa_pm_qprep_oracle"})
    );
}

// ------------------------------------------------------------ self-check

/// `fma(q,q,+0)` leaf squares of one head.
fn leaves(q: &[f32; D]) -> [f32; D] {
    let mut s = [0f32; D];
    for (o, &x) in s.iter_mut().zip(q) {
        *o = x.mul_add(x, 0.0);
    }
    s
}
/// Incumbent: one leaf per thread, LDS halvings 128, 64, ..., 1.
fn tree_incumbent(leaf: &[f32; D]) -> f32 {
    let mut s = *leaf;
    let mut stride = D / 2;
    while stride >= 1 {
        for t in 0..stride {
            s[t] += s[t + stride];
        }
        stride /= 2;
    }
    s[0]
}
/// One wave: lane `l` owns dims `l + 32k`; (k, k+4), (k, k+2), (k, k+1), then XOR
/// 16, 8, 4, 2, 1. Returns lane 0 and whether all 32 lanes agree bit for bit.
fn tree_wave(leaf: &[f32; D]) -> (f32, bool) {
    let mut lane = [0f32; 32];
    for (l, out) in lane.iter_mut().enumerate() {
        let mut s = [0f32; 8];
        for (k, v) in s.iter_mut().enumerate() {
            *v = leaf[l + 32 * k];
        }
        for k in 0..4 {
            s[k] = s[k] + s[k + 4];
        }
        for k in 0..2 {
            s[k] = s[k] + s[k + 2];
        }
        *out = s[0] + s[1];
    }
    for off in [16usize, 8, 4, 2, 1] {
        let prev = lane;
        for l in 0..32 {
            lane[l] = prev[l] + prev[l ^ off];
        }
    }
    (
        lane[0],
        lane.iter().all(|x| x.to_bits() == lane[0].to_bits()),
    )
}
/// Deterministic scalar model of the Q prep of one head; the reduction tree is the
/// only variable. `rsqrt` is a model function (identical for both trees).
fn model_head(
    src: &[f32; 2 * D],
    w: &[f32; D],
    angles: &[f32; NROT],
    eps: f32,
    wave: bool,
) -> ([f32; D], [f32; D]) {
    let q: [f32; D] = src[..D].try_into().unwrap();
    let gate: [f32; D] = src[D..].try_into().unwrap();
    let leaf = leaves(&q);
    let sum = if wave {
        tree_wave(&leaf).0
    } else {
        tree_incumbent(&leaf)
    };
    let rms = 1.0 / ((sum / D as f32 + eps) as f64).sqrt() as f32;
    let mut o = [0f32; D];
    for d in 0..D {
        o[d] = q[d] * w[d] * rms;
    }
    for i in 0..HALF {
        let (x0, x1) = (o[i], o[i + HALF]);
        let (c, s) = (angles[i], angles[HALF + i]);
        o[i] = x0.mul_add(c, -(x1 * s));
        o[i + HALF] = x0.mul_add(s, x1 * c);
    }
    (o, gate)
}
fn model_angles(pos: i64) -> [f32; NROT] {
    let mut t = [0f32; NROT];
    for i in 0..HALF {
        let (c, s) = angle_reference(pos, i);
        t[i] = c as f32;
        t[HALF + i] = s as f32;
    }
    t
}

fn self_check() -> usize {
    let mut n = 0usize;
    let mut ok = |cond: bool, what: &str| {
        assert!(cond, "self-check failed: {what}");
        n += 1;
    };
    // Launcher constants shared with the host module.
    ok(
        fa_qprep_pm::KERNARG_BYTES == QPREP_KERNARG_BYTES,
        "host kernarg bytes 48",
    );
    ok(fa_qprep_pm::BLOCK == [QPREP_BLOCK, 1, 1], "host block 128");
    ok(
        fa_qprep_pm::HEADS == H && fa_qprep_pm::HEAD_DIM == D && fa_qprep_pm::ANGLE_STRIDE == NROT,
        "host H16/D256/angle stride 64",
    );
    ok(
        fa_qprep_pm::N_ROT == NROT && fa_qprep_pm::MAX_ROWS == MAX_ROWS,
        "host n_rot 64 / max rows 8192",
    );
    ok(
        [1usize, 4, 63, 64, 65, 512, 4096, 8192]
            .iter()
            .all(|&r| fa_qprep_pm::grid(r) == qprep_grid(r)),
        "host grid == rows*16/4",
    );
    ok(fa_qprep_pm::MODULE == "fa_qprep_pm_gfx1201", "module key");
    ok(
        fa_qprep_pm::SYMBOL == "attention_fa2_qprep_pm_gfx1201",
        "symbol",
    );
    ok(
        fa_qprep_pm::IMAGE.starts_with(b"\x7fELF")
            || fa_qprep_pm::IMAGE.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"),
        "embedded bundle is a code object",
    );
    ok(
        DType::Raw.size() == 1 && DType::F32.size() == 4,
        "dtype sizes used by Buf",
    );
    ok(
        ROW == 4096 && SRC_ROW == 8192 && KROW == 512 && ROPE_LDS == 256,
        "dimension constants",
    );
    ok(
        FaPrefillKvCodec::Q8_0.cache_row_bytes() == KV * (D / 32) * 34
            && FaPrefillKvCodec::Q8_0.cache_row_bytes() == 544,
        "Q8 row remains 544 bytes",
    );
    ok(
        FaPrefillKvCodec::Fp8E4M3.cache_row_bytes() == KROW + KV * 2
            && FaPrefillKvCodec::Fp8E4M3.cache_row_bytes() == 516,
        "FP8 row: codes then f16 scales",
    );
    ok(
        FaPrefillKvCodec::Fp8E4M3.direct_module() == "fa_prefill_fp8_pm_gfx1201"
            && FaPrefillKvCodec::Fp8E4M3.direct_symbol()
                == "attention_fp8_e4m3_fa2_gqa_f16_pm_gfx1201",
        "native FP8 module and symbol",
    );
    ok(
        fa_prefill_pm::DIRECT_KERNARG_BYTES == 60
            && fa_prefill_pm::DIRECT_BLOCK == [160, 1, 1]
            && fa_prefill_pm::DIRECT_LDS == 65536,
        "shared typed direct ABI/block/LDS",
    );
    ok(
        fa_prefill_pm::FP8_IMAGE.starts_with(b"\x7fELF")
            || fa_prefill_pm::FP8_IMAGE.starts_with(b"__CLANG_OFFLOAD_BUNDLE__"),
        "embedded FP8 bundle is a code object",
    );
    // Kernarg layout (explicit-argument segment: no trailing pad).
    let p = |x: usize| x as *mut c_void;
    let b = qprep_kernargs(p(0x10), p(0x20), p(0x30), p(0x40), p(0x50), 77, 1e-6);
    let s = b.as_bytes();
    ok(s.len() == QPREP_KERNARG_BYTES, "qprep kernarg size 48");
    for i in 0..5 {
        ok(
            usize::from_ne_bytes(s[i * 8..i * 8 + 8].try_into().unwrap()) == 0x10 * (i + 1),
            "qprep pointer order interleaved,q16,gate,qnorm,angles @0..40",
        );
    }
    ok(
        i32::from_ne_bytes(s[40..44].try_into().unwrap()) == 77,
        "qprep rows @40",
    );
    ok(
        f32::from_ne_bytes(s[44..48].try_into().unwrap()) == 1e-6,
        "qprep eps @44",
    );
    ok(
        qprep_grid(8192) == [32768, 1, 1]
            && qprep_grid(512) == [2048, 1, 1]
            && qprep_grid(65) == [260, 1, 1]
            && qprep_grid(1) == [4, 1, 1],
        "qprep grids rows*16/4",
    );
    ok(rope_grid(4096) == [1, 4096, 1], "rope grid [1, rows, 1]");
    // Admission truth table.
    let std = Fixture::Std.positions(64);
    let good =
        |h, kv, d, r, rows, pos: &[i32], ctx, off| validate_cell(h, kv, d, r, rows, pos, ctx, off);
    ok(
        good(16, 2, 256, 64, 64, &std, 64, 0).is_ok(),
        "admits 64 rows",
    );
    ok(
        good(24, 4, 256, 64, 64, &std, 64, 0).is_err(),
        "refuses H24/KV4",
    );
    ok(
        good(16, 4, 256, 64, 64, &std, 64, 0).is_err(),
        "refuses KV4",
    );
    ok(
        good(16, 2, 128, 64, 64, &std, 64, 0).is_err(),
        "refuses D128",
    );
    ok(
        good(16, 2, 256, 32, 64, &std, 64, 0).is_err(),
        "refuses n_rot 32",
    );
    ok(
        good(16, 2, 256, 64, 0, &[], 64, 0).is_err(),
        "refuses 0 rows",
    );
    ok(
        good(16, 2, 256, 64, 8193, &vec![0; 8193], 9000, 0).is_err(),
        "refuses 8193 rows",
    );
    ok(
        good(16, 2, 256, 64, 63, &std, 64, 0).is_err(),
        "refuses length mismatch",
    );
    ok(
        good(16, 2, 256, 64, 1, &[-1], 64, 0).is_err(),
        "refuses negative position",
    );
    ok(
        good(16, 2, 256, 64, 1, &[64], 64, 0).is_err(),
        "refuses position >= context",
    );
    ok(
        good(16, 2, 256, 64, 64, &std, 64, -1).is_err(),
        "refuses negative pos_offset",
    );
    ok(
        good(16, 2, 256, 64, 64, &std, 64, i32::MAX).is_err(),
        "refuses position + pos_offset overflow",
    );
    ok(
        good(16, 2, 256, 64, 64, &std, 64, 37).is_ok(),
        "admits pos_offset 37",
    );
    // Cells.
    let cs = cells();
    ok(cs.len() == 20, "cell count 20: ten per codec");
    for codec in CODECS {
        let group: Vec<&Cell> = cs.iter().filter(|c| c.codec == codec).collect();
        ok(group.len() == 10, "ten cells per codec");
        for rows in [512usize, 4096, 8192] {
            for data in [Data::Zeros, Data::Random] {
                ok(
                    group.iter().any(|c| {
                        c.rows == rows
                            && c.data == data
                            && c.fixture == Fixture::Std
                            && c.pos_offset == 0
                    }),
                    "required rows x data cell present for each codec",
                );
            }
        }
        ok(
            group
                .iter()
                .filter(|c| c.timed)
                .map(|c| c.rows)
                .collect::<Vec<_>>()
                == [512, 4096, 8192],
            "three random timing cells per codec",
        );
        ok(
            group.iter().any(|c| c.rows == 65),
            "partial eight-row tile per codec",
        );
        ok(
            group
                .iter()
                .any(|c| c.pos_offset == 37 && c.fixture == Fixture::Scrambled),
            "nonaligned, nonmonotone, duplicate positions per codec",
        );
    }
    for pair in cs[..10].iter().zip(&cs[10..]) {
        let (q8, fp8) = pair;
        ok(
            q8.codec == FaPrefillKvCodec::Q8_0
                && fp8.codec == FaPrefillKvCodec::Fp8E4M3
                && q8.input_name() == fp8.input_name()
                && q8.name() == q8.input_name()
                && q8.name() != fp8.name(),
            "paired fixtures preserve Q8 identity and RNG seed",
        );
        let (uq8, ufp8) = (used(q8.rows, q8.codec), used(fp8.rows, fp8.codec));
        ok(
            uq8.kc - ufp8.kc == q8.ctx() * 28,
            "FP8 uses 28 fewer bytes per cache row",
        );
    }
    ok(
        parse_codec("q8").unwrap() == Some(FaPrefillKvCodec::Q8_0)
            && parse_codec("fp8").unwrap() == Some(FaPrefillKvCodec::Fp8E4M3)
            && parse_codec("all").unwrap().is_none()
            && parse_codec("bf16").is_err(),
        "CLI codec selection is typed and fail-closed",
    );
    ok(
        selected_cells(None).len() == 20
            && CODECS.iter().all(|&codec| {
                let selected = selected_cells(Some(codec));
                selected.len() == 10
                    && selected.iter().all(|c| c.codec == codec)
                    && codec_census(&selected)[0]["cells"] == 10
            }),
        "codec selection and census retain all ten fixtures",
    );
    ok(
        cs.iter().all(|c| {
            c.timed
                == (c.data == Data::Random
                    && c.fixture == Fixture::Std
                    && c.pos_offset == 0
                    && [512, 4096, 8192].contains(&c.rows))
        }),
        "timed flag only required random cells",
    );
    ok(
        cs.iter()
            .any(|c| c.pos_offset != 0 && c.fixture == Fixture::Scrambled),
        "offset + scrambled-position case present",
    );
    for c in &cs {
        let pos = c.fixture.positions(c.rows);
        ok(
            validate_cell(H, KV, D, NROT, c.rows, &pos, c.ctx(), c.pos_offset).is_ok(),
            "cell admission",
        );
        let u = used(c.rows, c.codec);
        ok(
            used_list(&u).iter().all(|&x| x % 4 == 0),
            "buffer bytes 4-aligned",
        );
        ok(
            u.q16_inc * 2 == u.q_inc && u.kc == c.ctx() * c.codec.cache_row_bytes() && u.kc == u.vc,
            "q16/codec cache extents",
        );
    }
    let (non_monotone, dup) = {
        let v = Fixture::Scrambled.positions(512);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        (
            v.windows(2).any(|w| w[1] < w[0]),
            sorted.windows(2).any(|w| w[0] == w[1]),
        )
    };
    ok(
        non_monotone && dup,
        "scrambled8 is nonmonotone with duplicates",
    );
    ok(
        Fixture::Std
            .positions(512)
            .windows(2)
            .all(|w| w[1] == w[0] + 1),
        "std positions monotone",
    );
    for rows in [65usize, 512, 1023] {
        let pos = Fixture::Scrambled.positions(rows);
        ok(
            pos.iter().enumerate().all(|(r, &x)| {
                let g = r / 8;
                (x as usize) >= g * 8 && (x as usize) < g * 8 + (rows - g * 8).min(8)
            }),
            "scrambled positions stay inside their 8-row group window",
        );
    }
    ok(
        arena_total(&cs)
            == used_list(&used(8192, FaPrefillKvCodec::Q8_0))
                .iter()
                .map(|b| b + 2 * GUARD)
                .sum::<usize>(),
        "arena total is the 8192-row cell",
    );
    // Layout mapping proof: every (row, head, dim) destination is written exactly
    // once from the interleaved row, and the RoPE partner of dim d lives in the
    // same lane (`l + 32k`) in the one-wave layout.
    let mut seen = vec![0u8; 2 * ROW];
    for h in 0..H {
        for d in 0..D {
            seen[h * 2 * D + d] += 1; // q slot
            seen[h * 2 * D + D + d] += 1; // gate slot
        }
    }
    ok(
        seen.iter().all(|&x| x == 1),
        "head-major q/gate slots tile the interleaved row",
    );
    for d in 0..NROT {
        ok(
            d % 32 == (d ^ 32) % 32 && d / 32 != (d ^ 32) / 32,
            "rope partner d^32 in same lane, k and k+1",
        );
    }
    // Reduction-tree proof: the one-wave tree equals the incumbent LDS halving tree
    // bit for bit (and all lanes agree) over many seeds and regimes.
    let mut rng = Rng(0x7ee5_0f5a_3b0c);
    let mut trees_ok = true;
    let mut lanes_ok = true;
    for trial in 0..4000 {
        let scale = match trial % 5 {
            0 => 1.0f32,
            1 => 1e-20,
            2 => 1e18,
            3 => 1e-5,
            _ => 3.0,
        };
        let mut q = [0f32; D];
        for x in q.iter_mut() {
            *x = rng.normal() * scale;
        }
        if trial % 7 == 0 {
            for x in q.iter_mut().skip(trial % 31).step_by(3) {
                *x = 0.0;
            }
        }
        let leaf = leaves(&q);
        let a = tree_incumbent(&leaf).to_bits();
        let (b, lanes) = tree_wave(&leaf);
        trees_ok &= a == b.to_bits();
        lanes_ok &= lanes;
    }
    ok(
        trees_ok,
        "one-wave tree == incumbent LDS tree (4000 random heads)",
    );
    ok(
        lanes_ok,
        "XOR butterfly leaves every lane with the same bits",
    );
    let adversarial: [[f32; D]; 4] = [
        [0.0; D],
        [1e20; D],
        [1e-30; D],
        std::array::from_fn(|i| if i % 2 == 0 { 3.0e19 } else { -1.0e-3 }),
    ];
    for q in &adversarial {
        let leaf = leaves(q);
        ok(
            tree_incumbent(&leaf).to_bits() == tree_wave(&leaf).0.to_bits(),
            "trees agree on zero / overflow / underflow heads",
        );
    }
    // End-to-end scalar model: wave tree vs incumbent tree, whole head output bits.
    let mut model_ok = true;
    for trial in 0..64usize {
        let mut src = [0f32; 2 * D];
        for x in src.iter_mut() {
            *x = rng.normal() * if trial % 3 == 0 { 40.0 } else { 1.0 };
        }
        let mut w = [0f32; D];
        for x in w.iter_mut() {
            *x = 1.0 + 0.25 * rng.normal();
        }
        let ang = model_angles((trial * 131 + 37) as i64);
        let (qa, ga) = model_head(&src, &w, &ang, EPS, false);
        let (qb, gb) = model_head(&src, &w, &ang, EPS, true);
        model_ok &= qa.iter().zip(&qb).all(|(x, y)| x.to_bits() == y.to_bits())
            && ga == gb
            && ga[..] == src[D..];
    }
    ok(
        model_ok,
        "scalar head model: q16 inputs and gate passthrough bit-identical",
    );
    // Angle sanity machinery.
    let pos = [0i32, 5, 8191];
    let mut table = Vec::new();
    for &p in &pos {
        table.extend_from_slice(&model_angles(p as i64 + 37));
    }
    ok(
        pass(&angle_sanity(&table, &pos, 37)),
        "angle_sanity accepts the reference table",
    );
    table[NROT + 3] = f32::NAN;
    ok(
        !pass(&angle_sanity(&table, &pos, 37)),
        "angle_sanity rejects an unwritten entry",
    );
    table[NROT + 3] = 5.0;
    ok(
        !pass(&angle_sanity(&table, &pos, 37)),
        "angle_sanity rejects a wrong entry",
    );
    // Compare machinery.
    let a32: Vec<f32> = (0..2 * ROW).map(|i| (i as f32) * 0.25 + 1.0).collect();
    let mut b32 = a32.clone();
    ok(
        pass(&compare(bytes(&a32), bytes(&b32), Elem::F32, H, D)),
        "identical passes",
    );
    b32[ROW + 3 * D + 7] = f32::from_bits(b32[ROW + 3 * D + 7].to_bits() + 3);
    let r = compare(bytes(&a32), bytes(&b32), Elem::F32, H, D);
    ok(
        !pass(&r) && r["mismatched_elements"] == 1,
        "single flip fails with count 1",
    );
    ok(
        r["first_mismatch"]["row"] == 1
            && r["first_mismatch"]["head"] == 3
            && r["first_mismatch"]["dim"] == 7
            && r["first_mismatch"]["index"] == ROW + 3 * D + 7,
        "first mismatch coordinates",
    );
    ok(
        r["first_mismatch"]["incumbent_bits"] != r["first_mismatch"]["new_bits"],
        "first mismatch carries both bit patterns",
    );
    b32[0] = f32::NAN;
    let r = compare(bytes(&b32), bytes(&b32), Elem::F32, H, D);
    ok(
        !pass(&r) && r["byte_identical"] == true && r["nonfinite_new"] == 1,
        "identical NaN still fails finite gate",
    );
    let h16: Vec<u16> = vec![0x3c00, 0x0000, 0x8000];
    let mut g16 = h16.clone();
    g16[1] = 0x8000;
    let r = compare(bytes(&h16), bytes(&g16), Elem::F16, 1, 1);
    ok(
        !pass(&r) && r["mismatched_elements"] == 1,
        "signed zero differs in f16 bytes",
    );
    ok(
        !pass(&compare(
            bytes(&[0x7c00u16]),
            bytes(&[0x7c00u16]),
            Elem::F16,
            1,
            1,
        )),
        "f16 inf is nonfinite",
    );
    ok(
        median(&[1.0, 3.0, 2.0, 4.0]) == 2.5 && median(&[3.0, 1.0, 2.0]) == 2.0,
        "median",
    );
    ok(
        sha_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha256",
    );
    n
}

// ------------------------------------------------------------------ main

struct Opts {
    plan: bool,
    check: bool,
    time: bool,
    iters: usize,
    budget_s: f64,
    codec: Option<FaPrefillKvCodec>,
}
fn parse() -> Result<Opts> {
    let mut o = Opts {
        plan: false,
        check: false,
        time: true,
        iters: DEFAULT_ITERS,
        budget_s: DEFAULT_BUDGET_S,
        codec: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--plan" => o.plan = true,
            "--self-check" => o.check = true,
            "--no-time" => o.time = false,
            "--kv-codec" => {
                o.codec = parse_codec(&it.next().ok_or("--kv-codec needs q8|fp8|all")?)?;
            }
            "--iters" => {
                o.iters = it.next().ok_or("--iters needs N")?.parse()?;
                if o.iters < MIN_ITERS {
                    return Err(format!("--iters must be >= {MIN_ITERS}").into());
                }
            }
            "--budget-s" => o.budget_s = it.next().ok_or("--budget-s needs SECONDS")?.parse()?,
            _ => {
                return Err(format!(
                    "unknown argument {a}; usage: fa_pm_qprep_oracle [--kv-codec q8|fp8|all] [--plan | --self-check | [--no-time] [--iters N] [--budget-s S]]"
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
    let cs = selected_cells(o.codec);
    if o.plan {
        plan(&cs);
        return Ok(true);
    }
    if o.check {
        println!(
            "{}",
            json!({"self_check":"passed","checks":n,"gpu_launched":false,
            "codec_census":codec_census(&cs),"cells":cs.len()})
        );
        return Ok(true);
    }
    if std::env::var("HIPFIRE_ROPE_INTERLEAVED_LEGACY").is_ok_and(|v| v == "1") {
        return Err("unset HIPFIRE_ROPE_INTERLEAVED_LEGACY: the F5 route is halfsplit only".into());
    }
    let t0 = Instant::now();
    println!(
        "{}",
        json!({"kind":"qprep_bundle","module_key":fa_qprep_pm::MODULE,"symbol":fa_qprep_pm::SYMBOL,
            "embedded_bytes":fa_qprep_pm::IMAGE.len(),"embedded_sha256":sha_hex(fa_qprep_pm::IMAGE),
            "budget_s":o.budget_s})
    );
    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1201" {
        return Err(format!("requires gfx1201, got {}", gpu.arch).into());
    }
    let arena = Arena::new(&mut gpu, &cs)?;
    let (mut identical, mut failures, mut all_identical) = (0usize, 0usize, true);
    let mut timings = Vec::new();
    for c in &cs {
        let (line, _p) = oracle_cell(&mut gpu, &arena, c)?;
        println!("{line}");
        identical += (line["byte_identical"] == true) as usize;
        failures += (line["pass"] != true) as usize;
        all_identical &= line["byte_identical"] == true;
        if o.time && c.timed {
            let t = time_cell(&mut gpu, &arena, c, o.iters)?;
            println!("{t}");
            timings.push(
                json!({"cell":c.name(),"incumbent_us_median":t["incumbent_us_median"],
                "codec":codec_name(c.codec),
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
            "codec_census":codec_census(&cs),
            "failures":failures,"all_byte_identical":all_identical,"timings":timings,
            "elapsed_s":t0.elapsed().as_secs_f64()})
    );
    println!("all_byte_identical {all_identical}");
    Ok(failures == 0 && all_identical)
}
fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("fa_pm_qprep_oracle: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_codec_cpu_contracts() {
        // Includes both-codec fixture/admission/census, ABI, mapping, exact
        // reduction-tree and mismatch-detection tests without HIP initialization.
        assert!(self_check() > 0);
    }

    #[test]
    fn fp8_only_arena_uses_native_cache_extents() {
        let q8 = selected_cells(Some(FaPrefillKvCodec::Q8_0));
        let fp8 = selected_cells(Some(FaPrefillKvCodec::Fp8E4M3));
        let q8_sizes = arena_sizes(&q8);
        let fp8_sizes = arena_sizes(&fp8);
        for i in 0..BUFFER_NAMES.len() {
            if i == 13 || i == 14 {
                assert_eq!(q8_sizes[i], 8192 * 544);
                assert_eq!(fp8_sizes[i], 8192 * 516);
            } else {
                assert_eq!(q8_sizes[i], fp8_sizes[i]);
            }
        }
        assert_eq!(arena_total(&cells()), arena_total(&q8));
        assert_eq!(arena_total(&q8) - arena_total(&fp8), 2 * 8192 * 28);
        assert_eq!(
            writer_name(FaPrefillKvCodec::Fp8E4M3),
            "kv_cache_write_fp8_e4m3_batched"
        );
    }
}
