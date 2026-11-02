// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C gate for the gfx942 qt=55 `FP8E4M3B128` kernels:
//! `dequant_fp8b128_to_bf16_gfx942` and `gemm_fp8b128_moe_grouped_bf16mfma`
//! (F32-x and BF16-x entries).
//!
//! 1. Decode identity: every non-NaN code x a scale sweep, BF16 bits vs the CPU
//!    teacher twin `rdna_compute::fp8_e4m3::dequant_b128_block`; with
//!    `--fixtures DIR` also vs bits written by the Python teacher
//!    (`fp8_teacher.dequant_e4m3_block`) for 256 codes x the checkpoint's scales
//!    and for one real gate_up/down expert (`mkfix.py` layout).
//! 2. Grouped GEMM vs an F64 reference on the CPU-decoded W and BF16 x (Qwen4
//!    gate_up r=1280 k=2560 x_row_div=10, down r=2560 k=640 x_row_div=1, top-10
//!    scatter over 12 experts, padding rows, a skipped tile); bound
//!    |err| <= k * 2^-24 * sum|w x| (F32 accumulation of exact products).
//! 3. F32-x and BF16-x entries byte-identical; compact tiles (nrows < 16)
//!    byte-identical to the padded layout and nothing past live rows written.
//! 4. Batch invariance: bytes(M=1) == row r of bytes(M=n), n in {2,7,8,16,17,128},
//!    every row; every row of the grouped launch == its single-row launch; and
//!    tile-position independence (a row at each of the 16 tile positions, tile
//!    index 0 and 5, == bytes(M=1)).
//! 5. `--byte-identity` (exact-byte A/B gate for any kernel candidate): the whole
//!    suite above runs with every grouped launch executed twice per X dtype,
//!    once through the retained P1 baseline (`cdc37659a` body, compiled from the
//!    standalone SHA-pinned snapshot
//!    `tests/fixtures/fp8_moe_grouped_reference_p1_cdc37659a.gfx942.hip`, no code
//!    shared with the production kernel; hidden launcher
//!    `gemm_fp8b128_moe_grouped_bf16mfma_reference`, never a production
//!    selection) and once through the production entry, over the same
//!    descriptors/X/POISON-filled Y; the raw F32 bits of the WHOLE Y (live,
//!    padding, skipped-tile and guard sentinel rows, signed zeros) must be
//!    identical, zero mismatches. The baseline's bytes stay the reference: it
//!    is never re-pinned to the candidate. The mode adds (a) the R in
//!    {16,48,80} x K in {128,256,384} (K=256 = one full D=2 group) output-wave/prefetch-tail shapes through the
//!    full existing case, (b) finite-FP8 +/-0/subnormal/max/sign codes over
//!    extreme/zero/negative/subnormal scales, (c) BF16 X with +/-0, subnormals,
//!    extreme exponents and cancellation-heavy (exactly mirrored W/X, so the
//!    result is pure accumulation-order residue) rows, and (d) the C1 route
//!    (10 distinct experts, one token) plus 16/512/2048-token Zipf-like
//!    top-10 routes over 512 experts at the real 1280x2560 / 2560x640 shapes,
//!    compared on every live output with no F64 oracle at those sizes.
//!
//! 6. `--time` (exclusive of the suite; needs no fixtures): reference-vs-production
//!    kernel timing, see [`time_mode`]: per X dtype x {gate_up, down} x tokens
//!    {1,16,512,2048} (top-10-of-512 routes; 1 token = C1), 5 warmups + 60
//!    alternating event-timed launches into reused buffers, whole-Y raw-bit
//!    mismatch counts, JSON with source/object/compiler hashes.
//!
//! Run: `cargo run --release -p rdna-compute --features lab --example test_fp8_moe_gfx942 -- [--fixtures DIR] [--byte-identity]`
//! Time: `... -- --time [--time-out PATH.json] [--time-tokens 1,16,512,2048] [--time-shapes gate_up,down] [--time-xdt f32|bf16|both]`

use rdna_compute::fp8_e4m3::{
    b128_block_stride, b128_scale_count, bf16_to_f32, dequant_b128_block,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::cell::{Cell, RefCell};

const POISON: f32 = 12345.625;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn f32_to_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

/// One qt=55 block: codes || s_inv LE || zero pad to the stride.
fn block_bytes(codes: &[u8], scales: &[u16], r: usize, k: usize) -> Vec<u8> {
    let stride = b128_block_stride(r, k).unwrap();
    let mut b = Vec::with_capacity(stride);
    b.extend_from_slice(codes);
    for s in scales {
        b.extend_from_slice(&s.to_le_bytes());
    }
    b.resize(stride, 0);
    b
}

/// Realistic positive block scale (the checkpoint's s_inv live in ~2^-14..2^-7).
fn rand_scale(st: &mut u64) -> u16 {
    let e = 113 + lcg(st) % 8; // 2^-14 .. 2^-7
    ((e << 7) | (lcg(st) & 0x7F)) as u16
}

fn rand_codes(n: usize, st: &mut u64) -> Vec<u8> {
    (0..n)
        .map(|_| {
            let b = (lcg(st) >> 8) as u8;
            if b & 0x7F == 0x7F {
                b ^ 1
            } else {
                b
            }
        })
        .collect()
}

fn upload_bytes(gpu: &Gpu, b: &[u8]) -> GpuTensor {
    gpu.upload_raw(b, &[b.len()]).expect("upload bytes")
}

fn upload_i32(gpu: &Gpu, v: &[i32]) -> GpuTensor {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut t = upload_bytes(gpu, &b);
    t.shape = vec![v.len()];
    t.dtype = DType::F32; // 4-byte carrier so byte_size() covers the payload
    t
}

struct Fail(Vec<String>);
impl Fail {
    fn check(&mut self, ok: bool, what: String) {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.0.push(what);
        }
    }
}

/// GPU dequant of `e` blocks (`r` x `k`) -> BF16 bits.
fn gpu_dequant(gpu: &mut Gpu, blocks: &[u8], e: usize, r: usize, k: usize) -> Vec<u16> {
    let d_blk = upload_bytes(gpu, blocks);
    let d_out = upload_bytes(gpu, &vec![0xA5u8; e * r * k * 2]);
    gpu.try_gfx942()
        .expect("exact gfx942")
        .dequant_fp8b128_to_bf16(&d_blk, &d_out, e, r, k)
        .expect("dequant_fp8b128_to_bf16");
    gpu.hip.device_synchronize().expect("sync");
    let bytes = gpu.download_raw_bytes(&d_out).expect("download");
    gpu.free_tensor(d_out).ok();
    gpu.free_tensor(d_blk).ok();
    bytes
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect()
}

struct Experts {
    r: usize,
    k: usize,
    n: usize,
    payload: Vec<u8>,
    w: Vec<Vec<u16>>, // CPU-decoded BF16 per expert
}

impl Experts {
    fn synthetic(r: usize, k: usize, n: usize, seed: u64) -> Self {
        let mut st = seed;
        let mut payload = Vec::new();
        let mut w = Vec::new();
        for _ in 0..n {
            let codes = rand_codes(r * k, &mut st);
            let scales: Vec<u16> = (0..b128_scale_count(r, k).unwrap())
                .map(|_| rand_scale(&mut st))
                .collect();
            w.push(dequant_b128_block(&codes, &scales, r, k).expect("cpu decode"));
            payload.extend(block_bytes(&codes, &scales, r, k));
        }
        Experts {
            r,
            k,
            n,
            payload,
            w,
        }
    }
}

struct Launch<'a> {
    ex: &'a Experts,
    d_payload: &'a GpuTensor,
    x_row_div: usize,
    x_rows: usize,
    x_f32: &'a GpuTensor,
    x_bf16: &'a GpuTensor,
    /// `--byte-identity`: every launch is also run through the retained P1
    /// baseline and compared bit-for-bit.
    ab: Option<&'a Ab>,
    tag: &'a str,
}

type Tile = [i32; 4]; // {expert, row0, nrows, 0}

impl Launch<'_> {
    /// One grouped launch (candidate = production entry, or the retained P1
    /// baseline when `reference`); returns Y [rows.len(), r] (POISON where the
    /// kernel wrote nothing).
    fn raw(
        &self,
        gpu: &mut Gpu,
        tiles: &[Tile],
        rows: &[i32],
        bf16_x: bool,
        reference: bool,
    ) -> Vec<f32> {
        let stride = b128_block_stride(self.ex.r, self.ex.k).unwrap();
        let flat: Vec<i32> = tiles.iter().flatten().copied().collect();
        let d_tiles = upload_i32(gpu, &flat);
        let d_rows = upload_i32(gpu, rows);
        let n = rows.len();
        let y = gpu
            .upload_f32(&vec![POISON; n * self.ex.r], &[n * self.ex.r])
            .expect("y");
        let x = if bf16_x { self.x_bf16 } else { self.x_f32 };
        let mut dev = gpu.try_gfx942().expect("exact gfx942");
        let res = if reference {
            dev.gemm_fp8b128_moe_grouped_bf16mfma_reference(
                self.d_payload,
                stride,
                self.ex.n,
                &d_tiles,
                tiles.len(),
                &d_rows,
                x,
                self.x_row_div,
                self.x_rows,
                &y,
                n,
                self.ex.r,
                self.ex.k,
            )
        } else {
            dev.gemm_fp8b128_moe_grouped_bf16mfma(
                self.d_payload,
                stride,
                self.ex.n,
                &d_tiles,
                tiles.len(),
                &d_rows,
                x,
                self.x_row_div,
                self.x_rows,
                &y,
                n,
                self.ex.r,
                self.ex.k,
            )
        };
        res.unwrap_or_else(|e| {
            panic!(
                "{}: {e:?}",
                if reference {
                    "gemm_fp8b128_moe_grouped_bf16mfma_reference"
                } else {
                    "gemm_fp8b128_moe_grouped_bf16mfma"
                }
            )
        });
        gpu.hip.device_synchronize().expect("sync");
        let out = gpu.download_f32(&y).expect("download y");
        for t in [y, d_rows, d_tiles] {
            gpu.free_tensor(t).ok();
        }
        out
    }

    /// Run one grouped launch over `rows` (sorted row ids); returns the
    /// candidate's Y. In `--byte-identity` mode the retained P1 baseline runs
    /// the same launch for BOTH X dtypes and the candidate's bytes must equal it
    /// exactly (whole Y, sentinel rows included).
    fn run(&self, gpu: &mut Gpu, tiles: &[Tile], rows: &[i32], bf16_x: bool) -> Vec<f32> {
        let out = self.raw(gpu, tiles, rows, bf16_x, false);
        if let Some(ab) = self.ab {
            let base = self.raw(gpu, tiles, rows, bf16_x, true);
            ab.compare(
                self.tag,
                if bf16_x { "bf16-x" } else { "f32-x" },
                &base,
                &out,
                self.ex.r,
            );
            drop(base);
            let other = self.raw(gpu, tiles, rows, !bf16_x, false);
            let other_base = self.raw(gpu, tiles, rows, !bf16_x, true);
            ab.compare(
                self.tag,
                if bf16_x { "f32-x" } else { "bf16-x" },
                &other_base,
                &other,
                self.ex.r,
            );
        }
        out
    }

    /// One row `flat` alone: a single tile with nrows = 1.
    fn single(&self, gpu: &mut Gpu, expert: i32, flat: i32) -> Vec<f32> {
        self.run(gpu, &[[expert, 0, 1, 0]], &[flat], false)
    }
}

/// Group `slot_expert[flat] = e` into tiles, expert ascending, flat ascending.
/// `padded`: MQ4 layout (every tile 16 rows, -1 padding); otherwise compact
/// (an expert's last tile has nrows < 16, no padding rows).
fn scatter(slot_expert: &[usize], n_exp: usize, padded: bool) -> (Vec<Tile>, Vec<i32>) {
    let mut tiles = Vec::new();
    let mut rows = Vec::new();
    for e in 0..n_exp {
        let mine: Vec<i32> = (0..slot_expert.len())
            .filter(|&i| slot_expert[i] == e)
            .map(|i| i as i32)
            .collect();
        for chunk in mine.chunks(16) {
            let row0 = rows.len() as i32;
            rows.extend_from_slice(chunk);
            let nrows = if padded {
                rows.extend(std::iter::repeat_n(-1, 16 - chunk.len()));
                16
            } else {
                chunk.len() as i32
            };
            tiles.push([e as i32, row0, nrows, 0]);
        }
    }
    (tiles, rows)
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// ── Byte A/B (`--byte-identity`) ─────────────────────────────────────────

/// Raw-bit comparison of baseline vs candidate Y over a whole case.
struct Ab {
    launches: Cell<usize>,
    values: Cell<u64>,
    mismatched: Cell<u64>,
    first: RefCell<Vec<String>>,
}

impl Ab {
    fn new() -> Self {
        Ab {
            launches: Cell::new(0),
            values: Cell::new(0),
            mismatched: Cell::new(0),
            first: RefCell::new(Vec::new()),
        }
    }

    fn compare(&self, tag: &str, what: &str, base: &[f32], cand: &[f32], r: usize) {
        self.launches.set(self.launches.get() + 1);
        self.values
            .set(self.values.get() + base.len().max(cand.len()) as u64);
        let mut bad = base.len().abs_diff(cand.len()) as u64;
        let mut first = None;
        for (i, (b, c)) in base.iter().zip(cand).enumerate() {
            if b.to_bits() != c.to_bits() {
                bad += 1;
                first.get_or_insert((i, b.to_bits(), c.to_bits()));
            }
        }
        if bad > 0 {
            self.mismatched.set(self.mismatched.get() + bad);
            let mut f = self.first.borrow_mut();
            if f.len() < 8 {
                f.push(match first {
                    Some((i, b, c)) => format!(
                        "{tag} [{what}] row {} col {}: baseline {b:#010x} candidate {c:#010x} ({bad} mismatched)",
                        i / r,
                        i % r
                    ),
                    None => format!("{tag} [{what}]: length baseline {} candidate {}", base.len(), cand.len()),
                });
            }
        }
    }

    fn check(&self, fail: &mut Fail, name: &str) {
        let mm = self.mismatched.get();
        fail.check(
            mm == 0,
            format!(
                "{name}: byte A/B retained P1 baseline == candidate, raw F32 bits incl. padding/skipped/guard sentinel rows, both X dtypes ({} compared launches, {} values, {mm} mismatched)",
                self.launches.get(),
                self.values.get()
            ),
        );
        for line in self.first.borrow().iter() {
            println!("  first mismatch: {line}");
        }
    }
}

impl Experts {
    /// Byte-A/B-only pool: no CPU-decoded W (the F64 oracle is not run on it),
    /// so 512-expert real-shape pools stay cheap. `make(e)` -> (codes, scales).
    fn ab_only(r: usize, k: usize, n: usize, make: impl Fn(usize) -> (Vec<u8>, Vec<u16>)) -> Self {
        let mut payload = Vec::with_capacity(n * b128_block_stride(r, k).unwrap());
        for e in 0..n {
            let (codes, scales) = make(e);
            assert_eq!(
                (codes.len(), scales.len()),
                (r * k, b128_scale_count(r, k).unwrap())
            );
            payload.extend(block_bytes(&codes, &scales, r, k));
        }
        Experts {
            r,
            k,
            n,
            payload,
            w: Vec::new(),
        }
    }
}

/// Every finite E4M3 code is `b & 0x7F != 0x7F`; map the two NaNs to +/-0.
fn finite_code(c: usize) -> u8 {
    let b = (c % 256) as u8;
    if b & 0x7F == 0x7F {
        b & 0x80
    } else {
        b
    }
}

/// Four random finite codes per LCG draw (fast enough for 512 x 3.3 MB).
fn rand_codes4(n: usize, st: &mut u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(n + 4);
    while v.len() < n {
        for b in lcg(st).to_le_bytes() {
            v.push(if b & 0x7F == 0x7F { b ^ 1 } else { b });
        }
    }
    v.truncate(n);
    v
}

fn expert_seed(seed: u64, e: usize) -> u64 {
    seed ^ (e as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Block scales that stress the decode: 1.0 and neighbours, checkpoint-range,
/// large, negative, BF16-subnormal / min-normal / +0 / -0 (decoded W is +/-0 or
/// BF16-subnormal-underflowed). All finite; none can overflow F32 with the X
/// generators below.
const SPECIAL_SCALES: [u16; 16] = [
    0x3F80, 0x3F81, 0x3FFF, 0x3C00, 0x3800, 0x2F80, 0x3B7F, 0x4321, 0xBF80, 0xBA83, 0x0D80, 0x0001,
    0x0080, 0x0000, 0x8000, 0x3A83,
];

/// Finite-FP8 stress codes per expert kind: +0, -0, subnormals with sign, +/-max
/// magnitude by K16 block, every finite code cycling, random.
fn special_codes(e: usize, r: usize, k: usize) -> Vec<u8> {
    let mut st = expert_seed(0xc0de_0000, e);
    (0..r * k)
        .map(|i| {
            let (o, kk) = (i / k, i % k);
            match e % 6 {
                0 => 0x00,
                1 => 0x80,
                2 => {
                    let m = (kk % 7 + 1) as u8; // 0x01..0x07: the E4M3 subnormals
                    if (kk / 7 + o) & 1 == 1 {
                        0x80 | m
                    } else {
                        m
                    }
                }
                3 => {
                    if (kk / 16 + o) & 1 == 1 {
                        0xFE
                    } else {
                        0x7E
                    }
                }
                4 => finite_code(kk + o * 17 + e * 5),
                _ => finite_code((lcg(&mut st) >> 8) as usize),
            }
        })
        .collect()
}

fn special_scales(e: usize, r: usize, k: usize) -> Vec<u16> {
    (0..b128_scale_count(r, k).unwrap())
        .map(|i| SPECIAL_SCALES[(i * 5 + e * 3) % SPECIAL_SCALES.len()])
        .collect()
}

/// Cancellation-heavy W: every scale equal and the codes mirrored (kind e%3:
/// half-shift w[k+K/2] = w[k]; reverse w[K-1-k] = w[k]; both), paired with
/// `cancel_x_row` (antisymmetric under both), so the exact sum is 0 and the
/// F32 result is purely accumulation-order residue.
fn cancel_codes(e: usize, r: usize, k: usize) -> Vec<u8> {
    let seed = expert_seed(0xca9c_e100, e);
    (0..r * k)
        .map(|i| {
            let (o, kk) = (i / k, i % k);
            let c = match e % 3 {
                0 => kk % (k / 2),
                1 => kk.min(k - 1 - kk),
                _ => {
                    let h = kk % (k / 2);
                    h.min(k / 2 - 1 - h)
                }
            };
            let mut s = seed ^ ((o * k + c) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            lcg(&mut s);
            finite_code((lcg(&mut s) >> 8) as usize)
        })
        .collect()
}

fn cancel_scales(e: usize, r: usize, k: usize) -> Vec<u16> {
    vec![[0x3C00u16, 0x3A80, 0x3F80][(e / 3) % 3]; b128_scale_count(r, k).unwrap()]
}

/// BF16 with a wide exponent (2^-30 .. 2^30), random sign/mantissa.
fn wide_bf16(st: &mut u64) -> u16 {
    let (s, e, m) = (lcg(st) & 1, 97 + lcg(st) % 61, lcg(st) & 0x7F);
    ((s << 15) | (e << 7) | m) as u16
}

/// Exactly antisymmetric X row (x[k+K/2] = -x[k], first half palindromic so
/// x[K-1-k] = -x[k]) with wide-exponent entries; odd rows get one 1-ulp bump so
/// the residue is a large-partial-sum rounding result rather than 0.
fn cancel_x_row(k: usize, row: usize, st: &mut u64) -> Vec<u16> {
    let v: Vec<u16> = (0..k / 4).map(|_| wide_bf16(st)).collect();
    let half: Vec<u16> = v.iter().copied().chain(v.iter().rev().copied()).collect();
    let mut x = half.clone();
    x.extend(half.iter().map(|b| b ^ 0x8000));
    if row % 2 == 1 {
        x[(row * 7 + 3) % k] ^= 1;
    }
    x
}

/// X rows of BF16 finite stress kinds: +0, -0, subnormals, huge/tiny
/// alternating, wide random, one-hot (position moves with the row), +/-0
/// interleaved, K16-block sign flips, plain random.
fn special_x_row(k: usize, row: usize, st: &mut u64) -> Vec<u16> {
    let sgn = |st: &mut u64| ((lcg(st) & 1) << 15) as u16;
    match row % 9 {
        0 => vec![0x0000; k],
        1 => vec![0x8000; k],
        2 => (0..k)
            .map(|_| sgn(st) | ((lcg(st) % 0x7F) as u16 + 1))
            .collect(),
        3 => (0..k)
            .map(|i| sgn(st) | if i % 2 == 0 { 0x4E80 } else { 0x3080 })
            .collect(),
        4 => (0..k).map(|_| wide_bf16(st)).collect(),
        5 => {
            let hot = (row * 37 + 11) % k;
            (0..k)
                .map(|i| {
                    if i == hot {
                        0x3F80 | sgn(st)
                    } else if i % 2 == 0 {
                        0x0000
                    } else {
                        0x8000
                    }
                })
                .collect()
        }
        6 => (0..k)
            .map(|i| {
                if i % 3 == 0 {
                    sgn(st)
                } else {
                    f32_to_bf16(((lcg(st) as f32 / u32::MAX as f32) - 0.5) * 4.0)
                }
            })
            .collect(),
        7 => (0..k)
            .map(|i| if (i / 16) % 2 == 0 { 0x3F80 } else { 0xBF80 })
            .collect(),
        _ => (0..k)
            .map(|_| f32_to_bf16(((lcg(st) as f32 / u32::MAX as f32) - 0.5) * 4.0))
            .collect(),
    }
}

/// Plain BF16-exact random X (signed zeros sprinkled) for the target routes.
fn random_x(x_rows: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut st = seed;
    (0..x_rows * k)
        .map(|i| match i % 97 {
            0 => 0x0000,
            1 => 0x8000,
            _ => f32_to_bf16(((lcg(&mut st) as f32 / u32::MAX as f32) - 0.5) * 4.0),
        })
        .collect()
}

/// Top-10 distinct experts per token. Skewed (cubic, Zipf-like); with
/// `uniform_mix` every other token is uniform so tiny pools reach all experts.
fn zipf_slots(tokens: usize, n_exp: usize, st: &mut u64, uniform_mix: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(tokens * 10);
    for t in 0..tokens {
        let mut chosen: Vec<usize> = Vec::with_capacity(10);
        while chosen.len() < 10 {
            let u = lcg(st) as f64 / 4294967296.0;
            let f = if uniform_mix && t % 2 == 0 {
                u
            } else {
                u * u * u
            };
            let e = ((f * n_exp as f64) as usize).min(n_exp - 1);
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        out.extend(chosen);
    }
    out
}

/// Slots whose M=1 launch is also A/B'd and compared to the grouped bytes:
/// first/last few plus an even stride through the route (tile and expert
/// boundaries included by the stride on large routes, every slot on small ones).
fn sample_flats(n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).step_by((n / 48).max(1)).collect();
    v.extend(0..4.min(n));
    v.extend(n.saturating_sub(4)..n);
    v.sort_unstable();
    v.dedup();
    v
}

fn upload_x(gpu: &mut Gpu, xb: &[u16]) -> (GpuTensor, GpuTensor) {
    let xf: Vec<f32> = xb.iter().map(|&b| bf16_to_f32(b)).collect();
    let x_f32 = gpu.upload_f32(&xf, &[xb.len()]).expect("x f32");
    let mut x_bf16 = upload_bytes(
        gpu,
        &xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    );
    x_bf16.shape = vec![xb.len()];
    x_bf16.dtype = DType::BF16;
    (x_f32, x_bf16)
}

/// Byte A/B of one route over an arbitrary expert pool / X (no F64 oracle):
/// padded layout + a skipped tile, compact layout + guard rows, and M=1
/// launches of sampled slots. Every launch is A/B'd against the retained P1
/// baseline for both X dtypes; on the candidate the padding/skipped/guard
/// sentinels, compact == padded and M=1 == grouped bytes are also checked.
fn ab_route(
    gpu: &mut Gpu,
    fail: &mut Fail,
    name: &str,
    ex: &Experts,
    xb: &[u16],
    x_row_div: usize,
    slot_expert: &[usize],
) {
    let (r, k) = (ex.r, ex.k);
    assert!(
        xb.len() % k == 0
            && !slot_expert.is_empty()
            && (slot_expert.len() - 1) / x_row_div < xb.len() / k,
        "{name}: route/X geometry"
    );
    let x_rows = xb.len() / k;
    let d_payload = upload_bytes(gpu, &ex.payload);
    let (x_f32, x_bf16) = upload_x(gpu, xb);
    let ab = Ab::new();
    let l = Launch {
        ex,
        d_payload: &d_payload,
        x_row_div,
        x_rows,
        x_f32: &x_f32,
        x_bf16: &x_bf16,
        ab: Some(&ab),
        tag: name,
    };

    let live = slot_expert.len();
    let (mut tiles, mut rows) = scatter(slot_expert, ex.n, true);
    let live_rows = rows.len();
    tiles.push([-1, live_rows as i32, 16, 0]);
    rows.extend(std::iter::repeat_n(0, 16));
    let y = l.run(gpu, &tiles, &rows, false);

    let mut flat_g = vec![usize::MAX; live];
    let (mut pad_ok, mut skip_ok) = (true, true);
    for (g, &flat) in rows[..live_rows].iter().enumerate() {
        let yrow = &y[g * r..(g + 1) * r];
        if flat < 0 {
            pad_ok &= yrow.iter().all(|v| v.to_bits() == 0);
        } else {
            flat_g[flat as usize] = g;
        }
    }
    skip_ok &= y[live_rows * r..]
        .iter()
        .all(|v| v.to_bits() == POISON.to_bits());
    let row_of = |y: &[f32], g: usize| -> Vec<f32> { y[g * r..(g + 1) * r].to_vec() };

    let (ct, cr) = scatter(slot_expert, ex.n, false);
    let mut cr_guard = cr.clone();
    cr_guard.extend(std::iter::repeat_n(0, 16));
    let yc = l.run(gpu, &ct, &cr_guard, true);
    let compact_ok = cr.iter().enumerate().all(|(g, &flat)| {
        bits_eq(
            &yc[g * r..(g + 1) * r],
            &y[flat_g[flat as usize] * r..][..r],
        )
    });
    let guard_ok = yc[cr.len() * r..]
        .iter()
        .all(|v| v.to_bits() == POISON.to_bits());

    let sample = sample_flats(live);
    let mut bad_single = 0usize;
    for &flat in &sample {
        let y1 = l.single(gpu, slot_expert[flat] as i32, flat as i32);
        bad_single += !bits_eq(&y1, &row_of(&y, flat_g[flat])) as usize;
    }
    drop(y);
    drop(yc);

    ab.check(fail, name);
    fail.check(
        pad_ok && skip_ok && guard_ok,
        format!("{name}: candidate padding rows +0, skipped tile and guard rows untouched"),
    );
    fail.check(
        compact_ok,
        format!("{name}: candidate compact tiles byte-identical to padded layout ({live} slots)"),
    );
    fail.check(bad_single == 0, format!("{name}: candidate sampled M=1 launches byte-identical to grouped rows ({} sampled, {bad_single} mismatched)", sample.len()));
    for t in [x_bf16, x_f32, d_payload] {
        gpu.free_tensor(t).ok();
    }
}

/// Mandatory `--byte-identity` additions (everything else is the full existing
/// suite with A/B enabled on every grouped launch).
fn byte_cases(gpu: &mut Gpu, fail: &mut Fail) {
    // (a) R in {16,48,80} (output-wave tails) x K in {128,256,384} (K=256: exactly
    // one full D=2 prefetch group; 128/384: odd/tail groups) through the FULL
    // existing case (F64 oracle, both X dtypes, padded/compact/skipped tiles,
    // M-invariance, tile positions).
    for (r, k) in [
        (16usize, 128usize),
        (16, 256),
        (16, 384),
        (48, 128),
        (48, 256),
        (48, 384),
        (80, 128),
        (80, 256),
        (80, 384),
    ] {
        let ex = Experts::synthetic(r, k, 12, 0x7a11_0000 ^ (r * 1000 + k) as u64);
        for (div, tokens) in [(10usize, 37usize), (1, 13)] {
            gemm_case(
                gpu,
                fail,
                &format!("tail r={r} k={k} x_row_div={div}"),
                &ex,
                div,
                tokens,
                true,
            );
        }
    }

    let shapes = [
        (16usize, 128usize, 10usize),
        (16, 256, 10),
        (48, 128, 1),
        (48, 256, 1),
        (80, 256, 10),
        (80, 384, 10),
        (80, 384, 1),
        (1280, 2560, 10),
        (2560, 640, 1),
    ];
    let tokens = 37usize;
    for (r, k, div) in shapes {
        let x_rows = if div == 1 { tokens * 10 } else { tokens };
        let mut st = 0x57ec_1a10_u64 ^ (r * 7919 + k) as u64;
        let slots = zipf_slots(tokens, 12, &mut st, true);

        // (b) finite-FP8 +/-0/subnormal/max/sign codes x extreme scales; (c) BF16 X stress.
        let ex = Experts::ab_only(r, k, 12, |e| {
            (special_codes(e, r, k), special_scales(e, r, k))
        });
        let xb: Vec<u16> = (0..x_rows)
            .flat_map(|row| special_x_row(k, row, &mut st))
            .collect();
        ab_route(
            gpu,
            fail,
            &format!("fp8-special r={r} k={k} x_row_div={div}"),
            &ex,
            &xb,
            div,
            &slots,
        );

        // BF16 X stress against ordinary random weights.
        let rnd = Experts::ab_only(r, k, 12, |e| {
            let mut s = expert_seed(0x4a4d_0000, e);
            let codes = rand_codes4(r * k, &mut s);
            (
                codes,
                (0..b128_scale_count(r, k).unwrap())
                    .map(|_| rand_scale(&mut s))
                    .collect(),
            )
        });
        ab_route(
            gpu,
            fail,
            &format!("bf16-x-special r={r} k={k} x_row_div={div}"),
            &rnd,
            &xb,
            div,
            &slots,
        );
        drop(rnd);
        drop(ex);

        // Cancellation-heavy: mirrored W against antisymmetric wide-exponent X.
        let canc = Experts::ab_only(r, k, 12, |e| {
            (cancel_codes(e, r, k), cancel_scales(e, r, k))
        });
        let xc: Vec<u16> = (0..x_rows)
            .flat_map(|row| cancel_x_row(k, row, &mut st))
            .collect();
        ab_route(
            gpu,
            fail,
            &format!("cancellation r={r} k={k} x_row_div={div}"),
            &canc,
            &xc,
            div,
            &slots,
        );
    }

    // (d) Target routes at the real shapes: C1 (one token, 10 distinct
    // experts, 10 one-row tiles) and 16/512/2048-token Zipf-like top-10 over
    // 512 experts. Baseline vs candidate on every live output; no F64 oracle.
    for (label, r, k, div) in [
        ("gate_up", 1280usize, 2560usize, 10usize),
        ("down", 2560, 640, 1),
    ] {
        let ex = Experts::ab_only(r, k, 512, |e| {
            let mut s = expert_seed(0x7a46_e700 ^ r as u64, e);
            let codes = rand_codes4(r * k, &mut s);
            (
                codes,
                (0..b128_scale_count(r, k).unwrap())
                    .map(|_| rand_scale(&mut s))
                    .collect(),
            )
        });
        for tokens in [1usize, 16, 512, 2048] {
            let mut st = 0x7a46_0000_u64 ^ tokens as u64 ^ (r as u64) << 24;
            let slots = if tokens == 1 {
                (0..10).map(|j| j * 51 + 7).collect()
            } else {
                zipf_slots(tokens, 512, &mut st, false)
            };
            let x_rows = if div == 1 { slots.len() } else { tokens };
            let xb = random_x(x_rows, k, 0x7a46_5eed ^ tokens as u64);
            let tag = if tokens == 1 {
                "C1 (10 experts)".to_string()
            } else {
                format!("{tokens}-token route")
            };
            ab_route(
                gpu,
                fail,
                &format!("target {label} r={r} k={k} {tag}"),
                &ex,
                &xb,
                div,
                &slots,
            );
        }
    }
}

fn gemm_case(
    gpu: &mut Gpu,
    fail: &mut Fail,
    name: &str,
    ex: &Experts,
    x_row_div: usize,
    tokens: usize,
    byte_ab: bool,
) {
    let (r, k) = (ex.r, ex.k);
    // x: BF16-exact values, including signed zeros. At least 128 rows so the
    // M-invariance sweep can reach n = 128.
    let x_rows = (if x_row_div == 1 { tokens * 10 } else { tokens }).max(128);
    let mut st = 0x5eed_0f00_u64 ^ (r as u64) << 20;
    let xb: Vec<u16> = (0..x_rows * k)
        .map(|i| match i % 97 {
            0 => 0x0000,
            1 => 0x8000,
            _ => f32_to_bf16(((lcg(&mut st) as f32 / u32::MAX as f32) - 0.5) * 4.0),
        })
        .collect();
    let xf: Vec<f32> = xb.iter().map(|&b| bf16_to_f32(b)).collect();
    let d_payload = upload_bytes(gpu, &ex.payload);
    let x_f32 = gpu.upload_f32(&xf, &[x_rows * k]).expect("x f32");
    let mut x_bf16 = upload_bytes(
        gpu,
        &xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    );
    x_bf16.shape = vec![x_rows * k];
    x_bf16.dtype = DType::BF16;
    let ab = byte_ab.then(Ab::new);
    let l = Launch {
        ex,
        d_payload: &d_payload,
        x_row_div,
        x_rows,
        x_f32: &x_f32,
        x_bf16: &x_bf16,
        ab: ab.as_ref(),
        tag: name,
    };

    // Top-10 distinct experts per token (deterministic, skewed).
    let mut slot_expert = Vec::with_capacity(tokens * 10);
    for t in 0..tokens {
        let mut chosen: Vec<usize> = Vec::new();
        while chosen.len() < 10 {
            let e = (lcg(&mut st) as usize % (ex.n * 2)).min(ex.n - 1 - (t % 2));
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        slot_expert.extend(chosen);
    }
    let live = slot_expert.len();

    // Padded (MQ4) layout + one skipped tile (expert -1) over 16 extra rows that
    // must stay POISON.
    let (mut tiles, mut rows) = scatter(&slot_expert, ex.n, true);
    tiles.push([-1, rows.len() as i32, 16, 0]);
    rows.extend(std::iter::repeat_n(0, 16));
    let y = l.run(gpu, &tiles, &rows, false);
    let yb = l.run(gpu, &tiles, &rows, true);
    fail.check(
        bits_eq(&y, &yb),
        format!(
            "{name}: F32-x and BF16-x entries byte-identical ({} values)",
            y.len()
        ),
    );

    // GPU tile-desc builder from padded-16 tile ids == the host descriptors.
    {
        let ids: Vec<i32> = tiles.iter().map(|t| t[0]).collect();
        let d_ids = upload_i32(gpu, &ids);
        let d_desc = upload_bytes(gpu, &vec![0xEEu8; ids.len() * 16]);
        gpu.try_gfx942()
            .expect("exact gfx942")
            .fp8b128_moe_tile_desc_pad16(&d_ids, &d_desc, ids.len())
            .expect("fp8b128_moe_tile_desc_pad16");
        gpu.hip.device_synchronize().expect("sync");
        let got: Vec<i32> = gpu
            .download_raw_bytes(&d_desc)
            .expect("download desc")
            .chunks_exact(4)
            .map(|p| i32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect();
        let want: Vec<i32> = tiles.iter().flatten().copied().collect();
        fail.check(
            got == want,
            format!(
                "{name}: fp8b128_moe_tile_desc_pad16 == host descriptors ({} tiles)",
                ids.len()
            ),
        );
        for t in [d_desc, d_ids] {
            gpu.free_tensor(t).ok();
        }
    }

    let bound_k = k as f64 * (2f64).powi(-24);
    let (mut worst, mut worst_abs, mut maxref) = (0f64, 0f64, 0f64);
    let (mut bad, mut pad_ok, mut skip_ok) = (0usize, true, true);
    // flat slot -> Y row bytes of the padded launch, for the layout checks below.
    let mut by_flat: Vec<Option<Vec<f32>>> = vec![None; live];
    for t in &tiles {
        for i in 0..16usize {
            let g = t[1] as usize + i;
            let yrow = &y[g * r..(g + 1) * r];
            if t[0] < 0 {
                skip_ok &= yrow.iter().all(|v| v.to_bits() == POISON.to_bits());
                continue;
            }
            let flat = rows[g];
            if flat < 0 {
                pad_ok &= yrow.iter().all(|v| v.to_bits() == 0);
                continue;
            }
            by_flat[flat as usize] = Some(yrow.to_vec());
            let xr = &xf[(flat as usize / x_row_div) * k..][..k];
            let w = &ex.w[t[0] as usize];
            for o in 0..r {
                let (mut acc, mut aa) = (0f64, 0f64);
                for (wk, xk) in w[o * k..(o + 1) * k].iter().zip(xr) {
                    let p = bf16_to_f32(*wk) as f64 * *xk as f64;
                    acc += p;
                    aa += p.abs();
                }
                let err = (yrow[o] as f64 - acc).abs();
                maxref = maxref.max(acc.abs());
                worst_abs = worst_abs.max(err);
                if aa > 0.0 {
                    worst = worst.max(err / aa);
                }
                bad += (err > bound_k * aa) as usize;
            }
        }
    }
    fail.check(
        bad == 0,
        format!(
            "{name}: grouped GEMM vs F64 ref, {live} live rows, max err/sum|wx| = {worst:.3e} (bound {bound_k:.3e}), max err/max|ref| = {:.3e}, violations {bad}",
            worst_abs / maxref.max(1e-30)
        ),
    );
    fail.check(
        pad_ok,
        format!("{name}: padding rows (-1) of live tiles written as +0"),
    );
    fail.check(
        skip_ok,
        format!("{name}: skipped tile (expert -1) left untouched"),
    );

    // Compact layout (nrows < 16 tails, no padding): same bytes per flat slot,
    // and nothing beyond the live rows written.
    let (ct, cr) = scatter(&slot_expert, ex.n, false);
    let mut cr_guard = cr.clone();
    cr_guard.extend(std::iter::repeat_n(0, 16)); // guard rows no tile covers
    let yc = l.run(gpu, &ct, &cr_guard, false);
    let compact_ok = cr.iter().enumerate().all(|(g, &flat)| {
        bits_eq(
            &yc[g * r..(g + 1) * r],
            by_flat[flat as usize].as_ref().unwrap(),
        )
    });
    let guard_ok = yc[cr.len() * r..]
        .iter()
        .all(|v| v.to_bits() == POISON.to_bits());
    fail.check(compact_ok && guard_ok, format!("{name}: compact tiles (nrows<16) byte-identical to padded layout, rows past the tiles untouched"));

    // Every grouped row == its single-row launch (same expert, same X row).
    let mut mismatched = 0usize;
    for (flat, e) in slot_expert.iter().enumerate() {
        let y1 = l.single(gpu, *e as i32, flat as i32);
        mismatched += !bits_eq(&y1, by_flat[flat].as_ref().unwrap()) as usize;
    }
    fail.check(mismatched == 0, format!("{name}: every grouped row byte-identical to its M=1 launch ({live} rows, {mismatched} mismatched)"));

    // bytes(M=1) == row of bytes(M=n), every row, single expert 0, x_row_div 1 view.
    let l1 = Launch { x_row_div: 1, ..l };
    let singles: Vec<Vec<f32>> = (0..128).map(|row| l1.single(gpu, 0, row)).collect();
    for n in [2usize, 7, 8, 16, 17, 128] {
        let (t, s) = scatter(&vec![0usize; n], 1, true);
        let yn = l1.run(gpu, &t, &s, false);
        let bad_rows = (0..n)
            .filter(|&row| !bits_eq(&yn[row * r..(row + 1) * r], &singles[row]))
            .count();
        fail.check(bad_rows == 0, format!("{name}: bytes(M=1) == row of bytes(M={n}) for all {n} rows ({bad_rows} mismatched)"));
    }

    // Tile-position independence: X row q placed at every position p of a full
    // 16-row tile (other positions hold other rows), with the tile at index 0 and
    // at index 5 of a launch; every placement must reproduce bytes(M=1).
    let mut bad_pos = 0usize;
    let checked_rows = [0i32, 37, 127];
    for &q in &checked_rows {
        for p in 0..16usize {
            let mut tile_rows: Vec<i32> = (0..16)
                .map(|i| ((q as usize + 1 + i * 7) % 128) as i32)
                .collect();
            tile_rows[p] = q;
            for tile_index in [0usize, 5] {
                let mut rows_l = Vec::new();
                let mut tiles_l = Vec::new();
                for t in 0..=tile_index {
                    tiles_l.push([0, rows_l.len() as i32, 16, 0]);
                    if t == tile_index {
                        rows_l.extend_from_slice(&tile_rows);
                    } else {
                        rows_l.extend((0..16).map(|i| ((t * 16 + i) % 128) as i32));
                    }
                }
                let yy = l1.run(gpu, &tiles_l, &rows_l, false);
                let g = tile_index * 16 + p;
                bad_pos += !bits_eq(&yy[g * r..(g + 1) * r], &singles[q as usize]) as usize;
            }
        }
    }
    fail.check(
        bad_pos == 0,
        format!("{name}: tile-position independence, rows {checked_rows:?} x 16 positions x tile index {{0,5}} == bytes(M=1) ({bad_pos} mismatched)"),
    );

    if let Some(ab) = &ab {
        ab.check(fail, name);
    }

    for t in [x_bf16, x_f32, d_payload] {
        gpu.free_tensor(t).ok();
    }
}

fn read(dir: &str, f: &str) -> Vec<u8> {
    std::fs::read(format!("{dir}/{f}")).unwrap_or_else(|e| panic!("read {dir}/{f}: {e}"))
}

// ── Timing (`--time`) ────────────────────────────────────────────────────

const TIME_WARMUPS: usize = 5;
const TIME_LAUNCHES: usize = 60;
const TIME_CACHE_POLICY: &str = "warm: no L2/MALL flush or eviction between launches; every launch re-reads the same \
                                 weight/X/descriptor buffers; reference and candidate alternate (order swapped every \
                                 iteration) over identical buffers";

/// Device buffers of one timed route, allocated once and reused by every launch.
struct TimedRoute<'a> {
    r: usize,
    k: usize,
    n_exp: usize,
    stride: usize,
    d_payload: &'a GpuTensor,
    d_tiles: &'a GpuTensor,
    n_tiles: usize,
    d_rows: &'a GpuTensor,
    n_sorted: usize,
    x_f32: &'a GpuTensor,
    x_bf16: &'a GpuTensor,
    x_row_div: usize,
    x_rows: usize,
}

/// One launch into caller-owned `y`: no allocation, sync or download.
fn timed_launch(gpu: &mut Gpu, rt: &TimedRoute, bf16_x: bool, reference: bool, y: &GpuTensor) {
    let x = if bf16_x { rt.x_bf16 } else { rt.x_f32 };
    let mut dev = gpu.try_gfx942().expect("exact gfx942");
    let res = if reference {
        dev.gemm_fp8b128_moe_grouped_bf16mfma_reference(
            rt.d_payload,
            rt.stride,
            rt.n_exp,
            rt.d_tiles,
            rt.n_tiles,
            rt.d_rows,
            x,
            rt.x_row_div,
            rt.x_rows,
            y,
            rt.n_sorted,
            rt.r,
            rt.k,
        )
    } else {
        dev.gemm_fp8b128_moe_grouped_bf16mfma(
            rt.d_payload,
            rt.stride,
            rt.n_exp,
            rt.d_tiles,
            rt.n_tiles,
            rt.d_rows,
            x,
            rt.x_row_div,
            rt.x_rows,
            y,
            rt.n_sorted,
            rt.r,
            rt.k,
        )
    };
    res.unwrap_or_else(|e| panic!("timed grouped launch (reference={reference}): {e:?}"));
}

/// Event-timed span (microseconds) of `n` back-to-back runs of `f`, synchronized
/// on the stop event only.
fn event_span_us(
    gpu: &mut Gpu,
    e0: &hip_bridge::Event,
    e1: &hip_bridge::Event,
    n: usize,
    mut f: impl FnMut(&mut Gpu),
) -> f64 {
    gpu.hip
        .event_record(e0, gpu.active_stream.as_ref())
        .expect("record start");
    for _ in 0..n {
        f(gpu);
    }
    gpu.hip
        .event_record(e1, gpu.active_stream.as_ref())
        .expect("record stop");
    gpu.hip.event_synchronize(e1).expect("sync stop");
    gpu.hip.event_elapsed_ms(e0, e1).expect("elapsed") as f64 * 1000.0
}

fn median_us(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn samples_json(samples: &[f64], burst_mean_us: f64, useful_flops: f64) -> serde_json::Value {
    let med = median_us(samples);
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    serde_json::json!({
        "median_us": med,
        "min_us": min,
        "max_us": samples.iter().copied().fold(0.0, f64::max),
        "useful_tflops_at_median": useful_flops / (med * 1e-6) / 1e12,
        "burst_mean_us": burst_mean_us,
        "useful_tflops_at_burst_mean": useful_flops / (burst_mean_us * 1e-6) / 1e12,
        "samples_us": samples,
    })
}

fn sha_hex(parts: &[&[u8]]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    format!("{:x}", h.finalize())
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn identity_json(id: &rdna_compute::cdna::gfx942::Fp8MoeBuildIdentity) -> serde_json::Value {
    serde_json::json!({
        "module": id.module,
        "source_sha256": id.source_sha256,
        "jit_cache_key": id.jit_cache_key,
        "object_sha256": id.object_sha256,
        "object_path": id.object_path,
        "compiler": id.toolchain_id,
    })
}

/// `--time [--time-out PATH] [--time-tokens 1,16,512,2048] [--time-shapes gate_up,down]
/// [--time-xdt f32|bf16|both]`: reference-vs-production kernel timing at the real
/// shapes (gate_up r=1280 k=2560 x_row_div=10, down r=2560 k=640 x_row_div=1),
/// synthetic top-10-of-512 routes (the byte gate's routes: tokens=1 is C1, ten
/// distinct experts), padded-16 tiles. Per (X dtype, shape, tokens) the
/// reference and the production entry run 5 warmups each, then 60 event-timed
/// single launches alternating (order swapped each iteration) into reused
/// buffers, plus one back-to-back burst of 60 per side (launch-gap-free mean).
/// After timing, the WHOLE Y of both sides is downloaded and compared on raw
/// F32 bits; any mismatch fails the run (exit 1) after the JSON is written.
fn time_mode(gpu: &mut Gpu, args: &[String]) {
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .map(|i| args[i + 1].clone())
    };
    let out_path = opt("--time-out").unwrap_or_else(|| "fp8_moe_time.json".to_string());
    let tokens_sel: Vec<usize> = opt("--time-tokens")
        .map(|s| {
            s.split(',')
                .map(|t| t.trim().parse().expect("--time-tokens"))
                .collect()
        })
        .unwrap_or_else(|| vec![1, 16, 512, 2048]);
    let shapes_sel = opt("--time-shapes").unwrap_or_else(|| "gate_up,down".to_string());
    let xdt = opt("--time-xdt").unwrap_or_else(|| "both".to_string());
    let dtypes: Vec<bool> = match xdt.as_str() {
        "f32" => vec![false],
        "bf16" => vec![true],
        "both" => vec![false, true],
        other => panic!("--time-xdt must be f32|bf16|both, got {other}"),
    };
    let shapes: Vec<(&str, usize, usize, usize)> = [
        ("gate_up", 1280usize, 2560usize, 10usize),
        ("down", 2560, 640, 1),
    ]
    .into_iter()
    .filter(|(l, ..)| shapes_sel.split(',').any(|s| s.trim() == *l))
    .collect();
    assert!(!shapes.is_empty(), "--time-shapes selects no shape");

    let e0 = gpu.hip.event_create().expect("start event");
    let e1 = gpu.hip.event_create().expect("stop event");
    let mut records = Vec::new();
    let mut total_mismatch = 0u64;
    let mut identities = None;

    for &(label, r, k, div) in &shapes {
        let ex = Experts::ab_only(r, k, 512, |e| {
            let mut s = expert_seed(0x7a46_e700 ^ r as u64, e);
            let codes = rand_codes4(r * k, &mut s);
            (
                codes,
                (0..b128_scale_count(r, k).unwrap())
                    .map(|_| rand_scale(&mut s))
                    .collect(),
            )
        });
        let stride = b128_block_stride(r, k).unwrap();
        let weights_digest = sha_hex(&[&ex.payload]);
        let d_payload = upload_bytes(gpu, &ex.payload);
        for &tokens in &tokens_sel {
            let mut st = 0x7a46_0000_u64 ^ tokens as u64 ^ (r as u64) << 24;
            let slots: Vec<usize> = if tokens == 1 {
                (0..10).map(|j| j * 51 + 7).collect()
            } else {
                zipf_slots(tokens, 512, &mut st, false)
            };
            let x_rows = if div == 1 { slots.len() } else { tokens };
            let xb = random_x(x_rows, k, 0x7a46_5eed ^ tokens as u64);
            let (tiles, rows) = scatter(&slots, ex.n, true);
            let flat_tiles: Vec<i32> = tiles.iter().flatten().copied().collect();
            let mut uniq: Vec<usize> = slots.clone();
            uniq.sort_unstable();
            uniq.dedup();
            let route_digest = sha_hex(&[
                &(r as u64).to_le_bytes(),
                &(k as u64).to_le_bytes(),
                &(div as u64).to_le_bytes(),
                &(ex.n as u64).to_le_bytes(),
                &i32_bytes(&flat_tiles),
                &i32_bytes(&rows),
            ]);
            let x_digest =
                sha_hex(&[&xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()]);
            let useful_flops = 2.0 * slots.len() as f64 * r as f64 * k as f64;

            let d_tiles = upload_i32(gpu, &flat_tiles);
            let d_rows = upload_i32(gpu, &rows);
            let (x_f32, x_bf16) = upload_x(gpu, &xb);
            let rt = TimedRoute {
                r,
                k,
                n_exp: ex.n,
                stride,
                d_payload: &d_payload,
                d_tiles: &d_tiles,
                n_tiles: tiles.len(),
                d_rows: &d_rows,
                n_sorted: rows.len(),
                x_f32: &x_f32,
                x_bf16: &x_bf16,
                x_row_div: div,
                x_rows,
            };

            for &bf16_x in &dtypes {
                let y_ref = gpu
                    .upload_f32(&vec![POISON; rows.len() * r], &[rows.len() * r])
                    .expect("y ref");
                let y_cand = gpu
                    .upload_f32(&vec![POISON; rows.len() * r], &[rows.len() * r])
                    .expect("y cand");
                if identities.is_none() {
                    let mut dev = gpu.try_gfx942().expect("exact gfx942");
                    let rid = dev
                        .fp8b128_moe_grouped_build_identity(true)
                        .expect("reference identity");
                    let pid = dev
                        .fp8b128_moe_grouped_build_identity(false)
                        .expect("production identity");
                    identities = Some((identity_json(&rid), identity_json(&pid)));
                }
                for _ in 0..TIME_WARMUPS {
                    timed_launch(gpu, &rt, bf16_x, true, &y_ref);
                    timed_launch(gpu, &rt, bf16_x, false, &y_cand);
                }
                gpu.hip.device_synchronize().expect("sync warmup");
                let (mut t_ref, mut t_cand) = (
                    Vec::with_capacity(TIME_LAUNCHES),
                    Vec::with_capacity(TIME_LAUNCHES),
                );
                for i in 0..TIME_LAUNCHES {
                    let one = |gpu: &mut Gpu, reference: bool| {
                        let y = if reference { &y_ref } else { &y_cand };
                        event_span_us(gpu, &e0, &e1, 1, |g| {
                            timed_launch(g, &rt, bf16_x, reference, y)
                        })
                    };
                    if i % 2 == 0 {
                        t_ref.push(one(gpu, true));
                        t_cand.push(one(gpu, false));
                    } else {
                        t_cand.push(one(gpu, false));
                        t_ref.push(one(gpu, true));
                    }
                }
                let burst_ref = event_span_us(gpu, &e0, &e1, TIME_LAUNCHES, |g| {
                    timed_launch(g, &rt, bf16_x, true, &y_ref)
                }) / TIME_LAUNCHES as f64;
                let burst_cand = event_span_us(gpu, &e0, &e1, TIME_LAUNCHES, |g| {
                    timed_launch(g, &rt, bf16_x, false, &y_cand)
                }) / TIME_LAUNCHES as f64;

                // Whole-Y raw-bit comparison (live, padding rows, every element).
                gpu.hip.device_synchronize().expect("sync compare");
                let a = gpu.download_f32(&y_ref).expect("download ref");
                let b = gpu.download_f32(&y_cand).expect("download cand");
                let mismatches = a.len().abs_diff(b.len()) as u64
                    + a.iter()
                        .zip(&b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count() as u64;
                let unwritten = b.iter().filter(|v| v.to_bits() == POISON.to_bits()).count();
                total_mismatch += mismatches;
                for t in [y_ref, y_cand] {
                    gpu.free_tensor(t).ok();
                }

                let dtype = if bf16_x { "bf16" } else { "f32" };
                let (rm, cm) = (median_us(&t_ref), median_us(&t_cand));
                println!(
                    "{} x={dtype} {label} r={r} k={k} tokens={tokens} slots={} tiles={} experts={}: reference {rm:.2} us ({:.2} TFLOPS) candidate {cm:.2} us ({:.2} TFLOPS) speedup {:.3}x burst {:.2}/{:.2} us mismatches={mismatches}",
                    if mismatches == 0 { "OK  " } else { "FAIL" },
                    slots.len(),
                    tiles.len(),
                    uniq.len(),
                    useful_flops / (rm * 1e-6) / 1e12,
                    useful_flops / (cm * 1e-6) / 1e12,
                    rm / cm,
                    burst_ref,
                    burst_cand,
                );
                records.push(serde_json::json!({
                    "x_dtype": dtype,
                    "shape": label,
                    "r": r,
                    "k": k,
                    "x_row_div": div,
                    "tokens": tokens,
                    "x_rows": x_rows,
                    "n_experts_pool": ex.n,
                    "unique_experts": uniq.len(),
                    "live_slots": slots.len(),
                    "tiles": tiles.len(),
                    "padded_rows": rows.len(),
                    "layout": "padded-16 (stable scatter; expert ascending, flat ascending)",
                    "route_digest": route_digest,
                    "x_digest": x_digest,
                    "weights_digest": weights_digest,
                    "useful_flops": useful_flops,
                    "y_values_compared": a.len(),
                    "raw_f32_bit_mismatches": mismatches,
                    "candidate_unwritten_poison_values": unwritten,
                    "reference": samples_json(&t_ref, burst_ref, useful_flops),
                    "candidate": samples_json(&t_cand, burst_cand, useful_flops),
                    "speedup_median": rm / cm,
                    "speedup_burst_mean": burst_ref / burst_cand,
                }));
            }
            for t in [x_bf16, x_f32, d_rows, d_tiles] {
                gpu.free_tensor(t).ok();
            }
        }
        gpu.free_tensor(d_payload).ok();
    }

    let (reference, production) = identities.expect("no timed record");
    let doc = serde_json::json!({
        "tool": "test_fp8_moe_gfx942 --time",
        "arch": gpu.arch,
        "args": args,
        "protocol": {
            "warmups": TIME_WARMUPS,
            "timed_launches": TIME_LAUNCHES,
            "timing": "per launch: event record, launch, event record, sync on stop; reference/candidate alternate with order swapped each iteration; median = sorted[n/2]",
            "burst": "one event span over 60 back-to-back launches / 60 (excludes host launch gaps); diagnostic",
            "cache_policy": TIME_CACHE_POLICY,
            "useful_flops": "2 * live_slots * r * k",
        },
        "reference_fixture": "crates/rdna-compute/tests/fixtures/fp8_moe_grouped_reference_p1_cdc37659a.gfx942.hip",
        "reference": reference,
        "production": production,
        "total_raw_f32_bit_mismatches": total_mismatch,
        "records": records,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap())
        .unwrap_or_else(|e| panic!("write {out_path}: {e}"));
    println!("timing JSON: {out_path}");
    for ev in [e0, e1] {
        gpu.hip.event_destroy(ev).ok();
    }
    if total_mismatch != 0 {
        println!(
            "FAILED: {total_mismatch} raw F32 bit mismatches between reference and production"
        );
        std::process::exit(1);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fixtures = args
        .iter()
        .position(|a| a == "--fixtures")
        .map(|i| args[i + 1].clone());
    let byte = args.iter().any(|a| a == "--byte-identity");
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!("detected arch: {}", gpu.arch);
    assert_eq!(gpu.arch, "gfx942", "this Tier C gate requires gfx942");
    if args.iter().any(|a| a == "--time") {
        time_mode(&mut gpu, &args);
        return;
    }
    let mut fail = Fail(Vec::new());

    // 1a. All 256 codes x scale sweep vs CPU teacher twin. One [128, 256] block
    // per scale pair: every 128x128 scale block holds each code 64 times.
    {
        let (r, k) = (128usize, 256usize);
        let mut st = 0xdec0_de42_u64;
        let mut sweep: Vec<u16> = vec![
            0x3F80, 0x3F81, 0x3FFF, 0x3C00, 0x3800, 0x3A83, 0x3B7F, 0x3901, 0x39FF, 0x4000, 0x4321,
            0x3E55, 0x3D2B, 0x2F80, 0x2FFF, 0xBA83, 0xBF80, 0x0D80, 0x7000,
        ];
        sweep.extend((0..45).map(|_| rand_scale(&mut st)));
        if sweep.len() % 2 == 1 {
            sweep.push(0x3B00);
        }
        let codes: Vec<u8> = (0..r * k)
            .map(|i| {
                let b = (i % 256) as u8;
                if b & 0x7F == 0x7F {
                    0x80
                } else {
                    b
                }
            })
            .collect();
        let e = sweep.len() / 2;
        let mut payload = Vec::new();
        let mut want = Vec::new();
        for p in sweep.chunks(2) {
            payload.extend(block_bytes(&codes, p, r, k));
            want.extend(
                dequant_b128_block(&codes, p, r, k).expect("cpu twin refused a sweep scale"),
            );
        }
        let got = gpu_dequant(&mut gpu, &payload, e, r, k);
        let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
        fail.check(
            bad == 0,
            format!("decode: 254 finite codes x {} scales, GPU dequant BF16 bits == CPU teacher twin ({} values, {bad} mismatched)", sweep.len(), got.len()),
        );
    }

    // 1b. Python-teacher bits (fixtures from mkfix.py), if given.
    if let Some(dir) = fixtures.as_deref() {
        let scales: Vec<u16> = read(dir, "codes256.scales")
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        let refs: Vec<u16> = read(dir, "codes256.ref")
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        let (r, k) = (1usize, 256usize);
        let codes: Vec<u8> = (0..256)
            .map(|b| if b & 0x7F == 0x7F { 0 } else { b as u8 })
            .collect();
        let payload: Vec<u8> = scales
            .iter()
            .flat_map(|&s| block_bytes(&codes, &[s, s], r, k))
            .collect();
        let got = gpu_dequant(&mut gpu, &payload, scales.len(), r, k);
        let (mut cmp, mut bad) = (0usize, 0usize);
        for (i, (&g, &w)) in got.iter().zip(&refs).enumerate() {
            if (i % 256) & 0x7F == 0x7F {
                continue;
            }
            cmp += 1;
            bad += (g != w) as usize;
        }
        fail.check(bad == 0, format!("decode: 256 codes x {} scales vs Python teacher bits ({cmp} compared, {bad} mismatched)", scales.len()));
        for (name, r, k) in [("gateup", 1280usize, 2560usize), ("down", 2560, 640)] {
            let blk = read(dir, &format!("{name}.blk"));
            let want: Vec<u16> = read(dir, &format!("{name}.ref"))
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect();
            let got = gpu_dequant(&mut gpu, &blk, 1, r, k);
            fail.check(got == want, format!("decode: real checkpoint {name} expert [{r},{k}] GPU dequant == Python teacher bits"));
            let cpu = rdna_compute::fp8_e4m3::dequant_b128_expert_bytes(&blk, r, k)
                .expect("cpu decode real");
            fail.check(
                cpu == want,
                format!("decode: real checkpoint {name} expert CPU twin == Python teacher bits"),
            );
        }
        // Real expert in the grouped GEMM (single expert, x_row_div 1).
        for (name, r, k) in [("gateup", 1280usize, 2560usize), ("down", 2560, 640)] {
            let blk = read(dir, &format!("{name}.blk"));
            let w: Vec<u16> = read(dir, &format!("{name}.ref"))
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect();
            let ex = Experts {
                r,
                k,
                n: 1,
                payload: blk,
                w: vec![w],
            };
            // Reuse the synthetic driver with a single-expert pool: tokens=1 slot row per
            // flat index, every slot on expert 0.
            real_gemm(&mut gpu, &mut fail, name, &ex, byte);
        }
    }

    // 2-4. Grouped GEMM Tier C at the Qwen4 shapes.
    let gate_up = Experts::synthetic(1280, 2560, 12, 0x6a7e_0001);
    gemm_case(
        &mut gpu,
        &mut fail,
        "gate_up r=1280 k=2560",
        &gate_up,
        10,
        37,
        byte,
    );
    drop(gate_up);
    let down = Experts::synthetic(2560, 640, 12, 0xd0e1_0002);
    gemm_case(&mut gpu, &mut fail, "down r=2560 k=640", &down, 1, 13, byte);
    drop(down);

    if byte {
        println!("byte-identity: baseline = retained P1 reference (standalone SHA-pinned snapshot), candidate = production entry");
        byte_cases(&mut gpu, &mut fail);
    }

    if fail.0.is_empty() {
        println!("ALL PASS test_fp8_moe_gfx942 (arch {})", gpu.arch);
    } else {
        println!("FAILED {} checks:", fail.0.len());
        for f in &fail.0 {
            println!("  {f}");
        }
        std::process::exit(1);
    }
}

/// Real checkpoint expert: 64 BF16 x rows through one expert vs F64 on the
/// Python-teacher W.
fn real_gemm(gpu: &mut Gpu, fail: &mut Fail, name: &str, ex: &Experts, byte_ab: bool) {
    let (r, k) = (ex.r, ex.k);
    let rows = 64usize;
    let mut st = 0x7ea1_u64;
    let xb: Vec<u16> = (0..rows * k)
        .map(|_| f32_to_bf16(((lcg(&mut st) as f32 / u32::MAX as f32) - 0.5) * 4.0))
        .collect();
    let xf: Vec<f32> = xb.iter().map(|&b| bf16_to_f32(b)).collect();
    let d_payload = upload_bytes(gpu, &ex.payload);
    let x_f32 = gpu.upload_f32(&xf, &[rows * k]).expect("x");
    let mut x_bf16 = upload_bytes(
        gpu,
        &xb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
    );
    x_bf16.shape = vec![rows * k];
    x_bf16.dtype = DType::BF16;
    let ab = byte_ab.then(Ab::new);
    let l = Launch {
        ex,
        d_payload: &d_payload,
        x_row_div: 1,
        x_rows: rows,
        x_f32: &x_f32,
        x_bf16: &x_bf16,
        ab: ab.as_ref(),
        tag: name,
    };
    let (t, s) = scatter(&vec![0usize; rows], 1, true);
    let y = l.run(gpu, &t, &s, false);
    let bound_k = k as f64 * (2f64).powi(-24);
    let (mut worst, mut bad) = (0f64, 0usize);
    for row in 0..rows {
        for o in 0..r {
            let (mut acc, mut aa) = (0f64, 0f64);
            for c in 0..k {
                let p = bf16_to_f32(ex.w[0][o * k + c]) as f64 * xf[row * k + c] as f64;
                acc += p;
                aa += p.abs();
            }
            let err = (y[row * r + o] as f64 - acc).abs();
            if aa > 0.0 {
                worst = worst.max(err / aa);
            }
            bad += (err > bound_k * aa) as usize;
        }
    }
    fail.check(bad == 0, format!("real {name}: GEMM vs F64 on teacher W, {rows} rows, max err/sum|wx| = {worst:.3e}, violations {bad}"));
    if let Some(ab) = &ab {
        ab.check(fail, &format!("real {name}"));
    }
    for t in [x_bf16, x_f32, d_payload] {
        gpu.free_tensor(t).ok();
    }
}
