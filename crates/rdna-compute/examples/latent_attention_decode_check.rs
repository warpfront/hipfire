// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Split-KV latent-attention decode check (`Gpu::latent_attention_decode_split`).
//!
//! Compares the launcher against an independent F64 host reference over
//!
//! * Raon-like geometries (token-major expanded mirrors, 192/128 and the
//!   absorbed 576/512 shape with `head_stride == 0`), an oracle-fixture-like
//!   geometry (H=8, nope=64, tail=32, v=96, latent=256) and non-Raon shapes
//!   (padded strides, per-head tails, widths that are multiples of 8 but not
//!   16, `dv/8` that does not divide 256, the width maxima 2048/2048);
//! * visible lengths `pos + 1` for `pos` in {0,1,15,16,17,63,64,65,200,1000,20000}
//!   reached through different `(*pos, pos_offset, max_rows, capacity)` splits:
//!   padded, tight, `pos_offset != 0`, `*pos` far beyond `max_rows` (truncation),
//!   `pos_offset`-only and (for short lengths) a much larger `max_rows`/buffer;
//!   the output bytes of all of them MUST be identical;
//! * empty visibility (`len == 0`, zeros);
//! * `round_y_bf16` on/off (on == BF16 RNE of off, bit-exact);
//! * poisoned memory: every row at or beyond the visible range, every view gap
//!   and every unused cache column holds 0x7149 (1e30-class BF16) or a NaN;
//! * byte-identical repeats with the workspace pre-filled with NaN / 1e30 words
//!   (stale workspace must never be read), output/workspace guard words;
//! * launcher refusals (rows, workspace, alignment, widths, extents);
//! * graph capture (blob path), replayed with different device `pos` values:
//!   bytes equal to eager launches.
//!
//! Bands vs the F64 reference: rel_l2 <= 1e-4 (pure FP32, `round_y_bf16 = false`),
//! rel_l2 <= 1e-2 and cosine >= 0.9999 with BF16 output rounding.
//!
//! Run (any RDNA card; gfx1201 for the lab runs):
//!   cargo build --release -p rdna-compute --example latent_attention_decode_check
//!   <target>/release/examples/latent_attention_decode_check

use rdna_compute::latent_attention_decode::{
    latent_attention_decode_workspace_elems, LATENT_DECODE_MAX_DQ, LATENT_DECODE_MAX_DV,
    LATENT_DECODE_SPLITS,
};
use rdna_compute::latent_attention_tiled::{LatentRows, TiledAttention};
use rdna_compute::{DType, Gpu, GpuTensor};

const GUARD: usize = 64;
/// A quiet NaN with a payload no arithmetic produces.
const SENTINEL: u32 = 0x7fc0_beef;
const WS_FILL_NAN: u32 = 0x7fc0_dead;
/// F32 of the 1e30 class.
const WS_FILL_BIG: u32 = 0x7149_7149;
const POSITIONS: [usize; 11] = [0, 1, 15, 16, 17, 63, 64, 65, 200, 1000, 20000];

const FP32_REL_L2: f64 = 1e-4;
const BF16_REL_L2: f64 = 1e-2;
const BF16_COS: f64 = 0.9999;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Round-to-nearest-even BF16 bits of a finite F32.
fn bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(bf16_bits(x))
}

/// Poison BF16 pattern: 1e30-class on even indices, a NaN on odd ones.
fn poison(i: usize) -> u16 {
    if i % 2 == 0 {
        0x7149
    } else {
        0x7fc1
    }
}

// ---------------------------------------------------------------------------
// Geometry

#[derive(Clone, Copy)]
struct View {
    buf: usize,
    offset: usize,
    head: usize,
    width: usize,
}

/// A BF16 buffer of `row` elements per token.
struct BufSpec {
    row: usize,
    amp: f32,
    bias: f32,
    /// Every 53rd row is 3x larger (peaky keys).
    spike: bool,
}

struct Geo {
    name: &'static str,
    heads: usize,
    bufs: Vec<BufSpec>,
    ka: View,
    kb: View,
    v: View,
    positions: Vec<usize>,
    scale_mults: Vec<f32>,
    seed: u64,
}

impl Geo {
    fn dq(&self) -> usize {
        self.ka.width + self.kb.width
    }
    fn dv(&self) -> usize {
        self.v.width
    }
    fn tok(&self, v: &View) -> usize {
        self.bufs[v.buf].row
    }
    fn check(&self) {
        for (name, v) in [("k_a", self.ka), ("k_b", self.kb), ("v", self.v)] {
            let row = self.bufs[v.buf].row;
            assert!(
                v.offset + (self.heads - 1) * v.head + v.width <= row,
                "{}: view {name} exceeds its row",
                self.name
            );
            assert!(row % 8 == 0 && v.offset % 8 == 0 && v.head % 8 == 0 && v.width % 8 == 0);
        }
    }
    /// Columns of buffer `b` covered by any view.
    fn covered(&self, b: usize) -> Vec<bool> {
        let mut c = vec![false; self.bufs[b].row];
        for v in [self.ka, self.kb, self.v] {
            if v.buf != b {
                continue;
            }
            for h in 0..self.heads {
                for d in 0..v.width {
                    c[v.offset + h * v.head + d] = true;
                }
            }
        }
        c
    }
}

fn spec(row: usize, amp: f32, bias: f32, spike: bool) -> BufSpec {
    BufSpec {
        row,
        amp,
        bias,
        spike,
    }
}

fn all_positions() -> Vec<usize> {
    POSITIONS.to_vec()
}

fn geometries() -> Vec<Geo> {
    vec![
        // Raon-like decompressed route: token-major mirrors [t][h][128], tail
        // from the 576-wide latent cache row (c512 || kpe64, c512 unused here).
        Geo {
            name: "raon-decomp H40 128+64/128",
            heads: 40,
            bufs: vec![
                spec(5120, 1.0, 0.0, true),
                spec(576, 1.0, 0.1, true),
                spec(5120, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 0,
                head: 128,
                width: 128,
            },
            kb: View {
                buf: 1,
                offset: 512,
                head: 0,
                width: 64,
            },
            v: View {
                buf: 2,
                offset: 0,
                head: 128,
                width: 128,
            },
            positions: all_positions(),
            scale_mults: vec![1.0],
            seed: 0x1001,
        },
        // Absorbed route: k_a / v are the latent channels of the cache row.
        Geo {
            name: "raon-absorbed H4 512+64/512",
            heads: 4,
            bufs: vec![spec(576, 1.0, 0.2, true)],
            ka: View {
                buf: 0,
                offset: 0,
                head: 0,
                width: 512,
            },
            kb: View {
                buf: 0,
                offset: 512,
                head: 0,
                width: 64,
            },
            v: View {
                buf: 0,
                offset: 0,
                head: 0,
                width: 512,
            },
            positions: all_positions(),
            scale_mults: vec![1.0],
            seed: 0x1002,
        },
        // Oracle-fixture-like: H=8, nope=64, tail=32, v=96, latent=256.
        Geo {
            name: "fixture-decomp H8 64+32/96",
            heads: 8,
            bufs: vec![
                spec(512, 1.0, 0.0, true),
                spec(288, 1.0, 0.1, true),
                spec(768, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 0,
                head: 64,
                width: 64,
            },
            kb: View {
                buf: 1,
                offset: 256,
                head: 0,
                width: 32,
            },
            v: View {
                buf: 2,
                offset: 0,
                head: 96,
                width: 96,
            },
            positions: vec![0, 1, 16, 17, 64, 65, 200, 1000, 4000],
            scale_mults: vec![1.0],
            seed: 0x1003,
        },
        Geo {
            name: "fixture-absorbed H8 256+32/256",
            heads: 8,
            bufs: vec![spec(288, 1.0, 0.2, true)],
            ka: View {
                buf: 0,
                offset: 0,
                head: 0,
                width: 256,
            },
            kb: View {
                buf: 0,
                offset: 256,
                head: 0,
                width: 32,
            },
            v: View {
                buf: 0,
                offset: 0,
                head: 0,
                width: 256,
            },
            positions: vec![0, 1, 16, 17, 64, 65, 200, 1000, 4000],
            scale_mults: vec![1.0],
            seed: 0x1004,
        },
        // Non-Raon: H=3, wa=32, wb=16, v=48; padded strides (poisoned gaps),
        // per-head tails, peaky and flat softmax scales.
        Geo {
            name: "odd H3 32+16/48 padded per-head-tail",
            heads: 3,
            bufs: vec![
                spec(104, 1.0, 0.0, true),
                spec(48, 1.0, 0.1, true),
                spec(160, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 8,
                head: 32,
                width: 32,
            },
            kb: View {
                buf: 1,
                offset: 0,
                head: 16,
                width: 16,
            },
            v: View {
                buf: 2,
                offset: 16,
                head: 48,
                width: 48,
            },
            positions: all_positions(),
            scale_mults: vec![1.0, 4.0, 0.0],
            seed: 0x1005,
        },
        // Widths multiples of 8 but not 16: 24+8/40, shared tail with a gap.
        Geo {
            name: "odd8 H2 24+8/40",
            heads: 2,
            bufs: vec![
                spec(48, 1.0, 0.0, true),
                spec(24, 1.0, 0.1, true),
                spec(80, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 0,
                head: 24,
                width: 24,
            },
            kb: View {
                buf: 1,
                offset: 16,
                head: 0,
                width: 8,
            },
            v: View {
                buf: 2,
                offset: 0,
                head: 40,
                width: 40,
            },
            positions: vec![0, 1, 15, 16, 17, 64, 65, 300],
            scale_mults: vec![1.0],
            seed: 0x1006,
        },
        // dv = 200: 25 value vectors, 10 key groups (256 / 25), 6 idle threads.
        Geo {
            name: "odd-vg H2 64+16/200",
            heads: 2,
            bufs: vec![
                spec(136, 1.0, 0.0, true),
                spec(32, 1.0, 0.1, true),
                spec(400, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 8,
                head: 64,
                width: 64,
            },
            kb: View {
                buf: 1,
                offset: 0,
                head: 16,
                width: 16,
            },
            v: View {
                buf: 2,
                offset: 0,
                head: 200,
                width: 200,
            },
            positions: vec![0, 1, 17, 65, 300],
            scale_mults: vec![1.0],
            seed: 0x1007,
        },
        // dv = 1024: 128 value vectors, 2 key groups.
        Geo {
            name: "wide-dv H2 256+64/1024",
            heads: 2,
            bufs: vec![
                spec(512, 1.0, 0.0, true),
                spec(64, 1.0, 0.1, true),
                spec(2048, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 0,
                head: 256,
                width: 256,
            },
            kb: View {
                buf: 1,
                offset: 0,
                head: 0,
                width: 64,
            },
            v: View {
                buf: 2,
                offset: 0,
                head: 1024,
                width: 1024,
            },
            positions: vec![0, 1, 17, 65, 300, 1000],
            scale_mults: vec![1.0],
            seed: 0x1008,
        },
        // Width maxima: dq = 2048, dv = 2048 (256 value vectors, 1 key group).
        Geo {
            name: "max H2 1536+512/2048",
            heads: 2,
            bufs: vec![
                spec(3072, 1.0, 0.0, true),
                spec(512, 1.0, 0.1, true),
                spec(4096, 1.0, 0.3, false),
            ],
            ka: View {
                buf: 0,
                offset: 0,
                head: 1536,
                width: 1536,
            },
            kb: View {
                buf: 1,
                offset: 0,
                head: 0,
                width: 512,
            },
            v: View {
                buf: 2,
                offset: 0,
                head: 2048,
                width: 2048,
            },
            positions: vec![0, 1, 17, 65, 300],
            scale_mults: vec![1.0],
            seed: 0x1009,
        },
    ]
}

// ---------------------------------------------------------------------------
// Host data and reference

/// Host buffer `b` with `rows_total` rows: rows `< data_rows` hold random data
/// in the view-covered columns (a pure function of `(seed, b, row)`, whatever
/// `rows_total` is) and poison in the gaps; every other row is poison.
fn gen_host_buf(geo: &Geo, b: usize, rows_total: usize, data_rows: usize) -> Vec<u16> {
    let s = &geo.bufs[b];
    let covered = geo.covered(b);
    let mut out = vec![0u16; rows_total * s.row];
    for t in 0..rows_total {
        let row = &mut out[t * s.row..(t + 1) * s.row];
        if t >= data_rows {
            for (c, e) in row.iter_mut().enumerate() {
                *e = poison(t + c);
            }
            continue;
        }
        let mut rng = Rng(geo.seed
            ^ (b as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (t as u64 + 1).wrapping_mul(0xA076_1D64_78BD_642F));
        let amp = s.amp * if s.spike && t % 53 == 7 { 3.0 } else { 1.0 };
        for (c, e) in row.iter_mut().enumerate() {
            let u = rng.unit();
            *e = if covered[c] {
                bf16_bits((u * 2.0 - 1.0) * amp + s.bias)
            } else {
                poison(c)
            };
        }
    }
    out
}

fn gen_q(geo: &Geo, p: usize) -> Vec<f32> {
    let mut rng = Rng(geo.seed ^ 0x51ED_270B ^ (p as u64 + 1).wrapping_mul(0x9E37_79B9));
    (0..geo.heads * geo.dq())
        .map(|_| (rng.unit() * 2.0 - 1.0) * 3.0)
        .collect()
}

/// F64 reference `[heads * dv]` for `len` visible keys (no BF16 output rounding).
fn reference(geo: &Geo, bufs: &[Vec<u16>], q: &[f32], len: usize, scale: f32) -> Vec<f64> {
    let (hn, wa, wb, wv) = (geo.heads, geo.ka.width, geo.kb.width, geo.dv());
    let dq = wa + wb;
    let mut y = vec![0.0f64; hn * wv];
    if len == 0 {
        return y;
    }
    let (ta, tb, tv) = (geo.tok(&geo.ka), geo.tok(&geo.kb), geo.tok(&geo.v));
    let (ba, bb, bv) = (&bufs[geo.ka.buf], &bufs[geo.kb.buf], &bufs[geo.v.buf]);
    let mut sc = vec![0.0f64; len];
    for h in 0..hn {
        let qh = &q[h * dq..(h + 1) * dq];
        let mut mx = f64::NEG_INFINITY;
        for (j, s_out) in sc.iter_mut().enumerate() {
            let mut s = 0.0f64;
            let a0 = j * ta + geo.ka.offset + h * geo.ka.head;
            for d in 0..wa {
                s += qh[d] as f64 * bf16_to_f32(ba[a0 + d]) as f64;
            }
            let b0 = j * tb + geo.kb.offset + h * geo.kb.head;
            for d in 0..wb {
                s += qh[wa + d] as f64 * bf16_to_f32(bb[b0 + d]) as f64;
            }
            s *= scale as f64;
            *s_out = s;
            mx = mx.max(s);
        }
        let mut den = 0.0f64;
        for s in sc.iter_mut() {
            *s = (*s - mx).exp();
            den += *s;
        }
        let yh = &mut y[h * wv..(h + 1) * wv];
        for (j, s) in sc.iter().enumerate() {
            let p = s / den;
            let v0 = j * tv + geo.v.offset + h * geo.v.head;
            for d in 0..wv {
                yh[d] += p * bf16_to_f32(bv[v0 + d]) as f64;
            }
        }
    }
    y
}

/// (rel_l2, cosine, max_abs) of `got` against `want`.
fn metrics(got: &[f32], want: &[f64]) -> (f64, f64, f64) {
    let (mut num, mut den, mut dot, mut gg, mut maxabs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        let g = *g as f64;
        let d = g - w;
        num += d * d;
        den += w * w;
        dot += g * w;
        gg += g * g;
        maxabs = maxabs.max(d.abs());
    }
    let rel = if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    };
    let cos = if gg > 0.0 && den > 0.0 {
        dot / (gg.sqrt() * den.sqrt())
    } else {
        1.0
    };
    (rel, cos, maxabs)
}

// ---------------------------------------------------------------------------
// Device helpers

fn upload_bf16(gpu: &mut Gpu, data: &[u16]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::BF16)
        .expect("alloc bf16");
    // SAFETY: u16 -> u8 view of the same bytes; the host is little endian.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    gpu.hip
        .device_synchronize()
        .expect("sync before bf16 upload");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("upload bf16");
    t
}

fn fill_words(gpu: &Gpu, t: &GpuTensor, word: u32) {
    let bytes: Vec<u8> = (0..t.numel()).flat_map(|_| word.to_le_bytes()).collect();
    gpu.hip.device_synchronize().expect("sync before fill");
    gpu.hip.memcpy_htod(&t.buf, &bytes).expect("fill words");
}

fn set_pos(gpu: &Gpu, pos: &GpuTensor, v: i32) {
    gpu.hip.device_synchronize().expect("sync before pos write");
    gpu.hip
        .memcpy_htod(&pos.buf, &v.to_le_bytes())
        .expect("pos write");
}

fn fetch_bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.hip.device_synchronize().expect("sync before fetch");
    gpu.download_f32(t)
        .expect("download")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn to_f32(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|b| f32::from_bits(*b)).collect()
}

fn view_rows<'a>(geo: &Geo, dev: &'a [GpuTensor], v: &View) -> LatentRows<'a> {
    LatentRows {
        buf: &dev[v.buf],
        offset: v.offset,
        tok_stride: geo.tok(v),
        head_stride: v.head,
        width: v.width,
    }
}

#[derive(Default)]
struct Stats {
    checks: usize,
    fails: usize,
    worst_rel_fp32: f64,
    worst_rel_bf16: f64,
    worst_cos_bf16: f64,
    identical_comparisons: usize,
    launches: usize,
}

impl Stats {
    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        self.checks += 1;
        if !ok {
            self.fails += 1;
            println!("  FAIL: {}", what());
        }
    }
}

struct Scn {
    tag: &'static str,
    pos_dev: i32,
    pos_offset: usize,
    max_rows: usize,
    cap: usize,
}

/// Every `(pos_dev, pos_offset, max_rows, cap)` split reaching `len = p + 1`.
fn scenarios(p: usize) -> Vec<Scn> {
    let len = p + 1;
    let mut v = vec![
        Scn {
            tag: "pad",
            pos_dev: p as i32,
            pos_offset: 0,
            max_rows: len + 37,
            cap: len + 37 + 13,
        },
        Scn {
            tag: "offset",
            pos_dev: p as i32 - 3,
            pos_offset: 3,
            max_rows: len + 5,
            cap: len + 5,
        },
        Scn {
            tag: "tight",
            pos_dev: p as i32,
            pos_offset: 0,
            max_rows: len,
            cap: len,
        },
        Scn {
            tag: "trunc",
            pos_dev: p as i32 + 500,
            pos_offset: 0,
            max_rows: len,
            cap: len + 70,
        },
        Scn {
            tag: "off-only",
            pos_dev: 0,
            pos_offset: p,
            max_rows: len + 9,
            cap: len + 9,
        },
    ];
    if p <= 1000 {
        let big = 2 * len + 4096;
        v.push(Scn {
            tag: "enlarged",
            pos_dev: p as i32,
            pos_offset: 0,
            max_rows: big,
            cap: big,
        });
    }
    v
}

struct Outs {
    noround: Vec<u32>,
    round: Vec<u32>,
}

/// Upload the first `scn.cap` rows of every host buffer, run both rounding
/// modes twice (different workspace fills), return the bytes; repeat
/// mismatches and guard damage are recorded in `st`.
#[allow(clippy::too_many_arguments)]
fn run_scn(
    gpu: &mut Gpu,
    st: &mut Stats,
    geo: &Geo,
    host: &[Vec<u16>],
    q_dev: &GpuTensor,
    pos_t: &GpuTensor,
    scn: &Scn,
    scale: f32,
    label: &str,
) -> Result<Outs, String> {
    let dev: Vec<GpuTensor> = host
        .iter()
        .zip(&geo.bufs)
        .map(|(h, s)| upload_bf16(gpu, &h[..scn.cap * s.row]))
        .collect();
    set_pos(gpu, pos_t, scn.pos_dev);
    let yn = geo.heads * geo.dv();
    let y = gpu
        .alloc_tensor(&[yn + GUARD], DType::F32)
        .map_err(|e| e.to_string())?;
    let ws_need = latent_attention_decode_workspace_elems(geo.heads, geo.dv());
    let ws = gpu
        .alloc_tensor(&[ws_need + GUARD], DType::F32)
        .map_err(|e| e.to_string())?;
    let mut runs: Vec<(bool, u32, Vec<u32>)> = Vec::new();
    for (round, fill) in [
        (false, WS_FILL_NAN),
        (false, WS_FILL_BIG),
        (true, WS_FILL_NAN),
        (true, WS_FILL_BIG),
    ] {
        fill_words(gpu, &ws, fill);
        fill_words(gpu, &y, SENTINEL);
        let a = TiledAttention {
            q: q_dev,
            y: &y,
            k_a: view_rows(geo, &dev, &geo.ka),
            k_b: view_rows(geo, &dev, &geo.kb),
            v: view_rows(geo, &dev, &geo.v),
            pos: pos_t,
            pos_offset: scn.pos_offset,
            n_rows: 1,
            n_heads: geo.heads,
            scale,
            max_rows: scn.max_rows,
            round_y_bf16: round,
        };
        gpu.latent_attention_decode_split(&a, &ws)
            .map_err(|e| e.to_string())?;
        st.launches += 1;
        let bits = fetch_bits(gpu, &y);
        let ws_bits = fetch_bits(gpu, &ws);
        st.check(bits[yn..].iter().all(|b| *b == SENTINEL), || {
            format!("{label}: output guard overwritten")
        });
        st.check(ws_bits[ws_need..].iter().all(|b| *b == fill), || {
            format!("{label}: workspace guard overwritten")
        });
        runs.push((round, fill, bits[..yn].to_vec()));
    }
    for t in dev {
        gpu.free_tensor(t).map_err(|e| e.to_string())?;
    }
    gpu.free_tensor(y).map_err(|e| e.to_string())?;
    gpu.free_tensor(ws).map_err(|e| e.to_string())?;
    st.check(runs[0].2 == runs[1].2, || {
        format!("{label}: repeat (fp32) not byte-identical")
    });
    st.check(runs[2].2 == runs[3].2, || {
        format!("{label}: repeat (bf16) not byte-identical")
    });
    st.identical_comparisons += 2;
    let (noround, round) = (runs[0].2.clone(), runs[2].2.clone());
    Ok(Outs { noround, round })
}

fn check_vs_reference(st: &mut Stats, label: &str, outs: &Outs, want: &[f64], len: usize) {
    let got = to_f32(&outs.noround);
    let got_r = to_f32(&outs.round);
    st.check(got.iter().chain(&got_r).all(|v| v.is_finite()), || {
        format!("{label}: non-finite output")
    });
    // BF16 output rounding is the RNE of the unrounded F32 result, bit for bit.
    st.check(
        got.iter()
            .zip(&got_r)
            .all(|(a, b)| round_bf16(*a).to_bits() == b.to_bits()),
        || format!("{label}: round_y_bf16 output is not bf16(unrounded)"),
    );
    if len == 0 {
        st.check(
            outs.noround.iter().chain(&outs.round).all(|b| *b == 0),
            || format!("{label}: len == 0 must write zeros"),
        );
        return;
    }
    let (rel, cos, maxabs) = metrics(&got, want);
    st.worst_rel_fp32 = st.worst_rel_fp32.max(rel);
    st.check(rel <= FP32_REL_L2, || {
        format!("{label}: fp32 rel_l2 {rel:.3e} > {FP32_REL_L2:.0e} (cos {cos:.9}, max_abs {maxabs:.3e})")
    });
    let want_r: Vec<f64> = want.iter().map(|w| round_bf16(*w as f32) as f64).collect();
    let (rel_r, cos_r, maxabs_r) = metrics(&got_r, &want_r);
    st.worst_rel_bf16 = st.worst_rel_bf16.max(rel_r);
    st.worst_cos_bf16 = if st.worst_cos_bf16 == 0.0 {
        cos_r
    } else {
        st.worst_cos_bf16.min(cos_r)
    };
    st.check(rel_r <= BF16_REL_L2 && cos_r >= BF16_COS, || {
        format!("{label}: bf16 rel_l2 {rel_r:.3e} cos {cos_r:.9} max_abs {maxabs_r:.3e} outside the band")
    });
    println!(
        "  {label}: len={len} fp32 rel_l2={rel:.2e} cos={cos:.9} | bf16 rel_l2={rel_r:.2e} cos={cos_r:.9}"
    );
}

fn run_geo(gpu: &mut Gpu, st: &mut Stats, geo: &Geo, pos_t: &GpuTensor) {
    geo.check();
    println!(
        "== {} (dq={}, dv={}, splits={})",
        geo.name,
        geo.dq(),
        geo.dv(),
        LATENT_DECODE_SPLITS
    );
    let base_scale = (geo.dq() as f32).powf(-0.5);

    // len == 0: nothing is visible (all rows poison).
    {
        let host: Vec<Vec<u16>> = (0..geo.bufs.len())
            .map(|b| gen_host_buf(geo, b, 13, 0))
            .collect();
        let q = gen_q(geo, 0);
        let q_dev = gpu.upload_f32(&q, &[q.len()]).expect("upload q");
        for scn in [
            Scn {
                tag: "empty:pos=-1",
                pos_dev: -1,
                pos_offset: 0,
                max_rows: 9,
                cap: 9,
            },
            Scn {
                tag: "empty:neg",
                pos_dev: -5,
                pos_offset: 2,
                max_rows: 9,
                cap: 13,
            },
        ] {
            let label = format!("{} / {}", geo.name, scn.tag);
            match run_scn(gpu, st, geo, &host, &q_dev, pos_t, &scn, base_scale, &label) {
                Ok(outs) => check_vs_reference(st, &label, &outs, &[], 0),
                Err(e) => st.check(false, || format!("{label}: launcher error: {e}")),
            }
        }
        gpu.free_tensor(q_dev).expect("free q");
    }

    for &p in &geo.positions {
        let len = p + 1;
        let scns = scenarios(p);
        let cap_max = scns.iter().map(|s| s.cap).max().unwrap();
        let host: Vec<Vec<u16>> = (0..geo.bufs.len())
            .map(|b| gen_host_buf(geo, b, cap_max, len))
            .collect();
        let q = gen_q(geo, p);
        let q_dev = gpu.upload_f32(&q, &[q.len()]).expect("upload q");
        for &mult in &geo.scale_mults {
            if p > 2000 && mult != 1.0 {
                continue;
            }
            let scale = base_scale * mult;
            let want = reference(geo, &host, &q, len, scale);
            let mut baseline: Option<Outs> = None;
            for scn in &scns {
                let label = format!("{} / pos={p} x{mult} / {}", geo.name, scn.tag);
                match run_scn(gpu, st, geo, &host, &q_dev, pos_t, scn, scale, &label) {
                    Err(e) => st.check(false, || format!("{label}: launcher error: {e}")),
                    Ok(outs) => {
                        match &baseline {
                            None => {
                                check_vs_reference(st, &label, &outs, &want, len);
                                baseline = Some(outs);
                            }
                            Some(b) => {
                                // Same len through a different (pos, offset, max_rows, capacity).
                                st.check(outs.noround == b.noround, || {
                                format!("{label}: fp32 bytes differ from the `pad` split (same len)")
                            });
                                st.check(outs.round == b.round, || {
                                format!("{label}: bf16 bytes differ from the `pad` split (same len)")
                            });
                                st.identical_comparisons += 2;
                            }
                        }
                    }
                }
            }
            // Negative control: make 5 poison rows visible; the harness must
            // see the difference (else the poison checks above prove nothing).
            if (p == 17 || p == 200) && mult == 1.0 {
                let scn = Scn {
                    tag: "control(reads-poison)",
                    pos_dev: p as i32 + 5,
                    pos_offset: 0,
                    max_rows: len + 37,
                    cap: len + 37 + 13,
                };
                let label = format!("{} / pos={p} / {}", geo.name, scn.tag);
                match run_scn(gpu, st, geo, &host, &q_dev, pos_t, &scn, scale, &label) {
                    Err(e) => st.check(false, || format!("{label}: launcher error: {e}")),
                    Ok(outs) => {
                        let differs = baseline.as_ref().is_some_and(|b| outs.noround != b.noround);
                        st.check(differs, || {
                            format!("{label}: poison rows made visible did not change the output")
                        });
                    }
                }
            }
        }
        gpu.free_tensor(q_dev).expect("free q");
    }
}

// ---------------------------------------------------------------------------
// Refusals

fn expect_refusal(
    gpu: &mut Gpu,
    st: &mut Stats,
    what: &str,
    needle: &str,
    a: &TiledAttention<'_>,
    ws: &GpuTensor,
) {
    match gpu.latent_attention_decode_split(a, ws) {
        Ok(()) => st.check(false, || {
            format!("refusal `{what}`: launcher accepted the call")
        }),
        Err(e) => {
            let msg = e.to_string();
            st.check(msg.contains(needle), || {
                format!("refusal `{what}`: message `{msg}` lacks `{needle}`")
            });
            println!("  refused `{what}`: {msg}");
        }
    }
}

fn refusal_checks(gpu: &mut Gpu, st: &mut Stats) {
    println!("== launcher refusals");
    let rows = 4usize;
    let row = 48usize;
    let buf = upload_bf16(gpu, &vec![0x3f80u16; rows * row]);
    let q = gpu.upload_f32(&vec![0.5f32; 64], &[64]).unwrap();
    let y = gpu.upload_f32(&vec![0.0f32; 4096], &[4096]).unwrap();
    let pos = gpu.upload_f32(&[0.0f32], &[1]).unwrap();
    set_pos(gpu, &pos, 1);
    let ws_need = latent_attention_decode_workspace_elems(1, 16);
    let ws = gpu.upload_f32(&vec![0.0f32; ws_need], &[ws_need]).unwrap();
    let view = |offset: usize, width: usize| LatentRows {
        buf: &buf,
        offset,
        tok_stride: row,
        head_stride: 0,
        width,
    };
    let base = TiledAttention {
        q: &q,
        y: &y,
        k_a: view(0, 16),
        k_b: view(16, 16),
        v: view(32, 16),
        pos: &pos,
        pos_offset: 0,
        n_rows: 1,
        n_heads: 1,
        scale: 0.25,
        max_rows: rows,
        round_y_bf16: false,
    };
    // Control: the unmodified call is accepted.
    st.check(
        gpu.latent_attention_decode_split(&base, &ws).is_ok(),
        || "refusal control: base call rejected".into(),
    );

    expect_refusal(
        gpu,
        st,
        "n_rows = 2",
        "one query row",
        &TiledAttention { n_rows: 2, ..base },
        &ws,
    );
    let ws_small = gpu
        .upload_f32(&vec![0.0f32; ws_need - 1], &[ws_need - 1])
        .unwrap();
    expect_refusal(
        gpu,
        st,
        "workspace too small",
        "workspace holds",
        &base,
        &ws_small,
    );
    let ws_bf16 = upload_bf16(gpu, &vec![0u16; ws_need * 2]);
    expect_refusal(
        gpu,
        st,
        "workspace not F32",
        "workspace must be F32",
        &base,
        &ws_bf16,
    );
    expect_refusal(
        gpu,
        st,
        "k_a offset 4",
        "offset",
        &TiledAttention {
            k_a: view(4, 16),
            ..base
        },
        &ws,
    );
    expect_refusal(
        gpu,
        st,
        "k_b tok_stride 44",
        "tok_stride",
        &TiledAttention {
            k_b: LatentRows {
                tok_stride: 44,
                ..view(16, 16)
            },
            ..base
        },
        &ws,
    );
    expect_refusal(
        gpu,
        st,
        "v head_stride 4",
        "head_stride",
        &TiledAttention {
            v: LatentRows {
                head_stride: 4,
                ..view(32, 16)
            },
            ..base
        },
        &ws,
    );
    expect_refusal(
        gpu,
        st,
        "k_a width 12",
        "width",
        &TiledAttention {
            k_a: view(0, 12),
            ..base
        },
        &ws,
    );
    expect_refusal(
        gpu,
        st,
        "v width 20",
        "width",
        &TiledAttention {
            v: view(28, 20),
            ..base
        },
        &ws,
    );
    expect_refusal(
        gpu,
        st,
        "max_rows 0",
        "max_rows",
        &TiledAttention {
            max_rows: 0,
            ..base
        },
        &ws,
    );
    expect_refusal(
        gpu,
        st,
        "max_rows beyond the view",
        "view needs",
        &TiledAttention {
            max_rows: rows + 1,
            ..base
        },
        &ws,
    );

    // Width maxima: one past the limit is refused, the limit itself is
    // exercised by the `max` geometry.
    let wide_row = 2112usize;
    let wide = upload_bf16(gpu, &vec![0x3f80u16; 2 * wide_row]);
    let wview = |offset: usize, width: usize| LatentRows {
        buf: &wide,
        offset,
        tok_stride: wide_row,
        head_stride: 0,
        width,
    };
    let wq = gpu.upload_f32(&vec![0.5f32; 4200], &[4200]).unwrap();
    let wy = gpu.upload_f32(&vec![0.0f32; 4200], &[4200]).unwrap();
    let wide_ws_n = latent_attention_decode_workspace_elems(1, 2056);
    let wide_ws = gpu
        .upload_f32(&vec![0.0f32; wide_ws_n], &[wide_ws_n])
        .unwrap();
    let wbase = TiledAttention {
        q: &wq,
        y: &wy,
        k_a: wview(0, 2048),
        k_b: wview(2048, 16),
        v: wview(0, 16),
        pos: &pos,
        pos_offset: 0,
        n_rows: 1,
        n_heads: 1,
        scale: 0.1,
        max_rows: 2,
        round_y_bf16: false,
    };
    expect_refusal(
        gpu,
        st,
        &format!("dq 2064 > {LATENT_DECODE_MAX_DQ}"),
        "query/key width",
        &wbase,
        &wide_ws,
    );
    expect_refusal(
        gpu,
        st,
        &format!("dv 2056 > {LATENT_DECODE_MAX_DV}"),
        "value width",
        &TiledAttention {
            k_a: wview(0, 16),
            k_b: wview(16, 16),
            v: wview(0, 2056),
            ..wbase
        },
        &wide_ws,
    );

    for t in [buf, q, y, pos, ws, ws_small, ws_bf16, wide, wq, wy, wide_ws] {
        gpu.free_tensor(t).expect("free");
    }
}

// ---------------------------------------------------------------------------
// Graph capture

fn graph_check(gpu: &mut Gpu, st: &mut Stats, pos_t: &GpuTensor) {
    println!("== graph capture (blob path) vs eager");
    let geos = geometries();
    let geo = &geos[0];
    let cap = 1100usize;
    let host: Vec<Vec<u16>> = (0..geo.bufs.len())
        .map(|b| gen_host_buf(geo, b, cap, cap))
        .collect();
    let dev: Vec<GpuTensor> = host.iter().map(|h| upload_bf16(gpu, h)).collect();
    let q = gen_q(geo, 4242);
    let q_dev = gpu.upload_f32(&q, &[q.len()]).unwrap();
    let yn = geo.heads * geo.dv();
    let y = gpu.alloc_tensor(&[yn + GUARD], DType::F32).unwrap();
    let ws_need = latent_attention_decode_workspace_elems(geo.heads, geo.dv());
    let ws = gpu.alloc_tensor(&[ws_need + GUARD], DType::F32).unwrap();
    let scale = (geo.dq() as f32).powf(-0.5);
    let a = TiledAttention {
        q: &q_dev,
        y: &y,
        k_a: view_rows(geo, &dev, &geo.ka),
        k_b: view_rows(geo, &dev, &geo.kb),
        v: view_rows(geo, &dev, &geo.v),
        pos: pos_t,
        pos_offset: 0,
        n_rows: 1,
        n_heads: geo.heads,
        scale,
        max_rows: cap,
        round_y_bf16: false,
    };
    // Warm the kernels up (a compile mid-capture is forbidden), then capture
    // while the device scalar holds some unrelated value.
    set_pos(gpu, pos_t, 300);
    gpu.latent_attention_decode_split(&a, &ws)
        .expect("eager warm-up");
    gpu.ensure_capture_stream().expect("capture stream");
    gpu.hip.device_synchronize().expect("sync");
    let dev_id = gpu.device_id;
    gpu.graphs
        .begin_graph_capture(
            &gpu.hip,
            dev_id,
            gpu.active_stream.as_ref().expect("stream"),
        )
        .expect("begin capture");
    let captured = gpu.latent_attention_decode_split(&a, &ws);
    if captured.is_err() {
        gpu.graphs.abort_graph_capture(
            &gpu.hip,
            dev_id,
            gpu.active_stream.as_ref().expect("stream"),
        );
    }
    captured.expect("captured launch");
    gpu.graphs
        .end_graph_capture(
            &gpu.hip,
            dev_id,
            gpu.active_stream.as_ref().expect("stream"),
        )
        .expect("end capture");
    for p in [300usize, 1099, 17, 0, 700, 1099] {
        let label = format!("graph pos={p}");
        set_pos(gpu, pos_t, p as i32);
        fill_words(gpu, &y, SENTINEL);
        fill_words(gpu, &ws, WS_FILL_NAN);
        gpu.graphs
            .graph_launch(
                &gpu.hip,
                dev_id,
                gpu.active_stream.as_ref().expect("stream"),
            )
            .expect("graph replay");
        let g_bits = fetch_bits(gpu, &y);
        fill_words(gpu, &y, SENTINEL);
        fill_words(gpu, &ws, WS_FILL_BIG);
        gpu.latent_attention_decode_split(&a, &ws)
            .expect("eager launch");
        let e_bits = fetch_bits(gpu, &y);
        st.launches += 2;
        st.check(g_bits == e_bits, || {
            format!("{label}: graph replay bytes differ from the eager launch")
        });
        st.identical_comparisons += 1;
        st.check(g_bits[yn..].iter().all(|b| *b == SENTINEL), || {
            format!("{label}: output guard overwritten")
        });
        let want = reference(geo, &host, &q, p + 1, scale);
        let (rel, cos, _) = metrics(&to_f32(&g_bits[..yn]), &want);
        st.worst_rel_fp32 = st.worst_rel_fp32.max(rel);
        st.check(rel <= FP32_REL_L2, || {
            format!("{label}: rel_l2 {rel:.3e} (cos {cos:.9})")
        });
        println!(
            "  {label}: len={} graph==eager bytes={} rel_l2={rel:.2e} cos={cos:.9}",
            p + 1,
            g_bits == e_bits
        );
    }
    gpu.graphs.drop_captured_graph(&gpu.hip, dev_id);
    for t in dev {
        gpu.free_tensor(t).expect("free");
    }
    for t in [q_dev, y, ws] {
        gpu.free_tensor(t).expect("free");
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("GPU: {}", gpu.arch);
    let mut st = Stats::default();
    let pos_t = gpu.alloc_tensor(&[1], DType::F32).expect("alloc pos");

    refusal_checks(&mut gpu, &mut st);
    for geo in geometries() {
        run_geo(&mut gpu, &mut st, &geo, &pos_t);
    }
    graph_check(&mut gpu, &mut st, &pos_t);

    println!(
        "SUMMARY arch={} splits={} launches={} checks={} failures={} byte-identity comparisons={} \
         worst fp32 rel_l2={:.3e} worst bf16 rel_l2={:.3e} min bf16 cos={:.9}",
        gpu.arch,
        LATENT_DECODE_SPLITS,
        st.launches,
        st.checks,
        st.fails,
        st.identical_comparisons,
        st.worst_rel_fp32,
        st.worst_rel_bf16,
        st.worst_cos_bf16
    );
    if st.fails != 0 {
        println!("latent_attention_decode_check: FAILED");
        std::process::exit(1);
    }
    println!("latent_attention_decode_check: PASS");
}
