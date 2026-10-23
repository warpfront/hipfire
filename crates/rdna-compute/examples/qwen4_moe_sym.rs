//! Qwen4 symmetric IU4 MoE ORACLE (fn-moe-sym, gfx1151 + gfx1201; developer
//! example).
//!
//! Drives the production `Gpu` launchers of the opt-in route, the same calls
//! the Qwen4 prefill makes under `HIPFIRE_QWEN4_MOE_SYM_IU4=1`:
//!     moe_scatter_stable_top10 + qwen4_moe_rotate256_i4 (gate A4)
//!     + gemm_qwen4_moe_gate_up_silu_iu4_sym + qwen4_moe_rotate128_i4 (down A4)
//!     + gemm_qwen4_moe_down_iu4_sym
//! over synthetic symmetric QT44/QT53 experts, and checks them against exact
//! host references: a CPU stable sort, the packed-A4 fold
//! `sum = fma(RN(sc*d), float(C_h), sum)`, the shipped SwiGLU expression (on
//! the device, over the CPU folds) and the down sidecar produced by the real
//! `mq_rotate_x_128_v2` plus the shared quantizer. The GEMMs are the certified
//! builder (PM) expert-run entries the launchers select for the residency and the case's real
//! slots (gfx1201 host-mapped: NT8 gate/up and down; resident: gate/up NT8 under
//! `HIPFIRE_MOE_IU4_WIDE_TILES` resolved on (unset on at exact gfx1201) when the slots per expert reach 128, else NT4, and the down
//! always NT4, so a widened resident pair is the mixed `pm_nt8gu_nt4dn`; reported as
//! `candidate_total_slots`), or the gfx1151 hipcc NT4 under
//! `HIPFIRE_QWEN4_MOE_SYM_PM=0`. On gfx1151 PM the down GEMM follows
//! `HIPFIRE_QWEN4_MOE_SYM_DOWN_RR` (1|2|4, row repeat; reported as `down_rr`).
//! Every other entry of the arch (PM NT1/NT4/NT8, down rr2/rr4 `pm_nt4_dr*`,
//! gfx1151 hipcc NT1/NT4) is an anchor: it runs on the same grouping and
//! sidecars and every byte of both output buffers (poison included) must
//! agree with the candidate's. The incumbent
//! (gfx1151: moe_scatter_fused_top10 + mq_rotate_x_f16 + F16 WMMA gate/up
//! SiLU + moe_unscatter_rotate128_f16 + F16 WMMA down; gfx1201: the scatter,
//! the gfx12 F16 WMMA gate/up over the upstream rotated X, the fused
//! unscatter/SiLU/FWHT128 and the SIMT down) runs on the same weights for
//! timing and an informative general-input delta.
//!
//! CLI: --mode check|graph|time|sweep|stress|rows --routing one|all|skew|FILE
//!      [--geometry fn|a3b] [--x FILE.f16] [--tokens T] [--seed N] [--iters N]
//!      [--residency vram|host]
//! Geometry (default `fn`): fn = 512 experts, top-10, hidden 2560, expert
//! intermediate 640; a3b = 256 experts, top-8, hidden 2048, intermediate 512.
//! Every mode but `rows` runs either geometry; `rows` is fn-only (its combine
//! is the top-10 BF16 kernel). The incumbent envelope (informative delta,
//! `time` arm) is the shipped fn route and is skipped for a3b.
//! Routing FILE = little-endian i32 expert ids [T*top_k]; X FILE = F16
//! [T*hidden] (captured activations). `--residency host` places the expert
//! weights in host-mapped memory. Output: one JSON object on stdout.
//! Needs `HIPFIRE_QWEN4_MOE_SYM_IU4=1` only for the route predicate report;
//! the launchers themselves run regardless.
//!
//! a3b on gfx1201 (`--mode check|graph`) also runs the PM BF16-row combine
//! (`Gpu::moe_down_combine_grouped_bf16in_pm`) over the route's own grouped
//! down rows and requires it bitwise equal to the HIP
//! `moe_down_combine_grouped_k8` over the same rows widened BF16 -> F32
//! (natural inverse permutation, and one with every 7th slot unrouted).
//!
//! `--mode rows` (prefill batch invariance, see `rows_mode`): per-token
//! routing (a function of the token's own index), reference at T = 8192
//! tokens, then every T of `--rows` on the first T tokens and the same T
//! tokens at row offsets inside a larger batch of other tokens; each token's
//! final combined MoE output row must be byte-identical to the reference's.
//! One JSON object per run on stdout; exit 1 on any difference.
//!      [--rows LIST] [--offsets LIST] [--scope on|off|both] [--ref-tokens N]
//!      [--distinct N] [--diag] [--print all|fail]

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor, Int4MmqPrepared};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// Expert geometry of this run (`--geometry`). QT44 gate/up and QT53 down
/// group bytes are shared by every geometry.
#[derive(Clone, Copy)]
struct Geo {
    name: &'static str,
    e: usize,
    topk: usize,
    hid: usize,
    mi: usize,
    gu_m: usize,
    dn_m: usize,
    gu_stride: usize,
    dn_stride: usize,
}

impl Geo {
    const fn new(name: &'static str, e: usize, topk: usize, hid: usize, mi: usize) -> Geo {
        Geo {
            name,
            e,
            topk,
            hid,
            mi,
            gu_m: 2 * mi,
            dn_m: hid,
            gu_stride: 2 * mi * (hid / 256) * GU_GB,
            dn_stride: hid * (mi / 128) * DN_GB,
        }
    }
}

const GEO_FN: Geo = Geo::new("fn", 512, 10, 2560, 640);
const GEO_A3B: Geo = Geo::new("a3b", 256, 8, 2048, 512);
/// Set once from `--geometry` (runtime input, so not a `LazyLock`).
static GEO: OnceLock<Geo> = OnceLock::new();

/// The selected geometry (`fn` until `main` selects another).
fn geo() -> &'static Geo {
    GEO.get().unwrap_or(&GEO_FN)
}

const GU_GB: usize = 136;
const DN_GB: usize = 68;
const BLK: usize = 72;
const GUARD: usize = 4096;

const QUANT: &str = include_str!("../../../kernels/src/block_i4_128_quant.hip");
/// The gfx1151 hipcc GEMMs (production source `kernels::QWEN4_MOE_IU4_SYM_GFX1151_SRC`).
const K11: &str = include_str!("../../../kernels/src/qwen4_moe_iu4_sym.gfx1151.hip");
const ROT128: &str = include_str!("../../../kernels/src/mq_rotate_x_128_v2.hip");
/// Oracle-only kernels: the shared quantizer over stored FWHT rows (the
/// down-sidecar reference), a lossless BF16 widen, and the shipped SwiGLU
/// expression over host-exact gate/up folds.
const ORACLE_KERNELS: &str = r#"
extern "C" __global__ __launch_bounds__(32) void qwen4_moe_sym_ref_quant(
    const float* __restrict__ rot, const int* __restrict__ sorted,
    block_i4_128* __restrict__ Xq, int K, int P, int L) {
    const int group = blockIdx.x;
    const int p = blockIdx.y;
    const int slot = sorted[p];
    if (slot < 0) return;
    const int tid = threadIdx.x;
    float xv[4];
    for (int i = 0; i < 4; ++i) xv[i] = rot[(size_t)p * K + group * 128 + tid * 4 + i];
    quantize_block_i4_128_wave<true>(xv, Xq + (size_t)group * L + slot, tid);
}
extern "C" __global__ void qwen4_moe_sym_ref_widen(const unsigned short* h, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((unsigned int)h[i] << 16);
}
__device__ __forceinline__ unsigned short r_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float r_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}
extern "C" __global__ void qwen4_moe_sym_ref_swiglu(const float* g, const float* u, const int* live, unsigned short* out, int n, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = r_rt(g[i]);
    const float w = r_rt(u[i]);
    out[i] = live[i / cols] ? r_bits(r_rt((v / (1.0f + expf(-v))) * w)) : (unsigned short)0;
}
"#;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------- host math

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn f16_to_f32(b: u16) -> f32 {
    let s = ((b >> 15) as u32) << 31;
    let e = ((b >> 10) & 0x1f) as u32;
    let m = (b & 0x3ff) as u32;
    let v = if e == 0 {
        if m == 0 {
            return f32::from_bits(s);
        }
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        return f32::from_bits(s | 0x7f80_0000 | (m << 13));
    } else {
        f32::from_bits(((e + 112) << 23) | (m << 13))
    };
    if s != 0 {
        -v
    } else {
        v
    }
}

/// Symmetric header pair (sc, zp) with zp == -8*sc exactly: independently
/// varied normal scales of both signs, zero groups with both zero signs.
fn sym_header(r: &mut Rng) -> (u16, u16) {
    match r.below(64) {
        0 => (0x0000, 0x8000),
        1 => (0x8000, 0x0000),
        _ => {
            let sign = (r.below(2) as u16) << 15;
            let exp = 3 + r.below(9) as u16; // 2^-12 .. 2^-4
            let man = r.below(1024) as u16;
            let sc = sign | (exp << 10) | man;
            let zp = (sign ^ 0x8000) | ((exp + 3) << 10) | man;
            (sc, zp)
        }
    }
}

fn headers_per_group(group_bytes: usize) -> usize {
    if group_bytes == GU_GB {
        2
    } else {
        1
    }
}

/// One expert's rows: QT44 (group_bytes 136, K%256) or QT53 (68, K%128),
/// every header symmetric, every nibble code drawn uniformly.
pub fn symmetric_weights(m: usize, k: usize, group_bytes: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed ^ 0x5eed_0000_0000_0001);
    let groups = if group_bytes == GU_GB {
        k / 256
    } else {
        k / 128
    };
    let mut w = vec![0u8; m * groups * group_bytes];
    for g in w.chunks_exact_mut(group_bytes) {
        let hdrs = headers_per_group(group_bytes);
        for h in 0..hdrs {
            let (sc, zp) = sym_header(&mut r);
            g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for c in g[4 * hdrs..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
    w
}

/// Byte offsets of K element `kk`'s header and nibble in row `row`.
fn locate(row: usize, kk: usize, k: usize, group_bytes: usize) -> (usize, usize) {
    let h = kk / 128;
    if group_bytes == GU_GB {
        let g = row * (k / 256) * GU_GB + (h / 2) * GU_GB;
        (g + 4 * (h % 2), g + 8 + 64 * (h % 2) + (kk % 128) / 2)
    } else {
        let g = row * (k / 128) * DN_GB + h * DN_GB;
        (g, g + 4 + (kk % 128) / 2)
    }
}

/// Exact packed-A4 fold: `xq` holds n tokens' blocks laid [k/128][n] (72 B);
/// returns [n][m] F32 sums with sum = fma(RN(sc*d), float(C_h), sum), +0 start,
/// ascending epochs. Integer dots skip zero terms on the sparser side, which
/// cannot change an exact integer sum.
pub fn fold_reference(
    w: &[u8],
    xq: &[u8],
    m: usize,
    k: usize,
    n: usize,
    group_bytes: usize,
) -> Vec<f32> {
    let epochs = k / 128;
    let mut xs: Vec<Vec<(u8, i32)>> = vec![Vec::new(); epochs * n];
    let mut xd = vec![0f32; epochs * n];
    for h in 0..epochs {
        for j in 0..n {
            let b = &xq[(h * n + j) * BLK..(h * n + j + 1) * BLK];
            xd[h * n + j] = f32::from_le_bytes(b[0..4].try_into().unwrap());
            for i in 0..128usize {
                let byte = b[8 + i / 2];
                let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                let v = ((nib << 4) as i8 >> 4) as i32;
                if v != 0 {
                    xs[h * n + j].push((i as u8, v));
                }
            }
        }
    }
    let mut out = vec![0f32; n * m];
    let mut ws = [0i32; 128];
    let mut wnz: Vec<(u8, i32)> = Vec::with_capacity(128);
    for row in 0..m {
        let mut sums = vec![0f32; n];
        for h in 0..epochs {
            let (hdr, _) = locate(row, h * 128, k, group_bytes);
            let sc = f16_to_f32(u16::from_le_bytes([w[hdr], w[hdr + 1]]));
            wnz.clear();
            for i in 0..128 {
                let (_, nb) = locate(row, h * 128 + i, k, group_bytes);
                let byte = w[nb];
                ws[i] = if i % 2 == 0 {
                    (byte & 15) as i32 - 8
                } else {
                    (byte >> 4) as i32 - 8
                };
                if ws[i] != 0 {
                    wnz.push((i as u8, ws[i]));
                }
            }
            for j in 0..n {
                let x = &xs[h * n + j];
                let c: i32 = if x.len() <= wnz.len() {
                    x.iter().map(|&(i, v)| ws[i as usize] * v).sum()
                } else {
                    let mut xv = [0i32; 128];
                    for &(i, v) in x {
                        xv[i as usize] = v;
                    }
                    wnz.iter().map(|&(i, v)| xv[i as usize] * v).sum()
                };
                let scale = sc * xd[h * n + j];
                sums[j] = scale.mul_add(c as f32, sums[j]);
            }
        }
        for j in 0..n {
            out[j * m + row] = sums[j];
        }
    }
    out
}

fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if !v.is_finite() {
        return (u >> 16) as u16;
    }
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}

fn md5_hex(b: &[u8]) -> String {
    let s: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let kk: Vec<u32> = (0..64)
        .map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32)
        .collect();
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut msg = b.to_vec();
    let bitlen = (b.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_le_bytes());
    for ch in msg.chunks_exact(64) {
        let m: Vec<u32> = ch
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        let (mut a, mut b1, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b1 & c) | (!b1 & d), i),
                1 => ((d & b1) | (!d & c), (5 * i + 1) % 16),
                2 => (b1 ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b1 | !d), (7 * i) % 16),
            };
            let f2 = f.wrapping_add(a).wrapping_add(kk[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b1;
            b1 = b1.wrapping_add(f2.rotate_left(s[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b1);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    [a0, b0, c0, d0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// CPU stable grouping: (expert, flat) order, pad16, padded counts.
struct Grouping {
    counts: Vec<i32>,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    inverse: Vec<i32>,
}
fn cpu_group(topk: &[i32], pmax: usize) -> Grouping {
    let l = topk.len();
    let mut raw = vec![0i32; geo().e];
    for &e in topk {
        raw[e as usize] += 1;
    }
    let counts: Vec<i32> = raw.iter().map(|c| (c + 15) / 16 * 16).collect();
    let mut offsets = vec![0i32; geo().e + 1];
    for e in 0..geo().e {
        offsets[e + 1] = offsets[e] + counts[e];
    }
    let mut sorted = vec![-1i32; pmax];
    let mut tiles = vec![-1i32; pmax / 16];
    let mut inverse = vec![0i32; l];
    let mut fill = offsets.clone();
    for (f, &e) in topk.iter().enumerate() {
        let p = fill[e as usize] as usize;
        fill[e as usize] += 1;
        sorted[p] = f as i32;
        inverse[f] = p as i32;
    }
    for e in 0..geo().e {
        for t in offsets[e] / 16..offsets[e + 1] / 16 {
            tiles[t as usize] = e as i32;
        }
    }
    Grouping {
        counts,
        offsets,
        sorted,
        tiles,
        inverse,
    }
}

fn routing(kind: &str, t: usize, seed: u64) -> Res<Vec<i32>> {
    let mut r = Rng(seed ^ 0xabcdef);
    Ok(match kind {
        // Every slot to expert 0: the hottest-possible single bucket.
        "one" => vec![0; t * geo().topk],
        // Round robin over all experts.
        "all" => (0..t * geo().topk).map(|i| (i % geo().e) as i32).collect(),
        // Five hot experts on every token plus five distinct uniform others.
        "skew" => {
            let mut v = Vec::with_capacity(t * geo().topk);
            for _ in 0..t {
                let mut row: Vec<i32> = (0..5).collect();
                while row.len() < geo().topk {
                    let e = 5 + r.below((geo().e - 5) as u64) as i32;
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                v.extend(row);
            }
            v
        }
        path => {
            let b = std::fs::read(path)?;
            let v: Vec<i32> = b
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            if v.len() < t * geo().topk {
                return Err(format!(
                    "routing file {path} has {} slots < {}",
                    v.len(),
                    t * geo().topk
                )
                .into());
            }
            v[..t * geo().topk].to_vec()
        }
    })
}

// ---------------------------------------------------------------- device

fn launch(gpu: &Gpu, f: &str, grid: [u32; 3], block: u32, args: &mut KernargBlob) -> Res<()> {
    args.pad_to(16);
    gpu.launch_kernel_blob(f, grid, [block, 1, 1], 0, args.as_mut_slice())?;
    Ok(())
}

fn p(t: &GpuTensor) -> *const std::ffi::c_void {
    t.buf.as_ptr() as *const _
}

fn alloc(gpu: &mut Gpu, bytes: usize) -> Res<GpuTensor> {
    Ok(gpu.zeros(&[bytes.div_ceil(4)], DType::F32)?)
}

fn download(gpu: &Gpu, t: &hip_bridge::DeviceBuffer, bytes: usize) -> Res<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, t)?;
    Ok(v)
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn to_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn ptr_table(base: &GpuTensor, stride: usize, distinct: usize) -> Vec<u8> {
    (0..geo().e)
        .flat_map(|e| ((base.buf.as_ptr() as u64) + ((e % distinct) * stride) as u64).to_le_bytes())
        .collect()
}

/// Weights, operands and every output buffer of one case.
struct Case {
    t: usize,
    l: usize,
    pmax: usize,
    topk_host: Vec<i32>,
    gu_w: Vec<u8>,
    dn_w: Vec<u8>,
    gu_distinct: usize,
    dn_distinct: usize,
    gu_t: GpuTensor,
    dn_t: GpuTensor,
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
    topk: GpuTensor,
    x_in: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    ygu: GpuTensor,
    ydn: GpuTensor,
    i_counts: GpuTensor,
    i_offsets: GpuTensor,
    i_sorted: GpuTensor,
    i_tiles: GpuTensor,
    i_inverse: GpuTensor,
    i_ygu: GpuTensor,
    i_ydn: GpuTensor,
    /// gfx1201 incumbent operands: upstream rotated X, compact SwiGLU rows.
    x_rot: GpuTensor,
    i_rot: GpuTensor,
    /// Anchor GEMM outputs.
    h_ygu: GpuTensor,
    h_ydn: GpuTensor,
    /// Expert weights in host-mapped memory (`--residency host`).
    host: bool,
}

#[allow(clippy::too_many_arguments)]
fn make_case(
    gpu: &mut Gpu,
    topk: Vec<i32>,
    x: &[f32],
    gu_w: Vec<u8>,
    dn_w: Vec<u8>,
    host: bool,
) -> Res<Case> {
    let l = topk.len();
    let t = l / geo().topk;
    let pmax = l + geo().e * 15;
    let pmax = pmax.div_ceil(16) * 16;
    let gu_distinct = gu_w.len() / geo().gu_stride;
    let dn_distinct = dn_w.len() / geo().dn_stride;
    let (gu_t, dn_t) = if host {
        (
            gpu.upload_raw_host_mapped(&gu_w, &[gu_w.len()])?,
            gpu.upload_raw_host_mapped(&dn_w, &[dn_w.len()])?,
        )
    } else {
        (
            gpu.upload_raw(&gu_w, &[gu_w.len()])?,
            gpu.upload_raw(&dn_w, &[dn_w.len()])?,
        )
    };
    let gu_ptrs = gpu.upload_raw(
        &ptr_table(&gu_t, geo().gu_stride, gu_distinct),
        &[geo().e * 8],
    )?;
    let dn_ptrs = gpu.upload_raw(
        &ptr_table(&dn_t, geo().dn_stride, dn_distinct),
        &[geo().e * 8],
    )?;
    Ok(Case {
        t,
        l,
        pmax,
        topk: gpu.upload_raw(&i32_bytes(&topk), &[l])?,
        topk_host: topk,
        gu_w,
        dn_w,
        gu_distinct,
        dn_distinct,
        gu_t,
        dn_t,
        gu_ptrs,
        dn_ptrs,
        x_in: gpu.upload_f32(x, &[t * geo().hid])?,
        counts: alloc(gpu, geo().e * 4)?,
        offsets: alloc(gpu, (geo().e + 1) * 4)?,
        sorted: alloc(gpu, pmax * 4)?,
        tiles: alloc(gpu, pmax / 16 * 4)?,
        inverse: alloc(gpu, l * 4)?,
        ygu: alloc(gpu, pmax * geo().mi * 2 + GUARD)?,
        ydn: alloc(gpu, pmax * geo().dn_m * 2 + GUARD)?,
        i_counts: alloc(gpu, geo().e * 4)?,
        i_offsets: alloc(gpu, (geo().e + 1) * 4)?,
        i_sorted: alloc(gpu, pmax * 4)?,
        i_tiles: alloc(gpu, pmax / 16 * 4)?,
        i_inverse: alloc(gpu, l * 4)?,
        i_ygu: alloc(gpu, pmax * geo().gu_m * 4)?,
        i_ydn: alloc(gpu, pmax * geo().dn_m * 4)?,
        x_rot: alloc(gpu, t * geo().hid * 4)?,
        i_rot: alloc(gpu, l * geo().mi * 4)?,
        h_ygu: alloc(gpu, pmax * geo().mi * 2 + GUARD)?,
        h_ydn: alloc(gpu, pmax * geo().dn_m * 2 + GUARD)?,
        host,
    })
}

/// The route's five stages, handles carried between them; `entry` names the
/// anchor GEMMs of [`anchor_stage`].
#[derive(Default)]
struct Handles {
    gate: Option<Int4MmqPrepared>,
    down: Option<Int4MmqPrepared>,
    entry: Option<Entry>,
}

fn cand_stage(gpu: &mut Gpu, c: &Case, h: &mut Handles, s: usize) -> Res<()> {
    match s {
        0 => gpu.moe_scatter_stable_top10(
            &c.topk,
            &c.counts,
            &c.offsets,
            &c.sorted,
            &c.tiles,
            &c.inverse,
            c.l,
            geo().e,
            c.pmax,
        )?,
        1 => h.gate = Some(gpu.qwen4_moe_rotate256_i4(&c.x_in, geo().hid, c.t)?),
        2 => gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            &c.gu_ptrs,
            &c.tiles,
            &c.sorted,
            h.gate.as_ref().ok_or("gate A4 missing")?,
            &c.ygu,
            geo().gu_m,
            geo().hid,
            geo().topk,
            c.pmax,
            c.t,
            c.host,
        )?,
        // The route's producer: the grouped-position plane where the PM twins apply (the dispatch call),
        // else the flat-slot sidecar. The anchors below always rebuild the flat sidecar.
        3 => {
            h.down =
                Some(gpu.qwen4_moe_rotate128_i4_gp(&c.ygu, &c.sorted, geo().mi, c.pmax, c.l)?)
        }
        _ => gpu.gemm_qwen4_moe_down_iu4_sym(
            &c.dn_ptrs,
            &c.tiles,
            &c.sorted,
            h.down.as_ref().ok_or("down A4 missing")?,
            &c.ydn,
            geo().dn_m,
            geo().mi,
            1,
            c.pmax,
            c.l,
            c.host,
        )?,
    }
    Ok(())
}

fn cand_all(gpu: &mut Gpu, c: &Case) -> Res<()> {
    let mut h = Handles::default();
    for s in 0..5 {
        cand_stage(gpu, c, &mut h, s)?;
    }
    Ok(())
}

/// One GEMM pair the route can run: the builder (PM) entries at each
/// expert-run tile width and, on gfx1151, the hipcc entries (the production
/// module `kernels::QWEN4_MOE_IU4_SYM_GFX1151_SRC`). NT1 gate/up runs block
/// 64; every NT>1 entry and every down runs block 128; the grid is
/// ceil(M/64) x P/16 for all, except a down with row repeat `down_rr` > 1
/// (gfx1151 PM `_nt4r2`/`_nt4r4`), whose grid.x is M / (64 * down_rr).
/// Every entry must produce the same bytes.
#[derive(Clone, Copy)]
struct Entry {
    name: &'static str,
    gate_up: &'static str,
    down: &'static str,
    gu_block: u32,
    pm: bool,
    down_rr: usize,
}

const fn entry(
    name: &'static str,
    gate_up: &'static str,
    down: &'static str,
    gu_block: u32,
    pm: bool,
) -> Entry {
    Entry {
        name,
        gate_up,
        down,
        gu_block,
        pm,
        down_rr: 1,
    }
}

const fn entry_rr(
    name: &'static str,
    gate_up: &'static str,
    down: &'static str,
    down_rr: usize,
) -> Entry {
    Entry {
        name,
        gate_up,
        down,
        gu_block: 128,
        pm: true,
        down_rr,
    }
}

const ENTRIES_GFX1151: [Entry; 9] = [
    entry(
        "pm_nt1",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151",
        "qwen4_moe_down_iu4_sym_pm_gfx1151",
        64,
        true,
    ),
    entry(
        "pm_nt4",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4",
        128,
        true,
    ),
    entry(
        "hip_nt1",
        "qwen4_moe_gate_up_silu_iu4_sym_gfx1151",
        "qwen4_moe_down_iu4_sym_gfx1151",
        64,
        false,
    ),
    entry(
        "hip_nt4",
        "qwen4_moe_gate_up_silu_iu4_sym_gfx1151_nt4",
        "qwen4_moe_down_iu4_sym_gfx1151_nt4",
        128,
        false,
    ),
    entry_rr(
        "pm_nt4_dr2",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r2",
        2,
    ),
    entry_rr(
        "pm_nt4_dr4",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r4",
        4,
    ),
    entry(
        "pm_nt8_stream2",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt8_stream2",
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4",
        128,
        true,
    ),
    entry_rr(
        "pm_nt8_stream2_dr2",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt8_stream2",
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r2",
        2,
    ),
    entry_rr(
        "pm_nt8_stream2_dr4",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt8_stream2",
        "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4r4",
        4,
    ),
];
const ENTRIES_GFX1201: [Entry; 3] = [
    entry(
        "pm_nt1",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201",
        "qwen4_moe_down_iu4_sym_pm_gfx1201",
        64,
        true,
    ),
    entry(
        "pm_nt4",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
        "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
        128,
        true,
    ),
    entry(
        "pm_nt8",
        "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
        "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8",
        128,
        true,
    ),
];

/// The resident gfx1201 pair the launchers run for a widened expert run
/// (`HIPFIRE_MOE_IU4_WIDE_TILES` resolved on, slots per expert >= 128): the NT8 gate/up with the incumbent
/// NT4 down. Not a row of [`ENTRIES_GFX1201`] (that table keeps the uniform NT1/NT4/NT8 anchors),
/// so [`cand_and_anchors`] matches the route's symbols against it after the table; the symbols
/// are the table's own `pm_nt8` gate/up and `pm_nt4` down. Every uniform entry stays an anchor.
const MIXED_GFX1201: [Entry; 1] = [entry(
    "pm_nt8gu_nt4dn",
    "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
    "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
    128,
    true,
)];
const HIP_SYM_MODULE: &str = "qwen4_moe_iu4_sym_gfx1151";

fn entries(gpu: &Gpu) -> &'static [Entry] {
    if gpu.arch == "gfx1201" {
        &ENTRIES_GFX1201
    } else {
        &ENTRIES_GFX1151
    }
}

/// Loads every entry of this arch.
fn load_entries(gpu: &mut Gpu) -> Res<()> {
    let src = format!("#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n{QUANT}{K11}");
    for e in entries(gpu) {
        for f in [e.gate_up, e.down] {
            if e.pm {
                gpu.ensure_qwen4_moe_sym_pm_entry(f)?;
            } else {
                gpu.ensure_kernel_public(HIP_SYM_MODULE, &src, f)?;
            }
        }
    }
    Ok(())
}

/// The candidate (the entry the launchers select for this residency and for a grouped call of
/// `total_slots` real routed slots over the geometry's `geo().e` expert-table entries, down row
/// repeat included) and its anchors: every other entry of the arch. The width of a resident
/// gfx1201 gate/up run follows `HIPFIRE_MOE_IU4_WIDE_TILES` and the slots per expert (the down
/// entry stays NT4 resident), so the shape is part of the question and the route's pair can be
/// the mixed [`MIXED_GFX1201`] entry, matched by its two reported symbols.
fn cand_and_anchors(gpu: &Gpu, host: bool, total_slots: usize) -> Res<(Entry, Vec<Entry>)> {
    let [gu, dn] = gpu
        .qwen4_moe_sym_gemm_symbols(host, total_slots, geo().e)
        .ok_or("no sym GEMM symbols on this arch")?;
    let gu = gpu
        .qwen4_moe_sym_gate_up_symbol(
            host,
            [
                geo().gu_m,
                geo().hid,
                geo().topk,
                total_slots / geo().topk,
                geo().e,
            ],
        )
        .unwrap_or(gu);
    let rr = gpu.qwen4_moe_sym_down_rr()?;
    let all = entries(gpu);
    let mixed: &[Entry] = if gpu.arch == "gfx1201" {
        &MIXED_GFX1201
    } else {
        &[]
    };
    let cand = *all
        .iter()
        .chain(mixed)
        .find(|e| e.gate_up == gu && e.down == dn)
        .ok_or("route GEMM is not an ORACLE entry")?;
    if cand.down_rr != rr {
        return Err(format!(
            "route down row repeat {rr} != ORACLE entry {} row repeat {}",
            cand.name, cand.down_rr
        )
        .into());
    }
    // Where the route runs the grouped-position `..._gp` twins, the flat entry of the same name is
    // not the candidate and stays an anchor (the byte-identity claim is twin vs original).
    let gp = gpu.qwen4_moe_rotate128_i4_gp_selected(geo().mi, total_slots, total_slots);
    Ok((
        cand,
        all.iter()
            .copied()
            .filter(|e| gp || e.name != cand.name)
            .collect(),
    ))
}

/// [`cand_and_anchors`] for one case of `total_slots` real routed slots, recorded in `out`: the
/// candidate's symbols, entry, down row repeat, residency and the anchor names. Returns the
/// candidate and the anchors.
fn report_candidate(
    gpu: &Gpu,
    host: bool,
    total_slots: usize,
    residency: &str,
    out: &mut BTreeMap<String, String>,
) -> Res<(Entry, Vec<Entry>)> {
    let q = |s: &str| format!("\"{s}\"");
    let (cand, anchors) = cand_and_anchors(gpu, host, total_slots)?;
    out.insert(
        "gemm_symbols".into(),
        format!("[{},{}]", q(cand.gate_up), q(cand.down)),
    );
    out.insert("candidate_entry".into(), q(cand.name));
    out.insert(
        "down_plane".into(),
        q(
            if gpu.qwen4_moe_rotate128_i4_gp_selected(geo().mi, total_slots, total_slots) {
                "grouped_position_gp_twins"
            } else {
                "flat_slot"
            },
        ),
    );
    out.insert("down_rr".into(), cand.down_rr.to_string());
    out.insert("residency".into(), q(residency));
    out.insert("candidate_total_slots".into(), total_slots.to_string());
    out.insert(
        "moe_iu4_wide_tiles".into(),
        gpu.flags.moe_iu4_wide_tiles.to_string(),
    );
    let names: Vec<String> = anchors.iter().map(|e| q(e.name)).collect();
    out.insert("anchors".into(), format!("[{}]", names.join(",")));
    Ok((cand, anchors))
}

fn diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Entry `e`'s GEMM `down` (else gate/up) from the current grouping and
/// production sidecar into `y` ([`load_entries`] first).
fn entry_gemm(gpu: &mut Gpu, c: &Case, e: Entry, down: bool, y: &GpuTensor) -> Res<()> {
    let (f, ptrs, xq, m, k, div, src_rows, block, rr) = if down {
        let xq = gpu
            .scratch
            .qwen4_moe_down_i4_scratch
            .as_ref()
            .ok_or("no down sidecar")?
            .as_ptr();
        (
            e.down,
            &c.dn_ptrs,
            xq,
            geo().dn_m,
            geo().mi,
            1,
            c.l,
            128,
            e.down_rr,
        )
    } else {
        let xq = gpu
            .scratch
            .int4_mmq_x_scratch
            .as_ref()
            .ok_or("no gate sidecar")?
            .as_ptr();
        (
            e.gate_up,
            &c.gu_ptrs,
            xq,
            geo().gu_m,
            geo().hid,
            geo().topk,
            c.t,
            e.gu_block,
            1,
        )
    };
    if m % (64 * rr) != 0 {
        return Err(format!("{f}: m {m} is not a multiple of 64 * {rr}").into());
    }
    let mut a = KernargBlob::new();
    for ptr in [p(ptrs), p(&c.tiles), p(&c.sorted), xq as *const _, p(y)] {
        a.push_ptr(ptr);
    }
    for v in [m, k, div, c.pmax, src_rows, ptrs.buf.size() / 8] {
        a.push_i32(v as i32);
    }
    launch(
        gpu,
        f,
        [(m / (64 * rr)) as u32, (c.pmax / 16) as u32, 1],
        block,
        &mut a,
    )
}

/// Every anchor's GEMMs over the current grouping and gate sidecar (its own
/// down sidecar from its own gate/up) into outputs poisoned with `v`, as the
/// candidate's were; per anchor, byte mismatches of both whole buffers
/// (poison and guard included) vs `ygu`, `ydn`.
fn anchor_mismatch(
    gpu: &mut Gpu,
    c: &Case,
    anchors: &[Entry],
    v: i32,
    ygu: &[u8],
    ydn: &[u8],
) -> Res<Vec<(&'static str, [usize; 2])>> {
    let mut out = Vec::with_capacity(anchors.len());
    for &e in anchors {
        gpu.hip
            .memset(&c.h_ygu.buf, v, c.pmax * geo().mi * 2 + GUARD)?;
        gpu.hip
            .memset(&c.h_ydn.buf, v, c.pmax * geo().dn_m * 2 + GUARD)?;
        let mut h = Handles {
            entry: Some(e),
            ..Handles::default()
        };
        for s in 2..5 {
            anchor_stage(gpu, c, &mut h, s)?;
        }
        gpu.hip.device_synchronize()?;
        let a = download(gpu, &c.h_ygu.buf, c.pmax * geo().mi * 2 + GUARD)?;
        let d = download(gpu, &c.h_ydn.buf, c.pmax * geo().dn_m * 2 + GUARD)?;
        out.push((e.name, [diff(&a, ygu), diff(&d, ydn)]));
    }
    Ok(out)
}

fn anchor_json(m: &[(&str, [usize; 2])]) -> String {
    let body: Vec<String> = m
        .iter()
        .map(|(n, [a, b])| format!("\"{n}\":[{a},{b}]"))
        .collect();
    format!("{{{}}}", body.join(","))
}

fn anchor_total(m: &[(&str, [usize; 2])]) -> usize {
    m.iter().map(|(_, [a, b])| a + b).sum()
}

/// The route with the GEMMs of `h.entry` (production grouping, producers).
fn anchor_stage(gpu: &mut Gpu, c: &Case, h: &mut Handles, s: usize) -> Res<()> {
    let e = h.entry.ok_or("anchor stage without an entry")?;
    match s {
        2 => entry_gemm(gpu, c, e, false, &c.h_ygu),
        3 => {
            h.down =
                Some(gpu.qwen4_moe_rotate128_i4(&c.h_ygu, &c.sorted, geo().mi, c.pmax, c.l)?);
            Ok(())
        }
        4 => entry_gemm(gpu, c, e, true, &c.h_ydn),
        _ => cand_stage(gpu, c, h, s),
    }
}

/// The shipped incumbent envelope on the same operands (per arch).
fn inc_stage(gpu: &mut Gpu, c: &Case, _h: &mut Handles, s: usize) -> Res<()> {
    let gfx1201 = gpu.arch == "gfx1201";
    match s {
        0 => gpu.moe_scatter_fused_top10(
            &c.topk,
            &c.i_counts,
            &c.i_offsets,
            &c.i_sorted,
            &c.i_tiles,
            &c.i_inverse,
            c.l,
            geo().e,
            c.pmax,
            16,
        )?,
        // gfx1201 consumes the upstream rotated X (`x_rot`, set up once).
        2 if gfx1201 => gpu.gemm_mq4g256v2_moe_grouped_top10(
            &c.gu_ptrs,
            &c.i_tiles,
            &c.i_sorted,
            &c.x_rot,
            &c.i_ygu,
            geo().gu_m,
            geo().hid,
            geo().topk,
            c.pmax,
            c.t,
        )?,
        2 => {
            let xf16 = gpu.rotate_x_mq_batched_f16(&c.x_in, geo().hid, c.t)?;
            gpu.gemm_mq4g256v2_moe_grouped_top10_silu_bf16out(
                &c.gu_ptrs,
                &c.i_tiles,
                &c.i_sorted,
                &xf16,
                &c.i_ygu,
                geo().gu_m,
                geo().hid,
                geo().topk,
                c.pmax,
                c.t,
            )?;
        }
        3 if gfx1201 => gpu.moe_gate_up_unscatter_silu_rotate128_top10(
            &c.i_ygu,
            &c.i_sorted,
            &c.i_rot,
            geo().mi,
            c.pmax,
            true,
            None,
        )?,
        4 if gfx1201 => gpu.gemm_mq4g128v2_moe_grouped_top10(
            &c.dn_ptrs,
            &c.i_tiles,
            &c.i_sorted,
            &c.i_rot,
            &c.i_ydn,
            geo().dn_m,
            geo().mi,
            1,
            c.pmax,
            c.l,
            geo().e,
        )?,
        4 => {
            let x16 =
                gpu.moe_unscatter_rotate128_f16(&c.i_ygu, &c.i_sorted, geo().mi, c.pmax, c.l)?;
            gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
                &c.dn_ptrs,
                &c.i_tiles,
                &c.i_sorted,
                &x16,
                &c.i_ydn,
                geo().dn_m,
                geo().mi,
                c.pmax,
                true,
            )?;
        }
        _ => {}
    }
    Ok(())
}

/// gfx1201 incumbent setup: the production `mq_rotate_x` output it consumes
/// (upstream of the MoE in the shipped route, not charged to either arm).
fn inc_setup(gpu: &mut Gpu, c: &Case) -> Res<()> {
    if gpu.arch == "gfx1201" {
        let r = gpu.reserve_int4_mmq(geo().hid, c.t)?;
        gpu.rotate_x_mq_i4_batched(&c.x_in, None, Some(&c.x_rot), r, geo().hid, c.t)?;
        gpu.hip.device_synchronize()?;
    }
    Ok(())
}

type Stage = fn(&mut Gpu, &Case, &mut Handles, usize) -> Res<()>;

fn timed(gpu: &mut Gpu, c: &Case, f: Stage, entry: Option<Entry>) -> Res<(Vec<f64>, f64)> {
    let ev: Vec<_> = (0..6)
        .map(|_| gpu.hip.event_create())
        .collect::<Result<_, _>>()?;
    let mut h = Handles {
        entry,
        ..Handles::default()
    };
    gpu.hip.event_record(&ev[0], None)?;
    for s in 0..5 {
        f(gpu, c, &mut h, s)?;
        gpu.hip.event_record(&ev[s + 1], None)?;
    }
    gpu.hip.event_synchronize(&ev[5])?;
    let st: Vec<f64> = (0..5)
        .map(|s| {
            gpu.hip
                .event_elapsed_ms(&ev[s], &ev[s + 1])
                .map(|x| x as f64)
        })
        .collect::<Result<_, _>>()?;
    Ok((st, gpu.hip.event_elapsed_ms(&ev[0], &ev[5])? as f64))
}

fn stats(v: &[f64]) -> String {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!("[{:.4},{:.4},{:.4}]", s[0], s[s.len() / 2], s[s.len() - 1])
}

fn poison(gpu: &Gpu, c: &Case, v: i32) -> Res<()> {
    gpu.hip
        .memset(&c.ygu.buf, v, c.pmax * geo().mi * 2 + GUARD)?;
    gpu.hip
        .memset(&c.ydn.buf, v, c.pmax * geo().dn_m * 2 + GUARD)?;
    Ok(())
}

fn outputs(gpu: &Gpu, c: &Case) -> Res<(Vec<u8>, Vec<u8>)> {
    Ok((
        download(gpu, &c.ygu.buf, c.pmax * geo().mi * 2 + GUARD)?,
        download(gpu, &c.ydn.buf, c.pmax * geo().dn_m * 2 + GUARD)?,
    ))
}

/// Live-tile bytes of both outputs (what the combine can read).
fn live_bytes(g: &Grouping, ygu: &[u8], ydn: &[u8]) -> Vec<u8> {
    let used = g.offsets[geo().e] as usize;
    [&ygu[..used * geo().mi * 2], &ydn[..used * geo().dn_m * 2]].concat()
}

fn load_oracle(gpu: &mut Gpu) -> Res<()> {
    let src =
        format!("#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1\n{QUANT}{ORACLE_KERNELS}");
    for f in [
        "qwen4_moe_sym_ref_quant",
        "qwen4_moe_sym_ref_widen",
        "qwen4_moe_sym_ref_swiglu",
    ] {
        gpu.ensure_kernel_public("qwen4_moe_sym_oracle", &src, f)?;
    }
    gpu.ensure_kernel_public("qwen4_moe_sym_oracle_rot128", ROT128, "mq_rotate_x_128_v2")?;
    load_entries(gpu)
}

/// Device SwiGLU of host-exact gate/up folds (`live` per row), BF16 bits.
fn ref_swiglu(gpu: &mut Gpu, g: &[f32], u: &[f32], live: &[i32]) -> Res<Vec<u16>> {
    let n = g.len();
    let gt = gpu.upload_f32(g, &[n])?;
    let ut = gpu.upload_f32(u, &[n])?;
    let lt = gpu.upload_raw(&i32_bytes(live), &[live.len()])?;
    let ot = alloc(gpu, n * 2)?;
    let mut a = KernargBlob::new();
    a.push_ptr(p(&gt));
    a.push_ptr(p(&ut));
    a.push_ptr(p(&lt));
    a.push_ptr(p(&ot));
    a.push_i32(n as i32);
    a.push_i32(geo().mi as i32);
    launch(
        gpu,
        "qwen4_moe_sym_ref_swiglu",
        [n.div_ceil(256) as u32, 1, 1],
        256,
        &mut a,
    )?;
    gpu.hip.device_synchronize()?;
    let r = download(gpu, &ot.buf, n * 2)?;
    for tsr in [gt, ut, lt, ot] {
        gpu.free_tensor(tsr)?;
    }
    Ok(r.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect())
}

/// Exact fold check of both GEMMs on `tiles` (grouped tile indices) against
/// the sidecars the route produced. Returns (gate/up mismatches, down
/// mismatches, nonzero padding values).
fn fold_check(
    gpu: &mut Gpu,
    c: &Case,
    g: &Grouping,
    tiles: &[usize],
    xqg: &[u8],
    xqd: &[u8],
    ygu: &[u8],
    ydn: &[u8],
) -> Res<(usize, usize, usize)> {
    let (mut gu_mis, mut dn_mis, mut pad_bad) = (0usize, 0usize, 0usize);
    let (mut gs, mut us, mut live, mut cand) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for &tile in tiles {
        let e = g.tiles[tile] as usize;
        let slots: Vec<i32> = (0..16).map(|i| g.sorted[tile * 16 + i]).collect();
        let mut xg = vec![0u8; (geo().hid / 128) * 16 * BLK];
        let mut xd = vec![0u8; (geo().mi / 128) * 16 * BLK];
        for (j, &s) in slots.iter().enumerate() {
            if s < 0 {
                continue;
            }
            let tok = s as usize / geo().topk;
            for h in 0..geo().hid / 128 {
                xg[(h * 16 + j) * BLK..(h * 16 + j + 1) * BLK]
                    .copy_from_slice(&xqg[(h * c.t + tok) * BLK..(h * c.t + tok + 1) * BLK]);
            }
            for h in 0..geo().mi / 128 {
                let o = (h * c.l + s as usize) * BLK;
                xd[(h * 16 + j) * BLK..(h * 16 + j + 1) * BLK].copy_from_slice(&xqd[o..o + BLK]);
            }
        }
        let ge = e % c.gu_distinct;
        let f = fold_reference(
            &c.gu_w[ge * geo().gu_stride..(ge + 1) * geo().gu_stride],
            &xg,
            geo().gu_m,
            geo().hid,
            16,
            GU_GB,
        );
        for j in 0..16 {
            live.push((slots[j] >= 0) as i32);
            for col in 0..geo().mi {
                gs.push(f[j * geo().gu_m + col]);
                us.push(f[j * geo().gu_m + geo().mi + col]);
                let o = ((tile * 16 + j) * geo().mi + col) * 2;
                cand.push(u16::from_le_bytes([ygu[o], ygu[o + 1]]));
            }
        }
        let de = e % c.dn_distinct;
        let fd = fold_reference(
            &c.dn_w[de * geo().dn_stride..(de + 1) * geo().dn_stride],
            &xd,
            geo().dn_m,
            geo().mi,
            16,
            DN_GB,
        );
        for j in 0..16 {
            for col in 0..geo().dn_m {
                let o = ((tile * 16 + j) * geo().dn_m + col) * 2;
                let got = u16::from_le_bytes([ydn[o], ydn[o + 1]]);
                let want = if slots[j] >= 0 {
                    bf16_bits(fd[j * geo().dn_m + col])
                } else {
                    0
                };
                dn_mis += (got != want) as usize;
                pad_bad += (slots[j] < 0 && got != 0) as usize;
            }
        }
    }
    let r = ref_swiglu(gpu, &gs, &us, &live)?;
    for (i, got) in cand.iter().enumerate() {
        gu_mis += (*got != r[i]) as usize;
        pad_bad += (live[i / geo().mi] == 0 && *got != 0) as usize;
    }
    Ok((gu_mis, dn_mis, pad_bad))
}

/// The down sidecar the real `mq_rotate_x_128_v2` + shared quantizer make
/// from the route's own gate/up output, vs the route's producer.
fn down_sidecar_mismatch(gpu: &mut Gpu, c: &Case, xqd: &[u8]) -> Res<usize> {
    let n = c.pmax * geo().mi;
    let rot = alloc(gpu, n * 4)?;
    let refq = alloc(gpu, (geo().mi / 128) * c.l * BLK)?;
    gpu.hip
        .memset(&refq.buf, 0x7B, (geo().mi / 128) * c.l * BLK)?;
    let mut a = KernargBlob::new();
    a.push_ptr(p(&c.ygu));
    a.push_ptr(p(&rot));
    a.push_i32(n as i32);
    launch(
        gpu,
        "qwen4_moe_sym_ref_widen",
        [n.div_ceil(256) as u32, 1, 1],
        256,
        &mut a,
    )?;
    gpu.ensure_mq_signs_128()?;
    let s1 = gpu.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
    let s2 = gpu.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
    // Both reference kernels are row-local with grid.y = rows; chunk the rows under HIP's grid.y limit
    // (a3b at 8192 tokens has 69376 grouped rows). `qwen4_moe_sym_ref_quant` keeps the full
    // `l` slot extent, so a chunk only offsets the row base of `rot` and `sorted`.
    const ROWS_PER_LAUNCH: usize = 32768;
    for first in (0..c.pmax).step_by(ROWS_PER_LAUNCH) {
        let rows = (c.pmax - first).min(ROWS_PER_LAUNCH);
        let rot_at = (p(&rot) as *const u8).wrapping_add(first * geo().mi * 4) as *const _;
        let sorted_at = (p(&c.sorted) as *const u8).wrapping_add(first * 4) as *const _;
        let mut a = KernargBlob::new();
        a.push_ptr(rot_at);
        a.push_ptr(rot_at);
        a.push_ptr(s1);
        a.push_ptr(s2);
        a.push_i32(geo().mi as i32);
        launch(
            gpu,
            "mq_rotate_x_128_v2",
            [(geo().mi / 128) as u32, rows as u32, 1],
            32,
            &mut a,
        )?;
        let mut a = KernargBlob::new();
        a.push_ptr(rot_at);
        a.push_ptr(sorted_at);
        a.push_ptr(p(&refq));
        a.push_i32(geo().mi as i32);
        a.push_i32(rows as i32);
        a.push_i32(c.l as i32);
        launch(
            gpu,
            "qwen4_moe_sym_ref_quant",
            [(geo().mi / 128) as u32, rows as u32, 1],
            32,
            &mut a,
        )?;
    }
    gpu.hip.device_synchronize()?;
    let r = download(gpu, &refq.buf, (geo().mi / 128) * c.l * BLK)?;
    gpu.free_tensor(rot)?;
    gpu.free_tensor(refq)?;
    Ok(r.iter().zip(xqd).filter(|(a, b)| a != b).count())
}

/// Whether the route's down producer writes the grouped-position plane for this case.
fn gp_plane(gpu: &Gpu, c: &Case) -> bool {
    gpu.qwen4_moe_rotate128_i4_gp_selected(geo().mi, c.pmax, c.l)
}

/// The gate and down sidecars. The grouped-position down plane (record `g * pmax + p`) is returned
/// remapped to the flat-slot layout (record `g * l + sorted[p]`, every slot a valid row names exactly
/// once), so every flat-sidecar comparison below applies to either producer.
fn sidecars(gpu: &Gpu, c: &Case) -> Res<(Vec<u8>, Vec<u8>)> {
    let g = gpu
        .scratch
        .int4_mmq_x_scratch
        .as_ref()
        .ok_or("no gate sidecar")?;
    let d = gpu
        .scratch
        .qwen4_moe_down_i4_scratch
        .as_ref()
        .ok_or("no down sidecar")?;
    let groups = geo().mi / 128;
    let down = if gp_plane(gpu, c) {
        let plane = download(gpu, d, groups * c.pmax * BLK)?;
        let sorted = download(gpu, &c.sorted.buf, c.pmax * 4)?;
        let mut flat = vec![0x7Bu8; groups * c.l * BLK];
        for grp in 0..groups {
            for (pos, w) in sorted.chunks_exact(4).enumerate() {
                let slot = i32::from_le_bytes([w[0], w[1], w[2], w[3]]);
                if slot >= 0 {
                    let (src, dst) = (
                        (grp * c.pmax + pos) * BLK,
                        (grp * c.l + slot as usize) * BLK,
                    );
                    flat[dst..dst + BLK].copy_from_slice(&plane[src..src + BLK]);
                }
            }
        }
        flat
    } else {
        download(gpu, d, groups * c.l * BLK)?
    };
    Ok((download(gpu, g, (geo().hid / 128) * c.t * BLK)?, down))
}

fn sample_tiles(used: usize) -> Vec<usize> {
    let step = (used / 24).max(1);
    let mut s: Vec<usize> = (0..used).step_by(step).collect();
    if used > 0 && !s.contains(&(used - 1)) {
        s.push(used - 1);
    }
    s
}

/// Header checker on this case's tables: (gate/up ok, down ok).
fn header_checks(gpu: &mut Gpu, c: &Case) -> Res<(bool, bool)> {
    Ok((
        gpu.qwen4_moe_sym_check(&c.gu_ptrs, geo().gu_m, geo().hid, geo().e, GU_GB)?,
        gpu.qwen4_moe_sym_check(&c.dn_ptrs, geo().dn_m, geo().mi, geo().e, DN_GB)?,
    ))
}

/// BF16 rows widened to F32 (lossless), `n` values: the HIP combine's input.
fn widen_bf16(gpu: &mut Gpu, src: &GpuTensor, n: usize) -> Res<GpuTensor> {
    let out = alloc(gpu, n * 4)?;
    let mut a = KernargBlob::new();
    a.push_ptr(p(src));
    a.push_ptr(p(&out));
    a.push_i32(n as i32);
    launch(
        gpu,
        "qwen4_moe_sym_ref_widen",
        [n.div_ceil(256) as u32, 1, 1],
        256,
        &mut a,
    )?;
    Ok(out)
}

/// The PM BF16-row combine over the route's grouped down rows `c.ydn`
/// (`x_residual += sum_k w * row`, ascending slots, unrouted slots skipped).
/// The only call site of the launcher.
fn pm_combine(
    gpu: &mut Gpu,
    c: &Case,
    inverse: &GpuTensor,
    weights: &GpuTensor,
    residual: &GpuTensor,
) -> Res<()> {
    gpu.moe_down_combine_grouped_bf16in_pm(
        &c.ydn,
        inverse,
        weights,
        residual,
        geo().dn_m,
        geo().topk,
        c.t,
        c.pmax,
    )?;
    Ok(())
}

/// Byte mismatches of the PM combine vs the HIP `moe_down_combine_grouped_k8`
/// over `c.ydn` widened to F32, for the natural inverse permutation and one
/// with every 7th flat slot unrouted (-1). Call after the route ran; random
/// positive weights and residual.
fn combine_mismatch(gpu: &mut Gpu, c: &Case, seed: u64) -> Res<[usize; 2]> {
    let (dim, k_top) = (geo().dn_m, geo().topk);
    let mut r = Rng(seed ^ 0xc0b1);
    let wts: Vec<f32> = (0..c.l).map(|_| 0.02 + r.unit()).collect();
    let res0: Vec<f32> = (0..c.t * dim).map(|_| (r.unit() - 0.5) * 8.0).collect();
    let natural = to_i32(&download(gpu, &c.inverse.buf, c.l * 4)?);
    let unrouted: Vec<i32> = natural
        .iter()
        .enumerate()
        .map(|(i, &s)| if i % 7 == 3 { -1 } else { s })
        .collect();
    let wide = widen_bf16(gpu, &c.ydn, c.pmax * dim)?;
    let wt = gpu.upload_f32(&wts, &[c.l])?;
    let mut mis = [0usize; 2];
    for (k, inv) in [natural, unrouted].iter().enumerate() {
        let it = gpu.upload_raw(&i32_bytes(inv), &[c.l])?;
        let pm = gpu.upload_f32(&res0, &[c.t * dim])?;
        let hip = gpu.upload_f32(&res0, &[c.t * dim])?;
        pm_combine(gpu, c, &it, &wt, &pm)?;
        gpu.moe_down_combine_grouped_k8(&wide, &it, &wt, &hip, dim, k_top, c.t)?;
        gpu.hip.device_synchronize()?;
        mis[k] = diff(
            &download(gpu, &pm.buf, c.t * dim * 4)?,
            &download(gpu, &hip.buf, c.t * dim * 4)?,
        );
        for tsr in [it, pm, hip] {
            gpu.free_tensor(tsr)?;
        }
    }
    gpu.free_tensor(wide)?;
    gpu.free_tensor(wt)?;
    Ok(mis)
}

// ---------------------------------------------------------------- sweeps

/// One-hot weight sweep: every K position of both formats x all 16 nibble
/// codes, one probe per (expert, row); every other code 8 (zero weight),
/// both half scales of each QT44 group distinct, so a nibble, K-order, half
/// or epoch mix-up changes the exact fold. X comes from the real producers.
fn weight_sweep(seed: u64) -> (Vec<u8>, Vec<u8>, Vec<i32>) {
    let mut r = Rng(seed ^ 0x0e0e);
    let gu_experts = geo().hid * 16 / geo().gu_m; // 32
    let dn_experts = geo().mi * 16 / geo().dn_m; // 4
    let mut probe = |m: usize, k: usize, gb: usize, experts: usize| {
        let mut w = Vec::new();
        for e in 0..experts {
            let mut ew = symmetric_weights(m, k, gb, seed * 7919 + e as u64);
            for row in 0..m {
                // Zero every code, then set the probe.
                for kk in 0..k {
                    let (_, nb) = locate(row, kk, k, gb);
                    ew[nb] = 0x88;
                }
                let id = e * m + row; // 0 .. k*16
                let (kk, code) = (id % k, (id / k) as u8);
                let (hdr, nb) = locate(row, kk, k, gb);
                if kk % 2 == 0 {
                    ew[nb] = (ew[nb] & 0xF0) | code;
                } else {
                    ew[nb] = (ew[nb] & 0x0F) | (code << 4);
                }
                // A nonzero, non-degenerate scale on the probe's header.
                let (sc, zp) = loop {
                    let h = sym_header(&mut r);
                    if h.0 & 0x7fff != 0 {
                        break h;
                    }
                };
                ew[hdr..hdr + 2].copy_from_slice(&sc.to_le_bytes());
                ew[hdr + 2..hdr + 4].copy_from_slice(&zp.to_le_bytes());
            }
            w.extend(ew);
        }
        w
    };
    let gu = probe(geo().gu_m, geo().hid, GU_GB, gu_experts);
    let dn = probe(geo().dn_m, geo().mi, DN_GB, dn_experts);
    // 512 tokens, each to top-k distinct probe experts (3j mod 32 distinct).
    let topk = (0..512 * geo().topk)
        .map(|i| (((i / geo().topk) + 3 * (i % geo().topk)) % gu_experts) as i32)
        .collect();
    (gu, dn, topk)
}

/// One-hot activation sweep through the candidate GEMM entry with crafted
/// sidecars: slot/token j carries value v = (j / 128) % 16 - 8 at in-block
/// position j % 128 of every epoch (d varied per block), the rest zero; random
/// symmetric weights. Every anchor runs on the same operands and must agree
/// with the candidate byte for byte. Returns (gate/up mismatches, down
/// mismatches, probes, per-anchor byte mismatches).
fn x_onehot_sweep(
    gpu: &mut Gpu,
    seed: u64,
    cand: Entry,
    anchors: &[Entry],
) -> Res<(usize, usize, usize, Vec<(&'static str, [usize; 2])>)> {
    let n = 128 * 16; // every position x every value
    let mut r = Rng(seed ^ 0x1e1e);
    let craft = |epochs: usize, r: &mut Rng| {
        let mut xq = vec![0u8; epochs * n * BLK];
        for h in 0..epochs {
            for j in 0..n {
                let b = &mut xq[(h * n + j) * BLK..(h * n + j + 1) * BLK];
                let v = ((j / 128) % 16) as i32 - 8;
                let pos = j % 128;
                let d = f32::from_bits(0x3c00_0000 + (r.below(1 << 22) as u32)); // ~2^-7 .. 2^-6
                b[0..4].copy_from_slice(&d.to_le_bytes());
                b[4..8].copy_from_slice(&v.to_le_bytes());
                let nib = (v as u8) & 15;
                b[8 + pos / 2] = if pos % 2 == 0 { nib } else { nib << 4 };
            }
        }
        xq
    };
    let xg = craft(geo().hid / 128, &mut r);
    let xd = craft(geo().mi / 128, &mut r);
    let gu_w = symmetric_weights(geo().gu_m, geo().hid, GU_GB, seed * 31 + 1);
    let dn_w = symmetric_weights(geo().dn_m, geo().mi, DN_GB, seed * 31 + 2);
    let gu_t = gpu.upload_raw(&gu_w, &[gu_w.len()])?;
    let dn_t = gpu.upload_raw(&dn_w, &[dn_w.len()])?;
    let gu_ptrs = gpu.upload_raw(&ptr_table(&gu_t, geo().gu_stride, 1), &[geo().e * 8])?;
    let dn_ptrs = gpu.upload_raw(&ptr_table(&dn_t, geo().dn_stride, 1), &[geo().e * 8])?;
    // Grouped rows j -> expert 0, source row j (x_row_div 1 for both GEMMs).
    let tiles = gpu.upload_raw(&i32_bytes(&vec![0; n / 16]), &[n / 16])?;
    let sorted = gpu.upload_raw(&i32_bytes(&(0..n as i32).collect::<Vec<_>>()), &[n])?;
    let xgt = gpu.upload_raw(&xg, &[xg.len()])?;
    let xdt = gpu.upload_raw(&xd, &[xd.len()])?;
    let ygu = alloc(gpu, n * geo().mi * 2)?;
    let ydn = alloc(gpu, n * geo().dn_m * 2)?;
    let run = |gpu: &mut Gpu, e: Entry| -> Res<(Vec<u8>, Vec<u8>)> {
        gpu.hip.memset(&ygu.buf, 0x7B, n * geo().mi * 2)?;
        gpu.hip.memset(&ydn.buf, 0x7B, n * geo().dn_m * 2)?;
        for (func, ptrs, xq, y, m, k, block, rr) in [
            (
                e.gate_up,
                &gu_ptrs,
                &xgt,
                &ygu,
                geo().gu_m,
                geo().hid,
                e.gu_block,
                1,
            ),
            (
                e.down,
                &dn_ptrs,
                &xdt,
                &ydn,
                geo().dn_m,
                geo().mi,
                128u32,
                e.down_rr,
            ),
        ] {
            let mut a = KernargBlob::new();
            a.push_ptr(p(ptrs));
            a.push_ptr(p(&tiles));
            a.push_ptr(p(&sorted));
            a.push_ptr(p(xq));
            a.push_ptr(p(y));
            for v in [m, k, 1, n, n, ptrs.buf.size() / 8] {
                a.push_i32(v as i32);
            }
            if m % (64 * rr) != 0 {
                return Err(format!("{func}: m {m} is not a multiple of 64 * {rr}").into());
            }
            launch(
                gpu,
                func,
                [(m / (64 * rr)) as u32, (n / 16) as u32, 1],
                block,
                &mut a,
            )?;
        }
        gpu.hip.device_synchronize()?;
        Ok((
            download(gpu, &ygu.buf, n * geo().mi * 2)?,
            download(gpu, &ydn.buf, n * geo().dn_m * 2)?,
        ))
    };
    let (got_gu, got_dn) = run(gpu, cand)?;
    let mut anchor_mis = Vec::with_capacity(anchors.len());
    for &e in anchors {
        let (ag, ad) = run(gpu, e)?;
        anchor_mis.push((e.name, [diff(&ag, &got_gu), diff(&ad, &got_dn)]));
    }
    let fg = fold_reference(&gu_w, &xg, geo().gu_m, geo().hid, n, GU_GB);
    let fd = fold_reference(&dn_w, &xd, geo().dn_m, geo().mi, n, DN_GB);
    let (mut gs, mut us) = (
        Vec::with_capacity(n * geo().mi),
        Vec::with_capacity(n * geo().mi),
    );
    for j in 0..n {
        gs.extend_from_slice(&fg[j * geo().gu_m..j * geo().gu_m + geo().mi]);
        us.extend_from_slice(&fg[j * geo().gu_m + geo().mi..(j + 1) * geo().gu_m]);
    }
    let r_gu = ref_swiglu(gpu, &gs, &us, &vec![1; n])?;
    let gu_mis = r_gu
        .iter()
        .enumerate()
        .filter(|(i, w)| u16::from_le_bytes([got_gu[2 * i], got_gu[2 * i + 1]]) != **w)
        .count();
    let dn_mis = fd
        .iter()
        .enumerate()
        .filter(|(i, v)| u16::from_le_bytes([got_dn[2 * i], got_dn[2 * i + 1]]) != bf16_bits(**v))
        .count();
    for tsr in [
        gu_t, dn_t, gu_ptrs, dn_ptrs, tiles, sorted, xgt, xdt, ygu, ydn,
    ] {
        gpu.free_tensor(tsr)?;
    }
    Ok((gu_mis, dn_mis, n, anchor_mis))
}

// ---------------------------------------------------------------- rows mode

const ROWS_FILLER_BASE: u64 = 1 << 40;

/// Default `--rows`: 1..=64, every multiple of 16 up to 1024, and the tile /
/// threshold neighbours up to 8191.
fn rows_default() -> Vec<usize> {
    let mut v: Vec<usize> = (1..=64).collect();
    v.extend((16..=1024).step_by(16));
    v.extend([
        255, 256, 257, 511, 512, 513, 1023, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 8191,
    ]);
    v.sort_unstable();
    v.dedup();
    v
}

/// `--rows` list: comma separated `N`, `A-B` or `A-B:STEP` (empty = default).
fn rows_list(spec: &str) -> Res<Vec<usize>> {
    if spec.is_empty() {
        return Ok(rows_default());
    }
    let mut v = Vec::new();
    for tok in spec.split(',').filter(|t| !t.is_empty()) {
        let (range, step) = match tok.split_once(':') {
            Some((r, s)) => (r, s.parse::<usize>()?),
            None => (tok, 1),
        };
        if step == 0 {
            return Err(format!("--rows {tok}: step 0").into());
        }
        match range.split_once('-') {
            Some((a, b)) => v.extend((a.parse::<usize>()?..=b.parse::<usize>()?).step_by(step)),
            None => v.push(range.parse()?),
        }
    }
    v.sort_unstable();
    v.dedup();
    if v.first() == Some(&0) {
        return Err("--rows: 0 tokens".into());
    }
    Ok(v)
}

/// Offsets (> 0) tested for `t` tokens; `t` = place the end of the rows at
/// row 4096 (only when t < 4096).
fn rows_offsets(spec: &str, t: usize) -> Res<Vec<usize>> {
    let mut v = Vec::new();
    for tok in spec.split(',').filter(|s| !s.is_empty()) {
        if tok == "t" {
            if t < 4096 {
                v.push(4096 - t);
            }
        } else {
            v.push(tok.parse()?);
        }
    }
    v.retain(|&o| o > 0);
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

fn rows_bytes<T>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slices (i32/f32), read as bytes of the same extent.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn rows_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Token `idx`'s top-10 experts and weights, a function of `(seed, idx)` only:
/// 0..5 picks from 8 hot experts (hit by a large share of all tokens, so one
/// expert's run spans many tokens and tiles), then warm (8..64) and cold
/// (64..512, often empty at small T) experts; all ten distinct, in a random
/// (not sorted) rank order.
fn row_route(seed: u64, idx: u64) -> (Vec<i32>, Vec<f32>) {
    let mut r = Rng(seed ^ idx.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ 0x0007_0077);
    let hot = [0usize, 1, 1, 2, 2, 3, 3, 5][r.below(8) as usize];
    let mut ids = vec![0i32; geo().topk];
    let mut n = 0;
    while n < geo().topk {
        let e = if n < hot {
            r.below(8) as i32
        } else if r.below(3) == 0 {
            8 + r.below(56) as i32
        } else {
            64 + r.below((geo().e - 64) as u64) as i32
        };
        if !ids[..n].contains(&e) {
            ids[n] = e;
            n += 1;
        }
    }
    for i in (1..geo().topk).rev() {
        let j = r.below(i as u64 + 1) as usize;
        ids.swap(i, j);
    }
    let mut w = vec![0f32; geo().topk];
    let mut s = 0f32;
    for x in w.iter_mut() {
        *x = 0.02 + r.unit();
        s += *x;
    }
    for x in w.iter_mut() {
        *x /= s;
    }
    (ids, w)
}

/// Token `idx`'s activation row (F32 [2560]), a function of `(seed, idx)`
/// only: per-token magnitude (0.25..2) and a sparse channel outlier.
fn row_x(seed: u64, idx: u64) -> Vec<f32> {
    let mut r = Rng(seed ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5858);
    let scale = [0.25f32, 0.5, 1.0, 2.0][(idx % 4) as usize];
    (0..geo().hid)
        .map(|i| {
            let v = (r.unit() - 0.5) * 4.0 * scale;
            if i % 97 == 13 {
                v * 8.0
            } else {
                v
            }
        })
        .collect()
}

/// `n` consecutive tokens' routing and activations from index `first`.
struct RowPool {
    topk: Vec<i32>,
    wts: Vec<f32>,
    x: Vec<f32>,
}

fn row_pool(seed: u64, first: u64, n: usize) -> RowPool {
    let mut p = RowPool {
        topk: Vec::with_capacity(n * geo().topk),
        wts: Vec::with_capacity(n * geo().topk),
        x: Vec::with_capacity(n * geo().hid),
    };
    for j in 0..n as u64 {
        let (ids, w) = row_route(seed, first + j);
        p.topk.extend(ids);
        p.wts.extend(w);
        p.x.extend(row_x(seed, first + j));
    }
    p
}

/// Device state of the rows-mode chain, sized for the largest batch.
struct RowBufs {
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
    topk: GpuTensor,
    wts: GpuTensor,
    x: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    ygu: GpuTensor,
    ydn: GpuTensor,
    order: GpuTensor,
    out: GpuTensor,
}

struct RowOut {
    /// Combined MoE output rows of the `t0` tested tokens (F32 [t0][2560]).
    out: Vec<u8>,
    /// Kernel of the last launch of each stage (scatter, rotate256, gate/up,
    /// rotate128, down, combine).
    kernels: Vec<String>,
    /// `--diag`: de-permuted per-slot SwiGLU (BF16 [t0*10][640]) and down
    /// (BF16 [t0*10][2560]) rows of the tested tokens.
    slots: Option<(Vec<u8>, Vec<u8>)>,
}

fn rows_slots(buf: &[u8], inv: &[i32], first: usize, n: usize, rb: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n * rb);
    for f in first..first + n {
        let row = inv[f] as usize;
        v.extend_from_slice(&buf[row * rb..(row + 1) * rb]);
    }
    v
}

/// Byte comparison by rows: `None` when identical, else the first differing
/// row and a JSON description.
fn rows_diff(
    got: &[u8],
    want: &[u8],
    rb: usize,
    elem: usize,
) -> Option<(usize, serde_json::Value)> {
    if got == want {
        return None;
    }
    let n = got.len().min(want.len()) / rb;
    let mut count = 0usize;
    let mut first: Option<(usize, usize)> = None;
    for r in 0..n {
        let (a, b) = (&got[r * rb..(r + 1) * rb], &want[r * rb..(r + 1) * rb]);
        if a != b {
            count += 1;
            if first.is_none() {
                first = Some((r, a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0)));
            }
        }
    }
    let (row, byte) = first.unwrap_or((0, 0));
    let col = byte / elem;
    let at = |v: &[u8]| rows_hex(&v[row * rb + col * elem..row * rb + (col + 1) * elem]);
    Some((
        row,
        serde_json::json!({
            "row": row, "col": col, "byte": byte,
            "got": at(got), "want": at(want), "rows_differing": count,
        }),
    ))
}

/// One chain run over a batch of `o + t0` tokens: `o` filler tokens (other
/// data) then the first `t0` pool tokens at row offset `o`.
#[allow(clippy::too_many_arguments)]
fn rows_run(
    gpu: &mut Gpu,
    b: &RowBufs,
    base: &RowPool,
    fill: &RowPool,
    o: usize,
    t0: usize,
    host: bool,
    diag: bool,
) -> Res<RowOut> {
    let t = o + t0;
    let l = t * geo().topk;
    let pmax = (l + geo().e * 15).div_ceil(16) * 16;
    if o > 0 {
        gpu.hip
            .memcpy_htod_offset(&b.topk.buf, 0, rows_bytes(&fill.topk[..o * geo().topk]))?;
        gpu.hip
            .memcpy_htod_offset(&b.wts.buf, 0, rows_bytes(&fill.wts[..o * geo().topk]))?;
        gpu.hip
            .memcpy_htod_offset(&b.x.buf, 0, rows_bytes(&fill.x[..o * geo().hid]))?;
    }
    gpu.hip.memcpy_htod_offset(
        &b.topk.buf,
        o * geo().topk * 4,
        rows_bytes(&base.topk[..t0 * geo().topk]),
    )?;
    gpu.hip.memcpy_htod_offset(
        &b.wts.buf,
        o * geo().topk * 4,
        rows_bytes(&base.wts[..t0 * geo().topk]),
    )?;
    gpu.hip.memcpy_htod_offset(
        &b.x.buf,
        o * geo().hid * 4,
        rows_bytes(&base.x[..t0 * geo().hid]),
    )?;
    gpu.hip.memset(&b.ygu.buf, 0x7B, pmax * geo().mi * 2)?;
    gpu.hip.memset(&b.ydn.buf, 0x7B, pmax * geo().dn_m * 2)?;
    gpu.hip.memset(&b.out.buf, 0xA5, t * geo().hid * 4)?;
    let mut kernels: Vec<String> = Vec::new();
    gpu.moe_scatter_stable_top10(
        &b.topk,
        &b.counts,
        &b.offsets,
        &b.sorted,
        &b.tiles,
        &b.inverse,
        l,
        geo().e,
        pmax,
    )?;
    kernels.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    let x_view = b.x.sub_offset(0, t * geo().hid);
    let gate = gpu.qwen4_moe_rotate256_i4(&x_view, geo().hid, t)?;
    kernels.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
        &b.gu_ptrs,
        &b.tiles,
        &b.sorted,
        &gate,
        &b.ygu,
        geo().gu_m,
        geo().hid,
        geo().topk,
        pmax,
        t,
        host,
    )?;
    kernels.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    let down = gpu.qwen4_moe_rotate128_i4(&b.ygu, &b.sorted, geo().mi, pmax, l)?;
    kernels.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    gpu.gemm_qwen4_moe_down_iu4_sym(
        &b.dn_ptrs,
        &b.tiles,
        &b.sorted,
        &down,
        &b.ydn,
        geo().dn_m,
        geo().mi,
        1,
        pmax,
        l,
        host,
    )?;
    kernels.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    // The prefill's combine: BF16 grouped rows, rank order by expert id,
    // written from +0.0 (`combine_initial_zero_applies` on this route).
    gpu.moe_down_combine_grouped_top10_bf16in(
        &b.ydn,
        &b.inverse,
        &b.topk,
        &b.wts,
        &b.out,
        &b.order,
        geo().hid,
        pmax,
        t,
        true,
        geo().e,
    )?;
    kernels.push(gpu.last_launched_kernel().unwrap_or("?").to_string());
    gpu.hip.device_synchronize()?;
    let mut out = vec![0u8; t0 * geo().hid * 4];
    gpu.hip
        .memcpy_dtoh_at(&mut out, &b.out.buf, o * geo().hid * 4)?;
    let slots = if diag {
        let mut inv = vec![0u8; l * 4];
        gpu.hip.memcpy_dtoh(&mut inv, &b.inverse.buf)?;
        let inv = to_i32(&inv);
        let mut ygu = vec![0u8; pmax * geo().mi * 2];
        gpu.hip.memcpy_dtoh(&mut ygu, &b.ygu.buf)?;
        let mut ydn = vec![0u8; pmax * geo().dn_m * 2];
        gpu.hip.memcpy_dtoh(&mut ydn, &b.ydn.buf)?;
        Some((
            rows_slots(&ygu, &inv, o * geo().topk, t0 * geo().topk, geo().mi * 2),
            rows_slots(&ydn, &inv, o * geo().topk, t0 * geo().topk, geo().dn_m * 2),
        ))
    } else {
        None
    };
    Ok(RowOut {
        out,
        kernels,
        slots,
    })
}

/// `--mode rows`: per-token batch invariance of the production grouped
/// symmetric IU4 MoE chain (stable top-10 scatter, rotate256 A4, gate/up +
/// SiLU, rotate128 A4, down, and the prefill's BF16-row combine).  Routing is
/// per token (`row_route`: hot, warm, cold and colliding experts across
/// tokens), the reference is the chain over T = `--ref-tokens` (8192) tokens,
/// and for every T of `--rows` the first T tokens, alone (offset 0) and at
/// each `--offsets` row offset inside a batch of T + o tokens whose first o
/// are other data, must reproduce the reference's per-token combined output
/// rows (`[token][2560]` F32, de-permuted by the combine; packed slots are
/// never compared except under `--diag`, which localizes a difference to the
/// gate/up, down or combine stage).
///
/// The example calls the sym launchers directly, so `--scope` (it sets
/// `Gpu::prefill_invariant_scope`) cannot change what runs: the launchers
/// never consult a row count to choose an entry (the `>= 512` rows predicate
/// `Gpu::qwen4_moe_sym_iu4_applies` is a dispatch-side gate, reported per run
/// as `sym_applies` for the row count the scope's routing sees).
fn rows_mode(mut gpu: Gpu, args: &[String], host: bool, seed: u64) -> Res<()> {
    let scopes: Vec<bool> = match arg(args, "--scope", "on").as_str() {
        "on" => vec![true],
        "off" => vec![false],
        "both" => vec![false, true],
        other => return Err(format!("--scope {other}: expected on, off or both").into()),
    };
    let ref_t: usize = arg(args, "--ref-tokens", "8192").parse()?;
    let rows = rows_list(&arg(args, "--rows", ""))?;
    let offset_spec = arg(args, "--offsets", "1,17,256,t");
    let diag = args.iter().any(|a| a == "--diag");
    let print_all = arg(args, "--print", "all") != "fail";
    let distinct: usize = arg(args, "--distinct", &geo().e.to_string()).parse()?;
    if distinct == 0 || distinct > geo().e || ref_t == 0 {
        return Err(format!("--distinct must be 1..={} and --ref-tokens > 0", geo().e).into());
    }
    let rmax = *rows.last().ok_or("--rows is empty")?;
    if rmax > ref_t {
        return Err(format!("--rows max {rmax} exceeds --ref-tokens {ref_t}").into());
    }
    let (mut tmax, mut omax) = (ref_t, 1usize);
    for &t in &rows {
        for o in rows_offsets(&offset_spec, t)? {
            tmax = tmax.max(t + o);
            omax = omax.max(o);
        }
    }

    let mut gu_w = Vec::with_capacity(distinct * geo().gu_stride);
    let mut dn_w = Vec::with_capacity(distinct * geo().dn_stride);
    for e in 0..distinct as u64 {
        gu_w.extend(symmetric_weights(
            geo().gu_m,
            geo().hid,
            GU_GB,
            seed * 1000 + e,
        ));
        dn_w.extend(symmetric_weights(
            geo().dn_m,
            geo().mi,
            DN_GB,
            seed * 1000 + 600 + e,
        ));
    }
    let (gu_t, dn_t) = if host {
        (
            gpu.upload_raw_host_mapped(&gu_w, &[gu_w.len()])?,
            gpu.upload_raw_host_mapped(&dn_w, &[dn_w.len()])?,
        )
    } else {
        (
            gpu.upload_raw(&gu_w, &[gu_w.len()])?,
            gpu.upload_raw(&dn_w, &[dn_w.len()])?,
        )
    };
    drop((gu_w, dn_w));
    let l_max = tmax * geo().topk;
    let pmax_max = (l_max + geo().e * 15).div_ceil(16) * 16;
    let bufs = RowBufs {
        gu_ptrs: gpu.upload_raw(&ptr_table(&gu_t, geo().gu_stride, distinct), &[geo().e * 8])?,
        dn_ptrs: gpu.upload_raw(&ptr_table(&dn_t, geo().dn_stride, distinct), &[geo().e * 8])?,
        topk: alloc(&mut gpu, l_max * 4)?,
        wts: alloc(&mut gpu, l_max * 4)?,
        x: alloc(&mut gpu, tmax * geo().hid * 4)?,
        counts: alloc(&mut gpu, geo().e * 4)?,
        offsets: alloc(&mut gpu, (geo().e + 1) * 4)?,
        sorted: alloc(&mut gpu, pmax_max * 4)?,
        tiles: alloc(&mut gpu, pmax_max / 16 * 4)?,
        inverse: alloc(&mut gpu, l_max * 4)?,
        ygu: alloc(&mut gpu, pmax_max * geo().mi * 2 + GUARD)?,
        ydn: alloc(&mut gpu, pmax_max * geo().dn_m * 2 + GUARD)?,
        order: alloc(&mut gpu, tmax * geo().topk * 8)?,
        out: alloc(&mut gpu, tmax * geo().hid * 4)?,
    };
    let base = row_pool(seed, 0, ref_t);
    let fill = row_pool(seed, ROWS_FILLER_BASE, omax);

    let slots = ref_t
        .checked_mul(geo().topk)
        .ok_or("--ref-tokens * top-k overflows")?;
    let (cand, _) = cand_and_anchors(&gpu, host, slots)?;
    println!(
        "{}",
        serde_json::json!({
            "event": "start", "mode": "rows", "arch": gpu.arch, "seed": seed,
            "residency": if host { "host" } else { "vram" },
            "gemm_symbols": [cand.gate_up, cand.down], "candidate_entry": cand.name,
            "down_rr": cand.down_rr, "distinct_experts": distinct,
            "ref_tokens": ref_t, "tokens_list": rows.len(), "offsets": offset_spec,
            "scopes": scopes.iter().map(|s| if *s { "on" } else { "off" }).collect::<Vec<_>>(),
            "route_requested": gpu.qwen4_moe_sym_iu4_requested(),
            "note": "the sym launchers are called directly; scope changes only the reported sym_applies",
        })
    );

    let (mut runs, mut fails) = (0usize, 0usize);
    let mut prev_ref: Option<Vec<u8>> = None;
    for &scope in &scopes {
        gpu.prefill_invariant_scope = scope;
        let sname = if scope { "on" } else { "off" };
        let rf = rows_run(&mut gpu, &bufs, &base, &fill, 0, ref_t, host, diag)?;
        println!(
            "{}",
            serde_json::json!({
                "event": "ref", "scope": sname, "tokens": ref_t, "kernels": rf.kernels,
                "sym_applies": gpu.qwen4_moe_sym_iu4_applies(gpu.route_rows(ref_t)),
            })
        );
        if let Some(p) = &prev_ref {
            let d = rows_diff(&rf.out, p, geo().hid * 4, 4);
            runs += 1;
            fails += usize::from(d.is_some());
            println!(
                "{}",
                serde_json::json!({
                    "event": "ref_cross_scope", "tokens": ref_t,
                    "verdict": if d.is_some() { "differs" } else { "identical" },
                    "first_diff": d.map(|x| x.1),
                })
            );
        }
        for &t0 in &rows {
            let mut offs = vec![0usize];
            offs.extend(rows_offsets(&offset_spec, t0)?);
            for o in offs {
                if o == 0 && t0 == ref_t {
                    continue;
                }
                let run = rows_run(&mut gpu, &bufs, &base, &fill, o, t0, host, diag)?;
                runs += 1;
                let d = rows_diff(&run.out, &rf.out[..t0 * geo().hid * 4], geo().hid * 4, 4);
                let mut stage = serde_json::Value::Null;
                let mut stage_diff = serde_json::Value::Null;
                if d.is_some() {
                    if let (Some((gy, dy)), Some((rg, rd))) = (&run.slots, &rf.slots) {
                        let n = t0 * geo().topk;
                        let hit = rows_diff(gy, &rg[..n * geo().mi * 2], geo().mi * 2, 2)
                            .map(|x| ("gate_up_silu", x))
                            .or_else(|| {
                                rows_diff(dy, &rd[..n * geo().dn_m * 2], geo().dn_m * 2, 2)
                                    .map(|x| ("down", x))
                            })
                            .or(Some(("combine", (0, serde_json::Value::Null))));
                        if let Some((name, (row, mut v))) = hit {
                            stage = serde_json::json!(name);
                            if name != "combine" {
                                v["token"] = serde_json::json!(row / geo().topk);
                                v["rank"] = serde_json::json!(row % geo().topk);
                                stage_diff = v;
                            }
                        }
                    }
                    fails += 1;
                }
                if d.is_some() || print_all {
                    let first = d.as_ref().map(|(tok, v)| {
                        let mut v = v.clone();
                        v["token"] = serde_json::json!(tok);
                        v["experts"] = serde_json::json!(base.topk
                            [tok * geo().topk..(tok + 1) * geo().topk]
                            .to_vec());
                        v
                    });
                    println!(
                        "{}",
                        serde_json::json!({
                            "event": "run", "scope": sname, "tokens": t0, "offset": o, "batch": t0 + o,
                            "route_rows": gpu.route_rows(t0 + o),
                            "sym_applies": gpu.qwen4_moe_sym_iu4_applies(gpu.route_rows(t0 + o)),
                            "verdict": if d.is_some() { "differs" } else { "identical" },
                            "first_diff": first, "stage": stage, "stage_first_diff": stage_diff,
                            "kernels": run.kernels,
                        })
                    );
                }
            }
        }
        prev_ref = Some(rf.out);
    }
    println!(
        "{}",
        serde_json::json!({ "event": "summary", "runs": runs, "differing": fails })
    );
    if fails > 0 {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------- main

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = arg(&args, "--mode", "check");
    let rkind = arg(&args, "--routing", "all");
    let seed: u64 = arg(&args, "--seed", "1").parse()?;
    let iters: usize = arg(&args, "--iters", "10").parse()?;
    let tokens: usize = arg(&args, "--tokens", "1536").parse()?;
    let xfile = arg(&args, "--x", "");
    let residency = arg(&args, "--residency", "vram");
    let host = match residency.as_str() {
        "vram" => false,
        "host" => true,
        other => return Err(format!("--residency {other}: expected vram or host").into()),
    };
    let geometry = arg(&args, "--geometry", "fn");
    let selected = match geometry.as_str() {
        "fn" => GEO_FN,
        "a3b" => GEO_A3B,
        other => return Err(format!("--geometry {other}: expected fn or a3b").into()),
    };
    GEO.set(selected).map_err(|_| "geometry selected twice")?;

    let mut gpu = Gpu::init()?;
    if !matches!(gpu.arch.as_str(), "gfx1151" | "gfx1201") {
        return Err(format!(
            "the symmetric IU4 MoE route is gfx1151/gfx1201-only, device is {}",
            gpu.arch
        )
        .into());
    }
    if mode == "rows" {
        if geo().name != "fn" {
            return Err("--mode rows is --geometry fn only (top-10 BF16 combine)".into());
        }
        return rows_mode(gpu, &args, host, seed);
    }
    load_oracle(&mut gpu)?;
    let mut out = BTreeMap::<String, String>::new();
    let q = |s: &str| format!("\"{s}\"");
    out.insert("arch".into(), q(&gpu.arch));
    out.insert("mode".into(), q(&mode));
    if geo().name != "fn" {
        out.insert("geometry".into(), q(geo().name));
        out.insert(
            "geometry_experts_topk_hidden_inter".into(),
            format!("[{},{},{},{}]", geo().e, geo().topk, geo().hid, geo().mi),
        );
    }
    out.insert("seed".into(), seed.to_string());
    out.insert("hipcc_extra_flags".into(), q(&gpu.flags.hipcc_extra_flags));
    out.insert(
        "route_requested".into(),
        gpu.qwen4_moe_sym_iu4_requested().to_string(),
    );
    let mut fails: Vec<String> = Vec::new();

    if mode == "sweep" {
        // (a) weight one-hot through the whole production envelope.
        let (gu_w, dn_w, topk) = weight_sweep(seed);
        let (cand, anchors) = report_candidate(&gpu, host, topk.len(), &residency, &mut out)?;
        let t = topk.len() / geo().topk;
        let mut r = Rng(seed ^ 0x77);
        let x: Vec<f32> = (0..t * geo().hid).map(|_| (r.unit() - 0.5) * 4.0).collect();
        let c = make_case(&mut gpu, topk, &x, gu_w, dn_w, host)?;
        let (gok, dok) = header_checks(&mut gpu, &c)?;
        poison(&gpu, &c, 0x7B)?;
        cand_all(&mut gpu, &c)?;
        gpu.hip.device_synchronize()?;
        let g = cpu_group(&c.topk_host, c.pmax);
        let (ygu, ydn) = outputs(&gpu, &c)?;
        let (xqg, xqd) = sidecars(&gpu, &c)?;
        let used = g.offsets[geo().e] as usize / 16;
        let all: Vec<usize> = (0..used).collect();
        let (gm, dm, pb) = fold_check(&mut gpu, &c, &g, &all, &xqg, &xqd, &ygu, &ydn)?;
        out.insert(
            "weight_onehot_probes_gate_up".into(),
            (geo().hid * 16).to_string(),
        );
        out.insert(
            "weight_onehot_probes_down".into(),
            (geo().mi * 16).to_string(),
        );
        out.insert("weight_onehot_tiles_checked".into(), used.to_string());
        out.insert(
            "weight_onehot_headers_symmetric".into(),
            format!("[{gok},{dok}]"),
        );
        out.insert("weight_onehot_gate_up_mismatch".into(), gm.to_string());
        out.insert("weight_onehot_down_mismatch".into(), dm.to_string());
        out.insert("weight_onehot_padding_nonzero".into(), pb.to_string());
        if !(gok && dok) || gm + dm + pb != 0 {
            fails.push("weight one-hot sweep".into());
        }
        let am = anchor_mismatch(&mut gpu, &c, &anchors, 0x7B, &ygu, &ydn)?;
        out.insert(
            "weight_onehot_anchor_byte_mismatch".into(),
            anchor_json(&am),
        );
        if anchor_total(&am) != 0 {
            fails.push("weight one-hot anchor mismatch".into());
        }
        // (b) activation one-hot, every position x value, every epoch.
        let (xg, xd, n, xa) = x_onehot_sweep(&mut gpu, seed, cand, &anchors)?;
        out.insert("x_onehot_slots".into(), n.to_string());
        out.insert("x_onehot_gate_up_mismatch".into(), xg.to_string());
        out.insert("x_onehot_down_mismatch".into(), xd.to_string());
        out.insert("x_onehot_anchor_byte_mismatch".into(), anchor_json(&xa));
        if xg + xd + anchor_total(&xa) != 0 {
            fails.push("activation one-hot sweep".into());
        }
    } else {
        let topk = routing(&rkind, tokens, seed)?;
        let (_, anchors) = report_candidate(&gpu, host, topk.len(), &residency, &mut out)?;
        let x: Vec<f32> = if xfile.is_empty() {
            let mut r = Rng(seed ^ 0x77);
            (0..tokens * geo().hid)
                .map(|_| (r.unit() - 0.5) * 4.0)
                .collect()
        } else {
            let b = std::fs::read(&xfile)?;
            b.chunks_exact(2)
                .take(tokens * geo().hid)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        let mut gu_w = Vec::with_capacity(geo().e * geo().gu_stride);
        let mut dn_w = Vec::with_capacity(geo().e * geo().dn_stride);
        for e in 0..geo().e as u64 {
            gu_w.extend(symmetric_weights(
                geo().gu_m,
                geo().hid,
                GU_GB,
                seed * 1000 + e,
            ));
            dn_w.extend(symmetric_weights(
                geo().dn_m,
                geo().mi,
                DN_GB,
                seed * 1000 + 600 + e,
            ));
        }
        out.insert("routing".into(), q(&rkind));
        out.insert("md5_routing".into(), q(&md5_hex(&i32_bytes(&topk))));
        out.insert(
            "md5_x".into(),
            q(&md5_hex(
                &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            )),
        );
        out.insert("md5_gu_w".into(), q(&md5_hex(&gu_w)));
        out.insert("md5_dn_w".into(), q(&md5_hex(&dn_w)));
        let c = make_case(&mut gpu, topk, &x, gu_w, dn_w, host)?;
        if geo().name == "fn" {
            inc_setup(&mut gpu, &c)?;
        }
        let g = cpu_group(&c.topk_host, c.pmax);
        let used = g.offsets[geo().e] as usize / 16;
        out.insert("tokens".into(), c.t.to_string());
        out.insert("slots_L".into(), c.l.to_string());
        out.insert("padded_P".into(), g.offsets[geo().e].to_string());
        out.insert(
            "active_experts".into(),
            g.counts.iter().filter(|&&n| n > 0).count().to_string(),
        );

        if mode == "check" || mode == "graph" {
            let (gok, dok) = header_checks(&mut gpu, &c)?;
            out.insert("sym_check_clean".into(), format!("[{gok},{dok}]"));
            if !(gok && dok) {
                fails.push("symmetric weights refused".into());
            }
            // Negative controls on one header each (offsets of the (sc, zp)
            // dword): a one-ulp zp flip in QT44 (second half) and QT53, and a
            // QT53 zero-scale group with a nonzero zero point.
            for (name, tsr, off) in [
                ("qt44_zp_ulp", &c.gu_t, 7 * geo().gu_stride + 3 * GU_GB + 4),
                (
                    "qt53_zp_ulp",
                    &c.dn_t,
                    (300 % geo().e) * geo().dn_stride + 11 * DN_GB,
                ),
                (
                    "qt53_zero_scale_nonzero_zp",
                    &c.dn_t,
                    (geo().e - 1) * geo().dn_stride + (geo().dn_m - 1) * (geo().mi / 128) * DN_GB,
                ),
            ] {
                let orig = download(&gpu, &tsr.buf, off + 4)?;
                let mut patch = [orig[off], orig[off + 1], orig[off + 2] ^ 1, orig[off + 3]];
                if name == "qt53_zero_scale_nonzero_zp" {
                    patch = [0x00, 0x00, 0x00, 0x3c];
                }
                gpu.hip.memcpy_htod_offset(&tsr.buf, off, &patch)?;
                let (g2, d2) = header_checks(&mut gpu, &c)?;
                gpu.hip
                    .memcpy_htod_offset(&tsr.buf, off, &orig[off..off + 4])?;
                let refused = !(g2 && d2);
                out.insert(
                    format!("sym_check_negative_{name}_refused"),
                    refused.to_string(),
                );
                if !refused {
                    fails.push(format!("negative control {name} accepted"));
                }
            }
            let (g3, d3) = header_checks(&mut gpu, &c)?;
            if !(g3 && d3) {
                fails.push("header restore".into());
            }

            poison(&gpu, &c, 0x7B)?;
            cand_all(&mut gpu, &c)?;
            gpu.hip.device_synchronize()?;
            let grp_ok = to_i32(&download(&gpu, &c.sorted.buf, c.pmax * 4)?) == g.sorted
                && to_i32(&download(&gpu, &c.tiles.buf, c.pmax / 16 * 4)?) == g.tiles
                && to_i32(&download(&gpu, &c.inverse.buf, c.l * 4)?) == g.inverse
                && to_i32(&download(&gpu, &c.counts.buf, geo().e * 4)?) == g.counts
                && to_i32(&download(&gpu, &c.offsets.buf, (geo().e + 1) * 4)?) == g.offsets;
            out.insert("group_matches_cpu_stable".into(), grp_ok.to_string());
            if !grp_ok {
                fails.push("grouping mismatch".into());
            }
            let (ygu, ydn) = outputs(&gpu, &c)?;
            let (xqg, xqd) = sidecars(&gpu, &c)?;
            let mut poison_bad = 0usize;
            for tile in 0..c.pmax / 16 {
                if g.tiles[tile] < 0 {
                    poison_bad += ygu[tile * 16 * geo().mi * 2..(tile + 1) * 16 * geo().mi * 2]
                        .iter()
                        .filter(|&&x| x != 0x7B)
                        .count();
                    poison_bad += ydn[tile * 16 * geo().dn_m * 2..(tile + 1) * 16 * geo().dn_m * 2]
                        .iter()
                        .filter(|&&x| x != 0x7B)
                        .count();
                }
            }
            poison_bad += ygu[c.pmax * geo().mi * 2..]
                .iter()
                .filter(|&&x| x != 0x7B)
                .count();
            poison_bad += ydn[c.pmax * geo().dn_m * 2..]
                .iter()
                .filter(|&&x| x != 0x7B)
                .count();
            out.insert("poison_violations".into(), poison_bad.to_string());
            if poison_bad != 0 {
                fails.push("poison violated".into());
            }
            let a4 = down_sidecar_mismatch(&mut gpu, &c, &xqd)?;
            out.insert("down_a4_sidecar_byte_mismatch".into(), a4.to_string());
            if a4 != 0 {
                fails.push("down A4 producer mismatch".into());
            }
            let sample = sample_tiles(used);
            let (gm, dm, pb) = fold_check(&mut gpu, &c, &g, &sample, &xqg, &xqd, &ygu, &ydn)?;
            out.insert("fold_checked_tiles".into(), sample.len().to_string());
            out.insert("gate_up_fold_mismatch_values".into(), gm.to_string());
            out.insert("down_fold_mismatch_values".into(), dm.to_string());
            out.insert("padding_nonzero_values".into(), pb.to_string());
            if gm + dm + pb != 0 {
                fails.push("GEMM fold mismatch".into());
            }
            let am = anchor_mismatch(&mut gpu, &c, &anchors, 0x7B, &ygu, &ydn)?;
            out.insert("anchor_byte_mismatch_gate_up_down".into(), anchor_json(&am));
            if anchor_total(&am) != 0 {
                fails.push("candidate GEMMs differ from an anchor".into());
            }
            if geo().name == "a3b" {
                let key = "pm_combine_vs_hip_byte_mismatch_natural_unrouted";
                if gpu.arch == "gfx1201" {
                    let mis = combine_mismatch(&mut gpu, &c, seed)?;
                    out.insert(key.into(), format!("[{},{}]", mis[0], mis[1]));
                    if mis[0] + mis[1] != 0 {
                        fails.push("PM combine differs from HIP combine".into());
                    }
                } else {
                    out.insert(key.into(), q("skipped_not_gfx1201"));
                }
            }
            let h_eager = md5_hex(&live_bytes(&g, &ygu, &ydn));
            out.insert("hash_live".into(), q(&h_eager));
            // Rerun with other poison: live bytes identical.
            poison(&gpu, &c, 0x00)?;
            cand_all(&mut gpu, &c)?;
            gpu.hip.device_synchronize()?;
            let (ygu2, ydn2) = outputs(&gpu, &c)?;
            let rerun = md5_hex(&live_bytes(&g, &ygu2, &ydn2)) == h_eager;
            out.insert("eager_rerun_identical".into(), rerun.to_string());
            if !rerun {
                fails.push("nondeterministic eager rerun".into());
            }
            if mode == "graph" {
                gpu.ensure_capture_stream()?;
                cand_all(&mut gpu, &c)?;
                gpu.hip.device_synchronize()?;
                gpu.begin_stream_capture()?;
                cand_all(&mut gpu, &c)?;
                let graph = gpu.end_stream_capture()?;
                let exec = gpu.hip.graph_instantiate(&graph)?;
                let mut all_eq = true;
                for r in 0..5 {
                    poison(&gpu, &c, 0x11 * (r + 1))?;
                    gpu.hip.device_synchronize()?;
                    gpu.launch_graph(&exec)?;
                    gpu.hip.device_synchronize()?;
                    let (a, d) = outputs(&gpu, &c)?;
                    all_eq &= md5_hex(&live_bytes(&g, &a, &d)) == h_eager;
                }
                out.insert("graph_replay_5x_equal_eager".into(), all_eq.to_string());
                if !all_eq {
                    fails.push("graph replay differs from eager".into());
                }

                // Changed-input replay: rewrite routing and activations at the
                // captured pointers, replay, and compare every output, grouping
                // array and both sidecars against a fresh eager run of the same
                // inputs; also against the CPU grouping and exact folds.
                let mut rounds: Vec<(&str, Vec<i32>, Vec<f32>)> = Vec::new();
                let mut r = Rng(seed ^ 0x6a6a);
                let distinct = |pool: &dyn Fn(&mut Rng) -> i32, r: &mut Rng| -> Vec<i32> {
                    let mut v = Vec::with_capacity(c.l);
                    for _ in 0..c.t {
                        let mut row: Vec<i32> = Vec::with_capacity(geo().topk);
                        while row.len() < geo().topk {
                            let e = pool(r);
                            if !row.contains(&e) {
                                row.push(e);
                            }
                        }
                        v.extend(row);
                    }
                    v
                };
                let uniform = distinct(&|r: &mut Rng| r.below(geo().e as u64) as i32, &mut r);
                // Skewed: experts 0..15 drawn 8x as often as the rest.
                let skewed = distinct(
                    &|r: &mut Rng| {
                        if r.below(2) == 0 {
                            r.below(16) as i32
                        } else {
                            r.below(geo().e as u64) as i32
                        }
                    },
                    &mut r,
                );
                // Empty experts: 64..512 only, so experts 0..63 receive no slot.
                let sparse = distinct(
                    &|r: &mut Rng| 64 + r.below((geo().e - 64) as u64) as i32,
                    &mut r,
                );
                let newx = |scale: f32, r: &mut Rng| -> Vec<f32> {
                    (0..c.t * geo().hid)
                        .map(|_| (r.unit() - 0.5) * scale)
                        .collect()
                };
                let (x1, x2, x3) = (newx(3.0, &mut r), newx(9.0, &mut r), newx(0.5, &mut r));
                rounds.push(("uniform", uniform, x1));
                rounds.push(("skewed", skewed, x2));
                rounds.push(("empty_experts_0_63", sparse, x3));
                rounds.push(("original", c.topk_host.clone(), x.clone()));
                let poison_all = |gpu: &Gpu, v: i32| -> Res<()> {
                    poison(gpu, &c, v)?;
                    for (t, n) in [
                        (&c.sorted, c.pmax * 4),
                        (&c.tiles, c.pmax / 16 * 4),
                        (&c.inverse, c.l * 4),
                        (&c.counts, geo().e * 4),
                        (&c.offsets, (geo().e + 1) * 4),
                    ] {
                        gpu.hip.memset(&t.buf, v, n)?;
                    }
                    let gs = gpu
                        .scratch
                        .int4_mmq_x_scratch
                        .as_ref()
                        .ok_or("no gate sidecar")?;
                    gpu.hip.memset(gs, v, (geo().hid / 128) * c.t * BLK)?;
                    let ds = gpu
                        .scratch
                        .qwen4_moe_down_i4_scratch
                        .as_ref()
                        .ok_or("no down sidecar")?;
                    gpu.hip
                        .memset(ds, v, (geo().mi / 128) * c.pmax.max(c.l) * BLK)?;
                    gpu.hip.device_synchronize()?;
                    Ok(())
                };
                let snapshot = |gpu: &Gpu| -> Res<Vec<(&'static str, Vec<u8>)>> {
                    let (a, d) = outputs(gpu, &c)?;
                    let (xg, xd) = sidecars(gpu, &c)?;
                    Ok(vec![
                        ("y_gate_up", a),
                        ("y_down", d),
                        ("sorted", download(gpu, &c.sorted.buf, c.pmax * 4)?),
                        ("tiles", download(gpu, &c.tiles.buf, c.pmax / 16 * 4)?),
                        ("inverse", download(gpu, &c.inverse.buf, c.l * 4)?),
                        ("counts", download(gpu, &c.counts.buf, geo().e * 4)?),
                        ("offsets", download(gpu, &c.offsets.buf, (geo().e + 1) * 4)?),
                        ("gate_sidecar", xg),
                        ("down_sidecar", xd),
                    ])
                };
                let mut prev_hash = h_eager.clone();
                let mut round_reports = Vec::new();
                for (name, topk, xr) in &rounds {
                    gpu.hip.memcpy_htod(&c.topk.buf, &i32_bytes(topk))?;
                    let xb: Vec<u8> = xr.iter().flat_map(|v| v.to_le_bytes()).collect();
                    gpu.hip.memcpy_htod(&c.x_in.buf, &xb)?;
                    poison_all(&gpu, 0x5A)?;
                    gpu.launch_graph(&exec)?;
                    gpu.hip.device_synchronize()?;
                    let replay = snapshot(&gpu)?;
                    poison_all(&gpu, 0x5A)?;
                    cand_all(&mut gpu, &c)?;
                    gpu.hip.device_synchronize()?;
                    let eager = snapshot(&gpu)?;
                    let am =
                        anchor_mismatch(&mut gpu, &c, &anchors, 0x5A, &eager[0].1, &eager[1].1)?;
                    let at = anchor_total(&am);
                    let mut mism = Vec::new();
                    let mut total = 0usize;
                    for ((n, a), (_, b)) in replay.iter().zip(&eager) {
                        let d = a.iter().zip(b).filter(|(x, y)| x != y).count();
                        total += d;
                        mism.push(format!("\"{n}\":{d}"));
                    }
                    // Host references on the replayed bytes.
                    let gr = cpu_group(topk, c.pmax);
                    let grp_ok = to_i32(&replay[2].1) == gr.sorted
                        && to_i32(&replay[3].1) == gr.tiles
                        && to_i32(&replay[4].1) == gr.inverse
                        && to_i32(&replay[5].1) == gr.counts
                        && to_i32(&replay[6].1) == gr.offsets;
                    let used = gr.offsets[geo().e] as usize / 16;
                    let (gm, dm, pb) = fold_check(
                        &mut gpu,
                        &c,
                        &gr,
                        &sample_tiles(used),
                        &replay[7].1,
                        &replay[8].1,
                        &replay[0].1,
                        &replay[1].1,
                    )?;
                    let hash = md5_hex(&live_bytes(&gr, &replay[0].1, &replay[1].1));
                    let changed = hash != prev_hash;
                    let empty = gr.counts.iter().filter(|&&n| n == 0).count();
                    let ok = total == 0
                        && grp_ok
                        && gm + dm + pb == 0
                        && at == 0
                        && (changed || *name == "original")
                        && (*name != "original" || hash == h_eager);
                    if !ok {
                        fails.push(format!("changed-input graph replay round {name}"));
                    }
                    round_reports.push(format!(
                        "{{\"round\":\"{name}\",\"empty_experts\":{empty},\"padded_P\":{},\"replay_vs_eager_mismatch\":{{{}}},\"replay_group_matches_cpu\":{grp_ok},\"replay_fold_mismatch\":[{gm},{dm},{pb}],\"eager_vs_anchor_byte_mismatch\":{},\"live_hash\":\"{hash}\",\"changed_vs_previous_round\":{changed},\"pass\":{ok}}}",
                        gr.offsets[geo().e],
                        mism.join(","),
                        anchor_json(&am)
                    ));
                    prev_hash = hash;
                }
                out.insert(
                    "graph_changed_input_rounds".into(),
                    format!("[{}]", round_reports.join(",")),
                );

                // Stale prepared handles must be refused once the producer
                // generation advances, never silently consumed.
                let stale_gate = gpu.qwen4_moe_rotate256_i4(&c.x_in, geo().hid, c.t)?;
                let _fresh_gate = gpu.qwen4_moe_rotate256_i4(&c.x_in, geo().hid, c.t)?;
                let gate_err = gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                    &c.gu_ptrs,
                    &c.tiles,
                    &c.sorted,
                    &stale_gate,
                    &c.ygu,
                    geo().gu_m,
                    geo().hid,
                    geo().topk,
                    c.pmax,
                    c.t,
                    c.host,
                );
                let stale_down =
                    gpu.qwen4_moe_rotate128_i4(&c.ygu, &c.sorted, geo().mi, c.pmax, c.l)?;
                let fresh_down =
                    gpu.qwen4_moe_rotate128_i4(&c.ygu, &c.sorted, geo().mi, c.pmax, c.l)?;
                let down_err = gpu.gemm_qwen4_moe_down_iu4_sym(
                    &c.dn_ptrs,
                    &c.tiles,
                    &c.sorted,
                    &stale_down,
                    &c.ydn,
                    geo().dn_m,
                    geo().mi,
                    1,
                    c.pmax,
                    c.l,
                    c.host,
                );
                // A live handle of the other sidecar is refused too.
                let cross_err = gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
                    &c.gu_ptrs,
                    &c.tiles,
                    &c.sorted,
                    &fresh_down,
                    &c.ygu,
                    geo().gu_m,
                    geo().hid,
                    geo().topk,
                    c.pmax,
                    c.t,
                    c.host,
                );
                gpu.hip.device_synchronize()?;
                let msg = |r: &hip_bridge::HipResult<()>| match r {
                    Ok(()) => "\"accepted\"".to_string(),
                    Err(e) => format!("\"{}\"", e.to_string().replace('"', "'")),
                };
                out.insert("stale_gate_handle".into(), msg(&gate_err));
                out.insert("stale_down_handle".into(), msg(&down_err));
                out.insert("cross_sidecar_handle".into(), msg(&cross_err));
                if gate_err.is_ok() || down_err.is_ok() || cross_err.is_ok() {
                    fails.push("stale or mismatched IU4 handle accepted".into());
                }
            }
            // General-input delta vs the incumbent (informative only; the
            // incumbent envelope is the shipped fn route, so a3b skips it).
            if geo().name == "fn" {
                let mut h = Handles::default();
                for s in 0..5 {
                    inc_stage(&mut gpu, &c, &mut h, s)?;
                }
                gpu.hip.device_synchronize()?;
                let isorted = to_i32(&download(&gpu, &c.i_sorted.buf, c.pmax * 4)?);
                // gfx1151's WMMA down stores BF16 rows, gfx1201's SIMT down F32.
                let f32_rows = gpu.arch == "gfx1201";
                let ydi = download(&gpu, &c.i_ydn.buf, c.pmax * geo().dn_m * 4)?;
                let (mut num, mut den) = (0f64, 0f64);
                for (pi, &s) in isorted.iter().enumerate() {
                    if s < 0 {
                        continue;
                    }
                    let pc = g.inverse[s as usize] as usize;
                    for col in (0..geo().dn_m).step_by(7) {
                        let i = pi * geo().dn_m + col;
                        let inc = if f32_rows {
                            f32::from_le_bytes(ydi[4 * i..4 * i + 4].try_into().unwrap()) as f64
                        } else {
                            f32::from_bits(
                                (u16::from_le_bytes([ydi[2 * i], ydi[2 * i + 1]]) as u32) << 16,
                            ) as f64
                        };
                        let o = (pc * geo().dn_m + col) * 2;
                        let cand =
                            f32::from_bits((u16::from_le_bytes([ydn[o], ydn[o + 1]]) as u32) << 16)
                                as f64;
                        num += (cand - inc) * (cand - inc);
                        den += inc * inc;
                    }
                }
                out.insert(
                    "down_vs_incumbent_rel_rms_informative".into(),
                    format!("{:.6e}", (num / den.max(1e-30)).sqrt()),
                );
            } else {
                out.insert(
                    "down_vs_incumbent_rel_rms_informative".into(),
                    q("n/a_fn_only"),
                );
            }
        }

        if mode == "stress" {
            // Rotating poison, every iteration's live bytes vs the first.
            let (mut first, mut bad) = (None, 0usize);
            for it in 0..iters {
                poison(&gpu, &c, (0x11 * (it % 15) + 0x0F) as i32)?;
                cand_all(&mut gpu, &c)?;
                gpu.hip.device_synchronize()?;
                let (a, d) = outputs(&gpu, &c)?;
                let h = md5_hex(&live_bytes(&g, &a, &d));
                match &first {
                    None => first = Some(h),
                    Some(f) => bad += (*f != h) as usize,
                }
            }
            out.insert("stress_pid".into(), std::process::id().to_string());
            out.insert("stress_iters".into(), iters.to_string());
            out.insert("stress_iterations_differing".into(), bad.to_string());
            out.insert("hash_live".into(), q(first.as_deref().unwrap_or("")));
            if bad != 0 {
                fails.push("stress mismatch".into());
            }
        }

        if mode == "time" {
            // Arms: the shipped incumbent, the candidate route and the same
            // route with each expert-run anchor's GEMMs (gfx1151: the hipcc
            // NT4; gfx1201: the other tile width). Two warm passes each, then
            // a palindromic order per iteration (ABC, CBA, ...) so drift
            // charges every arm alike.
            let mut arms: Vec<(&str, Stage, Option<Entry>)> = Vec::new();
            if geo().name == "fn" {
                arms.push(("incumbent", inc_stage, None));
            }
            arms.push(("candidate", cand_stage, None));
            for &e in anchors.iter().filter(|e| e.gu_block == 128) {
                arms.push((e.name, anchor_stage, Some(e)));
            }
            for _ in 0..2 {
                for &(_, f, entry) in &arms {
                    let mut h = Handles {
                        entry,
                        ..Handles::default()
                    };
                    for s in 0..5 {
                        f(&mut gpu, &c, &mut h, s)?;
                    }
                }
            }
            gpu.hip.device_synchronize()?;
            let mut tot = vec![Vec::new(); arms.len()];
            let mut st = vec![vec![Vec::new(); 5]; arms.len()];
            for i in 0..iters {
                let order: Vec<usize> = if i % 2 == 0 {
                    (0..arms.len()).collect()
                } else {
                    (0..arms.len()).rev().collect()
                };
                for a in order {
                    let (sv, t) = timed(&mut gpu, &c, arms[a].1, arms[a].2)?;
                    tot[a].push(t);
                    for s in 0..5 {
                        st[a][s].push(sv[s]);
                    }
                }
            }
            out.insert("time_iters".into(), iters.to_string());
            out.insert("time_pid".into(), std::process::id().to_string());
            for (a, (name, _, _)) in arms.iter().enumerate() {
                out.insert(format!("{name}_envelope_ms_min_med_max"), stats(&tot[a]));
                for s in 0..5 {
                    out.insert(format!("{name}_stage{s}_ms"), stats(&st[a][s]));
                }
            }
        }
    }
    out.insert("fails".into(), format!("{fails:?}"));
    out.insert(
        "result".into(),
        q(if fails.is_empty() { "PASS" } else { "FAIL" }),
    );
    let body: Vec<String> = out.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    println!("{{{}}}", body.join(","));
    if fails.is_empty() {
        Ok(())
    } else {
        std::process::exit(1)
    }
}

/// NT4 vs NT8 wide-tile byte oracle (`moe_iu4_wide_tiles_oracle` example): a
/// child module, so it reuses this file's private oracle machinery.
#[path = "moe_iu4_wide_tiles_oracle/lab.rs"]
#[allow(dead_code)]
pub mod wide_tiles;
