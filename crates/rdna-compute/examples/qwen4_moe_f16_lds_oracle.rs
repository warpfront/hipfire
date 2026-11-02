// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 F16 WMMA grouped MoE LDS-share ORACLE (gfx1151 only; developer
//! example).
//!
//! PR #832 adds two gfx1151 entries whose row-tile waves share each expert
//! run's X through LDS:
//!     gate/up SiLU  `gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out_x4`
//!                   (grid [M/64, P/16], block 128)
//!     down          `gemm_mq4g128v2_moe_grouped_wmma_gfx1151_x8_bf16out`
//!                   (grid [M/128, P/16], block 256)
//! and claims both are bitwise identical to the per-wave incumbents
//!     `gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out` (grid [M/16, P/16], block 32)
//!     `gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out` (grid [M/16, P/16], block 32).
//! This example proves it on the production Qwen4 envelope (E=512, top-10,
//! HID=2560, MI=640, gate/up M=1280, down M=2560) over synthetic ASYMMETRIC
//! QT44 / QT53 experts (every header's scale and zero point independent
//! normal F16, ~1/64 zero headers, random nibbles) and times the entries.
//!
//! Per case the production chain runs: `moe_scatter_fused_top10`,
//! `rotate_x_mq_batched_f16`, the gate/up arms, `moe_unscatter_rotate128_f16`
//! over the per-wave gate/up output, the down arms. Each GEMM runs three arms
//! into its own 0x5A-poisoned buffer (GEMM bytes + 4096 guard bytes):
//!     (a) the per-wave entry, launched directly (the oracle),
//!     (b) the LDS entry, launched directly,
//!     (c) the production launcher
//!         (`gemm_mq4g256v2_moe_grouped_top10_silu_bf16out`,
//!          `gemm_mq4g128v2_moe_grouped_top10_xf16`), which picks its entry
//!          from the device and the `HIPFIRE_QWEN4_MOE_X_LDS` flag (default
//!          `auto`: the LDS entry on gfx1151; with `=0`, (c) is the per-wave
//!          entry again; this example reads no environment variable itself).
//! The direct launches load the production module names, so the oracle and
//! the launcher share one compiled object. (b) and (c) must equal (a) over
//! the WHOLE buffer, poison and guard bytes included.
//!
//! Non-vacuity per case: live tiles, the histogram of expert-run lengths in
//! tiles, the count of non-poison non-zero BF16 values in the per-wave live
//! rows of each output (must be > 0), and md5s of the routing, X and the
//! per-wave buffers. Negative control (first case of the process): one header
//! byte of the weight blob of expert `tiles[0]` is changed on the device, only
//! the LDS arm reruns and must differ from the per-wave buffer, then the byte
//! is restored (gate/up with x4, down with x8).
//!
//! CLI: --mode check|time (default check) --routing LIST (one,all,skew)
//!      --tokens LIST (512,777,1131,2048,4096) --seeds LIST (7,11)
//!      --iters N (time mode, default 30) --distinct N (distinct expert
//!      blobs per projection, default 64; expert e uses blob e % N)
//! check runs every routing x tokens x seed in one process. time runs the
//! same grid: six arms (gu_wave gu_x4 gu_launcher dn_wave dn_x8 dn_launcher),
//! 2 warm passes then `--iters` hip-event timed iterations in palindromic
//! order (even iteration forward, odd reversed) per GEMM group.
//! Output: one JSON object per case on stdout, then one summary
//! `{"result":"PASS"|"FAIL","cases":N,"fails":[...]}`; exit 1 on any FAIL.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const E: usize = 512;
const TOPK: usize = 10;
const HID: usize = 2560;
const MI: usize = 640;
const GU_M: usize = 2 * MI;
const DN_M: usize = HID;
/// QT44 / MQ4G256V2: per 256 K, two 4-byte (sc, zp) headers + 128 nibble bytes.
const GU_GB: usize = 136;
/// QT53 / MQ4G128V2: per 128 K, one 4-byte (sc, zp) header + 64 nibble bytes.
const DN_GB: usize = 68;
const GUARD: usize = 4096;
const POISON: i32 = 0x5A;

const GU_STRIDE: usize = GU_M * (HID / 256) * GU_GB;
const DN_STRIDE: usize = DN_M * (MI / 128) * DN_GB;

const GU_SRC: &str =
    include_str!("../../../kernels/src/qwen4_gemm_mq4g256v2_moe_grouped_wmma_k2.hip");
const DN_SRC: &str =
    include_str!("../../../kernels/src/gemm_mq4g128v2_moe_grouped_wmma.gfx1151.hip");
const GU_WAVE: &str = "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out";
const GU_X4: &str = "gemm_mq4g256v2_moe_grouped_wmma_k2_silu_bf16out_x4";
const DN_MODULE: &str = "gemm_mq4g128v2_moe_grouped_wmma_gfx1151";
const DN_WAVE: &str = "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out";
const DN_X8: &str = "gemm_mq4g128v2_moe_grouped_wmma_gfx1151_x8_bf16out";

type Res<T> = Result<T, Box<dyn std::error::Error>>;
type Out = BTreeMap<String, Value>;

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

/// Random normal F16: random sign, exponent field 3..=11, random mantissa.
fn f16_normal(r: &mut Rng) -> u16 {
    let sign = (r.below(2) as u16) << 15;
    let exp = 3 + r.below(9) as u16;
    let man = r.below(1024) as u16;
    sign | (exp << 10) | man
}

/// Asymmetric header (sc, zp): independent normals, ~1/64 zero headers.
fn asym_header(r: &mut Rng) -> (u16, u16) {
    if r.below(64) == 0 {
        (0, 0)
    } else {
        (f16_normal(r), f16_normal(r))
    }
}

/// One expert's rows: QT44 (136 B groups, K % 256) or QT53 (68 B groups,
/// K % 128), independent headers, uniformly random nibble bytes.
fn asym_weights(m: usize, k: usize, group_bytes: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed ^ 0x5eed_0000_0000_0001);
    let (groups, hdrs) = if group_bytes == GU_GB {
        (k / 256, 2)
    } else {
        (k / 128, 1)
    };
    let mut w = vec![0u8; m * groups * group_bytes];
    for g in w.chunks_exact_mut(group_bytes) {
        for h in 0..hdrs {
            let (sc, zp) = asym_header(&mut r);
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

fn md5_block(st: &mut [u32; 4], ch: &[u8], kk: &[u32; 64]) {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let mut m = [0u32; 16];
    for (d, w) in m.iter_mut().zip(ch.chunks_exact(4)) {
        *d = u32::from_le_bytes(w.try_into().unwrap());
    }
    let (mut a, mut b1, mut c, mut d) = (st[0], st[1], st[2], st[3]);
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
        b1 = b1.wrapping_add(f2.rotate_left(S[i]));
    }
    st[0] = st[0].wrapping_add(a);
    st[1] = st[1].wrapping_add(b1);
    st[2] = st[2].wrapping_add(c);
    st[3] = st[3].wrapping_add(d);
}

fn md5_hex(b: &[u8]) -> String {
    let kk: [u32; 64] =
        std::array::from_fn(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32);
    let mut st = [0x67452301u32, 0xefcdab89, 0x98badcfe, 0x10325476];
    let mut chunks = b.chunks_exact(64);
    for ch in &mut chunks {
        md5_block(&mut st, ch, &kk);
    }
    let mut tail = chunks.remainder().to_vec();
    tail.push(0x80);
    while tail.len() % 64 != 56 {
        tail.push(0);
    }
    tail.extend_from_slice(&(b.len() as u64).wrapping_mul(8).to_le_bytes());
    for ch in tail.chunks_exact(64) {
        md5_block(&mut st, ch, &kk);
    }
    st.iter()
        .flat_map(|v| v.to_le_bytes())
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn routing(kind: &str, t: usize, seed: u64) -> Res<Vec<i32>> {
    let mut r = Rng(seed ^ 0xabcdef);
    Ok(match kind {
        // Every slot to expert 0: the hottest-possible single bucket.
        "one" => vec![0; t * TOPK],
        // Round robin over all 512 experts.
        "all" => (0..t * TOPK).map(|i| (i % E) as i32).collect(),
        // Five hot experts on every token plus five distinct uniform others.
        "skew" => {
            let mut v = Vec::with_capacity(t * TOPK);
            for _ in 0..t {
                let mut row: Vec<i32> = (0..5).collect();
                while row.len() < TOPK {
                    let e = 5 + r.below((E - 5) as u64) as i32;
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                v.extend(row);
            }
            v
        }
        other => return Err(format!("--routing {other}: expected one, all or skew").into()),
    })
}

/// Live tiles and the histogram of expert-run lengths in tiles
/// (1, 2, 3, 4, >4); a run is a maximal stretch of one expert's tiles.
fn tile_stats(tiles: &[i32]) -> (usize, [usize; 5]) {
    let (mut live, mut hist) = (0, [0usize; 5]);
    let mut i = 0;
    while i < tiles.len() {
        if tiles[i] < 0 {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < tiles.len() && tiles[j] == tiles[i] {
            j += 1;
        }
        live += j - i;
        hist[(j - i).min(5) - 1] += 1;
        i = j;
    }
    (live, hist)
}

/// Non-poison, non-zero BF16 values in the rows `sorted` marks live.
fn live_nonzero(buf: &[u8], sorted: &[i32], row_bytes: usize) -> usize {
    sorted
        .iter()
        .enumerate()
        .filter(|(_, &s)| s >= 0)
        .map(|(p, _)| {
            buf[p * row_bytes..(p + 1) * row_bytes]
                .chunks_exact(2)
                .filter(|c| {
                    let v = u16::from_le_bytes([c[0], c[1]]);
                    v != 0 && v != 0x5A5A
                })
                .count()
        })
        .sum()
}

fn mismatch(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn to_i32(b: &[u8]) -> Vec<i32> {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
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

fn download(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> Res<Vec<u8>> {
    let mut v = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf)?;
    Ok(v)
}

fn ptr_table(base: &GpuTensor, stride: usize, distinct: usize) -> Vec<u8> {
    (0..E)
        .flat_map(|e| ((base.buf.as_ptr() as u64) + ((e % distinct) * stride) as u64).to_le_bytes())
        .collect()
}

/// Overwrite one byte of the live allocation `t` on the device.
fn poke(gpu: &Gpu, t: &GpuTensor, off: usize, v: u8) -> Res<()> {
    if off >= t.buf.size() {
        return Err(format!("poke offset {off} outside a {} byte buffer", t.buf.size()).into());
    }
    let dst = t.buf.byte_view(off, 1);
    gpu.hip.memcpy_htod(&dst, &[v])?;
    gpu.hip.device_synchronize()?;
    Ok(())
}

/// The two GEMMs' kernarg ABI: 5 pointers, then M, K, x_row_div, m_total, n_experts (= E).
fn blob(ptrs: [*const std::ffi::c_void; 5], ints: [usize; 4]) -> KernargBlob {
    let mut a = KernargBlob::new();
    for q in ptrs {
        a.push_ptr(q);
    }
    for v in ints {
        a.push_i32(v as i32);
    }
    a.push_i32(E as i32);
    a
}

/// Synthetic experts of one seed: `distinct` blobs per projection (expert e
/// uses blob e % distinct) and the E-entry pointer tables.
struct Weights {
    distinct: usize,
    gu_w: Vec<u8>,
    dn_w: Vec<u8>,
    gu_t: GpuTensor,
    dn_t: GpuTensor,
    gu_ptrs: GpuTensor,
    dn_ptrs: GpuTensor,
    md5_gu: String,
    md5_dn: String,
}

impl Weights {
    fn new(gpu: &mut Gpu, seed: u64, distinct: usize) -> Res<Weights> {
        let mut gu_w = Vec::with_capacity(distinct * GU_STRIDE);
        let mut dn_w = Vec::with_capacity(distinct * DN_STRIDE);
        for b in 0..distinct as u64 {
            gu_w.extend(asym_weights(GU_M, HID, GU_GB, seed * 1000 + b));
            dn_w.extend(asym_weights(DN_M, MI, DN_GB, seed * 1000 + 600 + b));
        }
        let gu_t = gpu.upload_raw(&gu_w, &[gu_w.len()])?;
        let dn_t = gpu.upload_raw(&dn_w, &[dn_w.len()])?;
        let gu_ptrs = gpu.upload_raw(&ptr_table(&gu_t, GU_STRIDE, distinct), &[E * 8])?;
        let dn_ptrs = gpu.upload_raw(&ptr_table(&dn_t, DN_STRIDE, distinct), &[E * 8])?;
        Ok(Weights {
            distinct,
            md5_gu: md5_hex(&gu_w),
            md5_dn: md5_hex(&dn_w),
            gu_w,
            dn_w,
            gu_t,
            dn_t,
            gu_ptrs,
            dn_ptrs,
        })
    }

    fn free(self, gpu: &mut Gpu) -> Res<()> {
        for t in [self.gu_t, self.dn_t, self.gu_ptrs, self.dn_ptrs] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

/// One case's routing operands after the production scatter.
struct Ops {
    t: usize,
    l: usize,
    pmax: usize,
    topk: GpuTensor,
    x_in: GpuTensor,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
}

impl Ops {
    fn free(self, gpu: &mut Gpu) -> Res<()> {
        for t in [
            self.topk,
            self.x_in,
            self.counts,
            self.offsets,
            self.sorted,
            self.tiles,
            self.inverse,
        ] {
            gpu.free_tensor(t)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    GuWave,
    GuX4,
    GuLauncher,
    DnWave,
    DnX8,
    DnLauncher,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::GuWave => "gu_wave",
            Arm::GuX4 => "gu_x4",
            Arm::GuLauncher => "gu_launcher",
            Arm::DnWave => "dn_wave",
            Arm::DnX8 => "dn_x8",
            Arm::DnLauncher => "dn_launcher",
        }
    }

    /// Whole output buffer (GEMM bytes + guard) of this arm's GEMM.
    fn out_bytes(self, pmax: usize) -> usize {
        match self {
            Arm::GuWave | Arm::GuX4 | Arm::GuLauncher => pmax * MI * 2 + GUARD,
            Arm::DnWave | Arm::DnX8 | Arm::DnLauncher => pmax * DN_M * 2 + GUARD,
        }
    }
}

/// Weights + routing operands the arms run on.
struct Ctx<'a> {
    w: &'a Weights,
    o: &'a Ops,
}

impl Ctx<'_> {
    /// One launch of `arm` over F16 X `x` into `y` (gate/up: the rotated X
    /// scratch; down: the unscattered, rotated activations).
    fn run(&self, gpu: &mut Gpu, arm: Arm, x: &GpuTensor, y: &GpuTensor) -> Res<()> {
        let (w, o) = (self.w, self.o);
        let tiles = (o.pmax / 16) as u32;
        match arm {
            Arm::GuWave | Arm::GuX4 => {
                let x4 = arm == Arm::GuX4;
                let mut a = blob(
                    [p(&w.gu_ptrs), p(&o.tiles), p(&o.sorted), p(x), p(y)],
                    [GU_M, HID, TOPK, o.pmax],
                );
                let (f, rows_per_block, block) = if x4 {
                    (GU_X4, 64, 128)
                } else {
                    (GU_WAVE, 16, 32)
                };
                launch(
                    gpu,
                    f,
                    [(GU_M / rows_per_block) as u32, tiles, 1],
                    block,
                    &mut a,
                )
            }
            Arm::DnWave | Arm::DnX8 => {
                let x8 = arm == Arm::DnX8;
                let mut a = blob(
                    [p(&w.dn_ptrs), p(&o.tiles), p(&o.sorted), p(x), p(y)],
                    [DN_M, MI, 1, o.pmax],
                );
                let (f, rows_per_block, block) = if x8 {
                    (DN_X8, 128, 256)
                } else {
                    (DN_WAVE, 16, 32)
                };
                launch(
                    gpu,
                    f,
                    [(DN_M / rows_per_block) as u32, tiles, 1],
                    block,
                    &mut a,
                )
            }
            Arm::GuLauncher => Ok(gpu.gemm_mq4g256v2_moe_grouped_top10_silu_bf16out(
                &w.gu_ptrs, &o.tiles, &o.sorted, x, y, GU_M, HID, TOPK, o.pmax, o.t,
            )?),
            Arm::DnLauncher => Ok(gpu.gemm_mq4g128v2_moe_grouped_top10_xf16(
                &w.dn_ptrs, &o.tiles, &o.sorted, x, y, DN_M, MI, o.pmax, true,
            )?),
        }
    }

    /// Poison `y`, run `arm`, download the whole buffer.
    fn bytes(&self, gpu: &mut Gpu, arm: Arm, x: &GpuTensor, y: &GpuTensor) -> Res<Vec<u8>> {
        let n = arm.out_bytes(self.o.pmax);
        gpu.hip.memset(&y.buf, POISON, n)?;
        self.run(gpu, arm, x, y)?;
        gpu.hip.device_synchronize()?;
        download(gpu, y, n)
    }

    /// Negative control: change one header byte (sc high byte of row 0,
    /// group 0, XOR 0x40) of the blob expert `e0` uses, rerun the LDS `arm`
    /// only, restore the byte; returns the byte mismatches vs `wave`.
    fn negative(
        &self,
        gpu: &mut Gpu,
        arm: Arm,
        e0: usize,
        x: &GpuTensor,
        y: &GpuTensor,
        wave: &[u8],
    ) -> Res<usize> {
        let w = self.w;
        let blob_idx = e0 % w.distinct;
        let (t, off, orig) = if arm == Arm::GuX4 {
            let off = blob_idx * GU_STRIDE + 1;
            (&w.gu_t, off, w.gu_w[off])
        } else {
            let off = blob_idx * DN_STRIDE + 1;
            (&w.dn_t, off, w.dn_w[off])
        };
        poke(gpu, t, off, orig ^ 0x40)?;
        let r = self.bytes(gpu, arm, x, y);
        poke(gpu, t, off, orig)?;
        Ok(mismatch(&r?, wave))
    }
}

fn load_kernels(gpu: &mut Gpu) -> Res<()> {
    gpu.ensure_kernel_public(GU_WAVE, GU_SRC, GU_WAVE)?;
    gpu.ensure_kernel_public(GU_X4, GU_SRC, GU_X4)?;
    gpu.ensure_kernel_public(DN_MODULE, DN_SRC, DN_WAVE)?;
    gpu.ensure_kernel_public(DN_MODULE, DN_SRC, DN_X8)?;
    Ok(())
}

struct CaseId {
    kind: String,
    t: usize,
    seed: u64,
}

/// Routing, X and the production scatter of one case; fills the common
/// report keys. Returns the operands and the host sorted/tiles tables.
fn prepare(
    gpu: &mut Gpu,
    w: &Weights,
    id: &CaseId,
    mode: &str,
) -> Res<(Ops, Vec<i32>, Vec<i32>, Out)> {
    let topk = routing(&id.kind, id.t, id.seed)?;
    let mut r = Rng(id.seed ^ 0x77);
    let x: Vec<f32> = (0..id.t * HID).map(|_| (r.unit() - 0.5) * 4.0).collect();
    let l = id.t * TOPK;
    let pmax = (l + E * 15).div_ceil(16) * 16;
    let o = Ops {
        t: id.t,
        l,
        pmax,
        topk: gpu.upload_raw(&i32_bytes(&topk), &[l])?,
        x_in: gpu.upload_f32(&x, &[id.t * HID])?,
        counts: alloc(gpu, E * 4)?,
        offsets: alloc(gpu, (E + 1) * 4)?,
        sorted: alloc(gpu, pmax * 4)?,
        tiles: alloc(gpu, pmax / 16 * 4)?,
        inverse: alloc(gpu, l * 4)?,
    };
    gpu.moe_scatter_fused_top10(
        &o.topk, &o.counts, &o.offsets, &o.sorted, &o.tiles, &o.inverse, l, E, pmax, 16,
    )?;
    gpu.hip.device_synchronize()?;
    let sorted = to_i32(&download(gpu, &o.sorted, pmax * 4)?);
    let tiles = to_i32(&download(gpu, &o.tiles, pmax / 16 * 4)?);
    let (live, hist) = tile_stats(&tiles);
    let mut out = Out::new();
    out.insert("arch".into(), json!(gpu.arch));
    out.insert("mode".into(), json!(mode));
    out.insert("routing".into(), json!(id.kind));
    out.insert("tokens".into(), json!(id.t));
    out.insert("seed".into(), json!(id.seed));
    out.insert("distinct".into(), json!(w.distinct));
    out.insert("slots_L".into(), json!(l));
    out.insert("pmax".into(), json!(pmax));
    out.insert("live_tiles".into(), json!(live));
    out.insert(
        "run_len_tiles_hist".into(),
        json!({"1": hist[0], "2": hist[1], "3": hist[2], "4": hist[3], ">4": hist[4]}),
    );
    out.insert("md5_routing".into(), json!(md5_hex(&i32_bytes(&topk))));
    out.insert(
        "md5_x".into(),
        json!(md5_hex(
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()
        )),
    );
    out.insert("md5_sorted".into(), json!(md5_hex(&i32_bytes(&sorted))));
    out.insert("md5_tiles".into(), json!(md5_hex(&i32_bytes(&tiles))));
    out.insert("md5_gu_w".into(), json!(w.md5_gu));
    out.insert("md5_dn_w".into(), json!(w.md5_dn));
    Ok((o, sorted, tiles, out))
}

fn case_label(id: &CaseId) -> String {
    format!("{}/t{}/s{}", id.kind, id.t, id.seed)
}

/// One check case; returns its report and failure descriptions.
fn check_case(gpu: &mut Gpu, w: &Weights, id: &CaseId, first: bool) -> Res<(Out, Vec<String>)> {
    let (o, sorted, tiles, mut out) = prepare(gpu, w, id, "check")?;
    let ctx = Ctx { w, o: &o };
    let mut fails: Vec<String> = Vec::new();
    let label = case_label(id);
    let mut bad = |what: &str, n: usize| {
        if n != 0 {
            fails.push(format!("{label}: {what}={n}"));
        }
    };
    let gu_bytes = Arm::GuWave.out_bytes(o.pmax);
    let dn_bytes = Arm::DnWave.out_bytes(o.pmax);
    let ygu_wave = alloc(gpu, gu_bytes)?;
    let ydn_wave = alloc(gpu, dn_bytes)?;
    let ytmp_gu = alloc(gpu, gu_bytes)?;
    let ytmp_dn = alloc(gpu, dn_bytes)?;
    let e0 = tiles[0];
    let first_expert = if e0 >= 0 { Some(e0 as usize) } else { None };

    // Gate/up: the rotated F16 X is the production scratch, consumed before
    // the next producer (the unscatter below).
    let xf16 = gpu.rotate_x_mq_batched_f16(&o.x_in, HID, o.t)?;
    let gu_wave = ctx.bytes(gpu, Arm::GuWave, &xf16, &ygu_wave)?;
    let gu_x4 = ctx.bytes(gpu, Arm::GuX4, &xf16, &ytmp_gu)?;
    let gu_launcher = ctx.bytes(gpu, Arm::GuLauncher, &xf16, &ytmp_gu)?;
    let gu_nonzero = live_nonzero(&gu_wave, &sorted, MI * 2);
    out.insert("md5_gu_wave".into(), json!(md5_hex(&gu_wave)));
    out.insert("gu_buffer_bytes".into(), json!(gu_bytes));
    out.insert("gu_live_nonzero_bf16".into(), json!(gu_nonzero));
    out.insert(
        "gu_x4_vs_wave_mismatch".into(),
        json!(mismatch(&gu_x4, &gu_wave)),
    );
    out.insert(
        "gu_launcher_vs_wave_mismatch".into(),
        json!(mismatch(&gu_launcher, &gu_wave)),
    );
    bad("gu_x4_vs_wave", mismatch(&gu_x4, &gu_wave));
    bad("gu_launcher_vs_wave", mismatch(&gu_launcher, &gu_wave));
    if gu_nonzero == 0 {
        bad("gu_live_nonzero_bf16_is_zero", 1);
    }
    if first {
        match first_expert {
            Some(e) => {
                let n = ctx.negative(gpu, Arm::GuX4, e, &xf16, &ytmp_gu, &gu_wave)?;
                out.insert("neg_gu_x4_mismatch".into(), json!(n));
                if n == 0 {
                    bad("neg_gu_x4_control_undetected", 1);
                }
            }
            None => bad("neg_control_tiles0_not_live", 1),
        }
    }

    // Down: X is the unscattered, rotated per-slot activation of the
    // per-wave gate/up output (again the shared scratch).
    let x16 = gpu.moe_unscatter_rotate128_f16(&ygu_wave, &o.sorted, MI, o.pmax, o.l)?;
    let dn_wave = ctx.bytes(gpu, Arm::DnWave, &x16, &ydn_wave)?;
    let dn_x8 = ctx.bytes(gpu, Arm::DnX8, &x16, &ytmp_dn)?;
    let dn_launcher = ctx.bytes(gpu, Arm::DnLauncher, &x16, &ytmp_dn)?;
    let dn_nonzero = live_nonzero(&dn_wave, &sorted, DN_M * 2);
    out.insert("md5_dn_wave".into(), json!(md5_hex(&dn_wave)));
    out.insert("dn_buffer_bytes".into(), json!(dn_bytes));
    out.insert("dn_live_nonzero_bf16".into(), json!(dn_nonzero));
    out.insert(
        "dn_x8_vs_wave_mismatch".into(),
        json!(mismatch(&dn_x8, &dn_wave)),
    );
    out.insert(
        "dn_launcher_vs_wave_mismatch".into(),
        json!(mismatch(&dn_launcher, &dn_wave)),
    );
    bad("dn_x8_vs_wave", mismatch(&dn_x8, &dn_wave));
    bad("dn_launcher_vs_wave", mismatch(&dn_launcher, &dn_wave));
    if dn_nonzero == 0 {
        bad("dn_live_nonzero_bf16_is_zero", 1);
    }
    if first {
        if let Some(e) = first_expert {
            let n = ctx.negative(gpu, Arm::DnX8, e, &x16, &ytmp_dn, &dn_wave)?;
            out.insert("neg_dn_x8_mismatch".into(), json!(n));
            if n == 0 {
                bad("neg_dn_x8_control_undetected", 1);
            }
        }
    }

    out.insert("fails".into(), json!(fails));
    out.insert(
        "result".into(),
        json!(if fails.is_empty() { "PASS" } else { "FAIL" }),
    );
    for t in [ygu_wave, ydn_wave, ytmp_gu, ytmp_dn] {
        gpu.free_tensor(t)?;
    }
    o.free(gpu)?;
    Ok((out, fails))
}

/// Warm passes then `iters` palindromic iterations of `arms`, each launch
/// timed with its own pair of hip events; min/median/max ms per arm.
fn time_arms(
    gpu: &mut Gpu,
    ctx: &Ctx,
    arms: &[Arm],
    x: &GpuTensor,
    y: &GpuTensor,
    iters: usize,
    out: &mut Out,
) -> Res<()> {
    for _ in 0..2 {
        for &a in arms {
            ctx.run(gpu, a, x, y)?;
        }
    }
    gpu.hip.device_synchronize()?;
    let (e0, e1) = (gpu.hip.event_create()?, gpu.hip.event_create()?);
    let mut ms = vec![Vec::with_capacity(iters); arms.len()];
    for i in 0..iters {
        let order: Vec<usize> = if i % 2 == 0 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for a in order {
            gpu.hip.event_record(&e0, None)?;
            ctx.run(gpu, arms[a], x, y)?;
            gpu.hip.event_record(&e1, None)?;
            gpu.hip.event_synchronize(&e1)?;
            ms[a].push(gpu.hip.event_elapsed_ms(&e0, &e1)? as f64);
        }
    }
    for (a, v) in arms.iter().zip(ms.iter_mut()) {
        v.sort_by(|x, y| x.total_cmp(y));
        let r = |x: f64| (x * 1e4).round() / 1e4;
        out.insert(
            format!("{}_ms_min_med_max", a.name()),
            json!([r(v[0]), r(v[v.len() / 2]), r(v[v.len() - 1])]),
        );
    }
    Ok(())
}

/// One time case: the gate/up group over the rotated X, then the down group
/// over the activations of the per-wave gate/up output.
fn time_case(gpu: &mut Gpu, w: &Weights, id: &CaseId, iters: usize) -> Res<(Out, Vec<String>)> {
    let (o, _sorted, _tiles, mut out) = prepare(gpu, w, id, "time")?;
    let ctx = Ctx { w, o: &o };
    let ygu = alloc(gpu, Arm::GuWave.out_bytes(o.pmax))?;
    let ydn = alloc(gpu, Arm::DnWave.out_bytes(o.pmax))?;
    let xf16 = gpu.rotate_x_mq_batched_f16(&o.x_in, HID, o.t)?;
    time_arms(
        gpu,
        &ctx,
        &[Arm::GuWave, Arm::GuX4, Arm::GuLauncher],
        &xf16,
        &ygu,
        iters,
        &mut out,
    )?;
    ctx.bytes(gpu, Arm::GuWave, &xf16, &ygu)?;
    let x16 = gpu.moe_unscatter_rotate128_f16(&ygu, &o.sorted, MI, o.pmax, o.l)?;
    time_arms(
        gpu,
        &ctx,
        &[Arm::DnWave, Arm::DnX8, Arm::DnLauncher],
        &x16,
        &ydn,
        iters,
        &mut out,
    )?;
    out.insert("iters".into(), json!(iters));
    out.insert("fails".into(), json!(Vec::<String>::new()));
    out.insert("result".into(), json!("PASS"));
    for t in [ygu, ydn] {
        gpu.free_tensor(t)?;
    }
    o.free(gpu)?;
    Ok((out, Vec::new()))
}

// ---------------------------------------------------------------- main

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn list<T: std::str::FromStr>(args: &[String], k: &str, d: &str) -> Res<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    let v = arg(args, k, d)
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<T>()
                .map_err(|e| format!("{k} item {s:?}: {e}"))
        })
        .collect::<Result<Vec<T>, String>>()?;
    Ok(v)
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = arg(&args, "--mode", "check");
    if mode != "check" && mode != "time" {
        return Err(format!("--mode {mode}: expected check or time").into());
    }
    let kinds: Vec<String> = list(&args, "--routing", "one,all,skew")?;
    let tokens: Vec<usize> = list(&args, "--tokens", "512,777,1131,2048,4096")?;
    let seeds: Vec<u64> = list(&args, "--seeds", "7,11")?;
    let iters: usize = arg(&args, "--iters", "30").parse()?;
    let distinct: usize = arg(&args, "--distinct", "64").parse()?;
    if iters == 0 || distinct == 0 || distinct > E || tokens.contains(&0) {
        return Err("--iters and --tokens items must be > 0, --distinct in 1..=512".into());
    }
    for k in &kinds {
        routing(k, 1, 0)?;
    }

    let mut gpu = Gpu::init()?;
    if gpu.arch != "gfx1151" {
        return Err(format!(
            "the F16 WMMA LDS-share entries are gfx1151-only, device is {}",
            gpu.arch
        )
        .into());
    }
    load_kernels(&mut gpu)?;

    let mut all_fails: Vec<String> = Vec::new();
    let mut cases = 0usize;
    for &seed in &seeds {
        let w = Weights::new(&mut gpu, seed, distinct)?;
        for kind in &kinds {
            for &t in &tokens {
                let id = CaseId {
                    kind: kind.clone(),
                    t,
                    seed,
                };
                let (out, fails) = if mode == "check" {
                    check_case(&mut gpu, &w, &id, cases == 0)?
                } else {
                    time_case(&mut gpu, &w, &id, iters)?
                };
                println!("{}", serde_json::to_string(&out)?);
                all_fails.extend(fails);
                cases += 1;
            }
        }
        w.free(&mut gpu)?;
    }
    let pass = all_fails.is_empty();
    println!(
        "{}",
        json!({"result": if pass { "PASS" } else { "FAIL" }, "cases": cases, "fails": all_fails})
    );
    if !pass {
        std::process::exit(1);
    }
    Ok(())
}
