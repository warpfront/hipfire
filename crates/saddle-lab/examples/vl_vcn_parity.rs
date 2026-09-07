// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `vl_vcn_parity`: VCN JPEG → `pixel_values` parity vs the CPU oracles.
//!
//! Experiment `experiment/vcn-jpeg` + `experiment/vcn-ring` only; requires
//! `--features vcn-jpeg` (default-off, no default-build-graph change).
//!
//! Run (device 0, GPU lock held by the caller):
//! ```sh
//! exec 9>/tmp/hipfire-gpu.lock; flock -w 300 9
//! cargo run --release -p saddle-lab --features vcn-jpeg --example vl_vcn_parity -- --backend=redline
//! ```
//!
//! `--backend=va` (default) is the oracle path: va-bridge decode → dma-buf
//! import → `vl_yuv_preprocess` kernels via HIP → D2H. `--backend=redline`
//! runs each fixture through the direct-ring lane (redline `VcnJpegDecoder`
//! submit → ONE compute `CommandBuffer` submitted with
//! `SYNCOBJ_IN=ready_syncobj` + `EMIT_MEM_SYNC` → ONE terminal `wait_fence`
//! before D2H) and then the VA oracle for comparison.
//!
//! Per fixture (benchmarks/vision/images): repo CPU path
//! (`load_and_preprocess_from_bytes` + `extract_patches`, timed) vs pooled
//! VCN path (va-bridge decode → pooled dma-buf import → `vl_yuv_preprocess`
//! kernels → D2H, timed), plus a `libjpeg-turbo-rs`-vs-`image` decode
//! diagnostic. Prints rel-L1 of `pixel_values` and timing tables.
//! Surfaces are linear-only and pooled in the session, so a repeated size
//! costs only decode+sync; VCN-unsupported streams take the turbo fallback.
//!
//! Acceptance (per experiment contract):
//! - per-fixture rel-L1 within ~1e-2, on the VA path (retained) and on the
//!   redline path (the zune-equivalence argument: zune floor ≈ 4.6e-3 on
//!   doge, `docs/VALIDATION.md` VL parity row expects ≈ 5e-3 on 4:2:0);
//! - redline native planes byte-identical to `decode_jpeg_planes`
//!   (diff_count=0/max_abs=0 on every valid row; NV12 = 2 planes,
//!   444P = 3 planes), read back through `ReadyJpeg` after `wait_decode`;
//! - redline-vs-VA patch output f32 `to_bits` equality on the same HSACO
//!   whenever the native planes are byte-identical;
//! - a `[trace]` triple per chained run: JPEG submit → compute submit → ONE
//!   terminal wait, with no fence query between the two submits.

use hipfire_arch_qwen35_vl::image::{
    extract_patches, load_and_preprocess_from_bytes, smart_resize,
};
use redline::device::{Device, GpuBuffer};
use redline::dispatch::{CommandBuffer, Kernel, LoadedModule};
use redline::drm::AMDGPU_IB_FLAG_EMIT_MEM_SYNC;
use redline::queue::{ComputeQueue, Engine, SubmitSync, SyncObj};
use redline::vcn_jpeg::{JpegNativeFormat, JpegSurfaceLayout, VcnJpegDecoder};
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const PATCH: usize = 16;
const TEMPORAL: usize = 2; // vision_config_from_hfq default (qwen35_vl.rs:52-55)
const SMS: usize = 2;
const FACTOR: usize = PATCH * SMS;
const MIN_PX: usize = 65_536;
const MAX_PX: usize = 2_000_000; // VISION_MAX_PIXELS default (image.rs:67)

/// Terminal fence timeout for every redline wait (5 s). A timeout here is an
/// abandon-and-report event per the vcn-ring plan, never a retry.
const FENCE_TIMEOUT_NS: u64 = 5_000_000_000;

fn rel_l1(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        num += (*x as f64 - *y as f64).abs();
        den += (*x as f64).abs();
    }
    num / den.max(1e-30)
}

fn md5hex(data: &[u8]) -> String {
    // Minimal MD5 (RFC 1321) so prompts stay byte-pinned without a new dep.
    hex(&Md5::digest(data))
}

// ——— tiny MD5 + hex (no new deps) ———
struct Md5 {
    state: [u32; 4],
    count: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Md5 {
    fn digest(data: &[u8]) -> [u8; 16] {
        let mut m = Md5 {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            count: 0,
            buf: [0; 64],
            buf_len: 0,
        };
        m.update(data);
        m.finish()
    }
}

#[allow(clippy::many_single_char_names)]
impl Md5 {
    fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let b = self.buf;
                Self::block(&mut self.state, &b);
                self.count += 64;
                self.buf_len = 0;
            }
        }
    }
    fn finish(mut self) -> [u8; 16] {
        let bit_len = (self.count + self.buf_len as u64) * 8;
        self.update(&[0x80]);
        while self.buf_len != 56 {
            self.update(&[0x00]);
        }
        self.update(&bit_len.to_le_bytes());
        let mut out = [0u8; 16];
        for (i, s) in self.state.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&s.to_le_bytes());
        }
        out
    }
    fn block(s: &mut [u32; 4], b: &[u8; 64]) {
        const S: [u32; 64] = [
            7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20,
            5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
            6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
        ];
        const K: [u32; 64] = [
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
            0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
            0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
            0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
            0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
            0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
            0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
            0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
            0xeb86d391,
        ];
        let mut m = [0u32; 16];
        for (i, v) in m.iter_mut().enumerate() {
            *v = u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        }
        let (mut a, mut b_, mut c, mut d) = (s[0], s[1], s[2], s[3]);
        for i in 0..64 {
            let (f, g) = match i {
                0..16 => ((b_ & c) | (!b_ & d), i),
                16..32 => ((d & b_) | (!d & c), (5 * i + 1) % 16),
                32..48 => (b_ ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b_ | !d), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b_;
            b_ = b_.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = tmp;
        }
        s[0] = s[0].wrapping_add(a);
        s[1] = s[1].wrapping_add(b_);
        s[2] = s[2].wrapping_add(c);
        s[3] = s[3].wrapping_add(d);
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

// ——— shared geometry + kernarg prep (both backends) ———

/// Explicit args of `vl_nv12_to_rgb_norm`, in kernel signature order.
/// The HIP path passes these as a pointer vec; the redline path packs the
/// same fields little-endian into a GPU kernarg buffer.
#[derive(Clone)]
struct RgbArgs {
    surf: u64,
    y_pitch: u32,
    uv_pitch: u32,
    u_off: u32,
    v_off: u32,
    step: u32,
    src_w: u32,
    src_h: u32,
    sh: u32,
    sv: u32,
    dst_w: u32,
    dst_h: u32,
    chw: u64,
}

impl RgbArgs {
    /// HIP pointer vec (existing launch convention, order unchanged).
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            (&mut self.surf as *mut u64).cast(),
            (&mut self.y_pitch as *mut u32).cast(),
            (&mut self.uv_pitch as *mut u32).cast(),
            (&mut self.u_off as *mut u32).cast(),
            (&mut self.v_off as *mut u32).cast(),
            (&mut self.step as *mut u32).cast(),
            (&mut self.src_w as *mut u32).cast(),
            (&mut self.src_h as *mut u32).cast(),
            (&mut self.sh as *mut u32).cast(),
            (&mut self.sv as *mut u32).cast(),
            (&mut self.dst_w as *mut u32).cast(),
            (&mut self.dst_h as *mut u32).cast(),
            (&mut self.chw as *mut u64).cast(),
        ]
    }
    /// Redline explicit-arg bytes, same field order, packed per the AMDGPU
    /// kernarg ABI: each arg at its natural alignment. Verified against
    /// `llvm-readelf -n` of the gfx1201 HSACO: surf@0, u32s@8..52, 4-byte
    /// pad@52..56, chw@56 (64 bytes); hidden block counts follow at 64.
    fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(&self.surf.to_le_bytes());
        for v in [
            self.y_pitch,
            self.uv_pitch,
            self.u_off,
            self.v_off,
            self.step,
            self.src_w,
            self.src_h,
            self.sh,
            self.sv,
            self.dst_w,
            self.dst_h,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&[0u8; 4]); // pad so chw lands at offset 56
        b.extend_from_slice(&self.chw.to_le_bytes());
        b
    }
}

/// Explicit args of `vl_extract_patches`, in kernel signature order.
#[derive(Clone)]
struct PatchArgs {
    chw: u64,
    h: u32,
    w: u32,
    p: u32,
    t: u32,
    s: u32,
    out: u64,
}

impl PatchArgs {
    fn ptrs(&mut self) -> Vec<*mut c_void> {
        vec![
            (&mut self.chw as *mut u64).cast(),
            (&mut self.h as *mut u32).cast(),
            (&mut self.w as *mut u32).cast(),
            (&mut self.p as *mut u32).cast(),
            (&mut self.t as *mut u32).cast(),
            (&mut self.s as *mut u32).cast(),
            (&mut self.out as *mut u64).cast(),
        ]
    }
    /// Same ABI rule: chw@0, u32s@8..28, 4-byte pad@28..32, out@32
    /// (40 bytes); hidden block counts follow at 40.
    fn bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(40);
        b.extend_from_slice(&self.chw.to_le_bytes());
        for v in [self.h, self.w, self.p, self.t, self.s] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&[0u8; 4]); // pad so out lands at offset 32
        b.extend_from_slice(&self.out.to_le_bytes());
        b
    }
}

/// Shared target geometry + launch grids, derived from the source surface
/// size and pinned against the repo CPU path's `smart_resize` result.
struct PatchifyPlan {
    t_w: u32,
    t_h: u32,
    n_elem: usize,
    grid_rgb: [u32; 3],
    grid_patch: [u32; 3],
}

impl PatchifyPlan {
    fn compute(sw: usize, sh: usize, img_h: usize, img_w: usize, expect_elem: usize) -> Self {
        let (t_h, t_w) = smart_resize(sh, sw, FACTOR, MIN_PX, MAX_PX);
        assert_eq!(
            (t_h, t_w),
            (img_h, img_w),
            "smart_resize mismatch vs repo path"
        );
        let n_elem = (t_h / PATCH) * (t_w / PATCH) * TEMPORAL * 3 * PATCH * PATCH;
        assert_eq!(n_elem, expect_elem);
        Self {
            t_w: t_w as u32,
            t_h: t_h as u32,
            n_elem,
            grid_rgb: [(t_w as u32 + 15) / 16, (t_h as u32 + 15) / 16, 1],
            grid_patch: [(n_elem as u32 + 255) / 256, 1, 1],
        }
    }
}

/// Pack explicit args + hidden trailing args exactly like
/// `DispatchQueue::dispatch` (code object V5 layout): explicit bytes first,
/// hidden block-count/group-size/grid-dims at the 8-byte-aligned offset.
fn pack_kernargs(kernarg_size: u64, explicit: &[u8], grid: [u32; 3], block: [u32; 3]) -> Vec<u8> {
    let ka_size = std::cmp::max(kernarg_size as usize, explicit.len());
    let mut ka = vec![0u8; std::cmp::max(ka_size, 256)];
    ka[..explicit.len()].copy_from_slice(explicit);
    let hidden_off = (explicit.len() + 7) & !7;
    if ka_size > hidden_off {
        let mut w = |off: usize, val: &[u8]| {
            if off + val.len() <= ka.len() {
                ka[off..off + val.len()].copy_from_slice(val);
            }
        };
        w(hidden_off, &grid[0].to_le_bytes());
        w(hidden_off + 4, &grid[1].to_le_bytes());
        w(hidden_off + 8, &grid[2].to_le_bytes());
        w(hidden_off + 12, &(block[0] as u16).to_le_bytes());
        w(hidden_off + 14, &(block[1] as u16).to_le_bytes());
        w(hidden_off + 16, &(block[2] as u16).to_le_bytes());
        let ndims = if grid[2] > 1 {
            3u16
        } else if grid[1] > 1 {
            2
        } else {
            1
        };
        w(hidden_off + 64, &ndims.to_le_bytes());
    }
    ka
}

/// Surface pitch/offset convention shared by the VA zero-copy arm and the
/// redline native layout: NV12 carries an interleaved CbCr plane
/// (`v_off = u_off + 1`, chroma step 2, half shifts); 444P carries separate
/// Cb/Cr planes (unit step, zero shifts). Returns
/// (uv_pitch, u_off, v_off, chroma_step, (shift_h, shift_v)).
fn surf_kernel_params(
    is_444: bool,
    _y_pitch: u32,
    c_pitch: u32,
    u_off: u32,
    v_off: u32,
) -> (u32, u32, u32, u32, (u32, u32)) {
    if is_444 {
        (c_pitch, u_off, v_off, 1, (0, 0))
    } else {
        (c_pitch, u_off, u_off + 1, 2, (1, 1))
    }
}

/// (min, p50, p95) over devoted timing samples.
fn stats(mut xs: Vec<f64>) -> (f64, f64, f64) {
    assert!(!xs.is_empty());
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        let i = ((p * xs.len() as f64) as usize).min(xs.len() - 1);
        xs[i]
    };
    (xs[0], q(0.5), q(0.95))
}

/// Byte diff of redline native surface valid rows vs the `decode_jpeg_planes`
/// oracle. Only valid rows compare; surface padding is undefined.
fn native_plane_diff(
    surf: &[u8],
    layout: JpegSurfaceLayout,
    oracle: &va_bridge::DerivedPlanes,
) -> (usize, u8) {
    let w = layout.width as usize;
    assert_eq!(w, oracle.width as usize, "layout width vs oracle width");
    assert_eq!(
        layout.height, oracle.height,
        "layout height vs oracle height"
    );
    let is_444 = matches!(layout.format, JpegNativeFormat::Yuv444p);
    let expect_planes = if is_444 { 3 } else { 2 };
    assert_eq!(
        layout.plane_count as usize, expect_planes,
        "layout plane_count vs native format"
    );
    assert_eq!(
        oracle.planes.len(),
        expect_planes,
        "oracle plane count vs native format"
    );
    let row_bytes = |i: usize| -> usize {
        match (is_444, i) {
            (true, _) => w,
            (false, 0) => w,
            (false, _) => ((w + 1) / 2) * 2,
        }
    };
    let (mut diff_n, mut diff_max) = (0usize, 0u8);
    for i in 0..expect_planes {
        let (rw, nr) = (row_bytes(i), layout.plane_rows[i] as usize);
        let want = &oracle.planes[i];
        assert_eq!(
            want.len(),
            rw * nr,
            "oracle plane {i} bytes vs layout rows×stride"
        );
        let base = layout.plane_offsets[i] as usize;
        for r in 0..nr {
            let row =
                &surf[base + r * layout.pitch as usize..base + r * layout.pitch as usize + rw];
            for (a, b) in row.iter().zip(want[r * rw..(r + 1) * rw].iter()) {
                let d = a.abs_diff(*b);
                if d > 0 {
                    diff_n += 1;
                    diff_max = diff_max.max(d);
                }
            }
        }
    }
    (diff_n, diff_max)
}

/// Persistent redline context: one decoder for the whole run (reallocation
/// happens only while Idle on capacity/format change), one module.
struct Redline {
    dev: Device,
    queue: ComputeQueue,
    decoder: VcnJpegDecoder,
    module: LoadedModule,
}

/// Per-fixture redline buffers (allocated once per fixture, reused across
/// that fixture's warm + timed reps so reallocation is excluded from timing).
struct RlBufs {
    d_chw: GpuBuffer,
    d_out: GpuBuffer,
    ka_rgb: GpuBuffer,
    ka_patch: GpuBuffer,
    fence: GpuBuffer,
    ib: GpuBuffer,
}

/// Borrow bundle so the chained rep takes `&mut` decoder alongside the rest.
struct RlRep<'a> {
    dev: &'a Device,
    queue: &'a ComputeQueue,
    decoder: &'a mut VcnJpegDecoder,
    module: &'a LoadedModule,
    bufs: &'a RlBufs,
    plan: &'a PatchifyPlan,
    surf_va: u64,
}

/// One warmed decode->patchify rep: JPEG submit, compute submit_async with
/// SYNCOBJ_IN + EMIT_MEM_SYNC, then the single terminal wait_fence.
/// `fence_value` must differ from the previous rep's value (RELEASE_MEM
/// writes it, WAIT_REG_MEM polls it — a repeated value would pass on the
/// previous rep's stale write).
fn chained_rep(ctx: &mut RlRep<'_>, bytes: &[u8], fence_value: u32) {
    let pending = ctx
        .decoder
        .submit(ctx.dev, ctx.queue, bytes)
        .expect("jpeg submit");
    // The output BO persists while Idle on identical bytes/format; the
    // kernargs baked for the fixture stay valid only if this holds.
    assert_eq!(
        pending.surface().gpu_addr,
        ctx.surf_va,
        "surface VA moved between Idle submits"
    );
    let k_rgb = Kernel::find(ctx.module, "vl_nv12_to_rgb_norm").expect("k_rgb");
    let k_patch = Kernel::find(ctx.module, "vl_extract_patches").expect("k_patch");
    let mut cb = CommandBuffer::new();
    cb.dispatch(
        k_rgb,
        ctx.plan.grid_rgb,
        [16, 16, 1],
        ctx.bufs.ka_rgb.gpu_addr,
    );
    cb.barrier(ctx.bufs.fence.gpu_addr, fence_value);
    cb.dispatch(
        k_patch,
        ctx.plan.grid_patch,
        [256, 1, 1],
        ctx.bufs.ka_patch.gpu_addr,
    );
    // Per-rep IB upload (only the barrier value changes); the IB BO persists.
    ctx.dev
        .upload(&ctx.bufs.ib, &cb.as_bytes())
        .expect("upload ib");
    let wait: &[SyncObj] = &[*pending.ready_syncobj()];
    let sync = SubmitSync {
        wait,
        signal: None,
        ib_flags: AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
    };
    let bos: &[&GpuBuffer] = &[
        &ctx.bufs.ib,
        &ctx.bufs.ka_rgb,
        &ctx.bufs.ka_patch,
        &ctx.module.code_buf,
        pending.surface(),
        &ctx.bufs.d_chw,
        &ctx.bufs.d_out,
        &ctx.bufs.fence,
    ];
    let fence = ctx
        .queue
        .submit_async(
            ctx.dev,
            Engine::COMPUTE,
            &ctx.bufs.ib,
            cb.len_dwords(),
            bos,
            &sync,
        )
        .expect("compute submit_async");
    let done = ctx
        .queue
        .wait_fence(ctx.dev, &fence, FENCE_TIMEOUT_NS)
        .expect("wait_fence");
    assert!(done, "chained rep fence timeout");
    pending
        .complete_after(ctx.dev, ctx.queue, &fence, FENCE_TIMEOUT_NS)
        .expect("complete_after");
}
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cpu_only = args.iter().any(|a| a == "--cpu-only");
    let repeat: usize = args
        .iter()
        .filter_map(|a| a.strip_prefix("--repeat="))
        .filter_map(|v| v.parse().ok())
        .next()
        .unwrap_or(1)
        .max(1);
    let backend = args
        .iter()
        .find_map(|a| a.strip_prefix("--backend="))
        .unwrap_or("va");
    assert!(
        backend == "va" || backend == "redline",
        "--backend must be va|redline"
    );
    let reps: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--reps=").and_then(|v| v.parse().ok()))
        .unwrap_or(100);
    assert!(reps >= 1, "--reps must be >= 1");
    let warm: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--warm=").and_then(|v| v.parse().ok()))
        .unwrap_or(10);
    let only: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let fixtures = [
        "general_qa.jpg",
        "barney_cigar.jpg",
        "scene_1.jpg",
        "scene_2.jpg",
        "doge.jpeg",
    ];
    let img_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/vision/images");

    // ——— HIP + kernels (oracle kernel path; redline loads the same bytes) ———
    let hip = hip_bridge::HipRuntime::load().expect("HipRuntime::load");
    hip.set_device(0).expect("set_device(0)");
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1201".to_string());
    println!("[parity] hip arch={arch} backend={backend} reps={reps} warm={warm}");
    let stream = hip.stream_create().expect("stream_create");

    let ksrc =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../kernels/src/vl_yuv_preprocess.hip");
    let tmp = std::env::temp_dir().join("vcn_parity");
    std::fs::create_dir_all(&tmp).unwrap();
    let hsaco = tmp.join("vl_yuv_preprocess.hsaco");
    println!("[parity] hipcc {} ...", ksrc.display());
    let t_cc = Instant::now();
    let st = Command::new("hipcc")
        .args([
            "--genco",
            "-o",
            hsaco.to_str().unwrap(),
            &format!("--offload-arch={arch}"),
            ksrc.to_str().unwrap(),
        ])
        .status()
        .expect("hipcc spawn");
    assert!(st.success(), "hipcc failed");
    println!("[parity] hipcc {:.2}s", t_cc.elapsed().as_secs_f32());
    let hsaco_bytes = std::fs::read(&hsaco).unwrap();
    println!(
        "[parity] hsaco md5={} ({}B)",
        md5hex(&hsaco_bytes),
        hsaco_bytes.len()
    );
    let module = hip
        .module_load_data(&hsaco_bytes)
        .expect("module_load_data");
    let k_rgb = hip
        .module_get_function(&module, "vl_nv12_to_rgb_norm")
        .expect("vl_nv12_to_rgb_norm");
    let k_patch = hip
        .module_get_function(&module, "vl_extract_patches")
        .expect("vl_extract_patches");

    // ——— redline direct-ring context (redline backend only) ———
    let mut rl: Option<Redline> = if backend == "redline" && !cpu_only {
        let dev = Device::open(None).expect("redline Device::open");
        let queue = ComputeQueue::new(&dev).expect("ComputeQueue::new");
        let decoder = VcnJpegDecoder::new(&dev).expect("VcnJpegDecoder::new");
        let module = dev.load_module(&hsaco_bytes).expect("redline load_module");
        assert!(
            Kernel::find(&module, "vl_nv12_to_rgb_norm").is_some(),
            "redline module missing vl_nv12_to_rgb_norm"
        );
        assert!(
            Kernel::find(&module, "vl_extract_patches").is_some(),
            "redline module missing vl_extract_patches"
        );
        Some(Redline {
            dev,
            queue,
            decoder,
            module,
        })
    } else {
        None
    };

    // ——— VA session (None ⇒ CPU-fallback mode: oracles only) ———
    let mut session = match va_bridge::VaSession::open() {
        Ok(s) => {
            println!("[parity] VA vendor: {} (node {})", s.vendor(), s.node());
            Some(s)
        }
        Err(e) => {
            println!("[parity] VA unavailable ({e}) — oracle-only mode");
            None
        }
    };
    if backend == "redline" && session.is_none() && !cpu_only {
        panic!("--backend=redline needs the VA oracle for plane diffs and to_bits parity");
    }
    println!(
        "{:>16} {:>9} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10} {:>12}",
        "fixture", "dims", "cpu_ms", "vcn_dec", "vcn_kern", "vcn_tot", "speedup", "rel-L1", "path"
    );
    let mut worst_va = 0.0f64;
    let mut worst_rl = 0.0f64;
    // Timing accumulators for the final table (p50 columns + detail triples).
    struct Timing {
        name: String,
        fmt: String,
        dims: String,
        libva: Vec<f64>,
        rl_total: Vec<f64>,
        rl_ring: Vec<f64>,
        rl_chained: Vec<f64>,
        diff_n: usize,
        diff_max: u8,
    }
    let mut timings: Vec<Timing> = Vec::new();
    for name in fixtures {
        if !only.is_empty() && !only.iter().any(|o| o.as_str() == name) {
            continue;
        }
        let bytes = std::fs::read(img_dir.join(name)).expect("fixture read");
        println!(
            "[parity] --- {name} md5={} ({}B)",
            md5hex(&bytes),
            bytes.len()
        );

        // Oracle A (acceptance): the repo CPU path.
        let t0 = Instant::now();
        let (pixels, img_h, img_w) =
            load_and_preprocess_from_bytes(&bytes, PATCH, SMS).expect("cpu preprocess");
        let cpu_pre_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let cpu_patches = extract_patches(&pixels, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
        let cpu_patch_ms = t0.elapsed().as_secs_f64() * 1e3;
        let cpu_ms = cpu_pre_ms + cpu_patch_ms;

        // Oracle B (diagnostic): turbo vs image-crate decode, native resolution.
        let turbo = libjpeg_turbo_rs::decompress_to(&bytes, libjpeg_turbo_rs::PixelFormat::Rgb)
            .expect("turbo decode");
        let via_image = image::load_from_memory(&bytes)
            .expect("image decode")
            .to_rgb8();
        let (tw, th) = (turbo.width, turbo.height);
        assert_eq!(
            (tw, th),
            (via_image.width() as usize, via_image.height() as usize)
        );
        let raw = via_image.as_raw();
        let mut diff_n = 0usize;
        let mut diff_max = 0u8;
        for (a, b) in turbo.data.iter().zip(raw.iter()) {
            let d = a.abs_diff(*b);
            if d > 0 {
                diff_n += 1;
                diff_max = diff_max.max(d);
            }
        }
        println!(
            "[parity] decode-xcheck turbo({tw}x{th}) vs image-crate: differ={diff_n}/{} max_abs={diff_max}",
            raw.len()
        );
        // VCN path (driver-resolved pixels) with turbo CPU fallback, or a
        // --cpu-only timing row for the kill-gate A/B.
        if cpu_only {
            println!(
                "{:>16} {:>9} {:>10.2} {:>10} {:>10} {:>10} {:>8} {:>10} {:>12}",
                name,
                format!("{img_w}x{img_h}"),
                cpu_ms,
                "-",
                "-",
                "-",
                "-",
                "-",
                "cpu-only"
            );
            println!(
                "[parity] {name}: cpu-only cpu_pre={cpu_pre_ms:.2}ms cpu_patch={cpu_patch_ms:.2}ms"
            );
            continue;
        }
        let Some(sess) = session.as_mut() else {
            println!("[parity] {name}: VA unavailable, skipping VCN path");
            continue;
        };
        // Shared kernel stage: source surface (device) -> CHW f32 -> patches.
        // Returns (patches, kernel ms incl. sync; excludes the parity D2H).
        let run_kernels = |surf_a: u64,
                           y_pitch: u32,
                           uv_pitch: u32,
                           u_offset: u32,
                           v_offset: u32,
                           chroma_step: u32,
                           shifts: (u32, u32),
                           sw: usize,
                           sh: usize|
         -> (Vec<f32>, f64) {
            let plan = PatchifyPlan::compute(sw, sh, img_h, img_w, cpu_patches.len());
            let n_chw = 3 * img_h * img_w;
            let d_chw = hip.malloc(n_chw * 4).expect("malloc chw");
            let d_out = hip.malloc(plan.n_elem * 4).expect("malloc patches");
            let mut rgb = RgbArgs {
                surf: surf_a,
                y_pitch,
                uv_pitch,
                u_off: u_offset,
                v_off: v_offset,
                step: chroma_step,
                src_w: sw as u32,
                src_h: sh as u32,
                sh: shifts.0,
                sv: shifts.1,
                dst_w: plan.t_w,
                dst_h: plan.t_h,
                chw: d_chw.as_ptr() as u64,
            };
            let t1 = Instant::now();
            let mut p1 = rgb.ptrs();
            // SAFETY: module loaded, args are valid device pointers / values.
            unsafe {
                hip.launch_kernel(
                    &k_rgb,
                    [plan.grid_rgb[0], plan.grid_rgb[1], plan.grid_rgb[2]],
                    [16, 16, 1],
                    0,
                    Some(&stream),
                    &mut p1,
                )
                .expect("launch rgb");
            }
            let mut patch = PatchArgs {
                chw: rgb.chw,
                h: plan.t_h,
                w: plan.t_w,
                p: PATCH as u32,
                t: TEMPORAL as u32,
                s: SMS as u32,
                out: d_out.as_ptr() as u64,
            };
            let mut p2 = patch.ptrs();
            // SAFETY: same.
            unsafe {
                hip.launch_kernel(
                    &k_patch,
                    [plan.grid_patch[0], 1, 1],
                    [256, 1, 1],
                    0,
                    Some(&stream),
                    &mut p2,
                )
                .expect("launch patches");
            }
            hip.stream_synchronize(&stream).expect("sync");
            let kern_ms = t1.elapsed().as_secs_f64() * 1e3;
            let mut raw = vec![0u8; plan.n_elem * 4];
            hip.memcpy_dtoh(&mut raw, &d_out).expect("dtoh");
            // SAFETY: kernel wrote f32 elements; length checked above.
            let v: Vec<f32> =
                unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, plan.n_elem) }
                    .to_vec();
            hip.free(d_chw).unwrap();
            hip.free(d_out).unwrap();
            (v, kern_ms)
        };

        // Pooled zero-copy arm (linear-only surfaces): kernels read the
        // imported surface directly. The first decode of a size also pays
        // surface/context/export/import; repeats cost only parse+submit+sync.
        // `--repeat=N` times each decode to expose first vs steady-state.
        let mut dec_iters = Vec::with_capacity(repeat);
        let mut outcome = None;
        for _ in 0..repeat {
            let t0 = Instant::now();
            let o = sess.decode_jpeg(&bytes);
            dec_iters.push(t0.elapsed().as_secs_f64() * 1e3);
            if o.is_err() {
                outcome = Some(o);
                break;
            }
            outcome = Some(o);
        }
        let dec_ms = *dec_iters.last().unwrap_or(&f64::NAN);
        if repeat > 1 {
            println!(
                "[parity] {name}: dec iters (ms) = [{}]",
                dec_iters
                    .iter()
                    .map(|v| format!("{v:.2}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let outcome = outcome.expect("at least one decode attempt");
        let (path, got, kern_ms): (&str, Vec<f32>, f64) = match outcome {
            Ok(va_bridge::DecodeOutcome::Decoded(vf))
                if vf.fourcc == 0x3231_564E || vf.fourcc == 0x5034_3434 =>
            {
                let planar = vf.fourcc == 0x5034_3434;
                let (uv_pitch, u_off, v_off, step, shifts) = if planar {
                    assert!(vf.num_layers >= 3, "444P export must carry 3 layers");
                    assert_eq!(vf.layers[1].pitch[0], vf.layers[2].pitch[0]);
                    surf_kernel_params(
                        true,
                        vf.y_pitch(),
                        vf.layers[1].pitch[0],
                        vf.layers[1].offset[0],
                        vf.layers[2].offset[0],
                    )
                } else {
                    surf_kernel_params(false, vf.y_pitch(), vf.uv_pitch(), vf.uv_offset(), 0)
                };
                let (v, k) = run_kernels(
                    vf.device_ptr() as u64,
                    vf.y_pitch(),
                    uv_pitch,
                    u_off,
                    v_off,
                    step,
                    shifts,
                    vf.width as usize,
                    vf.height as usize,
                );
                (if planar { "vcn-zc-444" } else { "vcn-zc" }, v, k)
            }
            Ok(va_bridge::DecodeOutcome::Decoded(vf)) => {
                println!(
                    "[parity] {name}: VCN decoded fourcc=0x{:08x} — no kernel arm, CPU path",
                    vf.fourcc
                );
                let t1 = Instant::now();
                let rgb = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(tw as u32, th as u32, turbo.data.clone())
                        .expect("turbo rgb"),
                );
                let rgb = rgb
                    .resize_exact(
                        img_w as u32,
                        img_h as u32,
                        image::imageops::FilterType::CatmullRom,
                    )
                    .to_rgb8();
                let plane = img_h * img_w;
                let mut chw = vec![0f32; 3 * plane];
                for (i, px) in rgb.pixels().enumerate() {
                    chw[i] = px[0] as f32 / 127.5 - 1.0;
                    chw[plane + i] = px[1] as f32 / 127.5 - 1.0;
                    chw[2 * plane + i] = px[2] as f32 / 127.5 - 1.0;
                }
                let fb = extract_patches(&chw, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
                ("cpu-fallback", fb, t1.elapsed().as_secs_f64() * 1e3)
            }
            Ok(va_bridge::DecodeOutcome::Unsupported(reason)) => {
                // Product behavior: VCN-unsupported streams fall back to turbo.
                println!("[parity] {name}: VCN unsupported ({reason}) — turbo CPU fallback");
                let t1 = Instant::now();
                let rgb = image::DynamicImage::ImageRgb8(
                    image::RgbImage::from_raw(tw as u32, th as u32, turbo.data.clone())
                        .expect("turbo rgb"),
                );
                let rgb = rgb
                    .resize_exact(
                        img_w as u32,
                        img_h as u32,
                        image::imageops::FilterType::CatmullRom,
                    )
                    .to_rgb8();
                let plane = img_h * img_w;
                let mut chw = vec![0f32; 3 * plane];
                for (i, px) in rgb.pixels().enumerate() {
                    chw[i] = px[0] as f32 / 127.5 - 1.0;
                    chw[plane + i] = px[1] as f32 / 127.5 - 1.0;
                    chw[2 * plane + i] = px[2] as f32 / 127.5 - 1.0;
                }
                let fb = extract_patches(&chw, 3, img_h, img_w, PATCH, TEMPORAL, SMS);
                ("cpu-fallback", fb, t1.elapsed().as_secs_f64() * 1e3)
            }
            Err(e) => panic!("vcn decode of {name}: {e}"),
        };
        let r = rel_l1(&cpu_patches, &got);
        worst_va = worst_va.max(r);
        let tot = dec_ms + kern_ms;
        println!(
            "{:>16} {:>9} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>8.2}x {:>10.3e} {:>12}",
            name,
            format!("{img_w}x{img_h}"),
            cpu_ms,
            dec_ms,
            kern_ms,
            tot,
            cpu_ms / tot.max(1e-9),
            r,
            path
        );

        // ——— redline direct-ring lane (oracle planes + VA kernel output above
        // feed every comparison below) ———
        if let Some(ctx) = rl.as_mut() {
            // Format-generic oracle planes for the native-plane gate.
            let oracle = sess.decode_jpeg_planes(&bytes).expect("decode_jpeg_planes");
            let is_444 = oracle.fourcc == 0x5034_3434;
            assert!(
                oracle.fourcc == 0x3231_564E || is_444,
                "oracle fourcc=0x{:08x}, expected NV12 or 444P",
                oracle.fourcc
            );
            let fmt = if is_444 { "444P" } else { "NV12" };

            // (a) Decode-only parity run: submit -> wait_decode -> ReadyJpeg
            // readback (valid rows vs oracle) -> Drop returns Idle.
            eprintln!("[trace] {name}: jpeg submit (VCN_JPEG ring, no host wait)");
            let pending = ctx
                .decoder
                .submit(&ctx.dev, &ctx.queue, &bytes)
                .expect("jpeg submit");
            let layout = pending.layout();
            assert!(
                matches!(
                    (is_444, layout.format),
                    (true, JpegNativeFormat::Yuv444p) | (false, JpegNativeFormat::Nv12)
                ),
                "redline native format vs oracle format"
            );
            let plan = PatchifyPlan::compute(
                layout.width as usize,
                layout.height as usize,
                img_h,
                img_w,
                cpu_patches.len(),
            );
            let ready = pending
                .wait_decode(&ctx.dev, &ctx.queue, FENCE_TIMEOUT_NS)
                .expect("wait_decode");
            let surf = ready.surface();
            let surf_va = surf.gpu_addr;
            let mut surf_bytes = vec![0u8; surf.size as usize];
            ctx.dev
                .download(surf, &mut surf_bytes)
                .expect("surface readback");
            let (plane_diff_n, plane_diff_max) = native_plane_diff(&surf_bytes, layout, &oracle);
            println!(
                "[parity] {name}: native-plane diff_count={plane_diff_n} max_abs={plane_diff_max}"
            );
            assert_eq!(plane_diff_n, 0, "{name}: native-plane byte mismatch");
            assert_eq!(plane_diff_max, 0, "{name}: native-plane byte mismatch");
            drop(ready); // -> Idle; output no longer GPU-owned.

            // (b) Chained run: JPEG submit -> compute submit -> ONE terminal
            // wait. No fence query between the two submits. The surface VA is
            // stable: no realloc happens while Idle on identical bytes/format.
            // Kernel surface bases are relative to the LUMA plane base, not
            // the BO base: the decoder inserts canary guard regions before
            // each plane (e.g. luma at +256). Chroma offsets go relative to
            // luma so `surf + u_off` lands on the absolute chroma address.
            debug_assert!(layout.plane_offsets[1] >= layout.plane_offsets[0]);
            debug_assert!(layout.plane_offsets[2] >= layout.plane_offsets[0]);
            let luma_base = surf_va + layout.plane_offsets[0] as u64;
            let (uv_pitch, u_off, v_off, step, shifts) = if is_444 {
                surf_kernel_params(
                    true,
                    layout.pitch,
                    layout.pitch,
                    layout.plane_offsets[1] - layout.plane_offsets[0],
                    layout.plane_offsets[2] - layout.plane_offsets[0],
                )
            } else {
                surf_kernel_params(
                    false,
                    layout.pitch,
                    layout.pitch,
                    layout.plane_offsets[1] - layout.plane_offsets[0],
                    0,
                )
            };
            let n_chw = 3 * img_h * img_w;
            let d_chw = ctx.dev.alloc_vram((n_chw * 4) as u64).expect("alloc chw");
            let d_out = ctx
                .dev
                .alloc_vram((plan.n_elem * 4) as u64)
                .expect("alloc out");
            let rgb_args = RgbArgs {
                surf: luma_base,
                y_pitch: layout.pitch,
                uv_pitch,
                u_off,
                v_off,
                step,
                src_w: layout.width,
                src_h: layout.height,
                sh: shifts.0,
                sv: shifts.1,
                dst_w: plan.t_w,
                dst_h: plan.t_h,
                chw: d_chw.gpu_addr,
            };
            let patch_args = PatchArgs {
                chw: d_chw.gpu_addr,
                h: plan.t_h,
                w: plan.t_w,
                p: PATCH as u32,
                t: TEMPORAL as u32,
                s: SMS as u32,
                out: d_out.gpu_addr,
            };
            let k_rl_rgb = Kernel::find(&ctx.module, "vl_nv12_to_rgb_norm").expect("k_rgb");
            let k_rl_patch = Kernel::find(&ctx.module, "vl_extract_patches").expect("k_patch");
            let ka_rgb = pack_kernargs(
                k_rl_rgb.kernarg_size,
                &rgb_args.bytes(),
                plan.grid_rgb,
                [16, 16, 1],
            );
            let ka_patch = pack_kernargs(
                k_rl_patch.kernarg_size,
                &patch_args.bytes(),
                plan.grid_patch,
                [256, 1, 1],
            );
            let ka_rgb_buf = ctx
                .dev
                .alloc_vram(ka_rgb.len() as u64)
                .expect("alloc ka_rgb");
            let ka_patch_buf = ctx
                .dev
                .alloc_vram(ka_patch.len() as u64)
                .expect("alloc ka_patch");
            ctx.dev.upload(&ka_rgb_buf, &ka_rgb).expect("upload ka_rgb");
            ctx.dev
                .upload(&ka_patch_buf, &ka_patch)
                .expect("upload ka_patch");
            let fence_buf = ctx.dev.alloc_vram(4096).expect("alloc fence");
            ctx.dev
                .upload(&fence_buf, &vec![0u8; 4096])
                .expect("zero fence");
            // ONE command buffer: rgb norm, intra-IB barrier, patch extract.
            let mut cb = CommandBuffer::new();
            cb.dispatch(k_rl_rgb, plan.grid_rgb, [16, 16, 1], ka_rgb_buf.gpu_addr);
            cb.barrier(fence_buf.gpu_addr, 1);
            cb.dispatch(
                k_rl_patch,
                plan.grid_patch,
                [256, 1, 1],
                ka_patch_buf.gpu_addr,
            );
            let ib_buf = ctx
                .dev
                .alloc_vram(cb.as_bytes().len() as u64)
                .expect("alloc ib");
            ctx.dev.upload(&ib_buf, &cb.as_bytes()).expect("upload ib");
            let bufs = RlBufs {
                d_chw,
                d_out,
                ka_rgb: ka_rgb_buf,
                ka_patch: ka_patch_buf,
                fence: fence_buf,
                ib: ib_buf,
            };

            eprintln!("[trace] {name}: jpeg submit (VCN_JPEG ring, no host wait)");
            let t_submit = Instant::now();
            let pending = ctx
                .decoder
                .submit(&ctx.dev, &ctx.queue, &bytes)
                .expect("jpeg submit");
            assert_eq!(
                pending.surface().gpu_addr,
                surf_va,
                "surface VA moved between Idle submits"
            );
            let wait: &[SyncObj] = &[*pending.ready_syncobj()];
            let sync = SubmitSync {
                wait,
                signal: None,
                ib_flags: AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
            };
            eprintln!(
                "[trace] {name}: compute submit_async SYNCOBJ_IN=ready_syncobj + EMIT_MEM_SYNC"
            );
            let bos: &[&GpuBuffer] = &[
                &bufs.ib,
                &bufs.ka_rgb,
                &bufs.ka_patch,
                &ctx.module.code_buf,
                pending.surface(),
                &bufs.d_chw,
                &bufs.d_out,
                &bufs.fence,
            ];
            let fence = ctx
                .queue
                .submit_async(
                    &ctx.dev,
                    Engine::COMPUTE,
                    &bufs.ib,
                    cb.len_dwords(),
                    bos,
                    &sync,
                )
                .expect("compute submit_async");
            let submit_ms = t_submit.elapsed().as_secs_f64() * 1e3;
            eprintln!("[trace] {name}: terminal wait_fence (sole wait before D2H)");
            let t_wait = Instant::now();
            let done = ctx
                .queue
                .wait_fence(&ctx.dev, &fence, FENCE_TIMEOUT_NS)
                .expect("wait_fence");
            assert!(done, "redline compute fence timeout on {name}");
            let wait_ms = t_wait.elapsed().as_secs_f64() * 1e3;
            let mut out_bytes = vec![0u8; plan.n_elem * 4];
            ctx.dev
                .download(&bufs.d_out, &mut out_bytes)
                .expect("d2h patches");
            // SAFETY: kernel wrote f32 elements; length checked by the plan.
            let rl_patches: Vec<f32> = unsafe {
                std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, plan.n_elem)
            }
            .to_vec();
            pending
                .complete_after(&ctx.dev, &ctx.queue, &fence, FENCE_TIMEOUT_NS)
                .expect("complete_after");

            // Same-HSACO to_bits parity (inputs proven byte-identical above).
            let mut bit_diff = 0usize;
            for (a, b) in rl_patches.iter().zip(got.iter()) {
                if a.to_bits() != b.to_bits() {
                    bit_diff += 1;
                }
            }
            // to_bits equality is meaningful only when the VA side ran the
            // same HSACO on VCN-decoded pixels. On the turbo CPU fallback
            // (no VA kernel arm for the format) the inputs differ by decode
            // path, so only the rel-L1 gate below applies.
            let va_ran_kernels = path != "cpu-fallback";
            println!(
                "[parity] {name}: redline-vs-VA to_bits differ={bit_diff}/{}",
                rl_patches.len()
            );
            if va_ran_kernels {
                assert_eq!(bit_diff, 0, "{name}: redline-vs-VA bit mismatch");
            } else {
                println!("[parity] {name}: VA took the CPU fallback; skipping to_bits gate");
            }

            let r_rl = rel_l1(&cpu_patches, &rl_patches);
            worst_rl = worst_rl.max(r_rl);
            let chained_ms = submit_ms + wait_ms;
            println!(
                "{:>16} {:>9} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>8.2}x {:>10.3e} {:>12}",
                name,
                format!("{img_w}x{img_h}"),
                cpu_ms,
                submit_ms,
                wait_ms,
                chained_ms,
                cpu_ms / chained_ms.max(1e-9),
                r_rl,
                "redline"
            );

            // ——— warmed timing loops (persistent decoder/BOs; reallocation
            // happened above and is excluded) ———
            // libva end-to-ready.
            for _ in 0..warm {
                let _ = sess.decode_jpeg_planes(&bytes).expect("warm va");
            }
            let mut libva = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = sess.decode_jpeg_planes(&bytes).expect("timed va");
                libva.push(t.elapsed().as_secs_f64() * 1e3);
            }
            // Redline total-decode (submit + terminal wait_decode) and its
            // ring-only portion (wait_decode with everything resident).
            for _ in 0..warm {
                let p = ctx
                    .decoder
                    .submit(&ctx.dev, &ctx.queue, &bytes)
                    .expect("warm jpeg");
                let ready = p
                    .wait_decode(&ctx.dev, &ctx.queue, FENCE_TIMEOUT_NS)
                    .expect("warm wait");
                drop(ready);
            }
            let (mut rl_total, mut rl_ring) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
            for _ in 0..reps {
                let t = Instant::now();
                let p = ctx
                    .decoder
                    .submit(&ctx.dev, &ctx.queue, &bytes)
                    .expect("jpeg submit");
                let t_submit_side = t.elapsed().as_secs_f64() * 1e3;
                let t2 = Instant::now();
                let ready = p
                    .wait_decode(&ctx.dev, &ctx.queue, FENCE_TIMEOUT_NS)
                    .expect("wait_decode");
                let t_ring = t2.elapsed().as_secs_f64() * 1e3;
                drop(ready);
                rl_total.push(t_submit_side + t_ring);
                rl_ring.push(t_ring);
            }
            // Redline decode->two-kernel patchify with one terminal wait.
            // Only the intra-IB barrier value alternates per rep; the IB BO,
            // kernargs and data buffers persist.
            let mut rep_ctx = RlRep {
                dev: &ctx.dev,
                queue: &ctx.queue,
                decoder: &mut ctx.decoder,
                module: &ctx.module,
                bufs: &bufs,
                plan: &plan,
                surf_va,
            };
            // Correctness run left fence value 1 in the fence word; rep values
            // start at 2 so the first WAIT cannot pass on a stale write.
            for rep in 0..warm {
                chained_rep(&mut rep_ctx, &bytes, ((rep + 1) % 2) as u32 + 1);
            }
            let mut rl_chained = Vec::with_capacity(reps);
            for rep in 0..reps {
                let t = Instant::now();
                chained_rep(&mut rep_ctx, &bytes, ((rep + warm + 1) % 2) as u32 + 1);
                rl_chained.push(t.elapsed().as_secs_f64() * 1e3);
            }
            drop(rep_ctx);
            // Repeated-frame stability: the last timed rep's output must be
            // bit-identical to the correctness run (deterministic kernels +
            // EMIT_MEM_SYNC visibility across 110 resubmits).
            let mut rb = vec![0u8; plan.n_elem * 4];
            ctx.dev.download(&bufs.d_out, &mut rb).expect("d2h repeat");
            // SAFETY: kernel wrote f32 elements; length checked by the plan.
            let repeat: Vec<f32> =
                unsafe { std::slice::from_raw_parts(rb.as_ptr() as *const f32, plan.n_elem) }
                    .to_vec();
            let mut repeat_diff = 0usize;
            for (a, b) in repeat.iter().zip(rl_patches.iter()) {
                if a.to_bits() != b.to_bits() {
                    repeat_diff += 1;
                }
            }
            println!(
                "[parity] {name}: repeated-frame to_bits differ={repeat_diff}/{}",
                repeat.len()
            );
            assert_eq!(repeat_diff, 0, "{name}: repeated-frame instability");
            let (l_min, l_p50, l_p95) = stats(libva.clone());
            let (t_min, t_p50, t_p95) = stats(rl_total.clone());
            let (g_min, g_p50, g_p95) = stats(rl_ring.clone());
            let (c_min, c_p50, c_p95) = stats(rl_chained.clone());
            println!(
                "[parity] {name}: libva-ready min/p50/p95 = {l_min:.3}/{l_p50:.3}/{l_p95:.3}ms; \
                 rl-total {t_min:.3}/{t_p50:.3}/{t_p95:.3}ms; rl-ring {g_min:.3}/{g_p50:.3}/{g_p95:.3}ms; \
                 rl-chained {c_min:.3}/{c_p50:.3}/{c_p95:.3}ms"
            );
            timings.push(Timing {
                name: name.to_string(),
                fmt: fmt.to_string(),
                dims: format!("{img_w}x{img_h}"),
                libva,
                rl_total,
                rl_ring,
                rl_chained,
                diff_n: plane_diff_n,
                diff_max: plane_diff_max,
            });

            let RlBufs {
                d_chw,
                d_out,
                ka_rgb,
                ka_patch,
                fence,
                ib,
            } = bufs;
            for b in [d_chw, d_out, ka_rgb, ka_patch, fence, ib] {
                ctx.dev.free_buffer(b).expect("free rl buf");
            }
        }
    }
    println!("[parity] worst VA rel-L1 = {worst_va:.3e} (equivalence bound 1e-2)");
    assert!(worst_va <= 1e-2, "VA oracle parity bound violated");
    if backend == "redline" {
        println!("[parity] worst redline rel-L1 = {worst_rl:.3e} (equivalence bound 1e-2)");
        assert!(worst_rl <= 1e-2, "redline parity bound violated");
        println!(
            "{:>16} {:>6} {:>9} {:>12} {:>12} {:>12} {:>12} {:>10} {:>7} {:>14}",
            "fixture",
            "format",
            "dims",
            "libva p50",
            "rl-total p50",
            "rl-ring p50",
            "rl-chain p50",
            "delta ms",
            "ratio",
            "diff/max_abs"
        );
        for t in &timings {
            let (_, l_p50, _) = stats(t.libva.clone());
            let (_, t_p50, _) = stats(t.rl_total.clone());
            let (_, g_p50, _) = stats(t.rl_ring.clone());
            let (_, c_p50, _) = stats(t.rl_chained.clone());
            println!(
                "{:>16} {:>6} {:>9} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>10.3} {:>7.2}x {:>6}/{:<7}",
                t.name,
                t.fmt,
                t.dims,
                l_p50,
                t_p50,
                g_p50,
                c_p50,
                l_p50 - c_p50,
                l_p50 / c_p50.max(1e-9),
                t.diff_n,
                t.diff_max
            );
        }
    }
    // ——— redline teardown: explicit-destroy ownership, exactly once ———
    if let Some(ctx) = rl {
        let Redline {
            dev,
            queue,
            decoder,
            module,
        } = ctx;
        decoder.destroy(&dev);
        queue.destroy(&dev);
        let LoadedModule {
            kernels: _,
            code_buf,
        } = module;
        dev.free_buffer(code_buf).expect("free code");
    }
}
