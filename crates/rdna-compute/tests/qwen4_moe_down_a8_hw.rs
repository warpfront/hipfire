// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! E1 (KERNEL-PLAN §1.1 / §1.3 / §5): qwen4 Flash-Next routed-expert DOWN (qt53, M = 2560, K = 640,
//! top-10) with symmetric IU4 expert codes x block_i8_128 (A8) activations on exact gfx1201.
//!
//! Candidate source: `kernels/src/qwen4_moe_down_a8.gfx1201.hip` (JIT with the `block_i8_128`
//! prelude): the producer `qwen4_moe_down_a8_quant_gfx1201`, the D-indexed decode entries
//! (`..._indexed[_planar]_gfx1201`, rows 1..8) and the D-grouped verify entries
//! (`..._grouped[_planar]_gfx1201`, rows 9..64), shipped MQ4G128V2 and F1 planar MQ4G128V2P.
//!
//! `qwen4_moe_down_a8_oracle_hw`: an independent scalar Rust reference (bit-level nibble decode of the
//! UPLOADED layout, natural K order recovered from the documented nibble-pair qs order, exact i32 C per
//! 128 block, `sd = RN(f32(s_h) * d)`, `acc = fma(sd, C as f32, acc)` in increasing block order from +0.0)
//! is compared byte-for-byte against the GPU on
//!   * the producer's `block_i8_128` bytes (vs the CPU quantizer, and vs the shared reference producer
//!     `quantize_int8_mmq_ds128` of block_i8_128_quant.hip),
//!   * the per-block i32 C (test-only `-DA8_DEBUG_C` build of the same kernel source),
//!   * the final F32 words and guard words around Y (every grouped slot incl. pad / sentinel stores),
//! on (a) synthetic headers/activations incl. edge scales, zero groups, saturating activations, invalid
//! expert ids / sentinel tiles, and (b) REAL tensors of the shipped model (layers 0/17/47, only the
//! touched experts are read through pread) with REAL W1 routes at rows 1,2,4,8 (indexed) and
//! 9,16,24,32,48,56,64 (grouped); singleton (rows = 1) bytes == batched bytes for the same token; negative
//! controls (wrong scale byte, wrong block order) must be detected. The relative L2 error of the A8
//! output against the incumbent F32 kernel (same weights / same X) is reported (not a gate).
//!
//! `qwen4_moe_down_a8_timing_hw`: cold-weight (>= 6 real layers rotated + MALL flush) event timing of
//! A8 producer / down / producer+down against the incumbent production kernels on the same real routes.
//!
//! `#[ignore]`d: needs an exact gfx1201 device and the shipped model.
//!   cargo test -p rdna-compute --release --features lab --test qwen4_moe_down_a8_hw -- --ignored --nocapture --test-threads=1 <name>
//! Env: E1DN_OUT (summary dir), E1DN_MODEL, E1DN_ROUTES (dir), E1DN_DEFS ("A8_GR_NW=4,A8_GR_RG=2" A/B sweeps
//! prepended to the candidate source), E1DN_LAYERS (oracle layers), E1DN_ROWS (timing rows filter).

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::fmt::Write as _;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

const QUANT_SRC: &str = include_str!("../../../kernels/src/block_i8_128_quant.hip");
const A8_SRC: &str = include_str!("../../../kernels/src/qwen4_moe_down_a8.gfx1201.hip");
const INC_IDX_SHIPPED: &str =
    include_str!("../../../kernels/src/gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded.hip");
const INC_IDX_PLANAR: &str = include_str!(
    "../../../kernels/src/gemv_mq4g128v2p_moe_down_top10_indexed_batched_expanded.hip"
);
const INC_GRP_SHIPPED: &str =
    include_str!("../../../kernels/src/gemm_mq4g128v2_moe_grouped_top10_o8_r16.gfx1151.hip");
const INC_GRP_PLANAR: &str =
    include_str!("../../../kernels/src/gemm_mq4g128v2p_moe_grouped_top10_o8_r16.gfx1201.hip");

// Best-exact per-pick reference (the MoeDn32 `dni` kernel): first a lab-only source loaded by absolute
// path, now shipped as the default gfx1201 <= 8-row down (`HIPFIRE_QWEN4_MOE_DOWN_DNI`, 7a1f00cb4), so the
// timing arm reads the in-tree source and states "A8 vs the best exact decode" next to the incumbent.
const DNI_SRC: &str =
    include_str!("../../../kernels/src/gemv_mq4g128v2_moe_down_top10_dni.gfx1201.hip");
const MOD_DNI: &str = "e1dn_dni";
const F_DNI_S: &str = "gemv_mq4g128v2_moe_down_top10_dni_k640";
const F_DNI_P: &str = "gemv_mq4g128v2p_moe_down_top10_dni_k640";
const MOD_A8: &str = "e1dn_a8";
const MOD_A8_DBG: &str = "e1dn_a8_dbg";
const MOD_REFQ: &str = "e1dn_refq";
const MOD_INC_IDX_S: &str = "e1dn_inc_idx_s";
const MOD_INC_IDX_P: &str = "e1dn_inc_idx_p";
const MOD_INC_GRP_S: &str = "e1dn_inc_grp_s";
const MOD_INC_GRP_P: &str = "e1dn_inc_grp_p";
const MOD_FLUSH: &str = "e1dn_flush";

const F_QUANT: &str = "qwen4_moe_down_a8_quant_gfx1201";
const F_REFQ: &str = "quantize_int8_mmq_ds128";
const F_INC_IDX_S: &str = "gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded_k640_r4";
const F_INC_IDX_P: &str = "gemv_mq4g128v2p_moe_down_top10_indexed_batched_expanded_k640_r4";
const F_INC_GRP_S: &str = "gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151";
const F_INC_GRP_P: &str = "gemm_mq4g128v2p_moe_grouped_top10_o8_r16_gfx1201";
const F_FLUSH: &str = "e1dn_flush_stream";

const FLUSH_SRC: &str = r#"
#include <hip/hip_runtime.h>
extern "C" __global__ void e1dn_flush_stream(float4* p, long long n16) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    // READ-only sweep: evicts the MALL / L2 weights without leaving dirty lines whose write-back would
    // contend with the timed kernel's HBM reads (a write sweep measured the roofline at ~290 GB/s).
    float acc = 0.0f;
    for (; i < n16; i += stride) {
        const float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234567.0f) p[0] = make_float4(acc, acc, acc, acc);
}
// Roofline reference: stream every DISTINCT live expert (first tile of each expert run) once with b128 loads.
extern "C" __global__ void e1dn_stream_experts(const unsigned long long* table, const int* tiles, long long n16, float* sink, int n_exp) {
    const int t = blockIdx.y;
    const int e = tiles[t];
    if (e < 0 || e >= n_exp) return;
    if (t > 0 && tiles[t - 1] == e) return;
    const float4* p = (const float4*)table[e];
    float acc = 0.0f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (long long)gridDim.x * blockDim.x) {
        const float4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234567.0f) sink[0] = acc;
}
"#;
const F_STREAM: &str = "e1dn_stream_experts";

const M: usize = 2560;
const K: usize = 640;
const G: usize = 5;
const KTOP: usize = 10;
const TILE: usize = 16;
const SHIP: usize = M * G * 68; // 870_400
const PLAN: usize = M * K / 2 + M * G * 2; // 844_800
const BLK: usize = 136;
const ROWQ: usize = G * BLK; // 680
const N_LOCAL: usize = 256; // EP2 rank 0 owns the even global experts
const N_GLOBAL: usize = 512;
const GUARD: usize = 256; // guard bytes on each side of an output span
const SENT: u8 = 0xA5;

const DEFAULT_MODEL: &str =
    "/home/kaden/qcal/release-0.4.3/mgfn/model/qwen3.8-flash-next-base.mq4-xts";
const DEFAULT_ROUTES: &str = "/home/kaden/qcal/release-0.4.5/parity/kernel-runs/k0/routes";

// ---------------------------------------------------------------------------------------------
// sources / loading
// ---------------------------------------------------------------------------------------------

fn defs_text() -> String {
    let mut s = String::new();
    if let Ok(v) = std::env::var("E1DN_DEFS") {
        for kv in v.split(',').filter(|s| !s.is_empty()) {
            let (k, val) = kv.split_once('=').unwrap_or((kv, "1"));
            writeln!(s, "#define {k} {val}").ok();
        }
    }
    s
}

fn a8_src(debug: bool) -> String {
    let mut s = defs_text();
    if debug {
        s.push_str("#define A8_DEBUG_C 1\n#define A8_SUF _dbgc\n");
    }
    s.push_str("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n");
    s.push_str(QUANT_SRC);
    s.push_str(A8_SRC);
    s
}

/// Entry name of the test-only per-block-C build (`-DA8_DEBUG_C -DA8_SUF=_dbgc`).
fn dbg_name(f: &str) -> String {
    f.replace("_gfx1201", "_dbgc_gfx1201")
}

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}

/// Tunables the launch geometry depends on (must match the E1DN_DEFS / kernel defaults).
struct Geo {
    dn_wpb: usize,
    dn_rg: usize,
    gr_nw: usize,
    gr_rg: usize,
}

fn geo() -> Geo {
    Geo {
        dn_wpb: defs_get("A8_DN_WPB", 4),
        dn_rg: defs_get("A8_DN_RG", 1),
        gr_nw: defs_get("A8_GR_NW", 4),
        gr_rg: defs_get("A8_GR_RG", 1),
    }
}

fn defs_get(key: &str, d: usize) -> usize {
    if let Ok(v) = std::env::var("E1DN_DEFS") {
        for kv in v.split(',') {
            if let Some((k, val)) = kv.split_once('=') {
                if k == key {
                    return val.parse().unwrap_or(d);
                }
            }
        }
    }
    d
}

fn load_all(gpu: &mut Gpu) {
    let src = a8_src(false);
    let dbg = a8_src(true);
    gpu.ensure_kernel_public(MOD_A8, &src, F_QUANT)
        .unwrap_or_else(|e| panic!("load {F_QUANT}: {e}"));
    for f in [
        "qwen4_moe_down_a8_indexed_gfx1201",
        "qwen4_moe_down_a8_indexed_planar_gfx1201",
        "qwen4_moe_down_a8_grouped_gfx1201",
        "qwen4_moe_down_a8_grouped_planar_gfx1201",
        "qwen4_moe_down_a8_grouped_nodz_gfx1201",
        "qwen4_moe_down_a8_grouped_planar_nodz_gfx1201",
    ] {
        gpu.ensure_kernel_public(MOD_A8, &src, f)
            .unwrap_or_else(|e| panic!("load {f}: {e}"));
        gpu.ensure_kernel_public(MOD_A8_DBG, &dbg, &dbg_name(f))
            .unwrap_or_else(|e| panic!("load dbg {f}: {e}"));
    }
    gpu.ensure_kernel_public(
        MOD_REFQ,
        &format!("#define HIPFIRE_BLOCK_I8_128_QUANT_STANDALONE 1\n{QUANT_SRC}"),
        F_REFQ,
    )
    .expect("load ref quantizer");
    gpu.ensure_kernel_public(MOD_INC_IDX_S, INC_IDX_SHIPPED, F_INC_IDX_S)
        .expect("incumbent idx shipped");
    gpu.ensure_kernel_public(MOD_INC_IDX_P, INC_IDX_PLANAR, F_INC_IDX_P)
        .expect("incumbent idx planar");
    gpu.ensure_kernel_public(MOD_INC_GRP_S, INC_GRP_SHIPPED, F_INC_GRP_S)
        .expect("incumbent grp shipped");
    gpu.ensure_kernel_public(MOD_INC_GRP_P, INC_GRP_PLANAR, F_INC_GRP_P)
        .expect("incumbent grp planar");
    gpu.ensure_kernel_public(MOD_FLUSH, FLUSH_SRC, F_FLUSH)
        .expect("flush kernel");
    gpu.ensure_kernel_public(MOD_FLUSH, FLUSH_SRC, F_STREAM)
        .expect("stream kernel");
    gpu.ensure_kernel_public(MOD_DNI, DNI_SRC, F_DNI_S)
        .expect("dni shipped");
    gpu.ensure_kernel_public(MOD_DNI, DNI_SRC, F_DNI_P)
        .expect("dni planar");
}

fn gpu_gate() -> Option<Gpu> {
    let gpu = Gpu::init().expect("GPU required");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("SKIP: requires exact gfx1201.");
        return None;
    }
    Some(gpu)
}

// ---------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn gauss(&mut self) -> f32 {
        let r = self.next();
        let mut s = 0f32;
        for i in 0..4 {
            s += ((r >> (16 * i)) & 0xffff) as f32 / 32768.0 - 1.0;
        }
        s * 0.866_025_4
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(4)
}

fn upload_raw(gpu: &mut Gpu, data: &[u8]) -> GpuTensor {
    let t = gpu
        .alloc_tensor(&[data.len()], DType::Raw)
        .expect("alloc raw");
    gpu.hip.memcpy_htod(&t.buf, data).expect("htod");
    t
}

fn alloc_raw(gpu: &mut Gpu, bytes: usize) -> GpuTensor {
    gpu.alloc_tensor(&[bytes], DType::Raw).expect("alloc raw")
}

fn fill_bytes(gpu: &Gpu, t: &GpuTensor, byte: u8) {
    gpu.hip
        .memset(&t.buf, byte as i32, t.byte_size())
        .expect("memset");
}

fn download(gpu: &Gpu, t: &GpuTensor, off: usize, bytes: usize) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let mut out = vec![0u8; bytes];
    // SAFETY: a live tensor allocation; `off + bytes` is within it (callers size the spans).
    let view = unsafe {
        hip_bridge::DeviceBuffer::from_raw(
            (t.buf.as_ptr() as *mut u8).add(off) as *mut c_void,
            bytes,
        )
    };
    gpu.hip.memcpy_dtoh(&mut out, &view).expect("dtoh");
    out
}

fn upload_at(gpu: &Gpu, t: &GpuTensor, off: usize, data: &[u8]) {
    gpu.hip
        .memcpy_htod_offset(&t.buf, off, data)
        .expect("htod off");
}

fn ptr_at(t: &GpuTensor, off: usize) -> *mut c_void {
    // SAFETY: pointer arithmetic only (never dereferenced on the host); callers keep it in range.
    unsafe { (t.buf.as_ptr() as *mut u8).add(off) as *mut c_void }
}

fn i32b(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn par_for<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let nt = threads().min(n.max(1));
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    let chunk = n.div_ceil(nt.max(1)).max(1);
    std::thread::scope(|s| {
        for (ci, part) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, slot) in part.iter_mut().enumerate() {
                    *slot = Some(f(ci * chunk + j));
                }
            });
        }
    });
    out.into_iter().map(|o| o.unwrap()).collect()
}

// ---------------------------------------------------------------------------------------------
// HFQ index reader (container header + tensor index; payload via pread)
// ---------------------------------------------------------------------------------------------

struct Hfq {
    f: File,
    tensors: HashMap<String, (u64, u64)>, // name -> (absolute offset, size)
}

fn open_hfq(path: &Path) -> Hfq {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut header = [0u8; 32];
    f.read_exact_at(&mut header, 0).expect("hfq header");
    assert_eq!(&header[..4], b"HFQM", "bad HFQ magic");
    let n_tensors = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let metadata_off = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let data_off = u64::from_le_bytes(header[24..32].try_into().unwrap());
    let mut meta = vec![0u8; (data_off - metadata_off) as usize];
    f.read_exact_at(&mut meta, metadata_off)
        .expect("hfq metadata+index");
    let (mut depth, mut in_str, mut esc, mut json_end) = (0i32, false, false, 0usize);
    for (i, &b) in meta.iter().enumerate() {
        if esc {
            esc = false;
            continue;
        }
        if b == b'\\' && in_str {
            esc = true;
            continue;
        }
        if b == b'"' {
            in_str = !in_str;
            continue;
        }
        if !in_str {
            if b == b'{' {
                depth += 1;
            } else if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    json_end = i + 1;
                    break;
                }
            }
        }
    }
    assert!(json_end > 0, "metadata JSON not terminated");
    let mut pos = json_end;
    let idx_n = u32::from_le_bytes(meta[pos..pos + 4].try_into().unwrap()) as usize;
    assert_eq!(idx_n, n_tensors, "index count");
    pos += 4;
    let mut tensors = HashMap::new();
    let mut cum = data_off;
    for _ in 0..n_tensors {
        let nl = u16::from_le_bytes(meta[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        let name = String::from_utf8_lossy(&meta[pos..pos + nl]).to_string();
        pos += nl;
        pos += 1; // quant type
        let nd = meta[pos] as usize;
        pos += 1 + 4 * nd;
        pos += 4; // group size
        let size = u64::from_le_bytes(meta[pos..pos + 8].try_into().unwrap());
        pos += 8;
        tensors.insert(name, (cum, size));
        cum += size;
    }
    Hfq { f, tensors }
}

// ---------------------------------------------------------------------------------------------
// layouts
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Shipped,
    Planar,
}

impl Layout {
    fn per(self) -> usize {
        match self {
            Layout::Shipped => SHIP,
            Layout::Planar => PLAN,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Layout::Shipped => "shipped",
            Layout::Planar => "planar",
        }
    }
    fn a8_idx(self) -> &'static str {
        match self {
            Layout::Shipped => "qwen4_moe_down_a8_indexed_gfx1201",
            Layout::Planar => "qwen4_moe_down_a8_indexed_planar_gfx1201",
        }
    }
    fn a8_grp(self) -> &'static str {
        match self {
            Layout::Shipped => "qwen4_moe_down_a8_grouped_gfx1201",
            Layout::Planar => "qwen4_moe_down_a8_grouped_planar_gfx1201",
        }
    }
    fn a8_grp_nodz(self) -> &'static str {
        match self {
            Layout::Shipped => "qwen4_moe_down_a8_grouped_nodz_gfx1201",
            Layout::Planar => "qwen4_moe_down_a8_grouped_planar_nodz_gfx1201",
        }
    }
    fn inc_idx(self) -> &'static str {
        match self {
            Layout::Shipped => F_INC_IDX_S,
            Layout::Planar => F_INC_IDX_P,
        }
    }
    fn inc_grp(self) -> &'static str {
        match self {
            Layout::Shipped => F_INC_GRP_S,
            Layout::Planar => F_INC_GRP_P,
        }
    }
}

/// Shipped qt53 expert bytes -> F1 planar bytes (independent of hipfire_runtime::mq4_planar; the
/// reference below decodes the planar bytes themselves, so a repack bug would show as a mismatch
/// against the SHIPPED-layout decode in `layouts_decode_identically`).
fn repack_planar(ship: &[u8]) -> Vec<u8> {
    assert_eq!(ship.len(), SHIP);
    let mut out = vec![0u8; PLAN];
    for row in 0..M {
        for g in 0..G {
            let b = row * G * 68 + g * 68;
            out[row * 320 + g * 64..row * 320 + g * 64 + 64].copy_from_slice(&ship[b + 4..b + 68]);
            let so = M * 320 + row * 10 + 2 * g;
            out[so..so + 2].copy_from_slice(&ship[b..b + 2]);
        }
    }
    out
}

/// Row `row` of an expert in layout `l`: natural-order codes 0..15 and the five f16 scale bit patterns.
fn row_codes(l: Layout, exp: &[u8], row: usize) -> ([u8; K], [u16; G]) {
    let mut codes = [0u8; K];
    let mut sc = [0u16; G];
    for g in 0..G {
        let (pay, sb): (&[u8], [u8; 2]) = match l {
            Layout::Shipped => {
                let b = row * G * 68 + g * 68;
                (&exp[b + 4..b + 68], [exp[b], exp[b + 1]])
            }
            Layout::Planar => {
                let so = M * 320 + row * 10 + 2 * g;
                (
                    &exp[row * 320 + g * 64..row * 320 + g * 64 + 64],
                    [exp[so], exp[so + 1]],
                )
            }
        };
        sc[g] = u16::from_le_bytes(sb);
        for b in 0..64 {
            codes[128 * g + 2 * b] = pay[b] & 15;
            codes[128 * g + 2 * b + 1] = pay[b] >> 4;
        }
    }
    (codes, sc)
}

// ---------------------------------------------------------------------------------------------
// CPU reference (independent scalar)
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
struct QBlock {
    d: f32,
    s: i32,
    q: [i32; 128],
}

/// The shared quantizer contract (block_i8_128_quant.hip header) on 128 contiguous values.
fn quant_block(x: &[f32]) -> QBlock {
    let mut amax = 0f32;
    for &v in x {
        amax = amax.max(v.abs());
    }
    let mut b = QBlock {
        d: 0.0,
        s: 0,
        q: [0; 128],
    };
    if amax == 0.0 {
        return b;
    }
    b.d = amax / 127.0;
    for (i, &v) in x.iter().enumerate() {
        let q = (v / b.d).round_ties_even().clamp(-127.0, 127.0) as i32;
        b.q[i] = q;
        b.s += q;
    }
    b
}

fn quant_row(x: &[f32]) -> [QBlock; G] {
    std::array::from_fn(|g| quant_block(&x[128 * g..128 * g + 128]))
}

/// Documented qs byte order: qs[8g + i] = q[8g + 2i], qs[8g + 4 + i] = q[8g + 2i + 1].
fn qs_from_natural(q: &[i32; 128]) -> [u8; 128] {
    let mut qs = [0u8; 128];
    for g in 0..16 {
        for i in 0..4 {
            qs[8 * g + i] = q[8 * g + 2 * i] as i8 as u8;
            qs[8 * g + 4 + i] = q[8 * g + 2 * i + 1] as i8 as u8;
        }
    }
    qs
}

fn block_bytes(b: &QBlock) -> [u8; BLK] {
    let mut out = [0u8; BLK];
    out[0..4].copy_from_slice(&b.d.to_le_bytes());
    out[4..8].copy_from_slice(&b.s.to_le_bytes());
    out[8..].copy_from_slice(&qs_from_natural(&b.q));
    out
}

fn row_bytes(r: &[QBlock; G]) -> Vec<u8> {
    r.iter().flat_map(|b| block_bytes(b)).collect()
}

struct PickRef {
    y: Vec<f32>,
    c: Vec<i32>, // [G][M]
}

fn ref_pick(l: Layout, exp: &[u8], xq: &[QBlock; G], wrong_order: bool) -> PickRef {
    let mut y = vec![0f32; M];
    let mut c = vec![0i32; G * M];
    for row in 0..M {
        let (codes, sc) = row_codes(l, exp, row);
        let mut acc = 0f32;
        for g in 0..G {
            let xb = if wrong_order {
                &xq[(g + 1) % G]
            } else {
                &xq[g]
            };
            let mut ci = 0i32;
            for k in 0..128 {
                ci += (codes[128 * g + k] as i32 - 8) * xb.q[k];
            }
            c[g * M + row] = ci;
            let sd = f16_to_f32(sc[g]) * xb.d;
            acc = sd.mul_add(ci as f32, acc);
        }
        y[row] = acc;
    }
    PickRef { y, c }
}

// ---------------------------------------------------------------------------------------------
// expert stacks
// ---------------------------------------------------------------------------------------------

struct Stack {
    layout: Layout,
    buf: GpuTensor,
    table: GpuTensor,
    n: usize,
    per: usize,
    /// layout bytes of the experts kept for the CPU reference (local index -> bytes)
    host: HashMap<usize, Vec<u8>>,
}

fn upload_stack(
    gpu: &mut Gpu,
    layout: Layout,
    experts: &HashMap<usize, Vec<u8>>,
    n: usize,
    keep_host: bool,
) -> Stack {
    let per = layout.per();
    let buf = alloc_raw(gpu, n * per);
    fill_bytes(gpu, &buf, 0);
    let mut keys: Vec<usize> = experts.keys().copied().collect();
    keys.sort_unstable();
    for i in keys {
        assert_eq!(experts[&i].len(), per);
        upload_at(gpu, &buf, i * per, &experts[&i]);
    }
    let base = buf.buf.as_ptr() as usize as u64;
    let tb: Vec<u8> = (0..n)
        .flat_map(|i| (base + (i * per) as u64).to_le_bytes())
        .collect();
    let table = upload_raw(gpu, &tb);
    Stack {
        layout,
        buf,
        table,
        n,
        per,
        host: if keep_host {
            experts.clone()
        } else {
            HashMap::new()
        },
    }
}

fn free_stack(gpu: &mut Gpu, st: Stack) {
    let _ = gpu.free_tensor(st.buf);
    let _ = gpu.free_tensor(st.table);
}

/// Real down tensor of `layer`: owned (even global) experts `want` (local indices; None = all 256),
/// shipped layout bytes read through pread (and repacked to planar when requested).
fn real_experts(
    hfq: &Hfq,
    layer: usize,
    layout: Layout,
    want: Option<&BTreeSet<usize>>,
) -> HashMap<usize, Vec<u8>> {
    let name = format!("model.language_model.layers.{layer}.mlp.experts.down_proj");
    let &(off, size) = hfq
        .tensors
        .get(&name)
        .unwrap_or_else(|| panic!("tensor {name} not in artifact"));
    assert_eq!(size as usize, N_GLOBAL * SHIP, "{name}: unexpected size");
    let idx: Vec<usize> = match want {
        Some(w) => w.iter().copied().collect(),
        None => (0..N_LOCAL).collect(),
    };
    let blobs = par_for(idx.len(), |j| {
        let i = idx[j];
        let mut b = vec![0u8; SHIP];
        hfq.f
            .read_exact_at(&mut b, off + (2 * i * SHIP) as u64)
            .expect("pread expert");
        if layout == Layout::Planar {
            repack_planar(&b)
        } else {
            b
        }
    });
    idx.into_iter().zip(blobs).collect()
}

// ---------------------------------------------------------------------------------------------
// routes + grouping metadata
// ---------------------------------------------------------------------------------------------

struct Rec {
    layer: usize,
    ids: Vec<i32>, // rows * 10 global ids
}

fn read_routes(path: &Path) -> Vec<Rec> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let mut pos = 4;
    let mut out = Vec::new();
    for _ in 0..n {
        let layer = u16::from_le_bytes(b[pos..pos + 2].try_into().unwrap()) as usize;
        let rows = u16::from_le_bytes(b[pos + 2..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let ids = (0..rows * KTOP)
            .map(|i| i32::from_le_bytes(b[pos + 4 * i..pos + 4 * i + 4].try_into().unwrap()))
            .collect();
        pos += rows * KTOP * 4;
        out.push(Rec { layer, ids });
    }
    out
}

struct Routes {
    ar: Vec<Rec>,
    wide: Vec<Rec>,
}

fn load_routes() -> Routes {
    let dir = std::env::var("E1DN_ROUTES").unwrap_or_else(|_| DEFAULT_ROUTES.to_string());
    Routes {
        ar: read_routes(&Path::new(&dir).join("w1-real-k3-ar-rows2to8.bin")),
        wide: read_routes(&Path::new(&dir).join("w1-real-k3-wide-rows9plus.bin")),
    }
}

/// `rows` rows of real W1 routes for `layer`, variant `v`: the v-th record of that layer (ar file for
/// rows <= 8, wide file otherwise) truncated to `rows`, concatenating consecutive records of the layer
/// when no record is long enough. Returns global expert ids [rows * 10].
fn route_rows(r: &Routes, layer: usize, rows: usize, v: usize) -> Vec<i32> {
    let src = if rows <= 8 { &r.ar } else { &r.wide };
    let recs: Vec<&Rec> = src.iter().filter(|x| x.layer == layer).collect();
    assert!(!recs.is_empty(), "no route records for layer {layer}");
    let long: Vec<&&Rec> = recs.iter().filter(|x| x.ids.len() >= rows * KTOP).collect();
    if !long.is_empty() {
        return long[v % long.len()].ids[..rows * KTOP].to_vec();
    }
    let mut out = Vec::new();
    let mut i = v % recs.len();
    while out.len() < rows * KTOP {
        out.extend_from_slice(&recs[i % recs.len()].ids);
        i += 1;
    }
    out.truncate(rows * KTOP);
    out
}

#[derive(Clone)]
struct Meta {
    rows: usize,
    /// local expert per flat pick (-1 = not owned / invalid)
    topk: Vec<i32>,
    /// tiles/sorted over the PADDED cap `bound` (production hands the launcher this); `live` is the prefix
    tiles: Vec<i32>,
    sorted: Vec<i32>,
    live: usize,
    bound: usize,
    distinct: usize,
}

fn m_total_bound(slots: usize, n_exp: usize) -> usize {
    (slots + TILE.saturating_sub(1) * slots.min(n_exp)).div_ceil(TILE) * TILE
}

fn build_meta_local(topk: Vec<i32>, rows: usize, n_local: usize) -> Meta {
    let slots = rows * KTOP;
    let mut per: Vec<Vec<i32>> = vec![Vec::new(); n_local];
    for (flat, &e) in topk.iter().enumerate() {
        if e >= 0 && (e as usize) < n_local {
            per[e as usize].push(flat as i32);
        }
    }
    let live: usize = per.iter().map(|l| l.len().div_ceil(TILE) * TILE).sum();
    let bound = m_total_bound(slots, N_GLOBAL).max(live);
    let mut tiles = vec![-1i32; bound / TILE];
    let mut sorted = vec![-1i32; bound];
    let mut off = 0usize;
    let mut distinct = 0;
    for (e, l) in per.iter().enumerate() {
        if l.is_empty() {
            continue;
        }
        distinct += 1;
        for (j, &flat) in l.iter().enumerate() {
            sorted[off + j] = flat;
        }
        for t in 0..l.len().div_ceil(TILE) {
            tiles[off / TILE + t] = e as i32;
        }
        off += l.len().div_ceil(TILE) * TILE;
    }
    Meta {
        rows,
        topk,
        tiles,
        sorted,
        live,
        bound,
        distinct,
    }
}

/// EP2 rank 0 view of global ids.
fn build_meta(global: &[i32]) -> Meta {
    let rows = global.len() / KTOP;
    let topk = global
        .iter()
        .map(|&e| if e >= 0 && e % 2 == 0 { e / 2 } else { -1 })
        .collect();
    build_meta_local(topk, rows, N_LOCAL)
}

// ---------------------------------------------------------------------------------------------
// activations
// ---------------------------------------------------------------------------------------------

/// Real-shaped synthetic rotated activations: near-Gaussian blocks of varying scale with sparse outliers.
fn gen_x_gauss(rows: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = vec![0f32; rows * K];
    for r in 0..rows {
        for g in 0..G {
            let sigma = (0.5 * rng.gauss()).exp();
            for k in 0..128 {
                let mut v = sigma * rng.gauss();
                if rng.below(512) == 0 {
                    v *= 6.0;
                }
                x[r * K + 128 * g + k] = v;
            }
        }
    }
    x
}

const ACT_CLASSES: usize = 12;

fn gen_x_edge(rows: usize, rng: &mut Rng) -> Vec<f32> {
    let mut x = vec![0f32; rows * K];
    for r in 0..rows {
        for g in 0..G {
            let class = (r * 7 + g * 3 + (rng.next() % 2) as usize) % ACT_CLASSES;
            let b = &mut x[r * K + 128 * g..r * K + 128 * g + 128];
            match class {
                0 => b.iter_mut().for_each(|v| *v = rng.gauss()),
                1 => b.iter_mut().for_each(|v| *v = 0.0),
                2 => {
                    // one +-spike, everything else small: the spike quantizes to exactly +-127
                    b.iter_mut().for_each(|v| *v = rng.gauss());
                    b[rng.below(128) as usize] = if rng.below(2) == 0 { 1.0e4 } else { -1.0e4 };
                }
                3 => {
                    // ties-to-even: d = 0.25 exactly, values on the half-integer lattice
                    for v in b.iter_mut() {
                        *v = (rng.below(255) as i32 - 127) as f32 * 0.125;
                    }
                    b[rng.below(128) as usize] = 31.75;
                }
                4 => b.iter_mut().for_each(|v| *v = 1.0e-37 * rng.gauss()), // d subnormal
                5 => b.iter_mut().for_each(|v| *v = -3.0e3), // saturating, all same sign
                6 => b.iter_mut().for_each(|v| *v = 1.0e4 * rng.gauss()),
                7 => b.iter_mut().for_each(|v| *v = 1.0e-3 * rng.gauss()),
                8 => {
                    // alternating +-max (every code saturates)
                    for (i, v) in b.iter_mut().enumerate() {
                        *v = if i % 2 == 0 { 17.5 } else { -17.5 };
                    }
                }
                9 => {
                    b.iter_mut().for_each(|v| *v = 0.0);
                    b[rng.below(128) as usize] = 1.0;
                }
                10 => b.iter_mut().for_each(|v| *v = 1.0e-30 * rng.gauss()),
                _ => b.iter_mut().for_each(|v| *v = 8.0 * rng.gauss() + 1.0),
            }
        }
    }
    x
}

// ---------------------------------------------------------------------------------------------
// GPU launches
// ---------------------------------------------------------------------------------------------

fn launch_quant(gpu: &Gpu, x: *mut c_void, xq: *mut c_void, rows: usize) {
    let mut b = KernargBlob::new();
    b.push_ptr(x);
    b.push_ptr(xq);
    b.push_i32(K as i32);
    b.push_i32(rows as i32);
    assert_eq!(b.len(), 24, "producer ABI: 24 explicit kernarg bytes");
    b.pad_to(16);
    gpu.launch_kernel_blob(
        F_QUANT,
        [rows as u32, 1, 1],
        [(K / 128 * 32) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch producer");
}

fn idx_blob(
    table: *mut c_void,
    topk: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    n: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    b.push_ptr(table);
    b.push_ptr(topk);
    b.push_ptr(x);
    b.push_ptr(y);
    b.push_i32(M as i32);
    b.push_i32(K as i32);
    b.push_i32(n as i32);
    b.push_i32(N_LOCAL as i32);
    b.push_i32(KTOP as i32);
    assert_eq!(b.len(), 52, "D-indexed ABI: 52 explicit kernarg bytes");
    b.pad_to(16);
    b
}

fn launch_a8_idx(
    gpu: &Gpu,
    l: Layout,
    table: *mut c_void,
    topk: *mut c_void,
    xq: *mut c_void,
    y: *mut c_void,
    n: usize,
) {
    let g = geo();
    let mut b = idx_blob(table, topk, xq, y, n);
    let per_block = g.dn_wpb * 8 * g.dn_rg;
    gpu.launch_kernel_blob(
        l.a8_idx(),
        [(M / per_block) as u32, KTOP as u32, n as u32],
        [(g.dn_wpb * 32) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch a8 indexed");
}

fn launch_inc_idx(
    gpu: &Gpu,
    l: Layout,
    table: *mut c_void,
    topk: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    n: usize,
) {
    let mut b = idx_blob(table, topk, x, y, n);
    gpu.launch_kernel_blob(
        l.inc_idx(),
        [(M / 4) as u32, KTOP as u32, n as u32],
        [32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch incumbent indexed");
}

#[allow(clippy::too_many_arguments)]
fn grp_blob(
    table: *mut c_void,
    tiles: *mut c_void,
    sorted: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    x_row_div: usize,
    grouped_rows: usize,
    x_src_rows: usize,
) -> KernargBlob {
    let mut b = KernargBlob::new();
    b.push_ptr(table);
    b.push_ptr(tiles);
    b.push_ptr(sorted);
    b.push_ptr(x);
    b.push_ptr(y);
    b.push_i32(M as i32);
    b.push_i32(K as i32);
    b.push_i32(x_row_div as i32);
    b.push_i32(grouped_rows as i32);
    b.push_i32(x_src_rows as i32);
    b.push_i32(N_LOCAL as i32);
    assert_eq!(b.len(), 64, "D-grouped ABI: 64 explicit kernarg bytes");
    b
}

#[allow(clippy::too_many_arguments)]
fn launch_a8_grp(
    gpu: &Gpu,
    l: Layout,
    table: *mut c_void,
    tiles: *mut c_void,
    sorted: *mut c_void,
    xq: *mut c_void,
    y: *mut c_void,
    x_row_div: usize,
    grouped_rows: usize,
    x_src_rows: usize,
) {
    let g = geo();
    let mut b = grp_blob(
        table,
        tiles,
        sorted,
        xq,
        y,
        x_row_div,
        grouped_rows,
        x_src_rows,
    );
    let rows_per_cta = g.gr_nw * g.gr_rg * 16;
    gpu.launch_kernel_blob(
        l.a8_grp(),
        [
            M.div_ceil(rows_per_cta) as u32,
            grouped_rows.div_ceil(TILE) as u32,
            1,
        ],
        [(g.gr_nw * 32) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch a8 grouped");
}

#[allow(clippy::too_many_arguments)]
fn launch_inc_grp(
    gpu: &Gpu,
    l: Layout,
    table: *mut c_void,
    tiles: *mut c_void,
    sorted: *mut c_void,
    x: *mut c_void,
    y: *mut c_void,
    x_row_div: usize,
    grouped_rows: usize,
    x_src_rows: usize,
) {
    let mut b = grp_blob(
        table,
        tiles,
        sorted,
        x,
        y,
        x_row_div,
        grouped_rows,
        x_src_rows,
    );
    gpu.launch_kernel_blob(
        l.inc_grp(),
        [(M / 8) as u32, grouped_rows.div_ceil(TILE) as u32, 1],
        [32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch incumbent grouped");
}

// ---------------------------------------------------------------------------------------------
// oracle
// ---------------------------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Tally {
    cells: usize,
    pass: usize,
    mismatch_words: u64,
    nc_cells: usize,
    nc_detected: usize,
    nc_words: u64,
    lines: Vec<String>,
}

impl Tally {
    fn cell(&mut self, label: &str, mism: u64, detail: &str) {
        self.cells += 1;
        if mism == 0 {
            self.pass += 1;
        } else {
            self.mismatch_words += mism;
            let l = format!("FAIL {label}: {mism} mismatching words {detail}");
            println!("{l}");
            self.lines.push(l);
        }
    }
    fn nc(&mut self, label: &str, words: u64) {
        self.nc_cells += 1;
        self.nc_words += words;
        if words > 0 {
            self.nc_detected += 1;
        } else {
            let l = format!("FAIL negative control {label}: corruption NOT detected");
            println!("{l}");
            self.lines.push(l);
        }
    }
}
/// Expected output of a `_nodz` grouped launch into a sentinel-filled buffer: the production
/// words on every live slot, the untouched sentinel on every dead one.
fn nodz_expected(
    meta: &Meta,
    wy: &[f32],
    gr: usize,
    x_row_div: usize,
    x_src_rows: usize,
) -> Vec<u8> {
    let mut out = vec![SENT; gr * M * 4];
    for s in 0..gr {
        let t = meta.tiles[s / TILE];
        let flat = meta.sorted[s];
        if t < 0 || t as usize >= N_LOCAL || flat < 0 || (flat as usize / x_row_div) >= x_src_rows {
            continue;
        }
        out[s * M * 4..(s + 1) * M * 4].copy_from_slice(&f32b(&wy[s * M..(s + 1) * M]));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn launch_a8_grp_nodz(
    gpu: &Gpu,
    l: Layout,
    table: *mut c_void,
    tiles: *mut c_void,
    sorted: *mut c_void,
    xq: *mut c_void,
    y: *mut c_void,
    x_row_div: usize,
    grouped_rows: usize,
    x_src_rows: usize,
) {
    let g = geo();
    let mut b = grp_blob(
        table,
        tiles,
        sorted,
        xq,
        y,
        x_row_div,
        grouped_rows,
        x_src_rows,
    );
    let rows_per_cta = g.gr_nw * g.gr_rg * 16;
    gpu.launch_kernel_blob(
        l.a8_grp_nodz(),
        [
            M.div_ceil(rows_per_cta) as u32,
            grouped_rows.div_ceil(TILE) as u32,
            1,
        ],
        [(g.gr_nw * 32) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch a8 grouped nodz");
}

fn count_diff(got: &[u8], want: &[u8]) -> (u64, Option<usize>) {
    assert_eq!(got.len(), want.len());
    let mut n = 0u64;
    let mut first = None;
    for (i, (a, b)) in got.chunks_exact(4).zip(want.chunks_exact(4)).enumerate() {
        if a != b {
            n += 1;
            first.get_or_insert(i);
        }
    }
    (n, first)
}

/// Output span with guard bytes (sentinel-filled) around `words` 32-bit words.
struct OutBuf {
    t: GpuTensor,
    words: usize,
}

fn alloc_out(gpu: &mut Gpu, words: usize) -> OutBuf {
    let t = alloc_raw(gpu, GUARD * 2 + words * 4);
    OutBuf { t, words }
}

fn clear_out(gpu: &Gpu, o: &OutBuf) {
    fill_bytes(gpu, &o.t, SENT);
}

fn out_ptr(o: &OutBuf) -> *mut c_void {
    ptr_at(&o.t, GUARD)
}

/// Returns (span bytes, guards intact)
fn read_out(gpu: &Gpu, o: &OutBuf) -> (Vec<u8>, bool) {
    let all = download(gpu, &o.t, 0, GUARD * 2 + o.words * 4);
    let ok = all[..GUARD].iter().all(|&b| b == SENT)
        && all[GUARD + o.words * 4..].iter().all(|&b| b == SENT);
    (all[GUARD..GUARD + o.words * 4].to_vec(), ok)
}

/// Diagnostics accumulated per config.
#[derive(Default, Clone)]
struct Quality {
    num: f64,
    den: f64,
    max_abs: f64,
    n: u64,
}

impl Quality {
    fn add(&mut self, a8: &[f32], inc: &[f32]) {
        for (&a, &b) in a8.iter().zip(inc) {
            let d = (a as f64 - b as f64).abs();
            self.num += d * d;
            self.den += (b as f64) * (b as f64);
            self.max_abs = self.max_abs.max(d);
            self.n += 1;
        }
    }
    fn rel_l2(&self) -> f64 {
        if self.den == 0.0 {
            0.0
        } else {
            (self.num / self.den).sqrt()
        }
    }
}

struct Case<'a> {
    label: String,
    layout: Layout,
    stack: &'a Stack,
    meta: Meta,
    x_row_div: usize,
    x: Vec<f32>, // [x_src_rows][K]
    /// run the rows=1 vs batched comparison (off for metas whose dead-slot corner cases were mutated in)
    singleton: bool,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Everything one case needs on the CPU side.
struct CpuSide {
    xq: Vec<[QBlock; G]>,
    refs: HashMap<(usize, usize), PickRef>, // (local expert, x row)
}

fn cpu_side(c: &Case, keys: &BTreeSet<(usize, usize)>) -> CpuSide {
    let rows = c.x.len() / K;
    let (xs, host, layout) = (&c.x, &c.stack.host, c.layout);
    let xq: Vec<[QBlock; G]> = par_for(rows, |r| quant_row(&xs[r * K..(r + 1) * K]));
    let keys_v: Vec<(usize, usize)> = keys.iter().copied().collect();
    let out = par_for(keys_v.len(), |i| {
        let (e, r) = keys_v[i];
        ref_pick(layout, &host[&e], &xq[r], false)
    });
    CpuSide {
        xq,
        refs: keys_v.into_iter().zip(out).collect(),
    }
}

/// Expected D-indexed expanded output + C ([G][N*10][M]).
fn expected_indexed(meta: &Meta, cpu: &CpuSide, n_local: usize) -> (Vec<f32>, Vec<i32>) {
    let slots = meta.rows * KTOP;
    let mut y = vec![0f32; slots * M];
    let mut c = vec![0i32; G * slots * M];
    for flat in 0..slots {
        let e = meta.topk[flat];
        if e >= 0 && (e as usize) < n_local {
            let r = &cpu.refs[&(e as usize, flat)];
            y[flat * M..(flat + 1) * M].copy_from_slice(&r.y);
            for g in 0..G {
                c[(g * slots + flat) * M..(g * slots + flat + 1) * M]
                    .copy_from_slice(&r.c[g * M..(g + 1) * M]);
            }
        }
    }
    (y, c)
}

/// Expected D-grouped Y (+0.0 for every dead slot) + C ([G][grouped_rows][M]).
fn expected_grouped(
    meta: &Meta,
    cpu: &CpuSide,
    gr: usize,
    x_row_div: usize,
    x_src_rows: usize,
) -> (Vec<f32>, Vec<i32>) {
    let mut y = vec![0f32; gr * M];
    let mut c = vec![0i32; G * gr * M];
    for s in 0..gr {
        let t = meta.tiles[s / TILE];
        let flat = meta.sorted[s];
        if t < 0 || t as usize >= N_LOCAL || flat < 0 {
            continue;
        }
        let xr = flat as usize / x_row_div;
        if xr >= x_src_rows {
            continue;
        }
        let r = &cpu.refs[&(t as usize, xr)];
        y[s * M..(s + 1) * M].copy_from_slice(&r.y);
        for g in 0..G {
            c[(g * gr + s) * M..(g * gr + s + 1) * M].copy_from_slice(&r.c[g * M..(g + 1) * M]);
        }
    }
    (y, c)
}

fn needed_keys_indexed(meta: &Meta) -> BTreeSet<(usize, usize)> {
    let mut k = BTreeSet::new();
    for (flat, &e) in meta.topk.iter().enumerate() {
        if e >= 0 && (e as usize) < N_LOCAL {
            k.insert((e as usize, flat));
        }
    }
    k
}

fn needed_keys_grouped(
    meta: &Meta,
    gr: usize,
    x_row_div: usize,
    x_src_rows: usize,
) -> BTreeSet<(usize, usize)> {
    let mut k = BTreeSet::new();
    for s in 0..gr {
        let t = meta.tiles[s / TILE];
        let flat = meta.sorted[s];
        if t >= 0 && (t as usize) < N_LOCAL && flat >= 0 && (flat as usize / x_row_div) < x_src_rows
        {
            k.insert((t as usize, flat as usize / x_row_div));
        }
    }
    k
}

struct DevMeta {
    topk: GpuTensor,
    tiles: GpuTensor,
    sorted: GpuTensor,
}

fn upload_meta(gpu: &mut Gpu, m: &Meta) -> DevMeta {
    DevMeta {
        topk: upload_raw(gpu, &i32b(&m.topk)),
        tiles: upload_raw(gpu, &i32b(&m.tiles)),
        sorted: upload_raw(gpu, &i32b(&m.sorted)),
    }
}

fn free_meta(gpu: &mut Gpu, d: DevMeta) {
    let _ = gpu.free_tensor(d.topk);
    let _ = gpu.free_tensor(d.tiles);
    let _ = gpu.free_tensor(d.sorted);
}

fn run_case(
    gpu: &mut Gpu,
    c: &Case,
    tally: &mut Tally,
    quality: Option<&mut Quality>,
    use_idx: bool,
    use_grp: bool,
) {
    let _ = run_case_inner(gpu, c, tally, quality, use_idx, use_grp);
}

fn run_case_out(
    gpu: &mut Gpu,
    c: &Case,
    tally: &mut Tally,
    quality: Option<&mut Quality>,
) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    run_case_inner(gpu, c, tally, quality, true, true)
}

/// Run one case end to end (indexed + grouped ABIs; also the -DA8_DEBUG_C build for per-block C).
/// Returns the production indexed output and the production grouped (live rows) output bytes.
fn run_case_inner(
    gpu: &mut Gpu,
    c: &Case,
    tally: &mut Tally,
    quality: Option<&mut Quality>,
    use_idx: bool,
    use_grp: bool,
) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut ret_idx: Option<Vec<u8>> = None;
    let mut ret_grp: Option<Vec<u8>> = None;
    let meta = &c.meta;
    let x_src_rows = c.x.len() / K;
    let st = c.stack;
    let l = c.layout;
    let xt = upload_raw(gpu, &f32b(&c.x));
    let xq = alloc_raw(gpu, x_src_rows * ROWQ);
    fill_bytes(gpu, &xq, SENT);
    launch_quant(
        gpu,
        xt.buf.as_ptr() as *mut c_void,
        xq.buf.as_ptr() as *mut c_void,
        x_src_rows,
    );
    let got_xq = download(gpu, &xq, 0, x_src_rows * ROWQ);

    // ---- producer vs CPU quantizer
    let mut keys = BTreeSet::new();
    if use_idx {
        keys.extend(needed_keys_indexed(meta));
    }
    let gr_live = meta.live;
    let gr_bound = meta.bound;
    if use_grp {
        keys.extend(needed_keys_grouped(meta, gr_bound, c.x_row_div, x_src_rows));
    }
    let cpu = cpu_side(c, &keys);
    let want_xq: Vec<u8> = cpu.xq.iter().flat_map(|r| row_bytes(r)).collect();
    let xq_bad = got_xq.iter().zip(&want_xq).filter(|(a, b)| a != b).count() as u64;
    tally.cell(
        &format!("{} producer-vs-cpu-quantizer", c.label),
        xq_bad,
        "(bytes)",
    );

    // ---- producer vs the shared reference producer quantize_int8_mmq_ds128 ([kb][N] layout)
    {
        let refq = alloc_raw(gpu, x_src_rows * ROWQ);
        fill_bytes(gpu, &refq, SENT);
        let mut b = KernargBlob::new();
        b.push_ptr(xt.buf.as_ptr() as *mut c_void);
        b.push_ptr(refq.buf.as_ptr() as *mut c_void);
        b.push_i32(K as i32);
        b.push_i32(x_src_rows as i32);
        b.pad_to(16);
        let per_row = K.div_ceil(1024);
        gpu.launch_kernel_blob(
            F_REFQ,
            [(per_row * x_src_rows) as u32, 1, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .expect("launch ref quantizer");
        let rq = download(gpu, &refq, 0, x_src_rows * ROWQ);
        let mut bad = 0u64;
        for r in 0..x_src_rows {
            for g in 0..G {
                let a = &got_xq[(r * G + g) * BLK..(r * G + g + 1) * BLK];
                let bb = &rq[(g * x_src_rows + r) * BLK..(g * x_src_rows + r + 1) * BLK];
                bad += a.iter().zip(bb).filter(|(x, y)| x != y).count() as u64;
            }
        }
        tally.cell(
            &format!("{} producer-vs-quantize_int8_mmq_ds128", c.label),
            bad,
            "(bytes)",
        );
        let _ = gpu.free_tensor(refq);
    }

    let dm = upload_meta(gpu, meta);
    let mut q = quality;

    // ---- D-indexed
    if use_idx {
        let slots = meta.rows * KTOP;
        let (wy, wc) = expected_indexed(meta, &cpu, N_LOCAL);
        let o = alloc_out(gpu, slots * M);
        clear_out(gpu, &o);
        launch_a8_idx(
            gpu,
            l,
            st.table.buf.as_ptr() as *mut c_void,
            dm.topk.buf.as_ptr() as *mut c_void,
            xq.buf.as_ptr() as *mut c_void,
            out_ptr(&o),
            meta.rows,
        );
        let (got, guards) = read_out(gpu, &o);
        let (n, first) = count_diff(&got, &f32b(&wy));
        tally.cell(
            &format!("{} indexed F32 words", c.label),
            n + (!guards) as u64,
            &format!("first={first:?} guards_ok={guards}"),
        );
        ret_idx = Some(got.clone());
        // singleton == batched, per token
        let mut single_bad = 0u64;
        let og = alloc_out(gpu, KTOP * M);
        for tok in 0..meta.rows {
            clear_out(gpu, &og);
            launch_a8_idx(
                gpu,
                l,
                st.table.buf.as_ptr() as *mut c_void,
                ptr_at(&dm.topk, tok * KTOP * 4),
                ptr_at(&xq, tok * KTOP * ROWQ),
                out_ptr(&og),
                1,
            );
            let (g1, ok) = read_out(gpu, &og);
            single_bad += (!ok) as u64;
            single_bad += count_diff(&g1, &got[tok * KTOP * M * 4..(tok + 1) * KTOP * M * 4]).0;
        }
        tally.cell(
            &format!("{} indexed singleton(rows=1)==batched", c.label),
            single_bad,
            "(words)",
        );
        let _ = gpu.free_tensor(og.t);
        // per-block C via the debug build
        {
            let od = alloc_out(gpu, slots * M * (1 + G));
            clear_out(gpu, &od);
            launch_dbg_idx(gpu, l, st, &dm, &xq, &od, meta.rows);
            let (gotd, ok) = read_out(gpu, &od);
            let (ny, _) = count_diff(&gotd[..slots * M * 4], &f32b(&wy));
            let (nc, firstc) = count_diff(&gotd[slots * M * 4..], &i32b(&wc));
            tally.cell(
                &format!("{} indexed per-block-C(+Y) words", c.label),
                ny + nc + (!ok) as u64,
                &format!("firstC={firstc:?}"),
            );
            let _ = gpu.free_tensor(od.t);
        }
        if let Some(q) = q.as_deref_mut() {
            // quality vs the incumbent F32 kernel on the same weights / routes / X
            let oi = alloc_out(gpu, slots * M);
            clear_out(gpu, &oi);
            launch_inc_idx(
                gpu,
                l,
                st.table.buf.as_ptr() as *mut c_void,
                dm.topk.buf.as_ptr() as *mut c_void,
                xt.buf.as_ptr() as *mut c_void,
                out_ptr(&oi),
                meta.rows,
            );
            let (inc, _) = read_out(gpu, &oi);
            q.add(&f32s(&got), &f32s(&inc));
            let _ = gpu.free_tensor(oi.t);
        }
        let _ = gpu.free_tensor(o.t);
    }

    // ---- D-grouped (live rows prefix and the padded cap, the production default hands the cap)
    if use_grp {
        let mut by_idx: Option<Vec<u8>> = None;
        for (gname, gr) in [("live", gr_live), ("bound", gr_bound)] {
            if gr == 0 {
                continue;
            }
            let (wy, wc) = expected_grouped(meta, &cpu, gr, c.x_row_div, x_src_rows);
            let o = alloc_out(gpu, gr * M);
            clear_out(gpu, &o);
            launch_a8_grp(
                gpu,
                l,
                st.table.buf.as_ptr() as *mut c_void,
                dm.tiles.buf.as_ptr() as *mut c_void,
                dm.sorted.buf.as_ptr() as *mut c_void,
                xq.buf.as_ptr() as *mut c_void,
                out_ptr(&o),
                c.x_row_div,
                gr,
                x_src_rows,
            );
            let (got, guards) = read_out(gpu, &o);
            let (n, first) = count_diff(&got, &f32b(&wy));
            tally.cell(
                &format!("{} grouped[{gname} gr={gr}] F32 words", c.label),
                n + (!guards) as u64,
                &format!("first={first:?} guards_ok={guards}"),
            );
            {
                // `_nodz` twin: live rows == the production words, dead rows (sentinel tile, pad slot,
                // out-of-range x row) NOT written (the sentinel fill survives), guards intact.
                let on = alloc_out(gpu, gr * M);
                clear_out(gpu, &on);
                launch_a8_grp_nodz(
                    gpu,
                    l,
                    st.table.buf.as_ptr() as *mut c_void,
                    dm.tiles.buf.as_ptr() as *mut c_void,
                    dm.sorted.buf.as_ptr() as *mut c_void,
                    xq.buf.as_ptr() as *mut c_void,
                    out_ptr(&on),
                    c.x_row_div,
                    gr,
                    x_src_rows,
                );
                let (gotn, okn) = read_out(gpu, &on);
                let want_n = nodz_expected(meta, &wy, gr, c.x_row_div, x_src_rows);
                let (nn, firstn) = count_diff(&gotn, &want_n);
                tally.cell(
                    &format!(
                        "{} grouped[{gname} gr={gr}] nodz live words + untouched dead",
                        c.label
                    ),
                    nn + (!okn) as u64,
                    &format!("first={firstn:?} guards_ok={okn}"),
                );
                let _ = gpu.free_tensor(on.t);
            }
            {
                let od = alloc_out(gpu, gr * M * (1 + G));
                clear_out(gpu, &od);
                launch_dbg_grp(gpu, l, st, &dm, &xq, &od, c.x_row_div, gr, x_src_rows);
                let (gotd, ok) = read_out(gpu, &od);
                let (ny, _) = count_diff(&gotd[..gr * M * 4], &f32b(&wy));
                let (nc, firstc) = count_diff(&gotd[gr * M * 4..], &i32b(&wc));
                tally.cell(
                    &format!("{} grouped[{gname}] per-block-C(+Y) words", c.label),
                    ny + nc + (!ok) as u64,
                    &format!("firstC={firstc:?}"),
                );
                let _ = gpu.free_tensor(od.t);
            }
            if gname == "live" {
                by_idx = Some(got.clone());
                ret_grp = Some(got.clone());
                if let Some(q) = q.as_deref_mut() {
                    let oi = alloc_out(gpu, gr * M);
                    clear_out(gpu, &oi);
                    launch_inc_grp(
                        gpu,
                        l,
                        st.table.buf.as_ptr() as *mut c_void,
                        dm.tiles.buf.as_ptr() as *mut c_void,
                        dm.sorted.buf.as_ptr() as *mut c_void,
                        xt.buf.as_ptr() as *mut c_void,
                        out_ptr(&oi),
                        c.x_row_div,
                        gr,
                        x_src_rows,
                    );
                    let (inc, _) = read_out(gpu, &oi);
                    q.add(&f32s(&got), &f32s(&inc));
                    let _ = gpu.free_tensor(oi.t);
                }
            }
            let _ = gpu.free_tensor(o.t);
        }
        // grouped singleton == batched: one token's 10 picks as their own grouped launch
        if c.x_row_div == 1 && c.singleton && meta.rows >= 1 {
            let got = by_idx.expect("grouped live output");
            let mut bad = 0u64;
            let toks = meta.rows.min(3);
            for tok in 0..toks {
                let sub: Vec<i32> = (0..KTOP).map(|j| meta.topk[tok * KTOP + j]).collect();
                let sm = build_meta_local(sub, 1, N_LOCAL);
                let sdm = upload_meta(gpu, &sm);
                let og = alloc_out(gpu, sm.live.max(TILE) * M);
                clear_out(gpu, &og);
                launch_a8_grp(
                    gpu,
                    l,
                    st.table.buf.as_ptr() as *mut c_void,
                    sdm.tiles.buf.as_ptr() as *mut c_void,
                    sdm.sorted.buf.as_ptr() as *mut c_void,
                    ptr_at(&xq, tok * KTOP * ROWQ),
                    out_ptr(&og),
                    1,
                    sm.live,
                    KTOP,
                );
                let (g1, ok) = read_out(gpu, &og);
                bad += (!ok) as u64;
                // map: batched slot of (token, rank) <-> singleton slot of rank
                for rank in 0..KTOP {
                    if sm.topk[rank] < 0 || sm.topk[rank] as usize >= N_LOCAL {
                        continue;
                    }
                    let s1 = sm.sorted.iter().position(|&f| f == rank as i32).unwrap();
                    // (a mutated synthetic meta may have lost this pick's slot: nothing to compare)
                    let Some(sb) = meta.sorted[..meta.live]
                        .iter()
                        .position(|&f| f == (tok * KTOP + rank) as i32)
                    else {
                        continue;
                    };
                    bad += count_diff(
                        &g1[s1 * M * 4..(s1 + 1) * M * 4],
                        &got[sb * M * 4..(sb + 1) * M * 4],
                    )
                    .0;
                }
                free_meta(gpu, sdm);
                let _ = gpu.free_tensor(og.t);
            }
            tally.cell(
                &format!("{} grouped singleton(rows=1)==batched", c.label),
                bad,
                "(words)",
            );
        }
    }
    free_meta(gpu, dm);
    let _ = gpu.free_tensor(xt);
    let _ = gpu.free_tensor(xq);
    (ret_idx, ret_grp)
}

fn launch_dbg_idx(
    gpu: &Gpu,
    l: Layout,
    st: &Stack,
    dm: &DevMeta,
    xq: &GpuTensor,
    o: &OutBuf,
    n: usize,
) {
    let g = geo();
    let mut b = idx_blob(
        st.table.buf.as_ptr() as *mut c_void,
        dm.topk.buf.as_ptr() as *mut c_void,
        xq.buf.as_ptr() as *mut c_void,
        out_ptr(o),
        n,
    );
    let per_block = g.dn_wpb * 8 * g.dn_rg;
    gpu.launch_kernel_blob(
        &dbg_name(l.a8_idx()),
        [(M / per_block) as u32, KTOP as u32, n as u32],
        [(g.dn_wpb * 32) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch dbg indexed");
}

#[allow(clippy::too_many_arguments)]
fn launch_dbg_grp(
    gpu: &Gpu,
    l: Layout,
    st: &Stack,
    dm: &DevMeta,
    xq: &GpuTensor,
    o: &OutBuf,
    x_row_div: usize,
    gr: usize,
    x_src_rows: usize,
) {
    let g = geo();
    let mut b = grp_blob(
        st.table.buf.as_ptr() as *mut c_void,
        dm.tiles.buf.as_ptr() as *mut c_void,
        dm.sorted.buf.as_ptr() as *mut c_void,
        xq.buf.as_ptr() as *mut c_void,
        out_ptr(o),
        x_row_div,
        gr,
        x_src_rows,
    );
    let rows_per_cta = g.gr_nw * g.gr_rg * 16;
    gpu.launch_kernel_blob(
        &dbg_name(l.a8_grp()),
        [M.div_ceil(rows_per_cta) as u32, gr.div_ceil(TILE) as u32, 1],
        [(g.gr_nw * 32) as u32, 1, 1],
        0,
        b.as_mut_slice(),
    )
    .expect("launch dbg grouped");
}

// ---------------------------------------------------------------------------------------------
// synthetic experts
// ---------------------------------------------------------------------------------------------

const SYN_EXPERTS: usize = 12;

fn syn_scale(mode: usize, rng: &mut Rng) -> u16 {
    match mode % 6 {
        0 => half::f16::from_f32(0.002 + 0.028 * rng.unit()).to_bits(),
        1 => 0x0000,
        2 => 1 + (rng.below(0x3ff) as u16), // f16 subnormal
        3 => 0x6bff,                        // large (8 s still finite)
        4 => half::f16::from_f32(-(0.002 + 0.028 * rng.unit())).to_bits(),
        _ => match rng.below(5) {
            0 => 0x0000,
            1 => 1 + (rng.below(0x3ff) as u16),
            2 => 0x6bff,
            3 => half::f16::from_f32(-(0.01 + rng.unit())).to_bits(),
            _ => half::f16::from_f32(0.002 + 0.028 * rng.unit()).to_bits(),
        },
    }
}

/// Shipped bytes of a synthetic symmetric expert (z = f16(-8 s), or garbage when `z_garbage`: the kernels
/// must ignore z).
fn syn_shipped(e: usize, rng: &mut Rng, z_garbage: bool) -> Vec<u8> {
    let smode = e % 6;
    let cmode = (e / 2) % 6;
    let mut b = vec![0u8; SHIP];
    for row in 0..M {
        for g in 0..G {
            let o = row * G * 68 + g * 68;
            let s = syn_scale(smode, rng);
            let z = if z_garbage {
                rng.next() as u16
            } else {
                half::f16::from_f32(-8.0 * f16_to_f32(s)).to_bits()
            };
            b[o..o + 2].copy_from_slice(&s.to_le_bytes());
            b[o + 2..o + 4].copy_from_slice(&z.to_le_bytes());
            for j in 0..64 {
                b[o + 4 + j] = match cmode {
                    0 => rng.next() as u8,
                    1 => 0x00,
                    2 => 0xFF,
                    3 => 0x88,
                    4 => 0xF0,
                    _ => {
                        if rng.below(4) == 0 {
                            0xF0 | (rng.next() as u8 & 0x0F)
                        } else {
                            rng.next() as u8
                        }
                    }
                };
            }
        }
    }
    b
}

fn syn_experts(layout: Layout, seed: u64, z_garbage: bool) -> HashMap<usize, Vec<u8>> {
    let mut m = HashMap::new();
    for e in 0..SYN_EXPERTS {
        let mut rng = Rng(seed ^ (e as u64 * 0x1234_5678_9abc));
        let ship = syn_shipped(e, &mut rng, z_garbage);
        m.insert(
            e,
            if layout == Layout::Planar {
                repack_planar(&ship)
            } else {
                ship
            },
        );
    }
    // two valid-but-empty experts (zero scales, zero codes): the router may pick them
    for e in SYN_EXPERTS..SYN_EXPERTS + 2 {
        m.insert(e, vec![0u8; layout.per()]);
    }
    m
}

fn syn_meta(rows: usize, rng: &mut Rng, invalid: bool) -> Meta {
    let mut topk: Vec<i32> = Vec::new();
    for _ in 0..rows {
        let mut picks = Vec::new();
        while picks.len() < KTOP {
            let e = rng.below(SYN_EXPERTS as u64 + 2) as i32; // two extra ids = valid but empty experts
            if !picks.contains(&e) {
                picks.push(e);
            }
        }
        topk.extend(picks);
    }
    if invalid {
        for (i, t) in topk.iter_mut().enumerate() {
            if i % 7 == 3 {
                *t = -1;
            } else if i % 13 == 5 {
                *t = (N_LOCAL + 3) as i32;
            } else if i % 31 == 11 {
                *t = i32::MAX;
            }
        }
    }
    build_meta_local(topk, rows, N_LOCAL)
}

// ---------------------------------------------------------------------------------------------
// negative controls: corrupt the device state, the oracle must see it
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Nc {
    WrongScaleByte,
    WrongBlockOrder,
}

/// Returns (indexed diff words, grouped diff words) of the CORRUPTED run against the UNCORRUPTED reference.
fn nc_run(gpu: &mut Gpu, c: &Case, nc: Nc) -> (u64, u64) {
    let meta = &c.meta;
    let x_src_rows = c.x.len() / K;
    let st = c.stack;
    let l = c.layout;
    let mut keys = needed_keys_indexed(meta);
    keys.extend(needed_keys_grouped(
        meta,
        meta.live,
        c.x_row_div,
        x_src_rows,
    ));
    let cpu = cpu_side(c, &keys);
    let (wy_i, _) = expected_indexed(meta, &cpu, N_LOCAL);
    let (wy_g, _) = expected_grouped(meta, &cpu, meta.live, c.x_row_div, x_src_rows);
    let xt = upload_raw(gpu, &f32b(&c.x));
    let xq = alloc_raw(gpu, x_src_rows * ROWQ);
    launch_quant(
        gpu,
        xt.buf.as_ptr() as *mut c_void,
        xq.buf.as_ptr() as *mut c_void,
        x_src_rows,
    );
    // victim: the first owned pick
    let flat = meta
        .topk
        .iter()
        .position(|&e| e >= 0)
        .expect("an owned pick");
    let e = meta.topk[flat] as usize;
    let mut restore: Option<(usize, u8)> = None;
    match nc {
        Nc::WrongScaleByte => {
            // flip the high byte of the group-2 scale of row 0 of the victim expert (device only)
            let rel = match l {
                Layout::Shipped => 2 * 68 + 1,
                Layout::Planar => M * 320 + 2 * 2 + 1,
            };
            let orig = st.host[&e][rel];
            let off = e * st.per + rel;
            upload_at(gpu, &st.buf, off, &[orig ^ 0x40]);
            restore = Some((off, orig));
        }
        Nc::WrongBlockOrder => {
            let row = download(gpu, &xq, flat * ROWQ, ROWQ);
            let mut sw = row.clone();
            sw[BLK..2 * BLK].copy_from_slice(&row[2 * BLK..3 * BLK]);
            sw[2 * BLK..3 * BLK].copy_from_slice(&row[BLK..2 * BLK]);
            upload_at(gpu, &xq, flat * ROWQ, &sw);
        }
    }
    let dm = upload_meta(gpu, meta);
    let slots = meta.rows * KTOP;
    let o = alloc_out(gpu, slots * M);
    clear_out(gpu, &o);
    launch_a8_idx(
        gpu,
        l,
        st.table.buf.as_ptr() as *mut c_void,
        dm.topk.buf.as_ptr() as *mut c_void,
        xq.buf.as_ptr() as *mut c_void,
        out_ptr(&o),
        meta.rows,
    );
    let (got, _) = read_out(gpu, &o);
    let bad_i = count_diff(&got, &f32b(&wy_i)).0;
    let og = alloc_out(gpu, meta.live * M);
    clear_out(gpu, &og);
    launch_a8_grp(
        gpu,
        l,
        st.table.buf.as_ptr() as *mut c_void,
        dm.tiles.buf.as_ptr() as *mut c_void,
        dm.sorted.buf.as_ptr() as *mut c_void,
        xq.buf.as_ptr() as *mut c_void,
        out_ptr(&og),
        c.x_row_div,
        meta.live,
        x_src_rows,
    );
    let (gotg, _) = read_out(gpu, &og);
    let bad_g = count_diff(&gotg, &f32b(&wy_g)).0;
    if let Some((off, orig)) = restore {
        upload_at(gpu, &st.buf, off, &[orig]);
    }
    free_meta(gpu, dm);
    let _ = gpu.free_tensor(o.t);
    let _ = gpu.free_tensor(og.t);
    let _ = gpu.free_tensor(xt);
    let _ = gpu.free_tensor(xq);
    (bad_i, bad_g)
}

// ---------------------------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------------------------

#[test]
fn layouts_decode_identically() {
    let mut rng = Rng(7);
    let ship = syn_shipped(5, &mut rng, false);
    let plan = repack_planar(&ship);
    assert_eq!(plan.len(), PLAN);
    for row in [0usize, 1, 17, 1279, 2559] {
        let a = row_codes(Layout::Shipped, &ship, row);
        let b = row_codes(Layout::Planar, &plan, row);
        assert_eq!(a.0[..], b.0[..], "row {row} codes");
        assert_eq!(a.1, b.1, "row {row} scales");
    }
}

#[test]
fn cpu_quantizer_ties_and_order() {
    // 0.125 lattice with amax 31.75: d = 0.25 exactly, x / d = half-integers -> ties to even
    let mut x = [0f32; 128];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i % 255) as i32 - 127) as f32 * 0.125;
    }
    x[0] = 31.75;
    let b = quant_block(&x);
    assert_eq!(b.d, 0.25);
    assert_eq!(b.q[0], 127);
    // x[1] = (1 - 127) * 0.125 = -15.75 -> -63 exactly; x[2] = -15.625 -> -62.5 -> ties-to-even -62
    assert_eq!(b.q[1], -63);
    assert_eq!(b.q[2], -62);
    let qs = qs_from_natural(&b.q);
    assert_eq!(qs[0] as i8 as i32, b.q[0]);
    assert_eq!(qs[1] as i8 as i32, b.q[2]);
    assert_eq!(qs[4] as i8 as i32, b.q[1]);
}

fn out_dir() -> std::path::PathBuf {
    let d = std::env::var("E1DN_OUT").unwrap_or_else(|_| {
        format!(
            "/home/kaden/qcal/release-0.4.5/parity/kernel-runs/e1dn/out-{}",
            std::process::id()
        )
    });
    std::fs::create_dir_all(&d).ok();
    std::path::PathBuf::from(d)
}

#[test]
#[ignore = "needs an exact gfx1201 GPU and the shipped model"]
fn qwen4_moe_down_a8_oracle_hw() {
    let Some(mut gpu) = gpu_gate() else { return };
    load_all(&mut gpu);
    println!(
        "arch={} device={} geo: dn_wpb={} dn_rg={} gr_nw={} gr_rg={}",
        gpu.arch,
        gpu.device_id,
        geo().dn_wpb,
        geo().dn_rg,
        geo().gr_nw,
        geo().gr_rg
    );
    let mut tally = Tally::default();

    // ================= (a) synthetic headers / activations =================
    let mut syn_cells = 0usize;
    for layout in [Layout::Shipped, Layout::Planar] {
        for (zg, seed) in [(false, 0x51u64), (true, 0x52u64)] {
            let experts = syn_experts(layout, seed, zg);
            let st = upload_stack(&mut gpu, layout, &experts, N_LOCAL, true);
            for (rows, invalid, div) in [
                (1usize, false, 1usize),
                (3, true, 1),
                (8, false, 1),
                (8, true, 1),
                (16, true, 1),
                (24, false, 1),
                (48, true, 1),
                (16, true, 2),
                (9, false, 2),
            ] {
                let mut rng =
                    Rng(seed ^ ((rows as u64) << 8) ^ (invalid as u64) ^ ((div as u64) << 20));
                let mut meta = syn_meta(rows, &mut rng, invalid);
                let mut mutated = false;
                if invalid && meta.bound / TILE > 6 {
                    mutated = true;
                    // dead-slot corner cases of the grouped metadata: out-of-range tile id, out-of-range and negative sorted entries
                    meta.tiles[3] = (N_LOCAL + 7) as i32;
                    meta.sorted[20] = (rows * KTOP + 9) as i32;
                    meta.sorted[21] = -5;
                }
                let x_src_rows = rows * KTOP / div;
                let x = gen_x_edge(x_src_rows, &mut rng);
                let label = format!(
                    "syn[{} zgarbage={zg} rows={rows} invalid={invalid} div={div}]",
                    layout.name()
                );
                let case = Case {
                    label,
                    layout,
                    stack: &st,
                    meta,
                    x_row_div: div,
                    x,
                    singleton: !mutated,
                };
                // the indexed ABI has no x_row_div: its X rows are the flat picks
                let use_idx = div == 1;
                run_case(&mut gpu, &case, &mut tally, None, use_idx, true);
                syn_cells += 1;
            }
            free_stack(&mut gpu, st);
        }
    }
    println!(
        "synthetic: {syn_cells} cases, tally so far {}/{} cells pass",
        tally.pass, tally.cells
    );

    // ================= (b) real tensors + real W1 routes =================
    let model = std::env::var("E1DN_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let hfq = open_hfq(Path::new(&model));
    let routes = load_routes();
    let layers: Vec<usize> = std::env::var("E1DN_LAYERS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 17, 47]);
    let rows_list = [1usize, 2, 4, 8, 9, 16, 24, 32, 48, 56, 64];
    let mut quality: HashMap<(usize, &'static str), Quality> = HashMap::new();
    let mut real_cells = 0usize;
    let mut union_info: Vec<String> = Vec::new();
    for &layer in &layers {
        // route variants per rows for this layer
        let metas: Vec<(usize, Meta)> = rows_list
            .iter()
            .map(|&r| (r, build_meta(&route_rows(&routes, layer, r, 0))))
            .collect();
        let mut want: BTreeSet<usize> = BTreeSet::new();
        for (_, m) in &metas {
            for &e in &m.topk {
                if e >= 0 {
                    want.insert(e as usize);
                }
            }
        }
        println!(
            "layer {layer}: {} distinct owned experts over rows {:?}",
            want.len(),
            rows_list
        );
        let mut shipped_out: HashMap<usize, (Option<Vec<u8>>, Option<Vec<u8>>)> = HashMap::new();
        for layout in [Layout::Shipped, Layout::Planar] {
            let experts = real_experts(&hfq, layer, layout, Some(&want));
            let st = upload_stack(&mut gpu, layout, &experts, N_LOCAL, true);
            for (rows, meta) in &metas {
                let mut rng = Rng(0xE1D0_0000 + (layer * 100 + rows) as u64);
                let x = gen_x_gauss(rows * KTOP, &mut rng);
                union_info.push(format!(
                    "L{layer} rows={rows}: distinct owned={} live_rows={} bound={}",
                    meta.distinct, meta.live, meta.bound
                ));
                let case = Case {
                    label: format!("real[L{layer} {} rows={rows}]", layout.name()),
                    layout,
                    stack: &st,
                    meta: meta.clone(),
                    x_row_div: 1,
                    x,
                    singleton: true,
                };
                let q = quality.entry((*rows, layout.name())).or_default();
                let outs = run_case_out(&mut gpu, &case, &mut tally, Some(q));
                real_cells += 1;
                if layout == Layout::Shipped {
                    shipped_out.insert(*rows, outs);
                } else {
                    let (si, sg) = shipped_out.remove(rows).unwrap();
                    let (pi, pg) = outs;
                    let bad = si
                        .as_ref()
                        .zip(pi.as_ref())
                        .map(|(a, b)| count_diff(a, b).0)
                        .unwrap_or(0)
                        + sg.as_ref()
                            .zip(pg.as_ref())
                            .map(|(a, b)| count_diff(a, b).0)
                            .unwrap_or(0);
                    tally.cell(
                        &format!("real[L{layer} rows={rows}] planar==shipped output bytes"),
                        bad,
                        "(words)",
                    );
                }
            }
            // negative controls on layer 17 / the middle layer (ar rows 8 indexed; wide rows 16 grouped)
            if layer == layers[layers.len() / 2] {
                for (rows, meta) in metas.iter().filter(|(r, _)| *r == 8 || *r == 16) {
                    let mut rng = Rng(0xE1D0_0000 + (layer * 100 + rows) as u64);
                    let x = gen_x_gauss(rows * KTOP, &mut rng);
                    let case = Case {
                        label: format!("nc[L{layer} {} rows={rows}]", layout.name()),
                        layout,
                        stack: &st,
                        meta: meta.clone(),
                        x_row_div: 1,
                        x,
                        singleton: true,
                    };
                    for (nm, nc) in [
                        ("wrong-scale-byte", Nc::WrongScaleByte),
                        ("wrong-block-order", Nc::WrongBlockOrder),
                    ] {
                        let (bi, bg) = nc_run(&mut gpu, &case, nc);
                        tally.nc(
                            &format!("L{layer} {} rows={rows} {nm} indexed", layout.name()),
                            bi,
                        );
                        tally.nc(
                            &format!("L{layer} {} rows={rows} {nm} grouped", layout.name()),
                            bg,
                        );
                        println!("negative control L{layer} {} rows={rows} {nm}: indexed {bi} words, grouped {bg} words differ", layout.name());
                    }
                }
            }
            free_stack(&mut gpu, st);
        }
    }

    // ================= summary =================
    println!("\n==== ORACLE SUMMARY ====");
    println!(
        "cells: {} pass {} fail {} | mismatching words {}",
        tally.cells,
        tally.pass,
        tally.cells - tally.pass,
        tally.mismatch_words
    );
    println!(
        "negative controls: {} cells, {} detected, {} differing words",
        tally.nc_cells, tally.nc_detected, tally.nc_words
    );
    println!("synthetic cases {syn_cells}, real cases {real_cells}");
    let mut qjson = String::new();
    let mut qkeys: Vec<_> = quality.keys().cloned().collect();
    qkeys.sort();
    for k in qkeys {
        let q = &quality[&k];
        println!(
            "quality rows={} {}: rel_L2(A8 vs incumbent F32) = {:.6e}  max_abs = {:.4e}  (n = {})",
            k.0,
            k.1,
            q.rel_l2(),
            q.max_abs,
            q.n
        );
        write!(
            qjson,
            "{{\"rows\":{},\"layout\":\"{}\",\"rel_l2\":{:.8e},\"max_abs\":{:.6e},\"n\":{}}},",
            k.0,
            k.1,
            q.rel_l2(),
            q.max_abs,
            q.n
        )
        .ok();
    }
    let json = format!(
        "{{\"model\":\"{}\",\"tensors\":\"model.language_model.layers.{{L}}.mlp.experts.down_proj\",\"layers\":{:?},\"cells\":{},\"pass\":{},\"fail\":{},\"mismatch_words\":{},\"nc_cells\":{},\"nc_detected\":{},\"nc_words\":{},\"synthetic_cases\":{},\"real_cases\":{},\"quality\":[{}],\"failures\":{:?},\"unions\":{:?}}}\n",
        model, layers, tally.cells, tally.pass, tally.cells - tally.pass, tally.mismatch_words, tally.nc_cells, tally.nc_detected, tally.nc_words, syn_cells, real_cells,
        qjson.trim_end_matches(','), tally.lines, union_info
    );
    let od = out_dir();
    std::fs::write(od.join("oracle.json"), &json).expect("write oracle.json");
    println!("summary -> {}", od.join("oracle.json").display());
    assert_eq!(tally.cells, tally.pass, "oracle mismatches");
    assert_eq!(
        tally.nc_cells, tally.nc_detected,
        "a negative control went undetected"
    );
}

// ---------------------------------------------------------------------------------------------
// timing
// ---------------------------------------------------------------------------------------------

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p) as usize]
}

struct TStep {
    layer_ix: usize,
    meta: Meta,
    dm: DevMeta,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Arm {
    Inc,
    A8Incl,
    A8Down,
    A8Prod,
    Stream,
    Null,
    Dni,
    /// WMMA grouped kernel at decode row counts (live tiles; grouped Y layout) - dot4-vs-WMMA measurement
    A8GrpAtDecode,
    /// dot4 per-pick kernel at verify row counts - dot4-vs-WMMA measurement
    A8IdxAtWide,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Inc => "incumbent_f32",
            Arm::A8Incl => "a8_producer+down",
            Arm::A8Down => "a8_down_only",
            Arm::A8Prod => "a8_producer_only",
            Arm::Stream => "roofline_stream",
            Arm::Null => "null_launch_floor",
            Arm::Dni => "lab_dni_exact_f32",
            Arm::A8GrpAtDecode => "a8_down_only[wmma grouped]",
            Arm::A8IdxAtWide => "a8_down_only[dot4 per-pick]",
        }
    }
}

#[test]
#[ignore = "needs an exact gfx1201 GPU and the shipped model; run through the fabric on card C (--class timing)"]
fn qwen4_moe_down_a8_timing_hw() {
    let Some(mut gpu) = gpu_gate() else { return };
    load_all(&mut gpu);
    let model = std::env::var("E1DN_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let hfq = open_hfq(Path::new(&model));
    let routes = load_routes();
    let tlayers: Vec<usize> = std::env::var("E1DN_TLAYERS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 8, 17, 26, 35, 47]);
    assert!(tlayers.len() >= 6, "cold rotation needs >= 6 real layers");
    let rows_all = [1usize, 2, 4, 8, 9, 16, 24, 32, 48, 56, 64];
    let rows_list: Vec<usize> = std::env::var("E1DN_ROWS")
        .ok()
        .map(|v| v.split(',').filter_map(|s| s.parse().ok()).collect())
        .unwrap_or_else(|| rows_all.to_vec());
    let variants = env_usize("E1DN_VARIANTS", 4);
    let passes = env_usize("E1DN_PASSES", 12);
    let layouts: Vec<Layout> = match std::env::var("E1DN_LAYOUTS").ok().as_deref() {
        Some("shipped") => vec![Layout::Shipped],
        Some("planar") => vec![Layout::Planar],
        _ => vec![Layout::Shipped, Layout::Planar],
    };
    println!("arch={} device={} pid={} tlayers={tlayers:?} rows={rows_list:?} variants={variants} passes={passes}", gpu.arch, gpu.device_id, std::process::id());

    // MALL / L2 flush buffer (>= 4x the 64 MiB infinity cache)
    let flush_bytes = 256usize << 20;
    let flush = alloc_raw(&mut gpu, flush_bytes);
    fill_bytes(&gpu, &flush, 0x11);
    let do_flush = |g: &Gpu| {
        let mut blob = KernargBlob::new();
        blob.push_ptr(flush.buf.as_ptr() as *mut c_void);
        blob.push_u64((flush_bytes / 16) as u64);
        blob.pad_to(16);
        g.launch_kernel_blob(F_FLUSH, [4096, 1, 1], [256, 1, 1], 0, blob.as_mut_slice())
            .expect("flush launch");
    };

    let mut cells_json: Vec<String> = Vec::new();
    println!("\nTIMING (event-timed single sequences, cold: >= {} real layers rotated, {} MiB flush before each sample)", tlayers.len(), flush_bytes >> 20);
    println!(
        "{:<8} {:>4} {:<7} {:<18} {:>9} {:>9} {:>9} {:>10}",
        "layout", "rows", "grp", "arm", "us_med", "us_p10", "us_p90", "GBps_wts"
    );
    for &layout in &layouts {
        // all 256 owned experts of every rotated layer resident
        let mut stacks: Vec<Stack> = Vec::new();
        for &layer in &tlayers {
            let experts = real_experts(&hfq, layer, layout, None);
            stacks.push(upload_stack(&mut gpu, layout, &experts, N_LOCAL, false));
        }
        for &rows in &rows_list {
            let grouped = rows >= 9;
            let mut rng = Rng(0xE1D1_0000 + rows as u64);
            let x = gen_x_gauss(rows * KTOP, &mut rng);
            let xt = upload_raw(&mut gpu, &f32b(&x));
            let xq = alloc_raw(&mut gpu, rows * KTOP * ROWQ);
            launch_quant(
                &gpu,
                xt.buf.as_ptr() as *mut c_void,
                xq.buf.as_ptr() as *mut c_void,
                rows * KTOP,
            );
            let mut steps: Vec<TStep> = Vec::new();
            for (li, &layer) in tlayers.iter().enumerate() {
                for v in 0..variants {
                    let meta = build_meta(&route_rows(&routes, layer, rows, v));
                    let dm = upload_meta(&mut gpu, &meta);
                    steps.push(TStep {
                        layer_ix: li,
                        meta,
                        dm,
                    });
                }
            }
            let mean_distinct: f64 =
                steps.iter().map(|s| s.meta.distinct as f64).sum::<f64>() / steps.len() as f64;
            let mean_live: f64 =
                steps.iter().map(|s| s.meta.live as f64).sum::<f64>() / steps.len() as f64;
            let mean_bound: f64 =
                steps.iter().map(|s| s.meta.bound as f64).sum::<f64>() / steps.len() as f64;
            let ysz = steps
                .iter()
                .map(|s| s.meta.bound)
                .max()
                .unwrap()
                .max(rows * KTOP)
                * M;
            let y = alloc_out(&mut gpu, ysz);
            let gmodes: Vec<(&str, bool)> = if grouped {
                vec![("live", false), ("bound", true)]
            } else {
                vec![("-", false)]
            };
            for (gname, use_bound) in gmodes {
                let mut arms = vec![
                    Arm::Inc,
                    Arm::A8Incl,
                    Arm::A8Down,
                    Arm::A8Prod,
                    Arm::Stream,
                    Arm::Null,
                ];
                if !grouped {
                    arms.push(Arm::Dni);
                    arms.push(Arm::A8GrpAtDecode);
                } else {
                    arms.push(Arm::A8IdxAtWide);
                }
                let mut samples: HashMap<Arm, Vec<f64>> =
                    arms.iter().map(|&a| (a, Vec::new())).collect();
                let nsamp = steps.len() * passes;
                // warm-up
                for i in 0..steps.len().min(8) {
                    let s = &steps[i];
                    run_arm(
                        &gpu,
                        Arm::A8Incl,
                        layout,
                        &stacks[s.layer_ix],
                        s,
                        &xt,
                        &xq,
                        &y,
                        rows,
                        grouped,
                        use_bound,
                    );
                    run_arm(
                        &gpu,
                        Arm::Inc,
                        layout,
                        &stacks[s.layer_ix],
                        s,
                        &xt,
                        &xq,
                        &y,
                        rows,
                        grouped,
                        use_bound,
                    );
                }
                gpu.hip.device_synchronize().expect("sync");
                let ea = gpu.hip.event_create().expect("ev");
                let eb = gpu.hip.event_create().expect("ev");
                for i in 0..nsamp {
                    let s = &steps[i % steps.len()];
                    // alternate baseline / candidate inside the hold; rotate the order so no arm owns a slot
                    for k in 0..arms.len() {
                        let arm = arms[(k + i) % arms.len()];
                        do_flush(&gpu);
                        gpu.hip.event_record(&ea, None).expect("rec");
                        run_arm(
                            &gpu,
                            arm,
                            layout,
                            &stacks[s.layer_ix],
                            s,
                            &xt,
                            &xq,
                            &y,
                            rows,
                            grouped,
                            use_bound,
                        );
                        gpu.hip.event_record(&eb, None).expect("rec");
                        gpu.hip.event_synchronize(&eb).expect("sync");
                        let ms = gpu.hip.event_elapsed_ms(&ea, &eb).expect("elapsed") as f64;
                        samples.get_mut(&arm).unwrap().push(ms * 1000.0);
                    }
                }
                gpu.hip.event_destroy(ea).ok();
                gpu.hip.event_destroy(eb).ok();
                for &arm in &arms {
                    let v = &samples[&arm];
                    let med = median(v);
                    let bytes = mean_distinct * layout.per() as f64;
                    let gbps = bytes / (med * 1e-6) / 1e9;
                    println!(
                        "{:<8} {:>4} {:<7} {:<18} {:>9.1} {:>9.1} {:>9.1} {:>10.0}",
                        layout.name(),
                        rows,
                        gname,
                        arm.name(),
                        med,
                        pct(v, 0.1),
                        pct(v, 0.9),
                        gbps
                    );
                    cells_json.push(format!(
                        "{{\"layout\":\"{}\",\"rows\":{},\"grouped_rows_mode\":\"{}\",\"arm\":\"{}\",\"us_med\":{:.3},\"us_p10\":{:.3},\"us_p90\":{:.3},\"weights_gbps\":{:.1},\"mean_distinct_experts\":{:.2},\"mean_live_rows\":{:.1},\"mean_bound_rows\":{:.1},\"samples\":{}}}",
                        layout.name(), rows, gname, arm.name(), med, pct(v, 0.1), pct(v, 0.9), gbps, mean_distinct, mean_live, mean_bound, v.len()
                    ));
                }
                let inc = median(&samples[&Arm::Inc]);
                let incl = median(&samples[&Arm::A8Incl]);
                println!("   -> {} rows={} {}: a8 producer+down {:.1} us vs incumbent {:.1} us ({:+.1}%)  [mean distinct experts {:.1}]", layout.name(), rows, gname, incl, inc, (incl / inc - 1.0) * 100.0, mean_distinct);
            }
            for s in steps {
                free_meta(&mut gpu, s.dm);
            }
            let _ = gpu.free_tensor(xt);
            let _ = gpu.free_tensor(xq);
            let _ = gpu.free_tensor(y.t);
        }
        for st in stacks {
            free_stack(&mut gpu, st);
        }
    }
    let json = format!(
        "{{\"pid\":{},\"model\":\"{}\",\"layers\":{:?},\"cells\":[{}]}}\n",
        std::process::id(),
        model,
        tlayers,
        cells_json.join(",")
    );
    let od = out_dir();
    std::fs::write(od.join("timing.json"), json).expect("write timing.json");
    println!("timing summary -> {}", od.join("timing.json").display());
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    gpu: &Gpu,
    arm: Arm,
    l: Layout,
    st: &Stack,
    s: &TStep,
    xt: &GpuTensor,
    xq: &GpuTensor,
    y: &OutBuf,
    rows: usize,
    grouped: bool,
    use_bound: bool,
) {
    if arm == Arm::Stream {
        let mut b = KernargBlob::new();
        b.push_ptr(st.table.buf.as_ptr() as *mut c_void);
        b.push_ptr(s.dm.tiles.buf.as_ptr() as *mut c_void);
        b.push_u64((l.per() / 16) as u64);
        b.push_ptr(out_ptr(y));
        b.push_i32(N_LOCAL as i32);
        b.pad_to(16);
        gpu.launch_kernel_blob(
            F_STREAM,
            [128, (s.meta.live / TILE) as u32, 1],
            [256, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .expect("stream");
        return;
    }
    if arm == Arm::A8GrpAtDecode {
        launch_a8_grp(
            gpu,
            l,
            st.table.buf.as_ptr() as *mut c_void,
            s.dm.tiles.buf.as_ptr() as *mut c_void,
            s.dm.sorted.buf.as_ptr() as *mut c_void,
            xq.buf.as_ptr() as *mut c_void,
            out_ptr(y),
            1,
            s.meta.live,
            rows * KTOP,
        );
        return;
    }
    if arm == Arm::A8IdxAtWide {
        launch_a8_idx(
            gpu,
            l,
            st.table.buf.as_ptr() as *mut c_void,
            s.dm.topk.buf.as_ptr() as *mut c_void,
            xq.buf.as_ptr() as *mut c_void,
            out_ptr(y),
            rows,
        );
        return;
    }
    if arm == Arm::Dni {
        let mut b = idx_blob(
            st.table.buf.as_ptr() as *mut c_void,
            s.dm.topk.buf.as_ptr() as *mut c_void,
            xt.buf.as_ptr() as *mut c_void,
            out_ptr(y),
            rows,
        );
        let f = if l == Layout::Planar {
            F_DNI_P
        } else {
            F_DNI_S
        };
        gpu.launch_kernel_blob(
            f,
            [(M / 32) as u32, KTOP as u32, rows as u32],
            [128, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .expect("dni");
        return;
    }
    if arm == Arm::Null {
        let mut b = KernargBlob::new();
        b.push_ptr(xt.buf.as_ptr() as *mut c_void);
        b.push_ptr(xq.buf.as_ptr() as *mut c_void);
        b.push_i32(K as i32);
        b.push_i32(0);
        b.pad_to(16);
        gpu.launch_kernel_blob(
            F_QUANT,
            [1, 1, 1],
            [(K / 128 * 32) as u32, 1, 1],
            0,
            b.as_mut_slice(),
        )
        .expect("null");
        return;
    }
    let table = st.table.buf.as_ptr() as *mut c_void;
    let topk = s.dm.topk.buf.as_ptr() as *mut c_void;
    let tiles = s.dm.tiles.buf.as_ptr() as *mut c_void;
    let sorted = s.dm.sorted.buf.as_ptr() as *mut c_void;
    let xp = xt.buf.as_ptr() as *mut c_void;
    let xqp = xq.buf.as_ptr() as *mut c_void;
    let yp = out_ptr(y);
    let gr = if use_bound { s.meta.bound } else { s.meta.live };
    match arm {
        Arm::Inc => {
            if grouped {
                launch_inc_grp(gpu, l, table, tiles, sorted, xp, yp, 1, gr, rows * KTOP);
            } else {
                launch_inc_idx(gpu, l, table, topk, xp, yp, rows);
            }
        }
        Arm::A8Prod => launch_quant(gpu, xp, xqp, rows * KTOP),
        Arm::Stream | Arm::Null | Arm::Dni | Arm::A8GrpAtDecode | Arm::A8IdxAtWide => {
            unreachable!()
        }
        Arm::A8Down | Arm::A8Incl => {
            if arm == Arm::A8Incl {
                launch_quant(gpu, xp, xqp, rows * KTOP);
            }
            if grouped {
                launch_a8_grp(gpu, l, table, tiles, sorted, xqp, yp, 1, gr, rows * KTOP);
            } else {
                launch_a8_idx(gpu, l, table, topk, xqp, yp, rows);
            }
        }
    }
}
