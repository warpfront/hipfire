//! Throwaway Gate F1 FA2 oracle (repaired positions views).
//!
//! Historical bug (commit 203864992 oracle): `upload_raw` + shape `&[N]` made
//! positions `DType::Raw` (1-byte elements), then `sub_offset(HALF, HALF)`
//! advanced only 512 **bytes** instead of 512 i32s. Rebuild uses
//! `upload_raw(position_bytes, &[4*N])` and `sub_offset(4*row_start, 4*row_count)`.
//!
//! Fixtures (batch, start, end=L):
//!   N≤512 old-vs-new (same accepted FA2, bit-exact self / dual-launch):
//!     (1,0,1),(7,26,33),(64,0,64),(128,384,512),
//!     (512,0,512),(512,7680,8192),(512,32256,32768)
//!   N1024 one launch vs two N512 halves on identical complete KV:
//!     (1024,0,1024),(1024,7168,8192),(1024,31744,32768)
//! Both K modes (Q8 / fwht3). fwht3 also matches post-launch Q bits.
//!
//! Host asserts every position view's byte pointer delta + accessible length,
//! downloads i32s through the actual view, and checks each equals
//! `start+row_start+b`. Tiny `tmp_fa2_copy_positions` kernel reads
//! `const int*` exactly as FA2 does; negative control detects the historical
//! wrong view on the host BEFORE any undersized launch.
//!
//! TIME=1: interleaved host timings; fwht3 refreshes Q outside the timed
//! interval every iteration.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f1 \
//!     cargo run --release -p hipfire-runtime --example tmp_fa2_n1024_oracle --features lab
//!   TIME=1 … (same)   # interleaved timings
//!
//! Direct launchers: gfx1151 lab cap 1024 (H); production ingress stays 512.

use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};
use std::path::PathBuf;
use std::time::Instant;

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32; // 8
const Q8_BLOCK: usize = 34;
const ROW_STRIDE: usize = N_KV * BPH * Q8_BLOCK; // 1088
const FWHT3_HEAD: usize = 100;
const FWHT3_POS: usize = N_KV * FWHT3_HEAD; // 400
const QO_ROW: usize = N_HEADS * HD; // 6144
const HALF: usize = 512;

/// N≤512 fixtures: (batch, start_position, max_position+1).
const SHAPES_LE512: &[(usize, usize, usize)] = &[
    (1, 0, 1),
    (7, 26, 33),
    (64, 0, 64),
    (128, 384, 512),
    (512, 0, 512),
    (512, 7680, 8192),
    (512, 32256, 32768),
];

/// N=1024 fixtures: one full launch vs two N=512 halves.
const SHAPES_N1024: &[(usize, usize, usize)] = &[
    (1024, 0, 1024),
    (1024, 7168, 8192),
    (1024, 31744, 32768),
];

const COPY_POS_SRC: &str = r#"
#include <hip/hip_runtime.h>

// Debug twin of FA2's `const int* positions` load. ≤32 VGPR, zero LDS/spill.
extern "C" __global__ __launch_bounds__(64, 1) void tmp_fa2_copy_positions(
    const int* __restrict__ positions,
    int* __restrict__ echo,
    int n)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        echo[i] = positions[i];
    }
}
"#;

const COPY_POS_NAME: &str = "tmp_fa2_copy_positions";

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

fn prng_f32(i: u64, salt: u32) -> f32 {
    (prng_u32(i, salt) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7c00 | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return (sign << 15) | 0x7c00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}

fn pack_q8_block(scale: f32, codes: &[i8; 32]) -> [u8; 34] {
    let mut out = [0u8; 34];
    let s = f32_to_f16_bits(scale);
    out[0] = (s & 0xff) as u8;
    out[1] = (s >> 8) as u8;
    for w in 0..8 {
        let mut u = 0u32;
        for c in 0..4 {
            let v = codes[w * 4 + c] as u8;
            u |= (v as u32) << (c * 8);
        }
        out[2 + w * 4..2 + w * 4 + 4].copy_from_slice(&u.to_le_bytes());
    }
    out
}

fn fill_q8_cache(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * ROW_STRIDE];
    for g in 0..seq_len {
        for kv in 0..N_KV {
            for b in 0..BPH {
                let mut codes = [0i8; 32];
                for c in 0..32 {
                    let r = prng_u32((g * 64 + kv * 8 + b) as u64 * 32 + c as u64, salt);
                    codes[c] = ((r % 61) as i8).wrapping_sub(30);
                }
                let scale =
                    0.01 + (prng_u32((g * 32 + kv * 8 + b) as u64, salt ^ 0xabc) % 100) as f32 * 0.001;
                let blk = pack_q8_block(scale, &codes);
                let off = g * ROW_STRIDE + (kv * BPH + b) * Q8_BLOCK;
                buf[off..off + 34].copy_from_slice(&blk);
            }
        }
    }
    buf
}

fn fill_fwht3_k(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * FWHT3_POS];
    for g in 0..seq_len {
        for kv in 0..N_KV {
            let base = g * FWHT3_POS + kv * FWHT3_HEAD;
            let cnorm = 0.05 + (prng_u32((g * 4 + kv) as u64, salt) % 200) as f32 * 0.001;
            buf[base..base + 4].copy_from_slice(&cnorm.to_le_bytes());
            // 96 B of 3-bit codes: 32 groups of 3 B = 32 * 8 codes = 256 dims.
            for g3 in 0..32 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let code = (prng_u32((g * 256 + kv * 64 + g3 * 8 + i) as u64, salt) % 8) as u32;
                    packed |= code << (3 * i);
                }
                let bytes = packed.to_le_bytes();
                buf[base + 4 + g3 * 3..base + 4 + g3 * 3 + 3].copy_from_slice(&bytes[..3]);
            }
        }
    }
    buf
}

fn fill_q(batch: usize, salt: u32) -> Vec<f32> {
    let n = batch * N_HEADS * HD;
    (0..n)
        .map(|i| prng_f32(i as u64, salt) * 0.5)
        .collect()
}

fn fill_positions(batch: usize, start: usize) -> Vec<i32> {
    (0..batch).map(|b| (start + b) as i32).collect()
}

fn positions_to_bytes(pos: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pos.len() * 4);
    for &p in pos {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    bytes
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn bitwise_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn bytes_equal(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// Upload positions as Raw **bytes** with shape `[4*N]` (not `[N]`).
fn upload_positions(gpu: &mut Gpu, pos: &[i32]) -> GpuTensor {
    let bytes = positions_to_bytes(pos);
    let n_bytes = bytes.len();
    assert_eq!(n_bytes, pos.len() * 4);
    gpu.upload_raw(&bytes, &[n_bytes])
        .unwrap_or_else(|e| panic!("upload_positions: {e:?}"))
}

/// Correct Raw byte view: offset/len in **bytes** (= 4 * i32 count).
fn pos_view(pos: &GpuTensor, row_start: usize, row_count: usize) -> GpuTensor {
    assert_eq!(pos.dtype, DType::Raw, "positions must be DType::Raw");
    pos.sub_offset(4 * row_start, 4 * row_count)
}

/// Historical wrong view: sub_offset(HALF, HALF) on Raw → 512 **bytes**.
fn pos_view_historical_wrong(pos: &GpuTensor, half: usize) -> GpuTensor {
    assert_eq!(pos.dtype, DType::Raw);
    pos.sub_offset(half, half)
}

fn ptr_delta_bytes(base: &GpuTensor, view: &GpuTensor) -> usize {
    let b = base.buf.as_ptr() as usize;
    let v = view.buf.as_ptr() as usize;
    assert!(v >= b, "view pointer before base");
    v - b
}

fn download_i32_view(gpu: &Gpu, view: &GpuTensor) -> Vec<i32> {
    let nbytes = view.buf.size();
    assert_eq!(nbytes % 4, 0, "i32 view byte size not multiple of 4");
    let n = nbytes / 4;
    let mut data = vec![0i32; n];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, nbytes) };
    gpu.hip
        .memcpy_dtoh(bytes, &view.buf)
        .unwrap_or_else(|e| panic!("download_i32_view: {e:?}"));
    data
}

/// Assert host view geometry + every i32 through the view equals start+row_start+b.
fn assert_pos_view(
    gpu: &Gpu,
    base: &GpuTensor,
    view: &GpuTensor,
    start: usize,
    row_start: usize,
    row_count: usize,
    tag: &str,
) {
    let delta = ptr_delta_bytes(base, view);
    let expect_delta = 4 * row_start;
    let expect_len = 4 * row_count;
    assert_eq!(
        delta, expect_delta,
        "{tag}: ptr delta {delta} != {expect_delta} (4*row_start)"
    );
    assert_eq!(
        view.buf.size(),
        expect_len,
        "{tag}: accessible len {} != {expect_len}",
        view.buf.size()
    );
    assert_eq!(view.dtype, DType::Raw, "{tag}: dtype");
    // shape is element count in Raw (= bytes)
    assert_eq!(view.numel(), expect_len, "{tag}: numel (raw bytes)");

    let got = download_i32_view(gpu, view);
    assert_eq!(got.len(), row_count, "{tag}: downloaded i32 count");
    for (b, &v) in got.iter().enumerate() {
        let expect = (start + row_start + b) as i32;
        assert_eq!(
            v, expect,
            "{tag}: view[{b}]={v} != start+row_start+b={expect}"
        );
    }
}

fn ensure_copy_positions(gpu: &mut Gpu) {
    gpu.ensure_kernel_public(COPY_POS_NAME, COPY_POS_SRC, COPY_POS_NAME)
        .unwrap_or_else(|e| panic!("compile {COPY_POS_NAME}: {e:?}"));
}

fn gate_copy_positions_metadata(arch: &str) {
    // Parent also reads radiowave after first JIT; local hard gate before launches.
    let cache = std::env::var_os("HIPFIRE_KERNEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into());
            PathBuf::from(home).join(".hipfire_kernels")
        });
    let dir = cache.join(arch);
    // Hot cache names are `{name}.{hash}.radiowave.json` or `{name}.radiowave.json`.
    let mut found = None;
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let name = ent.file_name();
            let s = name.to_string_lossy();
            if s.starts_with(COPY_POS_NAME) && s.ends_with(".radiowave.json") {
                found = Some(ent.path());
                break;
            }
        }
    }
    let Some(path) = found else {
        eprintln!(
            "WARN: no radiowave.json for {COPY_POS_NAME} under {} — metadata deferred to parent",
            dir.display()
        );
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read radiowave");
    // Minimal parse without pulling serde into the example.
    let vgpr = json_u32_field(&text, "vgpr_count").unwrap_or(u32::MAX);
    let vgpr_spill = json_u32_field(&text, "vgpr_spill_count").unwrap_or(u32::MAX);
    let sgpr_spill = json_u32_field(&text, "sgpr_spill_count").unwrap_or(u32::MAX);
    let scratch = json_u32_field(&text, "private_segment_fixed_size").unwrap_or(u32::MAX);
    // group_segment may be absent on KernelReport; treat missing as 0 for this no-LDS kernel.
    let group = json_u32_field(&text, "group_segment_fixed_size").unwrap_or(0);
    eprintln!(
        "META {COPY_POS_NAME} vgpr={vgpr} spill_v={vgpr_spill} spill_s={sgpr_spill} \
         scratch={scratch} group={group}  ({})",
        path.display()
    );
    assert!(
        vgpr <= 32,
        "{COPY_POS_NAME} vgpr_count={vgpr} exceeds ceiling 32"
    );
    assert_eq!(vgpr_spill, 0, "{COPY_POS_NAME} vgpr spill");
    assert_eq!(sgpr_spill, 0, "{COPY_POS_NAME} sgpr spill");
    assert_eq!(scratch, 0, "{COPY_POS_NAME} scratch");
    assert_eq!(group, 0, "{COPY_POS_NAME} group_segment");
}

fn json_u32_field(text: &str, key: &str) -> Option<u32> {
    // First occurrence in the primary kernel report (schema has one kernels[0]).
    let pat = format!("\"{key}\"");
    let idx = text.find(&pat)?;
    let rest = &text[idx + pat.len()..];
    let colon = rest.find(':')?;
    let s = rest[colon + 1..].trim_start();
    // strip trailing comma/brace
    let end = s
        .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
        .unwrap_or(s.len());
    s[..end].parse().ok()
}

fn launch_copy_positions(gpu: &mut Gpu, pos: &GpuTensor, echo: &GpuTensor, n: i32) {
    let mut pos_ptr = pos.buf.as_ptr();
    let mut echo_ptr = echo.buf.as_ptr();
    let mut nn = n;
    let mut blob = hip_bridge::KernargBlob::new();
    blob.push_ptr(pos_ptr);
    blob.push_ptr(echo_ptr);
    blob.push_i32(nn);
    let _ = (&mut pos_ptr, &mut echo_ptr, &mut nn); // keep live across blob build
    let grid = ((n as u32) + 63) / 64;
    gpu.launch_kernel_blob(COPY_POS_NAME, [grid.max(1), 1, 1], [64, 1, 1], 0, blob.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {COPY_POS_NAME}: {e:?}"));
    gpu.hip.device_synchronize().unwrap();
}

/// Positive: correct view → echo equals expected i32s.
/// Negative: historical wrong view fails host geometry/value checks (no launch).
fn run_position_contract(
    gpu: &mut Gpu,
    start: usize,
    batch: usize,
    tag: &str,
) {
    let pos_h = fill_positions(batch, start);
    let d_pos = upload_positions(gpu, &pos_h);
    assert_eq!(d_pos.buf.size(), batch * 4);
    assert_eq!(d_pos.numel(), batch * 4);

    // Full view
    let full = pos_view(&d_pos, 0, batch);
    assert_pos_view(gpu, &d_pos, &full, start, 0, batch, &format!("{tag}/full"));

    if batch >= HALF * 2 {
        let p0 = pos_view(&d_pos, 0, HALF);
        let p1 = pos_view(&d_pos, HALF, HALF);
        assert_pos_view(gpu, &d_pos, &p0, start, 0, HALF, &format!("{tag}/p0"));
        assert_pos_view(gpu, &d_pos, &p1, start, HALF, HALF, &format!("{tag}/p1"));

        // Negative control: historical wrong second-half view — host only.
        let wrong = pos_view_historical_wrong(&d_pos, HALF);
        let wdelta = ptr_delta_bytes(&d_pos, &wrong);
        let wlen = wrong.buf.size();
        let wrong_vals = download_i32_view(gpu, &wrong);
        let expect_delta = 4 * HALF;
        let expect_len = 4 * HALF;
        let geometry_bad = wdelta != expect_delta || wlen != expect_len;
        let mut values_bad = wrong_vals.len() != HALF;
        if !values_bad {
            for (b, &v) in wrong_vals.iter().enumerate() {
                let expect = (start + HALF + b) as i32;
                if v != expect {
                    values_bad = true;
                    break;
                }
            }
        }
        assert!(
            geometry_bad || values_bad,
            "{tag}: negative control FAILED to detect historical wrong view \
             (delta={wdelta} want {expect_delta}, len={wlen} want {expect_len}, \
              values_bad={values_bad})"
        );
        eprintln!(
            "NEG  {tag:>28}  historical wrong view detected on host \
             (delta={wdelta} len={wlen} n_i32={} first={:?})",
            wrong_vals.len(),
            wrong_vals.first()
        );
        // Do NOT launch FA2 or copy-positions on the undersized/wrong view.
    }

    // Kernel-read contract on the correct full view.
    let d_echo = gpu
        .zeros(&[batch * 4], DType::Raw)
        .unwrap_or_else(|e| panic!("echo alloc: {e:?}"));
    launch_copy_positions(gpu, &full, &d_echo, batch as i32);
    let echo = download_i32_view(gpu, &d_echo);
    assert!(echo.len() >= batch, "echo alloc smaller than batch");
    let echo = &echo[..batch];
    for (b, &v) in echo.iter().enumerate() {
        let expect = (start + b) as i32;
        assert_eq!(v, expect, "{tag}: copy_positions[{b}]={v} != {expect}");
    }
}

fn run_q8(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    pos: &GpuTensor,
    batch: usize,
    seq_len: usize,
) {
    gpu.attention_q8_0_fa2_gqa_gfx11(q, k, v, out, pos, N_HEADS, N_KV, HD, seq_len, batch)
        .unwrap_or_else(|e| panic!("q8 fa2 launch batch={batch}: {e:?}"));
    gpu.hip.device_synchronize().unwrap();
}

fn run_fwht3(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    out: &GpuTensor,
    pos: &GpuTensor,
    s1: &GpuTensor,
    s2: &GpuTensor,
    batch: usize,
    seq_len: usize,
) {
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx11(
        q, k, v, out, pos, s1, s2, N_HEADS, N_KV, HD, seq_len, batch,
    )
    .unwrap_or_else(|e| panic!("fwht3 fa2 launch batch={batch}: {e:?}"));
    gpu.hip.device_synchronize().unwrap();
}

fn median_us(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn check_guards_raw(gpu: &Gpu, t: &GpuTensor, expect: &[u8], tag: &str) {
    let mut got = vec![0u8; expect.len()];
    // Download only the expect prefix (tensor may be larger Raw shape).
    let view = t.sub_offset(0, expect.len());
    gpu.hip
        .memcpy_dtoh(&mut got, &view.buf)
        .unwrap_or_else(|e| panic!("guard download {tag}: {e:?}"));
    assert!(
        bytes_equal(&got, expect),
        "{tag}: guard bytes changed after launch"
    );
}

fn fixture_le512_q8(
    gpu: &mut Gpu,
    batch: usize,
    start: usize,
    seq_len: usize,
    time: bool,
    n_fail: &mut usize,
) {
    let tag = format!("b{batch}_s{start}_L{seq_len}");
    let pos_h = fill_positions(batch, start);
    let q_h = fill_q(batch, 0x1111);
    let k_h = fill_q8_cache(seq_len, 0x2222);
    let v_h = fill_q8_cache(seq_len, 0x3333);

    let d_k = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
    let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_q = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_pos = upload_positions(gpu, &pos_h);
    let d_out = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();

    let pos_full = pos_view(&d_pos, 0, batch);
    assert_pos_view(gpu, &d_pos, &pos_full, start, 0, batch, &format!("Q8/{tag}/pos"));

    run_q8(gpu, &d_q, &d_k, &d_v, &d_out, &pos_full, batch, seq_len);
    let o_a = gpu.download_f32(&d_out).unwrap();

    // Guards: K/V/positions unchanged.
    check_guards_raw(gpu, &d_k, &k_h, &format!("Q8/{tag}/K"));
    check_guards_raw(gpu, &d_v, &v_h, &format!("Q8/{tag}/V"));
    check_guards_raw(gpu, &d_pos, &positions_to_bytes(&pos_h), &format!("Q8/{tag}/pos"));

    // Second launch (old-vs-new at N≤512: accepted FA2 is bit-exact baseline).
    let d_out2 = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
    let d_q2 = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    run_q8(gpu, &d_q2, &d_k, &d_v, &d_out2, &pos_full, batch, seq_len);
    let o_b = gpu.download_f32(&d_out2).unwrap();
    let mad = max_abs_diff(&o_a, &o_b);
    let beq = bitwise_equal(&o_a, &o_b);
    if !beq {
        *n_fail += 1;
    }
    eprintln!("Q8  {tag:>28}  max_abs={mad:.6e}  bitwise_eq={beq}  (dual-launch)");

    if time {
        const IT: usize = 10;
        let mut us = Vec::with_capacity(IT);
        for _ in 0..IT {
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            run_q8(gpu, &d_q, &d_k, &d_v, &d_out, &pos_full, batch, seq_len);
            us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!(
            "TIME Q8  {tag:>24}  med={:.1}us",
            median_us(&mut us)
        );
    }
}

fn fixture_le512_fwht3(
    gpu: &mut Gpu,
    d_s1: &GpuTensor,
    d_s2: &GpuTensor,
    batch: usize,
    start: usize,
    seq_len: usize,
    time: bool,
    n_fail: &mut usize,
) {
    let tag = format!("b{batch}_s{start}_L{seq_len}");
    let pos_h = fill_positions(batch, start);
    let q_h = fill_q(batch, 0x4444);
    let k_h = fill_fwht3_k(seq_len, 0x5555);
    let v_h = fill_q8_cache(seq_len, 0x6666);

    let d_k = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
    let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_pos = upload_positions(gpu, &pos_h);
    let pos_full = pos_view(&d_pos, 0, batch);
    assert_pos_view(gpu, &d_pos, &pos_full, start, 0, batch, &format!("F3/{tag}/pos"));

    let d_q_a = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_out_a = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
    run_fwht3(
        gpu, &d_q_a, &d_k, &d_v, &d_out_a, &pos_full, d_s1, d_s2, batch, seq_len,
    );
    let o_a = gpu.download_f32(&d_out_a).unwrap();
    let q_a = gpu.download_f32(&d_q_a).unwrap();

    check_guards_raw(gpu, &d_k, &k_h, &format!("F3/{tag}/K"));
    check_guards_raw(gpu, &d_v, &v_h, &format!("F3/{tag}/V"));
    check_guards_raw(gpu, &d_pos, &positions_to_bytes(&pos_h), &format!("F3/{tag}/pos"));

    // Second launch with fresh Q (in-place rotate).
    let d_q_b = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_out_b = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
    run_fwht3(
        gpu, &d_q_b, &d_k, &d_v, &d_out_b, &pos_full, d_s1, d_s2, batch, seq_len,
    );
    let o_b = gpu.download_f32(&d_out_b).unwrap();
    let q_b = gpu.download_f32(&d_q_b).unwrap();

    let mad = max_abs_diff(&o_a, &o_b);
    let beq = bitwise_equal(&o_a, &o_b);
    let q_beq = bitwise_equal(&q_a, &q_b);
    if !beq || !q_beq {
        *n_fail += 1;
    }
    eprintln!(
        "F3  {tag:>28}  max_abs={mad:.6e}  bitwise_eq={beq}  q_post_eq={q_beq}  (dual-launch)"
    );

    if time {
        const IT: usize = 5;
        let mut us = Vec::with_capacity(IT);
        for _ in 0..IT {
            // Refresh Q outside the timed interval (prologue mutates once).
            let qa = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            run_fwht3(
                gpu, &qa, &d_k, &d_v, &d_out_a, &pos_full, d_s1, d_s2, batch, seq_len,
            );
            us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!(
            "TIME F3  {tag:>24}  med={:.1}us",
            median_us(&mut us)
        );
    }
}

fn fixture_n1024_q8(
    gpu: &mut Gpu,
    batch: usize,
    start: usize,
    seq_len: usize,
    time: bool,
    n_fail: &mut usize,
) {
    assert_eq!(batch, 1024);
    let tag = format!("b{batch}_s{start}_L{seq_len}");
    let pos_h = fill_positions(batch, start);
    let q_h = fill_q(batch, 0x1111);
    let k_h = fill_q8_cache(seq_len, 0x2222);
    let v_h = fill_q8_cache(seq_len, 0x3333);

    // Position contract + negative control once per N1024 shape.
    run_position_contract(gpu, start, batch, &format!("Q8/{tag}"));

    // (a) one N=1024
    let d_k_a = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
    let d_v_a = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_q_a = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_pos_a = upload_positions(gpu, &pos_h);
    let d_out_a = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
    let pos_a = pos_view(&d_pos_a, 0, batch);
    assert_pos_view(gpu, &d_pos_a, &pos_a, start, 0, batch, &format!("Q8/{tag}/a"));
    run_q8(gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &pos_a, batch, seq_len);
    let o_a = gpu.download_f32(&d_out_a).unwrap();
    check_guards_raw(gpu, &d_k_a, &k_h, &format!("Q8/{tag}/a/K"));
    check_guards_raw(gpu, &d_v_a, &v_h, &format!("Q8/{tag}/a/V"));
    check_guards_raw(
        gpu,
        &d_pos_a,
        &positions_to_bytes(&pos_h),
        &format!("Q8/{tag}/a/pos"),
    );

    // (b) two N=512 on identical complete KV, separately cloned Q/out
    let d_k_b = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
    let d_v_b = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_q_b = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_pos_b = upload_positions(gpu, &pos_h);
    let d_out_b = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();

    let q0 = d_q_b.sub_offset(0, HALF * QO_ROW);
    let q1 = d_q_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
    let o0 = d_out_b.sub_offset(0, HALF * QO_ROW);
    let o1 = d_out_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
    let p0 = pos_view(&d_pos_b, 0, HALF);
    let p1 = pos_view(&d_pos_b, HALF, HALF);
    assert_pos_view(gpu, &d_pos_b, &p0, start, 0, HALF, &format!("Q8/{tag}/b0"));
    assert_pos_view(gpu, &d_pos_b, &p1, start, HALF, HALF, &format!("Q8/{tag}/b1"));

    run_q8(gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, HALF, seq_len);
    run_q8(gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, HALF, seq_len);
    let o_b = gpu.download_f32(&d_out_b).unwrap();

    let mad = max_abs_diff(&o_a, &o_b);
    let beq = bitwise_equal(&o_a, &o_b);
    if !beq {
        *n_fail += 1;
    }
    eprintln!("Q8  {tag:>28}  max_abs={mad:.6e}  bitwise_eq={beq}  (1x1024 vs 2x512)");

    if time {
        // Warm
        run_q8(gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &pos_a, batch, seq_len);
        run_q8(gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, HALF, seq_len);
        run_q8(gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, HALF, seq_len);
        const IT: usize = 10;
        let mut us_a = Vec::with_capacity(IT);
        let mut us_b = Vec::with_capacity(IT);
        for _ in 0..IT {
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            run_q8(gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &pos_a, batch, seq_len);
            us_a.push(t0.elapsed().as_secs_f64() * 1e6);

            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            run_q8(gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, HALF, seq_len);
            run_q8(gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, HALF, seq_len);
            us_b.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!(
            "TIME Q8  {tag:>24}  n1024_med={:.1}us  two512_sum_med={:.1}us",
            median_us(&mut us_a),
            median_us(&mut us_b)
        );
    }
}

fn fixture_n1024_fwht3(
    gpu: &mut Gpu,
    d_s1: &GpuTensor,
    d_s2: &GpuTensor,
    batch: usize,
    start: usize,
    seq_len: usize,
    time: bool,
    n_fail: &mut usize,
) {
    assert_eq!(batch, 1024);
    let tag = format!("b{batch}_s{start}_L{seq_len}");
    let pos_h = fill_positions(batch, start);
    let q_h = fill_q(batch, 0x4444);
    let k_h = fill_fwht3_k(seq_len, 0x5555);
    let v_h = fill_q8_cache(seq_len, 0x6666);

    // (a) N=1024 with dedicated Q clone
    let d_k_a = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
    let d_v_a = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_q_a = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_pos_a = upload_positions(gpu, &pos_h);
    let d_out_a = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();
    let pos_a = pos_view(&d_pos_a, 0, batch);
    assert_pos_view(gpu, &d_pos_a, &pos_a, start, 0, batch, &format!("F3/{tag}/a"));
    run_fwht3(
        gpu, &d_q_a, &d_k_a, &d_v_a, &d_out_a, &pos_a, d_s1, d_s2, batch, seq_len,
    );
    let o_a = gpu.download_f32(&d_out_a).unwrap();
    let q_a = gpu.download_f32(&d_q_a).unwrap();
    check_guards_raw(gpu, &d_k_a, &k_h, &format!("F3/{tag}/a/K"));
    check_guards_raw(gpu, &d_v_a, &v_h, &format!("F3/{tag}/a/V"));
    check_guards_raw(
        gpu,
        &d_pos_a,
        &positions_to_bytes(&pos_h),
        &format!("F3/{tag}/a/pos"),
    );

    // (b) two N=512 — fresh KV + fresh Q, write into full out/q
    let d_k_b = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
    let d_v_b = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_q_b = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
    let d_pos_b = upload_positions(gpu, &pos_h);
    let d_out_b = gpu.zeros(&[batch * QO_ROW], DType::F32).unwrap();

    let q0 = d_q_b.sub_offset(0, HALF * QO_ROW);
    let q1 = d_q_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
    let o0 = d_out_b.sub_offset(0, HALF * QO_ROW);
    let o1 = d_out_b.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
    let p0 = pos_view(&d_pos_b, 0, HALF);
    let p1 = pos_view(&d_pos_b, HALF, HALF);
    assert_pos_view(gpu, &d_pos_b, &p0, start, 0, HALF, &format!("F3/{tag}/b0"));
    assert_pos_view(gpu, &d_pos_b, &p1, start, HALF, HALF, &format!("F3/{tag}/b1"));

    run_fwht3(
        gpu, &q0, &d_k_b, &d_v_b, &o0, &p0, d_s1, d_s2, HALF, seq_len,
    );
    run_fwht3(
        gpu, &q1, &d_k_b, &d_v_b, &o1, &p1, d_s1, d_s2, HALF, seq_len,
    );
    let o_b = gpu.download_f32(&d_out_b).unwrap();
    let q_b = gpu.download_f32(&d_q_b).unwrap();

    let mad = max_abs_diff(&o_a, &o_b);
    let beq = bitwise_equal(&o_a, &o_b);
    let q_beq = bitwise_equal(&q_a, &q_b);
    if !beq || !q_beq {
        *n_fail += 1;
    }
    eprintln!(
        "F3  {tag:>28}  max_abs={mad:.6e}  bitwise_eq={beq}  q_post_eq={q_beq}  (1x1024 vs 2x512)"
    );

    if time {
        const IT: usize = 5;
        let mut us_a = Vec::with_capacity(IT);
        let mut us_b = Vec::with_capacity(IT);
        for _ in 0..IT {
            // Fresh Q each timed launch; upload OUTSIDE the timed interval.
            let qa = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            run_fwht3(
                gpu, &qa, &d_k_a, &d_v_a, &d_out_a, &pos_a, d_s1, d_s2, batch, seq_len,
            );
            us_a.push(t0.elapsed().as_secs_f64() * 1e6);

            let qb = gpu.upload_f32(&q_h, &[batch, N_HEADS, HD]).unwrap();
            let qb0 = qb.sub_offset(0, HALF * QO_ROW);
            let qb1 = qb.sub_offset(HALF * QO_ROW, HALF * QO_ROW);
            gpu.hip.device_synchronize().unwrap();
            let t0 = Instant::now();
            run_fwht3(
                gpu, &qb0, &d_k_b, &d_v_b, &o0, &p0, d_s1, d_s2, HALF, seq_len,
            );
            run_fwht3(
                gpu, &qb1, &d_k_b, &d_v_b, &o1, &p1, d_s1, d_s2, HALF, seq_len,
            );
            us_b.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        eprintln!(
            "TIME F3  {tag:>24}  n1024_med={:.1}us  two512_sum_med={:.1}us",
            median_us(&mut us_a),
            median_us(&mut us_b)
        );
    }
}

fn main() {
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "tmp_fa2_n1024_oracle arch={} time={}  (Gate F1 repaired positions)",
        gpu.arch, time
    );
    if !matches!(
        gpu.arch.as_str(),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151"
    ) {
        eprintln!("skip: need gfx11, got {}", gpu.arch);
        return;
    }
    if gpu.arch != "gfx1151" {
        eprintln!(
            "note: N1024 single-launch requires gfx1151 lab direct cap; \
             arch={} will still run N≤512 and may fail N1024 launches",
            gpu.arch
        );
    }

    // Metadata-first: compile copy-positions, gate resource ceiling, then any launch.
    ensure_copy_positions(&mut gpu);
    gate_copy_positions_metadata(&gpu.arch);

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let d_s1 = gpu.upload_f32(&signs1, &[256]).unwrap();
    let d_s2 = gpu.upload_f32(&signs2, &[256]).unwrap();

    let mut n_fail = 0usize;

    // ── N≤512: accepted FA2 dual-launch bit-exact ─────────────────────
    for &(batch, start, seq_len) in SHAPES_LE512 {
        run_position_contract(&mut gpu, start, batch, &format!("le512/b{batch}_s{start}"));
        fixture_le512_q8(&mut gpu, batch, start, seq_len, time, &mut n_fail);
        fixture_le512_fwht3(
            &mut gpu, &d_s1, &d_s2, batch, start, seq_len, time, &mut n_fail,
        );
    }

    // ── N1024: one launch vs two N512 ─────────────────────────────────
    for &(batch, start, seq_len) in SHAPES_N1024 {
        fixture_n1024_q8(&mut gpu, batch, start, seq_len, time, &mut n_fail);
        fixture_n1024_fwht3(
            &mut gpu, &d_s1, &d_s2, batch, start, seq_len, time, &mut n_fail,
        );
    }

    let n_shapes = SHAPES_LE512.len() + SHAPES_N1024.len();
    if n_fail == 0 {
        eprintln!("F1 PASS  shapes={n_shapes} modes=Q8,fwht3  positions=byte-views");
    } else {
        eprintln!("F1 FAIL  n_fail={n_fail}");
        std::process::exit(1);
    }
}
