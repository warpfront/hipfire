//! Throwaway resident pre-tiled oracle: shipping IU4 prefill GEMM vs a second
//! copy of the weights stored in exactly the byte image the LDS tile wants.
//!
//! Loads real qt=44 gate_proj (17408x5120) and down_proj (5120x17408) from an
//! MQ4-XT HFQ, builds the resident pre-tiled image on the host (per (128-row
//! tile, K256 group): the exact 128x42-dword LDS image the shipping fill
//! constructs — payload words verbatim, headers converted+replicated, pads
//! zero), quantizes X via `ensure_int4_mmq_x`, and raw-launches shipping
//! `*_full_set_lf16_col_gfx1151` / `*_full_add_occ3_col_gfx1151` against
//! `gemm_iu4_pretiled_{set,add,set_pf}_gfx1151` (same grids/blocks/LDS as
//! their shipping twins, except _pf which needs 52,224 B LDS) on identical
//! inputs. Asserts bitwise-equal Y per output plus A/PT/Xq immutability.
//!
//! Also covers phase edges with synthetic MQ4G256V2 data.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-pretiled HIPFIRE_GFX11_MQ4V2_IU4=1 \
//!     TIME=1 ./target/release/examples/tmp_halo_pretiled <model>
//!     (built with `cargo build --release -p hipfire-runtime --example
//!     tmp_halo_pretiled --features lab`)
//!
//! `--compile-only` JITs all symbols, prints the kernel cache path, and
//! returns before touching the model. TIME=1 prints interleaved GPU-event
//! us/launch (>=100 samples each, warm) with medians, TOPS and % of the
//! 107.8-TOPS peak for gate set (lf16 vs pretiled, lf16 vs _pf) and down
//! add (occ3 vs pretiled).

use hip_bridge::{DeviceBuffer, KernargBlob};
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;
use std::path::{Path, PathBuf};
use std::time::Instant;

const IU4_SRC: &str = concat!(
    include_str!("../../../kernels/src/block_i4_128_quant.hip"),
    include_str!("../../../kernels/src/gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip"),
    include_str!("tmp_halo_pretiled.hip"),
);
// Fresh module identity so no stale cache entry can alias these symbols.
const MODULE: &str = "tmp_halo_pretiled";

const SHIP_SET: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_set_lf16_col_gfx1151";
const SHIP_ADD: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_occ3_col_gfx1151";
const PT_SET: &str = "gemm_iu4_pretiled_set_gfx1151";
const PT_ADD: &str = "gemm_iu4_pretiled_add_gfx1151";
const PT_SET_PF: &str = "gemm_iu4_pretiled_set_pf_gfx1151";

const SHARED: u32 = (128 * 18 + 128 * 42) * 4; // 30720, same planes as shipping
const SHARED_PF: u32 = (128 * 18 + 2 * 128 * 42) * 4; // 52224, double-buffered A
const SHIP_SET_BLOCK: [u32; 3] = [32, 16, 1];
const SHIP_ADD_BLOCK: [u32; 3] = [32, 8, 1];
const PT_SET_BLOCK: [u32; 3] = [32, 16, 1];
const PT_ADD_BLOCK: [u32; 3] = [32, 8, 1];
const PT_PF_BLOCK: [u32; 3] = [32, 16, 1];
const QT_MQ4V2: u8 = 44;
const GATE_NAME: &str = "model.language_model.layers.0.mlp.gate_proj.weight";
const DOWN_NAME: &str = "model.language_model.layers.0.mlp.down_proj.weight";
const PEAK_TOPS: f64 = 107.8;

// Resident pre-tiled image geometry (dwords).
const PT_ROW_DWORDS: usize = 42;
const PT_TILE_DWORDS: usize = 128 * PT_ROW_DWORDS; // 5376
const PT_TILE_BYTES: usize = PT_TILE_DWORDS * 4; // 21504

fn prng_f32(i: usize, salt: u32) -> f32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

/// f32 -> f16 bits (round-to-nearest-even mantissa truncation is enough for
/// the benign scales used here; NaN/Inf pass through as Inf/NaN).
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ff_ff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x0200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    if e16 <= 0 {
        return s; // flush denormals; synthetic scales never land here
    }
    s | ((e16 as u16) << 10) | ((m >> 13) as u16)
}

/// Exact f16 bits -> f32 (all f16 values are exactly representable in f32).
fn f16_to_f32(b: u16) -> f32 {
    let s = ((b >> 15) & 1) as u32;
    let e = ((b >> 10) & 0x1f) as u32;
    let m = (b & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // Subnormal: normalize.
            let mut e32 = 127 - 14i32;
            let mut mm = m;
            while mm & 0x400 == 0 {
                mm <<= 1;
                e32 -= 1;
            }
            (s << 31) | ((e32 as u32) << 23) | ((mm & 0x3ff) << 13)
        }
    } else if e == 31 {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | (((e + 127 - 15)) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

/// Exact-ish f32 -> f16 bits with round-to-nearest-even, matching the
/// device's `( _Float16 )` conversion (v_cvt_f16_f32, RN). Overflow -> Inf;
/// NaN -> quiet NaN with the top payload bits preserved (never occurs in
/// practice; real and synthetic headers are finite).
fn f32_to_f16_rtne(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ff_ff;
    if e == 0xff {
        // Inf/NaN.
        return if m == 0 {
            s | 0x7c00
        } else {
            s | 0x7e00 | ((m >> 13) as u16 & 0x01ff)
        };
    }
    let e16 = e - 127 + 15;
    if e16 >= 31 {
        return s | 0x7c00;
    }
    if e16 <= 0 {
        // Subnormal or zero: round the 24-bit mantissa (with hidden 1 for
        // e16 == 0 edge, i.e. tiny normals) at the subnormal LSB.
        if e16 < -10 {
            return s; // rounds to zero
        }
        let with_hidden = if e == 0 { m } else { m | 0x80_0000 };
        let shift = (14 - e16) as u32; // 14..24
        let half = 1u32 << (shift - 1);
        let rounded = (with_hidden + half) >> shift;
        // rounded == 0x400 would carry into the normal range only when
        // e16 == 0 and rounding pushes 0x3ff up; handle via normal path.
        if rounded >= 0x400 && e16 == 0 {
            return s | (1 << 10);
        }
        return s | (rounded as u16 & 0x3ff);
    }
    // Normal range: round 13 mantissa bits to 10 with RN-even.
    let half = 1u32 << 12;
    let lower = m & 0x1fff;
    let mut mant = (m >> 13) as u32;
    if lower > half || (lower == half && (mant & 1) == 1) {
        mant += 1;
        if mant == 0x400 {
            // Carry into exponent.
            let e16p = e16 + 1;
            if e16p >= 31 {
                return s | 0x7c00;
            }
            return s | ((e16p as u16) << 10);
        }
    }
    s | ((e16 as u16) << 10) | (mant as u16)
}

/// Device header conversion for one raw group header u32: the shipping fill
/// computes dm = half2(f16(sc), f16(zp)) from the two f16 halves, i.e.
/// f16->f32 (exact) then RTNE f32->f16. For finite values this is the
/// identity; the software path below does not assume it.
fn convert_header(hs: u32) -> u32 {
    let sc = f16_to_f32((hs & 0xffff) as u16);
    let zp = f16_to_f32((hs >> 16) as u16);
    (f32_to_f16_rtne(sc) as u32) | ((f32_to_f16_rtne(zp) as u32) << 16)
}

/// Host relayout: MQ4G256V2 bytes -> resident pre-tiled image. Per (128-row
/// tile T, K256 group kb): 128 rows x 42 dwords = 21,504 B at byte offset
/// (T*G + kb) * 21,504, where row r holds payload words verbatim (group
/// bytes [8..136]), header dwords [32..40) (hs0 x4, hs1 x4 post-conversion),
/// and zero pad [40..42).
fn relayout_pretiled(a: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(m % 128, 0, "M={m} not full-tile");
    assert_eq!(k % 256, 0, "K={k} not multiple of 256");
    let gpr = k / 256;
    let ntiles = m / 128;
    assert_eq!(a.len(), m * gpr * 136, "MQ4 byte len");
    let mut out = vec![0u8; ntiles * gpr * PT_TILE_BYTES];
    for t in 0..ntiles {
        for kb in 0..gpr {
            let dst_tile = (t * gpr + kb) * PT_TILE_BYTES;
            for r in 0..128 {
                let gsrc = ((t * 128 + r) * gpr + kb) * 136;
                let drow = dst_tile + r * PT_ROW_DWORDS * 4;
                out[drow..drow + 128].copy_from_slice(&a[gsrc + 8..gsrc + 136]);
                let hs0 = u32::from_le_bytes(a[gsrc..gsrc + 4].try_into().unwrap());
                let hs1 = u32::from_le_bytes(a[gsrc + 4..gsrc + 8].try_into().unwrap());
                let dm0 = convert_header(hs0).to_le_bytes();
                let dm1 = convert_header(hs1).to_le_bytes();
                for l in 0..4 {
                    out[drow + 128 + l * 4..drow + 132 + l * 4].copy_from_slice(&dm0);
                    out[drow + 144 + l * 4..drow + 148 + l * 4].copy_from_slice(&dm1);
                }
            }
        }
    }
    out
}

/// Synthetic MQ4G256V2 weights: 136 B/group, dual fp16 headers, full 0..15
/// nibble codes, distinct h0/h1, non-power-of-two scales.
fn pack_mq4g256v2_synth(m: usize, k: usize, salt: u32) -> Vec<u8> {
    const GROUP: usize = 256;
    const HALF: usize = 128;
    const GB: usize = 136;
    assert_eq!(k % GROUP, 0, "K={k} not multiple of 256");
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GB];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GB;
            let sc0 = f16_bits(0.05 + (prng_u32((r * gpr + g) as u64, salt) % 20) as f32 * 0.001);
            let zp0 = f16_bits(-0.35);
            let sc1 =
                f16_bits(0.07 + (prng_u32((r * gpr + g) as u64, salt ^ 1) % 20) as f32 * 0.001);
            let zp1 = f16_bits(-0.45);
            blob[dst..dst + 2].copy_from_slice(&sc0.to_le_bytes());
            blob[dst + 2..dst + 4].copy_from_slice(&zp0.to_le_bytes());
            blob[dst + 4..dst + 6].copy_from_slice(&sc1.to_le_bytes());
            blob[dst + 6..dst + 8].copy_from_slice(&zp1.to_le_bytes());
            for i in 0..HALF {
                let lo = (prng_u32((r * 1024 + g * HALF + i) as u64, salt) % 16) as u8;
                let hi = (prng_u32((r * 1024 + g * HALF + i) as u64, salt ^ 0x55) % 16) as u8;
                blob[dst + 8 + i] = lo | (hi << 4);
            }
        }
    }
    blob
}

fn cache_dir(arch: &str) -> PathBuf {
    let root = std::env::var_os("HIPFIRE_KERNEL_CACHE").map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var_os("HOME").unwrap_or_else(|| "/tmp".into());
        PathBuf::from(home).join(".hipfire_kernels")
    });
    root.join(arch)
}

fn mk_blob(
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) -> Vec<u8> {
    let mut b = KernargBlob::new();
    b.push_ptr(a);
    b.push_ptr(xq);
    b.push_ptr(y);
    b.push_i32(m);
    b.push_i32(k);
    b.push_i32(n);
    b.push_i32(add);
    b.into_vec()
}

fn load_mq4v2(hfq: &HfqFile, name: &str) -> (usize, usize, Vec<u8>) {
    let (info, bytes) = hfq
        .tensor_data_vec(name)
        .unwrap_or_else(|| panic!("tensor not found: {name}"));
    assert_eq!(
        info.quant_type, QT_MQ4V2,
        "{name}: quant_type={} want {QT_MQ4V2}",
        info.quant_type
    );
    assert_eq!(info.shape.len(), 2, "{name}: expected 2D shape");
    let m = info.shape[0] as usize;
    let k = info.shape[1] as usize;
    assert_eq!(m % 128, 0, "{name}: M={m} not full-tile");
    assert_eq!(k % 256, 0, "{name}: K={k} not multiple of 256");
    let gpr = k / 256;
    let expect = m * gpr * 136;
    assert_eq!(bytes.len(), expect, "{name}: {m}x{k} byte len");
    (m, k, bytes)
}

fn bitwise_eq(a: &[f32], b: &[f32]) -> (bool, usize, Option<(usize, u32, u32)>) {
    assert_eq!(a.len(), b.len());
    let mut mism = 0usize;
    let mut first = None;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let (xb, yb) = (x.to_bits(), y.to_bits());
        if xb != yb {
            mism += 1;
            if first.is_none() {
                first = Some((i, xb, yb));
            }
        }
    }
    (mism == 0, mism, first)
}

fn dtoh_raw(gpu: &Gpu, ptr: *mut std::ffi::c_void, len: usize) -> Vec<u8> {
    let view = unsafe { DeviceBuffer::from_raw(ptr, len) };
    let mut host = vec![0u8; len];
    gpu.hip
        .memcpy_dtoh(&mut host, &view)
        .expect("memcpy_dtoh scratch snapshot");
    // Borrowed view: no free (DeviceBuffer has no Drop; nothing to do).
    let _ = view.is_borrowed();
    host
}

#[allow(clippy::too_many_arguments)]
fn launch_once(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) {
    let mut blob = mk_blob(a, xq, y, m, k, n, add);
    gpu.launch_kernel_blob(func, grid, block, shared, &mut blob)
        .unwrap_or_else(|e| panic!("launch {func}: {e}"));
}

#[allow(clippy::too_many_arguments)]
fn time_us(
    gpu: &mut Gpu,
    func: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    a: *const std::ffi::c_void,
    xq: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    m: i32,
    k: i32,
    n: i32,
    add: i32,
) -> f64 {
    let mut blob = mk_blob(a, xq, y, m, k, n, add);
    let start = gpu.hip.event_create().expect("event start");
    let stop = gpu.hip.event_create().expect("event stop");
    let stream = gpu.active_stream.as_ref();
    gpu.hip.event_record(&start, stream).expect("record start");
    gpu.launch_kernel_blob(func, grid, block, shared, &mut blob)
        .unwrap_or_else(|e| panic!("timed launch {func}: {e}"));
    gpu.hip.event_record(&stop, stream).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync stop");
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed");
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms as f64 * 1e3
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    gpu: &mut Gpu,
    label: &str,
    a_bytes: &[u8],
    pt_bytes: &[u8],
    m: usize,
    k: usize,
    n: usize,
    add: bool,
    ship_sym: &str,
    ship_block: [u32; 3],
    ship_shared: u32,
    pt_sym: &str,
    pt_block: [u32; 3],
    pt_shared: u32,
    salt: u32,
    time: bool,
) -> bool {
    assert_eq!(m % 128, 0, "{label}: M={m} not full-tile");
    assert_eq!(n % 128, 0, "{label}: N={n} not full-tile");
    assert_eq!(k % 256, 0, "{label}: K={k} not multiple of 256");
    let grid = [(n / 128) as u32, (m / 128) as u32, 1];

    // X through the real quantizer (N rows x K cols, N=batch here).
    let x: Vec<f32> = (0..n * k)
        .map(|i| {
            let col = i / k;
            let scale = 0.75 + (col % 5) as f32 * 0.15;
            prng_f32(i, salt) * scale
        })
        .collect();
    let d_x = gpu.upload_f32(&x, &[n, k]).expect("upload X");
    let xq_ptr = gpu
        .ensure_int4_mmq_x(&d_x, n, k)
        .expect("ensure_int4_mmq_x") as *mut std::ffi::c_void;
    let xq_len = (k / 128) * n * 72;
    let xq_before = dtoh_raw(gpu, xq_ptr, xq_len);

    let d_a = gpu.upload_raw(a_bytes, &[a_bytes.len()]).expect("upload A");
    let a_ptr = d_a.buf.as_ptr() as *const std::ffi::c_void;
    let d_pt = gpu.upload_raw(pt_bytes, &[pt_bytes.len()]).expect("upload PT");
    let pt_ptr = d_pt.buf.as_ptr() as *const std::ffi::c_void;

    // Y0 seed: set path zeros; add path NONZERO so accumulation is exercised.
    let y_elems = n * m;
    let y0: Vec<f32> = if add {
        (0..y_elems)
            .map(|i| prng_f32(i, 0xADD0_BEEF ^ salt) * 0.37 + 0.11)
            .collect()
    } else {
        vec![0.0f32; y_elems]
    };
    let d_y_ship = gpu.upload_f32(&y0, &[n, m]).expect("upload Y ship");
    let d_y_pt = gpu.upload_f32(&y0, &[n, m]).expect("upload Y pt");

    for sym in [ship_sym, pt_sym] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
    }

    let add_i = i32::from(add);
    let (mi, ki, ni) = (m as i32, k as i32, n as i32);
    launch_once(
        gpu,
        ship_sym,
        grid,
        ship_block,
        ship_shared,
        a_ptr,
        xq_ptr as *const _,
        d_y_ship.buf.as_ptr() as *const _,
        mi,
        ki,
        ni,
        add_i,
    );
    launch_once(
        gpu,
        pt_sym,
        grid,
        pt_block,
        pt_shared,
        pt_ptr,
        xq_ptr as *const _,
        d_y_pt.buf.as_ptr() as *const _,
        mi,
        ki,
        ni,
        add_i,
    );
    gpu.hip.device_synchronize().expect("sync parity");

    let y_ship = gpu.download_f32(&d_y_ship).expect("dl ship");
    let y_pt = gpu.download_f32(&d_y_pt).expect("dl pt");
    let finite = y_ship.iter().all(|v| v.is_finite()) && y_pt.iter().all(|v| v.is_finite());
    let (eq, mism, first) = bitwise_eq(&y_ship, &y_pt);
    let first_rc = first.map(|(idx, xb, yb)| {
        let row = idx % m;
        let col = idx / m;
        format!("(row={row},col={col}) ship=0x{xb:08x} pt=0x{yb:08x}")
    });
    let moved = if add {
        y_ship
            .iter()
            .zip(y0.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    } else {
        y_ship.iter().any(|v| v.to_bits() != 0)
    };

    // A/PT/Xq immutability over both launches.
    let mut a_back = vec![0u8; a_bytes.len()];
    gpu.hip
        .memcpy_dtoh(&mut a_back, &d_a.buf)
        .expect("dl A back");
    let a_same = a_back == a_bytes;
    let mut pt_back = vec![0u8; pt_bytes.len()];
    gpu.hip
        .memcpy_dtoh(&mut pt_back, &d_pt.buf)
        .expect("dl PT back");
    let pt_same = pt_back == pt_bytes;
    let xq_after = dtoh_raw(gpu, xq_ptr, xq_len);
    let xq_same = xq_after == xq_before;

    let ok = eq && finite && moved && a_same && pt_same && xq_same;
    eprintln!(
        "  [{label} M={m} K={k} N={n} add={add} {ship_sym} vs {pt_sym}] finite={finite} moved={moved} \
         bitwise_eq={eq} mism={mism} first={first_rc:?} a_immutable={a_same} pt_immutable={pt_same} xq_immutable={xq_same} [{}]",
        if ok { "PASS" } else { "FAIL" }
    );

    if time {
        const WARM: usize = 10;
        const SAMPLES: usize = 100;
        let y0_bytes: Vec<u8> = y0.iter().flat_map(|v| v.to_le_bytes()).collect();
        let restore = |gpu: &mut Gpu, y_ptr_buf: &DeviceBuffer| {
            gpu.hip
                .memcpy_htod(y_ptr_buf, &y0_bytes)
                .expect("restore Y0");
        };
        for _ in 0..WARM {
            if add {
                restore(gpu, &d_y_ship.buf);
            }
            launch_once(
                gpu, ship_sym, grid, ship_block, ship_shared, a_ptr, xq_ptr as *const _,
                d_y_ship.buf.as_ptr() as *const _, mi, ki, ni, add_i,
            );
            if add {
                restore(gpu, &d_y_pt.buf);
            }
            launch_once(
                gpu, pt_sym, grid, pt_block, pt_shared, pt_ptr, xq_ptr as *const _,
                d_y_pt.buf.as_ptr() as *const _, mi, ki, ni, add_i,
            );
        }
        gpu.hip.device_synchronize().expect("sync warm");

        let mut ship_us = Vec::with_capacity(SAMPLES);
        let mut pt_us = Vec::with_capacity(SAMPLES);
        for i in 0..SAMPLES {
            // Interleave ship/pt to share thermal state; add Y restored
            // outside each timed interval.
            if i % 2 == 0 {
                if add {
                    restore(gpu, &d_y_ship.buf);
                }
                ship_us.push(time_us(
                    gpu, ship_sym, grid, ship_block, ship_shared, a_ptr, xq_ptr as *const _,
                    d_y_ship.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
                if add {
                    restore(gpu, &d_y_pt.buf);
                }
                pt_us.push(time_us(
                    gpu, pt_sym, grid, pt_block, pt_shared, pt_ptr, xq_ptr as *const _,
                    d_y_pt.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
            } else {
                if add {
                    restore(gpu, &d_y_pt.buf);
                }
                pt_us.push(time_us(
                    gpu, pt_sym, grid, pt_block, pt_shared, pt_ptr, xq_ptr as *const _,
                    d_y_pt.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
                if add {
                    restore(gpu, &d_y_ship.buf);
                }
                ship_us.push(time_us(
                    gpu, ship_sym, grid, ship_block, ship_shared, a_ptr, xq_ptr as *const _,
                    d_y_ship.buf.as_ptr() as *const _, mi, ki, ni, add_i,
                ));
            }
        }
        let smed = median(&mut ship_us);
        let pmed = median(&mut pt_us);
        let ops = 2.0 * m as f64 * k as f64 * n as f64;
        let (stops, ptops) = (ops / smed / 1e6, ops / pmed / 1e6);
        let (spct, ppct) = (stops / PEAK_TOPS * 100.0, ptops / PEAK_TOPS * 100.0);
        let ratio = pmed / smed;
        eprintln!(
            "  TIME {label}: ship_med={smed:.3} us ({stops:.2} TOPS, {spct:.1}% peak)  \
             pt_med={pmed:.3} us ({ptops:.2} TOPS, {ppct:.1}% peak)  ratio={ratio:.4} (n={SAMPLES} interleaved)"
        );
    }

    let _ = gpu.free_tensor(d_x);
    let _ = gpu.free_tensor(d_a);
    let _ = gpu.free_tensor(d_pt);
    let _ = gpu.free_tensor(d_y_ship);
    let _ = gpu.free_tensor(d_y_pt);
    ok
}

fn resident_report(name: &str, m: usize, k: usize, a_len: usize, pt_len: usize) {
    let weights = m * k;
    let mq4_bpw = a_len as f64 / weights as f64 * 8.0;
    let pt_bpw = pt_len as f64 / weights as f64 * 8.0;
    eprintln!(
        "  RESIDENT {name} {m}x{k}: MQ4={a_len} B ({mq4_bpw:.3} bpw)  \
         pretiled={pt_len} B ({pt_bpw:.3} bpw)  ratio={:.4}x",
        pt_len as f64 / a_len as f64
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let compile_only = args.iter().any(|a| a == "--compile-only");
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("usage: tmp_halo_pretiled <model.mq4-xt> [--compile-only]");
            std::process::exit(2);
        });
    let time = std::env::var("TIME").ok().as_deref() == Some("1");

    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    eprintln!("tmp_halo_pretiled on {}  model={model}  TIME={time}", gpu.arch);
    if gpu.arch != "gfx1151" {
        eprintln!(
            "WARN: pretiled symbols are #if __gfx1151__; arch={} may fail JIT of pt entries",
            gpu.arch
        );
    }

    // Compile-only: JIT all five entries, print the cache path, return.
    for sym in [SHIP_SET, SHIP_ADD, PT_SET, PT_ADD, PT_SET_PF] {
        gpu.ensure_kernel_public(MODULE, IU4_SRC, sym)
            .unwrap_or_else(|e| panic!("JIT {sym}: {e}"));
        eprintln!("JIT OK: {sym}");
    }
    let dir = cache_dir(&gpu.arch);
    eprintln!("CACHE_PATH {}", dir.display());
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let s = ent.file_name().to_string_lossy().into_owned();
            if s.starts_with(MODULE) {
                eprintln!("  cached: {s}");
            }
        }
    }
    if compile_only {
        return;
    }

    let t0 = Instant::now();
    let hfq = HfqFile::open(Path::new(&model)).expect("open HFQ");
    let (gate_m, gate_k, gate) = load_mq4v2(&hfq, GATE_NAME);
    let (down_m, down_k, down) = load_mq4v2(&hfq, DOWN_NAME);
    eprintln!(
        "loaded gate_proj {gate_m}x{gate_k} ({} B)  down_proj {down_m}x{down_k} ({} B) in {:.2}s",
        gate.len(),
        down.len(),
        t0.elapsed().as_secs_f64()
    );
    assert_eq!((gate_m, gate_k), (17408, 5120), "gate shape");
    assert_eq!((down_m, down_k), (5120, 17408), "down shape");
    // Drop mmap pressure before large GPU allocs on UMA.
    drop(hfq);

    let t1 = Instant::now();
    let gate_pt = relayout_pretiled(&gate, gate_m, gate_k);
    let down_pt = relayout_pretiled(&down, down_m, down_k);
    eprintln!(
        "relayout host: gate_pt={} B down_pt={} B in {:.2}s",
        gate_pt.len(),
        down_pt.len(),
        t1.elapsed().as_secs_f64()
    );
    resident_report("gate_proj", gate_m, gate_k, gate.len(), gate_pt.len());
    resident_report("down_proj", down_m, down_k, down.len(), down_pt.len());
    eprintln!(
        "  RESIDENT total: MQ4={} B  pretiled={} B  duplicate overhead=+{:.2}%",
        gate.len() + down.len(),
        gate_pt.len() + down_pt.len(),
        (gate_pt.len() + down_pt.len()) as f64 / (gate.len() + down.len()) as f64 * 100.0 - 100.0
    );

    let mut ok = true;
    // Production bits: gate set (lf16 vs pretiled, lf16 vs _pf) and down
    // add (occ3 vs pretiled add); timing only on these three pairs.
    ok &= run_case(
        &mut gpu, "gate_proj/set", &gate, &gate_pt, gate_m, gate_k, 512, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET, PT_SET_BLOCK, SHARED, 0x9E71_0000, time,
    );
    ok &= run_case(
        &mut gpu, "gate_proj/set_pf", &gate, &gate_pt, gate_m, gate_k, 512, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET_PF, PT_PF_BLOCK, SHARED_PF, 0x9E71_0001, time,
    );
    ok &= run_case(
        &mut gpu, "down_proj/add", &down, &down_pt, down_m, down_k, 512, true,
        SHIP_ADD, SHIP_ADD_BLOCK, SHARED, PT_ADD, PT_ADD_BLOCK, SHARED, 0x9E71_0002, time,
    );
    // Parity-only production corners.
    ok &= run_case(
        &mut gpu, "gate_proj/add", &gate, &gate_pt, gate_m, gate_k, 512, true,
        SHIP_ADD, SHIP_ADD_BLOCK, SHARED, PT_ADD, PT_ADD_BLOCK, SHARED, 0x9E71_0003, false,
    );
    ok &= run_case(
        &mut gpu, "down_proj/set", &down, &down_pt, down_m, down_k, 512, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET, PT_SET_BLOCK, SHARED, 0x9E71_0004, false,
    );
    ok &= run_case(
        &mut gpu, "down_proj/set_pf", &down, &down_pt, down_m, down_k, 512, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET_PF, PT_PF_BLOCK, SHARED_PF, 0x9E71_0005, false,
    );

    // Phase edges, synthetic data (host relayout per case).
    for (ki, kk) in [256usize, 512, 768].iter().enumerate() {
        let a = pack_mq4g256v2_synth(256, *kk, 0xED6E_0000 + ki as u32);
        let pt = relayout_pretiled(&a, 256, *kk);
        ok &= run_case(
            &mut gpu, &format!("edgeK{kk}/set"), &a, &pt, 256, *kk, 256, false,
            SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET, PT_SET_BLOCK, SHARED, 0xE001 + ki as u32, false,
        );
        ok &= run_case(
            &mut gpu, &format!("edgeK{kk}/add"), &a, &pt, 256, *kk, 256, true,
            SHIP_ADD, SHIP_ADD_BLOCK, SHARED, PT_ADD, PT_ADD_BLOCK, SHARED, 0xE101 + ki as u32, false,
        );
    }
    let a1024 = pack_mq4g256v2_synth(1024, 5120, 0xED6E_1000);
    let pt1024 = relayout_pretiled(&a1024, 1024, 5120);
    ok &= run_case(
        &mut gpu, "m1024n128/set", &a1024, &pt1024, 1024, 5120, 128, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET, PT_SET_BLOCK, SHARED, 0xE010, false,
    );
    ok &= run_case(
        &mut gpu, "m1024n128/add", &a1024, &pt1024, 1024, 5120, 128, true,
        SHIP_ADD, SHIP_ADD_BLOCK, SHARED, PT_ADD, PT_ADD_BLOCK, SHARED, 0xE011, false,
    );
    let a128 = pack_mq4g256v2_synth(128, 5120, 0xED6E_2000);
    let pt128 = relayout_pretiled(&a128, 128, 5120);
    ok &= run_case(
        &mut gpu, "m128n512/set", &a128, &pt128, 128, 5120, 512, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET, PT_SET_BLOCK, SHARED, 0xE020, false,
    );
    ok &= run_case(
        &mut gpu, "m128n512/set_pf", &a128, &pt128, 128, 5120, 512, false,
        SHIP_SET, SHIP_SET_BLOCK, SHARED, PT_SET_PF, PT_PF_BLOCK, SHARED_PF, 0xE021, false,
    );
    ok &= run_case(
        &mut gpu, "m128n512/add", &a128, &pt128, 128, 5120, 512, true,
        SHIP_ADD, SHIP_ADD_BLOCK, SHARED, PT_ADD, PT_ADD_BLOCK, SHARED, 0xE022, false,
    );
    let a128b = pack_mq4g256v2_synth(128, 5120, 0xED6E_3000);
    let pt128b = relayout_pretiled(&a128b, 128, 5120);
    ok &= run_case(
        &mut gpu, "m128n256/add", &a128b, &pt128b, 128, 5120, 256, true,
        SHIP_ADD, SHIP_ADD_BLOCK, SHARED, PT_ADD, PT_ADD_BLOCK, SHARED, 0xE030, false,
    );

    if ok {
        eprintln!("PRETILED PASS: shipping vs pretiled bitwise-equal on all cases");
    } else {
        eprintln!("PRETILED FAIL");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip_is_identity_for_all_finite_f16() {
        // The device computes half2(f16(sc),f16(zp)) from the raw header
        // bits; the host relayout must reproduce those bits exactly.
        for b in 0..=0xffffu32 {
            let h = (b as u16) as u32;
            let sc16 = (h & 0xffff) as u16;
            let zp16 = (h >> 16) as u16;
            // f16->f32 is exact; RTNE back must be the identity for all
            // finite values (exponent 0..30).
            let e = (sc16 >> 10) & 0x1f;
            if e != 31 {
                assert_eq!(f32_to_f16_rtne(f16_to_f32(sc16)), sc16, "sc bits {sc16:04x}");
            }
            let e = (zp16 >> 10) & 0x1f;
            if e != 31 {
                assert_eq!(f32_to_f16_rtne(f16_to_f32(zp16)), zp16, "zp bits {zp16:04x}");
            }
        }
    }

    #[test]
    fn relayout_matches_expected_image() {
        // 128x256, one tile, one group: payload verbatim, headers x4, pad 0.
        let m = 128usize;
        let k = 256usize;
        let a = pack_mq4g256v2_synth(m, k, 0x1234);
        let pt = relayout_pretiled(&a, m, k);
        assert_eq!(pt.len(), PT_TILE_BYTES);
        for r in 0..128 {
            let gsrc = r * 136;
            let drow = r * PT_ROW_DWORDS * 4;
            assert_eq!(&pt[drow..drow + 128], &a[gsrc + 8..gsrc + 136], "row {r} payload");
            let hs0 = u32::from_le_bytes(a[gsrc..gsrc + 4].try_into().unwrap());
            let hs1 = u32::from_le_bytes(a[gsrc + 4..gsrc + 8].try_into().unwrap());
            let dm0 = convert_header(hs0);
            let dm1 = convert_header(hs1);
            // Finite synthetic headers: conversion must be the identity.
            assert_eq!(dm0, hs0, "row {r} hs0");
            assert_eq!(dm1, hs1, "row {r} hs1");
            for l in 0..4 {
                let w0 = u32::from_le_bytes(pt[drow + 128 + l * 4..drow + 132 + l * 4].try_into().unwrap());
                let w1 = u32::from_le_bytes(pt[drow + 144 + l * 4..drow + 148 + l * 4].try_into().unwrap());
                assert_eq!(w0, hs0, "row {r} hs0 rep {l}");
                assert_eq!(w1, hs1, "row {r} hs1 rep {l}");
            }
            assert_eq!(&pt[drow + 160..drow + 168], &[0u8; 8], "row {r} pad");
        }
    }

    #[test]
    fn relayout_sizes_and_bpw() {
        // Exact bpw: 42 dwords/row-group = 168 B per 256 weights = 5.25 bpw
        // vs MQ4 136 B = 4.25 bpw (ratio 168/136 = 1.2353x).
        let (m, k) = (256usize, 512usize);
        let a = pack_mq4g256v2_synth(m, k, 0x5678);
        let pt = relayout_pretiled(&a, m, k);
        assert_eq!(pt.len(), (m / 128) * (k / 256) * PT_TILE_BYTES);
        let weights = (m * k) as f64;
        let mq4_bpw = a.len() as f64 / weights * 8.0;
        let pt_bpw = pt.len() as f64 / weights * 8.0;
        assert!((mq4_bpw - 4.25).abs() < 1e-12, "{mq4_bpw}");
        assert!((pt_bpw - 5.25).abs() < 1e-12, "{pt_bpw}");
        // Gate production shape.
        assert_eq!((17408 / 128) * (5120 / 256) * PT_TILE_BYTES, 58_490_880);
    }
}
