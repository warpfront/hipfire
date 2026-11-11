// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! X1 oracle: `Gpu::argmax_f32_pair_merge` (kernel `argmax_f32_pair_merge`) over two
//! vocabulary shards equals `argmax_f32` over the concatenated full row, id for id.
//!
//! The peer shard is a second local buffer whose device address is the "peer
//! pointer". Row flavours: random, five-level ties, equal maxima straddling the
//! shard boundary (at w0-1 | w0, w0 | w0+1, 0 | w0), +0/-0 across the boundary in
//! both orders, ties inside each shard, NaN / +inf / -inf, a finite value in only
//! one shard, and rows with no finite value (answer 0). Widths: (124160, 124160)
//! and small odd widths, including more rows than one 256-row block. Each case also
//! runs with `out_top1` aliasing `local_top1`.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test x1_argmax_pair_merge_gpu -- --ignored --nocapture

use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32, ArgmaxPairMerge};
use rdna_compute::{DType, Gpu, GpuTensor};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

/// Number of flavours `make_row` knows.
const FLAVOURS: usize = 20;

/// One full row of `w0 + w1` values (shard 0 = `[0, w0)`), flavour `f`.
fn make_row(g: &mut Lcg, w0: usize, w1: usize, f: usize) -> Vec<f32> {
    let n = w0 + w1;
    // Strictly negative filler so planted maxima win.
    let neg = |g: &mut Lcg| -(1.0 + g.unit().abs() * 4.0);
    let mut r: Vec<f32> = (0..n).map(|_| neg(g)).collect();
    match f {
        0 => r = (0..n).map(|_| g.unit() * 8.0).collect(),
        1 => r = (0..n).map(|_| (g.next() % 5) as f32).collect(),
        2 => {
            r[w0 - 1] = 10.0;
            r[w0] = 10.0;
        }
        3 => {
            r[w0] = 10.0;
            r[(w0 + 1).min(n - 1)] = 10.0;
        }
        4 => {
            r[0] = 10.0;
            r[w0] = 10.0;
        }
        5 => {
            r[w0 - 1] = -0.0;
            r[w0] = 0.0;
        }
        6 => {
            r[w0 - 1] = 0.0;
            r[w0] = -0.0;
        }
        7 => {
            // Shard 0 has no finite value; shard 1 has some.
            for v in &mut r[..w0] {
                *v = match g.below(3) {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => f32::NEG_INFINITY,
                };
            }
        }
        8 => {
            // Shard 1 has no finite value; shard 0 has some.
            for v in &mut r[w0..] {
                *v = match g.below(3) {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => f32::NEG_INFINITY,
                };
            }
        }
        9 => {
            for v in &mut r {
                *v = match g.below(3) {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    _ => f32::NEG_INFINITY,
                };
            }
        }
        10 => {
            // Ties inside each shard.
            r[5usize.min(w0 - 1)] = 9.0;
            r[(w0 / 2).max(6).min(w0 - 1)] = 9.0;
            r[w0 + 3usize.min(w1 - 1)] = 9.0;
            r[w0 + (w1 / 2).max(4).min(w1 - 1)] = 9.0;
        }
        11 => {
            // Shard 1 max above shard 0 max.
            r[g.below(w0)] = 5.0;
            r[w0 + g.below(w1)] = 6.0;
        }
        12 => {
            // Shard 0 max above shard 1 max.
            r[g.below(w0)] = 6.0;
            r[w0 + g.below(w1)] = 5.0;
        }
        13 => {
            r[g.below(n)] = f32::INFINITY; // +inf never wins
            r[g.below(n)] = f32::NAN;
            r[g.below(n)] = 3.0;
        }
        14 => {
            r[g.below(n)] = f32::NEG_INFINITY;
            r[g.below(n)] = 2.0;
            r[g.below(n)] = 2.0;
        }
        15 => r = (0..n).map(|_| bf16(g.unit() * 14.0)).collect(),
        16 => r = vec![1.5; n], // constant: first index of shard 0
        17 => {
            // Equal maxima: last of shard 0 and last of shard 1.
            r[w0 - 1] = 7.0;
            r[n - 1] = 7.0;
        }
        18 => {
            // Single finite value, in shard 1 only, at its first element.
            r = vec![f32::NEG_INFINITY; n];
            r[w0] = -123.5;
        }
        _ => {
            // Single finite value, in shard 0 only, at its last element.
            r = vec![f32::NAN; n];
            r[w0 - 1] = -7.25;
        }
    }
    r
}

fn raw_i32(gpu: &mut Gpu, rows: usize) -> GpuTensor {
    gpu.zeros(&[rows * 4], DType::Raw).expect("alloc i32 rows")
}

fn read_i32(gpu: &Gpu, t: &GpuTensor, rows: usize) -> Vec<i32> {
    let b = gpu.download_raw_bytes(t).expect("download");
    b[..rows * 4]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn run_case(gpu: &mut Gpu, g: &mut Lcg, w0: usize, w1: usize, rows: usize) {
    let rows_v: Vec<Vec<f32>> = (0..rows)
        .map(|r| make_row(g, w0, w1, r % FLAVOURS))
        .collect();
    run_rows(gpu, w0, w1, rows_v);
}

/// Words of poisoned guard after every PM output buffer.
const GUARD: usize = 64;
const POISON: u32 = 0xA5A5_A5A5;

struct PmTwin {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
}

/// The committed `argmax_pair_merge_pm_gfx1201` bundle (artifact `kernels/argmax_pair_merge_pm_gfx1201.hxaco`).
fn load_pm(gpu: &Gpu) -> Option<PmTwin> {
    if gpu.arch != "gfx1201" {
        eprintln!("PM twin: skipped, gpu arch is {} (gfx1201 only)", gpu.arch);
        return None;
    }
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/argmax_pair_merge_pm_gfx1201.hxaco"
    );
    let image = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let module = gpu.hip.module_load_data(&image).expect("load PM bundle");
    let func = gpu
        .hip
        .module_get_function(&module, "argmax_pair_merge_pm_gfx1201")
        .expect("PM symbol");
    Some(PmTwin {
        _module: module,
        func,
    })
}

/// i32 buffer of `words.len() + GUARD` words, guard poisoned.
fn guarded(gpu: &Gpu, words: &[i32]) -> GpuTensor {
    let mut bytes = Vec::with_capacity((words.len() + GUARD) * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    for _ in 0..GUARD {
        bytes.extend_from_slice(&POISON.to_le_bytes());
    }
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("guarded buffer")
}

fn read_guarded(gpu: &Gpu, t: &GpuTensor, rows: usize) -> Vec<u8> {
    let b = gpu.download_raw_bytes(t).expect("download");
    b[..(rows + GUARD) * 4].to_vec()
}

fn assert_guard(bytes: &[u8], rows: usize, what: &str) {
    for w in 0..GUARD {
        let o = (rows + w) * 4;
        assert_eq!(
            u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()),
            POISON,
            "{what}: guard word {w} overwritten"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn pm_launch(
    gpu: &Gpu,
    pm: &PmTwin,
    l0: &GpuTensor,
    i0: &GpuTensor,
    peer_logits: u64,
    peer_top1: u64,
    out: &GpuTensor,
    rows: usize,
    w0: usize,
    w1: usize,
) {
    let (a0, a1) = (l0.buf.as_ptr() as u64, i0.buf.as_ptr() as u64);
    let ao = out.buf.as_ptr() as u64;
    let (r, x0, x1) = (rows as i32, w0 as i32, w1 as i32);
    let mut params = [
        &a0 as *const _ as *mut std::ffi::c_void,
        &a1 as *const _ as *mut std::ffi::c_void,
        &peer_logits as *const _ as *mut std::ffi::c_void,
        &peer_top1 as *const _ as *mut std::ffi::c_void,
        &ao as *const _ as *mut std::ffi::c_void,
        &r as *const _ as *mut std::ffi::c_void,
        &x0 as *const _ as *mut std::ffi::c_void,
        &x1 as *const _ as *mut std::ffi::c_void,
    ];
    // SAFETY: eight kernargs in the ABI order (five pointers, rows, w0, w1), all live device memory.
    unsafe {
        gpu.hip
            .launch_kernel(
                &pm.func,
                [rows.div_ceil(256) as u32, 1, 1],
                [256, 1, 1],
                0,
                None,
                &mut params,
            )
            .expect("PM launch");
    }
    gpu.hip.device_synchronize().expect("sync");
}

fn run_rows(gpu: &mut Gpu, w0: usize, w1: usize, row_vals: Vec<Vec<f32>>) {
    let rows = row_vals.len();
    let n = w0 + w1;
    let mut full = Vec::with_capacity(rows * n);
    let mut s0 = Vec::with_capacity(rows * w0);
    let mut s1 = Vec::with_capacity(rows * w1);
    for (r, row) in row_vals.into_iter().enumerate() {
        let _ = r;
        s0.extend_from_slice(&row[..w0]);
        s1.extend_from_slice(&row[w0..]);
        full.extend(row);
    }
    let t_full = gpu.upload_f32(&full, &[rows * n]).expect("full");
    let t0 = gpu.upload_f32(&s0, &[rows * w0]).expect("shard 0");
    let t1 = gpu.upload_f32(&s1, &[rows * w1]).expect("shard 1");
    let want_t = raw_i32(gpu, rows);
    let top0 = raw_i32(gpu, rows);
    let top1 = raw_i32(gpu, rows);
    let out = raw_i32(gpu, rows);
    argmax_f32(
        gpu,
        &ArgmaxF32 {
            logits: &t_full,
            indices: &want_t,
            rows,
            vocab: n,
        },
    )
    .expect("full argmax");
    argmax_f32(
        gpu,
        &ArgmaxF32 {
            logits: &t0,
            indices: &top0,
            rows,
            vocab: w0,
        },
    )
    .expect("shard 0 argmax");
    argmax_f32(
        gpu,
        &ArgmaxF32 {
            logits: &t1,
            indices: &top1,
            rows,
            vocab: w1,
        },
    )
    .expect("shard 1 argmax");
    let peer_logits = t1.buf.as_ptr() as u64;
    let peer_top1 = top1.buf.as_ptr() as u64;
    // Separate output.
    gpu.argmax_f32_pair_merge(&ArgmaxPairMerge {
        local_logits: &t0,
        local_top1: &top0,
        peer_logits,
        peer_top1,
        out_top1: &out,
        rows,
        w0,
        w1,
    })
    .expect("pair merge");
    gpu.hip.device_synchronize().expect("sync");
    let want = read_i32(gpu, &want_t, rows);
    let got = read_i32(gpu, &out, rows);
    for r in 0..rows {
        assert_eq!(
            got[r],
            want[r],
            "w0 {w0} w1 {w1} row {r} flavour {}",
            r % FLAVOURS
        );
    }
    let pm = load_pm(gpu);
    let top0_host = read_i32(gpu, &top0, rows);
    if let Some(pm) = &pm {
        // PM twin, separate output: raw bytes equal the hipcc entry and the full-row reference.
        let pout = guarded(gpu, &vec![0x5a5a_5a5a_u32 as i32; rows]);
        pm_launch(
            gpu,
            pm,
            &t0,
            &top0,
            peer_logits,
            peer_top1,
            &pout,
            rows,
            w0,
            w1,
        );
        let pb = read_guarded(gpu, &pout, rows);
        assert_guard(&pb, rows, "PM separate out");
        let hb = gpu.download_raw_bytes(&out).expect("download");
        assert_eq!(
            &pb[..rows * 4],
            &hb[..rows * 4],
            "PM vs hipcc bytes: w0 {w0} w1 {w1}"
        );
        let wb: Vec<u8> = want.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(
            &pb[..rows * 4],
            &wb[..],
            "PM vs full-row argmax: w0 {w0} w1 {w1}"
        );
        let _ = gpu.free_tensor(pout);
    }
    // Aliased output: out == local_top1.
    gpu.argmax_f32_pair_merge(&ArgmaxPairMerge {
        local_logits: &t0,
        local_top1: &top0,
        peer_logits,
        peer_top1,
        out_top1: &top0,
        rows,
        w0,
        w1,
    })
    .expect("pair merge (aliased)");
    gpu.hip.device_synchronize().expect("sync");
    assert_eq!(
        read_i32(gpu, &top0, rows),
        want,
        "aliased out: w0 {w0} w1 {w1}"
    );
    if let Some(pm) = &pm {
        // PM twin aliased: out == local_top1 (a guarded copy of the original top-1 ids).
        let alias = guarded(gpu, &top0_host);
        pm_launch(
            gpu,
            pm,
            &t0,
            &alias,
            peer_logits,
            peer_top1,
            &alias,
            rows,
            w0,
            w1,
        );
        let ab = read_guarded(gpu, &alias, rows);
        assert_guard(&ab, rows, "PM aliased out");
        let wb: Vec<u8> = want.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(
            &ab[..rows * 4],
            &wb[..],
            "PM aliased vs reference: w0 {w0} w1 {w1}"
        );
        let _ = gpu.free_tensor(alias);
    }
    for t in [t_full, t0, t1, want_t, top0, top1, out] {
        let _ = gpu.free_tensor(t);
    }
}

#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn pair_merge_equals_full_row_argmax() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut g = Lcg(0x71_5eed);
    // Small odd widths, more rows than one 256-row block, and the flash vocabulary.
    for (w0, w1, rows) in [
        (7usize, 9usize, 300usize),
        (1, 1, 40),
        (1023, 1025, 60),
        (4097, 3, 41),
        (124_160, 124_160, 40),
        // The exact EP vocabulary split and more than 256 rows.
        (50_176, 50_100, 300),
    ] {
        for _ in 0..2 {
            run_case(&mut gpu, &mut g, w0, w1, rows);
        }
        eprintln!("pair merge: widths ({w0}, {w1}) x {rows} rows == full argmax_f32");
    }
}

/// The merge refuses shapes it cannot run.
#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn pair_merge_refuses_bad_shapes() {
    let mut gpu = Gpu::init().expect("gpu init");
    let l = gpu.zeros(&[8 * 4], DType::F32).expect("l");
    let i = gpu.zeros(&[8 * 4], DType::Raw).expect("i");
    let peer = l.buf.as_ptr() as u64;
    let ok = |gpu: &mut Gpu, rows: usize, w0: usize, w1: usize, peer_top1: u64| {
        gpu.argmax_f32_pair_merge(&ArgmaxPairMerge {
            local_logits: &l,
            local_top1: &i,
            peer_logits: peer,
            peer_top1,
            out_top1: &i,
            rows,
            w0,
            w1,
        })
    };
    let peer_top1 = i.buf.as_ptr() as u64;
    assert!(ok(&mut gpu, 0, 4, 4, peer_top1).is_err(), "zero rows");
    assert!(ok(&mut gpu, 1, 0, 4, peer_top1).is_err(), "zero width");
    assert!(ok(&mut gpu, 1, 4, 4, 0).is_err(), "null peer");
    assert!(
        ok(&mut gpu, 9, 4, 4, peer_top1).is_err(),
        "rows beyond top1"
    );
    assert!(
        ok(&mut gpu, 8, 5, 4, peer_top1).is_err(),
        "logits too small"
    );
    assert!(ok(&mut gpu, 1, 4, 4, peer_top1).is_ok());
}

/// Rows with the planted maximum at global indices 50175 / 50176 / 50177 around the exact
/// width split (50176, 50100): single maxima, ties across the boundary, +-0, NaN / inf noise.
#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn pair_merge_exact_split_boundary_indices() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut g = Lcg(0xb0_0d);
    let (w0, w1) = (50_176usize, 50_100usize);
    let n = w0 + w1;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for &i in &[w0 - 1, w0, w0 + 1] {
        for variant in 0..6 {
            let mut r: Vec<f32> = (0..n).map(|_| -(1.0 + g.unit().abs() * 4.0)).collect();
            match variant {
                0 => r[i] = 10.0,
                1 => {
                    r[i] = 10.0;
                    r[w0 - 1] = 10.0;
                    r[w0] = 10.0;
                }
                2 => {
                    r[i] = 0.0;
                    r[w0 - 1] = -0.0;
                    r[w0] = -0.0;
                }
                3 => {
                    r[i] = 3.0;
                    r[0] = f32::NAN;
                    r[n - 1] = f32::INFINITY;
                }
                4 => {
                    r[w0 - 1] = f32::NAN;
                    r[w0] = f32::NEG_INFINITY;
                    r[i] = 2.0;
                    r[w0 + 1] = 2.0;
                }
                _ => {
                    r[i] = 5.0;
                    r[n - 1] = 5.0;
                    r[w0 - 2] = 5.0;
                }
            }
            rows.push(r);
        }
    }
    // More than one 256-row block of boundary-flavoured rows.
    while rows.len() < 290 {
        let f = rows.len() % FLAVOURS;
        rows.push(make_row(&mut g, w0, w1, f));
    }
    run_rows(&mut gpu, w0, w1, rows);
}
