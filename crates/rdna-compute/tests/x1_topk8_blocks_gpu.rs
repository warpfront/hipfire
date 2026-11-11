// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! X1 oracle: the draft-head top-8 reduction split into block ranges equals the
//! one 128-group launch.
//!
//! `Gpu::topk8_partial_blocks` (kernel `topk8_partial_f32_blocks`) over
//! `[0, 64)` and `[64, 128)` (and the reverse order, and 128 one-block launches)
//! must leave the partial buffer (pv then pi, all 8192 bytes) byte-identical to
//! the `topk8_partial_f32` grid-128 launch inside `Gpu::topk8_rescore_k2560`;
//! `Gpu::topk8_rescore_only` over that buffer must write the same 72-byte slot as
//! `topk8_rescore_k2560`.
//!
//! n = 100,276 (the front class; chunk 784, block-64 boundary at layout row 50,176)
//! and n = 248,320 (the full-hold class; chunk 1,940, boundary 124,160); identity,
//! front/special/tail orders; random, BF16-tie, five-level, constant, and rows with
//! equal top values straddling the block-64 boundary (rows 64*chunk-1 | 64*chunk),
//! plus -inf padding and NaN. Re-score is checked on a Q8_0 head and an MQ6G256V2
//! head of the n = 100,276 class (head bytes are pseudo-random; both arms read the
//! same bytes, so only the launch split can differ).
//!
//! `pm_partial_blocks_twin_is_byte_identical_to_hipcc` launches the PeaceMaker twin
//! `topk8_partial_f32_blocks_pm_gfx1201` straight from `kernels/topk8_partial_blocks_pm_gfx1201.hxaco`
//! (exact gfx1201 only; other devices skip) and compares every pv/pi word, with guard bytes, against
//! the hipcc `topk8_partial_f32` grid-128 launch and the hipcc `topk8_partial_f32_blocks` launches of
//! the same block plan: n = 5 | 1,000 | 100,276 | 248,320 | 262,144, the 64+64 / reverse / 128-singles
//! / subset plans and the packed-halves shard form, over finite, tied, constant, signed-zero, NaN,
//! +/-inf, all-non-finite (rows and chunks), random-bit-pattern rows and ties / zeros / infinities
//! around layout row 64 * chunk (50,176 for n = 100,276). Partial buffers only: no re-score
//! consumes the sentinel token ids that all-NaN groups produce.
//!
//! `#[ignore]`d: needs a HIP GPU. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --test x1_topk8_blocks_gpu -- --ignored --nocapture

use rdna_compute::gemv::TOPK8_GROUPS;
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
}

fn bf16(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

const FLAVOURS: usize = 9;

fn make_row(g: &mut Lcg, n: usize, f: usize) -> Vec<f32> {
    let chunk = n.div_ceil(TOPK8_GROUPS);
    let edge = 64 * chunk; // first layout row of group 64
    let mut r: Vec<f32> = (0..n).map(|_| g.unit() * 8.0).collect();
    match f {
        0 => {}
        1 => r = (0..n).map(|_| bf16(g.unit() * 14.0)).collect(),
        2 => r = (0..n).map(|_| (g.next() % 5) as f32).collect(),
        3 => r = vec![1.5; n],
        4 => {
            // Equal top values straddling the block-64 boundary.
            for v in &mut r {
                *v = -1.0 - v.abs();
            }
            r[edge - 1] = 20.0;
            r[edge] = 20.0;
        }
        5 => {
            // Many equal values on both sides of the boundary: eight tied winners
            // on each side, so the top-8 of each group is a pure tie-break.
            for v in &mut r {
                *v = -1.0 - v.abs();
            }
            for k in 0..8 {
                r[edge - 1 - k] = 20.0;
                r[edge + k] = 20.0;
            }
        }
        6 => {
            // Fewer than 8 finite values per group (-inf padding), NaN planted.
            r = vec![f32::NEG_INFINITY; n];
            for _ in 0..40 {
                let i = g.next() as usize % n;
                r[i] = bf16(g.unit());
            }
            r[edge] = f32::NAN;
            r[edge - 1] = 3.0;
        }
        7 => {
            for v in &mut r {
                *v = -1.0 - v.abs();
            }
            r[edge - 1] = 20.0;
            r[edge] = 20.0;
            r[edge + 1] = 20.0;
            r[0] = 20.0;
            r[n - 1] = 20.0;
        }
        _ => {
            // +inf / -inf mixed with finite values.
            r[edge] = f32::INFINITY;
            r[edge - 1] = f32::INFINITY;
            r[edge + 7] = f32::NEG_INFINITY;
        }
    }
    r
}

fn bytes_of(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.download_raw_bytes(t).expect("download")
}

fn raw(gpu: &mut Gpu, bytes: usize) -> GpuTensor {
    gpu.zeros(&[bytes], DType::Raw).expect("alloc raw")
}

/// A pseudo-random Q8_0 head of `rows` rows at K = 2560 (80 blocks of 34 bytes:
/// an f16 scale near 2^-7, then 32 int8).
fn q8_head(gpu: &mut Gpu, g: &mut Lcg, rows: usize) -> GpuTensor {
    let mut b = Vec::with_capacity(rows * 80 * 34);
    for _ in 0..rows * 80 {
        let s = 0x2000u16 + (g.next() & 0x3ff) as u16;
        b.extend_from_slice(&s.to_le_bytes());
        for _ in 0..32 {
            b.push(g.next() as u8);
        }
    }
    let mut t = gpu.upload_raw(&b, &[b.len()]).expect("q8 head");
    t.dtype = DType::Q8_0;
    t
}

/// A pseudo-random MQ6G256V2 head of `rows` rows at K = 2560 (10 groups of 200 bytes).
fn mq6_head(gpu: &mut Gpu, g: &mut Lcg, rows: usize) -> GpuTensor {
    let b: Vec<u8> = (0..rows * 10 * 200).map(|_| g.next() as u8).collect();
    let mut t = gpu.upload_raw(&b, &[b.len()]).expect("mq6 head");
    t.dtype = DType::MQ6G256V2;
    t
}

fn sentinel_partial(gpu: &mut Gpu) -> GpuTensor {
    let t = raw(gpu, Gpu::TOPK8_PARTIAL_BYTES);
    gpu.hip
        .memset(&t.buf, 0xAB, Gpu::TOPK8_PARTIAL_BYTES)
        .expect("sentinel");
    t
}

struct Heads {
    q8: Option<GpuTensor>,
    mq6: Option<GpuTensor>,
    x: GpuTensor,
}

fn run_case(
    gpu: &mut Gpu,
    g: &mut Lcg,
    n: usize,
    order: (usize, usize, usize),
    flavour: usize,
    heads: &Heads,
) {
    let data = make_row(g, n, flavour);
    let logits = gpu.upload_f32(&data, &[n]).expect("logits");
    let out_bytes = Gpu::TOPK8_OUT_BYTES;
    // Reference partial: the grid-128 launch inside topk8_rescore_k2560.
    let heads_to_run: Vec<(&str, &GpuTensor)> = [("q8", &heads.q8), ("mq6", &heads.mq6)]
        .into_iter()
        .filter_map(|(name, h)| h.as_ref().map(|h| (name, h)))
        .collect();
    // The partial comparison needs no head: any head the rescore accepts will do.
    let head0 = heads_to_run
        .first()
        .map(|(_, h)| *h)
        .expect("at least one head");
    let ctx = format!("n {n} order {order:?} flavour {flavour}");

    let p_ref = sentinel_partial(gpu);
    let out_ref = raw(gpu, out_bytes);
    gpu.topk8_rescore_k2560(&logits, n, order, head0, &heads.x, &p_ref, &out_ref)
        .expect("full rescore");
    gpu.hip.device_synchronize().expect("sync");
    let want_partial = bytes_of(gpu, &p_ref);
    assert_eq!(want_partial.len(), Gpu::TOPK8_PARTIAL_BYTES);
    assert!(
        want_partial.iter().any(|&b| b != 0xAB),
        "reference wrote nothing: {ctx}"
    );

    let half = TOPK8_GROUPS / 2;
    let splits: [(&str, Vec<(usize, usize)>); 3] = [
        ("0-64 then 64-128", vec![(0, half), (half, half)]),
        ("64-128 then 0-64", vec![(half, half), (0, half)]),
        (
            "128 single blocks",
            (0..TOPK8_GROUPS).map(|b| (b, 1)).collect(),
        ),
    ];
    for (name, launches) in &splits {
        let p = sentinel_partial(gpu);
        for &(block0, blocks) in launches {
            gpu.topk8_partial_blocks(&logits, 0, n, order, &p, block0, blocks)
                .expect("partial blocks");
        }
        gpu.hip.device_synchronize().expect("sync");
        assert_eq!(
            bytes_of(gpu, &p),
            want_partial,
            "partial pv/pi differ ({name}): {ctx}"
        );
        // Re-score over the split-built partial.
        for (hname, head) in &heads_to_run {
            let out_full = raw(gpu, out_bytes);
            gpu.topk8_rescore_k2560(&logits, n, order, head, &heads.x, &p_ref, &out_full)
                .expect("full rescore");
            let out_split = raw(gpu, out_bytes);
            gpu.topk8_rescore_only(&p, head, &heads.x, &out_split)
                .expect("rescore only");
            gpu.hip.device_synchronize().expect("sync");
            assert_eq!(
                bytes_of(gpu, &out_split),
                bytes_of(gpu, &out_full),
                "rescore slot differs ({name}, {hname} head): {ctx}"
            );
            let _ = gpu.free_tensor(out_full);
            let _ = gpu.free_tensor(out_split);
        }
        let _ = gpu.free_tensor(p);
    }
    // The vocabulary-shard form: each half is a PACKED buffer (rank 0 rows [0, split), rank 1
    // rows [split, n)) launched with `row0` = its first layout row; the partial equals the full
    // launch's byte for byte.
    {
        let chunk = n.div_ceil(TOPK8_GROUPS);
        let split = (TOPK8_GROUPS / 2 * chunk).min(n);
        let lo = gpu.upload_f32(&data[..split], &[split]).expect("lo half");
        let hi = gpu
            .upload_f32(&data[split..], &[n - split])
            .expect("hi half");
        let p = sentinel_partial(gpu);
        gpu.topk8_partial_blocks(&lo, 0, n, order, &p, 0, TOPK8_GROUPS / 2)
            .expect("lo half blocks");
        gpu.topk8_partial_blocks(&hi, split, n, order, &p, TOPK8_GROUPS / 2, TOPK8_GROUPS / 2)
            .expect("hi half blocks (row0 = 64 * chunk)");
        gpu.hip.device_synchronize().expect("sync");
        assert_eq!(
            bytes_of(gpu, &p),
            want_partial,
            "packed halves (row0 = {split}) differ from the full launch: {ctx}"
        );
        for t in [lo, hi, p] {
            let _ = gpu.free_tensor(t);
        }
    }
    for t in [logits, p_ref, out_ref] {
        let _ = gpu.free_tensor(t);
    }
}

#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn partial_blocks_and_rescore_only_equal_the_full_launch() {
    let mut gpu = Gpu::init().expect("gpu init");
    let mut g = Lcg(0x70_9c8);
    let xs: Vec<f32> = (0..2560).map(|_| g.unit()).collect();
    let x = gpu.upload_f32(&xs, &[2560]).expect("x");
    // n = 100,276: heads for both dtypes (token ids stay below n).
    {
        let n = 100_276usize;
        let heads = Heads {
            q8: Some(q8_head(&mut gpu, &mut g, n)),
            mq6: Some(mq6_head(&mut gpu, &mut g, n)),
            x: x.sub_offset(0, 2560),
        };
        // Orders that keep every token id below n: identity, and a front/special/tail
        // re-map of the same row range.
        for order in [(n, 0usize, 0usize), (90_000, 100_000, 276), (0, 0, 0)] {
            for flavour in 0..FLAVOURS {
                run_case(&mut gpu, &mut g, n, order, flavour, &heads);
            }
        }
        eprintln!("topk8 blocks: n {n} byte-identical (partial + rescore, Q8_0 + MQ6G256V2)");
        for h in [heads.q8, heads.mq6].into_iter().flatten() {
            let _ = gpu.free_tensor(h);
        }
    }
    // n = 248,320: partial blocks only need a rescore-capable head for the reference
    // launch; a Q8_0 head of the full vocabulary covers every token id the orders
    // below can produce.
    {
        let n = 248_320usize;
        let heads = Heads {
            q8: Some(q8_head(&mut gpu, &mut g, n)),
            mq6: None,
            x: x.sub_offset(0, 2560),
        };
        for order in [
            (n, 0usize, 0usize),
            (100_000, 248_044, 276),
            (124_000, 248_000, 320),
        ] {
            for flavour in 0..FLAVOURS {
                run_case(&mut gpu, &mut g, n, order, flavour, &heads);
            }
        }
        eprintln!("topk8 blocks: n {n} byte-identical (partial + rescore, Q8_0)");
        if let Some(h) = heads.q8 {
            let _ = gpu.free_tensor(h);
        }
    }
}

#[test]
#[ignore = "needs a GPU: run through the GPU fabric"]
fn partial_blocks_refuse_bad_shapes() {
    let mut gpu = Gpu::init().expect("gpu init");
    let n = 1000usize;
    let logits = gpu.zeros(&[n], DType::F32).expect("logits");
    let partial = raw(&mut gpu, Gpu::TOPK8_PARTIAL_BYTES);
    let small = raw(&mut gpu, Gpu::TOPK8_PARTIAL_BYTES - 1);
    let id = (n, 0usize, 0usize);
    assert!(
        gpu.topk8_partial_blocks(&logits, 0, n, id, &partial, 0, 0)
            .is_err(),
        "no blocks"
    );
    assert!(
        gpu.topk8_partial_blocks(&logits, 0, n, id, &partial, 100, 29)
            .is_err(),
        "past group 128"
    );
    assert!(
        gpu.topk8_partial_blocks(&logits, 0, n, id, &small, 0, 1)
            .is_err(),
        "small partial"
    );
    // n = 1,300 (chunk 11): groups 100..128 read rows [1100, 1300), past the 1,000-row view.
    assert!(
        gpu.topk8_partial_blocks(&logits, 0, 1300, (1300, 0, 0), &partial, 100, 28)
            .is_err(),
        "launched rows beyond the logits view"
    );
    // A view that starts at layout row 1,000 holds none of group 0's rows [0, 8).
    assert!(
        gpu.topk8_partial_blocks(&logits, 1000, n, id, &partial, 0, 1)
            .is_err(),
        "launched rows before row0"
    );
    // ... but group 125 (rows [1000, 1000), past n) is empty and needs no rows at all.
    assert!(gpu
        .topk8_partial_blocks(&logits, 1000, n, id, &partial, 125, 1)
        .is_ok());
    let big = 128 * 2049;
    let big_logits = gpu.zeros(&[big], DType::F32).expect("big logits");
    assert!(
        gpu.topk8_partial_blocks(&big_logits, 0, big, (big, 0, 0), &partial, 0, 1)
            .is_err(),
        "chunk beyond 2048"
    );
    assert!(gpu
        .topk8_partial_blocks(&logits, 0, n, id, &partial, 127, 1)
        .is_ok());
}

// ---- PeaceMaker twin ---------------------------------------------------------------
//
// `topk8_partial_f32_blocks_pm_gfx1201` (module `topk8_partial_blocks_pm_gfx1201`,
// `kernels/topk8_partial_blocks_pm_gfx1201.hxaco`) is launched directly from the bundle (no
// production routing) and its pv then pi bytes are compared word for word against the hipcc
// `topk8_partial_f32` grid-128 launch and against the hipcc `topk8_partial_f32_blocks` launched
// through `Gpu::topk8_partial_blocks` with the same block plan. These cases are partial-only: the
// all-NaN / no-candidate rows produce sentinel token ids (`INT_MAX - tail`) that no re-score may
// consume.

const PM_MODULE: &str = "topk8_partial_blocks_pm_gfx1201";
const PM_SYMBOL: &str = "topk8_partial_f32_blocks_pm_gfx1201";
/// Poison bytes on both sides of the PM arm's partial buffer.
const PM_GUARD: usize = 256;
const PM_FLAVOURS: usize = 14;

struct Pm {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
}

/// The PM image, or `None` (with a note) on an architecture other than gfx1201.
fn load_pm(gpu: &Gpu) -> Option<Pm> {
    if gpu.arch != "gfx1201" {
        eprintln!(
            "skip: {PM_MODULE} is an exact-gfx1201 image and this device is {}",
            gpu.arch
        );
        return None;
    }
    let path = format!(
        "{}/../../kernels/{PM_MODULE}.hxaco",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("{path}: {e} (emit it: peacemaker-isa emit --kernel topk8_partial_blocks_pm --arch gfx1201)")
    });
    let module = gpu.hip.module_load_data(&bytes).expect("load the PM image");
    let func = gpu
        .hip
        .module_get_function(&module, PM_SYMBOL)
        .expect("PM symbol");
    Some(Pm {
        _module: module,
        func,
    })
}

fn put(r: &mut [f32], i: usize, v: f32) {
    if i < r.len() {
        r[i] = v;
    }
}

/// Rows for the PM comparison: finite, ties, constants, signed zeros, all-non-finite rows and
/// chunks, raw random bit patterns, and ties / signed zeros around layout row `64 * chunk`
/// (50,176 for n = 100,276: rows 50,175 | 50,176 | 50,177).
fn make_pm_row(g: &mut Lcg, n: usize, f: usize) -> Vec<f32> {
    let chunk = n.div_ceil(TOPK8_GROUPS);
    let edge = (64 * chunk).min(n.saturating_sub(2)).max(1);
    let below = |g: &mut Lcg| -1.0 - (g.unit() * 8.0).abs();
    match f {
        0 => (0..n).map(|_| g.unit() * 8.0).collect(),
        1 => (0..n).map(|_| (g.next() % 5) as f32).collect(),
        2 => vec![1.5; n],
        3 => (0..n)
            .map(|_| match g.next() % 4 {
                0 | 2 => -0.0,
                1 => 0.0,
                _ => -1.0,
            })
            .collect(),
        4 => vec![f32::NAN; n],
        5 => vec![f32::NEG_INFINITY; n],
        6 => vec![f32::INFINITY; n],
        7 => (0..n)
            .map(|_| f32::from_bits((g.next() << 1) ^ g.next()))
            .collect(),
        8 => (0..n)
            .map(|_| match g.next() % 7 {
                0 => f32::NAN,
                1 => f32::NEG_INFINITY,
                2 => f32::INFINITY,
                3 => -0.0,
                4 => 0.0,
                _ => g.unit() * 8.0,
            })
            .collect(),
        9 => {
            // Whole chunks of one non-finite kind between finite chunks.
            let mut r: Vec<f32> = (0..n).map(|_| g.unit() * 8.0).collect();
            for k in 0..TOPK8_GROUPS {
                let (lo, hi) = ((k * chunk).min(n), ((k + 1) * chunk).min(n));
                for (t, x) in r[lo..hi].iter_mut().enumerate() {
                    match k % 5 {
                        0 => *x = f32::NAN,
                        1 => *x = f32::NEG_INFINITY,
                        2 => *x = f32::INFINITY,
                        3 => *x = if t % 2 == 0 { -0.0 } else { 0.0 },
                        _ => {}
                    }
                }
            }
            r
        }
        10 => {
            // Eight tied winners on each side of the boundary (edge - 1 and edge + 1 included).
            let mut r: Vec<f32> = (0..n).map(|_| below(g)).collect();
            for k in 0..8 {
                if edge > k {
                    put(&mut r, edge - 1 - k, 20.0);
                }
                put(&mut r, edge + k, 20.0);
            }
            r
        }
        11 => {
            // Zeros of both signs around the boundary win over negatives: raw bits must survive.
            let mut r: Vec<f32> = (0..n).map(|_| below(g)).collect();
            put(&mut r, edge - 1, 0.0);
            put(&mut r, edge, -0.0);
            put(&mut r, edge + 1, 0.0);
            if edge > 1 {
                put(&mut r, edge - 2, -0.0);
            }
            r
        }
        12 => {
            // +inf allowed and tied, -inf below, NaN planted next to the boundary.
            let mut r: Vec<f32> = (0..n).map(|_| g.unit() * 8.0).collect();
            put(&mut r, edge, f32::INFINITY);
            put(&mut r, edge - 1, f32::INFINITY);
            put(&mut r, edge + 1, f32::INFINITY);
            put(&mut r, edge + 7, f32::NEG_INFINITY);
            if edge > 1 {
                put(&mut r, edge - 2, f32::NAN);
            }
            r
        }
        _ => {
            // Fewer than eight candidates per group: -inf padding with signed zeros, a NaN, and
            // a handful of finite values.
            let mut r = vec![f32::NEG_INFINITY; n];
            for _ in 0..60 {
                let i = g.next() as usize % n;
                r[i] = match g.next() % 3 {
                    0 => -0.0,
                    1 => 0.0,
                    _ => bf16(g.unit()),
                };
            }
            put(&mut r, edge, f32::NAN);
            put(&mut r, edge - 1, 3.0);
            r
        }
    }
}

/// The kernel arguments of `topk8_partial_f32` (8 arguments) or, with `block0`,
/// `topk8_partial_f32_blocks` / the PM twin (9 arguments).
fn partial_kernargs(
    v: *mut std::ffi::c_void,
    n: usize,
    order: (usize, usize, usize),
    pv: *mut std::ffi::c_void,
    pi: *mut std::ffi::c_void,
    block0: Option<usize>,
) -> hip_bridge::KernargBlob {
    let mut b = hip_bridge::KernargBlob::new();
    b.push_ptr(v);
    b.push_i32(n as i32);
    b.push_i32(n.div_ceil(TOPK8_GROUPS) as i32);
    b.push_ptr(pv);
    b.push_ptr(pi);
    b.push_i32(order.0 as i32);
    b.push_i32(order.1 as i32);
    b.push_i32(order.2 as i32);
    if let Some(b0) = block0 {
        b.push_i32(b0 as i32);
    }
    b
}

/// The hipcc `topk8_partial_f32` grid-128 launch into a fresh sentinel buffer.
fn hip_full(gpu: &mut Gpu, logits: &GpuTensor, n: usize, order: (usize, usize, usize)) -> Vec<u8> {
    let p = sentinel_partial(gpu);
    gpu.ensure_kernel_public(
        "qwen4_gemv_q8_0",
        rdna_compute::kernels::QWEN4_GEMV_Q8_0_SRC,
        "topk8_partial_f32",
    )
    .expect("compile topk8_partial_f32");
    let pv = p.buf.as_ptr();
    let pi = unsafe { (pv as *mut u8).add(TOPK8_GROUPS * 8 * 4) } as *mut std::ffi::c_void;
    let mut blob = partial_kernargs(logits.buf.as_ptr(), n, order, pv, pi, None);
    assert_eq!(blob.len(), 44);
    gpu.launch_kernel_blob(
        "topk8_partial_f32",
        [TOPK8_GROUPS as u32, 1, 1],
        [256, 1, 1],
        0,
        blob.as_mut_slice(),
    )
    .expect("launch topk8_partial_f32");
    gpu.hip.device_synchronize().expect("sync");
    let out = bytes_of(gpu, &p);
    let _ = gpu.free_tensor(p);
    out
}

/// The hipcc `topk8_partial_f32_blocks` launches of `plan` (through `Gpu::topk8_partial_blocks`)
/// into a fresh sentinel buffer.
fn hip_blocks(
    gpu: &mut Gpu,
    logits: &GpuTensor,
    row0: usize,
    n: usize,
    order: (usize, usize, usize),
    plan: &[(usize, usize)],
) -> Vec<u8> {
    let p = sentinel_partial(gpu);
    for &(block0, blocks) in plan {
        gpu.topk8_partial_blocks(logits, row0, n, order, &p, block0, blocks)
            .expect("hipcc partial blocks");
    }
    gpu.hip.device_synchronize().expect("sync");
    let out = bytes_of(gpu, &p);
    let _ = gpu.free_tensor(p);
    out
}

/// The PM twin's launches of `plan` over `v` (a pointer to layout row 0: rebased for a packed
/// shard) into a poisoned buffer with guard bytes on both sides; returns pv then pi (8192 bytes)
/// after checking the guards.
fn pm_blocks(
    gpu: &mut Gpu,
    pm: &Pm,
    v: *mut std::ffi::c_void,
    n: usize,
    order: (usize, usize, usize),
    plan: &[(usize, usize)],
    ctx: &str,
) -> Vec<u8> {
    let total = 2 * PM_GUARD + Gpu::TOPK8_PARTIAL_BYTES;
    let buf = raw(gpu, total);
    gpu.hip.memset(&buf.buf, 0xAB, total).expect("poison");
    let base = buf.buf.as_ptr() as *mut u8;
    let pv = unsafe { base.add(PM_GUARD) } as *mut std::ffi::c_void;
    let pi = unsafe { base.add(PM_GUARD + TOPK8_GROUPS * 8 * 4) } as *mut std::ffi::c_void;
    gpu.hip
        .device_synchronize()
        .expect("sync before the PM launch");
    for &(block0, blocks) in plan {
        let mut blob = partial_kernargs(v, n, order, pv, pi, Some(block0));
        assert_eq!(blob.len(), 48, "the nine-argument ABI is 48 bytes");
        // SAFETY: the kernarg bytes follow the twin's nine-argument ABI (v@0 n@8 chunk@12 pv@16
        // pi@24 front@32 special@36 tail@40 block0@44); every pointer addresses a live buffer.
        unsafe {
            gpu.hip
                .launch_kernel_blob(
                    &pm.func,
                    [blocks as u32, 1, 1],
                    [256, 1, 1],
                    0,
                    None,
                    blob.as_mut_slice(),
                )
                .expect("PM launch");
        }
        gpu.hip
            .device_synchronize()
            .expect("sync after the PM launch");
    }
    let all = bytes_of(gpu, &buf);
    assert_eq!(all.len(), total);
    assert!(
        all[..PM_GUARD].iter().all(|&b| b == 0xAB),
        "PM wrote before pv ({ctx})"
    );
    assert!(
        all[PM_GUARD + Gpu::TOPK8_PARTIAL_BYTES..]
            .iter()
            .all(|&b| b == 0xAB),
        "PM wrote past pi ({ctx})"
    );
    let out = all[PM_GUARD..PM_GUARD + Gpu::TOPK8_PARTIAL_BYTES].to_vec();
    let _ = gpu.free_tensor(buf);
    out
}

/// Word-for-word equality of two partial buffers (pv words then pi words), naming the first
/// difference.
fn same_partial(got: &[u8], want: &[u8], what: &str, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length ({ctx})");
    let words = TOPK8_GROUPS * 8;
    for k in 0..got.len() / 4 {
        let g = u32::from_le_bytes(got[4 * k..4 * k + 4].try_into().unwrap());
        let w = u32::from_le_bytes(want[4 * k..4 * k + 4].try_into().unwrap());
        assert_eq!(
            g,
            w,
            "{what}: {} word {} (group {} round {}) got {g:#010x} want {w:#010x} ({ctx})",
            if k < words { "pv" } else { "pi" },
            k % words,
            (k % words) / 8,
            k % 8
        );
    }
}

fn pm_case(
    gpu: &mut Gpu,
    pm: &Pm,
    g: &mut Lcg,
    n: usize,
    order: (usize, usize, usize),
    flavour: usize,
) {
    let data = make_pm_row(g, n, flavour);
    let logits = gpu.upload_f32(&data, &[n]).expect("logits");
    let ctx = format!("n {n} order {order:?} flavour {flavour}");
    let half = TOPK8_GROUPS / 2;
    let want_full = hip_full(gpu, &logits, n, order);
    assert!(
        want_full.iter().any(|&b| b != 0xAB),
        "the hipcc full launch wrote nothing: {ctx}"
    );
    // (name, launches, covers all 128 groups)
    let plans: [(&str, Vec<(usize, usize)>, bool); 7] = [
        ("one launch of 128 blocks", vec![(0, TOPK8_GROUPS)], true),
        ("0-64 then 64-128", vec![(0, half), (half, half)], true),
        ("64-128 then 0-64", vec![(half, half), (0, half)], true),
        (
            "128 single blocks",
            (0..TOPK8_GROUPS).map(|b| (b, 1)).collect(),
            true,
        ),
        ("blocks 37..48", vec![(37, 11)], false),
        ("block 127", vec![(TOPK8_GROUPS - 1, 1)], false),
        ("blocks 64..128", vec![(half, half)], false),
    ];
    let v = logits.buf.as_ptr();
    for (name, plan, covers) in &plans {
        let hip = hip_blocks(gpu, &logits, 0, n, order, plan);
        let pmb = pm_blocks(gpu, pm, v, n, order, plan, &format!("{name}: {ctx}"));
        same_partial(&pmb, &hip, &format!("PM vs hipcc blocks ({name})"), &ctx);
        if *covers {
            same_partial(
                &hip,
                &want_full,
                &format!("hipcc blocks vs full ({name})"),
                &ctx,
            );
            same_partial(
                &pmb,
                &want_full,
                &format!("PM vs hipcc full ({name})"),
                &ctx,
            );
        }
    }
    // The vocabulary-shard form: each half is a PACKED buffer launched with `row0` = its first
    // layout row; the PM twin gets the same rebased pointer the hipcc launcher passes.
    // (Skipped when the upper half is empty: n <= 64 * chunk.)
    if n > half * n.div_ceil(TOPK8_GROUPS) {
        let chunk = n.div_ceil(TOPK8_GROUPS);
        let split = (half * chunk).min(n);
        let lo = gpu.upload_f32(&data[..split], &[split]).expect("lo half");
        let hi = gpu
            .upload_f32(&data[split..], &[n - split])
            .expect("hi half");
        let hip = {
            let p = sentinel_partial(gpu);
            gpu.topk8_partial_blocks(&lo, 0, n, order, &p, 0, half)
                .expect("hipcc lo half");
            gpu.topk8_partial_blocks(&hi, split, n, order, &p, half, half)
                .expect("hipcc hi half");
            gpu.hip.device_synchronize().expect("sync");
            let out = bytes_of(gpu, &p);
            let _ = gpu.free_tensor(p);
            out
        };
        same_partial(&hip, &want_full, "hipcc packed halves vs full", &ctx);
        let lo_ptr = lo.buf.as_ptr();
        let hi_ptr = (hi.buf.as_ptr() as *mut u8).wrapping_sub(split * 4) as *mut std::ffi::c_void;
        let guard_total = 2 * PM_GUARD + Gpu::TOPK8_PARTIAL_BYTES;
        let buf = raw(gpu, guard_total);
        gpu.hip.memset(&buf.buf, 0xAB, guard_total).expect("poison");
        let base = buf.buf.as_ptr() as *mut u8;
        let pv = unsafe { base.add(PM_GUARD) } as *mut std::ffi::c_void;
        let pi = unsafe { base.add(PM_GUARD + TOPK8_GROUPS * 8 * 4) } as *mut std::ffi::c_void;
        gpu.hip.device_synchronize().expect("sync");
        for (ptr, block0) in [(lo_ptr, 0usize), (hi_ptr, half)] {
            let mut blob = partial_kernargs(ptr, n, order, pv, pi, Some(block0));
            assert_eq!(blob.len(), 48);
            // SAFETY: nine-argument ABI as above; the rebased `hi_ptr` is only dereferenced at
            // layout rows of groups [64, 128), which lie inside the packed upper half.
            unsafe {
                gpu.hip
                    .launch_kernel_blob(
                        &pm.func,
                        [half as u32, 1, 1],
                        [256, 1, 1],
                        0,
                        None,
                        blob.as_mut_slice(),
                    )
                    .expect("PM shard launch");
            }
            gpu.hip.device_synchronize().expect("sync");
        }
        let all = bytes_of(gpu, &buf);
        assert!(
            all[..PM_GUARD].iter().all(|&b| b == 0xAB),
            "PM shard wrote before pv ({ctx})"
        );
        assert!(
            all[PM_GUARD + Gpu::TOPK8_PARTIAL_BYTES..]
                .iter()
                .all(|&b| b == 0xAB),
            "PM shard wrote past pi ({ctx})"
        );
        same_partial(
            &all[PM_GUARD..PM_GUARD + Gpu::TOPK8_PARTIAL_BYTES],
            &want_full,
            "PM packed halves vs hipcc full",
            &ctx,
        );
        for t in [lo, hi, buf] {
            let _ = gpu.free_tensor(t);
        }
    }
    let _ = gpu.free_tensor(logits);
}

#[test]
#[ignore = "needs a gfx1201 GPU and kernels/topk8_partial_blocks_pm_gfx1201.hxaco: run through the GPU fabric"]
fn pm_partial_blocks_twin_is_byte_identical_to_hipcc() {
    let mut gpu = Gpu::init().expect("gpu init");
    let Some(pm) = load_pm(&gpu) else {
        return;
    };
    let mut g = Lcg(0x70_9c9);
    // (n, token-map orders). n = 5 and 1,000 leave most groups empty (sentinel rows); 262,144 is
    // chunk 2,048, the widest the kernel takes; the two production vocabularies put the 64-chunk
    // boundary at layout rows 50,176 and 124,160.
    let cases: [(usize, [(usize, usize, usize); 3]); 5] = [
        (5, [(5, 0, 0), (0, 0, 0), (2, 3, 1)]),
        (1_000, [(1_000, 0, 0), (0, 0, 0), (300, 500, 200)]),
        (
            100_276,
            [(100_276, 0, 0), (90_000, 100_000, 276), (0, 0, 0)],
        ),
        (
            248_320,
            [
                (248_320, 0, 0),
                (100_000, 248_044, 276),
                (124_000, 248_000, 320),
            ],
        ),
        (
            262_144,
            [(262_144, 0, 0), (100_000, 200_000, 4_000), (0, 0, 0)],
        ),
    ];
    for (n, orders) in cases {
        for order in orders {
            for flavour in 0..PM_FLAVOURS {
                pm_case(&mut gpu, &pm, &mut g, n, order, flavour);
            }
        }
        eprintln!("topk8 PM twin: n {n} byte-identical (full, 64+64, reverse, 128 singles, subsets, packed halves)");
    }
}
