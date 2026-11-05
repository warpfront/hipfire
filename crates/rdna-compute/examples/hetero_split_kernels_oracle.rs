//! Oracle for the gfx1201 side of the device-side hetero split
//! (`kernels/src/hetero_split.hip`). Run on the R9700 (`HIPFIRE_DEVICES=gfx1201`):
//! a pinned mailbox, with a host thread playing the Strix responder.
//!
//! Cases:
//! * (a) publish copies x/ids/weights into the slot and writes masked ids;
//! * (b) the responder answers;
//! * (c) await + install copy exactly the cold rows;
//! * (d) a timeout writes the error word, installs nothing and returns promptly;
//! * (e) zero cold slots still pair publish and await.
//!
//! Prints `PASS` or `FAIL <case>: ...`; exits 1 on failure.
//!
//! ```text
//! HIPFIRE_DEVICES=gfx1201 cargo run --release -p rdna-compute --features lab --example hetero_split_kernels_oracle
//! ```

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use rdna_compute::{DType, Gpu, GpuTensor, HeteroSlotPtrs};

const LAYERS: u32 = 48;
const N_EXP: usize = 512;
const TOP_K: usize = 10;
const DIM: usize = 2560;
const SENTINEL: f32 = -7.0;

fn seq_for(t: u32, l: u32, n: u32) -> u32 {
    match t.wrapping_mul(n).wrapping_add(l).wrapping_add(1) {
        0 => 1,
        v => v,
    }
}

struct Mailbox {
    base: *mut u8,
}

impl Mailbox {
    // Slot layout mirrored from hipfire_dispatch::hetero_mailbox (max_rows 1).
    fn slot(&self, layer: usize) -> HeteroSlotPtrs {
        let align = |x: usize| x.div_ceil(256) * 256;
        let x_rot = 256;
        let ids = align(x_rot + DIM * 4);
        let weights = align(ids + TOP_K * 4);
        let cold_out = align(weights + TOP_K * 4);
        let bytes = align(cold_out + TOP_K * DIM * 4);
        let b = self.base as u64 + (layer * bytes) as u64;
        HeteroSlotPtrs {
            req_seq: b,
            resp_seq: b + 64,
            error: b + 128,
            x_rot: b + x_rot as u64,
            ids: b + ids as u64,
            weights: b + weights as u64,
            cold_out: b + cold_out as u64,
        }
    }
}

fn word(addr: u64) -> &'static AtomicU32 {
    // SAFETY: a 4-byte aligned word inside the live pinned mailbox.
    unsafe { &*(addr as *const AtomicU32) }
}

fn upload_i32(gpu: &mut Gpu, v: &[i32]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[v.len()], DType::F32).unwrap();
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
    gpu.memcpy_htod_auto(&t.buf, &b).unwrap();
    t
}

fn upload_u8(gpu: &mut Gpu, v: &[u8]) -> GpuTensor {
    let t = gpu.alloc_tensor(&[v.len().div_ceil(4)], DType::F32).unwrap();
    gpu.memcpy_htod_auto(&t.buf, v).unwrap();
    t
}

fn download_i32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<i32> {
    let mut b = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).unwrap();
    b.chunks(4).map(|c| i32::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn fail(case: &str, msg: String) -> ! {
    println!("FAIL {case}: {msg}");
    std::process::exit(1);
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    assert_eq!(gpu.arch, "gfx1201", "run with HIPFIRE_DEVICES=gfx1201");
    let flags = hip_bridge::HIP_HOST_MALLOC_MAPPED | hip_bridge::HIP_HOST_MALLOC_COHERENT;
    let total = 3 * 256 * 1024;
    let base = gpu.hip.host_malloc(total, flags).unwrap().cast::<u8>();
    // SAFETY: freshly allocated pinned memory of `total` bytes.
    unsafe { std::ptr::write_bytes(base, 0, total) };
    let mb = Mailbox { base };

    let counter = upload_i32(&mut gpu, &[0]);
    let x: Vec<f32> = (0..DIM).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let x_t = gpu.upload_f32(&x, &[DIM]).unwrap();
    let ids: Vec<i32> = (0..TOP_K as i32).map(|s| 3 + 37 * s).collect();
    let ids_t = upload_i32(&mut gpu, &ids);
    let w: Vec<f32> = (0..TOP_K).map(|s| 0.1 * (s as f32 + 1.0)).collect();
    let w_t = gpu.upload_f32(&w, &[TOP_K]).unwrap();
    let mut bitmap = vec![0u8; N_EXP];
    for (s, &e) in ids.iter().enumerate() {
        if s % 3 != 0 {
            bitmap[e as usize] = 1; // slots 1,2,4,5,7,8 cold
        }
    }
    let bitmap_t = upload_u8(&mut gpu, &bitmap);
    let masked_t = gpu.alloc_tensor(&[TOP_K], DType::F32).unwrap();
    let expanded = gpu.upload_f32(&vec![SENTINEL; TOP_K * DIM], &[TOP_K * DIM]).unwrap();
    let pattern = |s: usize, i: usize| (s * 10_000 + i) as f32;

    // ---- (a)-(c): one round trip on layer 0.
    let s0 = mb.slot(0);
    gpu.hetero_token_advance(&counter).unwrap();
    let want = seq_for(1, 0, LAYERS);
    let responder = std::thread::spawn(move || {
        let t0 = Instant::now();
        while word(s0.req_seq).load(Ordering::Acquire) != want {
            if t0.elapsed() > Duration::from_secs(5) {
                return false;
            }
        }
        // SAFETY: cold_out holds TOP_K * DIM f32 inside the mailbox.
        let out = unsafe { std::slice::from_raw_parts_mut(s0.cold_out as *mut f32, TOP_K * DIM) };
        for s in 0..TOP_K {
            for i in 0..DIM {
                out[s * DIM + i] = pattern(s, i);
            }
        }
        word(s0.resp_seq).store(want, Ordering::Release);
        true
    });
    gpu.hetero_publish_cold(&counter, 0, LAYERS, &x_t, &ids_t, &w_t, &bitmap_t, N_EXP as i32, 1, TOP_K as i32, DIM as i32, &s0, &masked_t)
        .unwrap();
    gpu.hetero_await_cold(&counter, 0, LAYERS, &s0, 50_000_000).unwrap();
    gpu.hetero_install_cold(&s0, &ids_t, &bitmap_t, N_EXP as i32, TOP_K as u32, DIM as i32, &expanded).unwrap();
    gpu.hip.device_synchronize().unwrap();
    if !responder.join().unwrap() {
        fail("b", "responder never saw the request".into());
    }
    // SAFETY: mailbox fields of slot 0.
    let (mx, mi, mw) = unsafe {
        (
            std::slice::from_raw_parts(s0.x_rot as *const f32, DIM),
            std::slice::from_raw_parts(s0.ids as *const i32, TOP_K),
            std::slice::from_raw_parts(s0.weights as *const f32, TOP_K),
        )
    };
    if mx != &x[..] || mi != &ids[..] || mw != &w[..] {
        fail("a", "mailbox slot contents differ from the inputs".into());
    }
    let masked = download_i32(&gpu, &masked_t, TOP_K);
    let want_masked: Vec<i32> = ids.iter().map(|&e| if bitmap[e as usize] != 0 { -1 } else { e }).collect();
    if masked != want_masked {
        fail("a", format!("masked ids {masked:?} != {want_masked:?}"));
    }
    let got = gpu.download_f32(&expanded).unwrap();
    for s in 0..TOP_K {
        let cold = bitmap[ids[s] as usize] != 0;
        for i in 0..DIM {
            let want = if cold { pattern(s, i) } else { SENTINEL };
            if got[s * DIM + i].to_bits() != want.to_bits() {
                fail("c", format!("slot {s} elem {i}: {} != {want}", got[s * DIM + i]));
            }
        }
    }
    if word(s0.error).load(Ordering::Acquire) != 0 {
        fail("c", "error word set on a successful round trip".into());
    }

    // ---- (d): timeout on layer 1 (nobody answers).
    let s1 = mb.slot(1);
    let before = gpu.download_f32(&expanded).unwrap();
    let t = Instant::now();
    gpu.hetero_publish_cold(&counter, 1, LAYERS, &x_t, &ids_t, &w_t, &bitmap_t, N_EXP as i32, 1, TOP_K as i32, DIM as i32, &s1, &masked_t)
        .unwrap();
    gpu.hetero_await_cold(&counter, 1, LAYERS, &s1, 20_000).unwrap();
    gpu.hetero_install_cold(&s1, &ids_t, &bitmap_t, N_EXP as i32, TOP_K as u32, DIM as i32, &expanded).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let dt = t.elapsed();
    let err = word(s1.error).load(Ordering::Acquire);
    if err != seq_for(1, 1, LAYERS) {
        fail("d", format!("error word {err}, expected {}", seq_for(1, 1, LAYERS)));
    }
    if gpu.download_f32(&expanded).unwrap() != before {
        fail("d", "install wrote rows after a timeout".into());
    }
    if dt > Duration::from_millis(100) {
        fail("d", format!("timed-out wait took {dt:?}"));
    }

    // ---- (e): zero cold slots still pair (layer 2).
    let none = upload_u8(&mut gpu, &vec![0u8; N_EXP]);
    let s2 = mb.slot(2);
    let want2 = seq_for(1, 2, LAYERS);
    let r2 = std::thread::spawn(move || {
        let t0 = Instant::now();
        while word(s2.req_seq).load(Ordering::Acquire) != want2 {
            if t0.elapsed() > Duration::from_secs(5) {
                return false;
            }
        }
        word(s2.resp_seq).store(want2, Ordering::Release);
        true
    });
    gpu.hetero_publish_cold(&counter, 2, LAYERS, &x_t, &ids_t, &w_t, &none, N_EXP as i32, 1, TOP_K as i32, DIM as i32, &s2, &masked_t)
        .unwrap();
    gpu.hetero_await_cold(&counter, 2, LAYERS, &s2, 50_000_000).unwrap();
    gpu.hip.device_synchronize().unwrap();
    if !r2.join().unwrap() || word(s2.error).load(Ordering::Acquire) != 0 {
        fail("e", "zero-cold layer did not pair publish and await".into());
    }
    if download_i32(&gpu, &masked_t, TOP_K) != ids {
        fail("e", "zero-cold masked ids should equal the ids".into());
    }
    // ---- (f)-(h): prefill slot (separate pinned slot, max 64 rows; layout
    // mirrored from hetero_mailbox::PrefillGeometry).
    const PROWS: usize = 64;
    let al = |x: usize| x.div_ceil(256) * 256;
    let (p_x, p_ids) = (256usize, al(256 + PROWS * DIM * 4));
    let p_w = al(p_ids + PROWS * TOP_K * 4);
    let p_part = al(p_w + PROWS * TOP_K * 4);
    let p_total = al(p_part + PROWS * DIM * 4);
    let pb = gpu.hip.host_malloc(p_total, flags).unwrap().cast::<u8>();
    // SAFETY: fresh pinned allocation of `p_total` bytes.
    unsafe { std::ptr::write_bytes(pb, 0, p_total) };
    let pa = pb as u64;
    let ps = HeteroSlotPtrs {
        req_seq: pa,
        resp_seq: pa + 64,
        error: pa + 128,
        x_rot: pa + p_x as u64,
        ids: pa + p_ids as u64,
        weights: pa + p_w as u64,
        cold_out: pa + p_part as u64,
    };
    let rows = 37usize; // a partial last chunk
    let xs: Vec<f32> = (0..rows * DIM).map(|i| ((i % 113) as f32 - 56.0) * 0.003).collect();
    let xs_t = gpu.upload_f32(&xs, &[rows * DIM]).unwrap();
    let pids: Vec<i32> = (0..rows * TOP_K).map(|s| ((s * 29 + 3) % N_EXP) as i32).collect();
    let pids_t = upload_i32(&mut gpu, &pids);
    let pw: Vec<f32> = (0..rows * TOP_K).map(|s| (s % 7) as f32 * 0.05).collect();
    let pw_t = gpu.upload_f32(&pw, &[rows * TOP_K]).unwrap();
    let pmasked = gpu.alloc_tensor(&[PROWS * TOP_K], DType::F32).unwrap();
    let base_x: Vec<f32> = (0..rows * DIM).map(|i| (i % 17) as f32).collect();
    let x_batch = gpu.upload_f32(&base_x, &[rows * DIM]).unwrap();
    let part = |i: usize| (i % 1009) as f32 * 0.25;
    gpu.hetero_token_advance(&counter).unwrap(); // token 2
    let want_f = seq_for(2, 5, LAYERS);
    let rf = std::thread::spawn(move || {
        let t0 = Instant::now();
        while word(ps.req_seq).load(Ordering::Acquire) != want_f {
            if t0.elapsed() > Duration::from_secs(5) {
                return false;
            }
        }
        // SAFETY: the partial area holds PROWS * DIM f32.
        let out = unsafe { std::slice::from_raw_parts_mut(ps.cold_out as *mut f32, rows * DIM) };
        for (i, v) in out.iter_mut().enumerate() {
            *v = part(i);
        }
        word(ps.resp_seq).store(want_f, Ordering::Release);
        true
    });
    gpu.hetero_publish_rows(&xs_t, &pids_t, &pw_t, &bitmap_t, N_EXP as i32, rows as i32, TOP_K as i32, DIM as i32, &ps, &pmasked)
        .unwrap();
    gpu.hetero_release_rows(&counter, 5, LAYERS, rows as i32, pa + 192, pa + 196, ps.req_seq).unwrap();
    gpu.hetero_await_cold(&counter, 5, LAYERS, &ps, 50_000_000).unwrap();
    gpu.hetero_add_partial(ps.error, ps.cold_out, (rows * DIM) as i32, &x_batch).unwrap();
    gpu.hip.device_synchronize().unwrap();
    if !rf.join().unwrap() {
        fail("f", "prefill responder never saw the request".into());
    }
    // SAFETY: slot fields.
    let (gx, gl, gr) = unsafe {
        (
            std::slice::from_raw_parts(ps.x_rot as *const f32, rows * DIM),
            *((pa + 192) as *const u32),
            *((pa + 196) as *const u32),
        )
    };
    if gx != &xs[..] || gl != 5 || gr != rows as u32 {
        fail("f", format!("prefill slot contents (layer {gl}, rows {gr})"));
    }
    let want_pm: Vec<i32> = pids.iter().map(|&e| if bitmap[e as usize] != 0 { -1 } else { e }).collect();
    if download_i32(&gpu, &pmasked, rows * TOP_K) != want_pm {
        fail("f", "prefill masked ids".into());
    }
    let gotx = gpu.download_f32(&x_batch).unwrap();
    for i in 0..rows * DIM {
        if gotx[i].to_bits() != (base_x[i] + part(i)).to_bits() {
            fail("f", format!("x_batch[{i}] {} != {}", gotx[i], base_x[i] + part(i)));
        }
    }
    // (g) timeout: nobody answers layer 6.
    let before = gpu.download_f32(&x_batch).unwrap();
    // SAFETY: zero the partial so a wrong add would show.
    unsafe { std::ptr::write_bytes(ps.cold_out as *mut u8, 0x41, rows * DIM * 4) };
    gpu.hetero_publish_rows(&xs_t, &pids_t, &pw_t, &bitmap_t, N_EXP as i32, rows as i32, TOP_K as i32, DIM as i32, &ps, &pmasked)
        .unwrap();
    gpu.hetero_release_rows(&counter, 6, LAYERS, rows as i32, pa + 192, pa + 196, ps.req_seq).unwrap();
    gpu.hetero_await_cold(&counter, 6, LAYERS, &ps, 20_000).unwrap();
    gpu.hetero_add_partial(ps.error, ps.cold_out, (rows * DIM) as i32, &x_batch).unwrap();
    gpu.hip.device_synchronize().unwrap();
    if word(ps.error).load(Ordering::Acquire) != seq_for(2, 6, LAYERS) {
        fail("g", "prefill timeout did not set the error word".into());
    }
    if gpu.download_f32(&x_batch).unwrap() != before {
        fail("g", "add_partial ran after a timeout".into());
    }
    // (h) zero cold: masked == ids.
    gpu.hetero_publish_rows(&xs_t, &pids_t, &pw_t, &none, N_EXP as i32, rows as i32, TOP_K as i32, DIM as i32, &ps, &pmasked)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    if download_i32(&gpu, &pmasked, rows * TOP_K) != pids {
        fail("h", "zero-cold prefill masked ids should equal the ids".into());
    }
    let _ = gpu.hip.host_free(pb.cast());
    let _ = gpu.hip.host_free(base.cast());
    println!("PASS");
}
