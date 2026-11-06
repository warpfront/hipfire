//! Oracle for the gfx1201 side of the device-side hetero split
//! (`kernels/src/hetero_split.hip`). Run on the R9700 (`HIPFIRE_DEVICES=gfx1201`):
//! a pinned mailbox, with a host thread playing the Strix responder.
//!
//! Cases:
//! * (a) publish copies x/ids/weights into the slot and writes masked ids;
//! * (b) the responder answers;
//! * (c) await + install copy exactly the cold rows;
//! * (d) a timeout writes the error word, installs nothing and returns promptly;
//! * (e) zero cold slots still pair publish and await;
//! * (f) prefill: F16 rows copied out per sub-chunk, meta + release, await +
//!   BF16 copy back, add (37 rows, 2 sub-chunks);
//! * (g) prefill timeouts (none answered / one of two answered) skip the add;
//! * (h) zero-cold prefill masks nothing; (i) the expert device's converts.
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
            rows: b + 192,
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
    let t = gpu
        .alloc_tensor(&[v.len().div_ceil(4)], DType::F32)
        .unwrap();
    gpu.memcpy_htod_auto(&t.buf, v).unwrap();
    t
}

fn download_i32(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<i32> {
    let mut b = vec![0u8; n * 4];
    gpu.hip.memcpy_dtoh(&mut b, &t.buf).unwrap();
    b.chunks(4)
        .map(|c| i32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn fail(case: &str, msg: String) -> ! {
    println!("FAIL {case}: {msg}");
    std::process::exit(1);
}

/// f32 -> IEEE half bits, round-to-nearest-even (the device `(_Float16)` cast).
fn f32_to_f16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    let sign = ((u >> 16) & 0x8000) as u16;
    let exp = ((u >> 23) & 0xff) as i32;
    let man = u & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1f => {
            if man == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn bf16_rne(v: f32) -> u16 {
    let u = v.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
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
    let expanded = gpu
        .upload_f32(&vec![SENTINEL; TOP_K * DIM], &[TOP_K * DIM])
        .unwrap();
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
    gpu.hetero_publish_cold(
        &counter,
        0,
        LAYERS,
        &x_t,
        &ids_t,
        &w_t,
        &bitmap_t,
        N_EXP as i32,
        1,
        TOP_K as i32,
        DIM as i32,
        &s0,
        &masked_t,
    )
    .unwrap();
    gpu.hetero_await_cold(&counter, 0, LAYERS, &s0, 50_000_000)
        .unwrap();
    gpu.hetero_install_cold(
        &s0,
        &ids_t,
        &bitmap_t,
        N_EXP as i32,
        TOP_K as u32,
        DIM as i32,
        &expanded,
    )
    .unwrap();
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
    let want_masked: Vec<i32> = ids
        .iter()
        .map(|&e| if bitmap[e as usize] != 0 { -1 } else { e })
        .collect();
    if masked != want_masked {
        fail("a", format!("masked ids {masked:?} != {want_masked:?}"));
    }
    let got = gpu.download_f32(&expanded).unwrap();
    for s in 0..TOP_K {
        let cold = bitmap[ids[s] as usize] != 0;
        for i in 0..DIM {
            let want = if cold { pattern(s, i) } else { SENTINEL };
            if got[s * DIM + i].to_bits() != want.to_bits() {
                fail(
                    "c",
                    format!("slot {s} elem {i}: {} != {want}", got[s * DIM + i]),
                );
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
    gpu.hetero_publish_cold(
        &counter,
        1,
        LAYERS,
        &x_t,
        &ids_t,
        &w_t,
        &bitmap_t,
        N_EXP as i32,
        1,
        TOP_K as i32,
        DIM as i32,
        &s1,
        &masked_t,
    )
    .unwrap();
    gpu.hetero_await_cold(&counter, 1, LAYERS, &s1, 20_000)
        .unwrap();
    gpu.hetero_install_cold(
        &s1,
        &ids_t,
        &bitmap_t,
        N_EXP as i32,
        TOP_K as u32,
        DIM as i32,
        &expanded,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let dt = t.elapsed();
    let err = word(s1.error).load(Ordering::Acquire);
    if err != seq_for(1, 1, LAYERS) {
        fail(
            "d",
            format!("error word {err}, expected {}", seq_for(1, 1, LAYERS)),
        );
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
    gpu.hetero_publish_cold(
        &counter,
        2,
        LAYERS,
        &x_t,
        &ids_t,
        &w_t,
        &none,
        N_EXP as i32,
        1,
        TOP_K as i32,
        DIM as i32,
        &s2,
        &masked_t,
    )
    .unwrap();
    gpu.hetero_await_cold(&counter, 2, LAYERS, &s2, 50_000_000)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();
    if !r2.join().unwrap() || word(s2.error).load(Ordering::Acquire) != 0 {
        fail("e", "zero-cold layer did not pair publish and await".into());
    }
    if download_i32(&gpu, &masked_t, TOP_K) != ids {
        fail("e", "zero-cold masked ids should equal the ids".into());
    }
    // ---- (f)-(i): prefill slot (separate pinned slot, max 64 rows, two
    // sub-chunks; layout mirrored from hetero_mailbox::PrefillGeometry).
    const PROWS: usize = 64;
    const SUBS: usize = 2;
    let al = |x: usize| x.div_ceil(256) * 256;
    let p_x = al(64 * (1 + 3 * SUBS));
    let p_ids = al(p_x + PROWS * DIM * 2);
    let p_w = al(p_ids + PROWS * TOP_K * 4);
    let p_part = al(p_w + PROWS * TOP_K * 4);
    let p_total = al(p_part + PROWS * DIM * 2);
    let pb = gpu.hip.host_malloc(p_total, flags).unwrap().cast::<u8>();
    // SAFETY: fresh pinned allocation of `p_total` bytes.
    unsafe { std::ptr::write_bytes(pb, 0, p_total) };
    let pa = pb as u64;
    let p_err = pa;
    let req = move |i: usize| pa + (64 * (1 + 3 * i)) as u64;
    let resp = move |i: usize| req(i) + 64;
    let meta = move |i: usize| req(i) + 128;
    let sub_range = |rows: usize, i: usize| {
        let (q, r) = (rows / SUBS, rows % SUBS);
        (i * q + i.min(r), q + usize::from(i < r))
    };
    let rows = 37usize; // a partial last chunk
    let xs: Vec<f32> = (0..rows * DIM)
        .map(|i| ((i % 113) as f32 - 56.0) * 0.003)
        .collect();
    let xs_t = gpu.upload_f32(&xs, &[rows * DIM]).unwrap();
    let pids: Vec<i32> = (0..rows * TOP_K)
        .map(|s| ((s * 29 + 3) % N_EXP) as i32)
        .collect();
    let pids_t = upload_i32(&mut gpu, &pids);
    let pw: Vec<f32> = (0..rows * TOP_K).map(|s| (s % 7) as f32 * 0.05).collect();
    let pw_t = gpu.upload_f32(&pw, &[rows * TOP_K]).unwrap();
    let pmasked = gpu.alloc_tensor(&[PROWS * TOP_K], DType::F32).unwrap();
    let x_f16 = gpu.alloc_tensor(&[PROWS * DIM / 2], DType::F32).unwrap();
    let part_local = gpu.alloc_tensor(&[PROWS * DIM / 2], DType::F32).unwrap();
    let base_x: Vec<f32> = (0..rows * DIM).map(|i| (i % 17) as f32).collect();
    let x_batch = gpu.upload_f32(&base_x, &[rows * DIM]).unwrap();
    // BF16-exact partial values (7 significant bits).
    let part = |i: usize| ((i % 61) as f32 - 30.0) * 0.25;
    // SAFETY: host views of the pinned slot.
    let mb_x = unsafe {
        hip_bridge::DeviceBuffer::from_raw(
            (pa + p_x as u64) as *mut std::ffi::c_void,
            PROWS * DIM * 2,
        )
    };
    let mb_part = unsafe {
        hip_bridge::DeviceBuffer::from_raw(
            (pa + p_part as u64) as *mut std::ffi::c_void,
            PROWS * DIM * 2,
        )
    };
    // One chunk: publish, per sub-chunk copy-out + release, per sub-chunk
    // await + copy-back, then the add. Stream order on the default stream.
    let chunk = |gpu: &mut Gpu, bitmap_t: &GpuTensor, layer: u32, polls: u32| {
        gpu.hetero_publish_rows(
            &xs_t,
            &pids_t,
            &pw_t,
            bitmap_t,
            N_EXP as i32,
            rows as i32,
            TOP_K as i32,
            DIM as i32,
            &x_f16,
            pa + p_ids as u64,
            pa + p_w as u64,
            &pmasked,
        )
        .unwrap();
        for i in 0..SUBS {
            let (off, n) = sub_range(rows, i);
            gpu.hip
                .memcpy_dtod_async_default_at(
                    &mb_x,
                    off * DIM * 2,
                    &x_f16.buf,
                    off * DIM * 2,
                    n * DIM * 2,
                )
                .unwrap();
            gpu.hetero_release_rows(
                &counter,
                layer,
                LAYERS,
                off as i32,
                n as i32,
                (i % 2) as u32,
                meta(i),
                req(i),
            )
            .unwrap();
        }
        for i in 0..SUBS {
            let (off, n) = sub_range(rows, i);
            let slot = HeteroSlotPtrs {
                req_seq: req(i),
                resp_seq: resp(i),
                error: p_err,
                rows: 0,
                x_rot: 0,
                ids: 0,
                weights: 0,
                cold_out: 0,
            };
            gpu.hetero_await_cold(&counter, layer, LAYERS, &slot, polls)
                .unwrap();
            gpu.hip
                .memcpy_dtod_async_default_at(
                    &part_local.buf,
                    off * DIM * 2,
                    &mb_part,
                    off * DIM * 2,
                    n * DIM * 2,
                )
                .unwrap();
        }
        gpu.hetero_add_partial(p_err, &part_local, (rows * DIM) as i32, &x_batch)
            .unwrap();
    };
    // The host responder: answer sub-chunks `answer` of layer `layer`, checking
    // each request's meta and F16 rows first.
    let x16_want: Vec<u16> = xs.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let respond = |layer: u32, answer: Vec<usize>| {
        let want = seq_for(2, layer, LAYERS);
        let x16_want = x16_want.clone();
        std::thread::spawn(move || -> Result<(), String> {
            for i in answer {
                let t0 = Instant::now();
                while word(req(i)).load(Ordering::Acquire) != want {
                    if t0.elapsed() > Duration::from_secs(5) {
                        return Err(format!("sub {i}: request never seen"));
                    }
                }
                let m = |k: u64| word(meta(i) + 4 * k).load(Ordering::Acquire);
                let (l, off, n, f) = (m(0), m(1), m(2), m(3));
                let (woff, wn) = sub_range(rows, i);
                if (l, off as usize, n as usize, f) != (layer, woff, wn, (i % 2) as u32) {
                    return Err(format!(
                        "sub {i}: meta ({l}, {off}, {n}, {f}) != ({layer}, {woff}, {wn}, {})",
                        i % 2
                    ));
                }
                // SAFETY: the slot's F16 rows and BF16 partial.
                let (xg, out) = unsafe {
                    (
                        std::slice::from_raw_parts((pa + p_x as u64) as *const u16, PROWS * DIM),
                        std::slice::from_raw_parts_mut(
                            (pa + p_part as u64) as *mut u16,
                            PROWS * DIM,
                        ),
                    )
                };
                let r = woff * DIM..(woff + wn) * DIM;
                if xg[r.clone()] != x16_want[r.clone()] {
                    return Err(format!("sub {i}: F16 rows differ"));
                }
                for k in r {
                    out[k] = (part(k).to_bits() >> 16) as u16;
                }
                word(resp(i)).store(want, Ordering::Release);
            }
            Ok(())
        })
    };
    // (f) both sub-chunks answered.
    gpu.hetero_token_advance(&counter).unwrap(); // token 2 (the decode cases used 1)
    let rf = respond(5, vec![0, 1]);
    chunk(&mut gpu, &bitmap_t, 5, 50_000_000);
    gpu.hip.device_synchronize().unwrap();
    if let Err(e) = rf.join().unwrap() {
        fail("f", e);
    }
    if word(p_err).load(Ordering::Acquire) != 0 {
        fail("f", "error word set".into());
    }
    // SAFETY: the slot's route fields.
    let (gi, gw) = unsafe {
        (
            std::slice::from_raw_parts((pa + p_ids as u64) as *const i32, rows * TOP_K),
            std::slice::from_raw_parts((pa + p_w as u64) as *const f32, rows * TOP_K),
        )
    };
    if gi != &pids[..] || gw != &pw[..] {
        fail("f", "prefill slot route".into());
    }
    let want_pm: Vec<i32> = pids
        .iter()
        .map(|&e| if bitmap[e as usize] != 0 { -1 } else { e })
        .collect();
    if download_i32(&gpu, &pmasked, rows * TOP_K) != want_pm {
        fail("f", "prefill masked ids".into());
    }
    let gotx = gpu.download_f32(&x_batch).unwrap();
    for i in 0..rows * DIM {
        if gotx[i].to_bits() != (base_x[i] + part(i)).to_bits() {
            fail(
                "f",
                format!("x_batch[{i}] {} != {}", gotx[i], base_x[i] + part(i)),
            );
        }
    }
    // (g1) nobody answers layer 6: error word set, nothing added.
    // (g2) only sub-chunk 0 of layer 7 answered: the whole add is still skipped.
    for (case, layer, answer) in [("g1", 6u32, vec![]), ("g2", 7, vec![0usize])] {
        // SAFETY: poison the partial so a wrong add would show; clear the error.
        unsafe { std::ptr::write_bytes((pa + p_part as u64) as *mut u8, 0x41, PROWS * DIM * 2) };
        word(p_err).store(0, Ordering::Release);
        let before = gpu.download_f32(&x_batch).unwrap();
        let r = respond(layer, answer);
        chunk(&mut gpu, &bitmap_t, layer, 20_000);
        gpu.hip.device_synchronize().unwrap();
        if let Err(e) = r.join().unwrap() {
            fail(case, e);
        }
        if word(p_err).load(Ordering::Acquire) != seq_for(2, layer, LAYERS) {
            fail(case, "timeout did not set the error word".into());
        }
        if gpu.download_f32(&x_batch).unwrap() != before {
            fail(case, "add_partial ran after a timeout".into());
        }
    }
    word(p_err).store(0, Ordering::Release);
    // (h) zero cold: masked == ids.
    gpu.hetero_publish_rows(
        &xs_t,
        &pids_t,
        &pw_t,
        &none,
        N_EXP as i32,
        rows as i32,
        TOP_K as i32,
        DIM as i32,
        &x_f16,
        pa + p_ids as u64,
        pa + p_w as u64,
        &pmasked,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    if download_i32(&gpu, &pmasked, rows * TOP_K) != pids {
        fail(
            "h",
            "zero-cold prefill masked ids should equal the ids".into(),
        );
    }
    // (i) the expert device's converts (same source on every arch).
    let specials = [
        0.0f32,
        -0.0,
        1.0,
        -2.5,
        65504.0,
        1e-6,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
    ];
    let mut vals: Vec<f32> = specials.to_vec();
    // RNE ties and near-ties around BF16 boundaries.
    for b in [
        0x3f80_8000u32,
        0x3f81_8000,
        0x3f80_7fff,
        0x3f80_8001,
        0xbf80_8000,
        0x0000_8000,
        0x7f7f_8000,
    ] {
        vals.push(f32::from_bits(b));
    }
    vals.extend((0..997).map(|k| ((k * 7919) % 20011) as f32 * 0.0137 - 137.0));
    let n = vals.len();
    let src = gpu.upload_f32(&vals, &[n]).unwrap();
    let b16 = gpu.hip.host_malloc(n * 2, flags).unwrap().cast::<u16>();
    gpu.hetero_f32_to_bf16(&src, b16 as u64, n as i32).unwrap();
    let f16s: Vec<u16> = vals.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let h16 = gpu.hip.host_malloc(n * 2, flags).unwrap().cast::<u16>();
    // SAFETY: pinned buffers of n u16.
    unsafe { std::ptr::copy_nonoverlapping(f16s.as_ptr(), h16, n) };
    let back = gpu.alloc_tensor(&[n], DType::F32).unwrap();
    gpu.hetero_f16_to_f32(h16 as u64, &back, n as i32).unwrap();
    gpu.hip.device_synchronize().unwrap();
    // SAFETY: as above.
    let got16 = unsafe { std::slice::from_raw_parts(b16, n) }.to_vec();
    for (k, &v) in vals.iter().enumerate() {
        let want = bf16_rne(v);
        let ok = if v.is_nan() {
            got16[k] & 0x7f80 == 0x7f80 && got16[k] & 0x7f != 0
        } else {
            got16[k] == want
        };
        if !ok {
            fail(
                "i",
                format!(
                    "bf16({v:e} = {:#010x}) = {:#06x}, want {want:#06x}",
                    v.to_bits(),
                    got16[k]
                ),
            );
        }
    }
    let gotf = gpu.download_f32(&back).unwrap();
    for (k, &h) in f16s.iter().enumerate() {
        let want = f16_bits_to_f32(h);
        if !(gotf[k].to_bits() == want.to_bits() || (want.is_nan() && gotf[k].is_nan())) {
            fail("i", format!("f16 {h:#06x} -> {} want {want}", gotf[k]));
        }
    }
    let _ = gpu.hip.host_free(b16.cast());
    let _ = gpu.hip.host_free(h16.cast());
    let _ = gpu.hip.host_free(pb.cast());
    let _ = gpu.hip.host_free(base.cast());
    println!("PASS");
}
