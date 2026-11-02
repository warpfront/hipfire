// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU as the glue device: STREAM-style CPU bandwidth (alone and under a concurrent GPU bandwidth load), and the
//! IEF15 A-stream pack (codes + D/b tails, byte-identical to `Ief15Layout::pack_in`) and the down ADD epilogue
//! `x = RN(x + Y)` done by CPU threads with non-temporal stores into the NPU-shared pages. The SiLU epilogue stays on the
//! GPU (its exp / rcp instruction DAG is not reproducible bit for bit on the CPU).
use super::*;
use core::arch::x86_64::*;

/// Worker threads (all logical CPUs).
pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Run `f(i)` for `i in 0..n` on all threads (static block split).
pub fn par_for(n: usize, f: &(dyn Fn(usize) + Sync)) {
    let th = threads().min(n.max(1));
    let per = n.div_ceil(th);
    std::thread::scope(|s| {
        for k in 0..th {
            s.spawn(move || {
                for i in k * per..((k + 1) * per).min(n) {
                    f(i)
                }
            });
        }
    });
}

#[derive(Clone, Copy)]
struct SendPtr(*mut u8);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Through a method so closures capture the whole `Send + Sync` wrapper, not its raw-pointer field.
    fn p(&self) -> *mut u8 {
        self.0
    }
}
#[target_feature(enable = "avx512f")]
unsafe fn nt_copy(dst: *mut u8, src: &[u8]) {
    debug_assert!(src.len() % 64 == 0 && dst as usize % 64 == 0);
    for i in (0..src.len()).step_by(64) {
        _mm512_stream_si512(
            dst.add(i) as *mut __m512i,
            _mm512_loadu_si512(src.as_ptr().add(i) as *const __m512i),
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn sum_read(p: *const u8, len: usize) -> u64 {
    let mut acc = _mm512_setzero_si512();
    for i in (0..len).step_by(64) {
        acc = _mm512_xor_si512(acc, _mm512_load_si512(p.add(i) as *const __m512i));
    }
    _mm512_reduce_add_epi64(acc) as u64
}

#[target_feature(enable = "avx512f")]
unsafe fn nt_fill(p: *mut u8, len: usize, v: i64) {
    let x = _mm512_set1_epi64(v);
    for i in (0..len).step_by(64) {
        _mm512_stream_si512(p.add(i) as *mut __m512i, x);
    }
}

/// STREAM read / write / copy GB/s over `bytes` per array with all threads (write and copy use NT stores).
pub fn stream(bytes: usize) -> Result<[f64; 3], String> {
    if !is_x86_feature_detected!("avx512f") {
        return Err("cpu: AVX-512F not available".into());
    }
    let (a, b) = (map(bytes)?, map(bytes)?);
    let th = threads();
    let chunk = (bytes / th) & !63;
    let run = |op: u8| -> f64 {
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for k in 0..th {
                let (pa, pb) = (SendPtr(a), SendPtr(b));
                s.spawn(move || unsafe {
                    let (pa, pb) = (pa.p().add(k * chunk), pb.p().add(k * chunk));
                    match op {
                        0 => {
                            std::hint::black_box(sum_read(pa, chunk));
                        }
                        1 => nt_fill(pa, chunk, k as i64),
                        _ => nt_copy(pb, core::slice::from_raw_parts(pa, chunk)),
                    }
                    _mm_sfence();
                });
            }
        });
        let dt = t0.elapsed().as_secs_f64();
        let moved = (chunk * th) as f64 * if op == 2 { 2.0 } else { 1.0 };
        moved / dt / 1e9
    };
    let mut best = [0f64; 3];
    for _ in 0..3 {
        for op in 0..3u8 {
            best[op as usize] = best[op as usize].max(run(op));
        }
    }
    unsafe {
        libc_mm::munmap(a as *mut core::ffi::c_void, bytes);
        libc_mm::munmap(b as *mut core::ffi::c_void, bytes);
    }
    Ok(best)
}

fn map(bytes: usize) -> Result<*mut u8, String> {
    let p = unsafe {
        libc_mm::mmap(
            core::ptr::null_mut(),
            bytes,
            libc_mm::PROT_RW,
            libc_mm::MAP_ANON_POPULATE,
            -1,
            0,
        )
    };
    if p as usize == usize::MAX {
        return Err(format!("mmap {bytes}: {}", std::io::Error::last_os_error()));
    }
    Ok(p as *mut u8)
}

/// IEF15 activation sidecars on all threads (per token: `iu4_ief15::grid15` over its epochs, as `encode_act_scales`).
pub fn act_scales(xq: &[u8], tokens: usize, e: usize) -> (Vec<u16>, Vec<i16>) {
    let mut d = vec![0u16; e * tokens];
    let mut b = vec![0i16; tokens];
    let (dp, bp) = (
        SendPtr(d.as_mut_ptr() as *mut u8),
        SendPtr(b.as_mut_ptr() as *mut u8),
    );
    par_for(tokens, &|t| {
        let v: Vec<f32> = (0..e)
            .map(|ep| f32::from_le_bytes(xq[(ep * tokens + t) * XBLK..][..4].try_into().unwrap()))
            .collect();
        let (m, x) = iu4_ief15::grid15(&v);
        unsafe {
            for (ep, &m) in m.iter().enumerate() {
                *(dp.p() as *mut u16).add(ep * tokens + t) = m as u16;
            }
            *(bp.p() as *mut i16).add(t) = x as i16;
        }
    });
    (d, b)
}

/// The A stream of `g` (byte-identical to `pack_in`'s arg0) from host Xq `[E][T][72]` into `a` (the NPU-shared arena,
/// page aligned), NT stores; returns (sidecar seconds, pack seconds).
pub fn pack(xq: &[u8], a: *mut u8, g: &Ief15Gemm) -> (f64, f64) {
    let t0 = Instant::now();
    let e = g.epochs();
    let (d, b) = act_scales(xq, T, e);
    let t1 = Instant::now();
    let (mw_n, nw_n) = g.layout().wave_grid();
    let aseg = g.design().args[0].bytes / 8;
    let ap = SendPtr(a);
    par_for(8 * mw_n, &|job| {
        let (c, mw) = (job / mw_n, job % mw_n);
        let (blk, hh) = (2 * (c / 4) + c % 2, (c / 2) % 2);
        let mut chunk = [0u8; 2176];
        for ep in 0..e {
            for mbl in 0..4 {
                for i in 0..8 {
                    let t = mw * 256 + blk * 64 + (2 * mbl + hh) * 8 + i;
                    let blk72 = &xq[(ep * T + t) * XBLK..][..XBLK];
                    for kb in 0..16 {
                        chunk[((mbl * 16 + kb) * 8 + i) * 4..][..4]
                            .copy_from_slice(&blk72[8 + kb * 4..8 + kb * 4 + 4]);
                    }
                    let lane = mbl * 8 + i;
                    chunk[2048 + 2 * lane..][..2].copy_from_slice(&d[ep * T + t].to_le_bytes());
                    chunk[2112 + 2 * lane..][..2].copy_from_slice(&b[t].to_le_bytes());
                }
            }
            for nw in 0..nw_n {
                let off = c * aseg + ((mw * nw_n + nw) * e + ep) * 2176;
                unsafe { nt_copy(ap.p().add(off), &chunk) };
            }
        }
        unsafe { _mm_sfence() };
    });
    (
        t1.duration_since(t0).as_secs_f64(),
        t1.elapsed().as_secs_f64(),
    )
}

/// The down ADD epilogue on the CPU: `x[t][col0 + r] = RN(x + Y)` (IEEE f32 add) for the NPU window of `g`, `x` a host
/// `[T][F]` f32 buffer, `c` the NPU's tiled C (host pages). Seconds.
pub fn addc(c: &[u8], x: *mut f32, g: &Ief15Gemm) -> f64 {
    let t0 = Instant::now();
    let y = g.unpack_out(c);
    let n = g.layout().features();
    let xp = SendPtr(x as *mut u8);
    par_for(T, &|t| unsafe {
        let row = (xp.p() as *mut f32).add(t * F);
        for r in 0..n {
            *row.add(r) += f32::from_le_bytes(y[(t * n + r) * 4..][..4].try_into().unwrap());
        }
    });
    t0.elapsed().as_secs_f64()
}

/// `--mode cpu`: CPU bandwidth alone and under a concurrent GPU read load; CPU pack (down n=1024, gate_up 2048-feature
/// command) byte-checked against `pack_in`, and CPU addc checked against the f32 reference; times.
pub fn run(cfg: &Cfg) -> Result<bool, String> {
    let gpu = Gpu::new()?;
    let rt = &gpu.rt;
    println!(
        "cpu: {} threads, avx512f {}",
        threads(),
        is_x86_feature_detected!("avx512f")
    );
    let alone = stream(1 << 30)?;
    println!(
        "CPUBW alone read={:.1} write={:.1} copy={:.1} GB/s",
        alone[0], alone[1], alone[2]
    );
    // Concurrent GPU load: PM gpu-bw read kernel over 1 GiB of device memory, enqueued ahead for > 1.5 s.
    let m_bw = rt.load_module(maps::BW_CO)?;
    let f_read = m_bw.function("read_bw")?;
    let gbuf = rt.allocate(1 << 30, None)?;
    let sink = rt.allocate(rt.compute_units()? as usize * 16 * 256 * 16, None)?;
    let blocks = rt.compute_units()? * 16;
    let (e0, e1) = (rt.event()?, rt.event()?);
    rt.record(&e0)?;
    for _ in 0..400 {
        let mut ka = [0u8; 32];
        put64(&mut ka, 0, gbuf.ptr());
        put64(&mut ka, 8, (1u64 << 30) / 16);
        put64(&mut ka, 16, sink.ptr());
        put64(&mut ka, 24, u64::from(blocks) * 256);
        rt.launch(&f_read, blocks, 256, &mut ka)?;
    }
    rt.record(&e1)?;
    std::thread::sleep(std::time::Duration::from_millis(20));
    let busy = stream(1 << 30)?;
    let ms = f64::from(rt.elapsed_ms(&e0, &e1)?);
    println!("CPUBW with GPU read load read={:.1} write={:.1} copy={:.1} GB/s; GPU load {:.1} GB/s over its {:.0} ms",
        busy[0], busy[1], busy[2], 400.0 * (1u64 << 30) as f64 / ms / 1e6, ms);
    let mut ok = true;
    // CPU pack, both projections; CPU addc (down).
    let inp = load_inputs(cfg)?;
    let o = oracle(&inp, 1024, true)?;
    let a = Shared::new(&gpu, None, o.g.design().args[0].bytes)?;
    a.fill(POISON);
    let mut tp = Vec::new();
    for _ in 0..3 {
        tp.push(pack(&inp.xq, a.host, &o.g));
    }
    let same = a.read(0, o.a_stream.len()) == o.a_stream;
    println!("CPUPACK down n=1024: A stream {} B byte-equal pack_in: {same}; sidecars ms [{}] pack ms [{}]", o.a_stream.len(),
        tp.iter().map(|x| format!("{:.2}", x.0 * 1e3)).collect::<Vec<_>>().join(","), tp.iter().map(|x| format!("{:.2}", x.1 * 1e3)).collect::<Vec<_>>().join(","));
    ok &= same;
    let c = o.g.design().args[2].bytes;
    let mut cbytes = vec![0u8; c];
    let mut s = 0x2545_f491_4f6c_dd1du64;
    for w in cbytes.chunks_exact_mut(4) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        w.copy_from_slice(
            &(((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32).to_le_bytes(),
        );
    }
    let mut x: Vec<f32> = inp
        .old
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let ta: Vec<f64> = (0..3)
        .map(|_| {
            x.copy_from_slice(
                &inp.old
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            );
            addc(&cbytes, x.as_mut_ptr(), &o.g)
        })
        .collect();
    let want = expect_add(&inp.old, &inp.old, &o.g.unpack_out(&cbytes), 1024);
    let got: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let exact = got == want;
    println!(
        "CPUADDC down n=1024: exact {exact}; ms [{}]",
        ta.iter()
            .map(|v| format!("{:.2}", v * 1e3))
            .collect::<Vec<_>>()
            .join(",")
    );
    ok &= exact;
    let gu_x = tile_xq(&cfg.xq.replace("down", "mlp"), gate_up::EG)?;
    let gg = Ief15Gemm::new(T, 2048, gate_up::KG, Control::Fast)?;
    let ga = Shared::new(&gpu, None, gg.design().args[0].bytes)?;
    let (gd, gb) = iu4_ief15::encode_act_scales(&Acts::new(T, gate_up::KG, &gu_x));
    let wdummy = vec![0u8; 2048 * gate_up::ROWG];
    let w = Weights::new(2048, gate_up::KG, &wdummy);
    let (sw, aw) = iu4_ief15::encode_weight_scales(&w);
    let [want_a, _] = gg.pack_in(&w, &Acts::new(T, gate_up::KG, &gu_x), &sw, &aw, &gd, &gb)?;
    let mut tg = Vec::new();
    for _ in 0..3 {
        tg.push(pack(&gu_x, ga.host, &gg));
    }
    let same = ga.read(0, want_a.len()) == want_a;
    println!("CPUPACK gate_up 2048 features: A stream {} B byte-equal pack_in: {same}; sidecars ms [{}] pack ms [{}]", want_a.len(),
        tg.iter().map(|x| format!("{:.2}", x.0 * 1e3)).collect::<Vec<_>>().join(","), tg.iter().map(|x| format!("{:.2}", x.1 * 1e3)).collect::<Vec<_>>().join(","));
    ok &= same;
    Ok(ok)
}
