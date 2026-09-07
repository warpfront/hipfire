//! Bare-PM4 device-dependency-flag kill test + two-line rescue.
//!
//! `poc_devdeps <elf> [MxK,...] [render-node]` builds IBs of N=100
//! producer/consumer pairs and reports GPU-side microseconds per pair as
//! (t(N) - t(1)) / (N - 1), min of 3 submissions, plus bit-compares of the
//! barrier arm vs the flag arm outputs.
//!
//! Arms (per pair i, ctr[i] zeroed and ready[i] zeroed by host before every submit):
//!   A (baseline small->gemv): [prod_small_noflag] BARRIER [gemv_plain] BARRIER
//!   B (flags   small->gemv): [prod_small -> ctr[i]] no barrier [cons_gemv waits ctr[i]] BARRIER
//!   C (baseline gemv->small): [gemv_plain] BARRIER [cons_small_noflag] BARRIER
//!   D (flags   gemv->small): [prod_gemv -> ctr[i]] no barrier [cons_small waits ctr[i]] BARRIER
//!   E (two-line small->gemv): [prod_small2 -> ctr[i]/ready[i]] no barrier [cons_gemv2 polls ready[i]] BARRIER
//!   F (two-line gemv->small): [prod_gemv2 -> ctr[i]/ready[i]] no barrier [cons_small2 polls ready[i]] BARRIER
//!
//! Two-line protocol: producers arrival-count on ctr; the last arrival fences
//! then release-stores 1 to ready[i] (own 256B cache line, separate buffer).
//! Consumers poll ONLY ready[i] with sleep_n s_sleep(1) between polls.
//!
//! Each pair writes its own x/y/out slice so a stale device-side read shows
//! up as bit diffs vs the barrier arm. Output slices are pre-filled with a
//! 0xFF sentinel (quiet NaN) before the correctness submit.
use redline::device::Device;
use redline::dispatch::{CommandBuffer, DispatchQueue, Kernel, KernargBuilder};
use std::time::Instant;

const N: usize = 100;
/// Per-pair kernarg stride; slots: prod@+0 (48B), gemv@+64 (64B), small@+128 (48B).
const KA_STRIDE: usize = 256;
/// Per-pair ready stride: own cache line, separate buffer from ctr.
const READY_STRIDE: usize = 256;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn pm4_barrier(cb: &mut CommandBuffer) {
    // Production gfx12 tape inter-node barrier (redline-rocr pm4.rs).
    cb.push_raw(&[0xc000_4600, 0x407]);
    cb.push_raw(&[0xc006_5800, 0, u32::MAX, 0x00ff_ffff, 0, 0, 0x0000_000a, 0x10180]);
}

fn word_diffs(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4).zip(b.chunks_exact(4)).filter(|(x, y)| x != y).count()
}

/// Arm config: (label, producer kind, consumer kind, sleep_n).
/// Kinds: 'A','B','C','D' legacy; 'E','F' two-line (sleep_n selects poll interval).
fn configs() -> Vec<(String, char, u32)> {
    let mut v = vec![("A".to_string(), 'A', 0), ("C".to_string(), 'C', 0)];
    for s in [2u32, 8, 32] {
        v.push((format!("E_s{s}"), 'E', s));
    }
    for s in [2u32, 8, 32] {
        v.push((format!("F_s{s}"), 'F', s));
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert!(!args.is_empty(), "usage: poc_devdeps <elf> [MxK,...] [render-node]");
    let shapes: Vec<(u32, u32)> = args
        .get(1)
        .map(|s| {
            s.split(',')
                .map(|p| {
                    let mut it = p.split(['x', 'X']);
                    (it.next().unwrap().parse().unwrap(), it.next().unwrap().parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(16480, 5120), (5120, 5120), (1024, 5120)]);
    let dev = Device::open(args.get(2).map(|s| s.as_str())).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    eprintln!(
        "gfx_arch={} gran={}",
        dev.info.gfx_arch,
        redline::dispatch::lds_granularity(&dev.info.gfx_arch)
    );
    let module = dev.load_module_file(&args[0]).unwrap();
    let fence = dev.alloc_vram(4096).unwrap();
    let gran = redline::dispatch::lds_granularity(&dev.info.gfx_arch);
    let k_prod = Kernel::find(&module, "prod_small").expect("prod_small");
    let k_prod_nf = Kernel::find(&module, "prod_small_noflag").expect("prod_small_noflag");
    let k_gemv = Kernel::find(&module, "gemv_plain").expect("gemv_plain");
    let k_cons_gemv = Kernel::find(&module, "cons_gemv").expect("cons_gemv");
    let k_prod_gemv = Kernel::find(&module, "prod_gemv").expect("prod_gemv");
    let k_cons_small = Kernel::find(&module, "cons_small").expect("cons_small");
    let k_cons_small_nf = Kernel::find(&module, "cons_small_noflag").expect("cons_small_noflag");
    let k_prod2 = Kernel::find(&module, "prod_small2").expect("prod_small2");
    let k_prod_gemv2 = Kernel::find(&module, "prod_gemv2").expect("prod_gemv2");
    let k_cons_gemv2 = Kernel::find(&module, "cons_gemv2").expect("cons_gemv2");
    let k_cons_small2 = Kernel::find(&module, "cons_small2").expect("cons_small2");

    for &(m, k) in &shapes {
        let groups = (k / 256) as usize;
        let wbytes = m as usize * groups * 136;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut w = vec![0u8; wbytes];
        for g in 0..(m as usize * groups) {
            let o = g * 136;
            w[o..o + 4].copy_from_slice(&0xB4CD_2E66u32.to_le_bytes());
            w[o + 4..o + 8].copy_from_slice(&0xB4CD_2E66u32.to_le_bytes());
            for c in w[o + 8..o + 136].chunks_mut(8) {
                c.copy_from_slice(&xorshift(&mut seed).to_le_bytes());
            }
        }
        let x0: Vec<f32> =
            (0..k).map(|_| (xorshift(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
        let d_w = dev.alloc_vram(wbytes as u64).unwrap();
        dev.upload(&d_w, &w).unwrap();
        let d_x0 = dev.alloc_vram(k as u64 * 4).unwrap();
        dev.upload(
            &d_x0,
            unsafe { std::slice::from_raw_parts(x0.as_ptr() as *const u8, k as usize * 4) },
        )
        .unwrap();
        let d_x = dev.alloc_vram(N as u64 * k as u64 * 4).unwrap();
        let d_y = dev.alloc_vram(N as u64 * m as u64 * 4).unwrap();
        let d_out = dev.alloc_vram(N as u64 * m as u64 * 4).unwrap();
        let d_ctr = dev.alloc_vram(N as u64 * 4).unwrap();
        let d_ready = dev.alloc_vram(N as u64 * READY_STRIDE as u64).unwrap();
        let d_ka = dev.alloc_vram(N as u64 * KA_STRIDE as u64).unwrap();

        let small_grid = ((k + 255) / 256) as u32; // prod_small grid (arms A/B/E)
        let cons_grid = ((m + 255) / 256) as u32; // cons_small grid (arms C/D/F)
        let tgt_b = small_grid; // arm B: producer blocks per pair
        let tgt_d = m; // arm D: producer WGs (one per row) per pair
        let zeros4 = vec![0u8; N * 4];
        let zeros_ready = vec![0u8; N * READY_STRIDE];
        let sentinel_x = vec![0xFFu8; N * k as usize * 4];
        let sentinel_y = vec![0xFFu8; N * m as usize * 4];

        // Kernarg arena builder per arm (+ sleep for E/F consumers).
        let build_arena = |arm: char, sleep: u32| -> Vec<u8> {
            let mut arena = vec![0u8; N * KA_STRIDE];
            for i in 0..N {
                let xs = d_x.gpu_addr + i as u64 * k as u64 * 4;
                let ys = d_y.gpu_addr + i as u64 * m as u64 * 4;
                let os = d_out.gpu_addr + i as u64 * m as u64 * 4;
                let cs = d_ctr.gpu_addr + i as u64 * 4;
                let rs = d_ready.gpu_addr + i as u64 * READY_STRIDE as u64;
                let base = i * KA_STRIDE;
                // prod slot: xin, xout, K, ctr, ready, nwg
                let mut ka = KernargBuilder::new(48);
                ka.write_ptr(0, d_x0.gpu_addr)
                    .write_ptr(8, xs)
                    .write_u32(16, k)
                    .write_ptr(24, cs)
                    .write_ptr(32, rs)
                    .write_u32(40, small_grid);
                arena[base..base + 48].copy_from_slice(ka.as_bytes());
                // gemv slot: A, x, y, M, K, ctr/target, ready/sleep or target
                let mut ka = KernargBuilder::new(64);
                let gx = match arm {
                    'A' | 'B' | 'E' => xs, // reads the small-producer output
                    _ => d_x0.gpu_addr,    // reads the shared input directly
                };
                ka.write_ptr(0, d_w.gpu_addr)
                    .write_ptr(8, gx)
                    .write_ptr(16, ys)
                    .write_u32(24, m)
                    .write_u32(28, k)
                    .write_ptr(32, cs);
                match arm {
                    'B' => {
                        ka.write_u32(40, tgt_b);
                    }
                    'E' => {
                        ka.write_ptr(32, rs).write_u32(40, sleep);
                    }
                    'F' => {
                        ka.write_ptr(32, cs).write_ptr(40, rs).write_u32(48, m);
                    }
                    _ => {
                        ka.write_u32(40, tgt_d);
                    }
                }
                arena[base + 64..base + 64 + 64].copy_from_slice(ka.as_bytes());
                // small slot: yin, out, M, ctr/target, ready/sleep or target
                let mut ka = KernargBuilder::new(48);
                ka.write_ptr(0, ys).write_ptr(8, os).write_u32(16, m);
                match arm {
                    'F' => {
                        ka.write_ptr(24, rs).write_u32(32, sleep);
                    }
                    _ => {
                        ka.write_ptr(24, cs).write_u32(32, tgt_d);
                    }
                }
                arena[base + 128..base + 128 + 48].copy_from_slice(ka.as_bytes());
            }
            arena
        };

        // IB builder: `pairs` pairs of arm `arm`.
        let build_ib = |arm: char, pairs: usize| -> CommandBuffer {
            let mut cb = CommandBuffer::new();
            for i in 0..pairs {
                let base = (i * KA_STRIDE) as u64;
                let ka_prod = d_ka.gpu_addr + base;
                let ka_gemv = d_ka.gpu_addr + base + 64;
                let ka_small = d_ka.gpu_addr + base + 128;
                match arm {
                    'A' => {
                        cb.dispatch_lds(k_prod_nf, [small_grid, 1, 1], [256, 1, 1], ka_prod, 0, gran);
                        pm4_barrier(&mut cb);
                        cb.dispatch_lds(k_gemv, [m, 1, 1], [32, 1, 1], ka_gemv, 0, gran);
                        pm4_barrier(&mut cb);
                    }
                    'B' => {
                        cb.dispatch_lds(k_prod, [small_grid, 1, 1], [256, 1, 1], ka_prod, 0, gran);
                        cb.dispatch_lds(k_cons_gemv, [m, 1, 1], [32, 1, 1], ka_gemv, 0, gran);
                        pm4_barrier(&mut cb);
                    }
                    'C' => {
                        cb.dispatch_lds(k_gemv, [m, 1, 1], [32, 1, 1], ka_gemv, 0, gran);
                        pm4_barrier(&mut cb);
                        cb.dispatch_lds(
                            k_cons_small_nf,
                            [cons_grid, 1, 1],
                            [256, 1, 1],
                            ka_small,
                            0,
                            gran,
                        );
                        pm4_barrier(&mut cb);
                    }
                    'D' => {
                        cb.dispatch_lds(k_prod_gemv, [m, 1, 1], [32, 1, 1], ka_gemv, 0, gran);
                        cb.dispatch_lds(
                            k_cons_small,
                            [cons_grid, 1, 1],
                            [256, 1, 1],
                            ka_small,
                            0,
                            gran,
                        );
                        pm4_barrier(&mut cb);
                    }
                    'E' => {
                        cb.dispatch_lds(k_prod2, [small_grid, 1, 1], [256, 1, 1], ka_prod, 0, gran);
                        cb.dispatch_lds(k_cons_gemv2, [m, 1, 1], [32, 1, 1], ka_gemv, 0, gran);
                        pm4_barrier(&mut cb);
                    }
                    'F' => {
                        cb.dispatch_lds(k_prod_gemv2, [m, 1, 1], [32, 1, 1], ka_gemv, 0, gran);
                        cb.dispatch_lds(
                            k_cons_small2,
                            [cons_grid, 1, 1],
                            [256, 1, 1],
                            ka_small,
                            0,
                            gran,
                        );
                        pm4_barrier(&mut cb);
                    }
                    _ => unreachable!(),
                }
            }
            cb
        };

        let bos = [
            &module.code_buf,
            &d_w,
            &d_x0,
            &d_x,
            &d_y,
            &d_out,
            &d_ctr,
            &d_ready,
            &d_ka,
            &fence,
        ];
        let cfgs = configs();
        // snapshots per config: x, y, out
        let mut snaps: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
        let mut per: Vec<Option<f64>> = Vec::new();
        for (label, arm, sleep) in &cfgs {
            dev.upload(&d_ka, &build_arena(*arm, *sleep)).unwrap();
            let ib1 = build_ib(*arm, 1);
            let ibn = build_ib(*arm, N);
            dev.upload(&d_x, &sentinel_x).unwrap();
            dev.upload(&d_y, &sentinel_y).unwrap();
            dev.upload(&d_out, &sentinel_y).unwrap();
            dev.upload(&d_ctr, &zeros4).unwrap();
            dev.upload(&d_ready, &zeros_ready).unwrap();
            match dq.submit(&dev, &ibn, &bos) {
                Ok(()) => {}
                Err(e) => eprintln!("M={m} K={k} arm{label}: CORRECTNESS SUBMIT FAILED: {e:?}"),
            }
            let mut xb = vec![0u8; N * k as usize * 4];
            let mut yb = vec![0u8; N * m as usize * 4];
            let mut ob = vec![0u8; N * m as usize * 4];
            dev.download(&d_x, &mut xb).unwrap_or_default();
            dev.download(&d_y, &mut yb).unwrap_or_default();
            dev.download(&d_out, &mut ob).unwrap_or_default();
            snaps.push((xb, yb, ob));

            let mut best = f64::INFINITY;
            let mut ok = true;
            dev.upload(&d_ctr, &zeros4).unwrap();
            dev.upload(&d_ready, &zeros_ready).unwrap();
            if let Err(e) = dq.submit(&dev, &ibn, &bos) {
                eprintln!("M={m} K={k} arm{label}: warm submit failed: {e:?}");
                ok = false;
            }
            let mut t1_last = 0.0;
            let mut tn_last = 0.0;
            for _ in 0..3 {
                dev.upload(&d_ctr, &zeros4).unwrap();
                dev.upload(&d_ready, &zeros_ready).unwrap();
                let t = Instant::now();
                if let Err(e) = dq.submit(&dev, &ib1, &bos) {
                    eprintln!("M={m} K={k} arm{label}: t1 submit failed: {e:?}");
                    ok = false;
                    break;
                }
                let t1 = t.elapsed().as_secs_f64() * 1e6;
                dev.upload(&d_ctr, &zeros4).unwrap();
                dev.upload(&d_ready, &zeros_ready).unwrap();
                let t = Instant::now();
                if let Err(e) = dq.submit(&dev, &ibn, &bos) {
                    eprintln!("M={m} K={k} arm{label}: tN submit failed: {e:?}");
                    ok = false;
                    break;
                }
                let tn = t.elapsed().as_secs_f64() * 1e6;
                t1_last = t1;
                tn_last = tn;
                best = best.min((tn - t1) / (N as f64 - 1.0));
            }
            if ok {
                per.push(Some(best));
                eprintln!(
                    "M={m} K={k} arm{label}: {best:.2} us/pair  (t1={t1_last:.0}us tN={tn_last:.0}us)"
                );
            } else {
                per.push(None);
                eprintln!("M={m} K={k} arm{label}: TIMING FAILED (hang? see above)");
            }
        }

        // Bit-exact: E_* vs A (x and y), F_* vs C (y and out).
        let (ax, ay) = (&snaps[0].0, &snaps[0].1);
        let (cy, co) = (&snaps[1].1, &snaps[1].2);
        for (idx, (label, arm, _)) in cfgs.iter().enumerate() {
            let (dx, dy, dout) = match arm {
                'E' => (
                    word_diffs(ax, &snaps[idx].0),
                    word_diffs(ay, &snaps[idx].1),
                    usize::MAX,
                ),
                'F' => (
                    usize::MAX,
                    word_diffs(cy, &snaps[idx].1),
                    word_diffs(co, &snaps[idx].2),
                ),
                _ => continue,
            };
            if *arm == 'E' {
                eprintln!(
                    "M={m} K={k} x-bit-diff A-vs-{label} words={dx} {}",
                    if dx == 0 { "BIT-EXACT" } else { "DIFFER" }
                );
                eprintln!(
                    "M={m} K={k} y-bit-diff A-vs-{label} words={dy} {}",
                    if dy == 0 { "BIT-EXACT" } else { "DIFFER" }
                );
            } else {
                eprintln!(
                    "M={m} K={k} y-bit-diff C-vs-{label} words={dy} {}",
                    if dy == 0 { "BIT-EXACT" } else { "DIFFER" }
                );
                eprintln!(
                    "M={m} K={k} out-bit-diff C-vs-{label} words={dout} {}",
                    if dout == 0 { "BIT-EXACT" } else { "DIFFER" }
                );
            }
        }
        let pa = per[0];
        let pc = per[1];
        for (idx, (label, arm, _)) in cfgs.iter().enumerate() {
            let base = if *arm == 'E' { pa } else { pc };
            if let (Some(b), Some(v)) = (base, per[idx]) {
                let dir = if *arm == 'E' { "A-E" } else { "C-F" };
                eprintln!("M={m} K={k} delta {dir} ({label} saves) = {:.2} us/pair", b - v);
            }
        }
    }
}
