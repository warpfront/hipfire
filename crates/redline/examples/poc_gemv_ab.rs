//! Bare-PM4 A/B of `gemv_mq4g256v2` code objects: no HIP runtime in the loop.
//!
//! `poc_gemv_ab <elf-a> <elf-b> [MxK,...] [render-node]` loads both ELFs,
//! builds one IB with N back-to-back dispatches (each followed by the
//! redline compute barrier, like a retained tape), and reports GPU-side
//! microseconds per dispatch as (t(N) - t(1)) / (N - 1) over the
//! submit-and-wait wall clock, plus a bit-compare of y between the two.
use redline::device::Device;
use redline::dispatch::{CommandBuffer, DispatchQueue, Kernel, KernargBuilder};
use std::time::Instant;

const N: usize = 200;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shapes: Vec<(u32, u32)> = args
        .get(2)
        .map(|s| {
            s.split(',')
                .map(|p| {
                    let (m, k) = p.split_once('x').unwrap();
                    (m.parse().unwrap(), k.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| vec![(4096, 1280), (16384, 1280), (5120, 5120), (248320, 5120)]);
    let dev = Device::open(args.get(3).map(|s| s.as_str())).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    eprintln!("gfx_arch={} gran={}", dev.info.gfx_arch, redline::dispatch::lds_granularity(&dev.info.gfx_arch));
    let mods: Vec<_> = args[..2].iter().map(|p| dev.load_module_file(p).unwrap()).collect();
    let fence = dev.alloc_vram(4096).unwrap();

    for &(m, k) in &shapes {
        let groups = (k / 256) as usize;
        let wbytes = m as usize * groups * 136;
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut w = vec![0u8; wbytes];
        for g in 0..(m as usize * groups) {
            let o = g * 136;
            // header: lanes 0-15 (sc 0x2E66 ~0.1, zp 0xB4CD ~-0.3) and lanes 16-31
            w[o..o + 4].copy_from_slice(&0xB4CD_2E66u32.to_le_bytes());
            w[o + 4..o + 8].copy_from_slice(&0xB4CD_2E66u32.to_le_bytes());
            for c in w[o + 8..o + 136].chunks_mut(8) {
                c.copy_from_slice(&xorshift(&mut seed).to_le_bytes());
            }
        }
        let x: Vec<f32> = (0..k).map(|_| (xorshift(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
        let d_w = dev.alloc_vram(wbytes as u64).unwrap();
        dev.upload(&d_w, &w).unwrap();
        let d_x = dev.alloc_vram(k as u64 * 4).unwrap();
        dev.upload(&d_x, unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, k as usize * 4) }).unwrap();
        let d_y = dev.alloc_vram(m as u64 * 4).unwrap();
        let d_ka = dev.alloc_vram(4096).unwrap();
        let mut ka = KernargBuilder::new(32);
        ka.write_ptr(0, d_w.gpu_addr).write_ptr(8, d_x.gpu_addr).write_ptr(16, d_y.gpu_addr);
        ka.write_u32(24, m).write_u32(28, k);
        dev.upload(&d_ka, ka.as_bytes()).unwrap();

        let mut ys: Vec<Vec<u8>> = Vec::new();
        let mut per: Vec<f64> = Vec::new();
        for (mi, module) in mods.iter().enumerate() {
            // GEMV_SYMBOL_A / _B, GEMV_BLOCK_A / _B, GEMV_GRID_DIV_A / _B override per arm.
            let sfx = if mi == 0 { "A" } else { "B" };
            let sym = std::env::var(format!("GEMV_SYMBOL_{sfx}")).unwrap_or_else(|_| "gemv_mq4g256v2".into());
            let block: u32 = std::env::var(format!("GEMV_BLOCK_{sfx}")).ok().and_then(|v| v.parse().ok()).unwrap_or(32);
            let grid_div: u32 = std::env::var(format!("GEMV_GRID_DIV_{sfx}")).ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let kern = Kernel::find(module, &sym).unwrap_or_else(|| panic!("{sym}"));
            let gran = redline::dispatch::lds_granularity(&dev.info.gfx_arch);
            let run = |n: usize| -> f64 {
                let mut cb = CommandBuffer::new();
                for i in 0..n {
                    cb.dispatch_lds(kern, [(m + grid_div - 1) / grid_div, 1, 1], [block, 1, 1], d_ka.gpu_addr, 0, gran);
                    // Production gfx12 tape inter-node barrier (redline-rocr pm4.rs).
                    cb.push_raw(&[0xc000_4600, 0x407]);
                    cb.push_raw(&[0xc006_5800, 0, u32::MAX, 0x00ff_ffff, 0, 0, 0x0000_000a, 0x10180]);
                    let _ = (i, mi);
                }
                let bos = [&module.code_buf, &d_w, &d_x, &d_y, &d_ka, &fence];
                dq.submit(&dev, &cb, &bos).unwrap(); // warm
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let t = Instant::now();
                    dq.submit(&dev, &cb, &bos).unwrap();
                    best = best.min(t.elapsed().as_secs_f64() * 1e6);
                }
                best
            };
            let t1 = run(1);
            let tn = run(N);
            let us = (tn - t1) / (N as f64 - 1.0);
            let mut y = vec![0u8; m as usize * 4];
            dev.download(&d_y, &mut y).unwrap();
            let bytes = wbytes as f64 + k as f64 * 4.0 + m as f64 * 4.0;
            eprintln!(
                "M={m} K={k} arm{mi} {}: {us:.2} us/dispatch  {:.0} GB/s  (t1={t1:.0}us tN={tn:.0}us)",
                args[mi].rsplit('/').next().unwrap(),
                bytes / us / 1e3
            );
            ys.push(y);
            per.push(us);
        }
        let diff = ys[0].chunks(4).zip(ys[1].chunks(4)).filter(|(a, b)| a != b).count();
        eprintln!("M={m} K={k} ratio b/a={:.3} y-bit-diff rows={diff}", per[1] / per[0]);
    }
}
