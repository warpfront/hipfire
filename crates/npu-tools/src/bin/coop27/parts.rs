// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! `--mode parts`: can amdxdna hold two hardware contexts on disjoint column partitions and run them without a context
//! switch? The design is the M2 single-core int8 GEMM (`gemm_i8::design(&[0], 64, 64, 64)`, partition-relative column 0,
//! the first Halo gate shape), a one-shot command (npu-gemm runs it on a fresh context per submit; resubmitting it on the
//! same context does not complete, measured W5), so every round creates two fresh contexts of `--part-tiles P` tiles each
//! (32 = whole array) and, CPU clock submit -> completion:
//! * `serial`: submit A, wait, submit B, wait;
//! * `overlap`: submit A and B back to back, then wait both (concurrent only if the firmware runs both partitions).
//! Every output is compared with the exact CPU reference.
use super::*;
use peacemaker_npu::kernels::gemm_i8::{cpu_reference, design, Design};

struct Side<'d> {
    ctx: HwCtx<'d>,
    _pdi: Bo,
    insts: Bo,
    inp: Bo,
    out: Bo,
    cmd: Bo,
}

fn side<'d>(dev: &'d Device, d: &Design, tiles: u32, input: &[u8]) -> Result<Side<'d>, String> {
    let ctx = HwCtx::create(dev, tiles, 2048)?;
    let pdi = dev.dev_bo(&d.pdi)?;
    ctx.config_cu(&pdi)?;
    let insts = dev.dev_bo(&d.insts)?;
    let mut inp = dev.shmem_bo(d.args[0].bytes)?;
    inp.as_mut_slice().copy_from_slice(input);
    inp.flush();
    let mut out = dev.shmem_bo(d.args[1].bytes)?;
    out.as_mut_slice().fill(POISON);
    out.flush();
    let cmd = dev.cmd_bo()?;
    Ok(Side {
        ctx,
        _pdi: pdi,
        insts,
        inp,
        out,
        cmd,
    })
}

impl Side<'_> {
    fn submit(&mut self) -> Result<u64, String> {
        self.ctx
            .submit(&mut self.cmd, &self.insts, &[&self.inp, &self.out])
    }
    fn wait(&self, seq: u64, timeout: u64) -> Result<(), String> {
        let st = self.ctx.wait(&self.cmd, seq, timeout)?;
        if st == ERT_STATE_COMPLETED {
            Ok(())
        } else {
            Err(format!("state {st}"))
        }
    }
    fn exact(&self, d: &Design, want: &[i32]) -> bool {
        self.out.flush();
        d.unpack_out(self.out.as_slice()) == want
    }
}

pub fn run(cfg: &Cfg) -> Result<bool, String> {
    let (m, n, k) = (64, 64, 64);
    let d = design(&[0], m, n, k);
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 37 + 11) % 15) as i8 - 7).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 53 + 5) % 15) as i8 - 7).collect();
    let want = cpu_reference(&a, &b, m, n, k);
    let input = d.pack_in(&a, &b).remove(0);
    let mut dev = Device::open()?;
    dev.map_heap(64 << 20)?;
    let all = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    println!("parts: device {} cols x {} core rows = {all} tiles; design gemm_i8 single core {m}x{n}x{k}", dev.meta.cols, dev.meta.core.0);
    let mut ok = true;
    for &tiles in &cfg.part_tiles {
        let tiles = tiles as u32;
        for mode in ["serial", "overlap"] {
            let mut busy = Vec::new();
            let mut good = true;
            for _ in 0..cfg.iters {
                let mut sa = match side(&dev, &d, tiles, &input) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("PARTS tiles={tiles}: context A refused: {e}");
                        good = false;
                        break;
                    }
                };
                let mut sb = match side(&dev, &d, tiles, &input) {
                    Ok(s) => s,
                    Err(e) => {
                        println!("PARTS tiles={tiles}: context B refused while A alive: {e}");
                        good = false;
                        break;
                    }
                };
                let t = Instant::now();
                if mode == "serial" {
                    let q = sa.submit()?;
                    sa.wait(q, cfg.timeout_ms)?;
                    let q = sb.submit()?;
                    sb.wait(q, cfg.timeout_ms)?;
                } else {
                    let (qa, qb) = (sa.submit()?, sb.submit()?);
                    sa.wait(qa, cfg.timeout_ms)?;
                    sb.wait(qb, cfg.timeout_ms)?;
                }
                busy.push(t.elapsed().as_secs_f64() * 1e6);
                good &= sa.exact(&d, &want) && sb.exact(&d, &want);
            }
            if busy.is_empty() {
                ok = false;
                continue;
            }
            println!("PARTS tiles={tiles} mode={mode} rounds={} pair_median_us={:.1} all=[{}] exact={good}", busy.len(), median(&busy),
                busy.iter().map(|x| format!("{x:.0}")).collect::<Vec<_>>().join(","));
            ok &= good;
        }
    }
    Ok(ok)
}
