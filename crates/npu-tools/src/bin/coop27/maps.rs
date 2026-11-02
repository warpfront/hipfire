// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! `--mode maps`: the shared-page mapping matrix (no NPU command). For every backing x `hipHostRegister` flags:
//! amdxdna userptr import (accepted?), hipHostRegister (accepted?), the alias verdicts (CPU->GPU, GPU->CPU, NPU map ==
//! CPU, NPU map -> CPU, GPU re-read after a foreign write = stale-cache detector), raw PM `gpu-bw` read / write GB/s on
//! the exact pages, and the co-op kernels on them: down `ief_pack` (n=1024, write-back, 303 MB, byte-checked against the
//! CPU pack), down `ief_addc` (reads a 33.5 MB NPU C from the pages) and gate_up `ief_silu` (reads 64 MB).
//! Backings: `4k` (MADV_NOHUGEPAGE), `thp` (2 MiB aligned, MADV_HUGEPAGE), `huge2m` (MAP_HUGETLB 2 MiB), `huge1g`
//! (MAP_HUGETLB 1 GiB); hugetlb needs pages reserved by the caller (the window script), else the row reports mmap
//! refused. Flags: 0 default, 0x1 Portable, 0x2 Mapped, 0x8 hipExtHostRegisterCoarseGrained.
use super::*;

pub(super) const BW_CO: &[u8] = include_bytes!("../../../native/gpu_bw_gfx1151.co");
const ARENA: usize = 320 << 20;

mod mm {
    use core::ffi::c_void;
    extern "C" {
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }
    pub const MADV_HUGEPAGE: i32 = 14;
    pub const MADV_NOHUGEPAGE: i32 = 15;
    pub const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;
    pub const MAP_HUGETLB: i32 = 0x40000;
    pub const MAP_POPULATE: i32 = 0x8000;
    pub const HUGE_SHIFT: i32 = 26;
}

/// One mapped arena with its backing description; unmapped on drop (after the registrations, which are dropped first).
struct Arena {
    host: *mut u8,
    map: *mut u8,
    map_len: usize,
    len: usize,
}
impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { libc_mm::munmap(self.map as *mut core::ffi::c_void, self.map_len) };
    }
}

fn map_arena(backing: &str, len: usize) -> Result<Arena, String> {
    use core::ffi::c_void;
    let (flags, map_len, align) = match backing {
        "4k" | "thp" => (mm::MAP_PRIVATE_ANON, len + (2 << 20), 2usize << 20),
        "huge2m" => (
            mm::MAP_PRIVATE_ANON | mm::MAP_HUGETLB | (21 << mm::HUGE_SHIFT),
            len.div_ceil(2 << 20) * (2 << 20),
            1,
        ),
        "huge1g" => (
            mm::MAP_PRIVATE_ANON | mm::MAP_HUGETLB | (30 << mm::HUGE_SHIFT),
            len.div_ceil(1 << 30) * (1 << 30),
            1,
        ),
        b => return Err(format!("backing {b}")),
    };
    let p = unsafe {
        libc_mm::mmap(
            core::ptr::null_mut(),
            map_len,
            libc_mm::PROT_RW,
            flags | mm::MAP_POPULATE * i32::from(align == 1),
            -1,
            0,
        )
    };
    if p as usize == usize::MAX {
        return Err(format!("mmap refused: {}", std::io::Error::last_os_error()));
    }
    let host = ((p as usize).div_ceil(align) * align) as *mut u8;
    if align > 1 {
        let adv = if backing == "thp" {
            mm::MADV_HUGEPAGE
        } else {
            mm::MADV_NOHUGEPAGE
        };
        if unsafe { mm::madvise(host as *mut c_void, len, adv) } != 0 {
            return Err(format!("madvise: {}", std::io::Error::last_os_error()));
        }
        unsafe { core::ptr::write_bytes(host, 0, len) };
    }
    unsafe { clflush(host, len) };
    Ok(Arena {
        host,
        map: p as *mut u8,
        map_len,
        len,
    })
}

/// `(AnonHugePages kB, KernelPageSize kB)` of the mapping containing `p`, from /proc/self/smaps.
fn page_info(p: *const u8) -> (u64, u64) {
    let s = std::fs::read_to_string("/proc/self/smaps").unwrap_or_default();
    let (mut inside, mut huge, mut kps) = (false, 0, 0);
    for line in s.lines() {
        if let Some((range, _)) = line
            .split_once(' ')
            .filter(|(r, _)| r.contains('-') && !r.ends_with(':'))
        {
            let (a, b) = range.split_once('-').unwrap();
            let (a, b) = (
                u64::from_str_radix(a, 16).unwrap_or(0),
                u64::from_str_radix(b, 16).unwrap_or(0),
            );
            if inside {
                break;
            }
            inside = (a..b).contains(&(p as u64));
        } else if inside {
            let v = |l: &str| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|x| x.parse().ok())
                    .unwrap_or(0)
            };
            if line.starts_with("AnonHugePages:") {
                huge = v(line);
            }
            if line.starts_with("KernelPageSize:") {
                kps = v(line);
            }
        }
    }
    (huge, kps)
}

pub fn run(cfg: &Cfg) -> Result<bool, String> {
    let inp = load_inputs(cfg)?;
    let gpu = Gpu::new()?;
    let rt = &gpu.rt;
    let m_bw = rt.load_module(BW_CO)?;
    let (f_read, f_write) = (m_bw.function("read_bw")?, m_bw.function("write_bw")?);
    let blocks = rt.compute_units()? * 16;
    let threads = u64::from(blocks) * 256;
    let sink = rt.allocate(threads as usize * 16, None)?;
    let mut dev = Device::open()?;
    dev.map_heap(64 << 20)?;
    // Down n=1024 geometry and CPU A stream; gate_up 2048-feature geometry (C size only).
    let o = oracle(&inp, 1024, false)?;
    let gd = Ief15Gemm::new(T, 2048, 5120, Control::Fast)?;
    let xq = gpu.up(&inp.xq)?;
    let x = gpu.up(&inp.old)?;
    let hbuf = rt.allocate(T * gate_up::H * 4, None)?;
    let dc = rt.allocate(E * T * 2, None)?;
    let bc = rt.allocate(T * 2, None)?;
    let gu = gate_up::silu_kernel(&gpu)?;
    let (e0, e1) = (rt.event()?, rt.event()?);
    let s = |b: bool| if b { "ok" } else { "MISMATCH" };
    let mut best: Option<(f64, String)> = None;
    for backing in ["4k", "thp", "huge2m", "huge1g"] {
        for flags in [0u32, 0x1, 0x2, 0x8] {
            let tag = format!("backing={backing} flags={flags:#x}");
            let arena = match map_arena(backing, ARENA) {
                Ok(a) => a,
                Err(e) => {
                    println!("MAP {tag}: {e}");
                    continue;
                }
            };
            let (huge_kb, kps) = page_info(arena.host);
            let npu = dev.userptr_bo(arena.host, arena.len);
            let npu_ok = npu.is_ok();
            let reg = rt.register_host_flags(arena.host, arena.len, flags);
            let reg = match reg {
                Ok(r) => r,
                Err(e) => {
                    println!("MAP {tag} thp_kB={huge_kb} kpage_kB={kps} npu_userptr={npu_ok}: hipHostRegister refused: {e}");
                    continue;
                }
            };
            let va = reg.ptr();
            // Alias verdicts on the first 64 KiB (GPU copy kernel, CPU with clflush, NPU BO mapping).
            const AB: usize = 64 << 10;
            let dbuf = rt.allocate(AB, None)?;
            let pat = |salt: u32| -> Vec<u8> {
                (0..AB as u32)
                    .map(|i| {
                        (i.wrapping_mul(2654435761)
                            .wrapping_add(salt.wrapping_mul(0x9e3779b9))
                            >> 13) as u8
                    })
                    .collect()
            };
            let copy = |dst: u64, src: u64| -> Result<(), String> {
                let mut a = coop_gpu::copy_args(
                    dst,
                    src,
                    (AB / 16) as u64,
                    (16 * coop_gpu::COPY_BLOCK) as u64,
                );
                rt.launch(&gpu.copy, 16, coop_gpu::COPY_BLOCK, &mut a)?;
                rt.synchronize()
            };
            let cpu_write = |v: &[u8]| unsafe {
                core::ptr::copy_nonoverlapping(v.as_ptr(), arena.host, AB);
                clflush(arena.host, AB)
            };
            let cpu_read = || unsafe {
                clflush(arena.host, AB);
                core::slice::from_raw_parts(arena.host, AB).to_vec()
            };
            let p1 = pat(1);
            cpu_write(&p1);
            copy(dbuf.ptr(), va)?;
            let c2g = gpu.down(&dbuf, AB)? == p1;
            let p2 = pat(2);
            rt.upload(&dbuf, &p2)?;
            copy(va, dbuf.ptr())?;
            let g2c = cpu_read() == p2;
            let (nview, n2c) = match &npu {
                Ok(bo) => {
                    bo.flush();
                    let v = bo.as_slice()[..AB] == p2[..];
                    let p3 = pat(3);
                    unsafe { core::ptr::copy_nonoverlapping(p3.as_ptr(), bo.host, AB) };
                    bo.flush();
                    (v, cpu_read() == p3)
                }
                Err(_) => {
                    cpu_write(&pat(3));
                    (false, false)
                }
            };
            copy(dbuf.ptr(), va)?;
            let reread = gpu.down(&dbuf, AB)? == pat(3);
            // Raw GPU bandwidth on the whole arena (PM gpu-bw kernels, 1 warm-up + 4 timed).
            let count = (arena.len / 16) as u64;
            let bw = |f: &HipFunction| -> Result<f64, String> {
                let mut ka = [0u8; 32];
                put64(&mut ka, 0, va);
                put64(&mut ka, 8, count);
                put64(&mut ka, 16, sink.ptr());
                put64(&mut ka, 24, threads);
                rt.launch(f, blocks, 256, &mut ka.clone())?;
                rt.record(&e0)?;
                for _ in 0..4 {
                    rt.launch(f, blocks, 256, &mut ka.clone())?;
                }
                rt.record(&e1)?;
                let ms = f64::from(rt.elapsed_ms(&e0, &e1)?) / 4.0;
                Ok(arena.len as f64 / ms / 1e6)
            };
            let (rd, wr) = (bw(&f_read)?, bw(&f_write)?);
            // Co-op kernels on the pages: down pack (wb) byte-checked, down addc and gate_up SiLU epilogue reads.
            unsafe {
                core::ptr::write_bytes(arena.host, POISON, arena.len);
                clflush(arena.host, arena.len)
            };
            gpu.db(xq.ptr(), dc.ptr(), bc.ptr(), E)?;
            gpu.pack(xq.ptr(), va, dc.ptr(), bc.ptr(), &o.g, true)?;
            rt.synchronize()?;
            let pack_ok = unsafe {
                clflush(arena.host, o.a_stream.len());
                core::slice::from_raw_parts(arena.host, o.a_stream.len())
            } == &o.a_stream[..];
            let time = |f: &dyn Fn() -> Result<(), String>| -> Result<f64, String> {
                f()?;
                rt.record(&e0)?;
                for _ in 0..3 {
                    f()?;
                }
                rt.record(&e1)?;
                Ok(f64::from(rt.elapsed_ms(&e0, &e1)?) / 3.0)
            };
            let t_pack = time(&|| gpu.pack(xq.ptr(), va, dc.ptr(), bc.ptr(), &o.g, true))?;
            let t_addc = time(&|| gpu.addc(va, x.ptr(), &o.g))?;
            let t_silu = time(&|| gu.epi(&gpu, va, hbuf.ptr(), 0, &gd))?;
            let alias_ok = c2g && g2c && nview && n2c && reread;
            println!("MAP {tag} thp_kB={huge_kb} kpage_kB={kps} npu_userptr={} hipreg=ok alias: cpu->gpu {} gpu->cpu {} npu==cpu {} npu->cpu {} \
                reread {} | gpu_bw read={rd:.1} write={wr:.1} GB/s | down pack_wb={t_pack:.3} ms ({}) addc={t_addc:.3} ms | gate_up silu_epi={t_silu:.3} ms",
                if npu_ok { "ok" } else { "REFUSED" }, s(c2g), s(g2c), s(nview), s(n2c), s(reread), if pack_ok { "exact" } else { "PACK MISMATCH" });
            if alias_ok && npu_ok && pack_ok && best.as_ref().is_none_or(|(b, _)| rd.min(wr) > *b) {
                best = Some((rd.min(wr), tag.clone()));
            }
            drop(reg);
            drop(npu);
            drop(arena);
        }
    }
    match &best {
        Some((bw, tag)) => {
            println!("MAPBEST alias-correct with NPU import: {tag} min(read,write)={bw:.1} GB/s")
        }
        None => println!("MAPBEST none alias-correct"),
    }
    Ok(best.is_some())
}
