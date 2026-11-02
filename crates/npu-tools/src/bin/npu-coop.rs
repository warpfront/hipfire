// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! npu-coop: CPU-free GPU -> NPU -> GPU proof on the persistent ring (`docs/npu/railgun-npu.md` §3.1, §4).
//!
//! usage: npu-coop [--mode empty|gemm|alias] [--shape M N K] [--slots S] [--nslots R] [--rounds N] [--timeout-ms T]
//!
//! `--mode alias` is the permanent transport gate (no NPU command): the three-way CPU / GPU-kernel / NPU-mapping alias
//! check must PASS on the userptr transport and must FAIL (detect) on the old HIP VMM dma-buf export path.
//!
//! The Halo iGPU (PCI `0000:bf:00.0`, gfx1151, PM-native kernels of `npu_tools::coop_gpu`) is the producer AND the
//! consumer; the CPU only builds, submits / re-arms the persistent NPU command once per round, enqueues the whole GPU
//! stream of the round ahead of time, then waits for both and verifies.
//!
//! Memory: the ring, the A arena and the C arena are anonymous host pages owned by this process, pinned by the NPU
//! as a userptr BO (`Device::userptr_bo`) and registered with HIP for the GPU (`HipRuntime::register_host`); the CPU
//! uses them directly (clflush around NPU-visible accesses). A HIP VMM host allocation exported as dma-buf is NOT
//! usable for the ring: after the import, its GPU writes are not seen through the dma-buf and vice versa (measured,
//! coop-w2 alias probe, `docs/coop-dense27b.md`). Before any NPU command a three-way aliasing check (CPU, GPU kernel,
//! NPU mapping) must pass. B (weights) is an NPU shmem BO; GPU-only buffers (A sources, C readback, publish records)
//! are `hipMalloc` device memory.
//!
//! NPU: ONE persistent command of `S` runs of G80 where the shape fits (N = 1280, K <= 2560), else of the A-repeat V9
//! (`ArrayDesign::with_a_repeat`: the A arena holds each run's A once, the shim replays it per N-wave) (`persistent_with`, `Bodies::Lean`: run 0 full, runs 1.. lean; rounds
//! after the first `Bodies::LeanFirst`, every run lean; `--mode empty`: `empty_persistent`, poll + done only), run `j`
//! on slot `j % R` with fixed slot arenas; seq of (round r, run j) is `1 + r*S + j`; rounds re-arm with `patch_seq`
//! (needs `S % R == 0`). Rounds are queued one ahead (`docs/npu/railgun-npu.md` 3.3): round r+1's command and GPU work
//! are issued (two retained command / insts copies) before the CPU waits for round r, so the next command's start-up
//! overlaps the current one. A round's wall time is the interval between consecutive round completions (NPU command
//! done and the consumer's last C read); every check below runs after the last round.
//!
//! GPU work per run j, on two in-order streams (`coop_publish` doubles as a wait,
//! `v` stored to a scratch line while polling `addr == v`, and as a store, polling its own line):
//! * producer: wait consumed(slot) == seq - R (slot released; skipped while seq <= R); gemm: `coop_copy` A source
//!   (operand set `((r*S + j) / R) % 2`, device memory) -> A arena slot (GPU-produced A; every reuse of a slot gets the
//!   other set, so a stale A cannot pass); store seq to the slot line (glc slc dlc + vscnt 0, the
//!   WRITE_DATA-with-confirm equivalent). It runs up to R runs ahead, so the next seq is ready while the NPU computes.
//! * consumer: poll done(slot) == seq (realtime stamp before every load); gemm: `coop_copy` C arena slot -> device
//!   readback (the GPU reads the NPU's C); store seq to consumed(slot) (device memory).
//!
//! Latencies, GPU realtime clock (`hipDeviceAttributeWallClockRate`), p50 / min / max over all runs of all rounds, from
//! the consumer's done wait (it starts when the previous run's C is read, so with the producer ahead these are
//! consumer-side wait times, not the ring round trip):
//! * `pub_to_done_gpu_us`: consumer wait start -> done line seen by the GPU;
//! * `done_to_gpu_obs_upper_us`: t_obs - issue time of the last poll load that missed; the done line became visible
//!   in that window, so this bounds done -> GPU observe from above;
//! * `poll_load_us`: issue -> return of the load that hit (one uncached load round trip, the observe floor);
//! * `publish_store_us`: scratch store issue -> store complete;
//! * per-run wall time (CPU, round / S) and, gemm, useful TOPS of the whole CPU-free pipeline.
//!
//! Exactness (gemm): two eager submits of the same design (one per operand set, shared B) on the same context first, each
//! compared with the exact closed-form CPU reference; then every run's GPU-read C (device readback) must be byte-equal
//! to the eager C of its set, every slot of the A arena must finally hold the GPU-copied A bytes of its last run,
//! and round 0 / the final round are also unpacked and compared with the CPU reference directly. Every consumer record
//! must carry status ok and its seq, every producer wait / publish / release kernel status ok and its value, and every
//! final done line seq, slot and 'DONE'. If the NPU command does not
//! complete, the CPU rescues it (publishes the missing seqs on free slots) so the array is never
//! left waiting, and the run FAILS. Exit 0 = PASS, 1 = FAIL, 2 = usage.
//!
//! Fabric clock: every mode (alias included) first takes `railgun::npu::fclk::FabricClockGuard`; by default it refuses
//! (exit 1) unless the Halo iGPU fclk is pinned, as inside `tools/npu/npu-window.sh`. `NPU_FCLK_GUARD=pin` pins it for
//! this process and restores the prior perf level on exit; `NPU_FCLK_GUARD=off` skips the check (loud warning).
use npu_tools::coop_gpu::{self, PublishRecord};
use peacemaker_npu::kernels::gemm_array::{
    apply_epilogue, design_v9, ArrayDesign, V8_DEFAULT_EPILOGUE,
};
use peacemaker_npu::kernels::gemm_core::Control;
use peacemaker_npu::kernels::gemm_g80::design_g80;
use peacemaker_npu::kernels::ring::{
    empty_persistent, patch_seq, persistent_with, Bodies, RingLayout, SlotPlan, DONE_MAGIC,
};
use railgun::npu::hip_runtime::{HipFunction, HipMemoryKind, HipRuntime, HostRegistration};
use railgun::npu::{clflush, Bo, Device, HwCtx, ERT_STATE_COMPLETED};
use std::time::{Duration, Instant};

const USAGE: &str = "usage: npu-coop [--mode empty|gemm|alias] [--shape M N K] [--slots S] [--nslots R] [--rounds N] [--timeout-ms T]";
const MAX_OPC: u32 = 2048;
const POISON: u8 = 0xA5;
const SETS: usize = 2;
/// GPU copy grid (blocks of 256 lanes).
const COPY_BLOCKS: u32 = 160;

struct Cfg {
    gemm: bool,
    alias: bool,
    m: usize,
    n: usize,
    k: usize,
    slots: usize,
    nslots: usize,
    rounds: usize,
    timeout_ms: u64,
}

mod libc_mm {
    use core::ffi::c_void;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
    pub const PROT_RW: i32 = 3;
    /// MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE
    pub const MAP_ANON_POPULATE: i32 = 0x02 | 0x20 | 0x8000;
}

/// Anonymous host pages shared by the three devices: pinned by the NPU (userptr BO), registered with HIP (GPU VA),
/// used directly by the CPU. Drop order: HIP registration, NPU BO (unpin), then munmap.
struct SharedMem {
    host: *mut u8,
    len: usize,
    gpu: Option<HostRegistration>,
    npu: Option<Bo>,
}
impl SharedMem {
    fn new(rt: &HipRuntime, dev: &Device, bytes: usize) -> Result<SharedMem, String> {
        let len = bytes.div_ceil(4096) * 4096;
        let p = unsafe {
            libc_mm::mmap(
                core::ptr::null_mut(),
                len,
                libc_mm::PROT_RW,
                libc_mm::MAP_ANON_POPULATE,
                -1,
                0,
            )
        };
        if p as usize == usize::MAX {
            return Err(format!(
                "mmap anonymous {len} B: {}",
                std::io::Error::last_os_error()
            ));
        }
        let mut s = SharedMem {
            host: p as *mut u8,
            len,
            gpu: None,
            npu: None,
        };
        s.npu = Some(dev.userptr_bo(s.host, len)?);
        s.gpu = Some(rt.register_host(s.host, len)?);
        Ok(s)
    }
    fn gpu(&self) -> u64 {
        self.gpu.as_ref().unwrap().ptr()
    }
    fn bo(&self) -> &Bo {
        self.npu.as_ref().unwrap()
    }
    fn read(&self, at: usize, len: usize) -> Vec<u8> {
        assert!(at + len <= self.len);
        unsafe {
            clflush(self.host.add(at), len);
            core::slice::from_raw_parts(self.host.add(at), len).to_vec()
        }
    }
    fn write(&self, at: usize, src: &[u8]) {
        assert!(at + src.len() <= self.len);
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.host.add(at), src.len());
            clflush(self.host.add(at), src.len());
            core::arch::x86_64::_mm_sfence();
        }
    }
    fn rd32(&self, at: usize) -> u32 {
        u32_at(&self.read(at, 4), 0)
    }
}
impl Drop for SharedMem {
    fn drop(&mut self) {
        self.gpu = None;
        self.npu = None;
        unsafe { libc_mm::munmap(self.host as *mut core::ffi::c_void, self.len) };
    }
}

/// Bytes of the three-way alias check buffer.
const ALIAS_BYTES: usize = 64 << 10;

/// Three-way aliasing check of one shared buffer, all writes made AFTER the NPU import (no NPU command): `gpu` is the
/// GPU VA, `cpu` the CPU view, `npu` the NPU BO (its own mapping is the NPU's SVA view of the pages). Verdicts
/// `[cpu->gpu, gpu->cpu, npu map == cpu, npu map -> cpu]`: a CPU write is seen by a GPU kernel load; a GPU kernel store
/// is seen by the CPU; the NPU mapping shows it too; a write through the NPU mapping is seen by the CPU.
fn alias_check(
    rt: &HipRuntime,
    f_copy: &HipFunction,
    gpu: u64,
    cpu: *mut u8,
    npu: &Bo,
) -> Result<[bool; 4], String> {
    const BYTES: usize = ALIAS_BYTES;
    let dev_buf = rt.allocate(BYTES, None)?;
    let pat = |salt: u32| -> Vec<u8> {
        (0..BYTES as u32)
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
            (BYTES / 16) as u64,
            (16 * coop_gpu::COPY_BLOCK) as u64,
        );
        rt.launch(f_copy, 16, coop_gpu::COPY_BLOCK, &mut a)?;
        rt.synchronize()
    };
    let cpu_write = |src: &[u8]| unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), cpu, BYTES);
        clflush(cpu, BYTES);
        core::arch::x86_64::_mm_sfence()
    };
    let cpu_read = || unsafe {
        clflush(cpu, BYTES);
        core::slice::from_raw_parts(cpu, BYTES).to_vec()
    };
    let p1 = pat(1);
    cpu_write(&p1);
    copy(dev_buf.ptr(), gpu)?;
    let mut got = vec![0u8; BYTES];
    rt.download(&dev_buf, &mut got)?;
    let cpu_to_gpu = got == p1;
    let p2 = pat(2);
    rt.upload(&dev_buf, &p2)?;
    copy(gpu, dev_buf.ptr())?;
    let gpu_to_cpu = cpu_read() == p2;
    npu.flush();
    let npu_view = npu.as_slice()[..BYTES] == p2[..];
    let p3 = pat(3);
    unsafe { core::ptr::copy_nonoverlapping(p3.as_ptr(), npu.host, BYTES) };
    npu.flush();
    let npu_to_cpu = cpu_read() == p3;
    Ok([cpu_to_gpu, gpu_to_cpu, npu_view, npu_to_cpu])
}

fn verdicts(v: [bool; 4]) -> String {
    let s = |b: bool| if b { "ok" } else { "MISMATCH" };
    format!(
        "cpu->gpu {} gpu->cpu {} npu-map==cpu {} npu-map->cpu {}",
        s(v[0]),
        s(v[1]),
        s(v[2]),
        s(v[3])
    )
}

/// The co-op transport: anonymous host pages (NPU userptr BO + hipHostRegister). Must alias all four ways.
fn check_userptr(rt: &HipRuntime, dev: &Device, f_copy: &HipFunction) -> Result<[bool; 4], String> {
    let sh = SharedMem::new(rt, dev, ALIAS_BYTES)?;
    alias_check(rt, f_copy, sh.gpu(), sh.host, sh.bo())
}

/// The old transport (negative control): a HIP VMM uncached host allocation exported with
/// `hipMemGetHandleForAddressRange` and imported by amdxdna; the CPU view is the import's mapping. Measured not to
/// alias after the import (coop-w2), so the check must FAIL here.
fn check_vmm_export(
    rt: &HipRuntime,
    dev: &Device,
    f_copy: &HipFunction,
) -> Result<[bool; 4], String> {
    let buf = rt.allocate(ALIAS_BYTES, Some(HipMemoryKind::VmmHostUncached))?;
    let fd = rt.export_dmabuf(&buf)?;
    let bo = dev
        .import_dmabuf(fd.fd(), ALIAS_BYTES)
        .map_err(|e| format!("vmm export import: {e}"))?;
    alias_check(rt, f_copy, buf.ptr(), bo.host, &bo)
}

/// `--mode alias` (no NPU command): the userptr transport must pass all four checks AND the old VMM export path must
/// fail at least one (the check detects the broken transport). Exit 0 only when both hold.
fn run_alias(rt: &HipRuntime, dev: &Device, f_copy: &HipFunction) -> Result<bool, String> {
    let good = check_userptr(rt, dev, f_copy)?;
    let old = check_vmm_export(rt, dev, f_copy)?;
    println!(
        "alias: userptr (anon pages, NPU userptr BO + hipHostRegister): {} -> {}",
        verdicts(good),
        if good.iter().all(|&b| b) {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "alias: vmm-export (HIP VMM uncached host, dma-buf export, negative control): {} -> {}",
        verdicts(old),
        if old.iter().all(|&b| b) {
            "NOT DETECTED (check broken or export now aliases)"
        } else {
            "detected (expected)"
        }
    );
    Ok(good.iter().all(|&b| b) && !old.iter().all(|&b| b))
}

fn parse(args: &[String]) -> Result<Cfg, String> {
    let mut c = Cfg {
        gemm: true,
        alias: false,
        m: 512,
        n: 1280,
        k: 2560,
        slots: 8,
        nslots: 4,
        rounds: 4,
        timeout_ms: 3000,
    };
    let mut it = args.iter();
    let num = |s: Option<&String>, what: &str| -> Result<usize, String> {
        s.ok_or_else(|| format!("{what} needs a value"))?
            .parse::<usize>()
            .map_err(|_| format!("{what}: bad integer"))
    };
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => match it.next().map(String::as_str) {
                Some("empty") => c.gemm = false,
                Some("gemm") => c.gemm = true,
                Some("alias") => c.alias = true,
                o => return Err(format!("--mode expects empty|gemm|alias, got {o:?}")),
            },
            "--shape" => {
                c.m = num(it.next(), "--shape M")?;
                c.n = num(it.next(), "--shape N")?;
                c.k = num(it.next(), "--shape K")?;
            }
            "--slots" => c.slots = num(it.next(), "--slots")?,
            "--nslots" => c.nslots = num(it.next(), "--nslots")?,
            "--rounds" => c.rounds = num(it.next(), "--rounds")?,
            "--timeout-ms" => c.timeout_ms = num(it.next(), "--timeout-ms")? as u64,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0)
            }
            o => return Err(format!("unknown argument {o:?}")),
        }
    }
    if c.slots == 0 || c.rounds == 0 || c.timeout_ms == 0 {
        return Err("--slots, --rounds, --timeout-ms must be >= 1".into());
    }
    if !c.nslots.is_power_of_two() || c.slots % c.nslots != 0 {
        return Err("--nslots must be a power of two dividing --slots".into());
    }
    Ok(c)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = match parse(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("npu-coop: {e}\n{USAGE}");
            std::process::exit(2)
        }
    };
    // Every mode runs GPU kernels against NPU-visible memory: fabric clock pinned for the whole run
    // (`railgun::npu::fclk`, `NPU_FCLK_GUARD`). The guard drops (and, mode `pin`, restores) before `exit`.
    let res = railgun::npu::fclk::FabricClockGuard::acquire("npu-coop").and_then(|_fclk| run(&cfg));
    match res {
        Ok(true) => println!("RESULT: PASS"),
        Ok(false) => {
            println!("RESULT: FAIL");
            std::process::exit(1)
        }
        Err(e) => {
            println!("error: {e}");
            println!("RESULT: FAIL");
            std::process::exit(1)
        }
    }
}

/// p50 / min / max.
fn pmm(v: &[f64]) -> String {
    if v.is_empty() {
        return "n/a".into();
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!(
        "p50={:.2} min={:.2} max={:.2} n={}",
        s[(s.len() - 1) / 2],
        s[0],
        s[s.len() - 1],
        s.len()
    )
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// Closed-form operand set `set` (npu-ring's form): A[r,kk] = ra(r) (depends on the set), B[kk,c] = cb(c) (shared
/// weights, identical for every set), C = epi(K ra cb) exactly.
fn operand_set(
    d: &ArrayDesign,
    m: usize,
    n: usize,
    k: usize,
    set: usize,
) -> ([Vec<u8>; 2], Vec<i32>) {
    let ra: Vec<i32> = (0..m)
        .map(|r| (7 + (r * 5 + set * 3) % 13) as i32 * if (r * 7 + set) % 3 == 0 { -1 } else { 1 })
        .collect();
    let cb: Vec<i32> = (0..n)
        .map(|c| (7 + (c * 3) % 13) as i32 * if (c * 5) % 4 == 1 { -1 } else { 1 })
        .collect();
    let a: Vec<i8> = (0..m * k).map(|i| ra[i / k] as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| cb[i % n] as i8).collect();
    let want = ra
        .iter()
        .flat_map(|&x| {
            cb.iter()
                .map(move |&y| apply_epilogue((k as i32).wrapping_mul(x * y), V8_DEFAULT_EPILOGUE))
        })
        .collect();
    (d.pack_in(&a, &b), want)
}

/// The NPU command did not complete: publish every run whose seq is not in its slot line yet and whose slot is free
/// (previous seq done), until the last run's done line appears or `timeout` passes. Never overwrites a busy slot.
fn rescue(ring: &SharedMem, layout: RingLayout, seq0: u32, runs: usize, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    let (r, last) = (layout.nslots as u32, seq0 + runs as u32 - 1);
    while ring.rd32(layout.done_line((runs - 1) % layout.nslots)) != last
        && Instant::now() < deadline
    {
        for j in 0..runs {
            let (seq, slot) = (seq0 + j as u32, j % layout.nslots);
            let cur = ring.rd32(layout.slot_line(slot));
            let free = seq <= r || ring.rd32(layout.done_line(slot)) == seq - r;
            if (cur.wrapping_sub(seq) as i32) < 0 && free {
                ring.write(layout.slot_line(slot), &seq.to_le_bytes());
            }
        }
    }
}

fn run(cfg: &Cfg) -> Result<bool, String> {
    let gemm = cfg.gemm;
    let (m, n, k) = if gemm {
        (cfg.m, cfg.n, cfg.k)
    } else {
        (512, 512, 64)
    };
    let (s_runs, r_slots) = (cfg.slots, cfg.nslots);
    let layout = RingLayout { nslots: r_slots };
    // NPU design: G80 (128x80 core, N = 1280 exactly, K <= 2560; the faster exact gate_up design) where the shape fits,
    // else the A-repeat V9: the GPU writes each run's A once and the shim replays it per N-wave (gemm_array
    // `with_a_repeat`), instead of the default packing that replicates A NW times in the shared arena.
    // Every run multiplies the same B (shared weights, as in chunked prefill through one layer): on G80 the lean runs
    // keep it resident in the memtile instead of refilling it from DDR (ring `b_shared`).
    let g80 = n == 1280 && k % 64 == 0 && k <= 2560;
    let v9 = || {
        if g80 {
            design_g80(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast)
        } else {
            design_v9(m, n, k, V8_DEFAULT_EPILOGUE, Control::Fast).with_a_repeat()
        }
    };
    let d = v9();
    let (a_bytes, c_bytes) = (d.args[0].bytes, d.args[2].bytes);
    let a_stride = a_bytes.div_ceil(4096) * 4096;
    let c_stride = c_bytes.div_ceil(4096) * 4096;
    let plans: Vec<SlotPlan> = (0..s_runs)
        .map(|j| {
            let s = j % r_slots;
            if gemm {
                SlotPlan {
                    slot: s,
                    a_off: (s * a_stride) as u64,
                    b_off: 0,
                    c_off: (s * c_stride) as u64,
                }
            } else {
                SlotPlan {
                    slot: s,
                    a_off: 0,
                    b_off: 0,
                    c_off: 0,
                }
            }
        })
        .collect();
    let pd = if gemm {
        persistent_with(v9(), layout, &plans, 1, Bodies::Lean, g80)
    } else {
        empty_persistent(layout, &plans, 1)
    };
    if pd.patch_sites.len() != 2 * s_runs {
        return Err(format!(
            "{} patch sites, expected {}",
            pd.patch_sites.len(),
            2 * s_runs
        ));
    }
    if gemm && pd.pdi != d.pdi {
        return Err("persistent PDI != eager design PDI".into());
    }
    let set_of = |r: usize, j: usize| ((r * s_runs + j) / r_slots) % SETS;

    // ---- GPU ----
    let pci = railgun::npu::fclk::igpu_pci();
    let rt = HipRuntime::load_for_pci(&pci)?;
    let arch = rt.device_arch()?;
    if arch != coop_gpu::ARCH {
        return Err(format!(
            "device {pci} is {arch}, kernels are {}-only",
            coop_gpu::ARCH
        ));
    }
    let khz = rt.wall_clock_khz()?;
    let us_per_tick = 1e3 / khz as f64;
    let module = rt.load_module(coop_gpu::CODE_OBJECT)?;
    let (f_copy, f_pub) = (
        module.function(coop_gpu::COPY)?,
        module.function(coop_gpu::PUBLISH)?,
    );
    let dev = {
        let mut dev = Device::open()?;
        dev.map_heap(64 << 20)
            .map_err(|e| format!("device heap map failed: {e}"))?;
        dev
    };
    if cfg.alias {
        return run_alias(&rt, &dev, &f_copy);
    }
    let v = check_userptr(&rt, &dev, &f_copy)?;
    println!(
        "alias: userptr (anon pages, NPU userptr BO + hipHostRegister): {}",
        verdicts(v)
    );
    if !v.iter().all(|&b| b) {
        return Err("shared host pages do not alias across CPU / GPU / NPU mapping; stop before any NPU command".into());
    }
    let ring_sh = SharedMem::new(&rt, &dev, layout.bytes())?;
    let a_sh = SharedMem::new(&rt, &dev, if gemm { r_slots * a_stride } else { 4096 })?;
    let c_sh = SharedMem::new(&rt, &dev, if gemm { r_slots * c_stride } else { 4096 })?;
    // Records and C readback for every run of every round: checked after the last round, so no check runs (or is
    // timed) between queued rounds.
    let stats = rt.allocate(cfg.rounds * s_runs * coop_gpu::PUBLISH_RECORD_BYTES, None)?;
    let src_a = rt.allocate(if gemm { SETS * a_bytes } else { 16 }, None)?;
    let readback = rt.allocate(
        if gemm {
            cfg.rounds * s_runs * c_bytes
        } else {
            16
        },
        None,
    )?;
    println!("gpu: arch={arch} pci={pci} CUs={} wall_clock={khz} kHz; ring/A/C = anon host pages {} / {} / {} B (NPU userptr BO, hipHostRegister)",
        rt.compute_units()?, ring_sh.len, a_sh.len, c_sh.len);
    let mut b_bo = dev.shmem_bo(d.args[1].bytes.max(4096))?;
    let sets: Vec<([Vec<u8>; 2], Vec<i32>)> = if gemm {
        (0..SETS).map(|s| operand_set(&d, m, n, k, s)).collect()
    } else {
        Vec::new()
    };
    let ntiles = dev.meta.cols as u32 * dev.meta.core.0 as u32;
    let ctx = HwCtx::create(&dev, ntiles, MAX_OPC)?;
    let pdi = dev.dev_bo(&pd.pdi)?;
    ctx.config_cu(&pdi)?;
    let mut eager: Vec<Vec<u8>> = Vec::new();
    if gemm {
        let insts = dev.dev_bo(&d.insts)?;
        let mut cmd = dev.cmd_bo()?;
        for (i, (p, want)) in sets.iter().enumerate() {
            let mut a = dev.shmem_bo(a_bytes)?;
            a.as_mut_slice().copy_from_slice(&p[0]);
            a.flush();
            let mut b = dev.shmem_bo(d.args[1].bytes)?;
            b.as_mut_slice().copy_from_slice(&p[1]);
            b.flush();
            let mut c = dev.shmem_bo(c_bytes)?;
            c.as_mut_slice().fill(POISON);
            c.flush();
            let seq = ctx.submit(&mut cmd, &insts, &[&a, &b, &c])?;
            let st = ctx.wait(&cmd, seq, cfg.timeout_ms)?;
            if st != ERT_STATE_COMPLETED {
                return Err(format!("eager set {i}: state {st}"));
            }
            c.flush();
            let bad = d
                .unpack_out(c.as_slice())
                .iter()
                .zip(want)
                .filter(|(x, y)| x != y)
                .count();
            println!("eager set {i}: mismatches vs CPU {bad}");
            if bad != 0 {
                return Err(format!("eager set {i} not exact"));
            }
            eager.push(c.as_slice().to_vec());
        }
        b_bo.as_mut_slice()[..d.args[1].bytes].copy_from_slice(&sets[0].0[1]);
        b_bo.flush();
        let src: Vec<u8> = sets
            .iter()
            .flat_map(|(p, _)| p[0].iter().copied())
            .collect();
        rt.upload(&src_a, &src)?;
        a_sh.write(0, &vec![POISON; a_sh.len]);
    }
    let mut img = vec![0u8; layout.bytes()];
    layout.initialize(&mut img);
    ring_sh.write(0, &img);
    let insts = dev.dev_bo(&pd.insts)?;
    // Rounds after the first: gemm the lean-first stream (the array is still configured by the previous round), empty
    // the same protocol stream. Two retained copies, so round r+1 is re-armed and queued while round r runs.
    let rearm = gemm.then(|| persistent_with(v9(), layout, &plans, 1, Bodies::LeanFirst, g80));
    let rearm_src = rearm.as_ref().unwrap_or(&pd);
    let mut rearm_bos = [
        (dev.dev_bo(&rearm_src.insts)?, rearm_src.insts.clone()),
        (dev.dev_bo(&rearm_src.insts)?, rearm_src.insts.clone()),
    ];
    let mut cmds = [dev.cmd_bo()?, dev.cmd_bo()?];
    let ring_gpu = ring_sh.gpu();
    // Two in-order streams. Producer: wait until the consumer released the slot (consumed[slot] == seq - R), copy A,
    // publish seq. Consumer: wait done == seq, read C, release the slot. The producer runs up to R runs ahead, so the
    // next seq is published while the NPU computes the current one. Both use `coop_publish`: a wait is a publish of
    // `v` to a scratch line polling `addr == v`; a store is a publish to `addr` polling the same line (one load).
    let (prod, cons) = (rt.stream()?, rt.stream()?);
    let consumed = rt.allocate(r_slots * 64, None)?;
    rt.memset(&consumed, 0)?;
    let scratch = rt.allocate(64, None)?;
    let aux = rt.allocate(
        cfg.rounds * 3 * s_runs * coop_gpu::PUBLISH_RECORD_BYTES,
        None,
    )?;
    let aux_rec = |r: usize, kind: usize, j: usize| {
        aux.ptr() + (((r * 3 + kind) * s_runs + j) * coop_gpu::PUBLISH_RECORD_BYTES) as u64
    };
    rt.memset(&stats, 0)?;
    rt.memset(&aux, 0)?;
    if gemm {
        c_sh.write(0, &vec![POISON; c_sh.len]);
    }
    rt.synchronize()?;
    let round_done = [rt.event()?, rt.event()?];
    let mut npu_seqs = [0u64; 2];
    // Poll cap per publish kernel (>= ~1 us per uncached load): a stalled round's S kernels x (free wait + done poll)
    // stay within about half the timeout, so the CPU rescue still runs inside it.
    let max_iters =
        (cfg.timeout_ms * 1000 / (4 * s_runs as u64)).clamp(1000, u32::MAX as u64) as u32;
    let publish = |st, seq_addr: u64, done_addr: u64, rec: u64, v: u32| -> Result<(), String> {
        let mut a = coop_gpu::publish_args(seq_addr, done_addr, rec, v, 0, max_iters, false);
        rt.launch_on(Some(st), &f_pub, 1, coop_gpu::LANE_BLOCK, &mut a)
    };

    let mut all_ok = true;
    let (mut pub_done, mut obs_upper, mut load_us, mut store_us, mut round_us, mut interval) = (
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let (
        mut c_ok,
        mut c_total,
        mut a_ok,
        mut a_total,
        mut cpu_ok,
        mut cpu_total,
        mut rec_ok,
        mut done_ok_n,
    ) = (
        0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
    );
    // Rounds are queued one ahead (docs/npu/railgun-npu.md 3.3): step t issues round t (re-arm, NPU submit, GPU stream)
    // and then completes round t-1 (NPU wait, GPU event, checks). A round's wall is the interval between consecutive
    // round completions; the first includes the pipeline start.
    let mut t_prev = Instant::now();
    for step in 0..=cfg.rounds {
        if step < cfg.rounds {
            let (r, b) = (step, step % 2);
            let seq0 = 1 + (r * s_runs) as u32;
            if r > 0 {
                // Re-arm: only the declared seq words change; this copy's previous round (r-2) has completed.
                let (bo, host) = &mut rearm_bos[b];
                patch_seq(host, &rearm_src.patch_sites, seq0);
                bo.as_mut_slice()[..host.len()].copy_from_slice(host);
                bo.flush();
            }
            let cur = if r == 0 { &insts } else { &rearm_bos[b].0 };
            npu_seqs[b] = ctx.submit(
                &mut cmds[b],
                cur,
                &[a_sh.bo(), &b_bo, c_sh.bo(), ring_sh.bo()],
            )?;
            for (j, p) in plans.iter().enumerate() {
                let seq = seq0 + j as u32;
                let release = consumed.ptr() + (p.slot * 64) as u64;
                if seq as usize > r_slots {
                    publish(
                        &prod,
                        scratch.ptr(),
                        release,
                        aux_rec(r, 0, j),
                        seq - r_slots as u32,
                    )?;
                }
                if gemm {
                    let mut a = coop_gpu::copy_args(
                        a_sh.gpu() + p.a_off,
                        src_a.ptr() + (set_of(r, j) * a_bytes) as u64,
                        (a_bytes / 16) as u64,
                        (COPY_BLOCKS * coop_gpu::COPY_BLOCK) as u64,
                    );
                    rt.launch_on(
                        Some(&prod),
                        &f_copy,
                        COPY_BLOCKS,
                        coop_gpu::COPY_BLOCK,
                        &mut a,
                    )?;
                }
                let slot_line = ring_gpu + layout.slot_line(p.slot) as u64;
                publish(&prod, slot_line, slot_line, aux_rec(r, 1, j), seq)?;
                publish(
                    &cons,
                    scratch.ptr(),
                    ring_gpu + layout.done_line(p.slot) as u64,
                    stats.ptr() + ((r * s_runs + j) * coop_gpu::PUBLISH_RECORD_BYTES) as u64,
                    seq,
                )?;
                if gemm {
                    let mut a = coop_gpu::copy_args(
                        readback.ptr() + ((r * s_runs + j) * c_bytes) as u64,
                        c_sh.gpu() + p.c_off,
                        (c_bytes / 16) as u64,
                        (COPY_BLOCKS * coop_gpu::COPY_BLOCK) as u64,
                    );
                    rt.launch_on(
                        Some(&cons),
                        &f_copy,
                        COPY_BLOCKS,
                        coop_gpu::COPY_BLOCK,
                        &mut a,
                    )?;
                }
                publish(&cons, release, release, aux_rec(r, 2, j), seq)?;
            }
            rt.record_on(&round_done[b], Some(&cons))?;
        }
        if step == 0 {
            continue;
        }
        let (r, b) = (step - 1, (step - 1) % 2);
        let seq0 = 1 + (r * s_runs) as u32;
        let st = ctx.wait(&cmds[b], npu_seqs[b], cfg.timeout_ms);
        if !matches!(st, Ok(s) if s == ERT_STATE_COMPLETED) {
            println!("round {r}: NPU command NOT completed ({st:?}); CPU rescue publishes the missing seqs; this run FAILS");
            rescue(
                &ring_sh,
                layout,
                seq0,
                s_runs,
                Duration::from_millis(cfg.timeout_ms),
            );
            println!(
                "round {r}: after rescue {:?}",
                ctx.wait(&cmds[b], npu_seqs[b], cfg.timeout_ms)
            );
            return Ok(false);
        }
        rt.sync_event(&round_done[b])?;
        let now = Instant::now();
        round_us.push((now - t_prev).as_secs_f64() * 1e6);
        t_prev = now;
    }
    rt.synchronize()?;
    for (r, &wall_us) in round_us.iter().enumerate() {
        let seq0 = 1 + (r * s_runs) as u32;
        let mut raw = vec![0u8; s_runs * coop_gpu::PUBLISH_RECORD_BYTES];
        rt.download_at(
            &stats,
            r * s_runs * coop_gpu::PUBLISH_RECORD_BYTES,
            &mut raw,
        )?;
        let recs: Vec<PublishRecord> = (0..s_runs)
            .map(|j| PublishRecord::parse(&raw[j * coop_gpu::PUBLISH_RECORD_BYTES..]))
            .collect();
        let mut raw_aux = vec![0u8; 3 * s_runs * coop_gpu::PUBLISH_RECORD_BYTES];
        rt.download_at(
            &aux,
            r * 3 * s_runs * coop_gpu::PUBLISH_RECORD_BYTES,
            &mut raw_aux,
        )?;
        for kind in 0..3 {
            for j in 0..s_runs {
                if kind == 0 && seq0 as usize + j <= r_slots {
                    continue;
                }
                let rec = PublishRecord::parse(
                    &raw_aux[(kind * s_runs + j) * coop_gpu::PUBLISH_RECORD_BYTES..],
                );
                if rec.status != coop_gpu::STATUS_OK
                    || rec.seq
                        != if kind == 0 {
                            seq0 + j as u32 - r_slots as u32
                        } else {
                            seq0 + j as u32
                        }
                {
                    println!(
                        "round {r} run {j}: {} kernel status {} seq {}",
                        ["slot-free wait", "publish", "release"][kind],
                        rec.status,
                        rec.seq
                    );
                    all_ok = false;
                }
            }
        }
        for (j, rec) in recs.iter().enumerate() {
            let seq = seq0 + j as u32;
            if rec.status != coop_gpu::STATUS_OK
                || rec.seq != seq
                || rec.last_done != seq
                || rec.t_obs < rec.t_pub
            {
                println!(
                    "round {r} run {j}: BAD record status={} seq={} last_done={} (want {seq})",
                    rec.status, rec.seq, rec.last_done
                );
                all_ok = false;
                continue;
            }
            rec_ok += 1;
            pub_done.push((rec.t_obs - rec.t_pub) as f64 * us_per_tick);
            obs_upper.push((rec.t_obs - rec.t_miss_issue) as f64 * us_per_tick);
            load_us.push((rec.t_obs - rec.t_hit_issue) as f64 * us_per_tick);
            store_us.push((rec.t_pub - rec.t_start) as f64 * us_per_tick);
            if j > 0 && recs[j - 1].status == coop_gpu::STATUS_OK {
                interval.push((rec.t_start - recs[j - 1].t_start) as f64 * us_per_tick);
            }
        }
        if gemm {
            let mut rb = vec![0u8; s_runs * c_bytes];
            rt.download_at(&readback, r * s_runs * c_bytes, &mut rb)?;
            for j in 0..s_runs {
                let set = set_of(r, j);
                let got = &rb[j * c_bytes..(j + 1) * c_bytes];
                c_total += 1;
                if got == eager[set].as_slice() {
                    c_ok += 1
                } else {
                    let i = got
                        .iter()
                        .zip(&eager[set])
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    println!("round {r} run {j}: GPU-read C != eager set {set} (first byte {i})");
                    all_ok = false;
                }
                if r == 0 || r + 1 == cfg.rounds {
                    cpu_total += 1;
                    let bad = d
                        .unpack_out(got)
                        .iter()
                        .zip(&sets[set].1)
                        .filter(|(a, b)| a != b)
                        .count();
                    if bad == 0 {
                        cpu_ok += 1
                    } else {
                        println!("round {r} run {j}: {bad} mismatches vs CPU");
                        all_ok = false;
                    }
                }
            }
        }
        println!(
            "round {r}: seq0={seq0} wall_us={wall_us:.1} per_run_us={:.2} records_ok={}/{s_runs}",
            wall_us / s_runs as f64,
            recs.iter()
                .filter(|x| x.status == coop_gpu::STATUS_OK)
                .count()
        );
    }
    // Final state: every done line carries the last round's last seq of its slot, every A slot the last copied set.
    let last = cfg.rounds - 1;
    let seq0 = 1 + (last * s_runs) as u32;
    let img = ring_sh.read(0, layout.bytes());
    for s in 0..r_slots {
        let want = (0..s_runs)
            .rev()
            .find(|&j| plans[j].slot == s)
            .map(|j| seq0 + j as u32)
            .unwrap();
        let at = layout.done_line(s);
        if u32_at(&img, at) == want
            && u32_at(&img, at + 4) == s as u32
            && u32_at(&img, at + 8) == DONE_MAGIC
        {
            done_ok_n += 1
        } else {
            println!(
                "slot {s}: final done line {:#x} {:#x} {:#x}, want seq {want:#x}",
                u32_at(&img, at),
                u32_at(&img, at + 4),
                u32_at(&img, at + 8)
            );
            all_ok = false;
        }
    }
    if gemm {
        let a_now = a_sh.read(0, a_sh.len);
        for s in 0..r_slots {
            let set = set_of(
                last,
                (0..s_runs).rev().find(|&j| plans[j].slot == s).unwrap(),
            );
            a_total += 1;
            if a_now[s * a_stride..s * a_stride + a_bytes] == *sets[set].0[0] {
                a_ok += 1
            } else {
                println!("slot {s}: final A arena != GPU-copied A of set {set}");
                all_ok = false;
            }
        }
    }

    let runs = cfg.rounds * s_runs;
    let per_run: Vec<f64> = round_us.iter().map(|u| u / s_runs as f64).collect();
    println!("latency: pub_to_done_gpu_us {}", pmm(&pub_done));
    println!("latency: done_to_gpu_obs_upper_us {}", pmm(&obs_upper));
    println!("latency: poll_load_us {}", pmm(&load_us));
    println!("latency: publish_store_us {}", pmm(&store_us));
    println!(
        "latency: gpu_run_interval_us {} (publish-to-publish of consecutive runs on the GPU clock)",
        pmm(&interval)
    );
    if gemm {
        let mean = per_run.iter().sum::<f64>() / per_run.len() as f64;
        println!("pipeline: shape={m}x{n}x{k} per_run_wall_us {} useful_tops_mean={:.2} (GPU A copy + publish + NPU GEMM + done + GPU C read, CPU-free)",
            pmm(&per_run), d.useful_ops() as f64 / mean / 1e6);
        println!("exact: gpu_read_c_bytes_equal_eager={c_ok}/{c_total} a_arena_equal_gpu_copy={a_ok}/{a_total} cpu_direct={cpu_ok}/{cpu_total}");
        all_ok &= c_ok == c_total && a_ok == a_total && cpu_ok == cpu_total;
    } else {
        println!("pipeline: empty ring per_run_wall_us {}", pmm(&per_run));
    }
    println!("coop: mode={} slots={s_runs} nslots={r_slots} rounds={} records_ok={rec_ok}/{runs} done_lines_ok={done_ok_n}/{r_slots} rescue=none",
        if gemm { "gemm" } else { "empty" }, cfg.rounds);
    all_ok &= rec_ok == runs && done_ok_n == r_slots;
    let _ = &pdi;
    Ok(all_ok)
}
