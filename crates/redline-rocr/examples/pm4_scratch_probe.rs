// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>

//! Hardware proof for GFX10/GFX11 PM4 dispatch with scratch (private segment).
//!
//! Every case runs the same hipcc code object twice on the same queue: once as
//! an ordinary AQL kernel-dispatch packet (ROCr/CP programs scratch from
//! `amd_queue_t`) and once as a PM4 indirect buffer whose scratch ring is
//! programmed by `Gfx10Pm4CommandBuffer::with_scratch`. The outputs must be
//! byte identical to each other and to a host reference.
//!
//! Cases: `dyn` (data-dependent private-array indexing), `chain` (scratch,
//! non-scratch, scratch kernels in ONE indirect buffer), `alias` (full
//! occupancy: every wave slot busy at once, each lane re-reading its private
//! tag so any two waves sharing a scratch slice corrupt each other), `mq4`
//! (the product `gemv_mq4g256v2_residual` code object over clean, wild and
//! NaN/Inf/denormal/-0 inputs).
//!
//! Build the code objects with `examples/pm4_scratch/build.sh <arch> <dir>`.
//!
//! ```text
//! pm4_scratch_probe --probe probe_gfx1100.co --mq4 mq4_residual_gfx1100.co
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use redline_rocr::packet::{KernelDispatchPacket, PacketImage};
use redline_rocr::{
    load_symbols, CompletionSignal, Executable, Gfx10KernelImage, Gfx10Pm4CommandBuffer,
    Gfx10Scratch, Gfx10ScratchFamily, Gfx10ScratchPlan, Gfx10ScratchTopology, GpuDevice,
    GpuSelector, KernargBuffer, KernargPool, Kernel, LaunchGeometry, QueueSet, Runtime,
};

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const RING_CAP_BYTES: u64 = 256 << 20;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

struct Rig {
    _runtime: Runtime,
    device: GpuDevice,
    host: KernargPool,
    local: KernargPool,
    queues: QueueSet,
    completion: CompletionSignal,
    family: Gfx10ScratchFamily,
    topology: Gfx10ScratchTopology,
}

struct Ring {
    scratch: Gfx10Scratch,
    _backing: KernargBuffer,
}

impl Rig {
    fn new() -> R<Self> {
        let runtime = Runtime::initialize(load_symbols().map_err(err)?).map_err(err)?;
        let device = runtime.select_gpu(GpuSelector::Ordinal(0)).map_err(err)?;
        let family = if device.name().starts_with("gfx11") {
            Gfx10ScratchFamily::Gfx11
        } else if device.name().starts_with("gfx10") {
            Gfx10ScratchFamily::Gfx10
        } else {
            return Err(format!("{} is neither gfx10 nor gfx11", device.name()));
        };
        let topology = device.scratch_topology().map_err(err)?;
        let host = KernargPool::discover(&device).map_err(err)?;
        let local = KernargPool::discover_device_local(&device).map_err(err)?;
        let completion = CompletionSignal::new(&device).map_err(err)?;
        let queues =
            QueueSet::create(&device, 1, *device.queue_size_range().start()).map_err(err)?;
        println!(
            "rig arch={} bus={} family={family:?} compute_units={} shader_engines={} max_waves_per_cu={}",
            device.name(),
            device.pci_bus_id(),
            topology.compute_units,
            topology.shader_engines,
            topology.max_waves_per_cu
        );
        Ok(Self {
            _runtime: runtime,
            device,
            host,
            local,
            queues,
            completion,
            family,
            topology,
        })
    }

    fn buffer(&self, bytes: &[u8]) -> R<KernargBuffer> {
        let mut buffer = self
            .host
            .allocate_fine_grained_bytes(bytes.len().max(16), 256)
            .map_err(err)?;
        buffer.as_mut_bytes()[..bytes.len()].copy_from_slice(bytes);
        Ok(buffer)
    }

    fn words(&self, words: &[u32]) -> R<KernargBuffer> {
        self.buffer(&words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>())
    }

    fn zeros(&self, bytes: usize) -> R<KernargBuffer> {
        self.buffer(&vec![0; bytes])
    }

    /// One ring for `kernels`: the plan covers the largest per-wave need.
    fn ring(&self, kernels: &[&Kernel]) -> R<Option<Ring>> {
        let mut required = 0;
        for kernel in kernels {
            required = required.max(
                Gfx10KernelImage::from_hsa(kernel)
                    .map_err(err)?
                    .scratch_bytes_per_wave(),
            );
        }
        if required == 0 {
            return Ok(None);
        }
        let plan = Gfx10ScratchPlan::new(self.family, self.topology, required, RING_CAP_BYTES)
            .map_err(err)?;
        let backing = self
            .local
            .allocate_scratch_backing(plan.backing_bytes() as usize)
            .map_err(err)?;
        let scratch = plan.bind(backing.address() as u64).map_err(err)?;
        println!(
            "ring family={:?} need_per_wave={required} bytes_per_wave={} slots={} backing={} base={:#x} tmpring={:#010x}",
            plan.family(),
            plan.bytes_per_wave(),
            plan.slots(),
            plan.backing_bytes(),
            scratch.base_address(),
            scratch.tmpring_size()
        );
        Ok(Some(Ring {
            scratch,
            _backing: backing,
        }))
    }

    fn submit(&mut self, packet: PacketImage) -> R<()> {
        self.completion.reset();
        self.queues.prepare_batches(&[vec![packet]]).map_err(err)?;
        self.queues.ring_prepared().map_err(err)?;
        self.queues
            .wait_signal(&self.completion, Duration::from_secs(60))
            .map_err(err)
    }

    /// Plain AQL kernel-dispatch packet: the anchor path (CP + ROCr scratch).
    fn run_aql(
        &mut self,
        kernel: &Kernel,
        geometry: LaunchGeometry,
        kernarg: &KernargBuffer,
    ) -> R<()> {
        let packet = KernelDispatchPacket::new(
            kernel.metadata(),
            geometry,
            0,
            kernarg.address(),
            self.completion.raw(),
        )
        .map_err(err)?;
        self.submit(PacketImage::kernel(&packet))
    }

    fn new_stream(ring: Option<&Ring>) -> Gfx10Pm4CommandBuffer {
        let mut commands = Gfx10Pm4CommandBuffer::new_stateful();
        if let Some(ring) = ring {
            commands = commands.with_scratch(ring.scratch);
        }
        commands.acquire_system();
        commands
    }

    fn run_pm4(&mut self, mut commands: Gfx10Pm4CommandBuffer) -> R<()> {
        commands.wait_compute_idle();
        commands.acquire_system();
        let bytes = commands.as_bytes();
        let mut indirect = self
            .host
            .allocate_executable_bytes(bytes.len())
            .map_err(err)?;
        indirect.write_exact(&bytes).map_err(err)?;
        let packet = PacketImage::pm4_indirect_buffer(
            indirect.address(),
            commands.len_dwords(),
            self.completion.raw(),
        )
        .map_err(err)?;
        self.submit(packet)
    }
}

/// Kernarg segment: explicit arguments, then the code-object-v5+ hidden
/// arguments HIP kernels read for `blockDim`/`gridDim` (LLVM `AMDGPUUsage`,
/// "Code Object V5 Metadata"): block counts (u32 x3) at +0, group sizes (u16
/// x3) at +12, remainders (u16 x3) at +18, grid dimensionality (u16) at +64,
/// after the explicit arguments rounded up to 8 bytes.
fn kernarg_for(
    rig: &Rig,
    kernel: &Kernel,
    geometry: LaunchGeometry,
    explicit: &[u8],
) -> R<KernargBuffer> {
    let mut buffer = rig.host.allocate_for(kernel.metadata()).map_err(err)?;
    let bytes = buffer.as_mut_bytes();
    bytes[..explicit.len()].copy_from_slice(explicit);
    let hidden = explicit.len().next_multiple_of(8);
    if bytes.len() >= hidden + 66 {
        for axis in 0..3 {
            let group = u32::from(geometry.workgroup[axis]);
            let grid = geometry.grid_workitems[axis];
            let count = grid.div_ceil(group);
            bytes[hidden + 4 * axis..hidden + 4 * axis + 4].copy_from_slice(&count.to_le_bytes());
            bytes[hidden + 12 + 2 * axis..hidden + 14 + 2 * axis]
                .copy_from_slice(&(group as u16).to_le_bytes());
            bytes[hidden + 18 + 2 * axis..hidden + 20 + 2 * axis]
                .copy_from_slice(&((grid % group) as u16).to_le_bytes());
        }
        bytes[hidden + 64..hidden + 66].copy_from_slice(&1_u16.to_le_bytes());
    }
    Ok(buffer)
}

fn ptr_bytes(parts: &[Part]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for part in parts {
        match part {
            Part::Ptr(buffer) => bytes.extend_from_slice(&(buffer.address() as u64).to_le_bytes()),
            Part::I32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }
    bytes
}

enum Part<'a> {
    Ptr(&'a KernargBuffer),
    I32(i32),
}

fn read_words(buffer: &mut KernargBuffer, words: usize) -> Vec<u32> {
    buffer.as_mut_bytes()[..words * 4]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Host reference of `scratch_dyn`.
fn scratch_dyn_ref(input: &[u32], threads: usize, rounds: i32) -> Vec<u32> {
    let n = input.len() as u32;
    (0..threads as u32)
        .map(|i| {
            let mut private = [0_u32; 48];
            for (j, slot) in private.iter_mut().enumerate() {
                *slot = input[((i.wrapping_mul(5).wrapping_add(j as u32)) % n) as usize]
                    .wrapping_add((j as u32).wrapping_mul(0x9e37_79b9));
            }
            let mut idx = input[(i % n) as usize] % 48;
            let mut acc = i;
            for _ in 0..rounds {
                acc = acc
                    .wrapping_mul(1_664_525)
                    .wrapping_add(private[idx as usize])
                    .wrapping_add(1_013_904_223);
                idx = (acc >> 9) % 48;
                private[idx as usize] ^= acc;
            }
            acc ^ private[((i + 1) % 48) as usize]
        })
        .collect()
}

/// Host reference of `plain_mix`.
fn plain_mix_ref(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .map(|v| v.wrapping_mul(3).wrapping_add(1))
        .collect()
}

/// Host reference of `scratch_alias` (`out` only; `fail` must stay zero).
fn scratch_alias_ref(threads: usize, spins: i32, seed: u32) -> Vec<u32> {
    (0..threads as u32)
        .map(|gid| {
            let tag = mix32(gid ^ seed);
            let mut sum = 0_u32;
            for s in 0..spins as u32 {
                let k = (s.wrapping_mul(7).wrapping_add(gid & 63)) & 63;
                sum = sum.wrapping_add(tag.wrapping_add(k));
            }
            sum ^ tag
        })
        .collect()
}

fn random_words(rng: &mut Rng, count: usize) -> Vec<u32> {
    const SPECIAL: [u32; 6] = [0, u32::MAX, 0x8000_0000, 1, 0x7fff_ffff, 0xdead_beef];
    (0..count)
        .map(|_| {
            let r = rng.next();
            if r % 16 == 0 {
                SPECIAL[(r >> 8) as usize % SPECIAL.len()]
            } else {
                rng.next()
            }
        })
        .collect()
}

struct Tally {
    cases: usize,
    failures: usize,
}

impl Tally {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        self.cases += 1;
        if !ok {
            self.failures += 1;
        }
        println!("case {name}: {} {detail}", if ok { "PASS" } else { "FAIL" });
    }
}

fn mismatches(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

fn case_dyn(rig: &mut Rig, tally: &mut Tally, probe: &Kernel) -> R<()> {
    let ring = rig.ring(&[probe])?;
    let threads = 65_536_usize;
    let geometry = LaunchGeometry::new([threads as u32, 1, 1], [64, 1, 1]).map_err(err)?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (trial, rounds) in [(0, 40), (1, 7), (2, 193)] {
        let input = random_words(&mut rng, 4096);
        let reference = scratch_dyn_ref(&input, threads, rounds);
        let input_buffer = rig.words(&input)?;
        let mut aql_out = rig.zeros(threads * 4)?;
        let aql_args = kernarg_for(
            rig,
            probe,
            geometry,
            &ptr_bytes(&[
                Part::Ptr(&input_buffer),
                Part::Ptr(&aql_out),
                Part::I32(input.len() as i32),
                Part::I32(rounds),
            ]),
        )?;
        rig.run_aql(probe, geometry, &aql_args)?;
        let aql = read_words(&mut aql_out, threads);

        let mut pm4_out = rig.zeros(threads * 4)?;
        let pm4_args = kernarg_for(
            rig,
            probe,
            geometry,
            &ptr_bytes(&[
                Part::Ptr(&input_buffer),
                Part::Ptr(&pm4_out),
                Part::I32(input.len() as i32),
                Part::I32(rounds),
            ]),
        )?;
        let mut commands = Rig::new_stream(ring.as_ref());
        commands
            .dispatch(probe, geometry, 0, pm4_args.address())
            .map_err(err)?;
        rig.run_pm4(commands)?;
        let pm4 = read_words(&mut pm4_out, threads);
        tally.check(
            &format!("dyn trial={trial} rounds={rounds}"),
            aql == pm4 && pm4 == reference,
            format!(
                "lanes={threads} pm4_vs_aql_mismatch={} pm4_vs_cpu_mismatch={} aql_vs_cpu_mismatch={}",
                mismatches(&pm4, &aql),
                mismatches(&pm4, &reference),
                mismatches(&aql, &reference)
            ),
        );
    }
    Ok(())
}

fn case_chain(rig: &mut Rig, tally: &mut Tally, dyn_kernel: &Kernel, plain: &Kernel) -> R<()> {
    let ring = rig.ring(&[dyn_kernel, plain])?;
    let threads = 32_768_usize;
    let geometry = LaunchGeometry::new([threads as u32, 1, 1], [64, 1, 1]).map_err(err)?;
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let input = random_words(&mut rng, 2048);
    let stage1 = scratch_dyn_ref(&input, threads, 24);
    let stage2 = plain_mix_ref(&stage1);
    let reference = scratch_dyn_ref(&stage2, threads, 24);

    let run = |rig: &mut Rig, pm4: bool| -> R<Vec<u32>> {
        let input_buffer = rig.words(&input)?;
        let o1 = rig.zeros(threads * 4)?;
        let o2 = rig.zeros(threads * 4)?;
        let mut o3 = rig.zeros(threads * 4)?;
        let a1 = kernarg_for(
            rig,
            dyn_kernel,
            geometry,
            &ptr_bytes(&[
                Part::Ptr(&input_buffer),
                Part::Ptr(&o1),
                Part::I32(input.len() as i32),
                Part::I32(24),
            ]),
        )?;
        let a2 = kernarg_for(
            rig,
            plain,
            geometry,
            &ptr_bytes(&[Part::Ptr(&o1), Part::Ptr(&o2)]),
        )?;
        let a3 = kernarg_for(
            rig,
            dyn_kernel,
            geometry,
            &ptr_bytes(&[
                Part::Ptr(&o2),
                Part::Ptr(&o3),
                Part::I32(threads as i32),
                Part::I32(24),
            ]),
        )?;
        if pm4 {
            let mut commands = Rig::new_stream(ring.as_ref());
            commands
                .dispatch(dyn_kernel, geometry, 0, a1.address())
                .map_err(err)?;
            commands.dependency_rmw_same_agent();
            commands
                .dispatch(plain, geometry, 0, a2.address())
                .map_err(err)?;
            commands.dependency_rmw_same_agent();
            commands
                .dispatch(dyn_kernel, geometry, 0, a3.address())
                .map_err(err)?;
            rig.run_pm4(commands)?;
        } else {
            rig.run_aql(dyn_kernel, geometry, &a1)?;
            rig.run_aql(plain, geometry, &a2)?;
            rig.run_aql(dyn_kernel, geometry, &a3)?;
        }
        Ok(read_words(&mut o3, threads))
    };
    let aql = run(rig, false)?;
    let pm4 = run(rig, true)?;
    tally.check(
        "chain scratch->plain->scratch in one IB",
        aql == pm4 && pm4 == reference,
        format!(
            "lanes={threads} pm4_vs_aql_mismatch={} pm4_vs_cpu_mismatch={}",
            mismatches(&pm4, &aql),
            mismatches(&pm4, &reference)
        ),
    );
    Ok(())
}

fn case_alias(rig: &mut Rig, tally: &mut Tally, alias: &Kernel) -> R<()> {
    let ring = rig.ring(&[alias])?;
    let lanes = if Gfx10KernelImage::from_hsa(alias).map_err(err)?.wave32 {
        32
    } else {
        64
    };
    let capacity_waves =
        (rig.topology.compute_units * rig.topology.max_waves_per_cu) as usize;
    // Twice the device's wave capacity so every slot is busy and more wait.
    let block = 128_usize;
    let threads = (capacity_waves * lanes * 2).next_multiple_of(block);
    let spins = 1500;
    let geometry = LaunchGeometry::new([threads as u32, 1, 1], [block as u16, 1, 1]).map_err(err)?;
    for seed in [0x5eed_0001_u32, 0xa11a_5000] {
        let reference = scratch_alias_ref(threads, spins, seed);
        let args = |rig: &Rig, out: &KernargBuffer, fail: &KernargBuffer| {
            kernarg_for(
                rig,
                alias,
                geometry,
                &ptr_bytes(&[
                    Part::Ptr(out),
                    Part::Ptr(fail),
                    Part::I32(spins),
                    Part::I32(seed as i32),
                ]),
            )
        };
        let mut aql_out = rig.zeros(threads * 4)?;
        let mut aql_fail = rig.zeros(16)?;
        let aql_args = args(rig, &aql_out, &aql_fail)?;
        rig.run_aql(alias, geometry, &aql_args)?;
        let aql = read_words(&mut aql_out, threads);
        let aql_bad = read_words(&mut aql_fail, 1)[0];

        let mut pm4_out = rig.zeros(threads * 4)?;
        let mut pm4_fail = rig.zeros(16)?;
        let pm4_args = args(rig, &pm4_out, &pm4_fail)?;
        let mut commands = Rig::new_stream(ring.as_ref());
        commands
            .dispatch(alias, geometry, 0, pm4_args.address())
            .map_err(err)?;
        rig.run_pm4(commands)?;
        let pm4 = read_words(&mut pm4_out, threads);
        let pm4_bad = read_words(&mut pm4_fail, 1)[0];
        tally.check(
            &format!("alias full-occupancy seed={seed:#x}"),
            aql == pm4 && pm4 == reference && aql_bad == 0 && pm4_bad == 0,
            format!(
                "lanes={threads} waves={} capacity_waves={capacity_waves} slots={} aql_bad_reads={aql_bad} pm4_bad_reads={pm4_bad} pm4_vs_aql_mismatch={} pm4_vs_cpu_mismatch={}",
                threads / lanes,
                ring.as_ref().map_or(0, |r| r.scratch.plan().slots()),
                mismatches(&pm4, &aql),
                mismatches(&pm4, &reference)
            ),
        );
    }
    Ok(())
}

/// Half-precision bit patterns the MQ4 headers are drawn from.
fn f16_bits(rng: &mut Rng, wild: bool) -> u16 {
    if wild {
        return rng.next() as u16;
    }
    // Sane scales/zero-points: sign, small exponent, random mantissa.
    let exponent = 10 + (rng.next() % 10) as u16;
    (((rng.next() & 1) as u16) << 15) | (exponent << 10) | (rng.next() as u16 & 0x3ff)
}

/// Negative control for `case_alias`: the same kernel on a ring whose
/// `WAVESIZE` is one unit (far below the kernel's per-wave need), dispatched
/// as if it were a zero-private kernel so the builder's provisioning check does
/// not refuse it. Waves then overrun into their neighbours' slices, so the
/// detector MUST report bad reads or wrong output: this proves the probe can
/// see aliasing and that the hardware really consumes `COMPUTE_TMPRING_SIZE`.
/// gfx11 only (gfx10's user SGPRs carry the V#, which the builder owns).
fn case_alias_control(rig: &mut Rig, tally: &mut Tally, alias: &Kernel) -> R<()> {
    if rig.family != Gfx10ScratchFamily::Gfx11 {
        println!("case alias-control: SKIP (gfx11 only)");
        return Ok(());
    }
    let image = Gfx10KernelImage::from_hsa(alias).map_err(err)?;
    let lanes = if image.wave32 { 32 } else { 64 };
    let small = Gfx10ScratchPlan::new(rig.family, rig.topology, 64, RING_CAP_BYTES).map_err(err)?;
    // Slack: the last slots overrun the ring by up to a full real wave slice.
    let slack = image.scratch_bytes_per_wave() as usize + (1 << 20);
    let backing = rig
        .local
        .allocate_scratch_backing(small.backing_bytes() as usize + slack)
        .map_err(err)?;
    let scratch = small.bind(backing.address() as u64).map_err(err)?;
    let capacity_waves = (rig.topology.compute_units * rig.topology.max_waves_per_cu) as usize;
    let block = 128_usize;
    let threads = (capacity_waves * lanes * 2).next_multiple_of(block);
    let spins = 1500;
    let seed = 0x5eed_0001_u32;
    let geometry = LaunchGeometry::new([threads as u32, 1, 1], [block as u16, 1, 1]).map_err(err)?;
    let reference = scratch_alias_ref(threads, spins, seed);
    let mut out = rig.zeros(threads * 4)?;
    let mut fail = rig.zeros(16)?;
    let args = kernarg_for(
        rig,
        alias,
        geometry,
        &ptr_bytes(&[
            Part::Ptr(&out),
            Part::Ptr(&fail),
            Part::I32(spins),
            Part::I32(seed as i32),
        ]),
    )?;
    let mut commands = Gfx10Pm4CommandBuffer::new_stateful().with_scratch(scratch);
    commands.acquire_system();
    let mut fudged = image;
    fudged.private_segment_size = 0;
    let kernarg = args.address() as usize as u64;
    commands
        .dispatch_image(
            &fudged,
            geometry,
            0,
            &[kernarg as u32, (kernarg >> 32) as u32],
        )
        .map_err(err)?;
    rig.run_pm4(commands)?;
    let got = read_words(&mut out, threads);
    let bad = read_words(&mut fail, 1)[0];
    let wrong = mismatches(&got, &reference);
    tally.check(
        "alias-control undersized ring (corruption MUST be detected)",
        bad != 0 || wrong != 0,
        format!(
            "bytes_per_wave={} real_need={} bad_reads={bad} wrong_outputs={wrong}/{threads}",
            small.bytes_per_wave(),
            image.scratch_bytes_per_wave()
        ),
    );
    Ok(())
}

fn case_mq4(rig: &mut Rig, tally: &mut Tally, residual: &Kernel) -> R<()> {
    let image = Gfx10KernelImage::from_hsa(residual).map_err(err)?;
    println!(
        "mq4 kernel private_segment_size={} wave32={} kernel_code_properties={:#06x} rsrc2={:#010x}",
        image.private_segment_size,
        image.wave32,
        image.kernel_code_properties,
        image.compute_pgm_rsrc2
    );
    let ring = rig.ring(&[residual])?;
    let (m, k) = (2048_usize, 4096_usize);
    let row_bytes = (k / 256) * 136;
    let geometry = LaunchGeometry::new([(m as u32 / 2) * 32, 1, 1], [32, 1, 1]).map_err(err)?;
    let mut rng = Rng(0xc0ff_ee00_dead_beef);
    for (name, wild, special_x) in [
        ("clean", false, false),
        ("wild-headers", true, false),
        ("edge-x-y", false, true),
    ] {
        let mut a = vec![0_u8; m * row_bytes];
        for row in 0..m {
            for group in 0..k / 256 {
                let base = row * row_bytes + group * 136;
                let scale = f16_bits(&mut rng, wild);
                let zero = f16_bits(&mut rng, wild);
                a[base..base + 2].copy_from_slice(&scale.to_le_bytes());
                a[base + 2..base + 4].copy_from_slice(&zero.to_le_bytes());
                for byte in &mut a[base + 4..base + 136] {
                    *byte = rng.next() as u8;
                }
            }
        }
        const SPECIAL_F32: [u32; 10] = [
            0x0000_0000, // +0
            0x8000_0000, // -0
            0x0000_0001, // smallest denormal
            0x807f_ffff, // largest negative denormal
            0x7f80_0000, // +inf
            0xff80_0000, // -inf
            0x7fc0_0000, // quiet NaN
            0xffc0_0001, // negative NaN payload
            0x7f7f_ffff, // max finite
            0x3f80_0000, // 1.0
        ];
        let x: Vec<u32> = (0..k)
            .map(|_| {
                let r = rng.next();
                if special_x && r % 7 == 0 {
                    SPECIAL_F32[(r >> 8) as usize % SPECIAL_F32.len()]
                } else {
                    // Normal-range values in roughly [-2, 2).
                    let mantissa = rng.next() & 0x7f_ffff;
                    ((rng.next() & 1) << 31) | ((126 + (r >> 24) % 3) << 23) | mantissa
                }
            })
            .collect();
        let y0: Vec<u32> = (0..m)
            .map(|_| {
                let r = rng.next();
                if special_x && r % 5 == 0 {
                    SPECIAL_F32[(r >> 8) as usize % SPECIAL_F32.len()]
                } else {
                    (r & 0x807f_ffff) | (127 << 23)
                }
            })
            .collect();
        let a_buffer = rig.buffer(&a)?;
        let x_buffer = rig.words(&x)?;
        let run = |rig: &mut Rig, pm4: bool| -> R<Vec<u32>> {
            let mut y = rig.words(&y0)?;
            let args = kernarg_for(
                rig,
                residual,
                geometry,
                &ptr_bytes(&[
                    Part::Ptr(&a_buffer),
                    Part::Ptr(&x_buffer),
                    Part::Ptr(&y),
                    Part::I32(m as i32),
                    Part::I32(k as i32),
                ]),
            )?;
            if pm4 {
                let mut commands = Rig::new_stream(ring.as_ref());
                commands
                    .dispatch(residual, geometry, 0, args.address())
                    .map_err(err)?;
                rig.run_pm4(commands)?;
            } else {
                rig.run_aql(residual, geometry, &args)?;
            }
            Ok(read_words(&mut y, m))
        };
        let aql = run(rig, false)?;
        let pm4 = run(rig, true)?;
        let pm4_again = run(rig, true)?;
        let nan_lanes = pm4.iter().filter(|w| (**w & 0x7fff_ffff) > 0x7f80_0000).count();
        tally.check(
            &format!("mq4 residual {name} M={m} K={k}"),
            aql == pm4 && pm4 == pm4_again,
            format!(
                "private={} rows={m} pm4_vs_aql_mismatch={} pm4_rerun_mismatch={} nan_rows={nan_lanes}",
                image.private_segment_size,
                mismatches(&pm4, &aql),
                mismatches(&pm4, &pm4_again)
            ),
        );
    }
    Ok(())
}

fn main() -> R<()> {
    let mut probe = PathBuf::new();
    let mut mq4 = PathBuf::new();
    let mut only = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--probe" => probe = value.into(),
            "--mq4" => mq4 = value.into(),
            "--only" => only = value,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let mut rig = Rig::new()?;
    let mut tally = Tally {
        cases: 0,
        failures: 0,
    };
    let wanted = |name: &str| only.is_empty() || only == name;
    if !probe.as_os_str().is_empty() {
        let exe = Executable::load(
            &rig.device,
            Arc::from(std::fs::read(&probe).map_err(err)?.into_boxed_slice()),
        )
        .map_err(err)?;
        let dyn_kernel = exe.kernel("scratch_dyn.kd").map_err(err)?;
        let plain = exe.kernel("plain_mix.kd").map_err(err)?;
        let alias = exe.kernel("scratch_alias.kd").map_err(err)?;
        for kernel in [&dyn_kernel, &plain, &alias] {
            let image = Gfx10KernelImage::from_hsa(kernel).map_err(err)?;
            println!(
                "kernel {} private_segment_size={} wave32={} kernel_code_properties={:#06x} rsrc2={:#010x}",
                kernel.name(),
                image.private_segment_size,
                image.wave32,
                image.kernel_code_properties,
                image.compute_pgm_rsrc2
            );
        }
        if wanted("dyn") {
            case_dyn(&mut rig, &mut tally, &dyn_kernel)?;
        }
        if wanted("chain") {
            case_chain(&mut rig, &mut tally, &dyn_kernel, &plain)?;
        }
        if wanted("alias") {
            case_alias(&mut rig, &mut tally, &alias)?;
        }
        if wanted("alias-control") {
            case_alias_control(&mut rig, &mut tally, &alias)?;
        }
    }
    if !mq4.as_os_str().is_empty() && wanted("mq4") {
        let exe = Executable::load(
            &rig.device,
            Arc::from(std::fs::read(&mq4).map_err(err)?.into_boxed_slice()),
        )
        .map_err(err)?;
        let residual = exe.kernel("gemv_mq4g256v2_residual.kd").map_err(err)?;
        case_mq4(&mut rig, &mut tally, &residual)?;
    }
    println!(
        "RESULT arch={} cases={} failures={}",
        rig.device.name(),
        tally.cases,
        tally.failures
    );
    if tally.failures != 0 {
        return Err(format!("{} case(s) failed", tally.failures));
    }
    Ok(())
}
