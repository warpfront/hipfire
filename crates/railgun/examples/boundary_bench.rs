//! gfx11 PM4 boundary-cost microbenchmark (the gfx1100 decode-floor rank-5
//! kill test): one retained PM4 IB of N dependent dispatches, one boundary
//! rung between every pair, timed against a HIP graph and back-to-back HIP
//! launches of the same N kernels from the same code object.
//!
//! One arm per process; the driver interleaves processes (ABBA+BAAB).
//!
//! ```sh
//! ROCR_VISIBLE_DEVICES=GPU-43390a851e296ee5 \
//!   boundary_bench --arch gfx1100 --bus 0000:66:00.0 --uuid GPU-43390a851e296ee5 \
//!   --empty-co empty.co --rmsnorm-co rmsnorm.hsaco --out DIR --label L \
//!   --arm pm4:redline --work rmsnorm --n 1000 --secs 10
//! ```
//!
//! Arms:
//! * `pm4:<rung>` — Redline's gfx11 tape framing (stateful SH registers,
//!   entry `acquire_system`, trailing wait + `acquire_system`), with `<rung>`
//!   between consecutive dispatches:
//!   - `redline`: what `replay.rs` emits on gfx1100 for a dependent boundary
//!     (`wait_compute_idle` + `acquire_inter_node(_, vmem_only=false)` =
//!     CS_PARTIAL_FLUSH + ACQUIRE_MEM GCR 0x380)
//!   - `wait_l1`: CS_PARTIAL_FLUSH + GCR 0x380 (the G4 `raw_smem` rung)
//!   - `wait_vmem`: CS_PARTIAL_FLUSH + GCR 0x300 (the G4 `raw_vmem` rung)
//!   - `wait`: CS_PARTIAL_FLUSH only (the G4 `war`/`waw` rung)
//!   - `release_wait`: RELEASE_MEM + WAIT_REG_MEM on a fence word, no cache op
//!     (gfx1010's `ReleaseWait` dependency; not a G4-probed rung on gfx11)
//!   - `wait_global`: CS_PARTIAL_FLUSH + GCR 0xc380
//!   - `wait_system`: CS_PARTIAL_FLUSH + ROCr system ACQUIRE_MEM
//!   - `none`: no packet (dispatches may overlap; lower bound only)
//! * `hip:graph` — the N launches stream-captured into one hipGraph, replayed
//!   (the decode-floor T0 method).
//! * `hip:stream` — N back-to-back `hipModuleLaunchKernel` on one stream.
//! * `soak` — every PM4 rung on both works, cycled for `--secs`.
//!
//! Works:
//! * `empty` — `bb_empty`, 1 workgroup × 64 lanes (the T0 `empty_1x64` shape);
//!   `empty:<wgs>x<lanes>` for the decode-floor large-grid shapes.
//! * `rmsnorm` — hipfire's `rmsnorm_f32` (H2's final-norm shape: n = 5120,
//!   grid 1, block 256, 1 KiB dynamic LDS), chained RAW by ping-pong: kernel i
//!   reads `x[i % 2]` and writes `x[(i + 1) % 2]`. Every process also runs
//!   the chain `--parity` times from a fixed input and hashes the result.
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hip_bridge::{DeviceBuffer, HipRuntime};
use redline_rocr::packet::PacketImage;
use redline_rocr::{
    CompletionSignal, Executable, Gfx10Pm4CommandBuffer, GpuDevice, GpuSelector, KernargBuffer, KernargPool, Kernel,
    LaunchGeometry, QueueSet, Runtime, load_symbols,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type R<T> = Result<T, String>;

fn err<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

const RMS_N: usize = 5120;
const RMS_BLOCK: u32 = 256;
const RMS_LDS: u32 = RMS_BLOCK * 4;
const RMS_EXPLICIT: usize = 32; // x, weight, out, n, eps
const EMPTY: Work = Work::Empty { wgs: 1, lanes: 64 };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rung {
    None,
    Wait,
    WaitVmem,
    WaitL1,
    Redline,
    ReleaseWait,
    WaitGlobal,
    WaitSystem,
}

const ALL_RUNGS: [Rung; 8] = [
    Rung::Redline,
    Rung::WaitL1,
    Rung::WaitVmem,
    Rung::Wait,
    Rung::ReleaseWait,
    Rung::WaitGlobal,
    Rung::WaitSystem,
    Rung::None,
];

impl Rung {
    fn parse(s: &str) -> R<Self> {
        Ok(match s {
            "none" => Self::None,
            "wait" => Self::Wait,
            "wait_vmem" => Self::WaitVmem,
            "wait_l1" => Self::WaitL1,
            "redline" => Self::Redline,
            "release_wait" => Self::ReleaseWait,
            "wait_global" => Self::WaitGlobal,
            "wait_system" => Self::WaitSystem,
            _ => return Err(format!("unknown rung {s}")),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Wait => "wait",
            Self::WaitVmem => "wait_vmem",
            Self::WaitL1 => "wait_l1",
            Self::Redline => "redline",
            Self::ReleaseWait => "release_wait",
            Self::WaitGlobal => "wait_global",
            Self::WaitSystem => "wait_system",
        }
    }

    fn emits(self) -> &'static str {
        match self {
            Self::None => "no packet",
            Self::Wait => "EVENT_WRITE CS_PARTIAL_FLUSH",
            Self::WaitVmem => "CS_PARTIAL_FLUSH + ACQUIRE_MEM GCR 0x300",
            Self::WaitL1 | Self::Redline => "CS_PARTIAL_FLUSH + ACQUIRE_MEM GCR 0x380",
            Self::ReleaseWait => "RELEASE_MEM(bottom-of-pipe, value) + WAIT_REG_MEM(==value)",
            Self::WaitGlobal => "CS_PARTIAL_FLUSH + ACQUIRE_MEM GCR 0xc380",
            Self::WaitSystem => "CS_PARTIAL_FLUSH + ROCr system ACQUIRE_MEM",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Work {
    /// `bb_empty` over `wgs` workgroups of `lanes` lanes.
    Empty { wgs: u32, lanes: u32 },
    Rmsnorm,
}

impl Work {
    /// `empty` (1x64), `empty:<wgs>x<lanes>`, or `rmsnorm`.
    fn parse(s: &str) -> R<Self> {
        if s == "empty" {
            return Ok(EMPTY);
        }
        if s == "rmsnorm" {
            return Ok(Self::Rmsnorm);
        }
        let shape = s.strip_prefix("empty:").ok_or_else(|| format!("unknown work {s}"))?;
        let (w, l) = shape.split_once('x').ok_or_else(|| format!("bad shape {shape}"))?;
        Ok(Self::Empty { wgs: w.parse().map_err(err)?, lanes: l.parse().map_err(err)? })
    }

    fn name(self) -> String {
        match self {
            EMPTY => "empty".into(),
            Self::Empty { wgs, lanes } => format!("empty:{wgs}x{lanes}"),
            Self::Rmsnorm => "rmsnorm".into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Pm4(Rung),
    HipGraph,
    HipStream,
    Soak,
}

impl Arm {
    fn parse(s: &str) -> R<Self> {
        Ok(match s {
            "hip:graph" => Self::HipGraph,
            "hip:stream" => Self::HipStream,
            "soak" => Self::Soak,
            _ => match s.strip_prefix("pm4:") {
                Some(r) => Self::Pm4(Rung::parse(r)?),
                None => return Err(format!("unknown arm {s}")),
            },
        })
    }

    fn name(self) -> String {
        match self {
            Self::Pm4(r) => format!("pm4:{}", r.name()),
            Self::HipGraph => "hip:graph".into(),
            Self::HipStream => "hip:stream".into(),
            Self::Soak => "soak".into(),
        }
    }
}

struct Args {
    arch: String,
    bus: String,
    uuid: String,
    empty_co: PathBuf,
    rmsnorm_co: PathBuf,
    out: PathBuf,
    label: String,
    arm: Arm,
    work: Work,
    n: usize,
    secs: f64,
    parity: usize,
    legacy_regs: bool,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        arch: String::new(),
        bus: String::new(),
        uuid: String::new(),
        empty_co: PathBuf::new(),
        rmsnorm_co: PathBuf::new(),
        out: PathBuf::new(),
        label: String::new(),
        arm: Arm::HipGraph,
        work: EMPTY,
        n: 1000,
        secs: 10.0,
        parity: 4,
        legacy_regs: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let k = argv[i].as_str();
        if k == "--legacy-regs" {
            a.legacy_regs = true;
            i += 1;
            continue;
        }
        let v = argv.get(i + 1).cloned().ok_or_else(|| format!("{k} needs a value"))?;
        match k {
            "--arch" => a.arch = v,
            "--bus" => a.bus = v,
            "--uuid" => a.uuid = v,
            "--empty-co" => a.empty_co = v.into(),
            "--rmsnorm-co" => a.rmsnorm_co = v.into(),
            "--out" => a.out = v.into(),
            "--label" => a.label = v,
            "--arm" => a.arm = Arm::parse(&v)?,
            "--work" => a.work = Work::parse(&v)?,
            "--n" => a.n = v.parse().map_err(err)?,
            "--secs" => a.secs = v.parse().map_err(err)?,
            "--parity" => a.parity = v.parse().map_err(err)?,
            _ => return Err(format!("unknown argument {k}")),
        }
        i += 2;
    }
    if a.arch.is_empty() || a.bus.is_empty() || a.uuid.is_empty() || a.out.as_os_str().is_empty() {
        return Err("--arch, --bus, --uuid and --out are required".into());
    }
    if a.label.is_empty() {
        a.label = format!("{}-{}", a.arm.name(), a.work.name()).replace(':', "_");
    }
    if a.n < 2 {
        return Err("--n must be at least 2".into());
    }
    Ok(a)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// KFD topology `unique_id` of the node at `bus`, as `GPU-%016x`.
fn kfd_uuid(bus: &str) -> R<Option<String>> {
    let (dom, rest) = bus.split_once(':').ok_or("bus")?;
    let parts: Vec<&str> = rest.split([':', '.']).collect();
    let domain = u32::from_str_radix(dom, 16).map_err(err)?;
    let b = u32::from_str_radix(parts[0], 16).map_err(err)?;
    let d = u32::from_str_radix(parts[1], 16).map_err(err)?;
    let f = u32::from_str_radix(parts[2], 16).map_err(err)?;
    let location = (b << 8) | (d << 3) | f;
    for entry in std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes").map_err(err)? {
        let props = std::fs::read_to_string(entry.map_err(err)?.path().join("properties")).unwrap_or_default();
        let field = |k: &str| {
            props
                .lines()
                .find_map(|l| l.strip_prefix(k).and_then(|v| v.trim().parse::<u64>().ok()))
        };
        if field("location_id ") == Some(location as u64) && field("domain ") == Some(domain as u64) {
            return Ok(field("unique_id ").filter(|u| *u != 0).map(|u| format!("GPU-{u:016x}")));
        }
    }
    Ok(None)
}

#[derive(Default)]
struct Samples {
    gpu_us: Vec<f64>,
    wall_us: Vec<f64>,
}

fn summary(v: &[f64], n: usize) -> Value {
    if v.is_empty() {
        return Value::Null;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    let per = |x: f64| x / n as f64;
    json!({
        "iterations": s.len(),
        "median_us_per_kernel": per(q(0.5)),
        "min_us_per_kernel": per(s[0]),
        "p10_us_per_kernel": per(q(0.1)),
        "p90_us_per_kernel": per(q(0.9)),
        "mean_us_per_kernel": per(s.iter().sum::<f64>() / s.len() as f64),
        "median_us_per_launch": q(0.5),
    })
}

/// HIP side: identity, the rmsnorm buffers (hipMalloc, as in hipfire), and
/// the HIP arms.
struct Hip {
    hip: HipRuntime,
    x: [DeviceBuffer; 2],
    weight: DeviceBuffer,
    x0_host: Vec<u8>,
}

fn hip_setup(args: &Args) -> R<(Hip, Value)> {
    let hip = HipRuntime::load().map_err(err)?;
    let count = hip.device_count().map_err(err)?;
    if count != 1 {
        return Err(format!("expected exactly one visible HIP device, found {count}"));
    }
    hip.set_device(0).map_err(err)?;
    let arch = hip.get_arch(0).map_err(err)?;
    let bus = hip.device_pci_bus_id(0).map_err(err)?.to_ascii_lowercase();
    let uuid = hip.device_uuid(0).map_err(err)?;
    if arch != args.arch || bus != args.bus.to_ascii_lowercase() || uuid != args.uuid {
        return Err(format!(
            "HIP identity mismatch: {arch} {bus} {uuid}, expected {} {} {}",
            args.arch, args.bus, args.uuid
        ));
    }
    let bytes = RMS_N * 4;
    let x = [hip.malloc(bytes).map_err(err)?, hip.malloc(bytes).map_err(err)?];
    let weight = hip.malloc(bytes).map_err(err)?;
    // Deterministic input; weights within 1e-3 of 1 so the 1000-step chain
    // keeps changing without under/overflow.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let x0: Vec<u8> = (0..RMS_N).flat_map(|_| ((next() * 2.0 - 1.0) as f32).to_le_bytes()).collect();
    let w: Vec<u8> = (0..RMS_N).flat_map(|_| ((1.0 + (next() * 2.0 - 1.0) * 1e-3) as f32).to_le_bytes()).collect();
    hip.memcpy_htod(&weight, &w).map_err(err)?;
    let identity = json!({"hip_arch": arch, "hip_bus": bus, "hip_uuid": uuid});
    Ok((Hip { hip, x, weight, x0_host: x0 }, identity))
}

impl Hip {
    fn reset_input(&self) -> R<()> {
        self.hip.memcpy_htod(&self.x[0], &self.x0_host).map_err(err)?;
        self.hip.memcpy_htod(&self.x[1], &vec![0u8; RMS_N * 4]).map_err(err)?;
        self.hip.device_synchronize().map_err(err)
    }

    fn result_hash(&self, n: usize) -> R<String> {
        self.hip.device_synchronize().map_err(err)?;
        let mut out = vec![0u8; RMS_N * 4];
        self.hip.memcpy_dtoh(&mut out, &self.x[n % 2]).map_err(err)?;
        Ok(sha256_hex(&out)[..16].to_owned())
    }

    /// Explicit rmsnorm_f32 kernarg bytes for chain step `i`.
    fn rms_explicit(&self, i: usize) -> [u8; RMS_EXPLICIT] {
        let mut b = [0u8; RMS_EXPLICIT];
        b[0..8].copy_from_slice(&(self.x[i % 2].as_ptr() as u64).to_le_bytes());
        b[8..16].copy_from_slice(&(self.weight.as_ptr() as u64).to_le_bytes());
        b[16..24].copy_from_slice(&(self.x[(i + 1) % 2].as_ptr() as u64).to_le_bytes());
        b[24..28].copy_from_slice(&(RMS_N as i32).to_le_bytes());
        b[28..32].copy_from_slice(&1e-6f32.to_le_bytes());
        b
    }
}

fn run_hip(args: &Args, hip: &Hip) -> R<Value> {
    let h = &hip.hip;
    let (co, symbol) = match args.work {
        Work::Empty { .. } => (&args.empty_co, "bb_empty"),
        Work::Rmsnorm => (&args.rmsnorm_co, "rmsnorm_f32"),
    };
    let image = std::fs::read(co).map_err(|e| format!("{}: {e}", co.display()))?;
    let module = h.module_load_data(&image).map_err(err)?;
    let func = h.module_get_function(&module, symbol).map_err(err)?;
    let stream = h.stream_create().map_err(err)?;
    let n = args.n;
    let mut blobs: Vec<[u8; RMS_EXPLICIT]> = (0..n).map(|i| hip.rms_explicit(i)).collect();
    let launch = |i: usize, blobs: &mut Vec<[u8; RMS_EXPLICIT]>| -> R<()> {
        unsafe {
            match args.work {
                Work::Empty { wgs, lanes } => {
                    h.launch_kernel(&func, [wgs, 1, 1], [lanes, 1, 1], 0, Some(&stream), &mut [])
                }
                Work::Rmsnorm => {
                    h.launch_kernel_blob(&func, [1, 1, 1], [RMS_BLOCK, 1, 1], RMS_LDS, Some(&stream), &mut blobs[i])
                }
            }
        }
        .map_err(err)
    };
    let graph = if args.arm == Arm::HipGraph {
        h.stream_begin_capture(&stream, 0).map_err(err)?;
        for i in 0..n {
            launch(i, &mut blobs)?;
        }
        let g = h.stream_end_capture(&stream).map_err(err)?;
        Some(h.graph_instantiate(&g).map_err(err)?)
    } else {
        None
    };
    let run_once = |blobs: &mut Vec<[u8; RMS_EXPLICIT]>| -> R<()> {
        match &graph {
            Some(g) => h.graph_launch(g, &stream).map_err(err),
            None => (0..n).try_for_each(|i| launch(i, blobs)),
        }
    };
    let mut parity = Vec::new();
    if args.work == Work::Rmsnorm {
        for _ in 0..args.parity {
            hip.reset_input()?;
            run_once(&mut blobs)?;
            h.stream_synchronize(&stream).map_err(err)?;
            parity.push(hip.result_hash(n)?);
        }
    }
    for _ in 0..3 {
        run_once(&mut blobs)?;
    }
    h.stream_synchronize(&stream).map_err(err)?;
    let e0 = h.event_create().map_err(err)?;
    let e1 = h.event_create().map_err(err)?;
    let mut s = Samples::default();
    let started = Instant::now();
    while started.elapsed().as_secs_f64() < args.secs {
        let t = Instant::now();
        h.event_record(&e0, Some(&stream)).map_err(err)?;
        run_once(&mut blobs)?;
        h.event_record(&e1, Some(&stream)).map_err(err)?;
        h.event_synchronize(&e1).map_err(err)?;
        s.wall_us.push(t.elapsed().as_secs_f64() * 1e6);
        s.gpu_us.push(h.event_elapsed_ms(&e0, &e1).map_err(err)? as f64 * 1e3);
    }
    Ok(json!({
        "timer": "hipEventElapsedTime around one launch of the N-kernel sequence",
        "gpu": summary(&s.gpu_us, n), "wall": summary(&s.wall_us, n),
        "seconds": started.elapsed().as_secs_f64(), "parity_hashes": parity,
        "code_object_sha256": sha256_hex(&image),
    }))
}

/// ROCr/PM4 side, Redline's transport: vendor PM4-IB AQL packet on a
/// Redline-created queue, IB in executable host memory, host kernargs.
struct Pm4 {
    queues: QueueSet,
    completion: CompletionSignal,
    host_pool: KernargPool,
    empty: Kernel,
    rmsnorm: Kernel,
    timestamps: KernargBuffer,
    fence: KernargBuffer,
    freq_hz: u64,
    _exes: Vec<Executable>,
}

struct Ib {
    buffer: KernargBuffer,
    dwords: u32,
    _kernargs: KernargBuffer,
    census: Value,
}

fn put_u16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// The HIP module-launch implicit suffix, as Redline's
/// `populate_gfx12_kernarg` synthesizes it (block counts, group sizes, zero
/// remainders, grid dims, dynamic LDS).
fn fill_implicit(b: &mut [u8], base: usize, wgs: u32, block: u16, lds: u32) {
    if b.len() < base + 256 {
        return;
    }
    for axis in 0..3 {
        put_u32(b, base + axis * 4, if axis == 0 { wgs } else { 1 });
        put_u16(b, base + 12 + axis * 2, if axis == 0 { block } else { 1 });
        put_u16(b, base + 18 + axis * 2, 0);
    }
    put_u16(b, base + 64, 1);
    put_u32(b, base + 120, lds);
}

impl Pm4 {
    fn setup(args: &Args) -> R<(Self, Value)> {
        let runtime = Runtime::initialize(load_symbols().map_err(err)?).map_err(err)?;
        let devices = runtime.gpu_devices().map_err(err)?;
        if devices.len() != 1 {
            return Err(format!("expected exactly one visible ROCr GPU, found {}", devices.len()));
        }
        let device: GpuDevice = runtime.select_gpu(GpuSelector::Ordinal(0)).map_err(err)?;
        let name = device.name().to_owned();
        let bus = device.pci_bus_id().to_string();
        let kfd = kfd_uuid(&bus)?;
        if name != args.arch || bus != args.bus || kfd.as_deref() != Some(args.uuid.as_str()) {
            return Err(format!(
                "ROCr identity mismatch: {name} {bus} {kfd:?}, expected {} {} {}",
                args.arch, args.bus, args.uuid
            ));
        }
        let cus = device.compute_unit_count().map_err(err)?;
        let load = |path: &PathBuf, symbol: &str| -> R<(Executable, Kernel)> {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let exe = Executable::load(&device, Arc::from(bytes.into_boxed_slice())).map_err(err)?;
            let k = exe.kernel(&format!("{symbol}.kd")).map_err(err)?;
            Ok((exe, k))
        };
        let (e0, empty) = load(&args.empty_co, "bb_empty")?;
        let (e1, rmsnorm) = load(&args.rmsnorm_co, "rmsnorm_f32")?;
        let rms_ka = rmsnorm.metadata().kernarg_segment_size as usize;
        if rms_ka != RMS_EXPLICIT + 256 {
            return Err(format!("rmsnorm_f32 kernarg segment {rms_ka} != {}", RMS_EXPLICIT + 256));
        }
        let host_pool = KernargPool::discover(&device).map_err(err)?;
        let timestamps = host_pool.allocate_fine_grained_bytes(64, 256).map_err(err)?;
        let fence = host_pool.allocate_fine_grained_bytes(64, 256).map_err(err)?;
        let completion = CompletionSignal::new(&device).map_err(err)?;
        let queue_size = *device.queue_size_range().start();
        let queues = QueueSet::create(&device, 1, queue_size).map_err(err)?;
        let freq_hz = device.gpu_timestamp_frequency_hz().map_err(err)?;
        let identity = json!({
            "rocr_arch": name, "rocr_bus": bus, "kfd_unique_id": kfd, "compute_units": cus,
            "gpu_timestamp_hz": freq_hz,
            "empty_kernarg_segment": empty.metadata().kernarg_segment_size,
            "rmsnorm_kernarg_segment": rms_ka,
        });
        Ok((
            Self { queues, completion, host_pool, empty, rmsnorm, timestamps, fence, freq_hz, _exes: vec![e0, e1] },
            identity,
        ))
    }

    fn build(&self, args: &Args, hip: &Hip, work: Work, rung: Rung) -> R<Ib> {
        let n = args.n;
        let kernel = match work {
            Work::Empty { .. } => &self.empty,
            Work::Rmsnorm => &self.rmsnorm,
        };
        let seg = (kernel.metadata().kernarg_segment_size as usize).max(64);
        let stride = seg.div_ceil(64) * 64;
        let mut kernargs = self.host_pool.allocate_fine_grained_bytes(n * stride, 256).map_err(err)?;
        let base = kernargs.address() as usize;
        {
            let bytes = kernargs.as_mut_bytes();
            bytes.fill(0);
            for i in 0..n {
                let b = &mut bytes[i * stride..i * stride + seg];
                match work {
                    Work::Empty { wgs, lanes } => fill_implicit(b, 0, wgs, lanes as u16, 0),
                    Work::Rmsnorm => {
                        b[..RMS_EXPLICIT].copy_from_slice(&hip.rms_explicit(i));
                        fill_implicit(b, RMS_EXPLICIT, 1, RMS_BLOCK as u16, RMS_LDS);
                    }
                }
            }
        }
        let (geometry, lds) = match work {
            Work::Empty { wgs, lanes } => (LaunchGeometry::new([wgs * lanes, 1, 1], [lanes as u16, 1, 1]), 0),
            Work::Rmsnorm => (LaunchGeometry::new([RMS_BLOCK, 1, 1], [RMS_BLOCK as u16, 1, 1]), RMS_LDS),
        };
        let geometry = geometry.map_err(err)?;
        let mut c = if args.legacy_regs { Gfx10Pm4CommandBuffer::new() } else { Gfx10Pm4CommandBuffer::new_stateful() };
        let fence = self.fence.address() as u64;
        let mut epoch = 0u32;
        if rung == Rung::ReleaseWait {
            // Redline's ReleaseWait entry sentinel (ABA across replays).
            c.dependency_fence(fence, 0);
        }
        c.acquire_system();
        for i in 0..n {
            if i > 0 {
                match rung {
                    Rung::None => {}
                    Rung::Wait => c.wait_compute_idle(),
                    Rung::WaitVmem => {
                        c.wait_compute_idle();
                        c.acquire_inter_node_vmem();
                    }
                    Rung::WaitL1 | Rung::Redline => {
                        c.wait_compute_idle();
                        c.acquire_inter_node_same_agent();
                    }
                    Rung::ReleaseWait => {
                        epoch += 1;
                        c.dependency_fence(fence, epoch);
                    }
                    Rung::WaitGlobal => c.dependency_rmw_global(),
                    Rung::WaitSystem => {
                        c.wait_compute_idle();
                        c.acquire_system();
                    }
                }
            }
            c.dispatch(kernel, geometry, lds, (base + i * stride) as *mut c_void).map_err(err)?;
        }
        // Redline's trailing release.
        c.wait_compute_idle();
        c.acquire_system();
        let ts = self.timestamps.address() as u64;
        let timed = c.with_gpu_timestamps(ts, ts + 8);
        let census: serde_json::Map<String, Value> = timed
            .packet_census()
            .map_err(|at| format!("malformed PM4 stream at dword {at}"))?
            .into_iter()
            .map(|((op, dw), count)| (format!("op{op:#04x}x{dw}"), json!(count)))
            .collect();
        let bytes = timed.as_bytes();
        let mut buffer = self.host_pool.allocate_executable_bytes(bytes.len()).map_err(err)?;
        buffer.as_mut_bytes().copy_from_slice(&bytes);
        Ok(Ib { buffer, dwords: timed.len_dwords(), _kernargs: kernargs, census: Value::Object(census) })
    }

    /// Submit one IB and wait. Returns (GPU µs between the IB's first and
    /// last timestamp packets, host wall µs from doorbell to completion).
    fn submit(&mut self, ib: &Ib) -> R<(f64, f64)> {
        {
            let t = self.timestamps.as_mut_bytes();
            t[..16].fill(0);
        }
        self.completion.reset();
        let packet = PacketImage::pm4_indirect_buffer(ib.buffer.address(), ib.dwords, self.completion.raw()).map_err(err)?;
        self.queues.prepare_batches(&[vec![packet]]).map_err(err)?;
        let t = Instant::now();
        self.queues.ring_prepared().map_err(err)?;
        self.queues.wait_signal(&self.completion, Duration::from_secs(30)).map_err(err)?;
        let wall = t.elapsed().as_secs_f64() * 1e6;
        let (start, end) = {
            let b = self.timestamps.as_mut_bytes();
            let rd = |o: usize| unsafe { std::ptr::read_volatile(b.as_ptr().add(o).cast::<u64>()) };
            (rd(0), rd(8))
        };
        if start == 0 || end <= start {
            return Err(format!("bad GPU timestamps start={start} end={end}"));
        }
        Ok(((end - start) as f64 * 1e6 / self.freq_hz as f64, wall))
    }
}

fn run_pm4(args: &Args, hip: &Hip, pm4: &mut Pm4, rung: Rung) -> R<Value> {
    let ib = pm4.build(args, hip, args.work, rung)?;
    let mut parity = Vec::new();
    if args.work == Work::Rmsnorm {
        for _ in 0..args.parity {
            hip.reset_input()?;
            pm4.submit(&ib)?;
            parity.push(hip.result_hash(args.n)?);
        }
    }
    for _ in 0..3 {
        pm4.submit(&ib)?;
    }
    let mut s = Samples::default();
    let started = Instant::now();
    while started.elapsed().as_secs_f64() < args.secs {
        let (gpu, wall) = pm4.submit(&ib)?;
        s.gpu_us.push(gpu);
        s.wall_us.push(wall);
    }
    Ok(json!({
        "timer": "PM4 COPY_DATA GPU clock at IB start -> RELEASE_MEM bottom-of-pipe timestamp at IB end",
        "rung": rung.name(), "emits": rung.emits(), "regs": if args.legacy_regs { "legacy" } else { "stateful" },
        "ib_dwords": ib.dwords, "packet_census": ib.census,
        "gpu": summary(&s.gpu_us, args.n), "wall": summary(&s.wall_us, args.n),
        "seconds": started.elapsed().as_secs_f64(), "parity_hashes": parity,
    }))
}

fn run_soak(args: &Args, hip: &Hip, pm4: &mut Pm4) -> R<Value> {
    let mut ibs = Vec::new();
    for work in [EMPTY, Work::Rmsnorm] {
        for rung in ALL_RUNGS {
            ibs.push((work, rung, pm4.build(args, hip, work, rung)?));
        }
    }
    let mut counts = vec![0u64; ibs.len()];
    let started = Instant::now();
    'outer: loop {
        for (k, (_, _, ib)) in ibs.iter().enumerate() {
            let t = Instant::now();
            while t.elapsed().as_secs_f64() < 2.0 {
                pm4.submit(ib)?;
                counts[k] += 1;
            }
            if started.elapsed().as_secs_f64() >= args.secs {
                break 'outer;
            }
        }
    }
    let rows: Vec<Value> = ibs
        .iter()
        .zip(&counts)
        .map(|((w, r, _), c)| json!({"work": w.name(), "rung": r.name(), "ibs": c, "dispatches": c * args.n as u64}))
        .collect();
    Ok(json!({"seconds": started.elapsed().as_secs_f64(), "cells": rows}))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[bb] error: {e}");
        std::process::exit(2);
    }
}

fn run() -> R<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).map_err(err)?;
    let started_at = std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();
    let (hip, hip_identity) = hip_setup(&args)?;
    let mut identity = hip_identity;
    let result = match args.arm {
        Arm::HipGraph | Arm::HipStream => run_hip(&args, &hip)?,
        Arm::Pm4(rung) => {
            let (mut pm4, id) = Pm4::setup(&args)?;
            identity.as_object_mut().unwrap().extend(id.as_object().unwrap().clone());
            run_pm4(&args, &hip, &mut pm4, rung)?
        }
        Arm::Soak => {
            let (mut pm4, id) = Pm4::setup(&args)?;
            identity.as_object_mut().unwrap().extend(id.as_object().unwrap().clone());
            run_soak(&args, &hip, &mut pm4)?
        }
    };
    let co = |p: &PathBuf| std::fs::read(p).map(|b| sha256_hex(&b)).unwrap_or_default();
    let doc = json!({
        "schema": "railgun.gfx11_boundary.v1", "label": args.label, "arm": args.arm.name(), "work": args.work.name(),
        "n": args.n, "secs": args.secs, "started_at": started_at, "identity": identity,
        "git_commit": std::env::var("BB_GIT_COMMIT").ok(),
        "rocr_visible_devices": std::env::var("ROCR_VISIBLE_DEVICES").ok(),
        "code_objects": {"empty": co(&args.empty_co), "rmsnorm": co(&args.rmsnorm_co)},
        "result": result,
    });
    let path = args.out.join(format!("{}.json", args.label));
    std::fs::write(&path, serde_json::to_vec_pretty(&doc).map_err(err)?).map_err(err)?;
    let med = |k: &str| doc["result"][k]["median_us_per_kernel"].as_f64().unwrap_or(f64::NAN);
    eprintln!(
        "[bb] {} {} n={} gpu={:.4} wall={:.4} us/kernel parity={} -> {}",
        args.arm.name(),
        args.work.name(),
        args.n,
        med("gpu"),
        med("wall"),
        doc["result"]["parity_hashes"],
        path.display()
    );
    Ok(())
}
