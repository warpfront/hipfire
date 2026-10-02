//! G4 silicon probe (railgun design §5 G4, §7 M2): per arch and per
//! cache-visibility row, adversarial producer→consumer kernel pairs inside
//! one retained PM4 IB, swept over the barrier rungs from none to a full
//! system acquire. A rung is disqualified by any stale observation; the
//! receipt names the weakest rung never observed stale and carries every
//! trial count.
//!
//! One pinned GPU, identity asserted in-process (name, PCI bus, KFD unique
//! id):
//!
//! ```sh
//! ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397 \
//!   cargo run --release -p railgun --features g4-probe --example g4_probe -- \
//!   --arch gfx1201 --bus 0000:13:00.0 --uuid GPU-e475645fe0200397 --out DIR
//! ```
//!
//! Rows (the per-trial IB body; `sys` = wait + the arch's system acquire,
//! which isolates trials; `RUNG` = the rung under test):
//!
//! * `raw_vmem`  — sys, warm(VMEM read), wait, produce, RUNG, consume(VMEM)
//! * `raw_smem`  — sys, warm(SMEM read), wait, produce, RUNG, consume(SMEM):
//!   the `e67c9cad0` shape (GPU-written data read through the scalar cache)
//! * `war`       — sys, read(VMEM, checked, slowed), RUNG, produce
//! * `waw`       — sys, produce(first value, slowed), RUNG, produce, sys, check
//! * `war_pre_writer` — sys, read(VMEM, checked), RUNG, produce, RAW, consume:
//!   the gfx12 retained-vector-line shape (`replay.rs:5386-5419`); RAW is the
//!   arch's shipped dependent rung (gfx12 inter-node, gfx11 same-agent), so
//!   RUNG is the writer-side (pre-writer) acquire alone
//! * `pre_writer_after_barrier` — sys, read(VMEM), sys, indep, RUNG, produce,
//!   RAW, consume: the MQ4R `moe_router → fused_silu_mul_mq_rotate` boundary,
//!   where the writer's last reader lies behind a full barrier and the node
//!   before the writer is independent of it (railgun emits no boundary there,
//!   Redline a wait + system acquire)
//!
//! Every row runs the trial budget split evenly over footprint × layout
//! cells. Footprints: ≤L0, ≤L1, ≤L2, >L2 (bytes on the command line; the
//! defaults fit gfx1100/gfx1151/gfx1201). Layouts: `contig` (producer
//! partition rotated against the reader's by a per-trial offset, so the
//! producer and the stale-line holder vary over WGPs; placement recorded from
//! HW_ID1) and `il32` (partitions interleaved at 32 bytes, so every cache
//! line has several writers on different WGPs).
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use redline_rocr::packet::PacketImage;
use redline_rocr::{
    CompletionSignal, Executable, Gfx10Pm4CommandBuffer, Gfx12Pm4CommandBuffer, Gfx12RmwAcquirePolicy, GpuDevice,
    GpuSelector, KernargBuffer, KernargPool, Kernel, LaunchGeometry, QueueSet, Runtime, load_symbols,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

type R<T> = Result<T, String>;

fn err<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

const HIP_SOURCE: &str = include_str!("g4_probe.hip");
const WG: u32 = 256;
const SLOTS: usize = 8;
const KA_STRIDE: usize = 256;
const FIRST_WRITER: u32 = 1;
const RECORD_KEYS: u32 = 2;
const CLASSIFY_OLD: u32 = 4;
// Result slots (g4_probe.hip `Slot`).
const OLD: usize = 1;
const FUTURE: usize = 2;
const FIRSTV: usize = 3;
const OTHER: usize = 4;
const CHECKED: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Gfx12,
    Gfx11,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Rung {
    None,
    Wait,
    WaitVmem,
    WaitInterNode,
    WaitL1,
    WaitGlobal,
    WaitSystem,
}

impl Rung {
    fn parse(s: &str) -> R<Self> {
        Ok(match s {
            "none" => Self::None,
            "wait" => Self::Wait,
            "wait_vmem" => Self::WaitVmem,
            "wait_inter_node" => Self::WaitInterNode,
            "wait_l1" => Self::WaitL1,
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
            Self::WaitInterNode => "wait_inter_node",
            Self::WaitL1 => "wait_l1",
            Self::WaitGlobal => "wait_global",
            Self::WaitSystem => "wait_system",
        }
    }

    /// What the rung emits on `family`, as it appears in a Redline/railgun IB.
    fn describe(self, family: Family) -> String {
        let acquire = match (family, self) {
            (_, Self::None) => return "no packet".into(),
            (_, Self::Wait) => return "CS_PARTIAL_FLUSH".into(),
            (Family::Gfx12, Self::WaitVmem) => "GCR 0x300 (HipLlvmVmemL1: GLV|GL1)",
            (Family::Gfx12, Self::WaitInterNode) => "GCR 0x10180 (acquire_inter_node_gfx12: GLK|GLV|SEQ)",
            (Family::Gfx12, Self::WaitL1) => "GCR 0x380 (SameAgentParallelL1: GLK|GLV|GL1)",
            (Family::Gfx12, Self::WaitGlobal) => "GCR 0xc380 (RadvGlobal: +GL2 INV|WB)",
            (Family::Gfx12, Self::WaitSystem) => "GCR 0xc3b1 (acquire_system_gfx12)",
            (Family::Gfx11, Self::WaitVmem) => "GCR 0x300 (acquire_inter_node_vmem)",
            (Family::Gfx11, Self::WaitInterNode) => "GCR 0x380 (acquire_inter_node_same_agent)",
            (Family::Gfx11, Self::WaitL1) => "GCR 0x380 (acquire_inter_node_same_agent)",
            (Family::Gfx11, Self::WaitGlobal) => "GCR 0xc380 (dependency_rmw_global)",
            (Family::Gfx11, Self::WaitSystem) => "ROCr system ACQUIRE_MEM (acquire_system)",
        };
        format!("CS_PARTIAL_FLUSH + ACQUIRE_MEM {acquire}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
    RawVmem,
    RawSmem,
    War,
    Waw,
    WarPreWriter,
    PreWriterAfterBarrier,
}

impl Row {
    fn parse(s: &str) -> R<Self> {
        Ok(match s {
            "raw_vmem" => Self::RawVmem,
            "raw_smem" => Self::RawSmem,
            "war" => Self::War,
            "waw" => Self::Waw,
            "war_pre_writer" => Self::WarPreWriter,
            "pre_writer_after_barrier" => Self::PreWriterAfterBarrier,
            _ => return Err(format!("unknown row {s}")),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::RawVmem => "raw_vmem",
            Self::RawSmem => "raw_smem",
            Self::War => "war",
            Self::Waw => "waw",
            Self::WarPreWriter => "war_pre_writer",
            Self::PreWriterAfterBarrier => "pre_writer_after_barrier",
        }
    }

    fn body(self) -> &'static str {
        match self {
            Self::RawVmem => "sys; read_v(warm); wait; produce; RUNG; consume_v",
            Self::RawSmem => "sys; read_s(warm); wait; produce; RUNG; consume_s",
            Self::War => "sys; read_v(checked, delay); RUNG; produce",
            Self::Waw => "sys; produce(first, delay); RUNG; produce; sys; consume_v",
            Self::WarPreWriter => "sys; read_v(checked); RUNG; produce; RAW; consume_v",
            Self::PreWriterAfterBarrier => "sys; read_v(checked); sys; indep; RUNG; produce; RAW; consume_v",
        }
    }

    /// Which counters make a trial stale for this row.
    fn stale_slots(self) -> &'static [usize] {
        match self {
            Self::RawVmem | Self::RawSmem => &[OLD, FIRSTV, OTHER],
            Self::War => &[FUTURE, OTHER],
            Self::Waw => &[FIRSTV, OLD, OTHER],
            Self::WarPreWriter | Self::PreWriterAfterBarrier => &[OLD, FUTURE, FIRSTV, OTHER],
        }
    }
}

struct Kernels {
    zero: Kernel,
    init: Kernel,
    produce: Kernel,
    read_v: Kernel,
    consume_v: Kernel,
    read_s: Kernel,
    consume_s: Kernel,
    indep: Kernel,
}

impl Kernels {
    fn all(&self) -> [(&'static str, &Kernel); 8] {
        [
            ("g4_zero", &self.zero),
            ("g4_init", &self.init),
            ("g4_produce", &self.produce),
            ("g4_read_v", &self.read_v),
            ("g4_consume_v", &self.consume_v),
            ("g4_read_s", &self.read_s),
            ("g4_consume_s", &self.consume_s),
            ("g4_indep", &self.indep),
        ]
    }
}

/// The by-value kernel argument (`P` in g4_probe.hip), 64 bytes.
#[derive(Clone, Copy, Default)]
struct Params {
    data: u64,
    result: u64,
    keys: u64,
    aux: u64,
    n: u32,
    cover: u32,
    epoch: u32,
    rot: u32,
    flags: u32,
    delay: u32,
    nwg: u32,
    layout: u32,
}

impl Params {
    fn bytes(&self) -> [u8; 64] {
        let mut b = [0u8; 64];
        b[0..8].copy_from_slice(&self.data.to_le_bytes());
        b[8..16].copy_from_slice(&self.result.to_le_bytes());
        b[16..24].copy_from_slice(&self.keys.to_le_bytes());
        b[24..32].copy_from_slice(&self.aux.to_le_bytes());
        for (k, v) in [self.n, self.cover, self.epoch, self.rot, self.flags, self.delay, self.nwg, self.layout]
            .into_iter()
            .enumerate()
        {
            b[32 + 4 * k..36 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        b
    }
}

/// One arch's command encoder.
enum Cmd {
    G12(Gfx12Pm4CommandBuffer),
    G11(Gfx10Pm4CommandBuffer),
}

impl Cmd {
    fn new(family: Family) -> Self {
        match family {
            Family::Gfx12 => Self::G12(Gfx12Pm4CommandBuffer::new()),
            Family::Gfx11 => Self::G11(Gfx10Pm4CommandBuffer::new()),
        }
    }

    fn wait(&mut self) {
        match self {
            Self::G12(c) => c.wait_compute_idle(),
            Self::G11(c) => c.wait_compute_idle(),
        }
    }

    fn rung(&mut self, rung: Rung) {
        if rung == Rung::None {
            return;
        }
        self.wait();
        match (self, rung) {
            (_, Rung::None | Rung::Wait) => {}
            (Self::G12(c), Rung::WaitVmem) => c.acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::HipLlvmVmemL1),
            (Self::G12(c), Rung::WaitInterNode) => c.acquire_inter_node_gfx12(),
            (Self::G12(c), Rung::WaitL1) => c.acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::SameAgentParallelL1),
            (Self::G12(c), Rung::WaitGlobal) => c.acquire_rmw_gfx12(Gfx12RmwAcquirePolicy::RadvGlobal),
            (Self::G12(c), Rung::WaitSystem) => c.acquire_system_gfx12(),
            (Self::G11(c), Rung::WaitVmem) => c.acquire_inter_node_vmem(),
            (Self::G11(c), Rung::WaitInterNode | Rung::WaitL1) => c.acquire_inter_node_same_agent(),
            // `dependency_rmw_global` carries its own wait; the extra
            // CS_PARTIAL_FLUSH is harmless.
            (Self::G11(c), Rung::WaitGlobal) => c.dependency_rmw_global(),
            (Self::G11(c), Rung::WaitSystem) => c.acquire_system(),
        }
    }

    /// The arch's shipped rung for a dependent (RAW) boundary.
    fn raw_default(&mut self, family: Family) {
        match family {
            Family::Gfx12 => self.rung(Rung::WaitInterNode),
            Family::Gfx11 => self.rung(Rung::WaitL1),
        }
    }

    fn dispatch(&mut self, kernel: &Kernel, workgroups: u32, kernarg: *mut c_void) -> R<()> {
        let geometry = LaunchGeometry::new([workgroups * WG, 1, 1], [WG as u16, 1, 1]).map_err(err)?;
        match self {
            Self::G12(c) => c.dispatch(kernel, geometry, 0, kernarg).map_err(err),
            Self::G11(c) => c.dispatch(kernel, geometry, 0, kernarg).map_err(err),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::G12(c) => c.as_bytes(),
            Self::G11(c) => c.as_bytes(),
        }
    }

    fn len_dwords(&self) -> u32 {
        match self {
            Self::G12(c) => c.len_dwords(),
            Self::G11(c) => c.len_dwords(),
        }
    }
}

struct Args {
    out: PathBuf,
    arch: String,
    bus: String,
    uuid: Option<String>,
    rows: Vec<Row>,
    rungs: Vec<Rung>,
    trials: u64,
    per_ib: u32,
    soak_secs: u64,
    footprints: Vec<u64>,
    layouts: Vec<u32>,
    diag_trials: u64,
    nwg: u32,
    stop_on_stale: bool,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        out: PathBuf::new(),
        arch: String::new(),
        bus: String::new(),
        uuid: None,
        rows: vec![],
        rungs: vec![],
        trials: 1 << 20,
        per_ib: 64,
        soak_secs: 60,
        footprints: vec![16 << 10, 128 << 10, 1 << 20, 16 << 20],
        layouts: vec![0, 1],
        diag_trials: 16384,
        nwg: 128,
        stop_on_stale: true,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    let list = |s: &str| s.split(',').map(str::to_owned).collect::<Vec<_>>();
    while i < argv.len() {
        let k = argv[i].as_str();
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        match k {
            "--out" => a.out = PathBuf::from(v),
            "--arch" => a.arch = v,
            "--bus" => a.bus = v,
            "--uuid" => a.uuid = Some(v),
            "--rows" => a.rows = list(&v).iter().map(|s| Row::parse(s)).collect::<R<_>>()?,
            "--rungs" => a.rungs = list(&v).iter().map(|s| Rung::parse(s)).collect::<R<_>>()?,
            "--trials" => a.trials = v.parse().map_err(err)?,
            "--per-ib" => a.per_ib = v.parse().map_err(err)?,
            "--soak" => a.soak_secs = v.parse().map_err(err)?,
            "--footprints" => a.footprints = list(&v).iter().map(|s| s.parse().map_err(err)).collect::<R<_>>()?,
            "--layouts" => {
                a.layouts = list(&v)
                    .iter()
                    .map(|s| match s.as_str() {
                        "contig" => Ok(0),
                        "il32" => Ok(1),
                        _ => Err(format!("unknown layout {s}")),
                    })
                    .collect::<R<_>>()?
            }
            "--diag-trials" => a.diag_trials = v.parse().map_err(err)?,
            "--nwg" => a.nwg = v.parse().map_err(err)?,
            "--no-stop" => {
                a.stop_on_stale = false;
                i += 1;
                continue;
            }
            _ => return Err(format!("unknown argument {k}")),
        }
        i += 2;
    }
    if a.out.as_os_str().is_empty() || a.arch.is_empty() || a.bus.is_empty() {
        return Err("--out, --arch and --bus are required".into());
    }
    if a.rows.is_empty() || a.rungs.is_empty() {
        return Err("--rows and --rungs are required".into());
    }
    Ok(a)
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

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

fn compile(out: &Path, arch: &str) -> R<(Vec<u8>, Value)> {
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "/opt/rocm/bin/hipcc".into());
    let dir = out.join("obj");
    std::fs::create_dir_all(&dir).map_err(err)?;
    let src = dir.join("g4_probe.hip");
    std::fs::write(&src, HIP_SOURCE).map_err(err)?;
    let co = dir.join(format!("g4_probe.{arch}.co"));
    let argv = ["--genco", &format!("--offload-arch={arch}"), "-O3", "-o"];
    let status = Command::new(&hipcc)
        .args(argv)
        .arg(&co)
        .arg(&src)
        .status()
        .map_err(|e| format!("{hipcc}: {e}"))?;
    if !status.success() {
        return Err(format!("hipcc failed: {status}"));
    }
    let bytes = std::fs::read(&co).map_err(err)?;
    let version = Command::new(&hipcc)
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().take(3).collect::<Vec<_>>().join(" | "))
        .unwrap_or_default();
    Ok((
        bytes.clone(),
        json!({
            "hipcc": hipcc, "hipcc_version": version, "argv": argv, "source_sha256": sha256_hex(HIP_SOURCE.as_bytes()),
            "code_object": co.display().to_string(), "code_object_sha256": sha256_hex(&bytes),
        }),
    ))
}

struct Gpu {
    family: Family,
    queues: QueueSet,
    completion: CompletionSignal,
    host_pool: KernargPool,
    kernels: Kernels,
    data: KernargBuffer,
    aux: KernargBuffer,
    result: KernargBuffer,
    keys: KernargBuffer,
    nwg: u32,
    per_ib: u32,
    epoch: u32,
}

/// Counts accumulated for one (row, rung, footprint, layout) cell.
#[derive(Default, Clone)]
struct Cell {
    trials: u64,
    stale_trials: u64,
    words: [u64; SLOTS],
    first_stale_trial: Option<u64>,
    ibs: u64,
    seconds: f64,
    placement: BTreeMap<String, u64>,
}

impl Gpu {
    fn submit(&mut self, ib: &KernargBuffer, dwords: u32) -> R<()> {
        self.completion.reset();
        let packet = PacketImage::pm4_indirect_buffer(ib.address(), dwords, self.completion.raw()).map_err(err)?;
        self.queues.prepare_batches(&[vec![packet]]).map_err(err)?;
        self.queues.ring_prepared().map_err(err)?;
        self.queues
            .wait_signal(&self.completion, Duration::from_secs(30))
            .map_err(err)
    }

    fn upload_ib(&self, cmd: &Cmd) -> R<KernargBuffer> {
        let bytes = cmd.bytes();
        let mut ib = self.host_pool.allocate_executable_bytes(bytes.len()).map_err(err)?;
        ib.as_mut_bytes().copy_from_slice(&bytes);
        Ok(ib)
    }

    fn base(&self, n: u32, cover: u32, layout: u32) -> Params {
        Params {
            data: self.data.address() as u64,
            result: self.result.address() as u64,
            keys: 0,
            aux: self.aux.address() as u64,
            n,
            cover,
            epoch: 0,
            rot: 0,
            flags: 0,
            delay: 0,
            nwg: self.nwg,
            layout,
        }
    }

    /// Build the IB of one batch: zero counters, re-init the footprint to
    /// `epoch0 - 1`, then `per_ib` trials of `row` with `rung`. Returns the
    /// IB and the kernarg slots with their per-trial role, so later batches
    /// only rewrite epochs/rotations.
    fn build(
        &self,
        row: Row,
        rung: Rung,
        n: u32,
        cover: u32,
        layout: u32,
        kernargs: &mut KernargBuffer,
    ) -> R<(Cmd, Vec<(usize, Option<u32>, Params)>)> {
        let family = self.family;
        let mut cmd = Cmd::new(family);
        let k = &self.kernels;
        let base = kernargs.address() as usize;
        let mut slots: Vec<(usize, Option<u32>, Params)> = Vec::new();
        let mut push = |cmd: &mut Cmd, kernel: &Kernel, wgs: u32, trial: Option<u32>, p: Params| -> R<()> {
            let offset = slots.len() * KA_STRIDE;
            slots.push((offset, trial, p));
            cmd.dispatch(kernel, wgs, (base + offset) as *mut c_void)
        };
        let sys = |cmd: &mut Cmd| cmd.rung(Rung::WaitSystem);
        cmd.rung(Rung::WaitSystem);
        let mut zero = self.base(self.per_ib * SLOTS as u32, 0, layout);
        zero.data = 0;
        push(&mut cmd, &k.zero, self.nwg, None, zero)?;
        let init = self.base(n, cover, layout);
        push(&mut cmd, &k.init, self.nwg, Some(u32::MAX), init)?; // epoch0 - 1
        for t in 0..self.per_ib {
            let mut p = self.base(n, cover, layout);
            p.result = self.result.address() as u64 + (t as u64) * (SLOTS as u64) * 4;
            if t == 0 && layout == 0 {
                p.keys = self.keys.address() as u64;
                p.flags |= RECORD_KEYS;
            }
            let tr = Some(t);
            sys(&mut cmd);
            match row {
                Row::RawVmem | Row::RawSmem => {
                    let (warm, consume) =
                        if row == Row::RawVmem { (&k.read_v, &k.consume_v) } else { (&k.read_s, &k.consume_s) };
                    push(&mut cmd, warm, self.nwg, tr, p)?;
                    cmd.wait();
                    push(&mut cmd, &k.produce, self.nwg, tr, p)?;
                    cmd.rung(rung);
                    push(&mut cmd, consume, self.nwg, tr, p)?;
                }
                Row::War => {
                    push(&mut cmd, &k.read_v, self.nwg, tr, Params { flags: p.flags | CLASSIFY_OLD, delay: 4, ..p })?;
                    cmd.rung(rung);
                    push(&mut cmd, &k.produce, self.nwg, tr, p)?;
                }
                Row::Waw => {
                    push(&mut cmd, &k.produce, self.nwg, tr, Params { flags: p.flags | FIRST_WRITER, delay: 4, ..p })?;
                    cmd.rung(rung);
                    push(&mut cmd, &k.produce, self.nwg, tr, p)?;
                    sys(&mut cmd);
                    push(&mut cmd, &k.consume_v, self.nwg, tr, p)?;
                }
                Row::WarPreWriter => {
                    push(&mut cmd, &k.read_v, self.nwg, tr, Params { flags: p.flags | CLASSIFY_OLD, ..p })?;
                    cmd.rung(rung);
                    push(&mut cmd, &k.produce, self.nwg, tr, p)?;
                    cmd.raw_default(family);
                    push(&mut cmd, &k.consume_v, self.nwg, tr, p)?;
                }
                Row::PreWriterAfterBarrier => {
                    push(&mut cmd, &k.read_v, self.nwg, tr, Params { flags: p.flags | CLASSIFY_OLD, ..p })?;
                    sys(&mut cmd);
                    push(&mut cmd, &k.indep, 16, tr, p)?;
                    cmd.rung(rung);
                    push(&mut cmd, &k.produce, self.nwg, tr, p)?;
                    cmd.raw_default(family);
                    push(&mut cmd, &k.consume_v, self.nwg, tr, p)?;
                }
            }
        }
        cmd.wait();
        if slots.len() * KA_STRIDE > kernargs.len() {
            return Err("kernarg slots overflow".into());
        }
        Ok((cmd, slots))
    }

    fn write_kernargs(&mut self, kernargs: &mut KernargBuffer, slots: &[(usize, Option<u32>, Params)], rots: &[u32]) {
        let epoch0 = self.epoch;
        let bytes = kernargs.as_mut_bytes();
        for (offset, trial, p) in slots {
            let mut p = *p;
            match trial {
                None => {}
                Some(u32::MAX) => p.epoch = epoch0.wrapping_sub(1),
                Some(t) => {
                    p.epoch = epoch0.wrapping_add(*t);
                    p.rot = rots[*t as usize];
                }
            }
            bytes[*offset..*offset + 64].copy_from_slice(&p.bytes());
        }
        self.epoch = epoch0.wrapping_add(self.per_ib).wrapping_add(1);
    }

    fn read_u32(buf: &mut KernargBuffer, count: usize) -> Vec<u32> {
        let bytes = buf.as_mut_bytes();
        // Device-local (BAR-mapped) memory: volatile word reads.
        (0..count)
            .map(|i| unsafe { std::ptr::read_volatile(bytes.as_ptr().add(4 * i).cast::<u32>()) })
            .collect()
    }
}

fn run_cell(
    gpu: &mut Gpu,
    row: Row,
    rung: Rung,
    footprint: u64,
    layout: u32,
    trials: u64,
    rng: &mut u64,
) -> R<Cell> {
    let n = (footprint / 4) as u32;
    if n % (gpu.nwg * 8) != 0 {
        return Err(format!("footprint {footprint} not a multiple of nwg*32 bytes"));
    }
    // ≤L0/≤L1 footprints: every reader workgroup reads the whole buffer, so
    // every WGP holds every (stale) line; larger ones: one slice per group.
    let cover = if footprint <= (128 << 10) { n } else { n / gpu.nwg };
    let mut kernargs = gpu
        .host_pool
        .allocate_fine_grained_bytes((gpu.per_ib as usize * 8 + 4) * KA_STRIDE, 256)
        .map_err(err)?;
    let (cmd, slots) = gpu.build(row, rung, n, cover, layout, &mut kernargs)?;
    let ib = gpu.upload_ib(&cmd)?;
    let dwords = cmd.len_dwords();
    let mut cell = Cell::default();
    let started = Instant::now();
    while cell.trials < trials {
        let rots: Vec<u32> = (0..gpu.per_ib)
            .map(|_| {
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                (*rng % gpu.nwg as u64) as u32
            })
            .collect();
        gpu.write_kernargs(&mut kernargs, &slots, &rots);
        gpu.submit(&ib, dwords)?;
        let counts = Gpu::read_u32(&mut gpu.result, gpu.per_ib as usize * SLOTS);
        for t in 0..gpu.per_ib as usize {
            let c = &counts[t * SLOTS..(t + 1) * SLOTS];
            for s in 0..SLOTS {
                cell.words[s] += c[s] as u64;
            }
            if c[CHECKED] == 0 {
                return Err(format!("{} {} trial {t}: nothing checked", row.name(), rung.name()));
            }
            if row.stale_slots().iter().any(|&s| c[s] != 0) {
                cell.stale_trials += 1;
                cell.first_stale_trial.get_or_insert(cell.trials + t as u64);
            }
        }
        if layout == 0 {
            // Roles recorded on trial 0: reader (0, by group), producer (1, by
            // partition), consumer (2, by group). With a sliced cover, reader
            // and consumer group w touch exactly partition w; with the whole
            // buffer per group (≤L1 footprints) every group touches every
            // partition, so only the WGP spread is meaningful.
            let keys = Gpu::read_u32(&mut gpu.keys, 3 * gpu.nwg as usize);
            let g = gpu.nwg as usize;
            let role = |r: usize| &keys[r * g..(r + 1) * g];
            let (has_reader, has_consumer) = match row {
                Row::War => (true, false),
                Row::Waw => (false, true),
                _ => (true, true),
            };
            let mut pairs: Vec<(&'static str, usize, usize)> = Vec::new();
            if has_reader {
                pairs.push(("reader_producer", 0, 1));
            }
            if has_consumer {
                pairs.push(("producer_consumer", 1, 2));
            }
            if has_reader && has_consumer {
                pairs.push(("reader_consumer", 0, 2));
            }
            for (label, x, y) in pairs {
                if cover != n {
                    let same = (0..g).filter(|&w| role(x)[w] == role(y)[w]).count() as u64;
                    *cell.placement.entry(format!("{label}_same_wgp")).or_default() += same;
                    *cell.placement.entry(format!("{label}_cross_wgp")).or_default() += g as u64 - same;
                }
            }
            for (label, r, present) in [("reader", 0, has_reader), ("producer", 1, true), ("consumer", 2, has_consumer)] {
                if present {
                    let distinct: std::collections::BTreeSet<u32> = role(r).iter().copied().collect();
                    let e = cell.placement.entry(format!("{label}_distinct_wgps_max")).or_default();
                    *e = (*e).max(distinct.len() as u64);
                }
            }
        }
        cell.trials += gpu.per_ib as u64;
        cell.ibs += 1;
    }
    cell.seconds = started.elapsed().as_secs_f64();
    Ok(cell)
}

fn soak(gpu: &mut Gpu, secs: u64) -> R<Value> {
    let mut report = serde_json::Map::new();
    let n = (1u32 << 20) / 4;
    let kernels: Vec<(&'static str, Kernel)> = gpu.kernels.all().iter().map(|(s, k)| (*s, (*k).clone())).collect();
    for (name, kernel) in kernels {
        let mut kernargs = gpu.host_pool.allocate_fine_grained_bytes(64 * KA_STRIDE, 256).map_err(err)?;
        let mut cmd = Cmd::new(gpu.family);
        cmd.rung(Rung::WaitSystem);
        let mut p = gpu.base(n, n / gpu.nwg, 0);
        if name == "g4_zero" {
            p.n = gpu.per_ib * SLOTS as u32;
        }
        let bytes = kernargs.as_mut_bytes();
        for i in 0..64 {
            let mut q = p;
            q.epoch = i as u32;
            bytes[i * KA_STRIDE..i * KA_STRIDE + 64].copy_from_slice(&q.bytes());
        }
        let base = kernargs.address() as usize;
        for i in 0..64 {
            cmd.dispatch(&kernel, gpu.nwg, (base + i * KA_STRIDE) as *mut c_void)?;
            cmd.rung(Rung::WaitInterNode);
        }
        let ib = gpu.upload_ib(&cmd)?;
        let started = Instant::now();
        let mut ibs = 0u64;
        while started.elapsed().as_secs() < secs {
            gpu.submit(&ib, cmd.len_dwords())?;
            ibs += 1;
        }
        eprintln!("[g4] soak {name}: {ibs} IBs x 64 dispatches in {:.1}s", started.elapsed().as_secs_f64());
        report.insert(name.into(), json!({"ibs": ibs, "dispatches": ibs * 64, "seconds": started.elapsed().as_secs_f64()}));
    }
    Ok(Value::Object(report))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[g4] error: {e}");
        std::process::exit(2);
    }
}

fn run() -> R<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).map_err(err)?;
    let runtime = Runtime::initialize(load_symbols().map_err(err)?).map_err(err)?;
    let devices = runtime.gpu_devices().map_err(err)?;
    if devices.len() != 1 {
        return Err(format!("expected exactly one visible GPU, found {}", devices.len()));
    }
    let device: GpuDevice = runtime.select_gpu(GpuSelector::Ordinal(0)).map_err(err)?;
    let name = device.name().to_owned();
    let bus = device.pci_bus_id().to_string();
    let kfd = kfd_uuid(&bus)?;
    if name != args.arch || bus != args.bus {
        return Err(format!("identity mismatch: device {name} {bus}, expected {} {}", args.arch, args.bus));
    }
    if let Some(want) = &args.uuid {
        if kfd.as_deref() != Some(want.as_str()) {
            return Err(format!("uuid mismatch: KFD {kfd:?} expected {want}"));
        }
    }
    let cus = device.compute_unit_count().map_err(err)?;
    eprintln!("[g4] asserted {name} {bus} {} cus={cus}", kfd.as_deref().unwrap_or("-"));
    let family = match name.as_str() {
        "gfx1200" | "gfx1201" => Family::Gfx12,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" => Family::Gfx11,
        other => return Err(format!("unsupported arch {other}")),
    };
    let (co, compile_info) = compile(&args.out, &name)?;
    let exe = Executable::load(&device, Arc::from(co.into_boxed_slice())).map_err(err)?;
    let kernel = |s: &str| exe.kernel(&format!("{s}.kd")).map_err(err);
    let kernels = Kernels {
        zero: kernel("g4_zero")?,
        init: kernel("g4_init")?,
        produce: kernel("g4_produce")?,
        read_v: kernel("g4_read_v")?,
        consume_v: kernel("g4_consume_v")?,
        read_s: kernel("g4_read_s")?,
        consume_s: kernel("g4_consume_s")?,
        indep: kernel("g4_indep")?,
    };
    let host_pool = KernargPool::discover(&device).map_err(err)?;
    let dev_pool = KernargPool::discover_device_local(&device).map_err(err)?;
    let max_fp = *args.footprints.iter().max().ok_or("no footprints")?;
    let data = dev_pool.allocate_fine_grained_bytes(max_fp as usize, 1 << 16).map_err(err)?;
    let aux = dev_pool.allocate_fine_grained_bytes(1 << 16, 4096).map_err(err)?;
    let result = dev_pool.allocate_fine_grained_bytes(args.per_ib as usize * SLOTS * 4, 4096).map_err(err)?;
    let keys = dev_pool.allocate_fine_grained_bytes(3 * args.nwg as usize * 4, 4096).map_err(err)?;
    let completion = CompletionSignal::new(&device).map_err(err)?;
    let queue_size = *device.queue_size_range().start();
    let queues = QueueSet::create(&device, 1, queue_size).map_err(err)?;
    let mut gpu = Gpu {
        family,
        queues,
        completion,
        host_pool,
        kernels,
        data,
        aux,
        result,
        keys,
        nwg: args.nwg,
        per_ib: args.per_ib,
        epoch: 0x1000,
    };
    let soak_report = if args.soak_secs > 0 { soak(&mut gpu, args.soak_secs)? } else { Value::Null };
    let driver = std::fs::read_to_string("/sys/module/amdgpu/version").unwrap_or_default().trim().to_owned();
    let rocm = std::fs::read_to_string("/opt/rocm/.info/version").unwrap_or_default().trim().to_owned();
    let kernel_release = std::fs::read_to_string("/proc/sys/kernel/osrelease").unwrap_or_default().trim().to_owned();
    let hostname = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default().trim().to_owned();
    let started_at = Command::new("date").arg("-u").arg("+%Y-%m-%dT%H:%M:%SZ").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned()).unwrap_or_default();
    let identity = json!({
        "arch": name, "bus": bus, "kfd_unique_id": kfd, "compute_units": cus, "host": hostname,
        "amdgpu_module_version": driver, "rocm_version": rocm, "kernel": kernel_release,
        "rocr_visible_devices": std::env::var("ROCR_VISIBLE_DEVICES").ok(),
        "git_commit": std::env::var("G4_GIT_COMMIT").ok(), "started_at": started_at,
    });
    let cells_per = (args.footprints.len() * args.layouts.len()) as u64;
    let per_cell = args.trials.div_ceil(cells_per).div_ceil(args.per_ib as u64) * args.per_ib as u64;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut rows_out = Vec::new();
    for &row in &args.rows {
        let mut rungs_out = Vec::new();
        for &rung in &args.rungs {
            let mut disqualified = false;
            let mut cells = Vec::new();
            let mut totals = Cell::default();
            for &fp in &args.footprints {
                for &layout in &args.layouts {
                    let quota = if disqualified && args.stop_on_stale { args.diag_trials.min(per_cell) } else { per_cell };
                    let cell = run_cell(&mut gpu, row, rung, fp, layout, quota, &mut rng)?;
                    eprintln!(
                        "[g4] {} {} fp={} layout={} trials={} stale_trials={} old={} future={} first={} other={} checked={} {:.1}s",
                        row.name(), rung.name(), fp, layout, cell.trials, cell.stale_trials, cell.words[OLD], cell.words[FUTURE],
                        cell.words[FIRSTV], cell.words[OTHER], cell.words[CHECKED], cell.seconds
                    );
                    disqualified |= cell.stale_trials > 0;
                    totals.trials += cell.trials;
                    totals.stale_trials += cell.stale_trials;
                    for s in 0..SLOTS {
                        totals.words[s] += cell.words[s];
                    }
                    totals.seconds += cell.seconds;
                    cells.push(json!({
                        "footprint_bytes": fp, "layout": if layout == 0 { "contig" } else { "il32" },
                        "reader_cover_dwords": if fp <= (128 << 10) { fp / 4 } else { fp / 4 / args.nwg as u64 },
                        "trials": cell.trials, "stale_trials": cell.stale_trials, "first_stale_trial": cell.first_stale_trial,
                        "words": {"old": cell.words[OLD], "future": cell.words[FUTURE], "first_writer": cell.words[FIRSTV],
                                  "other": cell.words[OTHER], "checked": cell.words[CHECKED]},
                        "ibs": cell.ibs, "seconds": cell.seconds, "placement": cell.placement,
                    }));
                }
            }
            rungs_out.push(json!({
                "rung": rung.name(), "emits": rung.describe(family), "trials": totals.trials,
                "stale_trials": totals.stale_trials, "stale": totals.stale_trials > 0,
                "words": {"old": totals.words[OLD], "future": totals.words[FUTURE], "first_writer": totals.words[FIRSTV],
                          "other": totals.words[OTHER], "checked": totals.words[CHECKED]},
                "seconds": totals.seconds, "cells": cells,
            }));
            let snapshot = json!({"identity": identity, "partial_row": row.name(), "rungs": rungs_out});
            std::fs::write(args.out.join(format!("partial.{}.json", row.name())), serde_json::to_vec_pretty(&snapshot).map_err(err)?).map_err(err)?;
        }
        // Weakest safe rung: the first rung in ladder order that is clean and
        // every rung after it that was run is clean too.
        let clean: Vec<bool> = rungs_out.iter().map(|r| !r["stale"].as_bool().unwrap_or(true)).collect();
        let weakest = (0..clean.len()).find(|&i| clean[i..].iter().all(|c| *c)).map(|i| rungs_out[i]["rung"].clone());
        let next_weaker_stale = weakest
            .as_ref()
            .and_then(|w| rungs_out.iter().position(|r| &r["rung"] == w))
            .map(|i| i > 0 && !clean[i - 1]);
        rows_out.push(json!({
            "row": row.name(), "body": row.body(), "weakest_safe_rung": weakest,
            "next_weaker_rung_shown_stale": next_weaker_stale, "rungs": rungs_out,
        }));
        let receipt = json!({
            "schema": "railgun.g4.receipt.v1", "identity": identity, "compile": compile_info,
            "trials_per_rung_target": args.trials, "per_ib": args.per_ib, "workgroups": args.nwg,
            "footprints_bytes": args.footprints, "row": rows_out.last().unwrap(),
        });
        let path = args.out.join(format!("receipt.{}.{}.json", name, row.name()));
        std::fs::write(&path, serde_json::to_vec_pretty(&receipt).map_err(err)?).map_err(err)?;
        eprintln!("[g4] wrote {}", path.display());
    }
    let table = json!({
        "schema": "railgun.g4.cache_table.v1", "identity": identity, "compile": compile_info, "soak": soak_report,
        "ladder": args.rungs.iter().map(|r| json!({"rung": r.name(), "emits": r.describe(family)})).collect::<Vec<_>>(),
        "rows": rows_out.iter().map(|r| json!({"row": r["row"], "weakest_safe_rung": r["weakest_safe_rung"],
            "next_weaker_rung_shown_stale": r["next_weaker_rung_shown_stale"],
            "trials": r["rungs"].as_array().map(|a| a.iter().map(|x| json!({"rung": x["rung"], "trials": x["trials"], "stale_trials": x["stale_trials"]})).collect::<Vec<_>>())})).collect::<Vec<_>>(),
    });
    let mut f = std::fs::File::create(args.out.join(format!("cache-table.{name}.json"))).map_err(err)?;
    f.write_all(&serde_json::to_vec_pretty(&table).map_err(err)?).map_err(err)?;
    Ok(())
}
