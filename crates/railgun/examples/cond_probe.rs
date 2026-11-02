// SPDX-License-Identifier: Apache-2.0
//! #867 conditional-node probe on a GFX11 compute queue (gfx1151): railgun
//! IF and bounded WHILE programs on a device-written ControlWord, in both
//! lowerings, against the host-branched program.
//!
//! The kernels are `kernels/railgun_guard_pm_gfx1151.co` (peacemaker-isa
//! `railgun_guard`), authored through their railgun-cert receipt
//! (`railgun-cert certify --out DIR <co>`), so the guard certificate is the
//! receipt's. Programs, per trial (fresh random `x`, `y`):
//! - IF: `set(flag, y, limit)` then IF(flag) { `add(flag, x, y, n)` };
//! - WHILE: `set` then `--iters` copies of `{ add; step(flag, y, limit) }`,
//!   each an IF on `flag` (`step` writes `y[0] < limit`).
//!
//! Each program runs as one retained PM4 IB on a Redline-created queue, in
//! two lowerings: `kernel_guard` (unchanged tape) and `cp_skip` (`COND_EXEC`
//! behind `RELEASE_MEM` + `WAIT_REG_MEM`; there `add` reads a constant set
//! flag, so only the CP can skip it). The reference is host-branched:
//! HIP launches of `add` with a constant set flag, the host reading `y[0]`
//! back before each decision. Every trial compares `y` and the final flag
//! byte for byte. A negative control (IF body unguarded) must differ on
//! every clear-flag trial. Boundaries are planned with every row at
//! `WaitIdle + System` (`RowPolicy::UnreceiptedSystem`; railgun has no gfx11
//! cache table), so timings carry that cost.
//!
//! ```sh
//! cond_probe --co kernels/railgun_guard_pm_gfx1151.co \
//!   --receipt R.receipt.json --out DIR --trials 2000 --iters 8 --n 1048576
//! ```
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hip_bridge::{DeviceBuffer, HipRuntime};
use railgun::kernel::{
    author, CondLowering, Conditional, KernelFacts, KernelProgram, KernelRef, LaunchRecord,
    NodeFacts,
};
use railgun::plan::{self, CacheTable, Pm4Op, RowPolicy, Rung};
use railgun::Binding;
use railgun_corpus::receipt::Receipt;
use redline_rocr::packet::PacketImage;
use redline_rocr::{
    load_symbols, CompletionSignal, Executable, Gfx10Pm4CommandBuffer, GpuSelector, KernargBuffer,
    KernargPool, Kernel, LaunchGeometry, QueueSet, Runtime,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const ADD: &str = "railgun_guard_add_pm";
const STEP: &str = "railgun_guard_step_pm";
const SET: &str = "railgun_flag_set_pm";
const BLOCK: u32 = 32;
/// IF threshold: `y[0] < 2^31` sets the flag for half the random `y[0]`.
const IF_LIMIT: u32 = 1 << 31;
/// WHILE: `y[0]` starts in `0..WHILE_Y0` and grows by `x[0]` in
/// `WHILE_D` per iteration, so trips span 0 to past `--iters`.
const WHILE_LIMIT: u32 = 1000;
const WHILE_Y0: u32 = 1300;
const WHILE_D: Range<u32> = 100..301;

struct Args {
    co: PathBuf,
    receipt: PathBuf,
    out: PathBuf,
    trials: usize,
    iters: usize,
    n: u32,
    secs: f64,
    /// Run only arms whose name contains this.
    arm: String,
}

fn args() -> R<Args> {
    let mut a = Args {
        co: PathBuf::new(),
        receipt: PathBuf::new(),
        out: PathBuf::new(),
        trials: 1000,
        iters: 8,
        n: 1 << 20,
        secs: 3.0,
        arm: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().ok_or(format!("{k} needs a value"))?;
        match k.as_str() {
            "--co" => a.co = v.into(),
            "--receipt" => a.receipt = v.into(),
            "--out" => a.out = v.into(),
            "--trials" => a.trials = v.parse().map_err(err)?,
            "--iters" => a.iters = v.parse().map_err(err)?,
            "--n" => a.n = v.parse().map_err(err)?,
            "--secs" => a.secs = v.parse().map_err(err)?,
            "--arm" => a.arm = v,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.n == 0 || !a.n.is_multiple_of(BLOCK) {
        return Err(format!("--n must be a positive multiple of {BLOCK}"));
    }
    Ok(a)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn range(&mut self, r: Range<u32>) -> u32 {
        r.start + self.next() % (r.end - r.start)
    }
}

struct Buffers {
    flag: DeviceBuffer,
    /// Constant 1: the host-branched reference's (and negative control's) flag.
    one: DeviceBuffer,
    x: DeviceBuffer,
    y: DeviceBuffer,
}

impl Buffers {
    fn binding(b: &DeviceBuffer) -> Binding {
        Binding {
            base: b.as_ptr() as u64,
            bytes: b.size() as u64,
        }
    }
}

/// Explicit kernarg bytes: pointers then one trailing `u32`.
fn kernarg(ptrs: &[&DeviceBuffer], scalar: u32, bytes: usize) -> Vec<u8> {
    let mut k = vec![0u8; bytes];
    for (i, p) in ptrs.iter().enumerate() {
        k[8 * i..8 * i + 8].copy_from_slice(&(p.as_ptr() as u64).to_le_bytes());
    }
    let at = 8 * ptrs.len();
    k[at..at + 4].copy_from_slice(&scalar.to_le_bytes());
    k
}

struct Facts(Vec<(String, Arc<NodeFacts>)>);

impl Facts {
    fn load(path: &PathBuf) -> R<(Self, String)> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let receipt: Receipt = serde_json::from_slice(&bytes).map_err(err)?;
        let mut out = Vec::new();
        for k in &receipt.railgun.kernels {
            let identity = receipt
                .kernels
                .iter()
                .find(|i| i.symbol == k.symbol)
                .ok_or(format!("no identity for {}", k.symbol))?;
            out.push((
                k.symbol.clone(),
                Arc::new(NodeFacts {
                    kref: KernelRef {
                        module: "railgun_guard_pm_gfx1151".into(),
                        symbol: k.symbol.clone(),
                        artifact_sha256: receipt.object_sha256.clone(),
                        kd_sha256: identity.kd_sha256.clone(),
                        receipt_sha256: sha256_hex(&bytes),
                        tier: receipt.tier.clone(),
                        obligations: receipt.obligations.clone(),
                    },
                    facts: KernelFacts::from_record(k)?,
                }),
            ));
        }
        Ok((Self(out), receipt.object_sha256))
    }

    fn get(&self, symbol: &str) -> R<Arc<NodeFacts>> {
        self.0
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, f)| f.clone())
            .ok_or(format!("receipt has no {symbol}"))
    }
}

/// The three launches, recorded as a composer would hand them to railgun.
struct Launches<'a> {
    facts: &'a Facts,
    b: &'a Buffers,
    n: u32,
}

impl Launches<'_> {
    fn record(
        &self,
        symbol: &str,
        grid: u32,
        kernarg: Vec<u8>,
        ptrs: &[&DeviceBuffer],
    ) -> R<LaunchRecord> {
        Ok(LaunchRecord {
            symbol: symbol.into(),
            grid: [grid, 1, 1],
            block: [BLOCK, 1, 1],
            dynamic_lds: 0,
            kernarg,
            declared: Vec::new(),
            facts: Ok(self.facts.get(symbol)?),
            allocations: ptrs
                .iter()
                .enumerate()
                .map(|(i, p)| (8 * i as u32, Ok(Buffers::binding(p))))
                .collect(),
        })
    }
    fn set(&self, limit: u32) -> R<LaunchRecord> {
        let p = [&self.b.flag, &self.b.y];
        self.record(SET, 1, kernarg(&p, limit, 24), &p)
    }
    fn step(&self, limit: u32) -> R<LaunchRecord> {
        let p = [&self.b.flag, &self.b.y];
        self.record(STEP, 1, kernarg(&p, limit, 24), &p)
    }
    fn add(&self, flag: &DeviceBuffer) -> R<LaunchRecord> {
        let p = [flag, &self.b.x, &self.b.y];
        self.record(ADD, self.n / BLOCK, kernarg(&p, self.n, 32), &p)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    If,
    While,
    /// IF program with the body run unconditionally (negative control).
    Ungated,
}

fn program(l: &Launches, shape: Shape, iters: usize, lowering: CondLowering) -> R<KernelProgram> {
    let mut launches = vec![];
    let limit = if shape == Shape::While {
        WHILE_LIMIT
    } else {
        IF_LIMIT
    };
    launches.push(l.set(limit)?);
    // Under CP skip `add` reads the constant set flag, so only COND_EXEC can
    // skip it (its own guard would otherwise mask an ignored COND_EXEC).
    let body_flag = match lowering {
        CondLowering::KernelGuard => &l.b.flag,
        CondLowering::CpSkip => &l.b.one,
    };
    match shape {
        Shape::If => launches.push(l.add(body_flag)?),
        Shape::Ungated => launches.push(l.add(&l.b.one)?),
        Shape::While => {
            for _ in 0..iters {
                launches.push(l.add(body_flag)?);
                launches.push(l.step(limit)?);
            }
        }
    }
    let mut p = author("gfx1151", &launches);
    for (i, node) in p.nodes.iter().enumerate() {
        if !node.lowering.is_pm4() {
            return Err(format!(
                "node {i} {} is not PM4: {:?}",
                node.symbol, node.lowering
            ));
        }
    }
    let flag = p.nodes[0].effects[0].resource;
    match shape {
        Shape::If => p.add_conditional(Conditional {
            flag,
            body: 1..2,
            lowering,
        })?,
        Shape::While => p.bounded_while(flag, 1, 2, iters, lowering)?,
        Shape::Ungated => {}
    }
    Ok(p)
}

struct Pm4 {
    queues: QueueSet,
    completion: CompletionSignal,
    pool: KernargPool,
    kernels: Vec<(String, Kernel)>,
    fence: KernargBuffer,
    timestamps: KernargBuffer,
    freq_hz: u64,
    _exe: Executable,
}

struct Ib {
    buffer: KernargBuffer,
    dwords: u32,
    dwords_hex: Vec<u32>,
    skipped_dwords: Vec<u32>,
    _kernargs: Vec<KernargBuffer>,
}

impl Pm4 {
    fn setup(co: &[u8]) -> R<(Self, Value)> {
        let runtime = Runtime::initialize(load_symbols().map_err(err)?).map_err(err)?;
        let device = runtime.select_gpu(GpuSelector::Ordinal(0)).map_err(err)?;
        if device.name() != "gfx1151" {
            return Err(format!("{} is not gfx1151", device.name()));
        }
        let exe =
            Executable::load(&device, Arc::from(co.to_vec().into_boxed_slice())).map_err(err)?;
        let mut kernels = Vec::new();
        for s in [ADD, STEP, SET] {
            kernels.push((s.to_owned(), exe.kernel(&format!("{s}.kd")).map_err(err)?));
        }
        let pool = KernargPool::discover(&device).map_err(err)?;
        let fence = pool.allocate_fine_grained_bytes(64, 256).map_err(err)?;
        let timestamps = pool.allocate_fine_grained_bytes(64, 256).map_err(err)?;
        let completion = CompletionSignal::new(&device).map_err(err)?;
        let queues =
            QueueSet::create(&device, 1, *device.queue_size_range().start()).map_err(err)?;
        let freq_hz = device.gpu_timestamp_frequency_hz().map_err(err)?;
        let identity = json!({
            "rocr_arch": device.name(), "rocr_bus": device.pci_bus_id().to_string(),
            "compute_units": device.compute_unit_count().map_err(err)?,
        });
        Ok((
            Self {
                queues,
                completion,
                pool,
                kernels,
                fence,
                timestamps,
                freq_hz,
                _exe: exe,
            },
            identity,
        ))
    }

    /// Lower `p` (one segment) to a Redline gfx11 IB. Boundaries: wait =
    /// `CS_PARTIAL_FLUSH`, `System` = ROCr's system `ACQUIRE_MEM`.
    fn build(&self, p: &KernelProgram) -> R<Ib> {
        let table = CacheTable {
            arch: "gfx1151".into(),
            rows: CacheTable::for_arch("gfx1201").unwrap().rows,
        };
        let pl = plan::plan(p, &table, RowPolicy::UnreceiptedSystem);
        if pl.segments.len() != 1 {
            return Err(format!("{} segments", pl.segments.len()));
        }
        let words = p.authored_words();
        let mut c = Gfx10Pm4CommandBuffer::new_stateful();
        let fence = self.fence.address() as u64;
        // Entry sentinel: later epochs never meet a stale fence value.
        c.dependency_fence(fence, 0);
        let mut epoch = 0u32;
        let mut open = None;
        let mut skipped = Vec::new();
        let mut kernargs = Vec::new();
        for op in pl.segment_ops(&pl.segments[0]) {
            match op {
                Pm4Op::Entry => c.acquire_system(),
                Pm4Op::Boundary(v) => {
                    if v.wait_idle {
                        c.wait_compute_idle();
                    }
                    match v.acquire {
                        Rung::None => {}
                        Rung::InterNode => c.acquire_inter_node_same_agent(),
                        Rung::System => c.acquire_system(),
                    }
                }
                Pm4Op::Dispatch(i) => {
                    let node = &p.nodes[i];
                    let kernel = &self
                        .kernels
                        .iter()
                        .find(|(s, _)| *s == node.symbol)
                        .ok_or("kernel")?
                        .1;
                    let metadata = kernel.metadata();
                    let image =
                        p.kernarg_image(i, metadata.kernarg_segment_size as usize, &words)?;
                    let mut buffer = self.pool.allocate_for(metadata).map_err(err)?;
                    buffer.write_exact(&image).map_err(err)?;
                    let geometry = LaunchGeometry::new(
                        [node.grid[0] * node.block[0], 1, 1],
                        [node.block[0] as u16, 1, 1],
                    )
                    .map_err(err)?;
                    c.dispatch(kernel, geometry, 0, buffer.address())
                        .map_err(err)?;
                    kernargs.push(buffer);
                }
                Pm4Op::SkipBegin(flag) => {
                    epoch += 1;
                    c.dependency_fence(fence, epoch);
                    let at = c.len_dwords();
                    open = Some((c.begin_cond_exec(p.resources[flag.0 as usize].base), at));
                }
                Pm4Op::SkipEnd => {
                    let (region, at) = open.take().ok_or("unbalanced skip")?;
                    c.end_cond_exec(region)
                        .map_err(|n| format!("COND_EXEC count {n}"))?;
                    skipped.push(c.len_dwords() - at - 5);
                }
                Pm4Op::Exit => {
                    c.wait_compute_idle();
                    c.acquire_system();
                }
            }
        }
        let ts = self.timestamps.address() as u64;
        let timed = c.with_gpu_timestamps(ts, ts + 8);
        let bytes = timed.as_bytes();
        let mut buffer = self
            .pool
            .allocate_executable_bytes(bytes.len())
            .map_err(err)?;
        buffer.as_mut_bytes().copy_from_slice(&bytes);
        Ok(Ib {
            buffer,
            dwords: timed.len_dwords(),
            dwords_hex: timed.dwords().to_vec(),
            skipped_dwords: skipped,
            _kernargs: kernargs,
        })
    }

    /// Submit and wait; GPU µs between the IB's timestamps.
    fn submit(&mut self, ib: &Ib) -> R<f64> {
        self.timestamps.as_mut_bytes()[..16].fill(0);
        self.completion.reset();
        let packet =
            PacketImage::pm4_indirect_buffer(ib.buffer.address(), ib.dwords, self.completion.raw())
                .map_err(err)?;
        self.queues.prepare_batches(&[vec![packet]]).map_err(err)?;
        self.queues.ring_prepared().map_err(err)?;
        self.queues
            .wait_signal(&self.completion, Duration::from_secs(30))
            .map_err(err)?;
        let b = self.timestamps.as_mut_bytes();
        let rd = |o: usize| unsafe { std::ptr::read_volatile(b.as_ptr().add(o).cast::<u64>()) };
        let (start, end) = (rd(0), rd(8));
        if start == 0 || end <= start {
            return Err(format!("bad GPU timestamps start={start} end={end}"));
        }
        Ok((end - start) as f64 * 1e6 / self.freq_hz as f64)
    }
}

struct Hip {
    hip: HipRuntime,
    add: hip_bridge::Function,
    stream: hip_bridge::Stream,
    _module: hip_bridge::Module,
}

impl Hip {
    fn setup(co: &[u8]) -> R<Self> {
        let hip = HipRuntime::load().map_err(err)?;
        hip.set_device(0).map_err(err)?;
        let module = hip.module_load_data(co).map_err(err)?;
        let add = hip.module_get_function(&module, ADD).map_err(err)?;
        let stream = hip.stream_create().map_err(err)?;
        Ok(Self {
            hip,
            add,
            stream,
            _module: module,
        })
    }

    fn read_u32(&self, b: &DeviceBuffer) -> R<u32> {
        let mut v = [0u8; 4];
        self.hip.memcpy_dtoh(&mut v, b).map_err(err)?;
        Ok(u32::from_le_bytes(v))
    }

    fn add(&self, b: &Buffers, n: u32) -> R<()> {
        let mut blob = kernarg(&[&b.one, &b.x, &b.y], n, 32);
        unsafe {
            self.hip.launch_kernel_blob(
                &self.add,
                [n / BLOCK, 1, 1],
                [BLOCK, 1, 1],
                0,
                Some(&self.stream),
                &mut blob,
            )
        }
        .map_err(err)?;
        self.hip.stream_synchronize(&self.stream).map_err(err)
    }

    /// The host-branched program: the host reads `y[0]` before every
    /// decision. Returns the final flag the device program must leave.
    fn reference(&self, b: &Buffers, shape: Shape, iters: usize, n: u32) -> R<u32> {
        let (limit, trips) = match shape {
            Shape::While => (WHILE_LIMIT, iters),
            _ => (IF_LIMIT, 1),
        };
        for _ in 0..trips {
            if self.read_u32(&b.y)? >= limit {
                return Ok(0);
            }
            self.add(b, n)?;
        }
        // IF: `add` leaves the set flag; WHILE: the last `step`'s decision.
        Ok(u32::from(
            shape != Shape::While || self.read_u32(&b.y)? < limit,
        ))
    }
}

#[derive(Default)]
struct Tally {
    trials: usize,
    set: usize,
    mismatches: usize,
    trips: Vec<usize>,
}

fn trial_inputs(rng: &mut Rng, shape: Shape, n: u32) -> (Vec<u8>, Vec<u8>) {
    let mut x: Vec<u32> = (0..n).map(|_| rng.next()).collect();
    let mut y: Vec<u32> = (0..n).map(|_| rng.next()).collect();
    if shape == Shape::While {
        x[0] = rng.range(WHILE_D);
        y[0] = rng.range(0..WHILE_Y0);
    }
    let le = |v: Vec<u32>| v.into_iter().flat_map(u32::to_le_bytes).collect();
    (le(x), le(y))
}

fn run() -> R<()> {
    let a = args()?;
    std::fs::create_dir_all(&a.out).map_err(err)?;
    let co = std::fs::read(&a.co).map_err(|e| format!("{}: {e}", a.co.display()))?;
    let (facts, object) = Facts::load(&a.receipt)?;
    if object != sha256_hex(&co) {
        return Err(format!(
            "receipt is for {object}, --co is {}",
            sha256_hex(&co)
        ));
    }
    let hip = Hip::setup(&co)?;
    let (mut pm4, identity) = Pm4::setup(&co)?;
    let bytes = a.n as usize * 4;
    let h = &hip.hip;
    let b = Buffers {
        flag: h.malloc(256).map_err(err)?,
        one: h.malloc(256).map_err(err)?,
        x: h.malloc(bytes).map_err(err)?,
        y: h.malloc(bytes).map_err(err)?,
    };
    h.memcpy_htod(&b.one, &1u32.to_le_bytes()).map_err(err)?;
    let l = Launches {
        facts: &facts,
        b: &b,
        n: a.n,
    };
    let arms = [
        ("if.kernel_guard", Shape::If, CondLowering::KernelGuard),
        ("if.cp_skip", Shape::If, CondLowering::CpSkip),
        (
            "while.kernel_guard",
            Shape::While,
            CondLowering::KernelGuard,
        ),
        ("while.cp_skip", Shape::While, CondLowering::CpSkip),
        (
            "if.ungated_negative_control",
            Shape::Ungated,
            CondLowering::KernelGuard,
        ),
    ];
    let mut ibs = Vec::new();
    for (name, shape, lowering) in arms.into_iter().filter(|(n, _, _)| n.contains(&a.arm)) {
        let p = program(&l, shape, a.iters, lowering)?;
        let ib = pm4.build(&p)?;
        let words: Vec<String> = ib.dwords_hex.iter().map(|w| format!("{w:08x}")).collect();
        std::fs::write(
            a.out.join(format!("{name}.ib.txt")),
            words.join("\n") + "\n",
        )
        .map_err(err)?;
        ibs.push((name, shape, ib));
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut rows = serde_json::Map::new();
    let mut out_y = vec![0u8; bytes];
    let mut ref_y = vec![0u8; bytes];
    for (name, shape, ib) in &ibs {
        let mut t = Tally::default();
        for _ in 0..a.trials {
            let (x, y) = trial_inputs(&mut rng, *shape, a.n);
            h.memcpy_htod(&b.x, &x).map_err(err)?;
            h.memcpy_htod(&b.y, &y).map_err(err)?;
            h.memcpy_htod(&b.flag, &0xdead_beefu32.to_le_bytes())
                .map_err(err)?;
            h.device_synchronize().map_err(err)?;
            pm4.submit(ib)?;
            h.memcpy_dtoh(&mut out_y, &b.y).map_err(err)?;
            let flag = hip.read_u32(&b.flag)?;

            h.memcpy_htod(&b.y, &y).map_err(err)?;
            let ref_shape = if *shape == Shape::Ungated {
                Shape::If
            } else {
                *shape
            };
            let ref_flag = hip.reference(&b, ref_shape, a.iters, a.n)?;
            h.memcpy_dtoh(&mut ref_y, &b.y).map_err(err)?;
            let y0 = u32::from_le_bytes(y[..4].try_into().unwrap());
            let limit = if *shape == Shape::While {
                WHILE_LIMIT
            } else {
                IF_LIMIT
            };
            t.trials += 1;
            t.set += usize::from(y0 < limit);
            if *shape == Shape::While {
                let (d, end) = (
                    u32::from_le_bytes(x[..4].try_into().unwrap()),
                    u32::from_le_bytes(ref_y[..4].try_into().unwrap()),
                );
                t.trips.push(((end.wrapping_sub(y0)) / d) as usize);
            }
            let flag_ok = *shape == Shape::Ungated || flag == ref_flag;
            t.mismatches += usize::from(out_y != ref_y || !flag_ok);
        }
        let mut hist = std::collections::BTreeMap::new();
        for k in &t.trips {
            *hist.entry(*k).or_insert(0usize) += 1;
        }
        rows.insert(
            (*name).into(),
            json!({
                "trials": t.trials, "flag_set_trials": t.set, "mismatches": t.mismatches,
                "trip_histogram": hist, "ib_dwords": ib.dwords, "cond_exec_skip_dwords": ib.skipped_dwords,
            }),
        );
        eprintln!(
            "{name}: {} trials, {} set, {} mismatches",
            t.trials, t.set, t.mismatches
        );
    }

    // Timing (indicative): the IF program with the flag held set or clear.
    let mut timing = serde_json::Map::new();
    for (y0, label) in [(0u32, "set"), (u32::MAX, "clear")] {
        let (mut x, mut y) = trial_inputs(&mut rng, Shape::If, a.n);
        // x[0] = 0 keeps y[0], hence the decision, fixed across replays.
        x[..4].fill(0);
        y[..4].copy_from_slice(&y0.to_le_bytes());
        for (name, shape, ib) in &ibs {
            if *shape != Shape::If {
                continue;
            }
            h.memcpy_htod(&b.x, &x).map_err(err)?;
            h.memcpy_htod(&b.y, &y).map_err(err)?;
            h.device_synchronize().map_err(err)?;
            let mut gpu = Vec::new();
            let started = Instant::now();
            while started.elapsed().as_secs_f64() < a.secs {
                gpu.push(pm4.submit(ib)?);
            }
            timing.insert(format!("{name}.{label}"), summary(gpu));
        }
        h.memcpy_htod(&b.y, &y).map_err(err)?;
        let mut wall = Vec::new();
        let started = Instant::now();
        while started.elapsed().as_secs_f64() < a.secs {
            let t = Instant::now();
            hip.reference(&b, Shape::If, 1, a.n)?;
            wall.push(t.elapsed().as_secs_f64() * 1e6);
        }
        timing.insert(format!("if.host_branched.{label}"), summary(wall));
    }
    let report = json!({
        "identity": identity, "co_sha256": sha256_hex(&co), "receipt": a.receipt,
        "n": a.n, "iters": a.iters, "row_policy": "unreceipted_system (every edge WaitIdle+System)",
        "correctness": rows,
        "timing_us": timing,
        "timing_note": "PM4 arms: GPU timestamps around the IB; host_branched: host wall of dtoh(y[0]) + hipModuleLaunchKernel + sync",
    });
    let path = a.out.join("cond_probe.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    println!("{}", path.display());
    let bad: Vec<_> = ibs
        .iter()
        .filter(|(name, shape, _)| {
            let m = rows[*name]["mismatches"].as_u64().unwrap();
            let clear = rows[*name]["trials"].as_u64().unwrap()
                - rows[*name]["flag_set_trials"].as_u64().unwrap();
            if *shape == Shape::Ungated {
                m != clear
            } else {
                m != 0
            }
        })
        .map(|(name, _, _)| *name)
        .collect();
    if !bad.is_empty() {
        return Err(format!("failed arms: {bad:?}"));
    }
    Ok(())
}

fn summary(mut v: Vec<f64>) -> Value {
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    json!({"samples": v.len(), "p50": q(0.5), "p10": q(0.1), "p99": q(0.99)})
}

fn main() {
    if let Err(e) = run() {
        eprintln!("cond_probe: {e}");
        std::process::exit(1);
    }
}
