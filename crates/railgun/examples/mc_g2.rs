//! Milestone MC gate: G2 on a program made only of copies (railgun design §7,
//! row MC), with the G3 replay stress, the size × alignment sweep of the copy
//! kernels against `hipMemcpyDtoD`, a table using every slot argument of
//! `railgun_copy_batch`, and the timing of the DFlash hidden scatter's ~80
//! D2D copies against one `CopyBatch` launch.
//!
//! One pinned GPU; the card's identity is asserted in-process:
//!
//! ```sh
//! ROCR_VISIBLE_DEVICES=GPU-e475645fe0200397 \
//!   cargo run --release -p railgun --features mc-g2 --example mc_g2 -- --out DIR
//! ```
//!
//! Every device image is compared byte for byte over whole allocations
//! (guard regions included) against a host simulation of the copies in
//! program order over the same poison, and against the `hipMemcpyDtoD` arm.
//! The kernel is compiled with the runtime's JIT recipe into `DIR/obj/`.
//! Results go to `DIR/g2.json`; exit status 1 on any mismatch or on a
//! negative control that is not detected exactly as predicted.
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hip_bridge::{DeviceBuffer, Function, GraphExec, HipRuntime, Stream};
use hipfire_config::{ConfigLayer, ProcessConfig, CONFIG_SCHEMA_VERSION};
use railgun::copy::{self, Launch};
use railgun::{
    Addr, Binding, CopySpec, Formula, Node, Prepared, PreparedNode, Program, Resource, ResourceId, ResourceKind, Slot, Word,
    WordId,
};
use rdna_compute::{FeatureFlags, KernelCompiler};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type R<T> = Result<T, String>;

fn err<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

struct Args {
    out: PathBuf,
    arch: String,
    bus: String,
    uuid: String,
    secs: f64,
    secs_secondary: f64,
    replays: usize,
    skip: Vec<String>,
}

fn parse_args() -> R<Args> {
    let mut a = Args {
        out: PathBuf::from("mc-g2"),
        arch: "gfx1201".into(),
        bus: "0000:13:00.0".into(),
        uuid: "GPU-e475645fe0200397".into(),
        secs: 10.0,
        secs_secondary: 10.0,
        replays: 200,
        skip: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--out" => a.out = value.into(),
            "--arch" => a.arch = value,
            "--bus" => a.bus = value,
            "--uuid" => a.uuid = value,
            "--secs" => a.secs = value.parse().map_err(err)?,
            "--secs-secondary" => a.secs_secondary = value.parse().map_err(err)?,
            "--replays" => a.replays = value.parse().map_err(err)?,
            "--skip" => a.skip.push(value),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(a)
}

// ── device ──────────────────────────────────────────────────────────────

struct Dev {
    hip: HipRuntime,
    copy: Function,
    batch: Function,
    stream: Stream,
}

fn view(addr: u64, len: u64) -> DeviceBuffer {
    // SAFETY: every address comes from a live allocation of this process and
    // the view is never freed.
    unsafe { DeviceBuffer::from_raw(addr as *mut c_void, len as usize) }
}

impl Dev {
    fn function(&self, l: &Launch) -> &Function {
        if l.symbol == copy::COPY_SYMBOL { &self.copy } else { &self.batch }
    }

    fn launch(&self, l: &Launch, stream: Option<&Stream>) -> R<()> {
        let mut blob = l.kernarg.clone();
        // SAFETY: `Launch` carries the kernel's explicit kernarg layout; its
        // pointers were resolved by `Program::prepare` from live allocations.
        unsafe { self.hip.launch_kernel_blob(self.function(l), l.grid, l.block, 0, stream, &mut blob) }.map_err(err)
    }

    fn sync(&self) -> R<()> {
        self.hip.device_synchronize().map_err(err)
    }

    fn upload_tables(&self, p: &Prepared) -> R<()> {
        for t in &p.tables {
            self.hip.memcpy_htod(&view(t.addr, t.bytes.len() as u64), &t.bytes).map_err(err)?;
        }
        Ok(())
    }

    fn dtod(&self, src: u64, dst: u64, len: u64) -> R<()> {
        // SAFETY: `prepare` checked both ranges against their bindings and
        // refused overlap.
        unsafe { self.hip.memcpy_dtod_raw(dst as *mut c_void, src as *const c_void, len as usize) }.map_err(err)
    }

    fn hip_memcpy(&self, src: u64, dst: u64, len: u64) -> R<()> {
        self.hip.memcpy_dtod_at(&view(dst, len), 0, &view(src, len), 0, len as usize).map_err(err)
    }

    /// The graph lowering: `Kernels` captures each node's copy-kernel launch,
    /// `Memcpy` a memcpy node per copy (§4.1).
    fn graph(&self, kind: Arm, p: &Prepared) -> R<GraphExec> {
        if kind == Arm::KernelGraph {
            self.upload_tables(p)?;
        }
        self.hip.stream_begin_capture(&self.stream, 0).map_err(err)?;
        let body = (|| -> R<()> {
            if kind == Arm::KernelGraph {
                for l in p.nodes.iter().filter_map(PreparedNode::launch) {
                    self.launch(&l, Some(&self.stream))?;
                }
            } else {
                for (s, d, n) in triples(p) {
                    self.hip.memcpy_dtod_async_at(&view(d, n), 0, &view(s, n), 0, n as usize, &self.stream).map_err(err)?;
                }
            }
            Ok(())
        })();
        let graph = self.hip.stream_end_capture(&self.stream).map_err(err)?;
        body?;
        let exec = self.hip.graph_instantiate(&graph).map_err(err)?;
        self.hip.graph_destroy(graph).map_err(err)?;
        Ok(exec)
    }

    fn replay(&self, g: &GraphExec) -> R<()> {
        self.hip.graph_launch(g, &self.stream).map_err(err)?;
        self.hip.stream_synchronize(&self.stream).map_err(err)
    }
}

/// How one arm lowers a prepared program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// Every copy through `hipMemcpyDtoD` (the reference).
    DtoD,
    /// Every copy through `hipMemcpy(kind = D2D)`, today's `memcpy_dtod_at`.
    HipMemcpy,
    /// Each node's copy kernel as a HIP launch.
    Kernels,
    /// Each node's copy kernel captured in a HIP graph.
    KernelGraph,
    /// A captured graph of memcpy nodes.
    MemcpyGraph,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::DtoD => "hipMemcpyDtoD",
            Arm::HipMemcpy => "hipMemcpy_d2d",
            Arm::Kernels => "kernel_launch",
            Arm::KernelGraph => "kernel_graph",
            Arm::MemcpyGraph => "memcpy_graph",
        }
    }
}

/// `(src, dst, len)` of every non-empty copy, in execution order.
fn triples(p: &Prepared) -> Vec<(u64, u64, u64)> {
    p.nodes
        .iter()
        .flat_map(|n| match n {
            PreparedNode::Copy(c) => vec![*c],
            PreparedNode::CopyBatch { copies, .. } => copies.clone(),
        })
        .filter(|c| c.dst.len > 0)
        .map(|c| (c.src_addr, c.dst_addr, c.dst.len))
        .collect()
}

fn run_arm(dev: &Dev, arm: Arm, p: &Prepared) -> R<()> {
    match arm {
        Arm::DtoD => triples(p).into_iter().try_for_each(|(s, d, n)| dev.dtod(s, d, n))?,
        Arm::HipMemcpy => triples(p).into_iter().try_for_each(|(s, d, n)| dev.hip_memcpy(s, d, n))?,
        Arm::Kernels => {
            dev.upload_tables(p)?;
            for l in p.nodes.iter().filter_map(PreparedNode::launch) {
                dev.launch(&l, None)?;
            }
        }
        Arm::KernelGraph | Arm::MemcpyGraph => {
            let g = dev.graph(arm, p)?;
            dev.replay(&g)?;
            dev.hip.graph_exec_destroy(g).map_err(err)?;
        }
    }
    dev.sync()
}

// ── memory: guarded allocations and their host simulation ────────────────

/// `[guard | size | guard]`; the resource is bound to the middle.
struct Alloc {
    buf: DeviceBuffer,
    guard: usize,
    size: usize,
}

impl Alloc {
    fn ptr(&self) -> u64 {
        self.buf.as_ptr() as u64
    }
    fn len(&self) -> usize {
        self.size + 2 * self.guard
    }
    fn binding(&self) -> Binding {
        Binding { base: self.ptr() + self.guard as u64, bytes: self.size as u64 }
    }
}

/// Allocations plus the host copy of what each holds initially.
struct Space {
    allocs: Vec<Alloc>,
    images: Vec<Vec<u8>>,
}

impl Space {
    fn new(dev: &Dev, sizes: &[usize], guard: usize, fill: impl Fn(usize, usize) -> Vec<u8>) -> R<Self> {
        let mut allocs = Vec::new();
        let mut images = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let len = size + 2 * guard;
            let buf = dev.hip.malloc(len).map_err(err)?;
            let image = fill(i, len);
            dev.hip.memcpy_htod(&buf, &image).map_err(err)?;
            allocs.push(Alloc { buf, guard, size });
            images.push(image);
        }
        Ok(Self { allocs, images })
    }

    fn bindings(&self) -> Vec<Binding> {
        self.allocs.iter().map(Alloc::binding).collect()
    }

    fn locate(&self, addr: u64, len: u64) -> R<(usize, usize)> {
        self.allocs
            .iter()
            .position(|a| addr >= a.ptr() && addr + len <= a.ptr() + a.len() as u64)
            .map(|i| (i, (addr - self.allocs[i].ptr()) as usize))
            .ok_or_else(|| format!("address {addr:#x}+{len} is in no allocation"))
    }

    /// Allocation `dst` after `copies` run in order over `start`; sources are
    /// read from the initial images (no copy here reads `dst`).
    fn expect(&self, dst: usize, start: &[u8], copies: &[(u64, u64, u64)]) -> R<Vec<u8>> {
        let mut image = start.to_vec();
        for &(s, d, n) in copies {
            let (si, so) = self.locate(s, n)?;
            let (di, off) = self.locate(d, n)?;
            if di != dst || si == dst {
                return Err(format!("copy {s:#x}->{d:#x} is not a read elsewhere / write into allocation {dst}"));
            }
            image[off..off + n as usize].copy_from_slice(&self.images[si][so..so + n as usize]);
        }
        Ok(image)
    }

    fn read(&self, dev: &Dev, i: usize) -> R<Vec<u8>> {
        let mut v = vec![0u8; self.allocs[i].len()];
        dev.hip.memcpy_dtoh(&mut v, &self.allocs[i].buf).map_err(err)?;
        Ok(v)
    }
}

/// Distinct seeds below 2^63 give distinct streams (`seed | 1` would give
/// seeds 2k and 2k+1 the same one, and two same-content allocations would
/// hide a slot mix-up).
fn xorshift(seed: u64, len: usize) -> Vec<u8> {
    let mut x = (seed << 1) | 1;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.extend_from_slice(&x.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
    out.truncate(len);
    out
}

const POISONS: [&str; 3] = ["0xA5 bytes", "f32 NaN 0x7FC00001", "xorshift bytes"];

fn poison(kind: usize, len: usize) -> Vec<u8> {
    match kind {
        0 => vec![0xA5; len],
        1 => (0..len).map(|i| [0x01, 0x00, 0xC0, 0x7F][i % 4]).collect(),
        _ => xorshift(0xD1B5_4A32_D192_ED03, len),
    }
}

/// `(mismatching bytes, first mismatching offset)`.
fn diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    assert_eq!(a.len(), b.len());
    let mut count = 0;
    let mut first = None;
    for (i, (x, y)) in a.chunks(4096).zip(b.chunks(4096)).enumerate() {
        if x != y {
            for (j, (p, q)) in x.iter().zip(y).enumerate() {
                if p != q {
                    count += 1;
                    first.get_or_insert(i * 4096 + j);
                }
            }
        }
    }
    (count, first)
}

/// Device-resident poison images for one destination allocation.
struct Poisons {
    bufs: Vec<DeviceBuffer>,
    host: Vec<Vec<u8>>,
}

impl Poisons {
    fn new(dev: &Dev, len: usize) -> R<Self> {
        let mut bufs = Vec::new();
        let mut host = Vec::new();
        for kind in 0..POISONS.len() {
            let image = poison(kind, len);
            let buf = dev.hip.malloc(len).map_err(err)?;
            dev.hip.memcpy_htod(&buf, &image).map_err(err)?;
            bufs.push(buf);
            host.push(image);
        }
        Ok(Self { bufs, host })
    }

    fn apply(&self, dev: &Dev, kind: usize, target: &Alloc) -> R<()> {
        dev.hip.memcpy_dtod(&target.buf, &self.bufs[kind], target.len()).map_err(err)?;
        dev.sync()
    }
}

// ── card, kernel ────────────────────────────────────────────────────────

fn product_config() -> ProcessConfig {
    ProcessConfig { schema_version: CONFIG_SCHEMA_VERSION, values: ConfigLayer::default() }
}

fn check_card(hip: &HipRuntime, a: &Args) -> R<Value> {
    let pinned = std::env::var("ROCR_VISIBLE_DEVICES").unwrap_or_default();
    if pinned != a.uuid {
        return Err(format!("ROCR_VISIBLE_DEVICES={pinned:?}: pin the card with ROCR_VISIBLE_DEVICES={}", a.uuid));
    }
    let count = hip.device_count().map_err(err)?;
    hip.set_device(0).map_err(err)?;
    let arch = hip.get_arch(0).map_err(err)?;
    let bus = hip.device_pci_bus_id(0).map_err(err)?;
    let uuid = hip.device_uuid(0).map_err(err)?;
    let runtime = hip.runtime_version().map_err(err)?;
    if count != 1 || arch != a.arch || !bus.eq_ignore_ascii_case(&a.bus) || uuid != a.uuid {
        return Err(format!("card mismatch: {count} devices, arch {arch}, bus {bus}, uuid {uuid}; expected 1, {}, {}, {}", a.arch, a.bus, a.uuid));
    }
    Ok(json!({ "devices": count, "arch": arch, "pci_bus": bus, "uuid": uuid, "hip_runtime": format!("{}.{}", runtime.0, runtime.1),
               "rocr_visible_devices": pinned }))
}

/// Compile `railgun_copy` exactly as the runtime JIT would (product-default
/// configuration), into `out/obj`.
fn compile(arch: &str, out: &Path) -> R<(Vec<u8>, Value)> {
    hipfire_config::install_process_config(product_config()).map_err(|_| "a process configuration is already installed")?;
    let extra = FeatureFlags::from_process_config(arch, &product_config()).hipcc_extra_flags;
    let compiler = KernelCompiler::new(arch, extra.clone()).map_err(err)?;
    let dir = out.join("obj");
    std::fs::create_dir_all(&dir).map_err(err)?;
    let src = dir.join(format!("{}.hip", copy::MODULE));
    let obj = dir.join(format!("{}.{arch}.hsaco", copy::MODULE));
    compiler.compile_to_paths(copy::MODULE, copy::SOURCE, &src, &obj).map_err(err)?;
    let bytes = std::fs::read(&obj).map_err(err)?;
    let sha = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>();
    let source_sha = Sha256::digest(copy::SOURCE.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>();
    let info = json!({ "object": obj, "object_sha256": sha, "source_sha256": source_sha, "extra_flags": extra,
                       "toolchain": compiler.toolchain_id(), "argv": compiler.jit_argv(copy::MODULE, copy::SOURCE, &src, &obj) });
    Ok((bytes, info))
}

// ── phase 1: every size × alignment against hipMemcpyDtoD ────────────────

fn sweep(dev: &Dev, failures: &mut Vec<String>) -> R<Value> {
    let pairs: Vec<(u64, u64)> = (0..16).flat_map(|s| (0..16).map(move |d| (s, d))).collect();
    let some: Vec<(u64, u64)> = [0, 1, 7, 15].iter().flat_map(|&s| [0, 3, 8, 13].map(|d| (s, d))).collect();
    let medium = [2047u64, 2048, 2049, 4095, 4096, 4097, 8191, 8192, 8193, 12345, 65535, 65536, 65537];
    let large = [(1u64 << 20) - 1, 1 << 20, (1 << 20) + 1, 3 * (1 << 20) + 7];
    // (size class, size, src alignment, dst alignment)
    let mut cases: Vec<(u8, u64, u64, u64)> = Vec::new();
    for size in 0..=1024 {
        cases.extend(pairs.iter().map(|&(s, d)| (0, size, s, d)));
    }
    for &size in &medium {
        cases.extend(pairs.iter().map(|&(s, d)| (1, size, s, d)));
    }
    for &size in &large {
        cases.extend(some.iter().map(|&(s, d)| (2, size, s, d)));
    }
    // Layout: sources at 64·k + sa; destinations at 64·k + da with at least
    // 64 poison bytes between consecutive destinations.
    let (mut cs, mut cd) = (0u64, 0u64);
    let mut offsets = Vec::with_capacity(cases.len());
    for &(_, size, sa, da) in &cases {
        let so = cs.next_multiple_of(64) + sa;
        let dof = (cd + 64).next_multiple_of(64) + da;
        offsets.push((so, dof));
        cs = so + size;
        cd = dof + size;
    }
    let (src_size, dst_size) = ((cs + 64) as usize, (cd + 64) as usize);
    // Tables: at most MAX_BATCH entries, one size class each.
    let mut tables: Vec<Vec<usize>> = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        match tables.last_mut() {
            Some(t) if t.len() < copy::MAX_BATCH as usize && cases[t[0]].0 == c.0 => t.push(i),
            _ => tables.push(vec![i]),
        }
    }
    const S: ResourceId = ResourceId(0);
    const D: ResourceId = ResourceId(1);
    let spec = |i: usize| CopySpec {
        src: Addr { resource: S, offset: Formula::constant(offsets[i].0) },
        dst: Addr { resource: D, offset: Formula::constant(offsets[i].1) },
        len: Formula::constant(cases[i].1),
    };
    let mut resources = vec![
        Resource { label: "sweep.src".into(), size: Formula::constant(src_size as u64), kind: ResourceKind::Scratch },
        Resource { label: "sweep.dst".into(), size: Formula::constant(dst_size as u64), kind: ResourceKind::Scratch },
    ];
    for (k, t) in tables.iter().enumerate() {
        resources.push(Resource {
            label: format!("sweep.table{k}"),
            size: Formula::constant(t.len() as u64 * copy::ENTRY_BYTES),
            kind: ResourceKind::PointerTable(t.iter().map(|&i| spec(i)).collect()),
        });
    }
    let per_copy = Program {
        arch: "gfx1201".into(),
        words: Vec::new(),
        resources: resources.clone(),
        nodes: (0..cases.len()).map(|i| Node::Copy(spec(i))).collect(),
    };
    let batched = Program {
        arch: "gfx1201".into(),
        words: Vec::new(),
        resources,
        nodes: tables
            .iter()
            .enumerate()
            .map(|(k, t)| Node::CopyBatch { table: ResourceId(2 + k as u32), n: Formula::constant(t.len() as u64) })
            .collect(),
    };
    let guard = 4096;
    let mut sizes = vec![src_size, dst_size];
    sizes.extend(tables.iter().map(|t| t.len() * copy::ENTRY_BYTES as usize));
    let space = Space::new(dev, &sizes, guard, |i, len| xorshift(0x5EED_0000 + i as u64, len))?;
    let bindings = space.bindings();
    let p_copy = per_copy.prepare(&[], &bindings).map_err(err)?;
    let p_batch = batched.prepare(&[], &bindings).map_err(err)?;
    let copies = triples(&p_copy);
    if copies != triples(&p_batch) {
        return Err("sweep: batch and per-copy programs resolve different copies".into());
    }
    let poisons = Poisons::new(dev, space.allocs[1].len())?;
    let where_is = |off: usize| -> Value {
        let at = (off as u64).saturating_sub(guard as u64);
        let i = offsets.partition_point(|&(_, d)| d <= at).saturating_sub(1);
        json!({ "alloc_offset": off, "case": { "size": cases[i].1, "src_align": cases[i].2, "dst_align": cases[i].3 } })
    };
    let arms = [Arm::DtoD, Arm::HipMemcpy, Arm::Kernels, Arm::KernelGraph];
    let mut rows = Vec::new();
    for kind in 0..POISONS.len() {
        let expected = space.expect(1, &poisons.host[kind], &copies)?;
        let mut reference: Option<Vec<u8>> = None;
        for arm in arms {
            poisons.apply(dev, kind, &space.allocs[1])?;
            let t = Instant::now();
            if arm == Arm::Kernels {
                // Zero-byte copies launch too, so the kernel sees n = 0.
                dev.upload_tables(&p_batch)?;
                for node in &p_copy.nodes {
                    let PreparedNode::Copy(c) = node else { unreachable!() };
                    dev.launch(&copy::copy_launch(c.src_addr, c.dst_addr, c.dst.len as u32), None)?;
                }
                dev.sync()?;
            } else if arm == Arm::KernelGraph {
                run_arm(dev, arm, &p_batch)?;
            } else {
                run_arm(dev, arm, &p_copy)?;
            }
            let secs = t.elapsed().as_secs_f64();
            let got = space.read(dev, 1)?;
            let (vs_expected, first) = diff(&got, &expected);
            let vs_dtod = reference.as_ref().map(|r| diff(&got, r).0);
            let arm_name = if arm == Arm::Kernels { "railgun_copy per copy" } else if arm == Arm::KernelGraph { "railgun_copy_batch graph" } else { arm.name() };
            if vs_expected != 0 || vs_dtod.unwrap_or(0) != 0 {
                failures.push(format!("sweep {arm_name} poison {kind}: {vs_expected} bytes differ from the simulation, first at {first:?}"));
            }
            rows.push(json!({ "arm": arm_name, "poison": POISONS[kind], "mismatch_vs_simulation": vs_expected,
                              "mismatch_vs_hipMemcpyDtoD": vs_dtod, "first_mismatch": first.map(&where_is), "seconds": secs }));
            if arm == Arm::DtoD {
                reference = Some(got);
            }
        }
        // The batch kernel launched directly, one launch per table.
        poisons.apply(dev, kind, &space.allocs[1])?;
        run_arm(dev, Arm::Kernels, &p_batch)?;
        let got = space.read(dev, 1)?;
        let (vs_expected, first) = diff(&got, &expected);
        let vs_dtod = diff(&got, reference.as_ref().expect("reference arm ran")).0;
        if vs_expected != 0 || vs_dtod != 0 {
            failures.push(format!("sweep railgun_copy_batch poison {kind}: {vs_expected} bytes differ, first at {first:?}"));
        }
        rows.push(json!({ "arm": "railgun_copy_batch launch", "poison": POISONS[kind], "mismatch_vs_simulation": vs_expected,
                          "mismatch_vs_hipMemcpyDtoD": vs_dtod, "first_mismatch": first.map(&where_is) }));
    }
    let src_after = space.read(dev, 0)?;
    let src_changed = diff(&src_after, &space.images[0]).0;
    if src_changed != 0 {
        failures.push(format!("sweep: {src_changed} source bytes changed"));
    }
    let written: u64 = copies.iter().map(|c| c.2).sum();
    Ok(json!({
        "cases": cases.len(), "sizes": "0..=1024 × 16×16 alignments; 13 sizes 2047..65537 × 16×16; 4 sizes 1 MiB-1..3 MiB+7 × 16",
        "medium_sizes": medium, "large_sizes": large, "tables": tables.len(),
        "dst_allocation_bytes": space.allocs[1].len(), "bytes_written": written,
        "bytes_checked_unwritten": space.allocs[1].len() as u64 - written, "source_bytes_changed": src_changed, "runs": rows,
    }))
}

// ── phase 1b: every slot argument of railgun_copy_batch ─────────────────

/// One table naming 8 source and 8 destination resources, one entry per
/// (source, destination) pair with varied sizes and alignments, so every
/// slot argument of `railgun_copy_batch` is read or written. Each run checks
/// all 8 destination allocations (guards included) against the host
/// simulation and the `hipMemcpyDtoD` arm, under the 3 poisons.
fn slots(dev: &Dev, failures: &mut Vec<String>) -> R<Value> {
    let (ns, nd) = (copy::SRC_SLOTS as u32, copy::DST_SLOTS as u32);
    let size = 64 * 1024u64;
    // Entry (i, j): source i at 4096·j + i, destination j at 8192·i + 3·j,
    // 1000 + 61·(8i + j) bytes (≤ 4843, so destinations stay disjoint).
    let entries: Vec<CopySpec> = (0..ns)
        .flat_map(|i| (0..nd).map(move |j| (i, j)))
        .map(|(i, j)| CopySpec {
            src: Addr { resource: ResourceId(i), offset: Formula::constant(4096 * u64::from(j) + u64::from(i)) },
            dst: Addr { resource: ResourceId(ns + j), offset: Formula::constant(8192 * u64::from(i) + 3 * u64::from(j)) },
            len: Formula::constant(1000 + 61 * u64::from(8 * i + j)),
        })
        .collect();
    let n = entries.len() as u64;
    let table = ResourceId(ns + nd);
    let mut resources: Vec<Resource> = (0..ns + nd)
        .map(|k| Resource { label: format!("slots.r{k}"), size: Formula::constant(size), kind: ResourceKind::Scratch })
        .collect();
    resources.push(Resource { label: "slots.table".into(), size: Formula::constant(n * copy::ENTRY_BYTES), kind: ResourceKind::PointerTable(entries) });
    let program = Program { arch: "gfx1201".into(), words: Vec::new(), resources, nodes: vec![Node::CopyBatch { table, n: Formula::constant(n) }] };
    let mut sizes = vec![size as usize; (ns + nd) as usize];
    sizes.push((n * copy::ENTRY_BYTES) as usize);
    let guard = 4096;
    let space = Space::new(dev, &sizes, guard, |i, len| xorshift(0x5107_0000 + i as u64, len))?;
    let p = program.prepare(&[], &space.bindings()).map_err(err)?;
    let PreparedNode::CopyBatch { src_slots, dst_slots, .. } = &p.nodes[0] else { unreachable!() };
    if (src_slots.len(), dst_slots.len()) != (copy::SRC_SLOTS, copy::DST_SLOTS) {
        return Err(format!("slots: {} source / {} destination slots bound", src_slots.len(), dst_slots.len()));
    }
    let copies = triples(&p);
    let dsts: Vec<usize> = (ns..ns + nd).map(|k| k as usize).collect();
    let per_dst = |d: usize| -> R<Vec<(u64, u64, u64)>> {
        let mut out = Vec::new();
        for &c in &copies {
            if space.locate(c.1, c.2)?.0 == d {
                out.push(c);
            }
        }
        Ok(out)
    };
    let poisons = Poisons::new(dev, space.allocs[dsts[0]].len())?;
    let mut rows = Vec::new();
    for kind in 0..POISONS.len() {
        let mut reference: Option<Vec<Vec<u8>>> = None;
        for (arm, name) in [(Arm::DtoD, "hipMemcpyDtoD"), (Arm::Kernels, "railgun_copy_batch launch"), (Arm::KernelGraph, "railgun_copy_batch graph")] {
            for &d in &dsts {
                poisons.apply(dev, kind, &space.allocs[d])?;
            }
            run_arm(dev, arm, &p)?;
            let (mut vs_expected, mut vs_dtod, mut first) = (0, 0, None);
            let mut images = Vec::new();
            for (j, &d) in dsts.iter().enumerate() {
                let got = space.read(dev, d)?;
                let (count, at) = diff(&got, &space.expect(d, &poisons.host[kind], &per_dst(d)?)?);
                vs_expected += count;
                first = first.or(at.map(|at| json!({ "dst_slot": j, "alloc_offset": at })));
                if let Some(r) = &reference {
                    vs_dtod += diff(&got, &r[j]).0;
                }
                images.push(got);
            }
            if vs_expected != 0 || vs_dtod != 0 {
                failures.push(format!("slots {name} poison {kind}: {vs_expected} bytes differ from the simulation, {vs_dtod} from hipMemcpyDtoD, first {first:?}"));
            }
            rows.push(json!({ "arm": name, "poison": POISONS[kind], "mismatch_vs_simulation": vs_expected,
                              "mismatch_vs_hipMemcpyDtoD": reference.as_ref().map(|_| vs_dtod), "first_mismatch": first }));
            if arm == Arm::DtoD {
                reference = Some(images);
            }
        }
    }
    let mut sources_changed = 0;
    for k in 0..ns as usize {
        sources_changed += diff(&space.read(dev, k)?, &space.images[k]).0;
    }
    let t = table.0 as usize;
    let table_got = space.read(dev, t)?;
    let table_changed = diff(&table_got[guard..guard + p.tables[0].bytes.len()], &p.tables[0].bytes).0
        + diff(&table_got[..guard], &space.images[t][..guard]).0
        + diff(&table_got[guard + sizes[t]..], &space.images[t][guard + sizes[t]..]).0;
    if sources_changed != 0 || table_changed != 0 {
        failures.push(format!("slots: {sources_changed} source bytes and {table_changed} table bytes changed"));
    }
    let written: u64 = copies.iter().map(|c| c.2).sum();
    Ok(json!({ "entries": n, "source_slots": ns, "destination_slots": nd, "bytes_written": written,
               "destination_bytes_checked": dsts.len() * space.allocs[dsts[0]].len(),
               "source_bytes_changed": sources_changed, "table_bytes_changed": table_changed, "runs": rows }))
}

// ── phase 2: the DFlash hidden scatter as a copies-only program ─────────

const NE: u64 = 5; // extract layers
const HIDDEN: u64 = 5120; // qwen3.8-27b hidden size
const ROW: u64 = HIDDEN * 4; // one f32 row
const BLOCK: u64 = 16; // DFlash B
const MAX_POS: u64 = 256; // ring slots
const MAX_CTX: u64 = 256; // target_hidden rows
const GUARD: usize = 64 * 1024;
const RINGS: u32 = NE as u32; // resources 0..5 are the rings
const TARGET: ResourceId = ResourceId(RINGS);
const TABLE: ResourceId = ResourceId(RINGS + 1);
const POSITION: WordId = WordId(0);
const ROWS: WordId = WordId(1);
fn slot(r: u64) -> WordId {
    WordId(2 + r as u32)
}

/// `scatter_hidden_block_to_interleaved` (`speculative.rs:3874-3951`): row
/// `r` of the last block, ring slot `slot_r` of extract layer `ext`, lands at
/// `target_hidden[position + r, ext, :]`. Entries are row-major so the first
/// `5·rows` are the first `rows` rows.
fn scatter_entries() -> Vec<CopySpec> {
    (0..BLOCK)
        .flat_map(|r| {
            (0..NE).map(move |ext| CopySpec {
                src: Addr { resource: ResourceId(ext as u32), offset: Formula::word(slot(r), ROW) },
                dst: Addr { resource: TARGET, offset: Formula::word(POSITION, NE * ROW).plus((r * NE + ext) * ROW) },
                len: Formula::constant(ROW),
            })
        })
        .collect()
}

fn scatter_program(nodes: Vec<Node>) -> Program {
    let mut resources: Vec<Resource> = (0..NE)
        .map(|e| Resource { label: format!("hidden_rb.layer_bufs[{e}]"), size: Formula::constant(MAX_POS * ROW), kind: ResourceKind::HiddenRing })
        .collect();
    resources.push(Resource { label: "draft_scratch.target_hidden".into(), size: Formula::constant(MAX_CTX * NE * ROW), kind: ResourceKind::DraftScratch });
    resources.push(Resource {
        label: "scatter.table".into(),
        size: Formula::constant(NE * BLOCK * copy::ENTRY_BYTES),
        kind: ResourceKind::PointerTable(scatter_entries()),
    });
    let mut words = vec![
        Word { name: "position".into(), domain: 0..=MAX_CTX - BLOCK },
        Word { name: "rows".into(), domain: 1..=BLOCK },
    ];
    words.extend((0..BLOCK).map(|r| Word { name: format!("slot_{r}"), domain: 0..=MAX_POS - 1 }));
    Program { arch: "gfx1201".into(), words, resources, nodes }
}

/// The batch program (one `CopyBatch` over `5·rows` entries) and the
/// per-copy program (today's loop: one `Copy` per row and layer).
fn scatter_programs(rows: u64) -> (Program, Program) {
    let batch = scatter_program(vec![Node::CopyBatch { table: TABLE, n: Formula::word(ROWS, NE) }]);
    let per_copy = scatter_program(scatter_entries().into_iter().take((NE * rows) as usize).map(Node::Copy).collect());
    (batch, per_copy)
}

struct Assignment {
    name: &'static str,
    head: u64,
    position: u64,
    rows: u64,
}

impl Assignment {
    /// Word values: ring slots as the scatter computes them from `head`.
    fn words(&self) -> Vec<u64> {
        let start = (self.head + MAX_POS - BLOCK) % MAX_POS;
        let mut w = vec![self.position, self.rows];
        w.extend((0..BLOCK).map(|r| (start + r) % MAX_POS));
        w
    }
}

const ASSIGNMENTS: [Assignment; 3] = [
    Assignment { name: "contiguous", head: 100, position: 37, rows: 16 },
    Assignment { name: "ring_wrap_at_resource_end", head: 5, position: MAX_CTX - BLOCK, rows: 16 },
    Assignment { name: "partial_rows_7", head: 130, position: 0, rows: 7 },
];

struct Scatter {
    space: Space,
    poisons: Poisons,
}

impl Scatter {
    fn new(dev: &Dev) -> R<Self> {
        let mut sizes = vec![(MAX_POS * ROW) as usize; NE as usize];
        sizes.push((MAX_CTX * NE * ROW) as usize);
        sizes.push((NE * BLOCK * copy::ENTRY_BYTES) as usize);
        let space = Space::new(dev, &sizes, GUARD, |i, len| xorshift(0xF1A5_0000 + i as u64, len))?;
        let poisons = Poisons::new(dev, space.allocs[TARGET.0 as usize].len())?;
        Ok(Self { space, poisons })
    }

    fn target(&self) -> &Alloc {
        &self.space.allocs[TARGET.0 as usize]
    }

    /// Poison the target, run `f`, read the target back.
    fn run(&self, dev: &Dev, kind: usize, f: impl FnOnce() -> R<()>) -> R<Vec<u8>> {
        self.poisons.apply(dev, kind, self.target())?;
        f()?;
        dev.sync()?;
        self.space.read(dev, TARGET.0 as usize)
    }

    /// Bytes of the table allocation that differ from `image` (the live
    /// entries) or from the initial guard fill. Entries past `image` belong
    /// to earlier, longer uploads and are not read.
    fn table_intact(&self, dev: &Dev, image: &[u8]) -> R<usize> {
        let got = self.space.read(dev, TABLE.0 as usize)?;
        let init = &self.space.images[TABLE.0 as usize];
        let end = GUARD + (NE * BLOCK * copy::ENTRY_BYTES) as usize;
        Ok(diff(&got[..GUARD], &init[..GUARD]).0
            + diff(&got[GUARD..GUARD + image.len()], image).0
            + diff(&got[end..], &init[end..]).0)
    }
}

fn scatter_g2(dev: &Dev, sc: &Scatter, failures: &mut Vec<String>) -> R<Value> {
    let bindings = sc.space.bindings();
    let mut out = Vec::new();
    for a in &ASSIGNMENTS {
        let (batch, per_copy) = scatter_programs(a.rows);
        let words = a.words();
        let p_batch = batch.prepare(&words, &bindings).map_err(err)?;
        let p_copy = per_copy.prepare(&words, &bindings).map_err(err)?;
        let copies = triples(&p_copy);
        if copies != triples(&p_batch) {
            return Err(format!("{}: the two programs resolve different copies", a.name));
        }
        let arms: [(Arm, &Prepared, &str); 6] = [
            (Arm::DtoD, &p_copy, "per-copy program"),
            (Arm::HipMemcpy, &p_copy, "per-copy program"),
            (Arm::Kernels, &p_copy, "per-copy program: railgun_copy launches"),
            (Arm::MemcpyGraph, &p_copy, "per-copy program: graph of memcpy nodes"),
            (Arm::Kernels, &p_batch, "CopyBatch program: one railgun_copy_batch launch"),
            (Arm::KernelGraph, &p_batch, "CopyBatch program: captured graph"),
        ];
        let mut runs = Vec::new();
        for kind in 0..POISONS.len() {
            let expected = sc.space.expect(TARGET.0 as usize, &sc.poisons.host[kind], &copies)?;
            let mut reference: Option<Vec<u8>> = None;
            for (arm, p, what) in arms {
                let got = sc.run(dev, kind, || run_arm(dev, arm, p))?;
                let (vs_expected, first) = diff(&got, &expected);
                let vs_dtod = reference.as_ref().map(|r| diff(&got, r).0);
                let table = if std::ptr::eq(p, &p_batch) { Some(sc.table_intact(dev, &p_batch.tables[0].bytes)?) } else { None };
                if vs_expected != 0 || vs_dtod.unwrap_or(0) != 0 || table.unwrap_or(0) != 0 {
                    failures.push(format!("scatter {} {} ({what}) poison {kind}: {vs_expected} bytes differ, first at {first:?}, table changed {table:?}", a.name, arm.name()));
                }
                runs.push(json!({ "arm": arm.name(), "program": what, "poison": POISONS[kind], "mismatch_vs_simulation": vs_expected,
                                  "mismatch_vs_hipMemcpyDtoD": vs_dtod, "first_mismatch": first, "table_bytes_changed": table }));
                if arm == Arm::DtoD {
                    reference = Some(got);
                }
            }
        }
        let written: u64 = copies.iter().map(|c| c.2).sum();
        out.push(json!({
            "assignment": a.name, "head": a.head, "position": a.position, "rows": a.rows, "words": words, "copies": copies.len(),
            "bytes_written": written, "target_allocation_bytes": sc.target().len(),
            "bytes_checked_unwritten": sc.target().len() as u64 - written,
            "effects": p_batch.nodes[0].effects().len(), "launch": p_batch.nodes[0].launch().map(|l| json!({ "grid": l.grid, "block": l.block })),
            "runs": runs,
        }));
    }
    let mut sources_changed = 0;
    for i in 0..NE as usize {
        sources_changed += diff(&sc.space.read(dev, i)?, &sc.space.images[i]).0;
    }
    if sources_changed != 0 {
        failures.push(format!("scatter: {sources_changed} ring bytes changed"));
    }
    Ok(json!({ "shape": { "extract_layers": NE, "hidden": HIDDEN, "row_bytes": ROW, "block": BLOCK, "ring_slots": MAX_POS,
                          "target_rows": MAX_CTX, "guard_bytes_each_side": GUARD },
               "assignments": out, "ring_bytes_changed": sources_changed }))
}

fn decode(image: &[u8]) -> Vec<copy::CopyEntry> {
    image
        .chunks(copy::ENTRY_BYTES as usize)
        .map(|e| {
            let w = |i: usize| u32::from_le_bytes(e[4 * i..4 * i + 4].try_into().unwrap());
            copy::CopyEntry { src_slot: w(0), src_offset: w(1), dst_slot: w(2), dst_offset: w(3), bytes: w(4) }
        })
        .collect()
}

/// The `(src, dst, len)` the kernel copies for `entries`: each through the
/// launch's slot bases, in table order. An entry naming a slot outside the
/// kernel's eight copies nothing.
fn resolve(entries: &[copy::CopyEntry], src: &[Slot], dst: &[Slot]) -> Vec<(u64, u64, u64)> {
    entries
        .iter()
        .filter(|e| (e.src_slot as usize) < copy::SRC_SLOTS && (e.dst_slot as usize) < copy::DST_SLOTS)
        .map(|e| {
            let at = |slots: &[Slot], k: u32, off: u32| slots[k as usize].base + u64::from(off);
            (at(src, e.src_slot, e.src_offset), at(dst, e.dst_slot, e.dst_offset), u64::from(e.bytes))
        })
        .collect()
}

/// Negative controls: a corrupted table (bypassing `prepare`) must change
/// the target exactly as the host simulation of the corrupted table says.
fn negative(dev: &Dev, sc: &Scatter, failures: &mut Vec<String>) -> R<Value> {
    let a = &ASSIGNMENTS[0];
    let (batch, _) = scatter_programs(a.rows);
    let p = batch.prepare(&a.words(), &sc.space.bindings()).map_err(err)?;
    let PreparedNode::CopyBatch { src_slots, dst_slots, .. } = &p.nodes[0] else { unreachable!() };
    let copies = triples(&p);
    if resolve(&decode(&p.tables[0].bytes), src_slots, dst_slots) != copies {
        return Err("negative: the prepared table image does not resolve to the program's copies".into());
    }
    let target = sc.target().binding();
    let victim = 17usize;
    let cases: [(&str, Box<dyn Fn(&mut copy::CopyEntry)>); 4] = [
        ("entry 17 one byte short", Box::new(|e| e.bytes -= 1)),
        ("entry 17 redirected into the trailing guard", Box::new(move |e| e.dst_offset = (target.bytes + 4096) as u32)),
        ("entry 17 reads the next ring (source slot + 1)", Box::new(|e| e.src_slot = (e.src_slot + 1) % NE as u32)),
        ("entry 17 names source slot 8 (outside the kernel's slots)", Box::new(|e| e.src_slot = copy::SRC_SLOTS as u32)),
    ];
    let mut out = Vec::new();
    for (name, corrupt) in &cases {
        let mut entries = decode(&p.tables[0].bytes);
        corrupt(&mut entries[victim]);
        let mut image = Vec::new();
        entries.iter().for_each(|e| e.encode(&mut image));
        let bad = resolve(&entries, src_slots, dst_slots);
        let launch = p.nodes[0].launch().expect("non-empty batch");
        let mut rows = Vec::new();
        for kind in 0..POISONS.len() {
            let expected = sc.space.expect(TARGET.0 as usize, &sc.poisons.host[kind], &copies)?;
            let predicted = sc.space.expect(TARGET.0 as usize, &sc.poisons.host[kind], &bad)?;
            let got = sc.run(dev, kind, || {
                dev.hip.memcpy_htod(&view(p.tables[0].addr, image.len() as u64), &image).map_err(err)?;
                dev.launch(&launch, None)
            })?;
            let detected = diff(&got, &expected).0;
            let predicted_count = diff(&predicted, &expected).0;
            let vs_predicted = diff(&got, &predicted).0;
            if detected == 0 || detected != predicted_count || vs_predicted != 0 {
                failures.push(format!("negative control '{name}' poison {kind}: detected {detected}, predicted {predicted_count}, device vs prediction {vs_predicted}"));
            }
            rows.push(json!({ "poison": POISONS[kind], "mismatch_vs_correct": detected, "predicted_mismatch": predicted_count,
                              "device_vs_predicted": vs_predicted }));
        }
        out.push(json!({ "case": name, "runs": rows }));
    }
    Ok(json!(out))
}

/// G3: replay the batch program `replays` times per lowering, every replay
/// over a fresh poison, every byte checked.
fn stress(dev: &Dev, sc: &Scatter, replays: usize, failures: &mut Vec<String>) -> R<Value> {
    let a = &ASSIGNMENTS[1];
    let (batch, per_copy) = scatter_programs(a.rows);
    let bindings = sc.space.bindings();
    let p = batch.prepare(&a.words(), &bindings).map_err(err)?;
    let p_copy = per_copy.prepare(&a.words(), &bindings).map_err(err)?;
    let copies = triples(&p);
    let expected: Vec<Vec<u8>> =
        (0..POISONS.len()).map(|k| sc.space.expect(TARGET.0 as usize, &sc.poisons.host[k], &copies)).collect::<R<_>>()?;
    let graph = dev.graph(Arm::KernelGraph, &p)?;
    let memcpy_graph = dev.graph(Arm::MemcpyGraph, &p_copy)?;
    let launch = p.nodes[0].launch().expect("non-empty batch");
    let mut out = Vec::new();
    for (name, f) in [
        ("CopyBatch captured graph", Box::new(|| dev.replay(&graph)) as Box<dyn Fn() -> R<()>>),
        ("CopyBatch HIP launch", Box::new(|| dev.launch(&launch, None))),
        ("per-copy memcpy graph", Box::new(|| dev.replay(&memcpy_graph))),
    ] {
        let mut identical = 0;
        let mut bad = Vec::new();
        for i in 0..replays {
            let got = sc.run(dev, i % POISONS.len(), &f)?;
            let (n, first) = diff(&got, &expected[i % POISONS.len()]);
            if n == 0 { identical += 1 } else { bad.push(json!({ "replay": i, "mismatch": n, "first": first })) }
        }
        if identical != replays {
            failures.push(format!("stress {name}: {identical}/{replays} replays identical"));
        }
        out.push(json!({ "lowering": name, "replays": replays, "identical": identical, "failed": bad }));
    }
    let table = sc.table_intact(dev, &p.tables[0].bytes)?;
    if table != 0 {
        failures.push(format!("stress: {table} table bytes changed"));
    }
    dev.hip.graph_exec_destroy(graph).map_err(err)?;
    dev.hip.graph_exec_destroy(memcpy_graph).map_err(err)?;
    Ok(json!({ "assignment": a.name, "runs": out, "table_bytes_changed": table }))
}

// ── phase 3: timing, ABBA ───────────────────────────────────────────────

fn stats(v: &mut [f64]) -> Value {
    v.sort_by(f64::total_cmp);
    let q = |f: f64| v[((v.len() - 1) as f64 * f).round() as usize];
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    json!({ "n": v.len(), "mean_us": mean, "median_us": q(0.5), "p10_us": q(0.1), "p90_us": q(0.9), "min_us": v[0], "max_us": v[v.len() - 1] })
}

/// One timed segment: cycles of `issue` then a device sync, for at least
/// `secs`. Returns per-cycle (enqueue, enqueue+sync) microseconds.
fn segment(dev: &Dev, secs: f64, issue: &mut dyn FnMut() -> R<()>) -> R<(Vec<f64>, Vec<f64>)> {
    for _ in 0..50 {
        issue()?;
        dev.sync()?;
    }
    let (mut enq, mut total) = (Vec::new(), Vec::new());
    let start = Instant::now();
    while start.elapsed().as_secs_f64() < secs {
        let t0 = Instant::now();
        issue()?;
        let t1 = Instant::now();
        dev.sync()?;
        let t2 = Instant::now();
        enq.push((t1 - t0).as_secs_f64() * 1e6);
        total.push((t2 - t0).as_secs_f64() * 1e6);
    }
    Ok((enq, total))
}

fn timing(dev: &Dev, sc: &Scatter, secs: f64, secs_secondary: f64, failures: &mut Vec<String>) -> R<Value> {
    let a = &ASSIGNMENTS[0];
    let (batch, per_copy) = scatter_programs(a.rows);
    let bindings = sc.space.bindings();
    let p = batch.prepare(&a.words(), &bindings).map_err(err)?;
    let p_copy = per_copy.prepare(&a.words(), &bindings).map_err(err)?;
    let copies = triples(&p_copy);
    dev.upload_tables(&p)?;
    let launch = p.nodes[0].launch().expect("non-empty batch");
    let mut batch_blob = launch.kernarg.clone();
    let copy_launches: Vec<Launch> = p_copy.nodes.iter().filter_map(PreparedNode::launch).collect();
    let mut copy_blobs: Vec<Vec<u8>> = copy_launches.iter().map(|l| l.kernarg.clone()).collect();
    let graph = dev.graph(Arm::KernelGraph, &p)?;
    let memcpy_graph = dev.graph(Arm::MemcpyGraph, &p_copy)?;
    let table = &p.tables[0];
    let table_view = view(table.addr, table.bytes.len() as u64);
    let PreparedNode::CopyBatch { src_slots, dst_slots, .. } = &p.nodes[0] else { unreachable!() };
    let bases = |slots: &[Slot]| slots.iter().map(|s| s.base).collect::<Vec<_>>();
    let floor = copy::batch_launch(table.addr, 1, &bases(src_slots), &bases(dst_slots), ROW);
    let mut floor_blob = floor.kernarg.clone();
    sc.poisons.apply(dev, 0, sc.target())?;

    let mut arms: Vec<(&str, Box<dyn FnMut() -> R<()> + '_>)> = vec![
        ("80 x hipMemcpyDtoD", Box::new(|| copies.iter().try_for_each(|&(s, d, n)| dev.dtod(s, d, n)))),
        ("1 x CopyBatch launch", Box::new(|| unsafe {
            dev.hip.launch_kernel_blob(&dev.batch, launch.grid, launch.block, 0, None, &mut batch_blob).map_err(err)
        })),
        ("80 x hipMemcpy D2D (memcpy_dtod_at)", Box::new(|| copies.iter().try_for_each(|&(s, d, n)| dev.hip_memcpy(s, d, n)))),
        ("80 x railgun_copy launch", Box::new(|| {
            copy_launches.iter().zip(copy_blobs.iter_mut()).try_for_each(|(l, b)| unsafe {
                dev.hip.launch_kernel_blob(&dev.copy, l.grid, l.block, 0, None, b).map_err(err)
            })
        })),
        ("graph of 80 memcpy nodes", Box::new(|| dev.hip.graph_launch(&memcpy_graph, &dev.stream).map_err(err))),
        ("CopyBatch graph", Box::new(|| dev.hip.graph_launch(&graph, &dev.stream).map_err(err))),
        ("table upload (1920 B H2D) + CopyBatch launch", Box::new(|| {
            dev.hip.memcpy_htod(&table_view, &table.bytes).map_err(err)?;
            let mut blob = launch.kernarg.clone();
            unsafe { dev.hip.launch_kernel_blob(&dev.batch, launch.grid, launch.block, 0, None, &mut blob) }.map_err(err)
        })),
        ("1 x CopyBatch launch, first entry only (launch + sync floor)", Box::new(|| unsafe {
            dev.hip.launch_kernel_blob(&dev.batch, floor.grid, floor.block, 0, None, &mut floor_blob).map_err(err)
        })),
    ];
    // Primary: A B B A. Secondary: C D E F G H H G F E D C.
    let primary = [0usize, 1, 1, 0];
    let secondary = [2usize, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2];
    let mut per_arm: Vec<(Vec<f64>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); arms.len()];
    let mut segments = Vec::new();
    let plan: Vec<(usize, f64)> =
        primary.iter().map(|&i| (i, secs)).chain(secondary.iter().map(|&i| (i, secs_secondary))).collect();
    for (order, &(i, s)) in plan.iter().enumerate() {
        let (mut enq, mut total) = segment(dev, s, &mut arms[i].1)?;
        let seg = json!({ "order": order, "arm": arms[i].0, "seconds": s, "total": stats(&mut total), "enqueue": stats(&mut enq) });
        eprintln!("timing {order}: {} median {:.2} us over {} cycles", arms[i].0, seg["total"]["median_us"].as_f64().unwrap_or(0.0), total.len());
        segments.push(seg);
        per_arm[i].0.extend(enq);
        per_arm[i].1.extend(total);
    }
    let pooled: Vec<Value> = arms
        .iter()
        .zip(per_arm.iter_mut())
        .map(|((name, _), (enq, total))| json!({ "arm": name, "total": stats(total), "enqueue": stats(enq) }))
        .collect();
    drop(arms);
    let expected = sc.space.expect(TARGET.0 as usize, &sc.poisons.host[0], &copies)?;
    let after = diff(&sc.space.read(dev, TARGET.0 as usize)?, &expected).0;
    if after != 0 {
        failures.push(format!("timing: target differs from the expected image by {after} bytes after all arms"));
    }
    dev.hip.graph_exec_destroy(graph).map_err(err)?;
    dev.hip.graph_exec_destroy(memcpy_graph).map_err(err)?;
    Ok(json!({ "assignment": a.name, "copies": copies.len(), "bytes_per_cycle": copies.iter().map(|c| c.2).sum::<u64>(),
               "unit": "one cycle = issue + hipDeviceSynchronize, host wall clock", "batch_launch": { "grid": launch.grid, "block": launch.block },
               "primary_order": "ABBA", "secondary_order": "CDEFGHHGFEDC", "segments": segments, "pooled": pooled,
               "target_mismatch_after": after }))
}

fn main() {
    let result = run();
    match result {
        Ok(ok) if ok => {}
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("mc_g2: {e}");
            std::process::exit(2);
        }
    }
}

fn run() -> R<bool> {
    let a = parse_args()?;
    std::fs::create_dir_all(&a.out).map_err(err)?;
    let hip = HipRuntime::load().map_err(err)?;
    let card = check_card(&hip, &a)?;
    eprintln!("card: {card}");
    let (object, kernel) = compile(&a.arch, &a.out)?;
    let module = hip.module_load_data(&object).map_err(err)?;
    let copy = hip.module_get_function(&module, copy::COPY_SYMBOL).map_err(err)?;
    let batch = hip.module_get_function(&module, copy::BATCH_SYMBOL).map_err(err)?;
    let stream = hip.stream_create().map_err(err)?;
    let dev = Dev { hip, copy, batch, stream };
    let mut failures = Vec::new();
    let mut report = json!({ "card": card, "kernel": kernel });
    let skip = |p: &str| a.skip.iter().any(|s| s == p);
    if !skip("sweep") {
        report["sweep"] = sweep(&dev, &mut failures)?;
        eprintln!("sweep done: {} failures", failures.len());
    }
    if !skip("slots") {
        report["slots"] = slots(&dev, &mut failures)?;
        eprintln!("slots done: {} failures", failures.len());
    }
    let sc = Scatter::new(&dev)?;
    if !skip("scatter") {
        report["scatter"] = scatter_g2(&dev, &sc, &mut failures)?;
        report["negative_control"] = negative(&dev, &sc, &mut failures)?;
        eprintln!("scatter done: {} failures", failures.len());
    }
    if !skip("stress") {
        report["stress"] = stress(&dev, &sc, a.replays, &mut failures)?;
        eprintln!("stress done: {} failures", failures.len());
    }
    if !skip("timing") {
        report["timing"] = timing(&dev, &sc, a.secs, a.secs_secondary, &mut failures)?;
    }
    report["failures"] = json!(failures);
    let path = a.out.join("g2.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&report).map_err(err)?).map_err(err)?;
    eprintln!("wrote {}; {} failures", path.display(), failures.len());
    for f in &failures {
        eprintln!("FAIL {f}");
    }
    Ok(failures.is_empty())
}
