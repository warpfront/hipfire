// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! PM decode twins: GPU whole-buffer byte-identity oracle (gfx1201).
//!
//! Incumbent = the frozen hipcc recipe from `inventory.json` (source snapshot
//! SHA-checked, loaded through `Gpu::ensure_kernel_public`). Candidate = a raw
//! PM code object (`.co`/`.hxaco` bytes) loaded with `module_load_data`.
//! Inputs default to real H2 tensors captured at the actual 27B decode callsite by
//! `crates/hipfire-arch-qwen35/tests/pm_decode_twins_hw.rs`.
//!
//! Modes
//! - `full`: launch incumbent and candidate with the captured kernarg layout,
//!   captured grid/block/LDS, every pointer argument re-backed by a fresh copy
//!   of the captured pre-launch bytes framed by guard bytes. Compares every
//!   argument buffer (outputs, RMW state, unchanged inputs) plus guards, and
//!   the incumbent against the captured callsite output. `--edges` adds
//!   admitted shapes derived from the same real tensors: K = 1..9 groups,
//!   real K-256 and K (quad/tail 0..3), and row counts 1/2/3/31/32/33/65 or
//!   per-matrix routing patterns, at the unchanged launcher grid rule.
//! - `region`: an author's G0 micro ABI. The reference is the incumbent source
//!   itself with exactly one insertion: before the reduction (`pre`) or before
//!   the `tid == 0` epilogue (`post`) every lane stores its value and returns.
//!   Its lane 0 (post) or an exact CPU emulation of the incumbent's shfl_down
//!   tree over its lanes (pre) is checked against the UNMODIFIED incumbent.
//!   Cases sweep every captured matrix x row x admitted group start.
//! - `selftest`: hipcc-vs-hipcc (`full` with the inventory hipcc object as the
//!   candidate) must be 0 mismatches; a perturbed weight byte and a perturbed
//!   output byte must both be detected.
//! - `time --time N [--rounds R] --module M --capture DIR --candidate CO
//!   --symbol S`: one checked warm launch per arm, then R (default 5) ABBA
//!   rounds of N launches per block. HIP events measure each launch on the
//!   default stream; framed PRE output buffers are restored outside each event
//!   interval, including RMW residual state. All buffers and guards are checked
//!   independently against hash-checked captured POST bytes before/after sampling.
//!   `timing.json` (numeric schema 1) accompanies `report.json` and its result.
//!   A factor is reported only when median separation exceeds the larger full
//!   sample spread AND sample ranges are strictly disjoint; otherwise the human
//!   summary says `not separable at this N`. JSON retains the diagnostic ratio.
//!   `time ... --synthetic` instead uses metadata-sized generated inputs, without
//!   reading PRE/POST binaries; kernarg and launch geometry remain captured.
//!   Weight/token bytes are nonzero xorshift64* data; outputs start at zero.
//!   The seed is the first eight SHA256(module) bytes as LE u64 (zero becomes 1).
//!   Each pointer offset starts state = seed XOR (offset * 0x9e3779b97f4a7c15),
//!   wrapping u64 multiplication, with zero state replaced by 1. Per byte, apply
//!   shifts XOR >>12, <<25, >>27; emit (state * 0x2545f4914f6cdd1d) % 255 + 1.
//!   Synthetic verification uses incumbent warm outputs, unchanged generated
//!   inputs, and independent constant guards, both before and after sampling.
//!   Timing JSON adds `synthetic`, `synthetic_seed` (base u64 or null),
//!   `arg_sizes` ([{arg: pointer offset, bytes: payload extent}]), and
//!   `expected_reference`. Mismatch `first_index` is a framed offset including
//!   guards; `first_region` and `first_payload_index` identify payload offsets.
//!
//! Example (card lock, one visible device):
//! ```text
//! pm_decode_twins region --module fused_qkvza_hfq4g256_mq4v2 \
//!   --candidate /abs/fused_qkvza_hfq4g256_mq4v2.g0.co --symbol pm_decode_qkvza_g0 \
//!   --capture /home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-inputs/fused_qkvza_mq4g256v2/occ0
//! ```

#[path = "support/abba.rs"]
mod abba;

use abba::*;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;

const INVENTORY: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/inventory.json";
const RESULTS: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/oracle-results";
const GUARD: usize = 4096;
const GUARD_BYTE: u8 = 0xA5;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(err))
        .collect()
}

fn abs_path(s: &str, what: &str) -> Result<PathBuf> {
    let p = PathBuf::from(s);
    if !p.is_absolute() {
        return Err(format!("{what} must be an absolute path: {s}"));
    }
    Ok(p)
}

fn rd_i32(b: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn wr_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn wr_i32(b: &mut [u8], off: usize, v: i32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

// ---------------------------------------------------------------------------
// Module tables (frozen clang24 kernarg layouts, inventory.json)

#[derive(Clone, Copy, PartialEq)]
enum Arg {
    Weight { ptr: usize, m: usize },
    X { ptr: usize },
    Y { ptr: usize, m: usize },
}

impl Arg {
    fn ptr(self) -> usize {
        match self {
            Arg::Weight { ptr, .. } | Arg::X { ptr } | Arg::Y { ptr, .. } => ptr,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Pre,
    Post,
}

#[derive(Clone, Copy, PartialEq)]
enum RegionAbi {
    /// 24 B: weights (4 contiguous groups of one row), activation (1024 f32), out (32 f32).
    QkvzaQuad,
    /// The module's own full ABI, one 136 B group of row `r`, 256 f32, 32 f32 out.
    OwnAbiOneGroup,
    /// 32 B multirow ABI, M=2 K=256: same group of rows r and r+1, out 64 f32.
    MultirowPair,
    /// 16 B: 32 lane records of 48 B (row r and r+1 half-headers/nibbles + 8 f32), out 32 x [f32; 2].
    ResidualRecords,
}

enum Rename {
    Macro(&'static str),
    Literal(&'static str),
}

struct ModuleSpec {
    module: &'static str,
    symbol: &'static str,
    k: usize,
    args: &'static [Arg],
    rename: Rename,
    pre_after: &'static str,
    post_before: &'static str,
    store: &'static str,
    region: RegionAbi,
    default_stage: Stage,
}

const MODULES: &[ModuleSpec] = &[
    ModuleSpec {
        module: "fused_qkvza_hfq4g256_mq4v2",
        symbol: "fused_qkvza_mq4g256v2",
        k: 88,
        args: &[
            Arg::Weight { ptr: 0, m: 72 },
            Arg::Weight { ptr: 8, m: 76 },
            Arg::Weight { ptr: 16, m: 80 },
            Arg::Weight { ptr: 24, m: 84 },
            Arg::X { ptr: 32 },
            Arg::Y { ptr: 40, m: 72 },
            Arg::Y { ptr: 48, m: 76 },
            Arg::Y { ptr: 56, m: 80 },
            Arg::Y { ptr: 64, m: 84 },
        ],
        rename: Rename::Macro("HIPFIRE_QKVZA_KERNEL_NAME"),
        pre_after: "    float acc = (acc0 + acc1) + (acc2 + acc3);\n",
        post_before: "    if (tid == 0) {\n#if defined(HIPFIRE_QKVZA_SCALAR_PREP)",
        store: "y[row * 32 + tid] = acc;",
        region: RegionAbi::QkvzaQuad,
        default_stage: Stage::Post,
    },
    ModuleSpec {
        module: "fused_qkv_hfq4g256_mq4v2",
        symbol: "fused_qkv_mq4g256v2",
        k: 68,
        args: &[
            Arg::Weight { ptr: 0, m: 56 },
            Arg::Weight { ptr: 8, m: 60 },
            Arg::Weight { ptr: 16, m: 64 },
            Arg::X { ptr: 24 },
            Arg::Y { ptr: 32, m: 56 },
            Arg::Y { ptr: 40, m: 60 },
            Arg::Y { ptr: 48, m: 64 },
        ],
        rename: Rename::Macro("HIPFIRE_QKV_KERNEL_NAME"),
        pre_after: "    float acc = (acc0 + acc1) + (acc2 + acc3);\n",
        post_before: "#if defined(HIPFIRE_QKV_WITH_BIAS)\n    if (tid == 0)",
        store: "y[row * 32 + tid] = acc;",
        region: RegionAbi::OwnAbiOneGroup,
        default_stage: Stage::Pre,
    },
    ModuleSpec {
        module: "fused_gate_up_hfq4g256_mq4v2",
        symbol: "fused_gate_up_mq4g256v2",
        k: 48,
        args: &[
            Arg::Weight { ptr: 0, m: 40 },
            Arg::Weight { ptr: 8, m: 44 },
            Arg::X { ptr: 16 },
            Arg::Y { ptr: 24, m: 40 },
            Arg::Y { ptr: 32, m: 44 },
        ],
        rename: Rename::Macro("HIPFIRE_FUSED_GATE_UP_KERNEL"),
        pre_after: "    float acc = (acc0 + acc1) + (acc2 + acc3);\n",
        post_before: "    if (tid == 0) y[local_row] = acc;",
        store: "y[local_row * 32 + tid] = acc;",
        region: RegionAbi::OwnAbiOneGroup,
        default_stage: Stage::Post,
    },
    ModuleSpec {
        module: "gemv_hfq4g256_residual_mq4v2",
        symbol: "gemv_mq4g256v2_residual",
        k: 28,
        args: &[
            Arg::Weight { ptr: 0, m: 24 },
            Arg::X { ptr: 8 },
            Arg::Y { ptr: 16, m: 24 },
        ],
        rename: Rename::Macro("HIPFIRE_RESIDUAL_KERNEL"),
        pre_after: "    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);\n",
        post_before: "    if (tid == 0) {\n        if (have_row1) {",
        store: "y[(row0 >> 1) * 64 + tid * 2] = acc; y[(row0 >> 1) * 64 + tid * 2 + 1] = bcc;",
        region: RegionAbi::ResidualRecords,
        default_stage: Stage::Post,
    },
    ModuleSpec {
        module: "gemv_hfq4g256_multirow_default_mq4v2",
        symbol: "gemv_mq4g256v2_multirow_r2",
        k: 28,
        args: &[
            Arg::Weight { ptr: 0, m: 24 },
            Arg::X { ptr: 8 },
            Arg::Y { ptr: 16, m: 24 },
        ],
        rename: Rename::Literal("gemv_mq4g256v2_multirow_r2"),
        pre_after: "    float s1 = (b0 + b1) + (b2 + b3);\n",
        post_before: "    if (tid == 0) {\n        y[row0] = s0;",
        store: "y[row0 * 32 + tid] = s0; y[row0 * 32 + 32 + tid] = s1;",
        region: RegionAbi::MultirowPair,
        default_stage: Stage::Pre,
    },
];

fn spec(module: &str) -> Result<&'static ModuleSpec> {
    MODULES.iter().find(|m| m.module == module).ok_or_else(|| {
        format!(
            "unknown module {module}; known: {}",
            MODULES
                .iter()
                .map(|m| m.module)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

// ---------------------------------------------------------------------------
// Inventory + incumbent

struct Incumbent {
    source: String,
    source_sha256: String,
    object_path: PathBuf,
    object_sha256: String,
    kernarg_size: usize,
}

fn incumbent(inventory: &Path, s: &ModuleSpec) -> Result<Incumbent> {
    let inv: Value =
        serde_json::from_slice(&std::fs::read(inventory).map_err(err)?).map_err(err)?;
    let m = inv["modules"]
        .as_array()
        .ok_or("inventory.modules missing")?
        .iter()
        .find(|m| m["module"] == s.module)
        .ok_or_else(|| format!("{} not in inventory", s.module))?;
    let snap = m["source_snapshot"]
        .as_str()
        .ok_or("source_snapshot missing")?;
    let source = std::fs::read_to_string(snap).map_err(err)?;
    let want = m["source_sha256"].as_str().ok_or("source_sha256 missing")?;
    let got = sha256_hex(source.as_bytes());
    if got != want {
        return Err(format!("{snap}: sha256 {got} != inventory {want}"));
    }
    let flags = &m["closure"]["flags"];
    let expect_flags = json!([
        "--genco",
        "--offload-arch=gfx1201",
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
        "-DIU4_A4_CANDIDATES=2"
    ]);
    if *flags != expect_flags {
        return Err(format!("{}: unexpected recipe flags {flags}", s.module));
    }
    let contract = &m["kernel_contracts"][s.symbol];
    let kernarg_size = contract["abi"][".kernarg_segment_size"]
        .as_u64()
        .ok_or("kernarg_segment_size missing")? as usize;
    Ok(Incumbent {
        source,
        source_sha256: got,
        object_path: PathBuf::from(m["object_path"].as_str().ok_or("object_path missing")?),
        object_sha256: m["object_sha256"]
            .as_str()
            .ok_or("object_sha256 missing")?
            .to_owned(),
        kernarg_size,
    })
}

fn variant_source(s: &ModuleSpec, inc: &Incumbent, stage: Stage, name: &str) -> Result<String> {
    let src = &inc.source;
    let insert = format!("    {{ {} return; }}\n", s.store);
    let (anchor, after) = match stage {
        Stage::Pre => (s.pre_after, true),
        Stage::Post => (s.post_before, false),
    };
    let hits = src.matches(anchor).count();
    if hits != 1 {
        return Err(format!(
            "{}: {stage:?} anchor occurs {hits} times",
            s.module
        ));
    }
    let at = src.find(anchor).unwrap() + if after { anchor.len() } else { 0 };
    let mut out = String::with_capacity(src.len() + 256);
    match s.rename {
        Rename::Macro(mac) => {
            out.push_str(&format!("#define {mac} {name}\n"));
            out.push_str(&src[..at]);
            out.push_str(&insert);
            out.push_str(&src[at..]);
        }
        Rename::Literal(sym) => {
            let decl = format!("void {sym}(");
            if src.matches(&decl).count() != 1 {
                return Err(format!(
                    "{}: literal symbol declaration not unique",
                    s.module
                ));
            }
            let body = format!("{}{}{}", &src[..at], insert, &src[at..]);
            out.push_str(&body.replacen(&decl, &format!("void {name}("), 1));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Capture

struct Capture {
    dir: PathBuf,
    meta: Value,
    kernarg: Vec<u8>,
    grid: [u32; 3],
    block: [u32; 3],
    lds: u32,
}

fn load_capture(dir: &Path, s: &ModuleSpec) -> Result<Capture> {
    let meta: Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).map_err(err)?).map_err(err)?;
    if meta["schema"] != "pmdt-capture-v1" || meta["symbol"] != s.symbol {
        return Err(format!("{}: not a {} capture", dir.display(), s.symbol));
    }
    let arr3 = |k: &str| -> Result<[u32; 3]> {
        let a = meta[k].as_array().ok_or(format!("meta.{k}"))?;
        Ok([0, 1, 2].map(|i| a[i].as_u64().unwrap_or(0) as u32))
    };
    Ok(Capture {
        dir: dir.to_owned(),
        kernarg: unhex(meta["kernarg_hex"].as_str().ok_or("kernarg_hex")?)?,
        grid: arr3("grid")?,
        block: arr3("block")?,
        lds: meta["dynamic_lds"].as_u64().unwrap_or(0) as u32,
        meta,
    })
}

impl Capture {
    fn arg(&self, off: usize, tag: &str) -> Result<Vec<u8>> {
        let name = format!("arg{off}.{tag}.bin");
        let b = std::fs::read(self.dir.join(&name)).map_err(err)?;
        let want = self.meta["files"][&name]["sha256"]
            .as_str()
            .ok_or(format!("no hash for {name}"))?;
        if sha256_hex(&b) != want {
            return Err(format!("{name}: sha256 mismatch"));
        }
        Ok(b)
    }
}

// ---------------------------------------------------------------------------
// Device helpers

struct Dev {
    buf: hip_bridge::DeviceBuffer,
}

impl Dev {
    fn new(gpu: &Gpu, bytes: usize) -> Result<Self> {
        Ok(Dev {
            buf: gpu.hip.malloc(bytes.max(16)).map_err(err)?,
        })
    }
    fn upload(gpu: &Gpu, bytes: &[u8]) -> Result<Self> {
        let d = Self::new(gpu, bytes.len())?;
        if !bytes.is_empty() {
            gpu.hip.memcpy_htod(&d.buf, bytes).map_err(err)?;
        }
        Ok(d)
    }
    fn zeroed(gpu: &Gpu, bytes: usize) -> Result<Self> {
        let d = Self::new(gpu, bytes)?;
        gpu.hip.memset(&d.buf, 0, d.buf.size()).map_err(err)?;
        Ok(d)
    }
    fn va(&self, off: usize) -> u64 {
        self.buf.as_ptr() as u64 + off as u64
    }
    fn download(&self, gpu: &Gpu, bytes: usize) -> Result<Vec<u8>> {
        let mut v = vec![0u8; bytes];
        gpu.hip.memcpy_dtoh(&mut v, &self.buf).map_err(err)?;
        Ok(v)
    }
}

fn pad16(mut v: Vec<u8>) -> Vec<u8> {
    let n = (v.len() + 15) & !15;
    v.resize(n, 0);
    v
}

struct Candidate {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
    path: PathBuf,
    sha256: String,
}

fn load_candidate(gpu: &Gpu, path: &Path, symbol: &str) -> Result<Candidate> {
    let bytes = std::fs::read(path).map_err(err)?;
    let module = gpu.hip.module_load_data(&bytes).map_err(err)?;
    let func = gpu.hip.module_get_function(&module, symbol).map_err(err)?;
    Ok(Candidate {
        _module: module,
        func,
        path: path.to_owned(),
        sha256: sha256_hex(&bytes),
    })
}

fn launch_candidate(
    gpu: &Gpu,
    c: &Candidate,
    grid: [u32; 3],
    block: [u32; 3],
    lds: u32,
    kernarg: &mut [u8],
) -> Result<()> {
    // SAFETY: the kernarg bytes follow the candidate's declared ABI and every
    // pointer addresses a live allocation owned by this process.
    unsafe {
        gpu.hip
            .launch_kernel_blob(&c.func, grid, block, lds, None, kernarg)
    }
    .map_err(err)
}

// ---------------------------------------------------------------------------
// Comparison

#[derive(Default)]
struct Cmp {
    bytes: usize,
    mismatches: usize,
    first: Option<usize>,
}

fn compare(a: &[u8], b: &[u8]) -> Cmp {
    let mut c = Cmp {
        bytes: a.len().max(b.len()),
        ..Default::default()
    };
    for i in 0..c.bytes {
        if a.get(i) != b.get(i) {
            c.mismatches += 1;
            c.first.get_or_insert(i);
        }
    }
    c
}

fn window(b: &[u8], at: usize) -> String {
    let lo = at & !15;
    hex(&b[lo.min(b.len())..(lo + 32).min(b.len())])
}

fn cmp_json(name: &str, a: &[u8], b: &[u8], labels: (&str, &str)) -> (usize, Value) {
    let c = compare(a, b);
    let mut v = json!({"buffer": name, "bytes": c.bytes, "mismatch_bytes": c.mismatches});
    if let Some(i) = c.first {
        v["first_index"] = json!(i);
        v[format!("{}_hex", labels.0)] = json!(window(a, i));
        v[format!("{}_hex", labels.1)] = json!(window(b, i));
        v["window_base"] = json!(i & !15);
    }
    (c.mismatches, v)
}

/// HIP `__shfl_down(v, d)` reduction over 32 lanes (out-of-range lane reads
/// itself), offsets 16..1, exactly as the incumbent loops. Returns lane 0.
fn shfl_down_tree(lanes: &[f32; 32]) -> f32 {
    let mut v = *lanes;
    let mut d = 16;
    while d > 0 {
        let prev = v;
        for i in 0..32 {
            let src = if i + d < 32 { prev[i + d] } else { prev[i] };
            v[i] = prev[i] + src;
        }
        d >>= 1;
    }
    v[0]
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---------------------------------------------------------------------------
// Full-symbol mode

struct Arm {
    bufs: Vec<(Arg, usize, Dev)>,
    kernarg: Vec<u8>,
}

fn arg_len(s: &ModuleSpec, kernarg: &[u8], a: Arg) -> usize {
    let k = rd_i32(kernarg, s.k) as usize;
    match a {
        Arg::Weight { m, .. } => rd_i32(kernarg, m) as usize * (k / 256) * 136,
        Arg::X { .. } => k * 4,
        Arg::Y { m, .. } => rd_i32(kernarg, m) as usize * 4,
    }
}

/// One launch's complete input: kernarg template (pointer slots are re-bound),
/// pre-launch bytes for every pointer argument, and launcher geometry.
struct Launch {
    kernarg: Vec<u8>,
    args: Vec<(Arg, Vec<u8>)>,
    grid: [u32; 3],
    block: [u32; 3],
    lds: u32,
}

fn build_arm(
    gpu: &Gpu,
    s: &ModuleSpec,
    l: &Launch,
    perturb: Option<(usize, usize)>,
) -> Result<Arm> {
    build_framed_arm(gpu, s, l, perturb, true)
}

fn build_framed_arm(
    gpu: &Gpu,
    s: &ModuleSpec,
    l: &Launch,
    perturb: Option<(usize, usize)>,
    check_lengths: bool,
) -> Result<Arm> {
    let mut kernarg = l.kernarg.clone();
    let mut bufs = Vec::new();
    for (a, pre) in &l.args {
        let len = pre.len();
        if check_lengths && len != arg_len(s, &l.kernarg, *a) {
            let len = arg_len(s, &l.kernarg, *a);
            return Err(format!(
                "arg{}: {} bytes, expected {len}",
                a.ptr(),
                pre.len()
            ));
        }
        let mut framed = vec![GUARD_BYTE; GUARD + len + GUARD];
        framed[GUARD..GUARD + len].copy_from_slice(pre);
        if let Some((off, byte)) = perturb {
            if off == a.ptr() {
                framed[GUARD + byte] ^= 0x10;
            }
        }
        let d = Dev::upload(gpu, &framed)?;
        wr_u64(&mut kernarg, a.ptr(), d.va(GUARD));
        bufs.push((*a, len, d));
    }
    Ok(Arm {
        bufs,
        kernarg: pad16(kernarg),
    })
}

/// Launch incumbent and candidate on identical copies; returns (mismatch
/// bytes, per-buffer json, incumbent post bytes per arg).
fn launch_pair(
    gpu: &mut Gpu,
    s: &ModuleSpec,
    cand: &Candidate,
    l: &Launch,
    perturb: Option<(usize, usize)>,
    flip_output: bool,
) -> Result<(usize, Vec<Value>, Vec<Vec<u8>>)> {
    let mut hip = build_arm(gpu, s, l, None)?;
    let mut pm = build_arm(gpu, s, l, perturb)?;
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.launch_kernel_blob(s.symbol, l.grid, l.block, l.lds, &mut hip.kernarg)
        .map_err(err)?;
    gpu.hip.device_synchronize().map_err(err)?;
    launch_candidate(gpu, cand, l.grid, l.block, l.lds, &mut pm.kernarg)?;
    gpu.hip.device_synchronize().map_err(err)?;
    let mut total = 0;
    let mut rows = Vec::new();
    let mut posts = Vec::new();
    let mut flipped = false;
    for ((a, len, dh), (_, _, dp)) in hip.bufs.iter().zip(&pm.bufs) {
        let fh = dh.download(gpu, GUARD + len + GUARD)?;
        let mut fp = dp.download(gpu, GUARD + len + GUARD)?;
        if flip_output && !flipped && matches!(a, Arg::Y { .. }) && *len > 0 {
            fp[GUARD] ^= 0x01;
            flipped = true;
        }
        let (n, v) = cmp_json(&format!("arg{}+guards", a.ptr()), &fh, &fp, ("hipcc", "pm"));
        total += n;
        rows.push(v);
        if !fh[..GUARD]
            .iter()
            .chain(&fh[GUARD + len..])
            .all(|&b| b == GUARD_BYTE)
        {
            return Err(format!("incumbent wrote outside arg{} extent", a.ptr()));
        }
        posts.push(fh[GUARD..GUARD + len].to_vec());
    }
    Ok((total, rows, posts))
}

fn captured_launch(s: &ModuleSpec, cap: &Capture, kernarg_size: usize) -> Result<Launch> {
    if cap.kernarg.len() < kernarg_size {
        return Err(format!(
            "captured kernarg {} < segment {kernarg_size}",
            cap.kernarg.len()
        ));
    }
    let args = s
        .args
        .iter()
        .map(|&a| cap.arg(a.ptr(), "pre").map(|b| (a, b)))
        .collect::<Result<_>>()?;
    Ok(Launch {
        kernarg: cap.kernarg.clone(),
        args,
        grid: cap.grid,
        block: cap.block,
        lds: cap.lds,
    })
}

fn synthetic_seed(module: &str) -> u64 {
    let digest = Sha256::digest(module.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap()).max(1)
}

fn synthetic_bytes(seed: u64, arg: Arg, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    if !matches!(arg, Arg::Y { .. }) {
        let mut state = (seed ^ (arg.ptr() as u64).wrapping_mul(0x9e3779b97f4a7c15)).max(1);
        for byte in &mut bytes {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            *byte = (state.wrapping_mul(0x2545f4914f6cdd1d) % 255 + 1) as u8;
        }
    }
    bytes
}

fn synthetic_launch(
    s: &ModuleSpec,
    cap: &Capture,
    kernarg_size: usize,
    seed: u64,
) -> Result<Launch> {
    if cap.kernarg.len() < kernarg_size {
        return Err(format!(
            "captured kernarg {} < segment {kernarg_size}",
            cap.kernarg.len()
        ));
    }
    let args = s
        .args
        .iter()
        .map(|&arg| {
            let name = format!("arg{}.pre.bin", arg.ptr());
            let bytes = cap.meta["files"][&name]["bytes"]
                .as_u64()
                .ok_or(format!("no byte extent for {name}"))?;
            let len =
                usize::try_from(bytes).map_err(|_| format!("{name}: byte extent exceeds usize"))?;
            Ok((arg, synthetic_bytes(seed, arg, len)))
        })
        .collect::<Result<_>>()?;
    Ok(Launch {
        kernarg: cap.kernarg.clone(),
        args,
        grid: cap.grid,
        block: cap.block,
        lds: cap.lds,
    })
}

/// Retain PRE for read-only data and constant guards even if warm wrote both.
fn synthetic_warm_reference(arg: Arg, pre: &[u8], warm: &[u8]) -> Vec<u8> {
    let mut framed = vec![GUARD_BYTE; GUARD + pre.len() + GUARD];
    let payload = if matches!(arg, Arg::Y { .. }) {
        &warm[GUARD..GUARD + pre.len()]
    } else {
        pre
    };
    framed[GUARD..GUARD + pre.len()].copy_from_slice(payload);
    framed
}

fn run_full(
    gpu: &mut Gpu,
    s: &ModuleSpec,
    inc: &Incumbent,
    cap: &Capture,
    cand: &Candidate,
    perturb: Option<(usize, usize)>,
    flip_output: bool,
) -> Result<(usize, Value)> {
    let l = captured_launch(s, cap, inc.kernarg_size)?;
    let (total, rows, posts) = launch_pair(gpu, s, cand, &l, perturb, flip_output)?;
    let mut callsite_mismatch = 0;
    for ((a, _), post) in l.args.iter().zip(&posts) {
        callsite_mismatch += compare(post, &cap.arg(a.ptr(), "post")?).mismatches;
    }
    Ok((
        total,
        json!({
            "mode": "full",
            "grid": cap.grid, "block": cap.block, "dynamic_lds": cap.lds,
            "kernarg_hex": hex(&cap.kernarg),
            "incumbent_vs_captured_callsite_mismatch_bytes": callsite_mismatch,
            "buffers": rows,
            "mismatch_bytes": total,
        }),
    ))
}

/// Launcher geometry for a derived shape: one workgroup per output row for the
/// fused projections; two rows per workgroup for residual / multirow-r2.
fn edge_grid(s: &ModuleSpec, ms: &[usize]) -> [u32; 3] {
    let total: usize = ms.iter().sum();
    let x = match s.region {
        RegionAbi::ResidualRecords | RegionAbi::MultirowPair => total.div_ceil(2),
        _ => total,
    };
    [x as u32, 1, 1]
}

/// Admitted edge shapes derived from the real tensors: group counts covering
/// quads 0/1/2+ with tail 0/1/2/3 plus the real K-1 group / K, and row counts
/// 1, 2/3 (odd row-pair tails), 31/32/33 and per-matrix routing.
fn run_edges(
    gpu: &mut Gpu,
    s: &ModuleSpec,
    cap: &Capture,
    cand: &Candidate,
) -> Result<(usize, Value)> {
    let real_groups = rd_i32(&cap.kernarg, s.k) as usize / 256;
    let mut gs: Vec<usize> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, real_groups - 1, real_groups];
    gs.retain(|&g| g >= 1 && g <= real_groups);
    gs.sort_unstable();
    gs.dedup();
    let weights: Vec<(Arg, usize)> = s
        .args
        .iter()
        .filter_map(|&a| {
            if let Arg::Weight { m, .. } = a {
                Some((a, m))
            } else {
                None
            }
        })
        .collect();
    let real_rows: Vec<usize> = weights
        .iter()
        .map(|&(_, m)| rd_i32(&cap.kernarg, m) as usize)
        .collect();
    let nw = weights.len();
    let patterns: Vec<Vec<usize>> = if nw == 1 {
        [1, 2, 3, 31, 32, 33, 65].iter().map(|&m| vec![m]).collect()
    } else {
        let base: &[[usize; 4]] = &[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [31, 32, 33, 1],
            [33, 1, 2, 31],
            [3, 0, 5, 0],
            [32, 32, 32, 32],
        ];
        let mut v: Vec<Vec<usize>> = base
            .iter()
            .map(|p| p[..nw].to_vec())
            .filter(|p| p.iter().any(|&m| m > 0))
            .collect();
        v.sort();
        v.dedup();
        v
    };
    let wbytes: Vec<Vec<u8>> = weights
        .iter()
        .map(|(a, _)| cap.arg(a.ptr(), "pre"))
        .collect::<Result<_>>()?;
    let xa = s
        .args
        .iter()
        .copied()
        .find(|a| matches!(a, Arg::X { .. }))
        .unwrap();
    let x = cap.arg(xa.ptr(), "pre")?;
    let mut total = 0;
    let mut shapes = Vec::new();
    for &g in &gs {
        for pat in &patterns {
            let ms: Vec<usize> = pat
                .iter()
                .zip(&real_rows)
                .map(|(&m, &r)| m.min(r))
                .collect();
            if ms.iter().all(|&m| m == 0) {
                continue;
            }
            let mut kernarg = cap.kernarg.clone();
            wr_i32(&mut kernarg, s.k, (g * 256) as i32);
            for (&(_, moff), &m) in weights.iter().zip(&ms) {
                wr_i32(&mut kernarg, moff, m as i32);
            }
            let real_row_bytes = real_groups * 136;
            let mut args = Vec::new();
            for &a in s.args {
                let bytes = match a {
                    Arg::Weight { ptr, .. } => {
                        let wi = weights.iter().position(|(w, _)| w.ptr() == ptr).unwrap();
                        let mut b = Vec::with_capacity(ms[wi] * g * 136);
                        for r in 0..ms[wi] {
                            b.extend_from_slice(&wbytes[wi][r * real_row_bytes..][..g * 136]);
                        }
                        b
                    }
                    Arg::X { .. } => x[..g * 1024].to_vec(),
                    Arg::Y { ptr, m } => {
                        let rows = rd_i32(&kernarg, m) as usize;
                        let mut pre = cap.arg(ptr, "pre")?;
                        pre.truncate(rows * 4);
                        pre
                    }
                };
                args.push((a, bytes));
            }
            let l = Launch {
                kernarg,
                args,
                grid: edge_grid(s, &ms),
                block: cap.block,
                lds: cap.lds,
            };
            let (n, rows, _) = launch_pair(gpu, s, cand, &l, None, false)?;
            total += n;
            let mut shape =
                json!({"groups": g, "K": g * 256, "rows": ms, "grid": l.grid, "mismatch_bytes": n});
            if n > 0 {
                shape["buffers"] = json!(rows);
            }
            shapes.push(shape);
        }
    }
    Ok((
        total,
        json!({"mode": "edges", "shapes": shapes.len(), "mismatch_bytes": total, "results": shapes}),
    ))
}

// ---------------------------------------------------------------------------
// Region mode

struct Case {
    matrix: usize,
    row: usize,
    group: usize,
}

struct RegionArms {
    cand_kernargs: Vec<Vec<u8>>,
    ref_kernargs: Vec<Vec<u8>>,
    inc_kernargs: Vec<Vec<u8>>,
    out_bytes: usize,
}

const LANE_OUT: usize = 128;

#[allow(clippy::too_many_arguments)]
fn region(
    gpu: &mut Gpu,
    s: &ModuleSpec,
    inc: &Incumbent,
    cap: &Capture,
    cand: &Candidate,
    stage: Stage,
    row_stride: usize,
    report: &Path,
) -> Result<(usize, Value)> {
    let k_total = rd_i32(&cap.kernarg, s.k) as usize;
    let groups = k_total / 256;
    let x = cap.arg(
        s.args
            .iter()
            .find(|a| matches!(a, Arg::X { .. }))
            .unwrap()
            .ptr(),
        "pre",
    )?;
    let weights: Vec<(Arg, usize, Vec<u8>)> = s
        .args
        .iter()
        .filter_map(|&a| match a {
            Arg::Weight { m, .. } => Some((a, rd_i32(&cap.kernarg, m) as usize, ())),
            _ => None,
        })
        .map(|(a, rows, ())| cap.arg(a.ptr(), "pre").map(|b| (a, rows, b)))
        .collect::<Result<_>>()?;
    let row_bytes = groups * 136;
    let (span, pair) = match s.region {
        RegionAbi::QkvzaQuad => (4, false),
        RegionAbi::OwnAbiOneGroup => (1, false),
        RegionAbi::MultirowPair | RegionAbi::ResidualRecords => (1, true),
    };
    let mut cases = Vec::new();
    for (mi, (_, rows, _)) in weights.iter().enumerate() {
        let last_row = if pair { rows.saturating_sub(1) } else { *rows };
        let mut picked: Vec<usize> = (0..last_row).step_by(row_stride.max(1)).collect();
        if last_row > 0 && *picked.last().unwrap() != last_row - 1 {
            picked.push(last_row - 1);
        }
        for row in picked {
            let mut g = 0;
            while g + span <= groups {
                cases.push(Case {
                    matrix: mi,
                    row,
                    group: g,
                });
                g += span;
            }
        }
    }
    if cases.is_empty() {
        return Err("no region cases".into());
    }

    // Device inputs: full matrices + x once; per-case synthetic blocks only where the ABI needs them.
    let xd = Dev::upload(gpu, &x)?;
    let zero_x = Dev::zeroed(gpu, 1024 * 4)?;
    let wd: Vec<Dev> = weights
        .iter()
        .map(|(_, _, b)| Dev::upload(gpu, b))
        .collect::<Result<_>>()?;
    let out_bytes = match s.region {
        RegionAbi::MultirowPair | RegionAbi::ResidualRecords => 2 * LANE_OUT,
        _ => LANE_OUT,
    };
    // Synthetic per-case blocks.
    let mut cand_blk: Vec<u8> = Vec::new();
    let mut ref_blk: Vec<u8> = Vec::new();
    let mut ref_x: Vec<u8> = Vec::new();
    let (cand_stride, ref_stride, refx_stride) = match s.region {
        RegionAbi::MultirowPair => (272, 272, 0),
        RegionAbi::ResidualRecords => (32 * 48, 2 * 4 * 136, 1024 * 4),
        _ => (0, 0, 0),
    };
    if cand_stride > 0 {
        cand_blk.reserve(cases.len() * cand_stride);
        ref_blk.reserve(cases.len() * ref_stride);
        for c in &cases {
            let w = &weights[c.matrix].2;
            let g0 = &w[c.row * row_bytes + c.group * 136..][..136];
            let g1 = &w[(c.row + 1) * row_bytes + c.group * 136..][..136];
            match s.region {
                RegionAbi::MultirowPair => {
                    cand_blk.extend_from_slice(g0);
                    cand_blk.extend_from_slice(g1);
                }
                RegionAbi::ResidualRecords => {
                    let xs = &x[c.group * 1024..][..1024];
                    for tid in 0..32 {
                        let h = if tid < 16 { 0 } else { 4 };
                        cand_blk.extend_from_slice(&g0[h..h + 4]);
                        cand_blk.extend_from_slice(&g0[8 + tid * 4..12 + tid * 4]);
                        cand_blk.extend_from_slice(&g1[h..h + 4]);
                        cand_blk.extend_from_slice(&g1[8 + tid * 4..12 + tid * 4]);
                        cand_blk.extend_from_slice(&xs[tid * 32..tid * 32 + 32]);
                    }
                    // Reference: M=2, K=1024, real group in slot 0, zero groups 1..3.
                    for g in [g0, g1] {
                        ref_blk.extend_from_slice(g);
                        ref_blk.extend(std::iter::repeat(0u8).take(3 * 136));
                    }
                    ref_x.extend_from_slice(xs);
                    ref_x.extend(std::iter::repeat(0u8).take(3 * 1024));
                }
                _ => unreachable!(),
            }
        }
    }
    let cand_blk_d = Dev::upload(gpu, &cand_blk)?;
    let ref_blk_d = Dev::upload(gpu, &ref_blk)?;
    let ref_x_d = Dev::upload(gpu, &ref_x)?;
    let out_c = Dev::zeroed(gpu, cases.len() * out_bytes)?;
    let out_r = Dev::zeroed(gpu, cases.len() * out_bytes)?;
    let out_i = Dev::zeroed(gpu, cases.len() * out_bytes)?;

    let mut arms = RegionArms {
        cand_kernargs: Vec::new(),
        ref_kernargs: Vec::new(),
        inc_kernargs: Vec::new(),
        out_bytes,
    };
    for (i, c) in cases.iter().enumerate() {
        let wva = wd[c.matrix].va(c.row * row_bytes + c.group * 136);
        let xva = xd.va(c.group * 1024);
        let o = |d: &Dev| d.va(i * out_bytes);
        // Incumbent-ABI kernarg builder for the region shape.
        let own = |out: u64| -> Vec<u8> {
            let mut ka = vec![0u8; inc.kernarg_size];
            for &a in s.args {
                match a {
                    Arg::Weight { ptr, .. } => wr_u64(&mut ka, ptr, wva),
                    Arg::X { ptr } => wr_u64(&mut ka, ptr, xva),
                    Arg::Y { ptr, .. } => wr_u64(&mut ka, ptr, out),
                }
            }
            let mut first = true;
            for &a in s.args {
                if let Arg::Weight { m, .. } = a {
                    wr_i32(&mut ka, m, if first { 1 } else { 0 });
                    first = false;
                }
            }
            wr_i32(
                &mut ka,
                s.k,
                (256 * match s.region {
                    RegionAbi::QkvzaQuad => 4,
                    _ => 1,
                }) as i32,
            );
            pad16(ka)
        };
        match s.region {
            RegionAbi::QkvzaQuad => {
                let mut ka = vec![0u8; 24];
                wr_u64(&mut ka, 0, wva);
                wr_u64(&mut ka, 8, xva);
                wr_u64(&mut ka, 16, o(&out_c));
                arms.cand_kernargs.push(pad16(ka));
                arms.ref_kernargs.push(own(o(&out_r)));
                arms.inc_kernargs.push(own(o(&out_i)));
            }
            RegionAbi::OwnAbiOneGroup => {
                arms.cand_kernargs.push(own(o(&out_c)));
                arms.ref_kernargs.push(own(o(&out_r)));
                arms.inc_kernargs.push(own(o(&out_i)));
            }
            RegionAbi::MultirowPair => {
                let pva = cand_blk_d.va(i * cand_stride);
                let mk = |out: u64| {
                    let mut ka = vec![0u8; 32];
                    wr_u64(&mut ka, 0, pva);
                    wr_u64(&mut ka, 8, xva);
                    wr_u64(&mut ka, 16, out);
                    wr_i32(&mut ka, 24, 2);
                    wr_i32(&mut ka, 28, 256);
                    ka
                };
                arms.cand_kernargs.push(mk(o(&out_c)));
                arms.ref_kernargs.push(mk(o(&out_r)));
                arms.inc_kernargs.push(mk(o(&out_i)));
            }
            RegionAbi::ResidualRecords => {
                let mut ka = vec![0u8; 16];
                wr_u64(&mut ka, 0, cand_blk_d.va(i * cand_stride));
                wr_u64(&mut ka, 8, o(&out_c));
                arms.cand_kernargs.push(ka);
                let mk = |out: u64| {
                    let mut ka = vec![0u8; 32];
                    wr_u64(&mut ka, 0, ref_blk_d.va(i * ref_stride));
                    wr_u64(&mut ka, 8, ref_x_d.va(i * refx_stride));
                    wr_u64(&mut ka, 16, out);
                    wr_i32(&mut ka, 24, 2);
                    wr_i32(&mut ka, 28, 1024);
                    ka
                };
                arms.ref_kernargs.push(mk(o(&out_r)));
                arms.inc_kernargs.push(mk(o(&out_i)));
            }
        }
    }
    let _ = &zero_x;

    let ref_name = format!(
        "pmdt_ref_{}_{}",
        if stage == Stage::Pre { "pre" } else { "post" },
        s.symbol
    );
    let ref_module = format!(
        "pmdt_ref_{}_{}",
        if stage == Stage::Pre { "pre" } else { "post" },
        s.module
    );
    let src = variant_source(s, inc, stage, &ref_name)?;
    gpu.ensure_kernel_public(&ref_module, &src, &ref_name)
        .map_err(err)?;
    let one = [1u32, 1, 1];
    let blk = [32u32, 1, 1];
    for ka in arms.cand_kernargs.iter_mut() {
        launch_candidate(gpu, cand, one, blk, 0, ka)?;
    }
    for ka in arms.ref_kernargs.iter_mut() {
        gpu.launch_kernel_blob(&ref_name, one, blk, 0, ka)
            .map_err(err)?;
    }
    for ka in arms.inc_kernargs.iter_mut() {
        gpu.launch_kernel_blob(s.symbol, one, blk, 0, ka)
            .map_err(err)?;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    let n = cases.len() * arms.out_bytes;
    let hc = out_c.download(gpu, n)?;
    let hr = out_r.download(gpu, n)?;
    let hi = out_i.download(gpu, n)?;

    // Reference-vs-unmodified-incumbent tie.
    let mut tie_mismatch = 0usize;
    let mut tie_first: Option<Value> = None;
    for (i, c) in cases.iter().enumerate() {
        let r = f32s(&hr[i * out_bytes..(i + 1) * out_bytes]);
        let inc_out = f32s(&hi[i * out_bytes..(i + 1) * out_bytes]);
        let (expect, got): (Vec<f32>, Vec<f32>) = match (s.region, stage) {
            (RegionAbi::MultirowPair, Stage::Post) => {
                (vec![r[0], r[32]], vec![inc_out[0], inc_out[1]])
            }
            (RegionAbi::MultirowPair, Stage::Pre) => {
                let a: [f32; 32] = r[..32].try_into().unwrap();
                let b: [f32; 32] = r[32..64].try_into().unwrap();
                (
                    vec![shfl_down_tree(&a), shfl_down_tree(&b)],
                    vec![inc_out[0], inc_out[1]],
                )
            }
            // Residual incumbent is RMW on zeroed y: y = 0 + lane0.
            (RegionAbi::ResidualRecords, Stage::Post) => {
                (vec![0.0 + r[0], 0.0 + r[1]], vec![inc_out[0], inc_out[1]])
            }
            (RegionAbi::ResidualRecords, Stage::Pre) => {
                let a: [f32; 32] = std::array::from_fn(|l| r[l * 2]);
                let b: [f32; 32] = std::array::from_fn(|l| r[l * 2 + 1]);
                (
                    vec![0.0 + shfl_down_tree(&a), 0.0 + shfl_down_tree(&b)],
                    vec![inc_out[0], inc_out[1]],
                )
            }
            (_, Stage::Post) => (vec![r[0]], vec![inc_out[0]]),
            (_, Stage::Pre) => {
                let a: [f32; 32] = r[..32].try_into().unwrap();
                (vec![shfl_down_tree(&a)], vec![inc_out[0]])
            }
        };
        for (e, g) in expect.iter().zip(&got) {
            if e.to_bits() != g.to_bits() {
                tie_mismatch += 1;
                tie_first.get_or_insert_with(|| {
                    json!({"case": i, "matrix_arg": weights[c.matrix].0.ptr(), "row": c.row, "group": c.group,
                           "reference": format!("{:08x}", e.to_bits()), "incumbent": format!("{:08x}", g.to_bits())})
                });
            }
        }
    }

    let cmp = compare(&hr, &hc);
    let mut v = json!({
        "mode": "region",
        "stage": format!("{stage:?}").to_lowercase(),
        "reference_symbol": ref_name,
        "reference_source_sha256": sha256_hex(src.as_bytes()),
        "cases": cases.len(),
        "row_stride": row_stride,
        "matrices": weights.iter().map(|(a, rows, _)| json!({"arg": a.ptr(), "rows": rows})).collect::<Vec<_>>(),
        "groups_per_row": groups,
        "out_bytes_per_case": out_bytes,
        "compared_bytes": n,
        "mismatch_bytes": cmp.mismatches,
        "mismatch_cases": (0..cases.len()).filter(|&i| hr[i * out_bytes..(i + 1) * out_bytes] != hc[i * out_bytes..(i + 1) * out_bytes]).count(),
        "reference_vs_unmodified_incumbent_mismatches": tie_mismatch,
    });
    if let Some(t) = tie_first {
        v["reference_tie_first"] = t;
    }
    if let Some(i) = cmp.first {
        let ci = i / out_bytes;
        let c = &cases[ci];
        let lane_bytes = if matches!(s.region, RegionAbi::ResidualRecords) {
            8
        } else {
            4
        };
        v["first_index"] = json!(i);
        v["first_case"] = json!({"case": ci, "matrix_arg": weights[c.matrix].0.ptr(), "row": c.row, "group": c.group,
                                 "byte_in_case": i - ci * out_bytes, "lane": (i - ci * out_bytes) % LANE_OUT / lane_bytes});
        v["hipcc_hex"] = json!(hex(&hr[ci * out_bytes..(ci + 1) * out_bytes]));
        v["pm_hex"] = json!(hex(&hc[ci * out_bytes..(ci + 1) * out_bytes]));
        // Dump the first failing case's exact inputs.
        let w = &weights[c.matrix].2;
        let nrows = if pair { 2 } else { 1 };
        let mut wdump = Vec::new();
        for r in 0..nrows {
            wdump.extend_from_slice(&w[(c.row + r) * row_bytes + c.group * 136..][..span * 136]);
        }
        std::fs::write(report.join("first_case_weights.bin"), &wdump).map_err(err)?;
        std::fs::write(
            report.join("first_case_x.bin"),
            &x[c.group * 1024..][..span * 1024],
        )
        .map_err(err)?;
    }
    Ok((cmp.mismatches + tie_mismatch, v))
}

// ---------------------------------------------------------------------------
// CLI

fn check_timing_arm(
    gpu: &Gpu,
    arm: &Arm,
    expected: &[Vec<u8>],
    name: &str,
    reference: &str,
) -> Result<(usize, Vec<Value>)> {
    let mut total = 0;
    let mut rows = Vec::new();
    for ((arg, len, dev), post) in arm.bufs.iter().zip(expected) {
        let got = dev.download(gpu, GUARD + len + GUARD)?;
        let (n, mut row) = cmp_json(
            &format!("{name}:arg{}+guards", arg.ptr()),
            &got,
            post,
            (name, reference),
        );
        if let Some(index) = row["first_index"].as_u64() {
            let index = index as usize;
            let region = if index < GUARD || index >= GUARD + len {
                "guard"
            } else {
                "payload"
            };
            row["first_index_kind"] = json!("framed_offset_including_guards");
            row["first_region"] = json!(region);
            if region == "payload" {
                row["first_payload_index"] = json!(index - GUARD);
            }
            if total == 0 {
                eprintln!(
                    "time: {name}:arg{} first mismatch at framed offset {index} ({region})",
                    arg.ptr()
                );
            }
        }
        total += n;
        rows.push(row);
    }
    Ok((total, rows))
}

fn timing_input_metadata(l: &Launch, seed: Option<u64>) -> serde_json::Map<String, Value> {
    serde_json::Map::from_iter([
        ("synthetic".into(), json!(seed.is_some())),
        ("synthetic_seed".into(), json!(seed)),
        (
            "arg_sizes".into(),
            json!(l
                .args
                .iter()
                .map(|(arg, bytes)| json!({"arg": arg.ptr(), "bytes": bytes.len()}))
                .collect::<Vec<_>>()),
        ),
    ])
}

fn timing_sysfs_reading(path: &Path, unit: &str) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    Some(json!({"path": path.display().to_string(), "raw": raw, "unit": unit}))
}

/// Enumerate AMD cards explicitly: HIP visibility can reorder device indices,
/// so no selected-device mapping is inferred from a DRM card number.
fn timing_sysfs_cards() -> Vec<Value> {
    let mut cards = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return cards;
    };
    let mut paths: Vec<_> = entries.flatten().map(|entry| entry.path()).collect();
    paths.sort();
    for card in paths {
        let Some(name) = card.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(index) = name.strip_prefix("card") else {
            continue;
        };
        if index.is_empty() || !index.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let device = card.join("device");
        let Ok(vendor) = std::fs::read_to_string(device.join("vendor")) else {
            continue;
        };
        if vendor.trim() != "0x1002" {
            continue;
        }
        let Ok(pci_path) = std::fs::canonicalize(&device) else {
            continue;
        };
        let Some(pci_identity) = pci_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let mut readings = Vec::new();
        for (file, unit) in [
            ("pp_dpm_sclk", "MHz table; * marks current level"),
            ("pp_dpm_mclk", "MHz table; * marks current level"),
            ("pp_od_clk_voltage", "driver-labelled clock/voltage table"),
            (
                "power_dpm_force_performance_level",
                "driver performance-level name",
            ),
            ("pp_power_profile_mode", "driver power-profile table"),
        ] {
            if let Some(reading) = timing_sysfs_reading(&device.join(file), unit) {
                readings.push(reading);
            }
        }
        if let Ok(hwmons) = std::fs::read_dir(device.join("hwmon")) {
            let mut hwmons: Vec<_> = hwmons.flatten().map(|entry| entry.path()).collect();
            hwmons.sort();
            for hwmon in hwmons {
                if std::fs::read_to_string(hwmon.join("name"))
                    .ok()
                    .as_deref()
                    .map(str::trim)
                    != Some("amdgpu")
                {
                    continue;
                }
                for (file, unit) in [
                    ("power1_average", "microwatts"),
                    ("power1_input", "microwatts"),
                    ("power1_cap", "microwatts"),
                    ("power1_cap_default", "microwatts"),
                    ("power1_cap_min", "microwatts"),
                    ("power1_cap_max", "microwatts"),
                    ("freq1_input", "Hz"),
                    ("freq1_label", "clock name"),
                    ("freq2_input", "Hz"),
                    ("freq2_label", "clock name"),
                    ("temp1_input", "millidegrees Celsius"),
                ] {
                    if let Some(reading) = timing_sysfs_reading(&hwmon.join(file), unit) {
                        readings.push(reading);
                    }
                }
            }
        }
        cards.push(json!({
            "drm_card": card.display().to_string(),
            "pci_identity": pci_identity,
            "pci_device_path": pci_path.display().to_string(),
            "readings": readings,
        }));
    }
    cards
}

/// Existing read-only HIP state APIs; unavailable readings are omitted.
fn timing_device_state(gpu: &Gpu) -> Value {
    let mut state = json!({"arch": gpu.arch, "device_id": gpu.device_id});
    if let Ok(device) = gpu.hip.current_device() {
        state["current_device"] = json!(device);
    }
    if let Ok((free, total)) = gpu.device_mem_info() {
        state["vram_free_bytes"] = json!(free);
        state["vram_total_bytes"] = json!(total);
    }
    let cards = timing_sysfs_cards();
    if !cards.is_empty() {
        state["amd_sysfs_cards"] = json!({
            "selection": "all AMD DRM cards; not mapped to the HIP device index",
            "cards": cards,
        });
    }
    state
}

#[allow(clippy::too_many_arguments)]
fn run_time(
    gpu: &mut Gpu,
    s: &ModuleSpec,
    inc: &Incumbent,
    cap: &Capture,
    cand: &Candidate,
    symbol: &str,
    n: usize,
    rounds: usize,
    synthetic: bool,
) -> Result<(usize, Value)> {
    let (calls, _) = timing_counts(n, rounds)?;
    let seed = synthetic.then(|| synthetic_seed(s.module));
    if synthetic {
        println!("inputs are synthetic; relative timing valid; not a production-data reproduction");
    }
    // This mode uses the freshly initialized Gpu's default stream for both arms.
    let l = if let Some(seed) = seed {
        synthetic_launch(s, cap, inc.kernarg_size, seed)?
    } else {
        captured_launch(s, cap, inc.kernarg_size)?
    };
    let mut incumbent_arm = build_framed_arm(gpu, s, &l, None, !synthetic)?;
    let mut candidate_arm = build_framed_arm(gpu, s, &l, None, !synthetic)?;
    let reference = if synthetic {
        "synthetic_incumbent_warm_and_pre"
    } else {
        "captured_post"
    };
    let mut expected = Vec::new();
    let mut resets = Vec::new();
    for (i, (arg, pre)) in l.args.iter().enumerate() {
        if !synthetic {
            let post = cap.arg(arg.ptr(), "post")?;
            if post.len() != pre.len() {
                return Err("captured POST extent differs from PRE".into());
            }
            let mut framed = vec![GUARD_BYTE; GUARD + post.len() + GUARD];
            framed[GUARD..GUARD + post.len()].copy_from_slice(&post);
            expected.push(framed);
        }
        if matches!(arg, Arg::Y { .. }) {
            let mut reset = vec![GUARD_BYTE; GUARD + pre.len() + GUARD];
            reset[GUARD..GUARD + pre.len()].copy_from_slice(pre);
            resets.push((i, reset));
        }
    }
    let mut result = json!({
        "schema": 1, "mode": "time", "module": s.module,
        "expected_reference": reference,
        "capture": cap.dir.display().to_string(), "capture_meta": cap.meta,
        "capture_meta_sha256": sha256_hex(&std::fs::read(cap.dir.join("meta.json")).map_err(err)?),
        "grid": l.grid, "block": l.block, "dynamic_lds": l.lds,
        "candidate_identity": {"path": cand.path.display().to_string(), "sha256": cand.sha256, "symbol": symbol},
        "incumbent_identity": {"symbol": s.symbol, "source_sha256": inc.source_sha256,
            "inventory_object_sha256": inc.object_sha256,
            "recipe": "inventory flags via Gpu::ensure_kernel_public"},
        "rounds": rounds, "calls_per_block": n, "calls_per_arm": 0,
        "incumbent": null, "candidate": null, "ratio": null,
        "separable": false, "observed_spread_ns": null, "median_delta_ns": null
    });
    result
        .as_object_mut()
        .unwrap()
        .extend(timing_input_metadata(&l, seed));
    result["device_state_before"] = timing_device_state(gpu);
    gpu.launch_kernel_blob(s.symbol, l.grid, l.block, l.lds, &mut incumbent_arm.kernarg)
        .map_err(err)?;
    launch_candidate(
        gpu,
        cand,
        l.grid,
        l.block,
        l.lds,
        &mut candidate_arm.kernarg,
    )?;
    gpu.hip.device_synchronize().map_err(err)?;
    if synthetic {
        for ((arg, pre), (_, len, dev)) in l.args.iter().zip(&incumbent_arm.bufs) {
            let warm = if matches!(arg, Arg::Y { .. }) {
                dev.download(gpu, GUARD + len + GUARD)?
            } else {
                Vec::new()
            };
            expected.push(synthetic_warm_reference(*arg, pre, &warm));
        }
    }
    let (a, ar) = check_timing_arm(gpu, &incumbent_arm, &expected, "incumbent", reference)?;
    let (b, br) = check_timing_arm(gpu, &candidate_arm, &expected, "candidate", reference)?;
    result["warm"] = json!({"incumbent": ar, "candidate": br, "mismatch_bytes": a + b});
    if a + b != 0 {
        result["device_state_after"] = timing_device_state(gpu);
        result["match"] = json!(false);
        result["status"] = json!("warm_mismatch_sampling_refused");
        eprintln!("time: warm mismatch; sampling refused");
        return Ok((a + b, result));
    }
    let mut incumbent_samples = Vec::with_capacity(calls);
    let mut candidate_samples = Vec::with_capacity(calls);
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = match gpu.hip.event_create() {
        Ok(event) => event,
        Err(e) => {
            let _ = gpu.hip.event_destroy(start);
            return Err(err(e));
        }
    };
    let measured = (|| -> Result<()> {
        for which in timing_schedule(n, rounds)? {
            let arm = match which {
                TimingArm::Incumbent => &mut incumbent_arm,
                TimingArm::Candidate => &mut candidate_arm,
            };
            for (i, bytes) in &resets {
                gpu.hip
                    .memcpy_htod(&arm.bufs[*i].2.buf, bytes)
                    .map_err(err)?;
            }
            gpu.hip.event_record(&start, None).map_err(err)?;
            match which {
                TimingArm::Incumbent => gpu
                    .launch_kernel_blob(s.symbol, l.grid, l.block, l.lds, &mut arm.kernarg)
                    .map_err(err)?,
                TimingArm::Candidate => {
                    launch_candidate(gpu, cand, l.grid, l.block, l.lds, &mut arm.kernarg)?
                }
            }
            gpu.hip.event_record(&stop, None).map_err(err)?;
            gpu.hip.event_synchronize(&stop).map_err(err)?;
            let ns = f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1e6;
            match which {
                TimingArm::Incumbent => incumbent_samples.push(ns),
                TimingArm::Candidate => candidate_samples.push(ns),
            }
        }
        Ok(())
    })();
    let destroy_start = gpu.hip.event_destroy(start).map_err(err);
    let destroy_stop = gpu.hip.event_destroy(stop).map_err(err);
    measured?;
    destroy_start?;
    destroy_stop?;
    result["device_state_after"] = timing_device_state(gpu);
    let (a, ar) = check_timing_arm(gpu, &incumbent_arm, &expected, "incumbent", reference)?;
    let (b, br) = check_timing_arm(gpu, &candidate_arm, &expected, "candidate", reference)?;
    result["post"] = json!({"incumbent": ar, "candidate": br, "mismatch_bytes": a + b});
    result["calls_per_arm"] = json!(calls);
    result["match"] = json!(a + b == 0);
    if a + b != 0 {
        result["status"] = json!("post_mismatch_no_performance_claim");
        eprintln!("time: post mismatch; no performance claim");
        return Ok((a + b, result));
    }
    let a = timing_stats(&incumbent_samples)?;
    let b = timing_stats(&candidate_samples)?;
    let (ratio, spread, delta, separable) = timing_relation(&a, &b);
    result["incumbent"] = a.json();
    result["candidate"] = b.json();
    result["ratio"] = json!(ratio);
    result["observed_spread_ns"] = json!(spread);
    result["median_delta_ns"] = json!(delta);
    result["separable"] = json!(separable);
    result["status"] = json!("matched");
    if separable {
        println!("time: candidate/incumbent median ratio {ratio:.6}");
    } else {
        println!("time: not separable at this N");
    }
    Ok((0, result))
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    #[test]
    fn synthetic_seed_and_fill_are_deterministic() {
        assert_eq!(synthetic_seed("abc"), 0xeacf018fbf1678ba);
        let seed = synthetic_seed(MODULES[0].module);
        assert_ne!(seed, 0);
        assert_ne!(seed, synthetic_seed(MODULES[1].module));
        let arg = Arg::Weight { ptr: 0, m: 72 };
        let bytes = synthetic_bytes(seed, arg, 4096);
        assert_eq!(bytes, synthetic_bytes(seed, arg, 4096));
        assert_eq!(&bytes[..64], synthetic_bytes(seed, arg, 64));
        assert_ne!(
            bytes,
            synthetic_bytes(seed, Arg::Weight { ptr: 8, m: 76 }, 4096)
        );
        assert_ne!(
            bytes,
            synthetic_bytes(synthetic_seed(MODULES[1].module), arg, 4096)
        );
        assert!(bytes.iter().all(|&byte| byte != 0));
        assert!(bytes.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn synthetic_tokens_nonzero_outputs_zero() {
        let seed = synthetic_seed(MODULES[0].module);
        let tokens = synthetic_bytes(seed, Arg::X { ptr: 32 }, 4096);
        assert!(tokens.iter().all(|&byte| byte != 0));
        assert!(tokens.windows(2).any(|pair| pair[0] != pair[1]));
        assert_ne!(tokens, synthetic_bytes(seed, Arg::X { ptr: 40 }, 4096));
        assert_eq!(
            synthetic_bytes(seed, Arg::Y { ptr: 40, m: 72 }, 129),
            vec![0; 129]
        );
    }

    fn synthetic_capture() -> (&'static ModuleSpec, Capture) {
        let s = &MODULES[0];
        let mut kernarg = vec![0x5a; s.k + 8];
        wr_i32(&mut kernarg, s.k, 256);
        let mut meta = json!({"files": {}});
        for (i, &arg) in s.args.iter().enumerate() {
            if let Arg::Weight { m, .. } | Arg::Y { m, .. } = arg {
                wr_i32(&mut kernarg, m, 1);
            }
            meta["files"][format!("arg{}.pre.bin", arg.ptr())] = json!({"bytes": (i + 1) * 17});
        }
        (
            s,
            Capture {
                dir: PathBuf::from("/nonexistent/pm-decode-twins-synthetic"),
                meta,
                kernarg,
                grid: [19, 2, 3],
                block: [64, 2, 1],
                lds: 1234,
            },
        )
    }

    #[test]
    fn synthetic_launch_uses_metadata_without_binaries() {
        let (s, cap) = synthetic_capture();
        let launch =
            synthetic_launch(s, &cap, cap.kernarg.len(), synthetic_seed(s.module)).unwrap();
        assert_eq!(launch.kernarg, cap.kernarg);
        assert_eq!(
            (launch.grid, launch.block, launch.lds),
            (cap.grid, cap.block, cap.lds)
        );
        assert_eq!(launch.args.len(), s.args.len());
        for (i, (arg, bytes)) in launch.args.iter().enumerate() {
            assert_eq!(arg.ptr(), s.args[i].ptr());
            assert_eq!(bytes.len(), (i + 1) * 17);
            assert_ne!(bytes.len(), arg_len(s, &cap.kernarg, *arg));
            assert!(bytes.iter().all(|&byte| if matches!(arg, Arg::Y { .. }) {
                byte == 0
            } else {
                byte != 0
            }));
        }
        assert!(synthetic_launch(s, &cap, cap.kernarg.len() + 1, 1).is_err());
        let mut missing_extent = cap;
        missing_extent.meta["files"]["arg0.pre.bin"]["bytes"] = Value::Null;
        assert!(synthetic_launch(s, &missing_extent, 0, 1).is_err());
    }

    #[test]
    fn synthetic_warm_reference_rejects_shared_guard_and_readonly_corruption() {
        let pre = vec![1, 2, 3, 4];
        let mut warm = vec![GUARD_BYTE; GUARD + pre.len() + GUARD];
        warm[GUARD..GUARD + pre.len()].copy_from_slice(&[9, 8, 7, 6]);
        warm[0] ^= 1;
        warm[GUARD + pre.len()] ^= 1;
        let y = synthetic_warm_reference(Arg::Y { ptr: 40, m: 72 }, &pre, &warm);
        assert_eq!(&y[GUARD..GUARD + pre.len()], &[9, 8, 7, 6]);
        assert_eq!(compare(&warm, &y).mismatches, 2);
        assert!(y[..GUARD]
            .iter()
            .chain(&y[GUARD + pre.len()..])
            .all(|&b| b == GUARD_BYTE));
        for arg in [Arg::Weight { ptr: 0, m: 72 }, Arg::X { ptr: 32 }] {
            let reference = synthetic_warm_reference(arg, &pre, &warm);
            assert_eq!(&reference[GUARD..GUARD + pre.len()], &pre);
            assert_eq!(compare(&warm, &reference).mismatches, 6);
        }
    }

    #[test]
    fn synthetic_json_seed_and_sizes_roundtrip() {
        let (s, cap) = synthetic_capture();
        let seed = synthetic_seed(s.module);
        let launch = synthetic_launch(s, &cap, 0, seed).unwrap();
        for seed in [Some(seed), Some(u64::MAX), None] {
            let metadata = Value::Object(timing_input_metadata(&launch, seed));
            let encoded = serde_json::to_string(&metadata).unwrap();
            let decoded: Value = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, metadata);
            assert_eq!(decoded["synthetic"], seed.is_some());
            assert_eq!(decoded["synthetic_seed"].as_u64(), seed);
            let sizes = decoded["arg_sizes"].as_array().unwrap();
            assert_eq!(sizes.len(), launch.args.len());
            for (entry, (arg, bytes)) in sizes.iter().zip(&launch.args) {
                assert_eq!(entry["arg"].as_u64(), Some(arg.ptr() as u64));
                assert_eq!(entry["bytes"].as_u64(), Some(bytes.len() as u64));
            }
        }
    }

    #[test]
    fn synthetic_cli_flag_is_time_only_and_consumes_no_value() {
        let args = |cmd: &str| {
            [
                cmd,
                "--synthetic",
                "--module",
                MODULES[0].module,
                "--capture",
                "/metadata-only",
                "--time",
                "2",
            ]
            .map(str::to_owned)
            .into_iter()
        };
        let time = parse_args(args("time")).unwrap();
        assert!(time.synthetic);
        assert_eq!(time.time, Some(2));
        assert_eq!(time.module, MODULES[0].module);
        for cmd in ["full", "region", "selftest"] {
            assert_eq!(
                parse_args(args(cmd)).err().unwrap(),
                "--synthetic is only valid with time"
            );
        }
    }
}

struct Opts {
    cmd: String,
    module: String,
    candidate: Option<PathBuf>,
    symbol: Option<String>,
    capture: PathBuf,
    stage: Option<Stage>,
    row_stride: usize,
    inventory: PathBuf,
    label: String,
    edges: bool,
    synthetic: bool,
    time: Option<usize>,
    rounds: usize,
    rounds_set: bool,
}

fn parse() -> Result<Opts> {
    parse_args(std::env::args().skip(1))
}

fn parse_args(mut a: impl Iterator<Item = String>) -> Result<Opts> {
    let cmd = a.next().ok_or("usage: pm_decode_twins <region|full|selftest|time> --module M --capture DIR [--candidate CO --symbol S] [--time N --rounds R --synthetic] [--stage pre|post] [--row-stride N] [--edges] [--label L]")?;
    let mut o = Opts {
        cmd,
        module: String::new(),
        candidate: None,
        symbol: None,
        capture: PathBuf::new(),
        stage: None,
        row_stride: 1,
        inventory: PathBuf::from(INVENTORY),
        label: String::new(),
        edges: false,
        synthetic: false,
        time: None,
        rounds: 5,
        rounds_set: false,
    };
    while let Some(k) = a.next() {
        if k == "--edges" {
            o.edges = true;
            continue;
        }
        if k == "--synthetic" {
            o.synthetic = true;
            continue;
        }
        let v = a.next().ok_or(format!("{k} needs a value"))?;
        match k.as_str() {
            "--module" => o.module = v,
            "--candidate" => o.candidate = Some(abs_path(&v, "--candidate")?),
            "--symbol" => o.symbol = Some(v),
            "--capture" => o.capture = abs_path(&v, "--capture")?,
            "--stage" => {
                o.stage = Some(match v.as_str() {
                    "pre" => Stage::Pre,
                    "post" => Stage::Post,
                    _ => return Err("--stage pre|post".into()),
                })
            }
            "--row-stride" => o.row_stride = v.parse().map_err(err)?,
            "--inventory" => o.inventory = abs_path(&v, "--inventory")?,
            "--label" => o.label = v,
            "--time" => o.time = Some(positive_count(&v)?),
            "--rounds" => {
                o.rounds = positive_count(&v)?;
                o.rounds_set = true;
            }
            _ => return Err(format!("unknown option {k}")),
        }
    }
    if o.synthetic && o.cmd != "time" {
        return Err("--synthetic is only valid with time".into());
    }
    if o.module.is_empty() || o.capture.as_os_str().is_empty() {
        return Err("--module and --capture are required".into());
    }
    if o.cmd == "time" {
        let n = o.time.ok_or("time requires positive --time N")?;
        timing_counts(n, o.rounds)?;
        if o.edges {
            return Err("--edges is not supported with time".into());
        }
    } else if o.time.is_some() || o.rounds_set {
        return Err("--time and --rounds are only valid with time".into());
    }
    Ok(o)
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("pm_decode_twins: error: {e}");
            std::process::exit(2);
        }
    }
}

fn run() -> Result<usize> {
    let o = parse()?;
    let s = spec(&o.module)?;
    let inc = incumbent(&o.inventory, s)?;
    let cap = load_capture(&o.capture, s)?;
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 only, got {}", gpu.arch));
    }
    gpu.ensure_kernel_public(s.module, &inc.source, s.symbol)
        .map_err(err)?;
    let label = if o.label.is_empty() {
        o.cmd.clone()
    } else {
        o.label.clone()
    };
    let report_dir = Path::new(RESULTS).join(s.module).join(&label);
    std::fs::create_dir_all(&report_dir).map_err(err)?;
    let argv: Vec<String> = std::env::args().collect();
    let mut report = json!({
        "schema": "pmdt-oracle-v1",
        "command": argv.join(" "),
        "module": s.module,
        "incumbent_symbol": s.symbol,
        "incumbent_source_sha256": inc.source_sha256,
        "incumbent_recipe": "inventory flags via Gpu::ensure_kernel_public",
        "capture": o.capture.display().to_string(),
        "capture_launch_index": cap.meta["launch_index"],
        "capture_position": cap.meta["position"],
        "arch": gpu.arch,
    });
    let failures = match o.cmd.as_str() {
        "full" | "region" | "time" => {
            let path = o.candidate.as_ref().ok_or("--candidate required")?;
            let sym = o.symbol.as_deref().ok_or("--symbol required")?;
            let cand = load_candidate(&gpu, path, sym)?;
            report["candidate"] = json!({"path": cand.path.display().to_string(), "sha256": cand.sha256, "symbol": sym});
            let (n, v) = if o.cmd == "time" {
                let (n, v) = run_time(
                    &mut gpu,
                    s,
                    &inc,
                    &cap,
                    &cand,
                    sym,
                    o.time.unwrap(),
                    o.rounds,
                    o.synthetic,
                )?;
                std::fs::write(
                    report_dir.join("timing.json"),
                    serde_json::to_string_pretty(&v).map_err(err)?,
                )
                .map_err(err)?;
                (n, v)
            } else if o.cmd == "full" {
                let (n, mut v) = run_full(&mut gpu, s, &inc, &cap, &cand, None, false)?;
                if o.edges {
                    let (ne, ve) = run_edges(&mut gpu, s, &cap, &cand)?;
                    v["edges"] = ve;
                    (n + ne, v)
                } else {
                    (n, v)
                }
            } else {
                let stage = o.stage.unwrap_or(s.default_stage);
                region(
                    &mut gpu,
                    s,
                    &inc,
                    &cap,
                    &cand,
                    stage,
                    o.row_stride,
                    &report_dir,
                )?
            };
            report["result"] = v;
            n
        }
        "selftest" => {
            let obj = std::fs::read(&inc.object_path).map_err(err)?;
            if sha256_hex(&obj) != inc.object_sha256 {
                return Err(format!(
                    "{}: object sha256 mismatch",
                    inc.object_path.display()
                ));
            }
            let cand = load_candidate(&gpu, &inc.object_path, s.symbol)?;
            report["candidate"] = json!({"path": cand.path.display().to_string(), "sha256": cand.sha256, "symbol": s.symbol, "kind": "inventory hipcc object"});
            let (clean, v0) = run_full(&mut gpu, s, &inc, &cap, &cand, None, false)?;
            let wptr = s.args[0].ptr();
            let (perturbed, v1) = run_full(&mut gpu, s, &inc, &cap, &cand, Some((wptr, 8)), false)?;
            let (flipped, v2) = run_full(&mut gpu, s, &inc, &cap, &cand, None, true)?;
            let (edge_diff, v3) = run_edges(&mut gpu, s, &cap, &cand)?;
            report["result"] = json!({
                "hipcc_vs_hipcc": v0,
                "perturbed_weight_byte": {"arg": wptr, "byte": 8, "xor": "0x10", "detected": perturbed > 0, "result": v1},
                "flipped_output_byte": {"detected": flipped == 1, "result": v2},
                "hipcc_vs_hipcc_edges": v3,
            });
            usize::from(clean != 0)
                + usize::from(perturbed == 0)
                + usize::from(flipped != 1)
                + usize::from(edge_diff != 0)
        }
        other => return Err(format!("unknown command {other}")),
    };
    report["pass"] = json!(failures == 0);
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    std::fs::write(report_dir.join("report.json"), &text).map_err(err)?;
    println!("{text}");
    println!(
        "{}: {} -> {}",
        if failures == 0 { "PASS" } else { "FAIL" },
        s.module,
        report_dir.join("report.json").display()
    );
    Ok(failures)
}
