// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! The contract between `hip_loop` (driver: compile, resources, identity, ABBA,
//! JSON) and one hipcc incumbent under test (`mq6.rs`, `hc.rs`, `qsa.rs`).
//!
//! A [`Node`] is constructed GPU-free from a `--shape` string (shape grammar,
//! launch geometry, incumbent source) so that `--offline` and the unit tests
//! never open a device. `setup` then allocates seeded inputs on the card,
//! `reset` restores every mutable output (residual GEMMs accumulate into Y)
//! to its pristine value, `dispatch` issues the ONE production dispatch given a
//! loaded function handle (the driver loads the incumbent and the candidate
//! as separate modules and calls `dispatch` with each handle), and
//! `read_output` returns the output bytes the identity gate compares.
//!
//! Included with `#[path = "hip_loop/node.rs"] mod node;` (siblings reach it
//! as `crate::node`).
#![allow(dead_code)]

use hip_bridge::{DeviceBuffer, Function, KernargBlob};
pub use rdna_compute::Gpu;
use serde_json::Value;
use std::collections::BTreeMap;

pub type Result<T, E = String> = std::result::Result<T, E>;

pub fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

/// The arch of the original `hip_loop` nodes (`mq6_res`, `hc_mix`, `qsa_hg4`); a node overrides
/// [`Node::arch`] (the lane nodes are gfx1201).
pub const ARCH: &str = "gfx1100";

/// What the production JIT compiles for the incumbent: the module name the
/// runtime passes to `KernelCompiler` (module-level flags and scheduler profile
/// are keyed on it), the full shipped source text, the launched kernel symbol
/// and any `extra_flags` the runtime compiler carries for it (normally empty).
/// A candidate FILE is compiled with the same `module`/`extra_flags` and must
/// define `symbol` with the same signature and launch contract.
#[derive(Clone, Debug)]
pub struct Incumbent {
    pub module: String,
    pub source: String,
    pub symbol: String,
    pub extra_flags: String,
    /// Where production selects this source/symbol (`file:line`), for the record.
    pub selection: String,
}

/// Launch geometry as passed to `hipModuleLaunchKernel`: grid in WORKGROUPS,
/// block in threads, dynamic LDS bytes. (rocprof's `grid_size` is
/// `grid * block` per axis; see [`Geometry::threads`].)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub lds: u32,
}

impl Geometry {
    /// The profiler's `grid_size` (work-items per axis): `grid[i] * block[i]`.
    pub fn threads(&self) -> [u64; 3] {
        [
            u64::from(self.grid[0]) * u64::from(self.block[0]),
            u64::from(self.grid[1]) * u64::from(self.block[1]),
            u64::from(self.grid[2]) * u64::from(self.block[2]),
        ]
    }
}

/// Useful work of one dispatch, for the record's achieved-rate fields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Work {
    pub flops: f64,
    pub bytes: f64,
}

/// One explicit kernel argument of an emulated launch (natural alignment, as [`Args`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EmuArg {
    /// Pointer to `EmuCase::bufs[i]` (resolved to the emulator address of that region).
    Buf(usize),
    /// Pointer `byte_offset` bytes into `EmuCase::bufs[i]` (a node that packs several output
    /// segments into one emulator region, separated by [`PAD_WORDS`] unwritten pad words).
    BufOff(usize, u64),
    /// A pointer argument that is null on the card too.
    Null,
    I32(i32),
    U32(u32),
    U64(u64),
    F32(f32),
}

/// One host-side input/output buffer of an emulated launch: mapped as its own exact-size
/// emulator region with unmapped guard gaps on both sides.
#[derive(Clone, Debug)]
pub struct EmuBuf {
    pub name: &'static str,
    pub bytes: Vec<u8>,
}

/// The host-side twin of a node's `setup`: seeded inputs and the production kernarg layout at the
/// node's small `emu` shape, with no device. `out` is the buffer the identity compare reads.
#[derive(Clone, Debug)]
pub struct EmuCase {
    pub bufs: Vec<EmuBuf>,
    pub args: Vec<EmuArg>,
    pub out: usize,
}

/// Output-owner map value of an element no workgroup may write (the unwritten pad between packed
/// output segments); an arm that writes it fails the gate.
pub const PAD_OWNER: [u32; 3] = [u32::MAX; 3];
/// Pad words (poison, never written by a correct kernel) after every packed output segment.
pub const PAD_WORDS: usize = 256;

/// One arm's code object plus its candidate header, for a node's own device-side identity matrix.
pub struct ArmImage<'a> {
    pub tag: &'a str,
    pub image: &'a [u8],
    /// The `// hip_loop:` header (empty for the incumbent: production geometry).
    pub header: &'a Header,
}

/// One hipcc incumbent at one shape.
pub trait Node {
    /// Node name as on the command line (`mq6_res`, `hc_mix`, `qsa_hg4`).
    fn name(&self) -> &'static str;
    /// Short shape label (`qkv`, `z`, ...), used for the record and file names.
    fn shape_label(&self) -> String;
    /// Full resolved shape as JSON (dimensions, tile sizes).
    fn shape_json(&self) -> Value;
    /// The shipped source + module + symbol (GPU-free).
    fn incumbent(&self) -> Result<Incumbent>;
    /// Launch geometry for a candidate whose `// hip_loop:` header is `hdr`.
    /// The node takes the keys it understands out of `hdr` (defaults = the
    /// production launch), refuses unknown keys (`hdr.no_leftover()`) and a
    /// geometry that does not cover the output (GPU-free).
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry>;
    /// Production launch geometry (no header; GPU-free).
    fn geometry(&self) -> Geometry {
        self.geometry_with(&mut Header::default())
            .expect("the production geometry covers the output")
    }
    /// FLOPs and bytes of one dispatch (GPU-free).
    fn work(&self) -> Work;
    /// Extra node-specific record fields (merged into the top-level record).
    fn record_extra(&self) -> Value {
        Value::Null
    }
    /// The host-side seeded inputs + explicit kernargs of the pre-card emulator gate (`--emu`),
    /// at this node's shape (the `emu` preset by default). Deterministic in `seed`, GPU-free.
    fn emu_case(&self, _seed: u64) -> Result<EmuCase> {
        Err(format!("{}: no host-side emulator case", self.name()))
    }
    /// Production-extent shapes (`--shape` strings of this node) the emulator gate also runs, sampled:
    /// the judge shapes, plus the capacity extremes the node's addressing depends on.
    fn extent_shapes(&self) -> Vec<String> {
        Vec::new()
    }
    /// Which workgroup of a launch with header `hdr` (no header = production geometry) writes output
    /// ELEMENT `e` (the output buffer is an array of 4-byte elements). GPU-free; the extent gate samples
    /// workgroups of both arms and compares exactly the output elements they own.
    fn out_owner<'a>(&'a self, _hdr: &Header) -> Result<Box<dyn Fn(usize) -> [u32; 3] + 'a>> {
        Err(format!("{}: no output-owner map", self.name()))
    }

    /// Allocate seeded valid inputs and the output; keep a pristine copy of
    /// every mutable output. Must be deterministic in `seed`.
    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()>;
    /// Restore every mutable output to its pristine post-`setup` value.
    fn reset(&mut self, gpu: &Gpu) -> Result<()>;
    /// The kernarg blob of the production launch (explicit args, natural
    /// alignment), valid after `setup`.
    fn kernarg(&self) -> Result<Vec<u8>>;
    /// Issue ONE dispatch with the loaded function `f` at geometry `g` (the
    /// production launch for the incumbent, the header-derived one for a
    /// candidate). `karg` is `self.kernarg()` built by the driver once, outside
    /// the timed interval. Default: `launch_kernel_blob`.
    fn dispatch(&self, gpu: &Gpu, f: &Function, g: &Geometry, karg: &mut [u8]) -> Result<()> {
        // SAFETY: `kernarg` packs the kernel's declared explicit arguments per the
        // production launcher and every pointer addresses a live allocation owned
        // by the node (allocated in `setup`).
        unsafe {
            gpu.hip
                .launch_kernel_blob(f, g.grid, g.block, g.lds, None, karg)
        }
        .map_err(err)
    }
    /// The output bytes the identity gate compares (after a `reset` + `dispatch`
    /// + device sync).
    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>>;
    /// The arch this node's incumbent is built for and judged on (`hipcc --offload-arch`, the
    /// device the GPU mode demands). Default gfx1100.
    fn arch(&self) -> &'static str {
        ARCH
    }
    /// Called by the driver before every timed dispatch number `i` (0-based, across both arms):
    /// the node may re-point `karg` at another weight set of its pool.
    fn select_set(&self, _karg: &mut [u8], _i: usize) {}
    /// The shape asked for a cache flush between timed dispatches (`flush=1`).
    fn flush_requested(&self) -> bool {
        false
    }
    /// `--emu` also runs the static hazard checks (`peacemaker_lift::hazards_gfx11` +
    /// `peacemaker_ir::passes::hazards::analyze`) on the arms' objects and fails the gate on the
    /// candidate's findings. Default off (the gfx1100 nodes' gate is unchanged).
    fn static_hazards(&self) -> bool {
        false
    }
    /// Shapes (`--shape` strings of this node) whose adversarial matrix `--identity-only` runs; empty: the
    /// node's own shape only.
    fn identity_shapes(&self) -> Vec<String> {
        Vec::new()
    }
    /// The extent cases run EVERY workgroup of the production grids and compare the whole output
    /// (`--emu-extent` overrides); default: sampled workgroups with the output-owner closure.
    fn extent_full_default(&self) -> bool {
        false
    }
    /// Extra device-side identity of a node (an adversarial input matrix run through both arms
    /// and a CPU reference), run after the basic identity passed and before timing. `Ok(None)`:
    /// the node has none. The returned JSON carries `pass` (bool).
    fn adversarial(
        &mut self,
        _gpu: &Gpu,
        _base: &ArmImage,
        _cand: &ArmImage,
        _seed: u64,
    ) -> Result<Option<Value>> {
        Ok(None)
    }
    /// Free device buffers (DeviceBuffer has no Drop).
    fn teardown(&mut self, gpu: &Gpu);
}

// ---------------------------------------------------------------------------
// Shape grammar

/// `--shape` grammar: `[PRESET][,key=value]...`; a bare token is the preset
/// (at most one), `key=value` pairs override the preset's values. Items in
/// any order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShapeSpec {
    pub preset: Option<String>,
    pub kv: BTreeMap<String, String>,
}

impl ShapeSpec {
    pub fn parse(s: Option<&str>) -> Result<Self> {
        let mut out = Self::default();
        for item in s
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|i| !i.is_empty())
        {
            match item.split_once('=') {
                Some((k, v)) => {
                    let (k, v) = (k.trim(), v.trim());
                    if k.is_empty() || v.is_empty() {
                        return Err(format!("--shape item `{item}`: expected key=value"));
                    }
                    if out.kv.insert(k.to_owned(), v.to_owned()).is_some() {
                        return Err(format!("--shape: `{k}` given twice"));
                    }
                }
                None => {
                    if let Some(p) = &out.preset {
                        return Err(format!("--shape: two presets `{p}` and `{item}`"));
                    }
                    out.preset = Some(item.to_owned());
                }
            }
        }
        Ok(out)
    }

    /// Take `key` as a non-negative integer, else `default`, else an error.
    pub fn take_usize(&mut self, key: &str, default: Option<usize>) -> Result<usize> {
        match self.kv.remove(key) {
            Some(v) => v
                .parse::<usize>()
                .map_err(|e| format!("--shape {key}={v}: {e}")),
            None => default.ok_or_else(|| format!("--shape: `{key}` is required")),
        }
    }

    /// Error naming every shape key nobody consumed.
    pub fn no_leftover(&self) -> Result<()> {
        if self.kv.is_empty() {
            return Ok(());
        }
        let keys: Vec<&str> = self.kv.keys().map(String::as_str).collect();
        Err(format!("--shape: unknown key(s) {}", keys.join(", ")))
    }
}

// ---------------------------------------------------------------------------
// Deterministic inputs

/// xorshift64* (`seed` 0 becomes 1).
#[derive(Clone, Debug)]
pub struct Xs64(u64);

impl Xs64 {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

pub fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

// ---------------------------------------------------------------------------
// Device buffers and kernargs

/// A device allocation with its byte size.
pub struct Buf {
    pub buf: DeviceBuffer,
    pub bytes: usize,
}

impl Buf {
    pub fn alloc(gpu: &Gpu, bytes: usize) -> Result<Self> {
        let buf = gpu.hip.malloc(bytes.max(1)).map_err(err)?;
        Ok(Self { buf, bytes })
    }
    pub fn from_bytes(gpu: &Gpu, data: &[u8]) -> Result<Self> {
        let b = Self::alloc(gpu, data.len())?;
        gpu.hip.memcpy_htod(&b.buf, data).map_err(err)?;
        Ok(b)
    }
    pub fn upload(&self, gpu: &Gpu, data: &[u8]) -> Result<()> {
        if data.len() > self.bytes {
            return Err(format!(
                "upload of {} B into a {} B buffer",
                data.len(),
                self.bytes
            ));
        }
        gpu.hip.memcpy_htod(&self.buf, data).map_err(err)
    }
    pub fn download(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let mut v = vec![0u8; self.bytes];
        gpu.hip.memcpy_dtoh(&mut v, &self.buf).map_err(err)?;
        Ok(v)
    }
    /// Device-to-device copy of the whole buffer from `src` (same size).
    pub fn copy_from(&self, gpu: &Gpu, src: &Buf) -> Result<()> {
        if src.bytes != self.bytes {
            return Err(format!("copy_from: {} B into {} B", src.bytes, self.bytes));
        }
        gpu.hip
            .memcpy_dtod(&self.buf, &src.buf, self.bytes)
            .map_err(err)
    }
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.buf.as_ptr()
    }
    pub fn free(self, gpu: &Gpu) {
        let _ = gpu.hip.free(self.buf);
    }
}

/// Kernarg builder: natural alignment, `push_ptr` takes a [`Buf`].
#[derive(Default)]
pub struct Args(KernargBlob);

impl Args {
    pub fn new() -> Self {
        Self(KernargBlob::new())
    }
    pub fn ptr(&mut self, b: &Buf) -> &mut Self {
        self.0.push_ptr(b.ptr() as *const _);
        self
    }
    pub fn raw_ptr(&mut self, p: u64) -> &mut Self {
        self.0.push_u64(p);
        self
    }
    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.0.push_i32(v);
        self
    }
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.push_u32(v);
        self
    }
    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.0.push_f32(v);
        self
    }
    pub fn finish(self) -> Vec<u8> {
        self.0.into_vec()
    }
}

// ---------------------------------------------------------------------------
// Candidate header

/// The structured header of a candidate `.hip`: every line of the form
/// `// hip_loop: key=value key=value` (several lines merge; a repeated key is
/// an error). It describes the launch geometry the candidate needs; no header
/// means the production launch exactly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header {
    pub kv: BTreeMap<String, String>,
}

impl Header {
    pub fn parse(source: &str) -> Result<Self> {
        const TAG: &str = "hip_loop:";
        let mut kv = BTreeMap::new();
        for line in source.lines() {
            let Some(rest) = line
                .trim_start()
                .strip_prefix("//")
                .map(str::trim_start)
                .and_then(|r| r.strip_prefix(TAG))
            else {
                continue;
            };
            for tok in rest.split_whitespace() {
                let (k, v) = tok
                    .split_once('=')
                    .filter(|(k, v)| !k.is_empty() && !v.is_empty())
                    .ok_or_else(|| format!("hip_loop header token `{tok}`: expected key=value"))?;
                if kv.insert(k.to_owned(), v.to_owned()).is_some() {
                    return Err(format!("hip_loop header: `{k}` given twice"));
                }
            }
        }
        Ok(Self { kv })
    }

    /// The same `key=value` tokens as a header line, separated by spaces or
    /// commas (the `--cand-geometry` argument).
    pub fn parse_list(s: &str) -> Result<Self> {
        let line = s.replace(',', " ");
        Self::parse(&format!("// hip_loop: {line}"))
    }

    /// Take `key` as a non-negative integer, else `default`.
    pub fn take_usize(&mut self, key: &str, default: usize) -> Result<usize> {
        match self.kv.remove(key) {
            Some(v) => v
                .parse::<usize>()
                .map_err(|e| format!("hip_loop header {key}={v}: {e}")),
            None => Ok(default),
        }
    }

    /// Error naming every header key nobody consumed.
    pub fn no_leftover(&self) -> Result<()> {
        if self.kv.is_empty() {
            return Ok(());
        }
        let keys: Vec<&str> = self.kv.keys().map(String::as_str).collect();
        Err(format!(
            "hip_loop header: unknown key(s) {}",
            keys.join(", ")
        ))
    }
}

impl Xs64 {
    /// Standard normal (Box-Muller, one value per call).
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 40) as f32 + 1.0) / (1u64 << 24) as f32;
        let u2 = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// The common `// hip_loop:` geometry of the tiled GEMM nodes: keys `tile_m`
/// (output rows per workgroup), `tile_n` (tokens per workgroup), `block`
/// (threads) and `lds_dynamic` (bytes), defaulting to the production values.
/// Grid = `[ceil(m / tile_m), ceil(n / tile_n), 1]` workgroups. Refuses unknown
/// keys, zero/oversized values and a grid that does not cover the `m x n`
/// output.
pub fn tiled_geometry(
    hdr: &mut Header,
    m: usize,
    n: usize,
    def_tile_m: usize,
    def_tile_n: usize,
    def_block: usize,
) -> Result<Geometry> {
    let tile_m = hdr.take_usize("tile_m", def_tile_m)?;
    let tile_n = hdr.take_usize("tile_n", def_tile_n)?;
    let block = hdr.take_usize("block", def_block)?;
    let lds = hdr.take_usize("lds_dynamic", 0)?;
    hdr.no_leftover()?;
    if tile_m == 0 || tile_n == 0 {
        return Err(format!(
            "hip_loop header: tile_m={tile_m} tile_n={tile_n} must be positive"
        ));
    }
    if block == 0 || block % 32 != 0 || block > 1024 {
        return Err(format!(
            "hip_loop header: block={block} must be a multiple of 32 in 32..=1024"
        ));
    }
    if lds > 64 * 1024 {
        return Err(format!(
            "hip_loop header: lds_dynamic={lds} exceeds the 64 KiB workgroup LDS"
        ));
    }
    let (gx, gy) = (m.div_ceil(tile_m), n.div_ceil(tile_n));
    if gx * tile_m < m || gy * tile_n < n {
        return Err(format!(
            "launch geometry does not cover the {m}x{n} output (grid {gx}x{gy}, tile {tile_m}x{tile_n})"
        ));
    }
    let g = |v: usize| u32::try_from(v).map_err(|_| format!("grid dimension {v} exceeds u32"));
    Ok(Geometry {
        grid: [g(gx)?, g(gy)?, 1],
        block: [block as u32, 1, 1],
        lds: lds as u32,
    })
}

/// Tile sizes `(tile_m, tile_n)` of a tiled-GEMM launch from a header, read without consuming it
/// (same keys and defaults as [`tiled_geometry`]).
pub fn header_tiles(hdr: &Header, def_tile_m: usize, def_tile_n: usize) -> Result<(usize, usize)> {
    let get = |k: &str, d: usize| -> Result<usize> {
        match hdr.kv.get(k) {
            Some(v) => v
                .parse::<usize>()
                .map_err(|e| format!("hip_loop header {k}={v}: {e}")),
            None => Ok(d),
        }
    };
    let (m, n) = (get("tile_m", def_tile_m)?, get("tile_n", def_tile_n)?);
    if m == 0 || n == 0 {
        return Err("hip_loop header: tile_m/tile_n must be positive".into());
    }
    Ok((m, n))
}
