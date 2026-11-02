// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! PM-ACO parity oracle: first MQ4 residual GEMV kill experiment (gfx1201).
//!
//! One kernel, one frozen real fixture: `gemv_mq4g256v2_residual` with the
//! real tensors captured at the 27B decode callsite (`pmdt-capture-v1`,
//! `oracle-inputs/gemv_mq4g256v2_residual/occ{0,127}`: whole pre/post A, x, y,
//! kernarg blob, grid/block, per-file SHA-256). Arms, all loaded through
//! `module_load_data` and launched through `launch_kernel_blob` with the
//! ORIGINAL 32-byte kernarg ABI (A@0 x@8 y@16 M@24 K@28):
//!
//! - `hipcc`: the frozen original hipcc object of `inventory.json` (source,
//!   recipe flags and object SHA-256 re-verified).
//! - `twin`: the accepted PM native twin, admitted through the repo's
//!   `rdna_compute::pm_decode_twins` (`pm_decode_module` + `verify`).
//! - `candidate`: the F3 ACO PM/HIP code object, bound by an
//!   `aco-pm-candidate-v1` manifest (object/SPIR-V/ACO-binary/sidecar/body
//!   SHA-256, wave, block, kernarg ABI).
//! - `vulkan`: the ORIGINAL RADV/Vulkan shader's output, produced by the
//!   Vulkan host tool and consumed here as an `aco-vulkan-ref-v1` receipt
//!   (framed post buffers + SPIR-V/binary/fixture/config/driver identity).
//!   Absent or mismatching receipt fields are refused, never inferred.
//!
//! Every pointer argument lives in its own VMM allocation (no legacy
//! `hipMalloc`), framed by 4096 guard bytes (0xA5) and padded to the VMM
//! granule with guard bytes. Whole A/x/y buffers including guards are compared
//! byte-for-byte: outputs, the read-modify-write y, and unchanged inputs.
//! Each arm runs eagerly and through a hipGraph capture/replay (twice, from
//! reset buffers) with the same function handle and kernarg blob.
//!
//! Modes (each separate; timing never runs implicitly)
//! - `full`: parity matrix + negative controls. Shapes: the real fixture, and
//!   a real-prefix derived odd/tail shape (M-1 rows, K-256: A cropped row-wise,
//!   x/y prefixes) at the captured launcher rule (grid = M) and at the minimal
//!   grid. A candidate that does not claim `shape_support.odd_tail` gets an
//!   explicit UNSUPPORTED refusal for the derived shapes; it is never skipped
//!   silently.
//! - `selftest`: hipcc-vs-hipcc (+ accepted twin) pipeline and negative
//!   controls, no Vulkan, no candidate.
//! - `prepare`: write the framed pre buffers of every shape for the Vulkan host.
//! - `time`: ONE diagnostic timing pass (candidate vs hipcc vs accepted twin,
//!   captured geometry; the candidate image is timed whole, so any adapter
//!   prologue cost is included). Requires a passing `full` report bound to the same
//!   candidate/capture/twin. Diagnostic only, never a product speed claim.
//! - `twin-time`: ONE hipcc-vs-accepted-twin timing pass for the default-on native PM decode
//!   twin; needs only `--capture` and `--out` (no candidate, ACO manifest, Vulkan receipt or
//!   parity report; `--iters/--reps/--warmup/--cache` as for `time`). The actual loaders
//!   (`load_hipcc`, `load_twin`) re-verify the inventory source/object and the accepted twin
//!   image. Before timing, both arms run eagerly and through a hipGraph on the captured
//!   M/K/grid/block and their whole A/x/y buffers plus guards are compared against the captured
//!   real post bytes and hipcc-vs-twin; any mismatch fails closed without timing. Timing is two
//!   arms only, rep-major with the arm order alternating per rep (executed order recorded),
//!   event timing, no ACO arm, no occupancy probes. Writes `twin-time-report.json`. This is a
//!   kernel-only diagnostic of the residual launch at the captured shape; the product A/B
//!   (`allpm_decode` default on/off at model level) is a separate measurement and is not
//!   replaced by it.
//!
//! ```text
//! pm_aco_oracle full --capture .../oracle-inputs/gemv_mq4g256v2_residual/occ0 \
//!   --candidate /abs/cand.co --symbol gemv_mq4g256v2_residual \
//!   --candidate-manifest /abs/cand.manifest.json --vulkan-receipt /abs/vk/receipt.json
//! ```

use hip_bridge::{Function, Module, Stream};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;

const MODULE: &str = "gemv_hfq4g256_residual_mq4v2";
const SYMBOL: &str = "gemv_mq4g256v2_residual";
const INVENTORY: &str = "/home/kaden/qcal/release-0.4.2/pm-decode-twins/inventory.json";
const RESULTS: &str = "/home/kaden/qcal/release-0.4.2/pm-aco-pilot/oracle-results";
const GUARD: usize = 4096;
const GUARD_BYTE: u8 = 0xA5;
const GROUP_BYTES: usize = 136;
const KERNARG_LEN: usize = 32;
/// Kernarg offsets of the pointer slots (A, x, y) and of M, K.
const PTR_OFF: [usize; 3] = [0, 8, 16];
const OFF_M: usize = 24;
const OFF_K: usize = 28;
const ARG_NAME: [&str; 3] = ["A", "x", "y"];
const VK_SCHEMA: &str = "aco-vulkan-ref-v1";
const MANIFEST_SCHEMA: &str = "aco-pm-candidate-v1";

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

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn wr_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn wr_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---------------------------------------------------------------------------
// JSON field access (every required field is checked, nothing is inferred)

fn jget<'a>(v: &'a Value, path: &str) -> Result<&'a Value> {
    let mut cur = v;
    for part in path.split('.') {
        cur = cur
            .get(part)
            .ok_or_else(|| format!("required field `{path}` is absent"))?;
    }
    Ok(cur)
}

fn jstr(v: &Value, path: &str) -> Result<String> {
    match jget(v, path)?.as_str() {
        Some(s) if !s.is_empty() => Ok(s.to_owned()),
        _ => Err(format!("field `{path}` must be a non-empty string")),
    }
}

fn jhex64(v: &Value, path: &str) -> Result<String> {
    let s = jstr(v, path)?;
    if s.len() != 64
        || !s
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(format!(
            "field `{path}` must be lowercase 64-hex sha256: {s}"
        ));
    }
    Ok(s)
}

fn ju64(v: &Value, path: &str) -> Result<u64> {
    jget(v, path)?
        .as_u64()
        .ok_or_else(|| format!("field `{path}` must be an unsigned integer"))
}

fn jarr3(v: &Value, path: &str) -> Result<[u32; 3]> {
    let a = jget(v, path)?
        .as_array()
        .ok_or_else(|| format!("field `{path}` must be an array"))?;
    if a.len() != 3 {
        return Err(format!("field `{path}` must have 3 entries"));
    }
    let mut out = [0u32; 3];
    for (i, e) in a.iter().enumerate() {
        out[i] = e
            .as_u64()
            .ok_or_else(|| format!("field `{path}`[{i}] not an integer"))? as u32;
    }
    Ok(out)
}

fn must_eq<T: PartialEq + std::fmt::Debug>(what: &str, got: T, want: T) -> Result<()> {
    if got != want {
        return Err(format!("{what}: {got:?} != expected {want:?}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Capture (frozen real fixture)

struct Capture {
    dir: PathBuf,
    meta: Value,
    meta_sha256: String,
    kernarg: Vec<u8>,
    grid: [u32; 3],
    block: [u32; 3],
    lds: u32,
    m: usize,
    k: usize,
    pre: [Vec<u8>; 3],
    post: [Vec<u8>; 3],
}

fn read_hashed(dir: &Path, meta: &Value, name: &str) -> Result<Vec<u8>> {
    let b = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
    let want = meta["files"][name]["sha256"]
        .as_str()
        .ok_or(format!("{name}: no hash in meta"))?
        .to_owned();
    if sha256_hex(&b) != want {
        return Err(format!("{name}: sha256 mismatch against capture meta"));
    }
    let bytes = meta["files"][name]["bytes"]
        .as_u64()
        .ok_or(format!("{name}: no byte count in meta"))?;
    if bytes as usize != b.len() {
        return Err(format!("{name}: {} bytes, meta says {bytes}", b.len()));
    }
    Ok(b)
}

fn load_capture(dir: &Path) -> Result<Capture> {
    let meta_bytes = std::fs::read(dir.join("meta.json")).map_err(|e| format!("meta.json: {e}"))?;
    let meta: Value = serde_json::from_slice(&meta_bytes).map_err(err)?;
    must_eq(
        "capture schema",
        jstr(&meta, "schema")?,
        "pmdt-capture-v1".to_owned(),
    )?;
    must_eq("capture symbol", jstr(&meta, "symbol")?, SYMBOL.to_owned())?;
    must_eq("capture arch", jstr(&meta, "arch")?, "gfx1201".to_owned())?;
    let kernarg = unhex(&jstr(&meta, "kernarg_hex")?)?;
    must_eq("captured kernarg length", kernarg.len(), KERNARG_LEN)?;
    let m = rd_u32(&kernarg, OFF_M) as usize;
    let k = rd_u32(&kernarg, OFF_K) as usize;
    if m == 0 || k == 0 || k % 256 != 0 {
        return Err(format!("capture M={m} K={k} outside the MQ4 domain"));
    }
    let read = |off: usize, tag: &str| read_hashed(dir, &meta, &format!("arg{off}.{tag}.bin"));
    let pre = [read(0, "pre")?, read(8, "pre")?, read(16, "pre")?];
    let post = [read(0, "post")?, read(8, "post")?, read(16, "post")?];
    let lens = [m * (k / 256) * GROUP_BYTES, k * 4, m * 4];
    for i in 0..3 {
        must_eq(
            &format!("arg{} pre length", PTR_OFF[i]),
            pre[i].len(),
            lens[i],
        )?;
        must_eq(
            &format!("arg{} post length", PTR_OFF[i]),
            post[i].len(),
            lens[i],
        )?;
    }
    Ok(Capture {
        dir: dir.to_owned(),
        meta_sha256: sha256_hex(&meta_bytes),
        grid: jarr3(&meta, "grid")?,
        block: jarr3(&meta, "block")?,
        lds: meta["dynamic_lds"].as_u64().unwrap_or(0) as u32,
        kernarg,
        m,
        k,
        pre,
        post,
        meta,
    })
}

/// The fixture must be deterministic, real and non-degenerate; otherwise a
/// "pass" would be a statement about constant or zero data.
fn fixture_quality(c: &Capture) -> Result<Value> {
    let mut seen = [false; 256];
    c.pre[0].iter().for_each(|&b| seen[b as usize] = true);
    let a_distinct = seen.iter().filter(|&&s| s).count();
    let a_nonzero = c.pre[0].iter().filter(|&&b| b != 0).count() as f64 / c.pre[0].len() as f64;
    let x = f32s(&c.pre[1]);
    let x_finite = x.iter().all(|v| v.is_finite());
    let x_distinct = x.iter().map(|v| v.to_bits()).collect::<HashSet<_>>().len();
    let y0 = f32s(&c.pre[2]);
    let y1 = f32s(&c.post[2]);
    let y_pre_finite = y0.iter().all(|v| v.is_finite());
    let y_post_finite = y1.iter().all(|v| v.is_finite());
    let y_pre_distinct = y0.iter().map(|v| v.to_bits()).collect::<HashSet<_>>().len();
    let y_pre_nonzero = y0.iter().filter(|v| v.to_bits() != 0).count();
    let y_rows_changed = y0
        .iter()
        .zip(&y1)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let a_unchanged = c.pre[0] == c.post[0];
    let x_unchanged = c.pre[1] == c.post[1];
    let v = json!({
        "capture": c.dir.display().to_string(),
        "meta_sha256": c.meta_sha256,
        "M": c.m, "K": c.k, "groups": c.k / 256, "grid": c.grid, "block": c.block, "dynamic_lds": c.lds,
        "kernarg_hex": hex(&c.kernarg),
        "model": c.meta["model"], "position": c.meta["position"], "launch_index": c.meta["launch_index"],
        "occurrence_ordinal": c.meta["occurrence_ordinal"],
        "A": {"bytes": c.pre[0].len(), "sha256": sha256_hex(&c.pre[0]), "distinct_byte_values": a_distinct, "nonzero_byte_fraction": a_nonzero, "unchanged_by_callsite": a_unchanged},
        "x": {"bytes": c.pre[1].len(), "sha256": sha256_hex(&c.pre[1]), "finite": x_finite, "distinct_f32": x_distinct, "unchanged_by_callsite": x_unchanged},
        "y": {"bytes": c.pre[2].len(), "pre_sha256": sha256_hex(&c.pre[2]), "post_sha256": sha256_hex(&c.post[2]),
              "pre_finite": y_pre_finite, "post_finite": y_post_finite, "pre_distinct_f32": y_pre_distinct,
              "pre_nonzero": y_pre_nonzero, "rows_changed_by_callsite": y_rows_changed},
    });
    let ok = a_distinct >= 128
        && a_nonzero > 0.9
        && x_finite
        && x_distinct >= 1024usize.min(x.len())
        && y_pre_finite
        && y_post_finite
        && y_pre_distinct >= 64usize.min(y0.len())
        && y_pre_nonzero > 0
        && y_rows_changed >= c.m / 2
        && a_unchanged
        && x_unchanged;
    if !ok {
        return Err(format!(
            "fixture is degenerate or not a pure A/x-read RMW-y callsite: {v}"
        ));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Shapes: the fixture and real-prefix derived odd/tail shapes

struct Shape {
    name: String,
    m: usize,
    k: usize,
    grid: [u32; 3],
    block: [u32; 3],
    lds: u32,
    /// Captured-ABI kernarg template (pointer slots zero).
    kernarg: Vec<u8>,
    pre: [Vec<u8>; 3],
    /// Captured post-launch A/x/y of the real callsite (fixture only).
    callsite_post: Option<[Vec<u8>; 3]>,
    /// Which part of the Vulkan receipt holds this shape's reference (fixture =
    /// receipt top level, derived = `derived.<key>`), or `NotRun`.
    vk: VkScope,
    derived: bool,
}

#[derive(Clone, PartialEq)]
enum VkScope {
    Fixture,
    Derived(String),
    NotRun,
}

impl Shape {
    fn lens(&self) -> [usize; 3] {
        [
            self.m * (self.k / 256) * GROUP_BYTES,
            self.k * 4,
            self.m * 4,
        ]
    }
    fn masked_kernarg(&self) -> Vec<u8> {
        let mut k = self.kernarg.clone();
        PTR_OFF.iter().for_each(|&o| wr_u64(&mut k, o, 0));
        k
    }
}

fn fixture_shape(c: &Capture) -> Shape {
    let mut kernarg = c.kernarg.clone();
    PTR_OFF.iter().for_each(|&o| wr_u64(&mut kernarg, o, 0));
    Shape {
        name: format!("fixture_m{}_k{}", c.m, c.k),
        m: c.m,
        k: c.k,
        grid: c.grid,
        block: c.block,
        lds: c.lds,
        kernarg,
        pre: c.pre.clone(),
        callsite_post: Some(c.post.clone()),
        vk: VkScope::Fixture,
        derived: false,
    }
}

/// Real-prefix derived odd/tail shape: first `m` rows, first `groups` groups of
/// each row. The A bytes are CROPPED row-wise (`groups*136` of each original
/// `G*136` row), x and y are prefixes; every arm receives the identical bytes.
fn derived_shape(c: &Capture, m: usize, groups: usize, minimal_grid: bool) -> Result<Shape> {
    let g_full = c.k / 256;
    if m == 0 || m > c.m || groups == 0 || groups > g_full {
        return Err(format!(
            "derived shape M={m} groups={groups} outside the fixture"
        ));
    }
    if !minimal_grid && c.grid != [c.m as u32, 1, 1] {
        return Err(format!(
            "captured launcher rule is not grid==M (grid {:?}, M {}); refusing to infer a derived grid",
            c.grid, c.m
        ));
    }
    let row_full = g_full * GROUP_BYTES;
    let row_crop = groups * GROUP_BYTES;
    let mut a = Vec::with_capacity(m * row_crop);
    for r in 0..m {
        a.extend_from_slice(&c.pre[0][r * row_full..r * row_full + row_crop]);
    }
    let x = c.pre[1][..groups * 1024].to_vec();
    let y = c.pre[2][..m * 4].to_vec();
    let mut kernarg = c.kernarg.clone();
    PTR_OFF.iter().for_each(|&o| wr_u64(&mut kernarg, o, 0));
    wr_u32(&mut kernarg, OFF_M, m as u32);
    wr_u32(&mut kernarg, OFF_K, (groups * 256) as u32);
    let grid_x = if minimal_grid {
        m.div_ceil(2) as u32
    } else {
        m as u32
    };
    let key = format!("odd_tail_m{m}_k{}", groups * 256);
    Ok(Shape {
        name: format!(
            "{key}.{}",
            if minimal_grid {
                "minimal_grid"
            } else {
                "captured_grid"
            }
        ),
        m,
        k: groups * 256,
        grid: [grid_x, 1, 1],
        block: c.block,
        lds: c.lds,
        kernarg,
        pre: [a, x, y],
        callsite_post: None,
        vk: if minimal_grid {
            VkScope::NotRun
        } else {
            VkScope::Derived(key)
        },
        derived: true,
    })
}

fn shapes_for(c: &Capture) -> Result<Vec<Shape>> {
    let g = c.k / 256;
    let mut v = vec![fixture_shape(c)];
    if c.m >= 2 && g >= 2 {
        v.push(derived_shape(c, c.m - 1, g - 1, false)?);
        v.push(derived_shape(c, c.m - 1, g - 1, true)?);
    }
    Ok(v)
}

fn framed(pre: &[u8]) -> Vec<u8> {
    let mut v = vec![GUARD_BYTE; GUARD + pre.len() + GUARD];
    v[GUARD..GUARD + pre.len()].copy_from_slice(pre);
    v
}

// ---------------------------------------------------------------------------
// Inventory incumbent (original hipcc) and accepted native twin

struct Incumbent {
    source: String,
    source_sha256: String,
    object_path: PathBuf,
    object_sha256: String,
    flags: Value,
}

fn incumbent(inventory: &Path) -> Result<Incumbent> {
    let inv: Value =
        serde_json::from_slice(&std::fs::read(inventory).map_err(err)?).map_err(err)?;
    let m = inv["modules"]
        .as_array()
        .ok_or("inventory.modules missing")?
        .iter()
        .find(|m| m["module"] == MODULE)
        .ok_or_else(|| format!("{MODULE} not in inventory"))?;
    let snap = m["source_snapshot"]
        .as_str()
        .ok_or("source_snapshot missing")?;
    let source = std::fs::read_to_string(snap).map_err(err)?;
    let want = m["source_sha256"].as_str().ok_or("source_sha256 missing")?;
    let got = sha256_hex(source.as_bytes());
    if got != want {
        return Err(format!("{snap}: sha256 {got} != inventory {want}"));
    }
    let flags = m["closure"]["flags"].clone();
    let expect = json!([
        "--genco",
        "--offload-arch=gfx1201",
        "-O3",
        "--no-offload-compress",
        "-fuse-cuid=none",
        "-mcode-object-version=6",
        "-DIU4_A4_CANDIDATES=2"
    ]);
    if flags != expect {
        return Err(format!("{MODULE}: unexpected recipe flags {flags}"));
    }
    let kernarg_size = m["kernel_contracts"][SYMBOL]["abi"][".kernarg_segment_size"]
        .as_u64()
        .ok_or("kernarg_segment_size missing")? as usize;
    must_eq("inventory kernarg_segment_size", kernarg_size, KERNARG_LEN)?;
    Ok(Incumbent {
        source,
        source_sha256: got,
        object_path: PathBuf::from(m["object_path"].as_str().ok_or("object_path missing")?),
        object_sha256: m["object_sha256"]
            .as_str()
            .ok_or("object_sha256 missing")?
            .to_owned(),
        flags,
    })
}

// ---------------------------------------------------------------------------
// Arms

struct Arm {
    label: &'static str,
    kind: String,
    _module: Module,
    func: Function,
    symbol: String,
    image_sha256: String,
    path: Option<PathBuf>,
    extra: Value,
}

fn load_arm(
    gpu: &Gpu,
    label: &'static str,
    kind: &str,
    image: &[u8],
    symbol: &str,
    path: Option<&Path>,
    extra: Value,
) -> Result<Arm> {
    let module = gpu
        .hip
        .module_load_data(image)
        .map_err(|e| format!("{label}: module_load_data: {e}"))?;
    let func = gpu
        .hip
        .module_get_function(&module, symbol)
        .map_err(|e| format!("{label}: symbol {symbol}: {e}"))?;
    Ok(Arm {
        label,
        kind: kind.to_owned(),
        _module: module,
        func,
        symbol: symbol.to_owned(),
        image_sha256: sha256_hex(image),
        path: path.map(Path::to_owned),
        extra,
    })
}

fn arm_identity(a: &Arm, sh: &Shape) -> Value {
    json!({
        "arm": a.label, "kind": a.kind, "symbol": a.symbol, "image_sha256": a.image_sha256,
        "path": a.path.as_ref().map(|p| p.display().to_string()),
        "abi_kernarg_bytes": KERNARG_LEN,
        "abi_kernarg_ptr_masked_hex": hex(&sh.masked_kernarg()),
        "abi_kernarg_ptr_masked_sha256": sha256_hex(&sh.masked_kernarg()),
        "detail": a.extra,
    })
}

fn load_hipcc(gpu: &Gpu, inc: &Incumbent) -> Result<Arm> {
    let obj = std::fs::read(&inc.object_path)
        .map_err(|e| format!("{}: {e}", inc.object_path.display()))?;
    if sha256_hex(&obj) != inc.object_sha256 {
        return Err(format!(
            "{}: object sha256 mismatch against inventory",
            inc.object_path.display()
        ));
    }
    load_arm(
        gpu,
        "hipcc",
        "original hipcc object (inventory recipe)",
        &obj,
        SYMBOL,
        Some(&inc.object_path),
        json!({"source_sha256": inc.source_sha256, "recipe_flags": inc.flags, "inventory_object_sha256": inc.object_sha256}),
    )
}

/// The accepted native twin through the repo admission (`pm_decode_twins`):
/// exact gfx1201, flag-style selection, pinned image digest, pinned incumbent
/// source digest, accepted export. Fails closed.
fn load_twin(gpu: &Gpu, inc: &Incumbent) -> Result<Arm> {
    let twin = rdna_compute::pm_decode_twins::pm_decode_module(&gpu.arch, true, MODULE)
        .ok_or_else(|| format!("no accepted pm_decode twin for {MODULE} on {}", gpu.arch))?;
    twin.verify(&inc.source, SYMBOL)?;
    load_arm(
        gpu,
        "twin",
        "accepted PM native twin (repo pm_decode_twins admission)",
        twin.image,
        SYMBOL,
        None,
        json!({
            "module": twin.module,
            "image_sha256": hex(&twin.image_sha256),
            "accepted_elf_sha256": hex(&twin.accepted_elf_sha256),
            "source_sha256": hex(&twin.source_sha256),
            "symbols": twin.symbols,
        }),
    )
}

// ---------------------------------------------------------------------------
// Candidate manifest (aco-pm-candidate-v1) and Vulkan receipt (aco-vulkan-ref-v1)

struct Manifest {
    json: Value,
    sha256: String,
    odd_tail: bool,
    object_sha256: String,
    spirv_sha256: String,
    aco_binary_sha256: String,
    aco_sidecar_sha256: String,
}

fn load_manifest(
    path: &Path,
    cand_sha: &str,
    symbol: &str,
    cap: &Capture,
    kernarg_abi_sha256: &str,
) -> Result<Manifest> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("candidate manifest {}: {e}", path.display()))?;
    let j: Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("candidate manifest: {e}"))?;
    let ctx = |e: String| format!("candidate manifest {}: {e}", path.display());
    let pre = || -> Result<()> {
        must_eq("schema", jstr(&j, "schema")?, MANIFEST_SCHEMA.to_owned())?;
        must_eq("symbol", jstr(&j, "symbol")?, symbol.to_owned())?;
        must_eq(
            "object_sha256 vs --candidate bytes",
            jhex64(&j, "object_sha256")?,
            cand_sha.to_owned(),
        )?;
        for f in [
            "spirv_sha256",
            "aco_binary_sha256",
            "aco_sidecar_sha256",
            "body_sha256",
        ] {
            jhex64(&j, f)?;
        }
        jstr(&j, "adapter_version")?;
        must_eq("wave_size", ju64(&j, "wave_size")?, 32)?;
        must_eq("block", jarr3(&j, "block")?, cap.block)?;
        must_eq(
            "kernarg_size",
            ju64(&j, "kernarg_size")? as usize,
            KERNARG_LEN,
        )?;
        must_eq(
            "kernarg_abi_sha256",
            jhex64(&j, "kernarg_abi_sha256")?,
            kernarg_abi_sha256.to_owned(),
        )?;
        must_eq(
            "shape_support.fixture",
            jget(&j, "shape_support.fixture")?.as_bool(),
            Some(true),
        )?;
        jget(&j, "shape_support.odd_tail")?
            .as_bool()
            .ok_or_else(|| "field `shape_support.odd_tail` must be a boolean".to_owned())?;
        Ok(())
    };
    pre().map_err(ctx)?;
    Ok(Manifest {
        sha256: sha256_hex(&bytes),
        odd_tail: j["shape_support"]["odd_tail"].as_bool().unwrap(),
        object_sha256: jhex64(&j, "object_sha256").unwrap(),
        spirv_sha256: jhex64(&j, "spirv_sha256").unwrap(),
        aco_binary_sha256: jhex64(&j, "aco_binary_sha256").unwrap(),
        aco_sidecar_sha256: jhex64(&j, "aco_sidecar_sha256").unwrap(),
        json: j,
    })
}

struct Pins {
    spirv: Option<String>,
    aco_binary: Option<String>,
}

struct VkRef {
    identity: Value,
    /// Per shape name: framed post buffers (exactly `GUARD + len + GUARD`).
    posts: Vec<(String, [Vec<u8>; 3])>,
}

fn vk_buffers(bucket: &Value, base: &Path, sh: &Shape) -> Result<[Vec<u8>; 3]> {
    let lens = sh.lens();
    let mut out: [Vec<u8>; 3] = Default::default();
    for i in 0..3 {
        let key = format!("buffers.arg{}", PTR_OFF[i]);
        let ctx = |e: String| format!("vulkan receipt, shape {}: {key}: {e}", sh.name);
        let want_pre = jhex64(bucket, &format!("{key}.pre_framed_sha256")).map_err(ctx)?;
        let got_pre = sha256_hex(&framed(&sh.pre[i]));
        must_eq(
            "pre_framed_sha256 vs oracle framing of the fixture bytes",
            want_pre,
            got_pre,
        )
        .map_err(ctx)?;
        let file = jstr(bucket, &format!("{key}.post_framed_file")).map_err(ctx)?;
        let want_post = jhex64(bucket, &format!("{key}.post_framed_sha256")).map_err(ctx)?;
        let p = if Path::new(&file).is_absolute() {
            PathBuf::from(&file)
        } else {
            base.join(&file)
        };
        let b = std::fs::read(&p).map_err(|e| ctx(format!("{}: {e}", p.display())))?;
        must_eq("post file length", b.len(), GUARD + lens[i] + GUARD).map_err(ctx)?;
        must_eq("post file sha256 vs receipt", sha256_hex(&b), want_post).map_err(ctx)?;
        out[i] = b;
    }
    Ok(out)
}

fn load_vulkan(
    path: &Path,
    cap: &Capture,
    man: &Manifest,
    shapes: &[Shape],
    pins: &Pins,
) -> Result<VkRef> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("vulkan receipt {}: {e}", path.display()))?;
    let j: Value = serde_json::from_slice(&bytes).map_err(|e| format!("vulkan receipt: {e}"))?;
    let base = path.parent().ok_or("receipt has no parent directory")?;
    let ctx = |e: String| format!("vulkan receipt {}: {e}", path.display());
    let top = || -> Result<()> {
        must_eq("schema", jstr(&j, "schema")?, VK_SCHEMA.to_owned())?;
        must_eq("symbol", jstr(&j, "symbol")?, SYMBOL.to_owned())?;
        let spirv = jhex64(&j, "spirv.sha256")?;
        must_eq(
            "spirv.sha256 vs candidate manifest",
            spirv.clone(),
            man.spirv_sha256.clone(),
        )?;
        if let Some(p) = &pins.spirv {
            must_eq("spirv.sha256 vs --expect-spirv-sha256", spirv, p.clone())?;
        }
        let bin = jhex64(&j, "aco_binary.sha256")?;
        must_eq(
            "aco_binary.sha256 vs candidate manifest",
            bin.clone(),
            man.aco_binary_sha256.clone(),
        )?;
        if let Some(p) = &pins.aco_binary {
            must_eq(
                "aco_binary.sha256 vs --expect-aco-binary-sha256",
                bin,
                p.clone(),
            )?;
        }
        must_eq(
            "aco_binary.sidecar_sha256 vs candidate manifest",
            jhex64(&j, "aco_binary.sidecar_sha256")?,
            man.aco_sidecar_sha256.clone(),
        )?;
        must_eq("config.wave_size", ju64(&j, "config.wave_size")?, 32)?;
        must_eq(
            "config.local_size vs capture block",
            jarr3(&j, "config.local_size")?,
            cap.block,
        )?;
        must_eq(
            "config.workgroups vs capture grid",
            jarr3(&j, "config.workgroups")?,
            cap.grid,
        )?;
        must_eq("config.M vs capture", ju64(&j, "config.M")? as usize, cap.m)?;
        must_eq("config.K vs capture", ju64(&j, "config.K")? as usize, cap.k)?;
        jhex64(&j, "config.specialization_sha256")?;
        jhex64(&j, "config.layout_sha256")?;
        must_eq(
            "fixture.meta_sha256 vs capture meta.json",
            jhex64(&j, "fixture.meta_sha256")?,
            cap.meta_sha256.clone(),
        )?;
        for f in [
            "driver.mesa_version",
            "driver.device_name",
            "driver.driver_id",
        ] {
            jstr(&j, f)?;
        }
        for f in ["driver.vendor_id", "driver.device_id"] {
            if jget(&j, f)?.is_null() {
                return Err(format!("field `{f}` is null"));
            }
        }
        must_eq(
            "framing.guard_bytes",
            ju64(&j, "framing.guard_bytes")? as usize,
            GUARD,
        )?;
        must_eq(
            "framing.guard_byte",
            jstr(&j, "framing.guard_byte")?,
            format!("{GUARD_BYTE:02x}"),
        )?;
        Ok(())
    };
    top().map_err(ctx)?;
    let mut posts = Vec::new();
    for sh in shapes {
        match &sh.vk {
            VkScope::NotRun => continue,
            VkScope::Fixture => posts.push((sh.name.clone(), vk_buffers(&j, base, sh)?)),
            VkScope::Derived(key) => {
                if !man.odd_tail {
                    continue; // explicit candidate refusal is reported by the matrix
                }
                let bucket = j["derived"].get(key).ok_or_else(|| {
                    ctx(format!("candidate claims shape_support.odd_tail but receipt has no `derived.{key}` bucket"))
                })?;
                let c2 = |e: String| format!("vulkan receipt derived.{key}: {e}");
                must_eq(
                    "config.M",
                    ju64(bucket, "config.M").map_err(c2)? as usize,
                    sh.m,
                )
                .map_err(c2)?;
                must_eq(
                    "config.K",
                    ju64(bucket, "config.K").map_err(c2)? as usize,
                    sh.k,
                )
                .map_err(c2)?;
                must_eq(
                    "config.workgroups",
                    jarr3(bucket, "config.workgroups").map_err(c2)?,
                    sh.grid,
                )
                .map_err(c2)?;
                posts.push((sh.name.clone(), vk_buffers(bucket, base, sh)?));
            }
        }
    }
    Ok(VkRef {
        identity: json!({
            "receipt": path.display().to_string(),
            "receipt_sha256": sha256_hex(&bytes),
            "spirv_sha256": j["spirv"]["sha256"], "aco_binary": j["aco_binary"], "config": j["config"],
            "fixture": j["fixture"], "driver": j["driver"],
            "shapes_with_vulkan_reference": posts.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
        }),
        posts,
    })
}

// ---------------------------------------------------------------------------
// VMM-framed buffers, eager launch, hipGraph capture/replay

struct Frame {
    t: GpuTensor,
    len: usize,
    mapped: usize,
}

struct Bufs {
    frames: Vec<Frame>,
    kernarg: Vec<u8>,
}

struct Ctx {
    stream: Stream,
    gran: usize,
}

fn mapped_frame(pre: &[u8], mapped: usize) -> Vec<u8> {
    let mut v = vec![GUARD_BYTE; mapped];
    v[GUARD..GUARD + pre.len()].copy_from_slice(pre);
    v
}

fn upload_frames(gpu: &Gpu, b: &Bufs, pre: &[Vec<u8>; 3]) -> Result<()> {
    for (f, p) in b.frames.iter().zip(pre) {
        gpu.hip
            .memcpy_htod(&f.t.buf, &mapped_frame(p, f.mapped))
            .map_err(err)?;
    }
    Ok(())
}

fn alloc_bufs(gpu: &mut Gpu, cx: &Ctx, sh: &Shape, pre: &[Vec<u8>; 3]) -> Result<Bufs> {
    let access = [gpu.device_id];
    let mut frames = Vec::new();
    let mut kernarg = sh.kernarg.clone();
    for (i, p) in pre.iter().enumerate() {
        let mapped = (GUARD + p.len() + GUARD).next_multiple_of(cx.gran);
        // SAFETY: the tensor is mapped at its full shape; every access below
        // stays inside `mapped` bytes.
        let t =
            unsafe { gpu.alloc_vmm_tensor(&[mapped], DType::Raw, mapped, &access) }.map_err(err)?;
        if t.buf.size() < mapped || !t.buf.is_vmm_owner() {
            return Err(format!(
                "arg{}: VMM allocation of {mapped} bytes not owner-mapped (size {})",
                PTR_OFF[i],
                t.buf.size()
            ));
        }
        wr_u64(
            &mut kernarg,
            PTR_OFF[i],
            t.buf.as_ptr() as u64 + GUARD as u64,
        );
        frames.push(Frame {
            t,
            len: p.len(),
            mapped,
        });
    }
    kernarg.resize(kernarg.len().next_multiple_of(16), 0);
    let b = Bufs { frames, kernarg };
    upload_frames(gpu, &b, pre)?;
    Ok(b)
}

fn free_bufs(gpu: &mut Gpu, b: Bufs) -> Result<()> {
    for f in b.frames {
        gpu.free_tensor(f.t).map_err(err)?;
    }
    Ok(())
}

struct RunOut {
    /// Framed core (`GUARD + len + GUARD`) per argument.
    core: [Vec<u8>; 3],
    /// Everything between the core and the VMM mapping end is still guard bytes.
    tail_clean: bool,
    guards_intact: bool,
    replay_identical: Option<bool>,
}

fn read_out(gpu: &Gpu, b: &Bufs) -> Result<([Vec<u8>; 3], bool, bool)> {
    let mut core: [Vec<u8>; 3] = Default::default();
    let mut tail = true;
    let mut guards = true;
    for (i, f) in b.frames.iter().enumerate() {
        let mut v = vec![0u8; f.mapped];
        gpu.hip.memcpy_dtoh(&mut v, &f.t.buf).map_err(err)?;
        let end = GUARD + f.len + GUARD;
        tail &= v[end..].iter().all(|&x| x == GUARD_BYTE);
        guards &= v[..GUARD]
            .iter()
            .chain(&v[GUARD + f.len..end])
            .all(|&x| x == GUARD_BYTE);
        v.truncate(end);
        core[i] = v;
    }
    Ok((core, tail, guards))
}

fn launch(gpu: &Gpu, a: &Arm, cx: &Ctx, sh: &Shape, kernarg: &mut [u8]) -> Result<()> {
    // SAFETY: the kernarg bytes follow the original 32-byte ABI and every
    // pointer addresses a live VMM allocation owned by this process.
    unsafe {
        gpu.hip.launch_kernel_blob(
            &a.func,
            sh.grid,
            sh.block,
            sh.lds,
            Some(&cx.stream),
            kernarg,
        )
    }
    .map_err(|e| format!("{}: launch {}: {e}", a.label, sh.name))
}

fn sync(gpu: &Gpu, cx: &Ctx) -> Result<()> {
    gpu.hip.device_synchronize().map_err(err)?;
    gpu.hip.stream_synchronize(&cx.stream).map_err(err)
}

fn run_eager(gpu: &mut Gpu, cx: &Ctx, a: &Arm, sh: &Shape, pre: &[Vec<u8>; 3]) -> Result<RunOut> {
    let mut b = alloc_bufs(gpu, cx, sh, pre)?;
    sync(gpu, cx)?;
    launch(gpu, a, cx, sh, &mut b.kernarg)?;
    sync(gpu, cx)?;
    let (core, tail_clean, guards_intact) = read_out(gpu, &b)?;
    free_bufs(gpu, b)?;
    Ok(RunOut {
        core,
        tail_clean,
        guards_intact,
        replay_identical: None,
    })
}

/// Capture the launch into a hipGraph on the oracle stream (the kernarg blob
/// stays alive and unmodified for the life of the graph), instantiate, then
/// replay twice, each time from freshly reset buffers.
fn run_graph(gpu: &mut Gpu, cx: &Ctx, a: &Arm, sh: &Shape, pre: &[Vec<u8>; 3]) -> Result<RunOut> {
    let mut b = alloc_bufs(gpu, cx, sh, pre)?;
    sync(gpu, cx)?;
    let dev = gpu.device_id;
    gpu.graphs
        .begin_graph_capture(&gpu.hip, dev, &cx.stream)
        .map_err(err)?;
    if let Err(e) = launch(gpu, a, cx, sh, &mut b.kernarg) {
        gpu.graphs.abort_graph_capture(&gpu.hip, dev, &cx.stream);
        return Err(e);
    }
    gpu.graphs
        .end_graph_capture(&gpu.hip, dev, &cx.stream)
        .map_err(err)?;
    // Capture does not execute; reset anyway so replay starts from the pre bytes.
    sync(gpu, cx)?;
    upload_frames(gpu, &b, pre)?;
    sync(gpu, cx)?;
    gpu.graphs
        .graph_launch(&gpu.hip, dev, &cx.stream)
        .map_err(err)?;
    sync(gpu, cx)?;
    let (core1, tail1, guards1) = read_out(gpu, &b)?;
    upload_frames(gpu, &b, pre)?;
    sync(gpu, cx)?;
    gpu.graphs
        .graph_launch(&gpu.hip, dev, &cx.stream)
        .map_err(err)?;
    sync(gpu, cx)?;
    let (core2, tail2, guards2) = read_out(gpu, &b)?;
    gpu.graphs.drop_captured_graph(&gpu.hip, dev);
    free_bufs(gpu, b)?;
    Ok(RunOut {
        replay_identical: Some(core1 == core2),
        core: core1,
        tail_clean: tail1 && tail2,
        guards_intact: guards1 && guards2,
    })
}

// ---------------------------------------------------------------------------
// Comparison / checks

struct Cmp {
    bytes: usize,
    mismatches: usize,
    first: Option<usize>,
}

fn compare(a: &[u8], b: &[u8]) -> Cmp {
    let mut c = Cmp {
        bytes: a.len().max(b.len()),
        mismatches: 0,
        first: None,
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

fn cmp_bufs(a: &[Vec<u8>; 3], b: &[Vec<u8>; 3], labels: (&str, &str)) -> (usize, Vec<Value>) {
    let mut total = 0;
    let mut rows = Vec::new();
    for i in 0..3 {
        let c = compare(&a[i], &b[i]);
        total += c.mismatches;
        let mut v = json!({"buffer": format!("arg{}:{}+guards", PTR_OFF[i], ARG_NAME[i]), "bytes": c.bytes, "mismatch_bytes": c.mismatches});
        if let Some(f) = c.first {
            v["first_index"] = json!(f);
            v["window_base"] = json!(f & !15);
            v[format!("{}_hex", labels.0)] = json!(window(&a[i], f));
            v[format!("{}_hex", labels.1)] = json!(window(&b[i], f));
            if i == 2 && f >= GUARD && f < a[i].len().min(b[i].len()).saturating_sub(GUARD) {
                v["y_row"] = json!((f - GUARD) / 4);
            }
        }
        rows.push(v);
    }
    (total, rows)
}

struct Matrix {
    failures: usize,
    entries: Vec<Value>,
}

impl Matrix {
    fn new() -> Self {
        Matrix {
            failures: 0,
            entries: Vec::new(),
        }
    }

    /// Whole-buffer byte identity (expected: zero differing bytes).
    fn eq(&mut self, name: &str, a: &[Vec<u8>; 3], b: &[Vec<u8>; 3], labels: (&str, &str)) {
        let (n, rows) = cmp_bufs(a, b, labels);
        self.failures += n;
        self.entries.push(
            json!({"check": name, "kind": "identical", "mismatch_bytes": n, "buffers": rows}),
        );
    }

    fn flag(&mut self, name: &str, ok: bool, detail: Value) {
        if !ok {
            self.failures += 1;
        }
        self.entries
            .push(json!({"check": name, "kind": "flag", "ok": ok, "detail": detail}));
    }

    /// Negative control: the compared buffers MUST differ.
    fn detect(
        &mut self,
        name: &str,
        a: &[Vec<u8>; 3],
        b: &[Vec<u8>; 3],
        labels: (&str, &str),
        require_y: bool,
    ) {
        let (n, rows) = cmp_bufs(a, b, labels);
        let y_diff = rows[2]["mismatch_bytes"].as_u64().unwrap_or(0);
        let ok = n > 0 && (!require_y || y_diff > 0);
        if !ok {
            self.failures += 1;
        }
        self.entries.push(json!({"check": name, "kind": "negative_control", "detected": ok, "mismatch_bytes": n, "y_mismatch_bytes": y_diff, "buffers": rows}));
    }
}

fn expected_callsite(sh: &Shape) -> Option<[Vec<u8>; 3]> {
    let post = sh.callsite_post.as_ref()?;
    Some([framed(&sh.pre[0]), framed(&sh.pre[1]), framed(&post[2])])
}

fn xor_edit(pre: &[Vec<u8>; 3], edits: &[(usize, usize, u8)]) -> [Vec<u8>; 3] {
    let mut p = pre.clone();
    for &(arg, off, x) in edits {
        p[arg][off] ^= x;
    }
    p
}

fn run_check(m: &mut Matrix, name: &str, r: &Result<RunOut>) -> bool {
    match r {
        Ok(o) => {
            m.flag(
                &format!("{name}: guards intact (pre/post guard bytes untouched)"),
                o.guards_intact,
                json!(null),
            );
            m.flag(
                &format!("{name}: VMM mapping tail beyond the frame untouched"),
                o.tail_clean,
                json!(null),
            );
            if let Some(same) = o.replay_identical {
                m.flag(
                    &format!("{name}: graph replay #2 from reset buffers == replay #1"),
                    same,
                    json!(null),
                );
            }
            true
        }
        Err(e) => {
            m.flag(&format!("{name}: run"), false, json!(e));
            false
        }
    }
}

// ---------------------------------------------------------------------------
// full / selftest

struct Setup<'a> {
    cap: &'a Capture,
    hipcc: Arm,
    twin: Arm,
    cand: Arm,
    vk: Option<&'a VkRef>,
    cand_odd_tail: bool,
}

fn parity(gpu: &mut Gpu, cx: &Ctx, s: &Setup) -> Result<(usize, Value)> {
    let shapes = shapes_for(s.cap)?;
    let mut shape_reports = Vec::new();
    let mut total_failures = 0;
    let mut odd_tail_claim = "not_applicable".to_owned();
    for sh in &shapes {
        let mut m = Matrix::new();
        let mut arms_run: Vec<Value> = Vec::new();
        let ref_run = run_eager(gpu, cx, &s.hipcc, sh, &sh.pre);
        if !run_check(&mut m, "hipcc eager", &ref_run) {
            total_failures += m.failures;
            shape_reports
                .push(json!({"shape": sh.name, "matrix": m.entries, "failures": m.failures}));
            continue;
        }
        let ref_run = ref_run.unwrap();
        if let Some(exp) = expected_callsite(sh) {
            m.eq(
                "hipcc eager vs captured real callsite (A/x pre==post, y post)",
                &ref_run.core,
                &exp,
                ("hipcc", "captured"),
            );
        }
        let mut runs: Vec<(&Arm, &str, Result<RunOut>)> = Vec::new();
        let arms: [&Arm; 3] = [&s.hipcc, &s.twin, &s.cand];
        for a in arms {
            let candidate_here = std::ptr::eq(a, &s.cand);
            if candidate_here && sh.derived && !s.cand_odd_tail {
                m.entries.push(json!({
                    "check": format!("candidate on {}", sh.name), "kind": "unsupported_refusal",
                    "result": "UNSUPPORTED: candidate manifest shape_support.odd_tail=false; odd/tail parity is NOT claimed and NOT run for the candidate",
                }));
                odd_tail_claim = "refused_unsupported".to_owned();
                continue;
            }
            if a.label != "hipcc" {
                runs.push((a, "eager", run_eager(gpu, cx, a, sh, &sh.pre)));
            }
            runs.push((a, "graph", run_graph(gpu, cx, a, sh, &sh.pre)));
        }
        if sh.derived && s.cand_odd_tail {
            odd_tail_claim = "claimed_and_checked".to_owned();
        }
        let vk_post =
            s.vk.and_then(|v| v.posts.iter().find(|(n, _)| *n == sh.name))
                .map(|(_, p)| p);
        for (a, mode, r) in &runs {
            let name = format!("{} {mode}", a.label);
            if !run_check(&mut m, &name, r) {
                continue;
            }
            let r = r.as_ref().unwrap();
            m.eq(
                &format!("{name} vs hipcc eager"),
                &r.core,
                &ref_run.core,
                (a.label, "hipcc"),
            );
            if a.label == "candidate" {
                if let Some(vp) = vk_post {
                    m.eq(
                        &format!("{name} vs original RADV/Vulkan reference"),
                        &r.core,
                        vp,
                        (a.label, "vulkan"),
                    );
                }
            }
            arms_run.push(json!({"arm": a.label, "mode": mode, "identity": arm_identity(a, sh)}));
        }
        if let Some(vp) = vk_post {
            m.eq(
                "original RADV/Vulkan reference vs hipcc eager",
                vp,
                &ref_run.core,
                ("vulkan", "hipcc"),
            );
        }
        if !sh.derived {
            // Negative controls on the real fixture (candidate arm, eager).
            {
                let g = s.cap.k / 256;
                let row_bytes = g * GROUP_BYTES;
                let invert = |row: usize, group: usize| -> Vec<(usize, usize, u8)> {
                    (0..128)
                        .map(|i| (0, row * row_bytes + group * GROUP_BYTES + 8 + i, 0xFFu8))
                        .collect()
                };
                let w_edits: Vec<(&str, Vec<(usize, usize, u8)>)> = vec![
                    ("wrong weight: byte 8 of A xor 0x10", vec![(0, 8, 0x10)]),
                    (
                        "wrong weight: all nibbles of row 0 group 0 inverted",
                        invert(0, 0),
                    ),
                    (
                        "wrong weight: all nibbles of mid row/group inverted",
                        invert(s.cap.m / 2, g / 2),
                    ),
                    (
                        "wrong weight: all nibbles of last row/last group inverted",
                        invert(s.cap.m - 1, g - 1),
                    ),
                    ("wrong input x: byte 0 xor 0x10", vec![(1, 0, 0x10)]),
                    (
                        "wrong input x: last byte xor 0x10",
                        vec![(1, s.cap.k * 4 - 1, 0x10)],
                    ),
                ];
                let vk_ref = vk_post;
                for (label, edits) in w_edits {
                    let pre = xor_edit(&sh.pre, &edits);
                    match run_eager(gpu, cx, &s.cand, sh, &pre) {
                        Ok(r) => {
                            m.detect(
                                &format!("control {label}: candidate vs hipcc"),
                                &r.core,
                                &ref_run.core,
                                ("candidate", "hipcc"),
                                label.starts_with("wrong weight: all")
                                    || label.contains("last byte"),
                            );
                            if let Some(vp) = vk_ref {
                                m.detect(
                                    &format!("control {label}: candidate vs Vulkan"),
                                    &r.core,
                                    vp,
                                    ("candidate", "vulkan"),
                                    false,
                                );
                            }
                        }
                        Err(e) => m.flag(&format!("control {label}: run"), false, json!(e)),
                    }
                }
                // Wrong output (post-hoc tamper of the candidate's real eager result).
                match run_eager(gpu, cx, &s.cand, sh, &sh.pre) {
                    Ok(r) => {
                        let tamper = |f: &dyn Fn(&mut [Vec<u8>; 3])| -> [Vec<u8>; 3] {
                            let mut c = r.core.clone();
                            f(&mut c);
                            c
                        };
                        let y_first = tamper(&|c| c[2][GUARD] ^= 0x01);
                        let y_last = tamper(&|c| {
                            let at = GUARD + s.cap.m * 4 - 1;
                            c[2][at] ^= 0x80;
                        });
                        let a_unchanged = tamper(&|c| {
                            let at = GUARD + c[0].len() / 2;
                            c[0][at] ^= 0x01;
                        });
                        let x_unchanged = tamper(&|c| c[1][GUARD] ^= 0x01);
                        let guard_pre = tamper(&|c| c[2][0] ^= 0x01);
                        let guard_post = tamper(&|c| {
                            let at = c[0].len() - 1;
                            c[0][at] ^= 0x01;
                        });
                        for (label, t) in [
                            ("wrong output: y[0] low bit", &y_first),
                            ("wrong output: y[M-1] sign bit", &y_last),
                            ("corrupted unchanged input A byte", &a_unchanged),
                            ("corrupted unchanged input x byte", &x_unchanged),
                            ("corrupted y pre-guard byte", &guard_pre),
                            ("corrupted A post-guard byte", &guard_post),
                        ] {
                            m.detect(
                                &format!("control {label}: vs hipcc"),
                                t,
                                &ref_run.core,
                                ("tampered", "hipcc"),
                                false,
                            );
                            if let Some(vp) = vk_ref {
                                m.detect(
                                    &format!("control {label}: vs Vulkan"),
                                    t,
                                    vp,
                                    ("tampered", "vulkan"),
                                    false,
                                );
                            }
                        }
                        if let Some(vp) = vk_ref {
                            let mut tv = vp.clone();
                            tv[2][GUARD + 4] ^= 0x01;
                            m.detect(
                                "control tampered Vulkan reference y[1] vs candidate",
                                &tv,
                                &r.core,
                                ("vulkan_tampered", "candidate"),
                                false,
                            );
                        }
                    }
                    Err(e) => m.flag("control wrong-output base run", false, json!(e)),
                }
            }
        }
        total_failures += m.failures;
        shape_reports.push(json!({
            "shape": sh.name, "M": sh.m, "K": sh.k, "grid": sh.grid, "block": sh.block, "dynamic_lds": sh.lds,
            "derived_from_real_prefix": sh.derived,
            "vulkan_scope": match &sh.vk { VkScope::Fixture => "fixture", VkScope::Derived(_) => "derived", VkScope::NotRun => "not_run (minimal-grid variant; Vulkan runs the captured launcher rule)" },
            "arms_run": arms_run, "failures": m.failures, "matrix": m.entries,
        }));
    }
    Ok((
        total_failures,
        json!({
            "shapes": shape_reports,
            "odd_tail_claim": odd_tail_claim,
            "failures": total_failures,
        }),
    ))
}

// ---------------------------------------------------------------------------
// time (diagnostic, one pass)

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn stats(v: &[f64]) -> Value {
    let mut s = v.to_vec();
    let med = median(&mut s);
    json!({"samples_ms_per_launch": v, "min": s[0], "median": med, "max": s[s.len() - 1]})
}

struct TimeCfg {
    iters: usize,
    reps: usize,
    warmup: usize,
    cold: bool,
}

fn time_sample(
    gpu: &mut Gpu,
    cx: &Ctx,
    a: &Arm,
    sh: &Shape,
    kernarg: &mut [u8],
    flush: Option<&GpuTensor>,
    cfg: &TimeCfg,
) -> Result<f64> {
    let t0 = gpu.hip.event_create().map_err(err)?;
    let t1 = gpu.hip.event_create().map_err(err)?;
    let ms = if let Some(fl) = flush {
        let mut pairs = Vec::new();
        for i in 0..cfg.iters {
            gpu.hip
                .memset_async(&fl.buf, (i & 0x7f) as i32, fl.buf.size(), &cx.stream)
                .map_err(err)?;
            let s = gpu.hip.event_create().map_err(err)?;
            let e = gpu.hip.event_create().map_err(err)?;
            gpu.hip.event_record(&s, Some(&cx.stream)).map_err(err)?;
            launch(gpu, a, cx, sh, kernarg)?;
            gpu.hip.event_record(&e, Some(&cx.stream)).map_err(err)?;
            pairs.push((s, e));
        }
        sync(gpu, cx)?;
        let mut v = Vec::new();
        for (s, e) in pairs {
            v.push(gpu.hip.event_elapsed_ms(&s, &e).map_err(err)? as f64);
            gpu.hip.event_destroy(s).map_err(err)?;
            gpu.hip.event_destroy(e).map_err(err)?;
        }
        median(&mut v)
    } else {
        gpu.hip.event_record(&t0, Some(&cx.stream)).map_err(err)?;
        for _ in 0..cfg.iters {
            launch(gpu, a, cx, sh, kernarg)?;
        }
        gpu.hip.event_record(&t1, Some(&cx.stream)).map_err(err)?;
        sync(gpu, cx)?;
        gpu.hip.event_elapsed_ms(&t0, &t1).map_err(err)? as f64 / cfg.iters as f64
    };
    gpu.hip.event_destroy(t0).map_err(err)?;
    gpu.hip.event_destroy(t1).map_err(err)?;
    Ok(ms)
}

fn md5_of(path: &Path) -> Value {
    match std::process::Command::new("md5sum").arg(path).output() {
        Ok(o) if o.status.success() => json!(String::from_utf8_lossy(&o.stdout)
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_owned()),
        _ => json!("unavailable: md5sum failed"),
    }
}

/// Rep-major timing of `arms` on one shape; every arm gets its own freshly allocated pre-image
/// frames. With `alternate`, odd reps run the arms in reverse order. Returns per-arm samples
/// (indexed like `arms`) and the executed arm order of every rep.
fn time_arms(
    gpu: &mut Gpu,
    cx: &Ctx,
    arms: &[&Arm],
    sh: &Shape,
    cfg: &TimeCfg,
    alternate: bool,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<&'static str>>)> {
    let mut bufs: Vec<Bufs> = Vec::new();
    for _ in arms {
        bufs.push(alloc_bufs(gpu, cx, sh, &sh.pre)?);
    }
    let flush = if cfg.cold {
        let bytes = (256usize << 20).next_multiple_of(cx.gran);
        let access = [gpu.device_id];
        // SAFETY: mapped at full shape; only memset over `buf.size()`.
        Some(unsafe { gpu.alloc_vmm_tensor(&[bytes], DType::Raw, bytes, &access) }.map_err(err)?)
    } else {
        None
    };
    // Warmup (untimed), per arm.
    for (i, a) in arms.iter().enumerate() {
        for _ in 0..cfg.warmup {
            launch(gpu, a, cx, sh, &mut bufs[i].kernarg)?;
        }
        sync(gpu, cx)?;
    }
    let mut full: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
    let mut order: Vec<Vec<&'static str>> = Vec::new();
    for rep in 0..cfg.reps {
        let mut idx: Vec<usize> = (0..arms.len()).collect();
        if alternate && rep % 2 == 1 {
            idx.reverse();
        }
        for &i in &idx {
            let ms = time_sample(
                gpu,
                cx,
                arms[i],
                sh,
                &mut bufs[i].kernarg,
                flush.as_ref(),
                cfg,
            )?;
            full[i].push(ms);
        }
        order.push(idx.iter().map(|&i| arms[i].label).collect());
    }
    for b in bufs {
        free_bufs(gpu, b)?;
    }
    if let Some(f) = flush {
        gpu.free_tensor(f).map_err(err)?;
    }
    Ok((full, order))
}

fn cache_protocol(cfg: &TimeCfg) -> &'static str {
    if cfg.cold {
        "cold: 256 MiB memset before every launch, per-launch event pair"
    } else {
        "warm: back-to-back launches, one event pair per sample"
    }
}

fn run_time(gpu: &mut Gpu, cx: &Ctx, s: &Setup, cfg: &TimeCfg) -> Result<Value> {
    let sh = fixture_shape(s.cap);
    let arms: [&Arm; 3] = [&s.hipcc, &s.twin, &s.cand];
    let (full, _) = time_arms(gpu, cx, &arms, &sh, cfg, false)?;
    let med = |v: &Vec<f64>| {
        let mut c = v.clone();
        median(&mut c)
    };
    let mut per_arm = serde_json::Map::new();
    for (i, a) in arms.iter().enumerate() {
        per_arm.insert(
            a.label.to_owned(),
            json!({"identity": arm_identity(a, &sh), "launch": stats(&full[i])}),
        );
    }
    let ratio = |a: usize, b: usize| med(&full[a]) / med(&full[b]);
    Ok(json!({
        "protocol": {
            "iters_per_sample": cfg.iters, "samples_per_arm": cfg.reps, "warmup_launches": cfg.warmup,
            "cache": cache_protocol(cfg),
            "order": "rep-major single pass: hipcc, twin, candidate per rep; no ABBA, no automatic expansion",
            "geometry": {"grid": sh.grid, "block": sh.block, "dynamic_lds": sh.lds, "M": sh.m, "K": sh.k},
            "prologue_cost_included": "each image is timed whole; any adapter prologue is inside the candidate code object",
            "diagnostic_only": "single-session kernel timing is NOT a product speed claim; product speed needs stock `hipfire bench` per docs/methodology/perf-benchmarking.md",
            "prompt": "not applicable (kernel-only fixture; input hashes in report)",
        },
        "arms": per_arm,
        "ratios_median": {"candidate_over_hipcc": ratio(2, 0), "candidate_over_twin": ratio(2, 1), "twin_over_hipcc": ratio(1, 0)},
    }))
}

// ---------------------------------------------------------------------------
// twin-time: hipcc vs accepted native twin only (no candidate / Vulkan)

/// Whole-buffer + guard parity of hipcc and twin on the captured fixture, eager and hipGraph,
/// each against the captured real post bytes and twin-vs-hipcc. Six byte comparisons are
/// REQUIRED to have run over a nonzero number of bytes.
fn twin_parity(
    gpu: &mut Gpu,
    cx: &Ctx,
    hipcc: &Arm,
    twin: &Arm,
    sh: &Shape,
) -> Result<(usize, Value)> {
    let exp = expected_callsite(sh).ok_or("fixture has no captured callsite post buffers")?;
    let mut m = Matrix::new();
    let mut runs: Vec<(&Arm, &str, RunOut)> = Vec::new();
    for a in [hipcc, twin] {
        for mode in ["eager", "graph"] {
            let r = if mode == "eager" {
                run_eager(gpu, cx, a, sh, &sh.pre)
            } else {
                run_graph(gpu, cx, a, sh, &sh.pre)
            };
            let name = format!("{} {mode}", a.label);
            if run_check(&mut m, &name, &r) {
                let r = r.unwrap();
                m.eq(
                    &format!("{name} vs captured real callsite (A/x pre==post, y post)"),
                    &r.core,
                    &exp,
                    (a.label, "captured"),
                );
                runs.push((a, mode, r));
            }
        }
    }
    let find = |label: &str, mode: &str| {
        runs.iter()
            .find(|(a, md, _)| a.label == label && *md == mode)
            .map(|(_, _, r)| r)
    };
    for mode in ["eager", "graph"] {
        match (find("twin", mode), find("hipcc", mode)) {
            (Some(t), Some(h)) => m.eq(
                &format!("twin {mode} vs hipcc {mode}"),
                &t.core,
                &h.core,
                ("twin", "hipcc"),
            ),
            _ => m.flag(
                &format!("twin {mode} vs hipcc {mode}: both runs available"),
                false,
                json!(null),
            ),
        }
    }
    let (mut n, mut compared, mut differing) = (0usize, 0u64, 0u64);
    for e in m.entries.iter().filter(|e| e["kind"] == "identical") {
        n += 1;
        differing += e["mismatch_bytes"].as_u64().unwrap_or(u64::MAX);
        compared += e["buffers"].as_array().map_or(0, |b| {
            b.iter()
                .map(|x| x["bytes"].as_u64().unwrap_or(0))
                .sum::<u64>()
        });
    }
    m.flag(
        "six whole-buffer comparisons ran over nonzero bytes",
        n == 6 && compared > 0,
        json!({"comparisons": n, "compared_bytes": compared}),
    );
    Ok((
        m.failures,
        json!({
            "comparisons": n, "compared_bytes": compared, "differing_bytes": differing,
            "failures": m.failures, "matrix": m.entries,
        }),
    ))
}

fn run_twin_time(
    gpu: &mut Gpu,
    cx: &Ctx,
    hipcc: &Arm,
    twin: &Arm,
    cap: &Capture,
    cfg: &TimeCfg,
) -> Result<(usize, Value)> {
    let sh = fixture_shape(cap);
    let geometry = json!({
        "name": sh.name, "M": sh.m, "K": sh.k, "grid": sh.grid, "block": sh.block, "dynamic_lds": sh.lds,
        "buffer_bytes": sh.lens(), "kernarg_ptr_masked_sha256": sha256_hex(&sh.masked_kernarg()),
    });
    let (failures, parity) = twin_parity(gpu, cx, hipcc, twin, &sh)?;
    if failures != 0 {
        return Ok((
            failures,
            json!({"geometry": geometry, "parity": parity, "timing": "NOT RUN: hipcc/twin parity failed closed"}),
        ));
    }
    let arms = [hipcc, twin];
    let (full, order) = time_arms(gpu, cx, &arms, &sh, cfg, true)?;
    let mut per_arm = serde_json::Map::new();
    for (i, a) in arms.iter().enumerate() {
        per_arm.insert(
            a.label.to_owned(),
            json!({"identity": arm_identity(a, &sh), "launch": stats(&full[i])}),
        );
    }
    let paired: Vec<f64> = full[1].iter().zip(&full[0]).map(|(t, h)| t / h).collect();
    let med = |v: &Vec<f64>| {
        let mut c = v.clone();
        median(&mut c)
    };
    let mut paired_sorted = paired.clone();
    Ok((
        0,
        json!({
            "geometry": geometry,
            "parity": parity,
            "protocol": {
                "iters_per_sample": cfg.iters, "samples_per_arm": cfg.reps, "warmup_launches": cfg.warmup,
                "cache": cache_protocol(cfg),
                "order": "rep-major, two arms (hipcc, twin), order alternates per rep (even reps hipcc,twin; odd reps twin,hipcc); executed order recorded; no ACO arm, no occupancy probes, no automatic expansion",
                "executed_order_per_rep": order,
                "diagnostic_only": "single-session kernel-only timing at the captured fixture; NOT a product speed claim and NOT the allpm_decode on/off product A/B",
            },
            "arms": per_arm,
            "twin_over_hipcc": {
                "median_of_arm_medians": med(&full[1]) / med(&full[0]),
                "paired_per_rep": paired,
                "paired_per_rep_median": median(&mut paired_sorted),
            },
        }),
    ))
}

// ---------------------------------------------------------------------------
// CLI

struct Opts {
    cmd: String,
    capture: PathBuf,
    candidate: Option<PathBuf>,
    symbol: String,
    manifest: Option<PathBuf>,
    receipt: Option<PathBuf>,
    parity_report: Option<PathBuf>,
    inventory: PathBuf,
    out: Option<PathBuf>,
    label: String,
    pins: Pins,
    iters: usize,
    reps: usize,
    warmup: usize,
    cold: bool,
}

const USAGE: &str = "usage: pm_aco_oracle <full|selftest|prepare|time|twin-time> --capture DIR [--candidate CO --symbol S --candidate-manifest JSON --vulkan-receipt JSON] [--expect-spirv-sha256 H] [--expect-aco-binary-sha256 H] [--out DIR] [--label L] [--inventory JSON]\n  time: --parity-report PASSING_FULL_REPORT --candidate .. --candidate-manifest .. [--iters N=200] [--reps R=5] [--warmup W=20] [--cache warm|cold]\n  twin-time: --capture DIR --out DIR [--iters N=200] [--reps R=5] [--warmup W=20] [--cache warm|cold] (hipcc vs accepted twin only; rejects candidate/manifest/receipt/parity-report/pin options)";

fn parse() -> Result<Opts> {
    let mut a = std::env::args().skip(1);
    let cmd = a.next().ok_or(USAGE)?;
    let mut o = Opts {
        cmd,
        capture: PathBuf::new(),
        candidate: None,
        symbol: SYMBOL.to_owned(),
        manifest: None,
        receipt: None,
        parity_report: None,
        inventory: PathBuf::from(INVENTORY),
        out: None,
        label: String::new(),
        pins: Pins {
            spirv: None,
            aco_binary: None,
        },
        iters: 200,
        reps: 5,
        warmup: 20,
        cold: false,
    };
    while let Some(k) = a.next() {
        let v = a.next().ok_or(format!("{k} needs a value"))?;
        match k.as_str() {
            "--capture" => o.capture = abs_path(&v, "--capture")?,
            "--candidate" => o.candidate = Some(abs_path(&v, "--candidate")?),
            "--symbol" => o.symbol = v,
            "--candidate-manifest" => o.manifest = Some(abs_path(&v, "--candidate-manifest")?),
            "--vulkan-receipt" => o.receipt = Some(abs_path(&v, "--vulkan-receipt")?),
            "--parity-report" => o.parity_report = Some(abs_path(&v, "--parity-report")?),
            "--expect-spirv-sha256" => o.pins.spirv = Some(v),
            "--expect-aco-binary-sha256" => o.pins.aco_binary = Some(v),
            "--inventory" => o.inventory = abs_path(&v, "--inventory")?,
            "--out" => o.out = Some(abs_path(&v, "--out")?),
            "--label" => o.label = v,
            "--iters" => o.iters = v.parse().map_err(err)?,
            "--reps" => o.reps = v.parse().map_err(err)?,
            "--warmup" => o.warmup = v.parse().map_err(err)?,
            "--cache" => {
                o.cold = match v.as_str() {
                    "warm" => false,
                    "cold" => true,
                    _ => return Err("--cache warm|cold".into()),
                }
            }
            _ => return Err(format!("unknown option {k}\n{USAGE}")),
        }
    }
    if o.capture.as_os_str().is_empty() {
        return Err(format!("--capture is required\n{USAGE}"));
    }
    if o.iters == 0 || o.reps == 0 {
        return Err("--iters and --reps must be > 0".into());
    }
    Ok(o)
}

fn main() {
    match run() {
        Ok(0) => {}
        Ok(_) => std::process::exit(1),
        Err(e) => {
            eprintln!("pm_aco_oracle: error: {e}");
            std::process::exit(2);
        }
    }
}

/// The ABI hash covers the captured 32-byte kernarg with ONLY the pointer slots
/// (0, 8, 16) zeroed; M and K stay as captured, so a manifest binds to one fixture.
fn kernarg_abi_sha256(cap: &Capture) -> String {
    sha256_hex(&fixture_shape(cap).masked_kernarg())
}

fn require_passing_parity(
    path: &Path,
    cand_sha: &str,
    cap: &Capture,
    twin_image: &str,
    man: &Manifest,
) -> Result<Value> {
    let b = std::fs::read(path).map_err(|e| format!("--parity-report {}: {e}", path.display()))?;
    let r: Value = serde_json::from_slice(&b).map_err(err)?;
    let ctx = |e: String| format!("--parity-report {}: {e}", path.display());
    let check = || -> Result<()> {
        must_eq("schema", jstr(&r, "schema")?, "pm-aco-oracle-v1".to_owned())?;
        must_eq("mode", jstr(&r, "mode")?, "full".to_owned())?;
        must_eq("pass", jget(&r, "pass")?.as_bool(), Some(true))?;
        must_eq(
            "candidate.image_sha256",
            jstr(&r, "identity.candidate.image_sha256")?,
            cand_sha.to_owned(),
        )?;
        must_eq(
            "candidate manifest sha256",
            jstr(&r, "identity.candidate_manifest.sha256")?,
            man.sha256.clone(),
        )?;
        must_eq(
            "capture meta_sha256",
            jstr(&r, "fixture.meta_sha256")?,
            cap.meta_sha256.clone(),
        )?;
        must_eq(
            "twin image_sha256",
            jstr(&r, "identity.twin.detail.image_sha256")?,
            twin_image.to_owned(),
        )?;
        jstr(&r, "identity.vulkan.receipt_sha256")?;
        Ok(())
    };
    check().map_err(ctx)?;
    Ok(json!({"path": path.display().to_string(), "sha256": sha256_hex(&b)}))
}

fn run() -> Result<usize> {
    let o = parse()?;
    let cap = load_capture(&o.capture)?;
    let quality = fixture_quality(&cap)?;
    let inc = incumbent(&o.inventory)?;
    let shapes = shapes_for(&cap)?;
    let occ_name = o
        .capture
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let label = if o.label.is_empty() {
        format!("{}-{occ_name}", o.cmd)
    } else {
        o.label.clone()
    };
    let out_dir = o
        .out
        .clone()
        .unwrap_or_else(|| Path::new(RESULTS).join(&label));
    let argv: Vec<String> = std::env::args().collect();

    if o.cmd == "prepare" {
        std::fs::create_dir_all(&out_dir).map_err(err)?;
        let mut listing = Vec::new();
        for sh in &shapes {
            let d = out_dir.join(&sh.name);
            std::fs::create_dir_all(&d).map_err(err)?;
            let mut files = serde_json::Map::new();
            for i in 0..3 {
                let f = framed(&sh.pre[i]);
                let name = format!("arg{}.pre.framed.bin", PTR_OFF[i]);
                std::fs::write(d.join(&name), &f).map_err(err)?;
                files.insert(name, json!({"bytes": f.len(), "sha256": sha256_hex(&f), "payload_offset": GUARD, "payload_len": sh.pre[i].len()}));
            }
            listing.push(json!({
                "shape": sh.name, "M": sh.m, "K": sh.k, "workgroups": sh.grid, "local_size": sh.block,
                "vulkan_scope": match &sh.vk { VkScope::Fixture => "fixture (receipt top level)".to_owned(), VkScope::Derived(k) => format!("derived.{k}"), VkScope::NotRun => "not run".to_owned() },
                "kernarg_ptr_masked_hex": hex(&sh.masked_kernarg()), "files": files,
            }));
        }
        let v = json!({"schema": "pm-aco-oracle-prepare-v1", "guard_bytes": GUARD, "guard_byte": format!("{GUARD_BYTE:02x}"),
                       "fixture": quality, "shapes": listing});
        let text = serde_json::to_string_pretty(&v).map_err(err)?;
        std::fs::write(out_dir.join("prepare.json"), &text).map_err(err)?;
        println!("{text}\nPREPARED -> {}", out_dir.display());
        return Ok(0);
    }

    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch != "gfx1201" {
        return Err(format!("gfx1201 only, got {}", gpu.arch));
    }
    let stream = gpu.hip.stream_create_non_blocking().map_err(err)?;
    let gran = gpu.vmm_recommended_granularity().map_err(err)?;
    let cx = Ctx { stream, gran };
    let hipcc = load_hipcc(&gpu, &inc)?;
    let twin = load_twin(&gpu, &inc)?;
    let twin_image = twin.extra["image_sha256"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    std::fs::create_dir_all(&out_dir).map_err(err)?;

    let mut report = json!({
        "schema": "pm-aco-oracle-v1",
        "mode": o.cmd,
        "command": argv.join(" "),
        "arch": gpu.arch,
        "visible_devices": {"HIP_VISIBLE_DEVICES": std::env::var("HIP_VISIBLE_DEVICES").ok(), "ROCR_VISIBLE_DEVICES": std::env::var("ROCR_VISIBLE_DEVICES").ok()},
        "device_id": gpu.device_id,
        "allocation": "VMM only (Gpu::alloc_vmm_tensor), per-argument 4096 B guard frames padded to the VMM granule",
        "vmm_granule_bytes": cx.gran,
        "fixture": quality,
        "executable": {"path": std::env::current_exe().ok().map(|p| p.display().to_string()), "md5": std::env::current_exe().ok().map(|p| md5_of(&p))},
    });
    let shape0 = &shapes[0];
    let failures = match o.cmd.as_str() {
        "selftest" => {
            // hipcc object as the "candidate": the pipeline, graph path and all
            // negative controls must behave on a known-identical arm.
            let obj = std::fs::read(&inc.object_path).map_err(err)?;
            let cand = load_arm(
                &gpu,
                "candidate",
                "selftest: hipcc inventory object",
                &obj,
                SYMBOL,
                Some(&inc.object_path),
                json!({}),
            )?;
            report["identity"] = json!({
                "hipcc": arm_identity(&hipcc, shape0), "twin": arm_identity(&twin, shape0),
                "candidate": arm_identity(&cand, shape0), "vulkan": "not applicable (selftest)",
            });
            let setup = Setup {
                cap: &cap,
                hipcc,
                twin,
                cand,
                vk: None,
                cand_odd_tail: true,
            };
            let (n, v) = parity(&mut gpu, &cx, &setup)?;
            report["result"] = v;
            n
        }
        "full" | "time" => {
            let path = o.candidate.as_ref().ok_or("--candidate required")?;
            let mpath = o
                .manifest
                .as_ref()
                .ok_or("--candidate-manifest required (refusing to infer candidate identity)")?;
            must_eq("--symbol", o.symbol.clone(), SYMBOL.to_owned())?;
            let bytes =
                std::fs::read(path).map_err(|e| format!("--candidate {}: {e}", path.display()))?;
            let man = load_manifest(
                mpath,
                &sha256_hex(&bytes),
                &o.symbol,
                &cap,
                &kernarg_abi_sha256(&cap),
            )?;
            let cand = load_arm(
                &gpu,
                "candidate",
                "F3 ACO PM/HIP candidate",
                &bytes,
                &o.symbol,
                Some(path),
                json!({"manifest": man.json}),
            )?;
            let mut ident = json!({
                "hipcc": arm_identity(&hipcc, shape0), "twin": arm_identity(&twin, shape0), "candidate": arm_identity(&cand, shape0),
                "candidate_manifest": {"path": mpath.display().to_string(), "sha256": man.sha256,
                                       "object_sha256": man.object_sha256, "spirv_sha256": man.spirv_sha256,
                                       "aco_binary_sha256": man.aco_binary_sha256, "aco_sidecar_sha256": man.aco_sidecar_sha256,
                                       "shape_support": man.json["shape_support"]},
            });
            if o.cmd == "full" {
                let rpath = o.receipt.as_ref().ok_or("--vulkan-receipt required (refusing to run without the original RADV/Vulkan reference)")?;
                let vk = load_vulkan(rpath, &cap, &man, &shapes, &o.pins)?;
                ident["vulkan"] = vk.identity.clone();
                report["identity"] = ident;
                let setup = Setup {
                    cap: &cap,
                    hipcc,
                    twin,
                    cand,
                    vk: Some(&vk),
                    cand_odd_tail: man.odd_tail,
                };
                let (n, v) = parity(&mut gpu, &cx, &setup)?;
                report["result"] = v;
                n
            } else {
                let prev = o.parity_report.as_ref().ok_or(
                    "time requires --parity-report of a passing `full` run for this candidate",
                )?;
                report["parity_report"] =
                    require_passing_parity(prev, &cand.image_sha256, &cap, &twin_image, &man)?;
                report["identity"] = ident;
                let setup = Setup {
                    cap: &cap,
                    hipcc,
                    twin,
                    cand,
                    vk: None,
                    cand_odd_tail: man.odd_tail,
                };
                let cfg = TimeCfg {
                    iters: o.iters,
                    reps: o.reps,
                    warmup: o.warmup,
                    cold: o.cold,
                };
                report["result"] = run_time(&mut gpu, &cx, &setup, &cfg)?;
                0
            }
        }
        "twin-time" => {
            if o.candidate.is_some()
                || o.manifest.is_some()
                || o.receipt.is_some()
                || o.parity_report.is_some()
                || o.pins.spirv.is_some()
                || o.pins.aco_binary.is_some()
                || o.symbol != SYMBOL
            {
                return Err("twin-time takes only --capture/--out (+ --iters/--reps/--warmup/--cache/--label/--inventory); candidate, manifest, receipt, parity-report, pin and symbol options are refused".into());
            }
            if o.out.is_none() {
                return Err("twin-time requires --out".into());
            }
            let exe = std::env::current_exe().map_err(err)?;
            let exe_bytes = std::fs::read(&exe).map_err(|e| format!("{}: {e}", exe.display()))?;
            report["schema"] = json!("pm-twin-time-v1");
            report["identity"] = json!({
                "hipcc": arm_identity(&hipcc, shape0), "twin": arm_identity(&twin, shape0),
                "inventory": {
                    "path": o.inventory.display().to_string(), "source_sha256": inc.source_sha256,
                    "object_path": inc.object_path.display().to_string(), "object_sha256": inc.object_sha256, "recipe_flags": inc.flags,
                },
                "executable": {"path": exe.display().to_string(), "sha256": sha256_hex(&exe_bytes), "md5": md5_of(&exe)},
                "card": {
                    "device_id": gpu.device_id, "arch": gpu.arch,
                    "uuid": gpu.hip.device_uuid(gpu.device_id).map_err(err)?,
                    "pci_bus_id": gpu.hip.device_pci_bus_id(gpu.device_id).map_err(err)?,
                },
                "capture": {
                    "dir": cap.dir.display().to_string(), "meta_sha256": cap.meta_sha256,
                    "kernarg_hex": hex(&cap.kernarg), "files": cap.meta["files"],
                },
            });
            let cfg = TimeCfg {
                iters: o.iters,
                reps: o.reps,
                warmup: o.warmup,
                cold: o.cold,
            };
            let (n, v) = run_twin_time(&mut gpu, &cx, &hipcc, &twin, &cap, &cfg)?;
            report["result"] = v;
            n
        }
        other => return Err(format!("unknown command {other}\n{USAGE}")),
    };
    report["pass"] = json!(failures == 0);
    if o.cmd == "full" {
        report["claim_scope"] = json!(match report["result"]["odd_tail_claim"].as_str() {
            Some("claimed_and_checked") => "fixture + odd/tail shapes",
            Some("refused_unsupported") =>
                "fixture only; odd/tail shapes explicitly UNSUPPORTED by the candidate",
            _ => "fixture",
        });
    }
    if o.cmd == "twin-time" {
        report["claim_scope"] = json!("kernel-only hipcc vs accepted native twin at the captured fixture geometry, eager event timing after eager+hipGraph whole-buffer parity; not a product speed claim");
    }
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    let file = match o.cmd.as_str() {
        "time" => "time-report.json",
        "twin-time" => "twin-time-report.json",
        _ => "report.json",
    };
    std::fs::write(out_dir.join(file), &text).map_err(err)?;
    println!("{text}");
    println!(
        "{}: {} {} -> {}",
        if failures == 0 { "PASS" } else { "FAIL" },
        o.cmd,
        MODULE,
        out_dir.join(file).display()
    );
    Ok(failures)
}
