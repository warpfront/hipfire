// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared utilities of `qwen4_e8p_gp`: errors, hashing, JSON, device upload helpers, HIP offload-bundle binding,
//! the expert-projection geometry, and the statistics of `gp_bench.py` (quantiles, medians, SplitMix64 bootstrap).

use std::io::Read;
use std::path::Path;

use hip_bridge::{DeviceBuffer, HipRuntime};
use hipfire_runtime::mq_i4_reference::DecodeKind;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub type Result<T, E = String> = std::result::Result<T, E>;
pub fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

pub const BLK: usize = 72;
pub const ALIGN: usize = 256;
pub const GUARD: usize = 256;
pub const TILE_ROWS: usize = 16;

// ---------------------------------------------------------------------------------------------------------------
// hashing / files / json
// ---------------------------------------------------------------------------------------------------------------

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
pub fn sha(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
pub fn sha_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 16 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf).map_err(err)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&h.finalize()), total))
}
pub fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
pub fn read_file(p: &Path) -> Result<Vec<u8>> {
    std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))
}
pub fn read_json(p: &Path) -> Result<(Value, Vec<u8>)> {
    let raw = read_file(p)?;
    let v = serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok((v, raw))
}
pub fn write_json(p: &Path, v: &Value) -> Result<Vec<u8>> {
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d).map_err(err)?;
    }
    let mut b = serde_json::to_vec_pretty(v).map_err(err)?;
    b.push(b'\n');
    std::fs::write(p, &b).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(b)
}
pub fn json_str<'a>(v: &'a Value, what: &str) -> Result<&'a str> {
    v.as_str()
        .ok_or_else(|| format!("{what}: expected a string"))
}
pub fn json_u64(v: &Value, what: &str) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| format!("{what}: expected an unsigned integer"))
}
pub fn fnv64(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}
pub fn align(n: usize) -> usize {
    n.div_ceil(ALIGN) * ALIGN
}
pub fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
pub fn u32s_le(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn i32s_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
pub fn le_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// splitmix64: deterministic, platform independent.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---------------------------------------------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------------------------------------------

pub fn up(hip: &HipRuntime, bytes: &[u8]) -> Result<DeviceBuffer> {
    let b = hip.malloc(bytes.len().max(8)).map_err(err)?;
    if !bytes.is_empty() {
        hip.memcpy_htod(&b, bytes).map_err(err)?;
    }
    Ok(b)
}
pub fn read_back(hip: &HipRuntime, b: &DeviceBuffer, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    if n > 0 {
        hip.memcpy_dtoh(&mut v, b).map_err(err)?;
    }
    Ok(v)
}
pub fn free_all(hip: &HipRuntime, bufs: Vec<DeviceBuffer>) -> Result<()> {
    for b in bufs {
        hip.free(b).map_err(err)?;
    }
    Ok(())
}

/// Shared 64-byte PM kernarg ABI: 5 pointers (expert table, tiles, sorted slots, x, y) then m, k, x_row_div, m_total,
/// x_rows and `n_experts@60` = the pointer table's actual entry count. `blob_len` >= 64 zero-pads to the code object's
/// `.kernarg_segment_size`.
#[allow(clippy::too_many_arguments)]
pub fn kernarg(
    table: &DeviceBuffer,
    tiles: &DeviceBuffer,
    sorted: &DeviceBuffer,
    x: &DeviceBuffer,
    y: &DeviceBuffer,
    m: usize,
    k: usize,
    div: usize,
    m_total: usize,
    x_rows: usize,
    blob_len: usize,
) -> Vec<u8> {
    let mut a = vec![0u8; blob_len.max(64)];
    for (i, p) in [table, tiles, sorted, x, y].iter().enumerate() {
        a[i * 8..i * 8 + 8].copy_from_slice(&(p.as_ptr() as u64).to_le_bytes());
    }
    for (off, v) in [
        (40usize, m),
        (44, k),
        (48, div),
        (52, m_total),
        (56, x_rows),
        (60, table.size() / 8),
    ] {
        a[off..off + 4].copy_from_slice(&(v as u32).to_le_bytes());
    }
    a
}

// ---------------------------------------------------------------------------------------------------------------
// msgpack `.kernarg_segment_size` scan (same scan as the real-G1 harness / A4 identity runner)
// ---------------------------------------------------------------------------------------------------------------

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from >= hay.len() {
        return None;
    }
    hay[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|p| p + from)
}
fn mp_uint(b: &[u8], at: usize) -> Option<u64> {
    let t = *b.get(at)?;
    match t {
        0x00..=0x7f => Some(t as u64),
        0xcc => Some(*b.get(at + 1)? as u64),
        0xcd => Some(u16::from_be_bytes(b.get(at + 1..at + 3)?.try_into().ok()?) as u64),
        0xce => Some(u32::from_be_bytes(b.get(at + 1..at + 5)?.try_into().ok()?) as u64),
        _ => None,
    }
}
fn mp_str(b: &[u8], at: usize) -> Option<&[u8]> {
    let t = *b.get(at)?;
    let (len, start) = match t {
        0xa0..=0xbf => ((t & 0x1f) as usize, at + 1),
        0xd9 => (*b.get(at + 1)? as usize, at + 2),
        _ => return None,
    };
    b.get(start..start + len)
}
/// `.kernarg_segment_size` of kernel `symbol` in the code object's msgpack note, if present.
pub fn metadata_kernarg_size(image: &[u8], symbol: &str) -> Option<usize> {
    let mut key = vec![0xa0 | 21u8];
    key.extend_from_slice(b".kernarg_segment_size");
    let mut name_key = vec![0xa5u8];
    name_key.extend_from_slice(b".name");
    let mut from = 0;
    while let Some(pos) = find(image, &key, from) {
        let vpos = pos + key.len();
        let next_key = find(image, &key, vpos).unwrap_or(image.len());
        if let (Some(val), Some(np)) = (mp_uint(image, vpos), find(image, &name_key, vpos)) {
            if np < next_key && mp_str(image, np + name_key.len()) == Some(symbol.as_bytes()) {
                return Some(val as usize);
            }
        }
        from = vpos;
    }
    None
}

// ---------------------------------------------------------------------------------------------------------------
// HIP offload bundle binding (standard uncompressed clang-offload-bundler format)
// ---------------------------------------------------------------------------------------------------------------

pub fn bundle_device_entry<'a>(bundle: &'a [u8], arch: &str) -> Result<(String, &'a [u8])> {
    const MAGIC: &[u8; 24] = b"__CLANG_OFFLOAD_BUNDLE__";
    if !bundle.starts_with(MAGIC) {
        return Err("bundle: missing __CLANG_OFFLOAD_BUNDLE__ magic (compressed CCOB bundles are not supported)".into());
    }
    let mut pos = MAGIC.len();
    let u64_at = |pos: &mut usize| -> Result<u64> {
        let w = bundle
            .get(*pos..*pos + 8)
            .ok_or_else(|| format!("bundle: header truncated at byte {pos}"))?;
        *pos += 8;
        Ok(u64::from_le_bytes(w.try_into().unwrap()))
    };
    let count = u64_at(&mut pos)?;
    let prefix = format!("hipv4-amdgcn-amd-amdhsa--{arch}");
    let mut hit: Vec<(String, &[u8])> = Vec::new();
    for i in 0..count {
        let (off, size, tl) = (
            u64_at(&mut pos)?,
            u64_at(&mut pos)?,
            u64_at(&mut pos)? as usize,
        );
        let triple = bundle
            .get(pos..pos + tl)
            .ok_or_else(|| format!("bundle: entry {i} triple truncated"))?;
        pos += tl;
        let triple = std::str::from_utf8(triple).map_err(err)?.to_owned();
        if triple == prefix
            || triple
                .strip_prefix(&prefix)
                .is_some_and(|r| r.starts_with(':'))
        {
            let payload = bundle
                .get(
                    off as usize
                        ..(off as usize)
                            .checked_add(size as usize)
                            .ok_or("bundle: payload overflow")?,
                )
                .ok_or_else(|| format!("bundle: entry {i} payload outside the file"))?;
            hit.push((triple, payload));
        }
    }
    if hit.len() != 1 {
        return Err(format!(
            "bundle: {} device entries match {prefix} (need exactly 1)",
            hit.len()
        ));
    }
    Ok(hit.pop().unwrap())
}

/// Receipt block binding a loaded native code object to its pinned SHA-256 (when pinned; refuses on mismatch) and its
/// optional source bundle (refuses unless the bundle's gfx1201 device entry equals the loaded ELF).
pub fn bind_module(
    label: &str,
    co_path: &Path,
    co: &[u8],
    pinned: Option<&str>,
    bundle_path: Option<&Path>,
) -> Result<Value> {
    let got = sha(co);
    if let Some(p) = pinned {
        if got != p {
            return Err(format!(
                "{label}: code object {} sha256 {got} != pinned {p}",
                co_path.display()
            ));
        }
    }
    let bundle = match bundle_path {
        None => Value::Null,
        Some(p) => {
            let b = read_file(p)?;
            let (triple, payload) = bundle_device_entry(&b, "gfx1201")
                .map_err(|e| format!("{label}: {} {e}", p.display()))?;
            if payload != co {
                return Err(format!(
                    "{label}: bundle {} device entry != loaded ELF",
                    p.display()
                ));
            }
            json!({"path": p.display().to_string(), "bytes": b.len(), "sha256": sha(&b), "device_entry": {"triple": triple, "bytes": payload.len(), "sha256": sha(payload)}, "device_entry_equals_loaded_elf": true})
        }
    };
    Ok(
        json!({"codeobject": {"path": co_path.display().to_string(), "bytes": co.len(), "sha256": got, "pinned_sha256": pinned}, "bundle": bundle}),
    )
}

// ---------------------------------------------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Proj {
    GateUp = 0,
    Down = 1,
}
impl Proj {
    pub const ALL: [Proj; 2] = [Proj::GateUp, Proj::Down];
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn name(self) -> &'static str {
        match self {
            Proj::GateUp => "gate_up",
            Proj::Down => "down",
        }
    }
    pub fn m(self) -> usize {
        match self {
            Proj::GateUp => 1280,
            Proj::Down => 2560,
        }
    }
    pub fn k(self) -> usize {
        match self {
            Proj::GateUp => 2560,
            Proj::Down => 640,
        }
    }
    /// BF16 output columns per slot.
    pub fn h(self) -> usize {
        match self {
            Proj::GateUp => 640,
            Proj::Down => 2560,
        }
    }
    pub fn groups(self) -> usize {
        self.k() / 128
    }
    pub fn threads(self) -> u32 {
        match self {
            Proj::GateUp => 64,
            Proj::Down => 128,
        }
    }
    /// grid.x of every PM kernel (W3, MQ4, E8P: identical launch ABI). `m` is the FULL fused row count (gate_up 1280 =
    /// gate 640 + up 640); a CTA owns 32 gate rows plus the matching 32 up rows (gate_up) or 64 rows (down), so
    /// grid.x = m / 64 for both kinds: gate_up 20 CTAs, down 40 CTAs.
    pub fn grid_x(self) -> u32 {
        (self.m() / 64) as u32
    }
    pub fn kind(self) -> DecodeKind {
        match self {
            Proj::GateUp => DecodeKind::GateUp,
            Proj::Down => DecodeKind::Down,
        }
    }
    /// Manifest key suffix of the solved E8P record file (`L<N>.<suffix>.bin`).
    pub fn record_suffix(self) -> &'static str {
        match self {
            Proj::GateUp => "gate_up_proj",
            Proj::Down => "down_proj",
        }
    }
    /// Planar words plane bytes of one expert's E8P blob: `m*k/8` u16.
    pub fn e8p_words_bytes(self) -> usize {
        self.m() * self.k() / 4
    }
    /// Scales plane bytes: `m*k/128` f16.
    pub fn e8p_scales_bytes(self) -> usize {
        self.m() * self.k() / 64
    }
    /// One expert's device blob: words plane then scales plane.
    pub fn e8p_blob_bytes(self) -> usize {
        self.e8p_words_bytes() + self.e8p_scales_bytes()
    }
    /// One expert's record in the solved `.bin`: 8-byte header, idx plane, signs plane, f16 scales.
    pub fn e8p_record_bytes(self) -> usize {
        8 + self.m() * self.k() / 8 * 2 + self.e8p_scales_bytes()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// statistics (the `gp_bench.py` definitions)
// ---------------------------------------------------------------------------------------------------------------

pub fn quantile(s: &[f64], q: f64) -> f64 {
    let pos = (s.len() - 1) as f64 * q;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}
fn median_sorted(s: &[f64]) -> f64 {
    let n = s.len();
    let m = n / 2;
    if n % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}
pub fn median(samples: &[f64]) -> f64 {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    median_sorted(&s)
}
pub fn summary(samples: &[f64]) -> Value {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median_sorted(&s);
    let (p25, p75) = (quantile(&s, 0.25), quantile(&s, 0.75));
    json!({"n": s.len(), "median_s": med, "min_s": s[0], "max_s": s[s.len() - 1], "p25_s": p25, "p75_s": p75, "iqr_s": p75 - p25, "rel_iqr": (p75 - p25) / med})
}

pub const BOOTSTRAP_SEED: u64 = 20_261_014;
pub const BOOTSTRAP_RESAMPLES: usize = 10_000;

/// `gp_bench.py::_ratio_ci`: deterministic percentile bootstrap (independent resampling, SplitMix64, index =
/// `(next * n) >> 64`, numerator draws then denominator draws) of `median(num)/median(den)`. Returns
/// `(time_ratio_ci {lo,hi}, inverse_ci {lo,hi})`.
pub fn ratio_ci(num: &[f64], den: &[f64], resamples: usize) -> (Value, Value) {
    let (na, nb) = (num.len(), den.len());
    let mut rng = Rng(BOOTSTRAP_SEED);
    let draw = |rng: &mut Rng, n: usize| -> usize {
        ((u128::from(rng.next()) * n as u128) >> 64) as usize
    };
    let mut ratios = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut ra: Vec<f64> = (0..na).map(|_| num[draw(&mut rng, na)]).collect();
        let mut rb: Vec<f64> = (0..nb).map(|_| den[draw(&mut rng, nb)]).collect();
        ra.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rb.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ratios.push(median_sorted(&ra) / median_sorted(&rb));
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut inv: Vec<f64> = ratios.iter().map(|x| 1.0 / x).collect();
    inv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        json!({"lo": quantile(&ratios, 0.025), "hi": quantile(&ratios, 0.975)}),
        json!({"lo": quantile(&inv, 0.025), "hi": quantile(&inv, 0.975)}),
    )
}

/// Refuse CU masks (forbidden on RDNA cards).
pub fn refuse_cu_masks() -> Result<()> {
    for var in ["HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"] {
        if std::env::var(var).is_ok_and(|v| !v.is_empty()) {
            return Err(format!(
                "{var} is set: CU masks are forbidden on RDNA cards; unset it"
            ));
        }
    }
    Ok(())
}

/// Fabric provenance block. Device runs only come from `fabric-run.sh --class <class> --card <card>`; the wrapper
/// exports `FABRIC_JOB` and `FABRIC_CARDS` (not the class or the owner), so those two are asserted by the command line.
pub fn fabric_provenance(
    card: &str,
    class: &str,
    owner: Option<&str>,
    wrapper: Option<&str>,
) -> Result<Value> {
    if !["A", "B", "C", "E"].contains(&card) {
        return Err(format!("--card {card}: expected A|B|C|E"));
    }
    let job = std::env::var("FABRIC_JOB").unwrap_or_default();
    let cards = std::env::var("FABRIC_CARDS").unwrap_or_default();
    if job.is_empty() || cards != card {
        return Err(format!("not running under `fabric-run.sh --class {class} --card {card}` (FABRIC_JOB={job:?} FABRIC_CARDS={cards:?}); device runs come only from fabric runs"));
    }
    let env_owner = std::env::var("FABRIC_OWNER").ok().filter(|s| !s.is_empty());
    let owner = owner
        .map(str::to_owned)
        .or(env_owner)
        .ok_or("fabric owner: pass --fabric-owner or run with --owner")?;
    let wrapper =
        wrapper.ok_or("--fabric-wrapper (path of fabric-run.sh) is required for a device run")?;
    if Path::new(wrapper).file_name().and_then(|s| s.to_str()) != Some("fabric-run.sh") {
        return Err(format!(
            "--fabric-wrapper {wrapper}: must be the fabric-run.sh path"
        ));
    }
    Ok(
        json!({"class": class, "card": card, "owner": owner, "wrapper": wrapper, "job": job,
        "class_asserted_by": "--fabric-class / --class on the fabric-run.sh command line (the wrapper does not export the class)"}),
    )
}

/// Launch-geometry known answer, checked on the CPU before any device is touched: the FULL fused M of gate_up (1280 =
/// gate 640 + up 640) launches 20 CTAs and down (2560) 40 CTAs at 64/128 threads. (A CTA owns 32 gate rows plus the
/// matching 32 up rows, so grid.x = M / 64 for both kinds; M / 32 would double-cover gate_up.)
pub fn launch_geometry_known_answer() -> Result<Value> {
    let got = [
        (Proj::GateUp, 1280usize, 20u32, 64u32),
        (Proj::Down, 2560, 40, 128),
    ];
    for (p, m, grid, thr) in got {
        if p.m() != m || p.grid_x() != grid || p.threads() != thr {
            return Err(format!("launch geometry {}: m {} grid.x {} threads {} != known answer m {m} grid.x {grid} threads {thr}", p.name(), p.m(), p.grid_x(), p.threads()));
        }
    }
    Ok(
        json!({"gate_up": {"m": 1280, "grid_x": 20, "threads": 64}, "down": {"m": 2560, "grid_x": 40, "threads": 128}, "ok": true}),
    )
}
