// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Captured MoE routes and activation rows as pinned judge inputs
//! (PM/rip-loop Slice 3, section B).
//!
//! * [`route_hist`] turns a `HIPFIRE_MOE_ROUTE_DUMP` file into the
//!   `route_hist.v1` artefact (`pm-census extract route-hist`).
//! * [`act_capture`] turns a `HIPFIRE_MOE_ACT_CAPTURE` file into the
//!   `act_capture.v1` artefact (`pm-census extract act-capture`).
//!
//! # Input formats (little endian, append-only)
//!
//! **Route dump**, per record: `u32 layer, u32 n (rows), u32 k (top-k), u32
//! position`, then `n*k` `i32` expert ids, slot = `row*k + j`.
//!
//! **Activation capture**, per record: `u32 magic 0x41435431 ('ACT1'), u32
//! site_len, site bytes (UTF-8), u32 layer, u32 rows, u32 k, u32 position`,
//! then `rows*k` `f32`.
//!
//! # Output
//!
//! Both artefacts are a JSON file plus a sibling bin. The JSON's `id` field is
//! `sha256(serde_json::to_vec(artefact with id omitted))` in the struct field
//! order below; it is stored in the file as the last key and is opaque to
//! consumers (integrity is `ids.sha256` / `bin.sha256` against the bin bytes).
//! `ids.path` / `bin.path` are the bin's file name, relative to the JSON's
//! directory.
//!
//! * `route_hist.v1` bin (`<stem>.ids.bin`): the dump's id payloads
//!   concatenated in record order, headers stripped (`i32 LE`, `row*k+j`).
//! * `act_capture.v1` bin (`<stem>.rows.f32`): rows only, `f32 LE`, grouped
//!   per site (sites sorted by name, record order within a site); site `s`
//!   occupies bytes `[offset_bytes, offset_bytes + rows_total * k * 4)`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ROUTE_HIST_SCHEMA: &str = "route_hist.v1";
pub const ACT_CAPTURE_SCHEMA: &str = "act_capture.v1";
pub const ROUTE_IDS_LAYOUT: &str = "i32 [layer-record order] row*k+j";
/// `'ACT1'` read as a little-endian `u32`.
pub const ACT_MAGIC: u32 = 0x4143_5431;

const USAGE: &str = "\
pm-census extract route-hist  --dump FILE [--model ID] [--experts E] --out route_hist.json
pm-census extract act-capture --dump FILE --out act_capture.json

route-hist   HIPFIRE_MOE_ROUTE_DUMP records -> route_hist.v1 + <out stem>.ids.bin
             E defaults to max id + 1; --experts is authoritative (ids outside [0,E) are refused)
act-capture  HIPFIRE_MOE_ACT_CAPTURE records -> act_capture.v1 + <out stem>.rows.f32 (rows grouped per site)
Both print {\"id\", \"path\"}; id = sha256 of the canonical JSON (id omitted).";

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------------
// artefact types (field order is the canonical serialization order)

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceRef {
    pub path: String,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouteRecordRef {
    pub layer: u32,
    pub n: u32,
    pub k: u32,
    pub position: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerCounts {
    pub layer: u32,
    pub tokens: u64,
    pub counts: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TotalCounts {
    pub tokens: u64,
    pub counts: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinRef {
    pub path: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layout: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RouteHist {
    pub schema: String,
    pub source: SourceRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub n_experts: u32,
    pub topk: u32,
    pub records: Vec<RouteRecordRef>,
    pub per_layer: Vec<LayerCounts>,
    pub total: TotalCounts,
    pub gini: f64,
    pub max_share: f64,
    pub ids: BinRef,
    /// `sha256(canonical JSON with this field omitted)`; set by [`RouteHist::seal`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteEntry {
    pub site: String,
    pub k: u32,
    pub rows_total: u64,
    pub records: u64,
    pub offset_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteStats {
    pub rms: f64,
    pub absmax: f64,
    pub mean: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActCapture {
    pub schema: String,
    pub source: SourceRef,
    pub sites: Vec<SiteEntry>,
    pub bin: BinRef,
    pub stats: BTreeMap<String, SiteStats>,
    /// `sha256(canonical JSON with this field omitted)`; set by [`ActCapture::seal`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

macro_rules! sealed {
    ($t:ty) => {
        impl $t {
            /// `sha256(serde_json::to_vec(self with id omitted))`.
            pub fn canonical_id(&self) -> String {
                let mut c = self.clone();
                c.id = None;
                sha256_hex(&serde_json::to_vec(&c).expect("artefact serializes"))
            }

            /// Store [`Self::canonical_id`] in `id` and return it.
            pub fn seal(&mut self) -> String {
                let id = self.canonical_id();
                self.id = Some(id.clone());
                id
            }
        }
    };
}
sealed!(RouteHist);
sealed!(ActCapture);

// ---------------------------------------------------------------------------
// dump readers

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn left(&self) -> usize {
        self.b.len() - self.at
    }

    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], String> {
        if self.left() < n {
            return Err(format!(
                "truncated at byte {}: {what} needs {n} bytes, {} left",
                self.at,
                self.left()
            ));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }

    fn u32(&mut self, what: &str) -> Result<u32, String> {
        let s = self.take(4, what)?;
        Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}

/// One route-dump record: header plus its `n*k` ids (still LE bytes).
pub struct RouteRecord<'a> {
    pub layer: u32,
    pub n: u32,
    pub k: u32,
    pub position: u32,
    pub ids: &'a [u8],
}

fn le_i32(b: &[u8]) -> impl Iterator<Item = i32> + '_ {
    b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

/// Split a route dump into records. A truncated record, `n == 0` or `k == 0`
/// is an error, never a silent drop.
pub fn parse_route_dump(bytes: &[u8]) -> Result<Vec<RouteRecord<'_>>, String> {
    let mut c = Cursor { b: bytes, at: 0 };
    let mut out = Vec::new();
    while c.left() > 0 {
        let idx = out.len();
        let what = format!("route record {idx}");
        let layer = c.u32(&what)?;
        let n = c.u32(&what)?;
        let k = c.u32(&what)?;
        let position = c.u32(&what)?;
        if n == 0 || k == 0 {
            return Err(format!("{what}: n={n} k={k}; both must be >= 1"));
        }
        let len = (n as usize)
            .checked_mul(k as usize)
            .and_then(|s| s.checked_mul(4))
            .ok_or_else(|| format!("{what}: n*k*4 overflows"))?;
        let ids = c.take(len, &format!("{what} ids (layer {layer}, n={n}, k={k})"))?;
        out.push(RouteRecord {
            layer,
            n,
            k,
            position,
            ids,
        });
    }
    Ok(out)
}

/// Gini coefficient of non-negative counts (0 = perfectly even); 0 for an
/// empty or all-zero histogram.
fn gini(counts: &[u64]) -> f64 {
    let total: u128 = counts.iter().map(|&c| c as u128).sum();
    if counts.is_empty() || total == 0 {
        return 0.0;
    }
    let mut s = counts.to_vec();
    s.sort_unstable();
    let n = s.len() as u128;
    let weighted: u128 = s
        .iter()
        .enumerate()
        .map(|(i, &c)| (i as u128 + 1) * c as u128)
        .sum();
    // G = 2 * sum(i * x_i) / (n * sum x) - (n + 1) / n
    (2 * weighted) as f64 / (n * total) as f64 - (n + 1) as f64 / n as f64
}

/// Build `route_hist.v1` (unsealed, `ids.path` unset) and the ids bin.
pub fn route_hist(
    dump: &[u8],
    source: SourceRef,
    model: Option<String>,
    experts: Option<u32>,
) -> Result<(RouteHist, Vec<u8>), String> {
    let recs = parse_route_dump(dump)?;
    let Some(first) = recs.first() else {
        return Err("route dump has no records".into());
    };
    let topk = first.k;
    let mut max_id = i32::MIN;
    for (i, r) in recs.iter().enumerate() {
        if r.k != topk {
            return Err(format!(
                "route record {i} (layer {}): k={} but record 0 has k={topk}",
                r.layer, r.k
            ));
        }
        for id in le_i32(r.ids) {
            if id < 0 {
                return Err(format!(
                    "route record {i} (layer {}): negative expert id {id}",
                    r.layer
                ));
            }
            max_id = max_id.max(id);
        }
    }
    let n_experts = match experts {
        Some(0) => return Err("--experts must be >= 1".into()),
        Some(e) => e,
        None => max_id as u32 + 1,
    };
    let mut per_layer: BTreeMap<u32, (u64, Vec<u64>)> = BTreeMap::new();
    let mut total = vec![0u64; n_experts as usize];
    let mut total_tokens = 0u64;
    let mut ids_bin = Vec::with_capacity(recs.iter().map(|r| r.ids.len()).sum());
    for (i, r) in recs.iter().enumerate() {
        let slot = per_layer
            .entry(r.layer)
            .or_insert_with(|| (0, vec![0u64; n_experts as usize]));
        slot.0 += r.n as u64;
        total_tokens += r.n as u64;
        for id in le_i32(r.ids) {
            let e = id as u32;
            if e >= n_experts {
                return Err(format!(
                    "route record {i} (layer {}): expert id {id} outside [0,{n_experts})",
                    r.layer
                ));
            }
            slot.1[e as usize] += 1;
            total[e as usize] += 1;
        }
        ids_bin.extend_from_slice(r.ids);
    }
    let slots: u64 = total.iter().sum();
    let max_share = total.iter().copied().max().unwrap_or(0) as f64 / slots as f64;
    let hist = RouteHist {
        schema: ROUTE_HIST_SCHEMA.into(),
        source,
        model,
        n_experts,
        topk,
        records: recs
            .iter()
            .map(|r| RouteRecordRef {
                layer: r.layer,
                n: r.n,
                k: r.k,
                position: r.position,
            })
            .collect(),
        per_layer: per_layer
            .into_iter()
            .map(|(layer, (tokens, counts))| LayerCounts {
                layer,
                tokens,
                counts,
            })
            .collect(),
        gini: gini(&total),
        max_share,
        total: TotalCounts {
            tokens: total_tokens,
            counts: total,
        },
        ids: BinRef {
            path: String::new(),
            sha256: sha256_hex(&ids_bin),
            layout: Some(ROUTE_IDS_LAYOUT.into()),
        },
        id: None,
    };
    Ok((hist, ids_bin))
}

struct SiteAcc {
    k: u32,
    records: u64,
    rows: u64,
    rows_bytes: Vec<u8>,
    sum: f64,
    sumsq: f64,
    absmax: f64,
}

/// Build `act_capture.v1` (unsealed, `bin.path` unset) and the grouped rows bin.
pub fn act_capture(dump: &[u8], source: SourceRef) -> Result<(ActCapture, Vec<u8>), String> {
    let mut c = Cursor { b: dump, at: 0 };
    let mut sites: BTreeMap<String, SiteAcc> = BTreeMap::new();
    let mut idx = 0usize;
    while c.left() > 0 {
        let what = format!("act record {idx}");
        let magic = c.u32(&what)?;
        if magic != ACT_MAGIC {
            return Err(format!(
                "{what}: bad magic {magic:#010x} at byte {} (want {ACT_MAGIC:#010x})",
                c.at - 4
            ));
        }
        let site_len = c.u32(&what)? as usize;
        let site = std::str::from_utf8(c.take(site_len, &format!("{what} site"))?)
            .map_err(|e| format!("{what}: site is not UTF-8: {e}"))?
            .to_owned();
        let _layer = c.u32(&what)?;
        let rows = c.u32(&what)?;
        let k = c.u32(&what)?;
        let _position = c.u32(&what)?;
        if rows == 0 || k == 0 {
            return Err(format!(
                "{what} ({site}): rows={rows} k={k}; both must be >= 1"
            ));
        }
        let len = (rows as usize)
            .checked_mul(k as usize)
            .and_then(|s| s.checked_mul(4))
            .ok_or_else(|| format!("{what}: rows*k*4 overflows"))?;
        let data = c.take(len, &format!("{what} ({site}) rows={rows} k={k}"))?;
        let acc = sites.entry(site.clone()).or_insert_with(|| SiteAcc {
            k,
            records: 0,
            rows: 0,
            rows_bytes: Vec::new(),
            sum: 0.0,
            sumsq: 0.0,
            absmax: 0.0,
        });
        if acc.k != k {
            return Err(format!(
                "{what}: site {site:?} has k={k} but earlier records have k={}",
                acc.k
            ));
        }
        for w in data.chunks_exact(4) {
            let x = f32::from_le_bytes([w[0], w[1], w[2], w[3]]);
            if !x.is_finite() {
                return Err(format!("{what} ({site}): non-finite activation {x}"));
            }
            let x = x as f64;
            acc.sum += x;
            acc.sumsq += x * x;
            acc.absmax = acc.absmax.max(x.abs());
        }
        acc.records += 1;
        acc.rows += rows as u64;
        acc.rows_bytes.extend_from_slice(data);
        idx += 1;
    }
    if sites.is_empty() {
        return Err("activation dump has no records".into());
    }
    let mut bin = Vec::new();
    let mut entries = Vec::new();
    let mut stats = BTreeMap::new();
    for (site, a) in sites {
        let count = (a.rows * a.k as u64) as f64;
        stats.insert(
            site.clone(),
            SiteStats {
                rms: (a.sumsq / count).sqrt(),
                absmax: a.absmax,
                mean: a.sum / count,
            },
        );
        entries.push(SiteEntry {
            site,
            k: a.k,
            rows_total: a.rows,
            records: a.records,
            offset_bytes: bin.len() as u64,
        });
        bin.extend_from_slice(&a.rows_bytes);
    }
    let cap = ActCapture {
        schema: ACT_CAPTURE_SCHEMA.into(),
        source,
        sites: entries,
        bin: BinRef {
            path: String::new(),
            sha256: sha256_hex(&bin),
            layout: None,
        },
        stats,
        id: None,
    };
    Ok((cap, bin))
}

// ---------------------------------------------------------------------------
// files + CLI

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn source_of(dump: &Path, bytes: &[u8]) -> SourceRef {
    SourceRef {
        path: dump.display().to_string(),
        sha256: sha256_hex(bytes),
    }
}

/// `<out dir>/<stem of out><suffix>`: the bin's path and its file name.
fn sibling(out: &Path, suffix: &str) -> Result<(PathBuf, String), String> {
    let stem = out
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{}: --out has no file name", out.display()))?;
    let name = format!("{stem}{suffix}");
    Ok((out.with_file_name(&name), name))
}

/// Write a [`RouteHist`] and its bin next to `out`; returns the sealed id.
pub fn write_route_hist(mut hist: RouteHist, ids_bin: &[u8], out: &Path) -> Result<String, String> {
    let (bin_path, bin_name) = sibling(out, ".ids.bin")?;
    hist.ids.path = bin_name;
    let id = hist.seal();
    write(&bin_path, ids_bin)?;
    write(
        out,
        &serde_json::to_vec_pretty(&hist).map_err(|e| e.to_string())?,
    )?;
    Ok(id)
}

/// Write an [`ActCapture`] and its bin next to `out`; returns the sealed id.
pub fn write_act_capture(mut cap: ActCapture, bin: &[u8], out: &Path) -> Result<String, String> {
    let (bin_path, bin_name) = sibling(out, ".rows.f32")?;
    cap.bin.path = bin_name;
    let id = cap.seal();
    write(&bin_path, bin)?;
    write(
        out,
        &serde_json::to_vec_pretty(&cap).map_err(|e| e.to_string())?,
    )?;
    Ok(id)
}

struct Args {
    values: BTreeMap<String, String>,
}

fn parse_args(args: &[String], allowed: &[&str]) -> Result<Args, String> {
    let mut values = BTreeMap::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let Some(flag) = a.strip_prefix("--") else {
            return Err(format!("unexpected argument {a:?}\n\n{USAGE}"));
        };
        let (name, inline) = match flag.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (flag, None),
        };
        if !allowed.contains(&name) {
            return Err(format!("unknown flag --{name}\n\n{USAGE}"));
        }
        let value = match inline {
            Some(v) => v,
            None => it
                .next()
                .cloned()
                .ok_or_else(|| format!("--{name} needs a value"))?,
        };
        if values.insert(name.to_string(), value).is_some() {
            return Err(format!("--{name} given twice"));
        }
    }
    Ok(Args { values })
}

impl Args {
    fn req(&self, name: &str) -> Result<&str, String> {
        self.values
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("--{name} is required\n\n{USAGE}"))
    }
}

fn print_result(id: &str, out: &Path) {
    println!(
        "{}",
        serde_json::json!({"id": id, "path": out.display().to_string()})
    );
}

/// `pm-census extract route-hist`.
pub fn cli_route_hist(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| matches!(a.as_str(), "--help" | "-h")) {
        println!("{USAGE}");
        return Ok(());
    }
    let a = parse_args(args, &["dump", "model", "experts", "out"])?;
    let dump = Path::new(a.req("dump")?);
    let out = Path::new(a.req("out")?);
    let experts = a
        .values
        .get("experts")
        .map(|s| {
            s.parse::<u32>()
                .map_err(|_| format!("--experts {s:?}: expected a positive integer"))
        })
        .transpose()?;
    let bytes = read(dump)?;
    let (hist, ids) = route_hist(
        &bytes,
        source_of(dump, &bytes),
        a.values.get("model").cloned(),
        experts,
    )?;
    let id = write_route_hist(hist, &ids, out)?;
    print_result(&id, out);
    Ok(())
}

/// `pm-census extract act-capture`.
pub fn cli_act_capture(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| matches!(a.as_str(), "--help" | "-h")) {
        println!("{USAGE}");
        return Ok(());
    }
    let a = parse_args(args, &["dump", "out"])?;
    let dump = Path::new(a.req("dump")?);
    let out = Path::new(a.req("out")?);
    let bytes = read(dump)?;
    let (cap, bin) = act_capture(&bytes, source_of(dump, &bytes))?;
    let id = write_act_capture(cap, &bin, out)?;
    print_result(&id, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("pm-census-capture-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn route_rec(layer: u32, n: u32, k: u32, position: u32, ids: &[i32]) -> Vec<u8> {
        assert_eq!(ids.len(), (n * k) as usize);
        let mut v = Vec::new();
        for x in [layer, n, k, position] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in ids {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    fn act_rec(site: &str, layer: u32, rows: u32, k: u32, position: u32, data: &[f32]) -> Vec<u8> {
        assert_eq!(data.len(), (rows * k) as usize);
        let mut v = Vec::new();
        v.extend_from_slice(&ACT_MAGIC.to_le_bytes());
        v.extend_from_slice(&(site.len() as u32).to_le_bytes());
        v.extend_from_slice(site.as_bytes());
        for x in [layer, rows, k, position] {
            v.extend_from_slice(&x.to_le_bytes());
        }
        for x in data {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    fn src() -> SourceRef {
        SourceRef {
            path: "d.bin".into(),
            sha256: "s".into(),
        }
    }

    /// Layer 0: two chunks (2 rows, then 1 row), layer 3: one chunk (2 rows); k=2, E=4 in use.
    fn route_dump() -> Vec<u8> {
        let mut d = route_rec(0, 2, 2, 0, &[0, 1, 0, 2]);
        d.extend(route_rec(0, 1, 2, 2, &[0, 3]));
        d.extend(route_rec(3, 2, 2, 0, &[1, 3, 0, 2]));
        d
    }

    #[test]
    fn gini_and_max_share_known_values() {
        assert_eq!(gini(&[5, 5, 5, 5]), 0.0);
        // one expert takes everything: G = (n-1)/n
        assert!((gini(&[0, 0, 0, 8]) - 0.75).abs() < 1e-12);
        assert_eq!(gini(&[0, 0]), 0.0);
        // [1,3]: G = 2*(1*1+2*3)/(2*4) - 3/2 = 0.25
        assert!((gini(&[3, 1]) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn route_hist_counts_records_and_stats() {
        let dump = route_dump();
        let (h, bin) = route_hist(&dump, src(), Some("m".into()), None).unwrap();
        assert_eq!((h.n_experts, h.topk), (4, 2));
        assert_eq!(h.records.len(), 3);
        assert_eq!(
            h.records[1],
            RouteRecordRef {
                layer: 0,
                n: 1,
                k: 2,
                position: 2
            }
        );
        assert_eq!(h.per_layer.len(), 2);
        assert_eq!((h.per_layer[0].layer, h.per_layer[0].tokens), (0, 3));
        assert_eq!(h.per_layer[0].counts, vec![3, 1, 1, 1]);
        assert_eq!((h.per_layer[1].layer, h.per_layer[1].tokens), (3, 2));
        assert_eq!(h.per_layer[1].counts, vec![1, 1, 1, 1]);
        assert_eq!(h.total.tokens, 5);
        assert_eq!(h.total.counts, vec![4, 2, 2, 2]);
        assert_eq!(h.total.counts.iter().sum::<u64>(), 5 * 2);
        assert!((h.max_share - 0.4).abs() < 1e-12);
        // gini of [4,2,2,2]: sorted [2,2,2,4], 2*28/(4*10) - 5/4 = 0.15
        let expect = 2.0 * (2.0 + 4.0 + 6.0 + 16.0) / 40.0 - 5.0 / 4.0;
        assert!((h.gini - expect).abs() < 1e-12);
        // ids bin = payloads concatenated, headers stripped
        let want: Vec<i32> = vec![0, 1, 0, 2, 0, 3, 1, 3, 0, 2];
        assert_eq!(le_i32(&bin).collect::<Vec<_>>(), want);
        assert_eq!(h.ids.sha256, sha256_hex(&bin));
        assert_eq!(h.ids.layout.as_deref(), Some(ROUTE_IDS_LAYOUT));
    }

    #[test]
    fn route_hist_experts_override_and_refusals() {
        let dump = route_dump();
        let (h, _) = route_hist(&dump, src(), None, Some(8)).unwrap();
        assert_eq!(h.n_experts, 8);
        assert_eq!(h.total.counts.len(), 8);
        assert_eq!(h.total.counts[4..], [0, 0, 0, 0]);
        // --experts smaller than the max id refuses
        let e = route_hist(&dump, src(), None, Some(3)).err().unwrap();
        assert!(e.contains("outside [0,3)"), "{e}");
        // negative id
        let neg = route_rec(0, 1, 2, 0, &[0, -1]);
        assert!(route_hist(&neg, src(), None, None)
            .err()
            .unwrap()
            .contains("negative"));
        // truncated tail
        let mut cut = route_dump();
        cut.pop();
        assert!(route_hist(&cut, src(), None, None)
            .err()
            .unwrap()
            .contains("truncated"));
        // mixed k
        let mut mixed = route_rec(0, 1, 2, 0, &[0, 1]);
        mixed.extend(route_rec(1, 1, 1, 0, &[0]));
        assert!(route_hist(&mixed, src(), None, None)
            .err()
            .unwrap()
            .contains("k=1"));
        // empty and zero-row
        assert!(route_hist(&[], src(), None, None).is_err());
        assert!(route_hist(&route_rec(0, 0, 2, 0, &[]), src(), None, None).is_err());
    }

    #[test]
    fn route_hist_identity_is_canonical_and_order_pinned() {
        let dump = route_dump();
        let (mut a, _) = route_hist(&dump, src(), Some("m".into()), None).unwrap();
        let (mut b, _) = route_hist(&dump, src(), Some("m".into()), None).unwrap();
        a.ids.path = "x.ids.bin".into();
        b.ids.path = "x.ids.bin".into();
        let ia = a.seal();
        assert_eq!(ia, b.seal(), "identical inputs give identical ids");
        assert_eq!(ia.len(), 64);
        // the id is the sha of the id-less struct-order serialization
        let mut bare = a.clone();
        bare.id = None;
        assert_eq!(ia, sha256_hex(&serde_json::to_vec(&bare).unwrap()));
        let json = String::from_utf8(serde_json::to_vec(&bare).unwrap()).unwrap();
        assert!(json.starts_with("{\"schema\":\"route_hist.v1\",\"source\":{\"path\":\"d.bin\""));
        let pos = |k: &str| json.find(&format!("\"{k}\":")).unwrap();
        let order = [
            "source",
            "model",
            "n_experts",
            "topk",
            "records",
            "per_layer",
            "total",
            "gini",
            "max_share",
            "ids",
        ];
        for w in order.windows(2) {
            assert!(pos(w[0]) < pos(w[1]), "{} before {}", w[0], w[1]);
        }
        // a different model changes the id
        b.model = Some("other".into());
        assert_ne!(ia, b.seal());
        // the id field itself does not feed the id
        assert_eq!(a.canonical_id(), ia);
    }

    #[test]
    fn route_hist_cli_writes_json_and_bin_and_is_idempotent() {
        let dir = tmpdir("rh");
        let dump = dir.join("routes.bin");
        std::fs::write(&dump, route_dump()).unwrap();
        let out = dir.join("a3b_t8192.route_hist.json");
        let args = |o: &Path| -> Vec<String> {
            vec![
                "--dump".into(),
                dump.display().to_string(),
                "--model".into(),
                "a3b".into(),
                "--experts".into(),
                "4".into(),
                "--out".into(),
                o.display().to_string(),
            ]
        };
        cli_route_hist(&args(&out)).unwrap();
        let doc: RouteHist = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(doc.ids.path, "a3b_t8192.route_hist.ids.bin");
        assert_eq!(doc.id.as_deref(), Some(doc.canonical_id().as_str()));
        let bin = std::fs::read(dir.join(&doc.ids.path)).unwrap();
        assert_eq!(sha256_hex(&bin), doc.ids.sha256);
        assert_eq!(doc.source.sha256, sha256_hex(&route_dump()));
        // same dump again -> byte-identical JSON
        let out2 = dir.join("again.json");
        cli_route_hist(&args(&out2)).unwrap();
        let doc2: RouteHist = serde_json::from_slice(&std::fs::read(&out2).unwrap()).unwrap();
        // only the bin file name differs (stem), so compare the stat payload
        assert_eq!(doc.total, doc2.total);
        assert_eq!(doc.ids.sha256, doc2.ids.sha256);
        // missing flag / unknown flag
        assert!(cli_route_hist(&["--out".into(), "x".into()]).is_err());
        assert!(cli_route_hist(&["--bogus".into(), "1".into()]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn act_dump() -> Vec<u8> {
        let mut d = act_rec(
            "moe.gate_up.x",
            0,
            2,
            3,
            0,
            &[1.0, -2.0, 3.0, 0.0, 0.0, 2.0],
        );
        d.extend(act_rec("moe.down.x", 0, 1, 2, 0, &[4.0, -4.0]));
        d.extend(act_rec("moe.gate_up.x", 1, 1, 3, 2, &[1.0, 1.0, 1.0]));
        d
    }

    #[test]
    fn act_capture_groups_rows_per_site_with_stats() {
        let (c, bin) = act_capture(&act_dump(), src()).unwrap();
        // sites sorted by name: moe.down.x then moe.gate_up.x
        assert_eq!(c.sites.len(), 2);
        assert_eq!(
            c.sites[0],
            SiteEntry {
                site: "moe.down.x".into(),
                k: 2,
                rows_total: 1,
                records: 1,
                offset_bytes: 0
            }
        );
        assert_eq!(
            c.sites[1],
            SiteEntry {
                site: "moe.gate_up.x".into(),
                k: 3,
                rows_total: 3,
                records: 2,
                offset_bytes: 8
            }
        );
        let f: Vec<f32> = bin
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(
            f,
            vec![4.0, -4.0, 1.0, -2.0, 3.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(bin.len() as u64, 8 + 3 * 3 * 4);
        let d = &c.stats["moe.down.x"];
        assert_eq!((d.rms, d.absmax, d.mean), (4.0, 4.0, 0.0));
        let g = &c.stats["moe.gate_up.x"];
        // values 1,-2,3,0,0,2,1,1,1: sum 7, sumsq 21, n 9
        assert!((g.mean - 7.0 / 9.0).abs() < 1e-12);
        assert!((g.rms - (21.0f64 / 9.0).sqrt()).abs() < 1e-12);
        assert_eq!(g.absmax, 3.0);
        assert_eq!(c.bin.sha256, sha256_hex(&bin));
    }

    #[test]
    fn act_capture_refusals() {
        let mut bad = act_dump();
        bad[0] ^= 0xff;
        assert!(act_capture(&bad, src())
            .err()
            .unwrap()
            .contains("bad magic"));
        let mut cut = act_dump();
        cut.pop();
        assert!(act_capture(&cut, src())
            .err()
            .unwrap()
            .contains("truncated"));
        let mut kmix = act_rec("s", 0, 1, 2, 0, &[1.0, 2.0]);
        kmix.extend(act_rec("s", 1, 1, 3, 0, &[1.0, 2.0, 3.0]));
        assert!(act_capture(&kmix, src()).err().unwrap().contains("k=3"));
        let nan = act_rec("s", 0, 1, 1, 0, &[f32::NAN]);
        assert!(act_capture(&nan, src())
            .err()
            .unwrap()
            .contains("non-finite"));
        assert!(act_capture(&[], src()).is_err());
    }

    #[test]
    fn act_capture_cli_writes_json_and_bin() {
        let dir = tmpdir("ac");
        let dump = dir.join("acts.bin");
        std::fs::write(&dump, act_dump()).unwrap();
        let out = dir.join("raon_t4096.act_capture.json");
        cli_act_capture(&[
            "--dump".into(),
            dump.display().to_string(),
            "--out".into(),
            out.display().to_string(),
        ])
        .unwrap();
        let doc: ActCapture = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
        assert_eq!(doc.bin.path, "raon_t4096.act_capture.rows.f32");
        assert_eq!(doc.id.as_deref(), Some(doc.canonical_id().as_str()));
        let bin = std::fs::read(dir.join(&doc.bin.path)).unwrap();
        assert_eq!(sha256_hex(&bin), doc.bin.sha256);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
