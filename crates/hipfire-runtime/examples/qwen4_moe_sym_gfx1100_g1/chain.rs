// SPDX-License-Identifier: Apache-2.0
//! Down-chain interchange. The oracle holds no K3 (rotation + A4 quantizer) code. `--prepare-chain DIR` (CPU only)
//! writes, per chain-source GateUp scenario, the CPU-expected grouped BF16 H and `DIR/chain-h.json`
//! (schema `consumer24.gfx1100.chain-h.v1`); the Python helper
//! `crates/peacemaker-isa/tools/qwen4_moe_gfx1100_chain_a4.py --chain-dir DIR` rotates (R128, seeds 43/1043) and quantizes
//! (`pilot_solve.a4_emulate`, candidates 2) every row and writes `DIR/<name>.a4.bin` + `DIR/chain-a4.json` (schema
//! `consumer24.gfx1100.chain-a4.v1`). `--chain-dir DIR` loads that result, validates every hash/schema/row/seed binding and
//! the CPU H, and uses the file only as the Down `XSet`; it never calls Python or the device.
//!
//! H file: raw little-endian BF16, row-major `[rows][640]` over ALL grouped rows (padding rows 0x0000, sentinel-tile rows
//! 0xa5a5; every live row finite). A4 file: epoch-major `[5][rows][72]`, record `{f32 d, i32 s, 64 qs}` (element j in byte
//! j/2, low nibble first), grouped row `p` of group `g` at record `g*rows + p`; the oracle reads live rows only.

use crate::model::*;
use hipfire_runtime::mq_i4_reference::A4Block;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const H_SCHEMA: &str = "consumer24.gfx1100.chain-h.v1";
pub const A4_SCHEMA: &str = "consumer24.gfx1100.chain-a4.v1";
pub const COLS: usize = 640;
pub const GROUPS: usize = 5;
pub const S1_SEED: u64 = 43;
pub const S2_SEED: u64 = 1043;
pub const HOOKS_QUALIFIED: &str = "qualified";
pub const HOOKS_CPU_ONLY: &str = "cpu-only-self-check";

/// Deterministic entry name: `{mode}-{codec}-{scenario}` (mode `percodec` | `mixed`; codec = the run codec, the cold
/// codec in mixed mode).
pub fn chain_name(mode: &str, codec: Codec, scenario: &str) -> String {
    format!("{mode}-{}-{scenario}", codec.name())
}

pub fn h_raw(h: &[u16]) -> Vec<u8> {
    h.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn live_rows(row_class: &[u8]) -> Vec<usize> {
    (0..row_class.len())
        .filter(|&p| row_class[p] == 0)
        .collect()
}

pub fn tiles_sha(tiles: &[Tile]) -> String {
    let mut b = Vec::new();
    for t in tiles {
        b.extend(t.0.to_le_bytes());
        for s in t.1 {
            b.extend(s.to_le_bytes());
        }
    }
    sha256_hex(&b)
}

fn validate_name(n: &str) -> Res<()> {
    let ok = !n.is_empty()
        && n.len() <= 120
        && n.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && n.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
        && n != "chain-h.json"
        && n != "chain-a4.json";
    if ok {
        Ok(())
    } else {
        Err(format!("chain entry name {n:?} violates the helper's [A-Za-z0-9][A-Za-z0-9_.-]{{0,119}} contract"))
    }
}

pub struct PrepEntry {
    pub name: String,
    pub mode: &'static str,
    pub codec: Codec,
    pub scenario: String,
    pub rows: usize,
    pub live: Vec<usize>,
    pub h: Vec<u16>,
    pub tiles_sha256: String,
    pub div: usize,
    pub x_seed: u32,
    pub experts: Value,
}

/// Write the raw H file and return the `chain-h.json` entry.
pub fn write_entry(dir: &Path, e: &PrepEntry) -> Res<Value> {
    validate_name(&e.name)?;
    if e.h.len() != e.rows * COLS {
        return Err(format!(
            "{}: H has {} words for {} rows x {COLS}",
            e.name,
            e.h.len(),
            e.rows
        ));
    }
    let bytes = h_raw(&e.h);
    let file = format!("{}.h.bf16.bin", e.name);
    std::fs::write(dir.join(&file), &bytes).map_err(|x| format!("{file}: {x}"))?;
    Ok(
        json!({"name": e.name, "mode": e.mode, "codec": e.codec.name(), "scenario": e.scenario, "down_scenario": format!("chain_down__{}", e.scenario),
        "kind": "gate_up_h", "h_file": file, "h_sha256": sha256_hex(&bytes), "h_bytes": bytes.len(), "dtype": "bf16-le",
        "layout": "row-major [rows][640] over all grouped rows; padding rows 0x0000, sentinel-tile rows 0xa5a5",
        "rows": e.rows, "cols": COLS, "live_rows": e.live, "s1_seed": S1_SEED, "s2_seed": S2_SEED,
        "tiles_sha256": e.tiles_sha256, "x_row_div": e.div, "x_seed": e.x_seed, "experts": e.experts}),
    )
}

/// Write `chain-h.json` (root fields + `entries`), remove a stale `chain-a4.json`, return the sha256 of the written file.
pub fn write_chain_h(dir: &Path, mut root: Map<String, Value>, entries: Vec<Value>) -> Res<String> {
    let _ = std::fs::remove_file(dir.join("chain-a4.json"));
    root.insert("schema".into(), json!(H_SCHEMA));
    root.insert("entries".into(), Value::Array(entries));
    let bytes = serde_json::to_vec_pretty(&Value::Object(root)).map_err(err)?;
    std::fs::write(dir.join("chain-h.json"), &bytes).map_err(|x| format!("chain-h.json: {x}"))?;
    Ok(sha256_hex(&bytes))
}

pub struct RootExpect {
    pub source_sha256: Value,
    pub probe_table_sha256: Option<String>,
    /// sha256 of the `--divide-table` merged into the hooks, if any.
    pub divide_table_sha256: Option<String>,
    /// sha256s of every `--intrinsic-table` merged into the hooks, sorted ascending.
    pub intrinsic_tables_sha256: Vec<String>,
    pub hooks: &'static str,
    /// `(codec name, manifest sha256)` of every sample set the run uses.
    pub sample_manifests: Vec<(String, String)>,
    pub seed: u32,
    pub oracle_binary_sha256: String,
}

pub struct ChainDir {
    dir: PathBuf,
    pub root: Value,
    entries: BTreeMap<String, Value>,
    h_entries: BTreeMap<String, Value>,
    h_json_sha256: String,
    a4_json_sha256: String,
}

fn entries_of(doc: &Value, what: &str) -> Res<BTreeMap<String, Value>> {
    let arr = doc
        .get("entries")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{what}: no `entries` array"))?;
    let mut m = BTreeMap::new();
    for e in arr {
        let n = e
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{what}: entry without `name`"))?
            .to_string();
        if m.insert(n.clone(), e.clone()).is_some() {
            return Err(format!("{what}: duplicate entry {n}"));
        }
    }
    Ok(m)
}

fn rel(dir: &Path, f: &str) -> Res<PathBuf> {
    let p = Path::new(f);
    if f.is_empty()
        || p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err(format!(
            "chain file reference {f:?} must be a plain relative name"
        ));
    }
    Ok(dir.join(p))
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl ChainDir {
    pub fn load(dir: &Path) -> Res<Self> {
        let hp = dir.join("chain-h.json");
        let ap = dir.join("chain-a4.json");
        let miss = |p: &Path, e: std::io::Error| {
            format!("{}: {e} (run --prepare-chain on hiptrx, then the A4 helper, before passing --chain-dir)", p.display())
        };
        let hb = std::fs::read(&hp).map_err(|e| miss(&hp, e))?;
        let ab = std::fs::read(&ap).map_err(|e| miss(&ap, e))?;
        let hdoc: Value =
            serde_json::from_slice(&hb).map_err(|e| format!("{}: {e}", hp.display()))?;
        let root: Value =
            serde_json::from_slice(&ab).map_err(|e| format!("{}: {e}", ap.display()))?;
        if hdoc.get("schema").and_then(Value::as_str) != Some(H_SCHEMA) {
            return Err(format!("{}: schema is not {H_SCHEMA}", hp.display()));
        }
        if root.get("schema").and_then(Value::as_str) != Some(A4_SCHEMA) {
            return Err(format!("{}: schema is not {A4_SCHEMA}", ap.display()));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            entries: entries_of(&root, "chain-a4.json")?,
            h_entries: entries_of(&hdoc, "chain-h.json")?,
            root,
            h_json_sha256: sha256_hex(&hb),
            a4_json_sha256: sha256_hex(&ab),
        })
    }

    pub fn names(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Everything the run can verify about the prepared interchange without a device.
    pub fn root_problems(&self, exp: &RootExpect) -> Vec<String> {
        let mut p = Vec::new();
        let r = &self.root;
        if r.get("input_schema").and_then(Value::as_str) != Some(H_SCHEMA) {
            p.push(format!("chain-a4.json input_schema is not {H_SCHEMA}"));
        }
        if r.get("input_manifest_sha256").and_then(Value::as_str)
            != Some(self.h_json_sha256.as_str())
        {
            p.push(format!(
                "chain-a4.json input_manifest_sha256 {:?} != sha256(chain-h.json) {}",
                r.get("input_manifest_sha256"),
                self.h_json_sha256
            ));
        }
        for k in ["pilot_solve", "harness", "rotation_source"] {
            let ok = r.get(k).is_some_and(|o| {
                o.get("path").and_then(Value::as_str).is_some()
                    && o.get("sha256")
                        .and_then(Value::as_str)
                        .is_some_and(is_hex64)
            });
            if !ok {
                p.push(format!(
                    "chain-a4.json root `{k}` must carry {{path, sha256}}"
                ));
            }
        }
        for (k, expected) in [
            (
                "pilot_solve",
                "53965f2328e32d86c7b87460a02573a065424e220383891da210441dc9f6c0df".to_string(),
            ),
            (
                "harness",
                sha256_hex(include_bytes!(
                    "../../../peacemaker-isa/tools/qwen4_moe_gfx1100_chain_a4.py"
                )),
            ),
            (
                "rotation_source",
                sha256_hex(include_bytes!(
                    "../../../peacemaker-isa/tools/mq_i4_a4_oracle.py"
                )),
            ),
        ] {
            if r[k]["sha256"].as_str() != Some(expected.as_str()) {
                p.push(format!(
                    "chain-a4.json {k} source hash differs from the certified preparation source"
                ));
            }
        }
        for k in ["numpy", "torch"] {
            if r.get(k).is_none() {
                p.push(format!("chain-a4.json root lacks `{k}` version provenance"));
            }
        }
        if r.get("candidates").and_then(Value::as_u64) != Some(2) {
            p.push("chain-a4.json candidates != 2 (the fused gfx1201 c2 producer route)".into());
        }
        if r.get("source_sha256") != Some(&exp.source_sha256) {
            p.push("chain root source_sha256 differs from this oracle's sources (re-prepare with the same sources)".into());
        }
        if r.get("probe_table_sha256")
            .and_then(Value::as_str)
            .map(str::to_string)
            != exp.probe_table_sha256
        {
            p.push(format!(
                "chain root probe_table_sha256 {:?} != {:?}",
                r.get("probe_table_sha256"),
                exp.probe_table_sha256
            ));
        }
        if r.get("divide_table_sha256")
            .and_then(Value::as_str)
            .map(str::to_string)
            != exp.divide_table_sha256
        {
            p.push(format!(
                "chain root divide_table_sha256 {:?} != {:?}",
                r.get("divide_table_sha256"),
                exp.divide_table_sha256
            ));
        }
        // generic exact-key tables (`--intrinsic-table`), sorted by sha256; the field must be present (an empty list pins "none")
        if r.get("intrinsic_tables_sha256") != Some(&json!(exp.intrinsic_tables_sha256)) {
            p.push(format!("chain root intrinsic_tables_sha256 {:?} != {:?} (re-prepare the chain with the same --intrinsic-table set)", r.get("intrinsic_tables_sha256"), exp.intrinsic_tables_sha256));
        }
        if r.get("hooks").and_then(Value::as_str) != Some(exp.hooks) {
            p.push(format!(
                "chain root hooks {:?} != {:?}",
                r.get("hooks"),
                exp.hooks
            ));
        }
        if r.get("seed").and_then(Value::as_u64) != Some(u64::from(exp.seed)) {
            p.push(format!(
                "chain root seed {:?} != {}",
                r.get("seed"),
                exp.seed
            ));
        }
        for (c, s) in &exp.sample_manifests {
            if r.get("sample_manifest_sha256")
                .and_then(|m| m.get(c.as_str()))
                .and_then(Value::as_str)
                != Some(s.as_str())
            {
                p.push(format!("chain root sample_manifest_sha256[{c}] != {s}"));
            }
        }
        p
    }

    pub fn root_info(&self, exp: &RootExpect) -> Value {
        json!({"dir": self.dir.display().to_string(), "chain_a4_json_sha256": self.a4_json_sha256, "chain_h_json_sha256": self.h_json_sha256,
            "entries": self.entries.len(), "oracle_binary_sha256_in_chain": self.root.get("oracle_binary_sha256"),
            "oracle_binary_sha256_now": exp.oracle_binary_sha256,
            "oracle_binary_match": self.root.get("oracle_binary_sha256").and_then(Value::as_str) == Some(exp.oracle_binary_sha256.as_str()),
            "pilot_solve": self.root.get("pilot_solve"), "harness": self.root.get("harness"), "rotation_source": self.root.get("rotation_source"),
            "numpy": self.root.get("numpy"), "torch": self.root.get("torch"), "candidates": self.root.get("candidates"), "mixed_mode": self.root.get("mixed_mode")})
    }

    /// Validate entry `name` against this run's CPU GateUp expectation and return the Down A4 blocks, token-major
    /// `[rows][5]` (non-live rows are zero blocks). Any binding mismatch is an error; nothing is recomputed or substituted.
    pub fn tok_for(
        &self,
        name: &str,
        exp_h: &[u16],
        row_class: &[u8],
        tiles_sha256: &str,
    ) -> Res<Vec<A4Block>> {
        let e = self.entries.get(name).ok_or_else(|| format!("chain entry {name} is missing from chain-a4.json (prepare and run the A4 helper for this scenario)"))?;
        let he = self
            .h_entries
            .get(name)
            .ok_or_else(|| format!("chain entry {name} is missing from chain-h.json"))?;
        for k in [
            "h_file",
            "h_sha256",
            "rows",
            "cols",
            "s1_seed",
            "s2_seed",
            "live_rows",
            "scenario",
            "mode",
            "codec",
            "tiles_sha256",
        ] {
            if e.get(k) != he.get(k) {
                return Err(format!(
                    "entry {name}: `{k}` differs between chain-h.json and chain-a4.json"
                ));
            }
        }
        let rows = row_class.len();
        let num = |k: &str| e.get(k).and_then(Value::as_u64);
        if num("rows") != Some(rows as u64) || num("cols") != Some(COLS as u64) {
            return Err(format!(
                "entry {name}: rows/cols {:?}/{:?} != {rows}/{COLS}",
                e.get("rows"),
                e.get("cols")
            ));
        }
        if num("s1_seed") != Some(S1_SEED) || num("s2_seed") != Some(S2_SEED) {
            return Err(format!(
                "entry {name}: rotation seeds {:?}/{:?} != {S1_SEED}/{S2_SEED}",
                e.get("s1_seed"),
                e.get("s2_seed")
            ));
        }
        if e.get("tiles_sha256").and_then(Value::as_str) != Some(tiles_sha256) {
            return Err(format!(
                "entry {name}: tiles_sha256 {:?} differs from this run's route {tiles_sha256}",
                e.get("tiles_sha256")
            ));
        }
        let live = live_rows(row_class);
        let got_live: Option<Vec<usize>> = e.get("live_rows").and_then(Value::as_array).map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect()
        });
        if got_live.as_deref() != Some(live.as_slice()) {
            return Err(format!(
                "entry {name}: live_rows differ from this run's route"
            ));
        }
        let cpu_h = h_raw(exp_h);
        let want = sha256_hex(&cpu_h);
        let h_sha = e
            .get("h_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("entry {name}: no h_sha256"))?;
        if h_sha != want {
            return Err(format!("entry {name}: this run's CPU GateUp H has sha256 {want}, the prepared H has {h_sha}"));
        }
        let hf = std::fs::read(rel(
            &self.dir,
            e.get("h_file").and_then(Value::as_str).unwrap_or(""),
        )?)
        .map_err(err)?;
        if hf != cpu_h {
            return Err(format!(
                "entry {name}: h_file bytes differ from this run's CPU GateUp H"
            ));
        }
        if e.get("a4_layout").and_then(Value::as_str) != Some("epoch-major") {
            return Err(format!(
                "entry {name}: a4_layout {:?} is not epoch-major",
                e.get("a4_layout")
            ));
        }
        let af = std::fs::read(rel(
            &self.dir,
            e.get("a4_file").and_then(Value::as_str).unwrap_or(""),
        )?)
        .map_err(err)?;
        let want_bytes = GROUPS * rows * BLK;
        if num("a4_bytes") != Some(want_bytes as u64) || af.len() != want_bytes {
            return Err(format!(
                "entry {name}: a4 file is {} bytes, entry says {:?}, expected {want_bytes}",
                af.len(),
                e.get("a4_bytes")
            ));
        }
        if e.get("a4_sha256").and_then(Value::as_str) != Some(sha256_hex(&af).as_str()) {
            return Err(format!(
                "entry {name}: a4 file sha256 differs from a4_sha256"
            ));
        }
        let mut tok = vec![block_from_values(1.0, &[0i8; 128]); rows * GROUPS];
        for &p in &live {
            for g in 0..GROUPS {
                let o = (g * rows + p) * BLK;
                let b = &af[o..o + BLK];
                let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let s = i32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let mut qs = [0u8; 64];
                qs.copy_from_slice(&b[8..BLK]);
                let blk = A4Block { d, s, qs };
                let sum: i32 = (0..128).map(|j| i32::from(blk.element(j))).sum();
                if !d.is_finite() || sum != s {
                    return Err(format!("entry {name}: A4 record (group {g}, row {p}) is malformed (d finite: {}, element sum {sum} vs s {s})", d.is_finite()));
                }
                tok[p * GROUPS + g] = blk;
            }
        }
        Ok(tok)
    }
}
