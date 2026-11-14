// SPDX-License-Identifier: Apache-2.0
//! Stored co-processor stage measurements (schema 4). Loading never measures or recalibrates a device.
//!
//! A record prices one [`StageMap`]: `gpu.trunk_in_ms_per_8192` and the in-proj `families` are keyed by the
//! map's group names, each family's `layer` is its kind's name, and the optional out families are the kinds'
//! `out_family` names. The per-kind GPU terms keep their schema-4 spellings: `gdn_core_per_layer_ms` prices
//! kind 0 (recurrent mixer), `qsa_*` and `qsa_order` kind 1 (attention mixer).
use super::stage::StageMap;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

pub const CALIBRATION_SCHEMA_VERSION: u32 = 4;
/// Identity of the NPU serving numerics/transport the stored calibrations were measured against.
/// Deliberately not the crate version: a release bump does not change the NPU route, and tying the
/// two would orphan every shipped calibration on each bump. Change this when serving
/// numerics/transport change (and re-measure the shipped calibrations).
pub fn calibration_build_identity() -> String {
    "hipfire-0.4.2:npu-trunk-bf16-rowmajor-v1".to_owned()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceKey {
    pub arch: String,
    /// PCI address of the selected GPU (HIP's PCI bus ID), not an NPU slot.
    pub pci_bdf: String,
    pub npu_fw_version: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelKey {
    pub uname_release: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalKey {
    pub device: DeviceKey,
    /// Stable release + NPU numerical/transport contract, never a git SHA.
    #[serde(default = "calibration_build_identity")]
    pub build: String,
    pub kernel: KernelKey,
    /// `model_trunk_index_sha256` of the HFQ: index-only, PLE-shard-agnostic (monolithic == split base).
    pub model_trunk_index_sha256: String,
    /// Always `bf16` (BF16-direct), matching NpuCalibration::epilogue.
    pub epilogue: String,
    /// GPU-native trunk identity: `stock` or `overlay:<artifact marker>`.
    pub trunk_route: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuCalibration {
    pub trunk_in_ms_per_8192: BTreeMap<String, f64>,
    pub uniform_per_layer_ms: f64,
    pub gdn_core_per_layer_ms: f64,
    pub qsa_per_layer_ms: f64,
    pub qsa_growth_per_layer_ms_per_8192_prefix: f64,
    pub fixed_per_chunk_ms: f64,
    pub boundary_tax_ms: f64,
    pub min_bank_rows: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FamilyCalibration {
    pub layer: String,
    pub cols: usize,
    pub out_dtype: String,
    /// `false` keeps the family out of `select_families` until it is measured.
    #[serde(default = "selectable_default", skip_serializing_if = "is_true")]
    pub selectable: bool,
    /// Out-projection families only (`gdn.out`, `qsa.o`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out: Option<OutFamily>,
}
/// Out-projection shape (N = `cols`, K = `k`, `panel_cols` wide panels), its simulator rate alone, its NPU DRAM bytes
/// per kilo-op and the GPU o-proj GEMM time it removes from the layer per 8192 rows.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutFamily {
    pub k: usize,
    pub panel_cols: usize,
    pub tops_alone: f64,
    pub bytes_per_kop: f64,
    pub gpu_ms_per_8192: f64,
}
fn selectable_default() -> bool {
    true
}
fn is_true(v: &bool) -> bool {
    *v
}
fn is_zero(v: &f64) -> bool {
    *v == 0.0
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlueCalibration {
    pub pack_ms_8192: f64,
    /// GPU overhead per 8192 rows of a bank visit that moves families (events, split prepare/pack, wake).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub visit_ms_8192: f64,
    pub sync_ms: f64,
    pub consume_mode: String,
    pub consume_ms_per_col_8192: BTreeMap<String, f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NpuCalibration {
    pub tops_corun: f64,
    /// In-proj alone rate the co-run rate is measured against; the out families scale their alone rate by
    /// `tops_corun / tops_alone`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tops_alone: Option<f64>,
    pub bytes_per_kop: f64,
    pub epilogue: String,
    pub panel_cols: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContentionCalibration {
    pub alpha_overlap: f64,
    /// Contention per NPU DRAM GB of the overlapped family chains (0 = burst model only).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub ms_per_gb: f64,
    pub model: String,
}
/// Measured per-8192-row QSA phase times at the 16K pair (bank A prefix 0, bank B prefix 8192).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QsaOrder {
    pub core_a_ms: f64,
    pub core_b_ms: f64,
    pub post_ms: f64,
    pub inproj_chain_ms: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    pub schema_version: u32,
    pub key: CalKey,
    pub gpu: GpuCalibration,
    pub families: BTreeMap<String, FamilyCalibration>,
    pub glue: GlueCalibration,
    pub npu: NpuCalibration,
    pub contention: ContentionCalibration,
    pub margin: f64,
    /// Measured QSA phase windows for the order `C_A C_B P_A P_B` that moving `qsa.o` forces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qsa_order: Option<QsaOrder>,
}
impl Calibration {
    pub fn load(path: impl AsRef<Path>, map: &StageMap) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let cal: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        cal.validate(map)?;
        Ok(cal)
    }
    /// `path` if it exists, else the compiled-in `builtin` JSON record. Only a missing file falls back;
    /// invalid/unreadable files are authoritative.
    pub fn load_with_builtin(
        path: impl AsRef<Path>,
        map: &StageMap,
        builtin: &'static str,
    ) -> Result<(Self, &'static str), String> {
        let (bytes, source) = match fs::read(path) {
            Ok(bytes) => (std::borrow::Cow::Owned(bytes), "file"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                (std::borrow::Cow::Borrowed(builtin.as_bytes()), "builtin")
            }
            Err(e) => return Err(e.to_string()),
        };
        let cal: Self = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        cal.validate(map)?;
        Ok((cal, source))
    }
    pub fn store(&self, path: impl AsRef<Path>, map: &StageMap) -> Result<(), String> {
        self.validate(map)?;
        let json = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        fs::write(path, json).map_err(|e| e.to_string())
    }
    /// First mismatch in stable device/build/kernel/model/epilogue/trunk-route order. Caller
    /// must leave the route off and log this reason; never silently recalibrate.
    pub fn key_matches(&self, key: &CalKey) -> Result<(), String> {
        macro_rules! check { ($($field:ident).+) => {
            if self.key.$($field).+ != key.$($field).+ {
                return Err(format!("calibration key mismatch: {}", stringify!($($field).+)));
            }
        }; }
        check!(device.arch);
        check!(device.pci_bdf);
        check!(device.npu_fw_version);
        check!(build);
        check!(kernel.uname_release);
        check!(model_trunk_index_sha256);
        check!(epilogue);
        check!(trunk_route);
        Ok(())
    }
    /// Binds the built-in calibration (the only record that may be bound) to this host. Checks the hard
    /// fields only, in order `device.arch`, `build`, `model_trunk_index_sha256`, `epilogue`, `trunk_route`
    /// (same `calibration key mismatch: <field>` errors as `key_matches`), then copies the host's
    /// `device.pci_bdf` (the GPU's PCI address), `device.npu_fw_version` and `kernel.uname_release` from
    /// `key` into this record's key and returns the names of those that differed. `margin` is unchanged
    /// (built-ins carry 0.0). A record is untouched on error. A user calibration file never goes through
    /// this and keeps the fully strict `key_matches`.
    pub fn bind_builtin(&mut self, key: &CalKey) -> Result<Vec<&'static str>, String> {
        macro_rules! check { ($($field:ident).+) => {
            if self.key.$($field).+ != key.$($field).+ {
                return Err(format!("calibration key mismatch: {}", stringify!($($field).+)));
            }
        }; }
        check!(device.arch);
        check!(build);
        check!(model_trunk_index_sha256);
        check!(epilogue);
        check!(trunk_route);
        let mut differ = Vec::new();
        if self.key.device.pci_bdf != key.device.pci_bdf {
            self.key.device.pci_bdf.clone_from(&key.device.pci_bdf);
            differ.push("device.pci_bdf");
        }
        if self.key.device.npu_fw_version != key.device.npu_fw_version {
            self.key
                .device
                .npu_fw_version
                .clone_from(&key.device.npu_fw_version);
            differ.push("device.npu_fw_version");
        }
        if self.key.kernel.uname_release != key.kernel.uname_release {
            self.key
                .kernel
                .uname_release
                .clone_from(&key.kernel.uname_release);
            differ.push("kernel.uname_release");
        }
        Ok(differ)
    }
    /// Schema-4 invariants plus the record's family table against `map` (group names, kinds, out families).
    pub fn validate(&self, map: &StageMap) -> Result<(), String> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION {
            return Err("unsupported calibration schema_version".into());
        }
        if !self.margin.is_finite() || !(0.0..1.0).contains(&self.margin) {
            return Err("margin must be finite and in [0,1)".into());
        }
        if self.npu.panel_cols == 0 || self.gpu.min_bank_rows == 0 {
            return Err("panel_cols and min_bank_rows must be positive".into());
        }
        let epilogue = map.contract.calibration_name();
        if self.npu.epilogue != epilogue {
            return Err(format!("npu.epilogue must be {epilogue}"));
        }
        if self.key.epilogue != self.npu.epilogue {
            return Err("key.epilogue differs from npu.epilogue".into());
        }
        if self.key.trunk_route != "stock"
            && !self
                .key
                .trunk_route
                .strip_prefix("overlay:")
                .is_some_and(|marker| !marker.trim().is_empty())
        {
            return Err("key.trunk_route must be stock or overlay:<marker>".into());
        }
        if self.glue.consume_mode != "retile" && self.glue.consume_mode != "direct" {
            return Err("glue.consume_mode must be retile or direct".into());
        }
        if self.glue.consume_ms_per_col_8192.len() != 2 {
            return Err("expected bf16,f32 consume_ms_per_col_8192".into());
        }
        for dtype in ["bf16", "f32"] {
            let cost = self
                .glue
                .consume_ms_per_col_8192
                .get(dtype)
                .ok_or_else(|| format!("missing glue.consume_ms_per_col_8192.{dtype}"))?;
            if !cost.is_finite() || *cost < 0.0 {
                return Err(format!("invalid glue.consume_ms_per_col_8192.{dtype}"));
            }
        }
        for (name, value) in [
            ("device.arch", self.key.device.arch.as_str()),
            ("device.pci_bdf", &self.key.device.pci_bdf),
            ("device.npu_fw_version", &self.key.device.npu_fw_version),
            ("build", &self.key.build),
            ("kernel.uname_release", &self.key.kernel.uname_release),
            ("contention.model", &self.contention.model),
        ] {
            if value.trim().is_empty() {
                return Err(format!("{name} must not be empty"));
            }
        }
        if self.key.model_trunk_index_sha256.len() != 64
            || !self
                .key
                .model_trunk_index_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
        {
            return Err("model_trunk_index_sha256 must be 64 hex digits".into());
        }
        let names = map.groups.map(|g| g.name);
        if self.gpu.trunk_in_ms_per_8192.len() != names.len() {
            return Err(format!("expected {} calibration families", names.join(",")));
        }
        for name in self.families.keys() {
            if !names.contains(&name.as_str())
                && !map.kinds.iter().any(|k| k.out_family == name.as_str())
            {
                return Err(format!("unknown families.{name}"));
            }
        }
        for group in &map.groups {
            let name = group.name;
            let f = self
                .families
                .get(name)
                .ok_or_else(|| format!("missing families.{name}"))?;
            if f.cols == 0 || f.layer != map.kinds[group.kind].name {
                return Err(format!("invalid families.{name}"));
            }
            if f.out_dtype != "bf16" && f.out_dtype != "f32" {
                return Err(format!("invalid families.{name}.out_dtype"));
            }
            let cost = self
                .gpu
                .trunk_in_ms_per_8192
                .get(name)
                .ok_or_else(|| format!("missing gpu.trunk_in_ms_per_8192.{name}"))?;
            if !cost.is_finite() || *cost < 0.0 {
                return Err(format!("invalid gpu.trunk_in_ms_per_8192.{name}"));
            }
        }
        for (name, value) in [
            ("uniform_per_layer_ms", self.gpu.uniform_per_layer_ms),
            ("gdn_core_per_layer_ms", self.gpu.gdn_core_per_layer_ms),
            ("qsa_per_layer_ms", self.gpu.qsa_per_layer_ms),
            (
                "qsa_growth_per_layer_ms_per_8192_prefix",
                self.gpu.qsa_growth_per_layer_ms_per_8192_prefix,
            ),
            ("fixed_per_chunk_ms", self.gpu.fixed_per_chunk_ms),
            ("boundary_tax_ms", self.gpu.boundary_tax_ms),
            ("pack_ms_8192", self.glue.pack_ms_8192),
            ("visit_ms_8192", self.glue.visit_ms_8192),
            ("sync_ms", self.glue.sync_ms),
            ("bytes_per_kop", self.npu.bytes_per_kop),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        if !self.npu.tops_corun.is_finite() || self.npu.tops_corun <= 0.0 {
            return Err("tops_corun must be finite and positive".into());
        }
        if !self.contention.alpha_overlap.is_finite() || self.contention.alpha_overlap < 1.0 {
            return Err("alpha_overlap must be finite and >= 1".into());
        }
        if !self.contention.ms_per_gb.is_finite() || self.contention.ms_per_gb < 0.0 {
            return Err("ms_per_gb must be finite and nonnegative".into());
        }
        self.validate_out_families(map)?;
        Ok(())
    }
    /// The optional out-projection families (one per kind): their own shape, rate and GPU o-proj cost, the
    /// in-proj alone rate their co-run rate is scaled by, and (for kind 1's) the measured attention phase windows.
    fn validate_out_families(&self, map: &StageMap) -> Result<(), String> {
        for kind in &map.kinds {
            let (name, layer) = (kind.out_family, kind.name);
            let Some(f) = self.families.get(name) else {
                continue;
            };
            let Some(o) = &f.out else {
                return Err(format!("families.{name} needs an out block"));
            };
            if f.cols == 0 || f.layer != layer || f.out_dtype != "bf16" {
                return Err(format!("invalid families.{name}"));
            }
            if o.k == 0 || o.panel_cols == 0 {
                return Err(format!("invalid families.{name}.out"));
            }
            for (field, value) in [
                ("tops_alone", o.tops_alone),
                ("bytes_per_kop", o.bytes_per_kop),
                ("gpu_ms_per_8192", o.gpu_ms_per_8192),
            ] {
                if !value.is_finite() || value < 0.0 || (field == "tops_alone" && value == 0.0) {
                    return Err(format!("invalid families.{name}.out.{field}"));
                }
            }
            if !self
                .npu
                .tops_alone
                .is_some_and(|t| t.is_finite() && t > 0.0)
            {
                return Err("out families need a positive npu.tops_alone".into());
            }
        }
        for name in map.groups.map(|g| g.name) {
            if self.families.get(name).is_some_and(|f| f.out.is_some()) {
                return Err(format!("families.{name} must not carry an out block"));
            }
        }
        let attention_out = map.kinds[1].out_family;
        if self.families.contains_key(attention_out) {
            let q = self
                .qsa_order
                .as_ref()
                .ok_or_else(|| format!("{attention_out} needs qsa_order"))?;
            for (field, value) in [
                ("core_a_ms", q.core_a_ms),
                ("core_b_ms", q.core_b_ms),
                ("post_ms", q.post_ms),
                ("inproj_chain_ms", q.inproj_chain_ms),
            ] {
                if !value.is_finite() || value < 0.0 {
                    return Err(format!("qsa_order.{field} must be finite and nonnegative"));
                }
            }
        } else if self.qsa_order.is_some() {
            return Err(format!("qsa_order without families.{attention_out}"));
        }
        Ok(())
    }
}

/// Pure construction shared by runtime and CPU identity tests.
pub fn calibration_key(
    arch: &str,
    pci_bdf: &str,
    firmware: &str,
    release: &str,
    index: &str,
    trunk_route: &str,
) -> CalKey {
    CalKey {
        device: DeviceKey {
            arch: arch.into(),
            pci_bdf: pci_bdf.to_ascii_lowercase(),
            npu_fw_version: firmware.trim().into(),
        },
        build: calibration_build_identity(),
        kernel: KernelKey {
            uname_release: release.trim().into(),
        },
        model_trunk_index_sha256: index.into(),
        epilogue: "bf16".into(),
        trunk_route: trunk_route.into(),
    }
}

/// Probe the exact accel node railgun opens, without opening the runtime.
#[cfg(unix)]
pub fn npu_firmware() -> Result<String, String> {
    use std::os::unix::fs::FileTypeExt;
    let node =
        fs::metadata("/dev/accel/accel0").map_err(|e| format!("NPU accel0 unavailable: {e}"))?;
    if !node.file_type().is_char_device() {
        return Err("NPU accel0 is not an accel device".into());
    }
    let device = Path::new("/sys/class/accel/accel0/device");
    let driver = fs::read_link(device.join("driver")).map_err(|e| format!("NPU driver: {e}"))?;
    if driver.file_name().is_none_or(|name| name != "amdxdna") {
        return Err("NPU accel0 is not amdxdna".into());
    }
    let pci = fs::canonicalize(device).map_err(|e| format!("NPU PCI device: {e}"))?;
    if pci
        .file_name()
        .is_none_or(|name| !name.to_string_lossy().contains(':'))
    {
        return Err("NPU accel0 PCI identity unavailable".into());
    }
    let firmware = fs::read_to_string(device.join("fw_version"))
        .map_err(|e| format!("NPU fw_version: {e}"))?;
    if firmware.trim().is_empty() {
        return Err("NPU fw_version is empty".into());
    }
    Ok(firmware)
}

#[cfg(not(unix))]
pub fn npu_firmware() -> Result<String, String> {
    Err("NPU accel0 unavailable on this platform".into())
}

/// Read only unprivileged metadata, never open a GPU/NPU or time work.
pub fn runtime_calibration_key(
    arch: &str,
    pci_bdf: &str,
    index: &str,
    trunk_route: &str,
) -> Result<CalKey, String> {
    // railgun opens accel0; key that exact device, not the first PCI directory.
    let firmware = npu_firmware()?;
    runtime_key_with_firmware(arch, pci_bdf, &firmware, index, trunk_route)
}

pub fn runtime_key_with_firmware(
    arch: &str,
    pci_bdf: &str,
    firmware: &str,
    index: &str,
    trunk_route: &str,
) -> Result<CalKey, String> {
    let release = std::process::Command::new("uname")
        .arg("-r")
        .output()
        .map_err(|e| format!("uname -r: {e}"))?;
    if !release.status.success() {
        return Err("uname -r failed".into());
    }
    let release = std::str::from_utf8(&release.stdout).map_err(|e| e.to_string())?;
    // PerfOpt's armed message is privileged dmesg, with no sysfs/proc ABI.
    // Schema 3 deliberately keys kernel identity on uname_release alone.
    Ok(calibration_key(
        arch,
        pci_bdf,
        firmware,
        release,
        index,
        trunk_route,
    ))
}

/// Domain tag of [`model_trunk_index_sha256`]'s preimage; bump it with any change to the encoding.
const TRUNK_INDEX_DOMAIN: &[u8] = b"hipfire-npu-trunk-index-v1\0";

/// Lowercase-hex SHA256 of the trunk-relevant HFQ index: the same for a monolithic MQ4 and its
/// split MQ4 XTS base, different for any other trunk. Index-only: it reads the parsed arch id,
/// effective metadata JSON and tensor index, never tensor payload or file offsets. The PLE
/// sidecar split (128 embedded `...ngram_embedding.shard_N.weight` tensors vs. a `hipfire_ple_rows`
/// marker) therefore does not change it.
///
/// Preimage (all integers little-endian):
/// 1. `"hipfire-npu-trunk-index-v1\0"`
/// 2. `u32` arch id
/// 3. `u64` byte length + canonical metadata JSON: the effective metadata minus its top-level
///    `hipfire_ple_rows` key, compact, every object's keys sorted by UTF-8 bytes at every depth,
///    scalars as serde_json renders them
/// 4. `u64` tensor count, then per tensor sorted by name (names are unique in an HFQ index; file
///    order is deliberately ignored): `u32` name length + UTF-8 name, `u8` quant type,
///    `u32` dim count + `u32` per dim, `u64` data size
///
/// Only a tensor named exactly `PLE_SHARD_PREFIX` + canonical decimal `N` (non-empty ASCII digits,
/// no leading zero except `0`) + `PLE_SHARD_SUFFIX` is excluded; a similarly named tensor (e.g.
/// zero-padded `N`) stays in the digest.
///
/// Evidence for the exclusion set: the tensor-walk proof `rcpt/mq4split.split.json` shows the
/// monolithic MQ4 and its split MQ4 XTS base share 1197 common tensors with byte-identical
/// payloads and equal order, differing only in the 128 PLE shards and the `hipfire_ple_rows` marker.
pub fn model_trunk_index_sha256(hfq: &crate::hfq::HfqFile) -> Result<String, String> {
    use crate::ple_sidecar::{PLE_BASE_MARKER_KEY, PLE_SHARD_PREFIX, PLE_SHARD_SUFFIX};
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    fn is_ple_shard(name: &str) -> bool {
        name.strip_prefix(PLE_SHARD_PREFIX)
            .and_then(|rest| rest.strip_suffix(PLE_SHARD_SUFFIX))
            .is_some_and(|n| {
                n.bytes().all(|b| b.is_ascii_digit())
                    && matches!(n.as_bytes(), [_] | [b'1'..=b'9', ..])
            })
    }
    fn canonical_json(value: &serde_json::Value, out: &mut String) {
        use serde_json::Value;
        match value {
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    canonical_json(item, out);
                }
                out.push(']');
            }
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
                out.push('{');
                for (i, (key, item)) in entries.into_iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key).expect("string serializes"));
                    out.push(':');
                    canonical_json(item, out);
                }
                out.push('}');
            }
            scalar => out.push_str(&serde_json::to_string(scalar).expect("scalar serializes")),
        }
    }
    let mut metadata: serde_json::Value =
        serde_json::from_str(&hfq.metadata_json).map_err(|e| format!("metadata JSON: {e}"))?;
    metadata
        .as_object_mut()
        .ok_or("metadata JSON is not an object")?
        .remove(PLE_BASE_MARKER_KEY);
    let mut canonical = String::new();
    canonical_json(&metadata, &mut canonical);
    let mut tensors: Vec<_> = hfq
        .tensors()
        .iter()
        .filter(|t| !is_ple_shard(&t.name))
        .collect();
    tensors.sort_unstable_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut hasher = Sha256::new();
    hasher.update(TRUNK_INDEX_DOMAIN);
    hasher.update(hfq.arch_id.to_le_bytes());
    hasher.update((canonical.len() as u64).to_le_bytes());
    hasher.update(canonical.as_bytes());
    hasher.update((tensors.len() as u64).to_le_bytes());
    for t in tensors {
        hasher.update((t.name.len() as u32).to_le_bytes());
        hasher.update(t.name.as_bytes());
        hasher.update([t.quant_type]);
        hasher.update((t.shape.len() as u32).to_le_bytes());
        for dim in &t.shape {
            hasher.update(dim.to_le_bytes());
        }
        hasher.update((t.data_size as u64).to_le_bytes());
    }
    let mut hex = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut hex, "{byte:02x}").expect("write to String");
    }
    Ok(hex)
}
