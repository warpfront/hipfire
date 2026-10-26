// SPDX-License-Identifier: Apache-2.0
//! Flash-Next calibration identity: the compiled-in measured Halo records, the model trunk index digest and the
//! host facts a record is keyed on. The schema-4 record itself is `hipfire_runtime::coproc::calibration`.
use super::{arch_admits_auto, shapes::FLASH_NEXT, SWITCH};
use hipfire_runtime::coproc::{calibration_key, CalKey, Calibration, Mode};
use std::{fs, path::Path};

/// The compiled-in measured Halo record of a trunk route (`stock` or the overlay).
pub fn builtin_record(route: &str) -> &'static str {
    match route {
        "stock" => include_str!("calibrations/halo-xdna2-stock-v2.json"),
        _ => include_str!("calibrations/halo-xdna2-overlay-v4.json"),
    }
}

/// The calibration at `path`, else the built-in record of `route`, validated against [`FLASH_NEXT`]. Only a
/// missing file falls back; invalid/unreadable files are authoritative.
pub fn load_calibration(
    path: impl AsRef<Path>,
    route: &str,
) -> Result<(Calibration, &'static str), String> {
    Calibration::load_with_builtin(path, &FLASH_NEXT, builtin_record(route))
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

/// Host facts the armed `npu-trunk:` line reports so stock-kernel reports are triageable:
/// `kernel=<k> iommu=gpu:<g>,npu:<n> amdxdna=<x> npu_fw=<f>`. `<k>` is the kernel release;
/// `<g>`/`<n>` are the IOMMU group domain types of the GPU (by PCI address) and `accel0`, i.e. the
/// kernel's `identity` (iommu=pt), `DMA`, or `DMA-FQ` (the default translated domain); `<x>` is the
/// out-of-tree xdna-driver version, `in-tree` for the mainline module without a version, else
/// `unknown`; `<f>` is the NPU firmware version. Values are trimmed, whitespace becomes `_`, and
/// missing/empty/unreadable values are `unknown`. Reads only unprivileged proc/sysfs files.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HostFacts {
    pub kernel: Option<String>,
    pub iommu_gpu: Option<String>,
    pub iommu_npu: Option<String>,
    pub amdxdna: Option<String>,
    pub npu_fw: Option<String>,
}

impl HostFacts {
    /// Read the facts under `root` (`/` in production, a fixture in tests).
    #[cfg(unix)]
    pub fn read(root: &Path, gpu_pci: Option<&str>) -> Self {
        fn fact(path: &Path) -> Option<String> {
            let text = fs::read_to_string(path).ok()?;
            let text = text.split_whitespace().collect::<Vec<_>>().join("_");
            (!text.is_empty()).then_some(text)
        }
        let accel = root.join("sys/class/accel/accel0/device");
        let module = root.join("sys/module/amdxdna");
        Self {
            kernel: fact(&root.join("proc/sys/kernel/osrelease")),
            iommu_gpu: gpu_pci.and_then(|pci| {
                fact(
                    &root
                        .join("sys/bus/pci/devices")
                        .join(pci.to_lowercase())
                        .join("iommu_group/type"),
                )
            }),
            iommu_npu: fact(&accel.join("iommu_group/type")),
            amdxdna: fact(&module.join("version"))
                .or_else(|| module.is_dir().then(|| "in-tree".into())),
            npu_fw: fact(&accel.join("fw_version")),
        }
    }

    #[cfg(not(unix))]
    pub fn read(_root: &Path, _gpu_pci: Option<&str>) -> Self {
        Self::default()
    }
}

impl std::fmt::Display for HostFacts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = |value: &Option<String>| value.as_deref().unwrap_or("unknown").to_owned();
        write!(
            f,
            "kernel={} iommu=gpu:{},npu:{} amdxdna={} npu_fw={}",
            v(&self.kernel),
            v(&self.iommu_gpu),
            v(&self.iommu_npu),
            v(&self.amdxdna),
            v(&self.npu_fw),
        )
    }
}

pub fn host_facts(gpu_pci: Option<&str>) -> String {
    HostFacts::read(Path::new("/"), gpu_pci).to_string()
}

#[cfg(unix)]
fn host_facts_from(root: &Path, gpu_pci: Option<&str>) -> String {
    HostFacts::read(root, gpu_pci).to_string()
}

/// Rows and banks of the serving trunk transport (what `forward.rs` opens).
#[cfg(target_os = "linux")]
pub(crate) const TRUNK_ROWS_MAX: usize = 8192;
#[cfg(target_os = "linux")]
pub(crate) const TRUNK_BANKS: usize = 2;

/// Bytes the armed trunk pins against `RLIMIT_MEMLOCK` (userptr arenas of both banks), computed by
/// the layout code the transport open itself uses, for the serving 8192-row, all-family programs.
#[cfg(target_os = "linux")]
pub fn trunk_pin_bytes() -> u64 {
    use super::shapes::{FamilyMask, Shape};
    use railgun::npu::trunk::{ArenaLayout, Shape as Transport};
    let programs: Vec<(Transport, usize)> = [
        (Shape::GdnIn, Transport::GdnIn),
        (Shape::QsaIn, Transport::QsaIn),
    ]
    .into_iter()
    .map(|(shape, transport)| (transport, shape.column_map(FamilyMask::ALL).padded_cols()))
    .filter(|&(_, pitch)| pitch != 0)
    .collect();
    ArenaLayout::trunk_pinned_bytes(TRUNK_ROWS_MAX, TRUNK_BANKS, &programs)
        .expect("serving trunk rows are a valid layout") as u64
}

// Host memlock bound for `required` pinned bytes: required x1.25 rounded up to the MiB.
#[cfg(target_os = "linux")]
pub use railgun::npu::bounded_memlock_bytes;

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

pub(crate) fn runtime_key_with_firmware(
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

/// Capture the base before it moves into the source owner. Off does nothing; on reads only the
/// index the open file already parsed.
pub fn model_index_identity(
    hfq: &hipfire_runtime::hfq::HfqFile,
    arch: &str,
) -> Option<Result<String, String>> {
    if Mode::from_config(SWITCH, Mode::Auto)
        .ok()
        .is_none_or(|mode| mode.resolve(arch_admits_auto(arch)).is_off())
    {
        return None;
    }
    Some(model_trunk_index_sha256(hfq).map_err(|e| format!("model index identity: {e}")))
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
pub fn model_trunk_index_sha256(hfq: &hipfire_runtime::hfq::HfqFile) -> Result<String, String> {
    use hipfire_runtime::ple_sidecar::{PLE_BASE_MARKER_KEY, PLE_SHARD_PREFIX, PLE_SHARD_SUFFIX};
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

#[cfg(test)]
mod builtin_tests {
    use super::*;
    use hipfire_runtime::coproc::*;

    /// Normalized trunk index of the real Flash-Next MQ4: the monolithic gptq3 file and its split XTS base.
    pub(super) const NORMALIZED_TRUNK_INDEX: &str =
        "098bd75fd3a6a5174d06033f2b3ec7febeb7777e7eba4696b64cecca5e3418b6";

    #[test]
    fn measured_builtin_keys_validate_for_this_build() {
        for (json, route) in [
            (
                include_str!("calibrations/halo-xdna2-overlay-v4.json"),
                "overlay:gfx1151-zab-v1",
            ),
            (
                include_str!("calibrations/halo-xdna2-stock-v2.json"),
                "stock",
            ),
        ] {
            let cal: Calibration = serde_json::from_str(json).unwrap();
            cal.validate(&FLASH_NEXT).unwrap();
            assert_eq!(cal.key.build, calibration_build_identity());
            assert_eq!(cal.key.trunk_route, route);
            let key = calibration_key(
                "gfx1151",
                "0000:c0:00.0",
                "1.1.2.65",
                "7.3.0-rc7-perfopt+",
                NORMALIZED_TRUNK_INDEX,
                route,
            );
            cal.key_matches(&key).unwrap();
            assert!(!auto_admission(&FLASH_NEXT, &cal, &key).unwrap().is_empty());
            let mut mismatched = key;
            mismatched.device.npu_fw_version.push('x');
            assert_eq!(
                cal.key_matches(&mismatched),
                Err("calibration key mismatch: device.npu_fw_version".into())
            );
        }
    }

    fn builtin_with_host(
        route: &str,
        uname: &str,
        fw: &str,
        bdf: &str,
    ) -> (Calibration, CalKey, Result<Vec<&'static str>, String>) {
        let (mut cal, source) =
            load_calibration("/nonexistent/npu-trunk-calibration.json", route).unwrap();
        assert_eq!(source, "builtin");
        let key = calibration_key("gfx1151", bdf, fw, uname, NORMALIZED_TRUNK_INDEX, route);
        let differ = cal.bind_builtin(&key);
        (cal, key, differ)
    }

    #[test]
    fn builtin_binds_any_kernel_firmware_gpu_address_at_margin_zero() {
        for route in ["overlay:gfx1151-zab-v1", "stock"] {
            let (cal, key, differ) =
                builtin_with_host(route, "7.3.0-rc7-perfopt+", "1.1.2.65", "0000:c0:00.0");
            assert!(differ.unwrap().is_empty());
            assert_eq!(cal.margin, 0.0);
            let (cal, key2, differ) =
                builtin_with_host(route, "6.17.0-stock", "1.1.9.9", "0000:c1:00.0");
            assert_eq!(
                differ.unwrap(),
                [
                    "device.pci_bdf",
                    "device.npu_fw_version",
                    "kernel.uname_release"
                ]
            );
            assert_eq!(cal.margin, 0.0);
            assert_eq!(cal.key, key2);
            assert_ne!(key2, key);
            cal.key_matches(&key2).unwrap();
            let families = auto_admission(&FLASH_NEXT, &cal, &key2).unwrap();
            assert!(!families.is_empty());
            let p = plan(&FLASH_NEXT, &cal, families.into(), 16384);
            assert!(
                p.steps.iter().any(|s| matches!(s, Step::Pipe { .. })),
                "{route}: {p:?}"
            );
            let (cal, _, differ) =
                builtin_with_host(route, "6.17.0-stock", "1.1.2.65", "0000:c0:00.0");
            assert_eq!(differ.unwrap(), ["kernel.uname_release"]);
            assert_eq!(cal.margin, 0.0);
        }
    }

    #[test]
    fn builtin_hard_fields_and_file_calibrations_stay_strict() {
        let route = "stock";
        let (cal, _) = load_calibration("/nonexistent/npu-trunk-calibration.json", route).unwrap();
        let base = calibration_key(
            "gfx1151",
            "0000:c0:00.0",
            "1.1.2.65",
            "6.17.0-stock",
            NORMALIZED_TRUNK_INDEX,
            route,
        );
        let mut arch = base.clone();
        arch.device.arch = "gfx1150".into();
        assert_eq!(
            cal.clone().bind_builtin(&arch),
            Err("calibration key mismatch: device.arch".into())
        );
        let mut build = base.clone();
        build.build.push('x');
        assert_eq!(
            cal.clone().bind_builtin(&build),
            Err("calibration key mismatch: build".into())
        );
        let mut model = base.clone();
        model.model_trunk_index_sha256 = "0".repeat(64);
        assert_eq!(
            cal.clone().bind_builtin(&model),
            Err("calibration key mismatch: model_trunk_index_sha256".into())
        );
        let mut epilogue = base.clone();
        epilogue.epilogue = "f32".into();
        assert_eq!(
            cal.clone().bind_builtin(&epilogue),
            Err("calibration key mismatch: epilogue".into())
        );
        let mut other_route = base.clone();
        other_route.trunk_route = "overlay:gfx1151-zab-v1".into();
        assert_eq!(
            cal.clone().bind_builtin(&other_route),
            Err("calibration key mismatch: trunk_route".into())
        );
        // A refused bind leaves the record untouched.
        let mut refused = cal.clone();
        assert!(refused.bind_builtin(&model).is_err());
        assert_eq!(refused.key, cal.key);
        // A calibration file is never bound: the host's kernel differs from the record's, so the strict key refuses it.
        assert_eq!(
            auto_admission(&FLASH_NEXT, &cal, &base),
            Err("calibration key mismatch: kernel.uname_release".into())
        );
    }

    #[test]
    fn file_precedes_builtin_and_invalid_file_never_falls_back() {
        let dir = std::env::temp_dir().join(format!("hipfire-builtin-cal-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("calibration.json");
        let (mut builtin, source) = load_calibration(&path, "stock").unwrap();
        assert_eq!(source, "builtin");
        assert_eq!(builtin.key.trunk_route, "stock");
        builtin.key.device.npu_fw_version = "custom-file".into();
        builtin.store(&path, &FLASH_NEXT).unwrap();
        let (file, source) = load_calibration(&path, "stock").unwrap();
        assert_eq!(source, "file");
        assert_eq!(file.key.device.npu_fw_version, "custom-file");
        fs::write(&path, b"{}").unwrap();
        assert!(load_calibration(&path, "stock").is_err());
        fs::remove_file(&path).unwrap();
        let (overlay, source) = load_calibration(&path, "overlay:gfx1151-zab-v1").unwrap();
        assert_eq!(source, "builtin");
        assert_eq!(overlay.key.trunk_route, "overlay:gfx1151-zab-v1");
        fs::remove_dir(dir).unwrap();
    }

    /// Hard-binds a built-in to the real normalized digest, then proves every other hard field and a
    /// foreign/zero index are still refused (the PLE split is invisible to the key, nothing else is).
    #[test]
    fn builtins_accept_only_the_normalized_index_with_all_other_hard_fields() {
        for (route, other_route) in [
            ("stock", "overlay:gfx1151-zab-v1"),
            ("overlay:gfx1151-zab-v1", "stock"),
        ] {
            let (cal, _) =
                load_calibration("/nonexistent/npu-trunk-calibration.json", route).unwrap();
            assert_eq!(cal.key.model_trunk_index_sha256, NORMALIZED_TRUNK_INDEX);
            let base = calibration_key(
                "gfx1151",
                "0000:c0:00.0",
                "1.1.2.65",
                "6.17.0-stock",
                NORMALIZED_TRUNK_INDEX,
                route,
            );
            assert!(cal.clone().bind_builtin(&base).is_ok());
            for index in [
                "0".repeat(64),
                "8703bb1626ff48b976a84d5f4f7d18b4aad8e9d1d9b780a52f963d73cb83c013".into(),
            ] {
                let mut other = base.clone();
                other.model_trunk_index_sha256 = index;
                assert_eq!(
                    cal.clone().bind_builtin(&other),
                    Err("calibration key mismatch: model_trunk_index_sha256".into())
                );
            }
            let mut device = base.clone();
            device.device.arch = "gfx1150".into();
            assert_eq!(
                cal.clone().bind_builtin(&device),
                Err("calibration key mismatch: device.arch".into())
            );
            let mut build = base.clone();
            build.build.push('x');
            assert_eq!(
                cal.clone().bind_builtin(&build),
                Err("calibration key mismatch: build".into())
            );
            let mut epilogue = base.clone();
            epilogue.epilogue = "f32".into();
            assert_eq!(
                cal.clone().bind_builtin(&epilogue),
                Err("calibration key mismatch: epilogue".into())
            );
            let mut wrong_route = base;
            wrong_route.trunk_route = other_route.into();
            assert_eq!(
                cal.clone().bind_builtin(&wrong_route),
                Err("calibration key mismatch: trunk_route".into())
            );
        }
    }
}

#[cfg(test)]
mod trunk_index_tests {
    use super::*;
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::ple_sidecar::ple_shard_name;
    use std::io::Write;

    type Tensor = (String, u8, Vec<u32>, usize);

    /// Minimal HFQM v1 container: header, metadata JSON, index, 4096-aligned zeroed payloads.
    fn write_hfq(path: &Path, arch_id: u32, metadata: &str, tensors: &[Tensor]) {
        let mut index = Vec::new();
        index.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
        for (name, quant_type, shape, data_size) in tensors {
            index.extend_from_slice(&(name.len() as u16).to_le_bytes());
            index.extend_from_slice(name.as_bytes());
            index.push(*quant_type);
            index.push(shape.len() as u8);
            for dim in shape {
                index.extend_from_slice(&dim.to_le_bytes());
            }
            index.extend_from_slice(&0u32.to_le_bytes());
            index.extend_from_slice(&(*data_size as u64).to_le_bytes());
        }
        let unaligned = 32 + metadata.len() + index.len();
        let data_offset = unaligned.next_multiple_of(4096);
        let mut file = fs::File::create(path).unwrap();
        file.write_all(b"HFQM").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&arch_id.to_le_bytes()).unwrap();
        file.write_all(&(tensors.len() as u32).to_le_bytes())
            .unwrap();
        file.write_all(&32u64.to_le_bytes()).unwrap();
        file.write_all(&(data_offset as u64).to_le_bytes()).unwrap();
        file.write_all(metadata.as_bytes()).unwrap();
        file.write_all(&index).unwrap();
        file.write_all(&vec![0u8; data_offset - unaligned]).unwrap();
        for (_, _, _, data_size) in tensors {
            file.write_all(&vec![0u8; *data_size]).unwrap();
        }
    }

    fn trunk() -> Vec<Tensor> {
        vec![
            ("embed.weight".into(), 13, vec![8, 256], 32),
            ("layers.0.qkv.weight".into(), 13, vec![4, 256], 16),
        ]
    }
    fn shards() -> Vec<Tensor> {
        (0..128)
            .map(|s| (ple_shard_name(s), 3, vec![2, 32], 8))
            .collect()
    }
    const META: &str = r#"{"b":{"y":2,"x":[1,{"d":4,"c":3}]},"a":"s"}"#;
    const MARKER: &str =
        r#","hipfire_ple_rows":{"version":1,"location":"sidecar","shard_count":128}"#;

    fn digest(label: &str, arch_id: u32, metadata: &str, tensors: &[Tensor]) -> String {
        let path = std::env::temp_dir().join(format!(
            "hipfire-trunk-index-{}-{label}.hfq",
            std::process::id()
        ));
        write_hfq(&path, arch_id, metadata, tensors);
        let hfq = HfqFile::open(&path).unwrap();
        let digest = model_trunk_index_sha256(&hfq).unwrap();
        fs::remove_file(path).unwrap();
        digest
    }
    fn with(extra: Vec<Tensor>) -> Vec<Tensor> {
        [trunk(), extra].concat()
    }
    fn meta_with_marker() -> String {
        format!("{}{MARKER}}}", &META[..META.len() - 1])
    }

    #[test]
    fn ple_split_and_key_order_do_not_change_the_digest_but_trunk_fields_do() {
        let monolithic = digest("mono", 16, META, &with(shards()));
        assert_eq!(monolithic.len(), 64);
        // Split base: marker in metadata, no shard tensors.
        assert_eq!(
            monolithic,
            digest("split", 16, &meta_with_marker(), &trunk())
        );
        // Object key order and tensor file order are irrelevant (canonical, sorted).
        let reordered = r#"{"a":"s","b":{"x":[1,{"c":3,"d":4}],"y":2}}"#;
        assert_eq!(
            monolithic,
            digest(
                "order",
                16,
                reordered,
                &with(shards()).into_iter().rev().collect::<Vec<_>>()
            )
        );
        // Offsets are not an input: whitespace only moves them.
        assert_eq!(
            monolithic,
            digest(
                "space",
                16,
                r#"{ "b": {"y":2, "x":[1,{"d":4,"c":3}]}, "a":"s" }"#,
                &with(shards())
            )
        );
        let changed = |label: &str, arch_id: u32, metadata: &str, tensors: Vec<Tensor>| {
            assert_ne!(
                monolithic,
                digest(label, arch_id, metadata, &tensors),
                "{label}"
            );
        };
        changed("arch", 17, META, with(shards()));
        let mut tensors = with(shards());
        tensors[0].1 = 15;
        changed("quant", 16, META, tensors);
        let mut tensors = with(shards());
        tensors[1].2 = vec![4, 128];
        changed("shape", 16, META, tensors);
        let mut tensors = with(shards());
        tensors[0].3 = 64;
        changed("size", 16, META, tensors);
        changed(
            "meta-scalar",
            16,
            r#"{"b":{"y":3,"x":[1,{"d":4,"c":3}]},"a":"s"}"#,
            with(shards()),
        );
        changed(
            "meta-nested-key",
            16,
            r#"{"b":{"y":2,"x":[1,{"d":4,"e":3}]},"a":"s"}"#,
            with(shards()),
        );
        changed(
            "meta-added",
            16,
            r#"{"b":{"y":2,"x":[1,{"d":4,"c":3}]},"a":"s","z":0}"#,
            with(shards()),
        );
        changed("tensor-removed", 16, META, with(shards())[1..].to_vec());
        // Only a canonical-decimal PLE shard name is excluded: look-alikes are trunk tensors.
        for (label, name) in [
            ("padded", ple_shard_name(0).replace("shard_0", "shard_00")),
            (
                "other-layer",
                ple_shard_name(0).replace("layers.1.", "layers.2."),
            ),
            ("suffix", ple_shard_name(0).replace(".weight", ".bias")),
            (
                "nonnumeric",
                ple_shard_name(0).replace("shard_0", "shard_x"),
            ),
        ] {
            changed(
                label,
                16,
                META,
                with(vec![(name, 3, vec![2, 32], 8)])
                    .into_iter()
                    .chain(shards())
                    .collect(),
            );
        }
    }

    #[test]
    fn marker_is_ignored_only_at_the_top_level() {
        // A nested look-alike key is trunk metadata, not the split marker.
        let nested = r#"{"a":{"hipfire_ple_rows":1}}"#;
        assert_ne!(
            digest("nested-marker", 16, nested, &trunk()),
            digest("no-nested-marker", 16, r#"{"a":{}}"#, &trunk())
        );
    }

    /// Everything the loader and the prefill planner derive from a model's trunk digest, computed
    /// with the shared production functions in the order `forward.rs` calls them: `bind_builtin`,
    /// `auto_admission`, `select_families` (eligibility table), `crossover`, and the per-request
    /// `plan_executable` + `executable_plan_for` (auto arms from `n_star`, forced modes from row 0).
    /// Exact `f64` equality is intended: the same record and the same pure functions must give
    /// bit-identical numbers.
    #[derive(Debug, PartialEq)]
    struct Resolved {
        bound_fields_differing: Vec<&'static str>,
        family_mask: u8,
        eligibility: Vec<(u8, bool, bool, [f64; 3])>,
        best_families: u8,
        n_star: Option<usize>,
        pairs_16k_auto: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>,
        pairs_16k_forced: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>,
        pairs_64k_auto: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>,
        pairs_64k_forced: Vec<(std::ops::Range<usize>, std::ops::Range<usize>)>,
    }

    fn resolve(template: &Calibration, route: &str, digest: &str) -> Result<Resolved, String> {
        use crate::npu_trunk::FamilyMask;
        use hipfire_runtime::coproc::{
            auto_admission, crossover, executable_plan_for, plan_executable, select_families, Step,
        };
        let mut cal = template.clone();
        let key = calibration_key(
            "gfx1151",
            "0000:c0:00.0",
            "1.1.2.65",
            "7.3.0-rc7-perfopt+",
            digest,
            route,
        );
        let bound_fields_differing = cal.bind_builtin(&key)?;
        let families = auto_admission(&FLASH_NEXT, &cal, &key)?;
        let (_, report) = select_families(&FLASH_NEXT, &cal);
        let n_star = crossover(&FLASH_NEXT, &cal, families.into());
        let pairs = |rows: usize, armed: Option<usize>| {
            let planned = plan_executable(&FLASH_NEXT, &cal, families.into(), rows);
            executable_plan_for(&FLASH_NEXT, &cal, families.into(), &planned, rows, armed)
                .steps
                .into_iter()
                .filter_map(|step| match step {
                    Step::Pipe { a, b } => Some((a, b)),
                    Step::Gpu { .. } => None,
                })
                .collect::<Vec<_>>()
        };
        Ok(Resolved {
            bound_fields_differing,
            family_mask: FamilyMask::from(families).bits(),
            eligibility: report
                .subsets
                .iter()
                .map(|row| {
                    (
                        FamilyMask::from(row.families).bits(),
                        row.beats_margin_16384,
                        row.selectable,
                        row.relative,
                    )
                })
                .collect(),
            best_families: FamilyMask::from(report.best_families).bits(),
            n_star,
            pairs_16k_auto: pairs(16384, n_star),
            pairs_16k_forced: pairs(16384, Some(0)),
            pairs_64k_auto: pairs(65536, n_star),
            pairs_64k_forced: pairs(65536, Some(0)),
        })
    }

    const ROUTES: [&str; 2] = ["stock", "overlay:gfx1151-zab-v1"];

    fn builtin(route: &str) -> Calibration {
        load_calibration("/nonexistent/npu-trunk-calibration.json", route)
            .unwrap()
            .0
    }

    /// What the real Flash-Next MQ4 (monolithic and split base alike) resolves to: the built-in
    /// record bound with its own normalized identity, with the invariants that make it a usable proof.
    fn builtin_baseline(route: &str) -> Resolved {
        let baseline = resolve(
            &builtin(route),
            route,
            super::builtin_tests::NORMALIZED_TRUNK_INDEX,
        )
        .unwrap();
        assert_ne!(
            baseline.family_mask, 0,
            "{route}: built-in admits no family"
        );
        assert!(
            baseline.n_star.is_some(),
            "{route}: built-in has no crossover"
        );
        assert!(
            baseline
                .eligibility
                .iter()
                .any(|(bits, beats, selectable, _)| *bits == baseline.family_mask
                    && *beats
                    && *selectable),
            "{route}: admitted mask is not an eligible, margin-beating row"
        );
        let natural = vec![(0..8192, 8192..16384)];
        assert_eq!(baseline.pairs_16k_forced, natural, "{route}");
        assert_eq!(baseline.pairs_16k_auto, natural, "{route}");
        baseline
    }

    fn ple_layout(hfq: &HfqFile) -> (bool, usize) {
        use hipfire_runtime::ple_sidecar::{PLE_BASE_MARKER_KEY, PLE_SHARD_PREFIX};
        let metadata: serde_json::Value = serde_json::from_str(&hfq.metadata_json).unwrap();
        (
            metadata.get(PLE_BASE_MARKER_KEY).is_some(),
            hfq.tensors()
                .iter()
                .filter(|t| t.name.starts_with(PLE_SHARD_PREFIX))
                .count(),
        )
    }

    /// Always runs: a monolithic fixture (128 embedded PLE shards) and its split twin (marker, no
    /// shards) resolve to the identical admission, eligibility table, family mask and executable
    /// 16K/64K pair ranges as the built-in record keyed by the real normalized digest, while a
    /// different trunk is refused on `model_trunk_index_sha256` alone.
    #[test]
    fn split_and_monolithic_resolve_the_same_calibration_admission_and_pairs() {
        let mono = digest("proof-mono", 16, META, &with(shards()));
        let split = digest("proof-split", 16, &meta_with_marker(), &trunk());
        assert_eq!(mono, split);
        let other_trunk = digest(
            "proof-other",
            16,
            META,
            &[
                trunk(),
                vec![("layers.1.qkv.weight".into(), 13, vec![4, 256], 16)],
            ]
            .concat(),
        );
        assert_ne!(mono, other_trunk);
        for route in ROUTES {
            let baseline = builtin_baseline(route);
            // The fixture record is the built-in with only its trunk identity replaced by the fixture's
            // digest; every measured number is untouched.
            let mut record = builtin(route);
            let measured = serde_json::to_string(&record.gpu).unwrap();
            record.key.model_trunk_index_sha256.clone_from(&mono);
            assert_eq!(serde_json::to_string(&record.gpu).unwrap(), measured);
            for (label, digest) in [("monolithic", &mono), ("split", &split)] {
                let resolved = resolve(&record, route, digest).unwrap();
                assert_eq!(resolved, baseline, "{label} ({route})");
            }
            assert_eq!(
                resolve(&record, route, &other_trunk),
                Err("calibration key mismatch: model_trunk_index_sha256".into()),
                "{route}"
            );
        }
    }

    /// `NPU_TRUNK_TEST_MODELS` (test-harness input only, not a product setting) = `:`-separated real
    /// HFQ paths: it MUST name at least one monolithic MQ4 (embedded PLE shards) and one split MQ4 XTS
    /// base (`hipfire_ple_rows` marker). Each must digest to the built-in calibration identity and,
    /// through the shared bind/admission/planner functions, resolve to the same admission,
    /// eligibility table, family mask and executable 16K/64K pair ranges. Skips when unset.
    #[test]
    fn real_models_match_the_builtin_trunk_identity() {
        let Ok(paths) = std::env::var("NPU_TRUNK_TEST_MODELS") else {
            eprintln!("skip: set NPU_TRUNK_TEST_MODELS to `:`-separated HFQ paths");
            return;
        };
        let (mut monolithic, mut split) = (0, 0);
        let digests: Vec<_> = paths
            .split(':')
            .filter(|p| !p.is_empty())
            .map(|path| {
                let hfq = HfqFile::open(Path::new(path)).unwrap();
                match ple_layout(&hfq) {
                    (false, shards) if shards > 0 => monolithic += 1,
                    (true, 0) => split += 1,
                    layout => panic!("{path}: neither monolithic nor split PLE layout: {layout:?}"),
                }
                let digest = model_trunk_index_sha256(&hfq).unwrap();
                println!("model_trunk_index_sha256 {digest} {path}");
                (path, digest)
            })
            .collect();
        assert!(
            monolithic > 0 && split > 0,
            "NPU_TRUNK_TEST_MODELS must name a monolithic and a split base (got {monolithic} monolithic, {split} split)"
        );
        for route in ROUTES {
            let baseline = builtin_baseline(route);
            for (path, digest) in &digests {
                assert_eq!(
                    *digest,
                    builtin(route).key.model_trunk_index_sha256,
                    "{path} ({route})"
                );
                let resolved = resolve(&builtin(route), route, digest).unwrap();
                println!(
                    "route={route} mask={:#04x} n_star={:?} pairs16k={:?} {path}",
                    resolved.family_mask, resolved.n_star, resolved.pairs_16k_auto
                );
                assert_eq!(resolved, baseline, "{path} ({route})");
            }
        }
    }
}

#[cfg(all(test, unix))]
mod host_facts_tests {
    use super::*;

    struct TempRoot(std::path::PathBuf);
    impl TempRoot {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "hipfire-host-facts-{tag}-{}-{nanos}",
                std::process::id()
            ));
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn put(&self, rel: &str, text: &str) {
            let path = self.0.join(rel);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
    }
    impl Drop for TempRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn host_facts_stock_kernel_fixture() {
        let root = TempRoot::new("stock");
        root.put("proc/sys/kernel/osrelease", "7.2.8\n");
        root.put(
            "sys/bus/pci/devices/0000:c0:00.0/iommu_group/type",
            "DMA-FQ\n",
        );
        root.put("sys/class/accel/accel0/device/iommu_group/type", "DMA-FQ\n");
        fs::create_dir_all(root.0.join("sys/module/amdxdna")).unwrap();
        root.put("sys/class/accel/accel0/device/fw_version", "1.1.2.65\n");
        assert_eq!(
            host_facts_from(&root.0, Some("0000:C0:00.0")),
            "kernel=7.2.8 iommu=gpu:DMA-FQ,npu:DMA-FQ amdxdna=in-tree npu_fw=1.1.2.65"
        );
    }

    #[test]
    fn host_facts_perfopt_kernel_fixture() {
        let root = TempRoot::new("perfopt");
        root.put("proc/sys/kernel/osrelease", "7.3.0-rc7-perfopt+\n");
        root.put(
            "sys/bus/pci/devices/0000:c0:00.0/iommu_group/type",
            "identity\n",
        );
        root.put(
            "sys/class/accel/accel0/device/iommu_group/type",
            "identity\n",
        );
        root.put("sys/module/amdxdna/version", "2.21.0\n");
        root.put("sys/class/accel/accel0/device/fw_version", "1.1.2.65\n");
        assert_eq!(host_facts_from(&root.0, Some("0000:c0:00.0")),
            "kernel=7.3.0-rc7-perfopt+ iommu=gpu:identity,npu:identity amdxdna=2.21.0 npu_fw=1.1.2.65");
    }

    #[test]
    fn host_facts_missing_values_are_unknown() {
        let root = TempRoot::new("empty");
        let unknown = "kernel=unknown iommu=gpu:unknown,npu:unknown amdxdna=unknown npu_fw=unknown";
        assert_eq!(host_facts_from(&root.0, Some("0000:c0:00.0")), unknown);
        assert_eq!(host_facts_from(&root.0, None), unknown);
        root.put("sys/bus/pci/devices/0000:c0:00.0/iommu_group/type", "DMA\n");
        root.put("proc/sys/kernel/osrelease", "7.2.8\n");
        assert_eq!(
            host_facts_from(&root.0, None),
            "kernel=7.2.8 iommu=gpu:unknown,npu:unknown amdxdna=unknown npu_fw=unknown"
        );
    }

    #[test]
    fn host_facts_struct_fields_match_the_line() {
        let root = TempRoot::new("fields");
        root.put("proc/sys/kernel/osrelease", "7.3.0-rc7-perfopt+\n");
        root.put(
            "sys/bus/pci/devices/0000:c0:00.0/iommu_group/type",
            "identity\n",
        );
        root.put("sys/class/accel/accel0/device/iommu_group/type", "DMA-FQ\n");
        fs::create_dir_all(root.0.join("sys/module/amdxdna")).unwrap();
        let facts = HostFacts::read(&root.0, Some("0000:C0:00.0"));
        assert_eq!(
            facts,
            HostFacts {
                kernel: Some("7.3.0-rc7-perfopt+".into()),
                iommu_gpu: Some("identity".into()),
                iommu_npu: Some("DMA-FQ".into()),
                amdxdna: Some("in-tree".into()),
                npu_fw: None,
            }
        );
        assert_eq!(
            facts.to_string(),
            host_facts_from(&root.0, Some("0000:C0:00.0"))
        );
        assert_eq!(facts.to_string(),
            "kernel=7.3.0-rc7-perfopt+ iommu=gpu:identity,npu:DMA-FQ amdxdna=in-tree npu_fw=unknown");
    }
}

#[cfg(all(test, target_os = "linux"))]
mod pin_tests {
    use super::*;

    #[test]
    fn serving_trunk_pins_two_banks_of_userptr_arenas() {
        // 2 x (A 8192x2880 + GdnIn C 272,629,760 + QsaIn C 230,686,720 + five 4 KiB pages).
        assert_eq!(trunk_pin_bytes(), 1_053_859_840);
        assert_eq!(bounded_memlock_bytes(trunk_pin_bytes()), 1_318_060_032);
    }
}
