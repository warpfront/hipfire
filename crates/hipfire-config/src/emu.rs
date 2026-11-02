//! SKU emulation profile registry (EMU-DESIGN.md §1). Capacity-only; no CU masks on RDNA.
//!
//! One authoritative table of target SKUs, the opt-in `HIPFIRE_EMU` parser and the
//! pure host-compatibility check. CPU-only: nothing here touches HIP, ROCr or the
//! ambient environment; every value is read from the immutable [`ProcessConfig`]
//! snapshot (`developer.emu*` keys, i.e. the `HIPFIRE_EMU*` compatibility names).
//!
//! Units: capacities are GiB (`n << 30`), cache sizes MiB (`n << 20`), register
//! files KiB (`n << 10`), bandwidth is decimal bytes/s (`GB/s * 1_000_000_000`).
//! Nominal clocks and PCIe link widths are deliberately not stored in the frozen
//! row; confirmed facts are recorded in the row's doc comment only.
//!
//! Safety: this module never creates or reads a CU mask. Mask environment variables
//! are refused by [`crate::devices::cu_mask_env_guard`], which this module does not
//! touch.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ProcessConfig;

/// Snapshot key (and compatibility env name) selecting the profile.
const ENV_PROFILE: &str = "HIPFIRE_EMU";
/// UMA scenario host-RAM override (`developer.emu_host_bytes`).
const ENV_HOST_BYTES: &str = "HIPFIRE_EMU_HOST_BYTES";
/// UMA scenario BIOS/display carve-out override (`developer.emu_carveout_bytes`).
const ENV_CARVEOUT_BYTES: &str = "HIPFIRE_EMU_CARVEOUT_BYTES";
/// UMA scenario OS/application headroom override (`developer.emu_host_reserve_bytes`).
const ENV_HOST_RESERVE_BYTES: &str = "HIPFIRE_EMU_HOST_RESERVE_BYTES";
/// UMA scenario aggregate virtual GTT limit override (`developer.emu_gtt_bytes`).
const ENV_GTT_BYTES: &str = "HIPFIRE_EMU_GTT_BYTES";

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const KIB: u64 = 1 << 10;
const GB_S: u64 = 1_000_000_000;

/// Target device memory model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryKind {
    /// Dedicated VRAM of `capacity_bytes` (advertised, not guaranteed allocatable).
    Discrete { capacity_bytes: u64 },
    /// Unified memory; capacity comes from the [`UmaScenario`].
    Uma,
}

/// How faithfully the host can stand in for the target SKU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Fidelity {
    /// Same die and CU count as the host's physical device; capacity cap only.
    CapacityOnly,
    /// Capacity cap plus a measured-input derating estimate.
    CapacityEstimate,
    /// Different die than the host; estimate is an approximation.
    CrossDieEstimate,
    /// Platform memory scenario, not execution on the named laptop part.
    UmaScenario,
}

/// One registry row. Target gfx / CUs / caches are metadata for reporting and
/// the estimate model only; they never rewrite the host's execution arch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct EmuProfile {
    pub id: &'static str,
    /// Exact detected gfx the host must report to be allowed this profile.
    pub host_gfx: &'static str,
    /// Target SKU gfx (metadata only).
    pub target_gfx: &'static str,
    /// Target physical CUs (not HIP WGP count).
    pub target_cus: u32,
    pub memory: MemoryKind,
    /// Peak external memory bandwidth, decimal bytes/s.
    pub peak_bw_bytes_s: u64,
    /// Infinity Cache bytes; `0` = none.
    pub infinity_cache_bytes: u64,
    pub l2_bytes: u64,
    pub vgpr_file_bytes: u64,
    pub fidelity: Fidelity,
}

/// Radeon RX 9070 XT, 16 GiB.
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html>
///
/// PCIe 5.0 x16. Closest profile to the R9700 host (same die, 64 CUs).
const RX9070XT: EmuProfile = EmuProfile {
    id: "rx9070xt",
    host_gfx: "gfx1201",
    target_gfx: "gfx1201",
    target_cus: 64,
    memory: MemoryKind::Discrete {
        capacity_bytes: 16 * GIB,
    },
    peak_bw_bytes_s: 640 * GB_S,
    infinity_cache_bytes: 64 * MIB,
    l2_bytes: 8 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CapacityOnly,
};

/// Radeon RX 9070, 16 GiB.
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html>
///
/// AMD page confirms 640 GB/s, 2520 MHz boost, 220 W; PCIe 5.0 x16.
const RX9070: EmuProfile = EmuProfile {
    id: "rx9070",
    host_gfx: "gfx1201",
    target_gfx: "gfx1201",
    target_cus: 56,
    memory: MemoryKind::Discrete {
        capacity_bytes: 16 * GIB,
    },
    peak_bw_bytes_s: 640 * GB_S,
    infinity_cache_bytes: 64 * MIB,
    l2_bytes: 8 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CapacityEstimate,
};

/// Radeon RX 9060 XT, 16 GiB (gfx1200 die, cross-die approximation on gfx1201).
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html>
///
/// AMD page confirms 320 GB/s, 3130 MHz boost, 160 W for the 16-GB variant; PCIe 5.0 x16.
const RX9060XT: EmuProfile = EmuProfile {
    id: "rx9060xt",
    host_gfx: "gfx1201",
    target_gfx: "gfx1200",
    target_cus: 32,
    memory: MemoryKind::Discrete {
        capacity_bytes: 16 * GIB,
    },
    peak_bw_bytes_s: 320 * GB_S,
    infinity_cache_bytes: 32 * MIB,
    l2_bytes: 4 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CrossDieEstimate,
};

/// Radeon RX 9060 XT, 8 GiB (gfx1200 die, cross-die approximation on gfx1201).
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html>
///
/// Same die and bandwidth as the 16 GiB variant; PCIe 5.0 x16.
const RX9060XT_8G: EmuProfile = EmuProfile {
    id: "rx9060xt-8g",
    host_gfx: "gfx1201",
    target_gfx: "gfx1200",
    target_cus: 32,
    memory: MemoryKind::Discrete {
        capacity_bytes: 8 * GIB,
    },
    peak_bw_bytes_s: 320 * GB_S,
    infinity_cache_bytes: 32 * MIB,
    l2_bytes: 4 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CrossDieEstimate,
};

/// Radeon RX 7900 XT, 20 GiB.
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html>
///
/// PCIe link generation/width not recorded.
const RX7900XT: EmuProfile = EmuProfile {
    id: "rx7900xt",
    host_gfx: "gfx1100",
    target_gfx: "gfx1100",
    target_cus: 84,
    memory: MemoryKind::Discrete {
        capacity_bytes: 20 * GIB,
    },
    peak_bw_bytes_s: 800 * GB_S,
    infinity_cache_bytes: 80 * MIB,
    l2_bytes: 6 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CapacityEstimate,
};

/// Radeon RX 7800 XT, 16 GiB (gfx1101 die, cross-die approximation on gfx1100).
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html>
///
/// PCIe link generation/width unknown (not recorded).
const RX7800XT: EmuProfile = EmuProfile {
    id: "rx7800xt",
    host_gfx: "gfx1100",
    target_gfx: "gfx1101",
    target_cus: 60,
    memory: MemoryKind::Discrete {
        capacity_bytes: 16 * GIB,
    },
    peak_bw_bytes_s: 624 * GB_S,
    infinity_cache_bytes: 64 * MIB,
    l2_bytes: 4 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CrossDieEstimate,
};

/// Radeon RX 7700 XT, 12 GiB (gfx1101 die, cross-die approximation on gfx1100).
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7700-xt.html>
///
/// PCIe link generation/width unknown (not recorded).
const RX7700XT: EmuProfile = EmuProfile {
    id: "rx7700xt",
    host_gfx: "gfx1100",
    target_gfx: "gfx1101",
    target_cus: 54,
    memory: MemoryKind::Discrete {
        capacity_bytes: 12 * GIB,
    },
    peak_bw_bytes_s: 432 * GB_S,
    infinity_cache_bytes: 48 * MIB,
    l2_bytes: 4 * MIB,
    vgpr_file_bytes: 768 * KIB,
    fidelity: Fidelity::CrossDieEstimate,
};

/// Radeon RX 7600, 8 GiB (gfx1102 die, cross-die/occupancy approximation on gfx1100).
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (external bandwidth): <https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7600.html>
///
/// PCIe link generation/width unknown (not recorded). The 512 KiB register file
/// is smaller than the gfx1100 host's, so occupancy differs.
const RX7600: EmuProfile = EmuProfile {
    id: "rx7600",
    host_gfx: "gfx1100",
    target_gfx: "gfx1102",
    target_cus: 32,
    memory: MemoryKind::Discrete {
        capacity_bytes: 8 * GIB,
    },
    peak_bw_bytes_s: 288 * GB_S,
    infinity_cache_bytes: 32 * MIB,
    l2_bytes: 2 * MIB,
    vgpr_file_bytes: 512 * KIB,
    fidelity: Fidelity::CrossDieEstimate,
};

/// Ryzen AI 9 HX 370 (Strix Point, gfx1150) platform scenario on a Strix Halo host.
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (platform memory): <https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html>
///
/// 128 GB/s is the LPDDR5X-8000 platform maximum, not an immutable laptop
/// bandwidth guarantee. Capacity is the configurable [`UmaScenario`]; this is a
/// platform scenario, not HX370 execution. No Infinity Cache.
const STRIX_POINT: EmuProfile = EmuProfile {
    id: "strix-point",
    host_gfx: "gfx1151",
    target_gfx: "gfx1150",
    target_cus: 16,
    memory: MemoryKind::Uma,
    peak_bw_bytes_s: 128 * GB_S,
    infinity_cache_bytes: 0,
    l2_bytes: 2 * MIB,
    vgpr_file_bytes: 512 * KIB,
    fidelity: Fidelity::UmaScenario,
};

/// Ryzen AI 7 350 (Krackan Point, gfx1152) platform scenario on a Strix Halo host.
///
/// - ROCm gfx/CUs/cache/register-file: <https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html>
/// - AMD product page (platform memory): <https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-7-350.html>
///
/// 128 GB/s is the LPDDR5X-8000 platform maximum, not an immutable laptop
/// bandwidth guarantee. Capacity is the configurable [`UmaScenario`]; this is a
/// platform scenario, not AI7-350 execution. No Infinity Cache.
const KRACKAN: EmuProfile = EmuProfile {
    id: "krackan",
    host_gfx: "gfx1151",
    target_gfx: "gfx1152",
    target_cus: 8,
    memory: MemoryKind::Uma,
    peak_bw_bytes_s: 128 * GB_S,
    infinity_cache_bytes: 0,
    l2_bytes: MIB,
    vgpr_file_bytes: 512 * KIB,
    fidelity: Fidelity::UmaScenario,
};

/// The single authoritative SKU table. No per-backend tables.
pub static PROFILES: &[EmuProfile] = &[
    RX9070XT,
    RX9070,
    RX9060XT,
    RX9060XT_8G,
    RX7900XT,
    RX7800XT,
    RX7700XT,
    RX7600,
    STRIX_POINT,
    KRACKAN,
];

/// Errors from profile selection, override parsing and host compatibility.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum EmuError {
    #[error("unknown HIPFIRE_EMU profile `{id}` (ids are exact lower-case; known: {known})")]
    UnknownProfile { id: String, known: String },
    #[error(
        "{keys} set without an active HIPFIRE_EMU profile; the UMA overrides require a profile"
    )]
    OverrideWithoutProfile { keys: String },
    #[error("{name} is a UMA-only override but profile `{profile}` is a discrete SKU")]
    OverrideOnDiscrete { name: &'static str, profile: String },
    #[error("invalid {name}=`{value}`: {reason}")]
    InvalidOverride {
        name: &'static str,
        value: String,
        reason: &'static str,
    },
    #[error("invalid UMA scenario: {0}")]
    InvalidUma(String),
    #[error("invalid emu options: {0}")]
    InvalidOptions(String),
    #[error("emu host fact unknown: {fact} (unknown facts fail closed)")]
    UnknownHostFact { fact: &'static str },
    #[error("emu profile `{profile}` requires host gfx `{required}`, detected `{detected}`")]
    WrongGfx {
        profile: &'static str,
        required: &'static str,
        detected: String,
    },
    #[error("emu profile `{profile}` requires an {required} host, detected an {detected} device")]
    WrongKind {
        profile: &'static str,
        required: &'static str,
        detected: &'static str,
    },
    #[error("emu profile `{profile}` requires >= {required} physical CUs, host has {detected}")]
    TooFewCus {
        profile: &'static str,
        required: u32,
        detected: u32,
    },
    #[error(
        "emu profile `{profile}` requires >= {required} bytes of real {what} memory, host has {detected}"
    )]
    TooLittleMemory {
        profile: &'static str,
        what: &'static str,
        required: u64,
        detected: u64,
    },
}

/// Look up a profile by exact lower-case id. Unknown ids fail; they never
/// silently disable emulation.
pub fn profile(id: &str) -> Result<&'static EmuProfile, EmuError> {
    PROFILES
        .iter()
        .find(|candidate| candidate.id == id)
        .ok_or_else(|| EmuError::UnknownProfile {
            id: id.to_owned(),
            known: PROFILES.iter().map(|p| p.id).collect::<Vec<_>>().join(", "),
        })
}

/// UMA laptop scenario (EMU-DESIGN.md §3). Explicit scenario assumptions, not
/// AMD limits. Defaults: H = 32 GiB, C = 1 GiB, R = 4 GiB, G = 16 GiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UmaScenario {
    /// H: host RAM of the emulated laptop.
    pub host_total_bytes: u64,
    /// C: BIOS/display carve-out reservation.
    pub carveout_bytes: u64,
    /// R: OS/ordinary CPU application headroom reservation.
    pub host_reserve_bytes: u64,
    /// G: aggregate virtual GTT limit (the emulated device capacity).
    pub gtt_limit_bytes: u64,
}

impl Default for UmaScenario {
    fn default() -> Self {
        Self {
            host_total_bytes: 32 * GIB,
            carveout_bytes: GIB,
            host_reserve_bytes: 4 * GIB,
            gtt_limit_bytes: 16 * GIB,
        }
    }
}

impl UmaScenario {
    /// S = H - C - R, the managed shared-host allowance. Fails on overflow or
    /// when `C + R >= H`.
    pub fn shared_host_bytes(&self) -> Result<u64, EmuError> {
        let reserved = self
            .carveout_bytes
            .checked_add(self.host_reserve_bytes)
            .ok_or_else(|| {
                EmuError::InvalidUma(format!(
                    "carve-out {} + host reserve {} overflows u64",
                    self.carveout_bytes, self.host_reserve_bytes
                ))
            })?;
        if reserved >= self.host_total_bytes {
            return Err(EmuError::InvalidUma(format!(
                "carve-out {} + host reserve {} must be < host total {}",
                self.carveout_bytes, self.host_reserve_bytes, self.host_total_bytes
            )));
        }
        // `reserved < host_total_bytes`, so this cannot underflow; checked anyway.
        self.host_total_bytes
            .checked_sub(reserved)
            .ok_or_else(|| EmuError::InvalidUma("shared-host allowance underflows u64".to_owned()))
    }

    /// Enforce `C + R < H` and `0 < G <= S = H - C - R` with checked arithmetic.
    pub fn validate(&self) -> Result<(), EmuError> {
        let shared = self.shared_host_bytes()?;
        if self.gtt_limit_bytes == 0 {
            return Err(EmuError::InvalidUma("GTT limit must be > 0".to_owned()));
        }
        if self.gtt_limit_bytes > shared {
            return Err(EmuError::InvalidUma(format!(
                "GTT limit {} exceeds the managed shared-host allowance S = H - C - R = {}",
                self.gtt_limit_bytes, shared
            )));
        }
        Ok(())
    }
}

/// Validated emulation request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmuOptions {
    pub profile_id: String,
    /// `Some` (validated) exactly for [`MemoryKind::Uma`] profiles.
    pub uma: Option<UmaScenario>,
}

/// Facts about the real host device, gathered by the caller (BDF-associated).
/// Zero / empty / `None` mean "unknown" and fail closed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostFacts {
    pub pci_bdf: String,
    /// Exact detected gfx (never the `gfx1010` fallback).
    pub detected_gfx: String,
    pub integrated: bool,
    /// Physical CUs (not the HIP WGP count; see [`wgp_to_physical_cus`]).
    pub physical_cus: u32,
    /// Raw HIP-reported total device memory.
    pub raw_device_total_bytes: u64,
    /// Real host RAM total; required for UMA profiles.
    pub host_total_bytes: Option<u64>,
}

/// Convert the HIP multiprocessor count to physical CUs. On gfx10/gfx11/gfx12
/// (wave32 RDNA) HIP reports WGPs, i.e. half the CUs (R9700: 32 -> 64). Other
/// architectures report CUs directly.
pub fn wgp_to_physical_cus(hip_multiprocessor_count: u32, gfx: &str) -> u32 {
    let rdna = ["gfx10", "gfx11", "gfx12"]
        .iter()
        .any(|prefix| gfx.starts_with(prefix));
    if rdna {
        hip_multiprocessor_count.saturating_mul(2)
    } else {
        hip_multiprocessor_count
    }
}

/// Read one UMA override from the snapshot. Unset or empty = not set.
fn byte_override(config: &ProcessConfig, name: &'static str) -> Result<Option<u64>, EmuError> {
    let Some(raw) = config.legacy_value(name) else {
        return Ok(None);
    };
    if raw.is_empty() {
        return Ok(None);
    }
    if !raw.bytes().all(|b| b.is_ascii_digit()) {
        return Err(EmuError::InvalidOverride {
            name,
            value: raw,
            reason: "expected an unsigned decimal byte count",
        });
    }
    raw.parse::<u64>()
        .map(Some)
        .map_err(|_| EmuError::InvalidOverride {
            name,
            value: raw,
            reason: "does not fit in u64",
        })
}

/// Resolve the emulation request from the process snapshot.
///
/// Unset, empty or `off` -> `Ok(None)`. An unknown id fails. UMA overrides
/// without an active profile, or on a discrete profile, fail. UMA profiles get
/// a validated [`UmaScenario`] (defaults overridden per knob).
pub fn from_process_config(config: &ProcessConfig) -> Result<Option<EmuOptions>, EmuError> {
    const KNOBS: [&str; 4] = [
        ENV_HOST_BYTES,
        ENV_CARVEOUT_BYTES,
        ENV_HOST_RESERVE_BYTES,
        ENV_GTT_BYTES,
    ];
    let requested = config.legacy_value(ENV_PROFILE);
    let profile_id = match requested.as_deref() {
        None | Some("") | Some("off") => None,
        Some(id) => Some(id),
    };
    let Some(profile_id) = profile_id else {
        let set: Vec<&str> = KNOBS
            .iter()
            .copied()
            .filter(|name| config.legacy_value(name).is_some_and(|v| !v.is_empty()))
            .collect();
        if set.is_empty() {
            return Ok(None);
        }
        return Err(EmuError::OverrideWithoutProfile {
            keys: set.join(", "),
        });
    };
    let target = profile(profile_id)?;
    let host = byte_override(config, ENV_HOST_BYTES)?;
    let carveout = byte_override(config, ENV_CARVEOUT_BYTES)?;
    let reserve = byte_override(config, ENV_HOST_RESERVE_BYTES)?;
    let gtt = byte_override(config, ENV_GTT_BYTES)?;
    let uma = match target.memory {
        MemoryKind::Discrete { .. } => {
            for (name, value) in [
                (ENV_HOST_BYTES, host),
                (ENV_CARVEOUT_BYTES, carveout),
                (ENV_HOST_RESERVE_BYTES, reserve),
                (ENV_GTT_BYTES, gtt),
            ] {
                if value.is_some() {
                    return Err(EmuError::OverrideOnDiscrete {
                        name,
                        profile: target.id.to_owned(),
                    });
                }
            }
            None
        }
        MemoryKind::Uma => {
            let defaults = UmaScenario::default();
            let scenario = UmaScenario {
                host_total_bytes: host.unwrap_or(defaults.host_total_bytes),
                carveout_bytes: carveout.unwrap_or(defaults.carveout_bytes),
                host_reserve_bytes: reserve.unwrap_or(defaults.host_reserve_bytes),
                gtt_limit_bytes: gtt.unwrap_or(defaults.gtt_limit_bytes),
            };
            scenario.validate()?;
            Some(scenario)
        }
    };
    Ok(Some(EmuOptions {
        profile_id: target.id.to_owned(),
        uma,
    }))
}

fn kind_name(integrated: bool) -> &'static str {
    if integrated {
        "integrated"
    } else {
        "discrete"
    }
}

/// Check that the real host can stand in for the requested profile.
///
/// Requires the exact allowed detected gfx, matching integrated/discrete kind,
/// `physical_cus >= target_cus` and enough real total memory for the target
/// managed capacity (UMA: real host total >= scenario host total). Unknown
/// facts fail closed.
pub fn check_host(options: &EmuOptions, host: &HostFacts) -> Result<(), EmuError> {
    let target = profile(&options.profile_id)?;
    let uma = match (target.memory, options.uma.as_ref()) {
        (MemoryKind::Discrete { .. }, None) => None,
        (MemoryKind::Discrete { .. }, Some(_)) => {
            return Err(EmuError::InvalidOptions(format!(
                "profile `{}` is discrete but a UMA scenario was supplied",
                target.id
            )));
        }
        (MemoryKind::Uma, Some(scenario)) => {
            scenario.validate()?;
            Some(scenario)
        }
        (MemoryKind::Uma, None) => {
            return Err(EmuError::InvalidOptions(format!(
                "profile `{}` is UMA but no UMA scenario was supplied",
                target.id
            )));
        }
    };

    if host.pci_bdf.is_empty() {
        return Err(EmuError::UnknownHostFact { fact: "pci_bdf" });
    }
    if host.detected_gfx.is_empty() {
        return Err(EmuError::UnknownHostFact {
            fact: "detected_gfx",
        });
    }
    if host.physical_cus == 0 {
        return Err(EmuError::UnknownHostFact {
            fact: "physical_cus",
        });
    }
    if host.raw_device_total_bytes == 0 {
        return Err(EmuError::UnknownHostFact {
            fact: "raw_device_total_bytes",
        });
    }

    if host.detected_gfx != target.host_gfx {
        return Err(EmuError::WrongGfx {
            profile: target.id,
            required: target.host_gfx,
            detected: host.detected_gfx.clone(),
        });
    }
    let required_integrated = uma.is_some();
    if host.integrated != required_integrated {
        return Err(EmuError::WrongKind {
            profile: target.id,
            required: kind_name(required_integrated),
            detected: kind_name(host.integrated),
        });
    }
    if host.physical_cus < target.target_cus {
        return Err(EmuError::TooFewCus {
            profile: target.id,
            required: target.target_cus,
            detected: host.physical_cus,
        });
    }

    match (target.memory, uma) {
        (MemoryKind::Discrete { capacity_bytes }, _) => {
            if host.raw_device_total_bytes < capacity_bytes {
                return Err(EmuError::TooLittleMemory {
                    profile: target.id,
                    what: "device",
                    required: capacity_bytes,
                    detected: host.raw_device_total_bytes,
                });
            }
        }
        (MemoryKind::Uma, Some(scenario)) => {
            let host_total = host.host_total_bytes.ok_or(EmuError::UnknownHostFact {
                fact: "host_total_bytes",
            })?;
            if host_total == 0 {
                return Err(EmuError::UnknownHostFact {
                    fact: "host_total_bytes",
                });
            }
            if host_total < scenario.host_total_bytes {
                return Err(EmuError::TooLittleMemory {
                    profile: target.id,
                    what: "host",
                    required: scenario.host_total_bytes,
                    detected: host_total,
                });
            }
        }
        (MemoryKind::Uma, None) => unreachable!("UMA scenario presence checked above"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{developer_key_for_env, ConfigLayer, CONFIG_SCHEMA_VERSION};

    fn config(pairs: &[(&str, &str)]) -> ProcessConfig {
        let mut values = ConfigLayer::default();
        for (key, raw) in pairs {
            values.set_cli(key, raw).unwrap();
        }
        ProcessConfig {
            schema_version: CONFIG_SCHEMA_VERSION,
            values,
        }
    }

    fn discrete(id: &str) -> EmuOptions {
        EmuOptions {
            profile_id: id.to_owned(),
            uma: None,
        }
    }

    fn uma(id: &str) -> EmuOptions {
        EmuOptions {
            profile_id: id.to_owned(),
            uma: Some(UmaScenario::default()),
        }
    }

    fn r9700() -> HostFacts {
        HostFacts {
            pci_bdf: "0000:03:00.0".into(),
            detected_gfx: "gfx1201".into(),
            integrated: false,
            physical_cus: 64,
            raw_device_total_bytes: 34_208_743_424,
            host_total_bytes: None,
        }
    }

    fn xtx() -> HostFacts {
        HostFacts {
            pci_bdf: "0000:0b:00.0".into(),
            detected_gfx: "gfx1100".into(),
            integrated: false,
            physical_cus: 96,
            raw_device_total_bytes: 24 * GIB,
            host_total_bytes: None,
        }
    }

    fn halo() -> HostFacts {
        HostFacts {
            pci_bdf: "0000:c5:00.0".into(),
            detected_gfx: "gfx1151".into(),
            integrated: true,
            physical_cus: 40,
            raw_device_total_bytes: 64 * GIB,
            host_total_bytes: Some(128 * GIB),
        }
    }

    #[test]
    fn developer_key_names_are_the_registered_surface() {
        // Developer keys are an open `HIPFIRE_*` namespace; these are the exact
        // keys the snapshot stores for the emu env names.
        for (env, key) in [
            ("HIPFIRE_EMU", "developer.emu"),
            ("HIPFIRE_EMU_HOST_BYTES", "developer.emu_host_bytes"),
            ("HIPFIRE_EMU_CARVEOUT_BYTES", "developer.emu_carveout_bytes"),
            (
                "HIPFIRE_EMU_HOST_RESERVE_BYTES",
                "developer.emu_host_reserve_bytes",
            ),
            ("HIPFIRE_EMU_GTT_BYTES", "developer.emu_gtt_bytes"),
        ] {
            assert_eq!(developer_key_for_env(env).as_deref(), Some(key));
        }
        // Existing unrelated names keep distinct keys.
        assert_ne!(
            developer_key_for_env("HIPFIRE_EMULATE_GPUS"),
            developer_key_for_env("HIPFIRE_EMU")
        );
        assert_ne!(
            developer_key_for_env("HIPFIRE_EMU_BF16_H"),
            developer_key_for_env("HIPFIRE_EMU")
        );
    }

    #[test]
    fn all_ten_profiles_resolve_with_table_values() {
        // id, host_gfx, target_gfx, cus, capacity GiB (None = UMA), bw GB/s,
        // IC MiB, L2 MiB, VGPR KiB, fidelity
        let table: &[(
            &str,
            &str,
            &str,
            u32,
            Option<u64>,
            u64,
            u64,
            u64,
            u64,
            Fidelity,
        )] = &[
            (
                "rx9070xt",
                "gfx1201",
                "gfx1201",
                64,
                Some(16),
                640,
                64,
                8,
                768,
                Fidelity::CapacityOnly,
            ),
            (
                "rx9070",
                "gfx1201",
                "gfx1201",
                56,
                Some(16),
                640,
                64,
                8,
                768,
                Fidelity::CapacityEstimate,
            ),
            (
                "rx9060xt",
                "gfx1201",
                "gfx1200",
                32,
                Some(16),
                320,
                32,
                4,
                768,
                Fidelity::CrossDieEstimate,
            ),
            (
                "rx9060xt-8g",
                "gfx1201",
                "gfx1200",
                32,
                Some(8),
                320,
                32,
                4,
                768,
                Fidelity::CrossDieEstimate,
            ),
            (
                "rx7900xt",
                "gfx1100",
                "gfx1100",
                84,
                Some(20),
                800,
                80,
                6,
                768,
                Fidelity::CapacityEstimate,
            ),
            (
                "rx7800xt",
                "gfx1100",
                "gfx1101",
                60,
                Some(16),
                624,
                64,
                4,
                768,
                Fidelity::CrossDieEstimate,
            ),
            (
                "rx7700xt",
                "gfx1100",
                "gfx1101",
                54,
                Some(12),
                432,
                48,
                4,
                768,
                Fidelity::CrossDieEstimate,
            ),
            (
                "rx7600",
                "gfx1100",
                "gfx1102",
                32,
                Some(8),
                288,
                32,
                2,
                512,
                Fidelity::CrossDieEstimate,
            ),
            (
                "strix-point",
                "gfx1151",
                "gfx1150",
                16,
                None,
                128,
                0,
                2,
                512,
                Fidelity::UmaScenario,
            ),
            (
                "krackan",
                "gfx1151",
                "gfx1152",
                8,
                None,
                128,
                0,
                1,
                512,
                Fidelity::UmaScenario,
            ),
        ];
        assert_eq!(PROFILES.len(), table.len());
        for &(id, host, target, cus, cap, bw, ic, l2, vgpr, fidelity) in table {
            let p = profile(id).unwrap();
            assert_eq!(p.id, id);
            assert_eq!(p.host_gfx, host, "{id}");
            assert_eq!(p.target_gfx, target, "{id}");
            assert_eq!(p.target_cus, cus, "{id}");
            let memory = match cap {
                Some(gib) => MemoryKind::Discrete {
                    capacity_bytes: gib << 30,
                },
                None => MemoryKind::Uma,
            };
            assert_eq!(p.memory, memory, "{id}");
            assert_eq!(p.peak_bw_bytes_s, bw * 1_000_000_000, "{id}");
            assert_eq!(p.infinity_cache_bytes, ic << 20, "{id}");
            assert_eq!(p.l2_bytes, l2 << 20, "{id}");
            assert_eq!(p.vgpr_file_bytes, vgpr << 10, "{id}");
            assert_eq!(p.fidelity, fidelity, "{id}");
        }
        let mut ids: Vec<_> = PROFILES.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 10, "ids unique");
    }

    #[test]
    fn unknown_id_and_wrong_case_fail() {
        for id in [
            "rx9070XT",
            "RX9070XT",
            "rx9080",
            " rx9070xt",
            "rx9070xt ",
            "off ",
            "strix_point",
        ] {
            assert!(
                matches!(profile(id), Err(EmuError::UnknownProfile { .. })),
                "{id:?}"
            );
        }
        let err = from_process_config(&config(&[("developer.emu", "RX9070XT")])).unwrap_err();
        assert!(matches!(err, EmuError::UnknownProfile { .. }));
    }

    #[test]
    fn off_empty_unset_disable() {
        assert_eq!(from_process_config(&config(&[])), Ok(None));
        assert_eq!(
            from_process_config(&config(&[("developer.emu", "")])),
            Ok(None)
        );
        assert_eq!(
            from_process_config(&config(&[("developer.emu", "off")])),
            Ok(None)
        );
        // Uppercase OFF is not the disable spelling.
        assert!(from_process_config(&config(&[("developer.emu", "OFF")])).is_err());
    }

    #[test]
    fn override_without_profile_is_an_error() {
        for key in [
            "developer.emu_host_bytes",
            "developer.emu_carveout_bytes",
            "developer.emu_host_reserve_bytes",
            "developer.emu_gtt_bytes",
        ] {
            let unset = config(&[(key, "1024")]);
            assert!(matches!(
                from_process_config(&unset),
                Err(EmuError::OverrideWithoutProfile { .. })
            ));
            let off = config(&[("developer.emu", "off"), (key, "1024")]);
            assert!(matches!(
                from_process_config(&off),
                Err(EmuError::OverrideWithoutProfile { .. })
            ));
            // An empty knob is unset.
            assert_eq!(from_process_config(&config(&[(key, "")])), Ok(None));
        }
    }

    #[test]
    fn discrete_profile_selects_without_uma_and_rejects_uma_overrides() {
        let options = from_process_config(&config(&[("developer.emu", "rx9070xt")]))
            .unwrap()
            .unwrap();
        assert_eq!(options, discrete("rx9070xt"));
        let err = from_process_config(&config(&[
            ("developer.emu", "rx9070xt"),
            ("developer.emu_gtt_bytes", "1024"),
        ]))
        .unwrap_err();
        assert!(matches!(err, EmuError::OverrideOnDiscrete { .. }));
    }

    #[test]
    fn uma_defaults_and_overrides() {
        for id in ["strix-point", "krackan"] {
            let options = from_process_config(&config(&[("developer.emu", id)]))
                .unwrap()
                .unwrap();
            let scenario = options.uma.unwrap();
            assert_eq!(scenario, UmaScenario::default());
            assert_eq!(scenario.host_total_bytes, 32 << 30);
            assert_eq!(scenario.carveout_bytes, 1 << 30);
            assert_eq!(scenario.host_reserve_bytes, 4 << 30);
            assert_eq!(scenario.gtt_limit_bytes, 16 << 30);
            assert_eq!(scenario.shared_host_bytes().unwrap(), 27 << 30);
        }
        let options = from_process_config(&config(&[
            ("developer.emu", "krackan"),
            ("developer.emu_host_bytes", "17179869184"), // 16 GiB
            ("developer.emu_carveout_bytes", "536870912"), // 512 MiB
            ("developer.emu_host_reserve_bytes", "2147483648"), // 2 GiB
            ("developer.emu_gtt_bytes", "8589934592"),   // 8 GiB
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            options.uma.unwrap(),
            UmaScenario {
                host_total_bytes: 16 << 30,
                carveout_bytes: 512 << 20,
                host_reserve_bytes: 2 << 30,
                gtt_limit_bytes: 8 << 30,
            }
        );
    }

    #[test]
    fn override_parsing_is_strict_unsigned_decimal() {
        for bad in [
            "-1",
            "+5",
            "1.5",
            "0x10",
            "16GiB",
            "99999999999999999999999",
        ] {
            let err = from_process_config(&config(&[
                ("developer.emu", "strix-point"),
                ("developer.emu_gtt_bytes", bad),
            ]))
            .unwrap_err();
            assert!(matches!(err, EmuError::InvalidOverride { .. }), "{bad}");
        }
    }

    #[test]
    fn uma_validity_rules() {
        let ok = UmaScenario::default();
        assert_eq!(ok.validate(), Ok(()));

        // C + R < H (equal and greater both fail).
        let mut s = ok;
        s.carveout_bytes = 28 * GIB;
        s.host_reserve_bytes = 4 * GIB; // C+R == H
        assert!(matches!(s.validate(), Err(EmuError::InvalidUma(_))));
        s.carveout_bytes = 40 * GIB; // C+R > H
        assert!(matches!(s.validate(), Err(EmuError::InvalidUma(_))));

        // G > 0.
        let mut s = ok;
        s.gtt_limit_bytes = 0;
        assert!(matches!(s.validate(), Err(EmuError::InvalidUma(_))));

        // G <= S = 27 GiB: exactly S passes, S + 1 fails.
        let mut s = ok;
        s.gtt_limit_bytes = 27 * GIB;
        assert_eq!(s.validate(), Ok(()));
        s.gtt_limit_bytes = 27 * GIB + 1;
        assert!(matches!(s.validate(), Err(EmuError::InvalidUma(_))));

        // Overflow of C + R is an error, not a wrap.
        let s = UmaScenario {
            host_total_bytes: u64::MAX,
            carveout_bytes: u64::MAX,
            host_reserve_bytes: 1,
            gtt_limit_bytes: 1,
        };
        assert!(matches!(s.validate(), Err(EmuError::InvalidUma(m)) if m.contains("overflow")));
        assert!(matches!(
            s.shared_host_bytes(),
            Err(EmuError::InvalidUma(_))
        ));

        // Through the config path too.
        let err = from_process_config(&config(&[
            ("developer.emu", "strix-point"),
            ("developer.emu_gtt_bytes", "0"),
        ]))
        .unwrap_err();
        assert!(matches!(err, EmuError::InvalidUma(_)));
        let err = from_process_config(&config(&[
            ("developer.emu", "strix-point"),
            ("developer.emu_host_bytes", "8589934592"), // 8 GiB < C+R+G
        ]))
        .unwrap_err();
        assert!(matches!(err, EmuError::InvalidUma(_)));
    }

    #[test]
    fn check_host_accepts_the_three_hosts() {
        assert_eq!(check_host(&discrete("rx9070xt"), &r9700()), Ok(()));
        assert_eq!(check_host(&discrete("rx9060xt-8g"), &r9700()), Ok(()));
        assert_eq!(check_host(&discrete("rx7600"), &xtx()), Ok(()));
        assert_eq!(check_host(&discrete("rx7900xt"), &xtx()), Ok(()));
        let facts = halo();
        assert_eq!(facts.host_total_bytes, Some(128 << 30));
        assert_eq!(check_host(&uma("strix-point"), &facts), Ok(()));
        assert_eq!(check_host(&uma("krackan"), &facts), Ok(()));
    }

    #[test]
    fn check_host_rejects_wrong_gfx() {
        // R9700 cannot be a 7600; XTX cannot be a 9070 XT; fallback gfx1010 never admits.
        assert!(matches!(
            check_host(&discrete("rx7600"), &r9700()),
            Err(EmuError::WrongGfx { .. })
        ));
        assert!(matches!(
            check_host(&discrete("rx9070xt"), &xtx()),
            Err(EmuError::WrongGfx { .. })
        ));
        let mut h = r9700();
        h.detected_gfx = "gfx1010".into();
        assert!(matches!(
            check_host(&discrete("rx9070xt"), &h),
            Err(EmuError::WrongGfx { .. })
        ));
        h.detected_gfx = "gfx1201:xnack-".into();
        assert!(matches!(
            check_host(&discrete("rx9070xt"), &h),
            Err(EmuError::WrongGfx { .. })
        ));
    }

    #[test]
    fn check_host_rejects_wrong_kind() {
        let mut h = r9700();
        h.integrated = true;
        assert!(matches!(
            check_host(&discrete("rx9070xt"), &h),
            Err(EmuError::WrongKind { .. })
        ));
        let mut h = halo();
        h.integrated = false;
        assert!(matches!(
            check_host(&uma("strix-point"), &h),
            Err(EmuError::WrongKind { .. })
        ));
    }

    #[test]
    fn check_host_rejects_too_few_cus() {
        let mut h = r9700();
        h.physical_cus = 63;
        assert!(matches!(
            check_host(&discrete("rx9070xt"), &h),
            Err(EmuError::TooFewCus {
                required: 64,
                detected: 63,
                ..
            })
        ));
        h.physical_cus = 56;
        assert_eq!(check_host(&discrete("rx9070"), &h), Ok(()));
        let mut h = halo();
        h.physical_cus = 7;
        assert!(matches!(
            check_host(&uma("krackan"), &h),
            Err(EmuError::TooFewCus { .. })
        ));
    }

    #[test]
    fn check_host_rejects_too_little_memory() {
        let mut h = r9700();
        h.raw_device_total_bytes = 16 * GIB - 1;
        assert!(matches!(
            check_host(&discrete("rx9070xt"), &h),
            Err(EmuError::TooLittleMemory { what: "device", .. })
        ));
        h.raw_device_total_bytes = 16 * GIB;
        assert_eq!(check_host(&discrete("rx9070xt"), &h), Ok(()));
        let mut h = halo();
        h.host_total_bytes = Some(16 * GIB);
        assert!(matches!(
            check_host(&uma("strix-point"), &h),
            Err(EmuError::TooLittleMemory { what: "host", .. })
        ));
        // Raising the scenario host past the real host also fails.
        let mut options = uma("strix-point");
        options.uma = Some(UmaScenario {
            host_total_bytes: 256 * GIB,
            ..UmaScenario::default()
        });
        assert!(matches!(
            check_host(&options, &halo()),
            Err(EmuError::TooLittleMemory { what: "host", .. })
        ));
    }

    #[test]
    fn check_host_unknown_facts_fail_closed() {
        let mutations: [(&str, fn(&mut HostFacts)); 4] = [
            ("pci_bdf", |h| h.pci_bdf.clear()),
            ("detected_gfx", |h| h.detected_gfx.clear()),
            ("physical_cus", |h| h.physical_cus = 0),
            ("raw_device_total_bytes", |h| h.raw_device_total_bytes = 0),
        ];
        for (fact, mutate) in mutations {
            let mut h = r9700();
            mutate(&mut h);
            assert_eq!(
                check_host(&discrete("rx9070xt"), &h),
                Err(EmuError::UnknownHostFact { fact })
            );
        }
        let mut h = halo();
        h.host_total_bytes = None;
        assert_eq!(
            check_host(&uma("strix-point"), &h),
            Err(EmuError::UnknownHostFact {
                fact: "host_total_bytes"
            })
        );
        h.host_total_bytes = Some(0);
        assert_eq!(
            check_host(&uma("strix-point"), &h),
            Err(EmuError::UnknownHostFact {
                fact: "host_total_bytes"
            })
        );
    }

    #[test]
    fn check_host_rejects_inconsistent_options() {
        assert!(matches!(
            check_host(&discrete("nope"), &r9700()),
            Err(EmuError::UnknownProfile { .. })
        ));
        // UMA profile without a scenario; discrete profile with one.
        assert!(matches!(
            check_host(&discrete("strix-point"), &halo()),
            Err(EmuError::InvalidOptions(_))
        ));
        assert!(matches!(
            check_host(&uma("rx9070xt"), &r9700()),
            Err(EmuError::InvalidOptions(_))
        ));
        // An invalid scenario supplied directly is rejected too.
        let mut options = uma("strix-point");
        options.uma.as_mut().unwrap().gtt_limit_bytes = 0;
        assert!(matches!(
            check_host(&options, &halo()),
            Err(EmuError::InvalidUma(_))
        ));
    }

    #[test]
    fn wgp_to_physical_cus_doubles_rdna() {
        assert_eq!(wgp_to_physical_cus(32, "gfx1201"), 64); // R9700
        assert_eq!(wgp_to_physical_cus(48, "gfx1100"), 96); // 7900 XTX
        assert_eq!(wgp_to_physical_cus(20, "gfx1151"), 40); // Strix Halo
        assert_eq!(wgp_to_physical_cus(18, "gfx1030"), 36);
        assert_eq!(wgp_to_physical_cus(32, "gfx1201:xnack-"), 64);
        // CDNA/GCN report CUs directly.
        assert_eq!(wgp_to_physical_cus(304, "gfx942"), 304);
        assert_eq!(wgp_to_physical_cus(60, "gfx906"), 60);
        assert_eq!(wgp_to_physical_cus(0, "gfx1201"), 0);
        assert_eq!(wgp_to_physical_cus(u32::MAX, "gfx1201"), u32::MAX);
    }

    #[test]
    fn receipt_types_roundtrip() {
        let options = uma("strix-point");
        let json = serde_json::to_string(&options).unwrap();
        assert_eq!(serde_json::from_str::<EmuOptions>(&json).unwrap(), options);
        let facts = halo();
        let json = serde_json::to_string(&facts).unwrap();
        assert_eq!(serde_json::from_str::<HostFacts>(&json).unwrap(), facts);
        let json = serde_json::to_string(profile("rx9070xt").unwrap()).unwrap();
        assert!(json.contains("\"id\":\"rx9070xt\""));
        let kind: MemoryKind =
            serde_json::from_str(&serde_json::to_string(&MemoryKind::Uma).unwrap()).unwrap();
        assert_eq!(kind, MemoryKind::Uma);
    }
}
