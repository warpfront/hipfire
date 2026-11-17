// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `HIPFIRE_EMU` in `hipfire fit`: the machine facts the daemon would see.
//!
//! A load under an emulation profile caps device capacity at the profile's and
//! commits the HIP runtime reserve at install
//! (`rdna_compute::emu::install_for_device`), so the first model allocation
//! finds `min(real free, capacity - reserve)` free of a `capacity` total.
//! Fit never opens the GPU; it rebuilds that figure from the same pure
//! function ([`rdna_compute::emu::initial_mem_info`]) and checks the host the
//! way the daemon's install does ([`hipfire_config::emu::check_host`]).

use hipfire_config::emu::{self, EmuOptions, HostFacts};
use hipfire_config::ProcessConfig;

use super::types::FitHost;

/// The emulation request of this process snapshot (`HIPFIRE_EMU`,
/// `developer.emu`), `None` when off. `Err` is the daemon's own refusal text.
pub(super) fn options(process: &ProcessConfig) -> Result<Option<EmuOptions>, String> {
    emu::from_process_config(process).map_err(|error| format!("HIPFIRE_EMU: {error}"))
}

/// HIP runtime bytes the emulated free memory already excludes: what the
/// capacity projection, which charges a cold device's runtime itself, must be
/// given back so it is not charged twice. 0 without emulation.
pub(super) fn runtime_in_free(process: &ProcessConfig) -> u64 {
    options(process)
        .ok()
        .flatten()
        .and_then(|options| rdna_compute::emu::runtime_reserve(&options).ok())
        .unwrap_or(0)
}

/// Replace GPU0's memory facts with the emulated card's and drop the other
/// GPUs (emulation is single-device; the daemon refuses a multi-device load
/// under a profile). Returns the report note on success; `Err` is a refusal
/// the daemon's install would also raise.
///
/// Fit reads no CU count, so the host check assumes the profile's CU target is
/// met (the daemon checks it); every other fact is the real device's.
pub(super) fn apply(options: &EmuOptions, machine: &mut FitHost) -> Result<String, String> {
    let profile =
        emu::profile(&options.profile_id).map_err(|error| format!("HIPFIRE_EMU: {error}"))?;
    let dropped = machine.devices.len().saturating_sub(1);
    machine.devices.truncate(1);
    let ram_total = machine.ram_total;
    let Some(device) = machine.devices.first_mut() else {
        return Ok(format!(
            "HIPFIRE_EMU={}: no GPU is visible, so there is no device to emulate on",
            profile.id
        ));
    };
    let facts = HostFacts {
        pci_bdf: "fit".to_owned(),
        detected_gfx: device.arch.clone(),
        integrated: device.unified_memory,
        physical_cus: profile.target_cus,
        raw_device_total_bytes: device.vram_total.unwrap_or(0),
        host_total_bytes: ram_total,
    };
    emu::check_host(options, &facts)
        .map_err(|error| format!("HIPFIRE_EMU host check failed: {error}"))?;
    let reserve = rdna_compute::emu::runtime_reserve(options)?;
    let capacity = match device.free {
        Some(raw_free) => {
            let (free, total) = rdna_compute::emu::initial_mem_info(options, raw_free)?;
            device.free = Some(free);
            device.vram_total = Some(total);
            total
        }
        None => device.vram_total.unwrap_or(0),
    };
    Ok(format!(
        "HIPFIRE_EMU={}: GPU0 reports the profile's {} B capacity less {} B of HIP runtime reserve, as the daemon sees it under the profile{}",
        profile.id,
        capacity,
        reserve,
        if dropped > 0 {
            format!("; the other {dropped} GPU(s) are not shown because emulation is single-device")
        } else {
            String::new()
        }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::types::FitDevice;
    use hipfire_loader::admission::HIP_RUNTIME_RESERVE_BYTES;

    const GIB: u64 = 1 << 30;

    fn gpu(index: usize, arch: &str, total: u64, free: u64) -> FitDevice {
        FitDevice {
            index,
            arch: arch.into(),
            unified_memory: false,
            vram_total: Some(total),
            free: Some(free),
            free_this_process_only: false,
            link: None,
            capacity: None,
            capacity_shown: None,
            env_selector: index.to_string(),
            config_selector: format!("GPU-{index}"),
        }
    }

    fn machine(devices: Vec<FitDevice>) -> FitHost {
        FitHost {
            ram_total: Some(128 * GIB),
            mem_available: Some(64 * GIB),
            ttm_pool: 0,
            gtt_limit: None,
            gtt_used: None,
            devices,
        }
    }

    fn rx7900xt() -> EmuOptions {
        EmuOptions {
            profile_id: "rx7900xt".into(),
            uma: None,
        }
    }

    #[test]
    fn a_7900_xt_profile_on_an_xtx_reports_20_gib_less_the_runtime_reserve() {
        // The 24 GiB XTX before any load: 23.98 GiB free of 23.98 GiB.
        let mut host = machine(vec![
            gpu(0, "gfx1100", 25_753_026_560, 25_700_000_000),
            gpu(1, "gfx1100", 25_753_026_560, 25_700_000_000),
        ]);
        let note = apply(&rx7900xt(), &mut host).unwrap();
        let reserve = HIP_RUNTIME_RESERVE_BYTES as u64;
        assert_eq!(host.devices.len(), 1, "emulation is single-device");
        assert_eq!(host.devices[0].vram_total, Some(20 * GIB));
        assert_eq!(host.devices[0].free, Some(20 * GIB - reserve));
        assert!(
            note.contains("rx7900xt") && note.contains("other 1 GPU"),
            "{note}"
        );
    }

    #[test]
    fn a_busy_card_keeps_its_own_smaller_free_figure() {
        let mut host = machine(vec![gpu(0, "gfx1100", 25_753_026_560, 6 * GIB)]);
        apply(&rx7900xt(), &mut host).unwrap();
        assert_eq!(host.devices[0].free, Some(6 * GIB));
        assert_eq!(host.devices[0].vram_total, Some(20 * GIB));
    }

    #[test]
    fn the_daemons_host_refusals_apply() {
        // A gfx1201 card cannot stand in for the gfx1100 profile.
        let mut wrong_gfx = machine(vec![gpu(0, "gfx1201", 34_000_000_000, 33_000_000_000)]);
        let error = apply(&rx7900xt(), &mut wrong_gfx).unwrap_err();
        assert!(error.contains("host check failed"), "{error}");
        // A real 20 GiB XT (21,458,059,264 B) is smaller than the profile's
        // 21,474,836,480 B.
        let mut small = machine(vec![gpu(0, "gfx1100", 21_458_059_264, 21_000_000_000)]);
        assert!(apply(&rx7900xt(), &mut small).is_err());
    }

    #[test]
    fn the_runtime_reserve_is_handed_back_to_the_projection_only_under_emulation() {
        assert_eq!(
            rdna_compute::emu::runtime_reserve(&rx7900xt()),
            Ok(HIP_RUNTIME_RESERVE_BYTES as u64)
        );
    }
}
