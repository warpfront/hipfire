// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A registry card's DFlash VRAM floor (#1151).
//!
//! Some registry tiers pair a DFlash draft that only fits next to the trunk's
//! KV growth on larger cards: the Qwen3.8-27B MQ5V2 tiers plus their draft
//! run out of memory at ~5k context on a 24 GB card. Such a card sets
//! `dflash_min_vram_gb`; the CLI forwards it as the
//! [`DFLASH_MIN_VRAM_PARAM`] load param beside the registry draft it wires, and
//! the daemon checks it against the total VRAM of the device it loads on.
//! Below the floor `speculation.dflash = auto` skips the draft; an explicit
//! `on` still loads it, with a warning.

/// Load param carrying the registry card's DFlash VRAM floor in GiB (the unit
/// of the registry's `min_vram_gb`). Sent only beside a registry-paired draft.
pub const DFLASH_MIN_VRAM_PARAM: &str = "dflash_min_vram_gb";

const GIB: f64 = (1u64 << 30) as f64;

/// What a DFlash VRAM floor decides for one device.
#[derive(Clone, Debug, PartialEq)]
pub enum DflashVramGate {
    /// No floor, the device total is unknown, or the device clears the floor:
    /// pair the draft as before.
    Pair,
    /// `auto` below the floor: skip the draft. The string names the reason.
    Skip(String),
    /// `on` below the floor: the user forced DFlash, so the draft still loads.
    /// The string is the warning to print.
    Forced(String),
}

/// Decide the DFlash pairing for a device with `device_total_bytes` of VRAM
/// (`None` when it could not be read) against a card floor of `min_vram_gb`
/// GiB. Only `auto` skips; `on` loads with a warning; any other mode (`off`
/// never carries a draft) is left alone.
pub fn dflash_vram_gate(
    dflash_mode: &str,
    min_vram_gb: Option<f64>,
    device_total_bytes: Option<u64>,
) -> DflashVramGate {
    let (Some(floor_gib), Some(total)) = (min_vram_gb, device_total_bytes) else {
        return DflashVramGate::Pair;
    };
    let total_gib = total as f64 / GIB;
    if total_gib >= floor_gib {
        return DflashVramGate::Pair;
    }
    let reason = format!(
        "the device has {total_gib:.1} GiB of VRAM, below this model's {floor_gib} GiB DFlash floor (the draft plus KV growth runs out of memory)"
    );
    match dflash_mode {
        "auto" => DflashVramGate::Skip(reason),
        "on" => DflashVramGate::Forced(format!(
            "speculation.dflash = on loads the DFlash draft anyway: {reason}; long contexts may fail"
        )),
        _ => DflashVramGate::Pair,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mem_info_vram_total` of a 7900 XTX (24 GB) and an R9700 (32 GB).
    const XTX_24GB: u64 = 25_753_026_560;
    const R9700_32GB: u64 = 34_208_743_424;

    #[test]
    fn auto_skips_below_the_floor_and_pairs_at_or_above_it() {
        // Simulated 24 GB card under a 28 GiB floor: auto skips, naming why.
        let DflashVramGate::Skip(reason) = dflash_vram_gate("auto", Some(28.0), Some(XTX_24GB))
        else {
            panic!("24 GB card must skip the draft under auto");
        };
        assert!(reason.contains("24.0 GiB"), "{reason}");
        assert!(reason.contains("28 GiB DFlash floor"), "{reason}");
        // A 32 GB card clears it.
        assert_eq!(
            dflash_vram_gate("auto", Some(28.0), Some(R9700_32GB)),
            DflashVramGate::Pair
        );
        // Exactly at the floor pairs.
        assert_eq!(
            dflash_vram_gate("auto", Some(28.0), Some(28 << 30)),
            DflashVramGate::Pair
        );
        assert!(matches!(
            dflash_vram_gate("auto", Some(28.0), Some((28 << 30) - 1)),
            DflashVramGate::Skip(_)
        ));
    }

    #[test]
    fn explicit_on_still_loads_below_the_floor_with_a_warning() {
        let DflashVramGate::Forced(warning) = dflash_vram_gate("on", Some(28.0), Some(XTX_24GB))
        else {
            panic!("on must force the draft");
        };
        assert!(warning.contains("speculation.dflash = on"), "{warning}");
        assert!(warning.contains("24.0 GiB"), "{warning}");
        assert_eq!(
            dflash_vram_gate("on", Some(28.0), Some(R9700_32GB)),
            DflashVramGate::Pair
        );
    }

    #[test]
    fn no_floor_unknown_total_or_off_leave_the_pairing_alone() {
        assert_eq!(
            dflash_vram_gate("auto", None, Some(XTX_24GB)),
            DflashVramGate::Pair
        );
        assert_eq!(
            dflash_vram_gate("auto", Some(28.0), None),
            DflashVramGate::Pair
        );
        assert_eq!(
            dflash_vram_gate("off", Some(28.0), Some(XTX_24GB)),
            DflashVramGate::Pair
        );
    }
}
