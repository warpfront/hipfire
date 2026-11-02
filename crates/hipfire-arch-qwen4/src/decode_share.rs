// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Decode-share developer knobs, mode enum and load admission.
//!
//! Decode-share places the long-prefill forward scratch on a contiguous tail
//! of cold resident experts' VRAM ("donors", mirrored in pinned host memory)
//! while decode, verify and small tiles run on a dedicated resident bank. This
//! module owns only the pure parts the loader consults: the
//! `HIPFIRE_QWEN4_DECODE_SHARE` mode, the verify knob and the admission
//! verdict. The donor layout is [`crate::expert_tier_plan`], the views and
//! mirror are [`crate::expert_tier_owners`].

use crate::gpu_forward::{Qwen4ForwardScratchMode, QWEN4_DECODE_SHARE_RESIDENT_ROWS};

/// Developer knob selecting the decode-share mode (`off` | `bind` | later
/// `bank` | `fixed` | `on`). Unset or any unrecognised value is `off`.
pub const DECODE_SHARE_ENV: &str = "HIPFIRE_QWEN4_DECODE_SHARE";

/// Developer knob: `1` verifies the donor tail against its mirror right after
/// materialization and, in `fixed` / `on`, after every donor restore (a debug
/// oracle; a mismatch fails the load or poisons the loan).
pub const DECODE_SHARE_VERIFY_ENV: &str = "HIPFIRE_QWEN4_DECODE_SHARE_VERIFY";

/// Developer knob: `1` makes the donor restore host-wait right after it is
/// enqueued (instead of at the next call that needs the donors), for A/B
/// timing of the overlapped restore.
pub const DECODE_SHARE_SYNC_ENV: &str = "HIPFIRE_QWEN4_DECODE_SHARE_SYNC";

/// Developer knob: `1` logs one line per borrow epoch (tiles, class span,
/// zeroed and restored bytes, host wait).
pub const DECODE_SHARE_LOG_ENV: &str = "HIPFIRE_QWEN4_DECODE_SHARE_LOG";

/// Developer knob (default on; exactly `0` keeps the shadows resident):
/// whether an admitted `on` / `fixed` gfx1201 load keeps the BF16 -> F16
/// weight shadows and the shared-expert INT8 shadows in the donor tail
/// (rebuilt at the first acquire of every borrow epoch) instead of resident
/// VRAM. Both shadows are read only by prefill tiles of at least 512 rows,
/// which on a decode-share load are always borrowed tiles.
pub const DECODE_SHARE_SHADOWS_ENV: &str = "HIPFIRE_QWEN4_DECODE_SHARE_SHADOWS";

/// The mode an unset (or `auto`) [`DECODE_SHARE_ENV`] selects. The product
/// default: an admitted load (exact gfx1201, single device, ranked hot set,
/// compact scratch, no lanes) shares its prefill scratch with the expert
/// donor tail and grows its resident expert set by the reclaimed bytes; every
/// other load behaves exactly as without decode-share. Flip this one constant
/// to ship the feature off by default.
pub const DECODE_SHARE_DEFAULT: DecodeShareMode = DecodeShareMode::On;

/// How a load uses decode-share.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeShareMode {
    /// No sharing: the plain placement (default).
    #[default]
    Off,
    /// Plan the donor tail, materialize the mirror and both read views; the
    /// forward does not use them yet.
    Bind,
    Bank,
    Fixed,
    On,
}

impl DecodeShareMode {
    /// Parse a knob value: ASCII case-insensitive, surrounding whitespace
    /// ignored. Unset and `auto` select [`DECODE_SHARE_DEFAULT`]; `off` and
    /// `0` and every unrecognised spelling are [`Self::Off`].
    pub fn parse(raw: Option<&str>) -> Self {
        match raw
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref()
        {
            None | Some("auto") => DECODE_SHARE_DEFAULT,
            Some("bind") => Self::Bind,
            Some("bank") => Self::Bank,
            Some("fixed") => Self::Fixed,
            Some("on") => Self::On,
            _ => Self::Off,
        }
    }

    /// Whether [`DECODE_SHARE_ENV`] is set (any value): a refusal is logged
    /// only for an explicit request, so a load the default does not admit
    /// stays silent and exactly as today.
    pub fn explicit() -> bool {
        hipfire_config::developer_var(DECODE_SHARE_ENV).is_ok()
    }

    /// The mode from the process snapshot ([`DECODE_SHARE_ENV`]).
    pub fn from_env() -> Self {
        Self::parse(
            hipfire_config::developer_var(DECODE_SHARE_ENV)
                .ok()
                .as_deref(),
        )
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Bind => "bind",
            Self::Bank => "bank",
            Self::Fixed => "fixed",
            Self::On => "on",
        }
    }

    /// Whether the loader plans a donor tail for this mode: `bind`, `fixed`
    /// and `on` share one placement plan (`bank` runs on an owned test
    /// backing and needs no donors; growing the hot budget under `on` is the
    /// loader's later step).
    pub fn plans_donors(self) -> bool {
        matches!(self, Self::Bind | Self::Fixed | Self::On)
    }

    /// Whether the forward builds its prefill banks over the real donor tail
    /// (`fixed`, `on`); `bank` uses an owned test backing.
    pub fn uses_donor_backing(self) -> bool {
        matches!(self, Self::Fixed | Self::On)
    }

    /// Whether this mode reclaims residency (`on`): it grows the expert hot
    /// budget by the reserve the dedicated scratch no longer needs.
    pub fn grows_hot_budget(self) -> bool {
        self == Self::On
    }

    /// Whether this load keeps the weight shadows in the donor tail: a mode
    /// with a real donor loan (`fixed`, `on`: [`Self::uses_donor_backing`]),
    /// exact gfx1201 (the only device whose F16 WMMA and INT8 shared-expert
    /// routes read the shadows only in prefill tiles of at least 512 rows) and
    /// [`DECODE_SHARE_SHADOWS_ENV`] not `0` (`knob`).
    pub fn borrows_shadows(self, arch: &str, knob: bool) -> bool {
        knob && arch == "gfx1201" && self.uses_donor_backing()
    }
}

/// [`DECODE_SHARE_VERIFY_ENV`] from the process snapshot: exactly `1`.
pub fn decode_share_verify_enabled() -> bool {
    hipfire_config::developer_bool(DECODE_SHARE_VERIFY_ENV, false)
}

/// [`DECODE_SHARE_SYNC_ENV`] from the process snapshot: exactly `1`.
pub fn decode_share_sync_enabled() -> bool {
    hipfire_config::developer_bool(DECODE_SHARE_SYNC_ENV, false)
}

/// [`DECODE_SHARE_LOG_ENV`] from the process snapshot: exactly `1`.
pub fn decode_share_log_enabled() -> bool {
    hipfire_config::developer_bool(DECODE_SHARE_LOG_ENV, false)
}

/// [`DECODE_SHARE_SHADOWS_ENV`] from the process snapshot: on unless exactly `0`.
pub fn decode_share_shadows_enabled() -> bool {
    hipfire_config::developer_bool(DECODE_SHARE_SHADOWS_ENV, true)
}

/// What the loader knows about a load when it decides decode-share.
#[derive(Clone, Copy, Debug)]
pub struct DecodeShareFacts<'a> {
    /// Device arch name (`gfx1201`).
    pub arch: &'a str,
    /// A `topology.pp` band load.
    pub pipeline_band: bool,
    /// An expert-parallel rank or an agreed expert-parallel layer split.
    pub expert_parallel: bool,
    /// The NPU trunk is selected (`on` / `emulate`).
    pub npu_trunk: bool,
    /// The expert placement is the ranked hot-set tier
    /// (`qwen4_plan_hotset_placement`): a sidecar rank applies and the
    /// whole-layer / whole-first placement does not.
    pub ranked_hotset_tier: bool,
    /// The load's forward scratch plan.
    pub scratch_mode: Qwen4ForwardScratchMode,
    /// An S1 slow-link prefill profile was admitted.
    pub slowlink_profile: bool,
    /// A DFlash speculator is requested.
    pub dflash: bool,
    /// The explicitly configured `memory.offload_exec`
    /// (`hipfire_config::memory::offload_exec_explicit`), `None` when unset.
    pub offload_exec: Option<hipfire_config::memory::OffloadExec>,
    /// `HIPFIRE_QWEN4_D2_DEFAULT=1` forces the D2 CPU split of an unset
    /// `memory.offload_exec`. The measured (unset) default is decided inside
    /// `attach_forward` after materialization, so the loader re-checks it there.
    pub d2_default_forced: bool,
    /// The load requests continuous batching (lanes staged after the load).
    pub continuous_batch: bool,
    /// The prefill chunk the placement is sized from.
    pub prefill_chunk: usize,
}

/// An explicit hot budget (`HIPFIRE_QWEN4_EXPERT_HOT_BUDGET_MIB`, any fixed
/// budget) does not grow, so `on` has no residency to reclaim and is `off` for
/// that load (`bind` / `fixed` are controls and still apply). `Some(line)` is
/// the log line naming it.
pub fn pinned_budget_refusal(
    mode: DecodeShareMode,
    hot_budget_fixed: bool,
) -> Option<&'static str> {
    (mode.grows_hot_budget() && hot_budget_fixed)
        .then_some("qwen4 decode-share: off (explicit hot budget: no residency to reclaim)")
}

/// Verdict change 5: whether a load may share donor VRAM. `Err` carries the
/// first refusal reason; the loader logs it and keeps the plain placement.
pub fn decode_share_admission(facts: &DecodeShareFacts<'_>) -> Result<(), &'static str> {
    use hipfire_config::memory::OffloadExec;
    if facts.arch != "gfx1201" {
        return Err("needs exact gfx1201");
    }
    if facts.pipeline_band {
        return Err("a topology.pp pipeline band load is not a single-device topology");
    }
    if facts.expert_parallel {
        return Err("expert parallelism is not a single-device topology");
    }
    if facts.npu_trunk {
        return Err("the NPU trunk is selected");
    }
    if !facts.ranked_hotset_tier {
        return Err("the expert placement is not the ranked hot-set tier");
    }
    if facts.scratch_mode != Qwen4ForwardScratchMode::FnSymGfx1201 {
        return Err("the forward scratch plan is not FnSymGfx1201");
    }
    if facts.slowlink_profile {
        return Err("an S1 slow-link prefill profile is admitted");
    }
    if facts.dflash {
        return Err("a DFlash speculator is requested");
    }
    if matches!(
        facts.offload_exec,
        Some(OffloadExec::Cpu | OffloadExec::Passback)
    ) {
        return Err("memory.offload_exec runs host experts on the CPU");
    }
    if facts.continuous_batch {
        return Err("continuous batching stages lanes (decode-share is single-stream)");
    }
    if facts.d2_default_forced {
        return Err("HIPFIRE_QWEN4_D2_DEFAULT=1 forces the D2 CPU expert split");
    }
    if facts.prefill_chunk <= QWEN4_DECODE_SHARE_RESIDENT_ROWS {
        return Err("the prefill chunk does not exceed the resident scratch bank rows");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_config::memory::OffloadExec;

    fn admitted() -> DecodeShareFacts<'static> {
        DecodeShareFacts {
            arch: "gfx1201",
            pipeline_band: false,
            expert_parallel: false,
            npu_trunk: false,
            ranked_hotset_tier: true,
            scratch_mode: Qwen4ForwardScratchMode::FnSymGfx1201,
            slowlink_profile: false,
            dflash: false,
            offload_exec: None,
            d2_default_forced: false,
            continuous_batch: false,
            prefill_chunk: 16_384,
        }
    }

    #[test]
    fn decode_share_mode_parser_defaults_to_the_product_default() {
        assert_eq!(DECODE_SHARE_DEFAULT, DecodeShareMode::On);
        assert_eq!(DecodeShareMode::parse(None), DecodeShareMode::On);
        assert_eq!(DecodeShareMode::parse(Some("auto")), DecodeShareMode::On);
        assert_eq!(DecodeShareMode::parse(Some(" AUTO ")), DecodeShareMode::On);
        assert_eq!(DecodeShareMode::parse(Some("off")), DecodeShareMode::Off);
        assert_eq!(DecodeShareMode::parse(Some("0")), DecodeShareMode::Off);
        assert_eq!(DecodeShareMode::default(), DecodeShareMode::Off);
        for raw in [
            "", "off", "OFF", "0", "1", "true", "bin", "bind1", "banked", "yes",
        ] {
            assert_eq!(
                DecodeShareMode::parse(Some(raw)),
                DecodeShareMode::Off,
                "{raw:?}"
            );
        }
        assert_eq!(DecodeShareMode::parse(Some("bind")), DecodeShareMode::Bind);
        assert_eq!(
            DecodeShareMode::parse(Some(" Bind\n")),
            DecodeShareMode::Bind
        );
        assert_eq!(DecodeShareMode::parse(Some("bank")), DecodeShareMode::Bank);
        assert_eq!(
            DecodeShareMode::parse(Some("fixed")),
            DecodeShareMode::Fixed
        );
        assert_eq!(DecodeShareMode::parse(Some("ON")), DecodeShareMode::On);
        for mode in [
            DecodeShareMode::Off,
            DecodeShareMode::Bind,
            DecodeShareMode::Bank,
            DecodeShareMode::Fixed,
            DecodeShareMode::On,
        ] {
            assert_eq!(DecodeShareMode::parse(Some(mode.name())), mode);
        }
        for mode in [
            DecodeShareMode::Bind,
            DecodeShareMode::Fixed,
            DecodeShareMode::On,
        ] {
            assert!(mode.plans_donors(), "{mode:?}");
        }
        for mode in [DecodeShareMode::Off, DecodeShareMode::Bank] {
            assert!(!mode.plans_donors(), "{mode:?}");
        }
    }

    #[test]
    fn decode_share_pinned_budget_turns_on_off_but_not_the_controls() {
        let line = pinned_budget_refusal(DecodeShareMode::On, true).unwrap();
        assert!(line.contains("explicit hot budget"));
        assert_eq!(pinned_budget_refusal(DecodeShareMode::On, false), None);
        for control in [
            DecodeShareMode::Off,
            DecodeShareMode::Bind,
            DecodeShareMode::Bank,
            DecodeShareMode::Fixed,
        ] {
            assert_eq!(pinned_budget_refusal(control, true), None, "{control:?}");
        }
    }

    #[test]
    fn decode_share_continuous_batching_is_refused() {
        let batch = DecodeShareFacts {
            continuous_batch: true,
            ..admitted()
        };
        let reason = decode_share_admission(&batch).unwrap_err();
        assert_eq!(
            reason,
            "continuous batching stages lanes (decode-share is single-stream)"
        );
        assert_eq!(decode_share_admission(&admitted()), Ok(()));
    }

    #[test]
    fn decode_share_admission_accepts_the_product_load() {
        assert_eq!(decode_share_admission(&admitted()), Ok(()));
        let unset_pcie = DecodeShareFacts {
            offload_exec: Some(OffloadExec::Pcie),
            ..admitted()
        };
        assert_eq!(decode_share_admission(&unset_pcie), Ok(()));
        let chunk = DecodeShareFacts {
            prefill_chunk: QWEN4_DECODE_SHARE_RESIDENT_ROWS + 1,
            ..admitted()
        };
        assert_eq!(decode_share_admission(&chunk), Ok(()));
    }

    #[test]
    fn decode_share_admission_names_each_refusal() {
        let cases: [(DecodeShareFacts<'static>, &str); 12] = [
            (
                DecodeShareFacts {
                    arch: "gfx1100",
                    ..admitted()
                },
                "gfx1201",
            ),
            (
                DecodeShareFacts {
                    arch: "gfx1201x",
                    ..admitted()
                },
                "gfx1201",
            ),
            (
                DecodeShareFacts {
                    pipeline_band: true,
                    ..admitted()
                },
                "pipeline band",
            ),
            (
                DecodeShareFacts {
                    expert_parallel: true,
                    ..admitted()
                },
                "expert parallelism",
            ),
            (
                DecodeShareFacts {
                    npu_trunk: true,
                    ..admitted()
                },
                "NPU trunk",
            ),
            (
                DecodeShareFacts {
                    ranked_hotset_tier: false,
                    ..admitted()
                },
                "ranked hot-set tier",
            ),
            (
                DecodeShareFacts {
                    scratch_mode: Qwen4ForwardScratchMode::Full,
                    ..admitted()
                },
                "FnSymGfx1201",
            ),
            (
                DecodeShareFacts {
                    slowlink_profile: true,
                    ..admitted()
                },
                "slow-link",
            ),
            (
                DecodeShareFacts {
                    dflash: true,
                    ..admitted()
                },
                "DFlash",
            ),
            (
                DecodeShareFacts {
                    offload_exec: Some(OffloadExec::Cpu),
                    ..admitted()
                },
                "offload_exec",
            ),
            (
                DecodeShareFacts {
                    d2_default_forced: true,
                    ..admitted()
                },
                "D2",
            ),
            (
                DecodeShareFacts {
                    prefill_chunk: QWEN4_DECODE_SHARE_RESIDENT_ROWS,
                    ..admitted()
                },
                "prefill chunk",
            ),
        ];
        for (facts, needle) in cases {
            match decode_share_admission(&facts) {
                Err(reason) => assert!(reason.contains(needle), "{reason:?} vs {needle:?}"),
                Ok(()) => panic!("expected a refusal naming {needle:?}"),
            }
        }
        // Passback also runs host experts on the CPU.
        let passback = DecodeShareFacts {
            offload_exec: Some(OffloadExec::Passback),
            ..admitted()
        };
        assert!(decode_share_admission(&passback).is_err());
        // A chunk below the resident rows is refused too.
        let small = DecodeShareFacts {
            prefill_chunk: 256,
            ..admitted()
        };
        assert!(decode_share_admission(&small).is_err());
    }
}
