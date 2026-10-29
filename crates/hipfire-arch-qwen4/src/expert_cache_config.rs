//! Admission and bounded resources for the opt-in exact expert cache.

use rdna_compute::railgun_check::CheckMode;

pub const EXPERT_CACHE_ENV: &str = "HIPFIRE_QWEN4_EXPERT_CACHE";
pub const EXPERT_CACHE_TAIL_PERCENT_ENV: &str = "HIPFIRE_QWEN4_EXPERT_CACHE_TAIL_PERCENT";
pub const EXPERT_CACHE_INFLIGHT_ENV: &str = "HIPFIRE_QWEN4_EXPERT_CACHE_INFLIGHT";

const IMMUTABLE_WEIGHTS_CHECK_REASON: &str =
    "is incompatible with HIPFIRE_RAILGUN_CHECK=always: physical donor-slot payloads are mutable, but the check compares a fixed Weights digest baseline";

/// Read the same resolved process snapshot and parser as CheckState, including
/// TOML developer.railgun_check. Raw environment access would miss that setting.
pub fn railgun_check_always() -> Result<bool, String> {
    let resolved = hipfire_config::developer_var("HIPFIRE_RAILGUN_CHECK").ok();
    let mode = resolved
        .as_deref()
        .map(CheckMode::parse)
        .transpose()?
        .unwrap_or(CheckMode::Off);
    Ok(mode == CheckMode::Always)
}

fn admit_railgun_check(cache_enabled: bool, always: bool) -> Result<(), String> {
    if cache_enabled && always {
        Err(format!(
            "exact expert cache {IMMUTABLE_WEIGHTS_CHECK_REASON}"
        ))
    } else {
        Ok(())
    }
}

/// Resource policy only. Routing scores and selected expert ids never change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertCacheConfig {
    pub tail_percent: u32,
    pub max_inflight: usize,
}

/// The immutable ranked load decision, used only to seed cache recency.
#[derive(Clone, Debug)]
pub struct ExpertCacheLoad {
    pub config: ExpertCacheConfig,
    pub warm_rank: Vec<u32>,
}

impl ExpertCacheConfig {
    pub fn from_env() -> Result<Option<Self>, String> {
        let cache = Self::parse(
            hipfire_config::developer_var(EXPERT_CACHE_ENV)
                .ok()
                .as_deref(),
            hipfire_config::developer_var(EXPERT_CACHE_TAIL_PERCENT_ENV)
                .ok()
                .as_deref(),
            hipfire_config::developer_var(EXPERT_CACHE_INFLIGHT_ENV)
                .ok()
                .as_deref(),
        )?;
        // Both preload planning and product loading call this before weight
        // fulfillment. Leave railgun behavior unchanged when the cache is off.
        if cache.is_some() {
            admit_railgun_check(true, railgun_check_always()?)?;
        }
        Ok(cache)
    }

    pub fn parse(
        enabled: Option<&str>,
        percent: Option<&str>,
        inflight: Option<&str>,
    ) -> Result<Option<Self>, String> {
        let enabled = match enabled {
            None => false,
            Some(value) => hipfire_config::parse_typed_bool(value)
                .ok_or_else(|| format!("{EXPERT_CACHE_ENV}: expected true/false/1/0"))?,
        };
        let number = |name: &str, value: Option<&str>, default: usize, max: usize| {
            let value = value.map_or(Ok(default), |raw| {
                raw.parse::<usize>()
                    .map_err(|_| format!("{name}: expected an integer in 1..={max}"))
            })?;
            if value == 0 || value > max {
                return Err(format!("{name}: expected an integer in 1..={max}"));
            }
            Ok(value)
        };
        // Reject invalid explicit subordinate settings even while disabled.
        let tail_percent = number(EXPERT_CACHE_TAIL_PERCENT_ENV, percent, 50, 100)? as u32;
        let max_inflight = number(EXPERT_CACHE_INFLIGHT_ENV, inflight, 64, 4096)?;
        Ok(enabled.then_some(Self {
            tail_percent,
            max_inflight,
        }))
    }

    pub fn tail_bytes(self, resident_owner_bytes: u64) -> Result<u64, String> {
        let bytes = resident_owner_bytes
            .checked_mul(u64::from(self.tail_percent))
            .ok_or("expert-cache donor size overflows")?
            / 100;
        if bytes == 0 {
            return Err("expert-cache needs a non-empty evictable expert tail".into());
        }
        Ok(bytes)
    }

    /// Device tables, retained route ids and a conservative extra queue/metadata
    /// reserve. The 32 MiB headroom is an admission margin, not a measured peak.
    pub fn device_reserve_bytes(
        self,
        layers: usize,
        experts: usize,
        topk: usize,
    ) -> Result<u64, String> {
        let tables = layers
            .checked_mul(experts)
            .and_then(|x| x.checked_mul(16))
            .ok_or("expert-cache table budget overflows")?;
        let routes = layers
            .checked_mul(topk)
            .and_then(|x| x.checked_mul(4))
            .ok_or("expert-cache route budget overflows")?;
        u64::try_from(tables)
            .ok()
            .and_then(|x| x.checked_add(routes as u64))
            .and_then(|x| x.checked_add(32 << 20))
            .ok_or_else(|| "expert-cache device reserve overflows".into())
    }

    /// Conservative margin for both additional runtimes (decode and immutable
    /// short prefill), their CPU projection/resource and binding metadata, LRU
    /// state, upload mirrors and route landing storage. Current layouts motivate
    /// this allowance; it is not measured usage or a bound for all future
    /// layouts/allocators. The payload mirror is charged by the actual tier plan.
    pub fn host_reserve_bytes(self) -> u64 {
        128 << 20
    }
}

pub struct ExpertCacheAdmission<'a> {
    pub arch: &'a str,
    pub flash_next: bool,
    pub single_device: bool,
    pub ranked_hotset: bool,
    pub uniform_mq4: bool,
    pub speculation: bool,
    pub continuous_batch: bool,
    pub cpu_experts: bool,
    pub decode_share: bool,
    pub npu: bool,
    pub slowlink: bool,
    pub expert_keep: bool,
    pub railgun_check_always: bool,
}

pub fn admit(facts: &ExpertCacheAdmission<'_>) -> Result<(), String> {
    let reason = if facts.arch != "gfx1201" {
        Some("requires exact gfx1201")
    } else if !facts.flash_next {
        Some("requires Flash-Next")
    } else if !facts.single_device {
        Some("requires one GPU without PP/TP/EP")
    } else if !facts.ranked_hotset {
        Some("requires an applied ranked expert hotset")
    } else if !facts.uniform_mq4 {
        Some("requires uniform QT44/QT53 experts")
    } else if facts.speculation {
        Some("requires speculation off")
    } else if facts.continuous_batch {
        Some("does not support continuous batching/lanes")
    } else if facts.cpu_experts {
        Some("requires PCIe expert execution without CPU/D2 splice")
    } else if facts.decode_share {
        Some("requires decode-share off")
    } else if facts.npu {
        Some("does not support the NPU trunk")
    } else if facts.slowlink {
        Some("does not support the S1 slow-link path")
    } else if facts.expert_keep {
        Some("does not support expert-keep pruning")
    } else if facts.railgun_check_always {
        Some(IMMUTABLE_WEIGHTS_CHECK_REASON)
    } else {
        None
    };
    reason.map_or(Ok(()), |why| Err(format!("exact expert cache {why}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_cache_config_is_strict_and_defaults_off() {
        assert_eq!(ExpertCacheConfig::parse(None, None, None).unwrap(), None);
        let c = ExpertCacheConfig::parse(Some("true"), None, None)
            .unwrap()
            .unwrap();
        assert_eq!((c.tail_percent, c.max_inflight), (50, 64));
        for raw in ["", "yes", "2"] {
            assert!(ExpertCacheConfig::parse(Some(raw), None, None).is_err());
        }
        for raw in ["0", "101", "-1", "NaN"] {
            assert!(ExpertCacheConfig::parse(Some("1"), Some(raw), None).is_err());
        }
        for raw in ["0", "4097", "-1"] {
            assert!(ExpertCacheConfig::parse(Some("1"), None, Some(raw)).is_err());
        }
        assert!(c.tail_bytes(u64::MAX).is_err());
        assert!(c.device_reserve_bytes(usize::MAX, 512, 10).is_err());
        assert_eq!(c.host_reserve_bytes(), 128 << 20);
    }

    fn facts() -> ExpertCacheAdmission<'static> {
        ExpertCacheAdmission {
            arch: "gfx1201",
            flash_next: true,
            single_device: true,
            ranked_hotset: true,
            uniform_mq4: true,
            speculation: false,
            continuous_batch: false,
            cpu_experts: false,
            decode_share: false,
            npu: false,
            slowlink: false,
            expert_keep: false,
            railgun_check_always: false,
        }
    }

    #[test]
    fn expert_cache_admission_refuses_each_unqualified_route() {
        assert!(admit(&facts()).is_ok());
        let mut f = facts();
        f.arch = "gfx1200";
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.single_device = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.flash_next = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.ranked_hotset = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.uniform_mq4 = false;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.speculation = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.continuous_batch = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.cpu_experts = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.decode_share = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.npu = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.slowlink = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.expert_keep = true;
        assert!(admit(&f).is_err());
        let mut f = facts();
        f.railgun_check_always = true;
        let error = admit(&f).unwrap_err();
        assert!(error.contains("physical donor-slot payloads"));
        assert!(error.contains("fixed Weights digest baseline"));
    }

    #[test]
    fn expert_cache_railgun_check_compatibility_truth_table() {
        // Cache-off preserves the checker's own behavior. Only Always has the
        // immutable physical-Weights baseline that conflicts with donor reuse.
        for (cache_enabled, setting, allowed) in [
            (false, "off", true),
            (false, "sample(1)", true),
            (false, "always", true),
            (true, "off", true),
            (true, "sample(1)", true),
            (true, "sample(64)", true),
            (true, "always", false),
            (true, " always ", false),
        ] {
            let mode = CheckMode::parse(setting).unwrap();
            assert_eq!(
                admit_railgun_check(cache_enabled, mode == CheckMode::Always).is_ok(),
                allowed,
                "cache={cache_enabled} check={setting:?}",
            );
        }
    }
}
