//! Pure KV-mode resolution: maps a raw `HIPFIRE_KV_MODE` string + a per-site
//! `KvModePolicy` to a concrete, validated [`KvMode`]. No GPU, no env read, no
//! allocation — the caller reads the env/override and passes the raw string.
//!
//! Qwen-family sites use [`parse_qwen_k_name`] / [`resolve_kv_pair`] (shared
//! name table, hard errors, arch-aware `auto`). Non-Qwen sites keep [`resolve`].

pub use saddle_core::kv::{KvMode, VMode};

/// Per-site alias table + accepted set + default. One const per load site.
pub struct KvModePolicy {
    /// Human label for diagnostics ("qwen35-hfq", "qwen35-pp", ...).
    pub site: &'static str,
    /// Site-LOCAL alias expansion: raw string → the [`KvMode`] it denotes ON
    /// THIS SITE, or `None` for "this site does not recognize this string".
    /// Used by [`resolve`] (non-Qwen / Maple / legacy single-string path).
    /// Qwen pair resolution goes through [`parse_qwen_k_name`] instead.
    pub normalize_alias: fn(&str) -> Option<KvMode>,
    /// Modes this site has a constructor for. [`resolve`] rejects anything else
    /// and falls to `default`; [`resolve_kv_pair`] returns [`KvPairError`].
    pub accepted: &'static [KvMode],
    /// What an unrecognized / recognized-but-unaccepted string resolves to
    /// under [`resolve`].
    pub default: KvMode,
}

/// The result of [`resolve`]: the concrete mode plus an optional operator-facing
/// warning (preserves each ladder's "unrecognized, defaulting to …" diagnostic).
pub struct ResolveResult {
    /// The resolved concrete mode.
    pub mode: KvMode,
    /// `Some` when a non-empty raw input was not honored as requested. `None`
    /// when the request was honored OR when the input was unset (`""`).
    pub warning: Option<&'static str>,
}

/// Error from Qwen K/V name parsing or pair resolution. No silent fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPairError {
    pub message: String,
}

impl KvPairError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for KvPairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for KvPairError {}

use KvMode::*;

/// Shared Qwen K-name table (site-independent). Explicit K / `--kv-mode` preset
/// K only — `auto` / `""` are **not** K formats (see [`resolve_kv_pair`]).
/// Native fp8/bf16 are indivisible whole-cache presets, never K overrides.
///
/// Mapping: `asymN`|`turboN` → `FwhtN`, bare `turbo` → `Fwht3`,
/// `legacy-asymN` → `AsymN`, `fwhtN` → `FwhtN`, plus `q8`.
pub fn parse_qwen_k_name(raw: &str) -> Result<KvMode, KvPairError> {
    match raw.trim() {
        "q8" => Ok(Q8),
        "fwht2" | "asym2" | "turbo2" => Ok(Fwht2),
        "fwht3" | "asym3" | "turbo3" | "turbo" => Ok(Fwht3),
        "fwht4" | "asym4" | "turbo4" => Ok(Fwht4),
        "legacy-asym2" => Ok(Asym2),
        "legacy-asym3" => Ok(Asym3),
        "legacy-asym4" => Ok(Asym4),
        other => Err(KvPairError::new(format!(
            "unrecognized Qwen K name '{other}' (expected q8|fwht2|fwht3|fwht4|asym2|asym3|asym4|turbo|turbo2|turbo3|turbo4|legacy-asym2|legacy-asym3|legacy-asym4; fp8/bf16 require --kv-mode)"
        ))),
    }
}

/// Shared Qwen V-name table: `q8` | `lloyd2` | `lloyd3` | `lloyd4`.
pub fn parse_qwen_v_name(raw: &str) -> Result<VMode, KvPairError> {
    match raw.trim() {
        "q8" => Ok(VMode::Q8),
        "lloyd2" => Ok(VMode::Lloyd2),
        "lloyd3" => Ok(VMode::Lloyd3),
        "lloyd4" => Ok(VMode::Lloyd4),
        other => Err(KvPairError::new(format!(
            "unrecognized Qwen V name '{other}' (expected q8|lloyd2|lloyd3|lloyd4)"
        ))),
    }
}

/// Canonical Qwen log/display name for a resolved K mode.
/// `AsymN` → `legacy-asymN`; `FwhtN` → `fwhtN` (bare `asymN` aliases already
/// folded to Fwht at parse time).
pub fn qwen_k_display_name(k: KvMode) -> &'static str {
    match k {
        Q8 => "q8",
        Asym2 => "legacy-asym2",
        Asym3 => "legacy-asym3",
        Asym4 => "legacy-asym4",
        Fwht2 => "fwht2",
        Fwht3 => "fwht3",
        Fwht4 => "fwht4",
        Fp8 => "fp8",
        Bf16 => "bf16",
    }
}

/// Canonical Qwen log/display name for a resolved V mode.
pub fn qwen_v_display_name(v: VMode) -> &'static str {
    match v {
        VMode::Q8 => "q8",
        VMode::Lloyd2 => "lloyd2",
        VMode::Lloyd3 => "lloyd3",
        VMode::Lloyd4 => "lloyd4",
    }
}


/// Non-empty authored override axis (`None` / `Some("")` / whitespace = absent).
fn authored_axis(raw: Option<&str>) -> Option<&str> {
    raw.map(str::trim).filter(|s| !s.is_empty())
}

fn is_fwht_k(k: KvMode) -> bool {
    matches!(k, Fwht2 | Fwht3 | Fwht4)
}

fn is_native_k(k: KvMode) -> bool {
    matches!(k, Fp8 | Bf16)
}

/// Qwen `""`|`auto` pair before axis overrides.
///
/// - `qwen_default_q8 == true` (shipped): Q8/Q8 everywhere except exact
///   `gfx1201` when `policy.accepted` contains `Fp8` → FP8/FP8.
/// - `qwen_default_q8 == false` (kill switch): restore prior HFQ/PaRo
///   distinctions — HFQ unset/auto → FWHT3/Q8 off gfx1201; PaRo raw unset
///   → Q8/Q8, PaRo `"auto"` → FWHT3/Q8; PP/Dir stay Q8/Q8. Eligible gfx1201
///   FP8 still wins when accepted.
fn qwen_auto_default_pair(
    mode_raw: &str,
    policy: &KvModePolicy,
    arch: &str,
    qwen_default_q8: bool,
) -> (KvMode, VMode) {
    if arch == "gfx1201" && policy.accepted.contains(&Fp8) {
        return (Fp8, VMode::Q8);
    }
    if qwen_default_q8 {
        return (Q8, VMode::Q8);
    }
    // Kill switch: historical HFQ/PaRo auto ladder (named aliases unchanged).
    match policy.site {
        "qwen35-hfq" => (Fwht3, VMode::Q8),
        "qwen35-paro" => {
            if mode_raw.trim().is_empty() {
                (Q8, VMode::Q8)
            } else {
                // "auto" (and any other auto-shaped input routed here)
                (Fwht3, VMode::Q8)
            }
        }
        _ => (Q8, VMode::Q8),
    }
}

fn ensure_accepted(k: KvMode, policy: &KvModePolicy) -> Result<(), KvPairError> {
    if policy.accepted.contains(&k) {
        Ok(())
    } else {
        Err(KvPairError::new(format!(
            "K mode {k:?} is not supported at site '{}' (accepted: {:?})",
            policy.site, policy.accepted
        )))
    }
}

fn ensure_lloyd_fwht(k: KvMode, v: VMode) -> Result<(), KvPairError> {
    if matches!(v, VMode::Q8) {
        return Ok(());
    }
    if is_fwht_k(k) {
        Ok(())
    } else {
        Err(KvPairError::new(format!(
            "Lloyd V ({v:?}) requires an FWHT K mode (fwht2|fwht3|fwht4); got K={k:?}"
        )))
    }
}

/// Resolve a Qwen-family `(K, V)` pair.
///
/// - `mode_raw` `""`|`auto`: arch-aware default (see [`qwen_auto_default_pair`]).
/// - Explicit mode supplies initial K with V=Q8 (native fp8/bf16 are an
///   indivisible pair represented as `(Fp8|Bf16, VMode::Q8)`).
/// - `k_raw` / `v_raw` override only their axis when present and non-empty.
/// - Authored `--kv-mode fp8|bf16` plus any axis override → error.
/// - Auto-selected native FP8: both axes together may replace the pair; a
///   single axis refuses (never FP8-K/Q8-V hybrid).
/// - Unsupported site/name combinations → [`KvPairError`] (no silent fallback).
pub fn resolve_kv_pair(
    mode_raw: &str,
    k_raw: Option<&str>,
    v_raw: Option<&str>,
    policy: &KvModePolicy,
    arch: &str,
    qwen_default_q8: bool,
) -> Result<(KvMode, VMode), KvPairError> {
    let mode_trim = mode_raw.trim();
    let is_auto = mode_trim.is_empty() || mode_trim == "auto";
    let k_axis = authored_axis(k_raw);
    let v_axis = authored_axis(v_raw);

    let (mut k, mut v, auto_native) = if is_auto {
        let (k, v) = qwen_auto_default_pair(mode_trim, policy, arch, qwen_default_q8);
        let native = is_native_k(k);
        (k, v, native)
    } else {
        let k = match mode_trim {
            "fp8" => Fp8,
            "bf16" => Bf16,
            _ => parse_qwen_k_name(mode_trim)?,
        };
        (k, VMode::Q8, false)
    };

    let authored_native = !is_auto && is_native_k(k);
    if authored_native && (k_axis.is_some() || v_axis.is_some()) {
        return Err(KvPairError::new(format!(
            "authored --kv-mode {mode_trim} is an indivisible native K/V pair; \
             refuse axis overrides (got k={k_raw:?} v={v_raw:?}). Use a non-native \
             --kv-mode or omit --kv-k/--kv-v"
        )));
    }

    if auto_native {
        match (k_axis, v_axis) {
            (None, None) => {}
            (Some(kr), Some(vr)) => {
                // Both axes replace the auto-native pair wholesale.
                k = parse_qwen_k_name(kr)?;
                v = parse_qwen_v_name(vr)?;
                if is_native_k(k) {
                    return Err(KvPairError::new(
                        "replacing auto-native FP8/BF16 with another native tier \
                         via axis overrides is not supported; set --kv-mode explicitly"
                            .to_string(),
                    ));
                }
            }
            _ => {
                return Err(KvPairError::new(
                    "auto selected native FP8/BF16 on eligible gfx1201; a single \
                     --kv-k/--kv-v axis cannot split the pair. Pass both axes, or \
                     --kv-mode q8 (or another non-native preset)"
                        .to_string(),
                ));
            }
        }
    } else {
        if let Some(kr) = k_axis {
            k = parse_qwen_k_name(kr)?;
        }
        if let Some(vr) = v_axis {
            v = parse_qwen_v_name(vr)?;
        }
    }

    if is_native_k(k) && !matches!(v, VMode::Q8) {
        return Err(KvPairError::new(format!(
            "native {k:?} K/V pair is indivisible; V must be q8 sentinel, got {v:?}"
        )));
    }

    ensure_lloyd_fwht(k, v)?;
    ensure_accepted(k, policy)?;
    Ok((k, v))
}

/// Single-GPU Qwen automatic KV string rewrite (legacy helper).
/// Exact gfx1201 → `"fp8"`; every other arch → `"q8"`. Explicit modes untouched.
/// Prefer [`resolve_kv_pair`] for new call sites.
pub fn qwen35_auto_for_arch<'a>(raw: &'a str, arch: &str) -> &'a str {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "auto" {
        if arch == "gfx1201" {
            "fp8"
        } else {
            "q8"
        }
    } else {
        raw
    }
}

const FULL_LADDER: &[KvMode] = &[Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4, Fp8, Bf16];

/// Qwen shared alias surface for the legacy single-string [`resolve`] path.
/// Named K values use [`parse_qwen_k_name`]; `""`|`auto` map to Q8 (arch-unaware —
/// pair resolution belongs in [`resolve_kv_pair`]).
fn normalize_qwen(raw: &str) -> Option<KvMode> {
    match raw.trim() {
        "" | "auto" => Some(Q8),
        other => parse_qwen_k_name(other).ok(),
    }
}

/// Site 1 — qwen35 HFQ in-place carrier (pp=1). Default q8 (pair path is
/// arch-aware via [`resolve_kv_pair`]). Admits native fp8/bf16 and full ladder.
pub const QWEN35_HFQ_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-hfq",
    normalize_alias: normalize_qwen,
    accepted: FULL_LADDER,
    default: Q8,
};

/// Site 2 — qwen35 PaRo loader. `Asym3` absent from `accepted` (no bare
/// legacy-asym3 constructor arm). `asym3`/`turbo3` parse to `Fwht3` (accepted).
pub const QWEN35_PARO_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-paro",
    normalize_alias: normalize_qwen,
    accepted: &[Q8, Asym2, Asym4, Fwht2, Fwht3, Fwht4, Fp8, Bf16],
    default: Q8,
};

/// Site 3 — Dir/safetensors llama/qwen3 PaRo (flat). FWHT has no Flat
/// constructor: only Q8 + legacy Asym3/Asym4. Shared name table still applies
/// (`legacy-asymN` → AsymN; bare `asymN`/`turboN` → FwhtN → rejected by pair path).
pub const DIR_SAFETENSORS_POLICY: KvModePolicy = KvModePolicy {
    site: "dir-safetensors",
    normalize_alias: normalize_qwen,
    accepted: &[Q8, Asym3, Asym4],
    default: Q8,
};

/// Non-Qwen (llama HFQ) alias table — preserved pre-migration semantics.
/// `asymN` stays Givens AsymN; `turbo`/`auto` → Fwht3 (unaccepted → q8 warn).
fn normalize_full(raw: &str) -> Option<KvMode> {
    match raw {
        "q8" => Some(Q8),
        "asym2" | "turbo2" => Some(Asym2),
        "asym3" => Some(Asym3),
        "auto" | "turbo" | "turbo3" => Some(Fwht3),
        "asym4" | "turbo4" => Some(Asym4),
        "fwht2" => Some(Fwht2),
        "fwht3" => Some(Fwht3),
        "fwht4" => Some(Fwht4),
        "fp8" => Some(Fp8),
        "bf16" => Some(Bf16),
        _ => None, // "" → default (silent); unrecognized → default (+warn)
    }
}

/// Site 4 — llama HFQ carrier. Default q8. Accepts Flat constructors only.
pub const LLAMA_HFQ_POLICY: KvModePolicy = KvModePolicy {
    site: "llama-hfq",
    normalize_alias: normalize_full,
    accepted: &[Q8, Asym3, Asym4],
    default: Q8,
};

/// Site 5 — minimax / lfm2moe HFQ carriers. Hardcoded q8.
pub const HFQ_Q8_ONLY_POLICY: KvModePolicy = KvModePolicy {
    site: "hfq-q8-only",
    normalize_alias: normalize_full,
    accepted: &[Q8],
    default: Q8,
};

/// Site 6 — qwen35 HFQ pp>1 (multi-GPU). Accepts {q8, asym3, fwht3, fwht2}.
/// Shared Qwen names: `turbo3`/`asym3` → Fwht3 (accepted); `legacy-asym3` → Asym3.
pub const QWEN35_PP_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-pp",
    normalize_alias: normalize_qwen,
    accepted: &[Q8, Asym3, Fwht3, Fwht2],
    default: Q8,
};

/// Qwen dense TP / MoE EP multi-GPU constructors. Full quantized ladder like
/// HFQ but native Fp8/Bf16 excluded so eligible gfx1201 `auto` stays Q8/Q8
/// (no single-GPU native pair on sharded routes).
pub const QWEN35_TP_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-tp",
    normalize_alias: normalize_qwen,
    accepted: &[Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4],
    default: Q8,
};


/// Site 7 — maple (arch 15). Unchanged: accept {Q8, Bf16}, default bf16.
fn normalize_maple(raw: &str) -> Option<KvMode> {
    match raw {
        "bf16" | "auto" | "" => Some(Bf16),
        "q8" => Some(Q8),
        _ => None, // every rotated/quantized tier → default (+warn)
    }
}
pub const MAPLE_POLICY: KvModePolicy = KvModePolicy {
    site: "maple",
    normalize_alias: normalize_maple,
    accepted: &[Q8, Bf16],
    default: Bf16,
};

/// Pure: `&str + &'static policy → ResolveResult`. No GPU, no env read.
/// Non-Qwen / Maple / legacy single-string path. Qwen pair sites prefer
/// [`resolve_kv_pair`].
pub fn resolve(raw: &str, policy: &KvModePolicy) -> ResolveResult {
    // 1. site-LOCAL alias expansion.
    let requested: Option<KvMode> = (policy.normalize_alias)(raw);

    // 2. accept only if the site supports it; else fall to the site's
    //    (unconditional) default — EXCEPT an explicit fp8/bf16 request, which
    //    is carried forward WITH a warning so no unsupported site can silently
    //    resolve it to q8/fwht3. Downstream construction fails closed on a
    //    mode it cannot build. Any other non-empty raw that normalize rejected
    //    (None) or that normalized to an unaccepted mode warns and defaults;
    //    unset ("") defaults silently.
    let (mode, warning) = match requested {
        Some(m) if policy.accepted.contains(&m) => (m, None),
        Some(m @ (Fp8 | Bf16)) => (
            m,
            Some(
                "explicit fp8/bf16 KV requested but unsupported at this site; load will fail closed",
            ),
        ),
        _ => {
            let warning = if raw.is_empty() {
                None
            } else {
                Some("unrecognized or unsupported HIPFIRE_KV_MODE; using site default")
            };
            (policy.default, warning)
        }
    };

    ResolveResult { mode, warning }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(
        mode: &str,
        k: Option<&str>,
        v: Option<&str>,
        policy: &KvModePolicy,
        arch: &str,
        q8_default: bool,
    ) -> Result<(KvMode, VMode), KvPairError> {
        resolve_kv_pair(mode, k, v, policy, arch, q8_default)
    }

    #[test]
    fn parse_qwen_k_name_shared_table() {
        assert_eq!(parse_qwen_k_name("q8").unwrap(), Q8);
        assert_eq!(parse_qwen_k_name("fwht2").unwrap(), Fwht2);
        assert_eq!(parse_qwen_k_name("fwht3").unwrap(), Fwht3);
        assert_eq!(parse_qwen_k_name("fwht4").unwrap(), Fwht4);
        // asymN / turboN → FwhtN (not legacy Asym)
        assert_eq!(parse_qwen_k_name("asym2").unwrap(), Fwht2);
        assert_eq!(parse_qwen_k_name("asym3").unwrap(), Fwht3);
        assert_eq!(parse_qwen_k_name("asym4").unwrap(), Fwht4);
        assert_eq!(parse_qwen_k_name("turbo2").unwrap(), Fwht2);
        assert_eq!(parse_qwen_k_name("turbo3").unwrap(), Fwht3);
        assert_eq!(parse_qwen_k_name("turbo").unwrap(), Fwht3);
        assert_eq!(parse_qwen_k_name("turbo4").unwrap(), Fwht4);
        // legacy-asymN keeps Givens Asym
        assert_eq!(parse_qwen_k_name("legacy-asym2").unwrap(), Asym2);
        assert_eq!(parse_qwen_k_name("legacy-asym3").unwrap(), Asym3);
        assert_eq!(parse_qwen_k_name("legacy-asym4").unwrap(), Asym4);
        assert!(parse_qwen_k_name("fp8").is_err());
        assert!(parse_qwen_k_name("bf16").is_err());
        // auto/empty are not K names
        assert!(parse_qwen_k_name("").is_err());
        assert!(parse_qwen_k_name("auto").is_err());
        assert!(parse_qwen_k_name("garbage").is_err());
    }

    #[test]
    fn parse_qwen_v_name_table() {
        assert_eq!(parse_qwen_v_name("q8").unwrap(), VMode::Q8);
        assert_eq!(parse_qwen_v_name("lloyd2").unwrap(), VMode::Lloyd2);
        assert_eq!(parse_qwen_v_name("lloyd3").unwrap(), VMode::Lloyd3);
        assert_eq!(parse_qwen_v_name("lloyd4").unwrap(), VMode::Lloyd4);
        assert!(parse_qwen_v_name("lloyd5").is_err());
    }

    #[test]
    fn qwen_auto_q8_default_off_gfx1201_fp8_on_eligible() {
        for p in [
            &QWEN35_HFQ_POLICY,
            &QWEN35_PARO_POLICY,
            &QWEN35_PP_POLICY,
            &DIR_SAFETENSORS_POLICY,
        ] {
            for raw in ["", "auto"] {
                // shipped default: Q8 everywhere without native FP8 admit
                for arch in ["gfx1100", "gfx1200", "gfx942", "gfx1151"] {
                    let (k, v) = pair(raw, None, None, p, arch, true).unwrap();
                    assert_eq!((k, v), (Q8, VMode::Q8), "site={} arch={} raw={raw}", p.site, arch);
                }
            }
        }
        // Eligible exact gfx1201 + Fp8 in accepted → native FP8
        for p in [&QWEN35_HFQ_POLICY, &QWEN35_PARO_POLICY] {
            for raw in ["", "auto"] {
                let (k, v) = pair(raw, None, None, p, "gfx1201", true).unwrap();
                assert_eq!((k, v), (Fp8, VMode::Q8), "site={} raw={raw}", p.site);
            }
        }
        // PP/Dir lack Fp8 in accepted → stay Q8 even on gfx1201
        for p in [&QWEN35_PP_POLICY, &DIR_SAFETENSORS_POLICY] {
            let (k, v) = pair("auto", None, None, p, "gfx1201", true).unwrap();
            assert_eq!((k, v), (Q8, VMode::Q8), "site={}", p.site);
        }
    }

    #[test]
    fn kill_switch_restores_hfq_paro_historical_auto() {
        // HFQ: both "" and auto → Fwht3 off gfx1201 when kill switch off
        for raw in ["", "auto"] {
            let (k, v) = pair(raw, None, None, &QWEN35_HFQ_POLICY, "gfx1100", false).unwrap();
            assert_eq!((k, v), (Fwht3, VMode::Q8), "hfq raw={raw}");
        }
        // PaRo: raw unset Q8; auto FWHT3
        let (k, v) = pair("", None, None, &QWEN35_PARO_POLICY, "gfx1100", false).unwrap();
        assert_eq!((k, v), (Q8, VMode::Q8));
        let (k, v) = pair("auto", None, None, &QWEN35_PARO_POLICY, "gfx1100", false).unwrap();
        assert_eq!((k, v), (Fwht3, VMode::Q8));
        // PP/Dir stay Q8 under kill switch
        for p in [&QWEN35_PP_POLICY, &DIR_SAFETENSORS_POLICY] {
            let (k, v) = pair("auto", None, None, p, "gfx1100", false).unwrap();
            assert_eq!((k, v), (Q8, VMode::Q8), "site={}", p.site);
        }
        // gfx1201 eligible still FP8 under kill switch
        let (k, v) = pair("auto", None, None, &QWEN35_HFQ_POLICY, "gfx1201", false).unwrap();
        assert_eq!((k, v), (Fp8, VMode::Q8));
    }

    #[test]
    fn qwen_named_k_same_table_all_sites_then_accepted() {
        // One table → same K value; site accepted decides construct vs error.
        let cases = [
            ("q8", Q8),
            ("fwht3", Fwht3),
            ("asym3", Fwht3),
            ("turbo3", Fwht3),
            ("turbo", Fwht3),
            ("legacy-asym3", Asym3),
            ("fwht2", Fwht2),
            ("asym2", Fwht2),
            ("legacy-asym2", Asym2),
            ("fwht4", Fwht4),
            ("asym4", Fwht4),
            ("turbo4", Fwht4),
            ("legacy-asym4", Asym4),
        ];
        for p in [
            &QWEN35_HFQ_POLICY,
            &QWEN35_PARO_POLICY,
            &QWEN35_PP_POLICY,
            &DIR_SAFETENSORS_POLICY,
        ] {
            for (name, want_k) in cases {
                let r = pair(name, None, None, p, "gfx1100", true);
                if p.accepted.contains(&want_k) {
                    let (k, v) = r.unwrap_or_else(|e| panic!("{} {} ok: {e}", p.site, name));
                    assert_eq!(k, want_k, "site={} name={name}", p.site);
                    assert_eq!(v, VMode::Q8);
                } else {
                    assert!(r.is_err(), "site={} name={name} must error", p.site);
                }
            }
        }
    }

    #[test]
    fn paro_rejects_legacy_asym3_hfq_accepts() {
        assert!(pair("legacy-asym3", None, None, &QWEN35_PARO_POLICY, "gfx1100", true).is_err());
        let (k, _) = pair("legacy-asym3", None, None, &QWEN35_HFQ_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Asym3);
        // bare asym3 is Fwht3 — accepted on PaRo
        let (k, _) = pair("asym3", None, None, &QWEN35_PARO_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht3);
    }

    #[test]
    fn dir_rejects_fwht_accepts_legacy_asym() {
        assert!(pair("fwht3", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).is_err());
        assert!(pair("turbo3", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).is_err());
        assert!(pair("asym3", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).is_err());
        let (k, _) = pair("legacy-asym3", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Asym3);
        let (k, _) = pair("legacy-asym4", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Asym4);
    }

    #[test]
    fn pp_turbo3_is_fwht3_legacy_asym3_ok() {
        let (k, _) = pair("turbo3", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht3);
        let (k, _) = pair("asym3", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht3);
        let (k, _) = pair("legacy-asym3", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Asym3);
        assert!(pair("fwht4", None, None, &QWEN35_PP_POLICY, "gfx1100", true).is_err());
        let (k, _) = pair("fwht2", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht2);
        let (k, _) = pair("asym2", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht2);
        assert!(pair("turbo4", None, None, &QWEN35_PP_POLICY, "gfx1100", true).is_err());
    }

    #[test]
    fn lloyd_v_requires_fwht_k() {
        let (k, v) = pair("fwht3", None, Some("lloyd3"), &QWEN35_HFQ_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Fwht3, VMode::Lloyd3));
        let (k, v) = pair("asym3", None, Some("lloyd3"), &QWEN35_HFQ_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Fwht3, VMode::Lloyd3));
        // q8 + lloyd fails
        assert!(pair("q8", None, Some("lloyd3"), &QWEN35_HFQ_POLICY, "gfx1100", true).is_err());
        // legacy-asym + lloyd fails
        assert!(pair(
            "legacy-asym3",
            None,
            Some("lloyd3"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true
        )
        .is_err());
        // axis form: mode q8, k fwht3, v lloyd3
        let (k, v) = pair(
            "q8",
            Some("fwht3"),
            Some("lloyd3"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Fwht3, VMode::Lloyd3));
    }

    #[test]
    fn native_fp8_bf16_indivisible() {
        // authored native + any axis → error
        assert!(pair("fp8", Some("q8"), None, &QWEN35_HFQ_POLICY, "gfx1201", true).is_err());
        assert!(pair("fp8", None, Some("q8"), &QWEN35_HFQ_POLICY, "gfx1201", true).is_err());
        assert!(pair("bf16", Some("fwht3"), Some("q8"), &QWEN35_HFQ_POLICY, "gfx1100", true).is_err());
        // explicit native alone ok when accepted
        let (k, v) = pair("fp8", None, None, &QWEN35_HFQ_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Fp8, VMode::Q8));
        let (k, v) = pair("bf16", None, None, &QWEN35_PARO_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Bf16, VMode::Q8));
        // explicit native on PP (no Fp8/Bf16 accepted) → error
        assert!(pair("fp8", None, None, &QWEN35_PP_POLICY, "gfx1201", true).is_err());
        assert!(pair("bf16", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).is_err());
    }

    #[test]
    fn auto_native_single_axis_refuses_both_axes_replace() {
        // auto FP8 on eligible gfx1201
        let (k, v) = pair("auto", None, None, &QWEN35_HFQ_POLICY, "gfx1201", true).unwrap();
        assert_eq!((k, v), (Fp8, VMode::Q8));
        // single axis refuses
        assert!(pair("auto", Some("q8"), None, &QWEN35_HFQ_POLICY, "gfx1201", true).is_err());
        assert!(pair("auto", None, Some("q8"), &QWEN35_HFQ_POLICY, "gfx1201", true).is_err());
        // both axes replace
        let (k, v) = pair(
            "auto",
            Some("legacy-asym3"),
            Some("q8"),
            &QWEN35_HFQ_POLICY,
            "gfx1201",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Asym3, VMode::Q8));
    }

    #[test]
    fn axis_overrides_independent_of_mode_preset() {
        // mode supplies initial pair; k overrides only K; v only V
        let (k, v) = pair(
            "fwht4",
            Some("fwht2"),
            None,
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Fwht2, VMode::Q8));
        let (k, v) = pair(
            "fwht3",
            None,
            Some("lloyd2"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Fwht3, VMode::Lloyd2));
        // explicit kv-k q8 against non-native mode → Q8/Q8
        let (k, v) = pair("fwht3", Some("q8"), None, &QWEN35_HFQ_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Q8, VMode::Q8));
    }

    #[test]
    fn truth_table_maple_unchanged() {
        let p = &MAPLE_POLICY;
        assert_eq!(resolve("bf16", p).mode, KvMode::Bf16);
        assert_eq!(resolve("q8", p).mode, KvMode::Q8);
        assert_eq!(resolve("", p).mode, KvMode::Bf16);
        assert!(resolve("", p).warning.is_none());
        assert_eq!(resolve("auto", p).mode, KvMode::Bf16);
        assert!(resolve("auto", p).warning.is_none());
        assert!(resolve("q8", p).warning.is_none());
        for m in [
            "asym2", "asym3", "asym4", "fwht2", "fwht3", "fwht4", "turbo", "fp8",
        ] {
            let r = resolve(m, p);
            assert_eq!(r.mode, KvMode::Bf16, "{m} must fall back to the default");
            assert!(r.warning.is_some(), "{m} must warn, not silently downgrade");
        }
        let garbage = resolve("garbage", p);
        assert_eq!(garbage.mode, KvMode::Bf16);
        assert!(garbage.warning.is_some());
    }

    #[test]
    fn truth_table_llama_hfq_expanded_unchanged() {
        let p = &LLAMA_HFQ_POLICY;
        assert_eq!(resolve("", p).mode, KvMode::Q8);
        assert!(resolve("", p).warning.is_none());
        assert_eq!(resolve("q8", p).mode, KvMode::Q8);
        assert_eq!(resolve("asym3", p).mode, KvMode::Asym3);
        assert!(resolve("asym3", p).warning.is_none());
        assert_eq!(resolve("asym4", p).mode, KvMode::Asym4);
        assert_eq!(resolve("turbo4", p).mode, KvMode::Asym4);
        // auto/turbo/turbo3 → Fwht3 unaccepted → Q8 + warn (legacy non-Qwen path)
        assert_eq!(resolve("auto", p).mode, KvMode::Q8);
        assert!(resolve("auto", p).warning.is_some());
        assert_eq!(resolve("turbo", p).mode, KvMode::Q8);
        assert!(resolve("turbo", p).warning.is_some());
        assert_eq!(resolve("turbo3", p).mode, KvMode::Q8);
        assert!(resolve("turbo3", p).warning.is_some());
        assert_eq!(resolve("asym2", p).mode, KvMode::Q8);
        assert!(resolve("asym2", p).warning.is_some());
        assert_eq!(resolve("fwht3", p).mode, KvMode::Q8);
        assert!(resolve("fwht3", p).warning.is_some());
        assert_eq!(resolve("garbage", p).mode, KvMode::Q8);
        assert!(resolve("garbage", p).warning.is_some());
    }

    #[test]
    fn truth_table_hfq_q8_only_unchanged() {
        let p = &HFQ_Q8_ONLY_POLICY;
        assert_eq!(resolve("", p).mode, KvMode::Q8);
        assert_eq!(resolve("q8", p).mode, KvMode::Q8);
        assert_eq!(resolve("asym3", p).mode, KvMode::Q8);
        assert!(resolve("asym3", p).warning.is_some());
        assert_eq!(resolve("auto", p).mode, KvMode::Q8);
        assert!(resolve("auto", p).warning.is_some());
        assert_eq!(resolve("fwht4", p).mode, KvMode::Q8);
        assert_eq!(resolve("garbage", p).mode, KvMode::Q8);
    }

    #[test]
    fn qwen35_auto_for_arch_helper() {
        for raw in ["", "auto"] {
            assert_eq!(qwen35_auto_for_arch(raw, "gfx1201"), "fp8");
            for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942", "gfx906"] {
                assert_eq!(qwen35_auto_for_arch(raw, arch), "q8", "arch={arch}");
            }
        }
        for raw in ["q8", "fwht3", "fp8", "bf16", "turbo", "garbage"] {
            for arch in ["gfx1201", "gfx1100", "gfx942"] {
                assert_eq!(qwen35_auto_for_arch(raw, arch), raw, "raw={raw} arch={arch}");
            }
        }
    }

    #[test]
    fn resolve_qwen_legacy_path_uses_shared_names() {
        // Plain resolve on Qwen policies: shared names, warn-default for unaccepted.
        let p = &QWEN35_HFQ_POLICY;
        assert_eq!(resolve("asym3", p).mode, Fwht3);
        assert_eq!(resolve("legacy-asym3", p).mode, Asym3);
        assert_eq!(resolve("turbo3", p).mode, Fwht3);
        assert_eq!(resolve("auto", p).mode, Q8);
        assert_eq!(resolve("", p).mode, Q8);
        assert!(resolve("", p).warning.is_none());
        // PaRo: legacy-asym3 unaccepted → default q8 + warn
        let r = resolve("legacy-asym3", &QWEN35_PARO_POLICY);
        assert_eq!(r.mode, Q8);
        assert!(r.warning.is_some());
        assert_eq!(resolve("asym3", &QWEN35_PARO_POLICY).mode, Fwht3);
    }
}
