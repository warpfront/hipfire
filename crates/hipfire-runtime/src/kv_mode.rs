//! Pure KV-mode resolution: maps a raw `HIPFIRE_KV_MODE` string + a per-site
//! `KvModePolicy` to a concrete, validated [`KvMode`]. No GPU, no env read, no
//! allocation — the caller reads the env/override and passes the raw string.
//!
//! Qwen-family sites use [`parse_qwen_k_name`] / [`resolve_kv_pair`] (shared
//! name table, hard errors, arch-aware `auto`). Non-Qwen sites keep [`resolve`].

pub use saddle_core::kv::{KvMode, SlotKvTierPlan, VMode};

/// Per-site alias table + accepted set + default. One const per load site.
#[derive(Clone, Copy)]
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

/// Exact `(n_heads, n_kv_heads, head_dim)` shapes with a native-FP8 cache
/// constructor on the single-GPU gfx1201 Qwen35 route. An authored
/// `--kv-mode fp8` is admitted for exactly these.
const QWEN35_NATIVE_FP8_SHAPES: &[(usize, usize, usize)] = &[(24, 4, 256), (16, 2, 256)];

/// Exact shapes with a native-BF16 cache constructor (authored `bf16` only).
const QWEN35_NATIVE_BF16_SHAPES: &[(usize, usize, usize)] = &[(24, 4, 256)];

/// Subset of [`QWEN35_NATIVE_FP8_SHAPES`] qualified for implicit selection:
/// unset / `auto` picks native FP8 only here and stays Q8 elsewhere.
const QWEN35_AUTO_FP8_SHAPES: &[(usize, usize, usize)] = &[(24, 4, 256), (16, 2, 256)];

/// Route-level conditions shared by every native Qwen35 cache: exact gfx1201,
/// single GPU, no adaptive tiering, no CASK eviction.
fn qwen35_native_route(arch: &str, pp: usize, adaptive: bool, cask: bool) -> bool {
    arch == "gfx1201" && pp == 1 && !adaptive && !cask
}

fn qwen35_native_shapes(mode: KvMode) -> &'static [(usize, usize, usize)] {
    match mode {
        Fp8 => QWEN35_NATIVE_FP8_SHAPES,
        Bf16 => QWEN35_NATIVE_BF16_SHAPES,
        _ => &[],
    }
}

/// Whether an AUTHORED native `mode` (`fp8` | `bf16`) has a constructor for
/// this route and geometry. This is support, not qualification: it never
/// decides what `auto` selects (see [`qwen35_native_auto_eligible`]). Every
/// non-native mode reports `false`.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_native_supported(
    mode: KvMode,
    arch: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pp: usize,
    adaptive: bool,
    cask: bool,
) -> bool {
    qwen35_native_route(arch, pp, adaptive, cask)
        && qwen35_native_shapes(mode).contains(&(n_heads, n_kv_heads, head_dim))
}

/// Whether unset / `auto` may select native FP8 for this route and geometry.
/// Always a subset of [`qwen35_native_supported`] for [`Fp8`].
pub fn qwen35_native_auto_eligible(
    arch: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pp: usize,
    adaptive: bool,
    cask: bool,
) -> bool {
    qwen35_native_route(arch, pp, adaptive, cask)
        && QWEN35_AUTO_FP8_SHAPES.contains(&(n_heads, n_kv_heads, head_dim))
}

/// Refusal text for an authored native `mode` the route cannot construct;
/// `None` when [`qwen35_native_supported`]. The one diagnostic every
/// preflight (loader admission, HFQ and safetensors carriers) reports.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_native_refusal(
    mode: KvMode,
    arch: &str,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    pp: usize,
    adaptive: bool,
    cask: bool,
) -> Option<String> {
    if qwen35_native_supported(
        mode, arch, n_heads, n_kv_heads, head_dim, pp, adaptive, cask,
    ) {
        return None;
    }
    let shapes = qwen35_native_shapes(mode)
        .iter()
        .map(|(h, kv, d)| format!("H{h}/Hkv{kv}/D{d}"))
        .collect::<Vec<_>>()
        .join(" or ");
    Some(format!(
        "native KV {} requires exact gfx1201 {shapes}, single GPU and no adaptive/CASK \
         (got {arch}, H{n_heads}/Hkv{n_kv_heads}/D{head_dim}, pp={pp}, adaptive={adaptive}, cask={cask}); \
         use --kv-mode q8",
        qwen_k_display_name(mode),
    ))
}

/// Retain explicit native-mode diagnostics, but don't choose native for an
/// implicit auto preset the model is not qualified for
/// ([`qwen35_native_auto_eligible`]). All names still use the one parser.
pub fn qwen35_policy_for_native(
    policy: &KvModePolicy,
    mode_raw: &str,
    auto_native_eligible: bool,
) -> KvModePolicy {
    let mut selected = *policy;
    if !auto_native_eligible && matches!(mode_raw.trim(), "" | "auto") {
        selected.accepted = match policy.site {
            "qwen35-hfq" => &[Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4],
            "qwen35-paro" => &[Q8, Asym2, Asym4, Fwht2, Fwht3, Fwht4],
            _ => policy.accepted,
        };
    }
    selected
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
/// Native K+V encodings have no independently selectable V axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPair {
    Native(KvMode),
    Split(KvMode, VMode),
}

impl KvPair {
    pub fn k(self) -> KvMode {
        match self {
            Self::Native(k) | Self::Split(k, _) => k,
        }
    }

    pub fn v(self) -> Option<VMode> {
        match self {
            Self::Native(_) => None,
            Self::Split(_, v) => Some(v),
        }
    }
}

/// Shared Qwen K-name table (site-independent). Explicit K / `--kv-mode` preset
/// K only — `auto` / `""` are **not** K formats (see [`resolve_kv_pair`]).
/// Native fp8/bf16 are indivisible whole-cache presets, never K overrides.
///
/// Mapping: `asymN`|`turboN` → `FwhtN`, bare `turbo` → `Fwht3`,
/// `legacy-asymN` → `AsymN`, `fwhtN` → `FwhtN`, plus `q8`.
pub fn parse_qwen_k_name(raw: &str) -> Result<KvMode, KvPairError> {
    match raw.trim() {
        "q8" => Ok(Q8),
        // lifecycle: deprecated since 0.4.0, removal 0.5.0 — asymN/turboN aliases of fwhtN; spell fwhtN
        "fwht2" | "asym2" | "turbo2" => Ok(Fwht2),
        "fwht3" | "asym3" | "turbo3" | "turbo" => Ok(Fwht3),
        "fwht4" | "asym4" | "turbo4" => Ok(Fwht4),
        // lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV is superseded by fwht3
        "legacy-asym2" => Ok(Asym2),
        "legacy-asym3" => Ok(Asym3),
        "legacy-asym4" => Ok(Asym4),
        other => Err(KvPairError::new(format!(
            "unrecognized Qwen K name '{other}' (expected q8|fwht2|fwht3|fwht4|asym2|asym3|asym4|turbo|turbo2|turbo3|turbo4|legacy-asym2|legacy-asym3|legacy-asym4; fp8/bf16/f16 require --kv-mode)"
        ))),
    }
}

// lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV and the asymN/turboN aliases are superseded by fwht3
/// KV names deprecated since 0.4.0 (removal in 0.5.0): Givens `legacy-asymN`
/// and the `asymN` / `turbo*` aliases (Qwen: FWHT aliases; llama: Givens).
pub const DEPRECATED_KV_NAMES: &[&str] = &[
    "asym2",
    "asym3",
    "asym4",
    "legacy-asym2",
    "legacy-asym3",
    "legacy-asym4",
    "turbo",
    "turbo2",
    "turbo3",
    "turbo4",
];

/// One-line deprecation warning for an authored KV name, `None` for every
/// supported name (including unset / `auto`).
pub fn deprecated_kv_name_warning(raw: &str) -> Option<String> {
    let name = raw.trim();
    DEPRECATED_KV_NAMES.contains(&name).then(|| {
        format!(
            "warning: KV name '{name}' is deprecated and will be removed in 0.5.0 \
             (Givens asym KV and the asymN/turboN aliases are superseded by fwht3; use fwhtN or q8)"
        )
    })
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
        F16 => "f16",
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

/// Qwen `""`|`auto` pair before axis overrides.
///
/// Shipped default is Q8/Q8 except eligible exact gfx1201, which uses native
/// FP8. The kill switch restores historical HFQ/PaRo FWHT3/Q8 selection.
fn qwen_auto_default_pair(
    mode_raw: &str,
    policy: &KvModePolicy,
    arch: &str,
    qwen_default_q8: bool,
) -> KvPair {
    if arch == "gfx1201" && policy.accepted.contains(&Fp8) {
        return KvPair::Native(Fp8);
    }
    if qwen_default_q8 {
        return KvPair::Split(Q8, VMode::Q8);
    }
    match policy.site {
        "qwen35-hfq" => KvPair::Split(Fwht3, VMode::Q8),
        "qwen35-paro" if !mode_raw.trim().is_empty() => KvPair::Split(Fwht3, VMode::Q8),
        _ => KvPair::Split(Q8, VMode::Q8),
    }
}

fn ensure_accepted(k: KvMode, policy: &KvModePolicy) -> Result<(), KvPairError> {
    if policy.accepted.contains(&k) {
        Ok(())
    } else {
        Err(KvPairError::new(format!(
            "K mode {k:?} is not supported for this model route '{}' (accepted: {:?}); choose one of those with --kv-mode (memory.kv_cache) or --kv-k",
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

/// Resolve a Qwen-family K+V encoding. Native fp8/bf16 have no V axis.
/// Auto-native accepts either no overrides or both axes replacing the pair.
/// An authored native preset refuses every axis override.
pub fn resolve_kv_pair(
    mode_raw: &str,
    k_raw: Option<&str>,
    v_raw: Option<&str>,
    policy: &KvModePolicy,
    arch: &str,
    qwen_default_q8: bool,
) -> Result<KvPair, KvPairError> {
    let mode_trim = mode_raw.trim();
    let is_auto = mode_trim.is_empty() || mode_trim == "auto";
    let k_axis = authored_axis(k_raw);
    let v_axis = authored_axis(v_raw);
    let initial = if is_auto {
        qwen_auto_default_pair(mode_trim, policy, arch, qwen_default_q8)
    } else {
        match mode_trim {
            "fp8" => KvPair::Native(Fp8),
            "bf16" => KvPair::Native(Bf16),
            "f16" => KvPair::Native(F16),
            _ => KvPair::Split(parse_qwen_k_name(mode_trim)?, VMode::Q8),
        }
    };
    let pair = match (initial, k_axis, v_axis) {
        (KvPair::Native(_), Some(_), _) | (KvPair::Native(_), _, Some(_)) if !is_auto => {
            return Err(KvPairError::new(format!(
                "authored --kv-mode {mode_trim} is an indivisible native K/V pair; \
                 refuse axis overrides (got k={k_raw:?} v={v_raw:?}). Use a non-native \
                 --kv-mode or omit --kv-k/--kv-v"
            )));
        }
        (KvPair::Native(k), None, None) => KvPair::Native(k),
        (KvPair::Native(_), Some(kr), Some(vr)) => {
            KvPair::Split(parse_qwen_k_name(kr)?, parse_qwen_v_name(vr)?)
        }
        (KvPair::Native(_), _, _) => {
            return Err(KvPairError::new(
                "auto selected native FP8/BF16 on eligible gfx1201; a single \
                 --kv-k/--kv-v axis cannot split the pair. Pass both axes, or \
                 --kv-mode q8 (or another non-native preset)",
            ));
        }
        (KvPair::Split(k, v), kr, vr) => KvPair::Split(
            kr.map(parse_qwen_k_name).transpose()?.unwrap_or(k),
            vr.map(parse_qwen_v_name).transpose()?.unwrap_or(v),
        ),
    };
    if let KvPair::Split(k, v) = pair {
        ensure_lloyd_fwht(k, v)?;
    }
    ensure_accepted(pair.k(), policy)?;
    Ok(pair)
}

const FULL_LADDER: &[KvMode] = &[Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4, Fp8, Bf16];

/// Qwen shared alias surface for the legacy single-string [`resolve`] path.
/// Named K values use [`parse_qwen_k_name`]; `""`|`auto` map to Q8 (arch-unaware —
/// pair resolution belongs in [`resolve_kv_pair`]). Native presets are not K
/// names, so an `fp8`/`bf16` string at these sites warns and takes the site
/// default (q8 on the PP/TP multi-GPU sites) instead of failing the load.
fn normalize_qwen(raw: &str) -> Option<KvMode> {
    match raw.trim() {
        "" | "auto" => Some(Q8),
        other => parse_qwen_k_name(other).ok(),
    }
}

/// [`normalize_qwen`] plus the three indivisible native presets, for the
/// multi-slot site only: bf16/f16 have slot readers, and fp8 resolves too
/// (gfx1201-gated — see [`resolve_qwen35_slots`], which hard-errors an
/// explicit fp8 off gfx1201 instead of letting the engine discover it).
fn normalize_qwen_slots(raw: &str) -> Option<KvMode> {
    match raw.trim() {
        "fp8" => Some(Fp8),
        "bf16" => Some(Bf16),
        "f16" => Some(F16),
        other => normalize_qwen(other),
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
/// `asymN` stays Givens AsymN; `auto` is the site default (q8, silent);
/// `turbo`/`turbo3` → Fwht3 (unaccepted → q8 warn).
fn normalize_full(raw: &str) -> Option<KvMode> {
    match raw {
        "q8" | "auto" => Some(Q8),
        // lifecycle: deprecated since 0.4.0, removal 0.5.0 — Givens asym KV and the asymN/turboN aliases are superseded by fwht3
        "asym2" | "turbo2" => Some(Asym2),
        "asym3" => Some(Asym3),
        "turbo" | "turbo3" => Some(Fwht3),
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

/// Gemma4 full-attention tier. Names use the shared Qwen table, so
/// bare `asym3`/`turbo3` mean FWHT (no hd512 FWHT tier here → q8 + warn) and
/// the legacy Givens tier needs the explicit `legacy-asym3` opt-out.
/// `fp8`/`bf16` are carried forward so the carrier fails closed on them.
fn normalize_gemma4(raw: &str) -> Option<KvMode> {
    match raw.trim() {
        "" | "auto" => Some(Q8),
        "fp8" => Some(Fp8),
        "bf16" => Some(Bf16),
        other => parse_qwen_k_name(other).ok(),
    }
}

/// Gemma4 carrier, full-attention tier (the sliding tier is always q8).
/// `auto` follows the shared policy: fp8 only where the site admits Fp8, else
/// q8. Fp8 is not admitted: every Gemma4 attend site builds its tier plan
/// with `quant_fp8: false`, and the batched prefill uses the Q8 plan, which
/// refuses fp8. So `auto` is q8 on every arch; `legacy-asym3` restores the
/// previous Givens asym3 tier.
pub const GEMMA4_FULL_POLICY: KvModePolicy = KvModePolicy {
    site: "gemma4-full",
    normalize_alias: normalize_gemma4,
    accepted: &[Q8, Asym3],
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

/// Site 8 — the multi-slot serve engine (qwen35). The slots engine ran
/// Q8_0-only for its whole life; the full static ladder is now wired
/// end-to-end (descriptor-aware K writers + flash tile kernels, legacy and
/// paged pools), so every rotated tier the sequential path accepts is
/// accepted here too — plus the flat 2-byte native tiers bf16 and f16, which
/// DO have slot readers (descriptor-aware `kv_cache_write_*` +
/// `attention_flash_*_tile_batched` kernels). fp8 is accepted at the string
/// level on exact gfx1201 ONLY: the descriptor-aware fp8 writer/reader
/// kernels exist there (`HIPFIRE_KV_FP8_E4M3` TU family), and
/// [`resolve_qwen35_slots`] hard-errors the tier everywhere else. This
/// policy stays arch-unaware by design — [`resolve`] is pure — so Fp8 in
/// `accepted` can never leak into an auto selection: the DEFAULT stays q8 on
/// EVERY arch (deliberately NOT the sequential site's fwht3, and NOT the
/// sequential hfq site's gfx1201-fp8 auto: the slots q8 path has production
/// mileage on every fixture, and an operator who wants a rotated or native
/// tier on the slots engine says so explicitly). "auto" therefore means q8
/// HERE (mirroring the qwen35-pp site's convention, not the hfq site's).
/// Names come from the shared Qwen table: `asymN`/`turboN` → FwhtN,
/// `legacy-asymN` → AsymN.
pub const QWEN35_SLOTS_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen35-slots",
    normalize_alias: normalize_qwen_slots,
    accepted: &[Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4, Bf16, F16, Fp8],
    default: Q8,
};

/// Strict, arch-aware qwen35 multi-slot resolution. `""`/`auto` is q8 on
/// EVERY arch (the slots engine's shipped default — never fp8, mirroring
/// the qwen35-pp convention); an explicit fp8 is honored only on exact
/// gfx1201, where the descriptor-aware fp8 writer/reader kernels exist, and
/// hard-errors everywhere else naming the arch and the fix (same shape as
/// [`resolve_qwen4`]). Every other mode falls through to the pure
/// [`resolve`] unchanged.
pub fn resolve_qwen35_slots(raw: &str, arch: &str) -> Result<ResolveResult, String> {
    let resolved = resolve(raw, &QWEN35_SLOTS_POLICY);
    if resolved.mode == Fp8 && arch != "gfx1201" {
        return Err(format!(
            "qwen35 slots kv mode \"fp8\" needs gfx1201 (have {arch}); use q8 or auto"
        ));
    }
    Ok(resolved)
}

/// Qwen4 (Flash-Next) QSA K/V. `bf16` is the exact reference state (F32
/// K/V arenas, BF16-valued index keys); `fp8` stores K/V as E4M3 with one
/// f16 scale per head and token (gfx12 kernels); `q8` stores K/V as the
/// dense q8 KV row, Q8_0 blocks of 32 (gfx11 and gfx12 kernels). Both
/// quantized modes keep the index keys as BF16. Aliases stay local so an
/// explicit rotated mode cannot be silently rewritten by the generic
/// fallback resolver.
fn normalize_qwen4(raw: &str) -> Option<KvMode> {
    match raw.trim() {
        "" | "auto" | "bf16" => Some(Bf16),
        "fp8" => Some(Fp8),
        "q8" => Some(Q8),
        _ => None,
    }
}

pub const QWEN4_POLICY: KvModePolicy = KvModePolicy {
    site: "qwen4",
    normalize_alias: normalize_qwen4,
    accepted: &[Bf16, Fp8, Q8],
    default: Bf16,
};

/// Strict, arch-aware Qwen4 resolution. `""`/`auto` is fp8 on exact gfx1201,
/// q8 on exact gfx1151 (Strix Halo) and exact gfx1100 (7900 XTX; the model's
/// geometry is checked where the state is built, and an auto request the
/// geometry cannot serve falls back to the F32 state there) and bf16
/// everywhere else (gfx942 included: auto never selects fp8 there); an
/// explicit `bf16` is the exact F32 state on every arch; explicit `fp8` is
/// accepted on exact gfx1201 and exact gfx942 and refused elsewhere, explicit
/// `q8` is refused off gfx11/gfx12, and any other unsupported explicit mode
/// is refused before the generic [`resolve`] fallback can rewrite it. The
/// model's head geometry is checked where the state is built
/// (`hipfire_arch_qwen4::resolve_qsa_format`).
/// Qwen4's `Bf16` mode denotes F32 state, not BF16 KV storage. On gfx942,
/// unset/`auto`/`bf16` select that F32 state; only an explicit `fp8` selects
/// the reduced FP8 K/V rows; `q8` is not supported there.
pub fn resolve_qwen4(raw: &str, arch: &str) -> Result<ResolveResult, String> {
    let auto_fp8_arch = arch == "gfx1201";
    let auto_q8_arch = matches!(arch, "gfx1151" | "gfx1100");
    let explicit_fp8_arch = auto_fp8_arch || arch == "gfx942";
    let q8_arch = arch.starts_with("gfx11") || arch.starts_with("gfx12");
    let mode = match raw.trim() {
        "" | "auto" if auto_fp8_arch => Fp8,
        "" | "auto" if auto_q8_arch => Q8,
        "" | "auto" | "bf16" => Bf16,
        "fp8" if explicit_fp8_arch => Fp8,
        "fp8" => {
            return Err(format!(
                "qwen4 kv mode \"fp8\" needs gfx1201 or gfx942 (have {arch}); use bf16 or auto"
            ))
        }
        "q8" if q8_arch => Q8,
        "q8" => {
            return Err(format!(
                "qwen4 kv mode \"q8\" needs a gfx11 or gfx12 GPU (have {arch}); use bf16 or auto"
            ))
        }
        other => {
            return Err(format!(
                "unsupported qwen4 kv mode {other:?}; --kv-mode (memory.kv_cache) must be one of: auto, bf16, fp8, q8"
            ))
        }
    };
    Ok(ResolveResult {
        mode,
        warning: None,
    })
}

/// Pure: `&str + &'static policy → ResolveResult`. No GPU, no env read.
/// Non-Qwen / Maple / legacy single-string path. Qwen pair sites prefer
/// [`resolve_kv_pair`].
pub fn resolve(raw: &str, policy: &KvModePolicy) -> ResolveResult {
    // 1. site-LOCAL alias expansion.
    let requested: Option<KvMode> = (policy.normalize_alias)(raw);

    // 2. accept only if the site supports it; else fall to the site's
    //    (unconditional) default — EXCEPT an explicit fp8/bf16/f16 request,
    //    which is carried forward WITH a warning so no unsupported site can
    //    silently resolve it to q8/fwht3. Downstream construction fails
    //    closed on a mode it cannot build. Any other non-empty raw that
    //    normalize rejected (None) or that normalized to an unaccepted mode
    //    warns and defaults; unset ("") defaults silently.
    let (mode, warning) = match requested {
        Some(m) if policy.accepted.contains(&m) => (m, None),
        Some(m @ (Fp8 | Bf16 | F16)) => (
            m,
            Some(
                "explicit fp8/bf16/f16 KV requested but this model route does not support it; the load will be refused. Use --kv-mode q8 or auto",
            ),
        ),
        _ => {
            let warning = if raw.is_empty() {
                None
            } else {
                // Once-per-load diagnostic: leak the formatted string to
                // satisfy the `&'static str` warning type.
                let leaked: &'static str = Box::leak(
                    format!(
                        "unrecognized or unsupported --kv-mode {raw:?} for this model route '{}'; \
                         using the route default {:?}",
                        policy.site, policy.default
                    )
                    .into_boxed_str(),
                );
                Some(leaked)
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
    ) -> Result<(KvMode, Option<VMode>), KvPairError> {
        resolve_kv_pair(mode, k, v, policy, arch, q8_default).map(|pair| (pair.k(), pair.v()))
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
    fn deprecated_kv_names_warn_supported_names_do_not() {
        for raw in [
            "", "auto", "q8", "fwht2", "fwht3", "fwht4", "fp8", "bf16", "f16", "lloyd3",
        ] {
            assert!(deprecated_kv_name_warning(raw).is_none(), "{raw}");
        }
        for raw in DEPRECATED_KV_NAMES {
            let w = deprecated_kv_name_warning(&format!(" {raw} ")).unwrap();
            assert!(w.contains("removed in 0.5.0"), "{w}");
        }
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
                    assert_eq!(
                        (k, v),
                        (Q8, Some(VMode::Q8)),
                        "site={} arch={} raw={raw}",
                        p.site,
                        arch
                    );
                }
            }
        }
        // Eligible exact gfx1201 + Fp8 in accepted → native FP8
        for p in [&QWEN35_HFQ_POLICY, &QWEN35_PARO_POLICY] {
            for raw in ["", "auto"] {
                let (k, v) = pair(raw, None, None, p, "gfx1201", true).unwrap();
                assert_eq!((k, v), (Fp8, None), "site={} raw={raw}", p.site);
                assert_eq!(
                    resolve_kv_pair(raw, None, None, p, "gfx1201", true).unwrap(),
                    KvPair::Native(Fp8),
                );
            }
        }
        // PP/Dir lack Fp8 in accepted → stay Q8 even on gfx1201
        for p in [&QWEN35_PP_POLICY, &DIR_SAFETENSORS_POLICY] {
            let (k, v) = pair("auto", None, None, p, "gfx1201", true).unwrap();
            assert_eq!((k, v), (Q8, Some(VMode::Q8)), "site={}", p.site);
        }
    }

    /// `hipfire_config::effective("memory.kv_cache")` is the Settings/CLI
    /// prediction of the Qwen auto default; it must name the same mode the
    /// loader resolves.
    #[test]
    fn effective_kv_cache_auto_matches_resolve_kv_pair() {
        use hipfire_config::effective::{effective, A8Prefill, ArchCtx};
        use hipfire_config::{resolve, ConfigValue};

        let resolved = resolve([]).unwrap();
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx1030"] {
            // An eligible-gfx1201 policy (accepted set contains Fp8).
            let loaded =
                resolve_kv_pair("auto", None, None, &QWEN35_HFQ_POLICY, arch, true).unwrap();
            let ctx = ArchCtx {
                gpu_arch: Some(arch.into()),
                gpu_count: 1,
                model_arch_id: None,
                registry_tag: Some("qwen3.8:27b".into()),
                a8_prefill: A8Prefill::NotLanded,
            };
            let predicted = effective("memory.kv_cache", &resolved, &ctx);
            assert_eq!(
                predicted.value,
                ConfigValue::String(qwen_k_display_name(loaded.k()).into()),
                "arch={arch}"
            );
        }
    }

    /// Native Qwen4 (`model_arch` 16) uses `resolve_qwen4`: fp8 on exact
    /// gfx1201, bf16 elsewhere.
    #[test]
    fn effective_kv_cache_auto_matches_resolve_qwen4() {
        use hipfire_config::effective::{effective, A8Prefill, ArchCtx};
        use hipfire_config::{resolve, ConfigValue};

        let resolved = resolve([]).unwrap();
        for arch in ["gfx1100", "gfx1151", "gfx1201", "gfx1030", "gfx942"] {
            let loaded = resolve_qwen4("auto", arch).unwrap();
            let ctx = ArchCtx {
                gpu_arch: Some(arch.into()),
                gpu_count: 1,
                model_arch_id: Some(16),
                registry_tag: None,
                a8_prefill: A8Prefill::NotLanded,
            };
            let predicted = effective("memory.kv_cache", &resolved, &ctx);
            assert_eq!(
                predicted.value,
                ConfigValue::String(qwen_k_display_name(loaded.mode).into()),
                "arch={arch}"
            );
        }
    }

    #[test]
    fn mismatched_qwen_geometry_auto_is_q8_but_explicit_native_is_preserved() {
        for base in [&QWEN35_HFQ_POLICY, &QWEN35_PARO_POLICY] {
            let eligible = qwen35_native_auto_eligible("gfx1201", 24, 4, 256, 1, false, false);
            let small = qwen35_native_auto_eligible("gfx1201", 8, 2, 256, 1, false, false);
            assert!(eligible);
            assert!(!small);
            assert!(!qwen35_native_auto_eligible(
                "gfx1201", 24, 4, 256, 1, false, true
            ));
            let native = qwen35_policy_for_native(base, "auto", eligible);
            let portable = qwen35_policy_for_native(base, "auto", small);
            assert_eq!(
                pair("auto", None, None, &native, "gfx1201", true)
                    .unwrap()
                    .0,
                Fp8
            );
            assert_eq!(
                pair("auto", None, None, &portable, "gfx1201", true)
                    .unwrap()
                    .0,
                Q8
            );
            assert_eq!(
                pair("auto", Some("q8"), None, &portable, "gfx1201", true)
                    .unwrap()
                    .0,
                Q8
            );
            let explicit = qwen35_policy_for_native(base, "fp8", small);
            assert_eq!(
                pair("fp8", None, None, &explicit, "gfx1201", true)
                    .unwrap()
                    .0,
                Fp8
            );
        }
    }

    /// What every Qwen35 preflight does: auto qualification picks the policy,
    /// the authored native mode then needs route + geometry support.
    #[allow(clippy::too_many_arguments)]
    fn native_preflight(
        raw: &str,
        arch: &str,
        shape: (usize, usize, usize),
        pp: usize,
        adaptive: bool,
        cask: bool,
    ) -> Result<KvMode, String> {
        let (h, kv, d) = shape;
        let auto = qwen35_native_auto_eligible(arch, h, kv, d, pp, adaptive, cask);
        let policy = qwen35_policy_for_native(&QWEN35_HFQ_POLICY, raw, auto);
        let k = resolve_kv_pair(raw, None, None, &policy, arch, true)
            .map_err(|e| e.to_string())?
            .k();
        if matches!(k, Fp8 | Bf16) {
            if let Some(why) = qwen35_native_refusal(k, arch, h, kv, d, pp, adaptive, cask) {
                return Err(why);
            }
        }
        Ok(k)
    }

    #[test]
    fn native_support_vs_auto_qualification_truth_table() {
        const DENSE: (usize, usize, usize) = (24, 4, 256);
        const A3B: (usize, usize, usize) = (16, 2, 256);
        let ok = |raw, arch, shape, pp, ad, cask| native_preflight(raw, arch, shape, pp, ad, cask);

        // Dense 27B: unchanged. auto/unset and explicit fp8/bf16 native; q8 override.
        for raw in ["", "auto", "fp8"] {
            assert_eq!(
                ok(raw, "gfx1201", DENSE, 1, false, false),
                Ok(Fp8),
                "dense {raw}"
            );
        }
        assert_eq!(ok("bf16", "gfx1201", DENSE, 1, false, false), Ok(Bf16));
        assert_eq!(ok("q8", "gfx1201", DENSE, 1, false, false), Ok(Q8));

        // Exact A3B H16/Hkv2/D256: auto/unset and explicit fp8 are qualified;
        // q8 override honored, bf16 NOT widened.
        for raw in ["", "auto", "fp8"] {
            assert_eq!(
                ok(raw, "gfx1201", A3B, 1, false, false),
                Ok(Fp8),
                "a3b {raw}"
            );
        }
        assert_eq!(ok("q8", "gfx1201", A3B, 1, false, false), Ok(Q8));
        assert!(ok("bf16", "gfx1201", A3B, 1, false, false).is_err());
        assert!(ok("fp8", "gfx1201", (8, 2, 256), 1, false, false).is_err());

        // Wrong shapes fail closed for explicit fp8/bf16; auto stays q8.
        for shape in [
            (8, 2, 256),
            (16, 2, 128),
            (24, 2, 256),
            (16, 4, 256),
            (24, 4, 128),
            (48, 8, 128),
            (16, 2, 512),
        ] {
            for raw in ["fp8", "bf16"] {
                assert!(
                    ok(raw, "gfx1201", shape, 1, false, false).is_err(),
                    "{shape:?} {raw}"
                );
            }
            for raw in ["", "auto"] {
                assert_eq!(
                    ok(raw, "gfx1201", shape, 1, false, false),
                    Ok(Q8),
                    "{shape:?} {raw}"
                );
            }
        }

        // pp>1, adaptive, CASK and non-gfx1201 refuse explicit native on both
        // supported shapes; auto stays q8; q8 override stays valid.
        for shape in [DENSE, A3B] {
            for (arch, pp, ad, cask) in [
                ("gfx1201", 2, false, false),
                ("gfx1201", 1, true, false),
                ("gfx1201", 1, false, true),
                ("gfx1100", 1, false, false),
                ("gfx1151", 1, false, false),
                ("gfx1200", 1, false, false),
                ("gfx942", 1, false, false),
            ] {
                for raw in ["fp8", "bf16"] {
                    assert!(
                        ok(raw, arch, shape, pp, ad, cask).is_err(),
                        "{shape:?} {arch} pp{pp} ad{ad} cask{cask} {raw}"
                    );
                }
                for raw in ["", "auto", "q8"] {
                    assert_eq!(
                        ok(raw, arch, shape, pp, ad, cask),
                        Ok(Q8),
                        "{shape:?} {arch} pp{pp} ad{ad} cask{cask} {raw}"
                    );
                }
            }
        }
    }

    #[test]
    fn native_predicates_per_mode_and_auto_is_subset_of_fp8_support() {
        let sup = |m, shape: (usize, usize, usize)| {
            qwen35_native_supported(m, "gfx1201", shape.0, shape.1, shape.2, 1, false, false)
        };
        assert!(sup(Fp8, (24, 4, 256)) && sup(Fp8, (16, 2, 256)));
        assert!(sup(Bf16, (24, 4, 256)) && !sup(Bf16, (16, 2, 256)));
        for m in [Q8, Asym2, Asym3, Asym4, Fwht2, Fwht3, Fwht4, F16] {
            assert!(!sup(m, (24, 4, 256)) && !sup(m, (16, 2, 256)), "{m:?}");
        }
        // Auto qualification includes both exact supported FP8 geometries.
        assert!(qwen35_native_auto_eligible(
            "gfx1201", 24, 4, 256, 1, false, false
        ));
        assert!(qwen35_native_auto_eligible(
            "gfx1201", 16, 2, 256, 1, false, false
        ));
        for arch in ["gfx1201", "gfx1100"] {
            for (h, kv, d) in [(24, 4, 256), (16, 2, 256), (8, 2, 256)] {
                for (pp, ad, cask) in [
                    (1, false, false),
                    (2, false, false),
                    (1, true, false),
                    (1, false, true),
                ] {
                    if qwen35_native_auto_eligible(arch, h, kv, d, pp, ad, cask) {
                        assert!(qwen35_native_supported(Fp8, arch, h, kv, d, pp, ad, cask));
                    }
                }
            }
        }
        assert!(qwen35_native_refusal(Fp8, "gfx1201", 16, 2, 256, 1, false, false).is_none());
        assert!(qwen35_native_refusal(Fp8, "gfx1201", 16, 2, 256, 2, false, false).is_some());
    }

    #[test]
    fn kill_switch_restores_hfq_paro_historical_auto() {
        // HFQ: both "" and auto → Fwht3 off gfx1201 when kill switch off
        for raw in ["", "auto"] {
            let (k, v) = pair(raw, None, None, &QWEN35_HFQ_POLICY, "gfx1100", false).unwrap();
            assert_eq!((k, v), (Fwht3, Some(VMode::Q8)), "hfq raw={raw}");
        }
        // PaRo: raw unset Q8; auto FWHT3
        let (k, v) = pair("", None, None, &QWEN35_PARO_POLICY, "gfx1100", false).unwrap();
        assert_eq!((k, v), (Q8, Some(VMode::Q8)));
        let (k, v) = pair("auto", None, None, &QWEN35_PARO_POLICY, "gfx1100", false).unwrap();
        assert_eq!((k, v), (Fwht3, Some(VMode::Q8)));
        // PP/Dir stay Q8 under kill switch
        for p in [&QWEN35_PP_POLICY, &DIR_SAFETENSORS_POLICY] {
            let (k, v) = pair("auto", None, None, p, "gfx1100", false).unwrap();
            assert_eq!((k, v), (Q8, Some(VMode::Q8)), "site={}", p.site);
        }
        // gfx1201 eligible still FP8 under kill switch
        let (k, v) = pair("auto", None, None, &QWEN35_HFQ_POLICY, "gfx1201", false).unwrap();
        assert_eq!((k, v), (Fp8, None));
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
                    assert_eq!(v, Some(VMode::Q8));
                } else {
                    assert!(r.is_err(), "site={} name={name} must error", p.site);
                }
            }
        }
    }

    #[test]
    fn paro_rejects_legacy_asym3_hfq_accepts() {
        assert!(pair(
            "legacy-asym3",
            None,
            None,
            &QWEN35_PARO_POLICY,
            "gfx1100",
            true
        )
        .is_err());
        let (k, _) = pair(
            "legacy-asym3",
            None,
            None,
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!(k, Asym3);
        // bare asym3 is Fwht3 — accepted on PaRo
        let (k, _) = pair("asym3", None, None, &QWEN35_PARO_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht3);
    }

    #[test]
    fn dir_rejects_fwht_accepts_legacy_asym() {
        assert!(pair(
            "fwht3",
            None,
            None,
            &DIR_SAFETENSORS_POLICY,
            "gfx1100",
            true
        )
        .is_err());
        assert!(pair(
            "turbo3",
            None,
            None,
            &DIR_SAFETENSORS_POLICY,
            "gfx1100",
            true
        )
        .is_err());
        assert!(pair(
            "asym3",
            None,
            None,
            &DIR_SAFETENSORS_POLICY,
            "gfx1100",
            true
        )
        .is_err());
        let (k, _) = pair(
            "legacy-asym3",
            None,
            None,
            &DIR_SAFETENSORS_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!(k, Asym3);
        let (k, _) = pair(
            "legacy-asym4",
            None,
            None,
            &DIR_SAFETENSORS_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!(k, Asym4);
    }

    #[test]
    fn pp_turbo3_is_fwht3_legacy_asym3_ok() {
        let (k, _) = pair("turbo3", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht3);
        let (k, _) = pair("asym3", None, None, &QWEN35_PP_POLICY, "gfx1100", true).unwrap();
        assert_eq!(k, Fwht3);
        let (k, _) = pair(
            "legacy-asym3",
            None,
            None,
            &QWEN35_PP_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
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
        let (k, v) = pair(
            "fwht3",
            None,
            Some("lloyd3"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Fwht3, Some(VMode::Lloyd3)));
        let (k, v) = pair(
            "asym3",
            None,
            Some("lloyd3"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Fwht3, Some(VMode::Lloyd3)));
        // q8 + lloyd fails
        assert!(pair(
            "q8",
            None,
            Some("lloyd3"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true
        )
        .is_err());
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
        assert_eq!((k, v), (Fwht3, Some(VMode::Lloyd3)));
    }

    #[test]
    fn native_fp8_bf16_indivisible() {
        // authored native + any axis → error
        assert!(pair("fp8", Some("q8"), None, &QWEN35_HFQ_POLICY, "gfx1201", true).is_err());
        assert!(pair("fp8", None, Some("q8"), &QWEN35_HFQ_POLICY, "gfx1201", true).is_err());
        assert!(
            resolve_kv_pair("fp8", None, Some("q8"), &QWEN35_HFQ_POLICY, "gfx1201", true)
                .unwrap_err()
                .to_string()
                .contains("indivisible")
        );
        assert!(pair(
            "bf16",
            Some("fwht3"),
            Some("q8"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true
        )
        .is_err());
        // An FP8 K axis with a Q8 V axis is not a supported native preset.
        assert!(pair(
            "auto",
            Some("fp8"),
            Some("q8"),
            &QWEN35_HFQ_POLICY,
            "gfx1201",
            true
        )
        .is_err());
        // explicit native alone ok when accepted
        let (k, v) = pair("fp8", None, None, &QWEN35_HFQ_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Fp8, None));
        let (k, v) = pair("bf16", None, None, &QWEN35_PARO_POLICY, "gfx1100", true).unwrap();
        assert_eq!((k, v), (Bf16, None));
        // explicit native on PP (no Fp8/Bf16 accepted) → error
        assert!(pair("fp8", None, None, &QWEN35_PP_POLICY, "gfx1201", true).is_err());
        assert!(pair("bf16", None, None, &DIR_SAFETENSORS_POLICY, "gfx1100", true).is_err());
    }

    #[test]
    fn auto_native_single_axis_refuses_both_axes_replace() {
        // auto FP8 on eligible gfx1201
        let (k, v) = pair("auto", None, None, &QWEN35_HFQ_POLICY, "gfx1201", true).unwrap();
        assert_eq!((k, v), (Fp8, None));
        // single axis refuses
        assert!(pair(
            "auto",
            Some("q8"),
            None,
            &QWEN35_HFQ_POLICY,
            "gfx1201",
            true
        )
        .is_err());
        assert!(pair(
            "auto",
            None,
            Some("q8"),
            &QWEN35_HFQ_POLICY,
            "gfx1201",
            true
        )
        .is_err());
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
        assert_eq!((k, v), (Asym3, Some(VMode::Q8)));
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
        assert_eq!((k, v), (Fwht2, Some(VMode::Q8)));
        let (k, v) = pair(
            "fwht3",
            None,
            Some("lloyd2"),
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Fwht3, Some(VMode::Lloyd2)));
        // explicit kv-k q8 against non-native mode → Q8/Q8
        let (k, v) = pair(
            "fwht3",
            Some("q8"),
            None,
            &QWEN35_HFQ_POLICY,
            "gfx1100",
            true,
        )
        .unwrap();
        assert_eq!((k, v), (Q8, Some(VMode::Q8)));
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
    fn truth_table_qwen35_slots_default_is_q8() {
        let p = &QWEN35_SLOTS_POLICY;
        // Unset and "auto" both mean q8, SILENTLY — q8 is the slots
        // engine's shipped default and its longest-validated path.
        assert_eq!(resolve("", p).mode, KvMode::Q8);
        assert!(resolve("", p).warning.is_none());
        assert_eq!(resolve("auto", p).mode, KvMode::Q8);
        assert!(resolve("auto", p).warning.is_none());
        assert_eq!(resolve("q8", p).mode, KvMode::Q8);
        assert!(resolve("q8", p).warning.is_none());
        // Every rotated tier is HONORED (the ladder is fully wired) and
        // must not warn — an explicit tier is an intentional choice.
        // Names use the shared Qwen table: asymN/turboN → FwhtN,
        // legacy-asymN → AsymN.
        for (raw, mode) in [
            ("asym2", KvMode::Fwht2),
            ("asym3", KvMode::Fwht3),
            ("turbo", KvMode::Fwht3),
            ("asym4", KvMode::Fwht4),
            ("fwht2", KvMode::Fwht2),
            ("fwht3", KvMode::Fwht3),
            ("fwht4", KvMode::Fwht4),
            ("legacy-asym2", KvMode::Asym2),
            ("legacy-asym3", KvMode::Asym3),
            ("legacy-asym4", KvMode::Asym4),
        ] {
            let r = resolve(raw, p);
            assert_eq!(r.mode, mode, "{raw} must be honored on the slots site");
            assert!(r.warning.is_none(), "{raw} must not warn");
        }
        // The flat 16-bit native tiers ARE allocatable on the slots site
        // (descriptor-aware write/attend kernels exist for both layouts) —
        // honored cleanly, no warning.
        for (raw, mode) in [("bf16", KvMode::Bf16), ("f16", KvMode::F16)] {
            let r = resolve(raw, p);
            assert_eq!(r.mode, mode, "{raw} must be honored on the slots site");
            assert!(r.warning.is_none(), "{raw} must not warn");
        }
        // fp8 resolves cleanly at the STRING level (the descriptor-aware
        // fp8 writer/reader kernels exist on gfx1201); the arch gate lives
        // in resolve_qwen35_slots below, and auto stays q8 on every arch
        // because the slots default is q8 — Fp8 in `accepted` can never
        // become an auto selection.
        let r = resolve("fp8", p);
        assert_eq!(r.mode, KvMode::Fp8);
        assert!(
            r.warning.is_none(),
            "fp8 must resolve cleanly on the slots site"
        );
        // Arch-parameterized gate: explicit fp8 on gfx1201 is honored, and
        // every other arch hard-errors naming the arch and the fix.
        let ok = resolve_qwen35_slots("fp8", "gfx1201").unwrap();
        assert_eq!(ok.mode, KvMode::Fp8);
        assert!(ok.warning.is_none());
        for arch in ["gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx942"] {
            let Err(error) = resolve_qwen35_slots("fp8", arch) else {
                panic!("fp8 off gfx1201 must be refused on {arch}");
            };
            assert!(error.contains("fp8") && error.contains(arch), "{error}");
            assert!(error.contains("gfx1201"), "{error}");
        }
        // auto/unset never select fp8, even on gfx1201 (slots default q8).
        for raw in ["", "auto"] {
            for arch in ["gfx1100", "gfx1201"] {
                let r = resolve_qwen35_slots(raw, arch).unwrap();
                assert_eq!(r.mode, KvMode::Q8, "{raw:?} on {arch} must stay q8");
                assert!(r.warning.is_none());
            }
        }
        // The pair path now honors fp8 too (it is in `accepted`): an
        // indivisible native pair resolves on gfx1201. The slots site itself
        // only consumes the single-string [`resolve`], so this is a
        // policy-level statement, not a production route — and the arch gate
        // in resolve_qwen35_slots applies identically either way.
        assert_eq!(
            resolve_kv_pair("fp8", None, None, p, "gfx1201", true).unwrap(),
            KvPair::Native(Fp8),
            "fp8 must resolve as an indivisible native pair once accepted"
        );
        let garbage = resolve("garbage", p);
        assert_eq!(garbage.mode, KvMode::Q8);
        assert!(garbage.warning.is_some());
    }

    #[test]
    fn native_presets_fall_back_to_default_off_slots_site() {
        // Only the slots site normalizes the native presets. Everywhere else
        // on the single-string path an fp8/bf16/f16 string is not a Qwen K
        // name: it warns and takes the site default (q8 on the PP/TP
        // multi-GPU sites) instead of failing the load.
        for p in [
            &QWEN35_HFQ_POLICY,
            &QWEN35_PARO_POLICY,
            &QWEN35_PP_POLICY,
            &QWEN35_TP_POLICY,
            &DIR_SAFETENSORS_POLICY,
        ] {
            for raw in ["fp8", "bf16", "f16"] {
                let r = resolve(raw, p);
                assert_eq!(r.mode, p.default, "site {} {raw}", p.site);
                assert!(r.warning.is_some(), "site {} must WARN on {raw}", p.site);
            }
        }
        // The slots site honors f16 and bf16 and fp8 (the latter
        // gfx1201-gated by resolve_qwen35_slots, not by the string resolver).
        assert_eq!(resolve("f16", &QWEN35_SLOTS_POLICY).mode, KvMode::F16);
        assert_eq!(resolve("bf16", &QWEN35_SLOTS_POLICY).mode, KvMode::Bf16);
        assert_eq!(resolve("fp8", &QWEN35_SLOTS_POLICY).mode, KvMode::Fp8);
    }

    #[test]
    fn qwen4_gfx942_auto_stays_f32_explicit_fp8_accepted_q8_refused() {
        // Unset/auto/bf16 never select fp8 on gfx942: they resolve to F32 state.
        for raw in ["", "auto", " auto ", "bf16"] {
            let resolved = resolve_qwen4(raw, "gfx942").expect("gfx942 F32 state mode");
            assert_eq!(resolved.mode, KvMode::Bf16, "{raw:?} on gfx942");
            assert!(resolved.warning.is_none());
        }
        // Only an explicit fp8 selects reduced K/V on gfx942.
        for raw in ["fp8", " fp8 "] {
            let resolved = resolve_qwen4(raw, "gfx942").expect("explicit gfx942 fp8");
            assert_eq!(resolved.mode, KvMode::Fp8, "{raw:?} on gfx942");
            assert!(resolved.warning.is_none());
        }
        let Err(error) = resolve_qwen4("q8", "gfx942") else {
            panic!("q8 reduced KV must be refused on gfx942");
        };
        assert!(
            error.contains("q8") && error.contains("gfx942") && error.contains("use bf16 or auto"),
            "{error}"
        );
        // Only exact gfx942 and gfx1201 accept explicit fp8; sibling CDNA
        // arches and gfx1200 do not.
        for arch in ["gfx940", "gfx941", "gfx90a", "gfx9420", "gfx1200"] {
            let Err(error) = resolve_qwen4("fp8", arch) else {
                panic!("fp8 must be refused on {arch}");
            };
            assert!(
                error.contains("fp8") && error.contains(arch) && error.contains("gfx942"),
                "{error}"
            );
        }
    }

    #[test]
    fn qwen4_auto_fp8_gfx1201_q8_gfx1151_gfx1100_bf16_elsewhere_rejects_before_fallback() {
        let p = &QWEN4_POLICY;
        // (raw, arch) -> mode. Only exact gfx1201 selects fp8 (not gfx1200);
        // only exact gfx1151 (the Halo) and exact gfx1100 (the XTX) select q8
        // under auto; every other arch keeps bf16 (the F32 state). Explicit
        // bf16 is the F32 state on every arch; explicit q8 is served on every
        // gfx11/gfx12 arch.
        for (raw, arch, mode) in [
            ("", "gfx1201", KvMode::Fp8),
            ("auto", "gfx1201", KvMode::Fp8),
            (" auto ", "gfx1201", KvMode::Fp8),
            ("bf16", "gfx1201", KvMode::Bf16),
            ("fp8", "gfx1201", KvMode::Fp8),
            ("q8", "gfx1201", KvMode::Q8),
            ("", "gfx1151", KvMode::Q8),
            ("auto", "gfx1151", KvMode::Q8),
            (" auto ", "gfx1151", KvMode::Q8),
            ("bf16", "gfx1151", KvMode::Bf16),
            (" bf16 ", "gfx1151", KvMode::Bf16),
            (" q8 ", "gfx1151", KvMode::Q8),
            ("", "gfx1100", KvMode::Q8),
            ("auto", "gfx1100", KvMode::Q8),
            (" auto ", "gfx1100", KvMode::Q8),
            ("bf16", "gfx1100", KvMode::Bf16),
            ("q8", "gfx1100", KvMode::Q8),
            ("", "gfx1101", KvMode::Bf16),
            ("auto", "gfx1101", KvMode::Bf16),
            ("auto", "gfx1102", KvMode::Bf16),
            ("auto", "gfx1150", KvMode::Bf16),
            ("auto", "gfx1152", KvMode::Bf16),
            ("auto", "gfx1200", KvMode::Bf16),
            ("q8", "gfx1200", KvMode::Q8),
        ] {
            let resolved = resolve_qwen4(raw, arch).expect("supported qwen4 mode");
            assert_eq!(resolved.mode, mode, "{raw:?} on {arch}");
            assert!(resolved.warning.is_none());
        }
        for arch in ["gfx1151", "gfx1100", "gfx1200"] {
            let Err(error) = resolve_qwen4("fp8", arch) else {
                panic!("fp8 off gfx1201 must be refused on {arch}");
            };
            assert!(error.contains("fp8") && error.contains(arch), "{error}");
        }
        for arch in ["gfx1030", "gfx906", "gfx942"] {
            let Err(error) = resolve_qwen4("q8", arch) else {
                panic!("q8 off gfx11/gfx12 must be refused on {arch}");
            };
            assert!(error.contains("q8") && error.contains(arch), "{error}");
        }
        for raw in ["asym3", "fwht3", "f16", "garbage"] {
            for arch in ["gfx1201", "gfx1151"] {
                let error = match resolve_qwen4(raw, arch) {
                    Ok(_) => panic!("explicit unsupported mode {raw} must be rejected"),
                    Err(error) => error,
                };
                assert!(error.contains("qwen4"));
                assert!(error.contains(raw));
            }
            assert_eq!(resolve(raw, p).mode, KvMode::Bf16);
            assert!(
                resolve(raw, p).warning.is_some(),
                "generic resolver remains fallback-and-warning only"
            );
        }
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
        // turbo/turbo3 → Fwht3 unaccepted → Q8 + warn (legacy non-Qwen path)
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
        assert!(resolve("fwht4", p).warning.is_some());
        assert_eq!(resolve("fwht4", p).mode, KvMode::Q8);
        assert_eq!(resolve("garbage", p).mode, KvMode::Q8);
    }

    /// `auto` is the site default on every non-Qwen q8 site: it resolves to
    /// q8 without the "unrecognized or unsupported" warning.
    #[test]
    fn non_qwen_auto_resolves_q8_without_warning() {
        for p in [&LLAMA_HFQ_POLICY, &HFQ_Q8_ONLY_POLICY] {
            let r = resolve("auto", p);
            assert_eq!(r.mode, KvMode::Q8, "site {}", p.site);
            assert!(r.warning.is_none(), "site {}: {:?}", p.site, r.warning);
        }
    }

    #[test]
    fn gemma4_full_policy() {
        let p = &GEMMA4_FULL_POLICY;
        for raw in ["", "auto", "q8"] {
            let r = resolve(raw, p);
            assert_eq!(r.mode, Q8, "{raw:?}");
            assert!(r.warning.is_none(), "{raw:?}");
        }
        let legacy = resolve("legacy-asym3", p);
        assert_eq!(legacy.mode, Asym3);
        assert!(legacy.warning.is_none());
        // Bare asymN/turboN/fwhtN are FWHT names; no hd512 FWHT tier → q8 + warn.
        for raw in [
            "asym3",
            "turbo3",
            "turbo",
            "fwht3",
            "asym4",
            "legacy-asym4",
            "garbage",
        ] {
            let r = resolve(raw, p);
            assert_eq!(r.mode, Q8, "{raw}");
            assert!(r.warning.is_some(), "{raw} must warn");
        }
        // Native presets are carried forward so the carrier fails closed.
        for (raw, mode) in [("fp8", Fp8), ("bf16", Bf16)] {
            let r = resolve(raw, p);
            assert_eq!(r.mode, mode);
            assert!(r.warning.is_some());
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
