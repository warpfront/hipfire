// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Typed prefill precision policy of Raon-1-mini (`HIPFIRE_RAON_PREFILL_PRECISION`).
//!
//! Every prefill GEMM *site* has an activation tier ([`PrefillActTier`]: `f16`, `a4`, `a8`,
//! `fp8`). The policy is parsed and resolved ONCE at load: the request (arch default table
//! overlaid with the environment) is intersected with what each site's weights can run
//! ([`effective_tier`]) and the effective tiers are stored as plain fields in `RaonWork`, so no
//! forward reads the environment. Only batched prefill at rows >= 64 outside graph capture and
//! Redline recording can run a quantized tier (the dispatch admission,
//! `hipfire_dispatch::pipeline::prefill_quant_admitted`); decode, speculative verify and
//! captured paths are F16 regardless.
//!
//! # Grammar (`HIPFIRE_RAON_PREFILL_PRECISION`)
//!
//! * unset / empty / `default`: the arch default table ([`default_table`]).
//! * `f16`: every site F16 (the global kill switch).
//! * `a4` | `a8` | `fp8`: every site requests that tier (sites whose weights cannot run it stay
//!   F16 and the load log says why).
//! * `[<base>,]<site>=<tier>,...` with `<base>` in `default|f16|a4|a8|fp8` (`default` when
//!   omitted) and `<site>` in `routed_gu|routed_dn|shared|dense|mla_q|mla_kv|mla_go|rda_qkvg|
//!   rda_o`. `site=f16` is that site's kill switch. Examples: `f16,mla_q=a8` (one site switched
//!   alone), `rda_o=f16`, `a8,routed_dn=fp8`.

use hipfire_dispatch::pipeline::PrefillActTier;
use hipfire_runtime::llama::WeightTensor;
use rdna_compute::DType;

use crate::weights::{RaonFfn, RaonMixer, RaonWeights};

/// Environment variable of the policy.
pub const PRECISION_ENV: &str = "HIPFIRE_RAON_PREFILL_PRECISION";

/// One prefill GEMM site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RaonPrecisionSite {
    /// Routed experts' fused gate/up GEMM (grouped, qt44).
    RoutedGateUp,
    /// Routed experts' down GEMM (grouped, qt44).
    RoutedDown,
    /// Shared expert gate/up/down of every MoE layer.
    Shared,
    /// First-layer dense MLP gate/up/down.
    Dense,
    /// MLA `q_a` and `q_b` projections.
    MlaQ,
    /// MLA `kv_a` projection and the `kv_b` expansion of the K/V mirror.
    MlaKv,
    /// MLA gate (`g_proj`) and output (`o_proj`) projections.
    MlaGo,
    /// RDA `q`/`k`/`v`/`g` projections.
    RdaQkvg,
    /// RDA output projection.
    RdaO,
}

impl RaonPrecisionSite {
    pub const ALL: [Self; 9] = [
        Self::RoutedGateUp,
        Self::RoutedDown,
        Self::Shared,
        Self::Dense,
        Self::MlaQ,
        Self::MlaKv,
        Self::MlaGo,
        Self::RdaQkvg,
        Self::RdaO,
    ];

    /// Name in `HIPFIRE_RAON_PREFILL_PRECISION` and in the load log.
    pub const fn name(self) -> &'static str {
        match self {
            Self::RoutedGateUp => "routed_gu",
            Self::RoutedDown => "routed_dn",
            Self::Shared => "shared",
            Self::Dense => "dense",
            Self::MlaQ => "mla_q",
            Self::MlaKv => "mla_kv",
            Self::MlaGo => "mla_go",
            Self::RdaQkvg => "rda_qkvg",
            Self::RdaO => "rda_o",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        Self::ALL.into_iter().find(|site| site.name() == s)
    }

    fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).expect("listed")
    }
}

/// An activation tier per site (the request, or the resolved effective tiers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaonSiteTiers {
    tiers: [PrefillActTier; 9],
}

impl Default for RaonSiteTiers {
    fn default() -> Self {
        Self::ALL_F16
    }
}

impl RaonSiteTiers {
    pub const ALL_F16: Self = Self {
        tiers: [PrefillActTier::F16; 9],
    };

    pub const fn all(tier: PrefillActTier) -> Self {
        Self { tiers: [tier; 9] }
    }

    pub fn get(&self, site: RaonPrecisionSite) -> PrefillActTier {
        self.tiers[site.index()]
    }

    pub fn set(&mut self, site: RaonPrecisionSite, tier: PrefillActTier) {
        self.tiers[site.index()] = tier;
    }
}

/// The arch default table. gfx1201: every site runs int8 activations (A8): W4A8 for the qt44
/// routed/shared/dense FFN, W8A8 for the Q8_0 attention-class projections. Chosen per site by
/// measurement on R9700 (`raon/pfx/ev/fp8/`, CHANGELOG): A8 and FP8 run at the same speed at every
/// site, A8 has the lower KLD at every site, A4 alone exceeds the +0.002 KLD budget at every FFN
/// site, and all-A8 costs code24 +0.0003 / WT2 +0.0011 vs all-F16 on the shipped artifact.
/// gfx1100 / gfx1151: the routed experts run W4A8 on the gfx11 twin of the grouped kernel
/// (7900 XTX pp512 +6.0%, Strix Halo +4.9%, KLD code24 -0.0005 / WT2 +0.00004 vs all-F16; receipts
/// `raon/pfx/ev/a8r3/`); shared and dense FFN can run it too but are unmeasured there, and the Q8_0
/// attention class has no gfx11 int8 route. Every other arch: all F16.
pub fn default_table(arch: &str) -> RaonSiteTiers {
    /// gfx1201 default, indexed like [`RaonPrecisionSite::ALL`].
    const GFX1201: RaonSiteTiers = RaonSiteTiers::all(PrefillActTier::A8);
    match arch {
        "gfx1201" => GFX1201,
        "gfx1100" | "gfx1151" => {
            let mut t = RaonSiteTiers::ALL_F16;
            t.set(RaonPrecisionSite::RoutedGateUp, PrefillActTier::A8);
            t.set(RaonPrecisionSite::RoutedDown, PrefillActTier::A8);
            t
        }
        _ => RaonSiteTiers::ALL_F16,
    }
}

fn grammar_help() -> String {
    let sites: Vec<&str> = RaonPrecisionSite::ALL.iter().map(|s| s.name()).collect();
    format!(
        "expected `default`, `f16`, `a4`, `a8`, `fp8`, or `[<base>,]<site>=<tier>,...` with base in \
         default|f16|a4|a8|fp8, site in {} and tier in f16|a4|a8|fp8",
        sites.join("|")
    )
}

/// The request: arch default table overlaid with the environment value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaonPrecisionPolicy {
    requested: RaonSiteTiers,
}

impl RaonPrecisionPolicy {
    /// The arch default table, nothing overridden.
    pub fn for_arch(arch: &str) -> Self {
        Self {
            requested: default_table(arch),
        }
    }

    /// Resolve `HIPFIRE_RAON_PREFILL_PRECISION` for `arch` (once, at load).
    pub fn from_env(arch: &str) -> Result<Self, String> {
        match hipfire_config::developer_var(PRECISION_ENV) {
            Err(_) => Ok(Self::for_arch(arch)),
            Ok(v) => Self::parse(&v, arch),
        }
    }

    /// Parse one value of the grammar in the module docs.
    pub fn parse(spec: &str, arch: &str) -> Result<Self, String> {
        let fail = |what: String| -> String {
            format!("{PRECISION_ENV}={spec:?}: {what}; {}", grammar_help())
        };
        let spec_l = spec.trim().to_ascii_lowercase();
        if spec_l.is_empty() {
            return Ok(Self::for_arch(arch));
        }
        let base_of = |tok: &str| -> Option<RaonSiteTiers> {
            if tok == "default" {
                Some(default_table(arch))
            } else {
                PrefillActTier::parse(tok).map(RaonSiteTiers::all)
            }
        };
        let mut tokens = spec_l.split(',').map(str::trim).peekable();
        let first = tokens.peek().copied().unwrap_or("");
        let mut requested = if first.contains('=') {
            default_table(arch)
        } else {
            tokens.next();
            base_of(first).ok_or_else(|| fail(format!("unknown base {first:?}")))?
        };
        let mut seen: Vec<RaonPrecisionSite> = Vec::new();
        for tok in tokens {
            let Some((site, tier)) = tok.split_once('=') else {
                return Err(fail(format!(
                    "{tok:?} is not `<site>=<tier>` (a base is only valid first)"
                )));
            };
            let site = RaonPrecisionSite::parse(site)
                .ok_or_else(|| fail(format!("unknown site {:?}", site.trim())))?;
            let tier = PrefillActTier::parse(tier)
                .ok_or_else(|| fail(format!("unknown tier {:?}", tier.trim())))?;
            if seen.contains(&site) {
                return Err(fail(format!("site {} given twice", site.name())));
            }
            seen.push(site);
            requested.set(site, tier);
        }
        Ok(Self { requested })
    }

    /// The requested tier of `site` (before the weight/arch intersection).
    pub fn requested(&self, site: RaonPrecisionSite) -> PrefillActTier {
        self.requested.get(site)
    }

    pub fn requested_tiers(&self) -> RaonSiteTiers {
        self.requested
    }
}

/// What a site's weight is, as far as the tier resolution cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteWeight {
    /// Q8_0 (Q8F16) weight, `k` input columns.
    Q8 {
        k: usize,
    },
    /// qt44 (`MQ4G256V2`) weight; `symmetric`: loader-verified `zp == -8 * sc`; `ffn`: the site
    /// has the grouped/one-expert qt44 cores (routed, shared, dense; not the attention-class
    /// projections).
    Qt44 {
        symmetric: bool,
        m: usize,
        k: usize,
        ffn: bool,
    },
    F32,
    Other,
}

impl SiteWeight {
    pub fn of(w: &WeightTensor, symmetric: bool, ffn: bool) -> Self {
        match w.gpu_dtype {
            DType::Q8_0 => Self::Q8 { k: w.k },
            DType::MQ4G256V2 => Self::Qt44 {
                symmetric,
                m: w.m,
                k: w.k,
                ffn,
            },
            DType::F32 => Self::F32,
            _ => Self::Other,
        }
    }

    /// `Ok` when this weight can run `tier` at prefill rows >= 64 on `arch`; otherwise the reason.
    fn can_run(&self, arch: &str, tier: PrefillActTier) -> Result<(), &'static str> {
        if !hipfire_dispatch::pipeline::grouped_sym_tier_admits(arch, tier) {
            return Err(
                if hipfire_dispatch::pipeline::grouped_sym_tier_admits(arch, PrefillActTier::A8) {
                    "A4/FP8 prefill tiers are gfx1201-only"
                } else {
                    "quantized prefill tiers are gfx1201/gfx1100/gfx1151-only"
                },
            );
        }
        if arch != "gfx1201" && matches!(*self, Self::Q8 { .. }) {
            return Err("Q8_0 has no gfx11 int8 route");
        }
        match *self {
            Self::Q8 { k } => {
                if tier == PrefillActTier::A4 {
                    Err("Q8_0 has no A4 route")
                } else if k % 32 != 0 {
                    Err("Q8_0 K is not a multiple of 32")
                } else {
                    Ok(())
                }
            }
            Self::Qt44 {
                symmetric,
                m,
                k,
                ffn,
            } => {
                if !ffn {
                    Err("qt44 attention-class projections have no tier route")
                } else if !symmetric {
                    Err("qt44 headers are not verified symmetric")
                } else if m % 64 != 0 || k % 256 != 0 {
                    Err("qt44 shape is not tileable (M % 64, K % 256)")
                } else {
                    Ok(())
                }
            }
            Self::F32 => Err("F32 weights are never tiered"),
            Self::Other => Err("weight dtype has no tier route"),
        }
    }
}

/// Resolved tier of one site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteResolution {
    pub requested: PrefillActTier,
    pub effective: PrefillActTier,
    /// Weights of the site that can run `effective` / all weights of the site.
    pub runnable: usize,
    pub total: usize,
    /// Why some (or all) weights stay F16 under a quantized request.
    pub reason: Option<&'static str>,
}

/// Intersect the request for one site with what its `weights` (across every layer) can run on
/// `arch`: the effective tier is the requested tier when at least one weight can run it (the
/// rest run F16 at the call site's own admission) and F16 otherwise.
pub fn effective_tier(
    arch: &str,
    requested: PrefillActTier,
    weights: &[SiteWeight],
) -> SiteResolution {
    let total = weights.len();
    if requested == PrefillActTier::F16 {
        return SiteResolution {
            requested,
            effective: PrefillActTier::F16,
            runnable: 0,
            total,
            reason: None,
        };
    }
    let mut runnable = 0;
    let mut reason = None;
    for w in weights {
        match w.can_run(arch, requested) {
            Ok(()) => runnable += 1,
            Err(why) => {
                reason.get_or_insert(why);
            }
        }
    }
    if total == 0 {
        reason = Some("the model has no such weights");
    }
    SiteResolution {
        requested,
        effective: if runnable > 0 {
            requested
        } else {
            PrefillActTier::F16
        },
        runnable,
        total,
        reason,
    }
}

/// The weights of every site, collected from the loaded model.
#[derive(Default)]
pub struct SiteWeights {
    sites: [Vec<SiteWeight>; 9],
}

impl SiteWeights {
    pub fn get(&self, site: RaonPrecisionSite) -> &[SiteWeight] {
        &self.sites[site.index()]
    }

    pub fn push(&mut self, site: RaonPrecisionSite, w: SiteWeight) {
        self.sites[site.index()].push(w);
    }

    /// Collect every tiered weight of `weights`. Sites map to the projections the forward
    /// routes through them (see `forward.rs`): `rda_qkvg` = q/k/v/g, `rda_o` = o, `mla_q` =
    /// q_a/q_b, `mla_kv` = kv_a/kv_b, `mla_go` = g/o; the F32 RDA projections (b, w, f_a, f_b)
    /// and the lm_head are never tiered and not collected.
    pub fn collect(weights: &RaonWeights) -> Self {
        use RaonPrecisionSite as S;
        let mut out = Self::default();
        for layer in &weights.layers {
            match &layer.mixer {
                RaonMixer::Rda(r) => {
                    for w in [&r.q_proj, &r.k_proj, &r.v_proj, &r.g_proj] {
                        out.push(S::RdaQkvg, SiteWeight::of(w, false, false));
                    }
                    out.push(S::RdaO, SiteWeight::of(&r.o_proj, false, false));
                }
                RaonMixer::Mla(m) => {
                    for w in [&m.q_a_proj, &m.q_b_proj] {
                        out.push(S::MlaQ, SiteWeight::of(w, false, false));
                    }
                    for w in [&m.kv_a_proj, &m.kv_b_proj] {
                        out.push(S::MlaKv, SiteWeight::of(w, false, false));
                    }
                    for w in [&m.g_proj, &m.o_proj] {
                        out.push(S::MlaGo, SiteWeight::of(w, false, false));
                    }
                }
            }
            match &layer.ffn {
                RaonFfn::Dense(d) => {
                    for w in [&d.gate, &d.up, &d.down] {
                        out.push(S::Dense, SiteWeight::of(w, d.iu4_symmetric, true));
                    }
                }
                RaonFfn::Moe(m) => {
                    for w in [&m.shared.gate, &m.shared.up, &m.shared.down] {
                        out.push(S::Shared, SiteWeight::of(w, m.shared.iu4_symmetric, true));
                    }
                    if let Some(e) = m.experts.first() {
                        out.push(
                            S::RoutedGateUp,
                            SiteWeight::of(&e.gate_up, m.iu4_symmetric, true),
                        );
                        out.push(
                            S::RoutedDown,
                            SiteWeight::of(&e.down, m.iu4_symmetric, true),
                        );
                    }
                }
            }
        }
        out
    }
}

/// The resolved policy: the request and, per site, what actually runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaonPrecisionPlan {
    pub requested: RaonSiteTiers,
    pub effective: RaonSiteTiers,
    pub sites: Vec<(RaonPrecisionSite, SiteResolution)>,
}

impl RaonPrecisionPolicy {
    /// Resolve every site against its weights on `arch`.
    pub fn resolve(&self, arch: &str, weights: &SiteWeights) -> RaonPrecisionPlan {
        let mut effective = RaonSiteTiers::ALL_F16;
        let mut sites = Vec::new();
        for site in RaonPrecisionSite::ALL {
            let r = effective_tier(arch, self.requested(site), weights.get(site));
            effective.set(site, r.effective);
            sites.push((site, r));
        }
        RaonPrecisionPlan {
            requested: self.requested,
            effective,
            sites,
        }
    }
}

impl RaonPrecisionPlan {
    /// The one load-log line: every site's effective tier (and, when it differs from the request,
    /// the request and why), plus how to restore all-F16.
    pub fn coverage_line(&self) -> String {
        let mut parts = Vec::new();
        for (site, r) in &self.sites {
            let mut p = format!("{}={}", site.name(), r.effective.name());
            if r.effective != r.requested {
                p.push_str(&format!(
                    " (requested {}: {})",
                    r.requested.name(),
                    r.reason.unwrap_or("not runnable")
                ));
            } else if r.effective != PrefillActTier::F16 && r.runnable < r.total {
                p.push_str(&format!(
                    " ({} of {} weights; the rest F16: {})",
                    r.runnable,
                    r.total,
                    r.reason.unwrap_or("not runnable")
                ));
            }
            parts.push(p);
        }
        format!(
            "raon1_mini: prefill precision: {}; quantized tiers apply at prefill rows >= 64 \
             outside graph capture (decode/verify rows stay F16); {PRECISION_ENV}=f16 restores all-F16",
            parts.join(" ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use PrefillActTier::{Fp8, A4, A8, F16};
    use RaonPrecisionSite as S;

    const R9700: &str = "gfx1201";

    fn parse(s: &str) -> Result<RaonPrecisionPolicy, String> {
        RaonPrecisionPolicy::parse(s, R9700)
    }

    #[test]
    fn default_tables_per_arch() {
        let d = default_table(R9700);
        assert_eq!(d, RaonSiteTiers::all(A8));
        let mut gfx11 = RaonSiteTiers::ALL_F16;
        gfx11.set(S::RoutedGateUp, A8);
        gfx11.set(S::RoutedDown, A8);
        for arch in ["gfx1100", "gfx1151"] {
            assert_eq!(default_table(arch), gfx11, "{arch}");
        }
        for arch in ["gfx1200", "gfx942", ""] {
            assert_eq!(default_table(arch), RaonSiteTiers::ALL_F16, "{arch}");
        }
        assert_eq!(parse("").unwrap().requested_tiers(), d);
        assert_eq!(parse("  default ").unwrap().requested_tiers(), d);
        assert_eq!(
            RaonPrecisionPolicy::parse("default", "gfx1100")
                .unwrap()
                .requested_tiers(),
            gfx11
        );
    }

    #[test]
    fn global_values_set_every_site() {
        assert_eq!(
            parse("f16").unwrap().requested_tiers(),
            RaonSiteTiers::ALL_F16
        );
        for (s, t) in [("a4", A4), ("A8", A8), (" fp8 ", Fp8)] {
            let p = parse(s).unwrap();
            for site in RaonPrecisionSite::ALL {
                assert_eq!(p.requested(site), t, "{s} {}", site.name());
            }
        }
    }

    #[test]
    fn base_plus_site_overrides() {
        // one site switched alone
        let p = parse("f16,mla_q=a8").unwrap();
        for site in RaonPrecisionSite::ALL {
            assert_eq!(p.requested(site), if site == S::MlaQ { A8 } else { F16 });
        }
        // a site kill switch on the default table
        let p = parse("rda_o=f16,routed_dn=f16").unwrap();
        assert_eq!(p.requested(S::RoutedGateUp), A8);
        assert_eq!(p.requested(S::RoutedDown), F16);
        // global base with a per-site override and a kill switch
        let p = parse("a8,routed_dn=fp8, mla_go = F16").unwrap();
        assert_eq!(p.requested(S::RoutedDown), Fp8);
        assert_eq!(p.requested(S::MlaGo), F16);
        assert_eq!(p.requested(S::Dense), A8);
        // explicit default base
        let p = parse("default,shared=fp8").unwrap();
        assert_eq!(p.requested(S::RoutedGateUp), A8);
        assert_eq!(p.requested(S::Shared), Fp8);
        // other arch: default base is all F16
        let p = RaonPrecisionPolicy::parse("dense=a8", "gfx1100").unwrap();
        assert_eq!(p.requested(S::RoutedGateUp), A8);
        assert_eq!(p.requested(S::Shared), F16);
        assert_eq!(p.requested(S::Dense), A8);
    }

    #[test]
    fn grammar_errors_name_the_valid_values() {
        for bad in [
            "a5",
            "bogus=a8",
            "mla_q=a5",
            "f16,a8",
            "mla_q=a8,f16",
            "mla_q=a8,mla_q=fp8",
            "mla_q",
            "f16,",
            ",f16",
            "mla_q=",
            "=a8",
        ] {
            let e = parse(bad).unwrap_err();
            assert!(e.contains(PRECISION_ENV), "{bad}: {e}");
            assert!(
                e.contains("routed_gu") && e.contains("rda_o") && e.contains("fp8"),
                "{bad}: {e}"
            );
        }
        let e = parse("bogus=a8").unwrap_err();
        assert!(e.contains("unknown site \"bogus\""), "{e}");
        let e = parse("mla_q=a5").unwrap_err();
        assert!(e.contains("unknown tier \"a5\""), "{e}");
    }

    #[test]
    fn effective_tier_per_dtype() {
        let q8 = SiteWeight::Q8 { k: 2560 };
        let sym = |ffn| SiteWeight::Qt44 {
            symmetric: true,
            m: 1088,
            k: 2560,
            ffn,
        };
        // Q8_0: A8/Fp8 run, A4 does not.
        for t in [A8, Fp8] {
            let r = effective_tier(R9700, t, &[q8, q8]);
            assert_eq!(
                (r.effective, r.runnable, r.total, r.reason),
                (t, 2, 2, None)
            );
        }
        let r = effective_tier(R9700, A4, &[q8]);
        assert_eq!(r.effective, F16);
        assert_eq!(r.reason, Some("Q8_0 has no A4 route"));
        // K not a multiple of 32
        let r = effective_tier(R9700, A8, &[SiteWeight::Q8 { k: 100 }]);
        assert_eq!(r.effective, F16);
        // qt44 FFN: every quantized tier runs on symmetric-verified weights
        for t in [A4, A8, Fp8] {
            assert_eq!(effective_tier(R9700, t, &[sym(true)]).effective, t);
        }
        // qt44 unverified / attention-class / misshapen
        let unverified = SiteWeight::Qt44 {
            symmetric: false,
            m: 1088,
            k: 2560,
            ffn: true,
        };
        let r = effective_tier(R9700, A8, &[unverified]);
        assert_eq!(
            (r.effective, r.reason),
            (F16, Some("qt44 headers are not verified symmetric"))
        );
        let r = effective_tier(R9700, A8, &[sym(false)]);
        assert_eq!(
            (r.effective, r.reason),
            (
                F16,
                Some("qt44 attention-class projections have no tier route")
            )
        );
        let odd = SiteWeight::Qt44 {
            symmetric: true,
            m: 1088,
            k: 1088,
            ffn: true,
        };
        assert_eq!(effective_tier(R9700, A8, &[odd]).effective, F16);
        // F32 is never tiered
        let r = effective_tier(R9700, A8, &[SiteWeight::F32]);
        assert_eq!(
            (r.effective, r.reason),
            (F16, Some("F32 weights are never tiered"))
        );
        // gfx11: A8 runs on the symmetric qt44 FFN weight (grouped gfx11 twin), never on Q8_0;
        // A4/FP8 stay gfx1201-only; other archs: F16 with the reason
        for arch in ["gfx1100", "gfx1151"] {
            let r = effective_tier(arch, A8, &[q8, sym(true)]);
            assert_eq!((r.effective, r.runnable, r.total), (A8, 1, 2), "{arch}");
            assert_eq!(effective_tier(arch, A8, &[q8]).effective, F16, "{arch}");
            let r = effective_tier(arch, Fp8, &[sym(true)]);
            assert_eq!(r.effective, F16, "{arch}");
            assert_eq!(r.reason, Some("A4/FP8 prefill tiers are gfx1201-only"));
        }
        let r = effective_tier("gfx942", A8, &[q8, sym(true)]);
        assert_eq!(r.effective, F16);
        assert_eq!(
            r.reason,
            Some("quantized prefill tiers are gfx1201/gfx1100/gfx1151-only")
        );
        let d = default_table("gfx1100");
        assert_eq!(d.get(RaonPrecisionSite::RoutedGateUp), A8);
        assert_eq!(d.get(RaonPrecisionSite::RoutedDown), A8);
        assert_eq!(d.get(RaonPrecisionSite::Shared), F16);
        assert_eq!(d.get(RaonPrecisionSite::MlaQ), F16);
        // an F16 request never has a reason, a site without weights falls back with one
        assert_eq!(effective_tier(R9700, F16, &[q8]).reason, None);
        let r = effective_tier(R9700, A8, &[]);
        assert_eq!(r.effective, F16);
        assert!(r.reason.is_some());
        // mixed site: partially runnable keeps the tier
        let r = effective_tier(R9700, A8, &[q8, SiteWeight::F32]);
        assert_eq!((r.effective, r.runnable, r.total), (A8, 1, 2));
    }

    #[test]
    fn resolve_and_coverage_line() {
        let mut w = SiteWeights::default();
        let q8 = SiteWeight::Q8 { k: 2560 };
        let sym = SiteWeight::Qt44 {
            symmetric: true,
            m: 2176,
            k: 2560,
            ffn: true,
        };
        w.push(S::RoutedGateUp, sym);
        w.push(S::RoutedDown, sym);
        for site in [S::MlaQ, S::MlaKv, S::MlaGo, S::RdaQkvg, S::RdaO] {
            w.push(site, q8);
        }
        w.push(S::Shared, sym);
        w.push(S::Dense, sym);
        let plan = parse("a4").unwrap().resolve(R9700, &w);
        assert_eq!(plan.effective.get(S::RoutedGateUp), A4);
        assert_eq!(plan.effective.get(S::MlaQ), F16);
        let line = plan.coverage_line();
        assert!(
            line.contains("routed_gu=a4 routed_dn=a4 shared=a4 dense=a4"),
            "{line}"
        );
        assert!(
            line.contains("mla_q=f16 (requested a4: Q8_0 has no A4 route)"),
            "{line}"
        );
        assert!(
            line.contains("HIPFIRE_RAON_PREFILL_PRECISION=f16"),
            "{line}"
        );
        let plan = parse("f16,mla_q=a8").unwrap().resolve(R9700, &w);
        assert_eq!(plan.effective.get(S::MlaQ), A8);
        assert_eq!(plan.effective.get(S::RoutedGateUp), F16);
        assert!(plan.coverage_line().contains("mla_q=a8"));
        // an all-A8 request resolved on gfx1100: the Q8_0 attention class falls back to F16, the
        // symmetric qt44 FFN sites keep A8 (gfx11 grouped twin)
        let plan = parse("default").unwrap().resolve("gfx1100", &w);
        assert_eq!(plan.effective.get(S::MlaQ), F16);
        assert_eq!(plan.effective.get(S::RoutedGateUp), A8);
    }
}
