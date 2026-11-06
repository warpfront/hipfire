//! Per-projection / per-layer mask of the Qwen4 dense IU4 trunk route
//! (`HIPFIRE_QWEN4_TRUNK_IU4`).
//!
//! Grammar: unset, empty or `0` = no projection; `1` or `all` = every family
//! on every layer; otherwise a comma list of `family[@lo-hi]` tokens. `lo-hi`
//! are inclusive full-layer indices (0-based); `@n` is a single layer; no `@`
//! is every layer. Families: `gdn.qkv gdn.z gdn.a gdn.b gdn.out qsa.q qsa.k
//! qsa.v qsa.idx qsa.o`; groups `gdn` (the five GDN), `qsa` (the five QSA) and
//! `all`.

use std::fmt::Write as _;

/// One trunk projection family (a bit of [`Qwen4TrunkMask::bits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrunkFamily {
    GdnQkv = 0,
    GdnZ = 1,
    GdnA = 2,
    GdnB = 3,
    GdnOut = 4,
    QsaQ = 5,
    QsaK = 6,
    QsaV = 7,
    QsaIdx = 8,
    QsaO = 9,
}

impl TrunkFamily {
    pub const ALL: [TrunkFamily; 10] = [
        Self::GdnQkv,
        Self::GdnZ,
        Self::GdnA,
        Self::GdnB,
        Self::GdnOut,
        Self::QsaQ,
        Self::QsaK,
        Self::QsaV,
        Self::QsaIdx,
        Self::QsaO,
    ];

    /// This family's bit in a per-layer `u16`.
    pub const fn bit(self) -> u16 {
        1 << self as u16
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::GdnQkv => "gdn.qkv",
            Self::GdnZ => "gdn.z",
            Self::GdnA => "gdn.a",
            Self::GdnB => "gdn.b",
            Self::GdnOut => "gdn.out",
            Self::QsaQ => "qsa.q",
            Self::QsaK => "qsa.k",
            Self::QsaV => "qsa.v",
            Self::QsaIdx => "qsa.idx",
            Self::QsaO => "qsa.o",
        }
    }
}

/// The five GDN families.
pub const TRUNK_GDN_BITS: u16 = 0x001f;
/// The five QSA families.
pub const TRUNK_QSA_BITS: u16 = 0x03e0;
/// Every family.
pub const TRUNK_ALL_BITS: u16 = 0x03ff;

/// Parsed `family[@lo-hi],...` mask. Immutable once parsed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qwen4TrunkMask {
    /// `(family bits, lo, hi)`, inclusive layer indices; `hi == u32::MAX` is
    /// open-ended.
    ranges: Vec<(u16, u32, u32)>,
    /// The source token of each range (same index), for error messages.
    tokens: Vec<String>,
}

impl Qwen4TrunkMask {
    /// Parse `raw`; the error names the offending token.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let text = raw.trim();
        match text {
            "" | "0" => return Ok(Self::default()),
            "1" => {
                return Ok(Self {
                    ranges: vec![(TRUNK_ALL_BITS, 0, u32::MAX)],
                    tokens: vec!["1".to_owned()],
                })
            }
            _ => {}
        }
        let mut ranges = Vec::new();
        let mut tokens = Vec::new();
        for token in text.split(',') {
            let token = token.trim();
            let bad = |why: &str| format!("bad token `{token}` ({why})");
            let (name, span) = match token.split_once('@') {
                Some((name, span)) => (name, Some(span)),
                None => (token, None),
            };
            let bits = match name {
                "all" => TRUNK_ALL_BITS,
                "gdn" => TRUNK_GDN_BITS,
                "qsa" => TRUNK_QSA_BITS,
                other => TrunkFamily::ALL
                    .iter()
                    .find(|f| f.name() == other)
                    .map(|f| f.bit())
                    .ok_or_else(|| bad("unknown family"))?,
            };
            let (lo, hi) = match span {
                None => (0, u32::MAX),
                Some(span) => {
                    let layer = |s: &str| {
                        // Digits only: `str::parse` also takes a leading `+`.
                        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                            .then(|| s.parse::<u32>().ok())
                            .flatten()
                            .filter(|n| *n != u32::MAX)
                            .ok_or_else(|| bad("layer index is not a non-negative integer"))
                    };
                    match span.split_once('-') {
                        Some((lo, hi)) => (layer(lo)?, layer(hi)?),
                        None => {
                            let n = layer(span)?;
                            (n, n)
                        }
                    }
                }
            };
            if lo > hi {
                return Err(bad("range start is above its end"));
            }
            ranges.push((bits, lo, hi));
            tokens.push(token.to_owned());
        }
        Ok(Self { ranges, tokens })
    }

    /// Reject a mask that names layers a `layers`-layer trunk does not have:
    /// any band with `lo >= layers`, or with `hi >= layers` unless it is
    /// open-ended (no `@`). The error names the token.
    pub fn check_layers(&self, layers: usize) -> Result<(), String> {
        let layers = u32::try_from(layers).unwrap_or(u32::MAX);
        for (&(_, lo, hi), token) in self.ranges.iter().zip(&self.tokens) {
            if lo >= layers || (hi >= layers && hi != u32::MAX) {
                return Err(format!(
                    "bad token `{token}` (layers {lo}..={hi} are outside the trunk's {layers} layers)"
                ));
            }
        }
        Ok(())
    }

    /// Families selected on `layer`, as [`TrunkFamily::bit`]s.
    pub fn bits(&self, layer: usize) -> u16 {
        let layer = u32::try_from(layer).unwrap_or(u32::MAX);
        self.ranges
            .iter()
            .filter(|(_, lo, hi)| (*lo..=*hi).contains(&layer))
            .fold(0, |bits, (b, _, _)| bits | b)
    }

    pub fn contains(&self, family: TrunkFamily, layer: usize) -> bool {
        self.bits(layer) & family.bit() != 0
    }

    /// No family on any layer.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Normalized spelling: per-family merged layer intervals; families with
    /// identical intervals collapse into `all` / `gdn` / `qsa`. `off` when
    /// empty. Parses back to an equal mask (up to range layout).
    pub fn canonical(&self) -> String {
        if self.is_empty() {
            return "off".into();
        }
        let mut per_family: Vec<(u16, Vec<(u32, u32)>)> = Vec::new();
        for family in TrunkFamily::ALL {
            let mut spans: Vec<(u32, u32)> = self
                .ranges
                .iter()
                .filter(|(b, _, _)| b & family.bit() != 0)
                .map(|&(_, lo, hi)| (lo, hi))
                .collect();
            if spans.is_empty() {
                continue;
            }
            spans.sort_unstable();
            let mut merged: Vec<(u32, u32)> = Vec::new();
            for (lo, hi) in spans {
                match merged.last_mut() {
                    Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                    _ => merged.push((lo, hi)),
                }
            }
            match per_family.iter_mut().find(|(_, s)| *s == merged) {
                Some((bits, _)) => *bits |= family.bit(),
                None => per_family.push((family.bit(), merged)),
            }
        }
        let mut out = String::new();
        for (mut bits, spans) in per_family {
            let mut names: Vec<&str> = Vec::new();
            if bits == TRUNK_ALL_BITS {
                names.push("all");
                bits = 0;
            }
            for (group, name) in [(TRUNK_GDN_BITS, "gdn"), (TRUNK_QSA_BITS, "qsa")] {
                if bits & group == group {
                    names.push(name);
                    bits &= !group;
                }
            }
            names.extend(
                TrunkFamily::ALL
                    .iter()
                    .filter(|f| bits & f.bit() != 0)
                    .map(|f| f.name()),
            );
            for name in names {
                for &(lo, hi) in &spans {
                    if !out.is_empty() {
                        out.push(',');
                    }
                    out.push_str(name);
                    match (lo, hi) {
                        (0, u32::MAX) => {}
                        (lo, hi) if lo == hi => {
                            let _ = write!(out, "@{lo}");
                        }
                        (lo, hi) => {
                            let _ = write!(out, "@{lo}-{hi}");
                        }
                    }
                }
            }
        }
        out
    }
}

/// The explicit environment override of the trunk route.
pub const TRUNK_IU4_ENV: &str = "HIPFIRE_QWEN4_TRUNK_IU4";

/// Top-level key of the HFQ metadata envelope that names a trunk default the
/// artifact's bake tooling vouches for (a string, see [`TrunkIu4Default`]).
pub const TRUNK_IU4_DEFAULT_KEY: &str = "qwen4_trunk_iu4_default";

/// A trunk route default that an artifact opts into through its embedded
/// [`TRUNK_IU4_DEFAULT_KEY`] marker. Only the bake tooling writes the marker;
/// an unmarked artifact keeps the route off unless
/// [`TRUNK_IU4_ENV`] selects a mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrunkIu4Default {
    /// `gfx1151-zab-v1`: the GDN Z, alpha and beta projections (the `zab`
    /// fold) on exact gfx1151.
    Gfx1151ZabV1,
    /// `gfx1201-q8f16-v1`: a Q8F16 (QT3) trunk overlay whose Q8_0 consumers
    /// carry G0 row-twin proofs on exact gfx1201. It selects no IU4 route (an
    /// empty mask): the marker only gates the overlay's `auto` application to
    /// the architecture its proofs cover.
    Gfx1201Q8f16V1,
}

impl TrunkIu4Default {
    /// The marker value this default is written and read as.
    pub const fn marker(self) -> &'static str {
        match self {
            Self::Gfx1151ZabV1 => "gfx1151-zab-v1",
            Self::Gfx1201Q8f16V1 => "gfx1201-q8f16-v1",
        }
    }

    /// The marker's default, or `None` for a value this build does not know.
    pub fn from_marker(marker: &str) -> Option<Self> {
        [Self::Gfx1151ZabV1, Self::Gfx1201Q8f16V1]
            .into_iter()
            .find(|default| default.marker() == marker)
    }

    /// The one exact architecture this default applies on.
    pub const fn arch(self) -> &'static str {
        match self {
            Self::Gfx1151ZabV1 => "gfx1151",
            Self::Gfx1201Q8f16V1 => "gfx1201",
        }
    }

    /// The mask this default selects, in the [`TRUNK_IU4_ENV`] grammar.
    pub const fn mask_spec(self) -> &'static str {
        match self {
            Self::Gfx1151ZabV1 => "gdn.z,gdn.a,gdn.b",
            Self::Gfx1201Q8f16V1 => "",
        }
    }
}

/// Where the effective mask of a load came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrunkIu4MaskSource {
    /// [`TRUNK_IU4_ENV`] was set (any value, `0` and empty included).
    Env,
    /// No env override; the artifact's marker selected its default on its
    /// exact architecture.
    ArtifactDefault(TrunkIu4Default),
    /// No env override and no applicable artifact default: the route is off.
    Off,
}

/// The effective mask of one load plus its provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4TrunkIu4Route {
    pub mask: Qwen4TrunkMask,
    pub source: TrunkIu4MaskSource,
}

impl Qwen4TrunkMask {
    /// Resolve the trunk mask of one artifact load. `env` is the raw
    /// [`TRUNK_IU4_ENV`] value (`None` = unset); `arch` the exact device
    /// architecture; `artifact` the artifact's embedded default marker.
    ///
    /// An explicit env value always wins and keeps the [`Qwen4TrunkMask::parse`]
    /// grammar (`0` / empty = no projection, which also switches a marked
    /// artifact's default off); a bad value is an error whatever the
    /// artifact. With the variable unset, a marked artifact takes its default
    /// mask only on the default's exact architecture; every other artifact
    /// and architecture stays off.
    pub fn resolve_iu4(
        env: Option<&str>,
        arch: &str,
        artifact: Option<TrunkIu4Default>,
    ) -> Result<Qwen4TrunkIu4Route, String> {
        match (env, artifact) {
            (Some(raw), _) => Ok(Qwen4TrunkIu4Route {
                mask: Self::parse(raw).map_err(|e| format!("{TRUNK_IU4_ENV}={raw:?}: {e}"))?,
                source: TrunkIu4MaskSource::Env,
            }),
            (None, Some(default)) if default.arch() == arch => Ok(Qwen4TrunkIu4Route {
                mask: Self::parse(default.mask_spec())
                    .map_err(|e| format!("trunk default {}: {e}", default.marker()))?,
                source: TrunkIu4MaskSource::ArtifactDefault(default),
            }),
            (None, _) => Ok(Qwen4TrunkIu4Route {
                mask: Self::default(),
                source: TrunkIu4MaskSource::Off,
            }),
        }
    }
}

/// [`Qwen4TrunkMask::resolve_iu4`] over the process's [`TRUNK_IU4_ENV`].
pub fn qwen4_trunk_iu4_route(
    arch: &str,
    artifact: Option<TrunkIu4Default>,
) -> Result<Qwen4TrunkIu4Route, String> {
    Qwen4TrunkMask::resolve_iu4(
        hipfire_config::developer_var(TRUNK_IU4_ENV).ok().as_deref(),
        arch,
        artifact,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layers(mask: &Qwen4TrunkMask, family: TrunkFamily) -> Vec<usize> {
        (0..48).filter(|&l| mask.contains(family, l)).collect()
    }

    #[test]
    fn all_and_one_select_every_family_everywhere() {
        for raw in ["all", "1"] {
            let mask = Qwen4TrunkMask::parse(raw).unwrap();
            assert_eq!(mask.bits(0), TRUNK_ALL_BITS, "{raw}");
            assert_eq!(mask.bits(47), TRUNK_ALL_BITS, "{raw}");
            assert_eq!(mask.canonical(), "all");
        }
    }

    #[test]
    fn zero_and_empty_are_off() {
        for raw in ["0", "", "  "] {
            let mask = Qwen4TrunkMask::parse(raw).unwrap();
            assert!(mask.is_empty());
            assert_eq!(mask.bits(3), 0);
            assert_eq!(mask.canonical(), "off");
        }
    }

    #[test]
    fn group_plus_ranged_family() {
        let mask = Qwen4TrunkMask::parse("gdn,qsa.o@3-15").unwrap();
        assert_eq!(mask.bits(0), TRUNK_GDN_BITS);
        assert_eq!(mask.bits(2), TRUNK_GDN_BITS);
        assert_eq!(mask.bits(3), TRUNK_GDN_BITS | TrunkFamily::QsaO.bit());
        assert_eq!(mask.bits(15), TRUNK_GDN_BITS | TrunkFamily::QsaO.bit());
        assert_eq!(mask.bits(16), TRUNK_GDN_BITS);
        assert_eq!(
            layers(&mask, TrunkFamily::QsaO),
            (3..=15).collect::<Vec<_>>()
        );
        assert!(!mask.contains(TrunkFamily::QsaQ, 5));
        assert_eq!(mask.canonical(), "gdn,qsa.o@3-15");
    }

    #[test]
    fn single_layer() {
        let mask = Qwen4TrunkMask::parse("gdn.qkv@7").unwrap();
        assert_eq!(layers(&mask, TrunkFamily::GdnQkv), vec![7]);
        assert_eq!(mask.bits(7), TrunkFamily::GdnQkv.bit());
        assert_eq!(mask.bits(6), 0);
        assert_eq!(mask.canonical(), "gdn.qkv@7");
    }

    #[test]
    fn canonical_merges_and_collapses() {
        let mask = Qwen4TrunkMask::parse(
            "gdn.qkv@0-3,gdn.qkv@4-9,gdn.z@0-9,gdn.a@0-9,gdn.b@0-9,gdn.out@0-9,qsa.idx",
        )
        .unwrap();
        assert_eq!(mask.canonical(), "gdn@0-9,qsa.idx");
        assert_eq!(
            Qwen4TrunkMask::parse(&mask.canonical()).unwrap().bits(5),
            TRUNK_GDN_BITS | TrunkFamily::QsaIdx.bit()
        );
    }

    #[test]
    fn bad_tokens_are_named() {
        for (raw, token) in [
            ("gdn,qsa.x", "qsa.x"),
            ("gdn.qkv@", "gdn.qkv@"),
            ("gdn.qkv@9-3", "gdn.qkv@9-3"),
            ("gdn.z@a", "gdn.z@a"),
            ("gdn,,qsa", ""),
            ("gdn,1", "1"),
        ] {
            let error = Qwen4TrunkMask::parse(raw).unwrap_err();
            assert!(error.contains(&format!("`{token}`")), "{raw}: {error}");
        }
    }

    #[test]
    fn layer_indices_are_digits_only() {
        for raw in [
            "gdn.z@+3",
            "gdn.z@+3-9",
            "gdn.z@3-+9",
            "gdn.z@-3",
            "gdn.z@3--9",
        ] {
            assert!(Qwen4TrunkMask::parse(raw).is_err(), "{raw}");
        }
        assert!(Qwen4TrunkMask::parse("gdn.z@3-9").is_ok());
    }

    #[test]
    fn check_layers_rejects_bands_outside_the_trunk() {
        let trunk = 48;
        for raw in [
            "all",
            "1",
            "0",
            "gdn",
            "gdn.qkv@0-47",
            "qsa.o@47",
            "gdn,qsa.o@3-15",
        ] {
            Qwen4TrunkMask::parse(raw)
                .unwrap()
                .check_layers(trunk)
                .unwrap_or_else(|e| {
                    panic!("{raw}: {e}");
                });
        }
        for (raw, token) in [
            ("gdn.qkv@48", "gdn.qkv@48"),
            ("gdn,qsa.o@40-48", "qsa.o@40-48"),
            ("gdn.z@0-47,gdn.a@50-60", "gdn.a@50-60"),
            ("qsa.k@100", "qsa.k@100"),
        ] {
            let error = Qwen4TrunkMask::parse(raw)
                .unwrap()
                .check_layers(trunk)
                .unwrap_err();
            assert!(error.contains(&format!("`{token}`")), "{raw}: {error}");
        }
    }

    const ZAB: Option<TrunkIu4Default> = Some(TrunkIu4Default::Gfx1151ZabV1);
    const ZAB_BITS: u16 =
        TrunkFamily::GdnZ.bit() | TrunkFamily::GdnA.bit() | TrunkFamily::GdnB.bit();

    fn route(
        env: Option<&str>,
        arch: &str,
        artifact: Option<TrunkIu4Default>,
    ) -> Qwen4TrunkIu4Route {
        Qwen4TrunkMask::resolve_iu4(env, arch, artifact).unwrap()
    }

    #[test]
    fn zab_marker_round_trips_and_names_the_s11_map() {
        let zab = TrunkIu4Default::Gfx1151ZabV1;
        assert_eq!(zab.marker(), "gfx1151-zab-v1");
        assert_eq!(TrunkIu4Default::from_marker("gfx1151-zab-v1"), Some(zab));
        for other in ["", "gfx1151-zab-v2", "GFX1151-ZAB-V1", "gfx1151", "zab"] {
            assert_eq!(TrunkIu4Default::from_marker(other), None, "{other:?}");
        }
        assert_eq!(TRUNK_IU4_DEFAULT_KEY, "qwen4_trunk_iu4_default");
        let mask = Qwen4TrunkMask::parse(zab.mask_spec()).unwrap();
        for layer in [0, 1, 3, 47] {
            assert_eq!(mask.bits(layer), ZAB_BITS, "layer {layer}");
        }
        assert_eq!(mask.canonical(), "gdn.z,gdn.a,gdn.b");
    }

    #[test]
    fn q8f16_marker_gates_on_gfx1201_and_selects_no_iu4_route() {
        let q8 = TrunkIu4Default::Gfx1201Q8f16V1;
        assert_eq!(q8.marker(), "gfx1201-q8f16-v1");
        assert_eq!(q8.arch(), "gfx1201");
        assert_eq!(TrunkIu4Default::from_marker("gfx1201-q8f16-v1"), Some(q8));
        for other in ["gfx1201-q8f16-v2", "GFX1201-Q8F16-V1", "gfx1201", "q8f16"] {
            assert_eq!(TrunkIu4Default::from_marker(other), None, "{other:?}");
        }
        assert!(Qwen4TrunkMask::parse(q8.mask_spec()).unwrap().is_empty());
        let on = route(None, "gfx1201", Some(q8));
        assert_eq!(on.source, TrunkIu4MaskSource::ArtifactDefault(q8));
        assert!(on.mask.is_empty(), "the Q8F16 marker turns no IU4 route on");
        for arch in ["gfx1151", "gfx1100", "gfx1201x", ""] {
            assert_eq!(route(None, arch, Some(q8)).source, TrunkIu4MaskSource::Off);
        }
    }

    #[test]
    fn marked_artifact_defaults_to_zab_on_exact_gfx1151_only() {
        let on = route(None, "gfx1151", ZAB);
        assert_eq!(
            on.source,
            TrunkIu4MaskSource::ArtifactDefault(TrunkIu4Default::Gfx1151ZabV1)
        );
        assert!(!on.mask.is_empty());
        assert_eq!(on.mask.bits(9), ZAB_BITS);
        for arch in [
            "gfx1201", "gfx1100", "gfx1150", "gfx11", "gfx1151x", "GFX1151", "gfx942", "",
        ] {
            let off = route(None, arch, ZAB);
            assert_eq!(off.source, TrunkIu4MaskSource::Off, "{arch:?}");
            assert!(off.mask.is_empty(), "{arch:?}");
        }
    }

    #[test]
    fn unmarked_artifacts_stay_off_on_every_arch() {
        for arch in ["gfx1151", "gfx1201", "gfx1100", "gfx942", ""] {
            let off = route(None, arch, None);
            assert_eq!(off.source, TrunkIu4MaskSource::Off, "{arch:?}");
            assert!(off.mask.is_empty(), "{arch:?}");
            assert_eq!(off.mask.bits(0), 0);
        }
    }

    #[test]
    fn explicit_env_wins_over_the_artifact_default() {
        // `0` and empty disable A4 even on a marked artifact on gfx1151.
        for raw in ["0", "", "  "] {
            let off = route(Some(raw), "gfx1151", ZAB);
            assert_eq!(off.source, TrunkIu4MaskSource::Env, "{raw:?}");
            assert!(off.mask.is_empty(), "{raw:?}");
        }
        // Any other value replaces the default wholesale (existing grammar).
        let all = route(Some("1"), "gfx1151", ZAB);
        assert_eq!(all.source, TrunkIu4MaskSource::Env);
        assert_eq!(all.mask.bits(5), TRUNK_ALL_BITS);
        let one = route(Some("gdn.qkv@7"), "gfx1151", ZAB);
        assert_eq!(one.mask.bits(7), TrunkFamily::GdnQkv.bit());
        assert_eq!(one.mask.bits(8), 0);
        // The mask grammar is not arch gated here: the arch gate lives in
        // `Gpu::qwen4_trunk_iu4_requested`, exactly as for an unmarked artifact.
        let other = route(Some("1"), "gfx1201", ZAB);
        assert_eq!(other.source, TrunkIu4MaskSource::Env);
        assert_eq!(other.mask.bits(0), TRUNK_ALL_BITS);
    }

    #[test]
    fn env_behaves_as_before_for_unmarked_artifacts() {
        assert!(route(Some("0"), "gfx1151", None).mask.is_empty());
        assert_eq!(
            route(Some("all"), "gfx1151", None).mask.bits(0),
            TRUNK_ALL_BITS
        );
        assert_eq!(
            route(Some("1"), "gfx1100", None).source,
            TrunkIu4MaskSource::Env
        );
        assert_eq!(
            route(Some("gdn"), "gfx1151", None).mask.bits(3),
            TRUNK_GDN_BITS
        );
    }

    #[test]
    fn bad_env_is_an_error_for_marked_and_unmarked_artifacts() {
        for artifact in [None, ZAB] {
            for arch in ["gfx1151", "gfx1201"] {
                let error =
                    Qwen4TrunkMask::resolve_iu4(Some("gdn,qsa.x"), arch, artifact).unwrap_err();
                assert!(error.contains("HIPFIRE_QWEN4_TRUNK_IU4"), "{error}");
                assert!(error.contains("`qsa.x`"), "{error}");
            }
        }
    }
}
