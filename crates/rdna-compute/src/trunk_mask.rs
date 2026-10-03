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
use std::sync::LazyLock;

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
            names.extend(TrunkFamily::ALL.iter().filter(|f| bits & f.bit() != 0).map(|f| f.name()));
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

fn parse_env(name: &str) -> Result<Qwen4TrunkMask, String> {
    match hipfire_config::developer_var(name) {
        Ok(raw) => Qwen4TrunkMask::parse(&raw).map_err(|e| format!("{name}={raw:?}: {e}")),
        Err(_) => Ok(Qwen4TrunkMask::default()),
    }
}

static TRUNK_IU4_MASK: LazyLock<Result<Qwen4TrunkMask, String>> =
    LazyLock::new(|| parse_env("HIPFIRE_QWEN4_TRUNK_IU4"));

/// `HIPFIRE_QWEN4_TRUNK_IU4` parsed once; the error names the bad token.
pub fn qwen4_trunk_iu4_mask() -> Result<&'static Qwen4TrunkMask, &'static str> {
    TRUNK_IU4_MASK.as_ref().map_err(String::as_str)
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
        assert_eq!(layers(&mask, TrunkFamily::QsaO), (3..=15).collect::<Vec<_>>());
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
        let mask = Qwen4TrunkMask::parse("gdn.qkv@0-3,gdn.qkv@4-9,gdn.z@0-9,gdn.a@0-9,gdn.b@0-9,gdn.out@0-9,qsa.idx").unwrap();
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
        for raw in ["gdn.z@+3", "gdn.z@+3-9", "gdn.z@3-+9", "gdn.z@-3", "gdn.z@3--9"] {
            assert!(Qwen4TrunkMask::parse(raw).is_err(), "{raw}");
        }
        assert!(Qwen4TrunkMask::parse("gdn.z@3-9").is_ok());
    }

    #[test]
    fn check_layers_rejects_bands_outside_the_trunk() {
        let trunk = 48;
        for raw in ["all", "1", "0", "gdn", "gdn.qkv@0-47", "qsa.o@47", "gdn,qsa.o@3-15"] {
            Qwen4TrunkMask::parse(raw).unwrap().check_layers(trunk).unwrap_or_else(|e| {
                panic!("{raw}: {e}");
            });
        }
        for (raw, token) in [
            ("gdn.qkv@48", "gdn.qkv@48"),
            ("gdn,qsa.o@40-48", "qsa.o@40-48"),
            ("gdn.z@0-47,gdn.a@50-60", "gdn.a@50-60"),
            ("qsa.k@100", "qsa.k@100"),
        ] {
            let error = Qwen4TrunkMask::parse(raw).unwrap().check_layers(trunk).unwrap_err();
            assert!(error.contains(&format!("`{token}`")), "{raw}: {error}");
        }
    }
}
