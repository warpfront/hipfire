// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Admission vocabulary of the symmetric IU4 grouped routed-expert
//! prefill route (pure CPU, unit-tested). The route is on by default by
//! capability (`HIPFIRE_MOE_SYM_IU4=0` / `HIPFIRE_IU4_PREFILL=0` kill it).
//!
//! The route is a capability, never a model or card name: it is admitted from the
//! expert recipe and geometry ([`ExpertIu4Spec`]) plus the per-call state of the
//! prefill ([`ExpertIu4Inputs`]). Nothing here launches a kernel or reads a device;
//! the runtime builds the two structs from loader-verified headers, the shape plan
//! and the routing state, and calls [`expert_iu4_admission`]. A rejection names the
//! first failing requirement ([`ExpertIu4Reject::reason`]) so the fallback is
//! explicit and loggable, never a silent success.

use rdna_compute::DType;

/// Row floor of this route: one row. The A3B-class route has no faster non-IU4
/// alternative at any size (the qt53 F32 fallback is ~3.7x slower at prefill
/// sizes) and its tile-based GEMM kernels (16-row tiles, -1 padding, runtime
/// `K`/`n_experts`) are generic in the row count, so no floor is needed. This is
/// deliberately not the Flash-Next F16-WMMA crossover
/// (`rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS`), which keeps its own route.
pub const EXPERT_IU4_MIN_ROWS: usize = 1;

/// The numerical boundary of the route's expert arithmetic: exactly the plan's
/// "Target expert numerical boundary".
///
/// F32 normalized input is rotated (FWHT256) into the C2 A4 activation record;
/// the symmetric gate/up integer WMMA feeds BF16-RNE
/// `rt(silu(rt(g)) * rt(u))`; that activation is rotated again (FWHT128, C2 A4)
/// into the symmetric down integer WMMA, whose rows are BF16-RNE slots. Each slot
/// is widened to F32 and combined in the existing ascending top-k slot order
/// (`acc = fma(w, v, acc)` from `+0.0`, then `residual += acc`), with no extra
/// rounding of the combined residual and no atomics. This is a separate numerical
/// candidate, not bit-exact to an F16 incumbent route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertNumerics {
    /// BF16 slot boundaries, F32 ordered top-k combine.
    SymmetricBf16SlotsF32OrderedCombine,
}

/// Expert recipe and geometry of one MoE layer family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertIu4Spec {
    /// Routed experts per layer.
    pub experts: u32,
    /// Experts selected per token.
    pub top_k: u32,
    /// Model width: gate/up input `K`, down output `M`.
    pub hidden: u32,
    /// Routed expert width: gate/up output is `2 * intermediate` rows, down input `K`.
    pub intermediate: u32,
    /// Codec of every routed gate/up tensor.
    pub gate_up_codec: DType,
    /// Codec of every routed down tensor.
    pub down_codec: DType,
    /// Numerical boundary the route computes.
    pub numerics: ExpertNumerics,
}

/// Why the route is refused: a spec requirement or an admission requirement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertIu4Reject {
    // Spec requirements ([`ExpertIu4Spec::validate`]).
    /// Gate/up codec is not the 256-group symmetric-capable 4-bit codec.
    GateUpCodec,
    /// Down codec is not the 128-group 4-bit codec.
    DownCodec,
    /// `experts == 0`.
    NoExperts,
    /// `top_k` outside `1..=8` (the ordered combine's certified domain).
    TopK,
    /// `hidden == 0` or `intermediate == 0`.
    ZeroExtent,
    /// `hidden % 64 != 0`.
    HiddenNotMultipleOf64,
    /// `hidden % 256 != 0`.
    HiddenNotMultipleOf256,
    /// `(2 * intermediate) % 64 != 0`.
    GateUpRowsNotMultipleOf64,
    /// `intermediate % 128 != 0`.
    IntermediateNotMultipleOf128,
    /// A per-expert or per-layer extent product exceeds 32 bits.
    ExtentOverflow,
    // Admission requirements ([`expert_iu4_admission`]).
    /// The route was not requested.
    NotRequested,
    /// The loader did not verify symmetric headers on every routed tensor.
    HeadersNotVerified,
    /// The prefill is not on the grouped (Path 2) expert route.
    NotPath2,
    /// Fewer routing rows than the route's floor.
    BelowMinRows,
    /// Routed tensors carry per-expert dtype tags (mixed-tier experts).
    MixedTags,
    /// The down tensors carry AWQ scales.
    DownAwq,
}

impl ExpertIu4Reject {
    /// Every reject reason, spec requirements first, in check order.
    pub const ALL: [ExpertIu4Reject; 16] = [
        Self::GateUpCodec,
        Self::DownCodec,
        Self::NoExperts,
        Self::TopK,
        Self::ZeroExtent,
        Self::HiddenNotMultipleOf64,
        Self::HiddenNotMultipleOf256,
        Self::GateUpRowsNotMultipleOf64,
        Self::IntermediateNotMultipleOf128,
        Self::ExtentOverflow,
        Self::NotRequested,
        Self::HeadersNotVerified,
        Self::NotPath2,
        Self::BelowMinRows,
        Self::MixedTags,
        Self::DownAwq,
    ];

    /// Stable one-line reason for logs.
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::GateUpCodec => "routed gate/up codec is not MQ4G256V2",
            Self::DownCodec => "routed down codec is not MQ4G128V2",
            Self::NoExperts => "no routed experts",
            Self::TopK => "top_k is outside 1..=8",
            Self::ZeroExtent => "hidden or intermediate width is zero",
            Self::HiddenNotMultipleOf64 => "hidden width is not a multiple of 64",
            Self::HiddenNotMultipleOf256 => "hidden width is not a multiple of 256",
            Self::GateUpRowsNotMultipleOf64 => {
                "gate/up rows (2 * intermediate) are not a multiple of 64"
            }
            Self::IntermediateNotMultipleOf128 => "intermediate width is not a multiple of 128",
            Self::ExtentOverflow => "an expert extent product exceeds 32 bits",
            Self::NotRequested => "route not requested",
            Self::HeadersNotVerified => "routed headers are not loader-verified symmetric",
            Self::NotPath2 => "prefill is not on the grouped expert route",
            Self::BelowMinRows => "fewer rows than the route floor",
            Self::MixedTags => "routed tensors carry per-expert dtype tags",
            Self::DownAwq => "down tensors carry AWQ scales",
        }
    }
}

impl std::fmt::Display for ExpertIu4Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.reason())
    }
}

impl ExpertIu4Spec {
    /// Check recipe and geometry. Order (first failure wins):
    /// gate/up codec == `MQ4G256V2`; down codec == `MQ4G128V2`; `experts >= 1`;
    /// `top_k` in `1..=8`; `hidden > 0` and `intermediate > 0`;
    /// `hidden % 64 == 0`; `hidden % 256 == 0`; `(2 * intermediate) % 64 == 0`;
    /// `intermediate % 128 == 0`; the per-expert extents `2 * intermediate * hidden`
    /// and `hidden * intermediate` and the per-layer extents `experts * 2 *
    /// intermediate` and `experts * hidden` fit 32 bits.
    pub fn validate(&self) -> Result<(), ExpertIu4Reject> {
        use ExpertIu4Reject as R;
        if self.gate_up_codec != DType::MQ4G256V2 {
            return Err(R::GateUpCodec);
        }
        // Kernel format: the PM down entries read 68 B per K128 (`qwen4_moe_sym` `Kind::Down`
        // addressing, `emit_down_gp`) and the A4 producer is an R128 rotation; qt44 (136 B per K256,
        // FWHT256) down experts are served by the generic grouped W4A4 core in `execute_grouped_moe`.
        if self.down_codec != DType::MQ4G128V2 {
            return Err(R::DownCodec);
        }
        if self.experts == 0 {
            return Err(R::NoExperts);
        }
        if !(1..=8).contains(&self.top_k) {
            return Err(R::TopK);
        }
        if self.hidden == 0 || self.intermediate == 0 {
            return Err(R::ZeroExtent);
        }
        if !self.hidden.is_multiple_of(64) {
            return Err(R::HiddenNotMultipleOf64);
        }
        if !self.hidden.is_multiple_of(256) {
            return Err(R::HiddenNotMultipleOf256);
        }
        if !(2 * u64::from(self.intermediate)).is_multiple_of(64) {
            return Err(R::GateUpRowsNotMultipleOf64);
        }
        // Kernel format: the PM down K loop is whole K128 epochs with no pad lane in the down A4
        // producer (`moe_rotate_i4_pm` row width is K); a padded physical K is the generic core's.
        if !self.intermediate.is_multiple_of(128) {
            return Err(R::IntermediateNotMultipleOf128);
        }
        let (e, h, i) = (
            u64::from(self.experts),
            u64::from(self.hidden),
            u64::from(self.intermediate),
        );
        let fits = |product: u64| product <= u64::from(u32::MAX);
        if !(fits(2 * i * h) && fits(h * i) && fits(e * 2 * i) && fits(e * h)) {
            return Err(R::ExtentOverflow);
        }
        Ok(())
    }
}

/// Per-call state of one prefill expert stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertIu4Inputs {
    /// The route was requested (flag and device capability).
    pub requested: bool,
    /// The loader verified symmetric headers on every routed gate/up and down tensor.
    pub symmetric_headers_verified: bool,
    /// The prefill runs the grouped (Path 2) expert route.
    pub use_path2: bool,
    /// Routing rows of this prefill chunk.
    pub rows: usize,
    /// The route's row floor.
    pub min_rows: usize,
    /// Routed tensors carry per-expert dtype tags (mixed-tier experts).
    pub has_dtype_tags: bool,
    /// The down tensors carry AWQ scales.
    pub has_down_awq: bool,
    /// Graph capture, replay capture or recording is active. Informational: the
    /// route is admitted under hipGraph capture (Redline recording is refused by
    /// the pipeline, not here).
    pub capturing_graph: bool,
    /// Routed experts are host-mapped. Informational: the route is admitted; the
    /// PM GEMM launchers must be called with `host_mapped = true`.
    pub host_mapped: bool,
}

/// Pure admission of the route for one prefill expert stage. Order (first failure
/// wins): not requested; the spec's own requirements ([`ExpertIu4Spec::validate`]);
/// headers not verified; not Path 2; `rows < min_rows`; mixed dtype tags; down AWQ.
/// Hipgraph capture and host-mapped experts are admitted ([`ExpertIu4Inputs::capturing_graph`],
/// [`ExpertIu4Inputs::host_mapped`] are informational): the launcher sequence is
/// capture/replay-safe and the PM GEMMs have host-mapped (NT8) entries. The
/// caller must pass the real `host_mapped` to the launchers.
pub fn expert_iu4_admission(
    spec: &ExpertIu4Spec,
    inputs: &ExpertIu4Inputs,
) -> Result<(), ExpertIu4Reject> {
    use ExpertIu4Reject as R;
    if !inputs.requested {
        return Err(R::NotRequested);
    }
    spec.validate()?;
    if !inputs.symmetric_headers_verified {
        return Err(R::HeadersNotVerified);
    }
    if !inputs.use_path2 {
        return Err(R::NotPath2);
    }
    if inputs.rows < inputs.min_rows {
        return Err(R::BelowMinRows);
    }
    if inputs.has_dtype_tags {
        return Err(R::MixedTags);
    }
    // Kernel format: the PM down A4 producer (`moe_rotate_i4_pm`) has no AWQ divide input, so a
    // per-expert down AWQ vector cannot be folded into the activation record on this route.
    if inputs.has_down_awq {
        return Err(R::DownAwq);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn spec(experts: u32, top_k: u32, hidden: u32, intermediate: u32) -> ExpertIu4Spec {
        ExpertIu4Spec {
            experts,
            top_k,
            hidden,
            intermediate,
            gate_up_codec: DType::MQ4G256V2,
            down_codec: DType::MQ4G128V2,
            numerics: ExpertNumerics::SymmetricBf16SlotsF32OrderedCombine,
        }
    }

    /// 256 experts, top-8, H2048, I512.
    fn narrow() -> ExpertIu4Spec {
        spec(256, 8, 2048, 512)
    }

    /// 512 experts, H2560, I640 (top-8: the ordered combine's ceiling).
    fn wide() -> ExpertIu4Spec {
        spec(512, 8, 2560, 640)
    }

    fn admitted() -> ExpertIu4Inputs {
        ExpertIu4Inputs {
            requested: true,
            symmetric_headers_verified: true,
            use_path2: true,
            rows: 4096,
            min_rows: EXPERT_IU4_MIN_ROWS,
            has_dtype_tags: false,
            has_down_awq: false,
            capturing_graph: false,
            host_mapped: false,
        }
    }

    #[test]
    fn both_geometries_validate() {
        assert_eq!(narrow().validate(), Ok(()));
        assert_eq!(wide().validate(), Ok(()));
        // Every certified top_k.
        for k in 1..=8 {
            assert_eq!(spec(256, k, 2048, 512).validate(), Ok(()), "{k}");
        }
    }

    #[test]
    fn spec_rejects_each_requirement() {
        use ExpertIu4Reject as R;
        let mut s = narrow();
        s.gate_up_codec = DType::MQ4G128V2;
        assert_eq!(s.validate(), Err(R::GateUpCodec));
        let mut s = narrow();
        s.down_codec = DType::MQ4G256V2;
        assert_eq!(s.validate(), Err(R::DownCodec));
        assert_eq!(spec(0, 8, 2048, 512).validate(), Err(R::NoExperts));
        assert_eq!(spec(256, 0, 2048, 512).validate(), Err(R::TopK));
        assert_eq!(spec(256, 9, 2048, 512).validate(), Err(R::TopK));
        assert_eq!(spec(256, 8, 0, 512).validate(), Err(R::ZeroExtent));
        assert_eq!(spec(256, 8, 2048, 0).validate(), Err(R::ZeroExtent));
        assert_eq!(
            spec(256, 8, 2080, 512).validate(),
            Err(R::HiddenNotMultipleOf64)
        );
        assert_eq!(
            spec(256, 8, 2112, 512).validate(),
            Err(R::HiddenNotMultipleOf256)
        );
        assert_eq!(
            spec(256, 8, 2048, 16).validate(),
            Err(R::GateUpRowsNotMultipleOf64)
        );
        assert_eq!(
            spec(256, 8, 2048, 96).validate(),
            Err(R::IntermediateNotMultipleOf128)
        );
        // 4096 * 2 * 2^20 elements per expert overflows 32 bits.
        assert_eq!(
            spec(2, 8, 1 << 20, 1 << 11).validate(),
            Err(R::ExtentOverflow)
        );
        // Per-layer extent: experts * 2 * intermediate.
        assert_eq!(
            spec(1 << 24, 8, 256, 256).validate(),
            Err(R::ExtentOverflow)
        );
    }

    #[test]
    fn spec_check_order_is_codec_then_shape() {
        use ExpertIu4Reject as R;
        let mut s = spec(0, 0, 0, 0);
        s.gate_up_codec = DType::MQ4G128V2;
        s.down_codec = DType::MQ4G256V2;
        assert_eq!(s.validate(), Err(R::GateUpCodec));
        s.gate_up_codec = DType::MQ4G256V2;
        assert_eq!(s.validate(), Err(R::DownCodec));
        s.down_codec = DType::MQ4G128V2;
        assert_eq!(s.validate(), Err(R::NoExperts));
        s.experts = 1;
        assert_eq!(s.validate(), Err(R::TopK));
        s.top_k = 1;
        assert_eq!(s.validate(), Err(R::ZeroExtent));
    }

    #[test]
    fn admission_accepts_the_verified_resident_path2_prefill() {
        for s in [narrow(), wide()] {
            assert_eq!(expert_iu4_admission(&s, &admitted()), Ok(()));
        }
        // Exactly at the floor (one row).
        let at_floor = ExpertIu4Inputs {
            rows: EXPERT_IU4_MIN_ROWS,
            ..admitted()
        };
        assert_eq!(EXPERT_IU4_MIN_ROWS, 1);
        assert_eq!(expert_iu4_admission(&narrow(), &at_floor), Ok(()));
    }

    #[test]
    fn admission_admits_graph_capture_and_host_mapped_experts() {
        for s in [narrow(), wide()] {
            for (capturing_graph, host_mapped) in [(true, false), (false, true), (true, true)] {
                let inputs = ExpertIu4Inputs {
                    capturing_graph,
                    host_mapped,
                    ..admitted()
                };
                assert_eq!(expert_iu4_admission(&s, &inputs), Ok(()));
            }
        }
    }

    #[test]
    fn admission_rejects_each_requirement() {
        use ExpertIu4Reject as R;
        let cases: [(ExpertIu4Inputs, R); 6] = [
            (
                ExpertIu4Inputs {
                    requested: false,
                    ..admitted()
                },
                R::NotRequested,
            ),
            (
                ExpertIu4Inputs {
                    symmetric_headers_verified: false,
                    ..admitted()
                },
                R::HeadersNotVerified,
            ),
            (
                ExpertIu4Inputs {
                    use_path2: false,
                    ..admitted()
                },
                R::NotPath2,
            ),
            (
                ExpertIu4Inputs {
                    rows: 0,
                    ..admitted()
                },
                R::BelowMinRows,
            ),
            (
                ExpertIu4Inputs {
                    has_dtype_tags: true,
                    ..admitted()
                },
                R::MixedTags,
            ),
            (
                ExpertIu4Inputs {
                    has_down_awq: true,
                    ..admitted()
                },
                R::DownAwq,
            ),
        ];
        for (inputs, want) in cases {
            assert_eq!(expert_iu4_admission(&narrow(), &inputs), Err(want));
        }
        // A spec failure surfaces through the admission.
        assert_eq!(
            expert_iu4_admission(&spec(256, 10, 2048, 512), &admitted()),
            Err(R::TopK)
        );
    }

    #[test]
    fn admission_check_order_is_the_documented_one() {
        use ExpertIu4Reject as R;
        // Everything wrong at once; clear one requirement at a time in documented order.
        let mut s = spec(256, 10, 2048, 512);
        let mut i = ExpertIu4Inputs {
            requested: false,
            symmetric_headers_verified: false,
            use_path2: false,
            rows: 0,
            min_rows: EXPERT_IU4_MIN_ROWS,
            has_dtype_tags: true,
            has_down_awq: true,
            capturing_graph: true,
            host_mapped: true,
        };
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::NotRequested));
        i.requested = true;
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::TopK));
        s.top_k = 8;
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::HeadersNotVerified));
        i.symmetric_headers_verified = true;
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::NotPath2));
        i.use_path2 = true;
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::BelowMinRows));
        i.rows = EXPERT_IU4_MIN_ROWS;
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::MixedTags));
        i.has_dtype_tags = false;
        assert_eq!(expert_iu4_admission(&s, &i), Err(R::DownAwq));
        i.has_down_awq = false;
        // Graph capture and host-mapped experts never refuse the route.
        assert_eq!(expert_iu4_admission(&s, &i), Ok(()));
    }

    #[test]
    fn every_reject_reason_is_reachable_and_distinct() {
        use ExpertIu4Reject as R;
        let mut seen = BTreeSet::new();
        let mut record = |r: Result<(), R>| {
            seen.insert(r.unwrap_err() as u8);
        };
        let mut s = narrow();
        s.gate_up_codec = DType::MQ4G128V2;
        record(s.validate());
        let mut s = narrow();
        s.down_codec = DType::MQ4G256V2;
        record(s.validate());
        record(spec(0, 8, 2048, 512).validate());
        record(spec(256, 9, 2048, 512).validate());
        record(spec(256, 8, 0, 512).validate());
        record(spec(256, 8, 2080, 512).validate());
        record(spec(256, 8, 2112, 512).validate());
        record(spec(256, 8, 2048, 16).validate());
        record(spec(256, 8, 2048, 96).validate());
        record(spec(2, 8, 1 << 20, 1 << 11).validate());
        for inputs in [
            ExpertIu4Inputs {
                requested: false,
                ..admitted()
            },
            ExpertIu4Inputs {
                symmetric_headers_verified: false,
                ..admitted()
            },
            ExpertIu4Inputs {
                use_path2: false,
                ..admitted()
            },
            ExpertIu4Inputs {
                rows: 0,
                ..admitted()
            },
            ExpertIu4Inputs {
                has_dtype_tags: true,
                ..admitted()
            },
            ExpertIu4Inputs {
                has_down_awq: true,
                ..admitted()
            },
        ] {
            record(expert_iu4_admission(&narrow(), &inputs));
        }
        let all: BTreeSet<u8> = R::ALL.iter().map(|r| *r as u8).collect();
        assert_eq!(all.len(), R::ALL.len(), "ALL lists a reason twice");
        assert_eq!(
            seen, all,
            "a reject reason is unreachable or missing from ALL"
        );
        let reasons: BTreeSet<&str> = R::ALL.iter().map(R::reason).collect();
        assert_eq!(reasons.len(), R::ALL.len(), "two reasons share a message");
        for r in R::ALL {
            assert_eq!(r.to_string(), r.reason());
        }
    }

    #[test]
    fn reasons_name_requirements_never_models_or_cards() {
        for r in ExpertIu4Reject::ALL {
            let text = r.reason().to_ascii_lowercase();
            for banned in [
                "qwen", "ornith", "a3b", "flash", "gemma", "llama", "gfx", "r9700", "mi300",
                "radeon", "strix",
            ] {
                assert!(!text.contains(banned), "{r:?}: {text:?} names {banned}");
            }
        }
    }
}
