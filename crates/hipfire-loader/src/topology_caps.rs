// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Which multi-GPU axes a [`Carrier`](crate::Carrier) declares.
//!
//! A carrier states what it can do (`topology.pp`, `topology.ep`, `--tp N`);
//! the admission front door refuses every other request with the one uniform
//! sentence ([`hipfire_runtime::loader_api::topology_refusal`]) and the
//! bench/serve routes read the same declaration. A new architecture opts into
//! an axis by returning it from [`Carrier::topology_caps`](crate::Carrier::topology_caps);
//! nothing else in the loader names an arch id for multi-card.

use hipfire_runtime::loader_api::{topology_refusal, TopologyAxis};

/// Which executor drives a declared pipeline (`topology.pp`) route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpExecutor {
    /// The Qwen3.5 hand-rolled per-layer loop (`boundary_copy` between
    /// `Gpus::init_layers` devices); layer counts come from the Qwen config.
    Legacy,
    /// The generic mesh executor: admitted ranks bind `OutputCardLast`, bands
    /// come from `BandPlacement`, edges from `provision_mesh_edges`.
    Mesh {
        /// Numerical-policy identity of the route (every band edge executes
        /// the carrier's cut-safe schedule).
        numerical_policy: &'static str,
        /// Minimum layers the first band must hold (an arch whose embed side
        /// state lives with the first layers; 1 when there is no such pin).
        first_band_min_layers: usize,
    },
}

/// Which bench entrypoint family drives a loaded pipeline-parallel model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PpBenchRoute {
    /// Qwen3.5 pipeline parallelism (`forward_prefill_batch_multi`,
    /// `forward_scratch_multi`).
    Qwen35Pp,
    /// Qwen4 mesh pipeline parallelism (`pp_prefill_final`,
    /// `pp_forward_token_or_argmax`).
    Qwen4Pp,
    /// The carrier's own bench hooks drive the bundle (they fan out over its
    /// bands); the mesh quiesce syncs every `pp_gpus` device.
    Carrier,
}

/// A carrier's pipeline-parallel capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PpCaps {
    /// Whether a safetensors model directory can be banded (false: HFQ only).
    pub dir_sources: bool,
    /// Largest `topology.pp` the route admits.
    pub max_degree: usize,
    pub executor: PpExecutor,
    pub bench: PpBenchRoute,
}

/// A carrier's expert-parallel (`topology.ep`) capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpCaps {
    /// Developer environment variable that forces `topology.ep` to 1 when set
    /// to `0` (Qwen4: `HIPFIRE_QWEN4_EP`), if the carrier has one.
    pub kill_switch_env: Option<&'static str>,
}

/// A carrier's `--tp N` capability: tensor parallelism for a dense model,
/// expert parallelism (`Carrier::tp_mesh_axis` picks the axis) for a MoE one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TpCaps {
    /// Whether a safetensors model directory can be sharded (false: HFQ only).
    pub dir_sources: bool,
}

/// The axes a carrier declares. The default declares nothing: every
/// multi-card request on such a carrier is refused at admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TopologyCaps {
    pub pp: Option<PpCaps>,
    pub ep: Option<EpCaps>,
    pub tp: Option<TpCaps>,
}

impl TopologyCaps {
    /// Whether this declaration covers `axis`.
    pub fn declares(&self, axis: TopologyAxis) -> bool {
        match axis {
            TopologyAxis::Pipeline => self.pp.is_some(),
            TopologyAxis::Expert => self.ep.is_some(),
            TopologyAxis::Tp => self.tp.is_some(),
        }
    }

    /// The `topology.ep` kill-switch variable the declaration names, if any.
    pub fn ep_kill_switch_env(&self) -> Option<&'static str> {
        self.ep.and_then(|ep| ep.kill_switch_env)
    }
}

/// The front door's one axis verdict. Every requested axis (degree > 1) the
/// carrier does not declare is refused with the uniform
/// [`topology_refusal`] sentence, `ep` then `tp` then `pp`; then a
/// safetensors directory is refused on a declared `tp`/`pp` axis that cannot
/// shard/band one. Pure: no source, device or environment access.
pub fn topology_axes_refusal(
    family: &str,
    caps: &TopologyCaps,
    is_dir: bool,
    pp: usize,
    tp: usize,
    ep: usize,
) -> Result<(), String> {
    for (axis, degree) in [
        (TopologyAxis::Expert, ep),
        (TopologyAxis::Tp, tp),
        (TopologyAxis::Pipeline, pp),
    ] {
        if degree > 1 && !caps.declares(axis) {
            return Err(topology_refusal(family, axis, degree, is_dir));
        }
    }
    if tp > 1 && is_dir && caps.tp.is_some_and(|c| !c.dir_sources) {
        return Err(TP_DIR_REFUSAL.to_string());
    }
    if pp > 1 && is_dir && caps.pp.is_some_and(|c| !c.dir_sources) {
        return Err(topology_refusal(family, TopologyAxis::Pipeline, pp, true));
    }
    Ok(())
}

/// A declared pipeline route's degree bound.
pub fn pp_degree_refusal(family: &str, caps: &TopologyCaps, pp: usize) -> Result<(), String> {
    match caps.pp {
        Some(pp_caps) if pp == 0 || pp > pp_caps.max_degree => Err(format!(
            "{family}: topology.pp = {pp} is outside the admitted 1..={}",
            pp_caps.max_degree
        )),
        _ => Ok(()),
    }
}

/// `--tp N` on a safetensors directory for a carrier whose tp route is
/// HFQ-only.
pub const TP_DIR_REFUSAL: &str = "expert/tensor parallelism (tp>1) is not supported for safetensors directory sources; load a single HFQ file or set tp=1";

#[cfg(test)]
mod tests {
    use super::*;

    const MESH: PpCaps = PpCaps {
        dir_sources: false,
        max_degree: 8,
        executor: PpExecutor::Mesh {
            numerical_policy: "test-v1",
            first_band_min_layers: 1,
        },
        bench: PpBenchRoute::Carrier,
    };

    #[test]
    fn a_carrier_with_no_caps_gets_exactly_the_uniform_sentence_per_axis() {
        let none = TopologyCaps::default();
        for is_dir in [false, true] {
            assert_eq!(topology_axes_refusal("fam", &none, is_dir, 1, 1, 1), Ok(()));
            assert_eq!(
                topology_axes_refusal("fam", &none, is_dir, 1, 1, 2),
                Err(topology_refusal("fam", TopologyAxis::Expert, 2, is_dir))
            );
            assert_eq!(
                topology_axes_refusal("fam", &none, is_dir, 1, 3, 1),
                Err(topology_refusal("fam", TopologyAxis::Tp, 3, is_dir))
            );
            assert_eq!(
                topology_axes_refusal("fam", &none, is_dir, 4, 1, 1),
                Err(topology_refusal("fam", TopologyAxis::Pipeline, 4, is_dir))
            );
        }
    }

    #[test]
    fn undeclared_axes_refuse_ep_then_tp_then_pp() {
        let none = TopologyCaps::default();
        assert_eq!(
            topology_axes_refusal("fam", &none, false, 2, 2, 2),
            Err(topology_refusal("fam", TopologyAxis::Expert, 2, false))
        );
        assert_eq!(
            topology_axes_refusal("fam", &none, false, 2, 2, 1),
            Err(topology_refusal("fam", TopologyAxis::Tp, 2, false))
        );
        let ep_only = TopologyCaps {
            ep: Some(EpCaps {
                kill_switch_env: None,
            }),
            ..TopologyCaps::default()
        };
        assert_eq!(
            topology_axes_refusal("fam", &ep_only, false, 2, 2, 2),
            Err(topology_refusal("fam", TopologyAxis::Tp, 2, false))
        );
    }

    #[test]
    fn declared_axes_pass_and_directories_are_refused_where_hfq_only() {
        let caps = TopologyCaps {
            pp: Some(MESH),
            ep: None,
            tp: Some(TpCaps { dir_sources: false }),
        };
        assert_eq!(topology_axes_refusal("fam", &caps, false, 2, 2, 1), Ok(()));
        assert_eq!(
            topology_axes_refusal("fam", &caps, true, 1, 2, 1),
            Err(TP_DIR_REFUSAL.to_string())
        );
        assert_eq!(
            topology_axes_refusal("fam", &caps, true, 2, 1, 1),
            Err(topology_refusal("fam", TopologyAxis::Pipeline, 2, true))
        );
        let dirs = TopologyCaps {
            pp: Some(PpCaps {
                dir_sources: true,
                ..MESH
            }),
            tp: Some(TpCaps { dir_sources: true }),
            ep: None,
        };
        assert_eq!(topology_axes_refusal("fam", &dirs, true, 2, 2, 1), Ok(()));
    }

    #[test]
    fn degree_bound_names_the_admitted_range() {
        let caps = TopologyCaps {
            pp: Some(MESH),
            ..TopologyCaps::default()
        };
        assert_eq!(pp_degree_refusal("fam", &caps, 8), Ok(()));
        assert_eq!(
            pp_degree_refusal("fam", &caps, 9),
            Err("fam: topology.pp = 9 is outside the admitted 1..=8".to_string())
        );
        assert_eq!(
            pp_degree_refusal("fam", &TopologyCaps::default(), 99),
            Ok(())
        );
    }
}
