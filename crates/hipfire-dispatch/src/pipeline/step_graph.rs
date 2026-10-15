// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//! GPU-free owned description of a step program (mesh splitter contract §1).
//!
//! A [`StepGraph`] is a plain-data record of a program in issue order: the
//! resources it touches ([`ResourceDecl`]) and the ops that read/write them
//! ([`OpDecl`]), each op carrying a [`SplitRule`] that tells a later stage
//! (`hipfire-runtime::mesh_split`) how the op may be partitioned across a
//! device mesh.
//!
//! Everything here is pure: no `Gpu`, no `GpuTensor`, no HIP, no file I/O, no
//! env reads. [`OpId`] and [`ResourceId`] are dense `u32` indices (the id equals
//! the position in [`StepGraph::ops`] / [`StepGraph::resources`]) and are the
//! stable handles every later stage (plan, transport ledger) uses.
//!
//! # Frozen op-kind IDs
//!
//! [`OpKind`] discriminants are part of the contract: one per [`Step`] variant
//! (`1..`) plus hand-route ops (`1000..`); a hand-route op that later gains a
//! `Step` variant keeps its ID (`Step::SiluMul` is `SiluMul = 1001`). Values
//! are frozen — never renumber, never reuse, append only. The
//! `op_kind_ids_frozen` test pins every value.

use super::steps::Step;

/// Dense op handle: the index of the op in [`StepGraph::ops`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct OpId(pub u32);

/// Dense resource handle: the index of the resource in [`StepGraph::resources`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ResourceId(pub u32);

/// Stable op-kind IDs. One per [`Step`] variant; values are frozen (never
/// renumber, append only).
#[repr(u16)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpKind {
    /// [`Step::Gemv`].
    Gemv = 1,
    /// [`Step::GemvResidual`].
    GemvResidual = 2,
    /// [`Step::RmsnormAutomatic`].
    RmsnormAutomatic = 3,
    /// [`Step::Attend`].
    Attend = 4,
    /// [`Step::Rope`].
    Rope = 5,
    /// [`Step::QkNorm`].
    QkNorm = 6,
    /// [`Step::BiasAdd`].
    BiasAdd = 7,
    /// [`Step::HyperRead`].
    HyperRead = 8,
    /// [`Step::GatedDeltaNet`].
    GatedDeltaNet = 9,
    /// [`Step::IndexedAttention`].
    IndexedAttention = 10,
    /// [`Step::HyperWrite`].
    HyperWrite = 11,
    /// [`Step::GroupedDepthwise`].
    GroupedDepthwise = 12,
    /// [`Step::Clear`].
    Clear = 13,
    /// [`Step::Embed`].
    Embed = 14,
    /// [`Step::HyperNorm`].
    HyperNorm = 15,
    /// [`Step::Project`].
    Project = 16,
    /// [`Step::BroadcastAdd`].
    BroadcastAdd = 17,
    /// [`Step::DenseFfn`].
    DenseFfn = 18,
    /// [`Step::SandwichAttention`].
    SandwichAttention = 19,
    /// [`Step::SandwichMlp`].
    SandwichMlp = 20,
    /// [`Step::ParallelMoeMlp`].
    ParallelMoeMlp = 21,
    /// [`Step::PerLayerInput`].
    PerLayerInput = 22,
    /// [`Step::Scale`].
    Scale = 23,
    /// [`Step::Softcap`].
    Softcap = 24,
    /// [`Step::Moe`].
    Moe = 25,
    /// [`Step::MoeStage`].
    MoeStage = 26,
    /// [`Step::GatedShortConv`] (#666 G10, LFM2).
    GatedShortConv = 27,
    /// [`Step::HcPre`] (#666 G9, DeepSeek4).
    HcPre = 28,
    /// [`Step::HcPost`] (#666 G9, DeepSeek4).
    HcPost = 29,
    /// [`Step::HcHead`] (#666 G9, DeepSeek4).
    HcHead = 30,
    /// [`Step::QLoraPrepare`] (#666 G9, DeepSeek4).
    QLoraPrepare = 31,
    /// [`Step::PackedProjections`] (#666 G9, DeepSeek4).
    PackedProjections = 32,
    /// [`Step::QLoraProject`] (#666 G9, DeepSeek4).
    QLoraProject = 33,
    /// [`Step::KvJoint`] (#666 G9, DeepSeek4).
    KvJoint = 34,
    /// [`Step::ApplyTailRope`] (#666 G9, DeepSeek4).
    ApplyTailRope = 35,
    /// [`Step::ExpertFfn`] (#666 G9, DeepSeek4).
    ExpertFfn = 36,
    /// [`Step::SparseIndex`] (#666 G9, DeepSeek4).
    SparseIndex = 37,
    /// [`Step::KvCompress`] (#666 G9, DeepSeek4).
    KvCompress = 38,
    /// [`Step::PreparedProjection`] (#666 G9, DeepSeek4).
    PreparedProjection = 39,
    /// [`Step::SwaRingWrite`] (#666 G9, DeepSeek4).
    SwaRingWrite = 40,
    /// [`Step::IdentityGather`] (#666 G9, DeepSeek4).
    IdentityGather = 41,
    /// [`Step::SwaAttention`] (#666 G9, DeepSeek4).
    SwaAttention = 42,
    /// [`Step::SparseAttention`] (#666 G9, DeepSeek4).
    SparseAttention = 43,
    /// [`Step::OLora`] (#666 G9, DeepSeek4).
    OLora = 44,
    /// [`Step::CompressorProject`] (#666 G9, DeepSeek4).
    CompressorProject = 45,
    /// [`Step::EmbedStreams`] (#666 G9 slice 2, DeepSeek4 rows).
    EmbedStreams = 46,
    /// [`Step::StreamMeanCapture`] (#666 G9 slice 2, DeepSeek4 rows).
    StreamMeanCapture = 47,
    /// [`Step::RingKv`] (#666 G9 slice 2, DeepSeek4 rows).
    RingKv = 48,
    /// [`Step::BlockStage`] (#666 G9 slice 2, DeepSeek4 rows).
    BlockStage = 49,
    /// [`Step::SwaStage`] (#666 G9 slice 2, DeepSeek4 rows).
    SwaStage = 50,
    /// Hand-route ops with no `Step` variant yet (dense TP decode): residual add.
    ResidualAdd = 1000,
    /// [`Step::SiluMul`]; also the hand-route SiLU-mul.
    SiluMul = 1001,
    /// Hand-route op with no `Step` variant yet: logits head argmax.
    Argmax = 1002,
}

/// Map a [`Step`] to its frozen [`OpKind`]. Exhaustive: adding a `Step`
/// variant is a compile error here until it is given a new (appended) ID.
pub fn op_kind_of(step: &Step<'_>) -> OpKind {
    match step {
        Step::Gemv { .. } => OpKind::Gemv,
        Step::GemvResidual { .. } => OpKind::GemvResidual,
        Step::RmsnormAutomatic { .. } => OpKind::RmsnormAutomatic,
        Step::Attend { .. } => OpKind::Attend,
        Step::Rope { .. } => OpKind::Rope,
        Step::QkNorm { .. } => OpKind::QkNorm,
        Step::BiasAdd { .. } => OpKind::BiasAdd,
        Step::HyperRead(..) => OpKind::HyperRead,
        Step::GatedDeltaNet(..) => OpKind::GatedDeltaNet,
        Step::IndexedAttention(..) => OpKind::IndexedAttention,
        Step::HyperWrite(..) => OpKind::HyperWrite,
        Step::GroupedDepthwise(..) => OpKind::GroupedDepthwise,
        Step::Clear(..) => OpKind::Clear,
        Step::Embed(..) => OpKind::Embed,
        Step::HyperNorm(..) => OpKind::HyperNorm,
        Step::Project(..) => OpKind::Project,
        Step::BroadcastAdd(..) => OpKind::BroadcastAdd,
        Step::DenseFfn(..) => OpKind::DenseFfn,
        Step::SandwichAttention(..) => OpKind::SandwichAttention,
        Step::SandwichMlp(..) => OpKind::SandwichMlp,
        Step::ParallelMoeMlp(..) => OpKind::ParallelMoeMlp,
        Step::PerLayerInput(..) => OpKind::PerLayerInput,
        Step::Scale(..) => OpKind::Scale,
        Step::Softcap(..) => OpKind::Softcap,
        Step::Moe(..) => OpKind::Moe,
        Step::MoeStage(..) => OpKind::MoeStage,
        Step::GatedShortConv(..) => OpKind::GatedShortConv,
        Step::SiluMul { .. } => OpKind::SiluMul,
        Step::HcPre(..) => OpKind::HcPre,
        Step::HcPost(..) => OpKind::HcPost,
        Step::HcHead(..) => OpKind::HcHead,
        Step::QLoraPrepare(..) => OpKind::QLoraPrepare,
        Step::PackedProjections(..) => OpKind::PackedProjections,
        Step::QLoraProject(..) => OpKind::QLoraProject,
        Step::KvJoint(..) => OpKind::KvJoint,
        Step::ApplyTailRope(..) => OpKind::ApplyTailRope,
        Step::ExpertFfn(..) => OpKind::ExpertFfn,
        Step::SparseIndex(..) => OpKind::SparseIndex,
        Step::KvCompress(..) => OpKind::KvCompress,
        Step::PreparedProjection(..) => OpKind::PreparedProjection,
        Step::SwaRingWrite(..) => OpKind::SwaRingWrite,
        Step::IdentityGather(..) => OpKind::IdentityGather,
        Step::SwaAttention(..) => OpKind::SwaAttention,
        Step::SparseAttention(..) => OpKind::SparseAttention,
        Step::OLora(..) => OpKind::OLora,
        Step::EmbedStreams(..) => OpKind::EmbedStreams,
        Step::StreamMeanCapture(..) => OpKind::StreamMeanCapture,
        Step::RingKv(..) => OpKind::RingKv,
        Step::BlockStage(..) => OpKind::BlockStage,
        Step::SwaStage(..) => OpKind::SwaStage,
        Step::CompressorProject(..) => OpKind::CompressorProject,
    }
}

/// Class of persistent state a [`ResourceClass::State`] resource holds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StateClass {
    /// Attention KV cache.
    Kv,
    /// Recurrent (e.g. DeltaNet) state.
    Recurrent,
    /// Convolution state.
    Conv,
}

/// What a resource is, for placement/sharding purposes.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ResourceClass {
    /// Per-step scratch activation (residual, q/k/v, partials, normed input ...).
    Activation,
    /// Manifest weight identity; `group` = packed-quant block (elements) along
    /// the policy's split axis (1 = unpacked).
    Weight {
        /// Manifest weight name.
        name: String,
        /// Layer index, `None` for non-layer weights.
        layer: Option<usize>,
        /// Packed-quant block (elements) along the split axis; 1 = unpacked.
        group: usize,
    },
    /// Persistent state; must match one `ManifestPlan.state` entry by
    /// (class, layer) (KV quant string ignored).
    State {
        /// State class.
        class: StateClass,
        /// Layer index.
        layer: usize,
    },
    /// Host-fed per-step input (token id, position).
    Input,
    /// Program result (logits).
    Output,
}

/// Element-offset view of an earlier resource.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Alias {
    /// Aliased resource; MUST be `<` the aliasing resource's id.
    pub of: ResourceId,
    /// Element offset into `of`.
    pub offset: usize,
}

/// Declaration of one resource.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ResourceDecl {
    /// Resource class.
    pub class: ResourceClass,
    /// Logical shape (elements per dim).
    pub shape: Vec<usize>,
    /// Optional alias view of an earlier resource.
    pub alias: Option<Alias>,
}

/// How an op may be split across a mesh group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitRule {
    /// `output = W · input` (overwrite). Layouts follow W's manifest
    /// `ShardPolicy`.
    Project {
        /// Weight resource (must be a [`ResourceClass::Weight`]).
        weight: ResourceId,
        /// Input activation.
        input: ResourceId,
        /// Output activation.
        output: ResourceId,
    },
    /// `dst += src`. dst must be Whole. src Whole → local add; src Partial →
    /// this op becomes ONE fused AllReduceAdd collective (today's
    /// `x += p0 + p1`); src Split → NeedsGather.
    Accumulate {
        /// Accumulation target.
        dst: ResourceId,
        /// Addend.
        src: ResourceId,
    },
    /// Elementwise over same-shape activation operands: all share one
    /// non-Partial layout.
    Pointwise,
    /// Needs the complete row of every activation operand (norms, argmax):
    /// all Whole.
    RowComplete,
    /// Independent per head (attention, DeltaNet, RoPE, qk-norm): activation
    /// operands all Whole or all head-Split; state operands take the same
    /// layout; requires `heads % parts == 0` and `kv_heads % parts == 0`.
    HeadLocal {
        /// Query heads.
        heads: usize,
        /// KV heads.
        kv_heads: usize,
    },
    /// No declared split point (compound sandwich/MoE/DenseFfn): every operand
    /// must be Whole.
    Opaque,
}

/// Declaration of one op in program order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct OpDecl {
    /// Frozen op kind.
    pub kind: OpKind,
    /// Layer index, `None` for non-layer ops.
    pub layer: Option<usize>,
    /// Split rule.
    pub rule: SplitRule,
    /// Resources read.
    pub reads: Vec<ResourceId>,
    /// Resources written. Effects; Project/Accumulate operands must appear in
    /// `reads` ∪ `writes`.
    pub writes: Vec<ResourceId>,
}

/// Refusals from the [`StepGraph`] builder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepGraphError {
    /// An op references a resource id that does not exist.
    UnknownResource(ResourceId),
    /// Alias target is not strictly earlier than the aliasing resource.
    AliasForward {
        /// The aliasing resource (the id it would have been given).
        resource: ResourceId,
        /// The offending alias target.
        of: ResourceId,
    },
    /// Alias `offset + numel` exceeds the target's element count (or the
    /// element count overflowed `usize`).
    AliasOutOfRange {
        /// The aliasing resource (the id it would have been given).
        resource: ResourceId,
    },
    /// A `SplitRule` operand is not in the op's `reads ∪ writes`.
    RuleOperandNotDeclared {
        /// The undeclared operand.
        operand: ResourceId,
    },
    /// `Project.weight` is not a [`ResourceClass::Weight`] resource.
    NotAWeight(ResourceId),
    /// More resources or ops than fit a dense `u32` id.
    TooMany,
}

impl std::fmt::Display for StepGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownResource(r) => write!(f, "unknown resource {}", r.0),
            Self::AliasForward { resource, of } => {
                write!(f, "resource {} aliases non-earlier resource {}", resource.0, of.0)
            }
            Self::AliasOutOfRange { resource } => {
                write!(f, "alias range of resource {} exceeds its target", resource.0)
            }
            Self::RuleOperandNotDeclared { operand } => {
                write!(f, "rule operand {} not in op reads/writes", operand.0)
            }
            Self::NotAWeight(r) => write!(f, "resource {} is not a weight", r.0),
            Self::TooMany => write!(f, "too many resources or ops for u32 ids"),
        }
    }
}

impl std::error::Error for StepGraphError {}

/// Checked element count of a shape (empty shape = scalar = 1).
fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Owned, GPU-free step program: resources plus ops in issue order.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub struct StepGraph {
    resources: Vec<ResourceDecl>,
    ops: Vec<OpDecl>,
}

impl StepGraph {
    /// Empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a resource; its id is its index. An alias must target a strictly
    /// earlier resource and its `offset + numel` must lie inside the target.
    pub fn add_resource(&mut self, decl: ResourceDecl) -> Result<ResourceId, StepGraphError> {
        let id = ResourceId(
            u32::try_from(self.resources.len()).map_err(|_| StepGraphError::TooMany)?,
        );
        if let Some(alias) = decl.alias {
            if alias.of.0 >= id.0 {
                return Err(StepGraphError::AliasForward {
                    resource: id,
                    of: alias.of,
                });
            }
            let target = &self.resources[alias.of.0 as usize];
            let out_of_range = StepGraphError::AliasOutOfRange { resource: id };
            let numel = checked_numel(&decl.shape).ok_or(out_of_range)?;
            let target_numel = checked_numel(&target.shape).ok_or(out_of_range)?;
            let end = alias.offset.checked_add(numel).ok_or(out_of_range)?;
            if end > target_numel {
                return Err(out_of_range);
            }
        }
        self.resources.push(decl);
        Ok(id)
    }

    /// Append an op in issue order; its id is its index. Every referenced
    /// resource must exist, rule operands must be a subset of `reads ∪ writes`,
    /// and `Project.weight` must be a Weight resource.
    pub fn add_op(&mut self, decl: OpDecl) -> Result<OpId, StepGraphError> {
        let id = OpId(u32::try_from(self.ops.len()).map_err(|_| StepGraphError::TooMany)?);
        let n = self.resources.len();
        let exists = |r: ResourceId| -> Result<(), StepGraphError> {
            if (r.0 as usize) < n {
                Ok(())
            } else {
                Err(StepGraphError::UnknownResource(r))
            }
        };
        for &r in decl.reads.iter().chain(decl.writes.iter()) {
            exists(r)?;
        }
        let declared = |r: ResourceId| decl.reads.contains(&r) || decl.writes.contains(&r);
        let check_operand = |r: ResourceId| -> Result<(), StepGraphError> {
            exists(r)?;
            if declared(r) {
                Ok(())
            } else {
                Err(StepGraphError::RuleOperandNotDeclared { operand: r })
            }
        };
        match decl.rule {
            SplitRule::Project {
                weight,
                input,
                output,
            } => {
                check_operand(weight)?;
                check_operand(input)?;
                check_operand(output)?;
                if !matches!(
                    self.resources[weight.0 as usize].class,
                    ResourceClass::Weight { .. }
                ) {
                    return Err(StepGraphError::NotAWeight(weight));
                }
            }
            SplitRule::Accumulate { dst, src } => {
                check_operand(dst)?;
                check_operand(src)?;
            }
            SplitRule::Pointwise
            | SplitRule::RowComplete
            | SplitRule::HeadLocal { .. }
            | SplitRule::Opaque => {}
        }
        self.ops.push(decl);
        Ok(id)
    }

    /// Resources in id order.
    pub fn resources(&self) -> &[ResourceDecl] {
        &self.resources
    }

    /// Ops in program (issue) order.
    pub fn ops(&self) -> &[OpDecl] {
        &self.ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn act(shape: &[usize]) -> ResourceDecl {
        ResourceDecl {
            class: ResourceClass::Activation,
            shape: shape.to_vec(),
            alias: None,
        }
    }

    fn weight(shape: &[usize]) -> ResourceDecl {
        ResourceDecl {
            class: ResourceClass::Weight {
                name: "w".to_string(),
                layer: Some(0),
                group: 1,
            },
            shape: shape.to_vec(),
            alias: None,
        }
    }

    fn alias(of: u32, offset: usize, shape: &[usize]) -> ResourceDecl {
        ResourceDecl {
            class: ResourceClass::Activation,
            shape: shape.to_vec(),
            alias: Some(Alias {
                of: ResourceId(of),
                offset,
            }),
        }
    }

    fn op(rule: SplitRule, reads: &[u32], writes: &[u32]) -> OpDecl {
        OpDecl {
            kind: OpKind::Project,
            layer: Some(0),
            rule,
            reads: reads.iter().map(|&r| ResourceId(r)).collect(),
            writes: writes.iter().map(|&r| ResourceId(r)).collect(),
        }
    }

    #[test]
    fn op_kind_ids_frozen() {
        let pinned: &[(OpKind, u16)] = &[
            (OpKind::Gemv, 1),
            (OpKind::GemvResidual, 2),
            (OpKind::RmsnormAutomatic, 3),
            (OpKind::Attend, 4),
            (OpKind::Rope, 5),
            (OpKind::QkNorm, 6),
            (OpKind::BiasAdd, 7),
            (OpKind::HyperRead, 8),
            (OpKind::GatedDeltaNet, 9),
            (OpKind::IndexedAttention, 10),
            (OpKind::HyperWrite, 11),
            (OpKind::GroupedDepthwise, 12),
            (OpKind::Clear, 13),
            (OpKind::Embed, 14),
            (OpKind::HyperNorm, 15),
            (OpKind::Project, 16),
            (OpKind::BroadcastAdd, 17),
            (OpKind::DenseFfn, 18),
            (OpKind::SandwichAttention, 19),
            (OpKind::SandwichMlp, 20),
            (OpKind::ParallelMoeMlp, 21),
            (OpKind::PerLayerInput, 22),
            (OpKind::Scale, 23),
            (OpKind::Softcap, 24),
            (OpKind::Moe, 25),
            (OpKind::MoeStage, 26),
            (OpKind::GatedShortConv, 27),
            (OpKind::HcPre, 28),
            (OpKind::HcPost, 29),
            (OpKind::HcHead, 30),
            (OpKind::QLoraPrepare, 31),
            (OpKind::PackedProjections, 32),
            (OpKind::QLoraProject, 33),
            (OpKind::KvJoint, 34),
            (OpKind::ApplyTailRope, 35),
            (OpKind::ExpertFfn, 36),
            (OpKind::SparseIndex, 37),
            (OpKind::KvCompress, 38),
            (OpKind::PreparedProjection, 39),
            (OpKind::SwaRingWrite, 40),
            (OpKind::IdentityGather, 41),
            (OpKind::SwaAttention, 42),
            (OpKind::SparseAttention, 43),
            (OpKind::OLora, 44),
            (OpKind::CompressorProject, 45),
            (OpKind::EmbedStreams, 46),
            (OpKind::StreamMeanCapture, 47),
            (OpKind::RingKv, 48),
            (OpKind::BlockStage, 49),
            (OpKind::SwaStage, 50),
            (OpKind::ResidualAdd, 1000),
            (OpKind::SiluMul, 1001),
            (OpKind::Argmax, 1002),
        ];
        for &(kind, value) in pinned {
            assert_eq!(kind as u16, value, "{kind:?}");
        }
    }

    #[test]
    fn dense_ids_follow_insertion_order() {
        let mut g = StepGraph::new();
        let w = g.add_resource(weight(&[4, 8])).unwrap();
        let x = g.add_resource(act(&[8])).unwrap();
        let y = g.add_resource(act(&[4])).unwrap();
        assert_eq!((w, x, y), (ResourceId(0), ResourceId(1), ResourceId(2)));
        let rule = SplitRule::Project {
            weight: w,
            input: x,
            output: y,
        };
        let a = g.add_op(op(rule, &[0, 1], &[2])).unwrap();
        let b = g.add_op(op(rule, &[0, 1], &[2])).unwrap();
        assert_eq!((a, b), (OpId(0), OpId(1)));
        assert_eq!(g.resources().len(), 3);
        assert_eq!(g.ops().len(), 2);
    }

    #[test]
    fn alias_valid_view_accepted() {
        let mut g = StepGraph::new();
        let base = g.add_resource(act(&[4, 8])).unwrap();
        let view = g.add_resource(alias(base.0, 8, &[8])).unwrap();
        assert_eq!(view, ResourceId(1));
        // Exact tail fit.
        g.add_resource(alias(base.0, 24, &[8])).unwrap();
        // Alias of an alias.
        g.add_resource(alias(view.0, 0, &[8])).unwrap();
    }

    #[test]
    fn alias_forward_and_self_refused() {
        let mut g = StepGraph::new();
        g.add_resource(act(&[8])).unwrap();
        // Target == new id (self alias).
        assert_eq!(
            g.add_resource(alias(1, 0, &[1])),
            Err(StepGraphError::AliasForward {
                resource: ResourceId(1),
                of: ResourceId(1)
            })
        );
        // Target beyond the new id.
        assert_eq!(
            g.add_resource(alias(7, 0, &[1])),
            Err(StepGraphError::AliasForward {
                resource: ResourceId(1),
                of: ResourceId(7)
            })
        );
        assert_eq!(g.resources().len(), 1);
    }

    #[test]
    fn alias_out_of_range_refused() {
        let mut g = StepGraph::new();
        g.add_resource(act(&[8])).unwrap();
        let oor = Err(StepGraphError::AliasOutOfRange {
            resource: ResourceId(1),
        });
        assert_eq!(g.add_resource(alias(0, 1, &[8])), oor);
        assert_eq!(g.add_resource(alias(0, 0, &[9])), oor);
        assert_eq!(g.add_resource(alias(0, 9, &[0])), oor);
        // offset + numel overflow.
        assert_eq!(g.add_resource(alias(0, usize::MAX, &[2])), oor);
        // numel overflow.
        assert_eq!(g.add_resource(alias(0, 0, &[usize::MAX, 2])), oor);
        assert_eq!(g.resources().len(), 1);
    }

    #[test]
    fn alias_target_numel_overflow_refused() {
        let mut g = StepGraph::new();
        g.add_resource(act(&[usize::MAX, 2])).unwrap();
        assert_eq!(
            g.add_resource(alias(0, 0, &[1])),
            Err(StepGraphError::AliasOutOfRange {
                resource: ResourceId(1)
            })
        );
    }

    #[test]
    fn op_unknown_resource_refused() {
        let mut g = StepGraph::new();
        g.add_resource(act(&[8])).unwrap();
        let bad = ResourceId(5);
        assert_eq!(
            g.add_op(op(SplitRule::Pointwise, &[5], &[])),
            Err(StepGraphError::UnknownResource(bad))
        );
        assert_eq!(
            g.add_op(op(SplitRule::Pointwise, &[0], &[5])),
            Err(StepGraphError::UnknownResource(bad))
        );
        // Unknown rule operand that is also declared: still unknown first.
        assert_eq!(
            g.add_op(op(
                SplitRule::Accumulate {
                    dst: ResourceId(0),
                    src: bad
                },
                &[0],
                &[]
            )),
            Err(StepGraphError::UnknownResource(bad))
        );
        assert!(g.ops().is_empty());
    }

    #[test]
    fn op_rule_operand_must_be_declared() {
        let mut g = StepGraph::new();
        let w = g.add_resource(weight(&[4, 8])).unwrap();
        let x = g.add_resource(act(&[8])).unwrap();
        let y = g.add_resource(act(&[4])).unwrap();
        let rule = SplitRule::Project {
            weight: w,
            input: x,
            output: y,
        };
        assert_eq!(
            g.add_op(op(rule, &[1], &[2])),
            Err(StepGraphError::RuleOperandNotDeclared { operand: w })
        );
        assert_eq!(
            g.add_op(op(rule, &[0], &[2])),
            Err(StepGraphError::RuleOperandNotDeclared { operand: x })
        );
        assert_eq!(
            g.add_op(op(rule, &[0, 1], &[])),
            Err(StepGraphError::RuleOperandNotDeclared { operand: y })
        );
        let acc = SplitRule::Accumulate { dst: x, src: y };
        assert_eq!(
            g.add_op(op(acc, &[1], &[])),
            Err(StepGraphError::RuleOperandNotDeclared { operand: y })
        );
        assert_eq!(
            g.add_op(op(acc, &[2], &[])),
            Err(StepGraphError::RuleOperandNotDeclared { operand: x })
        );
        assert!(g.ops().is_empty());
        // Writes alone satisfy declaration for an operand.
        g.add_op(op(acc, &[2], &[1])).unwrap();
    }

    #[test]
    fn project_weight_must_be_weight_class() {
        let mut g = StepGraph::new();
        let not_w = g.add_resource(act(&[4, 8])).unwrap();
        let x = g.add_resource(act(&[8])).unwrap();
        let y = g.add_resource(act(&[4])).unwrap();
        let rule = SplitRule::Project {
            weight: not_w,
            input: x,
            output: y,
        };
        assert_eq!(
            g.add_op(op(rule, &[0, 1], &[2])),
            Err(StepGraphError::NotAWeight(not_w))
        );
        assert!(g.ops().is_empty());
    }

    #[test]
    fn rules_without_operands_accept_any_declared_set() {
        let mut g = StepGraph::new();
        let x = g.add_resource(act(&[8])).unwrap();
        for rule in [
            SplitRule::Pointwise,
            SplitRule::RowComplete,
            SplitRule::HeadLocal {
                heads: 4,
                kv_heads: 2,
            },
            SplitRule::Opaque,
        ] {
            g.add_op(op(rule, &[x.0], &[x.0])).unwrap();
        }
        assert_eq!(g.ops().len(), 4);
    }
}
