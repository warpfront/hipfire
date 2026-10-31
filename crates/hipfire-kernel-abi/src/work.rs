// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `WorkV1`: what one launch of a contract computes, for ingestion and
//! roofline (`pm-census extract nodes`, `pm-census queue`).
//!
//! A [`WorkSpecV1`] is a sidecar of exactly one [`KernelAbiV1`]: the encoding
//! contract stays the launch authority and its digest is unchanged; the work
//! spec only names the contract's params. It states:
//!
//! * the op class and codec (taxonomy below), the compute pipe the roofline
//!   charges, useful multiply-accumulates, other arithmetic and minimal bytes
//!   as [`Expr`]s over the contract's params plus the bound params of
//!   [`WorkV1::bind`];
//! * which param carries the activation row count (the row class) and the row
//!   domain the product selector admits this kernel at ([`RowDomainV1`]);
//! * how params that a rocprof trace cannot carry (rocprof records grid and
//!   block, never kernargs) bind from the model: config integers, the weight
//!   tensor directory, the pass's chunk rows or causal prefix ([`BindSource`]).
//!
//! Nothing here is model-specific: config keys are HF naming synonyms, weights
//! are matched by codec and shape, never by tensor name.

use serde::Serialize;

use crate::{Expr, KernelAbiV1};

/// Op classes. Every [`WorkV1::op_class`] is one of these.
pub mod op_class {
    /// Dense weight GEMM (activation rows x weight rows x K).
    pub const GEMM: &str = "gemm";
    /// Expert-grouped GEMM over a stack of weights (routed rows).
    pub const GROUPED_GEMM: &str = "grouped_gemm";
    /// Weight GEMV (one activation row).
    pub const GEMV: &str = "gemv";
    /// Serial or chunked state recurrence (delta rule, SSM scan).
    pub const RECURRENCE: &str = "recurrence";
    /// Depthwise/causal convolution rows.
    pub const CONV: &str = "conv";
    /// Softmax attention (prefill or decode).
    pub const ATTENTION: &str = "attention";
    /// Multi-stream residual mixing (hyper-connections).
    pub const STREAM_MIX: &str = "stream_mix";
    /// Dtype conversion / rounding round trip.
    pub const CONVERT: &str = "convert";
    /// Router score + top-k.
    pub const MOE_ROUTE: &str = "moe_route";
    /// Routed-row scatter / slot planning.
    pub const MOE_SCATTER: &str = "moe_scatter";
    /// Weighted combine of expert outputs.
    pub const MOE_COMBINE: &str = "moe_combine";
    /// Gated activation (SiLU/SiTU/GELU products).
    pub const ACTIVATION: &str = "activation";
    /// Hadamard/rotation and activation quantize producers.
    pub const ROTATE: &str = "rotate";
    /// Norms and per-row prepare/epilogue kernels.
    pub const NORM: &str = "norm";
    /// Row/latent gather, scatter, pack, split.
    pub const GATHER: &str = "gather";
    /// Token embedding lookup.
    pub const EMBEDDING: &str = "embedding";
    /// Runtime buffer copy/fill.
    pub const COPY: &str = "copy";

    pub const ALL: &[&str] = &[
        GEMM,
        GROUPED_GEMM,
        GEMV,
        RECURRENCE,
        CONV,
        ATTENTION,
        STREAM_MIX,
        CONVERT,
        MOE_ROUTE,
        MOE_SCATTER,
        MOE_COMBINE,
        ACTIVATION,
        ROTATE,
        NORM,
        GATHER,
        EMBEDDING,
        COPY,
    ];
}

/// Compute pipe the roofline charges [`WorkV1::macs`] to (2 FLOP per MAC).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pipe {
    WmmaF16,
    WmmaBf16,
    WmmaFp8,
    WmmaIu8,
    WmmaIu4,
    /// Vector FP32 FMA.
    ValuF32,
    /// No arithmetic worth charging: bytes only.
    Memory,
}

/// Which axis of a 2-D weight tensor `[rows, cols]` (row-major, rows = output
/// features, cols = K) a binding reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightAxis {
    Rows,
    Cols,
}

/// Whether the launch reads one weight tensor per call or a stack of experts
/// (tensors whose name differs only in an inner numeric component; the
/// outermost numeric component is the layer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stacking {
    Single,
    Stack,
    /// Either; the binder picks the candidate set whose tensor count matches
    /// the node's calls per chunk.
    Either,
}

/// Where a param that the trace grid does not determine gets its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindSource {
    /// A model config integer: the first key present wins. Keys may be dotted
    /// paths (`rda_config.num_heads`); the binder also looks under
    /// `text_config.` (multimodal configs). Synonyms are HF naming variants,
    /// never one model's spelling only.
    Config(&'static [&'static str]),
    /// Weight-tensor match. Candidates are the 2-D tensors of the model's
    /// tensor directory (optionally restricted to `codecs`, the tensor-directory
    /// codec names) with `stacking`; with `fuse_siblings`, also row-wise
    /// concatenations of sibling tensors (same parent module, same cols and
    /// codec class), as runtimes fuse gate/up or q/k/v at load. A candidate
    /// matches when its `by_axis` extent lies in the interval the grid gives
    /// for param `by` (which is then bound exactly); this param binds to the
    /// candidate's other axis. Distinct values split the node into leaves.
    /// The binding also binds the implicit param [`STACK_PARAM`]: the number
    /// of tensors one launch reads from the matched group (stack members for
    /// a stack, 1 otherwise), for byte expressions.
    Weight {
        by: &'static str,
        by_axis: WeightAxis,
        codecs: &'static [&'static str],
        stacking: Stacking,
        fuse_siblings: bool,
    },
    /// Rows per prefill chunk in this pass (the mode of the [`WorkV1::rows`]
    /// params the grids bind exactly; cross-checked against pass tokens).
    ChunkRows,
    /// Tokens of the pass (`--tokens`).
    PassTokens,
    /// Rows already processed in this pass before this launch's chunk
    /// (chunk ordinal x chunk rows), for causal attention.
    CausalPrefix,
    /// Routed rows: chunk rows x the config top-k when the weight binding is a
    /// stack, else the value of param `dense` (the grid-bound row count).
    RoutedRows {
        topk: &'static [&'static str],
        dense: &'static str,
    },
    /// Token rows of a routed launch (the row class its admission and the
    /// row-class objective use): the chunk rows when the weight binding is a
    /// stack, else the value of param `dense` clamped to the chunk rows. Kept
    /// apart from [`BindSource::RoutedRows`] (useful slots = tokens x top-k,
    /// what the MACs count) so a token-domain predicate is never compared
    /// with a slot count.
    RoutedTokens { dense: &'static str },
    /// An expression over params already bound (in declaration order).
    Derived(Expr),
}

/// One bound param. Params are bound in declaration order; a param named here
/// need not exist in the encoding contract (pure work params such as `topk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BindV1 {
    pub param: &'static str,
    pub source: BindSource,
}

/// Activation-row domain the product selector admits this export at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RowDomainV1 {
    /// Smallest admitted row count (1 = all).
    pub min_rows: u64,
    /// Largest admitted row count; `None` = unbounded.
    pub max_rows: Option<u64>,
    /// Repo-relative `path::CONSTANT` (or `path:line`) of the predicate that
    /// enforces it; empty when the kernel is admitted unconditionally.
    pub predicate: &'static str,
    /// Env switch that disables the route, if any.
    pub kill_env: Option<&'static str>,
}

/// Implicit param every [`BindSource::Weight`] binding binds: tensors one
/// launch reads from the matched group (stack members for a stack, else 1).
pub const STACK_PARAM: &str = "stack";

/// Admitted at every row count.
pub const ALL_ROWS: RowDomainV1 = RowDomainV1 {
    min_rows: 1,
    max_rows: None,
    predicate: "",
    kill_env: None,
};

/// What one launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WorkV1 {
    /// One of [`op_class::ALL`].
    pub op_class: &'static str,
    /// Weight codec for GEMM-class ops (`q8_0`, `mq4g256v2`, `f32`, ...),
    /// the data dtype path otherwise (`f32`, `f32>f16`, `bf16`).
    pub codec: &'static str,
    pub pipe: Pipe,
    /// Useful multiply-accumulates per launch (padding and sentinel tiles
    /// excluded). `None` for ops with no MAC work.
    pub macs: Option<Expr>,
    /// Other arithmetic per launch in FLOP (charged to [`Pipe::ValuF32`]).
    pub flops: Option<Expr>,
    /// Minimal bytes one launch must move; `None` = the sum of the contract's
    /// operand extents.
    pub bytes: Option<Expr>,
    /// The param carrying the activation row count, if the op has rows.
    pub rows: Option<&'static str>,
    pub admit: RowDomainV1,
    pub bind: &'static [BindV1],
    /// Caveats (what the expressions approximate).
    pub note: &'static str,
}

/// A [`WorkV1`] attached to exactly one encoding contract.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct WorkSpecV1 {
    #[serde(serialize_with = "abi_id")]
    pub abi: &'static KernelAbiV1,
    pub work: WorkV1,
}

fn abi_id<S: serde::Serializer>(abi: &&'static KernelAbiV1, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(abi.id)
}

impl WorkSpecV1 {
    /// Names every expression of the work spec may use: the contract's
    /// params, every export constant, and the params of [`WorkV1::bind`].
    pub fn self_check(&self) -> Result<(), String> {
        let w = &self.work;
        if !op_class::ALL.contains(&w.op_class) {
            return Err(format!("{}: unknown op class {}", self.abi.id, w.op_class));
        }
        let mut names: Vec<&str> = self.abi.params.iter().map(|p| p.name).collect();
        for e in self.abi.exports {
            names.extend(e.consts.iter().map(|(k, _)| *k));
        }
        for b in w.bind {
            if let BindSource::Derived(e) = b.source {
                check_names(self.abi.id, &e, &names)?;
            }
            if let BindSource::Weight { by, .. } = b.source {
                if !names.contains(&by) {
                    return Err(format!(
                        "{}: weight binding by unknown param {by}",
                        self.abi.id
                    ));
                }
                names.push(STACK_PARAM);
            }
            if let BindSource::RoutedTokens { dense } = b.source {
                if !names.contains(&dense) {
                    return Err(format!(
                        "{}: routed tokens dense param {dense} unknown",
                        self.abi.id
                    ));
                }
            }
            if let BindSource::RoutedRows { dense, .. } = b.source {
                if !names.contains(&dense) {
                    return Err(format!(
                        "{}: routed rows dense param {dense} unknown",
                        self.abi.id
                    ));
                }
            }
            names.push(b.param);
        }
        for e in [w.macs, w.flops, w.bytes].into_iter().flatten() {
            check_names(self.abi.id, &e, &names)?;
        }
        if let Some(r) = w.rows {
            if !names.contains(&r) {
                return Err(format!("{}: rows param {r} unknown", self.abi.id));
            }
        }
        if w.admit.min_rows == 0 || w.admit.max_rows.is_some_and(|m| m < w.admit.min_rows) {
            return Err(format!("{}: empty row domain", self.abi.id));
        }
        Ok(())
    }
}

fn check_names(id: &str, e: &Expr, names: &[&str]) -> Result<(), String> {
    match e {
        Expr::Const(_) => Ok(()),
        Expr::Param(n) => {
            if names.contains(n) {
                Ok(())
            } else {
                Err(format!("{id}: work expression names unknown param {n}"))
            }
        }
        Expr::Add(xs) | Expr::Mul(xs) => xs.iter().try_for_each(|x| check_names(id, x, names)),
        Expr::Sub(a, b) => {
            check_names(id, a, names)?;
            check_names(id, b, names)
        }
        Expr::DivExact(a, _) | Expr::DivCeil(a, _) | Expr::Div(a, _) => check_names(id, a, names),
    }
}
