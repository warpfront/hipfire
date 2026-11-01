// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `contract:<abi id | export symbol>`: ONE generic [`Node`] derived from a `hipfire-kernel-abi`
//! [`KernelAbiV1`] export, for any contract whose operand formats have an input recipe here. No
//! per-node module: the incumbent, the launch geometry, the buffers, the kernarg image, the emulator
//! case and the work counts all come from the contract (`families::find_export` /
//! `families::work_for`).
//!
//! ```text
//! hip_loop --node contract:generic_gemm.bf16_xf32_tiled_rows.mpw16 --vs contract:generic_gemm.bf16_xf32_batched.wave_per_output \
//!          --shape M=128,K=2560,N=512 [--rowclasses [1,2,4,..]] [--pool] [--arch gfx1201|gfx1100] ...
//! hip_loop --node contract:gemm_q8_0_wmma_gfx12 --shape M=2048,K=2560,N=512 --null-control aa
//! hip_loop --node contract:moe_grouped_qt44_a8_sym_gfx1201_nt1r1 --shape M=2176,K=2560,n_experts=128,topk=8,tokens=512 \
//!          --null-control aa
//! ```
//!
//! # Contract resolution
//!
//! The spec after `contract:` is an ABI id (`generic_gemm.bf16_xf32_batched.wave_per_output`; it must have exactly
//! one export on the arch) or an export symbol (`gemm_f32_batched`, `gemm_q8_0_wmma_gfx12`,
//! `moe_grouped_qt44_a8_sym_gfx1201_nt1r1`). `--arch gfx1201|gfx1100` (default gfx1201) picks the export of
//! that arch. The symbol `gemm_bf16_xf32_tiled_rows` is the LAUNCHER entry of the tiled family: at `M <= 32` the
//! launcher (`Gpu::gemm_bf16_xf32_tiled_rows`, `gemm.rs`) launches `gemm_bf16_xf32_tiled_rows_mp4`, so that symbol resolves
//! to the mp4 export there (recorded in `contract.resolved_from`); the ABI id
//! `generic_gemm.bf16_xf32_tiled_rows.mpw16` is exact and never switches.
//!
//! # Shape (`--shape k=v,...`)
//!
//! Every contract param by name (`M`, `K`, `N`, `n_experts`, ...), plus the recipe knobs `tokens` (activation
//! rows of a ROUTED contract), `topk`, `route_seed`, `route=uniform|hist:<route_hist.v1>[@L]` (real recorded per-token
//! expert lists, sha-verified, cycled to `tokens`; E/topk must match; default uniform), `prefix` (the value of a position device scalar, default 0),
//! and `rows=` (an alias of the activation-row key: for a
//! dense contract the `WorkV1.rows` param, for a routed one the routed rows = `tokens * topk`). A missing
//! param is an error listing every param; the `WorkV1.rows` param (the row class) is what `--rowclasses`
//! injects per class. `m_total`, `x_src_rows`, `x_row_div` of a routed contract default from the routing plan
//! (`x_row_div = topk` = the gate/up form; give `x_row_div=1` for the down form). The bare preset `emu` fills
//! small ragged defaults (`N=37`, `K=256`, `n_experts=8`, `topk=2`, `tokens=37`, the smallest of M in
//! 32/64/128/256 the domain admits). Reserved keys (set by the CLI): `arch`, `vs`, `pool`.
//!
//! # Buffers and recipes
//!
//! One device buffer per operand, `eval(extent)` bytes (the larger of both arms' extents under `--vs`). Inputs
//! come from a recipe keyed by the operand's [`FormatId`] ([`recipe_for`]); a format without a recipe is an
//! error naming it (never fabricated bytes):
//!
//! | format | recipe |
//! |---|---|
//! | `f32.flat`, `f32.rows_n_k`, `f32.rows_x_s01`, `f32.m`, `f32.n_x_m` | uniform f32 in [-1, 1) |
//! | `f16.flat` | uniform f16 in [-1, 1) |
//! | `bf16.flat`, `bf16.token_head_view` | uniform bf16 in [-1, 1) over the whole extent (a strided view's padding included) |
//! | `i8.mirror_plane_view` | INT8 expanded-mirror plane of the Q8 flash prefill (operand `ka`: width `wa`, `vv`: width `dv`; `n_heads` from the shape): `max_rows` rows of `n_heads*(w+2)` B, each `n_heads*w` seeded i8 codes uniform in [-127, 127] then `n_heads` f16 scales, finite and positive, `(0.5 + 0.5 u)/127`, u in [0, 1) |
//! | `ggml.q8_0.34` | blocks of 34 B: f16 scale in [0.001, 0.021) + 32 random i8 codes |
//! | `moe_grouped.expert_ptr_table.u64` | `n_experts` slabs of `M x K/256` MQ4G256V2 groups (136 B, symmetric `zp = f16(-8 sc)`, `moe::synth_rows`), the table holds their device addresses |
//! | `moe_grouped.expert_tile_ids.i32.per_tile16`, `...sorted_slot_index.i32.minus1_padding`, `...expert_offsets.i32.e_plus_1` | the ONE grouped routing plan: `tokens` tokens routed to `topk` of `n_experts` experts (uniform Gumbel-top-k, shape-seeded), scattered like `moe_scatter_fused_k8` (`moe::scatter`) |
//! | `block_i8_128.kb_x_n.136b` | `[K/128][x_src_rows]` records `{f32 d in [0.005, 0.05); i32 s = sum(qs); i8 qs[128]}` |
//! | `block_i4_128.kb_x_n.72b` | `[K/128][x_src_rows]` records `{f32 d; i32 s = sum of the signed nibbles; u8 qs[64]}` ([INFERENCE]: `s` is not read by the symmetric kernels) |
//! | `i32.token_ids` | ids uniform in `[0, vocab)` |
//! | `i32.device_scalar` (operands `pos_ptr`, `base_ptr`: a position) | one `i32` = `prefix=` (default 0). A contract with such an operand and a `max_rows` param refuses `prefix + <rows> > max_rows` (a clippable domain error, so `--rowclasses` drops the classes that do not fit). Any other device scalar (`slot_idx`, `n_dev`) has no recipe |
//!
//! `WriteOnly` operands are pre-poisoned (f32 -1234.5 pattern), `ReadWrite` ones start from their input recipe;
//! `reset` restores both from a pristine device copy; `read_output` concatenates the outputs in operand order.
//! Seeds: operand name hash XOR `--seed`.
//!
//! # Launch
//!
//! Kernarg: an `InvocationV1` (params in contract order, bindings with the device addresses and allocation
//! sizes) goes through `validate_invocation` and the explicit image it encoded is passed padded to 16 bytes.
//! That is what the production launcher does under `launch_maybe_blob`'s blob path (`dispatch.rs`:
//! `blob.pad_to(16)` of the explicit arguments, no hidden words): `launch_kernel_blob` (`hip-bridge/ffi.rs`)
//! hands HIP `HIP_LAUNCH_PARAM_BUFFER_POINTER`, and the HIP runtime writes the hidden block itself (block
//! counts, group sizes, ... for the 13 hidden words / segment 296 of the tiled F32 contract) from the launch
//! geometry, exactly as for the `kernelParams` path. The emulator builds the hidden words from the arm's
//! geometry (`emu.rs`).
//!
//! Geometry: the export's `grid` / `block` / `dynamic_lds` evaluated with `hipfire_kernel_abi::eval` over the
//! params and the export consts. A candidate header `// hip_loop: grid=x,y,z block=x,y,z lds_dynamic=N` (a
//! `,` or `x` separates the 1-3 components; every key optional, unknown keys refused) overrides it; the
//! kernarg image always validates against the CONTRACT geometry.
//!
//! # `--vs contract:<id>`
//!
//! The BASELINE arm is the `--vs` contract's export (its own source, module, symbol and geometry), the
//! CANDIDATE arm is the `--node` contract's. The pair must be signature compatible: same param names/types,
//! same operand formats and access, same explicit fields; anything else is refused. Both arms receive the same
//! kernarg image and operand buffers. The identity gate is the driver's (bit-identical output required).
//!
//! # Row classes and regime
//!
//! `rows_param` is the activation-row key (`N` of the GEMMs, `tokens` of a routed contract). `--rowclasses`
//! (driver) rebuilds the node per class. `bytes_per_call` = `WorkV1.bytes` (else the operand extent sum);
//! `regime` is `hot` (the same buffers back to back) or, with `--pool` (`pool=1`) and `bytes_per_call` above
//! the arch's last-level cache (gfx1201 64 MiB, gfx1100 96 MiB), `pool`: the read-only operands (weights,
//! activations, expert slabs; not the routing index tables) are replicated into distinct seeded sets whose
//! total exceeds the cold-pool threshold (`support/cold.rs`; one set when a set alone exceeds it) and dispatch
//! `i` uses set `i % sets` (`select_set` re-points the kernarg pointers).
//!
//! Included with `#[path = "hip_loop/generic.rs"] mod generic;`.
#![allow(dead_code)]

use crate::node::*;
use hipfire_kernel_abi::families;
use hipfire_kernel_abi::work::{RowDomainV1, WorkSpecV1};
use hipfire_kernel_abi::{
    eval, validate_invocation, Access, Arch, ExportV1, Expr, FieldKind, FormatId, InvocationV1,
    KernelAbiV1, ObjectRefV1, OperandBinding, Rule, ScalarTy,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The default `--rowclasses`.
pub const DEFAULT_ROWCLASSES: [u64; 14] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384,
];

/// Output pre-fill: a finite value no kernel produces.
const SENTINEL: f32 = -1234.5;
const POISON: u32 = 0xDEAD_BEEF;
/// MQ4G256V2 group bytes (`moe::GROUP_BYTES`).
const GROUP_BYTES: usize = crate::moe::GROUP_BYTES;
/// Slab regions the emulator maps one per expert (larger tables: one concatenated region).
const EMU_SPLIT: usize = 16;
const SLAB_NAMES: [&str; EMU_SPLIT] = [
    "w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9", "w10", "w11", "w12", "w13", "w14",
    "w15",
];
/// Candidates the `emu` preset tries for `M` (smallest the contract domain admits).
const EMU_M: [u64; 4] = [32, 64, 128, 256];
const DEFAULT_ROUTE_SEED: u64 = crate::moe::DEFAULT_ROUTE_SEED;

/// Last-level cache of an arch (bytes): gfx1201 64 MiB (Infinity Cache), gfx1100 96 MiB.
pub fn llc_bytes(arch: Arch) -> Result<u64> {
    match arch {
        Arch::Gfx1201 => Ok(64 << 20),
        Arch::Gfx1100 => Ok(96 << 20),
        a => Err(format!(
            "no last-level cache size recorded for {}",
            a.name()
        )),
    }
}

/// Why a node could not be built: `Domain` is a shape the contract's domain rules reject (a row class
/// outside it is CLIPPED by `--rowclasses`), `Other` is a usage error (always fatal).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    Domain(String),
    Other(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Domain(s) | BuildError::Other(s) => f.write_str(s),
        }
    }
}

impl From<String> for BuildError {
    fn from(s: String) -> Self {
        BuildError::Other(s)
    }
}

type BResult<T> = std::result::Result<T, BuildError>;

// ---------------------------------------------------------------------------------------------
// Contract resolution

/// One resolved arm: the contract, its export for the arch and its work sidecar.
#[derive(Clone, Copy)]
pub struct ArmSpec {
    pub abi: &'static KernelAbiV1,
    pub export: &'static ExportV1,
    pub work: Option<&'static WorkSpecV1>,
}

impl ArmSpec {
    fn new(abi: &'static KernelAbiV1, export: &'static ExportV1) -> Self {
        Self {
            abi,
            export,
            work: families::work_for(abi.id),
        }
    }
}

/// `spec` (ABI id or export symbol) -> the arm on `arch`.
pub fn resolve(spec: &str, arch: Arch) -> Result<ArmSpec> {
    if let Some(abi) = families::all().find(|a| a.id == spec) {
        let ex: Vec<&'static ExportV1> = abi.exports.iter().filter(|e| e.arch == arch).collect();
        return match ex.as_slice() {
            [e] => Ok(ArmSpec::new(abi, e)),
            [] => Err(format!("contract {spec}: no export for {}", arch.name())),
            many => Err(format!(
                "contract {spec}: {} exports on {} ({}): name one by its symbol",
                many.len(),
                arch.name(),
                many.iter().map(|e| e.symbol).collect::<Vec<_>>().join(", ")
            )),
        };
    }
    families::find_export(arch, spec)
        .map(|(abi, e)| ArmSpec::new(abi, e))
        .ok_or_else(|| {
            format!(
                "unknown contract `{spec}`: neither a KernelAbiV1 id nor an export symbol on {} \
                 (`hipfire_kernel_abi::families::all`)",
                arch.name()
            )
        })
}

/// Signature compatibility of a `--vs` pair: params (names, types), operand formats/access and explicit
/// fields identical, so one kernarg image and one buffer set launch both.
pub fn signature_compat(base: &KernelAbiV1, cand: &KernelAbiV1) -> Result<()> {
    let mut why = Vec::new();
    let pn = |a: &KernelAbiV1| -> Vec<(&str, ScalarTy)> {
        a.params.iter().map(|p| (p.name, p.ty)).collect()
    };
    if pn(base) != pn(cand) {
        why.push(format!(
            "params differ ({:?} vs {:?})",
            pn(base).iter().map(|p| p.0).collect::<Vec<_>>(),
            pn(cand).iter().map(|p| p.0).collect::<Vec<_>>()
        ));
    }
    let ops = |a: &KernelAbiV1| -> Vec<(FormatId, Access)> {
        a.operands.iter().map(|o| (o.format, o.access)).collect()
    };
    if ops(base) != ops(cand) {
        why.push("operand formats/access differ".to_owned());
    }
    // The metadata `.name` is a label (BF16 batched names its pointers `W`/`X`, the tiled entries `A`/`B`);
    // the kernarg image is determined by offset, width and kind.
    let fl = |a: &KernelAbiV1| -> Vec<(u32, u32, FieldKind)> {
        a.fields
            .iter()
            .map(|f| (f.offset, f.width, f.kind))
            .collect()
    };
    if fl(base) != fl(cand) {
        why.push("explicit kernarg fields differ".to_owned());
    }
    if why.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "--vs: {} and {} are not signature compatible: {}",
            base.id,
            cand.id,
            why.join("; ")
        ))
    }
}

fn repo_root() -> PathBuf {
    match std::env::var_os("HIPFIRE_REPO_ROOT") {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
    }
}

/// The shipped source of an export, byte for byte what the runtime compiles: the runtime's
/// `kernels::*_SRC` is `include_str!` of this file and `KernelCompiler` resolves no local `#include`s
/// (system includes go to hipcc), so a source with a local `#include "..."` has an assembly step this node
/// would not reproduce and is refused.
pub fn load_source(export: &ExportV1) -> Result<(String, String, String)> {
    let ObjectRefV1::HipccSource { module, source } = export.object else {
        return Err(format!(
            "{}: the export is a committed object (ObjectRefV1::Embedded): hip_loop judges hipcc sources",
            export.symbol
        ));
    };
    let path = repo_root().join(source);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if let Some(l) = text
        .lines()
        .find(|l| l.trim_start().starts_with("#include \""))
    {
        return Err(format!(
            "{source}: local include `{}`: the runtime's source assembly for it is not reproduced here",
            l.trim()
        ));
    }
    Ok((module.to_owned(), text, source.to_owned()))
}

// ---------------------------------------------------------------------------------------------
// Expression evaluation with extra names (work specs bind `rows`, `stack`, ...)

fn ev_env(e: &Expr, env: &BTreeMap<String, u64>) -> Result<u64> {
    let ov = |what: &str| format!("work expression {what} overflows");
    match e {
        Expr::Const(v) => Ok(*v),
        Expr::Param(n) => env.get(*n).copied().ok_or_else(|| {
            format!("work expression names `{n}`, which no param, const or recipe knob binds")
        }),
        Expr::Add(xs) => xs.iter().try_fold(0u64, |a, x| {
            a.checked_add(ev_env(x, env)?).ok_or_else(|| ov("sum"))
        }),
        Expr::Mul(xs) => xs.iter().try_fold(1u64, |a, x| {
            a.checked_mul(ev_env(x, env)?).ok_or_else(|| ov("product"))
        }),
        Expr::Sub(a, b) => ev_env(a, env)?
            .checked_sub(ev_env(b, env)?)
            .ok_or_else(|| "work expression is negative".to_owned()),
        Expr::DivExact(a, d) => {
            let v = ev_env(a, env)?;
            if *d == 0 || v % d != 0 {
                return Err(format!("work expression: {v} is not divisible by {d}"));
            }
            Ok(v / d)
        }
        Expr::DivCeil(a, d) => {
            if *d == 0 {
                return Err("work expression: division by zero".into());
            }
            Ok(ev_env(a, env)?.div_ceil(*d))
        }
        Expr::Div(a, d) => {
            if *d == 0 {
                return Err("work expression: division by zero".into());
            }
            Ok(ev_env(a, env)? / d)
        }
    }
}

fn rule_holds(rule: &Rule, arm: &ArmSpec, params: &[u64]) -> Result<bool> {
    let ev = |e: &Expr| eval(e, arm.abi, arm.export, params);
    Ok(match rule {
        Rule::Positive(e) => ev(e)? > 0,
        Rule::MultipleOf(e, m) => *m != 0 && ev(e)? % m == 0,
        Rule::AtMost(e, l) => ev(e)? <= *l,
        Rule::AtLeast(e, l) => ev(e)? >= *l,
        Rule::Equal(a, b) => ev(a)? == ev(b)?,
        Rule::OneOf(e, s) => s.contains(&ev(e)?),
    })
}

/// The first domain rule of `arm` the params violate.
fn domain_violation(arm: &ArmSpec, params: &[u64]) -> Result<Option<String>> {
    for (p, v) in arm.abi.params.iter().zip(params) {
        let fits = match p.ty {
            ScalarTy::I32 => *v <= i32::MAX as u64,
            ScalarTy::U32 => *v <= u64::from(u32::MAX),
            ScalarTy::U64 => true,
        };
        if !fits {
            return Ok(Some(format!(
                "{}: param {} = {v} does not fit {:?}",
                arm.abi.id, p.name, p.ty
            )));
        }
    }
    for d in arm.abi.domain {
        match rule_holds(&d.rule, arm, params) {
            Ok(true) => {}
            Ok(false) => {
                return Ok(Some(format!(
                    "{} ({}): domain rule {:?} failed: {}",
                    arm.abi.id, arm.export.symbol, d.rule, d.why
                )))
            }
            // an inexact division / overflow inside a rule is the rule failing
            Err(e) => {
                return Ok(Some(format!(
                    "{} ({}): domain rule {:?} is not satisfiable: {e} ({})",
                    arm.abi.id, arm.export.symbol, d.rule, d.why
                )))
            }
        }
    }
    // The export's grid must evaluate (an exact `DivExact` tiling) and be non-empty.
    for (axis, e) in arm.export.grid.iter().enumerate() {
        match eval(e, arm.abi, arm.export, params) {
            Ok(0) => {
                return Ok(Some(format!(
                    "{} ({}): empty grid on axis {axis}",
                    arm.abi.id, arm.export.symbol
                )))
            }
            Ok(_) => {}
            Err(e) => {
                return Ok(Some(format!(
                    "{} ({}): grid axis {axis} does not tile the shape: {e}",
                    arm.abi.id, arm.export.symbol
                )))
            }
        }
    }
    Ok(None)
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

// ---------------------------------------------------------------------------------------------
// Input recipes

/// How an operand's bytes are produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recipe {
    F32,
    F16,
    Bf16,
    Q8_0,
    /// INT8 mirror plane (`[n_heads*w codes | n_heads f16 scales]` rows); geometry from `n_heads` and the
    /// operand's width param, see [`ContractNode::q8_plane_geom`].
    Q8Plane,
    /// u64 table of expert slab addresses (slabs of MQ4G256V2 groups).
    ExpertPtrTable,
    TileIds,
    SortedSlots,
    ExpertOffsets,
    BlockI8,
    BlockI4,
    TokenIds,
    /// A position device scalar (`pos_ptr`, `base_ptr`): one `i32` = the `prefix=` knob.
    DeviceScalar,
    /// Pre-poisoned output (any format).
    Poison,
}

impl Recipe {
    /// Index/routing tables are tiny and never replicated into a timing pool.
    fn is_index(self) -> bool {
        matches!(
            self,
            Recipe::TileIds
                | Recipe::SortedSlots
                | Recipe::ExpertOffsets
                | Recipe::TokenIds
                | Recipe::DeviceScalar
        )
    }
    fn is_routing(self) -> bool {
        matches!(
            self,
            Recipe::ExpertPtrTable | Recipe::TileIds | Recipe::SortedSlots | Recipe::ExpertOffsets
        )
    }
}

/// The recipe of an operand of `format` with `access`; a format without one is an error naming it.
pub fn recipe_for(format: FormatId, access: Access) -> Result<Recipe> {
    if access == Access::WriteOnly {
        return Ok(Recipe::Poison);
    }
    Ok(match format.0 {
        "f32.flat" | "f32.rows_n_k" | "f32.rows_x_s01" | "f32.m" | "f32.n_x_m" => Recipe::F32,
        "f16.flat" => Recipe::F16,
        "bf16.flat" | "bf16.token_head_view" => Recipe::Bf16,
        "i8.mirror_plane_view" => Recipe::Q8Plane,
        "ggml.q8_0.34" => Recipe::Q8_0,
        "moe_grouped.expert_ptr_table.u64" => Recipe::ExpertPtrTable,
        "moe_grouped.expert_tile_ids.i32.per_tile16" => Recipe::TileIds,
        "moe_grouped.sorted_slot_index.i32.minus1_padding" => Recipe::SortedSlots,
        "moe_grouped.expert_offsets.i32.e_plus_1" => Recipe::ExpertOffsets,
        "block_i8_128.kb_x_n.136b" => Recipe::BlockI8,
        "block_i4_128.kb_x_n.72b" => Recipe::BlockI4,
        "i32.token_ids" => Recipe::TokenIds,
        "i32.device_scalar" => Recipe::DeviceScalar,
        other => {
            return Err(format!(
                "no input recipe for format {other} ({access:?} operand): hip_loop's generic node does not \
                 fabricate bytes for a format it does not know"
            ))
        }
    })
}

fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}

/// `bytes` of recipe `r` seeded by `seed` (no routing: see [`ContractNode::routing_bytes`]).
fn gen_bytes(r: Recipe, bytes: usize, seed: u64, vocab: u64) -> Vec<u8> {
    let mut rng = Xs64::new(seed);
    let mut out = Vec::with_capacity(bytes);
    match r {
        Recipe::F32 => {
            for _ in 0..bytes / 4 {
                put_f32(&mut out, rng.unit());
            }
        }
        Recipe::F16 => {
            for _ in 0..bytes / 2 {
                out.extend_from_slice(&f16_bits(rng.unit()).to_le_bytes());
            }
        }
        Recipe::Bf16 => {
            for _ in 0..bytes / 2 {
                out.extend_from_slice(&half::bf16::from_f32(rng.unit()).to_bits().to_le_bytes());
            }
        }
        Recipe::Q8_0 => {
            for _ in 0..bytes / 34 {
                let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
                out.extend_from_slice(&f16_bits(0.001 + 0.02 * u).to_le_bytes());
                let mut codes = [0u8; 32];
                for c in codes.chunks_exact_mut(8) {
                    c.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                out.extend_from_slice(&codes);
            }
        }
        Recipe::BlockI8 => {
            for _ in 0..bytes / 136 {
                let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
                let mut qs = [0u8; 128];
                for c in qs.chunks_exact_mut(8) {
                    c.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                let s: i32 = qs.iter().map(|&b| i32::from(b as i8)).sum();
                put_f32(&mut out, 0.005 + 0.045 * u);
                out.extend_from_slice(&s.to_le_bytes());
                out.extend_from_slice(&qs);
            }
        }
        Recipe::BlockI4 => {
            for _ in 0..bytes / 72 {
                let u = (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
                let mut qs = [0u8; 64];
                for c in qs.chunks_exact_mut(8) {
                    c.copy_from_slice(&rng.next_u64().to_le_bytes());
                }
                let nib = |n: u8| i32::from(((n & 0xF) ^ 8) as i8 - 8);
                let s: i32 = qs.iter().map(|&b| nib(b) + nib(b >> 4)).sum();
                put_f32(&mut out, 0.005 + 0.045 * u);
                out.extend_from_slice(&s.to_le_bytes());
                out.extend_from_slice(&qs);
            }
        }
        Recipe::TokenIds => {
            for _ in 0..bytes / 4 {
                out.extend_from_slice(&((rng.next_u64() % vocab.max(1)) as i32).to_le_bytes());
            }
        }
        Recipe::Poison => {
            let pat = SENTINEL.to_le_bytes();
            out.extend((0..bytes).map(|i| pat[i % 4]));
        }
        // Routing tables and the position scalar are built by `host_operand`, never here.
        Recipe::ExpertPtrTable
        | Recipe::TileIds
        | Recipe::SortedSlots
        | Recipe::ExpertOffsets
        | Recipe::DeviceScalar
        | Recipe::Q8Plane => {}
    }
    out.resize(bytes, 0);
    out
}

/// An INT8 mirror plane of `bytes` bytes: rows of `row_bytes` = `[code_bytes codes | f16 scales]`. Codes are
/// uniform in [-127, 127] (the quantizer never emits -128); each scale is `(0.5 + 0.5 u) / 127`, u in [0, 1),
/// as f16, so `code * scale` spans about +-1 like the bf16 recipes.
fn gen_q8_plane(bytes: usize, row_bytes: usize, code_bytes: usize, seed: u64) -> Vec<u8> {
    let mut rng = Xs64::new(seed);
    let mut out = Vec::with_capacity(bytes);
    for _ in 0..bytes / row_bytes.max(1) {
        out.extend(
            (0..code_bytes).map(|_| (((rng.next_u64() >> 33) % 255) as i64 - 127) as i8 as u8),
        );
        for _ in 0..(row_bytes - code_bytes) / 2 {
            let u = (rng.unit() + 1.0) * 0.5;
            out.extend_from_slice(&f16_bits((0.5 + 0.5 * u) / 127.0).to_le_bytes());
        }
    }
    out.resize(bytes, 0);
    out
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ---------------------------------------------------------------------------------------------
// The node

/// One operand's plan.
#[derive(Clone, Debug)]
pub struct OperandPlan {
    pub name: &'static str,
    pub format: FormatId,
    pub access: Access,
    pub recipe: Recipe,
    /// Allocation bytes (the larger of both arms' extents).
    pub bytes: usize,
    /// The explicit-image offset of the pointer field that carries it.
    pub field_offset: u32,
}

struct Set {
    /// Per operand; sets after 0 hold only pooled operands.
    bufs: Vec<Option<Buf>>,
    slabs: Vec<Buf>,
}

struct State {
    sets: Vec<Set>,
    /// `(operand, pristine copy)` of every output.
    pristine: Vec<(usize, Buf)>,
}

pub struct ContractNode {
    /// The `--node` spec after `contract:` (ABI id or export symbol).
    spec: String,
    label: String,
    arch: Arch,
    /// The baseline arm (the `--vs` contract, or the `--node` contract itself).
    base: ArmSpec,
    /// The candidate arm under `--vs` (the `--node` contract).
    cand: Option<ArmSpec>,
    /// `Some(note)`: a launcher rule changed the export of the `--node` spec.
    resolved_from: Option<String>,
    params: Vec<u64>,
    /// The activation-row key and its value (`WorkV1.rows` param, or `tokens` of a routed contract).
    rows_key: Option<(String, u64)>,
    tokens: u64,
    topk: u64,
    routed: bool,
    routing: Option<crate::moe::Routing>,
    route_seed: u64,
    /// `route=hist:`: the recorded expert lists (None = the uniform shape-seeded routing).
    hist: Option<RouteHist>,
    /// Value of a position device scalar operand (`prefix=`).
    prefix: u64,
    operands: Vec<OperandPlan>,
    /// `(explicit-image offset, operand)` of every pointer field.
    ptr_fields: Vec<(u32, usize)>,
    pool: bool,
    pool_sets: usize,
    pool_set_bytes: usize,
    /// `--pool` and a launch streaming more than the last-level cache.
    pool_active: bool,
    /// Bytes one launch streams (operand extents, expert slabs of the touched experts).
    footprint_bytes: u64,
    bytes_per_call: u64,
    work_flops: f64,
    st: Option<State>,
}

fn param_value(abi: &KernelAbiV1, params: &[u64], name: &str) -> Option<u64> {
    abi.param_index(name).map(|i| params[i])
}

/// `--shape route=hist:<route_hist.v1 json>[@L]`: the per-token expert lists of a pinned `route_hist.v1`
/// (`pm-census extract route-hist`; headerless `i32` LE ids bin next to it, sha256-verified), layer `L` or
/// every layer's records concatenated in record order. Token `t` takes recorded row `t % rows` (cycled).
#[derive(Clone, Debug, PartialEq)]
struct RouteHist {
    path: String,
    id: String,
    layer: Option<u32>,
    n_experts: usize,
    topk: usize,
    /// Row-major `rows x topk`.
    ids: Vec<i32>,
}

impl RouteHist {
    fn parse(spec: &str) -> BResult<Self> {
        let (path, layer) = match spec.rsplit_once('@') {
            Some((p, l)) => (
                p,
                Some(
                    l.parse::<u32>()
                        .map_err(|e| format!("--shape route=hist:{spec}: layer `{l}`: {e}"))?,
                ),
            ),
            None => (spec, None),
        };
        Self::load(std::path::Path::new(path), layer).map_err(BuildError::Other)
    }

    fn load(path: &std::path::Path, layer: Option<u32>) -> std::result::Result<Self, String> {
        let err = |m: String| format!("--shape route=hist:{}: {m}", path.display());
        let doc: Value =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| err(e.to_string()))?)
                .map_err(|e| err(e.to_string()))?;
        if doc["schema"] != "route_hist.v1" {
            return Err(err(format!(
                "schema {}: expected route_hist.v1",
                doc["schema"]
            )));
        }
        let id = doc["id"]
            .as_str()
            .filter(|s| s.len() == 64)
            .map(str::to_owned)
            .ok_or_else(|| err("missing 64-hex top-level `id`".into()))?;
        let num = |v: &Value, what: &str| {
            v.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| err(format!("missing or non-integer {what}")))
        };
        let n_experts = num(&doc["n_experts"], "n_experts")?;
        let topk = num(&doc["topk"], "topk")?;
        let rel = doc["ids"]["path"]
            .as_str()
            .ok_or_else(|| err("missing ids.path".into()))?;
        let bin_path = if std::path::Path::new(rel).is_absolute() {
            PathBuf::from(rel)
        } else {
            path.parent().unwrap_or(std::path::Path::new(".")).join(rel)
        };
        let bin =
            std::fs::read(&bin_path).map_err(|e| err(format!("{}: {e}", bin_path.display())))?;
        let (want, got) = (
            doc["ids"]["sha256"].as_str().unwrap_or(""),
            crate::elf::sha256_hex(&bin),
        );
        if want != got {
            return Err(err(format!(
                "ids sha256 mismatch ({}): manifest {want}, file {got}",
                bin_path.display()
            )));
        }
        let records = doc["records"]
            .as_array()
            .ok_or_else(|| err("missing records".into()))?;
        let mut at = 0usize;
        let mut ids = Vec::new();
        for (i, r) in records.iter().enumerate() {
            let (l, n, k) = (
                num(&r["layer"], "record layer")?,
                num(&r["n"], "record n")?,
                num(&r["k"], "record k")?,
            );
            if k != topk {
                return Err(err(format!("record {i}: k={k}, topk={topk}")));
            }
            let end = at + n * k * 4;
            let part = bin.get(at..end).ok_or_else(|| {
                err(format!(
                    "record {i} runs past the ids bin ({} bytes)",
                    bin.len()
                ))
            })?;
            at = end;
            if layer.is_some_and(|want| want as usize != l) {
                continue;
            }
            for c in part.chunks_exact(4) {
                let e = i32::from_le_bytes(c.try_into().unwrap());
                if e < 0 || e as usize >= n_experts {
                    return Err(err(format!(
                        "record {i}: expert id {e} outside [0,{n_experts})"
                    )));
                }
                ids.push(e);
            }
        }
        if at != bin.len() {
            return Err(err(format!(
                "ids bin has {} bytes, records cover {at}",
                bin.len()
            )));
        }
        if ids.is_empty() {
            return Err(err(match layer {
                Some(l) => format!("no records for layer {l}"),
                None => "no rows".into(),
            }));
        }
        Ok(Self {
            path: path.display().to_string(),
            id,
            layer,
            n_experts,
            topk,
            ids,
        })
    }

    /// `tokens x topk` expert ids, token `t` = recorded row `t % rows`.
    fn table(&self, tokens: usize) -> Vec<i32> {
        let rows = self.ids.len() / self.topk;
        let mut out = Vec::with_capacity(tokens * self.topk);
        for t in 0..tokens {
            let r = t % rows;
            out.extend_from_slice(&self.ids[r * self.topk..(r + 1) * self.topk]);
        }
        out
    }
}

impl ContractNode {
    /// The `route` record of the info JSON: `{kind, path, id, layer}`.
    fn info_route(&self) -> Value {
        match &self.hist {
            Some(h) => json!({"kind": "hist", "path": h.path, "id": h.id, "layer": h.layer}),
            None => json!({"kind": "uniform"}),
        }
    }

    /// Build from the full `--node` string (`contract:<spec>`) and `--shape`.
    pub fn build(name: &str, shape: &ShapeSpec) -> BResult<Self> {
        let spec = name
            .strip_prefix("contract:")
            .ok_or_else(|| format!("`{name}` is not a contract: node"))?;
        let mut kv = shape.kv.clone();
        let arch = match kv.remove("arch") {
            None => Arch::Gfx1201,
            Some(a) => Arch::parse(&a)
                .ok_or_else(|| format!("arch={a}: expected one of gfx1100, gfx1151, gfx1201"))?,
        };
        let vs = kv
            .remove("vs")
            .map(|v| v.strip_prefix("contract:").unwrap_or(&v).to_owned());
        let pool = match kv.remove("pool").as_deref() {
            None | Some("0") => false,
            Some("1") => true,
            Some(o) => return Err(format!("pool={o}: 0 or 1").into()),
        };
        let preset = match shape.preset.as_deref() {
            None => None,
            Some("emu") => Some("emu"),
            Some(o) => {
                return Err(format!("contract node: unknown shape preset `{o}` (have: emu)").into())
            }
        };
        let take_u64 = |kv: &mut BTreeMap<String, String>, k: &str| -> Result<Option<u64>> {
            kv.remove(k)
                .map(|v| {
                    v.parse::<u64>()
                        .map_err(|e| format!("--shape {k}={v}: {e}"))
                })
                .transpose()
        };

        // The `--node` arm (launcher rules may re-resolve it once the params are known) and, under
        // `--vs`, the baseline arm.
        let mut node_arm = resolve(spec, arch)?;
        let abi = node_arm.abi;
        let rows_param_of =
            |a: &ArmSpec| -> Option<&'static str> { a.work.and_then(|w| w.work.rows) };
        let routed = abi
            .operands
            .iter()
            .any(|o| recipe_for(o.format, o.access).is_ok_and(|r| r.is_routing()));
        let rows_key_name: Option<String> = match rows_param_of(&node_arm) {
            Some(r) if abi.param_index(r).is_some() => Some(r.to_owned()),
            Some(_) if routed => Some("tokens".to_owned()),
            Some(r) => {
                return Err(format!(
                    "{}: WorkV1.rows names `{r}`, which is neither a contract param nor a routed contract's rows",
                    abi.id
                )
                .into())
            }
            None => None,
        };

        // ---- params from the shape
        let mut vals: Vec<Option<u64>> = Vec::new();
        for p in abi.params {
            vals.push(take_u64(&mut kv, p.name)?);
        }
        let tokens_kv = take_u64(&mut kv, "tokens")?;
        let topk_kv = take_u64(&mut kv, "topk")?;
        let rows_alias = take_u64(&mut kv, "rows")?;
        let route_seed = take_u64(&mut kv, "route_seed")?.unwrap_or(DEFAULT_ROUTE_SEED);
        let prefix = take_u64(&mut kv, "prefix")?.unwrap_or(0);
        let hist = match kv.remove("route") {
            None => None,
            Some(r) if r == "uniform" => None,
            Some(r) => match r.strip_prefix("hist:") {
                Some(spec) => Some(RouteHist::parse(spec)?),
                None => {
                    return Err(format!(
                        "--shape route={r}: expected uniform|hist:<route_hist.v1 json>[@layer]"
                    )
                    .into())
                }
            },
        };
        if hist.is_some() && !routed {
            return Err(format!(
                "{}: route= is a knob of routed (grouped) contracts only",
                abi.id
            )
            .into());
        }
        if let Some(k) = kv.keys().next() {
            let mut known: Vec<&str> = abi.params.iter().map(|p| p.name).collect();
            known.extend(["tokens", "topk", "rows", "route_seed", "route", "prefix"]);
            return Err(format!(
                "--shape: unknown key `{k}` for {} (params and knobs: {})",
                abi.id,
                known.join(", ")
            )
            .into());
        }

        // The `emu` preset fills small ragged defaults (a given value always wins).
        let emu = preset == Some("emu");
        let set_default = |vals: &mut Vec<Option<u64>>, name: &str, v: u64| {
            if let Some(i) = abi.param_index(name) {
                vals[i].get_or_insert(v);
            }
        };
        if emu {
            for (n, v) in [
                ("K", 256),
                ("N", 37),
                ("n_experts", 8),
                ("dim", 64),
                ("vocab", 100),
            ] {
                set_default(&mut vals, n, v);
            }
        }

        // ---- routed contracts: activation rows, routing plan, derived params
        let mut tokens = 0u64;
        let mut topk = 1u64;
        let mut routing = None;
        if routed {
            let ne_i = abi.param_index("n_experts").ok_or_else(|| {
                format!("{}: a routed contract without an n_experts param", abi.id)
            })?;
            let e = vals[ne_i].ok_or_else(|| {
                format!(
                    "{}: --shape n_experts=<experts> is required (a routed contract)",
                    abi.id
                )
            })?;
            topk = match topk_kv {
                Some(t) => t,
                None if e == 1 => 1,
                None if emu => 2,
                None => {
                    return Err(format!(
                        "{}: --shape topk=<experts per token> is required (n_experts = {e})",
                        abi.id
                    )
                    .into())
                }
            };
            if topk == 0 || topk > e {
                return Err(format!("topk={topk} must be in 1..=n_experts ({e})").into());
            }
            tokens = match (tokens_kv, rows_alias) {
                (Some(_), Some(_)) => return Err("give tokens= or rows=, not both".to_owned().into()),
                (Some(t), None) => t,
                (None, Some(r)) => {
                    if r % topk != 0 {
                        return Err(format!("rows={r} routed rows is not a multiple of topk={topk}").into());
                    }
                    r / topk
                }
                (None, None) if emu => 37,
                (None, None) => {
                    return Err(format!(
                        "{}: --shape tokens=<activation rows> is required (the row class of a routed contract)",
                        abi.id
                    )
                    .into())
                }
            };
            if tokens == 0 {
                return Err(BuildError::Domain(format!("{}: tokens = 0", abi.id)));
            }
            let slots = tokens
                .checked_mul(topk)
                .filter(|v| v.saturating_add(e * 16) < i32::MAX as u64)
                .ok_or_else(|| {
                    format!("tokens*topk = {tokens}*{topk} overflows the i32 slot indices")
                })?;
            let xrd = *vals[abi.param_index("x_row_div").unwrap_or(ne_i)].get_or_insert(topk);
            if abi.param_index("x_row_div").is_none() {
                return Err(format!("{}: a routed contract without x_row_div", abi.id).into());
            }
            if xrd == 0 {
                return Err(BuildError::Domain("x_row_div = 0".to_owned()));
            }
            set_default(&mut vals, "x_src_rows", slots.div_ceil(xrd));
            let bound = crate::moe::m_total_bound(slots as usize, e as usize) as u64;
            set_default(&mut vals, "m_total", bound);
            let m_total = vals[abi
                .param_index("m_total")
                .ok_or_else(|| format!("{}: a routed contract without m_total", abi.id))?]
            .unwrap();
            if m_total % 16 != 0 {
                return Err(format!(
                    "m_total = {m_total} is not a multiple of 16: the routing plan lays 16-slot tiles"
                )
                .into());
            }
            let topk_ids = match &hist {
                Some(h) => {
                    if h.n_experts as u64 != e {
                        return Err(format!(
                            "route=hist: the recording has n_experts={}, the shape n_experts={e}",
                            h.n_experts
                        )
                        .into());
                    }
                    if h.topk as u64 != topk {
                        return Err(format!(
                            "route=hist: the recording has topk={}, the shape topk={topk}",
                            h.topk
                        )
                        .into());
                    }
                    h.table(tokens as usize)
                }
                None => crate::moe::route_topk(
                    tokens as usize,
                    topk as usize,
                    e as usize,
                    crate::moe::Route::Uniform,
                    0.0,
                    route_seed,
                ),
            };
            routing = Some(
                crate::moe::scatter(&topk_ids, e as usize, m_total as usize)
                    .map_err(|err| BuildError::Other(format!("routing plan: {err}")))?,
            );
        } else if tokens_kv.is_some() || topk_kv.is_some() {
            return Err(format!(
                "{}: tokens/topk are recipe knobs of routed (grouped) contracts only",
                abi.id
            )
            .into());
        }

        // The dense activation-row key: `rows=` is its alias.
        let mut rows_value: Option<u64> = if routed { Some(tokens) } else { None };
        if !routed {
            if let Some(r) = rows_alias {
                let key = rows_key_name.as_deref().ok_or_else(|| {
                    format!("{}: has no rows param, `rows=` is meaningless", abi.id)
                })?;
                let i = abi.param_index(key).expect("dense rows key is a param");
                if vals[i].is_some_and(|v| v != r) {
                    return Err(
                        format!("rows={r} conflicts with {key}={}", vals[i].unwrap()).into(),
                    );
                }
                vals[i] = Some(r);
            }
            if let Some(k) = &rows_key_name {
                rows_value = vals[abi.param_index(k).expect("dense rows key is a param")];
            }
        }

        // `emu`: the smallest M of EMU_M the domain admits (the other params are all set by now).
        if emu {
            if let Some(mi) = abi.param_index("M") {
                if vals[mi].is_none()
                    && vals.iter().enumerate().all(|(i, v)| i == mi || v.is_some())
                {
                    for m in EMU_M {
                        let mut probe: Vec<u64> = vals.iter().map(|v| v.unwrap_or(m)).collect();
                        probe[mi] = m;
                        if domain_violation(&node_arm, &probe)?.is_none() {
                            vals[mi] = Some(m);
                            break;
                        }
                    }
                }
            }
        }

        let missing: Vec<String> = abi
            .params
            .iter()
            .zip(&vals)
            .filter(|(_, v)| v.is_none())
            .map(|(p, _)| format!("{} ({})", p.name, p.doc))
            .collect();
        if !missing.is_empty() {
            return Err(format!(
                "{}: missing params: {} (give --shape name=value,...{})",
                abi.id,
                missing.join("; "),
                rows_key_name
                    .as_deref()
                    .map(|k| format!("; `{k}` is the row class `--rowclasses` sets"))
                    .unwrap_or_default()
            )
            .into());
        }
        let params: Vec<u64> = vals.into_iter().map(|v| v.unwrap()).collect();

        // Position device scalars: `prefix=` is their value; the rest have no recipe.
        let mut has_position = false;
        for o in abi.operands {
            if recipe_for(o.format, o.access).is_ok_and(|r| r == Recipe::DeviceScalar) {
                if !matches!(o.name, "pos_ptr" | "base_ptr") {
                    return Err(format!(
                        "{} operand {}: an i32.device_scalar other than a position (pos_ptr, base_ptr) has no input recipe",
                        abi.id, o.name
                    )
                    .into());
                }
                has_position = true;
            }
        }
        if prefix > 0 && !has_position {
            return Err(format!(
                "{}: prefix={prefix}, but the contract has no position device scalar (pos_ptr, base_ptr)",
                abi.id
            )
            .into());
        }
        if prefix > i32::MAX as u64 {
            return Err(format!("prefix={prefix} does not fit the i32 device scalar").into());
        }
        if has_position {
            if let (Some(max_rows), Some(rows)) =
                (param_value(abi, &params, "max_rows"), rows_value)
            {
                if prefix.saturating_add(rows) > max_rows {
                    return Err(BuildError::Domain(format!(
                        "prefix {prefix} + rows {rows} exceeds max_rows {max_rows}: the keys of the last rows would lie outside the K/V views"
                    )));
                }
            }
        }

        // Launcher rule: `gemm_bf16_xf32_tiled_rows` at M <= 32 launches the mp4 entry.
        let mut resolved_from = None;
        if node_arm.export.symbol == "gemm_bf16_xf32_tiled_rows"
            && spec == "gemm_bf16_xf32_tiled_rows"
        {
            if let Some(m) = param_value(abi, &params, "M").filter(|m| *m <= 32) {
                node_arm = resolve("gemm_bf16_xf32_tiled_rows_mp4", arch)?;
                resolved_from = Some(format!(
                    "gemm_bf16_xf32_tiled_rows -> gemm_bf16_xf32_tiled_rows_mp4 (Gpu::gemm_bf16_xf32_tiled_rows launches the MPW=4 \
                     entry at M = {m} <= 32)"
                ));
            }
        }
        let (base, cand) = match &vs {
            None => (node_arm, None),
            Some(v) => {
                let mut b = resolve(v, arch)?;
                if b.export.symbol == "gemm_bf16_xf32_tiled_rows"
                    && v == "gemm_bf16_xf32_tiled_rows"
                    && param_value(b.abi, &params, "M").is_some_and(|m| m <= 32)
                {
                    b = resolve("gemm_bf16_xf32_tiled_rows_mp4", arch)?;
                }
                signature_compat(b.abi, node_arm.abi)?;
                if b.export.symbol == node_arm.export.symbol
                    && b.export.arch == node_arm.export.arch
                {
                    return Err(format!(
                        "--vs {v}: the same export as --node (use --null-control aa for an A/A)"
                    )
                    .into());
                }
                (b, Some(node_arm))
            }
        };
        for arm in std::iter::once(&base).chain(cand.iter()) {
            if let Some(v) = domain_violation(arm, &params)? {
                return Err(BuildError::Domain(v));
            }
            // Resolves the source now (GPU-free): an Embedded export or a local include is refused here.
            load_source(arm.export)?;
        }

        // ---- operand plans: bytes = the larger extent of the arms; recipe per format
        let mut operands = Vec::new();
        let mut ptr_fields = Vec::new();
        for (i, o) in base.abi.operands.iter().enumerate() {
            let recipe = recipe_for(o.format, o.access)
                .map_err(|e| format!("{} operand {}: {e}", base.abi.id, o.name))?;
            let mut bytes = 0u64;
            for arm in std::iter::once(&base).chain(cand.iter()) {
                let ex = arm.abi.operands[i].extent.ok_or_else(|| {
                    format!("{} operand {} has no extent expression", arm.abi.id, o.name)
                })?;
                bytes = bytes.max(
                    eval(&ex, arm.abi, arm.export, &params)
                        .map_err(|e| format!("{} operand {} extent: {e}", arm.abi.id, o.name))?,
                );
            }
            let off = base
                .abi
                .fields
                .iter()
                .find_map(|f| match f.kind {
                    FieldKind::Pointer { operand } if operand == o.name => Some(f.offset),
                    _ => None,
                })
                .ok_or_else(|| {
                    format!("{}: operand {} has no pointer field", base.abi.id, o.name)
                })?;
            ptr_fields.push((off, i));
            operands.push(OperandPlan {
                name: o.name,
                format: o.format,
                access: o.access,
                recipe,
                bytes: usize::try_from(bytes)
                    .map_err(|_| format!("operand {} is {bytes} B", o.name))?,
                field_offset: off,
            });
        }
        for o in operands.iter().filter(|o| o.recipe == Recipe::Q8Plane) {
            let w = match o.name {
                "ka" => "wa",
                "vv" => "dv",
                n => {
                    return Err(format!(
                        "{} operand {n}: the i8.mirror_plane_view recipe knows the planes `ka` (width wa) and `vv` (width dv) only",
                        base.abi.id
                    )
                    .into())
                }
            };
            for k in ["n_heads", w] {
                if base.abi.param_index(k).is_none() {
                    return Err(format!(
                        "{} operand {}: the i8.mirror_plane_view recipe needs the param `{k}`",
                        base.abi.id, o.name
                    )
                    .into());
                }
            }
        }
        if operands.iter().any(|o| o.recipe == Recipe::ExpertPtrTable) {
            for k in ["M", "K", "n_experts"] {
                if base.abi.param_index(k).is_none() {
                    return Err(format!(
                        "{}: the expert-slab recipe needs the params M, K and n_experts (`{k}` is missing)",
                        base.abi.id
                    )
                    .into());
                }
            }
        }
        if routed && routing.is_none() {
            return Err("internal: routed contract without a routing plan"
                .to_owned()
                .into());
        }

        // ---- work, bytes per call, regime
        let mut env: BTreeMap<String, u64> = BTreeMap::new();
        for (p, v) in base.abi.params.iter().zip(&params) {
            env.insert(p.name.to_owned(), *v);
        }
        for (k, v) in base.export.consts {
            env.insert((*k).to_owned(), *v);
        }
        env.insert("topk".into(), topk);
        env.insert("tokens".into(), tokens);
        env.insert("prefix".into(), prefix);
        let stack = param_value(base.abi, &params, "n_experts").unwrap_or(1);
        env.insert("stack".into(), stack);
        if let Some(w) = base.work {
            // Routed contracts: `RoutedRows` binds the useful slots (tokens x topk, what MACs count),
            // `RoutedTokens` the token row class (what admission and the sweep use).
            for b in w.work.bind {
                match b.source {
                    hipfire_kernel_abi::work::BindSource::RoutedRows { .. } => {
                        env.insert(b.param.to_owned(), tokens * topk);
                    }
                    hipfire_kernel_abi::work::BindSource::RoutedTokens { .. } => {
                        env.insert(b.param.to_owned(), tokens);
                    }
                    _ => {}
                }
            }
        }
        let operand_sum: u64 = operands.iter().map(|o| o.bytes as u64).sum();
        let (macs, other_flops, bytes_per_call) = match base.work {
            Some(w) => {
                let w = &w.work;
                let e = |x: &Option<Expr>| x.as_ref().map(|x| ev_env(x, &env)).transpose();
                (
                    e(&w.macs)?.unwrap_or(0),
                    e(&w.flops)?.unwrap_or(0),
                    e(&w.bytes)?.unwrap_or(operand_sum),
                )
            }
            None => (0, 0, operand_sum),
        };
        let work_flops = 2.0 * macs as f64 + other_flops as f64;
        // What one launch streams: every operand, an expert-slab table counted as the slabs of the experts
        // the routing touches (the work spec's `bytes` of a routed contract is a one-slab lower bound).
        let footprint_bytes: u64 = operands
            .iter()
            .map(|o| match (o.recipe, &routing) {
                (Recipe::ExpertPtrTable, Some(r)) => {
                    let m = param_value(base.abi, &params, "M").unwrap_or(0);
                    let k = param_value(base.abi, &params, "K").unwrap_or(0);
                    r.touched_experts() as u64 * m * (k / 256) * GROUP_BYTES as u64
                }
                _ => o.bytes as u64,
            })
            .sum();
        let pool_active = pool && bytes_per_call.max(footprint_bytes) > llc_bytes(arch)?;
        let pool_set_bytes: usize = operands
            .iter()
            .filter(|o| o.access == Access::ReadOnly && !o.recipe.is_index())
            .map(|o| {
                if o.recipe == Recipe::ExpertPtrTable {
                    let m = param_value(base.abi, &params, "M").unwrap_or(0) as usize;
                    let k = param_value(base.abi, &params, "K").unwrap_or(0) as usize;
                    let e = param_value(base.abi, &params, "n_experts").unwrap_or(0) as usize;
                    e * m * (k / 256) * GROUP_BYTES
                } else {
                    o.bytes
                }
            })
            .sum();
        let pool_sets = if !pool_active {
            1
        } else if pool_set_bytes > crate::cold::POOL_BYTES {
            1
        } else {
            crate::cold::pool_sets(pool_set_bytes)
        };

        let mut label = base
            .abi
            .params
            .iter()
            .zip(&params)
            .filter(|(p, _)| !matches!(p.name, "m_total" | "x_src_rows" | "x_row_div"))
            .map(|(p, v)| format!("{}{v}", p.name))
            .collect::<Vec<_>>()
            .join("-");
        if routed {
            label.push_str(&format!("-tokens{tokens}-topk{topk}"));
        }
        if prefix > 0 {
            label.push_str(&format!("-prefix{prefix}"));
        }
        Ok(Self {
            spec: spec.to_owned(),
            label,
            arch,
            base,
            cand,
            resolved_from,
            params,
            rows_key: rows_key_name.zip(rows_value),
            tokens,
            topk,
            routed,
            routing,
            route_seed,
            hist,
            prefix,
            operands,
            ptr_fields,
            pool,
            pool_sets,
            pool_set_bytes,
            pool_active,
            footprint_bytes,
            bytes_per_call,
            work_flops,
            st: None,
        })
    }

    /// The arm whose geometry/header `geometry_with` serves (the candidate under `--vs`).
    fn cand_arm(&self) -> &ArmSpec {
        self.cand.as_ref().unwrap_or(&self.base)
    }

    fn arm_incumbent(&self, arm: &ArmSpec) -> Result<Incumbent> {
        let (module, source, _) = load_source(arm.export)?;
        Ok(Incumbent {
            module,
            source,
            symbol: arm.export.symbol.to_owned(),
            extra_flags: String::new(),
            selection: arm.abi.id.to_owned(),
        })
    }

    /// The contract geometry of `arm` at this shape.
    pub fn arm_geometry(&self, arm: &ArmSpec) -> Result<Geometry> {
        let mut grid = [0u32; 3];
        for (g, e) in grid.iter_mut().zip(arm.export.grid.iter()) {
            let v = eval(e, arm.abi, arm.export, &self.params)?;
            *g = u32::try_from(v).map_err(|_| format!("grid dimension {v} exceeds u32"))?;
        }
        if grid.contains(&0) {
            return Err(format!("{}: empty grid {grid:?}", arm.export.symbol));
        }
        Ok(Geometry {
            grid,
            block: arm.export.block,
            lds: arm.export.dynamic_lds,
        })
    }

    /// The activation-row key and value (`N`, `tokens`, ...), if the contract has rows.
    pub fn rows(&self) -> Option<(&str, u64)> {
        self.rows_key.as_ref().map(|(k, v)| (k.as_str(), *v))
    }

    /// The size the PRODUCT launcher compares its row predicates against: for a routed (grouped) contract the launch's
    /// `m_total` (the sentinel-padded grouped-row bound), else the row-class value itself (`rows`).
    pub fn launch_rows(&self) -> Option<u64> {
        if self.routing.is_some() {
            if let Some(i) = self.base.abi.param_index("m_total") {
                return Some(self.params[i]);
            }
        }
        self.rows().map(|(_, v)| v)
    }

    pub fn bytes_per_call(&self) -> u64 {
        self.bytes_per_call
    }

    pub fn operands(&self) -> &[OperandPlan] {
        &self.operands
    }

    pub fn params(&self) -> &[u64] {
        &self.params
    }

    pub fn export_symbols(&self) -> (&'static str, &'static str) {
        (self.base.export.symbol, self.cand_arm().export.symbol)
    }

    pub fn regime(&self) -> &'static str {
        if self.pool_active {
            "pool"
        } else {
            "hot"
        }
    }

    fn is_pool_active(&self) -> bool {
        self.pool_active
    }

    pub fn pool_sets(&self) -> usize {
        self.pool_sets
    }

    pub fn admit(&self) -> (Option<RowDomainV1>, Option<RowDomainV1>) {
        (
            self.base.work.map(|w| w.work.admit),
            self.cand.and_then(|c| c.work).map(|w| w.work.admit),
        )
    }

    fn st(&self) -> Result<&State> {
        self.st
            .as_ref()
            .ok_or_else(|| "contract node: setup has not run".to_owned())
    }

    /// Whether operand `op` is replicated into the timing pool's sets.
    fn pooled(&self, op: usize) -> bool {
        self.pool_active
            && self.operands[op].access == Access::ReadOnly
            && !self.operands[op].recipe.is_index()
    }

    fn vocab(&self) -> u64 {
        param_value(self.base.abi, &self.params, "vocab").unwrap_or(1)
    }

    /// `M x K/256` groups per expert slab.
    fn slab_dims(&self) -> (usize, usize) {
        let m = param_value(self.base.abi, &self.params, "M").unwrap_or(0) as usize;
        let k = param_value(self.base.abi, &self.params, "K").unwrap_or(0) as usize;
        (m, k / 256)
    }

    pub fn slab_bytes(&self) -> usize {
        let (m, g) = self.slab_dims();
        m * g * GROUP_BYTES
    }

    fn n_experts(&self) -> usize {
        param_value(self.base.abi, &self.params, "n_experts").unwrap_or(0) as usize
    }

    /// One expert slab of set `set`.
    pub fn host_slab(&self, seed: u64, set: usize, e: usize) -> Vec<u8> {
        let (m, g) = self.slab_dims();
        crate::moe::synth_rows(m, g, seed ^ ((set as u64) << 40) ^ ((e as u64 + 1) << 16))
    }

    /// The routing-table bytes of operand `i` (`None` for other recipes).
    pub fn routing_bytes(&self, i: usize) -> Option<Vec<u8>> {
        let r = self.routing.as_ref()?;
        match self.operands[i].recipe {
            Recipe::TileIds => Some(i32s(&r.tiles)),
            Recipe::SortedSlots => Some(i32s(&r.sorted)),
            Recipe::ExpertOffsets => Some(i32s(&r.offsets)),
            _ => None,
        }
    }

    /// Host bytes of operand `i` (the expert pointer table is zero-filled: its addresses are a device /
    /// emulator matter). Deterministic in `seed` and `set`.
    pub fn host_operand(&self, i: usize, seed: u64, set: usize) -> Vec<u8> {
        let o = &self.operands[i];
        if o.recipe == Recipe::DeviceScalar {
            let mut b = (self.prefix as i32).to_le_bytes().to_vec();
            b.resize(o.bytes, 0);
            return b;
        }
        if let Some(b) = self.routing_bytes(i) {
            debug_assert_eq!(b.len(), o.bytes, "{}: routing table vs extent", o.name);
            return b;
        }
        if o.recipe == Recipe::ExpertPtrTable {
            return vec![0u8; o.bytes];
        }
        let seed = seed ^ fnv(o.name) ^ ((set as u64) << 40);
        if o.recipe == Recipe::Q8Plane {
            let (row_bytes, code_bytes) = self
                .q8_plane_geom(o.name)
                .expect("Q8Plane geometry is checked at build");
            return gen_q8_plane(o.bytes, row_bytes, code_bytes, seed);
        }
        gen_bytes(o.recipe, o.bytes, seed, self.vocab())
    }

    /// `(row bytes, code bytes per row)` of the INT8 mirror plane operand `name`: `ka` has width `wa`, `vv`
    /// width `dv`; a row is `n_heads*(w+2)` bytes, its codes `n_heads*w`.
    fn q8_plane_geom(&self, name: &str) -> Result<(usize, usize)> {
        let w = match name {
            "ka" => "wa",
            "vv" => "dv",
            _ => {
                return Err(format!(
                    "operand {name}: the i8.mirror_plane_view recipe knows the planes `ka` (width wa) and `vv` (width dv) only"
                ))
            }
        };
        let get = |k: &str| {
            param_value(self.base.abi, &self.params, k).ok_or_else(|| {
                format!("operand {name}: the i8.mirror_plane_view recipe needs the param `{k}`")
            })
        };
        let (h, w) = (get("n_heads")? as usize, get(w)? as usize);
        Ok((h * (w + 2), h * w))
    }

    /// The explicit kernarg image (padded to 16 B) for `arm` with `addrs[i]` the base address of operand
    /// `i`, validated by `validate_invocation` against the contract geometry.
    pub fn kernarg_image(&self, arm: &ArmSpec, addrs: &[u64]) -> Result<Vec<u8>> {
        let binds: Vec<OperandBinding> = self
            .operands
            .iter()
            .zip(addrs)
            .map(|(o, a)| OperandBinding {
                address: *a,
                bytes: Some(o.bytes as u64),
                format: o.format,
            })
            .collect();
        let g = self.arm_geometry(arm)?;
        let inv = InvocationV1 {
            arch: self.arch,
            symbol: arm.export.symbol,
            params: &self.params,
            operands: &binds,
            grid: g.grid,
            block: g.block,
            dynamic_lds: g.lds,
        };
        let v = validate_invocation(arm.abi, &inv).map_err(|e| e.to_string())?;
        let mut img = v.explicit().to_vec();
        img.resize(img.len().next_multiple_of(16), 0);
        Ok(img)
    }

    fn arms(&self) -> impl Iterator<Item = &ArmSpec> {
        std::iter::once(&self.base).chain(self.cand.iter())
    }

    fn contract_json(arm: &ArmSpec) -> Value {
        json!({
            "id": arm.abi.id, "version": arm.abi.version, "digest": arm.abi.digest(),
            "numeric": arm.abi.numeric.0, "symbol": arm.export.symbol, "arch": arm.export.arch.name(),
            "module": match arm.export.object {
                ObjectRefV1::HipccSource { module, .. } | ObjectRefV1::Embedded { module, .. } => module,
            },
            "segment_bytes": arm.abi.segment_bytes, "explicit_bytes": arm.abi.explicit_bytes,
            "hidden_words": arm.abi.hidden.len(),
            "work_op_class": arm.work.map(|w| w.work.op_class),
            "admit": arm.work.map(|w| json!({
                "min_rows": w.work.admit.min_rows, "max_rows": w.work.admit.max_rows,
                "predicate": w.work.admit.predicate, "kill_env": w.work.admit.kill_env,
            })),
        })
    }
}

/// `--rowclasses` plan of a contract node, resolved without a shape (GPU-free).
#[derive(Clone, Debug)]
pub struct RowPlan {
    pub rows_param: String,
    pub admit_base: RowDomainV1,
    pub admit_cand: RowDomainV1,
    pub llc_bytes: u64,
}

/// The row-class plan of `--node name [--vs vs] [--arch arch]`: the activation-row key and both arms'
/// admitted row domains.
pub fn row_plan(name: &str, vs: Option<&str>, arch: Option<&str>) -> Result<RowPlan> {
    let spec = name
        .strip_prefix("contract:")
        .ok_or_else(|| format!("--rowclasses: `{name}` is not a contract: node"))?;
    let arch = match arch {
        None => Arch::Gfx1201,
        Some(a) => Arch::parse(a)
            .ok_or_else(|| format!("arch={a}: expected gfx1100, gfx1151 or gfx1201"))?,
    };
    let cand = resolve(spec, arch)?;
    let base = match vs {
        Some(v) => resolve(v.strip_prefix("contract:").unwrap_or(v), arch)?,
        None => cand,
    };
    let routed = cand
        .abi
        .operands
        .iter()
        .any(|o| recipe_for(o.format, o.access).is_ok_and(|r| r.is_routing()));
    let rows = cand.work.and_then(|w| w.work.rows).ok_or_else(|| {
        format!(
            "{}: the contract has no activation-row param (WorkV1.rows): nothing to sweep",
            cand.abi.id
        )
    })?;
    let rows_param = if cand.abi.param_index(rows).is_some() {
        rows.to_owned()
    } else if routed {
        "tokens".to_owned()
    } else {
        return Err(format!(
            "{}: rows param `{rows}` is not a contract param",
            cand.abi.id
        ));
    };
    let admit = |a: &ArmSpec| {
        a.work
            .map(|w| w.work.admit)
            .unwrap_or(hipfire_kernel_abi::work::ALL_ROWS)
    };
    Ok(RowPlan {
        rows_param,
        admit_base: admit(&base),
        admit_cand: admit(&cand),
        llc_bytes: llc_bytes(arch)?,
    })
}

/// Whether `rows` lies in `d`.
pub fn admits(d: &RowDomainV1, rows: u64) -> bool {
    rows >= d.min_rows && d.max_rows.is_none_or(|m| rows <= m)
}

/// Parse a `--rowclasses` list (`1,2,4`): positive, sorted, deduplicated.
pub fn parse_rowclasses(s: &str) -> Result<Vec<u64>> {
    let mut v = Vec::new();
    for t in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let r = t
            .parse::<u64>()
            .map_err(|e| format!("--rowclasses `{t}`: {e}"))?;
        if r == 0 {
            return Err("--rowclasses: a class of 0 rows".into());
        }
        v.push(r);
    }
    if v.is_empty() {
        return Err("--rowclasses: empty class list".into());
    }
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

fn parse_dims(key: &str, v: &str) -> Result<[u32; 3]> {
    let mut out = [1u32; 3];
    let parts: Vec<&str> = v.split([',', 'x']).collect();
    if parts.is_empty() || parts.len() > 3 {
        return Err(format!("hip_loop header {key}={v}: 1 to 3 components"));
    }
    for (o, p) in out.iter_mut().zip(&parts) {
        *o = p
            .parse::<u32>()
            .map_err(|e| format!("hip_loop header {key}={v}: {e}"))?;
        if *o == 0 {
            return Err(format!("hip_loop header {key}={v}: a zero component"));
        }
    }
    Ok(out)
}

impl Node for ContractNode {
    /// `contract` for every contract node (the spec is in the record's `contract`; no extent cases rebuild a
    /// node by name, see [`Node::extent_shapes`]).
    fn name(&self) -> &'static str {
        "contract"
    }
    fn arch(&self) -> &'static str {
        self.arch.name()
    }
    fn shape_label(&self) -> String {
        self.label.clone()
    }
    fn shape_json(&self) -> Value {
        let params: BTreeMap<&str, u64> = self
            .base
            .abi
            .params
            .iter()
            .map(|p| p.name)
            .zip(self.params.iter().copied())
            .collect();
        let ops: Vec<Value> = self
            .operands
            .iter()
            .map(|o| {
                json!({"name": o.name, "format": o.format.0, "access": o.access.metadata_name(),
                    "recipe": format!("{:?}", o.recipe), "bytes": o.bytes})
            })
            .collect();
        let mut v = json!({
            "params": params, "operands": ops, "arch": self.arch.name(),
            "extent": "small emu shape over the full grid only (no production-extent cases for a contract node)",
        });
        if self.prefix > 0 {
            v["prefix"] = json!(self.prefix);
        }
        if let Some(r) = &self.routing {
            v["routing"] = json!({
                "tokens": self.tokens, "topk": self.topk, "n_experts": self.n_experts(),
                "route": self.info_route(), "route_seed": self.route_seed, "live_rows": r.live,
                "live_tiles": r.live_tiles(), "touched_experts": r.touched_experts(),
                "max_tiles_per_expert": r.max_tiles_per_expert(), "partial_tiles": r.partial_tiles(),
                "slab_bytes": self.slab_bytes(),
            });
        }
        if self.pool_active {
            v["timing_pool"] = crate::cold::pool_json(self.pool_sets, self.pool_set_bytes);
        }
        v
    }
    fn record_extra(&self) -> Value {
        let mut contract = Self::contract_json(self.cand_arm());
        if let Some(n) = &self.resolved_from {
            contract["resolved_from"] = json!(n);
        }
        let mut v = json!({
            "contract": contract,
            "rows_param": self.rows_key.as_ref().map(|r| r.0.clone()),
            "bytes_per_call": self.bytes_per_call,
            "footprint_bytes": self.footprint_bytes,
            "regime": self.regime(),
            "llc_bytes": llc_bytes(self.arch).ok(),
        });
        if self.cand.is_some() {
            v["contract_vs"] = Self::contract_json(&self.base);
        }
        v
    }
    fn incumbent(&self) -> Result<Incumbent> {
        self.arm_incumbent(&self.base)
    }
    fn candidate_incumbent(&self) -> Result<Option<Incumbent>> {
        self.cand
            .as_ref()
            .map(|c| self.arm_incumbent(c))
            .transpose()
    }
    fn geometry_with(&self, hdr: &mut Header) -> Result<Geometry> {
        let mut g = self.arm_geometry(self.cand_arm())?;
        if let Some(v) = hdr.kv.remove("grid") {
            g.grid = parse_dims("grid", &v)?;
        }
        if let Some(v) = hdr.kv.remove("block") {
            g.block = parse_dims("block", &v)?;
            let threads: u64 = g.block.iter().map(|b| u64::from(*b)).product();
            if threads > 1024 || threads % 32 != 0 {
                return Err(format!(
                    "hip_loop header: block={v} is {threads} threads, need a multiple of 32 up to 1024"
                ));
            }
        }
        g.lds = hdr.take_usize("lds_dynamic", g.lds as usize)? as u32;
        hdr.no_leftover()?;
        if g.lds > 64 * 1024 {
            return Err(format!(
                "hip_loop header: lds_dynamic={} exceeds the 64 KiB workgroup LDS",
                g.lds
            ));
        }
        Ok(g)
    }
    /// The production launch of the BASELINE arm.
    fn geometry(&self) -> Geometry {
        self.arm_geometry(&self.base)
            .expect("the contract geometry evaluates at a built shape")
    }
    fn work(&self) -> Work {
        Work {
            flops: self.work_flops,
            bytes: self.bytes_per_call as f64,
        }
    }
    fn extent_full_default(&self) -> bool {
        true
    }
    fn emu_case(&self, seed: u64) -> Result<EmuCase> {
        let e = self.n_experts();
        let table = self
            .operands
            .iter()
            .position(|o| o.recipe == Recipe::ExpertPtrTable);
        let outs: Vec<usize> = (0..self.operands.len())
            .filter(|&i| self.operands[i].access != Access::ReadOnly)
            .collect();
        // Regions in mapping order: expert slabs, then every non-output operand in operand order, then the
        // outputs (one region each, or one packed region with poison pads when several).
        let mut regions: Vec<(&'static str, Vec<u8>)> = Vec::new();
        let split = e <= EMU_SPLIT;
        let slab_bytes = self.slab_bytes();
        if table.is_some() {
            if split {
                for x in 0..e {
                    regions.push((SLAB_NAMES[x], self.host_slab(seed, 0, x)));
                }
            } else {
                let mut w = Vec::with_capacity(e * slab_bytes);
                for x in 0..e {
                    w.extend_from_slice(&self.host_slab(seed, 0, x));
                }
                regions.push(("w", w));
            }
        }
        let n_slab_regions = regions.len();
        let mut region_of: Vec<Option<usize>> = vec![None; self.operands.len()];
        for i in 0..self.operands.len() {
            if outs.contains(&i) {
                continue;
            }
            region_of[i] = Some(regions.len());
            regions.push((self.operands[i].name, self.host_operand(i, seed, 0)));
        }
        let mut packed_off: Vec<Option<u64>> = vec![None; self.operands.len()];
        let out_region;
        if outs.len() == 1 {
            out_region = regions.len();
            region_of[outs[0]] = Some(out_region);
            regions.push((
                self.operands[outs[0]].name,
                self.host_operand(outs[0], seed, 0),
            ));
        } else {
            let mut packed = Vec::new();
            let poison = || POISON.to_le_bytes().repeat(PAD_WORDS);
            for &i in &outs {
                packed_off[i] = Some(packed.len() as u64);
                packed.extend(self.host_operand(i, seed, 0));
                packed.extend(poison());
            }
            out_region = regions.len();
            for &i in &outs {
                region_of[i] = Some(out_region);
            }
            regions.push(("out_packed", packed));
        }
        // The pointer table holds the emulator addresses the slab regions will get.
        if let Some(t) = table {
            let sizes: Vec<usize> = regions.iter().map(|r| r.1.len()).collect();
            let bases = crate::moe::bump_bases(&sizes, crate::emu::GUARD_BYTES);
            let bytes: Vec<u8> = (0..e)
                .flat_map(|x| {
                    let a = if split {
                        bases[x]
                    } else {
                        bases[0] + (x * slab_bytes) as u64
                    };
                    a.to_le_bytes()
                })
                .collect();
            debug_assert!(n_slab_regions > 0);
            let r = region_of[t].expect("the table is an input operand");
            regions[r].1 = bytes;
        }
        let bufs: Vec<EmuBuf> = regions
            .into_iter()
            .map(|(name, bytes)| EmuBuf { name, bytes })
            .collect();
        let abi = self.base.abi;
        let mut args = Vec::with_capacity(abi.fields.len());
        for f in abi.fields {
            match f.kind {
                FieldKind::Pointer { operand } => {
                    let i = abi.operand_index(operand).expect("validated pointer field");
                    let r = region_of[i].expect("every operand has a region");
                    args.push(match packed_off[i] {
                        Some(off) => EmuArg::BufOff(r, off),
                        None => EmuArg::Buf(r),
                    });
                }
                FieldKind::Scalar { ty, value } => {
                    let v = eval(&value, abi, self.base.export, &self.params)?;
                    args.push(match ty {
                        ScalarTy::I32 => EmuArg::I32(v as i32),
                        ScalarTy::U32 => EmuArg::U32(v as u32),
                        ScalarTy::U64 => EmuArg::U64(v),
                    });
                }
            }
        }
        Ok(EmuCase {
            bufs,
            args,
            out: out_region,
        })
    }

    fn setup(&mut self, gpu: &Gpu, seed: u64) -> Result<()> {
        let n = self.operands.len();
        let table = self
            .operands
            .iter()
            .position(|o| o.recipe == Recipe::ExpertPtrTable);
        let mut sets: Vec<Set> = Vec::new();
        for s in 0..self.pool_sets {
            let mut bufs: Vec<Option<Buf>> = (0..n).map(|_| None).collect();
            let mut slabs = Vec::new();
            for i in 0..n {
                if s > 0 && !self.pooled(i) {
                    continue;
                }
                if Some(i) == table {
                    for x in 0..self.n_experts() {
                        slabs.push(Buf::from_bytes(gpu, &self.host_slab(seed, s, x))?);
                    }
                    let addrs: Vec<u8> = slabs
                        .iter()
                        .flat_map(|b| (b.ptr() as u64).to_le_bytes())
                        .collect();
                    bufs[i] = Some(Buf::from_bytes(gpu, &addrs)?);
                } else {
                    let data = self.host_operand(i, seed, s);
                    if data.len() != self.operands[i].bytes {
                        return Err(format!(
                            "operand {}: the recipe produced {} B, the contract extent is {} B",
                            self.operands[i].name,
                            data.len(),
                            self.operands[i].bytes
                        ));
                    }
                    bufs[i] = Some(Buf::from_bytes(gpu, &data)?);
                }
            }
            sets.push(Set { bufs, slabs });
        }
        let mut pristine = Vec::new();
        for (i, o) in self.operands.iter().enumerate() {
            if o.access != Access::ReadOnly {
                let b = Buf::alloc(gpu, o.bytes)?;
                b.copy_from(
                    gpu,
                    sets[0].bufs[i].as_ref().expect("set 0 holds every operand"),
                )?;
                pristine.push((i, b));
            }
        }
        self.st = Some(State { sets, pristine });
        Ok(())
    }
    fn reset(&mut self, gpu: &Gpu) -> Result<()> {
        let s = self.st()?;
        for (i, p) in &s.pristine {
            s.sets[0].bufs[*i]
                .as_ref()
                .expect("set 0 holds every operand")
                .copy_from(gpu, p)?;
        }
        Ok(())
    }
    fn kernarg(&self) -> Result<Vec<u8>> {
        let s = self.st()?;
        let addrs: Vec<u64> = s.sets[0]
            .bufs
            .iter()
            .map(|b| b.as_ref().expect("set 0 holds every operand").ptr() as u64)
            .collect();
        let img = self.kernarg_image(&self.base, &addrs)?;
        if let Some(c) = &self.cand {
            let other = self.kernarg_image(c, &addrs)?;
            if other != img {
                return Err(format!(
                    "--vs: {} and {} encode different kernarg images from the same operands",
                    self.base.abi.id, c.abi.id
                ));
            }
        }
        Ok(img)
    }
    fn select_set(&self, karg: &mut [u8], i: usize) {
        let Ok(s) = self.st() else { return };
        if s.sets.len() < 2 {
            return;
        }
        let set = &s.sets[i % s.sets.len()];
        for &(off, op) in &self.ptr_fields {
            if let Some(b) = &set.bufs[op] {
                if self.pooled(op) {
                    karg[off as usize..off as usize + 8]
                        .copy_from_slice(&(b.ptr() as u64).to_le_bytes());
                }
            }
        }
    }
    fn read_output(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let s = self.st()?;
        let mut v = Vec::new();
        for (i, _) in &s.pristine {
            v.extend(
                s.sets[0].bufs[*i]
                    .as_ref()
                    .expect("set 0 holds every operand")
                    .download(gpu)?,
            );
        }
        Ok(v)
    }
    fn teardown(&mut self, gpu: &Gpu) {
        if let Some(s) = self.st.take() {
            for set in s.sets {
                for b in set.bufs.into_iter().flatten() {
                    b.free(gpu);
                }
                for b in set.slabs {
                    b.free(gpu);
                }
            }
            for (_, b) in s.pristine {
                b.free(gpu);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_kernel_abi::encode_into;

    fn build(name: &str, shape: &str) -> BResult<ContractNode> {
        ContractNode::build(
            name,
            &ShapeSpec::parse(Some(shape)).map_err(BuildError::Other)?,
        )
    }

    fn node(name: &str, shape: &str) -> ContractNode {
        build(name, shape).unwrap_or_else(|e| panic!("{name} {shape}: {e}"))
    }

    const BF16_TILED: &str = "contract:generic_gemm.bf16_xf32_tiled_rows.mpw16";
    const BF16_BATCHED: &str = "contract:generic_gemm.bf16_xf32_batched.wave_per_output";
    const Q8: &str = "contract:gemm_q8_0_wmma_gfx12";
    const QT44: &str = "contract:moe_grouped_qt44_a8_sym_gfx1201_nt1r1";
    const QT44_SHAPE: &str = "M=128,K=512,n_experts=8,topk=2,tokens=70";

    /// 4 KiB-aligned fake base addresses, 1 MiB apart.
    fn fake_addrs(n: &ContractNode) -> Vec<u64> {
        (0..n.operands.len() as u64)
            .map(|i| 0x1000_0000 + i * 0x10_0000)
            .collect()
    }

    #[test]
    fn the_three_acceptance_contracts_construct() {
        let f = node(BF16_TILED, "M=128,K=2560,N=512");
        assert_eq!(f.incumbent().unwrap().symbol, "gemm_bf16_xf32_tiled_rows");
        assert_eq!(f.incumbent().unwrap().module, "gemm_bf16_xf32_tiled_rows");
        assert_eq!(f.arch(), "gfx1201");
        assert_eq!(
            f.incumbent().unwrap().selection,
            "generic_gemm.bf16_xf32_tiled_rows.mpw16"
        );
        let q = node(Q8, "M=2048,K=2560,N=512");
        assert_eq!(q.incumbent().unwrap().symbol, "gemm_q8_0_wmma_gfx12");
        assert!(q
            .incumbent()
            .unwrap()
            .source
            .contains("gemm_q8_0_wmma_gfx12"));
        let t = node(QT44, QT44_SHAPE);
        assert_eq!(
            t.incumbent().unwrap().symbol,
            "moe_grouped_qt44_a8_sym_gfx1201_nt1r1"
        );
        assert_eq!(
            t.incumbent().unwrap().module,
            "moe_grouped_qt44_a8_sym_gfx12"
        );
        // the routed params come from the routing plan: 70 tokens x top-2 = 140 slots, 8 experts
        assert_eq!(t.params[t.base.abi.param_index("x_row_div").unwrap()], 2);
        assert_eq!(t.params[t.base.abi.param_index("x_src_rows").unwrap()], 70);
        assert_eq!(
            t.params[t.base.abi.param_index("m_total").unwrap()],
            crate::moe::m_total_bound(140, 8) as u64
        );
        assert_eq!(t.rows(), Some(("tokens", 70)));
        assert_eq!(q.rows(), Some(("N", 512)));
        // an abi id with several exports names them
        let e = build("contract:generic_gemm.moe_grouped_qt44_a8_sym", QT44_SHAPE)
            .err()
            .unwrap();
        assert!(e.to_string().contains("nt2r2"), "{e}");
        assert!(build("contract:nope", "M=1")
            .err()
            .unwrap()
            .to_string()
            .contains("unknown contract"));
    }

    #[test]
    fn geometry_evaluates_the_contract_grids() {
        let t = node(BF16_TILED, "M=128,K=2560,N=512");
        let g = t.geometry();
        assert_eq!((g.grid, g.block, g.lds), ([1, 128, 1], [256, 1, 1], 0));
        let b = node(BF16_BATCHED, "M=128,K=2560,N=512");
        assert_eq!(b.geometry().grid, [128, 512, 1]);
        assert_eq!(b.geometry().block, [32, 1, 1]);
        let q = node(Q8, "M=2048,K=2560,N=512");
        assert_eq!(q.geometry().grid, [128, 32, 1]);
        let t = node(QT44, QT44_SHAPE);
        let mt = crate::moe::m_total_bound(140, 8);
        assert_eq!(t.geometry().grid, [2, (mt / 16) as u32, 1]);
        assert_eq!(t.geometry().block, [128, 1, 1]);
        // M = 40 is two tiled-mpw16 128-row blocks' worth: ceil(M/128) = 1; N = 5 -> ceil(5/4) = 2
        assert_eq!(
            node(BF16_TILED, "M=300,K=64,N=5").geometry().grid,
            [3, 2, 1]
        );
    }

    #[test]
    fn candidate_header_overrides_and_refuses() {
        let n = node(BF16_BATCHED, "M=128,K=2560,N=512");
        let mut h = Header::parse("// hip_loop: grid=8,4,2 block=64,2 lds_dynamic=512").unwrap();
        let g = n.geometry_with(&mut h).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([8, 4, 2], [64, 2, 1], 512));
        let mut h = Header::parse("// hip_loop: grid=8x4").unwrap();
        assert_eq!(n.geometry_with(&mut h).unwrap().grid, [8, 4, 1]);
        let mut h = Header::default();
        assert_eq!(n.geometry_with(&mut h).unwrap(), n.geometry());
        for bad in [
            "tile_m=16",
            "grid=0,1,1",
            "grid=1,2,3,4",
            "grid=a",
            "block=100",
            "block=2048",
            "block=32,64,2",
            "lds_dynamic=70000",
            "waves=4",
        ] {
            let mut h = Header::parse(&format!("// hip_loop: {bad}")).unwrap();
            assert!(n.geometry_with(&mut h).is_err(), "{bad}");
        }
    }

    #[test]
    fn missing_params_and_unknown_keys_are_listed() {
        let e = build(BF16_BATCHED, "M=128,K=2560")
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("missing params") && e.contains('N'), "{e}");
        let e = build(BF16_BATCHED, "M=128,K=2560,N=1,Q=3")
            .err()
            .unwrap()
            .to_string();
        assert!(
            e.contains("unknown key `Q`") && e.contains("n_experts") == false,
            "{e}"
        );
        let e = build(QT44, "M=128,K=512,n_experts=8,tokens=4")
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("topk"), "{e}");
        let e = build(QT44, "M=128,K=512,n_experts=8,topk=2")
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("tokens"), "{e}");
        assert!(build(BF16_BATCHED, "M=128,K=2560,N=1,topk=2").is_err());
        // rows= is the alias of the row key
        assert_eq!(node(BF16_BATCHED, "M=8,K=64,rows=9").rows(), Some(("N", 9)));
        assert_eq!(
            node(QT44, "M=128,K=512,n_experts=8,topk=2,rows=140").rows(),
            Some(("tokens", 70))
        );
        assert!(build(QT44, "M=128,K=512,n_experts=8,topk=2,rows=141").is_err());
    }

    #[test]
    fn domain_violations_are_clippable_and_usage_errors_are_not() {
        // N = 0, M = 33 on the mp4 entry, M not a multiple of 64 on the qt44 entry
        assert!(matches!(
            build(BF16_BATCHED, "M=8,K=64,N=0"),
            Err(BuildError::Domain(_))
        ));
        assert!(matches!(
            build(
                "contract:generic_gemm.bf16_xf32_tiled_rows.mpw4",
                "M=33,K=64,N=4"
            ),
            Err(BuildError::Domain(_))
        ));
        assert!(matches!(
            build(QT44, "M=100,K=512,n_experts=8,topk=2,tokens=8"),
            Err(BuildError::Domain(_))
        ));
        assert!(matches!(
            build(Q8, "M=8,K=33,N=4"),
            Err(BuildError::Domain(_))
        ));
        assert!(matches!(
            build(BF16_BATCHED, "M=8,K=64"),
            Err(BuildError::Other(_))
        ));
    }

    #[test]
    fn launcher_rule_resolves_the_mp4_export_at_small_m() {
        let a = node("contract:gemm_bf16_xf32_tiled_rows", "M=17,K=10240,N=64");
        assert_eq!(
            a.incumbent().unwrap().symbol,
            "gemm_bf16_xf32_tiled_rows_mp4"
        );
        assert_eq!(a.base.abi.id, "generic_gemm.bf16_xf32_tiled_rows.mpw4");
        assert_eq!(a.geometry().grid, [1, 16, 1]);
        assert!(a.record_extra()["contract"]["resolved_from"].is_string());
        let b = node("contract:gemm_bf16_xf32_tiled_rows", "M=128,K=2560,N=64");
        assert_eq!(b.incumbent().unwrap().symbol, "gemm_bf16_xf32_tiled_rows");
        assert!(b.record_extra()["contract"]["resolved_from"].is_null());
        // the ABI id is exact
        let c = node(BF16_TILED, "M=17,K=10240,N=64");
        assert_eq!(c.incumbent().unwrap().symbol, "gemm_bf16_xf32_tiled_rows");
        // under --vs the symbol form resolves the same way on both sides
        let v = node(
            "contract:gemm_bf16_xf32_tiled_rows",
            "M=17,K=10240,N=64,vs=contract:gemm_bf16_xf32_batched",
        );
        assert_eq!(
            v.export_symbols(),
            ("gemm_bf16_xf32_batched", "gemm_bf16_xf32_tiled_rows_mp4")
        );
    }

    #[test]
    fn vs_makes_the_node_the_candidate_arm() {
        let n = node(
            BF16_TILED,
            "M=128,K=2560,N=512,vs=contract:generic_gemm.bf16_xf32_batched.wave_per_output",
        );
        assert_eq!(n.incumbent().unwrap().symbol, "gemm_bf16_xf32_batched");
        assert_eq!(n.incumbent().unwrap().module, "gemm_bf16_xf32_batched");
        let c = n.candidate_incumbent().unwrap().unwrap();
        assert_eq!(
            (c.symbol.as_str(), c.module.as_str()),
            ("gemm_bf16_xf32_tiled_rows", "gemm_bf16_xf32_tiled_rows")
        );
        // baseline geometry = the vs contract's, candidate (no header) = the node contract's
        assert_eq!(n.geometry().grid, [128, 512, 1]);
        assert_eq!(
            n.geometry_with(&mut Header::default()).unwrap().grid,
            [1, 128, 1]
        );
        assert_eq!(
            n.export_symbols(),
            ("gemm_bf16_xf32_batched", "gemm_bf16_xf32_tiled_rows")
        );
        let r = n.record_extra();
        assert_eq!(r["contract"]["symbol"], "gemm_bf16_xf32_tiled_rows");
        assert_eq!(r["contract_vs"]["symbol"], "gemm_bf16_xf32_batched");
        // both arms get the same explicit image
        let addrs = fake_addrs(&n);
        let a = n.kernarg_image(&n.base, &addrs).unwrap();
        let b = n.kernarg_image(n.cand.as_ref().unwrap(), &addrs).unwrap();
        assert_eq!(a, b);
        // a null node without vs has no candidate incumbent
        assert!(node(BF16_BATCHED, "M=8,K=64,N=4")
            .candidate_incumbent()
            .unwrap()
            .is_none());
    }

    #[test]
    fn signature_compat_refuses_a_different_layout() {
        for (a, b) in [
            (Q8, BF16_BATCHED),
            (BF16_BATCHED, Q8),
            (QT44, BF16_BATCHED),
            ("contract:gemv_q8_0", Q8),
        ] {
            let e = build(
                a,
                &format!("M=128,K=512,N=8,n_experts=8,topk=2,tokens=8,vs={b}"),
            )
            .err()
            .unwrap_or_else(|| panic!("{a} vs {b} accepted"));
            assert!(
                e.to_string().contains("not signature compatible")
                    || matches!(e, BuildError::Other(_)),
                "{e}"
            );
        }
        let e = build(
            Q8,
            "M=128,K=512,N=8,vs=contract:generic_gemm.bf16_xf32_batched.wave_per_output",
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("not signature compatible"), "{e}");
        // the same export is not a vs
        let e = build(
            BF16_BATCHED,
            "M=8,K=64,N=4,vs=contract:gemm_bf16_xf32_batched",
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("A/A"), "{e}");
        // tiled vs mpw4 (identical fields, different domain/geometry) is compatible
        assert!(signature_compat(
            &hipfire_kernel_abi::families::generic_gemm::GEMM_BF16_XF32_BATCHED,
            &hipfire_kernel_abi::families::generic_gemm::GEMM_BF16_XF32_TILED_ROWS_MP4
        )
        .is_ok());
    }

    #[test]
    fn kernarg_image_equals_encode_into() {
        for (name, shape) in [
            (BF16_TILED, "M=128,K=2560,N=512"),
            (BF16_BATCHED, "M=128,K=2560,N=512"),
            (Q8, "M=2048,K=2560,N=512"),
            (QT44, QT44_SHAPE),
        ] {
            let n = node(name, shape);
            let addrs = fake_addrs(&n);
            let img = n.kernarg_image(&n.base, &addrs).unwrap();
            // independent encode: validate the invocation by hand, encode_into a buffer
            let binds: Vec<OperandBinding> = n
                .operands
                .iter()
                .zip(&addrs)
                .map(|(o, a)| OperandBinding {
                    address: *a,
                    bytes: Some(o.bytes as u64),
                    format: o.format,
                })
                .collect();
            let g = n.geometry();
            let inv = InvocationV1 {
                arch: Arch::Gfx1201,
                symbol: n.base.export.symbol,
                params: &n.params,
                operands: &binds,
                grid: g.grid,
                block: g.block,
                dynamic_lds: g.lds,
            };
            let v = validate_invocation(n.base.abi, &inv).unwrap();
            let mut want = vec![0u8; n.base.abi.explicit_bytes as usize];
            encode_into(&v, &mut want).unwrap();
            assert_eq!(&img[..want.len()], &want[..], "{name}");
            assert_eq!(img.len() % 16, 0);
            assert!(img[want.len()..].iter().all(|b| *b == 0));
            // pointers first, in field order, at the operands' addresses
            for (off, op) in &n.ptr_fields {
                let o = *off as usize;
                assert_eq!(
                    u64::from_le_bytes(img[o..o + 8].try_into().unwrap()),
                    addrs[*op]
                );
            }
        }
        // tiled: 13 hidden words, segment 296, explicit 36 -> the explicit image alone (48 B padded)
        let t = node(BF16_TILED, "M=128,K=2560,N=512");
        assert_eq!(t.base.abi.segment_bytes, 296);
        assert_eq!(t.kernarg_image(&t.base, &fake_addrs(&t)).unwrap().len(), 48);
        // a wrong extent is refused by validate_invocation (a short operand binding)
        let mut short = node(BF16_BATCHED, "M=8,K=64,N=4");
        short.operands[1].bytes -= 4;
        let e = short
            .kernarg_image(&short.base, &fake_addrs(&short))
            .unwrap_err();
        assert!(e.contains("needs") && e.contains("bytes"), "{e}");
    }

    #[test]
    fn recipes_are_deterministic_finite_and_valid() {
        let q = node(Q8, "M=64,K=256,N=8");
        let a = q.host_operand(0, 7, 0);
        assert_eq!(a, q.host_operand(0, 7, 0));
        assert_ne!(a, q.host_operand(0, 8, 0));
        assert_ne!(a, q.host_operand(0, 7, 1), "pool sets differ");
        assert_eq!(a.len(), 64 * 8 * 34);
        for blk in a.chunks_exact(34) {
            let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]]));
            assert!(d.is_finite() && d.to_f32() > 0.0 && d.to_f32() < 0.03);
        }
        // X f16 finite; Y poisoned
        let x = q.host_operand(1, 7, 0);
        assert!(x
            .chunks_exact(2)
            .all(|h| half::f16::from_bits(u16::from_le_bytes([h[0], h[1]])).is_finite()));
        let y = q.host_operand(2, 7, 0);
        assert!(y
            .chunks_exact(4)
            .all(|w| f32::from_le_bytes(w.try_into().unwrap()) == SENTINEL));
        let f = node(BF16_BATCHED, "M=8,K=64,N=4");
        assert!(f
            .host_operand(0, 1, 0)
            .chunks_exact(4)
            .all(|w| f32::from_le_bytes(w.try_into().unwrap()).abs() <= 1.0));
        // routing tables: tiles in -1..E, sorted slots < slots, offsets E+1 ascending 16-aligned
        let t = node(QT44, QT44_SHAPE);
        let by = |nm: &str| t.operands.iter().position(|o| o.name == nm).unwrap();
        let tiles: Vec<i32> = t
            .host_operand(by("expert_tile_ids"), 1, 0)
            .chunks_exact(4)
            .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        assert!(tiles.iter().all(|&e| (-1..8).contains(&e)));
        let sorted: Vec<i32> = t
            .host_operand(by("sorted_slot_index"), 1, 0)
            .chunks_exact(4)
            .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        assert!(sorted.iter().all(|&s| (-1..140).contains(&s)));
        assert_eq!(sorted.iter().filter(|&&s| s >= 0).count(), 140);
        let offs: Vec<i32> = t
            .host_operand(by("expert_offsets"), 1, 0)
            .chunks_exact(4)
            .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        assert_eq!(offs.len(), 9);
        assert!(offs.windows(2).all(|w| w[0] <= w[1]) && offs.iter().all(|o| o % 16 == 0));
        // routing is a function of the shape, not the seed
        assert_eq!(
            t.host_operand(by("sorted_slot_index"), 1, 0),
            t.host_operand(by("sorted_slot_index"), 99, 0)
        );
        // block_i8: finite d, s = sum(qs)
        let xq = t.host_operand(by("Xq"), 3, 0);
        assert_eq!(xq.len(), (512 / 128) * 70 * 136);
        for r in xq.chunks_exact(136) {
            let d = f32::from_le_bytes(r[0..4].try_into().unwrap());
            let s = i32::from_le_bytes(r[4..8].try_into().unwrap());
            assert!(d.is_finite() && d > 0.0);
            assert_eq!(s, r[8..].iter().map(|b| i32::from(*b as i8)).sum::<i32>());
        }
        // expert slabs are valid symmetric qt44 groups
        let slab = t.host_slab(5, 0, 3);
        assert_eq!(slab.len(), 128 * 2 * 136);
        assert_eq!(slab, t.host_slab(5, 0, 3));
        assert_ne!(slab, t.host_slab(5, 0, 4));
        for g in slab.chunks_exact(136) {
            for half_i in 0..2 {
                let d =
                    half::f16::from_bits(u16::from_le_bytes([g[4 * half_i], g[4 * half_i + 1]]));
                let z = half::f16::from_bits(u16::from_le_bytes([
                    g[4 * half_i + 2],
                    g[4 * half_i + 3],
                ]));
                assert!(
                    d.is_finite() && z.to_f32() == half::f16::from_f32(-8.0 * d.to_f32()).to_f32()
                );
            }
        }
    }

    #[test]
    fn a_format_without_a_recipe_is_refused_by_name() {
        let e = recipe_for(FormatId("ggml.iq4_xs.136"), Access::ReadOnly).unwrap_err();
        assert!(e.contains("ggml.iq4_xs.136"), "{e}");
        // an output of any format is just poisoned
        assert_eq!(
            recipe_for(FormatId("ggml.iq4_xs.136"), Access::WriteOnly).unwrap(),
            Recipe::Poison
        );
        // a real contract whose operand format has no recipe: the build error names the FormatId
        let e = build("contract:gemv_mmvq_iq4_xs", "M=1,K=256").err();
        let _ = e;
        let mut found = false;
        for abi in hipfire_kernel_abi::families::all() {
            for ex in abi.exports.iter().filter(|e| e.arch == Arch::Gfx1201) {
                let bad = abi
                    .operands
                    .iter()
                    .find(|o| recipe_for(o.format, o.access).is_err());
                if let (Some(o), ObjectRefV1::HipccSource { .. }) = (bad, ex.object) {
                    let spec = format!("contract:{}", ex.symbol);
                    // every param 1: the format check precedes any domain check
                    let shape: Vec<String> = abi
                        .params
                        .iter()
                        .map(|p| format!("{}=256", p.name))
                        .collect();
                    if let Err(err) = build(&spec, &shape.join(",")) {
                        if err.to_string().contains(o.format.0) {
                            found = true;
                        }
                    }
                }
            }
        }
        assert!(found, "no contract refused by an unknown format name");
    }

    #[test]
    fn contract_sources_have_no_local_includes() {
        let mut n = 0;
        for abi in hipfire_kernel_abi::families::all() {
            for ex in abi.exports {
                if let ObjectRefV1::HipccSource { .. } = ex.object {
                    let (module, text, _) =
                        load_source(ex).unwrap_or_else(|e| panic!("{}: {e}", ex.symbol));
                    assert!(!module.is_empty() && !text.is_empty(), "{}", ex.symbol);
                    n += 1;
                }
            }
        }
        assert!(n > 20, "{n}");
    }

    #[test]
    fn work_counts_follow_the_work_spec() {
        let n = node(BF16_TILED, "M=128,K=2560,N=512");
        let w = n.work();
        assert_eq!(w.flops, 2.0 * 128.0 * 512.0 * 2560.0);
        assert_eq!(
            w.bytes,
            ((128 * 2560 * 2 + (512 * 2560 + 512 * 128) * 4) as f64)
        );
        assert_eq!(
            n.bytes_per_call(),
            128 * 2560 * 2 + (512 * 2560 + 512 * 128) * 4
        );
        let t = node(QT44, QT44_SHAPE);
        // MACs = routed rows * M * K; bytes = the work spec's lower bound
        assert_eq!(t.work().flops, 2.0 * 140.0 * 128.0 * 512.0);
        assert_eq!(
            t.bytes_per_call(),
            128 * 2 * 136 + 140 * 4 * 136 + 140 * 128 * 4
        );
        assert_eq!(n.record_extra()["rows_param"], "N");
        assert_eq!(t.record_extra()["rows_param"], "tokens");
    }

    #[test]
    fn qt44_nt2r1_routed_work_evaluates() {
        let n = node(
            "contract:moe_grouped_qt44_a8_sym_gfx1201_nt2r1",
            "M=2176,K=2560,n_experts=128,topk=6,tokens=512",
        );
        assert_eq!(n.work().flops, 2.0 * (512.0 * 6.0) * 2176.0 * 2560.0);
    }

    #[test]
    fn launch_rows_is_m_total_for_routed_and_rows_otherwise() {
        let q = node(QT44, "M=2176,K=2560,n_experts=128,topk=6,tokens=512");
        let mt = q.params()[q.base.abi.param_index("m_total").unwrap()];
        assert_eq!(q.launch_rows(), Some(mt));
        assert!(mt >= 512 * 6 && mt % 16 == 0);
        assert_ne!(q.launch_rows(), q.rows().map(|(_, v)| v));
        let b = node(BF16_BATCHED, "M=128,K=2560,N=512");
        assert_eq!(b.launch_rows(), Some(512));
        assert_eq!(b.launch_rows(), b.rows().map(|(_, v)| v));
    }

    #[test]
    fn regime_is_hot_unless_pool_and_above_the_llc() {
        let small = node(BF16_BATCHED, "M=128,K=2560,N=512,pool=1");
        assert!(small.bytes_per_call() < 64 << 20);
        assert_eq!((small.regime(), small.pool_sets()), ("hot", 1));
        let big = node(BF16_BATCHED, "M=128,K=2560,N=16384,pool=1");
        assert!(big.bytes_per_call() > 64 << 20);
        assert_eq!(big.regime(), "pool");
        assert!(big.pool_sets() >= 1);
        // the same shape without --pool stays hot (the promotion side refuses it above the LLC)
        let hot = node(BF16_BATCHED, "M=128,K=2560,N=16384");
        assert_eq!((hot.regime(), hot.pool_sets()), ("hot", 1));
        assert_eq!(
            hot.record_extra()["bytes_per_call"],
            big.record_extra()["bytes_per_call"]
        );
        // gfx1100's cache is 96 MiB: the same shape is hot there under --pool at N = 16384? (168 MB > 96)
        assert_eq!(llc_bytes(Arch::Gfx1100).unwrap(), 96 << 20);
        assert_eq!(llc_bytes(Arch::Gfx1201).unwrap(), 64 << 20);
        assert!(llc_bytes(Arch::Gfx1151).is_err());
        // a routed contract's work-spec bytes are a one-slab lower bound: the launch streams every touched
        // expert slab, so `--pool` pools by the footprint (one set: a set alone exceeds the cold-pool threshold)
        let q = node(QT44, "M=2176,K=2560,n_experts=128,topk=6,tokens=512,pool=1");
        assert!(q.bytes_per_call() < 64 << 20);
        assert!(q.footprint_bytes > 64 << 20);
        assert_eq!((q.regime(), q.pool_sets()), ("pool", 1));
        assert_eq!(q.record_extra()["footprint_bytes"], q.footprint_bytes);
        let q = node(QT44, "M=2176,K=2560,n_experts=128,topk=6,tokens=512");
        assert_eq!(q.regime(), "hot");
    }

    #[test]
    fn row_plan_and_class_helpers() {
        let p = row_plan(BF16_TILED, Some("contract:gemm_bf16_xf32_batched"), None).unwrap();
        assert_eq!(p.rows_param, "N");
        assert_eq!((p.admit_cand.min_rows, p.admit_base.min_rows), (512, 1));
        assert!(
            admits(&p.admit_cand, 512) && !admits(&p.admit_cand, 511) && admits(&p.admit_base, 1)
        );
        let q = row_plan(QT44, None, None).unwrap();
        assert_eq!(q.rows_param, "tokens");
        assert_eq!(q.admit_cand.min_rows, 64);
        assert!(
            row_plan("contract:gemv_q8_0", None, None).is_err(),
            "no rows param"
        );
        assert!(row_plan("mq6_res", None, None).is_err());
        assert_eq!(parse_rowclasses("8, 1,4,4").unwrap(), vec![1, 4, 8]);
        assert!(
            parse_rowclasses("").is_err()
                && parse_rowclasses("0").is_err()
                && parse_rowclasses("x").is_err()
        );
        assert_eq!(DEFAULT_ROWCLASSES.len(), 14);
        assert!(DEFAULT_ROWCLASSES.windows(2).all(|w| w[0] < w[1]));
        // clipping: a class past the batched kernel's grid-y limit is a Domain error, not a fatal one
        assert!(matches!(
            build(BF16_BATCHED, "M=8,K=64,N=70000"),
            Err(BuildError::Domain(_))
        ));
        assert!(build(BF16_TILED, "M=8,K=64,N=70000").is_ok());
    }

    #[test]
    fn emu_case_has_the_contract_shape() {
        // f32: 3 pointers + M K N, output last, every operand its own exact-size region
        let n = node(BF16_BATCHED, "M=40,K=96,N=10");
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 6);
        assert_eq!(
            c.args[..3],
            [EmuArg::Buf(0), EmuArg::Buf(1), EmuArg::Buf(2)]
        );
        assert_eq!(
            c.args[3..],
            [EmuArg::I32(40), EmuArg::I32(96), EmuArg::I32(10)]
        );
        assert_eq!(c.out, 2);
        assert_eq!(c.bufs[0].bytes.len(), 40 * 96 * 2);
        assert_eq!(c.bufs[1].bytes.len(), 10 * 96 * 4);
        assert_eq!(c.bufs[2].bytes.len(), 10 * 40 * 4);
        assert_eq!(c.bufs[2].bytes, n.host_operand(2, 1, 0));
        let c2 = n.emu_case(1).unwrap();
        assert_eq!(c.bufs[0].bytes, c2.bufs[0].bytes);
        // the `emu` preset: ragged defaults, M the smallest the domain admits
        let e = node("contract:gemm_bf16_xf32_tiled_rows", "emu");
        assert_eq!(e.params, vec![32, 256, 37]);
        assert_eq!(
            e.incumbent().unwrap().symbol,
            "gemm_bf16_xf32_tiled_rows_mp4"
        );
        let q = node(Q8, "emu");
        assert_eq!(q.params, vec![32, 256, 37]);
        // qt44: the pointer table holds the emulator's addresses of the slab regions
        let t = node(QT44, QT44_SHAPE);
        let c = t.emu_case(2).unwrap();
        assert_eq!(c.args.len(), 12);
        assert_eq!(
            c.bufs[..8].iter().map(|b| b.name).collect::<Vec<_>>(),
            SLAB_NAMES[..8].to_vec()
        );
        let tbl = &c.bufs[8];
        assert_eq!(tbl.name, "expert_weight_ptrs");
        let sizes: Vec<usize> = c.bufs.iter().map(|b| b.bytes.len()).collect();
        let want = crate::moe::bump_bases(&sizes, crate::emu::GUARD_BYTES);
        for e in 0..8 {
            assert_eq!(
                u64::from_le_bytes(tbl.bytes[8 * e..8 * e + 8].try_into().unwrap()),
                want[e]
            );
        }
        let mut mem = peacemaker_ir::emu::Memory::default();
        for (i, b) in c.bufs.iter().enumerate() {
            let a = mem
                .map_guarded(b.name.to_owned(), b.bytes.clone(), crate::emu::GUARD_BYTES)
                .unwrap();
            assert_eq!(a, want[i], "{}", b.name);
        }
        assert_eq!(c.bufs[c.out].name, "Y");
        // the emu preset on the routed contract
        let e = node("contract:moe_grouped_qt44_a8_sym_gfx1201_nt1r2", "emu");
        assert_eq!((e.params[0], e.rows()), (128, Some(("tokens", 37))));
    }

    const FLASH: &str = "contract:latent_attention_flash_prefill";
    /// The Raon expanded-mirror layout (kb the latent cache's shared 64-wide tail: token stride 576,
    /// head stride 0), 2 heads.
    const FLASH_SHAPE: &str = "n_heads=2,wa=128,wb=64,dv=128,pos_offset=0,max_rows=200,ka_tok=256,\
        ka_head=128,kb_tok=576,kb_head=0,v_tok=256,v_head=128,scale_bits=1033096506,round_y=1";

    fn flash(extra: &str) -> String {
        format!("{FLASH_SHAPE},{extra}")
    }

    #[test]
    fn flash_view_and_position_recipes() {
        let n = node(FLASH, &flash("n_rows=48,prefix=100"));
        let by_name = |s: &str| n.operands.iter().position(|o| o.name == s).unwrap();
        for (name, extent) in [
            ("ka", ((200 - 1) * 256 + 128 + 128) * 2),
            ("kb", ((200 - 1) * 576 + 64) * 2),
            ("vv", ((200 - 1) * 256 + 128 + 128) * 2),
        ] {
            let i = by_name(name);
            assert_eq!(n.operands[i].recipe, Recipe::Bf16, "{name}");
            assert_eq!(n.operands[i].bytes, extent, "{name} extent");
            let b = n.host_operand(i, 3, 0);
            assert_eq!(b.len(), extent, "{name} fill covers the whole view extent");
            assert_eq!(b, n.host_operand(i, 3, 0), "{name} deterministic");
            assert!(b.chunks_exact(2).any(|c| c != [0, 0]), "{name} is filled");
            for c in b.chunks_exact(2) {
                let v = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                assert!(v.is_finite() && (-1.0..=1.0).contains(&v), "{name}: {v}");
            }
        }
        let p = by_name("pos_ptr");
        assert_eq!(n.operands[p].recipe, Recipe::DeviceScalar);
        assert_eq!(n.host_operand(p, 3, 0), 100i32.to_le_bytes().to_vec());
        // the default prefix is 0; the scalar is an index-like operand, never replicated into a pool
        let z = node(FLASH, &flash("n_rows=48"));
        assert_eq!(z.host_operand(p, 9, 0), 0i32.to_le_bytes().to_vec());
        assert!(n.operands[p].recipe.is_index());
        assert_eq!(n.shape_json()["prefix"], 100);
        assert!(z.shape_json().get("prefix").is_none());
        assert!(n.shape_label().ends_with("-prefix100"));
        // the scalar reaches the emulator case as its own region
        let c = n.emu_case(1).unwrap();
        let r = c.bufs.iter().find(|b| b.name == "pos_ptr").unwrap();
        assert_eq!(r.bytes, 100i32.to_le_bytes().to_vec());
    }

    #[test]
    fn flash_prefix_is_bounded_by_max_rows_and_counts_work() {
        // prefix + rows <= max_rows (a clippable domain error: --rowclasses drops the class)
        for shape in [
            flash("n_rows=48,prefix=153"),
            FLASH_SHAPE.replace("max_rows=200", "max_rows=48") + ",n_rows=48,prefix=1",
        ] {
            assert!(
                matches!(build(FLASH, &shape), Err(BuildError::Domain(_))),
                "{shape}"
            );
        }
        assert!(build(FLASH, &flash("n_rows=48,prefix=152")).is_ok());
        // the row class rebuilds with `rows=`: a class that no longer fits is clipped
        assert!(matches!(
            build(FLASH, &flash("prefix=152,rows=49")),
            Err(BuildError::Domain(_))
        ));
        assert!(build(FLASH, &flash("prefix=100,rows=100")).is_ok());
        // causal-triangle work with the prefix: H * (wa + wb + dv) * (rows * prefix + rows (rows + 1) / 2)
        let n = node(FLASH, &flash("n_rows=48,prefix=100"));
        let macs = 2.0 * 320.0 * (48.0 * 100.0 + 48.0 * 49.0 / 2.0);
        assert_eq!(n.work().flops, 2.0 * macs);
        assert_eq!(n.rows(), Some(("n_rows", 48)));
        // a prefix is meaningless without a position scalar; it never exceeds i32
        let e = build(BF16_BATCHED, "M=128,K=2560,N=512,prefix=4")
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("no position device scalar"), "{e}");
        let e = build(FLASH, &flash("n_rows=48,prefix=4294967296"))
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("i32"), "{e}");
    }

    const FLASH_Q8: &str = "contract:latent_attention_flash_prefill_q8";
    /// 8 heads: plane rows of 8 * 130 = 1040 B (a multiple of 16).
    const FLASH_Q8_SHAPE: &str =
        "n_heads=8,wa=128,wb=64,dv=128,pos_offset=0,max_rows=200,ka_tok=1040,\
        ka_head=128,kb_tok=576,kb_head=0,v_tok=1040,v_head=128,scale_bits=1033096506,round_y=1";

    fn flash_q8(extra: &str) -> String {
        format!("{FLASH_Q8_SHAPE},{extra}")
    }

    #[test]
    fn flash_q8_plane_and_view_recipes() {
        let n = node(FLASH_Q8, &flash_q8("n_rows=48,prefix=100"));
        let by_name = |s: &str| n.operands.iter().position(|o| o.name == s).unwrap();
        let extent = (200 - 1) * 1040 + 8 * 130;
        assert_eq!(extent, 200 * 1040);
        let mut planes = Vec::new();
        for name in ["ka", "vv"] {
            let i = by_name(name);
            assert_eq!(n.operands[i].recipe, Recipe::Q8Plane, "{name}");
            assert_eq!(n.operands[i].bytes, extent, "{name} extent");
            assert!(!n.operands[i].recipe.is_index());
            let b = n.host_operand(i, 3, 0);
            assert_eq!(b.len(), extent, "{name} fill covers the whole extent");
            assert_eq!(b, n.host_operand(i, 3, 0), "{name} deterministic");
            assert_ne!(b, n.host_operand(i, 3, 1), "{name} pool sets differ");
            for row in b.chunks_exact(1040) {
                let (codes, scales) = row.split_at(8 * 128);
                assert!(
                    codes.iter().all(|&c| (c as i8) >= -127),
                    "{name}: -128 code"
                );
                assert!(
                    codes.iter().any(|&c| (c as i8) > 100)
                        && codes.iter().any(|&c| (c as i8) < -100)
                );
                assert_eq!(scales.len(), 8 * 2);
                for c in scales.chunks_exact(2) {
                    let v = half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
                    assert!(
                        v.is_finite()
                            && v > 0.0
                            && (0.5 / 127.0 - 1e-5..=1.0 / 127.0 + 1e-5).contains(&v),
                        "{name}: scale {v}"
                    );
                }
            }
            planes.push(b);
        }
        assert_ne!(
            planes[0], planes[1],
            "the planes have different operand seeds"
        );
        let kb = by_name("kb");
        assert_eq!(n.operands[kb].recipe, Recipe::Bf16);
        assert_eq!(n.operands[kb].bytes, ((200 - 1) * 576 + 64) * 2);
        let p = by_name("pos_ptr");
        assert_eq!(n.operands[p].recipe, Recipe::DeviceScalar);
        assert_eq!(n.host_operand(p, 3, 0), 100i32.to_le_bytes().to_vec());
        // the emulator case maps the planes at their extent, the scale bytes after the codes of row 0
        let c = n.emu_case(1).unwrap();
        for name in ["ka", "vv"] {
            let r = c.bufs.iter().find(|b| b.name == name).unwrap();
            assert_eq!(r.bytes.len(), extent, "{name} region");
            let scale0 =
                half::f16::from_bits(u16::from_le_bytes([r.bytes[8 * 128], r.bytes[8 * 128 + 1]]))
                    .to_f32();
            assert!(
                scale0 > 0.0 && scale0 <= 1.0 / 127.0 + 1e-5,
                "{name}: {scale0}"
            );
            assert_eq!(r.bytes, n.host_operand(by_name(name), 1, 0));
        }
    }

    #[test]
    fn flash_q8_domain_geometry_and_work() {
        for shape in [
            flash_q8("n_rows=48").replace("ka_tok=1040", "ka_tok=1024"),
            flash_q8("n_rows=48").replace("v_tok=1040", "v_tok=1024"),
            flash_q8("n_rows=48").replace("ka_head=128", "ka_head=256"),
            flash_q8("n_rows=48").replace("v_head=128", "v_head=256"),
            flash_q8("n_rows=48").replace("n_heads=8", "n_heads=2"),
            flash_q8("n_rows=48").replace("wa=128", "wa=64"),
            flash_q8("n_rows=48,prefix=153"),
        ] {
            assert!(
                matches!(build(FLASH_Q8, &shape), Err(BuildError::Domain(_))),
                "{shape}"
            );
        }
        // n_heads = 2 with tok 260 (the whole plane row) fails the 16-byte alignment rule too
        let s = flash_q8("n_rows=48")
            .replace("n_heads=8", "n_heads=2")
            .replace("1040", "260");
        assert!(
            matches!(build(FLASH_Q8, &s), Err(BuildError::Domain(_))),
            "{s}"
        );
        assert!(build(FLASH_Q8, &flash_q8("n_rows=48,prefix=152")).is_ok());
        assert!(matches!(
            build(FLASH_Q8, &flash_q8("prefix=152,rows=49")),
            Err(BuildError::Domain(_))
        ));
        let n = node(FLASH_Q8, &flash_q8("n_rows=100,prefix=100"));
        let bf = node(
            FLASH,
            &flash("n_rows=100,prefix=100").replace("n_heads=2", "n_heads=8"),
        );
        assert_eq!(n.work().flops, bf.work().flops);
        assert_eq!(n.rows(), Some(("n_rows", 100)));
        let (prefix, rows, h) = (100u64, 100u64, 8u64);
        assert_eq!(
            n.bytes_per_call(),
            4 * rows * h * 320 + (prefix + rows) * h * (130 + 130) + 2 * (prefix + rows) * h * 64
        );
        let g = n.arm_geometry(&n.base).unwrap();
        assert_eq!((g.grid, g.block, g.lds), ([2, 8, 1], [128, 1, 1], 61440));
        // both archs resolve
        let o = node(FLASH_Q8, &flash_q8("n_rows=48,arch=gfx1100"));
        assert_eq!(o.arch(), "gfx1100");
        assert_eq!(o.base.export.arch, Arch::Gfx1100);
        assert_eq!(o.base.export.symbol, "latent_attention_flash_prefill_q8");
        assert_eq!(o.arm_geometry(&o.base).unwrap().lds, 61440);
        assert_eq!(node(FLASH_Q8, &flash_q8("n_rows=48")).arch(), "gfx1201");
    }

    #[test]
    fn a_non_position_device_scalar_has_no_recipe() {
        // gated_delta_net: `slot_idx` / `n_dev` are i32.device_scalar operands, not positions
        let e = build(
            "contract:gated_delta_net_vector_serial_f32",
            "n_tokens=4,n_heads=2,qk_group=1,decay_per_key=0,round_out_bf16=0,slot=0,q_row_stride=256,\
             k_row_stride=256,v_row_stride=256,decay_row_stride=2,beta_row_stride=2,out_row_stride=256,\
             state_head_stride=16384,state_row_stride=128,state_slot_stride=32768,n_qk_heads=2",
        )
        .err()
        .unwrap()
        .to_string();
        assert!(
            e.contains("slot_idx") && e.contains("no input recipe"),
            "{e}"
        );
    }

    #[test]
    fn offline_arch_picks_the_arch_export() {
        let n = node("contract:gemm_f32_batched", "M=8,K=64,N=4,arch=gfx1100");
        assert_eq!(n.arch(), "gfx1100");
        assert_eq!(n.base.export.arch, Arch::Gfx1100);
        assert!(build("contract:gemm_q8_0_wmma_gfx12", "M=8,K=64,N=4,arch=gfx1100").is_err());
        assert!(build("contract:gemm_f32_batched", "M=8,K=64,N=4,arch=gfx900").is_err());
    }

    /// A synthetic `route_hist.v1` (E=8, top-2): layer 0 has 3 rows, layer 1 has 2.
    fn hist_fixture(tag: &str, bad_sha: bool) -> PathBuf {
        let rows: [[i32; 2]; 5] = [[0, 1], [1, 2], [7, 7 - 1], [3, 4], [1, 4]];
        let mut bin = Vec::new();
        for r in rows {
            for e in r {
                bin.extend_from_slice(&e.to_le_bytes());
            }
        }
        let dir = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .join("moe_hist_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        let bin_name = format!("{tag}.ids.bin");
        std::fs::write(dir.join(&bin_name), &bin).unwrap();
        let sha = if bad_sha {
            "0".repeat(64)
        } else {
            crate::elf::sha256_hex(&bin)
        };
        let doc = json!({
            "schema": "route_hist.v1", "n_experts": 8, "topk": 2, "id": "ab".repeat(32),
            "ids": {"path": bin_name, "sha256": sha},
            "records": [{"layer": 0, "n": 3, "k": 2}, {"layer": 1, "n": 2, "k": 2}],
        });
        let p = dir.join(format!("{tag}.json"));
        std::fs::write(&p, serde_json::to_vec(&doc).unwrap()).unwrap();
        p
    }

    #[test]
    fn route_hist_replays_recorded_experts_cycled() {
        let p = hist_fixture("det", false);
        let shape = format!(
            "M=128,K=512,n_experts=8,topk=2,tokens=7,route=hist:{}",
            p.display()
        );
        let n = node(QT44, &shape);
        let r = n.routing.as_ref().unwrap();
        // token t takes recorded row t % 5 (all layers, record order)
        assert_eq!(&r.topk[..4], &[0, 1, 1, 2]);
        assert_eq!(&r.topk[10..14], &[0, 1, 1, 2], "token 5 replays row 0");
        assert_eq!(r.topk.len(), 14);
        // expert 1 owns tokens 0,1,4,5 -> 5 slots... recount from the table itself
        let c1 = r.topk.iter().filter(|&&e| e == 1).count();
        assert_eq!((r.offsets[2] - r.offsets[1]) as usize, c1.div_ceil(16) * 16);
        let rt = n.info_route();
        assert_eq!(
            (rt["kind"].as_str(), rt["id"].as_str()),
            (Some("hist"), Some("ab".repeat(32).as_str()))
        );
        assert!(rt["layer"].is_null());
        // `@1` selects layer 1's two rows: tokens alternate rows [3,4], [1,4]
        let n1 = node(QT44, &format!("{shape}@1"));
        assert_eq!(&n1.routing.as_ref().unwrap().topk[..4], &[3, 4, 1, 4]);
        assert_eq!(n1.info_route()["layer"], 1);
        // uniform stays the default
        let u = node(QT44, QT44_SHAPE);
        assert_eq!(u.info_route()["kind"], "uniform");
    }

    #[test]
    fn route_hist_refuses_mismatch_and_bad_sha() {
        let p = hist_fixture("ok", false);
        let at = |shape: &str| {
            build(QT44, &format!("{shape},route=hist:{}", p.display()))
                .err()
                .map(|e| e.to_string())
        };
        let e = at("M=128,K=512,n_experts=16,topk=2,tokens=7").unwrap();
        assert!(e.contains("n_experts"), "{e}");
        let e = at("M=128,K=512,n_experts=8,topk=3,tokens=7").unwrap();
        assert!(e.contains("topk"), "{e}");
        let e = build(
            QT44,
            &format!(
                "M=128,K=512,n_experts=8,topk=2,tokens=7,route=hist:{}@9",
                p.display()
            ),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("no records for layer 9"), "{e}");
        let b = hist_fixture("bad", true);
        let e = build(
            QT44,
            &format!(
                "M=128,K=512,n_experts=8,topk=2,tokens=7,route=hist:{}",
                b.display()
            ),
        )
        .err()
        .unwrap()
        .to_string();
        assert!(e.contains("sha256 mismatch"), "{e}");
        let e = build(QT44, "M=128,K=512,n_experts=8,topk=2,tokens=7,route=bogus")
            .err()
            .unwrap()
            .to_string();
        assert!(e.contains("route"), "{e}");
    }
}
