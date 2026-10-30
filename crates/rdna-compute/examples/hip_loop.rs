// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! hip_loop: the PM-KRSI judge for hipcc incumbents on gfx1100 (7900 XTX; the two lane nodes on gfx1201, R9700). A
//! candidate is a complete HIP source file that defines the incumbent's kernel
//! symbol with the identical signature and launch contract. Prints ONE JSON
//! object (schema 1) on stdout.
//!
//! ```text
//! hip_loop --node mq6_res|hc_mix|qsa_hg4|lane_proj|lane_wo|rda_serial|conv_rows|moe_k2 [--shape k=v,...] --cand FILE.hip
//!          [--offline] [--null-control aa] [--time N] [--rounds R] [--seed S] [--out DIR]
//! ```
//!
//! * Nodes: `mq6_res` (`gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds`; shapes
//!   `qkv|z|out|qg|ba`), `hc_mix` (`gemm_wmma_lds_64_64_32_64_k64_p`; shapes
//!   `mix|sel`), `qsa_hg4` (`indexed_attention_attention_q8_batched_hg4`). See
//!   `hip_loop/{mq6,hc,qsa}.rs` for the shape grammar and the production launch each
//!   one mirrors.
//! * Candidate header: lines `// hip_loop: key=value ...` in the candidate source
//!   choose its launch geometry (keys per node, e.g. `tile_m tile_n block
//!   lds_dynamic`; no header = the production launch exactly). Both arms'
//!   geometries are recorded; a geometry that does not cover the output is refused.
//!   Per-kernel compiler flags use the runtime's `// HIPFIRE_COMPILER_FLAGS: ...`.
//! * `--cand-object FILE.co [--cand-geometry "tile_m=.. tile_n=.. block=.. lds_dynamic=.."]`: a
//!   prebuilt candidate code object (ELF or offload bundle; PM twins, lifted objects) instead
//!   of a source. It must export the incumbent's symbol with the same explicit kernarg layout
//!   (AMDGPU metadata `.args`: offset, size, pointer-ness; refused with stage `abi` otherwise);
//!   `.text` sha256 and resources are reported as for a compiled candidate, static LDS comes
//!   from the object. The geometry string uses the `// hip_loop:` keys (default: production).
//! * `--base-object FILE.co`: use FILE as the incumbent instead of compiling the shipped source.
//!   `FILE.co.json` (written next to `DIR/baseline.co` by every run that compiles the incumbent)
//!   must certify the object: module, the sha256 of the shipped source it came from, the object
//!   sha256 and the launched kernel's `.text` sha256 and explicit kernarg layout. `--offline`
//!   also recompiles the production source and requires the same `.text` sha256; on the device the
//!   sidecar is trusted and nothing is compiled for the incumbent (`--null-control aa` then loads
//!   the object twice as separate modules, `--cand-object` makes the whole run compile-free).
//!   Every run that compiles writes `DIR/baseline.co` (+`.json`) and, for a `--cand` source,
//!   `DIR/candidate.co` (+`.json`) with `candidate_geometry_arg` (the header as a `--cand-geometry`
//!   string) in the record. hip_loop's own compiles never read the JIT hot cache; offline runs keep
//!   the runtime compiler's cache dir inside `--out` unless `HIPFIRE_KERNEL_CACHE`/`HIPFIRE_HOME`
//!   are set (then those are honoured), GPU runs leave the device runtime's cache alone.
//! * `--offline`: compile the incumbent (from the shipped source) and the
//!   candidate through rdna-compute's `KernelCompiler` for gfx1100 (the runtime
//!   JIT's own argv, module flags and scheduler profile), no device. The record has
//!   `baseline`/`candidate` = `{code_sha256 (the kernel symbol's .text), vgprs,
//!   sgprs, lds_bytes, scratch_bytes, spill, code_bytes, occupancy_waves_per_simd,
//!   geometry, ...}`.
//! * GPU mode (a device of the node's arch, gfx1100; the lane nodes gfx1201; anything else is refused before any timing):
//!   seeded valid inputs, an identity run (outputs reset from a pristine copy
//!   before every dispatch; bit-identical bytes required, `identical_to_baseline`),
//!   then `--rounds R` ABBA rounds of `--time N` launches per block, hipEvent
//!   timed per launch with outputs reset outside the interval. Per round ratio
//!   candidate/baseline, `ratio_median`, one-sided exact sign tests `p_faster`
//!   (P[>= faster rounds | fair coin], ties and the other sign count against) and
//!   `p_slower`. `--null-control aa`: the incumbent source compiled and loaded as
//!   a SEPARATE module is the candidate (no `--cand`). With `R` rounds the best
//!   possible p is `2^-R`: use R >= 11 for p < 1e-3.
//! * `--emu` (offline, no device; implies `--offline`; default shape `emu`): the PRE-CARD SAFETY GATE. Both
//!   arms' objects (`--base-object`/production compile, `--cand`/`--cand-object`) are lifted with
//!   `peacemaker_lift::lift_object` and EVERY workgroup of the small `emu` shape's grid is run on the
//!   PeaceMaker CPU emulator from the same host-side seeded inputs (`mq6_res` M=144 K=512 N=40, `hc_mix`
//!   M=144 K=256 N=24, `qsa_hg4` 3 rows at position 102; ragged tails, production head/dim constants).
//!   Each buffer is an exact-size emulator region with 1 MiB unmapped guards; the kernarg segment is its own
//!   region (explicit args at the object's offsets, hidden args from the arm's geometry). FAIL (exit 1,
//!   reason on stderr and in `emu_gate.reason`): any access outside a mapped buffer, LDS access beyond static +
//!   dynamic LDS, a read of an LDS byte never written in the workgroup, a write into an input buffer or the
//!   kernarg segment, a barrier deadlock, the instruction limit, an opcode without emulator semantics (named),
//!   an emulator panic, or output bytes not bit-identical to the emulated incumbent. EVERY arm runs under the
//!   emulator's gate model (`Emulator::hw_model`, no opt-out; `peacemaker-ir` `emu/hw.rs`): loads write their
//!   destination registers only when the matching `s_waitcnt` retires them (a read/write of a register still owed
//!   by a load is an `[in-flight]` failure naming both instructions); every LDS access is checked for cross-wave
//!   happens-before through retired accesses and `s_barrier` epochs (`[race]`); each workgroup runs under
//!   round-robin, run-ahead-until-blocked for every wave and seeded-random wave orders (the first 16 workgroups of an
//!   arm run all of them, later ones round-robin + a rotating run-ahead + one random) and its stores must be
//!   bit-identical across them (`[schedule]`). The receipt arms carry `schedules`. The gate model does NOT predict
//!   card safety for hardware-level events (R3Mq6Wp-02 passes it and still dropped the card). The receipt `emu_gate` has
//!   `pass, reason, shape, seed, instructions{baseline,candidate}, workgroups{..}, output_sha256{..},
//!   identical_to_baseline, guard_bytes, strict_lds, numerics_model, arms{..}`. The numerics are the gfx1151
//!   model (`Emulator::gfx1100_as_gfx1151`) for both arms: it proves memory safety and arm-vs-arm identity at
//!   the small shape, not gfx1100 hardware ulps. `--base-object` is trusted via its sidecar (no recompile).
//!   EXTENT CASES (`emu_gate.extent_cases`, on by default, `--no-extent` = dev only and not a pass): after the
//!   small shape passes, the same objects run at every production extent shape of the node
//!   (`mq6_res` qkv/z/out/qg/ba, `hc_mix` mix/sel, `qsa_hg4` pos=mid, pos=late, pos=late with a 262144-token
//!   K/V capacity and a far-end 16-row window of that cache) on exact-size guarded buffers, emulating only
//!   SAMPLED workgroups: every grid corner, the last tile of each dim beside the first/middle tiles of the
//!   others, eight seeded random ones, then the closure that makes the baseline and the candidate (different
//!   tilings) cover the same set of output elements. Those elements are compared bitwise; each arm must have
//!   written all of them and nothing outside its own workgroups' elements. A 32-bit address overflow that only
//!   appears at production extents is an unmapped access (or a diff) here. Receipt per case: label, shape, grids,
//!   sampled workgroup counts, elements compared, instructions, output sha256 of the compared elements.
//!
//! * LANE NODES (`lane_proj`, `lane_wo`: the A3B exact batch-route lane kernels of the R9700, gfx1201; see
//!   `hip_loop/{lane_proj,lane_wo,lane_common,lane_ref}.rs` and `hip_loop/incumbents/`). The node, not the tool,
//!   names the arch: these two are gfx1201 (offline compile for gfx1201, GPU mode refuses any other device),
//!   the three nodes above stay gfx1100 and unchanged. The incumbents are the shipped sources exactly as the
//!   runtime compiles them (`hip_loop/incumbents/{lane_proj,res_lanes}.hip`, `-DIU4_A4_CANDIDATES=2`, plus the
//!   source's own `// HIPFIRE_COMPILER_FLAGS:`); their objects reproduce the runtime's byte for byte (object
//!   sha256 68db168d... / 16f5f16d..., asserted by the tests). `lane_proj`: shape `group=dn|fa|moe,lanes=1..8`,
//!   header keys `block rows_per_wg wgs_per_wgp lds_per_lane grid_fixed`; `lane_wo`: shape `lanes=1..8,m=2048`,
//!   header keys `block rows_per_wg lds_dynamic`. Both take `flush=1` in the shape (or `--flush`): a 2 x 96 MiB
//!   device-to-device copy after every timed dispatch (outside the hipEvent interval), recorded in
//!   `shape_dims.flush` and `timing.flush`.
//!   TIMING cycles a pool of distinct seeded weight sets whose total exceeds 192 MiB (`shape_dims.timing_pool`;
//!   dispatch `i` uses set `i % sets`), the output reset outside the interval. IDENTITY runs, after the basic
//!   dispatch identity, the product gates' adversarial matrix through BOTH arms at the node's lane count and
//!   lanes 1,2,3,5,8 (`lane_wo`: 1..8) with outputs pre-poisoned (rows/lanes beyond `lanes` must stay
//!   untouched): `identity_matrix` = `{pass, cases, mismatches, nan_payload_diffs, guard_violations,
//!   cpu_ref_identical{baseline,candidate}, cpu_ref, failures}`. `lane_proj` uses the lab NaN policy (non-NaN
//!   words bit-equal, NaN meets NaN, payload diffs tallied); `lane_wo` requires RAW bits (the product gate does).
//!   `cpu_ref_identical` compares each arm with `lane_ref.rs` and is informational (loudly reported).
//!   `--emu`: small ragged shape over the FULL grid, then every production extent (lane_proj: every group x
//!   lanes {1,3,8}; lane_wo: lanes {1,5,8}, M = 2048) over the FULL grid of both arms (`--emu-extent
//!   full|sampled` overrides; `sampled` assumes the incumbent's contiguous row partition per workgroup).
//!   lane_proj packs its four y segments into one emulator region with unwritten poison pads between them, so
//!   a store past a segment fails. The candidate's static hazard checks (`peacemaker_lift::hazards_gfx11` +
//!   `peacemaker_ir::passes::hazards::analyze`) are part of the gate: any Required/Unchecked finding, any
//!   PM-IR obligation or an analysis error fails it (`emu_gate.hazards`).
//!
//! * `rda_serial` (`gated_delta_net_vector_serial_f32`, the generic gated delta rule kernel of the Raon RDA
//!   layers; `hip_loop/rda.rs` has the production launch, shape grammar `pp512|pp2048|tg|emu` +
//!   `t h qk_group per_key round slot slots dev arch=gfx1100|gfx1201`, header keys `rows_per_wg block
//!   lds_dynamic`). Judged on gfx1100 (7900 XTX) by default, `arch=gfx1201` (R9700) in the shape. The mutable
//!   outputs are the output AND the in-place state: identity compares both. `--emu` packs output, all state
//!   slots and poison pads into one region (a store into a pad or an untouched slot fails), runs the small
//!   `emu` shape over the full grid, then `pp512`, `pp512,dev=1` and a short slot-1/per-head-decay/no-round
//!   case (`t=64,slot=1,slots=2,per_key=0,round=0`) at sampled workgroups (`t=2048` costs ~9 min of emulation and
//!   `t=16384` hours, and neither reaches a 32-bit overflow: they are not in the default gate), plus
//!   the static hazard checks. E.g. `hip_loop --node rda_serial --shape pp512,arch=gfx1201 --emu --cand X.hip`.
//!
//! * `conv_rows` (`causal_conv_depthwise_silu_rows_f32`, the Raon depthwise causal short conv; `hip_loop/conv.rs`
//!   has the production launch, shape grammar `pp512|pp2048|tg|emu` + `n c k inplace silu round arch=gfx1100|gfx1201`,
//!   header keys `block ch_per_wg rows_per_wg lds_dynamic`). Outputs: `out` AND the (in-place) history; `--emu` packs
//!   both with poison pads into one region, `emu` shape over the full grid, then `pp512`, `n=4096,c=200` and
//!   `pp512,inplace=0` sampled, plus the static hazard checks.
//!
//! * `moe_k2` (`gemm_mq4g256v2_moe_grouped_wmma_k2`, Raon's hottest gfx1100 prefill kernel; `hip_loop/moe.rs` has the
//!   production call table, shape grammar `gu512|dn512|gu2048|dn2048|sg512|sd512|dg512|dd512|emu` + `tokens experts top_k
//!   M K route=uniform|skew skew route_seed`, header keys `rows_per_block slots_per_tile block lds_dynamic`). Seeded qt44
//!   experts behind a device pointer table, a seed-independent skewed top-k routing scattered like the product; one
//!   weight set per timing pool above the cold threshold. `--emu`: `emu` over the full grid, then the eight production
//!   shapes sampled (dead tiles are `PAD_OWNER`), plus the static hazard checks.
//!
//! Exit status: 0 ok; 1 identity failure (record still printed); 2 usage; 3 the
//! candidate (or the device) could not compile/launch (record printed with `error`).
//!
//! ```text
//! cargo build --release -p rdna-compute --features lab --example hip_loop
//! hip_loop --node mq6_res --shape qkv --cand copy_of_shipped.hip --offline
//! hip_loop --node lane_proj --shape group=dn,lanes=8 --cand copy_of_shipped.hip --offline
//! hip_loop --node lane_wo --emu --cand candidate.hip
//! hip_loop --node lane_proj --shape group=fa,lanes=8 --cand candidate.hip --flush --time 100 --rounds 15
//! ```
//! GPU-free tests: `cargo test --release -p rdna-compute --features lab --example hip_loop`.

#[path = "support/abba.rs"]
mod abba;
#[path = "support/cold.rs"]
mod cold;
#[path = "hip_loop/conv.rs"]
mod conv;
#[path = "hip_loop/elf.rs"]
mod elf;
#[path = "hip_loop/emu.rs"]
mod emu;
#[path = "hip_loop/hc.rs"]
mod hc;
#[path = "hip_loop/jit.rs"]
mod jit;
#[path = "hip_loop/lane_common.rs"]
mod lane_common;
#[path = "hip_loop/lane_proj.rs"]
mod lane_proj;
#[path = "hip_loop/lane_ref.rs"]
mod lane_ref;
#[path = "hip_loop/lane_wo.rs"]
mod lane_wo;
#[path = "hip_loop/moe.rs"]
mod moe;
#[path = "hip_loop/mq6.rs"]
mod mq6;
#[path = "hip_loop/node.rs"]
mod node;
#[path = "hip_loop/qsa.rs"]
mod qsa;
#[path = "hip_loop/rda.rs"]
mod rda;

use node::{err, Geometry, Header, Incumbent, Node, Result, ShapeSpec, ARCH};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const SCHEMA: u32 = 1;
const DEFAULT_TIME: usize = 50;
const DEFAULT_ROUNDS: usize = 12;
const NODES: [&str; 8] = [
    "mq6_res",
    "hc_mix",
    "qsa_hg4",
    "lane_proj",
    "lane_wo",
    "rda_serial",
    "conv_rows",
    "moe_k2",
];
const USAGE: &str = "--node mq6_res|hc_mix|qsa_hg4|lane_proj|lane_wo|rda_serial|conv_rows|moe_k2 [--shape k=v,...] \
(--cand FILE.hip | --cand-object FILE.co [--cand-geometry \"tile_m=.. tile_n=.. block=.. lds_dynamic=..\"] | --null-control aa) \
[--base-object FILE.co] [--offline] [--emu [--no-extent] [--emu-extent full|sampled]] [--flush] \
[--identity full|smoke|skip | --identity-only] [--time N] [--rounds R] [--seed S] [--out DIR]";

// ---------------------------------------------------------------------------
// CLI

#[derive(Debug, Clone, Default)]
struct Opts {
    node: String,
    shape: Option<String>,
    cand: Option<PathBuf>,
    offline: bool,
    emu: bool,
    no_extent: bool,
    /// `--flush` (lane nodes): a 2 x 96 MiB device copy between timed dispatches.
    flush: bool,
    /// `--emu-extent full|sampled`: how the extent cases run (default: the node's choice).
    emu_extent: Option<String>,
    /// `--identity full|smoke|skip` (default full): the device identity depth (see `gpu_run`).
    identity: Option<String>,
    /// `--identity-only`: run the node's adversarial matrix over all its shapes, no timing.
    identity_only: bool,
    null_control: bool,
    cand_object: Option<PathBuf>,
    base_object: Option<PathBuf>,
    cand_geometry: Option<String>,
    time: usize,
    rounds: usize,
    seed: u64,
    out: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Opts> {
    let mut o = Opts {
        node: String::new(),
        shape: None,
        cand: None,
        offline: false,
        emu: false,
        no_extent: false,
        null_control: false,
        cand_object: None,
        flush: false,
        emu_extent: None,
        identity: None,
        identity_only: false,
        cand_geometry: None,
        base_object: None,
        time: DEFAULT_TIME,
        rounds: DEFAULT_ROUNDS,
        seed: 1,
        out: None,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| -> Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("--{name} needs a value"))
        };
        match a.as_str() {
            "--node" => o.node = value("node")?,
            "--shape" => o.shape = Some(value("shape")?),
            "--cand" => o.cand = Some(PathBuf::from(value("cand")?)),
            "--cand-object" => o.cand_object = Some(PathBuf::from(value("cand-object")?)),
            "--base-object" => o.base_object = Some(PathBuf::from(value("base-object")?)),
            "--cand-geometry" => o.cand_geometry = Some(value("cand-geometry")?),
            "--offline" => o.offline = true,
            "--emu" => {
                o.emu = true;
                o.offline = true;
            }
            "--no-extent" => o.no_extent = true,
            "--flush" => o.flush = true,
            "--identity" => {
                let v = value("identity")?;
                if !["full", "smoke", "skip"].contains(&v.as_str()) {
                    return Err(format!(
                        "--identity {v}: expected `full`, `smoke` or `skip`"
                    ));
                }
                o.identity = Some(v);
            }
            "--identity-only" => o.identity_only = true,
            "--emu-extent" => {
                let v = value("emu-extent")?;
                if v != "full" && v != "sampled" {
                    return Err(format!("--emu-extent {v}: expected `full` or `sampled`"));
                }
                o.emu_extent = Some(v);
            }
            "--null-control" => {
                let v = value("null-control")?;
                if v != "aa" {
                    return Err(format!("--null-control {v}: the only null control is `aa`"));
                }
                o.null_control = true;
            }
            "--time" => o.time = abba::positive_count(&value("time")?)?,
            "--rounds" => o.rounds = abba::positive_count(&value("rounds")?)?,
            "--seed" => o.seed = value("seed")?.parse::<u64>().map_err(err)?,
            "--out" => o.out = Some(PathBuf::from(value("out")?)),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if !NODES.contains(&o.node.as_str()) {
        return Err(format!("--node must be one of {}", NODES.join(", ")));
    }
    let sources = usize::from(o.cand.is_some())
        + usize::from(o.cand_object.is_some())
        + usize::from(o.null_control);
    if sources != 1 {
        return Err(
            "exactly one of --cand FILE.hip, --cand-object FILE.co, --null-control aa is required"
                .into(),
        );
    }
    if o.identity_only && (o.offline || o.emu) {
        return Err("--identity-only runs on the device (not with --offline/--emu)".into());
    }
    if o.identity_only && o.identity.as_deref().is_some_and(|m| m != "full") {
        return Err("--identity-only runs the full matrix: it cannot be combined with --identity smoke|skip".into());
    }
    if o.cand_geometry.is_some() && o.cand_object.is_none() {
        return Err("--cand-geometry needs --cand-object (a source candidate carries a `// hip_loop:` header)".into());
    }
    abba::timing_counts(o.time, o.rounds)?;
    Ok(o)
}

fn build_node(name: &str, shape: Option<&str>) -> Result<Box<dyn Node>> {
    let spec = ShapeSpec::parse(shape)?;
    match name {
        "mq6_res" => Ok(Box::new(mq6::Mq6Res::new(&spec)?)),
        "hc_mix" => Ok(Box::new(hc::HcMix::new(&spec)?)),
        "qsa_hg4" => Ok(Box::new(qsa::QsaHg4::new(&spec)?)),
        "lane_proj" => Ok(Box::new(lane_proj::LaneProj::new(&spec)?)),
        "lane_wo" => Ok(Box::new(lane_wo::LaneWo::new(&spec)?)),
        "rda_serial" => Ok(Box::new(rda::Rda::new(&spec)?)),
        "conv_rows" => Ok(Box::new(conv::ConvRows::new(&spec)?)),
        "moe_k2" => Ok(Box::new(moe::MoeK2::new(&spec)?)),
        other => Err(format!("unknown node `{other}`")),
    }
}

// ---------------------------------------------------------------------------
// Records

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Ok,
    IdentityFailed,
    Unusable,
}

impl Outcome {
    fn code(&self) -> i32 {
        match self {
            Outcome::Ok => 0,
            Outcome::IdentityFailed => 1,
            Outcome::Unusable => 3,
        }
    }
}

fn fail(mut rec: Value, stage: &str, message: impl std::fmt::Display) -> (Value, Outcome) {
    rec["error"] = json!({"stage": stage, "message": message.to_string()});
    (rec, Outcome::Unusable)
}

fn geo_json(g: &Geometry) -> Value {
    json!({
        "grid_workgroups": g.grid,
        "block": g.block,
        "lds_dynamic_bytes": g.lds,
        "grid_work_items": g.threads(),
    })
}

fn arm_json(b: &jit::Built, symbol: &str, geo: &Geometry) -> Result<Value> {
    let mut v = b.kernel(symbol)?.json();
    v["geometry"] = geo_json(geo);
    v["object_sha256"] = json!(b.image_sha256);
    v["object_bytes"] = json!(b.image.len());
    v["object_path"] = json!(b.image_path.display().to_string());
    v["source_path"] = json!(b.source_path.display().to_string());
    v["kernels_in_object"] = json!(b.kernels.len());
    v["hipcc_argv"] = json!(b.argv);
    Ok(v)
}

fn out_dir_of(o: &Opts) -> PathBuf {
    o.out
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join(format!("hip_loop-{}", std::process::id())))
}

/// Everything the offline stage produces.
struct Prep {
    inc: Incumbent,
    base: jit::Built,
    cand: jit::Built,
    base_geo: Geometry,
    cand_geo: Geometry,
    /// The candidate's `// hip_loop:` header (before the node consumed it), for the extent gate.
    cand_header: Header,
}

/// Resolve the candidate, its geometry, compile both arms, and fill the offline
/// part of the record.
fn prepare(
    o: &Opts,
    node: &dyn Node,
    workdir: &Path,
    rec: &mut Value,
) -> std::result::Result<Prep, (String, String)> {
    let stage = |s: &str| {
        let s = s.to_owned();
        move |e: String| (s.clone(), e)
    };
    let inc = node.incumbent().map_err(stage("incumbent"))?;
    let arch = node.arch();
    rec["symbol"] = json!(inc.symbol);
    rec["module"] = json!(inc.module);
    rec["selection"] = json!(inc.selection);
    // A source candidate (`--cand`), the incumbent's own source (`--null-control aa`) or
    // a prebuilt object (`--cand-object`, no source).
    let cand_source = match (&o.cand, &o.cand_object) {
        (Some(p), _) => Some(
            std::fs::read_to_string(p)
                .map_err(|e| ("args".to_owned(), format!("--cand {}: {e}", p.display())))?,
        ),
        (None, Some(_)) => None,
        (None, None) => Some(inc.source.clone()),
    };
    let mut header = match &cand_source {
        Some(s) => Header::parse(s),
        None => Header::parse_list(o.cand_geometry.as_deref().unwrap_or("")),
    }
    .map_err(stage("header"))?;
    rec["candidate_header"] = json!(header.kv);
    let header_kv = header.kv.clone();
    let cand_geo = node.geometry_with(&mut header).map_err(stage("geometry"))?;
    let base_geo = node.geometry();
    if o.null_control && cand_geo != base_geo {
        return Err(("geometry".into(), "null control geometry differs".into()));
    }
    if o.offline {
        jit::isolate_caches(workdir).map_err(stage("compile"))?;
    }
    // Compile only what is needed: the incumbent unless `--base-object` supplies it (offline
    // still recompiles it, to verify the object, except under `--emu`, where the sidecar-certified
    // object is trusted as on the device), the candidate unless it is a prebuilt object
    // or an A/A against `--base-object` on the device (both arms load that object).
    let aa_from_object = o.null_control && o.base_object.is_some() && !o.offline;
    let want_base = o.base_object.is_none() || (o.offline && !o.emu);
    let want_cand = cand_source.is_some() && !aa_from_object;
    let (compiled_base, compiled_cand) = std::thread::scope(|s| {
        let b = want_base.then(|| s.spawn(|| jit::compile_for(arch, workdir, "baseline", &inc, &inc.source)));
        let c = want_cand.then(|| {
            s.spawn(|| jit::compile_for(arch, workdir, "candidate", &inc, cand_source.as_deref().unwrap_or_default()))
        });
        let join = |h: Option<std::thread::ScopedJoinHandle<'_, Result<jit::Built>>>| -> Result<Option<jit::Built>> {
            match h {
                None => Ok(None),
                Some(h) => h.join().map_err(|_| "compile thread panicked".to_owned())?.map(Some),
            }
        };
        let (b, c) = (join(b), join(c));
        Ok::<_, String>((b?, c?))
    })
    .map_err(stage("compile"))?;
    let base = match &o.base_object {
        Some(p) => {
            let b = jit::from_object_for(arch, "baseline", p).map_err(stage("base_object"))?;
            rec["base_object"] = verify_base_object_for(arch, p, &inc, &b, compiled_base.as_ref())
                .map_err(stage("base_object"))?;
            b
        }
        None => compiled_base.ok_or(("compile".to_owned(), "internal: no baseline".to_owned()))?,
    };
    let cand = if let Some(p) = &o.cand_object {
        jit::from_object_for(arch, "candidate", p).map_err(stage("object"))?
    } else if let Some(c) = compiled_cand {
        c
    } else {
        let p = o
            .base_object
            .as_ref()
            .expect("an A/A without a candidate build has --base-object");
        jit::from_object_for(arch, "candidate", p).map_err(stage("base_object"))?
    };
    if o.cand_object.is_some() {
        // The object must be launchable with the incumbent's kernargs and fit the LDS.
        let (bk, ck) = (
            base.kernel(&inc.symbol).map_err(stage("compile"))?,
            cand.kernel(&inc.symbol).map_err(stage("object"))?,
        );
        if let Some(why) = ck.kernarg_mismatch(bk) {
            return Err(("abi".into(), format!("{}: {why}", inc.symbol)));
        }
        let lds = ck.lds_bytes + u64::from(cand_geo.lds);
        if lds > 64 * 1024 {
            return Err((
                "abi".into(),
                format!("static + dynamic LDS is {lds} B, over the 64 KiB workgroup limit"),
            ));
        }
        rec["candidate_object"] = json!({
            "path": o.cand_object.as_ref().map(|p| p.display().to_string()),
            "sha256": cand.image_sha256,
            "geometry_arg": o.cand_geometry,
            "kernarg_size_matches_incumbent": ck.kernarg_bytes == bk.kernarg_bytes,
        });
    }
    // The compiled objects, reusable on a card without recompiling: `--base-object
    // DIR/baseline.co`, `--cand-object DIR/candidate.co --cand-geometry <candidate_geometry_arg>`.
    let publish = |name: &str,
                   b: &jit::Built,
                   source: &str,
                   kv: &std::collections::BTreeMap<String, String>| {
        let co = workdir.join(name);
        let side = jit::sidecar_json_for(arch, &o.node, &inc, b, source, kv);
        jit::publish(&co, b, &side).map(|()| {
            json!({"path": co.display().to_string(), "sha256": b.image_sha256,
                "sidecar": jit::sidecar_path(&co).display().to_string()})
        })
    };
    if o.base_object.is_none() {
        rec["baseline_co"] = publish("baseline.co", &base, &inc.source, &Default::default())
            .map_err(stage("write"))?;
    }
    if let (true, Some(src)) = (want_cand && o.cand_object.is_none(), &cand_source) {
        rec["candidate_co"] =
            publish("candidate.co", &cand, src, &header_kv).map_err(stage("write"))?;
        let arg: Vec<String> = header_kv.iter().map(|(k, v)| format!("{k}={v}")).collect();
        rec["candidate_geometry_arg"] = json!(arg.join(" "));
    }
    rec["baseline"] = arm_json(&base, &inc.symbol, &base_geo).map_err(stage("compile"))?;
    if o.base_object.is_some() {
        rec["baseline"]["kind"] = json!("object");
    }
    let cand_stage = if o.cand_object.is_some() {
        "object"
    } else {
        "compile"
    };
    let ck = arm_json(&cand, &inc.symbol, &cand_geo).map_err(stage(cand_stage))?;
    rec["candidate"] = ck;
    if o.cand_object.is_some() {
        rec["candidate"]["kind"] = json!("object");
    }
    let same = rec["baseline"]["code_sha256"] == rec["candidate"]["code_sha256"];
    rec["identical_code"] = json!(same);
    rec["same_geometry"] = json!(base_geo == cand_geo);
    if o.null_control {
        rec["null_control"] = json!({
            "kind": "aa",
            "identical_code": same,
            "identical_objects": base.image_sha256 == cand.image_sha256,
            "baseline_object_sha256": base.image_sha256,
            "candidate_object_sha256": cand.image_sha256,
            "note": if aa_from_object {
                "both arms load --base-object as separate modules (no compile on the device)"
            } else {
                "the candidate is the incumbent source compiled separately and loaded as a separate module"
            },
        });
    }
    Ok(Prep {
        inc,
        base,
        cand,
        base_geo,
        cand_geo,
        cand_header: Header { kv: header_kv },
    })
}

/// `--base-object FILE.co` as the incumbent: its sidecar `FILE.co.json` (written by an
/// earlier `--offline --out DIR`) must certify the object, the shipped source it was
/// compiled from and the `.text` sha256 of the launched kernel. Offline, `recompiled` (the
/// production-source compile just made) must also agree; on the device the sidecar is trusted.
#[allow(dead_code)]
fn verify_base_object(
    path: &Path,
    inc: &Incumbent,
    b: &jit::Built,
    recompiled: Option<&jit::Built>,
) -> Result<Value> {
    verify_base_object_for(ARCH, path, inc, b, recompiled)
}

/// [`verify_base_object`] for the node's own `arch`.
fn verify_base_object_for(
    arch: &str,
    path: &Path,
    inc: &Incumbent,
    b: &jit::Built,
    recompiled: Option<&jit::Built>,
) -> Result<Value> {
    let side = jit::sidecar_path(path);
    let text = std::fs::read_to_string(&side).map_err(|e| {
        format!(
            "{}: {e} (write it with `hip_loop --offline --out DIR`, which produces DIR/baseline.co and DIR/baseline.co.json)",
            side.display()
        )
    })?;
    let sc: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", side.display()))?;
    if sc["tool"] != "hip_loop" || sc["schema"] != SCHEMA {
        return Err(format!(
            "{}: not a hip_loop schema {SCHEMA} sidecar",
            side.display()
        ));
    }
    if sc["module"] != inc.module.as_str() || sc["arch"] != arch {
        return Err(format!(
            "{}: sidecar is for module {} on {}, the incumbent is {} on {arch}",
            side.display(),
            sc["module"],
            sc["arch"],
            inc.module
        ));
    }
    let src_sha = elf::sha256_hex(inc.source.as_bytes());
    if sc["source_sha256"] != src_sha.as_str() {
        return Err(format!(
            "{}: the object was compiled from a different shipped source (sidecar {}, this build {src_sha}): stale object",
            path.display(), sc["source_sha256"]
        ));
    }
    if sc["object_sha256"] != b.image_sha256.as_str() {
        return Err(format!(
            "{}: the object changed after its sidecar was written",
            path.display()
        ));
    }
    let k = b.kernel(&inc.symbol)?;
    let sk = &sc["kernels"][&inc.symbol];
    if sk.is_null() || sk["code_sha256"] != k.code_sha256.as_str() {
        return Err(format!(
            "{}: `{}` .text sha256 {} does not match the sidecar's {}",
            path.display(),
            inc.symbol,
            k.code_sha256,
            sk["code_sha256"]
        ));
    }
    if sk["explicit_args"] != k.json()["explicit_args"] {
        return Err(format!(
            "{}: `{}` explicit kernel arguments differ from the sidecar's",
            path.display(),
            inc.symbol
        ));
    }
    if let Some(r) = recompiled {
        let fresh = &r.kernel(&inc.symbol)?.code_sha256;
        if *fresh != k.code_sha256 {
            return Err(format!(
                "{}: `{}` .text sha256 {} differs from a fresh compile of the production source ({fresh})",
                path.display(), inc.symbol, k.code_sha256
            ));
        }
    }
    let local = rdna_compute::KernelCompiler::local_toolchain_id();
    Ok(json!({
        "path": path.display().to_string(),
        "sha256": b.image_sha256,
        "sidecar": side.display().to_string(),
        "verified": if recompiled.is_some() { "sidecar + recompiled production source" } else { "sidecar (source sha256, object sha256, kernel .text sha256, kernel ABI)" },
        "source_sha256": src_sha,
        "toolchain_recorded": sc["toolchain_id"],
        "toolchain_local": local,
        "toolchain_matches": sc["toolchain_id"] == json!(local),
    }))
}

fn run(o: &Opts) -> (Value, Outcome) {
    let mut rec = json!({
        "schema": SCHEMA, "tool": "hip_loop", "node": o.node, "arch": ARCH,
        "shape": Value::Null, "symbol": Value::Null, "error": Value::Null,
        "offline": o.offline,
    });
    let mut shape = o.shape.clone().or_else(|| o.emu.then(|| "emu".to_owned()));
    if o.flush {
        if !o.node.starts_with("lane_") {
            return fail(
                rec,
                "args",
                "--flush is supported by the lane_proj / lane_wo nodes only",
            );
        }
        shape = Some(match shape {
            Some(s) => format!("{s},flush=1"),
            None => "flush=1".to_owned(),
        });
    }
    let mut node = match build_node(&o.node, shape.as_deref()) {
        Ok(n) => n,
        Err(e) => return fail(rec, "args", e),
    };
    rec["arch"] = json!(node.arch());
    rec["shape"] = json!(node.shape_label());
    rec["shape_dims"] = node.shape_json();
    let w = node.work();
    rec["work"] = json!({"flops": w.flops, "bytes": w.bytes});
    if let Value::Object(extra) = node.record_extra() {
        for (k, v) in extra {
            rec[k] = v;
        }
    }
    let workdir = out_dir_of(o);
    if let Err(e) = std::fs::create_dir_all(&workdir) {
        return fail(rec, "args", format!("--out {}: {e}", workdir.display()));
    }
    rec["out_dir"] = json!(workdir.display().to_string());
    let prep = match prepare(o, node.as_ref(), &workdir, &mut rec) {
        Ok(p) => p,
        Err((stage, e)) => return fail(rec, &stage, e),
    };
    if o.emu {
        return emu_stage(o, node.as_ref(), &prep, rec);
    }
    if o.offline {
        return (rec, Outcome::Ok);
    }
    gpu_stage(o, node.as_mut(), &prep, rec)
}

// ---------------------------------------------------------------------------
// Emulator gate (`--emu`)

/// The pre-card gate. (1) Both arms' objects lifted and emulated over the FULL small grid on the PeaceMaker
/// CPU emulator, output bytes bit-compared. (2) Unless `--no-extent`, the same objects at every PRODUCTION
/// extent shape of the node (`Node::extent_shapes`: judge shapes, capacity extremes), emulating only sampled
/// workgroups (grid corners, partial-dim edges, seeded random ones, plus the closure that makes both arms
/// cover the same output elements) on exact-size guarded buffers; bits compared on those elements. Both
/// parts must pass: a 32-bit address overflow that only shows at production extents cannot slip through
/// the small shape. `rec["emu_gate"]` is the receipt (`extent_cases` lists part 2); exit 1 on failure.
fn emu_stage(o: &Opts, node: &dyn Node, prep: &Prep, mut rec: Value) -> (Value, Outcome) {
    let mut case = match node.emu_case(o.seed) {
        Ok(c) => c,
        Err(e) => return fail(rec, "emu_case", e),
    };
    let (small_pass, mut receipt) = emu::gate(
        &prep.inc.symbol,
        &mut case,
        (&prep.base.image, &prep.base_geo),
        (&prep.cand.image, &prep.cand_geo),
        node.shape_json(),
        o.seed,
    );
    drop(case);
    receipt["small_pass"] = json!(small_pass);
    let mut pass = small_pass;
    let mut why: Vec<String> = Vec::new();
    if !small_pass {
        why.push(format!(
            "small shape: {}",
            receipt["reason"].as_str().unwrap_or("?")
        ));
    }
    if o.no_extent {
        receipt["extent_cases"] = json!([]);
        receipt["extent_skipped"] = json!("--no-extent");
        pass = false;
        why.push("extent cases skipped (--no-extent): not a gate pass".into());
    } else if !small_pass {
        receipt["extent_cases"] = json!([]);
        receipt["extent_skipped"] = json!("the small shape already failed");
    } else {
        let mut cases = Vec::new();
        let shapes = node.extent_shapes();
        let name = node.name();
        // Full-grid cases are independent and each costs a good part of a minute to minutes of emulation:
        // a full-grid node runs them side by side (each also runs its two arms side by side).
        let results: Vec<(bool, Value)> = if node.extent_full_default() {
            lane_common::par_map(shapes.len(), |i| extent_case(o, name, prep, &shapes[i]))
        } else {
            shapes
                .iter()
                .map(|s| extent_case(o, name, prep, s))
                .collect()
        };
        for (shape, (ok, r)) in shapes.iter().zip(results) {
            if !ok {
                why.push(format!(
                    "extent {shape}: {}",
                    r["reason"].as_str().unwrap_or("?")
                ));
            }
            pass &= ok;
            cases.push(r);
        }
        receipt["extent_cases"] = json!(cases);
    }
    // Static hazard checks on the arms' objects (nodes that opt in): any Required/Unchecked finding,
    // any PM-IR hazard obligation or an analysis error on the CANDIDATE fails the gate.
    if node.static_hazards() {
        // Every entry the gate launches (the small shape's, each extent shape's): a lane_proj candidate
        // is launched through `_n{lanes}` at every lane count.
        let mut symbols = vec![prep.inc.symbol.clone()];
        for s in node.extent_shapes() {
            if let Some(sym) = build_node(node.name(), Some(&s))
                .ok()
                .and_then(|n| n.incumbent().ok())
                .map(|i| i.symbol)
            {
                if !symbols.contains(&sym) {
                    symbols.push(sym);
                }
            }
        }
        let mut by_symbol = serde_json::Map::new();
        let mut bad_symbols = Vec::new();
        for sym in &symbols {
            let (_, hb) = emu::hazards(&prep.base.image, sym);
            let (hc_ok, hc) = emu::hazards(&prep.cand.image, sym);
            if !hc_ok {
                bad_symbols.push(format!("{sym}: {}", hc["reason"].as_str().unwrap_or("?")));
            }
            by_symbol.insert(
                sym.clone(),
                json!({"pass": hc_ok, "candidate": hc, "baseline": hb}),
            );
        }
        receipt["hazards"] = json!({
            "pass": bad_symbols.is_empty(),
            "launched_symbol": by_symbol.get(&prep.inc.symbol),
            "symbols_checked": symbols,
            "by_symbol": by_symbol,
        });
        if !bad_symbols.is_empty() {
            pass = false;
            why.push(format!("hazards: {}", bad_symbols.join("; ")));
        }
    }
    receipt["pass"] = json!(pass);
    receipt["reason"] = if pass {
        Value::Null
    } else {
        json!(why.join(" | "))
    };
    rec["emu_gate"] = receipt;
    if pass {
        (rec, Outcome::Ok)
    } else {
        eprintln!(
            "hip_loop: emu_gate FAIL: {}",
            rec["emu_gate"]["reason"].as_str().unwrap_or("?")
        );
        (rec, Outcome::IdentityFailed)
    }
}

/// One production-extent case of `emu_stage` at `shape`.
fn extent_case(o: &Opts, name: &str, prep: &Prep, shape: &str) -> (bool, Value) {
    let t = std::time::Instant::now();
    let bad = |why: String| {
        (
            false,
            json!({"label": shape, "shape": shape, "pass": false, "reason": why}),
        )
    };
    let n = match build_node(name, Some(shape)) {
        Ok(n) => n,
        Err(e) => return bad(format!("extent shape: {e}")),
    };
    // The launched entry depends on the shape (lane_proj: `_n{lanes}`).
    let symbol = match n.incumbent() {
        Ok(i) => i.symbol,
        Err(e) => return bad(format!("incumbent: {e}")),
    };
    let base_geo = n.geometry();
    let cand_geo = match n.geometry_with(&mut prep.cand_header.clone()) {
        Ok(g) => g,
        Err(e) => return bad(format!("candidate geometry at this shape: {e}")),
    };
    let full = match o.emu_extent.as_deref() {
        Some("full") => true,
        Some(_) => false,
        None => n.extent_full_default(),
    };
    if full {
        // EVERY workgroup of both arms' production grids, the whole output compared bitwise: no
        // dependence on a workgroup-to-output ownership assumption about the candidate's tiling.
        let mut case = match n.emu_case(o.seed) {
            Ok(c) => c,
            Err(e) => return bad(format!("inputs: {e}")),
        };
        let built = t.elapsed().as_millis() as u64;
        let (ok, mut r) = emu::gate(
            &symbol,
            &mut case,
            (&prep.base.image, &base_geo),
            (&prep.cand.image, &cand_geo),
            n.shape_json(),
            o.seed,
        );
        r["label"] = json!(shape);
        r["mode"] = json!("full_grid");
        r["grid"] = json!({"baseline": base_geo.grid, "candidate": cand_geo.grid});
        r["inputs_ms"] = json!(built);
        r["total_ms"] = json!(t.elapsed().as_millis() as u64);
        return (ok, r);
    }
    let (ob, oc) = match (
        n.out_owner(&Header::default()),
        n.out_owner(&prep.cand_header),
    ) {
        (Ok(b), Ok(c)) => (b, c),
        (Err(e), _) | (_, Err(e)) => return bad(format!("output owner map: {e}")),
    };
    let mut case = match n.emu_case(o.seed) {
        Ok(c) => c,
        Err(e) => return bad(format!("inputs: {e}")),
    };
    let built = t.elapsed().as_millis() as u64;
    let (ok, mut r) = emu::extent_gate(
        shape,
        &symbol,
        &mut case,
        (&prep.base.image, &base_geo, &*ob),
        (&prep.cand.image, &cand_geo, &*oc),
        n.shape_json(),
        o.seed,
    );
    if n.extent_full_default() || o.emu_extent.is_some() {
        r["mode"] = json!("sampled");
    }
    r["inputs_ms"] = json!(built);
    r["total_ms"] = json!(t.elapsed().as_millis() as u64);
    (ok, r)
}

// ---------------------------------------------------------------------------
// GPU

struct Arm {
    _module: hip_bridge::Module,
    func: hip_bridge::Function,
    geo: Geometry,
    karg: Vec<u8>,
}

fn load(
    gpu: &rdna_compute::Gpu,
    b: &jit::Built,
    symbol: &str,
    geo: Geometry,
    karg: Vec<u8>,
) -> Result<Arm> {
    // A fresh module per arm: no name-keyed function cache sits between the two
    // arms, which share the kernel symbol.
    let module = gpu.hip.module_load_data(&b.image).map_err(err)?;
    let func = gpu.hip.module_get_function(&module, symbol).map_err(err)?;
    Ok(Arm {
        _module: module,
        func,
        geo,
        karg,
    })
}

fn sync(gpu: &rdna_compute::Gpu) -> Result<()> {
    gpu.hip.device_synchronize().map_err(err)
}

fn run_arm(gpu: &rdna_compute::Gpu, node: &mut dyn Node, arm: &mut Arm) -> Result<Vec<u8>> {
    node.reset(gpu)?;
    sync(gpu)?;
    node.dispatch(gpu, &arm.func, &arm.geo, &mut arm.karg)?;
    sync(gpu)?;
    node.read_output(gpu)
}

/// First differing byte and the number of differing bytes.
fn diff(a: &[u8], b: &[u8]) -> Option<(usize, usize)> {
    if a.len() != b.len() {
        return Some((a.len().min(b.len()), a.len().abs_diff(b.len())));
    }
    let first = a.iter().zip(b).position(|(x, y)| x != y)?;
    Some((first, a.iter().zip(b).filter(|(x, y)| x != y).count()))
}

fn clock_note(gpu: &rdna_compute::Gpu) -> Value {
    let bus = gpu
        .hip
        .device_pci_bus_id(gpu.device_id)
        .ok()
        .map(|s| s.to_lowercase());
    let read = |name: &str| -> Option<String> {
        let bus = bus.as_ref()?;
        std::fs::read_to_string(format!("/sys/bus/pci/devices/{bus}/{name}"))
            .ok()
            .map(|s| s.trim().replace('\n', " | "))
    };
    json!({
        "bus_id": bus,
        "sclk_dpm": read("pp_dpm_sclk"),
        "perf_level": read("power_dpm_force_performance_level"),
    })
}

/// P[X >= k], X ~ Binomial(n, 1/2): the one-sided exact sign test.
fn p_upper(k: usize, n: usize) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    // sum_{i=0}^{n-k} C(n, i) / 2^n, by symmetry.
    let mut term = 0.5f64.powi(n as i32);
    let mut sum = term;
    for i in 0..(n - k) {
        term *= (n - i) as f64 / (i + 1) as f64;
        sum += term;
    }
    sum.min(1.0)
}

fn stats_json(s: &abba::TimingStats) -> Value {
    json!({"median": s.median / 1e3, "min": s.min / 1e3, "max": s.max / 1e3,
        "p90": s.p90 / 1e3, "mean": s.mean / 1e3})
}

/// The timing record from the arms' samples (ns, schedule order, whole rounds).
fn timing_record(base_ns: &[f64], cand_ns: &[f64], n: usize, work: node::Work) -> Result<Value> {
    let bs = abba::timing_stats(base_ns)?;
    let cs = abba::timing_stats(cand_ns)?;
    let bm = abba::block_medians(base_ns, n)?;
    let cm = abba::block_medians(cand_ns, n)?;
    let ratios = abba::round_ratios(base_ns, cand_ns, n)?;
    let rounds: Vec<Value> = ratios
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let b = (bm[2 * i] + bm[2 * i + 1]) / 2.0;
            let c = (cm[2 * i] + cm[2 * i + 1]) / 2.0;
            json!({"round": i, "base_us": b / 1e3, "cand_us": c / 1e3, "ratio": r})
        })
        .collect();
    let mut sorted = ratios.clone();
    sorted.sort_by(f64::total_cmp);
    let r = sorted.len();
    let median = if r % 2 == 0 {
        (sorted[r / 2 - 1] + sorted[r / 2]) / 2.0
    } else {
        sorted[r / 2]
    };
    let faster = ratios.iter().filter(|x| **x < 1.0).count();
    let slower = ratios.iter().filter(|x| **x > 1.0).count();
    let tflops = |ns: f64| work.flops / ns / 1e3;
    let gbps = |ns: f64| work.bytes / ns;
    Ok(json!({
        "rounds": rounds,
        "ratio_median": median,
        "ratio_min": sorted[0],
        "ratio_max": sorted[r - 1],
        "rounds_faster": faster,
        "rounds_slower": slower,
        "rounds_tied": r - faster - slower,
        "p_faster": p_upper(faster, r),
        "p_slower": p_upper(slower, r),
        "timing": {
            "n": n, "rounds": r, "calls_per_arm": bs.count,
            "baseline_us": stats_json(&bs), "candidate_us": stats_json(&cs),
            "baseline_tflops": tflops(bs.median), "candidate_tflops": tflops(cs.median),
            "baseline_gbps": gbps(bs.median), "candidate_gbps": gbps(cs.median),
            "unit": "us per launch: hipEvent pair on the default stream around one dispatch; \
                     device idle at the start (output reset + sync outside the interval), \
                     so launch latency is inside the interval for both arms",
        },
    }))
}

fn sample(
    gpu: &rdna_compute::Gpu,
    node: &mut dyn Node,
    base: &mut Arm,
    cand: &mut Arm,
    n: usize,
    rounds: usize,
    flush: Option<&cold::Flush>,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (per_arm, _) = abba::timing_counts(n, rounds)?;
    let (mut b, mut c) = (Vec::with_capacity(per_arm), Vec::with_capacity(per_arm));
    let start = gpu.hip.event_create().map_err(err)?;
    let stop = match gpu.hip.event_create() {
        Ok(e) => e,
        Err(e) => {
            let _ = gpu.hip.event_destroy(start);
            return Err(err(e));
        }
    };
    // `--flush`: evict L2 / the last-level cache / TLBs / instruction caches between dispatches
    // the way the other kernels of a decode step do; always outside the timed interval.
    let evict = |gpu: &rdna_compute::Gpu| -> Result<()> { cold::evict(&gpu.hip, flush) };
    let measured = (|| -> Result<()> {
        evict(gpu)?;
        for (i, which) in abba::timing_schedule(n, rounds)?.into_iter().enumerate() {
            let (arm, sink) = match which {
                abba::TimingArm::Incumbent => (&mut *base, &mut b),
                abba::TimingArm::Candidate => (&mut *cand, &mut c),
            };
            // A node with a pool of weight sets points dispatch `i` at set `i % sets`.
            node.select_set(&mut arm.karg, i);
            node.reset(gpu)?;
            sync(gpu)?;
            gpu.hip.event_record(&start, None).map_err(err)?;
            node.dispatch(gpu, &arm.func, &arm.geo, &mut arm.karg)?;
            gpu.hip.event_record(&stop, None).map_err(err)?;
            gpu.hip.event_synchronize(&stop).map_err(err)?;
            sink.push(f64::from(gpu.hip.event_elapsed_ms(&start, &stop).map_err(err)?) * 1e6);
            evict(gpu)?;
        }
        Ok(())
    })();
    let d1 = gpu.hip.event_destroy(start).map_err(err);
    let d2 = gpu.hip.event_destroy(stop).map_err(err);
    measured?;
    d1?;
    d2?;
    Ok((b, c))
}

fn gpu_stage(o: &Opts, node: &mut dyn Node, prep: &Prep, mut rec: Value) -> (Value, Outcome) {
    let gpu = match rdna_compute::Gpu::init() {
        Ok(g) => g,
        Err(e) => return fail(rec, "device", e),
    };
    let arch = gpu.arch.split(':').next().unwrap_or("").to_owned();
    rec["device"] = json!({
        "arch": gpu.arch, "device_id": gpu.device_id,
        "uuid": std::env::var("ROCR_VISIBLE_DEVICES").ok().or_else(|| std::env::var("HIPFIRE_DEVICES").ok()),
        "fabric_job": std::env::var("FABRIC_JOB").ok(),
    });
    if arch != node.arch() {
        return fail(
            rec,
            "device",
            format!("device is {arch}, hip_loop requires {}", node.arch()),
        );
    }
    if o.identity_only {
        // No timing and no pool: the node's adversarial matrix over every shape it names.
        return identity_only(o, node, prep, &gpu, rec);
    }
    if let Err(e) = node.setup(&gpu, o.seed) {
        return fail(rec, "setup", e);
    }
    let res = gpu_run(o, node, prep, &gpu, &mut rec);
    node.teardown(&gpu);
    match res {
        Ok(outcome) => (rec, outcome),
        Err((stage, e)) => fail(rec, &stage, e),
    }
}

/// `--identity-only`: the adversarial identity matrix of the node over ALL its identity shapes (lane_proj: every
/// group; the node's own lane set per group), both arms, no timing. Exit 0 = every matrix passed.
fn identity_only(
    o: &Opts,
    node: &mut dyn Node,
    prep: &Prep,
    gpu: &rdna_compute::Gpu,
    mut rec: Value,
) -> (Value, Outcome) {
    let empty = Header::default();
    let mut shapes = node.identity_shapes();
    if shapes.is_empty() {
        shapes.push(String::new());
    }
    let mut results = Vec::new();
    let (mut pass, mut cases, mut mism, mut nanp) = (true, 0u64, 0u64, 0u64);
    let (mut cpu_b, mut cpu_c) = (true, true);
    let mut failures: Vec<Value> = Vec::new();
    for shape in shapes {
        let mut other = if shape.is_empty() {
            None
        } else {
            match build_node(node.name(), Some(&shape)) {
                Ok(n) => Some(n),
                Err(e) => {
                    return fail(
                        rec,
                        "identity_matrix",
                        format!("identity shape {shape}: {e}"),
                    )
                }
            }
        };
        let target: &mut dyn Node = match other.as_mut() {
            Some(n) => n.as_mut(),
            None => &mut *node,
        };
        let label = target.shape_label();
        let m = target.adversarial(
            gpu,
            &node::ArmImage {
                tag: "baseline",
                image: &prep.base.image,
                header: &empty,
            },
            &node::ArmImage {
                tag: "candidate",
                image: &prep.cand.image,
                header: &prep.cand_header,
            },
            o.seed,
        );
        match m {
            Ok(Some(m)) => {
                pass &= m["pass"] == json!(true);
                cases += m["cases"].as_u64().unwrap_or(0);
                mism += m["mismatches"].as_u64().unwrap_or(0);
                nanp += m["nan_payload_diffs"].as_u64().unwrap_or(0);
                cpu_b &= m["cpu_ref_identical"]["baseline"] == json!(true);
                cpu_c &= m["cpu_ref_identical"]["candidate"] == json!(true);
                if let Some(f) = m["failures"].as_array() {
                    failures.extend(f.iter().take(8).cloned());
                }
                results.push(json!({"shape": label, "matrix": m}));
            }
            Ok(None) => {}
            Err(e) => return fail(rec, "identity_matrix", e),
        }
    }
    if results.is_empty() {
        return fail(
            rec,
            "identity",
            format!("node {} has no adversarial identity matrix", node.name()),
        );
    }
    rec["identity_mode"] = json!("only");
    rec["identity_matrix"] = json!({
        "pass": pass, "cases": cases, "mismatches": mism, "nan_payload_diffs": nanp,
        "cpu_ref_identical": {"baseline": cpu_b, "candidate": cpu_c},
        "failures": failures, "shapes": results,
    });
    rec["identical_to_baseline"] = json!(pass);
    if pass {
        (rec, Outcome::Ok)
    } else {
        (rec, Outcome::IdentityFailed)
    }
}

fn gpu_run(
    o: &Opts,
    node: &mut dyn Node,
    prep: &Prep,
    gpu: &rdna_compute::Gpu,
    rec: &mut Value,
) -> std::result::Result<Outcome, (String, String)> {
    let stage = |s: &str| {
        let s = s.to_owned();
        move |e: String| (s.clone(), e)
    };
    let karg = node.kernarg().map_err(stage("setup"))?;
    let sym = &prep.inc.symbol;
    let mut base =
        load(gpu, &prep.base, sym, prep.base_geo, karg.clone()).map_err(stage("load"))?;
    let mut cand = load(gpu, &prep.cand, sym, prep.cand_geo, karg).map_err(stage("load"))?;

    // `--identity full|smoke|skip` (default full). `smoke`: only the basic dispatch identity below (the
    // timed shape's pristine-reset launches of both arms, byte-compared); `skip`: no identity at all (A/A and
    // control runs, profiler traces of the timing launches only); `full`: smoke plus the node's own
    // adversarial matrix.
    let mode = o.identity.as_deref().unwrap_or("full");
    rec["identity_mode"] = json!(mode);
    if mode == "skip" {
        rec["identical_to_baseline"] = Value::Null;
    } else {
        // Identity: every dispatch starts from the pristine output.
        node.reset(gpu).map_err(stage("identity"))?;
        sync(gpu).map_err(stage("identity"))?;
        let pristine = node.read_output(gpu).map_err(stage("identity"))?;
        let b1 = run_arm(gpu, node, &mut base).map_err(stage("identity"))?;
        let b2 = run_arm(gpu, node, &mut base).map_err(stage("identity"))?;
        let c1 = run_arm(gpu, node, &mut cand).map_err(stage("identity"))?;
        let c2 = run_arm(gpu, node, &mut cand).map_err(stage("identity"))?;
        let changed = diff(&pristine, &b1).is_some();
        let base_repeat = b1 == b2;
        let cand_repeat = c1 == c2;
        let d = diff(&b1, &c1);
        let identical = d.is_none();
        rec["identical_to_baseline"] = json!(identical);
        rec["identity"] = json!({
            "output_bytes": b1.len(),
            "dispatch_changed_output": changed,
            "baseline_repeatable": base_repeat,
            "candidate_repeatable": cand_repeat,
            "first_diff_byte": d.map(|x| x.0),
            "differing_bytes": d.map(|x| x.1),
        });
        if !changed {
            return Err((
                "identity".into(),
                "the baseline dispatch did not change the output: the launch is not exercised"
                    .into(),
            ));
        }
        if !base_repeat {
            return Err((
                "identity".into(),
                "the baseline output is not repeatable from a pristine state".into(),
            ));
        }
        if !identical || !cand_repeat {
            return Ok(Outcome::IdentityFailed);
        }

        // The node's own adversarial identity matrix (both arms through every input kind, the CPU
        // reference); a failure is an identity failure like a diff in the basic run.
        if mode == "full" {
            let empty = Header::default();
            let matrix = node
                .adversarial(
                    gpu,
                    &node::ArmImage {
                        tag: "baseline",
                        image: &prep.base.image,
                        header: &empty,
                    },
                    &node::ArmImage {
                        tag: "candidate",
                        image: &prep.cand.image,
                        header: &prep.cand_header,
                    },
                    o.seed,
                )
                .map_err(stage("identity_matrix"))?;
            if let Some(m) = matrix {
                let pass = m["pass"] == json!(true);
                rec["identity_matrix"] = m;
                if !pass {
                    rec["identical_to_baseline"] = json!(false);
                    return Ok(Outcome::IdentityFailed);
                }
            }
        }
    }

    let flush = if node.flush_requested() {
        Some(cold::Flush::alloc(&gpu.hip).map_err(stage("setup"))?)
    } else {
        None
    };
    let before = clock_note(gpu);
    let sampled = sample(
        gpu,
        node,
        &mut base,
        &mut cand,
        o.time,
        o.rounds,
        flush.as_ref(),
    );
    if let Some(f) = flush {
        f.free(&gpu.hip);
    }
    let (b_ns, c_ns) = sampled.map_err(stage("timing"))?;
    let after = clock_note(gpu);
    let t = timing_record(&b_ns, &c_ns, o.time, node.work()).map_err(stage("timing"))?;
    if let Value::Object(m) = t {
        for (k, v) in m {
            rec[k] = v;
        }
    }
    rec["timing"]["flush"] = json!(node.flush_requested());
    if node.flush_requested() {
        rec["timing"]["flush_method"] = json!(cold::flush_method());
    }
    rec["clock_note"] = json!({
        "clocks_pinned": false,
        "method": "ABBA-paired rounds cancel slow drift; sclk dpm table sampled before/after",
        "before": before, "after": after,
    });
    Ok(Outcome::Ok)
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: hip_loop {USAGE}");
        return;
    }
    let o = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("hip_loop: {e}\nusage: hip_loop {USAGE}");
            std::process::exit(2);
        }
    };
    let (rec, outcome) = run(&o);
    println!(
        "{}",
        serde_json::to_string_pretty(&rec).expect("record is JSON")
    );
    if let Some(e) = rec["error"].as_object() {
        eprintln!("hip_loop: {}: {}", e["stage"], e["message"]);
    }
    if outcome == Outcome::IdentityFailed {
        eprintln!("hip_loop: candidate output is not bit-identical to the baseline");
    }
    std::process::exit(outcome.code());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn cli_parses_and_refuses() {
        let o = parse_args(&args(
            "--node mq6_res --shape qkv --cand a.hip --offline --time 5 --rounds 3 --seed 9",
        ))
        .unwrap();
        assert!(o.offline && !o.null_control);
        assert_eq!((o.time, o.rounds, o.seed), (5, 3, 9));
        assert_eq!(o.shape.as_deref(), Some("qkv"));
        let o = parse_args(&args("--node hc_mix --null-control aa")).unwrap();
        assert!(o.null_control && o.cand.is_none());
        for bad in [
            "--node nope --cand a",
            "--node mq6_res",
            "--node mq6_res --cand a --null-control aa",
            "--node mq6_res --cand a --null-control bb",
            "--node mq6_res --cand a --time 0",
            "--node mq6_res --cand a --wat",
            "--node mq6_res --cand",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn one_sided_sign_test() {
        assert_eq!(p_upper(0, 10), 1.0);
        assert_eq!(p_upper(11, 10), 0.0);
        assert!((p_upper(10, 10) - 1.0 / 1024.0).abs() < 1e-15);
        assert!((p_upper(8, 10) - 56.0 / 1024.0).abs() < 1e-15);
        assert!((p_upper(5, 10) - 638.0 / 1024.0).abs() < 1e-15);
    }

    #[test]
    fn timing_record_pairs_rounds_and_signs() {
        // n = 2 launches per block, 3 rounds; the candidate is 10% faster each round.
        let n = 2;
        let base: Vec<f64> = vec![100.0; 2 * n * 3 * 1];
        let cand: Vec<f64> = vec![90.0; 2 * n * 3];
        let w = node::Work {
            flops: 1e6,
            bytes: 1e6,
        };
        let t = timing_record(&base, &cand, n, w).unwrap();
        assert_eq!(t["rounds"].as_array().unwrap().len(), 3);
        assert!((t["ratio_median"].as_f64().unwrap() - 0.9).abs() < 1e-12);
        assert_eq!(t["rounds_faster"], 3);
        assert!((t["p_faster"].as_f64().unwrap() - 0.125).abs() < 1e-15);
        assert_eq!(t["p_slower"].as_f64().unwrap(), 1.0);
        assert!((t["timing"]["baseline_tflops"].as_f64().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn diff_reports_first_byte_and_count() {
        assert_eq!(diff(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(diff(&[1, 2, 3, 4], &[1, 9, 3, 8]), Some((1, 2)));
        assert!(diff(&[1], &[1, 2]).is_some());
    }

    #[test]
    fn registered_nodes_build_and_reproduce_production_grids() {
        // (node, shape, PF-1 profiler grid)
        for (n, s, grid) in [
            ("mq6_res", "qkv", [163840u64, 6, 1]),
            ("mq6_res", "z", [98304, 6, 1]),
            ("mq6_res", "out", [40960, 6, 1]),
            ("mq6_res", "qg", [196608, 6, 1]),
            ("mq6_res", "ba", [768, 6, 1]),
            ("hc_mix", "mix", [320, 12, 1]),
        ] {
            let node = build_node(n, Some(s)).unwrap();
            assert_eq!(node.geometry().threads(), grid, "{n} {s}");
            assert_eq!(node.name(), n);
        }
    }

    /// Run `--offline` for `node` with the incumbent's own source (or `source`).
    fn offline(node: &str, shape: &str, source: Option<&str>, tag: &str) -> Value {
        let dir = tmp(tag);
        let cand = dir.join("cand.hip");
        let n = build_node(node, Some(shape)).unwrap();
        std::fs::write(
            &cand,
            source
                .map(str::to_owned)
                .unwrap_or(n.incumbent().unwrap().source),
        )
        .unwrap();
        let o = Opts {
            node: node.into(),
            shape: Some(shape.into()),
            cand: Some(cand),
            offline: true,
            emu: false,
            no_extent: false,
            flush: false,
            emu_extent: None,
            identity: None,
            identity_only: false,
            null_control: false,
            cand_object: None,
            cand_geometry: None,
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir.join("out")),
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        rec
    }

    #[test]
    fn offline_record_for_every_mq6_shape_matches_pf1_resources() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let mut shas = std::collections::BTreeSet::new();
        for (i, shape) in ["qkv", "z", "out", "qg", "ba"].iter().enumerate() {
            let rec = offline("mq6_res", shape, None, &format!("mq6-{i}"));
            assert_eq!(rec["node"], "mq6_res");
            assert_eq!(rec["shape"], *shape);
            assert_eq!(rec["symbol"], "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds");
            assert!(rec["error"].is_null());
            assert_eq!(rec["identical_code"], true);
            assert_eq!(
                rec["baseline"]["code_sha256"],
                rec["candidate"]["code_sha256"]
            );
            for arm in ["baseline", "candidate"] {
                let a = &rec[arm];
                assert_eq!(a["code_sha256"].as_str().unwrap().len(), 64);
                // PF-1: the MW8 kernel is 256 threads, 8 KiB LDS, 96 VGPR.
                assert_eq!(a["lds_bytes"], 8192, "{a}");
                // The metadata count is 92; the allocation granule (8) gives the 96 PF-1 reports.
                assert_eq!(a["vgprs_allocated"], 96, "{a}");
                assert!(a["vgprs"].as_u64().unwrap() <= 96, "{a}");
                assert_eq!(a["max_flat_workgroup_size"], 256);
                assert_eq!(a["scratch_bytes"], 0);
                assert_eq!(a["spill"], 0);
                assert!(a["code_bytes"].as_u64().unwrap() > 0);
                assert!(a["occupancy_waves_per_simd"].as_u64().unwrap() >= 1);
                assert_eq!(a["geometry"]["block"], json!([256, 1, 1]));
            }
            shas.insert(rec["baseline"]["code_sha256"].as_str().unwrap().to_owned());
        }
        // Shape does not change the code: one kernel, one sha.
        assert_eq!(shas.len(), 1);
    }

    #[test]
    fn code_sha_is_stable_across_compiles_and_changes_with_the_source() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let n = build_node("mq6_res", Some("qkv")).unwrap();
        let src = n.incumbent().unwrap().source;
        assert!(src.contains("#pragma unroll"));
        let a = offline("mq6_res", "qkv", None, "stab-a");
        let b = offline("mq6_res", "qkv", None, "stab-b");
        assert_eq!(a["baseline"]["code_sha256"], b["baseline"]["code_sha256"]);
        assert_eq!(
            a["baseline"]["object_sha256"],
            b["baseline"]["object_sha256"]
        );
        // Perturb the source: the unrolled loops become rolled.
        let perturbed = src.replace("#pragma unroll\n", "#pragma unroll 1\n");
        assert_ne!(perturbed, src);
        let p = offline("mq6_res", "qkv", Some(&perturbed), "pert");
        assert_eq!(p["identical_code"], false);
        assert_ne!(p["baseline"]["code_sha256"], p["candidate"]["code_sha256"]);
        assert_eq!(p["baseline"]["code_sha256"], a["baseline"]["code_sha256"]);
    }

    #[test]
    fn candidate_header_changes_geometry_not_code() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let n = build_node("mq6_res", Some("qkv")).unwrap();
        let src = format!(
            "// hip_loop: tile_m=32 tile_n=128 block=256\n{}",
            n.incumbent().unwrap().source
        );
        let r = offline("mq6_res", "qkv", Some(&src), "hdr");
        assert_eq!(
            r["candidate"]["geometry"]["grid_workgroups"],
            json!([320, 6, 1])
        );
        assert_eq!(
            r["baseline"]["geometry"]["grid_workgroups"],
            json!([640, 6, 1])
        );
        assert_eq!(r["same_geometry"], false);
        assert_eq!(r["candidate_header"]["tile_m"], "32");
    }

    #[test]
    fn missing_symbol_and_bad_header_are_loud() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("loud");
        let cand = dir.join("c.hip");
        std::fs::write(
            &cand,
            "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void other(){}\n",
        )
        .unwrap();
        let mut o = Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand: Some(cand.clone()),
            offline: true,
            emu: false,
            no_extent: false,
            flush: false,
            emu_extent: None,
            identity: None,
            identity_only: false,
            null_control: false,
            cand_object: None,
            cand_geometry: None,
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir.join("out")),
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "compile");
        assert!(
            rec["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not define kernel"),
            "{rec}"
        );
        std::fs::write(&cand, "// hip_loop: tile_m=0\n").unwrap();
        o.out = Some(dir.join("out2"));
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "geometry");
    }

    #[test]
    fn null_control_offline_compiles_the_incumbent_twice() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("aa");
        let o = Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand: None,
            offline: true,
            emu: false,
            no_extent: false,
            flush: false,
            emu_extent: None,
            identity: None,
            identity_only: false,
            null_control: true,
            cand_object: None,
            cand_geometry: None,
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir),
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        assert_eq!(rec["null_control"]["kind"], "aa");
        assert_eq!(rec["null_control"]["identical_objects"], true);
        assert_ne!(
            rec["baseline"]["source_path"],
            rec["candidate"]["source_path"]
        );
    }

    #[test]
    fn offline_record_for_conv_rows_on_both_arches() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        for (shape, arch, tag) in [
            ("pp512", "gfx1100", "conv-a"),
            ("pp512,arch=gfx1201", "gfx1201", "conv-b"),
        ] {
            let rec = &offline("conv_rows", shape, None, tag);
            assert_eq!(rec["symbol"], "causal_conv_depthwise_silu_rows_f32");
            assert_eq!(rec["identical_code"], true, "{arch}");
            assert_eq!(
                rec["baseline"]["geometry"]["grid_work_items"],
                json!([5120, 1, 1])
            );
            assert_eq!(rec["baseline"]["max_flat_workgroup_size"], 1024);
            assert_eq!(rec["baseline"]["spill"], 0);
            assert_eq!(
                rec["baseline"]["code_sha256"],
                rec["candidate"]["code_sha256"]
            );
            assert_eq!(rec["baseline"]["code_sha256"].as_str().unwrap().len(), 64);
        }
    }

    #[test]
    fn offline_record_for_hc_mix() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let rec = &offline("hc_mix", "mix", None, "hc");
        assert_eq!(rec["symbol"], "gemm_wmma_lds_64_64_32_64_k64_p");
        assert_eq!(rec["identical_code"], true);
        assert_eq!(
            rec["baseline"]["geometry"]["grid_work_items"],
            json!([320, 12, 1])
        );
        assert_eq!(rec["baseline"]["max_flat_workgroup_size"], 64);
        assert_eq!(rec["baseline"]["spill"], 0);
        assert_eq!(rec["baseline"]["code_sha256"].as_str().unwrap().len(), 64);
    }
}

#[cfg(test)]
mod object_tests {
    use super::*;

    fn opts(dir: &Path, object: PathBuf, geometry: Option<&str>) -> Opts {
        Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand: None,
            offline: true,
            emu: false,
            no_extent: false,
            flush: false,
            emu_extent: None,
            identity: None,
            identity_only: false,
            null_control: false,
            cand_object: Some(object),
            cand_geometry: geometry.map(str::to_owned),
            base_object: None,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(dir.join("out")),
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-objtest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn cli_takes_exactly_one_candidate_source() {
        let a = |s: &str| parse_args(&s.split_whitespace().map(str::to_owned).collect::<Vec<_>>());
        let o =
            a("--node mq6_res --cand-object x.co --cand-geometry tile_m=32,tile_n=256").unwrap();
        assert_eq!(o.cand_geometry.as_deref(), Some("tile_m=32,tile_n=256"));
        for bad in [
            "--node mq6_res --cand-object x.co --cand y.hip",
            "--node mq6_res --cand-object x.co --null-control aa",
            "--node mq6_res --cand-geometry tile_m=32 --cand y.hip",
            "--node mq6_res",
        ] {
            assert!(a(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prebuilt_object_is_judged_like_a_compiled_candidate() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("same");
        jit::isolate_caches(&dir).unwrap();
        let n = build_node("mq6_res", Some("ba")).unwrap();
        let inc = n.incumbent().unwrap();
        let built = jit::compile(&dir, "prebuilt", &inc, &inc.source).unwrap();
        let (rec, outcome) = run(&opts(
            &dir,
            built.image_path.clone(),
            Some("tile_m=32 tile_n=128 block=256"),
        ));
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        assert_eq!(rec["candidate"]["kind"], "object");
        assert_eq!(rec["identical_code"], true);
        assert_eq!(
            rec["candidate"]["geometry"]["grid_workgroups"],
            json!([2, 6, 1])
        );
        assert_eq!(rec["candidate"]["lds_bytes"], 8192);
        assert_eq!(rec["candidate_object"]["sha256"], json!(built.image_sha256));
        assert_eq!(
            rec["candidate_object"]["kernarg_size_matches_incumbent"],
            true
        );
        assert!(rec["candidate"]["hipcc_argv"]
            .as_array()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn object_with_another_kernarg_layout_or_symbol_is_refused() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("abi");
        jit::isolate_caches(&dir).unwrap();
        let n = build_node("mq6_res", Some("ba")).unwrap();
        let mut inc = n.incumbent().unwrap();
        let good = inc.clone();
        inc.source = format!(
            "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void {}(int* a) {{ a[0] = 1; }}\n",
            inc.symbol
        );
        let wrong_abi = jit::compile(&dir, "wrong-abi", &inc, &inc.source).unwrap();
        let (rec, outcome) = run(&opts(&dir, wrong_abi.image_path.clone(), None));
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "abi", "{rec}");
        inc.source = "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void other(int* a) { a[0] = 1; }\n".into();
        let other = jit::compile(&dir, "other", &inc, &inc.source).unwrap();
        let mut o = opts(&dir, other.image_path.clone(), None);
        o.out = Some(dir.join("out2"));
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "object");
        assert!(
            rec["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not define kernel"),
            "{rec}"
        );
        // A geometry outside the header grammar is refused before any compile.
        let built = jit::compile(&dir, "ok", &good, &good.source).unwrap();
        let mut o = opts(&dir, built.image_path.clone(), Some("block=100"));
        o.out = Some(dir.join("out3"));
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "geometry");
    }
}

#[cfg(test)]
mod base_object_tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("hip_loop-basetest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn opts(out: PathBuf, cand: Option<PathBuf>, base: Option<PathBuf>, aa: bool) -> Opts {
        Opts {
            node: "mq6_res".into(),
            shape: Some("ba".into()),
            cand,
            offline: true,
            emu: false,
            no_extent: false,
            flush: false,
            emu_extent: None,
            identity: None,
            identity_only: false,
            null_control: aa,
            cand_object: None,
            cand_geometry: None,
            base_object: base,
            time: 1,
            rounds: 1,
            seed: 1,
            out: Some(out),
        }
    }

    fn inc() -> Incumbent {
        build_node("mq6_res", Some("ba"))
            .unwrap()
            .incumbent()
            .unwrap()
    }

    #[test]
    fn cli_accepts_base_object_with_any_candidate() {
        let a = |s: &str| parse_args(&s.split_whitespace().map(str::to_owned).collect::<Vec<_>>());
        let o = a("--node mq6_res --base-object b.co --cand c.hip").unwrap();
        assert_eq!(o.base_object.as_deref(), Some(Path::new("b.co")));
        assert!(a("--node mq6_res --base-object b.co --null-control aa").is_ok());
        assert!(a("--node mq6_res --base-object b.co --cand-object c.co").is_ok());
        assert!(a("--node mq6_res --base-object").is_err());
    }

    #[test]
    fn offline_writes_both_objects_and_sidecars_and_base_object_round_trips() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("rt");
        let cand = dir.join("cand.hip");
        std::fs::write(
            &cand,
            format!("// hip_loop: tile_m=32 tile_n=128\n{}", inc().source),
        )
        .unwrap();
        let (rec, outcome) = run(&opts(dir.join("o1"), Some(cand.clone()), None, false));
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        for (key, name) in [
            ("baseline_co", "baseline.co"),
            ("candidate_co", "candidate.co"),
        ] {
            let p = dir.join("o1").join(name);
            assert_eq!(rec[key]["path"], json!(p.display().to_string()));
            let bytes = std::fs::read(&p).unwrap();
            assert_eq!(rec[key]["sha256"], json!(elf::sha256_hex(&bytes)));
            let side: Value =
                serde_json::from_str(&std::fs::read_to_string(jit::sidecar_path(&p)).unwrap())
                    .unwrap();
            assert_eq!(side["object_sha256"], rec[key]["sha256"]);
            let src = if key == "baseline_co" {
                inc().source
            } else {
                std::fs::read_to_string(&cand).unwrap()
            };
            assert_eq!(
                side["source_sha256"],
                json!(elf::sha256_hex(src.as_bytes()))
            );
            assert_eq!(
                side["kernels"][&inc().symbol]["code_sha256"],
                rec["baseline"]["code_sha256"]
            );
        }
        assert_eq!(rec["candidate_geometry_arg"], "tile_m=32 tile_n=128");

        // Offline: the prebuilt incumbent is verified against a fresh production compile.
        let base = dir.join("o1").join("baseline.co");
        let (rec2, outcome) = run(&opts(dir.join("o2"), Some(cand), Some(base.clone()), false));
        assert_eq!(outcome, Outcome::Ok, "{}", rec2["error"]);
        assert_eq!(rec2["baseline"]["kind"], "object");
        assert_eq!(
            rec2["base_object"]["verified"],
            "sidecar + recompiled production source"
        );
        assert_eq!(
            rec2["baseline"]["code_sha256"],
            rec["baseline"]["code_sha256"]
        );
        assert!(rec2.get("baseline_co").is_none());

        // A/A with --base-object offline recompiles the candidate; both arms identical.
        let (rec3, outcome) = run(&opts(dir.join("o3"), None, Some(base.clone()), true));
        assert_eq!(outcome, Outcome::Ok, "{}", rec3["error"]);
        assert_eq!(rec3["null_control"]["identical_objects"], true);

        // Device-mode verification (no recompile) trusts the sidecar and checks everything else.
        let b = jit::from_object("baseline", &base).unwrap();
        let v = verify_base_object(&base, &inc(), &b, None).unwrap();
        assert_eq!(
            v["verified"].as_str().unwrap().starts_with("sidecar ("),
            true
        );
    }

    #[test]
    fn base_object_is_refused_when_the_sidecar_does_not_certify_it() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("bad");
        let cand = dir.join("cand.hip");
        std::fs::write(&cand, inc().source).unwrap();
        let (rec, outcome) = run(&opts(dir.join("o1"), Some(cand), None, false));
        assert_eq!(outcome, Outcome::Ok, "{}", rec["error"]);
        let good = dir.join("o1").join("baseline.co");
        let side = jit::sidecar_path(&good);
        let sidecar: Value =
            serde_json::from_str(&std::fs::read_to_string(&side).unwrap()).unwrap();
        let load = |p: &Path| jit::from_object("baseline", p).unwrap();
        let sym = inc().symbol;

        // Missing sidecar.
        let lone = dir.join("lone.co");
        std::fs::copy(&good, &lone).unwrap();
        let e = verify_base_object(&lone, &inc(), &load(&lone), None).unwrap_err();
        assert!(e.contains("lone.co.json"), "{e}");

        let tampered = |name: &str, edit: &dyn Fn(&mut Value)| -> PathBuf {
            let p = dir.join(name);
            std::fs::copy(&good, &p).unwrap();
            let mut s = sidecar.clone();
            edit(&mut s);
            std::fs::write(jit::sidecar_path(&p), serde_json::to_string(&s).unwrap()).unwrap();
            p
        };
        // Wrong kernel .text sha, stale source, wrong module, object sha.
        let p = tampered("sha.co", &|s| {
            s["kernels"][&sym]["code_sha256"] = json!("00")
        });
        assert!(verify_base_object(&p, &inc(), &load(&p), None)
            .unwrap_err()
            .contains(".text sha256"));
        let p = tampered("src.co", &|s| s["source_sha256"] = json!("11"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None)
            .unwrap_err()
            .contains("stale"));
        let p = tampered("mod.co", &|s| s["module"] = json!("other_module"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None)
            .unwrap_err()
            .contains("other_module"));
        let p = tampered("obj.co", &|s| s["object_sha256"] = json!("22"));
        assert!(verify_base_object(&p, &inc(), &load(&p), None)
            .unwrap_err()
            .contains("changed"));
        let p = tampered("abi.co", &|s| {
            s["kernels"][&sym]["explicit_args"] = json!([])
        });
        assert!(verify_base_object(&p, &inc(), &load(&p), None)
            .unwrap_err()
            .contains("kernel arguments"));

        // Offline recompile disagreement: the sidecar and object agree with each other
        // but not with the production compile.
        let other = dir.join("other.co");
        let mut other_inc = inc();
        other_inc.source = inc()
            .source
            .replace("#pragma unroll\n", "#pragma unroll 1\n");
        let built =
            jit::compile(&dir.join("o1"), "perturbed", &other_inc, &other_inc.source).unwrap();
        std::fs::copy(&built.image_path, &other).unwrap();
        let mut s = jit::sidecar_json(
            "mq6_res",
            &inc(),
            &built,
            &inc().source,
            &Default::default(),
        );
        s["source_sha256"] = json!(elf::sha256_hex(inc().source.as_bytes()));
        std::fs::write(
            jit::sidecar_path(&other),
            serde_json::to_string(&s).unwrap(),
        )
        .unwrap();
        let fresh = jit::compile(&dir.join("o1"), "fresh", &inc(), &inc().source).unwrap();
        let e = verify_base_object(&other, &inc(), &load(&other), Some(&fresh)).unwrap_err();
        assert!(e.contains("fresh compile"), "{e}");
        // And through the driver: stage base_object, exit 3.
        let (rec, outcome) = run(&opts(dir.join("o9"), None, Some(other), true));
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "base_object");
    }
}

#[cfg(test)]
mod emu_tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-emutest-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn emu_shapes_are_small_ragged_and_use_the_production_kernarg_layout() {
        // mq6: M=144 K=512 N=40, forced MW8: grid 9x1, ragged under every candidate tile.
        let n = build_node("mq6_res", Some("emu")).unwrap();
        assert_eq!(n.shape_label(), "emu");
        assert_eq!(n.geometry().grid, [9, 1, 1]);
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 6);
        assert_eq!(c.bufs[0].bytes.len(), 144 * 2 * 200);
        assert_eq!(c.bufs[1].bytes.len(), 40 * 512 * 2);
        assert_eq!(c.bufs[2].bytes.len(), 40 * 144 * 4);
        assert_eq!(c.out, 2);
        for (g, grid) in [
            ("tile_m=128 tile_n=256 block=256", [2, 1, 1]),
            ("tile_m=64 tile_n=128 block=128", [3, 1, 1]),
        ] {
            let mut h = Header::parse_list(g).unwrap();
            assert_eq!(n.geometry_with(&mut h).unwrap().grid, grid, "{g}");
        }
        // Same seed, same bytes; another seed differs.
        assert_eq!(c.bufs[0].bytes, n.emu_case(1).unwrap().bufs[0].bytes);
        assert_ne!(c.bufs[0].bytes, n.emu_case(2).unwrap().bufs[0].bytes);
        // hc: M=144 K=256 N=24; (A, X, Y, Bias=null, M, K, N, has_bias, lda, ldx).
        let n = build_node("hc_mix", Some("emu")).unwrap();
        assert_eq!(n.geometry().grid, [3, 1, 1]);
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 10);
        assert_eq!(c.args[3], node::EmuArg::Null);
        let mut h = Header::parse_list("tile_m=80 tile_n=64 block=128").unwrap();
        assert_eq!(n.geometry_with(&mut h).unwrap().grid, [2, 1, 1]);
        // qsa: 3 rows at position 102 -> visible 103/104/105, selected_len below the budget, a row on a
        // block boundary (104 = 26 whole blocks), production head/dim constants.
        let n = build_node("qsa_hg4", Some("emu")).unwrap();
        assert_eq!(n.geometry().grid, [6, 1, 3]);
        let c = n.emu_case(1).unwrap();
        assert_eq!(c.args.len(), 14);
        assert_eq!(c.bufs[1].bytes.len(), 105 * 544);
        let sel: Vec<i32> = c.bufs[3]
            .bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let live: Vec<usize> = (0..3)
            .map(|r| {
                sel[r * 2051..(r + 1) * 2051]
                    .iter()
                    .filter(|&&t| t >= 0)
                    .count()
            })
            .collect();
        assert_eq!(live, [103, 104, 105]);
        assert!(c.bufs[4].bytes.iter().all(|&b| b == 0xff));
    }

    #[test]
    fn extent_sampling_covers_corners_edges_and_is_seeded() {
        let g = emu::sample_wgs([640, 6, 1], 1, "qkv");
        for corner in [[0, 0, 0], [639, 0, 0], [0, 5, 0], [639, 5, 0]] {
            assert!(g.contains(&corner), "{corner:?}");
        }
        assert!(g.len() >= 8 && g.len() <= 25, "{}", g.len());
        assert!(g.iter().all(|w| w[0] < 640 && w[1] < 6 && w[2] == 0));
        assert_eq!(g, emu::sample_wgs([640, 6, 1], 1, "qkv"));
        assert_ne!(g, emu::sample_wgs([640, 6, 1], 2, "qkv"));
        // The three-dimensional qsa grid: every corner, last tile along each dim.
        let q = emu::sample_wgs([6, 1, 768], 1, "q");
        for corner in [[0, 0, 0], [5, 0, 0], [0, 0, 767], [5, 0, 767]] {
            assert!(q.contains(&corner), "{corner:?}");
        }
        // A one-workgroup grid is just that workgroup.
        assert_eq!(emu::sample_wgs([1, 1, 1], 9, "x"), vec![[0, 0, 0]]);
    }

    #[test]
    fn output_owner_maps_follow_the_kernel_layouts() {
        let n = build_node("mq6_res", Some("qkv")).unwrap();
        // Y[token][row]: element (token 5, row 33) with 16 x 128 tiles -> workgroup (2, 0); production header.
        let own = n.out_owner(&Header::default()).unwrap();
        assert_eq!(own(5 * 10240 + 33), [2, 0, 0]);
        assert_eq!(own(767 * 10240 + 10239), [639, 5, 0]);
        let h = Header::parse_list("tile_m=128 tile_n=256 block=256").unwrap();
        let own = n.out_owner(&h).unwrap();
        assert_eq!(own(5 * 10240 + 33), [0, 0, 0]);
        assert_eq!(own(767 * 10240 + 10239), [79, 2, 0]);
        let n = build_node("qsa_hg4", Some("pos=mid")).unwrap();
        let own = n.out_owner(&Header::default()).unwrap();
        // out[row][head][d]: row 3, head 13, d 7 -> workgroup (head / 4, 0, row).
        assert_eq!(own((3 * 24 + 13) * 256 + 7), [3, 0, 3]);
        let h = Header::parse_list("heads_per_wg=12 block=1024").unwrap();
        assert_eq!(n.out_owner(&h).unwrap()((3 * 24 + 13) * 256 + 7), [1, 0, 3]);
        let n = build_node("hc_mix", Some("mix")).unwrap();
        let h = Header::parse_list("tile_m=80 tile_n=64 block=128").unwrap();
        assert_eq!(n.out_owner(&h).unwrap()(70 * 320 + 319), [3, 1, 0]);
        // Every node declares production extents; the qsa ones include a max_seq-scale capacity.
        assert!(n.extent_shapes().contains(&"mix".to_owned()));
        let q = build_node("qsa_hg4", Some("emu")).unwrap().extent_shapes();
        assert!(q.iter().any(|s| s.contains("cap=262144")) && q.contains(&"pos=late".to_owned()));
        for s in &q {
            build_node("qsa_hg4", Some(s)).unwrap();
        }
        assert_eq!(
            build_node("qsa_hg4", Some("pos=late,cap=262144"))
                .unwrap()
                .shape_json()["full_capacity"],
            262144
        );
        assert!(build_node("qsa_hg4", Some("pos=late,cap=100")).is_err());
    }

    /// The extent gate: the incumbent passes against itself at a production extent; a short input buffer
    /// is an unmapped access; a candidate whose geometry does not match its object leaves elements unwritten.
    #[test]
    fn extent_gate_passes_the_incumbent_and_fails_closed() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        let dir = tmp("extent");
        jit::isolate_caches(&dir).unwrap();
        let node = build_node("mq6_res", Some("ba")).unwrap();
        let inc = node.incumbent().unwrap();
        let built = jit::compile(&dir, "baseline", &inc, &inc.source).unwrap();
        let geo = node.geometry();
        let own = node.out_owner(&Header::default()).unwrap();
        let mut case = node.emu_case(1).unwrap();
        let (pass, rec) = emu::extent_gate(
            "ba",
            &inc.symbol,
            &mut case,
            (&built.image, &geo, &*own),
            (&built.image, &geo, &*own),
            node.shape_json(),
            1,
        );
        assert!(pass, "{rec}");
        assert_eq!(rec["identical_to_baseline"], true);
        assert_eq!(
            rec["output_sha256"]["baseline"],
            rec["output_sha256"]["candidate"]
        );
        assert!(rec["elements_compared"].as_u64().unwrap() > 1000);
        // The same case object serves the next gate: buffers were moved back, output pristine.
        let (pass, _) = emu::extent_gate(
            "ba",
            &inc.symbol,
            &mut case,
            (&built.image, &geo, &*own),
            (&built.image, &geo, &*own),
            node.shape_json(),
            1,
        );
        assert!(pass);
        // A short input: the kernel's reads past the end land in the unmapped guard.
        let mut short = case.clone();
        let x = short.bufs.iter_mut().find(|b| b.name == "x").unwrap();
        x.bytes.truncate(x.bytes.len() - 4);
        let (pass, rec) = emu::extent_gate(
            "ba",
            &inc.symbol,
            &mut short,
            (&built.image, &geo, &*own),
            (&built.image, &geo, &*own),
            node.shape_json(),
            1,
        );
        assert!(!pass);
        assert!(
            rec["reason"]
                .as_str()
                .unwrap()
                .contains("unmapped memory access"),
            "{rec}"
        );
        // Candidate geometry (32-row tiles) launched on the 16-row object: half the rows are never written.
        let h = Header::parse_list("tile_m=32 tile_n=128 block=256").unwrap();
        let mut hh = h.clone();
        let cgeo = node.geometry_with(&mut hh).unwrap();
        let cown = node.out_owner(&h).unwrap();
        let (pass, rec) = emu::extent_gate(
            "ba",
            &inc.symbol,
            &mut case,
            (&built.image, &geo, &*own),
            (&built.image, &cgeo, &*cown),
            node.shape_json(),
            1,
        );
        assert!(!pass, "{rec}");
        assert!(
            rec["reason"].as_str().unwrap().contains("did not write"),
            "{rec}"
        );
    }

    /// The gate passes an incumbent against itself and fails closed on an out-of-buffer access and on
    /// a launch that leaves output unwritten (different bytes from the incumbent's).
    #[test]
    fn emu_gate_passes_the_incumbent_and_fails_closed() {
        if !jit::hipcc_available() {
            eprintln!("SKIP: no hipcc");
            return;
        }
        for node_name in ["mq6_res", "hc_mix"] {
            let dir = tmp(node_name);
            jit::isolate_caches(&dir).unwrap();
            let node = build_node(node_name, Some("emu")).unwrap();
            let inc = node.incumbent().unwrap();
            let built = jit::compile(&dir, "baseline", &inc, &inc.source).unwrap();
            let geo = node.geometry();
            let case = node.emu_case(1).unwrap();
            let run = |case: &node::EmuCase, cand_geo: &Geometry| {
                emu::gate(
                    &inc.symbol,
                    &mut case.clone(),
                    (&built.image, &geo),
                    (&built.image, cand_geo),
                    node.shape_json(),
                    1,
                )
            };
            let (pass, rec) = run(&case, &geo);
            assert!(pass, "{node_name}: {rec}");
            assert_eq!(rec["identical_to_baseline"], true);
            assert_eq!(
                rec["output_sha256"]["baseline"],
                rec["output_sha256"]["candidate"]
            );
            let wgs = u64::from(geo.grid[0]) * u64::from(geo.grid[1]) * u64::from(geo.grid[2]);
            assert_eq!(rec["workgroups"]["baseline"], wgs);
            assert!(rec["instructions"]["baseline"].as_u64().unwrap() > 1000);
            assert!(rec["guard_bytes"].as_u64().unwrap() >= 64 * 1024);
            // A buffer one dword short: the kernel's last read lands in the unmapped guard.
            let mut short = case.clone();
            let x = short.bufs.iter_mut().find(|b| b.name == "x").unwrap();
            x.bytes.truncate(x.bytes.len() - 4);
            let (pass, rec) = run(&short, &geo);
            assert!(!pass, "{node_name}: a short input buffer must fail");
            assert!(
                rec["reason"]
                    .as_str()
                    .unwrap()
                    .contains("unmapped memory access"),
                "{rec}"
            );
            // A candidate launched on a grid that skips a workgroup: output differs from the incumbent's.
            let mut small = geo;
            small.grid[0] -= 1;
            let (pass, rec) = run(&case, &small);
            assert!(!pass);
            assert!(rec["reason"].as_str().unwrap().contains("differs"), "{rec}");
            assert_eq!(rec["identical_to_baseline"], false);
        }
    }
}

/// R3Mq6Wp-02-lag-w4n128 passed the pre-0.4.3 gate and then dropped the XTX during ABBA timing. Offline root-cause
/// work (gpu/rounds/r3/fault-0952/ROOT-CAUSE.md) found no ordering, race or address mechanism: the gate model (in-flight
/// memory, LDS happens-before, every schedule) also passes it. The one thing the strict gfx11 store-source reading sees
/// is the stub's `buffer_load_b128 v72:75` right after the `ds_store_b128 v72:75` (value-only; stock hipcc emits the
/// same shape, so no gate uses that reading). This test pins that fact: it must not be mistaken for a card-safety proof.
#[cfg(test)]
mod race_gate_tests {
    use super::*;

    const IMAGE: &[u8] = include_bytes!("hip_loop/fixtures/R3Mq6Wp-02-lag-w4n128.co");
    const SYMBOL: &str = "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds";

    fn arm(model: emu::Model) -> std::result::Result<emu::ArmRun, emu::ArmFail> {
        let n = build_node("mq6_res", Some("emu")).unwrap();
        let mut h = Header::parse_list("tile_m=64 tile_n=128 block=128").unwrap();
        let geo = n.geometry_with(&mut h).unwrap();
        let mut case = n.emu_case(1).unwrap();
        emu::run_arm_model(
            "cand",
            IMAGE,
            SYMBOL,
            &geo,
            &mut case,
            &emu::WgSel::All,
            model,
        )
    }

    #[test]
    fn r3mq6wp_02_passes_the_old_and_the_gate_model_and_only_the_strict_store_reading_names_the_stub(
    ) {
        arm(emu::Model::Legacy)
            .unwrap_or_else(|e| panic!("the pre-0.4.3 gate passes it: {}", e.reason));
        let rep = arm(emu::Model::Gate)
            .unwrap_or_else(|e| panic!("the gate model finds no defect in it: {}", e.reason));
        let sched = rep.json["schedules"].as_array().unwrap();
        assert!(sched.len() >= 7, "{sched:?}");
        let e = arm(emu::Model::StrictStoreSources)
            .err()
            .expect("the strict reading names the stub WAR");
        assert!(
            e.reason.contains("[in-flight]") && e.reason.contains("ds_store_b128"),
            "{}",
            e.reason
        );
    }
}

/// The gfx1201 lane nodes (`lane_proj`, `lane_wo`): incumbents are the runtime's objects, offline records,
/// the options only they take, and the emulator gate on the wave32 `wo` kernel.
#[cfg(test)]
mod lane_tests {
    use super::*;

    /// sha256 of the runtime-compiled incumbent OBJECTS (`hipcc --genco --offload-arch=gfx1201 -O3
    /// --no-offload-compress -fuse-cuid=none -mcode-object-version=6 -DIU4_A4_CANDIDATES=2` + the
    /// source's `HIPFIRE_COMPILER_FLAGS`): `KernelCompiler` reproduces them byte for byte, the whole file, not
    /// only the kernel `.text`.
    const LANE_PROJ_OBJECT_SHA: &str =
        "68db168d722305c1be594f2d16a8b76558a447353f04a15c19ddf53b29028d7b";
    const RES_LANES_OBJECT_SHA: &str =
        "16f5f16d35083682c6d12b0d5a813c6056006785d0a2d8188fdf8845873ad31c";
    const LANE_PROJ_SOURCE_SHA: &str =
        "32bec4741cd72e4e279ea09403c3e63f0e5ff8e401287e809be3f879594079d9";
    const RES_LANES_SOURCE_SHA: &str =
        "036fdd2cf77014d8ac70deaeee37d73a08fc3313318c5717fa122b65e6d70b71";

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hip_loop-lane-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn offline(node: &str, shape: &str, source: &str, tag: &str) -> Value {
        let dir = tmp(tag);
        let cand = dir.join("cand.hip");
        std::fs::write(&cand, source).unwrap();
        let o = Opts {
            node: node.into(),
            shape: Some(shape.into()),
            cand: Some(cand),
            offline: true,
            out: Some(dir.join("out")),
            ..Opts::default()
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Ok, "{rec}");
        rec
    }

    #[test]
    fn incumbent_sources_are_the_shipped_bytes_the_runtime_compiles() {
        let p = build_node("lane_proj", None).unwrap().incumbent().unwrap();
        let w = build_node("lane_wo", None).unwrap().incumbent().unwrap();
        assert_eq!(elf::sha256_hex(p.source.as_bytes()), LANE_PROJ_SOURCE_SHA);
        assert_eq!(elf::sha256_hex(w.source.as_bytes()), RES_LANES_SOURCE_SHA);
        assert!(p
            .source
            .starts_with("// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64\n"));
        assert!(w
            .source
            .starts_with("#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1\n"));
        assert_eq!(
            (p.extra_flags.as_str(), w.extra_flags.as_str()),
            ("-DIU4_A4_CANDIDATES=2", "-DIU4_A4_CANDIDATES=2")
        );
        for n in ["lane_proj", "lane_wo"] {
            assert_eq!(build_node(n, None).unwrap().arch(), "gfx1201");
        }
        for n in ["mq6_res", "hc_mix", "qsa_hg4"] {
            assert_eq!(build_node(n, Some("emu")).unwrap().arch(), ARCH);
        }
    }

    #[test]
    fn incumbent_objects_are_the_runtime_objects() {
        if !jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let dir = tmp("objects");
        for (node, want, wave) in [
            ("lane_proj", LANE_PROJ_OBJECT_SHA, 64),
            ("lane_wo", RES_LANES_OBJECT_SHA, 32),
        ] {
            let n = build_node(node, None).unwrap();
            let inc = n.incumbent().unwrap();
            let b = jit::compile_for(n.arch(), &dir, node, &inc, &inc.source).unwrap();
            assert_eq!(b.image_sha256, want, "{node}: not the runtime's object");
            assert_eq!(b.kernel(&inc.symbol).unwrap().wavefront_size, wave);
        }
    }

    #[test]
    fn offline_records_copy_equals_incumbent_and_a_different_candidate_differs() {
        if !jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let p = build_node("lane_proj", None)
            .unwrap()
            .incumbent()
            .unwrap()
            .source;
        let w = build_node("lane_wo", None)
            .unwrap()
            .incumbent()
            .unwrap()
            .source;
        let r = offline("lane_proj", "group=dn,lanes=8", &p, "p-copy");
        assert_eq!(r["identical_code"], true, "{r}");
        assert_eq!(r["candidate"]["code_sha256"], r["baseline"]["code_sha256"]);
        assert_eq!(r["candidate_co"]["sha256"], LANE_PROJ_OBJECT_SHA);
        assert_eq!(r["arch"], "gfx1201");
        assert_eq!(
            r["baseline"]["geometry"]["grid_workgroups"],
            json!([96, 1, 1])
        );
        let mutated = p.replace(
            "#define HIPFIRE_LANE_PROJ_COLGROUP 2",
            "#define HIPFIRE_LANE_PROJ_COLGROUP 4",
        );
        assert_ne!(mutated, p);
        let r = offline("lane_proj", "group=dn,lanes=8", &mutated, "p-mut");
        assert_eq!(r["identical_code"], false);
        assert_ne!(r["candidate"]["code_sha256"], r["baseline"]["code_sha256"]);
        let r = offline("lane_wo", "lanes=8,m=2048", &w, "w-copy");
        assert_eq!(r["identical_code"], true, "{r}");
        assert_eq!(r["candidate_co"]["sha256"], RES_LANES_OBJECT_SHA);
        assert_eq!(
            r["baseline"]["geometry"]["grid_workgroups"],
            json!([256, 1, 1])
        );
        let mutated = w.replace("row_base + r < M) y[", "row_base + r <= M) y[");
        assert_ne!(mutated, w);
        let r = offline("lane_wo", "lanes=8,m=2048", &mutated, "w-mut");
        assert_eq!(r["identical_code"], false);
    }

    #[test]
    fn header_geometry_reaches_the_record_and_null_control_runs() {
        if !jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let w = build_node("lane_wo", None)
            .unwrap()
            .incumbent()
            .unwrap()
            .source;
        let r = offline(
            "lane_wo",
            "lanes=4,m=2048",
            &format!("// hip_loop: rows_per_wg=16 block=256\n{w}"),
            "w-hdr",
        );
        assert_eq!(
            r["candidate"]["geometry"]["grid_workgroups"],
            json!([128, 1, 1])
        );
        assert_eq!(r["candidate"]["geometry"]["block"], json!([256, 1, 1]));
        assert_eq!(r["same_geometry"], false);
        let dir = tmp("aa");
        let o = Opts {
            node: "lane_proj".into(),
            shape: Some("group=moe,lanes=3".into()),
            null_control: true,
            offline: true,
            out: Some(dir.join("out")),
            ..Opts::default()
        };
        let (rec, outcome) = run(&o);
        assert_eq!(outcome, Outcome::Ok, "{rec}");
        assert_eq!(rec["null_control"]["identical_objects"], true, "{rec}");
        assert_eq!(rec["symbol"], "gemv_mq4g256v2_lane_proj_k2048_gfx1201_n3");
    }

    #[test]
    fn cli_takes_flush_and_emu_extent_and_refuses_them_elsewhere() {
        let o = parse_args(
            &[
                "--node",
                "lane_proj",
                "--cand",
                "x.hip",
                "--flush",
                "--emu-extent",
                "sampled",
            ]
            .map(String::from),
        )
        .unwrap();
        assert!(o.flush);
        assert_eq!(o.emu_extent.as_deref(), Some("sampled"));
        assert!(parse_args(
            &[
                "--node",
                "lane_wo",
                "--cand",
                "x.hip",
                "--emu-extent",
                "some"
            ]
            .map(String::from)
        )
        .is_err());
        let (rec, outcome) = run(&Opts {
            node: "mq6_res".into(),
            flush: true,
            null_control: true,
            offline: true,
            ..Opts::default()
        });
        assert_eq!(outcome, Outcome::Unusable);
        assert_eq!(rec["error"]["stage"], "args");
        // The flush shape key reaches the node and the record.
        let n = build_node("lane_wo", Some("lanes=2,flush=1")).unwrap();
        assert!(n.flush_requested());
        assert_eq!(n.shape_json()["flush"], true);
    }

    #[test]
    fn cli_takes_identity_depth_and_identity_only() {
        let p = |a: &[&str]| parse_args(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        assert_eq!(
            p(&["--node", "lane_wo", "--cand", "x.hip"])
                .unwrap()
                .identity,
            None
        );
        for m in ["full", "smoke", "skip"] {
            let o = p(&["--node", "lane_wo", "--cand", "x.hip", "--identity", m]).unwrap();
            assert_eq!(o.identity.as_deref(), Some(m));
        }
        assert!(p(&["--node", "lane_wo", "--cand", "x.hip", "--identity", "some"]).is_err());
        let o = p(&[
            "--node",
            "lane_proj",
            "--cand-object",
            "x.co",
            "--identity-only",
            "--seed",
            "3",
        ])
        .unwrap();
        assert!(o.identity_only);
        // The matrix needs the device and the full depth.
        assert!(p(&[
            "--node",
            "lane_proj",
            "--cand",
            "x.hip",
            "--identity-only",
            "--offline"
        ])
        .is_err());
        assert!(p(&[
            "--node",
            "lane_proj",
            "--cand",
            "x.hip",
            "--identity-only",
            "--identity",
            "skip"
        ])
        .is_err());
        // lane_proj's matrix runs over every group, lane_wo's over its own shape.
        assert_eq!(
            build_node("lane_proj", Some("lanes=8"))
                .unwrap()
                .identity_shapes(),
            ["group=dn,lanes=8", "group=fa,lanes=8", "group=moe,lanes=8"]
        );
        assert!(build_node("lane_wo", None)
            .unwrap()
            .identity_shapes()
            .is_empty());
    }

    /// The wave32 `wo` kernel through the pre-card gate at its small shape: the incumbent passes, a
    /// candidate that stores one row past `M` (the last lane's row `M` is past the end of `y`) fails with an
    /// unmapped access. The static hazard receipt is clean on the incumbent.
    #[test]
    fn lane_wo_emu_gate_passes_the_incumbent_and_fails_an_out_of_bounds_store() {
        if !jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let dir = tmp("wo-gate");
        let n = build_node("lane_wo", Some("emu")).unwrap();
        let inc = n.incumbent().unwrap();
        let good = jit::compile_for(n.arch(), &dir, "good", &inc, &inc.source).unwrap();
        let bad_src = inc
            .source
            .replace("row_base + r < M) y[", "row_base + r <= M) y[");
        assert_ne!(bad_src, inc.source);
        let bad = jit::compile_for(n.arch(), &dir, "bad", &inc, &bad_src).unwrap();
        let geo = n.geometry();
        let mut case = n.emu_case(1).unwrap();
        let (pass, rec) = emu::gate(
            &inc.symbol,
            &mut case,
            (&good.image, &geo),
            (&good.image, &geo),
            n.shape_json(),
            1,
        );
        assert!(pass, "{}", rec["reason"]);
        assert_eq!(rec["arms"]["baseline"]["arch"], "Gfx1201");
        let (pass, rec) = emu::gate(
            &inc.symbol,
            &mut case,
            (&good.image, &geo),
            (&bad.image, &geo),
            n.shape_json(),
            1,
        );
        assert!(!pass);
        assert!(
            rec["reason"].as_str().unwrap().contains("unmapped"),
            "{}",
            rec["reason"]
        );
        let (ok, hz) = emu::hazards(&good.image, &inc.symbol);
        assert!(ok, "{hz}");
        assert_eq!(hz["wave"], 32);
    }

    /// The wave64 `lane_proj` kernel through the gate at each group's small shape: the incumbent passes over the full
    /// grid; a candidate that stores one word past a segment's last row (into the unwritten pad between the packed
    /// output segments, or into the next lane's row 0) differs from the incumbent; the hazard receipt is clean.
    #[test]
    fn lane_proj_emu_gate_passes_the_incumbent_and_fails_a_store_past_a_segment() {
        if !jit::hipcc_available() {
            eprintln!("skip: no hipcc");
            return;
        }
        let dir = tmp("proj-gate");
        for shape in ["emu,group=fa,lanes=3", "emu,group=moe,lanes=1"] {
            let n = build_node("lane_proj", Some(shape)).unwrap();
            let inc = n.incumbent().unwrap();
            let good = jit::compile_for(n.arch(), &dir, "good", &inc, &inc.source).unwrap();
            let bad_src = inc.source.replace(
                "yp[r][(long long)b * mseg[r] + rowi[r]] = a;",
                "yp[r][(long long)b * mseg[r] + rowi[r] + (rowi[r] == mseg[r] - 1 ? 1 : 0)] = a;",
            );
            assert_ne!(bad_src, inc.source);
            let bad = jit::compile_for(n.arch(), &dir, "bad", &inc, &bad_src).unwrap();
            let geo = n.geometry();
            let mut case = n.emu_case(1).unwrap();
            let (pass, rec) = emu::gate(
                &inc.symbol,
                &mut case,
                (&good.image, &geo),
                (&good.image, &geo),
                n.shape_json(),
                1,
            );
            assert!(pass, "{shape}: {}", rec["reason"]);
            assert_eq!(rec["arms"]["baseline"]["wave"], 64);
            let (pass, rec) = emu::gate(
                &inc.symbol,
                &mut case,
                (&good.image, &geo),
                (&bad.image, &geo),
                n.shape_json(),
                1,
            );
            assert!(
                !pass,
                "{shape}: a store past the segment must fail the gate"
            );
            // Either the output differs from the incumbent's, or two waves now store the same word and the
            // schedules diverge (the gate model's own finding): both are failures of the store past the segment.
            let why = rec["reason"].as_str().unwrap();
            assert!(
                why.contains("differs") || why.contains("[schedule]"),
                "{why}"
            );
            let (ok, hz) = emu::hazards(&good.image, &inc.symbol);
            assert!(ok, "{hz}");
            assert_eq!(hz["wave"], 64);
        }
    }
}
