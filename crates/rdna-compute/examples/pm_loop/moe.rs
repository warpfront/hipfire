// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `--kernel qwen4_moe_sym`: the grouped symmetric IU4 MoE GEMMs of the Qwen4
//! prefill route on gfx1201 (default), gfx1151 or gfx1100 (`--arch`;
//! `peacemaker-isa emit --kernel qwen4_moe_sym`,
//! `crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rs`, weight codec
//! `WeightCodec::Mq4Sym`: pointer-addressed QT44 gate/up and QT53 down). The arch selects the
//! shipped baseline object/symbol, the candidate generator, the R1 contract exports and the
//! launch geometry; see `# Shipped baseline`.
//!
//! Included by the driver with `#[path = "pm_loop/moe.rs"] mod moe;`.
//!
//! # Knobs (what exists today)
//!
//! * `--epi gate_up|down|gate_up_ntN|down_ntN|down_nt4rR|down_gp_nt4|down_gp_nt8` (default
//!   `gate_up`): the candidate entry. Everything but `down_gp_*` is parsed by
//!   `Spec::from_epi(epi, arch)` and emitted with `qwen4_moe_sym::emit` (gfx1100: the NT1
//!   `qwen4_moe_sym_gfx11::emit`, `Spec::validate` refuses gfx1100). **`N` is the tile
//!   width** (16-slot tiles per weight stream: NT1/NT4/NT8 for gfx1201 gate/up and down, NT1/NT4
//!   for gfx1151, NT1 only for gfx1100) and
//!   `R` repeats a 64-row block per CTA (gfx1151 down NT4 only: `down_nt4r2|r4`; refused by
//!   `validate` on gfx1201). `gate_up_stream8` (gfx1151 only) is the streamed NT8 gate/up
//!   (`emit_gate_up_stream8`, object `QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151`).
//!   `down_gp_*` exists on gfx1201 only (a gfx1151/gfx1100 `--epi`/`--base-epi` naming it is refused,
//!   as are gfx1100 NT>1 and gfx1151 NT8: `Spec::validate` verbatim).
//!   `down_gp_nt4|nt8` is the grouped-position twin of the down NT4/NT8 entry
//!   (`qwen4_moe_sym::emit_down_gp`, bundle `qwen4_moe_down_gp_pm_gfx1201`): same 64-byte ABI,
//!   grid and output bytes; the activation plane is `[K/128][m_total]` records in grouped
//!   order (`g * m_total + row`) instead of `[K/128][x_src_rows]` flat slots, so each tile reads
//!   its 16 records contiguously. `validate` errors are returned verbatim by `candidate()`.
//! * `--base-epi <epi>`: emit the baseline from that entry instead of the shipped one (same kind
//!   as the candidate; a `down_gp_*` or flat `down_*` baseline each get their own plane).
//! * `--ref-rows full|sample:<n>` (default `sample:128`): how many live grouped rows the CPU
//!   reference computes (all output columns of each); `sample` rows are spaced evenly through the
//!   grouped order (first and last included), so they span experts, tiles, tile positions and the
//!   ragged last tile.
//!
//! The tile width and the `gp` plane are the only entry-level flags. **Ring depth, wait placement,
//! K-loop prefetch order, the per-tile branches of a fold and the LDS hand-off are script-level
//! (`--rip`)**: `--rip crates/peacemaker-isa/src/kernels/qwen4_moe_sym.rip` runs the committed
//! twin of every entry above (`gate_up|down` at NT1/NT4/NT8, and `down_gp_nt4|nt8`), composed from
//! the `moent::*` / `moe1::*` natives of `qwen4_moe_sym::{blocks, nt1_blocks}`; edit the script and
//! re-run with no rebuild. The twin is byte-identical to `qwen4_moe_sym::emit` /
//! `emit_down_gp` (`candidate.rip.identical_to_generator`); an edited script that keeps the ABI is
//! timed against the shipped baseline and reported as `identical_to_baseline: false`. The
//! chains/prefetch/grouped axes of `R4Spec` stay private Rust experiments (`pm_r4_screen`).
//!
//! `--cand-object FILE` (`--arch gfx1100` only; `pm_loop.rs`): FILE is the candidate code object
//! instead of the emit, e.g. `pm-krsi-1100/emit` builds a modified copy of
//! `crates/peacemaker-isa/src/kernels/qwen4_moe_sym_gfx1100/` into `gate_up.co` / `down.co`
//! (`tools/moe_check.sh`). It must expose the entry's symbol with the in-tree emit's ABI
//! (kernarg layout, workgroup size, LDS, wave mode; refused otherwise, stage `cand_object`); M7,
//! resources and `code_sha256` then run as for an emitted candidate (`candidate.cand_object`
//! = `{path, sha256, variant, identical_to_generator}`) and the GPU path (identity vs shipped,
//! CPU reference, timing) is unchanged.
//!
//! Correctness requires the candidate to be bitwise equal to the shipped baseline (every NT and
//! `gp` entry is byte-identical to the others on the same activations) AND to pass the CPU
//! reference below.
//!
//! # Presets
//!
//! `--shape fn:<name>` (plus any further `key=value` items) = Flash-Next `E=512,TOP=10,H=2560,
//! I=640` (`crates/hipfire-arch-qwen4/src/artifact.rs:1833-1843`: 512 experts, top-10;
//! `admission.rs:1019-1022`: `hidden_size 2560`, `moe_intermediate_size 640`) at the named token
//! count, with the named entry as the default `--epi`:
//!
//! | preset                    | entry         | T     | shipped baseline at this shape            |
//! |---------------------------|---------------|-------|-------------------------------------------|
//! | `fn:gate_up_nt4-t4096`    | `gate_up_nt4` | 4096  | gate/up NT4 (40960 slots < 128 * 512)     |
//! | `fn:gate_up_nt4-t16384`   | `gate_up_nt4` | 16384 | gate/up NT8 (163840 slots >= 128 * 512)   |
//! | `fn:down_gp_nt4-t4096`    | `down_gp_nt4` | 4096  | shipped gp down NT4                       |
//! | `fn:down_gp_nt4-t16384`   | `down_gp_nt4` | 16384 | shipped gp down NT4                       |
//!
//! The Flash-Next route plan `route_plans/qwen3.8-flash-next.gfx1201.tsv` walks AR decode only: it
//! lists the `qwen4_moe_iu4_sym_pm_gfx1201` bundle, not `qwen4_moe_down_gp_pm_gfx1201` (prefill's
//! grouped plane is selected by `qwen4_moe_rotate128_i4_gp_selected`, `moe_rotate_i4_pm.rs`).
//!
//! # Shape grammar (`--shape`)
//!
//! `[<preset>,] key=value, ...` (items in any order; empty = the default):
//!
//! * preset `fn` (default): Flash-Next `E=512,TOP=10,H=2560,I=640`
//!   (`crates/peacemaker-isa/tests/qwen4_moe_sym.rs` geometry test;
//!   `examples/qwen4_moe_sym.rs:96 GEO_FN`); preset `a3b`: `E=256,TOP=8,H=2048,I=512`
//!   (`examples/qwen4_moe_sym.rs:97 GEO_A3B`).
//! * `E` routed experts (pointer-table length = kernarg `n_experts`), `TOP`
//!   experts per token, `T` prefill tokens (default 4096), `H` hidden, `I` per-expert
//!   intermediate width, `D` distinct synthetic experts (default `E`; expert `e`
//!   aliases synthetic expert `e % D`, a memory cap), `seed` (default 1),
//!   `route` = `uniform` (default: `TOP` distinct uniform experts per token) |
//!   `skew` (experts 0..min(5,TOP) on every token, rest uniform distinct) |
//!   `all` (flat slot `f` -> expert `f % E`) | `one` (every slot to expert 0) |
//!   `hist:<route_hist.json>[@L]` (replay the recorded per-token expert lists of a pinned
//!   `route_hist.v1`, layer `L` or all layers' records in record order, cycling past the recording;
//!   E and TOP must equal the recording's `n_experts`/`topk`, the ids bin is sha256-verified, the
//!   record gets `route: {kind:"hist", path, id, layer}` and the shape `route_id`/`route_layer`).
//!
//! The kind (`gate_up` or `down`) comes from `--epi`; the GEMM dims follow
//! `ExpertGeometry::plan` (so its admission rules apply):
//!
//! | kind    | M (kernarg) | K (kernarg) | `x_row_div` | `x_src_rows` | Y row (BF16) |
//! |---------|-------------|-------------|-------------|--------------|--------------|
//! | gate_up | `2I`        | `H`         | `TOP`       | `T`          | `M/2 = I`    |
//! | down    | `H`         | `I`         | `1`         | `T * TOP`    | `M = H`      |
//!
//! # Launch ABI (reproduced from production)
//!
//! `qwen4_moe_sym_gemm` (`crates/rdna-compute/src/gemm.rs:52361`; callers
//! `gemm_qwen4_moe_{gate_up_silu,down}_iu4_sym` at `:51771`, `:51819`):
//! kernarg 64 B (`Spec::kernargs`, `KERNARG_BYTES`):
//! `expert_weight_ptrs@0, expert_tile_ids@8, sorted_slot_index@16, Xq@24,
//! Y_grouped@32` (device pointers) then `M@40, K@44, x_row_div@48, m_total@52,
//! x_src_rows@56, n_experts@60` (i32). `n_experts` is the pointer-table length
//! (`moe_expert_table_len`, `gemm.rs:1539`), `m_total` the grouped rows. Grid
//! `(M / (64 * rr), m_total / 16, 1)`, block `Spec::threads()` (64 for the NT1
//! gate/up, else 128), **no dynamic LDS** (the NT gate/up LDS is static in the
//! descriptor; `gemm.rs:52476` passes 0).
//!
//! Grouping (`examples/qwen4_moe_sym.rs:395-428 cpu_group`, `make_case:557-560`):
//! the flat slot `f = token * TOP + j` is stable-sorted by (expert, `f`); each
//! expert's run is padded to a multiple of 16 with `-1`; `sorted_slot_index[p]`
//! is the flat slot of grouped row `p` (or `-1`), `expert_tile_ids[t]` the expert
//! of tile `t = p / 16` (`-1` for unused tiles). `m_total = pmax =
//! ceil16(T*TOP + E*15)` so, as in production, trailing sentinel tiles exist and
//! exit before any load or store.
//!
//! Weights: expert `e`'s tensor is `M` rows of `row_bytes = (K/128)*68` B; gate/up
//! = QT44 groups of 136 B per K256 (`[f16 sc0, zp0, sc1, zp1][64 B nibbles epoch 2g]
//! [64 B epoch 2g+1]`), down = QT53 groups of 68 B per K128 (`[f16 sc, zp][64 B]`),
//! every header `zp == -8*sc` (`w = sc * (u - 8)`); `expert_weight_ptrs[e]` points at
//! it. Activations `Xq`: `block_i4_128` (`[f32 d, i32 s, 64 B]`, 72 B, nibble `j` of
//! byte `j/2` low first, signed), epoch-major `[K/128][x_src_rows]`, row
//! `slot / x_row_div` (`iu4_fold.rs:30`, HIP twin `qwen4_moe_iu4_sym.gfx1151.hip:12-20`).
//!
//! # Shipped baseline (VRAM-resident experts; one row per `--arch`, default gfx1201)
//!
//! | arch | gate/up | down | object (`rdna_compute::kernels`, `kernels.rs`) | selection source (`gemm.rs`) |
//! |------|---------|------|------------------------------------------------|------------------------------|
//! | gfx1201 | NT4; NT8 iff `total_slots >= 128 * E` | NT4 (gp NT4 with `down_gp_*`) | `QWEN4_MOE_IU4_SYM_PM_GFX1201` (`:4467`), `QWEN4_MOE_DOWN_GP_PM_GFX1201` (`:10783`) | `qwen4_moe_sym_nt8_selected :49334`, `..._wide_shape :49313`, tables `:49277-49296`, row `:50101-50125` |
//! | gfx1151 | NT4; streamed NT8 (`..._nt8_stream2`) iff `[M,K,top,rows,E] = [1280,2560,10,7168..=8192,512]` | NT4 rr1 (`HIPFIRE_QWEN4_MOE_SYM_DOWN_RR` unset; `nt4r2/r4` are opt-ins) | `QWEN4_MOE_IU4_SYM_PM_GFX1151` (`:4465`), `QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151` (`:4472`) | `qwen4_moe_sym_gemm_symbols :50704`, row `:50072-50100`, `qwen4_moe_sym_gate_up_symbol :50779`, `qwen4_moe_stream8_selected :56516`, `qwen4_moe_sym_down_rr :50802`, rr entries `:50307` |
//! | gfx1100 | NT1 | NT1 | `QWEN4_MOE_IU4_SYM_PM_GFX1100` (`:4495`) | row `:50038-50071`, NT1 early-out `:50711`, `qwen4_moe_sym_nt1_gemm :52067` |
//!
//! A shipped entry an arch lacks (the gp down off gfx1201, anything but NT1 on gfx1100) is
//! refused with the entry and arch named; a gfx1201 object is never retagged. The geometry plan
//! of gfx1100 is asked on gfx1151 (`Spec::validate` refuses gfx1100; the plan is pure
//! arithmetic and the K2 module shares its admission). The gfx1201 detail follows.
//!
//! Production loads `rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201`
//! (`kernels.rs:4467`, `kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco`, six exports: NT1,
//! NT4, NT8 x gate/up, down; `MoeSymIu4Kernel` row `gemm.rs:50101-50125`) and picks the
//! symbol per call: **down is always NT4**; **gate/up is NT4 unless the real slots
//! `x_src_rows * x_row_div` reach 128 per table expert (`total_slots >= 128 * E`),
//! then NT8** (`qwen4_moe_sym_nt8_selected`, `gemm.rs:49334`; `qwen4_moe_sym_wide_shape`,
//! `gemm.rs:49313`; flag `HIPFIRE_MOE_IU4_WIDE_TILES` unset = on at gfx1201,
//! `prefill_route_capabilities.rs:303,327`; the tables `gemm.rs:49277-49296`). Host-mapped
//! experts (NT8 for both GEMMs) are not modelled: weights here are VRAM-resident; use
//! `--base-epi gate_up_nt8` / `down_nt8` for that arm. All NT entries share the 64-byte
//! ABI, grid and output bytes (each output byte equals the NT1 entry's;
//! `qwen4_moe_sym.rs:1544-1554`), so a candidate of any `nt` is directly comparable
//! with the shipped symbol of a different `nt`; `baseline().source` records the nt.
//! The gfx1151 / gfx1100 records carry their own `const`, `artifact`, `sha256`, `symbol`,
//! `selection` and a `generator` field (the non-gfx1201 emitter), and the launch grid is
//! `M / (64 * rr)` (`Spec::rr`) for every arch.
//!
//! # Correctness reference
//!
//! The CPU reference reproduces the exact fold DAG of
//! `examples/qwen4_moe_sym.rs:262-328 fold_reference` (the oracle of the PM route; HIP twin
//! `kernels/src/qwen4_moe_iu4_sym.gfx1151.hip:19`): per output and ascending K128 epoch `h`,
//! `C_h` = exact int32 dot of the signed weight nibbles and signed activations, `sum =
//! fma(RN(sc_h * d_h), float(C_h), sum)` from `+0`. Down stores `bf16_rne(sum)` (a pure
//! rounding: **bit-exact, tolerance 0**). Gate/up stores `bf16_rne(silu(rt(g)) * rt(u))`
//! with `rt` the BF16 round trip; the SwiGLU is the portable-math twin of
//! `mq_i4_reference.rs:926-1068` (`f32::exp2`, exact `ldexp`, IEEE division), documented there as a
//! portable reference, **not a device byte oracle** (the kernel uses `v_exp_f32` / `v_rcp_f32`
//! sequences), so gate/up is compared to **1 BF16 ulp** (fold inputs are bit-exact). Padding
//! rows and sentinel-tile rows must be exactly `+0` / untouched (zero) and are always scanned
//! in full; only the sampled live rows are compared numerically.
//!
//! `Workload::emu` runs the candidate on the peacemaker emulator
//! (`peacemaker_ir::emu`) at a tiny shape (`E=2,TOP=2,T=20,H=512,I=128`) against the same
//! reference; if the emulator cannot lift/decode/execute some instruction (on any arch) it
//! refuses with its own message verbatim.
//!
//! The reference is arch-independent: gfx11 (WMMA `16x16x16`, eight K16 steps per K128 epoch,
//! C rows `2j + hi`) and gfx12 (`16x16x32`, four K32 steps, rows `8hi + j`) differ only in which
//! accumulator row a lane holds; per epoch `C_h` is the exact int32 chain (`|C_h| <= 8192`) so
//! the K-step grouping cannot change it, and the epoch fold `fma(RN(d_h * sc_h), C_h, sum)` is
//! ascending from `+0` on every arch (`qwen4_moe_sym.rs` module doc; `qwen4_moe_sym_gfx11/mod.rs`
//! `# Numerics`; the gfx1100 epilogue reuses `Epilogue::silu_dense`). Tolerances (down 0,
//! gate/up 1 BF16 ulp) and `requires_baseline_identity` are therefore the same on every arch.

use crate::common::sha256_hex;
use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use half::f16;
use hip_bridge::DeviceBuffer;
use hipfire_kernel_abi::families::{find_export, qwen4_moe_sym as moe_abi};
use hipfire_kernel_abi::{
    check_export, eval, validate_invocation, Arch as AbiArch, ArgEvidence, Drift, EvidenceSource,
    ExportEvidence, InvocationV1, OperandBinding,
};
use peacemaker_isa::kernels::qwen4_moe_down_gp;
use peacemaker_isa::kernels::qwen4_moe_sym::{
    self, ExpertGeometry, Kind, Spec, WeightCodec, KERNARG_BYTES, MAX_SLOTS,
};
use peacemaker_isa::kernels::qwen4_moe_sym_gfx11 as gfx11;
use peacemaker_isa::Arch;
use rayon::prelude::*;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Contract-crate twin of `arch`; every arch of this adapter has a `qwen4_moe_sym` contract.
fn abi_arch(arch: Arch) -> Result<AbiArch> {
    match arch {
        Arch::Gfx1201 => Ok(AbiArch::Gfx1201),
        Arch::Gfx1151 => Ok(AbiArch::Gfx1151),
        Arch::Gfx1100 => Ok(AbiArch::Gfx1100),
        other => Err(format!(
            "qwen4_moe_sym: no kernel-abi contract for {} (gfx1100, gfx1151, gfx1201 only)",
            other.name()
        )),
    }
}

/// The arch the (arch-independent) geometry plan is asked on. `ExpertGeometry::plan` validates the
/// `Spec` ladder of the arch it is given, and `Spec::validate` refuses gfx1100 (its NT1 entries
/// come from `qwen4_moe_sym_gfx11`, not `Spec`); the gfx1100 NT1 kernels share the gfx1151 plan's
/// 64-row / K-multiple / 24-bit-row admission (`qwen4_moe_sym_gfx11::validate_launch`), so gfx1100
/// plans on gfx1151. The plan only supplies `M/K/x_row_div/row_bytes`; no object is retagged.
fn plan_arch(arch: Arch) -> Arch {
    if arch == Arch::Gfx1100 {
        Arch::Gfx1151
    } else {
        arch
    }
}

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// Cap on device memory the synthetic problem may take.
const DEVICE_CAP: u64 = 3 << 30;
/// `block_i4_128` record bytes.
const XBLK: usize = 72;
const DEFAULT_TOKENS: usize = 4096;
const DEFAULT_SAMPLE: usize = 128;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

// ---------------------------------------------------------------------------
// Deterministic PRNG: splitmix64 (as `examples/qwen4_moe_sym.rs:158`).

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

// ---------------------------------------------------------------------------
// Shape

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Uniform,
    Skew,
    All,
    One,
    Hist,
}

/// `--shape route=hist:<route_hist.json>[@L]`: the expert ids of a pinned `route_hist.v1`
/// (`HIPFIRE_MOE_ROUTE_DUMP` records, headerless `i32` LE bin), layer `L` or every layer's
/// records concatenated in record order. Token `t` takes recorded row `t % rows` (cycled).
#[derive(Clone, Debug, PartialEq)]
struct Hist {
    path: String,
    id: String,
    layer: Option<u32>,
    n_experts: usize,
    topk: usize,
    /// Row-major `rows x topk`.
    ids: Vec<i32>,
}

impl Hist {
    fn parse(spec: &str) -> Result<Hist> {
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
        Self::load(std::path::Path::new(path), layer)
    }

    fn load(path: &std::path::Path, layer: Option<u32>) -> Result<Hist> {
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
        let id = crate::iu4::artefact_id(&doc).map_err(err)?;
        let num = |v: &Value, what: &str| {
            v.as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| err(format!("missing or non-integer {what}")))
        };
        let n_experts = num(&doc["n_experts"], "n_experts")?;
        let topk = num(&doc["topk"], "topk")?;
        let bin_path = crate::iu4::sibling_path(
            path,
            doc["ids"]["path"]
                .as_str()
                .ok_or_else(|| err("missing ids.path".into()))?,
        );
        let bin =
            std::fs::read(&bin_path).map_err(|e| err(format!("{}: {e}", bin_path.display())))?;
        let (want, got) = (
            doc["ids"]["sha256"].as_str().unwrap_or(""),
            sha256_hex(&bin),
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
        Ok(Hist {
            path: path.display().to_string(),
            id,
            layer,
            n_experts,
            topk,
            ids,
        })
    }

    fn json(&self) -> Value {
        json!({"kind": "hist", "path": self.path, "id": self.id, "layer": self.layer})
    }
}

impl Route {
    fn name(self) -> &'static str {
        match self {
            Route::Uniform => "uniform",
            Route::Skew => "skew",
            Route::All => "all",
            Route::One => "one",
            Route::Hist => "hist",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Geo {
    arch: Arch,
    preset: &'static str,
    experts: usize,
    top: usize,
    tokens: usize,
    hidden: usize,
    inter: usize,
    distinct: usize,
    route: Route,
    /// `route=hist:<file>`: the recorded per-token expert lists (`Route::Hist`).
    hist: Option<Hist>,
    seed: u64,
}

fn num<T: std::str::FromStr>(key: &str, v: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    v.trim()
        .parse::<T>()
        .map_err(|e| format!("--shape {key}={v}: {e}"))
}

impl Geo {
    fn parse(shape: &str, arch: Arch) -> Result<Geo> {
        let items: Vec<&str> = shape
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        // Presets first so explicit keys override whatever the order.
        let mut preset = ("fn", 512usize, 10usize, 2560usize, 640usize);
        for item in items.iter().filter(|i| !i.contains('=')) {
            preset = match *item {
                "fn" => ("fn", 512, 10, 2560, 640),
                "a3b" => ("a3b", 256, 8, 2048, 512),
                other => {
                    return Err(format!(
                        "--shape item `{other}`: expected a preset (fn|a3b) or key=value \
                         (E,TOP,T,H,I,D,seed,route)"
                    ))
                }
            };
        }
        let mut g = Geo {
            arch,
            preset: preset.0,
            experts: preset.1,
            top: preset.2,
            tokens: DEFAULT_TOKENS,
            hidden: preset.3,
            inter: preset.4,
            distinct: 0,
            route: Route::Uniform,
            hist: None,
            seed: 1,
        };
        let mut distinct = None;
        for item in items.iter().filter(|i| i.contains('=')) {
            let Some((k, v)) = item.split_once('=') else {
                continue;
            };
            match k.trim() {
                "E" => g.experts = num(k, v)?,
                "TOP" => g.top = num(k, v)?,
                "T" => g.tokens = num(k, v)?,
                "H" => g.hidden = num(k, v)?,
                "I" => g.inter = num(k, v)?,
                "D" => distinct = Some(num::<usize>(k, v)?),
                "seed" => g.seed = num(k, v)?,
                "route" => {
                    let v = v.trim();
                    if let Some(spec) = v.strip_prefix("hist:") {
                        g.hist = Some(Hist::parse(spec)?);
                        g.route = Route::Hist;
                        continue;
                    }
                    g.route = match v {
                        "uniform" => Route::Uniform,
                        "skew" => Route::Skew,
                        "all" => Route::All,
                        "one" => Route::One,
                        other => {
                            return Err(format!(
                                "--shape route={other}: expected uniform|skew|all|one|hist:<route_hist.json>[@L]"
                            ))
                        }
                    }
                }
                other => {
                    return Err(format!(
                        "--shape key `{other}`: expected E,TOP,T,H,I,D,seed,route"
                    ))
                }
            }
        }
        g.distinct = distinct.unwrap_or(g.experts);
        g.validate()?;
        Ok(g)
    }

    fn slots(&self) -> usize {
        self.tokens * self.top
    }

    fn expert_geometry(&self) -> Result<ExpertGeometry> {
        let u = |name: &str, v: usize| {
            u32::try_from(v).map_err(|_| format!("--shape {name}={v} exceeds u32"))
        };
        Ok(ExpertGeometry {
            experts: u("E", self.experts)?,
            top_k: u("TOP", self.top)?,
            hidden: u("H", self.hidden)?,
            intermediate: u("I", self.inter)?,
        })
    }

    fn validate(&self) -> Result<()> {
        if self.tokens == 0 {
            return Err("--shape T must be >= 1".into());
        }
        if self.distinct == 0 || self.distinct > self.experts {
            return Err(format!(
                "--shape D={} must be in 1..=E ({})",
                self.distinct, self.experts
            ));
        }
        let slots = self
            .tokens
            .checked_mul(self.top)
            .ok_or("--shape T*TOP overflows")?;
        if slots > MAX_SLOTS as usize {
            return Err(format!(
                "--shape T*TOP = {slots} exceeds the gather's exact slot range {MAX_SLOTS}"
            ));
        }
        let hot = self.hot();
        if matches!(self.route, Route::Uniform | Route::Skew) && self.top > self.experts {
            return Err(format!(
                "--shape route={} needs TOP ({}) <= E ({})",
                self.route.name(),
                self.top,
                self.experts
            ));
        }
        if hot > self.experts {
            return Err("--shape route=skew needs E >= min(5, TOP)".into());
        }
        if let Some(h) = &self.hist {
            if h.n_experts != self.experts || h.topk != self.top {
                return Err(format!(
                    "--shape route=hist:{}: recording has E={} TOP={}, the shape has E={} TOP={}",
                    h.path, h.n_experts, h.topk, self.experts, self.top
                ));
            }
        }
        self.expert_geometry()?
            .plan(plan_arch(self.arch), WeightCodec::Mq4Sym)?;
        Ok(())
    }

    /// Experts every token picks under `route=skew`.
    fn hot(&self) -> usize {
        if self.route == Route::Skew {
            self.top.min(5)
        } else {
            0
        }
    }

    /// Flat slot (`token * TOP + j`) -> expert id.
    fn route_table(&self) -> Vec<u32> {
        let l = self.slots();
        let mut rng = Rng(self.seed ^ 0xabcdef);
        match self.route {
            Route::One => vec![0; l],
            Route::All => (0..l).map(|f| (f % self.experts) as u32).collect(),
            Route::Hist => {
                let ids = &self
                    .hist
                    .as_ref()
                    .expect("Route::Hist carries its recording")
                    .ids;
                (0..l).map(|f| ids[f % ids.len()] as u32).collect()
            }
            Route::Uniform | Route::Skew => {
                let hot = self.hot();
                let mut v = Vec::with_capacity(l);
                for _ in 0..self.tokens {
                    let mut row: Vec<u32> = (0..hot as u32).collect();
                    while row.len() < self.top {
                        let e = (hot as u64 + rng.below((self.experts - hot) as u64)) as u32;
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    v.extend(row);
                }
                v
            }
        }
    }
}

/// How many live grouped rows the reference computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sample {
    Full,
    Rows(usize),
}

impl Sample {
    fn parse(s: &str) -> Result<Sample> {
        if s == "full" {
            return Ok(Sample::Full);
        }
        match s.strip_prefix("sample:") {
            Some(n) => match n.parse::<usize>() {
                Ok(n) if n >= 1 => Ok(Sample::Rows(n)),
                _ => Err(format!("--ref-rows {s}: sample:<n> needs n >= 1")),
            },
            None => Err(format!("--ref-rows {s}: expected full or sample:<n>")),
        }
    }
    fn label(self) -> String {
        match self {
            Sample::Full => "full".into(),
            Sample::Rows(n) => format!("sample:{n}"),
        }
    }
}

/// `n` live rows spaced evenly through `live` (first and last included).
fn pick_evenly(live: &[usize], n: usize) -> Vec<usize> {
    if n >= live.len() {
        return live.to_vec();
    }
    if n == 1 {
        return vec![live[0]];
    }
    (0..n)
        .map(|i| live[i * (live.len() - 1) / (n - 1)])
        .collect()
}

// ---------------------------------------------------------------------------
// Synthetic problem (host side, deterministic)

/// `(sc, zp)` header with `zp == -8*sc` exactly: independently varied normal
/// scales of both signs (2^-12 .. 2^-4), plus zero groups of both zero signs.
/// `examples/qwen4_moe_sym.rs:198-211`.
fn sym_header(r: &mut Rng, span: u64) -> (u16, u16) {
    match r.below(64) {
        0 => (0x0000, 0x8000),
        1 => (0x8000, 0x0000),
        _ => {
            let sign = (r.below(2) as u16) << 15;
            let exp = 3 + r.below(span) as u16;
            let man = r.below(1024) as u16;
            let sc = sign | (exp << 10) | man;
            let zp = (sign ^ 0x8000) | ((exp + 3) << 10) | man;
            (sc, zp)
        }
    }
}

/// One expert's rows: QT44 (136 B groups, `K % 256`) or QT53 (68 B groups,
/// `K % 128`), every header symmetric, every nibble code uniform.
/// `examples/qwen4_moe_sym.rs:223-244`.
fn symmetric_weights_into(w: &mut [u8], group_bytes: usize, seed: u64, span: u64) {
    let mut r = Rng(seed ^ 0x5eed_0000_0000_0001);
    let hdrs = group_bytes / 68;
    for g in w.chunks_exact_mut(group_bytes) {
        for h in 0..hdrs {
            let (sc, zp) = sym_header(&mut r, span);
            g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for c in g[4 * hdrs..].chunks_mut(8) {
            let v = r.next().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }
}

/// Byte offsets of the header word and the 64 nibble bytes of K128 epoch `h`
/// of weight row `row` (`examples/qwen4_moe_sym.rs:247-256`, kernel
/// `qwen4_moe_sym.rs` `header_off` / `nibble_off`).
fn locate(kind: Kind, row_bytes: usize, row: usize, h: usize) -> (usize, usize) {
    match kind {
        Kind::GateUp => {
            let g = row * row_bytes + (h / 2) * 136;
            (g + 4 * (h % 2), g + 8 + 64 * (h % 2))
        }
        Kind::Down => {
            let g = row * row_bytes + h * 68;
            (g, g + 4)
        }
    }
}

/// One activation row decoded: per-epoch scale and `epochs * 128` signed values.
pub(crate) struct XRow {
    pub(crate) d: Vec<f32>,
    pub(crate) v: Vec<i8>,
}

/// Exact fold of `m` weight rows against each activation row: `[xs.len()][m]`
/// f32 sums, `sum = fma(RN(sc_h * d_h), float(C_h), sum)` from `+0`, ascending
/// epochs (`examples/qwen4_moe_sym.rs:262-328`). `C_h` is an exact integer sum,
/// so its evaluation order is irrelevant.
pub(crate) fn fold_slots(
    kind: Kind,
    m: usize,
    epochs: usize,
    row_bytes: usize,
    w: &[u8],
    xs: &[XRow],
) -> Vec<f32> {
    let mut sums = vec![0f32; xs.len() * m];
    let mut w8 = [0i8; 128];
    for row in 0..m {
        for h in 0..epochs {
            let (hdr, nib) = locate(kind, row_bytes, row, h);
            let sc = f16::from_bits(u16::from_le_bytes([w[hdr], w[hdr + 1]])).to_f32();
            for j in 0..64 {
                let b = w[nib + j];
                w8[2 * j] = (b & 15) as i8 - 8;
                w8[2 * j + 1] = (b >> 4) as i8 - 8;
            }
            for (s, x) in xs.iter().enumerate() {
                let xv = &x.v[h * 128..(h + 1) * 128];
                let c: i32 = w8
                    .iter()
                    .zip(xv)
                    .map(|(&a, &b)| i32::from(a) * i32::from(b))
                    .sum();
                let acc = &mut sums[s * m + row];
                *acc = (sc * x.d[h]).mul_add(c as f32, *acc);
            }
        }
    }
    sums
}

/// BF16 RNE of raw f32 bits, non-finite passed through as the high half
/// (`mq_i4_reference.rs:926-936`, kernel `sym_bf16_bits`).
fn bf16_bits(v: f32) -> u16 {
    let u = v.to_bits();
    if !v.is_finite() {
        return (u >> 16) as u16;
    }
    (u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1) >> 16) as u16
}

/// BF16 round trip (`mq_i4_reference.rs:941-948`, kernel `sym_bf16_rt`).
pub(crate) fn rt_f32(v: f32) -> f32 {
    if !v.is_finite() {
        return v;
    }
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1) & 0xffff_0000)
}

/// Portable-math SwiGLU `RN(IEEE(g / (1 + e^-g)) * u)` of `silu_dense`
/// (`mq_i4_reference.rs:1046-1068` with `PortableEpilogue`): a portable reference,
/// not a device byte oracle.
pub(crate) fn portable_silu_mul(g: f32, u: f32) -> f32 {
    let neg_log2e = f32::from_bits(0xbfb8_aa3b);
    let neg_log2e_lo = f32::from_bits(0xb2a5_705f);
    let ph = neg_log2e * g;
    let mut pl = neg_log2e.mul_add(g, -ph);
    pl = neg_log2e_lo.mul_add(g, pl);
    let e = ph.round_ties_even();
    let a = (ph - e) + pl;
    // Exact power-of-two scaling in f64, one rounding to f32 (gradual underflow).
    let pow2 = f64::from_bits((((e as i32).clamp(-400, 400) + 1023) as u64) << 52);
    let mut r = (f64::from(a.exp2()) * pow2) as f32;
    if f32::from_bits(0x42ce_8ed0) < g {
        r = 0.0;
    }
    if g < f32::from_bits(0xc2b1_7218) {
        r = f32::INFINITY;
    }
    (g / (1.0 + r)) * u
}

fn bf16_to_f64(b: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(b) << 16))
}

fn bf16_nonfinite(b: u16) -> bool {
    b & 0x7f80 == 0x7f80
}

fn bf16_nan(b: u16) -> bool {
    bf16_nonfinite(b) && b & 0x7f != 0
}

/// Monotonic ordinal of a finite BF16 (`+0` and `-0` both 0): adjacent BF16 values differ by 1.
fn bf16_ord(b: u16) -> i32 {
    if b & 0x8000 != 0 {
        -i32::from(b & 0x7fff)
    } else {
        i32::from(b)
    }
}

/// One BF16 ulp at the magnitude of `b` (`2^(e-134)`, subnormals at `e = 1`).
fn bf16_ulp(b: u16) -> f64 {
    let e = i32::from((b >> 7) & 0xff).max(1);
    2f64.powi(e - 134)
}

/// Per-row comparison accumulator of [`Problem::compare`].
#[derive(Clone, Copy, Debug, Default)]
struct Acc {
    err: f64,
    tol: f64,
    bad: usize,
    compared_rows: usize,
    pad_rows: usize,
    pad_bad: usize,
    sentinel_rows: usize,
    sentinel_bad: usize,
}

impl Acc {
    fn merge(a: Acc, b: Acc) -> Acc {
        Acc {
            err: a.err.max(b.err),
            tol: a.tol.max(b.tol),
            bad: a.bad + b.bad,
            compared_rows: a.compared_rows + b.compared_rows,
            pad_rows: a.pad_rows + b.pad_rows,
            pad_bad: a.pad_bad + b.pad_bad,
            sentinel_rows: a.sentinel_rows + b.sentinel_rows,
            sentinel_bad: a.sentinel_bad + b.sentinel_bad,
        }
    }
}

/// One GEMM call of the route: shape, routing tables and every host-side
/// generator the device arms and the reference share.
struct Problem {
    geo: Geo,
    kind: Kind,
    m: usize,
    k: usize,
    epochs: usize,
    x_row_div: usize,
    x_src_rows: usize,
    /// BF16 elements per grouped output row (gate/up `M/2`, down `M`).
    row_len: usize,
    group_bytes: usize,
    row_bytes: usize,
    /// Bytes of one expert tensor.
    stride: usize,
    /// Grouped rows `m_total` (padded runs plus sentinel tail), a multiple of 16.
    pmax: usize,
    sorted: Vec<i32>,
    tiles: Vec<i32>,
    xq: OnceLock<Vec<u8>>,
    /// Emulator gate only: narrow the synthetic weight scales (header exponent span 4 instead of 9) so no
    /// accumulator reaches the SwiGLU divisions the emulator has not measured. Never set on a device run.
    tame: bool,
}

impl Problem {
    fn new(geo: &Geo, kind: Kind) -> Result<Problem> {
        geo.validate()?;
        let plan = geo
            .expert_geometry()?
            .plan(plan_arch(geo.arch), WeightCodec::Mq4Sym)?;
        let pg = match kind {
            Kind::GateUp => plan.gate_up,
            Kind::Down => plan.down,
        };
        let (m, k) = (pg.m as usize, pg.k as usize);
        let row_bytes = pg.row_bytes as usize;
        let x_row_div = pg.x_row_div as usize;
        let x_src_rows = match kind {
            Kind::GateUp => geo.tokens,
            Kind::Down => geo.slots(),
        };
        let route = geo.route_table();
        let l = route.len();
        let e = geo.experts;
        let pmax = (l + e * 15).div_ceil(16) * 16;
        // CPU stable grouping, pad16 (`examples/qwen4_moe_sym.rs:395-428`).
        let mut counts = vec![0usize; e];
        for &x in &route {
            counts[x as usize] += 1;
        }
        let mut offsets = vec![0usize; e + 1];
        for i in 0..e {
            offsets[i + 1] = offsets[i] + counts[i].div_ceil(16) * 16;
        }
        let mut sorted = vec![-1i32; pmax];
        let mut tiles = vec![-1i32; pmax / 16];
        let mut fill = offsets.clone();
        for (f, &x) in route.iter().enumerate() {
            sorted[fill[x as usize]] = f as i32;
            fill[x as usize] += 1;
        }
        for i in 0..e {
            for t in offsets[i] / 16..offsets[i + 1] / 16 {
                tiles[t] = i as i32;
            }
        }
        let row_len = match kind {
            Kind::GateUp => m / 2,
            Kind::Down => m,
        };
        // The kernel indexes Y with a 32-bit byte offset (`gemm.rs:1514 pm_check_moe_down_extent`).
        if (pmax as u64) * (row_len as u64) * 2 > u64::from(u32::MAX) {
            return Err(format!(
                "{} grouped rows x {row_len} x 2 B exceed the kernel's 32-bit Y byte offset",
                pmax
            ));
        }
        let stride = m * row_bytes;
        let p = Problem {
            geo: geo.clone(),
            kind,
            m,
            k,
            epochs: pg.epochs as usize,
            x_row_div,
            x_src_rows,
            row_len,
            group_bytes: kind.group_bytes() as usize,
            row_bytes,
            stride,
            pmax,
            sorted,
            tiles,
            xq: OnceLock::new(),
            tame: false,
        };
        let need = p.device_bytes();
        if need > DEVICE_CAP {
            return Err(format!(
                "shape needs {need} B of device memory (> {DEVICE_CAP} B cap): lower T, E or D"
            ));
        }
        Ok(p)
    }

    fn ntiles(&self) -> usize {
        self.pmax / 16
    }

    fn y_bytes(&self) -> usize {
        self.pmax * self.row_len * 2
    }

    fn xq_bytes(&self) -> usize {
        self.epochs * self.x_src_rows * XBLK
    }

    fn device_bytes(&self) -> u64 {
        self.device_bytes_with(false)
    }

    fn device_bytes_with(&self, gp: bool) -> u64 {
        (self.geo.distinct * self.stride
            + if gp { self.xq_gp_bytes() } else { 0 }
            + 2 * self.y_bytes()
            + self.xq_bytes()
            + self.geo.experts * 8
            + self.pmax * 4
            + self.ntiles() * 4) as u64
    }

    /// Grouped rows holding a real slot, ascending.
    fn live_rows(&self) -> Vec<usize> {
        (0..self.pmax)
            .filter(|&p| self.tiles[p / 16] >= 0 && self.sorted[p] >= 0)
            .collect()
    }

    fn total_slots(&self) -> usize {
        self.x_src_rows * self.x_row_div
    }

    fn fill_weights(&self, idx: usize, out: &mut [u8]) {
        let seed = self.geo.seed
            ^ (idx as u64).wrapping_mul(GOLDEN)
            ^ (self.group_bytes as u64).wrapping_mul(0x1_0000_0001);
        symmetric_weights_into(out, self.group_bytes, seed, if self.tame { 4 } else { 9 });
    }

    /// Epoch-major `[epochs][x_src_rows]` `block_i4_128` records: `d` a normal
    /// f32 in `[2^-3, 2)`, 128 uniform signed nibbles, `s` their sum.
    fn make_xq(&self) -> Vec<u8> {
        let mut rng = Rng(self.geo.seed ^ 0xA4A4_0000_0000_0000 ^ self.x_src_rows as u64);
        let mut out = vec![0u8; self.xq_bytes()];
        for rec in out.chunks_exact_mut(XBLK) {
            let d = f32::from_bits(
                ((124 + rng.below(4) as u32) << 23) | (rng.next() as u32 & 0x7f_ffff),
            );
            let mut s = 0i32;
            for c in rec[8..].chunks_exact_mut(8) {
                c.copy_from_slice(&rng.next().to_le_bytes());
            }
            for &b in &rec[8..] {
                s += i32::from(((b << 4) as i8) >> 4) + i32::from((b as i8) >> 4);
            }
            rec[0..4].copy_from_slice(&d.to_le_bytes());
            rec[4..8].copy_from_slice(&s.to_le_bytes());
        }
        out
    }

    fn xq(&self) -> &[u8] {
        self.xq.get_or_init(|| self.make_xq())
    }

    /// Bytes of the grouped-position plane `[epochs][m_total]` (down only).
    fn xq_gp_bytes(&self) -> usize {
        self.epochs * self.pmax * XBLK
    }

    /// The grouped-position plane of the down A4 records: record `g * m_total + p` is the
    /// flat record `g * x_src_rows + sorted[p] / x_row_div` for every live grouped row `p`;
    /// padding and sentinel rows stay zero (the `gp` entries never read them: a padding
    /// lane takes the out-of-range offset). Bytes equal what `moe_rotate_i4_gp_pm` writes
    /// for the same activations (`moe_rotate_i4_pm::emit_gp`: same record, other index).
    fn make_xq_gp(&self) -> Vec<u8> {
        let xq = self.xq();
        let mut out = vec![0u8; self.xq_gp_bytes()];
        out.par_chunks_mut(self.pmax * XBLK)
            .enumerate()
            .for_each(|(h, plane)| {
                for p in 0..self.pmax {
                    if self.tiles[p / 16] < 0 || self.sorted[p] < 0 {
                        continue;
                    }
                    let r = self.sorted[p] as usize / self.x_row_div;
                    let src = (h * self.x_src_rows + r) * XBLK;
                    plane[p * XBLK..(p + 1) * XBLK].copy_from_slice(&xq[src..src + XBLK]);
                }
            });
        out
    }

    fn x_row(&self, r: usize) -> XRow {
        let xq = self.xq();
        let mut d = Vec::with_capacity(self.epochs);
        let mut v = Vec::with_capacity(self.epochs * 128);
        for h in 0..self.epochs {
            let o = (h * self.x_src_rows + r) * XBLK;
            d.push(f32::from_le_bytes(
                xq[o..o + 4].try_into().expect("4 bytes"),
            ));
            for j in 0..128 {
                let b = xq[o + 8 + j / 2];
                let nib = if j % 2 == 0 { b } else { b >> 4 };
                v.push(((nib << 4) as i8) >> 4);
            }
        }
        XRow { d, v }
    }

    /// BF16 bits of one output row from its `[m]` sums.
    fn epilogue(&self, sums: &[f32]) -> Vec<u16> {
        match self.kind {
            Kind::GateUp => {
                let half = self.m / 2;
                (0..half)
                    .map(|c| bf16_bits(portable_silu_mul(rt_f32(sums[c]), rt_f32(sums[half + c]))))
                    .collect()
            }
            Kind::Down => sums.iter().map(|&s| bf16_bits(s)).collect(),
        }
    }

    /// Reference output rows `(grouped row, BF16 bits)`, ascending.
    fn reference(&self, sample: Sample) -> Vec<(usize, Vec<u16>)> {
        let live = self.live_rows();
        let picks = match sample {
            Sample::Full => live,
            Sample::Rows(n) => pick_evenly(&live, n),
        };
        let mut by_expert: std::collections::BTreeMap<i32, Vec<usize>> = Default::default();
        for p in picks {
            by_expert.entry(self.tiles[p / 16]).or_default().push(p);
        }
        let groups: Vec<(i32, Vec<usize>)> = by_expert.into_iter().collect();
        let mut rows: Vec<(usize, Vec<u16>)> = groups
            .into_par_iter()
            .flat_map_iter(|(e, ps)| {
                let mut w = vec![0u8; self.stride];
                self.fill_weights(e as usize % self.geo.distinct, &mut w);
                let xs: Vec<XRow> = ps
                    .iter()
                    .map(|&p| self.x_row(self.sorted[p] as usize / self.x_row_div))
                    .collect();
                let sums = fold_slots(self.kind, self.m, self.epochs, self.row_bytes, &w, &xs);
                ps.iter()
                    .enumerate()
                    .map(|(s, &p)| (p, self.epilogue(&sums[s * self.m..(s + 1) * self.m])))
                    .collect::<Vec<_>>()
            })
            .collect();
        rows.sort_by_key(|r| r.0);
        rows
    }

    /// Compare a downloaded `[pmax][row_len]` BF16 output with the reference rows.
    fn compare(
        &self,
        out: &[u8],
        reference: &[(usize, Vec<u16>)],
        sample: Sample,
    ) -> Result<RefReport> {
        if out.len() != self.y_bytes() {
            return Err(format!(
                "output is {} B, expected {} ({} grouped rows x {} BF16)",
                out.len(),
                self.y_bytes(),
                self.pmax,
                self.row_len
            ));
        }
        let down = self.kind == Kind::Down;
        let allow: i32 = if down { 0 } else { 1 };
        let rl = self.row_len;
        let acc = (0..self.pmax)
            .into_par_iter()
            .map(|p| {
                let row = &out[p * rl * 2..(p + 1) * rl * 2];
                let bits = |c: usize| u16::from_le_bytes([row[2 * c], row[2 * c + 1]]);
                let mut a = Acc::default();
                let expect_zero = |a: &mut Acc| -> usize {
                    let mut bad = 0;
                    for c in 0..rl {
                        let b = bits(c);
                        if b != 0 {
                            bad += 1;
                            let e = if bf16_nonfinite(b) {
                                f64::MAX
                            } else {
                                bf16_to_f64(b).abs()
                            };
                            a.err = a.err.max(e);
                        }
                    }
                    bad
                };
                if self.tiles[p / 16] < 0 {
                    a.sentinel_rows = 1;
                    a.sentinel_bad = usize::from(expect_zero(&mut a) > 0);
                    a.bad += a.sentinel_bad;
                } else if self.sorted[p] < 0 {
                    a.pad_rows = 1;
                    a.pad_bad = usize::from(expect_zero(&mut a) > 0);
                    a.bad += a.pad_bad;
                } else if let Ok(i) = reference.binary_search_by_key(&p, |r| r.0) {
                    a.compared_rows = 1;
                    for (c, &want) in reference[i].1.iter().enumerate() {
                        let got = bits(c);
                        if bf16_nonfinite(got) || bf16_nonfinite(want) {
                            if !(got == want || (bf16_nan(got) && bf16_nan(want))) {
                                a.bad += 1;
                                a.err = f64::MAX;
                            }
                            continue;
                        }
                        let dist = (bf16_ord(got) - bf16_ord(want)).abs();
                        let diff = (bf16_to_f64(got) - bf16_to_f64(want)).abs();
                        a.err = a.err.max(diff);
                        if !down {
                            a.tol = a.tol.max(bf16_ulp(want));
                        }
                        if dist > allow {
                            a.bad += 1;
                        }
                    }
                }
                a
            })
            .reduce(Acc::default, Acc::merge);
        let live = self.live_rows().len();
        let name = match self.kind {
            Kind::GateUp => {
                "cpu_qt44_a4_fold_f32 + portable_silu_bf16_rne (fold bit-exact; SwiGLU is a \
                 portable-math reference, not a device byte oracle; tolerance 1 BF16 ulp of the \
                 reference per element, `tolerance` = largest such ulp)"
            }
            Kind::Down => "cpu_qt53_a4_fold_f32 + bf16_rne (bit-exact, tolerance 0)",
        };
        Ok(RefReport {
            reference: format!(
                "{name}; ref-rows {}: {} of {live} live slot rows compared x {rl} cols \
                 ({} reference rows); padding rows zero-checked {} (bad {}), sentinel-tile rows \
                 zero-checked {} (bad {}), mismatching elements/rows {}",
                sample.label(),
                acc.compared_rows,
                reference.len(),
                acc.pad_rows,
                acc.pad_bad,
                acc.sentinel_rows,
                acc.sentinel_bad,
                acc.bad
            ),
            max_abs_err: acc.err,
            tolerance: acc.tol,
            pass: acc.bad == 0 && acc.compared_rows == reference.len(),
        })
    }

    /// Launch geometry `(grid, block, dynamic LDS)` of `spec`: grid `(M / (64 rr), tiles, 1)`,
    /// block `Spec::threads`, no dynamic LDS.
    fn geometry(&self, spec: Spec) -> ([u32; 3], [u32; 3], u32) {
        (
            [
                (self.m / (64 * usize::from(spec.rr.max(1)))) as u32,
                self.ntiles() as u32,
                1,
            ],
            [spec.threads(), 1, 1],
            0,
        )
    }

    /// The 64-byte kernarg image of `entry`'s launch, encoded by the `hipfire-kernel-abi`
    /// contract (`families::qwen4_moe_sym`): five device pointers, then
    /// `M, K, x_row_div, m_total, x_src_rows, n_experts` as i32.
    ///
    /// `ptrs` is `(address, allocation bytes)` of `expert_weight_ptrs, expert_tile_ids,
    /// sorted_slot_index, Xq, Y_grouped`. An entry whose symbol is a contract export is
    /// validated with its own geometry; any other (another NT or row repeat) encodes through
    /// the NT4 product export of the same kind, and its generator layout must agree with the
    /// contract on every argument and the segment.
    fn kernarg(&self, entry: Entry, ptrs: [(u64, u64); 5]) -> Result<Vec<u8>> {
        self.kernarg_with_layout(entry, entry.spec.kernargs(), ptrs)
    }

    fn kernarg_with_layout(
        &self,
        entry: Entry,
        layout: peacemaker_isa::KernargLayout,
        ptrs: [(u64, u64); 5],
    ) -> Result<Vec<u8>> {
        let spec = entry.spec;
        let abi_arch = abi_arch(spec.arch)?;
        let exact = if spec.codec == WeightCodec::Mq4Sym {
            find_export(abi_arch, &entry.symbol())
        } else {
            None
        };
        let (abi, export) = match exact {
            Some(found) => found,
            None => {
                let product = Entry {
                    spec: Spec {
                        arch: spec.arch,
                        kind: spec.kind,
                        nt: 4,
                        rr: 1,
                        codec: WeightCodec::Mq4Sym,
                    },
                    gp: entry.gp,
                };
                find_export(abi_arch, &product.symbol()).ok_or_else(|| {
                    format!("no kernel-abi contract export for {}", product.symbol())
                })?
            }
        };
        let evidence = ExportEvidence {
            source: EvidenceSource::Generator,
            arch: Some(abi_arch),
            symbol: export.symbol.to_owned(),
            args: layout
                .args
                .iter()
                .map(|a| ArgEvidence {
                    name: a.name.clone(),
                    offset: a.offset,
                    size: a.size,
                    value_kind: a.value_kind.clone(),
                    access: a.actual_access.clone(),
                })
                .collect(),
            segment_bytes: Some(layout.size),
            segment_align: None,
            max_flat_workgroup_size: None,
            threads: None,
            dynamic_lds: None,
            group_segment_fixed: None,
            wave: None,
            object_sha256: None,
        };
        let drift: Vec<String> = check_export(abi, export, &evidence)
            .into_iter()
            .filter(|d| {
                matches!(
                    d,
                    Drift::MissingArg { .. }
                        | Drift::ExtraArg { .. }
                        | Drift::ArgName { .. }
                        | Drift::ArgWidth { .. }
                        | Drift::ArgKind { .. }
                        | Drift::ArgAccess { .. }
                        | Drift::Segment { .. }
                        | Drift::SegmentAlign { .. }
                )
            })
            .map(|d| d.to_string())
            .collect();
        if !drift.is_empty() {
            return Err(format!(
                "{}: kernarg layout differs from contract {} ({}): {}",
                entry.variant(),
                abi.id,
                export.symbol,
                drift.join("; ")
            ));
        }
        let formats = match (spec.kind, entry.gp) {
            (Kind::GateUp, _) => [
                moe_abi::FORMAT_EXPERT_PTRS_GATE_UP_QT44,
                moe_abi::FORMAT_EXPERT_TILE_IDS,
                moe_abi::FORMAT_SORTED_SLOT_INDEX,
                moe_abi::FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
                moe_abi::FORMAT_Y_GATE_UP_SWIGLU_BF16,
            ],
            (Kind::Down, gp) => [
                moe_abi::FORMAT_EXPERT_PTRS_DOWN_QT53,
                moe_abi::FORMAT_EXPERT_TILE_IDS,
                moe_abi::FORMAT_SORTED_SLOT_INDEX,
                if gp {
                    moe_abi::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION
                } else {
                    moe_abi::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS
                },
                moe_abi::FORMAT_Y_DOWN_BF16,
            ],
        };
        let binds: [OperandBinding; 5] = std::array::from_fn(|i| OperandBinding {
            address: ptrs[i].0,
            bytes: Some(ptrs[i].1),
            format: formats[i],
        });
        let params = [
            self.m as u64,
            self.k as u64,
            self.x_row_div as u64,
            self.pmax as u64,
            self.x_src_rows as u64,
            self.geo.experts as u64,
        ];
        let (grid, block, dynamic_lds) = if exact.is_some() {
            self.geometry(spec)
        } else {
            let mut grid = [0u32; 3];
            for (g, e) in grid.iter_mut().zip(&export.grid) {
                let v = eval(e, abi, export, &params).map_err(|e| format!("{}: {e}", abi.id))?;
                *g = u32::try_from(v).map_err(|_| format!("{}: grid {v} exceeds u32", abi.id))?;
            }
            (grid, export.block, export.dynamic_lds)
        };
        let v = validate_invocation(
            abi,
            &InvocationV1 {
                arch: abi_arch,
                symbol: export.symbol,
                params: &params,
                operands: &binds,
                grid,
                block,
                dynamic_lds,
            },
        )
        .map_err(|e| format!("{}: {e}", entry.variant()))?;
        let image = v.explicit().to_vec();
        if image.len() != KERNARG_BYTES as usize {
            return Err(format!(
                "{}: contract image is {} B, the generator kernarg is {KERNARG_BYTES}",
                abi.id,
                image.len()
            ));
        }
        Ok(image)
    }

    fn launch_for(&self, entry: Entry, ptrs: [(u64, u64); 5]) -> Result<Launch> {
        let (grid, block, lds) = self.geometry(entry.spec);
        Ok(Launch {
            kernarg: self.kernarg(entry, ptrs)?,
            grid,
            block,
            lds,
        })
    }

    fn i32_bytes(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn shape_json(&self) -> Value {
        let g = &self.geo;
        let mut v = json!({
            "preset": g.preset, "experts": g.experts, "top_k": g.top, "tokens": g.tokens,
            "hidden": g.hidden, "intermediate": g.inter, "distinct_experts": g.distinct,
            "route": g.route.name(), "seed": g.seed,
            "kind": self.kind.tag(), "M": self.m, "K": self.k, "epochs": self.epochs,
            "x_row_div": self.x_row_div, "x_src_rows": self.x_src_rows,
            "grouped_rows": self.pmax, "tiles": self.ntiles(),
            "live_slots": g.slots(), "total_slots": self.total_slots(),
            "expert_tensor_bytes": self.stride, "y_bytes_per_arm": self.y_bytes(),
            "device_bytes": self.device_bytes(),
        });
        if let Some(h) = &g.hist {
            v["route_id"] = json!(h.id);
            v["route_layer"] = json!(h.layer);
        }
        v
    }
}

// ---------------------------------------------------------------------------
// Workload

/// Production's gfx1201 resident symbol width for one call: down is always NT4;
/// gate/up is NT8 when `total_slots >= 128 * n_experts`, else NT4
/// (`gemm.rs:49334 qwen4_moe_sym_nt8_selected`, `:49313 qwen4_moe_sym_wide_shape`).
fn shipped_nt(kind: Kind, total_slots: usize, experts: usize) -> u8 {
    let wide = kind == Kind::GateUp
        && experts != 0
        && experts
            .checked_mul(128)
            .is_some_and(|floor| total_slots >= floor);
    if wide {
        8
    } else {
        4
    }
}

/// The gfx1151 streamed NT8 gate/up (`qwen4_moe_sym::emit_gate_up_stream8`) selection: production's
/// `qwen4_moe_stream8_selected` (`gemm.rs:56516`; caller `qwen4_moe_sym_gate_up_symbol`,
/// `gemm.rs:50779`) takes it on a resident gfx1151 call exactly at `[M, K, top, rows, experts] =
/// [1280, 2560, 10, 7168..=8192, 512]` (gate/up only; `rows` = token rows = `x_src_rows`).
fn stream8_selected(p: &Problem) -> bool {
    p.kind == Kind::GateUp
        && p.m == 1280
        && p.k == 2560
        && p.x_row_div == 10
        && p.geo.experts == 512
        && (7168..=8192).contains(&p.x_src_rows)
}

/// What production (`HIPFIRE_*` unset, VRAM-resident experts) launches on `arch` for this call:
/// gfx1201 `shipped_nt`; gfx1151 gate/up NT4 (stream8 at its measured shape), down NT4 with row
/// repeat 1; gfx1100 NT1 for both. An entry the arch has no shipped object for is refused by name.
fn shipped_entry(arch: Arch, gp: bool, p: &Problem) -> Result<Entry> {
    let kind = p.kind;
    let plain = |nt: u8| Entry {
        spec: Spec {
            arch,
            kind,
            nt,
            rr: 1,
            codec: WeightCodec::Mq4Sym,
        },
        gp,
    };
    match arch {
        Arch::Gfx1201 => Ok(plain(shipped_nt(kind, p.total_slots(), p.geo.experts))),
        _ if gp => Err(format!(
            "no shipped grouped-position down entry on {}: `qwen4_moe_down_gp_pm_gfx1201` \
             (QWEN4_MOE_DOWN_GP_PM_GFX1201) is the only gp object",
            arch.name()
        )),
        Arch::Gfx1151 if stream8_selected(p) => Ok(Entry::stream8()),
        Arch::Gfx1151 => Ok(plain(4)),
        Arch::Gfx1100 => Ok(plain(1)),
        other => Err(format!(
            "no shipped qwen4_moe_sym object on {}",
            other.name()
        )),
    }
}

#[derive(Clone, Copy, Debug)]
enum Baseline {
    /// The embedded `qwen4_moe_iu4_sym_pm_gfx1201` bundle, this entry.
    Shipped(Entry),
    /// Emitted from `--base-epi`.
    Emitted(Entry),
}

impl Baseline {
    fn entry(self) -> Entry {
        match self {
            Baseline::Shipped(e) | Baseline::Emitted(e) => e,
        }
    }
}

struct Dev {
    _weights: DeviceBuffer,
    ptrs: DeviceBuffer,
    tiles: DeviceBuffer,
    sorted: DeviceBuffer,
    xq: DeviceBuffer,
    /// The grouped-position down plane (`[epochs][m_total]` records), when an arm is `gp`.
    xq_gp: Option<DeviceBuffer>,
    /// Per-arm output: `[Baseline, Candidate]`.
    y: [DeviceBuffer; 2],
}

struct Moe {
    problem: Problem,
    sample: Sample,
    cand: Entry,
    base: Baseline,
    dev: Option<Dev>,
    reference: OnceLock<Vec<(usize, Vec<u16>)>>,
    /// `--rip`: source path and text of the candidate script.
    rip: Option<(String, String)>,
    /// `--cand-object`: path and bytes of the prebuilt candidate code object (gfx1100 only).
    cand_object: Option<(String, Vec<u8>)>,
}

fn arm_index(arm: Arm) -> usize {
    match arm {
        Arm::Baseline => 0,
        Arm::Candidate => 1,
    }
}

/// One kernel entry: a `qwen4_moe_sym` spec, or (`gp`) the grouped-position twin of
/// the gfx1201 MQ4 down NT4/NT8 entry (`qwen4_moe_sym::emit_down_gp`), which reads the down A4
/// plane in grouped-position order (record `g * m_total + row`) instead of gathering
/// `sorted[row]`-indexed flat-slot records. Same 64-byte ABI, grid and output bytes.
///
/// The gfx1151 streamed NT8 gate/up (`emit_gate_up_stream8`, symbol `..._gfx1151_nt8_stream2`) has
/// no `Spec` of its own; it is the gfx1151 gate/up `nt = 8` spec, which `Spec::validate` refuses
/// for every other purpose, so [`Entry::is_stream8`] identifies it unambiguously.
#[derive(Clone, Copy, Debug)]
struct Entry {
    spec: Spec,
    gp: bool,
}

impl Entry {
    /// `--epi` grammar: the `Spec::from_epi` names plus `down_gp_nt4|down_gp_nt8` (gfx1201) and
    /// `gate_up_stream8` (gfx1151). gfx1100 builds only the NT1 entries; gfx1151 validates the
    /// spec at parse time (`Spec::validate` verbatim: NT8 and row-repeat rules).
    fn parse(epi: &str, arch: Arch) -> Result<Entry> {
        if epi == "gate_up_stream8" {
            if arch != Arch::Gfx1151 {
                return Err(format!(
                    "--epi {epi}: the streamed NT8 gate/up (`emit_gate_up_stream8`, object \
                     QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151) is built for gfx1151 only, not {}",
                    arch.name()
                ));
            }
            return Ok(Entry::stream8());
        }
        if let Some(tail) = epi.strip_prefix("down_gp") {
            if !tail.starts_with("_nt") {
                return Err(format!(
                    "--epi {epi}: the grouped-position twin needs a tile width (down_gp_nt4|down_gp_nt8)"
                ));
            }
            if arch != Arch::Gfx1201 {
                return Err(format!(
                    "--epi {epi}: the grouped-position down twin (`emit_down_gp`, bundle \
                     qwen4_moe_down_gp_pm_gfx1201) exists on gfx1201 only, not {}",
                    arch.name()
                ));
            }
            return Ok(Entry {
                spec: Spec::from_epi(&format!("down{tail}"), arch)?,
                gp: true,
            });
        }
        if epi.contains("_gp") {
            return Err(format!(
                "--epi {epi}: only the down entry has a grouped-position twin (down_gp_nt4|down_gp_nt8)"
            ));
        }
        let spec = Spec::from_epi(epi, arch)?;
        match arch {
            Arch::Gfx1201 => {}
            Arch::Gfx1151 => spec.validate()?,
            Arch::Gfx1100 if spec.nt != 1 || spec.rr != 1 => {
                return Err(format!(
                    "--epi {epi}: gfx1100 builds only the NT1 entries (gate_up, down; \
                     `qwen4_moe_sym_gfx11`), not nt{} rr{}",
                    spec.nt, spec.rr
                ));
            }
            Arch::Gfx1100 => {}
            other => {
                return Err(format!(
                    "--epi {epi}: no qwen4_moe_sym entry on {}",
                    other.name()
                ));
            }
        }
        Ok(Entry { spec, gp: false })
    }

    /// The gfx1151 streamed NT8 gate/up entry.
    fn stream8() -> Entry {
        Entry {
            spec: Spec {
                arch: Arch::Gfx1151,
                kind: Kind::GateUp,
                nt: 8,
                rr: 1,
                codec: WeightCodec::Mq4Sym,
            },
            gp: false,
        }
    }

    fn is_stream8(self) -> bool {
        !self.gp
            && self.spec.arch == Arch::Gfx1151
            && self.spec.kind == Kind::GateUp
            && self.spec.nt == 8
    }

    fn symbol(self) -> String {
        if self.gp {
            self.spec.symbol().replacen("_sym_pm_", "_sym_gp_pm_", 1)
        } else if self.is_stream8() {
            format!("{}_stream2", self.spec.symbol())
        } else {
            self.spec.symbol()
        }
    }

    fn variant(self) -> String {
        if self.gp {
            self.spec.variant().replacen("down_nt", "down_gp_nt", 1)
        } else if self.is_stream8() {
            "gate_up_stream8".to_owned()
        } else {
            self.spec.variant()
        }
    }

    /// The generator of this entry: `emit_down_gp`, `emit_gate_up_stream8`, the gfx11 K2 module
    /// for the gfx1100 NT1 entries (`Spec::validate` refuses gfx1100), else `qwen4_moe_sym::emit`.
    fn emit(self) -> Result<peacemaker_isa::Emitted> {
        if self.gp {
            qwen4_moe_sym::emit_down_gp(self.spec)
        } else if self.is_stream8() {
            qwen4_moe_sym::emit_gate_up_stream8(self.spec.arch)
        } else if self.spec.arch == Arch::Gfx1100 {
            gfx11::emit(gfx11::Spec {
                arch: self.spec.arch,
                codec: gfx11::Codec::Mq4,
                kind: self.spec.kind,
                acc: false,
                negative: gfx11::Negative::None,
            })
        } else {
            qwen4_moe_sym::emit(self.spec)
        }
    }
}

fn spec_json(e: Entry) -> Value {
    let spec = e.spec;
    json!({
        "epi": e.variant(), "kind": spec.kind.tag(), "nt": spec.nt, "rr": spec.rr,
        "gp": e.gp, "codec": format!("{:?}", spec.codec), "arch": spec.arch.name(),
        "symbol": e.symbol(), "threads": spec.threads(), "lds_bytes": spec.lds_bytes(),
    })
}

/// Emit and link exactly like `peacemaker-isa emit --kernel qwen4_moe_sym --co`
/// (`peacemaker-isa.rs:202-216` and the `native::assemble` block at `:1547`).
fn emit_object(entry: Entry, source: Value) -> Result<Object> {
    let e = entry.emit()?;
    let bytes = peacemaker_isa::native::assemble(&e.s_text, entry.spec.arch)?;
    let symbol = entry.symbol();
    Ok(Object {
        bytes,
        launch_symbol: symbol.clone(),
        exports: vec![symbol],
        desc: json!({"knobs": spec_json(entry)}),
        source,
    })
}

/// Provenance of a non-gfx1201 entry's generator (gfx1201 records keep their original `source`
/// text; this adds `generator` only elsewhere).
fn with_generator(entry: Entry, mut source: Value) -> Value {
    let generator = if entry.gp || entry.spec.arch == Arch::Gfx1201 {
        return source;
    } else if entry.is_stream8() {
        "peacemaker_isa::kernels::qwen4_moe_sym::emit_gate_up_stream8 + native::assemble"
    } else if entry.spec.arch == Arch::Gfx1100 {
        "peacemaker_isa::kernels::qwen4_moe_sym_gfx11::emit (Codec::Mq4; Spec::validate of \
         qwen4_moe_sym refuses gfx1100) + native::assemble"
    } else {
        "peacemaker_isa::kernels::qwen4_moe_sym::emit (Spec::from_epi(epi, gfx1151)) + native::assemble"
    };
    source["generator"] = json!(generator);
    source
}

/// Flash-Next prefill presets: `fn:<entry>-t<tokens>` = the Flash-Next geometry
/// (`E=512,TOP=10,H=2560,I=640`) at `T` prefill tokens with that entry as the default
/// `--epi` (an explicit `--epi` wins). `(name, epi, tokens)`.
const FN_PRESETS: [(&str, &str, usize); 4] = [
    ("gate_up_nt4-t4096", "gate_up_nt4", 4096),
    ("gate_up_nt4-t16384", "gate_up_nt4", 16384),
    ("down_gp_nt4-t4096", "down_gp_nt4", 4096),
    ("down_gp_nt4-t16384", "down_gp_nt4", 16384),
];

/// Rewrite `fn:<name>` shape items to `fn,T=<tokens>`; returns the preset's default epi.
fn expand_presets(shape: &str) -> Result<(String, Option<&'static str>)> {
    let mut epi = None;
    let mut items = Vec::new();
    for item in shape.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(name) = item.strip_prefix("fn:") {
            let &(_, e, t) = FN_PRESETS.iter().find(|p| p.0 == name).ok_or_else(|| {
                format!(
                    "unknown preset fn:{name} (one of {})",
                    FN_PRESETS
                        .iter()
                        .map(|p| format!("fn:{}", p.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
            if epi.replace(e).is_some() {
                return Err("--shape: more than one fn:<name> preset".into());
            }
            items.push("fn".to_owned());
            items.push(format!("T={t}"));
        } else {
            items.push(item.to_owned());
        }
    }
    Ok((items.join(","), epi))
}

/// Compile `.rip` source for `entry` (variant = `Entry::variant()`, e.g. `down_gp_nt4`), refuse
/// unless its ABI equals that of the Rust emit of the same entry, and link it with the native
/// writer. The record's `candidate.rip` = `{path, sha256, variant, identical_to_generator}`.
fn rip_object(entry: Entry, path: &str, source: &str) -> Result<Object> {
    let variant = entry.variant();
    let compiled = peacemaker_rip::compile(source, entry.spec.arch, &variant)?;
    let generator = entry.emit()?;
    let symbol = entry.symbol();
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, entry.spec.arch)?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, entry.spec.arch)?;
    let got = crate::iu4::abi_signature(&elf, &symbol)
        .map_err(|e| format!("compiled kernel does not expose {symbol}: {e}"))?;
    let want = crate::iu4::abi_signature(&gen_elf, &symbol)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with the Rust emit of {variant}:\n  .rip:      {got}\n  generator: {want}"
        ));
    }
    Ok(Object {
        source: json!({
            "source": format!("peacemaker_rip::compile({path}) + native::assemble"),
            "variant": variant,
            "sha256": sha256_hex(&elf),
        }),
        desc: json!({
            "knobs": spec_json(entry),
            "rip": {
                "path": path,
                "sha256": sha256_hex(source.as_bytes()),
                "variant": variant,
                "identical_to_generator": compiled.s_text == generator.s_text,
            },
        }),
        exports: vec![symbol.clone()],
        launch_symbol: symbol,
        bytes: elf,
    })
}

/// A prebuilt candidate code object for `entry` (`--cand-object`; gfx1100 NT1, built by the
/// standalone emitter of a candidate module copy): refused unless it exposes `entry`'s symbol with
/// the ABI (kernarg layout, workgroup size, LDS, wave mode) of the in-tree Rust emit of that
/// entry, so the launch contract, kernarg encoding and grid of the baseline arm stay valid. M7,
/// resources and `code_sha256` then run on it exactly as on an emitted candidate.
fn cand_object_object(entry: Entry, path: &str, bytes: &[u8]) -> Result<Object> {
    let symbol = entry.symbol();
    let generator = entry.emit()?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, entry.spec.arch)?;
    let got = crate::iu4::abi_signature(bytes, &symbol)
        .map_err(|e| format!("--cand-object {path}: not a code object exposing {symbol}: {e}"))?;
    // gfx1201: the reference is the shipped image's own entry (the object the baseline launches),
    // when that image exports the symbol; otherwise (an NT the shipped image lacks) the emit's.
    let (want, want_what) = match shipped_reference(entry) {
        Some(image) if crate::iu4::abi_signature(image, &symbol).is_ok() => (
            crate::iu4::abi_signature(image, &symbol)?,
            "shipped baseline image",
        ),
        _ => (
            crate::iu4::abi_signature(&gen_elf, &symbol)?,
            "in-tree emit",
        ),
    };
    if got != want {
        return Err(format!(
            "--cand-object {path}: ABI mismatch with the {want_what} of {} in field {}:\n  object:    {got}\n  reference: {want}",
            entry.variant(),
            abi_diff_field(&got, &want),
        ));
    }
    let text_sha256 = kernel_text_sha256(bytes, &symbol)?;
    Ok(Object {
        source: json!({
            "source": format!("--cand-object {path} (prebuilt candidate code object)"),
            "variant": entry.variant(),
            "sha256": sha256_hex(bytes),
            "symbol": symbol,
            "text_sha256": text_sha256,
        }),
        desc: json!({
            "knobs": spec_json(entry),
            "cand_object": {
                "path": path,
                "sha256": sha256_hex(bytes),
                "co_sha256": sha256_hex(bytes),
                "symbol": symbol,
                "text_sha256": text_sha256,
                "abi_reference": want_what,
                "variant": entry.variant(),
                "identical_to_generator": bytes == gen_elf.as_slice(),
            },
        }),
        exports: vec![symbol.clone()],
        launch_symbol: symbol,
        bytes: bytes.to_vec(),
    })
}

/// The shipped gfx1201 image holding `entry`'s symbol (None off gfx1201).
fn shipped_reference(entry: Entry) -> Option<&'static [u8]> {
    if entry.spec.arch != Arch::Gfx1201 {
        return None;
    }
    Some(if entry.gp {
        rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201
    } else {
        rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201
    })
}

/// sha256 of `symbol`'s re-encoded instruction stream (the `.text` of that kernel).
fn kernel_text_sha256(bytes: &[u8], symbol: &str) -> Result<String> {
    let program = crate::lift(bytes)?;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("object has no kernel {symbol}"))?;
    let code = peacemaker_lift::emit::bytes(kernel, program.target.arch).map_err(err)?;
    Ok(sha256_hex(&code))
}

/// Name of the first `abi_signature` field that differs between `a` and `b`.
fn abi_diff_field(a: &str, b: &str) -> String {
    const KEYS: [&str; 8] = [
        "symbol",
        "args",
        "kernarg_bytes",
        "kernarg_align",
        "lds_bytes",
        "max_flat_workgroup_size",
        "wavefront_size",
        "workgroup_processor_mode",
    ];
    let field = |s: &str, i: usize| -> String {
        let start = s.find(&format!("{}=", KEYS[i])).unwrap_or(s.len());
        let end = KEYS
            .get(i + 1)
            .and_then(|k| s.find(&format!(" {k}=")))
            .unwrap_or(s.len());
        s[start..end.max(start)].to_string()
    };
    (0..KEYS.len())
        .find(|&i| field(a, i) != field(b, i))
        .map(|i| KEYS[i].to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Constructor for `--kernel qwen4_moe_sym` without `--rip` (gfx1201).
#[cfg(test)]
pub fn new(shape: &str, knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    new_with_rip(shape, knobs, None, Arch::Gfx1201)
}

/// Test constructor: `--rip` only, no `--cand-object`.
#[cfg(test)]
pub fn new_with_rip(
    shape: &str,
    knobs: &mut Knobs,
    rip: Option<&std::path::Path>,
    arch: Arch,
) -> Result<Box<dyn Workload>> {
    new_with_cand_object(shape, knobs, rip, arch, None)
}

/// Constructor for `--kernel qwen4_moe_sym`; `rip` = `--rip FILE.rip` (every MQ4 entry);
/// `cand_object` = `--cand-object FILE` (a prebuilt candidate code object, gfx1100 only: read and
/// ABI-checked against the in-tree emit of the entry by `candidate()`); `arch` =
/// gfx1201 (default), gfx1151 or gfx1100: the shipped baseline, candidate emit, R1 contract and
/// geometry are all that arch's (see the module doc's `# Shipped baseline`).
pub fn new_with_cand_object(
    shape: &str,
    knobs: &mut Knobs,
    rip: Option<&std::path::Path>,
    arch: Arch,
    cand_object: Option<&std::path::Path>,
) -> Result<Box<dyn Workload>> {
    let cand_object = match cand_object {
        Some(p) => {
            if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1201) {
                return Err(format!(
                    "--cand-object is supported on gfx1100 and gfx1201 only (got {})",
                    arch.name()
                ));
            }
            Some((
                p.display().to_string(),
                std::fs::read(p).map_err(|e| format!("--cand-object {}: {e}", p.display()))?,
            ))
        }
        None => None,
    };
    abi_arch(arch)?;
    let rip = match rip {
        Some(p) => Some((
            p.display().to_string(),
            std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        )),
        None => None,
    };
    let (shape, preset_epi) = expand_presets(shape)?;
    let epi = knobs.take_or("epi", preset_epi.unwrap_or("gate_up"));
    let base_epi = knobs.take("base-epi");
    let sample = Sample::parse(&knobs.take_or("ref-rows", &format!("sample:{DEFAULT_SAMPLE}")))?;
    let geo = Geo::parse(&shape, arch)?;
    let cand = Entry::parse(&epi, arch)?;
    let problem = Problem::new(&geo, cand.spec.kind)?;
    let base = match base_epi {
        Some(b) => {
            let entry = Entry::parse(&b, arch).map_err(|e| format!("--base-epi: {e}"))?;
            if entry.spec.kind != cand.spec.kind {
                return Err(format!(
                    "--base-epi {b}: kind {} differs from the candidate's {} (outputs are not comparable)",
                    entry.spec.kind.tag(),
                    cand.spec.kind.tag()
                ));
            }
            Baseline::Emitted(entry)
        }
        // A gp candidate times against the shipped gp entry (production runs the grouped
        // plane whenever the PM rotate twin is selected), a flat one against the flat entry.
        None => Baseline::Shipped(shipped_entry(arch, cand.gp, &problem)?),
    };
    let any_gp = cand.gp || base.entry().gp;
    if any_gp && problem.device_bytes_with(true) > DEVICE_CAP {
        return Err(format!(
            "shape needs {} B of device memory with the grouped plane (> {DEVICE_CAP} B cap): lower T, E or D",
            problem.device_bytes_with(true)
        ));
    }
    Ok(Box::new(Moe {
        problem,
        sample,
        cand,
        base,
        dev: None,
        reference: OnceLock::new(),
        rip,
        cand_object,
    }))
}

/// Unmapped bytes between the exact-size buffers of an emulator-gate run (the gate demands >= 64 KiB).
const EMU_GUARD: u64 = 0x2_0000;
/// Shapes of the emulator gate: an aligned and a ragged token count (grouped rows pad to 16 per expert) at a tiny K
/// (gate_up K=512 / 4 epochs, down K=128 / 1 epoch) and at the production K (H=2560, I=640: gate_up K=2560 / 20
/// epochs, down K=640 / 5 epochs), so K-specialised paths (e.g. an epoch-count guard) are emulated too. ALL must pass.
const EMU_GATE_SHAPES: [(&str, &str); 4] = [
    ("tiny", "E=2,TOP=2,T=20,H=512,I=128"),
    ("ragged", "E=2,TOP=2,T=37,H=512,I=128"),
    ("prod", "E=2,TOP=2,T=20,H=2560,I=640"),
    ("prod_ragged", "E=2,TOP=2,T=37,H=2560,I=640"),
];
/// Production-extent cases of the gate: the judge shape (Flash-Next geometry, T=16896) and T=4096 with the real
/// kernargs and exact-size guarded buffers (hundreds of MB to ~1 GB of address space, lazily backed), emulating
/// only sampled workgroups ([`Moe::extent_wgs`]); the sampled outputs are compared with the emulated incumbent's.
/// Catches address-arithmetic faults that only appear at production record indices.
const EMU_EXTENT_SHAPES: [(&str, &str); 2] = [
    ("extent_t16896", "fn,T=16896"),
    ("extent_t4096", "fn,T=4096"),
];

/// One emulated arm of the gate: output bytes and counters.
struct EmuArm {
    y: Vec<u8>,
    y_bytes_written: u64,
    instructions: u64,
    workgroups: u64,
    lds_static: Option<u64>,
}

impl Moe {
    fn dev(&self) -> Result<&Dev> {
        self.dev
            .as_ref()
            .ok_or_else(|| "qwen4_moe_sym workload used before setup".to_owned())
    }

    /// `arm`'s launch with its kernarg encoded by the contract over the live allocations.
    fn checked_launch(&self, arm: Arm) -> Result<Launch> {
        let dev = self.dev()?;
        let entry = match arm {
            Arm::Baseline => self.base.entry(),
            Arm::Candidate => self.cand,
        };
        let op = |b: &DeviceBuffer| (b.as_ptr() as u64, b.size() as u64);
        let xq = match (&dev.xq_gp, entry.gp) {
            (Some(gp), true) => gp,
            _ => &dev.xq,
        };
        self.problem.launch_for(
            entry,
            [
                op(&dev.ptrs),
                op(&dev.tiles),
                op(&dev.sorted),
                op(xq),
                op(&dev.y[arm_index(arm)]),
            ],
        )
    }

    fn baseline_object(&self) -> Result<Object> {
        let entry = self.base.entry();
        let spec = entry.spec;
        match self.base {
            Baseline::Emitted(_) => emit_object(
                entry,
                with_generator(
                    entry,
                    json!({
                        "source": "emitted from --base-epi: peacemaker_isa::kernels::qwen4_moe_sym::emit \
                                   + native::assemble (peacemaker-isa.rs:202-216, :1547)",
                    }),
                ),
            ),
            Baseline::Shipped(_) if spec.arch != Arch::Gfx1201 => self.shipped_gfx11(entry),
            Baseline::Shipped(_) if entry.gp => self.shipped_gp(entry),
            Baseline::Shipped(_) => {
                let bytes = rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201.to_vec();
                let symbol = spec.symbol();
                let mut needle = symbol.clone().into_bytes();
                needle.push(0);
                if !bytes.windows(needle.len()).any(|w| w == needle.as_slice()) {
                    return Err(format!(
                        "shipped bundle QWEN4_MOE_IU4_SYM_PM_GFX1201 does not export {symbol}"
                    ));
                }
                let exports: Vec<String> = qwen4_moe_sym::module_specs(spec.arch)
                    .into_iter()
                    .map(Spec::symbol)
                    .collect();
                let p = &self.problem;
                Ok(Object {
                    launch_symbol: symbol.clone(),
                    exports,
                    desc: json!({
                        "artifact": "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco",
                        "knobs": spec_json(entry),
                    }),
                    source: json!({
                        "source": "crates/rdna-compute/src/gemm.rs:49334 qwen4_moe_sym_nt8_selected \
                                   (resident width; wide shape :49313; tables :49277-49296), \
                                   :50101-50125 gfx1201 MoeSymIu4Kernel row, \
                                   :52361-52503 qwen4_moe_sym_gemm launcher",
                        "const": "rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201 \
                                  (crates/rdna-compute/src/kernels.rs:4467)",
                        "artifact": "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco",
                        "sha256": sha256_hex(&bytes),
                        "symbol": symbol,
                        "nt": spec.nt,
                        "selection": {
                            "residency": "vram (host-mapped experts would take NT8 for both GEMMs)",
                            "rule": "down: NT4 always; gate_up: NT8 iff total_slots >= 128 * n_experts \
                                     (HIPFIRE_MOE_IU4_WIDE_TILES unset = on at gfx1201, \
                                     prefill_route_capabilities.rs:303,327), else NT4",
                            "total_slots": p.total_slots(),
                            "n_experts": p.geo.experts,
                        },
                        "launch": "grid (M/64, m_total/16, 1), block = Spec::threads, dynamic LDS 0 \
                                   (gemm.rs:52472,52476)",
                        "outputs_comparable": "yes: every NT entry shares the 64-byte ABI, grid and \
                                   output bytes (qwen4_moe_sym.rs:1544-1554), so the shipped nt may \
                                   differ from the candidate's",
                    }),
                    bytes,
                })
            }
        }
    }

    /// The shipped gfx1151 / gfx1100 baseline (`entry` from [`shipped_entry`]): that arch's own
    /// embedded bundle (never a gfx1201 object), selected as production does at VRAM residency.
    fn shipped_gfx11(&self, entry: Entry) -> Result<Object> {
        use rdna_compute::kernels as k;
        let spec = entry.spec;
        let arch = spec.arch;
        let symbol = entry.symbol();
        let p = &self.problem;
        let nt1_exports = || -> Vec<String> {
            Kind::ALL
                .into_iter()
                .map(|kind| {
                    Spec {
                        arch,
                        kind,
                        nt: 1,
                        rr: 1,
                        codec: WeightCodec::Mq4Sym,
                    }
                    .symbol()
                })
                .collect()
        };
        let (bytes, konst, artifact, exports, source, rule) = match (arch, entry.is_stream8()) {
            (Arch::Gfx1151, true) => (
                k::QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151.to_vec(),
                "rdna_compute::kernels::QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151 \
                 (crates/rdna-compute/src/kernels.rs:4472)",
                "kernels/qwen4_moe_gate_up_stream8_pm_gfx1151.hxaco",
                vec![symbol.clone()],
                "crates/rdna-compute/src/gemm.rs:50779 qwen4_moe_sym_gate_up_symbol (stream8 \
                 allowlist), :56516 qwen4_moe_stream8_selected, :52458 launcher module \
                 QWEN4_MOE_SYM_STREAM8_MODULE (:49228-49229)",
                "gate_up: streamed NT8 iff gfx1151 && native PM (HIPFIRE_QWEN4_MOE_SYM_PM unset = on) \
                 && resident && [M,K,top,rows,experts] = [1280,2560,10,7168..=8192,512] \
                 (gemm.rs:56516)",
            ),
            (Arch::Gfx1151, false) => (
                k::QWEN4_MOE_IU4_SYM_PM_GFX1151.to_vec(),
                "rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151 \
                 (crates/rdna-compute/src/kernels.rs:4465)",
                "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
                qwen4_moe_sym::module_specs(arch)
                    .into_iter()
                    .map(Spec::symbol)
                    .collect(),
                "crates/rdna-compute/src/gemm.rs:50704 qwen4_moe_sym_gemm_symbols (resident pair; \
                 gfx1151 MoeSymIu4Kernel row :50072-50100; tables :49277-49296; NT8 selection is \
                 gfx1201-only, :49334-49350), :50802 qwen4_moe_sym_down_rr \
                 (HIPFIRE_QWEN4_MOE_SYM_DOWN_RR unset = 1; rr2/rr4 entries :50307, env :50344), \
                 :50779 qwen4_moe_sym_gate_up_symbol (stream8 allowlist :56516)",
                "down: NT4 rr1 (HIPFIRE_QWEN4_MOE_SYM_DOWN_RR unset; nt4r2/nt4r4 are env opt-ins: \
                 use --base-epi down_nt4r2|down_nt4r4); gate_up: NT4 (the only tile width) except \
                 the measured stream8 shape (--epi gate_up_stream8; \
                 [M,K,top,rows,experts] = [1280,2560,10,7168..=8192,512])",
            ),
            (Arch::Gfx1100, _) => (
                k::QWEN4_MOE_IU4_SYM_PM_GFX1100.to_vec(),
                "rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1100 \
                 (crates/rdna-compute/src/kernels.rs:4495)",
                "kernels/qwen4_moe_iu4_sym_pm_gfx1100.hxaco",
                nt1_exports(),
                "crates/rdna-compute/src/gemm.rs:50038-50071 gfx1100 MoeSymIu4Kernel row \
                 (tiles_per_cta [1,1], block [64,128] = QWEN4_MOE_SYM_NT1_BLOCK :49995, no LDS), \
                 :50711 NT1 early-out of qwen4_moe_sym_gemm_symbols, launcher \
                 qwen4_moe_sym_nt1_gemm :52067; the descriptor's default_requested is false \
                 (route default OFF on gfx1100 until its Flash-Next qualification)",
                "gate_up and down: NT1 for both residencies (the only gfx1100 entries)",
            ),
            (other, _) => {
                return Err(format!("no shipped qwen4_moe_sym object on {}", other.name()))
            }
        };
        let mut needle = symbol.clone().into_bytes();
        needle.push(0);
        if !bytes.windows(needle.len()).any(|w| w == needle.as_slice()) {
            return Err(format!(
                "shipped {artifact} (arch {}) does not export {symbol}",
                arch.name()
            ));
        }
        Ok(Object {
            launch_symbol: symbol.clone(),
            exports,
            desc: json!({"artifact": artifact, "knobs": spec_json(entry)}),
            source: json!({
                "source": source,
                "const": konst,
                "artifact": artifact,
                "sha256": sha256_hex(&bytes),
                "symbol": symbol,
                "nt": spec.nt,
                "selection": {
                    "residency": "vram",
                    "rule": rule,
                    "total_slots": p.total_slots(),
                    "n_experts": p.geo.experts,
                },
                "launch": "grid (M/(64*rr), m_total/16, 1), block = Spec::threads, dynamic LDS 0 \
                           (static LDS in the object: gate/up NT4 8192 B, stream8 16384 B, else 0)",
                "outputs_comparable": "yes: every entry shares the 64-byte ABI, grid and output \
                           bytes (same ascending K128 fold and BF16/SwiGLU rounding as the NT1 \
                           anchor; qwen4_moe_sym.rs module doc), so the shipped nt may differ \
                           from the candidate's",
            }),
            bytes,
        })
    }

    /// The shipped grouped-position down entry: `QWEN4_MOE_DOWN_GP_PM_GFX1201`, the NT4
    /// twin (resident experts; the host-mapped NT8 twin is not modelled).
    fn shipped_gp(&self, entry: Entry) -> Result<Object> {
        let bytes = rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201.to_vec();
        let symbol = entry.symbol();
        let mut needle = symbol.clone().into_bytes();
        needle.push(0);
        if !bytes.windows(needle.len()).any(|w| w == needle.as_slice()) {
            return Err(format!(
                "shipped bundle QWEN4_MOE_DOWN_GP_PM_GFX1201 does not export {symbol}"
            ));
        }
        let p = &self.problem;
        Ok(Object {
            launch_symbol: symbol.clone(),
            exports: qwen4_moe_down_gp::symbols(entry.spec.arch).to_vec(),
            desc: json!({
                "artifact": "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
                "knobs": spec_json(entry),
            }),
            source: json!({
                "source": "crates/rdna-compute/src/gemm.rs:51851-51864 gemm_qwen4_moe_down_iu4_sym \
                           (grouped plane -> `..._gp` twin), :49236-49261 entries/image \
                           (qwen4_moe_down_gp_entry, qwen4_moe_down_gp_image, QWEN4_MOE_DOWN_GP_PM_ENTRIES), \
                           :51865-51872 launcher; down width is NT4 resident (:49334 rule); \
                           plane writer crates/rdna-compute/src/moe_rotate_i4_pm.rs:219-249 \
                           (qwen4_moe_rotate128_i4_gp_selected / qwen4_moe_rotate128_i4_gp)",
                "const": "rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201 \
                          (crates/rdna-compute/src/kernels.rs:10783)",
                "artifact": "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
                "sha256": sha256_hex(&bytes),
                "symbol": symbol,
                "nt": entry.spec.nt,
                "selection": {
                    "residency": "vram (host-mapped experts take the NT8 twin)",
                    "rule": "down: NT4 always (resident); grouped plane whenever the PM rotate twin is selected (HIPFIRE_MOE_ROTATE_I4_PM) on exact gfx1201",
                    "total_slots": p.total_slots(),
                    "n_experts": p.geo.experts,
                },
                "plane": "record g * m_total + row (72 B); built host-side from the same flat A4 records \
                          the flat entry reads (byte-identical to moe_rotate_i4_gp_pm_gfx1201 output)",
                "route_plan": "crates/rdna-compute/src/route_plans/qwen3.8-flash-next.gfx1201.tsv walks AR decode \
                               only: it lists qwen4_moe_iu4_sym_pm_gfx1201 but not qwen4_moe_down_gp_pm_gfx1201",
            }),
            bytes,
        })
    }

    fn reference(&self) -> &[(usize, Vec<u16>)] {
        self.reference
            .get_or_init(|| self.problem.reference(self.sample))
    }

    /// Tiny-shape emulator run of the candidate; `Err` is a refusal reason.
    fn emu_run(&self) -> Result<Value> {
        use peacemaker_ir::emu::{Emulator, Launch as EmuLaunch, Memory};
        const KERNARG: u64 = 0x1_0000;
        const TABLE: u64 = 0x10_0000;
        const TILES: u64 = 0x20_0000;
        const SORTED: u64 = 0x30_0000;
        const EXPERTS: u64 = 0x1000_0000;
        const EXPERT_STRIDE: u64 = 0x100_0000;
        const XQ: u64 = 0x4000_0000;
        const Y: u64 = 0x5000_0000;
        // The kernels prefetch one epoch ahead; slack keeps a harmless over-read mapped.
        const SLACK: usize = 4096;

        let geo = Geo {
            arch: self.problem.geo.arch,
            preset: "emu",
            experts: 2,
            top: 2,
            tokens: 20,
            hidden: 512,
            inter: 128,
            distinct: 2,
            route: Route::Uniform,
            hist: None,
            seed: 7,
        };
        let p = Problem::new(&geo, self.cand.spec.kind)?;
        let object = self.candidate()?;
        let program = peacemaker_lift::lift_object(
            &object.bytes,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .map_err(|e| format!("lift: {e}"))?
        .program;
        let emu = Emulator::new(&program, &object.launch_symbol)?.hw_model();

        let mut mem = Memory::default();
        let mut table = Vec::with_capacity(geo.experts * 8);
        for e in 0..geo.experts {
            let mut w = vec![0u8; p.stride + SLACK];
            p.fill_weights(e % geo.distinct, &mut w[..p.stride]);
            let base = EXPERTS + e as u64 * EXPERT_STRIDE;
            assert!((w.len() as u64) < EXPERT_STRIDE);
            table.extend_from_slice(&base.to_le_bytes());
            mem.map(format!("expert{e}"), base, w)?;
        }
        let table_len = table.len() as u64;
        mem.map("table".into(), TABLE, table)?;
        let tiles = Problem::i32_bytes(&p.tiles);
        let sorted = Problem::i32_bytes(&p.sorted);
        let (tiles_len, sorted_len) = (tiles.len() as u64, sorted.len() as u64);
        mem.map("tiles".into(), TILES, tiles)?;
        mem.map("sorted".into(), SORTED, sorted)?;
        let mut xq = if self.cand.gp {
            p.make_xq_gp()
        } else {
            p.xq().to_vec()
        };
        xq.resize(xq.len() + SLACK, 0);
        let xq_len = xq.len() as u64;
        mem.map("xq".into(), XQ, xq)?;
        let y_len = p.y_bytes();
        mem.map("y".into(), Y, vec![0u8; y_len])?;
        let launch = p.launch_for(
            self.cand,
            [
                (TABLE, table_len),
                (TILES, tiles_len),
                (SORTED, sorted_len),
                (XQ, xq_len),
                (Y, y_len as u64),
            ],
        )?;
        mem.map("kernarg".into(), KERNARG, launch.kernarg.clone())?;
        for gy in 0..launch.grid[1] {
            for gx in 0..launch.grid[0] {
                emu.run_workgroup(
                    &EmuLaunch {
                        kernarg_ptr: KERNARG,
                        workgroup: [gx, gy, 0],
                        block: launch.block,
                        dynamic_lds: launch.lds,
                    },
                    &mut mem,
                )
                .map_err(|e| format!("workgroup ({gx},{gy}): {e}"))?;
            }
        }
        let y = mem
            .regions
            .iter()
            .find(|r| r.name == "y")
            .map(|r| r.bytes.clone())
            .ok_or("emulator memory lost the y region")?;
        let reference = p.reference(Sample::Full);
        let report = p.compare(&y, &reference, Sample::Full)?;
        Ok(json!({
            "status": if report.pass { "pass" } else { "fail" },
            "symbol": object.launch_symbol,
            "shape": p.shape_json(),
            "workgroups": u64::from(launch.grid[0]) * u64::from(launch.grid[1]),
            "reference": report.reference,
            "max_abs_err": report.max_abs_err,
            "tolerance": report.tolerance,
        }))
    }

    /// One arm of [`Moe::emu_gate_run`]: lift `object`, run its complete grid on the emulator at `p` with
    /// definedness-tracked LDS and every buffer an exact-size region of its own, `EMU_GUARD` unmapped bytes apart
    /// (no slack: any access past a buffer is an unmapped-memory error). `Err` is the verbatim refusal.
    fn emu_gate_arm(
        p: &Problem,
        entry: Entry,
        object: &Object,
        wgs: Option<&[(u32, u32)]>,
    ) -> Result<EmuArm> {
        use peacemaker_ir::emu::{Emulator, Launch as EmuLaunch, Memory};
        let program = peacemaker_lift::lift_object(
            &object.bytes,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .map_err(|e| format!("lift: {e}"))?
        .program;
        let lds_static = program
            .kernels
            .iter()
            .find(|k| k.symbol.0 == object.launch_symbol)
            .and_then(|k| match &k.abi {
                peacemaker_ir::inst::Abi::Hsa { descriptor, .. } => {
                    Some(u64::from(descriptor.group_segment_fixed_size))
                }
                _ => None,
            });
        let emu = Emulator::new(&program, &object.launch_symbol)?
            .hw_model()
            .gfx1100_as_gfx1151();
        let geo = &p.geo;
        let mut mem = Memory::default();
        // Sampled (extent) runs back only the experts the sampled tiles use; every other table entry is 0 (an access
        // through it is an unmapped-memory error). `vec![0; n]` is calloc'd, so untouched pages cost no memory.
        let needed: Option<std::collections::BTreeSet<i32>> = wgs.map(|w| {
            w.iter()
                .map(|&(_, gy)| p.tiles[gy as usize])
                .filter(|&e| e >= 0)
                .collect()
        });
        let mut table = Vec::with_capacity(geo.experts * 8);
        for e in 0..geo.experts {
            if needed.as_ref().is_some_and(|n| !n.contains(&(e as i32))) {
                table.extend_from_slice(&0u64.to_le_bytes());
                continue;
            }
            let mut w = vec![0u8; p.stride];
            p.fill_weights(e % geo.distinct, &mut w);
            let base = mem.map_guarded(format!("expert{e}"), w, EMU_GUARD)?;
            table.extend_from_slice(&base.to_le_bytes());
        }
        let table_len = table.len() as u64;
        let table_base = mem.map_guarded("table".into(), table, EMU_GUARD)?;
        let tiles = Problem::i32_bytes(&p.tiles);
        let tiles_len = tiles.len() as u64;
        let tiles_base = mem.map_guarded("tiles".into(), tiles, EMU_GUARD)?;
        let sorted = Problem::i32_bytes(&p.sorted);
        let sorted_len = sorted.len() as u64;
        let sorted_base = mem.map_guarded("sorted".into(), sorted, EMU_GUARD)?;
        let xq = if entry.gp {
            p.make_xq_gp()
        } else {
            p.xq().to_vec()
        };
        let xq_len = xq.len() as u64;
        let xq_base = mem.map_guarded("xq".into(), xq, EMU_GUARD)?;
        let y_len = p.y_bytes();
        let y_base = mem.map_guarded("y".into(), vec![0u8; y_len], EMU_GUARD)?;
        let launch = p.launch_for(
            entry,
            [
                (table_base, table_len),
                (tiles_base, tiles_len),
                (sorted_base, sorted_len),
                (xq_base, xq_len),
                (y_base, y_len as u64),
            ],
        )?;
        let kernarg = mem.map_guarded("kernarg".into(), launch.kernarg.clone(), EMU_GUARD)?;
        let mut instructions = 0u64;
        let all: Vec<(u32, u32)>;
        let run: &[(u32, u32)] = match wgs {
            Some(w) => w,
            None => {
                all = (0..launch.grid[1])
                    .flat_map(|gy| (0..launch.grid[0]).map(move |gx| (gx, gy)))
                    .collect();
                &all
            }
        };
        for &(gx, gy) in run {
            if gx >= launch.grid[0] || gy >= launch.grid[1] {
                return Err(format!(
                    "workgroup ({gx},{gy}) outside the launch grid {:?}",
                    launch.grid
                ));
            }
            instructions += emu
                .run_workgroup_counted(
                    &EmuLaunch {
                        kernarg_ptr: kernarg,
                        workgroup: [gx, gy, 0],
                        block: launch.block,
                        dynamic_lds: launch.lds,
                    },
                    &mut mem,
                )
                .map_err(|e| format!("workgroup ({gx},{gy}): {e}"))?;
        }
        let region = mem
            .regions
            .iter()
            .find(|r| r.name == "y")
            .ok_or("emulator memory lost the y region")?;
        let y = match wgs {
            None => region.bytes.clone(),
            Some(w) => {
                // The output rows of every sampled tile (16 grouped rows each), ascending. A live tile that wrote
                // nothing is a failure (an all-zero output would otherwise compare equal to an all-zero incumbent).
                let tile_bytes = 16 * p.row_len * 2;
                let tiles: std::collections::BTreeSet<usize> =
                    w.iter().map(|&(_, gy)| gy as usize).collect();
                let mut y = Vec::with_capacity(tiles.len() * tile_bytes);
                for t in tiles {
                    let (lo, hi) = (t * tile_bytes, (t + 1) * tile_bytes);
                    let live =
                        p.tiles[t] >= 0 && p.sorted[16 * t..16 * t + 16].iter().any(|&x| x >= 0);
                    if live && !(lo..hi).any(|i| region.written.contains(i)) {
                        return Err(format!("sampled live tile {t} wrote no output"));
                    }
                    y.extend_from_slice(&region.bytes[lo..hi]);
                }
                y
            }
        };
        Ok(EmuArm {
            y,
            y_bytes_written: region.written.count() as u64,
            instructions,
            workgroups: run.len() as u64,
            lds_static,
        })
    }

    /// Sampled workgroups of an extent run: grid corners, the first/last live, ragged (padding slots) and sentinel
    /// (all-padding) tiles, the first tile of the last live expert, and eight seeded random (gx, tile) pairs.
    fn extent_wgs(p: &Problem, grid: [u32; 3]) -> Vec<(u32, u32)> {
        let nt = p.ntiles();
        let slots = |t: usize| &p.sorted[16 * t..16 * t + 16];
        let live = |t: usize| p.tiles[t] >= 0 && slots(t).iter().any(|&s| s >= 0);
        let ragged = |t: usize| live(t) && slots(t).iter().any(|&s| s < 0);
        let sentinel = |t: usize| p.tiles[t] < 0;
        let first = |f: &dyn Fn(usize) -> bool| (0..nt).find(|&t| f(t));
        let last = |f: &dyn Fn(usize) -> bool| (0..nt).rev().find(|&t| f(t));
        let mut tiles: Vec<usize> = vec![0, nt - 1];
        for f in [&live as &dyn Fn(usize) -> bool, &ragged, &sentinel] {
            tiles.extend(first(f));
            tiles.extend(last(f));
        }
        if let Some(l) = last(&live) {
            let e = p.tiles[l];
            tiles.extend((0..nt).find(|&t| p.tiles[t] == e));
        }
        let mut wgs: Vec<(u32, u32)> = tiles
            .iter()
            .flat_map(|&t| [0, grid[0] - 1].map(|gx| (gx, t as u32)))
            .collect();
        let mut rng = Rng(p.geo.seed ^ 0xE87E_0000_0000_0000 ^ nt as u64);
        for _ in 0..8 {
            let t = rng.below(nt as u64) as u32;
            let gx = rng.below(u64::from(grid[0])) as u32;
            wgs.push((gx, t));
        }
        wgs.sort_unstable();
        wgs.dedup();
        wgs
    }

    /// The pre-card emulator gate (`--emu`): `cand` and `base` (the incumbent) each run their complete grid on the
    /// emulator at the tiny shapes of [`EMU_GATE_SHAPES`]. It FAILS on: an access outside a mapped buffer, an LDS
    /// access out of bounds or a read of LDS bytes this workgroup never wrote, a barrier deadlock, the instruction
    /// limit, any instruction the emulator or lifter cannot execute (named), an output that is not bit-identical to
    /// the emulated incumbent, and an output that fails the CPU reference. The record is `emu_gate`.
    fn emu_gate_run(&self, cand: &Object, base: &Object) -> Value {
        let arch = self.problem.geo.arch;
        let (cand_entry, base_entry) = (self.cand, self.base.entry());
        let mut cases = Vec::new();
        let mut failures: Vec<String> = Vec::new();
        let (mut instructions, mut workgroups) = (0u64, 0u64);
        let (mut cand_hash, mut base_hash) = (Sha256::new(), Sha256::new());
        let short = |e: &str| e.split(" | ").next().unwrap_or(e).to_owned();
        let all_cases = EMU_GATE_SHAPES
            .iter()
            .map(|&(n, s)| (n, s, false))
            .chain(EMU_EXTENT_SHAPES.iter().map(|&(n, s)| (n, s, true)));
        for (name, shape, extent) in all_cases {
            let built = Geo::parse(&format!("{shape},seed=7"), arch)
                .and_then(|geo| Problem::new(&geo, cand_entry.spec.kind))
                .map(|mut p| {
                    p.tame = true;
                    p
                });
            let p = match built {
                Ok(p) => p,
                Err(e) => {
                    failures.push(format!("{name}: shape {shape}: {e}"));
                    cases.push(json!({"name": name, "status": "fail", "reason": e}));
                    continue;
                }
            };
            let mut case = json!({"name": name, "shape": p.shape_json()});
            let wgs = extent.then(|| Self::extent_wgs(&p, p.geometry(cand_entry.spec).0));
            if let Some(w) = &wgs {
                case["sampled_workgroups"] = json!(w.len());
                case["sampled"] = json!(w);
            }
            let c = Self::emu_gate_arm(&p, cand_entry, cand, wgs.as_deref());
            let b = Self::emu_gate_arm(&p, base_entry, base, wgs.as_deref());
            let mut reasons: Vec<String> = Vec::new();
            match (&c, &b) {
                (Ok(c), Ok(b)) => {
                    let identical = c.y == b.y;
                    if !identical {
                        let first = c.y.iter().zip(&b.y).position(|(x, y)| x != y);
                        reasons.push(format!(
                            "output not bit-identical to the emulated incumbent ({} of {} bytes differ, first at byte {})",
                            c.y.iter().zip(&b.y).filter(|(x, y)| x != y).count(),
                            c.y.len(),
                            first.map_or("n/a".to_owned(), |i| i.to_string())
                        ));
                    }
                    // Extent runs have no CPU reference (it needs every row); the tiny/prod cases carry it.
                    let reference = if extent {
                        Vec::new()
                    } else {
                        p.reference(Sample::Full)
                    };
                    let report = |y: &[u8]| match p.compare(y, &reference, Sample::Full) {
                        Ok(r) => (
                            r.pass,
                            json!({"reference": r.reference, "pass": r.pass,
                                   "max_abs_err": r.max_abs_err, "tolerance": r.tolerance}),
                        ),
                        Err(e) => (false, json!({"pass": false, "error": e})),
                    };
                    let (cand_ok, cand_ref) = if extent {
                        (true, json!("not computed at extent: bit-identity vs the emulated incumbent only"))
                    } else {
                        report(&c.y)
                    };
                    let (base_ok, base_ref) = if extent {
                        (true, json!(null))
                    } else {
                        report(&b.y)
                    };
                    if !cand_ok {
                        reasons.push(format!(
                            "candidate output fails the CPU reference: {cand_ref}"
                        ));
                    }
                    if !base_ok {
                        reasons.push(format!(
                            "emulated incumbent fails the CPU reference: {base_ref}"
                        ));
                    }
                    cand_hash.update(&c.y);
                    base_hash.update(&b.y);
                    instructions += c.instructions;
                    workgroups += c.workgroups;
                    case["workgroups"] = json!(c.workgroups);
                    case["identical_to_incumbent"] = json!(identical);
                    case["reference"] = json!({"candidate": cand_ref, "incumbent": base_ref});
                }
                _ => {}
            }
            for (who, r) in [("candidate", &c), ("incumbent", &b)] {
                match r {
                    Ok(a) => {
                        case[who] = json!({
                            "instructions": a.instructions,
                            "workgroups": a.workgroups,
                            "out_sha256": sha256_hex(&a.y),
                            "out_bytes": a.y.len(),
                            "out_bytes_written": a.y_bytes_written,
                            "lds_static_bytes": a.lds_static,
                        })
                    }
                    Err(e) => reasons.push(format!("{who}: {}", short(e))),
                }
            }
            if reasons.is_empty() {
                case["status"] = json!("pass");
            } else {
                case["status"] = json!("fail");
                case["reason"] = json!(reasons.join("; "));
                failures.push(format!("{name}: {}", reasons.join("; ")));
            }
            cases.push(case);
        }
        let status = if failures.is_empty() { "pass" } else { "fail" };
        json!({
            "status": status,
            "reason": if failures.is_empty() { "ok".to_owned() } else { failures.join(" | ") },
            "shape": EMU_GATE_SHAPES
                .iter()
                .chain(EMU_EXTENT_SHAPES.iter())
                .map(|(n, s)| format!("{n}:{s}"))
                .collect::<Vec<_>>(),
            "instructions": instructions,
            "workgroups": workgroups,
            "cand_out_sha256": format!("{:x}", cand_hash.finalize()),
            "base_out_sha256": format!("{:x}", base_hash.finalize()),
            "guard_bytes": EMU_GUARD,
            "strict_lds": true,
            "inputs": "synthetic, weight scale span narrowed (Problem::tame) so no accumulator reaches emulator-unmeasured SwiGLU divisions",
            "numeric_model": "gfx1100 executed with the gfx1151 WMMA + transcendental model on both arms (no hardware-qualified gfx1100 model); arms compared emulated-vs-emulated and against the CPU reference",
            "candidate_sha256": sha256_hex(&cand.bytes),
            "incumbent_sha256": sha256_hex(&base.bytes),
            "cases": cases,
        })
    }
}

impl Workload for Moe {
    fn kernel(&self) -> &'static str {
        "qwen4_moe_sym"
    }

    fn arch(&self) -> &'static str {
        self.problem.geo.arch.name()
    }

    fn shape_json(&self) -> Value {
        let mut v = self.problem.shape_json();
        v["ref_rows"] = json!(self.sample.label());
        v
    }

    fn record_extra(&self) -> Vec<(String, Value)> {
        match &self.problem.geo.hist {
            Some(h) => vec![("route".to_owned(), h.json())],
            None => Vec::new(),
        }
    }

    fn candidate_knobs(&self) -> Value {
        spec_json(self.cand)
    }

    fn requires_baseline_identity(&self) -> bool {
        // Every NT / gp entry is documented byte-identical to the shipped one.
        true
    }

    fn candidate(&self) -> Result<Object> {
        if let Some((path, bytes)) = &self.cand_object {
            return cand_object_object(self.cand, path, bytes);
        }
        if let Some((path, text)) = &self.rip {
            return rip_object(self.cand, path, text);
        }
        emit_object(
            self.cand,
            with_generator(
                self.cand,
                json!({
                    "source": "peacemaker_isa::kernels::qwen4_moe_sym::emit (emit_down_gp for down_gp_*) via Spec::from_epi + \
                               native::assemble (the `peacemaker-isa emit --kernel qwen4_moe_sym --co` \
                               path: crates/peacemaker-isa/src/bin/peacemaker-isa.rs:202-216, :1547)",
                }),
            ),
        )
    }

    fn baseline(&self) -> Result<Object> {
        self.baseline_object()
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let p = &self.problem;
        let distinct = p.geo.distinct;
        let mut host = vec![0u8; distinct * p.stride];
        host.par_chunks_mut(p.stride)
            .enumerate()
            .for_each(|(i, chunk)| p.fill_weights(i, chunk));
        let upload = |bytes: &[u8]| -> Result<DeviceBuffer> {
            let d = gpu.hip.malloc(bytes.len().max(16)).map_err(err)?;
            gpu.hip.memcpy_htod(&d, bytes).map_err(err)?;
            Ok(d)
        };
        let weights = upload(&host)?;
        drop(host);
        let wbase = weights.as_ptr() as u64;
        let table: Vec<u8> = (0..p.geo.experts)
            .flat_map(|e| (wbase + ((e % distinct) * p.stride) as u64).to_le_bytes())
            .collect();
        let y = || -> Result<DeviceBuffer> {
            let d = gpu.hip.malloc(p.y_bytes().max(16)).map_err(err)?;
            gpu.hip.memset(&d, 0, d.size()).map_err(err)?;
            Ok(d)
        };
        let xq_gp = if self.cand.gp || self.base.entry().gp {
            Some(upload(&p.make_xq_gp())?)
        } else {
            None
        };
        let dev = Dev {
            ptrs: upload(&table)?,
            tiles: upload(&Problem::i32_bytes(&p.tiles))?,
            sorted: upload(&Problem::i32_bytes(&p.sorted))?,
            xq: upload(p.xq())?,
            xq_gp,
            y: [y()?, y()?],
            _weights: weights,
        };
        self.dev = Some(dev);
        // Refuse a wrong layout / extent / alignment before any launch.
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.checked_launch(arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        self.checked_launch(arm)
            .expect("Workload::setup (kernarg contract) must succeed before Workload::launch")
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let y = &self.dev()?.y[arm_index(arm)];
        gpu.hip.memset(y, 0, y.size()).map_err(err)
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let y = &self.dev()?.y[arm_index(arm)];
        let mut v = vec![0u8; self.problem.y_bytes()];
        gpu.hip.memcpy_dtoh(&mut v, y).map_err(err)?;
        Ok(v)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        self.problem.compare(out, self.reference(), self.sample)
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() || a.len() % 2 != 0 {
            return Err(format!(
                "outputs differ in length ({} vs {}) or are not BF16",
                a.len(),
                b.len()
            ));
        }
        Ok(a.par_chunks(1 << 20)
            .zip(b.par_chunks(1 << 20))
            .map(|(x, y)| {
                let mut worst = 0f64;
                for (u, v) in x.chunks_exact(2).zip(y.chunks_exact(2)) {
                    let (u, v) = (
                        u16::from_le_bytes([u[0], u[1]]),
                        u16::from_le_bytes([v[0], v[1]]),
                    );
                    if u == v {
                        continue;
                    }
                    let d = if bf16_nan(u) != bf16_nan(v) {
                        f64::INFINITY
                    } else if bf16_nan(u) {
                        0.0
                    } else {
                        (bf16_to_f64(u) - bf16_to_f64(v)).abs()
                    };
                    worst = worst.max(d);
                }
                worst
            })
            .reduce(|| 0.0, f64::max))
    }

    fn emu(&self) -> Value {
        match self.emu_run() {
            Ok(v) => v,
            Err(reason) => Value::String(format!("refused ({reason})")),
        }
    }

    fn emu_gate(&self, cand: &Object, base: &Object) -> Option<Value> {
        Some(self.emu_gate_run(cand, base))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARCH: Arch = Arch::Gfx1201;

    /// Write a synthetic `route_hist.v1` (2 records: layer 0 rows 3, layer 1 rows 2; E=256 TOP=8).
    fn hist_fixture(tag: &str, tamper: bool) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("pm_loop_hist_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bin = Vec::new();
        for r in 0..5i32 {
            for j in 0..8i32 {
                bin.extend(((r * 17 + j * 3) % 256).to_le_bytes());
            }
        }
        let sha = sha256_hex(&bin);
        if tamper {
            bin[0] ^= 1;
        }
        std::fs::write(dir.join("h.ids.bin"), &bin).unwrap();
        let doc = json!({
            "schema": "route_hist.v1", "n_experts": 256, "topk": 8,
            "records": [
                {"layer": 0, "n": 3, "k": 8, "position": 0},
                {"layer": 1, "n": 2, "k": 8, "position": 0},
            ],
            "ids": {"path": "h.ids.bin", "sha256": sha},
            "id": "cd".repeat(32),
        });
        let p = dir.join("h.json");
        std::fs::write(&p, serde_json::to_vec(&doc).unwrap()).unwrap();
        p
    }

    #[test]
    fn route_hist_replays_deterministically_and_cycles() {
        let p = hist_fixture("det", false);
        let shape = format!("a3b,T=7,route=hist:{}", p.display());
        let a = Geo::parse(&shape, ARCH).unwrap();
        let b = Geo::parse(&shape, ARCH).unwrap();
        assert_eq!(a, b);
        let (ra, rb) = (a.route_table(), b.route_table());
        assert_eq!(ra, rb);
        assert_eq!(ra.len(), 7 * 8);
        // 5 recorded rows: token 5 replays row 0.
        assert_eq!(ra[5 * 8..6 * 8], ra[..8]);
        assert_eq!(ra[8], 17);
        assert_eq!(a.route, Route::Hist);
        let h = a.hist.as_ref().unwrap().json();
        assert_eq!(
            (h["kind"].as_str(), h["id"].as_str()),
            (Some("hist"), Some("cd".repeat(32).as_str()))
        );
        assert!(h["layer"].is_null());
        // `@1` selects layer 1's two rows.
        let l1 = Geo::parse(&format!("{shape}@1"), ARCH).unwrap();
        assert_eq!(l1.hist.as_ref().unwrap().json()["layer"], 1);
        assert_eq!(l1.route_table()[..8], ra[3 * 8..4 * 8]);
        assert_eq!(l1.route_table()[16..24], ra[3 * 8..4 * 8]);
        assert!(Geo::parse(&format!("{shape}@9"), ARCH).is_err());
    }

    #[test]
    fn route_hist_refuses_sha_and_shape_mismatch() {
        let p = hist_fixture("sha", true);
        let e = Geo::parse(&format!("a3b,route=hist:{}", p.display()), ARCH).unwrap_err();
        assert!(e.contains("sha256 mismatch"), "{e}");
        let p = hist_fixture("shape", false);
        let e = Geo::parse(&format!("fn,route=hist:{}", p.display()), ARCH).unwrap_err();
        assert!(
            e.contains("E=256 TOP=8") && e.contains("E=512 TOP=10"),
            "{e}"
        );
    }

    fn knobs(pairs: &[(&str, &str)]) -> Knobs {
        let mut k = Knobs::default();
        for (a, b) in pairs {
            k.0.insert((*a).to_owned(), (*b).to_owned());
        }
        k
    }

    fn tiny(kind: Kind) -> Problem {
        let geo = Geo::parse("E=3,TOP=2,T=12,H=256,I=128,D=2,seed=5", ARCH).unwrap();
        Problem::new(&geo, kind).unwrap()
    }

    #[test]
    fn shape_grammar() {
        let d = Geo::parse("", ARCH).unwrap();
        assert_eq!(
            (d.preset, d.experts, d.top, d.tokens, d.hidden, d.inter, d.distinct),
            ("fn", 512, 10, 4096, 2560, 640, 512)
        );
        assert_eq!((d.route, d.seed), (Route::Uniform, 1));
        let a = Geo::parse("a3b, T=8, route=skew,seed=9", ARCH).unwrap();
        assert_eq!(
            (a.preset, a.experts, a.top, a.tokens, a.hidden, a.inter, a.route, a.seed),
            ("a3b", 256, 8, 8, 2048, 512, Route::Skew, 9)
        );
        // Explicit keys win whatever the order.
        let b = Geo::parse("E=64,a3b", ARCH).unwrap();
        assert_eq!((b.preset, b.experts, b.top), ("a3b", 64, 8));
        for bad in [
            "bogus",
            "T=0",
            "E=4,TOP=8",
            "route=zigzag",
            "Q=1",
            "T=abc",
            "D=0",
            "D=999999",
            "I=100",
            "H=300",
            "T=900000",
        ] {
            assert!(Geo::parse(bad, ARCH).is_err(), "{bad}");
        }
    }

    #[test]
    fn dims_follow_the_plan() {
        let g = Geo::parse("T=100", ARCH).unwrap();
        let gu = Problem::new(&g, Kind::GateUp).unwrap();
        assert_eq!(
            (
                gu.m,
                gu.k,
                gu.epochs,
                gu.x_row_div,
                gu.x_src_rows,
                gu.row_len
            ),
            (1280, 2560, 20, 10, 100, 640)
        );
        assert_eq!((gu.row_bytes, gu.group_bytes), (20 * 68, 136));
        let dn = Problem::new(&g, Kind::Down).unwrap();
        assert_eq!(
            (
                dn.m,
                dn.k,
                dn.epochs,
                dn.x_row_div,
                dn.x_src_rows,
                dn.row_len
            ),
            (2560, 640, 5, 1, 1000, 2560)
        );
        assert_eq!((dn.row_bytes, dn.group_bytes), (5 * 68, 68));
        // pmax = ceil16(L + E*15)
        assert_eq!(gu.pmax, (1000 + 512 * 15usize).div_ceil(16) * 16);
    }

    #[test]
    fn grouping_is_stable_pad16_with_sentinel_tail() {
        let p = tiny(Kind::GateUp);
        let l = p.geo.slots();
        assert_eq!(p.pmax % 16, 0);
        assert_eq!(p.pmax, (l + 3 * 15usize).div_ceil(16) * 16);
        // Every flat slot appears exactly once, in its expert's tiles, ascending within a run.
        let mut seen = vec![false; l];
        let route = p.geo.route_table();
        let mut last: Vec<i32> = vec![-1; 3];
        for (row, &s) in p.sorted.iter().enumerate() {
            if s < 0 {
                continue;
            }
            let e = p.tiles[row / 16];
            assert_eq!(route[s as usize] as i32, e);
            assert!(s > last[e as usize]);
            last[e as usize] = s;
            assert!(!std::mem::replace(&mut seen[s as usize], true));
        }
        assert!(seen.iter().all(|&s| s));
        // Tiles of one expert are contiguous; a tile's padding is only at its run's tail.
        let mut runs = Vec::new();
        for &t in &p.tiles {
            if runs.last() != Some(&t) {
                runs.push(t);
            }
        }
        let live: Vec<i32> = runs.iter().copied().filter(|&t| t >= 0).collect();
        assert!(live.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*p.tiles.last().unwrap(), -1);
        assert_eq!(p.live_rows().len(), l);
    }

    use hipfire_kernel_abi::FormatId;

    /// Aligned fake device addresses and the exact allocation sizes the workload creates.
    fn fake_ptrs(p: &Problem, gp: bool) -> [(u64, u64); 5] {
        [
            (0x10_0000, p.geo.experts as u64 * 8),
            (0x20_0000, p.ntiles() as u64 * 4),
            (0x30_0000, p.pmax as u64 * 4),
            (
                0x40_0000,
                if gp { p.xq_gp_bytes() } else { p.xq().len() } as u64,
            ),
            (0x50_0000, p.y_bytes() as u64),
        ]
    }

    fn entry(epi: &str) -> Entry {
        Entry::parse(epi, ARCH).unwrap()
    }

    #[test]
    fn kernarg_packing_layout() {
        let p = tiny(Kind::Down);
        let a = p.kernarg(entry("down_nt4"), fake_ptrs(&p, false)).unwrap();
        assert_eq!(KERNARG_BYTES, 64);
        assert_eq!(a.len(), 64);
        let u64_at = |o: usize| u64::from_le_bytes(a[o..o + 8].try_into().unwrap());
        let i32_at = |o: usize| i32::from_le_bytes(a[o..o + 4].try_into().unwrap());
        assert_eq!(
            [u64_at(0), u64_at(8), u64_at(16), u64_at(24), u64_at(32)],
            [0x10_0000, 0x20_0000, 0x30_0000, 0x40_0000, 0x50_0000]
        );
        assert_eq!(
            [
                i32_at(40),
                i32_at(44),
                i32_at(48),
                i32_at(52),
                i32_at(56),
                i32_at(60)
            ],
            [
                p.m as i32,
                p.k as i32,
                1,
                p.pmax as i32,
                p.geo.slots() as i32,
                3
            ]
        );
        let g = tiny(Kind::GateUp);
        let a = g.kernarg(entry("gate_up"), fake_ptrs(&g, false)).unwrap();
        assert_eq!(i32::from_le_bytes(a[48..52].try_into().unwrap()), 2);
        assert_eq!(i32::from_le_bytes(a[56..60].try_into().unwrap()), 12);
        // Launch: grid (M/64, tiles), block per spec, no dynamic LDS.
        for (epi, grid_x, block) in [
            ("gate_up", 4, 64),
            ("gate_up_nt4", 4, 128),
            ("gate_up_nt8", 4, 128),
        ] {
            let l = g.launch_for(entry(epi), fake_ptrs(&g, false)).unwrap();
            assert_eq!(
                (l.grid, l.block, l.lds),
                ([grid_x, g.ntiles() as u32, 1], [block, 1, 1], 0),
                "{epi}"
            );
        }
        let l = p
            .launch_for(entry("down_nt4"), fake_ptrs(&p, false))
            .unwrap();
        assert_eq!(l.grid[0], (p.m / 64) as u32);
    }

    /// The workload's kernarg equals an independent contract encoding
    /// (`validate_invocation` + `encode_into` on the family's explicit static) for every
    /// shipped gfx1201 entry, and a non-exported entry (`nt4r2`) encodes through the NT4
    /// product's contract.
    #[test]
    fn moe_kernarg_is_contract_encoding() {
        use hipfire_kernel_abi::families::qwen4_moe_sym as f;
        use hipfire_kernel_abi::{encode_into, KernelAbiV1};
        let rr2 = Entry {
            spec: Spec {
                arch: ARCH,
                kind: Kind::Down,
                nt: 4,
                rr: 2,
                codec: WeightCodec::Mq4Sym,
            },
            gp: false,
        };
        let cases: [(Kind, Entry, &KernelAbiV1, [FormatId; 5]); 8] = [
            (
                Kind::GateUp,
                entry("gate_up"),
                &f::QWEN4_MOE_SYM_GATE_UP,
                gate_up_formats(),
            ),
            (
                Kind::GateUp,
                entry("gate_up_nt4"),
                &f::QWEN4_MOE_SYM_GATE_UP,
                gate_up_formats(),
            ),
            (
                Kind::GateUp,
                entry("gate_up_nt8"),
                &f::QWEN4_MOE_SYM_GATE_UP,
                gate_up_formats(),
            ),
            (
                Kind::Down,
                entry("down_nt4"),
                &f::QWEN4_MOE_SYM_DOWN,
                down_formats(false),
            ),
            (
                Kind::Down,
                entry("down_nt8"),
                &f::QWEN4_MOE_SYM_DOWN,
                down_formats(false),
            ),
            (
                Kind::Down,
                entry("down_gp_nt4"),
                &f::QWEN4_MOE_SYM_DOWN_GP,
                down_formats(true),
            ),
            (
                Kind::Down,
                entry("down_gp_nt8"),
                &f::QWEN4_MOE_SYM_DOWN_GP,
                down_formats(true),
            ),
            (Kind::Down, rr2, &f::QWEN4_MOE_SYM_DOWN, down_formats(false)),
        ];
        for (kind, e, abi, formats) in cases {
            let p = tiny(kind);
            let ptrs = fake_ptrs(&p, e.gp);
            let got = p.kernarg(e, ptrs).unwrap();
            // The reference export: the entry itself when the contract has it, else NT4.
            let reference = if find_export(AbiArch::Gfx1201, &e.symbol()).is_some() {
                e
            } else {
                Entry {
                    spec: Spec {
                        nt: 4,
                        rr: 1,
                        ..e.spec
                    },
                    gp: e.gp,
                }
            };
            let export = abi
                .export(AbiArch::Gfx1201, &reference.symbol())
                .unwrap_or_else(|| panic!("{} not in {}", reference.symbol(), abi.id));
            let binds: Vec<OperandBinding> = ptrs
                .iter()
                .zip(formats)
                .map(|(&(address, bytes), format)| OperandBinding {
                    address,
                    bytes: Some(bytes),
                    format,
                })
                .collect();
            let (grid, block, lds) = p.geometry(reference.spec);
            let v = validate_invocation(
                abi,
                &InvocationV1 {
                    arch: AbiArch::Gfx1201,
                    symbol: export.symbol,
                    params: &[
                        p.m as u64,
                        p.k as u64,
                        p.x_row_div as u64,
                        p.pmax as u64,
                        p.x_src_rows as u64,
                        p.geo.experts as u64,
                    ],
                    operands: &binds,
                    grid,
                    block,
                    dynamic_lds: lds,
                },
            )
            .unwrap();
            let mut want = [0u8; 64];
            encode_into(&v, &mut want).unwrap();
            assert_eq!(got, want, "{}", e.variant());
        }
    }

    fn gate_up_formats() -> [FormatId; 5] {
        use hipfire_kernel_abi::families::qwen4_moe_sym as f;
        [
            f::FORMAT_EXPERT_PTRS_GATE_UP_QT44,
            f::FORMAT_EXPERT_TILE_IDS,
            f::FORMAT_SORTED_SLOT_INDEX,
            f::FORMAT_XQ_GATE_UP_FWHT256_TOKEN_ROWS,
            f::FORMAT_Y_GATE_UP_SWIGLU_BF16,
        ]
    }

    fn down_formats(gp: bool) -> [FormatId; 5] {
        use hipfire_kernel_abi::families::qwen4_moe_sym as f;
        [
            f::FORMAT_EXPERT_PTRS_DOWN_QT53,
            f::FORMAT_EXPERT_TILE_IDS,
            f::FORMAT_SORTED_SLOT_INDEX,
            if gp {
                f::FORMAT_XQ_DOWN_ROT128_GROUPED_POSITION
            } else {
                f::FORMAT_XQ_DOWN_ROT128_FLAT_SLOTS
            },
            f::FORMAT_Y_DOWN_BF16,
        ]
    }

    /// A candidate whose generator layout drifts from the contract (a moved scalar, a renamed
    /// pointer, a larger segment) is refused before any launch; so are a short extent, a null
    /// operand and a misaligned one.
    #[test]
    fn moe_wrong_candidate_layout_is_rejected_before_launch() {
        let p = tiny(Kind::GateUp);
        let e = entry("gate_up_nt4");
        let ptrs = fake_ptrs(&p, false);
        assert!(p.kernarg_with_layout(e, e.spec.kernargs(), ptrs).is_ok());
        let mutate: [(&str, fn(&mut peacemaker_isa::KernargLayout)); 3] = [
            ("moved M", |l| l.args[5].offset = 44),
            ("renamed Xq", |l| l.args[3].name = "X".into()),
            ("segment", |l| l.size = 72),
        ];
        for (what, f) in mutate {
            let mut layout = e.spec.kernargs();
            f(&mut layout);
            let err = p.kernarg_with_layout(e, layout, ptrs).unwrap_err();
            assert!(err.contains("kernarg layout"), "{what}: {err}");
        }
        let mut short = ptrs;
        short[4].1 -= 2;
        assert!(p.kernarg(e, short).is_err());
        let mut null = ptrs;
        null[1].0 = 0;
        assert!(p.kernarg(e, null).is_err());
        let mut odd = ptrs;
        odd[4].0 += 8;
        assert!(p.kernarg(e, odd).is_err(), "Y needs 16-byte alignment");
    }

    #[test]
    fn synthetic_blobs_are_valid_symmetric_mq4() {
        for kind in [Kind::GateUp, Kind::Down] {
            let p = tiny(kind);
            let mut w = vec![0u8; p.stride];
            p.fill_weights(1, &mut w);
            let mut again = vec![0u8; p.stride];
            p.fill_weights(1, &mut again);
            assert_eq!(w, again, "deterministic");
            let mut other = vec![0u8; p.stride];
            p.fill_weights(0, &mut other);
            assert_ne!(w, other);
            for g in w.chunks_exact(p.group_bytes) {
                for h in 0..p.group_bytes / 68 {
                    let sc = f16::from_bits(u16::from_le_bytes([g[4 * h], g[4 * h + 1]]));
                    let zp = f16::from_bits(u16::from_le_bytes([g[4 * h + 2], g[4 * h + 3]]));
                    assert!(sc.is_finite() && zp.is_finite());
                    assert_eq!(zp.to_f32(), -8.0 * sc.to_f32(), "zp == -8*sc");
                }
            }
        }
    }

    #[test]
    fn xq_is_epoch_major_a4_blocks() {
        let p = tiny(Kind::Down);
        let xq = p.xq();
        assert_eq!(xq.len(), p.epochs * p.x_src_rows * 72);
        let row = p.x_row(5);
        for h in 0..p.epochs {
            let o = (h * p.x_src_rows + 5) * 72;
            let d = f32::from_le_bytes(xq[o..o + 4].try_into().unwrap());
            let s = i32::from_le_bytes(xq[o + 4..o + 8].try_into().unwrap());
            assert!(d.is_normal() && (0.125..2.0).contains(&d));
            assert_eq!(row.d[h], d);
            let vals = &row.v[h * 128..(h + 1) * 128];
            assert_eq!(vals.iter().map(|&v| i32::from(v)).sum::<i32>(), s);
            assert!(vals.iter().all(|&v| (-8..=7).contains(&v)));
            // element j = byte j/2, low nibble first
            assert_eq!(i32::from(((xq[o + 8] << 4) as i8) >> 4), i32::from(vals[0]));
            assert_eq!(i32::from((xq[o + 8] as i8) >> 4), i32::from(vals[1]));
        }
    }

    /// Hand-checkable fold: weights `+1` (nibble 9) against activations `3`.
    #[test]
    fn fold_hand_checked_down() {
        let (m, epochs, rb) = (2usize, 1usize, 68usize);
        let mut w = vec![0u8; m * rb];
        w[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // sc = 0.5
        w[4..68].fill(0x99); // u = 9 -> w = +1
        w[68..70].copy_from_slice(&0x3c00u16.to_le_bytes()); // sc = 1.0
        w[72..136].fill(0x77); // u = 7 -> w = -1
        let flat = |d: f32, v: Vec<i8>| XRow { d: vec![d], v };
        let xs = [
            flat(2.0, vec![3; 128]),
            flat(
                1.0,
                (0..128).map(|j| if j % 2 == 0 { 7 } else { -8 }).collect(),
            ),
        ];
        let s = fold_slots(Kind::Down, m, epochs, rb, &w, &xs);
        // x0: row0 0.5*2 * (128*1*3) = 384; row1 1*2 * (128*-1*3) = -768.
        assert_eq!((s[0], s[1]), (384.0, -768.0));
        // x1: row0 0.5 * (64*7 - 64*8) = -32; row1 1 * (-64*7 + 64*8) = 64.
        assert_eq!((s[2], s[3]), (-32.0, 64.0));
        assert_eq!(bf16_bits(384.0), 0x43c0);
    }

    #[test]
    fn fold_hand_checked_gate_up_layout() {
        // One K256 group of 136 B: [sc0 zp0 sc1 zp1][64 B epoch 0][64 B epoch 1].
        let (m, epochs, rb) = (2usize, 2usize, 136usize);
        let mut w = vec![0u8; m * rb];
        w[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // sc0 = 1.0
        w[4..6].copy_from_slice(&0x3800u16.to_le_bytes()); // sc1 = 0.5
        w[8..72].fill(0x99); // epoch 0: +1
        w[72..136].fill(0xaa); // epoch 1: +2
        let xs = [XRow {
            d: vec![1.0, 1.0],
            v: vec![3; 256],
        }];
        let s = fold_slots(Kind::GateUp, m, epochs, rb, &w, &xs);
        // epoch 0: 1.0 * 384 = 384; epoch 1: fma(0.5, 768, 384) = 768; row 1: scale 0 -> 0.
        assert_eq!((s[0], s[1]), (768.0, 0.0));
    }

    #[test]
    fn bf16_epilogue_pieces() {
        // RNE ties-to-even on exact halfway f32 bit patterns.
        assert_eq!(bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16_bits(f32::INFINITY), 0x7f80);
        assert_eq!(rt_f32(f32::from_bits(0x3f81_8000)).to_bits(), 0x3f82_0000);
        assert!(rt_f32(f32::NAN).is_nan());
        // silu(0) = 0; silu(16) * 1 rounds to 16.0 in BF16; silu(-200) = 0 (e^200 overflows to inf).
        assert_eq!(bf16_bits(portable_silu_mul(0.0, 1.0)), 0);
        assert_eq!(bf16_bits(portable_silu_mul(16.0, 1.0)), 0x4180);
        assert_eq!(portable_silu_mul(-200.0, 1.0), 0.0);
        let x = portable_silu_mul(1.0, 2.0); // 2 / (1 + e^-1) = 1.46211...
        assert!((x - 1.462_117_1).abs() < 1e-6, "{x}");
        assert_eq!(
            (bf16_ord(0x8000), bf16_ord(0), bf16_ord(0x8001)),
            (0, 0, -1)
        );
        assert_eq!(bf16_ulp(0x3f80), 2f64.powi(-7));
        assert_eq!(bf16_ulp(0), 2f64.powi(-133));
    }

    /// The comparison machinery against a hand-built output: exact for down, 1 ulp for gate/up,
    /// padding / sentinel rows must be zero.
    #[test]
    fn compare_tolerances_and_zero_checks() {
        for kind in [Kind::Down, Kind::GateUp] {
            let p = tiny(kind);
            let reference = p.reference(Sample::Full);
            assert_eq!(reference.len(), p.geo.slots());
            let mut out = vec![0u8; p.y_bytes()];
            for (row, bits) in &reference {
                for (c, b) in bits.iter().enumerate() {
                    let o = (row * p.row_len + c) * 2;
                    out[o..o + 2].copy_from_slice(&b.to_le_bytes());
                }
            }
            let ok = p.compare(&out, &reference, Sample::Full).unwrap();
            assert!(ok.pass, "{}", ok.reference);
            assert_eq!(ok.max_abs_err, 0.0);
            // A finite element nudged one BF16 step.
            let (row, bits) = &reference[reference.len() / 2];
            let c = bits
                .iter()
                .position(|&b| b & 0x7f80 != 0 && b & 0x7f80 != 0x7f80)
                .unwrap();
            let o = (row * p.row_len + c) * 2;
            let mut bumped = out.clone();
            let b = bits[c];
            bumped[o..o + 2].copy_from_slice(&(b ^ 1).to_le_bytes());
            let r1 = p.compare(&bumped, &reference, Sample::Full).unwrap();
            assert_eq!(r1.pass, kind == Kind::GateUp, "{kind:?}: 1 ulp");
            assert!(r1.max_abs_err > 0.0);
            if kind == Kind::GateUp {
                assert!(r1.max_abs_err <= r1.tolerance);
            }
            // Two steps fail for both kinds.
            let mut far = out.clone();
            let two = b ^ 2;
            far[o..o + 2].copy_from_slice(&two.to_le_bytes());
            assert!(!p.compare(&far, &reference, Sample::Full).unwrap().pass);
            // A padding row or a sentinel tile row that is not +0 fails.
            let pad = (0..p.pmax)
                .find(|&r| p.sorted[r] < 0 && p.tiles[r / 16] >= 0)
                .unwrap();
            let mut bad = out.clone();
            bad[pad * p.row_len * 2] = 1;
            assert!(!p.compare(&bad, &reference, Sample::Full).unwrap().pass);
            let mut bad = out.clone();
            let last = (p.pmax - 1) * p.row_len * 2;
            bad[last + 1] = 0x3f;
            let r = p.compare(&bad, &reference, Sample::Full).unwrap();
            assert!(
                !r.pass && r.reference.contains("(bad 1), mismatching"),
                "{}",
                r.reference
            );
            assert!(p.compare(&out[2..], &reference, Sample::Full).is_err());
        }
    }

    #[test]
    fn sampling_is_even_and_includes_the_ends() {
        let live: Vec<usize> = (0..100).map(|i| i * 3).collect();
        let s = pick_evenly(&live, 5);
        assert_eq!(s, vec![live[0], live[24], live[49], live[74], live[99]]);
        assert_eq!(pick_evenly(&live, 1), vec![0]);
        assert_eq!(pick_evenly(&live, 100), live);
        assert_eq!(pick_evenly(&live, 1000), live);
        assert_eq!(Sample::parse("full").unwrap(), Sample::Full);
        assert_eq!(Sample::parse("sample:7").unwrap(), Sample::Rows(7));
        assert!(Sample::parse("sample:0").is_err());
        assert!(Sample::parse("rows").is_err());
        let p = tiny(Kind::Down);
        let part = p.reference(Sample::Rows(10));
        assert_eq!(part.len(), 10);
        let full = p.reference(Sample::Full);
        for (row, bits) in &part {
            let f = full.iter().find(|r| r.0 == *row).unwrap();
            assert_eq!(&f.1, bits, "sampled row equals the full reference row");
        }
    }

    #[test]
    fn knob_to_spec_and_symbol_mapping() {
        let w = new("", &mut knobs(&[])).unwrap();
        let k = w.candidate_knobs();
        assert_eq!(k["epi"], "gate_up");
        assert_eq!(k["symbol"], "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201");
        assert_eq!(
            (k["threads"].as_u64(), k["lds_bytes"].as_u64()),
            (Some(64), Some(0))
        );
        for (epi, symbol, threads, lds) in [
            ("down", "qwen4_moe_down_iu4_sym_pm_gfx1201", 128, 0),
            ("down_nt4", "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4", 128, 0),
            ("down_nt8", "qwen4_moe_down_iu4_sym_pm_gfx1201_nt8", 128, 0),
            (
                "gate_up_nt4",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
                128,
                8192,
            ),
            (
                "gate_up_nt8",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
                128,
                16384,
            ),
        ] {
            let w = new("T=64", &mut knobs(&[("epi", epi)])).unwrap();
            let k = w.candidate_knobs();
            assert_eq!(k["symbol"], symbol);
            assert_eq!(
                (k["threads"].as_u64(), k["lds_bytes"].as_u64()),
                (Some(threads), Some(lds)),
                "{epi}"
            );
            assert_eq!(
                w.shape_json()["kind"],
                if epi.starts_with("down") {
                    "down"
                } else {
                    "gate_up"
                }
            );
        }
        // Unparsable knobs fail at construction; unsupported combos fail in candidate().
        assert!(new("", &mut knobs(&[("epi", "up")])).is_err());
        assert!(new("", &mut knobs(&[("epi", "down_ntx")])).is_err());
        assert!(new("", &mut knobs(&[("ref-rows", "some")])).is_err());
        for epi in ["down_nt4r2", "gate_up_nt16", "gate_up_nt3"] {
            let w = new("T=64", &mut knobs(&[("epi", epi)])).unwrap();
            assert!(w.candidate().is_err(), "{epi}");
        }
        // A base-epi of the other kind is refused; unknown extras stay for the driver's finish().
        assert!(new("", &mut knobs(&[("epi", "down"), ("base-epi", "gate_up")])).is_err());
        let mut rest = knobs(&[("epi", "down"), ("base-tile", "x")]);
        new("", &mut rest).unwrap();
        assert_eq!(rest.0.len(), 1);
        assert!(rest.0.contains_key("base-tile"));
    }

    #[test]
    fn shipped_baseline_selection_follows_production() {
        // Resident: down NT4 always; gate/up NT8 iff total_slots >= 128 * n_experts.
        assert_eq!(shipped_nt(Kind::Down, 10_000_000, 512), 4);
        assert_eq!(shipped_nt(Kind::GateUp, 128 * 512 - 1, 512), 4);
        assert_eq!(shipped_nt(Kind::GateUp, 128 * 512, 512), 8);
        assert_eq!(shipped_nt(Kind::GateUp, 5, 0), 4);
        for (shape, epi, want) in [
            (
                "T=4096",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
            ),
            (
                "T=6553",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4",
            ),
            (
                "T=6554",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8",
            ),
            (
                "T=8192",
                "down_nt8",
                "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4",
            ),
        ] {
            let w = new(shape, &mut knobs(&[("epi", epi)])).unwrap();
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, want, "{shape} {epi}");
            assert_eq!(b.source["symbol"], want);
            assert!(b.source["source"]
                .as_str()
                .unwrap()
                .contains("gemm.rs:49334"));
            assert_eq!(
                b.source["sha256"],
                sha256_hex(rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201)
            );
            assert_eq!(b.bytes, rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201);
            assert_eq!(b.exports.len(), 6);
            assert!(b.exports.contains(&b.launch_symbol));
            assert_eq!(
                b.desc["artifact"],
                "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco"
            );
        }
        // --base-epi emits that entry (a native-writer ELF) and inherits nothing else.
        let w = new(
            "T=64",
            &mut knobs(&[("epi", "down_nt8"), ("base-epi", "down_nt4")]),
        )
        .unwrap();
        let b = w.baseline().unwrap();
        assert_eq!(b.launch_symbol, "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4");
        assert_eq!(b.exports, [b.launch_symbol.clone()]);
        assert_eq!(&b.bytes[..4], b"\x7fELF");
        assert!(b.desc["knobs"]["epi"] == "down_nt4");
    }

    #[test]
    fn candidate_is_the_native_writer_link_of_the_emitted_text() {
        let w = new("T=64", &mut knobs(&[("epi", "down_nt4")])).unwrap();
        let c = w.candidate().unwrap();
        let spec = Spec::from_epi("down_nt4", ARCH).unwrap();
        let e = qwen4_moe_sym::emit(spec).unwrap();
        assert_eq!(
            c.bytes,
            peacemaker_isa::native::assemble(&e.s_text, ARCH).unwrap()
        );
        assert_eq!(c.launch_symbol, spec.symbol());
        assert_eq!(c.exports, [spec.symbol()]);
        assert_eq!(&c.bytes[..4], b"\x7fELF");
    }

    #[test]
    fn fn_presets_set_geometry_and_default_entry() {
        for (name, epi, t) in FN_PRESETS {
            let w = new(&format!("fn:{name}"), &mut knobs(&[])).unwrap();
            let sh = w.shape_json();
            assert_eq!(
                (
                    sh["experts"].as_u64(),
                    sh["top_k"].as_u64(),
                    sh["hidden"].as_u64(),
                    sh["intermediate"].as_u64(),
                    sh["tokens"].as_u64()
                ),
                (Some(512), Some(10), Some(2560), Some(640), Some(t as u64)),
                "{name}"
            );
            assert_eq!(w.candidate_knobs()["epi"], epi, "{name}");
        }
        // An explicit --epi wins; unknown presets and doubled presets are refused.
        let w = new("fn:down_gp_nt4-t4096", &mut knobs(&[("epi", "down_nt4")])).unwrap();
        assert_eq!(w.candidate_knobs()["epi"], "down_nt4");
        assert!(new("fn:nope", &mut knobs(&[])).is_err());
        assert!(new("fn:down_gp_nt4-t4096,fn:gate_up_nt4-t4096", &mut knobs(&[])).is_err());
        // Gate/up T=16384 is wide in production: the shipped baseline is NT8, the candidate NT4.
        let w = new("fn:gate_up_nt4-t16384", &mut knobs(&[])).unwrap();
        assert_eq!(
            w.baseline().unwrap().launch_symbol,
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8"
        );
        let w = new("fn:gate_up_nt4-t4096", &mut knobs(&[])).unwrap();
        assert_eq!(
            w.baseline().unwrap().launch_symbol,
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4"
        );
        assert!(w.requires_baseline_identity());
    }

    #[test]
    fn down_gp_entry_grammar_and_symbols() {
        let w = new("T=64", &mut knobs(&[("epi", "down_gp_nt4")])).unwrap();
        let k = w.candidate_knobs();
        assert_eq!(k["epi"], "down_gp_nt4");
        assert_eq!(k["gp"], true);
        assert_eq!(k["symbol"], "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4");
        let w = new("T=64", &mut knobs(&[("epi", "down_gp_nt8")])).unwrap();
        assert_eq!(
            w.candidate_knobs()["symbol"],
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8"
        );
        for bad in ["down_gp", "gate_up_gp_nt4", "down_gp_nt4x"] {
            assert!(new("T=64", &mut knobs(&[("epi", bad)])).is_err(), "{bad}");
        }
        // Only NT4/NT8 have a twin: the refusal is emit_down_gp's, surfaced by candidate().
        let w = new("T=64", &mut knobs(&[("epi", "down_gp_nt1")])).unwrap();
        assert!(w.candidate().is_err());
        // The candidate is the native-writer link of emit_down_gp.
        let w = new("T=64", &mut knobs(&[("epi", "down_gp_nt4")])).unwrap();
        let c = w.candidate().unwrap();
        let spec = Spec::from_epi("down_nt4", ARCH).unwrap();
        let e = qwen4_moe_sym::emit_down_gp(spec).unwrap();
        assert_eq!(
            c.bytes,
            peacemaker_isa::native::assemble(&e.s_text, ARCH).unwrap()
        );
        assert_eq!(c.exports, ["qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4"]);
    }

    #[test]
    fn down_gp_baseline_is_the_shipped_gp_bundle() {
        let w = new("fn:down_gp_nt4-t4096", &mut knobs(&[])).unwrap();
        let b = w.baseline().unwrap();
        assert_eq!(b.launch_symbol, "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4");
        assert_eq!(b.bytes, rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201);
        assert_eq!(
            b.source["sha256"],
            sha256_hex(rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201)
        );
        assert_eq!(b.exports, qwen4_moe_down_gp::symbols(ARCH));
        assert_eq!(
            b.desc["artifact"],
            "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco"
        );
        // An nt8 gp candidate still times against the resident NT4 twin.
        let w = new("T=64", &mut knobs(&[("epi", "down_gp_nt8")])).unwrap();
        assert_eq!(
            w.baseline().unwrap().launch_symbol,
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4"
        );
        // A flat --base-epi against a gp candidate is an emitted flat entry.
        let w = new(
            "T=64",
            &mut knobs(&[("epi", "down_gp_nt4"), ("base-epi", "down_nt4")]),
        )
        .unwrap();
        assert_eq!(
            w.baseline().unwrap().launch_symbol,
            "qwen4_moe_down_iu4_sym_pm_gfx1201_nt4"
        );
    }

    #[test]
    fn grouped_plane_holds_the_flat_records_at_their_positions() {
        let p = tiny(Kind::Down);
        let (flat, gp) = (p.xq(), p.make_xq_gp());
        assert_eq!(gp.len(), p.epochs * p.pmax * XBLK);
        let mut live = 0;
        for h in 0..p.epochs {
            for pos in 0..p.pmax {
                let got = &gp[(h * p.pmax + pos) * XBLK..][..XBLK];
                if p.tiles[pos / 16] >= 0 && p.sorted[pos] >= 0 {
                    let src = (h * p.x_src_rows + p.sorted[pos] as usize / p.x_row_div) * XBLK;
                    assert_eq!(got, &flat[src..src + XBLK]);
                    live += 1;
                } else {
                    assert!(got.iter().all(|&b| b == 0), "padding position stays zero");
                }
            }
        }
        assert_eq!(live, p.epochs * p.geo.slots());
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/qwen4_moe_sym.rip");

    #[test]
    fn rip_twin_is_identical_to_generator_for_every_entry() {
        for epi in [
            "gate_up",
            "gate_up_nt4",
            "gate_up_nt8",
            "down",
            "down_nt4",
            "down_nt8",
            "down_gp_nt4",
            "down_gp_nt8",
        ] {
            let e = Entry::parse(epi, ARCH).unwrap();
            let o = rip_object(e, "twin.rip", TWIN).unwrap();
            assert_eq!(o.launch_symbol, e.symbol(), "{epi}");
            assert_eq!(o.desc["rip"]["identical_to_generator"], true, "{epi}");
            assert_eq!(o.desc["rip"]["variant"], epi);
            assert_eq!(o.desc["rip"]["sha256"], sha256_hex(TWIN.as_bytes()));
        }
    }

    #[test]
    fn rip_reordered_kloop_keeps_the_abi_and_never_claims_identity() {
        let edited = TWIN.replace(
            "            moent::advance(wg)?;\n            if g.gate_up() && g.nt() == 8 {\n                wg.barrier(())?;\n            }\n            moent::load_w(wg, g, 0)?;\n",
            "            if g.gate_up() && g.nt() == 8 {\n                wg.barrier(())?;\n            }\n            moent::load_w(wg, g, 0)?;\n            moent::advance(wg)?;\n",
        );
        assert_ne!(edited, TWIN);
        for epi in ["down_gp_nt4", "gate_up_nt8"] {
            let e = Entry::parse(epi, ARCH).unwrap();
            if let Ok(o) = rip_object(e, "edited.rip", &edited) {
                assert_eq!(o.desc["rip"]["identical_to_generator"], false, "{epi}");
            }
        }
    }

    #[test]
    fn rip_is_refused_for_an_entry_the_script_does_not_declare() {
        let edited = TWIN.replace("        \"down_gp_nt8\" => Ok((\"down\", 8, true)),\n", "");
        assert_ne!(edited, TWIN);
        let e = Entry::parse("down_gp_nt8", ARCH).unwrap();
        assert!(rip_object(e, "edited.rip", &edited).is_err());
        let e = Entry::parse("down_gp_nt4", ARCH).unwrap();
        assert!(rip_object(e, "edited.rip", &edited).is_ok());
    }

    #[test]
    fn emu_never_fabricates() {
        for epi in ["down", "gate_up_nt4", "down_gp_nt4"] {
            let w = new("T=64", &mut knobs(&[("epi", epi)])).unwrap();
            match w.emu() {
                Value::String(s) => assert!(s.starts_with("refused ("), "{s}"),
                Value::Object(o) => {
                    assert!(matches!(o["status"].as_str(), Some("pass" | "fail")));
                    assert!(o["workgroups"].as_u64().unwrap() > 0);
                }
                other => panic!("{other}"),
            }
        }
    }

    /// The pre-card gate on the shipped gfx1100 object (candidate = the in-tree emit, incumbent = the shipped
    /// bundle): both run the whole grid on the emulator with guard gaps and strict LDS and agree bit-for-bit.
    #[test]
    fn emu_gate_passes_the_in_tree_gfx1100_entries() {
        for epi in ["gate_up", "down"] {
            let w = on("E=2,TOP=2,T=20,H=512,I=128", &[("epi", epi)], Arch::Gfx1100).unwrap();
            let g = w
                .emu_gate(&w.candidate().unwrap(), &w.baseline().unwrap())
                .unwrap();
            assert_eq!(g["status"], "pass", "{epi}: {g}");
            assert!(g["instructions"].as_u64().unwrap() > 0);
            assert_eq!(g["cand_out_sha256"], g["base_out_sha256"], "{epi}");
            let cases = g["cases"].as_array().unwrap();
            assert_eq!(cases.len(), EMU_GATE_SHAPES.len() + EMU_EXTENT_SHAPES.len());
            for c in &cases[EMU_GATE_SHAPES.len()..] {
                assert!(c["sampled_workgroups"].as_u64().unwrap() >= 8, "{c}");
            }
        }
    }

    fn tiny_on(kind: Kind, arch: Arch) -> Problem {
        let geo = Geo::parse("E=3,TOP=2,T=12,H=256,I=128,D=2,seed=5", arch).unwrap();
        Problem::new(&geo, kind).unwrap()
    }

    fn on(shape: &str, pairs: &[(&str, &str)], arch: Arch) -> Result<Box<dyn Workload>> {
        new_with_rip(shape, &mut knobs(pairs), None, arch)
    }

    /// gfx1151 / gfx1100 baselines are that arch's own shipped objects and symbols, with the
    /// artifact sha256 of the embedded const; the gfx1151 gate/up takes the streamed NT8 entry
    /// exactly on production's measured shape.
    #[test]
    fn gfx11_shipped_baselines_are_the_per_arch_objects() {
        use rdna_compute::kernels as k;
        let nt4 = k::QWEN4_MOE_IU4_SYM_PM_GFX1151;
        let cases: [(Arch, &str, &str, &str, &[u8], &str); 9] = [
            (
                Arch::Gfx1151,
                "fn,T=4096",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
                nt4,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1151,
                "fn,T=4096",
                "down",
                "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4",
                nt4,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1151,
                "fn,T=7167",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
                nt4,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1151,
                "fn,T=7168",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt8_stream2",
                k::QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151,
                "kernels/qwen4_moe_gate_up_stream8_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1151,
                "fn,T=8192",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt8_stream2",
                k::QWEN4_MOE_GATE_UP_STREAM8_PM_GFX1151,
                "kernels/qwen4_moe_gate_up_stream8_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1151,
                "fn,T=8193",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4",
                nt4,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1151,
                "fn,T=8192",
                "down",
                "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4",
                nt4,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
            ),
            (
                Arch::Gfx1100,
                "fn,T=4096",
                "gate_up",
                "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100",
                k::QWEN4_MOE_IU4_SYM_PM_GFX1100,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1100.hxaco",
            ),
            (
                Arch::Gfx1100,
                "fn,T=4096",
                "down",
                "qwen4_moe_down_iu4_sym_pm_gfx1100",
                k::QWEN4_MOE_IU4_SYM_PM_GFX1100,
                "kernels/qwen4_moe_iu4_sym_pm_gfx1100.hxaco",
            ),
        ];
        for (arch, shape, epi, symbol, image, artifact) in cases {
            let w = on(shape, &[("epi", epi)], arch).unwrap();
            assert_eq!(w.arch(), arch.name());
            let b = w.baseline().unwrap();
            assert_eq!(b.launch_symbol, symbol, "{} {shape} {epi}", arch.name());
            assert!(b.exports.contains(&symbol.to_owned()));
            assert_eq!(b.source["symbol"], symbol);
            assert_eq!(b.source["artifact"], artifact);
            assert_eq!(b.desc["artifact"], artifact);
            assert_eq!(b.source["sha256"], sha256_hex(image));
            assert_eq!(b.bytes, image);
            assert!(b.source["source"].as_str().unwrap().contains("gemm.rs:"));
            assert!(b.source["const"]
                .as_str()
                .unwrap()
                .contains(&arch.name().to_uppercase()));
            assert!(!b.source.to_string().contains("GFX1201"), "{}", b.source);
            assert_eq!(w.candidate_knobs()["arch"], arch.name());
            assert!(w.requires_baseline_identity());
        }
    }

    /// Candidate emit and the R1 contract (`check_export` + `validate_invocation` inside
    /// `Problem::kernarg`) succeed for every shipped entry of gfx1151 and gfx1100, including
    /// the gfx1151 row repeats whose grid x is `M / (64 * rr)`.
    #[test]
    fn gfx11_candidates_emit_and_satisfy_the_contract() {
        let cases: [(Arch, &[&str]); 2] = [
            (
                Arch::Gfx1151,
                &[
                    "gate_up",
                    "gate_up_nt4",
                    "gate_up_stream8",
                    "down",
                    "down_nt4",
                    "down_nt4r2",
                    "down_nt4r4",
                ],
            ),
            (Arch::Gfx1100, &["gate_up", "down"]),
        ];
        for (arch, epis) in cases {
            for &epi in epis {
                let e = Entry::parse(epi, arch).unwrap();
                let object = emit_object(e, json!({})).unwrap();
                assert_eq!(object.launch_symbol, e.symbol(), "{} {epi}", arch.name());
                crate::iu4::abi_signature(&object.bytes, &object.launch_symbol).unwrap();
                assert!(find_export(abi_arch(arch).unwrap(), &e.symbol()).is_some());
                let p = tiny_on(e.spec.kind, arch);
                let l = p.launch_for(e, fake_ptrs(&p, false)).unwrap();
                assert_eq!(
                    l.grid[0],
                    (p.m / (64 * usize::from(e.spec.rr))) as u32,
                    "{} {epi}",
                    arch.name()
                );
                assert_eq!(l.block[0], e.spec.threads());
            }
        }
    }

    /// `--cand-object` (gfx1100): the emitted object round-trips (identical to the generator, the
    /// same launch symbol/exports as an emitted candidate); garbage, a wrong-kind object and a
    /// non-gfx1100 arch are refused before any launch.
    #[test]
    fn cand_object_is_checked_against_the_in_tree_emit() {
        let dir = std::env::temp_dir().join(format!("pm_loop_cand_object_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let p = dir.join(name);
            std::fs::write(&p, bytes).unwrap();
            p
        };
        let build = |epi: &str, path: &std::path::Path, arch: Arch| {
            new_with_cand_object(
                "fn,T=64",
                &mut knobs(&[("epi", epi)]),
                None,
                arch,
                Some(path),
            )
        };
        for epi in ["gate_up", "down"] {
            let e = Entry::parse(epi, Arch::Gfx1100).unwrap();
            let emitted = emit_object(e, json!({})).unwrap();
            let p = write(&format!("{epi}.co"), &emitted.bytes);
            let c = build(epi, &p, Arch::Gfx1100).unwrap().candidate().unwrap();
            assert_eq!(c.bytes, emitted.bytes, "{epi}");
            assert_eq!(c.launch_symbol, emitted.launch_symbol, "{epi}");
            assert_eq!(c.exports, emitted.exports, "{epi}");
            assert_eq!(
                c.desc["cand_object"]["identical_to_generator"], true,
                "{epi}"
            );
            // The other kind's object does not expose this entry's symbol.
            let other = if epi == "down" { "gate_up" } else { "down" };
            let o = emit_object(Entry::parse(other, Arch::Gfx1100).unwrap(), json!({})).unwrap();
            let po = write(&format!("{epi}_wrong.co"), &o.bytes);
            let e = build(epi, &po, Arch::Gfx1100)
                .unwrap()
                .candidate()
                .unwrap_err();
            assert!(
                e.contains("--cand-object") && e.contains(&emitted.launch_symbol),
                "{e}"
            );
        }
        let junk = write("junk.co", b"not an elf");
        let e = build("gate_up", &junk, Arch::Gfx1100)
            .unwrap()
            .candidate()
            .unwrap_err();
        assert!(e.contains("--cand-object"), "{e}");
        let e = build("gate_up", &dir.join("missing.co"), Arch::Gfx1100)
            .err()
            .unwrap();
        assert!(
            e.contains("--cand-object") && e.contains("missing.co"),
            "{e}"
        );
        let e = build("gate_up", &junk, Arch::Gfx1151).err().unwrap();
        assert!(e.contains("gfx1100 and gfx1201 only"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An entry an arch has no shipped object for is refused naming the entry and the arch; so
    /// are gfx1151 NT8 (`Spec::validate` verbatim) and an arch without a contract.
    #[test]
    fn entries_missing_on_an_arch_are_refused_by_name() {
        let refused = |arch: Arch, epi: &str| {
            on("fn,T=64", &[("epi", epi)], arch)
                .err()
                .unwrap_or_else(|| panic!("{epi} accepted on {}", arch.name()))
        };
        for epi in ["down_gp_nt4", "down_gp_nt8"] {
            for arch in [Arch::Gfx1151, Arch::Gfx1100] {
                let e = refused(arch, epi);
                assert!(
                    e.contains(epi) && e.contains("gfx1201 only") && e.contains(arch.name()),
                    "{e}"
                );
            }
        }
        for epi in ["gate_up_nt4", "down_nt4", "gate_up_nt8", "down_nt4r2"] {
            let e = refused(Arch::Gfx1100, epi);
            assert!(
                e.contains(epi) && e.contains("gfx1100") && e.contains("NT1"),
                "{e}"
            );
        }
        for epi in ["gate_up_nt8", "down_nt8"] {
            let e = refused(Arch::Gfx1151, epi);
            assert!(e.contains("tile width 8"), "{e}");
        }
        for arch in [Arch::Gfx1201, Arch::Gfx1100] {
            let e = refused(arch, "gate_up_stream8");
            assert!(e.contains("gfx1151 only") && e.contains(arch.name()), "{e}");
        }
        // `--base-epi` follows the same grammar.
        let e = on(
            "fn,T=64",
            &[("epi", "down"), ("base-epi", "down_gp_nt4")],
            Arch::Gfx1151,
        )
        .err()
        .unwrap();
        assert!(
            e.starts_with("--base-epi:") && e.contains("gfx1201 only"),
            "{e}"
        );
        // Spec::validate refuses gfx1100 NT>1 on emit; the adapter refuses earlier, by name.
        let e = on("fn,T=64", &[], Arch::Gfx942).err().unwrap();
        assert!(e.contains("gfx942"), "{e}");
        // gfx1201 is unchanged: the stream8 / row-repeat spellings are not gfx1201 entries.
        assert!(on("fn,T=64", &[("epi", "down_nt4r2")], Arch::Gfx1201)
            .unwrap()
            .candidate()
            .is_err());
    }

    fn cand_object_on(epi: &str, shape: &str, tag: &str, bytes: &[u8]) -> Result<Object> {
        let p =
            std::env::temp_dir().join(format!("pm_loop_candobj_{}_{tag}.co", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        let mut k = knobs(&[("epi", epi)]);
        let w = new_with_cand_object(shape, &mut k, None, Arch::Gfx1201, Some(&p));
        let r = w.and_then(|w| w.candidate());
        let _ = std::fs::remove_file(&p);
        r
    }

    #[test]
    fn gfx1201_cand_object_accepts_the_shipped_hxaco_for_every_flat_epi() {
        let image = rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201;
        for epi in [
            "gate_up",
            "gate_up_nt4",
            "gate_up_nt8",
            "down",
            "down_nt4",
            "down_nt8",
        ] {
            let o =
                cand_object_on(epi, "T=64", epi, image).unwrap_or_else(|e| panic!("{epi}: {e}"));
            let c = &o.desc["cand_object"];
            assert_eq!(c["co_sha256"], sha256_hex(image), "{epi}");
            assert_eq!(c["abi_reference"], "shipped baseline image", "{epi}");
            assert_eq!(c["text_sha256"].as_str().unwrap().len(), 64, "{epi}");
            assert_eq!(c["symbol"], o.launch_symbol, "{epi}");
        }
    }

    #[test]
    fn gfx1201_cand_object_accepts_the_shipped_gp_hxaco() {
        let image = rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201;
        for epi in ["down_gp_nt4"] {
            let o =
                cand_object_on(epi, "T=64", epi, image).unwrap_or_else(|e| panic!("{epi}: {e}"));
            assert_eq!(o.desc["cand_object"]["co_sha256"], sha256_hex(image));
        }
    }

    #[test]
    fn gfx1201_cand_object_refuses_a_missing_symbol_and_non_objects() {
        // The gp image has no flat gate_up_nt8 export.
        let e = cand_object_on(
            "gate_up_nt8",
            "T=64",
            "missing",
            rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201,
        )
        .err()
        .unwrap();
        assert!(e.contains("not a code object exposing"), "{e}");
        let e = cand_object_on("gate_up_nt8", "T=64", "junk", b"not an object")
            .err()
            .unwrap();
        assert!(e.contains("--cand-object"), "{e}");
    }

    #[test]
    fn abi_diff_field_names_the_differing_field() {
        let a = "symbol=s args=[A, B] kernarg_bytes=64 kernarg_align=8 lds_bytes=0 max_flat_workgroup_size=256 wavefront_size=32 workgroup_processor_mode=false";
        assert_eq!(abi_diff_field(a, a), "unknown");
        assert_eq!(
            abi_diff_field(a, &a.replace("args=[A, B]", "args=[A]")),
            "args"
        );
        assert_eq!(
            abi_diff_field(a, &a.replace("kernarg_bytes=64", "kernarg_bytes=72")),
            "kernarg_bytes"
        );
        assert_eq!(
            abi_diff_field(a, &a.replace("size=256", "size=128")),
            "max_flat_workgroup_size"
        );
    }

    /// The pre-lockstep shipped object (a full hxaco from before NT8 lockstep).
    #[test]
    fn gfx1201_cand_object_pre_lockstep_object() {
        let p = "/home/kaden/qcal/release-0.4.3/pm-krsi/land/moe-route/obj/baseline.0eea1673a5308805.co";
        let Ok(bytes) = std::fs::read(p) else {
            eprintln!("pre-lockstep object absent: {p}");
            return;
        };
        let o = cand_object_on("gate_up_nt8", "T=64", "prelock", &bytes)
            .unwrap_or_else(|e| panic!("pre-lockstep gate_up_nt8: {e}"));
        eprintln!("pre-lockstep cand_object: {}", o.desc["cand_object"]);
    }
}
