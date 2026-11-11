// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Candidate arm of `--kernel qwen4_moe_verify`: the PM symmetric-IU4 grouped routed-expert chain
//! (production's `>= 512`-row prefill route of the qt44/qt53 MoE, `sym_iu4`), forced at the verify
//! row counts (T in {1,2,3,4,7,8}) for Flash-Next (H=2560, I=640, E=512, top10, QT44 gate/up
//! `MQ4G256` + QT53 down `MQ4G128`, symmetric headers). Included from `moe_verify.rs` as
//! `crate::moe_verify_iu4::Iu4Arm` (`#[path = "pm_loop/moe_verify_iu4.rs"] mod moe_verify_iu4;`).
//!
//! # The chain (production call order, read from source; no device run established anything here)
//!
//! `qt44_qt53_prefill.rs` `sym_iu4` branches, issued through the public `rdna_compute::Gpu`
//! launchers with production's literal arguments (`T` rows, `slots = 10 T`, `grouped_rows`
//! below, `host_mapped` where production passes `experts_host_mapped(p)`):
//!
//! | # | launcher | kernel (per arch) | production call |
//! |---|---|---|---|
//! | 1 | `moe_scatter_stable_top10(topk, counts, offsets, sorted, tiles, inverse, slots, E, grouped_rows)` | `qwen4_moe_group_{ranks,prefix,scatter}` (hipcc) | `:1619-1629` |
//! | 2 | `qwen4_moe_rotate256_i4(x_norm, H, T)` -> A4 sidecar | `mq_rotate_x_i4` (hipcc; FWHT256 + C2 `block_i4_128`) | `:1829` |
//! | 3 | `gemm_qwen4_moe_gate_up_silu_iu4_sym(gate_up_ptrs, tiles, sorted, &xq, y_gu, 2I, H, 10, grouped_rows, T, host_mapped)` | PM `..gate_up_silu_iu4_sym_pm_<arch>[_nt4\|_nt8]` | `:1831-1843` |
//! | 4 | `qwen4_moe_rotate128_i4_gp(y_gu, sorted, I, grouped_rows, slots)` | gfx1201 `moe_rotate_i4_gp_pm_gfx1201` (PM plane); gfx1151 `qwen4_moe_rotate128_i4` (hipcc, flat); gfx1100 `mq_i4_a4_r128_k640_gfx1100` (PM, flat) | `:2177-2183` |
//! | 5 | `gemm_qwen4_moe_down_iu4_sym(down_ptrs, tiles, sorted, &xd, y_down, H, I, 1, grouped_rows, slots, host_mapped)` | PM `..down_iu4_sym[_gp]_pm_<arch>[_nt4\|_nt8]` | `:2192-2204` |
//! | 6 | `moe_down_combine_grouped_top10_bf16in(y_down, inverse, topk_idx, topk_w, residual, order, H, grouped_rows, T, false, E)` | `moe_combine_order_top10` + `moe_down_combine_grouped_top10_bf16in` (hipcc) | `:2351-2363` |
//!
//! Symbols are NOT chosen here: setup asks the live `Gpu` selector
//! (`qwen4_moe_sym_gate_up_symbol`, `qwen4_moe_sym_gemm_symbols_wide`,
//! `qwen4_moe_rotate128_i4_gp_selected`) and refuses unless it agrees with the per-arch default
//! plan this module documents and returns from `pm_objects` (an env override such as
//! `HIPFIRE_QWEN4_MOE_SYM_PM=0`, `HIPFIRE_QWEN4_MOE_SYM_DOWN_RR`, `HIPFIRE_MOE_ROTATE_I4_PM=0`,
//! `HIPFIRE_MOE_IU4_WIDE_TILES=0` would otherwise change what is launched but not what is
//! certified). Expected plans: gfx1201 resident gate NT4 / down gp-NT4, host-mapped gate NT8 /
//! down gp-NT8 (gate NT8 also resident once `slots >= 128 E`); gfx1151 NT4/NT4 (rr 1);
//! gfx1100 NT1/NT1 (flat plane, embedded K3 producer, `k == 640`).
//!
//! # `grouped_rows`
//!
//! Production's `path2_m_total` is `grouped_m_total_bound(slots, E)`
//! (`hipfire-arch-qwen4/src/program.rs:391-404`, used at `:1188`): `live = min(slots, E)`,
//! `ceil16(slots + 15 live)`. T=1 -> 160, 2 -> 320, 4 -> 640, 8 -> 1280. It is a bound, not the
//! exact count: trailing sentinel tiles (`tiles[t] == -1`) exit in the GEMMs, and padding rows are
//! `sorted == -1`.
//!
//! # Policy bypassed (the only difference from production's admission)
//!
//! The arm forces the route below its `>= 512`-row floor. That floor is evaluated in three places
//! that this arm does not consult: `sym_iu4_rows_admit` (`gemm.rs:50245`, via
//! `Gpu::qwen4_moe_sym_iu4_applies` `gemm.rs:50604`, floor `QWEN4_MOE_SYM_IU4_MIN_ROWS`
//! `gemm.rs:49222`), `Qwen4MoeLayerRuntime::route_capability` (`gpu_forward.rs:3199-3205`, which
//! only declares `Qt44Qt53GroupedSymmetric` at `rows >= 512`), and the grouped-path selection of
//! the layer (`use_path2`, `grouped_path_for_rows`, `qt44_qt53_prefill.rs:137-143`; how the
//! layer routes T <= 8 before that predicate was not traced here). The launchers have no row
//! floor of their own (checks: `m % 64`, `k % 256|128`,
//! `grouped_rows % 16`, spans, `slots <= 2^22`). Production requests the route by default on gfx1100,
//! gfx1151 and gfx1201 (`default_requested: true`; `HIPFIRE_QWEN4_MOE_SYM_IU4=0` is the kill switch)
//! from `rows >= 512` on: forcing the chain below the floor is fine for a screen and is reported in
//! `stages_json`.
//!
//! # Combine semantics
//!
//! The arm accumulates: `residual += routed` (initial_zero = false: `moe_down_combine_..._bf16in`
//! loads `residual`, accumulates the ten rank-ordered BF16-rounded weighted rows in a BF16
//! accumulator, stores F32; `kernels/src/moe_down_combine_grouped_top10.hip:150-193`).
//! Production on gfx1151/gfx1201 defaults `HIPFIRE_QWEN4_MOE_COMBINE_ZINIT` on
//! (`feature_flags.rs:996-998`), which takes over the preceding zero fill and runs the `_zinit`
//! symbol (write-only target); per the kernel comment the `_zinit` stores are bytewise those of
//! "clear + the accumulating instantiation", so on a zeroed residual this is production's bytes;
//! the cost differs by the residual load only. gfx1100 production keeps the clear + accumulating
//! combine. No extra step is needed.
//!
//! # Not modelled
//!
//! * The fused front `qwen4_moe_front_producer` (production may replace step 2 on gfx1151/gfx1201
//!   when the router shares the F16 X; it writes the same sidecar bytes plus the router's F16 X,
//!   which is outside this chain).
//! * Router, shared expert, the CPU/EP/slab variants of the grouping.
//!
//! # Allocation / sync in `encode`
//!
//! None on the device after `setup`: every grouping/BF16/order buffer is allocated in `setup`;
//! `reserve_qwen4_moe_sym(T, ...)` sizes the launcher-internal scratch (grouping, gate sidecar
//! `int4_mmq_x_scratch`, down sidecar/plane) for this T; the launchers re-`reserve` on every call
//! (generation bump, `grow_scratch_buffer` is a no-op because the size already fits). The
//! `Int4MmqPrepared` handles are host values (pointer, k, n, generation) returned fresh by each
//! producer call; there is no prepared-plane reuse path to take (production re-produces them per
//! layer too). `setup` runs the chain once into a throwaway residual and synchronizes so lazy
//! `ensure_kernel` / `ensure_embedded_kernel` / `ensure_mq_signs*` loads happen outside timing.
//! Host side, every launcher builds a small params `Vec` / `KernargBlob` per launch (as in
//! production). No synchronization is issued (the profiler timers only record when the profiler
//! is active).
//!
//! # Not established from source
//!
//! Device behaviour at T < 512 (never run here), whether the gfx1100 descriptor is present on a
//! given device (`qwen4_moe_sym_kernel()` is checked in `setup`, which refuses otherwise), and the
//! resolved values of the `HIPFIRE_*` flags (read by `Gpu` at construction; `setup` verifies the
//! outcome, not the flags).
#![allow(dead_code)]

use crate::common::{contains, dev_err, sha256_hex};
use crate::moe_verify::ChainInputs;
use crate::workload::{Object, Result};
use peacemaker_isa::kernels::qwen4_moe_down_gp;
use peacemaker_isa::kernels::qwen4_moe_sym::{self, Kind, Spec, WeightCodec};
use peacemaker_isa::Arch as PmArch;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

/// Routed slots per token of the chain (the combine hard-codes ten ranks).
const TOP: usize = 10;
/// QT44 gate/up group bytes per K256 (`qwen4_moe_sym_check` `group_bytes`).
const GATE_GROUP_BYTES: usize = 136;
/// QT53 down group bytes per K128.
const DOWN_GROUP_BYTES: usize = 68;
/// Grouped-row tile (`MOE_GROUPED_BLOCK_M`).
const BLOCK_M: usize = 16;
/// Real slots per expert from which a resident gfx1201 gate/up takes NT8
/// (`QWEN4_MOE_SYM_WIDE_MIN_SLOTS_PER_EXPERT`, `gemm.rs:49301`).
const WIDE_MIN_SLOTS_PER_EXPERT: usize = 128;
/// Builder GEMMs' slot bound (`QWEN4_MOE_SYM_PM_MAX_SLOTS`, `gemm.rs:50357`).
const MAX_SLOTS: usize = 1 << 22;
/// `moe_scatter_stable_top10` expert bound (`gemm.rs:51179`).
const MAX_EXPERTS: usize = 1024;
/// Down `k` of the gfx1100 embedded K3 producer (`slab_moe::SLAB_DOWN_K`).
const GFX1100_DOWN_K: usize = rdna_compute::slab_moe::SLAB_DOWN_K;

const GATE_PRODUCER_SYMBOL: &str = "mq_rotate_x_i4";
const FLAT_DOWN_PRODUCER_SYMBOL: &str = "qwen4_moe_rotate128_i4";
const SCATTER_SYMBOLS: [&str; 3] = [
    "qwen4_moe_group_ranks",
    "qwen4_moe_group_prefix",
    "qwen4_moe_group_scatter",
];
const COMBINE_SYMBOLS: [&str; 2] = [
    "moe_combine_order_top10",
    "moe_down_combine_grouped_top10_bf16in",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Gfx1201,
    Gfx1151,
    Gfx1100,
}

impl Target {
    fn parse(arch: &str) -> Result<Self> {
        match arch {
            "gfx1201" => Ok(Self::Gfx1201),
            "gfx1151" => Ok(Self::Gfx1151),
            "gfx1100" => Ok(Self::Gfx1100),
            other => Err(format!(
                "qwen4_moe_verify IU4 arm: arch `{other}` has no symmetric-IU4 MoE descriptor \
                 (embedded PM GEMM bundles and A4 producers exist for gfx1201, gfx1151 and \
                 gfx1100 only: crates/rdna-compute/src/gemm.rs:50037-50126 MOE_SYM_IU4_KERNELS)"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gfx1201 => "gfx1201",
            Self::Gfx1151 => "gfx1151",
            Self::Gfx1100 => "gfx1100",
        }
    }

    fn pm(self) -> PmArch {
        match self {
            Self::Gfx1201 => PmArch::Gfx1201,
            Self::Gfx1151 => PmArch::Gfx1151,
            Self::Gfx1100 => PmArch::Gfx1100,
        }
    }

    /// Descriptor row of `MOE_SYM_IU4_KERNELS`.
    fn descriptor_lines(self) -> &'static str {
        match self {
            Self::Gfx1100 => "crates/rdna-compute/src/gemm.rs:50038-50071",
            Self::Gfx1151 => "crates/rdna-compute/src/gemm.rs:50072-50100",
            Self::Gfx1201 => "crates/rdna-compute/src/gemm.rs:50101-50125",
        }
    }
}

/// The `Spec` symbol of an MQ4 symmetric entry (`kind`, tile width `nt`, row repeat 1).
fn sym(target: Target, kind: Kind, nt: u8) -> String {
    Spec {
        arch: target.pm(),
        kind,
        nt,
        rr: 1,
        codec: WeightCodec::Mq4Sym,
    }
    .symbol()
}

/// The grouped-position twin of a down entry (`qwen4_moe_down_gp_entry`, `gemm.rs:49236-49242`;
/// `qwen4_moe_down_gp::symbols`).
fn gp_twin(symbol: &str) -> String {
    symbol.replacen("_sym_pm_", "_sym_gp_pm_", 1)
}

/// Production's grouped row bound for `slots` routed slots over `experts` experts:
/// `hipfire-arch-qwen4/src/program.rs:391-404 grouped_m_total_bound` (block 16).
pub fn grouped_rows(slots: usize, experts: usize) -> Result<usize> {
    let live = slots.min(experts);
    let padded = live
        .checked_mul(BLOCK_M - 1)
        .and_then(|pad| slots.checked_add(pad))
        .ok_or_else(|| "grouped row bound overflows".to_owned())?;
    Ok(padded.div_ceil(BLOCK_M) * BLOCK_M)
}

/// What the chain launches on one arch/residency/shape: the two GEMM entries and the down plane.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Plan {
    /// Gate/up GEMM entry.
    gate: String,
    /// Down GEMM entry as launched (the `_gp_` twin on a grouped plane).
    down: String,
    /// The down A4 producer writes the grouped-position plane.
    gp: bool,
    /// Down A4 producer kernel.
    down_producer: &'static str,
}

impl Plan {
    fn down_producer_origin(&self) -> &'static str {
        if self.down_producer == FLAT_DOWN_PRODUCER_SYMBOL {
            "hip"
        } else {
            "pm"
        }
    }
}

/// The per-arch default plan (no env overrides).
fn expected_plan(target: Target, host_mapped: bool, slots: usize, experts: usize) -> Plan {
    match target {
        Target::Gfx1201 => {
            // Host-mapped experts take NT8 for both GEMMs; resident widens only gate/up, from
            // 128 real slots per table expert (gemm.rs:50704-50745, :49301).
            let wide = slots >= WIDE_MIN_SLOTS_PER_EXPERT.saturating_mul(experts);
            let gate_nt = if host_mapped || wide { 8 } else { 4 };
            let down_nt = if host_mapped { 8 } else { 4 };
            Plan {
                gate: sym(target, Kind::GateUp, gate_nt),
                down: gp_twin(&sym(target, Kind::Down, down_nt)),
                gp: true,
                down_producer: rdna_compute::moe_rotate_i4_pm::MOE_ROTATE_I4_GP_PM_GFX1201_SYMBOL,
            }
        }
        Target::Gfx1151 => Plan {
            gate: sym(target, Kind::GateUp, 4),
            down: sym(target, Kind::Down, 4),
            gp: false,
            down_producer: FLAT_DOWN_PRODUCER_SYMBOL,
        },
        Target::Gfx1100 => Plan {
            gate: sym(target, Kind::GateUp, 1),
            down: sym(target, Kind::Down, 1),
            gp: false,
            down_producer: rdna_compute::slab_moe::MQ_I4_A4_GFX1100_SYMBOL,
        },
    }
}

/// GPU-free admission of the chain's launch extents (the launchers re-check; this refuses early
/// with the shape named).
fn validate_dims(
    target: Target,
    t: usize,
    h: usize,
    i: usize,
    e: usize,
    top: usize,
) -> Result<usize> {
    let slots = t
        .checked_mul(top)
        .ok_or_else(|| "T * top overflows".to_owned())?;
    let bad = |what: &str| Err(format!("qwen4_moe_verify IU4 arm: {what}"));
    if top != TOP {
        return bad(&format!(
            "top = {top}: the combine and the gate/up x_row_div are the top-10 route"
        ));
    }
    if t == 0 || slots > MAX_SLOTS {
        return bad(&format!(
            "T = {t} (slots {slots}): need 1 <= slots <= 2^22 (QWEN4_MOE_SYM_PM_MAX_SLOTS)"
        ));
    }
    if e == 0 || e > MAX_EXPERTS {
        return bad(&format!("E = {e}: need 0 < E <= 1024 (scatter bound)"));
    }
    if h == 0 || h % 256 != 0 {
        return bad(&format!(
            "H = {h}: gate/up k and the A4 FWHT256 need H % 256 == 0"
        ));
    }
    if i == 0 || i % 128 != 0 || (2 * i) % 64 != 0 {
        return bad(&format!(
            "I = {i}: down k / FWHT128 need I % 128 == 0 (and 2I % 64 == 0)"
        ));
    }
    if target == Target::Gfx1100 && i != GFX1100_DOWN_K {
        return bad(&format!(
            "I = {i}: the gfx1100 embedded K3 down producer is k == {GFX1100_DOWN_K} only \
             (`mq_i4_a4_r128_k640_gfx1100`)"
        ));
    }
    Ok(slots)
}

/// Device state of the arm: its own grouping/BF16/order buffers (the launcher-internal scratch
/// lives in the `Gpu`) and the plan the live selector confirmed.
struct State {
    t: usize,
    h: usize,
    i: usize,
    e: usize,
    host_mapped: bool,
    slots: usize,
    grouped_rows: usize,
    counts: GpuTensor,
    offsets: GpuTensor,
    sorted: GpuTensor,
    tiles: GpuTensor,
    inverse: GpuTensor,
    y_gate_up: GpuTensor,
    y_down: GpuTensor,
    order: GpuTensor,
    plan: Plan,
    /// `Gpu::qwen4_moe_sym_iu4_requested()` at setup (informational: production's request policy).
    production_requested: bool,
}

pub struct Iu4Arm {
    target: Target,
    state: Option<State>,
}

fn hip<T>(stage: &str, r: rdna_compute::HipResult<T>) -> Result<T> {
    r.map_err(|e| format!("{stage}: {}", dev_err(e)))
}

/// Bytes of the unreachable-from-source constants a record needs: `needle\0` is how a symbol
/// appears in a bundle's string table.
fn bundle_has(bytes: &[u8], symbol: &str) -> bool {
    let mut needle = symbol.as_bytes().to_vec();
    needle.push(0);
    contains(bytes, &needle)
}

impl Iu4Arm {
    /// GPU-free: `Err` naming the arch when it has no symmetric-IU4 MoE descriptor.
    pub fn new(arch: &'static str) -> Result<Self> {
        Ok(Self {
            target: Target::parse(arch)?,
            state: None,
        })
    }

    /// Allocate every buffer, size the launcher scratch, verify the selector, prime the chain.
    pub fn setup(&mut self, gpu: &mut Gpu, inp: &ChainInputs) -> Result<()> {
        if self.state.is_some() {
            return Err("qwen4_moe_verify IU4 arm: setup called twice".into());
        }
        let target = self.target;
        if inp.arch != target.name() || gpu.arch != target.name() {
            return Err(format!(
                "qwen4_moe_verify IU4 arm: built for {}, inputs say {}, device is {}",
                target.name(),
                inp.arch,
                gpu.arch
            ));
        }
        let (t, h, i, e) = (inp.t, inp.h, inp.i, inp.e);
        let slots = validate_dims(target, t, h, i, e, inp.top)?;
        if gpu.qwen4_moe_sym_kernel().is_none() {
            return Err(format!(
                "qwen4_moe_verify IU4 arm: {} has no symmetric-IU4 MoE descriptor on this device \
                 (MoeSymIu4Kernel requires IU4 WMMA and the K2 inventory row, gemm.rs:50135-50150)",
                target.name()
            ));
        }
        if !gpu.iu4_a4_c2_candidates_exact() {
            return Err(
                "qwen4_moe_verify IU4 arm: the A4 producers and K2 kernels need exactly \
                 -DIU4_A4_CANDIDATES=2 (gemm.rs:50209-50235 sym_iu4_request refuses otherwise)"
                    .into(),
            );
        }
        let grouped = grouped_rows(slots, e)?;

        // Weights must be symmetric for the sym-IU4 kernels (load-time gate in production,
        // gpu_forward.rs:4654-4725): gate/up QT44 [2I x H], down QT53 [H x I].
        if !hip(
            "qwen4_moe_sym_check(gate_up)",
            gpu.qwen4_moe_sym_check(&inp.gate_up_ptrs, 2 * i, h, e, GATE_GROUP_BYTES),
        )? {
            return Err(
                "qwen4_moe_verify IU4 arm: gate/up expert headers are not symmetric \
                        (zp != -8 * sc) or not finite"
                    .into(),
            );
        }
        if !hip(
            "qwen4_moe_sym_check(down)",
            gpu.qwen4_moe_sym_check(&inp.down_ptrs, h, i, e, DOWN_GROUP_BYTES),
        )? {
            return Err(
                "qwen4_moe_verify IU4 arm: down expert headers are not symmetric \
                        (zp != -8 * sc) or not finite"
                    .into(),
            );
        }

        // Launcher-internal scratch for exactly this T (grouping, gate sidecar, down sidecar and
        // grouped plane): nothing grows inside encode.
        hip(
            "reserve_qwen4_moe_sym",
            gpu.reserve_qwen4_moe_sym(t, TOP, e, h, i),
        )?;

        let raw =
            |gpu: &mut Gpu, what: &str, bytes: usize| hip(what, gpu.zeros(&[bytes], DType::Raw));
        let counts = raw(gpu, "alloc counts", e * 4)?;
        let offsets = raw(gpu, "alloc offsets", (e + 1) * 4)?;
        let sorted = raw(gpu, "alloc sorted_slot_index", grouped * 4)?;
        let tiles = raw(gpu, "alloc expert_tile_ids", grouped / BLOCK_M * 4)?;
        let inverse = raw(gpu, "alloc inverse_perm", slots * 4)?;
        // `moe_combine_order_top10` writes an int2 per route slot.
        let order = raw(gpu, "alloc combine order", slots * 8)?;
        let y_gate_up = hip(
            "alloc y_gate_up_grouped",
            gpu.zeros(&[grouped, i], DType::BF16),
        )?;
        let y_down = hip(
            "alloc y_down_grouped",
            gpu.zeros(&[grouped, h], DType::BF16),
        )?;

        // The live selector must agree with the documented default plan.
        let expected = expected_plan(target, inp.host_mapped, slots, e);
        let observed = {
            let gate = gpu
                .qwen4_moe_sym_gate_up_symbol(inp.host_mapped, [2 * i, h, TOP, t, e])
                .ok_or("qwen4_moe_sym_gate_up_symbol: no descriptor")?;
            let [_, down] = gpu
                .qwen4_moe_sym_gemm_symbols_wide(inp.host_mapped, slots, e)
                .ok_or("qwen4_moe_sym_gemm_symbols_wide: no descriptor")?;
            let gp = gpu.qwen4_moe_rotate128_i4_gp_selected(i, grouped, slots);
            Plan {
                gate: gate.to_owned(),
                down: if gp { gp_twin(down) } else { down.to_owned() },
                gp,
                down_producer: expected.down_producer,
            }
        };
        if observed != expected {
            return Err(format!(
                "qwen4_moe_verify IU4 arm: the live selector picks {observed:?} but the default \
                 {} plan certified and documented here is {expected:?} (an environment override \
                 such as HIPFIRE_QWEN4_MOE_SYM_PM / _SYM_DOWN_RR / MOE_ROTATE_I4_PM / \
                 MOE_IU4_WIDE_TILES is set); unset it",
                target.name()
            ));
        }

        let state = State {
            t,
            h,
            i,
            e,
            host_mapped: inp.host_mapped,
            slots,
            grouped_rows: grouped,
            counts,
            offsets,
            sorted,
            tiles,
            inverse,
            y_gate_up,
            y_down,
            order,
            plan: observed,
            production_requested: gpu.qwen4_moe_sym_iu4_requested(),
        };

        // Prime: one throwaway chain so module loads, sign tables and scratch binding happen
        // here, outside any timed interval.
        let prime = hip("alloc prime residual", gpu.zeros(&[t * h], DType::F32))?;
        run_chain(&state, gpu, inp, &prime)?;
        hip("prime synchronize", gpu.hip.device_synchronize())?;
        hip("free prime residual", gpu.free_tensor(prime))?;

        self.state = Some(state);
        Ok(())
    }

    /// Enqueue the chain: `residual += routed MoE output` (F32 `[T, H]`). No allocation, upload
    /// or synchronization.
    pub fn encode(&mut self, gpu: &mut Gpu, inp: &ChainInputs, residual: &GpuTensor) -> Result<()> {
        let state = self
            .state
            .as_ref()
            .ok_or("qwen4_moe_verify IU4 arm: encode before setup")?;
        run_chain(state, gpu, inp, residual)
    }

    /// `{stages: [{stage, launcher, symbols, origin, source, args}], forced_below_policy_floor,
    /// ...}`: before `setup` the default resident T-generic plan, after it the confirmed plan with
    /// literal arguments and scratch bytes.
    pub fn stages_json(&self) -> Value {
        let state = self.state.as_ref();
        let plan = state
            .map(|s| s.plan.clone())
            .unwrap_or_else(|| expected_plan(self.target, false, TOP, 512));
        let dim = |f: fn(&State) -> usize, name: &str| {
            state.map_or_else(|| name.to_owned(), |s| f(s).to_string())
        };
        let t = dim(|s| s.t, "T");
        let h = dim(|s| s.h, "H");
        let i = dim(|s| s.i, "I");
        let e = dim(|s| s.e, "E");
        let slots = dim(|s| s.slots, "10*T");
        let grouped = dim(|s| s.grouped_rows, "grouped_rows");
        let host = state.map_or_else(|| "host_mapped".to_owned(), |s| s.host_mapped.to_string());
        let down_producer_launcher = if plan.gp {
            "Gpu::qwen4_moe_rotate128_i4_gp (selected: grouped-position plane, \
             Gpu::qwen4_moe_rotate128_i4_gp_pm)"
        } else {
            "Gpu::qwen4_moe_rotate128_i4_gp (selected: falls through to Gpu::qwen4_moe_rotate128_i4, flat-slot sidecar)"
        };
        let gate_origin = "pm";
        let stages = vec![
            json!({
                "stage": "scatter",
                "launcher": "Gpu::moe_scatter_stable_top10",
                "symbols": SCATTER_SYMBOLS,
                "origin": "hip",
                "source": "crates/rdna-compute/src/gemm.rs:51101-51128 (three-launch body :51170-51325); \
                           production crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:1619-1629",
                "args": format!(
                    "topk_idx[{t},10], counts[{e}], offsets[{e}+1], sorted[{grouped}], \
                     tiles[{grouped}/16], inverse[{slots}], total_slots={slots}, experts={e}, \
                     grouped_rows={grouped}"
                ),
            }),
            json!({
                "stage": "gate_up_producer_a4",
                "launcher": "Gpu::qwen4_moe_rotate256_i4",
                "symbols": [GATE_PRODUCER_SYMBOL],
                "origin": "hip",
                "source": "crates/rdna-compute/src/gemm.rs:51330-51359 -> gemv.rs:6880-6987 \
                           rotate_x_mq_i4_batched (hipcc module mq_rotate_x_i4, FWHT256 + C2 \
                           block_i4_128); production qt44_qt53_prefill.rs:1829",
                "args": format!("x_norm F32[{t},{h}], k={h}, rows={t} -> Int4MmqPrepared (int4_mmq_x_scratch)"),
            }),
            json!({
                "stage": "gate_up_gemm_silu",
                "launcher": "Gpu::gemm_qwen4_moe_gate_up_silu_iu4_sym",
                "symbols": [plan.gate],
                "origin": gate_origin,
                "source": format!(
                    "crates/rdna-compute/src/gemm.rs:51771-51811 -> qwen4_moe_sym_gemm :52361-52503 \
                     (NT1 :52067-52142); entry rule :50704-50745 and :50779-50794; descriptor row {}; \
                     production qt44_qt53_prefill.rs:1831-1843",
                    self.target.descriptor_lines()
                ),
                "args": format!(
                    "gate_up_ptrs, tiles, sorted, &xq, y_gate_up[{grouped},{i}] BF16, m=2*{i}, k={h}, \
                     x_row_div=10, grouped_rows={grouped}, x_src_rows={t}, host_mapped={host}"
                ),
            }),
            json!({
                "stage": "down_producer_a4",
                "launcher": down_producer_launcher,
                "symbols": [plan.down_producer],
                "origin": plan.down_producer_origin(),
                "source": "crates/rdna-compute/src/moe_rotate_i4_pm.rs:237-345 (gp dispatch / gp_pm), \
                           gemm.rs:51522-51631 (flat), slab_moe.rs:1819-1910 (gfx1100 embedded K3); \
                           production qt44_qt53_prefill.rs:2177-2183",
                "args": format!(
                    "y_gate_up, sorted, k={i}, grouped_rows={grouped}, total_slots={slots} \
                     -> Int4MmqPrepared (down sidecar{})",
                    if plan.gp { ", grouped plane P=grouped_rows" } else { ", flat [k/128, total_slots]" }
                ),
            }),
            json!({
                "stage": "down_gemm",
                "launcher": "Gpu::gemm_qwen4_moe_down_iu4_sym",
                "symbols": [plan.down],
                "origin": "pm",
                "source": format!(
                    "crates/rdna-compute/src/gemm.rs:51819-51873 -> qwen4_moe_sym_gemm :52361-52503 \
                     (gp twin :49236-49258, :52433, :52465); descriptor row {}; production \
                     qt44_qt53_prefill.rs:2192-2204",
                    self.target.descriptor_lines()
                ),
                "args": format!(
                    "down_ptrs, tiles, sorted, &xd, y_down[{grouped},{h}] BF16, m={h}, k={i}, \
                     x_row_div=1, grouped_rows={grouped}, x_src_rows={slots}, host_mapped={host}"
                ),
            }),
            json!({
                "stage": "combine",
                "launcher": "Gpu::moe_down_combine_grouped_top10_bf16in",
                "symbols": COMBINE_SYMBOLS,
                "origin": "hip",
                "source": "crates/rdna-compute/src/moe.rs:2315-2422; kernels/src/moe_down_combine_grouped_top10.hip:97-134,150-205; \
                           production qt44_qt53_prefill.rs:2351-2363",
                "args": format!(
                    "y_down, inverse, topk_idx, topk_w, residual F32[{t},{h}], order[{slots}*8 B], \
                     hidden={h}, grouped_rows={grouped}, tokens={t}, initial_zero=false \
                     (accumulate: residual += routed), n_experts={e}"
                ),
            }),
        ];
        let mut out = json!({
            "route": "sym_iu4 grouped prefill chain (qt44/qt53, A4 activations)",
            "arch": self.target.name(),
            "stages": stages,
            "forced_below_policy_floor": {
                "policy": format!(
                    "sym_iu4_rows_admit rows>={}",
                    rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS
                ),
                "source": "crates/rdna-compute/src/gemm.rs:50245 (via Gpu::qwen4_moe_sym_iu4_applies \
                           :50604; floor QWEN4_MOE_SYM_IU4_MIN_ROWS :49222)",
                "also_not_consulted": [
                    {
                        "predicate": "Qwen4MoeLayerRuntime::route_capability: Qt44Qt53GroupedSymmetric only at rows>=512",
                        "source": "crates/hipfire-arch-qwen4/src/gpu_forward.rs:3199-3205",
                    },
                    {
                        "predicate": "grouped path selection (use_path2, grouped_path_for_rows) of the layer; how production routes T<=8 there was not traced here",
                        "source": "crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:137-143",
                    },
                    {
                        "predicate": "sym_iu4 composite (route policy capability, bf16 recipe, F32 x_norm, down_k == mi, applies)",
                        "source": "crates/hipfire-dispatch/src/pipeline/qt44_qt53_prefill.rs:1573-1585",
                    },
                ],
                "launcher_checks_kept": "m % 64, k % 256 (gate/up) | k % 128 (down), grouped_rows % 16, \
                                         spans, slots <= 2^22, header symmetry (qwen4_moe_sym_check at setup)",
                "default_requested_note": "production default_requested=true on gfx1100, gfx1151 and gfx1201 \
                                           (HIPFIRE_QWEN4_MOE_SYM_IU4=0 kills the route): the chain runs for \
                                           rows>=512; forced here below the floor for a screen",
            },
            "combine_mode": {
                "initial_zero": false,
                "meaning": "residual += routed (accumulating symbol); production gfx1151/gfx1201 default runs the \
                            _zinit symbol over a cleared target (feature_flags.rs:996-998, moe_program.rs:433-460), \
                            bytewise the same on a zeroed residual (moe_down_combine_grouped_top10.hip:142-147)",
            },
            "not_modelled": [
                "fused front producer qwen4_moe_front_producer (gemm.rs:51380; same sidecar bytes + router F16 X)",
                "router, shared expert, EP/slab/CPU-split variants",
            ],
            "encode_allocations": {
                "device": "none after setup (buffers + reserve_qwen4_moe_sym in setup; launchers re-reserve with no growth); \
                           setup primes the chain once so lazy module loads happen outside timing",
                "host": "small params Vec / KernargBlob per launch as in production; Int4MmqPrepared handles are host values",
                "sync": "none",
            },
            "selector_checked_at_setup": [
                "Gpu::qwen4_moe_sym_gate_up_symbol",
                "Gpu::qwen4_moe_sym_gemm_symbols_wide",
                "Gpu::qwen4_moe_rotate128_i4_gp_selected",
            ],
            "plan": {
                "gate_up": plan.gate,
                "down": plan.down,
                "grouped_position_plane": plan.gp,
                "down_producer": plan.down_producer,
            },
        });
        if let Some(s) = state {
            let plane_rows = rdna_compute::scratch::qwen4_grouped_rows_bound(s.slots, s.e)
                .unwrap_or(s.slots)
                .max(s.slots);
            out["residency"] = json!(if s.host_mapped { "host_mapped" } else { "vram" });
            out["production_request_policy_resolved"] = json!(s.production_requested);
            out["grouped_rows"] = json!(s.grouped_rows);
            out["scratch_bytes"] = json!({
                "own": {
                    "counts": s.e * 4,
                    "offsets": (s.e + 1) * 4,
                    "sorted_slot_index": s.grouped_rows * 4,
                    "expert_tile_ids": s.grouped_rows / BLOCK_M * 4,
                    "inverse_perm": s.slots * 4,
                    "y_gate_up_grouped_bf16": s.grouped_rows * s.i * 2,
                    "y_down_grouped_bf16": s.grouped_rows * s.h * 2,
                    "combine_order": s.slots * 8,
                },
                "launcher_reserved": {
                    "group": rdna_compute::scratch::qwen4_moe_group_layout(s.slots, s.e).bytes,
                    "gate_sidecar": s.h / 128 * s.t * 72,
                    "down_sidecar_plane": s.i / 128 * plane_rows * 72,
                },
            });
        }
        out
    }

    /// The embedded PM code objects the chain launches on this arch (bytes from
    /// `rdna_compute::kernels::*`; `exports` = every kernel symbol of the bundle, verified
    /// present in the bytes; `launch_symbol` = the symbol this chain launches there).
    ///
    /// `launch_symbol` and `source.launched` follow the plan `setup` confirmed with the live
    /// selector (residency and `T` aware); before `setup` they are the resident, small-`T`
    /// default plan. Every NT variant of a bundle is an export either way, so the certified set
    /// does not depend on it. `gfx1151` and `gfx1100` have PM GEMM entries but only `gfx1201`
    /// (grouped-position plane) and `gfx1100` (K3 `mq_i4_a4_r128_k640_gfx1100`) have a PM down
    /// A4 producer; the gate/up producer and the other launchers are hipcc kernels (no object).
    pub fn pm_objects(&self) -> Result<Vec<Object>> {
        let target = self.target;
        let plan = self
            .state
            .as_ref()
            .map(|s| s.plan.clone())
            .unwrap_or_else(|| expected_plan(target, false, TOP, 512));
        let mut objects = Vec::new();
        match target {
            Target::Gfx1201 => {
                let exports: Vec<String> = qwen4_moe_sym::module_specs(PmArch::Gfx1201)
                    .into_iter()
                    .map(Spec::symbol)
                    .collect();
                objects.push(bundle(
                    rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201,
                    "kernels/qwen4_moe_iu4_sym_pm_gfx1201.hxaco",
                    "rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201 (crates/rdna-compute/src/kernels.rs:4467)",
                    &format!("{} MoeSymIu4Kernel gfx1201 row (image, entries)", target.descriptor_lines()),
                    exports,
                    &plan.gate,
                    vec![plan.gate.clone()],
                )?);
                objects.push(bundle(
                    rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201,
                    "kernels/qwen4_moe_down_gp_pm_gfx1201.hxaco",
                    "rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201 (crates/rdna-compute/src/kernels.rs:10783)",
                    "crates/rdna-compute/src/gemm.rs:49231-49264 (gp entries/image), :52433,:52465 (launch), \
                     moe_rotate_i4_pm.rs:253-345 (gp producer)",
                    qwen4_moe_down_gp::symbols(PmArch::Gfx1201).to_vec(),
                    &plan.down,
                    vec![plan.down_producer.to_owned(), plan.down.clone()],
                )?);
            }
            Target::Gfx1151 => {
                let exports: Vec<String> = qwen4_moe_sym::module_specs(PmArch::Gfx1151)
                    .into_iter()
                    .map(Spec::symbol)
                    .collect();
                objects.push(bundle(
                    rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151,
                    "kernels/qwen4_moe_iu4_sym_pm_gfx1151.hxaco",
                    "rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151 (crates/rdna-compute/src/kernels.rs:4465)",
                    &format!("{} MoeSymIu4Kernel gfx1151 row (image, entries; native image selected by HIPFIRE_QWEN4_MOE_SYM_PM default)", target.descriptor_lines()),
                    exports,
                    &plan.gate,
                    vec![plan.gate.clone(), plan.down.clone()],
                )?);
            }
            Target::Gfx1100 => {
                let gate = sym(target, Kind::GateUp, 1);
                let down = sym(target, Kind::Down, 1);
                // `_acc` qualification twins are in the bundle (kernels.rs:4488-4494) and are
                // never launched by the route; every bundle export is certified.
                let exports = vec![
                    gate.clone(),
                    format!("{gate}_acc"),
                    down.clone(),
                    format!("{down}_acc"),
                ];
                objects.push(bundle(
                    rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1100,
                    "kernels/qwen4_moe_iu4_sym_pm_gfx1100.hxaco",
                    "rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1100 (crates/rdna-compute/src/kernels.rs:4495)",
                    &format!("{} MoeSymIu4Kernel gfx1100 row (NT1 entries; default_requested=false)", target.descriptor_lines()),
                    exports,
                    &plan.gate,
                    vec![plan.gate.clone(), plan.down.clone()],
                )?);
                objects.push(bundle(
                    rdna_compute::kernels::MQ_I4_A4_PM_GFX1100,
                    "kernels/mq_i4_a4_pm_gfx1100.hxaco",
                    "rdna_compute::kernels::MQ_I4_A4_PM_GFX1100 (crates/rdna-compute/src/kernels.rs:4977)",
                    "crates/rdna-compute/src/gemm.rs:50063-50068 (MoeSymIu4DownProducer::Embedded), \
                     slab_moe.rs:1819-1910 qwen4_moe_rotate128_i4_embedded",
                    vec![plan.down_producer.to_owned()],
                    plan.down_producer,
                    vec![plan.down_producer.to_owned()],
                )?);
            }
        }
        Ok(objects)
    }

    /// The recipe: A4 activations (NUMERIC against the F32 few-row chain), BF16 grouped
    /// intermediates, F32 residual.
    pub fn numeric_contract(&self) -> Value {
        json!({
            "recipe": "sym_iu4_grouped_prefill",
            "activations": "A4: gate/up input = FWHT256 rotation then per-128 K-group `block_i4_128` \
                            (72 B: f32 d, i32 s, 64 B signed nibbles) with the frozen C2 candidate set \
                            {5,7} (-DIU4_A4_CANDIDATES=2) (mq_rotate_x_i4); down input = the grouped BF16 \
                            SwiGLU rows rotated FWHT128 then the same per-128 A4 quantisation (once per \
                            routed slot)",
            "weights": "unchanged QT44 gate/up (MQ4G256, 136 B per K256) and QT53 down (MQ4G128, 68 B per \
                        K128); symmetric headers (zp == -8 * sc), read as w = sc * (u - 8)",
            "accumulate": "per K128 epoch exact int32 dot of the signed weight and activation nibbles, then \
                           sum = fma(RN(sc * d), float(C), sum) from +0 (examples/qwen4_moe_sym.rs fold_reference)",
            "gate_up_epilogue": "BF16 grouped rows = bf16_rne(silu(rt(g)) * rt(u)), rt = BF16 round trip; \
                                 padding rows +0, sentinel tiles untouched",
            "down_output": "BF16 per grouped row (bf16_rne of the fold), padding rows +0",
            "combine": "F32 residual accumulate in ascending expert-id rank order (moe_combine_order_top10), \
                        BF16-rounded accumulator, BF16-rounded outputs and BF16-rounded routing weights, \
                        F32 store (moe_down_combine_grouped_top10_bf16in, initial_zero=false)",
            "relation_vs_f32_few_row_baseline": "numeric",
            "reference": "cpu_iu4_chain (exact recipe; SiLU portable twin), pm_loop/moe_verify.rs: this \
                          chain's own arithmetic stage by stage; tolerance 0 except where a SwiGLU BF16 \
                          rounding is ambiguous between the portable SiLU twin and the device expf, then \
                          the enumerated alternative roundings' deviation",
            "reason": "A4 (4-bit) activations and BF16 grouped intermediates versus the few-row route's \
                       F32/BF16 recipe; not an AR-row-exact route",
        })
    }
}

/// The six-step chain; shared by `encode` and the priming run of `setup`.
fn run_chain(s: &State, gpu: &mut Gpu, inp: &ChainInputs, residual: &GpuTensor) -> Result<()> {
    if (inp.t, inp.h, inp.i, inp.e, inp.top) != (s.t, s.h, s.i, s.e, TOP)
        || inp.host_mapped != s.host_mapped
    {
        return Err(format!(
            "qwen4_moe_verify IU4 arm: inputs (T={}, H={}, I={}, E={}, top={}, host_mapped={}) differ \
             from setup (T={}, H={}, I={}, E={}, host_mapped={})",
            inp.t, inp.h, inp.i, inp.e, inp.top, inp.host_mapped, s.t, s.h, s.i, s.e, s.host_mapped
        ));
    }
    hip(
        "moe_scatter_stable_top10",
        gpu.moe_scatter_stable_top10(
            &inp.topk_idx,
            &s.counts,
            &s.offsets,
            &s.sorted,
            &s.tiles,
            &s.inverse,
            s.slots,
            s.e,
            s.grouped_rows,
        ),
    )?;
    let xq = hip(
        "qwen4_moe_rotate256_i4",
        gpu.qwen4_moe_rotate256_i4(&inp.x_norm, s.h, s.t),
    )?;
    hip(
        "gemm_qwen4_moe_gate_up_silu_iu4_sym",
        gpu.gemm_qwen4_moe_gate_up_silu_iu4_sym(
            &inp.gate_up_ptrs,
            &s.tiles,
            &s.sorted,
            &xq,
            &s.y_gate_up,
            2 * s.i,
            s.h,
            TOP,
            s.grouped_rows,
            s.t,
            s.host_mapped,
        ),
    )?;
    let xd = hip(
        "qwen4_moe_rotate128_i4_gp",
        gpu.qwen4_moe_rotate128_i4_gp(&s.y_gate_up, &s.sorted, s.i, s.grouped_rows, s.slots),
    )?;
    hip(
        "gemm_qwen4_moe_down_iu4_sym",
        gpu.gemm_qwen4_moe_down_iu4_sym(
            &inp.down_ptrs,
            &s.tiles,
            &s.sorted,
            &xd,
            &s.y_down,
            s.h,
            s.i,
            1,
            s.grouped_rows,
            s.slots,
            s.host_mapped,
        ),
    )?;
    hip(
        "moe_down_combine_grouped_top10_bf16in",
        gpu.moe_down_combine_grouped_top10_bf16in(
            &s.y_down,
            &s.inverse,
            &inp.topk_idx,
            &inp.topk_w,
            residual,
            &s.order,
            s.h,
            s.grouped_rows,
            s.t,
            false,
            s.e,
        ),
    )
}

/// One embedded PM bundle as an `Object`; `launch` and every export must occur in the bytes.
fn bundle(
    bytes: &'static [u8],
    artifact: &str,
    konst: &str,
    selection: &str,
    exports: Vec<String>,
    launch: &str,
    launched: Vec<String>,
) -> Result<Object> {
    let every = std::iter::once(launch)
        .chain(exports.iter().map(String::as_str))
        .chain(launched.iter().map(String::as_str));
    for symbol in every {
        if !bundle_has(bytes, symbol) {
            return Err(format!(
                "symbol {symbol} is not in the embedded bundle {artifact}"
            ));
        }
    }
    if !exports.iter().any(|e| e == launch) {
        return Err(format!(
            "launch symbol {launch} is not among the exports of {artifact}"
        ));
    }
    Ok(Object {
        bytes: bytes.to_vec(),
        launch_symbol: launch.to_owned(),
        exports,
        desc: json!({"artifact": artifact}),
        source: json!({
            "source": selection,
            "const": konst,
            "artifact": artifact,
            "sha256": sha256_hex(bytes),
            "launched": launched,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accepts_the_three_arches_and_names_the_refused() {
        for arch in ["gfx1201", "gfx1151", "gfx1100"] {
            assert!(Iu4Arm::new(arch).is_ok(), "{arch}");
        }
        for arch in ["gfx942", "gfx1030", "gfx1200", "gfx1201x", ""] {
            let err = Iu4Arm::new(arch).err().expect(arch);
            assert!(err.contains(&format!("`{arch}`")), "{err}");
            assert!(err.contains("gfx1201, gfx1151 and gfx1100"), "{err}");
        }
    }

    #[test]
    fn grouped_rows_is_productions_bound() {
        // program.rs:391-404: ceil16(slots + 15 * min(slots, E)).
        for (t, want) in [(1, 160), (2, 320), (3, 480), (4, 640), (7, 1120), (8, 1280)] {
            assert_eq!(grouped_rows(t * 10, 512).unwrap(), want, "T={t}");
        }
        // Past E slots every expert may carry padding: slots + 15 E.
        assert_eq!(grouped_rows(4096 * 10, 512).unwrap(), 4096 * 10 + 512 * 15);
        assert_eq!(grouped_rows(8192 * 10, 512).unwrap() % BLOCK_M, 0);
        assert!(grouped_rows(usize::MAX, 512).is_err());
    }

    #[test]
    fn dims_admission() {
        for target in [Target::Gfx1201, Target::Gfx1151, Target::Gfx1100] {
            for t in [1, 2, 3, 4, 7, 8] {
                assert_eq!(
                    validate_dims(target, t, 2560, 640, 512, 10).unwrap(),
                    t * 10
                );
            }
            assert!(validate_dims(target, 0, 2560, 640, 512, 10).is_err());
            assert!(validate_dims(target, 4, 2560, 640, 512, 8).is_err());
            assert!(validate_dims(target, 4, 2560, 640, 0, 10).is_err());
            assert!(validate_dims(target, 4, 2560, 640, 1025, 10).is_err());
            assert!(validate_dims(target, 4, 2561, 640, 512, 10).is_err());
            assert!(validate_dims(target, 4, 2560, 600, 512, 10).is_err());
            assert!(validate_dims(target, (1 << 22) / 10 + 1, 2560, 640, 512, 10).is_err());
        }
        // The gfx1100 embedded K3 producer is k == 640 only.
        assert!(validate_dims(Target::Gfx1100, 4, 2048, 512, 256, 10).is_err());
        assert!(validate_dims(Target::Gfx1201, 4, 2048, 512, 256, 10).is_ok());
    }

    #[test]
    fn expected_plans() {
        let p = expected_plan(Target::Gfx1201, false, 40, 512);
        assert_eq!(p.gate, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4");
        assert_eq!(p.down, "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4");
        assert!(p.gp && p.down_producer == "moe_rotate_i4_gp_pm_gfx1201");
        let p = expected_plan(Target::Gfx1201, true, 40, 512);
        assert_eq!(p.gate, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8");
        assert_eq!(p.down, "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt8");
        // Resident gate/up widens at 128 slots per expert; down stays NT4.
        let p = expected_plan(Target::Gfx1201, false, 128 * 512, 512);
        assert_eq!(p.gate, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt8");
        assert_eq!(p.down, "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4");
        for host in [false, true] {
            let p = expected_plan(Target::Gfx1151, host, 40, 512);
            assert_eq!(p.gate, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1151_nt4");
            assert_eq!(p.down, "qwen4_moe_down_iu4_sym_pm_gfx1151_nt4");
            assert!(!p.gp && p.down_producer_origin() == "hip");
            let p = expected_plan(Target::Gfx1100, host, 40, 512);
            assert_eq!(p.gate, "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1100");
            assert_eq!(p.down, "qwen4_moe_down_iu4_sym_pm_gfx1100");
            assert!(!p.gp && p.down_producer_origin() == "pm");
            assert_eq!(p.down_producer, "mq_i4_a4_r128_k640_gfx1100");
        }
    }

    #[test]
    fn pm_objects_per_arch() {
        for (arch, n_objects) in [("gfx1201", 2), ("gfx1151", 1), ("gfx1100", 2)] {
            let objects = Iu4Arm::new(arch).unwrap().pm_objects().unwrap();
            assert_eq!(objects.len(), n_objects, "{arch}");
            for o in &objects {
                assert!(!o.bytes.is_empty(), "{arch}");
                assert!(!o.exports.is_empty(), "{arch}");
                assert!(
                    o.exports.contains(&o.launch_symbol),
                    "{arch}: {}",
                    o.launch_symbol
                );
                for export in &o.exports {
                    assert!(bundle_has(&o.bytes, export), "{arch}: {export}");
                }
                assert!(o.desc["artifact"].as_str().unwrap().ends_with(".hxaco"));
                assert!(o.source["source"].as_str().unwrap().contains(".rs:"));
                for launched in o.source["launched"].as_array().unwrap() {
                    assert!(bundle_has(&o.bytes, launched.as_str().unwrap()), "{arch}");
                }
            }
        }
    }

    #[test]
    fn pm_object_exports_cover_the_documented_plan() {
        let objects = Iu4Arm::new("gfx1201").unwrap().pm_objects().unwrap();
        assert_eq!(
            objects[0].bytes,
            rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1201
        );
        assert_eq!(objects[0].exports.len(), 6);
        assert_eq!(
            objects[0].launch_symbol,
            "qwen4_moe_gate_up_silu_iu4_sym_pm_gfx1201_nt4"
        );
        assert_eq!(
            objects[1].bytes,
            rdna_compute::kernels::QWEN4_MOE_DOWN_GP_PM_GFX1201
        );
        assert_eq!(
            objects[1].launch_symbol,
            "qwen4_moe_down_iu4_sym_gp_pm_gfx1201_nt4"
        );
        assert!(objects[1]
            .exports
            .contains(&"moe_rotate_i4_gp_pm_gfx1201".to_owned()));
        // The host-mapped NT8 entries are exports of the same bundles.
        for host_plan in [expected_plan(Target::Gfx1201, true, 40, 512)] {
            assert!(objects[0].exports.contains(&host_plan.gate));
            assert!(objects[1].exports.contains(&host_plan.down));
        }

        let halo = Iu4Arm::new("gfx1151").unwrap().pm_objects().unwrap();
        assert_eq!(
            halo[0].bytes,
            rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1151
        );
        let plan = expected_plan(Target::Gfx1151, false, 40, 512);
        assert!(halo[0].exports.contains(&plan.gate) && halo[0].exports.contains(&plan.down));

        let navi = Iu4Arm::new("gfx1100").unwrap().pm_objects().unwrap();
        assert_eq!(
            navi[0].bytes,
            rdna_compute::kernels::QWEN4_MOE_IU4_SYM_PM_GFX1100
        );
        assert_eq!(navi[0].exports.len(), 4);
        assert_eq!(navi[1].bytes, rdna_compute::kernels::MQ_I4_A4_PM_GFX1100);
        assert_eq!(navi[1].launch_symbol, "mq_i4_a4_r128_k640_gfx1100");
    }

    #[test]
    fn stages_json_names_every_launcher_and_the_forcing() {
        for arch in ["gfx1201", "gfx1151", "gfx1100"] {
            let v = Iu4Arm::new(arch).unwrap().stages_json();
            let stages = v["stages"].as_array().unwrap();
            let launchers: Vec<&str> = stages
                .iter()
                .map(|s| s["launcher"].as_str().unwrap())
                .collect();
            for want in [
                "Gpu::moe_scatter_stable_top10",
                "Gpu::qwen4_moe_rotate256_i4",
                "Gpu::gemm_qwen4_moe_gate_up_silu_iu4_sym",
                "Gpu::qwen4_moe_rotate128_i4_gp",
                "Gpu::gemm_qwen4_moe_down_iu4_sym",
                "Gpu::moe_down_combine_grouped_top10_bf16in",
            ] {
                assert!(
                    launchers.iter().any(|l| l.starts_with(want)),
                    "{arch}: {want}"
                );
            }
            for s in stages {
                assert!(
                    matches!(s["origin"].as_str(), Some("pm" | "hip")),
                    "{arch}: {s}"
                );
                assert!(
                    s["source"].as_str().unwrap().contains(".rs:"),
                    "{arch}: {s}"
                );
                assert!(!s["symbols"].as_array().unwrap().is_empty());
            }
            assert_eq!(
                v["forced_below_policy_floor"]["policy"],
                "sym_iu4_rows_admit rows>=512"
            );
            assert!(v["forced_below_policy_floor"]["source"]
                .as_str()
                .unwrap()
                .contains("gemm.rs:50245"));
            assert!(v["forced_below_policy_floor"]["default_requested_note"]
                .as_str()
                .unwrap()
                .contains("default_requested=true"));
        }
        let v = Iu4Arm::new("gfx1201").unwrap().stages_json();
        assert_eq!(v["stages"][3]["origin"], "pm");
        let v = Iu4Arm::new("gfx1151").unwrap().stages_json();
        assert_eq!(v["stages"][3]["origin"], "hip");
        assert_eq!(v["stages"][3]["symbols"][0], "qwen4_moe_rotate128_i4");
    }

    #[test]
    fn numeric_contract_describes_the_recipe() {
        let v = Iu4Arm::new("gfx1201").unwrap().numeric_contract();
        assert_eq!(v["relation_vs_f32_few_row_baseline"], "numeric");
        for key in [
            "activations",
            "weights",
            "accumulate",
            "gate_up_epilogue",
            "combine",
        ] {
            assert!(v[key].as_str().unwrap().len() > 20, "{key}");
        }
        assert!(v["activations"].as_str().unwrap().contains("A4"));
        assert!(v["gate_up_epilogue"].as_str().unwrap().contains("BF16"));
    }
}
