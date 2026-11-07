// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! A8 (int8 per-128 activation, symmetric-IU4 weight) variant of the Qwen4 Flash-Next EP owner-routed
//! symmetric MoE prefill chain on exact gfx1201.
//!
//! The A4 chain ([`Gpu::qwen4_moe_rotate256_i4`] -> [`Gpu::gemm_qwen4_moe_gate_up_silu_iu4_sym`] ->
//! [`Gpu::qwen4_moe_rotate128_i4_gp`] -> [`Gpu::gemm_qwen4_moe_down_iu4_sym`]) quantizes both MoE
//! activations to `block_i4_128`; this module is the same chain with `block_i8_128` activations
//! (`d = RN(amax / 127)`, `q = clamp(rint(RN(x / d)), -127, 127)`) and signed-IU8 WMMA. Routing, grouping,
//! tile-id / sorted-slot contracts and the BF16 grouped outputs are those of the A4 launchers, so the
//! owner-routed combine reads the same bytes:
//!
//! * gate producer: [`Gpu::qwen4_moe_rotate256_i8`] = FWHT256 + A8 emit (`mq_rotate_x_i8_gfx12`, the
//!   `mq_rotate_x_i4.hip` body), sidecar `[K/128, rows]` in the shared int8 scratch.
//! * gate/up: [`Gpu::gemm_qwen4_moe_gate_up_silu_a8_sym`], `Y [grouped_rows, M/2]` BF16 bits of
//!   `rt(silu(rt(g)) * rt(u))`, padding slots +0, sentinel tiles untouched.
//! * down producer: [`Gpu::qwen4_moe_rotate128_i8`], grouped BF16 SwiGLU rows -> FWHT128 -> A8, flat-slot
//!   sidecar `[K/128, total_slots]`.
//! * down: [`Gpu::gemm_qwen4_moe_down_a8_sym`], `Y [grouped_rows, M]` BF16 bits of `rt(sum)`.
//!
//! Per output and ascending K128 epoch the GEMMs fold the exact int32 chain `C_h` as
//! `acc = fmaf(RN(f32(sc_h) * d_h), (float)C_h, acc)` from +0. Both weight layouts are served: the shipped
//! qt44 / qt53 groups and the F1 drop-zero planar `MQ4G256V2P` / `MQ4G128V2P` ([`Gpu::mq4_experts_planar`]).
//!
//! Developer env `HIPFIRE_QWEN4_MOE_EP_A8` (default on only for EP prefill forwards below 512 rows, `=0` off, `=1` on everywhere; exact gfx1201 only): when the chain runs,
//! [`Gpu::qwen4_moe_ep_a8_requested`] routes it through these kernels, any row count the chain serves.
//!
//! The same kernels also serve the Flash-Next EP2 decode / verify MoE of 9..=64 source rows on the trunk
//! shard's partial-FFN route (developer env `HIPFIRE_QWEN4_MOE_GU_A8`, default on, `=0` off, `=2` unmasked;
//! exact gfx1201): [`Gpu::qwen4_moe_gu_a8_applies`] is the launcher half of that route's predicate, the
//! dispatch stage predicate (`ep_a8_wide`, `qt44_qt53_prefill.rs`) the per-call half. By default that
//! route owner-masks its decode tiles ([`Gpu::qwen4_moe_gu_a8_masked`]).

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::gemm::moe_expert_table_len;
use crate::kernels;
use crate::scratch::Int8MmqPrepared;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;

/// Developer switch of the A8 EP chain. Unset: ON only inside an expert-parallel prefill forward
/// BELOW the owner-routed floor ([`Gpu::qwen4_ep_a8_scope`]: the sub-512-row prompts on the
/// lowered-floor tile route, where it is quality-neutral vs the exact route); every >= 512-row
/// forward and the single-card chain keep A4 (A8 measured +4.4% pp4096 prefill time there with no
/// measurable KLD gain). Exact `1`: on in every forward the chain serves; exact `0`: the kill
/// switch (the A4 chain everywhere).
pub const QWEN4_MOE_EP_A8_FLAG: &str = "HIPFIRE_QWEN4_MOE_EP_A8";

/// Developer switch of the wide-verify int8 routed-MoE chain of the Qwen4 Flash-Next EP2 decode /
/// verify path (exact gfx1201, source rows in the g32 window [`Gpu::moe_wide_verify_rows`]: 9..=64): this
/// module's A8 gate/up + down kernels and the BF16-row grouped combine replace the g32 gate/up, the
/// unscatter + rotation, the PM A8 down, the down unscatter and the expanded top-10 combine
/// ([`Gpu::qwen4_moe_gu_a8_applies`]). Default ON (unset / exact `1`): the decode tiles are owner-masked
/// (both A8 GEMMs skip the experts another rank owns, the combine reads only owned rows; see
/// [`Gpu::qwen4_moe_gu_a8_masked`]); exact `2`: the same chain over UNMASKED tiles (non-owned experts on the
/// zero dummy tables), byte-identical to the masked route and kept for the byte A/B; exact `0` restores the
/// incumbent chain, byte-identical.
pub const QWEN4_MOE_GU_A8_FLAG: &str = "HIPFIRE_QWEN4_MOE_GU_A8";

/// Module of the gate/up and down GEMM entries.
pub const EP_A8_GEMM_MODULE: &str = "qwen4_moe_ep_a8_gfx12";
/// Module and entry of the down A8 producer.
pub const EP_A8_ROTATE128_MODULE: &str = "qwen4_moe_rotate128_i8_gfx12";
pub const EP_A8_ROTATE128_SYMBOL: &str = "qwen4_moe_rotate128_i8";

/// `block_i8_128` bytes (`float d; int s; signed char qs[128]`).
const BLOCK_I8_128_BYTES: usize = 136;
/// Gate/up rows per CTA: 4 waves x 16 gate rows (the up rows ride along).
const ROWS_PER_CTA: usize = 64;

/// Developer override of the tile-run width (`1` or `2`; unset or anything else: [`ep_a8_select_nt`]).
pub const QWEN4_MOE_EP_A8_NT_ENV: &str = "HIPFIRE_QWEN4_MOE_EP_A8_NT";

static EP_A8: LazyLock<Option<bool>> = LazyLock::new(|| {
    match hipfire_config::developer_var(QWEN4_MOE_EP_A8_FLAG)
        .as_deref()
        .map(str::trim)
    {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    }
});
/// `HIPFIRE_QWEN4_MOE_GU_A8`, parsed once: exact `1` / `0`, anything else (unset and `2` included) is `None`
/// (the default: on). Same parse as [`EP_A8`]; the `2` (unmasked) half is [`GU_A8_UNMASKED`].
static GU_A8: LazyLock<Option<bool>> = LazyLock::new(|| {
    match hipfire_config::developer_var(QWEN4_MOE_GU_A8_FLAG)
        .as_deref()
        .map(str::trim)
    {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    }
});
/// `HIPFIRE_QWEN4_MOE_GU_A8=2`, parsed once: the wide-verify chain runs over unmasked tiles.
static GU_A8_UNMASKED: LazyLock<bool> = LazyLock::new(|| {
    gu_a8_unmasked(
        hipfire_config::developer_var(QWEN4_MOE_GU_A8_FLAG)
            .ok()
            .as_deref(),
    )
});
static EP_A8_NT: LazyLock<Option<usize>> = LazyLock::new(|| {
    hipfire_config::developer_var(QWEN4_MOE_EP_A8_NT_ENV)
        .ok()
        .and_then(|v| ep_a8_parse_nt(&v))
});

/// Pure parse of [`QWEN4_MOE_EP_A8_NT_ENV`]: a built width, else `None`.
fn ep_a8_parse_nt(raw: &str) -> Option<usize> {
    raw.trim()
        .parse::<usize>()
        .ok()
        .filter(|nt| EP_A8_GATE_UP_NTS.contains(nt) && EP_A8_DOWN_NTS.contains(nt))
}

/// Tile-run widths built for gate/up (16-slot tiles per CTA).
pub const EP_A8_GATE_UP_NTS: [usize; 2] = [1, 2];
/// Tile-run widths built for down (`nt4` spills the VGPR file and is not built).
pub const EP_A8_DOWN_NTS: [usize; 2] = [1, 2];

/// Symbol of the gate/up entry (`planar`: F1 drop-zero planar experts).
pub fn ep_a8_gate_up_symbol(planar: bool, nt: usize) -> Option<&'static str> {
    Some(match (planar, nt) {
        (false, 1) => "qwen4_moe_ep_a8_gate_up_nt1",
        (false, 2) => "qwen4_moe_ep_a8_gate_up_nt2",
        (true, 1) => "qwen4_moe_ep_a8_gate_up_planar_nt1",
        (true, 2) => "qwen4_moe_ep_a8_gate_up_planar_nt2",
        _ => return None,
    })
}

/// Symbol of the down entry (`planar`: F1 drop-zero planar experts).
pub fn ep_a8_down_symbol(planar: bool, nt: usize) -> Option<&'static str> {
    Some(match (planar, nt) {
        (false, 1) => "qwen4_moe_ep_a8_down_nt1",
        (false, 2) => "qwen4_moe_ep_a8_down_nt2",
        (true, 1) => "qwen4_moe_ep_a8_down_planar_nt1",
        (true, 2) => "qwen4_moe_ep_a8_down_planar_nt2",
        _ => return None,
    })
}

/// Tile-run width of a launch of `grouped_rows` grouped rows: one 16-slot tile per CTA below two tiles'
/// worth of work, two tiles per CTA (the weight stream is read once per pair) from there.
pub fn ep_a8_select_nt(_down: bool, grouped_rows: usize) -> usize {
    if grouped_rows >= 32 {
        2
    } else {
        1
    }
}

/// `k` the down sidecar is reserved with: the shared int8 scratch reserves whole K256 multiples, the down
/// width (`k % 128 == 0`) rounds up (the sidecar uses `k / 128` of the `k_reserved / 128` epochs).
pub fn ep_a8_down_reserve_k(k: usize) -> usize {
    k.next_multiple_of(256)
}

/// Pure launch-shape admission of the gate/up GEMM: `m` splits into gate and up halves of whole 64-row
/// CTAs, `k % 256 == 0` (qt44 groups of 256), whole 16-slot tiles.
pub fn ep_a8_gate_up_shape_ok(m: usize, k: usize, grouped_rows: usize, x_row_div: usize) -> bool {
    m != 0
        && m % (2 * ROWS_PER_CTA) == 0
        && k != 0
        && k % 256 == 0
        && grouped_rows != 0
        && grouped_rows % 16 == 0
        && x_row_div != 0
}

/// Pure launch-shape admission of the down GEMM: `m % 64 == 0`, `k % 128 == 0` (one K128 epoch per
/// qt53 group), whole 16-slot tiles.
pub fn ep_a8_down_shape_ok(m: usize, k: usize, grouped_rows: usize, x_row_div: usize) -> bool {
    m != 0
        && m % ROWS_PER_CTA == 0
        && k != 0
        && k % 128 == 0
        && grouped_rows != 0
        && grouped_rows % 16 == 0
        && x_row_div != 0
}

/// Pure form of [`Gpu::qwen4_moe_ep_a8_requested`].
fn ep_a8_requested(arch: &str, flag: Option<bool>, ep_scope: bool) -> bool {
    arch == "gfx1201" && flag.unwrap_or(ep_scope)
}

/// Pure form of [`Gpu::qwen4_moe_gu_a8_applies`]: exact gfx1201, the switch not `0`, an eager launch
/// (no graph capture, no replay record), and `rows` source rows in the g32 verify window.
fn gu_a8_applies(arch: &str, flag: Option<bool>, rows: usize, eager: bool) -> bool {
    gu_a8_enabled(arch, flag) && eager && Gpu::moe_wide_verify_rows(rows)
}

/// The switch half of [`gu_a8_applies`]: exact gfx1201 and not killed. Also the load-time condition
/// of the int8 scratch reservation ([`Gpu::reserve_qwen4_moe_ep_a8`]).
fn gu_a8_enabled(arch: &str, flag: Option<bool>) -> bool {
    arch == "gfx1201" && flag != Some(false)
}

/// Whether `HIPFIRE_QWEN4_MOE_GU_A8` is the exact value `2`: the wide-verify chain over unmasked tiles.
fn gu_a8_unmasked(raw: Option<&str>) -> bool {
    raw.map(str::trim) == Some("2")
}

/// Pure form of [`Gpu::qwen4_moe_gu_a8_masked`]: the switch half ([`gu_a8_enabled`]) and not the exact
/// value `2`.
fn gu_a8_masked(arch: &str, flag: Option<bool>, unmasked: bool) -> bool {
    gu_a8_enabled(arch, flag) && !unmasked
}

/// `(k, n)` int8-scratch reservations of a chain of up to `max_rows` tokens: the gate sidecar and the down
/// sidecar ([`ep_a8_down_reserve_k`] rounding), `n` saturating at `usize::MAX`.
pub fn ep_a8_reserve_shapes(
    max_rows: usize,
    k_top: usize,
    hidden: usize,
    intermediate: usize,
) -> [(usize, usize); 2] {
    [
        (hidden, max_rows),
        (
            ep_a8_down_reserve_k(intermediate),
            max_rows.saturating_mul(k_top),
        ),
    ]
}

fn err(what: &str) -> HipError {
    HipError::new(0, what)
}

fn first_launch_receipt(stage: usize, line: impl FnOnce() -> String) {
    static SEEN: [AtomicBool; 2] = [AtomicBool::new(false), AtomicBool::new(false)];
    if !SEEN[stage].swap(true, Ordering::Relaxed) {
        eprintln!("[INFO] qwen4 moe EP A8: first launch {}", line());
    }
}

impl Gpu {
    /// Whether the Flash-Next EP2 decode / verify MoE of `rows` source rows runs the int8 gate/up + down
    /// chain instead of the g32 + PM A8 down chain (`HIPFIRE_QWEN4_MOE_GU_A8`, default on, exact `0`
    /// kills; exact gfx1201; `rows` in 9..=[`crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS`], the g32
    /// window; eager only, like [`Gpu::qwen4_moe_ep_a8_requested`]: the EP row gather and the
    /// A8 chain's int8 scratch cannot be captured or recorded). Flag read once per process. The
    /// per-call conditions (grouped path, symmetric experts, geometry, no row gather) are the
    /// dispatch stage predicate's.
    pub fn qwen4_moe_gu_a8_applies(&self, rows: usize) -> bool {
        gu_a8_applies(
            &self.arch,
            *GU_A8,
            rows,
            !self.graphs.capture_mode && !self.replay.is_recording(),
        )
    }

    /// Whether the wide-verify int8 chain's switch is on for this device (flag and arch only).
    pub fn qwen4_moe_gu_a8_enabled(&self) -> bool {
        gu_a8_enabled(&self.arch, *GU_A8)
    }

    /// Whether the wide-verify int8 chain owner-masks its decode tiles: the switch is on for this device
    /// ([`Self::qwen4_moe_gu_a8_enabled`]) and `HIPFIRE_QWEN4_MOE_GU_A8` is not the exact value `2` (the
    /// unmasked developer A/B). The masked route is byte-identical to the unmasked one. Flag read once
    /// per process; also the load-time condition of the EP attach's owner table.
    pub fn qwen4_moe_gu_a8_masked(&self) -> bool {
        gu_a8_masked(&self.arch, *GU_A8, *GU_A8_UNMASKED)
    }

    /// One stderr line, once per process, the first time the wide-verify int8 chain launches:
    /// served logs prove the route ran. Per-stage entries are the `[INFO] qwen4 moe EP A8: first launch`
    /// lines and, under `HIPFIRE_CB_PHASES=1`, the `[moe-entry]` receipts of the A8 GEMMs.
    pub fn qwen4_moe_gu_a8_receipt(&self, rows: usize, grouped_rows: usize, masked: bool) {
        static SEEN: AtomicBool = AtomicBool::new(false);
        if !SEEN.swap(true, Ordering::Relaxed) {
            let tiles = if masked { "owner-masked" } else { "unmasked" };
            eprintln!(
                "[INFO] qwen4 moe GU A8: first wide-verify launch rows={rows} grouped_rows={grouped_rows} \
                 (int8 per-128 gate/up + down, {tiles} tiles, BF16 grouped combine; {QWEN4_MOE_GU_A8_FLAG}=0 restores the g32 + PM A8 down chain)"
            );
        }
    }
}

impl Gpu {
    /// Whether the EP owner-routed symmetric MoE prefill chain runs its A8 kernels
    /// (default on, `HIPFIRE_QWEN4_MOE_EP_A8=0` kills it; exact gfx1201, eager: never while a graph is captured or a replay
    /// recorded, where a lazy int8-scratch growth would invalidate the capture and the A4 chain's
    /// pre-reserved scratch is the one in force). Flag read once per process.
    pub fn qwen4_moe_ep_a8_requested(&self) -> bool {
        ep_a8_requested(&self.arch, *EP_A8, self.qwen4_ep_a8_scope)
            && !self.graphs.capture_mode
            && !self.replay.is_recording()
    }

    fn qwen4_moe_ep_a8_require(&self, what: &str) -> HipResult<()> {
        if self.arch != "gfx1201" {
            return Err(err(&format!(
                "{what}: the EP A8 chain is exact-gfx1201 only"
            )));
        }
        Ok(())
    }

    /// Grow the shared int8 scratch for A8 chains of up to `max_rows` tokens (`k_top` routed slots each) before
    /// any graph capture, alongside [`Gpu::reserve_qwen4_moe_sym`]'s A4 scratch: the gate sidecar `[hidden / 128,
    /// max_rows]` and the down sidecar `[intermediate / 128, slots]` (the larger one wins). A no-op unless the
    /// switch is on for this device ([`Gpu::qwen4_moe_ep_a8_requested`]'s flag and arch, capture state aside).
    pub fn reserve_qwen4_moe_ep_a8(
        &mut self,
        max_rows: usize,
        k_top: usize,
        hidden: usize,
        intermediate: usize,
    ) -> HipResult<()> {
        // Load-time (no forward scope yet): the caller is the EP load, so the unset flag counts as on.
        // The wide-verify chain (`HIPFIRE_QWEN4_MOE_GU_A8`) reads the same scratch, so it reserves
        // it even when the prefill switch is off.
        if !ep_a8_requested(&self.arch, *EP_A8, true) && !gu_a8_enabled(&self.arch, *GU_A8) {
            return Ok(());
        }
        // The wide-verify window never exceeds the chunk cap, but a smaller `max_rows` must still
        // cover it when the wide-verify chain is on: no lazy growth inside a decode step.
        let max_rows = if gu_a8_enabled(&self.arch, *GU_A8) {
            max_rows.max(crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS)
        } else {
            max_rows
        };
        for (k, n) in ep_a8_reserve_shapes(max_rows, k_top, hidden, intermediate) {
            self.reserve_int8_mmq(k, n)?;
        }
        Ok(())
    }

    /// Gate/up A8 producer of the EP chain: [`Gpu::qwen4_moe_rotate256_i4`]'s FWHT256 with the
    /// `block_i8_128` emit, epoch-major `[k / 128, rows]` in the shared int8 scratch. F32 input only.
    pub fn qwen4_moe_rotate256_i8(
        &mut self,
        x: &GpuTensor,
        k: usize,
        rows: usize,
    ) -> HipResult<Int8MmqPrepared> {
        self.qwen4_moe_ep_a8_require("qwen4_moe_rotate256_i8")?;
        if x.dtype != DType::F32 {
            return Err(err("qwen4_moe_rotate256_i8: F32 input only"));
        }
        crate::moe::require_span(x, &[rows, k, 4], "qwen4_moe_rotate256_i8 activations")?;
        let reservation = self.reserve_int8_mmq(k, rows)?;
        self.rotate_x_mq_i8_batched(x, None, reservation, k, rows)
    }

    /// Down A8 producer of the EP chain: grouped BF16 SwiGLU rows `h [grouped_rows, k]` -> FWHT128 -> A8,
    /// the compact flat-slot sidecar `Xq[group * total_slots + sorted[p]]` ([`Gpu::qwen4_moe_rotate128_i4`]
    /// layout with `block_i8_128` records). Rows with `sorted[p] < 0` emit nothing.
    pub fn qwen4_moe_rotate128_i8(
        &mut self,
        h: &GpuTensor,
        sorted: &GpuTensor,
        k: usize,
        grouped_rows: usize,
        total_slots: usize,
    ) -> HipResult<Int8MmqPrepared> {
        self.qwen4_moe_ep_a8_require("qwen4_moe_rotate128_i8")?;
        if k == 0 || k % 128 != 0 || grouped_rows == 0 || total_slots == 0 {
            return Err(err(
                "qwen4_moe_rotate128_i8: need k % 128 == 0 and nonzero rows",
            ));
        }
        crate::moe::require_span(
            h,
            &[grouped_rows, k, 2],
            "qwen4_moe_rotate128_i8 grouped BF16 rows",
        )?;
        crate::moe::require_span(
            sorted,
            &[grouped_rows, 4],
            "qwen4_moe_rotate128_i8 sorted slots",
        )?;
        if grouped_rows.max(total_slots).max(k) > (i32::MAX as usize) / 2 {
            return Err(err("qwen4_moe_rotate128_i8: extent exceeds 32-bit indices"));
        }
        let grid_x = u32::try_from((k / 128) * grouped_rows)
            .map_err(|_| err("qwen4_moe_rotate128_i8: grid exceeds u32"))?;
        self.bind_thread()?;
        self.ensure_mq_signs_128()?;
        self.ensure_kernel(
            EP_A8_ROTATE128_MODULE,
            kernels::QWEN4_MOE_ROTATE128_I8_SRC,
            EP_A8_ROTATE128_SYMBOL,
        )?;
        let reservation = self.reserve_int8_mmq(ep_a8_down_reserve_k(k), total_slots)?;
        let hp = h.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let xp = reservation.ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let (kv, pv, lv) = (k as i32, grouped_rows as i32, total_slots as i32);
        let mut params = [
            &hp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &pv as *const _ as *mut c_void,
            &lv as *const _ as *mut c_void,
        ];
        let timer = crate::profile::begin_timer(
            &self.hip,
            "fwht",
            EP_A8_ROTATE128_SYMBOL,
            grouped_rows * (k * 2 + 4) + k / 128 * total_slots * BLOCK_I8_128_BYTES,
        );
        let result = self.launch_maybe_blob(
            EP_A8_ROTATE128_SYMBOL,
            [grid_x, 1, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(hp);
                b.push_ptr(sp);
                b.push_ptr(xp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(kv);
                b.push_i32(pv);
                b.push_i32(lv);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result?;
        Ok(Int8MmqPrepared::from_reservation(reservation))
    }

    /// Grouped A8 gate/up with the SwiGLU epilogue: `y` `[grouped_rows, m/2]` BF16 bits of
    /// `rt(silu(rt(g)) * rt(u))`, padding rows +0, sentinel tiles untouched. `xq` is the
    /// [`Gpu::qwen4_moe_rotate256_i8`] handle of `x_src_rows` tokens; `x_row_div` = top-k. Weights are the
    /// shipped qt44 groups, or the planar `MQ4G256V2P` layout when [`Gpu::mq4_experts_planar`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qwen4_moe_gate_up_silu_a8_sym(
        &mut self,
        ptrs: &GpuTensor,
        tiles: &GpuTensor,
        sorted: &GpuTensor,
        xq: &Int8MmqPrepared,
        y: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        if !ep_a8_gate_up_shape_ok(m, k, grouped_rows, x_row_div) {
            return Err(err(
                "gemm_qwen4_moe_gate_up_silu_a8_sym: need m % 128 == 0, k % 256 == 0, grouped_rows % 16 == 0",
            ));
        }
        crate::moe::require_span(
            y,
            &[grouped_rows, m / 2, 2],
            "gemm_qwen4_moe_gate_up_silu_a8_sym BF16 grouped rows",
        )?;
        let (generation, live) = self.scratch.int8_mmq_live();
        let xp = xq.checked_ptr(generation, live, k, x_src_rows)?;
        self.qwen4_moe_ep_a8_gemm(
            false,
            ptrs,
            tiles,
            sorted,
            xp,
            y,
            [m, k, x_row_div, grouped_rows, x_src_rows],
        )
    }

    /// Grouped A8 down: `y` `[grouped_rows, m]` BF16 bits of `rt(sum)` per grouped row (no residual, no
    /// combine), padding rows +0, sentinel tiles untouched. `xq` is the [`Gpu::qwen4_moe_rotate128_i8`] handle
    /// of `x_src_rows` flat slots; `x_row_div` = 1. Weights are the shipped qt53 groups, or the planar
    /// `MQ4G128V2P` layout when [`Gpu::mq4_experts_planar`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_qwen4_moe_down_a8_sym(
        &mut self,
        ptrs: &GpuTensor,
        tiles: &GpuTensor,
        sorted: &GpuTensor,
        xq: &Int8MmqPrepared,
        y: &GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
    ) -> HipResult<()> {
        if !ep_a8_down_shape_ok(m, k, grouped_rows, x_row_div) {
            return Err(err(
                "gemm_qwen4_moe_down_a8_sym: need m % 64 == 0, k % 128 == 0, grouped_rows % 16 == 0",
            ));
        }
        crate::moe::require_span(
            y,
            &[grouped_rows, m, 2],
            "gemm_qwen4_moe_down_a8_sym BF16 grouped rows",
        )?;
        let (generation, live) = self.scratch.int8_mmq_live();
        let xp = xq.checked_ptr(generation, live, ep_a8_down_reserve_k(k), x_src_rows)?;
        self.qwen4_moe_ep_a8_gemm(
            true,
            ptrs,
            tiles,
            sorted,
            xp,
            y,
            [m, k, x_row_div, grouped_rows, x_src_rows],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn qwen4_moe_ep_a8_gemm(
        &mut self,
        down: bool,
        ptrs: &GpuTensor,
        tiles: &GpuTensor,
        sorted: &GpuTensor,
        xp: *mut c_void,
        y: &GpuTensor,
        [m, k, x_row_div, grouped_rows, x_src_rows]: [usize; 5],
    ) -> HipResult<()> {
        let what = if down {
            "gemm_qwen4_moe_down_a8_sym"
        } else {
            "gemm_qwen4_moe_gate_up_silu_a8_sym"
        };
        self.qwen4_moe_ep_a8_require(what)?;
        self.bind_thread()?;
        crate::moe::require_span(sorted, &[grouped_rows, 4], what)?;
        crate::moe::require_span(tiles, &[grouped_rows / 16, 4], what)?;
        if grouped_rows.max(x_src_rows).max(m).max(k) > (i32::MAX as usize) / 2 {
            return Err(err(&format!("{what}: extent exceeds 32-bit slot indices")));
        }
        let planar = self.mq4_experts_planar;
        let nt = (*EP_A8_NT).unwrap_or_else(|| ep_a8_select_nt(down, grouped_rows));
        let func = if down {
            ep_a8_down_symbol(planar, nt)
        } else {
            ep_a8_gate_up_symbol(planar, nt)
        }
        .ok_or_else(|| err(&format!("{what}: unbuilt tile-run width")))?;
        self.ensure_kernel(EP_A8_GEMM_MODULE, kernels::QWEN4_MOE_EP_A8_SRC, func)?;
        let ep = ptrs.buf.as_ptr();
        let tp = tiles.buf.as_ptr();
        let sp = sorted.buf.as_ptr();
        let yp = y.buf.as_ptr();
        let (mv, kv, dv, mt, xs) = (
            m as i32,
            k as i32,
            x_row_div as i32,
            grouped_rows as i32,
            x_src_rows as i32,
        );
        let nev = moe_expert_table_len(ptrs);
        let mut params: Vec<*mut c_void> = vec![
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &dv as *const _ as *mut c_void,
            &mt as *const _ as *mut c_void,
            &xs as *const _ as *mut c_void,
            &nev as *const _ as *mut c_void,
        ];
        let rows = if down { m } else { m / 2 };
        let grid = [(rows / ROWS_PER_CTA) as u32, (grouped_rows / 16) as u32, 1];
        first_launch_receipt(usize::from(down), || {
            format!(
                "stage={} entry={func} planar={planar} m={m} k={k} grouped_rows={grouped_rows} x_src_rows={x_src_rows}",
                if down { "down" } else { "gate_up" }
            )
        });
        // `HIPFIRE_CB_PHASES=1`: the same `[moe-entry]` receipt the g32 / PM A8 down launchers print,
        // rows = tokens (the down's source rows are flat slots, ten per token).
        crate::moe_entry_log::moe_entry_receipt(
            if down { "down" } else { "gate_up" },
            func,
            if down { x_src_rows / 10 } else { x_src_rows },
        );
        let timer = crate::profile::begin_timer(
            &self.hip,
            "gemm",
            func,
            grouped_rows * (k / 2 + m) + grouped_rows * k / 128 * BLOCK_I8_128_BYTES,
        );
        let result = self.launch_maybe_blob(func, grid, [128, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(ep);
            b.push_ptr(tp);
            b.push_ptr(sp);
            b.push_ptr(xp);
            b.push_ptr(yp);
            b.push_i32(mv);
            b.push_i32(kv);
            b.push_i32(dv);
            b.push_i32(mt);
            b.push_i32(xs);
            b.push_i32(nev);
            b
        });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_have_symbols_and_the_selector_picks_built_widths() {
        for planar in [false, true] {
            for nt in EP_A8_GATE_UP_NTS {
                let s = ep_a8_gate_up_symbol(planar, nt).unwrap();
                assert!(kernels::QWEN4_MOE_EP_A8_SRC.contains(s), "{s}");
                assert_eq!(s.contains("planar"), planar);
            }
            for nt in EP_A8_DOWN_NTS {
                let s = ep_a8_down_symbol(planar, nt).unwrap();
                assert!(kernels::QWEN4_MOE_EP_A8_SRC.contains(s), "{s}");
            }
            assert!(ep_a8_gate_up_symbol(planar, 4).is_none());
            assert!(ep_a8_down_symbol(planar, 4).is_none());
        }
        for down in [false, true] {
            for rows in [16, 32, 48, 64, 4096, 20480] {
                let nt = ep_a8_select_nt(down, rows);
                let built = if down {
                    EP_A8_DOWN_NTS
                } else {
                    EP_A8_GATE_UP_NTS
                };
                assert!(built.contains(&nt), "{down} {rows}");
            }
        }
        assert_eq!(ep_a8_select_nt(false, 16), 1);
        assert_eq!(ep_a8_select_nt(false, 32), 2);
    }

    #[test]
    fn producer_source_has_its_entry() {
        assert!(kernels::QWEN4_MOE_ROTATE128_I8_SRC.contains(EP_A8_ROTATE128_SYMBOL));
        assert!(kernels::QWEN4_MOE_ROTATE128_I8_SRC.contains("quantize_block_i8_128_wave"));
    }

    #[test]
    fn down_sidecar_reserve_rounds_to_k256_and_covers_the_epochs() {
        for k in [128usize, 640, 1280, 2560] {
            let kr = ep_a8_down_reserve_k(k);
            assert_eq!(kr % 256, 0);
            assert!(kr >= k && kr - k < 256);
            assert!(kr / 128 >= k / 128);
        }
        assert_eq!(ep_a8_down_reserve_k(640), 768);
    }

    #[test]
    fn shape_admission_matches_the_flash_next_geometry() {
        // Flash-Next EP: gate/up M = 2 * 640, K = 2560; down M = 2560, K = 640.
        assert!(ep_a8_gate_up_shape_ok(1280, 2560, 4096, 10));
        assert!(ep_a8_down_shape_ok(2560, 640, 4096, 1));
        assert!(!ep_a8_gate_up_shape_ok(1088, 2560, 4096, 10)); // M/2 not a whole 64-row CTA
        assert!(!ep_a8_gate_up_shape_ok(1280, 2688, 4096, 10)); // K % 256
        assert!(!ep_a8_gate_up_shape_ok(1280, 2560, 4090, 10)); // grouped_rows % 16
        assert!(!ep_a8_down_shape_ok(2560, 640, 0, 1));
        assert!(!ep_a8_down_shape_ok(2560, 600, 4096, 1));
        assert!(!ep_a8_down_shape_ok(2560, 640, 4096, 0));
    }

    #[test]
    fn nt_override_accepts_only_built_widths() {
        assert_eq!(ep_a8_parse_nt("1"), Some(1));
        assert_eq!(ep_a8_parse_nt(" 2 "), Some(2));
        for bad in ["0", "3", "4", "", "two", "-1"] {
            assert_eq!(ep_a8_parse_nt(bad), None, "{bad:?}");
        }
        assert_eq!(QWEN4_MOE_EP_A8_NT_ENV, "HIPFIRE_QWEN4_MOE_EP_A8_NT");
    }

    #[test]
    fn reserve_shapes_cover_both_sidecars_in_whole_k256_reservations() {
        // Flash-Next EP: hidden 2560, intermediate 640, top-10, 2048-row chain.
        let [(gk, gn), (dk, dn)] = ep_a8_reserve_shapes(2048, 10, 2560, 640);
        assert_eq!((gk, gn), (2560, 2048));
        assert_eq!((dk, dn), (768, 20480));
        for (k, _) in [(gk, gn), (dk, dn)] {
            assert_eq!(k % 256, 0);
        }
        // the down sidecar (5 K128 epochs x 20480 slots) fits its reservation
        assert!(640 / 128 * 20480 * BLOCK_I8_128_BYTES <= dk / 128 * dn * BLOCK_I8_128_BYTES);
        assert_eq!(
            ep_a8_reserve_shapes(usize::MAX, 10, 2560, 640)[1].1,
            usize::MAX
        );
    }

    #[test]
    fn requested_needs_exact_gfx1201_and_the_flag() {
        // Unset follows the EP scope; exact 1 / 0 override it.
        assert!(ep_a8_requested("gfx1201", None, true));
        assert!(!ep_a8_requested("gfx1201", None, false));
        assert!(ep_a8_requested("gfx1201", Some(true), false));
        assert!(!ep_a8_requested("gfx1201", Some(false), true));
        for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942", ""] {
            assert!(!ep_a8_requested(arch, Some(true), true), "{arch}");
        }
        assert_eq!(QWEN4_MOE_EP_A8_FLAG, "HIPFIRE_QWEN4_MOE_EP_A8");
    }

    /// `HIPFIRE_QWEN4_MOE_GU_A8`: default on (unset / exact `1`), exact `0` kills, exact gfx1201 only,
    /// eager only, and the row window is the g32 window (9..=WIDE_VERIFY_MAX_ROWS), never wider.
    #[test]
    fn gu_a8_window_kill_switch_and_arch_gating() {
        let max = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        for rows in 0..=max + 8 {
            let in_window = (9..=max).contains(&rows);
            // The window is exactly the g32 gate/up family's.
            assert_eq!(
                gu_a8_applies("gfx1201", None, rows, true),
                in_window,
                "default, rows {rows}"
            );
            assert_eq!(
                gu_a8_applies("gfx1201", None, rows, true),
                Gpu::moe32_gu_applies(1, rows, true),
                "g32 window, rows {rows}"
            );
            assert_eq!(gu_a8_applies("gfx1201", Some(true), rows, true), in_window);
            // Exact `0`: the incumbent chain at every row count.
            assert!(
                !gu_a8_applies("gfx1201", Some(false), rows, true),
                "kill, rows {rows}"
            );
            // Graph capture / replay record: never.
            assert!(
                !gu_a8_applies("gfx1201", None, rows, false),
                "eager, rows {rows}"
            );
            for arch in ["gfx1100", "gfx1151", "gfx1200", "gfx942", "gfx1030", ""] {
                assert!(!gu_a8_applies(arch, None, rows, true), "{arch} rows {rows}");
                assert!(
                    !gu_a8_applies(arch, Some(true), rows, true),
                    "{arch} rows {rows}"
                );
            }
        }
        // The row-gather-sized decode and the singleton few-row window keep the incumbent chain.
        for rows in 0..=8 {
            assert!(
                !gu_a8_applies("gfx1201", Some(true), rows, true),
                "rows {rows}"
            );
        }
        assert_eq!(max, 64);
        assert_eq!(QWEN4_MOE_GU_A8_FLAG, "HIPFIRE_QWEN4_MOE_GU_A8");
    }

    /// The load-time scratch reservation follows the switch alone (arch + not killed), whatever the
    /// prefill switch says.
    #[test]
    fn gu_a8_enabled_is_the_switch_half() {
        assert!(gu_a8_enabled("gfx1201", None));
        assert!(gu_a8_enabled("gfx1201", Some(true)));
        assert!(!gu_a8_enabled("gfx1201", Some(false)));
        for arch in ["gfx1100", "gfx1151", "gfx942", ""] {
            assert!(!gu_a8_enabled(arch, None), "{arch}");
        }
    }

    /// `HIPFIRE_QWEN4_MOE_GU_A8`: unset / `1` run the chain owner-masked, exact `2` the same chain over
    /// UNMASKED tiles (the byte A/B), exact `0` off; exact gfx1201 only. `2` is not `1` / `0`, so the
    /// switch half still reads it as "on" (`GU_A8` is `None`).
    #[test]
    fn gu_a8_masked_is_the_default_and_two_is_the_unmasked_ab() {
        assert!(!gu_a8_unmasked(None));
        assert!(!gu_a8_unmasked(Some("")));
        assert!(!gu_a8_unmasked(Some("1")));
        assert!(!gu_a8_unmasked(Some("0")));
        assert!(!gu_a8_unmasked(Some("22")));
        assert!(gu_a8_unmasked(Some("2")));
        assert!(gu_a8_unmasked(Some(" 2 ")));
        // Masked: switch on and not `2`.
        assert!(gu_a8_masked("gfx1201", None, false));
        assert!(gu_a8_masked("gfx1201", Some(true), false));
        // `2` parses to `GU_A8 = None` (on) with the unmasked bit: the chain applies, unmasked.
        assert!(!gu_a8_masked("gfx1201", None, true));
        assert!(gu_a8_applies("gfx1201", None, 16, true));
        // `0`: off, so never masked either.
        assert!(!gu_a8_masked("gfx1201", Some(false), false));
        assert!(!gu_a8_masked("gfx1201", Some(false), true));
        for arch in ["gfx1100", "gfx1151", "gfx942", ""] {
            assert!(!gu_a8_masked(arch, None, false), "{arch}");
        }
    }
}
