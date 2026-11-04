// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Int8-activation routed-expert DOWN route of Qwen4 Flash-Next (qt53 `MQ4G128V2` / planar
//! `MQ4G128V2P`, `M = 2560`, `K = 640`, top-10) on exact gfx1201.
//!
//! The experts are symmetric (`z == -8 s` on every group header, [`Gpu::qwen4_moe_sym_check`]), so
//! `w = s * (q - 8)` and the codes feed the signed-activation dot4 / iu8 WMMA kernels
//! (`kernels/src/qwen4_moe_down_a8.gfx1201.hip`) against `block_i8_128` activations produced from
//! the already rotated F32 rows by `qwen4_moe_down_a8_quant_gfx1201`. Per pick row `r` and output
//! row, every entry folds the same chain: for the five 128-K blocks in increasing order the exact
//! i32 `C`, `acc = fmaf(RN(f32(s_h) * d_x), (float)C, acc)`, `acc` starting at `+0.0f`; `Y = acc`.
//! One definition for both ABIs (D-indexed dot4, D-grouped WMMA), so a pick's bytes do not depend
//! on the row count, the entry or the tile position.
//!
//! Entries, ABIs exactly as the kernel header defines them (raw [`hip_bridge::KernargBlob`]
//! images; no typed `hipfire-kernel-abi` family: registering one needs edits to
//! `families/mod.rs` and `tests/kernel_abi_drift.rs`):
//! * producer `qwen4_moe_down_a8_quant_gfx1201(const f32* X, block_i8_128* Xq, i32 K, i32 rows)`
//!   (24 B), grid `[rows]`, block 160, no LDS. Xq is row-major `[rows][5]`, 680 B per row.
//! * `qwen4_moe_down_a8_indexed[_planar]_gfx1201(const u64* ptrs, const i32* topk,
//!   const block_i8_128* xq, f32* expanded, i32 M, i32 K, i32 N, i32 n_exp, i32 k_top)` (52 B),
//!   grid `[M / 32, k_top, N]`, block 128.
//! * `qwen4_moe_down_a8_grouped[_planar]_gfx1201(const u64* ptrs, const i32* tile_ids,
//!   const i32* sorted_slots, const block_i8_128* xq, f32* Y, i32 M, i32 K, i32 x_row_div,
//!   i32 grouped_rows, i32 x_src_rows, i32 n_exp)` (64 B), grid
//!   `[ceil(M / 64), ceil(grouped_rows / 16)]`, block 128, static LDS 10944.
//!
//! Env (developer): `HIPFIRE_QWEN4_MOE_DOWN_A8` (`developer.qwen4_moe_down_a8`, typed boolean):
//! `1` requests the GROUPED route (the 9..=64-row verify window) on exact gfx1201; unset and `0`
//! (the default: off) keep the incumbent routed down. The window is reached only by the >8-row
//! routes: singleton verify blocks of 9..=64 rows, prefill tails, and wide-class lane groups.
//! `HIPFIRE_QWEN4_MOE_DOWN_A8_INDEXED` (`developer.qwen4_moe_down_a8_indexed`) additionally
//! enables the per-pick dot4 arm (decode / few-row verify), default off. The route
//! additionally needs the loader's symmetric verdict ([`Gpu::set_qwen4_moe_down_a8_verified`]) and
//! a settled MTP-head verdict ([`Gpu::set_qwen4_moe_down_a8_head_pending`]).

use crate::dispatch::Gpu;
use hip_bridge::{HipError, HipResult, KernargBlob};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;

/// Largest token count (route rows) the route serves; both ABIs share one fold at 1..=64.
pub const QWEN4_MOE_DOWN_A8_ROUTE_CAP: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;

/// Down projection width the kernels are written for.
const DOWN_K: usize = 640;
/// Routed picks per token.
const DOWN_K_TOP: usize = 10;
/// Sidecar bytes per X row: five `block_i8_128` of 136 B.
pub const QWEN4_MOE_DOWN_A8_ROW_BYTES: usize = 5 * 136;

const MODULE: &str = "qwen4_moe_down_a8_gfx1201";
const QUANT_ENTRY: &str = "qwen4_moe_down_a8_quant_gfx1201";
const INDEXED_ENTRY: &str = "qwen4_moe_down_a8_indexed_gfx1201";
const INDEXED_PLANAR_ENTRY: &str = "qwen4_moe_down_a8_indexed_planar_gfx1201";
const GROUPED_ENTRY: &str = "qwen4_moe_down_a8_grouped_gfx1201";
const GROUPED_PLANAR_ENTRY: &str = "qwen4_moe_down_a8_grouped_planar_gfx1201";
const GROUPED_NODZ_ENTRY: &str = "qwen4_moe_down_a8_grouped_nodz_gfx1201";
const GROUPED_PLANAR_NODZ_ENTRY: &str = "qwen4_moe_down_a8_grouped_planar_nodz_gfx1201";

/// The grouped entry of a layout: `nodz` selects the twin that stores no dead row (sentinel tiles
/// and pad slots), live rows byte-identical (see the kernel header).
fn grouped_entry(planar: bool, nodz: bool) -> &'static str {
    match (planar, nodz) {
        (false, false) => GROUPED_ENTRY,
        (true, false) => GROUPED_PLANAR_ENTRY,
        (false, true) => GROUPED_NODZ_ENTRY,
        (true, true) => GROUPED_PLANAR_NODZ_ENTRY,
    }
}

/// `block_i8_128` prelude without its standalone quantizer, then the A8 down kernels (the byte
/// oracle `tests/qwen4_moe_down_a8_hw.rs` compiles the same concatenation).
pub const QWEN4_MOE_DOWN_A8_GFX1201_SRC: &str = concat!(
    "#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n",
    include_str!("../../../kernels/src/block_i8_128_quant.hip"),
    include_str!("../../../kernels/src/qwen4_moe_down_a8.gfx1201.hip")
);

/// `HIPFIRE_QWEN4_MOE_DOWN_A8` (`developer.qwen4_moe_down_a8`, typed boolean). Read once.
static QWEN4_MOE_DOWN_A8: LazyLock<Option<bool>> =
    LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_QWEN4_MOE_DOWN_A8"));

/// `HIPFIRE_QWEN4_MOE_DOWN_A8_INDEXED` (`developer.qwen4_moe_down_a8_indexed`, typed boolean):
/// the sub-switch of the per-pick dot4 arm. Read once.
static QWEN4_MOE_DOWN_A8_INDEXED: LazyLock<Option<bool>> =
    LazyLock::new(|| hipfire_config::developer_typed_bool("HIPFIRE_QWEN4_MOE_DOWN_A8_INDEXED"));

/// Pure form of the indexed sub-switch: explicitly on (default OFF).
fn down_a8_indexed_requested(flag: Option<bool>) -> bool {
    flag == Some(true)
}

/// Pure form of [`Gpu::qwen4_moe_down_a8_requested`]: exact gfx1201 and the flag explicitly on
/// (default OFF: `HIPFIRE_QWEN4_MOE_DOWN_A8=1` enables it; unset and `0` are the same).
fn down_a8_requested(arch: &str, flag: Option<bool>) -> bool {
    arch == "gfx1201" && flag == Some(true)
}

/// Pure inputs of [`down_a8_admits`].
#[derive(Clone, Copy, Debug)]
struct DownA8Gate<'a> {
    requested: bool,
    verified: bool,
    head_pending: bool,
    qwen4_scope: bool,
    prefill_invariant_scope: bool,
    recording: bool,
    capture: bool,
    arch: &'a str,
    m: usize,
    k: usize,
    k_top: usize,
}

/// Pure form of [`Gpu::qwen4_moe_down_a8_applies`].
fn down_a8_admits(g: DownA8Gate<'_>) -> bool {
    g.requested
        && g.verified
        && !g.head_pending
        && g.qwen4_scope
        && !g.prefill_invariant_scope
        && !g.recording
        && !g.capture
        && g.arch == "gfx1201"
        && g.k == DOWN_K
        && g.m > 0
        && g.m.is_multiple_of(32)
        && g.k_top == DOWN_K_TOP
}

/// Whether `tokens` route rows are inside the route's row window.
fn down_a8_rows_admit(tokens: usize) -> bool {
    (1..=QWEN4_MOE_DOWN_A8_ROUTE_CAP).contains(&tokens)
}

/// Token count of a grouped call whose X rows are one per top-10 pick (`x_row_div == 1`,
/// `x_src_rows == 10 * tokens`); `None` for any other shape.
fn grouped_tokens(x_row_div: usize, x_src_rows: usize) -> Option<usize> {
    (x_row_div == 1 && x_src_rows.is_multiple_of(DOWN_K_TOP)).then_some(x_src_rows / DOWN_K_TOP)
}

/// Kind of the two launch ABIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Indexed,
    Grouped,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Indexed => "indexed",
            Kind::Grouped => "grouped",
        }
    }

    fn entry(self, planar: bool) -> &'static str {
        match (self, planar) {
            (Kind::Indexed, false) => INDEXED_ENTRY,
            (Kind::Indexed, true) => INDEXED_PLANAR_ENTRY,
            (Kind::Grouped, false) => GROUPED_ENTRY,
            (Kind::Grouped, true) => GROUPED_PLANAR_ENTRY,
        }
    }
}

/// The one-time receipt line (without the `[INFO] ` prefix).
fn first_launch_line(kind: Kind, rows: usize, entry: &str) -> String {
    format!(
        "qwen4 moe down A8: first launch kind={} rows={rows} entry={entry}",
        kind.name()
    )
}

/// Row class of a receipt: 1, 2..=8, 9..=64.
fn row_class(rows: usize) -> usize {
    match rows {
        0 | 1 => 0,
        2..=8 => 1,
        _ => 2,
    }
}

/// Print the receipt once per process for each (kind, row class).
fn first_launch_receipt(kind: Kind, rows: usize, entry: &str) {
    static SEEN: [AtomicBool; 6] = [
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
        AtomicBool::new(false),
    ];
    let slot = row_class(rows) + 3 * usize::from(kind == Kind::Grouped);
    if !SEEN[slot].swap(true, Ordering::Relaxed) {
        eprintln!("[INFO] {}", first_launch_line(kind, rows, entry));
    }
}

impl Gpu {
    /// Whether the routed-down A8 route is requested on this device: exact gfx1201 and
    /// `HIPFIRE_QWEN4_MOE_DOWN_A8` explicitly on (`1`; unset and `0` leave the incumbent down).
    pub fn qwen4_moe_down_a8_requested(&self) -> bool {
        down_a8_requested(&self.arch, *QWEN4_MOE_DOWN_A8)
    }

    /// Record the loader's verdict: every routed MoE layer this device serves has symmetric qt53
    /// down experts. Defaults to false (fail closed).
    pub fn set_qwen4_moe_down_a8_verified(&mut self, on: bool) {
        self.qwen4_moe_down_a8_verified = on;
    }

    /// Mark the MTP head's MoE layer verdict as still owed (`true`) or settled (`false`). While
    /// pending the route declines, so a head whose experts were never verified never takes it.
    pub fn set_qwen4_moe_down_a8_head_pending(&mut self, pending: bool) {
        self.qwen4_moe_down_a8_head_pending = pending;
    }

    /// The loader's symmetric verdict as last recorded (false until set).
    pub fn qwen4_moe_down_a8_is_verified(&self) -> bool {
        self.qwen4_moe_down_a8_verified
    }

    /// Whether the MTP head layer's verdict is still owed ([`Gpu::set_qwen4_moe_down_a8_head_pending`]).
    pub fn qwen4_moe_down_a8_head_pending(&self) -> bool {
        self.qwen4_moe_down_a8_head_pending
    }

    /// Whether one routed down of `m` rows x `k` with `k_top` picks takes the A8 route: requested,
    /// verified symmetric, head verdict settled, inside the Qwen4 forward, eager (no replay
    /// recording / graph capture), outside the prefill-invariant scope, `K == 640`, `M % 32 == 0`,
    /// `k_top == 10`, exact gfx1201. The row window is checked by the launchers
    /// ([`QWEN4_MOE_DOWN_A8_ROUTE_CAP`]).
    pub fn qwen4_moe_down_a8_applies(&self, m: usize, k: usize, k_top: usize) -> bool {
        down_a8_admits(DownA8Gate {
            requested: self.qwen4_moe_down_a8_requested(),
            verified: self.qwen4_moe_down_a8_verified,
            head_pending: self.qwen4_moe_down_a8_head_pending,
            qwen4_scope: self.qwen4_scope,
            prefill_invariant_scope: self.prefill_invariant_scope,
            recording: self.replay.is_recording(),
            capture: self.graphs.capture_mode,
            arch: &self.arch,
            m,
            k,
            k_top,
        })
    }

    /// [`Gpu::qwen4_moe_down_a8_applies`] for a D-indexed (per-pick, decode) launch of
    /// `batch_size` tokens: additionally needs its own sub-switch
    /// (`HIPFIRE_QWEN4_MOE_DOWN_A8_INDEXED=1`, default off: the dot4 arm is a decode kernel, the
    /// verify window is the grouped arm).
    pub(crate) fn qwen4_moe_down_a8_applies_indexed(
        &self,
        m: usize,
        k: usize,
        batch_size: usize,
        k_top: usize,
    ) -> bool {
        down_a8_indexed_requested(*QWEN4_MOE_DOWN_A8_INDEXED)
            && down_a8_rows_admit(batch_size)
            && self.qwen4_moe_down_a8_applies(m, k, k_top)
    }

    /// [`Gpu::qwen4_moe_down_a8_applies`] for a D-grouped launch (one X row per top-10 pick).
    pub(crate) fn qwen4_moe_down_a8_applies_grouped(
        &self,
        m: usize,
        k: usize,
        x_row_div: usize,
        x_src_rows: usize,
    ) -> bool {
        grouped_tokens(x_row_div, x_src_rows).is_some_and(down_a8_rows_admit)
            && self.qwen4_moe_down_a8_applies(m, k, DOWN_K_TOP)
    }

    /// Grow the A8 sidecar to hold `x_rows` rows (`x_rows * 680` B, 8-byte aligned by the
    /// allocator) and return its pointer. Eager only: callable before capture to reserve; the
    /// launchers call it with the live row count (a no-op once large enough). Growth frees the
    /// previous buffer after a device sync; no captured graph embeds it (the route is eager-only).
    pub fn reserve_qwen4_moe_down_a8(&mut self, x_rows: usize) -> HipResult<*mut c_void> {
        if self.replay.is_recording() || self.graphs.capture_mode {
            return Err(HipError::new(
                0,
                "qwen4_moe_down_a8: sidecar growth is eager-only",
            ));
        }
        let needed = x_rows
            .checked_mul(QWEN4_MOE_DOWN_A8_ROW_BYTES)
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                HipError::new(0, "qwen4_moe_down_a8: need 0 < rows, extent fits usize")
            })?;
        self.bind_thread()?;
        if self.qwen4_moe_down_a8_scratch.is_none() || self.qwen4_moe_down_a8_scratch_bytes < needed
        {
            if let Some(old) = self.qwen4_moe_down_a8_scratch.take() {
                self.hip.device_synchronize()?;
                let _ = self.hip.free(old);
            }
            self.qwen4_moe_down_a8_scratch_bytes = 0;
            let fresh = self.hip.malloc(needed)?;
            self.qwen4_moe_down_a8_scratch = Some(fresh);
            self.qwen4_moe_down_a8_scratch_bytes = needed;
        }
        let ptr = self
            .qwen4_moe_down_a8_scratch
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        debug_assert_eq!(ptr as usize % 8, 0);
        Ok(ptr)
    }

    /// Producer: quantize `x_rows` rotated F32 rows of `x` (`[x_rows][640]`) into the sidecar and
    /// return the sidecar pointer.
    fn qwen4_moe_down_a8_quantize(
        &mut self,
        x: *mut c_void,
        x_rows: usize,
    ) -> HipResult<*mut c_void> {
        let xq = self.reserve_qwen4_moe_down_a8(x_rows)?;
        self.ensure_kernel(MODULE, QWEN4_MOE_DOWN_A8_GFX1201_SRC, QUANT_ENTRY)?;
        let kv = DOWN_K as i32;
        let rv = i32::try_from(x_rows)
            .map_err(|_| HipError::new(0, "qwen4_moe_down_a8: row count exceeds i32"))?;
        let mut params = [
            &x as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rv as *const _ as *mut c_void,
        ];
        let bytes = x_rows * (DOWN_K * 4 + QWEN4_MOE_DOWN_A8_ROW_BYTES);
        let timer = crate::profile::begin_timer(&self.hip, "quant", QUANT_ENTRY, bytes);
        let result = self.launch_maybe_blob(
            QUANT_ENTRY,
            [x_rows as u32, 1, 1],
            [(DOWN_K / 128 * 32) as u32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(x);
                b.push_ptr(xq);
                b.push_i32(kv);
                b.push_i32(rv);
                debug_assert_eq!(b.len(), 24, "producer ABI: 24 explicit kernarg bytes");
                b.pad_to(16);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result.map(|()| xq)
    }

    /// D-indexed A8 routed down (`1..=64` tokens, `k_top == 10`, `K == 640`, `M % 32 == 0`):
    /// producer over `hidden_batch` (`[batch][k_top][640]` rotated F32 picks), then the indexed
    /// entry (planar iff `mq4_experts_planar`); `expert_outputs` is `[batch][k_top][M]` unweighted
    /// F32 rows, the incumbent expanded layout. Callers check
    /// [`Gpu::qwen4_moe_down_a8_applies_indexed`] and validated the operand spans.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen4_moe_down_a8_indexed(
        &mut self,
        expert_ptrs: &crate::dispatch::GpuTensor,
        topk_indices: &crate::dispatch::GpuTensor,
        hidden_batch: &crate::dispatch::GpuTensor,
        expert_outputs: &crate::dispatch::GpuTensor,
        m: usize,
        k: usize,
        batch_size: usize,
        n_exp: usize,
        k_top: usize,
    ) -> HipResult<()> {
        debug_assert!(self.qwen4_moe_down_a8_applies_indexed(m, k, batch_size, k_top));
        let n_exp_i32 = i32::try_from(n_exp)
            .map_err(|_| HipError::new(0, "qwen4_moe_down_a8: expert count exceeds i32"))?;
        let planar = self.mq4_experts_planar;
        let entry = Kind::Indexed.entry(planar);
        let x_rows = batch_size * k_top;
        let xq = self.qwen4_moe_down_a8_quantize(hidden_batch.buf.as_ptr(), x_rows)?;
        self.ensure_kernel(MODULE, QWEN4_MOE_DOWN_A8_GFX1201_SRC, entry)?;
        first_launch_receipt(Kind::Indexed, batch_size, entry);
        crate::moe_entry_log::moe_entry_receipt("down", entry, batch_size);
        let pp = expert_ptrs.buf.as_ptr();
        let ip = topk_indices.buf.as_ptr();
        let yp = expert_outputs.buf.as_ptr();
        let (mv, kv, nv, ktv) = (m as i32, k as i32, batch_size as i32, k_top as i32);
        let mut params = [
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &nv as *const _ as *mut c_void,
            &n_exp_i32 as *const _ as *mut c_void,
            &ktv as *const _ as *mut c_void,
        ];
        let bytes = x_rows * (m * (k / 128 * 68) + QWEN4_MOE_DOWN_A8_ROW_BYTES + m * 4);
        let timer = crate::profile::begin_timer(&self.hip, "gemv", entry, bytes);
        let result = self.launch_maybe_blob(
            entry,
            [(m / 32) as u32, k_top as u32, batch_size as u32],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xq);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(nv);
                b.push_i32(n_exp_i32);
                b.push_i32(ktv);
                debug_assert_eq!(b.len(), 52, "D-indexed ABI: 52 explicit kernarg bytes");
                b.pad_to(16);
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// D-grouped A8 routed down (`x_row_div == 1`, `x_src_rows == 10 * tokens`, `1..=64` tokens):
    /// producer over `x_src` (`[x_src_rows][640]` rotated F32), then the grouped entry (planar iff
    /// `mq4_experts_planar`) with the incumbent `y_grouped` contract: `[grouped_rows][M]` F32,
    /// every grouped slot stored (+0 for dead/pad slots). Callers check
    /// [`Gpu::qwen4_moe_down_a8_applies_grouped`] and validated `y_grouped` / `x_src` spans.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qwen4_moe_down_a8_grouped(
        &mut self,
        expert_ptrs: &crate::dispatch::GpuTensor,
        expert_tile_ids: &crate::dispatch::GpuTensor,
        sorted_slot_index: &crate::dispatch::GpuTensor,
        x_src: &crate::dispatch::GpuTensor,
        y_grouped: &crate::dispatch::GpuTensor,
        m: usize,
        k: usize,
        x_row_div: usize,
        grouped_rows: usize,
        x_src_rows: usize,
        n_exp: usize,
    ) -> HipResult<()> {
        debug_assert!(self.qwen4_moe_down_a8_applies_grouped(m, k, x_row_div, x_src_rows));
        let planar = self.mq4_experts_planar;
        // Dead rows (sentinel tiles, pad slots) are not stored unless HIPFIRE_QWEN4_MOE_DEAD_STORES=1:
        // the one switch of the incumbent grouped down and of this route.
        let nodz = !self.moe_dead_stores;
        let entry = grouped_entry(planar, nodz);
        let xq = self.qwen4_moe_down_a8_quantize(x_src.buf.as_ptr(), x_src_rows)?;
        self.ensure_kernel(MODULE, QWEN4_MOE_DOWN_A8_GFX1201_SRC, entry)?;
        first_launch_receipt(Kind::Grouped, x_src_rows / DOWN_K_TOP, entry);
        crate::moe_entry_log::moe_entry_receipt("down", entry, x_src_rows / DOWN_K_TOP);
        let ep = expert_ptrs.buf.as_ptr();
        let tp = expert_tile_ids.buf.as_ptr();
        let sp = sorted_slot_index.buf.as_ptr();
        let yp = y_grouped.buf.as_ptr();
        let (mv, kv, rd, gr, xr, ne) = (
            m as i32,
            k as i32,
            x_row_div as i32,
            grouped_rows as i32,
            x_src_rows as i32,
            n_exp as i32,
        );
        let mut params = [
            &ep as *const _ as *mut c_void,
            &tp as *const _ as *mut c_void,
            &sp as *const _ as *mut c_void,
            &xq as *const _ as *mut c_void,
            &yp as *const _ as *mut c_void,
            &mv as *const _ as *mut c_void,
            &kv as *const _ as *mut c_void,
            &rd as *const _ as *mut c_void,
            &gr as *const _ as *mut c_void,
            &xr as *const _ as *mut c_void,
            &ne as *const _ as *mut c_void,
        ];
        let bytes = grouped_rows * (m * 4 + QWEN4_MOE_DOWN_A8_ROW_BYTES);
        let timer = crate::profile::begin_timer(&self.hip, "gemm", entry, bytes);
        let result = self.launch_maybe_blob(
            entry,
            [m.div_ceil(64) as u32, grouped_rows.div_ceil(16) as u32, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = KernargBlob::new();
                b.push_ptr(ep);
                b.push_ptr(tp);
                b.push_ptr(sp);
                b.push_ptr(xq);
                b.push_ptr(yp);
                b.push_i32(mv);
                b.push_i32(kv);
                b.push_i32(rd);
                b.push_i32(gr);
                b.push_i32(xr);
                b.push_i32(ne);
                debug_assert_eq!(b.len(), 64, "D-grouped ABI: 64 explicit kernarg bytes");
                b
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> DownA8Gate<'static> {
        DownA8Gate {
            requested: true,
            verified: true,
            head_pending: false,
            qwen4_scope: true,
            prefill_invariant_scope: false,
            recording: false,
            capture: false,
            arch: "gfx1201",
            m: 2560,
            k: 640,
            k_top: 10,
        }
    }

    #[test]
    fn requested_is_exact_gfx1201_and_opt_in() {
        // Default OFF: unset and `0` do not request it (they are the same), `1` does.
        assert!(!down_a8_requested("gfx1201", None));
        assert!(down_a8_requested("gfx1201", Some(true)));
        assert!(!down_a8_requested("gfx1201", Some(false)));
        for arch in ["gfx1151", "gfx1100", "gfx1200", "gfx942", ""] {
            assert!(!down_a8_requested(arch, None), "{arch}");
            assert!(!down_a8_requested(arch, Some(true)), "{arch}");
        }
    }

    #[test]
    fn the_grouped_entry_follows_layout_and_the_nodz_switch() {
        // The dead-store twin is selected by the landed `HIPFIRE_QWEN4_MOE_DEAD_STORES` switch
        // (default: skip dead rows), not by a flag of its own.
        assert_eq!(grouped_entry(false, false), GROUPED_ENTRY);
        assert_eq!(grouped_entry(true, false), GROUPED_PLANAR_ENTRY);
        assert_eq!(grouped_entry(false, true), GROUPED_NODZ_ENTRY);
        assert_eq!(grouped_entry(true, true), GROUPED_PLANAR_NODZ_ENTRY);
        // The nodz twins differ from the production entries by the template flag alone.
        assert!(QWEN4_MOE_DOWN_A8_GFX1201_SRC
            .contains("a8_dn_grouped_body<false, A8_GR_NW, A8_GR_RG, true>"));
        assert!(QWEN4_MOE_DOWN_A8_GFX1201_SRC
            .contains("a8_dn_grouped_body<true, A8_GR_NW, A8_GR_RG, true>"));
        assert!(QWEN4_MOE_DOWN_A8_GFX1201_SRC.contains("if (!NODZ || live) Y["));
    }

    #[test]
    fn the_indexed_arm_has_its_own_opt_in_sub_switch() {
        assert!(!down_a8_indexed_requested(None));
        assert!(!down_a8_indexed_requested(Some(false)));
        assert!(down_a8_indexed_requested(Some(true)));
        // The sub-switch alone never requests the route (the main flag + verdict gate it).
        assert!(!down_a8_requested("gfx1201", None));
    }

    #[test]
    fn admits_only_the_full_conjunction() {
        assert!(down_a8_admits(gate()));
        let off = |f: fn(&mut DownA8Gate<'static>)| {
            let mut g = gate();
            f(&mut g);
            assert!(!down_a8_admits(g), "{g:?}");
        };
        off(|g| g.requested = false);
        off(|g| g.verified = false);
        off(|g| g.head_pending = true);
        off(|g| g.qwen4_scope = false);
        off(|g| g.prefill_invariant_scope = true);
        off(|g| g.recording = true);
        off(|g| g.capture = true);
        off(|g| g.arch = "gfx1151");
        off(|g| g.k = 512);
        off(|g| g.k = 1280);
        off(|g| g.m = 0);
        off(|g| g.m = 2561);
        off(|g| g.m = 2048 + 16);
        off(|g| g.k_top = 8);
        off(|g| g.k_top = 16);
        let mut g = gate();
        g.m = 2048;
        assert!(down_a8_admits(g));
    }

    #[test]
    fn row_window_is_one_to_forty_eight() {
        assert!(!down_a8_rows_admit(0));
        assert!(down_a8_rows_admit(1));
        assert!(down_a8_rows_admit(8));
        assert!(down_a8_rows_admit(9));
        assert!(down_a8_rows_admit(48));
        assert!(down_a8_rows_admit(64));
        assert!(!down_a8_rows_admit(65));
        assert!(!down_a8_rows_admit(512));
    }

    #[test]
    fn grouped_tokens_need_one_x_row_per_pick() {
        assert_eq!(grouped_tokens(1, 10), Some(1));
        assert_eq!(grouped_tokens(1, 480), Some(48));
        assert_eq!(grouped_tokens(1, 650), Some(65));
        assert_eq!(grouped_tokens(1, 95), None);
        assert_eq!(grouped_tokens(10, 100), None);
        assert_eq!(grouped_tokens(1, 0), Some(0));
        assert!(!grouped_tokens(1, 0).is_some_and(down_a8_rows_admit));
    }

    #[test]
    fn entries_follow_kind_and_layout() {
        assert_eq!(
            Kind::Indexed.entry(false),
            "qwen4_moe_down_a8_indexed_gfx1201"
        );
        assert_eq!(
            Kind::Indexed.entry(true),
            "qwen4_moe_down_a8_indexed_planar_gfx1201"
        );
        assert_eq!(
            Kind::Grouped.entry(false),
            "qwen4_moe_down_a8_grouped_gfx1201"
        );
        assert_eq!(
            Kind::Grouped.entry(true),
            "qwen4_moe_down_a8_grouped_planar_gfx1201"
        );
    }

    #[test]
    fn receipt_line_and_row_classes() {
        assert_eq!(
            first_launch_line(Kind::Indexed, 4, "qwen4_moe_down_a8_indexed_gfx1201"),
            "qwen4 moe down A8: first launch kind=indexed rows=4 entry=qwen4_moe_down_a8_indexed_gfx1201"
        );
        assert_eq!(
            first_launch_line(Kind::Grouped, 12, "qwen4_moe_down_a8_grouped_planar_gfx1201"),
            "qwen4 moe down A8: first launch kind=grouped rows=12 entry=qwen4_moe_down_a8_grouped_planar_gfx1201"
        );
        assert_eq!([1, 2, 8, 9, 64].map(row_class), [0, 1, 1, 2, 2]);
    }

    #[test]
    fn source_carries_every_entry_and_the_prelude() {
        assert!(QWEN4_MOE_DOWN_A8_GFX1201_SRC.contains(&format!("void {QUANT_ENTRY}(")));
        for entry in [
            INDEXED_ENTRY,
            INDEXED_PLANAR_ENTRY,
            GROUPED_ENTRY,
            GROUPED_PLANAR_ENTRY,
            GROUPED_NODZ_ENTRY,
            GROUPED_PLANAR_NODZ_ENTRY,
        ] {
            let stem = entry.trim_end_matches("_gfx1201");
            assert!(
                QWEN4_MOE_DOWN_A8_GFX1201_SRC
                    .contains(&format!("A8_ENTRY({stem}, A8_SUF, _gfx1201)(")),
                "{entry}"
            );
        }
        assert!(QWEN4_MOE_DOWN_A8_GFX1201_SRC
            .starts_with("#define HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE 1\n"));
        assert!(QWEN4_MOE_DOWN_A8_GFX1201_SRC.contains("quantize_block_i8_128_wave"));
        assert_eq!(QWEN4_MOE_DOWN_A8_ROW_BYTES, 680);
    }
}
