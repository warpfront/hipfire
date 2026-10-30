// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP tensor operation wrappers.
//!
//! These functions enforce the tensor layout and F32/BF16 boundaries at the
//! shared HIP launch site; architecture-specific tuning remains an explicit
//! selector rather than an admission requirement.

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::prefill_invariant::{qwen4_gdn_prefill, Qwen4GdnPrefill};
use crate::{DType, Gpu, GpuTensor};

#[path = "npu_trunk_consumers.rs"]
pub(crate) mod npu_arena;
pub use npu_arena::*;

pub(crate) const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
pub(crate) const HYPER_READ_UP_WMMA_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_wmma.gfx1151.hip");
pub(crate) const HYPER_READ_UP_FUSED_TWIN_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_fused_twin.gfx1151.hip");
pub(crate) const HYPER_READ_UP_WMMA_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_wmma.gfx1201.hip");
pub(crate) const HYPER_READ_UP_WMMA_GFX1151_V2_SRC: &str =
    include_str!("../../../kernels/src/hyper_read_up_v2.gfx1151.hip");
pub(crate) const GATED_DELTA_CHUNK_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_chunk_wmma.gfx1151.hip");
pub(crate) const GATED_DELTA_CHUNK_Q8_WMMA_SRC: &str =
    include_str!("../../../kernels/src/gated_delta_chunk_q8_wmma.gfx1151.hip");

pub(crate) const INDEXED_ATTENTION_DENSE_WMMA_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_dense_wmma.gfx1151.hip");
pub(crate) const INDEXED_ATTENTION_GATHERED_WMMA_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_gathered_wmma.gfx1151.hip");
const INDEXED_ATTENTION_GATHERED_WMMA_GFX1201_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_gathered_wmma.gfx1201.hip");
const INDEXED_ATTENTION_SELECT_EXACT_SRC: &str =
    include_str!("../../../kernels/src/indexed_attention_select_exact.hip");
/// `HIPFIRE_QWEN4_QSA_WMMA_GATHER` (on unless `0`) routes QSA prefill
/// attention chunks (rows >= QWEN4_F16_WMMA_MIN_TOKENS) that the full-window
/// dense route does not take through the gathered F16 WMMA kernels: gfx1151 on
/// the F32 and q8 states, gfx1201 on the fp8 state.  Not bit-exact against the hg4
/// kernel; admitted because its error against an f64 reference is no worse
/// than BF16 storage of Q/K/V/P.  `0` keeps every launch of the incumbent
/// route.  Read once.
static QWEN4_QSA_WMMA_GATHER: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_WMMA_GATHER", true)
});
/// `HIPFIRE_QWEN4_GDN_PIPE=0` keeps the non-capture batched GDN recurrence of
/// gfx1201 (F32 and Q8 state) and gfx942 (F32 state) on
/// `gated_delta_step_halves_state128_persistent256_{f32,q8}` instead of the
/// register-resident `gated_delta_step_pipe128_{f32,q8}` (bytewise identical
/// results; the Q8 kernel is gfx1201-only).  Read once.
static QWEN4_GDN_PIPE: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_GDN_PIPE", true));
/// `HIPFIRE_QWEN4_QSA_PM` (on unless `0`) runs the gathered route's producer
/// and attention from the certified builder module (`kernels::QSA_GATHER_PM_*`,
/// same ABI, grid, LDS and output bytes as the hipcc kernels) instead of the
/// JIT source, on every gathered-route arch (gfx1151, gfx1201).  `0` keeps the
/// hipcc kernels.  Read once.
static QWEN4_QSA_PM: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_PM", true));
/// Persistent per-QSA-layer F16 K/V for exact gfx1151 F32-state prefill.
/// Opt-in (`HIPFIRE_QWEN4_QSA_F16_ARENA=1`); off keeps per-launch
/// conversion. Resolved once at load; decode keeps the F32 state.
static QWEN4_QSA_F16_ARENA: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_F16_ARENA", false)
});
/// `HIPFIRE_QWEN4_QSA_SCORE_PM` / `HIPFIRE_QWEN4_QSA_SELECT_PM` (each on
/// unless `0`) use the certified builder rows16 F32/BF16 score and
/// select-from-scores kernels on covered gfx1151 prefill and decode calls
/// ([`qsa_select_pm_fits`]), with identical output bytes. For F32, `0` keeps
/// the corresponding hipcc kernel; SCORE_PM=0 keeps the fused hipcc selector
/// for BF16 arenas. SELECT_PM chooses PM versus hipcc selection. Read once.
static QWEN4_QSA_SCORE_PM: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SCORE_PM", true));
/// Select PM versus hipcc from-scores for either dtype; BF16 only reaches
/// this choice when SCORE_PM is enabled, otherwise it stays fused hipcc.
static QWEN4_QSA_SELECT_PM: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SELECT_PM", true)
});
/// Disable PM admission below the prefill threshold with `0`. Read once.
static QWEN4_QSA_SELECT_PM_DECODE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SELECT_PM_DECODE", true)
});
/// `HIPFIRE_QWEN4_QSA_SELECT_EXACT=1` runs the batched QSA selector on the
/// `_exact` kernels (tile sort + fixed-order merge instead of the all-pairs
/// ranks; selected indices and mirror byte-identical) for complete <= 2048
/// pooled blocks, outside recording and capture.  Unset or `0` keeps the
/// incumbent selector.  Read once.
static QWEN4_QSA_SELECT_EXACT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SELECT_EXACT", false)
});
/// Pooled blocks the exact selector handles; larger contexts keep the
/// incumbent selector.
const QSA_SELECT_EXACT_MAX_BLOCKS: usize = 2048;
/// Budget blocks the exact selector's 512-entry merge bound covers.
const QSA_SELECT_EXACT_MAX_BUDGET_BLOCKS: usize = 512;
/// Dynamic LDS floor of the exact selector kernels
/// (`indexed_attention_select_exact.hip`), which keep no static LDS:
/// 6144 B even in global-score mode, otherwise 4 B per live block.
const QSA_SELECT_EXACT_DYNAMIC_LDS_FLOOR_BYTES: usize = 6 * 1024;
const QSA_SELECT_PARALLEL_THREADS: u32 = 256;
// gfx1151's 64-KiB dynamic LDS budget; other devices use the serial path.
// Oversized rows also use serial kernels without changing the contract.
const QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;
/// Static LDS of `indexed_attention_select_f32_batched` beside its dynamic
/// score row: 256 radix bins and the 512-entry chosen-block list
/// (`QSA_SELECT_LIST_CAPACITY` in tensor_ops.hip) plus scalars, rounded up.
/// The LDS route needs `shape_blocks * 4 + this` within the LDS limit, i.e. at
/// most 15360 pooled blocks (61440 tokens at compress 4); larger arenas keep
/// their score rows in global memory.
pub const QSA_SELECT_BATCHED_STATIC_LDS_BYTES: usize = 4 * 1024;
/// Rows per launch of the batched selector when its scores live in global
/// memory (arenas past the LDS row); longer batches launch in groups, so the
/// score scratch is `QSA_SELECT_GLOBAL_ROWS * padded blocks` floats.
const QSA_SELECT_GLOBAL_ROWS: usize = 256;

#[cfg(test)]
thread_local! {
    /// Test-only: route the batched selector to its serial oracle kernel.
    static SELECT_FORCE_SERIAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
fn select_forced_serial() -> bool {
    SELECT_FORCE_SERIAL.with(|flag| flag.get())
}
#[cfg(not(test))]
#[inline(always)]
fn select_forced_serial() -> bool {
    false
}
const QSA_ATTENTION_PARALLEL_THREADS: u32 = 256;
const QSA_ATTENTION_LDS_BYTES_PER_ROW: usize = 8; // F32 score + i32 token.
const QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeError {
    WrongDtype,
    WrongShape,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongDtype => write!(f, "tensor operation expects F32 tensors"),
            Self::WrongShape => write!(f, "tensor operation tensor shape mismatch"),
        }
    }
}

impl std::error::Error for ComputeError {}

pub(crate) fn ensure_f32(tensor: &GpuTensor) -> HipResult<()> {
    if tensor.dtype != DType::F32 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    Ok(())
}

const KERNEL_BLOCK: usize = 256;
const MAX_KERNEL_EXTENT: usize = i32::MAX as usize;

fn checked_extent(value: usize, what: &'static str) -> HipResult<usize> {
    if value <= MAX_KERNEL_EXTENT {
        Ok(value)
    } else {
        Err(HipError::new(0, &format!("{what} exceeds i32")))
    }
}

fn checked_product(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    checked_extent(value, what)
}

fn checked_product3(
    first: usize,
    second: usize,
    third: usize,
    what: &'static str,
) -> HipResult<usize> {
    checked_product(first, second, what).and_then(|value| checked_product(value, third, what))
}

fn checked_add(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_add(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    checked_extent(value, what)
}

fn checked_i32(value: usize, what: &'static str) -> HipResult<i32> {
    checked_extent(value, what).map(|value| value as i32)
}

fn checked_u32(value: usize, what: &'static str) -> HipResult<u32> {
    u32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds u32")))
}

pub(crate) fn blocks(elements: usize) -> HipResult<u32> {
    let elements = checked_extent(elements, "tensor operation flattened extent")?;
    checked_u32(
        elements.div_ceil(KERNEL_BLOCK),
        "tensor operation block grid",
    )
}

/// Storage format of a GDN recurrent state (`kernels/src/tensor_ops.hip`,
/// "Q8 GDN recurrent state").  The launchers take it explicitly
/// (`GatedDeltaStep::format`, `GatedDeltaStepBatched::format`): a Raw slot
/// is plain Q8 or Q8Ef, which the tensor dtype cannot tell apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GdnStateFormat {
    /// F32 `[value_head][key][value]`: the exact reference.
    F32,
    /// Qwen3.5's Q8 DeltaNet state as one Raw byte slot: per value head 128
    /// rows (value channel) of 128 i8 key-channel codes, then one F32 scale
    /// per row. 128x128 heads only.
    Q8,
    /// Q8 with a live F16 error-feedback residual: the recurrent slot is the
    /// plain [`GdnStateFormat::Q8`] slot (codes + scales; the ring, mirrors,
    /// snapshots and cache payloads carry only it), and the residual of the
    /// last requantization is a separate live F16 tensor per GDN layer and
    /// live owner ([`GdnStateFormat::residual_units`], element index
    /// `(head * 128 + row) * 128 + column`, the Q8 code index).  The kernels
    /// add the incoming residual once and requantize with round-to-nearest-even
    /// (no RNG, so no frame kernarg).  Explicit-only: `auto` never selects it.
    Q8Ef,
}

impl GdnStateFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Q8 => "q8",
            Self::Q8Ef => "q8_ef",
        }
    }

    pub const fn supports(self, key_dim: usize, value_dim: usize) -> bool {
        match self {
            Self::F32 => key_dim > 0 && value_dim > 0,
            Self::Q8 | Self::Q8Ef => key_dim == 128 && value_dim == 128,
        }
    }

    /// Dtype of the recurrent slot (the residual of [`Self::Q8Ef`] is a
    /// separate `DType::F16` tensor).
    pub const fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Q8 | Self::Q8Ef => DType::Raw,
        }
    }

    /// `dtype` units (F32 elements or bytes) of one state of `value_heads`
    /// heads of `key_dim x value_dim`; Q8Ef's slot equals Q8's.
    pub const fn state_units(self, value_heads: usize, key_dim: usize, value_dim: usize) -> usize {
        match self {
            Self::F32 => value_heads * key_dim * value_dim,
            Self::Q8 | Self::Q8Ef => value_heads * value_dim * (key_dim + 4),
        }
    }

    pub const fn state_bytes(self, value_heads: usize, key_dim: usize, value_dim: usize) -> usize {
        match self {
            Self::F32 => self.state_units(value_heads, key_dim, value_dim) * 4,
            Self::Q8 | Self::Q8Ef => self.state_units(value_heads, key_dim, value_dim),
        }
    }

    /// The format of the recurrent slot itself: what the ring, the
    /// transaction mirror, snapshots and cache payloads hold (Q8Ef -> Q8).
    pub const fn storage(self) -> Self {
        match self {
            Self::Q8Ef => Self::Q8,
            Self::F32 | Self::Q8 => self,
        }
    }

    /// Whether the format carries a separate live residual tensor.
    pub const fn has_residual(self) -> bool {
        match self {
            Self::Q8Ef => true,
            Self::F32 | Self::Q8 => false,
        }
    }

    /// F16 elements of the live residual of one state (0 without one).
    pub const fn residual_units(
        self,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> usize {
        match self {
            Self::Q8Ef => value_heads * value_dim * key_dim,
            Self::F32 | Self::Q8 => 0,
        }
    }

    /// Bytes of the live residual of one state (2 per F16 element).
    pub const fn residual_bytes(
        self,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> usize {
        2 * self.residual_units(value_heads, key_dim, value_dim)
    }

    /// Whether the plain-Q8 twin of a kernel takes a trailing `u32`
    /// requantization frame (stochastic rounding); Q8Ef rounds nearest-even
    /// and F32 never requantizes, so neither does.
    const fn has_frame_kernarg(self) -> bool {
        match self {
            Self::Q8 => true,
            Self::F32 | Self::Q8Ef => false,
        }
    }
}

/// Strict check of a GDN state descriptor shared by every launcher: `state`
/// is `format`'s slot (dtype and extent), a residual is present exactly for a
/// residual format and is an F16 tensor of [`GdnStateFormat::residual_units`]
/// elements that overlaps neither the state nor the ring, and a `row_states`
/// ring (verify capture) holds `rows` states of `format.storage()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_gdn_state(
    format: GdnStateFormat,
    state: &GpuTensor,
    residual: Option<&GpuTensor>,
    row_states: Option<&GpuTensor>,
    rows: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> HipResult<()> {
    if !format.supports(key_dim, value_dim) {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if state.dtype != format.dtype() {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let state_units = format.state_units(value_heads, key_dim, value_dim);
    if state.numel() != state_units {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if residual.is_some() != format.has_residual() {
        return Err(HipError::new(
            0,
            &format!(
                "GDN state format {} {} a live residual tensor",
                format.name(),
                if format.has_residual() {
                    "requires"
                } else {
                    "does not take"
                }
            ),
        ));
    }
    let state_range = tensor_byte_range(state, format.state_bytes(value_heads, key_dim, value_dim));
    if let Some(residual) = residual {
        if residual.dtype != DType::F16 {
            return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
        }
        let units = format.residual_units(value_heads, key_dim, value_dim);
        let bytes = format.residual_bytes(value_heads, key_dim, value_dim);
        if residual.numel() != units || residual.buf.size() < bytes {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        if ranges_overlap(tensor_byte_range(residual, bytes), state_range) {
            return Err(HipError::new(0, "GDN residual overlaps the state slot"));
        }
    }
    if let Some(ring) = row_states {
        let storage = format.storage();
        let slot = storage.state_units(value_heads, key_dim, value_dim);
        let needed = rows
            .checked_mul(slot)
            .ok_or_else(|| HipError::new(0, "GDN ring size overflow"))?;
        if ring.dtype != storage.dtype() || ring.numel() < needed {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        if let Some(residual) = residual {
            let bytes = format.residual_bytes(value_heads, key_dim, value_dim);
            let ring_bytes = needed * storage.dtype().size();
            if ranges_overlap(
                tensor_byte_range(residual, bytes),
                tensor_byte_range(ring, ring_bytes),
            ) {
                return Err(HipError::new(0, "GDN residual overlaps the state ring"));
            }
        }
    }
    Ok(())
}

/// `[start, end)` device address range of the first `bytes` of `tensor`.
fn tensor_byte_range(tensor: &GpuTensor, bytes: usize) -> (usize, usize) {
    let start = tensor.buf.as_ptr() as usize;
    (start, start.saturating_add(bytes))
}

fn ranges_overlap(left: (usize, usize), right: (usize, usize)) -> bool {
    left.0 < right.1 && right.0 < left.1
}

/// A GDN state as a kernel receives it: the slot pointer and, for
/// [`GdnStateFormat::Q8Ef`], the live F16 residual pointer that the kernel ABI
/// places **immediately after the slot pointer**.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnStateKernarg {
    format: GdnStateFormat,
    state: *mut std::ffi::c_void,
    residual: Option<*mut std::ffi::c_void>,
}

impl GdnStateKernarg {
    /// Refuses a residual that does not match [`GdnStateFormat::has_residual`].
    pub(crate) fn new(
        format: GdnStateFormat,
        state: *mut std::ffi::c_void,
        residual: Option<*mut std::ffi::c_void>,
    ) -> HipResult<Self> {
        if residual.is_some() != format.has_residual() {
            return Err(HipError::new(
                0,
                &format!(
                    "GDN state format {} residual pointer mismatch",
                    format.name()
                ),
            ));
        }
        Ok(Self {
            format,
            state,
            residual,
        })
    }

    /// The slot pointer, then the residual pointer iff the format has one.
    fn push_slot(&self, args: &mut KernargBlob) {
        args.push_ptr(self.state);
        if let Some(residual) = self.residual {
            args.push_ptr(residual);
        }
    }

    /// The trailing `u32` requantization frame iff the format takes one;
    /// returns its kernarg offset.
    fn push_frame(&self, args: &mut KernargBlob, frame: u32) -> Option<usize> {
        if self.format.has_frame_kernarg() {
            args.push_u32(frame);
            Some(args.len() - 4)
        } else {
            None
        }
    }
}

pub struct GatedDeltaStep<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    /// The recurrent slot in `format`'s storage: F32, or a Raw Q8 slot
    /// (also for Q8Ef).
    pub state: &'a GpuTensor,
    /// Format of `state`; strictly validated against its dtype and extent.
    pub format: GdnStateFormat,
    /// The live F16 residual: `Some` exactly for [`GdnStateFormat::Q8Ef`].
    pub residual: Option<&'a GpuTensor>,
    pub output: &'a GpuTensor,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// The token's position: seeds a Q8 state's requantization (declared to
    /// the recorder, so a replayed decode reseeds per position).  Q8Ef has no
    /// frame and ignores it.
    pub position: usize,
}

/// Head-pair arrival counters of the fused GDN step's optional rotation.
const GDN_PAIR_COUNTERS: usize = 256;

/// Bytes of the head-pair arrival counters [`ensure_gdn_pair_counters`] allocates.
pub const GDN_PAIR_COUNTERS_BYTES: usize = GDN_PAIR_COUNTERS * 4;

/// Allocate and zero the fused GDN step's head-pair arrival counters if they
/// do not exist yet. Each launch leaves them zero, so the zeroing memset runs
/// once. A retained-body boundary calls this before opening its capture
/// window: the tape replays dispatches only, and a memset inside the window
/// makes the capture effect-incomplete.
pub fn ensure_gdn_pair_counters(gpu: &mut Gpu) -> HipResult<()> {
    if gpu.scratch.gdn_pair_counters.is_none() {
        let counters = gpu.hip.malloc(GDN_PAIR_COUNTERS_BYTES)?;
        gpu.hip.memset(&counters, 0, GDN_PAIR_COUNTERS_BYTES)?;
        gpu.scratch.gdn_pair_counters = Some(counters);
    }
    Ok(())
}

pub fn gated_delta_step(gpu: &mut Gpu, p: &GatedDeltaStep<'_>) -> HipResult<()> {
    gated_delta_step_launch(gpu, p, None, None)
}

/// [`gated_delta_step`] followed by [`gated_delta_gate`] of its output
/// (`g.recurrent_output` must be `p.output`). On the gfx11+ 128x128 route one
/// launch keeps the recurrent output in LDS (bitwise the two launches; the
/// recurrent output is then not stored); elsewhere the two launches run.
/// With `rotate_into`, that launch also writes `mq_rotate_x(g.output)` there
/// (head pairs form the 256-wide groups) and the next matching
/// `Gpu::rotate_x_mq` skips (`ScratchState::prerotated`); returns whether it
/// did.
pub fn gated_delta_step_gated(
    gpu: &mut Gpu,
    p: &GatedDeltaStep<'_>,
    g: &GatedDeltaGate<'_>,
    rotate_into: Option<&GpuTensor>,
) -> HipResult<bool> {
    let fused = gpu.arch_caps.has_gfx11_plus_simt() && p.key_dim == 128 && p.value_dim == 128;
    if !fused || g.recurrent_output.buf.as_ptr() != p.output.buf.as_ptr() {
        gated_delta_step(gpu, p)?;
        gated_delta_gate(gpu, g)?;
        return Ok(false);
    }
    validate_gated_delta_gate(g)?;
    let width = g.output.numel();
    let rotate_into = rotate_into
        .filter(|r| p.value_heads.is_multiple_of(2) && r.dtype == DType::F32 && r.numel() >= width);
    gated_delta_step_launch(gpu, p, Some(g), rotate_into)?;
    if let Some(rotated) = rotate_into {
        gpu.scratch.prerotated = Some((
            g.output.buf.as_ptr() as usize,
            rotated.buf.as_ptr() as usize,
            width,
        ));
    }
    Ok(rotate_into.is_some())
}

/// `(kernel, shared_norm128)` of a single-token GDN step.  `gfx11_128`:
/// gfx11+ SIMT with 128x128 heads (the F32 shared-norm bodies).  Every Q8
/// family kernel is the 128x128 body on the dequantized state, so those
/// always take the shared-norm (128-thread) geometry; the Q8Ef entries are
/// the Q8 twins with a residual kernarg.
fn gdn_step_single_kernel(
    format: GdnStateFormat,
    gated: bool,
    gfx11_128: bool,
) -> (&'static str, bool) {
    match format {
        GdnStateFormat::Q8 => (
            if gated {
                "gated_delta_step_gate_norm128_q8"
            } else {
                "gated_delta_step_norm128_q8"
            },
            true,
        ),
        GdnStateFormat::Q8Ef => (
            if gated {
                "gated_delta_step_gate_norm128_q8ef"
            } else {
                "gated_delta_step_norm128_q8ef"
            },
            true,
        ),
        GdnStateFormat::F32 => match (gated, gfx11_128) {
            (true, true) => ("gated_delta_step_gate_norm128_gfx1151", true),
            (false, true) => ("gated_delta_step_shared_norm128_gfx1151", true),
            (_, false) => ("gated_delta_step_f32", false),
        },
    }
}

/// Output arguments of a single-token GDN step launch.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GdnStepOutput {
    /// The recurrent output tensor.
    Recurrent { output: *mut std::ffi::c_void },
    /// Fused with the gate: `z`, `norm`, gated `output`, then (shared-norm
    /// kernels only) the rotation `[rotated, signs1, signs2, counters]`
    /// pointers, null when no rotation was asked for.
    Gated {
        z: *mut std::ffi::c_void,
        norm: *mut std::ffi::c_void,
        output: *mut std::ffi::c_void,
        rotate: Option<[*mut std::ffi::c_void; 4]>,
    },
}

/// Resolved arguments of one single-token GDN step launch
/// ([`gdn_step_kernargs`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnStepKernargs {
    pub q: *mut std::ffi::c_void,
    pub k: *mut std::ffi::c_void,
    pub v: *mut std::ffi::c_void,
    pub gate: *mut std::ffi::c_void,
    pub beta: *mut std::ffi::c_void,
    pub state: GdnStateKernarg,
    pub output: GdnStepOutput,
    pub key_heads: i32,
    pub value_heads: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub query_scale: f32,
    /// The token's position (a Q8 frame; unused by F32 and Q8Ef).
    pub frame: u32,
}

/// The unpadded kernarg blob of a single-token GDN step and, when the format
/// takes a frame (plain Q8 only), the frame's offset for the replay binding.
/// Q8Ef: the residual pointer right after the slot pointer, no frame.
pub(crate) fn gdn_step_kernargs(a: &GdnStepKernargs) -> (KernargBlob, Option<usize>) {
    let mut args = KernargBlob::new();
    args.push_ptr(a.q);
    args.push_ptr(a.k);
    args.push_ptr(a.v);
    args.push_ptr(a.gate);
    args.push_ptr(a.beta);
    a.state.push_slot(&mut args);
    match a.output {
        GdnStepOutput::Recurrent { output } => args.push_ptr(output),
        GdnStepOutput::Gated {
            z, norm, output, ..
        } => {
            args.push_ptr(z);
            args.push_ptr(norm);
            args.push_ptr(output);
        }
    }
    args.push_i32(a.key_heads);
    args.push_i32(a.value_heads);
    args.push_i32(a.key_dim);
    args.push_i32(a.value_dim);
    args.push_f32(a.query_scale);
    if let GdnStepOutput::Gated {
        rotate: Some(rotate),
        ..
    } = a.output
    {
        for ptr in rotate {
            args.push_ptr(ptr);
        }
    }
    let frame_offset = a.state.push_frame(&mut args, a.frame);
    (args, frame_offset)
}

/// The per-launch shape checks of one GDN decode step: its state format and
/// the i32 shape arguments (key heads, value heads, key width, value width).
fn validate_gated_delta_step(p: &GatedDeltaStep<'_>) -> HipResult<(GdnStateFormat, [i32; 4])> {
    let format = p.format;
    for tensor in [p.q, p.k, p.v, p.gate, p.beta, p.output] {
        ensure_f32(tensor)?;
    }
    if p.key_heads == 0 || p.value_heads == 0 || p.value_heads % p.key_heads != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    validate_gdn_state(
        format,
        p.state,
        p.residual,
        None,
        1,
        p.value_heads,
        p.key_dim,
        p.value_dim,
    )?;
    let key_elements = checked_product(p.key_heads, p.key_dim, "GDN key extent")?;
    let value_elements = checked_product(p.value_heads, p.value_dim, "GDN value extent")?;
    checked_product(value_elements, p.key_dim, "GDN state extent")?;
    if p.q.numel() != key_elements
        || p.k.numel() != key_elements
        || p.v.numel() != value_elements
        || p.gate.numel() != p.value_heads
        || p.beta.numel() != p.value_heads
        || p.output.numel() != value_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok((
        format,
        [
            checked_i32(p.key_heads, "GDN key heads")?,
            checked_i32(p.value_heads, "GDN value heads")?,
            checked_i32(p.key_dim, "GDN key width")?,
            checked_i32(p.value_dim, "GDN value width")?,
        ],
    ))
}

fn gated_delta_step_launch(
    gpu: &mut Gpu,
    p: &GatedDeltaStep<'_>,
    gated: Option<&GatedDeltaGate<'_>>,
    rotate_into: Option<&GpuTensor>,
) -> HipResult<()> {
    let (format, [key_heads, value_heads, key_dim, value_dim]) = validate_gated_delta_step(p)?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN value-head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    let gfx11_128 = gpu.arch_caps.has_gfx11_plus_simt() && p.key_dim == 128 && p.value_dim == 128;
    let (kernel, shared_norm128) = gdn_step_single_kernel(format, gated.is_some(), gfx11_128);
    let block_x = if shared_norm128 { 128 } else { 256 };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let output = match gated {
        Some(g) => {
            let rotate = if shared_norm128 {
                let null = std::ptr::null_mut();
                Some(match rotate_into {
                    Some(rotated) => {
                        gpu.ensure_mq_signs()?;
                        ensure_gdn_pair_counters(gpu)?;
                        if p.value_heads / 2 > GDN_PAIR_COUNTERS {
                            return Err(HipError::new(0, "GDN rotate: too many head pairs"));
                        }
                        [
                            rotated.buf.as_ptr(),
                            gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr(),
                            gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr(),
                            gpu.scratch.gdn_pair_counters.as_ref().unwrap().as_ptr(),
                        ]
                    }
                    None => [null, null, null, null],
                })
            } else {
                None
            };
            GdnStepOutput::Gated {
                z: g.z.buf.as_ptr(),
                norm: g.norm.buf.as_ptr(),
                output: g.output.buf.as_ptr(),
                rotate,
            }
        }
        None => GdnStepOutput::Recurrent {
            output: p.output.buf.as_ptr(),
        },
    };
    let (mut args, frame_offset) = gdn_step_kernargs(&GdnStepKernargs {
        q: p.q.buf.as_ptr(),
        k: p.k.buf.as_ptr(),
        v: p.v.buf.as_ptr(),
        gate: p.gate.buf.as_ptr(),
        beta: p.beta.buf.as_ptr(),
        state: GdnStateKernarg::new(
            format,
            p.state.buf.as_ptr(),
            p.residual.map(|residual| residual.buf.as_ptr()),
        )?,
        output,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        query_scale: (p.key_dim as f32).sqrt().recip(),
        frame: if format.has_frame_kernarg() {
            checked_u32(p.position, "GDN Q8 frame")?
        } else {
            0
        },
    });
    args.pad_to(16);
    let frame_binding = frame_offset
        .map(|offset| [crate::replay::ReplayKernargBinding::PositionPlusU32 { offset, addend: 0 }]);
    gpu.launch_blob_recorded(
        kernel,
        [value_heads_grid, value_dim_grid, 1],
        [block_x, 1, 1],
        0,
        args.as_mut_slice(),
        match &frame_binding {
            Some(bindings) => crate::dispatch::ReplayLaunchBindings {
                grid: None,
                kernargs: bindings,
            },
            None => crate::dispatch::ReplayLaunchBindings::NONE,
        },
    )
}
/// Persistent row-batched GDN recurrence for the exact 128x128 geometry.
/// The kernel partitions each value channel's state column across two
/// 64-value halves and serializes cross-half dot chains in shared memory.
/// The older DeltaNet kernels use a different state orientation and omit the
/// source BF16 Q/K normalization, so they cannot replace this exact-state path.
pub struct GatedDeltaStepBatched<'a> {
    pub projection: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    /// The recurrent slot in `format`'s storage: F32, or a Raw Q8 slot
    /// (also for Q8Ef).
    pub state: &'a GpuTensor,
    /// Format of `state`; strictly validated against its dtype and extent.
    pub format: GdnStateFormat,
    /// The live F16 residual: `Some` exactly for [`GdnStateFormat::Q8Ef`].
    pub residual: Option<&'a GpuTensor>,
    pub output: &'a GpuTensor,
    /// Optional `[rows]` ring of `format.storage()` states (a Q8Ef ring is
    /// plain Q8, residual-free): the recurrent state after the last row lands
    /// in slot `rows - 1` instead of updating `state` (speculative verify;
    /// `state` keeps the pre-call value).  A Q8Ef capture also writes the
    /// final residual into the live `residual` plane.
    pub row_states: Option<&'a GpuTensor>,
    pub rows: usize,
    pub qkv_width: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    /// The first row's position; a Q8 state's requantization is seeded by
    /// the last row's. With `row_states` (verify capture) each row also
    /// requantizes at its own position, token-exact with [`gated_delta_step`].
    /// Q8Ef has no frame and ignores it.
    pub position: usize,
}

/// Inputs of the batched GDN recurrence kernel choice
/// ([`gdn_batched_kernel`]).
#[derive(Clone, Copy, Debug)]
struct GdnBatchedSelect {
    /// `row_states` present (verify capture).
    captured: bool,
    format: GdnStateFormat,
    /// `Gpu::qwen4_scope`.
    qwen4_scope: bool,
    /// `HIPFIRE_QWEN4_GDN_PREFILL`; read only for an uncaptured Q8-family
    /// state in the Qwen4 scope (other launches never resolve it, and carry
    /// `Wy` here).
    policy: Qwen4GdnPrefill,
    /// `gpu.arch == "gfx1201"`.
    arch_gfx1201: bool,
    /// `gpu.arch == "gfx942"`.
    arch_gfx942: bool,
    /// `HIPFIRE_QWEN4_GDN_PIPE`.
    pipe_enabled: bool,
}

impl GdnBatchedSelect {
    /// Whether the policy is consulted at all (see [`Self::policy`]).
    const fn reads_policy(captured: bool, format: GdnStateFormat, qwen4_scope: bool) -> bool {
        !captured && qwen4_scope && matches!(format, GdnStateFormat::Q8 | GdnStateFormat::Q8Ef)
    }
}

/// `(kernel, pipe128)` of a batched GDN recurrence launch.
///
/// `HIPFIRE_QWEN4_GDN_PREFILL=canon` (lab): a Q8-family state (Q8 or Q8Ef)
/// inside the Qwen4 forward (`Gpu::qwen4_scope`, so the arm also works with
/// the invariant profile off) runs the token-canonical persistent recurrence,
/// which requantizes every row at its own position and stores to `state`: the
/// result no longer depends on how rows are split into launches.  An F32
/// state keeps the persistent kernel (already token-exact).
///
/// gfx1201 (F32, Q8, Q8Ef) and gfx942 (F32 only) non-capture: the
/// register-resident one-thread-per-column recurrence (bitwise the
/// persistent256 kernel's result), 128 threads.
///
/// The Q8Ef entries are the Q8 twins (`..._q8` -> `..._q8ef`) taking the live
/// residual after the state pointer and no frame.
fn gdn_batched_kernel(s: &GdnBatchedSelect) -> (&'static str, bool) {
    let canon = GdnBatchedSelect::reads_policy(s.captured, s.format, s.qwen4_scope)
        && s.policy == Qwen4GdnPrefill::Canon;
    let pipe128_arch = s.arch_gfx1201
        || (s.arch_gfx942
            && match s.format {
                GdnStateFormat::F32 => true,
                GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => false,
            });
    let pipe128 = !canon && !s.captured && pipe128_arch && s.pipe_enabled;
    let kernel = match (s.captured, s.format) {
        (true, GdnStateFormat::F32) => "gated_delta_step_halves_state128_persistent256_capture_f32",
        (false, GdnStateFormat::F32) if pipe128 => "gated_delta_step_pipe128_f32",
        (false, GdnStateFormat::F32) => "gated_delta_step_halves_state128_persistent256_f32",
        (true, GdnStateFormat::Q8) => "gated_delta_step_halves_state128_persistent256_capture_q8",
        (false, GdnStateFormat::Q8) if canon => {
            "gated_delta_step_halves_state128_persistent256_canon_q8"
        }
        (false, GdnStateFormat::Q8) if pipe128 => "gated_delta_step_pipe128_q8",
        (false, GdnStateFormat::Q8) => "gated_delta_step_halves_state128_persistent256_q8",
        (true, GdnStateFormat::Q8Ef) => {
            "gated_delta_step_halves_state128_persistent256_capture_q8ef"
        }
        (false, GdnStateFormat::Q8Ef) if canon => {
            "gated_delta_step_halves_state128_persistent256_canon_q8ef"
        }
        (false, GdnStateFormat::Q8Ef) if pipe128 => "gated_delta_step_pipe128_q8ef",
        (false, GdnStateFormat::Q8Ef) => "gated_delta_step_halves_state128_persistent256_q8ef",
    };
    (kernel, pipe128)
}

/// Resolved arguments of one batched GDN recurrence launch
/// ([`gdn_batched_kernargs`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnBatchedKernargs {
    pub projection: *mut std::ffi::c_void,
    pub gate: *mut std::ffi::c_void,
    pub beta: *mut std::ffi::c_void,
    pub state: GdnStateKernarg,
    pub output: *mut std::ffi::c_void,
    pub row_states: Option<*mut std::ffi::c_void>,
    pub rows: i32,
    pub qkv_width: i32,
    pub key_heads: i32,
    pub value_heads: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub query_scale: f32,
    /// The last row's absolute position (a Q8 frame; unused by F32 and Q8Ef).
    pub frame: u32,
}

/// The unpadded kernarg blob of a batched GDN recurrence launch: the Q8 twin's
/// kernargs with the Q8Ef residual pointer **right after the state pointer**
/// (before `output` / `row_states`) and without the trailing frame.
pub(crate) fn gdn_batched_kernargs(a: &GdnBatchedKernargs) -> KernargBlob {
    let mut args = KernargBlob::new();
    args.push_ptr(a.projection);
    args.push_ptr(a.gate);
    args.push_ptr(a.beta);
    a.state.push_slot(&mut args);
    args.push_ptr(a.output);
    if let Some(states) = a.row_states {
        args.push_ptr(states);
    }
    args.push_i32(a.rows);
    args.push_i32(a.qkv_width);
    args.push_i32(a.key_heads);
    args.push_i32(a.value_heads);
    args.push_i32(a.key_dim);
    args.push_i32(a.value_dim);
    args.push_f32(a.query_scale);
    a.state.push_frame(&mut args, a.frame);
    args
}

pub fn gated_delta_step_batched(gpu: &mut Gpu, p: &GatedDeltaStepBatched<'_>) -> HipResult<()> {
    let format = p.format;
    for tensor in [p.projection, p.gate, p.beta, p.output] {
        ensure_f32(tensor)?;
    }
    if p.rows == 0
        || p.key_heads == 0
        || p.value_heads == 0
        || p.value_heads % p.key_heads != 0
        || p.key_dim != 128
        || p.value_dim != 128
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    validate_gdn_state(
        format,
        p.state,
        p.residual,
        p.row_states,
        p.rows,
        p.value_heads,
        p.key_dim,
        p.value_dim,
    )?;
    let qk = checked_product(p.key_heads, p.key_dim, "GDN batched qk extent")?;
    let value = checked_product(p.value_heads, p.value_dim, "GDN batched value extent")?;
    let expected_qkv = checked_add(
        checked_product(2, qk, "GDN batched qkv")?,
        value,
        "GDN batched qkv",
    )?;
    let rows_qkv = checked_product(p.rows, expected_qkv, "GDN batched projection extent")?;
    let rows_value = checked_product(p.rows, value, "GDN batched output extent")?;
    let rows_heads = checked_product(p.rows, p.value_heads, "GDN batched parameter extent")?;
    if p.qkv_width != expected_qkv
        || p.projection.numel() != rows_qkv
        || p.gate.numel() != rows_heads
        || p.beta.numel() != rows_heads
        || p.output.numel() != rows_value
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "GDN batched rows")?;
    let qkv_width = checked_i32(p.qkv_width, "GDN batched qkv width")?;
    let key_heads = checked_i32(p.key_heads, "GDN batched key heads")?;
    let value_heads = checked_i32(p.value_heads, "GDN batched value heads")?;
    let key_dim = checked_i32(p.key_dim, "GDN batched key width")?;
    let value_dim = checked_i32(p.value_dim, "GDN batched value width")?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN batched value-head grid")?;
    let captured = p.row_states.is_some();
    let (kernel, pipe128) = gdn_batched_kernel(&GdnBatchedSelect {
        captured,
        format,
        qwen4_scope: gpu.qwen4_scope,
        policy: if GdnBatchedSelect::reads_policy(captured, format, gpu.qwen4_scope) {
            qwen4_gdn_prefill()
        } else {
            Qwen4GdnPrefill::Wy
        },
        arch_gfx1201: gpu.arch == "gfx1201",
        arch_gfx942: gpu.arch == "gfx942",
        pipe_enabled: *QWEN4_GDN_PIPE,
    });
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let frame = if format.has_frame_kernarg() {
        checked_u32(p.position + p.rows - 1, "GDN Q8 frame")?
    } else {
        0
    };
    let mut args = gdn_batched_kernargs(&GdnBatchedKernargs {
        projection: p.projection.buf.as_ptr(),
        gate: p.gate.buf.as_ptr(),
        beta: p.beta.buf.as_ptr(),
        state: GdnStateKernarg::new(
            format,
            p.state.buf.as_ptr(),
            p.residual.map(|residual| residual.buf.as_ptr()),
        )?,
        output: p.output.buf.as_ptr(),
        row_states: p.row_states.map(|states| states.buf.as_ptr()),
        rows,
        qkv_width,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        query_scale: (p.key_dim as f32).sqrt().recip(),
        frame,
    });
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [value_heads_grid, 1, 1],
        [if pipe128 { 128 } else { 256 }, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Rollback kernel of a state's STORAGE format (`format.storage()`): the Q8Ef
/// rollback is the plain-Q8 replay (the ring is plain Q8); the Q8Ef entry set
/// has no rollback kernel.
fn gdn_rollback_kernel(storage: GdnStateFormat) -> &'static str {
    match storage.storage() {
        GdnStateFormat::F32 => "gated_delta_rollback_layers_f32",
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => "gated_delta_rollback_layers_q8",
    }
}

/// Few-row verify rollback of every GDN layer in one launch (kernel
/// `gated_delta_rollback_layers_{f32,q8}`): `table` is `layers` pairs of
/// device pointers (captured recurrence input, ring of `format.storage()`
/// states); each layer re-runs its first `keep` of `rows` captured rows from
/// ring slot `from` and leaves the last kept row's state in slot
/// `to + keep - 1`. 128-wide heads.
///
/// Rollback is storage-based: there is no EF rollback kernel.  A
/// [`GdnStateFormat::Q8Ef`] rollback replays the plain-Q8 ring with
/// `gated_delta_rollback_layers_q8` (the ring holds no residual), and the
/// CALLER then zeroes every rolled-back layer's live residual with
/// [`gdn_q8ef_residual_zero`] in stream order.
pub struct GatedDeltaRollbackLayers<'a> {
    pub table: &'a GpuTensor,
    pub discard: &'a GpuTensor,
    pub format: GdnStateFormat,
    pub layers: usize,
    pub rows: usize,
    pub keep: usize,
    pub from: usize,
    pub to: usize,
    pub qkv_width: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    /// Position of the last kept row: a Q8 state's kept row `r` requantizes
    /// at `position - (keep - 1) + r`, token-exact with [`gated_delta_step`].
    pub position: usize,
}

pub fn gated_delta_rollback_layers(
    gpu: &mut Gpu,
    p: &GatedDeltaRollbackLayers<'_>,
) -> HipResult<()> {
    let value = checked_product(p.value_heads, 128, "GDN rollback value width")?;
    if p.keep == 0
        || p.keep > p.rows
        || p.key_heads == 0
        || !p.value_heads.is_multiple_of(p.key_heads)
        || p.qkv_width != 2 * 128 * p.key_heads + value
        || p.table.buf.size() < 16 * p.layers
        || p.discard.numel() < checked_product(p.keep, value, "GDN rollback output")?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let storage = p.format.storage();
    let kernel = gdn_rollback_kernel(storage);
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.table.buf.as_ptr());
    args.push_ptr(p.discard.buf.as_ptr());
    for (value, label) in [
        (p.rows, "GDN rollback rows"),
        (p.keep, "GDN rollback kept rows"),
        (p.from, "GDN rollback source slot"),
        (p.to, "GDN rollback target slot"),
        (p.qkv_width, "GDN rollback qkv width"),
        (p.key_heads, "GDN rollback key heads"),
        (p.value_heads, "GDN rollback value heads"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_f32(128f32.sqrt().recip());
    if storage.has_frame_kernarg() {
        args.push_u32(checked_u32(p.position, "GDN Q8 frame")?);
    }
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [
            checked_u32(p.value_heads, "GDN rollback value-head grid")?,
            checked_u32(p.layers, "GDN rollback layer grid")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// Whether [`gated_delta_step_gate_wmma`] applies: the Qwen4 F16 prefill
/// route (gfx11 WMMA, >= QWEN4_F16_WMMA_MIN_TOKENS rows, no recorder or capture,
/// not opted out) with 128-wide heads.  Decided before the convolution, which
/// then stores its output as packed BF16 for that kernel.
pub fn gated_delta_chunk_route(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.arch_caps.has_wmma_w32()
        // Admitted wide verify keeps the token-canonical captured recurrence,
        // never WY/WMMA's F16 arithmetic (nor its capture refusal).
        && !(p.rows > crate::prefill_invariant::FEW_ROW_ROUTE_ROWS
            && p.rows <= gpu.few_row_limit()
            && gpu.route_rows(p.rows) <= gpu.few_row_limit())
        && gpu.route_rows(p.rows) >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && qwen4_gdn_prefill() == Qwen4GdnPrefill::Wy
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.key_dim == 128
        && p.value_dim == 128
        && p.key_heads > 0
        && p.value_heads % p.key_heads == 0
}

/// Whether H6's inline-Q8 chunk kernel (`gated_delta_chunk_gate_q8_wmma`) can
/// run this step, flag aside: exact gfx1151 on the chunked route, a plain Q8
/// slot without a residual (F32 never; Q8Ef has no inline kernel, so it is
/// never admitted and always runs on its F32 reconstruction) whose base is
/// 16-byte aligned (the kernel loads codes as `int4`).
pub fn gated_delta_q8_inline_geometry(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.arch_caps.is_gfx1151()
        && gpu.arch_caps.has_wmma_w32()
        && gated_delta_chunk_route(gpu, p)
        && gdn_chunk_inline_format(p.format, p.residual.is_some())
        && (p.state.buf.as_ptr() as usize) % 16 == 0
}

/// Format part of the inline chunk geometry: only plain Q8 without a residual
/// tensor.  F32 has no inline kernel, and Q8Ef has none either (it is not
/// GPU-gated yet).
const fn gdn_chunk_inline_format(format: GdnStateFormat, has_residual: bool) -> bool {
    matches!(format, GdnStateFormat::Q8) && !has_residual
}

/// [`gated_delta_q8_inline_geometry`] under `HIPFIRE_QWEN4_GDN_Q8_INLINE=1`
/// (default off): the selector [`gated_delta_step_gate_wmma`] uses.
pub fn gated_delta_q8_inline_route(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.flags.qwen4_gdn_q8_inline_enabled() && gated_delta_q8_inline_geometry(gpu, p)
}

/// Whether H5's convolution that also writes the normalized Q/K
/// (`gated_delta_conv_qknorm_bf16_f32_batched_k4`) can run this chunk, flag
/// aside: exact gfx1151 wave32 on the chunked route (so never a recorder, a
/// graph capture or the persistent gfx1201/gfx1100-persistent route), 16 key
/// heads of 128 in q|k|v channel order, `channels % 256 == 0`, K=4 with a
/// 3-row ring, and a BF16 convolution output.
pub fn gated_delta_conv_qknorm_geometry(
    gpu: &Gpu,
    step: &GatedDeltaStepBatched<'_>,
    conv: &GatedDeltaConvBatched<'_>,
) -> bool {
    gpu.arch_caps.is_gfx1151()
        && gpu.arch_caps.has_wmma_w32()
        && gated_delta_chunk_route(gpu, step)
        && step.key_heads == 16
        && step.key_dim == 128
        && step.qkv_width == conv.channels
        && conv.rows == step.rows
        && conv.channels >= 4096
        && conv.channels % 256 == 0
        && conv.history_rows == 3
        && conv.kernel_size == 4
        && conv.output.dtype == DType::BF16
}

/// [`gated_delta_conv_qknorm_geometry`] under
/// `HIPFIRE_QWEN4_GDN_CONV_QKNORM=1` (default off).  When it holds, the
/// caller launches [`gated_delta_conv_params_qknorm_batched`] in place of
/// [`gated_delta_conv_params_batched`] and then
/// [`gated_delta_step_gate_wmma_qknormed`] in place of
/// [`gated_delta_step_gate_wmma`].
pub fn gated_delta_conv_qknorm_route(
    gpu: &Gpu,
    step: &GatedDeltaStepBatched<'_>,
    conv: &GatedDeltaConvBatched<'_>,
) -> bool {
    gpu.flags.qwen4_gdn_conv_qknorm_enabled() && gated_delta_conv_qknorm_geometry(gpu, step, conv)
}

/// Opt-in arms of the chunked GDN launch.
#[derive(Clone, Copy, Default)]
struct GatedDeltaChunkArms {
    /// The convolution already wrote the normalized Q/K into the scratch.
    qk_prenormed: bool,
    /// Q8-family (Q8 or Q8Ef) state decoded/requantized inside the
    /// recurrence kernel.
    q8_inline: bool,
}

/// [`gated_delta_step_batched`] followed by [`gated_delta_gate_batched`] as
/// one chunked (16-row WY form) F16 WMMA kernel where
/// [`gated_delta_chunk_route`] holds; `p.projection` is the convolution
/// output as packed BF16.  `gate.output` receives the gated rows and
/// `p.output` is not written.  The recurrence's products round to F16, so
/// the route is KLD-gated; the gate is that kernel's expression.  A Q8 state
/// runs the inline-Q8 kernel where [`gated_delta_q8_inline_route`] holds.
pub fn gated_delta_step_gate_wmma(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
) -> HipResult<()> {
    let arms = GatedDeltaChunkArms {
        qk_prenormed: false,
        q8_inline: gated_delta_q8_inline_route(gpu, p),
    };
    gated_delta_step_gate_wmma_arms(gpu, p, gate, arms)
}

/// [`gated_delta_step_gate_wmma`] after
/// [`gated_delta_conv_params_qknorm_batched`] over the same rows: the
/// normalized Q/K are already in the F16 activation scratch, so the
/// Q/K-norm launch is skipped.  Nothing may use that scratch in between.
pub fn gated_delta_step_gate_wmma_qknormed(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
) -> HipResult<()> {
    let arms = GatedDeltaChunkArms {
        qk_prenormed: true,
        q8_inline: gated_delta_q8_inline_route(gpu, p),
    };
    gated_delta_step_gate_wmma_arms(gpu, p, gate, arms)
}

/// The chunk kernel module of a state: only the inline plain-Q8 kernel has its
/// own module over its own chunk source (it carries no EF text, so its code
/// object is the plain Q8 kernel's alone); every other launch (F32, or a
/// Q8-family state run on its F32 scratch, Q8Ef always) uses the plain F32
/// chunk kernel.  `arena`: the direct-arena twin.
pub(crate) const fn gdn_chunk_module_name(
    format: GdnStateFormat,
    inline: bool,
    arena: bool,
) -> &'static str {
    match (format, inline, arena) {
        (GdnStateFormat::Q8, true, false) => "gated_delta_chunk_q8_wmma",
        (GdnStateFormat::Q8, true, true) => "gated_delta_chunk_q8_wmma_arena",
        (GdnStateFormat::F32 | GdnStateFormat::Q8Ef, _, false)
        | (GdnStateFormat::Q8, false, false) => "gated_delta_chunk_wmma",
        (GdnStateFormat::F32 | GdnStateFormat::Q8Ef, _, true)
        | (GdnStateFormat::Q8, false, true) => "gated_delta_chunk_wmma_arena",
    }
}

/// Source text of [`gdn_chunk_module_name`]'s module: the inline Q8 kernel
/// lives in its own source file, so the plain F32 code object is unchanged.
pub(crate) const fn gdn_chunk_source(
    format: GdnStateFormat,
    inline: bool,
    arena: bool,
) -> &'static str {
    match (format, inline, arena) {
        (GdnStateFormat::Q8, true, false) => GATED_DELTA_CHUNK_Q8_WMMA_SRC,
        (GdnStateFormat::Q8, true, true) => npu_arena::CHUNK_Q8_ARENA_SRC,
        (GdnStateFormat::F32 | GdnStateFormat::Q8Ef, _, false)
        | (GdnStateFormat::Q8, false, false) => GATED_DELTA_CHUNK_WMMA_SRC,
        (GdnStateFormat::F32 | GdnStateFormat::Q8Ef, _, true)
        | (GdnStateFormat::Q8, false, true) => npu_arena::CHUNK_ARENA_SRC,
    }
}

/// Entry point of [`gdn_chunk_module_name`]'s module.
pub(crate) const fn gdn_chunk_kernel_name(
    format: GdnStateFormat,
    inline: bool,
    arena: bool,
) -> &'static str {
    match (format, inline, arena) {
        (GdnStateFormat::Q8, true, false) => "gated_delta_chunk_gate_q8_wmma",
        (GdnStateFormat::Q8, true, true) => "gated_delta_chunk_gate_q8_wmma_arena",
        (GdnStateFormat::F32 | GdnStateFormat::Q8Ef, _, false)
        | (GdnStateFormat::Q8, false, false) => "gated_delta_chunk_gate_wmma",
        (GdnStateFormat::F32 | GdnStateFormat::Q8Ef, _, true)
        | (GdnStateFormat::Q8, false, true) => "gated_delta_chunk_gate_wmma_arena",
    }
}

/// The state argument of a chunk kernel launch.
#[derive(Clone, Copy, Debug)]
pub(crate) enum GdnChunkState {
    /// An F32 state (the state itself, or the F32 scratch of a Q8-family state
    /// converted around the launch): `float* state`, no mode, no frame.
    F32 { state: *mut std::ffi::c_void },
    /// Inline plain-Q8 slot: mode 1, trailing requantization `frame`.
    Q8Inline {
        state: *mut std::ffi::c_void,
        frame: u32,
    },
}

/// Resolved arguments of one chunk GDN kernel launch
/// ([`gdn_chunk_kernargs`]); `z` is the gate `z` source (or the arena).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GdnChunkKernargs {
    pub projection: *mut std::ffi::c_void,
    pub qn: *mut std::ffi::c_void,
    pub kn: *mut std::ffi::c_void,
    pub gate: *mut std::ffi::c_void,
    pub beta: *mut std::ffi::c_void,
    pub state: GdnChunkState,
    pub z: *mut std::ffi::c_void,
    pub norm: *mut std::ffi::c_void,
    pub output: *mut std::ffi::c_void,
    pub rows: i32,
    pub qkv_width: i32,
    pub key_heads: i32,
    pub value_heads: i32,
    pub query_scale: f32,
    /// `1` when the gated output is stored as BF16 bits, else `0`.
    pub output_bf16: i32,
}

/// The UNPADDED kernarg blob of a chunk GDN launch (the arena twin appends its
/// pitch arguments, then the caller pads): `projection, qn, kn, gate, beta,
/// state, z, norm, output, rows, qkv_width, key_heads, value_heads,
/// query_scale, output_bf16`, then for the inline plain-Q8 kernel `i32 mode = 1`
/// and the `u32` frame.
pub(crate) fn gdn_chunk_kernargs(a: &GdnChunkKernargs) -> KernargBlob {
    let mut args = KernargBlob::new();
    args.push_ptr(a.projection);
    args.push_ptr(a.qn);
    args.push_ptr(a.kn);
    args.push_ptr(a.gate);
    args.push_ptr(a.beta);
    match a.state {
        GdnChunkState::F32 { state } | GdnChunkState::Q8Inline { state, .. } => {
            args.push_ptr(state)
        }
    }
    args.push_ptr(a.z);
    args.push_ptr(a.norm);
    args.push_ptr(a.output);
    args.push_i32(a.rows);
    args.push_i32(a.qkv_width);
    args.push_i32(a.key_heads);
    args.push_i32(a.value_heads);
    args.push_f32(a.query_scale);
    args.push_i32(a.output_bf16);
    match a.state {
        GdnChunkState::F32 { .. } => {}
        GdnChunkState::Q8Inline { frame, .. } => {
            args.push_i32(1);
            args.push_u32(frame);
        }
    }
    args
}

fn gated_delta_step_gate_wmma_arms(
    gpu: &mut Gpu,
    p: &GatedDeltaStepBatched<'_>,
    gate: &GatedDeltaGateBatched<'_>,
    arms: GatedDeltaChunkArms,
) -> HipResult<()> {
    if !gated_delta_chunk_route(gpu, p)
        || p.projection.dtype != DType::BF16
        || (arms.q8_inline && !gated_delta_q8_inline_geometry(gpu, p))
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let format = p.format;
    validate_gdn_state(
        format,
        p.state,
        p.residual,
        None,
        1,
        p.value_heads,
        p.key_dim,
        p.value_dim,
    )?;
    for tensor in [p.gate, p.beta, gate.z] {
        ensure_f32(tensor)?;
    }
    // The gated output is BF16-rounded: stored as BF16 bits or F32.
    let output_bf16 = gate.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(gate.output)?;
    }
    let qk = checked_product(p.key_heads, p.key_dim, "GDN chunk qk extent")?;
    let value = checked_product(p.value_heads, p.value_dim, "GDN chunk value extent")?;
    if p.qkv_width != 2 * qk + value
        || p.projection.numel() < p.rows * p.qkv_width
        || p.gate.numel() < p.rows * p.value_heads
        || p.beta.numel() < p.rows * p.value_heads
        || gate.rows != p.rows
        || gate.value_heads != p.value_heads
        || gate.value_dim != p.value_dim
        || gate.norm.dtype != DType::BF16
        || gate.norm.numel() != p.value_dim
        || gate.z.numel() < p.rows * value
        || gate.output.numel() < p.rows * value
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "GDN chunk rows")?;
    let qkv_width = checked_i32(p.qkv_width, "GDN chunk qkv width")?;
    let key_heads = checked_i32(p.key_heads, "GDN chunk key heads")?;
    let value_heads = checked_i32(p.value_heads, "GDN chunk value heads")?;
    let qn = gpu.qwen4_f16_x_scratch(2 * p.rows * qk)?;
    let qp = qn.buf.as_ptr();
    let kp = unsafe { (qp as *mut u8).add(p.rows * qk * 2) } as *mut std::ffi::c_void;
    if !arms.qk_prenormed {
        gpu.ensure_kernel_public(
            "tensor_ops",
            TENSOR_OPS_SRC,
            "gated_delta_qk_norm_bf16_batched",
        )?;
    }
    gpu.ensure_kernel_public(
        gdn_chunk_module_name(format, arms.q8_inline, false),
        gdn_chunk_source(format, arms.q8_inline, false),
        gdn_chunk_kernel_name(format, arms.q8_inline, false),
    )?;
    // With `qk_prenormed` the convolution already stored the normalized Q/K
    // in this same scratch (`gated_delta_conv_params_qknorm_batched`).
    if !arms.qk_prenormed {
        let mut args = KernargBlob::new();
        args.push_ptr(p.projection.buf.as_ptr());
        args.push_ptr(qp);
        args.push_ptr(kp);
        args.push_i32(rows);
        args.push_i32(qkv_width);
        args.push_i32(key_heads);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            "gated_delta_qk_norm_bf16_batched",
            [checked_u32(p.rows, "GDN chunk row grid")?, 1, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
    }
    // A Q8-family state runs the chunk on its F32 reconstruction (the state
    // scratch; Q8Ef adds the live residual) and is requantized from it after
    // the chunk (plain Q8 seeded by the last row; Q8Ef round-to-nearest-even
    // writing fresh codes, scales and residual).  Only the plain Q8 inline arm
    // (`gated_delta_chunk_gate_q8_wmma`) does both inside the kernel on the
    // slot itself.  A plain Q8 launch seeds the requantization with the last
    // row's absolute position, `position + rows - 1`.  The route excludes a
    // recorder and graph capture (`gated_delta_chunk_route`), so the frame is
    // never a replayed kernarg and needs no mutable-frame binding.
    let frame = if format.has_frame_kernarg() {
        checked_u32(p.position + p.rows - 1, "GDN Q8 frame")?
    } else {
        0
    };
    let slot = p.state.buf.as_ptr();
    let residual = p.residual.map(|residual| residual.buf.as_ptr());
    let mut scratch = None;
    // A Q8Ef inline request never reaches here: `arms.q8_inline &&
    // !gated_delta_q8_inline_geometry` above already refused it (Q8Ef has no
    // inline kernel), so a Q8Ef chunk always takes the scratch arm.
    let chunk_state = match (format, arms.q8_inline) {
        (GdnStateFormat::F32, _) => GdnChunkState::F32 { state: slot },
        (GdnStateFormat::Q8, true) => GdnChunkState::Q8Inline { state: slot, frame },
        (GdnStateFormat::Q8, false) | (GdnStateFormat::Q8Ef, _) => {
            let f32_state = gpu.gdn_state_f32_scratch(GdnStateFormat::F32.state_bytes(
                p.value_heads,
                p.key_dim,
                p.value_dim,
            ))?;
            gdn_state_to_f32(gpu, format, slot, residual, f32_state, p.value_heads)?;
            scratch = Some(f32_state);
            GdnChunkState::F32 { state: f32_state }
        }
    };
    let mut args = gdn_chunk_kernargs(&GdnChunkKernargs {
        projection: p.projection.buf.as_ptr(),
        qn: qp,
        kn: kp,
        gate: p.gate.buf.as_ptr(),
        beta: p.beta.buf.as_ptr(),
        state: chunk_state,
        z: gate.z.buf.as_ptr(),
        norm: gate.norm.buf.as_ptr(),
        output: gate.output.buf.as_ptr(),
        rows,
        qkv_width,
        key_heads,
        value_heads,
        query_scale: (p.key_dim as f32).sqrt().recip(),
        output_bf16: i32::from(output_bf16),
    });
    args.pad_to(16);
    gpu.launch_blob_recorded(
        gdn_chunk_kernel_name(format, arms.q8_inline, false),
        [checked_u32(p.value_heads, "GDN chunk head grid")?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some(f32_state) = scratch {
        gdn_state_from_f32(gpu, format, f32_state, slot, residual, p.value_heads, frame)?;
    }
    Ok(())
}

/// Converter kernel of a Q8-family `format`: `to_f32` reconstructs the F32
/// state (Q8Ef adds the live residual once), otherwise requantizes an F32
/// state (Q8Ef writes fresh codes, scales and residual and never re-adds the
/// old residual).  An F32 state needs no conversion and is refused.
fn gdn_convert_kernel(format: GdnStateFormat, to_f32: bool) -> HipResult<&'static str> {
    match (format, to_f32) {
        (GdnStateFormat::Q8, true) => Ok("gdn_state_q8_to_f32"),
        (GdnStateFormat::Q8, false) => Ok("gdn_state_f32_to_q8"),
        (GdnStateFormat::Q8Ef, true) => Ok("gdn_state_q8ef_to_f32"),
        (GdnStateFormat::Q8Ef, false) => Ok("gdn_state_f32_to_q8ef"),
        (GdnStateFormat::F32, _) => Err(HipError::new(
            0,
            "GDN state conversion: an F32 state has no Q8 slot",
        )),
    }
}

/// The unpadded kernarg blob of a state converter.  `to_f32`:
/// `(slot, [ef], f32_dst, value_heads)`; from F32: `(f32_src, slot, [ef],
/// value_heads, [frame])`, the frame for plain Q8 only.
fn gdn_convert_kernargs(
    to_f32: bool,
    state: &GdnStateKernarg,
    f32_state: *mut std::ffi::c_void,
    value_heads: i32,
    frame: u32,
) -> KernargBlob {
    let mut args = KernargBlob::new();
    if to_f32 {
        state.push_slot(&mut args);
        args.push_ptr(f32_state);
        args.push_i32(value_heads);
    } else {
        args.push_ptr(f32_state);
        state.push_slot(&mut args);
        args.push_i32(value_heads);
        state.push_frame(&mut args, frame);
    }
    args
}

/// Reconstruct the F32 state `[value_head][key][value]` of a Q8-family slot
/// into `f32_dst`: `gdn_state_q8_to_f32`, or for Q8Ef
/// `gdn_state_q8ef_to_f32(slot, ef, dst, value_heads)` with `residual` the
/// live F16 residual (`Some` exactly for Q8Ef, else an error).  Grid
/// `value_heads * 128` value rows, block 128.
pub(crate) fn gdn_state_to_f32(
    gpu: &mut Gpu,
    format: GdnStateFormat,
    slot: *mut std::ffi::c_void,
    residual: Option<*mut std::ffi::c_void>,
    f32_dst: *mut std::ffi::c_void,
    value_heads: usize,
) -> HipResult<()> {
    gdn_state_convert(gpu, format, true, slot, residual, f32_dst, value_heads, 0)
}

/// Requantize an F32 state into a Q8-family slot: `gdn_state_f32_to_q8` seeded
/// by `frame`, or for Q8Ef `gdn_state_f32_to_q8ef(src, slot, ef, value_heads)`
/// writing fresh codes, scales and `residual`; Q8Ef takes no frame (ignored
/// here, never a kernarg).
pub(crate) fn gdn_state_from_f32(
    gpu: &mut Gpu,
    format: GdnStateFormat,
    f32_src: *mut std::ffi::c_void,
    slot: *mut std::ffi::c_void,
    residual: Option<*mut std::ffi::c_void>,
    value_heads: usize,
    frame: u32,
) -> HipResult<()> {
    gdn_state_convert(
        gpu,
        format,
        false,
        slot,
        residual,
        f32_src,
        value_heads,
        frame,
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_state_convert(
    gpu: &mut Gpu,
    format: GdnStateFormat,
    to_f32: bool,
    slot: *mut std::ffi::c_void,
    residual: Option<*mut std::ffi::c_void>,
    f32_state: *mut std::ffi::c_void,
    value_heads: usize,
    frame: u32,
) -> HipResult<()> {
    let kernel = gdn_convert_kernel(format, to_f32)?;
    let state = GdnStateKernarg::new(format, slot, residual)?;
    let heads = checked_i32(value_heads, "GDN state heads")?;
    let grid = checked_u32(
        checked_product(value_heads, 128, "GDN state row grid")?,
        "GDN state row grid",
    )?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = gdn_convert_kernargs(to_f32, &state, f32_state, heads, frame);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [grid, 1, 1],
        [128, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Kernargs of `gdn_q8ef_residual_zero(unsigned short* ef, unsigned int count)`.
fn gdn_q8ef_zero_kernargs(ef: *mut std::ffi::c_void, count: u32) -> KernargBlob {
    let mut args = KernargBlob::new();
    args.push_ptr(ef);
    args.push_u32(count);
    args
}

/// Zero a live Q8Ef residual (`gdn_q8ef_residual_zero`: grid
/// `ceil(count / 256)`, block 256).  Recorded as a kernel dispatch, so it is
/// stream-ordered and graph/replay-visible (no `hipMemset` inside a
/// dispatch-only tape).  The lifecycle points that reset a Q8Ef state call it
/// per GDN layer: initial reset, keep-0 restore, partial rollback (after
/// [`gated_delta_rollback_layers`]), transaction/session/prefix/disk restore
/// and lane reinit.  An uninterrupted AR step, prefill launch or all-accept
/// commit keeps the residual.  Any non-empty F16 tensor is accepted (the
/// kernel zeroes `numel` elements); a non-F16 tensor is `WrongDtype`, an empty
/// or undersized one `WrongShape`.
pub fn gdn_q8ef_residual_zero(gpu: &mut Gpu, residual: &GpuTensor) -> HipResult<()> {
    if residual.dtype != DType::F16 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let count = residual.numel();
    if count == 0 || residual.buf.size() < checked_product(count, 2, "GDN residual bytes")? {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let count32 = checked_u32(
        checked_extent(count, "GDN residual extent")?,
        "GDN residual extent",
    )?;
    let grid = blocks(count)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gdn_q8ef_residual_zero")?;
    let mut args = gdn_q8ef_zero_kernargs(residual.buf.as_ptr(), count32);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gdn_q8ef_residual_zero",
        [grid, 1, 1],
        [KERNEL_BLOCK as u32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// BF16-stored HC streams (the Qwen4 F16 prefill route) widened to F32 for
/// the PLE block, `out[i] = state[i]`; `state` is F32-typed with its first
/// `n` halves holding the BF16 bits.
pub fn hc_state_bf16_to_f32(
    gpu: &mut Gpu,
    state: &GpuTensor,
    out: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    hc_state_bf16_elementwise(gpu, "hc_state_bf16_to_f32", state, out, n)
}

/// `state[i] = bf16(state[i] + addend[i])` on BF16-stored HC streams; every
/// HC reader rounds the F32 path's unrounded sum to BF16 first.
pub fn hc_state_bf16_add_f32(
    gpu: &mut Gpu,
    state: &GpuTensor,
    addend: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    hc_state_bf16_elementwise(gpu, "hc_state_bf16_add_f32", state, addend, n)
}

pub(crate) const HC_MEAN_CAPTURE_SRC: &str =
    include_str!("../../../kernels/src/hc_mean_capture.hip");

/// DFlash2 target features on Qwen4: the mean over the `hc` hyper-connection
/// streams of `rows` rows of HC state (`[row, stream, hidden]`, F32, or
/// BF16-stored when `bf16`), written as F32 to
/// `out[row * out_stride + out_offset ..][..hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn hc_mean_capture_f32(
    gpu: &mut Gpu,
    streams: &GpuTensor,
    out: &GpuTensor,
    rows: usize,
    hc: usize,
    hidden: usize,
    out_stride: usize,
    out_offset: usize,
    bf16: bool,
) -> HipResult<()> {
    ensure_f32(streams)?;
    ensure_f32(out)?;
    let elements = checked_product3(rows, hc, hidden, "HC mean capture streams")?;
    let stream_f32 = if bf16 { elements.div_ceil(2) } else { elements };
    let out_end = checked_add(
        checked_product(rows.saturating_sub(1), out_stride, "HC mean capture rows")?,
        checked_add(out_offset, hidden, "HC mean capture slot")?,
        "HC mean capture rows",
    )?;
    if rows == 0
        || hc == 0
        || out_offset + hidden > out_stride
        || streams.numel() < stream_f32
        || out.numel() < out_end
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let threads = checked_product(rows, hidden, "HC mean capture threads")?;
    gpu.ensure_kernel_public(
        "hc_mean_capture",
        HC_MEAN_CAPTURE_SRC,
        "hc_mean_capture_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(streams.buf.as_ptr());
    args.push_ptr(out.buf.as_ptr());
    args.push_i32(checked_i32(rows, "HC mean capture rows")?);
    args.push_i32(checked_i32(hc, "HC mean capture streams")?);
    args.push_i32(checked_i32(hidden, "HC mean capture hidden")?);
    args.push_i32(checked_i32(out_stride, "HC mean capture stride")?);
    args.push_i32(checked_i32(out_offset, "HC mean capture offset")?);
    args.push_i32(i32::from(bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hc_mean_capture_f32",
        [blocks(threads)?, 1, 1],
        [KERNEL_BLOCK as u32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

fn hc_state_bf16_elementwise(
    gpu: &mut Gpu,
    kernel: &str,
    state: &GpuTensor,
    other: &GpuTensor,
    n: usize,
) -> HipResult<()> {
    ensure_f32(state)?;
    ensure_f32(other)?;
    if n == 0 || state.numel() * 2 < n || other.numel() < n {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(state.buf.as_ptr());
    args.push_ptr(other.buf.as_ptr());
    args.push_i32(checked_i32(n, "HC state elements")?);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [blocks(n)?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Device-side F32 -> BF16 storage -> F32 conversion at a source activation
/// boundary.  `scratch` is caller-owned and is allocated with the forward
/// arena; no host transfer or per-token allocation occurs.
pub struct Bf16Roundtrip<'a> {
    pub input: &'a GpuTensor,
    pub scratch: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub elements: usize,
}

pub fn bf16_roundtrip_f32(gpu: &mut Gpu, p: &Bf16Roundtrip<'_>) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.output)?;
    let elements = checked_extent(p.elements, "BF16 roundtrip extent")?;
    if elements == 0
        || p.scratch.dtype != DType::BF16
        || p.input.numel() < elements
        || p.scratch.numel() < elements
        || p.output.numel() < elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "BF16 roundtrip extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_roundtrip_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.scratch.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(elements_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "bf16_roundtrip_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// HC-specific in-place fusion of three source BF16 boundaries, F32 scaling,
/// and the existing SiLU expression over a contiguous low-rank buffer.
pub struct HcActivationFused<'a> {
    pub values: &'a GpuTensor,
    pub scale: f32,
    /// Receives the (BF16-exact) results as packed BF16 instead of `values`.
    pub bf16_out: Option<&'a GpuTensor>,
}

pub fn hc_activation_fused_f32(gpu: &mut Gpu, p: &HcActivationFused<'_>) -> HipResult<()> {
    ensure_f32(p.values)?;
    let elements = checked_extent(p.values.numel(), "HC activation extent")?;
    if elements == 0
        || p.bf16_out
            .is_some_and(|out| out.dtype != DType::BF16 || out.numel() < elements)
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "HC activation extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hc_activation_fused_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_i32(elements_i);
    args.push_f32(p.scale);
    args.push_ptr(
        p.bf16_out
            .map_or(std::ptr::null_mut(), |out| out.buf.as_ptr()),
    );
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hc_activation_fused_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Source-exact BF16 product and residual add for the shared expert.
pub struct Bf16ScaledAdd<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub elements: usize,
}

pub fn bf16_scaled_add(gpu: &mut Gpu, p: &Bf16ScaledAdd<'_>) -> HipResult<()> {
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    let elements = checked_extent(p.elements, "BF16 scaled-add extent")?;
    if elements == 0
        || p.residual.numel() < elements
        || p.value.numel() < elements
        || p.scalar.numel() == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "BF16 scaled-add extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_scaled_add_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(elements_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "bf16_scaled_add_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// Source-exact BF16 product/residual add for a row-batched shared expert.
pub struct Bf16ScaledAddBatched<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub rows: usize,
    pub elements: usize,
}

pub fn bf16_scaled_add_batched(gpu: &mut Gpu, p: &Bf16ScaledAddBatched<'_>) -> HipResult<()> {
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    let flat_elements = checked_product(p.rows, p.elements, "BF16 batched scaled-add extent")?;
    if p.rows == 0
        || p.elements == 0
        || p.residual.numel() < flat_elements
        || p.value.numel() < flat_elements
        || p.scalar.numel() < p.rows
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "BF16 batched scaled-add rows")?;
    let elements = checked_i32(p.elements, "BF16 batched scaled-add width")?;
    let grid = blocks(flat_elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_scaled_add_batched_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(rows);
    args.push_i32(elements);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "bf16_scaled_add_batched_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct HyperRead<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub low: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    pub rank: usize,
}

pub fn hyper_read(gpu: &mut Gpu, p: &HyperRead<'_>) -> HipResult<()> {
    for tensor in [p.input, p.low, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 || p.rank == 0 || p.norm_weight.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let input_elements = checked_product(p.branches, p.hidden, "HC read input extent")?;
    let up_elements = checked_product(input_elements, p.rank, "HC read projection extent")?;
    let branches = checked_i32(p.branches, "HC read branch count")?;
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let rank = checked_i32(p.rank, "HC read rank")?;
    if p.input.numel() != input_elements
        || p.norm_weight.numel() != input_elements
        || p.low.numel() != p.rank
        || p.up.numel() != up_elements
        || p.normalized.numel() != input_elements
        || p.mixed.numel() != p.hidden
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.low, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rank);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_read_f32",
        [1, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// HC read where the up projection has already been evaluated to one F32
/// gate logit per branch/hidden column.
pub struct HyperReadProjected<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn hyper_read_projected(gpu: &mut Gpu, p: &HyperReadProjected<'_>) -> HipResult<()> {
    for tensor in [p.input, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.norm_weight.dtype != DType::BF16 || p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "projected HC width")?;
    let input_elements = checked_extent(p.input.numel(), "projected HC input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let mixed_elements = checked_product(rows, p.hidden, "projected HC mixed extent")?;
    let branches = checked_i32(p.branches, "projected HC branch count")?;
    let hidden = checked_i32(p.hidden, "projected HC hidden width")?;
    let rows_i = checked_i32(rows, "projected HC row count")?;
    let grid_x = blocks(wide)?;
    let grid_y = checked_u32(rows, "projected HC row grid")?;
    if p.norm_weight.numel() != wide
        || p.up.numel() != input_elements
        || p.normalized.numel() != input_elements
        || p.mixed.numel() != mixed_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_projected_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_read_projected_f32",
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct HyperWrite<'a> {
    pub input: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    /// `input` / `output` hold BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
}

/// BF16 HC-write kernel selection. [`hyper_write`] validates everything, then
/// picks [`HyperWriteBf16Kernel::X8Gfx1201`] automatically when the measured
/// crossover gate holds; tests force [`HyperWriteBf16Kernel::X2`] for the
/// incumbent reference, or [`HyperWriteBf16Kernel::X8Gfx1201`] below crossover
/// so timing labels stay truthful.
#[derive(Clone, Copy)]
enum HyperWriteBf16Kernel {
    X2,
    X8Gfx1201,
}

/// Shared seven-argument BF16 HC-write launcher (input/mixed/gates/output
/// pointers + branches/hidden/rows i32 + 16-byte padding; block `[256, 1, 1]`,
/// LDS 0, recorded funnel). X2 covers `wide / 2` pairs per row; X8 covers the
/// fixed `wide = 10240` geometry as grid `[5, rows, 1]`.
fn launch_hyper_write_bf16(
    gpu: &mut Gpu,
    p: &HyperWrite<'_>,
    rows: usize,
    kernel: HyperWriteBf16Kernel,
) -> HipResult<()> {
    let wide = checked_product(p.branches, p.hidden, "HC write width")?;
    let branches = checked_i32(p.branches, "HC write branch count")?;
    let hidden = checked_i32(p.hidden, "HC write hidden width")?;
    let rows_i = checked_i32(rows, "HC write row count")?;
    let grid_y = checked_u32(rows, "HC write row grid")?;
    let (name, grid_x) = match kernel {
        HyperWriteBf16Kernel::X2 => ("hyper_write_bf16x2", blocks(wide / 2)?),
        HyperWriteBf16Kernel::X8Gfx1201 => ("hyper_write_bf16x8_gfx1201", 5),
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.mixed, p.gates, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        name,
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Measured-crossover gate for the x8 widening: exact `gfx1201`,
/// `branches == 4`, `hidden == 2560`, 16-byte aligned `input`/`mixed`/`output`
/// bases (the kernel moves streams as `uint4` and mixed as `float4`), and
/// `rows >= 16`.
fn hyper_write_bf16x8_eligible(gpu: &Gpu, p: &HyperWrite<'_>, rows: usize) -> bool {
    gpu.arch == "gfx1201"
        && p.branches == 4
        && p.hidden == 2560
        && rows >= 16
        && (p.input.buf.as_ptr() as usize) % 16 == 0
        && (p.mixed.buf.as_ptr() as usize) % 16 == 0
        && (p.output.buf.as_ptr() as usize) % 16 == 0
}

pub fn hyper_write(gpu: &mut Gpu, p: &HyperWrite<'_>) -> HipResult<()> {
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "HC write width")?;
    let input_elements = checked_extent(p.input.numel(), "HC write input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let mixed_elements = checked_product(rows, p.hidden, "HC write mixed extent")?;
    let gate_elements = checked_product(rows, p.branches, "HC write gate extent")?;
    let branches = checked_i32(p.branches, "HC write branch count")?;
    let hidden = checked_i32(p.hidden, "HC write hidden width")?;
    let rows_i = checked_i32(rows, "HC write row count")?;
    let grid_x = blocks(wide)?;
    let grid_y = checked_u32(rows, "HC write row grid")?;
    if p.normalized.numel() != input_elements
        || p.mixed.numel() != mixed_elements
        || p.gates.numel() != gate_elements
        || p.output.numel() != input_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if p.state_bf16 {
        // Two columns (one BF16 pair) per thread; the x8 widening covers eight
        // columns per thread when the measured-crossover gate holds.
        if p.hidden % 2 != 0 {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        let kernel = if hyper_write_bf16x8_eligible(gpu, p, rows) {
            HyperWriteBf16Kernel::X8Gfx1201
        } else {
            HyperWriteBf16Kernel::X2
        };
        return launch_hyper_write_bf16(gpu, p, rows, kernel);
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_write_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_write_f32",
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// The next hyper write's gate inputs: its norm weight and BF16 `[4, 4 *
/// hidden]` inject projection, and the 16-float quarter-dot output.
pub struct HyperNextGates<'a> {
    pub norm_weight: &'a GpuTensor,
    pub inject: &'a GpuTensor,
    pub quarters: &'a GpuTensor,
}

/// [`hyper_write`] of up to [`crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS`]
/// F32 token rows (`hidden == 2560`, four branches; rows = `input` elements /
/// width) followed by [`hyper_norm`] of the written streams with
/// `norm_weight` into `normalized`, one launch; bitwise the two launches.
/// Block `(branch, row)` is the whole computation of its row (no state shared
/// across `blockIdx.y`), so every row is bitwise the one-row launch whatever
/// the row count and no row-group loop is needed. `quarters_in` (16 per row,
/// from a previous call's `next`) replaces `p.gates` as the gate source;
/// `next` also computes the next hyper write's gate quarters (its norm and k4
/// gate GEMV of these streams, bitwise); `clear` is zero-filled (a
/// `zero_f32`).
pub fn hyper_write_norm(
    gpu: &mut Gpu,
    p: &HyperWrite<'_>,
    norm_weight: &GpuTensor,
    normalized: &GpuTensor,
    quarters_in: Option<&GpuTensor>,
    next: Option<&HyperNextGates<'_>>,
    clear: Option<&GpuTensor>,
) -> HipResult<()> {
    for tensor in [p.input, p.mixed, p.gates, p.output, normalized] {
        ensure_f32(tensor)?;
    }
    let wide = checked_product(p.branches, p.hidden, "HC write width")?;
    // Token rows: one block per (branch, row).
    let rows = p.input.numel() / wide.max(1);
    let quarters_ok = |q: &GpuTensor| q.dtype == DType::F32 && q.numel() == 16 * rows;
    if p.state_bf16
        || p.branches != 4
        || p.hidden != 2560
        || rows == 0
        || rows > crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS
        || norm_weight.dtype != DType::BF16
        || p.input.numel() != rows * wide
        || p.output.numel() != rows * wide
        || normalized.numel() != rows * wide
        || norm_weight.numel() != wide
        || p.mixed.numel() != rows * p.hidden
        || p.gates.numel() != rows * p.branches
        || quarters_in.is_some_and(|q| !quarters_ok(q))
        || clear.is_some_and(|c| c.dtype != DType::F32 || c.numel() > i32::MAX as usize)
        || next.is_some_and(|n| {
            !quarters_ok(n.quarters)
                || n.norm_weight.dtype != DType::BF16
                || n.norm_weight.numel() != wide
                || n.inject.dtype != DType::BF16
                || n.inject.numel() != 4 * wide
        })
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel(
        "qwen4_gemv_bf16_xf32",
        crate::kernels::QWEN4_GEMV_BF16_XF32_SRC,
        "hyper_write_norm_f32",
    )?;
    let null = std::ptr::null_mut();
    let mut args = KernargBlob::new();
    for ptr in [
        p.input.buf.as_ptr(),
        p.mixed.buf.as_ptr(),
        p.gates.buf.as_ptr(),
        quarters_in.map_or(null, |q| q.buf.as_ptr()),
        p.output.buf.as_ptr(),
        norm_weight.buf.as_ptr(),
        normalized.buf.as_ptr(),
        next.map_or(null, |n| n.norm_weight.buf.as_ptr()),
        next.map_or(null, |n| n.inject.buf.as_ptr()),
        next.map_or(null, |n| n.quarters.buf.as_ptr()),
        clear.map_or(null, |c| c.buf.as_ptr()),
    ] {
        args.push_ptr(ptr);
    }
    args.push_i32(clear.map_or(0, |c| c.numel() as i32));
    args.push_i32(4);
    args.push_i32(2560);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_write_norm_f32",
        [4, rows as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct HyperNorm<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    /// `input` holds BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
}

pub fn hyper_norm(gpu: &mut Gpu, p: &HyperNorm<'_>) -> HipResult<()> {
    hyper_norm_impl(gpu, p, std::ptr::null_mut(), false)
}

/// [`hyper_norm`] that writes the normalized rows as F16 into
/// `normalized_f16` (same element count), the F16 WMMA projections' input,
/// and with `bf16_copy` also stores `normalized` as BF16 bits (the values are
/// BF16-rounded) in the first half of its buffer: read it with
/// `HyperReadUpFused::normalized_bf16`.  Without it `normalized` is untouched.
pub fn hyper_norm_f16(
    gpu: &mut Gpu,
    p: &HyperNorm<'_>,
    normalized_f16: &GpuTensor,
    bf16_copy: bool,
) -> HipResult<()> {
    if normalized_f16.dtype != DType::F16 || normalized_f16.numel() != p.normalized.numel() {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    hyper_norm_impl(gpu, p, normalized_f16.buf.as_ptr(), bf16_copy)
}

fn hyper_norm_impl(
    gpu: &mut Gpu,
    p: &HyperNorm<'_>,
    normalized_f16: *mut std::ffi::c_void,
    bf16_copy: bool,
) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.normalized)?;
    if p.norm_weight.dtype != DType::BF16 || p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "HC norm width")?;
    let input_elements = checked_extent(p.input.numel(), "HC norm input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let branches = checked_i32(p.branches, "HC norm branch count")?;
    let hidden = checked_i32(p.hidden, "HC norm hidden width")?;
    let rows_i = checked_i32(rows, "HC norm row count")?;
    let branch_grid = checked_u32(p.branches, "HC norm branch grid")?;
    let row_grid = checked_u32(rows, "HC norm row grid")?;
    if p.norm_weight.numel() != wide || p.normalized.numel() != input_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_norm_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(if normalized_f16.is_null() || bf16_copy {
        p.normalized.buf.as_ptr()
    } else {
        std::ptr::null_mut()
    });
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.push_ptr(normalized_f16);
    args.push_i32(i32::from(p.state_bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_norm_f32",
        [branch_grid, row_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// `hyper_norm` of each row followed by its BF16 `[branches, branches *
/// hidden]` gate projection (the multi-row BF16 GEMM), fused per row without
/// writing `normalized`; bitwise identical to the two-launch sequence.
pub struct HyperNormGate<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub gate_weight: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub rows: usize,
    pub branches: usize,
    pub hidden: usize,
    /// `input` holds BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
}

impl HyperNormGate<'_> {
    /// Geometry the fused kernel supports: four branches, `hidden` a multiple
    /// of 256 up to 2560.
    pub fn supports(branches: usize, hidden: usize) -> bool {
        branches == 4 && hidden % 256 == 0 && (256..=2560).contains(&hidden)
    }
}

pub fn hyper_norm_gate(gpu: &mut Gpu, p: &HyperNormGate<'_>) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.gates)?;
    let wide = checked_product(p.branches, p.hidden, "HC norm-gate width")?;
    if p.norm_weight.dtype != DType::BF16
        || p.gate_weight.dtype != DType::BF16
        || p.rows == 0
        || !HyperNormGate::supports(p.branches, p.hidden)
        || p.norm_weight.numel() != wide
        || p.gate_weight.numel() < p.branches * wide
        || p.input.numel() < p.rows * wide
        || p.gates.numel() < p.rows * p.branches
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let hidden = checked_i32(p.hidden, "HC norm-gate hidden width")?;
    let row_grid = checked_u32(p.rows, "HC norm-gate row grid")?;
    let lds_bytes = checked_u32((wide / 2 + 4 * 256) * 4, "HC norm-gate LDS")?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_norm_gate_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(p.gate_weight.buf.as_ptr());
    args.push_ptr(p.gates.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(i32::from(p.state_bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_norm_gate_f32",
        [row_grid, 1, 1],
        [256, 1, 1],
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// H4 read side: [`hyper_norm_gate`]'s norm + BF16 gate projection of the
/// streams and [`hyper_norm_f16`]'s F16 normalized row (the HC down GEMM's
/// input) in one launch (`hyper_norm_gate_outputs`).  `gates` receives the
/// gate logits the paired HC write would have computed from the same streams;
/// the write then skips its own norm + gate launch.  Bytewise the two
/// launches' outputs (hence not the F32/BF16 `normalized` copies).
pub struct HyperNormGateOutputs<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    /// The paired write's BF16 `[branches, branches * hidden]` gate weight.
    pub gate_weight: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    /// F16 `rows * branches * hidden` elements.
    pub normalized_f16: &'a GpuTensor,
    pub rows: usize,
    pub branches: usize,
    pub hidden: usize,
    /// `input` holds BF16 bits (see [`Gpu::qwen4_bf16_streams`]).
    pub state_bf16: bool,
}

pub fn hyper_norm_gate_outputs(gpu: &mut Gpu, p: &HyperNormGateOutputs<'_>) -> HipResult<()> {
    ensure_f32(p.input)?;
    ensure_f32(p.gates)?;
    let wide = checked_product(p.branches, p.hidden, "HC norm-gate-outputs width")?;
    if p.norm_weight.dtype != DType::BF16
        || p.gate_weight.dtype != DType::BF16
        || p.normalized_f16.dtype != DType::F16
        || p.rows == 0
        || !gpu.arch_caps.has_gfx11_plus_simt()
        || !HyperNormGate::supports(p.branches, p.hidden)
        || p.norm_weight.numel() != wide
        || p.gate_weight.numel() < p.branches * wide
        || p.input.numel() < p.rows * wide
        || p.gates.numel() < p.rows * p.branches
        || p.normalized_f16.numel() < p.rows * wide
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let hidden = checked_i32(p.hidden, "HC norm-gate-outputs hidden width")?;
    let row_grid = checked_u32(p.rows, "HC norm-gate-outputs row grid")?;
    let lds_bytes = checked_u32(p.hidden * 8 + 4 * 256 * 4, "HC norm-gate-outputs LDS")?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_norm_gate_outputs")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(p.gate_weight.buf.as_ptr());
    args.push_ptr(p.gates.buf.as_ptr());
    args.push_ptr(p.normalized_f16.buf.as_ptr());
    args.push_ptr(std::ptr::null());
    args.push_ptr(std::ptr::null());
    args.push_i32(hidden);
    args.push_i32(i32::from(p.state_bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "hyper_norm_gate_outputs",
        [row_grid, 1, 1],
        [256, 1, 1],
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// gfx1151 HC read tail: the BF16 `[4 * hidden, low_rank]` up projection of
/// `low` fused with `hyper_read_projected` (four branches), bitwise
/// identical to the multi-row BF16 GEMM followed by that kernel; the
/// projected logits are never written.
pub struct HyperReadUpFused<'a> {
    pub up_weight: &'a GpuTensor,
    pub low: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub rows: usize,
    pub hidden: usize,
    pub low_rank: usize,
    /// `normalized` holds BF16 bits ([`hyper_norm_f16`]), not F32.
    pub normalized_bf16: bool,
}

/// `HIPFIRE_QWEN4_HC_UP_TWIN` (default on; `0` restores `hyper_read_up_fused_f32`
/// for every row count): on exact gfx1151 the Qwen4 HC read tail runs the
/// bitwise twin `hyper_read_up_fused_f32_sx` on batches of
/// [`QWEN4_HC_UP_TWIN_MIN_ROWS`] rows or more. Read once.
static QWEN4_HC_UP_TWIN: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| hipfire_config::developer_bool("HIPFIRE_QWEN4_HC_UP_TWIN", true));

/// Smallest row count (inclusive) that takes the `hyper_read_up_fused` twin.
const QWEN4_HC_UP_TWIN_MIN_ROWS: usize = 16;

/// Whether [`hyper_read_up_fused`] launches `hyper_read_up_fused_f32_sx`
/// instead of `hyper_read_up_fused_f32`. The twin is bitwise the incumbent: only
/// its output stores move, so the wave-uniform X loads become scalar loads.
/// Decode/verify-sized batches below 16 rows keep the incumbent.
fn qwen4_hc_up_twin_applies(enabled: bool, exact_gfx1151: bool, rows: usize) -> bool {
    enabled && exact_gfx1151 && rows >= QWEN4_HC_UP_TWIN_MIN_ROWS
}

/// Same arguments, grid, block, dynamic LDS and kernarg blob for either entry:
/// on exact gfx1151 with at least 16 rows (and `HIPFIRE_QWEN4_HC_UP_TWIN`
/// not `0`) the launch is the bitwise twin `hyper_read_up_fused_f32_sx`
/// (module `hyper_read_up_fused_twin`); otherwise it is `hyper_read_up_fused_f32`.
pub fn hyper_read_up_fused(gpu: &mut Gpu, p: &HyperReadUpFused<'_>) -> HipResult<()> {
    ensure_f32(p.low)?;
    ensure_f32(p.normalized)?;
    ensure_f32(p.mixed)?;
    let wide = checked_product(4, p.hidden, "HC read width")?;
    if p.up_weight.dtype != DType::BF16
        || p.rows == 0
        || p.hidden == 0
        || p.hidden % 8 != 0
        || p.low_rank % 8 != 0
        || !(257..=512).contains(&p.low_rank)
        || p.up_weight.numel() < wide * p.low_rank
        || p.low.numel() < p.rows * p.low_rank
        || p.normalized.numel() < p.rows * wide
        || p.mixed.numel() < p.rows * p.hidden
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let low_rank = checked_i32(p.low_rank, "HC read low rank")?;
    let rows = checked_i32(p.rows, "HC read rows")?;
    let column_grid = checked_u32(p.hidden / 8, "HC read column grid")?;
    let row_grid = checked_u32(p.rows.div_ceil(128), "HC read row grid")?;
    let lds_bytes = checked_u32(32 * (p.low_rank / 2 + 1) * 4, "HC read LDS")?;
    let (module, source, kernel) =
        if qwen4_hc_up_twin_applies(*QWEN4_HC_UP_TWIN, gpu.arch_caps.is_gfx1151(), p.rows) {
            (
                "hyper_read_up_fused_twin",
                HYPER_READ_UP_FUSED_TWIN_SRC,
                "hyper_read_up_fused_f32_sx",
            )
        } else {
            ("tensor_ops", TENSOR_OPS_SRC, "hyper_read_up_fused_f32")
        };
    gpu.ensure_kernel_public(module, source, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.up_weight.buf.as_ptr());
    args.push_ptr(p.low.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.mixed.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(low_rank);
    args.push_i32(rows);
    args.push_i32(i32::from(p.normalized_bf16));
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [column_grid, row_grid, 1],
        [256, 1, 1],
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
/// [`hyper_read_up_fused`] on gfx11 BF16 WMMA, or on gfx1201's gfx12 BF16
/// WMMA when [`Gpu::qwen4_f16_wmma_gfx1201`] admits it; `low` is packed BF16
/// ([`HcActivationFused::bf16_out`]) and `normalized` is
/// [`hyper_norm_f16`]'s F16 copy (`normalized_bf16` is ignored). Not
/// bit-exact: the logits accumulate the same exact BF16 products in WMMA's F32
/// order, so a gate occasionally rounds one BF16 step apart; the epilogue is
/// unchanged.
pub fn hyper_read_up_wmma(gpu: &mut Gpu, p: &HyperReadUpFused<'_>) -> HipResult<()> {
    let select = HcUpSelect::Auto {
        tiled: gpu.flags.qwen4_hc_up_tile_enabled(),
        v2: gpu.flags.qwen4_hc_up_v2_enabled(),
    };
    hyper_read_up_wmma_launch(gpu, p, select)
}

/// Which launch of the gfx1201 HC read tail ([`hyper_read_up_wmma_route`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcUpRoute {
    /// `hyper_read_up_wmma_bf16_gfx1201`: 512 rows per block, dynamic LDS.
    Baseline,
    /// `hyper_read_up_wmma_bf16_gfx1201_t128`: retiled, 128 rows per block
    /// (`low_rank % 64 == 0`).
    T128,
    /// `hyper_read_up_wmma_bf16_gfx1201_v2`: persistent W-stationary
    /// (`low_rank == 320`), bytewise `T128`'s output.
    V2,
}

/// Smallest row count the persistent `_v2` HC up+mix entry takes in
/// [`hyper_read_up_wmma`]; below it the retiled `_t128` entry runs.
pub const HC_UP_V2_MIN_ROWS: usize = 128;

/// Row groups (grid y) of one `_v2` launch: one per 8192 rows, 1 to 4.
pub fn hc_up_v2_groups(rows: usize) -> usize {
    rows.div_ceil(8192).clamp(1, 4)
}

/// Dynamic LDS of the `_v2` entry: the 64 x (320 + 8) BF16 weight tile and the
/// 2 x 2048-entry gate table.
const HC_UP_V2_LDS_BYTES: u32 = 64 * (320 + 8) * 2 + 16384;

/// Waves per block of the gfx1151 `_v2` entry (`HRU3_WAVES` in
/// `hyper_read_up_v2.gfx1151.hip`).
const HC_UP_V2_1151_WAVES: u32 = 8;

/// Row groups (grid y) of one gfx1151 `_v2` launch: one per 4096 rows, 1 to 4
/// (tuned on Halo: 2 groups at 8192 rows, 4 at 16384).
pub fn hc_up_v2_1151_groups(rows: usize) -> usize {
    rows.div_ceil(4096).clamp(1, 4)
}

/// Whether [`hyper_read_up_wmma`] takes the persistent `_v2` entry: exact
/// gfx1201 / gfx1151 with the tiled path admitted (`tiled`: `_t128` on gfx1201,
/// `_swap` on gfx1151), `HIPFIRE_QWEN4_HC_UP_V2` on, `low_rank == 320`,
/// `hidden % 16 == 0`, at least [`HC_UP_V2_MIN_ROWS`] rows, not recording and
/// not capturing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_up_v2_selected(
    arch: &str,
    enabled: bool,
    tiled: bool,
    low_rank: usize,
    hidden: usize,
    rows: usize,
    recording: bool,
    capturing: bool,
) -> bool {
    enabled
        && matches!(arch, "gfx1201" | "gfx1151")
        && tiled
        && low_rank == 320
        && hidden != 0
        && hidden % 16 == 0
        && rows >= HC_UP_V2_MIN_ROWS
        && !recording
        && !capturing
}

/// [`hyper_read_up_wmma`] forced onto one gfx1201 launch. Never falls back:
/// an error when the arch is not gfx1201 on its F16 WMMA arm or the route's
/// shape does not hold (`T128`: `low_rank % 64 == 0`; `V2`: `low_rank == 320`).
/// `V2` skips the flag, minimum-rows, recording and capture gates of the
/// normal route, so any `rows >= 1` runs.
pub fn hyper_read_up_wmma_route(
    gpu: &mut Gpu,
    p: &HyperReadUpFused<'_>,
    route: HcUpRoute,
) -> HipResult<()> {
    hyper_read_up_wmma_launch(gpu, p, HcUpSelect::Forced(route))
}

/// [`hyper_read_up_wmma`] on its baseline entries (`tiled == false`) or the
/// retiled operand-swapped ones (`tiled == true`, bytewise the baseline's);
/// never the `_v2` entry.
#[cfg(test)]
fn hyper_read_up_wmma_tiled(gpu: &mut Gpu, p: &HyperReadUpFused<'_>, tiled: bool) -> HipResult<()> {
    hyper_read_up_wmma_launch(gpu, p, HcUpSelect::Auto { tiled, v2: false })
}

#[derive(Clone, Copy)]
enum HcUpSelect {
    Auto { tiled: bool, v2: bool },
    Forced(HcUpRoute),
}

fn hyper_read_up_wmma_launch(
    gpu: &mut Gpu,
    p: &HyperReadUpFused<'_>,
    select: HcUpSelect,
) -> HipResult<()> {
    ensure_f32(p.mixed)?;
    let wide = checked_product(4, p.hidden, "HC read width")?;
    let wrong_shape = || HipError::new(0, &ComputeError::WrongShape.to_string());
    let (module, source, entry) = if gpu.arch_caps.has_wmma_w32() {
        (
            "hyper_read_up_wmma",
            HYPER_READ_UP_WMMA_SRC,
            "hyper_read_up_wmma_bf16",
        )
    } else if gpu.qwen4_f16_wmma_gfx1201() {
        (
            "hyper_read_up_wmma_gfx1201",
            HYPER_READ_UP_WMMA_GFX1201_SRC,
            "hyper_read_up_wmma_bf16_gfx1201",
        )
    } else {
        return Err(wrong_shape());
    };
    let is_1201 = entry == "hyper_read_up_wmma_bf16_gfx1201";
    let is_1151 = gpu.arch == "gfx1151" && entry == "hyper_read_up_wmma_bf16";
    if matches!(select, HcUpSelect::Forced(_)) && !is_1201 && !is_1151 {
        return Err(wrong_shape());
    }
    if p.low.dtype != DType::BF16
        || p.normalized.dtype != DType::F16
        || p.up_weight.dtype != DType::BF16
        || p.rows == 0
        || p.hidden % 16 != 0
        || p.low_rank % 16 != 0
        || p.low_rank > 504
        || p.up_weight.numel() < wide * p.low_rank
        || p.low.numel() < p.rows * p.low_rank
        || p.normalized.numel() < p.rows * wide
        || p.mixed.numel() < p.rows * p.hidden
    {
        return Err(wrong_shape());
    }
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let low_rank = checked_i32(p.low_rank, "HC read low rank")?;
    let rows = checked_i32(p.rows, "HC read rows")?;
    let column_grid = checked_u32(p.hidden / 16, "HC read column grid")?;
    // H3 (`HIPFIRE_QWEN4_HC_UP_TILE`, exact gfx1151 / gfx1201): the retiled
    // operand-swapped entries.  gfx1151 keeps the baseline launch geometry;
    // gfx1201's takes 128 rows per block, static LDS only, `low_rank % 64 == 0`.
    let tiled = match select {
        HcUpSelect::Auto { tiled, .. } => tiled,
        HcUpSelect::Forced(route) => route == HcUpRoute::T128,
    };
    let tile_1201 = tiled && is_1201 && p.low_rank % 64 == 0;
    // `_v2` (`HIPFIRE_QWEN4_HC_UP_V2`): persistent W-stationary entry, bytewise the
    // tiled route's (`_t128` on gfx1201, `_swap` on gfx1151).
    let v2 = match select {
        HcUpSelect::Auto { v2, .. } => hc_up_v2_selected(
            gpu.arch.as_str(),
            v2,
            tile_1201 || (is_1151 && tiled),
            p.low_rank,
            p.hidden,
            p.rows,
            gpu.replay.is_recording(),
            gpu.graphs.capture_mode,
        ),
        HcUpSelect::Forced(route) => route == HcUpRoute::V2,
    };
    match select {
        HcUpSelect::Forced(HcUpRoute::T128) if !tile_1201 && !is_1151 => {
            return Err(wrong_shape());
        }
        HcUpSelect::Forced(HcUpRoute::V2) if p.low_rank != 320 => return Err(wrong_shape()),
        _ => {}
    }
    let entry = match (tiled, entry) {
        _ if v2 && is_1151 => "hyper_read_up_wmma_bf16_gfx1151_v2",
        _ if v2 => "hyper_read_up_wmma_bf16_gfx1201_v2",
        (true, "hyper_read_up_wmma_bf16") => "hyper_read_up_wmma_bf16_swap",
        (true, "hyper_read_up_wmma_bf16_gfx1201") if tile_1201 => {
            "hyper_read_up_wmma_bf16_gfx1201_t128"
        }
        (_, entry) => entry,
    };
    let (grid, block, lds_bytes) = if v2 && is_1151 {
        let groups = checked_u32(hc_up_v2_1151_groups(p.rows), "HC read row groups")?;
        (
            [column_grid, groups, 1],
            [HC_UP_V2_1151_WAVES * 32, 1, 1],
            HC_UP_V2_LDS_BYTES,
        )
    } else if v2 {
        let groups = checked_u32(hc_up_v2_groups(p.rows), "HC read row groups")?;
        ([column_grid, groups, 1], [512, 1, 1], HC_UP_V2_LDS_BYTES)
    } else if tile_1201 {
        let row_grid = checked_u32(p.rows.div_ceil(128), "HC read row grid")?;
        ([column_grid, row_grid, 1], [256, 1, 1], 0)
    } else {
        let row_grid = checked_u32(p.rows.div_ceil(512), "HC read row grid")?;
        let lds = checked_u32(64 * (p.low_rank + 8) * 2, "HC read LDS")?;
        ([column_grid, row_grid, 1], [256, 1, 1], lds)
    };
    let (module, source) = if v2 && is_1151 {
        (
            "hyper_read_up_wmma_gfx1151_v2",
            HYPER_READ_UP_WMMA_GFX1151_V2_SRC,
        )
    } else {
        (module, source)
    };
    gpu.ensure_kernel_public(module, source, entry)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.up_weight.buf.as_ptr());
    args.push_ptr(p.low.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.mixed.buf.as_ptr());
    args.push_i32(hidden);
    args.push_i32(low_rank);
    args.push_i32(rows);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        entry,
        grid,
        block,
        lds_bytes,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct GatedDeltaConv<'a> {
    pub input: &'a GpuTensor,
    pub kernel: &'a GpuTensor,
    pub history: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub next_history: &'a GpuTensor,
    pub channels: usize,
    pub history_rows: usize,
    pub kernel_size: usize,
    pub cursor: usize,
    /// Index of this row inside the chunk whose start position the tape replays
    /// at. The caller derives `cursor = (start_position + row_index) %
    /// history_rows`, so the declared replay binding re-derives exactly that.
    pub row_index: usize,
}

pub fn gated_delta_conv(gpu: &mut Gpu, p: &GatedDeltaConv<'_>) -> HipResult<()> {
    gated_delta_conv_launch(gpu, p, None)
}

/// [`gated_delta_conv`] and [`gated_delta_params`] of one decode row in one
/// launch (`heads <= 256`); both outputs are bitwise the two launches.
pub fn gated_delta_conv_params(
    gpu: &mut Gpu,
    p: &GatedDeltaConv<'_>,
    params: &GatedDeltaParams<'_>,
    heads: usize,
) -> HipResult<()> {
    gated_delta_conv_launch(gpu, p, Some((params, heads)))
}

/// The per-launch shape checks of one GDN decode convolution: the i32
/// arguments (channels, history rows, kernel width, cursor).
fn validate_gated_delta_conv(p: &GatedDeltaConv<'_>) -> HipResult<[i32; 4]> {
    for tensor in [p.input, p.history, p.output, p.next_history] {
        ensure_f32(tensor)?;
    }
    if p.kernel.dtype != DType::BF16 || p.channels == 0 || p.kernel_size == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let expected_history_rows = p.kernel_size - 1;
    let kernel_elements =
        checked_product(p.channels, p.kernel_size, "GDN convolution kernel extent")?;
    let history_elements = checked_product(
        p.channels,
        expected_history_rows,
        "GDN convolution history extent",
    )?;
    if p.history_rows != expected_history_rows
        || p.cursor >= expected_history_rows.max(1)
        || p.input.numel() != p.channels
        || p.kernel.numel() != kernel_elements
        || p.history.numel() != history_elements
        || p.output.numel() != p.channels
        || p.next_history.numel() != history_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok([
        checked_i32(p.channels, "GDN convolution channels")?,
        checked_i32(expected_history_rows, "GDN convolution history rows")?,
        checked_i32(p.kernel_size, "GDN convolution kernel width")?,
        checked_i32(p.cursor, "GDN convolution cursor")?,
    ])
}

fn gated_delta_conv_launch(
    gpu: &mut Gpu,
    p: &GatedDeltaConv<'_>,
    params: Option<(&GatedDeltaParams<'_>, usize)>,
) -> HipResult<()> {
    let [channels, history_rows, kernel_size, cursor] = validate_gated_delta_conv(p)?;
    let conv_blocks = blocks(p.channels)?;
    let (kernel_name, grid) = match params {
        None => ("gated_delta_conv_bf16_f32", conv_blocks),
        Some(_) => ("gated_delta_conv_params_bf16_f32", conv_blocks + 1),
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(cursor);
    // The offset the scalar actually landed at, not a hand-counted layout
    // constant: a changed argument list cannot silently move the binding.
    let cursor_offset = args.len() - 4;
    if let Some((params, heads)) = params {
        let heads_i = validate_gated_delta_params(params, heads)?;
        if heads > 256 {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
        for tensor in [
            params.a,
            params.b,
            params.a_log,
            params.dt_bias,
            params.gate,
            params.beta,
        ] {
            args.push_ptr(tensor.buf.as_ptr());
        }
        args.push_i32(heads_i);
        args.push_i32(checked_i32(conv_blocks as usize, "GDN convolution blocks")?);
    }
    args.pad_to(16);
    // The cursor is `(start_position + row_index) % history_rows` by
    // construction, so it is a declared dynamic field rather than an unexplained
    // kernarg difference: replay re-derives it instead of replaying the
    // capture-position ring slot.
    //
    // Kernel width 1 (`history_rows == 0`) has no ring to index: the cursor is
    // the constant 0, and a modulo binding would be meaningless (and rejected),
    // so nothing is declared. The placeholder below is never handed to the
    // recorder in that case.
    let cursor_binding = [crate::replay::ReplayKernargBinding::PositionModU32 {
        offset: cursor_offset,
        addend: u32::try_from(p.row_index)
            .map_err(|_| HipError::new(0, "GDN convolution row index exceeds u32"))?,
        modulus: u32::try_from(p.history_rows.max(1))
            .map_err(|_| HipError::new(0, "GDN convolution history rows exceed u32"))?,
    }];
    let declared: &[crate::replay::ReplayKernargBinding] = if p.history_rows > 0 {
        &cursor_binding
    } else {
        &[]
    };
    gpu.launch_blob_recorded(
        kernel_name,
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: declared,
        },
    )
}

/// `tensor_ops.hip` plus the multi-lane GDN core kernels (`gdn_lanes.hip`):
/// the lane kernels inline the same device bodies under the same flags, and
/// the `tensor_ops` module's source stays untouched.
pub(crate) const GDN_LANES_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip"),
    include_str!("../../../kernels/src/gdn_lanes.hip")
);

/// Most lanes one [`gated_delta_core_lanes`] call covers (`GDN_LANES_MAX`).
pub const GDN_CORE_LANES_MAX: usize = 8;

/// One AR lane's one-row GDN core: the arguments of its per-lane
/// [`gated_delta_conv_params`] and unrotated [`gated_delta_step_gated`].
pub struct GdnCoreLane<'a> {
    pub conv: GatedDeltaConv<'a>,
    pub params: GatedDeltaParams<'a>,
    pub step: GatedDeltaStep<'a>,
    pub gate: GatedDeltaGate<'a>,
}

/// A validated, admitted set of 2..=[`GDN_CORE_LANES_MAX`] lanes, ready for
/// [`GdnCoreLanesPlan::launch`]; produced only by [`gated_delta_core_lanes`].
pub struct GdnCoreLanesPlan<'a, 'l> {
    lanes: &'a [GdnCoreLane<'l>],
    heads: usize,
    /// `[channels, history_rows, kernel_size]` of the shared convolution.
    conv: [i32; 3],
    /// `[key_heads, value_heads, key_dim, value_dim]` of the shared step.
    step: [i32; 4],
}

/// Admission of the per-lane [`gated_delta_conv_params`] + unrotated
/// [`gated_delta_step_gated`] calls of 2..=[`GDN_CORE_LANES_MAX`] lanes as two
/// launches (see [`GdnCoreLanesPlan::launch`]). Lanes share the weights and
/// shapes, hold Q8 state slots on the gfx11+ 128x128 route, and each step reads
/// q | k | v contiguously from its convolution output. Every shape,
/// route and flag refusal happens here, before anything launches: `Ok(None)`
/// means the set does not take this route (nothing launched; run the lanes one
/// by one), so a caller that must act exactly once before the launch (the
/// interpreter's per-step hooks) does so only when a plan comes back. Never
/// while a replay tape records: the per-lane cursor and Q8 frame are not
/// declared to the recorder.
pub fn gated_delta_core_lanes<'a, 'l>(
    gpu: &Gpu,
    lanes: &'a [GdnCoreLane<'l>],
    heads: usize,
) -> HipResult<Option<GdnCoreLanesPlan<'a, 'l>>> {
    let Some(first) = lanes.first() else {
        return Ok(None);
    };
    if lanes.len() < 2
        || lanes.len() > GDN_CORE_LANES_MAX
        || gpu.replay.is_recording()
        || !gpu.flags.qwen4_gdn_lanes_enabled()
    {
        return Ok(None);
    }
    let (s0, c0, p0, g0) = (&first.step, &first.conv, &first.params, &first.gate);
    let qk = s0.key_heads * s0.key_dim;
    let mut shape: Option<([i32; 3], [i32; 4])> = None;
    for lane in lanes {
        let [channels, history_rows, kernel_size, _cursor] = validate_gated_delta_conv(&lane.conv)?;
        validate_gated_delta_params(&lane.params, heads)?;
        let (format, step_args) = validate_gated_delta_step(&lane.step)?;
        validate_gated_delta_gate(&lane.gate)?;
        let (c, p, s, g) = (&lane.conv, &lane.params, &lane.step, &lane.gate);
        let f32_at = |t: &GpuTensor, elements: usize| {
            c.output.buf.as_ptr() as usize + elements * 4 == t.buf.as_ptr() as usize
        };
        let fits = format == GdnStateFormat::Q8
            && s.key_dim == 128
            && s.value_dim == 128
            && heads == s.value_heads
            && heads <= 256
            && c.next_history.buf.as_ptr() == c.history.buf.as_ptr()
            && g.recurrent_output.buf.as_ptr() == s.output.buf.as_ptr()
            && s.gate.buf.as_ptr() == p.gate.buf.as_ptr()
            && s.beta.buf.as_ptr() == p.beta.buf.as_ptr()
            && f32_at(s.q, 0)
            && f32_at(s.k, qk)
            && f32_at(s.v, 2 * qk)
            && c.kernel.buf.as_ptr() == c0.kernel.buf.as_ptr()
            && p.a_log.buf.as_ptr() == p0.a_log.buf.as_ptr()
            && p.dt_bias.buf.as_ptr() == p0.dt_bias.buf.as_ptr()
            && g.norm.buf.as_ptr() == g0.norm.buf.as_ptr();
        let lane_shape = ([channels, history_rows, kernel_size], step_args);
        if !fits || *shape.get_or_insert(lane_shape) != lane_shape {
            return Ok(None);
        }
    }
    let Some((conv, step)) = shape else {
        return Ok(None);
    };
    Ok(Some(GdnCoreLanesPlan {
        lanes,
        heads,
        conv,
        step,
    }))
}

impl GdnCoreLanesPlan<'_, '_> {
    /// The admitted lanes as two launches (convolution + gate parameters, then
    /// step + gated norm): each lane's blocks run the same bodies on its own
    /// pointers, so every output and state byte equals the per-lane calls'.
    pub fn launch(self, gpu: &mut Gpu) -> HipResult<()> {
        let Self {
            lanes,
            heads,
            conv: [channels, history_rows, kernel_size],
            step: [key_heads, value_heads, key_dim, value_dim],
        } = self;
        let first = &lanes[0];
        let (s0, c0, p0, g0) = (&first.step, &first.conv, &first.params, &first.gate);
        let qk = s0.key_heads * s0.key_dim;
        let n = lanes.len();
        let pad = |args: &mut KernargBlob, filled: usize, ptr: bool| {
            for _ in filled..GDN_CORE_LANES_MAX {
                if ptr {
                    args.push_ptr(std::ptr::null());
                } else {
                    args.push_u32(0);
                }
            }
        };

        let conv_blocks = blocks(c0.channels)?;
        gpu.ensure_kernel_public(
            "gdn_lanes",
            GDN_LANES_SRC,
            "gated_delta_conv_params_bf16_f32_lanes",
        )?;
        let mut args = KernargBlob::new();
        for tensor in [c0.kernel, p0.a_log, p0.dt_bias] {
            args.push_ptr(tensor.buf.as_ptr());
        }
        for v in [
            channels,
            history_rows,
            kernel_size,
            checked_i32(heads, "GDN lane heads")?,
        ] {
            args.push_i32(v);
        }
        args.push_i32(checked_i32(
            conv_blocks as usize,
            "GDN lane convolution blocks",
        )?);
        // GdnConvLanes: seven pointer arrays, then the cursors.
        let conv_fields: [fn(&GdnCoreLane<'_>) -> *mut std::ffi::c_void; 7] = [
            |l| l.conv.input.buf.as_ptr(),
            |l| l.conv.output.buf.as_ptr(),
            |l| l.conv.history.buf.as_ptr(),
            |l| l.params.a.buf.as_ptr(),
            |l| l.params.b.buf.as_ptr(),
            |l| l.params.gate.buf.as_ptr(),
            |l| l.params.beta.buf.as_ptr(),
        ];
        for field in conv_fields {
            for lane in lanes {
                args.push_ptr(field(lane));
            }
            pad(&mut args, n, true);
        }
        for lane in lanes {
            args.push_i32(checked_i32(lane.conv.cursor, "GDN lane cursor")?);
        }
        pad(&mut args, n, false);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            "gated_delta_conv_params_bf16_f32_lanes",
            [conv_blocks + 1, n as u32, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        gpu.ensure_kernel_public(
            "gdn_lanes",
            GDN_LANES_SRC,
            "gated_delta_step_gate_norm128_q8_lanes",
        )?;
        let mut args = KernargBlob::new();
        args.push_ptr(g0.norm.buf.as_ptr());
        for v in [key_heads, value_heads, key_dim, value_dim] {
            args.push_i32(v);
        }
        args.push_f32((s0.key_dim as f32).sqrt().recip());
        args.push_i32(checked_i32(qk, "GDN lane q width")?);
        // GdnStepLanes: six pointer arrays, then the Q8 frames.
        let step_fields: [fn(&GdnCoreLane<'_>) -> *mut std::ffi::c_void; 6] = [
            |l| l.step.q.buf.as_ptr(),
            |l| l.step.gate.buf.as_ptr(),
            |l| l.step.beta.buf.as_ptr(),
            |l| l.step.state.buf.as_ptr(),
            |l| l.gate.z.buf.as_ptr(),
            |l| l.gate.output.buf.as_ptr(),
        ];
        for field in step_fields {
            for lane in lanes {
                args.push_ptr(field(lane));
            }
            pad(&mut args, n, true);
        }
        for lane in lanes {
            args.push_u32(checked_u32(lane.step.position, "GDN lane Q8 frame")?);
        }
        pad(&mut args, n, false);
        args.pad_to(16);
        gpu.launch_blob_recorded(
            "gated_delta_step_gate_norm128_q8_lanes",
            [
                checked_u32(s0.value_heads, "GDN lane value-head grid")?,
                1,
                n as u32,
            ],
            [128, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
        Ok(())
    }
}

/// Ordered K=4 causal convolution over row-major `[rows, channels]` input.
/// Each channel keeps its BF16-rounded history ring across the whole batch.
/// A BF16-typed `output` receives the (BF16-rounded) values as packed BF16.
pub struct GatedDeltaConvBatched<'a> {
    pub input: &'a GpuTensor,
    pub kernel: &'a GpuTensor,
    pub history: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub next_history: &'a GpuTensor,
    pub rows: usize,
    pub channels: usize,
    pub history_rows: usize,
    pub kernel_size: usize,
    pub start_cursor: usize,
}

pub fn gated_delta_conv_batched(gpu: &mut Gpu, p: &GatedDeltaConvBatched<'_>) -> HipResult<()> {
    gated_delta_conv_batched_impl(gpu, p, None, None)
}

/// [`gated_delta_conv_batched`] and [`gated_delta_params_batched`] in one
/// launch (the parameter blocks follow the convolution grid); bitwise both.
pub fn gated_delta_conv_params_batched(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
) -> HipResult<()> {
    validate_gated_delta_params_batched(params)?;
    gated_delta_conv_batched_impl(gpu, conv, Some(params), None)
}

/// [`gated_delta_conv_params_batched`] that also stores the Q/K the chunked
/// recurrence consumes already normalized
/// (`gated_delta_conv_qknorm_bf16_f32_batched_k4`): the convolution values
/// and parameters are bitwise the plain launch's, and the normalized Q/K are
/// bitwise what `gated_delta_qk_norm_bf16_batched` stores, written into the
/// shared F16 activation scratch ([`Gpu::qwen4_f16_x_scratch`]) where
/// [`gated_delta_step_gate_wmma_qknormed`] reads them.  `step` is the
/// recurrence this convolution feeds; it must satisfy
/// [`gated_delta_conv_qknorm_geometry`].  Without `keep_qk_conv` the Q/K
/// columns of `conv.output` are NOT written (the chunk kernel reads only V
/// from it): a caller that consumes them (debug / capture) passes `true`,
/// or stays on [`gated_delta_conv_params_batched`] and
/// [`gated_delta_step_gate_wmma`].
pub fn gated_delta_conv_params_qknorm_batched(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    step: &GatedDeltaStepBatched<'_>,
    keep_qk_conv: bool,
) -> HipResult<()> {
    if !gated_delta_conv_qknorm_geometry(gpu, step, conv) {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    validate_gated_delta_params_batched(params)?;
    let qk = checked_product(step.key_heads, step.key_dim, "GDN chunk qk extent")?;
    let scratch = gpu.qwen4_f16_x_scratch(2 * step.rows * qk)?;
    let qn = scratch.buf.as_ptr();
    let kn = unsafe { (qn as *mut u8).add(step.rows * qk * 2) } as *mut std::ffi::c_void;
    // The kernel stores qn / kn as 8-byte vectors.
    if (qn as usize) % 8 != 0 || (kn as usize) % 8 != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gated_delta_conv_batched_impl(
        gpu,
        conv,
        Some(params),
        Some(ConvQkNorm {
            qn,
            kn,
            keep_qk_conv,
        }),
    )
}

/// H5's extra outputs: the normalized Q/K (`[rows, 16, 128]` BF16 each).
#[derive(Clone, Copy)]
struct ConvQkNorm {
    qn: *mut std::ffi::c_void,
    kn: *mut std::ffi::c_void,
    /// Also store the unnormalized Q/K convolution into the output.
    keep_qk_conv: bool,
}

fn gated_delta_conv_batched_impl(
    gpu: &mut Gpu,
    p: &GatedDeltaConvBatched<'_>,
    params: Option<&GatedDeltaParamsBatched<'_>>,
    qk_norm: Option<ConvQkNorm>,
) -> HipResult<()> {
    for tensor in [p.history, p.next_history] {
        ensure_f32(tensor)?;
    }
    // Input and output are each F32 or BF16 bits (BF16-rounded values).
    let output_bf16 = p.output.dtype == DType::BF16;
    if !output_bf16 {
        ensure_f32(p.output)?;
    }
    let input_bf16 = p.input.dtype == DType::BF16;
    if !input_bf16 {
        ensure_f32(p.input)?;
    }
    if p.kernel.dtype != DType::BF16
        || p.rows == 0
        || p.channels == 0
        || p.history_rows != 3
        || p.kernel_size != 4
        || p.start_cursor >= p.history_rows
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let input_elements = checked_product(p.rows, p.channels, "GDN batched convolution input")?;
    let history_elements = checked_product(
        p.channels,
        p.history_rows,
        "GDN batched convolution history",
    )?;
    let kernel_elements =
        checked_product(p.channels, p.kernel_size, "GDN batched convolution kernel")?;
    if p.input.numel() != input_elements
        || p.output.numel() != input_elements
        || p.history.numel() != history_elements
        || p.next_history.numel() != history_elements
        || p.kernel.numel() != kernel_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "GDN batched convolution rows")?;
    let channels = checked_i32(p.channels, "GDN batched convolution channels")?;
    let history_rows = checked_i32(p.history_rows, "GDN batched convolution history rows")?;
    let kernel_size = checked_i32(p.kernel_size, "GDN batched convolution kernel width")?;
    let start_cursor = checked_i32(p.start_cursor, "GDN batched convolution cursor")?;
    let grid = blocks(p.channels)?;
    let row_grid = checked_u32(p.rows.div_ceil(16), "GDN batched convolution row grid")?;
    let kernel = if qk_norm.is_some() {
        "gated_delta_conv_qknorm_bf16_f32_batched_k4"
    } else {
        "gated_delta_conv_bf16_f32_batched_k4"
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(start_cursor);
    let start_cursor_offset = args.len() - 4;
    // H5's `output_bf16` is a flag word: bit 0 = packed BF16 output, bit 1 =
    // also store the unnormalized Q/K convolution (debug / capture).
    args.push_i32(i32::from(output_bf16) | qk_norm.map_or(0, |n| 2 * i32::from(n.keep_qk_conv)));
    args.push_i32(i32::from(input_bf16));
    let (param_elements, param_heads) = match params {
        Some(q) => {
            for tensor in [q.a, q.b, q.a_log, q.dt_bias, q.gate, q.beta] {
                args.push_ptr(tensor.buf.as_ptr());
            }
            (q.rows * q.heads, q.heads)
        }
        None => {
            for _ in 0..6 {
                args.push_ptr(std::ptr::null());
            }
            (0, 1)
        }
    };
    args.push_i32(checked_i32(param_elements, "GDN batched parameter extent")?);
    args.push_i32(checked_i32(param_heads, "GDN batched parameter heads")?);
    if let Some(norm) = qk_norm {
        args.push_ptr(norm.qn);
        args.push_ptr(norm.kn);
    }
    args.pad_to(16);
    let param_grid = checked_u32(param_elements.div_ceil(256), "GDN batched parameter grid")?;
    // Compact 1-D grid: `conv_blocks * row_grid` convolution tiles (block `i`
    // is channel block `i % conv_blocks`, row block `i / conv_blocks`), then the
    // parameter blocks.  The kernel re-derives the split from rows / channels,
    // so the grid depends only on those and the heads, fixed at capture.
    let conv_tiles = checked_product(
        grid as usize,
        row_grid as usize,
        "GDN batched convolution tile grid",
    )?;
    let total_grid = checked_u32(
        checked_add(
            conv_tiles,
            param_grid as usize,
            "GDN batched convolution launch grid",
        )?,
        "GDN batched convolution launch grid",
    )?;
    // `start_cursor` is `start_position % history_rows` for the chunk (the
    // kernel advances the ring per row from there), so the declared binding
    // re-derives it at the replay position.
    let start_cursor_binding = [crate::replay::ReplayKernargBinding::PositionModU32 {
        offset: start_cursor_offset,
        addend: 0,
        modulus: u32::try_from(p.history_rows)
            .map_err(|_| HipError::new(0, "GDN batched convolution history rows exceed u32"))?,
    }];
    gpu.launch_blob_recorded(
        kernel,
        [total_grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &start_cursor_binding,
        },
    )
}

pub struct GatedDeltaParams<'a> {
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
}

pub fn gated_delta_params(gpu: &mut Gpu, p: &GatedDeltaParams<'_>, heads: usize) -> HipResult<()> {
    let heads_i = validate_gated_delta_params(p, heads)?;
    let grid = blocks(heads)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_params_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_bf16_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

fn validate_gated_delta_params(p: &GatedDeltaParams<'_>, heads: usize) -> HipResult<i32> {
    for tensor in [p.a, p.b, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if heads == 0
        || p.a_log.dtype != DType::BF16
        || p.dt_bias.dtype != DType::BF16
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    checked_i32(heads, "GDN parameter head count")
}
/// Row-batched parameter expansion for the exact Qwen4 prefill route.
pub struct GatedDeltaParamsBatched<'a> {
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub rows: usize,
    pub heads: usize,
}

fn validate_gated_delta_params_batched(p: &GatedDeltaParamsBatched<'_>) -> HipResult<usize> {
    for tensor in [p.a, p.b, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if p.rows == 0 || p.heads == 0 || p.a_log.dtype != DType::BF16 || p.dt_bias.dtype != DType::BF16
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements = checked_product(p.rows, p.heads, "GDN batched parameter extent")?;
    if p.a.numel() != elements
        || p.b.numel() != elements
        || p.gate.numel() != elements
        || p.beta.numel() != elements
        || p.a_log.numel() != p.heads
        || p.dt_bias.numel() != p.heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok(elements)
}

pub fn gated_delta_params_batched(gpu: &mut Gpu, p: &GatedDeltaParamsBatched<'_>) -> HipResult<()> {
    let elements = validate_gated_delta_params_batched(p)?;
    let rows = checked_i32(p.rows, "GDN batched parameter rows")?;
    let heads = checked_i32(p.heads, "GDN batched parameter heads")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "gated_delta_params_bf16_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(heads);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_bf16_f32_batched",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn gated_delta_params_f32(
    gpu: &mut Gpu,
    p: &GatedDeltaParams<'_>,
    heads: usize,
) -> HipResult<()> {
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if heads == 0
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let heads_i = checked_i32(heads, "GDN parameter head count")?;
    let grid = blocks(heads)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_params_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads_i);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_params_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub struct GatedDeltaGate<'a> {
    pub recurrent_output: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub value_heads: usize,
    pub value_dim: usize,
}

pub fn gated_delta_gate(gpu: &mut Gpu, p: &GatedDeltaGate<'_>) -> HipResult<()> {
    let (value_heads, value_dim) = validate_gated_delta_gate(p)?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN gate head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_gate_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_gate_bf16_f32",
        [value_heads_grid, value_dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

fn validate_gated_delta_gate(p: &GatedDeltaGate<'_>) -> HipResult<(i32, i32)> {
    for tensor in [p.recurrent_output, p.z, p.output] {
        ensure_f32(tensor)?;
    }
    if p.norm.dtype != DType::BF16 || p.value_heads == 0 || p.value_dim == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements = checked_product(p.value_heads, p.value_dim, "GDN gate extent")?;
    if p.recurrent_output.numel() != elements
        || p.z.numel() != elements
        || p.norm.numel() != p.value_dim
        || p.output.numel() != elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok((
        checked_i32(p.value_heads, "GDN gate head count")?,
        checked_i32(p.value_dim, "GDN gate width")?,
    ))
}
/// Row-batched gated RMSNorm with the exact BF16 recurrent boundary folded
/// into the kernel.
pub struct GatedDeltaGateBatched<'a> {
    pub recurrent_output: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub value_heads: usize,
    pub value_dim: usize,
}

fn validate_gated_delta_gate_batched(p: &GatedDeltaGateBatched<'_>) -> HipResult<usize> {
    for tensor in [p.recurrent_output, p.z, p.output] {
        ensure_f32(tensor)?;
    }
    if p.norm.dtype != DType::BF16 || p.rows == 0 || p.value_heads == 0 || p.value_dim != 128 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let row_elements = checked_product(p.value_heads, p.value_dim, "GDN batched gate row extent")?;
    let elements = checked_product(p.rows, row_elements, "GDN batched gate extent")?;
    if p.recurrent_output.numel() != elements
        || p.z.numel() != elements
        || p.norm.numel() != p.value_dim
        || p.output.numel() != elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    Ok(elements)
}

pub fn gated_delta_gate_batched(gpu: &mut Gpu, p: &GatedDeltaGateBatched<'_>) -> HipResult<()> {
    let elements = validate_gated_delta_gate_batched(p)?;
    let rows = checked_i32(p.rows, "GDN batched gate rows")?;
    let value_heads = checked_i32(p.value_heads, "GDN batched gate heads")?;
    let value_dim = checked_i32(p.value_dim, "GDN batched gate width")?;
    let value_heads_grid = checked_u32(elements / p.value_dim, "GDN batched gate grid")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "gated_delta_gate_bf16_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "gated_delta_gate_bf16_f32_batched",
        [value_heads_grid, 1, 1],
        [32, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// [`gated_delta_gate_batched`] (even `value_heads`) that also writes
/// `mq_rotate_x(p.output)` into `rotated` (head pairs form the 256-wide
/// groups); the next matching `Gpu::rotate_x_mq_batched` then skips
/// (`ScratchState::prerotated`).
pub fn gated_delta_gate_batched_rotate(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    rotated: &GpuTensor,
) -> HipResult<()> {
    const KERNEL: &str = "gated_delta_gate_rotate_bf16_f32_batched";
    let elements = validate_gated_delta_gate_batched(p)?;
    ensure_f32(rotated)?;
    if !p.value_heads.is_multiple_of(2) || rotated.numel() < elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, KERNEL)?;
    gpu.ensure_mq_signs()?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output, rotated] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
    args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
    args.pad_to(16);
    gpu.launch_blob_recorded(
        KERNEL,
        [
            checked_u32(elements / 256, "GDN batched gate pair grid")?,
            1,
            1,
        ],
        [64, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    gpu.scratch.prerotated = Some((
        p.output.buf.as_ptr() as usize,
        rotated.buf.as_ptr() as usize,
        elements,
    ));
    Ok(())
}
pub struct IndexedAttentionNormRope<'a> {
    pub values: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub heads: usize,
    pub head_dim: usize,
    /// Distance, in elements, between consecutive query/key heads.
    pub head_stride: usize,
    pub position: usize,
    pub rotary_dim: usize,
    /// [`crate::rope::Rope::yarn`] of the model's RoPE at `rotary_dim`.
    pub yarn: crate::rope::RopeYarn,
}

/// One operand of a QSA append/norm/pool launch checked under
/// `qwen4_qsa_extent_check`: arena label, tensor, byte offset into the tensor and
/// the byte length the kernel actually accesses (guarded, not grid-rounded).
type QsaAppendExtent<'a> = (&'static str, &'a GpuTensor, usize, usize);

/// Pre-launch extent check of every operand (callers gate the call and the
/// computation of `extents` on `gpu.flags.qwen4_qsa_extent_check`).
fn qsa_append_extent_check(
    gpu: &Gpu,
    extents: &[QsaAppendExtent<'_>],
    depth: crate::dispatch::QsaExtentDepth,
    site: &'static str,
) {
    for &(arena, tensor, offset, len) in extents {
        gpu.qsa_extent_check(arena, tensor, offset, len, depth, site);
    }
}

/// Post-launch helper for every operand, called immediately after the launch
/// (mode 2 synchronizes inside the helper).
fn qsa_append_extent_post(
    gpu: &Gpu,
    extents: &[QsaAppendExtent<'_>],
    depth: crate::dispatch::QsaExtentDepth,
    site: &'static str,
) {
    for &(arena, tensor, offset, len) in extents {
        gpu.qsa_extent_post_launch(arena, tensor, offset, len, depth, site);
    }
}

pub fn indexed_attention_norm_rope(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRope<'_>,
) -> HipResult<()> {
    ensure_f32(p.values)?;
    if p.norm.dtype != DType::BF16
        || p.heads == 0
        || p.head_dim == 0
        || p.head_dim > 256
        || p.head_stride < p.head_dim
        || p.rotary_dim == 0
        || p.rotary_dim % 2 != 0
        || p.rotary_dim > p.head_dim
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let span = checked_product(p.heads - 1, p.head_stride, "indexed attention head span")
        .and_then(|base| checked_add(base, p.head_dim, "indexed attention head span"))?;
    let heads = checked_i32(p.heads, "indexed attention head count")?;
    let head_dim = checked_i32(p.head_dim, "indexed attention head width")?;
    let head_stride = checked_i32(p.head_stride, "indexed attention head stride")?;
    let position = checked_i32(p.position, "indexed attention position")?;
    let rotary_dim = checked_i32(p.rotary_dim, "indexed attention rotary width")?;
    let head_grid = checked_u32(p.heads, "indexed attention head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.values.numel() < span || p.norm.numel() != p.head_dim {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_norm_rope_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_ptr(p.norm.buf.as_ptr());
    args.push_i32(heads);
    args.push_i32(head_dim);
    args.push_i32(head_stride);
    args.push_i32(position);
    args.push_i32(rotary_dim);
    p.yarn.push(&mut args);
    args.pad_to(16);
    // Extents: grid `[heads, blocks(head_dim)]` (`blocks(head_dim)` is 1 for
    // `head_dim <= 256`; the 256-thread round-up never passes the `channel <
    // head_dim` guard before every access). Head `h` touches `h * head_stride +
    // channel`, so the guarded extent is `span` = `(heads - 1) * head_stride +
    // head_dim` F32 of `values`, plus `head_dim` BF16 of `norm` (RoPE partners
    // are `< rotary_dim <= head_dim`).
    let extents = || {
        (
            [
                ("qsa_norm_rope_values", p.values, 0usize, span * 4),
                ("qsa_norm_rope_norm", p.norm, 0, p.head_dim * 2),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: 1,
                end_position: p.position + 1,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_norm_rope");
    }
    gpu.launch_blob_recorded(
        "indexed_attention_norm_rope_f32",
        [head_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_norm_rope");
    }
    Ok(())
}

pub struct IndexedAttentionNormRopeBatch<'a> {
    pub values: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub rows: usize,
    pub row_stride: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub head_stride: usize,
    pub position_start: usize,
    pub rotary_dim: usize,
    /// [`crate::rope::Rope::yarn`] of the model's RoPE at `rotary_dim`.
    pub yarn: crate::rope::RopeYarn,
}

/// Storage format of a QSA layer's state: the full K/V caches and the
/// indexer's raw and pooled keys (`kernels/src/tensor_ops.hip`, "QSA state
/// formats"). The index keys are BF16 values in every format, so the
/// quantized formats' BF16 index arenas pool and select exactly what F32
/// arenas do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QsaKvFormat {
    /// F32 K/V and index keys: the exact reference.
    F32,
    /// E4M3 K/V with one f16 scale per head and token (the Qwen3.5 native fp8
    /// row; gfx12 only), BF16 index keys.
    Fp8,
    /// Q8_0 K/V (32-element blocks of one f16 scale and 32 int8 codes, the
    /// dense q8 KV row; gfx11 and gfx12), BF16 index keys.
    Q8,
    /// E4M3 FNUZ K/V with one f16 scale per head and token (the [`Self::Fp8`]
    /// row layout with the gfx942 FNUZ code points; gfx942 only), F32 index
    /// keys.
    Fp8Fnuz,
}

impl QsaKvFormat {
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Fp8 => "fp8",
            Self::Q8 => "q8",
            Self::Fp8Fnuz => "fp8_fnuz",
        }
    }

    /// Whether the kernels implement this format for a head geometry.
    pub const fn supports(self, kv_heads: usize, head_dim: usize) -> bool {
        match self {
            Self::F32 => head_dim > 0,
            // Whole-head scales over one 256-thread block, and 4-byte aligned
            // code rows (`kv_heads * 258` bytes).
            Self::Fp8 | Self::Fp8Fnuz => head_dim == 256 && kv_heads % 2 == 0,
            // One 256-thread block per head, eight 32-lane Q8_0 blocks, and
            // 4-byte aligned block pairs (272-byte heads).
            Self::Q8 => head_dim == 256 && kv_heads > 0,
        }
    }

    /// Dtype of the K/V tensors: F32 elements, or quantized byte rows as Raw.
    pub const fn kv_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Fp8 | Self::Q8 | Self::Fp8Fnuz => DType::Raw,
        }
    }

    /// `kv_dtype` units (elements or bytes) per token row of K or of V.
    pub const fn kv_row_units(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::F32 => kv_heads * head_dim,
            Self::Fp8 | Self::Fp8Fnuz => kv_heads * (head_dim + 2),
            Self::Q8 => kv_heads * (head_dim / 32) * 34,
        }
    }

    /// Bytes per token row of K or of V.
    pub const fn kv_row_bytes(self, kv_heads: usize, head_dim: usize) -> usize {
        match self {
            Self::F32 => kv_heads * head_dim * 4,
            Self::Fp8 | Self::Q8 | Self::Fp8Fnuz => self.kv_row_units(kv_heads, head_dim),
        }
    }

    /// Dtype of the raw and pooled index-key arenas.
    pub const fn index_dtype(self) -> DType {
        match self {
            Self::F32 | Self::Fp8Fnuz => DType::F32,
            Self::Fp8 | Self::Q8 => DType::BF16,
        }
    }

    /// This format's entry of a `[f32, fp8, q8, fp8_fnuz]` kernel-name table.
    fn kernel(self, names: [&'static str; 4]) -> &'static str {
        names[self as usize]
    }
}

/// Context storage of one QSA layer, including the optional derived F16
/// arena. A bare [`QsaKvFormat`] describes the unchanged state without one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QsaStateLayout {
    pub kv: QsaKvFormat,
    /// Persistent F16 K and block-transposed V arenas
    /// ([`QsaF16ArenaRef`]); each holds `qsa_f16_arena_rows(max_seq)` rows of
    /// `kv_heads * head_dim` F16 elements.
    pub f16_arena: bool,
    /// Derived BF16 mirror of the pooled index keys
    /// (`qsa_pooled_mirror::QsaPooledBf16MirrorRef`); holds `index_dim` BF16
    /// elements per pooled block, never snapshotted.
    pub pooled_bf16: bool,
}

impl QsaStateLayout {
    /// The exact F32 reference state with no extra storage.
    pub const F32: Self = Self::new(QsaKvFormat::F32);

    pub const fn new(kv: QsaKvFormat) -> Self {
        Self {
            kv,
            f16_arena: false,
            pooled_bf16: false,
        }
    }

    /// Dtype of the raw and pooled index-key arenas.
    pub const fn index_dtype(self) -> DType {
        self.kv.index_dtype()
    }

    /// Bytes per cached token of both F16 arenas (K and V, 2 bytes per
    /// channel), `0` without them.
    pub const fn f16_arena_bytes_per_token(self, kv_heads: usize, head_dim: usize) -> usize {
        if self.f16_arena {
            2 * kv_heads * head_dim * 2
        } else {
            0
        }
    }

    /// Bytes per pooled block of the BF16 mirror (2 bytes per index
    /// channel), `0` without it.
    pub const fn pooled_bf16_bytes_per_block(self, index_dim: usize) -> usize {
        if self.pooled_bf16 {
            2 * index_dim
        } else {
            0
        }
    }
}

impl From<QsaKvFormat> for QsaStateLayout {
    fn from(kv: QsaKvFormat) -> Self {
        Self::new(kv)
    }
}

/// Rows an F16 arena of `tokens` cached tokens holds: V is block-transposed
/// in four-token blocks, so its last block is written whole.
pub const fn qsa_f16_arena_rows(tokens: usize) -> usize {
    tokens.div_ceil(4).saturating_mul(4)
}

/// Load-time arena admission: enabled unless opted out, exact gfx1151,
/// F32 K/V, head_dim 256, and the gathered prefill route enabled.
pub fn qsa_f16_arena_enabled(gpu: &Gpu, format: QsaKvFormat, head_dim: usize) -> bool {
    *QWEN4_QSA_F16_ARENA
        && format == QsaKvFormat::F32
        && head_dim == 256
        && gpu.arch_caps.is_gfx1151()
        && qsa_gathered_wmma_enabled(gpu, format)
}

/// Whether the gathered route's shared F16 workspace is still needed: the
/// route is enabled for `format` and the QSA layers do not keep their own F16
/// arena (`f16_arena`; the MTP head never takes the gathered route, its
/// attention rows stay below `QWEN4_F16_WMMA_MIN_TOKENS`).
pub fn qsa_gather_workspace_needed(gpu: &Gpu, format: QsaKvFormat, f16_arena: bool) -> bool {
    !f16_arena && qsa_gathered_wmma_enabled(gpu, format)
}

/// Host watermark for a derived arena. It travels with its F32/F16 owners,
/// is never snapshotted, and only the successful converter advances it.
#[derive(Default)]
pub struct QsaF16ValidLen(std::sync::atomic::AtomicUsize);

impl QsaF16ValidLen {
    pub fn get(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn valid(&self, committed: usize) -> usize {
        self.get().min(committed)
    }

    pub fn clamp(&self, committed: usize) {
        self.0
            .fetch_min(committed, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn invalidate(&self) {
        self.0.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn converted(&self, end: usize) {
        self.0.store(end, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Conversion starts at a four-token boundary to preserve the existing
/// block-transposed V layout; at most three valid rows are reread.
fn qsa_f16_arena_start(valid: usize, committed: usize) -> usize {
    valid.min(committed) & !3
}

/// A QSA layer's persistent F16 arena as the gathered attention reads it:
/// `keys` is F16 `[token][kv_heads * 256]` and `values` the block-transposed
/// F16 `[token / 4][kv_heads * 256][token % 4]`
/// (`indexed_attention_kv_f16vb`'s layout), both sized for
/// [`qsa_f16_arena_rows`] of the cache capacity. `valid_len` counts the
/// leading tokens whose F16 rows are known to equal the F32 cache's; the owner
/// clamps it whenever cache rows are rewritten or rewound.
#[derive(Clone, Copy)]
pub struct QsaF16ArenaRef<'a> {
    pub keys: &'a GpuTensor,
    pub values: &'a GpuTensor,
    pub valid_len: &'a QsaF16ValidLen,
}

impl QsaF16ArenaRef<'_> {
    /// Leading tokens that are valid while the F32 cache holds `committed`.
    pub fn valid(&self, committed: usize) -> usize {
        self.valid_len.valid(committed)
    }
}

/// Refuse a format the device or head geometry has no kernels for, and K/V
/// tensors that are not in its dtype.
fn check_qsa_format(
    gpu: &Gpu,
    format: QsaKvFormat,
    kv_heads: usize,
    head_dim: usize,
    tensors: [&GpuTensor; 2],
) -> HipResult<()> {
    if !format.supports(kv_heads, head_dim) {
        return Err(HipError::new(
            0,
            &format!(
                "QSA {} K/V needs another head geometry ({kv_heads} KV heads x {head_dim})",
                format.name()
            ),
        ));
    }
    if format == QsaKvFormat::Fp8 && !(gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201()) {
        return Err(HipError::new(0, "QSA fp8 K/V needs a gfx12 device"));
    }
    if format == QsaKvFormat::Fp8Fnuz && !gpu.arch_caps.is_gfx942() {
        return Err(HipError::new(0, "QSA fp8 FNUZ K/V needs a gfx942 device"));
    }
    if format == QsaKvFormat::Q8 && !gpu.arch_caps.has_gfx11_plus_simt() {
        return Err(HipError::new(0, "QSA q8 K/V needs a gfx11 or gfx12 device"));
    }
    if tensors
        .iter()
        .any(|tensor| tensor.dtype != format.kv_dtype())
    {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    Ok(())
}

/// QSA prologue of `rows` consecutive rows (decode: one): the index-query,
/// query and key norm+RoPE, the key/value cache append and the index key's
/// BF16 round trip + raw-key copy, in one launch (bitwise the six launches it
/// replaces). Row buffers are row-major at their natural row widths. The
/// cache rows are `format`'s, the raw index keys its `index_dtype`.
pub struct IndexedAttentionDecodePrologue<'a> {
    /// `[index q (index_heads * index_dim) | index k (index_kv_width)]`.
    pub index_row: &'a GpuTensor,
    /// `[heads, 2 * head_dim]` query + gate.
    pub qgate: &'a GpuTensor,
    pub keys: &'a GpuTensor,
    pub values: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub index_q_norm: &'a GpuTensor,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_kv_width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub position: usize,
    pub rows: usize,
    pub format: QsaKvFormat,
    /// [`crate::rope::Rope::yarn`] at the prologue's rotary width
    /// (`min(dim, 64)`).
    pub yarn: crate::rope::RopeYarn,
}

pub fn indexed_attention_decode_prologue(
    gpu: &mut Gpu,
    p: &IndexedAttentionDecodePrologue<'_>,
) -> HipResult<()> {
    for tensor in [p.index_row, p.qgate, p.keys, p.values] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    if p.raw_index_keys.dtype != p.format.index_dtype() {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let row_units = p.format.kv_row_units(p.kv_heads, p.head_dim);
    let kv_width = checked_product(p.kv_heads, p.head_dim, "QSA prologue KV width")?;
    let end = checked_add(p.position, p.rows, "QSA prologue position")?;
    let index_width =
        checked_product(p.index_heads, p.index_dim, "QSA prologue index")? + p.index_kv_width;
    let bad = [p.index_q_norm, p.q_norm, p.k_norm]
        .iter()
        .any(|n| n.dtype != DType::BF16)
        || p.rows == 0
        || p.index_heads == 0
        || p.heads == 0
        || p.kv_heads == 0
        || p.index_dim == 0
        || p.index_dim > 256
        || p.head_dim == 0
        || p.head_dim > 256
        || p.index_q_norm.numel() != p.index_dim
        || p.q_norm.numel() != p.head_dim
        || p.k_norm.numel() != p.head_dim
        || p.index_row.numel() < checked_product(p.rows, index_width, "QSA prologue index")?
        || p.qgate.numel()
            < checked_product3(p.rows, 2 * p.heads, p.head_dim, "QSA prologue query")?
        || p.keys.numel() < checked_product(p.rows, kv_width, "QSA prologue keys")?
        || p.values.numel() < checked_product(p.rows, kv_width, "QSA prologue values")?
        || p.full_keys.numel() < checked_product(end, row_units, "QSA prologue cache")?
        || p.full_values.numel() < checked_product(end, row_units, "QSA prologue cache")?
        || p.raw_index_keys.numel()
            < checked_product(end, p.index_kv_width, "QSA prologue raw keys")?;
    if bad {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let blocks_x = checked_u32(
        p.index_heads + p.heads + p.kv_heads + 1,
        "QSA prologue block count",
    )?;
    let kernel = p.format.kernel([
        "indexed_attention_decode_prologue_f32",
        "indexed_attention_decode_prologue_fp8",
        "indexed_attention_decode_prologue_q8",
        "indexed_attention_decode_prologue_fp8_fnuz",
    ]);
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [
        p.index_row,
        p.qgate,
        p.keys,
        p.values,
        p.full_keys,
        p.full_values,
        p.raw_index_keys,
        p.index_q_norm,
        p.q_norm,
        p.k_norm,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for (value, label) in [
        (p.index_heads, "QSA prologue index heads"),
        (p.index_dim, "QSA prologue index width"),
        (p.index_kv_width, "QSA prologue index KV width"),
        (p.heads, "QSA prologue heads"),
        (p.kv_heads, "QSA prologue KV heads"),
        (p.head_dim, "QSA prologue head width"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_i32(checked_i32(p.position, "QSA prologue position")?);
    // Declared dynamic field: replay re-derives the position (angles and
    // both cache rows follow it).
    let position_offset = args.len() - 4;
    p.yarn.push(&mut args);
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents: grid `[index_heads + heads + kv_heads + 1, rows]`; every block does
    // work and each thread is guarded by `c < index_dim` / `c < head_dim` / `i <
    // index_kv_width`. Index rows are `index_width` F32 wide, `qgate` rows
    // `2 * heads * head_dim` wide of which only the query half of each head is
    // touched (`(rows - 1) * 2 * heads * head_dim + (heads - 1) * 2 * head_dim +
    // head_dim` elements), K/V rows `kv_width` F32; the caches get `rows` whole
    // format rows (`kv_row_bytes`: F32 `kv_width * 4`, fp8 `kv_heads *
    // (head_dim + 2)`, q8 `kv_heads * head_dim / 32 * 34`) at `position *
    // kv_row_bytes`; the raw index keys `rows * index_kv_width` elements of the
    // arena dtype at `position * index_kv_width`.
    let extents = || {
        let kv_row_bytes = p.format.kv_row_bytes(p.kv_heads, p.head_dim);
        let raw_elem = p.raw_index_keys.dtype.size();
        let qgate_elements =
            (p.rows - 1) * 2 * p.heads * p.head_dim + (p.heads - 1) * 2 * p.head_dim + p.head_dim;
        (
            [
                (
                    "qsa_index_row",
                    p.index_row,
                    0usize,
                    p.rows * index_width * 4,
                ),
                ("qsa_qgate", p.qgate, 0, qgate_elements * 4),
                ("qsa_keys", p.keys, 0, p.rows * kv_width * 4),
                ("qsa_values", p.values, 0, p.rows * kv_width * 4),
                (
                    "qsa_full_keys",
                    p.full_keys,
                    p.position * kv_row_bytes,
                    p.rows * kv_row_bytes,
                ),
                (
                    "qsa_full_values",
                    p.full_values,
                    p.position * kv_row_bytes,
                    p.rows * kv_row_bytes,
                ),
                (
                    "qsa_raw_index_keys",
                    p.raw_index_keys,
                    p.position * p.index_kv_width * raw_elem,
                    p.rows * p.index_kv_width * raw_elem,
                ),
                ("qsa_index_q_norm", p.index_q_norm, 0, p.index_dim * 2),
                ("qsa_q_norm", p.q_norm, 0, p.head_dim * 2),
                ("qsa_k_norm", p.k_norm, 0, p.head_dim * 2),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position: end,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_decode_prologue");
    }
    gpu.launch_blob_recorded(
        kernel,
        [blocks_x, checked_u32(p.rows, "QSA prologue rows")?, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_decode_prologue");
    }
    Ok(())
}

/// QSA prologue of the `MtpStep::Append`-style key-value-only fill of `rows`
/// consecutive prompt rows: the decode prologue kernel launched with
/// `heads == 0`, so the query/gate branch is dead and only the index-query,
/// key norm+RoPE, K/V cache append and index-key BF16 round trip + raw-key copy
/// run (same kernel, same per-element operations as `rows` decode-prologue
/// launches). Row buffers are row-major at their natural row widths; row `r`
/// sits at cache position `position + r`.
pub struct IndexedAttentionAppendPrologue<'a> {
    /// `rows x [index q (index_heads * index_dim) | index k (index_kv_width)]`.
    pub index_row: &'a GpuTensor,
    /// `rows x [kv_heads * head_dim]`, normed + roped in place.
    pub keys: &'a GpuTensor,
    /// `rows x [kv_heads * head_dim]`.
    pub values: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub index_q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_kv_width: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub position: usize,
    pub rows: usize,
    pub format: QsaKvFormat,
    /// See [`IndexedAttentionDecodePrologue::yarn`].
    pub yarn: crate::rope::RopeYarn,
}

pub fn indexed_attention_append_prologue(
    gpu: &mut Gpu,
    p: &IndexedAttentionAppendPrologue<'_>,
) -> HipResult<()> {
    for tensor in [p.index_row, p.keys, p.values] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    if p.raw_index_keys.dtype != p.format.index_dtype() {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let row_units = p.format.kv_row_units(p.kv_heads, p.head_dim);
    let kv_width = checked_product(p.kv_heads, p.head_dim, "QSA append prologue KV width")?;
    let end = checked_add(p.position, p.rows, "QSA append prologue position")?;
    let index_width = checked_product(p.index_heads, p.index_dim, "QSA append prologue index")?
        + p.index_kv_width;
    let bad = [p.index_q_norm, p.k_norm]
        .iter()
        .any(|n| n.dtype != DType::BF16)
        || p.rows == 0
        || p.index_heads == 0
        || p.kv_heads == 0
        || p.index_dim == 0
        || p.index_dim > 256
        || p.head_dim == 0
        || p.head_dim > 256
        || p.index_q_norm.numel() != p.index_dim
        || p.k_norm.numel() != p.head_dim
        || p.index_row.numel() < checked_product(p.rows, index_width, "QSA append prologue index")?
        || p.keys.numel() < checked_product(p.rows, kv_width, "QSA append prologue keys")?
        || p.values.numel() < checked_product(p.rows, kv_width, "QSA append prologue values")?
        || p.full_keys.numel() < checked_product(end, row_units, "QSA append prologue cache")?
        || p.full_values.numel() < checked_product(end, row_units, "QSA append prologue cache")?
        || p.raw_index_keys.numel()
            < checked_product(end, p.index_kv_width, "QSA append prologue raw keys")?;
    if bad {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let blocks_x = checked_u32(
        p.index_heads + p.kv_heads + 1,
        "QSA append prologue block count",
    )?;
    let kernel = p.format.kernel([
        "indexed_attention_decode_prologue_f32",
        "indexed_attention_decode_prologue_fp8",
        "indexed_attention_decode_prologue_q8",
        "indexed_attention_decode_prologue_fp8_fnuz",
    ]);
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    // heads == 0: the kernel never dereferences `qgate` (its row offset is
    // `r * 2 * 0 * head_dim`, and `block < heads` is never true) nor `q_norm`,
    // so they alias live buffers of the right dtype.
    for tensor in [
        p.index_row,
        p.keys,
        p.keys,
        p.values,
        p.full_keys,
        p.full_values,
        p.raw_index_keys,
        p.index_q_norm,
        p.k_norm,
        p.k_norm,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for (value, label) in [
        (p.index_heads, "QSA append prologue index heads"),
        (p.index_dim, "QSA append prologue index width"),
        (p.index_kv_width, "QSA append prologue index KV width"),
        (0, "QSA append prologue heads"),
        (p.kv_heads, "QSA append prologue KV heads"),
        (p.head_dim, "QSA append prologue head width"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_i32(checked_i32(p.position, "QSA append prologue position")?);
    // Declared dynamic field: replay re-derives the position (angles and
    // both cache rows follow it).
    let position_offset = args.len() - 4;
    p.yarn.push(&mut args);
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents (same kernel as the decode prologue with `heads == 0`; grid
    // `[index_heads + kv_heads + 1, rows]`, every block does work and each
    // thread is guarded by `c < index_dim` / `c < head_dim` / `i <
    // index_kv_width`): index rows are `index_width` F32 wide (query heads +
    // BF16-rounded key tail), K/V rows `kv_width` F32, the caches get `rows`
    // whole format rows at `position * kv_row_bytes`, the raw index keys
    // `rows * index_kv_width` elements of the arena dtype at `position *
    // index_kv_width`. `qgate`/`q_norm` alias `keys`/`k_norm` and are never
    // dereferenced (`heads == 0`), so they are not checked as separate operands.
    let extents = || {
        let kv_row_bytes = p.format.kv_row_bytes(p.kv_heads, p.head_dim);
        let raw_elem = p.raw_index_keys.dtype.size();
        (
            [
                (
                    "qsa_index_row",
                    p.index_row,
                    0usize,
                    p.rows * index_width * 4,
                ),
                ("qsa_keys", p.keys, 0, p.rows * kv_width * 4),
                ("qsa_values", p.values, 0, p.rows * kv_width * 4),
                (
                    "qsa_full_keys",
                    p.full_keys,
                    p.position * kv_row_bytes,
                    p.rows * kv_row_bytes,
                ),
                (
                    "qsa_full_values",
                    p.full_values,
                    p.position * kv_row_bytes,
                    p.rows * kv_row_bytes,
                ),
                (
                    "qsa_raw_index_keys",
                    p.raw_index_keys,
                    p.position * p.index_kv_width * raw_elem,
                    p.rows * p.index_kv_width * raw_elem,
                ),
                ("qsa_index_q_norm", p.index_q_norm, 0, p.index_dim * 2),
                ("qsa_k_norm", p.k_norm, 0, p.head_dim * 2),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position: end,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_append_prologue");
    }
    gpu.launch_blob_recorded(
        kernel,
        [
            blocks_x,
            checked_u32(p.rows, "QSA append prologue rows")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_append_prologue");
    }
    Ok(())
}

/// RoPE angle table for the fused QSA q/k norm+RoPE GEMM epilogue
/// (`indexed_attention_rope_table_f32`): row `r` of `table` (`rotary_dim`
/// F32) holds the cosines of the `rotary_dim / 2` bases, then their sines,
/// for position `position_start + r`, computed exactly as
/// `indexed_attention_norm_rope_batch` does. `table` needs at least
/// `rows * rotary_dim` F32.
pub fn indexed_attention_rope_table(
    gpu: &mut Gpu,
    table: &GpuTensor,
    rows: usize,
    position_start: usize,
    rotary_dim: usize,
    yarn: crate::rope::RopeYarn,
) -> HipResult<()> {
    ensure_f32(table)?;
    if rows == 0
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || position_start
            .checked_add(rows)
            .map_or(true, |end| end > i32::MAX as usize)
        || table.numel() < checked_product(rows, rotary_dim, "rope table extent")?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows_i = checked_i32(rows, "rope table rows")?;
    let position_i = checked_i32(position_start, "rope table position")?;
    let rotary_i = checked_i32(rotary_dim, "rope table rotary width")?;
    let threads = checked_product(rows, rotary_dim / 2, "rope table threads")?;
    let grid = checked_u32(threads.div_ceil(256), "rope table grid")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_rope_table_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(table.buf.as_ptr());
    args.push_i32(rows_i);
    args.push_i32(position_i);
    args.push_i32(rotary_i);
    yarn.push(&mut args);
    args.pad_to(16);
    // Extents: grid `[ceil(rows * rotary_dim / 2 / 256), 1, 1]`; a thread
    // writes `table[row * rotary_dim + base]` and `+ rotary_dim / 2` for
    // `row < rows`, so the guarded extent is `rows * rotary_dim` F32.
    gpu.launch_blob_recorded(
        "indexed_attention_rope_table_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn indexed_attention_norm_rope_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRopeBatch<'_>,
) -> HipResult<()> {
    ensure_f32(p.values)?;
    if p.norm.dtype != DType::BF16
        || p.rows == 0
        || p.row_stride == 0
        || p.heads == 0
        || p.head_dim == 0
        || p.head_dim > 256
        || p.head_stride < p.head_dim
        || p.rotary_dim == 0
        || p.rotary_dim % 2 != 0
        || p.rotary_dim > p.head_dim
        || p.position_start
            .checked_add(p.rows)
            .map_or(true, |end| end > i32::MAX as usize)
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_span = checked_product(
        p.heads - 1,
        p.head_stride,
        "indexed attention batch head span",
    )
    .and_then(|base| checked_add(base, p.head_dim, "indexed attention batch head span"))?;
    if p.row_stride < head_span
        || p.norm.numel() != p.head_dim
        || p.values.numel()
            < checked_product(p.rows - 1, p.row_stride, "indexed attention batch rows")?
                .checked_add(head_span)
                .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "indexed attention batch rows")?;
    let row_stride = checked_i32(p.row_stride, "indexed attention batch row stride")?;
    let heads = checked_i32(p.heads, "indexed attention batch head count")?;
    let head_dim = checked_i32(p.head_dim, "indexed attention batch head width")?;
    let head_stride = checked_i32(p.head_stride, "indexed attention batch head stride")?;
    let position_start = checked_i32(p.position_start, "indexed attention batch position")?;
    let rotary_dim = checked_i32(p.rotary_dim, "indexed attention batch rotary width")?;
    let head_grid = checked_u32(p.heads, "indexed attention batch head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    let row_grid = checked_u32(p.rows, "indexed attention batch row grid")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_norm_rope_f32_batched",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.values, p.norm] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [rows, row_stride, heads, head_dim, head_stride] {
        args.push_i32(value);
    }
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it
    // from its own position instead of replaying the capture-position angle
    // base; the kernel adds the row index itself.
    let position_offset = args.len() - 4;
    args.push_i32(rotary_dim);
    p.yarn.push(&mut args);
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents: grid `[heads, 1, rows]` is exact in x/z; the kernel returns for
    // `blockIdx.y != 0` and touches `channel = threadIdx.x < head_dim` of head
    // `h` of row `r` at `r * row_stride + h * head_stride`, so the guarded
    // extent is `(rows - 1) * row_stride + head_span` F32 (`head_span` =
    // `(heads - 1) * head_stride + head_dim`) plus `head_dim` norm BF16.
    let extents = || {
        (
            [
                (
                    "qsa_norm_rope_values",
                    p.values,
                    0usize,
                    ((p.rows - 1) * p.row_stride + head_span) * 4,
                ),
                ("qsa_norm_rope_norm", p.norm, 0, p.head_dim * 2),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position: p.position_start + p.rows,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_norm_rope_batch");
    }
    gpu.launch_blob_recorded(
        "indexed_attention_norm_rope_f32_batched",
        [head_grid, dim_grid, row_grid],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_norm_rope_batch");
    }
    Ok(())
}

pub struct IndexedAttentionCacheAppend<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub position: usize,
    pub kv_width: usize,
}

pub fn indexed_attention_cache_append(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppend<'_>,
) -> HipResult<()> {
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        ensure_f32(tensor)?;
    }
    if p.kv_width == 0
        || p.key.numel() != p.kv_width
        || p.value.numel() != p.kv_width
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let end = checked_product(p.position, p.kv_width, "indexed attention cache offset")
        .and_then(|base| checked_add(base, p.kv_width, "indexed attention cache offset"))?;
    if end > p.full_keys.numel() {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let position = checked_i32(p.position, "indexed attention cache position")?;
    let kv_width = checked_i32(p.kv_width, "indexed attention cache width")?;
    let grid = blocks(p.kv_width)?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_cache_append_f32",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(position);
    args.push_i32(kv_width);
    args.pad_to(16);
    // Extents: guard `channel < kv_width` (grid.x = blocks(kv_width) rounds up to
    // 256 threads, never past the guard): one F32 K/V row read, one row written
    // at `position * kv_width`.
    let extents = || {
        (
            [
                ("qsa_keys", p.key, 0usize, p.kv_width * 4),
                ("qsa_values", p.value, 0, p.kv_width * 4),
                (
                    "qsa_full_keys",
                    p.full_keys,
                    p.position * p.kv_width * 4,
                    p.kv_width * 4,
                ),
                (
                    "qsa_full_values",
                    p.full_values,
                    p.position * p.kv_width * 4,
                    p.kv_width * 4,
                ),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: 1,
                end_position: p.position + 1,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_cache_append");
    }
    gpu.launch_blob_recorded(
        "indexed_attention_cache_append_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_cache_append");
    }
    Ok(())
}

/// Append `rows` F32 K/V rows at `position_start` to caches in `format`.
pub struct IndexedAttentionCacheAppendBatch<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub rows: usize,
    pub position_start: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub format: QsaKvFormat,
}

pub fn indexed_attention_cache_append_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppendBatch<'_>,
) -> HipResult<()> {
    for tensor in [p.key, p.value] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    let kv_width = checked_product(p.kv_heads, p.head_dim, "indexed attention batch width")?;
    if p.rows == 0
        || kv_width == 0
        || p.key.numel() < checked_product(p.rows, kv_width, "indexed attention batch key")?
        || p.value.numel() < checked_product(p.rows, kv_width, "indexed attention batch value")?
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let end = checked_product(
        end_position,
        p.format.kv_row_units(p.kv_heads, p.head_dim),
        "indexed attention batch cache",
    )?;
    if end > p.full_keys.numel() || end_position > i32::MAX as usize {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "indexed attention batch rows")?;
    let position_start = checked_i32(p.position_start, "indexed attention batch position")?;
    let row_grid = checked_u32(p.rows, "indexed attention batch row grid")?;
    let kernel = p.format.kernel([
        "indexed_attention_cache_append_f32_batched",
        "indexed_attention_cache_append_fp8_batched",
        "indexed_attention_cache_append_q8_batched",
        "indexed_attention_cache_append_fp8_fnuz_batched",
    ]);
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(rows);
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it,
    // so the cache row this launch writes follows the replay position.
    let position_offset = args.len() - 4;
    // F32 copies channels across a flat grid; the quantized formats reduce
    // their scales over one block per KV head.
    let grid = if p.format == QsaKvFormat::F32 {
        args.push_i32(checked_i32(kv_width, "indexed attention batch width")?);
        [blocks(kv_width)?, row_grid, 1]
    } else {
        args.push_i32(checked_i32(p.kv_heads, "indexed attention batch KV heads")?);
        args.push_i32(checked_i32(
            p.head_dim,
            "indexed attention batch head width",
        )?);
        [
            checked_u32(p.kv_heads, "indexed attention batch head grid")?,
            row_grid,
            1,
        ]
    };
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents: guard `row < rows` and (F32) `channel < kv_width` / (quantized)
    // `head < kv_heads` with the 256-thread block over one head (`c < head_dim`
    // guards the reads, `qsa_kv_store` stores nothing for `c >= head_dim`); the
    // F32 grid.x round-up never passes the guard. K/V are read as
    // `rows * kv_width` F32; each cache gets exactly `rows` whole format rows
    // (`kv_row_bytes`: F32 `kv_width * 4`, fp8 `kv_heads * (head_dim + 2)`
    // codes+f16 scales, q8 `kv_heads * head_dim / 32 * 34`) at
    // `position_start * kv_row_bytes`.
    let extents = || {
        let row_bytes = p.format.kv_row_bytes(p.kv_heads, p.head_dim);
        (
            [
                ("qsa_keys", p.key, 0usize, p.rows * kv_width * 4),
                ("qsa_values", p.value, 0, p.rows * kv_width * 4),
                (
                    "qsa_full_keys",
                    p.full_keys,
                    p.position_start * row_bytes,
                    p.rows * row_bytes,
                ),
                (
                    "qsa_full_values",
                    p.full_values,
                    p.position_start * row_bytes,
                    p.rows * row_bytes,
                ),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_cache_append_batch");
    }
    gpu.launch_blob_recorded(
        kernel,
        grid,
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_cache_append_batch");
    }
    Ok(())
}

/// Prefill index-key tail into a BF16 raw-key arena
/// (`indexed_attention_index_key_append_bf16_batched`): rounds each row's
/// index key in `index_rows` to BF16 in place and appends it at
/// `position_start + row`, exactly as the decode prologue does per row.
pub struct IndexedAttentionIndexKeyAppendBatch<'a> {
    /// `rows` rows of `[index q (index_q_width) | index k (index_kv_width)]`.
    pub index_rows: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub rows: usize,
    pub index_q_width: usize,
    pub index_kv_width: usize,
    pub position_start: usize,
}

pub fn indexed_attention_index_key_append_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionIndexKeyAppendBatch<'_>,
) -> HipResult<()> {
    ensure_f32(p.index_rows)?;
    if p.raw_index_keys.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    let index_width = checked_add(p.index_q_width, p.index_kv_width, "QSA index-key width")?;
    let end = checked_add(p.position_start, p.rows, "QSA index-key position")?;
    if p.rows == 0
        || p.index_kv_width == 0
        || p.index_rows.numel() < checked_product(p.rows, index_width, "QSA index-key rows")?
        || p.raw_index_keys.numel() < checked_product(end, p.index_kv_width, "QSA raw keys")?
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    const NAME: &str = "indexed_attention_index_key_append_bf16_batched";
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, NAME)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.index_rows.buf.as_ptr());
    args.push_ptr(p.raw_index_keys.buf.as_ptr());
    for (value, label) in [
        (p.rows, "QSA index-key rows"),
        (p.index_q_width, "QSA index-key query width"),
        (index_width, "QSA index-key row width"),
        (p.index_kv_width, "QSA index-key width"),
    ] {
        args.push_i32(checked_i32(value, label)?);
    }
    args.push_i32(checked_i32(p.position_start, "QSA index-key position")?);
    // Declared dynamic field: the chunk's start position (the raw-key row).
    let position_offset = args.len() - 4;
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: 0,
    }];
    // Extents: guard `row < rows && i < index_kv_width` (grid.x = blocks(index_kv_width)
    // rounds threads up to 256, never past the guard). It reads and rewrites only
    // the key part of each index row, `[index_q_width, index_q_width +
    // index_kv_width)` of `index_width`-wide rows, and writes BF16 raw keys at
    // `(position_start + row) * index_kv_width + i`.
    let extents = || {
        (
            [
                (
                    "qsa_index_row",
                    p.index_rows,
                    p.index_q_width * 4,
                    ((p.rows - 1) * index_width + p.index_kv_width) * 4,
                ),
                (
                    "qsa_raw_index_keys",
                    p.raw_index_keys,
                    p.position_start * p.index_kv_width * 2,
                    p.rows * p.index_kv_width * 2,
                ),
            ],
            crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position: end,
                block_count: 0,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(
            gpu,
            &sites,
            depth,
            "indexed_attention_index_key_append_batch",
        );
    }
    gpu.launch_blob_recorded(
        NAME,
        [
            blocks(p.index_kv_width)?,
            checked_u32(p.rows, "QSA index-key row grid")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(
            gpu,
            &sites,
            depth,
            "indexed_attention_index_key_append_batch",
        );
    }
    Ok(())
}
pub struct IndexedAttentionSelect<'a> {
    pub query: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub block_count: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub visible: usize,
    pub capacity: usize,
}

pub fn indexed_attention_select(gpu: &mut Gpu, p: &IndexedAttentionSelect<'_>) -> HipResult<()> {
    for tensor in [p.query, p.pooled] {
        ensure_f32(tensor)?;
    }
    if p.selected.dtype != DType::Raw
        || p.index_heads == 0
        || p.index_dim == 0
        || p.compress == 0
        || p.capacity == 0
        || p.visible == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_elements = checked_product(p.index_heads, p.index_dim, "QSA select query extent")?;
    let pooled_elements = checked_product(p.block_count, p.index_dim, "QSA select pooled extent")?;
    checked_product(p.block_count, p.compress, "QSA select token extent")?;
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let block_count = checked_i32(p.block_count, "QSA select block count")?;
    let index_heads = checked_i32(p.index_heads, "QSA select index-head count")?;

    let index_dim = checked_i32(p.index_dim, "QSA select index width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA select budget")?;
    let compress = checked_i32(p.compress, "QSA select compression")?;
    let visible = checked_i32(p.visible, "QSA select visible count")?;
    let capacity = checked_i32(p.capacity, "QSA select capacity")?;
    if p.query.numel() != query_elements
        || p.pooled.numel() < pooled_elements
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    // The rank-parallel batched kernel emits the serial selection sort's
    // bytes; one row at `visible - 1` is this selection whenever the block
    // count follows the visible prefix (the single-row kernel is O(chosen x
    // blocks) in one thread).
    if p.block_count == p.visible / p.compress {
        return indexed_attention_select_batch(
            gpu,
            &IndexedAttentionSelectBatch {
                pooled_bf16: None,
                pooled_bf16_valid_blocks: 0,
                query: p.query,
                pooled: p.pooled,
                selected: p.selected,
                rows: 1,
                query_row_stride: query_elements,
                block_count: p.block_count,
                index_heads: p.index_heads,
                index_dim: p.index_dim,
                budget_blocks: p.budget_blocks,
                compress: p.compress,
                position_start: p.visible - 1,
                capacity: p.capacity,
                shape_blocks: p.block_count,
            },
        );
    }
    let (kernel_name, block, shared_mem) =
        match p.block_count.checked_mul(std::mem::size_of::<f32>()) {
            Some(bytes)
                if gpu.arch_caps.has_gfx11_plus_simt()
                    && bytes <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (
                    "indexed_attention_select_f32",
                    [QSA_SELECT_PARALLEL_THREADS, 1, 1],
                    bytes as u32,
                )
            }
            _ => ("indexed_attention_select_f32_serial", [1, 1, 1], 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.query, p.pooled, p.selected] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [
        block_count,
        index_heads,
        index_dim,
        budget_blocks,
        compress,
        visible,
        capacity,
    ] {
        args.push_i32(value);
    }
    args.pad_to(16);
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_single(
            gpu,
            p,
            kernel_name,
            kernel_name == "indexed_attention_select_f32_serial",
            false,
        );
    }
    gpu.launch_blob_recorded(
        kernel_name,
        [1, 1, 1],
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_single(
            gpu,
            p,
            kernel_name,
            kernel_name == "indexed_attention_select_f32_serial",
            true,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Developer-only QSA extent checks for the select launches
// (`gpu.flags.qwen4_qsa_extent_check`; mode 2 also synchronizes after every
// launch). Every helper below is called only from inside an
// `if gpu.flags.qwen4_qsa_extent_check` block at the launch sites, so the
// disabled path is that bool test alone.
//
// Each formula is the extent the *kernel* touches, read from the kernel source
// (`kernels/src/tensor_ops.hip`, `qsa_bf16_scores.hip`, `qsa_filter_select.hip`,
// `indexed_attention_select_exact.hip`, and the builder kernels
// `peacemaker-isa/src/kernels/qsa_{score,topk}.rip.rs`), not the extent the host
// validated:
// - "guarded": an in-kernel bound clamps the access (`block < block_count`,
//   `block < rbc`, `slot < cap`, `index < capacity`, or a raw-buffer descriptor
//   `num_records`), so only the guarded extent is checked;
// - "grid-round": the launch grid is rounded up (256-block score tiles, 4096-block
//   compact tiles, 16-row groups, the shape bound of a retained replay); the
//   surplus workgroups/lanes return before touching memory, so the rounded part
//   is never checked.
// `rbc` is the kernels' `min((position_start + row + 1) / compress,
// block_count)`; it is monotone in the row, so the last row of a launch is its
// maximum. Scores, selected and the control words are 4/4/8-byte-element rows:
// a `[rows][stride]` buffer is touched up to `(rows - 1) * stride + last_row`
// elements. All arithmetic saturates: an overflow reports as an over-extent.
// ---------------------------------------------------------------------------

type QsaSelDepth = crate::dispatch::QsaExtentDepth;

/// Byte size of the F32 / I32 words of the query, scores and selection rows.
const QSA_SEL_EXT_WORD: usize = 4;

fn qsa_sel_ext_depth(rows: usize, position_start: usize, block_count: usize) -> QsaSelDepth {
    QsaSelDepth {
        rows,
        end_position: position_start.saturating_add(rows),
        block_count,
    }
}

/// `rbc` of the last (maximal) of `rows` rows starting at `position_start`.
fn qsa_sel_ext_rbc_last(
    rows: usize,
    position_start: usize,
    compress: usize,
    block_count: usize,
) -> usize {
    position_start
        .saturating_add(rows)
        .checked_div(compress)
        .unwrap_or(0)
        .min(block_count)
}

/// Elements of a `[rows][stride]` buffer touched when every row uses
/// `last_row_elems` leading elements; nothing when a row touches none.
fn qsa_sel_ext_span(rows: usize, stride: usize, last_row_elems: usize) -> usize {
    if rows == 0 || last_row_elems == 0 {
        0
    } else {
        (rows - 1)
            .saturating_mul(stride)
            .saturating_add(last_row_elems)
    }
}

/// Elements the serial kernels read through `row_selected[prior * compress]`
/// (`prior < slot < chosen`, no `capacity` guard on the read): the last row can
/// reach `(chosen - 2) * compress + 1` elements, which may exceed `capacity`.
fn qsa_sel_ext_serial_prior(chosen: usize, compress: usize) -> usize {
    if chosen < 2 {
        0
    } else {
        (chosen - 2).saturating_mul(compress).saturating_add(1)
    }
}

/// Query rows `first_row..first_row + rows` (F32, `stride` apart): every row
/// reads its leading `index_heads * index_dim` elements (the score kernels stage
/// `4 * index_dim` per row into LDS/SGPRs; the fused kernels read per block).
#[allow(clippy::too_many_arguments)]
fn qsa_sel_ext_query(
    gpu: &Gpu,
    query: &GpuTensor,
    first_row: usize,
    rows: usize,
    stride: usize,
    width: usize,
    depth: QsaSelDepth,
    site: &'static str,
) {
    gpu.qsa_extent_check(
        "qsa_select_query",
        query,
        first_row
            .saturating_mul(stride)
            .saturating_mul(QSA_SEL_EXT_WORD),
        qsa_sel_ext_span(rows, stride, width).saturating_mul(QSA_SEL_EXT_WORD),
        depth,
        site,
    );
}

/// Pooled keys (or their BF16 mirror) rows `0..blocks`, `index_dim` elements of
/// the tensor's own element size each.
fn qsa_sel_ext_pooled(
    gpu: &Gpu,
    pooled: &GpuTensor,
    arena: &'static str,
    blocks: usize,
    index_dim: usize,
    depth: QsaSelDepth,
    site: &'static str,
) {
    gpu.qsa_extent_check(
        arena,
        pooled,
        0,
        blocks
            .saturating_mul(index_dim)
            .saturating_mul(pooled.dtype.size()),
        depth,
        site,
    );
}

/// Selected rows `first_row..first_row + rows` of `capacity` I32 each (every
/// kernel fills its whole row, `i < capacity`); `last_row_elems` is `capacity`
/// except for the serial kernels' unguarded prior read.
#[allow(clippy::too_many_arguments)]
fn qsa_sel_ext_selected(
    gpu: &Gpu,
    selected: &GpuTensor,
    first_row: usize,
    rows: usize,
    capacity: usize,
    last_row_elems: usize,
    depth: QsaSelDepth,
    site: &'static str,
) {
    gpu.qsa_extent_check(
        "qsa_select_selected",
        selected,
        first_row
            .saturating_mul(capacity)
            .saturating_mul(QSA_SEL_EXT_WORD),
        qsa_sel_ext_span(rows, capacity, last_row_elems).saturating_mul(QSA_SEL_EXT_WORD),
        depth,
        site,
    );
}

/// Mirror: the final row's `capacity` I32 (`mirror[i]`, `i < capacity`).
fn qsa_sel_ext_mirror(
    gpu: &Gpu,
    mirror: &GpuTensor,
    capacity: usize,
    depth: QsaSelDepth,
    site: &'static str,
) {
    gpu.qsa_extent_check(
        "qsa_select_mirror",
        mirror,
        0,
        capacity.saturating_mul(QSA_SEL_EXT_WORD),
        depth,
        site,
    );
}

/// Post-launch hook on the selection rows (the select kernels' output).
#[allow(clippy::too_many_arguments)]
fn qsa_sel_ext_post_selected(
    gpu: &Gpu,
    selected: &GpuTensor,
    first_row: usize,
    rows: usize,
    capacity: usize,
    position_start: usize,
    block_count: usize,
    site: &'static str,
) {
    gpu.qsa_extent_post_launch(
        "qsa_select_selected",
        selected,
        first_row
            .saturating_mul(capacity)
            .saturating_mul(QSA_SEL_EXT_WORD),
        qsa_sel_ext_span(rows, capacity, capacity).saturating_mul(QSA_SEL_EXT_WORD),
        qsa_sel_ext_depth(rows, position_start, block_count),
        site,
    );
}

/// `indexed_attention_select_f32{,_serial}`: one workgroup, one row. Query
/// `index_heads * index_dim`, pooled `block < block_count` (guarded), selected
/// `capacity` (every slot filled with -1 / written under `< capacity`); the
/// serial kernel also reads `selected[prior * compress]` unguarded. Scores are in
/// LDS (no global extent).
fn qsa_sel_ext_single(
    gpu: &Gpu,
    p: &IndexedAttentionSelect<'_>,
    kernel: &'static str,
    serial: bool,
    post: bool,
) {
    let depth = QsaSelDepth {
        rows: 1,
        end_position: p.visible,
        block_count: p.block_count,
    };
    let last_row = if serial {
        p.capacity.max(qsa_sel_ext_serial_prior(
            p.budget_blocks.min(p.block_count),
            p.compress,
        ))
    } else {
        p.capacity
    };
    if post {
        gpu.qsa_extent_post_launch(
            "qsa_select_selected",
            p.selected,
            0,
            last_row.saturating_mul(QSA_SEL_EXT_WORD),
            depth,
            kernel,
        );
        return;
    }
    let width = p.index_heads.saturating_mul(p.index_dim);
    qsa_sel_ext_query(gpu, p.query, 0, 1, width, width, depth, kernel);
    qsa_sel_ext_pooled(
        gpu,
        p.pooled,
        "qsa_pooled_keys",
        p.block_count,
        p.index_dim,
        depth,
        kernel,
    );
    qsa_sel_ext_selected(gpu, p.selected, 0, 1, p.capacity, last_row, depth, kernel);
}

/// Fused `indexed_attention_select_{f32,bf16}_batched{,_serial,_exact}`, grid
/// `rows` (no rounding). Per row: query `index_heads * index_dim` at
/// `row * query_row_stride`; pooled blocks `< row_block_count` (guarded; the last
/// row's `rbc` is the maximum); selected row `row * capacity + [0, capacity)`;
/// mirror (last row only) `[0, capacity)`; with global scores (`scores_global`
/// non-null) `row * scores_stride + [0, row_block_count)` floats, written and
/// read back (including the float4 reads, which stop at `row_block_count`). The
/// serial kernels also read `row_selected[prior * compress]` unguarded.
fn qsa_sel_ext_fused(
    gpu: &Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    scores_global: *mut std::ffi::c_void,
    scores_stride: usize,
    serial: bool,
    site: &'static str,
) {
    let live = qsa_sel_ext_rbc_last(p.rows, p.position_start, p.compress, p.block_count);
    let depth = qsa_sel_ext_depth(p.rows, p.position_start, p.block_count);
    qsa_sel_ext_query(
        gpu,
        p.query,
        0,
        p.rows,
        p.query_row_stride,
        p.index_heads.saturating_mul(p.index_dim),
        depth,
        site,
    );
    qsa_sel_ext_pooled(
        gpu,
        p.pooled,
        "qsa_pooled_keys",
        live,
        p.index_dim,
        depth,
        site,
    );
    let last_row = if serial {
        p.capacity.max(qsa_sel_ext_serial_prior(
            p.budget_blocks.min(live),
            p.compress,
        ))
    } else {
        p.capacity
    };
    qsa_sel_ext_selected(
        gpu, p.selected, 0, p.rows, p.capacity, last_row, depth, site,
    );
    if let Some(m) = mirror {
        qsa_sel_ext_mirror(gpu, m, p.capacity, depth, site);
    }
    if !scores_global.is_null() {
        gpu.qsa_extent_check_ptr(
            "qsa_select_scores",
            scores_global,
            0,
            qsa_sel_ext_span(p.rows, scores_stride, live).saturating_mul(QSA_SEL_EXT_WORD),
            depth,
            site,
        );
    }
}

/// Score launch of one row group (`n` rows from `g0`) into `scores`
/// (`[n][stride]` F32 scratch). Hipcc F32 rows8/16 and the PM F32/BF16 rows16
/// kernels guard the pooled read by the tile lane (`block < block_count`; the PM
/// descriptor is `num_records = min(block_count - 256 * wg_x, 256) * row bytes`),
/// not by row visibility, so `pooled_guarded_by_live == false` checks
/// `block_count` pooled rows; the hipcc BF16 split kernels return a tile past
/// `live = min(block_count, (position_start + row0 + nrows) / compress)` and
/// check `live` rows. Scores are written `row * stride + [0, min(block_count,
/// visible_blocks(row)))`: `(n - 1) * stride + live` floats. The query is
/// `4 * index_dim` per row, `(n - 1) * query_row_stride + 4 * index_dim`.
/// Grid-round: `ceil(block_count / 256)` (or `ceil(bound / 256)`) tiles and
/// `ceil(n / R)` row tiles; surplus tiles/lanes/rows return before any access.
#[allow(clippy::too_many_arguments)]
fn qsa_sel_ext_score(
    gpu: &Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    g0: usize,
    n: usize,
    pooled: &GpuTensor,
    pooled_guarded_by_live: bool,
    scores: *mut std::ffi::c_void,
    scores_arena: &'static str,
    stride: usize,
    site: &'static str,
    post: bool,
) {
    let position_start = p.position_start.saturating_add(g0);
    let live = qsa_sel_ext_rbc_last(n, position_start, p.compress, p.block_count);
    let depth = qsa_sel_ext_depth(n, position_start, p.block_count);
    let scores_bytes = qsa_sel_ext_span(n, stride, live).saturating_mul(QSA_SEL_EXT_WORD);
    // The actual operand read: the BF16 mirror (`pooled` is not `p.pooled`) or
    // the canonical pooled arena. Both report the mirror's real valid-blocks
    // watermark (`p.pooled_bf16_valid_blocks`; 0 = no valid mirror), so an F32
    // fallback taken while the mirror lags `block_count` stays visible.
    let valid_blocks = p.pooled_bf16_valid_blocks;
    let pooled_arena = if !std::ptr::eq(pooled, p.pooled) {
        "mirror"
    } else if pooled.dtype == DType::F32 {
        "pooled_f32"
    } else {
        "pooled_bf16"
    };
    let pooled_blocks = if pooled_guarded_by_live {
        live
    } else {
        p.block_count
    };
    let pooled_bytes = pooled_blocks
        .saturating_mul(p.index_dim)
        .saturating_mul(pooled.dtype.size());
    if post {
        gpu.qsa_extent_post_launch_pooled(
            pooled_arena,
            pooled,
            0,
            pooled_bytes,
            depth,
            site,
            valid_blocks,
        );
        return;
    }
    qsa_sel_ext_query(
        gpu,
        p.query,
        g0,
        n,
        p.query_row_stride,
        p.index_heads.saturating_mul(p.index_dim),
        depth,
        site,
    );
    gpu.qsa_extent_check_pooled(
        pooled_arena,
        pooled,
        0,
        pooled_bytes,
        depth,
        site,
        valid_blocks,
    );
    gpu.qsa_extent_check_ptr(scores_arena, scores, 0, scores_bytes, depth, site);
}

/// `indexed_attention_select_from_scores{,_pm_gfx1151}` of one row group, grid
/// `n` (no rounding). Scores are read `row * stride + [0, rbc(row))`
/// (guarded: the hipcc kernel loops `block < row_block_count`, the PM kernel's
/// 32-key packet lanes past `rbc` read 0 through the descriptor), selected
/// `row * capacity + [0, capacity)` (hipcc loop `i < capacity`; PM descriptor
/// `num_records` drops slots past `capacity`), mirror (last row, non-null) `[0,
/// capacity)`.
#[allow(clippy::too_many_arguments)]
fn qsa_sel_ext_from_scores(
    gpu: &Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    g0: usize,
    n: usize,
    scores: *mut std::ffi::c_void,
    scores_arena: &'static str,
    stride: usize,
    mirror: Option<&GpuTensor>,
    site: &'static str,
    post: bool,
) {
    let position_start = p.position_start.saturating_add(g0);
    if post {
        qsa_sel_ext_post_selected(
            gpu,
            p.selected,
            g0,
            n,
            p.capacity,
            position_start,
            p.block_count,
            site,
        );
        return;
    }
    let live = qsa_sel_ext_rbc_last(n, position_start, p.compress, p.block_count);
    let depth = qsa_sel_ext_depth(n, position_start, p.block_count);
    gpu.qsa_extent_check_ptr(
        scores_arena,
        scores,
        0,
        qsa_sel_ext_span(n, stride, live).saturating_mul(QSA_SEL_EXT_WORD),
        depth,
        site,
    );
    qsa_sel_ext_selected(gpu, p.selected, g0, n, p.capacity, p.capacity, depth, site);
    if let Some(m) = mirror {
        qsa_sel_ext_mirror(gpu, m, p.capacity, depth, site);
    }
}

/// One launch of the filter trio (`kernel` names it; `t.rows` rows, grid `rows`
/// for threshold/finish, `ceil(bound / 4096) x rows` for compact; the surplus
/// tiles return at `base >= rbc`):
/// - threshold: scores `row * stride + b`, `b < rbc` sampled (guarded; only rows
///   with `rbc > cap` read), ctl `[2 * row, 2 * row + 1]` u64 (`rows * 16` B);
/// - compact: scores `b < rbc` (guarded by `b < rbc`), ctl word 1 (atomic add),
///   cand `row * cap + slot`, `slot < cap` (guarded): `rows * cap * 8` B;
/// - finish: scores (repair path, `i < rbc`), ctl, cand `i < count <= cap`,
///   selected `row * capacity + [0, capacity)`, mirror `[0, capacity)`.
/// Scores are `(rows - 1) * stride + rbc_last` floats; scratch-owned operands go
/// through the raw-pointer helper, tensor-owned ones (selected, mirror, lab
/// scores) through the tensor helper at the group's row offset.
fn qsa_sel_ext_filter(gpu: &Gpu, t: &QsaFilterTrio<'_>, kernel: &'static str, post: bool) {
    let live = qsa_sel_ext_rbc_last(t.rows, t.position_start, t.compress, t.block_count);
    let depth = qsa_sel_ext_depth(t.rows, t.position_start, t.block_count);
    let ctl_bytes = t.rows.saturating_mul(16);
    if post {
        gpu.qsa_extent_post_launch_ptr("qsa_filter_ctl", t.ctl, 0, ctl_bytes, depth, kernel);
        return;
    }
    let scores_bytes = qsa_sel_ext_span(t.rows, t.stride, live).saturating_mul(QSA_SEL_EXT_WORD);
    // Built under the same flag as this call (`QsaFilterTrio::extent`).
    let extent = t
        .extent
        .as_ref()
        .expect("QSA filter extent metadata is built whenever the extent check is on");
    match extent.scores {
        Some(owner) => gpu.qsa_extent_check(
            "qsa_filter_scores",
            owner,
            extent
                .first_row
                .saturating_mul(t.stride)
                .saturating_mul(QSA_SEL_EXT_WORD),
            scores_bytes,
            depth,
            kernel,
        ),
        None => gpu.qsa_extent_check_ptr(
            "qsa_filter_scores",
            t.scores,
            0,
            scores_bytes,
            depth,
            kernel,
        ),
    }
    gpu.qsa_extent_check_ptr("qsa_filter_ctl", t.ctl, 0, ctl_bytes, depth, kernel);
    if kernel == "qsa_filter_threshold" {
        return;
    }
    gpu.qsa_extent_check_ptr(
        "qsa_filter_cand",
        t.cand,
        0,
        t.rows
            .saturating_mul(t.knobs.cap)
            .saturating_mul(std::mem::size_of::<u64>()),
        depth,
        kernel,
    );
    if kernel == "qsa_filter_finish" {
        qsa_sel_ext_selected(
            gpu,
            extent.selected,
            extent.first_row,
            t.rows,
            t.capacity,
            t.capacity,
            depth,
            kernel,
        );
        if !t.mirror.is_null() {
            if let Some(m) = extent.mirror {
                qsa_sel_ext_mirror(gpu, m, t.capacity, depth, kernel);
            }
        }
    }
}

pub struct IndexedAttentionSelectBatch<'a> {
    pub query: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    /// Exact BF16 copy of `pooled` rows [0, block_count) (pooled values are BF16-rounded
    /// at pooling). When Some (dtype BF16, numel >= block_count*index_dim) and the gfx1151
    /// PM BF16 score kernel covers the call (same admission as the PM F32 score: eager,
    /// `qsa_select_pm_fits`, HIPFIRE_QWEN4_QSA_SCORE_PM on, pooled F32), the score reads it;
    /// otherwise `pooled`. Selection bytes identical either way.
    pub pooled_bf16: Option<&'a GpuTensor>,
    /// Extent-check metadata only (never a kernel argument): the mirror's
    /// valid-blocks watermark, filled only under
    /// `gpu.flags.qwen4_qsa_extent_check` (0 elsewhere and for callers without a
    /// mirror).
    pub pooled_bf16_valid_blocks: usize,
    pub selected: &'a GpuTensor,
    pub rows: usize,
    pub query_row_stride: usize,
    pub block_count: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub position_start: usize,
    pub capacity: usize,
    /// Declared shape bound for the LDS reservation and the kernel symbol. A
    /// retained tape needs both to be position-independent, so callers pass the
    /// pooling capacity (`>= block_count`); a caller without a capacity passes
    /// the active count. Must be `>= block_count`.
    pub shape_blocks: usize,
}

pub fn indexed_attention_select_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
) -> HipResult<()> {
    indexed_attention_select_batch_impl(gpu, p, None).map(|_| ())
}

/// [`indexed_attention_select_batch`] that also writes the final row's
/// selection into `mirror` (`capacity` i32) on the parallel route; returns
/// whether it did (the caller copies otherwise).
pub fn indexed_attention_select_batch_mirrored(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: &GpuTensor,
) -> HipResult<bool> {
    indexed_attention_select_batch_impl(gpu, p, Some(mirror))
}

/// Shape, extent and integer admission of a batched select call, shared by the
/// live route and the lab entries.
fn qsa_select_batch_validate(p: &IndexedAttentionSelectBatch<'_>) -> HipResult<()> {
    ensure_f32(p.query)?;
    // Pooled keys are F32 or BF16 arenas (`QsaKvFormat::index_dtype`).
    if !matches!(p.pooled.dtype, DType::F32 | DType::BF16) {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    if p.selected.dtype != DType::Raw
        || p.rows == 0
        || p.query_row_stride == 0
        || p.block_count > 0 && p.compress == 0
        || p.index_heads == 0
        || p.index_dim == 0
        || p.compress == 0
        || p.capacity == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_elements = checked_product(p.index_heads, p.index_dim, "QSA batch select query")?;
    if p.query_row_stride < query_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_last = checked_product(
        p.rows - 1,
        p.query_row_stride,
        "QSA batch select query rows",
    )?
    .checked_add(query_elements)
    .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let pooled_elements = checked_product(p.block_count, p.index_dim, "QSA batch select pooled")?;
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA batch select rows")?,
        std::mem::size_of::<i32>(),
        "QSA batch select selected",
    )?;
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if end_position > i32::MAX as usize
        || p.query.numel() < query_last
        || p.pooled.numel() < pooled_elements
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    checked_i32(p.rows, "QSA batch select rows")?;
    checked_i32(p.query_row_stride, "QSA batch select query stride")?;
    checked_i32(p.block_count, "QSA batch select blocks")?;
    checked_i32(p.index_heads, "QSA batch select heads")?;
    checked_i32(p.index_dim, "QSA batch select dim")?;
    checked_i32(p.budget_blocks, "QSA batch select budget")?;
    checked_i32(p.compress, "QSA batch select compress")?;
    checked_i32(p.position_start, "QSA batch select position")?;
    checked_i32(p.capacity, "QSA batch select capacity")?;
    checked_u32(p.rows, "QSA batch select row grid")?;
    // The active count is declared to the recorder as `(position_start + rows) /
    // compress`, so replay re-derives it. Verify the caller used that formula:
    // a mismatch would make replay select against a different block count.
    let expected_blocks = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, "QSA batch select position overflow"))?
        / p.compress;
    if expected_blocks != p.block_count {
        return Err(HipError::new(
            0,
            &format!(
                "QSA batch select block count {} is not (position_start {} + rows {}) / compress {} = {expected_blocks}",
                p.block_count, p.position_start, p.rows, p.compress
            ),
        ));
    }
    // Reserved LDS bytes and kernel symbol come from the *bound*, not from the
    // active count, so a pinned bound makes both position-independent. The
    // kernel indexes its LDS by the active clamp (`row_block_count`), so a
    // larger reservation is never read.
    if p.shape_blocks < p.block_count {
        return Err(HipError::new(
            0,
            &format!(
                "QSA batch select shape bound {} is below the active block count {}",
                p.shape_blocks, p.block_count
            ),
        ));
    }
    p.shape_blocks
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipError::new(0, "QSA batch select shape overflow"))?;
    Ok(())
}

fn indexed_attention_select_batch_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
) -> HipResult<bool> {
    qsa_select_batch_validate(p)?;
    let shape_blocks = p.shape_blocks;
    let parallel =
        gpu.arch_caps.has_qwen4_parallel_qsa_gdn() && shape_blocks > 0 && !select_forced_serial();
    // Live F32 launches use the score/select pair. Covered BF16 prefill
    // launches use the builder score with either select kernel; SCORE_PM=0
    // keeps the fused BF16 selector. The opt-in exact selector keeps its route.
    if parallel
        && !*QWEN4_QSA_SELECT_EXACT
        && (p.pooled.dtype == DType::F32 || (qsa_select_pm_fits(gpu, p) && *QWEN4_QSA_SCORE_PM))
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.index_heads == 4
        && p.index_dim.is_multiple_of(4)
        && p.index_dim <= 128
        && p.block_count > 0
        && p.budget_blocks <= QSA_SELECT_FROM_SCORES_MAX_BUDGET
    {
        let pm = qsa_select_pm_arm(gpu, p);
        return indexed_attention_select_rows8(gpu, p, mirror, pm);
    }
    if let Some(route) = qsa_bf16_split_route(gpu, p) {
        return indexed_attention_select_bf16_split(
            gpu,
            p,
            mirror,
            Some(QSA_FILTER_PRODUCTION),
            route.bounded,
            None,
        );
    }
    indexed_attention_select_fused(gpu, p, mirror)
}

/// Incumbent fused selector, also used as the BF16 lab baseline.
fn indexed_attention_select_fused(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
) -> HipResult<bool> {
    let (parallel_kernel, serial_kernel) = match p.pooled.dtype {
        DType::F32 => (
            "indexed_attention_select_f32_batched",
            "indexed_attention_select_f32_batched_serial",
        ),
        DType::BF16 => (
            "indexed_attention_select_bf16_batched",
            "indexed_attention_select_bf16_batched_serial",
        ),
        _ => return Err(HipError::new(0, &ComputeError::WrongDtype.to_string())),
    };
    let rows = checked_i32(p.rows, "QSA batch select rows")?;
    let query_row_stride = checked_i32(p.query_row_stride, "QSA batch select query stride")?;
    let block_count = checked_i32(p.block_count, "QSA batch select blocks")?;
    let index_heads = checked_i32(p.index_heads, "QSA batch select heads")?;
    let index_dim = checked_i32(p.index_dim, "QSA batch select dim")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA batch select budget")?;
    let compress = checked_i32(p.compress, "QSA batch select compress")?;
    let position_start = checked_i32(p.position_start, "QSA batch select position")?;
    let capacity = checked_i32(p.capacity, "QSA batch select capacity")?;
    let row_grid = checked_u32(p.rows, "QSA batch select row grid")?;
    let shape_blocks = p.shape_blocks;
    let lds_bytes = shape_blocks
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| HipError::new(0, "QSA batch select shape overflow"))?;
    let parallel =
        gpu.arch_caps.has_qwen4_parallel_qsa_gdn() && shape_blocks > 0 && !select_forced_serial();
    // Past the LDS row the scores go to global rows (one per workgroup, at
    // most QSA_SELECT_GLOBAL_ROWS per launch); the selection is the same.
    let global = parallel
        && lds_bytes + QSA_SELECT_BATCHED_STATIC_LDS_BYTES > QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES;
    if global && p.rows > QSA_SELECT_GLOBAL_ROWS {
        let query_elements = p.query.numel();
        let selected_bytes = p.selected.numel();
        let row_bytes = p.capacity * std::mem::size_of::<i32>();
        let mut persisted = false;
        let mut first = 0;
        while first < p.rows {
            let rows = (p.rows - first).min(QSA_SELECT_GLOBAL_ROWS);
            let last = first + rows == p.rows;
            let query_offset = first * p.query_row_stride;
            let query = p
                .query
                .sub_offset(query_offset, query_elements - query_offset);
            let selected = p
                .selected
                .sub_offset(first * row_bytes, selected_bytes - first * row_bytes);
            let group = IndexedAttentionSelectBatch {
                pooled_bf16: p.pooled_bf16,
                pooled_bf16_valid_blocks: p.pooled_bf16_valid_blocks,
                query: &query,
                pooled: p.pooled,
                selected: &selected,
                rows,
                query_row_stride: p.query_row_stride,
                block_count: (p.position_start + first + rows) / p.compress,
                index_heads: p.index_heads,
                index_dim: p.index_dim,
                budget_blocks: p.budget_blocks,
                compress: p.compress,
                position_start: p.position_start + first,
                capacity: p.capacity,
                shape_blocks: p.shape_blocks,
            };
            let select = if p.pooled.dtype == DType::BF16 {
                indexed_attention_select_fused
            } else {
                indexed_attention_select_batch_impl
            };
            persisted = select(gpu, &group, if last { mirror } else { None })?;
            first += rows;
        }
        return Ok(persisted);
    }
    let scores_stride = shape_blocks.div_ceil(4) * 4;
    let scores_global = if global {
        let bytes = QSA_SELECT_GLOBAL_ROWS
            .checked_mul(scores_stride)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HipError::new(0, "QSA select score rows overflow"))?;
        gpu.qsa_select_scores(bytes)?
    } else {
        std::ptr::null_mut()
    };
    let (kernel_name, block, shared_mem) = if global {
        (parallel_kernel, [QSA_SELECT_PARALLEL_THREADS, 1, 1], 0u32)
    } else if parallel && lds_bytes <= u32::MAX as usize {
        (
            parallel_kernel,
            [QSA_SELECT_PARALLEL_THREADS, 1, 1],
            lds_bytes as u32,
        )
    } else {
        (serial_kernel, [1, 1, 1], 0)
    };
    // `HIPFIRE_QWEN4_QSA_SELECT_EXACT`: the exact selector replaces the
    // incumbent parallel kernel (same ABI, grid and block) for live
    // complete <= 2048 blocks and <= 512 budget blocks, never under a
    // recorder or capture; every other launch is the incumbent's.  The
    // exact kernel keeps no static LDS and needs max(6144, live-score-row)
    // bytes of dynamic LDS: the floor applies even in global-score mode,
    // otherwise 4 B per live block.
    let exact_dynamic_need = QSA_SELECT_EXACT_DYNAMIC_LDS_FLOOR_BYTES.max(if global {
        0
    } else {
        p.block_count.saturating_mul(std::mem::size_of::<f32>())
    });
    let launch_exact = qsa_select_exact_arch(&gpu.arch_caps)
        && *QWEN4_QSA_SELECT_EXACT
        && kernel_name == parallel_kernel
        && p.block_count <= QSA_SELECT_EXACT_MAX_BLOCKS
        && p.budget_blocks <= QSA_SELECT_EXACT_MAX_BUDGET_BLOCKS
        && exact_dynamic_need <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode;
    let (launch_kernel, shared_mem) = if launch_exact {
        let exact = match p.pooled.dtype {
            DType::F32 => "indexed_attention_select_f32_batched_exact",
            _ => "indexed_attention_select_bf16_batched_exact",
        };
        gpu.ensure_kernel_public(
            "indexed_attention_select_exact",
            INDEXED_ATTENTION_SELECT_EXACT_SRC,
            exact,
        )?;
        (exact, exact_dynamic_need as u32)
    } else {
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
        (kernel_name, shared_mem)
    };
    let mut args = KernargBlob::new();
    for tensor in [p.query, p.pooled, p.selected] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [rows, query_row_stride] {
        args.push_i32(value);
    }
    args.push_i32(block_count);
    let block_count_offset = args.len() - 4;
    for value in [index_heads, index_dim, budget_blocks, compress] {
        args.push_i32(value);
    }
    args.push_i32(position_start);
    let position_offset = args.len() - 4;
    args.push_i32(capacity);
    let mirror = mirror.filter(|m| {
        kernel_name == parallel_kernel
            && m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>()
    });
    if kernel_name == parallel_kernel {
        args.push_ptr(mirror.map_or(std::ptr::null_mut(), |m| m.buf.as_ptr()));
        args.push_ptr(scores_global);
        args.push_i32(checked_i32(scores_stride, "QSA select score stride")?);
    }
    args.pad_to(16);
    // Both declared fields make the selection follow the replay position instead
    // of the capture position.
    let addend =
        u32::try_from(p.rows).map_err(|_| HipError::new(0, "QSA batch select rows exceed u32"))?;
    let divisor = u32::try_from(p.compress)
        .map_err(|_| HipError::new(0, "QSA batch select compression exceeds u32"))?;
    let bindings = [
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: block_count_offset,
            addend,
            divisor,
        },
        crate::replay::ReplayKernargBinding::PositionPlusU32 {
            offset: position_offset,
            addend: 0,
        },
    ];
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_fused(
            gpu,
            p,
            mirror,
            scores_global,
            scores_stride,
            kernel_name == serial_kernel,
            launch_kernel,
        );
    }
    gpu.launch_blob_recorded(
        launch_kernel,
        [row_grid, 1, 1],
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &bindings,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_post_selected(
            gpu,
            p.selected,
            0,
            p.rows,
            p.capacity,
            p.position_start,
            p.block_count,
            launch_kernel,
        );
    }
    Ok(mirror.is_some())
}

/// Score-scratch budget of [`indexed_attention_select_rows8`]: rows are
/// selected in groups whose scores fit it.
const QSA_SELECT_SCORE_SCRATCH_BYTES: usize = 64 << 20;
/// `indexed_attention_select_from_scores` holds the chosen blocks in 512 LDS
/// entries; larger budgets take the batched kernel.
const QSA_SELECT_FROM_SCORES_MAX_BUDGET: usize = 512;

/// Which kernels of the live pair come from certified builder modules.
/// For BF16, `Hipcc` is the fused incumbent, `ScoreOnly` and `Both` use the
/// BF16 PM score; `SelectOnly` is unsupported (no hipcc BF16 rows score).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QsaSelectPmArm {
    Hipcc,
    ScoreOnly,
    SelectOnly,
    Both,
}
impl QsaSelectPmArm {
    fn score(self) -> bool {
        matches!(self, Self::ScoreOnly | Self::Both)
    }
    fn select(self) -> bool {
        matches!(self, Self::SelectOnly | Self::Both)
    }
}

/// Pooled blocks the builder selector serves: its per-lane bit-sliced
/// counters take at most 511 packets of 32 keys per wave.
const QSA_SELECT_PM_MAX_BLOCKS: usize = 511 * 8 * 32;
/// Prefill admission remains independent of the decode kill switch.
const QSA_SELECT_PM_MIN_ROWS: usize = 512;

/// The exact selector (`indexed_attention_select_exact.hip`) is `mov_dpp8`
/// code, gfx11+ ISA only: gfx942 gets the parallel incumbent selector, never
/// this one, even under `HIPFIRE_QWEN4_QSA_SELECT_EXACT=1`.
fn qsa_select_exact_arch(caps: &crate::arch_caps::ArchCaps) -> bool {
    caps.has_gfx11_plus_simt()
}

/// Whether the builder pair covers the live-pair call `p`: exact gfx1151,
/// admitted rows, the fixed four-head / 128-dim F32-query geometry with F32 or
/// BF16 pooled keys, and every buffer addressed through 32-bit raw-buffer
/// offsets below the out-of-range offset (pooled extents use its element size).
fn qsa_select_pm_fits(gpu: &Gpu, p: &IndexedAttentionSelectBatch<'_>) -> bool {
    let rows = gpu.route_rows(p.rows).max(p.rows);
    if rows < QSA_SELECT_PM_MIN_ROWS && !*QWEN4_QSA_SELECT_PM_DECODE {
        return false;
    }
    qsa_select_pm_shape_fits(
        // Threshold input under the invariant profile; never below the real
        // rows, which the shape's raw-offset bounds are checked against.
        gpu.arch_caps.is_gfx1151(),
        p.pooled.dtype,
        rows,
        p.index_heads,
        p.index_dim,
        p.block_count,
        p.query_row_stride,
        p.capacity,
    )
}

/// No tensor payload or GPU initialization is needed to choose the native
/// score/select image. Keep raw-offset admissions shared with live dispatch.
pub(crate) fn qsa_select_pm_shape_fits(
    halo: bool,
    dtype: DType,
    rows: usize,
    index_heads: usize,
    index_dim: usize,
    block_count: usize,
    query_row_stride: usize,
    capacity: usize,
) -> bool {
    const PM_OOB_OFFSET: usize = 0x7fff_ff00;
    let below = |n: Option<usize>| n.is_some_and(|bytes| bytes < PM_OOB_OFFSET);
    halo && matches!(dtype, DType::F32 | DType::BF16)
        && rows >= 1
        && index_heads == 4
        && index_dim == 128
        && block_count <= QSA_SELECT_PM_MAX_BLOCKS
        && below(
            rows.checked_mul(query_row_stride)
                .and_then(|n| n.checked_mul(4)),
        )
        && below(block_count.checked_mul(128 * dtype.size()))
        && below(rows.checked_mul(capacity).and_then(|n| n.checked_mul(4)))
}

pub(crate) fn qsa_select_pm_selected(fits: bool, score: bool, select: bool) -> QsaSelectPmArm {
    if !fits {
        return QsaSelectPmArm::Hipcc;
    }
    match (score, select) {
        (false, false) => QsaSelectPmArm::Hipcc,
        (true, false) => QsaSelectPmArm::ScoreOnly,
        (false, true) => QsaSelectPmArm::SelectOnly,
        (true, true) => QsaSelectPmArm::Both,
    }
}

/// The route's arm: `HIPFIRE_QWEN4_QSA_SCORE_PM` / `HIPFIRE_QWEN4_QSA_SELECT_PM`
/// (each on unless `0`) where the builder pair covers the call.
fn qsa_select_pm_arm(gpu: &Gpu, p: &IndexedAttentionSelectBatch<'_>) -> QsaSelectPmArm {
    if !qsa_select_pm_fits(gpu, p) {
        return QsaSelectPmArm::Hipcc;
    }
    qsa_select_pm_selected(true, *QWEN4_QSA_SCORE_PM, *QWEN4_QSA_SELECT_PM)
}

/// Lab entry with an explicit arm (grouping, scratch and mirror persistence;
/// only the env choice is bypassed). Refuses uncovered calls before launch.
/// BF16 `Hipcc` uses the fused incumbent; `ScoreOnly`/`Both` use PM scores
/// with hipcc/PM selection, and `SelectOnly` is refused.
#[cfg(any(test, feature = "lab"))]
pub fn indexed_attention_select_batch_pm_arm(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    arm: QsaSelectPmArm,
) -> HipResult<bool> {
    let live = gpu.arch_caps.has_gfx11_plus_simt()
        && matches!(p.pooled.dtype, DType::F32 | DType::BF16)
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.index_heads == 4
        && p.index_dim.is_multiple_of(4)
        && p.index_dim <= 128
        && p.rows >= 1
        && p.budget_blocks <= QSA_SELECT_FROM_SCORES_MAX_BUDGET;
    let fits = qsa_select_pm_shape_fits(
        gpu.arch_caps.is_gfx1151(),
        p.pooled.dtype,
        gpu.route_rows(p.rows).max(p.rows),
        p.index_heads,
        p.index_dim,
        p.block_count,
        p.query_row_stride,
        p.capacity,
    );
    if !live || (arm != QsaSelectPmArm::Hipcc && !fits) {
        return Err(HipError::new(0, "QSA select PM arm: call not covered"));
    }
    if p.block_count == 0 {
        return indexed_attention_select_fused(gpu, p, mirror);
    }
    if p.pooled.dtype == DType::BF16 {
        if arm == QsaSelectPmArm::SelectOnly {
            return Err(HipError::new(0, "QSA select BF16 requires PM score"));
        }
        if arm == QsaSelectPmArm::Hipcc {
            return indexed_attention_select_fused(gpu, p, mirror);
        }
    }
    indexed_attention_select_rows8(gpu, p, mirror, arm)
}

/// Whether an `rows`-row QSA call is an exact-wide verify call: more rows than
/// the historic few-row limit, admitted by the shared [`Gpu::few_row_limit`]
/// (eager verify only, `FEW_ROW_ROUTE_ROWS` while recording / capturing /
/// outside verify) and not under the prefill invariant profile (whose
/// `route_rows` is the canonical chunk, far above any limit). With the limit
/// at 8 this is never true, so prefill and every non-verify route are
/// unchanged.
fn qsa_exact_wide_rows(gpu: &Gpu, rows: usize) -> bool {
    let limit = gpu.few_row_limit();
    rows > crate::prefill_invariant::FEW_ROW_ROUTE_ROWS
        && rows <= limit
        && gpu.route_rows(rows) <= limit
}

/// Score implementation of a QSA select call, for [`qsa_select_score_symbol`].
#[derive(Clone, Copy)]
enum QsaScoreArm {
    PmBf16,
    PmF32,
    HipF32,
    HipBf16,
}

/// `HIPFIRE_QWEN4_QSA_SCORE_ROWS32` (developer, default off, `1` on): score
/// admitted exact-wide verify calls of 17..=32 rows ([`qsa_exact_wide_rows`])
/// with ONE rows32 tile (`indexed_attention_select_scores_rows32_*`, one pooled
/// read per 32 rows) instead of two rows16 tiles. Same score bits either way;
/// a performance choice kept unmeasured and off until the rows32 tile is shown
/// to beat two rows16 tiles at the claimed depths. Index width 4 x 128 only.
/// Read once.
static QWEN4_QSA_SCORE_ROWS32: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SCORE_ROWS32", false)
});

/// Whether a `rows`-row select call of `index_heads x index_dim` takes the
/// rows32 score symbols: the flag, 17..=few_row_limit admitted rows (never
/// prefill, recording or capture) and the only geometry the symbols implement.
fn qsa_score_rows32(gpu: &Gpu, rows: usize, index_heads: usize, index_dim: usize) -> bool {
    *QWEN4_QSA_SCORE_ROWS32
        && rows > 16
        && qsa_exact_wide_rows(gpu, rows)
        && index_heads == 4
        && index_dim == 128
}

/// Score kernel of a QSA select route: its rows per workgroup (the grid-y
/// divisor, ALWAYS the symbol's own tile) and whether its pooled read is
/// guarded by the tile's live block count (extent hooks).
struct QsaScoreSymbol {
    kernel: &'static str,
    rows: usize,
    /// The kernel returns a workgroup past `min(block_count, (position_start
    /// + row0 + nrows) / compress)` and loads no pooled key past it; false:
    /// every lane `< block_count` of the tile reads its key (old PM and hipcc
    /// F32 rows8/16 read B blocks, even where no row sees them).
    pooled_live_guarded: bool,
}

/// Score kernel and rows per score workgroup of a QSA select route.
/// `many_rows` is `route_rows > 8`: the PM arms are rows16 for every admitted
/// call, the hipcc arms rows16 above eight route rows else rows8. `rows32`
/// ([`qsa_score_rows32`]) selects the rows32 twin of the arm (PM: reads B
/// like rows16; hipcc: live-guarded).
///
/// Without rows32 every admitted call of 9..=32 rows is tiled by the rows16
/// symbols (9..=16 rows one tile, 17..=32 rows two tiles). Both the live
/// PM/F32 route and the gfx1201 BF16 split route
/// ([`indexed_attention_select_bf16_split`]) take symbol and tile rows from
/// this function and carry them into the grid, so symbol and divisor never
/// diverge.
fn qsa_select_score_symbol(arm: QsaScoreArm, many_rows: bool, rows32: bool) -> QsaScoreSymbol {
    let (kernel, rows, pooled_live_guarded) = match (arm, rows32, many_rows) {
        (QsaScoreArm::PmBf16, true, _) => (
            "indexed_attention_select_scores_rows32_bf16_pm_gfx1151",
            32,
            false,
        ),
        (QsaScoreArm::PmF32, true, _) => (
            "indexed_attention_select_scores_rows32_f32_pm_gfx1151",
            32,
            false,
        ),
        (QsaScoreArm::HipF32, true, _) => ("indexed_attention_select_scores_rows32_f32", 32, true),
        (QsaScoreArm::HipBf16, true, _) => {
            ("indexed_attention_select_scores_rows32_bf16", 32, true)
        }
        (QsaScoreArm::PmBf16, false, _) => (
            "indexed_attention_select_scores_rows16_bf16_pm_gfx1151",
            16,
            false,
        ),
        (QsaScoreArm::PmF32, false, _) => (
            "indexed_attention_select_scores_rows16_f32_pm_gfx1151",
            16,
            false,
        ),
        (QsaScoreArm::HipF32, false, true) => {
            ("indexed_attention_select_scores_rows16_f32", 16, false)
        }
        (QsaScoreArm::HipF32, false, false) => {
            ("indexed_attention_select_scores_rows8_f32", 8, false)
        }
        (QsaScoreArm::HipBf16, false, true) => {
            ("indexed_attention_select_scores_rows16_bf16", 16, true)
        }
        (QsaScoreArm::HipBf16, false, false) => {
            ("indexed_attention_select_scores_rows8_bf16", 8, true)
        }
    };
    QsaScoreSymbol {
        kernel,
        rows,
        pooled_live_guarded,
    }
}

/// Live score/select route: F32 uses hipcc rows8/16 or PM rows16 scores.
/// BF16 requires the PM rows16 score from `qsa_select_bf16_pm_gfx1151`;
/// there is no hipcc BF16 rows score. Both dtypes select using hipcc or the
/// existing PM select module, with identical kernargs, grids and output bytes.
/// `HIPFIRE_QWEN4_QSA_FILTER_SELECT=1` replaces the select launch of large,
/// eager, gfx11+ calls with the exact filter trio ([`qsa_filter_production`]).
fn indexed_attention_select_rows8(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    pm: QsaSelectPmArm,
) -> HipResult<bool> {
    let filter = qsa_filter_production(gpu, p, pm);
    indexed_attention_select_rows8_core(gpu, p, mirror, pm, filter, None)
}

/// [`indexed_attention_select_rows8`] with explicit filter knobs (`None` keeps
/// the select kernel) and optional per-row `[threshold, count | repaired << 32]`
/// words of the filter control block, collected in row order (a host
/// download per group: labs only; production passes `None`).
fn indexed_attention_select_rows8_core(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    pm: QsaSelectPmArm,
    filter: Option<QsaFilterCfg>,
    mut stats: Option<&mut Vec<[u64; 2]>>,
) -> HipResult<bool> {
    const PM_MODULE: &str = "qsa_select_pm_gfx1151";
    if p.pooled.dtype == DType::BF16 && !pm.score() {
        return Err(HipError::new(0, "QSA select BF16 requires PM score"));
    }
    if let Some(knobs) = filter {
        qsa_filter_cfg_check(knobs)?;
    }
    let stride = p.block_count;
    // PM scores sixteen rows per tile for every admitted call (thirty-two with
    // `HIPFIRE_QWEN4_QSA_SCORE_ROWS32` on 17..=32 verify rows). The hipcc
    // fallback retains eight-row scoring for decode and few-row verify.
    // A BF16 copy of an F32 pooled arena feeds the BF16 PM score when the call
    // would run the PM F32 score; it is never read otherwise.
    let bf16_copy = p.pooled_bf16.filter(|m| {
        m.dtype == DType::BF16
            && p.pooled.dtype == DType::F32
            && pm.score()
            && p.block_count
                .checked_mul(p.index_dim)
                .is_some_and(|elements| m.numel() >= elements)
    });
    let many_rows = gpu.route_rows(p.rows) > 8;
    let score_sym = qsa_select_score_symbol(
        if p.pooled.dtype == DType::BF16 || bf16_copy.is_some() {
            QsaScoreArm::PmBf16
        } else if pm.score() {
            QsaScoreArm::PmF32
        } else {
            QsaScoreArm::HipF32
        },
        many_rows,
        qsa_score_rows32(gpu, p.rows, p.index_heads, p.index_dim),
    );
    let (score_kernel, score_rows) = (score_sym.kernel, score_sym.rows);
    let select_kernel = if pm.select() {
        "indexed_attention_select_from_scores_pm_gfx1151"
    } else {
        "indexed_attention_select_from_scores"
    };
    let pooled_ptr = bf16_copy.unwrap_or(p.pooled).buf.as_ptr();
    let score_group = (QSA_SELECT_SCORE_SCRATCH_BYTES / (stride.max(1) * 4) / 16 * 16).max(16);
    // Growth goes through the accessor that invalidates captured state first.
    let (group, scores, cand, ctl) = if let Some(knobs) = filter {
        let score_rows = qsa_filter_score_group(stride).ok_or_else(|| {
            HipError::new(
                0,
                "QSA filter: one score row exceeds the score scratch budget",
            )
        })?;
        let group = score_rows
            .min(qsa_filter_group_limit(knobs.cap))
            .min(p.rows);
        let score_bytes = group
            .checked_mul(stride)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HipError::new(0, "QSA filter score scratch overflow"))?;
        let s = qsa_filter_scratch(gpu, score_bytes, group, knobs.cap)?;
        (group, s.scores, s.cand, s.ctl)
    } else {
        let group = score_group.min(p.rows);
        let scores = gpu.qwen4_f16_x_scratch(group * stride * 2)?.buf.as_ptr();
        (group, scores, std::ptr::null_mut(), std::ptr::null_mut())
    };
    if bf16_copy.is_some() || p.pooled.dtype == DType::BF16 {
        gpu.ensure_embedded_kernel(
            "qsa_select_bf16_pm_gfx1151",
            crate::kernels::QSA_SELECT_BF16_PM_GFX1151,
            score_kernel,
        )?;
    } else if pm.score() {
        gpu.ensure_embedded_kernel(
            PM_MODULE,
            crate::kernels::QSA_SELECT_PM_GFX1151,
            score_kernel,
        )?;
    } else {
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, score_kernel)?;
    }
    if filter.is_some() {
        qsa_filter_ensure_kernels(gpu)?;
    } else if pm.select() {
        gpu.ensure_embedded_kernel(
            PM_MODULE,
            crate::kernels::QSA_SELECT_PM_GFX1151,
            select_kernel,
        )?;
    } else {
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, select_kernel)?;
    }
    let mirror =
        mirror.filter(|m| m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>());
    let block_count = checked_i32(p.block_count, "QSA batch select blocks")?;
    let block_tiles = checked_u32(p.block_count.div_ceil(256), "QSA select block tiles")?;
    let mut g0 = 0usize;
    while g0 < p.rows {
        let n = group.min(p.rows - g0);
        let query = unsafe {
            (p.query.buf.as_ptr() as *mut f32).add(g0 * p.query_row_stride) as *mut std::ffi::c_void
        };
        let selected = unsafe {
            (p.selected.buf.as_ptr() as *mut i32).add(g0 * p.capacity) as *mut std::ffi::c_void
        };
        let position_start = checked_i32(p.position_start + g0, "QSA batch select position")?;
        let rows = checked_i32(n, "QSA batch select rows")?;
        let query_row_stride = checked_i32(p.query_row_stride, "QSA batch select query stride")?;
        // No live block: nothing to score (a zero grid is not launchable).
        if p.block_count > 0 {
            let mut args = KernargBlob::new();
            args.push_ptr(query);
            args.push_ptr(pooled_ptr);
            args.push_ptr(scores);
            args.push_i32(rows);
            args.push_i32(query_row_stride);
            args.push_i32(block_count);
            args.push_i32(checked_i32(p.index_dim, "QSA batch select dim")?);
            args.push_i32(checked_i32(p.compress, "QSA batch select compress")?);
            args.push_i32(position_start);
            args.push_i32(block_count);
            args.pad_to(16);
            if gpu.flags.qwen4_qsa_extent_check {
                qsa_sel_ext_score(
                    gpu,
                    p,
                    g0,
                    n,
                    bf16_copy.unwrap_or(p.pooled),
                    score_sym.pooled_live_guarded,
                    scores,
                    "fp16_x_scratch",
                    p.block_count,
                    score_kernel,
                    false,
                );
            }
            gpu.launch_blob_recorded(
                score_kernel,
                [
                    block_tiles,
                    checked_u32(n.div_ceil(score_rows), "QSA select row groups")?,
                    1,
                ],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
                crate::dispatch::ReplayLaunchBindings::NONE,
            )?;
            if gpu.flags.qwen4_qsa_extent_check {
                qsa_sel_ext_score(
                    gpu,
                    p,
                    g0,
                    n,
                    bf16_copy.unwrap_or(p.pooled),
                    score_sym.pooled_live_guarded,
                    scores,
                    "fp16_x_scratch",
                    p.block_count,
                    score_kernel,
                    true,
                );
            }
        }
        // The persistent selection is the final row's: the last group's.
        let last = g0 + n == p.rows;
        let mirror_ptr = mirror
            .filter(|_| last)
            .map_or(std::ptr::null_mut(), |m| m.buf.as_ptr());
        if let Some(knobs) = filter {
            let extent = gpu.flags.qwen4_qsa_extent_check.then(|| QsaFilterExtent {
                selected: p.selected,
                first_row: g0,
                mirror,
                scores: None,
            });
            qsa_filter_launch(
                gpu,
                &QsaFilterTrio {
                    scores,
                    stride: p.block_count.max(1),
                    ctl,
                    cand,
                    selected,
                    mirror: mirror_ptr,
                    rows: n,
                    block_count: p.block_count,
                    budget: p.budget_blocks,
                    compress: p.compress,
                    position_start: p.position_start + g0,
                    capacity: p.capacity,
                    knobs,
                    extent,
                },
            )?;
            if let Some(out) = stats.as_mut() {
                out.extend(qsa_filter_read_ctl(gpu, ctl, n)?);
            }
            g0 += n;
            continue;
        }
        let mut args = KernargBlob::new();
        args.push_ptr(scores);
        args.push_i32(block_count);
        args.push_ptr(selected);
        args.push_i32(rows);
        args.push_i32(block_count);
        args.push_i32(checked_i32(p.budget_blocks, "QSA batch select budget")?);
        args.push_i32(checked_i32(p.compress, "QSA batch select compress")?);
        args.push_i32(position_start);
        args.push_i32(checked_i32(p.capacity, "QSA batch select capacity")?);
        args.push_ptr(mirror_ptr);
        args.pad_to(16);
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_from_scores(
                gpu,
                p,
                g0,
                n,
                scores,
                "fp16_x_scratch",
                p.block_count,
                mirror.filter(|_| last),
                select_kernel,
                false,
            );
        }
        gpu.launch_blob_recorded(
            select_kernel,
            [rows as u32, 1, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_from_scores(
                gpu,
                p,
                g0,
                n,
                scores,
                "fp16_x_scratch",
                p.block_count,
                None,
                select_kernel,
                true,
            );
        }
        g0 += n;
    }
    Ok(mirror.is_some())
}

/// `HIPFIRE_QWEN4_QSA_BF16_SPLIT` (on unless `0`) routes BF16-pooled QSA selection on exact
/// gfx1201 through the split score / select pair
/// ([`indexed_attention_select_bf16_split`]): the exact hipcc BF16 score kernels
/// of `kernels/src/qsa_bf16_scores.hip`, then the exact filter trio
/// (`qsa_filter_select.hip`), both from the dedicated
/// [`Gpu::qsa_select_scores`] slot, from [`QSA_BF16_SPLIT_MIN_BLOCKS`] pooled
/// blocks. Selected indices and the mirror are byte-identical to the fused
/// `indexed_attention_select_bf16_batched` incumbent. Default on, and only on
/// exact gfx1201 and exact gfx1100 ([`qsa_bf16_split_arch`]; gfx1100 has the
/// kill switch `HIPFIRE_QWEN4_QSA_SPLIT_GFX1100=0`; the other architectures
/// keep the incumbent whatever the switch says); `0` is the kill switch and always
/// keeps the incumbent, as does `HIPFIRE_QWEN4_QSA_FILTER_SELECT=0` (the
/// select-from-scores arm never beats the filter on gfx1201). Read once.
static QWEN4_QSA_BF16_SPLIT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_BF16_SPLIT", true)
});

/// Pooled blocks from which the split route replaces the fused BF16 selector
/// for calls of at most [`QSA_BF16_SPLIT_MANY_ROWS`] rows (decode, verify, small
/// prefill tails). Measured on gfx1201 (R9700, graph-replayed GPU time per call,
/// graph of 16 calls): the filter route is ahead of the fused selector from
/// 1024 blocks at 1 and 8 rows and from 2048 blocks at 16 rows (1 / 8 / 16 rows
/// at 2048 blocks: fused 64.6 / 64.3 / 67.6 us, filter 40.1 / 43.6 / 65.8 us;
/// at 65536 blocks fused 552 / 546 / 545 us, filter 313 / 263 / 211 us), and
/// from 1024 blocks at 64 and 256 rows. Re-measured on gfx1100 (7900 XTX,
/// `qsa_bf16_split_crossover_table`, one launch per call, us per call fused vs
/// filter for 1 / 8 / 16 rows): at 1024 blocks 48.2 / 57.1 / 57.9 vs 38.3 /
/// 40.9 / 55.7; at 2048 blocks 61.8 / 61.8 / 64.6 vs 42.1 / 44.5 / 59.3; at
/// 8192 blocks 106.6 / 106.2 / 106.5 vs 50.6 / 53.5 / 69.7; at 32768 blocks
/// 326.0 / 330.1 / 345.3 vs 77.5 / 83.8 / 126.2; at 65536 blocks 616.3 / 644.3
/// / 657.8 vs 112.8 / 119.9 / 199.4, so the 2048 crossover holds on gfx1100
/// for up to 16 rows.
const QSA_BF16_SPLIT_MIN_BLOCKS: usize = 2048;
/// Rows above which a call counts as prefill for the crossover.
const QSA_BF16_SPLIT_MANY_ROWS: usize = 256;
/// Crossover for prefill-sized calls: score time dominates there and the filter
/// is 0.5% (1024 rows) to 7% (2048 rows) behind the fused selector at 2048
/// blocks and 2.5% to 14% behind at 4096, ahead from 8192 blocks (1024 / 2048
/// rows at 8192 blocks: fused 1838 / 3452 us, filter 1670 / 3309 us).
/// Measured on gfx1201 and reused unchanged on gfx1100.
const QSA_BF16_SPLIT_MIN_BLOCKS_MANY_ROWS: usize = 8192;
/// Rows per score / select group: with the 64 MiB score budget this bounds the
/// scratch at a constant ([`reserve_qsa_bf16_split_scratch`] reserves it), so
/// no launch grows the slot once reserved.
const QSA_BF16_SPLIT_MAX_GROUP_ROWS: usize = 512;

struct QsaBf16SplitRoute {
    /// Recording: grids, strides and scratch come from the position-independent
    /// shape bound and the position fields are declared to the recorder.
    bounded: bool,
}

/// `HIPFIRE_QWEN4_QSA_SPLIT_GFX1100` (on unless `0`) is the kill switch for
/// exact gfx1100 (7900 XTX) on the BF16 split route: unset/1 admits it under
/// exactly the gfx1201 conditions and crossovers (measured on gfx1100, see
/// [`QSA_BF16_SPLIT_MIN_BLOCKS`]); `0` restores the fused selector there.
/// Byte identity on gfx1100 is proven by the route tests (knob on/off) and the
/// exhaustive score gate (255 / 257 / 2048 / 8192 / 32768 / 65536 blocks,
/// 27,648 launches bit-exact over 4.3e9 cells). Read once.
static QWEN4_QSA_SPLIT_GFX1100: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_SPLIT_GFX1100", true)
});

/// Pure admission predicate behind [`qsa_bf16_split_arch`].
fn qsa_bf16_split_arch_admits(
    is_gfx1201: bool,
    is_gfx1100: bool,
    gfx1100_knob: bool,
    parallel_qsa_gdn: bool,
    forced_serial: bool,
) -> bool {
    (is_gfx1201 || (is_gfx1100 && gfx1100_knob)) && parallel_qsa_gdn && !forced_serial
}

/// Exact gfx1201, or exact gfx1100 unless
/// `HIPFIRE_QWEN4_QSA_SPLIT_GFX1100=0`; gfx1151 keeps its PM routes and every
/// other arch keeps the fused selector. The route, the scratch reserve and the
/// expert-residency census all call this one fn, so they agree in a process.
fn qsa_bf16_split_arch(gpu: &Gpu) -> bool {
    qsa_bf16_split_arch_admits(
        gpu.arch_caps.is_gfx1201(),
        gpu.arch_caps.is_gfx1100(),
        *QWEN4_QSA_SPLIT_GFX1100,
        gpu.arch_caps.has_qwen4_parallel_qsa_gdn(),
        select_forced_serial(),
    )
}

/// Geometry the BF16 score kernels cover: four heads, `index_dim % 4 == 0`
/// and at most 128, a budget the select-from-scores LDS bound holds.
fn qsa_bf16_split_covers(p: &IndexedAttentionSelectBatch<'_>) -> bool {
    p.pooled.dtype == DType::BF16
        && p.rows >= 1
        && p.index_heads == 4
        && p.index_dim.is_multiple_of(4)
        && p.index_dim <= 128
        && p.budget_blocks <= QSA_SELECT_FROM_SCORES_MAX_BUDGET
}

/// Production admission of the BF16 split route. Under a recorder the
/// decision reads only the shape bound (`shape_blocks`), so a retained tape
/// never depends on the capture position; eager calls read the live count.
/// Hipgraph capture keeps the fused selector (its replay cannot patch kernel
/// arguments).
fn qsa_bf16_split_route(
    gpu: &Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
) -> Option<QsaBf16SplitRoute> {
    if !*QWEN4_QSA_BF16_SPLIT
        || !QWEN4_QSA_FILTER_SELECT.unwrap_or(true)
        || *QWEN4_QSA_SELECT_EXACT
        || gpu.graphs.capture_mode
        || !qsa_bf16_split_arch(gpu)
        || !qsa_bf16_split_covers(p)
    {
        return None;
    }
    let bounded = gpu.replay.is_recording();
    let blocks = if bounded {
        p.shape_blocks
    } else {
        p.block_count
    };
    let min_blocks = if p.rows > QSA_BF16_SPLIT_MANY_ROWS {
        QSA_BF16_SPLIT_MIN_BLOCKS_MANY_ROWS
    } else {
        QSA_BF16_SPLIT_MIN_BLOCKS
    };
    if blocks == 0 || blocks < min_blocks {
        return None;
    }
    // A single row's scores must fit the score budget.
    qsa_filter_score_group(blocks)?;
    Some(QsaBf16SplitRoute { bounded })
}

/// Byte offsets of one split-route scratch request: `score_bytes` of F32
/// scores, then (filter only, `cap > 0`) `group * cap` u64 candidates and
/// `group` 16-byte control rows, each at a 256-byte aligned offset.
struct QsaBf16SplitLayout {
    cand_off: usize,
    ctl_off: usize,
    total: usize,
}

fn qsa_bf16_split_layout(
    score_bytes: usize,
    group: usize,
    cap: usize,
) -> HipResult<QsaBf16SplitLayout> {
    let cand_off = qsa_filter_align(score_bytes)?;
    let cand_bytes = group
        .checked_mul(cap)
        .and_then(|n| n.checked_mul(std::mem::size_of::<u64>()))
        .ok_or_else(qsa_filter_overflow)?;
    let ctl_off = cand_off
        .checked_add(qsa_filter_align(cand_bytes)?)
        .ok_or_else(qsa_filter_overflow)?;
    let ctl_bytes = if cap == 0 {
        0
    } else {
        group.checked_mul(16).ok_or_else(qsa_filter_overflow)?
    };
    let total = qsa_filter_align(
        ctl_off
            .checked_add(ctl_bytes)
            .ok_or_else(qsa_filter_overflow)?,
    )?;
    checked_extent(total, "QSA BF16 split scratch")?;
    Ok(QsaBf16SplitLayout {
        cand_off,
        ctl_off,
        total,
    })
}

/// Constant upper-bound extent of the split route's score / filter scratch
/// (`MAX_GROUP_ROWS` groups at the full score budget with the filter capacity).
/// Shared by the reservation and [`qsa_bf16_split_scratch_bytes`].
fn qsa_bf16_split_reserve_layout() -> HipResult<QsaBf16SplitLayout> {
    qsa_bf16_split_layout(
        QSA_SELECT_SCORE_SCRATCH_BYTES,
        QSA_BF16_SPLIT_MAX_GROUP_ROWS,
        QSA_FILTER_CAP,
    )
}

/// Whether [`reserve_qsa_bf16_split_scratch`] reserves anything for `pooled`:
/// switch on, exact gfx1201 parallel-QSA arch, BF16 pooled keys. Pure metadata
/// (no allocation, no device call); the runtime block-count crossover is not
/// part of the reservation.
fn qsa_bf16_split_reserves(gpu: &Gpu, pooled: DType) -> bool {
    *QWEN4_QSA_BF16_SPLIT && pooled == DType::BF16 && qsa_bf16_split_arch(gpu)
}

/// Bytes of the QSA split-route selector scratch that
/// [`reserve_qsa_bf16_split_scratch`] allocates for `pooled` keys on `gpu`:
/// the constant score + filter-candidate + control extent, or 0 when the route
/// cannot run (same gates as the reservation). Selector only: the gather path
/// is not included. Does not allocate or touch the device.
pub fn qsa_bf16_split_scratch_bytes(gpu: &Gpu, pooled: DType) -> usize {
    if !qsa_bf16_split_reserves(gpu, pooled) {
        return 0;
    }
    qsa_bf16_split_capacity_bytes()
}

/// The constant extent [`reserve_qsa_bf16_split_scratch`] reserves when the
/// split route can run (score + filter candidates + control rows). No `Gpu`,
/// arch or env gate: those decide only whether it is reserved at all.
pub fn qsa_bf16_split_capacity_bytes() -> usize {
    // The layout inputs are compile-time constants; the unit test pins the result.
    qsa_bf16_split_reserve_layout()
        .expect("QSA BF16 split reserve layout overflows its compile-time constants")
        .total
}

/// Reserves the split route's score / filter scratch at its constant upper
/// bound before any capture or record, so [`Gpu::qsa_select_scores`] never
/// grows (and the pointer never moves) under a recorder. A no-op, returning 0,
/// unless the route can run: switch on, exact gfx1201, BF16 pooled keys.
/// Returns the reserved bytes.
pub fn reserve_qsa_bf16_split_scratch(gpu: &mut Gpu, pooled: DType) -> HipResult<usize> {
    if !qsa_bf16_split_reserves(gpu, pooled) {
        return Ok(0);
    }
    let layout = qsa_bf16_split_reserve_layout()?;
    gpu.qsa_select_scores(layout.total)?;
    Ok(layout.total)
}

/// Declarations of a split-route launch's two position-derived slots: the
/// active block count `(position + call_rows) / compress` and this group's
/// `position_start` (`position + first_row`), exactly the fused selector's.
fn qsa_position_bindings(
    call_rows: usize,
    first_row: usize,
    compress: usize,
    block_count_offset: usize,
    position_offset: usize,
) -> HipResult<[crate::replay::ReplayKernargBinding; 2]> {
    let word = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| HipError::new(0, &format!("QSA split {what} exceeds u32")))
    };
    Ok([
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: block_count_offset,
            addend: word(call_rows, "rows")?,
            divisor: word(compress, "compression")?,
        },
        crate::replay::ReplayKernargBinding::PositionPlusU32 {
            offset: position_offset,
            addend: word(first_row, "row offset")?,
        },
    ])
}

/// BF16-pooled selection as a score / select pair, byte-identical to the fused
/// `indexed_attention_select_bf16_batched`. Scores come from the hipcc
/// `indexed_attention_select_scores_rows{8,16}_bf16` (bit-identical to the
/// fused per-block score), the selection from `indexed_attention_select_from_scores`
/// or, with `filter`, the exact filter trio. Scores and filter scratch live in
/// [`Gpu::qsa_select_scores`], never the shared F16 activation slot.
///
/// `bounded` is the retained-replay shape: the score stride, the score grid
/// (`ceil(shape_blocks / 256)`) and the filter compact grid
/// (`ceil(shape_blocks / 4096)`) come from `p.shape_blocks` instead of the live
/// count (the kernels mask by the kernarg block count), the block count and
/// `position_start` of every launch are declared to the recorder, no decision
/// reads a device result, and the scratch is the constant
/// [`reserve_qsa_bf16_split_scratch`] size. `stats` (labs, never recording)
/// downloads the filter control words per group.
fn indexed_attention_select_bf16_split(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    filter: Option<QsaFilterCfg>,
    bounded: bool,
    mut stats: Option<&mut Vec<[u64; 2]>>,
) -> HipResult<bool> {
    if p.pooled.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    if let Some(knobs) = filter {
        qsa_filter_cfg_check(knobs)?;
    }
    let bound = if bounded {
        p.shape_blocks
    } else {
        p.block_count
    };
    if bound == 0 {
        return indexed_attention_select_fused(gpu, p, mirror);
    }
    let group = qsa_filter_score_group(bound)
        .ok_or_else(|| {
            HipError::new(
                0,
                "QSA BF16 split: one score row exceeds the score scratch budget",
            )
        })?
        .min(QSA_BF16_SPLIT_MAX_GROUP_ROWS)
        .min(p.rows);
    let cap = filter.map_or(0, |knobs| knobs.cap);
    let score_bytes = group
        .checked_mul(bound)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| HipError::new(0, "QSA BF16 split score scratch overflow"))?;
    let layout = qsa_bf16_split_layout(score_bytes, group, cap)?;
    let base = gpu.qsa_select_scores(layout.total)? as *mut u8;
    // SAFETY: both offsets lie inside the `layout.total` bytes just requested.
    let (scores, cand, ctl) = unsafe {
        (
            base as *mut std::ffi::c_void,
            base.add(layout.cand_off) as *mut std::ffi::c_void,
            base.add(layout.ctl_off) as *mut std::ffi::c_void,
        )
    };
    let score_sym = qsa_select_score_symbol(
        QsaScoreArm::HipBf16,
        gpu.route_rows(p.rows) > 8,
        qsa_score_rows32(gpu, p.rows, p.index_heads, p.index_dim),
    );
    let (score_kernel, score_rows) = (score_sym.kernel, score_sym.rows);
    let select_kernel = "indexed_attention_select_from_scores";
    gpu.ensure_kernel_public(
        "qsa_bf16_scores",
        crate::kernels::QSA_BF16_SCORES_SRC,
        score_kernel,
    )?;
    if filter.is_some() {
        qsa_filter_ensure_kernels(gpu)?;
    } else {
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, select_kernel)?;
    }
    let mirror =
        mirror.filter(|m| m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>());
    let block_count = checked_i32(p.block_count, "QSA batch select blocks")?;
    let stride = checked_i32(bound, "QSA BF16 split score stride")?;
    let index_dim = checked_i32(p.index_dim, "QSA batch select dim")?;
    let compress = checked_i32(p.compress, "QSA batch select compress")?;
    let query_row_stride = checked_i32(p.query_row_stride, "QSA batch select query stride")?;
    let budget = checked_i32(p.budget_blocks, "QSA batch select budget")?;
    let capacity = checked_i32(p.capacity, "QSA batch select capacity")?;
    let score_tiles = checked_u32(bound.div_ceil(256), "QSA select block tiles")?;
    let mut g0 = 0usize;
    while g0 < p.rows {
        let n = group.min(p.rows - g0);
        // SAFETY: `qsa_select_batch_validate` bounded `(rows - 1) * stride`
        // plus a row and `rows * capacity` by the tensors' extents.
        let query = unsafe {
            (p.query.buf.as_ptr() as *mut f32).add(g0 * p.query_row_stride) as *mut std::ffi::c_void
        };
        let selected = unsafe {
            (p.selected.buf.as_ptr() as *mut i32).add(g0 * p.capacity) as *mut std::ffi::c_void
        };
        let position_start = checked_i32(p.position_start + g0, "QSA batch select position")?;
        let rows = checked_i32(n, "QSA batch select rows")?;
        let mut args = KernargBlob::new();
        args.push_ptr(query);
        args.push_ptr(p.pooled.buf.as_ptr());
        args.push_ptr(scores);
        args.push_i32(rows);
        args.push_i32(query_row_stride);
        args.push_i32(block_count);
        let block_count_offset = args.len() - 4;
        args.push_i32(index_dim);
        args.push_i32(compress);
        args.push_i32(position_start);
        let position_offset = args.len() - 4;
        args.push_i32(stride);
        args.pad_to(16);
        let bindings =
            qsa_position_bindings(p.rows, g0, p.compress, block_count_offset, position_offset)?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_score(
                gpu,
                p,
                g0,
                n,
                p.pooled,
                score_sym.pooled_live_guarded,
                scores,
                "qsa_select_scores",
                bound,
                score_kernel,
                false,
            );
        }
        gpu.launch_blob_recorded(
            score_kernel,
            [
                score_tiles,
                checked_u32(n.div_ceil(score_rows), "QSA select row groups")?,
                1,
            ],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings {
                grid: None,
                kernargs: &bindings,
            },
        )?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_score(
                gpu,
                p,
                g0,
                n,
                p.pooled,
                score_sym.pooled_live_guarded,
                scores,
                "qsa_select_scores",
                bound,
                score_kernel,
                true,
            );
        }
        // The persistent selection is the final row's: the last group's.
        let mirror_ptr = mirror
            .filter(|_| g0 + n == p.rows)
            .map_or(std::ptr::null_mut(), |m| m.buf.as_ptr());
        if let Some(knobs) = filter {
            let extent = gpu.flags.qwen4_qsa_extent_check.then(|| QsaFilterExtent {
                selected: p.selected,
                first_row: g0,
                mirror,
                scores: None,
            });
            qsa_filter_launch_shaped(
                gpu,
                &QsaFilterTrio {
                    scores,
                    stride: bound,
                    ctl,
                    cand,
                    selected,
                    mirror: mirror_ptr,
                    rows: n,
                    block_count: p.block_count,
                    budget: p.budget_blocks,
                    compress: p.compress,
                    position_start: p.position_start + g0,
                    capacity: p.capacity,
                    knobs,
                    extent,
                },
                Some(QsaFilterReplay {
                    bound,
                    call_rows: p.rows,
                    first_row: g0,
                }),
            )?;
            if let Some(out) = stats.as_mut() {
                out.extend(qsa_filter_read_ctl(gpu, ctl, n)?);
            }
            g0 += n;
            continue;
        }
        let mut args = KernargBlob::new();
        args.push_ptr(scores);
        args.push_i32(stride);
        args.push_ptr(selected);
        args.push_i32(rows);
        args.push_i32(block_count);
        let block_count_offset = args.len() - 4;
        args.push_i32(budget);
        args.push_i32(compress);
        args.push_i32(position_start);
        let position_offset = args.len() - 4;
        args.push_i32(capacity);
        args.push_ptr(mirror_ptr);
        args.pad_to(16);
        let bindings =
            qsa_position_bindings(p.rows, g0, p.compress, block_count_offset, position_offset)?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_from_scores(
                gpu,
                p,
                g0,
                n,
                scores,
                "qsa_select_scores",
                bound,
                mirror.filter(|_| g0 + n == p.rows),
                select_kernel,
                false,
            );
        }
        gpu.launch_blob_recorded(
            select_kernel,
            [rows as u32, 1, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings {
                grid: None,
                kernargs: &bindings,
            },
        )?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_from_scores(
                gpu,
                p,
                g0,
                n,
                scores,
                "qsa_select_scores",
                bound,
                None,
                select_kernel,
                true,
            );
        }
        g0 += n;
    }
    Ok(mirror.is_some())
}

/// Exact filter select (`qsa_filter_select.hip`): sampled threshold, tiled
/// candidate compaction, exact finish with in-kernel repair. Production knobs.
const QSA_FILTER_SAMPLES: usize = 2048;
const QSA_FILTER_TARGET: usize = 1024;
const QSA_FILTER_CAP: usize = 4096;
/// The kernels stage samples and candidates in LDS: neither may exceed 4096.
const QSA_FILTER_MAX_SAMPLES: usize = 4096;
const QSA_FILTER_MAX_CAP: usize = 4096;
/// Blocks above which the filter replaces select-from-scores. It beats the
/// hipcc select from 8192 blocks; against the PM select (gfx1151) only past
/// the measured crossovers (Halo, per call: 1-16 rows PM 18.6 / 50.7 us vs
/// filter 37 / 39 us at 8192 / 32768 blocks; 256 rows PM 0.26 / 1.15 ms vs
/// filter 0.35 / 0.80 ms at 32768 / 65536 blocks).
const QSA_FILTER_MIN_BLOCKS: usize = 4096;
const QSA_FILTER_MIN_BLOCKS_PM_FEW_ROWS: usize = 24576;
const QSA_FILTER_MIN_BLOCKS_PM: usize = 49152;
/// Calls of at most this many rows (decode, verify) use the few-row crossover.
const QSA_FILTER_FEW_ROWS: usize = 16;
/// `HIPFIRE_QWEN4_QSA_FILTER_FEW_ROWS32` (developer, default off, `1` on):
/// candidate crossover admission. Exact-wide verify calls of 17..=32 rows
/// ([`qsa_exact_wide_rows`]) use the few-row PM crossover
/// ([`QSA_FILTER_MIN_BLOCKS_PM_FEW_ROWS`]) instead of the many-row one. Exact
/// either way (every row selects over its own block count); a performance
/// choice to keep unmeasured until the PM-select versus filter crossover at
/// 16/32 rows is compared. Read once.
static QWEN4_QSA_FILTER_FEW_ROWS32: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    hipfire_config::developer_bool("HIPFIRE_QWEN4_QSA_FILTER_FEW_ROWS32", false)
});
/// `HIPFIRE_QWEN4_QSA_FILTER_SELECT` routes eager gfx11+ calls of more than
/// [`qsa_filter_min_blocks`] blocks and at most
/// [`QSA_SELECT_FROM_SCORES_MAX_BUDGET`] budget blocks through the filter
/// trio instead of select-from-scores. Unset: on for exact gfx1151 (where it
/// was measured), off elsewhere; `1` forces it on for any gfx11+, `0` is the
/// kill switch. Read once.
static QWEN4_QSA_FILTER_SELECT: std::sync::LazyLock<Option<bool>> =
    std::sync::LazyLock::new(|| {
        match hipfire_config::process_value("HIPFIRE_QWEN4_QSA_FILTER_SELECT").as_deref() {
            Some("1") => Some(true),
            Some("0") => Some(false),
            _ => None,
        }
    });

#[derive(Clone, Copy, Debug)]
struct QsaFilterCfg {
    samples: usize,
    target: usize,
    cap: usize,
}

const QSA_FILTER_PRODUCTION: QsaFilterCfg = QsaFilterCfg {
    samples: QSA_FILTER_SAMPLES,
    target: QSA_FILTER_TARGET,
    cap: QSA_FILTER_CAP,
};

fn qsa_filter_cfg_check(k: QsaFilterCfg) -> HipResult<()> {
    if k.samples == 0
        || k.samples > QSA_FILTER_MAX_SAMPLES
        || k.target == 0
        || k.cap == 0
        || k.cap > QSA_FILTER_MAX_CAP
    {
        return Err(HipError::new(
            0,
            &format!(
                "QSA filter knobs samples {} (1..={QSA_FILTER_MAX_SAMPLES}), target {} (>= 1), cap {} (1..={QSA_FILTER_MAX_CAP}) out of range",
                k.samples, k.target, k.cap
            ),
        ));
    }
    checked_i32(k.target, "QSA filter target")?;
    Ok(())
}

fn qsa_filter_min_blocks(few_rows: bool, pm_select: bool) -> usize {
    match (pm_select, few_rows) {
        (false, _) => QSA_FILTER_MIN_BLOCKS,
        (true, true) => QSA_FILTER_MIN_BLOCKS_PM_FEW_ROWS,
        (true, false) => QSA_FILTER_MIN_BLOCKS_PM,
    }
}

/// Whether `rows` use the few-row crossover: at most [`QSA_FILTER_FEW_ROWS`]
/// rows, or (candidate, default off) an exact-wide call under
/// [`QWEN4_QSA_FILTER_FEW_ROWS32`].
fn qsa_filter_few_rows(gpu: &Gpu, rows: usize) -> bool {
    rows <= QSA_FILTER_FEW_ROWS || (*QWEN4_QSA_FILTER_FEW_ROWS32 && qsa_exact_wide_rows(gpu, rows))
}

/// Production admission of the filter route for a call the score/select
/// route already admitted: switch on (default: exact gfx1151), gfx11+ SIMT, eager, more than
/// [`qsa_filter_min_blocks`] blocks for the select it replaces, budget within
/// the select bound.
fn qsa_filter_production(
    gpu: &Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    pm: QsaSelectPmArm,
) -> Option<QsaFilterCfg> {
    (QWEN4_QSA_FILTER_SELECT.unwrap_or(gpu.arch_caps.is_gfx1151())
        && gpu.arch_caps.has_gfx11_plus_simt()
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.block_count > qsa_filter_min_blocks(qsa_filter_few_rows(gpu, p.rows), pm.select())
        && p.budget_blocks <= QSA_SELECT_FROM_SCORES_MAX_BUDGET
        && qsa_filter_score_group(p.block_count).is_some())
    .then_some(QSA_FILTER_PRODUCTION)
}

/// Rows per group the candidate (`cap` u64 each) and control (16 B) scratch
/// of a filter group keeps within [`QSA_SELECT_SCORE_SCRATCH_BYTES`].
fn qsa_filter_group_limit(cap: usize) -> usize {
    (QSA_SELECT_SCORE_SCRATCH_BYTES / (cap * 8 + 16) / 16 * 16).max(16)
}

/// Rows per filter group whose F32 scores fit [`QSA_SELECT_SCORE_SCRATCH_BYTES`]:
/// a multiple of 16 when at least 16 rows fit, else 1; `None` when a single
/// row's scores already exceed the budget (the filter route is declined).
fn qsa_filter_score_group(stride: usize) -> Option<usize> {
    let fit = QSA_SELECT_SCORE_SCRATCH_BYTES / stride.max(1).checked_mul(4)?;
    match fit {
        0 => None,
        1..=15 => Some(1),
        _ => Some(fit / 16 * 16),
    }
}

fn qsa_filter_overflow() -> HipError {
    HipError::new(0, "QSA filter scratch overflow")
}

fn qsa_filter_align(bytes: usize) -> HipResult<usize> {
    bytes
        .checked_add(255)
        .map(|n| n & !255)
        .ok_or_else(qsa_filter_overflow)
}

struct QsaFilterScratch {
    scores: *mut std::ffi::c_void,
    cand: *mut std::ffi::c_void,
    ctl: *mut std::ffi::c_void,
}

/// One shared-scratch request holding `score_bytes` of scores, `group * cap`
/// u64 candidates and `group` 16-byte control rows at 256-byte aligned
/// offsets (the accessor counts F16 elements: bytes / 2).
fn qsa_filter_scratch(
    gpu: &mut Gpu,
    score_bytes: usize,
    group: usize,
    cap: usize,
) -> HipResult<QsaFilterScratch> {
    let cand_off = qsa_filter_align(score_bytes)?;
    let cand_bytes = group
        .checked_mul(cap)
        .and_then(|n| n.checked_mul(std::mem::size_of::<u64>()))
        .ok_or_else(qsa_filter_overflow)?;
    let ctl_off = cand_off
        .checked_add(qsa_filter_align(cand_bytes)?)
        .ok_or_else(qsa_filter_overflow)?;
    let ctl_bytes = group.checked_mul(16).ok_or_else(qsa_filter_overflow)?;
    let total = qsa_filter_align(
        ctl_off
            .checked_add(ctl_bytes)
            .ok_or_else(qsa_filter_overflow)?,
    )?;
    checked_extent(total, "QSA filter scratch")?;
    let base = gpu.qwen4_f16_x_scratch(total / 2)?.buf.as_ptr() as *mut u8;
    // SAFETY: every offset is within the `total`-byte request just made.
    unsafe {
        Ok(QsaFilterScratch {
            scores: base as *mut std::ffi::c_void,
            cand: base.add(cand_off) as *mut std::ffi::c_void,
            ctl: base.add(ctl_off) as *mut std::ffi::c_void,
        })
    }
}

fn qsa_filter_ensure_kernels(gpu: &mut Gpu) -> HipResult<()> {
    for name in [
        "qsa_filter_threshold",
        "qsa_filter_compact",
        "qsa_filter_finish",
    ] {
        gpu.ensure_kernel_public(
            "qsa_filter_select",
            crate::kernels::QSA_FILTER_SELECT_SRC,
            name,
        )?;
    }
    Ok(())
}

/// One row group of the filter trio. `mirror` is null except for the last
/// group of a call.
struct QsaFilterTrio<'a> {
    scores: *mut std::ffi::c_void,
    ctl: *mut std::ffi::c_void,
    cand: *mut std::ffi::c_void,
    selected: *mut std::ffi::c_void,
    mirror: *mut std::ffi::c_void,
    stride: usize,
    rows: usize,
    block_count: usize,
    budget: usize,
    compress: usize,
    position_start: usize,
    capacity: usize,
    knobs: QsaFilterCfg,
    /// Check-only tensor owners; `Some` only when the extent check is on (built
    /// under that flag, so the off path stores a `None` and nothing else).
    extent: Option<QsaFilterExtent<'a>>,
}

/// Tensor owners of a trio's tensor-derived operands, for the developer extent
/// check only: `selected` starts `first_row * capacity` words into `selected`,
/// the trio's `mirror` pointer is `mirror` when non-null, and `scores` is
/// `scores` at `first_row * stride` words (the from-scores labs); `None` means
/// the trio's `scores` pointer is scratch.
struct QsaFilterExtent<'a> {
    selected: &'a GpuTensor,
    first_row: usize,
    mirror: Option<&'a GpuTensor>,
    scores: Option<&'a GpuTensor>,
}

/// Replay declarations of a shape-bounded filter launch
/// ([`indexed_attention_select_bf16_split`] under a recorder).
#[derive(Clone, Copy)]
struct QsaFilterReplay {
    /// Position-independent block bound: the compact grid is
    /// `ceil(bound / 4096)` and is launched even for an empty live prefix.
    bound: usize,
    /// Rows of the whole call: the declared count is `(position + rows) / compress`.
    call_rows: usize,
    /// First row of this group inside the call.
    first_row: usize,
}

fn qsa_filter_launch(gpu: &mut Gpu, t: &QsaFilterTrio<'_>) -> HipResult<()> {
    qsa_filter_launch_shaped(gpu, t, None)
}

/// Pushes `values` as i32 kernargs; returns the kernarg byte offsets of the
/// values at `block_count_index` and `position_index`.
fn qsa_filter_push_ints(
    args: &mut KernargBlob,
    values: &[i32],
    block_count_index: usize,
    position_index: usize,
) -> (usize, usize) {
    let base = args.len();
    for value in values {
        args.push_i32(*value);
    }
    (base + 4 * block_count_index, base + 4 * position_index)
}

/// The filter trio. With `replay` the grids come from its bound and the
/// block-count / `position_start` slots of all three launches are declared to
/// the recorder; without it the launch is the eager one (grids from the live
/// count, nothing declared).
fn qsa_filter_launch_shaped(
    gpu: &mut Gpu,
    t: &QsaFilterTrio<'_>,
    replay: Option<QsaFilterReplay>,
) -> HipResult<()> {
    let rows = checked_i32(t.rows, "QSA filter rows")?;
    let row_grid = checked_u32(t.rows, "QSA filter row grid")?;
    let stride = checked_i32(t.stride, "QSA filter score stride")?;
    let block_count = checked_i32(t.block_count, "QSA filter blocks")?;
    let budget = checked_i32(t.budget, "QSA filter budget")?;
    let compress = checked_i32(t.compress, "QSA filter compress")?;
    let position_start = checked_i32(t.position_start, "QSA filter position")?;
    let capacity = checked_i32(t.capacity, "QSA filter capacity")?;
    let samples = checked_i32(t.knobs.samples, "QSA filter samples")?;
    let target = checked_i32(t.knobs.target, "QSA filter target")?;
    let cap = checked_i32(t.knobs.cap, "QSA filter cap")?;
    let bound = replay.map_or(t.block_count, |r| r.bound);
    let tiles = checked_u32(bound.div_ceil(4096).max(1), "QSA filter tiles")?;
    let bindings = |block_count_offset: usize, position_offset: usize| match replay {
        Some(r) => qsa_position_bindings(
            r.call_rows,
            r.first_row,
            t.compress,
            block_count_offset,
            position_offset,
        )
        .map(Some),
        None => Ok(None),
    };
    let mut args = KernargBlob::new();
    args.push_ptr(t.scores);
    args.push_i32(stride);
    args.push_ptr(t.ctl);
    let (bc, ps) = qsa_filter_push_ints(
        &mut args,
        &[
            rows,
            block_count,
            compress,
            position_start,
            samples,
            target,
            cap,
        ],
        1,
        3,
    );
    args.pad_to(16);
    let declared = bindings(bc, ps)?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_filter(gpu, t, "qsa_filter_threshold", false);
    }
    gpu.launch_blob_recorded(
        "qsa_filter_threshold",
        [row_grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: declared.as_ref().map_or(&[][..], |b| &b[..]),
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_filter(gpu, t, "qsa_filter_threshold", true);
    }
    // Zero live blocks: no tile to compact (a zero grid is not launchable);
    // the finish still writes the tail / sentinel slots and the stats. A
    // shape-bounded grid is never empty: tiles past the live prefix return.
    if t.block_count > 0 || replay.is_some() {
        let mut args = KernargBlob::new();
        args.push_ptr(t.scores);
        args.push_i32(stride);
        args.push_ptr(t.ctl);
        args.push_ptr(t.cand);
        let (bc, ps) = qsa_filter_push_ints(
            &mut args,
            &[rows, block_count, compress, position_start, cap],
            1,
            3,
        );
        args.pad_to(16);
        let declared = bindings(bc, ps)?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_filter(gpu, t, "qsa_filter_compact", false);
        }
        gpu.launch_blob_recorded(
            "qsa_filter_compact",
            [tiles, row_grid, 1],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings {
                grid: None,
                kernargs: declared.as_ref().map_or(&[][..], |b| &b[..]),
            },
        )?;
        if gpu.flags.qwen4_qsa_extent_check {
            qsa_sel_ext_filter(gpu, t, "qsa_filter_compact", true);
        }
    }
    let mut args = KernargBlob::new();
    args.push_ptr(t.scores);
    args.push_i32(stride);
    args.push_ptr(t.ctl);
    args.push_ptr(t.cand);
    args.push_ptr(t.selected);
    args.push_ptr(t.mirror);
    let (bc, ps) = qsa_filter_push_ints(
        &mut args,
        &[
            rows,
            block_count,
            budget,
            compress,
            position_start,
            capacity,
            cap,
        ],
        1,
        4,
    );
    args.pad_to(16);
    let declared = bindings(bc, ps)?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_filter(gpu, t, "qsa_filter_finish", false);
    }
    gpu.launch_blob_recorded(
        "qsa_filter_finish",
        [row_grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: declared.as_ref().map_or(&[][..], |b| &b[..]),
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_filter(gpu, t, "qsa_filter_finish", true);
    }
    Ok(())
}

/// Downloads `rows` control rows `[threshold, count | repaired << 32]`
/// (a host sync: labs only).
fn qsa_filter_read_ctl(
    gpu: &Gpu,
    ctl: *mut std::ffi::c_void,
    rows: usize,
) -> HipResult<Vec<[u64; 2]>> {
    let bytes = rows.checked_mul(16).ok_or_else(qsa_filter_overflow)?;
    let view = GpuTensor {
        // SAFETY: non-owning view of the `rows` control rows of the live scratch.
        buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ctl, bytes) },
        shape: vec![bytes],
        dtype: DType::Raw,
    };
    let raw = gpu.download_raw_bytes(&view)?;
    Ok(raw
        .chunks_exact(16)
        .map(|c| {
            [
                u64::from_le_bytes(c[..8].try_into().unwrap()),
                u64::from_le_bytes(c[8..].try_into().unwrap()),
            ]
        })
        .collect())
}

/// Filter knobs of the lab entries; `Default` is the production constants.
#[cfg(any(test, feature = "lab"))]
#[derive(Clone, Copy, Debug)]
pub struct QsaFilterKnobs {
    pub samples: usize,
    pub target: usize,
    pub cap: usize,
}

#[cfg(any(test, feature = "lab"))]
impl Default for QsaFilterKnobs {
    fn default() -> Self {
        Self {
            samples: QSA_FILTER_SAMPLES,
            target: QSA_FILTER_TARGET,
            cap: QSA_FILTER_CAP,
        }
    }
}

#[cfg(any(test, feature = "lab"))]
impl QsaFilterKnobs {
    fn cfg(self) -> QsaFilterCfg {
        QsaFilterCfg {
            samples: self.samples,
            target: self.target,
            cap: self.cap,
        }
    }
}

/// Per-row outcome of the filter: the sampled threshold key, the number of
/// keys at or above it, and whether the exact repair ran.
#[cfg(any(test, feature = "lab"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QsaFilterRowStat {
    pub threshold: u64,
    pub count: u32,
    pub repaired: bool,
}

#[cfg(any(test, feature = "lab"))]
fn qsa_filter_stats(raw: Vec<[u64; 2]>) -> Vec<QsaFilterRowStat> {
    raw.into_iter()
        .map(|[threshold, flags]| QsaFilterRowStat {
            threshold,
            count: flags as u32,
            repaired: flags >> 32 != 0,
        })
        .collect()
}

/// Frozen F32 score rows (`scores[row * score_stride + block]`) for the
/// from-scores labs; the live-block bound of row `r` is
/// `(position_start + r + 1) / compress` clamped by `block_count`.
#[cfg(any(test, feature = "lab"))]
#[derive(Clone, Copy)]
pub struct QsaFromScores<'a> {
    pub scores: &'a GpuTensor,
    pub score_stride: usize,
    pub selected: &'a GpuTensor,
    pub rows: usize,
    pub block_count: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub position_start: usize,
    pub capacity: usize,
}

#[cfg(any(test, feature = "lab"))]
fn qsa_from_scores_check(gpu: &Gpu, p: &QsaFromScores<'_>) -> HipResult<()> {
    let wrong = || HipError::new(0, &ComputeError::WrongShape.to_string());
    if !gpu.arch_caps.has_gfx11_plus_simt() || gpu.replay.is_recording() || gpu.graphs.capture_mode
    {
        return Err(HipError::new(0, "QSA select from scores: call not covered"));
    }
    ensure_f32(p.scores)?;
    if p.selected.dtype != DType::Raw
        || p.rows == 0
        || p.compress == 0
        || p.capacity == 0
        || p.budget_blocks > QSA_SELECT_FROM_SCORES_MAX_BUDGET
        || p.score_stride < p.block_count
    {
        return Err(wrong());
    }
    let score_need = if p.block_count == 0 {
        0
    } else {
        checked_product(p.rows - 1, p.score_stride, "QSA from-scores rows")?
            .checked_add(p.block_count)
            .ok_or_else(wrong)?
    };
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA from-scores selected")?,
        std::mem::size_of::<i32>(),
        "QSA from-scores selected",
    )?;
    let end_position = p.position_start.checked_add(p.rows).ok_or_else(wrong)?;
    if end_position > i32::MAX as usize
        || p.scores.numel() < score_need
        || p.selected.numel() < selected_bytes
    {
        return Err(wrong());
    }
    checked_i32(p.rows, "QSA from-scores rows")?;
    checked_u32(p.rows, "QSA from-scores row grid")?;
    checked_i32(p.score_stride, "QSA from-scores stride")?;
    checked_i32(p.block_count, "QSA from-scores blocks")?;
    checked_i32(p.budget_blocks, "QSA from-scores budget")?;
    checked_i32(p.compress, "QSA from-scores compress")?;
    checked_i32(p.capacity, "QSA from-scores capacity")?;
    Ok(())
}

/// Lab: the filter trio over frozen score rows, grouped by the candidate
/// scratch; returns the per-row stats in row order. `mirror` receives the
/// final row's selection, as on the live route.
#[cfg(any(test, feature = "lab"))]
pub fn qsa_filter_select_from_scores_lab(
    gpu: &mut Gpu,
    p: &QsaFromScores<'_>,
    mirror: Option<&GpuTensor>,
    knobs: QsaFilterKnobs,
) -> HipResult<Vec<QsaFilterRowStat>> {
    qsa_from_scores_check(gpu, p)?;
    let cfg = knobs.cfg();
    qsa_filter_cfg_check(cfg)?;
    qsa_filter_ensure_kernels(gpu)?;
    let mirror =
        mirror.filter(|m| m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>());
    let group = qsa_filter_group_limit(cfg.cap).min(p.rows);
    let scratch = qsa_filter_scratch(gpu, 0, group, cfg.cap)?;
    let mut raw = Vec::with_capacity(p.rows);
    let mut g0 = 0usize;
    while g0 < p.rows {
        let n = group.min(p.rows - g0);
        // SAFETY: `qsa_from_scores_check` bounded `(rows - 1) * stride + blocks`
        // and `rows * capacity` by the tensors' extents.
        let scores = unsafe {
            (p.scores.buf.as_ptr() as *mut f32).add(g0 * p.score_stride) as *mut std::ffi::c_void
        };
        let selected = unsafe {
            (p.selected.buf.as_ptr() as *mut i32).add(g0 * p.capacity) as *mut std::ffi::c_void
        };
        let last = g0 + n == p.rows;
        let extent = gpu.flags.qwen4_qsa_extent_check.then(|| QsaFilterExtent {
            selected: p.selected,
            first_row: g0,
            mirror,
            scores: Some(p.scores),
        });
        qsa_filter_launch(
            gpu,
            &QsaFilterTrio {
                scores,
                ctl: scratch.ctl,
                cand: scratch.cand,
                selected,
                mirror: mirror
                    .filter(|_| last)
                    .map_or(std::ptr::null_mut(), |m| m.buf.as_ptr()),
                stride: p.score_stride,
                rows: n,
                block_count: p.block_count,
                budget: p.budget_blocks,
                compress: p.compress,
                position_start: p.position_start + g0,
                capacity: p.capacity,
                knobs: cfg,
                extent,
            },
        )?;
        raw.extend(qsa_filter_read_ctl(gpu, scratch.ctl, n)?);
        g0 += n;
    }
    Ok(qsa_filter_stats(raw))
}

/// Lab: the incumbent exact select-from-scores kernel over frozen score
/// rows, the PM module (`pm`, gfx1151 only) or the hipcc kernel, same input
/// and output contract as [`qsa_filter_select_from_scores_lab`].
#[cfg(any(test, feature = "lab"))]
pub fn qsa_incumbent_select_from_scores_lab(
    gpu: &mut Gpu,
    p: &QsaFromScores<'_>,
    mirror: Option<&GpuTensor>,
    pm: bool,
) -> HipResult<()> {
    qsa_from_scores_check(gpu, p)?;
    if pm && (!gpu.arch_caps.is_gfx1151() || p.block_count > QSA_SELECT_PM_MAX_BLOCKS) {
        return Err(HipError::new(
            0,
            "QSA incumbent PM select: call not covered",
        ));
    }
    let kernel = if pm {
        gpu.ensure_embedded_kernel(
            "qsa_select_pm_gfx1151",
            crate::kernels::QSA_SELECT_PM_GFX1151,
            "indexed_attention_select_from_scores_pm_gfx1151",
        )?;
        "indexed_attention_select_from_scores_pm_gfx1151"
    } else {
        gpu.ensure_kernel_public(
            "tensor_ops",
            TENSOR_OPS_SRC,
            "indexed_attention_select_from_scores",
        )?;
        "indexed_attention_select_from_scores"
    };
    let mirror =
        mirror.filter(|m| m.numel() * m.dtype.size() >= p.capacity * std::mem::size_of::<i32>());
    let mut args = KernargBlob::new();
    args.push_ptr(p.scores.buf.as_ptr());
    args.push_i32(checked_i32(p.score_stride, "QSA from-scores stride")?);
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_i32(checked_i32(p.rows, "QSA from-scores rows")?);
    args.push_i32(checked_i32(p.block_count, "QSA from-scores blocks")?);
    args.push_i32(checked_i32(p.budget_blocks, "QSA from-scores budget")?);
    args.push_i32(checked_i32(p.compress, "QSA from-scores compress")?);
    args.push_i32(checked_i32(p.position_start, "QSA from-scores position")?);
    args.push_i32(checked_i32(p.capacity, "QSA from-scores capacity")?);
    args.push_ptr(mirror.map_or(std::ptr::null_mut(), |m| m.buf.as_ptr()));
    args.pad_to(16);
    if gpu.flags.qwen4_qsa_extent_check {
        // Scores are read `row * stride + [0, rbc)`; selected `capacity` words per
        // row; mirror (last row) `capacity` words; grid `rows`, no rounding.
        let live = qsa_sel_ext_rbc_last(p.rows, p.position_start, p.compress, p.block_count);
        let depth = qsa_sel_ext_depth(p.rows, p.position_start, p.block_count);
        gpu.qsa_extent_check(
            "qsa_filter_scores",
            p.scores,
            0,
            qsa_sel_ext_span(p.rows, p.score_stride, live).saturating_mul(QSA_SEL_EXT_WORD),
            depth,
            kernel,
        );
        qsa_sel_ext_selected(
            gpu, p.selected, 0, p.rows, p.capacity, p.capacity, depth, kernel,
        );
        if let Some(m) = mirror {
            qsa_sel_ext_mirror(gpu, m, p.capacity, depth, kernel);
        }
    }
    gpu.launch_blob_recorded(
        kernel,
        [checked_u32(p.rows, "QSA from-scores row grid")?, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        qsa_sel_ext_post_selected(
            gpu,
            p.selected,
            0,
            p.rows,
            p.capacity,
            p.position_start,
            p.block_count,
            kernel,
        );
    }
    Ok(())
}

/// Lab entry forcing the filter route (env and block minimum bypassed) with
/// the production grouping and the PM score when `score_pm` and the PM pair
/// covers the call, else the hipcc score; BF16 arenas need PM coverage.
/// Returns whether the mirror was persisted and the per-row stats.
#[cfg(any(test, feature = "lab"))]
pub fn indexed_attention_select_batch_filter_arm(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    score_pm: bool,
    knobs: QsaFilterKnobs,
) -> HipResult<(bool, Vec<QsaFilterRowStat>)> {
    qsa_select_batch_validate(p)?;
    let live = gpu.arch_caps.has_gfx11_plus_simt()
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.index_heads == 4
        && p.index_dim.is_multiple_of(4)
        && p.index_dim <= 128
        && p.budget_blocks <= QSA_SELECT_FROM_SCORES_MAX_BUDGET;
    if !live {
        return Err(HipError::new(0, "QSA select filter arm: call not covered"));
    }
    let fits = qsa_select_pm_shape_fits(
        gpu.arch_caps.is_gfx1151(),
        p.pooled.dtype,
        gpu.route_rows(p.rows).max(p.rows),
        p.index_heads,
        p.index_dim,
        p.block_count,
        p.query_row_stride,
        p.capacity,
    );
    let pm = if score_pm && fits {
        QsaSelectPmArm::ScoreOnly
    } else {
        QsaSelectPmArm::Hipcc
    };
    if p.pooled.dtype == DType::BF16 && !pm.score() {
        // No PM score: the hipcc BF16 score pair (never the fused selector).
        let mut raw = Vec::with_capacity(p.rows);
        let persisted = indexed_attention_select_bf16_split(
            gpu,
            p,
            mirror,
            Some(knobs.cfg()),
            false,
            Some(&mut raw),
        )?;
        return Ok((persisted, qsa_filter_stats(raw)));
    }
    let mut raw = Vec::with_capacity(p.rows);
    let persisted =
        indexed_attention_select_rows8_core(gpu, p, mirror, pm, Some(knobs.cfg()), Some(&mut raw))?;
    Ok((persisted, qsa_filter_stats(raw)))
}

/// Which selection the BF16 split lab entry runs after the hipcc BF16 score.
#[cfg(any(test, feature = "lab"))]
#[derive(Clone, Copy, Debug)]
pub enum QsaBf16SplitArm {
    /// `indexed_attention_select_from_scores`.
    Select,
    /// The exact filter trio with these knobs.
    Filter(QsaFilterKnobs),
}

/// Lab entry forcing the BF16 split route ([`indexed_attention_select_bf16_split`])
/// on any gfx11+ device: env, arch and crossover bypassed. `bounded` runs the
/// retained-replay shape (grids, stride and scratch from `p.shape_blocks`,
/// positions declared) outside a recorder; `collect_stats` downloads the
/// filter's per-row stats (a host sync: never under a recorder). Returns whether
/// the mirror was persisted and the stats (empty without a filter or
/// `collect_stats`).
#[cfg(any(test, feature = "lab"))]
pub fn indexed_attention_select_batch_bf16_split_arm(
    gpu: &mut Gpu,
    p: &IndexedAttentionSelectBatch<'_>,
    mirror: Option<&GpuTensor>,
    arm: QsaBf16SplitArm,
    bounded: bool,
    collect_stats: bool,
) -> HipResult<(bool, Vec<QsaFilterRowStat>)> {
    qsa_select_batch_validate(p)?;
    if !gpu.arch_caps.has_gfx11_plus_simt()
        || gpu.graphs.capture_mode
        || (collect_stats && gpu.replay.is_recording())
        || !qsa_bf16_split_covers(p)
    {
        return Err(HipError::new(0, "QSA BF16 split arm: call not covered"));
    }
    let filter = match arm {
        QsaBf16SplitArm::Select => None,
        QsaBf16SplitArm::Filter(knobs) => Some(knobs.cfg()),
    };
    let mut raw = Vec::new();
    let persisted = indexed_attention_select_bf16_split(
        gpu,
        p,
        mirror,
        filter,
        bounded || gpu.replay.is_recording(),
        (collect_stats && filter.is_some()).then_some(&mut raw),
    )?;
    Ok((persisted, qsa_filter_stats(raw)))
}

/// Device-side stable reuse of a prior MTP QSA selection row.
///
/// `selected` is a byte-addressed [`DType::Raw`] allocation containing i32
/// indices. `selected_len_out` is a persistent four-byte Raw scalar written by
/// the kernel; callers can read only that scalar when the logical span changes,
/// never the selected row itself.
pub struct IndexedAttentionReuseSelection<'a> {
    pub selected: &'a GpuTensor,
    pub selected_len: usize,
    pub position: usize,
    pub capacity: usize,
    pub selected_len_out: &'a GpuTensor,
}

pub fn indexed_attention_reuse_selection(
    gpu: &mut Gpu,
    p: &IndexedAttentionReuseSelection<'_>,
) -> HipResult<()> {
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if p.selected.dtype != DType::Raw
        || p.selected_len > p.capacity
        || p.capacity == 0
        || p.selected.numel() < selected_bytes
        || p.selected_len_out.dtype != DType::Raw
        || p.selected_len_out.numel() < std::mem::size_of::<i32>()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let selected_len = checked_i32(p.selected_len, "QSA reuse selected length")?;
    let position = checked_i32(p.position, "QSA reuse position")?;
    let capacity = checked_i32(p.capacity, "QSA reuse capacity")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_reuse_selection",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.selected_len_out.buf.as_ptr());
    args.push_i32(selected_len);
    args.push_i32(position);
    args.push_i32(capacity);
    args.pad_to(16);
    // The prior entries snapshot (the kernel compacts in place).
    let snapshot_bytes = p.selected_len.min(p.capacity) * std::mem::size_of::<i32>();
    if snapshot_bytes > QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES {
        return Err(HipError::new(
            0,
            "QSA reuse selection exceeds its LDS snapshot",
        ));
    }
    if gpu.flags.qwen4_qsa_extent_check {
        // One workgroup: reads `min(selected_len, capacity)` words, writes up to
        // `capacity` words of `selected` and the single `selected_len_out` word.
        let depth = QsaSelDepth {
            rows: 1,
            end_position: p.position,
            block_count: 0,
        };
        gpu.qsa_extent_check(
            "qsa_select_selected",
            p.selected,
            0,
            selected_bytes,
            depth,
            "indexed_attention_reuse_selection",
        );
        gpu.qsa_extent_check(
            "qsa_select_len_out",
            p.selected_len_out,
            0,
            std::mem::size_of::<i32>(),
            depth,
            "indexed_attention_reuse_selection",
        );
    }
    gpu.launch_blob_recorded(
        "indexed_attention_reuse_selection",
        [1, 1, 1],
        [256, 1, 1],
        snapshot_bytes as u32,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        gpu.qsa_extent_post_launch(
            "qsa_select_selected",
            p.selected,
            0,
            selected_bytes,
            QsaSelDepth {
                rows: 1,
                end_position: p.position,
                block_count: 0,
            },
            "indexed_attention_reuse_selection",
        );
    }
    Ok(())
}

/// Formula inputs for a QSA launch whose active count is a quotient of the
/// decode position (`(position_start + rows) / compress`).
///
/// Declaring them lets the retained recorder re-derive the count for a replay
/// position instead of replaying the capture-position count. The wrapper
/// verifies the caller's count *equals* the declared formula, so the
/// declaration cannot drift from the launch.
#[derive(Clone, Copy, Debug)]
pub struct QsaPositionBinding {
    pub position_start: usize,
    pub rows: usize,
}

pub struct IndexedAttentionPoolRope<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    /// Learned index-key RMSNorm. `None` is reserved for synthetic parity
    /// buffers, which intentionally retain the plain-RMS behavior.
    pub norm: Option<&'a GpuTensor>,
    pub block_count: usize,
    pub compress: usize,
    pub index_dim: usize,
    /// Declared source of `block_count`, or `None` for a synthetic caller whose
    /// count is not position-derived (which then declares nothing).
    pub position: Option<QsaPositionBinding>,
    /// `grid.x` bound for a full (non-incremental) launch. A retained tape needs
    /// the grid to be position-independent, so callers pass the capacity the
    /// kernel masks against (`>= block_count`); a caller without a capacity
    /// passes the active count. Must be `>= block_count`. An incremental launch
    /// instead uses `ceil(rows / compress)` (capped by this bound), which is
    /// also position-independent.
    pub grid_bound: usize,
    /// [`crate::rope::Rope::yarn`] at the pooled keys' rotary width
    /// (`min(index_dim, 64)`).
    pub yarn: crate::rope::RopeYarn,
}

pub fn indexed_attention_pool_rope(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
) -> HipResult<()> {
    indexed_attention_pool_rope_impl(gpu, p, false)
}

/// [`indexed_attention_pool_rope`] that pools only the blocks the launch's
/// rows complete: those below `position_start / compress` were pooled by an
/// earlier launch from the same raw keys and are left as they are (a declared
/// position is required).
pub fn indexed_attention_pool_rope_incremental(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
) -> HipResult<()> {
    indexed_attention_pool_rope_impl(gpu, p, true)
}

fn indexed_attention_pool_rope_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
    incremental: bool,
) -> HipResult<()> {
    if incremental && p.position.is_none() {
        return Err(HipError::new(
            0,
            "incremental QSA pooling needs a declared position",
        ));
    }
    // Raw and pooled keys share one arena dtype (`QsaKvFormat::index_dtype`).
    let kernel = match (p.raw_keys.dtype, p.pooled.dtype) {
        (DType::F32, DType::F32) => "indexed_attention_pool_rope_f32",
        (DType::BF16, DType::BF16) => "indexed_attention_pool_rope_bf16",
        _ => return Err(HipError::new(0, &ComputeError::WrongDtype.to_string())),
    };
    if p.block_count == 0 || p.compress == 0 || p.index_dim == 0 || p.index_dim % 2 != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if let Some(norm) = p.norm {
        if norm.dtype != DType::BF16 || norm.numel() != p.index_dim {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
    }
    let raw_elements = checked_product3(
        p.block_count,
        p.compress,
        p.index_dim,
        "QSA pool/RoPE raw extent",
    )?;
    let pooled_elements =
        checked_product(p.block_count, p.index_dim, "QSA pool/RoPE pooled extent")?;
    let block_count = checked_i32(p.block_count, "QSA pool/RoPE block count")?;
    let compress = checked_i32(p.compress, "QSA pool/RoPE compression")?;
    let index_dim = checked_i32(p.index_dim, "QSA pool/RoPE index width")?;
    // A declared position source must reproduce the count this launch performs;
    // otherwise replay would re-derive a different amount of pooling.
    if let Some(position) = p.position {
        let expected = position
            .position_start
            .checked_add(position.rows)
            .ok_or_else(|| HipError::new(0, "QSA pool/RoPE position overflow"))?
            / p.compress;
        if expected != p.block_count {
            return Err(HipError::new(
                0,
                &format!(
                    "QSA pool/RoPE block count {} is not (position_start {} + rows {}) / compress {} = {expected}",
                    p.block_count, position.position_start, position.rows, p.compress
                ),
            ));
        }
    }
    // The kernel masks inactive blocks before its first read, so a bound above
    // the active count only skips workgroups; a bound below it would leave work
    // undone and is refused.
    if p.grid_bound < p.block_count {
        return Err(HipError::new(
            0,
            &format!(
                "QSA pool/RoPE grid bound {} is below the active block count {}",
                p.grid_bound, p.block_count
            ),
        ));
    }
    // An incremental launch completes `(start + rows) / c - start / c <=
    // ceil(rows / c)` blocks from `first_block` (the kernel offsets
    // `blockIdx.x` by it), so its grid depends on `rows` only: constant across
    // replayed positions, with `first_block` arriving as a declared kernarg.
    let grid_x = match (incremental, p.position) {
        (true, Some(position)) => position.rows.div_ceil(p.compress).max(1).min(p.grid_bound),
        _ => p.grid_bound,
    };
    let block_grid = checked_u32(grid_x, "QSA pool/RoPE block grid")?;
    let dim_grid = blocks(p.index_dim)?;
    if p.raw_keys.numel() < raw_elements || p.pooled.numel() < pooled_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_ptr(
        p.norm
            .map(|norm| norm.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut()),
    );
    args.push_i32(block_count);
    // Declared dynamic field: replay re-derives `(position + rows) / compress`
    // rather than replaying the capture-position pooling count.
    let block_count_offset = args.len() - 4;
    args.push_i32(compress);
    args.push_i32(index_dim);
    let first_block = match (incremental, p.position) {
        (true, Some(position)) => position.position_start / p.compress,
        _ => 0,
    };
    args.push_i32(checked_i32(first_block, "QSA pool/RoPE first block")?);
    // Declared dynamic too when incremental: `position_start / compress`.
    let first_block_offset = args.len() - 4;
    p.yarn.push(&mut args);
    args.pad_to(16);
    let addend = u32::try_from(p.position.map_or(0, |position| position.rows))
        .map_err(|_| HipError::new(0, "QSA pool/RoPE rows exceed u32"))?;
    let divisor = u32::try_from(p.compress)
        .map_err(|_| HipError::new(0, "QSA pool/RoPE compression exceeds u32"))?;
    let bindings = [
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: block_count_offset,
            addend,
            divisor,
        },
        crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: first_block_offset,
            addend: 0,
            divisor,
        },
    ];
    let kernargs: &[crate::replay::ReplayKernargBinding] = match (p.position, incremental) {
        (None, _) => &[],
        (Some(_), false) => &bindings[..1],
        (Some(_), true) => &bindings[..],
    };
    // Extents the kernel touches (`qsa_pool_rope`): workgroup `blockIdx.x` pools
    // block `first_block + blockIdx.x` and returns before any access when that
    // is `>= block_count`, so `grid_bound` above the active count only skips
    // workgroups and the guarded block count is
    // `min(block_count - first_block, grid_x)`. Per pooled block it reads
    // `compress * index_dim` raw elements from `block * compress * index_dim`
    // (channel `< index_dim`, including the RoPE partner `< rotary_dim <=
    // index_dim`), `index_dim` norm BF16 elements when a norm is bound, and
    // writes `index_dim` pooled elements at `block * index_dim`. `dim_grid`
    // rounds channels up to 256; channels `>= index_dim` only join the
    // reduction (zero) and touch no memory.
    let extents = || {
        let done = p.block_count.saturating_sub(first_block).min(grid_x);
        let raw_elem = p.raw_keys.dtype.size();
        let pooled_elem = p.pooled.dtype.size();
        let mut sites = Vec::with_capacity(3);
        sites.push((
            "qsa_raw_index_keys",
            p.raw_keys,
            first_block * p.compress * p.index_dim * raw_elem,
            done * p.compress * p.index_dim * raw_elem,
        ));
        sites.push((
            "qsa_pooled_index_keys",
            p.pooled,
            first_block * p.index_dim * pooled_elem,
            done * p.index_dim * pooled_elem,
        ));
        if let Some(norm) = p.norm {
            sites.push(("qsa_index_key_norm", norm, 0, p.index_dim * 2));
        }
        let end_position = p.position.map_or(p.block_count * p.compress, |position| {
            position.position_start + position.rows
        });
        (
            sites,
            crate::dispatch::QsaExtentDepth {
                rows: p.position.map_or(0, |position| position.rows),
                end_position,
                block_count: p.block_count,
            },
        )
    };
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_check(gpu, &sites, depth, "indexed_attention_pool_rope");
    }
    gpu.launch_blob_recorded(
        kernel,
        [block_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs,
        },
    )?;
    if gpu.flags.qwen4_qsa_extent_check {
        let (sites, depth) = extents();
        qsa_append_extent_post(gpu, &sites, depth, "indexed_attention_pool_rope");
    }
    Ok(())
}

pub struct IndexedAttentionAttention<'a> {
    pub q_with_gate: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub selected_len: usize,
    /// Reserved cache rows (kernel argument; the only in-kernel token bound of
    /// the single-row kernels).
    pub full_capacity: usize,
    /// Developer extent-check metadata only (`qwen4_qsa_extent_check`); never a
    /// kernel argument. The caller guarantees every selected id is below
    /// `end_position` (the row's visible end, `position + 1`); the check bounds
    /// K/V by `min(end_position, full_capacity)` rows. Callers compute it
    /// only when the flag is on and otherwise pass `0`.
    pub end_position: usize,
    /// K/V cache format.
    pub format: QsaKvFormat,
}

/// One device-to-device copy for [`copy_regions`].
pub struct CopyRegion<'a> {
    pub dst: &'a hip_bridge::DeviceBuffer,
    pub dst_offset: usize,
    pub src: &'a hip_bridge::DeviceBuffer,
    pub src_offset: usize,
    pub bytes: usize,
}

/// Regions per `copy_regions_u32` launch (its by-value kernarg table).
const COPY_REGIONS_MAX: usize = 64;

/// Many small independent device copies in one launch per 64 regions instead
/// of one blit dispatch (and host API call) each. Regions of one call run
/// concurrently, so none may write another's source or destination. A region
/// that is not 4-byte aligned takes a plain copy.
///
/// This is the shared copy helper of the QSA snapshot/restore, session-cache and
/// VMM-prefix paths: it writes and reads the QSA K/V, raw/pooled index-key and
/// F16-arena VMM owners. Under `qwen4_qsa_extent_check` every region (also the
/// unaligned `memcpy_dtod_at_auto` fallback) has both byte ranges checked
/// against the buffer's mapped bytes before the copy, and the post-launch
/// helper runs right after each `copy_regions_u32` launch (and each fallback
/// copy). Guarded extent: the kernel copies exactly `words = bytes / 4`
/// words per region (`i < words`; the grid-stride loop and the `grid.x`
/// round-up never exceed it), so `[offset, offset + bytes)` is exact.
pub fn copy_regions(gpu: &mut Gpu, regions: &[CopyRegion<'_>]) -> HipResult<()> {
    // One flag test; the unchecked instantiation carries no check state.
    if gpu.flags.qwen4_qsa_extent_check {
        copy_regions_impl::<true>(gpu, regions)
    } else {
        copy_regions_impl::<false>(gpu, regions)
    }
}

/// Body of [`copy_regions`]; `CHECK` is `qwen4_qsa_extent_check`. Every region
/// is pre-checked; each flush gets ONE post-launch call on the last region
/// queued into it (the post helper synchronizes the whole device, so one
/// representative suffices), and each fallback copy gets its own.
fn copy_regions_impl<const CHECK: bool>(
    gpu: &mut Gpu,
    regions: &[CopyRegion<'_>],
) -> HipResult<()> {
    let mut table = [(0u64, 0u64, 0u32); COPY_REGIONS_MAX];
    let mut count = 0;
    let mut last_queued: Option<&CopyRegion<'_>> = None;
    let depth = crate::dispatch::QsaExtentDepth {
        rows: regions.len(),
        end_position: 0,
        block_count: 0,
    };
    for region in regions {
        if region.dst_offset + region.bytes > region.dst.size()
            || region.src_offset + region.bytes > region.src.size()
        {
            return Err(HipError::new(0, "copy_regions: region exceeds its buffer"));
        }
        if CHECK {
            qsa_copy_region_extent_check(gpu, region, depth);
        }
        let dst = region.dst.as_ptr() as u64 + region.dst_offset as u64;
        let src = region.src.as_ptr() as u64 + region.src_offset as u64;
        let words = region.bytes / 4;
        if !(dst | src | region.bytes as u64).is_multiple_of(4) || words > u32::MAX as usize {
            gpu.memcpy_dtod_at_auto(
                region.dst,
                region.dst_offset,
                region.src,
                region.src_offset,
                region.bytes,
            )?;
            if CHECK {
                qsa_copy_region_extent_post(gpu, region, depth);
            }
            continue;
        }
        if words == 0 {
            continue;
        }
        table[count] = (dst, src, words as u32);
        count += 1;
        if CHECK {
            last_queued = Some(region);
        }
        if count == COPY_REGIONS_MAX {
            launch_copy_regions(gpu, &table[..count])?;
            if CHECK {
                if let Some(region) = last_queued.take() {
                    qsa_copy_region_extent_post(gpu, region, depth);
                }
            }
            count = 0;
        }
    }
    if count > 0 {
        launch_copy_regions(gpu, &table[..count])?;
        if CHECK {
            if let Some(region) = last_queued.take() {
                qsa_copy_region_extent_post(gpu, region, depth);
            }
        }
    }
    Ok(())
}

/// Extent check of both byte ranges of one [`copy_regions`] region.
fn qsa_copy_region_extent_check(
    gpu: &Gpu,
    region: &CopyRegion<'_>,
    depth: crate::dispatch::QsaExtentDepth,
) {
    gpu.qsa_extent_check_buffer(
        "qsa_copy_dst",
        region.dst,
        region.dst_offset,
        region.bytes,
        depth,
        "copy_regions",
    );
    gpu.qsa_extent_check_buffer(
        "qsa_copy_src",
        region.src,
        region.src_offset,
        region.bytes,
        depth,
        "copy_regions",
    );
}

/// Post-launch sync/fault check of both byte ranges of one [`copy_regions`] region.
fn qsa_copy_region_extent_post(
    gpu: &Gpu,
    region: &CopyRegion<'_>,
    depth: crate::dispatch::QsaExtentDepth,
) {
    gpu.qsa_extent_post_launch_buffer(
        "qsa_copy_dst",
        region.dst,
        region.dst_offset,
        region.bytes,
        depth,
        "copy_regions",
    );
    gpu.qsa_extent_post_launch_buffer(
        "qsa_copy_src",
        region.src,
        region.src_offset,
        region.bytes,
        depth,
        "copy_regions",
    );
}

pub(crate) fn launch_copy_regions(gpu: &mut Gpu, table: &[(u64, u64, u32)]) -> HipResult<()> {
    const NAME: &str = "copy_regions_u32";
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, NAME)?;
    let mut args = KernargBlob::new();
    for index in 0..COPY_REGIONS_MAX {
        args.push_u64(table.get(index).map_or(0, |entry| entry.0));
    }
    for index in 0..COPY_REGIONS_MAX {
        args.push_u64(table.get(index).map_or(0, |entry| entry.1));
    }
    for index in 0..COPY_REGIONS_MAX {
        args.push_u32(table.get(index).map_or(0, |entry| entry.2));
    }
    args.push_i32(table.len() as i32);
    args.pad_to(16);
    let max_words = table.iter().map(|entry| entry.2).max().unwrap_or(0);
    gpu.launch_blob_recorded(
        NAME,
        [max_words.div_ceil(256).clamp(1, 64), table.len() as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Selected-list length of the batched QSA attention kernels for the row whose
/// visible window ends at `visible` tokens: the kernels' own derivation
/// (`chosen * compress + visible - block_count * compress`, capped by
/// `capacity`). Developer extent checks only.
fn qsa_attention_selected_len(
    visible: usize,
    budget_blocks: usize,
    compress: usize,
    capacity: usize,
) -> usize {
    let block_count = visible.checked_div(compress).unwrap_or(0);
    budget_blocks
        .min(block_count)
        .saturating_mul(compress)
        .saturating_add(visible - block_count.saturating_mul(compress))
        .min(capacity)
}

/// Exact end (exclusive byte) of the block-transposed F16 V
/// `[token / 4][width][token % 4]` that the first `tokens` cache rows touch:
/// the last token's last channel, `((last / 4) * width + width - 1) * 8 +
/// (last % 4) * 2 + 2`. Zero tokens touch nothing. Developer extent checks only.
fn qsa_vb16_extent(tokens: usize, width: usize) -> usize {
    if tokens == 0 || width == 0 {
        return 0;
    }
    let last = tokens - 1;
    (last / 4)
        .saturating_mul(width)
        .saturating_mul(8)
        .saturating_add((width - 1).saturating_mul(8))
        .saturating_add((last % 4) * 2 + 2)
}

/// Host extents of one batched QSA attention launch for the developer extent
/// checks (`gpu.flags.qwen4_qsa_extent_check`); built only inside that flag.
/// `kv_rows` is the visible end the kernels' token guard admits
/// (`token < full_capacity && token < visible`, visible <= `end_position`),
/// never the selected count.
struct QsaBatchExtents {
    depth: crate::dispatch::QsaExtentDepth,
    kv_rows: usize,
    kv_bytes: usize,
    q_bytes: usize,
    selected_bytes: usize,
    output_bytes: usize,
}

impl QsaBatchExtents {
    fn new(p: &IndexedAttentionAttentionBatch<'_>, end_position: usize) -> Self {
        let f32_bytes = std::mem::size_of::<f32>();
        let kv_rows = end_position.min(p.full_capacity);
        let selected_last =
            qsa_attention_selected_len(end_position, p.budget_blocks, p.compress, p.capacity);
        Self {
            depth: crate::dispatch::QsaExtentDepth {
                rows: p.rows,
                end_position,
                block_count: end_position.checked_div(p.compress).unwrap_or(0),
            },
            kv_rows,
            kv_bytes: kv_rows.saturating_mul(p.format.kv_row_bytes(p.n_kv_heads, p.head_dim)),
            // Whole `[rows][2 * n_heads * head_dim]` query + gate rows.
            q_bytes: p
                .rows
                .saturating_mul(p.n_heads)
                .saturating_mul(p.head_dim)
                .saturating_mul(2)
                .saturating_mul(f32_bytes),
            // Row r reads `[r * capacity, r * capacity + selected_len(r))`; the
            // last row ends furthest (earlier rows end by `r * capacity + capacity`).
            selected_bytes: p
                .rows
                .saturating_sub(1)
                .saturating_mul(p.capacity)
                .saturating_add(selected_last)
                .saturating_mul(std::mem::size_of::<i32>()),
            output_bytes: p
                .rows
                .saturating_mul(p.n_heads)
                .saturating_mul(p.head_dim)
                .saturating_mul(f32_bytes),
        }
    }

    fn check_query(&self, gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>, site: &'static str) {
        gpu.qsa_extent_check(
            "qsa_attn_q",
            p.q_with_gate,
            0,
            self.q_bytes,
            self.depth,
            site,
        );
    }

    fn check_kv(&self, gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>, site: &'static str) {
        gpu.qsa_extent_check(
            "qsa_full_keys",
            p.full_keys,
            0,
            self.kv_bytes,
            self.depth,
            site,
        );
        gpu.qsa_extent_check(
            "qsa_full_values",
            p.full_values,
            0,
            self.kv_bytes,
            self.depth,
            site,
        );
    }

    fn check_selected(
        &self,
        gpu: &Gpu,
        p: &IndexedAttentionAttentionBatch<'_>,
        site: &'static str,
    ) {
        gpu.qsa_extent_check(
            "qsa_attn_selected",
            p.selected,
            0,
            self.selected_bytes,
            self.depth,
            site,
        );
    }

    fn check_output(&self, gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>, site: &'static str) {
        gpu.qsa_extent_check(
            "qsa_attn_output",
            p.output,
            0,
            self.output_bytes,
            self.depth,
            site,
        );
    }

    /// Post-launch fault probe: one per launch, named by `site`; the operand
    /// shown on a fault is the output extent.
    fn post(&self, gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>, site: &'static str) {
        gpu.qsa_extent_post_launch(
            "qsa_attn_output",
            p.output,
            0,
            self.output_bytes,
            self.depth,
            site,
        );
    }
}

pub fn indexed_attention_attention(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttention<'_>,
) -> HipResult<()> {
    for tensor in [p.q_with_gate, p.output] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.n_kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    if p.selected.dtype != DType::Raw
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.head_dim == 0
        || p.n_heads % p.n_kv_heads != 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA attention head extent")?;
    let q_elements = checked_product(2, head_elements, "QSA attention query extent")?;
    let output_elements = head_elements;
    let full_elements = checked_product(
        p.full_capacity,
        p.format.kv_row_units(p.n_kv_heads, p.head_dim),
        "QSA attention cache extent",
    )?;
    let selected_bytes = p
        .selected_len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let n_heads = checked_i32(p.n_heads, "QSA attention head count")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA attention KV-head count")?;
    let head_dim = checked_i32(p.head_dim, "QSA attention head width")?;
    let selected_len = checked_i32(p.selected_len, "QSA attention selected length")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA attention capacity")?;
    let head_grid = checked_u32(p.n_heads, "QSA attention head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.q_with_gate.numel() != q_elements
        || p.output.numel() != output_elements
        || p.full_keys.numel() < full_elements
        || p.full_values.numel() < p.full_keys.numel()
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let [parallel, serial] = [
        p.format.kernel([
            "indexed_attention_attention_f32",
            "indexed_attention_attention_fp8",
            "indexed_attention_attention_q8",
            "indexed_attention_attention_fp8_fnuz",
        ]),
        p.format.kernel([
            "indexed_attention_attention_f32_serial",
            "indexed_attention_attention_fp8_serial",
            "indexed_attention_attention_q8_serial",
            "indexed_attention_attention_fp8_fnuz_serial",
        ]),
    ];
    let (kernel_name, shared_mem) =
        match p.selected_len.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if gpu.arch_caps.has_gfx11_plus_simt()
                    // The kernel adds 32 bytes of static LDS (per-wave maxes).
                    && bytes + 32 <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (parallel, bytes as u32)
            }
            _ => (serial, 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [
        p.q_with_gate,
        p.full_keys,
        p.full_values,
        p.selected,
        p.output,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [n_heads, n_kv_heads, head_dim, selected_len, full_capacity] {
        args.push_i32(value);
    }
    args.pad_to(16);
    // Single-row kernels take no position: `qsa_attention_single` passes
    // `visible = full_capacity` and `qsa_attention_serial` has no visible
    // check, so the kernel's only guard on an indirect K/V read is
    // `token >= 0 && token < full_capacity`. The host bound is the caller's
    // `end_position` (instrumentation only, never a kernel argument): callers
    // guarantee every selected id is `< end_position`, so K/V reads are
    // bounded by `min(end_position, full_capacity)` cache rows, not by the
    // reserved `full_capacity` nor the selection count. Selected is read for
    // `[0, selected_len)`; the per-head `dim_grid * 256` channel rounding is
    // guarded by `channel >= head_dim`.
    let single_extents = gpu.flags.qwen4_qsa_extent_check.then(|| {
        (
            crate::dispatch::QsaExtentDepth {
                rows: 1,
                end_position: p.end_position,
                block_count: 0,
            },
            p.end_position
                .min(p.full_capacity)
                .saturating_mul(p.format.kv_row_bytes(p.n_kv_heads, p.head_dim)),
        )
    });
    if let Some((depth, kv_bytes)) = single_extents {
        let f32_bytes = std::mem::size_of::<f32>();
        gpu.qsa_extent_check(
            "qsa_attn_q",
            p.q_with_gate,
            0,
            q_elements.saturating_mul(f32_bytes),
            depth,
            kernel_name,
        );
        gpu.qsa_extent_check(
            "qsa_full_keys",
            p.full_keys,
            0,
            kv_bytes,
            depth,
            kernel_name,
        );
        gpu.qsa_extent_check(
            "qsa_full_values",
            p.full_values,
            0,
            kv_bytes,
            depth,
            kernel_name,
        );
        gpu.qsa_extent_check(
            "qsa_attn_selected",
            p.selected,
            0,
            selected_bytes,
            depth,
            kernel_name,
        );
        gpu.qsa_extent_check(
            "qsa_attn_output",
            p.output,
            0,
            output_elements.saturating_mul(f32_bytes),
            depth,
            kernel_name,
        );
    }
    gpu.launch_blob_recorded(
        kernel_name,
        [head_grid, dim_grid, 1],
        [QSA_ATTENTION_PARALLEL_THREADS, 1, 1],
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some((depth, kv_bytes)) = single_extents {
        gpu.qsa_extent_post_launch(
            "qsa_full_keys",
            p.full_keys,
            0,
            kv_bytes,
            depth,
            kernel_name,
        );
    }
    Ok(())
}

pub struct IndexedAttentionAttentionBatch<'a> {
    pub q_with_gate: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
    pub position_start: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub capacity: usize,
    pub full_capacity: usize,
    /// K/V cache format.
    pub format: QsaKvFormat,
    /// Declared shape bound for the LDS reservation and the kernel symbol. A
    /// retained tape needs both to be position-independent, so callers pass the
    /// selected-row capacity (`>= max_selected`); a caller without a capacity
    /// passes the position-derived length. Must be `>= max_selected`.
    ///
    /// `grid.x` is derived from `n_heads` only (all heads, or head groups of
    /// four for the grouped kernel), so it stays position-independent.
    pub shape_selected: usize,
}

/// Query heads per workgroup in `indexed_attention_attention_f32_batched_hg4`.
const QSA_ATTENTION_HG4_HEADS: usize = 4;
/// Query heads per KV head of `indexed_attention_attention_f32_batched_hg12`
/// (one workgroup per KV group; its scores live in registers, QSA_T = 9
/// tiles of 256 selected rows).
const QSA_ATTENTION_HG12_HEADS: usize = 12;
const QSA_ATTENTION_HG12_MAX_SELECTED: usize = 9 * 256;
/// Few-row verify launches only `rows * n_kv_heads` hg12 workgroups; hg4's
/// three per KV group fill the GPU better there (4-row MTP verify at 32k
/// context: decode 51.4 -> 50.8 tok/s with hg12).
const QSA_ATTENTION_HG12_MIN_ROWS: usize = 16;

/// Below this many rows the grouped kernel launches too few workgroups
/// (`rows * n_heads / 4`) to fill the GPU and the per-head kernel is faster
/// (decode: 0.48 vs 0.73 ms per call at 1131 context on gfx1151). From two
/// rows (speculative verify) sharing each K/V row across four heads wins
/// (4 rows, 232 context: 55 -> 27 us per layer).
const QSA_ATTENTION_HG4_MIN_ROWS: usize = 2;

/// Whether the grouped prefill attention runs its low-LDS twin
/// (`indexed_attention_attention_q8_batched_hg4_lite`): q8 K/V on exact gfx1100
/// with `HIPFIRE_QWEN4_QSA_HG4_Q8_LITE` not `0`. The kernel symbol and the
/// dynamic-LDS size both follow from this one decision.
fn qsa_attention_hg4_lite(flags: &crate::feature_flags::FeatureFlags, format: QsaKvFormat) -> bool {
    format == QsaKvFormat::Q8 && flags.qwen4_qsa_hg4_q8_lite_enabled()
}

/// Dynamic LDS bytes of the grouped kernel for `shape_selected` selected rows.
/// Shipped kernels: weights[sel][4] + tokens[sel] + partial maxes[4][256] +
/// maxes[4]. The lite twin recomputes the validated token and reduces the
/// maxes with wave shuffles: weights[sel][4] + 8 waves x 4 partial maxes
/// (reused for the four denominators) = `selected * 16 + 128`.
fn qsa_attention_hg4_lds_bytes_for(shape_selected: usize, lite: bool) -> Option<usize> {
    let (per_row, fixed) = if lite {
        (4 * QSA_ATTENTION_HG4_HEADS, 4 * 8 * QSA_ATTENTION_HG4_HEADS)
    } else {
        (
            4 * QSA_ATTENTION_HG4_HEADS + 4,
            4 * (QSA_ATTENTION_HG4_HEADS * 256 + QSA_ATTENTION_HG4_HEADS),
        )
    };
    shape_selected.checked_mul(per_row)?.checked_add(fixed)
}

/// Dynamic LDS for the grouped kernel, or `None` when the shape is not the one
/// it is specialised for (gfx11+ ISA, head_dim 256, four heads per KV group,
/// prefill-sized row counts).
fn qsa_attention_hg4_lds_bytes(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    shape_selected: usize,
) -> Option<u32> {
    let group = p.n_heads / p.n_kv_heads;
    if !gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
        || p.head_dim != 256
        || group % QSA_ATTENTION_HG4_HEADS != 0
        || gpu.route_rows(p.rows) < QSA_ATTENTION_HG4_MIN_ROWS
    {
        return None;
    }
    let bytes = qsa_attention_hg4_lds_bytes_for(
        shape_selected,
        qsa_attention_hg4_lite(&gpu.flags, p.format),
    )?;
    (bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES).then_some(bytes as u32)
}

pub fn indexed_attention_attention_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
) -> HipResult<()> {
    indexed_attention_attention_batch_impl(gpu, p, true)
}

/// Developer-only exact reference for the QSA oracle: the same
/// `IndexedAttentionAttentionBatch` through the per-head kernels
/// (`allow_fast = false`), bypassing the grouped hg4 and dense/gathered WMMA
/// routes. Not a production route; Unit0 harness only. Gated behind `lab`
/// (or test) so production builds expose no new API.
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn indexed_attention_attention_batch_exact(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
) -> HipResult<()> {
    indexed_attention_attention_batch_impl(gpu, p, false)
}

/// The gathered route's launches for `p` on the hipcc kernels or, with `pm`,
/// the builder module (`HIPFIRE_QWEN4_QSA_PM`), whatever the route flags say.
/// Byte-equality and timing harness only (`examples/qsa_pm_check.rs`).
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn indexed_attention_gathered_batch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    pm: bool,
) -> HipResult<()> {
    let end_position = p.position_start + p.rows;
    let max_selected = end_position
        .min(p.capacity)
        .min(p.budget_blocks * p.compress + p.compress - 1);
    if pm && !qsa_gathered_pm_fits(p) {
        return Err(HipError::new(
            0,
            "QSA PM gathered route does not cover this shape",
        ));
    }
    qsa_gathered_wmma_launch_with(gpu, p, end_position, max_selected, pm, None)
}

/// `allow_fast` admits the grouped hg4 kernel and the dense F16 WMMA route;
/// without it the per-head kernels run (the exact reference).
fn indexed_attention_attention_batch_impl(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    allow_fast: bool,
) -> HipResult<()> {
    indexed_attention_attention_batch_arena(gpu, p, allow_fast, None)
}

/// [`indexed_attention_attention_batch`] for a layer that keeps a persistent
/// F16 K/V `arena` (F32 state only; `None` = the shared per-launch
/// workspace). The gathered route first brings the arena up to this launch's
/// rows ([`qsa_f16_arena_sync`]), then attends from it; every other route
/// ignores it.
pub fn indexed_attention_attention_batch_with_arena(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    arena: Option<&QsaF16ArenaRef<'_>>,
) -> HipResult<()> {
    indexed_attention_attention_batch_arena(gpu, p, true, arena)
}

fn indexed_attention_attention_batch_arena(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    allow_fast: bool,
    arena: Option<&QsaF16ArenaRef<'_>>,
) -> HipResult<()> {
    for tensor in [p.q_with_gate, p.output] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.n_kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    if p.selected.dtype != DType::Raw
        || p.rows == 0
        || p.position_start.checked_add(p.rows).is_none()
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.head_dim == 0
        || p.n_heads % p.n_kv_heads != 0
        || p.compress == 0
        || p.capacity == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA batch attention heads")?;
    let q_elements = checked_product(2, head_elements, "QSA batch attention query")?;
    let kv_elements = checked_product(
        p.full_capacity,
        p.format.kv_row_units(p.n_kv_heads, p.head_dim),
        "QSA batch attention cache",
    )?;
    let q_rows = checked_product(p.rows, q_elements, "QSA batch attention query rows")?;
    let output_rows = checked_product(p.rows, head_elements, "QSA batch attention output rows")?;
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA batch attention selected rows")?,
        std::mem::size_of::<i32>(),
        "QSA batch attention selected",
    )?;
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if end_position > i32::MAX as usize
        || end_position > p.full_capacity
        || p.q_with_gate.numel() < q_rows
        || p.output.numel() < output_rows
        || p.full_keys.numel() < kv_elements
        || p.full_values.numel() < p.full_keys.numel()
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let selected_bound = checked_product(
        p.budget_blocks,
        p.compress,
        "QSA batch attention selected bound",
    )?
    .checked_add(p.compress - 1)
    .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let max_selected = end_position.min(p.capacity).min(selected_bound);
    if p.shape_selected < max_selected {
        return Err(HipError::new(
            0,
            &format!(
                "QSA batch attention shape bound {} is below the active selected length {max_selected}",
                p.shape_selected
            ),
        ));
    }
    // A verify window can cross an LDS/policy boundary. Keep singleton
    // arithmetic per absolute position; adjacent identical recipes share a
    // launch. Runs are scalar ranges, not allocated tensor subviews.
    if qsa_decode_rows_admitted(gpu, p.rows) && qsa_decode_splitk_eager(gpu) {
        let policy_for = |gpu: &Gpu, row| {
            if allow_fast {
                qsa_decode_attention_policy_for_row(gpu, p, row)
            } else {
                QsaDecodeAttentionPolicy::PerHead
            }
        };
        let mut first = 0;
        while first < p.rows {
            let policy = policy_for(gpu, first);
            let parallel = qsa_decode_per_head_parallel(gpu, p, first);
            let mut end = first + 1;
            while end < p.rows
                && policy_for(gpu, end) == policy
                && (policy != QsaDecodeAttentionPolicy::PerHead
                    || qsa_decode_per_head_parallel(gpu, p, end) == parallel)
            {
                end += 1;
            }
            indexed_attention_attention_batch_launch(
                gpu,
                p,
                allow_fast,
                arena,
                Some(policy),
                first,
                end - first,
            )?;
            first = end;
        }
        return Ok(());
    }
    let policy = qsa_decode_attention_policy(gpu, p);
    indexed_attention_attention_batch_launch(gpu, p, allow_fast, arena, policy, 0, p.rows)
}

fn indexed_attention_attention_batch_launch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    allow_fast: bool,
    arena: Option<&QsaF16ArenaRef<'_>>,
    policy: Option<QsaDecodeAttentionPolicy>,
    first: usize,
    run_rows: usize,
) -> HipResult<()> {
    let end_position = p.position_start + first + run_rows;
    let max_selected = qsa_attention_selected_at(p, first + run_rows - 1);
    let decode = policy.is_some();
    if allow_fast && !decode && qsa_dense_wmma_applies(gpu, p, end_position) {
        return qsa_dense_wmma(gpu, p, end_position);
    }
    // Split decode, default on (HIPFIRE_QWEN4_QSA_DECODE_SPLITK=0 kill switch):
    // eager hd256 admitted decode calls, split-K for F32 K/V (exact gfx1151)
    // and split-KV for q8 K/V (exact gfx1151 and gfx1100); KLD-gated, not
    // bit-exact. The reference entry (`allow_fast = false`) stays per-head.
    if allow_fast {
        match policy {
            Some(QsaDecodeAttentionPolicy::SplitK { splits, layout }) => {
                return qsa_decode_splitk_launch_run(
                    gpu,
                    p,
                    max_selected,
                    splits,
                    layout,
                    first,
                    run_rows,
                );
            }
            Some(QsaDecodeAttentionPolicy::SplitKvQ8 { splits }) => {
                return qsa_decode_splitkv_q8_launch_run(gpu, p, splits, first, run_rows);
            }
            Some(QsaDecodeAttentionPolicy::PerHead) | None => {}
        }
    }
    if allow_fast && !decode && qsa_gathered_wmma_applies(gpu, p) {
        return qsa_gathered_wmma(gpu, p, end_position, max_selected, arena);
    }
    // Live (unrecorded) launches reserve LDS for this chunk's longest row
    // only; a recorded launch keeps the position-independent shape bound.
    let shape_selected = if !qsa_decode_reference_eager(gpu) {
        p.shape_selected
    } else {
        max_selected
    };
    let rows = checked_i32(run_rows, "QSA batch attention rows")?;
    let position_start = checked_i32(p.position_start + first, "QSA batch attention position")?;
    let n_heads = checked_i32(p.n_heads, "QSA batch attention heads")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA batch attention KV heads")?;
    let head_dim = checked_i32(p.head_dim, "QSA batch attention head width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA batch attention budget")?;
    let compress = checked_i32(p.compress, "QSA batch attention compress")?;
    let capacity = checked_i32(p.capacity, "QSA batch attention capacity")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA batch attention cache capacity")?;
    let row_grid = checked_u32(run_rows, "QSA batch attention row grid")?;
    let hg12 = allow_fast
        && !decode
        && p.format == QsaKvFormat::F32
        && gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
        && p.head_dim == 256
        && p.n_heads == p.n_kv_heads * QSA_ATTENTION_HG12_HEADS
        && gpu.route_rows(p.rows) >= QSA_ATTENTION_HG12_MIN_ROWS
        && shape_selected <= QSA_ATTENTION_HG12_MAX_SELECTED;
    let hg4_bytes = (allow_fast && !decode && !hg12)
        .then(|| qsa_attention_hg4_lds_bytes(gpu, p, shape_selected))
        .flatten();
    let (kernel_name, grid, shared_mem) = if hg12 {
        (
            "indexed_attention_attention_f32_batched_hg12",
            [
                checked_u32(p.n_kv_heads, "QSA batch attention KV head grid")?,
                1,
                row_grid,
            ],
            0,
        )
    } else if let Some(bytes) = hg4_bytes {
        (
            if qsa_attention_hg4_lite(&gpu.flags, p.format) {
                "indexed_attention_attention_q8_batched_hg4_lite"
            } else {
                p.format.kernel([
                    "indexed_attention_attention_f32_batched_hg4",
                    "indexed_attention_attention_fp8_batched_hg4",
                    "indexed_attention_attention_q8_batched_hg4",
                    "indexed_attention_attention_fp8_fnuz_batched_hg4",
                ])
            },
            [
                checked_u32(
                    p.n_heads / QSA_ATTENTION_HG4_HEADS,
                    "QSA batch attention head grid",
                )?,
                1,
                row_grid,
            ],
            bytes,
        )
    } else {
        let head_grid = checked_u32(p.n_heads, "QSA batch attention head grid")?;
        let dim_grid = blocks(p.head_dim)?;
        match shape_selected.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
                    && shape_selected > 0
                    // The kernel adds 32 bytes of static LDS (per-wave maxes).
                    && bytes + 32 <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
                    && bytes <= u32::MAX as usize =>
            {
                (
                    p.format.kernel([
                        "indexed_attention_attention_f32_batched",
                        "indexed_attention_attention_fp8_batched",
                        "indexed_attention_attention_q8_batched",
                        "indexed_attention_attention_fp8_fnuz_batched",
                    ]),
                    [head_grid, dim_grid, row_grid],
                    bytes as u32,
                )
            }
            _ => (
                p.format.kernel([
                    "indexed_attention_attention_f32_batched_serial",
                    "indexed_attention_attention_fp8_batched_serial",
                    "indexed_attention_attention_q8_batched_serial",
                    "indexed_attention_attention_fp8_fnuz_batched_serial",
                ]),
                [head_grid, dim_grid, row_grid],
                0,
            ),
        }
    };
    let block = [QSA_ATTENTION_PARALLEL_THREADS, 1, 1];
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for ptr in [
        qsa_attention_row_ptr(p.q_with_gate, first, 2 * p.n_heads * p.head_dim * 4),
        p.full_keys.buf.as_ptr(),
        p.full_values.buf.as_ptr(),
        qsa_attention_row_ptr(p.selected, first, p.capacity * 4),
        qsa_attention_row_ptr(p.output, first, p.n_heads * p.head_dim * 4),
    ] {
        args.push_ptr(ptr);
    }
    args.push_i32(rows);
    args.push_i32(position_start);
    // Declared dynamic field: the chunk's start position. Replay re-derives it,
    // so the attention window follows the replay position.
    let position_offset = args.len() - 4;
    for value in [
        n_heads,
        n_kv_heads,
        head_dim,
        budget_blocks,
        compress,
        capacity,
        full_capacity,
    ] {
        args.push_i32(value);
    }
    args.pad_to(16);
    let position_binding = [crate::replay::ReplayKernargBinding::PositionPlusU32 {
        offset: position_offset,
        addend: first as u32,
    }];
    // Guarded vs grid-rounded, all four kernels (per-head parallel/serial, hg4,
    // hg12): the row grid is exact; the per-head kernels' `dim_grid * 256`
    // channel rounding is guarded by `channel >= head_dim` (hg4/hg12 require
    // head_dim == 256 = block size). Every kernel validates
    // `token >= 0 && token < full_capacity && token < visible` with
    // `visible = position_start + row + 1 <= end_position`, so indirect K/V
    // reads (and the value reads of invalid entries, which redirect to row 0)
    // stay below `min(end_position, full_capacity)` cache rows whatever the
    // selected count or contents. The selected list is read below each row's
    // own `selected_len` (<= capacity); Q is read whole per (row, head) and
    // output is written for every `rows * n_heads * head_dim` element.
    let extents = gpu
        .flags
        .qwen4_qsa_extent_check
        .then(|| QsaBatchExtents::new(p, end_position));
    if let Some(e) = &extents {
        e.check_query(gpu, p, kernel_name);
        e.check_kv(gpu, p, kernel_name);
        e.check_selected(gpu, p, kernel_name);
        e.check_output(gpu, p, kernel_name);
    }
    gpu.launch_blob_recorded(
        kernel_name,
        grid,
        block,
        shared_mem,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings {
            grid: None,
            kernargs: &position_binding,
        },
    )?;
    if let Some(e) = &extents {
        e.post(gpu, p, kernel_name);
    }
    Ok(())
}

/// Split count of the production QSA split-K decode route
/// (`HIPFIRE_QWEN4_QSA_DECODE_SPLITK`, default on): 32. Chosen from an R9700 (gfx1201)
/// sweep as the lowest summed time over the 1131- and 8188-token cases (79.04 us),
/// pending Strix Halo KLD and performance. A compromise: S16 is fastest at
/// 1131 tokens (22.46 us vs 24.71 us at S32), S32 fastest at 8188 (54.33 us vs
/// 69.37 us at S16). The kernels accept 1..=48 (`QSA_DECODE_SPLITK_MAX_SPLITS`).
pub const QSA_DECODE_SPLITK_SPLITS: usize = 32;
/// Scoring layout of the production split-K route: 2 = KV-group layout, one
/// workgroup per KV head scoring its 12 query heads from a single K/V read
/// (needs `n_heads == 12 * n_kv_heads`; other shapes keep the ordinary
/// routes). 0 = one thread per token with the incumbent's serial dot order,
/// 1 = eight wave32 token groups with a shuffle-reduced dot, both reachable
/// through the direct test entry. Chosen from the same R9700 sweep as
/// `QSA_DECODE_SPLITK_SPLITS`, pending Strix Halo KLD and performance.
pub const QSA_DECODE_SPLITK_SCORING_LAYOUT: usize = 2;
/// Rows and splits the fixed workspace reserve covers (`MAX_SPLITS` is the
/// tuning ceiling; the workspace is reserved at this bound, and a launch maps
/// only the bytes it touches). `MAX_ROWS` is the widest exact verify block
/// (`WIDE_VERIFY_MAX_ROWS`); admission is `few_row_limit` alone (it never
/// exceeds this bound, and a larger launch would only raise the mapped bytes).
const QSA_DECODE_SPLITK_RESERVE_ROWS: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
const QSA_DECODE_SPLITK_MAX_SPLITS: usize = 48;
/// Highest scoring layout the partial kernel implements: 0 = one thread per
/// token, 1 = eight wave32 token groups per head, 2 = KV-group layout (one
/// workgroup per KV head scores its `QSA_ATTENTION_HG12_HEADS` query heads
/// from a single K/V read).
const QSA_DECODE_SPLITK_MAX_LAYOUT: usize = 2;
/// LDS bytes per slice entry: layouts 0/1 hold one F32 score and one i32
/// token (8 B); the KV-group layout holds 12 head scores and the token (52 B).
const QSA_DECODE_SPLITK_GROUP_LDS_BYTES_PER_ROW: usize = 52;
/// Static LDS budget reserved for `qsa_decode_splitk_partial_f32` beside its
/// dynamic slice: 416 B = the group layout's 384 B plus the per-head layout's
/// 32 B. Every layout is budgeted at this bound (the compiler may overlap the
/// disjoint arrays, so the true static size is at most this).
const QSA_DECODE_SPLITK_STATIC_LDS_BYTES: usize = 416;
const QSA_DECODE_SPLITK_MODULE: &str = "qsa_decode_splitk";
const QSA_DECODE_SPLITK_PARTIAL: &str = "qsa_decode_splitk_partial_f32";
const QSA_DECODE_SPLITK_COMBINE: &str = "qsa_decode_splitk_combine_f32";

/// The split-K route is eager only: its host-computed grid, LDS and position
/// scalars are baked into the launch, so no recorder or capture may be armed.
fn qsa_decode_splitk_eager(gpu: &Gpu) -> bool {
    !gpu.replay.is_recording() && !gpu.graphs.capture_mode
}

/// Dynamic LDS of the partial kernel for `scoring_layout`: layouts 0 and 1
/// take 8 bytes per entry of the longest slice, `ceil(max_selected / splits)`;
/// layout 2 takes 52. `None` when `splits` or the layout is out of range or the
/// slice plus the static budget does not fit the LDS limit.
fn qsa_decode_splitk_lds_bytes(
    max_selected: usize,
    splits: usize,
    scoring_layout: usize,
) -> Option<u32> {
    if splits == 0
        || splits > QSA_DECODE_SPLITK_MAX_SPLITS
        || scoring_layout > QSA_DECODE_SPLITK_MAX_LAYOUT
    {
        return None;
    }
    let per_row = if scoring_layout == 2 {
        QSA_DECODE_SPLITK_GROUP_LDS_BYTES_PER_ROW
    } else {
        QSA_ATTENTION_LDS_BYTES_PER_ROW
    };
    let bytes = max_selected.div_ceil(splits).checked_mul(per_row)?;
    (bytes.checked_add(QSA_DECODE_SPLITK_STATIC_LDS_BYTES)?
        <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES)
        .then_some(bytes as u32)
}

/// Whether `scoring_layout` fits the shape: layout 2 scores whole KV groups of
/// `QSA_ATTENTION_HG12_HEADS` query heads at head_dim 256; layouts 0 and 1 take
/// any grouping.
fn qsa_decode_splitk_layout_fits(
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    scoring_layout: usize,
) -> bool {
    scoring_layout <= QSA_DECODE_SPLITK_MAX_LAYOUT
        && (scoring_layout < 2
            || (head_dim == 256
                && n_kv_heads.checked_mul(QSA_ATTENTION_HG12_HEADS) == Some(n_heads)))
}

/// Whether an `rows`-row QSA attention call is an admitted decode call: `1..=`
/// the shared [`Gpu::few_row_limit`] (8, or the exact-wide verify rows, eager
/// only), both as launched and as routed (`route_rows`; the prefill invariant
/// profile routes at the canonical chunk, far above any limit). This is the
/// ONLY row-count term of [`qsa_decode_attention_policy`].
fn qsa_decode_rows_admitted(gpu: &Gpu, rows: usize) -> bool {
    let limit = gpu.few_row_limit();
    (1..=limit).contains(&rows) && gpu.route_rows(rows) <= limit
}

/// Attention implementation of an admitted QSA decode call (1..=`few_row_limit`
/// rows): the SAME choice for every row count, so n rows reproduce n one-row
/// decodes of the same architecture, format, flags and launch mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QsaDecodeAttentionPolicy {
    /// `qsa_decode_splitk_{partial,combine}_f32`; every row partitions its own
    /// selected length `ell_r` into `splits` slices.
    SplitK { splits: usize, layout: usize },
    /// `qsa_decode_splitkv_{partial_q8,merge_f32}`; every row splits its own
    /// selected list into `splits` slices. q8 K/V, 24 query heads over 2 KV
    /// heads, head_dim 256; the LDS is fixed, so no `max_selected` term.
    SplitKvQ8 { splits: usize },
    /// The one-row per-head kernels (`indexed_attention_attention_*_batched`
    /// / `_serial`), grid z = rows. Never hg4, hg12 or WMMA.
    PerHead,
}

/// The policy proper. Inputs only: the architecture and flag (F32 split-K:
/// gfx1151; q8 split-KV: gfx1151 or gfx1100; `HIPFIRE_QWEN4_QSA_DECODE_SPLITK`), K/V format, head_dim, head grouping, the
/// context-derived `max_selected` (LDS fit of the F32 slice; the q8 split-KV
/// LDS is fixed) and the launch mode (`eager`: no recorder or capture armed;
/// both split routes bake their grid, LDS and position into the launch and
/// are eager only). No row count.
fn qsa_decode_attention_policy_for(
    gpu: &Gpu,
    format: QsaKvFormat,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_selected: usize,
    eager: bool,
) -> QsaDecodeAttentionPolicy {
    qsa_decode_attention_policy_with(
        &gpu.flags,
        &gpu.arch_caps,
        format,
        head_dim,
        n_heads,
        n_kv_heads,
        max_selected,
        eager,
    )
}

/// [`qsa_decode_attention_policy_for`] over the flags and architecture alone
/// (no device), so the policy is unit-testable without a GPU.
fn qsa_decode_attention_policy_with(
    flags: &crate::feature_flags::FeatureFlags,
    arch_caps: &crate::arch_caps::ArchCaps,
    format: QsaKvFormat,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_selected: usize,
    eager: bool,
) -> QsaDecodeAttentionPolicy {
    let split_k = eager
        && flags.qwen4_qsa_decode_splitk_enabled()
        && arch_caps.is_gfx1151()
        && format == QsaKvFormat::F32
        && head_dim == 256
        && qsa_decode_splitk_layout_fits(
            head_dim,
            n_heads,
            n_kv_heads,
            QSA_DECODE_SPLITK_SCORING_LAYOUT,
        )
        && qsa_decode_splitk_lds_bytes(
            max_selected,
            QSA_DECODE_SPLITK_SPLITS,
            QSA_DECODE_SPLITK_SCORING_LAYOUT,
        )
        .is_some();
    if split_k {
        return QsaDecodeAttentionPolicy::SplitK {
            splits: QSA_DECODE_SPLITK_SPLITS,
            layout: QSA_DECODE_SPLITK_SCORING_LAYOUT,
        };
    }
    let split_kv_q8 = eager
        && flags.qwen4_qsa_decode_splitkv_q8_enabled()
        && (arch_caps.is_gfx1151() || arch_caps.is_gfx1100())
        && qsa_decode_splitkv_q8_geometry_fits(format, head_dim, n_heads, n_kv_heads);
    if split_kv_q8 {
        QsaDecodeAttentionPolicy::SplitKvQ8 {
            splits: QSA_DECODE_SPLITK_SPLITS,
        }
    } else {
        QsaDecodeAttentionPolicy::PerHead
    }
}

/// Raw byte offsets: selected is a raw i32 array, not an F32 tensor.
fn qsa_attention_row_ptr(
    tensor: &GpuTensor,
    first: usize,
    row_bytes: usize,
) -> *mut std::ffi::c_void {
    // Extent products and accessible rows were checked before run formation.
    unsafe {
        (tensor.buf.as_ptr() as *mut u8)
            .add(first * row_bytes)
            .cast()
    }
}

fn qsa_decode_reference_eager(gpu: &Gpu) -> bool {
    // Binding arithmetic must never make a real recording/capture split-safe.
    qsa_decode_splitk_eager(gpu)
        && !matches!(
            gpu.qwen4_verify_reference(),
            Some(crate::prefill_invariant::Qwen4VerifyReference::Recorded)
        )
}

fn qsa_decode_attention_policy_for_row(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    row: usize,
) -> QsaDecodeAttentionPolicy {
    qsa_decode_attention_policy_for(
        gpu,
        p.format,
        p.head_dim,
        p.n_heads,
        p.n_kv_heads,
        qsa_attention_selected_at(p, row),
        qsa_decode_reference_eager(gpu),
    )
}

/// Preserve singleton parallel/serial per-head choice across its LDS boundary.
fn qsa_decode_per_head_parallel(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    row: usize,
) -> bool {
    let selected = if qsa_decode_reference_eager(gpu) {
        qsa_attention_selected_at(p, row)
    } else {
        p.shape_selected
    };
    gpu.arch_caps.has_qwen4_parallel_qsa_gdn()
        && selected > 0
        && selected
            .checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW)
            .and_then(|bytes| bytes.checked_add(32))
            .is_some_and(|bytes| bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES)
}

/// Metadata-only policy hook for component qualification receipts.
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn qsa_decode_attention_policy_for_row_for_test(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    row: usize,
) -> Option<QsaDecodeAttentionPolicy> {
    (row < p.rows && qsa_decode_rows_admitted(gpu, p.rows))
        .then(|| qsa_decode_attention_policy_for_row(gpu, p, row))
}

/// Saturating twin of the selected extent validated by the production entry.
fn qsa_attention_selected_at(p: &IndexedAttentionAttentionBatch<'_>, row: usize) -> usize {
    let bound = p
        .budget_blocks
        .saturating_mul(p.compress)
        .saturating_add(p.compress.saturating_sub(1));
    p.position_start
        .saturating_add(row.saturating_add(1))
        .min(p.capacity)
        .min(bound)
}

/// Attention policy of `p` under the current launch mode: `Some` for an
/// admitted decode call (`1..=few_row_limit` rows; see
/// [`qsa_decode_rows_admitted`]), which MUST take exactly the returned
/// implementation - no hg4 / hg12 / dense or gathered WMMA, whatever the row
/// count - so the verify block is the n-fold one-row decode. `None`: not a
/// decode call (more rows than the limit, prefill profile); the row-routed
/// prefill and few-row selection applies.
fn qsa_decode_attention_policy(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
) -> Option<QsaDecodeAttentionPolicy> {
    qsa_decode_rows_admitted(gpu, p.rows).then(|| qsa_decode_attention_policy_for_row(gpu, p, 0))
}

/// Whether the eager decode policy differs from the recorded / captured one
/// for this shape on `gpu` (all else equal, fast routes allowed, rows
/// irrelevant). True means AR decode in a replay or graph and eager verify
/// are different attention arithmetic. Reads the flags and architecture only,
/// not the Gpu's current launch mode.
pub fn qsa_decode_attention_eager_differs_from_recorded(
    gpu: &Gpu,
    format: QsaKvFormat,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_selected: usize,
) -> bool {
    qsa_decode_attention_policy_for(
        gpu,
        format,
        head_dim,
        n_heads,
        n_kv_heads,
        max_selected,
        true,
    ) != qsa_decode_attention_policy_for(
        gpu,
        format,
        head_dim,
        n_heads,
        n_kv_heads,
        max_selected,
        false,
    )
}

/// The split-K partial + combine launches. `p` is already shape-validated
/// (the production arena checks or `qsa_decode_splitk_validate`);
/// `max_selected` is the chunk's longest selected list. The F32 partials live
/// in the QSA per-launch scratch ([`Gpu::qsa_gather_scratch`], not the F16
/// arena), reserved once at the fixed bound of `QSA_DECODE_SPLITK_RESERVE_ROWS`
/// rows x `QSA_DECODE_SPLITK_MAX_SPLITS` splits and kept if a larger buffer
/// is already held; only the launch's actual `rows x heads x splits` records
/// are mapped. Every row partitions its own selected length (the kernels
/// derive `ell_r` from `position_start + row`); both kernels run on the same
/// stream back to back.
#[cfg(any(test, feature = "lab"))]
fn qsa_decode_splitk_launch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    max_selected: usize,
    splits: usize,
    scoring_layout: usize,
) -> HipResult<()> {
    qsa_decode_splitk_launch_run(gpu, p, max_selected, splits, scoring_layout, 0, p.rows)
}

fn qsa_decode_splitk_launch_run(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    max_selected: usize,
    splits: usize,
    scoring_layout: usize,
    first: usize,
    run_rows: usize,
) -> HipResult<()> {
    if splits == 0
        || splits > QSA_DECODE_SPLITK_MAX_SPLITS
        || !qsa_decode_splitk_layout_fits(p.head_dim, p.n_heads, p.n_kv_heads, scoring_layout)
        || run_rows == 0
        || run_rows > gpu.few_row_limit()
        || p.head_dim != 256
        || p.format != QsaKvFormat::F32
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if !qsa_decode_splitk_eager(gpu) {
        return Err(HipError::new(0, "QSA split-K decode is eager-only"));
    }
    let lds = qsa_decode_splitk_lds_bytes(max_selected, splits, scoring_layout)
        .ok_or_else(|| HipError::new(0, "QSA split-K slice exceeds the dynamic LDS budget"))?;
    // The kernels index with 32-bit ints: row * capacity (selected), and
    // (row * heads) * splits (workspace record).
    checked_product(p.rows, p.capacity, "QSA split-K selected rows")?;
    checked_product3(p.rows, p.n_heads, splits, "QSA split-K workspace records")?;
    let rows = checked_i32(run_rows, "QSA split-K rows")?;
    let position_start = checked_i32(p.position_start + first, "QSA split-K position")?;
    let n_heads = checked_i32(p.n_heads, "QSA split-K heads")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA split-K KV heads")?;
    let head_dim = checked_i32(p.head_dim, "QSA split-K head width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA split-K budget")?;
    let compress = checked_i32(p.compress, "QSA split-K compress")?;
    let capacity = checked_i32(p.capacity, "QSA split-K capacity")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA split-K cache capacity")?;
    let splits_i = checked_i32(splits, "QSA split-K splits")?;
    let layout_i = checked_i32(scoring_layout, "QSA split-K scoring layout")?;
    // Layout 2 runs one workgroup per KV head, the others one per query head.
    let head_grid = checked_u32(
        if scoring_layout == 2 {
            p.n_kv_heads
        } else {
            p.n_heads
        },
        "QSA split-K head grid",
    )?;
    let combine_head_grid = checked_u32(p.n_heads, "QSA split-K combine head grid")?;
    let split_grid = checked_u32(splits, "QSA split-K split grid")?;
    let row_grid = checked_u32(run_rows, "QSA split-K row grid")?;
    // Actual demand: rows x n_heads x splits records of (head_dim + 2) F32
    // (m, l, acc[head_dim]); the reservation is the fixed max-rows x
    // max-splits bound, so one slot serves every admitted row count and a
    // wider launch never moves it.
    let record_bytes = |rows: usize, splits: usize| {
        rows.checked_mul(p.n_heads)
            .and_then(|value| value.checked_mul(splits))
            .and_then(|value| value.checked_mul(p.head_dim + 2))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| HipError::new(0, "QSA split-K workspace size overflows"))
    };
    let workspace_bytes = record_bytes(run_rows, splits)?;
    let reserve_bytes = record_bytes(QSA_DECODE_SPLITK_RESERVE_ROWS, QSA_DECODE_SPLITK_MAX_SPLITS)?;
    for kernel in [QSA_DECODE_SPLITK_PARTIAL, QSA_DECODE_SPLITK_COMBINE] {
        gpu.ensure_kernel_public(
            QSA_DECODE_SPLITK_MODULE,
            crate::kernels::QSA_DECODE_SPLITK_SRC,
            kernel,
        )?;
    }
    let workspace = gpu.qsa_gather_scratch(workspace_bytes, reserve_bytes)?;
    let mut partial = KernargBlob::new();
    for ptr in [
        qsa_attention_row_ptr(p.q_with_gate, first, 2 * p.n_heads * p.head_dim * 4),
        p.full_keys.buf.as_ptr(),
        p.full_values.buf.as_ptr(),
        qsa_attention_row_ptr(p.selected, first, p.capacity * 4),
        workspace,
    ] {
        partial.push_ptr(ptr);
    }
    for value in [
        rows,
        position_start,
        n_heads,
        n_kv_heads,
        head_dim,
        budget_blocks,
        compress,
        capacity,
        full_capacity,
        splits_i,
        layout_i,
    ] {
        partial.push_i32(value);
    }
    partial.pad_to(16);
    let mut combine = KernargBlob::new();
    combine.push_ptr(qsa_attention_row_ptr(
        p.q_with_gate,
        first,
        2 * p.n_heads * p.head_dim * 4,
    ));
    combine.push_ptr(workspace);
    combine.push_ptr(qsa_attention_row_ptr(
        p.output,
        first,
        p.n_heads * p.head_dim * 4,
    ));
    for value in [rows, n_heads, head_dim, splits_i] {
        combine.push_i32(value);
    }
    combine.pad_to(16);
    // Partial kernel, guarded vs grid-rounded: grid `[heads or kv heads,
    // splits, rows]`, every dimension exact (the kernels return on
    // `row >= rows` / `head >= n_heads` / `split >= splits`). Each split reads
    // its slice `[begin, end)` of the row's selected list (below the row's
    // `len`); tokens are validated `< full_capacity && < visible`
    // (visible = position_start + row + 1 <= end_position), so K/V reads are
    // bounded by the visible end `min(end_position, full_capacity)` rows, not
    // by the selected count. Every split writes its whole 258-float record
    // (`head_dim + 2`; empty slices write the sentinel record too), so the
    // workspace write extent is `rows * n_heads * splits` records, not the
    // fixed-bound reservation. Combine reads exactly those records.
    let splitk = gpu.flags.qwen4_qsa_extent_check.then(|| {
        let ws_len = run_rows
            .saturating_mul(p.n_heads)
            .saturating_mul(splits)
            .saturating_mul(p.head_dim + 2)
            .saturating_mul(std::mem::size_of::<f32>());
        (
            QsaBatchExtents::new(p, p.position_start.saturating_add(p.rows)),
            ws_len,
        )
    });
    if let Some((e, ws_len)) = &splitk {
        e.check_query(gpu, p, QSA_DECODE_SPLITK_PARTIAL);
        e.check_kv(gpu, p, QSA_DECODE_SPLITK_PARTIAL);
        e.check_selected(gpu, p, QSA_DECODE_SPLITK_PARTIAL);
        gpu.qsa_extent_check_ptr(
            "qsa_splitk_workspace",
            workspace,
            0,
            *ws_len,
            e.depth,
            QSA_DECODE_SPLITK_PARTIAL,
        );
    }
    gpu.launch_blob_recorded(
        QSA_DECODE_SPLITK_PARTIAL,
        [head_grid, split_grid, row_grid],
        [QSA_ATTENTION_PARALLEL_THREADS, 1, 1],
        lds,
        partial.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some((e, ws_len)) = &splitk {
        gpu.qsa_extent_post_launch_ptr(
            "qsa_splitk_workspace",
            workspace,
            0,
            *ws_len,
            e.depth,
            QSA_DECODE_SPLITK_PARTIAL,
        );
    }
    // Combine: grid `[n_heads, 1, rows]`, `channel = threadIdx.x < head_dim`
    // (256 = head_dim, exact). Reads the query gate (within the row's `q`
    // span), the `splits` records of each (row, head), writes the output row.
    if let Some((e, ws_len)) = &splitk {
        e.check_query(gpu, p, QSA_DECODE_SPLITK_COMBINE);
        gpu.qsa_extent_check_ptr(
            "qsa_splitk_workspace",
            workspace,
            0,
            *ws_len,
            e.depth,
            QSA_DECODE_SPLITK_COMBINE,
        );
        e.check_output(gpu, p, QSA_DECODE_SPLITK_COMBINE);
    }
    gpu.launch_blob_recorded(
        QSA_DECODE_SPLITK_COMBINE,
        [combine_head_grid, 1, row_grid],
        [QSA_ATTENTION_PARALLEL_THREADS, 1, 1],
        0,
        combine.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some((e, _)) = &splitk {
        e.post(gpu, p, QSA_DECODE_SPLITK_COMBINE);
    }
    Ok(())
}

/// Every shape, type and scalar check of the direct split-K entry; returns
/// the chunk's longest selected length. Stricter than the production route's
/// inline checks: all tensors' dtypes and byte extents, the full cache
/// capacities and lengths, the 1..=few_row_limit (<=64) row and 1..48 split limits, layouts 0..2
/// (layout 2 needs `n_heads == 12 * n_kv_heads`), a gfx11/gfx12
/// wave32 device, and no recorder or capture.
#[cfg(any(test, feature = "lab"))]
fn qsa_decode_splitk_validate(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    splits: usize,
    scoring_layout: usize,
) -> HipResult<usize> {
    let wrong_shape = || HipError::new(0, &ComputeError::WrongShape.to_string());
    if !gpu.arch_caps.has_gfx11_plus_simt() {
        return Err(HipError::new(
            0,
            "QSA split-K decode needs a gfx11 or gfx12 wave32 device",
        ));
    }
    if !qsa_decode_splitk_eager(gpu) {
        return Err(HipError::new(0, "QSA split-K decode is eager-only"));
    }
    if splits == 0
        || splits > QSA_DECODE_SPLITK_MAX_SPLITS
        || !qsa_decode_splitk_layout_fits(p.head_dim, p.n_heads, p.n_kv_heads, scoring_layout)
        || p.rows == 0
        || p.rows > gpu.few_row_limit()
        || p.format != QsaKvFormat::F32
        || p.head_dim != 256
    {
        return Err(wrong_shape());
    }
    for tensor in [p.q_with_gate, p.output] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.n_kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    if p.selected.dtype != DType::Raw
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.n_heads % p.n_kv_heads != 0
        || p.compress == 0
        || p.capacity == 0
        || p.full_capacity == 0
    {
        return Err(wrong_shape());
    }
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(wrong_shape)?;
    if end_position > i32::MAX as usize || end_position > p.full_capacity {
        return Err(wrong_shape());
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA split-K heads")?;
    let q_elements = checked_product(2, head_elements, "QSA split-K query")?;
    let kv_elements = checked_product(
        p.full_capacity,
        p.format.kv_row_units(p.n_kv_heads, p.head_dim),
        "QSA split-K cache",
    )?;
    let q_rows = checked_product(p.rows, q_elements, "QSA split-K query rows")?;
    let output_rows = checked_product(p.rows, head_elements, "QSA split-K output rows")?;
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA split-K selected rows")?,
        std::mem::size_of::<i32>(),
        "QSA split-K selected",
    )?;
    let f32_bytes = std::mem::size_of::<f32>();
    let kv_bytes = checked_product(kv_elements, f32_bytes, "QSA split-K cache bytes")?;
    if p.q_with_gate.numel() < q_rows
        || p.output.numel() < output_rows
        || p.full_keys.numel() < kv_elements
        || p.full_values.numel() < kv_elements
        || p.selected.numel() < selected_bytes
        || p.q_with_gate.buf.size() < checked_product(q_rows, f32_bytes, "QSA split-K query bytes")?
        || p.output.buf.size()
            < checked_product(output_rows, f32_bytes, "QSA split-K output bytes")?
        || p.full_keys.buf.size() < kv_bytes
        || p.full_values.buf.size() < kv_bytes
        || p.selected.buf.size() < selected_bytes
    {
        return Err(wrong_shape());
    }
    let selected_bound =
        checked_product(p.budget_blocks, p.compress, "QSA split-K selected bound")?
            .checked_add(p.compress - 1)
            .ok_or_else(wrong_shape)?;
    let max_selected = end_position.min(p.capacity).min(selected_bound);
    if p.shape_selected < max_selected {
        return Err(HipError::new(
            0,
            &format!(
                "QSA split-K shape bound {} is below the active selected length {max_selected}",
                p.shape_selected
            ),
        ));
    }
    Ok(max_selected)
}

/// Direct test route to the split-K partial + combine kernels with an explicit
/// `splits` (1..=48) and `scoring_layout` (0, 1, or the KV-group layout 2), bypassing the knob and the
/// route predicates: the full validation of [`qsa_decode_splitk_validate`], then
/// the production launch pair. F32 K/V, head_dim 256, 1..=few_row_limit (<=64) rows, eager only;
/// gfx11/gfx12 wave32. Gated behind `lab` (or test) like
/// [`indexed_attention_attention_batch_exact`].
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn indexed_attention_attention_splitk_for_test(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    splits: usize,
    scoring_layout: usize,
) -> HipResult<()> {
    let max_selected = qsa_decode_splitk_validate(gpu, p, splits, scoring_layout)?;
    qsa_decode_splitk_launch(gpu, p, max_selected, splits, scoring_layout)
}

/// Splits, shape and geometry of the q8 split-KV decode route (the measured
/// f1771197a kernels, `qsa_decode_splitkv_q8.hip`): the grid is one workgroup
/// of 384 threads (one wave32 per query head of a KV group) per (KV head,
/// split, row), so the kernels implement exactly 24 query heads over 2 KV
/// heads at head_dim 256 and at most 32 splits. The route splits
/// `QSA_DECODE_SPLITK_SPLITS` (32) ways.
const QSA_DECODE_SPLITKV_Q8_MAX_SPLITS: usize = 32;
const QSA_DECODE_SPLITKV_Q8_HEADS: usize = 24;
const QSA_DECODE_SPLITKV_Q8_KV_HEADS: usize = 2;
const QSA_DECODE_SPLITKV_Q8_HEAD_DIM: usize = 256;
/// Partial kernel block: one wave32 per query head of a KV group (12 waves).
const QSA_DECODE_SPLITKV_Q8_PARTIAL_THREADS: u32 = 384;
/// Merge kernel block: eight wave32, one (row, head) record per wave.
const QSA_DECODE_SPLITKV_Q8_MERGE_THREADS: u32 = 256;
const QSA_DECODE_SPLITKV_Q8_MERGE_RECORDS_PER_BLOCK: usize = 8;
/// Fixed dynamic LDS of the partial kernel (no static LDS): one 32 KiB F32
/// K/V bank (64 keys x 128 channels), 12 heads x 64 key weights and 64 token
/// ids = 32768 + 3072 + 256 = 36096 bytes.
const QSA_DECODE_SPLITKV_Q8_LDS_BYTES: usize =
    64 * 128 * 4 + QSA_ATTENTION_HG12_HEADS * 64 * 4 + 64 * 4;
const _: () = assert!(
    QSA_DECODE_SPLITKV_Q8_LDS_BYTES == 36096
        && QSA_DECODE_SPLITKV_Q8_LDS_BYTES <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
);
/// Workspace record: `rows * n_heads * splits` product-LSE records, an F32 LSE
/// plane then `head_dim` normalized F32 O values per record (257 F32 per
/// record). The kernel's record ABI differs from the F32 split-K scratch
/// layout, so the q8 route never reads or writes that layout; both share the
/// one QSA per-launch scratch allocation. The reservation bound is
/// `QSA_DECODE_SPLITK_RESERVE_ROWS` rows x 24 heads x
/// `QSA_DECODE_SPLITKV_Q8_MAX_SPLITS` splits.
const QSA_DECODE_SPLITKV_Q8_RECORD_BYTES: usize =
    (1 + QSA_DECODE_SPLITKV_Q8_HEAD_DIM) * std::mem::size_of::<f32>();
const QSA_DECODE_SPLITKV_Q8_MODULE: &str = "qsa_decode_splitkv_q8";
const QSA_DECODE_SPLITKV_Q8_PARTIAL: &str = "qsa_decode_splitkv_partial_q8";
const QSA_DECODE_SPLITKV_Q8_MERGE: &str = "qsa_decode_splitkv_merge_f32";

/// Whether a call of this format and geometry is one the q8 split-KV kernels
/// implement: q8 K/V, head_dim 256, 24 query heads over exactly 2 KV heads.
fn qsa_decode_splitkv_q8_geometry_fits(
    format: QsaKvFormat,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
) -> bool {
    format == QsaKvFormat::Q8
        && head_dim == QSA_DECODE_SPLITKV_Q8_HEAD_DIM
        && n_kv_heads == QSA_DECODE_SPLITKV_Q8_KV_HEADS
        && n_heads == QSA_DECODE_SPLITKV_Q8_HEADS
}

/// [`qsa_decode_splitkv_q8_geometry_fits`] for `p`.
fn qsa_decode_splitkv_q8_shape_fits(p: &IndexedAttentionAttentionBatch<'_>) -> bool {
    qsa_decode_splitkv_q8_geometry_fits(p.format, p.head_dim, p.n_heads, p.n_kv_heads)
}

/// The q8 split-KV partial + merge launches. `p` is already shape-validated
/// (the production arena checks or `qsa_decode_splitkv_q8_validate`). The F32
/// product-LSE records (`records = rows * n_heads * splits`, record
/// `(row * n_heads + head) * splits + split`; LSE plane, then normalized O)
/// live in the QSA per-launch scratch ([`Gpu::qsa_gather_scratch`], not the
/// F16 arena), reserved once at the fixed bound of
/// `QSA_DECODE_SPLITK_RESERVE_ROWS` rows x 24 heads x
/// `QSA_DECODE_SPLITKV_Q8_MAX_SPLITS` splits x 257 F32 and kept if a larger
/// buffer is already held (the F32 split-K reservation shares the slot); only
/// the launch's actual `rows x heads x splits` records are mapped. Every row
/// splits its own selected list; both kernels run on the same stream back to
/// back.
#[cfg(any(test, feature = "lab"))]
fn qsa_decode_splitkv_q8_launch(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    splits: usize,
) -> HipResult<()> {
    qsa_decode_splitkv_q8_launch_run(gpu, p, splits, 0, p.rows)
}

fn qsa_decode_splitkv_q8_launch_run(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    splits: usize,
    first: usize,
    run_rows: usize,
) -> HipResult<()> {
    if splits == 0
        || splits > QSA_DECODE_SPLITKV_Q8_MAX_SPLITS
        || run_rows == 0
        || run_rows > QSA_DECODE_SPLITK_RESERVE_ROWS
        || !qsa_decode_splitkv_q8_shape_fits(p)
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if !qsa_decode_splitk_eager(gpu) {
        return Err(HipError::new(0, "QSA q8 split-KV decode is eager-only"));
    }
    // The kernels index with 32-bit ints: row * capacity (selected), and
    // (row * heads) * splits (workspace record).
    checked_product(p.rows, p.capacity, "QSA q8 split-KV selected rows")?;
    checked_product3(
        p.rows,
        p.n_heads,
        splits,
        "QSA q8 split-KV workspace records",
    )?;
    let rows = checked_i32(run_rows, "QSA q8 split-KV rows")?;
    let position_start = checked_i32(p.position_start + first, "QSA q8 split-KV position")?;
    let n_heads = checked_i32(p.n_heads, "QSA q8 split-KV heads")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA q8 split-KV KV heads")?;
    let head_dim = checked_i32(p.head_dim, "QSA q8 split-KV head width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA q8 split-KV budget")?;
    let compress = checked_i32(p.compress, "QSA q8 split-KV compress")?;
    let capacity = checked_i32(p.capacity, "QSA q8 split-KV capacity")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA q8 split-KV cache capacity")?;
    let splits_i = checked_i32(splits, "QSA q8 split-KV splits")?;
    let kv_grid = checked_u32(p.n_kv_heads, "QSA q8 split-KV KV head grid")?;
    let split_grid = checked_u32(splits, "QSA q8 split-KV split grid")?;
    let row_grid = checked_u32(run_rows, "QSA q8 split-KV row grid")?;
    let merge_grid = checked_u32(
        (run_rows * p.n_heads).div_ceil(QSA_DECODE_SPLITKV_Q8_MERGE_RECORDS_PER_BLOCK),
        "QSA q8 split-KV merge grid",
    )?;
    for kernel in [QSA_DECODE_SPLITKV_Q8_PARTIAL, QSA_DECODE_SPLITKV_Q8_MERGE] {
        gpu.ensure_kernel_public(
            QSA_DECODE_SPLITKV_Q8_MODULE,
            crate::kernels::QSA_DECODE_SPLITKV_Q8_SRC,
            kernel,
        )?;
    }
    // Actual demand: rows x n_heads x splits records; the reservation is the
    // fixed max-rows x max-splits bound, so one slot serves every admitted row
    // count and a wider launch never moves it.
    let record_bytes = |rows: usize, splits: usize| {
        rows.checked_mul(p.n_heads)
            .and_then(|value| value.checked_mul(splits))
            .and_then(|value| value.checked_mul(QSA_DECODE_SPLITKV_Q8_RECORD_BYTES))
            .ok_or_else(|| HipError::new(0, "QSA q8 split-KV workspace size overflows"))
    };
    let workspace_bytes = record_bytes(run_rows, splits)?;
    let reserve_bytes = record_bytes(
        QSA_DECODE_SPLITK_RESERVE_ROWS,
        QSA_DECODE_SPLITKV_Q8_MAX_SPLITS,
    )?;
    let workspace = gpu.qsa_gather_scratch(workspace_bytes, reserve_bytes)?;
    let mut partial = KernargBlob::new();
    for ptr in [
        qsa_attention_row_ptr(p.q_with_gate, first, 2 * p.n_heads * p.head_dim * 4),
        p.full_keys.buf.as_ptr(),
        p.full_values.buf.as_ptr(),
        qsa_attention_row_ptr(p.selected, first, p.capacity * 4),
        workspace,
    ] {
        partial.push_ptr(ptr);
    }
    for value in [
        rows,
        position_start,
        n_heads,
        n_kv_heads,
        head_dim,
        budget_blocks,
        compress,
        capacity,
        full_capacity,
        splits_i,
    ] {
        partial.push_i32(value);
    }
    partial.pad_to(16);
    let mut merge = KernargBlob::new();
    merge.push_ptr(workspace);
    merge.push_ptr(qsa_attention_row_ptr(
        p.q_with_gate,
        first,
        2 * p.n_heads * p.head_dim * 4,
    ));
    merge.push_ptr(qsa_attention_row_ptr(
        p.output,
        first,
        p.n_heads * p.head_dim * 4,
    ));
    for value in [rows, n_heads, head_dim, splits_i] {
        merge.push_i32(value);
    }
    merge.pad_to(16);
    gpu.launch_blob_recorded(
        QSA_DECODE_SPLITKV_Q8_PARTIAL,
        [kv_grid, split_grid, row_grid],
        [QSA_DECODE_SPLITKV_Q8_PARTIAL_THREADS, 1, 1],
        QSA_DECODE_SPLITKV_Q8_LDS_BYTES as u32,
        partial.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    gpu.launch_blob_recorded(
        QSA_DECODE_SPLITKV_Q8_MERGE,
        [merge_grid, 1, 1],
        [QSA_DECODE_SPLITKV_Q8_MERGE_THREADS, 1, 1],
        0,
        merge.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Every shape, type and scalar check of the direct q8 split-KV entry.
/// Stricter than the production route's inline checks: all tensors' dtypes and
/// byte extents, the full cache capacities and lengths, the 1..=64 row and 1..=32
/// split limits, the fixed q8 / head_dim 256 / 24-over-2-head shape, the shape
/// bound against the chunk's longest selected length, a gfx11/gfx12 wave32
/// device, and no recorder or capture.
#[cfg(any(test, feature = "lab"))]
fn qsa_decode_splitkv_q8_validate(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    splits: usize,
) -> HipResult<()> {
    let wrong_shape = || HipError::new(0, &ComputeError::WrongShape.to_string());
    if !gpu.arch_caps.has_gfx11_plus_simt() {
        return Err(HipError::new(
            0,
            "QSA q8 split-KV decode needs a gfx11 or gfx12 wave32 device",
        ));
    }
    if !qsa_decode_splitk_eager(gpu) {
        return Err(HipError::new(0, "QSA q8 split-KV decode is eager-only"));
    }
    if splits == 0
        || splits > QSA_DECODE_SPLITKV_Q8_MAX_SPLITS
        || !qsa_decode_splitkv_q8_shape_fits(p)
        || p.rows == 0
        || p.rows > QSA_DECODE_SPLITK_RESERVE_ROWS
    {
        return Err(wrong_shape());
    }
    for tensor in [p.q_with_gate, p.output] {
        ensure_f32(tensor)?;
    }
    check_qsa_format(
        gpu,
        p.format,
        p.n_kv_heads,
        p.head_dim,
        [p.full_keys, p.full_values],
    )?;
    if p.selected.dtype != DType::Raw || p.compress == 0 || p.capacity == 0 || p.full_capacity == 0
    {
        return Err(wrong_shape());
    }
    let end_position = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(wrong_shape)?;
    if end_position > i32::MAX as usize || end_position > p.full_capacity {
        return Err(wrong_shape());
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA q8 split-KV heads")?;
    let q_elements = checked_product(2, head_elements, "QSA q8 split-KV query")?;
    let kv_elements = checked_product(
        p.full_capacity,
        p.format.kv_row_units(p.n_kv_heads, p.head_dim),
        "QSA q8 split-KV cache",
    )?;
    let q_rows = checked_product(p.rows, q_elements, "QSA q8 split-KV query rows")?;
    let output_rows = checked_product(p.rows, head_elements, "QSA q8 split-KV output rows")?;
    let selected_bytes = checked_product(
        checked_product(p.rows, p.capacity, "QSA q8 split-KV selected rows")?,
        std::mem::size_of::<i32>(),
        "QSA q8 split-KV selected",
    )?;
    let f32_bytes = std::mem::size_of::<f32>();
    let kv_bytes = checked_product(
        p.full_capacity,
        p.format.kv_row_bytes(p.n_kv_heads, p.head_dim),
        "QSA q8 split-KV cache bytes",
    )?;
    if p.q_with_gate.numel() < q_rows
        || p.output.numel() < output_rows
        || p.full_keys.numel() < kv_elements
        || p.full_values.numel() < kv_elements
        || p.selected.numel() < selected_bytes
        || p.q_with_gate.buf.size()
            < checked_product(q_rows, f32_bytes, "QSA q8 split-KV query bytes")?
        || p.output.buf.size()
            < checked_product(output_rows, f32_bytes, "QSA q8 split-KV output bytes")?
        || p.full_keys.buf.size() < kv_bytes
        || p.full_values.buf.size() < kv_bytes
        || p.selected.buf.size() < selected_bytes
    {
        return Err(wrong_shape());
    }
    let selected_bound = checked_product(
        p.budget_blocks,
        p.compress,
        "QSA q8 split-KV selected bound",
    )?
    .checked_add(p.compress - 1)
    .ok_or_else(wrong_shape)?;
    let max_selected = end_position.min(p.capacity).min(selected_bound);
    if p.shape_selected < max_selected {
        return Err(HipError::new(
            0,
            &format!(
                "QSA q8 split-KV shape bound {} is below the active selected length {max_selected}",
                p.shape_selected
            ),
        ));
    }
    Ok(())
}

/// Direct test route to the q8 split-KV partial + merge kernels with an
/// explicit `splits` (1..=32), bypassing the knob and the route predicates:
/// the full validation of [`qsa_decode_splitkv_q8_validate`], then the
/// production launch pair. q8 K/V, head_dim 256, 24 query heads over 2 KV
/// heads, 1..=64 rows, eager only; gfx11/gfx12 wave32. Gated behind `lab` (or
/// test) like [`indexed_attention_attention_batch_exact`]. The F32 split-K
/// kernels keep [`indexed_attention_attention_splitk_for_test`].
#[cfg(any(test, feature = "lab"))]
#[doc(hidden)]
pub fn indexed_attention_attention_splitkv_q8_for_test(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    splits: usize,
) -> HipResult<()> {
    qsa_decode_splitkv_q8_validate(gpu, p, splits)?;
    qsa_decode_splitkv_q8_launch(gpu, p, splits)
}

/// Whether the dense F16 WMMA attention applies: the Qwen4 F16 route (gfx11 WMMA,
/// >= QWEN4_F16_WMMA_MIN_TOKENS rows, no recorder or capture, not opted out),
/// head_dim 256 in four-head KV groups, and every row's selection is its
/// whole causal window: the budget covers every visible block and the
/// capacity every visible token (indexed_attention_select then emits all of
/// them). Only F32 caches take this route; q8 caches take the gathered route,
/// whose per-call F16 convert dequantizes them (fp8 is gfx12-only).
fn qsa_dense_wmma_applies(
    gpu: &Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
) -> bool {
    gpu.arch_caps.has_wmma_w32()
        && gpu.route_rows(p.rows) >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && *crate::gemm::QWEN4_F16_WMMA
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.head_dim == 256
        && p.format == QsaKvFormat::F32
        && (p.n_heads / p.n_kv_heads) % 4 == 0
        && end_position / p.compress <= p.budget_blocks
        && end_position <= p.capacity
}

/// Causal GQA flash attention over cache rows `[0, end_position)` in F16
/// WMMA (kernels/src/indexed_attention_dense_wmma.gfx1151.hip).  The F16 K
/// and V^T copies live in the shared FP16 X scratch.
fn qsa_dense_wmma(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
) -> HipResult<()> {
    let width = checked_product(p.n_kv_heads, 256, "QSA dense KV width")?;
    let tpad = end_position.div_ceil(32) * 32;
    let k_elements = checked_product(end_position, width, "QSA dense K")?;
    let vt_elements = checked_product(width, tpad, "QSA dense V")?;
    let scratch = gpu.qwen4_f16_x_scratch(k_elements + vt_elements)?;
    let k16 = scratch.buf.as_ptr();
    let vt16 = unsafe { (k16 as *mut u8).add(k_elements * 2) } as *mut std::ffi::c_void;
    let tokens = checked_i32(end_position, "QSA dense tokens")?;
    let kv_heads = checked_i32(p.n_kv_heads, "QSA dense KV heads")?;
    let tpad_i = checked_i32(tpad, "QSA dense padded tokens")?;
    for kernel in [
        "indexed_attention_kv_f16",
        "indexed_attention_dense_wmma_f16",
    ] {
        gpu.ensure_kernel_public(
            "indexed_attention_dense_wmma",
            INDEXED_ATTENTION_DENSE_WMMA_SRC,
            kernel,
        )?;
    }
    let mut args = KernargBlob::new();
    args.push_ptr(p.full_keys.buf.as_ptr());
    args.push_ptr(p.full_values.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vt16);
    args.push_i32(tokens);
    args.push_i32(kv_heads);
    args.push_i32(tpad_i);
    args.pad_to(16);
    // `indexed_attention_kv_f16`: grid `[tpad, n_kv_heads]` x 256 covers `width`
    // channels exactly; `tok >= tpad` returns. Reads F32 cache rows
    // `[0, end_position)` (`tok < tokens`); writes F16 K rows
    // `[0, end_position)` and V^T `[width][tpad]`, where the `tok >= tokens`
    // columns up to `tpad = ceil32(end_position)` are zero-filled: the V^T
    // write extent is grid-rounded to `width * tpad` F16, not `end_position`.
    let dense = gpu
        .flags
        .qwen4_qsa_extent_check
        .then(|| QsaBatchExtents::new(p, end_position));
    if let Some(e) = &dense {
        let site = "indexed_attention_kv_f16";
        let source = k_elements.saturating_mul(4);
        gpu.qsa_extent_check("qsa_dense_full_keys", p.full_keys, 0, source, e.depth, site);
        gpu.qsa_extent_check(
            "qsa_dense_full_values",
            p.full_values,
            0,
            source,
            e.depth,
            site,
        );
        gpu.qsa_extent_check_ptr(
            "qsa_dense_f16_k",
            k16,
            0,
            k_elements.saturating_mul(2),
            e.depth,
            site,
        );
        gpu.qsa_extent_check_ptr(
            "qsa_dense_f16_vt",
            vt16,
            0,
            vt_elements.saturating_mul(2),
            e.depth,
            site,
        );
    }
    gpu.launch_blob_recorded(
        "indexed_attention_kv_f16",
        [
            checked_u32(tpad, "QSA dense token grid")?,
            p.n_kv_heads as u32,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some(e) = &dense {
        gpu.qsa_extent_post_launch_ptr(
            "qsa_dense_f16_vt",
            vt16,
            0,
            vt_elements.saturating_mul(2),
            e.depth,
            "indexed_attention_kv_f16",
        );
    }
    let mut args = KernargBlob::new();
    args.push_ptr(p.q_with_gate.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vt16);
    args.push_i32(tpad_i);
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(checked_i32(p.rows, "QSA dense rows")?);
    args.push_i32(checked_i32(p.position_start, "QSA dense position")?);
    args.push_i32(checked_i32(p.n_heads, "QSA dense heads")?);
    args.push_i32(kv_heads);
    args.pad_to(16);
    // Guarded vs grid-rounded: block `t0 = blockIdx.x * 16` rows; K tiles of 32
    // keys are fetched only while `k0 + key < kv_hi` with
    // `kv_hi = position_start + min(t0 + 16, rows) <= end_position`, so K reads
    // are guarded to `end_position` rows. V^T tiles are fetched unguarded
    // per 32-key tile for `k0 < kv_hi`, i.e. up to `tpad = ceil32(end_position)`
    // columns of every head dim: the grid-rounded extent is all of
    // `width * tpad` F16. Query rows are guarded `t0 + q < rows` (rows
    // `[0, rows)` read), output rows `row >= rows` skip; the selection is not
    // read (whole causal window).
    if let Some(e) = &dense {
        let site = "indexed_attention_dense_wmma_f16";
        e.check_query(gpu, p, site);
        e.check_output(gpu, p, site);
        gpu.qsa_extent_check_ptr(
            "qsa_dense_f16_k",
            k16,
            0,
            k_elements.saturating_mul(2),
            e.depth,
            site,
        );
        gpu.qsa_extent_check_ptr(
            "qsa_dense_f16_vt",
            vt16,
            0,
            vt_elements.saturating_mul(2),
            e.depth,
            site,
        );
    }
    gpu.launch_blob_recorded(
        "indexed_attention_dense_wmma_f16",
        [
            checked_u32(p.rows.div_ceil(16), "QSA dense row grid")?,
            checked_u32(p.n_heads / 4, "QSA dense head grid")?,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some(e) = &dense {
        e.post(gpu, p, "indexed_attention_dense_wmma_f16");
    }
    Ok(())
}

/// Static LDS of the gathered kernels: F16 Q (16 heads x 256), two parity
/// sets of F16 probabilities (8 waves x 16 x 16) and of the per-wave max/sum.
const QSA_GATHERED_STATIC_LDS_BYTES: usize = 16 * 256 * 2 + 2 * 8 * 256 * 2 + 2 * 2 * 8 * 16 * 4;

/// Whether this process runs QSA prefill attention in `format` on the
/// gathered F16 WMMA route: `HIPFIRE_QWEN4_QSA_WMMA_GATHER` not `0` (read
/// first, so with it `0` nothing else is consulted), the Qwen4 F16 route not opted
/// out, and an arch with a kernel for the state format (gfx1151: F32 and
/// q8, gfx1201: fp8).  Loaders use it to reserve and account the route's scratch.
pub fn qsa_gathered_wmma_enabled(gpu: &Gpu, format: QsaKvFormat) -> bool {
    *QWEN4_QSA_WMMA_GATHER
        && *crate::gemm::QWEN4_F16_WMMA
        && match format {
            QsaKvFormat::F32 => gpu.arch_caps.is_gfx1151(),
            QsaKvFormat::Fp8 => gpu.arch_caps.is_gfx1201(),
            QsaKvFormat::Q8 => gpu.arch_caps.is_gfx1151(),
            QsaKvFormat::Fp8Fnuz => false,
        }
}

/// Bytes of the gathered route's F16 workspace a forward ending at `tokens`
/// touches, with `n_kv_heads` heads of 256 channels: F16 K of the rows
/// `[0, tokens)` followed by block-transposed V of their whole four-token
/// (pooled) blocks, 2 bytes per KV channel each. Every producer write and
/// attention read of that forward lies in this prefix (selected tokens are
/// validated below the row's visible end, and a whole-block V load needs all
/// four tokens visible); at `tokens` = the cache capacity it is the whole
/// reservation.
pub fn qsa_gathered_wmma_scratch_bytes(n_kv_heads: usize, tokens: usize) -> Option<usize> {
    let width = n_kv_heads.checked_mul(256)?;
    tokens
        .checked_add(tokens.div_ceil(4).checked_mul(4)?)?
        .checked_mul(width)?
        .checked_mul(2)
}

fn qsa_gathered_wmma_bytes(n_kv_heads: usize, tokens: usize) -> HipResult<usize> {
    qsa_gathered_wmma_scratch_bytes(n_kv_heads, tokens)
        .ok_or_else(|| HipError::new(0, "QSA gathered scratch size overflows"))
}

/// Reserve the gathered route's per-GPU workspace for a QSA cache of
/// `tokens` rows when the route is enabled for `format`. Where the VMM
/// workspace is admitted (gfx1151/gfx1201 on a VMM-certified platform) this is a
/// stable-VA reservation that commits no pages, released by
/// `Gpu::invalidate_weight_caches` (model unload); elsewhere the legacy
/// slot is allocated whole, as before. Returns the reserved bytes, 0 when
/// the route is off (nothing reserved). Call at attach, before any graph
/// capture or Redline record, so the workspace address never moves; map
/// each forward's prefix with [`ensure_qsa_gathered_wmma_workspace`].
pub fn reserve_qsa_gathered_wmma_workspace(
    gpu: &mut Gpu,
    format: QsaKvFormat,
    n_kv_heads: usize,
    tokens: usize,
) -> HipResult<usize> {
    if !qsa_gathered_wmma_enabled(gpu, format) {
        return Ok(0);
    }
    let bytes = qsa_gathered_wmma_bytes(n_kv_heads, tokens)?;
    gpu.qsa_gather_reserve(bytes)?;
    Ok(bytes)
}

/// Map the gathered route's workspace over everything a forward ending at
/// `end_tokens` touches ([`qsa_gathered_wmma_scratch_bytes`]), in place at
/// its reserved address (reserving at least that much first when nothing
/// is reserved; the legacy slot grows to it). Call before capture or
/// record: VMM growth while either is armed is refused. Returns the
/// committed bytes (VMM: granularity-rounded), 0 when the route is off.
pub fn ensure_qsa_gathered_wmma_workspace(
    gpu: &mut Gpu,
    format: QsaKvFormat,
    n_kv_heads: usize,
    end_tokens: usize,
) -> HipResult<usize> {
    if !qsa_gathered_wmma_enabled(gpu, format) {
        return Ok(0);
    }
    let bytes = qsa_gathered_wmma_bytes(n_kv_heads, end_tokens)?;
    gpu.qsa_gather_scratch(bytes, bytes)?;
    Ok(gpu.qsa_gather_scratch_bytes())
}

/// Reserve the gathered route's workspace for a QSA cache of `tokens` rows
/// and commit all of it when the route is enabled for `format`; returns the
/// bytes reserved (0 when it is not enabled, and nothing is allocated).
/// The whole-context commit of the attach-time contract, kept until the
/// caller moves to [`reserve_qsa_gathered_wmma_workspace`] +
/// [`ensure_qsa_gathered_wmma_workspace`]: call at load, before any graph
/// capture or Redline record, so a forward never maps more.
pub fn reserve_qsa_gathered_wmma_scratch(
    gpu: &mut Gpu,
    format: QsaKvFormat,
    n_kv_heads: usize,
    tokens: usize,
) -> HipResult<usize> {
    let bytes = reserve_qsa_gathered_wmma_workspace(gpu, format, n_kv_heads, tokens)?;
    if bytes > 0 {
        gpu.qsa_gather_scratch(bytes, bytes)?;
    }
    Ok(bytes)
}

/// Whether the gathered F16 WMMA attention applies (checked after the dense
/// route): the route is enabled for the state format
/// ([`qsa_gathered_wmma_enabled`]), >= QWEN4_F16_WMMA_MIN_TOKENS rows, no
/// recorder or capture, head_dim 256 with at most 16 query heads per KV head,
/// and a token list that fits the LDS budget.  With
/// `HIPFIRE_QWEN4_QSA_WMMA_GATHER=0` every launch is the incumbent's.
fn qsa_gathered_wmma_applies(gpu: &Gpu, p: &IndexedAttentionAttentionBatch<'_>) -> bool {
    qsa_gathered_wmma_enabled(gpu, p.format)
        && gpu.route_rows(p.rows) >= crate::gemm::QWEN4_F16_WMMA_MIN_TOKENS
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && p.head_dim == 256
        && p.n_heads / p.n_kv_heads <= 16
        && p.capacity.div_ceil(128) * 128 * 4 + QSA_GATHERED_STATIC_LDS_BYTES
            <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
}

/// QSA attention over each row's selection in F16 WMMA
/// (kernels/src/indexed_attention_gathered_wmma.gfx{1151,1201}.hip).  The
/// cache rows `[0, end_position)` are first written as F16 K and
/// block-transposed V into the route's own workspace
/// ([`reserve_qsa_gathered_wmma_workspace`]; 2 KiB per token at 2 x 256 KV
/// channels), mapped over that prefix before the launches.
fn qsa_gathered_wmma(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
    max_selected: usize,
    arena: Option<&QsaF16ArenaRef<'_>>,
) -> HipResult<()> {
    let pm = *QWEN4_QSA_PM && qsa_gathered_pm_fits(p);
    qsa_gathered_wmma_launch_with(gpu, p, end_position, max_selected, pm, arena)
}

/// Whether the builder module covers `p`: its raw buffer offsets are 32-bit,
/// so the F16 K/V scratch over the whole cache capacity and the query row
/// stay below its out-of-range offset.
fn qsa_gathered_pm_fits(p: &IndexedAttentionAttentionBatch<'_>) -> bool {
    qsa_gathered_pm_shape_fits(p.full_capacity, p.n_kv_heads, p.n_heads)
}

pub(crate) fn qsa_gathered_pm_shape_fits(
    full_capacity: usize,
    n_kv_heads: usize,
    n_heads: usize,
) -> bool {
    const PM_OOB_OFFSET: usize = 0x7fff_ff00;
    let scratch = full_capacity
        .div_ceil(4)
        .checked_mul(4)
        .and_then(|t| t.checked_mul(n_kv_heads * 512));
    scratch.is_some_and(|bytes| bytes + 4096 <= PM_OOB_OFFSET)
        && n_heads
            .checked_mul(2048)
            .is_some_and(|bytes| bytes + 4096 <= PM_OOB_OFFSET)
        && full_capacity < 1 << 24
        && n_kv_heads * 2048 < 1 << 24
}

/// The existing hipcc/PM producer and gathered attention launches. An arena
/// replaces only the per-launch F16 workspace; all attend arguments and
/// kernels are unchanged. The quantized formats retain their converters.
fn qsa_gathered_wmma_launch_with(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttentionBatch<'_>,
    end_position: usize,
    max_selected: usize,
    pm: bool,
    arena: Option<&QsaF16ArenaRef<'_>>,
) -> HipResult<()> {
    // (module, hipcc source or the builder image, kernel).
    type Code = (
        &'static str,
        Result<&'static str, &'static [u8]>,
        &'static str,
    );
    const GFX1151: &str = "indexed_attention_gathered_wmma";
    const GFX1201: &str = "indexed_attention_gathered_wmma_gfx1201";
    const PM_GFX1151: &str = "qsa_gather_pm_gfx1151";
    const PM_GFX1201: &str = "qsa_gather_pm_gfx1201";
    let hip = |module, src, kernel| -> Code { (module, Ok(src), kernel) };
    let built = |module, image, kernel| -> Code { (module, Err(image), kernel) };
    let attend_gfx1151 = if pm {
        built(
            PM_GFX1151,
            crate::kernels::QSA_GATHER_PM_GFX1151,
            "indexed_attention_gathered_wmma_f16_pm_gfx1151",
        )
    } else {
        hip(
            GFX1151,
            INDEXED_ATTENTION_GATHERED_WMMA_SRC,
            "indexed_attention_gathered_wmma_f16",
        )
    };
    let (convert, attend) = match (p.format, pm) {
        (QsaKvFormat::F32, false) => (
            hip(
                GFX1151,
                INDEXED_ATTENTION_GATHERED_WMMA_SRC,
                "indexed_attention_kv_f16vb",
            ),
            attend_gfx1151,
        ),
        (QsaKvFormat::F32, true) => (
            built(
                PM_GFX1151,
                crate::kernels::QSA_GATHER_PM_GFX1151,
                "indexed_attention_kv_f16vb_pm_gfx1151",
            ),
            attend_gfx1151,
        ),
        (QsaKvFormat::Q8, _) => (
            hip(
                GFX1151,
                INDEXED_ATTENTION_GATHERED_WMMA_SRC,
                "indexed_attention_kv_f16vb_q8",
            ),
            attend_gfx1151,
        ),
        (QsaKvFormat::Fp8, false) => (
            hip(
                GFX1201,
                INDEXED_ATTENTION_GATHERED_WMMA_GFX1201_SRC,
                "indexed_attention_kv_f16vb_fp8_gfx1201",
            ),
            hip(
                GFX1201,
                INDEXED_ATTENTION_GATHERED_WMMA_GFX1201_SRC,
                "indexed_attention_gathered_wmma_f16_gfx1201",
            ),
        ),
        (QsaKvFormat::Fp8, true) => (
            built(
                PM_GFX1201,
                crate::kernels::QSA_GATHER_PM_GFX1201,
                "indexed_attention_kv_f16vb_fp8_pm_gfx1201",
            ),
            built(
                PM_GFX1201,
                crate::kernels::QSA_GATHER_PM_GFX1201,
                "indexed_attention_gathered_wmma_f16_pm_gfx1201",
            ),
        ),
        (QsaKvFormat::Fp8Fnuz, _) => {
            return Err(HipError::new(
                0,
                "QSA fp8 FNUZ K/V has no gathered WMMA route",
            ));
        }
    };
    let width = checked_product(p.n_kv_heads, 256, "QSA gathered KV width")?;
    // The persistent arena has no per-launch convert kernel to ensure.
    for (module, code, kernel) in [convert, attend]
        .into_iter()
        .skip(usize::from(arena.is_some()))
    {
        match code {
            Ok(src) => {
                gpu.ensure_kernel_public(module, src, kernel)?;
            }
            Err(image) => {
                gpu.ensure_embedded_kernel(module, image, kernel)?;
            }
        }
    }
    let (convert, attend) = (convert.2, attend.2);
    let (k16, vb16) = match arena {
        // The persistent arena already holds every cache row below the ones
        // this op appended; convert only those (and the partial block they
        // continue), then attend straight from it.
        Some(arena) => {
            if p.format != QsaKvFormat::F32 {
                return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
            }
            qsa_f16_arena_sync(
                gpu,
                p.full_keys,
                p.full_values,
                arena,
                p.n_kv_heads,
                p.position_start,
                end_position,
                pm,
            )?;
            (arena.keys.buf.as_ptr(), arena.values.buf.as_ptr())
        }
        None => {
            let k_elements = checked_product(end_position, width, "QSA gathered K")?;
            let bytes = qsa_gathered_wmma_scratch_bytes(p.n_kv_heads, end_position)
                .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
            // Reserved for the whole cache (one stable address per model), mapped
            // over this forward's prefix.
            let reserve =
                qsa_gathered_wmma_scratch_bytes(p.n_kv_heads, p.full_capacity.max(end_position))
                    .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
            let k16 = gpu.qsa_gather_scratch(bytes, reserve)?;
            // V follows K: whole four-token blocks, the last one padded.
            let vb16 = unsafe { (k16 as *mut u8).add(k_elements * 2) } as *mut std::ffi::c_void;
            let tokens = checked_i32(end_position, "QSA gathered tokens")?;
            let kv_heads = checked_i32(p.n_kv_heads, "QSA gathered KV heads")?;
            let mut args = KernargBlob::new();
            args.push_ptr(p.full_keys.buf.as_ptr());
            args.push_ptr(p.full_values.buf.as_ptr());
            args.push_ptr(k16);
            args.push_ptr(vb16);
            args.push_i32(tokens);
            args.push_i32(kv_heads);
            args.pad_to(16);
            // Exact guards: grid `[end_position, n_kv_heads]` x 256 threads
            // covers `width` channels with no rounding (the q8/fp8 twins index
            // by head and `threadIdx.x`, same coverage); `tok >= tokens`
            // returns. Reads cache rows `[0, end_position)` of the format's
            // row bytes; writes F16 K rows `[0, end_position)` and V elements of
            // tokens `[0, end_position)` only (`tok >= tokens` exits; no padding
            // is written), i.e. the exact highest stored byte
            // [`qsa_vb16_extent`], not a whole last block. `bytes` above is the
            // allocation (K + whole-block V), unchanged.
            let convert_extents = gpu.flags.qwen4_qsa_extent_check.then(|| {
                (
                    QsaBatchExtents::new(p, end_position).depth,
                    end_position.saturating_mul(p.format.kv_row_bytes(p.n_kv_heads, 256)),
                    k_elements.saturating_mul(2),
                    qsa_vb16_extent(end_position, width),
                )
            });
            if let Some((depth, source_len, k_len, v_len)) = convert_extents {
                gpu.qsa_extent_check(
                    "qsa_gather_full_keys",
                    p.full_keys,
                    0,
                    source_len,
                    depth,
                    convert,
                );
                gpu.qsa_extent_check(
                    "qsa_gather_full_values",
                    p.full_values,
                    0,
                    source_len,
                    depth,
                    convert,
                );
                gpu.qsa_extent_check_ptr("qsa_gather_f16_k", k16, 0, k_len, depth, convert);
                gpu.qsa_extent_check_ptr("qsa_gather_f16_vb", vb16, 0, v_len, depth, convert);
            }
            gpu.launch_blob_recorded(
                convert,
                [
                    checked_u32(end_position, "QSA gathered token grid")?,
                    p.n_kv_heads as u32,
                    1,
                ],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
                crate::dispatch::ReplayLaunchBindings::NONE,
            )?;
            if let Some((depth, _, _, v_len)) = convert_extents {
                gpu.qsa_extent_post_launch_ptr("qsa_gather_f16_vb", vb16, 0, v_len, depth, convert);
            }
            (k16, vb16)
        }
    };
    let mut args = KernargBlob::new();
    args.push_ptr(p.q_with_gate.buf.as_ptr());
    args.push_ptr(k16);
    args.push_ptr(vb16);
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    for (value, what) in [
        (p.rows, "QSA gathered rows"),
        (p.position_start, "QSA gathered position"),
        (p.n_heads, "QSA gathered heads"),
        (p.n_kv_heads, "QSA gathered KV heads"),
        (p.budget_blocks, "QSA gathered budget"),
        (p.compress, "QSA gathered compress"),
        (p.capacity, "QSA gathered capacity"),
        (p.full_capacity, "QSA gathered cache capacity"),
    ] {
        args.push_i32(checked_i32(value, what)?);
    }
    args.pad_to(16);
    // The validated token list of the longest selection, in 128-entry tiles.
    let lds = checked_u32(max_selected.div_ceil(128) * 128 * 4, "QSA gathered LDS")?;
    // PM twin (crates/peacemaker-isa/src/kernels/qsa_gather.rip.rs, `prologue` /
    // `stage` / `attend`): same selection length, token validation and
    // `token < full_capacity && token < visible` mask. Its raw buffer
    // descriptors declare selected = `selected_len * 4`, Q = `n_heads * 2048`,
    // output = `n_heads * 1024` (all per-row bases), k16 = `full_capacity *
    // width * 2` and vb16 = `ceil4(full_capacity) * width * 2` records; an
    // invalid token's offset is `GATHER_OOB`, so it touches no memory. The
    // memory actually touched is therefore the extent checked below.
    // Guarded vs grid-rounded: one workgroup per (row, kv head), grid is exact.
    // Selected entries are read only below the row's `selected_len`; the token
    // list is validated `token >= 0 && token < full_capacity && token < visible`
    // (visible = position_start + row + 1 <= end_position), so K rows read are
    // `< min(end_position, full_capacity)` (invalid entries read row 0) and
    // the V elements of those tokens only: the whole-block 8-byte load needs
    // all four tokens valid and every other path loads 2 bytes per valid
    // token, so the highest byte read is [`qsa_vb16_extent`]`(kv_rows, width)`.
    // The visible end, not the selected count, bounds K/V. The 128-entry tile
    // rounding of `selected_len` is dynamic LDS only (`lds` above), never
    // global memory. Q/output are the full `rows` extents; `l16 >= group`
    // lanes return before touching output.
    let attend_extents = gpu
        .flags
        .qwen4_qsa_extent_check
        .then(|| QsaBatchExtents::new(p, end_position));
    if let Some(e) = &attend_extents {
        e.check_query(gpu, p, attend);
        e.check_selected(gpu, p, attend);
        e.check_output(gpu, p, attend);
        let k_len = e.kv_rows.saturating_mul(width).saturating_mul(2);
        let v_len = qsa_vb16_extent(e.kv_rows, width);
        match arena {
            Some(arena) => {
                gpu.qsa_extent_check("qsa_f16_arena_keys", arena.keys, 0, k_len, e.depth, attend);
                gpu.qsa_extent_check(
                    "qsa_f16_arena_values",
                    arena.values,
                    0,
                    v_len,
                    e.depth,
                    attend,
                );
            }
            None => {
                gpu.qsa_extent_check_ptr("qsa_gather_f16_k", k16, 0, k_len, e.depth, attend);
                gpu.qsa_extent_check_ptr("qsa_gather_f16_vb", vb16, 0, v_len, e.depth, attend);
            }
        }
    }
    gpu.launch_blob_recorded(
        attend,
        [
            checked_u32(p.rows, "QSA gathered row grid")?,
            checked_u32(p.n_kv_heads, "QSA gathered head grid")?,
            1,
        ],
        [256, 1, 1],
        lds,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some(e) = &attend_extents {
        e.post(gpu, p, attend);
    }
    Ok(())
}

/// Bring a QSA layer's persistent F16 arena up to `end_tokens` cache rows:
/// the F32 cache rows `[align_down4(valid), end_tokens)` through the
/// unchanged `indexed_attention_kv_f16vb` converter, where `valid` is the
/// arena's valid prefix while the cache holds `committed` unchanged rows
/// ([`QsaF16ArenaRef::valid`]). The converter's block-transposed V index uses
/// the token's offset inside its grid, so starting the grid at a four-token
/// boundary and offsetting the F32 inputs by `start * width * 4` bytes, the
/// K output by `start * width * 2` and the V output by `start / 4` blocks of
/// `width * 8` bytes makes the local layout equal the global one; at most
/// three already-valid rows are reread. Records `end_tokens` as the valid
/// prefix. Direct (unrecorded, uncaptured) launches only: the host start and
/// length are baked into the launch.
/// `pm` chooses the already-certified producer; callers must provide mapped
/// F32 sources and F16 destinations. The byte oracle also uses this helper
/// with an empty watermark for its independent whole-prefix reference.
pub fn qsa_f16_arena_sync(
    gpu: &mut Gpu,
    full_keys: &GpuTensor,
    full_values: &GpuTensor,
    arena: &QsaF16ArenaRef<'_>,
    n_kv_heads: usize,
    committed: usize,
    end_tokens: usize,
    pm: bool,
) -> HipResult<()> {
    let valid = arena.valid(committed);
    if end_tokens <= valid {
        return Ok(());
    }
    if !gpu.arch_caps.is_gfx1151() || gpu.replay.is_recording() || gpu.graphs.capture_mode {
        return Err(HipError::new(
            0,
            "QSA F16 arena sync needs a direct launch on gfx1151",
        ));
    }
    ensure_f32(full_keys)?;
    ensure_f32(full_values)?;
    let width = checked_product(n_kv_heads, 256, "QSA F16 arena KV width")?;
    let rows = checked_product(qsa_f16_arena_rows(end_tokens), width, "QSA F16 arena rows")?;
    let source = checked_product(end_tokens, width, "QSA F16 arena source")?;
    let arena_bytes = checked_product(rows, 2, "QSA F16 arena bytes")?;
    let source_bytes = checked_product(source, 4, "QSA F16 arena source bytes")?;
    if n_kv_heads == 0
        || arena.keys.dtype != DType::F16
        || arena.values.dtype != DType::F16
        || arena.keys.numel() < rows
        || arena.values.numel() < rows
        || full_keys.numel() < source
        || full_values.numel() < source
        || full_keys.buf.size() < source_bytes
        || full_values.buf.size() < source_bytes
        || arena.keys.buf.size() < arena_bytes
        || arena.values.buf.size() < arena_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let start = qsa_f16_arena_start(valid, committed);
    let tokens = checked_i32(end_tokens - start, "QSA F16 arena tokens")?;
    let kv_heads = checked_i32(n_kv_heads, "QSA F16 arena KV heads")?;
    let offset = |base: *mut std::ffi::c_void, bytes: usize| -> HipResult<*mut std::ffi::c_void> {
        Ok(unsafe { (base as *mut u8).add(bytes) } as *mut std::ffi::c_void)
    };
    let f32_offset = checked_product(start, width * 4, "QSA F16 arena source offset")?;
    let k_offset = checked_product(start, width * 2, "QSA F16 arena K offset")?;
    let v_offset = checked_product(start / 4, width * 8, "QSA F16 arena V offset")?;
    let kernel = if pm {
        let kernel = "indexed_attention_kv_f16vb_pm_gfx1151";
        gpu.ensure_embedded_kernel(
            "qsa_gather_pm_gfx1151",
            crate::kernels::QSA_GATHER_PM_GFX1151,
            kernel,
        )?;
        kernel
    } else {
        let kernel = "indexed_attention_kv_f16vb";
        gpu.ensure_kernel_public(
            "indexed_attention_gathered_wmma",
            INDEXED_ATTENTION_GATHERED_WMMA_SRC,
            kernel,
        )?;
        kernel
    };
    let mut args = KernargBlob::new();
    args.push_ptr(offset(full_keys.buf.as_ptr(), f32_offset)?);
    args.push_ptr(offset(full_values.buf.as_ptr(), f32_offset)?);
    args.push_ptr(offset(arena.keys.buf.as_ptr(), k_offset)?);
    args.push_ptr(offset(arena.values.buf.as_ptr(), v_offset)?);
    args.push_i32(tokens);
    args.push_i32(kv_heads);
    args.pad_to(16);
    // Exact guards: grid `[end_tokens - start, n_kv_heads]` x 256 threads covers
    // `width = n_kv_heads * 256` channels with no rounding, and the kernel's
    // `tok >= tokens` guard is the grid itself. Reads the F32 cache rows
    // `[start, end_tokens)` (offset `start * width * 4`); writes F16 K rows
    // `[start, end_tokens)` and the block-transposed V elements of tokens
    // `[start, end_tokens)` at byte offset `start / 4 * width * 8`. The kernel
    // stores only existing tokens (`tok >= tokens` exits, no padding is
    // written), so the V extent is the exact highest stored byte
    // ([`qsa_vb16_extent`] of the local token count; `start` is a multiple of
    // four), not a whole last block.
    let sync_extents = gpu.flags.qwen4_qsa_extent_check.then(|| {
        let rows = end_tokens - start;
        (
            crate::dispatch::QsaExtentDepth {
                rows,
                end_position: end_tokens,
                block_count: end_tokens / 4,
            },
            rows.saturating_mul(width).saturating_mul(4),
            rows.saturating_mul(width).saturating_mul(2),
            qsa_vb16_extent(rows, width),
        )
    });
    if let Some((depth, source_len, k_len, v_len)) = sync_extents {
        gpu.qsa_extent_check(
            "qsa_f16_sync_full_keys",
            full_keys,
            f32_offset,
            source_len,
            depth,
            kernel,
        );
        gpu.qsa_extent_check(
            "qsa_f16_sync_full_values",
            full_values,
            f32_offset,
            source_len,
            depth,
            kernel,
        );
        gpu.qsa_extent_check(
            "qsa_f16_arena_keys",
            arena.keys,
            k_offset,
            k_len,
            depth,
            kernel,
        );
        gpu.qsa_extent_check(
            "qsa_f16_arena_values",
            arena.values,
            v_offset,
            v_len,
            depth,
            kernel,
        );
    }
    gpu.launch_blob_recorded(
        kernel,
        [
            checked_u32(end_tokens - start, "QSA F16 arena token grid")?,
            n_kv_heads as u32,
            1,
        ],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some((depth, _, _, v_len)) = sync_extents {
        gpu.qsa_extent_post_launch(
            "qsa_f16_arena_values",
            arena.values,
            v_offset,
            v_len,
            depth,
            kernel,
        );
    }
    arena.valid_len.converted(end_tokens);
    Ok(())
}

pub struct IndexedAttentionPool<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub block_count: usize,
    pub head_dim: usize,
}

pub fn indexed_attention_pool(gpu: &mut Gpu, p: &IndexedAttentionPool<'_>) -> HipResult<()> {
    ensure_f32(p.raw_keys)?;
    ensure_f32(p.pooled)?;
    if p.block_count == 0 || p.head_dim == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let raw_elements = checked_product3(p.block_count, 4, p.head_dim, "QSA pool raw extent")?;
    let pooled_elements = checked_product(p.block_count, p.head_dim, "QSA pool output extent")?;
    let block_count = checked_i32(p.block_count, "QSA pool block count")?;
    let head_dim = checked_i32(p.head_dim, "QSA pool head width")?;
    let block_grid = checked_u32(p.block_count, "QSA pool block grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.raw_keys.numel() < raw_elements || p.pooled.numel() < pooled_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "indexed_attention_pool_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_i32(block_count);
    args.push_i32(head_dim);
    args.pad_to(16);
    // `indexed_attention_pool_f32` is not defined anywhere in kernels/src (this
    // entry point has no caller and `ensure_kernel_public` above fails first),
    // so these are the host shape-contract extents, not a measured kernel
    // guard: raw keys `[block_count * 4][head_dim]` read, pooled
    // `[block_count][head_dim]` written. `dim_grid * 256` may exceed
    // `head_dim` (grid-rounded); a kernel for this ABI must guard
    // `channel < head_dim`, which this check cannot see.
    let pool_depth = gpu
        .flags
        .qwen4_qsa_extent_check
        .then(|| crate::dispatch::QsaExtentDepth {
            rows: p.block_count,
            end_position: p.block_count.saturating_mul(4),
            block_count: p.block_count,
        });
    if let Some(depth) = pool_depth {
        gpu.qsa_extent_check(
            "qsa_pool_raw_keys",
            p.raw_keys,
            0,
            raw_elements.saturating_mul(4),
            depth,
            "indexed_attention_pool_f32",
        );
        gpu.qsa_extent_check(
            "qsa_pool_pooled",
            p.pooled,
            0,
            pooled_elements.saturating_mul(4),
            depth,
            "indexed_attention_pool_f32",
        );
    }
    gpu.launch_blob_recorded(
        "indexed_attention_pool_f32",
        [block_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )?;
    if let Some(depth) = pool_depth {
        gpu.qsa_extent_post_launch(
            "qsa_pool_pooled",
            p.pooled,
            0,
            pooled_elements.saturating_mul(4),
            depth,
            "indexed_attention_pool_f32",
        );
    }
    Ok(())
}

/// In-place scalar scaling for learned HC branch projections.
pub struct ScaleF32<'a> {
    pub values: &'a GpuTensor,
    pub scale: f32,
}

pub fn scale_f32(gpu: &mut Gpu, p: &ScaleF32<'_>) -> HipResult<()> {
    ensure_f32(p.values)?;
    let elements = checked_extent(p.values.numel(), "scale extent")?;
    if elements == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "scale extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "scale_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_i32(elements_i);
    args.push_f32(p.scale);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "scale_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Device-side greedy top-1 over contiguous F32 logits rows.
pub struct ArgmaxF32<'a> {
    pub logits: &'a GpuTensor,
    pub indices: &'a GpuTensor,
    pub rows: usize,
    pub vocab: usize,
}

pub fn argmax_f32(gpu: &mut Gpu, p: &ArgmaxF32<'_>) -> HipResult<()> {
    ensure_f32(p.logits)?;
    if p.rows == 0 || p.vocab == 0 || p.indices.dtype != DType::Raw {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let logits_elements = checked_product(p.rows, p.vocab, "argmax logits extent")?;
    let indices_bytes = p
        .rows
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let rows = checked_i32(p.rows, "argmax row count")?;
    let vocab = checked_i32(p.vocab, "argmax vocabulary width")?;
    let row_grid = checked_u32(p.rows, "argmax row grid")?;
    if p.logits.numel() != logits_elements || p.indices.numel() < indices_bytes {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "argmax_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.logits.buf.as_ptr());
    args.push_ptr(p.indices.buf.as_ptr());
    args.push_i32(rows);
    args.push_i32(vocab);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        "argmax_f32",
        [row_grid, 1, 1],
        [1024, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// [`argmax_f32`] of one logits row, read back: `llama::argmax(logits)`
/// without downloading the row. Uses a lazily allocated 4-byte scratch.
pub fn argmax_f32_host(gpu: &mut Gpu, logits: &GpuTensor) -> HipResult<u32> {
    argmax_f32_host_impl(gpu, logits, false)
}

/// [`argmax_f32_host`] for a decode loop that replays a graph on
/// `Gpu::active_stream`: the 4-byte result is read back ordered on that stream
/// ([`Gpu::download_f32_stream_ordered`]) instead of a blocking null-stream copy.
pub fn argmax_f32_host_stream_ordered(gpu: &mut Gpu, logits: &GpuTensor) -> HipResult<u32> {
    argmax_f32_host_impl(gpu, logits, true)
}

fn argmax_f32_host_impl(gpu: &mut Gpu, logits: &GpuTensor, ordered: bool) -> HipResult<u32> {
    if gpu.scratch.argmax_host.is_none() {
        gpu.scratch.argmax_host = Some(gpu.alloc_tensor(&[1], DType::F32)?);
    }
    let result = gpu.scratch.argmax_host.as_ref().unwrap().sub_offset(0, 1);
    let mut indices = result.sub_offset(0, 1);
    indices.dtype = DType::Raw;
    indices.shape = vec![4];
    argmax_f32(
        gpu,
        &ArgmaxF32 {
            logits,
            indices: &indices,
            rows: 1,
            vocab: logits.numel(),
        },
    )?;
    let row = if ordered {
        gpu.download_f32_stream_ordered(&result)?
    } else {
        gpu.download_f32(&result)?
    };
    Ok(row[0].to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qsa_bf16_split_arch_predicate_is_gfx1201_or_gfx1100_unless_killed() {
        let f = qsa_bf16_split_arch_admits;
        // gfx1201: admitted whatever the gfx1100 knob says.
        assert!(f(true, false, false, true, false));
        assert!(f(true, false, true, true, false));
        // gfx1100: admitted with the knob (default), not with the kill switch.
        assert!(f(false, true, true, true, false));
        assert!(!f(false, true, false, true, false));
        // other arch: never.
        assert!(!f(false, false, true, true, false));
        assert!(!f(false, false, false, true, false));
        // conjunction terms still gate both archs.
        for (a, b) in [(true, false), (false, true)] {
            assert!(!f(a, b, true, false, false));
            assert!(!f(a, b, true, true, true));
        }
    }

    /// The grouped attention's dynamic LDS: 20 bytes per selected row for the
    /// shipped kernels (weights[4] + token) plus a 4112-byte maxes/partial-max
    /// tail; the q8 lite twin keeps no token array and reduces maxes with wave
    /// shuffles: 16 bytes per selected row plus 128 bytes. The measured
    /// prefill shape (2051 selected rows) is 45132 / 32944 bytes.
    #[test]
    fn qsa_attention_hg4_lds_formula_drops_the_token_array_for_the_lite_twin() {
        assert_eq!(qsa_attention_hg4_lds_bytes_for(2051, false), Some(45132));
        assert_eq!(qsa_attention_hg4_lds_bytes_for(2051, true), Some(32944));
        assert_eq!(qsa_attention_hg4_lds_bytes_for(0, false), Some(4112));
        assert_eq!(qsa_attention_hg4_lds_bytes_for(0, true), Some(128));
        assert_eq!(qsa_attention_hg4_lds_bytes_for(usize::MAX, true), None);
        // Three lite workgroups fit a 128 KiB WGP at the measured shape; two
        // shipped ones do, a third does not. Four lite ones do not (the
        // weights alone are 32816 bytes at 2051 rows).
        assert!(3 * 32944 <= 128 * 1024 && 4 * 32944 > 128 * 1024 && 3 * 45132 > 128 * 1024);
        // The largest selected capacity under the 64 KiB dynamic limit.
        let limit = QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES;
        assert!(qsa_attention_hg4_lds_bytes_for(3071, false).unwrap() <= limit);
        assert!(qsa_attention_hg4_lds_bytes_for(3072, false).unwrap() > limit);
        assert!(qsa_attention_hg4_lds_bytes_for(4088, true).unwrap() <= limit);
        assert!(qsa_attention_hg4_lds_bytes_for(4089, true).unwrap() > limit);
    }

    /// The lite twin serves q8 K/V on exact gfx1100 unless
    /// `HIPFIRE_QWEN4_QSA_HG4_Q8_LITE=0`; every other format and architecture
    /// keeps the shipped kernel (and its LDS size).
    #[test]
    fn qsa_attention_hg4_lite_is_q8_exact_gfx1100_with_a_kill_switch() {
        let flags = |arch: &str, switch: bool| {
            let mut flags = crate::feature_flags::FeatureFlags::for_test(arch);
            flags.qwen4_qsa_hg4_q8_lite = switch;
            flags
        };
        for format in [
            QsaKvFormat::F32,
            QsaKvFormat::Fp8,
            QsaKvFormat::Q8,
            QsaKvFormat::Fp8Fnuz,
        ] {
            for arch in ["gfx1100", "gfx1101", "gfx1151", "gfx1201", "gfx942"] {
                let expect = format == QsaKvFormat::Q8 && arch == "gfx1100";
                assert_eq!(
                    qsa_attention_hg4_lite(&flags(arch, true), format),
                    expect,
                    "{arch} {format:?}"
                );
                assert!(
                    !qsa_attention_hg4_lite(&flags(arch, false), format),
                    "{arch} {format:?}: kill switch"
                );
            }
        }
    }

    /// The decode attention policy over the flags and architecture alone:
    /// F32 K/V keeps split-K (exact gfx1151), q8 K/V at 24 query heads over 2
    /// KV heads takes split-KV (exact gfx1151 and gfx1100), everything else (recorded/captured, kill switch, other
    /// geometry or architecture) stays per-head. Row counts are not an input.
    #[test]
    fn qsa_decode_attention_policy_routes_f32_split_k_and_q8_split_kv() {
        let policy = |arch: &str,
                      switch: bool,
                      format,
                      heads: usize,
                      kv_heads: usize,
                      head_dim: usize,
                      eager: bool| {
            let mut flags = crate::feature_flags::FeatureFlags::for_test(arch);
            flags.qwen4_qsa_decode_splitk = switch;
            let flags = std::sync::Arc::new(flags);
            let caps = crate::arch_caps::ArchCaps::new(arch, flags.clone());
            qsa_decode_attention_policy_with(
                &flags, &caps, format, head_dim, heads, kv_heads, 2051, eager,
            )
        };
        let q8 = QsaDecodeAttentionPolicy::SplitKvQ8 {
            splits: QSA_DECODE_SPLITK_SPLITS,
        };
        let f32_split_k = QsaDecodeAttentionPolicy::SplitK {
            splits: QSA_DECODE_SPLITK_SPLITS,
            layout: QSA_DECODE_SPLITK_SCORING_LAYOUT,
        };
        let per_head = QsaDecodeAttentionPolicy::PerHead;
        // q8: exact gfx1151, eager, 24 / 2 / 256.
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::Q8, 24, 2, 256, true),
            q8
        );
        // Recorded or captured launch.
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::Q8, 24, 2, 256, false),
            per_head
        );
        // Kill switch.
        assert_eq!(
            policy("gfx1151", false, QsaKvFormat::Q8, 24, 2, 256, true),
            per_head
        );
        // Wrong head geometry.
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::Q8, 12, 1, 256, true),
            per_head
        );
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::Q8, 24, 4, 256, true),
            per_head
        );
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::Q8, 24, 2, 128, true),
            per_head
        );
        // Another architecture.
        assert_eq!(
            policy("gfx1201", true, QsaKvFormat::Q8, 24, 2, 256, true),
            per_head
        );
        // gfx1100 takes q8 split-KV (eager, flag on) but never the F32 split-K.
        assert_eq!(
            policy("gfx1100", true, QsaKvFormat::Q8, 24, 2, 256, true),
            q8
        );
        assert_eq!(
            policy("gfx1100", true, QsaKvFormat::Q8, 24, 2, 256, false),
            per_head
        );
        assert_eq!(
            policy("gfx1100", false, QsaKvFormat::Q8, 24, 2, 256, true),
            per_head
        );
        assert_eq!(
            policy("gfx1100", true, QsaKvFormat::Q8, 12, 1, 256, true),
            per_head
        );
        assert_eq!(
            policy("gfx1100", true, QsaKvFormat::F32, 24, 2, 256, true),
            per_head
        );
        assert_eq!(
            policy("gfx1101", true, QsaKvFormat::Q8, 24, 2, 256, true),
            per_head
        );
        // F32 K/V is unchanged: split-K, never split-KV.
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::F32, 24, 2, 256, true),
            f32_split_k
        );
        assert_eq!(
            policy("gfx1151", true, QsaKvFormat::F32, 24, 2, 256, false),
            per_head
        );
        assert_eq!(
            policy("gfx1151", false, QsaKvFormat::F32, 24, 2, 256, true),
            per_head
        );
        assert_eq!(
            policy("gfx1201", true, QsaKvFormat::F32, 24, 2, 256, true),
            per_head
        );
        // The q8 split-KV LDS is fixed: a selected list far past the F32 slice
        // limit still takes it.
        let mut flags = crate::feature_flags::FeatureFlags::for_test("gfx1151");
        flags.qwen4_qsa_decode_splitk = true;
        let flags = std::sync::Arc::new(flags);
        let caps = crate::arch_caps::ArchCaps::new("gfx1151", flags.clone());
        assert_eq!(
            qsa_decode_attention_policy_with(
                &flags,
                &caps,
                QsaKvFormat::Q8,
                256,
                24,
                2,
                usize::MAX / 4,
                true
            ),
            q8
        );
        assert_eq!(
            qsa_decode_attention_policy_with(
                &flags,
                &caps,
                QsaKvFormat::F32,
                256,
                24,
                2,
                usize::MAX / 4,
                true
            ),
            per_head
        );
    }

    #[test]
    fn qsa_row_policy_thresholds_and_raw_selected_offsets() {
        let mut flags = crate::feature_flags::FeatureFlags::for_test("gfx1151");
        flags.qwen4_qsa_decode_splitk = true;
        let flags = std::sync::Arc::new(flags);
        let caps = crate::arch_caps::ArchCaps::new("gfx1151", flags.clone());
        let threshold = (QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES
            - QSA_DECODE_SPLITK_STATIC_LDS_BYTES)
            / QSA_DECODE_SPLITK_GROUP_LDS_BYTES_PER_ROW
            * QSA_DECODE_SPLITK_SPLITS;
        let dummy = GpuTensor::null_for_test();
        for rows in 1..=crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS {
            for compress in [1usize, 4, 8] {
                for budget in [0usize, 1, threshold / compress + 2] {
                    for capacity in [1usize, 33, 65, threshold + 64] {
                        let mut storage = vec![0u8; rows * capacity * 4];
                        let selected = GpuTensor {
                            buf: unsafe {
                                hip_bridge::DeviceBuffer::from_raw(
                                    storage.as_mut_ptr().cast(),
                                    storage.len(),
                                )
                            },
                            shape: vec![storage.len()],
                            dtype: DType::Raw,
                        };
                        let p = IndexedAttentionAttentionBatch {
                            q_with_gate: &dummy,
                            full_keys: &dummy,
                            full_values: &dummy,
                            selected: &selected,
                            output: &dummy,
                            rows,
                            position_start: threshold - rows / 2,
                            n_heads: 24,
                            n_kv_heads: 2,
                            head_dim: 256,
                            budget_blocks: budget,
                            compress,
                            capacity,
                            full_capacity: threshold + 128,
                            format: QsaKvFormat::F32,
                            shape_selected: capacity,
                        };
                        for row in 0..rows {
                            let want = (p.position_start + row + 1)
                                .min(capacity)
                                .min(budget * compress + compress - 1);
                            assert_eq!(qsa_attention_selected_at(&p, row), want);
                            assert_eq!(
                                qsa_attention_row_ptr(&selected, row, capacity * 4) as usize,
                                storage.as_ptr() as usize + row * capacity * 4,
                            );
                            let eager = qsa_decode_attention_policy_with(
                                &flags, &caps, p.format, 256, 24, 2, want, true,
                            );
                            assert_eq!(
                                eager,
                                if want <= threshold {
                                    QsaDecodeAttentionPolicy::SplitK {
                                        splits: QSA_DECODE_SPLITK_SPLITS,
                                        layout: QSA_DECODE_SPLITK_SCORING_LAYOUT,
                                    }
                                } else {
                                    QsaDecodeAttentionPolicy::PerHead
                                },
                                "rows={rows} row={row} selected={want}",
                            );
                            assert_eq!(
                                qsa_decode_attention_policy_with(
                                    &flags, &caps, p.format, 256, 24, 2, want, false,
                                ),
                                QsaDecodeAttentionPolicy::PerHead,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn qsa_exact_selector_stays_gfx11_plus_only() {
        let flags = std::sync::Arc::new(crate::feature_flags::FeatureFlags::for_test("gfx1100"));
        for arch in ["gfx1100", "gfx1151", "gfx1201"] {
            let caps = crate::arch_caps::ArchCaps::new(arch, flags.clone());
            assert!(qsa_select_exact_arch(&caps), "{arch}");
            assert!(caps.has_qwen4_parallel_qsa_gdn(), "{arch}");
        }
        // gfx942 gets the parallel incumbent selector, never the mov_dpp8 one.
        let gfx942 = crate::arch_caps::ArchCaps::new("gfx942", flags.clone());
        assert!(gfx942.has_qwen4_parallel_qsa_gdn());
        assert!(!qsa_select_exact_arch(&gfx942));
        for arch in ["gfx940", "gfx941"] {
            let caps = crate::arch_caps::ArchCaps::new(arch, flags.clone());
            assert!(
                !qsa_select_exact_arch(&caps) && !caps.has_qwen4_parallel_qsa_gdn(),
                "{arch}"
            );
        }
    }

    #[test]
    fn qsa_f16_arena_sizing_and_alignment() {
        let arena = QsaStateLayout {
            f16_arena: true,
            ..QsaStateLayout::F32
        };
        assert_eq!(QsaStateLayout::F32.f16_arena_bytes_per_token(2, 256), 0);
        assert_eq!(arena.f16_arena_bytes_per_token(2, 256), 2048);
        for tokens in [0usize, 1, 2, 3, 4, 5, 8191, 8192, 8193, 262144] {
            let rows = qsa_f16_arena_rows(tokens);
            assert_eq!(rows % 4, 0);
            assert!(rows >= tokens && rows - tokens < 4);
        }
        for committed in 0..1024 {
            for valid in [0, committed, committed + 17] {
                let start = qsa_f16_arena_start(valid, committed);
                assert_eq!(start % 4, 0);
                assert!(start <= valid.min(committed));
                assert!(valid.min(committed) - start < 4);
            }
        }
    }

    #[test]
    fn qsa_bf16_split_reserve_layout_is_83894272() {
        assert_eq!(qsa_bf16_split_reserve_layout().unwrap().total, 83_894_272);
    }

    #[test]
    fn qsa_f16_arena_watermark_clamp_invalidate_and_move() {
        let mark = QsaF16ValidLen::default();
        assert_eq!(mark.get(), 0);
        mark.converted(8193);
        assert_eq!(mark.valid(8192), 8192);
        mark.clamp(8191);
        assert_eq!(mark.get(), 8191);
        mark.clamp(9000); // A decode append does not claim converted rows.
        assert_eq!(mark.get(), 8191);
        assert_eq!(qsa_f16_arena_start(mark.get(), 8191), 8188);
        mark.converted(9000);
        let moved = mark; // Watermark moves with the storage owner.
        assert_eq!(moved.get(), 9000);
        moved.invalidate(); // Replaced prefix, reset, or failed append.
        assert_eq!(moved.valid(9000), 0);
        moved.converted(3);
        moved.clamp(0);
        assert_eq!(moved.get(), 0);
    }

    fn try_gpu() -> Option<crate::gpu_test::TestGpu> {
        crate::gpu_test::init().ok()
    }

    /// The batched gated RMSNorm (with its folded BF16 recurrent rounding)
    /// must equal round-trip + per-row gate bit for bit, for both the gated
    /// output and the in-place rounded recurrent buffer.
    #[test]
    fn gdn_gate_batch_is_bit_identical_to_per_row_gate() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (rows, heads, dim) = (5usize, 6usize, 128usize);
        let width = heads * dim;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let recurrent = wave(1, rows * width, 3.0);
        let z = wave(2, rows * width, 4.0);
        let norm_bits: Vec<u8> = wave(3, dim, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16 + 0x3f00).to_le_bytes())
            .collect();
        let mut norm = gpu
            .upload_raw(&norm_bits, &[norm_bits.len()])
            .expect("norm");
        norm.dtype = DType::BF16;
        norm.shape = vec![dim];
        let z_gpu = gpu.upload_f32(&z, &[z.len()]).expect("z");

        let rec_a = gpu.upload_f32(&recurrent, &[recurrent.len()]).expect("rec");
        let out_a = gpu.zeros(&[rows * width], DType::F32).expect("out");
        gated_delta_gate_batched(
            &mut gpu,
            &GatedDeltaGateBatched {
                recurrent_output: &rec_a,
                z: &z_gpu,
                norm: &norm,
                output: &out_a,
                rows,
                value_heads: heads,
                value_dim: dim,
            },
        )
        .expect("batched gate");

        let rec_b = gpu.upload_f32(&recurrent, &[recurrent.len()]).expect("rec");
        let out_b = gpu.zeros(&[rows * width], DType::F32).expect("out");
        gpu.bf16_round_trip_f32(&rec_b).expect("round trip");
        for row in 0..rows {
            gated_delta_gate(
                &mut gpu,
                &GatedDeltaGate {
                    recurrent_output: &rec_b.sub_offset(row * width, width),
                    z: &z_gpu.sub_offset(row * width, width),
                    norm: &norm,
                    output: &out_b.sub_offset(row * width, width),
                    value_heads: heads,
                    value_dim: dim,
                },
            )
            .expect("per-row gate");
        }
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &out_a), bits(&gpu, &out_b));
        assert!(a.iter().any(|v| *v != 0), "gate output is all zero");
        assert_eq!(a, b, "batched gate output differs");
        assert_eq!(
            bits(&gpu, &rec_a),
            bits(&gpu, &rec_b),
            "rounded recurrent differs"
        );
        for tensor in [norm, z_gpu, rec_a, out_a, rec_b, out_b] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The row-parallel batched convolution must equal the per-row ring kernel
    /// run row by row (in-place history), for outputs and the final history,
    /// across chunk boundaries, short batches and every start cursor.
    #[test]
    fn gdn_conv_batch_is_bit_identical_to_per_row_conv() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let channels = 300usize;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let kernel_bits: Vec<u8> = wave(1, channels * 4, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut kernel = gpu
            .upload_raw(&kernel_bits, &[kernel_bits.len()])
            .expect("kernel");
        kernel.dtype = DType::BF16;
        kernel.shape = vec![channels * 4];
        let history = wave(2, channels * 3, 2.0);
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for (rows, start_cursor) in [(1usize, 0usize), (2, 2), (3, 1), (17, 0), (37, 2), (40, 1)] {
            let input = gpu
                .upload_f32(&wave(3 + rows, rows * channels, 3.0), &[rows * channels])
                .expect("input");
            let hist_a = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let out_a = gpu.zeros(&[rows * channels], DType::F32).expect("out");
            gated_delta_conv_batched(
                &mut gpu,
                &GatedDeltaConvBatched {
                    input: &input,
                    kernel: &kernel,
                    history: &hist_a,
                    output: &out_a,
                    next_history: &hist_a,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor,
                },
            )
            .expect("batched conv");
            let hist_b = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let out_b = gpu.zeros(&[rows * channels], DType::F32).expect("out");
            for row in 0..rows {
                gated_delta_conv(
                    &mut gpu,
                    &GatedDeltaConv {
                        input: &input.sub_offset(row * channels, channels),
                        kernel: &kernel,
                        history: &hist_b,
                        output: &out_b.sub_offset(row * channels, channels),
                        next_history: &hist_b,
                        channels,
                        history_rows: 3,
                        kernel_size: 4,
                        cursor: (start_cursor + row) % 3,
                        row_index: row,
                    },
                )
                .expect("per-row conv");
            }
            assert_eq!(
                bits(&gpu, &out_a),
                bits(&gpu, &out_b),
                "rows {rows}: output"
            );
            assert_eq!(
                bits(&gpu, &hist_a),
                bits(&gpu, &hist_b),
                "rows {rows}: history"
            );
            for tensor in [input, hist_a, out_a, hist_b, out_b] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        gpu.free_tensor(kernel).expect("free");
    }

    /// The compact 1-D grid of the conv + parameter launch: the convolution
    /// output and ring must equal the convolution-only launch, and gate / beta
    /// the stand-alone batched parameter kernel's, with a ragged channel block
    /// (300 channels), rows that are not multiples of 16, several row blocks
    /// (3 .. 512) and several parameter blocks (the 48-head Flash-Next shape).
    #[test]
    fn gdn_conv_params_compact_grid_is_bit_identical_to_split_launches() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (channels, heads) = (300usize, 48usize);
        let mut kernel = gpu
            .upload_raw(
                &bf16_le_bytes(&test_wave(1, channels * 4, 1.0)),
                &[channels * 4 * 2],
            )
            .expect("kernel");
        kernel.dtype = DType::BF16;
        kernel.shape = vec![channels * 4];
        let mut a_log = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(11, heads, 1.0)), &[heads * 2])
            .expect("a_log");
        a_log.dtype = DType::BF16;
        a_log.shape = vec![heads];
        let mut dt_bias = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(12, heads, 1.0)), &[heads * 2])
            .expect("dt_bias");
        dt_bias.dtype = DType::BF16;
        dt_bias.shape = vec![heads];
        let history = test_wave(2, channels * 3, 2.0);
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for (rows, start_cursor) in [(37usize, 2usize), (513, 1), (8192, 0)] {
            let input = gpu
                .upload_f32(
                    &test_wave(3 + rows, rows * channels, 3.0),
                    &[rows * channels],
                )
                .expect("input");
            let a = gpu
                .upload_f32(&test_wave(9, rows * heads, 2.0), &[rows * heads])
                .expect("a");
            let b = gpu
                .upload_f32(&test_wave(10, rows * heads, 2.0), &[rows * heads])
                .expect("b");
            // Split launches: convolution only, then the stand-alone parameters.
            let hist_a = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let out_a = gpu.zeros(&[rows * channels], DType::F32).expect("out");
            let gate_a = gpu.zeros(&[rows * heads], DType::F32).expect("gate");
            let beta_a = gpu.zeros(&[rows * heads], DType::F32).expect("beta");
            gated_delta_conv_batched(
                &mut gpu,
                &GatedDeltaConvBatched {
                    input: &input,
                    kernel: &kernel,
                    history: &hist_a,
                    output: &out_a,
                    next_history: &hist_a,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor,
                },
            )
            .expect("conv");
            gated_delta_params_batched(
                &mut gpu,
                &GatedDeltaParamsBatched {
                    a: &a,
                    b: &b,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    gate: &gate_a,
                    beta: &beta_a,
                    rows,
                    heads,
                },
            )
            .expect("params");
            // One launch over the compact grid.
            let hist_b = gpu.upload_f32(&history, &[history.len()]).expect("hist");
            let out_b = gpu.zeros(&[rows * channels], DType::F32).expect("out");
            let gate_b = gpu.zeros(&[rows * heads], DType::F32).expect("gate");
            let beta_b = gpu.zeros(&[rows * heads], DType::F32).expect("beta");
            gated_delta_conv_params_batched(
                &mut gpu,
                &GatedDeltaConvBatched {
                    input: &input,
                    kernel: &kernel,
                    history: &hist_b,
                    output: &out_b,
                    next_history: &hist_b,
                    rows,
                    channels,
                    history_rows: 3,
                    kernel_size: 4,
                    start_cursor,
                },
                &GatedDeltaParamsBatched {
                    a: &a,
                    b: &b,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    gate: &gate_b,
                    beta: &beta_b,
                    rows,
                    heads,
                },
            )
            .expect("conv + params");
            assert!(
                bits(&gpu, &gate_a).iter().any(|v| *v != 0),
                "rows {rows}: gate reference is all zero"
            );
            assert_eq!(
                bits(&gpu, &out_a),
                bits(&gpu, &out_b),
                "rows {rows}: output"
            );
            assert_eq!(
                bits(&gpu, &hist_a),
                bits(&gpu, &hist_b),
                "rows {rows}: history"
            );
            assert_eq!(
                bits(&gpu, &gate_a),
                bits(&gpu, &gate_b),
                "rows {rows}: gate"
            );
            assert_eq!(
                bits(&gpu, &beta_a),
                bits(&gpu, &beta_b),
                "rows {rows}: beta"
            );
            for tensor in [
                input, a, b, hist_a, out_a, gate_a, beta_a, hist_b, out_b, gate_b, beta_b,
            ] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        for tensor in [kernel, a_log, dt_bias] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The fused HC norm + BF16 gate projection must equal hyper_norm followed
    /// by the multi-row BF16 GEMM bit for bit, at the production width.
    #[test]
    fn hyper_norm_gate_is_bit_identical_to_norm_then_gemm() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (rows, branches, hidden) = (37usize, 4usize, 2560usize);
        let wide = branches * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bf16 = |gpu: &mut Gpu, values: &[f32]| -> GpuTensor {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut tensor = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16");
            tensor.dtype = DType::BF16;
            tensor.shape = vec![values.len()];
            tensor
        };
        let input = gpu
            .upload_f32(&wave(1, rows * wide, 3.0), &[rows * wide])
            .expect("input");
        let norm = bf16(&mut gpu, &wave(2, wide, 0.5));
        let gate_weight = bf16(&mut gpu, &wave(3, branches * wide, 0.05));
        let fused = gpu.zeros(&[rows * branches], DType::F32).expect("fused");
        hyper_norm_gate(
            &mut gpu,
            &HyperNormGate {
                input: &input,
                norm_weight: &norm,
                gate_weight: &gate_weight,
                gates: &fused,
                rows,
                branches,
                hidden,
                state_bf16: false,
            },
        )
        .expect("fused");
        let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
        hyper_norm(
            &mut gpu,
            &HyperNorm {
                input: &input,
                norm_weight: &norm,
                normalized: &normalized,
                branches,
                hidden,
                state_bf16: false,
            },
        )
        .expect("norm");
        let reference = gpu
            .zeros(&[rows * branches], DType::F32)
            .expect("reference");
        gpu.gemm_bf16_xf32_multirow(&gate_weight, &normalized, &reference, branches, wide, rows)
            .expect("gemm");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &fused), bits(&gpu, &reference));
        assert!(b.iter().any(|v| *v != 0), "gates are all zero");
        assert_eq!(a, b, "fused norm-gate differs");
        for tensor in [input, norm, gate_weight, fused, normalized, reference] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// H4 read side: `hyper_norm_gate_outputs` stores, bytewise, the gates of
    /// `hyper_norm_gate` and the F16 normalized row of `hyper_norm_f16`, for F32
    /// streams and for BF16-bit streams, over ragged row counts.
    #[test]
    fn hyper_norm_gate_outputs_is_bytewise_the_norm_gate_and_norm_f16_pair() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs gfx11+ wave32");
            return;
        }
        let (branches, hidden) = (4usize, 2560usize);
        let wide = branches * hidden;
        let norm = {
            let mut t = gpu
                .upload_raw(&bf16_le_bytes(&test_wave(2, wide, 0.5)), &[wide * 2])
                .expect("norm");
            t.dtype = DType::BF16;
            t.shape = vec![wide];
            t
        };
        let gate_weight = {
            let mut t = gpu
                .upload_raw(
                    &bf16_le_bytes(&test_wave(3, branches * wide, 0.05)),
                    &[branches * wide * 2],
                )
                .expect("gate weight");
            t.dtype = DType::BF16;
            t.shape = vec![branches * wide];
            t
        };
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        for rows in [1usize, 37, 512, 530] {
            for state_bf16 in [false, true] {
                let values = test_wave(1, rows * wide, 3.0);
                let input = if state_bf16 {
                    // BF16 bits in the first half of an F32-typed buffer.
                    let mut raw = bf16_le_bytes(&values);
                    raw.resize(rows * wide * 4, 0);
                    gpu.upload_raw(&raw, &[rows * wide * 4])
                        .map(|mut t| {
                            t.dtype = DType::F32;
                            t.shape = vec![rows * wide];
                            t
                        })
                        .expect("input")
                } else {
                    gpu.upload_f32(&values, &[rows * wide]).expect("input")
                };
                let gates_ref = gpu.zeros(&[rows * branches], DType::F32).expect("gates");
                let x16_ref = gpu.zeros(&[rows * wide], DType::F16).expect("x16");
                let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
                hyper_norm_gate(
                    &mut gpu,
                    &HyperNormGate {
                        input: &input,
                        norm_weight: &norm,
                        gate_weight: &gate_weight,
                        gates: &gates_ref,
                        rows,
                        branches,
                        hidden,
                        state_bf16,
                    },
                )
                .expect("norm gate");
                hyper_norm_f16(
                    &mut gpu,
                    &HyperNorm {
                        input: &input,
                        norm_weight: &norm,
                        normalized: &normalized,
                        branches,
                        hidden,
                        state_bf16,
                    },
                    &x16_ref,
                    false,
                )
                .expect("norm f16");
                let gates = gpu.zeros(&[rows * branches], DType::F32).expect("gates");
                let x16 = gpu.zeros(&[rows * wide], DType::F16).expect("x16");
                hyper_norm_gate_outputs(
                    &mut gpu,
                    &HyperNormGateOutputs {
                        input: &input,
                        norm_weight: &norm,
                        gate_weight: &gate_weight,
                        gates: &gates,
                        normalized_f16: &x16,
                        rows,
                        branches,
                        hidden,
                        state_bf16,
                    },
                )
                .expect("outputs");
                assert!(
                    bytes(&gpu, &gates) == bytes(&gpu, &gates_ref),
                    "{rows} rows bf16={state_bf16}: gates differ"
                );
                assert!(
                    bytes(&gpu, &x16) == bytes(&gpu, &x16_ref),
                    "{rows} rows bf16={state_bf16}: F16 normalized row differs"
                );
                assert!(
                    bytes(&gpu, &gates).iter().any(|b| *b != 0),
                    "gates are all zero"
                );
                for tensor in [input, gates_ref, x16_ref, normalized, gates, x16] {
                    gpu.free_tensor(tensor).expect("free");
                }
            }
        }
        gpu.free_tensor(norm).expect("free");
        gpu.free_tensor(gate_weight).expect("free");
    }

    /// HC streams stored as BF16 bits (the Qwen4 F16 prefill route) must give
    /// every stream reader exactly the F32 stream's result: the norm, the
    /// norm-gate and the write all round the stream to BF16 on load, and the
    /// write's BF16 output is its F32 output's value.
    #[test]
    fn hc_bf16_streams_match_f32_streams() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (rows, branches, hidden) = (21usize, 4usize, 2560usize);
        let wide = branches * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bf16 = |gpu: &mut Gpu, values: &[f32]| -> GpuTensor {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut tensor = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16");
            tensor.dtype = DType::BF16;
            tensor.shape = vec![values.len()];
            tensor
        };
        // Arbitrary F32 stream values (not BF16-exact), and their RNE BF16
        // bits in the first half of an F32-typed buffer.
        let state = wave(1, rows * wide, 3.0);
        let mut state_bytes: Vec<u8> = state
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        state_bytes.resize(rows * wide * 4, 0);
        let f32_state = gpu.upload_f32(&state, &[state.len()]).expect("state");
        let mut bf16_state = gpu
            .upload_raw(&state_bytes, &[state_bytes.len()])
            .expect("bf16 state");
        bf16_state.dtype = DType::F32;
        bf16_state.shape = vec![rows * wide];
        let norm = bf16(&mut gpu, &wave(2, wide, 0.5));
        let gate_weight = bf16(&mut gpu, &wave(3, branches * wide, 0.05));
        let mixed = gpu
            .upload_f32(&wave(4, rows * hidden, 2.0), &[rows * hidden])
            .expect("mixed");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let mut results = Vec::new();
        for (input, state_bf16) in [(&f32_state, false), (&bf16_state, true)] {
            let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
            hyper_norm(
                &mut gpu,
                &HyperNorm {
                    input,
                    norm_weight: &norm,
                    normalized: &normalized,
                    branches,
                    hidden,
                    state_bf16,
                },
            )
            .expect("norm");
            let gates = gpu.zeros(&[rows * branches], DType::F32).expect("gates");
            hyper_norm_gate(
                &mut gpu,
                &HyperNormGate {
                    input,
                    norm_weight: &norm,
                    gate_weight: &gate_weight,
                    gates: &gates,
                    rows,
                    branches,
                    hidden,
                    state_bf16,
                },
            )
            .expect("norm-gate");
            hyper_write(
                &mut gpu,
                &HyperWrite {
                    input,
                    normalized: &normalized,
                    mixed: &mixed,
                    gates: &gates,
                    output: input,
                    branches,
                    hidden,
                    state_bf16,
                },
            )
            .expect("write");
            let mut written = bits(&gpu, input);
            if state_bf16 {
                // Widen the BF16 halves to the F32 values they encode.
                written = written
                    .iter()
                    .flat_map(|w| [w << 16, w & 0xFFFF_0000])
                    .take(rows * wide)
                    .collect();
            }
            results.push((bits(&gpu, &normalized), bits(&gpu, &gates), written));
            gpu.free_tensor(normalized).expect("free");
            gpu.free_tensor(gates).expect("free");
        }
        assert!(results[0].1.iter().any(|v| *v != 0), "gates are all zero");
        assert_eq!(results[0].0, results[1].0, "normalized rows differ");
        assert_eq!(results[0].1, results[1].1, "gates differ");
        assert_eq!(results[0].2, results[1].2, "written streams differ");
        for tensor in [f32_state, bf16_state, norm, gate_weight, mixed] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// HC-write x8 auto-select oracle: the public [`hyper_write`] candidate must
    /// equal the unchanged x2 production kernel FULL BF16 bytes (not an
    /// F32-widened proxy), over rows {1,15,16,17,37,64,131,530,16384} (the
    /// 15/16/17 neighbors bracket the measured rows>=16 crossover) x
    /// {in-place, disjoint} x {random, adversarial} payloads, with untouched
    /// guard tails, deterministic repeats, a one-bit negative control, and a
    /// real HIP-graph capture + two replays kept byte-equal to eager.
    /// The incumbent reference always forces
    /// [`HyperWriteBf16Kernel::X2`]; the candidate, repeat and capture always
    /// go through public [`hyper_write`]. `#[ignore]`: needs an exact gfx1201
    /// GPU.
    /// `cargo test -p rdna-compute --lib hyper_write_bf16x8_prototype_matches_x2 -- --ignored --nocapture`
    #[test]
    #[ignore = "needs an exact gfx1201 GPU: cargo test -p rdna-compute --lib hyper_write_bf16x8_prototype_matches_x2 -- --ignored --nocapture"]
    fn hyper_write_bf16x8_prototype_matches_x2() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if gpu.arch != "gfx1201" {
            eprintln!("skip: exact gfx1201 only");
            return;
        }
        const BRANCHES: usize = 4;
        const HIDDEN: usize = 2560;
        const WIDE: usize = BRANCHES * HIDDEN;
        const GUARD_WORDS: usize = 64;
        const ROWS: [usize; 9] = [1, 15, 16, 17, 37, 64, 131, 530, 16384];
        let rng_words = |seed: u64, n: usize| -> Vec<u32> {
            let mut state = seed;
            (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 33) as u32
                })
                .collect()
        };
        let stream_payload = |rows: usize, seed: u64, adversarial: bool| -> Vec<u8> {
            let mut words = rng_words(seed ^ (rows as u64).wrapping_mul(7919), rows * WIDE);
            if adversarial {
                // Signed zeros, infinities, mixed-sign and payload NaNs, then the
                // stream continues with random bits; every entry is a BF16 pair.
                let edge: [u32; 8] = [
                    0x0000_0000,
                    0x8000_0000,
                    0x7F80_7F80,
                    0xFF80_FF80,
                    0x7FC1_3F80,
                    0xFFC1_BF80,
                    0x7FFF_FFFF,
                    0xFFFF_0001,
                ];
                for (i, e) in edge.iter().enumerate() {
                    if i < words.len() {
                        words[i] = *e;
                    }
                }
            }
            words.iter().flat_map(|w| w.to_le_bytes()).collect()
        };
        let bytes_of = |gpu: &mut Gpu, t: &GpuTensor| -> Vec<u8> {
            gpu.hip.device_synchronize().expect("sync download");
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip
                .memcpy_dtoh(&mut out, &t.buf)
                .expect("download bytes");
            out
        };
        let check_aligned = |t: &GpuTensor, what: &str| {
            assert!(
                (t.buf.as_ptr() as usize) % 16 == 0,
                "x8 gate needs 16-byte {what} base, got {:p}",
                t.buf.as_ptr()
            );
        };
        let mk_state = |gpu: &mut Gpu, rows: usize, payload: &[u8]| -> GpuTensor {
            let mut t = gpu
                .upload_raw(payload, &[payload.len()])
                .expect("state upload");
            t.dtype = DType::F32;
            t.shape = vec![rows * WIDE];
            check_aligned(&t, "input");
            t
        };
        // Disjoint output: exact-size view over an owner with a GUARD_WORDS
        // sentinel tail; the kernel must leave the tail untouched. Returns the
        // owner (frees exactly once) and the view (never freed).
        let mk_guarded = |gpu: &mut Gpu, rows: usize| -> (GpuTensor, GpuTensor) {
            let owner = gpu
                .zeros(&[rows * WIDE + GUARD_WORDS], DType::F32)
                .expect("guarded out");
            check_aligned(&owner, "output");
            let sentinel = vec![0x5Au8; GUARD_WORDS * 4];
            gpu.hip
                .memcpy_htod_offset(&owner.buf, rows * WIDE * 4, &sentinel)
                .expect("sentinel tail");
            let view = owner.sub_offset(0, rows * WIDE);
            (owner, view)
        };
        for rows in ROWS {
            let norm = gpu.zeros(&[rows * WIDE], DType::F32).expect("normalized");
            let mixed_host: Vec<f32> = rng_words(0x2000_0000 + rows as u64, rows * HIDDEN)
                .iter()
                .map(|w| f32::from_bits(*w))
                .collect();
            let mixed = gpu
                .upload_f32(&mixed_host, &[mixed_host.len()])
                .expect("mixed");
            check_aligned(&mixed, "mixed");
            let gates_host: Vec<f32> = rng_words(0x3000_0000 + rows as u64, rows * BRANCHES)
                .iter()
                .map(|w| f32::from_bits(*w))
                .collect();
            let gates = gpu
                .upload_f32(&gates_host, &[gates_host.len()])
                .expect("gates");
            for adversarial in [false, true] {
                let payload = stream_payload(rows, 0x1000_0000, adversarial);
                let tag = if adversarial { "adversarial" } else { "random" };
                // In-place: explicit-X2 reference vs public-candidate on identical copies.
                let ref_state = mk_state(&mut gpu, rows, &payload);
                let ref_p = HyperWrite {
                    input: &ref_state,
                    normalized: &norm,
                    mixed: &mixed,
                    gates: &gates,
                    output: &ref_state,
                    branches: BRANCHES,
                    hidden: HIDDEN,
                    state_bf16: true,
                };
                launch_hyper_write_bf16(&mut gpu, &ref_p, rows, HyperWriteBf16Kernel::X2)
                    .expect("x2 production launch");
                let want = bytes_of(&mut gpu, &ref_state);
                assert!(
                    want.iter().any(|b| *b != 0),
                    "rows={rows} in-place {tag}: x2 reference wrote all zeros"
                );
                let cand_state = mk_state(&mut gpu, rows, &payload);
                let cand_p = HyperWrite {
                    input: &cand_state,
                    normalized: &norm,
                    mixed: &mixed,
                    gates: &gates,
                    output: &cand_state,
                    branches: BRANCHES,
                    hidden: HIDDEN,
                    state_bf16: true,
                };
                hyper_write(&mut gpu, &cand_p).expect("public candidate launch");
                let got = bytes_of(&mut gpu, &cand_state);
                assert!(
                    got == want,
                    "rows={rows} in-place {tag}: public candidate bytes differ from x2"
                );
                let mut bad = got.clone();
                bad[0] ^= 1;
                assert!(
                    bad != want,
                    "rows={rows} in-place {tag}: one-bit control equal"
                );
                // Deterministic repeat: refill and rerun the public candidate.
                gpu.hip
                    .memcpy_htod(&cand_state.buf, &payload)
                    .expect("refill state");
                hyper_write(&mut gpu, &cand_p).expect("public repeat");
                let repeat = bytes_of(&mut gpu, &cand_state);
                assert!(
                    repeat == want,
                    "rows={rows} in-place {tag}: public repeat differs"
                );
                for tensor in [ref_state, cand_state] {
                    gpu.free_tensor(tensor).expect("free in-place");
                }
                // Disjoint: explicit x2 vs public candidate into guarded outputs; tails must survive.
                let input = mk_state(&mut gpu, rows, &payload);
                let (ref_owner, ref_out) = mk_guarded(&mut gpu, rows);
                let (cand_owner, cand_out) = mk_guarded(&mut gpu, rows);
                let ref_dp = HyperWrite {
                    input: &input,
                    normalized: &norm,
                    mixed: &mixed,
                    gates: &gates,
                    output: &ref_out,
                    branches: BRANCHES,
                    hidden: HIDDEN,
                    state_bf16: true,
                };
                let cand_dp = HyperWrite {
                    input: &input,
                    normalized: &norm,
                    mixed: &mixed,
                    gates: &gates,
                    output: &cand_out,
                    branches: BRANCHES,
                    hidden: HIDDEN,
                    state_bf16: true,
                };
                launch_hyper_write_bf16(&mut gpu, &ref_dp, rows, HyperWriteBf16Kernel::X2)
                    .expect("x2 disjoint");
                hyper_write(&mut gpu, &cand_dp).expect("public disjoint");
                gpu.hip.device_synchronize().expect("sync disjoint");
                let ref_full = bytes_of(&mut gpu, &ref_owner);
                let cand_full = bytes_of(&mut gpu, &cand_owner);
                let split = rows * WIDE * 4;
                assert!(
                    cand_full[..split] == ref_full[..split],
                    "rows={rows} disjoint {tag}: public bytes differ from x2"
                );
                for (what, full) in [("x2", &ref_full), ("public", &cand_full)] {
                    assert!(
                        full[split..].iter().all(|b| *b == 0x5A),
                        "rows={rows} disjoint {tag}: {what} guard tail overwritten"
                    );
                }
                let mut bad = cand_full[..split].to_vec();
                bad[split / 2] ^= 1;
                assert!(
                    bad != ref_full[..split],
                    "rows={rows} disjoint {tag}: one-bit control equal"
                );
                gpu.free_tensor(input).expect("free input");
                for owner in [ref_owner, cand_owner] {
                    gpu.free_tensor(owner).expect("free guarded");
                }
            }
            for tensor in [norm, mixed, gates] {
                gpu.free_tensor(tensor).expect("free case");
            }
        }
        // Real HIP-graph capture of the public candidate (row 530 is eligible,
        // so the captured graph takes the x8 arm): eager bytes, then capture,
        // instantiate, refill, and two replays, all byte-equal. The taken blob
        // vec stays alive across both replays.
        {
            const CAP_ROWS: usize = 530;
            let payload = stream_payload(CAP_ROWS, 0xC0FFEE, true);
            let norm = gpu.zeros(&[CAP_ROWS * WIDE], DType::F32).expect("norm");
            let mixed_host: Vec<f32> = rng_words(77, CAP_ROWS * HIDDEN)
                .iter()
                .map(|w| f32::from_bits(*w))
                .collect();
            let mixed = gpu
                .upload_f32(&mixed_host, &[mixed_host.len()])
                .expect("mixed");
            let gates_host: Vec<f32> = rng_words(78, CAP_ROWS * BRANCHES)
                .iter()
                .map(|w| f32::from_bits(*w))
                .collect();
            let gates = gpu
                .upload_f32(&gates_host, &[gates_host.len()])
                .expect("gates");
            let eager = mk_state(&mut gpu, CAP_ROWS, &payload);
            let eager_p = HyperWrite {
                input: &eager,
                normalized: &norm,
                mixed: &mixed,
                gates: &gates,
                output: &eager,
                branches: BRANCHES,
                hidden: HIDDEN,
                state_bf16: true,
            };
            hyper_write(&mut gpu, &eager_p).expect("eager public");
            let eager_bytes = bytes_of(&mut gpu, &eager);
            gpu.ensure_capture_stream().expect("capture stream");
            let cap_state = mk_state(&mut gpu, CAP_ROWS, &payload);
            let cap_p = HyperWrite {
                input: &cap_state,
                normalized: &norm,
                mixed: &mixed,
                gates: &gates,
                output: &cap_state,
                branches: BRANCHES,
                hidden: HIDDEN,
                state_bf16: true,
            };
            gpu.hip.device_synchronize().expect("sync pre-capture");
            gpu.begin_stream_capture().expect("begin capture");
            let captured = hyper_write(&mut gpu, &cap_p);
            let graph = gpu.end_stream_capture().expect("end capture");
            captured.expect("captured public launch");
            let exec = gpu.hip.graph_instantiate(&graph).expect("instantiate");
            let blobs = std::mem::take(&mut gpu.graphs.capture_blobs);
            for replay in 0..2 {
                gpu.hip
                    .memcpy_htod(&cap_state.buf, &payload)
                    .expect("refill capture");
                gpu.launch_graph(&exec).expect("graph replay");
                gpu.hip.device_synchronize().expect("sync replay");
                let replayed = bytes_of(&mut gpu, &cap_state);
                assert!(
                    replayed == eager_bytes,
                    "capture replay {replay}: replayed bytes differ from eager public"
                );
            }
            let _ = blobs.len();
            gpu.hip.graph_exec_destroy(exec).expect("destroy exec");
            gpu.hip.graph_destroy(graph).expect("destroy graph");
            for tensor in [eager, cap_state, norm, mixed, gates] {
                gpu.free_tensor(tensor).expect("free capture");
            }
        }
    }
    /// The fused HC read tail (BF16 up projection + branch mix) must equal
    /// the multi-row BF16 GEMM followed by hyper_read_projected bit for bit.
    #[test]
    fn hyper_read_up_fused_is_bit_identical_to_gemm_then_read() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        let (rows, hidden, low_rank) = (131usize, 2560usize, 320usize);
        let wide = 4 * hidden;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bytes: Vec<u8> = wave(1, wide * low_rank, 0.2)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let mut up_weight = gpu.upload_raw(&bytes, &[bytes.len()]).expect("up weight");
        up_weight.dtype = DType::BF16;
        up_weight.shape = vec![wide * low_rank];
        // BF16 values, as hc_activation leaves them (the WMMA read needs it).
        let low_values: Vec<f32> = wave(2, rows * low_rank, 1.5)
            .iter()
            .map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000))
            .collect();
        let low = gpu
            .upload_f32(&low_values, &[rows * low_rank])
            .expect("low");
        let normalized = gpu
            .upload_f32(&wave(3, rows * wide, 2.0), &[rows * wide])
            .expect("normalized");
        let fused = gpu.zeros(&[rows * hidden], DType::F32).expect("fused");
        hyper_read_up_fused(
            &mut gpu,
            &HyperReadUpFused {
                up_weight: &up_weight,
                low: &low,
                normalized: &normalized,
                mixed: &fused,
                rows,
                hidden,
                low_rank,
                normalized_bf16: false,
            },
        )
        .expect("fused");
        // The same rows as BF16 bits (RNE) in the first half of an F32
        // buffer, as hyper_norm_f16 stores them.
        let mut bf16_bytes: Vec<u8> = wave(3, rows * wide, 2.0)
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        bf16_bytes.resize(rows * wide * 4, 0);
        let mut normalized_bf16 = gpu
            .upload_raw(&bf16_bytes, &[bf16_bytes.len()])
            .expect("normalized bf16");
        normalized_bf16.dtype = DType::F32;
        normalized_bf16.shape = vec![rows * wide];
        let fused_bf16 = gpu.zeros(&[rows * hidden], DType::F32).expect("fused bf16");
        hyper_read_up_fused(
            &mut gpu,
            &HyperReadUpFused {
                up_weight: &up_weight,
                low: &low,
                normalized: &normalized_bf16,
                mixed: &fused_bf16,
                rows,
                hidden,
                low_rank,
                normalized_bf16: true,
            },
        )
        .expect("fused bf16");
        let up = gpu.zeros(&[rows * wide], DType::F32).expect("up");
        gpu.gemm_bf16_xf32_multirow(&up_weight, &low, &up, wide, low_rank, rows)
            .expect("gemm");
        let reference = gpu.zeros(&[rows * hidden], DType::F32).expect("reference");
        let norm_weight = gpu.zeros(&[wide], DType::BF16).expect("norm weight");
        hyper_read_projected(
            &mut gpu,
            &HyperReadProjected {
                input: &normalized,
                norm_weight: &norm_weight,
                up: &up,
                normalized: &normalized,
                mixed: &reference,
                branches: 4,
                hidden,
            },
        )
        .expect("read");
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &fused), bits(&gpu, &reference));
        assert!(b.iter().any(|v| *v != 0), "mix is all zero");
        assert_eq!(a, b, "fused HC read differs");
        assert_eq!(
            bits(&gpu, &fused_bf16),
            b,
            "BF16-normalized HC read differs"
        );
        let wmma = gpu.zeros(&[rows * hidden], DType::F32).expect("wmma");
        if gpu.arch_caps.has_wmma_w32() || gpu.qwen4_f16_wmma_gfx1201() {
            // hyper_norm_f16's F16 copy: the BF16-rounded values, exact in F16
            // (all nonzero |v| here are normal F16s).
            let f16_bytes: Vec<u8> = wave(3, rows * wide, 2.0)
                .iter()
                .flat_map(|v| {
                    let u = v.to_bits();
                    let b = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000;
                    let sign = ((b >> 16) & 0x8000) as u16;
                    let bits = if b & 0x7FFF_FFFF == 0 {
                        sign
                    } else {
                        let exp = ((b >> 23) & 0xFF) as u16 + 15 - 127;
                        sign | (exp << 10) | ((b >> 13) & 0x3FF) as u16
                    };
                    bits.to_le_bytes()
                })
                .collect();
            let mut normalized_f16 = gpu
                .upload_raw(&f16_bytes, &[f16_bytes.len()])
                .expect("normalized f16");
            normalized_f16.dtype = DType::F16;
            normalized_f16.shape = vec![rows * wide];
            // `low` as packed BF16 bits, as hc_activation's bf16_out holds it.
            let low_bytes: Vec<u8> = low_values
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut low_bf16 = gpu
                .upload_raw(&low_bytes, &[low_bytes.len()])
                .expect("low bf16");
            low_bf16.dtype = DType::BF16;
            low_bf16.shape = vec![rows * low_rank];
            hyper_read_up_wmma(
                &mut gpu,
                &HyperReadUpFused {
                    up_weight: &up_weight,
                    low: &low_bf16,
                    normalized: &normalized_f16,
                    mixed: &wmma,
                    rows,
                    hidden,
                    low_rank,
                    normalized_bf16: true,
                },
            )
            .expect("wmma");
            // WMMA's F32 summation order may round a gate one BF16 step apart
            // from the chain's; anything more is a layout or indexing error.
            let reference = gpu.download_f32(&fused).expect("download");
            let got = gpu.download_f32(&wmma).expect("download");
            let far = reference
                .iter()
                .zip(&got)
                .filter(|(r, w)| (*r - *w).abs() > r.abs().max(w.abs()) / 64.0 + 1e-6)
                .count();
            let differ = reference.iter().zip(&got).filter(|(r, w)| r != w).count();
            assert_eq!(far, 0, "WMMA HC read beyond one BF16 step");
            assert!(
                differ * 100 < reference.len(),
                "WMMA HC read: {differ} differ"
            );
            // H3: the retiled operand-swapped entries are bytewise the
            // baseline entries' (also at the ragged row count 131).
            let tiled_out = gpu.zeros(&[rows * hidden], DType::F32).expect("tiled");
            hyper_read_up_wmma_tiled(
                &mut gpu,
                &HyperReadUpFused {
                    up_weight: &up_weight,
                    low: &low_bf16,
                    normalized: &normalized_f16,
                    mixed: &tiled_out,
                    rows,
                    hidden,
                    low_rank,
                    normalized_bf16: true,
                },
                true,
            )
            .expect("tiled wmma");
            assert_eq!(
                bits(&gpu, &tiled_out),
                bits(&gpu, &wmma),
                "retiled HC read differs from the baseline entry"
            );
            gpu.free_tensor(tiled_out).expect("free");
            gpu.free_tensor(normalized_f16).expect("free");
            gpu.free_tensor(low_bf16).expect("free");
        }
        for tensor in [
            up_weight,
            low,
            normalized,
            normalized_bf16,
            fused_bf16,
            fused,
            wmma,
            up,
            reference,
            norm_weight,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    #[test]
    fn hc_up_v2_predicate_covers_arch_shape_flags_and_boundaries() {
        let ok = |arch, en, tile, k, hidden, rows, rec, cap| {
            hc_up_v2_selected(arch, en, tile, k, hidden, rows, rec, cap)
        };
        let min = HC_UP_V2_MIN_ROWS;
        assert!(ok("gfx1201", true, true, 320, 2560, min, false, false));
        assert!(ok("gfx1201", true, true, 320, 2560, 16384, false, false));
        assert!(!ok("gfx1201", true, true, 320, 2560, min - 1, false, false));
        assert!(!ok("gfx1201", true, true, 320, 2560, 0, false, false));
        assert!(!ok("gfx1201", false, true, 320, 2560, 4096, false, false));
        assert!(!ok("gfx1201", true, false, 320, 2560, 4096, false, false));
        assert!(!ok("gfx1201", true, true, 320, 2560, 4096, true, false));
        assert!(!ok("gfx1201", true, true, 320, 2560, 4096, false, true));
        assert!(!ok("gfx1201", true, true, 256, 2560, 4096, false, false));
        assert!(!ok("gfx1201", true, true, 384, 2560, 4096, false, false));
        assert!(!ok("gfx1201", true, true, 320, 2568, 4096, false, false));
        assert!(!ok("gfx1201", true, true, 320, 0, 4096, false, false));
        assert!(ok("gfx1151", true, true, 320, 2560, min, false, false));
        assert!(!ok("gfx1151", true, false, 320, 2560, 4096, false, false));
        assert!(!ok("gfx1151", true, true, 320, 2560, min - 1, false, false));
        for arch in ["gfx1150", "gfx1200", "gfx1100", "gfx1030", "gfx1201x", ""] {
            assert!(
                !ok(arch, true, true, 320, 2560, 4096, false, false),
                "{arch}"
            );
        }
        assert_eq!(hc_up_v2_1151_groups(0), 1);
        assert_eq!(hc_up_v2_1151_groups(4096), 1);
        assert_eq!(hc_up_v2_1151_groups(8192), 2);
        assert_eq!(hc_up_v2_1151_groups(16384), 4);
        assert_eq!(hc_up_v2_1151_groups(1 << 20), 4);
        assert_eq!(hc_up_v2_groups(0), 1);
        assert_eq!(hc_up_v2_groups(1), 1);
        assert_eq!(hc_up_v2_groups(8192), 1);
        assert_eq!(hc_up_v2_groups(8193), 2);
        assert_eq!(hc_up_v2_groups(16384), 2);
        assert_eq!(hc_up_v2_groups(32768), 4);
        assert_eq!(hc_up_v2_groups(1 << 20), 4);
    }

    /// The persistent `_v2` HC read tail is bytewise the retiled `_t128` entry's
    /// (gfx1201), on random and on special (NaN, inf, zero, subnormal) operand bits.
    #[test]
    fn hyper_read_up_v2_is_bytewise_t128() {
        hyper_read_up_v2_bytewise("gfx1201");
    }

    /// gfx1151 twin: `_gfx1151_v2` is bytewise the retiled `_swap` entry's.
    #[test]
    fn hyper_read_up_v2_is_bytewise_swap_gfx1151() {
        hyper_read_up_v2_bytewise("gfx1151");
    }

    fn hyper_read_up_v2_bytewise(want_arch: &str) {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let admitted = if want_arch == "gfx1201" {
            gpu.arch == "gfx1201" && gpu.qwen4_f16_wmma_gfx1201()
        } else {
            gpu.arch == want_arch
        };
        if !admitted {
            eprintln!("skip: needs exact {want_arch}");
            return;
        }
        let (hidden, low_rank) = (2560usize, 320usize);
        let wide = 4 * hidden;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        const BF16_SPECIAL: [u16; 8] = [
            0x7fc0, 0xff80, 0x7f80, 0x0000, 0x8000, 0x0001, 0x807f, 0x7f7f,
        ];
        const F16_SPECIAL: [u16; 8] = [
            0x7e00, 0xfc00, 0x7c00, 0x0000, 0x8000, 0x0001, 0x03ff, 0xfbff,
        ];
        let bf16 = |next: &mut dyn FnMut() -> u64, special: bool| -> u16 {
            let r = next();
            if special && r % 97 == 0 {
                return BF16_SPECIAL[(r >> 20) as usize % BF16_SPECIAL.len()];
            }
            let sign = ((r >> 8) & 1) as u16;
            let exp = 120 + ((r >> 9) % 8) as u16;
            (sign << 15) | (exp << 7) | ((r >> 16) & 0x7f) as u16
        };
        let f16 = |next: &mut dyn FnMut() -> u64, special: bool| -> u16 {
            let r = next();
            if special && r % 53 == 0 {
                return F16_SPECIAL[(r >> 20) as usize % F16_SPECIAL.len()];
            }
            let sign = ((r >> 8) & 1) as u16;
            let exp = 12 + ((r >> 9) % 6) as u16;
            (sign << 15) | (exp << 10) | ((r >> 16) & 0x3ff) as u16
        };
        let upload = |gpu: &Gpu, bits: &[u16], dtype: DType| -> GpuTensor {
            let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
            let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("upload");
            t.dtype = dtype;
            t.shape = vec![bits.len()];
            t
        };
        for special in [false, true] {
            let w: Vec<u16> = (0..wide * low_rank)
                .map(|_| bf16(&mut next, special))
                .collect();
            let up_weight = upload(&gpu, &w, DType::BF16);
            for rows in [1usize, 16, 129, 4099] {
                let l: Vec<u16> = (0..rows * low_rank)
                    .map(|_| bf16(&mut next, special))
                    .collect();
                let n: Vec<u16> = (0..rows * wide).map(|_| f16(&mut next, special)).collect();
                let low = upload(&gpu, &l, DType::BF16);
                let normalized = upload(&gpu, &n, DType::F16);
                let mut outs = Vec::new();
                for route in [HcUpRoute::T128, HcUpRoute::V2] {
                    let mixed = gpu.zeros(&[rows * hidden], DType::F32).expect("mixed");
                    hyper_read_up_wmma_route(
                        &mut gpu,
                        &HyperReadUpFused {
                            up_weight: &up_weight,
                            low: &low,
                            normalized: &normalized,
                            mixed: &mixed,
                            rows,
                            hidden,
                            low_rank,
                            normalized_bf16: true,
                        },
                        route,
                    )
                    .expect("route launch");
                    let bits: Vec<u32> = gpu
                        .download_f32(&mixed)
                        .expect("download")
                        .iter()
                        .map(|v| v.to_bits())
                        .collect();
                    gpu.free_tensor(mixed).expect("free");
                    outs.push(bits);
                }
                assert!(
                    outs[0] == outs[1],
                    "v2 differs from t128 (rows {rows}, special {special})"
                );
                gpu.free_tensor(low).expect("free");
                gpu.free_tensor(normalized).expect("free");
            }
            gpu.free_tensor(up_weight).expect("free");
        }
    }

    /// HC K4 gates and read-down reproduce singleton bytes at every logical
    /// width 1..64, with gates-only and HC activation epilogues separately.
    #[test]
    fn gemv_bf16_k4_rows_are_bit_identical_to_single_row() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let max_rows = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        gpu.set_wide_verify_rows(max_rows);
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 131) % 8191;
                    (h as f32 - 4095.0) / 4095.0 * scale
                })
                .collect()
        };
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        // (M, K): HC write gates (4 x 4*2560) and an HC read down (320 x 4*2560).
        for (m, k) in [(4usize, 10240usize), (320, 10240)] {
            let bytes: Vec<u8> = wave(1, m * k, 0.05)
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect();
            let mut weight = gpu.upload_raw(&bytes, &[bytes.len()]).expect("weight");
            weight.dtype = DType::BF16;
            weight.shape = vec![m * k];
            let x = gpu
                .upload_f32(&wave(2, max_rows * k, 1.5), &[max_rows * k])
                .expect("x");
            for act in [None, Some(0.25f32)] {
                let single = gpu.zeros(&[max_rows * m], DType::F32).expect("single");
                for row in 0..max_rows {
                    gpu.gemv_bf16_xf32_k4(
                        &weight,
                        &x.sub_offset(row * k, k),
                        &single.sub_offset(row * m, m),
                        m,
                        k,
                        act,
                    )
                    .expect("k4");
                }
                let expected = bits(&gpu, &single);
                assert!(
                    expected.iter().any(|v| *v != 0),
                    "single-row output is all zero"
                );
                for rows in 1..=max_rows {
                    let multi = gpu.zeros(&[rows * m], DType::F32).expect("multi");
                    gpu.gemv_bf16_xf32_k4_rows(&weight, &x, &multi, m, k, act, rows)
                        .expect("k4 rows");
                    assert_eq!(
                        bits(&gpu, &multi),
                        expected[..rows * m],
                        "M={m} K={k} rows={rows} act={act:?}"
                    );
                    gpu.free_tensor(multi).expect("free");
                }
                gpu.free_tensor(single).expect("free");
            }
            gpu.free_tensor(x).expect("free");
            gpu.free_tensor(weight).expect("free");
        }
    }

    #[cfg(feature = "lab")]
    fn g1_launched_identity(gpu: &Gpu, context: std::fmt::Arguments<'_>) {
        let symbol = gpu
            .last_launched_kernel()
            .expect("G1: no actual launched kernel");
        let (module, container_sha256, object_sha256) = gpu
            .lab_kernel_device_identity(symbol)
            .expect("G1: launched kernel has no admitted device identity");
        eprintln!(
            "G1 arch={} module={module} symbol={symbol} container_sha256={container_sha256} object_sha256={object_sha256} {context}",
            gpu.arch
        );
    }

    #[test]
    fn hyper_write_norm_wide_rows_equal_singleton_and_unfused() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let max_rows = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        let (hidden, branches) = (2560usize, 4usize);
        let wide = hidden * branches;
        let input = gpu
            .upload_f32(&test_wave(201, max_rows * wide, 3.0), &[max_rows * wide])
            .expect("input");
        let mixed = gpu
            .upload_f32(
                &test_wave(203, max_rows * hidden, 2.0),
                &[max_rows * hidden],
            )
            .expect("mixed");
        let mut gate_values = test_wave(205, max_rows * branches, 8.0);
        for (i, gate) in gate_values.iter_mut().enumerate() {
            if i % 7 == 0 {
                *gate = 32.0;
            }
            if i % 11 == 0 {
                *gate = -32.0;
            }
        }
        let gates = gpu
            .upload_f32(&gate_values, &[gate_values.len()])
            .expect("gates");
        let quarters = gpu
            .upload_f32(&test_wave(207, max_rows * 16, 8.0), &[max_rows * 16])
            .expect("quarters");
        let norm = upload_bf16_values(&mut gpu, &test_wave(209, wide, 0.5));
        let inject = upload_bf16_values(&mut gpu, &test_wave(211, branches * wide, 0.05));
        let alloc = |gpu: &mut Gpu, n| {
            gpu.upload_f32(&vec![123.25; n], &[n])
                .expect("poison output")
        };
        for rows in 1..=max_rows {
            for use_quarters in [false, true] {
                for paired in [false, true] {
                    let output = alloc(&mut gpu, (rows + 1) * wide);
                    let normalized = alloc(&mut gpu, (rows + 1) * wide);
                    let next_quarters = alloc(&mut gpu, (rows + 1) * 16);
                    let single_output = alloc(&mut gpu, (rows + 1) * wide);
                    let single_norm = alloc(&mut gpu, (rows + 1) * wide);
                    let single_next = alloc(&mut gpu, (rows + 1) * 16);
                    let clear = alloc(&mut gpu, rows * hidden);
                    let single_clear = alloc(&mut gpu, rows * hidden);
                    let launch = |gpu: &mut Gpu,
                                  first: usize,
                                  count: usize,
                                  out: &GpuTensor,
                                  norm_out: &GpuTensor,
                                  next_out: &GpuTensor,
                                  clear_out: &GpuTensor| {
                        let next_view = next_out.sub_offset(first * 16, count * 16);
                        let next = HyperNextGates {
                            norm_weight: &norm,
                            inject: &inject,
                            quarters: &next_view,
                        };
                        let q = quarters.sub_offset(first * 16, count * 16);
                        hyper_write_norm(
                            gpu,
                            &HyperWrite {
                                input: &input.sub_offset(first * wide, count * wide),
                                normalized: &input.sub_offset(first * wide, count * wide),
                                mixed: &mixed.sub_offset(first * hidden, count * hidden),
                                gates: &gates.sub_offset(first * branches, count * branches),
                                output: &out.sub_offset(first * wide, count * wide),
                                branches,
                                hidden,
                                state_bf16: false,
                            },
                            &norm,
                            &norm_out.sub_offset(first * wide, count * wide),
                            use_quarters.then_some(&q),
                            paired.then_some(&next),
                            Some(&clear_out.sub_offset(first * hidden, count * hidden)),
                        )
                        .expect("HC fused");
                        #[cfg(feature = "lab")]
                        g1_launched_identity(gpu, format_args!(
                            "case=hyper_write_norm mode=eager format=F32 R={count} position={first} hidden={hidden} branches={branches} quarters={use_quarters} paired={paired} input_bytes={} output_bytes={} normalized_bytes={} clear_bytes={}",
                            count * wide * 4, count * wide * 4, count * wide * 4, count * hidden * 4
                        ));
                    };
                    launch(
                        &mut gpu,
                        0,
                        rows,
                        &output,
                        &normalized,
                        &next_quarters,
                        &clear,
                    );
                    for row in 0..rows {
                        launch(
                            &mut gpu,
                            row,
                            1,
                            &single_output,
                            &single_norm,
                            &single_next,
                            &single_clear,
                        );
                    }
                    for (a, b) in [
                        (&output, &single_output),
                        (&normalized, &single_norm),
                        (&next_quarters, &single_next),
                        (&clear, &single_clear),
                    ] {
                        assert_eq!(
                            download_bytes(&gpu, a),
                            download_bytes(&gpu, b),
                            "HC rows={rows} quarters={use_quarters} paired={paired}"
                        );
                    }
                    assert!(gpu
                        .download_f32(&clear)
                        .expect("clear")
                        .iter()
                        .all(|x| x.to_bits() == 0));
                    for (t, stride) in [(&output, wide), (&normalized, wide), (&next_quarters, 16)]
                    {
                        assert!(gpu
                            .download_f32(&t.sub_offset(rows * stride, stride))
                            .expect("canary")
                            .iter()
                            .all(|x| *x == 123.25));
                    }
                    if !use_quarters {
                        let reference = alloc(&mut gpu, rows * wide);
                        let reference_norm = alloc(&mut gpu, rows * wide);
                        hyper_write(
                            &mut gpu,
                            &HyperWrite {
                                input: &input.sub_offset(0, rows * wide),
                                normalized: &input.sub_offset(0, rows * wide),
                                mixed: &mixed.sub_offset(0, rows * hidden),
                                gates: &gates.sub_offset(0, rows * branches),
                                output: &reference,
                                branches,
                                hidden,
                                state_bf16: false,
                            },
                        )
                        .expect("HC write");
                        #[cfg(feature = "lab")]
                        g1_launched_identity(&gpu, format_args!(
                            "case=hyper_write_norm role=unfused_write mode=eager format=F32 R={rows} position=0 hidden={hidden} branches={branches} output_bytes={}",
                            reference.byte_size()
                        ));
                        hyper_norm(
                            &mut gpu,
                            &HyperNorm {
                                input: &reference,
                                norm_weight: &norm,
                                normalized: &reference_norm,
                                branches,
                                hidden,
                                state_bf16: false,
                            },
                        )
                        .expect("HC norm");
                        #[cfg(feature = "lab")]
                        g1_launched_identity(&gpu, format_args!(
                            "case=hyper_write_norm role=unfused_norm mode=eager format=F32 R={rows} position=0 hidden={hidden} branches={branches} output_bytes={}",
                            reference_norm.byte_size()
                        ));
                        assert_eq!(
                            download_bytes(&gpu, &reference),
                            download_bytes(&gpu, &output.sub_offset(0, rows * wide))
                        );
                        assert_eq!(
                            download_bytes(&gpu, &reference_norm),
                            download_bytes(&gpu, &normalized.sub_offset(0, rows * wide))
                        );
                        gpu.free_tensor(reference).expect("free");
                        gpu.free_tensor(reference_norm).expect("free");
                    }
                    for t in [
                        output,
                        normalized,
                        next_quarters,
                        single_output,
                        single_norm,
                        single_next,
                        clear,
                        single_clear,
                    ] {
                        gpu.free_tensor(t).expect("free");
                    }
                }
            }
        }
        for t in [input, mixed, gates, quarters, norm, inject] {
            gpu.free_tensor(t).expect("free");
        }
    }

    /// The persistent row-batched GDN recurrence must equal the per-row step
    /// kernel bit for bit (outputs and final state), including across the
    /// kernel's 256-row prologue block boundary.
    #[test]
    fn gdn_persistent_batch_is_bit_identical_to_per_row_steps() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 300usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta: Vec<f32> = wave(3, rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let state0 = wave(4, value * dim, 0.2);
        let proj_gpu = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");

        let batched_state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
        let batched_out = gpu.zeros(&[rows * value], DType::F32).expect("output");
        gated_delta_step_batched(
            &mut gpu,
            &GatedDeltaStepBatched {
                projection: &proj_gpu,
                gate: &gate_gpu,
                beta: &beta_gpu,
                state: &batched_state,
                format: GdnStateFormat::F32,
                residual: None,
                output: &batched_out,
                row_states: None,
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            },
        )
        .expect("batched GDN");

        let row_state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
        let row_out = gpu.zeros(&[rows * value], DType::F32).expect("output");
        for row in 0..rows {
            gated_delta_step(
                &mut gpu,
                &GatedDeltaStep {
                    q: &proj_gpu.sub_offset(row * qkv, qk),
                    k: &proj_gpu.sub_offset(row * qkv + qk, qk),
                    v: &proj_gpu.sub_offset(row * qkv + 2 * qk, value),
                    gate: &gate_gpu.sub_offset(row * value_heads, value_heads),
                    beta: &beta_gpu.sub_offset(row * value_heads, value_heads),
                    state: &row_state,
                    format: GdnStateFormat::F32,
                    residual: None,
                    output: &row_out.sub_offset(row * value, value),
                    key_heads,
                    value_heads,
                    key_dim: dim,
                    value_dim: dim,
                    position: row,
                },
            )
            .expect("per-row GDN");
        }
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let (a, b) = (bits(&gpu, &batched_out), bits(&gpu, &row_out));
        assert!(a.iter().any(|v| *v != 0), "batched output is all zero");
        let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        assert_eq!(differing, 0, "GDN outputs differ in {differing} cells");
        assert_eq!(
            bits(&gpu, &batched_state),
            bits(&gpu, &row_state),
            "GDN final state differs"
        );
        for tensor in [
            proj_gpu,
            gate_gpu,
            beta_gpu,
            batched_state,
            batched_out,
            row_state,
            row_out,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The register-resident `gated_delta_step_pipe128_{f32,q8}` must equal
    /// the persistent256 kernel byte for byte (output, final F32 state, and
    /// the Q8 codes and scales), across the 256-row parameter-block boundary
    /// and ragged row counts, on a nonzero random start state at a nonzero
    /// position.  Both kernels are launched by name, so the
    /// `HIPFIRE_QWEN4_GDN_PIPE` selection is not under test.  gfx1201 runs
    /// the F32 and Q8 kernels over rows 1..=1000; gfx942 has only the F32
    /// kernel (its persistent256 F32 arithmetic is `__fmaf_rn`-pinned) and
    /// adds a 1536-row case.  Other archs skip.
    #[test]
    fn gdn_pipe128_is_bytewise_the_persistent256_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let is_gfx942 = gpu.arch == "gfx942";
        if gpu.arch != "gfx1201" && !is_gfx942 {
            eprintln!("skip: gfx1201 / gfx942 only (arch {})", gpu.arch);
            return;
        }
        let (key_heads, value_heads, dim) = (16usize, 48usize, 128usize);
        let row_counts: &[usize] = if is_gfx942 {
            &[1, 2, 3, 255, 256, 257, 511, 513, 1000, 1536]
        } else {
            &[1, 2, 3, 255, 256, 257, 511, 513, 1000]
        };
        let max_rows = *row_counts.last().expect("row counts");
        let q8_arms: &[bool] = if is_gfx942 { &[false] } else { &[false, true] };
        let position = 37usize;
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let slot_bytes = GdnStateFormat::Q8.state_units(value_heads, dim, dim);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut unit = move || -> f32 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let projection: Vec<f32> = (0..max_rows * qkv)
            .map(|_| (unit() * 2.0 - 1.0) * 1.5)
            .collect();
        let gate: Vec<f32> = (0..max_rows * value_heads).map(|_| -3.0 * unit()).collect();
        let beta: Vec<f32> = (0..max_rows * value_heads).map(|_| unit()).collect();
        let state0: Vec<f32> = (0..value * dim)
            .map(|_| (unit() * 2.0 - 1.0) * 0.2)
            .collect();
        let proj_gpu = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let slot0 = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        if !is_gfx942 {
            gdn_state_from_f32(
                &mut gpu,
                GdnStateFormat::Q8,
                state0_gpu.buf.as_ptr(),
                slot0.buf.as_ptr(),
                None,
                value_heads,
                7,
            )
            .expect("quantize");
        }

        let launch = |gpu: &mut Gpu,
                      kernel: &str,
                      threads: u32,
                      q8: bool,
                      state: &GpuTensor,
                      output: &GpuTensor,
                      rows: usize| {
            gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)
                .expect("kernel");
            let mut args = KernargBlob::new();
            for tensor in [&proj_gpu, &gate_gpu, &beta_gpu, state, output] {
                args.push_ptr(tensor.buf.as_ptr());
            }
            args.push_i32(rows as i32);
            args.push_i32(qkv as i32);
            args.push_i32(key_heads as i32);
            args.push_i32(value_heads as i32);
            args.push_i32(dim as i32);
            args.push_i32(dim as i32);
            args.push_f32((dim as f32).sqrt().recip());
            if q8 {
                args.push_u32((position + rows - 1) as u32);
            }
            args.pad_to(16);
            gpu.launch_blob_recorded(
                kernel,
                [value_heads as u32, 1, 1],
                [threads, 1, 1],
                0,
                args.as_mut_slice(),
                crate::dispatch::ReplayLaunchBindings::NONE,
            )
            .expect("launch");
        };
        for &rows in row_counts {
            for &q8 in q8_arms {
                let fresh = |gpu: &mut Gpu| -> GpuTensor {
                    if q8 {
                        let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
                        gpu.copy_d2d(&slot0, &slot, slot_bytes).expect("copy slot");
                        slot
                    } else {
                        gpu.upload_f32(&state0, &[state0.len()]).expect("state")
                    }
                };
                let (old_kernel, new_kernel) = if q8 {
                    (
                        "gated_delta_step_halves_state128_persistent256_q8",
                        "gated_delta_step_pipe128_q8",
                    )
                } else {
                    (
                        "gated_delta_step_halves_state128_persistent256_f32",
                        "gated_delta_step_pipe128_f32",
                    )
                };
                let old_state = fresh(&mut gpu);
                let old_out = gpu.zeros(&[rows * value], DType::F32).expect("output");
                launch(&mut gpu, old_kernel, 256, q8, &old_state, &old_out, rows);
                let new_state = fresh(&mut gpu);
                let new_out = gpu.zeros(&[rows * value], DType::F32).expect("output");
                launch(&mut gpu, new_kernel, 128, q8, &new_state, &new_out, rows);

                let old_out_bytes = gpu.download_raw_bytes(&old_out).expect("download");
                let new_out_bytes = gpu.download_raw_bytes(&new_out).expect("download");
                let old_state_bytes = gpu.download_raw_bytes(&old_state).expect("download");
                let new_state_bytes = gpu.download_raw_bytes(&new_state).expect("download");
                let start_bytes = if q8 {
                    gpu.download_raw_bytes(&slot0).expect("download")
                } else {
                    gpu.download_raw_bytes(&state0_gpu).expect("download")
                };
                assert!(
                    old_out_bytes.iter().any(|b| *b != 0),
                    "{rows} rows q8={q8}: reference output is all zero"
                );
                assert_ne!(
                    old_state_bytes, start_bytes,
                    "{rows} rows q8={q8}: reference left the state untouched"
                );
                assert!(
                    old_out_bytes == new_out_bytes,
                    "{rows} rows q8={q8}: pipe128 output differs from persistent256"
                );
                assert!(
                    old_state_bytes == new_state_bytes,
                    "{rows} rows q8={q8}: pipe128 final state differs from persistent256"
                );
                for tensor in [old_state, old_out, new_state, new_out] {
                    gpu.free_tensor(tensor).expect("free");
                }
            }
        }
        for tensor in [proj_gpu, gate_gpu, beta_gpu, state0_gpu, slot0] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The chunked F16 WMMA GDN recurrence + gate (Qwen4 F16 prefill route)
    /// must track the exact persistent kernel followed by the gate kernel, in
    /// gated output and final state, across its 16-row chunks and a partial
    /// last chunk.
    #[test]
    fn gdn_chunk_gate_arm_matches_persistent_kernel_and_gate() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_wmma_w32() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx11 WMMA");
            return;
        }
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta: Vec<f32> = wave(3, rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let state0 = wave(4, value * dim, 0.2);
        let z = wave(5, rows * value, 2.0);
        let norm_bytes: Vec<u8> = wave(6, dim, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let proj_gpu = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let z_gpu = gpu.upload_f32(&z, &[z.len()]).expect("z");
        let mut norm_gpu = gpu
            .upload_raw(&norm_bytes, &[norm_bytes.len()])
            .expect("norm");
        norm_gpu.dtype = DType::BF16;
        norm_gpu.shape = vec![dim];
        // The chunked arm reads the convolution output as packed BF16: the
        // same values the exact kernel rounds the F32 projection to.
        let proj_bytes: Vec<u8> = projection
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        let mut proj_bf16 = gpu
            .upload_raw(&proj_bytes, &[proj_bytes.len()])
            .expect("projection bf16");
        proj_bf16.dtype = DType::BF16;
        proj_bf16.shape = vec![projection.len()];
        let run = |gpu: &mut Gpu, fused: bool| {
            let state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
            let recurrent = gpu.zeros(&[rows * value], DType::F32).expect("recurrent");
            let out = gpu.zeros(&[rows * value], DType::F32).expect("output");
            let step = GatedDeltaStepBatched {
                projection: if fused { &proj_bf16 } else { &proj_gpu },
                gate: &gate_gpu,
                beta: &beta_gpu,
                state: &state,
                format: GdnStateFormat::F32,
                residual: None,
                output: &recurrent,
                row_states: None,
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            };
            let gated = GatedDeltaGateBatched {
                recurrent_output: &recurrent,
                z: &z_gpu,
                norm: &norm_gpu,
                output: &out,
                rows,
                value_heads,
                value_dim: dim,
            };
            if fused {
                assert!(
                    gated_delta_chunk_route(gpu, &step),
                    "chunked route did not apply"
                );
                gated_delta_step_gate_wmma(gpu, &step, &gated).expect("fused GDN");
            } else {
                gated_delta_step_batched(gpu, &step).expect("batched GDN");
                gated_delta_gate_batched(gpu, &gated).expect("gate");
            }
            let values = (
                gpu.download_f32(&out).expect("download"),
                gpu.download_f32(&state).expect("download"),
            );
            for tensor in [out, recurrent, state] {
                gpu.free_tensor(tensor).expect("free");
            }
            values
        };
        let (ref_out, ref_state) = run(&mut gpu, false);
        let (col_out, col_state) = run(&mut gpu, true);
        let rel = |a: &[f32], b: &[f32]| {
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                err += (*x as f64 - *y as f64).powi(2);
                norm += (*x as f64).powi(2);
            }
            assert!(norm > 0.0, "reference is all zero");
            (err / norm).sqrt()
        };
        // F16 operands flip an occasional BF16 rounding of the gated output
        // (~2e-3 relative) and leave the state ~3e-4 off; a wrong column, row,
        // head, chunk boundary or gate lands near 1.
        let (out_rel, state_rel) = (rel(&ref_out, &col_out), rel(&ref_state, &col_state));
        assert!(out_rel < 1e-2, "GDN chunk gate output rel L2 {out_rel:.3e}");
        assert!(state_rel < 2e-3, "GDN chunk state rel L2 {state_rel:.3e}");
        for tensor in [proj_gpu, proj_bf16, gate_gpu, beta_gpu, z_gpu, norm_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// Sequential singleton decode from one nonzero live state; retain every
    /// state for captured-window and prefix-repair byte comparisons.
    #[allow(clippy::too_many_arguments)]
    fn gdn_sequential(
        gpu: &mut Gpu,
        projection: &GpuTensor,
        gate: &GpuTensor,
        beta: &GpuTensor,
        slot: &GpuTensor,
        n: usize,
        start: usize,
        key_heads: usize,
        value_heads: usize,
        format: GdnStateFormat,
    ) -> (Vec<f32>, Vec<Vec<u8>>) {
        let (qk, value) = (key_heads * 128, value_heads * 128);
        let qkv = 2 * qk + value;
        let out = gpu.zeros(&[n * value], DType::F32).expect("output");
        let mut states = Vec::with_capacity(n);
        for row in 0..n {
            gated_delta_step(
                gpu,
                &GatedDeltaStep {
                    q: &projection.sub_offset(row * qkv, qk),
                    k: &projection.sub_offset(row * qkv + qk, qk),
                    v: &projection.sub_offset(row * qkv + 2 * qk, value),
                    gate: &gate.sub_offset(row * value_heads, value_heads),
                    beta: &beta.sub_offset(row * value_heads, value_heads),
                    state: slot,
                    format,
                    residual: None,
                    output: &out.sub_offset(row * value, value),
                    key_heads,
                    value_heads,
                    key_dim: 128,
                    value_dim: 128,
                    position: start + row,
                },
            )
            .expect("GDN step");
            #[cfg(feature = "lab")]
            g1_launched_identity(gpu, format_args!(
                "case=gdn_sequential role=singleton mode=eager format={format:?} R=1 position={} key_heads={key_heads} value_heads={value_heads} dim=128 projection_bytes={} output_bytes={} state_bytes={}",
                start + row, qkv * 4, value * 4, slot.byte_size()
            ));
            gpu.hip.device_synchronize().expect("sync");
            let mut bytes = vec![0u8; slot.byte_size()];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &slot.buf)
                .expect("download");
            states.push(bytes);
        }
        let values = gpu.download_f32(&out).expect("download");
        gpu.free_tensor(out).expect("free");
        (values, states)
    }

    /// Q8 and F32 capture/prefix repair match sequential singleton decode for
    /// every width 1..64, small and production head geometries, both ring
    /// halves and eager/graph replay from nonzero state/absolute positions.
    #[test]
    fn gdn_q8_verify_matches_single_row() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        // Field-disjoint borrows (gpu.graphs / gpu.hip) need the plain `&mut Gpu`.
        let gpu = &mut *gpu;
        for format in [GdnStateFormat::Q8, GdnStateFormat::F32] {
            for (key_heads, value_heads) in [(2usize, 6usize), (16, 16), (16, 48)] {
                let dim = 128usize;
                let max_rows = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
                for start in [1157usize, 8191usize] {
                    let qk = key_heads * dim;
                    let value = value_heads * dim;
                    let qkv = 2 * qk + value;
                    let slot_units = format.state_units(value_heads, dim, dim);
                    let slot_bytes = slot_units * format.dtype().size();
                    let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
                        (0..n)
                            .map(|i| {
                                let h =
                                    i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                                (h as f32 - 5003.0) / 5003.0 * scale
                            })
                            .collect()
                    };
                    let projection = wave(11, max_rows * qkv, 1.5);
                    let gate: Vec<f32> = wave(12, max_rows * value_heads, 0.5)
                        .iter()
                        .map(|g| g - 0.6)
                        .collect();
                    let beta: Vec<f32> = wave(13, max_rows * value_heads, 0.45)
                        .iter()
                        .map(|b| b + 0.5)
                        .collect();
                    let state0 = wave(14, value * dim, 0.2);
                    let proj_gpu = gpu
                        .upload_f32(&projection, &[projection.len()])
                        .expect("projection");
                    let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
                    let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
                    let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
                    let slot0 = gpu.zeros(&[slot_units], format.dtype()).expect("slot");
                    if format == GdnStateFormat::Q8 {
                        gdn_state_from_f32(
                            gpu,
                            format,
                            state0_gpu.buf.as_ptr(),
                            slot0.buf.as_ptr(),
                            None,
                            value_heads,
                            start as u32 - 1,
                        )
                        .expect("quantize");
                    } else {
                        gpu.copy_d2d(&state0_gpu, &slot0, slot_bytes)
                            .expect("seed F32");
                    }
                    let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
                        gpu.hip.device_synchronize().expect("sync");
                        let mut out = vec![0u8; t.byte_size()];
                        gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
                        out
                    };
                    let seed_bytes = bytes(&*gpu, &slot0);
                    assert!(seed_bytes.iter().any(|b| *b != 0), "seed state is zero");
                    let fresh_slot = |gpu: &mut Gpu| -> GpuTensor {
                        let slot = gpu.zeros(&[slot_units], format.dtype()).expect("slot");
                        gpu.copy_d2d(&slot0, &slot, slot_bytes).expect("copy slot");
                        slot
                    };
                    gpu.ensure_capture_stream().expect("stream");

                    // Every mismatch is collected, then asserted, so one run reports
                    // exactly which rows and prefixes leave the sequential oracle.
                    let mut failures: Vec<String> = Vec::new();
                    let oracle_slot = fresh_slot(gpu);
                    let (seq_out, seq_states) = gdn_sequential(
                        gpu,
                        &proj_gpu,
                        &gate_gpu,
                        &beta_gpu,
                        &oracle_slot,
                        max_rows,
                        start,
                        key_heads,
                        value_heads,
                        format,
                    );
                    for graph in [false, true] {
                        for base in [0usize, max_rows] {
                            for n in 1..=max_rows {
                                let live = fresh_slot(gpu);
                                let out = gpu.zeros(&[n * value], DType::F32).expect("output");
                                // Two capture halves plus an immutable pre-window source slot.
                                let source = 2 * max_rows;
                                let repair_base = if base == 0 { max_rows } else { 0 };
                                let ring = gpu
                                    .zeros(&[(source + 1) * slot_units], format.dtype())
                                    .expect("ring");
                                gpu.copy_d2d(
                                    &slot0,
                                    &ring.sub_offset(source * slot_units, slot_units),
                                    slot_bytes,
                                )
                                .expect("seed ring");
                                let recurrence: Vec<f32> = projection[..n * qkv]
                                    .iter()
                                    .chain(&gate[..n * value_heads])
                                    .chain(&beta[..n * value_heads])
                                    .copied()
                                    .collect();
                                let recurrence_gpu = gpu
                                    .upload_f32(&recurrence, &[recurrence.len()])
                                    .expect("recurrence");
                                let pointers: Vec<u8> =
                                    [recurrence_gpu.buf.as_ptr(), ring.buf.as_ptr()]
                                        .iter()
                                        .flat_map(|p| (*p as u64).to_ne_bytes())
                                        .collect();
                                let table =
                                    gpu.upload_raw(&pointers, &[pointers.len()]).expect("table");
                                // One `[keep, value]` output region per kept prefix.
                                let discard =
                                    gpu.zeros(&[n * n * value], DType::F32).expect("discard");
                                let launch = |gpu: &mut Gpu| {
                                    gated_delta_step_batched(
                                        gpu,
                                        &GatedDeltaStepBatched {
                                            projection: &proj_gpu.sub_offset(0, n * qkv),
                                            gate: &gate_gpu.sub_offset(0, n * value_heads),
                                            beta: &beta_gpu.sub_offset(0, n * value_heads),
                                            state: &live,
                                            format,
                                            residual: None,
                                            output: &out,
                                            row_states: Some(
                                                &ring.sub_offset(base * slot_units, n * slot_units),
                                            ),
                                            rows: n,
                                            qkv_width: qkv,
                                            key_heads,
                                            value_heads,
                                            key_dim: dim,
                                            value_dim: dim,
                                            position: start,
                                        },
                                    )
                                    .expect("captured verify");
                                    #[cfg(feature = "lab")]
                                    g1_launched_identity(gpu, format_args!(
                                        "case=gdn_verify role=candidate mode={} format={format:?} R={n} position={start} key_heads={key_heads} value_heads={value_heads} dim={dim} base={base} projection_bytes={} output_bytes={} state_bytes={slot_bytes} ring_bytes={}",
                                        if graph { "graph_record_for_replay" } else { "eager" },
                                        n * qkv * 4, out.byte_size(), ring.byte_size()
                                    ));
                                    for keep in 1..=n {
                                        gated_delta_rollback_layers(
                                            gpu,
                                            &GatedDeltaRollbackLayers {
                                                table: &table,
                                                discard: &discard.sub_offset(
                                                    (keep - 1) * n * value,
                                                    keep * value,
                                                ),
                                                format,
                                                layers: 1,
                                                rows: n,
                                                keep,
                                                from: source,
                                                to: repair_base,
                                                qkv_width: qkv,
                                                key_heads,
                                                value_heads,
                                                position: start + keep - 1,
                                            },
                                        )
                                        .expect("rollback");
                                        #[cfg(feature = "lab")]
                                        g1_launched_identity(gpu, format_args!(
                                            "case=gdn_rollback mode={} format={format:?} R={n} keep={keep} position={} key_heads={key_heads} value_heads={value_heads} dim={dim} from={source} to={repair_base} state_bytes={slot_bytes} discard_bytes={}",
                                            if graph { "graph_record_for_replay" } else { "eager" },
                                            start + keep - 1, keep * value * 4
                                        ));
                                    }
                                };
                                if graph {
                                    // Kernels are warm from the direct pass; the capture only
                                    // records, so the buffers above are what the replay sees.
                                    let stream = gpu.active_stream.take().expect("stream");
                                    gpu.graphs
                                        .begin_graph_capture(&gpu.hip, gpu.device_id, &stream)
                                        .expect("begin capture");
                                    gpu.active_stream = Some(stream);
                                    launch(gpu);
                                    let stream = gpu.active_stream.take().expect("stream");
                                    gpu.graphs
                                        .end_graph_capture(&gpu.hip, gpu.device_id, &stream)
                                        .expect("end capture");
                                    gpu.graphs
                                        .graph_launch(&gpu.hip, gpu.device_id, &stream)
                                        .expect("replay");
                                    gpu.active_stream = Some(stream);
                                    gpu.hip.device_synchronize().expect("sync");
                                    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
                                } else {
                                    launch(gpu);
                                }

                                let tag = format!("{format:?} kh={key_heads} vh={value_heads} p={start} base={base} graph={graph} n={n}");
                                gpu.hip.device_synchronize().expect("sync");
                                let verify_out = gpu.download_f32(&out).expect("download");
                                let row_diff = |a: &[f32], b: &[f32]| -> Vec<(usize, usize)> {
                                    (0..a.len() / value)
                                        .map(|r| {
                                            let (x, y) = (
                                                &a[r * value..][..value],
                                                &b[r * value..][..value],
                                            );
                                            (
                                                r,
                                                x.iter()
                                                    .zip(y)
                                                    .filter(|(p, q)| p.to_bits() != q.to_bits())
                                                    .count(),
                                            )
                                        })
                                        .filter(|(_, d)| *d > 0)
                                        .collect()
                                };
                                let diff = row_diff(&verify_out, &seq_out);
                                if !diff.is_empty() {
                                    failures.push(format!(
                                        "{tag}: verify output (row, differing f32s) {diff:?}"
                                    ));
                                }
                                let slot_diff = |a: &[u8], b: &[u8]| {
                                    a.iter().zip(b).filter(|(p, q)| p != q).count()
                                };
                                let captured = bytes(
                                    &*gpu,
                                    &ring.sub_offset((base + n - 1) * slot_units, slot_units),
                                );
                                let d = slot_diff(&captured, &seq_states[n - 1]);
                                if d > 0 {
                                    failures.push(format!(
                                        "{tag}: captured slot differs from AR in {d} bytes"
                                    ));
                                }
                                if bytes(&*gpu, &live) != seed_bytes {
                                    failures.push(format!("{tag}: capture wrote the live state"));
                                }
                                if bytes(&*gpu, &ring.sub_offset(source * slot_units, slot_units))
                                    != seed_bytes
                                {
                                    failures.push(format!("{tag}: rollback wrote its source slot"));
                                }
                                let rollback_out = gpu.download_f32(&discard).expect("download");
                                for keep in 1..=n {
                                    let kept = bytes(
                                        &*gpu,
                                        &ring.sub_offset(
                                            (repair_base + keep - 1) * slot_units,
                                            slot_units,
                                        ),
                                    );
                                    let d = slot_diff(&kept, &seq_states[keep - 1]);
                                    if d > 0 {
                                        failures.push(format!(
                            "{tag} keep={keep}: rollback slot differs from AR in {d} bytes"
                        ));
                                    }
                                    let diff = row_diff(
                                        &rollback_out[(keep - 1) * n * value..][..keep * value],
                                        &seq_out[..keep * value],
                                    );
                                    if !diff.is_empty() {
                                        failures.push(format!(
                            "{tag} keep={keep}: rollback output (row, differing f32s) {diff:?}"
                        ));
                                    }
                                }
                                eprintln!("GDN_VERIFY_ROWS {tag} reached");
                                for tensor in [live, out, ring, recurrence_gpu, table, discard] {
                                    gpu.free_tensor(tensor).expect("free");
                                }
                            }
                        }
                    }
                    gpu.free_tensor(oracle_slot).expect("free");
                    for failure in &failures {
                        eprintln!("GDN_VERIFY_SEAM {failure}");
                    }
                    assert!(
                        failures.is_empty(),
                        "{} verify/rollback seams leave AR",
                        failures.len()
                    );
                    for tensor in [proj_gpu, gate_gpu, beta_gpu, state0_gpu, slot0] {
                        gpu.free_tensor(tensor).expect("free");
                    }
                }
            }
        }
    }

    /// Q8 GDN state (Qwen3.5's DeltaNet Q8 format): every route tracks its
    /// F32 twin (decode steps, the persistent batch, the chunked prefill),
    /// and the verify capture and a rollback re-run of its kept rows are
    /// token-exact with single-row decode (AR): the captured output and ring
    /// slot, and the rollback slot, equal decode steps from the same state at
    /// the same positions, bit for bit (the unarmed batch requantizes once,
    /// so it is not their oracle). A transposed row, a wrong scale or a slot
    /// offset lands near 1.
    #[test]
    fn gdn_q8_state_tracks_f32_on_every_route() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        // `rows` decode steps; the batch routes run every ragged row count
        // around the persistent kernel's 16-row unroll and 256-row blocks.
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 24usize);
        let batch_rows = [2usize, 15, 16, 17, 255, 256, 257];
        let max_rows = 257usize;
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let format = GdnStateFormat::Q8;
        let slot_bytes = format.state_units(value_heads, dim, dim);
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, max_rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, max_rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta: Vec<f32> = wave(3, max_rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let state0 = wave(4, value * dim, 0.2);
        let proj_gpu = gpu
            .upload_f32(&projection, &[projection.len()])
            .expect("projection");
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let slot0 = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        gdn_state_from_f32(
            &mut gpu,
            GdnStateFormat::Q8,
            state0_gpu.buf.as_ptr(),
            slot0.buf.as_ptr(),
            None,
            value_heads,
            7,
        )
        .expect("quantize");
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        let fresh_slot = |gpu: &mut Gpu| -> GpuTensor {
            let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
            gpu.copy_d2d(&slot0, &slot, slot_bytes).expect("copy slot");
            slot
        };
        let dequant = |gpu: &mut Gpu, slot: &GpuTensor| -> Vec<f32> {
            let f = gpu.zeros(&[value * dim], DType::F32).expect("f32");
            gdn_state_to_f32(
                gpu,
                GdnStateFormat::Q8,
                slot.buf.as_ptr(),
                None,
                f.buf.as_ptr(),
                value_heads,
            )
            .expect("dequantize");
            let values = gpu.download_f32(&f).expect("download");
            gpu.free_tensor(f).expect("free");
            values
        };
        let rel = |a: &[f32], b: &[f32]| {
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                err += (*x as f64 - *y as f64).powi(2);
                norm += (*x as f64).powi(2);
            }
            assert!(norm > 0.0, "reference is all zero");
            (err / norm).sqrt()
        };
        let q0 = dequant(&mut gpu, &slot0);
        let round_trip = rel(&state0, &q0);
        assert!(
            round_trip < 1e-2,
            "Q8 state round trip rel L2 {round_trip:.3e}"
        );

        // Decode: one step per row, F32 and Q8 states from the same start.
        let decode = |gpu: &mut Gpu, state: &GpuTensor| -> Vec<f32> {
            let out = gpu.zeros(&[rows * value], DType::F32).expect("output");
            for row in 0..rows {
                gated_delta_step(
                    gpu,
                    &GatedDeltaStep {
                        q: &proj_gpu.sub_offset(row * qkv, qk),
                        k: &proj_gpu.sub_offset(row * qkv + qk, qk),
                        v: &proj_gpu.sub_offset(row * qkv + 2 * qk, value),
                        gate: &gate_gpu.sub_offset(row * value_heads, value_heads),
                        beta: &beta_gpu.sub_offset(row * value_heads, value_heads),
                        state,
                        format: if state.dtype == DType::F32 {
                            GdnStateFormat::F32
                        } else {
                            GdnStateFormat::Q8
                        },
                        residual: None,
                        output: &out.sub_offset(row * value, value),
                        key_heads,
                        value_heads,
                        key_dim: dim,
                        value_dim: dim,
                        position: row,
                    },
                )
                .expect("GDN step");
            }
            let values = gpu.download_f32(&out).expect("download");
            gpu.free_tensor(out).expect("free");
            values
        };
        let f32_state = gpu.upload_f32(&q0, &[q0.len()]).expect("state");
        let f32_out = decode(&mut gpu, &f32_state);
        let f32_final = gpu.download_f32(&f32_state).expect("download");
        let q8_state = fresh_slot(&mut gpu);
        let q8_out = decode(&mut gpu, &q8_state);
        let q8_final = dequant(&mut gpu, &q8_state);
        let (out_rel, state_rel) = (rel(&f32_out, &q8_out), rel(&f32_final, &q8_final));
        eprintln!("Q8 decode: output rel {out_rel:.3e}, state rel {state_rel:.3e}");
        assert!(
            out_rel < 3e-2 && state_rel < 3e-2,
            "Q8 decode drifted: {out_rel:.3e} {state_rel:.3e}"
        );

        // Persistent batch, plain and captured, and the F32 twin.
        let batch =
            |gpu: &mut Gpu, state: &GpuTensor, ring: Option<&GpuTensor>, n: usize| -> Vec<f32> {
                let out = gpu.zeros(&[n * value], DType::F32).expect("output");
                gated_delta_step_batched(
                    gpu,
                    &GatedDeltaStepBatched {
                        projection: &proj_gpu.sub_offset(0, n * qkv),
                        gate: &gate_gpu.sub_offset(0, n * value_heads),
                        beta: &beta_gpu.sub_offset(0, n * value_heads),
                        state,
                        format: if state.dtype == DType::F32 {
                            GdnStateFormat::F32
                        } else {
                            GdnStateFormat::Q8
                        },
                        residual: None,
                        output: &out,
                        row_states: ring,
                        rows: n,
                        qkv_width: qkv,
                        key_heads,
                        value_heads,
                        key_dim: dim,
                        value_dim: dim,
                        position: 0,
                    },
                )
                .expect("batched GDN");
                let values = gpu.download_f32(&out).expect("download");
                gpu.free_tensor(out).expect("free");
                values
            };
        // Digest of every Q8 state this test writes: equal across processes
        // (the multi-process stress compares it).
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        let mut fold = |data: &[u8]| {
            for byte in data {
                digest = (digest ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        fold(&bytes(&gpu, &q8_state));
        for n in batch_rows {
            let keep = (n / 2).max(1);
            let f32_batch_state = gpu.upload_f32(&q0, &[q0.len()]).expect("state");
            let f32_batch_out = batch(&mut gpu, &f32_batch_state, None, n);
            let f32_batch_final = gpu.download_f32(&f32_batch_state).expect("download");
            let q8_batch_state = fresh_slot(&mut gpu);
            let q8_batch_out = batch(&mut gpu, &q8_batch_state, None, n);
            let q8_batch_final = dequant(&mut gpu, &q8_batch_state);
            let (out_rel, state_rel) = (
                rel(&f32_batch_out, &q8_batch_out),
                rel(&f32_batch_final, &q8_batch_final),
            );
            eprintln!("Q8 batch {n} rows: output rel {out_rel:.3e}, state rel {state_rel:.3e}");
            assert!(
                out_rel < 1e-2 && state_rel < 1e-2,
                "Q8 batch {n} rows drifted: {out_rel:.3e} {state_rel:.3e}"
            );
            fold(&bytes(&gpu, &q8_batch_state));

            // Verify capture: AR's tokens, not the unarmed batch. Its output
            // and ring slot are `n` single-row decode steps from the same
            // state at the same positions, bit for bit.
            let oracle_slot = fresh_slot(&mut gpu);
            let (seq_out, seq_states) = gdn_sequential(
                &mut gpu,
                &proj_gpu,
                &gate_gpu,
                &beta_gpu,
                &oracle_slot,
                n,
                0,
                key_heads,
                value_heads,
                GdnStateFormat::Q8,
            );
            let captured_state = fresh_slot(&mut gpu);
            let ring = gpu.zeros(&[2 * n * slot_bytes], DType::Raw).expect("ring");
            let captured_out = batch(&mut gpu, &captured_state, Some(&ring), n);
            assert!(
                captured_out
                    .iter()
                    .zip(&seq_out)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{n} rows: captured verify output is not single-row decode's"
            );
            assert_eq!(
                bytes(&gpu, &captured_state),
                bytes(&gpu, &slot0),
                "{n} rows: capture wrote the live state"
            );
            assert!(
                bytes(&gpu, &ring.sub_offset((n - 1) * slot_bytes, slot_bytes))
                    == seq_states[n - 1],
                "{n} rows: captured slot is not single-row decode's state"
            );

            // Rollback: `keep` rows re-run from ring slot 0 into slot `n`. Its
            // input is the rows' projection, then gate, then beta.
            let recurrence: Vec<f32> = projection[..n * qkv]
                .iter()
                .chain(&gate[..n * value_heads])
                .chain(&beta[..n * value_heads])
                .copied()
                .collect();
            let recurrence_gpu = gpu
                .upload_f32(&recurrence, &[recurrence.len()])
                .expect("recurrence");
            gpu.copy_d2d(&slot0, &ring, slot_bytes).expect("seed ring");
            let pointers: Vec<u8> = [recurrence_gpu.buf.as_ptr(), ring.buf.as_ptr()]
                .iter()
                .flat_map(|p| (*p as u64).to_ne_bytes())
                .collect();
            let table = gpu.upload_raw(&pointers, &[pointers.len()]).expect("table");
            let discard = gpu.zeros(&[n * value], DType::F32).expect("discard");
            gated_delta_rollback_layers(
                &mut gpu,
                &GatedDeltaRollbackLayers {
                    table: &table,
                    discard: &discard,
                    format,
                    layers: 1,
                    rows: n,
                    keep,
                    from: 0,
                    to: n,
                    qkv_width: qkv,
                    key_heads,
                    value_heads,
                    position: keep - 1,
                },
            )
            .expect("rollback");
            assert!(
                bytes(
                    &gpu,
                    &ring.sub_offset((n + keep - 1) * slot_bytes, slot_bytes)
                ) == seq_states[keep - 1],
                "{n} rows: rollback slot is not single-row decode's state after {keep} rows"
            );
            fold(&seq_states[keep - 1]);
            for tensor in [
                f32_batch_state,
                q8_batch_state,
                oracle_slot,
                captured_state,
                ring,
                recurrence_gpu,
                table,
                discard,
            ] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        eprintln!("GDN_Q8_DIGEST {digest:016x}");
        for tensor in [
            proj_gpu, gate_gpu, beta_gpu, state0_gpu, slot0, f32_state, q8_state,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// The chunked F16 WMMA prefill route on a Q8 state (dequantize, chunk,
    /// requantize) tracks the same route on the F32 state.
    #[test]
    fn gdn_q8_state_chunk_route_tracks_f32() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_wmma_w32() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx11 WMMA");
            return;
        }
        let (key_heads, value_heads, dim, rows) = (2usize, 6usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let wave = |seed: usize, n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                    (h as f32 - 5003.0) / 5003.0 * scale
                })
                .collect()
        };
        let projection = wave(1, rows * qkv, 1.5);
        let gate: Vec<f32> = wave(2, rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta: Vec<f32> = wave(3, rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let state0 = wave(4, value * dim, 0.2);
        let z = wave(5, rows * value, 2.0);
        let norm_bytes: Vec<u8> = wave(6, dim, 1.0)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let proj_bytes: Vec<u8> = projection
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect();
        let mut proj_bf16 = gpu
            .upload_raw(&proj_bytes, &[proj_bytes.len()])
            .expect("projection");
        proj_bf16.dtype = DType::BF16;
        proj_bf16.shape = vec![projection.len()];
        let gate_gpu = gpu.upload_f32(&gate, &[gate.len()]).expect("gate");
        let beta_gpu = gpu.upload_f32(&beta, &[beta.len()]).expect("beta");
        let z_gpu = gpu.upload_f32(&z, &[z.len()]).expect("z");
        let mut norm_gpu = gpu
            .upload_raw(&norm_bytes, &[norm_bytes.len()])
            .expect("norm");
        norm_gpu.dtype = DType::BF16;
        norm_gpu.shape = vec![dim];
        let slot_bytes = GdnStateFormat::Q8.state_units(value_heads, dim, dim);
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let seed_slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        gdn_state_from_f32(
            &mut gpu,
            GdnStateFormat::Q8,
            state0_gpu.buf.as_ptr(),
            seed_slot.buf.as_ptr(),
            None,
            value_heads,
            3,
        )
        .expect("quantize");
        let run = |gpu: &mut Gpu, state: &GpuTensor, n: usize| -> Vec<f32> {
            let recurrent = gpu.zeros(&[n * value], DType::F32).expect("recurrent");
            let out = gpu.zeros(&[n * value], DType::F32).expect("output");
            let step = GatedDeltaStepBatched {
                projection: &proj_bf16.sub_offset(0, n * qkv),
                gate: &gate_gpu.sub_offset(0, n * value_heads),
                beta: &beta_gpu.sub_offset(0, n * value_heads),
                state,
                format: if state.dtype == DType::F32 {
                    GdnStateFormat::F32
                } else {
                    GdnStateFormat::Q8
                },
                residual: None,
                output: &recurrent,
                row_states: None,
                rows: n,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            };
            let gated = GatedDeltaGateBatched {
                recurrent_output: &recurrent,
                z: &z_gpu.sub_offset(0, n * value),
                norm: &norm_gpu,
                output: &out,
                rows: n,
                value_heads,
                value_dim: dim,
            };
            gated_delta_step_gate_wmma(gpu, &step, &gated).expect("chunked GDN");
            let values = gpu.download_f32(&out).expect("download");
            gpu.free_tensor(recurrent).expect("free");
            gpu.free_tensor(out).expect("free");
            values
        };
        let rel = |a: &[f32], b: &[f32]| {
            let (mut err, mut norm) = (0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                err += (*x as f64 - *y as f64).powi(2);
                norm += (*x as f64).powi(2);
            }
            assert!(norm > 0.0, "reference is all zero");
            (err / norm).sqrt()
        };
        let mut digest = 0xcbf2_9ce4_8422_2325u64;
        // The route's minimum and the 16-row chunk boundary +-1.
        for n in [512usize, 513, 527, 528, 529, 530] {
            let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
            gpu.copy_d2d(&seed_slot, &slot, slot_bytes)
                .expect("copy slot");
            let f32_state = gpu.zeros(&[value * dim], DType::F32).expect("state");
            gdn_state_to_f32(
                &mut gpu,
                GdnStateFormat::Q8,
                slot.buf.as_ptr(),
                None,
                f32_state.buf.as_ptr(),
                value_heads,
            )
            .expect("dequantize");
            let f32_out = run(&mut gpu, &f32_state, n);
            let q8_out = run(&mut gpu, &slot, n);
            let f32_final = gpu.download_f32(&f32_state).expect("download");
            let q8_final = gpu.zeros(&[value * dim], DType::F32).expect("state");
            gdn_state_to_f32(
                &mut gpu,
                GdnStateFormat::Q8,
                slot.buf.as_ptr(),
                None,
                q8_final.buf.as_ptr(),
                value_heads,
            )
            .expect("dequantize");
            let q8_final_values = gpu.download_f32(&q8_final).expect("download");
            // The same inputs and dequantized start: the chunk output is
            // equal; the final state differs by one requantization.
            assert_eq!(
                f32_out, q8_out,
                "{n} rows: Q8 chunk output differs from its F32 twin"
            );
            let state_rel = rel(&f32_final, &q8_final_values);
            eprintln!("Q8 chunk {n} rows: state rel {state_rel:.3e}");
            assert!(
                state_rel < 1e-2,
                "{n} rows: Q8 chunk state rel L2 {state_rel:.3e}"
            );
            let mut slot_bytes_host = vec![0u8; slot_bytes];
            gpu.hip
                .memcpy_dtoh(&mut slot_bytes_host, &slot.buf)
                .expect("download");
            for byte in slot_bytes_host {
                digest = (digest ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
            for tensor in [slot, f32_state, q8_final] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        eprintln!("GDN_Q8_CHUNK_DIGEST {digest:016x}");
        for tensor in [
            proj_bf16, gate_gpu, beta_gpu, z_gpu, norm_gpu, state0_gpu, seed_slot,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// Round-to-nearest-even BF16 bits of `values`, little-endian.
    fn bf16_le_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| {
                let u = v.to_bits();
                (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16).to_le_bytes()
            })
            .collect()
    }

    fn test_wave(seed: usize, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let h = i.wrapping_mul(2_654_435_761).wrapping_add(seed * 97) % 10007;
                (h as f32 - 5003.0) / 5003.0 * scale
            })
            .collect()
    }

    /// H5 hookup: the opt-in convolution that also stores the normalized Q/K
    /// plus the Q/K-norm-skipping chunk launch is bytewise the plain
    /// convolution + `gated_delta_qk_norm_bf16_batched` + chunk launch pair:
    /// gated output, final state, convolution ring, gate/beta and the V
    /// columns of the convolution output (all columns with `keep_qk_conv`).
    /// Ragged last row tile, a nonzero ring cursor and the in-place ring the
    /// pipeline uses.  The flag is off by default, and the geometry guards
    /// refuse every other head layout.
    #[test]
    fn gdn_conv_qknorm_hookup_is_bytewise_the_two_launch_pair() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx1151 and the F16 WMMA route");
            return;
        }
        // 530: ragged last row tile (34 row blocks); 8192: the Flash-Next
        // prefill chunk (512 row blocks), where the compact 1-D conv grid is
        // 24 channel blocks x 512 row blocks + 512 parameter blocks.
        for rows in [530usize, 8192] {
            gdn_conv_qknorm_hookup_case(&mut gpu, rows);
        }
    }

    fn gdn_conv_qknorm_hookup_case(gpu: &mut Gpu, rows: usize) {
        let (key_heads, value_heads, dim) = (16usize, 16usize, 128usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let projection = gpu
            .upload_f32(&test_wave(1, rows * qkv, 1.5), &[rows * qkv])
            .expect("projection");
        let mut conv_kernel = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(7, qkv * 4, 0.6)), &[qkv * 4 * 2])
            .expect("conv kernel");
        conv_kernel.dtype = DType::BF16;
        conv_kernel.shape = vec![qkv * 4];
        let history0 = test_wave(8, 3 * qkv, 1.0);
        let a = gpu
            .upload_f32(
                &test_wave(9, rows * value_heads, 2.0),
                &[rows * value_heads],
            )
            .expect("a");
        let b = gpu
            .upload_f32(
                &test_wave(10, rows * value_heads, 2.0),
                &[rows * value_heads],
            )
            .expect("b");
        let mut a_log = gpu
            .upload_raw(
                &bf16_le_bytes(&test_wave(11, value_heads, 1.0)),
                &[value_heads * 2],
            )
            .expect("a_log");
        a_log.dtype = DType::BF16;
        a_log.shape = vec![value_heads];
        let mut dt_bias = gpu
            .upload_raw(
                &bf16_le_bytes(&test_wave(12, value_heads, 1.0)),
                &[value_heads * 2],
            )
            .expect("dt_bias");
        dt_bias.dtype = DType::BF16;
        dt_bias.shape = vec![value_heads];
        let state0 = test_wave(4, value_heads * dim * dim, 0.2);
        let z = gpu
            .upload_f32(&test_wave(5, rows * value, 2.0), &[rows * value])
            .expect("z");
        let mut norm = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(6, dim, 1.0)), &[dim * 2])
            .expect("norm");
        norm.dtype = DType::BF16;
        norm.shape = vec![dim];
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        // (gated output, state, ring, gate, beta, convolution output)
        let run = |gpu: &mut Gpu, fused: bool, keep_qk_conv: bool| -> [Vec<u8>; 6] {
            let history = gpu
                .upload_f32(&history0, &[history0.len()])
                .expect("history");
            let conv_out = gpu.zeros(&[rows * qkv], DType::BF16).expect("conv out");
            let gate = gpu.zeros(&[rows * value_heads], DType::F32).expect("gate");
            let beta = gpu.zeros(&[rows * value_heads], DType::F32).expect("beta");
            let state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
            let recurrent = gpu.zeros(&[rows * value], DType::F32).expect("recurrent");
            let out = gpu.zeros(&[rows * value], DType::F32).expect("output");
            let conv = GatedDeltaConvBatched {
                input: &projection,
                kernel: &conv_kernel,
                history: &history,
                output: &conv_out,
                next_history: &history,
                rows,
                channels: qkv,
                history_rows: 3,
                kernel_size: 4,
                start_cursor: 1,
            };
            let params = GatedDeltaParamsBatched {
                a: &a,
                b: &b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                gate: &gate,
                beta: &beta,
                rows,
                heads: value_heads,
            };
            let step = GatedDeltaStepBatched {
                projection: &conv_out,
                gate: &gate,
                beta: &beta,
                state: &state,
                format: GdnStateFormat::F32,
                residual: None,
                output: &recurrent,
                row_states: None,
                rows,
                qkv_width: qkv,
                key_heads,
                value_heads,
                key_dim: dim,
                value_dim: dim,
                position: 0,
            };
            let gated = GatedDeltaGateBatched {
                recurrent_output: &recurrent,
                z: &z,
                norm: &norm,
                output: &out,
                rows,
                value_heads,
                value_dim: dim,
            };
            assert!(
                gated_delta_chunk_route(gpu, &step),
                "chunked route did not apply"
            );
            assert!(gated_delta_conv_qknorm_geometry(gpu, &step, &conv));
            // Default off: the flag alone selects the new kernels.
            assert_eq!(
                gated_delta_conv_qknorm_route(gpu, &step, &conv),
                gpu.flags.qwen4_gdn_conv_qknorm_enabled()
            );
            if fused {
                gated_delta_conv_params_qknorm_batched(gpu, &conv, &params, &step, keep_qk_conv)
                    .expect("conv + qk norm");
                gated_delta_step_gate_wmma_qknormed(gpu, &step, &gated).expect("chunk");
            } else {
                gated_delta_conv_params_batched(gpu, &conv, &params).expect("conv");
                gated_delta_step_gate_wmma(gpu, &step, &gated).expect("chunk");
            }
            let values = [
                bytes(gpu, &out),
                bytes(gpu, &state),
                bytes(gpu, &history),
                bytes(gpu, &gate),
                bytes(gpu, &beta),
                bytes(gpu, &conv_out),
            ];
            for tensor in [history, conv_out, gate, beta, state, recurrent, out] {
                gpu.free_tensor(tensor).expect("free");
            }
            values
        };
        let plain = run(gpu, false, false);
        let fused = run(gpu, true, false);
        let kept = run(gpu, true, true);
        for (i, label) in ["gated output", "state", "conv ring", "gate", "beta"]
            .iter()
            .enumerate()
        {
            assert!(plain[i] == fused[i], "{label} differs (qk columns dropped)");
            assert!(plain[i] == kept[i], "{label} differs (qk columns kept)");
        }
        assert!(
            plain[5] == kept[5],
            "convolution output differs with keep_qk_conv"
        );
        // Gate / beta against the stand-alone parameter kernel (a separate
        // grid), so the compact conv + parameter grid is not only compared to
        // itself: `rows * value_heads` spans several parameter blocks.
        let ref_gate = gpu
            .zeros(&[rows * value_heads], DType::F32)
            .expect("ref gate");
        let ref_beta = gpu
            .zeros(&[rows * value_heads], DType::F32)
            .expect("ref beta");
        gated_delta_params_batched(
            gpu,
            &GatedDeltaParamsBatched {
                a: &a,
                b: &b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                gate: &ref_gate,
                beta: &ref_beta,
                rows,
                heads: value_heads,
            },
        )
        .expect("stand-alone params");
        assert!(
            bytes(gpu, &ref_gate) == plain[3],
            "rows {rows}: gate vs stand-alone params"
        );
        assert!(
            bytes(gpu, &ref_beta) == plain[4],
            "rows {rows}: beta vs stand-alone params"
        );
        for tensor in [ref_gate, ref_beta] {
            gpu.free_tensor(tensor).expect("free");
        }
        let (row_bytes, v_from) = (qkv * 2, 2 * qk * 2);
        for r in 0..rows {
            assert!(
                plain[5][r * row_bytes + v_from..(r + 1) * row_bytes]
                    == fused[5][r * row_bytes + v_from..(r + 1) * row_bytes],
                "row {r}: V convolution columns differ"
            );
        }
        // The geometry guards: only 16 key heads of 128 in q|k|v order.
        let conv_out = gpu.zeros(&[rows * qkv], DType::BF16).expect("conv out");
        let history = gpu
            .upload_f32(&history0, &[history0.len()])
            .expect("history");
        let state = gpu.upload_f32(&state0, &[state0.len()]).expect("state");
        let gate = gpu.zeros(&[rows * value_heads], DType::F32).expect("gate");
        let recurrent = gpu.zeros(&[rows * value], DType::F32).expect("recurrent");
        let step = GatedDeltaStepBatched {
            projection: &conv_out,
            gate: &gate,
            beta: &gate,
            state: &state,
            format: GdnStateFormat::F32,
            residual: None,
            output: &recurrent,
            row_states: None,
            rows,
            qkv_width: qkv,
            key_heads,
            value_heads,
            key_dim: dim,
            value_dim: dim,
            position: 0,
        };
        let conv = GatedDeltaConvBatched {
            input: &projection,
            kernel: &conv_kernel,
            history: &history,
            output: &conv_out,
            next_history: &history,
            rows,
            channels: qkv,
            history_rows: 3,
            kernel_size: 4,
            start_cursor: 0,
        };
        assert!(gated_delta_conv_qknorm_geometry(&gpu, &step, &conv));
        let eight = GatedDeltaStepBatched {
            key_heads: 8,
            ..step
        };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &eight, &conv));
        let short = GatedDeltaStepBatched { rows: 511, ..eight };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &short, &conv));
        let f32_out = gpu.zeros(&[rows * qkv], DType::F32).expect("f32 conv out");
        let f32_conv = GatedDeltaConvBatched {
            output: &f32_out,
            ..conv
        };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &step, &f32_conv));
        let ring4 = GatedDeltaConvBatched {
            history_rows: 4,
            ..f32_conv
        };
        assert!(!gated_delta_conv_qknorm_geometry(&gpu, &step, &ring4));
        for tensor in [
            projection,
            conv_kernel,
            a,
            b,
            a_log,
            dt_bias,
            z,
            norm,
            conv_out,
            history,
            state,
            gate,
            recurrent,
            f32_out,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// H6 hookup: the inline-Q8 chunk module is bytewise the
    /// dequantize + chunk + requantize arm it replaces (gated output and the
    /// whole Q8 slot), over the route's minimum rows and the 16-row chunk
    /// boundary +-1, at a nonzero start position (the requantization seed).
    #[test]
    fn gdn_q8_inline_hookup_is_bytewise_the_conversion_arm() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx1151 and the F16 WMMA route");
            return;
        }
        let (key_heads, value_heads, dim, max_rows) = (2usize, 6usize, 128usize, 530usize);
        let qk = key_heads * dim;
        let value = value_heads * dim;
        let qkv = 2 * qk + value;
        let mut proj_bf16 = gpu
            .upload_raw(
                &bf16_le_bytes(&test_wave(1, max_rows * qkv, 1.5)),
                &[max_rows * qkv * 2],
            )
            .expect("projection");
        proj_bf16.dtype = DType::BF16;
        proj_bf16.shape = vec![max_rows * qkv];
        let gate_values: Vec<f32> = test_wave(2, max_rows * value_heads, 0.5)
            .iter()
            .map(|g| g - 0.6)
            .collect();
        let beta_values: Vec<f32> = test_wave(3, max_rows * value_heads, 0.45)
            .iter()
            .map(|b| b + 0.5)
            .collect();
        let gate_gpu = gpu
            .upload_f32(&gate_values, &[gate_values.len()])
            .expect("gate");
        let beta_gpu = gpu
            .upload_f32(&beta_values, &[beta_values.len()])
            .expect("beta");
        let z_gpu = gpu
            .upload_f32(&test_wave(5, max_rows * value, 2.0), &[max_rows * value])
            .expect("z");
        let mut norm_gpu = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(6, dim, 1.0)), &[dim * 2])
            .expect("norm");
        norm_gpu.dtype = DType::BF16;
        norm_gpu.shape = vec![dim];
        let state0 = test_wave(4, value_heads * dim * dim, 0.2);
        let slot_bytes = GdnStateFormat::Q8.state_units(value_heads, dim, dim);
        let state0_gpu = gpu.upload_f32(&state0, &[state0.len()]).expect("state0");
        let seed_slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
        gdn_state_from_f32(
            &mut gpu,
            GdnStateFormat::Q8,
            state0_gpu.buf.as_ptr(),
            seed_slot.buf.as_ptr(),
            None,
            value_heads,
            3,
        )
        .expect("quantize");
        let bytes = |gpu: &Gpu, t: &GpuTensor| -> Vec<u8> {
            let mut out = vec![0u8; t.byte_size()];
            gpu.hip.memcpy_dtoh(&mut out, &t.buf).expect("download");
            out
        };
        for (n, position) in [
            (512usize, 0usize),
            (513, 7),
            (527, 4096),
            (528, 1),
            (529, 99),
            (530, 262_143),
        ] {
            let mut results = Vec::new();
            for inline in [false, true] {
                let slot = gpu.zeros(&[slot_bytes], DType::Raw).expect("slot");
                gpu.copy_d2d(&seed_slot, &slot, slot_bytes)
                    .expect("copy slot");
                let recurrent = gpu.zeros(&[n * value], DType::F32).expect("recurrent");
                let out = gpu.zeros(&[n * value], DType::F32).expect("output");
                let step = GatedDeltaStepBatched {
                    projection: &proj_bf16.sub_offset(0, n * qkv),
                    gate: &gate_gpu.sub_offset(0, n * value_heads),
                    beta: &beta_gpu.sub_offset(0, n * value_heads),
                    state: &slot,
                    format: GdnStateFormat::Q8,
                    residual: None,
                    output: &recurrent,
                    row_states: None,
                    rows: n,
                    qkv_width: qkv,
                    key_heads,
                    value_heads,
                    key_dim: dim,
                    value_dim: dim,
                    position,
                };
                let gated = GatedDeltaGateBatched {
                    recurrent_output: &recurrent,
                    z: &z_gpu.sub_offset(0, n * value),
                    norm: &norm_gpu,
                    output: &out,
                    rows: n,
                    value_heads,
                    value_dim: dim,
                };
                assert!(gated_delta_q8_inline_geometry(&gpu, &step));
                // Default off: the flag alone selects the new module.
                assert_eq!(
                    gated_delta_q8_inline_route(&gpu, &step),
                    gpu.flags.qwen4_gdn_q8_inline_enabled()
                );
                let arms = GatedDeltaChunkArms {
                    qk_prenormed: false,
                    q8_inline: inline,
                };
                gated_delta_step_gate_wmma_arms(&mut gpu, &step, &gated, arms)
                    .expect("chunked GDN");
                results.push((bytes(&gpu, &out), bytes(&gpu, &slot)));
                for tensor in [slot, recurrent, out] {
                    gpu.free_tensor(tensor).expect("free");
                }
            }
            assert!(
                results[0].0 == results[1].0,
                "{n} rows @ {position}: gated output differs"
            );
            assert!(
                results[0].1 == results[1].1,
                "{n} rows @ {position}: Q8 slot differs"
            );
        }
        // An F32 state or a misaligned slot never takes the inline arm.
        let f32_state = gpu
            .zeros(&[value_heads * dim * dim], DType::F32)
            .expect("f32 state");
        let step = GatedDeltaStepBatched {
            projection: &proj_bf16,
            gate: &gate_gpu,
            beta: &beta_gpu,
            state: &f32_state,
            format: GdnStateFormat::F32,
            residual: None,
            output: &f32_state,
            row_states: None,
            rows: 512,
            qkv_width: qkv,
            key_heads,
            value_heads,
            key_dim: dim,
            value_dim: dim,
            position: 0,
        };
        assert!(!gated_delta_q8_inline_geometry(&gpu, &step));
        for tensor in [
            proj_bf16, gate_gpu, beta_gpu, z_gpu, norm_gpu, state0_gpu, seed_slot, f32_state,
        ] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    fn null_tensor(shape: &[usize], dtype: DType) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.shape = shape.to_vec();
        tensor.dtype = dtype;
        tensor
    }

    #[test]
    fn recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let input = [1.0f32, -2.0, 3.5, 0.25];
        let values = gpu
            .upload_f32(&input, &[input.len()])
            .expect("scale upload");

        // Ordinary path: no recorder armed, so the same launch leaves no tape.
        scale_f32(
            &mut gpu,
            &ScaleF32 {
                values: &values,
                scale: 2.0,
            },
        )
        .expect("scale");
        assert_eq!(gpu.replay.recorded_launches().len(), 0);

        // Open a recording window: the launch must now enter the tape with the
        // exact bytes it launched, resolve its owning artifact, AND still run.
        gpu.replay =
            crate::replay::ReplayController::new_armed(crate::replay::ReplayBackendRequest::Auto);
        gpu.replay.begin_capture().expect("open recording window");
        scale_f32(
            &mut gpu,
            &ScaleF32 {
                values: &values,
                scale: 3.0,
            },
        )
        .expect("scale");
        let recorded = gpu.replay.recorded_launches();
        assert_eq!(recorded.len(), 1, "one launch, one tape entry");
        assert_eq!(recorded[0].kernel, "scale_f32");
        assert!(
            recorded[0].artifact.is_some(),
            "the owning artifact must resolve, or preparation cannot lower the tape"
        );
        assert_eq!(recorded[0].grid, [1, 1, 1]);
        let kernarg = &recorded[0].kernarg;
        assert_eq!(kernarg.len() % 16, 0, "blob is tail-padded");
        assert_eq!(
            u64::from_ne_bytes(kernarg[0..8].try_into().unwrap()),
            values.buf.as_ptr() as u64
        );
        assert_eq!(
            i32::from_ne_bytes(kernarg[8..12].try_into().unwrap()),
            input.len() as i32
        );
        assert_eq!(f32::from_ne_bytes(kernarg[12..16].try_into().unwrap()), 3.0);

        // The recorded launch executed: (input * 2.0) * 3.0, all exact in f32.
        let actual = gpu.download_f32(&values).expect("scale download");
        assert_eq!(actual, vec![6.0, -12.0, 21.0, 1.5]);
    }

    #[test]
    fn checked_extents_reject_signed_flattening_overflow() {
        let max = i32::MAX as usize;
        assert_eq!(checked_product(max, 1, "boundary").unwrap(), max);
        assert!(checked_product(max, 2, "boundary").is_err());
        assert!(checked_add(max, 1, "boundary").is_err());
        assert!(blocks(max).is_ok());
        assert!(blocks(max + 1).is_err());
    }

    #[test]
    fn host_argmax_matches_sampler_semantics() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut long = vec![0.25f32; 248_320];
        long[200_001] = 7.0;
        long[5] = 7.0;
        long[9] = f32::INFINITY;
        let cases: [(Vec<f32>, u32); 4] = [
            // Ties resolve to the first index.
            (vec![1.0, 5.0, 3.0, 5.0], 1),
            // Non-finite values never win.
            (
                vec![f32::NAN, f32::INFINITY, 2.0, f32::NEG_INFINITY, 2.0],
                2,
            ),
            // No finite value: index 0.
            (vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 0),
            (long, 5),
        ];
        for (values, expected) in cases {
            let logits = gpu.upload_f32(&values, &[values.len()]).unwrap();
            assert_eq!(argmax_f32_host(&mut gpu, &logits).unwrap(), expected);
            gpu.free_tensor(logits).unwrap();
        }
    }

    #[test]
    fn raw_index_outputs_require_i32_byte_capacity() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };

        let logits = null_tensor(&[1], DType::F32);
        let indices = null_tensor(&[1], DType::Raw);
        let error = argmax_f32(
            &mut gpu,
            &ArgmaxF32 {
                logits: &logits,
                indices: &indices,
                rows: 1,
                vocab: 1,
            },
        )
        .expect_err("argmax must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let query = null_tensor(&[1], DType::F32);
        let pooled = null_tensor(&[1], DType::F32);
        let selected = null_tensor(&[1], DType::Raw);
        let error = indexed_attention_select(
            &mut gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: &pooled,
                selected: &selected,
                block_count: 1,
                index_heads: 1,
                index_dim: 1,
                budget_blocks: 1,
                compress: 1,
                visible: 1,
                capacity: 1,
            },
        )
        .expect_err("QSA selection must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let q_with_gate = null_tensor(&[2], DType::F32);
        let full_keys = null_tensor(&[1], DType::F32);
        let full_values = null_tensor(&[1], DType::F32);
        let output = null_tensor(&[1], DType::F32);
        let error = indexed_attention_attention(
            &mut gpu,
            &IndexedAttentionAttention {
                q_with_gate: &q_with_gate,
                full_keys: &full_keys,
                full_values: &full_values,
                selected: &selected,
                output: &output,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 1,
                selected_len: 1,
                full_capacity: 1,
                end_position: 1,
                format: QsaKvFormat::F32,
            },
        )
        .expect_err("QSA attention must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn gdn_rejects_shape_before_kernel_launch() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let q = null_tensor(&[1], DType::F32);
        let k = null_tensor(&[2], DType::F32);
        let v = null_tensor(&[12], DType::F32);
        let gate = null_tensor(&[3], DType::F32);
        let beta = null_tensor(&[3], DType::F32);
        let state = null_tensor(&[24], DType::F32);
        let output = null_tensor(&[12], DType::F32);
        let params = GatedDeltaStep {
            q: &q,
            k: &k,
            v: &v,
            gate: &gate,
            beta: &beta,
            state: &state,
            format: GdnStateFormat::F32,
            residual: None,
            output: &output,
            key_heads: 1,
            value_heads: 3,
            key_dim: 2,
            value_dim: 4,
            position: 0,
        };

        let error = gated_delta_step(&mut gpu, &params).expect_err("invalid shape must fail");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn qsa_reuse_selection_rejects_byte_capacity_mismatch() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let selected = null_tensor(&[3], DType::Raw);
        let selected_len_out = null_tensor(&[4], DType::Raw);
        let params = IndexedAttentionReuseSelection {
            selected: &selected,
            selected_len: 1,
            position: 0,
            capacity: 1,
            selected_len_out: &selected_len_out,
        };

        let error = indexed_attention_reuse_selection(&mut gpu, &params)
            .expect_err("byte capacity must be checked");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    /// Every position-derived QSA field must reach the recorder as a declaration
    /// pointing at the right kernarg slot: nothing else re-derives these values
    /// for a replay position. The assertion decodes the recorded bytes at each
    /// declared offset, so a wrong offset fails here rather than at replay.
    #[test]
    fn qsa_position_fields_are_declared_to_the_recorder() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        gpu.replay =
            crate::replay::ReplayController::new_armed(crate::replay::ReplayBackendRequest::Auto);
        gpu.replay.begin_capture().expect("open recording window");

        let compress = 4usize;
        let blocks = 2usize;
        let index_dim = 8usize;
        let index_heads = 2usize;
        let rows = 1usize;
        let position_start = blocks * compress - rows;
        let capacity = 8usize;
        let budget_blocks = 2usize;
        let cells = blocks * compress * index_dim;
        let raw: Vec<f32> = (0..cells).map(|i| (i % 17) as f32 * 0.5).collect();
        let raw_gpu = gpu.upload_f32(&raw, &[cells]).expect("raw upload");
        let pooled_gpu = gpu
            .zeros(&[blocks * index_dim], DType::F32)
            .expect("pooled allocation");
        indexed_attention_pool_rope(
            &mut gpu,
            &IndexedAttentionPoolRope {
                raw_keys: &raw_gpu,
                pooled: &pooled_gpu,
                norm: None,
                block_count: blocks,
                compress,
                index_dim,
                position: Some(QsaPositionBinding {
                    position_start,
                    rows,
                }),
                grid_bound: blocks,
                yarn: crate::rope::RopeYarn::OFF,
            },
        )
        .expect("QSA pool/RoPE");

        let query_elements = index_heads * index_dim;
        let query: Vec<f32> = (0..query_elements)
            .map(|i| (i % 13) as f32 * 0.25)
            .collect();
        let query_gpu = gpu
            .upload_f32(&query, &[query_elements])
            .expect("query upload");
        let selected_gpu = gpu
            .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        indexed_attention_select_batch(
            &mut gpu,
            &IndexedAttentionSelectBatch {
                pooled_bf16: None,
                pooled_bf16_valid_blocks: 0,
                query: &query_gpu,
                pooled: &pooled_gpu,
                selected: &selected_gpu,
                rows,
                query_row_stride: query_elements,
                block_count: blocks,
                index_heads,
                index_dim,
                budget_blocks,
                compress,
                position_start,
                capacity,
                shape_blocks: blocks,
            },
        )
        .expect("QSA select");

        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let full_capacity = 8usize;
        let q_with_gate_gpu = gpu
            .zeros(&[rows * n_heads * 2 * head_dim], DType::F32)
            .expect("q allocation");
        let full_keys_gpu = gpu
            .zeros(&[full_capacity * n_kv_heads * head_dim], DType::F32)
            .expect("keys allocation");
        let full_values_gpu = gpu
            .zeros(&[full_capacity * n_kv_heads * head_dim], DType::F32)
            .expect("values allocation");
        let output_gpu = gpu
            .zeros(&[rows * n_heads * head_dim], DType::F32)
            .expect("attention output allocation");
        indexed_attention_attention_batch(
            &mut gpu,
            &IndexedAttentionAttentionBatch {
                q_with_gate: &q_with_gate_gpu,
                full_keys: &full_keys_gpu,
                full_values: &full_values_gpu,
                selected: &selected_gpu,
                output: &output_gpu,
                rows,
                position_start,
                n_heads,
                n_kv_heads,
                head_dim,
                budget_blocks,
                compress,
                capacity,
                full_capacity,
                format: QsaKvFormat::F32,
                shape_selected: capacity,
            },
        )
        .expect("QSA attention");

        // The index-key write: a row offset that moves with the position.
        let index_kv_width = 128usize;
        let dco = position_start * index_kv_width;
        let copy_src = gpu
            .zeros(&[index_kv_width], DType::F32)
            .expect("copy src allocation");
        let copy_dst = gpu
            .zeros(&[dco + index_kv_width], DType::F32)
            .expect("copy dst allocation");
        gpu.copy_rows_strided_f32(
            &copy_src,
            &copy_dst,
            rows,
            index_kv_width,
            index_kv_width,
            index_kv_width,
            dco,
            Some(index_kv_width),
        )
        .expect("index-key write");

        // The incremental pool at a large capacity: `first_block` is declared
        // and the grid is `ceil(rows / compress)`, not the capacity.
        let inc_rows = 3usize;
        let inc_start = blocks * compress - inc_rows;
        indexed_attention_pool_rope_incremental(
            &mut gpu,
            &IndexedAttentionPoolRope {
                raw_keys: &raw_gpu,
                pooled: &pooled_gpu,
                norm: None,
                block_count: blocks,
                compress,
                index_dim,
                position: Some(QsaPositionBinding {
                    position_start: inc_start,
                    rows: inc_rows,
                }),
                grid_bound: 65_536,
                yarn: crate::rope::RopeYarn::OFF,
            },
        )
        .expect("incremental QSA pool/RoPE");

        let launches = gpu.replay.recorded_launches();
        assert_eq!(launches.len(), 5, "the recorded QSA sequence changed");

        // The declared slot must hold the value this very launch used.
        let check = |launch: &crate::replay::RecordedHipLaunch,
                     binding: crate::replay::ReplayKernargBinding,
                     expected: u32,
                     label: &str| {
            assert!(
                launch.declared_kernarg_bindings().contains(&binding),
                "{label}: missing declaration {binding:?} in {:?}",
                launch.declared_kernarg_bindings()
            );
            let offset = binding.offset();
            let recorded = u32::from_ne_bytes(
                launch.kernarg[offset..offset + 4]
                    .try_into()
                    .expect("binding offset"),
            );
            assert_eq!(
                recorded, expected,
                "{label}: declaration points at the wrong slot"
            );
        };

        let pool_blocks = crate::replay::ReplayKernargBinding::PositionDivU32 {
            offset: 24,
            addend: rows as u32,
            divisor: compress as u32,
        };
        let pool_binding = launches[0]
            .declared_kernarg_bindings()
            .iter()
            .find(|binding| {
                matches!(
                    binding,
                    crate::replay::ReplayKernargBinding::PositionDivU32 { .. }
                )
            })
            .copied()
            .expect("pool declares its block count");
        assert_eq!(pool_binding, pool_blocks, "pool block-count offset drifted");
        check(
            &launches[0],
            pool_binding,
            blocks as u32,
            "pool block count",
        );

        let select_div = launches[1]
            .declared_kernarg_bindings()
            .iter()
            .find(|binding| {
                matches!(
                    binding,
                    crate::replay::ReplayKernargBinding::PositionDivU32 { .. }
                )
            })
            .copied()
            .expect("select declares its block count");
        let select_pos = launches[1]
            .declared_kernarg_bindings()
            .iter()
            .find(|binding| {
                matches!(
                    binding,
                    crate::replay::ReplayKernargBinding::PositionPlusU32 { .. }
                )
            })
            .copied()
            .expect("select declares its start position");
        check(
            &launches[1],
            select_div,
            blocks as u32,
            "select block count",
        );
        check(
            &launches[1],
            select_pos,
            position_start as u32,
            "select position",
        );

        let attention_pos = launches[2]
            .declared_kernarg_bindings()
            .first()
            .copied()
            .expect("attention declares its start position");
        check(
            &launches[2],
            attention_pos,
            position_start as u32,
            "attention position",
        );

        let copy_mul = launches[3]
            .declared_kernarg_bindings()
            .first()
            .copied()
            .expect("index-key write declares its row offset");
        assert_eq!(
            copy_mul,
            crate::replay::ReplayKernargBinding::PositionMulU32 {
                offset: copy_mul.offset(),
                factor: index_kv_width as u32,
            },
            "index-key write must declare `position * index_kv_width`"
        );
        check(&launches[3], copy_mul, dco as u32, "index-key row offset");

        let inc = &launches[4];
        assert_eq!(
            inc.grid[0],
            inc_rows.div_ceil(compress) as u32,
            "incremental pool grid must be ceil(rows / compress), not the capacity"
        );
        assert!(
            inc.grid_binding.is_none(),
            "incremental pool grid must be constant"
        );
        check(
            inc,
            crate::replay::ReplayKernargBinding::PositionDivU32 {
                offset: 24,
                addend: inc_rows as u32,
                divisor: compress as u32,
            },
            blocks as u32,
            "incremental pool block count",
        );
        check(
            inc,
            crate::replay::ReplayKernargBinding::PositionDivU32 {
                offset: 36,
                addend: 0,
                divisor: compress as u32,
            },
            (inc_start / compress) as u32,
            "incremental pool first block",
        );

        gpu.free_tensor(raw_gpu).expect("free raw");
        gpu.free_tensor(pooled_gpu).expect("free pooled");
        gpu.free_tensor(query_gpu).expect("free query");
        gpu.free_tensor(selected_gpu).expect("free selected");
        gpu.free_tensor(q_with_gate_gpu).expect("free q");
        gpu.free_tensor(full_keys_gpu).expect("free keys");
        gpu.free_tensor(full_values_gpu).expect("free values");
        gpu.free_tensor(output_gpu).expect("free attention output");
        gpu.free_tensor(copy_src).expect("free copy src");
        gpu.free_tensor(copy_dst).expect("free copy dst");
    }

    /// The BF16 split route's position fields (block count, group
    /// `position_start`) reach the recorder at the right kernarg slots on every
    /// score / select / filter launch, and its grids come from the shape bound,
    /// not the live count. A one-million-block bound leaves 16 score rows, so
    /// 17 rows run as two groups with grouped addends.
    #[test]
    fn qsa_bf16_split_position_fields_are_declared_to_the_recorder() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11+ device");
            return;
        }
        let (blocks, compress, index_dim, index_heads, rows) =
            (8usize, 4usize, 8usize, 4usize, 17usize);
        let position_start = blocks * compress - rows;
        let (capacity, budget_blocks, shape_blocks) = (16usize, 2usize, 1_000_000usize);
        let pooled = gpu
            .zeros(&[blocks * index_dim], DType::BF16)
            .expect("pooled allocation");
        let query = gpu
            .zeros(&[rows * index_heads * index_dim], DType::F32)
            .expect("query allocation");
        let selected = gpu
            .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let params = IndexedAttentionSelectBatch {
            pooled_bf16: None,
            pooled_bf16_valid_blocks: 0,
            query: &query,
            pooled: &pooled,
            selected: &selected,
            rows,
            query_row_stride: index_heads * index_dim,
            block_count: blocks,
            index_heads,
            index_dim,
            budget_blocks,
            compress,
            position_start,
            capacity,
            shape_blocks,
        };
        let score_tiles = shape_blocks.div_ceil(256) as u32;
        let compact_tiles = shape_blocks.div_ceil(4096) as u32;
        for (arm, per_group) in [
            (QsaBf16SplitArm::Select, 2usize),
            (QsaBf16SplitArm::Filter(QsaFilterKnobs::default()), 4usize),
        ] {
            gpu.replay = crate::replay::ReplayController::new_armed(
                crate::replay::ReplayBackendRequest::Auto,
            );
            gpu.replay.begin_capture().expect("open recording window");
            indexed_attention_select_batch_bf16_split_arm(
                &mut gpu, &params, None, arm, false, false,
            )
            .expect("recorded BF16 split select");
            gpu.replay.finish_capture().expect("close recording window");
            let launches = gpu.replay.recorded_launches();
            assert_eq!(
                launches.len(),
                2 * per_group,
                "{arm:?}: two groups of {per_group} launches"
            );
            for (index, launch) in launches.iter().enumerate() {
                let first_row = (index / per_group) * 16;
                let label = format!("{arm:?} launch {index} ({})", launch.kernel);
                let declared = launch.declared_kernarg_bindings();
                assert_eq!(
                    declared.len(),
                    2,
                    "{label}: exactly the block count and position"
                );
                let count = declared
                    .iter()
                    .find(|b| {
                        matches!(
                            b,
                            crate::replay::ReplayKernargBinding::PositionDivU32 { .. }
                        )
                    })
                    .copied()
                    .unwrap_or_else(|| panic!("{label}: no block-count declaration"));
                let start = declared
                    .iter()
                    .find(|b| {
                        matches!(
                            b,
                            crate::replay::ReplayKernargBinding::PositionPlusU32 { .. }
                        )
                    })
                    .copied()
                    .unwrap_or_else(|| panic!("{label}: no position declaration"));
                // Whole-call count `(position + rows) / compress`, group start
                // `position + first_row`.
                assert!(
                    matches!(count, crate::replay::ReplayKernargBinding::PositionDivU32 { addend, divisor, .. }
                        if addend == rows as u32 && divisor == compress as u32),
                    "{label}: block-count declaration {count:?}"
                );
                assert!(
                    matches!(start, crate::replay::ReplayKernargBinding::PositionPlusU32 { addend, .. }
                        if addend == first_row as u32),
                    "{label}: position declaration {start:?}"
                );
                let word = |binding: crate::replay::ReplayKernargBinding| {
                    let offset = binding.offset();
                    u32::from_ne_bytes(launch.kernarg[offset..offset + 4].try_into().expect("slot"))
                };
                assert_eq!(word(count), blocks as u32, "{label}: block-count slot");
                assert_eq!(
                    word(start),
                    (position_start + first_row) as u32,
                    "{label}: position slot"
                );
                match launch.kernel.as_str() {
                    k if k.starts_with("indexed_attention_select_scores_rows") => {
                        assert_eq!(
                            launch.grid[0], score_tiles,
                            "{label}: score grid is ceil(shape / 256)"
                        );
                    }
                    "qsa_filter_compact" => {
                        assert_eq!(
                            launch.grid[0], compact_tiles,
                            "{label}: compact grid is ceil(shape / 4096)"
                        );
                    }
                    _ => {}
                }
                assert!(launch.grid_binding.is_none(), "{label}: grids are constant");
            }
        }
        gpu.free_tensor(pooled).expect("free pooled");
        gpu.free_tensor(query).expect("free query");
        gpu.free_tensor(selected).expect("free selected");
    }

    /// A retained PM4 tape owns raw device addresses, not allocations. The QSA
    /// K/V and raw index-key arenas are VMM tensors: virtual ranges reserved up
    /// front whose mapped prefix grows in place, so a base address recorded in
    /// the tape must keep naming the same bytes after growth. This records the
    /// full QSA sequence (index-key write, pool/RoPE, select, attention) while
    /// only the first granule of each arena is mapped, prepares the PM4 tape,
    /// and only then maps two more granules. Replaying positions inside the
    /// newly mapped granules must then read and write the grown pages (the
    /// attention tail and the raw index-key write land there, and for the last
    /// position the pooled blocks also cover grown raw keys) and match, byte
    /// for byte, the same sequence launched through plain HIP. Outputs, the
    /// selection, the pooled prefix and the written index-key row are poisoned
    /// before every execution so a stale result cannot pass.
    #[test]
    fn retained_pm4_tape_replays_over_grown_vmm_qsa_arena() {
        use crate::replay::{ReplayBackendRequest, ReplayController};

        const COMPRESS: usize = 4;
        const INDEX_HEADS: usize = 2;
        const N_HEADS: usize = 2;
        const N_KV_HEADS: usize = 1;
        const DIM: usize = 128;
        const BUDGET_BLOCKS: usize = 2;
        const SELECTED_CAPACITY: usize = BUDGET_BLOCKS * COMPRESS + COMPRESS - 1;
        const CAPTURE_POSITION: usize = 3;
        const F32_BYTES: usize = std::mem::size_of::<f32>();

        struct Qsa {
            reserved_rows: usize,
            raw_keys: GpuTensor,
            full_keys: GpuTensor,
            full_values: GpuTensor,
            pooled: GpuTensor,
            index_src: GpuTensor,
            query: GpuTensor,
            q_with_gate: GpuTensor,
            selected: GpuTensor,
            output: GpuTensor,
        }

        struct Observed {
            raw_row: Vec<u8>,
            pooled: Vec<u8>,
            selected: Vec<u8>,
            output: Vec<f32>,
        }

        /// The QSA sequence at one position, launched through whatever route
        /// `gpu.replay` currently selects (recorded when capturing).
        fn run_qsa(gpu: &mut Gpu, t: &Qsa, position: usize) {
            let blocks = (position + 1) / COMPRESS;
            let max_blocks = t.reserved_rows / COMPRESS;
            gpu.copy_rows_strided_f32(
                &t.index_src,
                &t.raw_keys,
                1,
                DIM,
                DIM,
                DIM,
                position * DIM,
                Some(DIM),
            )
            .expect("index-key write");
            indexed_attention_pool_rope(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &t.raw_keys,
                    pooled: &t.pooled,
                    norm: None,
                    block_count: blocks,
                    compress: COMPRESS,
                    index_dim: DIM,
                    position: Some(QsaPositionBinding {
                        position_start: position,
                        rows: 1,
                    }),
                    grid_bound: max_blocks,
                    yarn: crate::rope::RopeYarn::OFF,
                },
            )
            .expect("QSA pool/RoPE");
            indexed_attention_select_batch(
                gpu,
                &IndexedAttentionSelectBatch {
                    pooled_bf16: None,
                    pooled_bf16_valid_blocks: 0,
                    query: &t.query,
                    pooled: &t.pooled,
                    selected: &t.selected,
                    rows: 1,
                    query_row_stride: INDEX_HEADS * DIM,
                    block_count: blocks,
                    index_heads: INDEX_HEADS,
                    index_dim: DIM,
                    budget_blocks: BUDGET_BLOCKS,
                    compress: COMPRESS,
                    position_start: position,
                    capacity: SELECTED_CAPACITY,
                    shape_blocks: max_blocks,
                },
            )
            .expect("QSA select");
            indexed_attention_attention_batch(
                gpu,
                &IndexedAttentionAttentionBatch {
                    q_with_gate: &t.q_with_gate,
                    full_keys: &t.full_keys,
                    full_values: &t.full_values,
                    selected: &t.selected,
                    output: &t.output,
                    rows: 1,
                    position_start: position,
                    n_heads: N_HEADS,
                    n_kv_heads: N_KV_HEADS,
                    head_dim: DIM,
                    budget_blocks: BUDGET_BLOCKS,
                    compress: COMPRESS,
                    capacity: SELECTED_CAPACITY,
                    full_capacity: t.reserved_rows,
                    format: QsaKvFormat::F32,
                    shape_selected: SELECTED_CAPACITY,
                },
            )
            .expect("QSA attention");
        }

        /// Poison every result location, execute the position through the
        /// retained PM4 tape or (replay disabled) plain HIP, and read back.
        fn observe(gpu: &mut Gpu, t: &Qsa, position: usize, replay: bool) -> Observed {
            let blocks = (position + 1) / COMPRESS;
            let row_offset = position * DIM * F32_BYTES;
            let nan_bytes = |count: usize| f32::NAN.to_ne_bytes().repeat(count);
            gpu.hip
                .memcpy_htod_offset(&t.raw_keys.buf, row_offset, &nan_bytes(DIM))
                .expect("poison index-key row");
            gpu.hip
                .memcpy_htod(&t.pooled.buf, &nan_bytes(t.pooled.numel()))
                .expect("poison pooled");
            gpu.hip
                .memcpy_htod(&t.selected.buf, &vec![0xFFu8; SELECTED_CAPACITY * 4])
                .expect("poison selected");
            gpu.hip
                .memcpy_htod(&t.output.buf, &nan_bytes(t.output.numel()))
                .expect("poison output");
            gpu.hip.device_synchronize().expect("sync poison");
            if replay {
                // SAFETY: every tensor the tape recorded is live in `t`, at the
                // base address captured (the arenas only grew in place).
                unsafe { gpu.replay.replay_pm4(position) }
                    .unwrap_or_else(|reason| panic!("PM4 replay at {position}: {reason}"));
            } else {
                let retained = std::mem::replace(
                    &mut gpu.replay,
                    ReplayController::new(ReplayBackendRequest::Hip),
                );
                run_qsa(gpu, t, position);
                gpu.replay = retained;
            }
            gpu.hip.device_synchronize().expect("sync execution");
            let mut raw_row = vec![0u8; DIM * F32_BYTES];
            gpu.hip
                .memcpy_dtoh_at(&mut raw_row, &t.raw_keys.buf, row_offset)
                .expect("read index-key row");
            let mut pooled = vec![0u8; blocks * DIM * F32_BYTES];
            gpu.hip
                .memcpy_dtoh_at(&mut pooled, &t.pooled.buf, 0)
                .expect("read pooled prefix");
            Observed {
                raw_row,
                pooled,
                selected: gpu.download_raw_bytes(&t.selected).expect("read selected"),
                output: gpu.download_f32(&t.output).expect("read output"),
            }
        }

        fn selected_tokens(observed: &Observed) -> Vec<i32> {
            observed
                .selected
                .chunks_exact(4)
                .map(|word| i32::from_ne_bytes(word.try_into().expect("selected word")))
                .collect()
        }

        fn assert_same(label: &str, replayed: &Observed, reference: &Observed) {
            assert_eq!(
                replayed.raw_row, reference.raw_row,
                "{label}: index-key row differs from the HIP reference"
            );
            assert!(
                replayed.pooled == reference.pooled,
                "{label}: pooled prefix differs from the HIP reference"
            );
            assert_eq!(
                replayed.selected, reference.selected,
                "{label}: selection differs from the HIP reference"
            );
            assert!(
                replayed
                    .output
                    .iter()
                    .map(|value| value.to_bits())
                    .eq(reference.output.iter().map(|value| value.to_bits())),
                "{label}: attention output differs from the HIP reference: {:?} vs {:?}",
                &replayed.output[..4],
                &reference.output[..4]
            );
        }

        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        // Field-disjoint borrows (gpu.replay / gpu.device_id) need the plain `&mut Gpu`.
        let gpu = &mut *gpu;
        if gpu.arch != "gfx1201" {
            eprintln!("skip: needs gfx1201, found {}", gpu.arch);
            return;
        }
        let Ok(gran) = gpu.vmm_recommended_granularity() else {
            eprintln!("skip: no VMM allocation granularity");
            return;
        };
        // The retained PM4 route must be available on this very HSA agent, not
        // merely configured: kernel admission and prepare failures below are
        // real failures and must not be skipped.
        let device_ordinal = gpu.device_id as usize;
        let pm4_agent = redline_dispatch::aql::load_symbols()
            .ok()
            .and_then(|symbols| {
                let runtime = redline_dispatch::aql::Runtime::initialize(symbols).ok()?;
                let device = runtime
                    .select_gpu(redline_dispatch::aql::GpuSelector::Ordinal(device_ordinal))
                    .ok()?;
                Some(device.name().eq_ignore_ascii_case("gfx1201"))
            })
            .unwrap_or(false);
        if !pm4_agent {
            eprintln!("skip: no gfx1201 HSA agent for retained PM4");
            return;
        }

        // One row of every arena is DIM F32 values; the first granule holds
        // `initial_rows` of them and each arena reserves three granules.
        assert_eq!(gran % (DIM * F32_BYTES), 0, "granule must hold whole rows");
        eprintln!("gfx1201 VMM granularity: {gran} bytes");
        let initial_rows = gran / (DIM * F32_BYTES);
        assert_eq!(initial_rows % COMPRESS, 0, "granule must hold whole blocks");
        assert!(initial_rows > CAPTURE_POSITION + 1);
        let reserved_rows = 3 * initial_rows;
        let max_blocks = reserved_rows / COMPRESS;
        assert!(
            max_blocks * F32_BYTES + QSA_SELECT_BATCHED_STATIC_LDS_BYTES
                <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES,
            "granule {gran} reserves a select LDS row beyond the dynamic LDS limit"
        );
        // `(position + 1) % COMPRESS == 3`: each replay position has a full
        // three-token tail that starts at a grown row and is always selected.
        let p0 = CAPTURE_POSITION;
        let p1 = initial_rows + 2;
        let p2 = 2 * initial_rows + 2;
        for position in [p1, p2] {
            assert_eq!((position + 1) % COMPRESS, 3);
            assert!(position + 1 <= reserved_rows);
        }
        assert!(p0 < initial_rows && p1 >= initial_rows && p1 < 2 * initial_rows);
        assert!(p2 >= 2 * initial_rows && p2 < reserved_rows);

        let device = gpu.device_id;
        let baseline_vmm = gpu.vmm_allocation_count();
        let vmm_tensor = |gpu: &mut Gpu, label: &str| {
            // SAFETY: only the mapped prefix is touched until the arena grows.
            unsafe { gpu.alloc_vmm_tensor(&[reserved_rows * DIM], DType::F32, gran, &[device]) }
                .unwrap_or_else(|error| panic!("{label} VMM tensor: {error}"))
        };
        let raw_keys = vmm_tensor(gpu, "raw index keys");
        let full_keys = vmm_tensor(gpu, "full keys");
        let full_values = vmm_tensor(gpu, "full values");
        assert_eq!(gpu.vmm_allocation_count(), baseline_vmm + 3);
        for tensor in [&raw_keys, &full_keys, &full_values] {
            assert_eq!(gpu.vmm_granularity(tensor), Some(gran));
            assert_eq!(gpu.vmm_mapped_bytes(tensor), Some(gran));
        }

        let rows_f32 = |first: usize, count: usize, value: &dyn Fn(usize, usize) -> f32| {
            let mut bytes = Vec::with_capacity(count * DIM * F32_BYTES);
            for row in first..first + count {
                for channel in 0..DIM {
                    bytes.extend_from_slice(&value(row, channel).to_ne_bytes());
                }
            }
            bytes
        };
        let raw_key =
            |row: usize, channel: usize| (((row * 13 + channel * 7) % 31) as f32 - 15.0) / 16.0;
        let full_key =
            |row: usize, channel: usize| (((row * 5 + channel * 3) % 11) as f32 - 5.0) / 20.0;
        // The value level steps by 256 per granule, so attention output that
        // misses (or misreads) a grown row is far from the expected level.
        let full_value = |row: usize, channel: usize| {
            (row / initial_rows) as f32 * 256.0 + 1.0 + ((row * 7 + channel * 3) % 13) as f32 / 16.0
        };

        // The initial pages are written with plain HIP copies.
        gpu.hip
            .memcpy_htod(&raw_keys.buf, &rows_f32(0, initial_rows, &raw_key))
            .expect("initial raw keys");
        gpu.hip
            .memcpy_htod(&full_keys.buf, &rows_f32(0, initial_rows, &full_key))
            .expect("initial full keys");
        gpu.hip
            .memcpy_htod(&full_values.buf, &rows_f32(0, initial_rows, &full_value))
            .expect("initial full values");

        let index_source: Vec<f32> = (0..DIM)
            .map(|channel| ((channel * 5 + 3) % 17) as f32 / 8.0 - 1.0)
            .collect();
        let index_row_bytes: Vec<u8> = index_source
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let index_query: Vec<f32> = (0..INDEX_HEADS * DIM)
            .map(|i| ((i * 3) % 7) as f32 * 0.1 + 0.05)
            .collect();
        let mut q_with_gate = Vec::with_capacity(N_HEADS * 2 * DIM);
        for head in 0..N_HEADS {
            q_with_gate.extend((0..DIM).map(|c| 0.02 * (1 + (c + head) % 5) as f32));
            q_with_gate.extend(std::iter::repeat(0.5f32).take(DIM));
        }

        let mut t = Qsa {
            reserved_rows,
            raw_keys,
            full_keys,
            full_values,
            pooled: gpu.zeros(&[max_blocks * DIM], DType::F32).expect("pooled"),
            index_src: gpu.upload_f32(&index_source, &[DIM]).expect("index source"),
            query: gpu
                .upload_f32(&index_query, &[INDEX_HEADS * DIM])
                .expect("index query"),
            q_with_gate: gpu
                .upload_f32(&q_with_gate, &[N_HEADS * 2 * DIM])
                .expect("q with gate"),
            selected: gpu
                .zeros(&[SELECTED_CAPACITY * 4], DType::Raw)
                .expect("selected"),
            output: gpu.zeros(&[N_HEADS * DIM], DType::F32).expect("output"),
        };

        // Record the tape at p0 while only the first granule is mapped.
        gpu.replay = ReplayController::new_manual_pm4();
        assert!(gpu.replay.uses_pm4_transport());
        gpu.replay.begin_capture().expect("open recording window");
        run_qsa(gpu, &t, p0);
        gpu.hip.device_synchronize().expect("sync capture");
        gpu.replay.finish_capture().expect("close recording window");
        let launches = gpu.replay.recorded_launches().len();
        assert_eq!(launches, 4, "the recorded QSA sequence changed");
        for launch in gpu.replay.recorded_launches() {
            assert!(
                !launch.declared_kernarg_bindings().is_empty(),
                "{}: position bindings were not declared",
                launch.kernel
            );
        }
        let (dispatches, _, _) = gpu
            .replay
            .prepare_pm4_prefix(gpu.device_id as usize, launches)
            .unwrap_or_else(|reason| panic!("PM4 prepare: {reason}"));
        assert_eq!(dispatches, launches);
        assert!(gpu.replay.prepared_pm4_plan_ready());
        assert!(gpu.replay.uses_pm4_transport());

        // Control inside the mapped granule, before any growth.
        let replayed_p0 = observe(gpu, &t, p0, true);
        let reference_p0 = observe(gpu, &t, p0, false);
        assert_same("pre-growth p0", &replayed_p0, &reference_p0);
        assert_eq!(replayed_p0.raw_row, index_row_bytes);

        // Map two more granules into each arena in place; the tape is not told.
        let bases = [
            t.raw_keys.buf.as_ptr() as usize,
            t.full_keys.buf.as_ptr() as usize,
            t.full_values.buf.as_ptr() as usize,
        ];
        for tensor in [&mut t.raw_keys, &mut t.full_keys, &mut t.full_values] {
            let mapped = gpu
                .grow_vmm_tensor(tensor, 2 * gran, &[device])
                .expect("grow VMM arena");
            assert_eq!(mapped, 3 * gran);
        }
        for (tensor, base) in [&t.raw_keys, &t.full_keys, &t.full_values]
            .into_iter()
            .zip(bases)
        {
            assert_eq!(tensor.buf.as_ptr() as usize, base, "growth moved the arena");
            let mapped = gpu.vmm_mapped_bytes(tensor).expect("registered VMM owner");
            assert_eq!(mapped, 3 * gran);
            assert!(mapped > gran);
        }

        // The grown pages are written with plain HIP copies after growth.
        let grown_rows = reserved_rows - initial_rows;
        gpu.hip
            .memcpy_htod_offset(
                &t.raw_keys.buf,
                gran,
                &rows_f32(initial_rows, grown_rows, &raw_key),
            )
            .expect("grown raw keys");
        gpu.hip
            .memcpy_htod_offset(
                &t.full_keys.buf,
                gran,
                &rows_f32(initial_rows, grown_rows, &full_key),
            )
            .expect("grown full keys");
        gpu.hip
            .memcpy_htod_offset(
                &t.full_values.buf,
                gran,
                &rows_f32(initial_rows, grown_rows, &full_value),
            )
            .expect("grown full values");

        let mut replayed = Vec::new();
        for position in [p1, p2] {
            let blocks = (position + 1) / COMPRESS;
            let tail_start = (blocks * COMPRESS) as i32;
            let tape = observe(gpu, &t, position, true);
            let reference = observe(gpu, &t, position, false);
            let label = format!("grown position {position}");
            assert_same(&label, &tape, &reference);

            // The kernel wrote the index-key row into a newly mapped page.
            assert_eq!(tape.raw_row, index_row_bytes, "{label}: index-key row");
            // Every pooled block (grown raw keys included) replaced the poison.
            assert!(
                tape.pooled
                    .chunks_exact(4)
                    .all(|word| f32::from_ne_bytes(word.try_into().unwrap()).is_finite()),
                "{label}: pooled prefix was not fully written"
            );
            // Selection: the chosen blocks' tokens, then the three-token tail
            // that starts at a grown row.
            let tokens = selected_tokens(&tape);
            assert_eq!(tokens.len(), SELECTED_CAPACITY);
            let chosen = BUDGET_BLOCKS * COMPRESS;
            assert!(
                tokens[..chosen]
                    .iter()
                    .all(|&token| token >= 0 && token < tail_start),
                "{label}: chosen tokens {:?}",
                &tokens[..chosen]
            );
            assert_eq!(
                &tokens[chosen..],
                &[tail_start, tail_start + 1, tail_start + 2]
            );
            assert!(
                tokens.iter().any(|&token| token as usize >= initial_rows),
                "{label}: selection never reached the grown region"
            );
            // Attention reads the grown tail's values (level >= 257), so every
            // channel lies far above the first granule's 1..2 level.
            assert!(
                tape.output
                    .iter()
                    .all(|value| value.is_finite() && *value > 8.0),
                "{label}: attention output {:?}",
                &tape.output[..4]
            );
            replayed.push(tape);
        }
        assert!(
            replayed[0]
                .output
                .iter()
                .zip(&replayed[1].output)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "positions in different grown granules produced identical outputs"
        );

        // The tape owns the addresses: keep every tensor live until it is gone.
        gpu.replay.shutdown().expect("retained replay quiescence");
        let Qsa {
            raw_keys,
            full_keys,
            full_values,
            pooled,
            index_src,
            query,
            q_with_gate,
            selected,
            output,
            ..
        } = t;
        for tensor in [
            raw_keys,
            full_keys,
            full_values,
            pooled,
            index_src,
            query,
            q_with_gate,
            selected,
            output,
        ] {
            gpu.free_tensor(tensor).expect("free tensor");
        }
        assert_eq!(gpu.vmm_allocation_count(), baseline_vmm);
    }

    /// The retained-replay shape pin (a fixed grid or LDS bound larger than the
    /// active length) must not change a single output byte: the kernels mask, so
    /// a bigger reservation is never read. This is the premise the G3 shape
    /// decision rests on, tested directly for all three QSA launches, including
    /// the `block_count == 0` case where the wrapper's symbol choice differs.
    #[test]
    fn pinned_qsa_shapes_are_bit_identical_to_derived_shapes() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };

        // ── pool: an oversized masked grid ──
        let index_dim = 8usize;
        let compress = 4usize;
        let blocks = 2usize;
        let cells = blocks * compress * index_dim;
        let raw: Vec<f32> = (0..cells)
            .map(|i| ((i * 37 % 251) as f32 - 125.0) / 37.0)
            .collect();
        let raw_gpu = gpu.upload_f32(&raw, &[cells]).expect("raw upload");
        let pool_run = |gpu: &mut Gpu, bound: usize| {
            let pooled = gpu
                .zeros(&[blocks * index_dim], DType::F32)
                .expect("pooled allocation");
            indexed_attention_pool_rope(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &raw_gpu,
                    pooled: &pooled,
                    norm: None,
                    block_count: blocks,
                    compress,
                    index_dim,
                    position: None,
                    grid_bound: bound,
                    yarn: crate::rope::RopeYarn::OFF,
                },
            )
            .expect("QSA pool/RoPE");
            let values = gpu.download_f32(&pooled).expect("pooled download");
            gpu.free_tensor(pooled).expect("free pooled");
            values
        };
        let pooled_derived = pool_run(&mut gpu, blocks);
        let pooled_pinned = pool_run(&mut gpu, blocks + 6);
        assert_eq!(
            pooled_pinned, pooled_derived,
            "a masked pool grid above the active block count changed the result"
        );

        // ── incremental pool: `first_block > 0` at a large capacity ──
        // The grid is `ceil(rows / compress)` offset by `first_block`, not the
        // capacity: every block the rows complete must equal the full pool,
        // and every block below `first_block` must stay untouched.
        let inc_blocks = 40usize;
        let inc_capacity = 65_536usize;
        let inc_cells = inc_blocks * compress * index_dim;
        let inc_raw: Vec<f32> = (0..inc_cells)
            .map(|i| ((i * 37 % 251) as f32 - 125.0) / 37.0)
            .collect();
        let inc_raw_gpu = gpu.upload_f32(&inc_raw, &[inc_cells]).expect("raw upload");
        let inc_pool = |gpu: &mut Gpu, position: Option<QsaPositionBinding>| {
            let sentinel = vec![-7.5f32; inc_blocks * index_dim];
            let pooled = gpu
                .upload_f32(&sentinel, &[sentinel.len()])
                .expect("pooled upload");
            let p = IndexedAttentionPoolRope {
                raw_keys: &inc_raw_gpu,
                pooled: &pooled,
                norm: None,
                block_count: inc_blocks,
                compress,
                index_dim,
                position,
                grid_bound: inc_capacity,
                yarn: crate::rope::RopeYarn::OFF,
            };
            match position {
                Some(_) => indexed_attention_pool_rope_incremental(gpu, &p),
                None => indexed_attention_pool_rope(gpu, &p),
            }
            .expect("QSA pool/RoPE");
            let values = gpu.download_f32(&pooled).expect("pooled download");
            gpu.free_tensor(pooled).expect("free pooled");
            values
        };
        let full = inc_pool(&mut gpu, None);
        assert!(
            full.iter().all(|v| *v != -7.5),
            "full pool left a block unwritten"
        );
        // `(position_start + rows) / compress == inc_blocks`, aligned and
        // unaligned, completing zero to three blocks.
        for (position_start, rows) in [
            (150, 10),
            (153, 7),
            (155, 5),
            (156, 4),
            (157, 3),
            (158, 2),
            (159, 1),
            (160, 3),
        ] {
            let inc = inc_pool(
                &mut gpu,
                Some(QsaPositionBinding {
                    position_start,
                    rows,
                }),
            );
            let split = position_start / compress * index_dim;
            assert!(
                inc[..split].iter().all(|v| *v == -7.5),
                "incremental pool at {position_start}+{rows} wrote below first_block"
            );
            assert_eq!(
                inc[split..],
                full[split..],
                "incremental pool at {position_start}+{rows} diverged from the full pool"
            );
        }
        gpu.free_tensor(inc_raw_gpu).expect("free raw");

        // ── select: a larger LDS reservation, at an active count and at zero ──
        let index_heads = 2usize;
        let query_elements = index_heads * index_dim;
        let rows = 1usize;
        let capacity = 8usize;
        let budget_blocks = 2usize;
        let query: Vec<f32> = (0..rows * query_elements)
            .map(|i| ((i * 53 % 199) as f32 - 100.0) / 199.0)
            .collect();
        let query_gpu = gpu
            .upload_f32(&query, &[rows * query_elements])
            .expect("query upload");
        let pooled_gpu = gpu
            .zeros(&[blocks * index_dim], DType::F32)
            .expect("pooled allocation");
        // `position_start` must satisfy the declared formula
        // `(position_start + rows) / compress == active`, so the active count and
        // the position are always consistent (the wrapper refuses a mismatch).
        let select_run = |gpu: &mut Gpu, active: usize, bound: usize, position_start: usize| {
            let selected = gpu
                .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
                .expect("selected allocation");
            indexed_attention_select_batch(
                gpu,
                &IndexedAttentionSelectBatch {
                    pooled_bf16: None,
                    pooled_bf16_valid_blocks: 0,
                    query: &query_gpu,
                    pooled: &pooled_gpu,
                    selected: &selected,
                    rows,
                    query_row_stride: query_elements,
                    block_count: active,
                    index_heads,
                    index_dim,
                    budget_blocks,
                    compress,
                    position_start,
                    capacity,
                    shape_blocks: bound,
                },
            )
            .expect("QSA select");
            let mut bytes = vec![0u8; rows * capacity * std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &selected.buf)
                .expect("selected download");
            gpu.free_tensor(selected).expect("free selected");
            bytes
        };
        let active_position = blocks * compress - rows;
        assert_eq!(
            select_run(&mut gpu, blocks, capacity, active_position),
            select_run(&mut gpu, blocks, blocks, active_position),
            "a larger select LDS reservation changed the selection"
        );
        // Zero complete blocks: position 0 with one row. This is the shape the
        // production lowering hits on the first `compress` tokens, so the
        // batched symbol at zero must equal the serial symbol it replaces.
        assert_eq!(
            select_run(&mut gpu, 0, capacity, 0),
            select_run(&mut gpu, 0, 0, 0),
            "the batched select symbol at an active count of zero diverged from the serial symbol"
        );

        // ── attention: a larger LDS reservation ──
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 4usize;
        let full_capacity = 8usize;
        let position_start = 2usize;
        let q_with_gate: Vec<f32> = (0..rows * n_heads * 2 * head_dim)
            .map(|i| ((i * 29 % 173) as f32 - 86.0) / 173.0)
            .collect();
        let kv: Vec<f32> = (0..full_capacity * n_kv_heads * head_dim)
            .map(|i| ((i * 41 % 211) as f32 - 105.0) / 211.0)
            .collect();
        let selected_tokens: Vec<i32> = vec![0, 1, 2, -1, -1, -1, -1, -1];
        let q_with_gate_gpu = gpu
            .upload_f32(&q_with_gate, &[q_with_gate.len()])
            .expect("q upload");
        let full_keys_gpu = gpu.upload_f32(&kv, &[kv.len()]).expect("keys upload");
        let full_values_gpu = gpu.upload_f32(&kv, &[kv.len()]).expect("values upload");
        let selected_gpu = gpu
            .zeros(&[rows * capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let selected_bytes = selected_tokens
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        gpu.hip
            .memcpy_htod(&selected_gpu.buf, &selected_bytes)
            .expect("selected upload");
        let attention_run = |gpu: &mut Gpu, bound: usize| {
            let output = gpu
                .zeros(&[rows * n_heads * head_dim], DType::F32)
                .expect("attention output allocation");
            indexed_attention_attention_batch(
                gpu,
                &IndexedAttentionAttentionBatch {
                    q_with_gate: &q_with_gate_gpu,
                    full_keys: &full_keys_gpu,
                    full_values: &full_values_gpu,
                    selected: &selected_gpu,
                    output: &output,
                    rows,
                    position_start,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    budget_blocks,
                    compress,
                    capacity,
                    full_capacity,
                    format: QsaKvFormat::F32,
                    shape_selected: bound,
                },
            )
            .expect("QSA attention");
            let values = gpu.download_f32(&output).expect("attention download");
            gpu.free_tensor(output).expect("free attention output");
            values
        };
        let derived_selected = (position_start + rows).min(capacity);
        assert_eq!(
            attention_run(&mut gpu, capacity),
            attention_run(&mut gpu, derived_selected),
            "a larger attention LDS reservation changed the output"
        );

        gpu.free_tensor(raw_gpu).expect("free raw");
        gpu.free_tensor(query_gpu).expect("free query");
        gpu.free_tensor(pooled_gpu).expect("free pooled");
        gpu.free_tensor(q_with_gate_gpu).expect("free q");
        gpu.free_tensor(full_keys_gpu).expect("free keys");
        gpu.free_tensor(full_values_gpu).expect("free values");
        gpu.free_tensor(selected_gpu).expect("free selected");
    }

    /// The grouped QSA attention kernels must equal the per-head batched
    /// kernel bit for bit (24 heads, head_dim 256; 2 KV heads is the
    /// production shape): permuted selections, invalid slots, a partial key
    /// tile and rows with and without a tail all included.
    #[test]
    fn qsa_attention_grouped_is_bit_identical_to_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        // hg12 (2 KV heads) over three and nine 256-row tiles, hg4 (6 KV heads).
        for (n_kv_heads, position_start, full_capacity, budget_blocks) in [
            (2usize, 610usize, 640usize, 150usize),
            (2, 2100, 2200, 512),
            (6, 610, 640, 150),
        ] {
            let (n_heads, head_dim, compress) = (24usize, 256usize, 4usize);
            let rows = 20usize;
            let capacity = budget_blocks * compress + compress - 1;
            let lcg = |seed: usize, n: usize| -> Vec<f32> {
                (0..n)
                    .map(|i| {
                        ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2003) as f32 - 1001.0)
                            / 997.0
                    })
                    .collect()
            };
            let q = lcg(1, rows * n_heads * 2 * head_dim);
            let keys = lcg(7, full_capacity * n_kv_heads * head_dim);
            let values = lcg(13, full_capacity * n_kv_heads * head_dim);
            let mut selected = vec![-1i32; rows * capacity];
            for row in 0..rows {
                let visible = position_start + row + 1;
                let blocks = visible / compress;
                let chosen = budget_blocks.min(blocks);
                // Descending-stride block choice, then the tail, as the selector emits.
                for slot in 0..chosen {
                    let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                    for r in 0..compress {
                        selected[row * capacity + slot * compress + r] =
                            block * compress as i32 + r as i32;
                    }
                }
                let mut offset = chosen * compress;
                for token in blocks * compress..visible {
                    selected[row * capacity + offset] = token as i32;
                    offset += 1;
                }
                // An invalid slot inside the active length must be skipped.
                selected[row * capacity + 3] = -1;
            }
            let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
            let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
            let values_gpu = gpu
                .upload_f32(&values, &[values.len()])
                .expect("values upload");
            let selected_gpu = gpu
                .zeros(&[selected.len() * std::mem::size_of::<i32>()], DType::Raw)
                .expect("selected allocation");
            let bytes = selected
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected_gpu.buf, &bytes)
                .expect("selected upload");
            let run = |gpu: &mut Gpu, allow_fast: bool| {
                let output = gpu
                    .zeros(&[rows * n_heads * head_dim], DType::F32)
                    .expect("output allocation");
                indexed_attention_attention_batch_impl(
                    gpu,
                    &IndexedAttentionAttentionBatch {
                        q_with_gate: &q_gpu,
                        full_keys: &keys_gpu,
                        full_values: &values_gpu,
                        selected: &selected_gpu,
                        output: &output,
                        rows,
                        position_start,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        budget_blocks,
                        compress,
                        capacity,
                        full_capacity,
                        format: QsaKvFormat::F32,
                        shape_selected: capacity,
                    },
                    allow_fast,
                )
                .expect("QSA attention");
                let values = gpu.download_f32(&output).expect("output download");
                gpu.free_tensor(output).expect("free output");
                values
            };
            let reference = run(&mut gpu, false);
            let grouped = run(&mut gpu, true);
            assert!(
                reference.iter().any(|v| *v != 0.0),
                "reference output is all zero"
            );
            let differing = reference
                .iter()
                .zip(&grouped)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "grouped QSA attention differs in {differing} cells"
            );
            for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
    }

    /// The dense F16 WMMA route (every row selects its whole causal window)
    /// must match the exact per-head kernel to F16-rounding accuracy at the
    /// production head shape, with a chunk offset and a partial row tile.
    #[test]
    fn qsa_dense_wmma_matches_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_wmma_w32() || !*crate::gemm::QWEN4_F16_WMMA {
            eprintln!("skip: needs gfx11 WMMA");
            return;
        }
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        let (rows, position_start, full_capacity) = (521usize, 100usize, 640usize);
        let (budget_blocks, capacity) = (2048usize, 640usize);
        let lcg = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2003) as f32 - 1001.0)
                        / 997.0
                })
                .collect()
        };
        let q = lcg(1, rows * n_heads * 2 * head_dim);
        let keys = lcg(7, full_capacity * n_kv_heads * head_dim);
        let values = lcg(13, full_capacity * n_kv_heads * head_dim);
        // The whole window, blocks then tail, as indexed_attention_select
        // emits it when the budget covers every block.
        let mut selected = vec![-1i32; rows * capacity];
        for row in 0..rows {
            for token in 0..position_start + row + 1 {
                selected[row * capacity + token] = token as i32;
            }
        }
        let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
        let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
        let values_gpu = gpu
            .upload_f32(&values, &[values.len()])
            .expect("values upload");
        let selected_gpu = gpu
            .zeros(&[selected.len() * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let bytes = selected
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>();
        gpu.hip
            .memcpy_htod(&selected_gpu.buf, &bytes)
            .expect("selected upload");
        let run = |gpu: &mut Gpu, allow_fast: bool| {
            let output = gpu
                .zeros(&[rows * n_heads * head_dim], DType::F32)
                .expect("output allocation");
            indexed_attention_attention_batch_impl(
                gpu,
                &IndexedAttentionAttentionBatch {
                    q_with_gate: &q_gpu,
                    full_keys: &keys_gpu,
                    full_values: &values_gpu,
                    selected: &selected_gpu,
                    output: &output,
                    rows,
                    position_start,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    budget_blocks,
                    compress,
                    capacity,
                    full_capacity,
                    format: QsaKvFormat::F32,
                    shape_selected: capacity,
                },
                allow_fast,
            )
            .expect("QSA attention");
            let values = gpu.download_f32(&output).expect("output download");
            gpu.free_tensor(output).expect("free output");
            values
        };
        let reference = run(&mut gpu, false);
        let dense = run(&mut gpu, true);
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for (r, d) in reference.iter().zip(&dense) {
            err += (*r as f64 - *d as f64).powi(2);
            norm += (*r as f64).powi(2);
        }
        let rel = (err / norm).sqrt();
        assert!(norm > 0.0, "reference output is all zero");
        // F16 operands and probabilities: ~1e-3 relative; a wrong row, head,
        // key range or dim mapping lands near 1.
        assert!(rel < 5e-3, "dense QSA WMMA rel L2 {rel:.3e}");
        // It is the WMMA route (F16 rounding), not the exact kernel.
        assert!(rel > 0.0, "dense QSA WMMA route did not run");
        for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// Device tensors of one gathered-route QSA case: `rows` query rows from
    /// `position_start` over a `full_capacity`-row cache, 24 heads over 2
    /// KV heads of 256, a 512-block budget of compress-4 blocks. Rows before
    /// and past the budget, causal tails, a `-1` slot per row, and rows whose
    /// blocks are emitted in descending token order (the per-token V path).
    /// The query is scaled so the softmax is peaked: with the flat LCG query
    /// a mis-paired key and value barely moves the output.
    struct GatheredCase {
        q: GpuTensor,
        k: GpuTensor,
        v: GpuTensor,
        selected: GpuTensor,
        output: GpuTensor,
        rows: usize,
        position_start: usize,
        full_capacity: usize,
        format: QsaKvFormat,
    }

    const GATHERED_BUDGET_BLOCKS: usize = 512;
    const GATHERED_CAPACITY: usize = GATHERED_BUDGET_BLOCKS * 4 + 4 - 1;

    impl GatheredCase {
        fn new(
            gpu: &mut Gpu,
            format: QsaKvFormat,
            rows: usize,
            position_start: usize,
            full_capacity: usize,
        ) -> Self {
            let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
            let (budget_blocks, capacity) = (GATHERED_BUDGET_BLOCKS, GATHERED_CAPACITY);
            let mut q = qsa_lcg(1, rows * n_heads * 2 * head_dim);
            for head_row in q.chunks_mut(2 * head_dim) {
                head_row[..head_dim].iter_mut().for_each(|v| *v *= 8.0);
            }
            let keys = qsa_lcg(7, full_capacity * n_kv_heads * head_dim);
            let values = qsa_lcg(13, full_capacity * n_kv_heads * head_dim);
            let mut selected = vec![-1i32; rows * capacity];
            for row in 0..rows {
                let visible = position_start + row + 1;
                let blocks = visible / compress;
                let chosen = budget_blocks.min(blocks);
                for slot in 0..chosen {
                    let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                    for r in 0..compress {
                        // Odd rows list each block's tokens in descending order.
                        let token = if row % 2 == 1 { compress - 1 - r } else { r };
                        selected[row * capacity + slot * compress + r] =
                            block * compress as i32 + token as i32;
                    }
                }
                let mut offset = chosen * compress;
                for token in blocks * compress..visible {
                    selected[row * capacity + offset] = token as i32;
                    offset += 1;
                }
                selected[row * capacity + 3] = -1;
            }
            let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
            let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
            let values_gpu = gpu
                .upload_f32(&values, &[values.len()])
                .expect("values upload");
            let (k, v) = if format == QsaKvFormat::F32 {
                (keys_gpu, values_gpu)
            } else {
                let units = format.kv_row_units(n_kv_heads, head_dim);
                let k = gpu
                    .zeros(&[full_capacity * units], format.kv_dtype())
                    .expect("K");
                let v = gpu
                    .zeros(&[full_capacity * units], format.kv_dtype())
                    .expect("V");
                indexed_attention_cache_append_batch(
                    gpu,
                    &IndexedAttentionCacheAppendBatch {
                        key: &keys_gpu,
                        value: &values_gpu,
                        full_keys: &k,
                        full_values: &v,
                        rows: full_capacity,
                        position_start: 0,
                        kv_heads: n_kv_heads,
                        head_dim,
                        format,
                    },
                )
                .expect("QSA append");
                gpu.free_tensor(keys_gpu).expect("free");
                gpu.free_tensor(values_gpu).expect("free");
                (k, v)
            };
            let selected_gpu = gpu
                .zeros(&[selected.len() * std::mem::size_of::<i32>()], DType::Raw)
                .expect("selected allocation");
            let bytes = selected
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected_gpu.buf, &bytes)
                .expect("selected upload");
            let output = gpu
                .zeros(&[rows * n_heads * head_dim], DType::F32)
                .expect("output allocation");
            Self {
                q: q_gpu,
                k,
                v,
                selected: selected_gpu,
                output,
                rows,
                position_start,
                full_capacity,
                format,
            }
        }

        fn params(&self) -> IndexedAttentionAttentionBatch<'_> {
            IndexedAttentionAttentionBatch {
                q_with_gate: &self.q,
                full_keys: &self.k,
                full_values: &self.v,
                selected: &self.selected,
                output: &self.output,
                rows: self.rows,
                position_start: self.position_start,
                n_heads: 24,
                n_kv_heads: 2,
                head_dim: 256,
                budget_blocks: GATHERED_BUDGET_BLOCKS,
                compress: 4,
                capacity: GATHERED_CAPACITY,
                full_capacity: self.full_capacity,
                format: self.format,
                shape_selected: GATHERED_CAPACITY,
            }
        }

        /// Output bytes of the gathered launches (hipcc, or the `pm`
        /// module) over a poisoned output.
        fn gathered(&self, gpu: &mut Gpu, pm: bool) -> Vec<u8> {
            gpu.hip
                .memset(&self.output.buf, 0x7f, self.output.byte_size())
                .expect("poison");
            let end = self.position_start + self.rows;
            qsa_gathered_wmma_launch_with(gpu, &self.params(), end, GATHERED_CAPACITY, pm, None)
                .expect("gathered QSA");
            gpu.hip.device_synchronize().expect("sync");
            let mut bytes = vec![0u8; self.output.byte_size()];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &self.output.buf)
                .expect("download");
            bytes
        }

        fn free(self, gpu: &mut Gpu) {
            for tensor in [self.q, self.k, self.v, self.selected, self.output] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
    }

    /// State formats the gathered route serves on this device.
    fn gathered_formats(gpu: &Gpu) -> Vec<QsaKvFormat> {
        if gpu.arch_caps.is_gfx1151() {
            vec![QsaKvFormat::F32, QsaKvFormat::Q8]
        } else if gpu.arch_caps.is_gfx1201() {
            vec![QsaKvFormat::Fp8]
        } else {
            Vec::new()
        }
    }

    /// The gathered F16 WMMA route must match the exact per-head kernel of
    /// the same state format to F16-rounding accuracy ([`GatheredCase`]),
    /// through the hipcc and the PM attend alike.
    #[test]
    fn qsa_gathered_wmma_matches_per_head_kernel() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let formats = gathered_formats(&gpu);
        if formats.is_empty() {
            eprintln!("skip: the gathered route is gfx1151 / gfx1201 only");
            return;
        }
        for format in formats {
            // Visible 1901..2420 against a 512-block budget: 475..605 blocks.
            let case = GatheredCase::new(&mut gpu, format, 520, 1900, 2432);
            let p = case.params();
            indexed_attention_attention_batch_impl(&mut gpu, &p, false).expect("per-head QSA");
            let reference = gpu.download_f32(&case.output).expect("reference download");
            qsa_gathered_wmma(&mut gpu, &p, p.position_start + p.rows, p.capacity, None)
                .expect("gathered QSA");
            let gathered = gpu.download_f32(&case.output).expect("gathered download");
            let rel = rel_l2(&reference, &gathered);
            eprintln!("gathered QSA WMMA ({}) rel L2 {rel:.3e}", format.name());
            // F16 operands and probabilities: ~1e-4 here; a wrong entry, head,
            // dim or key/value pairing lands near 1e-1 or above.
            assert!(
                rel < 5e-3,
                "gathered QSA WMMA ({}) rel L2 {rel:.3e}",
                format.name()
            );
            assert!(rel > 0.0, "gathered QSA WMMA route did not run");
            let hip = case.gathered(&mut gpu, false);
            let pm = case.gathered(&mut gpu, true);
            assert!(
                hip == pm,
                "PM vs hipcc gathered bytes differ ({})",
                format.name()
            );
            case.free(&mut gpu);
        }
    }

    /// Where admitted (`Gpu::qsa_gather_vmm_capable`), the gathered workspace
    /// is one stable-VA reservation per cache, demand-mapped: each forward
    /// maps exactly the granularity-rounded
    /// prefix it touches (K rows plus whole V blocks to its end) at an
    /// unchanged address, hipcc and the PM module stay byte-identical on it
    /// across growth, the output equals a whole-context commit's, growth is
    /// refused while a capture is armed, and weight-cache invalidation
    /// releases the reservation.
    #[test]
    fn qsa_gathered_workspace_demand_maps_one_stable_reservation() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let Some(&format) = gathered_formats(&gpu).first() else {
            eprintln!("skip: the gathered route is gfx1151 / gfx1201 only");
            return;
        };
        if !qsa_gathered_wmma_enabled(&gpu, format) {
            eprintln!("skip: the gathered route is disabled");
            return;
        }
        if !gpu.qsa_gather_vmm_capable() {
            eprintln!("skip: the gathered workspace keeps its legacy slot here");
            return;
        }
        let (heads, capacity) = (2usize, 8192usize);
        let bytes = |end: usize| qsa_gathered_wmma_scratch_bytes(heads, end).unwrap();
        gpu.invalidate_weight_caches();
        let owners = gpu.vmm_allocation_count();
        let reserved = reserve_qsa_gathered_wmma_workspace(&mut gpu, format, heads, capacity)
            .expect("reserve");
        assert_eq!(reserved, bytes(capacity));
        assert_eq!(gpu.qsa_gather_reserved_bytes(), reserved);
        assert_eq!(gpu.vmm_allocation_count(), owners + 1);
        assert!(
            gpu.scratch.qsa_gather_f16.is_none(),
            "legacy slot allocated on a VMM Gpu"
        );
        let workspace = gpu.scratch.qsa_gather_vmm.as_ref().unwrap();
        let (base, granularity) = (
            workspace.buf.as_ptr(),
            gpu.vmm_granularity(workspace).unwrap(),
        );
        assert_eq!(
            gpu.qsa_gather_scratch_bytes(),
            0,
            "a reservation commits nothing"
        );
        // Chunk 1 ends at 523 (a partial V block), chunk 2 at 5003: several
        // pages more, mapped in place.
        let mut demand_output = Vec::new();
        for (rows, start) in [(520usize, 3usize), (2480, 2523)] {
            let end = start + rows;
            let case = GatheredCase::new(&mut gpu, format, rows, start, capacity);
            let hip = case.gathered(&mut gpu, false);
            let pm = case.gathered(&mut gpu, true);
            assert!(hip == pm, "PM vs hipcc gathered bytes differ at end {end}");
            assert_eq!(
                gpu.qsa_gather_scratch_bytes(),
                bytes(end).next_multiple_of(granularity)
            );
            assert_eq!(
                gpu.scratch.qsa_gather_vmm.as_ref().unwrap().buf.as_ptr(),
                base,
                "address moved"
            );
            assert_eq!(gpu.vmm_allocation_count(), owners + 1);
            demand_output = hip;
            case.free(&mut gpu);
        }
        // Armed: a covered prefix is fine, any growth is refused.
        let mapped = gpu.qsa_gather_scratch_bytes();
        gpu.graphs.capture_mode = true;
        let covered = ensure_qsa_gathered_wmma_workspace(&mut gpu, format, heads, 5003);
        let grown = ensure_qsa_gathered_wmma_workspace(&mut gpu, format, heads, capacity);
        gpu.graphs.capture_mode = false;
        assert_eq!(covered.expect("covered while armed"), mapped);
        assert!(grown.is_err(), "growth while armed must be refused");
        assert_eq!(gpu.qsa_gather_scratch_bytes(), mapped);
        // Invalidation releases it; the whole-context commit gives the same
        // output bytes.
        gpu.invalidate_weight_caches();
        assert_eq!(gpu.qsa_gather_reserved_bytes(), 0);
        assert_eq!(gpu.vmm_allocation_count(), owners);
        assert_eq!(
            reserve_qsa_gathered_wmma_scratch(&mut gpu, format, heads, capacity).expect("commit"),
            reserved
        );
        assert!(gpu.qsa_gather_scratch_bytes() >= reserved);
        let case = GatheredCase::new(&mut gpu, format, 2480, 2523, capacity);
        assert!(
            case.gathered(&mut gpu, false) == demand_output,
            "demand-mapped output differs from whole commit"
        );
        case.free(&mut gpu);
        gpu.invalidate_weight_caches();
        assert_eq!(gpu.vmm_allocation_count(), owners);
    }

    struct SelectCase {
        compress: usize,
        index_heads: usize,
        index_dim: usize,
        rows: usize,
        block_count: usize,
        budget_blocks: usize,
        capacity: usize,
        position_start: usize,
    }

    fn run_select_case(
        gpu: &mut Gpu,
        case: &SelectCase,
        pooled: &[f32],
        query: &[f32],
        shape_blocks: usize,
    ) -> Vec<i32> {
        let pooled_gpu = gpu
            .upload_f32(pooled, &[pooled.len().max(1)])
            .expect("pooled upload");
        let query_gpu = gpu
            .upload_f32(query, &[query.len().max(1)])
            .expect("query upload");
        let selected = gpu
            .zeros(&[case.rows * case.capacity * 4], DType::Raw)
            .expect("selected allocation");
        indexed_attention_select_batch(
            gpu,
            &IndexedAttentionSelectBatch {
                pooled_bf16: None,
                pooled_bf16_valid_blocks: 0,
                query: &query_gpu,
                pooled: &pooled_gpu,
                selected: &selected,
                rows: case.rows,
                query_row_stride: case.index_heads * case.index_dim,
                block_count: case.block_count,
                index_heads: case.index_heads,
                index_dim: case.index_dim,
                budget_blocks: case.budget_blocks,
                compress: case.compress,
                position_start: case.position_start,
                capacity: case.capacity,
                shape_blocks,
            },
        )
        .expect("QSA select");
        let mut bytes = vec![0u8; case.rows * case.capacity * 4];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &selected.buf)
            .expect("selected download");
        gpu.free_tensor(selected).expect("free selected");
        gpu.free_tensor(pooled_gpu).expect("free pooled");
        gpu.free_tensor(query_gpu).expect("free query");
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 chunk")))
            .collect()
    }

    /// Arena bound whose score row exceeds the LDS limit: the global route.
    const GLOBAL_BOUND: usize = QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES / 4 + 1;

    /// The serial selection-sort oracle kernel for `case`.
    fn run_select_serial(
        gpu: &mut Gpu,
        case: &SelectCase,
        pooled: &[f32],
        query: &[f32],
    ) -> Vec<i32> {
        SELECT_FORCE_SERIAL.with(|flag| flag.set(true));
        let selected = run_select_case(gpu, case, pooled, query, case.block_count);
        SELECT_FORCE_SERIAL.with(|flag| flag.set(false));
        selected
    }

    /// Scores that increase with the block index must select blocks in
    /// descending order, and equal scores must break ties toward the lower
    /// block index — the order the serial scan's strict `>` comparison
    /// produced. This pins the semantics rather than mutual agreement.
    #[test]
    fn batched_select_orders_blocks_by_score_then_index() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let compress = 4usize;
        let index_dim = 4usize;
        // Block b contributes b to the dot product, so score(b) = b / 2.
        let pooled: Vec<f32> = (0..4 * index_dim)
            .map(|i| {
                if i % index_dim == 0 {
                    (i / index_dim) as f32
                } else {
                    0.0
                }
            })
            .collect();
        let query = vec![1.0f32; index_dim];
        let case = SelectCase {
            compress,
            index_heads: 1,
            index_dim,
            rows: 1,
            block_count: 4,
            budget_blocks: 8,
            capacity: 18,
            position_start: 4 * compress,
        };
        // Slot 16 is the tail: block 4 covers tokens 16..19 but `visible` is 17,
        // so the partial block still contributes its one visible token.
        let selected = run_select_case(&mut gpu, &case, &pooled, &query, case.block_count);
        assert_eq!(
            selected,
            vec![12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 16, -1],
            "descending-score selection changed"
        );

        // All-zero pooled keys tie every block at score 0.
        let tied = vec![0.0f32; 4 * index_dim];
        let selected = run_select_case(&mut gpu, &case, &tied, &query, case.block_count);
        assert_eq!(
            selected,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1],
            "tie-break by block index changed"
        );
    }

    /// `indexed_attention_select_from_scores` on synthetic score rows with
    /// tens of thousands of blocks tied at the threshold (65,536 is 262K
    /// context at compress 4). The above/equal counts once shared one packed
    /// int whose 16-bit equal field went negative past 32,767 ties and
    /// wrapped past 65,535. Up to 65,536 blocks the old code only wrote to
    /// negative LDS indices, which neither the output check nor the canary
    /// can see; only the 70,000-block cases, where later ties wrap back into
    /// the first selection slots, detect it. The rest pin the selection.
    #[test]
    fn select_from_scores_survives_tens_of_thousands_of_ties() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        gpu.ensure_kernel_public(
            "tensor_ops",
            TENSOR_OPS_SRC,
            "indexed_attention_select_from_scores",
        )
        .expect("select kernel");
        const CANARY: i32 = 0x5A5A_5A5A;
        const GUARD: usize = 64;
        let (compress, budget, tail) = (4usize, 512usize, 3usize);
        let capacity = budget * compress + tail;
        let tied = |blocks: usize| vec![1.0f32; blocks];
        // 300 blocks scored above the tie, scattered, with some repeats so
        // the above blocks also break ties by index.
        let partial = |blocks: usize| {
            let mut scores = vec![1.0f32; blocks];
            for i in 0..300usize {
                scores[(i * 7919 + 13) % blocks] = 2.0 + (i % 97) as f32 * 0.25;
            }
            scores
        };
        let cases: Vec<(&str, Vec<f32>)> = vec![
            ("all tied 65536", tied(65_536)),
            ("all tied 40000", tied(40_000)),
            ("partial tie 65536", partial(65_536)),
            ("partial tie 40000", partial(40_000)),
            ("all tied 70000", tied(70_000)),
            ("partial tie 70000", partial(70_000)),
        ];
        for (name, scores) in cases {
            let blocks = scores.len();
            // Visible spans every block plus `tail` tail tokens.
            let position_start = blocks * compress + tail - 1;
            let mut order: Vec<usize> = (0..blocks).collect();
            order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
            let mut expected = vec![-1i32; capacity];
            for (rank, &block) in order.iter().take(budget).enumerate() {
                for slot in 0..compress {
                    expected[rank * compress + slot] = (block * compress + slot) as i32;
                }
            }
            for t in 0..tail {
                expected[budget * compress + t] = (blocks * compress + t) as i32;
            }
            let scores_gpu = gpu.upload_f32(&scores, &[blocks]).expect("scores upload");
            let init: Vec<u8> = std::iter::repeat(CANARY)
                .take(capacity + GUARD)
                .flat_map(i32::to_ne_bytes)
                .collect();
            let selected = gpu
                .upload_raw(&init, &[init.len()])
                .expect("selected upload");
            let mut args = KernargBlob::new();
            args.push_ptr(scores_gpu.buf.as_ptr());
            args.push_i32(blocks as i32);
            args.push_ptr(selected.buf.as_ptr());
            args.push_i32(1);
            args.push_i32(blocks as i32);
            args.push_i32(budget as i32);
            args.push_i32(compress as i32);
            args.push_i32(position_start as i32);
            args.push_i32(capacity as i32);
            args.push_ptr(std::ptr::null_mut());
            args.pad_to(16);
            gpu.launch_blob_recorded(
                "indexed_attention_select_from_scores",
                [1, 1, 1],
                [256, 1, 1],
                0,
                args.as_mut_slice(),
                crate::dispatch::ReplayLaunchBindings::NONE,
            )
            .expect("select launch");
            let mut bytes = vec![0u8; init.len()];
            gpu.hip
                .memcpy_dtoh(&mut bytes, &selected.buf)
                .expect("selected download");
            gpu.free_tensor(selected).expect("free selected");
            gpu.free_tensor(scores_gpu).expect("free scores");
            let got: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 chunk")))
                .collect();
            assert!(
                got[capacity..].iter().all(|&v| v == CANARY),
                "{name}: guard canary after the selection overwritten"
            );
            let first_bad = (0..capacity).find(|&i| got[i] != expected[i]);
            assert!(
                first_bad.is_none(),
                "{name}: selection differs from CPU reference at slot {first_bad:?}: got {:?} want {:?}",
                first_bad.map(|i| got[i]),
                first_bad.map(|i| expected[i])
            );
        }
    }

    /// The parallel rank path and the serial selection-sort fallback are two
    /// symbols for the same logical selection, chosen only by the LDS
    /// reservation; they must emit identical `selected` bytes. A
    /// `shape_blocks` above the dynamic-LDS limit selects the serial symbol.
    #[test]
    fn batched_select_ranking_matches_the_serial_selection_sort() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let mut state = 0x9e37_79b9u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 65_536.0) - 8.0
        };

        for &compress in &[2usize, 4, 8] {
            for &index_dim in &[8usize, 128] {
                for &index_heads in &[1usize, 4] {
                    // 37 rows: three sixteen-row scoring groups, the last
                    // partial, each row with its own visible block count.
                    for &rows in &[1usize, 5, 37] {
                        // `(position_start + rows) / compress` must equal the
                        // wrapper's declared block count.
                        if rows < 37 && rows > compress - 1 {
                            continue;
                        }
                        for &block_count in &[0usize, 1, 3, 17, 72, 128, 500] {
                            for &budget_blocks in &[1usize, 4, 64, 512] {
                                // Many rows: the pinned geometry (the live
                                // rows16 route), smaller budgets.
                                let pinned = compress == 4 && index_heads == 4 && index_dim == 128;
                                if rows == 37
                                    && (!pinned
                                        || budget_blocks > 64
                                        || block_count * compress + compress - 1 < rows)
                                {
                                    continue;
                                }
                                let case = SelectCase {
                                    compress,
                                    index_heads,
                                    index_dim,
                                    rows,
                                    block_count,
                                    budget_blocks,
                                    capacity: budget_blocks * compress + compress - 1,
                                    position_start: if rows < compress {
                                        block_count * compress
                                    } else {
                                        block_count * compress + compress - 1 - rows
                                    },
                                };
                                let pooled: Vec<f32> = (0..block_count * index_dim + index_dim)
                                    .map(|_| next())
                                    .collect();
                                let query: Vec<f32> = (0..rows * index_heads * index_dim)
                                    .map(|_| next())
                                    .collect();
                                let parallel =
                                    run_select_case(&mut gpu, &case, &pooled, &query, block_count);
                                let serial = run_select_serial(&mut gpu, &case, &pooled, &query);
                                let global =
                                    run_select_case(&mut gpu, &case, &pooled, &query, GLOBAL_BOUND);
                                assert_eq!(
                                    global, serial,
                                    "global-score selection diverged from the serial selection sort: \
                                     compress={compress} heads={index_heads} dim={index_dim} \
                                     rows={rows} blocks={block_count} budget={budget_blocks}"
                                );
                                assert_eq!(
                                    parallel, serial,
                                    "parallel ranking diverged from the serial selection sort: \
                                     compress={compress} heads={index_heads} dim={index_dim} \
                                     rows={rows} blocks={block_count} budget={budget_blocks}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Production QSA geometry past its budget (512 of 4096 blocks, 4 heads x
    /// 128, compress 4): the parallel radix-select route must emit the serial
    /// selection sort's bytes, including when the budget boundary falls inside
    /// a run of equal scores (ReLU zeros), where the lower block index wins.
    #[test]
    fn past_budget_select_matches_serial_on_production_geometry() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (compress, index_heads, index_dim, rows, block_count, budget_blocks) =
            (4usize, 4usize, 128usize, 3usize, 4096usize, 512usize);
        let mut state = 0x2545_f491u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0
        };
        // Positive queries: a block whose pooled row is all negative scores 0.
        let query: Vec<f32> = (0..rows * index_heads * index_dim)
            .map(|_| next())
            .collect();
        // positive_every: 1 = every block scores > 0 (threshold above zero);
        // 10 = ~410 positive blocks, so the boundary is inside the zero run;
        // 0 = every block ties at zero.
        for positive_every in [1usize, 10, 0] {
            let pooled: Vec<f32> = (0..block_count * index_dim)
                .map(|i| {
                    let block = i / index_dim;
                    let value = next() - 0.25;
                    if positive_every != 0 && block % positive_every == 0 {
                        value
                    } else {
                        -value.abs() - 0.01
                    }
                })
                .collect();
            let case = SelectCase {
                compress,
                index_heads,
                index_dim,
                rows,
                block_count,
                budget_blocks,
                capacity: budget_blocks * compress + compress - 1,
                position_start: block_count * compress,
            };
            let parallel = run_select_case(&mut gpu, &case, &pooled, &query, block_count);
            let serial = run_select_serial(&mut gpu, &case, &pooled, &query);
            let global = run_select_case(&mut gpu, &case, &pooled, &query, GLOBAL_BOUND);
            assert_eq!(
                parallel, serial,
                "past-budget radix select diverged (positive_every={positive_every})"
            );
            assert_eq!(
                global, serial,
                "past-budget global-score select diverged (positive_every={positive_every})"
            );
            if positive_every == 0 {
                // All tied: the lowest 512 block indices, in index order.
                let expected: Vec<i32> = (0..(budget_blocks * compress) as i32).collect();
                assert_eq!(&parallel[..budget_blocks * compress], &expected[..]);
            }
        }
    }

    /// A prefill-sized batch (more rows than one global-score launch holds)
    /// runs in row groups; each row's selection must equal the single-launch
    /// LDS route's, including rows whose visible prefix ends mid-block.
    #[test]
    fn global_score_row_groups_match_the_lds_route() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (compress, index_heads, index_dim, rows, block_count, budget_blocks) = (
            4usize,
            4usize,
            128usize,
            2 * QSA_SELECT_GLOBAL_ROWS + 37,
            4096usize,
            512usize,
        );
        let mut state = 0x7f4a_7c15u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 8_388_608.0 - 1.0
        };
        let query: Vec<f32> = (0..rows * index_heads * index_dim)
            .map(|_| next())
            .collect();
        let pooled: Vec<f32> = (0..block_count * index_dim).map(|_| next()).collect();
        let case = SelectCase {
            compress,
            index_heads,
            index_dim,
            rows,
            block_count,
            budget_blocks,
            capacity: budget_blocks * compress + compress - 1,
            // (position_start + rows) / compress == block_count.
            position_start: block_count * compress - rows + 2,
        };
        let lds = run_select_case(&mut gpu, &case, &pooled, &query, block_count);
        let global = run_select_case(&mut gpu, &case, &pooled, &query, GLOBAL_BOUND);
        assert_eq!(
            global, lds,
            "grouped global-score selection diverged from the LDS route"
        );
    }

    #[test]
    fn qsa_reuse_selection_preserves_order_and_boundaries() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let capacity = 4usize;
        let mut selected = gpu
            .zeros(&[capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let selected_len_out = gpu
            .zeros(&[std::mem::size_of::<i32>()], DType::Raw)
            .expect("length allocation");
        let cases: [([i32; 4], usize, usize, [i32; 4], i32); 4] = [
            ([-1, -1, -1, -1], 0usize, 7usize, [7, -1, -1, -1], 1),
            ([0, 3, 1, 99], 4, 2, [0, 1, 2, -1], 3),
            ([0, 2, 1, -1], 3, 2, [0, 2, 1, -1], 3),
            ([0, 1, 2, 3], 4, 4, [0, 1, 2, 3], 4),
        ];
        for (input, selected_len, position, expected, expected_len) in cases {
            let input_bytes = input
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected.buf, &input_bytes)
                .expect("upload selected row");
            gpu.hip
                .memset(&selected_len_out.buf, 0, selected_len_out.byte_size())
                .expect("clear selected length");
            indexed_attention_reuse_selection(
                &mut gpu,
                &IndexedAttentionReuseSelection {
                    selected: &selected,
                    selected_len,
                    position,
                    capacity,
                    selected_len_out: &selected_len_out,
                },
            )
            .expect("reuse selection");

            let mut output_bytes = vec![0u8; capacity * std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut output_bytes, &selected.buf)
                .expect("download selected row");
            let output = output_bytes
                .chunks_exact(std::mem::size_of::<i32>())
                .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
                .collect::<Vec<_>>();
            assert_eq!(output, expected);
            let mut length_bytes = [0u8; std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut length_bytes, &selected_len_out.buf)
                .expect("download selected length");
            assert_eq!(i32::from_ne_bytes(length_bytes), expected_len);
        }
        gpu.free_tensor(selected).expect("free selected");
        gpu.free_tensor(selected_len_out)
            .expect("free selected length");
    }
    #[test]
    fn qsa_norm_rope_matches_production_shape_and_preserves_gates() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const HEAD_STRIDE: usize = 2 * HEAD_DIM;
        const ROTARY_DIM: usize = 64;
        let mut input = vec![0.0f32; HEADS * HEAD_STRIDE];
        for head in 0..HEADS {
            for channel in 0..HEAD_DIM {
                input[head * HEAD_STRIDE + channel] =
                    (((head * HEAD_DIM + channel) * 37 % 251) as f32 - 125.0) / 37.0;
                input[head * HEAD_STRIDE + HEAD_DIM + channel] =
                    1000.0 + (head * HEAD_DIM + channel) as f32;
            }
        }
        let norm = gpu
            .zeros(&[HEAD_DIM], DType::BF16)
            .expect("zero norm allocation");
        // Plain RoPE at a short position, and the Qwen3.8 README YaRN override
        // (factor 4 over 262144) against the host HF reference, at a position
        // where interpolated pairs and the cos/sin factor are both visible.
        use crate::rope::{Rope, RopeScaling, Yarn};
        let yarn = RopeScaling::Yarn(Yarn::new(4.0, 262_144, None, None, None, None));
        for (scaling, position, tolerance) in
            [(RopeScaling::None, 11, 1.0e-4), (yarn, 4099, 2.0e-3)]
        {
            let rope = Rope {
                theta: 10_000_000.0,
                scaling,
            };
            let values = gpu
                .upload_f32(&input, &[input.len()])
                .expect("query/gate upload");
            indexed_attention_norm_rope(
                &mut gpu,
                &IndexedAttentionNormRope {
                    values: &values,
                    norm: &norm,
                    heads: HEADS,
                    head_dim: HEAD_DIM,
                    head_stride: HEAD_STRIDE,
                    position,
                    rotary_dim: ROTARY_DIM,
                    yarn: rope.yarn(ROTARY_DIM),
                },
            )
            .expect("QSA norm/RoPE");
            let actual = gpu.download_f32(&values).expect("query/gate download");
            gpu.free_tensor(values).expect("free query/gate");
            let (inv_freq, mscale) = rope.inv_freq(ROTARY_DIM);
            let mut expected = input.clone();
            for head in 0..HEADS {
                let start = head * HEAD_STRIDE;
                let inv = (input[start..start + HEAD_DIM]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / HEAD_DIM as f32
                    + 1.0e-6)
                    .sqrt()
                    .recip();
                for channel in 0..HEAD_DIM {
                    expected[start + channel] = input[start + channel] * inv;
                }
                for channel in 0..ROTARY_DIM / 2 {
                    let (sine, cosine) = (position as f64 * inv_freq[channel]).sin_cos();
                    let (sine, cosine) = ((sine * mscale) as f32, (cosine * mscale) as f32);
                    let first = input[start + channel] * inv;
                    let second = input[start + ROTARY_DIM / 2 + channel] * inv;
                    expected[start + channel] = first * cosine - second * sine;
                    expected[start + ROTARY_DIM / 2 + channel] = first * sine + second * cosine;
                }
            }
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() <= tolerance,
                    "QSA norm/RoPE {scaling:?} mismatch at {index}: got {got}, expected {want}"
                );
            }
        }
        gpu.free_tensor(norm).expect("free norm");
    }

    /// Formats this device has QSA kernels for, beyond F32 (fp8: gfx12;
    /// fp8 FNUZ: gfx942; q8: gfx11 and gfx12).
    fn quantized_qsa_formats(gpu: &Gpu) -> Vec<QsaKvFormat> {
        let mut formats = Vec::new();
        if gpu.arch_caps.is_gfx1200() || gpu.arch_caps.is_gfx1201() {
            formats.push(QsaKvFormat::Fp8);
        }
        if gpu.arch_caps.is_gfx942() {
            formats.push(QsaKvFormat::Fp8Fnuz);
        }
        if gpu.arch_caps.has_gfx11_plus_simt() {
            formats.push(QsaKvFormat::Q8);
        }
        if formats.is_empty() {
            eprintln!("skip: quantized QSA K/V needs a gfx11, gfx12 or gfx942 GPU");
        }
        formats
    }

    fn qsa_lcg(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) % 2003) as f32 - 1001.0) / 997.0
            })
            .collect()
    }

    fn f16_to_f32(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1F) as i32;
        let mantissa = (bits & 0x3FF) as f32;
        sign * if exponent == 0 {
            mantissa / 1024.0 * 2f32.powi(-14)
        } else {
            (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
        }
    }

    /// Round-to-nearest-even BF16 bits of a finite value.
    fn bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn e4m3_to_f32(code: u8) -> f32 {
        let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
        let exponent = ((code >> 3) & 0xF) as i32;
        let mantissa = (code & 7) as f32;
        sign * if exponent == 0 {
            mantissa / 8.0 * 2f32.powi(-6)
        } else {
            (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
        }
    }

    /// Host dequantization of `tokens` quantized K or V rows.
    fn dequantize_qsa_rows(
        format: QsaKvFormat,
        bytes: &[u8],
        tokens: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let row = format.kv_row_bytes(kv_heads, head_dim);
        let half = |at: usize| f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
        let mut values = Vec::with_capacity(tokens * kv_heads * head_dim);
        for token in 0..tokens {
            let base = token * row;
            for element in 0..kv_heads * head_dim {
                values.push(match format {
                    QsaKvFormat::Fp8 | QsaKvFormat::Fp8Fnuz => {
                        let scale = half(base + kv_heads * head_dim + element / head_dim * 2);
                        e4m3_to_f32(bytes[base + element]) * scale
                    }
                    QsaKvFormat::Q8 => {
                        let (head, d) = (element / head_dim, element % head_dim);
                        let block = base + (head * (head_dim / 32) + d / 32) * 34;
                        bytes[block + 2 + d % 32] as i8 as f32 * half(block)
                    }
                    QsaKvFormat::F32 => unreachable!("F32 rows are not quantized"),
                });
            }
        }
        values
    }

    fn rel_l2(reference: &[f32], actual: &[f32]) -> f64 {
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for (r, a) in reference.iter().zip(actual) {
            err += (*r as f64 - *a as f64).powi(2);
            norm += (*r as f64).powi(2);
        }
        (err / norm).sqrt()
    }

    fn download_bytes(gpu: &Gpu, tensor: &GpuTensor) -> Vec<u8> {
        let mut bytes = vec![0u8; tensor.byte_size()];
        gpu.hip
            .memcpy_dtoh(&mut bytes, &tensor.buf)
            .expect("download");
        bytes
    }

    /// The decode prologue and the prefill appends must write the same
    /// quantized K/V rows and BF16 raw index keys for the same rows (decode
    /// after prefill reads what token-by-token decode would have written),
    /// and those rows must dequantize to the keys and values they encode.
    #[test]
    fn qsa_quantized_prologue_and_prefill_appends_write_identical_rows() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (heads, kv_heads, head_dim) = (4usize, 2usize, 256usize);
        let (index_heads, index_dim, index_kv_width) = (4usize, 128usize, 128usize);
        let (rows, position, capacity) = (3usize, 5usize, 16usize);
        let index_width = index_heads * index_dim + index_kv_width;
        let kv_width = kv_heads * head_dim;
        for format in quantized_qsa_formats(&gpu) {
            let row_units = format.kv_row_units(kv_heads, head_dim);
            let index_row = qsa_lcg(3, rows * index_width);
            let index_gpu = gpu
                .upload_f32(&index_row, &[index_row.len()])
                .expect("index");
            let qgate = qsa_lcg(5, rows * 2 * heads * head_dim);
            let qgate_gpu = gpu.upload_f32(&qgate, &[qgate.len()]).expect("qgate");
            let keys = qsa_lcg(7, rows * kv_width);
            let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys");
            let values = qsa_lcg(11, rows * kv_width);
            let values_gpu = gpu.upload_f32(&values, &[values.len()]).expect("values");
            let index_norm = gpu.zeros(&[index_dim], DType::BF16).expect("index norm");
            let head_norm = gpu.zeros(&[head_dim], DType::BF16).expect("head norm");
            let cache = |gpu: &mut Gpu| {
                gpu.zeros(&[capacity * row_units], format.kv_dtype())
                    .expect("cache")
            };
            let (decode_k, decode_v, prefill_k, prefill_v) = (
                cache(&mut gpu),
                cache(&mut gpu),
                cache(&mut gpu),
                cache(&mut gpu),
            );
            let decode_raw = gpu
                .zeros(&[capacity * index_kv_width], DType::BF16)
                .expect("raw");
            let prefill_raw = gpu
                .zeros(&[capacity * index_kv_width], DType::BF16)
                .expect("raw");
            indexed_attention_decode_prologue(
                &mut gpu,
                &IndexedAttentionDecodePrologue {
                    index_row: &index_gpu,
                    qgate: &qgate_gpu,
                    keys: &keys_gpu,
                    values: &values_gpu,
                    full_keys: &decode_k,
                    full_values: &decode_v,
                    raw_index_keys: &decode_raw,
                    index_q_norm: &index_norm,
                    q_norm: &head_norm,
                    k_norm: &head_norm,
                    index_heads,
                    index_dim,
                    index_kv_width,
                    heads,
                    kv_heads,
                    head_dim,
                    position,
                    rows,
                    format,
                    yarn: crate::rope::RopeYarn::OFF,
                },
            )
            .expect("QSA prologue");
            // The prologue leaves the normed keys and rounded index keys in
            // its scratch: the prefill appends start from those.
            indexed_attention_cache_append_batch(
                &mut gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &keys_gpu,
                    value: &values_gpu,
                    full_keys: &prefill_k,
                    full_values: &prefill_v,
                    rows,
                    position_start: position,
                    kv_heads,
                    head_dim,
                    format,
                },
            )
            .expect("QSA append");
            indexed_attention_index_key_append_batch(
                &mut gpu,
                &IndexedAttentionIndexKeyAppendBatch {
                    index_rows: &index_gpu,
                    raw_index_keys: &prefill_raw,
                    rows,
                    index_q_width: index_heads * index_dim,
                    index_kv_width,
                    position_start: position,
                },
            )
            .expect("QSA index-key append");
            let name = format.name();
            for (decode, prefill, what) in [
                (&decode_k, &prefill_k, "keys"),
                (&decode_v, &prefill_v, "values"),
                (&decode_raw, &prefill_raw, "raw index keys"),
            ] {
                assert_eq!(
                    download_bytes(&gpu, decode),
                    download_bytes(&gpu, prefill),
                    "{name} prologue and prefill {what} differ"
                );
            }
            let row_bytes = format.kv_row_bytes(kv_heads, head_dim);
            let tail =
                |tensor: &GpuTensor| download_bytes(&gpu, tensor)[position * row_bytes..].to_vec();
            let normed = gpu.download_f32(&keys_gpu).expect("normed keys");
            let (k_rows, v_rows) = (tail(&decode_k), tail(&decode_v));
            let dequant_k = dequantize_qsa_rows(format, &k_rows, rows, kv_heads, head_dim);
            let dequant_v = dequantize_qsa_rows(format, &v_rows, rows, kv_heads, head_dim);
            // E4M3 keeps 3 mantissa bits, Q8_0 7 bits per 32-element block. A
            // wrong head, block or scale lands near 1.
            let tolerance = if format == QsaKvFormat::Q8 {
                1e-2
            } else {
                5e-2
            };
            for (reference, actual, bytes, what) in [
                (&normed, &dequant_k, &k_rows, "keys"),
                (&values, &dequant_v, &v_rows, "values"),
            ] {
                let rel = rel_l2(reference, actual);
                assert!(rel < tolerance, "{name} {what} rel L2 {rel:.3e}");
                if format != QsaKvFormat::Q8 {
                    continue;
                }
                // Q8_0 rounds against the f32 scale amax / 127 and stores it
                // as f16: every element is within half an f32 step plus 127
                // times the scale's f16 rounding. Blocks are consecutive
                // 34-byte runs, in element order.
                for (block, (want, got)) in reference
                    .chunks_exact(32)
                    .zip(actual.chunks_exact(32))
                    .enumerate()
                {
                    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
                    let stored = f16_to_f32(u16::from_le_bytes([
                        bytes[34 * block],
                        bytes[34 * block + 1],
                    ]));
                    let bound = 0.5 * scale * (1.0 + 1e-4) + 127.0 * (scale - stored).abs();
                    for (w, g) in want.iter().zip(got) {
                        assert!(
                            (w - g).abs() <= bound,
                            "{name} {what} block {block}: {g} vs {w} past the int8 bound {bound:.3e}"
                        );
                    }
                }
            }
            for tensor in [
                index_gpu,
                qgate_gpu,
                keys_gpu,
                values_gpu,
                index_norm,
                head_norm,
                decode_k,
                decode_v,
                prefill_k,
                prefill_v,
                decode_raw,
                prefill_raw,
            ] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
    }

    /// Every quantized attention kernel must agree bit for bit with its
    /// per-head twin (grouped hg4 batch, single-row) and stay near the F32
    /// reference at the production head shape.
    #[test]
    fn qsa_quantized_attention_kernels_agree() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        let (n_heads, n_kv_heads, head_dim, compress) = (24usize, 2usize, 256usize, 4usize);
        let (rows, position_start, full_capacity) = (20usize, 610usize, 640usize);
        let budget_blocks = 150usize;
        let capacity = budget_blocks * compress + compress - 1;
        let kv_width = n_kv_heads * head_dim;
        let q = qsa_lcg(1, rows * n_heads * 2 * head_dim);
        let keys = qsa_lcg(7, full_capacity * kv_width);
        let values = qsa_lcg(13, full_capacity * kv_width);
        let mut selected = vec![-1i32; rows * capacity];
        for row in 0..rows {
            let visible = position_start + row + 1;
            let blocks = visible / compress;
            let chosen = budget_blocks.min(blocks);
            for slot in 0..chosen {
                let block = (blocks - 1 - (slot * 7 + row) % blocks) as i32;
                for r in 0..compress {
                    selected[row * capacity + slot * compress + r] =
                        block * compress as i32 + r as i32;
                }
            }
            let mut offset = chosen * compress;
            for token in blocks * compress..visible {
                selected[row * capacity + offset] = token as i32;
                offset += 1;
            }
            selected[row * capacity + 3] = -1;
        }
        let q_gpu = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
        let keys_gpu = gpu.upload_f32(&keys, &[keys.len()]).expect("keys upload");
        let values_gpu = gpu
            .upload_f32(&values, &[values.len()])
            .expect("values upload");
        let selected_gpu = gpu
            .zeros(&[selected.len() * 4], DType::Raw)
            .expect("selected allocation");
        let selected_bytes: Vec<u8> = selected.iter().flat_map(|v| v.to_ne_bytes()).collect();
        gpu.hip
            .memcpy_htod(&selected_gpu.buf, &selected_bytes)
            .expect("selected upload");
        let run =
            |gpu: &mut Gpu, format: QsaKvFormat, k: &GpuTensor, v: &GpuTensor, allow_fast: bool| {
                let output = gpu
                    .zeros(&[rows * n_heads * head_dim], DType::F32)
                    .expect("output");
                indexed_attention_attention_batch_impl(
                    gpu,
                    &IndexedAttentionAttentionBatch {
                        q_with_gate: &q_gpu,
                        full_keys: k,
                        full_values: v,
                        selected: &selected_gpu,
                        output: &output,
                        rows,
                        position_start,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        budget_blocks,
                        compress,
                        capacity,
                        full_capacity,
                        format,
                        shape_selected: capacity,
                    },
                    allow_fast,
                )
                .expect("QSA attention");
                let result = gpu.download_f32(&output).expect("output download");
                gpu.free_tensor(output).expect("free output");
                result
            };
        let reference = run(&mut gpu, QsaKvFormat::F32, &keys_gpu, &values_gpu, false);
        for format in quantized_qsa_formats(&gpu) {
            let name = format.name();
            let row_units = format.kv_row_units(n_kv_heads, head_dim);
            let k = gpu
                .zeros(&[full_capacity * row_units], format.kv_dtype())
                .expect("K");
            let v = gpu
                .zeros(&[full_capacity * row_units], format.kv_dtype())
                .expect("V");
            indexed_attention_cache_append_batch(
                &mut gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &keys_gpu,
                    value: &values_gpu,
                    full_keys: &k,
                    full_values: &v,
                    rows: full_capacity,
                    position_start: 0,
                    kv_heads: n_kv_heads,
                    head_dim,
                    format,
                },
            )
            .expect("QSA append");
            let per_head = run(&mut gpu, format, &k, &v, false);
            let grouped = run(&mut gpu, format, &k, &v, true);
            let differing = per_head
                .iter()
                .zip(&grouped)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                differing, 0,
                "{name} grouped attention differs in {differing} cells"
            );
            let rel = rel_l2(&reference, &per_head);
            let tolerance = if format == QsaKvFormat::Q8 {
                1e-2
            } else {
                5e-2
            };
            assert!(
                rel > 0.0 && rel < tolerance,
                "{name} attention rel L2 {rel:.3e}"
            );
            // The single-row entry (explicit selection length) is the batched
            // per-head body for that row.
            let row = rows - 1;
            let visible = position_start + row + 1;
            let blocks = visible / compress;
            let selected_len = budget_blocks.min(blocks) * compress + visible - blocks * compress;
            let row_q = q_gpu.sub_offset(row * n_heads * 2 * head_dim, n_heads * 2 * head_dim);
            let row_selected = selected_gpu.sub_offset(row * capacity * 4, capacity * 4);
            let output = gpu
                .zeros(&[n_heads * head_dim], DType::F32)
                .expect("output");
            indexed_attention_attention(
                &mut gpu,
                &IndexedAttentionAttention {
                    q_with_gate: &row_q,
                    full_keys: &k,
                    full_values: &v,
                    selected: &row_selected,
                    output: &output,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    selected_len,
                    full_capacity: visible,
                    end_position: visible,
                    format,
                },
            )
            .expect("QSA single-row attention");
            let single = gpu.download_f32(&output).expect("single download");
            let batched_row = &per_head[row * n_heads * head_dim..];
            assert!(
                single
                    .iter()
                    .zip(batched_row)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name} single-row attention differs from the batched row"
            );
            for tensor in [output, k, v] {
                gpu.free_tensor(tensor).expect("free");
            }
        }
        for tensor in [q_gpu, keys_gpu, values_gpu, selected_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// POOL primitive only: all live bytes agree with real singleton launches,
    /// including every kept prefix, abort, and continuation. This is not a
    /// select -> attend -> commit ticket or the full-model strict G2 proof.
    #[test]
    fn qsa_incremental_pool_wide_rows_equal_singletons_and_rewind() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let max_rows = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        let dim = 128usize;
        let norm = upload_bf16_values(&mut gpu, &test_wave(301, dim, 0.25));
        let norm_bytes = download_bytes(&gpu, &norm);
        // config.rs pins FN index heads/KV heads/dim/compress/budget to
        // 4/1/128/4/2048 (not attention's 24/2/256). state.rs admits any
        // nonzero max_seq <= context_limit, and exercises max_seq=8193 in
        // its allocation tests. Its raw capacity is max_seq; pooled capacity
        // is ceil(max_seq/compress). mtp_gpu.rs passes that pooled capacity as
        // grid_bound and end/compress as block_count.
        //
        // Thus 8193 raw rows require 2049 pooled rows, NOT 2051: 2051 is
        // config.qsa_selected_capacity() = budget + compress - 1. Budget is
        // a selector concept, not a pooling bound. Exercise ordinary block
        // edges around its 2048-token seam and the actual allocation end;
        // selector-budget acceptance remains with the original G2/G3 gates.
        for (raw_rows, compress, production) in
            [(256usize, 4usize, false), (256, 8, false), (8193, 4, true)]
        {
            let capacity = raw_rows.div_ceil(compress);
            let starts: Vec<usize> = if production {
                (0..compress)
                    .map(|residue| 2048 - max_rows + residue)
                    .chain((0..compress).map(|residue| raw_rows - max_rows - 1 - residue))
                    .collect()
            } else {
                (0..compress).map(|residue| 64 + residue).collect()
            };
            for dtype in [DType::F32, DType::BF16] {
                let values = test_wave(303, raw_rows * dim, 1.0);
                let raw = if dtype == DType::F32 {
                    gpu.upload_f32(&values, &[values.len()]).expect("raw F32")
                } else {
                    upload_bf16_values(&mut gpu, &values)
                };
                let raw_bytes = download_bytes(&gpu, &raw);
                let pooled = |gpu: &mut Gpu| {
                    // One extra row is an allocation canary, never a capacity.
                    let t = gpu.zeros(&[(capacity + 1) * dim], dtype).expect("pooled");
                    gpu.hip.memset(&t.buf, 0xa5, t.byte_size()).expect("poison");
                    t
                };
                let launch = |gpu: &mut Gpu, out: &GpuTensor, start: usize, rows: usize| {
                    assert!(start + rows <= raw_rows);
                    indexed_attention_pool_rope_incremental(
                        gpu,
                        &IndexedAttentionPoolRope {
                            raw_keys: &raw,
                            pooled: out,
                            norm: Some(&norm),
                            block_count: (start + rows) / compress,
                            compress,
                            index_dim: dim,
                            position: Some(QsaPositionBinding {
                                position_start: start,
                                rows,
                            }),
                            grid_bound: capacity,
                            yarn: crate::rope::RopeYarn::OFF,
                        },
                    )
                    .expect("incremental pool");
                    #[cfg(feature = "lab")]
                    g1_launched_identity(gpu, format_args!(
                        "case=qsa_incremental_pool mode=eager format={dtype:?} R={rows} position={start} compress={compress} dim={dim} raw_bytes={} pooled_bytes={}",
                        raw.byte_size(), out.byte_size()
                    ));
                };
                // Reuse device buffers and the singleton prefix receipts for
                // every R/keep; no sampled keeps or row-count tuning cap.
                let seed = pooled(&mut gpu);
                let single = pooled(&mut gpu);
                let batch = pooled(&mut gpu);
                let speculative = pooled(&mut gpu);
                for start in starts.iter().copied() {
                    gpu.hip
                        .memset(&seed.buf, 0xa5, seed.byte_size())
                        .expect("seed poison");
                    launch(&mut gpu, &seed, 0, start);
                    let seed_bytes = download_bytes(&gpu, &seed);
                    gpu.copy_d2d(&seed, &single, seed.byte_size())
                        .expect("seed oracle");
                    let mut prefixes = vec![seed_bytes.clone()];
                    for row in 0..=max_rows {
                        launch(&mut gpu, &single, start + row, 1);
                        prefixes.push(download_bytes(&gpu, &single));
                    }
                    let first_byte = start / compress * dim * dtype.size();
                    for rows in 1..=max_rows {
                        gpu.copy_d2d(&seed, &speculative, seed.byte_size())
                            .expect("seed speculative");
                        launch(&mut gpu, &speculative, start, rows);
                        let speculative_bytes = download_bytes(&gpu, &speculative);
                        assert_eq!(
                            speculative_bytes, prefixes[rows],
                            "pool {dtype:?} c={compress} p={start} rows={rows}"
                        );
                        let speculative_live = (start + rows) / compress * dim * dtype.size();
                        assert_eq!(speculative_bytes[..first_byte], seed_bytes[..first_byte]);
                        assert!(
                            speculative_bytes[speculative_live..]
                                .iter()
                                .all(|b| *b == 0xa5),
                            "pool published future blocks or overwrote its canary"
                        );
                        for keep in 0..=rows {
                            // Real logical abort (keep=0) is a zero-row API
                            // launch against the speculative bytes. Rewind does
                            // not erase rejected blocks; it must not write them.
                            gpu.copy_d2d(&speculative, &batch, batch.byte_size())
                                .expect("restore speculative");
                            launch(&mut gpu, &batch, start, keep);
                            let kept = download_bytes(&gpu, &batch);
                            let live = (start + keep) / compress * dim * dtype.size();
                            assert_eq!(
                                kept[..live], prefixes[keep][..live],
                                "pool rewind {dtype:?} c={compress} p={start} rows={rows} keep={keep}"
                            );
                            assert_eq!(kept[live..], speculative_bytes[live..]);
                            if keep == 0 {
                                assert_eq!(kept, speculative_bytes, "abort wrote pooled bytes");
                            }
                            // Actually pool the next row after every kept end,
                            // including abort and full acceptance. Compare the
                            // complete live prefix, not just the newest block.
                            launch(&mut gpu, &batch, start + keep, 1);
                            let continued = download_bytes(&gpu, &batch);
                            let continued_live = (start + keep + 1) / compress * dim * dtype.size();
                            assert_eq!(
                                continued[..continued_live], prefixes[keep + 1][..continued_live],
                                "pool continuation {dtype:?} c={compress} p={start} rows={rows} keep={keep}"
                            );
                            assert_eq!(continued[..first_byte], seed_bytes[..first_byte]);
                            assert_eq!(
                                continued[continued_live..],
                                speculative_bytes[continued_live..],
                                "continuation changed future bytes or its canary"
                            );
                        }
                    }
                    assert_eq!(download_bytes(&gpu, &seed), seed_bytes, "seed ring changed");
                    assert_eq!(download_bytes(&gpu, &raw), raw_bytes, "raw ring changed");
                    assert_eq!(download_bytes(&gpu, &norm), norm_bytes, "norm changed");
                }
                for tensor in [seed, single, batch, speculative, raw] {
                    gpu.free_tensor(tensor).expect("free");
                }
            }
        }
        gpu.free_tensor(norm).expect("free");
    }

    /// BF16 raw/pooled index arenas hold the same BF16 values as F32 arenas,
    /// so pooling and both selection kernels must produce identical pooled
    /// keys and selections.
    #[test]
    fn qsa_bf16_index_arenas_pool_and_select_like_f32() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        let (compress, index_heads, index_dim) = (4usize, 4usize, 128usize);
        let (rows, position_start, budget_blocks) = (3usize, 1021usize, 64usize);
        let block_count = (position_start + rows) / compress;
        let capacity = budget_blocks * compress + compress - 1;
        // Raw keys are BF16-rounded before they reach the arena.
        let raw: Vec<f32> = qsa_lcg(17, block_count * compress * index_dim)
            .into_iter()
            .map(|v| f32::from_bits((bf16_bits(v) as u32) << 16))
            .collect();
        let raw_bits: Vec<u8> = raw
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        let raw_f32 = gpu.upload_f32(&raw, &[raw.len()]).expect("raw f32");
        let raw_bf16 = gpu.zeros(&[raw.len()], DType::BF16).expect("raw bf16");
        gpu.hip
            .memcpy_htod(&raw_bf16.buf, &raw_bits)
            .expect("raw bf16 upload");
        let norm_bits: Vec<u8> = qsa_lcg(19, index_dim)
            .iter()
            .flat_map(|v| bf16_bits(v * 0.25).to_le_bytes())
            .collect();
        let norm = gpu.zeros(&[index_dim], DType::BF16).expect("norm");
        gpu.hip
            .memcpy_htod(&norm.buf, &norm_bits)
            .expect("norm upload");
        let pooled_f32 = gpu
            .zeros(&[block_count * index_dim], DType::F32)
            .expect("pooled f32");
        let pooled_bf16 = gpu
            .zeros(&[block_count * index_dim], DType::BF16)
            .expect("pooled bf16");
        for (raw_keys, pooled) in [(&raw_f32, &pooled_f32), (&raw_bf16, &pooled_bf16)] {
            indexed_attention_pool_rope(
                &mut gpu,
                &IndexedAttentionPoolRope {
                    raw_keys,
                    pooled,
                    norm: Some(&norm),
                    block_count,
                    compress,
                    index_dim,
                    position: None,
                    grid_bound: block_count,
                    yarn: crate::rope::RopeYarn::OFF,
                },
            )
            .expect("QSA pool");
        }
        let pooled_reference = gpu.download_f32(&pooled_f32).expect("pooled f32");
        let pooled_widened: Vec<f32> = download_bytes(&gpu, &pooled_bf16)
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect();
        assert!(
            pooled_reference
                .iter()
                .zip(&pooled_widened)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "BF16 pooled keys differ from the F32 arena"
        );
        let query = qsa_lcg(23, rows * index_heads * index_dim);
        let query_gpu = gpu.upload_f32(&query, &[query.len()]).expect("query");
        // The parallel kernel, then the serial one (a shape bound past its LDS).
        for shape_blocks in [block_count, QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES / 4 + 1] {
            let mut selections = Vec::new();
            for pooled in [&pooled_f32, &pooled_bf16] {
                let selected = gpu
                    .zeros(&[rows * capacity * 4], DType::Raw)
                    .expect("selected");
                indexed_attention_select_batch(
                    &mut gpu,
                    &IndexedAttentionSelectBatch {
                        pooled_bf16: None,
                        pooled_bf16_valid_blocks: 0,
                        query: &query_gpu,
                        pooled,
                        selected: &selected,
                        rows,
                        query_row_stride: index_heads * index_dim,
                        block_count,
                        index_heads,
                        index_dim,
                        budget_blocks,
                        compress,
                        position_start,
                        capacity,
                        shape_blocks,
                    },
                )
                .expect("QSA select");
                selections.push(download_bytes(&gpu, &selected));
                gpu.free_tensor(selected).expect("free selected");
            }
            assert_eq!(
                selections[0], selections[1],
                "BF16 selection differs (shape bound {shape_blocks})"
            );
        }
        for tensor in [raw_f32, raw_bf16, norm, pooled_f32, pooled_bf16, query_gpu] {
            gpu.free_tensor(tensor).expect("free");
        }
    }

    /// H4 shared-down fold: the BF16 shared-down GEMM that applies the scaled
    /// add and the paired HC write in its epilogue must leave the HC streams
    /// (and, when asked, the routed rows) bytewise as the F16 WMMA GEMM,
    /// `bf16_scaled_add_batched` and `hyper_write` do, F32 and BF16-bit
    /// streams, ragged row counts.
    #[test]
    fn shared_down_hcsd_is_bytewise_the_gemm_scaled_add_hyper_write_chain() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let (m, k) = (2560usize, 640usize);
        let wide = 4 * m;
        let mut weight = gpu
            .upload_raw(&bf16_le_bytes(&test_wave(1, m * k, 0.3)), &[m * k * 2])
            .expect("weight");
        weight.dtype = DType::BF16;
        weight.shape = vec![m * k];
        let bits = |gpu: &Gpu, t: &GpuTensor| -> Vec<u32> {
            gpu.download_f32(t)
                .expect("download")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for rows in [512usize, 530, 1100] {
            assert!(gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd_applies(&weight, m, k, rows));
            let x = gpu
                .upload_f32(&test_wave(2, rows * k, 2.0), &[rows * k])
                .expect("x");
            let selector = gpu
                .upload_f32(&test_wave(3, rows, 1.0), &[rows])
                .expect("selector");
            let gates = gpu
                .upload_f32(&test_wave(4, rows * 4, 1.5), &[rows * 4])
                .expect("gates");
            let routed_values = test_wave(5, rows * m, 3.0);
            let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
            for state_bf16 in [false, true] {
                let initial = test_wave(6, rows * wide, 3.0);
                let raw: Vec<u8> = if state_bf16 {
                    let mut raw = bf16_le_bytes(&initial);
                    raw.resize(rows * wide * 4, 0);
                    raw
                } else {
                    initial.iter().flat_map(|v| v.to_le_bytes()).collect()
                };
                let mut streams = |gpu: &mut Gpu| {
                    let mut t = gpu.upload_raw(&raw, &[rows * wide * 4]).expect("streams");
                    t.dtype = DType::F32;
                    t.shape = vec![rows * wide];
                    t
                };
                let reference_streams = streams(&mut gpu);
                let reference_routed = gpu.upload_f32(&routed_values, &[rows * m]).expect("routed");
                let projected = gpu.zeros(&[rows * m], DType::F32).expect("projected");
                assert!(gpu
                    .gemm_bf16_xf32_f16_wmma_qwen4(&[(&weight, &projected, m)], &x, k, rows)
                    .expect("gemm"));
                bf16_scaled_add_batched(
                    &mut gpu,
                    &Bf16ScaledAddBatched {
                        residual: &reference_routed,
                        value: &projected,
                        scalar: &selector,
                        rows,
                        elements: m,
                    },
                )
                .expect("scaled add");
                hyper_write(
                    &mut gpu,
                    &HyperWrite {
                        input: &reference_streams,
                        normalized: &normalized,
                        mixed: &reference_routed,
                        gates: &gates,
                        output: &reference_streams,
                        branches: 4,
                        hidden: m,
                        state_bf16,
                    },
                )
                .expect("write");
                for write_routed in [false, true] {
                    let fused_streams = streams(&mut gpu);
                    let fused_routed = gpu.upload_f32(&routed_values, &[rows * m]).expect("routed");
                    gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
                        &weight,
                        &x,
                        m,
                        k,
                        rows,
                        &fused_routed,
                        &selector,
                        &fused_streams,
                        &gates,
                        state_bf16,
                        write_routed,
                    )
                    .expect("hcsd");
                    let (want, got) = (bits(&gpu, &reference_streams), bits(&gpu, &fused_streams));
                    assert_ne!(want.len(), 0);
                    let differing = want.iter().zip(&got).filter(|(a, b)| a != b).count();
                    assert_eq!(
                        differing, 0,
                        "{differing} stream words differ: rows={rows} state_bf16={state_bf16} \
                         write_routed={write_routed}"
                    );
                    if write_routed {
                        assert_eq!(
                            bits(&gpu, &reference_routed),
                            bits(&gpu, &fused_routed),
                            "routed rows differ: rows={rows}"
                        );
                    } else {
                        assert_eq!(
                            routed_values
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            bits(&gpu, &fused_routed),
                            "routed rows were rewritten without hc_write_routed"
                        );
                    }
                }
            }
        }
    }

    /// HC row fold (`HIPFIRE_QWEN4_HC_ROW_FOLD`): the BF16-store shared-down GEMM
    /// plus `hc_row_fold_norm_gate` must leave the HC streams, the next read's F16
    /// normalized row and its paired write's gate logits bytewise as the
    /// zero-initialized combine, the `hcsd` fold + HC write GEMM and
    /// `hyper_norm_gate_outputs` do, with dead (-1) routes, tied experts and
    /// ragged token counts.
    #[test]
    fn hc_row_fold_is_bytewise_combine_hcsd_norm_gate_chain() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let (m, k) = (2560usize, 640usize);
        let wide = 4 * m;
        let bf16_tensor = |gpu: &mut Gpu, values: &[f32]| -> GpuTensor {
            let mut t = gpu
                .upload_raw(&bf16_le_bytes(values), &[values.len() * 2])
                .expect("bf16 upload");
            t.dtype = DType::BF16;
            t.shape = vec![values.len()];
            t
        };
        let weight = bf16_tensor(&mut gpu, &test_wave(1, m * k, 0.3));
        let norm_weight = bf16_tensor(&mut gpu, &test_wave(7, wide, 0.3));
        let gate_weight = bf16_tensor(&mut gpu, &test_wave(8, 4 * wide, 0.05));
        let ints = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for rows in [512usize, 530, 1100] {
            let grouped_rows = rows * 10 + 16;
            let grouped = gpu
                .upload_raw(
                    &bf16_le_bytes(&test_wave(9, grouped_rows * m, 2.0)),
                    &[grouped_rows * m * 2],
                )
                .expect("grouped");
            let inverse: Vec<i32> = (0..rows * 10)
                .map(|i| {
                    if i % 13 == 5 {
                        -1
                    } else {
                        ((i * 7919) % grouped_rows) as i32
                    }
                })
                .collect();
            let experts: Vec<i32> = (0..rows * 10).map(|i| ((i * 37) % 64) as i32).collect();
            let inverse = gpu
                .upload_raw(&ints(&inverse), &[rows * 40])
                .expect("inverse");
            let experts = gpu
                .upload_raw(&ints(&experts), &[rows * 40])
                .expect("experts");
            let route_weights: Vec<f32> = test_wave(10, rows * 10, 1.0)
                .iter()
                .map(|v| v.abs() / 3.0)
                .collect();
            let route_weights = gpu
                .upload_f32(&route_weights, &[rows * 10])
                .expect("weights");
            let order = gpu.zeros(&[rows * 20], DType::F32).expect("order");
            let x = gpu
                .upload_f32(&test_wave(2, rows * k, 2.0), &[rows * k])
                .expect("x");
            let selector = gpu
                .upload_f32(&test_wave(3, rows, 1.0), &[rows])
                .expect("selector");
            let gates = gpu
                .upload_f32(&test_wave(4, rows * 4, 1.5), &[rows * 4])
                .expect("gates");
            let mut raw = bf16_le_bytes(&test_wave(6, rows * wide, 3.0));
            raw.resize(rows * wide * 4, 0);
            let streams = |gpu: &mut Gpu| {
                let mut t = gpu.upload_raw(&raw, &[rows * wide * 4]).expect("streams");
                t.dtype = DType::F32;
                t.shape = vec![rows * wide];
                t
            };
            // Reference: zero-initialized combine, hcsd, hyper_norm_gate_outputs.
            let routed = gpu.zeros(&[rows * m], DType::F32).expect("routed");
            gpu.moe_down_combine_grouped_top10_bf16in(
                &grouped,
                &inverse,
                &experts,
                &route_weights,
                &routed,
                &order,
                m,
                grouped_rows,
                rows,
                true,
                64,
            )
            .expect("combine");
            let ref_streams = streams(&mut gpu);
            gpu.gemm_bf16_xf32_f16_wmma_qwen4_hcsd(
                &weight,
                &x,
                m,
                k,
                rows,
                &routed,
                &selector,
                &ref_streams,
                &gates,
                true,
                false,
            )
            .expect("hcsd");
            let ref_gates = gpu.zeros(&[rows * 4], DType::F32).expect("ref gates");
            let ref_row = gpu.zeros(&[rows * wide], DType::F16).expect("ref row");
            hyper_norm_gate_outputs(
                &mut gpu,
                &HyperNormGateOutputs {
                    input: &ref_streams,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    gates: &ref_gates,
                    normalized_f16: &ref_row,
                    rows,
                    branches: 4,
                    hidden: m,
                    state_bf16: true,
                },
            )
            .expect("norm gate");
            // Row fold.
            let new_streams = streams(&mut gpu);
            let shared = gpu.zeros(&[rows * m], DType::F32).expect("shared");
            let new_gates = gpu.zeros(&[rows * 4], DType::F32).expect("new gates");
            let new_row = gpu.zeros(&[rows * wide], DType::F16).expect("new row");
            gpu.moe_combine_order_top10(
                &inverse,
                &experts,
                &route_weights,
                &order,
                grouped_rows,
                rows,
                64,
            )
            .expect("order");
            gpu.gemm_bf16_xf32_f16_wmma_qwen4_bf16st(&weight, &x, m, k, rows, &shared)
                .expect("bf16st");
            gpu.hc_row_fold_norm_gate(&crate::hc_row_fold::HcRowFold {
                grouped_down: &grouped,
                order: &order,
                shared_bf16: &shared,
                selector: &selector,
                streams: &new_streams,
                gates: &gates,
                norm_weight: &norm_weight,
                gate_weight: &gate_weight,
                next_gates: &new_gates,
                normalized_f16: &new_row,
                rows,
                hidden: m,
            })
            .expect("row fold");
            let words = |gpu: &Gpu, t: &GpuTensor| gpu.download_raw_bytes(t).expect("download");
            for (name, want, got) in [
                (
                    "streams",
                    words(&gpu, &ref_streams),
                    words(&gpu, &new_streams),
                ),
                (
                    "gate logits",
                    words(&gpu, &ref_gates),
                    words(&gpu, &new_gates),
                ),
                (
                    "normalized f16 row",
                    words(&gpu, &ref_row),
                    words(&gpu, &new_row),
                ),
            ] {
                assert_ne!(want.len(), 0);
                assert!(want == got, "{name} differ: rows={rows}");
            }
        }
    }

    // ── gfx1201 HC row folds: byte oracles ──────────────────────────────────────────────

    /// xorshift64* generator for the gfx1201 row-fold oracles (deterministic, no deps).
    struct FoldRng(u64);

    impl FoldRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / 8_388_608.0 - 1.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// BF16 bit patterns a row fold must treat exactly like the unfused kernels: NaNs of
    /// both signs and distinct payloads (quiet and signaling), +-inf, +-0, subnormals,
    /// the largest and smallest normals, and exact 1.0.
    const FOLD_BF16_SPECIALS: [u16; 22] = [
        0x7fc0, 0xffc0, 0x7f81, 0xff81, 0x7fff, 0xffff, 0x7fc5, 0xffd3, 0x7f80, 0xff80, 0x0000,
        0x8000, 0x0001, 0x8001, 0x007f, 0x807f, 0x7f7f, 0xff7f, 0x0080, 0x8080, 0x3f80, 0xbf80,
    ];

    /// The same as F32 bit patterns (payload bits below the BF16 mantissa included).
    const FOLD_F32_SPECIALS: [u32; 26] = [
        0x7fc0_0000,
        0xffc0_0000,
        0x7f80_0001,
        0xff80_0001,
        0x7fc1_2345,
        0xffc5_4321,
        0x7fff_ffff,
        0xffff_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7fbf_ffff,
        0xffa0_0001,
        0x7f7f_0000,
        0x0000_8000,
    ];

    /// How a row of the adversarial data behaves (`row * 7 + 3` mod 13): rows of only
    /// specials, sparse specials, a single NaN, huge and subnormal-range magnitudes
    /// among clean rows.
    fn fold_row_kind(row: usize) -> usize {
        (row * 7 + 3) % 13
    }

    fn fold_value(rng: &mut FoldRng, kind: usize, adversarial: bool, scale: f32) -> f32 {
        let v = rng.unit();
        if !adversarial {
            return v * scale;
        }
        match kind {
            3 => v * 1.0e30,
            4 => v * 1.0e-39,
            5 => v * 3.0e38,
            _ => v * scale,
        }
    }

    fn fold_special_f32(rng: &mut FoldRng) -> f32 {
        match rng.below(8) {
            0 => f32::from_bits(
                0x7fc0_0000 | (rng.next() as u32 & 0x003f_ffff) | ((rng.next() as u32 & 1) << 31),
            ),
            1 => f32::from_bits(
                0x7f80_0000
                    | (1 + (rng.next() as u32 % 0x003f_fffe))
                    | ((rng.next() as u32 & 1) << 31),
            ),
            _ => f32::from_bits(FOLD_F32_SPECIALS[rng.below(FOLD_F32_SPECIALS.len())]),
        }
    }

    /// F32 data `[rows, width]` (per-row kinds as above when `adversarial`).
    fn fold_f32_rows(
        rng: &mut FoldRng,
        rows: usize,
        width: usize,
        adversarial: bool,
        scale: f32,
    ) -> Vec<f32> {
        let mut data = Vec::with_capacity(rows * width);
        for row in 0..rows {
            let kind = fold_row_kind(row);
            for _ in 0..width {
                data.push(if adversarial && kind == 0 {
                    fold_special_f32(rng)
                } else if adversarial && kind == 1 && rng.below(64) == 0 {
                    fold_special_f32(rng)
                } else {
                    fold_value(rng, kind, adversarial, scale)
                });
            }
            if adversarial && kind == 2 {
                data[row * width + rng.below(width)] = fold_special_f32(rng);
            }
        }
        data
    }

    /// BF16 bits `[rows, width]`, the BF16 round-to-nearest-even of F32 data with the
    /// specials spliced in as raw bit patterns.
    fn fold_bf16_rows(
        rng: &mut FoldRng,
        rows: usize,
        width: usize,
        adversarial: bool,
        scale: f32,
    ) -> Vec<u16> {
        let mut data = Vec::with_capacity(rows * width);
        let special = |rng: &mut FoldRng| FOLD_BF16_SPECIALS[rng.below(FOLD_BF16_SPECIALS.len())];
        let rne = |v: f32| {
            let u = v.to_bits();
            ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
        };
        for row in 0..rows {
            let kind = fold_row_kind(row);
            for _ in 0..width {
                data.push(if adversarial && kind == 0 {
                    special(rng)
                } else if adversarial && kind == 1 && rng.below(64) == 0 {
                    special(rng)
                } else {
                    rne(fold_value(rng, kind, adversarial, scale))
                });
            }
            if adversarial && kind == 2 {
                data[row * width + rng.below(width)] = special(rng);
            }
        }
        data
    }

    /// Oracle data of `variant`: 0 clean; 1 / 2 row-adversarial ([`fold_row_kind`]);
    /// 3 clean with isolated specials (one lane in 32768, so about a quarter of the HC
    /// rows hold one and the rest stay finite, with finite gate logits).
    fn fold_f32_data(
        rng: &mut FoldRng,
        rows: usize,
        width: usize,
        variant: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut data = fold_f32_rows(rng, rows, width, variant == 1 || variant == 2, scale);
        if variant == 3 {
            for v in data.iter_mut() {
                if rng.below(32768) == 0 {
                    *v = fold_special_f32(rng);
                }
            }
        }
        data
    }

    fn fold_bf16_data(
        rng: &mut FoldRng,
        rows: usize,
        width: usize,
        variant: usize,
        scale: f32,
    ) -> Vec<u16> {
        let mut data = fold_bf16_rows(rng, rows, width, variant == 1 || variant == 2, scale);
        if variant == 3 {
            for v in data.iter_mut() {
                if rng.below(32768) == 0 {
                    *v = FOLD_BF16_SPECIALS[rng.below(FOLD_BF16_SPECIALS.len())];
                }
            }
        }
        data
    }

    fn fold_bf16_tensor(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
        let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut t = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16 upload");
        t.dtype = DType::BF16;
        t.shape = vec![bits.len()];
        t
    }

    /// BF16 HC streams as the prefill holds them: BF16 bits in an F32-typed tensor of
    /// `rows * wide` elements (twice the bytes the bits need).
    fn fold_streams_tensor(gpu: &mut Gpu, bits: &[u16]) -> GpuTensor {
        let mut bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();
        bytes.resize(bits.len() * 4, 0);
        let mut t = gpu
            .upload_raw(&bytes, &[bytes.len()])
            .expect("streams upload");
        t.dtype = DType::F32;
        t.shape = vec![bits.len()];
        t
    }

    /// Appends a line to `$ROWFOLD_RECEIPT` (when set) and stderr.
    fn fold_receipt(line: &str) {
        eprintln!("{line}");
        if let Ok(path) = std::env::var("ROWFOLD_RECEIPT") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open receipt");
            writeln!(file, "{line}").expect("write receipt");
        }
    }

    /// Byte-compares `want` / `got` and returns the number of differing bytes.
    fn fold_byte_diff(want: &[u8], got: &[u8]) -> usize {
        assert_eq!(want.len(), got.len());
        want.iter().zip(got).filter(|(a, b)| a != b).count()
    }

    fn fold_nan_words(bytes: &[u8], width: usize) -> usize {
        match width {
            2 => bytes
                .chunks_exact(2)
                .filter(|w| {
                    let u = u16::from_le_bytes([w[0], w[1]]);
                    (u & 0x7f80) == 0x7f80 && (u & 0x7f) != 0
                })
                .count(),
            _ => bytes
                .chunks_exact(4)
                .filter(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]).is_nan())
                .count(),
        }
    }

    /// The weights of the next read's norm and its paired write's gate projection:
    /// clean, or with NaN / inf / zero / subnormal entries spliced in.
    fn fold_norm_gate_weights(
        gpu: &mut Gpu,
        rng: &mut FoldRng,
        wide: usize,
        adversarial: bool,
    ) -> (GpuTensor, GpuTensor) {
        let mut norm = fold_bf16_rows(rng, 1, wide, false, 0.3);
        let mut gate = fold_bf16_rows(rng, 4, wide, false, 0.05);
        if adversarial {
            for _ in 0..6 {
                let i = rng.below(norm.len());
                norm[i] = FOLD_BF16_SPECIALS[rng.below(FOLD_BF16_SPECIALS.len())];
            }
            for _ in 0..10 {
                let i = rng.below(gate.len());
                gate[i] = FOLD_BF16_SPECIALS[rng.below(FOLD_BF16_SPECIALS.len())];
            }
        }
        (fold_bf16_tensor(gpu, &norm), fold_bf16_tensor(gpu, &gate))
    }

    /// The unfused tail after the mixed rows: `hyper_write` (BF16 streams, in place)
    /// then `hyper_norm_gate_outputs`.  Returns (gates, F16 row) tensors.
    #[allow(clippy::too_many_arguments)]
    fn fold_reference_write_norm(
        gpu: &mut Gpu,
        streams: &GpuTensor,
        mixed: &GpuTensor,
        gates: &GpuTensor,
        norm_weight: &GpuTensor,
        gate_weight: &GpuTensor,
        rows: usize,
        m: usize,
    ) -> (GpuTensor, GpuTensor) {
        let wide = 4 * m;
        let normalized = gpu.zeros(&[rows * wide], DType::F32).expect("normalized");
        hyper_write(
            gpu,
            &HyperWrite {
                input: streams,
                normalized: &normalized,
                mixed,
                gates,
                output: streams,
                branches: 4,
                hidden: m,
                state_bf16: true,
            },
        )
        .expect("reference hyper_write");
        let ref_gates = gpu.zeros(&[rows * 4], DType::F32).expect("ref gates");
        let ref_row = gpu.zeros(&[rows * wide], DType::F16).expect("ref row");
        hyper_norm_gate_outputs(
            gpu,
            &HyperNormGateOutputs {
                input: streams,
                norm_weight,
                gate_weight,
                gates: &ref_gates,
                normalized_f16: &ref_row,
                rows,
                branches: 4,
                hidden: m,
                state_bf16: true,
            },
        )
        .expect("reference norm gate");
        gpu.free_tensor(normalized).expect("free");
        (ref_gates, ref_row)
    }

    /// gfx1201 MoE-tail row fold: `hc_row_fold_moe_gfx1201` leaves the HC streams, the
    /// next read's F16 normalized row and its paired write's gate logits bytewise as the
    /// default chain does: zero-initialized combine -> `bf16_scaled_add_batched` (shared
    /// down F32 rows) -> `hyper_write` (the x8 gfx1201 kernel) -> `hyper_norm_gate_outputs`.
    /// Four data variants (0 clean; 1 row-adversarial: NaN payloads of both signs, inf,
    /// +-0, subnormals, huge rows; 2 the same with NaN / inf / zero entries in the next
    /// read's norm and gate weights; 3 clean with isolated specials), dead -1 routes and
    /// unrouted experts, at 16, 17, 4096 and 4097 rows.  Set `ROWFOLD_RECEIPT`
    /// to append one result line per case to a file.
    #[test]
    fn hc_row_fold_moe_gfx1201_is_bytewise_the_default_chain() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1201() {
            eprintln!("skip: needs gfx1201");
            return;
        }
        let m = 2560usize;
        let wide = 4 * m;
        let ints = |v: &[i32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for variant in 0..4usize {
            let adversarial_data = variant >= 1;
            let adversarial_weights = variant == 2;
            for rows in [16usize, 17, 4096, 4097] {
                let mut rng =
                    FoldRng(0x9e37_79b9_7f4a_7c15 ^ ((variant as u64) << 32) ^ rows as u64);
                let grouped_rows = (rows * 10 + 16).min(6000);
                let (norm_weight, gate_weight) =
                    fold_norm_gate_weights(&mut gpu, &mut rng, wide, adversarial_weights);
                let grouped_bits = fold_bf16_data(&mut rng, grouped_rows, m, variant, 2.0);
                let grouped = fold_bf16_tensor(&mut gpu, &grouped_bits);
                let inverse: Vec<i32> = (0..rows * 10)
                    .map(|i| {
                        if i % 13 == 5 {
                            -1
                        } else {
                            ((i * 7919 + variant) % grouped_rows) as i32
                        }
                    })
                    .collect();
                // Tied experts, and ids outside [0, 64) (unrouted: the order slot stays dead).
                let experts: Vec<i32> = (0..rows * 10)
                    .map(|i| match i % 29 {
                        7 => -1,
                        19 => 64 + (i % 5) as i32,
                        _ => ((i * 37) % 64) as i32,
                    })
                    .collect();
                let inverse = gpu
                    .upload_raw(&ints(&inverse), &[rows * 40])
                    .expect("inverse");
                let experts = gpu
                    .upload_raw(&ints(&experts), &[rows * 40])
                    .expect("experts");
                let mut weights = fold_f32_rows(&mut rng, rows, 10, false, 1.0);
                for w in weights.iter_mut() {
                    *w = w.abs() / 3.0;
                }
                if adversarial_data {
                    let dirty = if variant == 3 {
                        (rows / 64).max(1)
                    } else {
                        (rows / 4).max(1)
                    };
                    for _ in 0..dirty {
                        let i = rng.below(rows * 10);
                        weights[i] = fold_special_f32(&mut rng);
                    }
                }
                let route_weights = gpu.upload_f32(&weights, &[rows * 10]).expect("weights");
                let shared_values = fold_f32_data(&mut rng, rows, m, variant, 2.0);
                let shared = gpu.upload_f32(&shared_values, &[rows * m]).expect("shared");
                let selector_values = fold_f32_data(&mut rng, rows, 1, variant, 1.0);
                let selector = gpu.upload_f32(&selector_values, &[rows]).expect("selector");
                let gate_values = fold_f32_data(&mut rng, rows, 4, variant, 3.0);
                let gates = gpu.upload_f32(&gate_values, &[rows * 4]).expect("gates");
                let stream_bits = fold_bf16_data(&mut rng, rows, wide, variant, 3.0);
                let order = gpu.zeros(&[rows * 20], DType::F32).expect("order");

                // Reference: the default gfx1201 chain.
                let routed = gpu.zeros(&[rows * m], DType::F32).expect("routed");
                gpu.moe_down_combine_grouped_top10_bf16in(
                    &grouped,
                    &inverse,
                    &experts,
                    &route_weights,
                    &routed,
                    &order,
                    m,
                    grouped_rows,
                    rows,
                    true,
                    64,
                )
                .expect("combine");
                bf16_scaled_add_batched(
                    &mut gpu,
                    &Bf16ScaledAddBatched {
                        residual: &routed,
                        value: &shared,
                        scalar: &selector,
                        rows,
                        elements: m,
                    },
                )
                .expect("scaled add");
                let ref_streams = fold_streams_tensor(&mut gpu, &stream_bits);
                let (ref_gates, ref_row) = fold_reference_write_norm(
                    &mut gpu,
                    &ref_streams,
                    &routed,
                    &gates,
                    &norm_weight,
                    &gate_weight,
                    rows,
                    m,
                );

                // Fold: rank order + one row kernel (the combine output never exists).
                let new_streams = fold_streams_tensor(&mut gpu, &stream_bits);
                let new_gates = gpu.zeros(&[rows * 4], DType::F32).expect("new gates");
                let new_row = gpu.zeros(&[rows * wide], DType::F16).expect("new row");
                gpu.moe_combine_order_top10(
                    &inverse,
                    &experts,
                    &route_weights,
                    &order,
                    grouped_rows,
                    rows,
                    64,
                )
                .expect("order");
                gpu.hc_row_fold_moe_gfx1201(&crate::hc_row_fold::HcRowFoldMoeGfx1201 {
                    grouped_down: &grouped,
                    order: &order,
                    shared_f32: &shared,
                    selector: &selector,
                    streams: &new_streams,
                    gates: &gates,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    next_gates: &new_gates,
                    normalized_f16: &new_row,
                    rows,
                    hidden: m,
                })
                .expect("moe row fold");

                let words = |gpu: &Gpu, t: &GpuTensor| gpu.download_raw_bytes(t).expect("download");
                let mut report = String::new();
                let mut bad = 0usize;
                for (name, want, got, width) in [
                    (
                        "streams",
                        words(&gpu, &ref_streams),
                        words(&gpu, &new_streams),
                        2usize,
                    ),
                    (
                        "gate_logits",
                        words(&gpu, &ref_gates),
                        words(&gpu, &new_gates),
                        4,
                    ),
                    ("f16_row", words(&gpu, &ref_row), words(&gpu, &new_row), 2),
                ] {
                    assert_ne!(want.len(), 0);
                    let diff = fold_byte_diff(&want, &got);
                    bad += diff;
                    report.push_str(&format!(
                        " {name}:bytes={} diff={diff} nan_words={}",
                        want.len(),
                        if name == "f16_row" {
                            0
                        } else {
                            fold_nan_words(&want, width)
                        }
                    ));
                }
                if bad != 0 {
                    // Where the fold and the chain part: the stream word, its row / column,
                    // the HC write's inputs there and the BF16 scaled add's.
                    let want = words(&gpu, &ref_streams);
                    let got = words(&gpu, &new_streams);
                    let routed_bytes = words(&gpu, &routed);
                    let mut shown = 0;
                    for i in 0..rows * wide {
                        let (w, g) = (
                            u16::from_le_bytes([want[2 * i], want[2 * i + 1]]),
                            u16::from_le_bytes([got[2 * i], got[2 * i + 1]]),
                        );
                        if w != g && shown < 8 {
                            shown += 1;
                            let (row, col) = (i / wide, (i % wide) % m);
                            let mixed_bits = u32::from_le_bytes(
                                routed_bytes[4 * (row * m + col)..4 * (row * m + col) + 4]
                                    .try_into()
                                    .unwrap(),
                            );
                            fold_receipt(&format!(
                                "  diff row={row} branch={} col={col}: want={w:04x} got={g:04x} stream_in={:04x} \
                                 ref_mixed={mixed_bits:08x} shared={:08x} selector={:08x} gate_logit={:08x}",
                                (i % wide) / m,
                                stream_bits[i],
                                shared_values[row * m + col].to_bits(),
                                selector_values[row].to_bits(),
                                gate_values[row * 4 + (i % wide) / m].to_bits(),
                            ));
                        }
                    }
                }
                fold_receipt(&format!(
                    "hc_row_fold_moe_gfx1201 variant={variant} rows={rows}{report} {}",
                    if bad == 0 { "BYTE-EQUAL" } else { "MISMATCH" }
                ));
                assert_eq!(bad, 0, "moe fold differs: variant={variant} rows={rows}");
                for t in [
                    norm_weight,
                    gate_weight,
                    grouped,
                    inverse,
                    experts,
                    route_weights,
                    shared,
                    selector,
                    gates,
                    order,
                    routed,
                    ref_streams,
                    ref_gates,
                    ref_row,
                    new_streams,
                    new_gates,
                    new_row,
                ] {
                    gpu.free_tensor(t).expect("free");
                }
            }
        }
    }

    /// gfx1201 mixer-tail row fold: `hc_row_write_norm_gate_gfx1201` leaves the HC
    /// streams, the next read's F16 row and its paired write's gates bytewise as
    /// `hyper_write` (the x8 gfx1201 kernel, F32 `mixed` of the out projection) followed
    /// by `hyper_norm_gate_outputs` do; rows below 16 and other widths fail closed.
    #[test]
    fn hc_row_write_norm_gate_gfx1201_is_bytewise_the_default_chain() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1201() {
            eprintln!("skip: needs gfx1201");
            return;
        }
        let m = 2560usize;
        let wide = 4 * m;
        for variant in 0..4usize {
            let adversarial_weights = variant == 2;
            for rows in [1usize, 15, 16, 17, 4096, 4097] {
                let mut rng =
                    FoldRng(0x1234_5678_9abc_def1 ^ ((variant as u64) << 32) ^ rows as u64);
                let (norm_weight, gate_weight) =
                    fold_norm_gate_weights(&mut gpu, &mut rng, wide, adversarial_weights);
                let mixed_values = fold_f32_data(&mut rng, rows, m, variant, 2.0);
                let mixed = gpu.upload_f32(&mixed_values, &[rows * m]).expect("mixed");
                let gate_values = fold_f32_data(&mut rng, rows, 4, variant, 3.0);
                let gates = gpu.upload_f32(&gate_values, &[rows * 4]).expect("gates");
                let stream_bits = fold_bf16_data(&mut rng, rows, wide, variant, 3.0);
                let new_streams = fold_streams_tensor(&mut gpu, &stream_bits);
                let new_gates = gpu.zeros(&[rows * 4], DType::F32).expect("new gates");
                let new_row = gpu.zeros(&[rows * wide], DType::F16).expect("new row");
                let fold = crate::hc_row_fold::HcRowWriteNormGateGfx1201 {
                    mixed: &mixed,
                    streams: &new_streams,
                    gates: &gates,
                    norm_weight: &norm_weight,
                    gate_weight: &gate_weight,
                    next_gates: &new_gates,
                    normalized_f16: &new_row,
                    rows,
                    hidden: m,
                };
                if rows < 16 {
                    assert!(
                        gpu.hc_row_write_norm_gate_gfx1201(&fold).is_err(),
                        "rows={rows} must fail closed (hyper_write takes the x2 kernel there)"
                    );
                    fold_receipt(&format!(
                        "hc_row_write_norm_gate_gfx1201 variant={variant} rows={rows} FAIL-CLOSED (rows < 16)"
                    ));
                    for t in [
                        norm_weight,
                        gate_weight,
                        mixed,
                        gates,
                        new_streams,
                        new_gates,
                        new_row,
                    ] {
                        gpu.free_tensor(t).expect("free");
                    }
                    continue;
                }
                let ref_streams = fold_streams_tensor(&mut gpu, &stream_bits);
                let (ref_gates, ref_row) = fold_reference_write_norm(
                    &mut gpu,
                    &ref_streams,
                    &mixed,
                    &gates,
                    &norm_weight,
                    &gate_weight,
                    rows,
                    m,
                );
                gpu.hc_row_write_norm_gate_gfx1201(&fold)
                    .expect("mixer row fold");

                let words = |gpu: &Gpu, t: &GpuTensor| gpu.download_raw_bytes(t).expect("download");
                let mut report = String::new();
                let mut bad = 0usize;
                for (name, want, got, width) in [
                    (
                        "streams",
                        words(&gpu, &ref_streams),
                        words(&gpu, &new_streams),
                        2usize,
                    ),
                    (
                        "gate_logits",
                        words(&gpu, &ref_gates),
                        words(&gpu, &new_gates),
                        4,
                    ),
                    ("f16_row", words(&gpu, &ref_row), words(&gpu, &new_row), 2),
                ] {
                    assert_ne!(want.len(), 0);
                    let diff = fold_byte_diff(&want, &got);
                    bad += diff;
                    report.push_str(&format!(
                        " {name}:bytes={} diff={diff} nan_words={}",
                        want.len(),
                        if name == "f16_row" {
                            0
                        } else {
                            fold_nan_words(&want, width)
                        }
                    ));
                }
                fold_receipt(&format!(
                    "hc_row_write_norm_gate_gfx1201 variant={variant} rows={rows}{report} {}",
                    if bad == 0 { "BYTE-EQUAL" } else { "MISMATCH" }
                ));
                assert_eq!(bad, 0, "mixer fold differs: variant={variant} rows={rows}");
                for t in [
                    norm_weight,
                    gate_weight,
                    mixed,
                    gates,
                    new_streams,
                    new_gates,
                    new_row,
                    ref_streams,
                    ref_gates,
                    ref_row,
                ] {
                    gpu.free_tensor(t).expect("free");
                }
            }
        }
        // Another width never takes the kernel.
        let wide_bad = 4 * 2048;
        let mixed = gpu.zeros(&[32 * 2048], DType::F32).expect("mixed");
        let gates = gpu.zeros(&[32 * 4], DType::F32).expect("gates");
        let streams = gpu.zeros(&[32 * wide_bad], DType::F32).expect("streams");
        let gate_logits = gpu.zeros(&[32 * 4], DType::F32).expect("gate logits");
        let row = gpu.zeros(&[32 * wide_bad], DType::F16).expect("row");
        let mut norm = gpu.zeros(&[wide_bad], DType::F32).expect("norm");
        norm.dtype = DType::BF16;
        let mut gate = gpu.zeros(&[4 * wide_bad], DType::F32).expect("gate");
        gate.dtype = DType::BF16;
        assert!(gpu
            .hc_row_write_norm_gate_gfx1201(&crate::hc_row_fold::HcRowWriteNormGateGfx1201 {
                mixed: &mixed,
                streams: &streams,
                gates: &gates,
                norm_weight: &norm,
                gate_weight: &gate,
                next_gates: &gate_logits,
                normalized_f16: &row,
                rows: 32,
                hidden: 2048,
            })
            .is_err());
    }

    const BF16_PM_ROWS: usize = 600;
    const BF16_PM_BLOCKS: usize = 2000;
    const BF16_PM_COMPRESS: usize = 4;
    const BF16_PM_HEADS: usize = 4;
    const BF16_PM_DIM: usize = 128;
    const BF16_PM_BUDGET: usize = 256;
    const BF16_PM_CAPACITY: usize = BF16_PM_BUDGET * BF16_PM_COMPRESS + BF16_PM_COMPRESS - 1;
    /// Padded past `heads * dim` so the query row stride is not the extent.
    const BF16_PM_QUERY_STRIDE: usize = BF16_PM_HEADS * BF16_PM_DIM + 16;
    const BF16_PM_POISON: u8 = 0xa5;
    /// Poisoned bytes after `selected` / `mirror`: a stray write shows.
    const BF16_PM_CANARY: usize = 4096;

    /// Prefill-sized BF16-pooled selection (600 rows, so the last sixteen-row
    /// score group is ragged: 600 = 37 * 16 + 8; 2000 blocks) on synthetic
    /// keys with many equal scores: a run of 40 and a run of 50 identical
    /// blocks (the budget boundary falls inside the second), a 400-block
    /// region of four constant key levels, a run of all-negative (ReLU zero)
    /// keys, and every seventh block a copy of the block three before it.
    struct Bf16PmSelect {
        query: GpuTensor,
        pooled: GpuTensor,
        selected: GpuTensor,
        mirror: GpuTensor,
    }

    struct Bf16PmOut {
        selected: Vec<u8>,
        mirror: Vec<u8>,
        persisted: bool,
    }

    impl Bf16PmOut {
        fn selected_len() -> usize {
            BF16_PM_ROWS * BF16_PM_CAPACITY * 4
        }

        fn mirror_len() -> usize {
            BF16_PM_CAPACITY * 4
        }

        fn canaries_intact(&self) -> bool {
            self.selected[Self::selected_len()..]
                .iter()
                .all(|b| *b == BF16_PM_POISON)
                && self.mirror[Self::mirror_len()..]
                    .iter()
                    .all(|b| *b == BF16_PM_POISON)
        }

        fn untouched(&self) -> bool {
            self.selected.iter().all(|b| *b == BF16_PM_POISON)
                && self.mirror.iter().all(|b| *b == BF16_PM_POISON)
        }

        fn last_row(&self) -> &[u8] {
            &self.selected[Self::selected_len() - Self::mirror_len()..Self::selected_len()]
        }

        fn selected_entries(&self) -> usize {
            self.selected[..Self::selected_len()]
                .chunks_exact(4)
                .filter(|w| i32::from_ne_bytes((*w).try_into().expect("i32 word")) >= 0)
                .count()
        }
    }

    impl Bf16PmSelect {
        fn new(gpu: &mut Gpu) -> Self {
            let mut state = 0x51ca_17b3u32;
            let mut next = move || {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / 16_777_216.0
            };
            // Positive queries: a block whose keys are all negative scores 0.
            let query: Vec<f32> = (0..BF16_PM_ROWS * BF16_PM_QUERY_STRIDE)
                .map(|_| next())
                .collect();
            let mut keys = vec![0f32; BF16_PM_BLOCKS * BF16_PM_DIM];
            for block in 0..BF16_PM_BLOCKS {
                for d in 0..BF16_PM_DIM {
                    let random = next();
                    keys[block * BF16_PM_DIM + d] = match block {
                        1000..=1039 => 0.9,
                        1200..=1249 => 0.6,
                        300..=699 => [0.05f32, 0.15, 0.25, 0.35]
                            [(block.wrapping_mul(2_654_435_761) >> 5) & 3],
                        1500..=1599 => -random - 0.01,
                        _ => random - 0.25,
                    };
                }
            }
            for block in (8..BF16_PM_BLOCKS).filter(|b| b % 7 == 3) {
                keys.copy_within(
                    (block - 3) * BF16_PM_DIM..(block - 2) * BF16_PM_DIM,
                    block * BF16_PM_DIM,
                );
            }
            let pooled_bytes: Vec<u8> = keys
                .iter()
                .flat_map(|v| {
                    // Round to nearest even, the BF16 the index-key append stores.
                    let bits = v.to_bits();
                    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16;
                    (rounded as u16).to_le_bytes()
                })
                .collect();
            let mut pooled = gpu
                .upload_raw(&pooled_bytes, &[pooled_bytes.len()])
                .expect("pooled upload");
            pooled.dtype = DType::BF16;
            pooled.shape = vec![keys.len()];
            let selected_len = Bf16PmOut::selected_len() + BF16_PM_CANARY;
            let mirror_len = Bf16PmOut::mirror_len() + BF16_PM_CANARY;
            Self {
                query: gpu
                    .upload_f32(&query, &[query.len()])
                    .expect("query upload"),
                pooled,
                selected: gpu
                    .upload_raw(&vec![BF16_PM_POISON; selected_len], &[selected_len])
                    .expect("selected upload"),
                mirror: gpu
                    .upload_raw(&vec![BF16_PM_POISON; mirror_len], &[mirror_len])
                    .expect("mirror upload"),
            }
        }

        fn params(&self) -> IndexedAttentionSelectBatch<'_> {
            IndexedAttentionSelectBatch {
                pooled_bf16: None,
                pooled_bf16_valid_blocks: 0,
                query: &self.query,
                pooled: &self.pooled,
                selected: &self.selected,
                rows: BF16_PM_ROWS,
                query_row_stride: BF16_PM_QUERY_STRIDE,
                block_count: BF16_PM_BLOCKS,
                index_heads: BF16_PM_HEADS,
                index_dim: BF16_PM_DIM,
                budget_blocks: BF16_PM_BUDGET,
                compress: BF16_PM_COMPRESS,
                // (position_start + rows) / compress == block_count.
                position_start: BF16_PM_BLOCKS * BF16_PM_COMPRESS - BF16_PM_ROWS,
                capacity: BF16_PM_CAPACITY,
                shape_blocks: BF16_PM_BLOCKS,
            }
        }

        fn poison(&self, gpu: &Gpu) {
            for t in [&self.selected, &self.mirror] {
                gpu.hip
                    .memset(&t.buf, BF16_PM_POISON as i32, t.byte_size())
                    .expect("poison outputs");
            }
        }

        fn download(&self, gpu: &Gpu, persisted: bool) -> Bf16PmOut {
            Bf16PmOut {
                selected: gpu
                    .download_raw_bytes(&self.selected)
                    .expect("selected download"),
                mirror: gpu
                    .download_raw_bytes(&self.mirror)
                    .expect("mirror download"),
                persisted,
            }
        }

        /// The lab helper with `arm`, outputs poisoned first.
        fn run_arm(&self, gpu: &mut Gpu, arm: QsaSelectPmArm, with_mirror: bool) -> Bf16PmOut {
            self.poison(gpu);
            let persisted = indexed_attention_select_batch_pm_arm(
                gpu,
                &self.params(),
                with_mirror.then_some(&self.mirror),
                arm,
            )
            .unwrap_or_else(|e| panic!("BF16 select arm {arm:?}: {e}"));
            self.download(gpu, persisted)
        }

        /// The default route (`indexed_attention_select_batch[_mirrored]`),
        /// outputs poisoned first.
        fn run_default(&self, gpu: &mut Gpu, with_mirror: bool) -> Bf16PmOut {
            self.poison(gpu);
            let persisted = if with_mirror {
                indexed_attention_select_batch_mirrored(gpu, &self.params(), &self.mirror)
                    .expect("default mirrored BF16 select")
            } else {
                indexed_attention_select_batch(gpu, &self.params()).expect("default BF16 select");
                false
            };
            self.download(gpu, persisted)
        }

        fn free(self, gpu: &mut Gpu) {
            for t in [self.query, self.pooled, self.selected, self.mirror] {
                gpu.free_tensor(t).expect("free");
            }
        }
    }

    /// The fused BF16 arm (`Hipcc`) is the reference; the builder score pair
    /// (`ScoreOnly`, `Both`) must reproduce its selection, mirror and
    /// persisted flag byte for byte, with and without a mirror. `SelectOnly`
    /// has no hipcc BF16 rows score kernel and must refuse before any launch.
    #[test]
    fn qsa_bf16_pm_select_arms_match_the_fused_bf16_arm_byte_for_byte() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let case = Bf16PmSelect::new(&mut gpu);

        // SelectOnly on BF16 keys: refused, and nothing was launched.
        case.poison(&gpu);
        let refused = indexed_attention_select_batch_pm_arm(
            &mut gpu,
            &case.params(),
            Some(&case.mirror),
            QsaSelectPmArm::SelectOnly,
        );
        assert!(refused.is_err(), "BF16 SelectOnly must be refused");
        assert!(
            case.download(&gpu, false).untouched(),
            "refused BF16 SelectOnly wrote outputs (launched before refusing)"
        );

        let reference = case.run_arm(&mut gpu, QsaSelectPmArm::Hipcc, true);
        assert!(
            reference.persisted,
            "fused BF16 arm did not persist the mirror"
        );
        assert!(
            reference.canaries_intact(),
            "fused BF16 arm wrote past its outputs"
        );
        // Every row emits its 256 blocks (1024 tokens); the rows' tails follow.
        assert!(
            reference.selected_entries() >= BF16_PM_ROWS * BF16_PM_BUDGET * BF16_PM_COMPRESS,
            "fused BF16 arm selected too few tokens"
        );
        assert_eq!(
            &reference.mirror[..Bf16PmOut::mirror_len()],
            reference.last_row(),
            "mirror is not the final row's selection"
        );
        for arm in [QsaSelectPmArm::ScoreOnly, QsaSelectPmArm::Both] {
            let out = case.run_arm(&mut gpu, arm, true);
            assert!(out.persisted, "{arm:?}: mirror not persisted");
            assert!(out.canaries_intact(), "{arm:?}: wrote past its outputs");
            assert!(
                out.selected == reference.selected,
                "{arm:?}: selection differs from the fused BF16 arm"
            );
            assert!(
                out.mirror == reference.mirror,
                "{arm:?}: mirror differs from the fused BF16 arm"
            );
        }
        // Without a mirror: the same selection, nothing persisted, mirror untouched.
        for arm in [
            QsaSelectPmArm::Hipcc,
            QsaSelectPmArm::ScoreOnly,
            QsaSelectPmArm::Both,
        ] {
            let out = case.run_arm(&mut gpu, arm, false);
            assert!(!out.persisted, "{arm:?}: persisted without a mirror");
            assert!(
                out.selected == reference.selected,
                "{arm:?}: selection differs without a mirror"
            );
            assert!(
                out.mirror.iter().all(|b| *b == BF16_PM_POISON),
                "{arm:?}: wrote a mirror it was not given"
            );
        }
        case.free(&mut gpu);
    }

    /// Exact-gfx1151 decode/verify coverage of the F32 builder pair. `Both`
    /// (PM rows16 score + PM select) must reproduce the hipcc pair's selected
    /// indices, tails, `-1` fill and mirror bytes for 1..=64 rows (and every row of
    /// a batch must equal the one-row call at its own position, bitwise) over block
    /// counts spanning the empty route, partial and full 256-block score
    /// tiles, the 511/512/513 boundary around the 512 budget blocks, and the
    /// 2047/8192 contexts. Geometry is the real Qwen4 decode call of
    /// `layer_ops.rs`: `budget_blocks = indexer_budget / compress` (2048 / 4),
    /// `capacity = indexer_budget + compress - 1`, query stride
    /// `heads * dim`, and `position_start + rows` completing `block_count`
    /// blocks (the tail 0..compress is swept). Zero blocks take the live
    /// empty route (fused) under both arms.
    #[test]
    #[ignore = "needs an exact gfx1151 GPU: cargo test -p rdna-compute --lib --features lab qsa_f32_decode_pm_both_matches_hipcc -- --ignored --nocapture"]
    fn qsa_f32_decode_pm_both_matches_hipcc_for_one_to_thirty_two_rows() {
        const HEADS: usize = 4;
        const DIM: usize = 128;
        const BUDGET: usize = 2048;
        const COMPRESS: usize = 4;
        const BUDGET_BLOCKS: usize = BUDGET / COMPRESS;
        const CAPACITY: usize = BUDGET + COMPRESS - 1;
        const STRIDE: usize = HEADS * DIM;
        const MAX_ROWS: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        const MAX_BLOCKS: usize = 8192;
        const POISON: u8 = 0xa5;
        const CANARY: usize = 4096;
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }

        let mut state = 0x7d3b_91a5u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0
        };
        // Positive random F32 queries: a block with all-negative keys scores 0.
        let query: Vec<f32> = (0..MAX_ROWS * STRIDE).map(|_| next()).collect();
        // Random F32 keys with deliberate score ties: every fifth block copies
        // its predecessor, a 32-block run per 512 is one constant block, a
        // 64-block run per 512 takes four quantised levels, and every 97th
        // block is all-negative (ReLU zero). The runs straddle the 256-block
        // score tiles and the budget boundary.
        let mut keys = vec![0f32; MAX_BLOCKS * DIM];
        for block in 0..MAX_BLOCKS {
            for d in 0..DIM {
                let random = next();
                keys[block * DIM + d] = match block % 512 {
                    64..=95 => 0.5,
                    200..=263 => {
                        [0.0f32, 0.125, 0.25, 0.5][(block.wrapping_mul(2_654_435_761) >> 5) & 3]
                    }
                    _ if block % 97 == 13 => -random - 0.01,
                    _ => random - 0.25,
                };
            }
            if block > 0 && block % 5 == 0 {
                keys.copy_within((block - 1) * DIM..block * DIM, block * DIM);
            }
        }
        let query_t = gpu
            .upload_f32(&query, &[query.len()])
            .expect("query upload");
        // Row r of the batch as its own tensor: the one-row call at position_start + r.
        let row_q: Vec<GpuTensor> = (0..MAX_ROWS)
            .map(|r| {
                gpu.upload_f32(&query[r * STRIDE..(r + 1) * STRIDE], &[STRIDE])
                    .expect("row query upload")
            })
            .collect();
        let pooled = gpu.upload_f32(&keys, &[keys.len()]).expect("pooled upload");
        let selected_len = MAX_ROWS * CAPACITY * 4 + CANARY;
        let mirror_len = CAPACITY * 4 + CANARY;
        let selected = gpu
            .upload_raw(&vec![POISON; selected_len], &[selected_len])
            .expect("selected upload");
        let mirror = gpu
            .upload_raw(&vec![POISON; mirror_len], &[mirror_len])
            .expect("mirror upload");

        fn run(
            gpu: &mut Gpu,
            p: &IndexedAttentionSelectBatch<'_>,
            mirror: &GpuTensor,
            arm: QsaSelectPmArm,
            with_mirror: bool,
        ) -> (bool, Vec<u8>, Vec<u8>) {
            // A verify forward of two or more rows runs row-exact to the widest
            // rows; a one-row call is the decode itself.
            gpu.set_wide_verify_rows(if p.rows >= 2 { MAX_ROWS } else { 0 });
            for t in [p.selected, mirror] {
                gpu.hip
                    .memset(&t.buf, POISON as i32, t.byte_size())
                    .expect("poison outputs");
            }
            let persisted =
                indexed_attention_select_batch_pm_arm(gpu, p, with_mirror.then_some(mirror), arm)
                    .unwrap_or_else(|e| {
                        panic!(
                            "{arm:?} rows={} blocks={} ps={}: {e}",
                            p.rows, p.block_count, p.position_start
                        )
                    });
            let out = (
                persisted,
                gpu.download_raw_bytes(p.selected)
                    .expect("selected download"),
                gpu.download_raw_bytes(mirror).expect("mirror download"),
            );
            gpu.set_wide_verify_rows(0);
            out
        }

        let mut cases = 0usize;
        for blocks in [0usize, 1, 3, 255, 256, 511, 512, 513, 2047, 8192] {
            for rows in 1..=MAX_ROWS {
                for tail in 0..COMPRESS {
                    // `(position_start + rows) / compress == blocks`; empty
                    // calls with more rows than a three-token tail clamp the
                    // start to zero, other infeasible starts do not exist.
                    let visible = blocks * COMPRESS + tail;
                    let position_start = match visible.checked_sub(rows) {
                        Some(start) => start,
                        None if blocks == 0 => 0,
                        None => continue,
                    };
                    let p = IndexedAttentionSelectBatch {
                        pooled_bf16: None,
                        pooled_bf16_valid_blocks: 0,
                        query: &query_t,
                        pooled: &pooled,
                        selected: &selected,
                        rows,
                        query_row_stride: STRIDE,
                        block_count: blocks,
                        index_heads: HEADS,
                        index_dim: DIM,
                        budget_blocks: BUDGET_BLOCKS,
                        compress: COMPRESS,
                        position_start,
                        capacity: CAPACITY,
                        shape_blocks: MAX_BLOCKS,
                    };
                    let ctx =
                        format!("rows={rows} blocks={blocks} tail={tail} ps={position_start}");
                    let (ref_persisted, ref_selected, ref_mirror) =
                        run(&mut gpu, &p, &mirror, QsaSelectPmArm::Hipcc, true);
                    assert!(ref_persisted, "hipcc did not persist the mirror: {ctx}");
                    let used = rows * CAPACITY * 4;
                    assert!(
                        ref_selected[used..].iter().all(|b| *b == POISON),
                        "hipcc wrote past its rows (sentinel): {ctx}"
                    );
                    assert!(
                        ref_mirror[CAPACITY * 4..].iter().all(|b| *b == POISON),
                        "hipcc wrote past the mirror (sentinel): {ctx}"
                    );
                    if blocks > 0 {
                        let live = ref_selected[..used]
                            .chunks_exact(4)
                            .filter(|w| i32::from_ne_bytes((*w).try_into().expect("i32 word")) >= 0)
                            .count();
                        assert!(live > 0, "hipcc selected nothing: {ctx}");
                        assert_eq!(
                            &ref_mirror[..CAPACITY * 4],
                            &ref_selected[(rows - 1) * CAPACITY * 4..used],
                            "mirror is not the final row's selection: {ctx}"
                        );
                    }

                    let (persisted, got_selected, got_mirror) =
                        run(&mut gpu, &p, &mirror, QsaSelectPmArm::Both, true);
                    assert!(persisted, "Both did not persist the mirror: {ctx}");
                    assert!(
                        got_selected == ref_selected,
                        "Both selection differs from hipcc: {ctx}"
                    );
                    assert!(
                        got_mirror == ref_mirror,
                        "Both mirror differs from hipcc: {ctx}"
                    );

                    // Wide verify: row r of the batch is bitwise the one-row call of
                    // the same arm at `position_start + r` (its own block count).
                    if blocks > 0 {
                        for (arm, batch_selected, batch_mirror) in [
                            (QsaSelectPmArm::Hipcc, &ref_selected, &ref_mirror),
                            (QsaSelectPmArm::Both, &got_selected, &got_mirror),
                        ] {
                            for r in 0..rows {
                                let p1 = IndexedAttentionSelectBatch {
                                    query: &row_q[r],
                                    rows: 1,
                                    block_count: (position_start + r + 1) / COMPRESS,
                                    position_start: position_start + r,
                                    ..p
                                };
                                let (one_persisted, one_selected, one_mirror) =
                                    run(&mut gpu, &p1, &mirror, arm, true);
                                assert!(
                                    one_persisted,
                                    "{arm:?} one-row call {r} did not persist: {ctx}"
                                );
                                assert!(
                                    one_selected[..CAPACITY * 4]
                                        == batch_selected[r * CAPACITY * 4..(r + 1) * CAPACITY * 4],
                                    "{arm:?} row {r} differs from its one-row call: {ctx}"
                                );
                                assert!(
                                    one_selected[CAPACITY * 4..].iter().all(|b| *b == POISON),
                                    "{arm:?} one-row call {r} wrote past its row: {ctx}"
                                );
                                if r + 1 == rows {
                                    assert!(
                                        one_mirror[..CAPACITY * 4] == batch_mirror[..CAPACITY * 4],
                                        "{arm:?} batch mirror differs from the last one-row call: {ctx}"
                                    );
                                }
                            }
                        }
                    }

                    // No mirror given: same selection, nothing persisted, the
                    // mirror buffer is untouched (every byte still the sentinel).
                    for arm in [QsaSelectPmArm::Hipcc, QsaSelectPmArm::Both] {
                        let (persisted, got_selected, got_mirror) =
                            run(&mut gpu, &p, &mirror, arm, false);
                        assert!(!persisted, "{arm:?} persisted without a mirror: {ctx}");
                        assert!(
                            got_selected == ref_selected,
                            "{arm:?} selection differs without a mirror: {ctx}"
                        );
                        assert!(
                            got_mirror.iter().all(|b| *b == POISON),
                            "{arm:?} wrote a mirror it was not given: {ctx}"
                        );
                    }
                    cases += 1;
                }
            }
        }
        eprintln!("qsa_f32_decode_pm_both: {cases} cases identical");
        for t in [query_t, pooled, selected, mirror].into_iter().chain(row_q) {
            gpu.free_tensor(t).expect("free");
        }
    }

    /// The default mirrored entry on BF16 keys (whichever arm the route's
    /// environment picks) must equal the fused BF16 arm byte for byte.
    #[test]
    fn qsa_bf16_default_mirrored_select_matches_the_fused_bf16_arm() {
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: needs gfx1151");
            return;
        }
        let case = Bf16PmSelect::new(&mut gpu);
        let reference = case.run_arm(&mut gpu, QsaSelectPmArm::Hipcc, true);
        assert!(
            reference.persisted,
            "fused BF16 arm did not persist the mirror"
        );
        let mirrored = case.run_default(&mut gpu, true);
        assert!(
            mirrored.persisted,
            "default BF16 route did not persist the mirror"
        );
        assert!(
            mirrored.canaries_intact(),
            "default BF16 route wrote past its outputs"
        );
        assert!(
            mirrored.selected == reference.selected,
            "default BF16 selection differs from the fused arm"
        );
        assert!(
            mirrored.mirror == reference.mirror,
            "default BF16 mirror differs from the fused arm"
        );
        let plain = case.run_default(&mut gpu, false);
        assert!(!plain.persisted);
        assert!(
            plain.selected == reference.selected,
            "default BF16 selection without a mirror differs"
        );
        case.free(&mut gpu);
    }

    /// Wide-verify oracle of the gfx1151 BF16 PM arms and the default mirrored
    /// route: for n = 1..=64 rows at end positions straddling the compress-4
    /// boundaries (E = 8000/4000/1030 minus 0..3), each arm's batched call
    /// equals the fused hipcc arm and, row by row, the one-row call of the same
    /// arm at `position_start + r` (selection row bytes, final mirror). The wide
    /// gate is set through `Gpu::set_wide_verify_rows` (32 for n >= 2, else 0).
    #[test]
    #[ignore = "needs an exact gfx1151 GPU: cargo test -p rdna-compute --lib --features lab qsa_bf16_pm_wide_rows -- --ignored --nocapture"]
    fn qsa_bf16_pm_wide_rows_match_one_row_calls() {
        const WIDE: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        const CAP: usize = BF16_PM_CAPACITY;
        #[derive(Clone, Copy, Debug)]
        enum Arm {
            Pm(QsaSelectPmArm),
            Default,
        }
        fn run(
            gpu: &mut Gpu,
            p: &IndexedAttentionSelectBatch<'_>,
            mirror: &GpuTensor,
            arm: Arm,
        ) -> (bool, Vec<u8>, Vec<u8>) {
            for t in [p.selected, mirror] {
                gpu.hip
                    .memset(&t.buf, BF16_PM_POISON as i32, t.byte_size())
                    .expect("poison outputs");
            }
            gpu.set_wide_verify_rows(if p.rows >= 2 { WIDE } else { 0 });
            let persisted = match arm {
                Arm::Pm(a) => indexed_attention_select_batch_pm_arm(gpu, p, Some(mirror), a),
                Arm::Default => indexed_attention_select_batch_mirrored(gpu, p, mirror),
            }
            .unwrap_or_else(|e| {
                panic!(
                    "BF16 {arm:?} rows={} blocks={} ps={}: {e}",
                    p.rows, p.block_count, p.position_start
                )
            });
            #[cfg(feature = "lab")]
            g1_launched_identity(gpu, format_args!(
                "case=qsa_bf16_pm identity_scope=terminal_launch_only arm={arm:?} mode=eager format=BF16 R={} position={} blocks={} heads={} dim={} compress={} query_bytes={} pooled_bytes={} selected_bytes={} mirror_bytes={}",
                p.rows, p.position_start, p.block_count, p.index_heads, p.index_dim,
                p.compress, p.query.byte_size(), p.pooled.byte_size(),
                p.selected.byte_size(), mirror.byte_size()
            ));
            gpu.set_wide_verify_rows(0);
            (
                persisted,
                gpu.download_raw_bytes(p.selected)
                    .expect("selected download"),
                gpu.download_raw_bytes(mirror).expect("mirror download"),
            )
        }
        let Some(mut gpu) = try_gpu() else {
            assert!(!cfg!(feature = "lab"), "G1 refused: no GPU");
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.is_gfx1151() {
            assert!(!cfg!(feature = "lab"), "G1 refused: needs gfx1151");
            eprintln!("skip: needs gfx1151");
            return;
        }
        let case = Bf16PmSelect::new(&mut gpu);
        let host_query = gpu.download_f32(&case.query).expect("query download");
        let row_q: Vec<GpuTensor> = (0..WIDE)
            .map(|r| {
                gpu.upload_f32(
                    &host_query[r * BF16_PM_QUERY_STRIDE..(r + 1) * BF16_PM_QUERY_STRIDE],
                    &[BF16_PM_QUERY_STRIDE],
                )
                .expect("row query upload")
            })
            .collect();
        let one_selected_len = CAP * 4 + BF16_PM_CANARY;
        let selected1 = gpu
            .upload_raw(&vec![BF16_PM_POISON; one_selected_len], &[one_selected_len])
            .expect("one-row selected upload");
        let mirror1 = gpu
            .upload_raw(&vec![BF16_PM_POISON; one_selected_len], &[one_selected_len])
            .expect("one-row mirror upload");
        let arms = [
            Arm::Pm(QsaSelectPmArm::Hipcc),
            Arm::Pm(QsaSelectPmArm::ScoreOnly),
            Arm::Pm(QsaSelectPmArm::Both),
            Arm::Default,
        ];
        let mut cases = 0usize;
        for base in [8000usize, 4000, 1030] {
            for disp in 0..BF16_PM_COMPRESS {
                let end = base - disp;
                for n in 1..=WIDE {
                    let ps = end - n;
                    let p = IndexedAttentionSelectBatch {
                        pooled_bf16: None,
                        pooled_bf16_valid_blocks: 0,
                        query: &case.query,
                        pooled: &case.pooled,
                        selected: &case.selected,
                        rows: n,
                        query_row_stride: BF16_PM_QUERY_STRIDE,
                        block_count: end / BF16_PM_COMPRESS,
                        index_heads: BF16_PM_HEADS,
                        index_dim: BF16_PM_DIM,
                        budget_blocks: BF16_PM_BUDGET,
                        compress: BF16_PM_COMPRESS,
                        position_start: ps,
                        capacity: CAP,
                        shape_blocks: BF16_PM_BLOCKS,
                    };
                    let reference = run(&mut gpu, &p, &case.mirror, Arm::Pm(QsaSelectPmArm::Hipcc));
                    assert!(reference.0, "fused BF16 arm did not persist: E={end} n={n}");
                    for arm in arms {
                        let ctx = format!("{arm:?} E={end} n={n}");
                        let got = run(&mut gpu, &p, &case.mirror, arm);
                        assert!(got.0, "{ctx}: mirror not persisted");
                        assert!(
                            got.1 == reference.1,
                            "{ctx}: selection differs from the fused BF16 arm"
                        );
                        assert!(
                            got.2 == reference.2,
                            "{ctx}: mirror differs from the fused BF16 arm"
                        );
                        for r in 0..n {
                            let p1 = IndexedAttentionSelectBatch {
                                query: &row_q[r],
                                selected: &selected1,
                                rows: 1,
                                block_count: (ps + r + 1) / BF16_PM_COMPRESS,
                                position_start: ps + r,
                                ..p
                            };
                            let one = run(&mut gpu, &p1, &mirror1, arm);
                            assert!(one.0, "{ctx}: one-row call {r} did not persist");
                            assert!(
                                one.1[..CAP * 4] == got.1[r * CAP * 4..(r + 1) * CAP * 4],
                                "{ctx}: row {r} differs from its one-row call"
                            );
                            assert!(
                                one.1[CAP * 4..].iter().all(|b| *b == BF16_PM_POISON),
                                "{ctx}: one-row call {r} wrote past its row"
                            );
                            if r + 1 == n {
                                assert!(
                                    one.2[..CAP * 4] == got.2[..CAP * 4],
                                    "{ctx}: batch mirror differs from the last one-row call"
                                );
                            }
                        }
                        cases += 1;
                    }
                }
            }
        }
        eprintln!("qsa_bf16_pm_wide_rows: {cases} cases identical");
        for t in row_q.into_iter().chain([selected1, mirror1]) {
            gpu.free_tensor(t).expect("free");
        }
        case.free(&mut gpu);
    }

    fn upload_bf16_values(gpu: &mut Gpu, values: &[f32]) -> GpuTensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect();
        let mut tensor = gpu.upload_raw(&bytes, &[bytes.len()]).expect("bf16 upload");
        tensor.dtype = DType::BF16;
        tensor.shape = vec![values.len()];
        tensor
    }

    /// Wide-verify oracle of the QSA decode prologue (`indexed_attention_decode_prologue`,
    /// grid.y = rows) for F32 and every quantized cache format of the device: a
    /// batch of n = 1..=64 rows at position P writes, bit for bit, what n one-row
    /// prologues at P + r write (normalized index query, Q with the untouched gate,
    /// normalized K, the V source, every K/V cache byte of rows P..P+n and the raw
    /// index keys), the poisoned rows just before and after the window stay
    /// poisoned. End depths 8192 / 32768 / 131072 / 262144 minus displacement
    /// 0..3 (RoPE positions straddling compress-4 boundaries); every n at
    /// displacement 0, ragged n elsewhere. One cache of 262144 rows per format
    /// (about 1.1 GiB for F32).
    #[test]
    #[ignore = "needs a GPU with ~1.2 GiB free: cargo test -p rdna-compute --lib --features lab qsa_prologue_wide_rows -- --ignored --nocapture"]
    fn qsa_prologue_wide_rows_equal_one_row_calls() {
        const WIDE: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        let Some(mut gpu) = try_gpu() else {
            assert!(!cfg!(feature = "lab"), "G1 refused: no GPU");
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            assert!(!cfg!(feature = "lab"), "G1 refused: needs gfx11/gfx12");
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        let (heads, kv_heads, head_dim) = (24usize, 2usize, 256usize);
        let (index_heads, index_dim, index_kv_width) = (4usize, 128usize, 128usize);
        let index_width = index_heads * index_dim + index_kv_width;
        let kv_width = kv_heads * head_dim;
        let q_width = 2 * heads * head_dim;
        // 262144 rows plus the canary row after a window that ends at the deepest end.
        let full_cap = 262_144usize + 1;
        let mut formats = vec![QsaKvFormat::F32];
        formats.extend(quantized_qsa_formats(&gpu));
        let norm_values = |len: usize, seed: usize| -> Vec<f32> {
            qsa_lcg(seed, len)
                .into_iter()
                .map(|v| 0.5 + 0.25 * v)
                .collect()
        };
        let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for format in formats {
            let row_units = format.kv_row_units(kv_heads, head_dim);
            let row_bytes = format.kv_row_bytes(kv_heads, head_dim);
            let raw_bytes = index_kv_width * format.index_dtype().size();
            let k_cache = gpu
                .zeros(&[full_cap * row_units], format.kv_dtype())
                .expect("K cache");
            let v_cache = gpu
                .zeros(&[full_cap * row_units], format.kv_dtype())
                .expect("V cache");
            let raw = gpu
                .zeros(&[full_cap * index_kv_width], format.index_dtype())
                .expect("raw index keys");
            let index_norm = upload_bf16_values(&mut gpu, &norm_values(index_dim, 101));
            let q_norm = upload_bf16_values(&mut gpu, &norm_values(head_dim, 103));
            let k_norm = upload_bf16_values(&mut gpu, &norm_values(head_dim, 107));
            // Branch A (one batched call) and branch B (n one-row calls on views).
            let alloc =
                |gpu: &mut Gpu, elems: usize| gpu.zeros(&[elems], DType::F32).expect("input");
            let a = [
                alloc(&mut gpu, WIDE * index_width),
                alloc(&mut gpu, WIDE * q_width),
                alloc(&mut gpu, WIDE * kv_width),
                alloc(&mut gpu, WIDE * kv_width),
            ];
            let b = [
                alloc(&mut gpu, WIDE * index_width),
                alloc(&mut gpu, WIDE * q_width),
                alloc(&mut gpu, WIDE * kv_width),
                alloc(&mut gpu, WIDE * kv_width),
            ];
            let run = |gpu: &mut Gpu, t: [&GpuTensor; 4], position: usize, rows: usize| {
                gpu.set_wide_verify_rows(if rows >= 2 { WIDE } else { 0 });
                let result = indexed_attention_decode_prologue(
                    gpu,
                    &IndexedAttentionDecodePrologue {
                        index_row: t[0],
                        qgate: t[1],
                        keys: t[2],
                        values: t[3],
                        full_keys: &k_cache,
                        full_values: &v_cache,
                        raw_index_keys: &raw,
                        index_q_norm: &index_norm,
                        q_norm: &q_norm,
                        k_norm: &k_norm,
                        index_heads,
                        index_dim,
                        index_kv_width,
                        heads,
                        kv_heads,
                        head_dim,
                        position,
                        rows,
                        format,
                        yarn: crate::rope::RopeYarn::OFF,
                    },
                );
                gpu.set_wide_verify_rows(0);
                result.expect("QSA prologue");
                #[cfg(feature = "lab")]
                g1_launched_identity(gpu, format_args!(
                    "case=qsa_prologue mode=eager format={} R={rows} position={position} heads={heads} kv_heads={kv_heads} head_dim={head_dim} index_heads={index_heads} index_dim={index_dim} index_bytes={} qgate_bytes={} keys_bytes={} values_bytes={} cache_row_bytes={row_bytes} raw_row_bytes={raw_bytes}",
                    format.name(), t[0].byte_size(), t[1].byte_size(),
                    t[2].byte_size(), t[3].byte_size()
                ));
            };
            let name = format.name();
            let mut window = 0usize;
            let mut cases = 0usize;
            for depth in [8192usize, 32768, 131072, 262144] {
                for disp in 0..4usize {
                    let end = depth - disp;
                    for n in 1..=WIDE {
                        window += 1;
                        let ps = end - n;
                        let seed = 31 * window + n;
                        let host = [
                            qsa_lcg(seed + 3, n * index_width),
                            qsa_lcg(seed + 5, n * q_width),
                            qsa_lcg(seed + 7, n * kv_width),
                            qsa_lcg(seed + 11, n * kv_width),
                        ];
                        for branch in [&a, &b] {
                            for (t, h) in branch.iter().zip(&host) {
                                gpu.hip
                                    .memcpy_htod(&t.buf, &to_bytes(h))
                                    .expect("input upload");
                            }
                        }
                        // Rows ps-1 ..= ps+n are poisoned: the window plus one canary row each side.
                        let poison = |gpu: &Gpu| {
                            for (cache, unit) in [
                                (&k_cache, row_bytes),
                                (&v_cache, row_bytes),
                                (&raw, raw_bytes),
                            ] {
                                gpu.hip
                                    .memcpy_htod_offset(
                                        &cache.buf,
                                        (ps - 1) * unit,
                                        &vec![0xA5u8; (n + 2) * unit],
                                    )
                                    .expect("poison cache rows");
                            }
                        };
                        let observe = |gpu: &Gpu, t: &[GpuTensor; 4]| -> Vec<Vec<u8>> {
                            let mut out = Vec::new();
                            for (tensor, len) in [
                                (&t[0], n * index_width * 4),
                                (&t[1], n * q_width * 4),
                                (&t[2], n * kv_width * 4),
                                (&t[3], n * kv_width * 4),
                            ] {
                                out.push(download_bytes(gpu, tensor)[..len].to_vec());
                            }
                            for (cache, unit) in [
                                (&k_cache, row_bytes),
                                (&v_cache, row_bytes),
                                (&raw, raw_bytes),
                            ] {
                                let mut rows = vec![0u8; (n + 2) * unit];
                                gpu.hip
                                    .memcpy_dtoh_at(&mut rows, &cache.buf, (ps - 1) * unit)
                                    .expect("read cache rows");
                                out.push(rows);
                            }
                            out
                        };
                        poison(&gpu);
                        run(&mut gpu, [&a[0], &a[1], &a[2], &a[3]], ps, n);
                        gpu.hip.device_synchronize().expect("sync");
                        let batched = observe(&gpu, &a);
                        poison(&gpu);
                        for r in 0..n {
                            let views = [
                                b[0].sub_offset(r * index_width, index_width),
                                b[1].sub_offset(r * q_width, q_width),
                                b[2].sub_offset(r * kv_width, kv_width),
                                b[3].sub_offset(r * kv_width, kv_width),
                            ];
                            run(
                                &mut gpu,
                                [&views[0], &views[1], &views[2], &views[3]],
                                ps + r,
                                1,
                            );
                        }
                        gpu.hip.device_synchronize().expect("sync");
                        let one_row = observe(&gpu, &b);
                        for (i, what) in [
                            "index_row",
                            "qgate",
                            "keys",
                            "values",
                            "K cache rows",
                            "V cache rows",
                            "raw index key rows",
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            assert!(
                                batched[i] == one_row[i],
                                "{name} E={end} n={n}: {what} of the batch differ from the n one-row prologues"
                            );
                        }
                        // The V source is never normalized; the poisoned rows around the window stay.
                        assert_eq!(
                            batched[3],
                            to_bytes(&host[3]),
                            "{name} E={end} n={n}: values were modified"
                        );
                        for (i, unit) in [(4usize, row_bytes), (5, row_bytes), (6, raw_bytes)] {
                            assert!(
                                batched[i][..unit].iter().all(|b| *b == 0xA5)
                                    && batched[i][(n + 1) * unit..].iter().all(|b| *b == 0xA5),
                                "{name} E={end} n={n}: wrote outside rows [P, P+n) (cache {i})"
                            );
                        }
                        cases += 1;
                    }
                }
            }
            eprintln!("qsa_prologue_wide_rows {name}: {cases} windows identical");
            for t in [k_cache, v_cache, raw, index_norm, q_norm, k_norm]
                .into_iter()
                .chain(a)
                .chain(b)
            {
                gpu.free_tensor(t).expect("free");
            }
        }
    }

    /// Wide-verify oracle of the split-K partial/combine pair at the production
    /// geometry (compress 4, budget 512, capacity 2051, S = 32, layout 2, F32
    /// K/V): for n = 1..=64 rows (end depths 8192 and 262144 minus 0..3, so the
    /// active lengths 2048..2051 occur in every tail phase) every row's whole
    /// workspace region (`[m, l, acc[256]]` per (head, split), sentinel records of
    /// empty slices included) and attention output equal the one-row launch at
    /// `position_start + r`, bit for bit. The wide gate is set through
    /// `Gpu::set_wide_verify_rows`.
    #[test]
    #[ignore = "needs a gfx11/gfx12 GPU with ~1.2 GiB free: cargo test -p rdna-compute --lib --features lab qsa_splitk_wide_workspace -- --ignored --nocapture"]
    fn qsa_splitk_wide_workspace_records_equal_one_row_records() {
        const WIDE: usize = crate::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
        const H: usize = 24;
        const HKV: usize = 2;
        const D: usize = 256;
        const COMPRESS: usize = 4;
        const BUDGET: usize = 512;
        const CAP: usize = 2051;
        const RECORD_BYTES: usize = (D + 2) * 4;
        let Some(mut gpu) = try_gpu() else {
            eprintln!("skip: no GPU");
            return;
        };
        if !gpu.arch_caps.has_gfx11_plus_simt() {
            eprintln!("skip: needs a gfx11/gfx12 GPU");
            return;
        }
        let (splits, layout) = (QSA_DECODE_SPLITK_SPLITS, QSA_DECODE_SPLITK_SCORING_LAYOUT);
        let region = H * splits * RECORD_BYTES;
        let reserve =
            QSA_DECODE_SPLITK_RESERVE_ROWS * H * QSA_DECODE_SPLITK_MAX_SPLITS * RECORD_BYTES;
        let to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        // Production-shaped selection of `visible` keys: `min(512, blocks)` pseudo-random
        // blocks in token order, the causal tail, -1 padding.
        let selection = |seed: usize, visible: usize| -> Vec<i32> {
            let blocks = visible / COMPRESS;
            let mut row = vec![-1i32; CAP];
            let mut state = (seed as u32).wrapping_mul(2_654_435_761) | 1;
            let mut i = 0usize;
            for _ in 0..BUDGET.min(blocks) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let block = (state >> 8) as usize % blocks;
                for j in 0..COMPRESS {
                    row[i] = (block * COMPRESS + j) as i32;
                    i += 1;
                }
            }
            for t in blocks * COMPRESS..visible {
                row[i] = t as i32;
                i += 1;
            }
            row
        };
        for depth in [8192usize, 262_144] {
            let row_f32 = HKV * D;
            let make_cache = |gpu: &mut Gpu, seed: usize| {
                let t = gpu
                    .alloc_tensor(&[depth * row_f32], DType::F32)
                    .expect("K/V alloc");
                let chunk = 8192usize;
                let mut first = 0usize;
                while first < depth {
                    let rows = chunk.min(depth - first);
                    let bytes = to_bytes(&qsa_lcg(seed + first / chunk, rows * row_f32));
                    gpu.hip
                        .memcpy_htod_offset(&t.buf, first * row_f32 * 4, &bytes)
                        .expect("K/V chunk upload");
                    first += rows;
                }
                t
            };
            let keys = make_cache(&mut gpu, 13);
            let values = make_cache(&mut gpu, 29);
            let launch = |gpu: &mut Gpu,
                          q: &GpuTensor,
                          sel: &GpuTensor,
                          out: &GpuTensor,
                          rows: usize,
                          ps: usize| {
                gpu.set_wide_verify_rows(if rows >= 2 { WIDE } else { 0 });
                let result = indexed_attention_attention_splitk_for_test(
                    gpu,
                    &IndexedAttentionAttentionBatch {
                        q_with_gate: q,
                        full_keys: &keys,
                        full_values: &values,
                        selected: sel,
                        output: out,
                        rows,
                        position_start: ps,
                        n_heads: H,
                        n_kv_heads: HKV,
                        head_dim: D,
                        budget_blocks: BUDGET,
                        compress: COMPRESS,
                        capacity: CAP,
                        full_capacity: depth,
                        format: QsaKvFormat::F32,
                        shape_selected: CAP,
                    },
                    splits,
                    layout,
                );
                gpu.set_wide_verify_rows(0);
                result.expect("split-K launch");
                gpu.hip.device_synchronize().expect("sync");
                // The launch's own workspace slot (same size arguments, so the same mapping).
                let bytes = rows * region;
                let ptr = gpu
                    .qsa_gather_scratch(bytes, reserve)
                    .expect("workspace slot");
                // SAFETY: a non-owning view of the mapped scratch prefix, read once.
                let view = unsafe { hip_bridge::DeviceBuffer::from_raw(ptr, bytes) };
                let mut records = vec![0u8; bytes];
                gpu.hip
                    .memcpy_dtoh(&mut records, &view)
                    .expect("workspace download");
                (records, gpu.download_f32(out).expect("output download"))
            };
            let mut cases = 0usize;
            for disp in 0..COMPRESS {
                let end = depth - disp;
                for n in 1..=WIDE {
                    let ps = end - n;
                    let q = qsa_lcg(5 + n + end % 977, n * 2 * H * D);
                    let sel: Vec<Vec<i32>> =
                        (0..n).map(|r| selection(end + r, ps + r + 1)).collect();
                    let q_t = gpu.upload_f32(&q, &[q.len()]).expect("q upload");
                    let sel_bytes: Vec<u8> =
                        sel.iter().flatten().flat_map(|t| t.to_le_bytes()).collect();
                    let sel_t = gpu
                        .upload_raw(&sel_bytes, &[sel_bytes.len()])
                        .expect("selected upload");
                    let out = gpu.zeros(&[n * H * D], DType::F32).expect("output");
                    let (ws, got) = launch(&mut gpu, &q_t, &sel_t, &out, n, ps);
                    for r in 0..n {
                        let q1 = gpu
                            .upload_f32(&q[r * 2 * H * D..(r + 1) * 2 * H * D], &[2 * H * D])
                            .expect("row q upload");
                        let s1_bytes: Vec<u8> =
                            sel[r].iter().flat_map(|t| t.to_le_bytes()).collect();
                        let s1 = gpu
                            .upload_raw(&s1_bytes, &[s1_bytes.len()])
                            .expect("row selected upload");
                        let o1 = gpu.zeros(&[H * D], DType::F32).expect("row output");
                        let (ws1, got1) = launch(&mut gpu, &q1, &s1, &o1, 1, ps + r);
                        assert!(
                            ws[r * region..(r + 1) * region] == ws1[..region],
                            "depth {depth} E={end} n={n}: row {r} workspace records differ from the one-row launch"
                        );
                        assert!(
                            got[r * H * D..(r + 1) * H * D]
                                .iter()
                                .map(|v| v.to_bits())
                                .eq(got1.iter().map(|v| v.to_bits())),
                            "depth {depth} E={end} n={n}: row {r} attention output differs from the one-row launch"
                        );
                        for t in [q1, s1, o1] {
                            gpu.free_tensor(t).expect("free");
                        }
                    }
                    for t in [q_t, sel_t, out] {
                        gpu.free_tensor(t).expect("free");
                    }
                    cases += 1;
                }
            }
            eprintln!("qsa_splitk_wide_workspace depth {depth}: {cases} windows identical");
            gpu.free_tensor(keys).expect("free");
            gpu.free_tensor(values).expect("free");
        }
    }
}

#[cfg(test)]
mod route_selector_tests {
    use super::*;

    #[test]
    fn qsa_select_switches_preserve_all_four_arms() {
        for fits in [false, true] {
            for score in [false, true] {
                for select in [false, true] {
                    let old = if !fits {
                        QsaSelectPmArm::Hipcc
                    } else {
                        match (score, select) {
                            (false, false) => QsaSelectPmArm::Hipcc,
                            (true, false) => QsaSelectPmArm::ScoreOnly,
                            (false, true) => QsaSelectPmArm::SelectOnly,
                            (true, true) => QsaSelectPmArm::Both,
                        }
                    };
                    assert_eq!(qsa_select_pm_selected(fits, score, select), old);
                }
            }
        }
    }

    #[test]
    fn qsa_select_shape_thresholds_and_raw_offsets() {
        for halo in [false, true] {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                for rows in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 511, 512, 513] {
                    for blocks in [
                        0,
                        QSA_SELECT_PM_MAX_BLOCKS - 1,
                        QSA_SELECT_PM_MAX_BLOCKS,
                        QSA_SELECT_PM_MAX_BLOCKS + 1,
                    ] {
                        for stride in [512, usize::MAX] {
                            let below = |n: Option<usize>| n.is_some_and(|v| v < 0x7fff_ff00);
                            let old = halo
                                && matches!(dtype, DType::F32 | DType::BF16)
                                && rows >= 1
                                && blocks <= QSA_SELECT_PM_MAX_BLOCKS
                                && below(rows.checked_mul(stride).and_then(|v| v.checked_mul(4)))
                                && below(blocks.checked_mul(128 * dtype.size()))
                                && below(rows.checked_mul(1027).and_then(|v| v.checked_mul(4)));
                            assert_eq!(
                                qsa_select_pm_shape_fits(
                                    halo, dtype, rows, 4, 128, blocks, stride, 1027,
                                ),
                                old
                            );
                        }
                    }
                }
            }
        }
        for rows in [1usize, 8, 512] {
            assert!(!qsa_select_pm_shape_fits(
                true,
                DType::F32,
                rows,
                3,
                128,
                128,
                512,
                1027
            ));
            assert!(!qsa_select_pm_shape_fits(
                true,
                DType::F32,
                rows,
                4,
                127,
                128,
                512,
                1027
            ));
            assert!(qsa_select_pm_shape_fits(
                true,
                DType::F32,
                rows,
                4,
                128,
                0,
                512,
                1027
            ));
        }
        assert!(!qsa_select_pm_shape_fits(
            true,
            DType::F32,
            0,
            4,
            128,
            128,
            512,
            1027
        ));
    }

    #[test]
    fn qsa_gathered_raw_offset_admission_is_unchanged() {
        for capacity in [32767usize, 32768, 32769, (1 << 24) - 1, 1 << 24] {
            for kv in [1usize, 4, 8] {
                for heads in [4usize, 24, 32] {
                    let scratch = capacity
                        .div_ceil(4)
                        .checked_mul(4)
                        .and_then(|t| t.checked_mul(kv * 512));
                    let old = scratch.is_some_and(|v| v + 4096 <= 0x7fff_ff00)
                        && heads
                            .checked_mul(2048)
                            .is_some_and(|v| v + 4096 <= 0x7fff_ff00)
                        && capacity < 1 << 24
                        && kv * 2048 < 1 << 24;
                    assert_eq!(qsa_gathered_pm_shape_fits(capacity, kv, heads), old);
                }
            }
        }
    }
}

#[cfg(test)]
mod hc_up_twin_route_tests {
    use super::*;

    #[test]
    fn qwen4_hc_up_twin_route_table() {
        // (enabled, exact_gfx1151, rows, expected)
        let cases: [(bool, bool, usize, bool); 10] = [
            (false, true, 145, false),
            (false, true, 16, false),
            (true, false, 145, false),
            (true, true, 1, false),
            (true, true, 15, false),
            (true, true, 16, true),
            (true, true, 145, true),
            (true, true, 401, true),
            (true, true, 529, true),
            (true, false, 16, false),
        ];
        for (enabled, exact, rows, want) in cases {
            assert_eq!(
                qwen4_hc_up_twin_applies(enabled, exact, rows),
                want,
                "enabled={enabled} exact_gfx1151={exact} rows={rows}"
            );
        }
    }
}

/// CPU-only tests of the Q8Ef live-layout ABI: format metadata, strict state
/// validation, kernel selection and the kernargs the production launchers
/// build (the tests call the very builders the launchers call).
#[cfg(test)]
mod gdn_q8ef_tests {
    use super::*;
    use hip_bridge::DeviceBuffer;
    use std::ffi::c_void;

    const F32: GdnStateFormat = GdnStateFormat::F32;
    const Q8: GdnStateFormat = GdnStateFormat::Q8;
    const EF: GdnStateFormat = GdnStateFormat::Q8Ef;
    const FORMATS: [GdnStateFormat; 3] = [F32, Q8, EF];

    fn ptr(address: usize) -> *mut c_void {
        address as *mut c_void
    }

    /// A non-owning tensor over fabricated device addresses (never launched).
    fn tensor(address: usize, shape: &[usize], dtype: DType, bytes: usize) -> GpuTensor {
        GpuTensor {
            buf: unsafe { DeviceBuffer::from_raw(ptr(address), bytes) },
            shape: shape.to_vec(),
            dtype,
        }
    }

    #[derive(Clone, Copy)]
    enum Arg {
        P(usize),
        I(i32),
        U(u32),
        F(f32),
    }

    /// The kernarg bytes of a hand-written argument list.
    fn expected(spec: &[Arg]) -> Vec<u8> {
        let mut blob = KernargBlob::new();
        for arg in spec {
            match *arg {
                Arg::P(value) => blob.push_ptr(ptr(value)),
                Arg::I(value) => blob.push_i32(value),
                Arg::U(value) => blob.push_u32(value),
                Arg::F(value) => blob.push_f32(value),
            }
        }
        blob.as_bytes().to_vec()
    }

    const HEADS: usize = 2;
    const STATE_BASE: usize = 0x1_0000_0000;
    const RESIDUAL_BASE: usize = 0x2_0000_0000;
    const RING_BASE: usize = 0x3_0000_0000;

    fn state_tensor(format: GdnStateFormat) -> GpuTensor {
        tensor(
            STATE_BASE,
            &[format.state_units(HEADS, 128, 128)],
            format.dtype(),
            format.state_bytes(HEADS, 128, 128),
        )
    }

    fn residual_tensor(address: usize) -> GpuTensor {
        let units = EF.residual_units(HEADS, 128, 128);
        tensor(
            address,
            &[units],
            DType::F16,
            EF.residual_bytes(HEADS, 128, 128),
        )
    }

    fn ring_tensor(format: GdnStateFormat, rows: usize) -> GpuTensor {
        let storage = format.storage();
        tensor(
            RING_BASE,
            &[rows * storage.state_units(HEADS, 128, 128)],
            storage.dtype(),
            rows * storage.state_bytes(HEADS, 128, 128),
        )
    }

    fn validate(
        format: GdnStateFormat,
        state: &GpuTensor,
        residual: Option<&GpuTensor>,
        ring: Option<&GpuTensor>,
        rows: usize,
    ) -> HipResult<()> {
        validate_gdn_state(format, state, residual, ring, rows, HEADS, 128, 128)
    }

    #[test]
    fn gdn_q8ef_live_layout() {
        assert_eq!(EF.name(), "q8_ef");
        assert_eq!(Q8.name(), "q8");
        assert_eq!(F32.name(), "f32");
        assert_eq!(EF.storage(), Q8);
        assert_eq!(Q8.storage(), Q8);
        assert_eq!(F32.storage(), F32);
        assert!(EF.has_residual());
        assert!(!Q8.has_residual());
        assert!(!F32.has_residual());
        for (heads, key, value) in [(48, 128, 128), (16, 128, 128), (1, 128, 128), (48, 64, 128)] {
            assert_eq!(EF.supports(key, value), Q8.supports(key, value));
            assert_eq!(EF.dtype(), Q8.dtype());
            assert_eq!(EF.dtype(), DType::Raw);
            assert_eq!(
                EF.state_units(heads, key, value),
                Q8.state_units(heads, key, value)
            );
            assert_eq!(
                EF.state_bytes(heads, key, value),
                Q8.state_bytes(heads, key, value)
            );
            assert_eq!(F32.residual_units(heads, key, value), 0);
            assert_eq!(Q8.residual_units(heads, key, value), 0);
            assert_eq!(Q8.residual_bytes(heads, key, value), 0);
        }
        assert!(!EF.supports(64, 128));
        // 48 value heads: 811,008 Q8 slot bytes, 1,572,864 residual bytes per
        // layer, 54 MiB over 36 layers.
        assert_eq!(EF.state_bytes(48, 128, 128), 811_008);
        assert_eq!(EF.residual_units(48, 128, 128), 48 * 128 * 128);
        assert_eq!(EF.residual_bytes(48, 128, 128), 1_572_864);
        assert_eq!(36 * EF.residual_bytes(48, 128, 128), 54 * 1024 * 1024);
        // The residual index is the Q8 code index: (head * 128 + row) * 128 + col.
        let index = |head: usize, row: usize, col: usize| (head * 128 + row) * 128 + col;
        assert_eq!(index(0, 0, 0), 0);
        assert_eq!(index(47, 127, 127), EF.residual_units(48, 128, 128) - 1);
    }

    #[test]
    fn gdn_q8ef_live_layout_validation() {
        let state = state_tensor(EF);
        let residual = residual_tensor(RESIDUAL_BASE);
        let ring = ring_tensor(EF, 4);
        validate(EF, &state, Some(&residual), None, 1).expect("Q8Ef state");
        validate(EF, &state, Some(&residual), Some(&ring), 4).expect("Q8Ef capture ring");
        validate(Q8, &state_tensor(Q8), None, Some(&ring_tensor(Q8, 4)), 4).expect("Q8 ring");
        validate(F32, &state_tensor(F32), None, Some(&ring_tensor(F32, 2)), 2).expect("F32 ring");

        // A residual is present exactly for a residual format.
        assert!(validate(EF, &state, None, None, 1).is_err());
        assert!(validate(Q8, &state_tensor(Q8), Some(&residual), None, 1).is_err());
        assert!(validate(F32, &state_tensor(F32), Some(&residual), None, 1).is_err());
        // Residual dtype and extent.
        let wrong_dtype = tensor(
            RESIDUAL_BASE,
            &[EF.residual_units(HEADS, 128, 128)],
            DType::F32,
            EF.residual_bytes(HEADS, 128, 128),
        );
        assert_eq!(
            validate(EF, &state, Some(&wrong_dtype), None, 1)
                .unwrap_err()
                .to_string(),
            HipError::new(0, &ComputeError::WrongDtype.to_string()).to_string()
        );
        let short = tensor(
            RESIDUAL_BASE,
            &[EF.residual_units(HEADS, 128, 128) - 1],
            DType::F16,
            EF.residual_bytes(HEADS, 128, 128),
        );
        assert_eq!(
            validate(EF, &state, Some(&short), None, 1)
                .unwrap_err()
                .to_string(),
            HipError::new(0, &ComputeError::WrongShape.to_string()).to_string()
        );
        let long = tensor(
            RESIDUAL_BASE,
            &[EF.residual_units(HEADS, 128, 128) + 1],
            DType::F16,
            EF.residual_bytes(HEADS, 128, 128) + 2,
        );
        assert!(validate(EF, &state, Some(&long), None, 1).is_err());
        let undersized_buffer = tensor(
            RESIDUAL_BASE,
            &[EF.residual_units(HEADS, 128, 128)],
            DType::F16,
            EF.residual_bytes(HEADS, 128, 128) - 2,
        );
        assert!(validate(EF, &state, Some(&undersized_buffer), None, 1).is_err());
        // State dtype and extent follow the format (Q8Ef is a Raw slot).
        assert!(validate(EF, &state_tensor(F32), Some(&residual), None, 1).is_err());
        assert!(validate(F32, &state_tensor(EF), None, None, 1).is_err());
        let short_state = tensor(
            STATE_BASE,
            &[EF.state_units(HEADS, 128, 128) - 1],
            DType::Raw,
            EF.state_bytes(HEADS, 128, 128),
        );
        assert!(validate(EF, &short_state, Some(&residual), None, 1).is_err());
        // Geometry: Q8 family is 128x128 only.
        assert!(validate_gdn_state(EF, &state, Some(&residual), None, 1, HEADS, 64, 128).is_err());
        // The ring is plain Q8 (storage) with `rows` slots.
        let f32_ring = ring_tensor(F32, 4);
        assert!(validate(EF, &state, Some(&residual), Some(&f32_ring), 4).is_err());
        assert!(validate(EF, &state, Some(&residual), Some(&ring_tensor(EF, 3)), 4).is_err());
        assert!(validate(F32, &state_tensor(F32), None, Some(&ring_tensor(Q8, 4)), 4).is_err());
        // The residual never aliases the state or the ring.
        let in_state = residual_tensor(STATE_BASE + 4096);
        assert!(validate(EF, &state, Some(&in_state), None, 1).is_err());
        let in_ring = residual_tensor(RING_BASE + 16);
        assert!(validate(EF, &state, Some(&in_ring), Some(&ring), 4).is_err());
        // The state itself may sit adjacent to the residual.
        let adjacent = residual_tensor(STATE_BASE + EF.state_bytes(HEADS, 128, 128));
        validate(EF, &state, Some(&adjacent), None, 1).expect("adjacent ranges do not overlap");
    }

    #[test]
    fn gdn_q8ef_live_layout_kernarg_state_refuses_mismatch() {
        for format in FORMATS {
            let with = GdnStateKernarg::new(format, ptr(1), Some(ptr(2)));
            let without = GdnStateKernarg::new(format, ptr(1), None);
            assert_eq!(with.is_ok(), format.has_residual(), "{}", format.name());
            assert_eq!(without.is_ok(), !format.has_residual(), "{}", format.name());
        }
    }

    /// `(kernel, pipe128)` of one batched launch for a format.
    fn batched(
        format: GdnStateFormat,
        captured: bool,
        qwen4_scope: bool,
        policy: Qwen4GdnPrefill,
        arch: &str,
        pipe_enabled: bool,
    ) -> (&'static str, bool) {
        gdn_batched_kernel(&GdnBatchedSelect {
            captured,
            format,
            qwen4_scope,
            policy,
            arch_gfx1201: arch == "gfx1201",
            arch_gfx942: arch == "gfx942",
            pipe_enabled,
        })
    }

    /// The pre-Q8Ef selector, verbatim, for F32 and Q8.
    fn frozen_plain(
        format: GdnStateFormat,
        captured: bool,
        qwen4_scope: bool,
        policy: Qwen4GdnPrefill,
        arch: &str,
        pipe_enabled: bool,
    ) -> (&'static str, bool) {
        let canon = !captured && format == Q8 && qwen4_scope && policy == Qwen4GdnPrefill::Canon;
        let pipe128_arch = arch == "gfx1201" || (arch == "gfx942" && format == F32);
        let pipe128 = !canon && !captured && pipe128_arch && pipe_enabled;
        let kernel = match (captured, format) {
            (true, F32) => "gated_delta_step_halves_state128_persistent256_capture_f32",
            (false, F32) if pipe128 => "gated_delta_step_pipe128_f32",
            (false, F32) => "gated_delta_step_halves_state128_persistent256_f32",
            (true, Q8) => "gated_delta_step_halves_state128_persistent256_capture_q8",
            (false, Q8) if canon => "gated_delta_step_halves_state128_persistent256_canon_q8",
            (false, Q8) if pipe128 => "gated_delta_step_pipe128_q8",
            (false, Q8) => "gated_delta_step_halves_state128_persistent256_q8",
            (_, GdnStateFormat::Q8Ef) => unreachable!("not a plain format"),
        };
        (kernel, pipe128)
    }

    #[test]
    fn gdn_q8ef_live_layout_batched_selection() {
        let policies = [
            Qwen4GdnPrefill::Wy,
            Qwen4GdnPrefill::Persistent,
            Qwen4GdnPrefill::Canon,
        ];
        let mut checked = 0usize;
        for captured in [false, true] {
            for scope in [false, true] {
                for policy in policies {
                    for arch in ["gfx1151", "gfx1201", "gfx942", "gfx1100"] {
                        for pipe in [false, true] {
                            // Plain F32 / Q8 selection is the frozen selector.
                            for format in [F32, Q8] {
                                assert_eq!(
                                    batched(format, captured, scope, policy, arch, pipe),
                                    frozen_plain(format, captured, scope, policy, arch, pipe),
                                    "{} captured={captured} scope={scope} {policy:?} {arch} pipe={pipe}",
                                    format.name()
                                );
                            }
                            // Q8Ef is the Q8 twin in every branch: same
                            // branch structure and pipe128 geometry, `_q8` ->
                            // `_q8ef`.
                            let (q8_kernel, q8_pipe) =
                                batched(Q8, captured, scope, policy, arch, pipe);
                            let (ef_kernel, ef_pipe) =
                                batched(EF, captured, scope, policy, arch, pipe);
                            let twin = format!("{}ef", q8_kernel);
                            assert!(q8_kernel.ends_with("_q8"), "{q8_kernel}");
                            assert_eq!(ef_kernel, twin);
                            assert_eq!(ef_pipe, q8_pipe);
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 2 * 2 * 3 * 4 * 2);
        // The named branches of the Q8Ef set.
        let wy = Qwen4GdnPrefill::Wy;
        assert_eq!(
            batched(EF, true, true, wy, "gfx1201", true),
            (
                "gated_delta_step_halves_state128_persistent256_capture_q8ef",
                false
            )
        );
        assert_eq!(
            batched(EF, false, true, Qwen4GdnPrefill::Canon, "gfx1201", true),
            (
                "gated_delta_step_halves_state128_persistent256_canon_q8ef",
                false
            )
        );
        // Canon needs the Qwen4 scope.
        assert_eq!(
            batched(EF, false, false, Qwen4GdnPrefill::Canon, "gfx1201", true),
            ("gated_delta_step_pipe128_q8ef", true)
        );
        assert_eq!(
            batched(EF, false, false, wy, "gfx1151", true),
            ("gated_delta_step_halves_state128_persistent256_q8ef", false)
        );
        // gfx942 pipes F32 only.
        assert_eq!(
            batched(EF, false, false, wy, "gfx942", true),
            ("gated_delta_step_halves_state128_persistent256_q8ef", false)
        );
        assert_eq!(
            batched(F32, false, false, wy, "gfx942", true),
            ("gated_delta_step_pipe128_f32", true)
        );
        // Only a Q8-family state outside capture in the scope reads the policy.
        for (captured, scope, format, reads) in [
            (false, true, Q8, true),
            (false, true, EF, true),
            (false, true, F32, false),
            (true, true, Q8, false),
            (true, true, EF, false),
            (false, false, EF, false),
        ] {
            assert_eq!(
                GdnBatchedSelect::reads_policy(captured, format, scope),
                reads
            );
        }
    }

    #[test]
    fn gdn_q8ef_live_layout_single_step_selection() {
        for gated in [false, true] {
            for gfx11_128 in [false, true] {
                // Plain formats: frozen selection.
                let (q8, q8_shared) = gdn_step_single_kernel(Q8, gated, gfx11_128);
                assert!(q8_shared);
                assert_eq!(
                    q8,
                    if gated {
                        "gated_delta_step_gate_norm128_q8"
                    } else {
                        "gated_delta_step_norm128_q8"
                    }
                );
                let (f32_kernel, f32_shared) = gdn_step_single_kernel(F32, gated, gfx11_128);
                assert_eq!(f32_shared, gfx11_128);
                assert_eq!(
                    f32_kernel,
                    match (gated, gfx11_128) {
                        (true, true) => "gated_delta_step_gate_norm128_gfx1151",
                        (false, true) => "gated_delta_step_shared_norm128_gfx1151",
                        _ => "gated_delta_step_f32",
                    }
                );
                // Q8Ef: the Q8 twin on the same shared-norm geometry.
                let (ef, ef_shared) = gdn_step_single_kernel(EF, gated, gfx11_128);
                assert!(ef_shared);
                assert_eq!(ef, format!("{q8}ef"));
            }
        }
    }

    #[test]
    fn gdn_q8ef_live_layout_rollback_converters_and_chunk_selection() {
        // Storage-based rollback: no EF rollback kernel.
        assert_eq!(gdn_rollback_kernel(F32), "gated_delta_rollback_layers_f32");
        assert_eq!(gdn_rollback_kernel(Q8), "gated_delta_rollback_layers_q8");
        assert_eq!(
            gdn_rollback_kernel(EF.storage()),
            "gated_delta_rollback_layers_q8"
        );
        assert_eq!(gdn_rollback_kernel(EF), "gated_delta_rollback_layers_q8");
        assert!(EF.storage().has_frame_kernarg());
        // Frames: plain Q8 only.
        assert!(Q8.has_frame_kernarg());
        assert!(!EF.has_frame_kernarg());
        assert!(!F32.has_frame_kernarg());
        // Converters.
        assert_eq!(gdn_convert_kernel(Q8, true).unwrap(), "gdn_state_q8_to_f32");
        assert_eq!(
            gdn_convert_kernel(Q8, false).unwrap(),
            "gdn_state_f32_to_q8"
        );
        assert_eq!(
            gdn_convert_kernel(EF, true).unwrap(),
            "gdn_state_q8ef_to_f32"
        );
        assert_eq!(
            gdn_convert_kernel(EF, false).unwrap(),
            "gdn_state_f32_to_q8ef"
        );
        assert!(gdn_convert_kernel(F32, true).is_err());
        assert!(gdn_convert_kernel(F32, false).is_err());
        // Chunk kernels and modules (registry names).
        let table = [
            (
                Q8,
                true,
                false,
                "gated_delta_chunk_q8_wmma",
                "gated_delta_chunk_gate_q8_wmma",
            ),
            (
                Q8,
                true,
                true,
                "gated_delta_chunk_q8_wmma_arena",
                "gated_delta_chunk_gate_q8_wmma_arena",
            ),
        ];
        for (format, inline, arena, module, kernel) in table {
            assert_eq!(gdn_chunk_module_name(format, inline, arena), module);
            assert_eq!(gdn_chunk_kernel_name(format, inline, arena), kernel);
        }
        for format in FORMATS {
            assert_eq!(
                gdn_chunk_module_name(format, false, false),
                "gated_delta_chunk_wmma"
            );
            assert_eq!(
                gdn_chunk_kernel_name(format, false, false),
                "gated_delta_chunk_gate_wmma"
            );
            assert_eq!(
                gdn_chunk_module_name(format, false, true),
                "gated_delta_chunk_wmma_arena"
            );
            assert_eq!(
                gdn_chunk_kernel_name(format, false, true),
                "gated_delta_chunk_gate_wmma_arena"
            );
        }
        // Sources follow the format: only the plain Q8 inline kernel has its own file (it
        // carries no EF text, so its code object is the plain Q8 kernel's alone), every other
        // launch is the F32 chunk source.
        for (format, inline, arena, source) in [
            (Q8, true, false, GATED_DELTA_CHUNK_Q8_WMMA_SRC),
            (Q8, true, true, npu_arena::CHUNK_Q8_ARENA_SRC),
        ] {
            assert_eq!(gdn_chunk_source(format, inline, arena), source);
        }
        for format in FORMATS {
            assert_eq!(
                gdn_chunk_source(format, false, false),
                GATED_DELTA_CHUNK_WMMA_SRC
            );
            assert_eq!(
                gdn_chunk_source(format, false, true),
                npu_arena::CHUNK_ARENA_SRC
            );
        }
        for source in [GATED_DELTA_CHUNK_Q8_WMMA_SRC, npu_arena::CHUNK_Q8_ARENA_SRC] {
            assert!(
                !source.contains("q8ef"),
                "the plain Q8 source must not carry the EF entry"
            );
            assert!(
                !source.contains("ef_row"),
                "the plain Q8 source must not carry EF code"
            );
        }
        // Inline format predicate: only plain Q8 without a residual; Q8Ef has no inline kernel.
        for (format, has_residual, want) in [
            (F32, false, false),
            (F32, true, false),
            (Q8, false, true),
            (Q8, true, false),
            (EF, true, false),
            (EF, false, false),
        ] {
            assert_eq!(
                gdn_chunk_inline_format(format, has_residual),
                want,
                "{} residual={has_residual}",
                format.name()
            );
        }
        assert!(!gdn_chunk_inline_format(GdnStateFormat::Q8Ef, true));
        assert!(!gdn_chunk_inline_format(GdnStateFormat::Q8Ef, false));
        assert!(gdn_chunk_inline_format(GdnStateFormat::Q8, false));
        assert!(!gdn_chunk_inline_format(GdnStateFormat::Q8, true));
        assert!(!gdn_chunk_inline_format(GdnStateFormat::F32, false));
        // A Q8Ef chunk always runs the F32 chunk kernel on its reconstruction.
        assert_eq!(
            gdn_chunk_module_name(GdnStateFormat::Q8Ef, false, false),
            "gated_delta_chunk_wmma"
        );
        assert_eq!(
            gdn_chunk_module_name(GdnStateFormat::Q8Ef, false, true),
            "gated_delta_chunk_wmma_arena"
        );
    }

    #[test]
    fn gdn_q8ef_live_layout_step_kernargs() {
        let base = |state: GdnStateKernarg, output: GdnStepOutput, frame: u32| GdnStepKernargs {
            q: ptr(0x10),
            k: ptr(0x20),
            v: ptr(0x30),
            gate: ptr(0x40),
            beta: ptr(0x50),
            state,
            output,
            key_heads: 16,
            value_heads: 48,
            key_dim: 128,
            value_dim: 128,
            query_scale: 0.0883883,
            frame,
        };
        let head = |tail: &[Arg]| {
            let mut spec = vec![
                Arg::P(0x10),
                Arg::P(0x20),
                Arg::P(0x30),
                Arg::P(0x40),
                Arg::P(0x50),
            ];
            spec.extend_from_slice(tail);
            spec
        };
        let dims = [
            Arg::I(16),
            Arg::I(48),
            Arg::I(128),
            Arg::I(128),
            Arg::F(0.0883883),
        ];
        let recurrent = GdnStepOutput::Recurrent { output: ptr(0x60) };

        // Plain Q8: state, output, dims, trailing frame (+ replay offset).
        let q8 = GdnStateKernarg::new(Q8, ptr(0x70), None).unwrap();
        let (blob, frame_offset) = gdn_step_kernargs(&base(q8, recurrent, 9));
        let mut spec = head(&[Arg::P(0x70), Arg::P(0x60)]);
        spec.extend_from_slice(&dims);
        spec.push(Arg::U(9));
        assert_eq!(blob.as_bytes(), expected(&spec));
        assert_eq!(frame_offset, Some(blob.len() - 4));

        // F32: no frame.
        let f32_state = GdnStateKernarg::new(F32, ptr(0x70), None).unwrap();
        let (blob, frame_offset) = gdn_step_kernargs(&base(f32_state, recurrent, 9));
        let mut spec = head(&[Arg::P(0x70), Arg::P(0x60)]);
        spec.extend_from_slice(&dims);
        assert_eq!(blob.as_bytes(), expected(&spec));
        assert_eq!(frame_offset, None);

        // Q8Ef: ef right after the state pointer, no frame, no replay binding.
        let ef = GdnStateKernarg::new(EF, ptr(0x70), Some(ptr(0x80))).unwrap();
        let (blob, frame_offset) = gdn_step_kernargs(&base(ef, recurrent, 9));
        let mut spec = head(&[Arg::P(0x70), Arg::P(0x80), Arg::P(0x60)]);
        spec.extend_from_slice(&dims);
        assert_eq!(blob.as_bytes(), expected(&spec));
        assert_eq!(frame_offset, None);

        // Gated + rotation: z, norm, output, dims, rotation pointers (then
        // the Q8 frame for plain Q8 only).
        let gated = GdnStepOutput::Gated {
            z: ptr(0x90),
            norm: ptr(0xa0),
            output: ptr(0xb0),
            rotate: Some([ptr(0xc0), ptr(0xd0), ptr(0xe0), ptr(0xf0)]),
        };
        let (blob, frame_offset) = gdn_step_kernargs(&base(ef, gated, 9));
        let mut spec = head(&[
            Arg::P(0x70),
            Arg::P(0x80),
            Arg::P(0x90),
            Arg::P(0xa0),
            Arg::P(0xb0),
        ]);
        spec.extend_from_slice(&dims);
        spec.extend_from_slice(&[Arg::P(0xc0), Arg::P(0xd0), Arg::P(0xe0), Arg::P(0xf0)]);
        assert_eq!(blob.as_bytes(), expected(&spec));
        assert_eq!(frame_offset, None);
        let (blob, frame_offset) = gdn_step_kernargs(&base(q8, gated, 9));
        let mut spec = head(&[Arg::P(0x70), Arg::P(0x90), Arg::P(0xa0), Arg::P(0xb0)]);
        spec.extend_from_slice(&dims);
        spec.extend_from_slice(&[
            Arg::P(0xc0),
            Arg::P(0xd0),
            Arg::P(0xe0),
            Arg::P(0xf0),
            Arg::U(9),
        ]);
        assert_eq!(blob.as_bytes(), expected(&spec));
        assert_eq!(frame_offset, Some(blob.len() - 4));
    }

    #[test]
    fn gdn_q8ef_live_layout_batched_kernargs() {
        let make = |state: GdnStateKernarg, ring: Option<*mut c_void>| GdnBatchedKernargs {
            projection: ptr(0x10),
            gate: ptr(0x20),
            beta: ptr(0x30),
            state,
            output: ptr(0x50),
            row_states: ring,
            rows: 4,
            qkv_width: 10240,
            key_heads: 16,
            value_heads: 48,
            key_dim: 128,
            value_dim: 128,
            query_scale: 0.0883883,
            frame: 77,
        };
        let dims = [
            Arg::I(4),
            Arg::I(10240),
            Arg::I(16),
            Arg::I(48),
            Arg::I(128),
            Arg::I(128),
            Arg::F(0.0883883),
        ];
        let q8 = GdnStateKernarg::new(Q8, ptr(0x40), None).unwrap();
        let ef = GdnStateKernarg::new(EF, ptr(0x40), Some(ptr(0x45))).unwrap();
        let f32_state = GdnStateKernarg::new(F32, ptr(0x40), None).unwrap();
        for ring in [None, Some(ptr(0x60))] {
            let ring_arg: Vec<Arg> = ring.iter().map(|_| Arg::P(0x60)).collect();
            // Plain Q8: ..., state, output, [ring], dims, frame.
            let blob = gdn_batched_kernargs(&make(q8, ring));
            let mut spec = vec![
                Arg::P(0x10),
                Arg::P(0x20),
                Arg::P(0x30),
                Arg::P(0x40),
                Arg::P(0x50),
            ];
            spec.extend_from_slice(&ring_arg);
            spec.extend_from_slice(&dims);
            spec.push(Arg::U(77));
            assert_eq!(
                blob.as_bytes(),
                expected(&spec),
                "Q8 ring={}",
                ring.is_some()
            );
            // F32: no frame.
            let blob = gdn_batched_kernargs(&make(f32_state, ring));
            let mut spec = vec![
                Arg::P(0x10),
                Arg::P(0x20),
                Arg::P(0x30),
                Arg::P(0x40),
                Arg::P(0x50),
            ];
            spec.extend_from_slice(&ring_arg);
            spec.extend_from_slice(&dims);
            assert_eq!(
                blob.as_bytes(),
                expected(&spec),
                "F32 ring={}",
                ring.is_some()
            );
            // Q8Ef: ef right after state (before output / ring), no frame.
            let blob = gdn_batched_kernargs(&make(ef, ring));
            let mut spec = vec![
                Arg::P(0x10),
                Arg::P(0x20),
                Arg::P(0x30),
                Arg::P(0x40),
                Arg::P(0x45),
                Arg::P(0x50),
            ];
            spec.extend_from_slice(&ring_arg);
            spec.extend_from_slice(&dims);
            assert_eq!(
                blob.as_bytes(),
                expected(&spec),
                "Q8Ef ring={}",
                ring.is_some()
            );
        }
    }

    #[test]
    fn gdn_q8ef_live_layout_chunk_kernargs() {
        let make = |state: GdnChunkState| GdnChunkKernargs {
            projection: ptr(0x10),
            qn: ptr(0x20),
            kn: ptr(0x30),
            gate: ptr(0x40),
            beta: ptr(0x50),
            state,
            z: ptr(0x70),
            norm: ptr(0x80),
            output: ptr(0x90),
            rows: 512,
            qkv_width: 10240,
            key_heads: 16,
            value_heads: 48,
            query_scale: 0.0883883,
            output_bf16: 1,
        };
        let head = [
            Arg::P(0x10),
            Arg::P(0x20),
            Arg::P(0x30),
            Arg::P(0x40),
            Arg::P(0x50),
        ];
        let tail = [Arg::P(0x70), Arg::P(0x80), Arg::P(0x90)];
        let dims = [
            Arg::I(512),
            Arg::I(10240),
            Arg::I(16),
            Arg::I(48),
            Arg::F(0.0883883),
            Arg::I(1),
        ];
        let build = |state: &[Arg], mode: &[Arg]| {
            let mut spec = head.to_vec();
            spec.extend_from_slice(state);
            spec.extend_from_slice(&tail);
            spec.extend_from_slice(&dims);
            spec.extend_from_slice(mode);
            expected(&spec)
        };
        // F32 chunk: no mode, no frame.
        let blob = gdn_chunk_kernargs(&make(GdnChunkState::F32 { state: ptr(0x60) }));
        assert_eq!(blob.as_bytes(), build(&[Arg::P(0x60)], &[]));
        // Inline Q8: mode 1, frame.
        let blob = gdn_chunk_kernargs(&make(GdnChunkState::Q8Inline {
            state: ptr(0x60),
            frame: 531,
        }));
        assert_eq!(
            blob.as_bytes(),
            build(&[Arg::P(0x60)], &[Arg::I(1), Arg::U(531)])
        );
    }

    #[test]
    fn gdn_q8ef_live_layout_converter_and_zero_kernargs() {
        let q8 = GdnStateKernarg::new(Q8, ptr(0x10), None).unwrap();
        let ef = GdnStateKernarg::new(EF, ptr(0x10), Some(ptr(0x20))).unwrap();
        // To F32: (slot, [ef], dst, heads).
        assert_eq!(
            gdn_convert_kernargs(true, &q8, ptr(0x30), 48, 5).as_bytes(),
            expected(&[Arg::P(0x10), Arg::P(0x30), Arg::I(48)])
        );
        assert_eq!(
            gdn_convert_kernargs(true, &ef, ptr(0x30), 48, 5).as_bytes(),
            expected(&[Arg::P(0x10), Arg::P(0x20), Arg::P(0x30), Arg::I(48)])
        );
        // From F32: (src, slot, [ef], heads, [frame]); the frame is Q8's only.
        assert_eq!(
            gdn_convert_kernargs(false, &q8, ptr(0x30), 48, 5).as_bytes(),
            expected(&[Arg::P(0x30), Arg::P(0x10), Arg::I(48), Arg::U(5)])
        );
        assert_eq!(
            gdn_convert_kernargs(false, &ef, ptr(0x30), 48, 5).as_bytes(),
            expected(&[Arg::P(0x30), Arg::P(0x10), Arg::P(0x20), Arg::I(48)])
        );
        // Residual zero: (ef, count), grid ceil(count / 256).
        assert_eq!(
            gdn_q8ef_zero_kernargs(ptr(0x40), 1_572_864 / 2).as_bytes(),
            expected(&[Arg::P(0x40), Arg::U(1_572_864 / 2)])
        );
        for (count, grid) in [(1usize, 1u32), (256, 1), (257, 2), (48 * 128 * 128, 3072)] {
            assert_eq!(blocks(count).unwrap(), grid, "count {count}");
        }
    }
}
