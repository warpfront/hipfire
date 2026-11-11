// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next's gfx1201 prefill GDN on the dense C64 chunk scan.
//!
//! The fused producer preserves the lab route's serial BF16 Q/K norm,
//! F16 packing and chunk-local gate cumsum. The consumer reads the scan's
//! BF16 output directly. The Q8 state aliases the scan's codes/scales. A plain Q8 state's EF
//! enters as zero (thread-local scratch, cleared by the producer) and is discarded on exit
//! (FN keeps none). A Q8Ef state scans against its live per-layer residual (`p.residual`,
//! F16 `[head][row][key]`): the producer's clear is disabled (`ef_words == 0`), the scan
//! rebuilds `code * scale + ef` on entry and stores the new residual on exit, so the residual
//! persists across prefill launches. Exit Q8 uses
//! round-to-nearest-even, not the pipe128 recurrence's stochastic rounding.
//! Default on for admitted steps; `HIPFIRE_FN_GDN_DENSE_SCAN=0` opts out.
//!
//! `HIPFIRE_QWEN4_GDN_CHAIN_FUSE` (exact gfx1201, default on, `0` restores the v1 producer and
//! the F32 projection): the qkv projection arrives as BF16 bits (the producer only reads it
//! through its BF16 rounding) and the load-batched `fn_gdn_dense_producer_v2_{f32,bf16}` entries
//! run; every stored byte is the v1 producer's.
//!
//! Head geometry: the kernels are compiled for the Flash-Next 16 key / 48 value heads and for the
//! TP2 trunk shard's rank-local 8 / 24 ([`crate::scratch::QWEN4_DENSE_GDN_GEOMETRIES`], the one
//! pure admission shared with the prefill-workspace census). The 16 / 48 entries are byte for byte
//! the pre-shard ones; the 8 / 24 geometry has its own load-batched producer entries
//! (`..._k8v24`) and gate entry (`fn_gdn_dense_gate_i8_p12`), so it needs
//! `HIPFIRE_QWEN4_GDN_CHAIN_FUSE` (the v1 producer is 16 / 48 only). The KKT solve and the chunk
//! scan take their head counts as kernel arguments. The thread-local A / EF scratch is sized at
//! the 48-head upper bound, which is also what the census charges.

use std::cell::RefCell;
use std::ffi::c_void;

use hip_bridge::{DeviceBuffer, HipResult, KernargBlob};

use crate::tensor_ops::{
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched, GatedDeltaStepBatched,
    GdnStateFormat,
};
use crate::{DType, Gpu, GpuTensor};

/// Bytes of the thread-local zero-entry EF scratch at the 48-value-head upper bound over every
/// admitted geometry ([`crate::scratch::QWEN4_DENSE_GDN_GEOMETRIES`]): a smaller geometry zeroes
/// and reads only the first [`Geometry::ef_bytes`] of it.
const EF_BYTES: usize = 48 * 128 * 128 * 2;
/// Bytes of one A row of the KKT solve at the same upper bound: 48 value heads x 64 BF16 columns
/// (A is `[row][value head][64]`, so a 24-head geometry strides its rows by half of this).
const A_ROW_BYTES: usize = 48 * 64 * 2;
/// Admission floor of [`applies`]: below this row count the route never runs.
const MIN_ROWS: usize = 64;

/// One admitted head geometry ([`crate::scratch::QWEN4_DENSE_GDN_GEOMETRIES`]; 128-wide heads).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Geometry {
    key_heads: usize,
    value_heads: usize,
}

impl Geometry {
    /// `Some` for an admitted `(key heads, value heads)` pair.
    fn of(key_heads: usize, value_heads: usize) -> Option<Self> {
        crate::scratch::QWEN4_DENSE_GDN_GEOMETRIES
            .contains(&(key_heads, value_heads))
            .then_some(Self {
                key_heads,
                value_heads,
            })
    }

    /// Q (or K) channels of the convolution.
    fn qk(self) -> usize {
        self.key_heads * 128
    }

    /// V channels of the convolution (= the gated output width = the out projection's K).
    fn value_width(self) -> usize {
        self.value_heads * 128
    }

    /// Convolution channels: q | k | v.
    fn channels(self) -> usize {
        2 * self.qk() + self.value_width()
    }

    /// Producer grid x: one block per q head, k head and v head, the gate-params block and the
    /// EF-zero block (82 for 16 / 48).
    fn producer_blocks(self) -> usize {
        2 * self.key_heads + self.value_heads + 2
    }

    /// Bytes of one state's F16 EF plane (`[head][row][key]`).
    fn ef_bytes(self) -> usize {
        self.value_heads * 128 * 128 * 2
    }

    /// Bytes of the Q8 codes of the state slot (`[head][row][key]` i8); the F32 row scales follow.
    fn state_codes(self) -> usize {
        self.value_heads * 128 * 128
    }

    /// Bytes of the Q8 row scales of the state slot (`[head][row]` f32).
    fn state_scales(self) -> usize {
        self.value_heads * 128 * 4
    }

    /// Producer entry of this geometry: the v1 entry (16 / 48 with the chain fusion off) or the
    /// load-batched entry by input dtype; `None` where the geometry has no such entry.
    fn producer_entry(self, chain_fuse: bool, bf16_input: bool) -> Option<&'static str> {
        match (self.key_heads, self.value_heads, chain_fuse, bf16_input) {
            (16, 48, false, _) => Some("fn_gdn_dense_producer"),
            (16, 48, true, true) => Some("fn_gdn_dense_producer_v2_bf16"),
            (16, 48, true, false) => Some("fn_gdn_dense_producer_v2_f32"),
            (8, 24, true, true) => Some("fn_gdn_dense_producer_v2_bf16_k8v24"),
            (8, 24, true, false) => Some("fn_gdn_dense_producer_v2_f32_k8v24"),
            _ => None,
        }
    }
}

/// The `fn_gdn_dense_gate_i8` entry of a value-head count (24 head pairs per row for 48 heads, 12
/// for 24); `None` where no entry exists.
fn gate_i8_entry(value_heads: usize, value_dim: usize) -> Option<&'static str> {
    match (value_heads, value_dim) {
        (48, 128) => Some("fn_gdn_dense_gate_i8"),
        (24, 128) => Some("fn_gdn_dense_gate_i8_p12"),
        _ => None,
    }
}

/// The row-count part of [`applies`]: at least [`MIN_ROWS`] rows. For callers
/// aligned to 512-row seams, every final segment, including 1..63 rows, stays
/// on the scan route. KKT and scan mask its partial C64 microtile
/// (`gdn_chunk_tail_kill`); this does not change the call-local tile origin.
fn rows_admitted(n: usize) -> bool {
    n >= MIN_ROWS
}

/// The geometry part of [`applies`]: an admitted head geometry
/// ([`crate::scratch::qwen4_dense_gdn_geometry_admits`]) with a producer entry for the chain
/// fusion setting. The 8 / 24 shard geometry has only the load-batched entries.
fn geometry_runs(
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    qkv_width: usize,
    chain_fuse: bool,
) -> bool {
    crate::scratch::qwen4_dense_gdn_geometry_admits(
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        qkv_width,
    ) && Geometry::of(key_heads, value_heads)
        .is_some_and(|geometry| geometry.producer_entry(chain_fuse, false).is_some())
}

/// Bytes of the thread-local A scratch for a `rows`-row step: `round64(rows)`
/// rows of [`A_ROW_BYTES`] (KKT writes whole 64-row blocks). 0 below the
/// [`applies`] row floor, where the route never allocates; `None` on overflow.
pub fn workspace_a_bytes(rows: usize) -> Option<usize> {
    if rows < MIN_ROWS {
        return Some(0);
    }
    rows.div_ceil(64).checked_mul(64)?.checked_mul(A_ROW_BYTES)
}

/// Bytes of the thread-local zero-entry EF scratch of a plain Q8 state (one per allocation,
/// rows-independent); 0 below the [`applies`] row floor. A Q8Ef state scans against its live
/// residual instead and never allocates this ([`workspace_ef_bytes_for`]).
pub fn workspace_ef_bytes(rows: usize) -> usize {
    if rows < MIN_ROWS {
        0
    } else {
        EF_BYTES
    }
}

/// A scratch bytes the dense scan charges for a state stored in `format`: the
/// [`workspace_a_bytes`] of an admitted Q8 / Q8Ef state, 0 for F32 (the route is unreachable).
pub fn workspace_a_bytes_for(format: GdnStateFormat, rows: usize) -> Option<usize> {
    match format {
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => workspace_a_bytes(rows),
        GdnStateFormat::F32 => Some(0),
    }
}

/// Thread-local EF scratch bytes the dense scan charges for `format`: [`workspace_ef_bytes`]
/// for plain Q8 only. A Q8Ef state's persistent residual belongs to the state owner and is
/// not charged here; F32 never reaches the route.
pub fn workspace_ef_bytes_for(format: GdnStateFormat, rows: usize) -> usize {
    match format {
        GdnStateFormat::Q8 => workspace_ef_bytes(rows),
        GdnStateFormat::Q8Ef | GdnStateFormat::F32 => 0,
    }
}

/// Total thread-local workspace (A + EF) of the dense scan for `rows` rows of a plain Q8
/// state, `None` on overflow. Excludes the aliased Q/K/V/G/scan-output views.
pub fn workspace_bytes(rows: usize) -> Option<usize> {
    workspace_bytes_for(GdnStateFormat::Q8, rows)
}

/// [`workspace_bytes`] for a state stored in `format`.
pub fn workspace_bytes_for(format: GdnStateFormat, rows: usize) -> Option<usize> {
    workspace_a_bytes_for(format, rows)?.checked_add(workspace_ef_bytes_for(format, rows))
}

struct Scratch {
    rows: usize,
    a: GpuTensor,
    /// Zero-entry EF plane of a plain Q8 state; allocated on first plain-Q8 use and never for
    /// Q8Ef (which scans against the owner's live residual).
    ef: Option<GpuTensor>,
    /// Dedicated Q/K/V planes of an edge step (`rows` too small for the alias into the F32
    /// projection, see [`edge_planes_needed`]); never persistent carry.
    planes: Option<GpuTensor>,
}

/// Whether a `rows`-row step must scan from dedicated Q/K/V planes: the planes are
/// `round64(rows)` BF16 rows of every channel, which fits the dead F32 conv destination
/// (`rows` F32 rows) only for `rows >= 32`. Below that an alias would overrun it.
pub const fn edge_planes_needed(rows: usize) -> bool {
    rows.div_ceil(64) * 64 > 2 * rows
}

/// Device bytes of the dedicated edge workspace of a `rows`-row dense step on a head geometry:
/// the A scratch for the padded 64-row block, plus (rows < 32) the dedicated Q/K/V planes. Charged
/// once per device by the scratch census, never per layer or per snapshot. `None` for an
/// unadmitted geometry or on overflow.
pub fn edge_workspace_bytes(key_heads: usize, value_heads: usize, rows: usize) -> Option<usize> {
    let geometry = Geometry::of(key_heads, value_heads)?;
    let n64 = rows.max(1).div_ceil(64).checked_mul(64)?;
    let a = n64.checked_mul(A_ROW_BYTES)?;
    let planes = if edge_planes_needed(rows.max(1)) {
        n64.checked_mul(geometry.channels())?.checked_mul(2)?
    } else {
        0
    };
    a.checked_add(planes)
}

/// [`applies`] without the row floor: the static admission of the dense scan for the edge
/// executor (every row count, 1..63 included), whose workspace is dedicated
/// ([`edge_workspace_bytes`]). The caller wires the scan's edge entry/exit contract
/// (S2: Q8/EF/F32 tile-entry state, absolute seams); this predicate never selects the route
/// by row count.
pub fn applies_edge(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.flags.fn_gdn_dense_scan
        && admits_state(&gpu.arch, p.format, p.residual.is_some())
        && p.row_states.is_none()
        && p.rows > 0
        && geometry_runs(
            p.key_heads,
            p.value_heads,
            p.key_dim,
            p.value_dim,
            p.qkv_width,
            gpu.flags.qwen4_gdn_chain_fuse_enabled(),
        )
        && p.projection.dtype == DType::F32
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
}

/// Static admission of the dense C64 stream (Scan64 executor) for a state in `format` on
/// `arch`: the dense scan's architecture / format / geometry admission with the chain-fusion
/// producer entry, no row count and no device handle (the census of an admission decision has
/// none). The stream carries a Q8-family state only (F32 has no seam).
pub fn stream_geometry_admitted(
    arch: &str,
    format: GdnStateFormat,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    chain_fuse: bool,
) -> bool {
    matches!(format, GdnStateFormat::Q8 | GdnStateFormat::Q8Ef)
        && crate::scratch::qwen4_dense_gdn_admits(arch, format)
        && geometry_runs(
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            (2 * key_heads + value_heads) * key_dim,
            chain_fuse,
        )
}

/// [`stream_geometry_admitted`] under this device's flags (`HIPFIRE_FN_GDN_DENSE_SCAN`,
/// `HIPFIRE_QWEN4_GDN_CHAIN_FUSE`).
pub fn stream_admitted(
    gpu: &Gpu,
    format: GdnStateFormat,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> bool {
    gpu.flags.fn_gdn_dense_scan
        && stream_geometry_admitted(
            &gpu.arch,
            format,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            gpu.flags.qwen4_gdn_chain_fuse_enabled(),
        )
}

thread_local! {
    /// One scratch per device id: a single submission thread can drive several
    /// devices (pipeline-parallel bands), and a buffer allocated on one device
    /// must never be handed to a kernel running on another.
    static SCRATCH: RefCell<Vec<(i32, Scratch)>> = const { RefCell::new(Vec::new()) };
}

fn raw(t: &GpuTensor, off: usize, len: usize) -> GpuTensor {
    GpuTensor {
        buf: unsafe { DeviceBuffer::from_raw((t.buf.as_ptr() as *mut u8).add(off) as *mut _, len) },
        shape: vec![len],
        dtype: DType::Raw,
    }
}

/// Pure admission by architecture, state format and live residual: exact gfx1201
/// ([`crate::scratch::qwen4_dense_gdn_admits`]), a plain Q8 state (scratch EF, no residual
/// tensor) or a Q8Ef state with its live residual. F32 and every other architecture retain
/// the incumbent recurrence.
fn admits_state(arch: &str, format: GdnStateFormat, has_residual: bool) -> bool {
    crate::scratch::qwen4_dense_gdn_admits(arch, format) && has_residual == format.has_residual()
}

/// Exact gfx1201 FN geometry (16 / 48 heads, or the TP2 shard's 8 / 24), Q8 or Q8Ef state (with
/// its live residual), no capture/recording, and the mseg segment geometry. Other steps retain
/// the incumbent recurrence.
pub fn applies(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    gpu.flags.fn_gdn_dense_scan
        && admits_state(&gpu.arch, p.format, p.residual.is_some())
        && p.row_states.is_none()
        && geometry_runs(
            p.key_heads,
            p.value_heads,
            p.key_dim,
            p.value_dim,
            p.qkv_width,
            gpu.flags.qwen4_gdn_chain_fuse_enabled(),
        )
        && p.projection.dtype == DType::F32
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && rows_admitted(p.rows)
}

fn launch(
    gpu: &mut Gpu,
    sym: &str,
    grid: [u32; 3],
    block: u32,
    args: &mut KernargBlob,
) -> HipResult<()> {
    // Uses the registered module/pack lookup, with the same compiler options
    // as tensor_ops (no chunk-scan-only floating-point flags).
    gpu.ensure_kernel("fn_gdn_dense", crate::kernels::FN_GDN_DENSE_SRC, sym)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        sym,
        grid,
        [block, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Fused convolution/parameters/packing, then KKT and mseg. Caller checked
/// [`applies`]; the input projection and convolution destination are distinct.
pub fn run(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    p: &GatedDeltaStepBatched<'_>,
) -> HipResult<()> {
    // Refuse before any launch: only an admitted format with its exact residual is run.
    if !admits_state(&gpu.arch, p.format, p.residual.is_some()) {
        return Err(hip_bridge::HipError::new(
            0,
            "fn_gdn_dense: GDN state format/residual not admitted by the dense scan",
        ));
    }
    crate::tensor_ops::validate_gdn_state(
        p.format,
        p.state,
        p.residual,
        None,
        p.rows,
        p.value_heads,
        p.key_dim,
        p.value_dim,
    )?;
    let geometry = Geometry::of(p.key_heads, p.value_heads)
        .filter(|_| {
            crate::scratch::qwen4_dense_gdn_geometry_admits(
                p.key_heads,
                p.value_heads,
                p.key_dim,
                p.value_dim,
                p.qkv_width,
            )
        })
        .ok_or_else(|| {
            hip_bridge::HipError::new(
                0,
                "fn_gdn_dense: head geometry not admitted by the dense scan",
            )
        })?;
    if conv.channels != geometry.channels() {
        return Err(hip_bridge::HipError::new(
            0,
            "fn_gdn_dense: convolution width is not the q | k | v width of the head geometry",
        ));
    }
    let n = p.rows;
    let n64 = n.div_ceil(64) * 64;
    // KKT clamps K loads to the last real row and guards G/beta; mseg clamps
    // Q/K/V loads, guards its controls, and only stores output for tok<rows.
    // Neither reads Q/K/V/G/beta pad rows. KKT DOES write a full 64-row A
    // block, so only A needs padded scratch. Q/K/V alias the dead F32 conv
    // destination (n64 * channels * 2 <= n * channels * 4 for n >= 32; below that they
    // get dedicated planes); the producer also zeros their pad rows. G aliases gate,
    // beta stays in its original plane, and BF16 scan output aliases the F32 recurrent
    // destination.
    let out = raw(p.output, 0, n * geometry.value_width() * 2);
    let mut slot = SCRATCH.with(|s| {
        let mut slots = s.borrow_mut();
        let index = slots
            .iter()
            .position(|(device, _)| *device == gpu.device_id)?;
        Some(slots.swap_remove(index).1)
    });
    if slot.as_ref().is_some_and(|s| s.rows < n64) {
        let s = slot.take().unwrap();
        gpu.free_tensor(s.a)?;
        if let Some(ef) = s.ef {
            gpu.free_tensor(ef)?;
        }
        if let Some(planes) = s.planes {
            gpu.free_tensor(planes)?;
        }
    }
    let mut s = match slot {
        Some(s) => s,
        None => Scratch {
            rows: n64,
            a: gpu.zeros(
                &[n64.checked_mul(A_ROW_BYTES).ok_or_else(|| {
                    hip_bridge::HipError::new(0, "fn_gdn_dense: A scratch extent overflow")
                })?],
                DType::Raw,
            )?,
            ef: None,
            planes: None,
        },
    };
    if edge_planes_needed(n) && s.planes.is_none() {
        match gpu.zeros(&[n64 * geometry.channels() * 2], DType::Raw) {
            Ok(planes) => s.planes = Some(planes),
            Err(error) => {
                SCRATCH.with(|c| c.borrow_mut().push((gpu.device_id, s)));
                return Err(error);
            }
        }
    }
    let (plane_src, plane_off) = match (edge_planes_needed(n), s.planes.as_ref()) {
        (true, Some(planes)) => (planes, 0),
        _ => (p.projection, 0),
    };
    let q_bytes = n64 * geometry.qk() * 2;
    let q = raw(plane_src, plane_off, q_bytes);
    let k = raw(plane_src, plane_off + q_bytes, q_bytes);
    let v = raw(
        plane_src,
        plane_off + q_bytes * 2,
        n64 * geometry.value_width() * 2,
    );
    if p.residual.is_none() && s.ef.is_none() {
        match gpu.zeros(&[EF_BYTES], DType::Raw) {
            Ok(ef) => s.ef = Some(ef),
            Err(error) => {
                SCRATCH.with(|c| c.borrow_mut().push((gpu.device_id, s)));
                return Err(error);
            }
        }
    }
    let result = (|| -> HipResult<()> {
        // Plain Q8: the cleared thread-local scratch (`ef_words` words zeroed by the
        // producer's block x=81). Q8Ef: the owner's live residual, which the producer must
        // not clear (`ef_words == 0`) and the scan reads on entry and rewrites on exit.
        let (ef_plane, ef_words): (&GpuTensor, usize) = match (p.residual, s.ef.as_ref()) {
            (Some(residual), _) => (residual, 0),
            (None, Some(ef)) => (ef, geometry.ef_bytes() / 4),
            (None, None) => {
                return Err(hip_bridge::HipError::new(
                    0,
                    "fn_gdn_dense: missing EF plane",
                ));
            }
        };
        let mut args = KernargBlob::new();
        for t in [
            conv.input,
            conv.kernel,
            conv.history,
            conv.next_history,
            &q,
            &k,
            &v,
            p.gate,
            p.beta,
            params.a,
            params.b,
            params.a_log,
            params.dt_bias,
        ] {
            args.push_ptr(t.buf.as_ptr());
        }
        args.push_ptr(ef_plane.buf.as_ptr());
        args.push_i32(n as i32);
        args.push_i32(n64 as i32);
        args.push_i32(conv.start_cursor as i32);
        args.push_i32(i32::from(conv.input.dtype == DType::BF16));
        args.push_f32((p.key_dim as f32).sqrt().recip());
        args.push_i32(ef_words as i32);
        // `HIPFIRE_QWEN4_GDN_CHAIN_FUSE`: the load-batched producer (same bytes), entry by
        // the projection's dtype; `=0` keeps the v1 entry (which reads either dtype). The
        // 8 / 24 shard geometry has only the load-batched entries (`applies` requires them).
        let chain_fuse = gpu.flags.qwen4_gdn_chain_fuse_enabled();
        if geometry.producer_entry(chain_fuse, false).is_none() {
            return Err(hip_bridge::HipError::new(
                0,
                "fn_gdn_dense: no producer entry for this head geometry",
            ));
        }
        // One literal per arm (the launcher ABI gate resolves the kernel name from them); the
        // check above admits only 16 / 48 (any entry) and 8 / 24 (load-batched entries).
        let producer = match (geometry.key_heads, chain_fuse, conv.input.dtype) {
            (16, false, _) => "fn_gdn_dense_producer",
            (16, true, DType::BF16) => "fn_gdn_dense_producer_v2_bf16",
            (16, true, _) => "fn_gdn_dense_producer_v2_f32",
            (_, _, DType::BF16) => "fn_gdn_dense_producer_v2_bf16_k8v24",
            (_, _, _) => "fn_gdn_dense_producer_v2_f32_k8v24",
        };
        launch(
            gpu,
            producer,
            [geometry.producer_blocks() as u32, (n64 / 32) as u32, 1],
            256,
            &mut args,
        )?;
        gpu.gdn_chunk_kkt_solve_batched(&k, p.gate, p.beta, &s.a, n, p.key_heads, p.value_heads)?;
        let codes = raw(p.state, 0, geometry.state_codes());
        let scales = raw(p.state, geometry.state_codes(), geometry.state_scales());
        gpu.gdn_chunk_scan_layer_mseg(
            &q,
            &k,
            &v,
            &s.a,
            p.gate,
            p.beta,
            &codes,
            &scales,
            ef_plane,
            &out,
            n,
            p.key_heads,
            p.value_heads,
        )
    })();
    SCRATCH.with(|c| c.borrow_mut().push((gpu.device_id, s)));
    result
}

/// Gate the packed BF16 scan output, optionally also producing the output
/// projection's Aligned256 rotation. No later reader needs recurrent_output.
pub fn gate(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    rotated: Option<&GpuTensor>,
) -> HipResult<()> {
    let elements = p.rows * p.value_heads * p.value_dim;
    let mut args = KernargBlob::new();
    for t in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(t.buf.as_ptr());
    }
    if let Some(rotated) = rotated {
        gpu.ensure_mq_signs()?;
        args.push_ptr(rotated.buf.as_ptr());
        args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
        args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
        launch(
            gpu,
            "fn_gdn_dense_gate_rotate_bf16in",
            [(elements / 256) as u32, 1, 1],
            64,
            &mut args,
        )?;
        gpu.scratch.prerotated = Some((
            p.output.buf.as_ptr() as usize,
            rotated.buf.as_ptr() as usize,
            elements,
        ));
        Ok(())
    } else {
        args.push_i32(p.rows as i32);
        args.push_i32(p.value_heads as i32);
        args.push_i32(p.value_dim as i32);
        launch(
            gpu,
            "fn_gdn_dense_gate_bf16in",
            [(elements / 128) as u32, 1, 1],
            32,
            &mut args,
        )
    }
}

/// Whether [`gate_i8`] replaces the gate + the MQ6 out projection's A8 producer
/// (`HIPFIRE_QWEN4_GDN_I8_FUSE`, default on for exact gfx1201): the 48 x 128 FN
/// value geometry (6144 columns = 24 whole 256-wide rotation groups of two heads) or the
/// 24 x 128 TP2 shard geometry (3072 columns = 12 groups), with the recurrent plane, `z` and
/// `norm` 16-byte aligned (the kernel loads eight elements per lane). The caller has checked
/// [`applies`] and the out projection's INT8 route ([`Gpu::gemm_mq6g256v2_i8_applies`], eager
/// only), so no recorder or capture state is consulted here.
pub fn gate_i8_applies(gpu: &Gpu, p: &GatedDeltaGateBatched<'_>) -> bool {
    gpu.flags.qwen4_gdn_i8_fuse_enabled()
        && p.rows > 0
        && gate_i8_entry(p.value_heads, p.value_dim).is_some()
        && [p.recurrent_output, p.z, p.norm]
            .iter()
            .all(|t| (t.buf.as_ptr() as usize).is_multiple_of(16))
}

/// Gated RMSNorm of the packed BF16 scan output fused with the out
/// projection's rotation and A8 quantization: one launch writes the
/// `block_i8_128` blocks (`[K/128][rows]`, K = `value_heads * value_dim`) that
/// `mq_rotate_x_i8_gfx12` would emit from the F32 gate output, bytewise, and
/// nothing else (`p.output` is not written). Returns the blocks' device pointer for
/// [`Gpu::gemm_mq6g256v2_i8_prequant`]; valid until the next A8 producer.
pub fn gate_i8(gpu: &mut Gpu, p: &GatedDeltaGateBatched<'_>) -> HipResult<*mut c_void> {
    if gate_i8_entry(p.value_heads, p.value_dim).is_none() {
        return Err(hip_bridge::HipError::new(
            0,
            "fn_gdn_dense: no gate_i8 entry for this value-head geometry",
        ));
    }
    let entry = match p.value_heads {
        24 => "fn_gdn_dense_gate_i8_p12",
        _ => "fn_gdn_dense_gate_i8",
    };
    let k = p.value_heads * p.value_dim;
    // The unfused producer flushes a deferred residual add before reading.
    gpu.flush_residual_fold()?;
    gpu.ensure_mq_signs()?;
    let reservation = gpu.reserve_int8_mmq(k, p.rows)?;
    let mut args = KernargBlob::new();
    for t in [p.recurrent_output, p.z, p.norm] {
        args.push_ptr(t.buf.as_ptr());
    }
    args.push_ptr(reservation.ptr());
    args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
    args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
    args.push_i32(p.rows as i32);
    launch(
        gpu,
        entry,
        // One wave per 256-wide group, two groups per workgroup (rows * (24 or 12) is even).
        [(p.rows * (k / 256) / 2) as u32, 1, 1],
        64,
        &mut args,
    )?;
    let prepared = crate::scratch::Int8MmqPrepared::from_reservation(reservation);
    gpu.int8_mmq_prepared_ptr(&prepared, k, p.rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_bytes_match_flash_next_4096() {
        assert_eq!(workspace_a_bytes(4096), Some(25_165_824));
        assert_eq!(workspace_ef_bytes(4096), 1_572_864);
        assert_eq!(workspace_bytes(4096), Some(26_738_688));
        // A rounds up to whole 64-row blocks; the route needs n >= 64.
        assert_eq!(workspace_a_bytes(65), workspace_a_bytes(128));
        assert_eq!(workspace_bytes(63), Some(0));
        assert_eq!(workspace_a_bytes(usize::MAX), None);
    }

    /// Dense A is charged for every admitted state; the 1.5 MiB thread-local EF scratch
    /// only for plain Q8 (Q8Ef scans against the owner's live residual, which this scratch
    /// census never charges); F32 never reaches the route.
    #[test]
    fn fn_gdn_dense_workspace_charges_by_format() {
        const A_4096: usize = 25_165_824;
        const EF: usize = 1_572_864;
        assert_eq!(EF_BYTES, EF);
        for (format, a, ef) in [
            (GdnStateFormat::Q8, A_4096, EF),
            (GdnStateFormat::Q8Ef, A_4096, 0),
            (GdnStateFormat::F32, 0, 0),
        ] {
            assert_eq!(workspace_a_bytes_for(format, 4096), Some(a), "{format:?} A");
            assert_eq!(workspace_ef_bytes_for(format, 4096), ef, "{format:?} EF");
            assert_eq!(
                workspace_bytes_for(format, 4096),
                Some(a + ef),
                "{format:?} total"
            );
            // Below the row floor the route never allocates, whatever the format.
            assert_eq!(
                workspace_bytes_for(format, MIN_ROWS - 1),
                Some(0),
                "{format:?} floor"
            );
        }
        assert_eq!(
            workspace_bytes(4096),
            workspace_bytes_for(GdnStateFormat::Q8, 4096)
        );
        // The residual tensor Q8Ef scans against is exactly the scratch plane's extent.
        assert_eq!(GdnStateFormat::Q8Ef.residual_bytes(48, 128, 128), EF_BYTES);
        for format in [
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
            GdnStateFormat::F32,
        ] {
            assert_eq!(
                workspace_a_bytes_for(format, usize::MAX),
                if format == GdnStateFormat::F32 {
                    Some(0)
                } else {
                    None
                }
            );
        }
    }

    /// `applies`' format/architecture/residual predicate: exact gfx1201, Q8 without a
    /// residual tensor, Q8Ef with one; never F32, never another architecture (gfx942 is
    /// F32-only), never a residual/format mismatch.
    #[test]
    fn fn_gdn_dense_admission_table() {
        use GdnStateFormat::{Q8Ef, F32, Q8};
        let arches = ["gfx1201", "gfx1151", "gfx1100", "gfx942", "gfx1200", ""];
        for arch in arches {
            for (format, residual, want) in [
                (Q8, false, true),
                (Q8, true, false),
                (Q8Ef, true, true),
                (Q8Ef, false, false),
                (F32, false, false),
                (F32, true, false),
            ] {
                assert_eq!(
                    admits_state(arch, format, residual),
                    arch == "gfx1201" && want,
                    "{arch} {format:?} residual={residual}"
                );
            }
        }
    }

    /// The pre-shard 16 / 48 geometry keeps every launch constant it had as a literal (10240
    /// channels, 82 producer blocks, the 1.5 MiB EF plane, the 48 x 128 x 128 Q8 codes), and the
    /// rank-local 8 / 24 shard geometry is exactly half of each.
    #[test]
    fn geometry_constants_pin_flash_next_and_halve_for_the_shard() {
        let full = Geometry::of(16, 48).expect("flash-next");
        assert_eq!(full.qk(), 2048);
        assert_eq!(full.value_width(), 6144);
        assert_eq!(full.channels(), 10240);
        assert_eq!(full.producer_blocks(), 82);
        assert_eq!(full.ef_bytes(), EF_BYTES);
        assert_eq!(full.state_codes(), 48 * 128 * 128);
        assert_eq!(full.state_scales(), 48 * 128 * 4);
        assert_eq!(
            GdnStateFormat::Q8.state_bytes(48, 128, 128),
            full.state_codes() + full.state_scales()
        );
        let half = Geometry::of(8, 24).expect("rank-local shard");
        assert_eq!(half.qk(), 1024);
        assert_eq!(half.value_width(), 3072);
        assert_eq!(half.channels(), 5120);
        assert_eq!(half.producer_blocks(), 42);
        assert_eq!(half.ef_bytes() * 2, full.ef_bytes());
        assert_eq!(half.state_codes() * 2, full.state_codes());
        assert_eq!(half.state_scales() * 2, full.state_scales());
        assert_eq!(
            GdnStateFormat::Q8.state_bytes(24, 128, 128),
            half.state_codes() + half.state_scales()
        );
        assert_eq!(half.channels() * 2, full.channels());
        for (kh, vh) in [(8, 48), (16, 24), (4, 12), (32, 96), (16, 32), (0, 0)] {
            assert_eq!(Geometry::of(kh, vh), None, "{kh}/{vh}");
        }
    }

    /// The thread-local scratch is sized at the 48-head upper bound the census charges: every
    /// admitted geometry's EF plane and A rows fit it.
    #[test]
    fn scratch_upper_bound_covers_every_admitted_geometry() {
        for (key_heads, value_heads) in crate::scratch::QWEN4_DENSE_GDN_GEOMETRIES {
            let g = Geometry::of(key_heads, value_heads).expect("admitted");
            assert!(g.ef_bytes() <= EF_BYTES, "{key_heads}/{value_heads}");
            assert!(
                value_heads * 64 * 2 <= A_ROW_BYTES,
                "{key_heads}/{value_heads}"
            );
            assert_eq!(g.ef_bytes() % 16, 0, "the EF zero block stores whole uint4");
        }
    }

    /// Producer entry by (geometry, chain fusion, input dtype): the three pre-shard entries are
    /// unchanged; the shard geometry has the two load-batched entries only.
    #[test]
    fn producer_entries_by_geometry() {
        let full = Geometry::of(16, 48).unwrap();
        assert_eq!(
            full.producer_entry(false, false),
            Some("fn_gdn_dense_producer")
        );
        assert_eq!(
            full.producer_entry(false, true),
            Some("fn_gdn_dense_producer")
        );
        assert_eq!(
            full.producer_entry(true, false),
            Some("fn_gdn_dense_producer_v2_f32")
        );
        assert_eq!(
            full.producer_entry(true, true),
            Some("fn_gdn_dense_producer_v2_bf16")
        );
        let half = Geometry::of(8, 24).unwrap();
        assert_eq!(half.producer_entry(false, false), None);
        assert_eq!(half.producer_entry(false, true), None);
        assert_eq!(
            half.producer_entry(true, false),
            Some("fn_gdn_dense_producer_v2_f32_k8v24")
        );
        assert_eq!(
            half.producer_entry(true, true),
            Some("fn_gdn_dense_producer_v2_bf16_k8v24")
        );
    }

    /// `applies`' geometry part: the 16 / 48 pin (10240 channels) holds with the chain fusion on
    /// or off, the 8 / 24 shard geometry (5120 channels) only with it on, nothing else runs.
    #[test]
    fn geometry_admission_pins_old_results_and_adds_the_shard_shape() {
        for chain_fuse in [false, true] {
            assert!(geometry_runs(16, 48, 128, 128, 10240, chain_fuse));
            assert_eq!(
                geometry_runs(8, 24, 128, 128, 5120, chain_fuse),
                chain_fuse,
                "the shard geometry has no v1 producer"
            );
            for (kh, vh, kd, vd, qkv) in [
                (16, 48, 128, 128, 5120),
                (8, 24, 128, 128, 10240),
                (16, 32, 128, 128, 8192),
                (16, 48, 64, 128, 10240),
                (16, 48, 128, 64, 10240),
                (8, 48, 128, 128, 8192),
            ] {
                assert!(!geometry_runs(kh, vh, kd, vd, qkv, chain_fuse));
            }
        }
    }

    /// A short final segment must not change the algorithm for all preceding
    /// rows of an outer staging chunk.
    #[test]
    fn row_admission_is_the_mseg_segment_geometry() {
        for (rows, want) in [
            (0, false),
            (63, false),
            (64, true),
            (65, true),
            (511, true),
            (512, true),
            (513, true),
            (575, true),
            (576, true),
            (1024, true),
            (1025, true),
            (4096, true),
            (4096 + 63, true),
            (4096 + 64, true),
        ] {
            assert_eq!(rows_admitted(rows), want, "{rows} rows");
        }
    }

    /// The fused gate + A8 producer: 48 value heads (24 pairs per row) as before, 24 value
    /// heads (12 pairs) for the shard; a head width other than 128 or any other head count has no
    /// entry (the unfused gate + `mq_rotate_x_i8_gfx12` chain runs).
    #[test]
    fn gate_i8_entries_by_value_heads() {
        assert_eq!(gate_i8_entry(48, 128), Some("fn_gdn_dense_gate_i8"));
        assert_eq!(gate_i8_entry(24, 128), Some("fn_gdn_dense_gate_i8_p12"));
        for (heads, dim) in [
            (48, 64),
            (24, 64),
            (32, 128),
            (12, 128),
            (0, 128),
            (16, 128),
        ] {
            assert_eq!(gate_i8_entry(heads, dim), None, "{heads} x {dim}");
        }
        // Whole 256-wide rotation groups of head pairs, an even number of groups per launch row
        // pair: the launch grid `rows * pairs / 2` is exact for every row count.
        for heads in [48usize, 24] {
            assert_eq!(heads * 128 % 256, 0);
            assert_eq!((heads / 2) % 2, 0);
        }
    }
}
