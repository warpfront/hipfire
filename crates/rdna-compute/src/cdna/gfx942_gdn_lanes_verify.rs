// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The exact-gfx942 GDN Core of a whole lane group in three launches per layer.
//!
//! A verify lane's GDN Core is `copy_d2d(projection -> capture inputs)`, convolution + gate
//! parameters, the captured recurrence, the gated norm; the one-row AR lane's is the same chain
//! without the copy and with the state updated in place. The kernels of
//! `kernels/src/qwen4_g942_gdn_lanes_verify.gfx942.hip` run each step for every lane of a group
//! in ONE launch, each lane on its own pointers (a device lane table, content-keyed and cached)
//! and with its own row count (4 bits per lane in the launch arguments, so a window whose lane
//! depths change never misses the table). A lane writes only its own bytes, so every output and
//! state byte equals the per-lane chain's (Tier C:
//! `examples/gdn_lanes_verify_f32_gfx942_oracle.rs`).

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use super::{Gfx942Device, GDN_LANES_TABLE_CACHE_MAX};
use crate::{DType, GpuTensor};

pub(super) const QWEN4_G942_GDN_LANES_VERIFY_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_g942_gdn_lanes_verify.gfx942.hip");
/// Module key of [`QWEN4_G942_GDN_LANES_VERIFY_SRC`] (its own cache key and twin seat).
pub const QWEN4_G942_GDN_LANES_VERIFY_MODULE: &str = "qwen4_g942_gdn_lanes_verify";
/// Convolution + gate parameters (+ the capture-input copy) of every lane.
pub const QWEN4_G942_GDN_LANES_CONV_PARAMS: &str = "qwen4_g942_gdn_lanes_conv_params_k4";
/// The (captured) recurrence of every lane, ragged rows.
pub const QWEN4_G942_GDN_LANES_STEP: &str = "qwen4_g942_gdn_lanes_step_f32";
/// The gated norm of every lane.
pub const QWEN4_G942_GDN_LANES_GATE: &str = "qwen4_g942_gdn_lanes_gate_batched";
/// Most lanes one launch covers: the per-lane row counts and conv cursors travel as 4 bits a
/// lane in one 32-bit launch argument each.
pub const QWEN4_G942_GDN_LANES_VERIFY_MAX: usize = 8;
/// Most rows one lane covers (the 4-bit field; the verify window of the lane steps is 8).
pub const QWEN4_G942_GDN_LANES_VERIFY_ROWS: usize = 8;

/// Bind `kernel` of the module: the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE=/absolute/device.elf` names one, that explicitly
/// supplied code object (the Route-A twin proof: same module, same entry names, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_gdn_lanes_verify_ensure(gpu: &mut crate::Gpu, kernel: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_GDN_LANES_VERIFY_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_G942_GDN_LANES_VERIFY_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                kernel,
            )
        }
        _ => gpu.ensure_kernel(
            QWEN4_G942_GDN_LANES_VERIFY_MODULE,
            QWEN4_G942_GDN_LANES_VERIFY_SRC,
            kernel,
        ),
    }
}

/// One lane of the device lane table (`struct Qwen4GdnLaneVerifyDesc` in the kernel file): the
/// seventeen pointers of one lane's Core chain.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Qwen4GdnLaneVerifyDesc {
    projection: u64,
    inputs: u64,
    kernel: u64,
    history: u64,
    conv_output: u64,
    a: u64,
    b: u64,
    a_log: u64,
    dt_bias: u64,
    gate: u64,
    beta: u64,
    state: u64,
    final_state: u64,
    recurrent_output: u64,
    z: u64,
    norm: u64,
    gdn_output: u64,
}
const _: () = assert!(std::mem::size_of::<Qwen4GdnLaneVerifyDesc>() == 136);

impl Qwen4GdnLaneVerifyDesc {
    fn bytes(&self) -> impl Iterator<Item = u8> {
        [
            self.projection,
            self.inputs,
            self.kernel,
            self.history,
            self.conv_output,
            self.a,
            self.b,
            self.a_log,
            self.dt_bias,
            self.gate,
            self.beta,
            self.state,
            self.final_state,
            self.recurrent_output,
            self.z,
            self.norm,
            self.gdn_output,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes)
    }
}

/// `[key_heads, value_heads]` and the lane descriptors a cached table was built from.
pub(crate) type GdnLanesVerifyTableKey = ([usize; 2], Vec<Qwen4GdnLaneVerifyDesc>);

/// One lane of a [`Gfx942Device::qwen4_gdn_lanes_verify`] launch: the tensors that lane's
/// per-lane Core chain names (`copy_d2d`, `gated_delta_conv_params_batched`,
/// `gated_delta_step_batched`, `gated_delta_gate_batched`).
pub struct Qwen4GdnLanesVerifyLane<'a> {
    /// F32 `[rows, qkv]` qkv in-projection rows (q | k | v per row), read only.
    pub projection: &'a GpuTensor,
    /// The capture's retained convolution input (F32 `[rows, qkv]`) the projection rows are
    /// copied to; `None` when the projection already is that tensor (or the lane has none).
    pub inputs: Option<&'a GpuTensor>,
    /// BF16 `[qkv, 4]` convolution weights.
    pub kernel: &'a GpuTensor,
    /// F32 `[3, qkv]` convolution ring, updated in place.
    pub history: &'a GpuTensor,
    /// F32 `[rows, qkv]` convolution output (the recurrence's projection).
    pub conv_output: &'a GpuTensor,
    /// F32 `[rows, value_heads]` raw `in_proj_a` / `in_proj_b` rows.
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    /// BF16 `[value_heads]` parameters.
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    /// F32 `[rows, value_heads]` decay gate and beta (the recurrence input).
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    /// F32 `[value_heads, 128, 128]` recurrent state.
    pub state: &'a GpuTensor,
    /// The lane's capture ring (F32 slots of the state's size, at least `rows`): the state
    /// after the last row lands in slot `rows - 1` and `state` is only read. `None`: the AR
    /// lane, `state` is updated in place.
    pub ring: Option<&'a GpuTensor>,
    /// F32 `[rows, value_heads * 128]` recurrent output, gated in place by the norm.
    pub recurrent_output: &'a GpuTensor,
    /// F32 `[rows, value_heads * 128]` z rows.
    pub z: &'a GpuTensor,
    /// BF16 `[128]` gated-norm weight.
    pub norm: &'a GpuTensor,
    /// F32 `[rows, value_heads * 128]` gated output.
    pub gdn_output: &'a GpuTensor,
    /// This lane's rows (`1..=`[`QWEN4_G942_GDN_LANES_VERIFY_ROWS`]).
    pub rows: usize,
    /// The convolution ring cursor of the lane's first row (`position % 3`).
    pub start_cursor: usize,
}

/// Shape shared by every lane of one [`Gfx942Device::qwen4_gdn_lanes_verify`] launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4GdnLanesVerifyShape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel: usize,
}

/// A validated group: the lane descriptors and the launch geometry.
struct Prepared {
    key: GdnLanesVerifyTableKey,
    shape: Qwen4GdnLanesVerifyShape,
    qkv: usize,
    value: usize,
    state_elems: usize,
    max_rows: usize,
    rows_packed: u32,
    cursor_packed: u32,
    total_rows: usize,
}

/// Every refusal of [`Gfx942Device::qwen4_gdn_lanes_verify`], without launching (and without a
/// device): the route's admission.
fn prepare(
    lanes: &[Qwen4GdnLanesVerifyLane<'_>],
    shape: Qwen4GdnLanesVerifyShape,
) -> HipResult<Prepared> {
    const WHAT: &str = "qwen4_gdn_lanes_verify";
    let refuse = |msg: String| Err(HipError::new(0, &msg));
    let Qwen4GdnLanesVerifyShape {
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        conv_kernel,
    } = shape;
    if lanes.is_empty() || lanes.len() > QWEN4_G942_GDN_LANES_VERIFY_MAX {
        return refuse(format!(
            "{WHAT}: {} lanes (1..={QWEN4_G942_GDN_LANES_VERIFY_MAX})",
            lanes.len()
        ));
    }
    if key_heads == 0
        || value_heads == 0
        || value_heads % key_heads != 0
        || key_dim != 128
        || value_dim != 128
        || conv_kernel != 4
    {
        return refuse(format!(
                "{WHAT}: needs the exact 128x128 GDN shape with the k4 convolution (key_heads={key_heads} \
                 value_heads={value_heads} key_dim={key_dim} value_dim={value_dim} conv_kernel={conv_kernel})"
            ));
    }
    let value = value_heads * 128;
    let qkv = 2 * key_heads * 128 + value;
    let state_elems = value * 128;
    let mut rows_packed = 0u32;
    let mut cursor_packed = 0u32;
    let mut max_rows = 0usize;
    let mut descs = Vec::with_capacity(lanes.len());
    for (i, lane) in lanes.iter().enumerate() {
        let rows = lane.rows;
        if !(1..=QWEN4_G942_GDN_LANES_VERIFY_ROWS).contains(&rows) || lane.start_cursor >= 3 {
            return refuse(format!(
                "{WHAT}: lane {i} rows={rows} cursor={} (1..={QWEN4_G942_GDN_LANES_VERIFY_ROWS} \
                     rows, cursor < 3)",
                lane.start_cursor
            ));
        }
        rows_packed |= (rows as u32) << (4 * i);
        cursor_packed |= (lane.start_cursor as u32) << (4 * i);
        max_rows = max_rows.max(rows);
        let heads = rows * value_heads;
        let mut checks: Vec<(&str, &GpuTensor, DType, usize, bool)> = vec![
            ("projection", lane.projection, DType::F32, rows * qkv, true),
            ("kernel", lane.kernel, DType::BF16, qkv * 4, true),
            ("history", lane.history, DType::F32, qkv * 3, true),
            (
                "conv_output",
                lane.conv_output,
                DType::F32,
                rows * qkv,
                true,
            ),
            ("a", lane.a, DType::F32, heads, true),
            ("b", lane.b, DType::F32, heads, true),
            ("a_log", lane.a_log, DType::BF16, value_heads, true),
            ("dt_bias", lane.dt_bias, DType::BF16, value_heads, true),
            ("gate", lane.gate, DType::F32, heads, true),
            ("beta", lane.beta, DType::F32, heads, true),
            ("state", lane.state, DType::F32, state_elems, true),
            (
                "recurrent_output",
                lane.recurrent_output,
                DType::F32,
                rows * value,
                true,
            ),
            ("z", lane.z, DType::F32, rows * value, true),
            ("norm", lane.norm, DType::BF16, 128, true),
            (
                "gdn_output",
                lane.gdn_output,
                DType::F32,
                rows * value,
                true,
            ),
        ];
        if let Some(inputs) = lane.inputs {
            checks.push(("inputs", inputs, DType::F32, rows * qkv, true));
        }
        if let Some(ring) = lane.ring {
            // The ring only has to HOLD `rows` slots (a compact capture view may be larger).
            checks.push(("ring", ring, DType::F32, rows * state_elems, false));
        }
        for (name, tensor, dtype, elems, exact) in checks {
            let extent_ok = if exact {
                tensor.numel() == elems
            } else {
                tensor.numel() >= elems
            };
            if tensor.dtype != dtype || !extent_ok || tensor.buf.size() < elems * dtype.size() {
                return refuse(format!(
                    "{WHAT}: lane {i} {name} must be {dtype:?} with {}{elems} elements (got {:?}, \
                         {} elements, {} bytes)",
                    if exact { "" } else { "at least " },
                    tensor.dtype,
                    tensor.numel(),
                    tensor.buf.size()
                ));
            }
        }
        let ptr = |t: &GpuTensor| t.buf.as_ptr() as u64;
        descs.push(Qwen4GdnLaneVerifyDesc {
            projection: ptr(lane.projection),
            inputs: lane.inputs.map_or(0, ptr),
            kernel: ptr(lane.kernel),
            history: ptr(lane.history),
            conv_output: ptr(lane.conv_output),
            a: ptr(lane.a),
            b: ptr(lane.b),
            a_log: ptr(lane.a_log),
            dt_bias: ptr(lane.dt_bias),
            gate: ptr(lane.gate),
            beta: ptr(lane.beta),
            state: ptr(lane.state),
            final_state: lane.ring.map_or(ptr(lane.state), |ring| {
                ptr(ring) + ((rows - 1) * state_elems * 4) as u64
            }),
            recurrent_output: ptr(lane.recurrent_output),
            z: ptr(lane.z),
            norm: ptr(lane.norm),
            gdn_output: ptr(lane.gdn_output),
        });
    }
    // The table is a pure function of the lanes' pointers; the row counts and cursors ride
    // in the launch arguments, so a window whose lane depths change reuses it. (The final-state
    // slot pointer moves with `rows`, so a depth change of a captured lane is a new key.)
    let key: GdnLanesVerifyTableKey = ([key_heads, value_heads], descs);
    let refused_overlap = {
        // Validated on every call: the row counts are not in the key.
        let mut writes: Vec<(u64, u64)> = Vec::with_capacity(lanes.len() * 8);
        let mut reads: Vec<(u64, u64)> = Vec::with_capacity(lanes.len() * 9);
        for (lane, d) in lanes.iter().zip(&key.1) {
            let rows = lane.rows as u64;
            let (qkv, heads, value, state) = (
                qkv as u64 * 4,
                rows * value_heads as u64 * 4,
                rows * value as u64 * 4,
                state_elems as u64 * 4,
            );
            writes.push((d.history, d.history + 3 * qkv));
            writes.push((d.conv_output, d.conv_output + rows * qkv));
            writes.push((d.gate, d.gate + heads));
            writes.push((d.beta, d.beta + heads));
            writes.push((d.final_state, d.final_state + state));
            writes.push((d.recurrent_output, d.recurrent_output + value));
            writes.push((d.gdn_output, d.gdn_output + value));
            if d.inputs != 0 {
                writes.push((d.inputs, d.inputs + rows * qkv));
            }
            reads.push((d.projection, d.projection + rows * qkv));
            reads.push((d.kernel, d.kernel + 2 * qkv));
            reads.push((d.a, d.a + heads));
            reads.push((d.b, d.b + heads));
            reads.push((d.a_log, d.a_log + value_heads as u64 * 2));
            reads.push((d.dt_bias, d.dt_bias + value_heads as u64 * 2));
            reads.push((d.z, d.z + value));
            reads.push((d.norm, d.norm + 256));
            if d.final_state != d.state {
                reads.push((d.state, d.state + state));
            }
        }
        writes.sort_unstable();
        writes.windows(2).any(|w| w[0].1 > w[1].0) || {
            let hits_write = |(start, end): (u64, u64)| {
                // The last write starting before `end` is the only one that can reach back
                // over `start` (the writes are disjoint and sorted).
                let at = writes.partition_point(|w| w.0 < end);
                at > 0 && writes[at - 1].1 > start
            };
            reads.into_iter().any(hits_write)
        }
    };
    if refused_overlap {
        return refuse(format!(
            "{WHAT}: two lanes write overlapping ranges, or a lane reads a range a lane writes"
        ));
    }
    Ok(Prepared {
        key,
        shape,
        qkv,
        value,
        state_elems,
        max_rows,
        rows_packed,
        cursor_packed,
        total_rows: lanes.iter().map(|lane| lane.rows).sum(),
    })
}

/// Whether [`Gfx942Device::qwen4_gdn_lanes_verify`] takes this group (the lane count, shape,
/// every dtype / extent and the write-overlap rule): `Ok` means the launch's refusals have all
/// passed.
pub fn qwen4_gdn_lanes_verify_admitted(
    lanes: &[Qwen4GdnLanesVerifyLane<'_>],
    shape: Qwen4GdnLanesVerifyShape,
) -> HipResult<()> {
    prepare(lanes, shape).map(|_| ())
}

impl Gfx942Device<'_> {
    /// The GDN Core chain of up to [`QWEN4_G942_GDN_LANES_VERIFY_MAX`] lanes in three launches:
    /// [`QWEN4_G942_GDN_LANES_CONV_PARAMS`] (grid `[ceil(qkv/256) + ceil(max_rows*heads/256),
    /// lanes]`), [`QWEN4_G942_GDN_LANES_STEP`] (`[value_heads, lanes]`, block 128) and
    /// [`QWEN4_G942_GDN_LANES_GATE`] (`[max_rows * value_heads, lanes]`, block 32). Lane `i`
    /// ends in exactly the bytes its own per-lane chain leaves (see the kernel file).
    ///
    /// Refused (`Err`, nothing launched) unless every tensor is the dtype and extent the
    /// per-lane launches validate, the shape is the exact 128 x 128 GDN one with the k4
    /// convolution, and no written range (capture input, convolution ring and output, gate,
    /// beta, final state, recurrent and gated output) overlaps another written range or any
    /// read range (projection, weights, a / b, parameters, a capture lane's state, z) of the
    /// group; during hipGraph capture the table must already be cached.
    pub fn qwen4_gdn_lanes_verify(
        &mut self,
        lanes: &[Qwen4GdnLanesVerifyLane<'_>],
        shape: Qwen4GdnLanesVerifyShape,
    ) -> HipResult<()> {
        const WHAT: &str = "qwen4_gdn_lanes_verify";
        let refuse = |msg: String| Err(HipError::new(0, &msg));
        let Prepared {
            key,
            shape:
                Qwen4GdnLanesVerifyShape {
                    key_heads,
                    value_heads,
                    key_dim,
                    value_dim,
                    conv_kernel,
                },
            qkv,
            value,
            state_elems,
            max_rows,
            rows_packed,
            cursor_packed,
            total_rows,
        } = prepare(lanes, shape)?;
        let channel_blocks = qkv.div_ceil(256);
        if !self
            .gpu
            .gdn_lanes_f32_tables
            .verify_tables
            .contains_key(&key)
        {
            if self.gpu.graphs.capture_mode {
                return refuse(format!("{WHAT}: lane table staging during graph capture"));
            }
            self.gpu.bind_thread()?;
            if self.gpu.gdn_lanes_f32_tables.verify_tables.len() >= GDN_LANES_TABLE_CACHE_MAX {
                // Queued launches may still read a table: drain before freeing any.
                self.gpu.hip.device_synchronize()?;
                self.gpu.gdn_lanes_f32_tables.verify_tables.clear();
            }
            let bytes: Vec<u8> = key
                .1
                .iter()
                .flat_map(Qwen4GdnLaneVerifyDesc::bytes)
                .collect();
            let buf = self.gpu.hip.malloc(bytes.len().next_multiple_of(256))?;
            // Stream-ordered before this launch; `bytes` lives in the cache entry, so the
            // (possibly staged) copy never reads a dropped source and nothing synchronizes.
            match self.gpu.active_stream.as_ref() {
                Some(stream) => self.gpu.hip.memcpy_htod_async(&buf, &bytes, stream)?,
                None => self.gpu.hip.memcpy_htod(&buf, &bytes)?,
            }
            self.gpu
                .gdn_lanes_f32_tables
                .verify_tables
                .insert(key.clone(), (buf, bytes));
        }
        let table = self.gpu.gdn_lanes_f32_tables.verify_tables[&key].0.as_ptr();
        self.gpu.bind_thread()?;
        for kernel in [
            QWEN4_G942_GDN_LANES_CONV_PARAMS,
            QWEN4_G942_GDN_LANES_STEP,
            QWEN4_G942_GDN_LANES_GATE,
        ] {
            g942_gdn_lanes_verify_ensure(&mut *self.gpu, kernel)?;
        }
        let lane_count = lanes.len() as u32;
        let (channels, history_rows, kernel_size, heads) =
            (qkv as i32, 3i32, conv_kernel as i32, value_heads as i32);
        let (qkv_i, kh, vh, kd, vd) = (
            qkv as i32,
            key_heads as i32,
            value_heads as i32,
            key_dim as i32,
            value_dim as i32,
        );
        let scale = (key_dim as f32).sqrt().recip();
        let bytes = total_rows * (qkv * 4 * 3 + value * 4 * 5) + lanes.len() * 2 * state_elems * 4;

        // Convolution + parameters (+ the capture-input copy).
        {
            let mut params: Vec<*mut c_void> = vec![
                &table as *const _ as *mut c_void,
                &rows_packed as *const _ as *mut c_void,
                &cursor_packed as *const _ as *mut c_void,
                &channels as *const _ as *mut c_void,
                &history_rows as *const _ as *mut c_void,
                &kernel_size as *const _ as *mut c_void,
                &heads as *const _ as *mut c_void,
            ];
            let timer = crate::profile::begin_timer(
                &self.gpu.hip,
                "deltanet",
                QWEN4_G942_GDN_LANES_CONV_PARAMS,
                bytes,
            );
            let result = self.gpu.launch_maybe_blob(
                QWEN4_G942_GDN_LANES_CONV_PARAMS,
                [
                    (channel_blocks + (max_rows * value_heads).div_ceil(256)) as u32,
                    lane_count,
                    1,
                ],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut blob = KernargBlob::new();
                    blob.push_ptr(table);
                    blob.push_u32(rows_packed);
                    blob.push_u32(cursor_packed);
                    blob.push_i32(channels);
                    blob.push_i32(history_rows);
                    blob.push_i32(kernel_size);
                    blob.push_i32(heads);
                    blob
                },
            );
            if let Some(timer) = timer {
                timer.finish(&self.gpu.hip);
            }
            result?;
        }
        // The recurrence.
        {
            let mut params: Vec<*mut c_void> = vec![
                &table as *const _ as *mut c_void,
                &rows_packed as *const _ as *mut c_void,
                &qkv_i as *const _ as *mut c_void,
                &kh as *const _ as *mut c_void,
                &vh as *const _ as *mut c_void,
                &kd as *const _ as *mut c_void,
                &vd as *const _ as *mut c_void,
                &scale as *const _ as *mut c_void,
            ];
            let timer = crate::profile::begin_timer(
                &self.gpu.hip,
                "deltanet",
                QWEN4_G942_GDN_LANES_STEP,
                bytes,
            );
            let result = self.gpu.launch_maybe_blob(
                QWEN4_G942_GDN_LANES_STEP,
                [value_heads as u32, lane_count, 1],
                [128, 1, 1],
                0,
                &mut params,
                || {
                    let mut blob = KernargBlob::new();
                    blob.push_ptr(table);
                    blob.push_u32(rows_packed);
                    blob.push_i32(qkv_i);
                    blob.push_i32(kh);
                    blob.push_i32(vh);
                    blob.push_i32(kd);
                    blob.push_i32(vd);
                    blob.push_f32(scale);
                    blob
                },
            );
            if let Some(timer) = timer {
                timer.finish(&self.gpu.hip);
            }
            result?;
        }
        // The gated norm.
        {
            let mut params: Vec<*mut c_void> = vec![
                &table as *const _ as *mut c_void,
                &rows_packed as *const _ as *mut c_void,
                &vh as *const _ as *mut c_void,
                &vd as *const _ as *mut c_void,
            ];
            let timer = crate::profile::begin_timer(
                &self.gpu.hip,
                "deltanet",
                QWEN4_G942_GDN_LANES_GATE,
                bytes,
            );
            let result = self.gpu.launch_maybe_blob(
                QWEN4_G942_GDN_LANES_GATE,
                [(max_rows * value_heads) as u32, lane_count, 1],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut blob = KernargBlob::new();
                    blob.push_ptr(table);
                    blob.push_u32(rows_packed);
                    blob.push_i32(vh);
                    blob.push_i32(vd);
                    blob
                },
            );
            if let Some(timer) = timer {
                timer.finish(&self.gpu.hip);
            }
            result?;
        }
        Ok(())
    }
}
