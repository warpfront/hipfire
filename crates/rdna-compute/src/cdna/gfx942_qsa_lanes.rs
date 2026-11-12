// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Batched QSA decode core of many lanes on exact gfx942 (`HIPFIRE_QWEN4_QSA_LANES`).
//!
//! A lane's QSA Core on gfx942 is five launches (decode prologue, incremental pool, row-8
//! score, select-from-scores with the persistent-selection mirror, per-head attention). The
//! kernels of `kernels/src/qwen4_qsa_lanes_f32.gfx942.hip` run each of them for every lane in
//! ONE launch: the production device template of that launch with the lane on `blockIdx.z`,
//! the lane's own state / scratch pointers from a device lane table, and its position and
//! rows from a by-value per-step table. A lane writes only its own bytes, so every output and
//! state byte equals the per-lane launches' (Tier C: `examples/qsa_lanes_gfx942_oracle.rs`).
//!
//! The admission (which per-lane route a lane would take, extents, aliasing) lives in
//! `tensor_ops::qsa_core_lanes_gfx942`, a child of `tensor_ops` so it shares the per-lane
//! launchers' private validators; it builds a [`Qwen4QsaLanesPlan`] of plain data that
//! [`Gfx942Device::qwen4_qsa_lanes_core`] launches.

use std::collections::HashMap;

use hip_bridge::{DeviceBuffer, HipError, HipResult, KernargBlob};

use super::Gfx942Device;
use crate::rope::RopeYarn;
use crate::tensor_ops::QsaKvFormat;

pub(super) const QWEN4_QSA_LANES_SRC: &str =
    include_str!("../../../../kernels/src/qwen4_qsa_lanes_f32.gfx942.hip");
/// Module key of [`QWEN4_QSA_LANES_SRC`].
pub const QWEN4_QSA_LANES_MODULE: &str = "qwen4_qsa_lanes_f32_gfx942";
pub const QWEN4_QSA_LANES_PROLOGUE_F32: &str = "qwen4_qsa_lanes_prologue_f32_gfx942";
pub const QWEN4_QSA_LANES_PROLOGUE_FP8_FNUZ: &str = "qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942";
pub const QWEN4_QSA_LANES_POOL: &str = "qwen4_qsa_lanes_pool_f32_gfx942";
pub const QWEN4_QSA_LANES_SCORES: &str = "qwen4_qsa_lanes_scores_rows8_f32_gfx942";
pub const QWEN4_QSA_LANES_SELECT: &str = "qwen4_qsa_lanes_select_from_scores_gfx942";
pub const QWEN4_QSA_LANES_ATTENTION_F32: &str = "qwen4_qsa_lanes_attention_f32_gfx942";
pub const QWEN4_QSA_LANES_ATTENTION_FP8_FNUZ: &str = "qwen4_qsa_lanes_attention_fp8_fnuz_gfx942";

/// Bind `kernel` of the QSA lanes module: the hipcc source (the anchor object), or, only when
/// `HIPFIRE_QWEN4_G942_QSA_LANES_IMAGE=/absolute/device.elf` names one, that explicitly supplied
/// code object (the Route-A twin proof: same module, same entry names, loaded through
/// `ensure_embedded_kernel`). Unset: the source, unchanged.
fn g942_qsa_lanes_ensure(gpu: &mut crate::Gpu, kernel: &str) -> HipResult<()> {
    match hipfire_config::developer_var("HIPFIRE_QWEN4_G942_QSA_LANES_IMAGE") {
        Ok(path) if !path.is_empty() => {
            let bytes = std::fs::read(&path).map_err(|error| {
                HipError::new(
                    0,
                    &format!("HIPFIRE_QWEN4_G942_QSA_LANES_IMAGE {path}: {error}"),
                )
            })?;
            gpu.ensure_embedded_kernel(
                QWEN4_QSA_LANES_MODULE,
                std::sync::Arc::<[u8]>::from(bytes),
                kernel,
            )
        }
        _ => gpu.ensure_kernel(QWEN4_QSA_LANES_MODULE, QWEN4_QSA_LANES_SRC, kernel),
    }
}

/// Most lanes one launch covers (`QWEN4_QSA_LANES_MAX` in the kernel file: the by-value
/// per-step table is 8 bytes a lane inside the 4 KiB kernarg limit).
pub const QWEN4_QSA_LANES_MAX: usize = 128;
/// Most rows one lane covers (the verify window of the lane steps, `LANE_MAX_ROWS`; the
/// rows8 score tile and the few-row decode limit).
pub const QWEN4_QSA_LANE_ROWS: usize = 8;
/// Lane tables the cache keeps before it is dropped whole (12 QSA layers x a few lane sets).
const QSA_LANES_TABLE_CACHE_MAX: usize = 2048;

/// One lane of the device lane table (`struct Qwen4QsaLaneDesc` in the kernel file): the
/// eleven pointers of one lane's QSA Core.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Qwen4QsaLaneDesc {
    pub(crate) full_keys: u64,
    pub(crate) full_values: u64,
    pub(crate) raw_idx: u64,
    pub(crate) pooled: u64,
    pub(crate) sel_persist: u64,
    pub(crate) index_row: u64,
    pub(crate) qgate: u64,
    pub(crate) keys: u64,
    pub(crate) values: u64,
    pub(crate) selected: u64,
    pub(crate) output: u64,
}
const _: () = assert!(std::mem::size_of::<Qwen4QsaLaneDesc>() == 88);

impl Qwen4QsaLaneDesc {
    fn bytes(descs: &[Self]) -> Vec<u8> {
        descs
            .iter()
            .flat_map(|d| {
                [
                    d.full_keys,
                    d.full_values,
                    d.raw_idx,
                    d.pooled,
                    d.sel_persist,
                    d.index_row,
                    d.qgate,
                    d.keys,
                    d.values,
                    d.selected,
                    d.output,
                ]
                .into_iter()
                .flat_map(u64::to_le_bytes)
            })
            .collect()
    }
}

/// Device lane tables of [`Gfx942Device::qwen4_qsa_lanes_core`], keyed by content (every
/// lane's eleven pointers): a table is a pure function of its key, so a hit is always
/// current and a lane set that changes simply misses once per layer. The per-step scalars
/// (positions, rows) never enter a table. Each entry keeps its host bytes until the table is
/// dropped, so its stream-ordered upload never races the source.
#[derive(Default)]
pub(crate) struct QsaLanesTableCache {
    tables: HashMap<Vec<Qwen4QsaLaneDesc>, (DeviceBuffer, Vec<u8>)>,
}

/// Launch shape every lane of a plan agrees on (admitted by `qsa_core_lanes_gfx942`).
#[derive(Clone, Debug)]
pub(crate) struct Qwen4QsaLanesShape {
    pub(crate) format: QsaKvFormat,
    pub(crate) index_heads: usize,
    pub(crate) index_dim: usize,
    pub(crate) index_kv_width: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) budget_blocks: usize,
    pub(crate) compress: usize,
    pub(crate) capacity: usize,
    pub(crate) full_capacity: usize,
    /// The pooling capacity every lane declares (`shape_blocks`): the score row stride.
    pub(crate) pooled_capacity: usize,
    pub(crate) query_row_stride: usize,
    pub(crate) index_q_norm: u64,
    pub(crate) q_norm: u64,
    pub(crate) k_norm: u64,
    /// The pool's norm (every lane shares it).
    pub(crate) pool_norm: u64,
    pub(crate) yarn: RopeYarn,
    /// The pool's RoPE (the same `RopeYarn` the lanes' prologue uses, as the interpreter builds it).
    pub(crate) pool_yarn: RopeYarn,
    /// Widest lane (rows).
    pub(crate) max_rows: usize,
    /// `grid.x` of the pool launch: the widest `ceil(rows / compress)`.
    pub(crate) pool_grid: usize,
    /// Most blocks any lane scores (`(position + rows) / compress`).
    pub(crate) max_blocks: usize,
    /// Dynamic LDS of the attention launch (the longest selected list x 8 bytes).
    pub(crate) attention_lds: usize,
}

/// An admitted set of lanes ready for [`Gfx942Device::qwen4_qsa_lanes_core`]; produced only
/// by `tensor_ops::qsa_core_lanes_gfx942`. Plain data: the device pointers are the ones the
/// admitted lanes borrowed, so launch the plan before releasing them.
pub struct Qwen4QsaLanesPlan {
    pub(crate) shape: Qwen4QsaLanesShape,
    pub(crate) lanes: Vec<Qwen4QsaLaneDesc>,
    /// `state.position` of each lane.
    pub(crate) positions: Vec<i32>,
    /// Rows of each lane (1..=[`QWEN4_QSA_LANE_ROWS`]).
    pub(crate) rows: Vec<i32>,
}

impl Qwen4QsaLanesPlan {
    /// Lanes in the plan.
    pub fn lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Kernel launches [`Gfx942Device::qwen4_qsa_lanes_core`] issues: prologue, pool, scores,
    /// select, attention.
    pub fn launch_count(&self) -> usize {
        5
    }

    /// The scratch (F32 words) of the score launch: every lane's `max_rows` score rows at the
    /// pooled-capacity stride.
    fn score_words(&self) -> Option<usize> {
        self.lanes
            .len()
            .checked_mul(self.shape.max_rows)?
            .checked_mul(self.shape.pooled_capacity)
    }
}

impl Gfx942Device<'_> {
    /// The QSA Core of every lane of `plan` (F32 or E4M3-FNUZ K/V, F32 index keys) as five
    /// launches over all lanes: decode prologue, incremental pool, row-8 scores,
    /// select-from-scores (writing each lane's persistent selection), per-head attention. Lane
    /// `i`'s state and outputs are bytewise what its per-lane launches write.
    ///
    /// The lanes' pointers ride in a device lane table, built once per distinct lane set and
    /// layer and reused until the set changes (the cache is keyed by content). The score
    /// scratch is the shared F16-X scratch, as the per-lane select's.
    pub fn qwen4_qsa_lanes_core(&mut self, plan: &Qwen4QsaLanesPlan) -> HipResult<()> {
        const WHAT: &str = "qwen4_qsa_lanes_core";
        let refuse = |msg: String| Err(HipError::new(0, &format!("{WHAT}: {msg}")));
        let n = plan.lanes.len();
        let c = &plan.shape;
        if n == 0
            || n > QWEN4_QSA_LANES_MAX
            || plan.positions.len() != n
            || plan.rows.len() != n
            || c.max_rows == 0
            || c.max_rows > QWEN4_QSA_LANE_ROWS
            || c.pool_grid == 0
            || c.max_blocks == 0
        {
            return refuse(format!(
                "a malformed plan ({n} lanes, {} max rows)",
                c.max_rows
            ));
        }
        let (prologue, attention) = match c.format {
            QsaKvFormat::F32 => (QWEN4_QSA_LANES_PROLOGUE_F32, QWEN4_QSA_LANES_ATTENTION_F32),
            QsaKvFormat::Fp8Fnuz => (
                QWEN4_QSA_LANES_PROLOGUE_FP8_FNUZ,
                QWEN4_QSA_LANES_ATTENTION_FP8_FNUZ,
            ),
            other => {
                return refuse(format!(
                    "K/V format {} has no gfx942 lane kernels",
                    other.name()
                ))
            }
        };
        let int = |v: usize, what: &str| {
            i32::try_from(v).map_err(|_| HipError::new(0, &format!("{WHAT}: {what} exceeds i32")))
        };
        let score_words = plan
            .score_words()
            .ok_or_else(|| HipError::new(0, &format!("{WHAT}: score scratch extent overflows")))?;
        let lane_stride = int(c.max_rows * c.pooled_capacity, "score lane stride")?;
        let score_stride = int(c.pooled_capacity, "score row stride")?;
        let n_lanes = n as u32;
        let max_rows = c.max_rows as u32;

        self.gpu.bind_thread()?;
        // The device lane table: staged on first use of this lane set.
        if !self
            .gpu
            .qsa_lanes_gfx942_tables
            .tables
            .contains_key(&plan.lanes)
        {
            if self.gpu.graphs.capture_mode {
                return refuse("lane table staging during graph capture".into());
            }
            if self.gpu.qsa_lanes_gfx942_tables.tables.len() >= QSA_LANES_TABLE_CACHE_MAX {
                // Queued launches may still read a table: drain before freeing any.
                self.gpu.hip.device_synchronize()?;
                self.gpu.qsa_lanes_gfx942_tables.tables.clear();
            }
            let bytes = Qwen4QsaLaneDesc::bytes(&plan.lanes);
            let buf = self.gpu.hip.malloc(bytes.len().next_multiple_of(256))?;
            // Stream-ordered before this launch; `bytes` lives in the cache entry, so the
            // (possibly staged) copy never reads a dropped source and nothing synchronizes.
            match self.gpu.active_stream.as_ref() {
                Some(stream) => self.gpu.hip.memcpy_htod_async(&buf, &bytes, stream)?,
                None => self.gpu.hip.memcpy_htod(&buf, &bytes)?,
            }
            self.gpu
                .qsa_lanes_gfx942_tables
                .tables
                .insert(plan.lanes.clone(), (buf, bytes));
        }
        let table = self.gpu.qsa_lanes_gfx942_tables.tables[&plan.lanes]
            .0
            .as_ptr();
        // Scores of every lane's rows: the shared scratch, grown (monotonically) to the shape
        // bound so the block count growing by one every `compress` tokens never regrows it.
        let scores = self.gpu.qwen4_f16_x_scratch(score_words * 2)?.buf.as_ptr();
        for kernel in [
            prologue,
            QWEN4_QSA_LANES_POOL,
            QWEN4_QSA_LANES_SCORES,
            QWEN4_QSA_LANES_SELECT,
            attention,
        ] {
            g942_qsa_lanes_ensure(self.gpu, kernel)?;
        }
        // By-value per-step table (`struct Qwen4QsaLaneInts`): positions, then rows.
        let push_ints = |args: &mut KernargBlob| {
            for lanes in [&plan.positions, &plan.rows] {
                for slot in 0..QWEN4_QSA_LANES_MAX {
                    args.push_i32(lanes.get(slot).copied().unwrap_or(0));
                }
            }
        };

        // 1. Prologue: grid [index_heads + heads + kv_heads + 1, max_rows, lanes].
        let mut args = KernargBlob::with_capacity(1280);
        args.push_ptr(table);
        for norm in [c.index_q_norm, c.q_norm, c.k_norm] {
            args.push_u64(norm);
        }
        for v in [
            c.index_heads,
            c.index_dim,
            c.index_kv_width,
            c.heads,
            c.kv_heads,
            c.head_dim,
        ] {
            args.push_i32(int(v, "prologue extent")?);
        }
        c.yarn.push(&mut args);
        push_ints(&mut args);
        args.pad_to(16);
        self.gpu.launch_blob_recorded(
            prologue,
            [
                u32::try_from(c.index_heads + c.heads + c.kv_heads + 1)
                    .map_err(|_| HipError::new(0, "qwen4_qsa_lanes_core: prologue grid"))?,
                max_rows,
                n_lanes,
            ],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        // 2. Pool: grid [pool_grid, ceil(index_dim / 256), lanes] (a lane completing no block
        // exits at its first check).
        let mut args = KernargBlob::with_capacity(1280);
        args.push_ptr(table);
        args.push_u64(c.pool_norm);
        args.push_i32(int(c.compress, "compress")?);
        args.push_i32(int(c.index_kv_width, "index width")?);
        c.pool_yarn.push(&mut args);
        push_ints(&mut args);
        args.pad_to(16);
        self.gpu.launch_blob_recorded(
            QWEN4_QSA_LANES_POOL,
            [
                u32::try_from(c.pool_grid)
                    .map_err(|_| HipError::new(0, "qwen4_qsa_lanes_core: pool grid"))?,
                u32::try_from(c.index_kv_width.div_ceil(256))
                    .map_err(|_| HipError::new(0, "qwen4_qsa_lanes_core: pool grid"))?,
                n_lanes,
            ],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        // 3. Scores: grid [ceil(max_blocks / 256), 1, lanes].
        let mut args = KernargBlob::with_capacity(1280);
        args.push_ptr(table);
        args.push_ptr(scores);
        for v in [
            lane_stride,
            score_stride,
            int(c.query_row_stride, "query row stride")?,
            int(c.index_dim, "index dim")?,
            int(c.compress, "compress")?,
        ] {
            args.push_i32(v);
        }
        push_ints(&mut args);
        args.pad_to(16);
        self.gpu.launch_blob_recorded(
            QWEN4_QSA_LANES_SCORES,
            [
                u32::try_from(c.max_blocks.div_ceil(256))
                    .map_err(|_| HipError::new(0, "qwen4_qsa_lanes_core: score grid"))?,
                1,
                n_lanes,
            ],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        // 4. Select from scores (+ the persistent-selection mirror): grid [max_rows, 1, lanes].
        let mut args = KernargBlob::with_capacity(1280);
        args.push_ptr(table);
        args.push_ptr(scores);
        for v in [
            lane_stride,
            score_stride,
            int(c.budget_blocks, "budget blocks")?,
            int(c.compress, "compress")?,
            int(c.capacity, "selected capacity")?,
        ] {
            args.push_i32(v);
        }
        push_ints(&mut args);
        args.pad_to(16);
        self.gpu.launch_blob_recorded(
            QWEN4_QSA_LANES_SELECT,
            [max_rows, 1, n_lanes],
            [256, 1, 1],
            0,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;

        // 5. Attention: grid [heads, 1, lanes * max_rows]; lane = z / max_rows, row = z % max_rows.
        let mut args = KernargBlob::with_capacity(1280);
        args.push_ptr(table);
        for v in [
            c.max_rows,
            c.heads,
            c.kv_heads,
            c.head_dim,
            c.budget_blocks,
            c.compress,
            c.capacity,
            c.full_capacity,
        ] {
            args.push_i32(int(v, "attention extent")?);
        }
        push_ints(&mut args);
        args.pad_to(16);
        self.gpu.launch_blob_recorded(
            attention,
            [
                u32::try_from(c.heads)
                    .map_err(|_| HipError::new(0, "qwen4_qsa_lanes_core: attention grid"))?,
                1,
                n_lanes
                    .checked_mul(max_rows)
                    .ok_or_else(|| HipError::new(0, "qwen4_qsa_lanes_core: attention grid"))?,
            ],
            [256, 1, 1],
            u32::try_from(c.attention_lds)
                .map_err(|_| HipError::new(0, "qwen4_qsa_lanes_core: attention LDS exceeds u32"))?,
            args.as_mut_slice(),
            crate::dispatch::ReplayLaunchBindings::NONE,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane table entry is eleven 8-byte pointers on both sides of the launch.
    #[test]
    fn lane_desc_matches_kernel_struct() {
        assert_eq!(std::mem::size_of::<Qwen4QsaLaneDesc>(), 88);
        let src = QWEN4_QSA_LANES_SRC;
        let at = src
            .find("struct Qwen4QsaLaneDesc {")
            .expect("kernel lane descriptor");
        let body = &src[at..src[at..].find("};").unwrap() + at];
        assert_eq!(body.matches(';').count(), 11);
        let fields = [
            "full_keys",
            "full_values",
            "raw_idx",
            "pooled",
            "sel_persist",
            "index_row",
            "qgate",
            "keys",
            "values",
            "selected",
            "output",
        ];
        let mut last = 0;
        for field in fields {
            let hit = body
                .find(&format!("* {field};"))
                .unwrap_or_else(|| panic!("{field}"));
            assert!(hit > last, "{field} out of order");
            last = hit;
        }
        let d = Qwen4QsaLaneDesc {
            full_keys: 1,
            full_values: 2,
            raw_idx: 3,
            pooled: 4,
            sel_persist: 5,
            index_row: 6,
            qgate: 7,
            keys: 8,
            values: 9,
            selected: 10,
            output: 11,
        };
        let bytes = Qwen4QsaLaneDesc::bytes(&[d, d]);
        assert_eq!(bytes.len(), 2 * 88);
        for (i, w) in bytes[..88].chunks(8).enumerate() {
            assert_eq!(u64::from_le_bytes(w.try_into().unwrap()), i as u64 + 1);
        }
    }

    /// The per-step table is `QWEN4_QSA_LANES_MAX` positions then as many rows, on both sides.
    #[test]
    fn per_step_table_matches_kernel_struct() {
        let src = QWEN4_QSA_LANES_SRC;
        assert!(src.contains(&format!(
            "#define QWEN4_QSA_LANES_MAX {QWEN4_QSA_LANES_MAX}\n"
        )));
        let at = src
            .find("struct Qwen4QsaLaneInts {")
            .expect("kernel per-step table");
        let body = &src[at..src[at..].find("};").unwrap() + at];
        let pos = body
            .find("int pos[QWEN4_QSA_LANES_MAX];")
            .expect("pos table");
        let rows = body
            .find("int rows[QWEN4_QSA_LANES_MAX];")
            .expect("rows table");
        assert!(pos < rows);
        // Two 4-byte tables plus the head of a launch stay under the 4 KiB kernarg limit.
        assert!(2 * 4 * QWEN4_QSA_LANES_MAX + 256 <= 4096);
    }
    /// The lane kernels run the production templates, byte for byte: every block between the
    /// `BEGIN` / `END verbatim tensor_ops.hip` markers must occur in the production source (the
    /// HIP source `tensor_ops.hip` behind `rope_yarn.h`), the `select_from_scores` body must be the
    /// whole body of the production kernel, and the attention row derivation the wrapper repeats
    /// must be `qsa_attention_batched`'s.
    #[test]
    fn qsa_lanes_gfx942_source_is_tensor_ops_verbatim() {
        const BEGIN: &str = "// BEGIN verbatim tensor_ops.hip\n";
        const END: &str = "// END verbatim tensor_ops.hip";
        let src = QWEN4_QSA_LANES_SRC;
        let production = crate::tensor_ops::TENSOR_OPS_SRC;
        let mut blocks = 0;
        let mut rest = src;
        while let Some(at) = rest.find(BEGIN) {
            let body = &rest[at + BEGIN.len()..];
            let end = body.find(END).expect("END marker after every BEGIN marker");
            let block = body[..end].trim_end_matches('\n');
            assert!(
                production.contains(block),
                "verbatim block {blocks} of qwen4_qsa_lanes_f32.gfx942.hip is no longer a \
                 substring of tensor_ops.hip: mirror the production edit into the lanes kernel\n\
                 block starts: {:?}",
                &block[..block.len().min(120)]
            );
            blocks += 1;
            rest = &body[end + END.len()..];
        }
        assert_eq!(
            blocks, 8,
            "bf16 helpers, RopeYarn, K/V helpers, prologue, pool, row scores, attention row, \
             select_from_scores body"
        );
        // The whole body of the production select_from_scores kernel, and nothing else.
        let kernel = production
            .find("void indexed_attention_select_from_scores(")
            .expect("production select_from_scores");
        let body_at = production[kernel..].find("    int* mirror) {\n").unwrap()
            + kernel
            + "    int* mirror) {\n".len();
        let production_body =
            &production[body_at..body_at + production[body_at..].find("\n}\n").unwrap()];
        let ours_at = src
            .find("static __device__ __forceinline__ void qsa_lanes_select_from_scores_body(")
            .expect("lane select_from_scores body");
        let ours_start = ours_at
            + src[ours_at..].find("    int* mirror) {\n").unwrap()
            + "    int* mirror) {\n".len();
        let ours_body = &src[ours_start..ours_start + src[ours_start..].find("\n}\n").unwrap()];
        let ours_body: String = ours_body
            .lines()
            .filter(|l| !l.starts_with("// BEGIN verbatim") && !l.starts_with("// END verbatim"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            ours_body, production_body,
            "select_from_scores body drifted"
        );
        // The per-row attention derivation of `qsa_attention_batched` (the wrapper owns the lane
        // split of grid z, so it repeats these four statements).
        for line in [
            "int visible = position_start + row + 1;",
            "int block_count = visible / compress;",
            "int chosen = budget_blocks < block_count ? budget_blocks : block_count;",
            "int selected_len = chosen * compress + visible - block_count * compress;",
            "if (selected_len > capacity) selected_len = capacity;",
        ] {
            assert!(production.contains(line), "production lost `{line}`");
        }
        assert!(production.contains(
            "q_with_gate + row * 2 * n_heads * head_dim + (int)blockIdx.x * head_dim * 2,\n        full_keys, full_values, selected_indices + row * capacity,\n        output + row * n_heads * head_dim, selected_len, visible,"
        ));
        assert!(src.contains("int visible = ints.pos[l] + row + 1;"));
        for line in [
            "int block_count = visible / compress;",
            "int chosen = budget_blocks < block_count ? budget_blocks : block_count;",
            "int selected_len = chosen * compress + visible - block_count * compress;",
            "if (selected_len > capacity) selected_len = capacity;",
        ] {
            assert!(src.contains(line), "lane wrapper lost `{line}`");
        }
        // Every exported lane symbol is defined, once.
        for k in [
            QWEN4_QSA_LANES_PROLOGUE_F32,
            QWEN4_QSA_LANES_PROLOGUE_FP8_FNUZ,
            QWEN4_QSA_LANES_POOL,
            QWEN4_QSA_LANES_SCORES,
            QWEN4_QSA_LANES_SELECT,
            QWEN4_QSA_LANES_ATTENTION_F32,
            QWEN4_QSA_LANES_ATTENTION_FP8_FNUZ,
        ] {
            assert!(src.contains(k), "kernel file lacks export {k}");
        }
        // The production bodies the wrappers call are the instantiations the per-lane launches use.
        assert_eq!(src.matches("qsa_decode_prologue<F, float>(").count(), 1);
        assert!(production.contains(
            "QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_f32, QSA_KV_F32, float, float)"
        ));
        assert!(production.contains(
            "QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_fp8_fnuz, QSA_KV_FP8_FNUZ, unsigned char,\n                    float)"
        ));
        assert!(src
            .contains("QWEN4_QSA_LANES_PROLOGUE(qwen4_qsa_lanes_prologue_f32_gfx942, QSA_KV_F32)"));
        assert!(src.contains(
            "QWEN4_QSA_LANES_PROLOGUE(qwen4_qsa_lanes_prologue_fp8_fnuz_gfx942, QSA_KV_FP8_FNUZ)"
        ));
        assert!(production.contains("QSA_ATTENTION_KERNELS(f32, QSA_KV_F32, float)"));
        assert!(
            production.contains("QSA_ATTENTION_KERNELS(fp8_fnuz, QSA_KV_FP8_FNUZ, unsigned char)")
        );
        assert!(src.contains(
            "QWEN4_QSA_LANES_ATTENTION(qwen4_qsa_lanes_attention_f32_gfx942, QSA_KV_F32)"
        ));
        assert!(src.contains(
            "QWEN4_QSA_LANES_ATTENTION(qwen4_qsa_lanes_attention_fp8_fnuz_gfx942, QSA_KV_FP8_FNUZ)"
        ));
        assert!(src.contains("indexed_attention_select_scores_rows<8>("));
        assert!(production.contains(
            "indexed_attention_select_scores_rows<8>(query, pooled, scores, rows, query_row_stride,"
        ));
        assert!(src.contains("qsa_pool_rope<float>("));
        assert!(production.contains("qsa_pool_rope(raw_keys, pooled, norm, block_count, compress, index_dim, first_block, yarn);"));
        assert!(!src.contains("HIPFIRE_COMPILER_FLAGS"));
    }
}
