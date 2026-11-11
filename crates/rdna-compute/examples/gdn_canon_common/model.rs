// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU stream models for the S0 kills (CHUNK-CANONICAL-DESIGN §3.3, §6 S0 kill 3).
//!
//! * [`RetainedReferenceStream`] — the slow **full-512 retained-input
//!   reference**. It keeps every raw row of the current 512-row commit frame
//!   and recomputes the whole frame from the frame-entry state on every
//!   `append`, so its numerics cannot depend on how the caller partitioned the
//!   stream. Library-quality: later slices call it as the oracle.
//! * [`BoundedTileStream`] — the **bounded-tile replay** model the production
//!   carry is meant to match: FP32 tile-entry state + conv history at the tile
//!   entry + raw pre-conv inputs of the incomplete microtile only; replay only
//!   the incomplete tile; quantize at absolute 512 seams and at `seal`.
//!   With a non-[`Fault::None`] value it becomes a deliberately broken
//!   implementation (one fault per design §6 S0 kill-4 item) used to prove
//!   that the oracle actually discriminates.
//!
//! Tile arithmetic ([`tile_scan`]). This is **not** a model of the WMMA
//! kernel's numerics (the GPU legs of the kills cover those). It is an exact,
//! deterministic stand-in that preserves the two structural properties the
//! carry design hinges on, so that state-management bugs are visible:
//!
//! 1. the per-row gate is formed from a *tile-local* running log-gate
//!    cumulation (`eff_t = fl(cum_t - cum_{t-1})`, cum restarting at the tile
//!    origin) — the CPU analogue of "tile-local gate prefix / gl-dependent
//!    state update" — so the result depends on the absolute tile grid, and a
//!    partial tile's end state is **not** a valid entry state for the next
//!    row once the grid moves;
//! 2. outputs of row `i` depend only on rows `<= i` of the same tile and its
//!    entry state/conv history (causal prefix property).
//!
//! Everything else (Q8/Q8Ef codec, EF, plain-Q8 stochastic rounding keys)
//! reuses `gdn_q8ef_reference` unchanged.

use super::q8ef::{self as q8, Geometry, LiveState, RowInput};
use super::stream::*;
use super::util::*;
use rdna_compute::gdn_stream::*;

pub const CONV_TAPS: usize = 4;

// ───────────────────────────── constants & data ──────────────────────────────

#[derive(Clone, Debug)]
pub struct ModelConsts {
    /// `qkv_width * CONV_TAPS`, `[channel][tap]`.
    pub conv_w: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
}

impl ModelConsts {
    pub fn new(g: &StreamGeometry, seed: u64) -> Self {
        Self {
            conv_w: lcg_vec(seed ^ 0xC0, g.qkv_width() * CONV_TAPS, -0.5, 0.5),
            a_log: lcg_vec(seed ^ 0xA1, g.hv, 0.0, 2.0),
            dt_bias: lcg_vec(seed ^ 0xD7, g.hv, -4.0, -2.0),
        }
    }
}

pub fn gen_rows(g: &StreamGeometry, seed: u64, rows: usize) -> RawRows {
    RawRows {
        rows,
        qkv: lcg_vec(seed ^ 1, rows * g.qkv_width(), -1.732_050_8, 1.732_050_8),
        alpha: lcg_vec(seed ^ 4, rows * g.hv, -1.732_050_8, 1.732_050_8),
        beta: lcg_vec(seed ^ 3, rows * g.hv, -2.0, 2.0),
    }
}

/// Nonzero incoming Q8(+EF) state at `position` (EF is nonzero for Q8Ef: the
/// state is encoded from a dense F32 state).
pub fn gen_initial(g: &StreamGeometry, seed: u64, position: u64) -> InitialState {
    let dense = lcg_vec(seed ^ 0x51, g.state_elems(), -0.5, 0.5);
    let live = match g.state_quant {
        GdnStateQuant::Q8Ef => LiveState::from_dense(&dense),
        GdnStateQuant::Q8 => LiveState {
            slot: q8::quantize_plain(&dense, 0),
            ef: vec![0; dense.len()],
        },
        GdnStateQuant::F32 => panic!("F32 streams have no Q8 plane (no artificial commit)"),
    };
    InitialState {
        position,
        live,
        conv: lcg_vec(seed ^ 0xC9, g.conv_hist_rows * g.qkv_width(), -0.5, 0.5),
    }
}

pub fn cpu_geometry(state_quant: GdnStateQuant) -> StreamGeometry {
    StreamGeometry {
        hk: 1,
        hv: 2,
        micro_rows: 64,
        commit_rows: GDN_COMMIT_ROWS,
        conv_hist_rows: CONV_TAPS - 1,
        state_quant,
    }
}

pub fn model_plan(g: &StreamGeometry) -> GdnPrefillPlan {
    let mut p = GdnPrefillPlan {
        recipe_id: 0,
        algorithm: match g.micro_rows {
            64 => GdnScanAlgorithm::Scan64,
            16 => GdnScanAlgorithm::Wy16,
            _ => GdnScanAlgorithm::Sequential,
        },
        micro_rows: g.micro_rows,
        commit_rows: g.commit_rows,
        projection_recipe: GdnProjectionRecipe {
            qkv: 1,
            z: 1,
            alpha: 1,
            beta: 1,
            out: 1,
        },
        scan_operand_format: GdnTensorFormat::F32,
        scan_output_format: GdnTensorFormat::F32,
        state_quant: g.state_quant,
        producer_tile_rows: 64,
    };
    p.recipe_id = p.compute_recipe_id();
    p
}

/// Frozen metadata of a stream at `position` under the absolute-grid rules
/// (independent of any implementation).
pub fn expected_meta(
    g: &StreamGeometry,
    recipe_id: u64,
    phase: GdnStreamPhase,
    position: u64,
    phase_start: u64,
    owner_generation: u64,
) -> GdnStreamMeta {
    let c = g.micro_rows as u64;
    let commit = g.commit_rows as u64;
    let (tile_base, frame_base) = match phase {
        GdnStreamPhase::Sealed => (position, position),
        GdnStreamPhase::Open => (
            phase_start.max(position / c * c),
            phase_start.max(position / commit * commit),
        ),
    };
    GdnStreamMeta {
        schema_version: GDN_STREAM_SCHEMA_VERSION,
        recipe_id,
        phase,
        position,
        phase_start: match phase {
            GdnStreamPhase::Sealed => position,
            GdnStreamPhase::Open => phase_start,
        },
        frame_base,
        tile_base,
        pending_rows: (position - tile_base) as u32,
        owner_generation,
    }
}

// ───────────────────────────── tile arithmetic ───────────────────────────────

/// Causal depthwise conv (taps over `hist ++ rows`) followed by SiLU.
fn conv_silu(
    g: &StreamGeometry,
    c: &ModelConsts,
    hist: &[f32],
    qkv: &[f32],
    rows: usize,
) -> Vec<f32> {
    let w = g.qkv_width();
    let h = g.conv_hist_rows;
    let mut out = vec![0.0f32; rows * w];
    for t in 0..rows {
        for ch in 0..w {
            let mut acc = 0.0f32;
            for j in 0..CONV_TAPS {
                let idx = t as i64 + j as i64 - h as i64;
                let x = if idx < 0 {
                    hist[(h as i64 + idx) as usize * w + ch]
                } else {
                    qkv[idx as usize * w + ch]
                };
                acc = c.conv_w[ch * CONV_TAPS + j].mul_add(x, acc);
            }
            out[t * w + ch] = acc / (1.0 + (-acc).exp());
        }
    }
    out
}

/// Last `conv_hist_rows` raw rows of `hist ++ qkv`.
pub fn advance_conv(g: &StreamGeometry, hist: &[f32], qkv: &[f32]) -> Vec<f32> {
    let w = g.qkv_width();
    let h = g.conv_hist_rows;
    let mut all: Vec<f32> = Vec::with_capacity(hist.len() + qkv.len());
    all.extend_from_slice(hist);
    all.extend_from_slice(qkv);
    let rows = all.len() / w;
    all[(rows - h) * w..].to_vec()
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Scan one tile (all supplied rows) from `entry` (dense `[head][row][key]`)
/// with conv history `hist`. Returns the tile's outputs (`rows * hv * 128`) and
/// the end state. See the module doc for the two structural properties.
pub fn tile_scan(
    g: &StreamGeometry,
    consts: &ModelConsts,
    entry: &[f32],
    hist: &[f32],
    rows: &RawRows,
) -> (Vec<f32>, Vec<f32>) {
    let w = g.qkv_width();
    let (hk, hv) = (g.hk, g.hv);
    let conv = conv_silu(g, consts, hist, &rows.qkv, rows.rows);
    let geom = Geometry {
        key_heads: hk,
        value_heads: hv,
    };
    let mut dense = entry.to_vec();
    let mut cum = vec![0.0f32; hv];
    let mut outs = vec![0.0f32; rows.rows * hv * HD];
    for t in 0..rows.rows {
        let row = &conv[t * w..(t + 1) * w];
        let mut gate = vec![0.0f32; hv];
        let mut beta = vec![0.0f32; hv];
        for h in 0..hv {
            let raw = -consts.a_log[h].exp() * softplus(rows.alpha[t * hv + h] + consts.dt_bias[h]);
            let prev = cum[h];
            cum[h] = prev + raw;
            gate[h] = cum[h] - prev;
            beta[h] = 1.0 / (1.0 + (-rows.beta[t * hv + h]).exp());
        }
        let r = RowInput {
            q: row[..hk * HD].to_vec(),
            k: row[hk * HD..2 * hk * HD].to_vec(),
            v: row[2 * hk * HD..].to_vec(),
            gate,
            beta,
        };
        q8::step_f32(
            geom,
            &mut dense,
            &r,
            &mut outs[t * hv * HD..(t + 1) * hv * HD],
        );
    }
    (outs, dense)
}

// ───────────────────────────── codec glue ────────────────────────────────────

/// Dense state of a Q8 plane. `ignore_ef` models the "missing EF" fault.
fn plane_to_dense(g: &StreamGeometry, plane: &LiveState, ignore_ef: bool) -> Vec<f32> {
    match g.state_quant {
        GdnStateQuant::Q8Ef if !ignore_ef => plane.to_dense(),
        _ => q8::dequantize_plain(&plane.slot),
    }
}

fn dense_to_plane(g: &StreamGeometry, dense: &[f32], key: u32) -> LiveState {
    match g.state_quant {
        GdnStateQuant::Q8Ef => LiveState::from_dense(dense),
        GdnStateQuant::Q8 => LiveState {
            slot: q8::quantize_plain(dense, key),
            ef: vec![0; dense.len()],
        },
        GdnStateQuant::F32 => panic!("F32 streams have no Q8 plane"),
    }
}

fn sealed_of(g: &StreamGeometry, position: u64, plane: &LiveState, conv: &[f32]) -> SealedState {
    SealedState {
        position,
        codes: plane.slot.codes.clone(),
        scales: plane.slot.scales.clone(),
        ef: match g.state_quant {
            GdnStateQuant::Q8Ef => plane.ef.clone(),
            _ => vec![0; plane.ef.len()],
        },
        conv: conv.to_vec(),
    }
}

fn concat(g: &StreamGeometry, a: &RawRows, b: &RawRows) -> RawRows {
    let _ = g;
    let mut r = a.clone();
    r.rows += b.rows;
    r.qkv.extend_from_slice(&b.qkv);
    r.alpha.extend_from_slice(&b.alpha);
    r.beta.extend_from_slice(&b.beta);
    r
}

fn garbage_f32(n: usize) -> Vec<f32> {
    vec![f32::from_bits(0x7fc0_dead); n]
}

fn garbage_plane(g: &StreamGeometry) -> LiveState {
    LiveState {
        slot: q8::Q8Slot {
            codes: vec![0x5a; g.state_elems()],
            scales: vec![f32::NAN; g.hv * HD],
        },
        ef: vec![0xa5a5; g.state_elems()],
    }
}

// ───────────────────────────── retained reference ────────────────────────────

#[derive(Clone)]
struct RefState {
    phase: GdnStreamPhase,
    position: u64,
    phase_start: u64,
    plane: LiveState,
    plane_pos: u64,
    frame_start: u64,
    frame_entry: Vec<f32>,
    frame_conv: Vec<f32>,
    retained: RawRows,
    conv_live: Vec<f32>,
}

/// Full-512 retained-input reference. Retains every raw row since the current
/// frame start (≤ 512 rows) and recomputes the frame on every append.
pub struct RetainedReferenceStream {
    g: StreamGeometry,
    consts: ModelConsts,
    recipe_id: u64,
    st: RefState,
}

impl RetainedReferenceStream {
    pub fn new(g: StreamGeometry, consts: ModelConsts, init: &InitialState) -> Self {
        let recipe_id = model_plan(&g).recipe_id;
        let mut s = Self {
            st: RefState {
                phase: GdnStreamPhase::Sealed,
                position: 0,
                phase_start: 0,
                plane: init.live.clone(),
                plane_pos: 0,
                frame_start: 0,
                frame_entry: vec![],
                frame_conv: vec![],
                retained: RawRows::empty(),
                conv_live: vec![],
            },
            g,
            consts,
            recipe_id,
        };
        s.reset(init).expect("reference reset");
        s
    }

    /// Recompute the retained frame from its entry state. Returns all retained
    /// rows' outputs and the end state of the (possibly partial) last tile.
    fn recompute(&self, st: &RefState) -> (Vec<f32>, Vec<f32>) {
        let g = &self.g;
        let c = g.micro_rows as u64;
        let mut entry = st.frame_entry.clone();
        let mut hist = st.frame_conv.clone();
        let mut tb = st.frame_start;
        let mut idx = 0usize;
        let mut outs: Vec<f32> = Vec::new();
        let mut cur_end = entry.clone();
        while idx < st.retained.rows {
            let b = (tb / c + 1) * c;
            let len = ((b - tb) as usize).min(st.retained.rows - idx);
            let tile = st.retained.slice(g, idx, idx + len);
            let (o, end) = tile_scan(g, &self.consts, &entry, &hist, &tile);
            outs.extend(o);
            cur_end = end.clone();
            idx += len;
            if tb + len as u64 == b {
                hist = advance_conv(g, &hist, &tile.qkv);
                entry = end;
                tb = b;
            }
        }
        (outs, cur_end)
    }

    fn open(&mut self) {
        let st = &mut self.st;
        st.frame_start = st.position;
        st.frame_entry = plane_to_dense(&self.g, &st.plane, false);
        st.frame_conv = st.conv_live.clone();
        st.retained = RawRows::empty();
        st.phase_start = st.position;
        st.phase = GdnStreamPhase::Open;
    }

    fn seal_compute(&self, st: &RefState) -> (LiveState, u64) {
        if st.phase == GdnStreamPhase::Sealed {
            return (st.plane.clone(), st.plane_pos);
        }
        let end = if st.retained.rows > 0 {
            self.recompute(st).1
        } else {
            st.frame_entry.clone()
        };
        if st.plane_pos != st.position {
            (
                dense_to_plane(&self.g, &end, st.position as u32),
                st.position,
            )
        } else {
            (st.plane.clone(), st.plane_pos)
        }
    }
}

impl GdnStreamUnderTest for RetainedReferenceStream {
    fn label(&self) -> String {
        "retained-512-reference".into()
    }
    fn geometry(&self) -> StreamGeometry {
        self.g
    }
    fn reset(&mut self, init: &InitialState) -> R<()> {
        let st = &mut self.st;
        st.phase = GdnStreamPhase::Sealed;
        st.position = init.position;
        st.phase_start = init.position;
        st.plane = init.live.clone();
        st.plane_pos = init.position;
        st.frame_start = init.position;
        st.frame_entry = vec![];
        st.frame_conv = vec![];
        st.retained = RawRows::empty();
        st.conv_live = init.conv.clone();
        Ok(())
    }
    fn meta(&self) -> GdnStreamMeta {
        expected_meta(
            &self.g,
            self.recipe_id,
            self.st.phase,
            self.st.position,
            self.st.phase_start,
            0,
        )
    }
    fn append(&mut self, rows: &RawRows) -> R<Vec<f32>> {
        if rows.rows == 0 {
            return Ok(vec![]);
        }
        if self.st.phase == GdnStreamPhase::Sealed {
            self.open();
        }
        let g = self.g;
        let ow = g.out_width();
        let commit = g.commit_rows as u64;
        let mut out: Vec<f32> = Vec::with_capacity(rows.rows * ow);
        let mut consumed = 0usize;
        while consumed < rows.rows {
            let frame_end = (self.st.frame_start / commit + 1) * commit;
            let space = (frame_end - self.st.frame_start) as usize - self.st.retained.rows;
            let take = space.min(rows.rows - consumed);
            let old_len = self.st.retained.rows;
            let piece = rows.slice(&g, consumed, consumed + take);
            self.st.retained = concat(&g, &self.st.retained, &piece);
            let (o, end) = self.recompute(&self.st);
            out.extend_from_slice(&o[old_len * ow..]);
            self.st.position += take as u64;
            self.st.conv_live = advance_conv(&g, &self.st.conv_live, &piece.qkv);
            consumed += take;
            if self.st.frame_start + self.st.retained.rows as u64 == frame_end {
                let plane = dense_to_plane(&g, &end, frame_end as u32);
                self.st.frame_conv = advance_conv(&g, &self.st.frame_conv, &self.st.retained.qkv);
                self.st.frame_entry = plane_to_dense(&g, &plane, false);
                self.st.plane = plane;
                self.st.plane_pos = frame_end;
                self.st.frame_start = frame_end;
                self.st.retained = RawRows::empty();
            }
        }
        Ok(out)
    }
    fn seal(&mut self) -> R<SealedState> {
        let (plane, plane_pos) = self.seal_compute(&self.st);
        self.st.plane = plane;
        self.st.plane_pos = plane_pos;
        self.st.phase = GdnStreamPhase::Sealed;
        self.st.retained = RawRows::empty();
        Ok(sealed_of(
            &self.g,
            self.st.position,
            &self.st.plane,
            &self.st.conv_live,
        ))
    }
    fn peek_seal(&mut self) -> R<SealedState> {
        let (plane, _) = self.seal_compute(&self.st);
        Ok(sealed_of(
            &self.g,
            self.st.position,
            &plane,
            &self.st.conv_live,
        ))
    }
    fn snapshot(&mut self) -> R<OpaqueSnapshot> {
        Ok(OpaqueSnapshot(Box::new(self.st.clone())))
    }
    fn restore(&mut self, snap: &OpaqueSnapshot) -> R<()> {
        self.st = snap
            .0
            .downcast_ref::<RefState>()
            .ok_or("snapshot type mismatch")?
            .clone();
        Ok(())
    }
    fn clobber_live(&mut self) -> R<()> {
        let g = self.g;
        let st = &mut self.st;
        st.phase = GdnStreamPhase::Sealed;
        st.position = 0xDEAD_BEEF;
        st.phase_start = 0xDEAD;
        st.plane = garbage_plane(&g);
        st.plane_pos = 0xBEEF;
        st.frame_start = 0xFEED;
        st.frame_entry = garbage_f32(g.state_elems());
        st.frame_conv = garbage_f32(g.conv_hist_rows * g.qkv_width());
        st.retained = RawRows::empty();
        st.conv_live = garbage_f32(g.conv_hist_rows * g.qkv_width());
        Ok(())
    }
}

// ───────────────────────────── bounded-tile stream ───────────────────────────

/// One deliberately wrong behavior per design §6 S0 kill-4 item (plus the
/// current production call-end quantization). `None` is the correct bounded
/// replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fault {
    None,
    /// Current production: tile grid and Q8 seams restart at every outer call
    /// start, final quantization at call end, always sealed (design §1 / §3.1).
    CallEndQuant,
    /// F32 state carried across calls but the tile grid restarts at the call
    /// start and no raw prefix is replayed (a partial tile's end state used as
    /// the next tile's entry; design §3.1).
    NaiveEntryCarry,
    /// Snapshot omits the pending raw rows (carry) of the incomplete tile.
    MissingPendingRaw,
    /// Snapshot omits the FP32 tile-entry state (restore falls back to the
    /// last-seam Q8 plane).
    MissingTileEntry,
    /// Q8Ef plane read without its F16 residual.
    MissingEf,
    /// Snapshot omits the F16 residual plane (visible on restored Sealed state).
    SnapshotDropsEf,
    /// Hidden read of the (stale) Q8 plane whenever `pending_rows == 0`.
    StaleQ8,
    /// Replay of the incomplete tile consumes the *already advanced* live conv
    /// history instead of the saved tile-entry history.
    DuplicateConv,
    /// Plain-Q8 stochastic rounding keyed by a process/call counter instead of
    /// the absolute seam position.
    CallIndexRounding,
    /// Replay assembles the pending prefix from the full carry capacity
    /// (poison-filled beyond `pending_rows`) instead of `pending_rows`.
    PendingCapacityRead,
}

impl Fault {
    pub const ALL_FAULTS: [Fault; 10] = [
        Fault::CallEndQuant,
        Fault::NaiveEntryCarry,
        Fault::MissingPendingRaw,
        Fault::MissingTileEntry,
        Fault::MissingEf,
        Fault::SnapshotDropsEf,
        Fault::StaleQ8,
        Fault::DuplicateConv,
        Fault::CallIndexRounding,
        Fault::PendingCapacityRead,
    ];
    /// Whether the fault is observable in the given quantization mode.
    pub fn applies(self, q: GdnStateQuant) -> bool {
        match self {
            Fault::MissingEf | Fault::SnapshotDropsEf => q == GdnStateQuant::Q8Ef,
            Fault::CallIndexRounding => q == GdnStateQuant::Q8,
            _ => true,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Fault::None => "none",
            Fault::CallEndQuant => "call_end_quantization",
            Fault::NaiveEntryCarry => "f32_entry_without_raw_carry",
            Fault::MissingPendingRaw => "missing_pending_raw_carry",
            Fault::MissingTileEntry => "missing_tile_entry_state",
            Fault::MissingEf => "missing_ef_read",
            Fault::SnapshotDropsEf => "snapshot_drops_ef",
            Fault::StaleQ8 => "stale_q8_read",
            Fault::DuplicateConv => "duplicate_conv",
            Fault::CallIndexRounding => "call_index_rounding",
            Fault::PendingCapacityRead => "pending_capacity_read_poison",
        }
    }
}

#[derive(Clone)]
struct BState {
    phase: GdnStreamPhase,
    position: u64,
    phase_start: u64,
    plane: LiveState,
    plane_pos: u64,
    entry: Vec<f32>,
    entry_conv: Vec<f32>,
    conv_live: Vec<f32>,
    /// Carry capacity `micro_rows` rows; poison-filled beyond `pending_rows`.
    pending: RawRows,
    pending_rows: usize,
    call_counter: u32,
}

pub struct BoundedTileStream {
    g: StreamGeometry,
    consts: ModelConsts,
    recipe_id: u64,
    fault: Fault,
    st: BState,
}

impl BoundedTileStream {
    pub fn new(g: StreamGeometry, consts: ModelConsts, fault: Fault, init: &InitialState) -> Self {
        let recipe_id = model_plan(&g).recipe_id;
        let mut s = Self {
            st: BState {
                phase: GdnStreamPhase::Sealed,
                position: 0,
                phase_start: 0,
                plane: init.live.clone(),
                plane_pos: 0,
                entry: vec![],
                entry_conv: vec![],
                conv_live: vec![],
                pending: RawRows::empty(),
                pending_rows: 0,
                call_counter: 0,
            },
            g,
            consts,
            recipe_id,
            fault,
        };
        s.reset(init).expect("bounded reset");
        s
    }

    pub fn fault(&self) -> Fault {
        self.fault
    }

    fn poison_pending(&self) -> RawRows {
        let g = &self.g;
        let c = g.micro_rows;
        RawRows {
            rows: c,
            qkv: garbage_f32(c * g.qkv_width()),
            alpha: garbage_f32(c * g.hv),
            beta: garbage_f32(c * g.hv),
        }
    }

    fn dense_of(&self, plane: &LiveState) -> Vec<f32> {
        plane_to_dense(&self.g, plane, self.fault == Fault::MissingEf)
    }

    fn pending_valid(&self, st: &BState) -> RawRows {
        st.pending
            .slice(&self.g, 0, st.pending_rows.min(self.g.micro_rows))
    }

    fn store_pending(&self, st: &mut BState, rows: &RawRows) {
        let g = &self.g;
        let c = g.micro_rows;
        let mut p = self.poison_pending();
        let w = g.qkv_width();
        // Capacity is `micro_rows`; a correct stream never exceeds it (a
        // partial tile has < micro_rows rows). Faulted streams may; clamp.
        let n = rows.rows.min(c);
        p.qkv[..n * w].copy_from_slice(&rows.qkv[..n * w]);
        p.alpha[..n * g.hv].copy_from_slice(&rows.alpha[..n * g.hv]);
        p.beta[..n * g.hv].copy_from_slice(&rows.beta[..n * g.hv]);
        st.pending = p;
        st.pending_rows = rows.rows;
    }

    fn open(&mut self) {
        let entry = self.dense_of(&self.st.plane);
        let st = &mut self.st;
        st.entry = entry;
        st.entry_conv = st.conv_live.clone();
        st.pending_rows = 0;
        st.phase_start = st.position;
        st.phase = GdnStreamPhase::Open;
    }

    /// Real bounded replay (also the `StaleQ8` / `DuplicateConv` /
    /// `PendingCapacityRead` / `CallIndexRounding` carriers).
    fn append_replay(&mut self, rows: &RawRows) -> Vec<f32> {
        let g = self.g;
        let ow = g.out_width();
        let c = g.micro_rows as u64;
        let commit = g.commit_rows as u64;
        self.st.call_counter += 1;
        if self.st.phase == GdnStreamPhase::Sealed {
            self.open();
        }
        if self.fault == Fault::StaleQ8 && self.st.pending_rows == 0 {
            // hidden read of the stale Q8 plane at pending==0
            self.st.entry = self.dense_of(&self.st.plane);
        }
        let old_pending = self.st.pending_rows;
        let prefix = if self.fault == Fault::PendingCapacityRead {
            self.st.pending.clone() // full capacity incl. poison rows
        } else {
            self.pending_valid(&self.st)
        };
        let pend_eff = prefix.rows;
        let comb = concat(&g, &prefix, rows);
        let total = comb.rows;
        let mut tile_base = self.st.position - old_pending as u64;
        let mut idx = 0usize;
        let mut out: Vec<f32> = Vec::with_capacity(rows.rows * ow);
        let mut first_tile = true;
        while idx < total {
            let b = (tile_base / c + 1) * c;
            let len = ((b - tile_base) as usize).min(total - idx);
            let tile = comb.slice(&g, idx, idx + len);
            let hist: Vec<f32> =
                if self.fault == Fault::DuplicateConv && first_tile && old_pending > 0 {
                    self.st.conv_live.clone()
                } else {
                    self.st.entry_conv.clone()
                };
            first_tile = false;
            let (tout, end) = tile_scan(&g, &self.consts, &self.st.entry, &hist, &tile);
            for j in 0..len {
                if idx + j >= pend_eff {
                    out.extend_from_slice(&tout[j * ow..(j + 1) * ow]);
                }
            }
            if tile_base + len as u64 == b {
                self.st.entry_conv = advance_conv(&g, &self.st.entry_conv, &tile.qkv);
                self.st.entry = end.clone();
                tile_base = b;
                idx += len;
                if b % commit == 0 {
                    let key = if self.fault == Fault::CallIndexRounding {
                        self.st.call_counter
                    } else {
                        b as u32
                    };
                    let plane = dense_to_plane(&g, &end, key);
                    self.st.entry = self.dense_of(&plane);
                    self.st.plane = plane;
                    self.st.plane_pos = b;
                }
            } else {
                break; // partial: stays pending
            }
        }
        let new_pending = comb.slice(&g, idx, total);
        let mut st = std::mem::replace(&mut self.st, dummy_state());
        self.store_pending(&mut st, &new_pending);
        st.pending_rows = total - idx;
        if self.fault == Fault::PendingCapacityRead {
            // keep bookkeeping sane for the *next* call's tile_base
            st.pending_rows = ((st.position + rows.rows as u64).saturating_sub(tile_base) as usize)
                .min(g.micro_rows);
        }
        st.position += rows.rows as u64;
        st.conv_live = advance_conv(&g, &st.conv_live, &rows.qkv);
        self.st = st;
        out
    }

    /// Current production behavior: every outer call is a fresh local phase.
    fn append_call_end(&mut self, rows: &RawRows) -> Vec<f32> {
        let g = self.g;
        let ow = g.out_width();
        let c = g.micro_rows;
        let commit = g.commit_rows;
        self.st.call_counter += 1;
        let mut dense = self.dense_of(&self.st.plane);
        let mut hist = self.st.conv_live.clone();
        let mut out = Vec::with_capacity(rows.rows * ow);
        let mut seg0 = 0usize;
        while seg0 < rows.rows {
            let seg1 = (seg0 + commit).min(rows.rows);
            let mut t0 = seg0;
            while t0 < seg1 {
                let t1 = (t0 + c).min(seg1);
                let tile = rows.slice(&g, t0, t1);
                let (o, end) = tile_scan(&g, &self.consts, &dense, &hist, &tile);
                out.extend(o);
                hist = advance_conv(&g, &hist, &tile.qkv);
                dense = end;
                t0 = t1;
            }
            let plane = dense_to_plane(&g, &dense, self.st.call_counter);
            if seg1 < rows.rows {
                dense = plane_to_dense(&g, &plane, false);
            } else {
                self.st.plane = plane;
            }
            seg0 = seg1;
        }
        self.st.position += rows.rows as u64;
        self.st.plane_pos = self.st.position;
        self.st.conv_live = hist;
        out
    }

    /// F32 state carried, tile grid restarted at call start, no raw carry.
    fn append_naive(&mut self, rows: &RawRows) -> Vec<f32> {
        let g = self.g;
        let ow = g.out_width();
        let c = g.micro_rows;
        if self.st.phase == GdnStreamPhase::Sealed {
            self.open();
        }
        let mut dense = self.st.entry.clone();
        let mut hist = self.st.conv_live.clone();
        let mut out = Vec::with_capacity(rows.rows * ow);
        let mut t0 = 0usize;
        while t0 < rows.rows {
            let t1 = (t0 + c).min(rows.rows);
            let tile = rows.slice(&g, t0, t1);
            let (o, end) = tile_scan(&g, &self.consts, &dense, &hist, &tile);
            out.extend(o);
            hist = advance_conv(&g, &hist, &tile.qkv);
            dense = end;
            t0 = t1;
        }
        self.st.entry = dense;
        self.st.entry_conv = hist.clone();
        self.st.conv_live = hist;
        self.st.position += rows.rows as u64;
        self.st.pending_rows = 0;
        out
    }

    fn seal_on(&self, st: &mut BState) -> SealedState {
        let g = &self.g;
        if st.phase == GdnStreamPhase::Open {
            let end = if self.fault == Fault::NaiveEntryCarry {
                st.entry.clone()
            } else if st.pending_rows > 0 {
                let pend = self.pending_valid(st);
                tile_scan(g, &self.consts, &st.entry, &st.entry_conv, &pend).1
            } else {
                st.entry.clone()
            };
            if st.plane_pos != st.position {
                st.plane = dense_to_plane(g, &end, st.position as u32);
                st.plane_pos = st.position;
            }
            st.phase = GdnStreamPhase::Sealed;
            st.pending_rows = 0;
        }
        sealed_of(g, st.position, &st.plane, &st.conv_live)
    }
}

fn dummy_state() -> BState {
    BState {
        phase: GdnStreamPhase::Sealed,
        position: 0,
        phase_start: 0,
        plane: LiveState::zeros(0),
        plane_pos: 0,
        entry: vec![],
        entry_conv: vec![],
        conv_live: vec![],
        pending: RawRows::empty(),
        pending_rows: 0,
        call_counter: 0,
    }
}

impl GdnStreamUnderTest for BoundedTileStream {
    fn label(&self) -> String {
        format!("bounded-tile-replay[fault={}]", self.fault.name())
    }
    fn geometry(&self) -> StreamGeometry {
        self.g
    }
    fn reset(&mut self, init: &InitialState) -> R<()> {
        let pending = self.poison_pending();
        self.st = BState {
            phase: GdnStreamPhase::Sealed,
            position: init.position,
            phase_start: init.position,
            plane: init.live.clone(),
            plane_pos: init.position,
            entry: vec![],
            entry_conv: vec![],
            conv_live: init.conv.clone(),
            pending,
            pending_rows: 0,
            call_counter: 0,
        };
        Ok(())
    }
    fn meta(&self) -> GdnStreamMeta {
        expected_meta(
            &self.g,
            self.recipe_id,
            self.st.phase,
            self.st.position,
            self.st.phase_start,
            0,
        )
    }
    fn append(&mut self, rows: &RawRows) -> R<Vec<f32>> {
        if rows.rows == 0 {
            return Ok(vec![]);
        }
        Ok(match self.fault {
            Fault::CallEndQuant => self.append_call_end(rows),
            Fault::NaiveEntryCarry => self.append_naive(rows),
            _ => self.append_replay(rows),
        })
    }
    fn seal(&mut self) -> R<SealedState> {
        let mut st = std::mem::replace(&mut self.st, dummy_state());
        let s = self.seal_on(&mut st);
        self.st = st;
        Ok(s)
    }
    fn peek_seal(&mut self) -> R<SealedState> {
        let mut st = self.st.clone();
        Ok(self.seal_on(&mut st))
    }
    fn snapshot(&mut self) -> R<OpaqueSnapshot> {
        let mut snap = self.st.clone();
        match self.fault {
            Fault::MissingPendingRaw => {
                let rows = snap.pending_rows;
                let _ = rows;
                snap.pending = RawRows {
                    rows: self.g.micro_rows,
                    qkv: vec![0.0; self.g.micro_rows * self.g.qkv_width()],
                    alpha: vec![0.0; self.g.micro_rows * self.g.hv],
                    beta: vec![0.0; self.g.micro_rows * self.g.hv],
                };
            }
            Fault::MissingTileEntry => {
                if snap.phase == GdnStreamPhase::Open {
                    snap.entry = plane_to_dense(&self.g, &snap.plane, false);
                }
            }
            Fault::SnapshotDropsEf => {
                snap.plane.ef = vec![0; snap.plane.ef.len()];
            }
            _ => {}
        }
        Ok(OpaqueSnapshot(Box::new(snap)))
    }
    fn restore(&mut self, snap: &OpaqueSnapshot) -> R<()> {
        self.st = snap
            .0
            .downcast_ref::<BState>()
            .ok_or("snapshot type mismatch")?
            .clone();
        Ok(())
    }
    fn clobber_live(&mut self) -> R<()> {
        let g = self.g;
        self.st = BState {
            phase: GdnStreamPhase::Sealed,
            position: 0xDEAD_BEEF,
            phase_start: 0xDEAD,
            plane: garbage_plane(&g),
            plane_pos: 0xBEEF,
            entry: garbage_f32(g.state_elems()),
            entry_conv: garbage_f32(g.conv_hist_rows * g.qkv_width()),
            conv_live: garbage_f32(g.conv_hist_rows * g.qkv_width()),
            pending: self.poison_pending(),
            pending_rows: 777,
            call_counter: 0xFFFF,
        };
        Ok(())
    }
}
