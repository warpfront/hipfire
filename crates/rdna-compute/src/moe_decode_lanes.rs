// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lane-batched twins of the A3B (`qwen3_5_moe`, `.mq4rs`) decode MoE block for
//! the VMM continuous-batching exact route, exact gfx1201.
//!
//! After the caller's batched post-attention rmsnorm+rotate and the four
//! batched projections (router, selector, shared gate, shared up), the
//! singleton decode runs, per layer and per lane,
//!
//! 1. `moe_router_softmax_topk_k8_wave64`
//! 2. `fused_silu_mul_mq_rotate` (K = 512, shared expert)
//! 3. `gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2` (shared down, residual)
//! 4. `gemv_mq4g256v2_moe_gate_up_k8_indexed`
//! 5. `gemv_mq4g128v2_moe_ninepath_d4` (routed down + combine, residual)
//!
//! [`Gpu::moe_decode_lanes_mq4rs`] launches those five for `lanes` (1..=8)
//! lanes in that stream order, one launch per step. Steps 1, 3, 4, 5 are the
//! singleton kernels' own bodies (`kernels/src/moe_decode_lanes_mq4rs.gfx1201.hip`
//! wraps each as an inlined device function behind a lane-indexed entry, so
//! per-lane arithmetic, reduction order, FMA contraction and compiler flags are
//! the singleton's by construction); step 2 reuses the existing grid-folded
//! batched launch of the very same `fused_silu_mul_mq_rotate` kernel. The lab
//! test `tests/moe_decode_lanes.rs` proves every lane's bytes (and every
//! scratch buffer) equal the real singleton launches on random and adversarial
//! inputs.
//!
//! With 2+ lanes steps 4 and 5 run expert-deduplicating kernels instead of the
//! twins (`kernels/src/moe_dedup_mq4rs.gfx1201.hip`): across the up to 64
//! (lane, slot) picks of a layer-step only ~46 experts are distinct on a real
//! decode batch, so each distinct expert's weights are read once and every
//! pair that selected it is computed from that read, per column with the
//! singleton kernel's arithmetic (same bytes as the twins, proven by
//! `tests/moe_dedup.rs`). The routed down is two launches: stage A writes each
//! pair's pre-combine slot parts into the expanded scratch
//! ([`MoeDecodeLanesArgs::down_expanded`]), stage B is the singleton's
//! ascending-slot weighted fold into the residual. `HIPFIRE_A3B_MOE_DEDUP=0`
//! keeps the twins; `HIPFIRE_A3B_MOE_DEDUP_DOWN=0` keeps the twin for the
//! routed down alone ([`MoeDedup`]).
//!
//! Layout contract: row-major `[lanes][len]` buffers (the
//! `PrefillBatchScratch` fields), see [`MoeDecodeLanesArgs`].

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

/// Most lanes one launch covers.
pub const MOE_LANES_MAX: usize = 8;
/// Production geometry the twins are compiled for.
pub const MOE_LANES_HIDDEN: usize = 2048;
pub const MOE_LANES_N_EXP: usize = 256;
pub const MOE_LANES_TOPK: usize = 8;
pub const MOE_LANES_MI: usize = 512;
pub const MOE_LANES_SHARED_MI: usize = 512;

/// Module (and entry) names of the four new twins.
pub const MOE_ROUTER_ROWS: &str = "moe_router_softmax_topk_k8_wave64_rows";
pub const MOE_SHARED_DOWN_ROWS: &str = "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_rows";
pub const MOE_GATE_UP_ROWS: &str = "gemv_mq4g256v2_moe_gate_up_k8_indexed_rows";
pub const MOE_NINEPATH_ROWS: &str = "gemv_mq4g128v2_moe_ninepath_d4_rows";

/// The four module names, in launch order.
pub const MOE_LANE_MODULES: [&str; 4] = [
    MOE_ROUTER_ROWS,
    MOE_SHARED_DOWN_ROWS,
    MOE_GATE_UP_ROWS,
    MOE_NINEPATH_ROWS,
];

/// Expert-deduplicating routed kernels (one module, three entries): gate/up
/// and routed down stage A read each distinct expert's weights once for all
/// (lane, slot) pairs that selected it; stage B is the singleton's combine.
pub const MOE_DEDUP_MODULE: &str = "moe_dedup_mq4rs_gfx1201";
pub const MOE_GATE_UP_DEDUP_ROWS: &str = "gemv_mq4g256v2_moe_gate_up_k8_dedup_rows";
pub const MOE_DOWN_DEDUP_ROWS: &str = "gemv_mq4g128v2_moe_down_dedup_rows";
pub const MOE_DOWN_COMBINE_ROWS: &str = "gemv_mq4g128v2_moe_down_combine_rows";
/// The module's device symbols.
pub const MOE_DEDUP_SYMBOLS: [&str; 3] = [
    MOE_GATE_UP_DEDUP_ROWS,
    MOE_DOWN_DEDUP_ROWS,
    MOE_DOWN_COMBINE_ROWS,
];
/// Gate/up row pairs of one dedup wave (`MD_GU_ROWS` in the kernel).
pub const MOE_DEDUP_GATE_UP_ROWS: usize = 4;
/// Output rows of one stage-A down workgroup (`MD_DOWN_ROWS` in the kernel).
pub const MOE_DEDUP_DOWN_ROWS: usize = 16;
/// Dedup module source.
pub const MOE_DEDUP_SRC: &str = include_str!("../../../kernels/src/moe_dedup_mq4rs.gfx1201.hip");

/// Which routed stages use the expert-deduplicating kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeDedup {
    pub gate_up: bool,
    pub down: bool,
}

impl MoeDedup {
    pub const OFF: Self = Self {
        gate_up: false,
        down: false,
    };
    pub const ALL: Self = Self {
        gate_up: true,
        down: true,
    };

    /// `HIPFIRE_A3B_MOE_DEDUP` kill switch (developer, read once per
    /// process): on unless `0`; `0` keeps the non-dedup lane twins for the
    /// routed gate/up and down + combine. `HIPFIRE_A3B_MOE_DEDUP_DOWN=0`
    /// (same reading) keeps the lane twin for the down + combine alone.
    pub fn from_env() -> Self {
        static CHOICE: std::sync::LazyLock<MoeDedup> = std::sync::LazyLock::new(|| {
            let on = |name: &str| hipfire_config::developer_var(name).as_deref() != Ok("0");
            let all = on("HIPFIRE_A3B_MOE_DEDUP");
            MoeDedup {
                gate_up: all,
                down: all && on("HIPFIRE_A3B_MOE_DEDUP_DOWN"),
            }
        });
        *CHOICE
    }
}

fn lane_err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

/// Assemble a lane twin module: the singleton source with its entry header
/// (`header`, found exactly once, including any `__launch_bounds__` and the
/// `extern "C" __global__` qualifiers) rewritten to an inlined device function
/// `{kernel}_row_body(`, then the wrapper TU selected by `twin_define`. Any
/// drift in a singleton's entry header makes assembly fail instead of building
/// a different kernel.
fn lane_twin_source(src: &str, header: &str, kernel: &str, twin_define: &str) -> HipResult<String> {
    if src.matches(header).count() != 1 {
        return Err(lane_err(format!(
            "moe lane twin {kernel}: singleton entry header not found exactly once"
        )));
    }
    let body = src.replacen(
        header,
        &format!("__device__ __forceinline__ void {kernel}_row_body("),
        1,
    );
    Ok(format!(
        "{body}\n#define {twin_define} 1\n{}",
        kernels::MOE_DECODE_LANES_MQ4RS_GFX1201_SRC
    ))
}

/// Module source by twin name (also used by the kernel registry).
pub fn moe_lane_module_source(module: &str) -> HipResult<String> {
    match module {
        MOE_ROUTER_ROWS => lane_twin_source(
            kernels::MOE_ROUTER_SOFTMAX_TOPK_K8_WAVE64_SRC,
            "extern \"C\" __launch_bounds__(64, 4)\n__global__ void moe_router_softmax_topk_k8_wave64(",
            "moe_router_softmax_topk_k8_wave64",
            "HIPFIRE_MOE_LANES_ROUTER",
        ),
        MOE_SHARED_DOWN_ROWS => lane_twin_source(
            kernels::GEMV_MQ4G256V2_RESIDUAL_SIGMOID_SCALED_K512_R2_SRC,
            "__launch_bounds__(32, 16)\nextern \"C\" __global__ void HIPFIRE_MQ4G256V2_K512_R2_KERNEL(",
            "gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2",
            "HIPFIRE_MOE_LANES_SHARED_DOWN",
        ),
        MOE_GATE_UP_ROWS => lane_twin_source(
            kernels::GEMV_MQ4G256V2_MOE_GATE_UP_K8_INDEXED_SRC,
            "#if defined(HIPFIRE_MQ4V2_GATE_UP_NOLDS)\n__launch_bounds__(32, 8)\n#else\n__launch_bounds__(32, 16)\n#endif\nextern \"C\" __global__ void HIPFIRE_MQ4V2_GATE_UP_KERNEL(",
            "gemv_mq4g256v2_moe_gate_up_k8_indexed",
            "HIPFIRE_MOE_LANES_GATE_UP",
        ),
        MOE_NINEPATH_ROWS => lane_twin_source(
            kernels::GEMV_MQ4G128V2_MOE_NINEPATH_D4_SRC,
            "__launch_bounds__(256, 4)\nextern \"C\" __global__ void HIPFIRE_MQ4G128V2_NINEPATH_KERNEL(",
            "gemv_mq4g128v2_moe_ninepath_d4",
            "HIPFIRE_MOE_LANES_NINEPATH",
        ),
        other => Err(lane_err(format!("unknown moe lane twin {other}"))),
    }
}

/// Buffers of one lane-batched MoE block step. Every `[lanes][len]` tensor is
/// row-major f32 (i32 for the top-k ids) with lane `r` at element offset
/// `r * len`; each must hold at least `lanes * len` elements.
pub struct MoeDecodeLanesArgs<'a> {
    /// `[lanes][256]` router logits (batched projection output).
    pub router_logits: &'a GpuTensor,
    /// `[lanes][1]` shared-expert selector logit (batched projection output).
    pub shared_scalar: &'a GpuTensor,
    /// `[lanes][512]` shared gate / up (batched projection outputs).
    pub shared_gate: &'a GpuTensor,
    pub shared_up: &'a GpuTensor,
    /// `[lanes][512]` OUT: rotated `silu(shared_gate) * shared_up`.
    pub shared_rot: &'a GpuTensor,
    /// `[lanes][8]` OUT: top-8 expert ids (i32) / weights (f32).
    pub topk_idx: &'a GpuTensor,
    pub topk_w: &'a GpuTensor,
    /// `[lanes][8][512]` OUT: routed gate / up.
    pub gate_batch: &'a GpuTensor,
    pub up_batch: &'a GpuTensor,
    /// `[lanes][2048]` rotated post-attention activation (the projections'
    /// input; the routed gate/up GEMV reads it).
    pub x_rot_batch: &'a GpuTensor,
    /// `[lanes][2048]` residual stream: the shared down then the routed
    /// nine-path accumulate into it (read-modify-write).
    pub x_batch: &'a GpuTensor,
    /// Shared down weights, one MQ4G256V2 matrix `[2048 rows][K 512]` shared by
    /// all lanes.
    pub shared_down_w: &'a GpuTensor,
    /// Routed gate/up pointer table (MQ4G256V2, `n_exp` u64) and down pointer
    /// table (MQ4G128V2, `n_exp` u64), shared by all lanes.
    pub gate_up_ptrs: &'a GpuTensor,
    pub down_ptrs: &'a GpuTensor,
    /// `[lanes][8][2048]` f32 scratch of the expert-deduplicating routed down
    /// (`PrefillBatchScratch::moe_down_expanded_batch`): every (lane, slot)
    /// pair's pre-combine down dot products. Written only on the dedup path.
    pub down_expanded: &'a GpuTensor,
    /// Number of experts (256).
    pub n_exp: usize,
    /// Renormalize the selected top-k weights.
    pub norm_topk: bool,
}

impl MoeDecodeLanesArgs<'_> {
    fn validate(&self, lanes: usize) -> HipResult<()> {
        if self.n_exp != MOE_LANES_N_EXP {
            return Err(lane_err("moe_decode_lanes_mq4rs: requires n_exp == 256"));
        }
        // (name, bytes held, bytes needed)
        let n = lanes;
        let spans: [(&str, usize, usize); 15] = [
            ("router_logits", self.router_logits.buf.size(), n * 256 * 4),
            ("shared_scalar", self.shared_scalar.buf.size(), n * 4),
            ("shared_gate", self.shared_gate.buf.size(), n * 512 * 4),
            ("shared_up", self.shared_up.buf.size(), n * 512 * 4),
            ("shared_rot", self.shared_rot.buf.size(), n * 512 * 4),
            ("topk_idx", self.topk_idx.buf.size(), n * 8 * 4),
            ("topk_w", self.topk_w.buf.size(), n * 8 * 4),
            ("gate_batch", self.gate_batch.buf.size(), n * 8 * 512 * 4),
            ("up_batch", self.up_batch.buf.size(), n * 8 * 512 * 4),
            (
                "x_rot_batch",
                self.x_rot_batch.buf.size(),
                n * MOE_LANES_HIDDEN * 4,
            ),
            ("x_batch", self.x_batch.buf.size(), n * MOE_LANES_HIDDEN * 4),
            (
                "shared_down_w",
                self.shared_down_w.buf.size(),
                2048 * 2 * 136,
            ),
            ("gate_up_ptrs", self.gate_up_ptrs.buf.size(), 256 * 8),
            ("down_ptrs", self.down_ptrs.buf.size(), 256 * 8),
            (
                "down_expanded",
                self.down_expanded.buf.size(),
                n * 8 * MOE_LANES_HIDDEN * 4,
            ),
        ];
        for (what, have, need) in spans {
            if have < need {
                return Err(lane_err(format!(
                    "moe_decode_lanes_mq4rs: operand span too small: {what} holds {have} B, needs {need} B for {lanes} lane(s)"
                )));
            }
        }
        Ok(())
    }
}

impl Gpu {
    fn ensure_moe_lane_twin(&mut self, module: &'static str) -> HipResult<()> {
        if !self.functions.contains_key(module) {
            let src = moe_lane_module_source(module)?;
            self.ensure_kernel(module, &src, module)?;
        }
        Ok(())
    }

    /// Whether [`Self::moe_decode_lanes_mq4rs`] is admitted on this device:
    /// exact gfx1201 and `lanes` in `1..=8`. The geometry (hidden 2048, 256
    /// experts, top-8, moe/shared intermediate 512) is fixed by the kernels.
    pub fn moe_decode_lanes_admitted(&self, lanes: usize) -> bool {
        self.arch_caps.is_gfx1201() && (1..=MOE_LANES_MAX).contains(&lanes)
    }

    /// Lane-batched A3B MoE block, after the caller's batched projections:
    /// router top-k, shared silu+rotate, shared down (+residual), routed
    /// gate/up, routed down + combine (+residual), for `lanes` lanes, in the
    /// singleton's launch order. Lane `r`'s results (`topk_idx`/`topk_w` rows,
    /// `shared_rot`, `gate_batch`/`up_batch`, `x_batch` row) are byte-identical
    /// to the five singleton launches run on lane `r`'s rows alone. With 2+
    /// lanes the routed gate/up and down + combine use the expert-deduplicating
    /// kernels (one weight read per distinct expert, same bytes) unless
    /// `HIPFIRE_A3B_MOE_DEDUP=0`.
    pub fn moe_decode_lanes_mq4rs(
        &mut self,
        a: &MoeDecodeLanesArgs<'_>,
        lanes: usize,
    ) -> HipResult<()> {
        self.moe_decode_lanes_mq4rs_with(a, lanes, MoeDedup::from_env())
    }

    /// [`Self::moe_decode_lanes_mq4rs`] with the dedup choice explicit (lab
    /// entry; ignored for one lane).
    pub fn moe_decode_lanes_mq4rs_with(
        &mut self,
        a: &MoeDecodeLanesArgs<'_>,
        lanes: usize,
        dedup: MoeDedup,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(lane_err("moe_decode_lanes_mq4rs: requires exact gfx1201"));
        }
        if !(1..=MOE_LANES_MAX).contains(&lanes) {
            return Err(lane_err(format!(
                "moe_decode_lanes_mq4rs: lanes must be in 1..={MOE_LANES_MAX}, got {lanes}"
            )));
        }
        a.validate(lanes)?;
        self.ensure_mq_signs()?;
        self.ensure_mq_signs_128()?;
        self.ensure_moe_lane_twin(MOE_ROUTER_ROWS)?;
        self.ensure_moe_lane_twin(MOE_SHARED_DOWN_ROWS)?;
        let n_lanes = lanes as u32;

        // 1. router top-k: grid = lanes, one wave64 per lane.
        {
            let lp = a.router_logits.buf.as_ptr();
            let ip = a.topk_idx.buf.as_ptr();
            let wp = a.topk_w.buf.as_ptr();
            let n = a.n_exp as i32;
            let nr = i32::from(a.norm_topk);
            let mut params: Vec<*mut c_void> = vec![
                &lp as *const _ as *mut c_void,
                &ip as *const _ as *mut c_void,
                &wp as *const _ as *mut c_void,
                &n as *const _ as *mut c_void,
                &nr as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                MOE_ROUTER_ROWS,
                [n_lanes, 1, 1],
                [64, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(lp);
                    b.push_ptr(ip);
                    b.push_ptr(wp);
                    b.push_i32(n);
                    b.push_i32(nr);
                    b
                },
            )?;
        }

        // 2. shared silu + rotate: the singleton kernel's own grid-folded
        // batched launch (grid.x = lane * 2 + group).
        self.fused_silu_mul_rotate_mq_batched(
            a.shared_gate,
            a.shared_up,
            a.shared_rot,
            MOE_LANES_SHARED_MI,
            lanes,
        )?;

        // 3. shared down (+ sigmoid-gated residual): grid (M/2, 1, lanes).
        {
            let a_ptr = a.shared_down_w.buf.as_ptr();
            let x_ptr = a.shared_rot.buf.as_ptr();
            let y_ptr = a.x_batch.buf.as_ptr();
            let c_ptr = a.shared_scalar.buf.as_ptr();
            let m = MOE_LANES_HIDDEN as i32;
            let k = MOE_LANES_SHARED_MI as i32;
            let mut params: Vec<*mut c_void> = vec![
                &a_ptr as *const _ as *mut c_void,
                &x_ptr as *const _ as *mut c_void,
                &y_ptr as *const _ as *mut c_void,
                &c_ptr as *const _ as *mut c_void,
                &m as *const _ as *mut c_void,
                &k as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                MOE_SHARED_DOWN_ROWS,
                [(MOE_LANES_HIDDEN / 2) as u32, 1, n_lanes],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(a_ptr);
                    b.push_ptr(x_ptr);
                    b.push_ptr(y_ptr);
                    b.push_ptr(c_ptr);
                    b.push_i32(m);
                    b.push_i32(k);
                    b
                },
            )?;
        }

        // 4. routed gate/up and 5. routed down + combine (+ residual): the
        // expert-deduplicating kernels when 2+ lanes and `dedup` (one lane has
        // nothing to dedup), else the lane twins.
        let gate_up = dedup.gate_up && lanes >= 2;
        let down = dedup.down && lanes >= 2;
        self.moe_routed_gate_up_launch(a, lanes, gate_up)?;
        self.moe_routed_down_launch(a, lanes, down)
    }

    /// Routed gate/up of `lanes` lanes alone (the block's step 4), with the
    /// expert-deduplicating kernel (`dedup`) or the lane twin of the singleton
    /// kernel. Both write byte-identical `gate_batch`/`up_batch`; lab entry.
    pub fn moe_routed_gate_up_lanes(
        &mut self,
        a: &MoeDecodeLanesArgs<'_>,
        lanes: usize,
        dedup: bool,
    ) -> HipResult<()> {
        self.moe_routed_prologue(a, lanes)?;
        self.moe_routed_gate_up_launch(a, lanes, dedup)
    }

    /// Routed down + combine of `lanes` lanes alone (the block's step 5), with
    /// the expert-deduplicating kernel pair (`dedup`) or the lane twin of the
    /// singleton nine-path kernel. Both accumulate byte-identical rows into
    /// `x_batch`; lab entry.
    pub fn moe_routed_down_lanes(
        &mut self,
        a: &MoeDecodeLanesArgs<'_>,
        lanes: usize,
        dedup: bool,
    ) -> HipResult<()> {
        self.moe_routed_prologue(a, lanes)?;
        self.moe_routed_down_launch(a, lanes, dedup)
    }

    fn moe_routed_prologue(&mut self, a: &MoeDecodeLanesArgs<'_>, lanes: usize) -> HipResult<()> {
        self.bind_thread()?;
        if !self.arch_caps.is_gfx1201() {
            return Err(lane_err("moe routed lanes: requires exact gfx1201"));
        }
        if !(1..=MOE_LANES_MAX).contains(&lanes) {
            return Err(lane_err(format!(
                "moe routed lanes: lanes must be in 1..={MOE_LANES_MAX}, got {lanes}"
            )));
        }
        a.validate(lanes)?;
        self.ensure_mq_signs_128()
    }

    fn ensure_moe_dedup(&mut self, symbol: &'static str) -> HipResult<()> {
        self.ensure_kernel(MOE_DEDUP_MODULE, MOE_DEDUP_SRC, symbol)
    }

    /// Step 4 launch (operands validated by the caller).
    fn moe_routed_gate_up_launch(
        &mut self,
        a: &MoeDecodeLanesArgs<'_>,
        lanes: usize,
        dedup: bool,
    ) -> HipResult<()> {
        let n_lanes = lanes as u32;
        let pp = a.gate_up_ptrs.buf.as_ptr();
        let ip = a.topk_idx.buf.as_ptr();
        let xp = a.x_rot_batch.buf.as_ptr();
        let ygp = a.gate_batch.buf.as_ptr();
        let yup = a.up_batch.buf.as_ptr();
        // The expert count is the table size / 8, exactly as the singleton
        // launcher derives it.
        let n_experts = (a.gate_up_ptrs.buf.size() / 8) as i32;
        if dedup {
            self.ensure_moe_dedup(MOE_GATE_UP_DEDUP_ROWS)?;
            let nl = lanes as i32;
            let mut params: Vec<*mut c_void> = vec![
                &pp as *const _ as *mut c_void,
                &ip as *const _ as *mut c_void,
                &xp as *const _ as *mut c_void,
                &ygp as *const _ as *mut c_void,
                &yup as *const _ as *mut c_void,
                &nl as *const _ as *mut c_void,
                &n_experts as *const _ as *mut c_void,
            ];
            return self.launch_maybe_blob(
                MOE_GATE_UP_DEDUP_ROWS,
                [(MOE_LANES_MI / MOE_DEDUP_GATE_UP_ROWS) as u32, 8, n_lanes],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(pp);
                    b.push_ptr(ip);
                    b.push_ptr(xp);
                    b.push_ptr(ygp);
                    b.push_ptr(yup);
                    b.push_i32(nl);
                    b.push_i32(n_experts);
                    b
                },
            );
        }
        self.ensure_moe_lane_twin(MOE_GATE_UP_ROWS)?;
        // grid (M/2, 8, lanes), M = 2 * 512, K = 2048.
        let m = (2 * MOE_LANES_MI) as i32;
        let k = MOE_LANES_HIDDEN as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &xp as *const _ as *mut c_void,
            &ygp as *const _ as *mut c_void,
            &yup as *const _ as *mut c_void,
            &m as *const _ as *mut c_void,
            &k as *const _ as *mut c_void,
            &n_experts as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            MOE_GATE_UP_ROWS,
            [MOE_LANES_MI as u32, 8, n_lanes],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(xp);
                b.push_ptr(ygp);
                b.push_ptr(yup);
                b.push_i32(m);
                b.push_i32(k);
                b.push_i32(n_experts);
                b
            },
        )
    }

    /// Step 5 launch(es) (operands validated by the caller).
    fn moe_routed_down_launch(
        &mut self,
        a: &MoeDecodeLanesArgs<'_>,
        lanes: usize,
        dedup: bool,
    ) -> HipResult<()> {
        let n_lanes = lanes as u32;
        let pp = a.down_ptrs.buf.as_ptr();
        let ip = a.topk_idx.buf.as_ptr();
        let wp = a.topk_w.buf.as_ptr();
        let gp = a.gate_batch.buf.as_ptr();
        let up_ptr = a.up_batch.buf.as_ptr();
        let s1 = self.scratch.mq_signs1_128.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2_128.as_ref().unwrap().buf.as_ptr();
        let op = a.x_batch.buf.as_ptr();
        let n_experts = a.n_exp as i32;
        if dedup {
            self.ensure_moe_dedup(MOE_DOWN_DEDUP_ROWS)?;
            self.ensure_moe_dedup(MOE_DOWN_COMBINE_ROWS)?;
            let ex = a.down_expanded.buf.as_ptr();
            let nl = lanes as i32;
            // Stage A: every distinct expert's slot parts, one weight read.
            let mut params: Vec<*mut c_void> = vec![
                &pp as *const _ as *mut c_void,
                &ip as *const _ as *mut c_void,
                &gp as *const _ as *mut c_void,
                &up_ptr as *const _ as *mut c_void,
                &s1 as *const _ as *mut c_void,
                &s2 as *const _ as *mut c_void,
                &ex as *const _ as *mut c_void,
                &nl as *const _ as *mut c_void,
                &n_experts as *const _ as *mut c_void,
            ];
            self.launch_maybe_blob(
                MOE_DOWN_DEDUP_ROWS,
                [(MOE_LANES_HIDDEN / MOE_DEDUP_DOWN_ROWS) as u32, 8, n_lanes],
                [32, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(pp);
                    b.push_ptr(ip);
                    b.push_ptr(gp);
                    b.push_ptr(up_ptr);
                    b.push_ptr(s1);
                    b.push_ptr(s2);
                    b.push_ptr(ex);
                    b.push_i32(nl);
                    b.push_i32(n_experts);
                    b
                },
            )?;
            // Stage B: the singleton's ascending-slot weighted fold into the
            // residual, per lane.
            let mut params: Vec<*mut c_void> = vec![
                &wp as *const _ as *mut c_void,
                &ex as *const _ as *mut c_void,
                &op as *const _ as *mut c_void,
            ];
            return self.launch_maybe_blob(
                MOE_DOWN_COMBINE_ROWS,
                [(MOE_LANES_HIDDEN / 256) as u32, n_lanes, 1],
                [256, 1, 1],
                0,
                &mut params,
                || {
                    let mut b = hip_bridge::KernargBlob::new();
                    b.push_ptr(wp);
                    b.push_ptr(ex);
                    b.push_ptr(op);
                    b
                },
            );
        }
        self.ensure_moe_lane_twin(MOE_NINEPATH_ROWS)?;
        // grid (down_m / 16, lanes).
        let dm = MOE_LANES_HIDDEN as i32;
        let dk = MOE_LANES_MI as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &ip as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &gp as *const _ as *mut c_void,
            &up_ptr as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &dm as *const _ as *mut c_void,
            &dk as *const _ as *mut c_void,
            &n_experts as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            MOE_NINEPATH_ROWS,
            [(MOE_LANES_HIDDEN / 16) as u32, n_lanes, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(ip);
                b.push_ptr(wp);
                b.push_ptr(gp);
                b.push_ptr(up_ptr);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(dm);
                b.push_i32(dk);
                b.push_i32(n_experts);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_lane_twin_module_assembles_with_its_wrapper() {
        for module in MOE_LANE_MODULES {
            let src = moe_lane_module_source(module).expect(module);
            assert!(src.contains("_row_body("), "{module}: body not wrapped");
            assert!(
                src.contains(&format!("void {module}(")),
                "{module}: lane entry missing"
            );
            // The singleton's own entry is gone: no __global__ before the
            // selected wrapper (the wrapper TU holds one entry per twin, three
            // of them compiled out).
            let split = src.find("#define HIPFIRE_MOE_LANES_").expect(module);
            assert_eq!(src[..split].matches("__global__").count(), 0, "{module}");
        }
    }

    #[test]
    fn router_twin_keeps_the_wave64_translation_unit_flag() {
        let src = moe_lane_module_source(MOE_ROUTER_ROWS).unwrap();
        assert!(src.contains("// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64"));
        for module in &MOE_LANE_MODULES[1..] {
            assert!(
                !moe_lane_module_source(module)
                    .unwrap()
                    .contains("HIPFIRE_COMPILER_FLAGS:"),
                "{module} must add no compiler flags (the singleton adds none)"
            );
        }
    }

    #[test]
    fn unknown_module_is_refused() {
        assert!(moe_lane_module_source("nope").is_err());
    }
}
