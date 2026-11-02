// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Rows-batched twins of singleton decode kernels for continuous batching.
//!
//! Each twin compiles the singleton kernel's own source with its entry
//! rewritten into an inlined device function (`rows_twin_source`) and adds a
//! wrapper (`kernels/src/rows_batched_twins.hip`) that runs one request row
//! per `blockIdx.z`, taking that row's pointers from a row table. The wrapper
//! passes exactly the arguments a per-row singleton launch passes, with the
//! singleton's grid x/y and block, so each row is the singleton by
//! construction. Any change to a singleton's entry signature, entry macro or
//! edited line makes module assembly fail instead of building a different
//! kernel.
//!
//! Use: `stage_*_rows` validates the rows against the plan and uploads the
//! table into caller-owned device staging (outside graph capture); `*_rows`
//! launches from that staging.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const ROWS_TWINS_SRC: &str = include_str!("../../../kernels/src/rows_batched_twins.hip");

fn twin_err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

/// Assemble a rows-batched twin module.
///
/// * `entry_macro`/`kernel`: the source's `#define {entry_macro} {kernel}\n`
///   line (exactly one) is rewritten to `{kernel}_row_body`;
/// * every `extern "C" [__launch_bounds__(..)] __global__ void {entry_macro}(`
///   (exactly `entries` of them) becomes
///   `__device__ __forceinline__ void {entry_macro}(`;
/// * each `(from, to)` in `edits` must occur exactly once;
/// * `#define {twin_define} 1` and `#define {twin_define}_ENTRY {kernel}_rows`
///   select the wrapper appended from `rows_batched_twins.hip`.
pub fn rows_twin_source(
    src: &str,
    entry_macro: &str,
    kernel: &str,
    entries: usize,
    edits: &[(&str, &str)],
    twin_define: &str,
) -> HipResult<String> {
    let define = format!("#define {entry_macro} {kernel}\n");
    if src.matches(&define).count() != 1 {
        return Err(twin_err(format!(
            "rows twin {kernel}: entry define `{}` not found exactly once",
            define.trim()
        )));
    }
    let mut out = src.replacen(
        &define,
        &format!("#define {entry_macro} {kernel}_row_body\n"),
        1,
    );
    let sig = format!("__global__ void {entry_macro}(");
    let mut rewritten = 0usize;
    let mut search_from = 0usize;
    while let Some(rel) = out[search_from..].find(&sig) {
        let at = search_from + rel;
        let ext = out[..at]
            .rfind("extern \"C\"")
            .ok_or_else(|| twin_err(format!("rows twin {kernel}: entry without extern \"C\"")))?;
        let between = out[ext + "extern \"C\"".len()..at].trim();
        if !(between.is_empty()
            || (between.starts_with("__launch_bounds__(")
                && between.ends_with(')')
                && !between.contains(';')))
        {
            return Err(twin_err(format!(
                "rows twin {kernel}: unexpected entry qualifiers `{between}`"
            )));
        }
        let replacement = format!("__device__ __forceinline__ void {entry_macro}(");
        out.replace_range(ext..at + sig.len(), &replacement);
        rewritten += 1;
        search_from = ext + replacement.len();
    }
    if rewritten != entries {
        return Err(twin_err(format!(
            "rows twin {kernel}: found {rewritten} entry signatures, expected {entries}"
        )));
    }
    for (from, to) in edits {
        if out.matches(from).count() != 1 {
            return Err(twin_err(format!(
                "rows twin {kernel}: checked edit `{from}` not found exactly once"
            )));
        }
        out = out.replacen(from, to, 1);
    }
    Ok(format!(
        "#define {twin_define} 1\n#define {twin_define}_ENTRY {kernel}_rows\n{out}\n{ROWS_TWINS_SRC}"
    ))
}

/// One row's device byte ranges the twin WRITES; rows of one launch must
/// never overlap (state/output bleed between requests).
type WriteExtent = (u64, usize, &'static str);

/// Host gate before staging a row table: one tag per row equal to the
/// plan's tag for that row (wrong-row / stale-epoch), non-null written
/// pointers, and pairwise-disjoint written ranges across rows.
pub fn validate_rows(
    kernel: &str,
    row_tags: &[u64],
    plan_tags: &[u64],
    writes: &[Vec<WriteExtent>],
) -> Result<(), String> {
    if row_tags.len() != plan_tags.len() || row_tags.len() != writes.len() {
        return Err(format!(
            "{kernel} rows: tags={} plan={} rows={} disagree",
            row_tags.len(),
            plan_tags.len(),
            writes.len()
        ));
    }
    if writes.is_empty() {
        return Err(format!("{kernel} rows: empty batch"));
    }
    for (r, (t, p)) in row_tags.iter().zip(plan_tags).enumerate() {
        if t != p {
            return Err(format!("{kernel} row {r}: row tag {t} != plan tag {p}"));
        }
    }
    let mut all: Vec<(u64, u64, usize, &str)> = vec![];
    for (r, extents) in writes.iter().enumerate() {
        for &(base, len, what) in extents {
            if base == 0 {
                return Err(format!("{kernel} row {r}: null {what} pointer"));
            }
            if len > 0 {
                all.push((base, base + len as u64, r, what));
            }
        }
    }
    all.sort_unstable();
    for w in all.windows(2) {
        let (a, b) = (w[0], w[1]);
        if b.0 < a.1 && a.2 != b.2 {
            return Err(format!(
                "{kernel}: row {} {} [0x{:x},0x{:x}) overlaps row {} {} at 0x{:x}",
                a.2, a.3, a.0, a.1, b.2, b.3, b.0
            ));
        }
    }
    Ok(())
}

fn as_bytes<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: row structs are repr(C) and consist only of u64 fields.
    unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, std::mem::size_of_val(rows)) }
}

macro_rules! row_struct {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* $f:ident),* $(,)? }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
        pub struct $name { $($(#[$fm])* pub $f: u64,)* }
    };
}

row_struct!(
    /// `HfGdnRow`: device pointers of one request's GDN step.
    GdnRow { q, k, v, gate, beta, s_q8, s_scales, output,
        /// 0 = no error feedback, exactly as the singleton's `None`.
        s_ef_residual }
);
row_struct!(
    /// `HfConvRow`: one request's conv+SiLU+QK-norm step.
    ConvRow { q_out, k_out, v_out, input, state }
);
row_struct!(
    /// `HfGatedNormRow`: one request's gated norm + MQ rotate.
    GatedNormRow { x, z, x_rot }
);
row_struct!(
    /// `HfFaPrepRow`: one request's FA prep; `pos` = its position word.
    FaPrepRow { q_interleaved, q, gate, k, pos }
);

const _: () = assert!(std::mem::size_of::<GdnRow>() == 72);
const _: () = assert!(std::mem::size_of::<ConvRow>() == 40);
const _: () = assert!(std::mem::size_of::<GatedNormRow>() == 24);
const _: () = assert!(std::mem::size_of::<FaPrepRow>() == 40);

/// gfx1201 27B GDN head geometry the compact3 twin admits.
pub const GDN_HD: usize = 128;

/// Module names of the 27B twins.
pub const GDN_COMPACT3_B2_ROWS: &str = "gated_delta_net_q8_compact3_b2_rows";
pub const CONV_QKNORM_B256_ROWS: &str = "conv1d_silu_split_qknorm_b256_rows";
pub const GATED_NORM_AWQ_K6144_ROWS: &str = "gated_norm_mq_rotate_awq_k6144_gfx1201_rows";
pub const QWEN36_27B_FA_PREP_ROWS: &str = "qwen36_27b_fa_prep_gfx1201_rows";
pub const QWEN35_A3B_FA_PREP_ROWS: &str = "qwen35_fa_prep_gfx1201_rows";

/// Module names of the A3B (32 value heads, 16 QK heads, HD 128, K = 4096)
/// DeltaNet twins; the conv twin is the shared [`CONV_QKNORM_B256_ROWS`].
pub const GDN_COMPACT2_B2_ROWS: &str = "gated_delta_net_q8_compact2_b2_rows";
pub const GATED_NORM_K4096_ROWS: &str = "gated_norm_mq_rotate_gfx1201_rows";

/// Whether the singleton A3B DeltaNet launchers run the kernels the A3B rows
/// twins are built from: `gated_delta_net_q8_compact2_b2` (no
/// `HIPFIRE_GDN_COMPACT2_SHAPE` override) and `conv1d_silu_split_qknorm_b256`
/// (no `HIPFIRE_CONV_QKNORM_SHAPE` override).
pub fn a3b_dn_default_kernel_shapes() -> bool {
    let unset_or = |var: &str, ok: &str| {
        hipfire_config::developer_var(var)
            .ok()
            .is_none_or(|v| v == ok)
    };
    unset_or("HIPFIRE_GDN_COMPACT2_SHAPE", "b2") && unset_or("HIPFIRE_CONV_QKNORM_SHAPE", "b256")
}

/// Twin module source by twin name (also used by any precompile registry).
pub fn rows_twin_module_source(twin: &str) -> HipResult<String> {
    match twin {
        GDN_COMPACT3_B2_ROWS => rows_twin_source(
            kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC,
            "HIPFIRE_GDN_KERNEL",
            "gated_delta_net_q8_compact3_b2",
            2,
            &[(
                "    const int tile = blockIdx.y;\n    const int lane = blockIdx.z;\n#endif",
                "    const int tile = blockIdx.y;\n    const int lane = 0;\n#endif",
            )],
            "HIPFIRE_ROWS_GDN",
        ),
        CONV_QKNORM_B256_ROWS => rows_twin_source(
            kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC,
            "HIPFIRE_CQN_KERNEL",
            "conv1d_silu_split_qknorm_b256",
            1,
            &[],
            "HIPFIRE_ROWS_CQN",
        ),
        GATED_NORM_AWQ_K6144_ROWS => rows_twin_source(
            kernels::gated_norm_mq_rotate_awq_k6144_gfx1201_src(),
            "HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL",
            "gated_norm_mq_rotate_awq_k6144_gfx1201",
            1,
            &[],
            "HIPFIRE_ROWS_GATED_NORM",
        ),
        QWEN36_27B_FA_PREP_ROWS => rows_twin_source(
            kernels::qwen36_27b_fa_prep_gfx1201_src(),
            "HIPFIRE_QWEN35_FA_PREP_KERNEL",
            "qwen36_27b_fa_prep_gfx1201",
            1,
            &[],
            "HIPFIRE_ROWS_FA_PREP",
        ),
        GDN_COMPACT2_B2_ROWS => rows_twin_source(
            kernels::GATED_DELTA_NET_Q8_COMPACT2_B2_SRC,
            "HIPFIRE_GDN_KERNEL",
            "gated_delta_net_q8_compact2_b2",
            2,
            &[(
                "    const int tile = blockIdx.y;\n    const int lane = blockIdx.z;\n#endif",
                "    const int tile = blockIdx.y;\n    const int lane = 0;\n#endif",
            )],
            "HIPFIRE_ROWS_GDN",
        ),
        GATED_NORM_K4096_ROWS => rows_twin_source(
            kernels::GATED_NORM_MQ_ROTATE_GFX1201_SRC,
            "HIPFIRE_GATED_NORM_MQ_ROTATE_KERNEL",
            "gated_norm_mq_rotate_gfx1201",
            1,
            &[],
            "HIPFIRE_ROWS_GATED_NORM",
        ),
        QWEN35_A3B_FA_PREP_ROWS => rows_twin_source(
            kernels::QWEN35_FA_PREP_GFX1201_SRC,
            "HIPFIRE_QWEN35_FA_PREP_KERNEL",
            "qwen35_fa_prep_gfx1201",
            1,
            &[],
            "HIPFIRE_ROWS_FA_PREP",
        ),
        other => Err(twin_err(format!("unknown rows twin {other}"))),
    }
}

/// Kernarg byte offset of `frame_base` in the GDN twin: table pointer, then
/// `n_tokens`, `n_heads`, `head_dim`.
const GDN_ROWS_FRAME_KERNARG_OFFSET: u32 = 8 + 3 * 4;

impl Gpu {
    fn ensure_rows_twin(&mut self, twin: &'static str) -> HipResult<()> {
        if !self.arch_caps.is_gfx1201() {
            return Err(twin_err(format!(
                "{twin}: rows twins are certified on gfx1201 only"
            )));
        }
        if !self.functions.contains_key(twin) {
            let src = rows_twin_module_source(twin)?;
            self.ensure_kernel(twin, &src, twin)?;
        }
        Ok(())
    }

    fn stage_rows_table<T: Copy>(
        &self,
        what: &str,
        table: &GpuTensor,
        rows: &[T],
    ) -> HipResult<()> {
        let bytes = as_bytes(rows);
        if table.buf.size() < bytes.len() {
            return Err(twin_err(format!(
                "{what}: row table staging holds {} B, needs {} B",
                table.buf.size(),
                bytes.len()
            )));
        }
        if self.graphs.capture_mode {
            return Err(twin_err(format!(
                "{what}: row table staging during graph capture"
            )));
        }
        self.bind_thread()?;
        self.hip.memcpy_htod(&table.buf, bytes)
    }

    // ── GatedDeltaNet compact3 (27B: 48 value heads, 16 QK heads, HD 128) ──
    // ── and compact2 (A3B: 32 value heads, 16 QK heads, HD 128) ──

    /// Validate and stage GDN rows. `n_tokens` per row as the singleton.
    fn stage_gdn_rows(
        &self,
        table: &GpuTensor,
        rows: &[GdnRow],
        row_tags: &[u64],
        plan_tags: &[u64],
        n_tokens: usize,
        n_heads: usize,
    ) -> HipResult<()> {
        let st = n_heads * GDN_HD * GDN_HD;
        let writes: Vec<Vec<WriteExtent>> = rows
            .iter()
            .map(|r| {
                let mut w = vec![
                    (r.s_q8, st, "s_q8"),
                    (r.s_scales, n_heads * GDN_HD * 4, "s_scales"),
                    (r.output, n_tokens * n_heads * GDN_HD * 4, "output"),
                ];
                if r.s_ef_residual != 0 {
                    w.push((r.s_ef_residual, st * 2, "s_ef_residual"));
                }
                w
            })
            .collect();
        for (i, r) in rows.iter().enumerate() {
            if [r.q, r.k, r.v, r.gate, r.beta].contains(&0) {
                return Err(twin_err(format!(
                    "gdn rows: row {i} has a null input pointer"
                )));
            }
        }
        validate_rows("gdn rows", row_tags, plan_tags, &writes).map_err(twin_err)?;
        self.stage_rows_table("gdn rows", table, rows)
    }

    /// Launch GDN twin `module` (compact QK ratio `qk_div`) over `rows` rows
    /// of an already staged table, reserving `rows` requant frames.
    #[allow(clippy::too_many_arguments)]
    fn launch_gdn_rows(
        &mut self,
        module: &'static str,
        qk_div: usize,
        table: &GpuTensor,
        rows: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        if head_dim != GDN_HD || n_heads % qk_div != 0 || rows == 0 || rows > u16::MAX as usize {
            return Err(twin_err(format!(
                "gdn rows: requires head_dim 128, n_heads % {qk_div} == 0, 1..=65535 rows"
            )));
        }
        self.bind_thread()?;
        self.ensure_rows_twin(module)?;
        let tp = table.buf.as_ptr();
        let nt = n_tokens as i32;
        let nh = n_heads as i32;
        let hd = head_dim as i32;
        let fr = crate::norm::reserve_gdn_requant_frames(rows as u32) as i32;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &nt as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &fr as *const _ as *mut c_void,
        ];
        let declared = [railgun::kernel::DeclaredWord {
            offset: GDN_ROWS_FRAME_KERNARG_OFFSET,
            role: railgun::kernel::WordRole::GdnFrame {
                frames: rows as u32,
            },
        }];
        self.launch_maybe_blob_declared(
            module,
            [n_heads as u32, (GDN_HD / 4) as u32, rows as u32],
            [32, 1, 1],
            0,
            &mut params,
            &declared,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nt);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_i32(fr);
                b
            },
        )
    }

    /// Validate and stage GDN compact3 rows. `n_tokens` per row as the
    /// singleton.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_gated_delta_net_q8_compact3_b2_rows(
        &self,
        table: &GpuTensor,
        rows: &[GdnRow],
        row_tags: &[u64],
        plan_tags: &[u64],
        n_tokens: usize,
        n_heads: usize,
    ) -> HipResult<()> {
        self.stage_gdn_rows(table, rows, row_tags, plan_tags, n_tokens, n_heads)
    }

    /// Rows-batched `gated_delta_net_q8_compact3_b2`: row r == the singleton
    /// `gated_delta_net_q8_compact(.., qk_head_div=3, ..)` on row r's state,
    /// with requant frame `base + r` (reserves `rows` frames, as `rows`
    /// back-to-back singleton launches would).
    pub fn gated_delta_net_q8_compact3_b2_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.launch_gdn_rows(
            GDN_COMPACT3_B2_ROWS,
            3,
            table,
            rows,
            n_tokens,
            n_heads,
            head_dim,
        )
    }

    /// Validate and stage GDN compact2 rows (A3B). Same row layout and
    /// checks as the compact3 staging.
    pub fn stage_gated_delta_net_q8_compact2_b2_rows(
        &self,
        table: &GpuTensor,
        rows: &[GdnRow],
        row_tags: &[u64],
        plan_tags: &[u64],
        n_tokens: usize,
        n_heads: usize,
    ) -> HipResult<()> {
        self.stage_gdn_rows(table, rows, row_tags, plan_tags, n_tokens, n_heads)
    }

    /// Rows-batched `gated_delta_net_q8_compact2_b2` (A3B: 32 value heads,
    /// 16 QK heads): row r == the singleton
    /// `gated_delta_net_q8_compact(.., qk_head_div=2, ..)` on row r's state,
    /// with requant frame `base + r` (reserves `rows` frames, as `rows`
    /// back-to-back singleton launches would).
    pub fn gated_delta_net_q8_compact2_b2_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        self.launch_gdn_rows(
            GDN_COMPACT2_B2_ROWS,
            2,
            table,
            rows,
            n_tokens,
            n_heads,
            head_dim,
        )
    }

    // ── conv1d + SiLU + split + QK norm, block 256 ──

    #[allow(clippy::too_many_arguments)]
    pub fn stage_conv1d_silu_split_qknorm_b256_rows(
        &self,
        table: &GpuTensor,
        rows: &[ConvRow],
        row_tags: &[u64],
        plan_tags: &[u64],
        k_dim: usize,
        v_dim: usize,
    ) -> HipResult<()> {
        let writes: Vec<Vec<WriteExtent>> = rows
            .iter()
            .map(|r| {
                vec![
                    (r.q_out, k_dim * 4, "q_out"),
                    (r.k_out, k_dim * 4, "k_out"),
                    (r.v_out, v_dim * 4, "v_out"),
                    (r.state, (2 * k_dim + v_dim) * 3 * 4, "state"),
                ]
            })
            .collect();
        if let Some(i) = rows.iter().position(|r| r.input == 0) {
            return Err(twin_err(format!(
                "conv rows: row {i} has a null input pointer"
            )));
        }
        validate_rows("conv rows", row_tags, plan_tags, &writes).map_err(twin_err)?;
        self.stage_rows_table("conv rows", table, rows)
    }

    /// Rows-batched `conv1d_silu_split_qknorm_b256` (the default
    /// `conv1d_silu_split_qknorm` shape); `weight` is shared by all rows.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_silu_split_qknorm_b256_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        weight: &GpuTensor,
        k_dim: usize,
        v_dim: usize,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
        eps: f32,
    ) -> HipResult<()> {
        if rows == 0 || rows > u16::MAX as usize {
            return Err(twin_err("conv rows: 1..=65535 rows"));
        }
        self.bind_thread()?;
        self.ensure_rows_twin(CONV_QKNORM_B256_ROWS)?;
        let block = 256u32;
        let grid = n_heads as u32 + (v_dim as u32).div_ceil(block);
        let tp = table.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let (kd, vd, nh, hd) = (k_dim as i32, v_dim as i32, n_heads as i32, head_dim as i32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &kd as *const _ as *mut c_void,
            &vd as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &q_scale as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            CONV_QKNORM_B256_ROWS,
            [grid, 1, rows as u32],
            [block, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(wp);
                b.push_i32(kd);
                b.push_i32(vd);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(q_scale);
                b.push_f32(eps);
                b
            },
        )
    }

    // ── gated norm + MQ rotate, AWQ, 48 heads x 128 (gfx1201) ──

    pub fn stage_gated_norm_mq_rotate_awq_k6144_rows(
        &self,
        table: &GpuTensor,
        rows: &[GatedNormRow],
        row_tags: &[u64],
        plan_tags: &[u64],
    ) -> HipResult<()> {
        let writes: Vec<Vec<WriteExtent>> = rows
            .iter()
            .map(|r| vec![(r.x_rot, 48 * 128 * 4, "x_rot")])
            .collect();
        if let Some(i) = rows.iter().position(|r| r.x == 0 || r.z == 0) {
            return Err(twin_err(format!(
                "gated norm rows: row {i} has a null input pointer"
            )));
        }
        validate_rows("gated norm rows", row_tags, plan_tags, &writes).map_err(twin_err)?;
        self.stage_rows_table("gated norm rows", table, rows)
    }

    /// Rows-batched `gated_norm_mq_rotate_awq_k6144_gfx1201` (the
    /// `gated_norm_rotate_mq_gfx1100` route for 48 heads + AWQ on gfx1201).
    pub fn gated_norm_mq_rotate_awq_k6144_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        weight: &GpuTensor,
        awq_scale: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        const N_HEADS: usize = 48;
        const HEAD_DIM: usize = 128;
        if rows == 0 || rows > u16::MAX as usize {
            return Err(twin_err("gated norm rows: 1..=65535 rows"));
        }
        if awq_scale.numel() < N_HEADS * HEAD_DIM || weight.numel() < HEAD_DIM {
            return Err(twin_err("gated norm rows: undersized weight/awq_scale"));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_rows_twin(GATED_NORM_AWQ_K6144_ROWS)?;
        let tp = table.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let ap = awq_scale.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let (nh, hd) = (N_HEADS as i32, HEAD_DIM as i32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &ap as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            GATED_NORM_AWQ_K6144_ROWS,
            [(N_HEADS / 2) as u32, 1, rows as u32],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(wp);
                b.push_ptr(ap);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(eps);
                b
            },
        )
    }

    // ── gated norm + MQ rotate, no AWQ, 32 heads x 128 (A3B, gfx1201) ──

    pub fn stage_gated_norm_mq_rotate_k4096_rows(
        &self,
        table: &GpuTensor,
        rows: &[GatedNormRow],
        row_tags: &[u64],
        plan_tags: &[u64],
    ) -> HipResult<()> {
        let writes: Vec<Vec<WriteExtent>> = rows
            .iter()
            .map(|r| vec![(r.x_rot, 32 * 128 * 4, "x_rot")])
            .collect();
        if let Some(i) = rows.iter().position(|r| r.x == 0 || r.z == 0) {
            return Err(twin_err(format!(
                "gated norm rows: row {i} has a null input pointer"
            )));
        }
        validate_rows("gated norm k4096 rows", row_tags, plan_tags, &writes).map_err(twin_err)?;
        self.stage_rows_table("gated norm k4096 rows", table, rows)
    }

    /// Rows-batched `gated_norm_mq_rotate_gfx1201` (the
    /// `gated_norm_rotate_mq_gfx1100` route for 32 heads without AWQ on
    /// gfx1201): row r's `x_rot` is the singleton's output for row r's
    /// `x`/`z`.
    pub fn gated_norm_mq_rotate_k4096_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        weight: &GpuTensor,
        eps: f32,
    ) -> HipResult<()> {
        const N_HEADS: usize = 32;
        const HEAD_DIM: usize = 128;
        if rows == 0 || rows > u16::MAX as usize {
            return Err(twin_err("gated norm rows: 1..=65535 rows"));
        }
        if weight.numel() < HEAD_DIM {
            return Err(twin_err("gated norm rows: undersized weight"));
        }
        self.bind_thread()?;
        self.ensure_mq_signs()?;
        self.ensure_rows_twin(GATED_NORM_K4096_ROWS)?;
        let tp = table.buf.as_ptr();
        let wp = weight.buf.as_ptr();
        let s1 = self.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr();
        let s2 = self.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr();
        let (nh, hd) = (N_HEADS as i32, HEAD_DIM as i32);
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &wp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            GATED_NORM_K4096_ROWS,
            [(N_HEADS / 2) as u32, 1, rows as u32],
            [64, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(wp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_i32(nh);
                b.push_i32(hd);
                b.push_f32(eps);
                b
            },
        )
    }

    // ── Qwen3.6-27B FA prep (24Q / 4K, head_dim 256, n_rot 64) ──

    pub fn stage_qwen36_27b_fa_prep_rows(
        &self,
        table: &GpuTensor,
        rows: &[FaPrepRow],
        row_tags: &[u64],
        plan_tags: &[u64],
    ) -> HipResult<()> {
        let writes: Vec<Vec<WriteExtent>> = rows
            .iter()
            .map(|r| {
                vec![
                    (r.q, 24 * 256 * 4, "q"),
                    (r.gate, 24 * 256 * 4, "gate"),
                    (r.k, 4 * 256 * 4, "k"),
                ]
            })
            .collect();
        if let Some(i) = rows.iter().position(|r| r.q_interleaved == 0 || r.pos == 0) {
            return Err(twin_err(format!(
                "fa prep rows: row {i} has a null input pointer"
            )));
        }
        validate_rows("fa prep rows", row_tags, plan_tags, &writes).map_err(twin_err)?;
        self.stage_rows_table("fa prep rows", table, rows)
    }

    /// Rows-batched `qwen36_27b_fa_prep_gfx1201` (`qwen35_fa_prep_gfx1100`
    /// with 24Q/4K on gfx1201).
    pub fn qwen36_27b_fa_prep_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        q_weight: &GpuTensor,
        k_weight: &GpuTensor,
        eps: f32,
        freq_base: f32,
    ) -> HipResult<()> {
        if rows == 0 || rows > u16::MAX as usize {
            return Err(twin_err("fa prep rows: 1..=65535 rows"));
        }
        self.bind_thread()?;
        self.ensure_rows_twin(QWEN36_27B_FA_PREP_ROWS)?;
        let tp = table.buf.as_ptr();
        let qw = q_weight.buf.as_ptr();
        let kw = k_weight.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &qw as *const _ as *mut c_void,
            &kw as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &freq_base as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            QWEN36_27B_FA_PREP_ROWS,
            [28, 1, rows as u32],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(qw);
                b.push_ptr(kw);
                b.push_f32(eps);
                b.push_f32(freq_base);
                b
            },
        )
    }

    // ── A3B FA prep (16Q / 2K, head_dim 256, n_rot 64) ──

    pub fn stage_qwen35_a3b_fa_prep_rows(
        &self,
        table: &GpuTensor,
        rows: &[FaPrepRow],
        row_tags: &[u64],
        plan_tags: &[u64],
    ) -> HipResult<()> {
        let writes: Vec<Vec<WriteExtent>> = rows
            .iter()
            .map(|r| {
                vec![
                    (r.q, 16 * 256 * 4, "q"),
                    (r.gate, 16 * 256 * 4, "gate"),
                    (r.k, 2 * 256 * 4, "k"),
                ]
            })
            .collect();
        if let Some(i) = rows.iter().position(|r| r.q_interleaved == 0 || r.pos == 0) {
            return Err(twin_err(format!(
                "a3b fa prep rows: row {i} has a null input pointer"
            )));
        }
        validate_rows("a3b fa prep rows", row_tags, plan_tags, &writes).map_err(twin_err)?;
        self.stage_rows_table("a3b fa prep rows", table, rows)
    }

    /// Rows-batched `qwen35_fa_prep_gfx1201` (`qwen35_fa_prep_gfx1100` with
    /// 16Q/2K on gfx1201): one 18-workgroup grid per row.
    pub fn qwen35_a3b_fa_prep_rows(
        &mut self,
        table: &GpuTensor,
        rows: usize,
        q_weight: &GpuTensor,
        k_weight: &GpuTensor,
        eps: f32,
        freq_base: f32,
    ) -> HipResult<()> {
        if rows == 0 || rows > u16::MAX as usize {
            return Err(twin_err("a3b fa prep rows: 1..=65535 rows"));
        }
        self.bind_thread()?;
        self.ensure_rows_twin(QWEN35_A3B_FA_PREP_ROWS)?;
        let tp = table.buf.as_ptr();
        let qw = q_weight.buf.as_ptr();
        let kw = k_weight.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &qw as *const _ as *mut c_void,
            &kw as *const _ as *mut c_void,
            &eps as *const _ as *mut c_void,
            &freq_base as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            QWEN35_A3B_FA_PREP_ROWS,
            [18, 1, rows as u32],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(qw);
                b.push_ptr(kw);
                b.push_f32(eps);
                b.push_f32(freq_base);
                b
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_twin_modules_assemble() {
        for t in [
            GDN_COMPACT3_B2_ROWS,
            CONV_QKNORM_B256_ROWS,
            GATED_NORM_AWQ_K6144_ROWS,
            QWEN36_27B_FA_PREP_ROWS,
            GDN_COMPACT2_B2_ROWS,
            GATED_NORM_K4096_ROWS,
            QWEN35_A3B_FA_PREP_ROWS,
        ] {
            let s = rows_twin_module_source(t).unwrap();
            assert!(s.contains(&format!("_ENTRY {t}\n")), "{t}");
            assert!(!s.contains("__global__ void HIPFIRE_GDN_KERNEL("), "{t}");
        }
    }

    #[test]
    fn signature_change_refused() {
        let changed = kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC
            .replace("__launch_bounds__(HIPFIRE_CQN_BLOCK)\n", "");
        assert!(rows_twin_source(
            &changed,
            "HIPFIRE_CQN_KERNEL",
            "conv1d_silu_split_qknorm_b256",
            1,
            &[],
            "X"
        )
        .is_ok());
        let renamed = kernels::CONV1D_SILU_SPLIT_QKNORM_B256_SRC
            .replace("HIPFIRE_CQN_KERNEL conv1d", "HIPFIRE_CQN_KERNEL conv2d");
        assert!(rows_twin_source(
            &renamed,
            "HIPFIRE_CQN_KERNEL",
            "conv1d_silu_split_qknorm_b256",
            1,
            &[],
            "X"
        )
        .is_err());
        assert!(rows_twin_source(
            kernels::GATED_DELTA_NET_Q8_COMPACT3_B2_SRC,
            "HIPFIRE_GDN_KERNEL",
            "gated_delta_net_q8_compact3_b2",
            2,
            &[("no such line", "")],
            "X"
        )
        .is_err());
    }

    #[test]
    fn overlap_and_tags_refused() {
        let ok = vec![vec![(0x1000, 0x100, "s")], vec![(0x1100, 0x100, "s")]];
        assert!(validate_rows("t", &[1, 2], &[1, 2], &ok).is_ok());
        let bleed = vec![vec![(0x1000, 0x100, "s")], vec![(0x10ff, 0x100, "s")]];
        assert!(validate_rows("t", &[1, 2], &[1, 2], &bleed).is_err());
        assert!(validate_rows("t", &[1, 3], &[1, 2], &ok).is_err());
        assert!(validate_rows(
            "t",
            &[1, 2],
            &[1, 2],
            &[vec![(0, 4, "s")], vec![(8, 4, "s")]]
        )
        .is_err());
    }
}
