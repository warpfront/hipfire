// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Segment twins of singleton verify kernels for cross-request batched
//! speculative verify.
//!
//! A singleton verify launches one kernel per request over that request's
//! `n` verify rows. Each twin compiles the singleton kernel's own source
//! with its entry rewritten into an inlined device function and appends a
//! wrapper (`kernels/src/verify_segment_twins.hip`) that runs one request's
//! row segment per `blockIdx.z`, taking that segment's pointers and row
//! count from a table, with the singleton launch's grid x/y and block. Each
//! segment is therefore the singleton launch by construction; ragged
//! segments size grid y to the longest and exit the surplus blocks. A twin
//! may carry checked source edits that change only operand loads, never an
//! arithmetic operation or its order (`ATTN_FP8_LOAD_EDITS`).
//!
//! Use: `stage_*_segs` validates and uploads the tables (outside graph
//! capture), `*_segs` launches from a staged table.

use crate::{kernels, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};
use std::ffi::c_void;

const SEG_TWINS_SRC: &str = include_str!("../../../kernels/src/verify_segment_twins.hip");

fn seg_err(msg: impl AsRef<str>) -> HipError {
    HipError::new(0, msg.as_ref())
}

/// Rewrite the singleton entry `extern "C" __global__ void {kernel}(` (it
/// must occur exactly once) into `__device__ __forceinline__ void
/// {kernel}_seg_body(`, apply `edits` (each `from` must occur exactly once)
/// and append the wrapper selected by `twin_define`.
pub fn seg_twin_source(
    src: &str,
    kernel: &str,
    twin_define: &str,
    edits: &[(&str, &str)],
) -> HipResult<String> {
    let entry = format!("extern \"C\" __global__ void {kernel}(");
    if src.matches(&entry).count() != 1 {
        return Err(seg_err(format!(
            "segment twin {kernel}: entry `{entry}` not found exactly once"
        )));
    }
    let mut body = src.replacen(
        &entry,
        &format!("__device__ __forceinline__ void {kernel}_seg_body("),
        1,
    );
    for (from, to) in edits {
        if body.matches(from).count() != 1 {
            return Err(seg_err(format!(
                "segment twin {kernel}: checked edit `{from}` not found exactly once"
            )));
        }
        body = body.replacen(from, to, 1);
    }
    Ok(format!("#define {twin_define} 1\n{body}\n{SEG_TWINS_SRC}"))
}

/// The fp8 scalar-batched attention's two inner loops issue one dependent
/// global byte load per step, so the singleton is load-latency bound. These
/// edits keep every arithmetic operation and its order (same expressions,
/// same sequential accumulation) and change only how the operands are
/// loaded: K codes in 8-byte words (when aligned), V codes and scales
/// hoisted eight positions ahead of their in-order accumulation.
const ATTN_FP8_LOAD_EDITS: &[(&str, &str)] = &[
    (
        "        for (int j = 0; j < head_dim; j++)\n            dot += q_shared[j] * (ks * hipfire_fp8_e4m3_to_f32_batched(kc[j]));\n",
        "        if ((head_dim & 7) == 0 && (((unsigned long long)kc) & 7) == 0) {
            const uint2* kc8 = (const uint2*)kc;
            const int words = head_dim >> 3;
            for (int c = 0; c < words; c += 4) {
                uint2 w[4];
                #pragma unroll
                for (int u = 0; u < 4; u++) w[u] = (c + u < words) ? kc8[c + u] : make_uint2(0u, 0u);
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    if (c + u < words) {
                        #pragma unroll
                        for (int b = 0; b < 8; b++) {
                            const int j = ((c + u) << 3) + b;
                            const unsigned int word = b < 4 ? w[u].x : w[u].y;
                            const unsigned char code = (unsigned char)(word >> ((b & 3) * 8));
                            dot += q_shared[j] * (ks * hipfire_fp8_e4m3_to_f32_batched(code));
                        }
                    }
                }
            }
        } else {
            for (int j = 0; j < head_dim; j++)
                dot += q_shared[j] * (ks * hipfire_fp8_e4m3_to_f32_batched(kc[j]));
        }
",
    ),
    (
        "        for (int t = 0; t < eff_seq_len; t++) {
            const unsigned char* vrow = v_cache + kv_offset_for_v(desc, t, per_pos_bytes);
            const float vs = (float)*((const _Float16*)(vrow + scale_off));
            val += scores[t] * (vs * hipfire_fp8_e4m3_to_f32_batched(vrow[k_head_off + d]));
        }
",
        "        int t = 0;
        for (; t + 8 <= eff_seq_len; t += 8) {
            float vsu[8];
            unsigned char vbu[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                const unsigned char* vrow = v_cache + kv_offset_for_v(desc, t + u, per_pos_bytes);
                vsu[u] = (float)*((const _Float16*)(vrow + scale_off));
                vbu[u] = vrow[k_head_off + d];
            }
            #pragma unroll
            for (int u = 0; u < 8; u++)
                val += scores[t + u] * (vsu[u] * hipfire_fp8_e4m3_to_f32_batched(vbu[u]));
        }
        for (; t < eff_seq_len; t++) {
            const unsigned char* vrow = v_cache + kv_offset_for_v(desc, t, per_pos_bytes);
            const float vs = (float)*((const _Float16*)(vrow + scale_off));
            val += scores[t] * (vs * hipfire_fp8_e4m3_to_f32_batched(vrow[k_head_off + d]));
        }
",
    ),
];

/// `HfAttnFp8Seg`: one request's `attention_fp8_e4m3_kv_batched` launch.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttnFp8Seg {
    pub q: u64,
    pub k_cache: u64,
    pub v_cache: u64,
    pub out: u64,
    pub positions: u64,
    pub n_rows: u64,
}
const _: () = assert!(std::mem::size_of::<AttnFp8Seg>() == 48);

/// Module/function name of the fp8 scalar-batched attention twin.
pub const ATTN_FP8_KV_BATCHED_SEGS: &str = "attention_fp8_e4m3_kv_batched_segs";

/// Twin module source by name.
pub fn seg_twin_module_source(twin: &str) -> HipResult<String> {
    match twin {
        ATTN_FP8_KV_BATCHED_SEGS => {
            // Same source the singleton launcher compiles: kv_slot_desc.h
            // stripped and prepended.
            let stripped = kernels::ATTENTION_FP8_E4M3_KV_BATCHED_SRC
                .replace("#include \"kv_slot_desc.h\"", "");
            let src = format!("{}\n{}", kernels::KV_SLOT_DESC_H, stripped);
            seg_twin_source(
                &src,
                "attention_fp8_e4m3_kv_batched",
                "HIPFIRE_SEGS_ATTN_FP8",
                ATTN_FP8_LOAD_EDITS,
            )
        }
        other => Err(seg_err(format!("unknown segment twin {other}"))),
    }
}

/// Singleton block size of `attention_fp8_e4m3_kv_batched` for a launch
/// with `max_ctx_len` (the singleton launcher's formula).
fn attn_batched_block(max_ctx_len: usize, head_dim: usize) -> u32 {
    (max_ctx_len.max(head_dim) as u32)
        .next_power_of_two()
        .min(256)
}

fn as_bytes<T: Copy>(rows: &[T]) -> &[u8] {
    // SAFETY: segment structs are repr(C) and consist only of u64 fields.
    unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, std::mem::size_of_val(rows)) }
}

impl Gpu {
    fn ensure_seg_twin(&mut self, twin: &'static str) -> HipResult<()> {
        if !self.arch_caps.is_gfx1201() {
            return Err(seg_err(format!(
                "{twin}: segment twins are certified on gfx1201 only"
            )));
        }
        if !self.functions.contains_key(twin) {
            let src = seg_twin_module_source(twin)?;
            self.ensure_kernel(twin, &src, twin)?;
        }
        Ok(())
    }

    /// Validate fp8 attention segments (non-null pointers, pairwise-disjoint
    /// outputs) and upload them to `table` at segment offset `at`.
    pub fn stage_attention_fp8_e4m3_kv_batched_segs(
        &self,
        table: &GpuTensor,
        at: usize,
        segs: &[AttnFp8Seg],
        n_heads: usize,
        head_dim: usize,
    ) -> HipResult<()> {
        let mut outs: Vec<(u64, u64)> = Vec::with_capacity(segs.len());
        for (i, s) in segs.iter().enumerate() {
            if [s.q, s.k_cache, s.v_cache, s.out, s.positions].contains(&0) || s.n_rows == 0 {
                return Err(seg_err(format!(
                    "attn fp8 segs: segment {i} has a null pointer or no rows"
                )));
            }
            outs.push((s.out, s.out + s.n_rows * (n_heads * head_dim * 4) as u64));
        }
        outs.sort_unstable();
        if outs.windows(2).any(|w| w[1].0 < w[0].1) {
            return Err(seg_err("attn fp8 segs: overlapping segment outputs"));
        }
        let bytes = as_bytes(segs);
        let off = at * std::mem::size_of::<AttnFp8Seg>();
        if table.buf.size() < off + bytes.len() {
            return Err(seg_err(format!(
                "attn fp8 segs: table holds {} B, needs {} B",
                table.buf.size(),
                off + bytes.len()
            )));
        }
        if self.graphs.capture_mode {
            return Err(seg_err("attn fp8 segs: table staging during graph capture"));
        }
        self.bind_thread()?;
        self.hip.memcpy_htod_offset(&table.buf, off, bytes)
    }

    /// Segment-batched `attention_fp8_e4m3_kv_batched`: segment `z` (table
    /// entries `at..at + segs`) == the singleton launch over its rows with
    /// `max_ctx_lens[z]`. Refuses segments whose singleton block size
    /// differs (one launch has one block size).
    #[allow(clippy::too_many_arguments)]
    pub fn attention_fp8_e4m3_kv_batched_segs(
        &mut self,
        table: &GpuTensor,
        at: usize,
        max_rows: usize,
        max_ctx_lens: &[usize],
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> HipResult<()> {
        let segs = max_ctx_lens.len();
        if segs == 0 || max_rows == 0 || segs > u16::MAX as usize {
            return Err(seg_err(
                "attn fp8 segs: requires 1..=65535 segments and rows",
            ));
        }
        let block = attn_batched_block(max_ctx_lens[0], head_dim);
        if max_ctx_lens
            .iter()
            .any(|&c| attn_batched_block(c, head_dim) != block)
        {
            return Err(seg_err(
                "attn fp8 segs: segments disagree on the singleton block size",
            ));
        }
        let max_ctx = *max_ctx_lens.iter().max().expect("non-empty");
        self.bind_thread()?;
        self.ensure_seg_twin(ATTN_FP8_KV_BATCHED_SEGS)?;
        let tp =
            unsafe { (table.buf.as_ptr() as *mut u8).add(at * std::mem::size_of::<AttnFp8Seg>()) }
                as *mut c_void;
        let nh = n_heads as i32;
        let nkv = n_kv_heads as i32;
        let hd = head_dim as i32;
        let ms = max_seq as i32;
        let sc = 1.0f32 / (head_dim as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &nh as *const _ as *mut c_void,
            &nkv as *const _ as *mut c_void,
            &hd as *const _ as *mut c_void,
            &ms as *const _ as *mut c_void,
            &sc as *const _ as *mut c_void,
        ];
        let shared_mem = ((max_ctx + block as usize + head_dim) * 4) as u32;
        self.launch_maybe_blob(
            ATTN_FP8_KV_BATCHED_SEGS,
            [n_heads as u32, max_rows as u32, segs as u32],
            [block, 1, 1],
            shared_mem,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_i32(nh);
                b.push_i32(nkv);
                b.push_i32(hd);
                b.push_i32(ms);
                b.push_f32(sc);
                b
            },
        )
    }
}
