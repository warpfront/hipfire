// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU dispatch for the FLUX **text encoders** — T5-XXL (the `txt` stream)
//! and CLIP-L (the pooled `vec`). Companions of the CPU references in
//! `hipfire_arch_diffusion::{t5, clip}`.
//!
//! Only three ops live here. Everything else the two encoders need is already
//! a shared primitive: [`Gpu::rmsnorm_batched`] is T5's LayerNorm,
//! [`Gpu::layernorm_batched`] is CLIP's affine LayerNorm, and every linear is
//! [`Gpu::gemm_f16_x_f16_wmma_lds_auto`]. Kept in its own file rather than
//! bolted onto `norm.rs`, which is already ~7 kLOC of unrelated element-wise
//! dispatch.
//!
//! These are correctness-first: the encoders run once per prompt (and then
//! get cached), not once per denoise step, so the 55 s of host scalar
//! `nn::linear` they replace is the whole win — a hand-tuned attention kernel
//! at n = 256 would buy nothing measurable on top.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};

/// Workgroup size for [`Gpu::attention_text_f32`]. Must be a power of two:
/// the kernel's max/sum tree reductions halve `blockDim.x` each step.
const ATTN_BLOCK: u32 = 256;

impl Gpu {
    /// Small-sequence f32 self-attention for the text encoders.
    ///
    /// `q`/`k`/`v`/`out` are `[n, heads*hd]` f32, head-interleaved exactly as
    /// the CPU references index them (`x[pos * d + h * hd + t]`), so no
    /// transpose is needed on either side.
    ///
    /// `bias`, when present, is `[heads, n, n]` f32 and is added to the logits
    /// after `scale` — T5's relative-position bias. `causal` masks `k > q`
    /// (CLIP). Padding is NOT masked: both CPU references deliberately ignore
    /// their `attention_mask` (ComfyUI builds T5-XXL with
    /// `enable_attention_masks=False`; diffusers calls `CLIPTextModel` with no
    /// mask), and the goldens were captured against that.
    ///
    /// T5 passes `scale = 1.0` — transformers 5 folds the scaling into the
    /// relative bias and does not scale the dot product. CLIP passes
    /// `1/sqrt(hd)`.
    ///
    /// `k`/`v` rows are `n_kv_heads * hd` wide (GQA): head `h` reads KV head
    /// `h / (heads / n_kv_heads)`. T5 and CLIP pass `n_kv_heads == heads`,
    /// which is exactly the pre-GQA indexing this kernel always used.
    ///
    /// `key_mask`, when present, is F32 `[n]` (1.0 = visible, 0.0 = masked) —
    /// the Qwen3 key-padding mask (Task 15). T5 and CLIP pass `None`: see the
    /// kernel doc comment for why padding is deliberately unmasked there.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_text_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        bias: Option<&GpuTensor>,
        key_mask: Option<&GpuTensor>,
        out: &GpuTensor,
        n: usize,
        heads: usize,
        n_kv_heads: usize,
        hd: usize,
        scale: f32,
        causal: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if n == 0 || heads == 0 || n_kv_heads == 0 || hd == 0 {
            return Err(HipError::new(
                0,
                "attention_text_f32: n/heads/n_kv_heads/hd must be > 0",
            ));
        }
        if heads % n_kv_heads != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "attention_text_f32: heads ({heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
                ),
            ));
        }
        for (name, t) in [("q", q), ("k", k), ("v", v), ("out", out)] {
            if t.dtype != DType::F32 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: {name} dtype must be F32 (got {:?})",
                        t.dtype
                    ),
                ));
            }
        }
        if let Some(b) = bias {
            if b.dtype != DType::F32 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: bias dtype must be F32 (got {:?})",
                        b.dtype
                    ),
                ));
            }
            let need = heads
                .checked_mul(n)
                .and_then(|x| x.checked_mul(n))
                .and_then(|x| x.checked_mul(DType::F32.size()))
                .ok_or_else(|| HipError::new(0, "attention_text_f32: bias size overflow"))?;
            if b.buf.size() < need {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: bias buffer too small (have {} need {need})",
                        b.buf.size()
                    ),
                ));
            }
        }
        if let Some(m) = key_mask {
            if m.dtype != DType::F32 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: key_mask dtype must be F32 (got {:?})",
                        m.dtype
                    ),
                ));
            }
            let need = n
                .checked_mul(DType::F32.size())
                .ok_or_else(|| HipError::new(0, "attention_text_f32: key_mask size overflow"))?;
            if m.buf.size() < need {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: key_mask buffer too small (have {} need {need})",
                        m.buf.size()
                    ),
                ));
            }
        }
        let need_q = n
            .checked_mul(heads)
            .and_then(|x| x.checked_mul(hd))
            .and_then(|x| x.checked_mul(DType::F32.size()))
            .ok_or_else(|| HipError::new(0, "attention_text_f32: q/out size overflow"))?;
        for (name, t) in [("q", q), ("out", out)] {
            if t.buf.size() < need_q {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: {name} buffer too small (have {} need {need_q})",
                        t.buf.size()
                    ),
                ));
            }
        }
        let need_kv = n
            .checked_mul(n_kv_heads)
            .and_then(|x| x.checked_mul(hd))
            .and_then(|x| x.checked_mul(DType::F32.size()))
            .ok_or_else(|| HipError::new(0, "attention_text_f32: k/v size overflow"))?;
        for (name, t) in [("k", k), ("v", v)] {
            if t.buf.size() < need_kv {
                return Err(HipError::new(
                    0,
                    &format!(
                        "attention_text_f32: {name} buffer too small (have {} need {need_kv})",
                        t.buf.size()
                    ),
                ));
            }
        }
        const KERNEL: &str = "attention_t5_bias_f32";
        self.ensure_kernel(KERNEL, kernels::ATTENTION_T5_BIAS_F32_SRC, KERNEL)?;

        let q_ptr = q.buf.as_ptr();
        let k_ptr = k.buf.as_ptr();
        let v_ptr = v.buf.as_ptr();
        // A null bias/key_mask pointer is the kernel's "no additive
        // bias"/"no key-padding mask" signal.
        let b_ptr = bias.map_or(std::ptr::null_mut(), |b| b.buf.as_ptr());
        let m_ptr = key_mask.map_or(std::ptr::null_mut(), |m| m.buf.as_ptr());
        let o_ptr = out.buf.as_ptr();
        let n_i = n as i32;
        let heads_i = heads as i32;
        let n_kv_heads_i = n_kv_heads as i32;
        let hd_i = hd as i32;
        let scale_v = scale;
        let causal_i = i32::from(causal);

        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &k_ptr as *const _ as *mut c_void,
            &v_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &m_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
            &heads_i as *const _ as *mut c_void,
            &n_kv_heads_i as *const _ as *mut c_void,
            &hd_i as *const _ as *mut c_void,
            &scale_v as *const _ as *mut c_void,
            &causal_i as *const _ as *mut c_void,
        ];
        // LDS: the whole prob row plus the reduction scratch.
        let shared = ((n as u32) + ATTN_BLOCK) * 4;
        let bytes = crate::profile::elementwise_bytes(n * heads * hd);
        let timer = crate::profile::begin_timer(&self.hip, "attention", KERNEL, bytes);
        let result = self.launch_maybe_blob(
            KERNEL,
            [n as u32, heads as u32, 1],
            [ATTN_BLOCK, 1, 1],
            shared,
            &mut params,
            || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(q_ptr);
                blob.push_ptr(k_ptr);
                blob.push_ptr(v_ptr);
                blob.push_ptr(b_ptr);
                blob.push_ptr(m_ptr);
                blob.push_ptr(o_ptr);
                blob.push_i32(n_i);
                blob.push_i32(heads_i);
                blob.push_i32(n_kv_heads_i);
                blob.push_i32(hd_i);
                blob.push_f32(scale_v);
                blob.push_i32(causal_i);
                blob
            },
        );
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// T5 v1.1 gated-GELU FFN term: `out[i] = gelu_new(a[i]) * b[i]`, the
    /// middle of `wo(gelu(wi_0 x) * wi_1 x)`. `out` may alias `a` or `b`.
    pub fn gelu_new_mul_f32(
        &mut self,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.elementwise2_f32(
            "gelu_new_mul_f32",
            kernels::GELU_NEW_MUL_F32_SRC,
            a,
            b,
            out,
            n,
        )
    }

    /// OpenAI CLIP quick-GELU: `out[i] = x[i] * sigmoid(1.702 x[i])`.
    /// In-place capable (`out` may alias `x`).
    pub fn quick_gelu_f32(&mut self, x: &GpuTensor, out: &GpuTensor, n: usize) -> HipResult<()> {
        self.bind_thread()?;
        if n == 0 {
            return Err(HipError::new(0, "quick_gelu_f32: n must be > 0"));
        }
        for (name, t) in [("x", x), ("out", out)] {
            if t.dtype != DType::F32 {
                return Err(HipError::new(
                    0,
                    &format!(
                        "quick_gelu_f32: {name} dtype must be F32 (got {:?})",
                        t.dtype
                    ),
                ));
            }
            if t.buf.size() < n * DType::F32.size() {
                return Err(HipError::new(
                    0,
                    &format!(
                        "quick_gelu_f32: {name} buffer too small (have {} need {})",
                        t.buf.size(),
                        n * DType::F32.size()
                    ),
                ));
            }
        }
        const KERNEL: &str = "quick_gelu_f32";
        self.ensure_kernel(KERNEL, kernels::QUICK_GELU_F32_SRC, KERNEL)?;
        let x_ptr = x.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let n_i = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &x_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = n.div_ceil(block as usize) as u32;
        let bytes = crate::profile::elementwise_bytes(n);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", KERNEL, bytes);
        let result =
            self.launch_maybe_blob(KERNEL, [grid, 1, 1], [block, 1, 1], 0, &mut params, || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(x_ptr);
                blob.push_ptr(o_ptr);
                blob.push_i32(n_i);
                blob
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Shared launch body for the two-input element-wise text-encoder
    /// kernels (`(a, b, out, n)` signature, 256-thread flat grid).
    fn elementwise2_f32(
        &mut self,
        kernel: &'static str,
        src: &'static str,
        a: &GpuTensor,
        b: &GpuTensor,
        out: &GpuTensor,
        n: usize,
    ) -> HipResult<()> {
        if n == 0 {
            return Err(HipError::new(0, &format!("{kernel}: n must be > 0")));
        }
        for (name, t) in [("a", a), ("b", b), ("out", out)] {
            if t.dtype != DType::F32 {
                return Err(HipError::new(
                    0,
                    &format!("{kernel}: {name} dtype must be F32 (got {:?})", t.dtype),
                ));
            }
            if t.buf.size() < n * DType::F32.size() {
                return Err(HipError::new(
                    0,
                    &format!(
                        "{kernel}: {name} buffer too small (have {} need {})",
                        t.buf.size(),
                        n * DType::F32.size()
                    ),
                ));
            }
        }
        self.ensure_kernel(kernel, src, kernel)?;
        let a_ptr = a.buf.as_ptr();
        let b_ptr = b.buf.as_ptr();
        let o_ptr = out.buf.as_ptr();
        let n_i = n as i32;
        let mut params: Vec<*mut c_void> = vec![
            &a_ptr as *const _ as *mut c_void,
            &b_ptr as *const _ as *mut c_void,
            &o_ptr as *const _ as *mut c_void,
            &n_i as *const _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = n.div_ceil(block as usize) as u32;
        let bytes = crate::profile::elementwise_bytes(n);
        let timer = crate::profile::begin_timer(&self.hip, "elementwise", kernel, bytes);
        let result =
            self.launch_maybe_blob(kernel, [grid, 1, 1], [block, 1, 1], 0, &mut params, || {
                let mut blob = hip_bridge::KernargBlob::new();
                blob.push_ptr(a_ptr);
                blob.push_ptr(b_ptr);
                blob.push_ptr(o_ptr);
                blob.push_i32(n_i);
                blob
            });
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn text_attention_kernel_takes_kv_heads_and_key_mask() {
        let s = crate::kernels::ATTENTION_T5_BIAS_F32_SRC;
        assert!(s.contains("int n_kv_heads") && s.contains("const float* __restrict__ key_mask"));
    }
}
