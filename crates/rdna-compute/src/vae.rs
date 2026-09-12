// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU dispatch for the FLUX VAE decoder kernels (see `kernels/src/vae_*.hip`).
//!
//! These are the GPU companions of the CPU reference in
//! `hipfire_arch_diffusion::vae`. They operate on f32, channel-major
//! `[c][h][w]` tensors and are written correctness-first (one thread per
//! output element, plain FMA accumulation) so that the latent->pixels decode
//! matches the CPU reference and ComfyUI before any WMMA/tiled optimization.
//! The CPU decode of a 1024x1024 image is single-threaded and takes minutes;
//! these kernels bring that to GPU speed while preserving the exact summation
//! structure needed for a parity gate.

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};

/// Workgroup tile for `vae_im2col_f16_lds`: `(c_tile, th, tw)`.
///
/// `c_tile` sets the length of the contiguous store run (`c_tile*9` halves
/// per output pixel), so bigger is better for the write side, which is 18 of
/// the ~22 bytes moved per input element. It must divide `c_in` to avoid
/// partial tiles — the kernel handles a partial tile correctly, this only
/// keeps the runs full-width. Every real FLUX VAE conv has `c_in` in
/// {16, 128, 256, 512}, and the GEMM route already requires `c_in % 16 == 0`
/// (K = c_in*9 must be a multiple of 16 and 9 is odd), so the search below
/// always lands on 64, 32 or 16.
///
/// LDS is `c_tile*(th+2)*(tw+2)*2` bytes; the default 64/8/16 is 22.5 KB,
/// which leaves room for more than one workgroup per WGP. Override with
/// `HIPFIRE_VAE_IM2COL_TILE=<c_tile>:<th>:<tw>` for A/B.
fn im2col_tile(c_in: usize) -> HipResult<(usize, usize, usize)> {
    if let Ok(spec) = hipfire_config::developer_var("HIPFIRE_VAE_IM2COL_TILE") {
        let parts: Vec<Option<usize>> = spec.split(':').map(|p| p.parse().ok()).collect();
        return match parts[..] {
            [Some(ct), Some(th), Some(tw)] if ct > 0 && th > 0 && tw > 0 => {
                Ok((ct.min(c_in.max(1)), th, tw))
            }
            // A typo here used to fall through to the default tile silently,
            // so a sweep would report the default's number under the typo's
            // name.
            _ => Err(HipError::new(
                0,
                &format!(
                    "HIPFIRE_VAE_IM2COL_TILE=`{spec}` is not `<c_tile>:<th>:<tw>` \
                     with three positive integers"
                ),
            )),
        };
    }
    let c_tile = [64usize, 32, 16, 8, 4, 2]
        .into_iter()
        .find(|d| c_in % d == 0)
        .unwrap_or(1);
    Ok((c_tile, 8, 16))
}

impl Gpu {
    /// 3x3 stride-1 pad-1 convolution, f32, channel-major. `x` is
    /// `[c_in][h][w]`, `w` is `[c_out][c_in*9]` (the loader flattens
    /// `[c_out][c_in][3][3]`), `bias` is `[c_out]`, `y` is `[c_out][h][w]`.
    pub fn vae_conv3x3_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
        c_in: usize,
        c_out: usize,
        h: usize,
        wdt: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_conv3x3", kernels::VAE_CONV3X3_SRC, "vae_conv3x3_f32")?;
        let func = &self.functions["vae_conv3x3_f32"];

        let mut xp = x.buf.as_ptr();
        let mut wp = w.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let (mut ci, mut co) = (c_in as i32, c_out as i32);
        let (mut hv, mut wv) = (h as i32, wdt as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut hv as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
        ];
        let total = (c_out * h * wdt) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// 3x3 STRIDE-2 convolution with the diffusers `Downsample2D`
    /// asymmetric pad (left 0, right 1, top 0, bottom 1), f32, channel-major.
    /// `x` is `[c_in][h][wdt]`, `w` is `[c_out][c_in*9]`, `bias` is
    /// `[c_out]`, `y` is `[c_out][h/2][wdt/2]`. The VAE encoder's per-block
    /// downsampler; the decoder never uses it.
    ///
    /// `h` and `wdt` must be even. Not because the arithmetic breaks — for
    /// odd `h` this kernel's `h/2` happens to equal diffusers'
    /// `(h + 1 - 3)/2 + 1` — but because the encoder is only ever fed a
    /// multiple-of-16 image, so an odd extent partway down the block chain
    /// means the CALLER mis-snapped its input. Failing loudly beats halving a
    /// shape nobody intended.
    pub fn vae_conv3x3_s2_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
        c_in: usize,
        c_out: usize,
        h: usize,
        wdt: usize,
    ) -> HipResult<()> {
        if h % 2 != 0 || wdt % 2 != 0 {
            return Err(HipError::new(
                0,
                &format!(
                    "vae_conv3x3_s2_f32: {h}x{wdt} is not even; stride-2 output would truncate"
                ),
            ));
        }
        self.bind_thread()?;
        self.ensure_kernel(
            "vae_conv3x3_s2",
            kernels::VAE_CONV3X3_S2_SRC,
            "vae_conv3x3_s2_f32",
        )?;
        let func = &self.functions["vae_conv3x3_s2_f32"];

        let mut xp = x.buf.as_ptr();
        let mut wp = w.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let (mut ci, mut co) = (c_in as i32, c_out as i32);
        let (mut hv, mut wv) = (h as i32, wdt as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut hv as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
        ];
        let total = (c_out * (h / 2) * (wdt / 2)) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// 1x1 convolution (per-pixel linear), f32. `w` is `[c_out][c_in]`,
    /// `bias` is `[c_out]`, `n = h*w`. `in_pos_major` selects the input
    /// layout: false reads channel-major `[c_in][n]`, true reads
    /// position-major `[n][c_in]`. `out_pos_major` selects the output:
    /// false writes channel-major `[c_out][n]` (residual shortcut path),
    /// true writes position-major `[n][c_out]` (mid-attention q/k/v/proj).
    pub fn vae_conv1x1_f32(
        &mut self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
        c_in: usize,
        c_out: usize,
        n: usize,
        in_pos_major: bool,
        out_pos_major: bool,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_conv1x1", kernels::VAE_CONV1X1_SRC, "vae_conv1x1_f32")?;
        let func = &self.functions["vae_conv1x1_f32"];

        let mut xp = x.buf.as_ptr();
        let mut wp = w.buf.as_ptr();
        let mut bp = bias.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let (mut ci, mut co, mut nv) = (c_in as i32, c_out as i32, n as i32);
        let mut in_flag: i32 = if in_pos_major { 1 } else { 0 };
        let mut out_flag: i32 = if out_pos_major { 1 } else { 0 };
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut wp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
            &mut co as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut in_flag as *mut _ as *mut c_void,
            &mut out_flag as *mut _ as *mut c_void,
        ];
        let total = (c_out * n) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// GroupNorm over a whole-group reduction, f32. `x`/`y` are `[c][h*w]`,
    /// `gamma`/`beta` are `[c]`. `hw = h*w`. One block per group.
    pub fn vae_groupnorm_f32(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        y: &GpuTensor,
        c: usize,
        hw: usize,
        groups: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.vae_groupnorm_entry(x, gamma, beta, y, c, hw, groups, eps, "vae_groupnorm_f32")
    }

    /// GroupNorm with the SiLU that follows it in every VAE resnet folded
    /// into the normalize pass. Bit-identical to `vae_groupnorm_f32` followed
    /// by `silu_f32` (same f32 expression, same order), but it saves a full
    /// read+write of the tensor — 512 MB each way at the 1024x1024 tail.
    pub fn vae_groupnorm_silu_f32(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        y: &GpuTensor,
        c: usize,
        hw: usize,
        groups: usize,
        eps: f32,
    ) -> HipResult<()> {
        self.vae_groupnorm_entry(
            x,
            gamma,
            beta,
            y,
            c,
            hw,
            groups,
            eps,
            "vae_groupnorm_silu_f32",
        )
    }

    fn vae_groupnorm_entry(
        &mut self,
        x: &GpuTensor,
        gamma: &GpuTensor,
        beta: &GpuTensor,
        y: &GpuTensor,
        c: usize,
        hw: usize,
        groups: usize,
        eps: f32,
        entry: &str,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_groupnorm", kernels::VAE_GROUPNORM_SRC, entry)?;
        let func = &self.functions[entry];

        let mut xp = x.buf.as_ptr();
        let mut gp = gamma.buf.as_ptr();
        let mut bp = beta.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let (mut cv, mut hwv, mut gv) = (c as i32, hw as i32, groups as i32);
        let mut epsv = eps;
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut gp as *mut _ as *mut c_void,
            &mut bp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
            &mut hwv as *mut _ as *mut c_void,
            &mut gv as *mut _ as *mut c_void,
            &mut epsv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let shared = block * 4;
        unsafe {
            self.hip.launch_kernel(
                func,
                [groups as u32, 1, 1],
                [block, 1, 1],
                shared,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Nearest-neighbour 2x upsample, f32. `x` is `[c][h][w]`, `y` is
    /// `[c][2h][2w]`.
    pub fn vae_upsample2x_f32(
        &mut self,
        x: &GpuTensor,
        y: &GpuTensor,
        c: usize,
        h: usize,
        wdt: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "vae_upsample2x",
            kernels::VAE_UPSAMPLE2X_SRC,
            "vae_upsample2x_f32",
        )?;
        let func = &self.functions["vae_upsample2x_f32"];

        let mut xp = x.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let (mut cv, mut hv, mut wv) = (c as i32, h as i32, wdt as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
            &mut hv as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
        ];
        let total = (c * h * wdt) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Mid-attention scores: `s[qp][kp] = scale * dot(q[qp], k[kp])`, with
    /// `q`/`k` position-major `[n][c]` and `s` `[n][n]`.
    pub fn vae_attn_scores_f32(
        &mut self,
        q: &GpuTensor,
        k: &GpuTensor,
        s: &GpuTensor,
        n: usize,
        c: usize,
        scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_attn", kernels::VAE_ATTN_SRC, "vae_attn_scores_f32")?;
        let func = &self.functions["vae_attn_scores_f32"];

        let mut qp = q.buf.as_ptr();
        let mut kp = k.buf.as_ptr();
        let mut sp = s.buf.as_ptr();
        let (mut nv, mut cv) = (n as i32, c as i32);
        let mut sv = scale;
        let mut params: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
        ];
        let total = (n as u64) * (n as u64);
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Mid-attention context: `ctx[qp][ch] = sum_kp probs[qp][kp] * v[kp][ch]`,
    /// with `probs` `[n][n]`, `v`/`ctx` position-major `[n][c]`.
    pub fn vae_attn_ctx_f32(
        &mut self,
        probs: &GpuTensor,
        v: &GpuTensor,
        ctx: &GpuTensor,
        n: usize,
        c: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_attn", kernels::VAE_ATTN_SRC, "vae_attn_ctx_f32")?;
        let func = &self.functions["vae_attn_ctx_f32"];

        let mut pp = probs.buf.as_ptr();
        let mut vp = v.buf.as_ptr();
        let mut cp = ctx.buf.as_ptr();
        let (mut nv, mut cv) = (n as i32, c as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut pp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut cp as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
        ];
        let total = (n * c) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Mid-attention fused transpose-residual: `y[ch][p] = x[ch][p] + out[p][ch]`,
    /// with `x`/`y` channel-major `[c][n]` and `out` position-major `[n][c]`.
    pub fn vae_attn_residual_f32(
        &mut self,
        x: &GpuTensor,
        out: &GpuTensor,
        y: &GpuTensor,
        c: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_attn", kernels::VAE_ATTN_SRC, "vae_attn_residual_f32")?;
        let func = &self.functions["vae_attn_residual_f32"];

        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let mut yp = y.buf.as_ptr();
        let (mut cv, mut nv) = (c as i32, n as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void,
            &mut cv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let total = (c * n) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// im2col for the 3x3 stride-1 pad-1 conv GEMM route, over the
    /// horizontal band `[y0, y0+rows)` of the image. `x` is f32
    /// channel-major `[c_in][h][w]`; `out` holds only that band, an f16
    /// `[rows*w, c_in*9]` row matrix with column order `ci*9 + ky*3 + kx`
    /// (matching the flattened `[c_out][c_in*9]` weight layout). Pad
    /// positions are zeroed. Taps still read the full image (so the band's
    /// top/bottom halo is real data, not padding), which is what lets
    /// `conv3x3` bound the f16 column matrix instead of materialising
    /// `h*w*c_in*9` halves — 2.4 GB for a 128-channel 1024x1024 conv, and
    /// multi-GB allocations are exactly what a device already holding the
    /// transformer weights is slowest at serving.
    pub fn vae_im2col_f16_band(
        &mut self,
        x: &GpuTensor,
        out: &GpuTensor,
        c_in: usize,
        h: usize,
        wdt: usize,
        y0: usize,
        rows: usize,
    ) -> HipResult<()> {
        let map = hipfire_config::developer_var("HIPFIRE_VAE_IM2COL_MAP").unwrap_or_default();
        self.vae_im2col_f16_variant(x, out, c_in, h, wdt, y0, rows, &map)
    }

    /// Explicit lane-mapping selection, the A/B entry point that
    /// `HIPFIRE_VAE_IM2COL_MAP` drives and that the parity example uses to
    /// run two maps back to back in one process:
    ///
    /// - `""` / `"lds"` — LDS-tiled (default). One workgroup stages a
    ///   `[c_tile][(th+2)x(tw+2)]` halo patch through LDS, so each input
    ///   element crosses DRAM once instead of nine times, and each output
    ///   pixel's `c_tile*9` halves leave as one contiguous run.
    /// - `"c"` — the previous default: channel-fastest scalar gather
    ///   (contiguous 9-tap writes, scattered plane reads).
    /// - `"p"` — pixel-fastest scalar gather (measured ~6x worse).
    #[allow(clippy::too_many_arguments)]
    pub fn vae_im2col_f16_variant(
        &mut self,
        x: &GpuTensor,
        out: &GpuTensor,
        c_in: usize,
        h: usize,
        wdt: usize,
        y0: usize,
        rows: usize,
        map: &str,
    ) -> HipResult<()> {
        self.bind_thread()?;
        // An unknown map used to fall through to `lds`, which silently turns
        // an A/B of a misspelled variant into a measurement of the default.
        let entry = match map {
            "" | "lds" => "vae_im2col_f16_lds",
            "c" => "vae_im2col_f16_cfast",
            "p" => "vae_im2col_f16_pfast",
            other => {
                return Err(HipError::new(
                    0,
                    &format!("unknown im2col map `{other}`: expected `lds`, `c` or `p`"),
                ));
            }
        };
        self.ensure_kernel("vae_im2col", kernels::VAE_IM2COL_SRC, entry)?;
        let func = &self.functions[entry];

        let (c_tile, th, tw) = im2col_tile(c_in)?;
        let mut xp = x.buf.as_ptr();
        let mut op = out.buf.as_ptr();
        let (mut ci, mut hv, mut wv) = (c_in as i32, h as i32, wdt as i32);
        let (mut y0v, mut rowsv) = (y0 as i32, rows as i32);
        let (mut ctv, mut thv, mut twv) = (c_tile as i32, th as i32, tw as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut xp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut ci as *mut _ as *mut c_void,
            &mut hv as *mut _ as *mut c_void,
            &mut wv as *mut _ as *mut c_void,
            &mut y0v as *mut _ as *mut c_void,
            &mut rowsv as *mut _ as *mut c_void,
            &mut ctv as *mut _ as *mut c_void,
            &mut thv as *mut _ as *mut c_void,
            &mut twv as *mut _ as *mut c_void,
        ];
        if entry == "vae_im2col_f16_lds" {
            let block = 256u32;
            let shared = (c_tile * (th + 2) * (tw + 2) * 2) as u32;
            let grid = [
                wdt.div_ceil(tw) as u32,
                rows.div_ceil(th) as u32,
                c_in.div_ceil(c_tile) as u32,
            ];
            return unsafe {
                self.hip.launch_kernel(
                    func,
                    grid,
                    [block, 1, 1],
                    shared,
                    self.stream_ref(),
                    &mut params,
                )
            };
        }
        let total = (c_in * rows * wdt) as u64;
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Transpose `src [m][n]` into `dst [n][m]`, f32. Used to turn the
    /// position-major GEMM conv output back into channel-major layout.
    /// Default is the one-thread-per-element scatter variant — on the VAE's
    /// skinny shapes (n = 128..512 channels, m up to 1M pixels) it measured
    /// faster than the 32x32 LDS tile (347 vs 524 ms over a decode), which
    /// stays available behind `HIPFIRE_VAE_TRANSPOSE=tiled`.
    pub fn vae_transpose_f32(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        m: usize,
        n: usize,
    ) -> HipResult<()> {
        self.vae_transpose_f32_banded(src, dst, m, n, m, 0)
    }

    /// Transpose one band of a position-major `[m][n]` block into a
    /// channel-major image of row stride `dst_stride`, starting at column
    /// `dst_off`: `dst[j*dst_stride + dst_off + i] = src[i*n + j]`. The
    /// unbanded call is `dst_stride = m, dst_off = 0`.
    pub fn vae_transpose_f32_banded(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        m: usize,
        n: usize,
        dst_stride: usize,
        dst_off: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let tiled =
            hipfire_config::developer_var("HIPFIRE_VAE_TRANSPOSE").is_ok_and(|v| v == "tiled");
        let entry = if tiled {
            "vae_transpose_f32_banded"
        } else {
            "vae_transpose_f32_naive_banded"
        };
        self.ensure_kernel("vae_layout", kernels::VAE_LAYOUT_SRC, entry)?;
        let func = &self.functions[entry];

        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let (mut mv, mut nv) = (m as i32, n as i32);
        let (mut stride, mut off) = (dst_stride as i64, dst_off as i64);
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
            &mut stride as *mut _ as *mut c_void,
            &mut off as *mut _ as *mut c_void,
        ];
        if tiled {
            let grid_x = ((n + 31) / 32) as u32;
            let grid_y = ((m + 31) / 32) as u32;
            unsafe {
                return self.hip.launch_kernel(
                    func,
                    [grid_x, grid_y, 1],
                    [32, 8, 1],
                    0,
                    self.stream_ref(),
                    &mut params,
                );
            }
        }
        let total = (m as u64) * (n as u64);
        let block = 256u32;
        let grid = ((total + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Fused transpose + f32 -> f16 cast: `src [m][n]` f32 into `dst [n][m]`
    /// f16. `dst` must be allocated `DType::F16` with shape `[n, m]`. Same
    /// 32x32 tiled launch as [`Self::vae_transpose_f32`].
    pub fn vae_transpose_cast_f16(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        m: usize,
        n: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "vae_layout",
            kernels::VAE_LAYOUT_SRC,
            "vae_transpose_cast_f16",
        )?;
        let func = &self.functions["vae_transpose_cast_f16"];

        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let (mut mv, mut nv) = (m as i32, n as i32);
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let grid_x = ((n + 31) / 32) as u32;
        let grid_y = ((m + 31) / 32) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid_x, grid_y, 1],
                [32, 8, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Elementwise f32 -> f16 cast folded with a scale: `dst[i] = src[i] * scale`.
    /// `dst` must be allocated `DType::F16` with the same element count.
    pub fn vae_cast_scale_f16(
        &mut self,
        src: &GpuTensor,
        dst: &GpuTensor,
        scale: f32,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("vae_layout", kernels::VAE_LAYOUT_SRC, "vae_cast_scale_f16")?;
        let func = &self.functions["vae_cast_scale_f16"];

        let n = src.numel() as i64;
        let mut sp = src.buf.as_ptr();
        let mut dp = dst.buf.as_ptr();
        let mut sv = scale;
        let mut nv = n;
        let mut params: Vec<*mut c_void> = vec![
            &mut sp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut sv as *mut _ as *mut c_void,
            &mut nv as *mut _ as *mut c_void,
        ];
        let block = 256u32;
        let grid = ((n as u64 + block as u64 - 1) / block as u64) as u32;
        unsafe {
            self.hip.launch_kernel(
                func,
                [grid, 1, 1],
                [block, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels;

    /// Source pin for the encoder's stride-2 downsampler. The whole reason
    /// this kernel is separate from `vae_conv3x3_f32` is the tap arithmetic:
    /// diffusers' `Downsample2D` pads (0,1,0,1), so the input row is
    /// `2*oy + ky` with no `-1` recentering. Getting that wrong shifts the
    /// encoded latent by one pixel per downsample level — which surfaces as a
    /// slightly blurred round trip, not as an error.
    #[test]
    fn vae_conv3x3_s2_source_is_the_stride2_kernel() {
        let src = kernels::VAE_CONV3X3_S2_SRC;
        assert!(
            src.contains("vae_conv3x3_s2_f32"),
            "stride-2 conv source must define `vae_conv3x3_s2_f32`"
        );
        assert!(
            src.contains("2 * oy + ky"),
            "stride-2 conv must tap `2 * oy + ky` (pad top 0, bottom 1), not the \
             stride-1 kernel's `oy + ky - 1`"
        );
        assert!(
            src.contains("2 * ox + kx"),
            "stride-2 conv must tap `2 * ox + kx` (pad left 0, right 1)"
        );
    }
}
