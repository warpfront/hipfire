; Lean conv1d+SiLU for the F2 QKVZA+GDN epilogue. Not a hipcc slice: the four
; conv MACs are fp8_gemm.gdn.conv_silu.region.s's first four lines verbatim;
; the SiLU that follows returns the same bits as that golden's SiLU for every
; one of the 2^32 f32 conv results x (NaN payloads, signed zeros, denormals,
; infinities included), checked exhaustively on gfx1201 by
; `tools/gdn/silu_exhaust.py` (round-nearest, f32 denormals kept: the F2 mode).
; - `v_med3_num_f32 x, -128, 128` feeds the exp range reduction instead of the two
;   compare+select range fixups (both saturate to the same 1 + e).
; - The IEEE division keeps `v_div_fixup_f32` (special operands) but replaces
;   div_scale / refinement / div_fmas: rcp of d * 2^-32 rescaled by 2^-32
;   (d = 1 + e >= 1, so only d >= 2^126 would flush rcp), then one residual
;   correction q + (x - d q) r.
; clamp must hold 128.0 (0x43000000).
; inputs: w2=v46 win2=v13 w3=v44 cur=v17 w1=v42 win1=v9 w0=v18 win0=v5
; sinputs: clamp=s30
; outputs: out=v2
v_mul_f32_e32 v27, v13, v46
v_fmac_f32_e32 v27, v44, v17
v_fmac_f32_e32 v27, v42, v9
v_fmac_f32_e32 v27, v18, v5
v_med3_num_f32 v3, v27, -s30, s30
v_mul_f32_e32 v2, 0xbfb8aa3b, v3
v_fma_f32 v50, 0xbfb8aa3b, v3, -v2
v_rndne_f32_e32 v51, v2
v_fmac_f32_e32 v50, 0xb2a5705f, v3
v_sub_f32_e32 v2, v2, v51
v_add_f32_e32 v2, v2, v50
v_cvt_i32_f32_e32 v50, v51
v_exp_f32_e32 v2, v2
v_ldexp_f32 v2, v2, v50
v_add_f32_e32 v2, 1.0, v2
v_mul_f32_e32 v50, 0x2f800000, v2
v_rcp_f32_e32 v51, v50
v_mul_f32_e32 v51, 0x2f800000, v51
v_mul_f32_e32 v3, v27, v51
v_fma_f32 v50, -v2, v3, v27
v_fmac_f32_e32 v3, v50, v51
v_div_fixup_f32 v2, v3, v2, v27
