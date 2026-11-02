; hipcc SiLU region of gemm_mq4g256v2_gate_up_silu_iu4_v2c_gfx11 (IU4_V2C_SILU_MUL: h = g / (1 + expf(-g)) * u), gfx1100.
; Backward dataflow slice of the first h value, from llvm-objdump of the shipped runtime JIT code object
; (source kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip md5 ed398b4f884aa0f60ef001819bc528fc, hipx hipcc --genco --offload-arch=gfx1100 -O3 --no-offload-compress,
;  AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961); bundle sha256 04b7b46f9a86aae7df353bf0c44495676896a3aac3caa71946e7bf95762cb945,
;  device ELF sha256 5b9bf3dda9f1f3732f89411e6d22cbc85492fc785da3f0b1db491ba5276a2036).
; Lane masks renamed to SGPRs (VOP3 compare/select forms); hazard/issue waits are placed by the builder.
; Inputs: v91 = g (gate accumulator), v90 = u (up accumulator). Output: last definition.
; Regenerate and compare with `peacemaker-isa region-import --arch gfx1100 --disassembly <objdump.txt>`.
v_mul_f32_e32 v0, 0xbfb8aa3b, v91
v_cmp_nlt_f32_e64 s0, 0x42ce8ed0, v91
v_rndne_f32_e32 v3, v0
v_fma_f32 v2, 0xbfb8aa3b, v91, -v0
v_sub_f32_e32 v4, v0, v3
v_cvt_i32_f32_e32 v3, v3
v_fmac_f32_e32 v2, 0xb2a5705f, v91
v_dual_add_f32 v2, v4, v2
v_exp_f32_e32 v2, v2
v_ldexp_f32 v2, v2, v3
v_cndmask_b32_e64 v2, 0, v2, s0
v_cmp_ngt_f32_e64 s1, 0xc2b17218, v91
v_cndmask_b32_e64 v2, 0x7f800000, v2, s1
v_dual_add_f32 v2, 1.0, v2
v_div_scale_f32 v12, null, v2, v2, v91
v_rcp_f32_e32 v14, v12
v_fma_f32 v15, -v12, v14, 1.0
v_dual_fmac_f32 v14, v15, v14
v_div_scale_f32 v3, vcc_lo, v91, v2, v91
v_mul_f32_e32 v15, v3, v14
v_fma_f32 v19, -v12, v15, v3
v_fmac_f32_e32 v15, v19, v14
v_fma_f32 v3, -v12, v15, v3
v_div_fmas_f32 v3, v3, v14, v15
v_div_fixup_f32 v1, v3, v2, v91
v_mul_f32_e32 v1, v1, v90
