; hipcc SiLU region of gemm_mq4g256v2_gate_up_silu_mmq_iu4_v3 (IU4_SILU_MUL: h = g / (1 + expf(-g)) * u).
; Backward dataflow slice of the first h value, taken from llvm-objdump of the hipcc code object
; (source kernels/src/gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip md5 cd3defef06a2077a9c17dd030e629da5,
;  defines IU4_SYMMETRIC_FOLD=1 IU4_G12_RASTER=1, hipcc --genco --offload-arch=gfx1201 -O3,
;  AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961), device ELF sha256 f0672fd64ab06920b1c5e40f9de29c4ba0982aae0696cd9f2a325a510f443973).
; Inputs: v162 = g (gate accumulator), v164 = u (up accumulator). Output: last definition.
; Regenerate and compare with `hipfire-isa region-import --disassembly <objdump.txt>`.
v_dual_mul_f32 v8, 0xbfb8aa3b, v162
v_cmp_nlt_f32_e64 s14, 0x42ce8ed0, v162
v_cmp_ngt_f32_e64 s15, 0xc2b17218, v162
v_rndne_f32_e32 v13, v8
v_fma_f32 v16, 0xbfb8aa3b, v162, -v8
v_dual_sub_f32 v15, v8, v13
v_dual_fmac_f32 v16, 0xb2a5705f, v162
v_cvt_i32_f32_e32 v13, v13
v_add_f32_e32 v15, v15, v16
v_exp_f32_e32 v15, v15
v_ldexp_f32 v13, v15, v13
v_cndmask_b32_e64 v13, 0, v13, s14
v_cndmask_b32_e64 v13, 0x7f800000, v13, s15
v_add_f32_e32 v13, 1.0, v13
v_div_scale_f32 v15, null, v13, v13, v162
v_rcp_f32_e32 v16, v15
v_fma_f32 v17, -v15, v16, 1.0
v_fmac_f32_e32 v16, v17, v16
v_div_scale_f32 v17, vcc_lo, v162, v13, v162
v_mul_f32_e32 v18, v17, v16
v_fma_f32 v19, -v15, v18, v17
v_fmac_f32_e32 v18, v19, v16
v_fma_f32 v15, -v15, v18, v17
s_wait_alu depctr_va_vcc(0)
v_div_fmas_f32 v15, v15, v16, v18
v_div_fixup_f32 v13, v15, v13, v162
v_dual_mul_f32 v13, v164, v13
