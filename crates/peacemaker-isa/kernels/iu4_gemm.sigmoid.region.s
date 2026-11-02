; hipcc sigmoid region of gemm_mq4g256v2_residual_mmq_iu4_full_add_v3 (iu4_shared_sigmoid: sig = 1.0f / (1.0f + __expf(-s))).
; The per-token gate lowering of the `add == 2` shared-expert sigmoid residual epilogue, copied from llvm-objdump
; of the hipcc code object (source kernels/src/gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip,
;  defines IU4_SYMMETRIC_FOLD=1 IU4_G12_RASTER=1, hipcc --genco --offload-arch=gfx1201 -O3,
;  AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb691863e9258034ab48c1dfce21689d5606)).
; Only registers are renamed (hipcc reuses v0 for the loaded logit and the exp argument): input v10 = s (the gate logit).
; Output: last definition. Used by the `_b1` sigmoid residual ADD epilogue (`Epi::AddSigmoid`).
v_mul_f32_e32 v0, 0xbfb8aa3b, v10
v_exp_f32_e32 v0, v0
v_add_f32_e32 v0, 1.0, v0
v_div_scale_f32 v1, null, v0, v0, 1.0
v_rcp_f32_e32 v5, v1
v_fma_f32 v6, -v1, v5, 1.0
v_fmac_f32_e32 v5, v6, v5
v_div_scale_f32 v6, vcc_lo, 1.0, v0, 1.0
v_mul_f32_e32 v8, v6, v5
v_fma_f32 v9, -v1, v8, v6
v_fmac_f32_e32 v8, v9, v5
v_fma_f32 v1, -v1, v8, v6
s_wait_alu depctr_va_vcc(0)
v_div_fmas_f32 v1, v1, v5, v8
v_div_fixup_f32 v9, v1, v0, 1.0
