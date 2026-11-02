; hipcc regions of gdn_chunk_prep (kernels/src/gdn_chunk_scan_prep.gfx1201.hip md5 1ecb5a312c9a7b962421a7424cee641b),
; hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -ffp-contract=off (the rdna-compute recipe for this module),
; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961).
; Lifted with peacemaker-lift from /home/kaden/qcal/perf/fp8-4k5/gdnprep-fuse/kill/prep_cache.hsaco (sha256 b2b10a2fd241fcdef8c6309799e3d5e0204d9efa6fad189d8ac0a694f50e501a); region `conv_silu`.
; Regenerate/compare: `peacemaker-isa region-import --gdn-object <object>` (feature `lift`).
; inputs: w2=v46 win2=v13 w3=v44 cur=v17 w1=v42 win1=v9 w0=v18 win0=v5
; outputs: out=v2
v_mul_f32_e32 v27, v13, v46
v_fmac_f32_e32 v27, v44, v17
v_fmac_f32_e32 v27, v42, v9
v_fmac_f32_e32 v27, v18, v5
v_mul_f32_e32 v2, 0xbfb8aa3b, v27
v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v27
v_fma_f32 v50, 0xbfb8aa3b, v27, -v2
v_rndne_f32_e32 v51, v2
v_fmac_f32_e32 v50, 0xb2a5705f, v27
v_sub_f32_e32 v2, v2, v51
v_add_f32_e32 v2, v2, v50
v_cvt_i32_f32_e32 v50, v51
v_exp_f32_e32 v2, v2
v_ldexp_f32 v2, v2, v50
v_cndmask_b32_e32 v2, 0, v2, vcc_lo
v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v27
v_cndmask_b32_e32 v2, 0x7f800000, v2, vcc_lo
v_add_f32_e32 v2, 1.0, v2
v_div_scale_f32 v50, null, v2, v2, v27
v_div_scale_f32 v58, vcc_lo, v27, v2, v27
v_rcp_f32_e32 v54, v50
v_fma_f32 v61, -v50, v54, 1.0
v_fmac_f32_e32 v54, v61, v54
v_mul_f32_e32 v61, v58, v54
v_fma_f32 v66, -v50, v61, v58
v_fmac_f32_e32 v61, v66, v54
v_fma_f32 v50, -v50, v61, v58
s_wait_alu depctr_va_vcc(0)
v_div_fmas_f32 v50, v50, v54, v61
v_div_fixup_f32 v2, v50, v2, v27
