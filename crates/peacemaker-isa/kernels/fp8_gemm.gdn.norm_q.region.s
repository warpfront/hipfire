; hipcc regions of gdn_chunk_prep (kernels/src/gdn_chunk_scan_prep.gfx1201.hip md5 1ecb5a312c9a7b962421a7424cee641b),
; hipcc --genco --offload-arch=gfx1201 -O3 --no-offload-compress -ffp-contract=off (the rdna-compute recipe for this module),
; AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961).
; Lifted with peacemaker-lift from /home/kaden/qcal/perf/fp8-4k5/gdnprep-fuse/kill/prep_cache.hsaco (sha256 b2b10a2fd241fcdef8c6309799e3d5e0204d9efa6fad189d8ac0a694f50e501a); region `norm_q`.
; Regenerate/compare: `peacemaker-isa region-import --gdn-object <object>` (feature `lift`).
; inputs: o0=v5 o1=v4 o2=v3 o3=v2
; sinputs: eps=s30 qscale=s29
; lane: v47
; outputs: h0=v5.l h1=v5.h h2=v3.l h3=v48.l
v_mul_f32_e32 v27, v4, v4
v_mul_f32_e32 v28, v5, v5
v_xor_b32_e32 v48, 16, v47
v_mul_f32_e32 v49, v3, v3
v_cmp_gt_u32_e32 vcc_lo, 32, v48
v_add_f32_e32 v27, v28, v27
v_mul_f32_e32 v28, v2, v2
v_cndmask_b32_e32 v48, v47, v48, vcc_lo
v_add_f32_e32 v27, v27, v49
v_lshlrev_b32_e32 v48, 2, v48
v_add_f32_e32 v27, v27, v28
ds_bpermute_b32 v28, v48, v27
v_xor_b32_e32 v48, 8, v47
v_cmp_gt_u32_e32 vcc_lo, 32, v48
v_cndmask_b32_e32 v48, v47, v48, vcc_lo
v_lshlrev_b32_e32 v48, 2, v48
v_add_f32_e32 v27, v27, v28
ds_bpermute_b32 v28, v48, v27
v_xor_b32_e32 v48, 4, v47
v_cmp_gt_u32_e32 vcc_lo, 32, v48
v_cndmask_b32_e32 v48, v47, v48, vcc_lo
v_lshlrev_b32_e32 v48, 2, v48
v_add_f32_e32 v27, v27, v28
ds_bpermute_b32 v28, v48, v27
v_xor_b32_e32 v48, 2, v47
v_cmp_gt_u32_e32 vcc_lo, 32, v48
v_cndmask_b32_e32 v48, v47, v48, vcc_lo
v_lshlrev_b32_e32 v48, 2, v48
v_add_f32_e32 v27, v27, v28
ds_bpermute_b32 v28, v48, v27
v_xor_b32_e32 v48, 1, v47
v_cmp_gt_u32_e32 vcc_lo, 32, v48
v_cndmask_b32_e32 v48, v47, v48, vcc_lo
v_lshlrev_b32_e32 v48, 2, v48
v_add_f32_e32 v27, v27, v28
ds_bpermute_b32 v28, v48, v27
v_add_f32_e32 v27, v27, v28
v_add_f32_e32 v27, s30, v27
v_mul_f32_e32 v28, 0x4b800000, v27
v_cmp_gt_f32_e32 vcc_lo, 0x800000, v27
v_cndmask_b32_e32 v27, v27, v28, vcc_lo
v_rsq_f32_e32 v27, v27
v_mul_f32_e32 v28, 0x45800000, v27
v_cndmask_b32_e32 v27, v27, v28, vcc_lo
v_mul_f32_e32 v3, v27, v3
v_mul_f32_e32 v5, v27, v5
v_mul_f32_e32 v4, v27, v4
v_mul_f32_e32 v2, v27, v2
v_mul_f32_e32 v48, s29, v3
v_mul_f32_e32 v28, s29, v5
v_mul_f32_e32 v49, s29, v4
v_mov_b32_e32 v3, v48
v_mul_f32_e32 v48, s29, v2
v_mov_b32_e32 v5, v28
v_mov_b32_e32 v4, v49
v_cvt_f16_f32_e32 v3.l, v3
v_mov_b32_e32 v2, v48
v_cvt_f16_f32_e32 v5.l, v5
v_cvt_f16_f32_e32 v5.h, v4
v_cvt_f16_f32_e32 v48.l, v2
