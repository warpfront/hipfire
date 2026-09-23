	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	gated_norm_mq_rotate_awq_i4_gfx12 ; -- Begin function gated_norm_mq_rotate_awq_i4_gfx12
	.globl	gated_norm_mq_rotate_awq_i4_gfx12
	.p2align	8
	.type	gated_norm_mq_rotate_awq_i4_gfx12,@function
gated_norm_mq_rotate_awq_i4_gfx12:      ; @gated_norm_mq_rotate_awq_i4_gfx12
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b128 s[16:19], s[0:1], 0x40
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s17, 0x80
	s_cbranch_scc1 .LBB0_55
; %bb.1:
	s_mov_b32 s2, exec_lo
	v_cmpx_gt_u32_e32 64, v0
	s_cbranch_execz .LBB0_55
; %bb.2:
	s_load_b32 s20, s[0:1], 0x50
	s_ashr_i32 s25, s19, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s2, s25, 24
	s_add_co_i32 s2, s19, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s2, s2, 8
	s_cmp_ge_i32 ttmp9, s2
	s_cbranch_scc1 .LBB0_55
; %bb.3:
	v_lshrrev_b32_e32 v5, 5, v0
	s_lshl_b32 s22, ttmp9, 1
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_or_b32_e32 v1, s22, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_i32_e32 vcc_lo, s16, v1
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_55
; %bb.4:
	s_load_b512 s[0:15], s[0:1], 0x0
	v_ashrrev_i32_e32 v2, 31, v1
	s_mov_b32 s16, ttmp7
	s_ashr_i32 s17, ttmp7, 31
	s_mov_b32 s24, s19
	v_and_b32_e32 v10, 31, v0
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[24:25], s[24:25], s[16:17]
	v_lshlrev_b64_e32 v[6:7], 9, v[1:2]
	s_lshl_b64 s[24:25], s[24:25], 2
	v_lshlrev_b64_e32 v[1:2], 7, v[1:2]
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v3, v10
	v_mov_b32_e32 v8, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[0:1], s[24:25]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, s0, v6
	v_add_co_ci_u32_e64 v7, null, s1, v7, vcc_lo
	s_mov_b32 s0, 0
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[11:12], 2, v[3:4]
	v_add_co_u32 v11, vcc_lo, v6, v11
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v12, null, v7, v12, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 0x5f, v3
	global_load_b32 v9, v[11:12], off
	v_add_nc_u32_e32 v11, 32, v3
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, vcc_lo, s0
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v8, v9, v9 :: v_dual_mov_b32 v3, v11
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_5
; %bb.6:                                ; %.preheader
	s_or_b32 exec_lo, exec_lo, s0
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_lshlrev_b64_e32 v[16:17], 2, v[1:2]
	s_add_nc_u64 s[0:1], s[2:3], s[24:25]
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_bfi_b32 v4, v3, 0, 32
	v_xor_b32_e32 v9, 16, v3
	v_xor_b32_e32 v12, 8, v3
	v_xor_b32_e32 v13, 4, v3
	v_xor_b32_e32 v14, 2, v3
	v_cmp_lt_u32_e32 vcc_lo, v9, v4
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v3, v9, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v12, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v12, v3, v12 :: v_dual_lshlrev_b32 v11, 2, v9
	v_cmp_lt_u32_e32 vcc_lo, v13, v4
	v_lshlrev_b32_e32 v12, 2, v12
	ds_bpermute_b32 v9, v11, v8
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v13, v3, v13, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, v14, v4
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v14, v3, v14 :: v_dual_lshlrev_b32 v13, 2, v13
	v_lshlrev_b32_e32 v15, 2, v14
	v_xor_b32_e32 v14, 1, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_u32_e32 vcc_lo, v14, v4
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_add_f32 v8, v8, v9 :: v_dual_cndmask_b32 v3, v3, v14
	ds_bpermute_b32 v9, v12, v8
	s_wait_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	ds_bpermute_b32 v9, v13, v8
	s_wait_dscnt 0x0
	v_add_f32_e32 v8, v8, v9
	ds_bpermute_b32 v9, v15, v8
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v8, v9 :: v_dual_lshlrev_b32 v14, 2, v3
	ds_bpermute_b32 v4, v14, v3
	s_wait_dscnt 0x0
	v_dual_mov_b32 v8, s18 :: v_dual_add_f32 v3, v3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmamk_f32 v3, v3, 0x3c000000, v8
	v_mul_f32_e32 v4, 0x4b800000, v3
	v_cmp_gt_f32_e32 vcc_lo, 0x800000, v3
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v3, v3, v4, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, s0, s0, v16
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v8, null, s1, v17, s0
	v_rsq_f32_e32 v1, v3
	v_add_co_u32 v9, s0, s6, v16
	v_lshlrev_b32_e32 v3, 9, v5
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, s7, v17, s0
	s_mov_b32 s0, 0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v18, 0x45800000, v1
	v_cndmask_b32_e32 v17, v1, v18, vcc_lo
	v_mov_b32_e32 v1, v10
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[18:19], 2, v[1:2]
	v_add_co_u32 v20, vcc_lo, v4, v18
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v21, null, v8, v19, vcc_lo
	global_load_b32 v24, v[20:21], off
	v_add_co_u32 v20, vcc_lo, v6, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, v7, v19, vcc_lo
	v_add_co_u32 v22, vcc_lo, s4, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, s5, v19, vcc_lo
	global_load_b32 v20, v[20:21], off
	global_load_b32 v21, v[22:23], off
	v_add_co_u32 v18, vcc_lo, v9, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, v16, v19, vcc_lo
	global_load_b32 v18, v[18:19], off
	s_wait_loadcnt 0x3
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v24
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v20, v17, v20
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v19, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v20, v20, v21
	v_fma_f32 v22, 0xbfb8aa3b, v24, -v19
	v_rndne_f32_e32 v23, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v22, 0xb2a5705f, v24 :: v_dual_sub_f32 v19, v19, v23
	v_add_f32_e32 v19, v19, v22
	v_cvt_i32_f32_e32 v22, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_exp_f32_e32 v19, v19
	v_ldexp_f32 v19, v19, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v19, 0, v19, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0x7f800000, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v19, 1.0, v19
	v_div_scale_f32 v22, null, v19, v19, v24
	v_div_scale_f32 v26, vcc_lo, v24, v19, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v23, v22
	v_fma_f32 v25, -v22, v23, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v25, v23
	v_mul_f32_e32 v25, v26, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v27, -v22, v25, v26
	v_fmac_f32_e32 v25, v27, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v22, v25, v26
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v22, v22, v23, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v19, v22, v19, v24
	v_mul_f32_e32 v19, v20, v19
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v20, null, v18, v18, v19
	v_div_scale_f32 v23, vcc_lo, v19, v18, v19
	v_rcp_f32_e32 v21, v20
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v20, v21, 1.0
	v_fmac_f32_e32 v21, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, v23, v21
	v_fma_f32 v24, -v20, v22, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v24, v21
	v_fma_f32 v20, -v20, v22, v23
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_div_fmas_f32 v20, v20, v21, v22
	v_add_nc_u32_e32 v21, 32, v1
	v_cmp_lt_u32_e32 vcc_lo, 0x5f, v1
	v_lshl_add_u32 v22, v1, 2, v3
	v_div_fixup_f32 v18, v20, v18, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_mov_b32_e32 v1, v21
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s0, vcc_lo, s0
	ds_store_b32 v22, v18
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_55
; %bb.9:
	v_lshlrev_b32_e32 v9, 5, v0
	s_cmp_eq_u64 s[12:13], 0
	s_clause 0x3
	global_load_b128 v[1:4], v9, s[8:9]
	global_load_b128 v[5:8], v9, s[8:9] offset:16
	global_load_b128 v[16:19], v9, s[10:11]
	global_load_b128 v[20:23], v9, s[10:11] offset:16
	ds_load_b128 v[24:27], v9
	ds_load_b128 v[28:31], v9 offset:16
	s_wait_loadcnt_dscnt 0x301
	v_dual_mul_f32 v2, v25, v2 :: v_dual_and_b32 v9, 1, v0
	s_wait_loadcnt_dscnt 0x200
	v_mul_f32_e32 v6, v29, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_cmp_eq_u32_e64 s0, 0, v9
	v_and_b32_e32 v9, 2, v0
	v_fma_f32 v25, v24, v1, v2
	v_fma_f32 v1, v24, v1, -v2
	v_mul_f32_e32 v4, v27, v4
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	v_and_b32_e32 v9, 4, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_fma_f32 v2, v26, v3, v4
	v_fma_f32 v3, v26, v3, -v4
	v_fma_f32 v4, v28, v5, v6
	v_mul_f32_e32 v8, v31, v8
	v_fma_f32 v5, v28, v5, -v6
	v_dual_add_f32 v24, v1, v3 :: v_dual_sub_f32 v1, v1, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v6, v30, v7, v8
	v_fma_f32 v7, v30, v7, -v8
	v_add_f32_e32 v3, v5, v7
	v_add_f32_e32 v8, v25, v2
	v_sub_f32_e32 v2, v25, v2
	v_dual_add_f32 v25, v4, v6 :: v_dual_sub_f32 v4, v4, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_sub_f32 v5, v5, v7 :: v_dual_add_f32 v6, v8, v25
	v_sub_f32_e32 v7, v8, v25
	v_dual_add_f32 v8, v24, v3 :: v_dual_sub_f32 v3, v24, v3
	ds_swizzle_b32 v28, v7 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v25, v8 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v29, v3 offset:swizzle(SWAP,1)
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v3, -v3, v3, s0
	v_add_f32_e32 v24, v2, v4
	v_sub_f32_e32 v2, v2, v4
	v_dual_add_f32 v4, v1, v5 :: v_dual_sub_f32 v1, v1, v5
	ds_swizzle_b32 v5, v6 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v6, -v6, v6, s0
	v_cndmask_b32_e64 v8, -v8, v8, s0
	v_cndmask_b32_e64 v7, -v7, v7, s0
	ds_swizzle_b32 v31, v1 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v1, -v1, v1, s0
	s_wait_dscnt 0x4
	v_add_f32_e32 v7, v7, v28
	s_wait_dscnt 0x2
	v_add_f32_e32 v3, v3, v29
	ds_swizzle_b32 v26, v24 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v24, -v24, v24, s0
	ds_swizzle_b32 v28, v7 offset:swizzle(SWAP,2)
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	ds_swizzle_b32 v29, v3 offset:swizzle(SWAP,2)
	s_wait_dscnt 0x4
	v_add_f32_e32 v5, v6, v5
	ds_swizzle_b32 v27, v4 offset:swizzle(SWAP,1)
	v_add_f32_e32 v6, v8, v25
	v_cndmask_b32_e64 v4, -v4, v4, s0
	s_wait_dscnt 0x4
	v_add_f32_e32 v1, v1, v31
	ds_swizzle_b32 v30, v2 offset:swizzle(SWAP,1)
	v_cndmask_b32_e64 v2, -v2, v2, s0
	ds_swizzle_b32 v25, v6 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	ds_swizzle_b32 v31, v1 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x6
	v_add_f32_e32 v8, v24, v26
	ds_swizzle_b32 v24, v5 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	s_wait_dscnt 0x6
	v_add_f32_e32 v7, v7, v28
	s_wait_dscnt 0x5
	v_add_f32_e32 v3, v3, v29
	ds_swizzle_b32 v26, v8 offset:swizzle(SWAP,2)
	s_wait_dscnt 0x5
	v_add_f32_e32 v4, v4, v27
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	ds_swizzle_b32 v28, v7 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v29, v3 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x6
	v_add_f32_e32 v2, v2, v30
	s_wait_dscnt 0x4
	v_dual_add_f32 v6, v6, v25 :: v_dual_add_f32 v1, v1, v31
	ds_swizzle_b32 v30, v2 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	ds_swizzle_b32 v25, v6 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v31, v1 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x6
	v_add_f32_e32 v5, v5, v24
	ds_swizzle_b32 v27, v4 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_dscnt 0x6
	v_add_f32_e32 v8, v8, v26
	ds_swizzle_b32 v24, v5 offset:swizzle(SWAP,4)
	v_and_b32_e32 v9, 8, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	ds_swizzle_b32 v26, v8 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	s_wait_dscnt 0x5
	v_add_f32_e32 v2, v2, v30
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x4
	v_dual_add_f32 v6, v6, v25 :: v_dual_add_f32 v7, v7, v28
	v_add_f32_e32 v3, v3, v29
	s_wait_dscnt 0x3
	v_add_f32_e32 v1, v1, v31
	s_wait_dscnt 0x2
	v_add_f32_e32 v4, v4, v27
	ds_swizzle_b32 v25, v6 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v28, v7 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v29, v3 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x4
	v_add_f32_e32 v5, v5, v24
	ds_swizzle_b32 v27, v4 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v30, v2 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	ds_swizzle_b32 v24, v5 offset:swizzle(SWAP,8)
	v_cmp_eq_u32_e32 vcc_lo, 0, v9
	s_wait_dscnt 0x6
	v_add_f32_e32 v8, v8, v26
	ds_swizzle_b32 v31, v1 offset:swizzle(SWAP,8)
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	ds_swizzle_b32 v26, v8 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x6
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_add_f32 v6, v6, v25 :: v_dual_add_f32 v7, v7, v28
	s_wait_dscnt 0x4
	v_add_f32_e32 v4, v4, v27
	s_wait_dscnt 0x3
	v_dual_add_f32 v2, v2, v30 :: v_dual_add_f32 v3, v3, v29
	s_wait_dscnt 0x2
	v_add_f32_e32 v5, v5, v24
	ds_swizzle_b32 v27, v4 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v30, v2 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	s_wait_dscnt 0x3
	v_dual_add_f32 v8, v8, v26 :: v_dual_add_f32 v1, v1, v31
	ds_swizzle_b32 v24, v6 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v28, v3 offset:swizzle(SWAP,16)
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	ds_swizzle_b32 v25, v8 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	s_wait_dscnt 0x5
	v_add_f32_e32 v4, v4, v27
	s_wait_dscnt 0x4
	v_add_f32_e32 v2, v2, v30
	ds_swizzle_b32 v27, v7 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v30, v1 offset:swizzle(SWAP,16)
	s_wait_dscnt 0x5
	v_add_f32_e32 v5, v5, v9
	ds_swizzle_b32 v26, v4 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v29, v2 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x6
	v_add_f32_e32 v6, v6, v24
	s_wait_dscnt 0x4
	v_dual_add_f32 v8, v8, v25 :: v_dual_add_f32 v3, v3, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v5, 0x3d800000, v5 :: v_dual_mul_f32 v8, 0x3d800000, v8
	v_mul_f32_e32 v9, 0x3d800000, v3
	s_wait_dscnt 0x3
	v_add_f32_e32 v7, v7, v27
	s_wait_dscnt 0x2
	v_add_f32_e32 v1, v1, v30
	v_mul_f32_e32 v6, 0x3d800000, v6
	s_wait_dscnt 0x1
	v_add_f32_e32 v4, v4, v26
	s_wait_dscnt 0x0
	v_dual_add_f32 v2, v2, v29 :: v_dual_mul_f32 v7, 0x3d800000, v7
	v_mul_f32_e32 v25, 0x3d800000, v1
	s_wait_loadcnt 0x1
	v_dual_mul_f32 v1, v16, v5 :: v_dual_mul_f32 v4, 0x3d800000, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v24, 0x3d800000, v2
	v_dual_mul_f32 v2, v17, v6 :: v_dual_mul_f32 v3, v18, v8
	s_wait_loadcnt 0x0
	v_dual_mul_f32 v5, v20, v7 :: v_dual_mul_f32 v4, v19, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mul_f32 v6, v21, v9 :: v_dual_mul_f32 v7, v22, v24
	v_mul_f32_e32 v8, v23, v25
	s_cbranch_scc1 .LBB0_11
; %bb.10:
	v_lshlrev_b32_e32 v9, 3, v0
	s_mov_b32 s2, ttmp9
	s_ashr_i32 s3, ttmp9, 31
	s_add_nc_u64 s[4:5], s[12:13], s[24:25]
	s_lshl_b64 s[2:3], s[2:3], 10
	v_lshlrev_b32_e32 v9, 2, v9
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	s_clause 0x1
	global_store_b128 v9, v[1:4], s[2:3]
	global_store_b128 v9, v[5:8], s[2:3] offset:16
.LBB0_11:
	v_lshrrev_b32_e32 v9, 1, v0
	s_mov_b32 s5, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v16, 2, v9
	ds_bpermute_b32 v9, v16, v1
	ds_bpermute_b32 v17, v16, v2
	ds_bpermute_b32 v18, v16, v3
	ds_bpermute_b32 v21, v16, v4
	ds_bpermute_b32 v19, v16, v5
	ds_bpermute_b32 v22, v16, v6
	ds_bpermute_b32 v23, v16, v7
	ds_bpermute_b32 v24, v16, v8
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v20, v19, v9, s0
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v19, v22, v17, s0
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v18, v23, v18, s0
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v17, v24, v21, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v9, |v20|, |v19|, |v18|
	v_max_num_f32_e64 v21, |v17|, |v17|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v9, v9, v21
	ds_bpermute_b32 v21, v11, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v21, v21, v21
	v_max_num_f32_e32 v9, v9, v21
	ds_bpermute_b32 v21, v12, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v21, v21, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v9, v9, v21
	ds_bpermute_b32 v21, v13, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v21, v21, v21
	v_max_num_f32_e32 v9, v9, v21
	ds_bpermute_b32 v21, v15, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v21, v21, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v9, v9, v21
	ds_bpermute_b32 v21, v14, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v21, v21, v21
	v_max_num_f32_e32 v21, v9, v21
	v_mov_b32_e32 v9, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_neq_f32_e32 0, v21
	s_cbranch_execz .LBB0_23
; %bb.12:
	v_div_scale_f32 v9, null, 0x40e00000, 0x40e00000, v21
	v_div_scale_f32 v24, vcc_lo, v21, 0x40e00000, v21
	s_mov_b32 s6, 0xc1000000
	v_rcp_f32_e32 v22, v9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v23, -v9, v22, 1.0
	v_fmac_f32_e32 v22, v23, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v24, v22
	v_fma_f32 v25, -v9, v23, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v25, v22
	v_fma_f32 v9, -v9, v23, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v9, v9, v22, v23
	v_mov_b32_e32 v23, 0x7149f2ca
	v_div_fixup_f32 v22, v9, 0x40e00000, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v9, 1.0 :: v_dual_mul_f32 v22, 0.5, v22
	s_branch .LBB0_14
.LBB0_13:                               ; %.preheader.preheader.i.i
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mul_f32_e32 v24, s1, v22
	s_add_co_i32 s5, s5, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s5, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v25, null, v24, v24, v20
	v_rcp_f32_e32 v29, v25
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v35, -v25, v29, 1.0
	v_fmac_f32_e32 v29, v35, v29
	v_div_scale_f32 v26, null, v24, v24, v19
	v_div_scale_f32 v27, null, v24, v24, v18
	v_div_scale_f32 v28, null, v24, v24, v17
	v_div_scale_f32 v33, vcc_lo, v20, v24, v20
	v_div_scale_f32 v34, s1, v19, v24, v19
	v_rcp_f32_e32 v30, v26
	v_rcp_f32_e32 v31, v27
	v_rcp_f32_e32 v32, v28
	v_div_scale_f32 v38, s2, v18, v24, v18
	v_div_scale_f32 v35, s3, v17, v24, v17
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v36, -v26, v30, 1.0
	v_fma_f32 v37, -v27, v31, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v39, -v28, v32, 1.0
	v_dual_fmac_f32 v30, v36, v30 :: v_dual_fmac_f32 v31, v37, v31
	v_mul_f32_e32 v36, v33, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v32, v39, v32 :: v_dual_mul_f32 v37, v34, v30
	v_mul_f32_e32 v39, v38, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v41, -v25, v36, v33
	v_fma_f32 v42, -v26, v37, v34
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v43, -v27, v39, v38
	v_fmac_f32_e32 v36, v41, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v40, v35, v32 :: v_dual_fmac_f32 v37, v42, v30
	v_fmac_f32_e32 v39, v43, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v25, -v25, v36, v33
	v_fma_f32 v44, -v28, v40, v35
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v26, -v26, v37, v34
	v_fma_f32 v27, -v27, v39, v38
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v25, v25, v29, v36
	s_mov_b32 vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v26, v26, v30, v37
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v25, v25, v24, v20
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v27, v27, v31, v39
	s_mov_b32 vcc_lo, s3
	v_div_fixup_f32 v26, v26, v24, v19
	v_rndne_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v27, v27, v24, v18
	v_rndne_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v25, v25, s6, 0x40e00000
	v_rndne_f32_e32 v27, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v26, v26, s6, 0x40e00000
	v_fma_f32 v25, -v25, v24, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v27, v27, s6, 0x40e00000
	v_fma_f32 v26, -v26, v24, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v25, v25, v25, 0
	v_fmac_f32_e32 v40, v44, v32
	v_fma_f32 v27, -v27, v24, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v25, v26, v26
	v_fma_f32 v28, -v28, v40, v35
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v25, v27, v27
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v28, v28, v32, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v28, v28, v24, v17
	v_rndne_f32_e32 v28, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_med3_num_f32 v28, v28, s6, 0x40e00000
	v_fma_f32 v26, -v28, v24, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v26
	ds_bpermute_b32 v26, v11, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v25, v25, v26
	ds_bpermute_b32 v26, v12, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v25, v25, v26
	ds_bpermute_b32 v26, v13, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v25, v25, v26
	ds_bpermute_b32 v26, v15, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v25, v25, v26
	ds_bpermute_b32 v26, v14, v25
	s_wait_dscnt 0x0
	v_add_f32_e32 v25, v25, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v25, v23
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v23, v23, v25, vcc_lo
	v_cndmask_b32_e32 v9, v9, v24, vcc_lo
	s_cbranch_scc0 .LBB0_23
.LBB0_14:                               ; %NodeBlock582
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s5, 1
	s_mov_b32 s1, 1.0
	s_cbranch_scc1 .LBB0_13
; %bb.15:                               ; %NodeBlock
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_cmp_lt_i32 s5, 2
	s_mov_b32 s2, -1
                                        ; implicit-def: $sgpr1
	s_cbranch_scc1 .LBB0_21
; %bb.16:                               ; %LeafBlock
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_cmp_lg_u32 s5, 2
	s_mov_b32 s1, -1
	s_cbranch_scc0 .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_14 Depth=1
	s_mov_b32 s1, 0
.LBB0_18:                               ; %Flow596
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, 2.0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_14 Depth=1
	s_mov_b32 s1, 0x3fdb6db7
.LBB0_20:                               ; %Flow597
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_mov_b32 s2, 0
.LBB0_21:                               ; %Flow598
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_13
; %bb.22:                               ;   in Loop: Header=BB0_14 Depth=1
	s_mov_b32 s1, 0x3fa49249
	s_branch .LBB0_13
.LBB0_23:                               ; %Flow601
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_neq_f32_e64 s1, 0, v21
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_div_scale_f32 v22, null, v9, v9, v20
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v23, v22
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v24, -v22, v23, 1.0
	v_fmac_f32_e32 v23, v24, v23
	v_div_scale_f32 v24, vcc_lo, v20, v9, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v25, v24, v23
	v_fma_f32 v26, -v22, v25, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v23
	v_fma_f32 v22, -v22, v25, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v22, v22, v23, v25
	v_div_fixup_f32 v20, v22, v9, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v20, v20
	v_maxmin_num_f32 v20, v20, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v22, v20
.LBB0_25:
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_div_scale_f32 v20, null, v9, v9, v19
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v21, v20
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v23, -v20, v21, 1.0
	v_fmac_f32_e32 v21, v23, v21
	v_div_scale_f32 v23, vcc_lo, v19, v9, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v24, v23, v21
	v_fma_f32 v25, -v20, v24, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v25, v21
	v_fma_f32 v20, -v20, v24, v23
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v20, v20, v21, v24
	v_div_fixup_f32 v19, v20, v9, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v19, v19
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v19, v19, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v21, v19
.LBB0_27:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_div_scale_f32 v20, null, v9, v9, v18
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v23, v20
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v24, -v20, v23, 1.0
	v_fmac_f32_e32 v23, v24, v23
	v_div_scale_f32 v24, vcc_lo, v18, v9, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v25, v24, v23
	v_fma_f32 v26, -v20, v25, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v25, v26, v23
	v_fma_f32 v20, -v20, v25, v24
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v20, v20, v23, v25
	v_div_fixup_f32 v18, v20, v9, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v18, v18
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v18, v18, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v20, v18
.LBB0_29:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_div_scale_f32 v18, null, v9, v9, v17
	s_mov_b32 s1, 0xc1000000
	v_rcp_f32_e32 v19, v18
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v23, -v18, v19, 1.0
	v_fmac_f32_e32 v19, v23, v19
	v_div_scale_f32 v23, vcc_lo, v17, v9, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v24, v23, v19
	v_fma_f32 v25, -v18, v24, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v25, v19
	v_fma_f32 v18, -v18, v24, v23
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v18, v18, v19, v24
	v_div_fixup_f32 v17, v18, v9, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v17, v17
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v17, v17, s1, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v19, v17
.LBB0_31:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v17, v21, v22
	v_and_b32_e32 v22, 15, v22
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s23, s22, 31
	s_mul_u64 s[2:3], s[16:17], 0x48
	v_add3_u32 v17, v17, v20, v19
	v_lshl_or_b32 v21, v21, 4, v22
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[4:5], s[20:21], s[22:23]
	v_cmp_eq_u32_e64 s1, 0, v10
	s_add_nc_u64 s[2:3], s[14:15], s[2:3]
	ds_bpermute_b32 v18, v11, v17
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[4:5], s[4:5], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[4:5], s[2:3], s[4:5]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v17, v17, v18
	ds_bpermute_b32 v18, v12, v17
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v17, v17, v18
	ds_bpermute_b32 v18, v13, v17
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v17, v17, v18
	ds_bpermute_b32 v18, v15, v17
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v18, v17, v18
	v_and_b32_e32 v17, 15, v20
	ds_bpermute_b32 v20, v14, v18
	v_lshl_or_b32 v17, v19, 4, v17
	v_and_b16 v19.h, 0xff, v21.l
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b16 v19.l, 8, v17.l
	v_dual_mov_b32 v0, 0 :: v_dual_lshlrev_b32 v17, 1, v0
	v_or_b16 v10.l, v19.h, v19.l
	global_store_b16 v17, v10, s[4:5] offset:8
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_33
; %bb.32:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v10, v18, v20
	global_store_b64 v0, v[9:10], s[4:5]
.LBB0_33:                               ; %_Z26quantize_block_i4_128_waveILb1EEvPKfP12block_i4_128i.exit.i
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	ds_bpermute_b32 v0, v16, v1 offset:64
	ds_bpermute_b32 v1, v16, v2 offset:64
	ds_bpermute_b32 v2, v16, v3 offset:64
	ds_bpermute_b32 v9, v16, v4 offset:64
	ds_bpermute_b32 v3, v16, v5 offset:64
	ds_bpermute_b32 v5, v16, v6 offset:64
	ds_bpermute_b32 v6, v16, v7 offset:64
	ds_bpermute_b32 v7, v16, v8 offset:64
	s_mov_b32 s6, exec_lo
	s_wait_dscnt 0x3
	v_cndmask_b32_e64 v4, v3, v0, s0
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v3, v5, v1, s0
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v2, v6, v2, s0
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v1, v7, v9, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_max3_num_f32 v0, |v4|, |v3|, |v2|
	v_max_num_f32_e64 v5, |v1|, |v1|
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v5
	ds_bpermute_b32 v5, v11, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v0, v0, v5
	ds_bpermute_b32 v5, v12, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v5
	ds_bpermute_b32 v5, v13, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_max_num_f32_e32 v0, v0, v5
	ds_bpermute_b32 v5, v15, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v0, v0, v5
	ds_bpermute_b32 v5, v14, v0
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v5, v5, v5
	v_dual_max_num_f32 v5, v0, v5 :: v_dual_mov_b32 v0, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v5
	s_cbranch_execz .LBB0_45
; %bb.34:
	v_div_scale_f32 v0, null, 0x40e00000, 0x40e00000, v5
	v_div_scale_f32 v8, vcc_lo, v5, 0x40e00000, v5
	s_mov_b32 s7, 0
	s_mov_b32 s8, 0xc1000000
	v_rcp_f32_e32 v6, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v7, -v0, v6, 1.0
	v_fmac_f32_e32 v6, v7, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, v8, v6
	v_fma_f32 v9, -v0, v7, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v7, v9, v6
	v_fma_f32 v0, -v0, v7, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v0, v0, v6, v7
	v_mov_b32_e32 v7, 0x7149f2ca
	v_div_fixup_f32 v6, v0, 0x40e00000, v5
	v_mov_b32_e32 v0, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v6, 0.5, v6
	s_branch .LBB0_36
.LBB0_35:                               ; %.preheader.preheader.i.1.i
                                        ;   in Loop: Header=BB0_36 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mul_f32_e32 v8, s0, v6
	s_add_co_i32 s7, s7, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s7, 4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v9, null, v8, v8, v4
	v_rcp_f32_e32 v19, v9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v25, -v9, v19, 1.0
	v_fmac_f32_e32 v19, v25, v19
	v_div_scale_f32 v10, null, v8, v8, v3
	v_div_scale_f32 v16, null, v8, v8, v2
	v_div_scale_f32 v18, null, v8, v8, v1
	v_div_scale_f32 v23, vcc_lo, v4, v8, v4
	v_div_scale_f32 v24, s0, v3, v8, v3
	v_rcp_f32_e32 v20, v10
	v_rcp_f32_e32 v21, v16
	v_rcp_f32_e32 v22, v18
	v_div_scale_f32 v28, s2, v2, v8, v2
	v_div_scale_f32 v25, s3, v1, v8, v1
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v26, -v10, v20, 1.0
	v_fma_f32 v27, -v16, v21, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v29, -v18, v22, 1.0
	v_dual_fmac_f32 v20, v26, v20 :: v_dual_fmac_f32 v21, v27, v21
	v_mul_f32_e32 v26, v23, v19
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v22, v29, v22 :: v_dual_mul_f32 v27, v24, v20
	v_mul_f32_e32 v29, v28, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v31, -v9, v26, v23
	v_fma_f32 v32, -v10, v27, v24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v33, -v16, v29, v28
	v_fmac_f32_e32 v26, v31, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v30, v25, v22 :: v_dual_fmac_f32 v27, v32, v20
	v_fmac_f32_e32 v29, v33, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v9, -v9, v26, v23
	v_fma_f32 v34, -v18, v30, v25
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v10, -v10, v27, v24
	v_fma_f32 v16, -v16, v29, v28
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v9, v9, v19, v26
	s_mov_b32 vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v10, v10, v20, v27
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v9, v9, v8, v4
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v16, v16, v21, v29
	s_mov_b32 vcc_lo, s3
	v_div_fixup_f32 v10, v10, v8, v3
	v_rndne_f32_e32 v9, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v16, v16, v8, v2
	v_rndne_f32_e32 v10, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v9, v9, s8, 0x40e00000
	v_rndne_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v10, v10, s8, 0x40e00000
	v_fma_f32 v9, -v9, v8, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v16, v16, s8, 0x40e00000
	v_fma_f32 v10, -v10, v8, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v9, v9, v9, 0
	v_fma_f32 v16, -v16, v8, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v10
	v_dual_fmac_f32 v30, v34, v22 :: v_dual_fmac_f32 v9, v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v18, v30, v25
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v22, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v18, v18, v8, v1
	v_rndne_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_med3_num_f32 v18, v18, s8, 0x40e00000
	v_fma_f32 v10, -v18, v8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v10
	ds_bpermute_b32 v10, v11, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v12, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v13, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v15, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v14, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v9, v7
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v7, v7, v9 :: v_dual_cndmask_b32 v0, v0, v8
	s_cbranch_scc0 .LBB0_45
.LBB0_36:                               ; %NodeBlock588
                                        ; =>This Inner Loop Header: Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lt_i32 s7, 1
	s_mov_b32 s0, 1.0
	s_cbranch_scc1 .LBB0_35
; %bb.37:                               ; %NodeBlock586
                                        ;   in Loop: Header=BB0_36 Depth=1
	s_cmp_lt_i32 s7, 2
	s_mov_b32 s2, -1
                                        ; implicit-def: $sgpr0
	s_cbranch_scc1 .LBB0_43
; %bb.38:                               ; %LeafBlock584
                                        ;   in Loop: Header=BB0_36 Depth=1
	s_cmp_lg_u32 s7, 2
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_36 Depth=1
	s_mov_b32 s0, 0
.LBB0_40:                               ; %Flow590
                                        ;   in Loop: Header=BB0_36 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, 2.0
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_36 Depth=1
	s_mov_b32 s0, 0x3fdb6db7
.LBB0_42:                               ; %Flow591
                                        ;   in Loop: Header=BB0_36 Depth=1
	s_mov_b32 s2, 0
.LBB0_43:                               ; %Flow592
                                        ;   in Loop: Header=BB0_36 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_35
; %bb.44:                               ;   in Loop: Header=BB0_36 Depth=1
	s_mov_b32 s0, 0x3fa49249
	s_branch .LBB0_35
.LBB0_45:                               ; %Flow595
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s6
	v_cmp_neq_f32_e64 s0, 0, v5
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_div_scale_f32 v6, null, v0, v0, v4
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v7, v6
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v8, -v6, v7, 1.0
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v4, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v9, v8, v7
	v_fma_f32 v10, -v6, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v7
	v_fma_f32 v6, -v6, v9, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v6, v6, v7, v9
	v_div_fixup_f32 v4, v6, v0, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v4, v4
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v4, v4, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v6, v4
.LBB0_47:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_div_scale_f32 v4, null, v0, v0, v3
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v5, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v7, -v4, v5, 1.0
	v_fmac_f32_e32 v5, v7, v5
	v_div_scale_f32 v7, vcc_lo, v3, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v8, v7, v5
	v_fma_f32 v9, -v4, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v8, v9, v5
	v_fma_f32 v4, -v4, v8, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v5, v8
	v_div_fixup_f32 v3, v4, v0, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v3, v3
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v3, v3, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v5, v3
.LBB0_49:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_div_scale_f32 v4, null, v0, v0, v2
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v7, v4
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v8, -v4, v7, 1.0
	v_fmac_f32_e32 v7, v8, v7
	v_div_scale_f32 v8, vcc_lo, v2, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v9, v8, v7
	v_fma_f32 v10, -v4, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v7
	v_fma_f32 v4, -v4, v9, v8
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v4, v4, v7, v9
	v_div_fixup_f32 v2, v4, v0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v2, v2
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v2, v2, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v4, v2
.LBB0_51:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_53
; %bb.52:
	v_div_scale_f32 v2, null, v0, v0, v1
	s_mov_b32 s0, 0xc1000000
	v_rcp_f32_e32 v3, v2
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v7, -v2, v3, 1.0
	v_fmac_f32_e32 v3, v7, v3
	v_div_scale_f32 v7, vcc_lo, v1, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v8, v7, v3
	v_fma_f32 v9, -v2, v8, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v8, v9, v3
	v_fma_f32 v2, -v2, v8, v7
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v2, v2, v3, v8
	v_div_fixup_f32 v1, v2, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v1, v1
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v1, v1, s0, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v3, v1
.LBB0_53:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v1, v5, v6
	v_and_b32_e32 v6, 15, v6
	s_mul_u64 s[2:3], s[20:21], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[4:5], s[2:3]
	v_add3_u32 v1, v1, v4, v3
	v_and_b32_e32 v4, 15, v4
	ds_bpermute_b32 v2, v11, v1
	v_lshl_or_b32 v3, v3, 4, v4
	v_lshl_or_b32 v4, v5, 4, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b16 v3.l, 8, v3.l
	v_and_b16 v3.h, 0xff, v4.l
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, s0, s2, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s3, 0, s0
	v_or_b16 v3.l, v3.h, v3.l
	global_store_b16 v[4:5], v3, off offset:8
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v12, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v13, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v15, v1
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v14, v1
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_55
; %bb.54:
	s_wait_dscnt 0x0
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v1, v1, v2
	global_store_b64 v2, v[0:1], s[2:3]
.LBB0_55:                               ; %_Z31emit_iu4_sidecar_from_producer8ffffffffP12block_i4_128iiii.exit
	s_endpgm
.Lfunc_end0:
	.size	gated_norm_mq_rotate_awq_i4_gfx12, .Lfunc_end0-gated_norm_mq_rotate_awq_i4_gfx12
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gated_norm_mq_rotate_awq_i4_gfx12
		.amdhsa_group_segment_fixed_size 1024
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 84
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 45
		.amdhsa_next_free_sgpr 26
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-gated_norm_mq_rotate_awq_i4_gfx12)<<4)&4080)>>4
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.num_vgpr, 45
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.num_agpr, 0
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.numbered_sgpr, 26
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.num_named_barrier, 0
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.private_seg_size, 0
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.uses_vcc, 1
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.uses_flat_scratch, 0
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.has_dyn_sized_stack, 0
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.has_recursion, 0
	.set .Lgated_norm_mq_rotate_awq_i4_gfx12.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6652
; TotalNumSgprs: 28
; NumVgprs: 45
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 1024 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 28
; NumVGPRsForWavesPerEU: 45
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_9062a07188db61c1,@object ; @__hip_cuid_9062a07188db61c1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9062a07188db61c1
__hip_cuid_9062a07188db61c1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9062a07188db61c1, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9062a07188db61c1
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 84
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           gated_norm_mq_rotate_awq_i4_gfx12
    .private_segment_fixed_size: 0
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         gated_norm_mq_rotate_awq_i4_gfx12.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     45
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
