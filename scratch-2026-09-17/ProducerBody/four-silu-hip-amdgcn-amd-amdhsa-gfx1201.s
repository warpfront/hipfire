	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 6
	.text
	.protected	fused_silu_mul_mq_rotate_awq_i4_gfx12 ; -- Begin function fused_silu_mul_mq_rotate_awq_i4_gfx12
	.globl	fused_silu_mul_mq_rotate_awq_i4_gfx12
	.p2align	8
	.type	fused_silu_mul_mq_rotate_awq_i4_gfx12,@function
fused_silu_mul_mq_rotate_awq_i4_gfx12:  ; @fused_silu_mul_mq_rotate_awq_i4_gfx12
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b64 s[12:13], s[0:1], 0x38
	s_wait_kmcnt 0x0
	s_ashr_i32 s3, s12, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s2, s3, 24
	s_add_co_i32 s2, s12, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s2, s2, 8
	s_cmp_ge_i32 ttmp9, s2
	s_cbranch_scc1 .LBB0_47
; %bb.1:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 3, v0
	s_mov_b32 s14, ttmp7
	s_mov_b32 s15, 0
	s_mov_b32 s2, s12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_lshl_or_b32 v1, ttmp9, 8, v1
	s_mul_u64 s[2:3], s[2:3], s[14:15]
	s_lshl_b64 s[14:15], s[2:3], 2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshlrev_b64_e32 v[33:34], 2, v[1:2]
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[2:3], s[4:5], s[14:15]
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_add_co_u32 v5, vcc_lo, s2, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v6, null, s3, v34, vcc_lo
	s_add_nc_u64 s[2:3], s[6:7], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v7, vcc_lo, s2, v33
	global_load_b128 v[22:25], v[5:6], off
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, s3, v34, vcc_lo
	v_add_co_u32 v17, vcc_lo, s8, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s9, v34, vcc_lo
	global_load_b128 v[26:29], v[7:8], off
	global_load_b128 v[1:4], v[17:18], off
	global_load_b128 v[9:12], v[5:6], off offset:16
	s_wait_loadcnt 0x3
	v_dual_mul_f32 v5, 0xbfb8aa3b, v22 :: v_dual_mul_f32 v20, 0xbfb8aa3b, v23
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v6, 0xbfb8aa3b, v22, -v5
	v_rndne_f32_e32 v13, v5
	v_fma_f32 v30, 0xbfb8aa3b, v23, -v20
	v_rndne_f32_e32 v31, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v6, 0xb2a5705f, v22 :: v_dual_sub_f32 v5, v5, v13
	v_fmac_f32_e32 v30, 0xb2a5705f, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	v_dual_sub_f32 v20, v20, v31 :: v_dual_add_f32 v5, v5, v6
	v_cvt_i32_f32_e32 v6, v13
	global_load_b128 v[13:16], v[7:8], off offset:16
	v_add_f32_e32 v20, v20, v30
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v20, v20
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_ldexp_f32 v5, v5, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0, v5, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v22
	global_load_b128 v[5:8], v[17:18], off offset:16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v17, 0x7f800000, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v17, 1.0, v17
	v_div_scale_f32 v18, null, v17, v17, v22
	v_div_scale_f32 v32, vcc_lo, v22, v17, v22
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v19, v18
	v_fma_f32 v21, -v18, v19, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v19, v21, v19
	v_mul_f32_e32 v21, v32, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v30, -v18, v21, v32
	v_fmac_f32_e32 v21, v30, v19
	v_cvt_i32_f32_e32 v30, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v21, v32
	v_ldexp_f32 v20, v20, v30
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fmas_f32 v18, v18, v19, v21
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v23
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0, v20, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v23
	v_mul_f32_e32 v20, 0xbfb8aa3b, v24
	v_div_fixup_f32 v17, v18, v17, v22
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, 0x7f800000, v19, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_rndne_f32_e32 v30, v20
	s_wait_loadcnt 0x4
	v_mul_f32_e32 v35, v26, v17
	v_fma_f32 v26, 0xbfb8aa3b, v24, -v20
	v_add_f32_e32 v17, 1.0, v18
	v_sub_f32_e32 v20, v20, v30
	v_cvt_i32_f32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v26, 0xb2a5705f, v24
	v_div_scale_f32 v19, null, v17, v17, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, v20, v26
	s_wait_loadcnt 0x3
	v_div_scale_f32 v18, null, v1, v1, v35
	v_rcp_f32_e32 v22, v19
	v_div_scale_f32 v36, vcc_lo, v35, v1, v35
	v_exp_f32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v21, v18
	v_fma_f32 v32, -v19, v22, 1.0
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_ldexp_f32 v20, v20, v30
	v_fma_f32 v31, -v18, v21, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v22, v32, v22 :: v_dual_fmac_f32 v21, v31, v21
	v_mul_f32_e32 v32, v36, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v37, -v18, v32, v36
	v_fmac_f32_e32 v32, v37, v21
	v_div_scale_f32 v31, s2, v23, v17, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v18, -v18, v32, v36
	v_mul_f32_e32 v26, v31, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v36, v18, v21, v32
	v_fma_f32 v38, -v19, v26, v31
	s_mov_b32 vcc_lo, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v1, v36, v1, v35
	v_fmac_f32_e32 v26, v38, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v19, -v19, v26, v31
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v19, v22, v26
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v24
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v17, v18, v17, v23
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0, v20, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v24
	v_mul_f32_e32 v20, 0xbfb8aa3b, v25
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v37, v27, v17 :: v_dual_cndmask_b32 v18, 0x7f800000, v19
	v_fma_f32 v23, 0xbfb8aa3b, v25, -v20
	v_rndne_f32_e32 v26, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_div_scale_f32 v31, vcc_lo, v37, v2, v37
	v_add_f32_e32 v17, 1.0, v18
	v_div_scale_f32 v18, null, v2, v2, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v23, 0xb2a5705f, v25 :: v_dual_sub_f32 v20, v20, v26
	v_cvt_i32_f32_e32 v26, v26
	v_div_scale_f32 v19, null, v17, v17, v24
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v21, v18
	v_rcp_f32_e32 v22, v19
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v18, v21, 1.0
	v_fma_f32 v30, -v19, v22, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v21, v27, v21
	v_div_scale_f32 v27, s2, v24, v17, v24
	v_fmac_f32_e32 v22, v30, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v20, v20, v23 :: v_dual_mul_f32 v23, v27, v22
	v_exp_f32_e32 v20, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v38, -v19, v23, v27
	v_dual_mul_f32 v30, v31, v21 :: v_dual_fmac_f32 v23, v38, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v32, -v18, v30, v31
	v_ldexp_f32 v20, v20, v26
	v_lshlrev_b32_e32 v26, 5, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v19, -v19, v23, v27
	v_fmac_f32_e32 v30, v32, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v18, v30, v31
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v38, v18, v21, v30
	s_mov_b32 vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v19, v22, v23
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v25
	v_div_fixup_f32 v2, v38, v2, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v17, v18, v17, v24
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v19, 0, v20, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v25
	v_mul_f32_e32 v39, v28, v17
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v17, 0xbfb8aa3b, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v18, 0x7f800000, v19, vcc_lo
	v_div_scale_f32 v28, null, v3, v3, v39
	v_div_scale_f32 v40, vcc_lo, v39, v3, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_f32_e32 v27, 1.0, v18
	v_fma_f32 v18, 0xbfb8aa3b, v9, -v17
	v_rcp_f32_e32 v31, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v18, 0xb2a5705f, v9
	v_div_scale_f32 v30, null, v27, v27, v25
	v_div_scale_f32 v41, s2, v25, v27, v25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v32, v30
	v_fma_f32 v19, -v28, v31, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v20, -v30, v32, 1.0
	v_fmac_f32_e32 v32, v20, v32
	v_rndne_f32_e32 v21, v17
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v43, v41, v32
	v_sub_f32_e32 v17, v17, v21
	v_fmac_f32_e32 v31, v19, v31
	v_cvt_i32_f32_e32 v45, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_add_f32_e32 v22, v17, v18
	global_load_b128 v[17:20], v26, s[10:11]
	v_mul_f32_e32 v42, v40, v31
	v_exp_f32_e32 v44, v22
	v_fma_f32 v22, -v30, v43, v41
	v_fma_f32 v23, -v28, v42, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_fmac_f32 v43, v22, v32 :: v_dual_fmac_f32 v42, v23, v31
	global_load_b128 v[21:24], v26, s[10:11] offset:16
	v_ldexp_f32 v44, v44, v45
	s_load_b256 s[4:11], s[0:1], 0x20
	v_fma_f32 v30, -v30, v43, v41
	v_fma_f32 v28, -v28, v42, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v40, v28, v31, v42
	s_mov_b32 vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v28, v30, v32, v43
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v9
	v_div_fixup_f32 v3, v40, v3, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v25, v28, v27, v25
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v30, 0, v44, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v9
	v_mul_f32_e32 v41, v29, v25
	v_mul_f32_e32 v29, 0xbfb8aa3b, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v27, 0x7f800000, v30, vcc_lo
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[6:7], 0
	v_div_scale_f32 v45, vcc_lo, v41, v4, v41
	v_fma_f32 v32, 0xbfb8aa3b, v10, -v29
	v_add_f32_e32 v25, 1.0, v27
	v_div_scale_f32 v27, null, v4, v4, v41
	v_rndne_f32_e32 v42, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fmac_f32_e32 v32, 0xb2a5705f, v10
	v_div_scale_f32 v28, null, v25, v25, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v30, v27
	v_rcp_f32_e32 v31, v28
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v43, -v27, v30, 1.0
	v_fma_f32 v44, -v28, v31, 1.0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v30, v43, v30
	v_div_scale_f32 v43, s2, v9, v25, v9
	v_dual_fmac_f32 v31, v44, v31 :: v_dual_mul_f32 v44, v45, v30
	v_sub_f32_e32 v29, v29, v42
	v_cvt_i32_f32_e32 v42, v42
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v46, -v27, v44, v45
	v_dual_add_f32 v29, v29, v32 :: v_dual_mul_f32 v32, v43, v31
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v44, v46, v30
	v_exp_f32_e32 v29, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v47, -v28, v32, v43
	v_fma_f32 v27, -v27, v44, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fmac_f32_e32 v32, v47, v31
	v_ldexp_f32 v29, v29, v42
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fmas_f32 v42, v27, v30, v44
	v_fma_f32 v28, -v28, v32, v43
	s_mov_b32 vcc_lo, s2
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v4, v42, v4, v41
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v27, v28, v31, v32
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_div_fixup_f32 v9, v27, v25, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v28, 0, v29, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v10
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v9, v13, v9
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v25, 0x7f800000, v28, vcc_lo
	v_mul_f32_e32 v28, 0xbfb8aa3b, v11
	s_wait_loadcnt 0x2
	v_div_scale_f32 v13, null, v5, v5, v9
	v_div_scale_f32 v45, vcc_lo, v9, v5, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_rndne_f32_e32 v32, v28
	v_add_f32_e32 v25, 1.0, v25
	v_fma_f32 v31, 0xbfb8aa3b, v11, -v28
	v_rcp_f32_e32 v29, v13
	v_sub_f32_e32 v28, v28, v32
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_scale_f32 v27, null, v25, v25, v10
	v_fmac_f32_e32 v31, 0xb2a5705f, v11
	v_cvt_i32_f32_e32 v32, v32
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v30, v27
	v_fma_f32 v43, -v13, v29, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_add_f32 v28, v28, v31 :: v_dual_fmac_f32 v29, v43, v29
	v_div_scale_f32 v43, s2, v10, v25, v10
	v_exp_f32_e32 v28, v28
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v44, -v27, v30, 1.0
	v_fmac_f32_e32 v30, v44, v30
	v_mul_f32_e32 v44, v45, v29
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_ldexp_f32 v28, v28, v32
	v_mul_f32_e32 v31, v43, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v46, -v13, v44, v45
	v_fma_f32 v47, -v27, v31, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v44, v46, v29 :: v_dual_fmac_f32 v31, v47, v30
	v_fma_f32 v13, -v13, v44, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v27, -v27, v31, v43
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v13, v13, v29, v44
	s_mov_b32 vcc_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v27, v27, v30, v31
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v11
	v_div_fixup_f32 v5, v13, v5, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v10, v27, v25, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v28, 0, v28, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v11
	s_wait_alu depctr_va_vcc(0)
	v_dual_mul_f32 v10, v14, v10 :: v_dual_cndmask_b32 v25, 0x7f800000, v28
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_div_scale_f32 v44, null, v6, v6, v10
	v_div_scale_f32 v48, vcc_lo, v10, v6, v10
	v_dual_add_f32 v14, 1.0, v25 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v46, v44
	v_div_scale_f32 v43, null, v14, v14, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fma_f32 v27, 0xbfb8aa3b, v12, -v25
	v_rndne_f32_e32 v47, v25
	v_div_scale_f32 v50, s0, v11, v14, v11
	v_rcp_f32_e32 v45, v43
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v28, -v44, v46, 1.0
	v_fmac_f32_e32 v27, 0xb2a5705f, v12
	v_sub_f32_e32 v25, v25, v47
	v_cvt_i32_f32_e32 v47, v47
	v_fmac_f32_e32 v46, v28, v46
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v29, -v43, v45, 1.0
	v_add_f32_e32 v25, v25, v27
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v49, v48, v46
	v_fmac_f32_e32 v45, v29, v45
	global_load_b128 v[29:32], v26, s[4:5]
	v_exp_f32_e32 v53, v25
	v_fma_f32 v52, -v44, v49, v48
	global_load_b128 v[25:28], v26, s[4:5] offset:16
	v_mul_f32_e32 v51, v50, v45
	v_fmac_f32_e32 v49, v52, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_f32 v54, -v43, v51, v50
	v_ldexp_f32 v47, v53, v47
	v_fma_f32 v44, -v44, v49, v48
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v51, v54, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v44, v44, v46, v49
	s_mov_b32 vcc_lo, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v43, -v43, v51, v50
	v_div_fixup_f32 v6, v44, v6, v10
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v43, v43, v45, v51
	v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v12
	v_div_fixup_f32 v11, v43, v14, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v45, 0, v47, vcc_lo
	v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v12
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v11, v15, v11 :: v_dual_cndmask_b32 v14, 0x7f800000, v45
	v_add_f32_e32 v14, 1.0, v14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_scale_f32 v43, null, v14, v14, v12
	v_rcp_f32_e32 v46, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v48, -v43, v46, 1.0
	v_fmac_f32_e32 v46, v48, v46
	s_wait_loadcnt 0x3
	v_mul_f32_e32 v2, v2, v18
	v_div_scale_f32 v15, null, v7, v7, v11
	v_div_scale_f32 v49, vcc_lo, v11, v7, v11
	v_mul_f32_e32 v4, v4, v20
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_fma_f32 v10, v1, v17, v2
	v_rcp_f32_e32 v45, v15
	v_fma_f32 v1, v1, v17, -v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
	v_fma_f32 v2, v3, v19, -v4
	s_wait_loadcnt 0x2
	v_mul_f32_e32 v6, v6, v22
	v_fma_f32 v47, -v15, v45, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v45, v47, v45
	v_div_scale_f32 v47, s0, v12, v14, v12
	v_mul_f32_e32 v48, v49, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v50, v47, v46
	v_fma_f32 v51, -v15, v48, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v52, -v43, v50, v47
	v_fmac_f32_e32 v48, v51, v45
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fmac_f32_e32 v50, v52, v46
	v_fma_f32 v15, -v15, v48, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v43, -v43, v50, v47
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v15, v15, v45, v48
	s_mov_b32 vcc_lo, s0
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v43, v43, v46, v50
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_div_fixup_f32 v7, v15, v7, v11
	v_div_fixup_f32 v12, v43, v14, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v12, v16, v12
	v_div_scale_f32 v14, null, v8, v8, v12
	v_div_scale_f32 v45, vcc_lo, v12, v8, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v16, v14
	v_fma_f32 v43, -v14, v16, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v43, v16
	v_mul_f32_e32 v43, v45, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v46, -v14, v43, v45
	v_fmac_f32_e32 v43, v46, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v14, -v14, v43, v45
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v14, v14, v16, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v8, v14, v8, v12
	v_fma_f32 v12, v3, v19, v4
	v_fma_f32 v4, v5, v21, v6
	v_fma_f32 v5, v5, v21, -v6
	v_add_f32_e32 v3, v10, v12
	v_mul_f32_e32 v8, v8, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_fma_f32 v9, v7, v23, v8
	v_fma_f32 v6, v7, v23, -v8
	v_and_b32_e32 v16, 1, v0
	v_dual_sub_f32 v7, v10, v12 :: v_dual_add_f32 v8, v1, v2
	v_dual_add_f32 v10, v4, v9 :: v_dual_sub_f32 v1, v1, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v2, v5, v6
	v_dual_sub_f32 v4, v4, v9 :: v_dual_sub_f32 v5, v5, v6
	v_dual_add_f32 v6, v3, v10 :: v_dual_sub_f32 v3, v3, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_add_f32 v9, v8, v2 :: v_dual_sub_f32 v2, v8, v2
	v_add_f32_e32 v8, v7, v4
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_sub_f32 v4, v7, v4 :: v_dual_add_f32 v7, v1, v5
	v_sub_f32_e32 v1, v1, v5
	ds_swizzle_b32 v5, v6 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v13, v3 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v10, v9 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v12, v7 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v17, v1 offset:swizzle(SWAP,1)
	v_cmp_eq_u32_e64 s0, 0, v16
	ds_swizzle_b32 v14, v2 offset:swizzle(SWAP,1)
	s_wait_alu depctr_va_sdst(0)
	v_cndmask_b32_e64 v6, -v6, v6, s0
	v_cndmask_b32_e64 v7, -v7, v7, s0
	v_cndmask_b32_e64 v3, -v3, v3, s0
	v_and_b32_e32 v16, 2, v0
	v_cndmask_b32_e64 v9, -v9, v9, s0
	v_cndmask_b32_e64 v1, -v1, v1, s0
	v_cndmask_b32_e64 v2, -v2, v2, s0
	s_wait_dscnt 0x5
	v_add_f32_e32 v5, v6, v5
	s_wait_dscnt 0x4
	v_add_f32_e32 v3, v3, v13
	v_cmp_eq_u32_e32 vcc_lo, 0, v16
	s_wait_dscnt 0x2
	v_add_f32_e32 v7, v7, v12
	ds_swizzle_b32 v11, v8 offset:swizzle(SWAP,1)
	s_wait_dscnt 0x2
	v_add_f32_e32 v1, v1, v17
	ds_swizzle_b32 v15, v4 offset:swizzle(SWAP,1)
	ds_swizzle_b32 v13, v3 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v12, v7 offset:swizzle(SWAP,2)
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	v_add_f32_e32 v6, v9, v10
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v17, v1 offset:swizzle(SWAP,2)
	v_cndmask_b32_e64 v8, -v8, v8, s0
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	v_cndmask_b32_e64 v4, -v4, v4, s0
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x6
	v_add_f32_e32 v2, v2, v14
	s_wait_dscnt 0x5
	v_add_f32_e32 v8, v8, v11
	s_wait_dscnt 0x3
	v_dual_add_f32 v4, v4, v15 :: v_dual_add_f32 v3, v3, v13
	s_wait_dscnt 0x2
	v_add_f32_e32 v7, v7, v12
	ds_swizzle_b32 v10, v6 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v14, v2 offset:swizzle(SWAP,2)
	s_wait_dscnt 0x3
	v_add_f32_e32 v5, v5, v9
	s_wait_dscnt 0x2
	v_add_f32_e32 v1, v1, v17
	ds_swizzle_b32 v12, v7 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v11, v8 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v15, v4 offset:swizzle(SWAP,2)
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,4)
	v_and_b32_e32 v16, 4, v0
	ds_swizzle_b32 v13, v3 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v17, v1 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	s_wait_dscnt 0x7
	v_add_f32_e32 v6, v6, v10
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x4
	v_dual_add_f32 v8, v8, v11 :: v_dual_add_f32 v7, v7, v12
	s_wait_dscnt 0x2
	v_dual_add_f32 v5, v5, v9 :: v_dual_add_f32 v2, v2, v14
	ds_swizzle_b32 v10, v6 offset:swizzle(SWAP,4)
	s_wait_dscnt 0x2
	v_add_f32_e32 v3, v3, v13
	s_wait_dscnt 0x1
	v_add_f32_e32 v1, v1, v17
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,8)
	v_add_f32_e32 v4, v4, v15
	ds_swizzle_b32 v12, v7 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v11, v8 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v13, v3 offset:swizzle(SWAP,8)
	v_and_b32_e32 v16, 8, v0
	ds_swizzle_b32 v15, v4 offset:swizzle(SWAP,4)
	ds_swizzle_b32 v17, v1 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v14, v2 offset:swizzle(SWAP,4)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	s_wait_dscnt 0x7
	v_add_f32_e32 v6, v6, v10
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x6
	v_add_f32_e32 v5, v5, v9
	s_wait_dscnt 0x4
	v_dual_add_f32 v7, v7, v12 :: v_dual_add_f32 v8, v8, v11
	ds_swizzle_b32 v10, v6 offset:swizzle(SWAP,8)
	s_wait_dscnt 0x4
	v_add_f32_e32 v3, v3, v13
	s_wait_dscnt 0x2
	v_add_f32_e32 v1, v1, v17
	ds_swizzle_b32 v9, v5 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v12, v7 offset:swizzle(SWAP,16)
	v_add_f32_e32 v4, v4, v15
	s_wait_dscnt 0x3
	v_add_f32_e32 v2, v2, v14
	ds_swizzle_b32 v11, v8 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v13, v3 offset:swizzle(SWAP,16)
	v_and_b32_e32 v16, 16, v0
	ds_swizzle_b32 v15, v4 offset:swizzle(SWAP,8)
	ds_swizzle_b32 v17, v1 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v2 offset:swizzle(SWAP,8)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v5, -v5, v5, vcc_lo
	v_cndmask_b32_e64 v1, -v1, v1, vcc_lo
	s_wait_dscnt 0x7
	v_add_f32_e32 v6, v6, v10
	v_cndmask_b32_e64 v7, -v7, v7, vcc_lo
	v_cndmask_b32_e64 v3, -v3, v3, vcc_lo
	s_wait_dscnt 0x4
	v_dual_add_f32 v5, v5, v9 :: v_dual_add_f32 v8, v8, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_e32 v7, v7, v12
	s_wait_dscnt 0x1
	v_add_f32_e32 v1, v1, v17
	ds_swizzle_b32 v10, v6 offset:swizzle(SWAP,16)
	v_dual_mul_f32 v5, 0x3d800000, v5 :: v_dual_add_f32 v4, v4, v15
	s_wait_dscnt 0x1
	v_dual_add_f32 v2, v2, v14 :: v_dual_mul_f32 v7, 0x3d800000, v7
	ds_swizzle_b32 v11, v8 offset:swizzle(SWAP,16)
	v_mul_f32_e32 v12, 0x3d800000, v1
	s_wait_loadcnt 0x1
	v_mul_f32_e32 v1, v5, v29
	ds_swizzle_b32 v15, v4 offset:swizzle(SWAP,16)
	ds_swizzle_b32 v14, v2 offset:swizzle(SWAP,16)
	v_cndmask_b32_e64 v6, -v6, v6, vcc_lo
	v_add_f32_e32 v3, v3, v13
	v_cndmask_b32_e64 v8, -v8, v8, vcc_lo
	v_cndmask_b32_e64 v4, -v4, v4, vcc_lo
	v_cndmask_b32_e64 v2, -v2, v2, vcc_lo
	s_wait_dscnt 0x3
	v_dual_mul_f32 v9, 0x3d800000, v3 :: v_dual_add_f32 v6, v6, v10
	s_wait_loadcnt_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v5, v9, v25 :: v_dual_add_f32 v4, v4, v15
	v_dual_add_f32 v8, v8, v11 :: v_dual_mul_f32 v11, 0x3d800000, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, 0x3d800000, v8
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v2, v14
	v_mul_f32_e32 v6, 0x3d800000, v6
	v_mul_f32_e32 v4, v7, v32
	v_mul_f32_e32 v7, v11, v27
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v3, v8, v31 :: v_dual_mul_f32 v10, 0x3d800000, v2
	v_mul_f32_e32 v2, v6, v30
	v_mul_f32_e32 v8, v12, v28
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v6, v10, v26
	s_cbranch_scc1 .LBB0_3
; %bb.2:
	s_add_nc_u64 s[2:3], s[6:7], s[14:15]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v9, vcc_lo, s2, v33
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v10, null, s3, v34, vcc_lo
	s_clause 0x1
	global_store_b128 v[9:10], v[1:4], off
	global_store_b128 v[9:10], v[5:8], off offset:16
.LBB0_3:
	v_mbcnt_lo_u32_b32 v21, -1, 0
	v_lshrrev_b32_e32 v9, 1, v0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_xor_b32_e32 v22, 16, v21
	v_cmp_gt_u32_e32 vcc_lo, 32, v22
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v16, 2, v9
	ds_bpermute_b32 v9, v16, v1
	ds_bpermute_b32 v13, v16, v5
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v19, v13, v9, s0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v9, v21, v22, vcc_lo
	ds_bpermute_b32 v10, v16, v2
	ds_bpermute_b32 v11, v16, v3
	ds_bpermute_b32 v12, v16, v4
	ds_bpermute_b32 v14, v16, v6
	ds_bpermute_b32 v15, v16, v7
	ds_bpermute_b32 v20, v16, v8
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v18, v14, v10, s0
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v17, v15, v11, s0
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v10, v20, v12, s0
	v_lshlrev_b32_e32 v11, 2, v9
	v_xor_b32_e32 v20, 1, v21
	v_max3_num_f32 v12, |v19|, |v18|, |v17|
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max_num_f32_e64 v13, |v10|, |v10|
	v_max_num_f32_e32 v9, v12, v13
	v_xor_b32_e32 v13, 8, v21
	ds_bpermute_b32 v12, v11, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v13
	s_wait_dscnt 0x0
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v13, v21, v13 :: v_dual_max_num_f32 v14, v12, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_max_num_f32 v9, v9, v14 :: v_dual_lshlrev_b32 v12, 2, v13
	v_xor_b32_e32 v14, 4, v21
	ds_bpermute_b32 v13, v12, v9
	v_cmp_gt_u32_e32 vcc_lo, 32, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v14, v21, v14, vcc_lo
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v15, v13, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v9, v9, v15
	v_xor_b32_e32 v15, 2, v21
	v_cmp_gt_u32_e32 vcc_lo, 32, v15
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v15, v21, v15, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v20
	v_lshlrev_b32_e32 v13, 2, v14
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v20, v21, v20 :: v_dual_lshlrev_b32 v15, 2, v15
	ds_bpermute_b32 v14, v13, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v14, v14, v14
	v_max_num_f32_e32 v9, v9, v14
	ds_bpermute_b32 v14, v15, v9
	s_wait_dscnt 0x0
	v_dual_max_num_f32 v21, v14, v14 :: v_dual_lshlrev_b32 v14, 2, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_max_num_f32_e32 v9, v9, v21
	ds_bpermute_b32 v20, v14, v9
	s_wait_dscnt 0x0
	v_max_num_f32_e32 v20, v20, v20
	v_dual_max_num_f32 v20, v9, v20 :: v_dual_mov_b32 v9, 1.0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0, v20
	s_cbranch_execz .LBB0_15
; %bb.4:                                ; %.preheader79.i.i
	v_div_scale_f32 v9, null, 0x40e00000, 0x40e00000, v20
	v_div_scale_f32 v23, vcc_lo, v20, 0x40e00000, v20
	s_mov_b32 s5, 0x40e00000
	s_mov_b32 s6, 0
	s_mov_b32 s7, 0xc1000000
	v_rcp_f32_e32 v21, v9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v9, v21, 1.0
	v_fmac_f32_e32 v21, v22, v21
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v22, v23, v21
	v_fma_f32 v24, -v9, v22, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v22, v24, v21
	v_fma_f32 v9, -v9, v22, v23
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_fmas_f32 v9, v9, v21, v22
	v_mov_b32_e32 v22, 0x7149f2ca
	v_div_fixup_f32 v21, v9, 0x40e00000, v20
	v_mov_b32_e32 v9, 1.0
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v21, 0.5, v21
	s_branch .LBB0_6
.LBB0_5:                                ; %.preheader.preheader.i.i
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_div_scale_f32 v23, null, s5, s5, s1
	v_div_scale_f32 v26, vcc_lo, s1, 0x40e00000, s1
	s_add_co_i32 s6, s6, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s6, 4
	v_rcp_f32_e32 v24, v23
	v_xor_b32_e32 v23, 0x80000000, v23
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v25, v23, v24, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v25, v24
	v_mul_f32_e32 v25, v26, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v27, v23, v25, v26
	v_fmac_f32_e32 v25, v27, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v26, v23, v25
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v23, v26, v24, v25
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v23, v23, 0x40e00000, s1
	v_add_f32_e32 v23, 1.0, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v21, v23
	v_div_scale_f32 v24, null, v23, v23, v19
	v_div_scale_f32 v25, null, v23, v23, v18
	v_div_scale_f32 v26, null, v23, v23, v17
	v_div_scale_f32 v27, null, v23, v23, v10
	v_div_scale_f32 v33, s1, v18, v23, v18
	v_rcp_f32_e32 v28, v24
	v_rcp_f32_e32 v29, v25
	v_rcp_f32_e32 v30, v26
	v_rcp_f32_e32 v31, v27
	v_div_scale_f32 v32, vcc_lo, v19, v23, v19
	v_div_scale_f32 v37, s2, v17, v23, v17
	v_fma_f32 v34, -v24, v28, 1.0
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fma_f32 v35, -v25, v29, 1.0
	v_fma_f32 v36, -v26, v30, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v38, -v27, v31, 1.0
	v_dual_fmac_f32 v28, v34, v28 :: v_dual_fmac_f32 v29, v35, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_fmac_f32 v30, v36, v30 :: v_dual_fmac_f32 v31, v38, v31
	v_div_scale_f32 v34, s3, v10, v23, v10
	v_mul_f32_e32 v36, v33, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v38, v37, v30
	v_fma_f32 v41, -v25, v36, v33
	v_mul_f32_e32 v35, v32, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v42, -v26, v38, v37
	v_fmac_f32_e32 v36, v41, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v40, -v24, v35, v32
	v_fmac_f32_e32 v38, v42, v30
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v25, -v25, v36, v33
	v_fmac_f32_e32 v35, v40, v28
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v26, -v26, v38, v37
	v_fma_f32 v24, -v24, v35, v32
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fmas_f32 v24, v24, v28, v35
	s_mov_b32 vcc_lo, s1
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v25, v25, v29, v36
	s_mov_b32 vcc_lo, s2
	v_div_fixup_f32 v24, v24, v23, v19
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v26, v26, v30, v38
	s_mov_b32 vcc_lo, s3
	v_div_fixup_f32 v25, v25, v23, v18
	v_rndne_f32_e32 v24, v24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_div_fixup_f32 v26, v26, v23, v17
	v_rndne_f32_e32 v25, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v24, v24, s7, 0x40e00000
	v_rndne_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v25, v25, s7, 0x40e00000
	v_fma_f32 v24, -v24, v23, v19
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v26, v26, s7, 0x40e00000
	v_fma_f32 v25, -v25, v23, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_f32 v24, v24, v24, 0
	v_mul_f32_e32 v39, v34, v31
	v_fma_f32 v26, -v26, v23, v17
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fmac_f32_e32 v24, v25, v25
	v_fma_f32 v43, -v27, v39, v34
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v24, v26, v26 :: v_dual_fmac_f32 v39, v43, v31
	v_fma_f32 v27, -v27, v39, v34
	s_wait_alu depctr_sa_sdst(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v27, v27, v31, v39
	v_div_fixup_f32 v27, v27, v23, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v27, v27
	v_med3_num_f32 v27, v27, s7, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v25, -v27, v23, v10
	v_fmac_f32_e32 v24, v25, v25
	ds_bpermute_b32 v25, v11, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v25
	ds_bpermute_b32 v25, v12, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v25
	ds_bpermute_b32 v25, v13, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v25
	ds_bpermute_b32 v25, v15, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v25
	ds_bpermute_b32 v25, v14, v24
	s_wait_dscnt 0x0
	v_add_f32_e32 v24, v24, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_lt_f32_e32 vcc_lo, v24, v22
	s_wait_alu depctr_va_vcc(0)
	v_dual_cndmask_b32 v22, v22, v24 :: v_dual_cndmask_b32 v9, v9, v23
	s_cbranch_scc0 .LBB0_15
.LBB0_6:                                ; %NodeBlock546
                                        ; =>This Inner Loop Header: Depth=1
	s_cmp_lt_i32 s6, 1
	s_mov_b32 s1, 0
	s_cbranch_scc1 .LBB0_5
; %bb.7:                                ; %NodeBlock
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_cmp_lt_i32 s6, 2
	s_mov_b32 s2, -1
                                        ; implicit-def: $sgpr1
	s_cbranch_scc1 .LBB0_13
; %bb.8:                                ; %LeafBlock
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_cmp_lg_u32 s6, 2
	s_mov_b32 s1, -1
	s_cbranch_scc0 .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s1, 0
.LBB0_10:                               ; %Flow560
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, 0x40e00000
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s1, 0x40a00000
.LBB0_12:                               ; %Flow561
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s2, 0
.LBB0_13:                               ; %Flow562
                                        ;   in Loop: Header=BB0_6 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_5
; %bb.14:                               ;   in Loop: Header=BB0_6 Depth=1
	s_mov_b32 s1, 2.0
	s_branch .LBB0_5
.LBB0_15:                               ; %Flow565
	s_or_b32 exec_lo, exec_lo, s4
	v_cmp_neq_f32_e64 s1, 0, v20
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_17
; %bb.16:
	v_div_scale_f32 v21, null, v9, v9, v19
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v22, v21
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v23, -v21, v22, 1.0
	v_fmac_f32_e32 v22, v23, v22
	v_div_scale_f32 v23, vcc_lo, v19, v9, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v24, v23, v22
	v_fma_f32 v25, -v21, v24, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v25, v22
	v_fma_f32 v21, -v21, v24, v23
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v21, v21, v22, v24
	v_div_fixup_f32 v19, v21, v9, v19
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v19, v19
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v19, v19, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v21, v19
.LBB0_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_div_scale_f32 v19, null, v9, v9, v18
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v20, v19
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v19, v20, 1.0
	v_fmac_f32_e32 v20, v22, v20
	v_div_scale_f32 v22, vcc_lo, v18, v9, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v22, v20
	v_fma_f32 v24, -v19, v23, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v24, v20
	v_fma_f32 v19, -v19, v23, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v19, v19, v20, v23
	v_div_fixup_f32 v18, v19, v9, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v18, v18
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v18, v18, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v20, v18
.LBB0_19:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_div_scale_f32 v19, null, v9, v9, v17
	s_mov_b32 s3, 0xc1000000
	v_rcp_f32_e32 v22, v19
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v23, -v19, v22, 1.0
	v_fmac_f32_e32 v22, v23, v22
	v_div_scale_f32 v23, vcc_lo, v17, v9, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v24, v23, v22
	v_fma_f32 v25, -v19, v24, v23
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v24, v25, v22
	v_fma_f32 v19, -v19, v24, v23
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v19, v19, v22, v24
	v_div_fixup_f32 v17, v19, v9, v17
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v17, v17
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v17, v17, s3, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v19, v17
.LBB0_21:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_div_scale_f32 v17, null, v9, v9, v10
	s_mov_b32 s1, 0xc1000000
	v_rcp_f32_e32 v18, v17
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v22, -v17, v18, 1.0
	v_fmac_f32_e32 v18, v22, v18
	v_div_scale_f32 v22, vcc_lo, v10, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v23, v22, v18
	v_fma_f32 v24, -v17, v23, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v23, v24, v18
	v_fma_f32 v17, -v17, v23, v22
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v17, v17, v18, v23
	v_div_fixup_f32 v10, v17, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_rndne_f32_e32 v10, v10
	s_wait_alu depctr_sa_sdst(0)
	v_maxmin_num_f32 v10, v10, s1, 0x40e00000
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_i32_f32_e32 v18, v10
.LBB0_23:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v10, v20, v21
	v_and_b32_e32 v21, 15, v21
	s_lshl_b32 s6, ttmp9, 1
	s_mov_b32 s2, ttmp7
	s_ashr_i32 s3, ttmp7, 31
	v_add3_u32 v10, v10, v19, v18
	s_ashr_i32 s5, s13, 31
	s_mov_b32 s4, s13
	s_wait_alu depctr_sa_sdst(0)
	s_ashr_i32 s7, s6, 31
	s_mul_u64 s[2:3], s[2:3], 0x48
	ds_bpermute_b32 v17, v11, v10
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[6:7], s[4:5], s[6:7]
	v_cmp_eq_u32_e64 s1, 0, v0
	s_add_nc_u64 s[2:3], s[8:9], s[2:3]
	s_wait_alu depctr_sa_sdst(0)
	s_mul_u64 s[6:7], s[6:7], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[6:7], s[2:3], s[6:7]
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v10, v10, v17
	ds_bpermute_b32 v17, v12, v10
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v10, v10, v17
	ds_bpermute_b32 v17, v13, v10
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v10, v10, v17
	ds_bpermute_b32 v17, v15, v10
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v10, v10, v17
	v_and_b32_e32 v17, 15, v19
	ds_bpermute_b32 v19, v14, v10
	v_lshl_or_b32 v17, v18, 4, v17
	v_lshl_or_b32 v18, v20, 4, v21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b16 v20.l, 8, v17.l
	v_and_b16 v20.h, 0xff, v18.l
	v_dual_mov_b32 v18, 0 :: v_dual_lshlrev_b32 v17, 1, v0
	s_delay_alu instid0(VALU_DEP_2)
	v_or_b16 v0.l, v20.h, v20.l
	global_store_b16 v17, v0, s[6:7] offset:8
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_25
; %bb.24:
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v10, v10, v19
	global_store_b64 v18, v[9:10], s[6:7]
.LBB0_25:                               ; %_Z26quantize_block_i4_128_waveILb1EEvPKfP12block_i4_128i.exit.i
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
	s_mov_b32 s8, exec_lo
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
	s_cbranch_execz .LBB0_37
; %bb.26:                               ; %.preheader79.i.1.i
	v_div_scale_f32 v0, null, 0x40e00000, 0x40e00000, v5
	v_div_scale_f32 v8, vcc_lo, v5, 0x40e00000, v5
	s_mov_b32 s9, 0x40e00000
	s_mov_b32 s10, 0
	s_mov_b32 s11, 0xc1000000
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
	s_branch .LBB0_28
.LBB0_27:                               ; %.preheader.preheader.i.1.i
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_div_scale_f32 v8, null, s9, s9, s0
	v_div_scale_f32 v16, vcc_lo, s0, 0x40e00000, s0
	s_add_co_i32 s10, s10, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_lg_u32 s10, 4
	v_rcp_f32_e32 v9, v8
	v_xor_b32_e32 v8, 0x80000000, v8
	s_delay_alu instid0(TRANS32_DEP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v10, v8, v9, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v9
	v_mul_f32_e32 v10, v16, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v18, v8, v10, v16
	v_fmac_f32_e32 v10, v18, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v16, v8, v10
	s_wait_alu depctr_va_vcc(0)
	v_div_fmas_f32 v8, v16, v9, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v8, v8, 0x40e00000, s0
	v_add_f32_e32 v8, 1.0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v8, v6, v8
	v_div_scale_f32 v16, null, v8, v8, v2
	v_div_scale_f32 v18, null, v8, v8, v1
	v_div_scale_f32 v28, s2, v2, v8, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v21, v16
	v_rcp_f32_e32 v22, v18
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_fma_f32 v27, -v16, v21, 1.0
	v_fma_f32 v29, -v18, v22, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v21, v27, v21 :: v_dual_fmac_f32 v22, v29, v22
	v_mul_f32_e32 v29, v28, v21
	v_div_scale_f32 v9, null, v8, v8, v4
	v_div_scale_f32 v10, null, v8, v8, v3
	v_div_scale_f32 v23, vcc_lo, v4, v8, v4
	v_div_scale_f32 v24, s0, v3, v8, v3
	v_fma_f32 v33, -v16, v29, v28
	v_rcp_f32_e32 v19, v9
	v_rcp_f32_e32 v20, v10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_fmac_f32_e32 v29, v33, v21
	v_fma_f32 v25, -v9, v19, 1.0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v26, -v10, v20, 1.0
	v_fma_f32 v16, -v16, v29, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_fmac_f32 v19, v25, v19 :: v_dual_fmac_f32 v20, v26, v20
	v_div_scale_f32 v25, s3, v1, v8, v1
	v_dual_mul_f32 v26, v23, v19 :: v_dual_mul_f32 v27, v24, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_fma_f32 v31, -v9, v26, v23
	v_fma_f32 v32, -v10, v27, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v26, v31, v19 :: v_dual_fmac_f32 v27, v32, v20
	v_fma_f32 v9, -v9, v26, v23
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_fma_f32 v10, -v10, v27, v24
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
	v_med3_num_f32 v9, v9, s11, 0x40e00000
	v_rndne_f32_e32 v16, v16
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v10, v10, s11, 0x40e00000
	v_fma_f32 v9, -v9, v8, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_med3_num_f32 v16, v16, s11, 0x40e00000
	v_fma_f32 v10, -v10, v8, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_f32 v9, v9, v9, 0
	v_fma_f32 v16, -v16, v8, v2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v9, v10, v10
	v_dual_mul_f32 v30, v25, v22 :: v_dual_fmac_f32 v9, v16, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v34, -v18, v30, v25
	v_fmac_f32_e32 v30, v34, v22
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_f32 v18, -v18, v30, v25
	s_wait_alu depctr_sa_sdst(0)
	v_div_fmas_f32 v18, v18, v22, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fixup_f32 v18, v18, v8, v1
	v_rndne_f32_e32 v18, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_med3_num_f32 v18, v18, s11, 0x40e00000
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
	s_cbranch_scc0 .LBB0_37
.LBB0_28:                               ; %NodeBlock552
                                        ; =>This Inner Loop Header: Depth=1
	s_cmp_lt_i32 s10, 1
	s_mov_b32 s0, 0
	s_cbranch_scc1 .LBB0_27
; %bb.29:                               ; %NodeBlock550
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_lt_i32 s10, 2
	s_mov_b32 s2, -1
                                        ; implicit-def: $sgpr0
	s_cbranch_scc1 .LBB0_35
; %bb.30:                               ; %LeafBlock548
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_cmp_lg_u32 s10, 2
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s0, 0
.LBB0_32:                               ; %Flow554
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s0, 0x40e00000
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s0, 0x40a00000
.LBB0_34:                               ; %Flow555
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s2, 0
.LBB0_35:                               ; %Flow556
                                        ;   in Loop: Header=BB0_28 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB0_27
; %bb.36:                               ;   in Loop: Header=BB0_28 Depth=1
	s_mov_b32 s0, 2.0
	s_branch .LBB0_27
.LBB0_37:                               ; %Flow559
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s8
	v_cmp_neq_f32_e64 s0, 0, v5
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_39
; %bb.38:
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
.LBB0_39:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_41
; %bb.40:
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
.LBB0_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_43
; %bb.42:
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
.LBB0_43:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	s_and_saveexec_b32 s2, s0
	s_cbranch_execz .LBB0_45
; %bb.44:
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
.LBB0_45:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v1, v5, v6
	v_and_b32_e32 v6, 15, v6
	s_mul_u64 s[2:3], s[4:5], 0x48
	s_wait_alu depctr_sa_sdst(0)
	s_add_nc_u64 s[2:3], s[6:7], s[2:3]
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
	s_and_saveexec_b32 s0, s1
	s_cbranch_execz .LBB0_47
; %bb.46:
	s_wait_dscnt 0x0
	v_dual_mov_b32 v2, 0 :: v_dual_add_nc_u32 v1, v1, v2
	global_store_b64 v2, v[0:1], s[2:3]
.LBB0_47:                               ; %_Z31emit_iu4_sidecar_from_producer8ffffffffP12block_i4_128iiii.exit
	s_endpgm
.Lfunc_end0:
	.size	fused_silu_mul_mq_rotate_awq_i4_gfx12, .Lfunc_end0-fused_silu_mul_mq_rotate_awq_i4_gfx12
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fused_silu_mul_mq_rotate_awq_i4_gfx12
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
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
		.amdhsa_next_free_vgpr 55
		.amdhsa_next_free_sgpr 16
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-fused_silu_mul_mq_rotate_awq_i4_gfx12)<<4)&4080)>>4
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
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.num_vgpr, 55
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.num_agpr, 0
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.numbered_sgpr, 16
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.num_named_barrier, 0
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.private_seg_size, 0
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.uses_vcc, 1
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.uses_flat_scratch, 0
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.has_dyn_sized_stack, 0
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.has_recursion, 0
	.set .Lfused_silu_mul_mq_rotate_awq_i4_gfx12.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8476
; TotalNumSgprs: 18
; NumVgprs: 55
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 55
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
	.type	__hip_cuid_4c708912eda94d18,@object ; @__hip_cuid_4c708912eda94d18
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4c708912eda94d18
__hip_cuid_4c708912eda94d18:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4c708912eda94d18, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 8f497e0992fb7513f7f78a6f6b6f1056c375e961)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_4c708912eda94d18
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
      - .actual_access:  write_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 32
    .name:           fused_silu_mul_mq_rotate_awq_i4_gfx12
    .private_segment_fixed_size: 0
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         fused_silu_mul_mq_rotate_awq_i4_gfx12.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     55
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
